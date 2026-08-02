from __future__ import annotations

import numpy as np
import pandas as pd


def _as_num(x):
    return pd.to_numeric(x, errors="coerce")


def _optional_num(out: pd.DataFrame, *columns: str, default: float = np.nan) -> pd.Series:
    for col in columns:
        if col in out.columns:
            return _as_num(out[col])
    return pd.Series(default, index=out.index, dtype=float)


def _hour_group(hour: int) -> str:
    h = int(hour)
    if 0 <= h <= 5:
        return "Overnight"
    if 6 <= h <= 9:
        return "Morning"
    if 10 <= h <= 15:
        return "Midday"
    if 16 <= h <= 20:
        return "Peak"
    return "LateEvening"


def apply_peak_risk_correction(df: pd.DataFrame, config: dict | None = None, base_col: str = "Calibrated_Forecast_MWH") -> pd.DataFrame:
    """Apply a conservative conditional peak-risk uplift.

    This is deliberately *not* a Prophet/CatBoost blend.  It only uses those benchmark
    models as warning signals when they are materially above the corrected production
    forecast during likely warm peak hours.  The goal is to reduce the specific V12.5
    failure mode: warm spring evening peaks that remain low after the normal correction chain.
    """
    out = df.copy().sort_values("DT").reset_index(drop=True)
    cfg = (((config or {}).get("calibration", {}) or {}).get("peak_risk", {}) or {})
    enabled = bool(cfg.get("enabled", True))

    if base_col not in out.columns:
        base_col = "Calibrated_Forecast_MWH" if "Calibrated_Forecast_MWH" in out.columns else "Raw_Forecast_MWH"

    out["Peak_Risk_Cal_MWH"] = 0.0
    out["Peak_Risk_Source"] = "none"
    out["Peak_Risk_Adjusted_Forecast_MWH"] = _as_num(out[base_col]).clip(lower=0.0)
    if not enabled or out.empty:
        out["Calibrated_Forecast_MWH"] = out["Peak_Risk_Adjusted_Forecast_MWH"]
        return out

    if "Hour" not in out.columns:
        out["Hour"] = pd.to_datetime(out["DT"], errors="coerce").dt.hour.astype("Int64").astype(float)
    if "HourGroup" not in out.columns:
        out["HourGroup"] = out["Hour"].fillna(-1).astype(int).map(_hour_group)

    hours = [int(h) for h in cfg.get("hours", [17, 18, 19, 20, 21])]
    min_maxtemp = float(cfg.get("min_maxtemp_f", 78.0))
    prophet_threshold = float(cfg.get("prophet_gap_threshold_mwh", 6.0))
    cat_threshold = float(cfg.get("catboost_gap_threshold_mwh", 7.5))
    tree_threshold = float(cfg.get("tree_gap_threshold_mwh", 0.0) or 0.0)
    tree_strength = float(cfg.get("tree_gap_signal_strength", 0.0) or 0.0)
    tree_cols = list(cfg.get("tree_gap_model_cols", ["XGB_Pred_MWH", "LGB_Pred_MWH", "CatBoost_Pred_MWH"]) or [])
    blend = float(cfg.get("blend", 0.45))
    cap = float(cfg.get("cap_mwh", 8.0))
    min_recent_under = float(cfg.get("min_recent_underforecast_correction_mwh", 0.0))
    extreme_cfg = cfg.get("extreme_hot_peak", {}) or {}
    extreme_enabled = bool(extreme_cfg.get("enabled", False))
    extreme_hours = [int(h) for h in extreme_cfg.get("hours", [16, 17, 18])]
    extreme_min_maxtemp = float(extreme_cfg.get("min_maxtemp_f", 110.0))
    extreme_min_day = int(extreme_cfg.get("min_forecast_day", 2))
    extreme_max_day = int(extreme_cfg.get("max_forecast_day", 7))
    extreme_cap = float(extreme_cfg.get("cap_mwh", cap))

    base = _as_num(out[base_col])
    hour = _as_num(out["Hour"]).fillna(-1).astype(int)
    forecast_day = _as_num(out.get("Forecast_Day", pd.Series(np.nan, index=out.index)))
    if "Temperature_DailyMax" in out.columns:
        maxt = _as_num(out["Temperature_DailyMax"])
    elif "Temperature" in out.columns:
        maxt = _as_num(out["Temperature"])
    else:
        maxt = pd.Series(np.nan, index=out.index)

    eligible = hour.isin(hours) & maxt.ge(min_maxtemp)
    if "Recent_Level_Correction_MWH" in out.columns and min_recent_under > 0:
        eligible &= _as_num(out["Recent_Level_Correction_MWH"]).fillna(0.0).ge(min_recent_under)

    signal = pd.Series(0.0, index=out.index, dtype=float)
    sources = [[] for _ in range(len(out))]

    if "Prophet_Pred_MWH" in out.columns:
        gap = (_as_num(out["Prophet_Pred_MWH"]) - base).fillna(0.0)
        add = (gap - prophet_threshold).clip(lower=0.0)
        use = eligible & add.gt(0.0)
        signal.loc[use] = np.maximum(signal.loc[use], add.loc[use])
        for ix in out.index[use]:
            sources[ix].append("prophet_peak_gap")

    if "CatBoost_Pred_MWH" in out.columns:
        gap = (_as_num(out["CatBoost_Pred_MWH"]) - base).fillna(0.0)
        add = (gap - cat_threshold).clip(lower=0.0)
        use = eligible & add.gt(0.0)
        # CatBoost is a weaker warning than Prophet here, so give it partial signal strength.
        signal.loc[use] = np.maximum(signal.loc[use], add.loc[use] * 0.65)
        for ix in out.index[use]:
            sources[ix].append("catboost_peak_gap")

    if tree_threshold > 0 and tree_strength > 0 and tree_cols:
        present_tree_cols = [col for col in tree_cols if col in out.columns]
        if present_tree_cols:
            tree_preds = pd.concat([_as_num(out[col]) for col in present_tree_cols], axis=1)
            gap = (tree_preds.max(axis=1, skipna=True) - base).fillna(0.0)
            add = (gap - tree_threshold).clip(lower=0.0)
            use = eligible & add.gt(0.0)
            signal.loc[use] = np.maximum(signal.loc[use], add.loc[use] * tree_strength)
            for ix in out.index[use]:
                sources[ix].append("tree_peak_gap")

    cap_by_row = pd.Series(cap, index=out.index, dtype=float)
    if extreme_enabled:
        extreme_mask = (
            eligible
            & hour.isin(extreme_hours)
            & maxt.ge(extreme_min_maxtemp)
            & forecast_day.between(extreme_min_day, extreme_max_day)
        )
        cap_by_row.loc[extreme_mask] = np.maximum(cap_by_row.loc[extreme_mask], extreme_cap)
        for ix in out.index[extreme_mask & signal.gt(0.0)]:
            sources[ix].append("extreme_hot_peak_cap")

    correction = pd.Series(np.minimum((signal * blend).clip(lower=0.0), cap_by_row), index=out.index, dtype=float)

    positive_guard_cfg = cfg.get("positive_guard", {}) or {}
    overforecast_guard_cfg = positive_guard_cfg.get("overforecast_risk", {}) or {}
    positive_guard_enabled = bool(positive_guard_cfg.get("enabled", False)) and bool(
        overforecast_guard_cfg.get("enabled", True)
    )
    if positive_guard_enabled:
        guard_hours = [int(h) for h in overforecast_guard_cfg.get("hours", hours)]
        guard_min_maxtemp = float(overforecast_guard_cfg.get("min_maxtemp_f", min_maxtemp))
        guard_min_raw_minus_7day = float(
            overforecast_guard_cfg.get("min_raw_minus_samehour_7day_mean_mwh", np.inf)
        )
        guard_min_next3_drop = float(overforecast_guard_cfg.get("min_forecast_drop_next3hr_f", np.inf))
        guard_max_positive = float(overforecast_guard_cfg.get("max_positive_correction_mwh", 0.0))
        blocked_source = str(
            overforecast_guard_cfg.get("blocked_source", "peak_risk_overforecast_guard_blocked")
        )

        raw = _optional_num(out, "Raw_Forecast_MWH", "Forecast_MWH", default=np.nan)
        raw = raw.where(raw.notna(), base)
        raw_minus_samehour_7day = _optional_num(out, "Raw_Minus_SameHour7DayMean_MWH", default=np.nan)
        if raw_minus_samehour_7day.isna().all():
            samehour_7day = _optional_num(
                out,
                "MWH_SameHour7DayMean",
                "Baseline_Rolling7DaySameHourAvg_MWH",
                default=np.nan,
            )
            raw_minus_samehour_7day = raw - samehour_7day
        next3_drop = _optional_num(out, "TempDrop_Next3Hr_F", default=np.nan)

        guard_mask = (
            correction.gt(guard_max_positive)
            & hour.isin(guard_hours)
            & maxt.ge(guard_min_maxtemp)
            & raw_minus_samehour_7day.ge(guard_min_raw_minus_7day)
            & next3_drop.ge(guard_min_next3_drop)
        )
        if guard_max_positive <= 0.0:
            correction.loc[guard_mask] = 0.0
            for ix in out.index[guard_mask]:
                sources[ix] = [blocked_source]
        else:
            capped_mask = guard_mask & correction.gt(guard_max_positive)
            correction.loc[capped_mask] = guard_max_positive
            for ix in out.index[capped_mask]:
                if blocked_source not in sources[ix]:
                    sources[ix].append(blocked_source)

    out["Peak_Risk_Cal_MWH"] = correction
    out["Peak_Risk_Source"] = ["+".join(s) if s else "none" for s in sources]
    out["Peak_Risk_Adjusted_Forecast_MWH"] = (base + correction).clip(lower=0.0)
    out["Calibrated_Forecast_MWH"] = out["Peak_Risk_Adjusted_Forecast_MWH"]
    return out
