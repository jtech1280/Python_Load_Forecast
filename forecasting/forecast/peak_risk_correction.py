from __future__ import annotations

from datetime import timedelta

import numpy as np
import pandas as pd


def _as_num(x):
    return pd.to_numeric(x, errors="coerce")


def _optional_num(out: pd.DataFrame, *columns: str, default: float = np.nan) -> pd.Series:
    for col in columns:
        if col in out.columns:
            return _as_num(out[col])
    return pd.Series(default, index=out.index, dtype=float)


def _append_source(sources: list[list[str]], index_values: pd.Index, label: str) -> None:
    for ix in index_values:
        if label not in sources[ix]:
            sources[ix].append(label)


def _local_datetime_series(values, index: pd.Index | None = None) -> pd.Series:
    raw = values if isinstance(values, pd.Series) else pd.Series(values, index=index)
    try:
        return pd.to_datetime(raw, errors="coerce")
    except ValueError:
        cleaned = raw.astype(str).str.strip().str.replace(r"(?:[+-]\d{2}:?\d{2}|Z)$", "", regex=True)
        return pd.to_datetime(cleaned, errors="coerce")


def _forecast_day_series(out: pd.DataFrame, dt: pd.Series | None = None) -> pd.Series:
    if "Forecast_Day" in out.columns:
        day = _as_num(out["Forecast_Day"])
        if day.notna().any():
            return day
    if dt is None:
        if "DT" not in out.columns:
            return pd.Series(np.nan, index=out.index, dtype=float)
        dt = _local_datetime_series(out["DT"], out.index)
    if dt.notna().any():
        first_day = dt.dropna().min().normalize()
        return ((dt.dt.normalize() - first_day).dt.days + 1).astype(float)
    return pd.Series(np.nan, index=out.index, dtype=float)


def _actual_col(df: pd.DataFrame) -> str | None:
    for col in ["MWH", "Actual_MWH", "Actual"]:
        if col in df.columns:
            return col
    return None


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


def _merged_hot_ramp_cfg(config: dict | None, peak_cfg: dict | None = None) -> dict:
    raw = config or {}
    cal_cfg = (raw.get("calibration", {}) or {}) if isinstance(raw, dict) else {}
    root_cfg = cal_cfg.get("hot_ramp_override", {}) or {}
    nested_cfg = (peak_cfg or {}).get("hot_ramp_override", {}) or {}
    if "hot_ramp_override" in raw:
        root_cfg = raw.get("hot_ramp_override", {}) or {}
    merged = dict(root_cfg)
    merged.update(nested_cfg)
    return merged


def _day1_live_ramp_cfg(config: dict | None) -> dict:
    raw = config or {}
    if isinstance(raw, dict) and "calibration" in raw:
        return ((raw.get("calibration", {}) or {}).get("day1_live_ramp_override", {}) or {})
    if isinstance(raw, dict) and "day1_live_ramp_override" in raw:
        return raw.get("day1_live_ramp_override", {}) or {}
    return raw if isinstance(raw, dict) else {}


def _multiday_live_heat_anchor_cfg(config: dict | None) -> dict:
    raw = config or {}
    if isinstance(raw, dict) and "calibration" in raw:
        return ((raw.get("calibration", {}) or {}).get("multiday_live_heat_anchor_override", {}) or {})
    if isinstance(raw, dict) and "multiday_live_heat_anchor_override" in raw:
        return raw.get("multiday_live_heat_anchor_override", {}) or {}
    return raw if isinstance(raw, dict) else {}


def _day1_live_ramp_state(history_df: pd.DataFrame | None, cfg: dict) -> dict:
    state = {
        "enabled": False,
        "latest_date": None,
        "latest_hour": np.nan,
        "pair_count": 0,
        "observed_delta_mwh": np.nan,
        "source": "disabled_or_empty",
    }
    if history_df is None or history_df.empty:
        return state
    actual_col = _actual_col(history_df)
    if actual_col is None or "DT" not in history_df.columns:
        state["source"] = "missing_history_actuals"
        return state

    hist = history_df[["DT", actual_col]].copy()
    hist["_dt"] = _local_datetime_series(hist["DT"], hist.index)
    hist["_actual"] = _as_num(hist[actual_col])
    hist = hist[hist["_dt"].notna() & hist["_actual"].notna()].sort_values("_dt")
    if hist.empty:
        state["source"] = "missing_history_actuals"
        return state
    hist["_date"] = hist["_dt"].dt.date
    hist["_hour"] = hist["_dt"].dt.hour.astype(int)
    latest_date = hist["_date"].max()
    previous_date = (pd.Timestamp(latest_date) - timedelta(days=1)).date()
    today = hist[hist["_date"].eq(latest_date)].copy()
    previous = hist[hist["_date"].eq(previous_date)].copy()
    if today.empty or previous.empty:
        state["source"] = "missing_previous_day_pairs"
        return state

    latest_hour = int(today["_hour"].max())
    min_latest_hour = int(cfg.get("min_latest_actual_hour", 6))
    if latest_hour < min_latest_hour:
        state.update(
            {
                "latest_date": str(latest_date),
                "latest_hour": latest_hour,
                "source": "latest_actual_too_early",
            }
        )
        return state

    merged = today[["_hour", "_actual"]].merge(
        previous[["_hour", "_actual"]].rename(columns={"_actual": "_previous_actual"}),
        on="_hour",
        how="inner",
    )
    if merged.empty:
        state["source"] = "missing_previous_day_pairs"
        return state

    observation_hours = cfg.get("observation_hours")
    if observation_hours:
        hours = {int(h) for h in observation_hours}
        merged = merged[merged["_hour"].isin(hours)]
    merged = merged[merged["_hour"].le(latest_hour)].sort_values("_hour")
    tail_n = int(cfg.get("observation_tail_hours", 3))
    if tail_n > 0:
        merged = merged.tail(tail_n)
    if len(merged) < int(cfg.get("min_pair_hours", 3)):
        state.update(
            {
                "latest_date": str(latest_date),
                "latest_hour": latest_hour,
                "pair_count": int(len(merged)),
                "source": "insufficient_previous_day_pairs",
            }
        )
        return state

    deltas = (_as_num(merged["_actual"]) - _as_num(merged["_previous_actual"])).dropna()
    if deltas.empty:
        state["source"] = "missing_previous_day_pairs"
        return state
    stat = str(cfg.get("observed_delta_stat", "latest")).strip().lower()
    if stat == "median":
        observed_delta = float(deltas.median())
    elif stat == "mean":
        observed_delta = float(deltas.mean())
    elif stat == "max":
        observed_delta = float(deltas.max())
    else:
        observed_delta = float(deltas.iloc[-1])

    min_delta = float(cfg.get("min_observed_delta_mwh", 15.0))
    if observed_delta < min_delta:
        state.update(
            {
                "latest_date": str(latest_date),
                "latest_hour": latest_hour,
                "pair_count": int(len(deltas)),
                "observed_delta_mwh": observed_delta,
                "source": "observed_delta_below_threshold",
            }
        )
        return state

    state.update(
        {
            "enabled": True,
            "latest_date": str(latest_date),
            "latest_hour": latest_hour,
            "pair_count": int(len(deltas)),
            "observed_delta_mwh": observed_delta,
            "source": "latest_same_day_vs_yesterday",
        }
    )
    return state


def _derive_daily_series(out: pd.DataFrame, maxt: pd.Series, date: pd.Series, col: str) -> pd.Series:
    if col in out.columns:
        values = _as_num(out[col])
        if values.notna().any():
            return values
    daily = (
        pd.DataFrame({"Date": date, "Temperature_DailyMax": maxt})
        .dropna(subset=["Date"])
        .drop_duplicates(subset=["Date"])
        .sort_values("Date")
        .reset_index(drop=True)
    )
    if daily.empty:
        return pd.Series(np.nan, index=out.index, dtype=float)
    max_temp = _as_num(daily["Temperature_DailyMax"])
    if col == "PriorDay_DailyMaxTemp":
        daily[col] = max_temp.shift(1)
    elif col == "DailyMaxTemp_Ramp_1Day":
        daily[col] = max_temp - max_temp.shift(1)
    elif col == "DailyMaxTemp_2DayMean":
        daily[col] = max_temp.rolling(2, min_periods=1).mean()
    elif col == "DailyMaxTemp_3DayMean":
        daily[col] = max_temp.rolling(3, min_periods=1).mean()
    elif col == "ConsecutiveHotDays90":
        daily[col] = _consecutive_daily_count(max_temp.ge(90.0))
    elif col == "ConsecutiveVeryHotDays95":
        daily[col] = _consecutive_daily_count(max_temp.ge(95.0))
    elif col == "ConsecutiveExtremeHotDays100":
        daily[col] = _consecutive_daily_count(max_temp.ge(100.0))
    else:
        daily[col] = np.nan
    return date.map(daily.set_index("Date")[col]).reindex(out.index).astype(float)


def _consecutive_daily_count(flag: pd.Series) -> pd.Series:
    values = flag.fillna(False).astype(bool)
    groups = values.ne(values.shift(fill_value=False)).cumsum()
    counts = values.groupby(groups).cumsum()
    return counts.where(values, 0).astype(float)


def _hot_ramp_context(
    out: pd.DataFrame,
    cfg: dict,
    hour: pd.Series,
    maxt: pd.Series,
    forecast_day: pd.Series,
) -> dict[str, pd.Series]:
    if not bool(cfg.get("enabled", False)) or out.empty:
        false = pd.Series(False, index=out.index, dtype=bool)
        zero = pd.Series(0.0, index=out.index, dtype=float)
        return {
            "gate": false,
            "persistence": false,
            "consecutive_hot_90": zero,
            "consecutive_very_hot_95": zero,
            "consecutive_extreme_hot_100": zero,
            "dailymax_3day_mean": zero,
            "dailymax_ramp_1day": zero,
        }

    dt = _local_datetime_series(out["DT"], out.index) if "DT" in out.columns else pd.Series(pd.NaT, index=out.index)
    date = out["Date"] if "Date" in out.columns else dt.dt.date
    date = pd.Series(date, index=out.index)

    consec90 = _derive_daily_series(out, maxt, date, "ConsecutiveHotDays90").fillna(0.0)
    consec95 = _derive_daily_series(out, maxt, date, "ConsecutiveVeryHotDays95").fillna(0.0)
    consec100 = _derive_daily_series(out, maxt, date, "ConsecutiveExtremeHotDays100").fillna(0.0)
    dailymax_3day = _derive_daily_series(out, maxt, date, "DailyMaxTemp_3DayMean")
    dailymax_ramp = _derive_daily_series(out, maxt, date, "DailyMaxTemp_Ramp_1Day").fillna(0.0)
    prior_maxt = _derive_daily_series(out, maxt, date, "PriorDay_DailyMaxTemp")

    min_day = int(cfg.get("min_forecast_day", 1))
    max_day = int(cfg.get("max_forecast_day", 16))
    if forecast_day.notna().any():
        day_gate = forecast_day.between(min_day, max_day)
    else:
        day_gate = pd.Series(True, index=out.index, dtype=bool)

    hours = {int(h) for h in cfg.get("hours", [17, 18, 19, 20])}
    temp_gate = maxt.ge(float(cfg.get("min_maxtemp_f", 100.0))) & maxt.le(float(cfg.get("max_maxtemp_f", 109.9)))
    hour_gate = hour.isin(hours)

    persistence_terms: list[pd.Series] = []
    min_consec90 = float(cfg.get("min_consecutive_hot_days_90", np.inf))
    min_consec95 = float(cfg.get("min_consecutive_very_hot_days_95", 2.0))
    min_consec100 = float(cfg.get("min_consecutive_extreme_hot_days_100", 2.0))
    min_3day = float(cfg.get("min_dailymax_3day_mean_f", np.inf))
    min_ramp = float(cfg.get("min_dailymax_ramp_1day_f", np.inf))
    min_prior_for_ramp = float(cfg.get("min_prior_dailymax_for_ramp_f", 95.0))
    if np.isfinite(min_consec90):
        persistence_terms.append(consec90.ge(min_consec90))
    if np.isfinite(min_consec95):
        persistence_terms.append(consec95.ge(min_consec95))
    if np.isfinite(min_consec100):
        persistence_terms.append(consec100.ge(min_consec100))
    if np.isfinite(min_3day):
        persistence_terms.append(dailymax_3day.ge(min_3day))
    if np.isfinite(min_ramp):
        persistence_terms.append(dailymax_ramp.ge(min_ramp) & prior_maxt.ge(min_prior_for_ramp))
    persistence = pd.Series(False, index=out.index, dtype=bool)
    for term in persistence_terms:
        persistence |= term.fillna(False).astype(bool)

    gate = temp_gate & hour_gate & day_gate & persistence
    return {
        "gate": gate.fillna(False).astype(bool),
        "persistence": persistence.fillna(False).astype(bool),
        "consecutive_hot_90": consec90,
        "consecutive_very_hot_95": consec95,
        "consecutive_extreme_hot_100": consec100,
        "dailymax_3day_mean": dailymax_3day,
        "dailymax_ramp_1day": dailymax_ramp,
    }


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
    out["Hot_Ramp_Override_Gate"] = 0
    out["Hot_Ramp_Override_HeatPersistence_Flag"] = 0
    out["Hot_Ramp_Override_PeakRisk_Protected_MWH"] = 0.0
    out["Hot_Ramp_Override_Scenario_Gap_MWH"] = 0.0
    out["Hot_Ramp_Override_Scenario_Lift_MWH"] = 0.0
    out["Hot_Ramp_Override_Cal_MWH"] = 0.0
    out["Hot_Ramp_Override_Source"] = "none"
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

    hot_ramp_cfg = _merged_hot_ramp_cfg(config, cfg)
    hot_ramp = _hot_ramp_context(out, hot_ramp_cfg, hour, maxt, forecast_day)
    hot_ramp_gate = hot_ramp["gate"]
    out["Hot_Ramp_Override_Gate"] = hot_ramp_gate.astype(int)
    out["Hot_Ramp_Override_HeatPersistence_Flag"] = hot_ramp["persistence"].astype(int)

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

    if bool(hot_ramp_cfg.get("scenario_lift_enabled", True)):
        scenario_col = str(hot_ramp_cfg.get("scenario_gap_col", "WeatherScenario_hot_stress_5f_P50_MWH"))
        if scenario_col in out.columns:
            total_cap = float(
                hot_ramp_cfg.get(
                    "total_cap_mwh",
                    hot_ramp_cfg.get("cap_mwh", hot_ramp_cfg.get("scenario_lift_cap_mwh", 8.0)),
                )
            )
            scenario_gap = (_as_num(out[scenario_col]) - (base + correction)).clip(lower=0.0).fillna(0.0)
            scenario_lift = (
                scenario_gap
                * float(hot_ramp_cfg.get("scenario_gap_blend", 0.40))
            ).clip(lower=0.0, upper=float(hot_ramp_cfg.get("scenario_lift_cap_mwh", hot_ramp_cfg.get("cap_mwh", 8.0))))
            scenario_lift = np.minimum(scenario_lift, float(total_cap))
            scenario_lift = scenario_lift.where(hot_ramp_gate, 0.0).fillna(0.0)
            correction = np.maximum(correction, scenario_lift)
            out["Hot_Ramp_Override_Scenario_Gap_MWH"] = scenario_gap.where(hot_ramp_gate, 0.0)
            out["Hot_Ramp_Override_Scenario_Lift_MWH"] = scenario_lift
        else:
            out["Hot_Ramp_Override_Scenario_Gap_MWH"] = 0.0
            out["Hot_Ramp_Override_Scenario_Lift_MWH"] = 0.0

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
        bypass_mask = guard_mask & hot_ramp_gate & bool(hot_ramp_cfg.get("guard_bypass_enabled", True))
        protected_cap = float(hot_ramp_cfg.get("peak_risk_protected_cap_mwh", hot_ramp_cfg.get("cap_mwh", 8.0)))
        protected = correction.clip(lower=0.0, upper=protected_cap).where(bypass_mask, 0.0).fillna(0.0)
        if bypass_mask.any():
            correction.loc[bypass_mask] = protected.loc[bypass_mask]
            _append_source(sources, out.index[bypass_mask], str(hot_ramp_cfg.get("guard_bypass_source", "hot_ramp_guard_bypass")))
            out.loc[bypass_mask, "Hot_Ramp_Override_PeakRisk_Protected_MWH"] = protected.loc[bypass_mask]
        guard_mask = guard_mask & ~bypass_mask
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
    total_cap = float(
        hot_ramp_cfg.get(
            "total_cap_mwh",
            hot_ramp_cfg.get("cap_mwh", hot_ramp_cfg.get("scenario_lift_cap_mwh", 8.0)),
        )
    )
    out["Hot_Ramp_Override_Cal_MWH"] = np.maximum(
        _as_num(out["Hot_Ramp_Override_PeakRisk_Protected_MWH"]).fillna(0.0),
        _as_num(out["Hot_Ramp_Override_Scenario_Lift_MWH"]).fillna(0.0),
    ).clip(lower=0.0, upper=total_cap)
    hot_source = pd.Series("none", index=out.index, dtype="object")
    hot_source.loc[_as_num(out["Hot_Ramp_Override_PeakRisk_Protected_MWH"]).gt(0.0)] = str(
        hot_ramp_cfg.get("guard_bypass_source", "hot_ramp_guard_bypass")
    )
    scenario_source = str(hot_ramp_cfg.get("scenario_lift_source", "hot_ramp_hot_stress_scenario_lift"))
    hot_source.loc[_as_num(out["Hot_Ramp_Override_Scenario_Lift_MWH"]).gt(0.0) & hot_source.eq("none")] = scenario_source
    hot_source.loc[_as_num(out["Hot_Ramp_Override_Scenario_Lift_MWH"]).gt(0.0) & ~hot_source.eq("none")] = (
        hot_source.loc[_as_num(out["Hot_Ramp_Override_Scenario_Lift_MWH"]).gt(0.0) & ~hot_source.eq("none")] + "+" + scenario_source
    )
    out["Hot_Ramp_Override_Source"] = hot_source
    out["Peak_Risk_Adjusted_Forecast_MWH"] = (base + correction).clip(lower=0.0)
    out["Calibrated_Forecast_MWH"] = out["Peak_Risk_Adjusted_Forecast_MWH"]
    return out


def apply_hot_ramp_scenario_override(
    df: pd.DataFrame,
    config: dict | None = None,
    base_col: str = "Final_Forecast_MWH",
    also_update_cols: tuple[str, ...] = ("Stage_Selected_Forecast_MWH",),
) -> pd.DataFrame:
    """Lift persistent 100-109F peak-hour forecasts toward the hot-stress scenario."""
    out = df.copy()
    cal_cfg = ((config or {}).get("calibration", {}) or {}) if isinstance(config, dict) else {}
    cfg = _merged_hot_ramp_cfg(config, (cal_cfg.get("peak_risk", {}) or {}))
    for col, default in {
        "Hot_Ramp_Override_Gate": 0,
        "Hot_Ramp_Override_HeatPersistence_Flag": 0,
        "Hot_Ramp_Override_PeakRisk_Protected_MWH": 0.0,
        "Hot_Ramp_Override_Scenario_Gap_MWH": 0.0,
        "Hot_Ramp_Override_Scenario_Lift_MWH": 0.0,
        "Hot_Ramp_Override_Cal_MWH": 0.0,
        "Hot_Ramp_Override_Source": "none",
    }.items():
        if col not in out.columns:
            out[col] = default

    if not bool(cfg.get("enabled", False)) or not bool(cfg.get("scenario_lift_enabled", True)) or out.empty:
        return out
    if base_col not in out.columns:
        return out
    scenario_col = str(cfg.get("scenario_gap_col", "WeatherScenario_hot_stress_5f_P50_MWH"))
    if scenario_col not in out.columns:
        return out

    if "Hour" not in out.columns:
        out["Hour"] = _local_datetime_series(out["DT"], out.index).dt.hour.astype("Int64").astype(float) if "DT" in out.columns else np.nan
    hour = _as_num(out["Hour"]).fillna(-1).astype(int)
    if "Temperature_DailyMax" in out.columns:
        maxt = _as_num(out["Temperature_DailyMax"])
    elif "Temperature" in out.columns:
        maxt = _as_num(out["Temperature"])
    else:
        maxt = pd.Series(np.nan, index=out.index)
    forecast_day = _forecast_day_series(out)
    hot_ramp = _hot_ramp_context(out, cfg, hour, maxt, forecast_day)
    gate = hot_ramp["gate"]

    base = _as_num(out[base_col])
    scenario = _as_num(out[scenario_col])
    total_cap = float(cfg.get("total_cap_mwh", cfg.get("cap_mwh", cfg.get("scenario_lift_cap_mwh", 8.0))))
    prior_override = _as_num(out["Hot_Ramp_Override_Cal_MWH"]).fillna(0.0).clip(lower=0.0)
    remaining_cap = (total_cap - prior_override).clip(lower=0.0)
    gap = (scenario - base).clip(lower=0.0).fillna(0.0)
    lift = (
        gap
        * float(cfg.get("scenario_gap_blend", 0.40))
    ).clip(lower=0.0, upper=float(cfg.get("scenario_lift_cap_mwh", cfg.get("cap_mwh", 8.0))))
    lift = np.minimum(lift, remaining_cap)
    lift = lift.where(gate, 0.0).fillna(0.0)
    if not lift.gt(0.0).any():
        out["Hot_Ramp_Override_Gate"] = np.maximum(_as_num(out["Hot_Ramp_Override_Gate"]).fillna(0).astype(int), gate.astype(int))
        out["Hot_Ramp_Override_HeatPersistence_Flag"] = np.maximum(
            _as_num(out["Hot_Ramp_Override_HeatPersistence_Flag"]).fillna(0).astype(int),
            hot_ramp["persistence"].astype(int),
        )
        return out

    out["Hot_Ramp_Override_Gate"] = np.maximum(_as_num(out["Hot_Ramp_Override_Gate"]).fillna(0).astype(int), gate.astype(int))
    out["Hot_Ramp_Override_HeatPersistence_Flag"] = np.maximum(
        _as_num(out["Hot_Ramp_Override_HeatPersistence_Flag"]).fillna(0).astype(int),
        hot_ramp["persistence"].astype(int),
    )
    out["Hot_Ramp_Override_Scenario_Gap_MWH"] = np.maximum(
        _as_num(out["Hot_Ramp_Override_Scenario_Gap_MWH"]).fillna(0.0),
        gap.where(gate, 0.0),
    )
    out["Hot_Ramp_Override_Scenario_Lift_MWH"] = _as_num(out["Hot_Ramp_Override_Scenario_Lift_MWH"]).fillna(0.0) + lift
    out["Hot_Ramp_Override_Cal_MWH"] = (prior_override + lift).clip(lower=0.0, upper=total_cap)
    source = str(cfg.get("scenario_lift_source", "hot_ramp_hot_stress_scenario_lift"))
    existing_source = out["Hot_Ramp_Override_Source"].astype(str)
    apply_source = lift.gt(0.0)
    out.loc[apply_source & existing_source.eq("none"), "Hot_Ramp_Override_Source"] = source
    out.loc[apply_source & ~existing_source.eq("none"), "Hot_Ramp_Override_Source"] = (
        existing_source.loc[apply_source & ~existing_source.eq("none")] + "+" + source
    )

    out[base_col] = (base + lift).clip(lower=0.0)
    update_cols = list(also_update_cols)
    if base_col == "Final_Forecast_MWH" and "Calibrated_Forecast_MWH" in out.columns and "Calibrated_Forecast_MWH" not in update_cols:
        update_cols.append("Calibrated_Forecast_MWH")
    for col in update_cols:
        if col in out.columns:
            out[col] = (_as_num(out[col]) + lift).clip(lower=0.0)
    return out


def apply_day1_live_ramp_override(
    df: pd.DataFrame,
    history_df: pd.DataFrame | None,
    config: dict | None = None,
    base_col: str = "Final_Forecast_MWH",
    also_update_cols: tuple[str, ...] = ("Stage_Selected_Forecast_MWH",),
) -> pd.DataFrame:
    """Day-1 hot-peak lift from same-day actual load running above yesterday."""
    out = df.copy()
    cfg = _day1_live_ramp_cfg(config)
    for col, default in {
        "Day1_Live_Ramp_Gate": 0,
        "Day1_Live_Ramp_Cal_MWH": 0.0,
        "Day1_Live_Ramp_Observed_Delta_MWH": np.nan,
        "Day1_Live_Ramp_Target_Delta_MWH": np.nan,
        "Day1_Live_Ramp_Target_MWH": np.nan,
        "Day1_Live_Ramp_Yesterday_MWH": np.nan,
        "Day1_Live_Ramp_Latest_Actual_Hour": np.nan,
        "Day1_Live_Ramp_Pair_Count": 0,
        "Day1_Live_Ramp_Source": "none",
    }.items():
        if col not in out.columns:
            out[col] = default

    if not bool(cfg.get("enabled", False)) or out.empty or base_col not in out.columns:
        return out
    state = _day1_live_ramp_state(history_df, cfg)
    out["Day1_Live_Ramp_Observed_Delta_MWH"] = state.get("observed_delta_mwh", np.nan)
    out["Day1_Live_Ramp_Latest_Actual_Hour"] = state.get("latest_hour", np.nan)
    out["Day1_Live_Ramp_Pair_Count"] = int(state.get("pair_count", 0) or 0)
    if not bool(state.get("enabled", False)):
        out["Day1_Live_Ramp_Source"] = str(state.get("source", "disabled_or_empty"))
        return out

    if "Hour" not in out.columns:
        out["Hour"] = _local_datetime_series(out["DT"], out.index).dt.hour.astype("Int64").astype(float) if "DT" in out.columns else np.nan
    hour = _as_num(out["Hour"]).fillna(-1).astype(int)
    forecast_day = _forecast_day_series(out)
    if "Temperature_DailyMax" in out.columns:
        maxt = _as_num(out["Temperature_DailyMax"])
    elif "Temperature" in out.columns:
        maxt = _as_num(out["Temperature"])
    else:
        maxt = pd.Series(np.nan, index=out.index)

    if "MWH_Lag24" in out.columns:
        yesterday = _as_num(out["MWH_Lag24"])
    else:
        out["Day1_Live_Ramp_Source"] = "missing_yesterday_lag24"
        return out

    hours = {int(h) for h in cfg.get("hours", [16, 17, 18, 19, 20])}
    gate = (
        forecast_day.between(int(cfg.get("min_forecast_day", 1)), int(cfg.get("max_forecast_day", 1)))
        & hour.isin(hours)
        & maxt.ge(float(cfg.get("min_maxtemp_f", 100.0)))
        & maxt.le(float(cfg.get("max_maxtemp_f", 115.0)))
        & yesterday.notna()
    )
    if "IsHoliday" in out.columns and bool(cfg.get("exclude_holidays", False)):
        gate &= _as_num(out["IsHoliday"]).fillna(0.0).eq(0.0)

    observed_delta = float(state.get("observed_delta_mwh", 0.0))
    target_delta = observed_delta * float(cfg.get("carry_fraction", 0.90))
    target = yesterday + target_delta
    base = _as_num(out[base_col])
    lift = (target - base).clip(lower=0.0).fillna(0.0)
    lift = lift.clip(lower=0.0, upper=float(cfg.get("cap_mwh", 16.0)))
    lift = lift.where(gate, 0.0).fillna(0.0)

    out["Day1_Live_Ramp_Gate"] = gate.astype(int)
    out["Day1_Live_Ramp_Target_Delta_MWH"] = target_delta
    out["Day1_Live_Ramp_Target_MWH"] = target.where(gate, np.nan)
    out["Day1_Live_Ramp_Yesterday_MWH"] = yesterday.where(gate, np.nan)
    out["Day1_Live_Ramp_Cal_MWH"] = lift
    source = str(cfg.get("source", "day1_live_same_day_ramp"))
    out["Day1_Live_Ramp_Source"] = np.where(
        lift.gt(0.0),
        source,
        np.where(gate, "already_at_live_ramp_target", "out_of_scope"),
    )

    if not lift.gt(0.0).any():
        return out
    out[base_col] = (base + lift).clip(lower=0.0)
    update_cols = list(also_update_cols)
    if base_col == "Final_Forecast_MWH" and "Calibrated_Forecast_MWH" in out.columns and "Calibrated_Forecast_MWH" not in update_cols:
        update_cols.append("Calibrated_Forecast_MWH")
    for col in update_cols:
        if col in out.columns:
            out[col] = (_as_num(out[col]) + lift).clip(lower=0.0)
    return out


def apply_multiday_live_heat_anchor_override(
    df: pd.DataFrame,
    config: dict | None = None,
    base_col: str = "Final_Forecast_MWH",
    also_update_cols: tuple[str, ...] = ("Stage_Selected_Forecast_MWH",),
) -> pd.DataFrame:
    """Carry an observed day-1 heat-wave level into days 2-7 with bounded discounts."""
    out = df.copy()
    cfg = _multiday_live_heat_anchor_cfg(config)
    defaults = {
        "MultiDay_Heat_Anchor_Gate": 0,
        "MultiDay_Heat_Anchor_Cal_MWH": 0.0,
        "MultiDay_Heat_Anchor_Target_MWH": np.nan,
        "MultiDay_Heat_Anchor_Anchor_Peak_MWH": np.nan,
        "MultiDay_Heat_Anchor_Anchor_Temp_F": np.nan,
        "MultiDay_Heat_Anchor_Observed_Delta_MWH": np.nan,
        "MultiDay_Heat_Anchor_Lead_Day": np.nan,
        "MultiDay_Heat_Anchor_Weekend_Discount_MWH": 0.0,
        "MultiDay_Heat_Anchor_Source": "none",
    }
    for col, default in defaults.items():
        if col not in out.columns:
            out[col] = default

    if not bool(cfg.get("enabled", False)) or out.empty or base_col not in out.columns or "DT" not in out.columns:
        return out

    dt = _local_datetime_series(out["DT"], out.index)
    if "Hour" not in out.columns:
        out["Hour"] = dt.dt.hour.astype("Int64").astype(float)
    hour = _as_num(out["Hour"]).fillna(-1).astype(int)
    forecast_day = _forecast_day_series(out, dt)
    if "Temperature_DailyMax" in out.columns:
        maxt = _as_num(out["Temperature_DailyMax"])
    elif "Temperature" in out.columns:
        maxt = _as_num(out["Temperature"])
    else:
        maxt = pd.Series(np.nan, index=out.index)

    base = _as_num(out[base_col])
    live_cal = _optional_num(out, "Day1_Live_Ramp_Cal_MWH", default=0.0).fillna(0.0)
    observed = _optional_num(out, "Day1_Live_Ramp_Observed_Delta_MWH", default=np.nan)
    observed_delta = float(observed.dropna().max()) if observed.notna().any() else np.nan
    out["MultiDay_Heat_Anchor_Observed_Delta_MWH"] = observed_delta

    min_observed = float(cfg.get("min_anchor_observed_delta_mwh", 20.0))
    if not np.isfinite(observed_delta) or observed_delta < min_observed:
        out["MultiDay_Heat_Anchor_Source"] = "missing_live_observed_delta"
        return out

    anchor_hours = {int(h) for h in cfg.get("anchor_hours", cfg.get("hours", [16, 17, 18, 19]))}
    anchor_mask = (
        forecast_day.eq(1)
        & hour.isin(anchor_hours)
        & maxt.ge(float(cfg.get("min_anchor_maxtemp_f", cfg.get("min_maxtemp_f", 102.0))))
        & base.notna()
    )
    min_live_lift = float(cfg.get("min_anchor_live_ramp_mwh", 8.0))
    if "Day1_Live_Ramp_Cal_MWH" in out.columns:
        anchor_mask &= live_cal.ge(min_live_lift)
    if not anchor_mask.any():
        out["MultiDay_Heat_Anchor_Source"] = "missing_day1_live_anchor"
        return out

    anchor_peak = float(base.loc[anchor_mask].max())
    anchor_temp = float(maxt.loc[forecast_day.eq(1)].max()) if maxt.loc[forecast_day.eq(1)].notna().any() else np.nan
    if not np.isfinite(anchor_peak) or not np.isfinite(anchor_temp):
        out["MultiDay_Heat_Anchor_Source"] = "missing_day1_live_anchor"
        return out

    out["MultiDay_Heat_Anchor_Anchor_Peak_MWH"] = anchor_peak
    out["MultiDay_Heat_Anchor_Anchor_Temp_F"] = anchor_temp

    hours = {int(h) for h in cfg.get("hours", [16, 17, 18, 19, 20])}
    lead_day = forecast_day - 1.0
    weekend = dt.dt.dayofweek.ge(5).fillna(False)
    sunday = dt.dt.dayofweek.eq(6).fillna(False)
    weekend_discount = pd.Series(0.0, index=out.index)
    weekend_discount = weekend_discount.mask(weekend, float(cfg.get("weekend_discount_mwh", 8.0)))
    weekend_discount = weekend_discount + sunday.astype(float) * float(cfg.get("sunday_extra_discount_mwh", 6.0))

    raw_offsets = cfg.get("hour_target_offsets_mwh", {}) or {}
    hour_offsets = hour.map(
        lambda h: float(raw_offsets.get(h, raw_offsets.get(str(h), 0.0))) if isinstance(raw_offsets, dict) else 0.0
    ).astype(float)
    temp_shortfall = (anchor_temp - maxt).clip(lower=0.0).fillna(0.0)
    temp_excess = (maxt - anchor_temp).clip(lower=0.0).fillna(0.0)
    target = (
        anchor_peak
        - float(cfg.get("anchor_discount_mwh", 5.0))
        - lead_day.fillna(0.0) * float(cfg.get("lead_decay_mwh_per_day", 0.75))
        - temp_shortfall * float(cfg.get("temp_shortfall_discount_mwh_per_f", 1.75))
        + temp_excess * float(cfg.get("temp_excess_credit_mwh_per_f", 1.0))
        - weekend_discount
        + hour_offsets
    )
    target = target.clip(upper=float(cfg.get("max_target_mwh", 365.0)))

    gate = (
        forecast_day.between(int(cfg.get("min_forecast_day", 2)), int(cfg.get("max_forecast_day", 7)))
        & hour.isin(hours)
        & maxt.ge(float(cfg.get("min_maxtemp_f", 102.0)))
        & maxt.le(float(cfg.get("max_maxtemp_f", 115.0)))
        & base.notna()
    )
    min_extreme_days = cfg.get("min_consecutive_extreme_hot_days_100")
    if min_extreme_days is not None and "ConsecutiveExtremeHotDays100" in out.columns:
        gate &= _as_num(out["ConsecutiveExtremeHotDays100"]).fillna(0.0).ge(float(min_extreme_days))
    if "IsHoliday" in out.columns and bool(cfg.get("exclude_holidays", False)):
        gate &= _as_num(out["IsHoliday"]).fillna(0.0).eq(0.0)

    lift = (target - base).clip(lower=0.0).fillna(0.0)
    lift = lift.clip(lower=0.0, upper=float(cfg.get("cap_mwh", 14.0)))
    lift = lift.where(gate, 0.0).fillna(0.0)

    out["MultiDay_Heat_Anchor_Gate"] = gate.astype(int)
    out["MultiDay_Heat_Anchor_Target_MWH"] = target.where(gate, np.nan)
    out["MultiDay_Heat_Anchor_Lead_Day"] = lead_day.where(gate, np.nan)
    out["MultiDay_Heat_Anchor_Weekend_Discount_MWH"] = weekend_discount.where(gate, 0.0)
    out["MultiDay_Heat_Anchor_Cal_MWH"] = lift
    source = str(cfg.get("source", "multiday_live_heat_anchor"))
    out["MultiDay_Heat_Anchor_Source"] = np.where(
        lift.gt(0.0),
        source,
        np.where(gate, "already_at_heat_anchor_target", "out_of_scope"),
    )

    if not lift.gt(0.0).any():
        return out
    out[base_col] = (base + lift).clip(lower=0.0)
    update_cols = list(also_update_cols)
    if base_col == "Final_Forecast_MWH" and "Calibrated_Forecast_MWH" in out.columns and "Calibrated_Forecast_MWH" not in update_cols:
        update_cols.append("Calibrated_Forecast_MWH")
    for col in update_cols:
        if col in out.columns:
            out[col] = (_as_num(out[col]) + lift).clip(lower=0.0)
    return out
