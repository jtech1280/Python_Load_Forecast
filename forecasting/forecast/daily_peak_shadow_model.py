from __future__ import annotations

"""Shadow daily-peak learner.

This layer models the daily system peak separately from the hourly forecast. It is
diagnostic by default: it writes a candidate peak-reconciled forecast column and
summary metrics, while leaving the production forecast untouched unless explicitly
promoted later.
"""

from typing import Any

import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingRegressor


DAILY_PEAK_SHADOW_COLUMNS = [
    "Daily_Peak_Model_Version",
    "Daily_Peak_Shadow_Mode",
    "Daily_Peak_Base_Forecast_MWH",
    "Daily_Peak_Correction_MWH",
    "Daily_Peak_Shadow_Adjusted_Forecast_MWH",
    "Daily_Peak_Correction_Applied_Flag",
    "Daily_Peak_Source",
    "Daily_Peak_Evaluation_Mode",
    "Daily_Peak_Base_DailyPeak_MWH",
    "Daily_Peak_Predicted_Residual_MWH",
    "Daily_Peak_Predicted_DailyPeak_MWH",
    "Daily_Peak_Base_PeakHour",
    "Daily_Peak_Predicted_PeakHour",
    "Daily_Peak_Timing_Shift_Hours",
    "Daily_Peak_Residual_MWH",
    "Daily_Peak_AbsError_MWH",
    "Daily_Peak_Delta_AbsError_MWH",
]


def _cfg(config: dict | None) -> dict:
    raw = config or {}
    if "daily_peak_shadow_model" in raw:
        return raw.get("daily_peak_shadow_model", {}) or {}
    cal = raw.get("calibration", {}) or {}
    if "daily_peak_shadow_model" in cal:
        return cal.get("daily_peak_shadow_model", {}) or {}
    stage_selector = (cal.get("stage_selector", {}) or {})
    return stage_selector.get("daily_peak_shadow_model", {}) or {}


def _as_num(value: Any, index: pd.Index | None = None, default: float = np.nan) -> pd.Series:
    if isinstance(value, pd.Series):
        raw = value
    else:
        raw = pd.Series(default, index=index)
    return pd.to_numeric(raw, errors="coerce")


def _optional_num(values: pd.DataFrame, *cols: str, default: float = np.nan) -> pd.Series:
    for col in cols:
        if col in values.columns:
            return _as_num(values[col], values.index).fillna(default)
    return pd.Series(default, index=values.index, dtype=float)


def _local_datetime(values: pd.DataFrame) -> pd.Series:
    raw = values.get("DT", pd.Series(pd.NaT, index=values.index))
    try:
        return pd.to_datetime(raw, errors="coerce")
    except ValueError:
        cleaned = raw.astype(str).str.strip().str.replace(r"(?:[+-]\d{2}:?\d{2}|Z)$", "", regex=True)
        return pd.to_datetime(cleaned, errors="coerce")


def _date(values: pd.DataFrame, dt: pd.Series | None = None) -> pd.Series:
    if "Date" in values.columns:
        raw = values["Date"]
        if raw.notna().any():
            return raw
    dt = dt if dt is not None else _local_datetime(values)
    return dt.dt.date


def _hour(values: pd.DataFrame, dt: pd.Series | None = None) -> pd.Series:
    dt = dt if dt is not None else _local_datetime(values)
    hour = _as_num(values.get("Hour", pd.Series(np.nan, index=values.index)), values.index)
    return hour.where(hour.notna(), dt.dt.hour).fillna(0.0)


def _month(values: pd.DataFrame, dt: pd.Series | None = None) -> pd.Series:
    dt = dt if dt is not None else _local_datetime(values)
    month = _as_num(values.get("Month", pd.Series(np.nan, index=values.index)), values.index)
    return month.where(month.notna(), dt.dt.month).fillna(1.0)


def _dow(values: pd.DataFrame, dt: pd.Series | None = None) -> pd.Series:
    dt = dt if dt is not None else _local_datetime(values)
    dow = _as_num(values.get("DOW", pd.Series(np.nan, index=values.index)), values.index)
    return dow.where(dow.notna(), dt.dt.dayofweek).fillna(0.0)


def _forecast_day(values: pd.DataFrame, dt: pd.Series | None = None) -> pd.Series:
    if "Forecast_Day" in values.columns:
        day = _as_num(values["Forecast_Day"], values.index)
        if day.notna().any():
            return day
    dt = dt if dt is not None else _local_datetime(values)
    valid = dt.notna()
    out = pd.Series(np.nan, index=values.index, dtype=float)
    if valid.any():
        first_day = dt.loc[valid].min().normalize()
        out.loc[valid] = (dt.loc[valid].dt.normalize() - first_day).dt.days.astype(float) + 1.0
    return out


def _cloud_norm(values: pd.DataFrame) -> pd.Series:
    cloud = _optional_num(values, "CloudCover_Norm", default=np.nan)
    if cloud.notna().any() and cloud.max(skipna=True) > 1.5:
        cloud = cloud / 100.0
    return cloud.clip(0.0, 1.0)


def _base_forecast(values: pd.DataFrame, forecast_col: str) -> pd.Series:
    if forecast_col in values.columns:
        return _as_num(values[forecast_col], values.index)
    for col in [
        "Final_Backtest_Forecast_MWH",
        "Final_Forecast_MWH",
        "Stage_Selected_Forecast_MWH",
        "Auto_Residual_Adjusted_Forecast_MWH",
        "Recent_Corrected_Forecast_MWH",
        "Calibrated_Forecast_MWH",
        "Raw_Forecast_MWH",
    ]:
        if col in values.columns:
            return _as_num(values[col], values.index)
    return pd.Series(np.nan, index=values.index, dtype=float)


def _first_valid(series: pd.Series, default: float = np.nan) -> float:
    valid = _as_num(series).dropna()
    return float(valid.iloc[0]) if not valid.empty else float(default)


def _max_valid(series: pd.Series, default: float = np.nan) -> float:
    valid = _as_num(series).dropna()
    return float(valid.max()) if not valid.empty else float(default)


def _mean_valid(series: pd.Series, default: float = np.nan) -> float:
    valid = _as_num(series).dropna()
    return float(valid.mean()) if not valid.empty else float(default)


def _sum_valid(series: pd.Series, default: float = np.nan) -> float:
    valid = _as_num(series).dropna()
    return float(valid.sum()) if not valid.empty else float(default)


def _peak_idx(group: pd.DataFrame, col: str) -> Any | None:
    if col not in group.columns:
        return None
    values = _as_num(group[col], group.index)
    values = values.dropna()
    if values.empty:
        return None
    return values.idxmax()


def _component_peak(group: pd.DataFrame, col: str) -> float:
    return _max_valid(group[col]) if col in group.columns else np.nan


def _model_frame(values: pd.DataFrame, forecast_col: str, config: dict | None) -> pd.DataFrame:
    """Collapse hourly rows to one model row per forecast date."""
    cfg = _cfg(config)
    peak_hours = {int(h) for h in cfg.get("peak_hours", [14, 15, 16, 17, 18, 19, 20, 21])}
    min_peak_rows = int(cfg.get("min_peak_window_rows_per_day", 3))

    work = values.copy()
    dt = _local_datetime(work)
    work["_DailyPeak_DT"] = dt
    work["_DailyPeak_Date"] = _date(work, dt=dt)
    work["_DailyPeak_Hour"] = _hour(work, dt=dt)
    work["_DailyPeak_Base"] = _base_forecast(work, forecast_col=forecast_col)
    actual_col = "Actual_MWH" if "Actual_MWH" in work.columns else ("Actual" if "Actual" in work.columns else None)
    work["_DailyPeak_Actual"] = (
        _as_num(work[actual_col], work.index) if actual_col is not None else pd.Series(np.nan, index=work.index)
    )
    work["_DailyPeak_Cloud"] = _cloud_norm(work)

    rows: list[dict[str, Any]] = []
    for date, group in work.groupby("_DailyPeak_Date", dropna=False):
        group = group[group["_DailyPeak_DT"].notna()].copy()
        if group.empty:
            continue
        peak_group = group[group["_DailyPeak_Hour"].astype("Int64").isin(peak_hours).fillna(False)].copy()
        if peak_group.empty:
            peak_group = group.copy()
        base_peak_idx = _peak_idx(peak_group, "_DailyPeak_Base")
        if base_peak_idx is None:
            continue
        actual_peak_idx = _peak_idx(peak_group, "_DailyPeak_Actual")
        base_peak_row = group.loc[base_peak_idx]
        actual_peak_row = group.loc[actual_peak_idx] if actual_peak_idx is not None else None

        base_peak = float(base_peak_row["_DailyPeak_Base"])
        actual_peak = float(actual_peak_row["_DailyPeak_Actual"]) if actual_peak_row is not None else np.nan
        base_peak_hour = float(base_peak_row["_DailyPeak_Hour"])
        actual_peak_hour = float(actual_peak_row["_DailyPeak_Hour"]) if actual_peak_row is not None else np.nan

        forecast_day = _forecast_day(group, dt=group["_DailyPeak_DT"])
        month = _month(group, dt=group["_DailyPeak_DT"])
        dow = _dow(group, dt=group["_DailyPeak_DT"])
        temp = _optional_num(group, "Temperature", default=np.nan)
        daily_max = _optional_num(group, "Temperature_DailyMax", default=np.nan)
        if not daily_max.notna().any():
            daily_max = temp.groupby(group["_DailyPeak_Date"]).transform("max")

        raw = _optional_num(group, "Raw_Forecast_MWH", default=np.nan)
        same7 = _optional_num(group, "MWH_SameHour7DayMean", "Baseline_Rolling7DaySameHourAvg_MWH", default=np.nan)
        lag24 = _optional_num(group, "MWH_Lag24", "Baseline_SameHourYesterday_MWH", default=np.nan)
        xgb = _optional_num(group, "XGB_Pred_MWH", default=np.nan)
        lgb = _optional_num(group, "LGB_Pred_MWH", default=np.nan)
        cat = _optional_num(group, "CatBoost_Pred_MWH", default=np.nan)
        prophet = _optional_num(group, "Prophet_Pred_MWH", default=np.nan)
        components = pd.concat([xgb, lgb, cat, prophet], axis=1)

        base_peak_features = group.loc[[base_peak_idx]]
        peak_components = pd.Series(
            [
                _first_valid(base_peak_features.get("XGB_Pred_MWH", pd.Series(np.nan))),
                _first_valid(base_peak_features.get("LGB_Pred_MWH", pd.Series(np.nan))),
                _first_valid(base_peak_features.get("CatBoost_Pred_MWH", pd.Series(np.nan))),
                _first_valid(base_peak_features.get("Prophet_Pred_MWH", pd.Series(np.nan))),
            ],
            dtype=float,
        )

        row = {
            "Date": date,
            "Forecast_Day": _mean_valid(forecast_day),
            "Forecast_Lead_Hour_Mean": _mean_valid(_optional_num(group, "Forecast_Lead_Hour", default=np.nan)),
            "Month": _first_valid(month),
            "DOW": _first_valid(dow),
            "IsWeekend": _max_valid(_optional_num(group, "IsWeekend", default=0.0), default=0.0),
            "IsHoliday": _max_valid(_optional_num(group, "IsHoliday", default=0.0), default=0.0),
            "Peak_Window_Row_Count": int(len(peak_group)),
            "Peak_Window_Complete_Flag": int(len(peak_group) >= min_peak_rows),
            "Base_DailyPeak_MWH": base_peak,
            "Base_PeakHour": base_peak_hour,
            "Base_PeakHour_Sin": float(np.sin(2.0 * np.pi * base_peak_hour / 24.0)),
            "Base_PeakHour_Cos": float(np.cos(2.0 * np.pi * base_peak_hour / 24.0)),
            "Base_DailyEnergy_MWH": _sum_valid(group["_DailyPeak_Base"]),
            "Base_PeakMinusDailyMean_MWH": base_peak - _mean_valid(group["_DailyPeak_Base"]),
            "Base_PeakMinusSameHour7DayPeak_MWH": base_peak - _component_peak(peak_group, "MWH_SameHour7DayMean"),
            "Base_PeakMinusLag24Peak_MWH": base_peak - _component_peak(peak_group, "MWH_Lag24"),
            "Raw_DailyPeak_MWH": _component_peak(peak_group, "Raw_Forecast_MWH"),
            "XGB_DailyPeak_MWH": _component_peak(peak_group, "XGB_Pred_MWH"),
            "LGB_DailyPeak_MWH": _component_peak(peak_group, "LGB_Pred_MWH"),
            "CatBoost_DailyPeak_MWH": _component_peak(peak_group, "CatBoost_Pred_MWH"),
            "Prophet_DailyPeak_MWH": _component_peak(peak_group, "Prophet_Pred_MWH"),
            "ModelSpread_AtBasePeak_MWH": _max_valid(peak_components) - _mean_valid(peak_components, default=0.0),
            "ComponentSpread_DailyPeak_MWH": _max_valid(components.max(axis=0, skipna=True))
            - _max_valid(components.min(axis=0, skipna=True)),
            "Temperature_DailyMax": _max_valid(daily_max),
            "Temperature_AtBasePeak": _first_valid(base_peak_features.get("Temperature", pd.Series(np.nan))),
            "TempDrop_FromDailyMax_AtBasePeak_F": _first_valid(
                base_peak_features.get("Temperature_Drop_From_DailyMax_F", pd.Series(np.nan)),
                default=_max_valid(daily_max) - _first_valid(base_peak_features.get("Temperature", pd.Series(np.nan))),
            ),
            "TempDrop_Next1Hr_AtBasePeak_F": _first_valid(base_peak_features.get("TempDrop_Next1Hr_F", pd.Series(np.nan))),
            "TempDrop_Next2Hr_AtBasePeak_F": _first_valid(base_peak_features.get("TempDrop_Next2Hr_F", pd.Series(np.nan))),
            "TempDrop_Next3Hr_AtBasePeak_F": _first_valid(base_peak_features.get("TempDrop_Next3Hr_F", pd.Series(np.nan))),
            "CloudCover_Max": _max_valid(group["_DailyPeak_Cloud"]),
            "CloudCover_MeanPeakWindow": _mean_valid(peak_group["_DailyPeak_Cloud"]),
            "ClearSkyIndex_MinPeakWindow": _max_valid(-_optional_num(peak_group, "ClearSky_Index", default=np.nan)) * -1.0,
            "BTM_Solar_Loss_Max_MW": _max_valid(
                _optional_num(peak_group, "BTM_Solar_Loss_From_ClearSky_MW", "Midday_Overcast_Solar_Loss_MW", default=0.0),
                default=0.0,
            ),
            "BTM_Solar_Proxy_Max_MW": _max_valid(_optional_num(peak_group, "BTM_Solar_Proxy_MW", default=0.0), default=0.0),
            "WindSpeed_Max_Mph": _max_valid(_optional_num(peak_group, "WindSpeed_Mph", default=np.nan)),
            "Westerly_Flow_Max_Mph": _max_valid(_optional_num(peak_group, "Westerly_Flow_Mph", default=0.0), default=0.0),
            "WesterlyFlow_Next3Hr_Ramp_Max_Mph": _max_valid(
                _optional_num(peak_group, "WesterlyFlow_Next3Hr_Ramp_Mph", default=np.nan)
            ),
            "WindRamp_Next3Hr_Max_Mph": _max_valid(_optional_num(peak_group, "WindRamp_Next3Hr_Mph", default=np.nan)),
            "DeltaBreeze_Cooling_Flag_Max": _max_valid(_optional_num(peak_group, "DeltaBreeze_Cooling_Flag", default=0.0), default=0.0),
            "DeltaBreeze_Westerly_Flow_Flag_Max": _max_valid(
                _optional_num(peak_group, "DeltaBreeze_Westerly_Flow_Flag", "Westerly_Flow_Flag", default=0.0),
                default=0.0,
            ),
            "DeltaBreeze_Cooling_Signal_Max": _max_valid(_optional_num(peak_group, "DeltaBreeze_Cooling_Signal", default=0.0), default=0.0),
            "DeltaBreeze_PostPeak_LoadDecay_Signal_Max": _max_valid(
                _optional_num(peak_group, "DeltaBreeze_PostPeak_LoadDecay_Signal", default=0.0),
                default=0.0,
            ),
            "PostPeak_LoadDecay_1Hr_Min_MWH": _max_valid(-_optional_num(peak_group, "PostPeak_LoadDecay_1Hr_MWH", default=np.nan)) * -1.0,
            "PostPeak_LoadDecay_2Hr_Min_MWH": _max_valid(-_optional_num(peak_group, "PostPeak_LoadDecay_2Hr_MWH", default=np.nan)) * -1.0,
            "PostPeak_DecayVs7Day_Min_MWH": _max_valid(
                -_optional_num(peak_group, "PostPeak_LoadDecay_VsSameHour7DayMean_MWH", default=np.nan)
            ) * -1.0,
            "Recent_Level_Correction_Max_MWH": _max_valid(_optional_num(peak_group, "Recent_Level_Correction_MWH", default=0.0), default=0.0),
            "Focused_Guard_Max_MWH": _max_valid(_optional_num(peak_group, "Focused_Scorecard_Guard_MWH", default=0.0), default=0.0),
            "Auto_Residual_Max_MWH": _max_valid(_optional_num(peak_group, "Auto_Residual_Correction_MWH", default=0.0), default=0.0),
            "Weather_Robustness_Hedge_Max_MWH": _max_valid(
                _optional_num(peak_group, "Weather_Robustness_Hedge_MWH", default=0.0),
                default=0.0,
            ),
            "Actual_DailyPeak_MWH": actual_peak,
            "Actual_PeakHour": actual_peak_hour,
            "Target_PeakResidual_MWH": actual_peak - base_peak if np.isfinite(actual_peak) else np.nan,
            "Target_TimingShift_Hours": actual_peak_hour - base_peak_hour
            if np.isfinite(actual_peak_hour) and np.isfinite(base_peak_hour)
            else np.nan,
        }
        rows.append(row)

    out = pd.DataFrame(rows)
    if out.empty:
        return out
    for col in out.columns:
        if col != "Date":
            out[col] = pd.to_numeric(out[col], errors="coerce")
    return out


def _feature_columns(daily: pd.DataFrame) -> list[str]:
    blocked = {
        "Date",
        "Actual_DailyPeak_MWH",
        "Actual_PeakHour",
        "Target_PeakResidual_MWH",
        "Target_TimingShift_Hours",
    }
    return [c for c in daily.columns if c not in blocked]


def _new_model(cfg: dict, *, min_samples_leaf: int) -> HistGradientBoostingRegressor:
    return HistGradientBoostingRegressor(
        loss=str(cfg.get("loss", "absolute_error")),
        learning_rate=float(cfg.get("learning_rate", 0.04)),
        max_iter=int(cfg.get("max_iter", 120)),
        max_leaf_nodes=int(cfg.get("max_leaf_nodes", 16)),
        min_samples_leaf=max(2, int(min_samples_leaf)),
        l2_regularization=float(cfg.get("l2_regularization", 5.0)),
        random_state=int(cfg.get("random_state", 42)),
    )


def _optional_cfg_float(cfg: dict, key: str) -> float | None:
    value = cfg.get(key)
    if value in {None, ""}:
        return None
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def _valid_training_mask(daily: pd.DataFrame, cfg: dict) -> tuple[pd.Series, pd.Series]:
    """Return (clipped target, valid-day mask) shared by fit and walk-forward eval."""
    target = _as_num(daily["Target_PeakResidual_MWH"], daily.index)
    target_clip = float(cfg.get("target_clip_mwh", 35.0))
    if target_clip > 0.0:
        target = target.clip(-target_clip, target_clip)
    valid = (
        target.notna()
        & _as_num(daily["Base_DailyPeak_MWH"], daily.index).notna()
        & _as_num(daily["Peak_Window_Complete_Flag"], daily.index).eq(1)
    )
    return target, valid


def _fit_daily_peak_artifact(
    daily: pd.DataFrame,
    cfg: dict,
    *,
    forecast_col: str = "Final_Backtest_Forecast_MWH",
    fit_timing: bool = True,
    enforce_min_days: bool = True,
) -> dict | None:
    """Fit the residual (and optional timing) learner from a prepared daily frame.

    Factored out of :func:`build_daily_peak_shadow_model` so the leakage-safe
    walk-forward evaluator can refit on earlier-only origins without duplicating
    the training logic.
    """
    if daily is None or daily.empty:
        return None
    target, valid = _valid_training_mask(daily, cfg)
    min_days = int(cfg.get("min_days", 45))
    if enforce_min_days and int(valid.sum()) < min_days:
        return None
    if int(valid.sum()) < 2:
        return None

    features = daily.loc[valid, _feature_columns(daily)].copy()
    fill = features.median(numeric_only=True).replace([np.inf, -np.inf], np.nan).fillna(0.0)
    x = features.fillna(fill).replace([np.inf, -np.inf], 0.0).fillna(0.0)
    y = target.loc[valid].copy()

    model = _new_model(cfg, min_samples_leaf=int(cfg.get("min_samples_leaf", 10)))
    model.fit(x, y)

    timing_model = None
    if fit_timing:
        timing_valid = _as_num(daily.loc[valid, "Target_TimingShift_Hours"], daily.loc[valid].index)
        max_timing = float(cfg.get("target_timing_clip_hours", 4.0))
        timing_valid = timing_valid.clip(-max_timing, max_timing)
        if bool(cfg.get("timing_model_enabled", True)) and timing_valid.notna().sum() >= min_days:
            timing_model = _new_model(cfg, min_samples_leaf=int(cfg.get("timing_min_samples_leaf", cfg.get("min_samples_leaf", 10))))
            timing_model.fit(x.loc[timing_valid.notna()], timing_valid.loc[timing_valid.notna()])

    return {
        "residual_model": model,
        "timing_model": timing_model,
        "fill_values": fill,
        "feature_columns": list(x.columns),
        "metadata": {
            "model_version": str(cfg.get("model_version", "daily_peak_shadow_v1")),
            "training_days": int(valid.sum()),
            "training_start_date": str(daily.loc[valid, "Date"].min()),
            "training_end_date": str(daily.loc[valid, "Date"].max()),
            "target_peak_residual_mean_mwh": float(y.mean()),
            "target_peak_residual_mae_mwh": float(y.abs().mean()),
            "forecast_col": forecast_col,
            "shadow_mode": bool(cfg.get("shadow_mode", True)),
            "timing_model_enabled": timing_model is not None,
        },
    }


def _predict_daily_peak_residual(daily_subset: pd.DataFrame, artifact: dict, cfg: dict) -> pd.Series:
    """Blend/cap/threshold a fitted artifact's raw prediction into a peak correction."""
    columns = list(artifact.get("feature_columns") or [])
    features = daily_subset.reindex(columns=columns)
    fill_values = artifact.get("fill_values", pd.Series(dtype=float))
    x = features.fillna(fill_values).replace([np.inf, -np.inf], 0.0).fillna(0.0)
    pred = pd.Series(np.asarray(artifact["residual_model"].predict(x), dtype=float), index=daily_subset.index)
    pred = (pred * float(cfg.get("blend", 0.50))).clip(
        -float(cfg.get("cap_mwh", 10.0)),
        float(cfg.get("cap_mwh", 10.0)),
    )
    min_abs = float(cfg.get("min_abs_correction_mwh", 0.50))
    if min_abs > 0.0:
        pred.loc[pred.abs().lt(min_abs)] = 0.0
    return pred


def build_daily_peak_shadow_model(
    backtest_df: pd.DataFrame,
    config: dict | None,
    *,
    forecast_col: str = "Final_Backtest_Forecast_MWH",
) -> dict | None:
    cfg = _cfg(config)
    if backtest_df is None or backtest_df.empty or not bool(cfg.get("enabled", False)):
        return None

    daily = _model_frame(backtest_df, forecast_col=forecast_col, config=config)
    if daily.empty:
        return None
    return _fit_daily_peak_artifact(daily, cfg, forecast_col=forecast_col, fit_timing=True)


def walk_forward_daily_peak_eval(daily: pd.DataFrame, config: dict | None) -> pd.DataFrame:
    """Leakage-safe expanding-window evaluation of the daily-peak learner.

    The daily rows are ordered by date; using an expanding origin, the residual
    learner is refit on all valid days strictly *before* each holdout block and
    used to predict the held-out later days. Every scored day is therefore
    out-of-sample, so the resulting delta is an honest estimate of the layer's
    benefit -- unlike the in-sample summary, which grades the learner on the same
    days it was trained on. The correction is scored at the daily-peak level
    (Base_DailyPeak_MWH + predicted residual vs Actual_DailyPeak_MWH), which is
    exactly the object the model predicts.
    """
    cfg = _cfg(config)
    if daily is None or daily.empty:
        return pd.DataFrame()

    work = daily.sort_values("Date").reset_index(drop=True)
    _, valid = _valid_training_mask(work, cfg)
    actual_ok = _as_num(work["Actual_DailyPeak_MWH"], work.index).notna()
    work = work.loc[valid & actual_ok].sort_values("Date").reset_index(drop=True)

    n = len(work)
    min_days = int(cfg.get("min_days", 45))
    if n <= min_days:
        return pd.DataFrame()

    step = max(1, int(cfg.get("holdout_eval_step_days", 7)))
    min_app = _optional_cfg_float(cfg, "min_application_forecast_day")
    max_app = _optional_cfg_float(cfg, "max_application_forecast_day")

    rows: list[dict[str, Any]] = []
    start = min_days
    while start < n:
        train = work.iloc[:start]
        test = work.iloc[start:start + step]
        artifact = _fit_daily_peak_artifact(train, cfg, fit_timing=False, enforce_min_days=False)
        pred = (
            _predict_daily_peak_residual(test, artifact, cfg)
            if artifact is not None
            else pd.Series(0.0, index=test.index)
        )
        for pos, drow in test.iterrows():
            base_peak = float(drow.get("Base_DailyPeak_MWH", np.nan))
            actual_peak = float(drow.get("Actual_DailyPeak_MWH", np.nan))
            if not (np.isfinite(base_peak) and np.isfinite(actual_peak)):
                continue
            correction = float(pred.loc[pos]) if pos in pred.index else 0.0
            forecast_day = float(drow.get("Forecast_Day", np.nan))
            in_horizon = True
            if min_app is not None and (not np.isfinite(forecast_day) or forecast_day < min_app):
                in_horizon = False
            if max_app is not None and (not np.isfinite(forecast_day) or forecast_day > max_app):
                in_horizon = False
            applied = bool(in_horizon and abs(correction) > 0.0)
            effective = correction if applied else 0.0
            shadow_peak = base_peak + effective
            rows.append({
                "Date": drow.get("Date"),
                "Applied": int(applied),
                "Base_DailyPeak_AbsError_MWH": abs(actual_peak - base_peak),
                "Shadow_DailyPeak_AbsError_MWH": abs(actual_peak - shadow_peak),
                "Correction_MWH": effective,
            })
        start += step
    return pd.DataFrame(rows)


def _init_columns(
    out: pd.DataFrame,
    base: pd.Series,
    *,
    model_version: str,
    source: str,
    evaluation_mode: str,
    shadow_mode: bool,
) -> pd.DataFrame:
    out["Daily_Peak_Model_Version"] = model_version
    out["Daily_Peak_Shadow_Mode"] = int(bool(shadow_mode))
    out["Daily_Peak_Base_Forecast_MWH"] = base
    out["Daily_Peak_Correction_MWH"] = 0.0
    out["Daily_Peak_Shadow_Adjusted_Forecast_MWH"] = base
    out["Daily_Peak_Correction_Applied_Flag"] = 0
    out["Daily_Peak_Source"] = source
    out["Daily_Peak_Evaluation_Mode"] = evaluation_mode
    out["Daily_Peak_Base_DailyPeak_MWH"] = np.nan
    out["Daily_Peak_Predicted_Residual_MWH"] = np.nan
    out["Daily_Peak_Predicted_DailyPeak_MWH"] = np.nan
    out["Daily_Peak_Base_PeakHour"] = np.nan
    out["Daily_Peak_Predicted_PeakHour"] = np.nan
    out["Daily_Peak_Timing_Shift_Hours"] = 0.0
    actual_col = "Actual_MWH" if "Actual_MWH" in out.columns else ("Actual" if "Actual" in out.columns else None)
    actual = _as_num(out.get(actual_col, pd.Series(np.nan, index=out.index)), out.index)
    residual = actual - base
    out["Daily_Peak_Residual_MWH"] = residual
    out["Daily_Peak_AbsError_MWH"] = residual.abs()
    out["Daily_Peak_Delta_AbsError_MWH"] = 0.0
    return out


def _hour_weights(hours: pd.Series, center_hour: float, spread_hours: float) -> pd.Series:
    h = _as_num(hours, hours.index)
    if spread_hours <= 0:
        weights = h.eq(round(center_hour)).astype(float)
    else:
        weights = np.exp(-0.5 * ((h - float(center_hour)) / float(spread_hours)) ** 2)
    out = pd.Series(weights, index=hours.index, dtype=float).fillna(0.0)
    max_weight = out.max(skipna=True)
    if not np.isfinite(max_weight) or max_weight <= 0.0:
        return pd.Series(0.0, index=hours.index, dtype=float)
    return out / max_weight


def apply_daily_peak_shadow_model(
    df: pd.DataFrame,
    artifact: dict | None,
    config: dict | None,
    *,
    forecast_col: str = "Final_Forecast_MWH",
    also_update_cols: tuple[str, ...] = (),
    force_shadow: bool | None = None,
    evaluation_mode: str = "shadow",
) -> pd.DataFrame:
    """Apply the daily peak model.

    By default this is a shadow stage. Promotion can be tested later by setting
    ``shadow_mode: false`` in the model config, but the current config keeps it
    diagnostic only.
    """
    out = df.copy()
    cfg = _cfg(config)
    shadow_mode = bool(cfg.get("shadow_mode", True)) if force_shadow is None else bool(force_shadow)
    model_version = str(cfg.get("model_version", "daily_peak_shadow_v1"))
    base = _base_forecast(out, forecast_col=forecast_col)
    out = _init_columns(
        out,
        base,
        model_version=model_version,
        source="disabled",
        evaluation_mode=evaluation_mode,
        shadow_mode=shadow_mode,
    )
    if out.empty or not bool(cfg.get("enabled", False)):
        return out
    if not artifact or artifact.get("residual_model") is None:
        out["Daily_Peak_Source"] = "insufficient_history"
        return out
    columns = list(artifact.get("feature_columns") or [])
    if not columns:
        out["Daily_Peak_Source"] = "no_feature_columns"
        return out

    daily = _model_frame(out, forecast_col=forecast_col, config=config)
    if daily.empty:
        out["Daily_Peak_Source"] = "no_daily_peak_rows"
        return out

    features = daily.reindex(columns=columns)
    fill_values = artifact.get("fill_values", pd.Series(dtype=float))
    x = features.fillna(fill_values).replace([np.inf, -np.inf], 0.0).fillna(0.0)
    pred_residual = pd.Series(np.asarray(artifact["residual_model"].predict(x), dtype=float), index=daily.index)
    pred_residual = (pred_residual * float(cfg.get("blend", 0.50))).clip(
        -float(cfg.get("cap_mwh", 10.0)),
        float(cfg.get("cap_mwh", 10.0)),
    )
    min_abs = float(cfg.get("min_abs_correction_mwh", 0.50))
    if min_abs > 0.0:
        pred_residual.loc[pred_residual.abs().lt(min_abs)] = 0.0

    timing_shift = pd.Series(0.0, index=daily.index, dtype=float)
    timing_model = artifact.get("timing_model")
    if timing_model is not None:
        raw_timing = pd.Series(np.asarray(timing_model.predict(x), dtype=float), index=daily.index)
        timing_shift = raw_timing * float(cfg.get("timing_blend", 0.50))
    max_shift = float(cfg.get("max_timing_shift_hours", 3.0))
    timing_shift = timing_shift.clip(-max_shift, max_shift)

    adjustment_hours = {int(h) for h in cfg.get("adjustment_hours", cfg.get("peak_hours", [14, 15, 16, 17, 18, 19, 20, 21]))}
    spread_hours = float(cfg.get("spread_hours", 1.5))
    row_dt = _local_datetime(out)
    row_date = _date(out, dt=row_dt)
    row_hour = _hour(out, dt=row_dt)
    total_correction = pd.Series(0.0, index=out.index, dtype=float)
    source = pd.Series("out_of_scope", index=out.index, dtype="object")

    for pos, drow in daily.iterrows():
        date = drow.get("Date")
        day_mask = row_date.eq(date)
        if not day_mask.any():
            continue
        forecast_day = float(drow.get("Forecast_Day", np.nan))
        min_application_day = _optional_cfg_float(cfg, "min_application_forecast_day")
        max_application_day = _optional_cfg_float(cfg, "max_application_forecast_day")
        out.loc[day_mask, "Daily_Peak_Base_DailyPeak_MWH"] = float(drow.get("Base_DailyPeak_MWH", np.nan))
        out.loc[day_mask, "Daily_Peak_Predicted_Residual_MWH"] = float(pred_residual.loc[pos])
        out.loc[day_mask, "Daily_Peak_Predicted_DailyPeak_MWH"] = (
            float(drow.get("Base_DailyPeak_MWH", np.nan)) + float(pred_residual.loc[pos])
        )
        base_peak_hour = float(drow.get("Base_PeakHour", np.nan))
        if not np.isfinite(base_peak_hour):
            source.loc[day_mask] = "invalid_base_peak"
            continue
        if (
            (min_application_day is not None and (not np.isfinite(forecast_day) or forecast_day < min_application_day))
            or (max_application_day is not None and (not np.isfinite(forecast_day) or forecast_day > max_application_day))
        ):
            source.loc[day_mask] = "horizon_out_of_scope"
            out.loc[day_mask, "Daily_Peak_Base_PeakHour"] = base_peak_hour
            out.loc[day_mask, "Daily_Peak_Predicted_PeakHour"] = base_peak_hour
            out.loc[day_mask, "Daily_Peak_Timing_Shift_Hours"] = 0.0
            continue
        center = base_peak_hour + float(timing_shift.loc[pos])
        if adjustment_hours:
            center = float(min(max(round(center), min(adjustment_hours)), max(adjustment_hours)))
        else:
            center = float(round(center))
        candidate_mask = day_mask & row_hour.astype("Int64").isin(adjustment_hours).fillna(False)
        if not candidate_mask.any():
            source.loc[day_mask] = "no_adjustment_window_rows"
            continue
        weights = _hour_weights(row_hour.loc[candidate_mask], center, spread_hours=spread_hours)
        correction = float(pred_residual.loc[pos])
        total_correction.loc[candidate_mask] = correction * weights
        source.loc[day_mask] = "daily_peak_shadow_zeroed" if abs(correction) <= 1e-12 else "daily_peak_shadow"
        out.loc[day_mask, "Daily_Peak_Base_PeakHour"] = base_peak_hour
        out.loc[day_mask, "Daily_Peak_Predicted_PeakHour"] = center
        out.loc[day_mask, "Daily_Peak_Timing_Shift_Hours"] = float(timing_shift.loc[pos])

    valid = base.notna()
    total_correction = total_correction.where(valid, 0.0)
    adjusted = (base + total_correction).clip(lower=0.0)
    source.loc[base.isna()] = "invalid_base_forecast"

    out["Daily_Peak_Correction_MWH"] = total_correction
    out["Daily_Peak_Shadow_Adjusted_Forecast_MWH"] = adjusted
    out["Daily_Peak_Correction_Applied_Flag"] = total_correction.ne(0.0).astype(int)
    out["Daily_Peak_Source"] = source

    actual_col = "Actual_MWH" if "Actual_MWH" in out.columns else ("Actual" if "Actual" in out.columns else None)
    actual = _as_num(out.get(actual_col, pd.Series(np.nan, index=out.index)), out.index)
    base_residual = actual - base
    shadow_residual = actual - adjusted
    out["Daily_Peak_Residual_MWH"] = shadow_residual
    out["Daily_Peak_AbsError_MWH"] = shadow_residual.abs()
    out["Daily_Peak_Delta_AbsError_MWH"] = shadow_residual.abs() - base_residual.abs()

    if not shadow_mode and forecast_col in out.columns:
        out[forecast_col] = adjusted
        if forecast_col == "Final_Backtest_Forecast_MWH" and "Final_Forecast_MWH" in out.columns:
            out["Final_Forecast_MWH"] = adjusted
        for col in also_update_cols:
            if col in out.columns and col != forecast_col:
                out[col] = adjusted
        if forecast_col in {"Final_Backtest_Forecast_MWH", "Final_Forecast_MWH"} and actual.notna().any():
            out["Final_Residual_MWH"] = actual - _as_num(out[forecast_col], out.index)
            out["Final_AbsError_MWH"] = out["Final_Residual_MWH"].abs()
            out["Final_APE"] = np.where(
                actual.abs() > 1e-9,
                out["Final_AbsError_MWH"] / actual.abs() * 100.0,
                np.nan,
            )
    return out


def _daily_peak_eval_rows(df: pd.DataFrame, config: dict | None = None) -> pd.DataFrame:
    if df is None or df.empty or "Daily_Peak_Shadow_Adjusted_Forecast_MWH" not in df.columns:
        return pd.DataFrame()
    work = df.copy()
    dt = _local_datetime(work)
    work["_Date"] = _date(work, dt=dt)
    # Score the daily peak on the same afternoon window used to train the learner
    # (config peak_hours), so the training target, application window, and this
    # evaluation share one peak definition instead of three.
    peak_hours = {int(h) for h in _cfg(config).get("peak_hours", [14, 15, 16, 17, 18, 19, 20, 21])}
    work["_PeakHour"] = _hour(work, dt=dt)
    actual_col = "Actual_MWH" if "Actual_MWH" in work.columns else ("Actual" if "Actual" in work.columns else None)
    if actual_col is None:
        return pd.DataFrame()
    rows = []
    for date, group in work.groupby("_Date", dropna=False):
        group = group.dropna(subset=[actual_col, "Daily_Peak_Base_Forecast_MWH", "Daily_Peak_Shadow_Adjusted_Forecast_MWH"])
        if group.empty:
            continue
        peak_group = group[group["_PeakHour"].astype("Int64").isin(peak_hours).fillna(False)]
        if not peak_group.empty:
            group = peak_group
        actual = _as_num(group[actual_col], group.index)
        base = _as_num(group["Daily_Peak_Base_Forecast_MWH"], group.index)
        shadow = _as_num(group["Daily_Peak_Shadow_Adjusted_Forecast_MWH"], group.index)
        actual_peak_idx = actual.idxmax()
        row = {
            "Date": date,
            "Actual_DailyPeak_MWH": float(actual.max()),
            "Baseline_DailyPeak_MWH": float(base.max()),
            "Shadow_DailyPeak_MWH": float(shadow.max()),
            "Baseline_DailyPeak_AbsError_MWH": float(abs(actual.max() - base.max())),
            "Shadow_DailyPeak_AbsError_MWH": float(abs(actual.max() - shadow.max())),
            "Baseline_AtActualPeak_AbsError_MWH": float(abs(actual.loc[actual_peak_idx] - base.loc[actual_peak_idx])),
            "Shadow_AtActualPeak_AbsError_MWH": float(abs(actual.loc[actual_peak_idx] - shadow.loc[actual_peak_idx])),
            "Daily_Peak_Correction_Applied_Flag": int(
                _as_num(group.get("Daily_Peak_Correction_Applied_Flag", pd.Series(0, index=group.index)), group.index).eq(1).any()
            ),
            "Mean_Correction_MWH": float(_as_num(group.get("Daily_Peak_Correction_MWH", pd.Series(0.0, index=group.index)), group.index).mean()),
            "MaxAbs_Correction_MWH": float(_as_num(group.get("Daily_Peak_Correction_MWH", pd.Series(0.0, index=group.index)), group.index).abs().max()),
        }
        row["Delta_DailyPeak_AbsError_MWH"] = row["Shadow_DailyPeak_AbsError_MWH"] - row["Baseline_DailyPeak_AbsError_MWH"]
        row["Delta_AtActualPeak_AbsError_MWH"] = row["Shadow_AtActualPeak_AbsError_MWH"] - row["Baseline_AtActualPeak_AbsError_MWH"]
        rows.append(row)
    return pd.DataFrame(rows)


def daily_peak_shadow_summary(backtest_df: pd.DataFrame | None, artifact: dict | None, config: dict | None) -> dict:
    cfg = _cfg(config)
    summary: dict[str, Any] = {
        "enabled": bool(cfg.get("enabled", False)),
        "shadow_mode": bool(cfg.get("shadow_mode", True)),
        "model_version": str(cfg.get("model_version", "daily_peak_shadow_v1")),
    }
    if artifact and artifact.get("metadata"):
        summary["artifact"] = dict(artifact.get("metadata") or {})

    # Leakage-safe, out-of-sample evaluation. The in-sample block below grades the
    # learner on the same days it was trained on and is therefore optimistic; this
    # expanding walk-forward refits on earlier-only origins and scores held-out
    # later days, so the delta here is the honest number for a promote decision.
    if bool(cfg.get("out_of_sample_eval_enabled", True)) and backtest_df is not None and not backtest_df.empty:
        forecast_col = "Final_Backtest_Forecast_MWH"
        if artifact and isinstance(artifact.get("metadata"), dict):
            forecast_col = str(artifact["metadata"].get("forecast_col", forecast_col))
        try:
            oos_daily = _model_frame(backtest_df, forecast_col=forecast_col, config=config)
            oos = walk_forward_daily_peak_eval(oos_daily, config)
        except Exception:
            oos = pd.DataFrame()
        if not oos.empty:
            applied_oos = oos[_as_num(oos["Applied"], oos.index).eq(1)].copy()
            summary["out_of_sample_eval_method"] = "expanding_walk_forward"
            summary["out_of_sample_scored_days"] = int(len(oos))
            summary["out_of_sample_applied_days"] = int(len(applied_oos))
            if not applied_oos.empty:
                delta = applied_oos["Shadow_DailyPeak_AbsError_MWH"] - applied_oos["Base_DailyPeak_AbsError_MWH"]
                summary["out_of_sample_baseline_daily_peak_mae_mwh"] = float(applied_oos["Base_DailyPeak_AbsError_MWH"].mean())
                summary["out_of_sample_daily_peak_shadow_mae_mwh"] = float(applied_oos["Shadow_DailyPeak_AbsError_MWH"].mean())
                summary["out_of_sample_delta_daily_peak_mae_mwh"] = float(delta.mean())
                summary["out_of_sample_improved_days"] = int((delta < 0.0).sum())
                summary["out_of_sample_worsened_days"] = int((delta > 0.0).sum())
                summary["out_of_sample_mean_abs_correction_mwh"] = float(applied_oos["Correction_MWH"].abs().mean())

    daily = _daily_peak_eval_rows(backtest_df if backtest_df is not None else pd.DataFrame(), config)
    if daily.empty:
        summary["evaluation_days"] = 0
        return summary

    applied = _as_num(daily["Daily_Peak_Correction_Applied_Flag"], daily.index).eq(1)
    eval_days = daily[applied].copy()
    summary["evaluation_days"] = int(len(eval_days))
    summary["total_days_with_actuals"] = int(len(daily))
    if eval_days.empty:
        return summary

    summary["baseline_daily_peak_mae_mwh"] = float(eval_days["Baseline_DailyPeak_AbsError_MWH"].mean())
    summary["daily_peak_shadow_mae_mwh"] = float(eval_days["Shadow_DailyPeak_AbsError_MWH"].mean())
    summary["delta_daily_peak_mae_mwh"] = float(eval_days["Delta_DailyPeak_AbsError_MWH"].mean())
    summary["baseline_at_actual_peak_mae_mwh"] = float(eval_days["Baseline_AtActualPeak_AbsError_MWH"].mean())
    summary["shadow_at_actual_peak_mae_mwh"] = float(eval_days["Shadow_AtActualPeak_AbsError_MWH"].mean())
    summary["delta_at_actual_peak_mae_mwh"] = float(eval_days["Delta_AtActualPeak_AbsError_MWH"].mean())
    summary["improved_days"] = int((eval_days["Delta_DailyPeak_AbsError_MWH"] < 0.0).sum())
    summary["worsened_days"] = int((eval_days["Delta_DailyPeak_AbsError_MWH"] > 0.0).sum())
    summary["mean_daily_correction_mwh"] = float(eval_days["Mean_Correction_MWH"].mean())
    summary["mean_max_abs_correction_mwh"] = float(eval_days["MaxAbs_Correction_MWH"].mean())

    actual = _as_num(backtest_df.get("Actual_MWH", pd.Series(np.nan, index=backtest_df.index)), backtest_df.index)
    base = _as_num(backtest_df.get("Daily_Peak_Base_Forecast_MWH", pd.Series(np.nan, index=backtest_df.index)), backtest_df.index)
    shadow = _as_num(backtest_df.get("Daily_Peak_Shadow_Adjusted_Forecast_MWH", pd.Series(np.nan, index=backtest_df.index)), backtest_df.index)
    row_applied = _as_num(
        backtest_df.get("Daily_Peak_Correction_Applied_Flag", pd.Series(0, index=backtest_df.index)),
        backtest_df.index,
    ).eq(1)
    row_valid = actual.notna() & base.notna() & shadow.notna() & row_applied
    if row_valid.any():
        base_abs = (actual[row_valid] - base[row_valid]).abs()
        shadow_abs = (actual[row_valid] - shadow[row_valid]).abs()
        summary["applied_hourly_rows"] = int(row_valid.sum())
        summary["baseline_hourly_mae_on_applied_rows_mwh"] = float(base_abs.mean())
        summary["shadow_hourly_mae_on_applied_rows_mwh"] = float(shadow_abs.mean())
        summary["delta_hourly_mae_on_applied_rows_mwh"] = float((shadow_abs - base_abs).mean())
    return summary
