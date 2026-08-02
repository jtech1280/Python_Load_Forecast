from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingRegressor


AUTO_RESIDUAL_COLUMNS = [
    "Auto_Residual_Model_Version",
    "Auto_Residual_Shadow_Mode",
    "Auto_Residual_Production_Scope",
    "Auto_Residual_Base_Forecast_MWH",
    "Auto_Residual_Correction_MWH",
    "Auto_Residual_Adjusted_Forecast_MWH",
    "Auto_Residual_Correction_Applied_Flag",
    "Auto_Residual_Source",
    "Auto_Residual_Evaluation_Mode",
    "Auto_Residual_Residual_MWH",
    "Auto_Residual_AbsError_MWH",
    "Auto_Residual_Delta_AbsError_MWH",
    "Auto_Residual_Full_Shadow_Correction_MWH",
    "Auto_Residual_Full_Shadow_Adjusted_Forecast_MWH",
    "Auto_Residual_Full_Shadow_Correction_Applied_Flag",
    "Auto_Residual_Full_Shadow_Source",
    "Auto_Residual_Full_Shadow_Residual_MWH",
    "Auto_Residual_Full_Shadow_AbsError_MWH",
    "Auto_Residual_Full_Shadow_Delta_AbsError_MWH",
]


def _cfg(config: dict | None) -> dict:
    raw = config or {}
    if "operational_residual_learner" in raw:
        return raw.get("operational_residual_learner", {}) or {}
    return ((raw.get("calibration", {}) or {}).get("operational_residual_learner", {}) or {})


def _production_scope(cfg: dict | None) -> str:
    """Return the configured subset allowed to affect the applied auto-residual stage."""
    raw = str((cfg or {}).get("production_scope", "all") or "all").strip().lower()
    aliases = {
        "": "all",
        "full": "all",
        "global_and_hot_peak": "all",
        "hot": "hot_peak_only",
        "gated_hot_peak": "hot_peak_only",
        "gated_hot_peak_only": "hot_peak_only",
        "none": "shadow_only",
        "disabled": "shadow_only",
    }
    return aliases.get(raw, raw)


def _as_num(value: Any, index: pd.Index | None = None, default: float = np.nan) -> pd.Series:
    if isinstance(value, pd.Series):
        raw = value
    else:
        raw = pd.Series(default, index=index)
    return pd.to_numeric(raw, errors="coerce")


def _local_datetime(values: pd.DataFrame) -> pd.Series:
    raw = values.get("DT", pd.Series(pd.NaT, index=values.index))
    try:
        return pd.to_datetime(raw, errors="coerce")
    except ValueError:
        cleaned = raw.astype(str).str.strip().str.replace(r"(?:[+-]\d{2}:?\d{2}|Z)$", "", regex=True)
        return pd.to_datetime(cleaned, errors="coerce")


def _hour(values: pd.DataFrame, dt: pd.Series | None = None) -> pd.Series:
    dt = dt if dt is not None else _local_datetime(values)
    return _as_num(values.get("Hour", dt.dt.hour), values.index).fillna(dt.dt.hour).fillna(0.0)


def _month(values: pd.DataFrame, dt: pd.Series | None = None) -> pd.Series:
    dt = dt if dt is not None else _local_datetime(values)
    return _as_num(values.get("Month", dt.dt.month), values.index).fillna(dt.dt.month).fillna(1.0)


def _dow(values: pd.DataFrame, dt: pd.Series | None = None) -> pd.Series:
    dt = dt if dt is not None else _local_datetime(values)
    return _as_num(values.get("DOW", dt.dt.dayofweek), values.index).fillna(dt.dt.dayofweek).fillna(0.0)


def _cloud_norm(values: pd.DataFrame) -> pd.Series:
    cloud = _as_num(values.get("CloudCover_Norm", pd.Series(np.nan, index=values.index)), values.index)
    if cloud.notna().any() and cloud.max(skipna=True) > 1.5:
        cloud = cloud / 100.0
    return cloud.clip(0.0, 1.0)


def _optional_num(values: pd.DataFrame, *cols: str, default: float = 0.0) -> pd.Series:
    for col in cols:
        if col in values.columns:
            return _as_num(values[col], values.index).fillna(default)
    return pd.Series(default, index=values.index, dtype=float)


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


def _solar_loss(values: pd.DataFrame) -> pd.Series:
    return _optional_num(
        values,
        "BTM_Solar_Loss_From_ClearSky_MW",
        "Midday_Overcast_Solar_Loss_MW",
        default=0.0,
    ).clip(lower=0.0)


def _base_forecast(values: pd.DataFrame, forecast_col: str) -> pd.Series:
    if forecast_col in values.columns:
        return _as_num(values[forecast_col], values.index)
    if "Final_Forecast_MWH" in values.columns:
        return _as_num(values["Final_Forecast_MWH"], values.index)
    if "Final_Backtest_Forecast_MWH" in values.columns:
        return _as_num(values["Final_Backtest_Forecast_MWH"], values.index)
    return _as_num(values.get("Calibrated_Forecast_MWH", values.get("Raw_Forecast_MWH")), values.index)


def _hot_peak_mask(values: pd.DataFrame, config: dict | None) -> pd.Series:
    cfg = _cfg(config)
    hot_cfg = cfg.get("hot_peak", {}) or {}
    dt = _local_datetime(values)
    hour = _hour(values, dt=dt).astype(int)
    hours = [int(h) for h in hot_cfg.get("hours", [16, 17, 18, 19, 20])]
    daily_max = _as_num(values.get("Temperature_DailyMax", pd.Series(np.nan, index=values.index)), values.index)
    min_temp = float(hot_cfg.get("min_maxtemp_f", cfg.get("hot_peak_min_maxtemp_f", 90.0)))
    return hour.isin(hours) & daily_max.ge(min_temp)


def _hot_peak_low_forecast_gate(values: pd.DataFrame, config: dict | None, base: pd.Series) -> pd.Series:
    """Return hot-peak rows where an upward residual lift is operationally plausible.

    The gate uses only forecast-origin-available state. Missing criteria are ignored.
    If no criteria are configured, all hot-peak rows pass.
    """
    cfg = _cfg(config)
    gate_cfg = ((cfg.get("hot_peak", {}) or {}).get("positive_gate", {}) or {})
    if not bool(gate_cfg.get("enabled", False)):
        return pd.Series(True, index=values.index, dtype=bool)

    raw = _optional_num(values, "Raw_Forecast_MWH", default=np.nan)
    same7 = _optional_num(values, "MWH_SameHour7DayMean", "Baseline_Rolling7DaySameHourAvg_MWH", default=np.nan)
    lag24 = _optional_num(values, "MWH_Lag24", "Baseline_SameHourYesterday_MWH", default=np.nan)
    raw_minus_same7 = _optional_num(values, "Raw_Minus_SameHour7DayMean_MWH", default=np.nan)
    raw_minus_lag24 = _optional_num(values, "Raw_Minus_SameHourYesterday_MWH", default=np.nan)
    if raw_minus_same7.isna().all():
        raw_minus_same7 = raw - same7
    if raw_minus_lag24.isna().all():
        raw_minus_lag24 = raw - lag24
    base_minus_raw = base - raw
    raw_minus_base = raw - base
    base_minus_same7 = base - same7
    base_minus_lag24 = base - lag24

    gate = pd.Series(True, index=values.index, dtype=bool)
    configured = False

    def require_min(series: pd.Series, key: str) -> None:
        nonlocal gate, configured
        if key not in gate_cfg:
            return
        configured = True
        gate &= series.ge(float(gate_cfg[key])).fillna(False)

    def require_max(series: pd.Series, key: str) -> None:
        nonlocal gate, configured
        if key not in gate_cfg:
            return
        configured = True
        gate &= series.le(float(gate_cfg[key])).fillna(False)

    require_min(raw_minus_same7, "min_raw_minus_samehour_7day_mean_mwh")
    require_max(raw_minus_same7, "max_raw_minus_samehour_7day_mean_mwh")
    require_min(base_minus_same7, "min_base_minus_samehour_7day_mean_mwh")
    require_max(base_minus_same7, "max_base_minus_samehour_7day_mean_mwh")
    require_min(raw_minus_lag24, "min_raw_minus_samehour_yesterday_mwh")
    require_max(raw_minus_lag24, "max_raw_minus_samehour_yesterday_mwh")
    require_min(raw_minus_base, "min_raw_minus_final_forecast_mwh")
    require_max(raw_minus_base, "max_raw_minus_final_forecast_mwh")
    require_min(base_minus_raw, "min_final_minus_raw_forecast_mwh")
    require_max(base_minus_raw, "max_final_minus_raw_forecast_mwh")
    require_min(base_minus_lag24, "min_base_minus_samehour_yesterday_mwh")
    require_max(base_minus_lag24, "max_base_minus_samehour_yesterday_mwh")
    require_min(_optional_num(values, "Recent_Level_Correction_MWH", default=0.0), "min_recent_correction_mwh")
    require_max(_optional_num(values, "Recent_Level_Correction_MWH", default=0.0), "max_recent_correction_mwh")
    require_min(_optional_num(values, "Peak_Risk_Cal_MWH", default=0.0), "min_peak_risk_correction_mwh")
    require_max(_optional_num(values, "Peak_Risk_Cal_MWH", default=0.0), "max_peak_risk_correction_mwh")
    require_min(_as_num(values.get("Temperature_DailyMax", pd.Series(np.nan, index=values.index)), values.index), "min_maxtemp_f")
    require_min(_cloud_norm(values), "min_cloud_cover_norm")
    require_max(_cloud_norm(values), "max_cloud_cover_norm")
    require_max(_forecast_day(values), "max_forecast_day")
    require_min(_forecast_day(values), "min_forecast_day")

    return gate if configured else pd.Series(True, index=values.index, dtype=bool)


def _hot_peak_cooling_underway_guard(values: pd.DataFrame, config: dict | None) -> pd.Series:
    """Return hot-peak rows where evening cooling argues against an upward lift.

    The guard is intentionally separate from the low-forecast gate: the low gate
    asks whether an upside correction is plausible versus recent load baselines,
    while this guard suppresses that upside when the weather shape indicates the
    post-peak decay regime is already underway.
    """
    cfg = _cfg(config)
    hot_cfg = cfg.get("hot_peak", {}) or {}
    gate_cfg = (hot_cfg.get("positive_gate", {}) or {})
    guard_cfg = (gate_cfg.get("cooling_underway_guard", {}) or gate_cfg.get("cooling_guard", {}) or {})
    if not bool(guard_cfg.get("enabled", False)):
        return pd.Series(False, index=values.index, dtype=bool)

    masks: list[pd.Series] = []

    def add_min(series: pd.Series, key: str) -> None:
        if key not in guard_cfg:
            return
        masks.append(series.ge(float(guard_cfg[key])).fillna(False))

    drop_from_max = _optional_num(values, "Temperature_Drop_From_DailyMax_F", default=np.nan)
    if drop_from_max.isna().all():
        daily_max = _as_num(values.get("Temperature_DailyMax", pd.Series(np.nan, index=values.index)), values.index)
        temp = _as_num(values.get("Temperature", pd.Series(np.nan, index=values.index)), values.index)
        drop_from_max = daily_max - temp
    add_min(drop_from_max, "min_drop_from_dailymax_f")

    add_min(_optional_num(values, "TempDrop_Next1Hr_F", default=np.nan), "min_forecast_drop_next1hr_f")
    add_min(_optional_num(values, "TempDrop_Next2Hr_F", default=np.nan), "min_forecast_drop_next2hr_f")
    add_min(_optional_num(values, "TempDrop_Next3Hr_F", default=np.nan), "min_forecast_drop_next3hr_f")
    add_min(_optional_num(values, "PostPeak_LoadDecay_1Hr_MWH", default=np.nan), "min_post_peak_load_decay_1hr_mwh")
    add_min(_optional_num(values, "PostPeak_LoadDecay_2Hr_MWH", default=np.nan), "min_post_peak_load_decay_2hr_mwh")

    if not masks:
        return pd.Series(False, index=values.index, dtype=bool)

    mode = str(guard_cfg.get("mode", "all") or "all").strip().lower()
    guard = masks[0].copy()
    for mask in masks[1:]:
        guard = guard | mask if mode in {"any", "or"} else guard & mask

    min_hour = guard_cfg.get("min_hour")
    if min_hour is not None:
        guard &= _hour(values).ge(float(min_hour)).fillna(False)
    max_hour = guard_cfg.get("max_hour")
    if max_hour is not None:
        guard &= _hour(values).le(float(max_hour)).fillna(False)
    max_cloud = guard_cfg.get("max_cloud_cover_norm")
    if max_cloud is not None:
        guard &= _cloud_norm(values).le(float(max_cloud)).fillna(False)

    return guard.fillna(False)


def _feature_frame(values: pd.DataFrame, forecast_col: str) -> pd.DataFrame:
    out = pd.DataFrame(index=values.index)
    dt = _local_datetime(values)
    hour = _hour(values, dt=dt)
    month = _month(values, dt=dt)
    dow = _dow(values, dt=dt)
    cloud = _cloud_norm(values)
    daily_max = _as_num(values.get("Temperature_DailyMax", pd.Series(np.nan, index=values.index)), values.index)
    temp = _as_num(values.get("Temperature", pd.Series(np.nan, index=values.index)), values.index)
    base = _base_forecast(values, forecast_col=forecast_col)
    raw = _optional_num(values, "Raw_Forecast_MWH", default=np.nan)
    prophet = _optional_num(values, "Prophet_Pred_MWH", default=np.nan)
    xgb = _optional_num(values, "XGB_Pred_MWH", default=np.nan)
    lgb = _optional_num(values, "LGB_Pred_MWH", default=np.nan)
    cat = _optional_num(values, "CatBoost_Pred_MWH", default=np.nan)
    components = pd.concat([xgb, lgb, cat, prophet], axis=1)
    tree_components = pd.concat([xgb, lgb, cat], axis=1)
    same7 = _optional_num(values, "MWH_SameHour7DayMean", "Baseline_Rolling7DaySameHourAvg_MWH", default=np.nan)
    lag24 = _optional_num(values, "MWH_Lag24", "Baseline_SameHourYesterday_MWH", default=np.nan)
    solar_loss = _solar_loss(values)
    forecast_day = _forecast_day(values, dt=dt)

    out["Base_Forecast_MWH"] = base
    out["Raw_Forecast_MWH"] = raw
    out["Forecast_Day"] = forecast_day
    out["Hour"] = hour
    out["Month"] = month
    out["DOW"] = dow
    out["IsWeekend"] = _optional_num(values, "IsWeekend", default=0.0)
    out["IsHoliday"] = _optional_num(values, "IsHoliday", default=0.0)
    out["IsLikelySystemPeakHour"] = _optional_num(values, "IsLikelySystemPeakHour", default=0.0)
    out["Temperature"] = temp
    out["Temperature_DailyMax"] = daily_max
    out["Temperature_Above_90F"] = (daily_max - 90.0).clip(lower=0.0)
    out["Temperature_Above_100F"] = (daily_max - 100.0).clip(lower=0.0)
    out["CloudCover_Norm"] = cloud
    out["Humidity_Norm"] = _optional_num(values, "Humidity_Norm", default=np.nan)
    out["WindSpeed_Mph"] = _optional_num(values, "WindSpeed_Mph", default=np.nan)
    out["PrecipIn"] = _optional_num(values, "PrecipIn", default=0.0).clip(lower=0.0)
    out["BTM_Solar_Proxy_MW"] = _optional_num(values, "BTM_Solar_Proxy_MW", default=0.0).clip(lower=0.0)
    out["BTM_Solar_Loss_MW"] = solar_loss
    out["ClearSky_Index"] = _optional_num(values, "ClearSky_Index", default=np.nan)
    out["MWH_Lag24"] = lag24
    out["MWH_SameHour7DayMean"] = same7
    out["Raw_Minus_SameHour7DayMean_MWH"] = raw - same7
    out["Raw_Minus_SameHourYesterday_MWH"] = raw - lag24
    out["Base_Minus_SameHour7DayMean_MWH"] = base - same7
    out["Base_Minus_SameHourYesterday_MWH"] = base - lag24
    out["Prophet_Gap_MWH"] = prophet - base
    out["Prophet_Raw_Gap_MWH"] = prophet - raw
    out["CatBoost_Gap_MWH"] = cat - base
    out["Tree_Spread_MWH"] = tree_components.max(axis=1, skipna=True) - tree_components.min(axis=1, skipna=True)
    out["Component_Spread_MWH"] = components.max(axis=1, skipna=True) - components.min(axis=1, skipna=True)
    out["Peak_Risk_Cal_MWH"] = _optional_num(values, "Peak_Risk_Cal_MWH", default=0.0)
    out["Recent_Level_Correction_MWH"] = _optional_num(values, "Recent_Level_Correction_MWH", default=0.0)
    out["Focused_Scorecard_Guard_MWH"] = _optional_num(values, "Focused_Scorecard_Guard_MWH", default=0.0)
    out["Weather_Robustness_Hedge_MWH"] = _optional_num(values, "Weather_Robustness_Hedge_MWH", default=0.0)
    out["WeatherScenario_Spread_MWH"] = _optional_num(values, "WeatherScenario_Spread_MWH", default=0.0)
    out["WeatherScenario_MaxAbsDelta_MWH"] = _optional_num(values, "WeatherScenario_MaxAbsDelta_MWH", default=0.0)
    out["Weather_Input_Risk_Multiplier"] = _optional_num(values, "Weather_Input_Risk_Multiplier", default=1.0)
    out["Hour_Sin"] = np.sin(2.0 * np.pi * hour / 24.0)
    out["Hour_Cos"] = np.cos(2.0 * np.pi * hour / 24.0)
    out["Month_Sin"] = np.sin(2.0 * np.pi * month / 12.0)
    out["Month_Cos"] = np.cos(2.0 * np.pi * month / 12.0)
    out["ForecastDay_Log1p"] = np.log1p(forecast_day.clip(lower=0.0))
    out["Hot_Peak_Flag"] = _hot_peak_mask(values, {"operational_residual_learner": {"hot_peak": {}}}).astype(float)
    out["Prophet_Gap_x_HotPeak"] = out["Prophet_Gap_MWH"] * out["Hot_Peak_Flag"]
    out["TempAbove90_x_HotPeak"] = out["Temperature_Above_90F"] * out["Hot_Peak_Flag"]
    return out


def _new_model(cfg: dict, min_samples_leaf: int) -> HistGradientBoostingRegressor:
    return HistGradientBoostingRegressor(
        loss=str(cfg.get("loss", "absolute_error")),
        learning_rate=float(cfg.get("learning_rate", 0.05)),
        max_iter=int(cfg.get("max_iter", 80)),
        max_leaf_nodes=int(cfg.get("max_leaf_nodes", 12)),
        min_samples_leaf=max(2, int(min_samples_leaf)),
        l2_regularization=float(cfg.get("l2_regularization", 4.0)),
        random_state=int(cfg.get("random_state", 42)),
    )


def _training_matrix(
    values: pd.DataFrame,
    target: pd.Series,
    forecast_col: str,
) -> tuple[pd.DataFrame, pd.Series, pd.Series]:
    features = _feature_frame(values, forecast_col=forecast_col)
    target = _as_num(target, values.index)
    valid = features["Base_Forecast_MWH"].notna() & target.notna()
    x = features.loc[valid].copy()
    y = target.loc[valid].copy()
    fill = x.median(numeric_only=True).replace([np.inf, -np.inf], np.nan).fillna(0.0)
    return x.fillna(fill).replace([np.inf, -np.inf], 0.0).fillna(0.0), y, fill


def _predict_model(model, fill_values: pd.Series, columns: list[str], values: pd.DataFrame, forecast_col: str) -> pd.Series:
    features = _feature_frame(values, forecast_col=forecast_col).reindex(columns=columns)
    valid = features["Base_Forecast_MWH"].notna()
    out = pd.Series(0.0, index=values.index, dtype=float)
    if model is None or not valid.any():
        return out
    x = features.loc[valid].fillna(fill_values).replace([np.inf, -np.inf], 0.0).fillna(0.0)
    out.loc[valid] = np.asarray(model.predict(x), dtype=float)
    return out


def _target_residual(values: pd.DataFrame, forecast_col: str) -> pd.Series:
    if "Final_Residual_MWH" in values.columns and forecast_col in {"Final_Backtest_Forecast_MWH", "Final_Forecast_MWH"}:
        return _as_num(values["Final_Residual_MWH"], values.index)
    actual_col = "Actual_MWH" if "Actual_MWH" in values.columns else "Actual"
    actual = _as_num(values.get(actual_col, pd.Series(np.nan, index=values.index)), values.index)
    base = _base_forecast(values, forecast_col=forecast_col)
    return actual - base


def build_operational_residual_learner(
    backtest_df: pd.DataFrame,
    config: dict | None,
    *,
    forecast_col: str = "Final_Backtest_Forecast_MWH",
) -> dict | None:
    """Fit a bounded second-pass learner for residuals left after the production correction chain."""
    cfg = _cfg(config)
    if backtest_df is None or backtest_df.empty or not bool(cfg.get("enabled", False)):
        return None

    work = backtest_df.copy()
    target = _target_residual(work, forecast_col=forecast_col)
    target_clip = float(cfg.get("target_clip_mwh", 30.0))
    if target_clip > 0:
        target = target.clip(-target_clip, target_clip)

    x_global, y_global, fill_global = _training_matrix(work, target, forecast_col=forecast_col)
    min_rows = int(cfg.get("min_rows", 336))
    if len(x_global) < min_rows:
        return None

    global_model = _new_model(cfg, min_samples_leaf=int(cfg.get("min_samples_leaf", 24)))
    global_model.fit(x_global, y_global)
    feature_columns = list(x_global.columns)

    global_train_pred = pd.Series(
        np.asarray(global_model.predict(x_global), dtype=float),
        index=x_global.index,
    )
    global_train_corr = (
        global_train_pred * float(cfg.get("blend", 0.35))
    ).clip(-float(cfg.get("cap_mwh", 6.0)), float(cfg.get("cap_mwh", 6.0)))

    hot_cfg = cfg.get("hot_peak", {}) or {}
    hot_model = None
    fill_hot = pd.Series(dtype=float)
    hot_training_rows = 0
    if bool(hot_cfg.get("enabled", True)):
        hot_mask = _hot_peak_mask(work, config).reindex(x_global.index).fillna(False)
        hot_target = y_global - global_train_corr
        hot_training_rows = int(hot_mask.sum())
        if hot_training_rows >= int(hot_cfg.get("min_rows", 48)):
            x_hot = x_global.loc[hot_mask].copy()
            y_hot = hot_target.loc[hot_mask].copy()
            fill_hot = x_hot.median(numeric_only=True).replace([np.inf, -np.inf], np.nan).fillna(0.0)
            hot_model = _new_model(cfg | hot_cfg, min_samples_leaf=int(hot_cfg.get("min_samples_leaf", 12)))
            hot_model.fit(x_hot.fillna(fill_hot).replace([np.inf, -np.inf], 0.0).fillna(0.0), y_hot)

    return {
        "global_model": global_model,
        "global_fill_values": fill_global,
        "hot_peak_model": hot_model,
        "hot_peak_fill_values": fill_hot,
        "feature_columns": feature_columns,
        "metadata": {
            "model_version": str(cfg.get("model_version", "operational_residual_shadow_v1")),
            "training_rows": int(len(x_global)),
            "hot_peak_training_rows": int(hot_training_rows),
            "target_residual_mean_mwh": float(y_global.mean()),
            "target_residual_mae_mwh": float(y_global.abs().mean()),
            "forecast_col": forecast_col,
            "shadow_mode": bool(cfg.get("shadow_mode", True)),
            "production_scope": _production_scope(cfg),
        },
    }


def _init_auto_columns(
    out: pd.DataFrame,
    base: pd.Series,
    *,
    model_version: str,
    shadow_mode: bool,
    production_scope: str,
    source: str,
    evaluation_mode: str,
) -> pd.DataFrame:
    out["Auto_Residual_Model_Version"] = model_version
    out["Auto_Residual_Shadow_Mode"] = int(bool(shadow_mode))
    out["Auto_Residual_Production_Scope"] = production_scope
    out["Auto_Residual_Base_Forecast_MWH"] = base
    out["Auto_Residual_Correction_MWH"] = 0.0
    out["Auto_Residual_Adjusted_Forecast_MWH"] = base
    out["Auto_Residual_Correction_Applied_Flag"] = 0
    out["Auto_Residual_Source"] = source
    out["Auto_Residual_Evaluation_Mode"] = evaluation_mode
    actual_col = "Actual_MWH" if "Actual_MWH" in out.columns else "Actual"
    actual = _as_num(out.get(actual_col, pd.Series(np.nan, index=out.index)), out.index)
    auto_residual = actual - out["Auto_Residual_Adjusted_Forecast_MWH"]
    base_residual = actual - base
    out["Auto_Residual_Residual_MWH"] = auto_residual
    out["Auto_Residual_AbsError_MWH"] = auto_residual.abs()
    out["Auto_Residual_Delta_AbsError_MWH"] = auto_residual.abs() - base_residual.abs()
    out["Auto_Residual_Full_Shadow_Correction_MWH"] = 0.0
    out["Auto_Residual_Full_Shadow_Adjusted_Forecast_MWH"] = base
    out["Auto_Residual_Full_Shadow_Correction_Applied_Flag"] = 0
    out["Auto_Residual_Full_Shadow_Source"] = source
    out["Auto_Residual_Full_Shadow_Residual_MWH"] = auto_residual
    out["Auto_Residual_Full_Shadow_AbsError_MWH"] = auto_residual.abs()
    out["Auto_Residual_Full_Shadow_Delta_AbsError_MWH"] = auto_residual.abs() - base_residual.abs()
    return out


def apply_operational_residual_learner(
    df: pd.DataFrame,
    artifact: dict | None,
    config: dict | None,
    *,
    forecast_col: str = "Final_Forecast_MWH",
    also_update_cols: tuple[str, ...] = (),
    force_shadow: bool | None = None,
    evaluation_mode: str = "artifact_shadow",
) -> pd.DataFrame:
    """Apply the residual learner and emit shadow diagnostics by default.

    Residual convention is Actual - Forecast. Positive model output increases the forecast.
    With ``shadow_mode: true`` the production forecast column is not changed.
    """
    out = df.copy()
    cfg = _cfg(config)
    shadow_mode = bool(cfg.get("shadow_mode", True)) if force_shadow is None else bool(force_shadow)
    production_scope = _production_scope(cfg)
    model_version = str(cfg.get("model_version", "operational_residual_shadow_v1"))
    base = _base_forecast(out, forecast_col=forecast_col)
    out = _init_auto_columns(
        out,
        base,
        model_version=model_version,
        shadow_mode=shadow_mode,
        production_scope=production_scope,
        source="disabled",
        evaluation_mode=evaluation_mode,
    )
    if out.empty or not bool(cfg.get("enabled", False)):
        return out
    if not artifact or artifact.get("global_model") is None:
        out["Auto_Residual_Source"] = "insufficient_history"
        out["Auto_Residual_Full_Shadow_Source"] = "insufficient_history"
        return out

    columns = list(artifact.get("feature_columns") or [])
    if not columns:
        out["Auto_Residual_Source"] = "no_feature_columns"
        out["Auto_Residual_Full_Shadow_Source"] = "no_feature_columns"
        return out

    global_cap = float(cfg.get("cap_mwh", 6.0))
    global_corr = _predict_model(
        artifact.get("global_model"),
        artifact.get("global_fill_values", pd.Series(dtype=float)),
        columns,
        out,
        forecast_col=forecast_col,
    )
    global_corr = (global_corr * float(cfg.get("blend", 0.35))).clip(-global_cap, global_cap)

    hot_corr = pd.Series(0.0, index=out.index, dtype=float)
    source = pd.Series("global", index=out.index, dtype="object")
    hot_cfg = cfg.get("hot_peak", {}) or {}
    hot_mask = _hot_peak_mask(out, config)
    if artifact.get("hot_peak_model") is not None and bool(hot_cfg.get("enabled", True)) and hot_mask.any():
        hot_cap = float(hot_cfg.get("cap_mwh", 6.0))
        hot_pred = _predict_model(
            artifact.get("hot_peak_model"),
            artifact.get("hot_peak_fill_values", pd.Series(dtype=float)),
            columns,
            out.loc[hot_mask],
            forecast_col=forecast_col,
        )
        hot_corr.loc[hot_mask] = (hot_pred * float(hot_cfg.get("blend", 0.45))).clip(-hot_cap, hot_cap)
        source.loc[hot_mask] = "global+hot_peak"

    total_cap = float(cfg.get("total_cap_mwh", 8.0))
    total_corr = (global_corr + hot_corr).clip(-total_cap, total_cap)
    if hot_mask.any():
        hot_gate_cfg = (hot_cfg.get("positive_gate", {}) or {})
        if bool(hot_gate_cfg.get("enabled", False)):
            low_forecast_gate = _hot_peak_low_forecast_gate(out, config, base)
            block_positive = hot_mask & total_corr.gt(0.0) & ~low_forecast_gate
            allow_negative = bool(hot_gate_cfg.get("allow_negative_correction", False))
            block_negative = (
                pd.Series(False, index=out.index, dtype=bool)
                if allow_negative
                else hot_mask & total_corr.lt(0.0)
            )
            blocked = block_positive | block_negative
            if blocked.any():
                total_corr.loc[blocked] = 0.0
                source.loc[blocked] = str(hot_gate_cfg.get("blocked_source", "hot_peak_low_forecast_gate_blocked"))

            cooling_guard_cfg = (
                hot_gate_cfg.get("cooling_underway_guard", {})
                or hot_gate_cfg.get("cooling_guard", {})
                or {}
            )
            if bool(cooling_guard_cfg.get("enabled", False)):
                cooling_underway = _hot_peak_cooling_underway_guard(out, config)
                guard_positive = hot_mask & total_corr.gt(0.0) & cooling_underway
                if guard_positive.any():
                    cap = float(cooling_guard_cfg.get("cap_positive_correction_mwh", 0.0))
                    if cap <= 0.0:
                        total_corr.loc[guard_positive] = 0.0
                        source.loc[guard_positive] = str(
                            cooling_guard_cfg.get(
                                "blocked_source",
                                "hot_peak_cooling_underway_guard_blocked",
                            )
                        )
                    else:
                        over_cap = guard_positive & total_corr.gt(cap)
                        if over_cap.any():
                            total_corr.loc[over_cap] = cap
                            source.loc[over_cap] = str(
                                cooling_guard_cfg.get(
                                    "capped_source",
                                    "global+hot_peak_cooling_underway_guard_capped",
                                )
                            )

    valid = base.notna()
    full_corr = total_corr.where(valid, 0.0)
    full_adjusted = (base + full_corr).clip(lower=0.0)
    full_source = pd.Series(np.where(valid, source, "invalid_base_forecast"), index=out.index, dtype="object")

    applied_corr = full_corr.copy()
    applied_source = full_source.copy()
    if production_scope == "hot_peak_only":
        allow_applied = full_source.astype(str).str.startswith("global+hot_peak")
        scoped_out = valid & full_corr.ne(0.0) & ~allow_applied
        applied_corr = full_corr.where(allow_applied, 0.0)
        applied_source.loc[scoped_out & full_source.eq("global")] = "global_shadow_only"
        applied_source.loc[scoped_out & ~full_source.eq("global")] = (
            full_source.loc[scoped_out & ~full_source.eq("global")].astype(str) + "_shadow_only"
        )
    elif production_scope == "shadow_only":
        scoped_out = valid & full_corr.ne(0.0)
        applied_corr = pd.Series(0.0, index=out.index, dtype=float)
        applied_source.loc[scoped_out] = full_source.loc[scoped_out].astype(str) + "_shadow_only"
    elif production_scope != "all":
        # Unknown scopes fail closed for the applied stage but still preserve full-shadow diagnostics.
        scoped_out = valid & full_corr.ne(0.0)
        applied_corr = pd.Series(0.0, index=out.index, dtype=float)
        applied_source.loc[scoped_out] = "unknown_production_scope_shadow_only"

    adjusted = (base + applied_corr).clip(lower=0.0)

    out["Auto_Residual_Correction_MWH"] = applied_corr
    out["Auto_Residual_Adjusted_Forecast_MWH"] = adjusted
    out["Auto_Residual_Correction_Applied_Flag"] = applied_corr.ne(0.0).astype(int)
    out["Auto_Residual_Source"] = applied_source
    out["Auto_Residual_Full_Shadow_Correction_MWH"] = full_corr
    out["Auto_Residual_Full_Shadow_Adjusted_Forecast_MWH"] = full_adjusted
    out["Auto_Residual_Full_Shadow_Correction_Applied_Flag"] = full_corr.ne(0.0).astype(int)
    out["Auto_Residual_Full_Shadow_Source"] = full_source

    actual_col = "Actual_MWH" if "Actual_MWH" in out.columns else "Actual"
    actual = _as_num(out.get(actual_col, pd.Series(np.nan, index=out.index)), out.index)
    auto_residual = actual - adjusted
    full_shadow_residual = actual - full_adjusted
    base_residual = actual - base
    out["Auto_Residual_Residual_MWH"] = auto_residual
    out["Auto_Residual_AbsError_MWH"] = auto_residual.abs()
    out["Auto_Residual_Delta_AbsError_MWH"] = auto_residual.abs() - base_residual.abs()
    out["Auto_Residual_Full_Shadow_Residual_MWH"] = full_shadow_residual
    out["Auto_Residual_Full_Shadow_AbsError_MWH"] = full_shadow_residual.abs()
    out["Auto_Residual_Full_Shadow_Delta_AbsError_MWH"] = full_shadow_residual.abs() - base_residual.abs()

    if not shadow_mode and forecast_col in out.columns:
        out[forecast_col] = adjusted
        if forecast_col == "Final_Backtest_Forecast_MWH" and "Final_Forecast_MWH" in out.columns:
            out["Final_Forecast_MWH"] = adjusted
        for col in also_update_cols:
            if col in out.columns and col != forecast_col:
                out[col] = adjusted
        actual_col = "Actual_MWH" if "Actual_MWH" in out.columns else "Actual"
        actual = _as_num(out.get(actual_col, pd.Series(np.nan, index=out.index)), out.index)
        if forecast_col in {"Final_Backtest_Forecast_MWH", "Final_Forecast_MWH"} and actual.notna().any():
            out["Final_Residual_MWH"] = actual - _as_num(out[forecast_col], out.index)
            out["Final_AbsError_MWH"] = out["Final_Residual_MWH"].abs()
            out["Final_APE"] = np.where(
                actual.abs() > 1e-9,
                out["Final_AbsError_MWH"] / actual.abs() * 100.0,
                np.nan,
            )
    return out


def simulate_operational_residual_learner_backtest(
    backtest_df: pd.DataFrame,
    config: dict | None,
    *,
    forecast_col: str = "Final_Backtest_Forecast_MWH",
    force_shadow: bool | None = True,
) -> pd.DataFrame:
    """Walk-forward evaluation using only earlier corrected backtest rows for each day."""
    out = backtest_df.copy().sort_values("DT").reset_index(drop=True)
    cfg = _cfg(config)
    base = _base_forecast(out, forecast_col=forecast_col)
    shadow_mode = bool(cfg.get("shadow_mode", True)) if force_shadow is None else bool(force_shadow)
    production_scope = _production_scope(cfg)
    model_version = str(cfg.get("model_version", "operational_residual_shadow_v1"))
    out = _init_auto_columns(
        out,
        base,
        model_version=model_version,
        shadow_mode=shadow_mode,
        production_scope=production_scope,
        source="disabled",
        evaluation_mode="walk_forward_shadow",
    )
    if out.empty or not bool(cfg.get("enabled", False)):
        return out

    dt = _local_datetime(out)
    if dt.notna().sum() == 0:
        out["Auto_Residual_Source"] = "no_datetime"
        out["Auto_Residual_Full_Shadow_Source"] = "no_datetime"
        return out

    min_train_rows = int(cfg.get("backtest_min_train_rows", cfg.get("min_rows", 336)))
    pieces: list[pd.DataFrame] = []
    for day in sorted(dt.dropna().dt.normalize().unique()):
        day_mask = dt.dt.normalize().eq(day)
        prior = out.loc[dt.lt(day)].copy()
        if len(prior) < min_train_rows:
            group = out.loc[day_mask].copy()
            group["Auto_Residual_Source"] = "insufficient_walk_forward_history"
            group["Auto_Residual_Full_Shadow_Source"] = "insufficient_walk_forward_history"
            pieces.append(group)
            continue
        artifact = build_operational_residual_learner(prior, config, forecast_col=forecast_col)
        group = apply_operational_residual_learner(
            out.loc[day_mask],
            artifact,
            config,
            forecast_col=forecast_col,
            force_shadow=force_shadow,
            evaluation_mode="walk_forward_shadow",
        )
        pieces.append(group)

    if not pieces:
        return out
    return pd.concat(pieces, ignore_index=True, sort=False).sort_values("DT").reset_index(drop=True)


def operational_residual_learner_summary(backtest_df: pd.DataFrame | None, artifact: dict | None, config: dict | None) -> dict:
    cfg = _cfg(config)
    summary: dict[str, Any] = {
        "enabled": bool(cfg.get("enabled", False)),
        "shadow_mode": bool(cfg.get("shadow_mode", True)),
        "production_scope": _production_scope(cfg),
        "model_version": str(cfg.get("model_version", "operational_residual_shadow_v1")),
    }
    if artifact and artifact.get("metadata"):
        summary["artifact"] = dict(artifact.get("metadata") or {})
    if backtest_df is None or backtest_df.empty or "Auto_Residual_Adjusted_Forecast_MWH" not in backtest_df.columns:
        summary["evaluation_rows"] = 0
        return summary

    actual = _as_num(backtest_df.get("Actual_MWH", pd.Series(np.nan, index=backtest_df.index)), backtest_df.index)
    base = _as_num(
        backtest_df.get(
            "Auto_Residual_Base_Forecast_MWH",
            backtest_df.get(
                "Final_Backtest_Forecast_MWH",
                backtest_df.get("Final_Forecast_MWH", pd.Series(np.nan, index=backtest_df.index)),
            ),
        ),
        backtest_df.index,
    )
    auto = _as_num(backtest_df["Auto_Residual_Adjusted_Forecast_MWH"], backtest_df.index)
    applied = _as_num(backtest_df.get("Auto_Residual_Correction_Applied_Flag", pd.Series(0, index=backtest_df.index)), backtest_df.index).eq(1)
    valid = actual.notna() & base.notna() & auto.notna() & applied
    summary["evaluation_rows"] = int(valid.sum())
    if not valid.any():
        return summary

    base_abs = (actual[valid] - base[valid]).abs()
    auto_abs = (actual[valid] - auto[valid]).abs()
    delta = auto_abs - base_abs
    summary["baseline_mae_mwh"] = float(base_abs.mean())
    summary["auto_shadow_mae_mwh"] = float(auto_abs.mean())
    summary["delta_mae_mwh"] = float(delta.mean())
    summary["improved_rows"] = int((delta < 0).sum())
    summary["worsened_rows"] = int((delta > 0).sum())
    summary["mean_correction_mwh"] = float(
        _as_num(backtest_df.loc[valid, "Auto_Residual_Correction_MWH"], backtest_df.loc[valid].index).mean()
    )

    hot = _hot_peak_mask(backtest_df, config).reindex(backtest_df.index).fillna(False) & valid
    summary["hot_peak_evaluation_rows"] = int(hot.sum())
    if hot.any():
        hot_base_abs = (actual[hot] - base[hot]).abs()
        hot_auto_abs = (actual[hot] - auto[hot]).abs()
        summary["hot_peak_baseline_mae_mwh"] = float(hot_base_abs.mean())
        summary["hot_peak_auto_shadow_mae_mwh"] = float(hot_auto_abs.mean())
        summary["hot_peak_delta_mae_mwh"] = float((hot_auto_abs - hot_base_abs).mean())

    if "Auto_Residual_Full_Shadow_Adjusted_Forecast_MWH" in backtest_df.columns:
        full_shadow = _as_num(backtest_df["Auto_Residual_Full_Shadow_Adjusted_Forecast_MWH"], backtest_df.index)
        full_applied = _as_num(
            backtest_df.get("Auto_Residual_Full_Shadow_Correction_Applied_Flag", pd.Series(0, index=backtest_df.index)),
            backtest_df.index,
        ).eq(1)
        full_valid = actual.notna() & base.notna() & full_shadow.notna() & full_applied
        summary["full_shadow_evaluation_rows"] = int(full_valid.sum())
        if full_valid.any():
            full_base_abs = (actual[full_valid] - base[full_valid]).abs()
            full_auto_abs = (actual[full_valid] - full_shadow[full_valid]).abs()
            summary["full_shadow_baseline_mae_mwh"] = float(full_base_abs.mean())
            summary["full_shadow_auto_mae_mwh"] = float(full_auto_abs.mean())
            summary["full_shadow_delta_mae_mwh"] = float((full_auto_abs - full_base_abs).mean())
    return summary
