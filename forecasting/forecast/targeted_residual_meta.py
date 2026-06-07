from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingRegressor


def _as_num(value) -> pd.Series:
    return pd.to_numeric(value, errors="coerce")


def _hour(values: pd.DataFrame) -> pd.Series:
    dt = pd.to_datetime(values["DT"], errors="coerce")
    return _as_num(values.get("Hour", dt.dt.hour)).fillna(dt.dt.hour).fillna(0.0)


def _month(values: pd.DataFrame) -> pd.Series:
    dt = pd.to_datetime(values["DT"], errors="coerce")
    return _as_num(values.get("Month", dt.dt.month)).fillna(dt.dt.month).fillna(1.0)


def _cloud_norm(values: pd.DataFrame) -> pd.Series:
    cloud = _as_num(values.get("CloudCover_Norm", pd.Series(np.nan, index=values.index)))
    if cloud.notna().any() and cloud.max(skipna=True) > 1.5:
        cloud = cloud / 100.0
    return cloud.clip(0.0, 1.0)


def _solar_loss(values: pd.DataFrame) -> pd.Series:
    col = (
        "BTM_Solar_Loss_From_ClearSky_MW"
        if "BTM_Solar_Loss_From_ClearSky_MW" in values.columns
        else "Midday_Overcast_Solar_Loss_MW"
    )
    return _as_num(values.get(col, pd.Series(0.0, index=values.index))).fillna(0.0).clip(lower=0.0)


def _cloud_solar_midday_mask(values: pd.DataFrame, min_loss_mw: float) -> pd.Series:
    hour = _hour(values)
    cloud = _cloud_norm(values)
    solar_loss = _solar_loss(values)
    return hour.between(10, 16) & (cloud.ge(0.60) | solar_loss.ge(float(min_loss_mw)))


def _hot_peak_mask(values: pd.DataFrame, config: dict) -> pd.Series:
    hours = [int(hour) for hour in config.get("hot_peak_hours", [16, 17, 18, 19, 20])]
    daily_max = _as_num(values.get("Temperature_DailyMax", pd.Series(np.nan, index=values.index)))
    return _hour(values).astype(int).isin(hours) & daily_max.ge(float(config.get("hot_peak_min_maxtemp_f", 90.0)))


def _feature_frame(values: pd.DataFrame, min_loss_mw: float) -> pd.DataFrame:
    out = pd.DataFrame(index=values.index)
    hour = _hour(values)
    month = _month(values)
    cloud = _cloud_norm(values)
    solar_loss = _solar_loss(values)
    solar_proxy = _as_num(values.get("BTM_Solar_Proxy_MW", pd.Series(0.0, index=values.index))).clip(lower=0.0)

    out["Raw_Forecast_MWH"] = _as_num(values.get("Raw_Forecast_MWH", pd.Series(np.nan, index=values.index)))
    out["Temperature"] = _as_num(values.get("Temperature", pd.Series(np.nan, index=values.index)))
    out["Temperature_DailyMax"] = _as_num(values.get("Temperature_DailyMax", pd.Series(np.nan, index=values.index)))
    out["CloudCover_Norm"] = cloud
    out["Humidity_Norm"] = _as_num(values.get("Humidity_Norm", pd.Series(np.nan, index=values.index)))
    out["WindSpeed_Mph"] = _as_num(values.get("WindSpeed_Mph", pd.Series(np.nan, index=values.index)))
    out["PrecipIn"] = _as_num(values.get("PrecipIn", pd.Series(np.nan, index=values.index))).clip(lower=0.0)
    out["BTM_Solar_Proxy_MW"] = solar_proxy
    out["BTM_Solar_Loss_MW"] = solar_loss
    out["ClearSky_Index"] = _as_num(values.get("ClearSky_Index", pd.Series(np.nan, index=values.index)))
    out["Solar_Loss_x_Cloud"] = solar_loss * cloud.fillna(0.0)
    out["Cloudy_Solar_Midday"] = _cloud_solar_midday_mask(values, min_loss_mw).astype(float)
    out["IsWeekend"] = _as_num(values.get("IsWeekend", pd.Series(0.0, index=values.index))).fillna(0.0)
    out["IsLikelySystemPeakHour"] = _as_num(
        values.get("IsLikelySystemPeakHour", pd.Series(0.0, index=values.index))
    ).fillna(0.0)
    out["Hour_Sin"] = np.sin(2.0 * np.pi * hour / 24.0)
    out["Hour_Cos"] = np.cos(2.0 * np.pi * hour / 24.0)
    out["Month_Sin"] = np.sin(2.0 * np.pi * month / 12.0)
    out["Month_Cos"] = np.cos(2.0 * np.pi * month / 12.0)
    return out


def _new_model(config: dict, min_samples_leaf: int) -> HistGradientBoostingRegressor:
    return HistGradientBoostingRegressor(
        loss="absolute_error",
        learning_rate=float(config.get("learning_rate", 0.05)),
        max_iter=int(config.get("max_iter", 80)),
        max_leaf_nodes=int(config.get("max_leaf_nodes", 8)),
        min_samples_leaf=max(2, int(min_samples_leaf)),
        l2_regularization=float(config.get("l2_regularization", 3.0)),
        random_state=int(config.get("random_state", 42)),
    )


def _training_matrix(values: pd.DataFrame, target: pd.Series, min_loss_mw: float) -> tuple[pd.DataFrame, pd.Series, pd.Series]:
    features = _feature_frame(values, min_loss_mw=min_loss_mw)
    target = _as_num(target)
    valid = features["Raw_Forecast_MWH"].notna() & target.notna()
    x = features.loc[valid].copy()
    y = target.loc[valid].copy()
    fill = x.median(numeric_only=True).fillna(0.0)
    return x.fillna(fill).fillna(0.0), y, fill


def _predict(model, fill_values: pd.Series, values: pd.DataFrame, min_loss_mw: float) -> pd.Series:
    features = _feature_frame(values, min_loss_mw=min_loss_mw)
    valid = features["Raw_Forecast_MWH"].notna()
    out = pd.Series(0.0, index=values.index, dtype=float)
    if model is None or not valid.any():
        return out
    x = features.loc[valid].fillna(fill_values).fillna(0.0)
    out.loc[valid] = np.asarray(model.predict(x), dtype=float)
    return out


def build_targeted_residual_meta_model(raw_backtest_df: pd.DataFrame, config: dict) -> dict | None:
    """Fit a compact residual layer on origin-available raw holdout errors."""
    if raw_backtest_df is None or raw_backtest_df.empty:
        return None

    cfg = ((config or {}).get("calibration", {}) or {}).get("targeted_residual_meta", {}) or {}
    if not bool(cfg.get("enabled", True)):
        return None

    work = raw_backtest_df.copy()
    actual = _as_num(work.get("Actual_MWH", pd.Series(np.nan, index=work.index)))
    raw = _as_num(work.get("Raw_Forecast_MWH", pd.Series(np.nan, index=work.index)))
    raw_residual = _as_num(work.get("Residual_MWH", actual - raw))
    min_loss_mw = float(cfg.get("solar_cloud_min_loss_mw", 1.25))

    x_global, y_global, fill_global = _training_matrix(work, raw_residual, min_loss_mw=min_loss_mw)
    if len(x_global) < int(cfg.get("min_rows", 336)):
        return None

    global_model = _new_model(cfg, min_samples_leaf=int(cfg.get("min_samples_leaf", 24)))
    global_model.fit(x_global, y_global)
    global_correction = _predict(global_model, fill_global, work, min_loss_mw=min_loss_mw)
    global_correction = (
        global_correction
        * float(cfg.get("global_blend", 0.35))
    ).clip(-float(cfg.get("global_cap_mwh", 6.0)), float(cfg.get("global_cap_mwh", 6.0)))

    event_model = None
    fill_event = pd.Series(dtype=float)
    event_mask = _cloud_solar_midday_mask(work, min_loss_mw=min_loss_mw)
    event_target = raw_residual - global_correction
    x_event, y_event, fill_event = _training_matrix(work.loc[event_mask], event_target.loc[event_mask], min_loss_mw=min_loss_mw)
    if len(x_event) >= int(cfg.get("min_event_rows", 48)):
        event_model = _new_model(cfg, min_samples_leaf=int(cfg.get("event_min_samples_leaf", 12)))
        event_model.fit(x_event, y_event)

    return {
        "global_model": global_model,
        "global_fill_values": fill_global,
        "event_model": event_model,
        "event_fill_values": fill_event,
        "metadata": {
            "training_rows": int(len(x_global)),
            "event_training_rows": int(len(x_event)),
            "raw_residual_mean_mwh": float(y_global.mean()),
            "raw_residual_mae_mwh": float(y_global.abs().mean()),
            "solar_cloud_min_loss_mw": min_loss_mw,
        },
    }


def apply_targeted_residual_meta_correction(
    future_df: pd.DataFrame,
    artifact: dict | None,
    config: dict,
) -> pd.DataFrame:
    """Apply capped broad raw-bias and cloudy-solar-midday residual predictions."""
    out = future_df.copy()
    out["Targeted_Meta_Bias_Cal_MWH"] = 0.0
    out["Targeted_Meta_SolarCloud_Cal_MWH"] = 0.0
    out["Targeted_Meta_Cal_MWH"] = 0.0
    out["Targeted_Meta_Source"] = "none"
    raw = _as_num(out.get("Raw_Forecast_MWH", pd.Series(np.nan, index=out.index)))
    out["Targeted_Meta_Adjusted_Forecast_MWH"] = raw.clip(lower=0.0)

    cfg = ((config or {}).get("calibration", {}) or {}).get("targeted_residual_meta", {}) or {}
    if not bool(cfg.get("enabled", True)) or not artifact or artifact.get("global_model") is None:
        return out

    min_loss_mw = float(cfg.get("solar_cloud_min_loss_mw", 1.25))
    global_corr = _predict(
        artifact.get("global_model"),
        artifact.get("global_fill_values", pd.Series(dtype=float)),
        out,
        min_loss_mw=min_loss_mw,
    )
    global_cap = float(cfg.get("global_cap_mwh", 6.0))
    global_corr = (global_corr * float(cfg.get("global_blend", 0.35))).clip(-global_cap, global_cap)

    event_corr = pd.Series(0.0, index=out.index, dtype=float)
    event_mask = _cloud_solar_midday_mask(out, min_loss_mw=min_loss_mw)
    if artifact.get("event_model") is not None and event_mask.any():
        event_pred = _predict(
            artifact.get("event_model"),
            artifact.get("event_fill_values", pd.Series(dtype=float)),
            out.loc[event_mask],
            min_loss_mw=min_loss_mw,
        )
        event_cap = float(cfg.get("event_cap_mwh", 5.0))
        event_corr.loc[event_mask] = (
            event_pred * float(cfg.get("event_blend", 0.45))
        ).clip(-event_cap, event_cap)

    hot_peak = _hot_peak_mask(out, cfg)
    if hot_peak.any():
        hot_peak_scale = float(cfg.get("hot_peak_scale", 0.0))
        global_corr.loc[hot_peak] *= hot_peak_scale
        event_corr.loc[hot_peak] *= hot_peak_scale
    total_cap = float(cfg.get("total_cap_mwh", 8.0))
    total_corr = (global_corr + event_corr).clip(-total_cap, total_cap)
    valid = raw.notna()
    out.loc[valid, "Targeted_Meta_Bias_Cal_MWH"] = global_corr.loc[valid]
    out.loc[valid, "Targeted_Meta_SolarCloud_Cal_MWH"] = event_corr.loc[valid]
    out.loc[valid, "Targeted_Meta_Cal_MWH"] = total_corr.loc[valid]
    out.loc[valid, "Targeted_Meta_Source"] = "raw_bias"
    out.loc[valid & event_mask, "Targeted_Meta_Source"] = "raw_bias+solar_cloud_midday"
    out["Targeted_Meta_Adjusted_Forecast_MWH"] = (raw + total_corr).clip(lower=0.0)
    return out
