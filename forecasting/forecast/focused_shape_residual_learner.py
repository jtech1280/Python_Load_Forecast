from __future__ import annotations

"""Shadow learner intended to replace hand-tuned focused-scorecard YAML rules.

The learner is deliberately diagnostic by default. It learns residuals against the
pre-focused-guard forecast and writes a separate shadow adjusted forecast, leaving
the production forecast unchanged.
"""

from typing import Any

import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingRegressor

from forecasting.forecast.focused_scorecard_guard import focused_scorecard_rule_union_mask


FOCUSED_SHAPE_COLUMNS = [
    "Focused_Shape_Model_Version",
    "Focused_Shape_Shadow_Mode",
    "Focused_Shape_Base_Forecast_MWH",
    "Focused_Shape_Correction_MWH",
    "Focused_Shape_Adjusted_Forecast_MWH",
    "Focused_Shape_Correction_Applied_Flag",
    "Focused_Shape_Source",
    "Focused_Shape_Evaluation_Mode",
    "Focused_Shape_Residual_MWH",
    "Focused_Shape_AbsError_MWH",
    "Focused_Shape_Delta_AbsError_MWH",
    "Focused_Shape_RuleUnion_Flag",
    "Focused_Shape_Scope_Flag",
]


def _cfg(config: dict | None) -> dict:
    raw = config or {}
    if "focused_shape_residual_learner" in raw:
        return raw.get("focused_shape_residual_learner", {}) or {}
    stage_selector = ((raw.get("calibration", {}) or {}).get("stage_selector", {}) or {})
    return stage_selector.get("focused_shape_residual_learner", {}) or {}


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


def _optional_num(values: pd.DataFrame, *cols: str, default: float = np.nan) -> pd.Series:
    for col in cols:
        if col in values.columns:
            return _as_num(values[col], values.index).fillna(default)
    return pd.Series(default, index=values.index, dtype=float)


def _hour(values: pd.DataFrame, dt: pd.Series | None = None) -> pd.Series:
    dt = dt if dt is not None else _local_datetime(values)
    return _as_num(values.get("Hour", dt.dt.hour), values.index).fillna(dt.dt.hour).fillna(0.0)


def _month(values: pd.DataFrame, dt: pd.Series | None = None) -> pd.Series:
    dt = dt if dt is not None else _local_datetime(values)
    return _as_num(values.get("Month", dt.dt.month), values.index).fillna(dt.dt.month).fillna(1.0)


def _dow(values: pd.DataFrame, dt: pd.Series | None = None) -> pd.Series:
    dt = dt if dt is not None else _local_datetime(values)
    return _as_num(values.get("DOW", dt.dt.dayofweek), values.index).fillna(dt.dt.dayofweek).fillna(0.0)


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
        "Pre_Focused_Guard_Forecast_MWH",
        "Stage_Selected_Forecast_MWH",
        "Final_Backtest_Forecast_MWH",
        "Final_Forecast_MWH",
        "Recent_Corrected_Forecast_MWH",
        "Calibrated_Forecast_MWH",
        "Raw_Forecast_MWH",
    ]:
        if col in values.columns:
            return _as_num(values[col], values.index)
    return pd.Series(np.nan, index=values.index, dtype=float)


def _promotion_reference_forecast(values: pd.DataFrame, forecast_col: str, guard_cfg: dict) -> pd.Series:
    ref_col = str(guard_cfg.get("reference_col", "current_final")).strip()
    aliases = {
        "current_final": [
            "Final_Backtest_Forecast_MWH",
            "Final_Forecast_MWH",
            "Stage_Selected_Forecast_MWH",
            "Calibrated_Forecast_MWH",
            forecast_col,
        ],
        "final": ["Final_Backtest_Forecast_MWH", "Final_Forecast_MWH", forecast_col],
        "stage_selected": ["Stage_Selected_Forecast_MWH", forecast_col],
        "base": [forecast_col],
    }
    candidates = aliases.get(ref_col, [ref_col])
    for col in candidates:
        if col in values.columns:
            ref = _as_num(values[col], values.index)
            if ref.notna().any():
                return ref
    return _base_forecast(values, forecast_col=forecast_col)


def _promotion_risk_slice_mask(values: pd.DataFrame, guard_cfg: dict) -> pd.Series:
    if not bool(guard_cfg.get("enabled", False)):
        return pd.Series(False, index=values.index, dtype=bool)
    dt = _local_datetime(values)
    hour = _hour(values, dt=dt)
    daily_max = _optional_num(values, "Temperature_DailyMax", default=np.nan)

    mask = pd.Series(False, index=values.index, dtype=bool)
    if bool(guard_cfg.get("include_peak_window", True)):
        peak_hours = [int(h) for h in guard_cfg.get("peak_window_hours", [14, 15, 16, 17, 18])]
        mask |= hour.astype("Int64").isin(peak_hours).fillna(False)
    if bool(guard_cfg.get("include_hot_peak", True)):
        hot_hours = [int(h) for h in guard_cfg.get("hot_peak_hours", [16, 17, 18, 19, 20])]
        hot_peak = (
            hour.astype("Int64").isin(hot_hours).fillna(False)
            & daily_max.ge(float(guard_cfg.get("hot_peak_min_maxtemp_f", 90.0))).fillna(False)
        )
        mask |= hot_peak
    return mask.fillna(False)


def _solar_loss(values: pd.DataFrame) -> pd.Series:
    return _optional_num(
        values,
        "BTM_Solar_Loss_From_ClearSky_MW",
        "Midday_Overcast_Solar_Loss_MW",
        default=0.0,
    ).clip(lower=0.0)


def _shape_scope(values: pd.DataFrame, config: dict | None, forecast_col: str) -> tuple[pd.Series, pd.Series]:
    cfg = _cfg(config)
    scope_cfg = cfg.get("scope", {}) or {}
    dt = _local_datetime(values)
    hour = _hour(values, dt=dt)
    forecast_day = _forecast_day(values, dt=dt)
    daily_max = _optional_num(values, "Temperature_DailyMax", default=np.nan)
    cloud = _cloud_norm(values)
    solar_loss = _solar_loss(values)

    rule_union = pd.Series(False, index=values.index, dtype=bool)
    if bool(scope_cfg.get("use_focused_guard_rule_union", True)):
        rule_forecast_col = forecast_col if forecast_col in values.columns else (
            "Pre_Focused_Guard_Forecast_MWH"
            if "Pre_Focused_Guard_Forecast_MWH" in values.columns
            else "Final_Backtest_Forecast_MWH"
        )
        rule_union = focused_scorecard_rule_union_mask(
            values,
            config,
            forecast_col=rule_forecast_col,
            include_disabled=bool(scope_cfg.get("include_disabled_rules", False)),
            runtime_context=bool(scope_cfg.get("use_runtime_rule_context", True)),
        )

    hot_hours = [int(h) for h in scope_cfg.get("hot_peak_hours", [14, 15, 16, 17, 18, 19, 20, 21])]
    hot_peak = (
        hour.astype("Int64").isin(hot_hours).fillna(False)
        & daily_max.ge(float(scope_cfg.get("hot_peak_min_maxtemp_f", 90.0))).fillna(False)
    ) if bool(scope_cfg.get("include_hot_peak", True)) else pd.Series(False, index=values.index, dtype=bool)

    cloud_hours = [int(h) for h in scope_cfg.get("cloud_solar_hours", [10, 11, 12, 13, 14, 15, 16])]
    cloud_solar = (
        hour.astype("Int64").isin(cloud_hours).fillna(False)
        & (
            cloud.ge(float(scope_cfg.get("cloud_solar_min_cloud_cover_norm", 0.60))).fillna(False)
            | solar_loss.ge(float(scope_cfg.get("cloud_solar_min_solar_loss_mw", 1.25))).fillna(False)
        )
    ) if bool(scope_cfg.get("include_cloud_solar", True)) else pd.Series(False, index=values.index, dtype=bool)

    delta_hours = [int(h) for h in scope_cfg.get("delta_breeze_hours", [16, 17, 18, 19, 20, 21, 22])]
    delta_flag = _optional_num(values, "DeltaBreeze_Cooling_Flag", default=0.0).gt(0.0)
    westerly_flag = _optional_num(values, "DeltaBreeze_Westerly_Flow_Flag", "Westerly_Flow_Flag", default=0.0).gt(0.0)
    next3_drop = _optional_num(values, "TempDrop_Next3Hr_F", default=np.nan)
    delta_breeze = (
        hour.astype("Int64").isin(delta_hours).fillna(False)
        & daily_max.ge(float(scope_cfg.get("delta_breeze_min_maxtemp_f", 95.0))).fillna(False)
        & cloud.le(float(scope_cfg.get("delta_breeze_max_cloud_cover_norm", 0.35))).fillna(False)
        & (delta_flag | westerly_flag | next3_drop.ge(float(scope_cfg.get("delta_breeze_min_forecast_drop_next3hr_f", 6.0))).fillna(False))
    ) if bool(scope_cfg.get("include_delta_breeze", True)) else pd.Series(False, index=values.index, dtype=bool)

    long_horizon = (
        forecast_day.ge(float(scope_cfg.get("long_horizon_min_forecast_day", 8))).fillna(False)
        & forecast_day.le(float(scope_cfg.get("long_horizon_max_forecast_day", 16))).fillna(False)
        & daily_max.ge(float(scope_cfg.get("long_horizon_min_maxtemp_f", 75.0))).fillna(False)
        & hour.between(
            float(scope_cfg.get("long_horizon_min_hour", 10)),
            float(scope_cfg.get("long_horizon_max_hour", 22)),
        ).fillna(False)
    ) if bool(scope_cfg.get("include_long_horizon_heat", True)) else pd.Series(False, index=values.index, dtype=bool)

    scope = rule_union | hot_peak | cloud_solar | delta_breeze | long_horizon
    if not bool(scope_cfg.get("require_scope_for_application", True)):
        scope = pd.Series(True, index=values.index, dtype=bool)
    return rule_union.fillna(False), scope.fillna(False)


def _feature_frame(values: pd.DataFrame, forecast_col: str, config: dict | None) -> pd.DataFrame:
    out = pd.DataFrame(index=values.index)
    dt = _local_datetime(values)
    hour = _hour(values, dt=dt)
    month = _month(values, dt=dt)
    dow = _dow(values, dt=dt)
    forecast_day = _forecast_day(values, dt=dt)
    base = _base_forecast(values, forecast_col=forecast_col)
    raw = _optional_num(values, "Raw_Forecast_MWH", default=np.nan)
    daily_max = _optional_num(values, "Temperature_DailyMax", default=np.nan)
    temp = _optional_num(values, "Temperature", default=np.nan)
    cloud = _cloud_norm(values)
    solar_loss = _solar_loss(values)
    same7 = _optional_num(values, "MWH_SameHour7DayMean", "Baseline_Rolling7DaySameHourAvg_MWH", default=np.nan)
    lag24 = _optional_num(values, "MWH_Lag24", "Baseline_SameHourYesterday_MWH", default=np.nan)
    xgb = _optional_num(values, "XGB_Pred_MWH", default=np.nan)
    lgb = _optional_num(values, "LGB_Pred_MWH", default=np.nan)
    cat = _optional_num(values, "CatBoost_Pred_MWH", default=np.nan)
    prophet = _optional_num(values, "Prophet_Pred_MWH", default=np.nan)
    components = pd.concat([xgb, lgb, cat, prophet], axis=1)
    tree_components = pd.concat([xgb, lgb, cat], axis=1)
    rule_union, scope = _shape_scope(values, config, forecast_col)

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
    out["Temperature_Above_85F"] = (daily_max - 85.0).clip(lower=0.0)
    out["Temperature_Above_90F"] = (daily_max - 90.0).clip(lower=0.0)
    out["Temperature_Above_100F"] = (daily_max - 100.0).clip(lower=0.0)
    out["Temperature_Drop_From_DailyMax_F"] = _optional_num(
        values,
        "Temperature_Drop_From_DailyMax_F",
        default=np.nan,
    ).where(lambda s: s.notna(), daily_max - temp)
    out["TempDrop_Next1Hr_F"] = _optional_num(values, "TempDrop_Next1Hr_F", default=np.nan)
    out["TempDrop_Next2Hr_F"] = _optional_num(values, "TempDrop_Next2Hr_F", default=np.nan)
    out["TempDrop_Next3Hr_F"] = _optional_num(values, "TempDrop_Next3Hr_F", default=np.nan)
    out["CloudCover_Norm"] = cloud
    out["Humidity_Norm"] = _optional_num(values, "Humidity_Norm", default=np.nan)
    out["WindSpeed_Mph"] = _optional_num(values, "WindSpeed_Mph", default=np.nan)
    out["WindDirection_Deg"] = _optional_num(values, "WindDirection_Deg", default=np.nan)
    out["Westerly_Flow_Mph"] = _optional_num(values, "Westerly_Flow_Mph", default=0.0)
    out["Westerly_Flow_Flag"] = _optional_num(values, "Westerly_Flow_Flag", default=0.0)
    out["WindRamp_Next3Hr_Mph"] = _optional_num(values, "WindRamp_Next3Hr_Mph", default=np.nan)
    out["WesterlyFlow_Next3Hr_Ramp_Mph"] = _optional_num(values, "WesterlyFlow_Next3Hr_Ramp_Mph", default=np.nan)
    out["DeltaBreeze_Cooling_Flag"] = _optional_num(values, "DeltaBreeze_Cooling_Flag", default=0.0)
    out["DeltaBreeze_Westerly_Flow_Flag"] = _optional_num(values, "DeltaBreeze_Westerly_Flow_Flag", default=0.0)
    out["DeltaBreeze_Cooling_Signal"] = _optional_num(values, "DeltaBreeze_Cooling_Signal", default=0.0)
    out["DeltaBreeze_ClearHotEvening_Signal"] = _optional_num(values, "DeltaBreeze_ClearHotEvening_Signal", default=0.0)
    out["PostPeak_LoadDecay_1Hr_MWH"] = _optional_num(values, "PostPeak_LoadDecay_1Hr_MWH", default=np.nan)
    out["PostPeak_LoadDecay_2Hr_MWH"] = _optional_num(values, "PostPeak_LoadDecay_2Hr_MWH", default=np.nan)
    out["PostPeak_LoadDecay_VsSameHour7DayMean_MWH"] = _optional_num(values, "PostPeak_LoadDecay_VsSameHour7DayMean_MWH", default=np.nan)
    out["DeltaBreeze_PostPeak_LoadDecay_Signal"] = _optional_num(values, "DeltaBreeze_PostPeak_LoadDecay_Signal", default=0.0)
    out["BTM_Solar_Loss_MW"] = solar_loss
    out["BTM_Solar_Proxy_MW"] = _optional_num(values, "BTM_Solar_Proxy_MW", default=0.0)
    out["ClearSky_Index"] = _optional_num(values, "ClearSky_Index", default=np.nan)
    out["MWH_Lag24"] = lag24
    out["MWH_SameHour7DayMean"] = same7
    out["Raw_Minus_SameHour7DayMean_MWH"] = _optional_num(
        values,
        "Raw_Minus_SameHour7DayMean_MWH",
        default=np.nan,
    ).where(lambda s: s.notna(), raw - same7)
    out["Raw_Minus_SameHourYesterday_MWH"] = _optional_num(
        values,
        "Raw_Minus_SameHourYesterday_MWH",
        default=np.nan,
    ).where(lambda s: s.notna(), raw - lag24)
    out["Base_Minus_Raw_MWH"] = base - raw
    out["Base_Minus_SameHour7DayMean_MWH"] = base - same7
    out["Base_Minus_SameHourYesterday_MWH"] = base - lag24
    out["XGB_Gap_MWH"] = xgb - base
    out["LGB_Gap_MWH"] = lgb - base
    out["CatBoost_Gap_MWH"] = cat - base
    out["Prophet_Gap_MWH"] = prophet - base
    out["Tree_Spread_MWH"] = tree_components.max(axis=1, skipna=True) - tree_components.min(axis=1, skipna=True)
    out["Component_Spread_MWH"] = components.max(axis=1, skipna=True) - components.min(axis=1, skipna=True)
    out["Peak_Risk_Cal_MWH"] = _optional_num(values, "Peak_Risk_Cal_MWH", default=0.0)
    out["Recent_Level_Correction_MWH"] = _optional_num(values, "Recent_Level_Correction_MWH", default=0.0)
    out["Weather_Robustness_Hedge_MWH"] = _optional_num(values, "Weather_Robustness_Hedge_MWH", default=0.0)
    out["RuleUnion_Flag"] = rule_union.astype(float)
    out["ShapeScope_Flag"] = scope.astype(float)
    out["Hour_Sin"] = np.sin(2.0 * np.pi * hour / 24.0)
    out["Hour_Cos"] = np.cos(2.0 * np.pi * hour / 24.0)
    out["Month_Sin"] = np.sin(2.0 * np.pi * month / 12.0)
    out["Month_Cos"] = np.cos(2.0 * np.pi * month / 12.0)
    out["ForecastDay_Log1p"] = np.log1p(forecast_day.clip(lower=0.0))
    out["HotPeak_x_TempAbove90"] = out["ShapeScope_Flag"] * out["Temperature_Above_90F"]
    out["RuleUnion_x_BaseMinusRaw"] = out["RuleUnion_Flag"] * out["Base_Minus_Raw_MWH"]
    return out


def _new_model(cfg: dict, min_samples_leaf: int) -> HistGradientBoostingRegressor:
    return HistGradientBoostingRegressor(
        loss=str(cfg.get("loss", "absolute_error")),
        learning_rate=float(cfg.get("learning_rate", 0.04)),
        max_iter=int(cfg.get("max_iter", 120)),
        max_leaf_nodes=int(cfg.get("max_leaf_nodes", 16)),
        min_samples_leaf=max(2, int(min_samples_leaf)),
        l2_regularization=float(cfg.get("l2_regularization", 5.0)),
        random_state=int(cfg.get("random_state", 42)),
    )


def _target_residual(values: pd.DataFrame, forecast_col: str) -> pd.Series:
    actual_col = "Actual_MWH" if "Actual_MWH" in values.columns else "Actual"
    actual = _as_num(values.get(actual_col, pd.Series(np.nan, index=values.index)), values.index)
    return actual - _base_forecast(values, forecast_col=forecast_col)


def _training_matrix(
    values: pd.DataFrame,
    target: pd.Series,
    forecast_col: str,
    config: dict | None,
) -> tuple[pd.DataFrame, pd.Series, pd.Series, pd.Series, pd.Series]:
    features = _feature_frame(values, forecast_col=forecast_col, config=config)
    target = _as_num(target, values.index)
    rule_union = features["RuleUnion_Flag"].fillna(0.0).gt(0.0)
    scope = features["ShapeScope_Flag"].fillna(0.0).gt(0.0)
    valid = features["Base_Forecast_MWH"].notna() & target.notna() & scope
    x = features.loc[valid].copy()
    y = target.loc[valid].copy()
    fill = x.median(numeric_only=True).replace([np.inf, -np.inf], np.nan).fillna(0.0)
    return x.fillna(fill).replace([np.inf, -np.inf], 0.0).fillna(0.0), y, fill, rule_union.loc[valid], scope.loc[valid]


def build_focused_shape_residual_learner(
    backtest_df: pd.DataFrame,
    config: dict | None,
    *,
    forecast_col: str = "Final_Backtest_Forecast_MWH",
) -> dict | None:
    cfg = _cfg(config)
    if backtest_df is None or backtest_df.empty or not bool(cfg.get("enabled", False)):
        return None

    target = _target_residual(backtest_df, forecast_col=forecast_col)
    target_clip = float(cfg.get("target_clip_mwh", 35.0))
    if target_clip > 0:
        target = target.clip(-target_clip, target_clip)

    x, y, fill, rule_union, scope = _training_matrix(backtest_df, target, forecast_col, config)
    min_rows = int(cfg.get("min_rows", 168))
    if len(x) < min_rows:
        return None

    model = _new_model(cfg, min_samples_leaf=int(cfg.get("min_samples_leaf", 16)))
    model.fit(x, y)
    return {
        "model": model,
        "fill_values": fill,
        "feature_columns": list(x.columns),
        "metadata": {
            "model_version": str(cfg.get("model_version", "focused_shape_residual_shadow_v1")),
            "training_rows": int(len(x)),
            "rule_union_training_rows": int(rule_union.sum()),
            "shape_scope_training_rows": int(scope.sum()),
            "target_residual_mean_mwh": float(y.mean()),
            "target_residual_mae_mwh": float(y.abs().mean()),
            "forecast_col": forecast_col,
            "shadow_mode": bool(cfg.get("shadow_mode", True)),
        },
    }


def _init_columns(
    out: pd.DataFrame,
    base: pd.Series,
    *,
    model_version: str,
    source: str,
    evaluation_mode: str,
    shadow_mode: bool,
) -> pd.DataFrame:
    out["Focused_Shape_Model_Version"] = model_version
    out["Focused_Shape_Shadow_Mode"] = int(bool(shadow_mode))
    out["Focused_Shape_Base_Forecast_MWH"] = base
    out["Focused_Shape_Correction_MWH"] = 0.0
    out["Focused_Shape_Adjusted_Forecast_MWH"] = base
    out["Focused_Shape_Correction_Applied_Flag"] = 0
    out["Focused_Shape_Source"] = source
    out["Focused_Shape_Evaluation_Mode"] = evaluation_mode
    out["Focused_Shape_RuleUnion_Flag"] = 0
    out["Focused_Shape_Scope_Flag"] = 0
    actual_col = "Actual_MWH" if "Actual_MWH" in out.columns else "Actual"
    actual = _as_num(out.get(actual_col, pd.Series(np.nan, index=out.index)), out.index)
    residual = actual - base
    out["Focused_Shape_Residual_MWH"] = residual
    out["Focused_Shape_AbsError_MWH"] = residual.abs()
    out["Focused_Shape_Delta_AbsError_MWH"] = 0.0
    return out


def apply_focused_shape_residual_learner(
    df: pd.DataFrame,
    artifact: dict | None,
    config: dict | None,
    *,
    forecast_col: str = "Pre_Focused_Guard_Forecast_MWH",
    also_update_cols: tuple[str, ...] = (),
    force_shadow: bool | None = None,
    update_forecast_col: bool = True,
    evaluation_mode: str = "shadow",
) -> pd.DataFrame:
    """Apply the focused-shape learner.

    With ``shadow_mode: true`` the production forecast column is not changed.
    When explicitly promoted with ``shadow_mode: false``, callers can keep the
    pre-guard baseline intact by setting ``update_forecast_col=False`` and
    passing production columns through ``also_update_cols``.
    """
    out = df.copy()
    cfg = _cfg(config)
    shadow_mode = bool(cfg.get("shadow_mode", True)) if force_shadow is None else bool(force_shadow)
    model_version = str(cfg.get("model_version", "focused_shape_residual_shadow_v1"))
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
    rule_union, scope = _shape_scope(out, config, forecast_col)
    out["Focused_Shape_RuleUnion_Flag"] = rule_union.astype(int)
    out["Focused_Shape_Scope_Flag"] = scope.astype(int)

    if not artifact or artifact.get("model") is None:
        out["Focused_Shape_Source"] = "insufficient_history"
        return out
    columns = list(artifact.get("feature_columns") or [])
    if not columns:
        out["Focused_Shape_Source"] = "no_feature_columns"
        return out

    features = _feature_frame(out, forecast_col=forecast_col, config=config).reindex(columns=columns)
    valid = base.notna() & scope
    if not valid.any():
        out["Focused_Shape_Source"] = "out_of_scope"
        return out

    fill_values = artifact.get("fill_values", pd.Series(dtype=float))
    x = features.loc[valid].fillna(fill_values).replace([np.inf, -np.inf], 0.0).fillna(0.0)
    raw_pred = pd.Series(np.asarray(artifact["model"].predict(x), dtype=float), index=x.index)
    correction = pd.Series(0.0, index=out.index, dtype=float)
    cap = float(cfg.get("cap_mwh", 12.0))
    correction.loc[valid] = (raw_pred * float(cfg.get("blend", 0.55))).clip(-cap, cap)
    min_abs = float(cfg.get("min_abs_correction_mwh", 0.25))
    if min_abs > 0:
        correction.loc[correction.abs().lt(min_abs)] = 0.0

    guard_cfg = cfg.get("promotion_delta_guard", {}) or {}
    application_base = base
    if bool(cfg.get("apply_correction_to_reference_forecast", False)):
        reference_base = _promotion_reference_forecast(out, forecast_col, guard_cfg)
        application_base = reference_base.where(reference_base.notna(), base)
        out["Focused_Shape_Base_Forecast_MWH"] = application_base

    adjusted = (application_base + correction).clip(lower=0.0)
    delta_guard_capped = pd.Series(False, index=out.index, dtype=bool)
    if not shadow_mode and bool(guard_cfg.get("enabled", False)):
        reference = _promotion_reference_forecast(out, forecast_col, guard_cfg)
        ref_valid = valid & reference.notna() & adjusted.notna()
        if ref_valid.any():
            default_cap = abs(float(guard_cfg.get("max_abs_delta_vs_reference_mwh", 3.0)))
            cap_values = pd.Series(default_cap, index=out.index, dtype=float)
            risk_cap_raw = guard_cfg.get("risk_slice_max_abs_delta_vs_reference_mwh")
            if risk_cap_raw is not None:
                risk_cap = abs(float(risk_cap_raw))
                risk_mask = _promotion_risk_slice_mask(out, guard_cfg)
                cap_values.loc[risk_mask] = risk_cap
            guarded = adjusted.copy()
            delta_vs_reference = (adjusted.loc[ref_valid] - reference.loc[ref_valid]).clip(
                lower=-cap_values.loc[ref_valid],
                upper=cap_values.loc[ref_valid],
            )
            guarded.loc[ref_valid] = reference.loc[ref_valid] + delta_vs_reference
            delta_guard_capped = ref_valid & (adjusted - guarded).abs().gt(1e-9)
            adjusted = guarded.clip(lower=0.0)
            correction = (adjusted - application_base).where(valid, 0.0).fillna(0.0)

    source_label = "focused_shape_shadow" if shadow_mode else "focused_shape_production"
    source = pd.Series("out_of_scope", index=out.index, dtype="object")
    source.loc[valid] = source_label
    source.loc[valid & rule_union] = f"{source_label}+rule_union_scope"
    source.loc[base.isna()] = "invalid_base_forecast"
    source.loc[valid & correction.eq(0.0)] = f"{source_label}_zeroed"
    source.loc[delta_guard_capped] = source.loc[delta_guard_capped].astype(str) + "+promotion_delta_guard"

    out["Focused_Shape_Correction_MWH"] = correction
    out["Focused_Shape_Adjusted_Forecast_MWH"] = adjusted
    out["Focused_Shape_Correction_Applied_Flag"] = correction.ne(0.0).astype(int)
    out["Focused_Shape_Source"] = source

    actual_col = "Actual_MWH" if "Actual_MWH" in out.columns else "Actual"
    actual = _as_num(out.get(actual_col, pd.Series(np.nan, index=out.index)), out.index)
    base_residual = actual - application_base
    shadow_residual = actual - adjusted
    out["Focused_Shape_Residual_MWH"] = shadow_residual
    out["Focused_Shape_AbsError_MWH"] = shadow_residual.abs()
    out["Focused_Shape_Delta_AbsError_MWH"] = shadow_residual.abs() - base_residual.abs()

    if not shadow_mode:
        promote_cols: list[str] = []
        if update_forecast_col and forecast_col in out.columns:
            promote_cols.append(forecast_col)
        promote_cols.extend(col for col in also_update_cols if col in out.columns)
        if "Final_Backtest_Forecast_MWH" in promote_cols and "Final_Forecast_MWH" in out.columns:
            promote_cols.append("Final_Forecast_MWH")
        for col in dict.fromkeys(promote_cols):
            out[col] = adjusted

        final_metric_col = None
        if "Final_Backtest_Forecast_MWH" in out.columns and "Final_Backtest_Forecast_MWH" in promote_cols:
            final_metric_col = "Final_Backtest_Forecast_MWH"
        elif "Final_Forecast_MWH" in out.columns and "Final_Forecast_MWH" in promote_cols:
            final_metric_col = "Final_Forecast_MWH"
        if final_metric_col is not None and actual.notna().any():
            final_residual = actual - _as_num(out[final_metric_col], out.index)
            out["Final_Residual_MWH"] = final_residual
            out["Final_AbsError_MWH"] = final_residual.abs()
            out["Final_APE"] = np.where(
                actual.abs() > 1e-9,
                out["Final_AbsError_MWH"] / actual.abs() * 100.0,
                np.nan,
            )
    return out


def focused_shape_residual_summary(backtest_df: pd.DataFrame | None, artifact: dict | None, config: dict | None) -> dict:
    cfg = _cfg(config)
    summary: dict[str, Any] = {
        "enabled": bool(cfg.get("enabled", False)),
        "shadow_mode": bool(cfg.get("shadow_mode", True)),
        "model_version": str(cfg.get("model_version", "focused_shape_residual_shadow_v1")),
    }
    if artifact and artifact.get("metadata"):
        summary["artifact"] = dict(artifact.get("metadata") or {})
    if backtest_df is None or backtest_df.empty or "Focused_Shape_Adjusted_Forecast_MWH" not in backtest_df.columns:
        summary["evaluation_rows"] = 0
        return summary

    actual = _as_num(backtest_df.get("Actual_MWH", pd.Series(np.nan, index=backtest_df.index)), backtest_df.index)
    base = _as_num(backtest_df.get("Focused_Shape_Base_Forecast_MWH", pd.Series(np.nan, index=backtest_df.index)), backtest_df.index)
    shadow = _as_num(backtest_df["Focused_Shape_Adjusted_Forecast_MWH"], backtest_df.index)
    applied = _as_num(
        backtest_df.get("Focused_Shape_Correction_Applied_Flag", pd.Series(0, index=backtest_df.index)),
        backtest_df.index,
    ).eq(1)
    valid = actual.notna() & base.notna() & shadow.notna() & applied
    summary["evaluation_rows"] = int(valid.sum())
    if not valid.any():
        return summary

    base_abs = (actual[valid] - base[valid]).abs()
    shadow_abs = (actual[valid] - shadow[valid]).abs()
    delta = shadow_abs - base_abs
    summary["baseline_mae_mwh"] = float(base_abs.mean())
    summary["focused_shape_shadow_mae_mwh"] = float(shadow_abs.mean())
    summary["delta_mae_mwh"] = float(delta.mean())
    summary["improved_rows"] = int((delta < 0).sum())
    summary["worsened_rows"] = int((delta > 0).sum())
    summary["mean_correction_mwh"] = float(
        _as_num(backtest_df.loc[valid, "Focused_Shape_Correction_MWH"], backtest_df.loc[valid].index).mean()
    )

    rule_union = _as_num(
        backtest_df.get("Focused_Shape_RuleUnion_Flag", pd.Series(0, index=backtest_df.index)),
        backtest_df.index,
    ).eq(1) & valid
    summary["rule_union_evaluation_rows"] = int(rule_union.sum())
    if rule_union.any():
        rule_base_abs = (actual[rule_union] - base[rule_union]).abs()
        rule_shadow_abs = (actual[rule_union] - shadow[rule_union]).abs()
        summary["rule_union_baseline_mae_mwh"] = float(rule_base_abs.mean())
        summary["rule_union_shadow_mae_mwh"] = float(rule_shadow_abs.mean())
        summary["rule_union_delta_mae_mwh"] = float((rule_shadow_abs - rule_base_abs).mean())
    return summary
