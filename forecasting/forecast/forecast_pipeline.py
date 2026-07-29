from __future__ import annotations

import pandas as pd
import numpy as np
from typing import Callable, Optional

from forecasting.data.history_loader import load_hourly_system_mwh
from forecasting.data.weather_loader import fetch_historical_weather, fetch_forecast_weather
from forecasting.data.btm_loader import load_btm_monthly_capacity
from forecasting.data.five_min_load_loader import load_five_min_system_load
from forecasting.data.solar_loader import load_solar_forecast
from forecasting.data.local_weather_loader import (
    apply_temperature_bias_calibration,
    apply_dynamic_temperature_calibration,
    build_temperature_bias_lookup,
    load_local_station_weather,
    local_temperature_bias_summary,
    merge_local_station_temperature,
)

from forecasting.features.feature_builder import build_training_frame, build_forecast_frame
from forecasting.features.intraday_load_features import (
    append_recent_five_min_hourly_load,
    build_hourly_load_from_five_min,
    build_intraday_load_feature_frame,
)

from forecasting.model.xgb_model import DEFAULT_FEATURES as XGB_FEATURES
from forecasting.model.trainers import train_tree_models
from forecasting.model.prophet_model import train_prophet, DEFAULT_PROPHET_REGRESSORS, prophet_enabled
from forecasting.model.catboost_model import train_catboost, catboost_enabled

from forecasting.forecast.recursive_engine import recursive_forecast
from forecasting.forecast.calibration import (
    build_learned_residual_lookups,
    apply_learned_calibration,
    build_heat_peak_lookup,
    apply_heat_peak_calibration,
    build_warm_ramp_lookup,
    apply_warm_ramp_correction,
)
from forecasting.forecast.recent_residual_correction import (
    build_recent_residual_profile,
    apply_recent_residual_correction,
    simulate_recent_residual_correction_backtest,
)
from forecasting.forecast.uncertainty_bands import build_residual_band_lookup, apply_bands
from forecasting.forecast.event_shape_corrections import build_cloud_solar_shape_lookup, apply_cloud_solar_shape_correction
from forecasting.forecast.peak_risk_correction import apply_peak_risk_correction
from forecasting.forecast.weather_robustness_hedge import apply_weather_robustness_hedge
from forecasting.forecast.focused_scorecard_guard import apply_focused_scorecard_guard
from forecasting.forecast.anomaly_exclusions import drop_excluded_intervals
from forecasting.forecast.targeted_residual_meta import (
    apply_targeted_residual_meta_correction,
    build_targeted_residual_meta_model,
)
from forecasting.forecast.operational_residual_learner import (
    apply_operational_residual_learner,
    build_operational_residual_learner,
    operational_residual_learner_summary,
    simulate_operational_residual_learner_backtest,
)
from forecasting.forecast.weather_scenarios import (
    add_scenario_summary_columns,
    apply_weather_scenario_delta_caps,
    apply_conformal_weather_bands,
    build_weather_stress_summary,
    make_weather_scenario_frame,
    scenario_column_name,
    scenario_definitions,
)
from forecasting.backtest.rolling_backtest import run_rolling_backtest
from forecasting.diagnostics import build_diagnostics_bundle

ProgressCallback = Callable[[str, int, int | None], None]


def _progress(
    progress_callback: ProgressCallback | None,
    label: str,
    advance: int = 0,
    total: int | None = None,
) -> None:
    if progress_callback is not None:
        progress_callback(label, advance, total)


def _trim_incomplete_future_weather(future_frame: pd.DataFrame, required_cols: list[str] | None = None) -> pd.DataFrame:
    """Drop incomplete weather rows at the live forecast tail before recursive prediction.

    Open-Meteo can occasionally return the requested horizon plus partially populated trailing
    hours. Keeping those rows lets zero-filled weather features leak into the forecast. Treat an
    incomplete run of weather rows as the end of the usable operational horizon.
    """
    if future_frame is None or future_frame.empty:
        return future_frame
    required = required_cols or ["Temperature"]
    required = [col for col in required if col in future_frame.columns]
    if not required:
        return future_frame

    out = future_frame.sort_values("DT").copy()
    bad = out[required].isna().any(axis=1)
    if not bad.any():
        return out

    first_bad_pos = int(np.flatnonzero(bad.to_numpy())[0])
    if first_bad_pos == 0:
        raise RuntimeError(
            "Forecast weather is incomplete from the first future hour; refresh weather inputs before forecasting."
        )
    dropped = len(out) - first_bad_pos
    first_bad_dt = out.iloc[first_bad_pos]["DT"]
    print(f"WARNING: Dropping {dropped} incomplete future weather rows starting at {first_bad_dt}.")
    return out.iloc[:first_bad_pos].copy()


def _extend_historical_weather_with_recent_forecast(
    hist_wx: pd.DataFrame,
    fut_wx: pd.DataFrame,
    latest_load_dt: pd.Timestamp | None,
) -> pd.DataFrame:
    """Use forecast API past-hours rows only to cover recent 5-minute load additions.

    Open-Meteo archive history normally ends yesterday local. If the 5-minute load feed
    supplies completed hours for today, those rows need weather features. The forecast API
    is already requested with `forecast_past_hours`, so use those recent rows without
    replacing archive weather where archive rows exist.
    """
    if hist_wx is None or hist_wx.empty or fut_wx is None or fut_wx.empty or latest_load_dt is None:
        return hist_wx
    latest = pd.Timestamp(latest_load_dt)
    recent = fut_wx.copy()
    recent["DT"] = pd.to_datetime(recent["DT"], errors="coerce")
    recent = recent[recent["DT"].notna() & (recent["DT"] <= latest)].copy()
    if recent.empty:
        return hist_wx
    combined = pd.concat([recent, hist_wx], ignore_index=True, sort=False)
    combined.sort_values("DT", inplace=True)
    return combined.drop_duplicates(subset=["DT"], keep="last").reset_index(drop=True)


def _production_ensemble_weights(config: dict) -> dict[str, float]:
    """Return weights for the production raw forecast.

    V12 treats Prophet as a benchmark by default. It is still trained/exported when enabled, but it does
    not affect Raw_Forecast_MWH unless model.prophet.blend_into_production is explicitly true.
    """
    weights = dict(config.get("model", {}).get("ensemble_weights", {}) or {})
    prop_cfg = config.get("model", {}).get("prophet", {}) or {}
    if not bool(prop_cfg.get("blend_into_production", False)):
        weights["prophet"] = 0.0
    cat_cfg = config.get("model", {}).get("catboost", {}) or {}
    if not bool(cat_cfg.get("blend_into_production", False)):
        weights["catboost"] = 0.0
    return weights


def build_correction_artifacts(raw_backtest_df: pd.DataFrame, config: dict) -> dict:
    """Build correction lookups and the origin-available recent profile from raw residuals."""
    cal_cfg = config.get("calibration", {})
    artifacts = {
        "targeted_meta_artifact": None,
        "lookup_bundle": {},
        "heat_lookup": None,
        "warm_lookup": None,
        "cloud_solar_lookup": None,
        "recent_profile": None,
        "operational_residual_artifact": None,
        "pre_recent_frame": pd.DataFrame(),
    }
    if raw_backtest_df is None or raw_backtest_df.empty:
        return artifacts

    # Pure exclusion: drop configured anomalous intervals (e.g. DER dispatch hours) so they
    # never enter the targeted-meta model, the learned calibration lookups, or the
    # recent-residual profile. This runs for both production and rolling-origin replay,
    # which are the only two callers of build_correction_artifacts.
    raw_backtest_df = drop_excluded_intervals(raw_backtest_df, config)
    if raw_backtest_df is None or raw_backtest_df.empty:
        return artifacts

    residual_basis_df = raw_backtest_df.copy()
    targeted_meta_cfg = cal_cfg.get("targeted_residual_meta", {}) or {}
    if bool(targeted_meta_cfg.get("enabled", True)):
        artifacts["targeted_meta_artifact"] = build_targeted_residual_meta_model(raw_backtest_df, config)
        if artifacts["targeted_meta_artifact"]:
            residual_basis_df = apply_targeted_residual_meta_correction(
                raw_backtest_df,
                artifacts["targeted_meta_artifact"],
                config,
            )
            residual_basis_df["Residual_MWH"] = (
                pd.to_numeric(residual_basis_df["Actual_MWH"], errors="coerce")
                - pd.to_numeric(residual_basis_df["Targeted_Meta_Adjusted_Forecast_MWH"], errors="coerce")
            )

    if bool(cal_cfg.get("seasonal_enabled", True)):
        artifacts["lookup_bundle"] = build_learned_residual_lookups(
            backtest_df=residual_basis_df,
            blend=float(cal_cfg.get("blend", 0.85)),
            cap_mwh=float(cal_cfg.get("cap_mwh", 22.0)),
            shrink_k=float(cal_cfg.get("shrink_k", 24.0)),
        )
    if bool(cal_cfg.get("heat_peak_enabled", True)):
        artifacts["heat_lookup"] = build_heat_peak_lookup(
            backtest_df=residual_basis_df,
            min_maxtemp_f=float(cal_cfg.get("heat_peak_min_maxtemp_f", 88.0)),
            hours=list(cal_cfg.get("heat_peak_hours", [14, 15, 16, 17, 18, 19, 20])),
            blend=float(cal_cfg.get("heat_peak_blend", 0.50)),
            cap_mwh=float(cal_cfg.get("heat_peak_cap_mwh", 12.0)),
            shrink_k=float(cal_cfg.get("heat_peak_shrink_k", 18.0)),
        )
    if bool(cal_cfg.get("warm_ramp_enabled", True)):
        artifacts["warm_lookup"] = build_warm_ramp_lookup(
            backtest_df=residual_basis_df,
            min_maxtemp_f=float(cal_cfg.get("warm_ramp_min_maxtemp_f", 75.0)),
            max_maxtemp_f=float(cal_cfg.get("warm_ramp_max_maxtemp_f", 93.0)),
            hours=list(cal_cfg.get("warm_ramp_hours", [12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22])),
            blend=float(cal_cfg.get("warm_ramp_blend", 0.88)),
            cap_mwh=float(cal_cfg.get("warm_ramp_cap_mwh", 16.0)),
            shrink_k=float(cal_cfg.get("warm_ramp_shrink_k", 8.0)),
        )
    if bool(cal_cfg.get("cloud_solar_shape_enabled", True)):
        pre_cloud_lookup_frame = _apply_pre_cloud_correction_chain_to_frame(
            raw_df=raw_backtest_df,
            config=config,
            targeted_meta_artifact=artifacts["targeted_meta_artifact"],
            lookup_bundle=artifacts["lookup_bundle"],
            heat_lookup=artifacts["heat_lookup"],
            warm_lookup=artifacts["warm_lookup"],
        )
        pre_cloud_lookup_frame["Pre_Cloud_Residual_MWH"] = (
            pd.to_numeric(pre_cloud_lookup_frame["Actual_MWH"], errors="coerce")
            - pd.to_numeric(pre_cloud_lookup_frame["Calibrated_Forecast_MWH"], errors="coerce")
        )
        artifacts["cloud_solar_lookup"] = build_cloud_solar_shape_lookup(
            backtest_df=pre_cloud_lookup_frame,
            hours=list(cal_cfg.get("cloud_solar_shape_hours", [10, 11, 12, 13, 14, 15, 16])),
            blend=float(cal_cfg.get("cloud_solar_shape_blend", 0.78)),
            cap_mwh=float(cal_cfg.get("cloud_solar_shape_cap_mwh", 16.0)),
            shrink_k=float(cal_cfg.get("cloud_solar_shape_shrink_k", 10.0)),
            min_loss_mw=float(cal_cfg.get("cloud_solar_shape_min_loss_mw", 1.25)),
            residual_col="Pre_Cloud_Residual_MWH",
            forecast_col="Calibrated_Forecast_MWH",
        )

    pre_recent_frame = _apply_v126_correction_chain_to_frame(
        raw_df=raw_backtest_df,
        config=config,
        targeted_meta_artifact=artifacts["targeted_meta_artifact"],
        lookup_bundle=artifacts["lookup_bundle"],
        heat_lookup=artifacts["heat_lookup"],
        warm_lookup=artifacts["warm_lookup"],
        cloud_solar_lookup=artifacts["cloud_solar_lookup"],
        simulate_recent=False,
        apply_auto_residual=False,
    )
    artifacts["pre_recent_frame"] = pre_recent_frame
    if bool((cal_cfg.get("recent_residual", {}) or {}).get("enabled", True)):
        recent_profile_frame = pre_recent_frame.copy()
        recent_basis_col = (
            "Pre_Recent_Forecast_MWH"
            if "Pre_Recent_Forecast_MWH" in recent_profile_frame.columns
            else "Calibrated_Forecast_MWH"
        )
        if recent_basis_col in recent_profile_frame.columns:
            recent_profile_frame["Residual_MWH"] = (
                pd.to_numeric(recent_profile_frame["Actual_MWH"], errors="coerce")
                - pd.to_numeric(recent_profile_frame[recent_basis_col], errors="coerce")
            )
        artifacts["recent_profile"] = build_recent_residual_profile(recent_profile_frame, config=config)
    if bool((cal_cfg.get("operational_residual_learner", {}) or {}).get("enabled", False)):
        auto_residual_basis = _apply_v126_correction_chain_to_frame(
            raw_df=raw_backtest_df,
            config=config,
            targeted_meta_artifact=artifacts["targeted_meta_artifact"],
            lookup_bundle=artifacts["lookup_bundle"],
            heat_lookup=artifacts["heat_lookup"],
            warm_lookup=artifacts["warm_lookup"],
            cloud_solar_lookup=artifacts["cloud_solar_lookup"],
            simulate_recent=True,
            apply_auto_residual=False,
        )
        artifacts["operational_residual_artifact"] = build_operational_residual_learner(
            auto_residual_basis,
            config,
            forecast_col="Final_Backtest_Forecast_MWH",
        )
    return artifacts


def apply_origin_available_correction_chain(raw_df: pd.DataFrame, config: dict, artifacts: dict | None) -> pd.DataFrame:
    """Apply corrections using only lookup/profile state available before the scored origin."""
    artifacts = artifacts or {}
    out = _apply_v126_correction_chain_to_frame(
        raw_df=raw_df,
        config=config,
        targeted_meta_artifact=artifacts.get("targeted_meta_artifact"),
        lookup_bundle=artifacts.get("lookup_bundle"),
        heat_lookup=artifacts.get("heat_lookup"),
        warm_lookup=artifacts.get("warm_lookup"),
        cloud_solar_lookup=artifacts.get("cloud_solar_lookup"),
        simulate_recent=False,
        apply_auto_residual=False,
    )
    cal_cfg = config.get("calibration", {})
    if bool((cal_cfg.get("recent_residual", {}) or {}).get("enabled", True)):
        out = apply_recent_residual_correction(
            future_df=out,
            profile=artifacts.get("recent_profile"),
            config=config,
            base_col="Calibrated_Forecast_MWH",
        )
        out["Final_Backtest_Forecast_MWH"] = out["Final_Forecast_MWH"]
    elif "Final_Backtest_Forecast_MWH" not in out.columns:
        out["Final_Backtest_Forecast_MWH"] = out["Calibrated_Forecast_MWH"]

    out = apply_operational_stage_selector(out, config=config, forecast_col="Final_Backtest_Forecast_MWH")
    final_col = "Final_Backtest_Forecast_MWH"
    out["Final_Residual_MWH"] = pd.to_numeric(out["Actual_MWH"], errors="coerce") - pd.to_numeric(out[final_col], errors="coerce")
    out["Final_AbsError_MWH"] = out["Final_Residual_MWH"].abs()
    out["Final_APE"] = np.where(
        pd.to_numeric(out["Actual_MWH"], errors="coerce").abs() > 1e-9,
        out["Final_AbsError_MWH"] / pd.to_numeric(out["Actual_MWH"], errors="coerce").abs() * 100.0,
        np.nan,
    )
    return out


def _forecast_day_index(out: pd.DataFrame) -> pd.Series:
    if "Forecast_Day" in out.columns:
        day = pd.to_numeric(out["Forecast_Day"], errors="coerce")
        if day.notna().any():
            return day.fillna(999).astype(int)
    dt = pd.to_datetime(out["DT"], errors="coerce")
    if dt.dropna().empty:
        return pd.Series(999, index=out.index, dtype=int)
    first_day = dt.min().normalize()
    return ((dt.dt.normalize() - first_day).dt.days + 1).fillna(999).astype(int)


def _horizon_bucket_for_stage_selector(day: pd.Series) -> pd.Series:
    day_num = pd.to_numeric(day, errors="coerce")
    return pd.Series(
        np.select(
            [
                day_num.le(1),
                day_num.between(2, 7),
                day_num.between(8, 16),
            ],
            ["Day1", "Days2to7", "Days8to16"],
            default="Days17Plus",
        ),
        index=day.index,
        dtype="object",
    )


def _configured_numeric_lookup(
    values: pd.Series,
    lookup_cfg: dict | None,
    default: float = 0.0,
) -> pd.Series:
    if not lookup_cfg:
        return pd.Series(default, index=values.index, dtype=float)
    lookup = {str(k): float(v) for k, v in lookup_cfg.items()}
    direct = values.astype("object").map(lambda x: lookup.get(str(x), np.nan))
    if direct.notna().any():
        return direct.fillna(default).astype(float)
    numeric = pd.to_numeric(values, errors="coerce").round()
    return numeric.map(lambda x: lookup.get(str(int(x)), default) if pd.notna(x) else default).astype(float)


def _configured_pair_numeric_lookup(
    primary_values: pd.Series,
    secondary_values: pd.Series,
    lookup_cfg: dict | None,
    default: float = 1.0,
) -> pd.Series:
    out = pd.Series(default, index=primary_values.index, dtype=float)
    if not lookup_cfg:
        return out

    primary_num = pd.to_numeric(primary_values, errors="coerce").round()
    secondary_num = pd.to_numeric(secondary_values, errors="coerce").round()
    for primary_key, nested_lookup in lookup_cfg.items():
        if not isinstance(nested_lookup, dict):
            continue
        try:
            primary_match = int(float(primary_key))
        except (TypeError, ValueError):
            continue
        primary_mask = primary_num.eq(primary_match)
        for secondary_key, value in nested_lookup.items():
            try:
                secondary_match = int(float(secondary_key))
                scale = float(value)
            except (TypeError, ValueError):
                continue
            out.loc[primary_mask & secondary_num.eq(secondary_match)] = scale
    return out


def apply_operational_stage_selector(df: pd.DataFrame, config: dict, forecast_col: str) -> pd.DataFrame:
    selector_cfg = ((config.get("calibration", {}) or {}).get("stage_selector", {}) or {})
    out = df.copy()
    base_col = forecast_col if forecast_col in out.columns else (
        "Final_Forecast_MWH" if "Final_Forecast_MWH" in out.columns else "Calibrated_Forecast_MWH"
    )
    if not bool(selector_cfg.get("enabled", True)):
        out["Stage_Selected_Forecast_MWH"] = pd.to_numeric(out[base_col], errors="coerce")
        out["Stage_Selector_Source"] = base_col
        out["Stage_Selector_Reason"] = "disabled"
        return out

    selected = pd.to_numeric(out[base_col], errors="coerce").copy()
    source = pd.Series(base_col, index=out.index, dtype="object")
    reason = pd.Series("default_final", index=out.index, dtype="object")
    day = _forecast_day_index(out)
    explicit_forecast_day = (
        "Forecast_Day" in out.columns
        and pd.to_numeric(out.get("Forecast_Day"), errors="coerce").notna().any()
    )
    horizon_policy_enabled = bool(explicit_forecast_day or "Actual_MWH" not in out.columns)
    hour = pd.to_numeric(out.get("Hour", pd.Series(np.nan, index=out.index)), errors="coerce")
    daily_max = pd.to_numeric(out.get("Temperature_DailyMax", pd.Series(np.nan, index=out.index)), errors="coerce")
    hot_peak = hour.between(16, 20) & daily_max.ge(float(selector_cfg.get("hot_peak_min_maxtemp_f", 90.0)))

    raw_col = "Raw_Forecast_MWH"
    targeted_col = "Targeted_Meta_Adjusted_Forecast_MWH"
    peak_col = "Peak_Risk_Adjusted_Forecast_MWH"
    recent_col = "Recent_Corrected_Forecast_MWH"
    residual_col = "Residual_Calibrated_Forecast_MWH"
    heat_col = "Heat_Adjusted_Forecast_MWH"
    warm_col = "Warm_Ramp_Adjusted_Forecast_MWH"
    cloud_col = "Cloud_Solar_Adjusted_Forecast_MWH"
    out["Long_Horizon_Peak_Month_Correction_MWH"] = 0.0
    out["Long_Horizon_Hot_Month_Correction_MWH"] = 0.0
    suppress_bias_for_overrides = bool(selector_cfg.get("suppress_bias_uplift_for_stage_overrides", False))
    suppress_hot_for_overrides = bool(selector_cfg.get("suppress_hot_uplift_for_stage_overrides", False))
    allow_bias_uplift = pd.Series(True, index=out.index, dtype=bool)
    allow_hot_uplift = pd.Series(True, index=out.index, dtype=bool)
    if not horizon_policy_enabled:
        allow_bias_uplift.loc[:] = bool(selector_cfg.get("non_horizon_apply_bias_uplift", False))
        allow_hot_uplift.loc[:] = bool(selector_cfg.get("non_horizon_apply_hot_uplift", False))

    def _stage_values(stage_name: str, fallback_col: str) -> tuple[str, pd.Series]:
        stage = str(stage_name or "").strip().lower()
        if stage in {"raw", "raw_forecast"} and raw_col in out.columns:
            return raw_col, pd.to_numeric(out[raw_col], errors="coerce")
        if stage in {"targeted", "targeted_meta", "targeted_meta_adjusted"} and targeted_col in out.columns:
            return targeted_col, pd.to_numeric(out[targeted_col], errors="coerce")
        if stage in {"peak", "peak_risk", "peak_risk_adjusted"} and peak_col in out.columns:
            return peak_col, pd.to_numeric(out[peak_col], errors="coerce")
        if stage in {"recent", "recent_corrected"} and recent_col in out.columns:
            return recent_col, pd.to_numeric(out[recent_col], errors="coerce")
        if stage in {"residual", "residual_calibrated"} and residual_col in out.columns:
            return residual_col, pd.to_numeric(out[residual_col], errors="coerce")
        if stage in {"heat", "heat_peak", "heat_adjusted"} and heat_col in out.columns:
            return heat_col, pd.to_numeric(out[heat_col], errors="coerce")
        if stage in {"warm", "warm_ramp", "warm_ramp_adjusted"} and warm_col in out.columns:
            return warm_col, pd.to_numeric(out[warm_col], errors="coerce")
        if stage in {"cloud", "cloud_solar", "cloud_solar_adjusted"} and cloud_col in out.columns:
            return cloud_col, pd.to_numeric(out[cloud_col], errors="coerce")
        use_col = fallback_col if fallback_col in out.columns else base_col
        return use_col, pd.to_numeric(out[use_col], errors="coerce")

    def _mark_stage_override(mask: pd.Series) -> None:
        if suppress_bias_for_overrides:
            allow_bias_uplift.loc[mask] = False
        if suppress_hot_for_overrides:
            allow_hot_uplift.loc[mask] = False

    def _conditional_allowed_mask(values: pd.Series, allowed, *, numeric: bool = True) -> pd.Series:
        if allowed is None:
            return pd.Series(True, index=out.index, dtype=bool)
        if numeric:
            allowed_set = {int(x) for x in allowed}
            return pd.to_numeric(values, errors="coerce").round().astype("Int64").isin(allowed_set).fillna(False)
        allowed_set = {str(x).strip() for x in allowed}
        return values.astype(str).str.strip().isin(allowed_set)

    def _month_values() -> pd.Series:
        if "Month" in out.columns:
            month = pd.to_numeric(out["Month"], errors="coerce")
            if month.notna().any():
                return month
        if "DT" in out.columns:
            return pd.to_datetime(out["DT"], errors="coerce").dt.month.astype(float)
        return pd.Series(np.nan, index=out.index, dtype=float)

    def _season_values(month: pd.Series) -> pd.Series:
        if "Season" in out.columns:
            season = out["Season"].where(pd.notna(out["Season"]), "").astype(str).str.strip()
            season = season.mask(season.str.lower().isin({"nan", "nat", "none"}), "")
            if season.ne("").any():
                return season
        season = pd.Series("", index=out.index, dtype="object")
        month_int = pd.to_numeric(month, errors="coerce").round().astype("Int64")
        season.loc[month_int.isin([12, 1, 2]).fillna(False)] = "Winter"
        season.loc[month_int.isin([3, 4, 5]).fillna(False)] = "Spring"
        season.loc[month_int.isin([6, 7, 8, 9]).fillna(False)] = "Summer"
        season.loc[month_int.isin([10, 11]).fillna(False)] = "Fall"
        return season

    def _conditional_stage_mask(rule: dict) -> pd.Series:
        mask = pd.Series(True, index=out.index, dtype=bool)
        month = _month_values()
        season = _season_values(month)
        temp_bucket = pd.to_numeric(
            out.get("DailyMaxTempBucket", out.get("DailyMaxTempBin", pd.Series(np.nan, index=out.index))),
            errors="coerce",
        )
        mask &= _conditional_allowed_mask(month, rule.get("months"))
        mask &= _conditional_allowed_mask(season, rule.get("seasons"), numeric=False)
        mask &= _conditional_allowed_mask(hour, rule.get("hours"))
        if "min_forecast_day" in rule:
            mask &= day.ge(float(rule["min_forecast_day"]))
        if "max_forecast_day" in rule:
            mask &= day.le(float(rule["max_forecast_day"]))
        if "min_maxtemp_f" in rule:
            mask &= daily_max.ge(float(rule["min_maxtemp_f"]))
        if "max_maxtemp_f" in rule:
            mask &= daily_max.lt(float(rule["max_maxtemp_f"]))
        min_temp_bucket = rule.get("min_daily_max_temp_bucket", rule.get("min_daily_max_temp_bin"))
        if min_temp_bucket is not None:
            mask &= temp_bucket.ge(float(min_temp_bucket))
        max_temp_bucket = rule.get("max_daily_max_temp_bucket", rule.get("max_daily_max_temp_bin"))
        if max_temp_bucket is not None:
            mask &= temp_bucket.le(float(max_temp_bucket))
        return mask.fillna(False)

    day1 = horizon_policy_enabled & day.eq(1)
    if day1.any():
        day1_stage = str(selector_cfg.get("day1_stage", "blend_raw_peak")).strip().lower()
        if day1_stage in {"raw", "raw_forecast"}:
            selected_col, values = _stage_values("raw", raw_col)
            selected.loc[day1] = values.loc[day1]
            source.loc[day1] = selected_col
            reason.loc[day1] = "day1_raw"
            _mark_stage_override(day1)
        elif day1_stage in {"blend", "blend_raw_peak", "raw_peak_blend"}:
            peak_weight = float(selector_cfg.get("day1_peak_risk_weight", 0.40))
            raw = pd.to_numeric(out[raw_col], errors="coerce")
            peak = pd.to_numeric(out[peak_col], errors="coerce")
            selected.loc[day1] = raw.loc[day1] * (1.0 - peak_weight) + peak.loc[day1] * peak_weight
            source.loc[day1] = f"{raw_col}+{peak_col}"
            reason.loc[day1] = "day1_raw_peak_blend"
            _mark_stage_override(day1)
        else:
            selected_col, values = _stage_values(day1_stage, residual_col)
            selected.loc[day1] = values.loc[day1]
            source.loc[day1] = selected_col
            reason.loc[day1] = "day1_" + day1_stage
            _mark_stage_override(day1)

    d23 = horizon_policy_enabled & day.between(2, 3)
    if d23.any():
        selected_col, values = _stage_values(selector_cfg.get("days2to3_stage", "peak_risk"), peak_col)
        selected.loc[d23] = values.loc[d23]
        source.loc[d23] = selected_col
        reason.loc[d23] = "days2to3_" + str(selector_cfg.get("days2to3_stage", "peak_risk"))
        _mark_stage_override(d23)

    d47 = horizon_policy_enabled & day.between(4, 7)
    if d47.any():
        selected_col, values = _stage_values(selector_cfg.get("days4to7_stage", "peak_risk"), peak_col)
        selected.loc[d47] = values.loc[d47]
        source.loc[d47] = selected_col
        reason.loc[d47] = "days4to7_" + str(selector_cfg.get("days4to7_stage", "peak_risk"))
        _mark_stage_override(d47)

    d8p = horizon_policy_enabled & day.ge(8) & out.get(residual_col, pd.Series(np.nan, index=out.index)).notna()
    if d8p.any():
        selected.loc[d8p] = pd.to_numeric(out.loc[d8p, base_col], errors="coerce")
        source.loc[d8p] = base_col
        reason.loc[d8p] = "days8plus_final_low_confidence"

    # The full replay showed hot-peak rows still did best with the final correction stack.
    if horizon_policy_enabled and bool(selector_cfg.get("hot_peak_final_stack_override", True)) and hot_peak.any() and base_col in out.columns:
        selected.loc[hot_peak] = pd.to_numeric(out.loc[hot_peak, base_col], errors="coerce")
        source.loc[hot_peak] = base_col
        reason.loc[hot_peak] = "hot_peak_final_stack"

    peak_window_cfg = selector_cfg.get("peak_window_stage_override", {}) or {}
    if horizon_policy_enabled and bool(peak_window_cfg.get("enabled", False)):
        peak_hours = [int(h) for h in peak_window_cfg.get("hours", [14, 15, 16, 17, 18])]
        min_day = int(peak_window_cfg.get("min_forecast_day", 1))
        max_day = int(peak_window_cfg.get("max_forecast_day", 16))
        min_maxtemp = float(peak_window_cfg.get("min_maxtemp_f", -999.0))
        peak_window = day.between(min_day, max_day) & hour.isin(peak_hours) & daily_max.ge(min_maxtemp)
        if bool(peak_window_cfg.get("exclude_hot_peak", True)):
            peak_window = peak_window & ~hot_peak
        if peak_window.any():
            selected_col, values = _stage_values(peak_window_cfg.get("stage", "warm_ramp"), warm_col)
            selected.loc[peak_window] = values.loc[peak_window]
            source.loc[peak_window] = selected_col
            reason.loc[peak_window] = reason.loc[peak_window].astype(str) + "+peak_window_stage_override"
            if bool(peak_window_cfg.get("suppress_bias_uplift", True)):
                allow_bias_uplift.loc[peak_window] = False
            if bool(peak_window_cfg.get("suppress_hot_uplift", True)):
                allow_hot_uplift.loc[peak_window] = False

    cloud_override_cfg = selector_cfg.get("cloud_solar_raw_override", {}) or {}
    if horizon_policy_enabled and bool(cloud_override_cfg.get("enabled", False)):
        override_hours = [int(h) for h in cloud_override_cfg.get("hours", [10, 11, 12, 13, 14, 15, 16])]
        min_cloud = float(cloud_override_cfg.get("min_cloud_cover_norm", 0.60))
        min_loss = float(cloud_override_cfg.get("min_solar_loss_mw", 1.25))
        cloud = pd.to_numeric(out.get("CloudCover_Norm", pd.Series(np.nan, index=out.index)), errors="coerce")
        solar_loss = pd.to_numeric(out.get("BTM_Solar_Loss_From_ClearSky_MW", pd.Series(np.nan, index=out.index)), errors="coerce")
        cloud_override = hour.isin(override_hours) & (cloud.ge(min_cloud) | solar_loss.ge(min_loss))
        if cloud_override.any():
            selected_col, values = _stage_values(cloud_override_cfg.get("stage", "raw"), raw_col)
            selected.loc[cloud_override] = values.loc[cloud_override]
            source.loc[cloud_override] = selected_col
            reason.loc[cloud_override] = reason.loc[cloud_override].astype(str) + "+cloud_solar_raw_override"
            if bool(cloud_override_cfg.get("suppress_bias_uplift", True)):
                allow_bias_uplift.loc[cloud_override] = False
            if bool(cloud_override_cfg.get("suppress_hot_uplift", True)):
                allow_hot_uplift.loc[cloud_override] = False

    long_horizon_cfg = selector_cfg.get("long_horizon_peak_hot_month_correction", {}) or {}
    if horizon_policy_enabled and bool(long_horizon_cfg.get("enabled", False)):
        min_day = int(long_horizon_cfg.get("min_forecast_day", 8))
        max_day = int(long_horizon_cfg.get("max_forecast_day", 16))
        peak_hours = [int(h) for h in long_horizon_cfg.get("peak_hours", [14, 15, 16, 17, 18])]
        hot_hours = [int(h) for h in long_horizon_cfg.get("hot_hours", [16, 17, 18, 19, 20])]
        hot_min_maxtemp = float(long_horizon_cfg.get("hot_min_maxtemp_f", 90.0))
        month_values = out.get("Month")
        if month_values is None:
            dt_values = out.get("DT", pd.Series(pd.NaT, index=out.index))
            month_values = pd.to_datetime(dt_values, errors="coerce").dt.month
        numeric_month = pd.to_numeric(month_values, errors="coerce")
        hot_min_by_month = _configured_numeric_lookup(
            numeric_month,
            long_horizon_cfg.get("hot_min_maxtemp_f_by_month", {}),
            default=hot_min_maxtemp,
        )
        peak_corr = _configured_numeric_lookup(
            numeric_month,
            long_horizon_cfg.get("peak_month_offsets_mwh", {}),
            default=0.0,
        )
        hot_corr = _configured_numeric_lookup(
            numeric_month,
            long_horizon_cfg.get("hot_month_offsets_mwh", {}),
            default=0.0,
        )
        peak_cap = abs(float(long_horizon_cfg.get("peak_cap_mwh", 8.0)))
        hot_cap = abs(float(long_horizon_cfg.get("hot_cap_mwh", 15.0)))
        peak_corr = peak_corr.clip(lower=-peak_cap, upper=peak_cap)
        hot_corr = hot_corr.clip(lower=-hot_cap, upper=hot_cap)
        peak_day_scale = _configured_pair_numeric_lookup(
            numeric_month,
            day,
            long_horizon_cfg.get("peak_month_forecast_day_scales", {}),
            default=1.0,
        )
        hot_day_scale = _configured_pair_numeric_lookup(
            numeric_month,
            day,
            long_horizon_cfg.get("hot_month_forecast_day_scales", {}),
            default=1.0,
        )
        peak_corr = peak_corr * peak_day_scale
        hot_corr = hot_corr * hot_day_scale
        if "IsHoliday" in out.columns:
            is_holiday = pd.to_numeric(out["IsHoliday"], errors="coerce").fillna(0).ne(0)
        else:
            is_holiday = pd.Series(False, index=out.index, dtype=bool)
        if bool(long_horizon_cfg.get("peak_exclude_holidays", False)):
            peak_corr = peak_corr.where(~is_holiday, 0.0)
        else:
            peak_holiday_scale = float(long_horizon_cfg.get("peak_holiday_scale", 1.0))
            if abs(peak_holiday_scale - 1.0) > 1e-9:
                peak_corr = peak_corr.where(~is_holiday, peak_corr * peak_holiday_scale)
        if bool(long_horizon_cfg.get("hot_exclude_holidays", False)):
            hot_corr = hot_corr.where(~is_holiday, 0.0)
        else:
            holiday_scale = float(long_horizon_cfg.get("hot_holiday_scale", 1.0))
            if abs(holiday_scale - 1.0) > 1e-9:
                hot_corr = hot_corr.where(~is_holiday, hot_corr * holiday_scale)
        long_horizon = day.between(min_day, max_day)
        peak_month_mask = long_horizon & hour.isin(peak_hours) & peak_corr.ne(0.0)
        hot_month_mask = (
            long_horizon
            & hour.isin(hot_hours)
            & daily_max.ge(hot_min_by_month)
            & hot_corr.ne(0.0)
        )
        if peak_month_mask.any():
            selected.loc[peak_month_mask] = selected.loc[peak_month_mask] + peak_corr.loc[peak_month_mask]
            out.loc[peak_month_mask, "Long_Horizon_Peak_Month_Correction_MWH"] = peak_corr.loc[peak_month_mask]
            reason.loc[peak_month_mask] = reason.loc[peak_month_mask].astype(str) + "+long_horizon_peak_month_correction"
        if hot_month_mask.any():
            overlap = hot_month_mask & peak_month_mask
            # Hot rows get their own correction; this avoids stacking two tuned offsets
            # on the HE16-18 overlap where the signs can differ by month.
            if overlap.any():
                selected.loc[overlap] = selected.loc[overlap] - peak_corr.loc[overlap]
                out.loc[overlap, "Long_Horizon_Peak_Month_Correction_MWH"] = 0.0
            selected.loc[hot_month_mask] = selected.loc[hot_month_mask] + hot_corr.loc[hot_month_mask]
            out.loc[hot_month_mask, "Long_Horizon_Hot_Month_Correction_MWH"] = hot_corr.loc[hot_month_mask]
            reason.loc[hot_month_mask] = reason.loc[hot_month_mask].astype(str) + "+long_horizon_hot_month_correction"

    underforecast_uplift = float(selector_cfg.get("underforecast_bias_uplift_mwh", 0.0) or 0.0)
    if abs(underforecast_uplift) > 1e-9:
        selected.loc[allow_bias_uplift] = selected.loc[allow_bias_uplift] + underforecast_uplift
        reason.loc[allow_bias_uplift] = reason.loc[allow_bias_uplift].astype(str) + "+underforecast_bias_uplift"

    hot_uplift_cfg = selector_cfg.get("hot_peak_uplift", {}) or {}
    if bool(hot_uplift_cfg.get("enabled", False)):
        uplift_hours = [int(h) for h in hot_uplift_cfg.get("hours", [14, 15, 16, 17, 18, 19, 20])]
        min_day = int(hot_uplift_cfg.get("min_forecast_day", 2))
        max_day = int(hot_uplift_cfg.get("max_forecast_day", 16))
        min_maxtemp = float(hot_uplift_cfg.get("min_maxtemp_f", 90.0))
        max_uplift = float(hot_uplift_cfg.get("max_uplift_mwh", 5.0))
        base_uplift = float(hot_uplift_cfg.get("base_uplift_mwh", 0.0))
        horizon_bucket = _horizon_bucket_for_stage_selector(day)
        horizon_uplift = _configured_numeric_lookup(
            horizon_bucket,
            hot_uplift_cfg.get("horizon_uplift_mwh", {}),
            default=base_uplift,
        )
        temp_bucket = out.get("DailyMaxTempBucket", out.get("DailyMaxTempBin", pd.Series(np.nan, index=out.index)))
        temp_uplift = _configured_numeric_lookup(
            temp_bucket,
            hot_uplift_cfg.get("temp_bucket_uplift_mwh", {}),
            default=base_uplift,
        )
        uplift = np.maximum(horizon_uplift, temp_uplift).clip(lower=0.0, upper=max_uplift)
        uplift_mask = (
            day.between(min_day, max_day)
            & hour.isin(uplift_hours)
            & daily_max.ge(min_maxtemp)
            & uplift.gt(0.0)
            & allow_hot_uplift
        )
        if uplift_mask.any():
            selected.loc[uplift_mask] = selected.loc[uplift_mask] + uplift.loc[uplift_mask]
            reason.loc[uplift_mask] = reason.loc[uplift_mask].astype(str) + "+guarded_hot_peak_uplift"

    ultra_heat_cfg = selector_cfg.get("ultra_extreme_heat_wave_uplift", {}) or {}
    if horizon_policy_enabled and bool(ultra_heat_cfg.get("enabled", False)):
        min_day = int(ultra_heat_cfg.get("min_forecast_day", 2))
        max_day = int(ultra_heat_cfg.get("max_forecast_day", 7))
        min_maxtemp = float(ultra_heat_cfg.get("min_maxtemp_f", 115.0))
        ramp_full = float(ultra_heat_cfg.get("ramp_full_maxtemp_f", min_maxtemp))
        max_uplift = float(ultra_heat_cfg.get("max_uplift_mwh", 16.0))
        hour_uplift = _configured_numeric_lookup(
            hour,
            ultra_heat_cfg.get("hour_uplift_mwh", {}),
            default=0.0,
        ).clip(lower=0.0, upper=max_uplift)
        # Temperature ramp: 0 below min_maxtemp, full magnitude at/above ramp_full,
        # linear in between. With ramp_full == min_maxtemp this reduces to the prior step.
        if ramp_full > min_maxtemp:
            ramp_scale = ((daily_max - min_maxtemp) / (ramp_full - min_maxtemp)).clip(lower=0.0, upper=1.0)
        else:
            ramp_scale = daily_max.ge(min_maxtemp).astype(float)
        hour_uplift = (hour_uplift * ramp_scale).clip(lower=0.0, upper=max_uplift)
        uplift_mask = (
            day.between(min_day, max_day)
            & daily_max.ge(min_maxtemp)
            & hour_uplift.gt(0.0)
        )
        if uplift_mask.any():
            selected.loc[uplift_mask] = selected.loc[uplift_mask] + hour_uplift.loc[uplift_mask]
            reason.loc[uplift_mask] = reason.loc[uplift_mask].astype(str) + "+ultra_extreme_heat_wave_uplift"

    conditional_overrides = selector_cfg.get("conditional_stage_overrides", []) or []
    if horizon_policy_enabled and conditional_overrides:
        for raw_rule in conditional_overrides:
            rule = raw_rule or {}
            if not bool(rule.get("enabled", True)):
                continue
            override_mask = _conditional_stage_mask(rule)
            if not override_mask.any():
                continue
            selected_col, values = _stage_values(rule.get("stage", "raw"), raw_col)
            selected.loc[override_mask] = values.loc[override_mask]
            source.loc[override_mask] = selected_col
            name = str(rule.get("name", "conditional_stage_override")).strip() or "conditional_stage_override"
            reason.loc[override_mask] = reason.loc[override_mask].astype(str) + f"+conditional_stage_override:{name}"

    out["Stage_Selected_Forecast_MWH"] = selected.clip(lower=0.0)
    out["Stage_Selector_Source"] = source
    out["Stage_Selector_Reason"] = reason
    out[forecast_col] = out["Stage_Selected_Forecast_MWH"]
    if forecast_col == "Final_Backtest_Forecast_MWH":
        out["Final_Forecast_MWH"] = out[forecast_col]
    return out


def _apply_future_correction_chain(
    raw_future: pd.DataFrame,
    config: dict,
    targeted_meta_artifact: dict | None,
    lookup_bundle: dict | None,
    heat_lookup,
    warm_lookup: dict | None,
    cloud_solar_lookup: dict | None,
    recent_profile: dict | None,
) -> pd.DataFrame:
    cal_cfg = config.get("calibration", {})
    targeted_meta_cfg = cal_cfg.get("targeted_residual_meta", {}) or {}
    if bool(targeted_meta_cfg.get("enabled", True)):
        cal_future = apply_targeted_residual_meta_correction(
            future_df=raw_future,
            artifact=targeted_meta_artifact,
            config=config,
        )
    else:
        cal_future = raw_future.copy()
        cal_future["Targeted_Meta_Bias_Cal_MWH"] = 0.0
        cal_future["Targeted_Meta_SolarCloud_Cal_MWH"] = 0.0
        cal_future["Targeted_Meta_Cal_MWH"] = 0.0
        cal_future["Targeted_Meta_Source"] = "disabled"
        cal_future["Targeted_Meta_Adjusted_Forecast_MWH"] = cal_future["Raw_Forecast_MWH"].astype(float).clip(lower=0.0)

    calibration_base_col = (
        "Targeted_Meta_Adjusted_Forecast_MWH"
        if "Targeted_Meta_Adjusted_Forecast_MWH" in cal_future.columns
        else "Raw_Forecast_MWH"
    )
    if bool(cal_cfg.get("seasonal_enabled", True)):
        cal_future = apply_learned_calibration(
            cal_future,
            lookup_bundle,
            level_weights=cal_cfg.get("level_weights", {}),
            cap_mwh=float(cal_cfg.get("cap_mwh", 22.0)),
            base_col=calibration_base_col,
            hot_peak_cfg=cal_cfg.get("residual_calibration_hot_peak", {}),
        )
    else:
        cal_future["Residual_Cal_MWH"] = 0.0
        cal_future["Calibration_Level"] = "disabled"
        cal_future["Calibration_Matched_Levels"] = ""
        cal_future["Residual_Calibrated_Forecast_MWH"] = cal_future[calibration_base_col].astype(float).clip(lower=0.0)
        cal_future["Calibrated_Forecast_MWH"] = cal_future["Residual_Calibrated_Forecast_MWH"]

    if bool(cal_cfg.get("heat_peak_enabled", True)):
        cal_future = apply_heat_peak_calibration(
            future_df=cal_future,
            heat_lookup=heat_lookup,
            min_maxtemp_f=float(cal_cfg.get("heat_peak_min_maxtemp_f", 88.0)),
            hours=list(cal_cfg.get("heat_peak_hours", [14, 15, 16, 17, 18, 19, 20])),
        )
    if bool(cal_cfg.get("warm_ramp_enabled", True)):
        cal_future = apply_warm_ramp_correction(
            future_df=cal_future,
            warm_lookup=warm_lookup,
            min_maxtemp_f=float(cal_cfg.get("warm_ramp_min_maxtemp_f", 75.0)),
            max_maxtemp_f=float(cal_cfg.get("warm_ramp_max_maxtemp_f", 93.0)),
            hours=list(cal_cfg.get("warm_ramp_hours", [12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22])),
            cap_mwh=float(cal_cfg.get("warm_ramp_cap_mwh", 16.0)),
        )
    if bool(cal_cfg.get("cloud_solar_shape_enabled", True)):
        cal_future = apply_cloud_solar_shape_correction(
            future_df=cal_future,
            lookup_bundle=cloud_solar_lookup,
            hours=list(cal_cfg.get("cloud_solar_shape_hours", [10, 11, 12, 13, 14, 15, 16])),
            cap_mwh=float(cal_cfg.get("cloud_solar_shape_cap_mwh", 16.0)),
            min_loss_mw=float(cal_cfg.get("cloud_solar_shape_min_loss_mw", 1.25)),
            level_weights=cal_cfg.get("cloud_solar_shape_level_weights", {}),
            use_event_multiplier=bool(cal_cfg.get("cloud_solar_event_multiplier_enabled", True)),
        )
    if bool(((cal_cfg.get("peak_risk", {}) or {}).get("enabled", True))):
        cal_future = apply_peak_risk_correction(cal_future, config=config, base_col="Calibrated_Forecast_MWH")
    if bool((cal_cfg.get("recent_residual", {}) or {}).get("enabled", True)):
        cal_future = apply_recent_residual_correction(
            future_df=cal_future,
            profile=recent_profile,
            config=config,
            base_col="Calibrated_Forecast_MWH",
        )
    else:
        cal_future["Recent_Level_Correction_MWH"] = 0.0
        cal_future["Recent_Correction_Source"] = "disabled_or_empty"
        cal_future["AR_Residual_Correction_MWH"] = 0.0
        cal_future["AR_Residual_Phi"] = np.nan
        cal_future["AR_Residual_Latest_MWH"] = np.nan
        cal_future["AR_Residual_Source"] = "ar_disabled_or_empty"
        cal_future["OriginDay_State_Correction_MWH"] = 0.0
        cal_future["OriginDay_State_MWH"] = np.nan
        cal_future["OriginDay_Latest_Day_MWH"] = np.nan
        cal_future["OriginDay_State_Source"] = "origin_day_disabled_or_empty"
        cal_future["Recent_Corrected_Forecast_MWH"] = cal_future["Calibrated_Forecast_MWH"]
        cal_future["Final_Forecast_MWH"] = cal_future["Calibrated_Forecast_MWH"]
    cal_future = apply_operational_stage_selector(cal_future, config=config, forecast_col="Final_Forecast_MWH")
    return cal_future


def run_pipeline(
    config: dict,
    override_horizon_days: Optional[int] = None,
    progress_callback: ProgressCallback | None = None,
):
    """
    Orchestrates data loading, leakage-safe backtest, model training, recursive forecast,
    learned residual calibration, warm-ramp calibration, recent residual level correction,
    and residual-based uncertainty bands.
    """
    scenario_defs = list(scenario_definitions(config))
    _progress(progress_callback, "Starting forecast pipeline", total=21 + len(scenario_defs))

    _progress(progress_callback, "Loading hourly system history")
    load_df = load_hourly_system_mwh(config)
    _progress(progress_callback, "Loaded hourly system history", advance=1)

    _progress(progress_callback, "Fetching historical weather")
    hist_wx = fetch_historical_weather(config)
    _progress(progress_callback, "Fetched historical weather", advance=1)

    _progress(progress_callback, "Fetching forecast weather")
    fut_wx = fetch_forecast_weather(config)
    _progress(progress_callback, "Fetched forecast weather", advance=1)

    _progress(progress_callback, "Loading solar forecast")
    solar_df = load_solar_forecast(config)
    _progress(progress_callback, "Loaded solar forecast", advance=1)

    _progress(progress_callback, "Loading local weather")
    local_wx = load_local_station_weather(config)
    local_temp_lookup = pd.DataFrame()
    local_temp_matched = pd.DataFrame()
    if not local_wx.empty:
        local_temp_lookup, local_temp_matched = build_temperature_bias_lookup(hist_wx, local_wx, config)
        hist_wx = merge_local_station_temperature(hist_wx, local_wx)
        temp_cal_cfg = ((config.get("local_weather", {}) or {}).get("temperature_calibration", {}) or {})
        if bool(temp_cal_cfg.get("enabled", False)) and not local_temp_lookup.empty:
            hist_wx = apply_temperature_bias_calibration(hist_wx, local_temp_lookup, config)
            fut_wx = apply_temperature_bias_calibration(fut_wx, local_temp_lookup, config)
        
        if bool(temp_cal_cfg.get("dynamic_enabled", True)):
            fut_wx = apply_dynamic_temperature_calibration(fut_wx, hist_wx, config)
    _progress(progress_callback, "Processed local weather", advance=1)

    official_hourly_latest_dt = load_df["DT"].max() if "DT" in load_df.columns and not load_df.empty else pd.NaT
    _progress(progress_callback, "Loading five-minute load")
    five_min_load = load_five_min_system_load(config)
    five_min_cfg = config.get("five_min_load", {}) or {}
    five_min_hourly = build_hourly_load_from_five_min(
        five_min_load,
        timezone=str(config.get("project", {}).get("timezone") or "America/Los_Angeles"),
        min_intervals_per_hour=int(five_min_cfg.get("min_completed_intervals_per_hour", 10)),
    )
    if bool(five_min_cfg.get("use_as_recent_hourly_load", True)) and not five_min_hourly.empty:
        load_df = append_recent_five_min_hourly_load(
            load_df,
            five_min_hourly,
            replace_overlap_hours=int(five_min_cfg.get("replace_overlap_hours", 0)),
        )
        hist_wx = _extend_historical_weather_with_recent_forecast(
            hist_wx,
            fut_wx,
            latest_load_dt=load_df["DT"].max() if "DT" in load_df.columns and not load_df.empty else None,
        )
    _progress(progress_callback, "Processed five-minute load", advance=1)

    _progress(progress_callback, "Loading BTM solar capacity")
    btm_monthly = load_btm_monthly_capacity(config)
    _progress(progress_callback, "Loaded BTM solar capacity", advance=1)

    _progress(progress_callback, "Building intraday load features")
    intraday_features = build_intraday_load_feature_frame(five_min_load)
    _progress(progress_callback, "Built intraday load features", advance=1)

    _progress(progress_callback, "Building training frame")
    train_df = build_training_frame(load_df, hist_wx, btm_monthly, intraday_load_features=intraday_features, solar_df=solar_df)
    train_df.attrs["config"] = config
    _progress(progress_callback, "Built training frame", advance=1)

    features = [c for c in XGB_FEATURES if c in train_df.columns]
    ensemble_weights = _production_ensemble_weights(config)

    backtest_days = int(config["training"].get("backtest_days", 45))
    _progress(progress_callback, "Running leakage-safe backtest")
    backtest_raw_df = run_rolling_backtest(
        train_df=train_df,
        features=features,
        ensemble_weights=ensemble_weights,
        backtest_days=backtest_days,
        config=config,
    )
    _progress(progress_callback, "Completed leakage-safe backtest", advance=1)

    # Build correction lookups from the leakage-safe raw holdout residuals. These lookups are then
    # applied to both the future forecast and the backtest frame so stage-aware diagnostics can judge
    # the full V12.8 correction chain rather than stopping at raw/recent-only performance.
    cal_cfg = config.get("calibration", {})
    targeted_meta_cfg = cal_cfg.get("targeted_residual_meta", {}) or {}
    residual_band_lookup = None

    _progress(progress_callback, "Building correction artifacts")
    correction_artifacts = build_correction_artifacts(backtest_raw_df, config)
    lookup_bundle = correction_artifacts["lookup_bundle"]
    targeted_meta_artifact = correction_artifacts["targeted_meta_artifact"]
    heat_lookup = correction_artifacts["heat_lookup"]
    warm_lookup = correction_artifacts["warm_lookup"]
    cloud_solar_lookup = correction_artifacts["cloud_solar_lookup"]
    recent_profile = correction_artifacts["recent_profile"]
    operational_residual_artifact = correction_artifacts.get("operational_residual_artifact")
    _progress(progress_callback, "Built correction artifacts", advance=1)

    _progress(progress_callback, "Applying backtest correction chain")
    backtest_df = _apply_v126_correction_chain_to_frame(
        raw_df=backtest_raw_df,
        config=config,
        targeted_meta_artifact=targeted_meta_artifact,
        lookup_bundle=lookup_bundle,
        heat_lookup=heat_lookup,
        warm_lookup=warm_lookup,
        cloud_solar_lookup=cloud_solar_lookup,
        simulate_recent=True,
    )
    _progress(progress_callback, "Applied backtest correction chain", advance=1)

    # Train final models on all available data after the leakage-safe backtest is complete.
    _progress(progress_callback, "Training final tree models")
    xgb_model, lgb_model, xgb_feats = train_tree_models(train_df, features, config=config, stage_name="final full-history")
    _progress(progress_callback, "Trained final tree models", advance=1)

    _progress(progress_callback, "Training benchmark models")
    prophet_fit = train_prophet(train_df, DEFAULT_PROPHET_REGRESSORS, config=config) if prophet_enabled(config) else None
    prophet_model = prophet_fit.model if prophet_fit is not None else None
    prophet_features = prophet_fit.regressors if prophet_fit is not None else []
    catboost_model, catboost_features = train_catboost(train_df, xgb_feats, config=config) if catboost_enabled(config) else (None, xgb_feats)
    features = xgb_feats
    _progress(progress_callback, "Trained benchmark models", advance=1)

    five_min_cfg = config.get("five_min_load", {}) or {}
    _progress(progress_callback, "Building future feature frame")
    future_frame = build_forecast_frame(
        fut_wx,
        btm_monthly,
        intraday_load_features=intraday_features,
        max_intraday_carry_forward_hours=int(five_min_cfg.get("max_carry_forward_hours", 24)),
        use_future_intraday_load_features=bool(five_min_cfg.get("future_model_features_enabled", False)),
        solar_df=solar_df,
    )
    latest_hist_dt = train_df["DT"].max()
    future_frame = future_frame[future_frame["DT"] > latest_hist_dt].copy()

    if override_horizon_days is not None:
        end_dt = latest_hist_dt + pd.Timedelta(days=int(override_horizon_days))
        future_frame = future_frame[future_frame["DT"] <= end_dt].copy()
    future_frame = _trim_incomplete_future_weather(future_frame)
    _progress(progress_callback, "Built future feature frame", advance=1)

    historical_seed = train_df[["DT", "MWH"]].copy().sort_values("DT")
    _progress(progress_callback, "Running recursive forecast")
    raw_future = recursive_forecast(
        future_frame=future_frame,
        historical_seed=historical_seed,
        xgb_model=xgb_model,
        lgb_model=lgb_model,
        features=features,
        ensemble_weights=ensemble_weights,
        prophet_fit=prophet_fit,
        prophet_features=prophet_features,
        catboost_model=catboost_model,
    )
    _progress(progress_callback, "Completed recursive forecast", advance=1)

    _progress(progress_callback, "Applying future correction chain")
    if bool(targeted_meta_cfg.get("enabled", True)):
        cal_future = apply_targeted_residual_meta_correction(
            future_df=raw_future,
            artifact=targeted_meta_artifact,
            config=config,
        )
    else:
        cal_future = raw_future.copy()
        cal_future["Targeted_Meta_Bias_Cal_MWH"] = 0.0
        cal_future["Targeted_Meta_SolarCloud_Cal_MWH"] = 0.0
        cal_future["Targeted_Meta_Cal_MWH"] = 0.0
        cal_future["Targeted_Meta_Source"] = "disabled"
        cal_future["Targeted_Meta_Adjusted_Forecast_MWH"] = cal_future["Raw_Forecast_MWH"].astype(float).clip(lower=0.0)

    calibration_base_col = (
        "Targeted_Meta_Adjusted_Forecast_MWH"
        if "Targeted_Meta_Adjusted_Forecast_MWH" in cal_future.columns
        else "Raw_Forecast_MWH"
    )
    if bool(cal_cfg.get("seasonal_enabled", True)):
        cal_future = apply_learned_calibration(
            cal_future,
            lookup_bundle,
            level_weights=cal_cfg.get("level_weights", {}),
            cap_mwh=float(cal_cfg.get("cap_mwh", 22.0)),
            base_col=calibration_base_col,
            hot_peak_cfg=cal_cfg.get("residual_calibration_hot_peak", {}),
        )
    else:
        cal_future["Residual_Cal_MWH"] = 0.0
        cal_future["Calibration_Level"] = "disabled"
        cal_future["Calibration_Matched_Levels"] = ""
        cal_future["Residual_Calibrated_Forecast_MWH"] = cal_future[calibration_base_col].astype(float).clip(lower=0.0)
        cal_future["Calibrated_Forecast_MWH"] = cal_future["Residual_Calibrated_Forecast_MWH"]

    if bool(cal_cfg.get("heat_peak_enabled", True)):
        cal_future = apply_heat_peak_calibration(
            future_df=cal_future,
            heat_lookup=heat_lookup,
            min_maxtemp_f=float(cal_cfg.get("heat_peak_min_maxtemp_f", 88.0)),
            hours=list(cal_cfg.get("heat_peak_hours", [14, 15, 16, 17, 18, 19, 20])),
        )

    if bool(cal_cfg.get("warm_ramp_enabled", True)):
        cal_future = apply_warm_ramp_correction(
            future_df=cal_future,
            warm_lookup=warm_lookup,
            min_maxtemp_f=float(cal_cfg.get("warm_ramp_min_maxtemp_f", 75.0)),
            max_maxtemp_f=float(cal_cfg.get("warm_ramp_max_maxtemp_f", 93.0)),
            hours=list(cal_cfg.get("warm_ramp_hours", [12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22])),
            cap_mwh=float(cal_cfg.get("warm_ramp_cap_mwh", 16.0)),
        )

    if bool(cal_cfg.get("cloud_solar_shape_enabled", True)):
        cal_future = apply_cloud_solar_shape_correction(
            future_df=cal_future,
            lookup_bundle=cloud_solar_lookup,
            hours=list(cal_cfg.get("cloud_solar_shape_hours", [10, 11, 12, 13, 14, 15, 16])),
            cap_mwh=float(cal_cfg.get("cloud_solar_shape_cap_mwh", 16.0)),
            min_loss_mw=float(cal_cfg.get("cloud_solar_shape_min_loss_mw", 1.25)),
            level_weights=cal_cfg.get("cloud_solar_shape_level_weights", {}),
            use_event_multiplier=bool(cal_cfg.get("cloud_solar_event_multiplier_enabled", True)),
        )

    if bool(((cal_cfg.get("peak_risk", {}) or {}).get("enabled", True))):
        cal_future = apply_peak_risk_correction(
            cal_future,
            config=config,
            base_col="Calibrated_Forecast_MWH",
        )

    if bool((cal_cfg.get("recent_residual", {}) or {}).get("enabled", True)):
        cal_future = apply_recent_residual_correction(
            future_df=cal_future,
            profile=recent_profile,
            config=config,
            base_col="Calibrated_Forecast_MWH",
        )
    else:
        cal_future["Recent_Level_Correction_MWH"] = 0.0
        cal_future["Recent_Correction_Source"] = "disabled_or_empty"
        cal_future["AR_Residual_Correction_MWH"] = 0.0
        cal_future["AR_Residual_Phi"] = np.nan
        cal_future["AR_Residual_Latest_MWH"] = np.nan
        cal_future["AR_Residual_Source"] = "ar_disabled_or_empty"
        cal_future["OriginDay_State_Correction_MWH"] = 0.0
        cal_future["OriginDay_State_MWH"] = np.nan
        cal_future["OriginDay_Latest_Day_MWH"] = np.nan
        cal_future["OriginDay_State_Source"] = "origin_day_disabled_or_empty"
        cal_future["Recent_Corrected_Forecast_MWH"] = cal_future["Calibrated_Forecast_MWH"]
        cal_future["Final_Forecast_MWH"] = cal_future["Calibrated_Forecast_MWH"]
    cal_future = apply_operational_stage_selector(cal_future, config=config, forecast_col="Final_Forecast_MWH")
    _progress(progress_callback, "Applied future correction chain", advance=1)

    scenario_columns: list[str] = []
    for scenario in scenario_defs:
        name = str(scenario.get("name", "scenario"))
        _progress(progress_callback, f"Running weather scenario: {name}")
        scenario_frame = make_weather_scenario_frame(future_frame, scenario)
        scenario_raw = recursive_forecast(
            future_frame=scenario_frame,
            historical_seed=historical_seed,
            xgb_model=xgb_model,
            lgb_model=lgb_model,
            features=features,
            ensemble_weights=ensemble_weights,
            prophet_fit=prophet_fit,
            prophet_features=prophet_features,
            catboost_model=catboost_model,
        )
        scenario_cal = _apply_future_correction_chain(
            raw_future=scenario_raw,
            config=config,
            targeted_meta_artifact=targeted_meta_artifact,
            lookup_bundle=lookup_bundle,
            heat_lookup=heat_lookup,
            warm_lookup=warm_lookup,
            cloud_solar_lookup=cloud_solar_lookup,
            recent_profile=recent_profile,
        )
        col = scenario_column_name(name)
        cal_future[col] = pd.to_numeric(scenario_cal["Final_Forecast_MWH"], errors="coerce").to_numpy()
        scenario_columns.append(col)
        _progress(progress_callback, f"Completed weather scenario: {name}", advance=1)

    _progress(progress_callback, "Summarizing weather scenarios")
    cal_future = apply_weather_scenario_delta_caps(cal_future, scenario_columns, config=config)
    cal_future = add_scenario_summary_columns(cal_future, scenario_columns)
    _progress(progress_callback, "Summarized weather scenarios", advance=1)

    # V12.9: lead-aware weather-uncertainty peak hedge. Lifts the hot/peak point
    # forecast using the warmer/cooler scenario re-predictions, scaled by the
    # daily-max temperature forecast error at each lead. Applied after stage
    # selection and before band construction so bands build on the hedged point.
    _progress(progress_callback, "Applying production forecast guards")
    cal_future = apply_weather_robustness_hedge(
        cal_future,
        config=config,
        base_col="Final_Forecast_MWH",
        also_update_cols=("Stage_Selected_Forecast_MWH",),
    )
    cal_future = apply_focused_scorecard_guard(
        cal_future,
        config=config,
        forecast_col="Final_Forecast_MWH",
        also_update_cols=("Stage_Selected_Forecast_MWH",),
    )
    cal_future = apply_operational_residual_learner(
        cal_future,
        operational_residual_artifact,
        config,
        forecast_col="Final_Forecast_MWH",
        also_update_cols=("Stage_Selected_Forecast_MWH", "Calibrated_Forecast_MWH"),
        evaluation_mode="future_shadow",
    )
    _progress(progress_callback, "Applied production forecast guards", advance=1)

    _progress(progress_callback, "Building forecast bands")
    bands_cfg = config.get("bands", {})
    band_basis_df = backtest_df.copy()
    if "Final_Residual_MWH" in band_basis_df.columns:
        band_basis_df["Residual_MWH"] = band_basis_df["Final_Residual_MWH"]
    residual_band_lookup = build_residual_band_lookup(
        band_basis_df,
        shrink_floor_mwh=float(bands_cfg.get("band_floor_mwh", 5.0)),
    ) if bool(bands_cfg.get("residual_based", True)) else None

    final_future = apply_bands(
        cal_future,
        percent_band=float(bands_cfg.get("default_percent_band", 0.08)),
        floor_mwh=float(bands_cfg.get("band_floor_mwh", 5.0)),
        residual_lookup=residual_band_lookup,
        band_scale=float(bands_cfg.get("band_scale", 1.0)),
        weather_input_risk=bands_cfg.get("weather_input_risk", {}),
        hot_bucket_band_floor=bands_cfg.get("hot_bucket_band_floor", {}),
    )
    final_future = apply_conformal_weather_bands(
        final_future,
        config=config,
        output_dir=config.get("project", {}).get("output_dir", "forecast_outputs"),
    )
    _progress(progress_callback, "Built forecast bands", advance=1)

    _progress(progress_callback, "Building dashboard display frame")
    display_df = build_display_df(train_df, final_future)
    _progress(progress_callback, "Built dashboard display frame", advance=1)

    _progress(progress_callback, "Building diagnostics")
    diagnostics = build_diagnostics_bundle(
        backtest_df=backtest_df,
        forecast_display_df=display_df,
        features=features,
        xgb_model=xgb_model,
        lgb_model=lgb_model,
        prophet_model=prophet_model,
        prophet_features=prophet_features,
        catboost_model=catboost_model,
        calibration_lookup_bundle=lookup_bundle,
        heat_peak_lookup=heat_lookup,
        warm_ramp_lookup=warm_lookup,
        cloud_solar_shape_lookup=cloud_solar_lookup,
        recent_residual_profile=recent_profile,
        residual_band_lookup=residual_band_lookup,
        config=config,
    )
    diagnostics["operational_residual_learner_summary"] = operational_residual_learner_summary(
        backtest_df,
        operational_residual_artifact,
        config,
    )
    diagnostics["local_weather_temperature_bias_summary"] = local_temperature_bias_summary(local_temp_matched, local_temp_lookup)
    diagnostics["local_weather_temperature_bias_lookup"] = local_temp_lookup
    diagnostics["forecast_weather_used"] = fut_wx.copy() if isinstance(fut_wx, pd.DataFrame) else pd.DataFrame()
    diagnostics["forecast_weather_snapshot_summary"] = {
        "source": str(getattr(fut_wx, "attrs", {}).get("weather_source", "")),
        "snapshot_path": str(getattr(fut_wx, "attrs", {}).get("weather_snapshot_path", "")),
        "cache_path": str(getattr(fut_wx, "attrs", {}).get("weather_cache_path", "")),
        "rows": int(len(fut_wx)) if isinstance(fut_wx, pd.DataFrame) else 0,
        "first_dt": str(fut_wx["DT"].min()) if isinstance(fut_wx, pd.DataFrame) and "DT" in fut_wx.columns and not fut_wx.empty else None,
        "last_dt": str(fut_wx["DT"].max()) if isinstance(fut_wx, pd.DataFrame) and "DT" in fut_wx.columns and not fut_wx.empty else None,
    }
    diagnostics["weather_stress_test_summary"] = build_weather_stress_summary(final_future, scenario_columns)
    diagnostics["five_min_hourly_load_debug"] = five_min_hourly
    if not five_min_hourly.empty:
        added = five_min_hourly[pd.to_datetime(five_min_hourly["DT"], errors="coerce") > official_hourly_latest_dt].copy()
        diagnostics["five_min_hourly_load_summary"] = {
            "five_min_rows": int(len(five_min_load)),
            "completed_hourly_rows": int(len(five_min_hourly)),
            "latest_completed_five_min_hour": str(five_min_hourly["DT"].max()),
            "official_hourly_latest_dt": str(official_hourly_latest_dt),
            "appended_recent_hourly_rows": int(len(added)),
            "appended_latest_dt": str(added["DT"].max()) if not added.empty else None,
        }
    _progress(progress_callback, "Built diagnostics", advance=1)

    return {
        "historical_fit_df": train_df,
        "future": {
            "raw": raw_future,
            "calibrated": final_future,
            "display": display_df,
        },
        "backtest": backtest_df,
        "features": features,
        "prophet_features": prophet_features,
        "catboost_features": catboost_features if catboost_model is not None else [],
        "backtest_metrics": backtest_df.attrs.get("metrics", {}) if backtest_df is not None else {},
        "diagnostics": diagnostics,
    }



def _apply_pre_cloud_correction_chain_to_frame(
    raw_df: pd.DataFrame,
    config: dict,
    targeted_meta_artifact: dict | None,
    lookup_bundle: dict | None,
    heat_lookup,
    warm_lookup: dict | None,
) -> pd.DataFrame:
    """Apply correction stages up through warm-ramp only.

    This is used to train the V12.8 cloud/solar lookup on residuals that remain after the broader
    calibration layers, which makes the overcast-midday correction less blunt.
    """
    if raw_df is None or raw_df.empty:
        return pd.DataFrame()
    cal_cfg = config.get("calibration", {})
    if bool((cal_cfg.get("targeted_residual_meta", {}) or {}).get("enabled", True)):
        out = apply_targeted_residual_meta_correction(raw_df, targeted_meta_artifact, config)
    else:
        out = raw_df.copy()
        out["Targeted_Meta_Adjusted_Forecast_MWH"] = out["Raw_Forecast_MWH"].astype(float).clip(lower=0.0)
    base_col = "Targeted_Meta_Adjusted_Forecast_MWH"
    if bool(cal_cfg.get("seasonal_enabled", True)):
        out = apply_learned_calibration(
            out,
            lookup_bundle,
            level_weights=cal_cfg.get("level_weights", {}),
            cap_mwh=float(cal_cfg.get("cap_mwh", 22.0)),
            base_col=base_col,
            hot_peak_cfg=cal_cfg.get("residual_calibration_hot_peak", {}),
        )
    else:
        out["Residual_Cal_MWH"] = 0.0
        out["Residual_Calibrated_Forecast_MWH"] = out[base_col].astype(float).clip(lower=0.0)
        out["Calibrated_Forecast_MWH"] = out["Residual_Calibrated_Forecast_MWH"]

    if bool(cal_cfg.get("heat_peak_enabled", True)):
        out = apply_heat_peak_calibration(
            future_df=out,
            heat_lookup=heat_lookup,
            min_maxtemp_f=float(cal_cfg.get("heat_peak_min_maxtemp_f", 88.0)),
            hours=list(cal_cfg.get("heat_peak_hours", [14, 15, 16, 17, 18, 19, 20])),
        )
    if bool(cal_cfg.get("warm_ramp_enabled", True)):
        out = apply_warm_ramp_correction(
            future_df=out,
            warm_lookup=warm_lookup,
            min_maxtemp_f=float(cal_cfg.get("warm_ramp_min_maxtemp_f", 75.0)),
            max_maxtemp_f=float(cal_cfg.get("warm_ramp_max_maxtemp_f", 93.0)),
            hours=list(cal_cfg.get("warm_ramp_hours", [12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22])),
            cap_mwh=float(cal_cfg.get("warm_ramp_cap_mwh", 16.0)),
        )
    return out


def _apply_v126_correction_chain_to_frame(
    raw_df: pd.DataFrame,
    config: dict,
    targeted_meta_artifact: dict | None,
    lookup_bundle: dict | None,
    heat_lookup,
    warm_lookup: dict | None,
    cloud_solar_lookup: dict | None,
    simulate_recent: bool = True,
    apply_auto_residual: bool = True,
) -> pd.DataFrame:
    """Apply the same V12.8 correction chain to a backtest-like frame.

    This fixes the V12.5 diagnostics gap where the future forecast included cloud/solar shape
    correction but the backtest stage metrics did not.  The recent residual step is simulated
    using only earlier rows in the holdout and uses the residual left after the event-correction
    chain, not the original raw residual.
    """
    if raw_df is None or raw_df.empty:
        return pd.DataFrame()
    cal_cfg = config.get("calibration", {})
    if bool((cal_cfg.get("targeted_residual_meta", {}) or {}).get("enabled", True)):
        out = apply_targeted_residual_meta_correction(raw_df, targeted_meta_artifact, config)
    else:
        out = raw_df.copy()
        out["Targeted_Meta_Adjusted_Forecast_MWH"] = out["Raw_Forecast_MWH"].astype(float).clip(lower=0.0)
    base_col = "Targeted_Meta_Adjusted_Forecast_MWH"

    if bool(cal_cfg.get("seasonal_enabled", True)):
        out = apply_learned_calibration(
            out,
            lookup_bundle,
            level_weights=cal_cfg.get("level_weights", {}),
            cap_mwh=float(cal_cfg.get("cap_mwh", 22.0)),
            base_col=base_col,
            hot_peak_cfg=cal_cfg.get("residual_calibration_hot_peak", {}),
        )
    else:
        out["Residual_Cal_MWH"] = 0.0
        out["Residual_Calibrated_Forecast_MWH"] = out[base_col].astype(float).clip(lower=0.0)
        out["Calibrated_Forecast_MWH"] = out["Residual_Calibrated_Forecast_MWH"]

    if bool(cal_cfg.get("heat_peak_enabled", True)):
        out = apply_heat_peak_calibration(
            future_df=out,
            heat_lookup=heat_lookup,
            min_maxtemp_f=float(cal_cfg.get("heat_peak_min_maxtemp_f", 88.0)),
            hours=list(cal_cfg.get("heat_peak_hours", [14, 15, 16, 17, 18, 19, 20])),
        )
    if bool(cal_cfg.get("warm_ramp_enabled", True)):
        out = apply_warm_ramp_correction(
            future_df=out,
            warm_lookup=warm_lookup,
            min_maxtemp_f=float(cal_cfg.get("warm_ramp_min_maxtemp_f", 75.0)),
            max_maxtemp_f=float(cal_cfg.get("warm_ramp_max_maxtemp_f", 93.0)),
            hours=list(cal_cfg.get("warm_ramp_hours", [12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22])),
            cap_mwh=float(cal_cfg.get("warm_ramp_cap_mwh", 16.0)),
        )
    if bool(cal_cfg.get("cloud_solar_shape_enabled", True)):
        out = apply_cloud_solar_shape_correction(
            future_df=out,
            lookup_bundle=cloud_solar_lookup,
            hours=list(cal_cfg.get("cloud_solar_shape_hours", [10, 11, 12, 13, 14, 15, 16])),
            cap_mwh=float(cal_cfg.get("cloud_solar_shape_cap_mwh", 16.0)),
            min_loss_mw=float(cal_cfg.get("cloud_solar_shape_min_loss_mw", 1.25)),
            level_weights=cal_cfg.get("cloud_solar_shape_level_weights", {}),
            use_event_multiplier=bool(cal_cfg.get("cloud_solar_event_multiplier_enabled", True)),
        )
    if bool(((cal_cfg.get("peak_risk", {}) or {}).get("enabled", True))):
        out = apply_peak_risk_correction(out, config=config, base_col="Calibrated_Forecast_MWH")

    if simulate_recent and bool((cal_cfg.get("recent_residual", {}) or {}).get("enabled", True)):
        out = simulate_recent_residual_correction_backtest(out, config=config, base_col="Calibrated_Forecast_MWH")
    else:
        out["Recent_Level_Correction_MWH"] = 0.0
        out["Recent_Correction_Source"] = "disabled_or_empty"
        out["AR_Residual_Correction_MWH"] = 0.0
        out["AR_Residual_Phi"] = np.nan
        out["AR_Residual_Latest_MWH"] = np.nan
        out["AR_Residual_Source"] = "ar_disabled_or_empty"
        out["OriginDay_State_Correction_MWH"] = 0.0
        out["OriginDay_State_MWH"] = np.nan
        out["OriginDay_Latest_Day_MWH"] = np.nan
        out["OriginDay_State_Source"] = "origin_day_disabled_or_empty"
        out["Pre_Recent_Forecast_MWH"] = out["Calibrated_Forecast_MWH"]
        out["Recent_Corrected_Forecast_MWH"] = out["Calibrated_Forecast_MWH"]
        out["Final_Backtest_Forecast_MWH"] = out["Calibrated_Forecast_MWH"]
        out["Final_Forecast_MWH"] = out["Final_Backtest_Forecast_MWH"]

    out = apply_operational_stage_selector(out, config=config, forecast_col="Final_Backtest_Forecast_MWH")
    out = apply_focused_scorecard_guard(
        out,
        config=config,
        forecast_col="Final_Backtest_Forecast_MWH",
        also_update_cols=("Stage_Selected_Forecast_MWH",),
    )
    final_col = "Final_Backtest_Forecast_MWH" if "Final_Backtest_Forecast_MWH" in out.columns else "Calibrated_Forecast_MWH"
    out["Final_Residual_MWH"] = pd.to_numeric(out["Actual_MWH"], errors="coerce") - pd.to_numeric(out[final_col], errors="coerce")
    out["Final_AbsError_MWH"] = out["Final_Residual_MWH"].abs()
    out["Final_APE"] = np.where(
        pd.to_numeric(out["Actual_MWH"], errors="coerce").abs() > 1e-9,
        out["Final_AbsError_MWH"] / pd.to_numeric(out["Actual_MWH"], errors="coerce").abs() * 100.0,
        np.nan,
    )
    if apply_auto_residual:
        out = simulate_operational_residual_learner_backtest(
            out,
            config,
            forecast_col=final_col,
            force_shadow=None,
        )
    return out

def build_display_df(train_df: pd.DataFrame, future_df: pd.DataFrame) -> pd.DataFrame:
    """Combine recent historical actuals with future forecast + diagnostics for dashboard display."""
    hist = train_df.copy()
    hist = hist[[c for c in [
        "DT", "MWH", "Load_Source", "FiveMin_Interval_Count", "FiveMin_Hourly_Last_MW", "FiveMin_Hourly_Range_MW",
        "Temperature", "Temperature_DailyMax", "BTM_Solar_Proxy_MW", "CloudCover_Norm", "Humidity_Norm",
        "WindSpeed_Mph", "WindDirection_Deg", "WindDirection_Available_Flag", "Westerly_Flow_Mph",
        "Westerly_Flow_Flag", "WindRamp_1Hr_Mph", "WindRamp_3Hr_Mph", "WindRamp_Next1Hr_Mph",
        "WindRamp_Next3Hr_Mph", "WesterlyFlow_Ramp_1Hr_Mph", "WesterlyFlow_Ramp_3Hr_Mph",
        "WesterlyFlow_Next1Hr_Ramp_Mph", "WesterlyFlow_Next3Hr_Ramp_Mph",
        "Temperature_Drop_From_DailyMax_F", "TempDrop_1Hr_F", "TempDrop_2Hr_F", "TempDrop_3Hr_F",
        "TempDrop_Next1Hr_F", "TempDrop_Next2Hr_F", "TempDrop_Next3Hr_F",
        "IsPostPeakEvening18to23", "ClearHotEvening_Flag", "ClearVeryHotEvening_Flag",
        "ClearHotEvening_x_TempDropFromDailyMax", "ClearHotEvening_x_ForecastDropNext3Hr",
        "ClearHotEvening_x_WesterlyFlow", "ClearHotEvening_x_WesterlyFlowRamp",
        "DeltaBreeze_Westerly_Flow_Flag", "DeltaBreeze_EveningWindRamp_Flag",
        "DeltaBreeze_Cooling_Flag", "DeltaBreeze_Cooling_Signal",
        "DeltaBreeze_CoolingNoDirection_Signal", "DeltaBreeze_ClearHotEvening_Signal",
        "Load_Decay_1Hr_MWH", "Load_Decay_2Hr_MWH",
        "Lag1_Minus_SameHourYesterday_MWH", "Lag1_Minus_SameHour7DayMean_MWH",
        "PostPeak_LoadDecay_1Hr_MWH", "PostPeak_LoadDecay_2Hr_MWH",
        "PostPeak_LoadDecay_VsSameHourYesterday_MWH", "PostPeak_LoadDecay_VsSameHour7DayMean_MWH",
        "ClearHotEvening_LoadDecay_Vs7Day_MWH", "DeltaBreeze_PostPeak_LoadDecay_Signal",
        "PrecipIn",
    ] if c in hist.columns]]
    hist.rename(columns={"MWH": "Actual"}, inplace=True)
    hist["Forecast"] = pd.NA
    hist["Raw_Forecast_MWH"] = pd.NA
    hist["Upper_Band"] = pd.NA
    hist["Lower_Band"] = pd.NA
    hist["Band"] = pd.NA
    hist["P10_Forecast_MWH"] = pd.NA
    hist["P50_Forecast_MWH"] = pd.NA
    hist["P90_Forecast_MWH"] = pd.NA
    hist["Forecast_Low_MWH"] = pd.NA
    hist["Forecast_Expected_MWH"] = pd.NA
    hist["Forecast_High_MWH"] = pd.NA
    hist["Weather_Input_Risk_Multiplier"] = pd.NA
    hist["Weather_Input_Risk_Reason"] = pd.NA
    hist["Weather_Input_Risk_Class"] = pd.NA
    hist["Production_Caution_Flag"] = pd.NA
    hist["Production_Caution_Reason"] = pd.NA
    hist["Production_Confidence_Label"] = pd.NA
    hist["Production_Risk_Code"] = pd.NA
    hist["Calibration_Level"] = "actual"

    forecast_col = "Final_Forecast_MWH" if "Final_Forecast_MWH" in future_df.columns else "Calibrated_Forecast_MWH"
    fut_cols = [
        "DT", forecast_col, "Calibrated_Forecast_MWH", "Targeted_Meta_Adjusted_Forecast_MWH", "Residual_Calibrated_Forecast_MWH", "Heat_Adjusted_Forecast_MWH",
        "Warm_Ramp_Adjusted_Forecast_MWH", "Cloud_Solar_Adjusted_Forecast_MWH", "Peak_Risk_Adjusted_Forecast_MWH", "Recent_Corrected_Forecast_MWH", "Raw_Forecast_MWH",
        "XGB_Pred_MWH", "LGB_Pred_MWH", "CatBoost_Pred_MWH", "Prophet_Pred_MWH", "Prophet_Lower_MWH", "Prophet_Upper_MWH",
        "Targeted_Meta_Bias_Cal_MWH", "Targeted_Meta_SolarCloud_Cal_MWH", "Targeted_Meta_Cal_MWH", "Residual_Cal_MWH", "Heat_Peak_Cal_MWH", "Warm_Ramp_Cal_MWH", "Cloud_Solar_Shape_Cal_MWH", "Cloud_Solar_Shape_Raw_Cal_MWH", "Peak_Risk_Cal_MWH", "Recent_Level_Correction_MWH",
        "AR_Residual_Correction_MWH", "AR_Residual_Phi", "AR_Residual_Latest_MWH",
        "OriginDay_State_Correction_MWH", "OriginDay_State_MWH", "OriginDay_Latest_Day_MWH",
        "Band", "Upper_Band", "Lower_Band", "P10_Forecast_MWH", "P50_Forecast_MWH", "P90_Forecast_MWH",
        "Forecast_Low_MWH", "Forecast_Expected_MWH", "Forecast_High_MWH",
        "Band_Method", "Quantile_Method", "Operational_Horizon_Label", "Weather_Input_Risk_Multiplier",
        "Weather_Input_Risk_Reason", "Weather_Input_Risk_Class", "Production_Caution_Flag",
        "Production_Caution_Reason", "Production_Confidence_Label", "Production_Risk_Code",
        "Pre_Conformal_Band_MWH", "Conformal_Weather_Band_MWH", "Conformal_Weather_Source",
        "WeatherScenario_Min_P50_MWH", "WeatherScenario_Max_P50_MWH", "WeatherScenario_Spread_MWH",
        "WeatherScenario_HalfSpread_MWH", "WeatherScenario_MaxAbsDelta_MWH", "WeatherScenario_Cap_Applied",
        "Weather_Robustness_Hedge_MWH", "Weather_Robustness_Hedge_Source",
        "Weather_Robustness_Jensen_MWH", "Weather_Robustness_Upper_MWH",
        "Weather_Robustness_Warmer_Delta_MWH", "Weather_Robustness_Temp_Sigma_F",
        "Weather_Robustness_Temp_Bias_Damping", "Weather_Robustness_Gate",
        "Pre_Focused_Guard_Forecast_MWH", "Post_Focused_Guard_Forecast_MWH",
        "Focused_Guard_Applied_Flag",
        "Focused_Scorecard_Guard_MWH", "Focused_Scorecard_Guard_Source",
        "Auto_Residual_Model_Version", "Auto_Residual_Shadow_Mode",
        "Auto_Residual_Production_Scope",
        "Auto_Residual_Base_Forecast_MWH", "Auto_Residual_Correction_MWH",
        "Auto_Residual_Adjusted_Forecast_MWH", "Auto_Residual_Correction_Applied_Flag",
        "Auto_Residual_Source", "Auto_Residual_Evaluation_Mode",
        "Auto_Residual_Residual_MWH", "Auto_Residual_AbsError_MWH",
        "Auto_Residual_Delta_AbsError_MWH",
        "Auto_Residual_Full_Shadow_Correction_MWH",
        "Auto_Residual_Full_Shadow_Adjusted_Forecast_MWH",
        "Auto_Residual_Full_Shadow_Correction_Applied_Flag",
        "Auto_Residual_Full_Shadow_Source",
        "Auto_Residual_Full_Shadow_Residual_MWH",
        "Auto_Residual_Full_Shadow_AbsError_MWH",
        "Auto_Residual_Full_Shadow_Delta_AbsError_MWH",
        "Calibration_Level", "Calibration_Matched_Levels", "Targeted_Meta_Source", "Warm_Ramp_Correction_Source", "Cloud_Solar_Correction_Source", "Peak_Risk_Source", "Recent_Correction_Source", "AR_Residual_Source", "OriginDay_State_Source",
        "Long_Horizon_Peak_Month_Correction_MWH", "Long_Horizon_Hot_Month_Correction_MWH",
        "Stage_Selected_Forecast_MWH", "Stage_Selector_Source", "Stage_Selector_Reason",
        "Temperature", "Temperature_DailyMax", "Humidity_Norm", "CloudCover_Norm",
        "WindSpeed_Mph", "WindDirection_Deg", "WindDirection_Available_Flag", "Westerly_Flow_Mph",
        "Westerly_Flow_Flag", "WindRamp_1Hr_Mph", "WindRamp_3Hr_Mph", "WindRamp_Next1Hr_Mph",
        "WindRamp_Next3Hr_Mph", "WesterlyFlow_Ramp_1Hr_Mph", "WesterlyFlow_Ramp_3Hr_Mph",
        "WesterlyFlow_Next1Hr_Ramp_Mph", "WesterlyFlow_Next3Hr_Ramp_Mph",
        "Temperature_Drop_From_DailyMax_F", "TempDrop_1Hr_F", "TempDrop_2Hr_F", "TempDrop_3Hr_F",
        "TempDrop_Next1Hr_F", "TempDrop_Next2Hr_F", "TempDrop_Next3Hr_F",
        "IsPostPeakEvening18to23", "ClearHotEvening_Flag", "ClearVeryHotEvening_Flag",
        "ClearHotEvening_x_TempDropFromDailyMax", "ClearHotEvening_x_ForecastDropNext3Hr",
        "ClearHotEvening_x_WesterlyFlow", "ClearHotEvening_x_WesterlyFlowRamp",
        "DeltaBreeze_Westerly_Flow_Flag", "DeltaBreeze_EveningWindRamp_Flag",
        "DeltaBreeze_Cooling_Flag", "DeltaBreeze_Cooling_Signal",
        "DeltaBreeze_CoolingNoDirection_Signal", "DeltaBreeze_ClearHotEvening_Signal",
        "Load_Decay_1Hr_MWH", "Load_Decay_2Hr_MWH",
        "Lag1_Minus_SameHourYesterday_MWH", "Lag1_Minus_SameHour7DayMean_MWH",
        "PostPeak_LoadDecay_1Hr_MWH", "PostPeak_LoadDecay_2Hr_MWH",
        "PostPeak_LoadDecay_VsSameHourYesterday_MWH", "PostPeak_LoadDecay_VsSameHour7DayMean_MWH",
        "ClearHotEvening_LoadDecay_Vs7Day_MWH", "DeltaBreeze_PostPeak_LoadDecay_Signal",
        "PrecipIn",
        "Load_Source", "FiveMin_Interval_Count", "FiveMin_Hourly_Last_MW", "FiveMin_Hourly_Range_MW",
        "FiveMin_Load_Available", "FiveMin_Data_Age_Hours", "FiveMin_PrevHour_Avg_MW", "FiveMin_PrevHour_Max_MW",
        "FiveMin_PrevHour_Min_MW", "FiveMin_PrevHour_Last_MW", "FiveMin_PrevHour_Range_MW", "FiveMin_PrevHour_Ramp_MW",
        "FiveMin_PrevHour_Count", "FiveMin_Ramp_15Min_MW", "FiveMin_Ramp_30Min_MW", "FiveMin_Ramp_60Min_MW",
        "BTM_Solar_Proxy_MW", "BTM_Solar_Loss_From_ClearSky_MW", "Midday_Overcast_Solar_Loss_MW", "CloudSolarEventClass", "CloudSolarEventMultiplier", "CloudSolarBaseBucket",
        "Daily_BTM_Solar_Proxy_Max_MW", "Daily_BTM_Solar_Loss_Max_MW", "Solar_Irradiance", "ClearSky_Index",
    ]
    scenario_cols = [c for c in future_df.columns if c.startswith("WeatherScenario_") and c.endswith("_P50_MWH")]
    selected_cols = list(dict.fromkeys(c for c in fut_cols + scenario_cols if c in future_df.columns))
    fut = future_df[selected_cols].copy()
    fut.rename(columns={forecast_col: "Forecast"}, inplace=True)
    if "Forecast" not in fut.columns and "Calibrated_Forecast_MWH" in fut.columns:
        fut["Forecast"] = fut["Calibrated_Forecast_MWH"]
    fut["Actual"] = pd.NA

    out = pd.concat([hist, fut], ignore_index=True, sort=False)
    out.sort_values("DT", inplace=True)
    return out.reset_index(drop=True)
