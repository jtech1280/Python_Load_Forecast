from __future__ import annotations

import json
import warnings
from pathlib import Path
from typing import Any
import inspect

import numpy as np
import pandas as pd
import xgboost as xgb

from forecasting.features.intraday_load_features import INTRADAY_LOAD_FEATURES
from forecasting.utils.performance import resolve_n_jobs

BASE_FEATURES = [
    # Weather response. Tree ensembles are robust to correlated split candidates, and
    # replay showed that pruning this family removed useful hot-peak signal. Keep the
    # broader heat response here; Prophet keeps its pruned regressor set separately.
    "Temperature", "Temperature_DailyMax", "Temperature_DailyMin", "Temperature_DailyMean",
    "CDD", "HDD", "Daily_CDD", "Daily_HDD", "Cooling_Stress",
    "Temp_Squared", "CDD_Squared", "HDD_Squared",
    "Extreme_Heat_80", "Extreme_Heat_85", "Extreme_Heat_90", "Extreme_Heat_95", "Extreme_Heat_100",
    "Temp_Bin", "DailyMaxTempBin",
    "HeatIndexF", "HeatIndex_CDD", "DailyMax_x_PeakHour",
    "PriorDay_DailyMaxTemp", "PriorDay_DailyMinTemp",
    "DailyMaxTemp_Ramp_1Day", "DailyMinTemp_Ramp_1Day",
    "DailyMaxTemp_2DayMean", "DailyMaxTemp_3DayMean",
    "DailyMinTemp_2DayMean", "DailyMinTemp_3DayMean", "DailyMeanTemp_3DayMean",
    "ConsecutiveHotDays90", "ConsecutiveVeryHotDays95", "ConsecutiveExtremeHotDays100",
    "HeatPersistenceStress90", "HeatPersistenceStress95", "DailyMax3DayMean_x_PeakHour",
    "OvernightHeatStress", "OvernightHeatStress_x_PeakHour",
    "Humidity_Norm", "CloudCover_Norm", "WindSpeed_Mph", "PrecipIn", "Is_Raining",
    "Wind_x_Temp", "Rain_x_IsWeekend", "Hot_Humid_Stress",
    # Time/calendar/load-shape
    "Hour", "DOW", "Month", "DayOfYear", "WeekOfYear",
    "HourSin", "HourCos", "DOWSin", "DOWCos", "MonthSin", "MonthCos", "DayOfYearSin", "DayOfYearCos",
    "IsWeekend", "IsBusinessDay", "IsHoliday", "IsPreHoliday", "IsPostHoliday", "IsHolidayAdjacent",
    "IsMonday", "IsFriday", "IsSummerSeason", "IsWinterSeason", "IsOffPeak", "IsOnPeak", "IsSuperPeak", "IsLikelySystemPeakHour",
    # Solar / BTM. Restore the broader family for the controlled replay; the heat
    # family remains pruned above.
    "Nameplate_MW", "Capacity_Ratio_To_Current", "Impact_Cap_MW", "Solar_Irradiance",
    "BTM_Solar_Proxy_MW", "Daily_BTM_Solar_Proxy_Total_MWh", "Daily_BTM_Solar_Proxy_Max_MW",
    "BTM_x_GHI",
    "BTM_x_Cloud", "Solar_Midday_Flag", "Solar_Evening_Ramp_Flag", "BTM_Evening_Ramp_Impact",
    "Solar_Hour_Shape", "Cloud_x_Solar_Hour", "Solar_Season_Factor", "ClearSky_Index",
    "ClearSky_GHI_Proxy_Wm2", "BTM_ClearSky_Proxy_MW",
    "BTM_Solar_Cloud_Adjusted_MW", "BTM_Solar_Loss_From_ClearSky_MW",
    "Cloud_x_GHI", "Cloud_x_ClearSky_GHI", "Daily_BTM_ClearSky_Max_MW",
    "Daily_BTM_Solar_Loss_MWh", "Daily_BTM_Solar_Loss_Max_MW", "Midday_Overcast_Solar_Loss_MW",
    "BTM_Midday_Impact", "Solar_Ramp_Down_1hr", "Solar_Ramp_Down_2hr", "Solar_Ramp_Up_1hr",
    "Humidity_x_Temp",
    # Load memory
    "MWH_Lag1", "MWH_Lag2", "MWH_Lag3", "MWH_Lag24", "MWH_Lag48", "MWH_Lag72", "MWH_Lag168",
    "MWH_Rolling3", "MWH_Rolling6", "MWH_Rolling12", "MWH_Rolling24", "MWH_Rolling48", "MWH_Rolling168",
    "MWH_Rolling24Std", "MWH_SameHour7DayMean",
] + INTRADAY_LOAD_FEATURES

DEFAULT_FEATURES = BASE_FEATURES
_LAST_XGB_TRAINING_INFO: dict[str, Any] = {}


def _cfg(config: dict | None, *keys, default=None):
    cur = config or {}
    for k in keys:
        if not isinstance(cur, dict) or k not in cur:
            return default
        cur = cur[k]
    return cur


def _as_bool(value, default: bool = False) -> bool:
    if value is None:
        return default
    if isinstance(value, bool):
        return value
    if isinstance(value, (int, float)):
        return bool(value)
    if isinstance(value, str):
        return value.strip().lower() in {"1", "true", "yes", "y", "on"}
    return default


def _available_features(df: pd.DataFrame, features: list[str]) -> list[str]:
    return [c for c in features if c in df.columns]


def _prepare_x(df: pd.DataFrame, features: list[str]) -> pd.DataFrame:
    X = df[features].apply(pd.to_numeric, errors="coerce")
    return X.replace([np.inf, -np.inf], np.nan).fillna(0.0).astype(float)


def _split_train_validation(
    df: pd.DataFrame,
    validation_days: float,
    min_train_rows: int,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    if df is None or df.empty:
        return df, pd.DataFrame()
    if "DT" not in df.columns:
        return df, pd.DataFrame()
    df = df.copy().sort_values("DT").reset_index(drop=True)
    if validation_days <= 0:
        return df, pd.DataFrame()
    cutoff = df["DT"].max() - pd.Timedelta(days=float(validation_days))
    train = df[df["DT"] < cutoff].copy()
    valid = df[df["DT"] >= cutoff].copy()
    if len(train) < int(min_train_rows) or len(valid) < 24:
        return df, pd.DataFrame()
    return train, valid


def _early_stopping_cfg(config: dict | None) -> dict[str, Any]:
    cfg = config or {}
    es = _cfg(cfg, "model", "early_stopping", default={}) or {}
    return {
        "enabled": _as_bool(es.get("enabled", True), default=True),
        "validation_days": float(es.get("validation_days", 45)),
        "min_train_rows": int(es.get("min_train_rows", 2000)),
        "rounds": int(es.get("rounds", 75)),
        "metric": str(es.get("metric", "mae")).strip().lower() or "mae",
    }


def build_sample_weights(df: pd.DataFrame, config: dict | None = None) -> np.ndarray:
    """Emphasize recent observations and summer/peak/high-load hours without overfitting one event."""
    sw_cfg = _cfg(config, "model", "sample_weight", default={}) or {}
    y = pd.to_numeric(df["MWH"], errors="coerce").astype(float)
    weights = np.ones(len(df), dtype=float)

    q90 = y.quantile(float(sw_cfg.get("peak_q90", 0.90)))
    q95 = y.quantile(float(sw_cfg.get("peak_q95", 0.95)))
    weights[y >= q90] *= float(sw_cfg.get("peak_q90_weight", 1.8))
    weights[y >= q95] *= float(sw_cfg.get("peak_q95_weight", 3.0))

    if "Temperature_DailyMax" in df.columns:
        hot_peak_mask = (
            (pd.to_numeric(df["Temperature_DailyMax"], errors="coerce") >= float(sw_cfg.get("hot_day_min_f", 90.0)))
            & (pd.to_numeric(df.get("IsLikelySystemPeakHour", 0), errors="coerce").fillna(0).astype(int) == 1)
        )
        weights[hot_peak_mask.to_numpy()] *= float(sw_cfg.get("hot_peak_weight", 1.7))

    recency_end_weight = float(sw_cfg.get("recency_end_weight", 1.35))
    if recency_end_weight > 1 and len(df) > 1:
        ranks = np.linspace(1.0, recency_end_weight, len(df))
        weights *= ranks

    return np.clip(weights, 0.25, float(sw_cfg.get("max_weight", 8.0)))


def _base_xgb_params(config: dict | None) -> dict[str, Any]:
    p = _cfg(config, "model", "xgb", default={}) or {}
    return {
        "objective": "reg:squarederror",
        "n_estimators": int(p.get("n_estimators", 1400)),
        "learning_rate": float(p.get("learning_rate", 0.025)),
        "max_depth": int(p.get("max_depth", 8)),
        "min_child_weight": float(p.get("min_child_weight", 3)),
        "subsample": float(p.get("subsample", 0.85)),
        "colsample_bytree": float(p.get("colsample_bytree", 0.85)),
        "reg_lambda": float(p.get("reg_lambda", 2.0)),
        "reg_alpha": float(p.get("reg_alpha", 0.05)),
        "gamma": float(p.get("gamma", 0.0)),
        "random_state": int(p.get("random_state", 42)),
        "n_jobs": int(resolve_n_jobs(config, "xgb", default=p.get("n_jobs", -1))),
    }



def _xgb_major_version() -> int:
    try:
        return int(str(getattr(xgb, "__version__", "0")).split(".")[0])
    except Exception:
        return 0

def _gpu_requested(config: dict | None) -> bool:
    hw = _cfg(config, "hardware", default={}) or {}
    p = _cfg(config, "model", "xgb", default={}) or {}
    explicit = hw.get("use_gpu", None)
    if explicit is not None:
        return _as_bool(explicit, default=False)
    return str(p.get("device", "")).lower() in {"cuda", "gpu"} or str(p.get("tree_method", "")).lower() == "gpu_hist"


def _xgb_attempts(config: dict | None) -> list[tuple[str, dict[str, Any]]]:
    """Return ordered XGBoost parameter attempts.

    Modern XGBoost uses tree_method='hist' + device='cuda'. Older releases often need
    tree_method='gpu_hist'. We try both before CPU when fallback is enabled.
    """
    p = _cfg(config, "model", "xgb", default={}) or {}
    hw = _cfg(config, "hardware", default={}) or {}
    gpu_api = str(hw.get("xgb_gpu_api", p.get("gpu_api", "auto"))).lower()
    fallback_to_cpu = _as_bool(hw.get("fallback_to_cpu", p.get("fallback_to_cpu", True)), default=True)
    base = _base_xgb_params(config)

    cpu_params = dict(base)
    cpu_params.update({"tree_method": "hist", "device": "cpu"})

    if not _gpu_requested(config):
        return [("cpu", cpu_params)]

    modern = dict(base)
    modern.update({"tree_method": "hist", "device": "cuda"})

    legacy = dict(base)
    # Legacy GPU mode intentionally omits 'device' because older XGBoost versions may reject it.
    legacy.update({"tree_method": "gpu_hist"})

    attempts: list[tuple[str, dict[str, Any]]] = []
    major = _xgb_major_version()

    # Avoid false "GPU success" on old XGBoost versions where the modern `device` parameter
    # can be ignored. XGBoost 2.x+ should use hist + device=cuda. 1.x usually needs gpu_hist.
    if gpu_api == "auto":
        if major >= 2:
            attempts.extend([("cuda", modern), ("gpu_hist", legacy)])
        else:
            attempts.extend([("gpu_hist", legacy), ("cuda", modern)])
    else:
        if gpu_api in {"cuda", "modern", "xgboost_cuda"}:
            attempts.append(("cuda", modern))
        if gpu_api in {"legacy", "gpu_hist"}:
            attempts.append(("gpu_hist", legacy))

    if fallback_to_cpu:
        attempts.append(("cpu_fallback", cpu_params))
    return attempts


def make_xgb_model(config: dict | None, params_override: dict[str, Any] | None = None) -> xgb.XGBRegressor:
    params = params_override or _xgb_attempts(config)[0][1]
    return xgb.XGBRegressor(**params)


def get_last_xgb_training_info() -> dict[str, Any]:
    return dict(_LAST_XGB_TRAINING_INFO)


def write_xgb_training_info(config: dict | None) -> None:
    """Optionally write the most recent XGB training backend info for review."""
    try:
        out_dir = Path(_cfg(config, "project", "output_dir", default="forecast_outputs"))
        out_dir.mkdir(parents=True, exist_ok=True)
        (out_dir / "xgb_training_backend.json").write_text(
            json.dumps(get_last_xgb_training_info(), indent=2, default=str),
            encoding="utf-8",
        )
    except Exception:
        pass


def train_xgb(df: pd.DataFrame, features: list[str] | None = None, config: dict | None = None):
    global _LAST_XGB_TRAINING_INFO

    cfg = config or df.attrs.get("config", {}) or {}
    features = _available_features(df, features or DEFAULT_FEATURES)
    es_cfg = _early_stopping_cfg(cfg)
    train_df, valid_df = (df, pd.DataFrame())
    if bool(es_cfg.get("enabled", True)):
        train_df, valid_df = _split_train_validation(
            df=df,
            validation_days=float(es_cfg.get("validation_days", 45)),
            min_train_rows=int(es_cfg.get("min_train_rows", 2000)),
        )

    X = _prepare_x(train_df, features)
    y = pd.to_numeric(train_df["MWH"], errors="coerce").astype(float)
    sample_weight = build_sample_weights(train_df.reset_index(drop=True), cfg)

    X_valid = None
    y_valid = None
    valid_weight = None
    if valid_df is not None and not valid_df.empty:
        X_valid = _prepare_x(valid_df, features)
        y_valid = pd.to_numeric(valid_df["MWH"], errors="coerce").astype(float)
        valid_weight = build_sample_weights(valid_df.reset_index(drop=True), cfg)

    errors: list[str] = []
    attempts = _xgb_attempts(cfg)
    for backend_name, params in attempts:
        model = make_xgb_model(cfg, params_override=params)
        try:
            if backend_name in {"cuda", "gpu_hist"}:
                print(f"Training XGBoost with GPU backend '{backend_name}'...")
            elif backend_name == "cpu_fallback":
                print("Training XGBoost on CPU after GPU fallback...")
            with warnings.catch_warnings(record=True) as caught_warnings:
                warnings.simplefilter("always")
                fit_kwargs: dict[str, Any] = {"sample_weight": sample_weight}
                if X_valid is not None and y_valid is not None and len(X_valid) and len(y_valid):
                    fit_kwargs["eval_set"] = [(X_valid, y_valid)]
                    try:
                        params = inspect.signature(model.fit).parameters
                    except Exception:
                        params = {}
                    if "sample_weight_eval_set" in params:
                        fit_kwargs["sample_weight_eval_set"] = [valid_weight] if valid_weight is not None else None
                    if "eval_metric" in params:
                        fit_kwargs["eval_metric"] = str(es_cfg.get("metric", "mae"))
                    if "early_stopping_rounds" in params:
                        fit_kwargs["early_stopping_rounds"] = int(es_cfg.get("rounds", 75))
                    if "verbose" in params:
                        fit_kwargs["verbose"] = False
                model.fit(X, y, **fit_kwargs)

            booster_attrs = {}
            booster_config = {}
            actual_device = None
            try:
                booster = model.get_booster()
                booster_attrs = booster.attributes()
                booster_config = json.loads(booster.save_config())
                actual_device = (booster_config.get("learner", {}).get("generic_param", {}) or {}).get("device")
            except Exception:
                booster_attrs = {}
                booster_config = {}

            if backend_name in {"cuda", "gpu_hist"}:
                warning_text = "\n".join(str(w.message) for w in caught_warnings)
                gpu_warning_markers = [
                    "No visible GPU",
                    "Device is changed from GPU to CPU",
                    "not compiled with GPU",
                    "GPU is not enabled",
                ]
                if any(marker.lower() in warning_text.lower() for marker in gpu_warning_markers):
                    raise RuntimeError("XGBoost GPU request fell back or failed according to warnings: " + warning_text[:1000])
                if actual_device is not None and str(actual_device).lower().startswith("cpu"):
                    raise RuntimeError(
                        f"XGBoost accepted GPU parameters but trained on device={actual_device!r}; retrying fallback."
                    )

            actual_backend = backend_name
            _LAST_XGB_TRAINING_INFO = {
                "requested_gpu": _gpu_requested(cfg),
                "selected_backend": actual_backend,
                "xgboost_version": getattr(xgb, "__version__", "unknown"),
                "params": {k: v for k, v in params.items() if k not in {"objective"}},
                "failed_attempts": errors,
                "early_stopping": {
                    "enabled": bool(X_valid is not None and y_valid is not None),
                    "metric": str(es_cfg.get("metric", "mae")),
                    "rounds": int(es_cfg.get("rounds", 75)),
                    "validation_days": float(es_cfg.get("validation_days", 45)),
                    "best_iteration": getattr(model, "best_iteration", None),
                    "best_score": getattr(model, "best_score", None),
                },
                "booster_attributes": booster_attrs,
                "actual_device": actual_device,
                "n_rows": int(len(df)),
                "n_features": int(len(features)),
            }
            if actual_backend in {"cuda", "gpu_hist"}:
                print(f"XGBoost GPU training succeeded using backend '{actual_backend}'.")
            return model, features
        except Exception as exc:
            msg = f"{backend_name}: {type(exc).__name__}: {exc}"
            errors.append(msg)
            if backend_name in {"cuda", "gpu_hist"}:
                print(f"WARNING: XGBoost GPU backend '{backend_name}' failed. Details: {exc}")
            else:
                print(f"ERROR: XGBoost CPU training failed. Details: {exc}")

    _LAST_XGB_TRAINING_INFO = {
        "requested_gpu": _gpu_requested(cfg),
        "selected_backend": None,
        "xgboost_version": getattr(xgb, "__version__", "unknown"),
        "failed_attempts": errors,
        "n_rows": int(len(df)),
        "n_features": int(len(features)),
    }
    raise RuntimeError("All XGBoost training attempts failed:\n" + "\n".join(errors))
