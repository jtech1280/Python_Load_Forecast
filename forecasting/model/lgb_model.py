from __future__ import annotations

import json
from pathlib import Path
from typing import Any
import inspect

import numpy as np
import pandas as pd

try:
    import lightgbm as lgb
except Exception as exc:
    raise RuntimeError(
        "LightGBM is required. Install with `pip install lightgbm`."
    ) from exc

from forecasting.model.xgb_model import (
    DEFAULT_FEATURES,
    build_sample_weights,
    hot_peak_scope_mask,
    make_asymmetric_hot_peak_objective,
)
from forecasting.utils.performance import resolve_n_jobs
from forecasting.model.monotonic_constraints import lgb_monotone_param

_LAST_LGB_TRAINING_INFO: dict[str, Any] = {}


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


def _early_stopping_callback(rounds: int):
    try:
        if hasattr(lgb, "early_stopping"):
            return lgb.early_stopping(stopping_rounds=int(rounds), verbose=False)
        if hasattr(lgb, "callback") and hasattr(lgb.callback, "early_stopping"):
            return lgb.callback.early_stopping(
                stopping_rounds=int(rounds), verbose=False
            )
    except Exception:
        return None
    return None


def _base_lgb_params(config: dict | None) -> dict[str, Any]:
    p = _cfg(config, "model", "lgb", default={}) or {}
    return {
        "objective": "regression",
        "n_estimators": int(p.get("n_estimators", 1200)),
        "learning_rate": float(p.get("learning_rate", 0.025)),
        "num_leaves": int(p.get("num_leaves", 96)),
        "min_child_samples": int(p.get("min_child_samples", 25)),
        "subsample": float(p.get("subsample", 0.85)),
        "colsample_bytree": float(p.get("colsample_bytree", 0.85)),
        "reg_lambda": float(p.get("reg_lambda", 2.0)),
        "reg_alpha": float(p.get("reg_alpha", 0.05)),
        "random_state": int(p.get("random_state", 42)),
        "n_jobs": int(resolve_n_jobs(config, "lgb", default=p.get("n_jobs", -1))),
        "verbose": -1,
    }


def _lgb_gpu_requested(config: dict | None) -> bool:
    hw = _cfg(config, "hardware", default={}) or {}
    p = _cfg(config, "model", "lgb", default={}) or {}
    # Keep LightGBM GPU opt-in because Windows wheels frequently lack usable OpenCL/CUDA support.
    return _as_bool(hw.get("use_lgb_gpu", p.get("use_gpu", False)), default=False)


def _lgb_attempts(config: dict | None) -> list[tuple[str, dict[str, Any]]]:
    hw = _cfg(config, "hardware", default={}) or {}
    p = _cfg(config, "model", "lgb", default={}) or {}
    base = _base_lgb_params(config)
    fallback_to_cpu = _as_bool(
        hw.get("fallback_to_cpu", p.get("fallback_to_cpu", True)), default=True
    )

    cpu_params = dict(base)
    attempts: list[tuple[str, dict[str, Any]]] = []

    if _lgb_gpu_requested(config):
        gpu_params = dict(base)
        gpu_params["device_type"] = str(
            p.get("device_type", hw.get("lgb_device_type", "gpu"))
        )
        gpu_platform_id = p.get("gpu_platform_id", hw.get("lgb_gpu_platform_id", None))
        gpu_device_id = p.get("gpu_device_id", hw.get("lgb_gpu_device_id", None))
        if gpu_platform_id is not None:
            gpu_params["gpu_platform_id"] = int(gpu_platform_id)
        if gpu_device_id is not None:
            gpu_params["gpu_device_id"] = int(gpu_device_id)
        attempts.append(("gpu", gpu_params))

    if not _lgb_gpu_requested(config) or fallback_to_cpu:
        attempts.append(("cpu", cpu_params))
    return attempts


def make_lgb_model(
    config: dict | None, params_override: dict[str, Any] | None = None
) -> lgb.LGBMRegressor:
    params = params_override or _lgb_attempts(config)[0][1]
    return lgb.LGBMRegressor(**params)


def get_last_lgb_training_info() -> dict[str, Any]:
    return dict(_LAST_LGB_TRAINING_INFO)


def write_lgb_training_info(config: dict | None) -> None:
    try:
        out_dir = Path(
            _cfg(config, "project", "output_dir", default="forecast_outputs")
        )
        out_dir.mkdir(parents=True, exist_ok=True)
        (out_dir / "lgb_training_backend.json").write_text(
            json.dumps(get_last_lgb_training_info(), indent=2, default=str),
            encoding="utf-8",
        )
    except Exception:
        pass


def train_lgb(
    df: pd.DataFrame, features: list[str] | None = None, config: dict | None = None
):
    global _LAST_LGB_TRAINING_INFO

    cfg = config or {}
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

    mono_vector = lgb_monotone_param(features, cfg)

    errors: list[str] = []
    attempts = _lgb_attempts(cfg)
    if mono_vector is not None:
        for _, attempt_params in attempts:
            attempt_params["monotone_constraints"] = mono_vector

    asym_cfg = _cfg(cfg, "model", "asymmetric_loss", default={}) or {}
    if bool(asym_cfg.get("enabled", False)):
        hot_peak_mask = hot_peak_scope_mask(
            train_df.reset_index(drop=True), cfg
        ).to_numpy()
        objective = make_asymmetric_hot_peak_objective(
            hot_peak_mask,
            under_forecast_penalty=float(
                asym_cfg.get("under_forecast_penalty_multiplier", 2.0)
            ),
        )
        for _, attempt_params in attempts:
            attempt_params["objective"] = objective

    for backend_name, params in attempts:
        model = make_lgb_model(cfg, params_override=params)
        try:
            if backend_name == "gpu":
                print("Training LightGBM with GPU backend...")
            elif errors:
                print("Training LightGBM on CPU after GPU fallback...")

            fit_kwargs: dict[str, Any] = {"sample_weight": sample_weight}
            callbacks = []
            if (
                X_valid is not None
                and y_valid is not None
                and len(X_valid)
                and len(y_valid)
            ):
                fit_kwargs["eval_set"] = [(X_valid, y_valid)]
                try:
                    fit_sig_params = inspect.signature(model.fit).parameters
                except Exception:
                    fit_sig_params = {}
                if "eval_metric" in fit_sig_params:
                    fit_kwargs["eval_metric"] = str(es_cfg.get("metric", "mae"))
                if "eval_sample_weight" in fit_sig_params:
                    fit_kwargs["eval_sample_weight"] = (
                        [valid_weight] if valid_weight is not None else None
                    )
                cb = _early_stopping_callback(int(es_cfg.get("rounds", 75)))
                if cb is not None:
                    callbacks.append(cb)
            if callbacks:
                fit_kwargs["callbacks"] = callbacks

            model.fit(X, y, **fit_kwargs)
            _LAST_LGB_TRAINING_INFO = {
                "requested_gpu": _lgb_gpu_requested(cfg),
                "selected_backend": backend_name,
                "lightgbm_version": getattr(lgb, "__version__", "unknown"),
                "params": params,
                "failed_attempts": errors,
                "early_stopping": {
                    "enabled": bool(X_valid is not None and y_valid is not None),
                    "metric": str(es_cfg.get("metric", "mae")),
                    "rounds": int(es_cfg.get("rounds", 75)),
                    "validation_days": float(es_cfg.get("validation_days", 45)),
                    "best_iteration": getattr(model, "best_iteration_", None),
                },
                "n_rows": int(len(df)),
                "n_features": int(len(features)),
            }
            if backend_name == "gpu":
                print("LightGBM GPU training succeeded.")
            return model, features
        except Exception as exc:
            msg = f"{backend_name}: {type(exc).__name__}: {exc}"
            errors.append(msg)
            print(f"WARNING: LightGBM backend '{backend_name}' failed. Details: {exc}")

    _LAST_LGB_TRAINING_INFO = {
        "requested_gpu": _lgb_gpu_requested(cfg),
        "selected_backend": None,
        "lightgbm_version": getattr(lgb, "__version__", "unknown"),
        "failed_attempts": errors,
        "n_rows": int(len(df)),
        "n_features": int(len(features)),
    }
    raise RuntimeError("All LightGBM training attempts failed:\n" + "\n".join(errors))
