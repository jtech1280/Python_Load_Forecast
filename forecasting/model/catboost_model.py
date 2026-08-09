from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from forecasting.model.xgb_model import DEFAULT_FEATURES, build_sample_weights
from forecasting.utils.performance import resolve_n_jobs
from forecasting.model.monotonic_constraints import catboost_monotone_param

_LAST_CATBOOST_TRAINING_INFO: dict[str, Any] = {}


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


def catboost_enabled(config: dict | None) -> bool:
    return _as_bool(
        _cfg(config, "model", "catboost", "enabled", default=False), default=False
    )


def catboost_blend_enabled(config: dict | None) -> bool:
    return catboost_enabled(config) and _as_bool(
        _cfg(config, "model", "catboost", "blend_into_production", default=False),
        default=False,
    )


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


def _import_catboost():
    try:
        from catboost import CatBoostRegressor

        return CatBoostRegressor, None
    except Exception as exc:
        return None, exc


def _base_params(config: dict | None) -> dict[str, Any]:
    p = _cfg(config, "model", "catboost", default={}) or {}
    return {
        "iterations": int(p.get("iterations", 1400)),
        "depth": int(p.get("depth", 8)),
        "learning_rate": float(p.get("learning_rate", 0.025)),
        "l2_leaf_reg": float(p.get("l2_leaf_reg", 4.0)),
        "loss_function": str(p.get("loss_function", "RMSE")),
        "random_seed": int(p.get("random_seed", 42)),
        "thread_count": int(
            resolve_n_jobs(config, "catboost", default=p.get("thread_count", -1))
        ),
        "verbose": False,
        "allow_writing_files": False,
    }


def _gpu_requested(config: dict | None) -> bool:
    hw_use_gpu = _as_bool(
        _cfg(config, "hardware", "use_gpu", default=True), default=True
    )
    p = _cfg(config, "model", "catboost", default={}) or {}
    task = str(p.get("task_type", "GPU")).upper()
    return hw_use_gpu and task == "GPU"


def _attempts(config: dict | None) -> list[tuple[str, dict[str, Any]]]:
    p = _cfg(config, "model", "catboost", default={}) or {}
    hw = _cfg(config, "hardware", default={}) or {}
    fallback_to_cpu = _as_bool(hw.get("fallback_to_cpu", True), default=True)
    base = _base_params(config)
    attempts: list[tuple[str, dict[str, Any]]] = []
    if _gpu_requested(config):
        gp = dict(base)
        gp["task_type"] = "GPU"
        if p.get("devices") is not None:
            gp["devices"] = str(p.get("devices"))
        attempts.append(("gpu", gp))
    cp = dict(base)
    cp["task_type"] = "CPU"
    if not _gpu_requested(config) or fallback_to_cpu:
        attempts.append(("cpu", cp))
    return attempts


def train_catboost(
    df: pd.DataFrame, features: list[str] | None = None, config: dict | None = None
):
    """Train CatBoost as an optional benchmark. Returns (model, features) or (None, features)."""
    global _LAST_CATBOOST_TRAINING_INFO
    cfg = config or {}
    features = _available_features(df, features or DEFAULT_FEATURES)
    CatBoostRegressor, import_error = _import_catboost()
    if CatBoostRegressor is None:
        _LAST_CATBOOST_TRAINING_INFO = {
            "enabled": catboost_enabled(cfg),
            "selected_backend": None,
            "skipped": True,
            "reason": f"catboost is not installed or could not be imported: {import_error}",
            "n_rows": int(len(df)),
            "n_features": int(len(features)),
        }
        print(
            "WARNING: CatBoost benchmark skipped because catboost is not installed or failed to import."
        )
        return None, features

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
    if valid_df is not None and not valid_df.empty:
        X_valid = _prepare_x(valid_df, features)
        y_valid = pd.to_numeric(valid_df["MWH"], errors="coerce").astype(float)
    has_eval_set = (
        X_valid is not None and y_valid is not None and len(X_valid) and len(y_valid)
    )

    mono_vector = catboost_monotone_param(features, cfg)

    errors: list[str] = []
    attempts = _attempts(cfg)
    if mono_vector is not None:
        for _, attempt_params in attempts:
            attempt_params["monotone_constraints"] = mono_vector
    if has_eval_set:
        for _, attempt_params in attempts:
            attempt_params["eval_metric"] = (
                str(es_cfg.get("metric", "mae")).strip().upper()
            )

    for backend_name, params in attempts:
        try:
            print(f"Training CatBoost benchmark with {backend_name.upper()} backend...")
            model = CatBoostRegressor(**params)
            fit_kwargs: dict[str, Any] = {"sample_weight": sample_weight}
            if has_eval_set:
                fit_kwargs["eval_set"] = (X_valid, y_valid)
                fit_kwargs["early_stopping_rounds"] = int(es_cfg.get("rounds", 75))
                fit_kwargs["use_best_model"] = True
                fit_kwargs["verbose"] = False
            model.fit(X, y, **fit_kwargs)
            best_iteration = None
            if has_eval_set:
                try:
                    raw_best = model.get_best_iteration()
                    best_iteration = int(raw_best) if raw_best is not None else None
                except Exception:
                    best_iteration = None
            _LAST_CATBOOST_TRAINING_INFO = {
                "enabled": catboost_enabled(cfg),
                "requested_gpu": _gpu_requested(cfg),
                "selected_backend": backend_name,
                "params": params,
                "failed_attempts": errors,
                "early_stopping": {
                    "enabled": bool(has_eval_set),
                    "metric": str(es_cfg.get("metric", "mae")),
                    "rounds": int(es_cfg.get("rounds", 75)),
                    "validation_days": float(es_cfg.get("validation_days", 45)),
                    "best_iteration": best_iteration,
                    "tree_count": (
                        int(model.tree_count_)
                        if hasattr(model, "tree_count_")
                        else None
                    ),
                },
                "n_rows": int(len(df)),
                "n_features": int(len(features)),
            }
            return model, features
        except Exception as exc:
            msg = f"{backend_name}: {type(exc).__name__}: {exc}"
            errors.append(msg)
            print(f"WARNING: CatBoost backend '{backend_name}' failed. Details: {exc}")

    _LAST_CATBOOST_TRAINING_INFO = {
        "enabled": catboost_enabled(cfg),
        "requested_gpu": _gpu_requested(cfg),
        "selected_backend": None,
        "skipped": True,
        "failed_attempts": errors,
        "n_rows": int(len(df)),
        "n_features": int(len(features)),
    }
    print("WARNING: CatBoost benchmark skipped after all training attempts failed.")
    return None, features


def get_last_catboost_training_info() -> dict[str, Any]:
    return dict(_LAST_CATBOOST_TRAINING_INFO)


def write_catboost_training_info(config: dict | None) -> None:
    try:
        out_dir = Path(
            _cfg(config, "project", "output_dir", default="forecast_outputs")
        )
        out_dir.mkdir(parents=True, exist_ok=True)
        (out_dir / "catboost_training_backend.json").write_text(
            json.dumps(get_last_catboost_training_info(), indent=2, default=str),
            encoding="utf-8",
        )
    except Exception:
        pass
