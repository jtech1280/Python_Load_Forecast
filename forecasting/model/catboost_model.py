from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from forecasting.model.xgb_model import DEFAULT_FEATURES, build_sample_weights
from forecasting.utils.performance import resolve_n_jobs

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
    return _as_bool(_cfg(config, "model", "catboost", "enabled", default=False), default=False)


def catboost_blend_enabled(config: dict | None) -> bool:
    return catboost_enabled(config) and _as_bool(_cfg(config, "model", "catboost", "blend_into_production", default=False), default=False)


def _available_features(df: pd.DataFrame, features: list[str]) -> list[str]:
    return [c for c in features if c in df.columns]


def _prepare_x(df: pd.DataFrame, features: list[str]) -> pd.DataFrame:
    X = df[features].apply(pd.to_numeric, errors="coerce")
    return X.replace([np.inf, -np.inf], np.nan).fillna(0.0).astype(float)


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
        "thread_count": int(resolve_n_jobs(config, "catboost", default=p.get("thread_count", -1))),
        "verbose": False,
        "allow_writing_files": False,
    }


def _gpu_requested(config: dict | None) -> bool:
    hw_use_gpu = _as_bool(_cfg(config, "hardware", "use_gpu", default=True), default=True)
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


def train_catboost(df: pd.DataFrame, features: list[str] | None = None, config: dict | None = None):
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
        print("WARNING: CatBoost benchmark skipped because catboost is not installed or failed to import.")
        return None, features

    X = _prepare_x(df, features)
    y = pd.to_numeric(df["MWH"], errors="coerce").astype(float)
    sample_weight = build_sample_weights(df.reset_index(drop=True), cfg)

    errors: list[str] = []
    for backend_name, params in _attempts(cfg):
        try:
            print(f"Training CatBoost benchmark with {backend_name.upper()} backend...")
            model = CatBoostRegressor(**params)
            model.fit(X, y, sample_weight=sample_weight)
            _LAST_CATBOOST_TRAINING_INFO = {
                "enabled": catboost_enabled(cfg),
                "requested_gpu": _gpu_requested(cfg),
                "selected_backend": backend_name,
                "params": params,
                "failed_attempts": errors,
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
        out_dir = Path(_cfg(config, "project", "output_dir", default="forecast_outputs"))
        out_dir.mkdir(parents=True, exist_ok=True)
        (out_dir / "catboost_training_backend.json").write_text(
            json.dumps(get_last_catboost_training_info(), indent=2, default=str),
            encoding="utf-8",
        )
    except Exception:
        pass
