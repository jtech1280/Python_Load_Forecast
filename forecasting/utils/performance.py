from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Any


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


def resolve_cpu_threads(config: dict | None = None) -> int:
    """Return the requested worker/thread count.

    A value of -1, 0, or null means "all visible logical CPU cores".
    """
    raw = _cfg(config, "hardware", "cpu_threads", default=-1)
    try:
        n = int(raw)
    except Exception:
        n = -1
    if n <= 0:
        return max(1, int(os.cpu_count() or 1))
    return max(1, n)


def resolve_n_jobs(config: dict | None, model_name: str, default: int = -1) -> int:
    """Resolve model n_jobs using model-specific config first, then hardware.cpu_threads."""
    model_val = _cfg(config, "model", model_name, "n_jobs", default=None)
    if model_val is None:
        model_val = default
    try:
        n = int(model_val)
    except Exception:
        n = int(default)
    if n <= 0:
        return resolve_cpu_threads(config)
    return n


def apply_runtime_thread_settings(config: dict | None = None) -> dict[str, Any]:
    """Set process-level thread env vars before heavy ML libraries are imported.

    This helps avoid accidental single-threaded NumPy/OpenMP execution. It also makes
    the run easier to diagnose because the settings are written to runtime_performance.json.
    """
    hw = _cfg(config, "hardware", default={}) or {}
    set_env = _as_bool(hw.get("set_thread_env", True), default=True)
    threads = resolve_cpu_threads(config)
    force = _as_bool(hw.get("overwrite_thread_env", False), default=False)

    env_names = [
        "OMP_NUM_THREADS",
        "OPENBLAS_NUM_THREADS",
        "MKL_NUM_THREADS",
        "NUMEXPR_NUM_THREADS",
        "VECLIB_MAXIMUM_THREADS",
        "BLIS_NUM_THREADS",
    ]
    applied: dict[str, str] = {}
    if set_env:
        for name in env_names:
            if force or not os.environ.get(name):
                os.environ[name] = str(threads)
            applied[name] = os.environ.get(name, "")

    # Avoid joblib/loky inner thread oversubscription when other packages use it.
    if set_env and (force or not os.environ.get("LOKY_MAX_CPU_COUNT")):
        os.environ["LOKY_MAX_CPU_COUNT"] = str(threads)
    if set_env:
        applied["LOKY_MAX_CPU_COUNT"] = os.environ.get("LOKY_MAX_CPU_COUNT", "")

    info = {
        "cpu_count_visible": int(os.cpu_count() or 1),
        "resolved_cpu_threads": int(threads),
        "set_thread_env": bool(set_env),
        "overwrite_thread_env": bool(force),
        "env": applied,
        "performance_mode": hw.get("performance_mode", "max"),
        "xgb_gpu_requested": xgb_gpu_requested(config),
        "lgb_gpu_requested": lgb_gpu_requested(config),
        "parallel_tree_training": parallel_tree_training_enabled(config),
    }
    return info


def write_runtime_performance_info(
    config: dict | None, info: dict[str, Any] | None = None
) -> None:
    try:
        out_dir = Path(
            _cfg(config, "project", "output_dir", default="forecast_outputs")
        )
        out_dir.mkdir(parents=True, exist_ok=True)
        payload = info or apply_runtime_thread_settings(config)
        (out_dir / "runtime_performance.json").write_text(
            json.dumps(payload, indent=2, default=str),
            encoding="utf-8",
        )
    except Exception:
        pass


def xgb_gpu_requested(config: dict | None = None) -> bool:
    hw = _cfg(config, "hardware", default={}) or {}
    p = _cfg(config, "model", "xgb", default={}) or {}
    explicit = hw.get("use_gpu", None)
    if explicit is not None:
        return _as_bool(explicit, default=False)
    return (
        str(p.get("device", "")).lower() in {"cuda", "gpu"}
        or str(p.get("tree_method", "")).lower() == "gpu_hist"
    )


def lgb_gpu_requested(config: dict | None = None) -> bool:
    hw = _cfg(config, "hardware", default={}) or {}
    p = _cfg(config, "model", "lgb", default={}) or {}
    return _as_bool(hw.get("use_lgb_gpu", p.get("use_gpu", False)), default=False)


def parallel_tree_training_enabled(config: dict | None = None) -> bool:
    hw = _cfg(config, "hardware", default={}) or {}
    if not _as_bool(hw.get("parallel_tree_training", True), default=True):
        return False

    mode = str(hw.get("performance_mode", "max")).lower()
    if mode in {"safe", "sequential", "single"}:
        return False

    # Best throughput case: XGB uses GPU while LightGBM uses CPU.
    # If both are GPU, sequential avoids fighting for the same card.
    if xgb_gpu_requested(config) and not lgb_gpu_requested(config):
        return True

    # Allow advanced users to force parallel CPU/CPU training, but leave it off by default
    # because two all-core jobs often oversubscribe and run slower than sequential all-core jobs.
    return _as_bool(hw.get("force_parallel_cpu_training", False), default=False)


def max_worker_count(config: dict | None = None) -> int:
    raw = _cfg(config, "hardware", "max_workers", default=-1)
    try:
        n = int(raw)
    except Exception:
        n = -1
    if n <= 0:
        return resolve_cpu_threads(config)
    return max(1, n)
