"""GPU/CPU device utilities for model inference consistency.

Handles device mismatches between training and inference, preventing
performance degradation from GPU↔CPU data transfers.

Usage:
    import xgboost as xgb
    from forecasting.utils.device_utils import prepare_for_prediction

    # Automatically handles device consistency
    predictions = model.predict(prepare_for_prediction(X, model))
"""

from __future__ import annotations

import logging
import warnings
from typing import Any

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)


def get_model_device(model: Any) -> str:
    """Detect which device a trained XGBoost model is on.

    Returns:
        Device string: 'cuda', 'gpu_hist', or 'cpu'.
    """
    try:
        import xgboost as xgb

        if not isinstance(model, (xgb.XGBRegressor, xgb.XGBClassifier)):
            return "cpu"

        booster = model.get_booster()
        booster_config = booster.save_config()

        # Check modern XGBoost 2.x+ device parameter
        if "device" in booster_config and "cuda" in booster_config:
            return "cuda"

        # Check for legacy gpu_hist
        if "gpu_hist" in booster_config.lower():
            return "gpu_hist"

        return "cpu"
    except Exception:
        return "cpu"


def _move_to_device(data: Any, target_device: str) -> Any:
    """Move data to target device if needed.

    Args:
        data: pandas DataFrame, numpy array, or similar
        target_device: 'cuda' or 'gpu_hist' to move to GPU, 'cpu' for CPU

    Returns:
        Data on target device, or original data if conversion not possible.
    """
    if target_device == "cpu":
        # Ensure data is on CPU (numpy/pandas are already CPU)
        if isinstance(data, (pd.DataFrame, pd.Series)):
            return data
        if hasattr(data, "get") and data.__class__.__module__.startswith(
            "cupy"
        ):  # CuPy array
            try:
                return np.asarray(data.get())
            except Exception:
                return data
        return data

    if target_device in {"cuda", "gpu_hist"}:
        # Try to move to GPU using CuPy
        try:
            with warnings.catch_warnings():
                warnings.filterwarnings(
                    "ignore",
                    message="CUDA path could not be detected.*",
                    category=UserWarning,
                    module=r"cupy\._environment",
                )
                import cupy as cp

            # Convert pandas to numpy first if needed
            if isinstance(data, pd.DataFrame):
                data = data.values

            # Move to GPU if not already there
            if not isinstance(data, cp.ndarray):
                data = cp.asarray(data)
            return data
        except ImportError:
            logger.warning(
                "CuPy not installed; cannot move data to GPU. Falling back to CPU prediction. "
                "Install with: pip install cupy-cuda11x (replace 11x with your CUDA version)"
            )
            return data
        except Exception as e:
            logger.warning(f"Failed to move data to GPU: {e}. Falling back to CPU.")
            return data

    return data


def prepare_for_prediction(data: Any, model: Any) -> Any:
    """Prepare data for model prediction with device consistency.

    Detects which device the model trained on and automatically moves
    prediction data to match, preventing GPU↔CPU transfer overhead.

    Args:
        data: Input features as pandas DataFrame, numpy array, or similar.
        model: Trained XGBoost model.

    Returns:
        Data on the same device as the model, ready for prediction.

    Example:
        >>> import xgboost as xgb
        >>> from forecasting.utils.device_utils import prepare_for_prediction
        >>> model = xgb.XGBRegressor(device='cuda')
        >>> model.fit(X_train, y_train)
        >>> X_test_prepared = prepare_for_prediction(X_test, model)
        >>> predictions = model.predict(X_test_prepared)
    """
    model_device = get_model_device(model)

    if model_device in {"cuda", "gpu_hist"}:
        logger.debug(f"Model trained on {model_device}; moving prediction data to GPU.")
        return _move_to_device(data, "cuda")

    # Model is on CPU; ensure data is also on CPU
    return _move_to_device(data, "cpu")


def ensure_device_consistency(model: Any, config: dict | None = None) -> None:
    """Log device consistency info for debugging.

    Args:
        model: Trained XGBoost model.
        config: Optional config dict with hardware settings.
    """
    model_device = get_model_device(model)
    gpu_requested = False

    if config:
        hw = config.get("hardware", {}) or {}
        xgb_cfg = config.get("model", {}).get("xgb", {}) or {}
        gpu_requested = (
            hw.get("use_gpu") is not None
            and str(hw.get("use_gpu")).lower() in {"1", "true", "yes"}
        ) or str(xgb_cfg.get("device", "")).lower() in {"cuda", "gpu"}

    if gpu_requested and model_device == "cpu":
        logger.warning(
            "GPU was requested but model trained on CPU. Check hardware availability "
            "or set hardware.use_gpu=false to avoid this warning."
        )
    elif not gpu_requested and model_device != "cpu":
        logger.info(f"Model successfully using {model_device} for training.")
