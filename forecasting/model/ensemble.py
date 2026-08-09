from __future__ import annotations

import numpy as np


def _as_array_or_none(values):
    if values is None:
        return None
    arr = np.asarray(values, dtype=float)
    return arr


def _pad_to_length(arr: np.ndarray, length: int) -> np.ndarray:
    if len(arr) == length:
        return arr
    out = np.full(length, np.nan, dtype=float)
    keep = min(len(arr), length)
    if keep:
        out[:keep] = arr[:keep]
    return out


def blend_predictions(
    xgb_pred, lgb_pred, weights: dict[str, float], prophet_pred=None, catboost_pred=None
) -> np.ndarray:
    """Blend available model predictions using normalized configured weights.

    Supports the original XGB+LGB ensemble plus an optional Prophet component. Missing or NaN-only
    components are skipped row-by-row so the pipeline remains robust when Prophet is disabled.
    """
    preds = {
        "xgb": _as_array_or_none(xgb_pred),
        "lgb": _as_array_or_none(lgb_pred),
        "prophet": _as_array_or_none(prophet_pred),
        "catboost": _as_array_or_none(catboost_pred),
    }
    arrays = [v for v in preds.values() if v is not None]
    if not arrays:
        return np.asarray([], dtype=float)

    n = max(len(a) for a in arrays)
    out = np.zeros(n, dtype=float)
    wsum = np.zeros(n, dtype=float)

    defaults = {"xgb": 0.50, "lgb": 0.30, "prophet": 0.20, "catboost": 0.20}
    for name, arr in preds.items():
        if arr is None:
            continue
        arr = _pad_to_length(arr, n)
        w = float((weights or {}).get(name, defaults.get(name, 0.0)))
        if w <= 0:
            continue
        valid = np.isfinite(arr)
        out[valid] += w * arr[valid]
        wsum[valid] += w

    # If an entire row has no valid component, return NaN for that row rather than silently inventing 0.
    result = np.full(n, np.nan, dtype=float)
    mask = wsum > 0
    result[mask] = out[mask] / wsum[mask]
    return result
