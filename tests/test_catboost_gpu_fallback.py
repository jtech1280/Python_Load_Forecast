from __future__ import annotations

import unittest
from unittest.mock import patch

import numpy as np
import pandas as pd

from forecasting.model.catboost_model import (
    _attempts,
    get_last_catboost_training_info,
    train_catboost,
)


def _synthetic_frame(n_days: int = 20) -> pd.DataFrame:
    rng = np.random.default_rng(3)
    dt = pd.date_range("2026-01-01", periods=24 * n_days, freq="h")
    hour = dt.hour.values.astype(float)
    temperature = 70 + 15 * np.sin((hour - 6) / 24 * 2 * np.pi)
    mwh = 500 + 4 * temperature + rng.normal(0, 1, len(dt))
    return pd.DataFrame(
        {"DT": dt, "MWH": mwh, "Temperature": temperature, "Hour": hour}
    )


class CatBoostAttemptOrderingTests(unittest.TestCase):
    """config.yaml now defaults model.catboost.task_type to "GPU"; these lock in the
    backend-selection contract that change relies on."""

    def test_gpu_requested_and_fallback_enabled_tries_gpu_then_cpu(self):
        config = {
            "hardware": {"use_gpu": True, "fallback_to_cpu": True},
            "model": {"catboost": {"task_type": "GPU"}},
        }
        backends = [name for name, _ in _attempts(config)]
        self.assertEqual(backends, ["gpu", "cpu"])

    def test_gpu_requested_but_fallback_disabled_only_tries_gpu(self):
        config = {
            "hardware": {"use_gpu": True, "fallback_to_cpu": False},
            "model": {"catboost": {"task_type": "GPU"}},
        }
        backends = [name for name, _ in _attempts(config)]
        self.assertEqual(backends, ["gpu"])

    def test_hardware_use_gpu_false_skips_gpu_even_if_task_type_gpu(self):
        config = {
            "hardware": {"use_gpu": False, "fallback_to_cpu": True},
            "model": {"catboost": {"task_type": "GPU"}},
        }
        backends = [name for name, _ in _attempts(config)]
        self.assertEqual(backends, ["cpu"])

    def test_task_type_cpu_never_attempts_gpu(self):
        config = {
            "hardware": {"use_gpu": True, "fallback_to_cpu": True},
            "model": {"catboost": {"task_type": "CPU"}},
        }
        backends = [name for name, _ in _attempts(config)]
        self.assertEqual(backends, ["cpu"])


class FakeCatBoostRegressor:
    """Stand-in that fails to fit on GPU (as it would with no CUDA device visible)
    but succeeds on CPU, so train_catboost's runtime fallback can be exercised without
    a real GPU."""

    def __init__(self, **params):
        self.params = params
        self.tree_count_ = 5

    def fit(self, X, y, **kwargs):
        if self.params.get("task_type") == "GPU":
            raise RuntimeError("CatBoostError: no CUDA-capable device is detected")
        return self

    def predict(self, X):
        return np.zeros(len(X))

    def get_best_iteration(self):
        return 4


class TrainCatboostRuntimeFallbackTests(unittest.TestCase):
    def test_train_catboost_falls_back_to_cpu_when_gpu_attempt_raises(self):
        df = _synthetic_frame()
        config = {
            "hardware": {"use_gpu": True, "fallback_to_cpu": True},
            "model": {
                "early_stopping": {"enabled": False},
                "monotonic_constraints": {"enabled": False},
                "catboost": {
                    "enabled": True,
                    "iterations": 10,
                    "depth": 2,
                    "task_type": "GPU",
                },
            },
        }
        with patch(
            "forecasting.model.catboost_model._import_catboost",
            return_value=(FakeCatBoostRegressor, None),
        ):
            model, feats = train_catboost(
                df, features=["Temperature", "Hour"], config=config
            )

        self.assertIsNotNone(model)
        info = get_last_catboost_training_info()
        self.assertEqual(info["selected_backend"], "cpu")
        self.assertTrue(info["requested_gpu"])
        self.assertEqual(len(info["failed_attempts"]), 1)
        self.assertIn("gpu", info["failed_attempts"][0])


if __name__ == "__main__":
    unittest.main()
