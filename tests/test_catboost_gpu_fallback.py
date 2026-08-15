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

    def test_require_gpu_disables_cpu_fallback_even_when_global_fallback_enabled(self):
        config = {
            "hardware": {"use_gpu": True, "fallback_to_cpu": True},
            "model": {"catboost": {"task_type": "GPU", "require_gpu": True}},
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

    def test_require_gpu_overrides_cpu_task_type(self):
        config = {
            "hardware": {"use_gpu": True, "fallback_to_cpu": True},
            "model": {"catboost": {"task_type": "CPU", "require_gpu": True}},
        }
        backends = [name for name, _ in _attempts(config)]
        self.assertEqual(backends, ["gpu"])


class CatBoostGpuRamPartTests(unittest.TestCase):
    """Regression tests for the CUDA out-of-memory fix: replay.parallel.processes runs
    multiple origins concurrently against one GPU, and CatBoost's own default
    (gpu_ram_part=0.95, grab ~95% of free VRAM) causes every process after the first to
    fail with a bad-allocation error. gpu_ram_part must reach the GPU attempt only.
    """

    def test_configured_gpu_ram_part_is_applied_to_gpu_attempt(self):
        config = {
            "hardware": {"use_gpu": True, "fallback_to_cpu": True},
            "model": {"catboost": {"task_type": "GPU", "gpu_ram_part": 0.2}},
        }
        attempts = dict(_attempts(config))
        self.assertAlmostEqual(attempts["gpu"]["gpu_ram_part"], 0.2)

    def test_gpu_ram_part_not_set_on_cpu_attempt(self):
        config = {
            "hardware": {"use_gpu": True, "fallback_to_cpu": True},
            "model": {"catboost": {"task_type": "GPU", "gpu_ram_part": 0.2}},
        }
        attempts = dict(_attempts(config))
        self.assertNotIn("gpu_ram_part", attempts["cpu"])

    def test_unset_gpu_ram_part_leaves_catboost_default_untouched(self):
        config = {
            "hardware": {"use_gpu": True, "fallback_to_cpu": True},
            "model": {"catboost": {"task_type": "GPU"}},
        }
        attempts = dict(_attempts(config))
        self.assertNotIn("gpu_ram_part", attempts["gpu"])


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

    def test_train_catboost_require_gpu_raises_when_gpu_attempt_raises(self):
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
                    "require_gpu": True,
                },
            },
        }
        with patch(
            "forecasting.model.catboost_model._import_catboost",
            return_value=(FakeCatBoostRegressor, None),
        ):
            with self.assertRaisesRegex(RuntimeError, "CatBoost GPU is required"):
                train_catboost(df, features=["Temperature", "Hour"], config=config)

        info = get_last_catboost_training_info()
        self.assertIsNone(info["selected_backend"])
        self.assertTrue(info["requested_gpu"])
        self.assertTrue(info["require_gpu"])
        self.assertFalse(info["cpu_fallback_allowed"])
        self.assertEqual(len(info["failed_attempts"]), 1)
        self.assertIn("gpu", info["failed_attempts"][0])

    def test_train_catboost_require_gpu_rejects_cpu_only_config(self):
        df = _synthetic_frame()
        config = {
            "hardware": {"use_gpu": False, "fallback_to_cpu": True},
            "model": {
                "early_stopping": {"enabled": False},
                "catboost": {
                    "enabled": True,
                    "iterations": 10,
                    "depth": 2,
                    "task_type": "GPU",
                    "require_gpu": True,
                },
            },
        }
        with self.assertRaisesRegex(ValueError, "hardware.use_gpu=false"):
            train_catboost(df, features=["Temperature", "Hour"], config=config)

        info = get_last_catboost_training_info()
        self.assertIsNone(info["selected_backend"])
        self.assertTrue(info["require_gpu"])
        self.assertIn("hardware.use_gpu=false", info["reason"])


class AlwaysSucceedsCatBoostRegressor:
    """Stand-in that succeeds on whichever backend it's given, so the GPU attempt is
    the one actually selected (mirrors a real GPU-capable machine)."""

    def __init__(self, **params):
        self.params = params
        self.tree_count_ = 5

    def fit(self, X, y, **kwargs):
        return self

    def predict(self, X):
        return np.zeros(len(X))

    def get_best_iteration(self):
        return 4


class CatBoostEvalMetricSelectionTests(unittest.TestCase):
    def _config(
        self,
        *,
        task_type: str,
        require_gpu: bool = False,
        gpu_eval_metric: str | None = None,
    ) -> dict:
        catboost_cfg = {
            "enabled": True,
            "iterations": 10,
            "depth": 2,
            "task_type": task_type,
            "require_gpu": require_gpu,
        }
        if gpu_eval_metric is not None:
            catboost_cfg["gpu_eval_metric"] = gpu_eval_metric
        return {
            "hardware": {"use_gpu": True, "fallback_to_cpu": True},
            "model": {
                "early_stopping": {
                    "enabled": True,
                    "validation_days": 3,
                    "min_train_rows": 100,
                    "rounds": 5,
                    "metric": "mae",
                },
                "monotonic_constraints": {"enabled": False},
                "catboost": catboost_cfg,
            },
        }

    def test_gpu_attempt_uses_configured_gpu_eval_metric_for_mae_early_stopping(
        self,
    ):
        df = _synthetic_frame()
        with patch(
            "forecasting.model.catboost_model._import_catboost",
            return_value=(AlwaysSucceedsCatBoostRegressor, None),
        ):
            model, feats = train_catboost(
                df,
                features=["Temperature", "Hour"],
                config=self._config(
                    task_type="GPU", require_gpu=True, gpu_eval_metric="RMSE"
                ),
            )

        self.assertIsNotNone(model)
        info = get_last_catboost_training_info()
        self.assertEqual(info["selected_backend"], "gpu")
        self.assertEqual(info["params"]["eval_metric"], "RMSE")
        self.assertNotIn("_eval_metric_note", info["params"])
        self.assertEqual(info["early_stopping"]["metric"], "mae")
        self.assertEqual(info["early_stopping"]["actual_eval_metric"], "RMSE")
        self.assertEqual(
            info["early_stopping"]["eval_metric_note"], "configured_gpu_eval_metric"
        )

    def test_cpu_attempt_keeps_requested_mae_eval_metric(self):
        df = _synthetic_frame()
        with patch(
            "forecasting.model.catboost_model._import_catboost",
            return_value=(AlwaysSucceedsCatBoostRegressor, None),
        ):
            model, feats = train_catboost(
                df,
                features=["Temperature", "Hour"],
                config=self._config(task_type="CPU"),
            )

        self.assertIsNotNone(model)
        info = get_last_catboost_training_info()
        self.assertEqual(info["selected_backend"], "cpu")
        self.assertEqual(info["params"]["eval_metric"], "MAE")
        self.assertEqual(info["early_stopping"]["actual_eval_metric"], "MAE")
        self.assertIsNone(info["early_stopping"]["eval_metric_note"])


class CatBoostGpuMonotonicConstraintTests(unittest.TestCase):
    """CatBoost's GPU backend does not support monotone_constraints (a hard engine
    limitation, confirmed against CatBoost's own docs/FAQ) -- regression tests for the
    accuracy/speed tradeoff: GPU attempts must omit the constraint, CPU attempts (either
    chosen outright or reached via fallback) must still get it.
    """

    def _config(self, *, task_type: str) -> dict:
        return {
            "hardware": {"use_gpu": True, "fallback_to_cpu": True},
            "model": {
                "early_stopping": {"enabled": False},
                "monotonic_constraints": {"enabled": True},
                "catboost": {
                    "enabled": True,
                    "iterations": 10,
                    "depth": 2,
                    "task_type": task_type,
                },
            },
        }

    def test_gpu_attempt_omits_monotone_constraints(self):
        df = _synthetic_frame()
        with patch(
            "forecasting.model.catboost_model._import_catboost",
            return_value=(AlwaysSucceedsCatBoostRegressor, None),
        ):
            model, feats = train_catboost(
                df,
                features=["Temperature", "Hour"],
                config=self._config(task_type="GPU"),
            )

        self.assertIsNotNone(model)
        info = get_last_catboost_training_info()
        self.assertEqual(info["selected_backend"], "gpu")
        self.assertNotIn("monotone_constraints", info["params"])
        self.assertTrue(info["monotonic_constraints_requested"])
        self.assertFalse(info["monotonic_constraints_applied"])

    def test_cpu_fallback_still_gets_monotone_constraints_after_gpu_failure(self):
        df = _synthetic_frame()
        with patch(
            "forecasting.model.catboost_model._import_catboost",
            return_value=(FakeCatBoostRegressor, None),
        ):
            model, feats = train_catboost(
                df,
                features=["Temperature", "Hour"],
                config=self._config(task_type="GPU"),
            )

        self.assertIsNotNone(model)
        info = get_last_catboost_training_info()
        self.assertEqual(info["selected_backend"], "cpu")
        self.assertIn("monotone_constraints", info["params"])
        self.assertTrue(info["monotonic_constraints_requested"])
        self.assertTrue(info["monotonic_constraints_applied"])

    def test_cpu_only_task_type_still_gets_monotone_constraints(self):
        df = _synthetic_frame()
        with patch(
            "forecasting.model.catboost_model._import_catboost",
            return_value=(AlwaysSucceedsCatBoostRegressor, None),
        ):
            model, feats = train_catboost(
                df,
                features=["Temperature", "Hour"],
                config=self._config(task_type="CPU"),
            )

        self.assertIsNotNone(model)
        info = get_last_catboost_training_info()
        self.assertEqual(info["selected_backend"], "cpu")
        self.assertIn("monotone_constraints", info["params"])
        self.assertTrue(info["monotonic_constraints_applied"])


if __name__ == "__main__":
    unittest.main()
