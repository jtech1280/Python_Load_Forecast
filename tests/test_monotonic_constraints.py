from __future__ import annotations

import unittest

import numpy as np
import pandas as pd

from forecasting.model.monotonic_constraints import (
    DEFAULT_INCREASING_FEATURES,
    build_monotone_vector,
)
from forecasting.model.xgb_model import train_xgb, get_last_xgb_training_info
from forecasting.model.lgb_model import train_lgb, get_last_lgb_training_info
from forecasting.model.catboost_model import train_catboost


def _synthetic_frame(n_days: int = 40) -> pd.DataFrame:
    """Hourly synthetic load with a noisy-but-mostly-increasing heat response.

    A handful of very hot hours get an intentionally *inverted* noise spike so an
    unconstrained tree can pick up a locally non-monotonic split; the constrained
    model should not reproduce that inversion at prediction time.
    """
    rng = np.random.default_rng(7)
    dt = pd.date_range("2026-06-01", periods=24 * n_days, freq="h")
    hour = dt.hour.values.astype(float)
    temperature = (
        70 + 25 * np.sin((hour - 6) / 24 * 2 * np.pi) + rng.normal(0, 2, len(dt))
    )
    temperature = np.clip(temperature, 40, 115)
    cdd = np.clip(temperature - 65, 0, None)

    base_load = 500 + 6 * cdd + 3 * np.sin(hour / 24 * 2 * np.pi) * 20
    noise = rng.normal(0, 5, len(dt))
    mwh = base_load + noise

    # Inject a few sparse "hot but noisy-low" points to tempt an unconstrained model
    # into a non-monotonic split in the sparsely populated extreme-heat region.
    hot_idx = np.where(temperature > 100)[0]
    if len(hot_idx):
        dip_idx = hot_idx[: max(1, len(hot_idx) // 3)]
        mwh[dip_idx] -= 80

    return pd.DataFrame(
        {
            "DT": dt,
            "MWH": mwh,
            "Temperature": temperature,
            "Temperature_DailyMax": temperature,
            "CDD": cdd,
            "Hour": hour,
        }
    )


class MonotoneVectorTests(unittest.TestCase):
    def test_disabled_returns_none(self):
        vector = build_monotone_vector(
            ["Temperature", "MWH_Lag1"],
            {"model": {"monotonic_constraints": {"enabled": False}}},
        )
        self.assertIsNone(vector)

    def test_default_marks_known_heat_features_increasing(self):
        features = ["MWH_Lag1", "Temperature", "DOW", "CDD"]
        vector = build_monotone_vector(features, {})
        self.assertEqual(vector, (0, 1, 0, 1))

    def test_custom_increasing_and_decreasing_features(self):
        features = ["Temperature", "TempDrop_1Hr_F", "DOW"]
        config = {
            "model": {
                "monotonic_constraints": {
                    "enabled": True,
                    "increasing_features": ["Temperature"],
                    "decreasing_features": ["TempDrop_1Hr_F"],
                }
            }
        }
        vector = build_monotone_vector(features, config)
        self.assertEqual(vector, (1, -1, 0))

    def test_conflicting_feature_direction_raises(self):
        config = {
            "model": {
                "monotonic_constraints": {
                    "enabled": True,
                    "increasing_features": ["Temperature"],
                    "decreasing_features": ["Temperature"],
                }
            }
        }
        with self.assertRaises(ValueError):
            build_monotone_vector(["Temperature"], config)

    def test_no_overlap_with_present_features_returns_none(self):
        vector = build_monotone_vector(["DOW", "IsWeekend"], {})
        self.assertIsNone(vector)

    def test_default_increasing_list_is_nonempty(self):
        self.assertGreater(len(DEFAULT_INCREASING_FEATURES), 10)


class TreeTrainingMonotonicityTests(unittest.TestCase):
    def setUp(self):
        self.df = _synthetic_frame()
        self.features = ["Temperature", "Temperature_DailyMax", "CDD", "Hour"]
        self.hot_hours = pd.DataFrame(
            {
                "Temperature": np.linspace(95, 115, 21),
                "Temperature_DailyMax": np.linspace(95, 115, 21),
                "CDD": np.linspace(30, 50, 21),
                "Hour": np.full(21, 17.0),
            }
        )

    def _is_monotone_nondecreasing(self, preds: np.ndarray) -> bool:
        return bool(np.all(np.diff(preds) >= -1e-6))

    def test_xgb_predictions_monotone_increasing_with_constraint_enabled(self):
        config = {
            "model": {
                "early_stopping": {"enabled": False},
                "monotonic_constraints": {"enabled": True},
                "xgb": {"n_estimators": 60, "max_depth": 4},
            }
        }
        model, feats = train_xgb(self.df, self.features, config=config)
        preds = model.predict(self.hot_hours[feats].to_numpy(dtype=float))
        self.assertTrue(self._is_monotone_nondecreasing(preds))

    def test_xgb_training_info_logs_real_hyperparameters(self):
        config = {
            "model": {
                "early_stopping": {
                    "enabled": True,
                    "validation_days": 5,
                    "min_train_rows": 50,
                    "rounds": 5,
                },
                "monotonic_constraints": {"enabled": False},
                "xgb": {"n_estimators": 40, "max_depth": 3, "learning_rate": 0.1},
            }
        }
        train_xgb(self.df, self.features, config=config)
        info = get_last_xgb_training_info()
        # Regression check: previously the eval_set branch clobbered the logged params
        # with the model.fit() call signature instead of the actual hyperparameters.
        self.assertIn("params", info)
        self.assertEqual(info["params"].get("max_depth"), 3)
        self.assertAlmostEqual(info["params"].get("learning_rate"), 0.1)

    def test_lgb_predictions_monotone_increasing_with_constraint_enabled(self):
        config = {
            "model": {
                "early_stopping": {"enabled": False},
                "monotonic_constraints": {"enabled": True},
                "lgb": {"n_estimators": 60, "num_leaves": 15, "min_child_samples": 5},
            }
        }
        model, feats = train_lgb(self.df, self.features, config=config)
        preds = model.predict(self.hot_hours[feats].to_numpy(dtype=float))
        self.assertTrue(self._is_monotone_nondecreasing(preds))

    def test_lgb_training_info_logs_real_hyperparameters(self):
        config = {
            "model": {
                "early_stopping": {
                    "enabled": True,
                    "validation_days": 5,
                    "min_train_rows": 50,
                    "rounds": 5,
                },
                "monotonic_constraints": {"enabled": False},
                "lgb": {"n_estimators": 40, "num_leaves": 20, "learning_rate": 0.1},
            }
        }
        train_lgb(self.df, self.features, config=config)
        info = get_last_lgb_training_info()
        self.assertIn("params", info)
        self.assertEqual(info["params"].get("num_leaves"), 20)
        self.assertAlmostEqual(info["params"].get("learning_rate"), 0.1)

    def test_catboost_predictions_monotone_increasing_with_constraint_enabled(self):
        config = {
            "model": {
                "monotonic_constraints": {"enabled": True},
                "catboost": {
                    "enabled": True,
                    "iterations": 60,
                    "depth": 4,
                    "task_type": "CPU",
                },
            },
            "hardware": {"use_gpu": False},
        }
        model, feats = train_catboost(self.df, self.features, config=config)
        self.assertIsNotNone(model)
        preds = model.predict(self.hot_hours[feats].to_numpy(dtype=float))
        self.assertTrue(self._is_monotone_nondecreasing(preds))


if __name__ == "__main__":
    unittest.main()
