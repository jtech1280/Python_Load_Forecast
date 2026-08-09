from __future__ import annotations

import unittest

import numpy as np
import pandas as pd

from forecasting.model.catboost_model import get_last_catboost_training_info, train_catboost


def _synthetic_frame(n_days: int = 90) -> pd.DataFrame:
    """Hourly synthetic load with a simple, low-noise, easily-learnable signal so a model
    converges well before a generous iteration cap -- exactly the condition early stopping
    should catch."""
    rng = np.random.default_rng(11)
    dt = pd.date_range("2026-01-01", periods=24 * n_days, freq="h")
    hour = dt.hour.values.astype(float)
    temperature = 70 + 15 * np.sin((hour - 6) / 24 * 2 * np.pi) + rng.normal(0, 1, len(dt))
    mwh = 500 + 4 * temperature + 10 * np.sin(hour / 24 * 2 * np.pi) + rng.normal(0, 1, len(dt))
    return pd.DataFrame({"DT": dt, "MWH": mwh, "Temperature": temperature, "Hour": hour})


class CatBoostEarlyStoppingTests(unittest.TestCase):
    def setUp(self):
        self.df = _synthetic_frame()
        self.features = ["Temperature", "Hour"]

    def test_early_stopping_enabled_stops_before_iterations_cap(self):
        config = {
            "model": {
                "early_stopping": {"enabled": True, "validation_days": 14, "min_train_rows": 200, "rounds": 20, "metric": "mae"},
                "catboost": {"enabled": True, "iterations": 2000, "depth": 6, "task_type": "CPU"},
            },
            "hardware": {"use_gpu": False},
        }
        model, feats = train_catboost(self.df, self.features, config=config)
        self.assertIsNotNone(model)
        info = get_last_catboost_training_info()
        self.assertTrue(info["early_stopping"]["enabled"])
        self.assertIsNotNone(info["early_stopping"]["best_iteration"])
        # The model actually stopped short of the 2000-iteration cap on this easy signal.
        self.assertLess(model.tree_count_, 2000)
        self.assertEqual(model.tree_count_, info["early_stopping"]["tree_count"])

    def test_early_stopping_disabled_trains_full_iteration_count(self):
        config = {
            "model": {
                "early_stopping": {"enabled": False},
                "catboost": {"enabled": True, "iterations": 60, "depth": 4, "task_type": "CPU"},
            },
            "hardware": {"use_gpu": False},
        }
        model, feats = train_catboost(self.df, self.features, config=config)
        self.assertIsNotNone(model)
        info = get_last_catboost_training_info()
        self.assertFalse(info["early_stopping"]["enabled"])
        self.assertIsNone(info["early_stopping"]["best_iteration"])
        self.assertEqual(model.tree_count_, 60)

    def test_too_little_data_for_validation_split_falls_back_to_full_training(self):
        small_df = self.df.iloc[:100].copy()
        config = {
            "model": {
                "early_stopping": {"enabled": True, "validation_days": 45, "min_train_rows": 2000, "rounds": 10},
                "catboost": {"enabled": True, "iterations": 40, "depth": 3, "task_type": "CPU"},
            },
            "hardware": {"use_gpu": False},
        }
        model, feats = train_catboost(small_df, self.features, config=config)
        self.assertIsNotNone(model)
        info = get_last_catboost_training_info()
        self.assertFalse(info["early_stopping"]["enabled"])
        self.assertEqual(model.tree_count_, 40)


if __name__ == "__main__":
    unittest.main()
