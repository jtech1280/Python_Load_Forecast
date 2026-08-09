from __future__ import annotations

import importlib.util
import json
import sys
import tempfile
import unittest
from pathlib import Path

import pandas as pd

SCRIPTS_DIR = Path(__file__).resolve().parents[1] / "scripts"
if str(SCRIPTS_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPTS_DIR))

try:
    import optuna  # noqa: F401
    OPTUNA_AVAILABLE = True
except ImportError:
    OPTUNA_AVAILABLE = False

from forecasting.tuning.calibration_search import RawOriginBundle, save_raw_origin_bundles

if OPTUNA_AVAILABLE:
    tune_calibration_optuna = importlib.import_module("tune_calibration_optuna")


def _minimal_bundle(origin_number: int) -> RawOriginBundle:
    dt = pd.date_range("2026-06-01 00:00", periods=48, freq="h")
    base = pd.DataFrame(
        {
            "DT": dt,
            "Actual_MWH": 520.0,
            "Raw_Forecast_MWH": 500.0,
            "XGB_Pred_MWH": 500.0,
            "LGB_Pred_MWH": 500.0,
            "Hour": dt.hour.astype(float),
            "DOW": dt.dayofweek.astype(float),
            "Month": dt.month.astype(float),
            "Season": "Summer",
            "Temperature": 90.0,
            "Temperature_DailyMax": 95.0,
            "CloudCover_Norm": 0.1,
            "IsWeekend": 0,
            "IsHoliday": 0,
            "Forecast_Day": ((dt - dt[0]).days + 1).astype(float),
        }
    )
    return RawOriginBundle(
        origin_number=origin_number,
        origin_dt=dt[0] + pd.Timedelta(days=origin_number),
        calibration_days=3,
        raw_calibration=base.copy(),
        raw_origin=base.copy(),
        raw_weather_realism=pd.DataFrame(),
        raw_realized_scenarios={},
        raw_weather_scenarios={},
    )


_DISABLED_STAGES_CONFIG = {
    "calibration": {
        "targeted_residual_meta": {"enabled": False},
        "seasonal_enabled": False,
        "heat_peak_enabled": False,
    },
    "focused_shape_residual_learner": {"enabled": False},
    "operational_residual_learner": {"enabled": False},
    "daily_peak_shadow_model": {"enabled": False},
    "hot_ramp_peak_capture": {"enabled": False},
    "heat_persistence_peak_capture": {"enabled": False},
}


@unittest.skipUnless(OPTUNA_AVAILABLE, "optuna not installed")
class SplitBundlesTests(unittest.TestCase):
    def test_split_is_deterministic_and_covers_all_origins(self):
        bundles = [_minimal_bundle(i) for i in range(1, 11)]
        search1, holdout1 = tune_calibration_optuna.split_bundles(bundles, holdout_fraction=0.3, seed=7)
        search2, holdout2 = tune_calibration_optuna.split_bundles(bundles, holdout_fraction=0.3, seed=7)
        self.assertEqual([b.origin_number for b in search1], [b.origin_number for b in search2])
        self.assertEqual([b.origin_number for b in holdout1], [b.origin_number for b in holdout2])
        self.assertEqual(set(b.origin_number for b in search1) | set(b.origin_number for b in holdout1), set(range(1, 11)))
        self.assertEqual(set(b.origin_number for b in search1) & set(b.origin_number for b in holdout1), set())
        self.assertEqual(len(holdout1), 3)

    def test_zero_holdout_fraction_keeps_everything_in_search(self):
        bundles = [_minimal_bundle(i) for i in range(1, 5)]
        search, holdout = tune_calibration_optuna.split_bundles(bundles, holdout_fraction=0.0, seed=1)
        self.assertEqual(len(search), 4)
        self.assertEqual(holdout, [])

    def test_always_leaves_at_least_one_search_origin(self):
        bundles = [_minimal_bundle(i) for i in range(1, 3)]
        search, holdout = tune_calibration_optuna.split_bundles(bundles, holdout_fraction=0.99, seed=1)
        self.assertGreaterEqual(len(search), 1)


@unittest.skipUnless(OPTUNA_AVAILABLE, "optuna not installed")
class RunSearchEndToEndTests(unittest.TestCase):
    def test_run_search_writes_outputs_and_scores_holdout(self):
        bundles = [_minimal_bundle(i) for i in range(1, 9)]
        with tempfile.TemporaryDirectory() as tmp:
            cache_dir = Path(tmp) / "cache"
            output_dir = Path(tmp) / "out"
            save_raw_origin_bundles(bundles, cache_dir)

            summary = tune_calibration_optuna.run_search(
                config=_DISABLED_STAGES_CONFIG,
                cache_dir=cache_dir,
                n_trials=3,
                holdout_fraction=0.25,
                seed=1,
                output_dir=output_dir,
                study_name="test_study",
                storage=None,
                objective_weights=None,
            )

            self.assertEqual(len(summary["search_origins"]) + len(summary["holdout_origins"]), 8)
            self.assertIn("best_params", summary)
            self.assertIsInstance(summary["search_set_objective"], float)
            self.assertIsInstance(summary["holdout_set_objective"], float)

            self.assertTrue((output_dir / "calibration_search_trials.csv").exists())
            self.assertTrue((output_dir / "calibration_search_holdout_scorecard.csv").exists())
            best_params_path = output_dir / "calibration_search_best_params.json"
            self.assertTrue(best_params_path.exists())
            on_disk = json.loads(best_params_path.read_text(encoding="utf-8"))
            self.assertEqual(on_disk["n_trials"], 3)

            trials_df = pd.read_csv(output_dir / "calibration_search_trials.csv")
            self.assertEqual(len(trials_df), 3)


if __name__ == "__main__":
    unittest.main()
