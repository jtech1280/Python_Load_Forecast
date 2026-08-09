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

from forecasting.tuning.calibration_search import (
    RawOriginBundle,
    save_raw_origin_bundles,
)

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
        search1, holdout1 = tune_calibration_optuna.split_bundles(
            bundles, holdout_fraction=0.3, seed=7
        )
        search2, holdout2 = tune_calibration_optuna.split_bundles(
            bundles, holdout_fraction=0.3, seed=7
        )
        self.assertEqual(
            [b.origin_number for b in search1], [b.origin_number for b in search2]
        )
        self.assertEqual(
            [b.origin_number for b in holdout1], [b.origin_number for b in holdout2]
        )
        self.assertEqual(
            set(b.origin_number for b in search1)
            | set(b.origin_number for b in holdout1),
            set(range(1, 11)),
        )
        self.assertEqual(
            set(b.origin_number for b in search1)
            & set(b.origin_number for b in holdout1),
            set(),
        )
        self.assertEqual(len(holdout1), 3)

    def test_zero_holdout_fraction_keeps_everything_in_search(self):
        bundles = [_minimal_bundle(i) for i in range(1, 5)]
        search, holdout = tune_calibration_optuna.split_bundles(
            bundles, holdout_fraction=0.0, seed=1
        )
        self.assertEqual(len(search), 4)
        self.assertEqual(holdout, [])

    def test_always_leaves_at_least_one_search_origin(self):
        bundles = [_minimal_bundle(i) for i in range(1, 3)]
        search, holdout = tune_calibration_optuna.split_bundles(
            bundles, holdout_fraction=0.99, seed=1
        )
        self.assertGreaterEqual(len(search), 1)


@unittest.skipUnless(OPTUNA_AVAILABLE, "optuna not installed")
class StabilityAndCentralRepeatTests(unittest.TestCase):
    def test_identical_repeats_report_zero_spread(self):
        params = {
            name: (low + high) / 2
            for name, _path, low, high in tune_calibration_optuna.V125_PARAM_SPACE
        }
        report = tune_calibration_optuna.stability_report(
            [dict(params), dict(params), dict(params)]
        )
        for stats in report.values():
            self.assertEqual(stats["range_fraction_of_search_space"], 0.0)

    def test_central_repeat_is_not_the_outlier(self):
        name0, _path0, low0, high0 = tune_calibration_optuna.V125_PARAM_SPACE[0]
        mid = (low0 + high0) / 2
        base = {
            name: (low + high) / 2
            for name, _path, low, high in tune_calibration_optuna.V125_PARAM_SPACE
        }
        near_median_a = dict(base)
        near_median_a[name0] = mid + 0.01 * (high0 - low0)
        near_median_b = dict(base)
        near_median_b[name0] = mid - 0.01 * (high0 - low0)
        outlier = dict(base)
        outlier[name0] = high0  # far from the other two

        all_params = [near_median_a, near_median_b, outlier]
        central_idx = tune_calibration_optuna.pick_central_repeat(all_params)
        self.assertIn(central_idx, (0, 1))

    def test_wide_spread_flagged_unstable(self):
        name0, _path0, low0, high0 = tune_calibration_optuna.V125_PARAM_SPACE[0]
        base = {
            name: (low + high) / 2
            for name, _path, low, high in tune_calibration_optuna.V125_PARAM_SPACE
        }
        p1 = dict(base)
        p1[name0] = low0
        p2 = dict(base)
        p2[name0] = high0
        report = tune_calibration_optuna.stability_report([p1, p2])
        self.assertAlmostEqual(report[name0]["range_fraction_of_search_space"], 1.0)


@unittest.skipUnless(OPTUNA_AVAILABLE, "optuna not installed")
class RunMultiSeedSearchEndToEndTests(unittest.TestCase):
    def test_run_multi_seed_search_writes_outputs_and_scores_final_holdout(self):
        bundles = [_minimal_bundle(i) for i in range(1, 9)]
        with tempfile.TemporaryDirectory() as tmp:
            cache_dir = Path(tmp) / "cache"
            output_dir = Path(tmp) / "out"
            save_raw_origin_bundles(bundles, cache_dir)

            n_repeats = 3
            n_trials = 2
            summary = tune_calibration_optuna.run_multi_seed_search(
                config=_DISABLED_STAGES_CONFIG,
                cache_dir=cache_dir,
                n_trials=n_trials,
                n_repeats=n_repeats,
                holdout_fraction=0.25,
                final_holdout_fraction=0.2,
                seed=1,
                output_dir=output_dir,
                study_name="test_study",
                storage=None,
                objective_weights=None,
            )

            self.assertEqual(summary["n_repeats"], n_repeats)
            self.assertEqual(len(summary["repeats"]), n_repeats)
            for repeat in summary["repeats"]:
                self.assertIsInstance(repeat["search_set_objective"], float)

            param_names = [
                n for n, _p, _l, _h in tune_calibration_optuna.V125_PARAM_SPACE
            ]
            self.assertEqual(
                set(summary["parameter_stability"].keys()), set(param_names)
            )
            self.assertIn(summary["recommended_repeat_index"], range(n_repeats))
            self.assertEqual(
                summary["recommended_params"],
                summary["repeats"][summary["recommended_repeat_index"]]["best_params"],
            )

            # 8 origins total; final_holdout_fraction=0.2 carves at least 1 into the final set,
            # which is never part of any repeat's search or repeat-holdout origins.
            final_holdout = set(summary["final_holdout_origins"])
            self.assertGreaterEqual(len(final_holdout), 1)
            for repeat in summary["repeats"]:
                self.assertEqual(final_holdout & set(repeat["search_origins"]), set())
                self.assertEqual(
                    final_holdout & set(repeat["repeat_holdout_origins"]), set()
                )

            self.assertTrue((output_dir / "calibration_search_trials.csv").exists())
            self.assertTrue(
                (output_dir / "calibration_search_final_holdout_scorecard.csv").exists()
            )
            best_params_path = output_dir / "calibration_search_best_params.json"
            self.assertTrue(best_params_path.exists())
            on_disk = json.loads(best_params_path.read_text(encoding="utf-8"))
            self.assertEqual(on_disk["n_repeats"], n_repeats)

            trials_df = pd.read_csv(output_dir / "calibration_search_trials.csv")
            self.assertEqual(len(trials_df), n_repeats * n_trials)
            self.assertEqual(
                set(trials_df["repeat_index"].unique()), set(range(n_repeats))
            )

    def test_n_repeats_one_still_works(self):
        bundles = [_minimal_bundle(i) for i in range(1, 6)]
        with tempfile.TemporaryDirectory() as tmp:
            cache_dir = Path(tmp) / "cache"
            output_dir = Path(tmp) / "out"
            save_raw_origin_bundles(bundles, cache_dir)

            summary = tune_calibration_optuna.run_multi_seed_search(
                config=_DISABLED_STAGES_CONFIG,
                cache_dir=cache_dir,
                n_trials=2,
                n_repeats=1,
                holdout_fraction=0.25,
                final_holdout_fraction=0.2,
                seed=5,
                output_dir=output_dir,
                study_name="single_repeat_study",
                storage=None,
                objective_weights=None,
            )
            self.assertEqual(summary["n_repeats"], 1)
            self.assertEqual(summary["recommended_repeat_index"], 0)
            for stats in summary["parameter_stability"].values():
                self.assertEqual(stats["range_fraction_of_search_space"], 0.0)


if __name__ == "__main__":
    unittest.main()
