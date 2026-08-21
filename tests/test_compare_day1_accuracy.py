from __future__ import annotations

import importlib
import sys
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import numpy as np
import pandas as pd
import yaml

SCRIPTS_DIR = Path(__file__).resolve().parents[1] / "scripts"
if str(SCRIPTS_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPTS_DIR))
TESTS_DIR = Path(__file__).resolve().parent
if str(TESTS_DIR) not in sys.path:
    sys.path.insert(0, str(TESTS_DIR))

compare_mod = importlib.import_module("compare_day1_accuracy")

from test_ablate_correction_stages import _config, _synthetic_bundle  # noqa: E402
from forecasting.tuning.calibration_search import save_raw_origin_bundles  # noqa: E402


class DayOneSliceTests(unittest.TestCase):
    def test_day1_slice_filters_to_forecast_day_one(self):
        df = pd.DataFrame({"Forecast_Day": [1, 1, 2, 3, np.nan]})
        sliced = compare_mod._day1_slice(df)
        self.assertEqual(len(sliced), 2)


class BreakdownByTests(unittest.TestCase):
    def test_breakdown_by_missing_column_returns_none(self):
        df = pd.DataFrame({"Actual_MWH": [1.0], "Final_Backtest_Forecast_MWH": [1.0]})
        self.assertIsNone(compare_mod._breakdown_by(df, "Replay_Origin_ID"))

    def test_compare_returns_empty_frame_when_group_col_missing_on_either_side(self):
        with_col = pd.DataFrame(
            {
                "Hour": [1, 1],
                "Actual_MWH": [10.0, 12.0],
                "Final_Backtest_Forecast_MWH": [9.0, 11.0],
            }
        )
        without_col = pd.DataFrame(
            {"Actual_MWH": [10.0], "Final_Backtest_Forecast_MWH": [9.0]}
        )
        result = compare_mod.compare(with_col, without_col, "Hour", "a", "b")
        self.assertTrue(result.empty)

    def test_compare_computes_delta_mae_per_group(self):
        a = pd.DataFrame(
            {
                "Hour": [1, 1, 2, 2],
                "Actual_MWH": [10.0, 10.0, 20.0, 20.0],
                "Final_Backtest_Forecast_MWH": [8.0, 8.0, 18.0, 18.0],
            }
        )
        b = pd.DataFrame(
            {
                "Hour": [1, 1, 2, 2],
                "Actual_MWH": [10.0, 10.0, 20.0, 20.0],
                "Final_Backtest_Forecast_MWH": [9.0, 9.0, 15.0, 15.0],
            }
        )
        result = compare_mod.compare(a, b, "Hour", "a", "b").set_index("Hour")
        # Hour 1: |10-8|=2 (a) vs |10-9|=1 (b) -> delta -1 (b improved)
        # Hour 2: |20-18|=2 (a) vs |20-15|=5 (b) -> delta +3 (b worsened)
        self.assertAlmostEqual(result.loc[1, "Delta_MAE_MWH"], -1.0)
        self.assertAlmostEqual(result.loc[2, "Delta_MAE_MWH"], 3.0)


class MainCliTests(unittest.TestCase):
    def test_main_runs_end_to_end_and_writes_csv(self):
        rng_a = np.random.default_rng(1)
        rng_b = np.random.default_rng(2)
        bundles_a = [_synthetic_bundle(1, hot=True, rng=rng_a)]
        bundles_b = [_synthetic_bundle(1, hot=True, rng=rng_b)]

        with tempfile.TemporaryDirectory() as tmp:
            tmp_path = Path(tmp)
            cache_a = tmp_path / "a"
            cache_b = tmp_path / "b"
            save_raw_origin_bundles(bundles_a, cache_a)
            save_raw_origin_bundles(bundles_b, cache_b)
            config_path = tmp_path / "config.yaml"
            config_path.write_text(yaml.safe_dump(_config()), encoding="utf-8")
            output_csv = tmp_path / "out.csv"

            argv = [
                "compare_day1_accuracy.py",
                "--config",
                str(config_path),
                "--cache-a",
                str(cache_a),
                "--label-a",
                "std",
                "--cache-b",
                str(cache_b),
                "--label-b",
                "ext",
                "--output-csv",
                str(output_csv),
            ]
            with patch.object(sys, "argv", argv):
                exit_code = compare_mod.main()
            self.assertEqual(exit_code, 0)
            self.assertTrue(output_csv.exists())
            written = pd.read_csv(output_csv)
            self.assertIn("Hour", written["Group_By"].tolist())


if __name__ == "__main__":
    unittest.main()
