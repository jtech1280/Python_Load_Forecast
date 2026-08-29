from __future__ import annotations

import importlib
import sys
import unittest
from pathlib import Path

import numpy as np
import pandas as pd

SCRIPTS_DIR = Path(__file__).resolve().parents[1] / "scripts"
if str(SCRIPTS_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPTS_DIR))

inspect_mod = importlib.import_module("inspect_load_level_at_matched_temperature")


class BuildMatchedScopeTests(unittest.TestCase):
    def test_keeps_only_hot_peak_hours_on_matched_temperature_days(self):
        dt = pd.date_range("2026-07-27", periods=48, freq="h", tz="America/Los_Angeles")
        load_df = pd.DataFrame({"DT": dt, "MWH": np.arange(len(dt), dtype=float)})
        daily_max = pd.DataFrame(
            {
                "Date": pd.to_datetime(["2026-07-27", "2026-07-28"]),
                "Temperature_DailyMax": [96.0, 70.0],
            }
        )
        scoped = inspect_mod.build_matched_scope(
            load_df, daily_max, hot_peak_hours={16, 17, 18}, temp_band_low=94.0, temp_band_high=99.0
        )
        self.assertTrue((scoped["Date"] == pd.Timestamp("2026-07-27")).all())
        self.assertTrue(scoped["Hour"].isin([16, 17, 18]).all())
        self.assertEqual(len(scoped), 3)

    def test_no_matching_days_returns_empty(self):
        dt = pd.date_range("2026-07-27", periods=24, freq="h", tz="America/Los_Angeles")
        load_df = pd.DataFrame({"DT": dt, "MWH": 1.0})
        daily_max = pd.DataFrame(
            {"Date": pd.to_datetime(["2026-07-27"]), "Temperature_DailyMax": [70.0]}
        )
        scoped = inspect_mod.build_matched_scope(
            load_df, daily_max, hot_peak_hours={16, 17}, temp_band_low=94.0, temp_band_high=99.0
        )
        self.assertTrue(scoped.empty)


class YearlySummaryTests(unittest.TestCase):
    def test_groups_by_year_with_expected_stats(self):
        scoped = pd.DataFrame(
            {
                "Year": [2020, 2020, 2021],
                "MWH": [100.0, 200.0, 300.0],
            }
        )
        summary = inspect_mod.yearly_summary(scoped)
        row_2020 = summary[summary["Year"] == 2020].iloc[0]
        self.assertEqual(row_2020["N_Hours"], 2)
        self.assertAlmostEqual(row_2020["Mean_MWH"], 150.0)
        self.assertAlmostEqual(row_2020["Max_MWH"], 200.0)

    def test_empty_input_returns_empty_with_expected_columns(self):
        summary = inspect_mod.yearly_summary(pd.DataFrame())
        self.assertTrue(summary.empty)
        self.assertIn("Mean_MWH", summary.columns)


class FitTrendAndPredictTests(unittest.TestCase):
    def test_extrapolates_a_clean_linear_trend(self):
        summary = pd.DataFrame(
            {
                "Year": [2020, 2021, 2022, 2023, 2024, 2025],
                "Mean_MWH": [500.0, 510.0, 520.0, 530.0, 540.0, 550.0],
            }
        )
        result = inspect_mod.fit_trend_and_predict(summary, target_year=2026)
        self.assertIsNotNone(result)
        predicted, slope = result
        self.assertAlmostEqual(slope, 10.0, places=6)
        self.assertAlmostEqual(predicted, 560.0, places=6)

    def test_insufficient_prior_years_returns_none(self):
        summary = pd.DataFrame({"Year": [2025], "Mean_MWH": [500.0]})
        result = inspect_mod.fit_trend_and_predict(summary, target_year=2026)
        self.assertIsNone(result)

    def test_only_uses_years_strictly_before_target(self):
        summary = pd.DataFrame(
            {
                "Year": [2020, 2021, 2026],
                "Mean_MWH": [500.0, 510.0, 9999.0],
            }
        )
        result = inspect_mod.fit_trend_and_predict(summary, target_year=2026)
        predicted, slope = result
        self.assertAlmostEqual(slope, 10.0, places=6)


if __name__ == "__main__":
    unittest.main()
