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

compare = importlib.import_module("compare_origin_bias_across_caches")


def _replay_df(origin_id: str, hour: list[int], daily_max: float, actual: float, forecast: float) -> pd.DataFrame:
    n = len(hour)
    return pd.DataFrame(
        {
            "Replay_Origin_ID": origin_id,
            "Hour": hour,
            "Temperature_DailyMax": daily_max,
            "Actual_MWH": actual,
            "Final_Backtest_Forecast_MWH": forecast,
        }
    )


class GateMaskTests(unittest.TestCase):
    def test_hot_peak_requires_both_hour_window_and_hot_temp(self):
        df = _replay_df("origin_01", hour=list(range(24)), daily_max=95.0, actual=500, forecast=490)
        mask = compare._gate_mask(df, compare.HOT_PEAK_TEST_NAME)
        self.assertTrue((df.loc[mask, "Hour"].between(16, 20)).all())
        self.assertTrue(mask.any())

    def test_hot_peak_excludes_cool_days_even_in_hour_window(self):
        df = _replay_df("origin_01", hour=[17], daily_max=75.0, actual=500, forecast=490)
        mask = compare._gate_mask(df, compare.HOT_PEAK_TEST_NAME)
        self.assertFalse(mask.any())

    def test_peak_window_ignores_temperature(self):
        df = _replay_df("origin_01", hour=[15], daily_max=60.0, actual=500, forecast=490)
        mask = compare._gate_mask(df, compare.PEAK_WINDOW_TEST_NAME)
        self.assertTrue(mask.any())

    def test_unsupported_test_name_raises(self):
        df = _replay_df("origin_01", hour=[17], daily_max=95.0, actual=500, forecast=490)
        with self.assertRaises(ValueError):
            compare._gate_mask(df, "Some other gate")


class OriginSignedBiasTests(unittest.TestCase):
    def test_sign_convention_is_actual_minus_forecast(self):
        # Actual runs 10 MWH above forecast at every hot-peak hour -> positive bias
        # (under-forecasting), matching build_production_readiness_scorecard's convention.
        df = _replay_df("origin_01", hour=[16, 17, 18, 19, 20], daily_max=95.0, actual=510, forecast=500)
        bias = compare._origin_signed_bias(df, compare.HOT_PEAK_TEST_NAME)
        self.assertAlmostEqual(bias["origin_01"], 10.0)

    def test_over_forecasting_gives_negative_bias(self):
        df = _replay_df("origin_01", hour=[16, 17, 18], daily_max=95.0, actual=490, forecast=500)
        bias = compare._origin_signed_bias(df, compare.HOT_PEAK_TEST_NAME)
        self.assertAlmostEqual(bias["origin_01"], -10.0)

    def test_multiple_origins_kept_separate(self):
        df = pd.concat(
            [
                _replay_df("origin_01", hour=[17], daily_max=95.0, actual=505, forecast=500),
                _replay_df("origin_02", hour=[17], daily_max=95.0, actual=480, forecast=500),
            ],
            ignore_index=True,
        )
        bias = compare._origin_signed_bias(df, compare.HOT_PEAK_TEST_NAME)
        self.assertAlmostEqual(bias["origin_01"], 5.0)
        self.assertAlmostEqual(bias["origin_02"], -20.0)

    def test_missing_replay_origin_id_returns_empty_series_not_a_crash(self):
        df = _replay_df("origin_01", hour=[17], daily_max=95.0, actual=505, forecast=500).drop(
            columns=["Replay_Origin_ID"]
        )
        bias = compare._origin_signed_bias(df, compare.HOT_PEAK_TEST_NAME)
        self.assertTrue(bias.empty)

    def test_empty_or_none_input_returns_empty_series(self):
        self.assertTrue(compare._origin_signed_bias(pd.DataFrame(), compare.HOT_PEAK_TEST_NAME).empty)
        self.assertTrue(compare._origin_signed_bias(None, compare.HOT_PEAK_TEST_NAME).empty)


if __name__ == "__main__":
    unittest.main()
