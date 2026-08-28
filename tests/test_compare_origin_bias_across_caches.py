from __future__ import annotations

import importlib
import io
import contextlib
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

compare = importlib.import_module("compare_origin_bias_across_caches")

from forecasting.tuning.calibration_search import RawOriginBundle, save_raw_origin_bundles


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


class MainConfigSelectionTests(unittest.TestCase):
    """Regression coverage for the exact mistake that undermined an earlier real
    validation: without --config-b, both caches were silently scored with the same
    config, hiding a correction-chain-parameter difference entirely."""

    def _bundle(self) -> RawOriginBundle:
        dt = pd.date_range("2026-07-15", periods=24, freq="h")
        df = pd.DataFrame({"DT": dt, "Temperature_DailyMax": 95.0})
        return RawOriginBundle(
            origin_number=1,
            origin_dt=dt[0],
            calibration_days=3,
            raw_calibration=df.copy(),
            raw_origin=df.copy(),
            raw_weather_realism=pd.DataFrame(),
            raw_realized_scenarios={},
            raw_weather_scenarios={},
        )

    def test_without_config_b_both_caches_are_scored_with_the_same_config(self):
        with tempfile.TemporaryDirectory() as tmp:
            tmp_path = Path(tmp)
            cache_a, cache_b = tmp_path / "a", tmp_path / "b"
            save_raw_origin_bundles([self._bundle()], cache_a)
            save_raw_origin_bundles([self._bundle()], cache_b)
            config_a_path = tmp_path / "config_a.yaml"
            config_a_path.write_text(yaml.safe_dump({"marker": "A"}))

            seen_configs = []

            def fake_score_bundles(bundles, config):
                seen_configs.append(config)
                return pd.DataFrame()

            with patch.object(compare, "score_bundles", side_effect=fake_score_bundles):
                with contextlib.redirect_stdout(io.StringIO()) as buf:
                    sys.argv = [
                        "prog",
                        "--cache-a", str(cache_a),
                        "--cache-b", str(cache_b),
                        "--config", str(config_a_path),
                    ]
                    with self.assertRaises(SystemExit):
                        compare.main()

            self.assertEqual(len(seen_configs), 2)
            self.assertEqual(seen_configs[0].get("marker"), "A")
            self.assertEqual(seen_configs[1].get("marker"), "A")
            self.assertIn("NOTE: --config-b not given", buf.getvalue())

    def test_with_config_b_each_cache_is_scored_with_its_own_config(self):
        with tempfile.TemporaryDirectory() as tmp:
            tmp_path = Path(tmp)
            cache_a, cache_b = tmp_path / "a", tmp_path / "b"
            save_raw_origin_bundles([self._bundle()], cache_a)
            save_raw_origin_bundles([self._bundle()], cache_b)
            config_a_path = tmp_path / "config_a.yaml"
            config_a_path.write_text(yaml.safe_dump({"marker": "A"}))
            config_b_path = tmp_path / "config_b.yaml"
            config_b_path.write_text(yaml.safe_dump({"marker": "B"}))

            seen_configs = []

            def fake_score_bundles(bundles, config):
                seen_configs.append(config)
                return pd.DataFrame()

            with patch.object(compare, "score_bundles", side_effect=fake_score_bundles):
                with contextlib.redirect_stdout(io.StringIO()) as buf:
                    sys.argv = [
                        "prog",
                        "--cache-a", str(cache_a),
                        "--cache-b", str(cache_b),
                        "--config", str(config_a_path),
                        "--config-b", str(config_b_path),
                    ]
                    with self.assertRaises(SystemExit):
                        compare.main()

            self.assertEqual(len(seen_configs), 2)
            self.assertEqual(seen_configs[0].get("marker"), "A")
            self.assertEqual(seen_configs[1].get("marker"), "B")
            self.assertNotIn("NOTE: --config-b not given", buf.getvalue())


if __name__ == "__main__":
    unittest.main()
