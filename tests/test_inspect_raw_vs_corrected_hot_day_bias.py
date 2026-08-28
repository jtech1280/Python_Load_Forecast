from __future__ import annotations

import contextlib
import importlib
import io
import sys
import tempfile
import unittest
from pathlib import Path

import numpy as np
import pandas as pd

SCRIPTS_DIR = Path(__file__).resolve().parents[1] / "scripts"
if str(SCRIPTS_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPTS_DIR))

inspect_mod = importlib.import_module("inspect_raw_vs_corrected_hot_day_bias")

from forecasting.tuning.calibration_search import RawOriginBundle, save_raw_origin_bundles


def _replay_df(hour, daily_max, actual, raw, final, origin_id=None) -> pd.DataFrame:
    df = pd.DataFrame(
        {
            "Hour": hour,
            "Temperature_DailyMax": daily_max,
            "Actual_MWH": actual,
            "Raw_Forecast_MWH": raw,
            "Final_Backtest_Forecast_MWH": final,
        }
    )
    if origin_id is not None:
        df["Replay_Origin_ID"] = origin_id
    return df


class GateMaskTests(unittest.TestCase):
    def test_hot_peak_requires_hour_window_and_hot_temp(self):
        df = _replay_df(hour=list(range(24)), daily_max=95.0, actual=500, raw=490, final=495)
        mask = inspect_mod._gate_mask(df, inspect_mod.HOT_PEAK_TEST_NAME)
        self.assertTrue((df.loc[mask, "Hour"].between(16, 20)).all())
        self.assertTrue(mask.any())

    def test_hot_peak_excludes_cool_days(self):
        df = _replay_df(hour=[17], daily_max=75.0, actual=500, raw=490, final=495)
        mask = inspect_mod._gate_mask(df, inspect_mod.HOT_PEAK_TEST_NAME)
        self.assertFalse(mask.any())

    def test_peak_window_ignores_temperature(self):
        df = _replay_df(hour=[15], daily_max=60.0, actual=500, raw=490, final=495)
        mask = inspect_mod._gate_mask(df, inspect_mod.PEAK_WINDOW_TEST_NAME)
        self.assertTrue(mask.any())

    def test_unsupported_test_name_raises(self):
        df = _replay_df(hour=[17], daily_max=95.0, actual=500, raw=490, final=495)
        with self.assertRaises(ValueError):
            inspect_mod._gate_mask(df, "Some other gate")


class PooledMetricsTests(unittest.TestCase):
    def test_sign_convention_is_actual_minus_forecast(self):
        df = _replay_df(hour=[17, 17], daily_max=95.0, actual=[510, 510], raw=[500, 500], final=[505, 505])
        mae, bias, n = inspect_mod._pooled_metrics(df, "Raw_Forecast_MWH")
        self.assertAlmostEqual(mae, 10.0)
        self.assertAlmostEqual(bias, 10.0)
        self.assertEqual(n, 2)

    def test_missing_column_returns_none(self):
        df = _replay_df(hour=[17], daily_max=95.0, actual=500, raw=490, final=495)
        self.assertIsNone(inspect_mod._pooled_metrics(df, "CatBoost_Pred_MWH"))

    def test_missing_actual_column_returns_none(self):
        df = pd.DataFrame({"Raw_Forecast_MWH": [500.0]})
        self.assertIsNone(inspect_mod._pooled_metrics(df, "Raw_Forecast_MWH"))


class MainTests(unittest.TestCase):
    def _bundle(self, hour, daily_max, actual, raw, final) -> RawOriginBundle:
        n = len(hour)
        dt = pd.date_range("2026-07-15 00:00", periods=n, freq="h")
        df = pd.DataFrame(
            {
                "DT": dt,
                "Actual_MWH": actual,
                "Raw_Forecast_MWH": raw,
                "XGB_Pred_MWH": raw,
                "LGB_Pred_MWH": raw,
                "Hour": hour,
                "DOW": dt.dayofweek.astype(float),
                "Month": dt.month.astype(float),
                "Season": "Summer",
                "Temperature": daily_max,
                "Temperature_DailyMax": daily_max,
                "CloudCover_Norm": 0.1,
                "IsWeekend": 0,
                "IsHoliday": 0,
                "Forecast_Day": ((dt - dt[0]).days + 2).astype(float),
            }
        )
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

    def test_prints_pooled_raw_vs_final_comparison(self):
        dt_count = 24
        bundle = self._bundle(
            hour=list(range(dt_count)),
            daily_max=95.0,
            actual=520.0,
            raw=500.0,
            final=505.0,
        )
        with tempfile.TemporaryDirectory() as tmp:
            save_raw_origin_bundles([bundle], tmp)
            buf = io.StringIO()
            with contextlib.redirect_stdout(buf):
                sys.argv = ["prog", "--cache-dir", tmp, "--config", "forecasting/config.yaml"]
                rc = inspect_mod.main()
            out = buf.getvalue()
        self.assertEqual(rc, 0)
        self.assertIn("Raw ensemble (pre-correction)", out)
        self.assertIn("Final (post-correction)", out)

    def test_empty_cache_dir_raises(self):
        with tempfile.TemporaryDirectory() as tmp:
            sys.argv = ["prog", "--cache-dir", tmp, "--config", "forecasting/config.yaml"]
            with self.assertRaises(SystemExit):
                inspect_mod.main()

    def test_no_matching_gate_rows_is_a_clean_no_op(self):
        bundle = self._bundle(
            hour=[10],
            daily_max=60.0,
            actual=500.0,
            raw=495.0,
            final=498.0,
        )
        with tempfile.TemporaryDirectory() as tmp:
            save_raw_origin_bundles([bundle], tmp)
            buf = io.StringIO()
            with contextlib.redirect_stdout(buf):
                sys.argv = ["prog", "--cache-dir", tmp, "--config", "forecasting/config.yaml"]
                rc = inspect_mod.main()
            out = buf.getvalue()
        self.assertEqual(rc, 0)
        self.assertIn("No rows matched", out)


if __name__ == "__main__":
    unittest.main()
