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

inspect_mod = importlib.import_module("inspect_raw_bias_by_forecast_day")

from forecasting.tuning.calibration_search import RawOriginBundle, save_raw_origin_bundles


class BiasByOriginAndForecastDayTests(unittest.TestCase):
    def test_pivots_mean_signed_bias_by_origin_and_day(self):
        gate_rows = pd.DataFrame(
            {
                "Replay_Origin_ID": ["origin_01", "origin_01", "origin_01", "origin_01", "origin_02"],
                "Forecast_Day": [1, 1, 2, 2, 1],
                "Actual_MWH": [510.0, 510.0, 530.0, 530.0, 500.0],
                "Raw_Forecast_MWH": [500.0, 500.0, 500.0, 500.0, 495.0],
            }
        )
        pivot = inspect_mod.bias_by_origin_and_forecast_day(gate_rows, "Raw_Forecast_MWH")
        self.assertAlmostEqual(pivot.loc["origin_01", "Day1"], 10.0)
        self.assertAlmostEqual(pivot.loc["origin_01", "Day2"], 30.0)
        self.assertAlmostEqual(pivot.loc["origin_02", "Day1"], 5.0)
        self.assertTrue(pd.isna(pivot.loc["origin_02"].get("Day2", np.nan)))

    def test_rows_missing_forecast_day_are_dropped(self):
        gate_rows = pd.DataFrame(
            {
                "Replay_Origin_ID": ["origin_01", "origin_01"],
                "Forecast_Day": [1, np.nan],
                "Actual_MWH": [510.0, 999.0],
                "Raw_Forecast_MWH": [500.0, 0.0],
            }
        )
        pivot = inspect_mod.bias_by_origin_and_forecast_day(gate_rows, "Raw_Forecast_MWH")
        self.assertEqual(list(pivot.columns), ["Day1"])


class GateMaskTests(unittest.TestCase):
    def test_hot_peak_requires_hour_and_temp(self):
        df = pd.DataFrame({"Hour": [17, 10], "Temperature_DailyMax": [95.0, 95.0]})
        mask = inspect_mod._gate_mask(df, inspect_mod.HOT_PEAK_TEST_NAME)
        self.assertEqual(list(mask), [True, False])

    def test_unsupported_test_name_raises(self):
        df = pd.DataFrame({"Hour": [17], "Temperature_DailyMax": [95.0]})
        with self.assertRaises(ValueError):
            inspect_mod._gate_mask(df, "nonsense")


class MainTests(unittest.TestCase):
    def _bundle(self, origin_number: int, start: str, n_days: int) -> RawOriginBundle:
        dt = pd.date_range(start, periods=n_days * 24, freq="h")
        df = pd.DataFrame(
            {
                "DT": dt,
                "Replay_Origin_ID": f"origin_{origin_number:02d}",
                "Actual_MWH": 520.0,
                "Raw_Forecast_MWH": 500.0,
                "XGB_Pred_MWH": 500.0,
                "LGB_Pred_MWH": 500.0,
                "Hour": dt.hour,
                "DOW": dt.dayofweek.astype(float),
                "Month": dt.month.astype(float),
                "Season": "Summer",
                "Temperature": 95.0,
                "Temperature_DailyMax": 95.0,
                "CloudCover_Norm": 0.1,
                "IsWeekend": 0,
                "IsHoliday": 0,
                "Forecast_Day": ((dt - dt[0]).days + 1).astype(float),
            }
        )
        return RawOriginBundle(
            origin_number=origin_number,
            origin_dt=dt[0],
            calibration_days=3,
            raw_calibration=df.copy(),
            raw_origin=df.copy(),
            raw_weather_realism=pd.DataFrame(),
            raw_realized_scenarios={},
            raw_weather_scenarios={},
        )

    def test_prints_pivot_table(self):
        bundle = self._bundle(1, "2026-07-27 00:00", 3)
        with tempfile.TemporaryDirectory() as tmp:
            save_raw_origin_bundles([bundle], tmp)
            buf = io.StringIO()
            with contextlib.redirect_stdout(buf):
                sys.argv = [
                    "prog",
                    "--cache-dir",
                    tmp,
                    "--config",
                    "forecasting/config.yaml",
                    "--test-name",
                    "Peak window hours 14-18",
                ]
                rc = inspect_mod.main()
            out = buf.getvalue()
        self.assertEqual(rc, 0)
        self.assertIn("Day1", out)

    def test_empty_cache_dir_raises(self):
        with tempfile.TemporaryDirectory() as tmp:
            sys.argv = ["prog", "--cache-dir", tmp, "--config", "forecasting/config.yaml"]
            with self.assertRaises(SystemExit):
                inspect_mod.main()


if __name__ == "__main__":
    unittest.main()
