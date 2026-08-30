from __future__ import annotations

import contextlib
import importlib
import io
import sys
import tempfile
import unittest
from pathlib import Path

import pandas as pd

SCRIPTS_DIR = Path(__file__).resolve().parents[1] / "scripts"
if str(SCRIPTS_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPTS_DIR))

inspect_mod = importlib.import_module("inspect_operational_residual_shadow_signal")

from forecasting.tuning.calibration_search import RawOriginBundle, save_raw_origin_bundles


class HotPeakMaskTests(unittest.TestCase):
    def test_requires_hour_and_temp(self):
        df = pd.DataFrame({"Hour": [17, 10], "Temperature_DailyMax": [95.0, 95.0]})
        mask = inspect_mod._hot_peak_mask(df)
        self.assertEqual(list(mask), [True, False])


class SummarizeShadowSignalTests(unittest.TestCase):
    def test_reports_mean_and_max_per_origin(self):
        gate_rows = pd.DataFrame(
            {
                "Replay_Origin_ID": ["origin_25", "origin_25", "origin_26"],
                "Actual_MWH": [520.0, 530.0, 500.0],
                "Raw_Forecast_MWH": [500.0, 500.0, 495.0],
                "Final_Backtest_Forecast_MWH": [501.0, 502.0, 496.0],
                "Auto_Residual_Correction_MWH": [1.0, 1.0, 0.5],
                "Auto_Residual_Full_Shadow_Correction_MWH": [15.0, 25.0, 3.0],
            }
        )
        summary = inspect_mod.summarize_shadow_signal(gate_rows)
        row25 = summary[summary["Replay_Origin_ID"] == "origin_25"].iloc[0]
        self.assertAlmostEqual(row25["Raw_Bias_MWH"], 25.0)
        self.assertAlmostEqual(row25["Auto_Residual_Full_Shadow_Correction_MWH_mean"], 20.0)
        self.assertAlmostEqual(row25["Auto_Residual_Full_Shadow_Correction_MWH_max"], 25.0)

    def test_sorted_by_absolute_raw_bias_descending(self):
        gate_rows = pd.DataFrame(
            {
                "Replay_Origin_ID": ["origin_a", "origin_b"],
                "Actual_MWH": [500.0, 500.0],
                "Raw_Forecast_MWH": [495.0, 520.0],
                "Final_Backtest_Forecast_MWH": [496.0, 519.0],
            }
        )
        summary = inspect_mod.summarize_shadow_signal(gate_rows)
        self.assertEqual(summary.iloc[0]["Replay_Origin_ID"], "origin_b")


class MainTests(unittest.TestCase):
    def _bundle(self, origin_number: int) -> RawOriginBundle:
        dt = pd.date_range("2026-07-27 00:00", periods=24, freq="h")
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
                "Forecast_Day": 1.0,
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

    def test_prints_summary_table(self):
        bundle = self._bundle(1)
        with tempfile.TemporaryDirectory() as tmp:
            save_raw_origin_bundles([bundle], tmp)
            buf = io.StringIO()
            with contextlib.redirect_stdout(buf):
                sys.argv = ["prog", "--cache-dir", tmp, "--config", "forecasting/config.yaml"]
                rc = inspect_mod.main()
            out = buf.getvalue()
        self.assertEqual(rc, 0)
        self.assertIn("Raw_Bias_MWH", out)

    def test_empty_cache_dir_raises(self):
        with tempfile.TemporaryDirectory() as tmp:
            sys.argv = ["prog", "--cache-dir", tmp, "--config", "forecasting/config.yaml"]
            with self.assertRaises(SystemExit):
                inspect_mod.main()


if __name__ == "__main__":
    unittest.main()
