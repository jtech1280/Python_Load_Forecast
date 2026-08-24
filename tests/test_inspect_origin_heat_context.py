from __future__ import annotations

import importlib
import sys
import tempfile
import unittest
from pathlib import Path

import pandas as pd

SCRIPTS_DIR = Path(__file__).resolve().parents[1] / "scripts"
if str(SCRIPTS_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPTS_DIR))

inspect_mod = importlib.import_module("inspect_origin_heat_context")

from forecasting.tuning.calibration_search import RawOriginBundle, save_raw_origin_bundles


def _bundle(origin_number: int, origin_date: str, cal_days: int = 3, extra_origin_cols: dict | None = None) -> RawOriginBundle:
    origin_dt = pd.Timestamp(f"{origin_date} 00:00:00")
    cal_dates = pd.date_range(end=origin_dt - pd.Timedelta(days=1), periods=cal_days, freq="D")
    cal_rows = [
        {"DT": d + pd.Timedelta(hours=h), "Temperature_DailyMax": 90.0 + i}
        for i, d in enumerate(cal_dates)
        for h in range(24)
    ]
    origin_rows = {
        "DT": pd.date_range(origin_dt, periods=24, freq="h"),
        "Temperature_DailyMax": 101.0,
    }
    origin_rows.update(extra_origin_cols or {})
    return RawOriginBundle(
        origin_number=origin_number,
        origin_dt=origin_dt,
        calibration_days=cal_days,
        raw_calibration=pd.DataFrame(cal_rows),
        raw_origin=pd.DataFrame(origin_rows),
        raw_weather_realism=pd.DataFrame(),
        raw_realized_scenarios={},
        raw_weather_scenarios={},
    )


class DailyMaxTests(unittest.TestCase):
    def test_collapses_hourly_rows_to_one_per_date(self):
        bundle = _bundle(1, "2026-07-27")
        daily = inspect_mod._daily_max(bundle.raw_calibration)
        self.assertEqual(len(daily), 3)
        self.assertTrue((daily == [90.0, 91.0, 92.0]).all())

    def test_empty_or_missing_columns_returns_empty_series(self):
        self.assertTrue(inspect_mod._daily_max(pd.DataFrame()).empty)
        self.assertTrue(inspect_mod._daily_max(pd.DataFrame({"DT": [pd.Timestamp("2026-01-01")]})).empty)


class MainSmokeTests(unittest.TestCase):
    def test_prints_origin_and_trailing_context_without_climatology_columns(self):
        with tempfile.TemporaryDirectory() as tmp:
            save_raw_origin_bundles([_bundle(25, "2026-07-27")], tmp)
            import io
            import contextlib

            buf = io.StringIO()
            with contextlib.redirect_stdout(buf):
                sys.argv = ["prog", "--cache-dir", tmp, "--origins", "25"]
                inspect_mod.main()
            out = buf.getvalue()
            self.assertIn("origin_25", out)
            self.assertIn("Origin-day Temperature_DailyMax: 101.0", out)
            self.assertIn("climatology columns not present", out)
            self.assertIn("NOT evidence the feature was off", out)

    def test_prints_climatology_columns_when_present(self):
        with tempfile.TemporaryDirectory() as tmp:
            save_raw_origin_bundles(
                [
                    _bundle(
                        25,
                        "2026-07-27",
                        extra_origin_cols={
                            "Climatology_Temp_PXX_F": 97.0,
                            "Temp_Climatology_Reference_Years": 6,
                            "Temp_Excess_Over_Climatology_F": 4.0,
                        },
                    )
                ],
                tmp,
            )
            import io
            import contextlib

            buf = io.StringIO()
            with contextlib.redirect_stdout(buf):
                sys.argv = ["prog", "--cache-dir", tmp, "--origins", "25"]
                inspect_mod.main()
            out = buf.getvalue()
            self.assertIn("Temp_Excess_Over_Climatology_F on origin day: 4.0", out)

    def test_missing_origin_warns_and_continues(self):
        with tempfile.TemporaryDirectory() as tmp:
            save_raw_origin_bundles([_bundle(25, "2026-07-27")], tmp)
            import io
            import contextlib

            buf = io.StringIO()
            with contextlib.redirect_stdout(buf):
                sys.argv = ["prog", "--cache-dir", tmp, "--origins", "25", "99"]
                inspect_mod.main()
            out = buf.getvalue()
            self.assertIn("origin(s) not found", out)
            self.assertIn("origin_25", out)

    def test_empty_cache_dir_raises(self):
        with tempfile.TemporaryDirectory() as tmp:
            sys.argv = ["prog", "--cache-dir", tmp, "--origins", "1"]
            with self.assertRaises(SystemExit):
                inspect_mod.main()


if __name__ == "__main__":
    unittest.main()
