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

inspect_mod = importlib.import_module("summarize_monthly_replay_totals")


class DeduplicateToNearestHorizonTests(unittest.TestCase):
    def test_keeps_smallest_forecast_day_per_hour(self):
        df = pd.DataFrame(
            {
                "DT": [
                    "2026-08-05 10:00:00-07:00",
                    "2026-08-05 10:00:00-07:00",
                    "2026-08-06 10:00:00-07:00",
                ],
                "Forecast_Day": [7, 2, 1],
                "Actual_MWH": [500.0, 500.0, 510.0],
                "Final_Backtest_Forecast_MWH": [480.0, 495.0, 505.0],
            }
        )
        out = inspect_mod.deduplicate_to_nearest_horizon(df)
        self.assertEqual(len(out), 2)
        row = out[out["DT"] == "2026-08-05 10:00:00-07:00"].iloc[0]
        self.assertEqual(row["Forecast_Day"], 2)
        self.assertEqual(row["Final_Backtest_Forecast_MWH"], 495.0)

    def test_empty_input_returns_empty(self):
        out = inspect_mod.deduplicate_to_nearest_horizon(pd.DataFrame())
        self.assertTrue(out.empty)

    def test_single_row_per_hour_is_unaffected(self):
        df = pd.DataFrame(
            {
                "DT": ["2026-08-05 10:00", "2026-08-06 10:00"],
                "Forecast_Day": [3, 1],
                "Actual_MWH": [500.0, 510.0],
                "Final_Backtest_Forecast_MWH": [495.0, 505.0],
            }
        )
        out = inspect_mod.deduplicate_to_nearest_horizon(df)
        self.assertEqual(len(out), 2)


class ParseDtAsLocalWallClockTests(unittest.TestCase):
    def test_mixed_dst_offsets_resolve_to_correct_local_time(self):
        series = pd.Series(
            ["2026-08-05 10:00:00-07:00", "2026-01-15 10:00:00-08:00"]
        )
        parsed = inspect_mod.parse_dt_as_local_wall_clock(series, "America/Los_Angeles")
        self.assertIsNone(parsed.dt.tz)
        self.assertEqual(parsed.iloc[0], pd.Timestamp("2026-08-05 10:00:00"))
        self.assertEqual(parsed.iloc[1], pd.Timestamp("2026-01-15 10:00:00"))

    def test_late_evening_pdt_stays_in_the_same_local_month(self):
        series = pd.Series(["2026-07-31 23:00:00-07:00"])
        parsed = inspect_mod.parse_dt_as_local_wall_clock(series, "America/Los_Angeles")
        self.assertEqual(parsed.iloc[0].month, 7)
        self.assertEqual(parsed.iloc[0].day, 31)

    def test_unparseable_values_become_nat(self):
        series = pd.Series(["not a date"])
        parsed = inspect_mod.parse_dt_as_local_wall_clock(series, "America/Los_Angeles")
        self.assertTrue(parsed.isna().all())


class MonthlyTotalsTests(unittest.TestCase):
    def test_sums_actual_and_forecast(self):
        df = pd.DataFrame(
            {
                "Actual_MWH": [500.0, 520.0, 510.0],
                "Final_Backtest_Forecast_MWH": [490.0, 500.0, 505.0],
            }
        )
        out = inspect_mod.monthly_totals(df, "Final_Backtest_Forecast_MWH")
        self.assertEqual(out["N_Hours"], 3)
        self.assertAlmostEqual(out["Actual_Sum_MWH"], 1530.0)
        self.assertAlmostEqual(out["Forecast_Sum_MWH"], 1495.0)
        self.assertAlmostEqual(out["Diff_MWH"], 35.0)

    def test_rows_with_missing_values_are_excluded(self):
        df = pd.DataFrame(
            {
                "Actual_MWH": [500.0, None],
                "Final_Backtest_Forecast_MWH": [490.0, 500.0],
            }
        )
        out = inspect_mod.monthly_totals(df, "Final_Backtest_Forecast_MWH")
        self.assertEqual(out["N_Hours"], 1)
        self.assertAlmostEqual(out["Actual_Sum_MWH"], 500.0)


class MainTests(unittest.TestCase):
    def _write_results(self, tmp: str) -> Path:
        df = pd.DataFrame(
            {
                "DT": [
                    "2026-08-05 10:00:00-07:00",
                    "2026-08-05 10:00:00-07:00",
                    "2026-08-06 10:00:00-07:00",
                    "2026-07-31 10:00:00-07:00",
                ],
                "Replay_Origin_ID": ["origin_01", "origin_02", "origin_01", "origin_01"],
                "Forecast_Day": [7, 2, 6, 5],
                "Actual_MWH": [500.0, 500.0, 510.0, 400.0],
                "Final_Backtest_Forecast_MWH": [480.0, 495.0, 505.0, 390.0],
            }
        )
        path = Path(tmp) / "rolling_origin_replay_results.csv"
        df.to_csv(path, index=False)
        return path

    def test_prints_both_dedup_and_all_rows_totals(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = self._write_results(tmp)
            buf = io.StringIO()
            with contextlib.redirect_stdout(buf):
                sys.argv = [
                    "prog",
                    "--results-path",
                    str(path),
                    "--year",
                    "2026",
                    "--month",
                    "8",
                ]
                rc = inspect_mod.main()
            out = buf.getvalue()
        self.assertEqual(rc, 0)
        self.assertIn("Deduplicated", out)
        self.assertIn("All-rows", out)
        # The July 31 row must not leak into an August-only summary.
        self.assertNotIn("origin_01", out.split("Origins whose horizon")[0])

    def test_tz_aware_dt_keeps_local_wall_clock_month(self):
        """Regression test: converting to UTC before dropping the tz label shifts the
        wall-clock hour and can push a late-evening PDT timestamp into the next UTC day
        (and, near a month boundary, the next month) -- DT must be parsed keeping the
        local calendar date, not the UTC-shifted one."""
        df = pd.DataFrame(
            {
                # 2026-07-31 23:00 PDT is 2026-08-01 06:00 UTC -- must still count as July.
                "DT": ["2026-07-31 23:00:00-07:00"],
                "Replay_Origin_ID": ["origin_01"],
                "Forecast_Day": [1],
                "Actual_MWH": [500.0],
                "Final_Backtest_Forecast_MWH": [490.0],
            }
        )
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "rolling_origin_replay_results.csv"
            df.to_csv(path, index=False)
            buf = io.StringIO()
            with contextlib.redirect_stdout(buf):
                sys.argv = [
                    "prog",
                    "--results-path",
                    str(path),
                    "--year",
                    "2026",
                    "--month",
                    "7",
                ]
                rc = inspect_mod.main()
        self.assertEqual(rc, 0)
        self.assertIn("Actual_Sum_MWH: 500.00", buf.getvalue())

    def test_mixed_dst_offsets_in_one_file_do_not_crash(self):
        """Regression test for the real failure: a year-long fixed_origins replay
        crosses DST transitions, so the same DT column has some rows at -07:00
        (PDT) and others at -08:00 (PST). pandas raises ValueError on mixed
        offsets unless parsed with utc=True first."""
        df = pd.DataFrame(
            {
                "DT": [
                    "2026-08-05 10:00:00-07:00",  # PDT
                    "2026-01-15 10:00:00-08:00",  # PST
                ],
                "Replay_Origin_ID": ["origin_01", "origin_01"],
                "Forecast_Day": [1, 1],
                "Actual_MWH": [500.0, 400.0],
                "Final_Backtest_Forecast_MWH": [490.0, 390.0],
            }
        )
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "rolling_origin_replay_results.csv"
            df.to_csv(path, index=False)
            buf = io.StringIO()
            with contextlib.redirect_stdout(buf):
                sys.argv = [
                    "prog",
                    "--results-path",
                    str(path),
                    "--year",
                    "2026",
                    "--month",
                    "8",
                ]
                rc = inspect_mod.main()
        self.assertEqual(rc, 0)
        self.assertIn("Actual_Sum_MWH: 500.00", buf.getvalue())

    def test_missing_file_raises(self):
        sys.argv = [
            "prog",
            "--results-path",
            "/nonexistent/path.csv",
        ]
        with self.assertRaises(SystemExit):
            inspect_mod.main()

    def test_no_rows_for_month_raises(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = self._write_results(tmp)
            sys.argv = [
                "prog",
                "--results-path",
                str(path),
                "--year",
                "2020",
                "--month",
                "1",
            ]
            with self.assertRaises(SystemExit):
                inspect_mod.main()


if __name__ == "__main__":
    unittest.main()
