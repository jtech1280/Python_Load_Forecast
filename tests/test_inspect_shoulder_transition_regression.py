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

inspect_mod = importlib.import_module("inspect_shoulder_transition_regression")


def _row(origin: str, hour: int, season: str, temp: float, actual: float, forecast: float, **extra) -> dict:
    row = {
        "Replay_Origin_ID": origin,
        "Hour": hour,
        "Season": season,
        "Temperature_DailyMax": temp,
        "Actual_MWH": actual,
        "Final_Backtest_Forecast_MWH": forecast,
    }
    row.update(extra)
    return row


class ShoulderMaskTests(unittest.TestCase):
    def test_matches_spring_fall_hour_and_temp_window(self):
        df = pd.DataFrame(
            [
                _row("o1", 15, "Spring", 85.0, 500, 495),  # in
                _row("o1", 15, "Summer", 85.0, 500, 495),  # wrong season
                _row("o1", 10, "Spring", 85.0, 500, 495),  # wrong hour
                _row("o1", 15, "Spring", 60.0, 500, 495),  # too cool
                _row("o1", 15, "Spring", 100.0, 500, 495),  # too hot
            ]
        )
        mask = inspect_mod._shoulder_transition_mask(df)
        self.assertEqual(list(mask), [True, False, False, False, False])


class MainTests(unittest.TestCase):
    def test_flags_an_origin_touched_by_the_persistence_stage_elsewhere_in_its_horizon(self):
        rows = [_row("origin_A", h, "Spring", 85.0, 500.0, 495.0,
                      Heat_Persistence_Peak_Correction_MWH=0.0,
                      Heat_Persistence_Peak_Strong_Flag=0)
                for h in range(12, 23)]
        rows += [_row("origin_B", h, "Fall", 88.0, 500.0, 480.0,
                       Heat_Persistence_Peak_Correction_MWH=0.0,
                       Heat_Persistence_Peak_Strong_Flag=0)
                 for h in range(12, 23)]
        rows += [_row("origin_B", h, "Summer", 96.0, 550.0, 545.0,
                       Heat_Persistence_Peak_Correction_MWH=3.5,
                       Heat_Persistence_Peak_Strong_Flag=0)
                 for h in range(16, 21)]
        df = pd.DataFrame(rows)
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "backtest_results.csv"
            df.to_csv(path, index=False)
            buf = io.StringIO()
            with contextlib.redirect_stdout(buf):
                sys.argv = ["prog", "--csv", str(path)]
                inspect_mod.main()
            out = buf.getvalue()
        self.assertIn("origin_B", out)
        self.assertIn("origin_A", out)
        self.assertIn("1/2 origins", out)

    def test_strong_tier_correction_is_tracked_separately_from_moderate(self):
        rows = [_row("origin_C", h, "Spring", 85.0, 500.0, 495.0,
                      Heat_Persistence_Peak_Correction_MWH=0.0,
                      Heat_Persistence_Peak_Strong_Flag=0)
                for h in range(12, 23)]
        rows += [_row("origin_C", h, "Summer", 102.0, 550.0, 540.0,
                       Heat_Persistence_Peak_Correction_MWH=6.0,
                       Heat_Persistence_Peak_Strong_Flag=1)
                 for h in range(16, 21)]
        df = pd.DataFrame(rows)
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "backtest_results.csv"
            df.to_csv(path, index=False)
            buf = io.StringIO()
            with contextlib.redirect_stdout(buf):
                sys.argv = ["prog", "--csv", str(path)]
                inspect_mod.main()
            out = buf.getvalue()
        self.assertIn("origin_C", out)
        self.assertIn("1/1 origins", out)

    def test_missing_persistence_columns_still_prints_the_breakdown(self):
        rows = [_row("origin_A", h, "Spring", 85.0, 500.0, 495.0) for h in range(12, 23)]
        df = pd.DataFrame(rows)
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "backtest_results.csv"
            df.to_csv(path, index=False)
            buf = io.StringIO()
            with contextlib.redirect_stdout(buf):
                sys.argv = ["prog", "--csv", str(path)]
                inspect_mod.main()
            out = buf.getvalue()
        self.assertIn("origin_A", out)
        self.assertIn("Heat_Persistence_Peak_Correction_MWH not present", out)

    def test_no_matching_rows_is_a_clean_no_op(self):
        df = pd.DataFrame([_row("origin_A", 15, "Summer", 85.0, 500.0, 495.0)])
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "backtest_results.csv"
            df.to_csv(path, index=False)
            buf = io.StringIO()
            with contextlib.redirect_stdout(buf):
                sys.argv = ["prog", "--csv", str(path)]
                rc = inspect_mod.main()
            out = buf.getvalue()
        self.assertEqual(rc, 0)
        self.assertIn("No rows matched", out)

    def test_missing_csv_raises(self):
        sys.argv = ["prog", "--csv", "/no/such/file.csv"]
        with self.assertRaises(SystemExit):
            inspect_mod.main()

    def test_missing_required_column_raises(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "backtest_results.csv"
            pd.DataFrame([{"Hour": 15}]).to_csv(path, index=False)
            sys.argv = ["prog", "--csv", str(path)]
            with self.assertRaises(SystemExit):
                inspect_mod.main()


if __name__ == "__main__":
    unittest.main()
