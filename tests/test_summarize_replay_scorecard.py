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

summarize = importlib.import_module("summarize_replay_scorecard")


def _scorecard_df(mae_by_test: dict[str, float], bias_by_test: dict[str, float] | None = None) -> pd.DataFrame:
    bias_by_test = bias_by_test or {}
    rows = []
    for test, mae in mae_by_test.items():
        rows.append(
            {
                "Test": test,
                "Pass": mae < 8.0,
                "N": 100,
                "MAE_MWH": mae,
                "MAPE_PCT": 5.0,
                "Bias_MWH": bias_by_test.get(test, 0.0),
                "P90_AbsError_MWH": mae * 1.5,
                "Max_Underforecast_MWH": mae * 2.0,
                "Underforecast_At_Actual_Peak_MWH": mae,
            }
        )
    return pd.DataFrame(rows)


class SummarizeRowsTests(unittest.TestCase):
    def test_only_key_tests_are_kept_in_declared_order(self):
        df = _scorecard_df({"Hot peak days": 8.0, "SomeOtherTest": 99.0, "Day 1 only": 3.5})
        rows = summarize._summarize_rows(df)
        self.assertEqual([r["Test"] for r in rows], ["Day 1 only", "Hot peak days"])

    def test_missing_test_is_simply_absent_not_an_error(self):
        df = _scorecard_df({"Hot peak days": 8.0})
        rows = summarize._summarize_rows(df)
        self.assertEqual(len(rows), 1)
        self.assertEqual(rows[0]["Test"], "Hot peak days")


class MainCompareTests(unittest.TestCase):
    def test_no_compare_flag_prints_only_the_current_scorecard(self):
        with tempfile.TemporaryDirectory() as tmp:
            out_dir = Path(tmp)
            _scorecard_df({"Hot peak days": 8.0}).to_csv(
                out_dir / "production_readiness_scorecard.csv", index=False
            )
            buf = io.StringIO()
            with contextlib.redirect_stdout(buf):
                sys.argv = ["prog", "--output-dir", str(out_dir)]
                summarize.main()
            out = buf.getvalue()
            self.assertIn("Hot peak days", out)
            self.assertNotIn("Comparison against", out)

    def test_compare_path_prints_a_delta_table_with_correct_sign(self):
        """Delta = current - compare, so an improvement (lower MAE now) must show as
        negative Delta_MAE -- this is exactly the sign convention documented in the
        printed header, and the one a reader will actually act on."""
        with tempfile.TemporaryDirectory() as tmp:
            out_dir = Path(tmp)
            _scorecard_df({"Hot peak days": 8.0}, {"Hot peak days": 4.0}).to_csv(
                out_dir / "production_readiness_scorecard.csv", index=False
            )
            before_path = out_dir / "before.csv"
            _scorecard_df({"Hot peak days": 8.4}, {"Hot peak days": 4.5}).to_csv(
                before_path, index=False
            )
            buf = io.StringIO()
            with contextlib.redirect_stdout(buf):
                sys.argv = [
                    "prog",
                    "--output-dir",
                    str(out_dir),
                    "--compare-path",
                    str(before_path),
                ]
                summarize.main()
            out = buf.getvalue()
            self.assertIn("Comparison against", out)
            self.assertIn("-0.4000", out)  # Delta_MAE: 8.0 - 8.4
            self.assertIn("-0.5000", out)  # Delta_Bias: 4.0 - 4.5

    def test_compare_label_resolves_via_the_same_label_convention_as_the_primary_scorecard(self):
        with tempfile.TemporaryDirectory() as tmp:
            out_dir = Path(tmp)
            _scorecard_df({"Hot peak days": 8.0}).to_csv(
                out_dir / "production_readiness_scorecard.csv", index=False
            )
            _scorecard_df({"Hot peak days": 9.0}).to_csv(
                out_dir / "production_readiness_scorecard_before_moderate_tier.csv", index=False
            )
            buf = io.StringIO()
            with contextlib.redirect_stdout(buf):
                sys.argv = [
                    "prog",
                    "--output-dir",
                    str(out_dir),
                    "--compare-label",
                    "before_moderate_tier",
                ]
                summarize.main()
            out = buf.getvalue()
            self.assertIn("production_readiness_scorecard_before_moderate_tier.csv", out)
            self.assertIn("-1.0000", out)  # Delta_MAE: 8.0 - 9.0

    def test_missing_compare_path_raises(self):
        with tempfile.TemporaryDirectory() as tmp:
            out_dir = Path(tmp)
            _scorecard_df({"Hot peak days": 8.0}).to_csv(
                out_dir / "production_readiness_scorecard.csv", index=False
            )
            sys.argv = [
                "prog",
                "--output-dir",
                str(out_dir),
                "--compare-path",
                str(out_dir / "does_not_exist.csv"),
            ]
            with self.assertRaises(FileNotFoundError):
                summarize.main()

    def test_test_present_only_on_one_side_shows_dashes_not_a_crash(self):
        with tempfile.TemporaryDirectory() as tmp:
            out_dir = Path(tmp)
            _scorecard_df({"Hot peak days": 8.0, "Day 1 only": 3.5}).to_csv(
                out_dir / "production_readiness_scorecard.csv", index=False
            )
            before_path = out_dir / "before.csv"
            _scorecard_df({"Hot peak days": 8.4}).to_csv(before_path, index=False)
            buf = io.StringIO()
            with contextlib.redirect_stdout(buf):
                sys.argv = [
                    "prog",
                    "--output-dir",
                    str(out_dir),
                    "--compare-path",
                    str(before_path),
                ]
                summarize.main()
            out = buf.getvalue()
            self.assertIn("Day 1 only", out)
            self.assertIn("--", out)


if __name__ == "__main__":
    unittest.main()
