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

describe_mod = importlib.import_module("describe_cloud_solar_midday_coverage")


def _replay_df(n_rows: int = 20) -> pd.DataFrame:
    hour = np.array([10, 11, 12, 13, 14, 15, 16, 9, 17, 20] * 2, dtype=float)[:n_rows]
    cloud = np.linspace(0.0, 1.0, n_rows)
    loss = np.linspace(0.0, 3.0, n_rows)
    origin = np.array([1, 1, 1, 2, 2, 2, 3, 3, 3, 3] * 2)[:n_rows]
    return pd.DataFrame(
        {
            "Hour": hour,
            "CloudCover_Norm": cloud,
            "BTM_Solar_Loss_From_ClearSky_MW": loss,
            "Replay_Origin_ID": origin,
        }
    )


class DescribeCloudSolarMiddayCoverageTests(unittest.TestCase):
    def test_reports_candidate_window_and_qualifying_counts(self):
        df = _replay_df()
        hour = df["Hour"]
        candidate = hour.between(10, 16)
        cloud = df["CloudCover_Norm"]
        loss = df["BTM_Solar_Loss_From_ClearSky_MW"]
        expected_qualifying = int((candidate & (cloud.ge(0.60) | loss.ge(1.25))).sum())

        buf = []
        import contextlib
        import io

        stdout = io.StringIO()
        with contextlib.redirect_stdout(stdout):
            describe_mod.describe(df)
        output = stdout.getvalue()
        self.assertIn(f"Rows in the hour-10-16 candidate window: {int(candidate.sum())}", output)
        self.assertIn(f"{expected_qualifying} (", output)

    def test_no_qualifying_rows_does_not_crash(self):
        df = pd.DataFrame(
            {
                "Hour": [10.0, 11.0, 12.0],
                "CloudCover_Norm": [0.1, 0.05, 0.2],
                "BTM_Solar_Loss_From_ClearSky_MW": [0.1, 0.2, 0.3],
                "Replay_Origin_ID": [1, 1, 1],
            }
        )
        import contextlib
        import io

        stdout = io.StringIO()
        with contextlib.redirect_stdout(stdout):
            describe_mod.describe(df)
        self.assertIn("No qualifying rows", stdout.getvalue())

    def test_no_rows_in_candidate_window_does_not_crash(self):
        df = pd.DataFrame(
            {
                "Hour": [0.0, 1.0, 2.0],
                "CloudCover_Norm": [0.9, 0.9, 0.9],
                "BTM_Solar_Loss_From_ClearSky_MW": [2.0, 2.0, 2.0],
                "Replay_Origin_ID": [1, 1, 1],
            }
        )
        import contextlib
        import io

        stdout = io.StringIO()
        with contextlib.redirect_stdout(stdout):
            describe_mod.describe(df)
        self.assertIn("nothing to describe", stdout.getvalue())

    def test_top3_origin_concentration_flags_a_single_dominant_origin(self):
        # All qualifying rows come from one origin -- the concentration warning should
        # report 100%.
        df = pd.DataFrame(
            {
                "Hour": [12.0] * 5,
                "CloudCover_Norm": [0.9] * 5,
                "BTM_Solar_Loss_From_ClearSky_MW": [0.0] * 5,
                "Replay_Origin_ID": [1, 1, 1, 1, 1],
            }
        )
        import contextlib
        import io

        stdout = io.StringIO()
        with contextlib.redirect_stdout(stdout):
            describe_mod.describe(df)
        self.assertIn("Top 3 origins account for 100.0%", stdout.getvalue())


if __name__ == "__main__":
    unittest.main()
