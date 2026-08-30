from __future__ import annotations

import importlib
import sys
import unittest
from pathlib import Path

import pandas as pd

SCRIPTS_DIR = Path(__file__).resolve().parents[1] / "scripts"
if str(SCRIPTS_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPTS_DIR))

inspect_mod = importlib.import_module("inspect_persistence_depth_training_sparsity")


class RowCountHistogramTests(unittest.TestCase):
    def test_buckets_days_by_depth(self):
        daily = pd.DataFrame({"ConsecutiveVeryHotDays95": [0, 0, 1, 2, 3, 5, 6, 14, 14, 14]})
        hist = inspect_mod.row_count_histogram(daily)
        as_dict = dict(zip(hist["Bucket"], hist["N_Days"]))
        self.assertEqual(as_dict["0 (not hot)"], 2)
        self.assertEqual(as_dict["1-3"], 3)
        self.assertEqual(as_dict["4-7"], 2)
        self.assertEqual(as_dict["8-12"], 0)
        self.assertEqual(as_dict["13-18"], 3)
        self.assertEqual(as_dict["19+"], 0)

    def test_missing_values_treated_as_zero(self):
        daily = pd.DataFrame({"ConsecutiveVeryHotDays95": [None, 1.0]})
        hist = inspect_mod.row_count_histogram(daily)
        as_dict = dict(zip(hist["Bucket"], hist["N_Days"]))
        self.assertEqual(as_dict["0 (not hot)"], 1)


class IndependentEpisodeHistogramTests(unittest.TestCase):
    def test_counts_streaks_reaching_each_depth_not_total_days(self):
        streaks = pd.DataFrame({"Length_Days": [18, 18, 2, 5]})
        hist = inspect_mod.independent_episode_histogram(streaks)
        as_dict = dict(zip(hist["Bucket"], hist["N_Independent_Streaks_Reaching_Depth"]))
        self.assertEqual(as_dict["1+"], 4)
        self.assertEqual(as_dict["4+"], 3)
        self.assertEqual(as_dict["8+"], 2)
        self.assertEqual(as_dict["13+"], 2)
        self.assertEqual(as_dict["19+"], 0)

    def test_empty_streaks_returns_empty(self):
        hist = inspect_mod.independent_episode_histogram(pd.DataFrame())
        self.assertTrue(hist.empty)


if __name__ == "__main__":
    unittest.main()
