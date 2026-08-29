from __future__ import annotations

import importlib
import sys
import unittest
from pathlib import Path

import pandas as pd

SCRIPTS_DIR = Path(__file__).resolve().parents[1] / "scripts"
if str(SCRIPTS_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPTS_DIR))

inspect_mod = importlib.import_module("inspect_historical_heat_streaks")


class FindStreaksTests(unittest.TestCase):
    def _daily(self, temps: list[float], start: str = "2026-01-01") -> pd.DataFrame:
        return pd.DataFrame(
            {
                "Date": pd.date_range(start, periods=len(temps)),
                "Temperature_DailyMax": temps,
            }
        )

    def test_finds_the_longest_streak_first(self):
        daily = self._daily([80, 96, 97, 98, 60, 95, 95, 95, 95, 95])
        out = inspect_mod.find_streaks(daily, 95.0)
        self.assertEqual(len(out), 2)
        self.assertEqual(out.iloc[0]["Length_Days"], 5)
        self.assertEqual(out.iloc[1]["Length_Days"], 3)
        self.assertEqual(out.iloc[0]["Max_Temp_F"], 95.0)
        self.assertEqual(out.iloc[1]["Max_Temp_F"], 98.0)

    def test_a_calendar_gap_breaks_a_streak_even_if_both_sides_are_hot(self):
        daily = pd.DataFrame(
            {
                "Date": [
                    pd.Timestamp("2026-01-01"),
                    pd.Timestamp("2026-01-02"),
                    pd.Timestamp("2026-01-05"),  # gap: Jan 3-4 missing
                    pd.Timestamp("2026-01-06"),
                ],
                "Temperature_DailyMax": [96, 97, 98, 99],
            }
        )
        out = inspect_mod.find_streaks(daily, 95.0)
        self.assertEqual(len(out), 2)
        self.assertTrue((out["Length_Days"] == 2).all())

    def test_no_qualifying_days_returns_empty(self):
        daily = self._daily([60, 65, 70])
        out = inspect_mod.find_streaks(daily, 95.0)
        self.assertTrue(out.empty)

    def test_empty_or_none_input_returns_empty(self):
        self.assertTrue(inspect_mod.find_streaks(pd.DataFrame(), 95.0).empty)
        self.assertTrue(inspect_mod.find_streaks(None, 95.0).empty)

    def test_single_hot_day_is_a_length_one_streak(self):
        daily = self._daily([60, 96, 60])
        out = inspect_mod.find_streaks(daily, 95.0)
        self.assertEqual(len(out), 1)
        self.assertEqual(out.iloc[0]["Length_Days"], 1)

    def test_threshold_is_inclusive(self):
        daily = self._daily([95.0])
        out = inspect_mod.find_streaks(daily, 95.0)
        self.assertEqual(len(out), 1)


if __name__ == "__main__":
    unittest.main()
