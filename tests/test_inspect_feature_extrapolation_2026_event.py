from __future__ import annotations

import importlib
import sys
import unittest
from pathlib import Path

import pandas as pd

SCRIPTS_DIR = Path(__file__).resolve().parents[1] / "scripts"
if str(SCRIPTS_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPTS_DIR))

inspect_mod = importlib.import_module("inspect_feature_extrapolation_2026_event")


class BuildDailyFeaturesTests(unittest.TestCase):
    def test_collapses_hourly_weather_to_one_row_per_date_with_persistence_features(self):
        dt = pd.date_range("2026-07-24", periods=4 * 24, freq="h")
        temps = []
        for day in range(4):
            temps.extend([70.0 + day] * 12 + [96.0 + day] * 12)
        weather_df = pd.DataFrame({"DT": dt, "TempF": temps})

        daily = inspect_mod.build_daily_features(weather_df)
        self.assertEqual(len(daily), 4)
        self.assertTrue((daily["ConsecutiveVeryHotDays95"] == [1.0, 2.0, 3.0, 4.0]).all())
        self.assertAlmostEqual(daily.iloc[-1]["Temperature_DailyMax"], 99.0)


class FindNewRecordsTests(unittest.TestCase):
    def _daily(self, dates: list[str], max_temp: list[float]) -> pd.DataFrame:
        return pd.DataFrame(
            {
                "Date": pd.to_datetime(dates),
                "Temperature_DailyMax": max_temp,
                "DailyMaxTemp_3DayMean": max_temp,
                "ConsecutiveHotDays90": [1.0] * len(dates),
                "ConsecutiveVeryHotDays95": [1.0] * len(dates),
                "ConsecutiveExtremeHotDays100": [0.0] * len(dates),
            }
        )

    def test_event_value_above_prior_history_is_flagged_a_new_record(self):
        daily = self._daily(
            ["2020-07-01", "2021-07-01", "2026-07-27"],
            [95.0, 97.0, 99.0],
        )
        out = inspect_mod.find_new_records(
            daily, pd.Timestamp("2026-07-27"), pd.Timestamp("2026-07-27")
        )
        row = out[out["Feature"] == "Temperature_DailyMax"].iloc[0]
        self.assertTrue(row["New_Record"])
        self.assertEqual(row["Historical_Max_Before_Event"], 97.0)

    def test_event_value_within_prior_history_is_not_flagged(self):
        daily = self._daily(
            ["2020-07-01", "2021-07-01", "2026-07-27"],
            [95.0, 105.0, 99.0],
        )
        out = inspect_mod.find_new_records(
            daily, pd.Timestamp("2026-07-27"), pd.Timestamp("2026-07-27")
        )
        row = out[out["Feature"] == "Temperature_DailyMax"].iloc[0]
        self.assertFalse(row["New_Record"])

    def test_empty_daily_frame_returns_empty(self):
        daily = self._daily([], [])
        out = inspect_mod.find_new_records(
            daily, pd.Timestamp("2026-07-27"), pd.Timestamp("2026-07-27")
        )
        self.assertTrue(out.empty)


if __name__ == "__main__":
    unittest.main()
