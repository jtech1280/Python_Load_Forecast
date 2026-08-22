from __future__ import annotations

import unittest

import numpy as np
import pandas as pd

from forecasting.features.record_breaking_heat import (
    add_record_breaking_heat_features,
    build_daily_max_temp_reference,
    compute_climatology_lookup,
    _circular_doy_distance,
    _trailing_percentile_for_date,
)


def _hourly_weather(dates: list[str], daily_max_by_date: dict[str, float]) -> pd.DataFrame:
    """One synthetic hourly frame: 24 hours/day, with the day's max temp reached at
    hour 16 and other hours a bit cooler, so build_daily_max_temp_reference has to
    actually take a max, not just echo a single value."""
    rows = []
    for date in dates:
        peak = daily_max_by_date[date]
        for hour in range(24):
            rows.append(
                {
                    "DT": pd.Timestamp(f"{date} {hour:02d}:00:00"),
                    "TempF": peak - abs(hour - 16) * 0.5,
                }
            )
    return pd.DataFrame(rows)


class BuildDailyMaxTempReferenceTests(unittest.TestCase):
    def test_aggregates_hourly_to_one_row_per_date_with_true_max(self):
        wx = _hourly_weather(["2020-07-04"], {"2020-07-04": 101.0})
        ref = build_daily_max_temp_reference(wx)
        self.assertEqual(len(ref), 1)
        self.assertAlmostEqual(ref.iloc[0]["Temperature_DailyMax"], 101.0)
        self.assertEqual(int(ref.iloc[0]["Year"]), 2020)

    def test_empty_or_missing_column_returns_empty_frame(self):
        self.assertTrue(build_daily_max_temp_reference(pd.DataFrame()).empty)
        self.assertTrue(
            build_daily_max_temp_reference(pd.DataFrame({"DT": [pd.Timestamp("2020-01-01")]})).empty
        )


class CircularDoyDistanceTests(unittest.TestCase):
    def test_wraps_around_year_boundary(self):
        # Dec 30 (DOY ~364) vs Jan 3 (DOY 3) should be close (4 days), not ~361 days.
        distance = _circular_doy_distance(np.array([364]), 3)
        self.assertLessEqual(distance[0], 5)


class TrailingPercentileTests(unittest.TestCase):
    def _reference(self, rows: list[tuple[int, int, float]]) -> pd.DataFrame:
        return pd.DataFrame(rows, columns=["Year", "DOY", "Temperature_DailyMax"])

    def test_only_uses_years_strictly_before_target_year(self):
        """The core leakage-safety guarantee: a huge future value must not leak into
        a past date's climatology, even though it's a perfect day-of-year match."""
        reference = self._reference(
            [(2018, 200, 90.0), (2019, 200, 91.0), (2020, 200, 92.0), (2025, 200, 130.0)]
        )
        value, n_years = _trailing_percentile_for_date(
            reference,
            target_year=2021,
            target_doy=200,
            percentile=95.0,
            window_days=5,
            min_reference_years=3,
        )
        # 2025's 130.0 must never influence a 2021 lookup.
        self.assertLess(value, 100.0)
        self.assertEqual(n_years, 3)

    def test_returns_nan_not_zero_when_reference_years_below_minimum(self):
        """The exact failure mode of the earlier, degenerate prototype: too few
        reference points should surface as missing, not a silently-wrong zero."""
        reference = self._reference([(2018, 200, 90.0), (2019, 200, 91.0)])
        value, n_years = _trailing_percentile_for_date(
            reference,
            target_year=2021,
            target_doy=200,
            percentile=95.0,
            window_days=5,
            min_reference_years=3,
        )
        self.assertTrue(np.isnan(value))
        self.assertEqual(n_years, 2)

    def test_no_prior_years_at_all_returns_nan(self):
        reference = self._reference([(2021, 200, 90.0)])
        value, n_years = _trailing_percentile_for_date(
            reference,
            target_year=2021,
            target_doy=200,
            percentile=95.0,
            window_days=5,
            min_reference_years=1,
        )
        self.assertTrue(np.isnan(value))
        self.assertEqual(n_years, 0)

    def test_day_of_year_window_excludes_far_dates(self):
        # DOY 50 is far outside a 5-day window around DOY 200.
        reference = self._reference([(2018, 50, 40.0), (2019, 200, 90.0), (2020, 200, 92.0)])
        value, n_years = _trailing_percentile_for_date(
            reference,
            target_year=2021,
            target_doy=200,
            percentile=50.0,
            window_days=5,
            min_reference_years=2,
        )
        self.assertGreater(value, 80.0)  # not dragged down by the DOY-50 row
        self.assertEqual(n_years, 2)


class ComputeClimatologyLookupTests(unittest.TestCase):
    def test_produces_one_row_per_unique_date_with_expected_columns(self):
        reference = build_daily_max_temp_reference(
            _hourly_weather(
                ["2018-07-15", "2019-07-15", "2020-07-15"],
                {"2018-07-15": 90.0, "2019-07-15": 92.0, "2020-07-15": 94.0},
            )
        )
        dates = pd.Series(pd.to_datetime(["2021-07-15", "2021-07-15", "2018-07-15"]))
        lookup = compute_climatology_lookup(
            reference,
            dates,
            {"features": {"record_breaking_heat": {"min_reference_years": 2}}},
        )
        self.assertEqual(len(lookup), 2)  # de-duplicated
        row_2021 = lookup[lookup["Date"] == pd.Timestamp("2021-07-15")].iloc[0]
        self.assertGreater(row_2021["Temp_Climatology_Reference_Years"], 0)
        row_2018 = lookup[lookup["Date"] == pd.Timestamp("2018-07-15")].iloc[0]
        self.assertTrue(np.isnan(row_2018["Climatology_Temp_PXX_F"]))  # no prior years


class AddRecordBreakingHeatFeaturesTests(unittest.TestCase):
    def _df(self) -> pd.DataFrame:
        dt = pd.date_range("2021-07-15", periods=3, freq="h")
        return pd.DataFrame(
            {
                "DT": dt,
                "Temperature_DailyMax": [96.0, 96.0, 96.0],
            }
        )

    def _lookup(self) -> pd.DataFrame:
        return pd.DataFrame(
            {
                "Date": [pd.Timestamp("2021-07-15")],
                "Climatology_Temp_PXX_F": [92.0],
                "Temp_Climatology_Reference_Years": [5],
            }
        )

    def test_disabled_is_a_true_no_op(self):
        out = add_record_breaking_heat_features(
            self._df(), self._lookup(), {"features": {"record_breaking_heat": {"enabled": False}}}
        )
        self.assertNotIn("Temp_Excess_Over_Climatology_F", out.columns)

    def test_enabled_computes_excess_over_climatology(self):
        out = add_record_breaking_heat_features(
            self._df(), self._lookup(), {"features": {"record_breaking_heat": {"enabled": True}}}
        )
        self.assertTrue((out["Temp_Excess_Over_Climatology_F"] == 4.0).all())  # 96 - 92

    def test_enabled_with_no_lookup_is_a_no_op(self):
        out = add_record_breaking_heat_features(
            self._df(), None, {"features": {"record_breaking_heat": {"enabled": True}}}
        )
        self.assertNotIn("Temp_Excess_Over_Climatology_F", out.columns)

    def test_missing_required_columns_is_a_no_op(self):
        df = pd.DataFrame({"DT": pd.date_range("2021-07-15", periods=2, freq="h")})
        out = add_record_breaking_heat_features(
            df, self._lookup(), {"features": {"record_breaking_heat": {"enabled": True}}}
        )
        self.assertNotIn("Temp_Excess_Over_Climatology_F", out.columns)


if __name__ == "__main__":
    unittest.main()
