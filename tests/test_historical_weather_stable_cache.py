from __future__ import annotations

import datetime as dt
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import pandas as pd

from forecasting.data.weather_loader import fetch_historical_weather


def _config(
    *,
    tmp: str,
    historical_start: str,
    historical_end: str,
    revision_window_days: int = 8,
) -> dict:
    return {
        "project": {"timezone": "America/Los_Angeles"},
        "openmeteo": {
            "latitude": 38.7521,
            "longitude": -121.2880,
            "hourly_vars": ["temperature_2m"],
            "timezone": "America/Los_Angeles",
            "temperature_unit": "fahrenheit",
            "wind_speed_unit": "mph",
            "precipitation_unit": "inch",
            "historical_url": "https://example.test/archive",
            "historical_start": historical_start,
            "historical_end": historical_end,
            "historical_revision_window_days": revision_window_days,
            "cache_dir": tmp,
            "ssl_verify": True,
            "ssl_use_os_truststore": False,
        },
        "quality": {
            "valid_temp_min_f": -50.0,
            "valid_temp_max_f": 130.0,
            "weather_timestamp_shift_hours": 0,
            "max_interpolation_gap_hours": 0,
        },
        "project_output_dir": tmp,
    }


def _payload_for_range(start: str, end: str, temp: float = 80.0) -> dict:
    times = [
        t.strftime("%Y-%m-%dT%H:%M")
        for t in pd.date_range(start, end, freq="h", inclusive="both")
    ]
    return {"hourly": {"time": times, "temperature_2m": [temp] * len(times)}}


class HistoricalWeatherStableCacheTests(unittest.TestCase):
    def test_first_fetch_splits_into_stable_and_volatile_tail_calls(self):
        with tempfile.TemporaryDirectory() as tmp:
            config = _config(
                tmp=tmp,
                historical_start="2026-01-01",
                historical_end="2026-01-20",
                revision_window_days=8,
            )
            responses = [
                _payload_for_range("2026-01-01T00:00", "2026-01-12T23:00", temp=70.0),
                _payload_for_range("2026-01-13T00:00", "2026-01-20T23:00", temp=75.0),
            ]
            with patch(
                "forecasting.data.weather_loader._fetch_json", side_effect=responses
            ) as mock_fetch:
                df = fetch_historical_weather(config)

            self.assertEqual(mock_fetch.call_count, 2)
            first_params = mock_fetch.call_args_list[0][0][1]
            second_params = mock_fetch.call_args_list[1][0][1]
            self.assertEqual(first_params["start_date"], "2026-01-01")
            self.assertEqual(first_params["end_date"], "2026-01-12")
            self.assertEqual(second_params["start_date"], "2026-01-13")
            self.assertEqual(second_params["end_date"], "2026-01-20")
            self.assertEqual(len(df), 20 * 24)

            stable_cache = Path(tmp) / "historical_weather_stable_from_2026-01-01.csv"
            self.assertTrue(stable_cache.exists())
            cached = pd.read_csv(stable_cache)
            self.assertEqual(len(cached), 12 * 24)

    def test_second_run_next_day_only_fetches_the_new_gap_and_tail(self):
        with tempfile.TemporaryDirectory() as tmp:
            config_day1 = _config(
                tmp=tmp,
                historical_start="2026-01-01",
                historical_end="2026-01-20",
                revision_window_days=8,
            )
            with patch(
                "forecasting.data.weather_loader._fetch_json",
                side_effect=[
                    _payload_for_range("2026-01-01T00:00", "2026-01-12T23:00", temp=70.0),
                    _payload_for_range("2026-01-13T00:00", "2026-01-20T23:00", temp=75.0),
                ],
            ):
                fetch_historical_weather(config_day1)

            # Day 2: end advances by one day. The stable cutoff advances by one day
            # too (2026-01-13), so the stable cache only needs one new day (01-13)
            # appended, not a full re-fetch of 01-01..01-13.
            config_day2 = _config(
                tmp=tmp,
                historical_start="2026-01-01",
                historical_end="2026-01-21",
                revision_window_days=8,
            )
            with patch(
                "forecasting.data.weather_loader._fetch_json",
                side_effect=[
                    _payload_for_range("2026-01-13T00:00", "2026-01-13T23:00", temp=71.0),
                    _payload_for_range("2026-01-14T00:00", "2026-01-21T23:00", temp=76.0),
                ],
            ) as mock_fetch_day2:
                df_day2 = fetch_historical_weather(config_day2)

            self.assertEqual(mock_fetch_day2.call_count, 2)
            gap_params = mock_fetch_day2.call_args_list[0][0][1]
            tail_params = mock_fetch_day2.call_args_list[1][0][1]
            self.assertEqual(gap_params["start_date"], "2026-01-13")
            self.assertEqual(gap_params["end_date"], "2026-01-13")
            self.assertEqual(tail_params["start_date"], "2026-01-14")
            self.assertEqual(tail_params["end_date"], "2026-01-21")

            # The already-settled 01-01..01-12 values must be untouched (still 70.0),
            # not silently overwritten by a fresh archive re-pull -- that's the whole
            # point of caching the stable portion permanently.
            df_day2["DT"] = pd.to_datetime(df_day2["DT"])
            settled = df_day2[df_day2["DT"].dt.date < dt.date(2026, 1, 13)]
            self.assertTrue((settled["TempF"] == 70.0).all())

    def test_short_range_below_revision_window_treats_everything_as_volatile(self):
        with tempfile.TemporaryDirectory() as tmp:
            config = _config(
                tmp=tmp,
                historical_start="2026-01-10",
                historical_end="2026-01-12",
                revision_window_days=8,
            )
            with patch(
                "forecasting.data.weather_loader._fetch_json",
                side_effect=[
                    _payload_for_range("2026-01-10T00:00", "2026-01-12T23:00", temp=80.0)
                ],
            ) as mock_fetch:
                df = fetch_historical_weather(config)

            self.assertEqual(mock_fetch.call_count, 1)
            called_params = mock_fetch.call_args_list[0][0][1]
            self.assertEqual(called_params["start_date"], "2026-01-10")
            self.assertEqual(called_params["end_date"], "2026-01-12")
            self.assertEqual(len(df), 3 * 24)
            stable_cache = Path(tmp) / "historical_weather_stable_from_2026-01-10.csv"
            self.assertFalse(stable_cache.exists())

    def test_falls_back_to_latest_cache_when_gap_fetch_fails(self):
        with tempfile.TemporaryDirectory() as tmp:
            config_day1 = _config(
                tmp=tmp,
                historical_start="2026-01-01",
                historical_end="2026-01-20",
                revision_window_days=8,
            )
            with patch(
                "forecasting.data.weather_loader._fetch_json",
                side_effect=[
                    _payload_for_range("2026-01-01T00:00", "2026-01-12T23:00", temp=70.0),
                    _payload_for_range("2026-01-13T00:00", "2026-01-20T23:00", temp=75.0),
                ],
            ):
                fetch_historical_weather(config_day1)

            config_day2 = _config(
                tmp=tmp,
                historical_start="2026-01-01",
                historical_end="2026-01-21",
                revision_window_days=8,
            )
            with patch(
                "forecasting.data.weather_loader._fetch_json",
                side_effect=RuntimeError("network error"),
            ):
                with self.assertWarns(RuntimeWarning):
                    df = fetch_historical_weather(config_day2)

            self.assertFalse(df.empty)


if __name__ == "__main__":
    unittest.main()
