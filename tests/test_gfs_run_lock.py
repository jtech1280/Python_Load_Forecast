from __future__ import annotations

import datetime as dt
import unittest
from unittest.mock import patch

import pandas as pd

from forecasting.data.weather_loader import (
    _fetch_gfs_locked_forecast,
    fetch_forecast_weather,
)


def _config(*, enabled: bool = True, allow_previous_day_fallback: bool = True) -> dict:
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
            "forecast_days": 16,
            "forecast_past_hours": 0,
            "forecast_url": "https://example.test/forecast",
            "cache_dir": "/tmp/does-not-matter",
            "ssl_verify": True,
            "ssl_use_os_truststore": False,
            "gfs_run_lock": {
                "enabled": enabled,
                "model": "gfs_global",
                "run_hour_utc": 6,
                "allow_previous_day_fallback": allow_previous_day_fallback,
            },
        },
        "quality": {
            "valid_temp_min_f": -50.0,
            "valid_temp_max_f": 130.0,
            "weather_timestamp_shift_hours": 0,
            "max_interpolation_gap_hours": 0,
        },
    }


def _payload(times: list[str], temps: list[float]) -> dict:
    return {"hourly": {"time": times, "temperature_2m": temps}}


class FetchGfsLockedForecastTests(unittest.TestCase):
    def test_uses_todays_utc_06z_run_on_first_try(self):
        config = _config()
        payload = _payload(["2026-08-19T00:00", "2026-08-19T01:00"], [90.0, 91.0])
        with patch(
            "forecasting.data.weather_loader._fetch_json", return_value=payload
        ) as mock_fetch:
            df, run_tag = _fetch_gfs_locked_forecast(config)

        self.assertFalse(df.empty)
        self.assertEqual(mock_fetch.call_count, 1)
        called_params = mock_fetch.call_args[0][1]
        self.assertEqual(called_params["models"], "gfs_global")
        today_utc = dt.datetime.now(dt.timezone.utc).date()
        self.assertEqual(called_params["run"], f"{today_utc.isoformat()}T06:00")
        self.assertTrue(run_tag.endswith("06Z"))

    def test_falls_back_to_previous_day_when_todays_run_is_empty(self):
        config = _config()
        empty_then_real = [
            {"hourly": {}},
            _payload(["2026-08-19T00:00"], [90.0]),
        ]
        with patch(
            "forecasting.data.weather_loader._fetch_json",
            side_effect=empty_then_real,
        ) as mock_fetch:
            df, run_tag = _fetch_gfs_locked_forecast(config)

        self.assertFalse(df.empty)
        self.assertEqual(mock_fetch.call_count, 2)
        today_utc = dt.datetime.now(dt.timezone.utc).date()
        yesterday_utc = today_utc - dt.timedelta(days=1)
        second_call_params = mock_fetch.call_args_list[1][0][1]
        self.assertEqual(
            second_call_params["run"], f"{yesterday_utc.isoformat()}T06:00"
        )

    def test_no_fallback_when_disabled_in_config(self):
        config = _config(allow_previous_day_fallback=False)
        with patch(
            "forecasting.data.weather_loader._fetch_json",
            return_value={"hourly": {}},
        ) as mock_fetch:
            df, run_tag = _fetch_gfs_locked_forecast(config)

        self.assertTrue(df.empty)
        self.assertEqual(run_tag, "")
        self.assertEqual(mock_fetch.call_count, 1)

    def test_total_failure_returns_empty_without_raising(self):
        config = _config()
        with patch(
            "forecasting.data.weather_loader._fetch_json",
            side_effect=RuntimeError("network error"),
        ):
            with self.assertWarns(RuntimeWarning):
                df, run_tag = _fetch_gfs_locked_forecast(config)

        self.assertTrue(df.empty)
        self.assertEqual(run_tag, "")


class FetchForecastWeatherGfsLockWiringTests(unittest.TestCase):
    def test_disabled_by_default_uses_standard_endpoint_unchanged(self):
        config = _config(enabled=False)
        config["openmeteo"]["forecast_import_policy"] = {"enabled": False}
        payload = _payload(["2026-08-19T00:00"], [90.0])
        with (
            patch(
                "forecasting.data.weather_loader._fetch_json", return_value=payload
            ) as mock_fetch,
            patch(
                "forecasting.data.weather_loader._archive_forecast_weather",
                return_value=None,
            ),
        ):
            out = fetch_forecast_weather(config)

        self.assertEqual(out.attrs["weather_source"], "open_meteo_forecast")
        # Disabled means the run-locked path must never even attempt a request: exactly
        # one _fetch_json call (the standard endpoint), not two.
        self.assertEqual(mock_fetch.call_count, 1)

    def test_enabled_and_successful_tags_source_with_run(self):
        config = _config(enabled=True)
        config["openmeteo"]["forecast_import_policy"] = {"enabled": False}
        payload = _payload(["2026-08-19T00:00"], [90.0])
        with (
            patch("forecasting.data.weather_loader._fetch_json", return_value=payload),
            patch(
                "forecasting.data.weather_loader._archive_forecast_weather",
                return_value=None,
            ),
        ):
            out = fetch_forecast_weather(config)

        self.assertTrue(
            out.attrs["weather_source"].startswith("open_meteo_gfs_run_locked_")
        )

    def test_enabled_but_run_unavailable_falls_back_to_standard_endpoint(self):
        config = _config(enabled=True, allow_previous_day_fallback=False)
        config["openmeteo"]["forecast_import_policy"] = {"enabled": False}
        responses = [{"hourly": {}}, _payload(["2026-08-19T00:00"], [90.0])]
        with (
            patch(
                "forecasting.data.weather_loader._fetch_json",
                side_effect=responses,
            ),
            patch(
                "forecasting.data.weather_loader._archive_forecast_weather",
                return_value=None,
            ),
        ):
            with self.assertWarns(RuntimeWarning):
                out = fetch_forecast_weather(config)

        self.assertEqual(out.attrs["weather_source"], "open_meteo_forecast")


if __name__ == "__main__":
    unittest.main()
