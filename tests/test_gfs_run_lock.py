from __future__ import annotations

import datetime as dt
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import pandas as pd

from forecasting.data.weather_loader import (
    _fetch_gfs_locked_forecast,
    fetch_forecast_weather,
)


def _config(
    *,
    enabled: bool = True,
    allow_previous_day_fallback: bool = True,
    allow_standard_forecast_fallback: bool = False,
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
            "forecast_days": 16,
            "forecast_past_hours": 0,
            "forecast_url": "https://example.test/forecast",
            "cache_dir": "/tmp/does-not-matter",
            "ssl_verify": True,
            "ssl_use_os_truststore": False,
            "gfs_run_lock": {
                "enabled": enabled,
                "single_runs_url": "https://single-runs-api.open-meteo.com/v1/forecast",
                "model": "gfs_seamless",
                "run_hour_utc": 6,
                "latest_cache_stem": "forecast_weather_gfs_06z_latest",
                "allow_import_outside_window": True,
                "allow_previous_day_fallback": allow_previous_day_fallback,
                "allow_standard_forecast_fallback": allow_standard_forecast_fallback,
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
        called_url = mock_fetch.call_args[0][0]
        called_params = mock_fetch.call_args[0][1]
        self.assertEqual(
            called_url, "https://single-runs-api.open-meteo.com/v1/forecast"
        )
        self.assertEqual(called_params["models"], "gfs_seamless")
        self.assertEqual(called_params["forecast_days"], 16)
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

    def test_enabled_but_run_unavailable_does_not_use_standard_endpoint_by_default(
        self,
    ):
        with tempfile.TemporaryDirectory() as tmp:
            config = _config(enabled=True, allow_previous_day_fallback=False)
            config["openmeteo"]["cache_dir"] = tmp
            config["project"]["output_dir"] = tmp
            config["openmeteo"]["forecast_import_policy"] = {"enabled": False}
            pd.DataFrame(
                {
                    "DT": ["2026-08-19T00:00:00-07:00"],
                    "Temperature": [77.0],
                }
            ).to_csv(f"{tmp}/forecast_results.csv", index=False)
            with (
                patch(
                    "forecasting.data.weather_loader._fetch_json",
                    return_value={"hourly": {}},
                ) as mock_fetch,
                patch(
                    "forecasting.data.weather_loader._archive_forecast_weather",
                    return_value=None,
                ),
            ):
                with self.assertRaises(RuntimeError):
                    fetch_forecast_weather(config)

        self.assertEqual(mock_fetch.call_count, 1)

    def test_enabled_but_run_unavailable_can_explicitly_fall_back_to_standard_endpoint(
        self,
    ):
        config = _config(
            enabled=True,
            allow_previous_day_fallback=False,
            allow_standard_forecast_fallback=True,
        )
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


def _payload_with_trailing_null_gap(
    *, start: str, total_hours: int, populated_hours: int
) -> dict:
    """A payload with a full 'hourly.time' array (so an empty-check alone won't catch
    it) but TempF null for every hour after `populated_hours` -- the shape of the
    2026-08-17 incident where Open-Meteo's 06Z single run returned 384 hourly rows
    through 2026-09-01 but temperature_2m was only populated through 2026-08-19.
    """
    times = [
        t.strftime("%Y-%m-%dT%H:%M")
        for t in pd.date_range(start, periods=total_hours, freq="h")
    ]
    temps = [70.0] * populated_hours + [None] * (total_hours - populated_hours)
    return _payload(times, temps)


class PartialNullForecastPayloadTests(unittest.TestCase):
    def test_fetch_gfs_locked_forecast_rejects_large_temp_gap_and_falls_back(self):
        config = _config()
        incomplete = _payload_with_trailing_null_gap(
            start="2026-08-17T00:00", total_hours=384, populated_hours=72
        )
        complete = _payload(["2026-08-16T00:00"], [90.0])
        with patch(
            "forecasting.data.weather_loader._fetch_json",
            side_effect=[incomplete, complete],
        ) as mock_fetch:
            with self.assertWarns(RuntimeWarning):
                df, run_tag = _fetch_gfs_locked_forecast(config)

        self.assertFalse(df.empty)
        self.assertFalse(df["TempF"].isna().any())
        self.assertEqual(mock_fetch.call_count, 2)
        today_utc = dt.datetime.now(dt.timezone.utc).date()
        yesterday_utc = today_utc - dt.timedelta(days=1)
        second_call_params = mock_fetch.call_args_list[1][0][1]
        self.assertEqual(
            second_call_params["run"], f"{yesterday_utc.isoformat()}T06:00"
        )

    def test_gfs_lock_never_caches_a_partially_null_payload(self):
        with tempfile.TemporaryDirectory() as tmp:
            config = _config(allow_previous_day_fallback=False)
            config["openmeteo"]["cache_dir"] = tmp
            config["project"]["output_dir"] = tmp
            config["openmeteo"]["forecast_import_policy"] = {"enabled": False}
            incomplete = _payload_with_trailing_null_gap(
                start="2026-08-17T00:00", total_hours=384, populated_hours=72
            )
            with (
                patch(
                    "forecasting.data.weather_loader._fetch_json",
                    return_value=incomplete,
                ),
                patch(
                    "forecasting.data.weather_loader._archive_forecast_weather",
                    return_value=None,
                ),
            ):
                with self.assertRaises(RuntimeError):
                    fetch_forecast_weather(config)

            # No good prior cache and standard/forecast_results fallback both disabled
            # by this config, so it must raise rather than write the null-laden frame
            # to the daily/latest cache where a later run would silently reuse it.
            self.assertFalse((Path(tmp) / "forecast_weather_gfs_06z_latest.csv").exists())

    def test_standard_forecast_with_large_temp_gap_falls_back_to_cache(self):
        with tempfile.TemporaryDirectory() as tmp:
            config = _config(enabled=False)
            config["openmeteo"]["cache_dir"] = tmp
            config["project"]["output_dir"] = tmp
            config["openmeteo"]["forecast_import_policy"] = {"enabled": False}
            good_cache = pd.DataFrame(
                {
                    "DT": ["2026-08-16T00:00:00-07:00"],
                    "TempF": [88.0],
                }
            )
            good_cache.to_csv(f"{tmp}/forecast_weather_latest.csv", index=False)
            incomplete = _payload_with_trailing_null_gap(
                start="2026-08-17T00:00", total_hours=384, populated_hours=72
            )
            with (
                patch(
                    "forecasting.data.weather_loader._fetch_json",
                    return_value=incomplete,
                ),
                patch(
                    "forecasting.data.weather_loader._archive_forecast_weather",
                    return_value=None,
                ),
            ):
                with self.assertWarns(RuntimeWarning):
                    out = fetch_forecast_weather(config)

        self.assertEqual(out.attrs["weather_source"], "forecast_weather_latest_cache")
        self.assertFalse(out["TempF"].isna().any())


if __name__ == "__main__":
    unittest.main()
