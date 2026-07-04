import os
import sys
import tempfile
import time
import unittest
from datetime import date
from pathlib import Path
from unittest.mock import patch

import pandas as pd
import requests


PROJECT_ROOT = Path(__file__).resolve().parents[1]
SOLAR_DIR = PROJECT_ROOT / "forecasting" / "solar"
if str(SOLAR_DIR) not in sys.path:
    sys.path.insert(0, str(SOLAR_DIR))

import solar_forecaster


class _FakeResponse:
    def __init__(self, payload):
        self._payload = payload

    def raise_for_status(self):
        return None

    def json(self):
        return self._payload


class SolarForecasterRetryTests(unittest.TestCase):
    def test_open_meteo_get_json_retries_transient_connection_reset(self):
        calls = {"count": 0}

        def fake_get(*_args, **_kwargs):
            calls["count"] += 1
            if calls["count"] == 1:
                raise requests.ConnectionError("connection reset")
            return _FakeResponse({"hourly": {"time": []}})

        with patch.object(solar_forecaster.requests, "get", side_effect=fake_get), patch.object(
            solar_forecaster.time,
            "sleep",
            return_value=None,
        ):
            payload = solar_forecaster._open_meteo_get_json(
                "https://api.open-meteo.com/v1/forecast",
                {"latitude": "1", "longitude": "2"},
                source_name="forecast",
                start_date=date(2026, 7, 3),
                end_date=date(2026, 7, 18),
            )

        self.assertEqual(payload, {"hourly": {"time": []}})
        self.assertEqual(calls["count"], 2)

    def test_hourly_forecast_weather_reuses_fresh_cache_without_api_call(self):
        sites = pd.DataFrame(
            [{"SolarSiteKey": 1, "Latitude": 38.7522, "Longitude": -121.2880}]
        )
        payload = {
            "hourly": {
                "time": ["2026-07-03T00:00", "2026-07-03T01:00"],
                "shortwave_radiation": [0.0, 100.0],
                "cloud_cover": [10.0, 20.0],
                "cloud_cover_low": [1.0, 2.0],
                "cloud_cover_mid": [3.0, 4.0],
                "cloud_cover_high": [5.0, 6.0],
            },
            "hourly_units": {"shortwave_radiation": "W/m2"},
        }

        with tempfile.TemporaryDirectory() as tmp:
            with patch.object(solar_forecaster, "_open_meteo_get_json", return_value=payload) as fetch:
                first = solar_forecaster.fetch_open_meteo_hourly_weather(
                    sites,
                    date(2026, 7, 3),
                    date(2026, 7, 3),
                    use_forecast=True,
                    timezone_name="America/Los_Angeles",
                    cache_dir=tmp,
                    forecast_cache_max_age_hours=24.0,
                )

            self.assertEqual(fetch.call_count, 1)
            self.assertEqual(len(first), 2)

            with patch.object(
                solar_forecaster,
                "_open_meteo_get_json",
                side_effect=AssertionError("API should not be called for fresh cache"),
            ):
                second = solar_forecaster.fetch_open_meteo_hourly_weather(
                    sites,
                    date(2026, 7, 3),
                    date(2026, 7, 3),
                    use_forecast=True,
                    timezone_name="America/Los_Angeles",
                    cache_dir=tmp,
                    forecast_cache_max_age_hours=24.0,
                )

            self.assertEqual(len(second), 2)
            self.assertEqual(float(second.loc[1, "GHI_kWh_per_m2"]), 0.1)

    def test_hourly_forecast_weather_uses_stale_cache_after_api_failure(self):
        sites = pd.DataFrame(
            [{"SolarSiteKey": 1, "Latitude": 38.7522, "Longitude": -121.2880}]
        )
        payload = {
            "hourly": {
                "time": ["2026-07-03T00:00"],
                "shortwave_radiation": [50.0],
                "cloud_cover": [25.0],
                "cloud_cover_low": [5.0],
                "cloud_cover_mid": [10.0],
                "cloud_cover_high": [15.0],
            },
            "hourly_units": {"shortwave_radiation": "W/m2"},
        }

        with tempfile.TemporaryDirectory() as tmp:
            with patch.object(solar_forecaster, "_open_meteo_get_json", return_value=payload):
                solar_forecaster.fetch_open_meteo_hourly_weather(
                    sites,
                    date(2026, 7, 3),
                    date(2026, 7, 3),
                    use_forecast=True,
                    timezone_name="America/Los_Angeles",
                    cache_dir=tmp,
                    forecast_cache_max_age_hours=24.0,
                )

            for path in Path(tmp).glob("solar_hourly_forecast_*.csv"):
                old = time.time() - 48 * 3600
                os.utime(path, (old, old))

            with patch.object(
                solar_forecaster,
                "_open_meteo_get_json",
                side_effect=RuntimeError("api unavailable"),
            ):
                cached = solar_forecaster.fetch_open_meteo_hourly_weather(
                    sites,
                    date(2026, 7, 3),
                    date(2026, 7, 3),
                    use_forecast=True,
                    timezone_name="America/Los_Angeles",
                    cache_dir=tmp,
                    forecast_cache_max_age_hours=1.0,
                )

            self.assertEqual(len(cached), 1)
            self.assertEqual(float(cached.loc[0, "GHI_kWh_per_m2"]), 0.05)


if __name__ == "__main__":
    unittest.main()
