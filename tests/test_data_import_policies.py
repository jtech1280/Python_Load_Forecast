from __future__ import annotations

import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch
from zoneinfo import ZoneInfo

import pandas as pd

from forecasting.data.history_loader import actuals_import_cutoff_dt
from forecasting.data.weather_loader import fetch_forecast_weather
from forecasting.features.intraday_load_features import (
    build_hourly_load_from_five_min,
    build_intraday_load_feature_frame,
)
from forecasting.forecast.forecast_pipeline import _filter_to_actuals_cutoff


class DataImportPolicyTests(unittest.TestCase):
    def _weather_config(self, cache_dir: Path) -> dict:
        return {
            "project": {
                "timezone": "America/Los_Angeles",
                "output_dir": str(cache_dir),
            },
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
                "cache_dir": str(cache_dir),
                "ssl_verify": True,
                "ssl_use_os_truststore": False,
                "forecast_import_policy": {
                    "enabled": True,
                    "target_local_time": "06:30",
                    "import_window_start_local": "05:30",
                    "import_window_end_local": "08:00",
                    "daily_cache_stem": "forecast_weather_morning",
                },
            },
            "quality": {
                "valid_temp_min_f": -50.0,
                "valid_temp_max_f": 130.0,
                "weather_timestamp_shift_hours": 0,
                "max_interpolation_gap_hours": 0,
            },
            "output_sql": {"enabled": False},
        }

    def test_forecast_weather_import_freezes_to_one_morning_snapshot_per_day(self):
        tz = ZoneInfo("America/Los_Angeles")
        payload = {
            "hourly": {
                "time": ["2026-08-01T00:00", "2026-08-01T01:00"],
                "temperature_2m": [71.0, 72.0],
            }
        }
        with tempfile.TemporaryDirectory() as tmp:
            cache_dir = Path(tmp)
            config = self._weather_config(cache_dir)

            with (
                patch(
                    "forecasting.data.weather_loader._now_local",
                    return_value=pd.Timestamp("2026-08-01 06:35", tz=tz),
                ),
                patch(
                    "forecasting.data.weather_loader._fetch_json", return_value=payload
                ) as fetch_json,
            ):
                first = fetch_forecast_weather(config)

            daily_cache = cache_dir / "forecast_weather_morning_2026-08-01.csv"
            self.assertTrue(daily_cache.exists())
            self.assertEqual(fetch_json.call_count, 1)
            self.assertEqual(first.attrs["weather_source"], "open_meteo_forecast")

            with (
                patch(
                    "forecasting.data.weather_loader._now_local",
                    return_value=pd.Timestamp("2026-08-01 13:00", tz=tz),
                ),
                patch(
                    "forecasting.data.weather_loader._fetch_json",
                    side_effect=AssertionError("unexpected API call"),
                ),
            ):
                second = fetch_forecast_weather(config)

            self.assertEqual(
                second.attrs["weather_source"], "forecast_weather_morning_daily_cache"
            )
            self.assertEqual(second["TempF"].tolist(), [71.0, 72.0])

    def test_forecast_weather_outside_window_reuses_latest_cache_without_api_import(
        self,
    ):
        tz = ZoneInfo("America/Los_Angeles")
        with tempfile.TemporaryDirectory() as tmp:
            cache_dir = Path(tmp)
            config = self._weather_config(cache_dir)
            pd.DataFrame(
                {
                    "DT": pd.date_range("2026-08-01 00:00", periods=2, freq="h", tz=tz),
                    "TempF": [80.0, 81.0],
                }
            ).to_csv(cache_dir / "forecast_weather_latest.csv", index=False)

            with (
                patch(
                    "forecasting.data.weather_loader._now_local",
                    return_value=pd.Timestamp("2026-08-01 13:00", tz=tz),
                ),
                patch(
                    "forecasting.data.weather_loader._fetch_json",
                    side_effect=AssertionError("unexpected API call"),
                ),
            ):
                out = fetch_forecast_weather(config)

            self.assertEqual(
                out.attrs["weather_source"],
                "forecast_weather_latest_cache_outside_morning_window",
            )
            self.assertEqual(out["TempF"].tolist(), [80.0, 81.0])

    def test_actuals_import_cutoff_uses_prior_day_completed_hour_label(self):
        config = {
            "project": {"timezone": "America/Los_Angeles"},
            "actuals_import": {
                "limit_to_prior_day_he24": True,
                "cutoff_days_back": 1,
                "cutoff_hour_ending": 24,
            },
        }

        cutoff = actuals_import_cutoff_dt(
            config, now=pd.Timestamp("2026-08-01 13:00", tz="America/Los_Angeles")
        )

        self.assertEqual(
            cutoff, pd.Timestamp("2026-07-31 23:00", tz="America/Los_Angeles")
        )

    def test_actuals_cutoff_filters_five_min_hourly_rows_after_prior_day_he24(self):
        cutoff = pd.Timestamp("2026-07-31 23:00", tz="America/Los_Angeles")
        rows = pd.DataFrame(
            {
                "DT": pd.date_range(
                    "2026-07-31 22:00", periods=4, freq="h", tz="America/Los_Angeles"
                ),
                "MWH": [100.0, 101.0, 102.0, 103.0],
            }
        )

        out = _filter_to_actuals_cutoff(rows, cutoff)

        self.assertEqual(out["DT"].tolist(), rows["DT"].iloc[:2].tolist())

    def test_completed_five_min_hourly_load_uses_completed_hour_label(self):
        tz = "America/Los_Angeles"
        rows = pd.DataFrame(
            {
                "DT": pd.date_range("2026-07-15 14:00", periods=12, freq="5min", tz=tz),
                "FiveMin_Load_MW": [300.0 + i for i in range(12)],
            }
        )

        out = build_hourly_load_from_five_min(
            rows, timezone=tz, min_intervals_per_hour=12
        )

        self.assertEqual(len(out), 1)
        self.assertEqual(out.loc[0, "DT"], pd.Timestamp("2026-07-15 15:00", tz=tz))
        self.assertEqual(
            out.loc[0, "FiveMin_Hour_End"], pd.Timestamp("2026-07-15 15:00", tz=tz)
        )
        self.assertAlmostEqual(out.loc[0, "MWH"], 305.5)

    def test_intraday_previous_hour_features_do_not_use_current_completed_hour(self):
        tz = "America/Los_Angeles"
        rows = pd.DataFrame(
            {
                "DT": list(
                    pd.date_range("2026-07-15 14:00", periods=12, freq="5min", tz=tz)
                )
                + list(
                    pd.date_range("2026-07-15 15:00", periods=12, freq="5min", tz=tz)
                ),
                "FiveMin_Load_MW": [300.0 + i for i in range(12)]
                + [400.0 + i for i in range(12)],
            }
        )

        out = build_intraday_load_feature_frame(rows)
        out["DT"] = pd.to_datetime(out["DT"], errors="coerce", utc=True).dt.tz_convert(
            tz
        )

        row = out.loc[out["DT"].eq(pd.Timestamp("2026-07-15 16:00", tz=tz))].iloc[0]

        self.assertAlmostEqual(row["FiveMin_PrevHour_Avg_MW"], 305.5)
        self.assertAlmostEqual(row["FiveMin_PrevHour_Max_MW"], 311.0)
        self.assertFalse(out["DT"].eq(pd.Timestamp("2026-07-15 15:00", tz=tz)).any())


if __name__ == "__main__":
    unittest.main()
