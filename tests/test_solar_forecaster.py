import os
import sys
import tempfile
import time
import unittest
from datetime import date
from pathlib import Path
from unittest.mock import patch

import numpy as np
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

        with patch.object(
            solar_forecaster.requests, "get", side_effect=fake_get
        ), patch.object(
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
            with patch.object(
                solar_forecaster, "_open_meteo_get_json", return_value=payload
            ) as fetch:
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
            with patch.object(
                solar_forecaster, "_open_meteo_get_json", return_value=payload
            ):
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

    def test_hourly_weather_range_reuses_overlapping_disk_caches_without_api_call(self):
        sites = pd.DataFrame(
            [{"SolarSiteKey": 1, "Latitude": 38.7522, "Longitude": -121.2880}]
        )

        with tempfile.TemporaryDirectory() as tmp:
            first_path = solar_forecaster._solar_weather_cache_path(
                cache_dir=tmp,
                kind="hourly",
                source_name="historical",
                start_date=date(2026, 6, 1),
                end_date=date(2026, 6, 2),
                sites=sites,
                timezone_name="America/Los_Angeles",
                variables=solar_forecaster.HOURLY_WEATHER_VARIABLES,
            )
            second_path = solar_forecaster._solar_weather_cache_path(
                cache_dir=tmp,
                kind="hourly",
                source_name="historical",
                start_date=date(2026, 6, 3),
                end_date=date(2026, 6, 3),
                sites=sites,
                timezone_name="America/Los_Angeles",
                variables=solar_forecaster.HOURLY_WEATHER_VARIABLES,
            )
            solar_forecaster._write_solar_weather_cache(
                pd.DataFrame(
                    {
                        "SolarSiteKey": [1, 1],
                        "IntervalStartDT": [
                            "2026-06-01 12:00:00",
                            "2026-06-02 12:00:00",
                        ],
                        "GHI_kWh_per_m2": [0.80, 0.85],
                    }
                ),
                first_path,
            )
            solar_forecaster._write_solar_weather_cache(
                pd.DataFrame(
                    {
                        "SolarSiteKey": [1],
                        "IntervalStartDT": ["2026-06-03 12:00:00"],
                        "GHI_kWh_per_m2": [0.90],
                    }
                ),
                second_path,
            )

            with patch.object(
                solar_forecaster,
                "_open_meteo_get_json",
                side_effect=AssertionError("API should not be called"),
            ):
                out = solar_forecaster.fetch_hourly_weather_for_date_range(
                    sites,
                    date(2026, 6, 1),
                    date(2026, 6, 3),
                    timezone_name="America/Los_Angeles",
                    cache_dir=tmp,
                )

        self.assertEqual(len(out), 3)
        self.assertEqual(
            set(out["date"]),
            {date(2026, 6, 1), date(2026, 6, 2), date(2026, 6, 3)},
        )
        self.assertAlmostEqual(float(out["GHI_kWh_per_m2"].max()), 0.90)

    def test_hourly_weather_range_chunks_missing_api_fetches(self):
        sites = pd.DataFrame(
            [{"SolarSiteKey": 1, "Latitude": 38.7522, "Longitude": -121.2880}]
        )
        calls = []

        def fake_fetch(
            _sites,
            start_date,
            end_date,
            use_forecast,
            _timezone_name,
            **_kwargs,
        ):
            calls.append((start_date, end_date, use_forecast))
            return pd.DataFrame(
                {
                    "SolarSiteKey": [1 for _ in pd.date_range(start_date, end_date)],
                    "IntervalStartDT": [
                        pd.Timestamp(day) + pd.Timedelta(hours=12)
                        for day in pd.date_range(start_date, end_date)
                    ],
                    "GHI_kWh_per_m2": [0.8 for _ in pd.date_range(start_date, end_date)],
                }
            )

        with tempfile.TemporaryDirectory() as tmp, patch.object(
            solar_forecaster,
            "fetch_open_meteo_hourly_weather",
            side_effect=fake_fetch,
        ):
            out = solar_forecaster.fetch_hourly_weather_for_date_range(
                sites,
                date(2026, 6, 1),
                date(2026, 6, 17),
                timezone_name="America/Los_Angeles",
                cache_dir=tmp,
            )

        self.assertEqual(
            calls,
            [
                (date(2026, 6, 1), date(2026, 6, 16), False),
                (date(2026, 6, 17), date(2026, 6, 17), False),
            ],
        )
        self.assertEqual(len(out), 17)

    def test_interval_forecast_preserves_hourly_output_contract_with_model_columns(
        self,
    ):
        weather = pd.DataFrame(
            {
                "IntervalStartDT": pd.date_range(
                    "2026-07-05 12:00", periods=2, freq="h"
                ),
                "GHI_kWh_per_m2": [0.8, 0.7],
                "WeatherGHI_Wm2": [800.0, 700.0],
                "CloudCoverPct": [10.0, 20.0],
                "CloudCoverLowPct": [5.0, 10.0],
                "CloudCoverMidPct": [2.0, 5.0],
                "CloudCoverHighPct": [1.0, 3.0],
            }
        )
        intrahour_shape = pd.DataFrame(
            {
                "hour": [12, 12, 12, 12, 13, 13, 13, 13],
                "minute": [0, 15, 30, 45, 0, 15, 30, 45],
                "IntraHourCoefficient": [0.25] * 8,
            }
        )
        model = solar_forecaster.PerformanceModel(
            estimator=None,
            fallback_ratio=0.75,
            feature_columns=solar_forecaster.PERFORMANCE_FEATURE_COLUMNS,
        )

        interval = solar_forecaster.build_interval_forecast(
            weather,
            intrahour_shape,
            1000.0,
            model,
            solar_forecaster.ROSEVILLE_LATITUDE,
            solar_forecaster.ROSEVILLE_LONGITUDE,
            "America/Los_Angeles",
            0.0,
        )
        hourly = solar_forecaster.resample_interval_forecast_to_hourly(interval, 1000.0)

        self.assertIn("Forecast_MW", hourly.columns)
        self.assertIn("BaseForecast_MW", hourly.columns)
        self.assertIn("PerformanceRatio", hourly.columns)
        self.assertIn("TotalCalibrationFactor", hourly.columns)
        self.assertAlmostEqual(float(hourly.loc[0, "Forecast_MW"]), 0.6)
        self.assertAlmostEqual(float(hourly.loc[0, "BaseForecast_MW"]), 0.6)


class SolarForecasterFeatureAndCapacityTests(unittest.TestCase):
    def test_new_weather_features_are_registered(self):
        weather_columns = [
            "DirectRadiation_Wm2",
            "DiffuseRadiation_Wm2",
            "Temperature_C",
            "WindSpeed_ms",
        ]
        for column in weather_columns:
            self.assertIn(column, solar_forecaster.WEATHER_OUTPUT_COLUMNS)

        for column in [
            *weather_columns,
            "ClearSkyGHI_Wm2",
            "ClearSkyIndex",
        ]:
            self.assertIn(column, solar_forecaster.PERFORMANCE_FEATURE_COLUMNS)
        for variable in [
            "direct_radiation",
            "diffuse_radiation",
            "temperature_2m",
            "wind_speed_10m",
        ]:
            self.assertIn(variable, solar_forecaster.HOURLY_WEATHER_VARIABLES)

    def test_hourly_weather_parses_temperature_and_beam_diffuse_features(self):
        sites = pd.DataFrame(
            [{"SolarSiteKey": 1, "Latitude": 38.7522, "Longitude": -121.2880}]
        )
        payload = {
            "hourly": {
                "time": ["2026-07-03T12:00"],
                "shortwave_radiation": [800.0],
                "direct_radiation": [600.0],
                "diffuse_radiation": [180.0],
                "temperature_2m": [37.5],
                "wind_speed_10m": [3.2],
                "cloud_cover": [12.0],
                "cloud_cover_low": [4.0],
                "cloud_cover_mid": [3.0],
                "cloud_cover_high": [1.0],
            },
            "hourly_units": {"shortwave_radiation": "W/m2"},
        }

        with tempfile.TemporaryDirectory() as tmp:
            with patch.object(
                solar_forecaster, "_open_meteo_get_json", return_value=payload
            ):
                out = solar_forecaster.fetch_open_meteo_hourly_weather(
                    sites,
                    date(2026, 7, 3),
                    date(2026, 7, 3),
                    use_forecast=True,
                    timezone_name="America/Los_Angeles",
                    cache_dir=tmp,
                    forecast_cache_max_age_hours=24.0,
                )

        row = out.iloc[0]
        self.assertEqual(float(row["Temperature_C"]), 37.5)
        self.assertEqual(float(row["DirectRadiation_Wm2"]), 600.0)
        self.assertEqual(float(row["DiffuseRadiation_Wm2"]), 180.0)
        self.assertEqual(float(row["WindSpeed_ms"]), 3.2)

    def test_build_daily_active_capacity_reflects_interconnection_growth(self):
        sites = pd.DataFrame(
            [
                {"SolarCECkW": 100.0, "InterconnectionDate": "2025-01-10"},
                {"SolarCECkW": 50.0, "InterconnectionDate": "2025-01-20"},
                {"SolarCECkW": 25.0, "InterconnectionDate": None},
            ]
        )
        daily = solar_forecaster.build_daily_active_capacity(
            sites, date(2025, 1, 1), date(2025, 1, 31)
        )
        self.assertEqual(len(daily), 31)
        by_date = daily.set_index("Date")["ActiveCapacity_kW"]
        self.assertEqual(float(by_date.loc[date(2025, 1, 1)]), 25.0)
        self.assertEqual(float(by_date.loc[date(2025, 1, 9)]), 25.0)
        self.assertEqual(float(by_date.loc[date(2025, 1, 10)]), 125.0)
        self.assertEqual(float(by_date.loc[date(2025, 1, 19)]), 125.0)
        self.assertEqual(float(by_date.loc[date(2025, 1, 20)]), 175.0)
        self.assertEqual(float(by_date.loc[date(2025, 1, 31)]), 175.0)

    def test_build_daily_active_capacity_without_dates_is_flat(self):
        sites = pd.DataFrame([{"SolarCECkW": 100.0}, {"SolarCECkW": 40.0}])
        daily = solar_forecaster.build_daily_active_capacity(
            sites, date(2025, 6, 1), date(2025, 6, 3)
        )
        self.assertEqual(len(daily), 3)
        self.assertTrue((daily["ActiveCapacity_kW"] == 140.0).all())

    def test_resolve_row_capacity_uses_daily_series_and_fallback(self):
        daily = pd.DataFrame(
            {"Date": [date(2025, 6, 1)], "ActiveCapacity_kW": [1234.0]}
        )
        timestamps = pd.Series(pd.to_datetime(["2025-06-01 09:00", "2025-06-05 09:00"]))
        resolved = solar_forecaster._resolve_row_capacity(timestamps, daily, 9999.0)
        self.assertEqual(float(resolved.iloc[0]), 1234.0)
        self.assertEqual(float(resolved.iloc[1]), 9999.0)

        resolved_none = solar_forecaster._resolve_row_capacity(timestamps, None, 7.0)
        self.assertTrue((resolved_none == 7.0).all())

    def test_build_interval_forecast_scales_by_daily_active_capacity(self):
        weather = pd.DataFrame(
            {
                "IntervalStartDT": [
                    pd.Timestamp("2025-06-01 12:00"),
                    pd.Timestamp("2025-06-02 12:00"),
                ],
                "GHI_kWh_per_m2": [1.0, 1.0],
                "WeatherGHI_Wm2": [1000.0, 1000.0],
                "CloudCoverPct": [0.0, 0.0],
                "CloudCoverLowPct": [0.0, 0.0],
                "CloudCoverMidPct": [0.0, 0.0],
                "CloudCoverHighPct": [0.0, 0.0],
            }
        )
        intrahour_shape = pd.DataFrame(
            {
                "hour": [12, 12, 12, 12],
                "minute": [0, 15, 30, 45],
                "IntraHourCoefficient": [0.25, 0.25, 0.25, 0.25],
            }
        )
        model = solar_forecaster.PerformanceModel(
            estimator=None,
            fallback_ratio=0.5,
            feature_columns=solar_forecaster.PERFORMANCE_FEATURE_COLUMNS,
        )
        daily_capacity = pd.DataFrame(
            {
                "Date": [date(2025, 6, 1), date(2025, 6, 2)],
                "ActiveCapacity_kW": [1000.0, 2000.0],
            }
        )

        interval = solar_forecaster.build_interval_forecast(
            weather,
            intrahour_shape,
            5000.0,
            model,
            solar_forecaster.ROSEVILLE_LATITUDE,
            solar_forecaster.ROSEVILLE_LONGITUDE,
            "America/Los_Angeles",
            0.0,
            daily_active_capacity=daily_capacity,
        )
        interval["Date"] = interval["IntervalStartDT"].dt.date
        day1 = float(
            interval.loc[interval["Date"] == date(2025, 6, 1), "Forecast_kWh"].sum()
        )
        day2 = float(
            interval.loc[interval["Date"] == date(2025, 6, 2), "Forecast_kWh"].sum()
        )
        self.assertAlmostEqual(day1, 500.0, places=3)
        self.assertAlmostEqual(day2, 1000.0, places=3)

    def test_capacity_weighted_weather_aggregates_per_site_rows(self):
        timestamps = [
            pd.Timestamp("2026-07-01 12:00"),
            pd.Timestamp("2026-07-01 12:00"),
        ]
        weather = pd.DataFrame(
            {
                "SolarSiteKey": [1, 2],
                "IntervalStartDT": timestamps,
                "GHI_kWh_per_m2": [0.4, 1.0],
                "WeatherGHI_Wm2": [400.0, 1000.0],
                "DirectRadiation_Wm2": [300.0, 900.0],
                "DiffuseRadiation_Wm2": [80.0, 120.0],
                "Temperature_C": [30.0, 36.0],
                "WindSpeed_ms": [2.0, 4.0],
                "CloudCoverPct": [80.0, 20.0],
                "CloudCoverLowPct": [70.0, 10.0],
                "CloudCoverMidPct": [50.0, 5.0],
                "CloudCoverHighPct": [30.0, 1.0],
            }
        )
        sites = pd.DataFrame(
            {
                "SolarSiteKey": [1, 2],
                "SolarCECkW": [100.0, 300.0],
            }
        )

        out = solar_forecaster.aggregate_capacity_weighted_weather(weather, sites)

        self.assertEqual(len(out), 1)
        self.assertAlmostEqual(float(out.loc[0, "GHI_kWh_per_m2"]), 0.85)
        self.assertAlmostEqual(float(out.loc[0, "WeatherGHI_Wm2"]), 850.0)
        self.assertAlmostEqual(float(out.loc[0, "CloudCoverPct"]), 35.0)

    def test_capacity_weighted_weather_aggregates_cluster_rows(self):
        weather = pd.DataFrame(
            {
                "SolarSiteKey": [0, 1],
                "IntervalStartDT": [pd.Timestamp("2026-07-01 12:00")] * 2,
                "GHI_kWh_per_m2": [0.2, 1.0],
                "WeatherGHI_Wm2": [200.0, 1000.0],
                "DirectRadiation_Wm2": [100.0, 900.0],
                "DiffuseRadiation_Wm2": [50.0, 150.0],
                "Temperature_C": [28.0, 34.0],
                "WindSpeed_ms": [1.0, 3.0],
                "CloudCoverPct": [90.0, 10.0],
                "CloudCoverLowPct": [80.0, 5.0],
                "CloudCoverMidPct": [70.0, 5.0],
                "CloudCoverHighPct": [60.0, 5.0],
            }
        )
        sites = pd.DataFrame(
            {
                "SolarSiteKey": [10, 11, 12],
                "SolarCECkW": [100.0, 300.0, 600.0],
                "WeatherCluster": [0, 1, 1],
            }
        )

        out = solar_forecaster.aggregate_capacity_weighted_weather(weather, sites)

        self.assertEqual(len(out), 1)
        self.assertAlmostEqual(float(out.loc[0, "GHI_kWh_per_m2"]), 0.92)
        self.assertAlmostEqual(float(out.loc[0, "WeatherGHI_Wm2"]), 920.0)
        self.assertAlmostEqual(float(out.loc[0, "CloudCoverPct"]), 18.0)

    def test_add_performance_features_builds_clear_sky_metrics(self):
        features = solar_forecaster.add_performance_features(
            pd.DataFrame(
                {
                    "IntervalStartDT": [pd.Timestamp("2026-06-21 12:00")],
                    "GHI_kWh_per_m2": [0.85],
                    "WeatherGHI_Wm2": [850.0],
                    "DirectRadiation_Wm2": [650.0],
                    "DiffuseRadiation_Wm2": [180.0],
                    "Temperature_C": [34.0],
                    "WindSpeed_ms": [2.5],
                    "CloudCoverPct": [10.0],
                    "CloudCoverLowPct": [2.0],
                    "CloudCoverMidPct": [3.0],
                    "CloudCoverHighPct": [4.0],
                }
            ),
            latitude=solar_forecaster.ROSEVILLE_LATITUDE,
            longitude=solar_forecaster.ROSEVILLE_LONGITUDE,
            timezone_name="America/Los_Angeles",
        )
        self.assertIn("ClearSkyGHI_Wm2", features.columns)
        self.assertIn("ClearSkyIndex", features.columns)
        self.assertGreater(float(features.loc[0, "ClearSkyGHI_Wm2"]), 0.0)
        self.assertGreaterEqual(float(features.loc[0, "ClearSkyIndex"]), 0.0)

    def test_interval_and_hourly_outputs_keep_new_weather_columns(self):
        weather = pd.DataFrame(
            {
                "IntervalStartDT": [pd.Timestamp("2026-07-05 12:00")],
                "GHI_kWh_per_m2": [0.8],
                "WeatherGHI_Wm2": [800.0],
                "DirectRadiation_Wm2": [600.0],
                "DiffuseRadiation_Wm2": [150.0],
                "Temperature_C": [36.0],
                "WindSpeed_ms": [3.0],
                "CloudCoverPct": [12.0],
                "CloudCoverLowPct": [5.0],
                "CloudCoverMidPct": [4.0],
                "CloudCoverHighPct": [3.0],
            }
        )
        intrahour_shape = pd.DataFrame(
            {
                "hour": [12, 12, 12, 12],
                "minute": [0, 15, 30, 45],
                "IntraHourCoefficient": [0.25, 0.25, 0.25, 0.25],
            }
        )
        model = solar_forecaster.PerformanceModel(
            estimator=None,
            fallback_ratio=0.75,
            feature_columns=solar_forecaster.PERFORMANCE_FEATURE_COLUMNS,
        )
        interval = solar_forecaster.build_interval_forecast(
            weather,
            intrahour_shape,
            1000.0,
            model,
            solar_forecaster.ROSEVILLE_LATITUDE,
            solar_forecaster.ROSEVILLE_LONGITUDE,
            "America/Los_Angeles",
            0.0,
        )
        hourly = solar_forecaster.resample_interval_forecast_to_hourly(interval, 1000.0)
        for column in [
            "DirectRadiation_Wm2",
            "DiffuseRadiation_Wm2",
            "Temperature_C",
            "WindSpeed_ms",
            "ClearSkyGHI_Wm2",
            "ClearSkyIndex",
        ]:
            self.assertIn(column, interval.columns)
            self.assertIn(column, hourly.columns)

    def test_predict_performance_ratio_respects_model_upper_bound(self):
        class _HighPredictor:
            @staticmethod
            def predict(_):
                return np.array([1.40])

        model = solar_forecaster.PerformanceModel(
            estimator=_HighPredictor(),
            fallback_ratio=0.8,
            feature_columns=solar_forecaster.PERFORMANCE_FEATURE_COLUMNS,
            upper_bound=1.1,
        )
        row = pd.DataFrame(
            {
                "GHI_kWh_per_m2": [0.9],
                "WeatherGHI_Wm2": [900.0],
                "DirectRadiation_Wm2": [700.0],
                "DiffuseRadiation_Wm2": [140.0],
                "ClearSkyGHI_Wm2": [950.0],
                "ClearSkyIndex": [0.95],
                "Temperature_C": [35.0],
                "WindSpeed_ms": [2.0],
                "CloudCoverPct": [5.0],
                "CloudCoverLowPct": [1.0],
                "CloudCoverMidPct": [2.0],
                "CloudCoverHighPct": [2.0],
                "SolarElevationDeg": [60.0],
                "HourSin": [0.0],
                "HourCos": [1.0],
                "DayOfYearSin": [0.0],
                "DayOfYearCos": [1.0],
            }
        )
        ratio = solar_forecaster.predict_performance_ratio(model, row)
        self.assertAlmostEqual(float(ratio.iloc[0]), 1.1, places=6)

    def test_peak_intrahour_shape_boost_reweights_peak_minutes(self):
        weather = pd.DataFrame(
            {
                "IntervalStartDT": [pd.Timestamp("2026-07-05 12:00")],
                "GHI_kWh_per_m2": [1.2],
                "WeatherGHI_Wm2": [1200.0],
                "DirectRadiation_Wm2": [900.0],
                "DiffuseRadiation_Wm2": [180.0],
                "Temperature_C": [37.0],
                "WindSpeed_ms": [2.0],
                "CloudCoverPct": [0.0],
                "CloudCoverLowPct": [0.0],
                "CloudCoverMidPct": [0.0],
                "CloudCoverHighPct": [0.0],
            }
        )
        intrahour_shape = pd.DataFrame(
            {
                "hour": [12, 12, 12, 12],
                "minute": [0, 15, 30, 45],
                "IntraHourCoefficient": [0.25, 0.25, 0.25, 0.25],
            }
        )
        model = solar_forecaster.PerformanceModel(
            estimator=None,
            fallback_ratio=1.0,
            feature_columns=solar_forecaster.PERFORMANCE_FEATURE_COLUMNS,
            upper_bound=1.1,
        )
        interval = solar_forecaster.build_interval_forecast(
            weather,
            intrahour_shape,
            1000.0,
            model,
            solar_forecaster.ROSEVILLE_LATITUDE,
            solar_forecaster.ROSEVILLE_LONGITUDE,
            "America/Los_Angeles",
            0.0,
            peak_hourly_kwh_quantile=0.5,
        )
        by_minute = interval.set_index(interval["IntervalStartDT"].dt.minute)[
            "Forecast_kWh"
        ]
        self.assertGreater(float(by_minute.loc[45]), float(by_minute.loc[0]))


class SolarForecasterBacktestDiagnosticTests(unittest.TestCase):
    @staticmethod
    def _sample_backtest():
        timestamps = pd.to_datetime(
            [
                "2026-04-01 10:00",
                "2026-04-01 12:00",
                "2026-04-01 14:00",
                "2026-04-01 22:00",
            ]
        )
        actual_mw = np.array([10.0, 20.0, 5.0, 0.0])
        forecast_mw = np.array([8.0, 24.0, 2.0, 0.0])
        base_mw = np.array([7.0, 22.0, 3.0, 0.0])
        out = pd.DataFrame(
            {
                "IntervalStartDT": timestamps,
                "HE": [11, 13, 15, 23],
                "Actual_MW": actual_mw,
                "Forecast_MW": forecast_mw,
                "BaseForecast_MW": base_mw,
                "Actual_kWh": actual_mw * 1000.0,
                "Forecast_kWh": forecast_mw * 1000.0,
                "BaseForecast_kWh": base_mw * 1000.0,
                "WeatherGHI_Wm2": [500.0, 800.0, 300.0, 0.0],
                "GHI_kWh_per_m2": [0.5, 0.8, 0.3, 0.0],
                "ClearSkyGHI_Wm2": [700.0, 900.0, 500.0, 0.0],
                "ClearSkyIndex": [0.71, 0.89, 0.60, 0.0],
                "CloudCoverPct": [25.0, 10.0, 90.0, 100.0],
                "CloudCoverLowPct": [10.0, 5.0, 80.0, 100.0],
                "CloudCoverMidPct": [20.0, 5.0, 10.0, 50.0],
                "CloudCoverHighPct": [5.0, 0.0, 20.0, 100.0],
                "PerformanceRatio": [0.5, 0.6, 0.4, 0.0],
                "ResidualCalibrationFactor": [1.0, 1.0, 1.0, 1.0],
                "SeasonalCalibrationFactor": [1.0, 1.0, 1.0, 1.0],
                "TotalCalibrationFactor": [1.0, 1.0, 1.0, 1.0],
                "ForecastSource": ["test"] * 4,
            }
        )
        out["Error_MW"] = out["Forecast_MW"] - out["Actual_MW"]
        out["AbsError_MW"] = out["Error_MW"].abs()
        return out

    def test_solar_backtest_diagnostics_include_operational_slices(self):
        diagnostics = solar_forecaster.calculate_solar_backtest_diagnostic_metrics(
            self._sample_backtest(),
            daylight_threshold_mw=0.1,
        )

        slices = set(diagnostics["Slice"])
        self.assertIn("Overall", slices)
        self.assertIn("DaylightActual", slices)
        self.assertIn("PeakSolarHours11to15", slices)
        self.assertIn("Hour:12", slices)
        self.assertIn("CloudCover:80-100", slices)

        overall = diagnostics.loc[diagnostics["Slice"] == "Overall"].iloc[0]
        self.assertAlmostEqual(float(overall["Actual_MWh"]), 35.0)
        self.assertAlmostEqual(
            float(overall["WMAPE_PCT"]), (2.0 + 4.0 + 3.0) / 35.0 * 100.0
        )

    def test_solar_backtest_top_errors_ranks_under_and_over_forecasts(self):
        top_errors = solar_forecaster.build_solar_backtest_top_errors(
            self._sample_backtest(),
            top_n=1,
            daylight_threshold_mw=0.1,
        )

        self.assertEqual(
            set(top_errors["ErrorType"]), {"Underforecast", "Overforecast"}
        )
        under = top_errors.loc[top_errors["ErrorType"] == "Underforecast"].iloc[0]
        over = top_errors.loc[top_errors["ErrorType"] == "Overforecast"].iloc[0]
        self.assertAlmostEqual(float(under["Underforecast_MW"]), 3.0)
        self.assertAlmostEqual(float(over["Overforecast_MW"]), 4.0)

    def test_ami_suppressed_actuals_are_excluded_from_headline_scorecards(self):
        timestamps = pd.to_datetime(
            [
                "2026-04-29 12:00",
                "2026-04-29 13:00",
                "2026-04-30 12:00",
            ]
        )
        backtest = pd.DataFrame(
            {
                "IntervalStartDT": timestamps,
                "HE": [13, 14, 13],
                "Actual_kWh": [100.0, 200.0, 8000.0],
                "Forecast_kWh": [20000.0, 21000.0, 10000.0],
                "BaseForecast_kWh": [19000.0, 20000.0, 9000.0],
                "GHI_kWh_per_m2": [0.85, 0.90, 0.80],
                "ClearSkyIndex": [0.95, 0.98, 0.88],
                "CloudCoverPct": [0.0, 0.0, 10.0],
            }
        )
        backtest["Actual_MW"] = backtest["Actual_kWh"] / 1000.0
        backtest["Forecast_MW"] = backtest["Forecast_kWh"] / 1000.0
        backtest["BaseForecast_MW"] = backtest["BaseForecast_kWh"] / 1000.0
        backtest["Error_MW"] = backtest["Forecast_MW"] - backtest["Actual_MW"]
        backtest["AbsError_MW"] = backtest["Error_MW"].abs()
        backtest["BaseError_MW"] = backtest["BaseForecast_MW"] - backtest["Actual_MW"]
        backtest["BaseAbsError_MW"] = backtest["BaseError_MW"].abs()
        backtest["Error_kWh"] = backtest["Forecast_kWh"] - backtest["Actual_kWh"]
        backtest["AbsError_kWh"] = backtest["Error_kWh"].abs()
        backtest["BaseError_kWh"] = (
            backtest["BaseForecast_kWh"] - backtest["Actual_kWh"]
        )
        backtest["BaseAbsError_kWh"] = backtest["BaseError_kWh"].abs()
        backtest["ActualQualityExpected_kWh"] = backtest[
            ["Forecast_kWh", "BaseForecast_kWh"]
        ].max(axis=1)

        flagged = solar_forecaster.add_solar_actual_quality_flags(
            backtest,
            actual_kwh_col="Actual_kWh",
            expected_kwh_col="ActualQualityExpected_kWh",
            actual_to_expected_ratio_threshold=0.15,
        )

        self.assertEqual(int(flagged["SolarBacktestExcluded"].sum()), 2)
        self.assertEqual(
            flagged.loc[0, "ActualQualityFlag"],
            solar_forecaster.ACTUAL_QUALITY_AMI_SUPPRESSED,
        )
        self.assertEqual(
            flagged.loc[2, "ActualQualityFlag"], solar_forecaster.ACTUAL_QUALITY_OK
        )

        summary = solar_forecaster.calculate_backtest_summary(flagged).iloc[0]
        self.assertEqual(int(summary["RawIntervals"]), 3)
        self.assertEqual(int(summary["ExcludedIntervals"]), 2)
        self.assertEqual(int(summary["Intervals"]), 1)
        self.assertAlmostEqual(float(summary["Actual_MWh"]), 8.0)
        self.assertAlmostEqual(float(summary["RawActual_MWh"]), 8.3)

        diagnostics = solar_forecaster.calculate_solar_backtest_diagnostic_metrics(
            flagged
        )
        self.assertIn("RawOverall", set(diagnostics["Slice"]))
        self.assertIn("ActualQualityExcluded", set(diagnostics["Slice"]))
        overall = diagnostics.loc[diagnostics["Slice"] == "Overall"].iloc[0]
        self.assertEqual(int(overall["N"]), 1)

        top_errors = solar_forecaster.build_solar_backtest_top_errors(
            flagged,
            top_n=1,
            daylight_threshold_mw=0.1,
        )
        over = top_errors.loc[top_errors["ErrorType"] == "Overforecast"].iloc[0]
        self.assertTrue(bool(over["SolarBacktestExcluded"]))

    def test_solar_temporal_holdout_returns_status_when_disabled(self):
        scorecard, hourly = solar_forecaster.build_solar_temporal_holdout_backtest(
            rec_interval_df=pd.DataFrame(),
            weather_df=pd.DataFrame(),
            capacity_kw=1000.0,
            fallback_ratio=0.75,
            latitude=solar_forecaster.ROSEVILLE_LATITUDE,
            longitude=solar_forecaster.ROSEVILLE_LONGITUDE,
            timezone_name="America/Los_Angeles",
            min_solar_elevation=0.0,
            daily_active_capacity=None,
            max_performance_ratio=1.1,
            use_performance_model_energy_weighting=True,
            intrahour_shape_method="median",
            shape_quantile=0.75,
            peak_hourly_kwh_quantile=0.90,
            holdout_days=0,
            residual_calibration_enabled=True,
            residual_lower_bound=0.25,
            residual_upper_bound=1.75,
            residual_min_forecast_kwh=25.0,
            residual_min_training_rows=96,
            residual_energy_weighting=True,
            seasonal_calibration_enabled=True,
            seasonal_prior_mwh=500.0,
            seasonal_lower_bound=0.85,
            seasonal_upper_bound=1.15,
        )

        self.assertEqual(scorecard.loc[0, "Status"], "skipped")
        self.assertTrue(hourly.empty)


class SolarForecasterPerformanceModelTests(unittest.TestCase):
    class _FakeGradientBoostingRegressor:
        last_sample_weight = "not-called"
        last_fit_rows = None

        def __init__(self, *_args, **_kwargs):
            pass

        def fit(self, X, _y, sample_weight=None):
            type(self).last_sample_weight = sample_weight
            type(self).last_fit_rows = len(X)
            self._rows = len(X)
            return self

        def predict(self, X):
            return np.full(len(X), 0.50)

    @staticmethod
    def _performance_training_frames():
        timestamps = pd.date_range("2026-06-01 10:00", periods=4, freq="h")
        rec = pd.DataFrame(
            {
                "IntervalStartDT": timestamps,
                "Export_kWh": [100.0, 400.0, 800.0, 300.0],
            }
        )
        weather = pd.DataFrame(
            {
                "IntervalStartDT": timestamps,
                "GHI_kWh_per_m2": [0.4, 0.8, 1.0, 0.5],
                "WeatherGHI_Wm2": [400.0, 800.0, 1000.0, 500.0],
                "DirectRadiation_Wm2": [300.0, 650.0, 800.0, 350.0],
                "DiffuseRadiation_Wm2": [80.0, 120.0, 150.0, 90.0],
                "Temperature_C": [28.0, 32.0, 35.0, 30.0],
                "WindSpeed_ms": [2.0, 2.5, 3.0, 2.2],
                "CloudCoverPct": [10.0, 15.0, 5.0, 20.0],
                "CloudCoverLowPct": [2.0, 3.0, 1.0, 4.0],
                "CloudCoverMidPct": [3.0, 4.0, 2.0, 5.0],
                "CloudCoverHighPct": [5.0, 8.0, 2.0, 10.0],
            }
        )
        return rec, weather

    def test_performance_model_energy_weighting_passes_sample_weight(self):
        rec, weather = self._performance_training_frames()
        fake = self._FakeGradientBoostingRegressor
        fake.last_sample_weight = "not-called"

        with patch.object(solar_forecaster, "GradientBoostingRegressor", fake):
            solar_forecaster.train_performance_model(
                rec_intervals=rec,
                weather_df=weather,
                capacity_kw=1000.0,
                fallback_ratio=0.75,
                latitude=solar_forecaster.ROSEVILLE_LATITUDE,
                longitude=solar_forecaster.ROSEVILLE_LONGITUDE,
                timezone_name="America/Los_Angeles",
                min_training_available_kwh=1.0,
                min_training_rows=2,
                use_energy_weighting=True,
            )

        self.assertIsNotNone(fake.last_sample_weight)
        self.assertEqual(len(fake.last_sample_weight), 4)
        self.assertAlmostEqual(float(fake.last_sample_weight.iloc[2]), 800.0)

    def test_performance_model_energy_weighting_can_be_disabled(self):
        rec, weather = self._performance_training_frames()
        fake = self._FakeGradientBoostingRegressor
        fake.last_sample_weight = "not-called"

        with patch.object(solar_forecaster, "GradientBoostingRegressor", fake):
            solar_forecaster.train_performance_model(
                rec_intervals=rec,
                weather_df=weather,
                capacity_kw=1000.0,
                fallback_ratio=0.75,
                latitude=solar_forecaster.ROSEVILLE_LATITUDE,
                longitude=solar_forecaster.ROSEVILLE_LONGITUDE,
                timezone_name="America/Los_Angeles",
                min_training_available_kwh=1.0,
                min_training_rows=2,
                use_energy_weighting=False,
            )

        self.assertIsNone(fake.last_sample_weight)

    def test_performance_model_excludes_ami_suppressed_training_rows(self):
        timestamps = pd.to_datetime(
            [
                "2026-06-01 11:00",
                "2026-06-01 12:00",
                "2026-06-02 11:00",
                "2026-06-02 12:00",
                "2026-06-02 13:00",
            ]
        )
        rec = pd.DataFrame(
            {
                "IntervalStartDT": timestamps,
                "Export_kWh": [20.0, 30.0, 3000.0, 3600.0, 3200.0],
            }
        )
        weather = pd.DataFrame(
            {
                "IntervalStartDT": timestamps,
                "GHI_kWh_per_m2": [0.75, 0.85, 0.75, 0.85, 0.80],
                "WeatherGHI_Wm2": [750.0, 850.0, 750.0, 850.0, 800.0],
                "DirectRadiation_Wm2": [650.0, 750.0, 650.0, 750.0, 700.0],
                "DiffuseRadiation_Wm2": [100.0, 100.0, 100.0, 100.0, 100.0],
                "Temperature_C": [30.0, 32.0, 30.0, 32.0, 31.0],
                "WindSpeed_ms": [2.0, 2.0, 2.0, 2.0, 2.0],
                "CloudCoverPct": [0.0, 0.0, 5.0, 5.0, 5.0],
                "CloudCoverLowPct": [0.0, 0.0, 2.0, 2.0, 2.0],
                "CloudCoverMidPct": [0.0, 0.0, 2.0, 2.0, 2.0],
                "CloudCoverHighPct": [0.0, 0.0, 1.0, 1.0, 1.0],
            }
        )
        fake = self._FakeGradientBoostingRegressor
        fake.last_sample_weight = "not-called"
        fake.last_fit_rows = None

        with patch.object(solar_forecaster, "GradientBoostingRegressor", fake):
            solar_forecaster.train_performance_model(
                rec_intervals=rec,
                weather_df=weather,
                capacity_kw=10000.0,
                fallback_ratio=0.75,
                latitude=solar_forecaster.ROSEVILLE_LATITUDE,
                longitude=solar_forecaster.ROSEVILLE_LONGITUDE,
                timezone_name="America/Los_Angeles",
                min_training_available_kwh=1000.0,
                min_training_rows=2,
                use_energy_weighting=True,
            )

        self.assertEqual(fake.last_fit_rows, 3)
        self.assertEqual(len(fake.last_sample_weight), 3)


if __name__ == "__main__":
    unittest.main()
