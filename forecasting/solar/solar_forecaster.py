#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Solar Generation Forecaster
===========================

Purpose
-------
This script creates a system-wide solar generation forecast by combining
site-specific capacity data with REC/NET export actuals, hourly irradiance
weather data, and a normalized intra-hour production shape derived from
historical meter reads.

What this script does
---------------------
1.  Connects to the destination Forecast DB.
2.  Loads forecast-eligible solar site data (capacity, location) from the
    `Forecasting.ForecastSolarSite` table.
3.  Groups sites into geographic clusters to reduce weather API calls.
4.  Loads REC channel export and negative NET interval export data from parquet
    files for forecast-eligible solar sites.
5.  Trains a linear regression model on historical weather and solar generation
    to predict a dynamic performance ratio.
6.  Calculates a historical average 15-minute solar generation shape.
7.  Fetches hourly GHI and cloud-cover weather data from the Open-Meteo API for
    both historical and forecast periods.
8.  Builds a continuous hourly forecast (hindcast and forecast) using the
    trained model.
9.  Splits the hourly forecast into 15-minute intervals using the intra-hour
    historical shape.
10. Optionally corrects remaining same-day intervals using completed actual
    export intervals already observed today.
11. Saves the final forecast, actuals, and production shapes to CSV files.

Requirements
------------
pip install pyodbc pandas requests scikit-learn SQLAlchemy pyarrow
"""

from __future__ import annotations

import argparse
import hashlib
import json
import logging
import math
import os
import sys
import time
from dataclasses import dataclass, field
from datetime import date, datetime, timedelta
from pathlib import Path
from typing import Optional
from zoneinfo import ZoneInfo

import numpy as np
import pandas as pd
import requests
import urllib3
from sklearn.cluster import KMeans
from sklearn.ensemble import GradientBoostingRegressor
from sqlalchemy.engine import Engine

try:
    from .db_utils import connect, read_sql
except ImportError:
    from db_utils import connect, read_sql

# Suppress only the single InsecureRequestWarning from urllib3 needed for this script
urllib3.disable_warnings(urllib3.exceptions.InsecureRequestWarning)


# =============================================================================
# Logging
# =============================================================================

LOG_FORMAT = "%(asctime)s | %(levelname)-8s | %(message)s"
INTERVAL_HOURS = 0.25
DEFAULT_PERFORMANCE_RATIO = 0.75
DEFAULT_PARQUET_ROOT = Path(
    os.environ.get("FORECAST_SOLAR_PARQUET_ROOT")
    or os.environ.get("FORECAST_DATA_ROOT")
    or r"C:\PY_LRS"
)


def _default_solar_weather_cache_dir() -> Path:
    explicit = os.environ.get("FORECAST_SOLAR_WEATHER_CACHE_DIR")
    if explicit:
        return Path(explicit)
    base_cache = os.environ.get("FORECAST_WEATHER_CACHE_DIR")
    if base_cache:
        return Path(base_cache) / "solar_weather"
    return Path("weather_cache") / "solar_weather"


DEFAULT_SOLAR_WEATHER_CACHE_DIR = _default_solar_weather_cache_dir()
DEFAULT_FORECAST_WEATHER_CACHE_MAX_AGE_HOURS = 6.0
DEFAULT_HOURLY_WEATHER_FETCH_CHUNK_DAYS = 16
INDEX_CACHE_ROOT = Path("_shape_analysis_cache") / "spid_file_index"
CATALOG_COLUMNS = [
    "FileID",
    "filepath",
    "filename",
    "relative_path",
    "folder",
    "segment",
    "channel",
    "rate_group",
    "nem_status",
    "month",
    "size_mb",
    "size_bytes",
    "modified_time",
    "modified_time_ns",
    "columns_json",
    "schema_error",
]
ROSEVILLE_LATITUDE = 38.7522
ROSEVILLE_LONGITUDE = -121.2880
CUSTOMER_SEGMENT_TOTAL = "TOTAL"
CUSTOMER_SEGMENT_NEM = "NEM"
CUSTOMER_SEGMENT_SOLAR_20 = "SOLAR_2_0"
CUSTOMER_SEGMENT_OTHER = "OTHER"
CUSTOMER_SEGMENT_LABELS = {
    CUSTOMER_SEGMENT_TOTAL: "Total",
    CUSTOMER_SEGMENT_NEM: "NEM",
    CUSTOMER_SEGMENT_SOLAR_20: "Solar 2.0",
    CUSTOMER_SEGMENT_OTHER: "Other",
}
DEFAULT_DAILY_SHAPE_METHOD = "upper-quantile"
DEFAULT_INTRAHOUR_SHAPE_METHOD = "upper-quantile"
DEFAULT_SHAPE_QUANTILE = 0.75
DEFAULT_PERFORMANCE_RATIO_UPPER_BOUND = 1.10
DEFAULT_MAX_PERFORMANCE_RATIO = DEFAULT_PERFORMANCE_RATIO_UPPER_BOUND
DEFAULT_PEAK_HOURLY_KWH_QUANTILE = 0.90
DEFAULT_SOLAR_BACKTEST_DAYLIGHT_THRESHOLD_MW = 0.10
DEFAULT_SOLAR_BACKTEST_TOP_ERROR_COUNT = 100
DEFAULT_SOLAR_BACKTEST_HOLDOUT_DAYS = 30
DEFAULT_ACTUAL_QUALITY_MIN_EXPECTED_KWH = 2500.0
DEFAULT_ACTUAL_QUALITY_MIN_GHI_KWH_M2 = 0.35
DEFAULT_ACTUAL_QUALITY_MIN_CLEAR_SKY_INDEX = 0.55
DEFAULT_ACTUAL_QUALITY_MIN_BAD_HOURS_PER_DAY = 2
DEFAULT_ACTUAL_QUALITY_FORECAST_RATIO_THRESHOLD = 0.15
DEFAULT_ACTUAL_QUALITY_AVAILABLE_RATIO_THRESHOLD = 0.08
DEFAULT_ACTUAL_QUALITY_READ_COVERAGE_RATIO_THRESHOLD = 0.25
ACTUAL_QUALITY_OK = "OK"
ACTUAL_QUALITY_AMI_SUPPRESSED = "AMI_SUPPRESSED_ACTUAL"
DEFAULT_TEMPERATURE_C = 25.0
DEFAULT_ARRAY_TILT_DEGREES = 25.0
DEFAULT_ARRAY_AZIMUTH_DEGREES = 180.0
GROUND_ALBEDO = 0.20
PV_TEMPERATURE_COEFFICIENT_PER_C = -0.004
PV_NOCT_C = 45.0
PV_WIND_COOLING_C_PER_MPS = 0.35
NET_METER_TYPES = {"AMI_NET", "AMI_NET_D"}
SOLAR_20_METER_PATTERNS = ("SLR2", "TSL2")
HOURLY_WEATHER_VARIABLES = [
    "shortwave_radiation",
    "direct_radiation",
    "direct_normal_irradiance",
    "diffuse_radiation",
    "global_tilted_irradiance",
    "temperature_2m",
    "relative_humidity_2m",
    "dew_point_2m",
    "apparent_temperature",
    "surface_pressure",
    "wind_speed_10m",
    "wind_direction_10m",
    "wind_gusts_10m",
    "vapour_pressure_deficit",
    "precipitation",
    "weather_code",
    "sunshine_duration",
    "is_day",
    "cloud_cover",
    "cloud_cover_low",
    "cloud_cover_mid",
    "cloud_cover_high",
]
CLOUD_COVER_OUTPUT_COLUMNS = [
    "CloudCoverPct",
    "CloudCoverLowPct",
    "CloudCoverMidPct",
    "CloudCoverHighPct",
]
ADVANCED_WEATHER_OUTPUT_COLUMNS = [
    "DirectRadiation_kWh_per_m2",
    "DirectRadiation_Wm2",
    "DirectNormalIrradiance_Wm2",
    "DiffuseRadiation_kWh_per_m2",
    "DiffuseRadiation_Wm2",
    "GlobalTiltedIrradiance_kWh_per_m2",
    "GlobalTiltedIrradiance_Wm2",
    "Temperature2m_C",
    "RelativeHumidity2mPct",
    "DewPoint2m_C",
    "ApparentTemperature_C",
    "SurfacePressure_hPa",
    "WindSpeed10m_mps",
    "WindDirection10mDeg",
    "WindGusts10m_mps",
    "VapourPressureDeficit_kPa",
    "Precipitation_mm",
    "WeatherCode",
    "SunshineDurationSec",
    "IsDay",
    "Temperature_C",
    "WindSpeed_ms",
]
WEATHER_OUTPUT_COLUMNS = [
    "GHI_kWh_per_m2",
    "WeatherGHI_Wm2",
    *ADVANCED_WEATHER_OUTPUT_COLUMNS,
    *CLOUD_COVER_OUTPUT_COLUMNS,
]
PHYSICS_FEATURE_COLUMNS = [
    "ClearSkyGHI_Wm2",
    "ClearSkyIndex",
    "SolarAzimuthDeg",
    "IncidenceAngleCos",
    "PlaneOfArrayIrradiance_Wm2",
    "PlaneOfArray_kWh_per_m2",
    "DirectPOA_Wm2",
    "DiffusePOA_Wm2",
    "CellTemperature_C",
    "TemperatureDerate",
    "PVWatts_kWh_per_kW",
]
PERFORMANCE_FEATURE_COLUMNS = [
    "GHI_kWh_per_m2",
    "WeatherGHI_Wm2",
    "DirectRadiation_kWh_per_m2",
    "DirectRadiation_Wm2",
    "DirectNormalIrradiance_Wm2",
    "DiffuseRadiation_kWh_per_m2",
    "DiffuseRadiation_Wm2",
    "GlobalTiltedIrradiance_kWh_per_m2",
    "GlobalTiltedIrradiance_Wm2",
    "Temperature2m_C",
    "RelativeHumidity2mPct",
    "DewPoint2m_C",
    "ApparentTemperature_C",
    "SurfacePressure_hPa",
    "WindSpeed10m_mps",
    "WindDirection10mDeg",
    "WindGusts10m_mps",
    "VapourPressureDeficit_kPa",
    "Precipitation_mm",
    "WeatherCode",
    "SunshineDurationSec",
    "IsDay",
    "Temperature_C",
    "WindSpeed_ms",
    *CLOUD_COVER_OUTPUT_COLUMNS,
    *PHYSICS_FEATURE_COLUMNS,
    "SolarElevationDeg",
    "HourSin",
    "HourCos",
    "DayOfYearSin",
    "DayOfYearCos",
]
CALIBRATION_FEATURE_COLUMNS = PERFORMANCE_FEATURE_COLUMNS + [
    "Forecast_kWh",
    "Forecast_kW",
    "CapacityFactor",
    "PerformanceRatio",
]
FORECAST_DIAGNOSTIC_COLUMNS = [
    "SolarElevationDeg",
    *WEATHER_OUTPUT_COLUMNS,
    *PHYSICS_FEATURE_COLUMNS,
]
DEFAULT_WEATHER_CLUSTERS = 10
DEFAULT_WEATHER_LOCATIONS_PER_REQUEST = 50
DEFAULT_MAX_WEATHER_API_CALLS = 1
DEFAULT_RESIDUAL_CALIBRATION_ENERGY_WEIGHT_POWER = 1.35
DEFAULT_CALIBRATION_DAYLIGHT_ROW_COVERAGE_MIN = 0.95
DEFAULT_CALIBRATION_DAY_ACTUAL_FORECAST_RATIO_MIN = 0.20
DEFAULT_CALIBRATION_DAY_FORECAST_MWH_MIN = 20.0
DEFAULT_REGIME_CALIBRATION_MIN_ROWS = 24
DEFAULT_REGIME_CALIBRATION_MIN_FORECAST_MWH = 50.0
DEFAULT_REGIME_CALIBRATION_PRIOR_MWH = 250.0
DEFAULT_REGIME_CALIBRATION_LOWER_BOUND = 0.85
DEFAULT_REGIME_CALIBRATION_UPPER_BOUND = 1.15
DEFAULT_PEAK_CALIBRATION_QUANTILE = 0.98
DEFAULT_PEAK_CALIBRATION_MIN_ROWS = 24
DEFAULT_PEAK_CALIBRATION_MIN_FORECAST_MWH = 50.0
DEFAULT_PEAK_CALIBRATION_LOWER_BOUND = 0.90
DEFAULT_PEAK_CALIBRATION_TARGET_CAPTURE = 1.00
REGIME_HOLDOUT_WMAPE_TOLERANCE = 0.0005
REGIME_HOLDOUT_MIN_ROWS = 12
REGIME_HOLDOUT_MIN_FORECAST_MWH = 10.0
PEAK_HOLDOUT_WMAPE_TOLERANCE = 0.0015
PEAK_CAPTURE_TOLERANCE = 0.001
REGIME_CLEAR_SKY_BINS = [-0.01, 0.20, 0.50, 0.80, 1.05, 1.60]
REGIME_CLEAR_SKY_LABELS = [
    "csi_000_020",
    "csi_020_050",
    "csi_050_080",
    "csi_080_105",
    "csi_105_160",
]
REGIME_CLOUD_BINS = [-1.0, 30.0, 70.0, 100.0]
REGIME_CLOUD_LABELS = ["cloud_000_030", "cloud_030_070", "cloud_070_100"]
REGIME_ELEVATION_BINS = [-90.0, 15.0, 35.0, 90.0]
REGIME_ELEVATION_LABELS = ["elev_low", "elev_mid", "elev_high"]
OPEN_METEO_TRANSIENT_STATUS_CODES = {429, 500, 502, 503, 504}
OPEN_METEO_MAX_RETRIES = 4
OPEN_METEO_RETRY_BASE_SECONDS = 8.0
OPEN_METEO_RETRY_MAX_SECONDS = 120.0
OPEN_METEO_TIMEOUT_SECONDS = 60
OPEN_METEO_MAX_ATTEMPTS = OPEN_METEO_MAX_RETRIES
OPEN_METEO_RETRY_BACKOFF_SECONDS = 3.0


@dataclass
class PerformanceModel:
    estimator: Optional[GradientBoostingRegressor]
    fallback_ratio: float
    feature_columns: list[str]
    upper_bound: float = DEFAULT_PERFORMANCE_RATIO_UPPER_BOUND

    @property
    def ratio_upper_bound(self) -> float:
        return self.upper_bound


@dataclass
class ResidualCalibrationModel:
    estimator: Optional[GradientBoostingRegressor]
    fallback_factor: float
    feature_columns: list[str]
    lower_bound: float
    upper_bound: float
    seasonal_factors: dict[int, float] = field(default_factory=dict)
    seasonal_default_factor: float = 1.0
    regime_factors: dict[str, float] = field(default_factory=dict)
    regime_default_factor: float = 1.0
    peak_calibration_factor: float = 1.0
    peak_calibration_threshold_cf: Optional[float] = None


def configure_logging(verbose: bool = False) -> None:
    logging.basicConfig(
        level=logging.DEBUG if verbose else logging.INFO,
        format=LOG_FORMAT,
        datefmt="%Y-%m-%d %H:%M:%S",
        force=True,
    )


def add_hour_ending_column(
    df: pd.DataFrame,
    timestamp_col: str = "IntervalStartDT",
    column_name: str = "HE",
) -> pd.DataFrame:
    """
    Add hour-ending number from the interval start timestamp.

    HE1 covers intervals starting at 00:00, HE2 covers intervals starting at 01:00,
    and HE24 covers intervals starting at 23:00.
    """
    out = df.copy()
    out[column_name] = pd.to_datetime(out[timestamp_col]).dt.hour + 1
    return out


def current_local_timestamp(timezone_name: str) -> pd.Timestamp:
    """
    Return the current local wall-clock timestamp without timezone info.
    """
    return pd.Timestamp(datetime.now(ZoneInfo(timezone_name))).tz_localize(None)


def _open_meteo_get_json(
    url: str,
    params: dict,
    *,
    source_name: str,
    start_date: date,
    end_date: date,
) -> dict | list:
    """
    Fetch Open-Meteo JSON with bounded retries for transient TLS/network resets.
    """
    last_error: Exception | None = None
    for attempt in range(1, OPEN_METEO_MAX_ATTEMPTS + 1):
        try:
            response = requests.get(
                url,
                params=params,
                verify=False,
                timeout=OPEN_METEO_TIMEOUT_SECONDS,
            )
            response.raise_for_status()
            return response.json()
        except requests.HTTPError as exc:
            last_error = exc
            status_code = exc.response.status_code if exc.response is not None else None
            if (
                status_code is not None
                and 400 <= status_code < 500
                and status_code != 429
            ):
                raise
        except (requests.RequestException, ValueError) as exc:
            last_error = exc

        if attempt >= OPEN_METEO_MAX_ATTEMPTS:
            break
        sleep_seconds = OPEN_METEO_RETRY_BACKOFF_SECONDS * attempt
        logging.warning(
            "Open-Meteo %s weather request failed for %s to %s on attempt %s/%s: %s. Retrying in %.1f seconds.",
            source_name,
            start_date,
            end_date,
            attempt,
            OPEN_METEO_MAX_ATTEMPTS,
            last_error,
            sleep_seconds,
        )
        time.sleep(sleep_seconds)

    raise RuntimeError(
        f"Open-Meteo {source_name} weather request failed for {start_date} to {end_date} "
        f"after {OPEN_METEO_MAX_ATTEMPTS} attempts."
    ) from last_error


def _safe_cache_token(value: object) -> str:
    token = "".join(ch if ch.isalnum() else "_" for ch in str(value))
    return "_".join(part for part in token.split("_") if part)


def _solar_weather_cache_root(cache_dir: str | Path | None) -> Path:
    path = Path(cache_dir) if cache_dir else DEFAULT_SOLAR_WEATHER_CACHE_DIR
    if not path.is_absolute():
        path = Path.cwd() / path
    return path


def _solar_weather_site_signature(sites: pd.DataFrame) -> str:
    if sites is None or sites.empty:
        return "no_sites"
    cols = [
        col for col in ["SolarSiteKey", "Latitude", "Longitude"] if col in sites.columns
    ]
    if not cols:
        return "no_site_columns"
    work = sites[cols].copy()
    sort_cols = [
        col for col in ["SolarSiteKey", "Latitude", "Longitude"] if col in work.columns
    ]
    work.sort_values(sort_cols, inplace=True, ignore_index=True)
    payload = work.to_csv(index=False, float_format="%.6f")
    return hashlib.sha1(payload.encode("utf-8")).hexdigest()[:12]


def _weather_variables_signature(variables: list[str]) -> str:
    payload = ",".join(str(variable) for variable in variables)
    return hashlib.sha1(payload.encode("utf-8")).hexdigest()[:8]


def _solar_weather_cache_path(
    *,
    cache_dir: str | Path | None,
    kind: str,
    source_name: str,
    start_date: date,
    end_date: date,
    sites: pd.DataFrame,
    timezone_name: str,
    variables: Optional[list[str]] = None,
) -> Path:
    root = _solar_weather_cache_root(cache_dir)
    site_hash = _solar_weather_site_signature(sites)
    stem_parts = [
        "solar",
        kind,
        source_name,
        start_date.isoformat(),
        end_date.isoformat(),
        _safe_cache_token(timezone_name),
        site_hash,
    ]
    if variables:
        stem_parts.append(_weather_variables_signature(variables))
    return root / f"{'_'.join(stem_parts)}.csv"


def _read_solar_weather_cache(
    path: Path,
    *,
    start_date: date,
    end_date: date,
    timestamp_col: str,
) -> pd.DataFrame:
    if not path.exists():
        return pd.DataFrame()
    try:
        out = pd.read_csv(path)
    except Exception as exc:
        logging.warning("Ignoring unreadable solar weather cache %s: %s", path, exc)
        return pd.DataFrame()
    if timestamp_col not in out.columns:
        return pd.DataFrame()

    out = out.copy()
    if timestamp_col == "date":
        out[timestamp_col] = pd.to_datetime(out[timestamp_col], errors="coerce").dt.date
        in_range = (
            out[timestamp_col].notna()
            & out[timestamp_col].ge(start_date)
            & out[timestamp_col].le(end_date)
        )
    else:
        out[timestamp_col] = pd.to_datetime(out[timestamp_col], errors="coerce")
        in_range = (
            out[timestamp_col].notna()
            & out[timestamp_col].dt.date.ge(start_date)
            & out[timestamp_col].dt.date.le(end_date)
        )
    out = out.loc[in_range].copy()
    if out.empty:
        return pd.DataFrame()
    return out.sort_values(timestamp_col).reset_index(drop=True)


def _write_solar_weather_cache(df: pd.DataFrame, path: Path) -> None:
    if df is None or df.empty:
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(path, index=False)


def _solar_weather_cache_is_fresh(path: Path, max_age_hours: float) -> bool:
    if not path.exists():
        return False
    try:
        age_seconds = time.time() - path.stat().st_mtime
    except OSError:
        return False
    return age_seconds <= max(0.0, float(max_age_hours)) * 3600.0


def _read_overlapping_solar_weather_caches(
    *,
    cache_dir: str | Path | None,
    kind: str,
    source_name: str,
    start_date: date,
    end_date: date,
    sites: pd.DataFrame,
    timezone_name: str,
    variables: Optional[list[str]],
    timestamp_col: str,
    forecast_cache_max_age_hours: float = DEFAULT_FORECAST_WEATHER_CACHE_MAX_AGE_HOURS,
) -> pd.DataFrame:
    root = _solar_weather_cache_root(cache_dir)
    if not root.exists():
        return pd.DataFrame()

    site_hash = _solar_weather_site_signature(sites)
    suffix_parts = [_safe_cache_token(timezone_name), site_hash]
    if variables:
        suffix_parts.append(_weather_variables_signature(variables))
    expected_suffix = "_" + "_".join(suffix_parts)

    frames: list[pd.DataFrame] = []
    for path in root.glob(f"solar_{kind}_{source_name}_*.csv"):
        stem = path.stem
        if not stem.endswith(expected_suffix):
            continue
        parts = stem.split("_")
        if len(parts) < 7:
            continue
        try:
            cache_start = date.fromisoformat(parts[3])
            cache_end = date.fromisoformat(parts[4])
        except ValueError:
            continue
        if cache_end < start_date or cache_start > end_date:
            continue
        if source_name == "forecast" and not _solar_weather_cache_is_fresh(
            path, forecast_cache_max_age_hours
        ):
            continue

        overlap_start = max(start_date, cache_start)
        overlap_end = min(end_date, cache_end)
        cached = _read_solar_weather_cache(
            path,
            start_date=overlap_start,
            end_date=overlap_end,
            timestamp_col=timestamp_col,
        )
        if not cached.empty:
            frames.append(cached)

    if not frames:
        return pd.DataFrame()
    return pd.concat(frames, ignore_index=True)


def _archive_solar_forecast_weather(
    df: pd.DataFrame, cache_dir: str | Path | None, source_path: Path
) -> None:
    if df is None or df.empty:
        return
    archive_dir = _solar_weather_cache_root(cache_dir) / "forecast_weather_runs"
    timestamp = pd.Timestamp.now(tz="UTC").strftime("%Y%m%d_%H%M%S")
    _write_solar_weather_cache(df, archive_dir / f"{source_path.stem}_{timestamp}.csv")


def classify_customer_segments(df: pd.DataFrame) -> pd.Series:
    """
    Classify forecast-eligible solar sites into customer tariff/meter cohorts.
    """
    meter_type = (
        df.get("MeterType", pd.Series("", index=df.index))
        .fillna("")
        .astype(str)
        .str.strip()
        .str.upper()
    )
    rate_schedule = (
        df.get("RateSchedule", pd.Series("", index=df.index))
        .fillna("")
        .astype(str)
        .str.strip()
        .str.upper()
    )

    segment = pd.Series(CUSTOMER_SEGMENT_OTHER, index=df.index, dtype="object")
    nem_meter = meter_type.isin(NET_METER_TYPES)
    segment.loc[nem_meter] = CUSTOMER_SEGMENT_NEM
    solar_20_meter = meter_type.apply(
        lambda value: any(pattern in value for pattern in SOLAR_20_METER_PATTERNS)
    )
    solar_20_rate = rate_schedule.str.startswith("E2") | rate_schedule.str.contains(
        "SL", na=False
    )
    segment.loc[(solar_20_meter | solar_20_rate) & ~nem_meter] = (
        CUSTOMER_SEGMENT_SOLAR_20
    )
    return segment


def parse_customer_segments(value: str) -> list[str]:
    segments = [
        segment.strip().upper() for segment in str(value).split(",") if segment.strip()
    ]
    valid_segments = {
        CUSTOMER_SEGMENT_NEM,
        CUSTOMER_SEGMENT_SOLAR_20,
        CUSTOMER_SEGMENT_OTHER,
    }
    invalid_segments = sorted(set(segments) - valid_segments)
    if invalid_segments:
        raise ValueError(
            "--customer-segments contains unsupported values: "
            + ", ".join(invalid_segments)
            + ". Use NEM, SOLAR_2_0, and/or OTHER."
        )
    return list(dict.fromkeys(segments))


def aggregate_segment_export_to_total(segment_intervals: pd.DataFrame) -> pd.DataFrame:
    """
    Collapse segmented 15-minute export rows back to the legacy total series.
    """
    if segment_intervals.empty or "CustomerSegment" not in segment_intervals.columns:
        return segment_intervals.copy()

    aggregations = {
        "Export_kWh": ("Export_kWh", "sum"),
        "Export_kW": ("Export_kW", "sum"),
        "ExportSource": (
            "ExportSource",
            lambda values: "+".join(sorted(set(values.dropna().astype(str)))),
        ),
    }
    if "SolarElevationDeg" in segment_intervals.columns:
        aggregations["SolarElevationDeg"] = ("SolarElevationDeg", "mean")

    return (
        segment_intervals.groupby("IntervalStartDT", as_index=False)
        .agg(**aggregations)
        .sort_values("IntervalStartDT")
    )


def weather_for_sites(
    weather_df: pd.DataFrame,
    sites: Optional[pd.DataFrame],
    use_capacity_weighted_weather: bool,
) -> pd.DataFrame:
    """
    Return one hourly weather series for the provided site subset.
    """
    if weather_df.empty:
        return weather_df.copy()
    if (
        use_capacity_weighted_weather
        and sites is not None
        and "SolarSiteKey" in weather_df.columns
    ):
        return aggregate_capacity_weighted_weather(weather_df, sites)
    return aggregate_weather_to_hourly(weather_df)


# =============================================================================
# Data Loading
# =============================================================================


def get_total_forecast_eligible_capacity(conn: Engine) -> float:
    """
    Calculate the total forecast-eligible solar capacity from the database.
    """
    logging.info("Calculating total forecast-eligible solar capacity")
    sql = """
    SELECT SUM(SolarCECkW) as TotalKw
    FROM Forecasting.ForecastSolarSite
    WHERE IsForecastEligible = 1;
    """
    total_capacity = pd.read_sql(sql, conn).iloc[0]["TotalKw"]
    if pd.isna(total_capacity) or float(total_capacity) <= 0:
        raise ValueError("No positive forecast-eligible solar capacity found.")
    total_capacity = float(total_capacity)
    logging.info(f"Total forecast-eligible solar capacity: {total_capacity:,.2f} kW")
    return total_capacity


def load_forecast_eligible_solar_sites(conn: Engine) -> pd.DataFrame:
    """
    Load forecast-eligible solar sites used to match REC service point parquet rows.
    """
    logging.info("Loading forecast-eligible solar sites")
    sql = """
    SELECT
        SolarSiteKey,
        LocationNumber,
        SolarCECkW,
        Latitude,
        Longitude,
        LocationClass,
        RateSchedule,
        MeterType,
        InterconnectionDate
    FROM Forecasting.ForecastSolarSite
    WHERE IsForecastEligible = 1
      AND LocationNumber IS NOT NULL
      AND SolarCECkW IS NOT NULL
      AND SolarCECkW > 0;
    """
    df = read_sql(conn, sql)
    if df.empty:
        raise ValueError("No forecast-eligible solar sites found.")

    df["LocationNumber"] = df["LocationNumber"].astype("Int64").astype(str)
    df["SolarCECkW"] = pd.to_numeric(df["SolarCECkW"], errors="coerce")
    df["InterconnectionDate"] = pd.to_datetime(
        df["InterconnectionDate"], errors="coerce"
    ).dt.date
    df = df.dropna(subset=["LocationNumber", "SolarCECkW"])
    df["CustomerSegment"] = classify_customer_segments(df)
    segment_summary = (
        df.groupby("CustomerSegment", as_index=False)
        .agg(Sites=("LocationNumber", "count"), SolarCECkW=("SolarCECkW", "sum"))
        .sort_values("SolarCECkW", ascending=False)
    )
    logging.info(
        "Loaded %s forecast-eligible solar sites totaling %s kW",
        len(df),
        f"{df['SolarCECkW'].sum():,.2f}",
    )
    logging.info(
        "Customer segment capacity: %s",
        "; ".join(
            f"{CUSTOMER_SEGMENT_LABELS.get(row.CustomerSegment, row.CustomerSegment)}="
            f"{row.Sites:,} sites / {row.SolarCECkW:,.2f} kW"
            for row in segment_summary.itertuples(index=False)
        ),
    )
    return df


def load_production_interval_data(conn: Engine) -> pd.DataFrame:
    """
    Load 15-minute historical solar production data for a single representative site.
    """
    logging.info("Loading 15-minute historical production data for a single site")
    sql = """
    WITH LocationWithMostReadings AS (
        SELECT TOP 1
            LocationNumber = SUBSTRING(ServicePointID, 1, CHARINDEX('_', ServicePointID) - 1)
        FROM Forecasting.ForecastSolarProductionInterval15Min
        WHERE CHARINDEX('_', ServicePointID) > 0
        GROUP BY SUBSTRING(ServicePointID, 1, CHARINDEX('_', ServicePointID) - 1)
        ORDER BY COUNT(*) DESC
    )
    SELECT
        IntervalStartDT,
        IntervalValue
    FROM Forecasting.ForecastSolarProductionInterval15Min
    WHERE SUBSTRING(ServicePointID, 1, CHARINDEX('_', ServicePointID) - 1) = (SELECT LocationNumber FROM LocationWithMostReadings)
    AND IntervalValue > 0;
    """
    df = read_sql(conn, sql)
    df["IntervalValue"] = pd.to_numeric(df["IntervalValue"], errors="coerce")
    df = df.dropna(subset=["IntervalStartDT", "IntervalValue"])
    logging.info("Loaded %s historical production intervals for single site", len(df))
    return df


# =============================================================================
# REC Parquet Data
# =============================================================================


def month_range(start_date: date, end_date: date) -> list[int]:
    """
    Return YYYYMM integers touched by the inclusive date range.
    """
    months = []
    current = date(start_date.year, start_date.month, 1)
    end_month = date(end_date.year, end_date.month, 1)

    while current <= end_month:
        months.append(current.year * 100 + current.month)
        if current.month == 12:
            current = date(current.year + 1, 1, 1)
        else:
            current = date(current.year, current.month + 1, 1)

    return months


def add_months(value: date, months: int) -> date:
    """
    Add whole calendar months to a date.
    """
    month_index = value.year * 12 + value.month - 1 + months
    year = month_index // 12
    month = month_index % 12 + 1
    day = min(value.day, pd.Timestamp(year=year, month=month, day=1).days_in_month)
    return date(year, month, day)


def calculate_solar_position(
    timestamps: pd.Series,
    latitude: float,
    longitude: float,
    timezone_name: str,
) -> pd.DataFrame:
    """
    Approximate solar elevation and azimuth for naive local timestamps using NOAA equations.
    """
    timestamp_index = (
        timestamps.index
        if hasattr(timestamps, "index")
        else pd.RangeIndex(len(timestamps))
    )
    local_timestamps = pd.Series(pd.to_datetime(timestamps), index=timestamp_index)
    local_timestamps = local_timestamps + pd.Timedelta(minutes=INTERVAL_HOURS * 60 / 2)
    timezone = ZoneInfo(timezone_name)

    day_of_year = local_timestamps.dt.dayofyear.astype(float)
    local_hour = (
        local_timestamps.dt.hour
        + local_timestamps.dt.minute / 60.0
        + local_timestamps.dt.second / 3600.0
    )
    utc_offset_hours = local_timestamps.apply(
        lambda value: (
            value.to_pydatetime().replace(tzinfo=timezone).utcoffset().total_seconds()
            / 3600.0
            if pd.notna(value)
            else np.nan
        )
    )

    fractional_year = (
        2.0 * math.pi / 365.0 * (day_of_year - 1.0 + (local_hour - 12.0) / 24.0)
    )
    equation_of_time = 229.18 * (
        0.000075
        + 0.001868 * np.cos(fractional_year)
        - 0.032077 * np.sin(fractional_year)
        - 0.014615 * np.cos(2.0 * fractional_year)
        - 0.040849 * np.sin(2.0 * fractional_year)
    )
    declination = (
        0.006918
        - 0.399912 * np.cos(fractional_year)
        + 0.070257 * np.sin(fractional_year)
        - 0.006758 * np.cos(2.0 * fractional_year)
        + 0.000907 * np.sin(2.0 * fractional_year)
        - 0.002697 * np.cos(3.0 * fractional_year)
        + 0.00148 * np.sin(3.0 * fractional_year)
    )

    time_offset_minutes = equation_of_time + 4.0 * longitude - 60.0 * utc_offset_hours
    true_solar_time_minutes = (local_hour * 60.0 + time_offset_minutes) % 1440.0
    hour_angle_degrees = true_solar_time_minutes / 4.0 - 180.0

    latitude_radians = math.radians(latitude)
    hour_angle_radians = np.radians(hour_angle_degrees)
    elevation_radians = np.arcsin(
        np.sin(latitude_radians) * np.sin(declination)
        + np.cos(latitude_radians) * np.cos(declination) * np.cos(hour_angle_radians)
    )
    azimuth_radians = (
        np.arctan2(
            np.sin(hour_angle_radians),
            np.cos(hour_angle_radians) * np.sin(latitude_radians)
            - np.tan(declination) * np.cos(latitude_radians),
        )
        + math.pi
    ) % (2.0 * math.pi)

    return pd.DataFrame(
        {
            "SolarElevationDeg": np.degrees(elevation_radians),
            "SolarAzimuthDeg": np.degrees(azimuth_radians),
        },
        index=timestamp_index,
    )


def calculate_solar_elevation_degrees(
    timestamps: pd.Series,
    latitude: float,
    longitude: float,
    timezone_name: str,
) -> pd.Series:
    """
    Approximate solar elevation for naive local timestamps using NOAA equations.
    """
    return calculate_solar_position(
        timestamps,
        latitude=latitude,
        longitude=longitude,
        timezone_name=timezone_name,
    )["SolarElevationDeg"]


def apply_solar_plausibility_filter(
    df: pd.DataFrame,
    timestamp_col: str,
    energy_col: str,
    power_col: str,
    latitude: float,
    longitude: float,
    timezone_name: str,
    min_solar_elevation: float,
    label: str,
) -> pd.DataFrame:
    """
    Zero energy/power outside the configured solar elevation window.
    """
    out = df.copy()
    out["SolarElevationDeg"] = calculate_solar_elevation_degrees(
        out[timestamp_col],
        latitude=latitude,
        longitude=longitude,
        timezone_name=timezone_name,
    )
    non_solar_mask = out["SolarElevationDeg"] < min_solar_elevation
    removed_kwh = out.loc[non_solar_mask, energy_col].sum()
    removed_intervals = int((non_solar_mask & (out[energy_col] > 0)).sum())
    if removed_intervals:
        logging.info(
            "Zeroed %s %s intervals below %.2f solar elevation degrees; removed %.2f kWh",
            f"{removed_intervals:,}",
            label,
            min_solar_elevation,
            removed_kwh,
        )

    out.loc[non_solar_mask, energy_col] = 0.0
    out.loc[non_solar_mask, power_col] = 0.0
    return out


def _catalog_path(parquet_root: Path) -> Path:
    return parquet_root / INDEX_CACHE_ROOT / "file_catalog.parquet"


def _csv_index_path(parquet_root: Path) -> Path:
    return parquet_root / "_interval_parquet_index.csv"


def _lookup_root(parquet_root: Path) -> Path:
    return parquet_root / INDEX_CACHE_ROOT / "lookup"


def _rate_group_from_path_part(value: str, segment: str) -> str:
    if segment == "RES":
        return "RES"
    cleaned = value.strip().upper()
    if cleaned.startswith("GS") and "-" not in cleaned and len(cleaned) > 2:
        return f"GS-{cleaned[2:]}"
    return cleaned


def _parse_interval_parquet_metadata(
    path: Path, parquet_root: Path
) -> dict[str, str] | None:
    try:
        relative_path = path.relative_to(parquet_root)
    except ValueError:
        return None

    parts = list(relative_path.parts)
    upper_parts = [part.upper() for part in parts]
    if len(parts) < 4 or upper_parts[0] not in {"COM", "RES"}:
        return None

    month_part = next(
        (part for part in parts if part.upper().startswith("MONTHLY_")), ""
    )
    month = month_part.split("_", 1)[1] if "_" in month_part else ""
    if not (len(month) == 6 and month.isdigit()):
        return None

    channel = next(
        (part.upper() for part in parts if part.upper() in {"DEL", "NET", "REC"}), ""
    )
    if not channel:
        return None

    segment = upper_parts[0]
    rate_group_part = next(
        (part for part in parts if part.upper().startswith("GS")), "RES"
    )
    return {
        "relative_path": str(relative_path),
        "segment": segment,
        "channel": channel,
        "rate_group": _rate_group_from_path_part(rate_group_part, segment),
        "nem_status": "NEM" if "NEM" in upper_parts else "Non-NEM",
        "month": month,
    }


def _parquet_columns(path: Path) -> tuple[list[str], str]:
    try:
        import pyarrow.parquet as pq

        return list(pq.read_schema(path).names), ""
    except Exception as exc:
        return [], str(exc)


def rebuild_rec_file_catalog(parquet_root: Path) -> pd.DataFrame:
    """
    Rebuild the interval parquet catalog from the COM/RES parquet folder tree.
    """
    if not parquet_root.exists():
        raise FileNotFoundError(
            f"Parquet root does not exist: {parquet_root}. "
            "Copy the interval parquet folders or run with --parquet-root pointing at them."
        )

    source_files = sorted(
        path
        for segment in ("COM", "RES")
        for path in (parquet_root / segment).rglob("*.parquet")
        if path.is_file()
    )
    if not source_files:
        raise FileNotFoundError(
            f"No COM/RES interval parquet files found under {parquet_root}."
        )

    logging.info(
        "Rebuilding interval parquet catalog from %s parquet files under %s",
        f"{len(source_files):,}",
        parquet_root,
    )
    rows = []
    for file_id, path in enumerate(source_files):
        meta = _parse_interval_parquet_metadata(path, parquet_root)
        if meta is None:
            continue
        stat = path.stat()
        columns, schema_error = _parquet_columns(path)
        rows.append(
            {
                "FileID": file_id,
                "filepath": str(path),
                "filename": path.name,
                "relative_path": meta["relative_path"],
                "folder": str(path.parent),
                "segment": meta["segment"],
                "channel": meta["channel"],
                "rate_group": meta["rate_group"],
                "nem_status": meta["nem_status"],
                "month": meta["month"],
                "size_mb": round(stat.st_size / (1024 * 1024), 3),
                "size_bytes": int(stat.st_size),
                "modified_time": datetime.fromtimestamp(stat.st_mtime).strftime(
                    "%Y-%m-%d %H:%M:%S"
                ),
                "modified_time_ns": int(stat.st_mtime_ns),
                "columns_json": json.dumps(columns),
                "schema_error": schema_error,
            }
        )

    if not rows:
        raise FileNotFoundError(
            f"No recognizable COM/RES Monthly_YYYYMM interval parquet files found under {parquet_root}."
        )

    catalog = pd.DataFrame(rows, columns=CATALOG_COLUMNS)
    catalog_path = _catalog_path(parquet_root)
    catalog_path.parent.mkdir(parents=True, exist_ok=True)
    catalog.to_parquet(catalog_path, index=False)
    catalog.drop(columns=["FileID"]).to_csv(_csv_index_path(parquet_root), index=False)
    logging.info(
        "Wrote interval parquet catalog: %s rows to %s",
        f"{len(catalog):,}",
        catalog_path,
    )
    return catalog


def _normalize_rec_file_catalog(
    catalog: pd.DataFrame, parquet_root: Path
) -> pd.DataFrame:
    """
    Normalize legacy interval indexes and remove non-service-point aggregate parquet files.
    """
    if catalog is None or catalog.empty:
        return pd.DataFrame(columns=CATALOG_COLUMNS)

    out = catalog.copy()
    if "FileID" not in out.columns:
        out = out.reset_index(names="FileID")
    if "filepath" not in out.columns:
        return pd.DataFrame(columns=CATALOG_COLUMNS)

    def _missing(value: object) -> bool:
        return value is None or pd.isna(value)

    normalized_rows = []
    skipped_rows = 0
    for fallback_file_id, row in enumerate(out.to_dict("records")):
        path = Path(str(row.get("filepath", "")))
        meta = _parse_interval_parquet_metadata(path, parquet_root)
        if meta is None:
            skipped_rows += 1
            continue

        stat = path.stat() if path.exists() else None
        file_id = row.get("FileID", fallback_file_id)
        try:
            file_id = int(file_id)
        except Exception:
            file_id = fallback_file_id
        modified_time_ns = row.get("modified_time_ns", None)
        if _missing(modified_time_ns):
            modified_time_ns = int(stat.st_mtime_ns) if stat is not None else 0
        size_bytes = row.get("size_bytes", None)
        if _missing(size_bytes):
            size_bytes = int(stat.st_size) if stat is not None else 0
        modified_time = row.get("modified_time", None)
        if _missing(modified_time):
            modified_time = (
                datetime.fromtimestamp(stat.st_mtime).strftime("%Y-%m-%d %H:%M:%S")
                if stat is not None
                else ""
            )

        normalized_rows.append(
            {
                "FileID": file_id,
                "filepath": str(path),
                "filename": row.get("filename") or path.name,
                "relative_path": meta["relative_path"],
                "folder": row.get("folder") or str(path.parent),
                "segment": meta["segment"],
                "channel": meta["channel"],
                "rate_group": meta["rate_group"],
                "nem_status": meta["nem_status"],
                "month": meta["month"],
                "size_mb": (
                    row.get("size_mb")
                    if not _missing(row.get("size_mb"))
                    else round(float(size_bytes) / (1024 * 1024), 3)
                ),
                "size_bytes": int(size_bytes),
                "modified_time": str(modified_time),
                "modified_time_ns": int(modified_time_ns),
                "columns_json": (
                    row.get("columns_json")
                    if not _missing(row.get("columns_json"))
                    else "[]"
                ),
                "schema_error": (
                    ""
                    if _missing(row.get("schema_error", ""))
                    else str(row.get("schema_error", ""))
                ),
            }
        )

    normalized = pd.DataFrame(normalized_rows, columns=CATALOG_COLUMNS)
    if skipped_rows:
        logging.info(
            "Ignored %s non-interval parquet index rows outside COM/RES Monthly_YYYYMM folders",
            f"{skipped_rows:,}",
        )
    return normalized


def _write_rec_file_catalog_cache(catalog: pd.DataFrame, parquet_root: Path) -> None:
    catalog_path = _catalog_path(parquet_root)
    catalog_path.parent.mkdir(parents=True, exist_ok=True)
    catalog.to_parquet(catalog_path, index=False)
    catalog.drop(columns=["FileID"]).to_csv(_csv_index_path(parquet_root), index=False)


def _read_or_rebuild_rec_file_catalog(parquet_root: Path) -> pd.DataFrame:
    catalog_path = _catalog_path(parquet_root)
    csv_index_path = _csv_index_path(parquet_root)

    if catalog_path.exists():
        catalog = pd.read_parquet(catalog_path)
        normalized = _normalize_rec_file_catalog(catalog, parquet_root)
        if normalized.empty:
            logging.warning(
                "Interval parquet catalog at %s had no usable COM/RES interval rows; rebuilding it.",
                catalog_path,
            )
            return rebuild_rec_file_catalog(parquet_root)
        _write_rec_file_catalog_cache(normalized, parquet_root)
        return normalized
    elif csv_index_path.exists():
        catalog = pd.read_csv(csv_index_path)
        normalized = _normalize_rec_file_catalog(catalog, parquet_root)
        if normalized.empty:
            logging.warning(
                "Interval parquet CSV index at %s had no usable COM/RES interval rows; rebuilding it.",
                csv_index_path,
            )
            return rebuild_rec_file_catalog(parquet_root)
        _write_rec_file_catalog_cache(normalized, parquet_root)
        return normalized

    logging.warning(
        "Interval parquet catalog cache is missing; rebuilding it under %s",
        parquet_root,
    )
    return rebuild_rec_file_catalog(parquet_root)


def rebuild_spid_file_lookup(
    parquet_root: Path, catalog: pd.DataFrame | None = None
) -> pd.DataFrame:
    """
    Rebuild ServicePointID-to-FileID lookup parts from the interval parquet files.
    """
    catalog = (
        catalog.copy()
        if catalog is not None
        else _read_or_rebuild_rec_file_catalog(parquet_root)
    )
    if catalog.empty:
        raise ValueError(
            "Cannot rebuild service point lookup from an empty parquet catalog."
        )
    if "FileID" not in catalog.columns:
        catalog = catalog.reset_index(names="FileID")

    lookup_root = _lookup_root(parquet_root)
    lookup_root.mkdir(parents=True, exist_ok=True)
    for stale_part in lookup_root.glob("bucket=*/*.parquet"):
        stale_part.unlink()

    logging.info(
        "Rebuilding service point file lookup from %s catalog files",
        f"{len(catalog):,}",
    )
    bucket_frames: dict[int, list[pd.DataFrame]] = {bucket: [] for bucket in range(256)}
    read_errors = 0
    for idx, row in enumerate(catalog.itertuples(index=False), start=1):
        path = Path(str(getattr(row, "filepath", "")))
        if not path.exists():
            read_errors += 1
            continue
        try:
            service_points = pd.read_parquet(path, columns=["ServicePointID"])
        except Exception as exc:
            read_errors += 1
            logging.warning(
                "Skipping %s while rebuilding service point lookup: %s", path, exc
            )
            continue

        if service_points.empty or "ServicePointID" not in service_points.columns:
            continue
        spids = service_points["ServicePointID"].dropna().astype(str).str.strip()
        spids = spids[spids.ne("")]
        if spids.empty:
            continue

        counts = spids.value_counts().rename_axis("SPID").reset_index(name="RowCount")
        counts["SPID_BASE"] = counts["SPID"].str.split("_", n=1).str[0]
        counts["FileID"] = int(getattr(row, "FileID"))
        counts = counts[["SPID", "SPID_BASE", "FileID", "RowCount"]]
        bucket = (
            pd.util.hash_pandas_object(counts["SPID_BASE"], index=False)
            .mod(256)
            .astype(int)
        )
        for bucket_id, group in counts.groupby(bucket):
            bucket_frames[int(bucket_id)].append(group.reset_index(drop=True))

        if idx % 500 == 0:
            logging.info(
                "Indexed service points from %s / %s parquet files",
                f"{idx:,}",
                f"{len(catalog):,}",
            )

    part_count = 0
    lookup_rows = 0
    for bucket_id, frames in bucket_frames.items():
        if not frames:
            continue
        part = pd.concat(frames, ignore_index=True)
        part["FileID"] = pd.to_numeric(part["FileID"], errors="coerce").astype("int32")
        part["RowCount"] = (
            pd.to_numeric(part["RowCount"], errors="coerce").fillna(0).astype("int64")
        )
        bucket_dir = lookup_root / f"bucket={bucket_id:03d}"
        bucket_dir.mkdir(parents=True, exist_ok=True)
        part.to_parquet(bucket_dir / "part_00000.parquet", index=False)
        part_count += 1
        lookup_rows += len(part)

    if lookup_rows == 0:
        raise FileNotFoundError(
            f"Could not rebuild service point lookup from parquet files under {parquet_root}; "
            f"{read_errors} files failed or were missing."
        )

    metadata = {
        "created_at": datetime.now().isoformat(timespec="seconds"),
        "catalog_rows": int(len(catalog)),
        "lookup_rows": int(lookup_rows),
        "lookup_parts": int(part_count),
        "read_errors": int(read_errors),
    }
    (_catalog_path(parquet_root).parent / "metadata.json").write_text(
        json.dumps(metadata, indent=2), encoding="utf-8"
    )
    logging.info(
        "Wrote service point lookup: %s rows across %s bucket parts%s",
        f"{lookup_rows:,}",
        f"{part_count:,}",
        f"; skipped {read_errors:,} files" if read_errors else "",
    )
    return load_spid_file_lookup(parquet_root)


def load_rec_file_catalog(
    parquet_root: Path, channels: Optional[set[str]] = None
) -> pd.DataFrame:
    """
    Load the cached interval parquet catalog and keep requested channel files.
    """
    selected_channels = channels or {"REC"}
    catalog = _read_or_rebuild_rec_file_catalog(parquet_root)

    rec_catalog = catalog[catalog["channel"].isin(selected_channels)].copy()
    if rec_catalog.empty:
        raise ValueError(
            f"No {sorted(selected_channels)} parquet files found under {parquet_root}."
        )

    rec_catalog["month"] = pd.to_numeric(rec_catalog["month"], errors="coerce").astype(
        "Int64"
    )
    return rec_catalog.dropna(subset=["month", "filepath", "FileID"])


def get_available_rec_date_range(parquet_root: Path) -> tuple[date, date]:
    """
    Return the approximate available REC date range based on catalog months.
    """
    rec_catalog = load_rec_file_catalog(parquet_root)
    first_month = int(rec_catalog["month"].min())
    last_month = int(rec_catalog["month"].max())
    first_date = date(first_month // 100, first_month % 100, 1)
    last_month_start = pd.Timestamp(
        year=last_month // 100, month=last_month % 100, day=1
    )
    last_date = last_month_start + pd.offsets.MonthEnd(0)
    return first_date, last_date.date()


def load_spid_file_lookup(parquet_root: Path) -> pd.DataFrame:
    """
    Load cached ServicePointID-to-FileID mappings.
    """
    lookup_root = _lookup_root(parquet_root)
    if not lookup_root.exists():
        logging.warning(
            "Service point lookup cache is missing; rebuilding it under %s",
            parquet_root,
        )
        return rebuild_spid_file_lookup(parquet_root)

    lookup_files = sorted(lookup_root.glob("bucket=*/*.parquet"))
    if not lookup_files:
        logging.warning(
            "Service point lookup cache has no parquet parts; rebuilding it under %s",
            parquet_root,
        )
        return rebuild_spid_file_lookup(parquet_root)

    logging.info(
        "Loading service point file lookup from %s parquet parts", len(lookup_files)
    )
    lookup = pd.concat(
        [
            pd.read_parquet(path, columns=["SPID", "SPID_BASE", "FileID", "RowCount"])
            for path in lookup_files
        ],
        ignore_index=True,
    )
    lookup["SPID_BASE"] = lookup["SPID_BASE"].astype(str)
    lookup["FileID"] = pd.to_numeric(lookup["FileID"], errors="coerce").astype("Int64")
    lookup["RowCount"] = pd.to_numeric(lookup["RowCount"], errors="coerce").fillna(0)
    return lookup.dropna(subset=["SPID", "SPID_BASE", "FileID"])


def load_rec_interval_data(
    parquet_root: Path,
    sites: pd.DataFrame,
    start_date: date,
    end_date: date,
    net_meter_export_source: str = "net",
    latitude: float = ROSEVILLE_LATITUDE,
    longitude: float = ROSEVILLE_LONGITUDE,
    timezone_name: str = "America/Los_Angeles",
    min_solar_elevation: float = 0.0,
    group_by_customer_segment: bool = False,
) -> pd.DataFrame:
    """
    Aggregate parquet rows for forecast-eligible solar locations into 15-minute system export.

    REC rows are interval kWh received by the utility from the customer. This is
    solar exported to the grid, not gross behind-the-meter PV production. For
    AMI_NET/AMI_NET_D sites, export can be derived from negative NET intervals.
    """
    logging.info("Loading solar export intervals from %s to %s", start_date, end_date)

    if net_meter_export_source not in {"net", "rec"}:
        raise ValueError("net_meter_export_source must be 'net' or 'rec'.")

    catalog_channels = {"REC", "NET"} if net_meter_export_source == "net" else {"REC"}
    rec_catalog = load_rec_file_catalog(parquet_root, catalog_channels)
    wanted_months = set(month_range(start_date, end_date))
    rec_catalog = rec_catalog[
        rec_catalog["month"].astype(int).isin(wanted_months)
    ].copy()
    if rec_catalog.empty:
        raise ValueError(
            f"No export parquet files found for months {sorted(wanted_months)}."
        )

    sites = sites.copy()
    sites["MeterType"] = (
        sites["MeterType"].fillna("").astype(str).str.strip().str.upper()
    )
    if "CustomerSegment" not in sites.columns:
        sites["CustomerSegment"] = classify_customer_segments(sites)
    net_meter_locations = set(
        sites.loc[sites["MeterType"].isin(NET_METER_TYPES), "LocationNumber"].astype(
            str
        )
    )
    all_solar_locations = set(sites["LocationNumber"].astype(str))
    rec_locations = (
        all_solar_locations - net_meter_locations
        if net_meter_export_source == "net"
        else all_solar_locations
    )
    lookup = load_spid_file_lookup(parquet_root)

    source_lookups = []
    if rec_locations:
        rec_files = rec_catalog[rec_catalog["channel"].eq("REC")]
        rec_file_ids = set(rec_files["FileID"].astype(int))
        rec_lookup = lookup[
            lookup["FileID"].astype(int).isin(rec_file_ids)
            & lookup["SPID_BASE"].isin(rec_locations)
        ].copy()
        rec_lookup["ExportSource"] = "REC"
        source_lookups.append(rec_lookup)

    if net_meter_export_source == "net" and net_meter_locations:
        net_files = rec_catalog[
            rec_catalog["channel"].eq("NET") & rec_catalog["nem_status"].eq("NEM")
        ]
        net_file_ids = set(net_files["FileID"].astype(int))
        net_lookup = lookup[
            lookup["FileID"].astype(int).isin(net_file_ids)
            & lookup["SPID_BASE"].isin(net_meter_locations)
        ].copy()
        net_lookup["ExportSource"] = "NET_NEGATIVE"
        source_lookups.append(net_lookup)

    lookup = (
        pd.concat(source_lookups, ignore_index=True)
        if source_lookups
        else pd.DataFrame()
    )
    if lookup.empty:
        raise ValueError(
            "No forecast-eligible solar service points found in export parquet lookup for requested dates."
        )

    lookup["FileID"] = lookup["FileID"].astype(int)
    dedupe_meta = rec_catalog[["FileID", "month", "modified_time_ns"]].copy()
    dedupe_meta["FileID"] = dedupe_meta["FileID"].astype(int)
    lookup = lookup.merge(dedupe_meta, on="FileID", how="left")
    lookup = lookup.merge(
        sites[["LocationNumber", "CustomerSegment"]].drop_duplicates("LocationNumber"),
        left_on="SPID_BASE",
        right_on="LocationNumber",
        how="left",
    )
    lookup["CustomerSegment"] = lookup["CustomerSegment"].fillna(CUSTOMER_SEGMENT_OTHER)
    lookup["month"] = pd.to_numeric(lookup["month"], errors="coerce").astype("Int64")
    lookup["modified_time_ns"] = pd.to_numeric(
        lookup["modified_time_ns"], errors="coerce"
    ).fillna(0)

    lookup_rows_before_dedupe = len(lookup)
    duplicate_file_mappings = int(
        lookup.duplicated(["SPID", "ExportSource", "month"], keep=False).sum()
    )
    lookup = lookup.sort_values(
        ["SPID", "ExportSource", "month", "RowCount", "modified_time_ns", "FileID"],
        ascending=[True, True, True, False, False, False],
    ).drop_duplicates(["SPID", "ExportSource", "month"], keep="first")
    removed_file_mappings = lookup_rows_before_dedupe - len(lookup)
    if removed_file_mappings:
        logging.info(
            "Removed %s duplicate SPID/month/source file mappings from %s duplicated mappings",
            f"{removed_file_mappings:,}",
            f"{duplicate_file_mappings:,}",
        )

    service_points = lookup["SPID"].nunique()
    matched_locations = lookup["SPID_BASE"].nunique()
    logging.info(
        "Matched %s export service points across %s forecast-eligible solar locations and %s files",
        service_points,
        matched_locations,
        lookup["FileID"].nunique(),
    )

    filepath_by_id = rec_catalog.set_index("FileID")["filepath"].to_dict()
    source_by_id = lookup.groupby("FileID")["ExportSource"].first().to_dict()
    spids_by_file = lookup.groupby("FileID")["SPID"].apply(list).to_dict()
    segment_by_spid = (
        lookup.drop_duplicates("SPID").set_index("SPID")["CustomerSegment"].to_dict()
    )
    interval_start_min = pd.Timestamp(start_date)
    interval_start_max = pd.Timestamp(end_date) + pd.Timedelta(days=1)

    pieces = []
    total_rows = 0
    total_duplicate_interval_rows = 0
    for index, (file_id, service_points_in_file) in enumerate(
        spids_by_file.items(), start=1
    ):
        path = filepath_by_id.get(int(file_id))
        if not path:
            continue

        if index % 50 == 0:
            logging.info(
                "Processed %s / %s export parquet files", index, len(spids_by_file)
            )

        df = pd.read_parquet(
            path,
            columns=["ServicePointID", "ReadingValue_kWh", "EndTimePST"],
            filters=[("ServicePointID", "in", service_points_in_file)],
        )
        if df.empty:
            continue

        df["IntervalStartDT"] = pd.to_datetime(
            df["EndTimePST"], errors="coerce"
        ) - pd.Timedelta(minutes=15)
        reading_kwh = pd.to_numeric(df["ReadingValue_kWh"], errors="coerce")
        export_source = source_by_id.get(file_id, "REC")
        if export_source == "NET_NEGATIVE":
            df["Export_kWh"] = (-reading_kwh).clip(lower=0)
        else:
            df["Export_kWh"] = reading_kwh.clip(lower=0)

        df = df[
            (df["IntervalStartDT"] >= interval_start_min)
            & (df["IntervalStartDT"] < interval_start_max)
        ].dropna(subset=["IntervalStartDT", "Export_kWh"])
        if df.empty:
            continue

        duplicate_interval_rows = int(
            df.duplicated(["ServicePointID", "IntervalStartDT"]).sum()
        )
        if duplicate_interval_rows:
            total_duplicate_interval_rows += duplicate_interval_rows
            df = df.drop_duplicates(["ServicePointID", "IntervalStartDT"], keep="last")

        total_rows += len(df)
        if group_by_customer_segment:
            df["CustomerSegment"] = (
                df["ServicePointID"].map(segment_by_spid).fillna(CUSTOMER_SEGMENT_OTHER)
            )
            group_columns = ["IntervalStartDT", "CustomerSegment"]
        else:
            group_columns = ["IntervalStartDT"]
        interval_piece = df.groupby(group_columns, as_index=False)["Export_kWh"].sum()
        interval_piece["ExportSource"] = export_source
        pieces.append(interval_piece)

    if not pieces:
        raise ValueError(
            "No export interval rows found for requested dates after filtering."
        )

    if group_by_customer_segment:
        group_columns = ["IntervalStartDT", "CustomerSegment"]
    else:
        group_columns = ["IntervalStartDT"]
    intervals = (
        pd.concat(pieces, ignore_index=True)
        .groupby(group_columns, as_index=False)
        .agg(
            Export_kWh=("Export_kWh", "sum"),
            ExportSource=("ExportSource", lambda values: "+".join(sorted(set(values)))),
        )
        .sort_values(group_columns)
    )
    intervals["Export_kW"] = intervals["Export_kWh"] / INTERVAL_HOURS
    intervals = apply_solar_plausibility_filter(
        intervals,
        timestamp_col="IntervalStartDT",
        energy_col="Export_kWh",
        power_col="Export_kW",
        latitude=latitude,
        longitude=longitude,
        timezone_name=timezone_name,
        min_solar_elevation=min_solar_elevation,
        label="historical export",
    )
    logging.info(
        "Aggregated %s unique export rows into %s %s intervals; total export %.2f MWh, peak %.2f MW",
        f"{total_rows:,}",
        f"{len(intervals):,}",
        "segmented" if group_by_customer_segment else "system",
        intervals["Export_kWh"].sum() / 1000.0,
        intervals["Export_kW"].max() / 1000.0,
    )
    if group_by_customer_segment:
        segment_summary = (
            intervals.groupby("CustomerSegment", as_index=False)
            .agg(
                Export_MWh=("Export_kWh", lambda values: values.sum() / 1000.0),
                Peak_MW=("Export_kW", lambda values: values.max() / 1000.0),
            )
            .sort_values("Export_MWh", ascending=False)
        )
        logging.info(
            "Segment export totals: %s",
            "; ".join(
                f"{CUSTOMER_SEGMENT_LABELS.get(row.CustomerSegment, row.CustomerSegment)}="
                f"{row.Export_MWh:,.2f} MWh / {row.Peak_MW:,.2f} MW peak"
                for row in segment_summary.itertuples(index=False)
            ),
        )
    if total_duplicate_interval_rows:
        logging.info(
            "Skipped %s duplicate ServicePointID/interval rows inside selected parquet files",
            f"{total_duplicate_interval_rows:,}",
        )
    return intervals


def build_complete_time_shape() -> pd.DataFrame:
    """
    Build a complete 15-minute clock-time index for shape outputs.
    """
    return pd.DataFrame(
        {"time": pd.date_range("2000-01-01", periods=96, freq="15min").time}
    )


def build_average_daily_shape(
    interval_df: pd.DataFrame,
    power_col: str,
    method: str = DEFAULT_DAILY_SHAPE_METHOD,
    quantile: float = DEFAULT_SHAPE_QUANTILE,
) -> pd.DataFrame:
    """
    Build a robust 15-minute daily export shape from historical interval data.
    """
    shape_df = interval_df.copy()
    shape_df["IntervalStartDT"] = pd.to_datetime(shape_df["IntervalStartDT"])
    shape_df["date"] = shape_df["IntervalStartDT"].dt.date
    shape_df["time"] = shape_df["IntervalStartDT"].dt.time
    shape_df[power_col] = pd.to_numeric(shape_df[power_col], errors="coerce").fillna(
        0.0
    )

    complete_shape = build_complete_time_shape()
    if method == "mean":
        average_shape = (
            shape_df.groupby("time", as_index=False)[power_col]
            .mean()
            .rename(columns={power_col: "Average_kW"})
        )
    else:
        daily_peaks = (
            shape_df.groupby("date", as_index=False)[power_col]
            .max()
            .rename(columns={power_col: "DailyPeak_kW"})
        )
        valid_daily_peaks = daily_peaks[daily_peaks["DailyPeak_kW"] > 0].copy()
        if valid_daily_peaks.empty:
            average_shape = complete_shape.copy()
            average_shape["Average_kW"] = 0.0
        else:
            shape_df = shape_df.merge(valid_daily_peaks, on="date", how="inner")
            shape_df["NormalizedPower"] = shape_df[power_col] / shape_df["DailyPeak_kW"]
            if method == "median":
                normalized_shape = shape_df.groupby("time", as_index=False)[
                    "NormalizedPower"
                ].median()
                reference_peak_kw = valid_daily_peaks["DailyPeak_kW"].median()
            elif method == "upper-quantile":
                normalized_shape = shape_df.groupby("time", as_index=False)[
                    "NormalizedPower"
                ].quantile(quantile)
                reference_peak_kw = valid_daily_peaks["DailyPeak_kW"].quantile(quantile)
            else:
                raise ValueError(f"Unsupported daily shape method: {method!r}")

            average_shape = complete_shape.merge(
                normalized_shape, on="time", how="left"
            )
            max_normalized_power = average_shape["NormalizedPower"].max(skipna=True)
            if pd.isna(max_normalized_power) or max_normalized_power <= 0:
                average_shape["Average_kW"] = 0.0
            else:
                average_shape["Average_kW"] = (
                    average_shape["NormalizedPower"].fillna(0.0)
                    / max_normalized_power
                    * reference_peak_kw
                )
            average_shape = average_shape.drop(columns=["NormalizedPower"])

    average_shape = complete_shape.merge(average_shape, on="time", how="left")
    average_shape["Average_kW"] = average_shape["Average_kW"].fillna(0.0)
    average_shape["ShapeMethod"] = method
    average_shape["ShapeQuantile"] = quantile if method == "upper-quantile" else pd.NA
    return average_shape


def build_intrahour_production_shape(
    interval_df: pd.DataFrame,
    energy_col: str,
    method: str = DEFAULT_INTRAHOUR_SHAPE_METHOD,
    quantile: float = DEFAULT_SHAPE_QUANTILE,
) -> pd.DataFrame:
    """
    Build normalized 15-minute weights within each hour from interval energy data.
    """
    shape_df = interval_df.copy()
    shape_df["IntervalStartDT"] = pd.to_datetime(shape_df["IntervalStartDT"])
    shape_df["date"] = shape_df["IntervalStartDT"].dt.date
    shape_df["hour"] = shape_df["IntervalStartDT"].dt.hour
    shape_df["minute"] = shape_df["IntervalStartDT"].dt.minute
    shape_df[energy_col] = pd.to_numeric(shape_df[energy_col], errors="coerce").fillna(
        0.0
    )

    hourly_totals = (
        shape_df.groupby(["date", "hour"], as_index=False)[energy_col]
        .sum()
        .rename(columns={energy_col: "TotalHourlyProduction_kWh"})
    )
    shape_df = shape_df.merge(hourly_totals, on=["date", "hour"], how="left")
    shape_df = shape_df[shape_df["TotalHourlyProduction_kWh"] > 0].copy()
    shape_df["IntraHourCoefficient"] = (
        shape_df[energy_col] / shape_df["TotalHourlyProduction_kWh"]
    )

    if method == "mean":
        observed_shape = shape_df.groupby(["hour", "minute"], as_index=False)[
            "IntraHourCoefficient"
        ].mean()
    elif method == "median":
        observed_shape = shape_df.groupby(["hour", "minute"], as_index=False)[
            "IntraHourCoefficient"
        ].median()
    elif method == "upper-quantile":
        observed_shape = shape_df.groupby(["hour", "minute"], as_index=False)[
            "IntraHourCoefficient"
        ].quantile(quantile)
    else:
        raise ValueError(f"Unsupported intra-hour shape method: {method!r}")

    complete_shape = pd.MultiIndex.from_product(
        [range(24), [0, 15, 30, 45]],
        names=["hour", "minute"],
    ).to_frame(index=False)
    complete_shape = complete_shape.merge(
        observed_shape, on=["hour", "minute"], how="left"
    )

    normalized_groups = []
    for hour, group in complete_shape.groupby("hour", sort=True):
        group = group.copy()
        coefficient_sum = group["IntraHourCoefficient"].sum(skipna=True)
        if pd.isna(coefficient_sum) or coefficient_sum <= 0:
            group["IntraHourCoefficient"] = 0.25
        else:
            group["IntraHourCoefficient"] = (
                group["IntraHourCoefficient"].fillna(0.0) / coefficient_sum
            )
        normalized_groups.append(group)

    intrahour_shape = pd.concat(normalized_groups, ignore_index=True)
    intrahour_shape["ShapeMethod"] = method
    intrahour_shape["ShapeQuantile"] = quantile if method == "upper-quantile" else pd.NA
    return intrahour_shape


def get_default_rec_history_window(
    parquet_root: Path, history_months: int
) -> tuple[date, date]:
    """
    Pick the most recent complete REC parquet months for calibration.
    """
    _, available_end = get_available_rec_date_range(parquet_root)
    history_start_month = add_months(
        date(available_end.year, available_end.month, 1), -(history_months - 1)
    )
    return history_start_month, available_end


def calculate_active_capacity_for_timestamps(
    timestamps: pd.Series,
    sites: Optional[pd.DataFrame],
    default_capacity_kw: float,
) -> pd.Series:
    """
    Estimate active system capacity by timestamp using site interconnection dates.
    """
    timestamp_series = pd.to_datetime(pd.Series(timestamps), errors="coerce")
    fallback = pd.Series(
        float(default_capacity_kw), index=timestamp_series.index, dtype="float64"
    )
    if (
        timestamp_series.dropna().empty
        or sites is None
        or sites.empty
        or "SolarCECkW" not in sites.columns
    ):
        return fallback

    site_capacity = pd.to_numeric(sites["SolarCECkW"], errors="coerce")
    valid_capacity = site_capacity.notna() & (site_capacity > 0)
    if not valid_capacity.any():
        return fallback

    site_total_capacity = float(site_capacity[valid_capacity].sum())
    if "InterconnectionDate" in sites.columns:
        interconnection_series = pd.to_datetime(
            sites["InterconnectionDate"], errors="coerce"
        )
    else:
        interconnection_series = pd.Series(pd.NaT, index=sites.index)
    if interconnection_series.isna().all():
        return pd.Series(
            site_total_capacity, index=timestamp_series.index, dtype="float64"
        )

    min_date = timestamp_series.dropna().min().date()
    max_date = timestamp_series.dropna().max().date()
    # Sites already interconnected before the modeled window must contribute
    # from the first modeled day. Without this floor, capacity is understated
    # whenever the history window starts after existing fleet buildout.
    effective_dates = interconnection_series.dt.date.where(
        interconnection_series.notna(), min_date
    )
    effective_dates = pd.Series(effective_dates, index=sites.index)
    effective_dates = effective_dates.where(effective_dates >= min_date, min_date)
    increments = (
        pd.DataFrame(
            {
                "EffectiveDate": effective_dates[valid_capacity],
                "SolarCECkW": site_capacity[valid_capacity],
            }
        )
        .groupby("EffectiveDate", as_index=True)["SolarCECkW"]
        .sum()
        .sort_index()
    )

    date_index = pd.date_range(min_date, max_date, freq="D")
    cumulative_capacity = increments.reindex(date_index.date, fill_value=0.0).cumsum()
    active_capacity = pd.Series(
        timestamp_series.dt.date.map(cumulative_capacity.to_dict()),
        index=timestamp_series.index,
        dtype="float64",
    )
    if active_capacity.max(skipna=True) <= 0:
        return fallback

    return active_capacity.clip(lower=0.0, upper=site_total_capacity).fillna(fallback)


def build_daily_active_capacity(
    sites: pd.DataFrame,
    start_date: date,
    end_date: date,
) -> pd.DataFrame:
    """
    Build one active-capacity row per date from site interconnection dates.
    """
    empty = pd.DataFrame(columns=["Date", "ActiveCapacity_kW"])
    if (
        sites is None
        or sites.empty
        or "SolarCECkW" not in sites.columns
        or start_date > end_date
    ):
        return empty

    dates = pd.date_range(start_date, end_date, freq="D")
    capacity = calculate_active_capacity_for_timestamps(
        pd.Series(dates),
        sites=sites,
        default_capacity_kw=float(
            pd.to_numeric(sites["SolarCECkW"], errors="coerce").fillna(0.0).sum()
        ),
    )
    return pd.DataFrame(
        {
            "Date": dates.date,
            "ActiveCapacity_kW": pd.to_numeric(capacity, errors="coerce")
            .fillna(0.0)
            .to_numpy(),
        }
    )


def _resolve_row_capacity(
    timestamps: pd.Series,
    daily_active_capacity: Optional[pd.DataFrame],
    fallback_capacity_kw: float,
) -> pd.Series:
    """
    Resolve timestamp capacity from a daily active-capacity frame, falling back to a scalar.
    """
    timestamp_series = pd.to_datetime(pd.Series(timestamps), errors="coerce")
    fallback = pd.Series(
        float(fallback_capacity_kw), index=timestamp_series.index, dtype="float64"
    )
    if daily_active_capacity is None or daily_active_capacity.empty:
        return fallback
    if (
        "Date" not in daily_active_capacity.columns
        or "ActiveCapacity_kW" not in daily_active_capacity.columns
    ):
        return fallback

    daily = daily_active_capacity.copy()
    daily["Date"] = pd.to_datetime(daily["Date"], errors="coerce").dt.date
    daily["ActiveCapacity_kW"] = pd.to_numeric(
        daily["ActiveCapacity_kW"], errors="coerce"
    )
    lookup = (
        daily.dropna(subset=["Date"])
        .drop_duplicates("Date", keep="last")
        .set_index("Date")["ActiveCapacity_kW"]
    )
    resolved = timestamp_series.dt.date.map(lookup)
    return pd.to_numeric(resolved, errors="coerce").fillna(fallback).clip(lower=0.0)


# =============================================================================
# Weather Data
# =============================================================================


def shortwave_radiation_to_kwh_per_m2(values: pd.Series, unit: str) -> pd.Series:
    """
    Convert Open-Meteo shortwave_radiation_sum values to kWh/m^2.
    """
    normalized_unit = (
        (unit or "").replace("\u00b2", "2").replace("^2", "2").strip().lower()
    )
    numeric_values = pd.to_numeric(values, errors="coerce")

    if normalized_unit == "mj/m2":
        return numeric_values / 3.6
    if normalized_unit == "wh/m2":
        return numeric_values / 1000.0
    if normalized_unit == "kwh/m2":
        return numeric_values

    raise ValueError(f"Unsupported Open-Meteo shortwave_radiation_sum unit: {unit!r}")


def hourly_irradiance_to_kwh_per_m2(values: pd.Series, unit: str) -> pd.Series:
    """
    Convert Open-Meteo hourly irradiance values to hourly kWh/m^2.
    """
    normalized_unit = (
        (unit or "").replace("\u00b2", "2").replace("^2", "2").strip().lower()
    )
    numeric_values = pd.to_numeric(values, errors="coerce")

    if normalized_unit in {"w/m2", "wm2"}:
        return numeric_values / 1000.0
    if normalized_unit == "wh/m2":
        return numeric_values / 1000.0
    if normalized_unit == "kwh/m2":
        return numeric_values
    if normalized_unit == "mj/m2":
        return numeric_values / 3.6

    raise ValueError(f"Unsupported Open-Meteo hourly irradiance unit: {unit!r}")


def temperature_to_celsius(values: pd.Series, unit: str) -> pd.Series:
    """
    Convert hourly temperature values to Celsius.
    """
    normalized_unit = (unit or "").replace("°", "").replace(" ", "").strip().lower()
    numeric_values = pd.to_numeric(values, errors="coerce")

    if normalized_unit in {"", "c", "degc", "celsius"}:
        return numeric_values
    if normalized_unit in {"f", "degf", "fahrenheit"}:
        return (numeric_values - 32.0) * (5.0 / 9.0)
    if normalized_unit in {"k", "kelvin"}:
        return numeric_values - 273.15

    logging.warning(
        "Unsupported hourly temperature unit %r; leaving values as-is", unit
    )
    return numeric_values


def wind_speed_to_mps(values: pd.Series, unit: str) -> pd.Series:
    """
    Convert hourly wind speed values to meters per second.
    """
    normalized_unit = (unit or "").replace(" ", "").strip().lower()
    numeric_values = pd.to_numeric(values, errors="coerce")

    if normalized_unit in {
        "",
        "m/s",
        "ms",
        "ms-1",
        "meterpersecond",
        "meterspersecond",
    }:
        return numeric_values
    if normalized_unit in {"km/h", "kmh", "kph"}:
        return numeric_values / 3.6
    if normalized_unit in {"mph", "mi/h"}:
        return numeric_values * 0.44704
    if normalized_unit in {"kn", "knot", "knots"}:
        return numeric_values * 0.514444

    logging.warning("Unsupported hourly wind-speed unit %r; leaving values as-is", unit)
    return numeric_values


def array_azimuth_to_open_meteo(array_azimuth_degrees: float) -> float:
    """
    Convert compass azimuth (0 north, 180 south) to Open-Meteo panel azimuth.

    Open-Meteo uses 0 for south, -90 for east, 90 for west, and +/-180 for north.
    """
    return ((float(array_azimuth_degrees) - 180.0 + 180.0) % 360.0) - 180.0


def retry_after_seconds(response: requests.Response, fallback_seconds: float) -> float:
    """
    Parse Retry-After when Open-Meteo provides it; otherwise use exponential fallback.
    """
    retry_after = response.headers.get("Retry-After")
    if retry_after:
        try:
            return min(float(retry_after), OPEN_METEO_RETRY_MAX_SECONDS)
        except ValueError:
            logging.debug("Could not parse Retry-After header %r", retry_after)

    return min(fallback_seconds, OPEN_METEO_RETRY_MAX_SECONDS)


def request_open_meteo_json(
    url: str,
    params: dict,
    source_name: str,
    batch_num: int = 1,
    batch_count: int = 1,
) -> list[dict]:
    """
    Execute an Open-Meteo request with bounded retries for rate limits/transient errors.
    """
    for attempt in range(OPEN_METEO_MAX_RETRIES + 1):
        response = requests.get(url, params=params, verify=False, timeout=60)
        if response.status_code not in OPEN_METEO_TRANSIENT_STATUS_CODES:
            response.raise_for_status()
            results = response.json()
            return results if isinstance(results, list) else [results]

        if attempt >= OPEN_METEO_MAX_RETRIES:
            logging.error(
                "Open-Meteo %s request failed with HTTP %s after %s attempts for batch %s/%s",
                source_name,
                response.status_code,
                attempt + 1,
                batch_num,
                batch_count,
            )
            response.raise_for_status()

        fallback_delay = OPEN_METEO_RETRY_BASE_SECONDS * (2**attempt)
        delay_seconds = retry_after_seconds(response, fallback_delay)
        logging.warning(
            "Open-Meteo %s request returned HTTP %s for batch %s/%s; retrying in %.1f seconds",
            source_name,
            response.status_code,
            batch_num,
            batch_count,
            delay_seconds,
        )
        time.sleep(delay_seconds)

    raise RuntimeError("Open-Meteo retry loop exited unexpectedly.")


def normalize_weather_site_keys(sites: pd.DataFrame) -> set[str]:
    """
    Return comparable site keys for checking reusable weather coverage.
    """
    if sites.empty or "SolarSiteKey" not in sites.columns:
        return set()
    return set(sites["SolarSiteKey"].dropna().astype(str))


def normalize_hourly_weather_frame(weather_df: pd.DataFrame) -> pd.DataFrame:
    """
    Normalize an hourly weather frame enough for date filtering and de-duplication.
    """
    if weather_df.empty or "IntervalStartDT" not in weather_df.columns:
        return pd.DataFrame()

    out = weather_df.copy()
    out["IntervalStartDT"] = pd.to_datetime(out["IntervalStartDT"], errors="coerce")
    out = out.dropna(subset=["IntervalStartDT"])
    out["date"] = out["IntervalStartDT"].dt.date
    return out


def collect_reusable_hourly_weather(
    weather_frames: Optional[list[pd.DataFrame]],
    start_date: date,
    end_date: date,
    expected_site_keys: set[str],
) -> pd.DataFrame:
    """
    Return already-fetched hourly weather rows that can satisfy part of a requested window.
    """
    if not weather_frames:
        return pd.DataFrame()

    normalized_frames = [
        normalize_hourly_weather_frame(frame)
        for frame in weather_frames
        if frame is not None and not frame.empty
    ]
    normalized_frames = [frame for frame in normalized_frames if not frame.empty]
    if not normalized_frames:
        return pd.DataFrame()

    reusable = pd.concat(normalized_frames, ignore_index=True)
    reusable = reusable[
        (reusable["date"] >= start_date) & (reusable["date"] <= end_date)
    ].copy()
    if reusable.empty:
        return reusable

    if expected_site_keys and "SolarSiteKey" in reusable.columns:
        site_key_mask = (
            reusable["SolarSiteKey"]
            .where(
                reusable["SolarSiteKey"].notna(),
                "",
            )
            .astype(str)
            .isin(expected_site_keys)
        )
        reusable = reusable[site_key_mask].copy()

    dedupe_keys = ["IntervalStartDT"]
    if "SolarSiteKey" in reusable.columns:
        dedupe_keys.insert(0, "SolarSiteKey")
    return reusable.drop_duplicates(subset=dedupe_keys, keep="last")


def contiguous_date_ranges(dates: list[date]) -> list[tuple[date, date]]:
    """
    Collapse individual dates into inclusive contiguous ranges.
    """
    if not dates:
        return []

    sorted_dates = sorted(dates)
    ranges: list[tuple[date, date]] = []
    range_start = sorted_dates[0]
    previous = sorted_dates[0]
    for current in sorted_dates[1:]:
        if current == previous + timedelta(days=1):
            previous = current
            continue
        ranges.append((range_start, previous))
        range_start = current
        previous = current
    ranges.append((range_start, previous))
    return ranges


def chunk_date_range(
    range_start: date, range_end: date, max_days: int
) -> list[tuple[date, date]]:
    """
    Split an inclusive date range into smaller inclusive ranges.
    """
    if range_start > range_end:
        return []
    chunk_days = max(1, int(max_days))
    chunks: list[tuple[date, date]] = []
    current = range_start
    while current <= range_end:
        chunk_end = min(range_end, current + timedelta(days=chunk_days - 1))
        chunks.append((current, chunk_end))
        current = chunk_end + timedelta(days=1)
    return chunks


def missing_hourly_weather_ranges(
    reusable_weather: pd.DataFrame,
    start_date: date,
    end_date: date,
    expected_site_keys: set[str],
) -> list[tuple[date, date]]:
    """
    Find dates not fully covered by reusable weather rows.
    """
    requested_dates = [
        value.date() for value in pd.date_range(start_date, end_date, freq="D")
    ]
    if reusable_weather.empty:
        return contiguous_date_ranges(requested_dates)

    reusable = normalize_hourly_weather_frame(reusable_weather)
    if reusable.empty:
        return contiguous_date_ranges(requested_dates)

    if expected_site_keys and "SolarSiteKey" in reusable.columns:
        coverage = reusable[["date", "SolarSiteKey"]].dropna().drop_duplicates()
        coverage["SolarSiteKey"] = coverage["SolarSiteKey"].astype(str)
        covered_dates = {
            weather_date
            for weather_date, group in coverage.groupby("date")
            if expected_site_keys.issubset(set(group["SolarSiteKey"]))
        }
    else:
        covered_dates = set(reusable["date"].dropna())

    missing_dates = [
        weather_date
        for weather_date in requested_dates
        if weather_date not in covered_dates
    ]
    return contiguous_date_ranges(missing_dates)


def fetch_historical_weather(
    sites: pd.DataFrame,
    start_date: date,
    end_date: date,
) -> pd.DataFrame:
    """
    Fetch historical GHI from Open-Meteo for a list of sites.
    """
    return fetch_open_meteo_weather(sites, start_date, end_date, use_forecast=False)


def fetch_forecast_weather(
    sites: pd.DataFrame,
    start_date: date,
    end_date: date,
) -> pd.DataFrame:
    """
    Fetch forecast GHI from Open-Meteo for a list of sites.
    """
    return fetch_open_meteo_weather(sites, start_date, end_date, use_forecast=True)


def fetch_open_meteo_weather(
    sites: pd.DataFrame,
    start_date: date,
    end_date: date,
    use_forecast: bool,
) -> pd.DataFrame:
    """
    Fetch daily GHI from Open-Meteo archive or forecast API.
    The Open-Meteo forecast API is limited to 16 days.
    """
    source_name = "forecast" if use_forecast else "historical"
    logging.info(
        "Fetching %s weather data from %s to %s", source_name, start_date, end_date
    )

    url = (
        "https://api.open-meteo.com/v1/forecast"
        if use_forecast
        else "https://archive-api.open-meteo.com/v1/archive"
    )

    # API can handle multiple locations in one request
    params = {
        "latitude": sites["Latitude"].tolist(),
        "longitude": sites["Longitude"].tolist(),
        "start_date": start_date.isoformat(),
        "end_date": end_date.isoformat(),
        "daily": "shortwave_radiation_sum",
        "timezone": "auto",
    }

    results = request_open_meteo_json(url, params, source_name)

    all_weather_data = []
    for i, site_weather in enumerate(results):
        site_id = sites.iloc[i]["SolarSiteKey"]
        temp_df = pd.DataFrame(site_weather["daily"])
        temp_df["SolarSiteKey"] = site_id
        radiation_unit = site_weather.get("daily_units", {}).get(
            "shortwave_radiation_sum", ""
        )
        temp_df["GHI_kWh_per_m2"] = shortwave_radiation_to_kwh_per_m2(
            temp_df["shortwave_radiation_sum"],
            radiation_unit,
        )
        all_weather_data.append(temp_df)

    weather_df = pd.concat(all_weather_data, ignore_index=True)
    weather_df.rename(columns={"time": "date"}, inplace=True)
    weather_df["date"] = pd.to_datetime(weather_df["date"]).dt.date

    logging.info("Fetched %s weather data points", len(weather_df))
    return weather_df


def fetch_open_meteo_hourly_weather(
    sites: pd.DataFrame,
    start_date: date,
    end_date: date,
    use_forecast: bool,
    timezone_name: str,
    array_tilt_degrees: float = DEFAULT_ARRAY_TILT_DEGREES,
    array_azimuth_degrees: float = DEFAULT_ARRAY_AZIMUTH_DEGREES,
    weather_locations_per_request: int = DEFAULT_WEATHER_LOCATIONS_PER_REQUEST,
    cache_dir: str | Path | None = DEFAULT_SOLAR_WEATHER_CACHE_DIR,
    forecast_cache_max_age_hours: float = DEFAULT_FORECAST_WEATHER_CACHE_MAX_AGE_HOURS,
    use_cache: bool = True,
) -> pd.DataFrame:
    """
    Fetch hourly GHI and cloud cover from Open-Meteo archive or forecast API.
    """
    source_name = "forecast" if use_forecast else "historical"
    logging.info(
        "Fetching hourly %s weather data from %s to %s",
        source_name,
        start_date,
        end_date,
    )
    cache_path = _solar_weather_cache_path(
        cache_dir=cache_dir,
        kind="hourly",
        source_name=source_name,
        start_date=start_date,
        end_date=end_date,
        sites=sites,
        timezone_name=timezone_name,
        variables=HOURLY_WEATHER_VARIABLES,
    )
    cached = pd.DataFrame()
    if use_cache:
        cached = _read_solar_weather_cache(
            cache_path,
            start_date=start_date,
            end_date=end_date,
            timestamp_col="IntervalStartDT",
        )
        if not cached.empty and (
            not use_forecast
            or _solar_weather_cache_is_fresh(cache_path, forecast_cache_max_age_hours)
        ):
            logging.info(
                "Using cached %s hourly solar weather: %s", source_name, cache_path
            )
            return cached

    url = (
        "https://api.open-meteo.com/v1/forecast"
        if use_forecast
        else "https://archive-api.open-meteo.com/v1/archive"
    )
    all_weather_data = []
    locations_per_request = max(1, int(weather_locations_per_request))
    site_batches = [
        sites.iloc[i : i + locations_per_request]
        for i in range(0, len(sites), locations_per_request)
    ]

    for batch_num, site_batch in enumerate(site_batches, start=1):
        if len(site_batches) > 1:
            logging.info(
                "Fetching weather batch %s of %s (%s sites)",
                batch_num,
                len(site_batches),
                len(site_batch),
            )
        params = {
            "latitude": site_batch["Latitude"].tolist(),
            "longitude": site_batch["Longitude"].tolist(),
            "start_date": start_date.isoformat(),
            "end_date": end_date.isoformat(),
            "hourly": ",".join(HOURLY_WEATHER_VARIABLES),
            "timezone": timezone_name,
            "wind_speed_unit": "ms",
            "tilt": float(array_tilt_degrees),
            "azimuth": array_azimuth_to_open_meteo(array_azimuth_degrees),
        }

        try:
            results = _open_meteo_get_json(
                url,
                params,
                source_name=source_name,
                start_date=start_date,
                end_date=end_date,
            )
        except Exception as exc:
            if use_cache and not cached.empty:
                logging.warning(
                    "Open-Meteo %s hourly solar weather refresh failed; using stale cache %s. Details: %s",
                    source_name,
                    cache_path,
                    exc,
                )
                return cached
            raise
        if isinstance(results, dict):
            results = [results]

        for i, site_weather in enumerate(results):
            hourly = site_weather.get("hourly")
            if not hourly:
                continue

            site_id = site_batch.iloc[i]["SolarSiteKey"]
            temp_df = pd.DataFrame(hourly)
            temp_df["SolarSiteKey"] = site_id
            temp_df.rename(columns={"time": "IntervalStartDT"}, inplace=True)
            temp_df["IntervalStartDT"] = pd.to_datetime(
                temp_df["IntervalStartDT"], errors="coerce"
            )

            hourly_units = site_weather.get("hourly_units", {})
            radiation_unit = hourly_units.get("shortwave_radiation", "W/m\u00b2")
            direct_radiation_unit = hourly_units.get("direct_radiation", radiation_unit)
            direct_normal_unit = hourly_units.get(
                "direct_normal_irradiance", radiation_unit
            )
            diffuse_radiation_unit = hourly_units.get(
                "diffuse_radiation", radiation_unit
            )
            global_tilted_unit = hourly_units.get(
                "global_tilted_irradiance", radiation_unit
            )
            temperature_unit = hourly_units.get("temperature_2m", "\u00b0C")
            dew_point_unit = hourly_units.get("dew_point_2m", temperature_unit)
            apparent_temperature_unit = hourly_units.get(
                "apparent_temperature", temperature_unit
            )
            wind_speed_unit = hourly_units.get("wind_speed_10m", "m/s")
            wind_gust_unit = hourly_units.get("wind_gusts_10m", wind_speed_unit)
            temp_df["WeatherGHI_Wm2"] = pd.to_numeric(
                temp_df.get("shortwave_radiation"), errors="coerce"
            )
            temp_df["GHI_kWh_per_m2"] = hourly_irradiance_to_kwh_per_m2(
                temp_df["shortwave_radiation"],
                radiation_unit,
            )

            direct_radiation_values = temp_df.get("direct_radiation")
            if direct_radiation_values is not None:
                temp_df["DirectRadiation_Wm2"] = pd.to_numeric(
                    direct_radiation_values, errors="coerce"
                )
                temp_df["DirectRadiation_kWh_per_m2"] = hourly_irradiance_to_kwh_per_m2(
                    direct_radiation_values,
                    direct_radiation_unit,
                )
            else:
                temp_df["DirectRadiation_Wm2"] = pd.NA
                temp_df["DirectRadiation_kWh_per_m2"] = pd.NA

            direct_normal_values = temp_df.get("direct_normal_irradiance")
            if direct_normal_values is not None:
                temp_df["DirectNormalIrradiance_Wm2"] = pd.to_numeric(
                    direct_normal_values, errors="coerce"
                )
                if direct_normal_unit != radiation_unit:
                    temp_df["DirectNormalIrradiance_Wm2"] = (
                        hourly_irradiance_to_kwh_per_m2(
                            direct_normal_values, direct_normal_unit
                        )
                        * 1000.0
                    )
            else:
                temp_df["DirectNormalIrradiance_Wm2"] = pd.NA

            diffuse_radiation_values = temp_df.get("diffuse_radiation")
            if diffuse_radiation_values is not None:
                temp_df["DiffuseRadiation_Wm2"] = pd.to_numeric(
                    diffuse_radiation_values, errors="coerce"
                )
                temp_df["DiffuseRadiation_kWh_per_m2"] = (
                    hourly_irradiance_to_kwh_per_m2(
                        diffuse_radiation_values,
                        diffuse_radiation_unit,
                    )
                )
            else:
                temp_df["DiffuseRadiation_Wm2"] = pd.NA
                temp_df["DiffuseRadiation_kWh_per_m2"] = pd.NA

            global_tilted_values = temp_df.get("global_tilted_irradiance")
            if global_tilted_values is not None:
                temp_df["GlobalTiltedIrradiance_Wm2"] = pd.to_numeric(
                    global_tilted_values, errors="coerce"
                )
                temp_df["GlobalTiltedIrradiance_kWh_per_m2"] = (
                    hourly_irradiance_to_kwh_per_m2(
                        global_tilted_values,
                        global_tilted_unit,
                    )
                )
            else:
                temp_df["GlobalTiltedIrradiance_Wm2"] = pd.NA
                temp_df["GlobalTiltedIrradiance_kWh_per_m2"] = pd.NA

            temperature_values = temp_df.get("temperature_2m")
            if temperature_values is not None:
                temp_df["Temperature2m_C"] = temperature_to_celsius(
                    temperature_values, temperature_unit
                )
            else:
                temp_df["Temperature2m_C"] = pd.NA
            temp_df["Temperature_C"] = temp_df["Temperature2m_C"]

            relative_humidity_values = temp_df.get("relative_humidity_2m")
            temp_df["RelativeHumidity2mPct"] = (
                pd.to_numeric(relative_humidity_values, errors="coerce")
                if relative_humidity_values is not None
                else pd.NA
            )

            dew_point_values = temp_df.get("dew_point_2m")
            temp_df["DewPoint2m_C"] = (
                temperature_to_celsius(dew_point_values, dew_point_unit)
                if dew_point_values is not None
                else pd.NA
            )

            apparent_temperature_values = temp_df.get("apparent_temperature")
            temp_df["ApparentTemperature_C"] = (
                temperature_to_celsius(
                    apparent_temperature_values, apparent_temperature_unit
                )
                if apparent_temperature_values is not None
                else pd.NA
            )

            surface_pressure_values = temp_df.get("surface_pressure")
            temp_df["SurfacePressure_hPa"] = (
                pd.to_numeric(surface_pressure_values, errors="coerce")
                if surface_pressure_values is not None
                else pd.NA
            )

            wind_speed_values = temp_df.get("wind_speed_10m")
            if wind_speed_values is not None:
                temp_df["WindSpeed10m_mps"] = wind_speed_to_mps(
                    wind_speed_values, wind_speed_unit
                )
            else:
                temp_df["WindSpeed10m_mps"] = pd.NA
            temp_df["WindSpeed_ms"] = temp_df["WindSpeed10m_mps"]

            wind_direction_values = temp_df.get("wind_direction_10m")
            temp_df["WindDirection10mDeg"] = (
                pd.to_numeric(wind_direction_values, errors="coerce")
                if wind_direction_values is not None
                else pd.NA
            )

            wind_gust_values = temp_df.get("wind_gusts_10m")
            temp_df["WindGusts10m_mps"] = (
                wind_speed_to_mps(wind_gust_values, wind_gust_unit)
                if wind_gust_values is not None
                else pd.NA
            )

            vpd_values = temp_df.get("vapour_pressure_deficit")
            temp_df["VapourPressureDeficit_kPa"] = (
                pd.to_numeric(vpd_values, errors="coerce")
                if vpd_values is not None
                else pd.NA
            )

            precipitation_values = temp_df.get("precipitation")
            temp_df["Precipitation_mm"] = (
                pd.to_numeric(precipitation_values, errors="coerce")
                if precipitation_values is not None
                else pd.NA
            )

            weather_code_values = temp_df.get("weather_code")
            temp_df["WeatherCode"] = (
                pd.to_numeric(weather_code_values, errors="coerce")
                if weather_code_values is not None
                else pd.NA
            )

            sunshine_duration_values = temp_df.get("sunshine_duration")
            temp_df["SunshineDurationSec"] = (
                pd.to_numeric(sunshine_duration_values, errors="coerce")
                if sunshine_duration_values is not None
                else pd.NA
            )

            is_day_values = temp_df.get("is_day")
            temp_df["IsDay"] = (
                pd.to_numeric(is_day_values, errors="coerce")
                if is_day_values is not None
                else pd.NA
            )

            cloud_columns = {
                "cloud_cover": "CloudCoverPct",
                "cloud_cover_low": "CloudCoverLowPct",
                "cloud_cover_mid": "CloudCoverMidPct",
                "cloud_cover_high": "CloudCoverHighPct",
            }
            for source_col, target_col in cloud_columns.items():
                if source_col in temp_df.columns:
                    temp_df[target_col] = pd.to_numeric(
                        temp_df[source_col], errors="coerce"
                    )
                else:
                    temp_df[target_col] = pd.NA

            temp_df["date"] = temp_df["IntervalStartDT"].dt.date
            for column in WEATHER_OUTPUT_COLUMNS:
                if column not in temp_df.columns:
                    temp_df[column] = pd.NA
            keep_columns = [
                "SolarSiteKey",
                "IntervalStartDT",
                "date",
                *WEATHER_OUTPUT_COLUMNS,
            ]
            temp_df = temp_df[keep_columns].dropna(
                subset=["IntervalStartDT", "GHI_kWh_per_m2"]
            )
            temp_df = temp_df[
                (temp_df["date"] >= start_date) & (temp_df["date"] <= end_date)
            ]
            all_weather_data.append(temp_df)

        if len(site_batches) > 1 and batch_num < len(site_batches):
            time.sleep(1)  # Respect API rate limits

    if not all_weather_data:
        raise ValueError(
            f"No hourly weather data returned from Open-Meteo for {start_date} to {end_date}."
        )

    weather_df = pd.concat(all_weather_data, ignore_index=True)
    if use_cache:
        _write_solar_weather_cache(weather_df, cache_path)
        if use_forecast:
            _archive_solar_forecast_weather(weather_df, cache_dir, cache_path)
    logging.info("Fetched %s hourly weather data points", len(weather_df))
    return weather_df


def fetch_weather_for_date_range(
    sites: pd.DataFrame,
    start_date: date,
    end_date: date,
) -> pd.DataFrame:
    """
    Fetch archive data for past dates and forecast data for today/future dates.
    """
    today = date.today()
    frames = []

    archive_end = min(end_date, today - timedelta(days=1))
    if start_date <= archive_end:
        frames.append(fetch_historical_weather(sites, start_date, archive_end))

    forecast_start = max(start_date, today)
    if forecast_start <= end_date:
        frames.append(fetch_forecast_weather(sites, forecast_start, end_date))

    if not frames:
        frames.append(fetch_historical_weather(sites, start_date, end_date))

    return pd.concat(frames, ignore_index=True)


def fetch_hourly_weather_for_date_range(
    sites: pd.DataFrame,
    start_date: date,
    end_date: date,
    timezone_name: str,
    array_tilt_degrees: float = DEFAULT_ARRAY_TILT_DEGREES,
    array_azimuth_degrees: float = DEFAULT_ARRAY_AZIMUTH_DEGREES,
    weather_locations_per_request: int = DEFAULT_WEATHER_LOCATIONS_PER_REQUEST,
    cache_dir: str | Path | None = DEFAULT_SOLAR_WEATHER_CACHE_DIR,
    forecast_cache_max_age_hours: float = DEFAULT_FORECAST_WEATHER_CACHE_MAX_AGE_HOURS,
    reusable_weather_frames: Optional[list[pd.DataFrame]] = None,
) -> pd.DataFrame:
    """
    Fetch hourly archive data for past dates and hourly forecast data for today/future dates.
    """
    today = current_local_timestamp(timezone_name).date()
    frames = []
    expected_site_keys = normalize_weather_site_keys(sites)
    reusable_weather = collect_reusable_hourly_weather(
        reusable_weather_frames,
        start_date,
        end_date,
        expected_site_keys,
    )
    if not reusable_weather.empty:
        logging.info(
            "Reusing %s previously fetched hourly weather rows from %s to %s",
            f"{len(reusable_weather):,}",
            reusable_weather["date"].min(),
            reusable_weather["date"].max(),
        )

    def append_weather_for_missing_ranges(
        range_start: date,
        range_end: date,
        use_forecast: bool,
    ) -> None:
        source_name = "forecast" if use_forecast else "historical"
        if reusable_weather.empty or "date" not in reusable_weather.columns:
            reusable_subset = pd.DataFrame()
        else:
            reusable_subset = reusable_weather[
                (reusable_weather["date"] >= range_start)
                & (reusable_weather["date"] <= range_end)
            ].copy()
        cached_subset = _read_overlapping_solar_weather_caches(
            cache_dir=cache_dir,
            kind="hourly",
            source_name=source_name,
            start_date=range_start,
            end_date=range_end,
            sites=sites,
            timezone_name=timezone_name,
            variables=HOURLY_WEATHER_VARIABLES,
            timestamp_col="IntervalStartDT",
            forecast_cache_max_age_hours=forecast_cache_max_age_hours,
        )
        available_subset = collect_reusable_hourly_weather(
            [reusable_subset, cached_subset],
            range_start,
            range_end,
            expected_site_keys,
        )
        if not available_subset.empty:
            logging.info(
                "Reusing %s %s hourly solar weather rows from memory/cache for %s to %s",
                f"{len(available_subset):,}",
                source_name,
                range_start,
                range_end,
            )
            frames.append(available_subset)

        for missing_start, missing_end in missing_hourly_weather_ranges(
            available_subset,
            range_start,
            range_end,
            expected_site_keys,
        ):
            for fetch_start, fetch_end in chunk_date_range(
                missing_start,
                missing_end,
                DEFAULT_HOURLY_WEATHER_FETCH_CHUNK_DAYS,
            ):
                frames.append(
                    fetch_open_meteo_hourly_weather(
                        sites,
                        fetch_start,
                        fetch_end,
                        use_forecast,
                        timezone_name,
                        array_tilt_degrees=array_tilt_degrees,
                        array_azimuth_degrees=array_azimuth_degrees,
                        weather_locations_per_request=weather_locations_per_request,
                        cache_dir=cache_dir,
                        forecast_cache_max_age_hours=forecast_cache_max_age_hours,
                    )
                )

    archive_end = min(end_date, today - timedelta(days=1))
    if start_date <= archive_end:
        append_weather_for_missing_ranges(start_date, archive_end, use_forecast=False)

    forecast_start = max(start_date, today)
    if forecast_start <= end_date:
        append_weather_for_missing_ranges(forecast_start, end_date, use_forecast=True)

    if not frames:
        frames.append(
            fetch_open_meteo_hourly_weather(
                sites,
                start_date,
                end_date,
                False,
                timezone_name,
                array_tilt_degrees=array_tilt_degrees,
                array_azimuth_degrees=array_azimuth_degrees,
                weather_locations_per_request=weather_locations_per_request,
                cache_dir=cache_dir,
            )
        )

    weather = normalize_hourly_weather_frame(pd.concat(frames, ignore_index=True))
    if weather.empty:
        return weather

    dedupe_keys = ["IntervalStartDT"]
    if "SolarSiteKey" in weather.columns:
        dedupe_keys.insert(0, "SolarSiteKey")
    return weather.drop_duplicates(subset=dedupe_keys, keep="last").sort_values(
        dedupe_keys
    )


# =============================================================================
# Forecast Model
# =============================================================================


def build_system_weather_site(latitude: float, longitude: float) -> pd.DataFrame:
    """
    Use a representative Roseville point for system-wide weather.
    """
    return pd.DataFrame(
        [
            {
                "SolarSiteKey": 1,
                "LocationNumber": "Roseville, CA",
                "Latitude": latitude,
                "Longitude": longitude,
            }
        ]
    )


def build_weather_clusters(
    sites: pd.DataFrame, n_clusters: int
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """
    Group sites into geographic clusters and sample one real site per cluster.

    The sampled rows keep Open-Meteo calls bounded while avoiding synthetic
    coordinates. Each sampled row uses the cluster id as ``SolarSiteKey`` so it
    can be capacity-weighted back to the full fleet by ``WeatherCluster``.
    """
    sites_with_coords = sites.dropna(subset=["Latitude", "Longitude"]).copy()
    for column in ["Latitude", "Longitude", "SolarCECkW"]:
        sites_with_coords[column] = pd.to_numeric(
            sites_with_coords[column], errors="coerce"
        )
    sites_with_coords = sites_with_coords.dropna(subset=["Latitude", "Longitude"])
    if len(sites_with_coords) < n_clusters:
        logging.warning(
            "Number of sites with coordinates (%s) is less than n_clusters (%s). "
            "Using number of sites as n_clusters.",
            len(sites_with_coords),
            n_clusters,
        )
        n_clusters = len(sites_with_coords)

    if n_clusters == 0:
        return sites, sites_with_coords

    kmeans = KMeans(n_clusters=n_clusters, random_state=42, n_init=10)
    sites_with_coords["WeatherCluster"] = kmeans.fit_predict(
        sites_with_coords[["Latitude", "Longitude"]]
    )

    # Merge the cluster information back into the original sites dataframe
    out_sites = sites.merge(
        sites_with_coords[["SolarSiteKey", "WeatherCluster"]],
        on="SolarSiteKey",
        how="left",
    )

    sampled_rows = []
    for cluster_id, group in sites_with_coords.groupby("WeatherCluster", sort=True):
        group = group.copy()
        weights = pd.to_numeric(group["SolarCECkW"], errors="coerce").fillna(0.0)
        if weights.gt(0).any():
            target_latitude = float(np.average(group["Latitude"], weights=weights))
            target_longitude = float(np.average(group["Longitude"], weights=weights))
        else:
            target_latitude = float(group["Latitude"].mean())
            target_longitude = float(group["Longitude"].mean())

        distance = (group["Latitude"] - target_latitude) ** 2 + (
            group["Longitude"] - target_longitude
        ) ** 2
        representative = group.loc[distance.idxmin()]
        cluster_capacity_kw = float(weights.sum())
        sampled_rows.append(
            {
                "SolarSiteKey": int(cluster_id),
                "LocationNumber": f"Weather Sample {int(cluster_id)}: {representative.get('LocationNumber', '')}",
                "Latitude": float(representative["Latitude"]),
                "Longitude": float(representative["Longitude"]),
                "SampleSolarSiteKey": representative.get("SolarSiteKey"),
                "SampleLocationNumber": representative.get("LocationNumber"),
                "ClusterCapacity_kW": cluster_capacity_kw,
            }
        )

    weather_samples = (
        pd.DataFrame(sampled_rows).sort_values("SolarSiteKey").reset_index(drop=True)
    )
    logging.info(
        "Selected %s representative solar sites for weather sampling; largest cluster %.2f kW, total %.2f kW",
        len(weather_samples),
        (
            weather_samples["ClusterCapacity_kW"].max()
            if not weather_samples.empty
            else 0.0
        ),
        (
            weather_samples["ClusterCapacity_kW"].sum()
            if not weather_samples.empty
            else 0.0
        ),
    )
    return out_sites, weather_samples


def weighted_weather_average(
    group: pd.DataFrame, weather_columns: list[str]
) -> pd.Series:
    """
    Capacity-weight weather columns for one timestamp.
    """
    weights = pd.to_numeric(group["SolarCECkW"], errors="coerce")
    valid_weight = weights.notna() & (weights > 0)
    values = {}

    for column in weather_columns:
        series = pd.to_numeric(group[column], errors="coerce")
        valid = valid_weight & series.notna()
        if valid.any():
            values[column] = float(
                (series[valid] * weights[valid]).sum() / weights[valid].sum()
            )
        else:
            values[column] = pd.NA

    return pd.Series(values)


def aggregate_capacity_weighted_weather(
    weather_df: pd.DataFrame, sites: pd.DataFrame
) -> pd.DataFrame:
    """
    Convert per-site or per-cluster weather rows to one capacity-weighted system series.
    """
    if weather_df.empty:
        return weather_df
    if "SolarSiteKey" not in weather_df.columns:
        logging.warning(
            "Weather rows have no SolarSiteKey; using unweighted hourly weather average."
        )
        return aggregate_weather_to_hourly(weather_df)

    if "WeatherCluster" in sites.columns:
        capacity_by_key = sites[["WeatherCluster", "SolarCECkW"]].copy()
        capacity_by_key["WeightKey"] = pd.to_numeric(
            capacity_by_key["WeatherCluster"], errors="coerce"
        )
        key_label = "weather clusters"
    else:
        capacity_by_key = sites[["SolarSiteKey", "SolarCECkW"]].copy()
        capacity_by_key["WeightKey"] = pd.to_numeric(
            capacity_by_key["SolarSiteKey"], errors="coerce"
        )
        key_label = "sites"

    capacity_by_key["SolarCECkW"] = pd.to_numeric(
        capacity_by_key["SolarCECkW"], errors="coerce"
    )
    capacity_by_key = (
        capacity_by_key.dropna(subset=["WeightKey", "SolarCECkW"])
        .query("SolarCECkW > 0")
        .groupby("WeightKey", as_index=False)["SolarCECkW"]
        .sum()
    )
    if capacity_by_key.empty:
        logging.warning(
            "No positive capacity weights found for weather aggregation; using unweighted hourly average."
        )
        return aggregate_weather_to_hourly(weather_df)

    weather_key = pd.to_numeric(weather_df["SolarSiteKey"], errors="coerce")
    weather = pd.DataFrame(
        {
            "IntervalStartDT": pd.to_datetime(
                weather_df["IntervalStartDT"], errors="coerce"
            ),
            "WeightKey": weather_key,
        }
    )
    for column in WEATHER_OUTPUT_COLUMNS:
        if column in weather_df.columns:
            weather[column] = pd.to_numeric(weather_df[column], errors="coerce").astype(
                "float64"
            )
        else:
            weather[column] = np.nan

    weather = weather.dropna(subset=["IntervalStartDT", "WeightKey"]).merge(
        capacity_by_key,
        on="WeightKey",
        how="left",
    )
    weights = pd.to_numeric(weather["SolarCECkW"], errors="coerce").astype("float64")
    valid_weight = weights.notna() & (weights > 0)
    if not valid_weight.any():
        logging.warning(
            "Weather rows did not match positive capacity weights; using unweighted hourly average."
        )
        return aggregate_weather_to_hourly(weather_df)

    work = pd.DataFrame({"IntervalStartDT": weather["IntervalStartDT"]})
    sum_columns = []
    output_pairs = {}
    for index, column in enumerate(WEATHER_OUTPUT_COLUMNS):
        values = pd.to_numeric(weather[column], errors="coerce").astype("float64")
        valid = valid_weight & values.notna()
        weighted_column = f"WeightedValue{index}"
        weight_column = f"Weight{index}"
        work[weighted_column] = np.where(valid, values * weights, 0.0)
        work[weight_column] = np.where(valid, weights, 0.0)
        sum_columns.extend([weighted_column, weight_column])
        output_pairs[column] = (weighted_column, weight_column)

    grouped = work.groupby("IntervalStartDT", sort=True, as_index=False)[
        sum_columns
    ].sum()
    out = pd.DataFrame({"IntervalStartDT": grouped["IntervalStartDT"]})
    for column, (weighted_column, weight_column) in output_pairs.items():
        denominator = grouped[weight_column].replace(0.0, np.nan)
        out[column] = grouped[weighted_column] / denominator

    logging.info(
        "Capacity-weighted weather across %s from %s source rows to %s hourly rows",
        key_label,
        f"{len(weather_df):,}",
        f"{len(out):,}",
    )
    return out


def aggregate_weather_to_hourly(weather_df: pd.DataFrame) -> pd.DataFrame:
    """
    Collapse weather rows to one row per timestamp.
    """
    weather = weather_df.copy()
    weather["IntervalStartDT"] = pd.to_datetime(weather["IntervalStartDT"])
    for column in WEATHER_OUTPUT_COLUMNS:
        if column not in weather.columns:
            weather[column] = np.nan
        weather[column] = pd.to_numeric(weather[column], errors="coerce")

    return (
        weather.groupby("IntervalStartDT", as_index=False)
        .agg({column: "mean" for column in WEATHER_OUTPUT_COLUMNS})
        .sort_values("IntervalStartDT")
    )


def add_physics_features(
    df: pd.DataFrame,
    latitude: float,
    longitude: float,
    timezone_name: str,
    array_tilt_degrees: float = DEFAULT_ARRAY_TILT_DEGREES,
    array_azimuth_degrees: float = DEFAULT_ARRAY_AZIMUTH_DEGREES,
) -> pd.DataFrame:
    """
    Add PVWatts-style clear-sky, plane-of-array, and temperature derate features.
    """
    out = df.loc[:, ~df.columns.duplicated()].copy()
    solar_position = calculate_solar_position(
        out["IntervalStartDT"],
        latitude=latitude,
        longitude=longitude,
        timezone_name=timezone_name,
    )
    out["SolarElevationDeg"] = solar_position["SolarElevationDeg"]
    out["SolarAzimuthDeg"] = solar_position["SolarAzimuthDeg"]

    elevation_radians = np.radians(out["SolarElevationDeg"])
    solar_zenith_cos = pd.Series(np.sin(elevation_radians), index=out.index).clip(
        lower=0.0
    )
    clear_sky_ghi = pd.Series(0.0, index=out.index, dtype="float64")
    daylight_mask = solar_zenith_cos > 0.0
    clear_sky_ghi.loc[daylight_mask] = (
        1098.0
        * solar_zenith_cos.loc[daylight_mask]
        * np.exp(-0.059 / solar_zenith_cos.loc[daylight_mask].clip(lower=0.065))
    )
    out["ClearSkyGHI_Wm2"] = clear_sky_ghi.clip(lower=0.0)
    out["ClearSkyIndex"] = (
        (out["WeatherGHI_Wm2"] / out["ClearSkyGHI_Wm2"].replace(0.0, np.nan))
        .clip(lower=0.0, upper=1.6)
        .fillna(0.0)
    )

    ghi_wm2 = out["WeatherGHI_Wm2"].clip(lower=0.0).fillna(0.0)
    direct_wm2 = out["DirectRadiation_Wm2"].clip(lower=0.0)
    dni_wm2 = out["DirectNormalIrradiance_Wm2"].clip(lower=0.0)
    diffuse_wm2 = out["DiffuseRadiation_Wm2"].clip(lower=0.0)
    global_tilted_wm2 = out["GlobalTiltedIrradiance_Wm2"].clip(lower=0.0)
    missing_direct = direct_wm2.isna()
    missing_diffuse = diffuse_wm2.isna()
    fallback_diffuse_fraction = (
        0.25 + 0.55 * (1.0 - out["ClearSkyIndex"].clip(0.0, 1.0))
    ).clip(0.20, 0.85)
    diffuse_wm2 = diffuse_wm2.fillna(ghi_wm2 * fallback_diffuse_fraction)
    direct_wm2 = direct_wm2.fillna((ghi_wm2 - diffuse_wm2).clip(lower=0.0))
    direct_wm2 = direct_wm2.where(
        ~missing_diffuse | ~missing_direct, (ghi_wm2 - diffuse_wm2).clip(lower=0.0)
    )

    tilt_radians = math.radians(array_tilt_degrees)
    array_azimuth_radians = math.radians(array_azimuth_degrees)
    solar_zenith_radians = np.radians(90.0 - out["SolarElevationDeg"])
    solar_azimuth_radians = np.radians(out["SolarAzimuthDeg"])
    incidence_cos = np.cos(solar_zenith_radians) * math.cos(tilt_radians) + np.sin(
        solar_zenith_radians
    ) * math.sin(tilt_radians) * np.cos(solar_azimuth_radians - array_azimuth_radians)
    out["IncidenceAngleCos"] = pd.Series(incidence_cos, index=out.index).clip(
        lower=0.0, upper=1.0
    )

    beam_geometry_factor = (
        (out["IncidenceAngleCos"] / solar_zenith_cos.replace(0.0, np.nan))
        .clip(lower=0.0, upper=4.0)
        .fillna(0.0)
    )
    direct_poa_from_horizontal = (direct_wm2 * beam_geometry_factor).clip(lower=0.0)
    direct_poa_from_dni = (dni_wm2 * out["IncidenceAngleCos"]).clip(lower=0.0)
    out["DirectPOA_Wm2"] = direct_poa_from_dni.where(
        dni_wm2.notna(), direct_poa_from_horizontal
    )
    out["DiffusePOA_Wm2"] = (diffuse_wm2 * (1.0 + math.cos(tilt_radians)) / 2.0).clip(
        lower=0.0
    )
    ground_reflected_wm2 = (
        ghi_wm2 * GROUND_ALBEDO * (1.0 - math.cos(tilt_radians)) / 2.0
    )
    modeled_poa_wm2 = (
        out["DirectPOA_Wm2"] + out["DiffusePOA_Wm2"] + ground_reflected_wm2
    ).clip(lower=0.0)
    out["PlaneOfArrayIrradiance_Wm2"] = global_tilted_wm2.where(
        global_tilted_wm2.notna(),
        modeled_poa_wm2,
    ).clip(lower=0.0)
    out["PlaneOfArray_kWh_per_m2"] = out["PlaneOfArrayIrradiance_Wm2"] / 1000.0

    cell_temperature = (
        out["Temperature2m_C"]
        + (out["PlaneOfArrayIrradiance_Wm2"] / 800.0) * (PV_NOCT_C - 20.0)
        - out["WindSpeed10m_mps"].clip(lower=0.0) * PV_WIND_COOLING_C_PER_MPS
    )
    out["CellTemperature_C"] = cell_temperature.fillna(out["Temperature2m_C"]).fillna(
        25.0
    )
    out["TemperatureDerate"] = (
        1.0 + PV_TEMPERATURE_COEFFICIENT_PER_C * (out["CellTemperature_C"] - 25.0)
    ).clip(lower=0.70, upper=1.08)
    out["PVWatts_kWh_per_kW"] = (
        out["PlaneOfArray_kWh_per_m2"] * out["TemperatureDerate"]
    ).clip(lower=0.0)
    return out


def add_performance_features(
    df: pd.DataFrame,
    latitude: float,
    longitude: float,
    timezone_name: str,
    array_tilt_degrees: float = DEFAULT_ARRAY_TILT_DEGREES,
    array_azimuth_degrees: float = DEFAULT_ARRAY_AZIMUTH_DEGREES,
) -> pd.DataFrame:
    """
    Add weather/time features used by the performance-ratio model.
    """
    out = df.loc[:, ~df.columns.duplicated()].copy()
    out["IntervalStartDT"] = pd.to_datetime(out["IntervalStartDT"])
    for column in WEATHER_OUTPUT_COLUMNS:
        if column not in out.columns:
            out[column] = np.nan
        out[column] = pd.to_numeric(out[column], errors="coerce")

    out["Temperature2m_C"] = out["Temperature2m_C"].fillna(out["Temperature_C"])
    out["Temperature_C"] = out["Temperature_C"].fillna(out["Temperature2m_C"])
    out["WindSpeed10m_mps"] = out["WindSpeed10m_mps"].fillna(out["WindSpeed_ms"])
    out["WindSpeed_ms"] = out["WindSpeed_ms"].fillna(out["WindSpeed10m_mps"])
    out["GHI_kWh_per_m2"] = (
        out["GHI_kWh_per_m2"].fillna(out["WeatherGHI_Wm2"] / 1000.0).clip(lower=0)
    )
    out["WeatherGHI_Wm2"] = out["WeatherGHI_Wm2"].fillna(out["GHI_kWh_per_m2"] * 1000.0)
    out["DirectRadiation_Wm2"] = out["DirectRadiation_Wm2"].clip(lower=0)
    out["DirectRadiation_kWh_per_m2"] = (
        out["DirectRadiation_kWh_per_m2"]
        .fillna(out["DirectRadiation_Wm2"] / 1000.0)
        .clip(lower=0)
    )
    out["DirectRadiation_Wm2"] = out["DirectRadiation_Wm2"].fillna(
        out["DirectRadiation_kWh_per_m2"] * 1000.0
    )
    out["DirectNormalIrradiance_Wm2"] = out["DirectNormalIrradiance_Wm2"].clip(
        lower=0.0
    )

    out["DiffuseRadiation_Wm2"] = out["DiffuseRadiation_Wm2"].clip(lower=0)
    out["DiffuseRadiation_kWh_per_m2"] = (
        out["DiffuseRadiation_kWh_per_m2"]
        .fillna(out["DiffuseRadiation_Wm2"] / 1000.0)
        .clip(lower=0)
    )
    out["DiffuseRadiation_Wm2"] = out["DiffuseRadiation_Wm2"].fillna(
        out["DiffuseRadiation_kWh_per_m2"] * 1000.0
    )

    out["GlobalTiltedIrradiance_Wm2"] = out["GlobalTiltedIrradiance_Wm2"].clip(
        lower=0.0
    )
    out["GlobalTiltedIrradiance_kWh_per_m2"] = (
        out["GlobalTiltedIrradiance_kWh_per_m2"]
        .fillna(out["GlobalTiltedIrradiance_Wm2"] / 1000.0)
        .clip(lower=0.0)
    )
    out["GlobalTiltedIrradiance_Wm2"] = out["GlobalTiltedIrradiance_Wm2"].fillna(
        out["GlobalTiltedIrradiance_kWh_per_m2"] * 1000.0
    )

    out["Temperature2m_C"] = (
        out["Temperature2m_C"].fillna(out["Temperature2m_C"].median()).fillna(25.0)
    )
    out["Temperature_C"] = out["Temperature_C"].fillna(out["Temperature2m_C"])
    out["RelativeHumidity2mPct"] = (
        out["RelativeHumidity2mPct"].clip(lower=0, upper=100).fillna(50.0)
    )
    out["DewPoint2m_C"] = out["DewPoint2m_C"].fillna(out["Temperature2m_C"])
    out["ApparentTemperature_C"] = out["ApparentTemperature_C"].fillna(
        out["Temperature2m_C"]
    )
    out["SurfacePressure_hPa"] = (
        out["SurfacePressure_hPa"]
        .fillna(out["SurfacePressure_hPa"].median())
        .fillna(1013.25)
    )
    out["WindSpeed10m_mps"] = out["WindSpeed10m_mps"].clip(lower=0).fillna(0.0)
    out["WindSpeed_ms"] = (
        out["WindSpeed_ms"].clip(lower=0).fillna(out["WindSpeed10m_mps"])
    )
    out["WindDirection10mDeg"] = out["WindDirection10mDeg"].mod(360.0).fillna(0.0)
    out["WindGusts10m_mps"] = (
        out["WindGusts10m_mps"].clip(lower=0.0).fillna(out["WindSpeed10m_mps"])
    )
    out["WindGusts10m_mps"] = out["WindGusts10m_mps"].where(
        out["WindGusts10m_mps"] >= out["WindSpeed10m_mps"],
        out["WindSpeed10m_mps"],
    )
    out["VapourPressureDeficit_kPa"] = (
        out["VapourPressureDeficit_kPa"].clip(lower=0.0).fillna(0.0)
    )
    out["Precipitation_mm"] = out["Precipitation_mm"].clip(lower=0.0).fillna(0.0)
    out["WeatherCode"] = out["WeatherCode"].fillna(0.0)
    out["SunshineDurationSec"] = (
        out["SunshineDurationSec"].clip(lower=0.0, upper=3600.0).fillna(0.0)
    )

    for column in CLOUD_COVER_OUTPUT_COLUMNS:
        out[column] = out[column].clip(lower=0, upper=100).fillna(0.0)

    out["hour"] = out["IntervalStartDT"].dt.hour
    day_of_year = out["IntervalStartDT"].dt.dayofyear
    out["HourSin"] = np.sin(2.0 * np.pi * out["hour"] / 24.0)
    out["HourCos"] = np.cos(2.0 * np.pi * out["hour"] / 24.0)
    out["DayOfYearSin"] = np.sin(2.0 * np.pi * day_of_year / 366.0)
    out["DayOfYearCos"] = np.cos(2.0 * np.pi * day_of_year / 366.0)
    out = add_physics_features(
        out,
        latitude=latitude,
        longitude=longitude,
        timezone_name=timezone_name,
        array_tilt_degrees=array_tilt_degrees,
        array_azimuth_degrees=array_azimuth_degrees,
    )
    out["IsDay"] = (
        out["IsDay"]
        .fillna((out["SolarElevationDeg"] > 0).astype(float))
        .clip(lower=0.0, upper=1.0)
    )
    missing_gti = out["GlobalTiltedIrradiance_Wm2"].isna()
    out.loc[missing_gti, "GlobalTiltedIrradiance_Wm2"] = out.loc[
        missing_gti, "PlaneOfArrayIrradiance_Wm2"
    ]
    out.loc[missing_gti, "GlobalTiltedIrradiance_kWh_per_m2"] = out.loc[
        missing_gti,
        "PlaneOfArray_kWh_per_m2",
    ]
    return out


def _boolean_column(df: pd.DataFrame, column: str, default: bool = False) -> pd.Series:
    """
    Return a nullable-safe boolean view of a dataframe column.
    """
    if column not in df.columns:
        return pd.Series(default, index=df.index, dtype="bool")
    series = df[column]
    if pd.api.types.is_bool_dtype(series):
        return series.fillna(default).astype(bool)
    if pd.api.types.is_numeric_dtype(series):
        return pd.to_numeric(series, errors="coerce").fillna(int(default)).astype(bool)
    return (
        series.astype(str).str.strip().str.lower().isin({"1", "true", "t", "yes", "y"})
    )


def _actual_quality_exclusion_mask(df: pd.DataFrame) -> pd.Series:
    """
    Identify rows excluded from model scoring/training by actual-data quality flags.
    """
    return _boolean_column(df, "SolarBacktestExcluded", default=False)


def add_solar_actual_quality_flags(
    df: pd.DataFrame,
    actual_kwh_col: str,
    expected_kwh_col: str,
    actual_to_expected_ratio_threshold: float,
    actual_read_count_col: Optional[str] = None,
    min_expected_kwh: float = DEFAULT_ACTUAL_QUALITY_MIN_EXPECTED_KWH,
    min_ghi_kwh_m2: float = DEFAULT_ACTUAL_QUALITY_MIN_GHI_KWH_M2,
    min_clear_sky_index: float = DEFAULT_ACTUAL_QUALITY_MIN_CLEAR_SKY_INDEX,
    min_bad_hours_per_day: int = DEFAULT_ACTUAL_QUALITY_MIN_BAD_HOURS_PER_DAY,
    read_coverage_ratio_threshold: float = DEFAULT_ACTUAL_QUALITY_READ_COVERAGE_RATIO_THRESHOLD,
) -> pd.DataFrame:
    """
    Flag likely AMI-suppressed actuals on high-expected, high-irradiance solar hours.
    """
    out = df.copy()
    if out.empty:
        out["ActualQualityFlag"] = pd.Series(dtype="object")
        out["SolarBacktestExcluded"] = pd.Series(dtype="bool")
        out["ActualToExpectedRatio"] = pd.Series(dtype="float64")
        out["ActualQualityExpected_kWh"] = pd.Series(dtype="float64")
        out["ActualQualitySuspiciousHour"] = pd.Series(dtype="bool")
        out["ActualReadCoverageRatio"] = pd.Series(dtype="float64")
        return out

    out["IntervalStartDT"] = pd.to_datetime(out["IntervalStartDT"])

    def numeric_series(column: str, default: float = np.nan) -> pd.Series:
        if column not in out.columns:
            return pd.Series(default, index=out.index, dtype="float64")
        return pd.to_numeric(out[column], errors="coerce")

    actual_kwh = numeric_series(actual_kwh_col, default=0.0).fillna(0.0)
    expected_kwh = numeric_series(expected_kwh_col)
    ghi = numeric_series("GHI_kWh_per_m2")
    clear_sky_index = numeric_series("ClearSkyIndex")
    read_count = (
        numeric_series(actual_read_count_col)
        if actual_read_count_col and actual_read_count_col in out.columns
        else pd.Series(np.nan, index=out.index, dtype="float64")
    )

    ratio = (actual_kwh / expected_kwh.where(expected_kwh > 0)).replace(
        [np.inf, -np.inf], np.nan
    )
    physical_suspicious_hour = (
        (expected_kwh >= min_expected_kwh)
        & (ghi >= min_ghi_kwh_m2)
        & (clear_sky_index >= min_clear_sky_index)
        & ratio.notna()
        & (ratio <= actual_to_expected_ratio_threshold)
    )

    positive_read_count = read_count[read_count > 0].dropna()
    typical_read_count = (
        float(positive_read_count.quantile(0.90))
        if not positive_read_count.empty
        else np.nan
    )
    if pd.notna(typical_read_count) and typical_read_count > 0:
        read_coverage_ratio = read_count / typical_read_count
    else:
        read_coverage_ratio = pd.Series(np.nan, index=out.index, dtype="float64")
    read_suspicious_hour = (
        expected_kwh.ge(min_expected_kwh)
        & ghi.ge(min_ghi_kwh_m2)
        & read_coverage_ratio.notna()
        & read_coverage_ratio.le(read_coverage_ratio_threshold)
    )
    if actual_read_count_col and actual_read_count_col in out.columns:
        severe_physical = physical_suspicious_hour & ratio.le(
            actual_to_expected_ratio_threshold * 0.50
        )
        suspicious_hour = read_suspicious_hour | severe_physical
    else:
        suspicious_hour = physical_suspicious_hour

    dates = out["IntervalStartDT"].dt.date
    min_bad_hours = max(1, int(min_bad_hours_per_day))
    bad_hour_counts = suspicious_hour.groupby(dates).sum()
    if actual_read_count_col and actual_read_count_col in out.columns:
        daily_read_coverage = read_coverage_ratio.groupby(dates).median()
        bad_dates = set(
            bad_hour_counts[
                (bad_hour_counts >= min_bad_hours)
                & (daily_read_coverage <= read_coverage_ratio_threshold)
            ].index
        )
    else:
        bad_dates = set(bad_hour_counts[bad_hour_counts >= min_bad_hours].index)
    quality_excluded = suspicious_hour | dates.isin(bad_dates)

    out["SolarBacktestExcluded"] = _actual_quality_exclusion_mask(
        out
    ) | quality_excluded.fillna(False).astype(bool)
    if "ActualQualityFlag" not in out.columns:
        out["ActualQualityFlag"] = ACTUAL_QUALITY_OK
    out["ActualQualityFlag"] = (
        out["ActualQualityFlag"].fillna(ACTUAL_QUALITY_OK).astype(str)
    )
    out.loc[quality_excluded, "ActualQualityFlag"] = ACTUAL_QUALITY_AMI_SUPPRESSED
    out["ActualToExpectedRatio"] = ratio
    out["ActualQualityExpected_kWh"] = expected_kwh
    out["ActualQualitySuspiciousHour"] = suspicious_hour.fillna(False).astype(bool)
    out["ActualReadCoverageRatio"] = read_coverage_ratio
    return out


def train_performance_model(
    rec_intervals: pd.DataFrame,
    weather_df: pd.DataFrame,
    capacity_kw: float,
    fallback_ratio: float,
    performance_ratio_upper_bound: float = DEFAULT_PERFORMANCE_RATIO_UPPER_BOUND,
    sites: Optional[pd.DataFrame] = None,
    latitude: float = ROSEVILLE_LATITUDE,
    longitude: float = ROSEVILLE_LONGITUDE,
    timezone_name: str = "America/Los_Angeles",
    array_tilt_degrees: float = DEFAULT_ARRAY_TILT_DEGREES,
    array_azimuth_degrees: float = DEFAULT_ARRAY_AZIMUTH_DEGREES,
    min_training_available_kwh: float = 25.0,
    min_training_rows: int = 24,
    daily_active_capacity: Optional[pd.DataFrame] = None,
    max_performance_ratio: Optional[float] = None,
    use_energy_weighting: bool = True,
    exclude_suppressed_actuals: bool = True,
) -> PerformanceModel:
    """
    Train a bounded model that predicts export performance ratio from weather and seasonality.
    """
    if max_performance_ratio is not None:
        performance_ratio_upper_bound = max_performance_ratio
    export_agg_spec = {"Export_kWh": ("Export_kWh", "sum")}
    if "ExportReadCount" in rec_intervals.columns:
        export_agg_spec["ExportReadCount"] = ("ExportReadCount", "sum")
    hourly_export = (
        rec_intervals.copy()
        .set_index("IntervalStartDT")
        .resample("h")
        .agg(**export_agg_spec)
        .reset_index()
    )
    hourly_weather = add_performance_features(
        aggregate_weather_to_hourly(weather_df),
        latitude=latitude,
        longitude=longitude,
        timezone_name=timezone_name,
        array_tilt_degrees=array_tilt_degrees,
        array_azimuth_degrees=array_azimuth_degrees,
    )

    training_data = pd.merge(
        hourly_export, hourly_weather, on="IntervalStartDT", how="inner"
    )
    if daily_active_capacity is not None and not daily_active_capacity.empty:
        training_data["ActiveCapacity_kW"] = _resolve_row_capacity(
            training_data["IntervalStartDT"],
            daily_active_capacity,
            capacity_kw,
        )
    else:
        training_data["ActiveCapacity_kW"] = calculate_active_capacity_for_timestamps(
            training_data["IntervalStartDT"],
            sites=sites,
            default_capacity_kw=capacity_kw,
        )
    available_per_kw = training_data["PVWatts_kWh_per_kW"].where(
        training_data["PVWatts_kWh_per_kW"] > 0.0,
        training_data["GHI_kWh_per_m2"],
    )
    training_data["ModeledAvailable_kWh"] = (
        available_per_kw * training_data["ActiveCapacity_kW"]
    )
    training_data = training_data[
        (training_data["ActiveCapacity_kW"] > 0)
        & (training_data["ModeledAvailable_kWh"] >= min_training_available_kwh)
        & training_data["Export_kWh"].notna()
    ].copy()
    if exclude_suppressed_actuals and not training_data.empty:
        training_data = add_solar_actual_quality_flags(
            training_data,
            actual_kwh_col="Export_kWh",
            expected_kwh_col="ModeledAvailable_kWh",
            actual_to_expected_ratio_threshold=DEFAULT_ACTUAL_QUALITY_AVAILABLE_RATIO_THRESHOLD,
            actual_read_count_col="ExportReadCount",
        )
        excluded_rows = int(_actual_quality_exclusion_mask(training_data).sum())
        if excluded_rows:
            logging.info(
                "Excluded %s AMI-suppressed daylight rows from performance-ratio training.",
                excluded_rows,
            )
            training_data = training_data[
                ~_actual_quality_exclusion_mask(training_data)
            ].copy()

    if training_data.empty:
        logging.warning(
            "No usable model training rows found; using fallback performance ratio %.3f",
            fallback_ratio,
        )
        return PerformanceModel(
            None,
            fallback_ratio,
            PERFORMANCE_FEATURE_COLUMNS,
            performance_ratio_upper_bound,
        )

    training_data["PerformanceRatio"] = (
        training_data["Export_kWh"] / training_data["ModeledAvailable_kWh"]
    ).clip(lower=0.0, upper=performance_ratio_upper_bound)
    learned_fallback = float(training_data["PerformanceRatio"].median())
    if pd.isna(learned_fallback) or learned_fallback <= 0:
        learned_fallback = fallback_ratio
    learned_fallback = float(
        np.clip(learned_fallback, 0.0, performance_ratio_upper_bound)
    )

    if len(training_data) < min_training_rows:
        logging.warning(
            "Only %s usable model training rows found; using median performance ratio %.3f",
            len(training_data),
            learned_fallback,
        )
        return PerformanceModel(
            None,
            learned_fallback,
            PERFORMANCE_FEATURE_COLUMNS,
            performance_ratio_upper_bound,
        )

    model = GradientBoostingRegressor(
        n_estimators=160,
        learning_rate=0.04,
        max_depth=2,
        min_samples_leaf=8,
        subsample=0.85,
        random_state=42,
    )
    X = training_data[PERFORMANCE_FEATURE_COLUMNS].fillna(0.0)
    y = training_data["PerformanceRatio"]
    sample_weight = None
    if use_energy_weighting:
        sample_weight = training_data["Export_kWh"].clip(
            lower=min_training_available_kwh
        )
    model.fit(X, y, sample_weight=sample_weight)

    fitted = pd.Series(model.predict(X), index=training_data.index).clip(
        0.0, performance_ratio_upper_bound
    )
    fitted_kwh = training_data["ModeledAvailable_kWh"] * fitted
    actual_kwh_sum = training_data["Export_kWh"].sum()
    wmape = (
        (fitted_kwh - training_data["Export_kWh"]).abs().sum() / actual_kwh_sum
        if actual_kwh_sum > 0
        else np.nan
    )
    logging.info(
        "Trained physics-residual performance model on %s hourly daylight rows; median ratio %.3f, "
        "energy weighting %s, in-sample daylight WMAPE %.2f%% "
        "(modeled capacity range %.2f to %.2f kW, ratio upper %.2f)",
        len(training_data),
        learned_fallback,
        "enabled" if use_energy_weighting else "disabled",
        wmape * 100 if pd.notna(wmape) else float("nan"),
        training_data["ActiveCapacity_kW"].min(),
        training_data["ActiveCapacity_kW"].max(),
        performance_ratio_upper_bound,
    )
    return PerformanceModel(
        model,
        learned_fallback,
        PERFORMANCE_FEATURE_COLUMNS,
        performance_ratio_upper_bound,
    )


def predict_performance_ratio(
    model: PerformanceModel, feature_df: pd.DataFrame
) -> pd.Series:
    """
    Predict bounded performance ratio for weather feature rows.
    """
    if model.estimator is None:
        return pd.Series(model.fallback_ratio, index=feature_df.index)

    ratio = pd.Series(
        model.estimator.predict(
            feature_df.reindex(columns=model.feature_columns).fillna(0.0)
        ),
        index=feature_df.index,
    )
    return ratio.clip(lower=0.0, upper=model.ratio_upper_bound).fillna(
        model.fallback_ratio
    )


def identity_residual_calibration_model(
    lower_bound: float = 0.25,
    upper_bound: float = 1.75,
) -> ResidualCalibrationModel:
    """
    Build a no-op residual calibration model.
    """
    return ResidualCalibrationModel(
        estimator=None,
        fallback_factor=1.0,
        feature_columns=CALIBRATION_FEATURE_COLUMNS,
        lower_bound=lower_bound,
        upper_bound=upper_bound,
    )


def add_residual_calibration_features(
    df: pd.DataFrame,
    total_capacity_kw: float,
    latitude: float,
    longitude: float,
    timezone_name: str,
    array_tilt_degrees: float = DEFAULT_ARRAY_TILT_DEGREES,
    array_azimuth_degrees: float = DEFAULT_ARRAY_AZIMUTH_DEGREES,
    performance_ratio_upper_bound: float = DEFAULT_PERFORMANCE_RATIO_UPPER_BOUND,
) -> pd.DataFrame:
    """
    Add weather, time, and base-forecast features used by the residual calibrator.
    """
    out = add_performance_features(
        df,
        latitude=latitude,
        longitude=longitude,
        timezone_name=timezone_name,
        array_tilt_degrees=array_tilt_degrees,
        array_azimuth_degrees=array_azimuth_degrees,
    )

    for column in [
        "Forecast_kWh",
        "Forecast_kW",
        "Forecast_MW",
        "CapacityFactor",
        "PerformanceRatio",
    ]:
        if column not in out.columns:
            out[column] = np.nan
        out[column] = pd.to_numeric(out[column], errors="coerce")

    if not out["Forecast_MW"].isna().all():
        out["Forecast_kW"] = out["Forecast_kW"].fillna(out["Forecast_MW"] * 1000.0)
    if out["Forecast_kW"].isna().any() and not out["Forecast_kWh"].isna().all():
        out["Forecast_kW"] = out["Forecast_kW"].fillna(
            out["Forecast_kWh"] / INTERVAL_HOURS
        )
    if out["Forecast_MW"].isna().any() and not out["Forecast_kW"].isna().all():
        out["Forecast_MW"] = out["Forecast_MW"].fillna(out["Forecast_kW"] / 1000.0)
    if out["Forecast_kWh"].isna().any() and not out["Forecast_kW"].isna().all():
        out["Forecast_kWh"] = out["Forecast_kWh"].fillna(
            out["Forecast_kW"] * INTERVAL_HOURS
        )

    if total_capacity_kw > 0:
        out["CapacityFactor"] = out["CapacityFactor"].fillna(
            out["Forecast_kW"] / total_capacity_kw
        )
    out["CapacityFactor"] = out["CapacityFactor"].clip(lower=0.0)
    out["PerformanceRatio"] = (
        out["PerformanceRatio"]
        .fillna(0.0)
        .clip(
            lower=0.0,
            upper=performance_ratio_upper_bound,
        )
    )
    return out


def filter_residual_calibration_training_quality(
    training_data: pd.DataFrame,
    min_training_forecast_kwh: float,
    min_row_coverage: float,
    min_actual_forecast_ratio: float,
    min_day_forecast_mwh: float,
) -> pd.DataFrame:
    """
    Remove daylight days that look like incomplete actual-feed days.
    """
    if training_data.empty:
        return training_data

    work = training_data.copy()
    work["IntervalStartDT"] = pd.to_datetime(work["IntervalStartDT"], errors="coerce")
    work["Forecast_kWh"] = pd.to_numeric(work["Forecast_kWh"], errors="coerce")
    work["Actual_kWh"] = pd.to_numeric(work["Actual_kWh"], errors="coerce")
    daylight = work[
        work["Forecast_kWh"].fillna(0.0) >= min_training_forecast_kwh
    ].copy()
    if daylight.empty:
        return training_data

    daylight["CalibrationDate"] = daylight["IntervalStartDT"].dt.date
    daily = daylight.groupby("CalibrationDate", as_index=False).agg(
        ExpectedDaylightHours=("Forecast_kWh", "count"),
        ObservedActualHours=("Actual_kWh", lambda values: values.notna().sum()),
        Actual_kWh=("Actual_kWh", "sum"),
        Forecast_kWh=("Forecast_kWh", "sum"),
    )
    daily["RowCoveragePct"] = daily["ObservedActualHours"] / daily[
        "ExpectedDaylightHours"
    ].where(daily["ExpectedDaylightHours"] > 0)
    daily["ActualForecastRatio"] = daily["Actual_kWh"] / daily["Forecast_kWh"].where(
        daily["Forecast_kWh"] > 0
    )
    material_day_mask = daily["Forecast_kWh"] >= min_day_forecast_mwh * 1000.0
    usable_day_mask = daily["RowCoveragePct"].fillna(0.0).ge(min_row_coverage) & (
        ~material_day_mask
        | daily["ActualForecastRatio"].fillna(0.0).ge(min_actual_forecast_ratio)
    )
    usable_dates = set(daily.loc[usable_day_mask, "CalibrationDate"])
    filtered = work[work["IntervalStartDT"].dt.date.isin(usable_dates)].copy()
    excluded_days = int((~usable_day_mask).sum())
    excluded_rows = len(work) - len(filtered)

    if filtered.empty:
        logging.warning(
            "Residual calibration quality filter would exclude all %s training rows across %s days; "
            "keeping unfiltered training data.",
            len(work),
            len(daily),
        )
        return training_data

    if excluded_days > 0:
        logging.info(
            "Residual calibration quality filter excluded %s likely incomplete actual-feed days "
            "and %s daylight rows; keeping %s rows.",
            excluded_days,
            excluded_rows,
            len(filtered),
        )
    return filtered


def build_regime_key_frame(df: pd.DataFrame) -> pd.DataFrame:
    """
    Build stable weather-regime lookup keys from clear-sky, cloud, and solar elevation.
    """
    index = df.index
    clear_sky_index = pd.to_numeric(
        df.get("ClearSkyIndex", pd.Series(0.0, index=index)), errors="coerce"
    )
    cloud_cover = pd.to_numeric(
        df.get("CloudCoverPct", pd.Series(0.0, index=index)), errors="coerce"
    )
    solar_elevation = pd.to_numeric(
        df.get("SolarElevationDeg", pd.Series(90.0, index=index)), errors="coerce"
    )

    csi_bucket = (
        pd.cut(
            clear_sky_index.clip(lower=0.0, upper=1.6).fillna(0.0),
            bins=REGIME_CLEAR_SKY_BINS,
            labels=REGIME_CLEAR_SKY_LABELS,
            include_lowest=True,
        )
        .astype("object")
        .fillna(REGIME_CLEAR_SKY_LABELS[0])
        .astype(str)
    )
    cloud_bucket = (
        pd.cut(
            cloud_cover.clip(lower=0.0, upper=100.0).fillna(0.0),
            bins=REGIME_CLOUD_BINS,
            labels=REGIME_CLOUD_LABELS,
            include_lowest=True,
        )
        .astype("object")
        .fillna(REGIME_CLOUD_LABELS[0])
        .astype(str)
    )
    elevation_bucket = (
        pd.cut(
            solar_elevation.clip(lower=-90.0, upper=90.0).fillna(90.0),
            bins=REGIME_ELEVATION_BINS,
            labels=REGIME_ELEVATION_LABELS,
            include_lowest=True,
        )
        .astype("object")
        .fillna(REGIME_ELEVATION_LABELS[-1])
        .astype(str)
    )

    return pd.DataFrame(
        {
            "RegimeKey": "regime|"
            + csi_bucket
            + "|"
            + cloud_bucket
            + "|"
            + elevation_bucket,
            "RegimeFallbackKey": "csi|" + csi_bucket,
        },
        index=index,
    )


def build_regime_factor_map(
    calibration_data: pd.DataFrame,
    preliminary_forecast_kwh: pd.Series,
    min_rows: int,
    min_forecast_mwh: float,
    prior_mwh: float,
    lower_bound: float,
    upper_bound: float,
) -> dict[str, float]:
    """
    Build bounded correction factors for weather regimes after residual/seasonal calibration.
    """
    if calibration_data.empty:
        return {}

    work = calibration_data.copy()
    work["Actual_kWh"] = pd.to_numeric(work["Actual_kWh"], errors="coerce")
    work["PreliminaryForecast_kWh"] = pd.to_numeric(
        preliminary_forecast_kwh.reindex(work.index),
        errors="coerce",
    )
    work = work.dropna(subset=["Actual_kWh", "PreliminaryForecast_kWh"])
    work = work[
        (work["Actual_kWh"] >= 0) & (work["PreliminaryForecast_kWh"] > 0)
    ].copy()
    if work.empty:
        return {}

    regime_keys = build_regime_key_frame(work)
    work["RegimeKey"] = regime_keys["RegimeKey"]
    work["RegimeFallbackKey"] = regime_keys["RegimeFallbackKey"]
    prior_kwh = prior_mwh * 1000.0
    min_forecast_kwh = min_forecast_mwh * 1000.0
    factors: dict[str, float] = {}

    for group_column in ["RegimeKey", "RegimeFallbackKey"]:
        grouped = work.groupby(group_column, as_index=False).agg(
            Rows=("Actual_kWh", "count"),
            Actual_kWh=("Actual_kWh", "sum"),
            Forecast_kWh=("PreliminaryForecast_kWh", "sum"),
        )
        grouped = grouped[
            (grouped["Rows"] >= min_rows)
            & (grouped["Forecast_kWh"] >= min_forecast_kwh)
            & (grouped["Forecast_kWh"] > 0)
        ].copy()
        for _, row in grouped.iterrows():
            raw_factor = (row["Actual_kWh"] + prior_kwh) / (
                row["Forecast_kWh"] + prior_kwh
            )
            factors[str(row[group_column])] = float(
                np.clip(raw_factor, lower_bound, upper_bound)
            )

    return factors


def lookup_regime_calibration_factor_from_map(
    feature_df: pd.DataFrame,
    regime_factors: dict[str, float],
    default_factor: float = 1.0,
) -> pd.Series:
    """
    Lookup exact weather-regime factors, then clear-sky fallback factors.
    """
    if feature_df.empty:
        return pd.Series(dtype="float64")
    if not regime_factors:
        return pd.Series(default_factor, index=feature_df.index, dtype="float64")

    regime_keys = build_regime_key_frame(feature_df)
    exact_factor = regime_keys["RegimeKey"].map(regime_factors)
    fallback_factor = regime_keys["RegimeFallbackKey"].map(regime_factors)
    return exact_factor.fillna(fallback_factor).fillna(default_factor).astype("float64")


def prune_regime_factors_by_holdout(
    validation_data: pd.DataFrame,
    pre_regime_forecast_kwh: pd.Series,
    candidate_factors: dict[str, float],
) -> dict[str, float]:
    """
    Keep only regime factors that improve their own validation slice.
    """
    if validation_data.empty or not candidate_factors:
        return {}

    work = validation_data.copy()
    work["Actual_kWh"] = pd.to_numeric(work["Actual_kWh"], errors="coerce")
    work["PreRegimeForecast_kWh"] = pd.to_numeric(
        pre_regime_forecast_kwh.reindex(work.index),
        errors="coerce",
    )
    work = work.dropna(subset=["Actual_kWh", "PreRegimeForecast_kWh"])
    work = work[(work["Actual_kWh"] >= 0) & (work["PreRegimeForecast_kWh"] > 0)].copy()
    if work.empty:
        return {}

    regime_keys = build_regime_key_frame(work)
    work["RegimeKey"] = regime_keys["RegimeKey"]
    work["RegimeFallbackKey"] = regime_keys["RegimeFallbackKey"]
    kept: dict[str, float] = {}

    for key_column in ["RegimeKey", "RegimeFallbackKey"]:
        for key, factor in candidate_factors.items():
            if key not in set(work[key_column]):
                continue
            subset = work[work[key_column].eq(key)].copy()
            forecast_mwh = subset["PreRegimeForecast_kWh"].sum() / 1000.0
            if (
                len(subset) < REGIME_HOLDOUT_MIN_ROWS
                or forecast_mwh < REGIME_HOLDOUT_MIN_FORECAST_MWH
            ):
                continue
            base_wmape = calculate_energy_wmape(
                subset["Actual_kWh"],
                subset["PreRegimeForecast_kWh"],
            )
            corrected_wmape = calculate_energy_wmape(
                subset["Actual_kWh"],
                subset["PreRegimeForecast_kWh"] * factor,
            )
            if (
                pd.notna(base_wmape)
                and pd.notna(corrected_wmape)
                and corrected_wmape <= base_wmape - REGIME_HOLDOUT_WMAPE_TOLERANCE
            ):
                kept[key] = factor

    return kept


def calculate_energy_wmape(actual_kwh: pd.Series, forecast_kwh: pd.Series) -> float:
    """
    Calculate energy-weighted MAPE for aligned actual and forecast energy series.
    """
    actual = pd.to_numeric(actual_kwh, errors="coerce")
    forecast = pd.to_numeric(forecast_kwh, errors="coerce")
    mask = actual.notna() & forecast.notna()
    actual_sum = actual.loc[mask].sum()
    if actual_sum <= 0:
        return np.nan
    return float((forecast.loc[mask] - actual.loc[mask]).abs().sum() / actual_sum)


def calculate_peak_capture(
    actual_power_kw: pd.Series, forecast_power_kw: pd.Series
) -> float:
    """
    Calculate forecast peak divided by actual peak for aligned power series.
    """
    actual = pd.to_numeric(actual_power_kw, errors="coerce")
    forecast = pd.to_numeric(forecast_power_kw, errors="coerce")
    mask = actual.notna() & forecast.notna()
    if not mask.any():
        return np.nan
    actual_peak_kw = actual.loc[mask].max()
    forecast_peak_kw = forecast.loc[mask].max()
    if pd.isna(actual_peak_kw) or actual_peak_kw <= 0:
        return np.nan
    return float(forecast_peak_kw / actual_peak_kw)


def active_capacity_for_peak_calibration(
    df: pd.DataFrame, total_capacity_kw: float
) -> pd.Series:
    """
    Return positive active capacity for forecast capacity-factor peak calibration.
    """
    if "ActiveCapacity_kW" in df.columns:
        capacity_kw = pd.to_numeric(df["ActiveCapacity_kW"], errors="coerce")
    else:
        capacity_kw = pd.Series(total_capacity_kw, index=df.index, dtype="float64")
    fallback_capacity_kw = total_capacity_kw if total_capacity_kw > 0 else np.nan
    capacity_kw = capacity_kw.where(capacity_kw > 0, fallback_capacity_kw)
    return capacity_kw


def actual_power_kw_for_peak_calibration(df: pd.DataFrame) -> pd.Series:
    """
    Use actual kW when present; otherwise hourly kWh is equivalent to average kW.
    """
    if "Actual_kW" in df.columns:
        actual_kw = pd.to_numeric(df["Actual_kW"], errors="coerce")
        if actual_kw.notna().any():
            return actual_kw
    return pd.to_numeric(
        df.get("Actual_kWh", pd.Series(np.nan, index=df.index)), errors="coerce"
    )


def build_peak_calibration_params(
    calibration_data: pd.DataFrame,
    pre_peak_forecast_kwh: pd.Series,
    total_capacity_kw: float,
    quantile: float,
    min_rows: int,
    min_forecast_mwh: float,
    lower_bound: float,
    target_capture: float,
) -> tuple[float, Optional[float], dict[str, float]]:
    """
    Learn an upper-tail peak factor from calibrated forecast capacity factor.
    """
    if calibration_data.empty:
        return 1.0, None, {}

    work = calibration_data.copy()
    work["ActualPower_kW"] = actual_power_kw_for_peak_calibration(work)
    work["PrePeakForecast_kWh"] = pd.to_numeric(
        pre_peak_forecast_kwh.reindex(work.index),
        errors="coerce",
    )
    work["PrePeakForecastPower_kW"] = work["PrePeakForecast_kWh"]
    work["ActiveCapacity_kW"] = active_capacity_for_peak_calibration(
        work, total_capacity_kw
    )
    work = work.dropna(
        subset=[
            "ActualPower_kW",
            "PrePeakForecast_kWh",
            "PrePeakForecastPower_kW",
            "ActiveCapacity_kW",
        ]
    )
    work = work[
        (work["ActualPower_kW"] >= 0)
        & (work["PrePeakForecast_kWh"] > 0)
        & (work["PrePeakForecastPower_kW"] > 0)
        & (work["ActiveCapacity_kW"] > 0)
    ].copy()
    if work.empty:
        return 1.0, None, {}

    work["PrePeakCapacityFactor"] = (
        work["PrePeakForecastPower_kW"] / work["ActiveCapacity_kW"]
    )
    threshold_cf = float(work["PrePeakCapacityFactor"].quantile(quantile))
    if pd.isna(threshold_cf) or threshold_cf <= 0:
        return 1.0, None, {}

    top = work[work["PrePeakCapacityFactor"] >= threshold_cf].copy()
    top_forecast_mwh = top["PrePeakForecast_kWh"].sum() / 1000.0
    if len(top) < min_rows or top_forecast_mwh < min_forecast_mwh:
        return (
            1.0,
            None,
            {
                "Rows": float(len(top)),
                "Forecast_MWh": float(top_forecast_mwh),
                "ThresholdCF": float(threshold_cf),
            },
        )

    actual_peak_kw = work["ActualPower_kW"].max()
    forecast_peak_kw = work["PrePeakForecastPower_kW"].max()
    if (
        pd.isna(actual_peak_kw)
        or actual_peak_kw <= 0
        or pd.isna(forecast_peak_kw)
        or forecast_peak_kw <= 0
    ):
        return 1.0, None, {}

    raw_factor = target_capture * actual_peak_kw / forecast_peak_kw
    factor = float(np.clip(raw_factor, lower_bound, 1.0))
    peak_capture = calculate_peak_capture(
        work["ActualPower_kW"], work["PrePeakForecastPower_kW"]
    )
    if factor >= 1.0 - PEAK_CAPTURE_TOLERANCE:
        return (
            1.0,
            None,
            {
                "Rows": float(len(top)),
                "Forecast_MWh": float(top_forecast_mwh),
                "ThresholdCF": float(threshold_cf),
                "PeakCapture": (
                    float(peak_capture) if pd.notna(peak_capture) else np.nan
                ),
            },
        )

    return (
        factor,
        threshold_cf,
        {
            "Rows": float(len(top)),
            "Forecast_MWh": float(top_forecast_mwh),
            "ThresholdCF": float(threshold_cf),
            "PeakCapture": float(peak_capture) if pd.notna(peak_capture) else np.nan,
        },
    )


def lookup_peak_calibration_factor(
    feature_df: pd.DataFrame,
    forecast_power_kw: pd.Series,
    total_capacity_kw: float,
    factor: float,
    threshold_cf: Optional[float],
) -> pd.Series:
    """
    Apply a learned peak factor only to upper-tail forecast capacity factors.
    """
    if feature_df.empty:
        return pd.Series(dtype="float64")
    if threshold_cf is None or factor >= 1.0:
        return pd.Series(1.0, index=feature_df.index, dtype="float64")

    capacity_kw = active_capacity_for_peak_calibration(feature_df, total_capacity_kw)
    forecast_power_kw = pd.to_numeric(
        forecast_power_kw.reindex(feature_df.index), errors="coerce"
    )
    forecast_cf = forecast_power_kw / capacity_kw.where(capacity_kw > 0)
    peak_mask = forecast_cf.ge(threshold_cf).fillna(False)
    out = pd.Series(1.0, index=feature_df.index, dtype="float64")
    out.loc[peak_mask] = factor
    return out


def tune_peak_calibration_from_hourly_backtest(
    calibration_model: ResidualCalibrationModel,
    hourly_backtest: pd.DataFrame,
    total_capacity_kw: float,
    quantile: float,
    min_rows: int,
    min_forecast_mwh: float,
    lower_bound: float,
    target_capture: float,
    label: str,
) -> bool:
    """
    Tighten peak calibration against the hourly backtest produced by the full 15-minute pipeline.
    """
    if hourly_backtest.empty:
        return False

    work = hourly_backtest.copy()
    work["ActualPower_kW"] = (
        pd.to_numeric(
            work.get("Actual_MW", pd.Series(np.nan, index=work.index)), errors="coerce"
        )
        * 1000.0
    )
    if work["ActualPower_kW"].isna().all():
        work["ActualPower_kW"] = actual_power_kw_for_peak_calibration(work)
    work["ForecastPower_kW"] = (
        pd.to_numeric(
            work.get("Forecast_MW", pd.Series(np.nan, index=work.index)),
            errors="coerce",
        )
        * 1000.0
    )
    if work["ForecastPower_kW"].isna().all():
        work["ForecastPower_kW"] = pd.to_numeric(
            work.get("Forecast_kWh"), errors="coerce"
        )
    work["Forecast_kWh"] = pd.to_numeric(
        work.get("Forecast_kWh", pd.Series(np.nan, index=work.index)),
        errors="coerce",
    )
    work["ActiveCapacity_kW"] = active_capacity_for_peak_calibration(
        work, total_capacity_kw
    )
    work = work.dropna(
        subset=[
            "ActualPower_kW",
            "ForecastPower_kW",
            "Forecast_kWh",
            "ActiveCapacity_kW",
        ]
    )
    work = work[
        (work["ActualPower_kW"] >= 0)
        & (work["ForecastPower_kW"] > 0)
        & (work["Forecast_kWh"] > 0)
        & (work["ActiveCapacity_kW"] > 0)
    ].copy()
    if work.empty:
        return False

    peak_capture = calculate_peak_capture(
        work["ActualPower_kW"], work["ForecastPower_kW"]
    )
    if pd.isna(peak_capture) or peak_capture <= target_capture + PEAK_CAPTURE_TOLERANCE:
        return False

    work["ForecastCapacityFactor"] = (
        work["ForecastPower_kW"] / work["ActiveCapacity_kW"]
    )
    threshold_cf = calibration_model.peak_calibration_threshold_cf
    if threshold_cf is None or pd.isna(threshold_cf) or threshold_cf <= 0:
        threshold_cf = float(work["ForecastCapacityFactor"].quantile(quantile))
    if pd.isna(threshold_cf) or threshold_cf <= 0:
        return False

    top = work[work["ForecastCapacityFactor"] >= threshold_cf].copy()
    top_forecast_mwh = top["Forecast_kWh"].sum() / 1000.0
    if len(top) < min_rows or top_forecast_mwh < min_forecast_mwh:
        logging.info(
            "Peak calibration pipeline tightening skipped for %s; only %s upper-tail rows and %.1f MWh",
            label,
            len(top),
            top_forecast_mwh,
        )
        return False

    current_factor = calibration_model.peak_calibration_factor
    additional_factor = target_capture / peak_capture
    new_factor = float(np.clip(current_factor * additional_factor, lower_bound, 1.0))
    if new_factor >= current_factor - PEAK_CAPTURE_TOLERANCE:
        return False

    calibration_model.peak_calibration_factor = new_factor
    calibration_model.peak_calibration_threshold_cf = threshold_cf
    logging.info(
        "Peak calibration pipeline tightened for %s: capture %.2f%% toward %.2f%%, "
        "factor %.3f -> %.3f above forecast CF %.3f (%s rows, %.1f MWh)",
        label,
        peak_capture * 100,
        target_capture * 100,
        current_factor,
        new_factor,
        threshold_cf,
        len(top),
        top_forecast_mwh,
    )
    return True


def train_residual_calibration_model(
    backtest_df: pd.DataFrame,
    total_capacity_kw: float,
    latitude: float,
    longitude: float,
    timezone_name: str,
    array_tilt_degrees: float,
    array_azimuth_degrees: float,
    performance_ratio_upper_bound: float,
    lower_bound: float,
    upper_bound: float,
    min_training_forecast_kwh: float,
    min_training_rows: int,
    validation_fraction: float,
    min_validation_rows: int,
    use_energy_weighting: bool,
    energy_weight_power: float,
    use_seasonal_calibration: bool,
    seasonal_prior_mwh: float,
    seasonal_lower_bound: float,
    seasonal_upper_bound: float,
    use_quality_filter: bool,
    quality_min_row_coverage: float,
    quality_min_actual_forecast_ratio: float,
    quality_min_day_forecast_mwh: float,
    use_regime_calibration: bool,
    regime_min_rows: int,
    regime_min_forecast_mwh: float,
    regime_prior_mwh: float,
    regime_lower_bound: float,
    regime_upper_bound: float,
    use_peak_calibration: bool,
    peak_quantile: float,
    peak_min_rows: int,
    peak_min_forecast_mwh: float,
    peak_lower_bound: float,
    peak_target_capture: float,
) -> ResidualCalibrationModel:
    """
    Train a bounded model that corrects repeatable actual-vs-forecast residual bias.
    """
    model = identity_residual_calibration_model(
        lower_bound=lower_bound, upper_bound=upper_bound
    )
    if backtest_df.empty:
        logging.info("Residual calibration skipped; no backtest rows are available.")
        return model

    training_data = backtest_df.copy()
    training_data["IntervalStartDT"] = pd.to_datetime(
        training_data["IntervalStartDT"], errors="coerce"
    )
    training_data["Actual_kWh"] = pd.to_numeric(
        training_data["Actual_kWh"], errors="coerce"
    )
    training_data["Forecast_kWh"] = pd.to_numeric(
        training_data["Forecast_kWh"], errors="coerce"
    )
    training_data = training_data.dropna(subset=["Actual_kWh", "Forecast_kWh"])
    training_data = training_data[
        (training_data["Actual_kWh"] >= 0)
        & (training_data["Forecast_kWh"] >= min_training_forecast_kwh)
    ].copy()
    if training_data.empty:
        logging.info(
            "Residual calibration skipped; no daylight forecast rows met the training threshold."
        )
        return model
    if use_quality_filter:
        training_data = filter_residual_calibration_training_quality(
            training_data,
            min_training_forecast_kwh=min_training_forecast_kwh,
            min_row_coverage=quality_min_row_coverage,
            min_actual_forecast_ratio=quality_min_actual_forecast_ratio,
            min_day_forecast_mwh=quality_min_day_forecast_mwh,
        )
        training_data = training_data.dropna(subset=["Actual_kWh", "Forecast_kWh"])
        training_data = training_data[
            (training_data["Actual_kWh"] >= 0)
            & (training_data["Forecast_kWh"] >= min_training_forecast_kwh)
        ].copy()
        if training_data.empty:
            logging.info(
                "Residual calibration skipped; quality filter removed all daylight training rows."
            )
            return model

    aggregate_factor = (
        training_data["Actual_kWh"].sum() / training_data["Forecast_kWh"].sum()
    )
    aggregate_factor = float(np.clip(aggregate_factor, lower_bound, upper_bound))
    training_data["ResidualCalibrationFactor"] = (
        training_data["Actual_kWh"] / training_data["Forecast_kWh"]
    ).clip(lower=lower_bound, upper=upper_bound)

    if len(training_data) < min_training_rows:
        logging.info(
            "Only %s residual calibration rows found; using aggregate correction factor %.3f",
            len(training_data),
            aggregate_factor,
        )
        return ResidualCalibrationModel(
            estimator=None,
            fallback_factor=aggregate_factor,
            feature_columns=CALIBRATION_FEATURE_COLUMNS,
            lower_bound=lower_bound,
            upper_bound=upper_bound,
        )

    training_data = training_data.sort_values("IntervalStartDT").copy()
    feature_data = add_residual_calibration_features(
        training_data,
        total_capacity_kw=total_capacity_kw,
        latitude=latitude,
        longitude=longitude,
        timezone_name=timezone_name,
        array_tilt_degrees=array_tilt_degrees,
        array_azimuth_degrees=array_azimuth_degrees,
        performance_ratio_upper_bound=performance_ratio_upper_bound,
    )
    estimator_params = {
        "n_estimators": 220,
        "learning_rate": 0.03,
        "max_depth": 4,
        "min_samples_leaf": 6,
        "subsample": 0.90,
        "random_state": 84,
    }
    X = feature_data[CALIBRATION_FEATURE_COLUMNS].fillna(0.0)
    y = training_data["ResidualCalibrationFactor"]
    sample_weight = None
    if use_energy_weighting:
        sample_weight = (
            training_data["Forecast_kWh"].clip(lower=min_training_forecast_kwh)
            ** energy_weight_power
        )

    disable_regime_due_holdout = False
    disable_peak_due_holdout = False
    validated_regime_factor_keys: Optional[set[str]] = None
    holdout_rows = int(math.ceil(len(training_data) * validation_fraction))
    holdout_rows = (
        max(min_validation_rows, holdout_rows) if validation_fraction > 0 else 0
    )
    holdout_rows = min(holdout_rows, len(training_data) - min_training_rows)
    if holdout_rows >= min_validation_rows:
        fit_index = training_data.index[:-holdout_rows]
        validation_index = training_data.index[-holdout_rows:]
        validation_estimator = GradientBoostingRegressor(**estimator_params)
        validation_weights = (
            sample_weight.loc[fit_index] if sample_weight is not None else None
        )
        validation_estimator.fit(
            X.loc[fit_index], y.loc[fit_index], sample_weight=validation_weights
        )
        validation_subset = training_data.loc[validation_index].copy()
        fit_residual_factor = pd.Series(
            validation_estimator.predict(X.loc[fit_index]),
            index=fit_index,
        ).clip(lower_bound, upper_bound)
        validation_factor = pd.Series(
            validation_estimator.predict(X.loc[validation_index]),
            index=validation_index,
        ).clip(lower_bound, upper_bound)
        fit_factor = fit_residual_factor.copy()
        if use_seasonal_calibration:
            validation_seasonal_factors, validation_default_factor = (
                build_seasonal_calibration_factors(
                    backtest_df=training_data.loc[fit_index],
                    residual_factors=fit_residual_factor,
                    use_seasonal_calibration=True,
                    prior_mwh=seasonal_prior_mwh,
                    lower_bound=seasonal_lower_bound,
                    upper_bound=seasonal_upper_bound,
                )
            )
            if validation_seasonal_factors:
                fit_month_factor = (
                    training_data.loc[fit_index, "IntervalStartDT"]
                    .dt.month.map(validation_seasonal_factors)
                    .fillna(validation_default_factor)
                )
                validation_month_factor = (
                    validation_subset["IntervalStartDT"]
                    .dt.month.map(validation_seasonal_factors)
                    .fillna(validation_default_factor)
                )
                fit_factor = fit_factor * fit_month_factor
                validation_factor = validation_factor * validation_month_factor
                validation_factor = validation_factor.clip(
                    lower_bound * seasonal_lower_bound,
                    upper_bound * seasonal_upper_bound,
                )
                fit_factor = fit_factor.clip(
                    lower_bound * seasonal_lower_bound,
                    upper_bound * seasonal_upper_bound,
                )

        validation_actual_kwh = validation_subset["Actual_kWh"].sum()
        validation_base_wmape = (
            (validation_subset["Forecast_kWh"] - validation_subset["Actual_kWh"])
            .abs()
            .sum()
            / validation_actual_kwh
            if validation_actual_kwh > 0
            else np.nan
        )
        validation_pre_regime_kwh = (
            validation_subset["Forecast_kWh"] * validation_factor
        )
        validation_pre_regime_wmape = calculate_energy_wmape(
            validation_subset["Actual_kWh"],
            validation_pre_regime_kwh,
        )
        validation_calibrated_kwh = validation_pre_regime_kwh
        validation_regime_factor_count = 0
        if use_regime_calibration:
            validation_regime_factors = build_regime_factor_map(
                calibration_data=training_data.loc[fit_index],
                preliminary_forecast_kwh=training_data.loc[fit_index, "Forecast_kWh"]
                * fit_factor,
                min_rows=regime_min_rows,
                min_forecast_mwh=regime_min_forecast_mwh,
                prior_mwh=regime_prior_mwh,
                lower_bound=regime_lower_bound,
                upper_bound=regime_upper_bound,
            )
            validation_regime_factors = prune_regime_factors_by_holdout(
                validation_data=validation_subset,
                pre_regime_forecast_kwh=validation_pre_regime_kwh,
                candidate_factors=validation_regime_factors,
            )
            validated_regime_factor_keys = set(validation_regime_factors)
            validation_regime_factor_count = len(validation_regime_factors)
            validation_regime_factor = lookup_regime_calibration_factor_from_map(
                validation_subset,
                validation_regime_factors,
            )
            validation_calibrated_kwh = (
                validation_pre_regime_kwh * validation_regime_factor
            )
        validation_calibrated_wmape = (
            (validation_calibrated_kwh - validation_subset["Actual_kWh"]).abs().sum()
            / validation_actual_kwh
            if validation_actual_kwh > 0
            else np.nan
        )
        validation_pre_peak_wmape = validation_calibrated_wmape
        validation_pre_peak_capture = calculate_peak_capture(
            actual_power_kw_for_peak_calibration(validation_subset),
            validation_calibrated_kwh,
        )
        validation_peak_kwh = validation_calibrated_kwh
        validation_peak_factor = 1.0
        validation_peak_threshold_cf: Optional[float] = None
        validation_peak_rows = 0
        if use_peak_calibration:
            (
                validation_peak_factor,
                validation_peak_threshold_cf,
                validation_peak_diag,
            ) = build_peak_calibration_params(
                calibration_data=training_data.loc[fit_index],
                pre_peak_forecast_kwh=training_data.loc[fit_index, "Forecast_kWh"]
                * fit_factor,
                total_capacity_kw=total_capacity_kw,
                quantile=peak_quantile,
                min_rows=peak_min_rows,
                min_forecast_mwh=peak_min_forecast_mwh,
                lower_bound=peak_lower_bound,
                target_capture=peak_target_capture,
            )
            validation_peak_rows = int(validation_peak_diag.get("Rows", 0.0))
            validation_peak_factor_series = lookup_peak_calibration_factor(
                validation_subset,
                validation_calibrated_kwh,
                total_capacity_kw=total_capacity_kw,
                factor=validation_peak_factor,
                threshold_cf=validation_peak_threshold_cf,
            )
            validation_peak_kwh = (
                validation_calibrated_kwh * validation_peak_factor_series
            )
        validation_peak_wmape = calculate_energy_wmape(
            validation_subset["Actual_kWh"],
            validation_peak_kwh,
        )
        validation_peak_capture = calculate_peak_capture(
            actual_power_kw_for_peak_calibration(validation_subset),
            validation_peak_kwh,
        )
        logging.info(
            "Residual calibration holdout on latest %s rows (%s to %s): "
            "WMAPE %.2f%% -> %.2f%% residual/seasonal -> %.2f%% regime (%s factors)",
            len(validation_subset),
            validation_subset["IntervalStartDT"].min(),
            validation_subset["IntervalStartDT"].max(),
            (
                validation_base_wmape * 100
                if pd.notna(validation_base_wmape)
                else float("nan")
            ),
            (
                validation_pre_regime_wmape * 100
                if pd.notna(validation_pre_regime_wmape)
                else float("nan")
            ),
            (
                validation_calibrated_wmape * 100
                if pd.notna(validation_calibrated_wmape)
                else float("nan")
            ),
            validation_regime_factor_count,
        )
        if use_peak_calibration:
            logging.info(
                "Peak calibration holdout: capture %.2f%% -> %.2f%%, WMAPE %.2f%% -> %.2f%% "
                "(factor %.3f above forecast CF %.3f; %s fit rows)",
                (
                    validation_pre_peak_capture * 100
                    if pd.notna(validation_pre_peak_capture)
                    else float("nan")
                ),
                (
                    validation_peak_capture * 100
                    if pd.notna(validation_peak_capture)
                    else float("nan")
                ),
                (
                    validation_pre_peak_wmape * 100
                    if pd.notna(validation_pre_peak_wmape)
                    else float("nan")
                ),
                (
                    validation_peak_wmape * 100
                    if pd.notna(validation_peak_wmape)
                    else float("nan")
                ),
                validation_peak_factor,
                (
                    validation_peak_threshold_cf
                    if validation_peak_threshold_cf is not None
                    else float("nan")
                ),
                validation_peak_rows,
            )
        if (
            use_regime_calibration
            and pd.notna(validation_pre_regime_wmape)
            and pd.notna(validation_calibrated_wmape)
            and validation_calibrated_wmape
            > validation_pre_regime_wmape + REGIME_HOLDOUT_WMAPE_TOLERANCE
        ):
            disable_regime_due_holdout = True
            logging.info(
                "Regime calibration disabled for final model because selective holdout WMAPE did not improve "
                "(%.2f%% residual/seasonal vs %.2f%% regime).",
                validation_pre_regime_wmape * 100,
                validation_calibrated_wmape * 100,
            )
        if (
            use_peak_calibration
            and validation_peak_factor < 1.0
            and pd.notna(validation_pre_peak_wmape)
            and pd.notna(validation_peak_wmape)
        ):
            pre_peak_excess = (
                max(0.0, validation_pre_peak_capture - peak_target_capture)
                if pd.notna(validation_pre_peak_capture)
                else np.nan
            )
            post_peak_excess = (
                max(0.0, validation_peak_capture - peak_target_capture)
                if pd.notna(validation_peak_capture)
                else np.nan
            )
            if (
                validation_peak_wmape
                > validation_pre_peak_wmape + PEAK_HOLDOUT_WMAPE_TOLERANCE
                or (
                    pd.notna(pre_peak_excess)
                    and pd.notna(post_peak_excess)
                    and post_peak_excess > pre_peak_excess + PEAK_CAPTURE_TOLERANCE
                )
            ):
                disable_peak_due_holdout = True
                logging.info(
                    "Peak calibration disabled for final model because holdout did not improve cleanly "
                    "(capture %.2f%% -> %.2f%%, WMAPE %.2f%% -> %.2f%%).",
                    (
                        validation_pre_peak_capture * 100
                        if pd.notna(validation_pre_peak_capture)
                        else float("nan")
                    ),
                    (
                        validation_peak_capture * 100
                        if pd.notna(validation_peak_capture)
                        else float("nan")
                    ),
                    (
                        validation_pre_peak_wmape * 100
                        if pd.notna(validation_pre_peak_wmape)
                        else float("nan")
                    ),
                    (
                        validation_peak_wmape * 100
                        if pd.notna(validation_peak_wmape)
                        else float("nan")
                    ),
                )
    else:
        logging.info(
            "Residual calibration holdout skipped; %s rows are insufficient for %s training and %s validation rows",
            len(training_data),
            min_training_rows,
            min_validation_rows,
        )

    estimator = GradientBoostingRegressor(**estimator_params)
    estimator.fit(X, y, sample_weight=sample_weight)

    fitted_factor = pd.Series(estimator.predict(X), index=training_data.index).clip(
        lower_bound, upper_bound
    )
    seasonal_factors, seasonal_default_factor = build_seasonal_calibration_factors(
        backtest_df=training_data,
        residual_factors=fitted_factor,
        use_seasonal_calibration=use_seasonal_calibration,
        prior_mwh=seasonal_prior_mwh,
        lower_bound=seasonal_lower_bound,
        upper_bound=seasonal_upper_bound,
    )
    fitted_pre_regime_kwh = training_data["Forecast_kWh"] * fitted_factor
    if use_seasonal_calibration and seasonal_factors:
        fitted_month_factor = (
            training_data["IntervalStartDT"]
            .dt.month.map(seasonal_factors)
            .fillna(seasonal_default_factor)
        )
        fitted_pre_regime_kwh = fitted_pre_regime_kwh * fitted_month_factor

    regime_factors: dict[str, float] = {}
    fitted_kwh = fitted_pre_regime_kwh
    if use_regime_calibration and not disable_regime_due_holdout:
        regime_factors = build_regime_factor_map(
            calibration_data=training_data,
            preliminary_forecast_kwh=fitted_pre_regime_kwh,
            min_rows=regime_min_rows,
            min_forecast_mwh=regime_min_forecast_mwh,
            prior_mwh=regime_prior_mwh,
            lower_bound=regime_lower_bound,
            upper_bound=regime_upper_bound,
        )
        if validated_regime_factor_keys is not None:
            regime_factors = {
                key: factor
                for key, factor in regime_factors.items()
                if key in validated_regime_factor_keys
            }
        fitted_regime_factor = lookup_regime_calibration_factor_from_map(
            training_data, regime_factors
        )
        fitted_kwh = fitted_pre_regime_kwh * fitted_regime_factor

    peak_calibration_factor = 1.0
    peak_calibration_threshold_cf: Optional[float] = None
    pre_peak_fitted_kwh = fitted_kwh
    if use_peak_calibration and not disable_peak_due_holdout:
        peak_calibration_factor, peak_calibration_threshold_cf, peak_diag = (
            build_peak_calibration_params(
                calibration_data=training_data,
                pre_peak_forecast_kwh=pre_peak_fitted_kwh,
                total_capacity_kw=total_capacity_kw,
                quantile=peak_quantile,
                min_rows=peak_min_rows,
                min_forecast_mwh=peak_min_forecast_mwh,
                lower_bound=peak_lower_bound,
                target_capture=peak_target_capture,
            )
        )
        fitted_peak_factor = lookup_peak_calibration_factor(
            training_data,
            pre_peak_fitted_kwh,
            total_capacity_kw=total_capacity_kw,
            factor=peak_calibration_factor,
            threshold_cf=peak_calibration_threshold_cf,
        )
        fitted_kwh = pre_peak_fitted_kwh * fitted_peak_factor
        if peak_calibration_factor < 1.0 and peak_calibration_threshold_cf is not None:
            pre_peak_capture = calculate_peak_capture(
                actual_power_kw_for_peak_calibration(training_data),
                pre_peak_fitted_kwh,
            )
            post_peak_capture = calculate_peak_capture(
                actual_power_kw_for_peak_calibration(training_data),
                fitted_kwh,
            )
            logging.info(
                "Built peak calibration factor %.3f above forecast CF %.3f from %s upper-tail rows; "
                "in-sample peak capture %.2f%% -> %.2f%%",
                peak_calibration_factor,
                peak_calibration_threshold_cf,
                int(peak_diag.get("Rows", 0.0)),
                pre_peak_capture * 100 if pd.notna(pre_peak_capture) else float("nan"),
                (
                    post_peak_capture * 100
                    if pd.notna(post_peak_capture)
                    else float("nan")
                ),
            )

    pre_regime_wmape = calculate_energy_wmape(
        training_data["Actual_kWh"],
        fitted_pre_regime_kwh,
    )
    pre_peak_wmape = calculate_energy_wmape(
        training_data["Actual_kWh"],
        pre_peak_fitted_kwh,
    )
    calibrated_wmape = calculate_energy_wmape(
        training_data["Actual_kWh"],
        fitted_kwh,
    )
    base_wmape = calculate_energy_wmape(
        training_data["Actual_kWh"],
        training_data["Forecast_kWh"],
    )
    logging.info(
        "Trained residual calibration on %s daylight rows; aggregate factor %.3f, "
        "energy weight power %.2f, in-sample daylight WMAPE %.2f%% -> %.2f%% residual/seasonal "
        "-> %.2f%% regime (%s factors) -> %.2f%% peak",
        len(training_data),
        aggregate_factor,
        energy_weight_power if use_energy_weighting else 0.0,
        base_wmape * 100 if pd.notna(base_wmape) else float("nan"),
        pre_regime_wmape * 100 if pd.notna(pre_regime_wmape) else float("nan"),
        pre_peak_wmape * 100 if pd.notna(pre_peak_wmape) else float("nan"),
        len(regime_factors),
        calibrated_wmape * 100 if pd.notna(calibrated_wmape) else float("nan"),
    )
    return ResidualCalibrationModel(
        estimator=estimator,
        fallback_factor=aggregate_factor,
        feature_columns=CALIBRATION_FEATURE_COLUMNS,
        lower_bound=lower_bound,
        upper_bound=upper_bound,
        seasonal_factors=seasonal_factors,
        seasonal_default_factor=seasonal_default_factor,
        regime_factors=regime_factors,
        regime_default_factor=1.0,
        peak_calibration_factor=peak_calibration_factor,
        peak_calibration_threshold_cf=peak_calibration_threshold_cf,
    )


def build_seasonal_calibration_factors(
    backtest_df: pd.DataFrame,
    residual_factors: pd.Series,
    use_seasonal_calibration: bool,
    prior_mwh: float,
    lower_bound: float,
    upper_bound: float,
) -> tuple[dict[int, float], float]:
    """
    Build small month-level energy correction factors after residual calibration.
    """
    if not use_seasonal_calibration or backtest_df.empty:
        return {}, 1.0

    calibration_data = backtest_df.copy()
    calibration_data["IntervalStartDT"] = pd.to_datetime(
        calibration_data["IntervalStartDT"]
    )
    calibration_data["Forecast_kWh"] = pd.to_numeric(
        calibration_data["Forecast_kWh"], errors="coerce"
    )
    calibration_data["Actual_kWh"] = pd.to_numeric(
        calibration_data["Actual_kWh"], errors="coerce"
    )
    calibration_data["ResidualCalibrationFactor"] = residual_factors.reindex(
        calibration_data.index
    ).fillna(1.0)
    calibration_data["ResidualForecast_kWh"] = (
        calibration_data["Forecast_kWh"] * calibration_data["ResidualCalibrationFactor"]
    )
    calibration_data = calibration_data.dropna(
        subset=["Actual_kWh", "ResidualForecast_kWh"]
    )
    calibration_data = calibration_data[
        (calibration_data["Actual_kWh"] >= 0)
        & (calibration_data["ResidualForecast_kWh"] > 0)
    ].copy()
    if calibration_data.empty:
        logging.info(
            "Seasonal calibration skipped; no positive residual forecast rows are available."
        )
        return {}, 1.0

    aggregate_factor = (
        calibration_data["Actual_kWh"].sum()
        / calibration_data["ResidualForecast_kWh"].sum()
    )
    aggregate_factor = float(np.clip(aggregate_factor, lower_bound, upper_bound))
    prior_kwh = prior_mwh * 1000.0
    month_totals = (
        calibration_data.assign(Month=calibration_data["IntervalStartDT"].dt.month)
        .groupby("Month", as_index=False)
        .agg(
            Actual_kWh=("Actual_kWh", "sum"),
            ResidualForecast_kWh=("ResidualForecast_kWh", "sum"),
        )
    )

    seasonal_factors = {}
    for _, row in month_totals.iterrows():
        factor = (row["Actual_kWh"] + prior_kwh * aggregate_factor) / (
            row["ResidualForecast_kWh"] + prior_kwh
        )
        seasonal_factors[int(row["Month"])] = float(
            np.clip(factor, lower_bound, upper_bound)
        )

    logging.info(
        "Built seasonal calibration factors: %s",
        ", ".join(
            f"{month}={factor:.3f}"
            for month, factor in sorted(seasonal_factors.items())
        ),
    )
    return seasonal_factors, aggregate_factor


def predict_residual_calibration_factor(
    model: ResidualCalibrationModel,
    feature_df: pd.DataFrame,
    total_capacity_kw: float,
    latitude: float,
    longitude: float,
    timezone_name: str,
    array_tilt_degrees: float,
    array_azimuth_degrees: float,
    performance_ratio_upper_bound: float,
) -> pd.Series:
    """
    Predict bounded residual correction factors.
    """
    if feature_df.empty:
        return pd.Series(dtype="float64")

    if model.estimator is None:
        return pd.Series(model.fallback_factor, index=feature_df.index)

    calibration_features = add_residual_calibration_features(
        feature_df,
        total_capacity_kw=total_capacity_kw,
        latitude=latitude,
        longitude=longitude,
        timezone_name=timezone_name,
        array_tilt_degrees=array_tilt_degrees,
        array_azimuth_degrees=array_azimuth_degrees,
        performance_ratio_upper_bound=performance_ratio_upper_bound,
    )
    factors = pd.Series(
        model.estimator.predict(
            calibration_features[model.feature_columns].fillna(0.0)
        ),
        index=feature_df.index,
    )
    return factors.clip(lower=model.lower_bound, upper=model.upper_bound).fillna(
        model.fallback_factor
    )


def lookup_seasonal_calibration_factor(
    model: ResidualCalibrationModel,
    timestamps: pd.Series,
) -> pd.Series:
    """
    Lookup month-level seasonal factors, using the nearest trained month for unseen months.
    """
    timestamps = pd.to_datetime(timestamps)
    if not model.seasonal_factors:
        return pd.Series(1.0, index=timestamps.index)

    trained_months = sorted(model.seasonal_factors)

    def nearest_month_factor(month: int) -> float:
        if month in model.seasonal_factors:
            return model.seasonal_factors[month]
        nearest_month = min(trained_months, key=lambda trained: abs(trained - month))
        return model.seasonal_factors.get(nearest_month, model.seasonal_default_factor)

    return pd.Series(
        [nearest_month_factor(int(month)) for month in timestamps.dt.month],
        index=timestamps.index,
    ).fillna(model.seasonal_default_factor)


def lookup_regime_calibration_factor(
    model: ResidualCalibrationModel,
    feature_df: pd.DataFrame,
) -> pd.Series:
    """
    Lookup weather-regime residual correction factors.
    """
    return lookup_regime_calibration_factor_from_map(
        feature_df,
        model.regime_factors,
        default_factor=model.regime_default_factor,
    )


def apply_residual_calibration(
    interval_forecast: pd.DataFrame,
    calibration_model: ResidualCalibrationModel,
    total_capacity_kw: float,
    latitude: float,
    longitude: float,
    timezone_name: str,
    array_tilt_degrees: float,
    array_azimuth_degrees: float,
    performance_ratio_upper_bound: float,
) -> pd.DataFrame:
    """
    Apply learned residual calibration while preserving the base forecast columns.
    """
    out = interval_forecast.copy()
    if out.empty:
        return out

    out["BaseForecast_kWh"] = pd.to_numeric(
        out["Forecast_kWh"], errors="coerce"
    ).fillna(0.0)
    out["BaseForecast_kW"] = pd.to_numeric(out["Forecast_kW"], errors="coerce").fillna(
        0.0
    )
    factors = predict_residual_calibration_factor(
        calibration_model,
        out,
        total_capacity_kw=total_capacity_kw,
        latitude=latitude,
        longitude=longitude,
        timezone_name=timezone_name,
        array_tilt_degrees=array_tilt_degrees,
        array_azimuth_degrees=array_azimuth_degrees,
        performance_ratio_upper_bound=performance_ratio_upper_bound,
    )
    out["ResidualCalibrationFactor"] = factors.reindex(out.index).fillna(
        calibration_model.fallback_factor
    )
    out["SeasonalCalibrationFactor"] = lookup_seasonal_calibration_factor(
        calibration_model,
        out["IntervalStartDT"],
    )
    out["RegimeCalibrationFactor"] = lookup_regime_calibration_factor(
        calibration_model,
        out,
    )
    out["TotalCalibrationFactor"] = (
        out["ResidualCalibrationFactor"]
        * out["SeasonalCalibrationFactor"]
        * out["RegimeCalibrationFactor"]
    )
    active_mask = out["BaseForecast_kWh"] > 0
    out.loc[active_mask, "Forecast_kWh"] = (
        out.loc[active_mask, "BaseForecast_kWh"]
        * out.loc[active_mask, "TotalCalibrationFactor"]
    )
    out.loc[active_mask, "Forecast_kW"] = (
        out.loc[active_mask, "BaseForecast_kW"]
        * out.loc[active_mask, "TotalCalibrationFactor"]
    )
    out["PeakCalibrationFactor"] = lookup_peak_calibration_factor(
        out,
        out["Forecast_kW"],
        total_capacity_kw=total_capacity_kw,
        factor=calibration_model.peak_calibration_factor,
        threshold_cf=calibration_model.peak_calibration_threshold_cf,
    )
    out.loc[active_mask, "Forecast_kWh"] = (
        out.loc[active_mask, "Forecast_kWh"]
        * out.loc[active_mask, "PeakCalibrationFactor"]
    )
    out.loc[active_mask, "Forecast_kW"] = (
        out.loc[active_mask, "Forecast_kW"]
        * out.loc[active_mask, "PeakCalibrationFactor"]
    )
    out["TotalCalibrationFactor"] = (
        out["ResidualCalibrationFactor"]
        * out["SeasonalCalibrationFactor"]
        * out["RegimeCalibrationFactor"]
        * out["PeakCalibrationFactor"]
    )
    out.loc[~active_mask, "ResidualCalibrationFactor"] = 1.0
    out.loc[~active_mask, "SeasonalCalibrationFactor"] = 1.0
    out.loc[~active_mask, "RegimeCalibrationFactor"] = 1.0
    out.loc[~active_mask, "PeakCalibrationFactor"] = 1.0
    out.loc[~active_mask, "TotalCalibrationFactor"] = 1.0
    out.loc[~active_mask, "Forecast_kWh"] = 0.0
    out.loc[~active_mask, "Forecast_kW"] = 0.0
    return out


def build_interval_forecast(
    weather_df: pd.DataFrame,
    intrahour_shape: pd.DataFrame,
    capacity_kw: float,
    model: PerformanceModel,
    sites: Optional[pd.DataFrame] = None,
    latitude: float = ROSEVILLE_LATITUDE,
    longitude: float = ROSEVILLE_LONGITUDE,
    timezone_name: str = "America/Los_Angeles",
    min_solar_elevation: float = 0.0,
    array_tilt_degrees: float = DEFAULT_ARRAY_TILT_DEGREES,
    array_azimuth_degrees: float = DEFAULT_ARRAY_AZIMUTH_DEGREES,
    forecast_source: str = "forecast",
    daily_active_capacity: Optional[pd.DataFrame] = None,
    peak_hourly_kwh_quantile: Optional[float] = None,
) -> pd.DataFrame:
    """
    Build 15-minute kW forecast from hourly GHI and intra-hour interval shape.
    """
    legacy_capacity_mode = False
    if (
        sites is not None
        and not isinstance(sites, pd.DataFrame)
        and isinstance(longitude, str)
    ):
        old_latitude = float(sites)
        old_longitude = float(latitude)
        old_timezone = str(longitude)
        old_min_solar_elevation = float(timezone_name)
        sites = None
        latitude = old_latitude
        longitude = old_longitude
        timezone_name = old_timezone
        min_solar_elevation = old_min_solar_elevation
        legacy_capacity_mode = True

    forecast_df = add_performance_features(
        aggregate_weather_to_hourly(weather_df),
        latitude=latitude,
        longitude=longitude,
        timezone_name=timezone_name,
        array_tilt_degrees=array_tilt_degrees,
        array_azimuth_degrees=array_azimuth_degrees,
    )
    forecast_df = forecast_df.dropna(subset=["GHI_kWh_per_m2"])
    forecast_df["PerformanceRatio"] = predict_performance_ratio(model, forecast_df)
    if daily_active_capacity is not None and not daily_active_capacity.empty:
        forecast_df["ActiveCapacity_kW"] = _resolve_row_capacity(
            forecast_df["IntervalStartDT"],
            daily_active_capacity,
            capacity_kw,
        ).clip(lower=0.0)
        legacy_capacity_mode = True
    else:
        forecast_df["ActiveCapacity_kW"] = calculate_active_capacity_for_timestamps(
            forecast_df["IntervalStartDT"],
            sites=sites,
            default_capacity_kw=capacity_kw,
        ).clip(lower=0.0)
    if legacy_capacity_mode:
        available_per_kw = forecast_df["GHI_kWh_per_m2"]
    else:
        available_per_kw = forecast_df["PVWatts_kWh_per_kW"].where(
            forecast_df["PVWatts_kWh_per_kW"] > 0.0,
            forecast_df["GHI_kWh_per_m2"],
        )
    forecast_df["Hourly_kWh"] = (
        available_per_kw
        * forecast_df["ActiveCapacity_kW"]
        * forecast_df["PerformanceRatio"]
    )

    intrahour_shape = intrahour_shape.copy()
    interval_forecast = forecast_df.merge(intrahour_shape, on="hour", how="left")
    if peak_hourly_kwh_quantile is not None and not interval_forecast.empty:
        peak_threshold = float(
            interval_forecast["Hourly_kWh"].quantile(float(peak_hourly_kwh_quantile))
        )
        if pd.notna(peak_threshold) and peak_threshold > 0:
            peak_mask = interval_forecast["Hourly_kWh"].ge(
                peak_threshold
            ) & interval_forecast["SolarElevationDeg"].ge(25.0)
            if bool(peak_mask.any()):
                peak_shape = {0: 0.20, 15: 0.24, 30: 0.27, 45: 0.29}
                base = interval_forecast.loc[peak_mask, "IntraHourCoefficient"].fillna(
                    0.25
                )
                target = (
                    interval_forecast.loc[peak_mask, "minute"]
                    .map(peak_shape)
                    .fillna(0.25)
                )
                interval_forecast.loc[peak_mask, "IntraHourCoefficient"] = (
                    0.6 * base + 0.4 * target
                )
                affected_hours = interval_forecast.loc[
                    peak_mask, "IntervalStartDT"
                ].drop_duplicates()
                affected_mask = interval_forecast["IntervalStartDT"].isin(
                    affected_hours
                )
                coefficient_sum = (
                    interval_forecast.loc[affected_mask]
                    .groupby("IntervalStartDT")["IntraHourCoefficient"]
                    .transform("sum")
                )
                interval_forecast.loc[affected_mask, "IntraHourCoefficient"] = (
                    interval_forecast.loc[affected_mask, "IntraHourCoefficient"]
                    / coefficient_sum.replace(0, np.nan)
                ).fillna(0.25)
    interval_forecast["IntervalStartDT"] = interval_forecast[
        "IntervalStartDT"
    ] + pd.to_timedelta(interval_forecast["minute"], unit="m")
    interval_forecast["Forecast_kWh"] = (
        interval_forecast["Hourly_kWh"] * interval_forecast["IntraHourCoefficient"]
    )
    interval_forecast["Forecast_kW"] = (
        interval_forecast["Forecast_kWh"] / INTERVAL_HOURS
    )
    interval_forecast["ForecastSource"] = forecast_source
    interval_forecast = apply_solar_plausibility_filter(
        interval_forecast,
        timestamp_col="IntervalStartDT",
        energy_col="Forecast_kWh",
        power_col="Forecast_kW",
        latitude=latitude,
        longitude=longitude,
        timezone_name=timezone_name,
        min_solar_elevation=min_solar_elevation,
        label="forecast",
    )
    interval_forecast["BaseForecast_kWh"] = interval_forecast["Forecast_kWh"]
    interval_forecast["BaseForecast_kW"] = interval_forecast["Forecast_kW"]
    interval_forecast["ResidualCalibrationFactor"] = 1.0
    interval_forecast["SeasonalCalibrationFactor"] = 1.0
    interval_forecast["RegimeCalibrationFactor"] = 1.0
    interval_forecast["PeakCalibrationFactor"] = 1.0
    interval_forecast["TotalCalibrationFactor"] = 1.0
    interval_forecast["SameDayCorrectionFactor"] = 1.0
    return interval_forecast[
        [
            "IntervalStartDT",
            "Forecast_kWh",
            "Forecast_kW",
            "BaseForecast_kWh",
            "BaseForecast_kW",
            "ActiveCapacity_kW",
            "ResidualCalibrationFactor",
            "SeasonalCalibrationFactor",
            "RegimeCalibrationFactor",
            "PeakCalibrationFactor",
            "TotalCalibrationFactor",
            *FORECAST_DIAGNOSTIC_COLUMNS,
            "PerformanceRatio",
            "SameDayCorrectionFactor",
            "ForecastSource",
        ]
    ].sort_values("IntervalStartDT")


def load_same_day_export_actuals(
    parquet_root: Path,
    sites: pd.DataFrame,
    forecast_start_date: date,
    forecast_end_date: date,
    net_meter_export_source: str,
    latitude: float,
    longitude: float,
    timezone_name: str,
    min_solar_elevation: float,
    preloaded_intervals: Optional[pd.DataFrame] = None,
) -> pd.DataFrame:
    """
    Load completed same-day export actuals for forecast correction.
    """
    local_now = current_local_timestamp(timezone_name)
    local_today = local_now.date()
    if not (forecast_start_date <= local_today <= forecast_end_date):
        logging.info(
            "Same-day correction skipped; %s is outside the forecast window",
            local_today,
        )
        return pd.DataFrame()

    complete_interval_cutoff = local_now.floor("15min") - pd.Timedelta(minutes=15)
    if complete_interval_cutoff.date() < local_today:
        logging.info(
            "Same-day correction skipped; no completed intervals are available yet for %s",
            local_today,
        )
        return pd.DataFrame()

    if preloaded_intervals is not None and not preloaded_intervals.empty:
        preloaded = preloaded_intervals.copy()
        preloaded["IntervalStartDT"] = pd.to_datetime(preloaded["IntervalStartDT"])
        preloaded_today = preloaded[
            preloaded["IntervalStartDT"].dt.date == local_today
        ].copy()
        if not preloaded_today.empty:
            actuals = preloaded_today
        else:
            actuals = pd.DataFrame()
    else:
        actuals = pd.DataFrame()

    if actuals.empty:
        try:
            actuals = load_rec_interval_data(
                parquet_root=parquet_root,
                sites=sites,
                start_date=local_today,
                end_date=local_today,
                net_meter_export_source=net_meter_export_source,
                latitude=latitude,
                longitude=longitude,
                timezone_name=timezone_name,
                min_solar_elevation=min_solar_elevation,
            )
        except (FileNotFoundError, ValueError) as exc:
            logging.info(
                "Same-day correction skipped; no same-day export actuals found: %s", exc
            )
            return pd.DataFrame()

    actuals = actuals.copy()
    actuals["IntervalStartDT"] = pd.to_datetime(actuals["IntervalStartDT"])
    actuals = actuals[
        (actuals["IntervalStartDT"].dt.date == local_today)
        & (actuals["IntervalStartDT"] <= complete_interval_cutoff)
    ].copy()
    if actuals.empty:
        logging.info(
            "Same-day correction skipped; no completed same-day export intervals found"
        )
        return actuals

    logging.info(
        "Loaded %s completed same-day export intervals through %s for correction",
        f"{len(actuals):,}",
        actuals["IntervalStartDT"].max(),
    )
    return actuals


def apply_same_day_actual_correction(
    interval_forecast: pd.DataFrame,
    same_day_actuals: pd.DataFrame,
    timezone_name: str,
    min_observed_intervals: int,
    min_observed_forecast_kwh: float,
    lower_bound: float,
    upper_bound: float,
    half_life_hours: float,
    weather_similarity_floor: float,
) -> pd.DataFrame:
    """
    Scale remaining same-day intervals with a weather-aware, lead-time-decayed correction.
    """
    out = interval_forecast.copy()
    out["IntervalStartDT"] = pd.to_datetime(out["IntervalStartDT"])
    if "SameDayCorrectionFactor" not in out.columns:
        out["SameDayCorrectionFactor"] = 1.0

    if same_day_actuals.empty:
        return out

    local_today = current_local_timestamp(timezone_name).date()
    actuals = same_day_actuals.copy()
    actuals["IntervalStartDT"] = pd.to_datetime(actuals["IntervalStartDT"])
    actuals = actuals[actuals["IntervalStartDT"].dt.date == local_today].copy()
    if actuals.empty:
        return out

    comparison_columns = [
        "IntervalStartDT",
        "Forecast_kWh",
        "CloudCoverPct",
        "ClearSkyIndex",
        "WeatherGHI_Wm2",
    ]
    comparison = actuals[["IntervalStartDT", "Export_kWh"]].merge(
        out[[column for column in comparison_columns if column in out.columns]],
        on="IntervalStartDT",
        how="inner",
    )
    comparison = comparison[
        comparison["Forecast_kWh"].notna() & (comparison["Forecast_kWh"] > 0)
    ].copy()

    observed_intervals = len(comparison)
    observed_forecast_kwh = comparison["Forecast_kWh"].sum()
    observed_actual_kwh = comparison["Export_kWh"].sum()
    if (
        observed_intervals < min_observed_intervals
        or observed_forecast_kwh < min_observed_forecast_kwh
    ):
        logging.info(
            "Same-day correction skipped; observed %s daylight intervals and %.2f forecast kWh",
            observed_intervals,
            observed_forecast_kwh,
        )
        return out

    raw_factor = (
        observed_actual_kwh / observed_forecast_kwh
        if observed_forecast_kwh > 0
        else 1.0
    )
    correction_factor = float(np.clip(raw_factor, lower_bound, upper_bound))
    last_observed_interval = actuals["IntervalStartDT"].max()
    future_same_day_mask = (out["IntervalStartDT"].dt.date == local_today) & (
        out["IntervalStartDT"] > last_observed_interval
    )
    intervals_to_correct = int(future_same_day_mask.sum())
    if intervals_to_correct == 0:
        logging.info(
            "Same-day correction calculated %.3f but no remaining same-day intervals need correction",
            correction_factor,
        )
        return out

    future = out.loc[future_same_day_mask].copy()
    lead_hours = (
        (future["IntervalStartDT"] - last_observed_interval).dt.total_seconds() / 3600.0
    ).clip(lower=0.0)
    lead_weight = 0.5 ** (lead_hours / half_life_hours)
    weather_similarity = pd.Series(1.0, index=future.index, dtype="float64")

    comparison_weights = comparison["Forecast_kWh"].clip(lower=0.0)
    if comparison_weights.sum() <= 0:
        comparison_weights = pd.Series(1.0, index=comparison.index)

    similarity_components = []
    if "CloudCoverPct" in comparison.columns and "CloudCoverPct" in future.columns:
        observed_cloud = float(
            np.average(
                pd.to_numeric(comparison["CloudCoverPct"], errors="coerce").fillna(0.0),
                weights=comparison_weights,
            )
        )
        future_cloud = pd.to_numeric(future["CloudCoverPct"], errors="coerce")
        similarity_components.append(
            (1.0 - (future_cloud - observed_cloud).abs() / 100.0).clip(0.0, 1.0)
        )
    if "ClearSkyIndex" in comparison.columns and "ClearSkyIndex" in future.columns:
        observed_clear_sky_index = float(
            np.average(
                pd.to_numeric(comparison["ClearSkyIndex"], errors="coerce").fillna(0.0),
                weights=comparison_weights,
            )
        )
        future_clear_sky_index = pd.to_numeric(future["ClearSkyIndex"], errors="coerce")
        similarity_components.append(
            (1.0 - (future_clear_sky_index - observed_clear_sky_index).abs()).clip(
                0.0, 1.0
            )
        )

    if similarity_components:
        weather_similarity = (
            pd.concat(similarity_components, axis=1)
            .mean(axis=1)
            .clip(lower=weather_similarity_floor, upper=1.0)
            .fillna(weather_similarity_floor)
        )

    effective_factor = (
        1.0 + (correction_factor - 1.0) * lead_weight * weather_similarity
    )
    effective_factor = effective_factor.clip(lower=lower_bound, upper=upper_bound)
    out.loc[future_same_day_mask, "Forecast_kWh"] *= effective_factor
    out.loc[future_same_day_mask, "Forecast_kW"] *= effective_factor
    out.loc[future_same_day_mask, "SameDayCorrectionFactor"] = effective_factor
    logging.info(
        "Applied weather-aware same-day correction %.3f raw / %.3f average effective factor "
        "to %s remaining intervals (actual %.2f kWh / forecast %.2f kWh over %s observed daylight intervals)",
        correction_factor,
        effective_factor.mean(),
        intervals_to_correct,
        observed_actual_kwh,
        observed_forecast_kwh,
        observed_intervals,
    )
    return out


def resample_interval_forecast_to_hourly(
    interval_forecast: pd.DataFrame, total_capacity_kw: float
) -> pd.DataFrame:
    """
    Resample 15-minute forecast rows to hourly forecast output.
    """
    logging.info("Resampling forecast to hourly and converting to MW")
    interval_forecast = interval_forecast.copy()
    if "BaseForecast_kWh" not in interval_forecast.columns:
        interval_forecast["BaseForecast_kWh"] = interval_forecast["Forecast_kWh"]
    if "BaseForecast_kW" not in interval_forecast.columns:
        interval_forecast["BaseForecast_kW"] = interval_forecast["Forecast_kW"]
    if "ResidualCalibrationFactor" not in interval_forecast.columns:
        interval_forecast["ResidualCalibrationFactor"] = 1.0
    if "SeasonalCalibrationFactor" not in interval_forecast.columns:
        interval_forecast["SeasonalCalibrationFactor"] = 1.0
    if "RegimeCalibrationFactor" not in interval_forecast.columns:
        interval_forecast["RegimeCalibrationFactor"] = 1.0
    if "PeakCalibrationFactor" not in interval_forecast.columns:
        interval_forecast["PeakCalibrationFactor"] = 1.0
    if "TotalCalibrationFactor" not in interval_forecast.columns:
        interval_forecast["TotalCalibrationFactor"] = (
            interval_forecast["ResidualCalibrationFactor"]
            * interval_forecast["SeasonalCalibrationFactor"]
            * interval_forecast["RegimeCalibrationFactor"]
            * interval_forecast["PeakCalibrationFactor"]
        )
    if "ActiveCapacity_kW" not in interval_forecast.columns:
        interval_forecast["ActiveCapacity_kW"] = total_capacity_kw

    for column in WEATHER_OUTPUT_COLUMNS:
        if column not in interval_forecast.columns:
            interval_forecast[column] = np.nan
    for column in PHYSICS_FEATURE_COLUMNS:
        if column not in interval_forecast.columns:
            interval_forecast[column] = np.nan
    if "SolarElevationDeg" not in interval_forecast.columns:
        interval_forecast["SolarElevationDeg"] = np.nan
    if "SolarAzimuthDeg" not in interval_forecast.columns:
        interval_forecast["SolarAzimuthDeg"] = np.nan

    hourly_aggregations = {
        "Forecast_kWh": ("Forecast_kWh", "sum"),
        "Forecast_kW": ("Forecast_kW", "mean"),
        "BaseForecast_kWh": ("BaseForecast_kWh", "sum"),
        "BaseForecast_kW": ("BaseForecast_kW", "mean"),
        "ActiveCapacity_kW": ("ActiveCapacity_kW", "mean"),
    }
    for column in [
        *WEATHER_OUTPUT_COLUMNS,
        "SolarElevationDeg",
        "SolarAzimuthDeg",
        *PHYSICS_FEATURE_COLUMNS,
    ]:
        hourly_aggregations.setdefault(column, (column, "mean"))
    hourly_aggregations.update(
        {
            "PerformanceRatio": ("PerformanceRatio", "mean"),
            "ResidualCalibrationFactor": ("ResidualCalibrationFactor", "mean"),
            "SeasonalCalibrationFactor": ("SeasonalCalibrationFactor", "mean"),
            "RegimeCalibrationFactor": ("RegimeCalibrationFactor", "mean"),
            "PeakCalibrationFactor": ("PeakCalibrationFactor", "mean"),
            "TotalCalibrationFactor": ("TotalCalibrationFactor", "mean"),
            "SameDayCorrectionFactor": ("SameDayCorrectionFactor", "max"),
            "ForecastSource": ("ForecastSource", "last"),
        }
    )
    if "CustomerSegment" in interval_forecast.columns:
        hourly_forecast = (
            interval_forecast.groupby(
                [pd.Grouper(key="IntervalStartDT", freq="h"), "CustomerSegment"],
                as_index=False,
            )
            .agg(**hourly_aggregations)
            .sort_values(["IntervalStartDT", "CustomerSegment"])
        )
    else:
        hourly_forecast = (
            interval_forecast.set_index("IntervalStartDT")
            .resample("h")
            .agg(**hourly_aggregations)
        )
        hourly_forecast.reset_index(inplace=True)
    hourly_forecast["Forecast_MW"] = hourly_forecast["Forecast_kW"] / 1000.0
    hourly_forecast["BaseForecast_MW"] = hourly_forecast["BaseForecast_kW"] / 1000.0
    capacity_denominator = pd.to_numeric(
        hourly_forecast["ActiveCapacity_kW"], errors="coerce"
    )
    if total_capacity_kw > 0:
        capacity_denominator = capacity_denominator.where(
            capacity_denominator > 0, total_capacity_kw
        )
    else:
        capacity_denominator = capacity_denominator.where(capacity_denominator > 0)
    if capacity_denominator.notna().any():
        hourly_forecast["CapacityFactor"] = (
            (hourly_forecast["Forecast_kW"] / capacity_denominator)
            .clip(lower=0.0)
            .fillna(0.0)
        )
        hourly_forecast["BaseCapacityFactor"] = (
            (hourly_forecast["BaseForecast_kW"] / capacity_denominator)
            .clip(lower=0.0)
            .fillna(0.0)
        )
    else:
        hourly_forecast["CapacityFactor"] = 0.0
        hourly_forecast["BaseCapacityFactor"] = 0.0
    return add_hour_ending_column(hourly_forecast)


def resample_actual_export_to_hourly(rec_interval_df: pd.DataFrame) -> pd.DataFrame:
    """
    Resample 15-minute actual export rows to hourly actual output.
    """
    if "CustomerSegment" in rec_interval_df.columns:
        rec_hourly = (
            rec_interval_df.groupby(
                [pd.Grouper(key="IntervalStartDT", freq="h"), "CustomerSegment"],
                as_index=False,
            )
            .agg(
                Export_kWh=("Export_kWh", "sum"),
                Export_kW=("Export_kW", "mean"),
            )
            .sort_values(["IntervalStartDT", "CustomerSegment"])
        )
    else:
        rec_hourly = (
            rec_interval_df.set_index("IntervalStartDT")
            .resample("h")
            .agg(
                Export_kWh=("Export_kWh", "sum"),
                Export_kW=("Export_kW", "mean"),
            )
        )
        rec_hourly.reset_index(inplace=True)
    rec_hourly["Export_kW"] = rec_hourly["Export_kW"].fillna(rec_hourly["Export_kWh"])
    rec_hourly["Export_MW"] = rec_hourly["Export_kW"] / 1000.0
    return add_hour_ending_column(rec_hourly)


def build_hourly_backtest(
    rec_interval_df: pd.DataFrame,
    interval_backtest_forecast: pd.DataFrame,
    total_capacity_kw: float,
) -> pd.DataFrame:
    """
    Compare hourly model forecast against REC/NET-derived actual export.
    """
    actual_hourly = resample_actual_export_to_hourly(rec_interval_df).rename(
        columns={
            "Export_MW": "Actual_MW",
            "Export_kWh": "Actual_kWh",
            "Export_kW": "Actual_kW",
        }
    )
    forecast_hourly = resample_interval_forecast_to_hourly(
        interval_backtest_forecast,
        total_capacity_kw,
    )
    merge_keys = ["IntervalStartDT"]
    actual_columns = ["IntervalStartDT", "HE", "Actual_MW", "Actual_kWh", "Actual_kW"]
    forecast_columns = [
        "IntervalStartDT",
        "Forecast_MW",
        "Forecast_kWh",
        "BaseForecast_MW",
        "BaseForecast_kWh",
        "CapacityFactor",
        "BaseCapacityFactor",
        "ActiveCapacity_kW",
        *FORECAST_DIAGNOSTIC_COLUMNS,
        "PerformanceRatio",
        "ResidualCalibrationFactor",
        "SeasonalCalibrationFactor",
        "RegimeCalibrationFactor",
        "PeakCalibrationFactor",
        "TotalCalibrationFactor",
        "SameDayCorrectionFactor",
        "ForecastSource",
    ]
    if (
        "CustomerSegment" in actual_hourly.columns
        and "CustomerSegment" in forecast_hourly.columns
    ):
        merge_keys.append("CustomerSegment")
        actual_columns.insert(1, "CustomerSegment")
        forecast_columns.insert(1, "CustomerSegment")

    backtest = actual_hourly[actual_columns].merge(
        forecast_hourly[forecast_columns],
        on=merge_keys,
        how="inner",
    )
    if "BaseForecast_MW" not in backtest.columns:
        backtest["BaseForecast_MW"] = backtest["Forecast_MW"]
    if "BaseForecast_kWh" not in backtest.columns:
        backtest["BaseForecast_kWh"] = backtest["Forecast_kWh"]
    if "ResidualCalibrationFactor" not in backtest.columns:
        backtest["ResidualCalibrationFactor"] = 1.0
    if "SeasonalCalibrationFactor" not in backtest.columns:
        backtest["SeasonalCalibrationFactor"] = 1.0
    if "RegimeCalibrationFactor" not in backtest.columns:
        backtest["RegimeCalibrationFactor"] = 1.0
    if "PeakCalibrationFactor" not in backtest.columns:
        backtest["PeakCalibrationFactor"] = 1.0
    if "TotalCalibrationFactor" not in backtest.columns:
        backtest["TotalCalibrationFactor"] = (
            backtest["ResidualCalibrationFactor"]
            * backtest["SeasonalCalibrationFactor"]
            * backtest["RegimeCalibrationFactor"]
            * backtest["PeakCalibrationFactor"]
        )

    backtest["Error_MW"] = backtest["Forecast_MW"] - backtest["Actual_MW"]
    backtest["Error_kWh"] = backtest["Forecast_kWh"] - backtest["Actual_kWh"]
    backtest["AbsError_MW"] = backtest["Error_MW"].abs()
    backtest["AbsError_kWh"] = backtest["Error_kWh"].abs()
    backtest["BaseError_MW"] = backtest["BaseForecast_MW"] - backtest["Actual_MW"]
    backtest["BaseError_kWh"] = backtest["BaseForecast_kWh"] - backtest["Actual_kWh"]
    backtest["BaseAbsError_MW"] = backtest["BaseError_MW"].abs()
    backtest["BaseAbsError_kWh"] = backtest["BaseError_kWh"].abs()
    backtest["APE"] = pd.NA
    backtest["BaseAPE"] = pd.NA
    positive_actual_mask = backtest["Actual_kWh"] > 0
    backtest.loc[positive_actual_mask, "APE"] = (
        backtest.loc[positive_actual_mask, "AbsError_kWh"]
        / backtest.loc[positive_actual_mask, "Actual_kWh"]
    )
    backtest.loc[positive_actual_mask, "BaseAPE"] = (
        backtest.loc[positive_actual_mask, "BaseAbsError_kWh"]
        / backtest.loc[positive_actual_mask, "Actual_kWh"]
    )
    backtest["BacktestForecast"] = True
    return backtest


def calculate_backtest_summary(backtest_df: pd.DataFrame) -> pd.DataFrame:
    """
    Build one-row backtest accuracy metrics for the hourly comparison file.
    """
    if backtest_df.empty:
        return pd.DataFrame(
            [
                {
                    "Start": pd.NaT,
                    "End": pd.NaT,
                    "Intervals": 0,
                    "Actual_MWh": 0.0,
                    "Forecast_MWh": 0.0,
                    "BaseForecast_MWh": 0.0,
                    "Bias_MWh": 0.0,
                    "BaseBias_MWh": 0.0,
                    "BiasPct": pd.NA,
                    "BaseBiasPct": pd.NA,
                    "MAE_MW": pd.NA,
                    "BaseMAE_MW": pd.NA,
                    "RMSE_MW": pd.NA,
                    "BaseRMSE_MW": pd.NA,
                    "WMAPE": pd.NA,
                    "BaseWMAPE": pd.NA,
                    "WMAPEImprovementPct": pd.NA,
                    "MAPE": pd.NA,
                    "BaseMAPE": pd.NA,
                    "ActualPeak_MW": pd.NA,
                    "ForecastPeak_MW": pd.NA,
                    "BaseForecastPeak_MW": pd.NA,
                    "RawIntervals": 0,
                    "ExcludedIntervals": 0,
                    "ExcludedActual_MWh": 0.0,
                    "ExcludedForecast_MWh": 0.0,
                    "RawActual_MWh": 0.0,
                    "RawForecast_MWh": 0.0,
                    "RawBaseForecast_MWh": 0.0,
                }
            ]
        )

    raw_backtest_df = backtest_df.copy()
    excluded_mask = _actual_quality_exclusion_mask(raw_backtest_df)
    if bool(excluded_mask.any()):
        filtered = raw_backtest_df[~excluded_mask].copy()
        summary = calculate_backtest_summary(filtered)
        excluded_df = raw_backtest_df[excluded_mask].copy()
        summary["RawIntervals"] = len(raw_backtest_df)
        summary["ExcludedIntervals"] = int(excluded_mask.sum())
        summary["ExcludedActual_MWh"] = excluded_df["Actual_kWh"].sum() / 1000.0
        summary["ExcludedForecast_MWh"] = excluded_df["Forecast_kWh"].sum() / 1000.0
        summary["RawActual_MWh"] = raw_backtest_df["Actual_kWh"].sum() / 1000.0
        summary["RawForecast_MWh"] = raw_backtest_df["Forecast_kWh"].sum() / 1000.0
        base_col = (
            "BaseForecast_kWh"
            if "BaseForecast_kWh" in raw_backtest_df.columns
            else "Forecast_kWh"
        )
        summary["RawBaseForecast_MWh"] = raw_backtest_df[base_col].sum() / 1000.0
        return summary

    actual_mwh = backtest_df["Actual_kWh"].sum() / 1000.0
    forecast_mwh = backtest_df["Forecast_kWh"].sum() / 1000.0
    if "BaseForecast_kWh" not in backtest_df.columns:
        backtest_df = backtest_df.copy()
        backtest_df["BaseForecast_kWh"] = backtest_df["Forecast_kWh"]
        backtest_df["BaseForecast_MW"] = backtest_df["Forecast_MW"]
        backtest_df["BaseError_MW"] = backtest_df["Error_MW"]
        backtest_df["BaseAbsError_MW"] = backtest_df["AbsError_MW"]
        backtest_df["BaseAbsError_kWh"] = backtest_df["AbsError_kWh"]
        backtest_df["BaseAPE"] = backtest_df["APE"]
    base_forecast_mwh = backtest_df["BaseForecast_kWh"].sum() / 1000.0
    bias_mwh = forecast_mwh - actual_mwh
    base_bias_mwh = base_forecast_mwh - actual_mwh
    rmse_mw = math.sqrt(float((backtest_df["Error_MW"] ** 2).mean()))
    base_rmse_mw = math.sqrt(float((backtest_df["BaseError_MW"] ** 2).mean()))
    if "APE" not in backtest_df.columns:
        backtest_df = backtest_df.copy()
        backtest_df["APE"] = pd.NA
        positive_actual_mask = (
            pd.to_numeric(backtest_df["Actual_kWh"], errors="coerce") > 0
        )
        backtest_df.loc[positive_actual_mask, "APE"] = (
            backtest_df.loc[positive_actual_mask, "AbsError_kWh"]
            / backtest_df.loc[positive_actual_mask, "Actual_kWh"]
        )
    if "BaseAPE" not in backtest_df.columns:
        backtest_df = backtest_df.copy()
        backtest_df["BaseAPE"] = pd.NA
        positive_actual_mask = (
            pd.to_numeric(backtest_df["Actual_kWh"], errors="coerce") > 0
        )
        backtest_df.loc[positive_actual_mask, "BaseAPE"] = (
            backtest_df.loc[positive_actual_mask, "BaseAbsError_kWh"]
            / backtest_df.loc[positive_actual_mask, "Actual_kWh"]
        )
    wmape = (
        backtest_df["AbsError_kWh"].sum() / backtest_df["Actual_kWh"].sum()
        if backtest_df["Actual_kWh"].sum() > 0
        else pd.NA
    )
    base_wmape = (
        backtest_df["BaseAbsError_kWh"].sum() / backtest_df["Actual_kWh"].sum()
        if backtest_df["Actual_kWh"].sum() > 0
        else pd.NA
    )
    wmape_improvement = (
        (base_wmape - wmape) / base_wmape
        if pd.notna(base_wmape) and base_wmape > 0 and pd.notna(wmape)
        else pd.NA
    )
    mape = backtest_df["APE"].dropna().mean()
    base_mape = backtest_df["BaseAPE"].dropna().mean()
    return pd.DataFrame(
        [
            {
                "Start": backtest_df["IntervalStartDT"].min(),
                "End": backtest_df["IntervalStartDT"].max(),
                "Intervals": len(backtest_df),
                "Actual_MWh": actual_mwh,
                "Forecast_MWh": forecast_mwh,
                "BaseForecast_MWh": base_forecast_mwh,
                "Bias_MWh": bias_mwh,
                "BaseBias_MWh": base_bias_mwh,
                "BiasPct": bias_mwh / actual_mwh if actual_mwh > 0 else pd.NA,
                "BaseBiasPct": base_bias_mwh / actual_mwh if actual_mwh > 0 else pd.NA,
                "MAE_MW": backtest_df["AbsError_MW"].mean(),
                "BaseMAE_MW": backtest_df["BaseAbsError_MW"].mean(),
                "RMSE_MW": rmse_mw,
                "BaseRMSE_MW": base_rmse_mw,
                "WMAPE": wmape,
                "BaseWMAPE": base_wmape,
                "WMAPEImprovementPct": wmape_improvement,
                "MAPE": mape,
                "BaseMAPE": base_mape,
                "ActualPeak_MW": backtest_df["Actual_MW"].max(),
                "ForecastPeak_MW": backtest_df["Forecast_MW"].max(),
                "BaseForecastPeak_MW": backtest_df["BaseForecast_MW"].max(),
            }
        ]
    )


def _solar_metric_row(
    data: pd.DataFrame,
    slice_name: str,
    slice_group: str,
    slice_value: object,
) -> dict:
    """
    Build one solar backtest metrics row for a supplied slice.
    """
    row = {
        "Slice": slice_name,
        "SliceGroup": slice_group,
        "SliceValue": slice_value,
        "N": int(len(data)),
        "Actual_MWh": np.nan,
        "Forecast_MWh": np.nan,
        "BaseForecast_MWh": np.nan,
        "Bias_MWh": np.nan,
        "BaseBias_MWh": np.nan,
        "BiasPct": np.nan,
        "BaseBiasPct": np.nan,
        "MAE_MW": np.nan,
        "BaseMAE_MW": np.nan,
        "RMSE_MW": np.nan,
        "BaseRMSE_MW": np.nan,
        "WMAPE_PCT": np.nan,
        "BaseWMAPE_PCT": np.nan,
        "WMAPEImprovementPct": np.nan,
        "MAPE_PCT": np.nan,
        "BaseMAPE_PCT": np.nan,
        "Underforecast_Rate_PCT": np.nan,
        "P90_AbsError_MW": np.nan,
        "Max_Underforecast_MW": np.nan,
        "Max_Overforecast_MW": np.nan,
        "ActualPeak_MW": np.nan,
        "ForecastPeak_MW": np.nan,
        "BaseForecastPeak_MW": np.nan,
        "ForecastAtActualPeak_MW": np.nan,
        "UnderforecastAtActualPeak_MW": np.nan,
    }
    if data.empty:
        return row

    def numeric_column(column: str) -> pd.Series:
        if column not in data.columns:
            return pd.Series(0.0, index=data.index, dtype="float64")
        return pd.to_numeric(data[column], errors="coerce").fillna(0.0)

    actual_kwh = numeric_column("Actual_kWh")
    forecast_kwh = numeric_column("Forecast_kWh")
    base_forecast_kwh = numeric_column("BaseForecast_kWh")
    actual_mw = numeric_column("Actual_MW")
    forecast_mw = numeric_column("Forecast_MW")
    base_forecast_mw = numeric_column("BaseForecast_MW")
    error_mw = forecast_mw - actual_mw
    base_error_mw = base_forecast_mw - actual_mw
    error_kwh = forecast_kwh - actual_kwh
    base_error_kwh = base_forecast_kwh - actual_kwh
    actual_kwh_sum = float(actual_kwh.sum())
    forecast_kwh_sum = float(forecast_kwh.sum())
    base_forecast_kwh_sum = float(base_forecast_kwh.sum())
    wmape = error_kwh.abs().sum() / actual_kwh_sum if actual_kwh_sum > 0 else np.nan
    base_wmape = (
        base_error_kwh.abs().sum() / actual_kwh_sum if actual_kwh_sum > 0 else np.nan
    )
    wmape_improvement = (
        (base_wmape - wmape) / base_wmape
        if pd.notna(base_wmape) and base_wmape > 0 and pd.notna(wmape)
        else np.nan
    )
    positive_actual = actual_kwh > 0
    mape = (
        (error_kwh[positive_actual].abs() / actual_kwh[positive_actual]).mean()
        if positive_actual.any()
        else np.nan
    )
    base_mape = (
        (base_error_kwh[positive_actual].abs() / actual_kwh[positive_actual]).mean()
        if positive_actual.any()
        else np.nan
    )
    actual_peak = float(actual_mw.max()) if len(actual_mw) else np.nan
    if pd.notna(actual_peak) and len(actual_mw):
        actual_peak_index = actual_mw.idxmax()
        forecast_at_actual_peak = float(forecast_mw.loc[actual_peak_index])
        under_at_actual_peak = actual_peak - forecast_at_actual_peak
    else:
        forecast_at_actual_peak = np.nan
        under_at_actual_peak = np.nan

    row.update(
        {
            "Actual_MWh": actual_kwh_sum / 1000.0,
            "Forecast_MWh": forecast_kwh_sum / 1000.0,
            "BaseForecast_MWh": base_forecast_kwh_sum / 1000.0,
            "Bias_MWh": (forecast_kwh_sum - actual_kwh_sum) / 1000.0,
            "BaseBias_MWh": (base_forecast_kwh_sum - actual_kwh_sum) / 1000.0,
            "BiasPct": (
                (forecast_kwh_sum - actual_kwh_sum) / actual_kwh_sum
                if actual_kwh_sum > 0
                else np.nan
            ),
            "BaseBiasPct": (
                (base_forecast_kwh_sum - actual_kwh_sum) / actual_kwh_sum
                if actual_kwh_sum > 0
                else np.nan
            ),
            "MAE_MW": float(error_mw.abs().mean()),
            "BaseMAE_MW": float(base_error_mw.abs().mean()),
            "RMSE_MW": math.sqrt(float((error_mw**2).mean())),
            "BaseRMSE_MW": math.sqrt(float((base_error_mw**2).mean())),
            "WMAPE_PCT": wmape * 100.0 if pd.notna(wmape) else np.nan,
            "BaseWMAPE_PCT": base_wmape * 100.0 if pd.notna(base_wmape) else np.nan,
            "WMAPEImprovementPct": (
                wmape_improvement * 100.0 if pd.notna(wmape_improvement) else np.nan
            ),
            "MAPE_PCT": mape * 100.0 if pd.notna(mape) else np.nan,
            "BaseMAPE_PCT": base_mape * 100.0 if pd.notna(base_mape) else np.nan,
            "Underforecast_Rate_PCT": float((error_mw < 0).mean() * 100.0),
            "P90_AbsError_MW": float(error_mw.abs().quantile(0.90)),
            "Max_Underforecast_MW": float((actual_mw - forecast_mw).max()),
            "Max_Overforecast_MW": float((forecast_mw - actual_mw).max()),
            "ActualPeak_MW": actual_peak,
            "ForecastPeak_MW": float(forecast_mw.max()),
            "BaseForecastPeak_MW": float(base_forecast_mw.max()),
            "ForecastAtActualPeak_MW": forecast_at_actual_peak,
            "UnderforecastAtActualPeak_MW": under_at_actual_peak,
        }
    )
    return row


def _append_grouped_solar_metrics(
    rows: list[dict],
    data: pd.DataFrame,
    slice_name: str,
    group_col: str,
    mask: Optional[pd.Series] = None,
) -> None:
    """
    Append solar metrics for every value of a grouping column.
    """
    if group_col not in data.columns:
        return
    grouped_data = data if mask is None else data[mask].copy()
    if grouped_data.empty:
        return
    for value, group in grouped_data.groupby(group_col, dropna=False, observed=False):
        rows.append(_solar_metric_row(group, f"{slice_name}:{value}", group_col, value))


def calculate_solar_backtest_diagnostic_metrics(
    backtest_df: pd.DataFrame,
    daylight_threshold_mw: float = DEFAULT_SOLAR_BACKTEST_DAYLIGHT_THRESHOLD_MW,
) -> pd.DataFrame:
    """
    Build slice-level diagnostics for the solar hourly backtest.
    """
    rows: list[dict] = []
    if backtest_df.empty:
        return pd.DataFrame([_solar_metric_row(backtest_df, "Overall", "all", "all")])

    work = backtest_df.copy()
    work["IntervalStartDT"] = pd.to_datetime(work["IntervalStartDT"])
    for column in [
        "Actual_MW",
        "Forecast_MW",
        "BaseForecast_MW",
        "Actual_kWh",
        "Forecast_kWh",
        "BaseForecast_kWh",
        "CloudCoverPct",
        "ClearSkyIndex",
        "GHI_kWh_per_m2",
    ]:
        if column not in work.columns:
            work[column] = np.nan
        work[column] = pd.to_numeric(work[column], errors="coerce")

    excluded_mask = _actual_quality_exclusion_mask(work)
    raw_work = work.copy()
    if bool(excluded_mask.any()):
        rows.append(_solar_metric_row(raw_work, "RawOverall", "all", "all"))
        rows.append(
            _solar_metric_row(
                raw_work[excluded_mask],
                "ActualQualityExcluded",
                "ActualQualityFlag",
                ACTUAL_QUALITY_AMI_SUPPRESSED,
            )
        )
        work = raw_work[~excluded_mask].copy()
    if work.empty:
        rows.append(_solar_metric_row(work, "Overall", "all", "all"))
        return pd.DataFrame(rows)

    work["Hour"] = work["IntervalStartDT"].dt.hour
    work["Month"] = work["IntervalStartDT"].dt.to_period("M").astype(str)
    daylight_mask = work["Actual_MW"].fillna(0.0) > daylight_threshold_mw
    active_mask = daylight_mask | (
        work["Forecast_MW"].fillna(0.0) > daylight_threshold_mw
    )
    peak_hour_mask = daylight_mask & work["Hour"].between(11, 15, inclusive="both")
    actual_peak = (
        work.loc[daylight_mask, "Actual_MW"].max() if daylight_mask.any() else np.nan
    )
    high_output_mask = (
        daylight_mask & (work["Actual_MW"] >= actual_peak * 0.75)
        if pd.notna(actual_peak) and actual_peak > 0
        else pd.Series(False, index=work.index)
    )

    rows.append(_solar_metric_row(work, "Overall", "all", "all"))
    rows.append(
        _solar_metric_row(
            work[daylight_mask],
            "DaylightActual",
            "Actual_MW",
            f">{daylight_threshold_mw:g}",
        )
    )
    rows.append(
        _solar_metric_row(
            work[active_mask],
            "ActiveSolar",
            "ActualOrForecast_MW",
            f">{daylight_threshold_mw:g}",
        )
    )
    rows.append(
        _solar_metric_row(work[peak_hour_mask], "PeakSolarHours11to15", "Hour", "11-15")
    )
    rows.append(
        _solar_metric_row(
            work[high_output_mask],
            "HighOutputActualGE75PctPeak",
            "ActualPeakShare",
            ">=0.75",
        )
    )

    work["CloudCoverBucket"] = pd.cut(
        work["CloudCoverPct"],
        bins=[-np.inf, 20.0, 50.0, 80.0, np.inf],
        labels=["0-20", "20-50", "50-80", "80-100"],
    )
    work["ClearSkyIndexBucket"] = pd.cut(
        work["ClearSkyIndex"],
        bins=[-np.inf, 0.25, 0.50, 0.75, 1.00, np.inf],
        labels=["0-0.25", "0.25-0.50", "0.50-0.75", "0.75-1.00", ">1.00"],
    )
    work["GHIBucket"] = pd.cut(
        work["GHI_kWh_per_m2"],
        bins=[-np.inf, 0.10, 0.30, 0.50, 0.70, np.inf],
        labels=["0-0.10", "0.10-0.30", "0.30-0.50", "0.50-0.70", ">0.70"],
    )

    _append_grouped_solar_metrics(rows, work, "Month", "Month")
    _append_grouped_solar_metrics(rows, work, "Hour", "Hour", daylight_mask)
    _append_grouped_solar_metrics(
        rows, work, "CloudCover", "CloudCoverBucket", active_mask
    )
    _append_grouped_solar_metrics(
        rows, work, "ClearSkyIndex", "ClearSkyIndexBucket", active_mask
    )
    _append_grouped_solar_metrics(rows, work, "GHI", "GHIBucket", active_mask)
    return pd.DataFrame(rows)


def build_solar_backtest_top_errors(
    backtest_df: pd.DataFrame,
    top_n: int = DEFAULT_SOLAR_BACKTEST_TOP_ERROR_COUNT,
    daylight_threshold_mw: float = DEFAULT_SOLAR_BACKTEST_DAYLIGHT_THRESHOLD_MW,
) -> pd.DataFrame:
    """
    Return ranked underforecast and overforecast hours for solar backtest review.
    """
    if backtest_df.empty or top_n <= 0:
        return pd.DataFrame()

    work = backtest_df.copy()
    work["IntervalStartDT"] = pd.to_datetime(work["IntervalStartDT"])
    for column in [
        "Actual_MW",
        "Forecast_MW",
        "BaseForecast_MW",
        "Error_MW",
        "AbsError_MW",
    ]:
        if column not in work.columns:
            work[column] = np.nan
        work[column] = pd.to_numeric(work[column], errors="coerce")
    work["Error_MW"] = work["Error_MW"].fillna(work["Forecast_MW"] - work["Actual_MW"])
    work["AbsError_MW"] = work["AbsError_MW"].fillna(work["Error_MW"].abs())
    work["Underforecast_MW"] = work["Actual_MW"] - work["Forecast_MW"]
    work["Overforecast_MW"] = work["Forecast_MW"] - work["Actual_MW"]
    active_mask = work["Actual_MW"].fillna(0.0).gt(daylight_threshold_mw) | work[
        "Forecast_MW"
    ].fillna(0.0).gt(daylight_threshold_mw)
    work = work[active_mask].copy()
    if work.empty:
        return pd.DataFrame()

    review_columns = [
        "IntervalStartDT",
        "HE",
        "Actual_MW",
        "Forecast_MW",
        "BaseForecast_MW",
        "ActualReadCount",
        "Error_MW",
        "AbsError_MW",
        "Underforecast_MW",
        "Overforecast_MW",
        "WeatherGHI_Wm2",
        "GHI_kWh_per_m2",
        "ClearSkyGHI_Wm2",
        "ClearSkyIndex",
        "CloudCoverPct",
        "CloudCoverLowPct",
        "CloudCoverMidPct",
        "CloudCoverHighPct",
        "PerformanceRatio",
        "ResidualCalibrationFactor",
        "SeasonalCalibrationFactor",
        "RegimeCalibrationFactor",
        "PeakCalibrationFactor",
        "TotalCalibrationFactor",
        "ActualQualityFlag",
        "SolarBacktestExcluded",
        "ActualToExpectedRatio",
        "ActualQualityExpected_kWh",
        "ActualQualitySuspiciousHour",
        "ActualReadCoverageRatio",
        "ForecastSource",
    ]
    for column in review_columns:
        if column not in work.columns:
            work[column] = np.nan

    under = work.sort_values("Underforecast_MW", ascending=False).head(top_n).copy()
    under.insert(0, "ErrorType", "Underforecast")
    under.insert(1, "Rank", range(1, len(under) + 1))
    over = work.sort_values("Overforecast_MW", ascending=False).head(top_n).copy()
    over.insert(0, "ErrorType", "Overforecast")
    over.insert(1, "Rank", range(1, len(over) + 1))
    return pd.concat([under, over], ignore_index=True)[
        ["ErrorType", "Rank", *review_columns]
    ]


def build_solar_temporal_holdout_backtest(
    rec_interval_df: pd.DataFrame,
    weather_df: pd.DataFrame,
    capacity_kw: float,
    fallback_ratio: float,
    latitude: float,
    longitude: float,
    timezone_name: str,
    min_solar_elevation: float,
    daily_active_capacity: Optional[pd.DataFrame],
    max_performance_ratio: float,
    use_performance_model_energy_weighting: bool,
    intrahour_shape_method: str,
    shape_quantile: float,
    peak_hourly_kwh_quantile: float,
    holdout_days: int,
    residual_calibration_enabled: bool,
    residual_lower_bound: float,
    residual_upper_bound: float,
    residual_min_forecast_kwh: float,
    residual_min_training_rows: int,
    residual_energy_weighting: bool,
    seasonal_calibration_enabled: bool,
    seasonal_prior_mwh: float,
    seasonal_lower_bound: float,
    seasonal_upper_bound: float,
    actual_quality_filter_enabled: bool = True,
    daylight_threshold_mw: float = DEFAULT_SOLAR_BACKTEST_DAYLIGHT_THRESHOLD_MW,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """
    Train solar models before the holdout window and score the final period.
    """
    status_columns = [
        "Evaluation",
        "Status",
        "Reason",
        "HoldoutDays",
        "HoldoutStart",
        "TrainRows",
        "HoldoutRows",
    ]

    def status_frame(
        status: str, reason: str, holdout_start: object = pd.NaT
    ) -> pd.DataFrame:
        return pd.DataFrame(
            [
                {
                    "Evaluation": "temporal_holdout",
                    "Status": status,
                    "Reason": reason,
                    "HoldoutDays": holdout_days,
                    "HoldoutStart": holdout_start,
                    "TrainRows": 0,
                    "HoldoutRows": 0,
                }
            ],
            columns=status_columns,
        )

    if holdout_days <= 0:
        return status_frame("skipped", "holdout_days <= 0"), pd.DataFrame()
    if rec_interval_df.empty or weather_df.empty:
        return (
            status_frame("skipped", "missing REC intervals or weather rows"),
            pd.DataFrame(),
        )

    rec = rec_interval_df.copy()
    weather = weather_df.copy()
    rec["IntervalStartDT"] = pd.to_datetime(rec["IntervalStartDT"])
    weather["IntervalStartDT"] = pd.to_datetime(weather["IntervalStartDT"])
    max_timestamp = rec["IntervalStartDT"].max()
    if pd.isna(max_timestamp):
        return (
            status_frame("skipped", "REC intervals have no valid timestamps"),
            pd.DataFrame(),
        )

    holdout_start = max_timestamp.normalize() - pd.Timedelta(days=holdout_days - 1)
    train_rec = rec[rec["IntervalStartDT"] < holdout_start].copy()
    holdout_rec = rec[rec["IntervalStartDT"] >= holdout_start].copy()
    train_weather = weather[weather["IntervalStartDT"] < holdout_start].copy()
    holdout_weather = weather[weather["IntervalStartDT"] >= holdout_start].copy()
    if (
        train_rec.empty
        or holdout_rec.empty
        or train_weather.empty
        or holdout_weather.empty
    ):
        return (
            status_frame(
                "skipped", "insufficient train or holdout rows", holdout_start
            ),
            pd.DataFrame(),
        )

    holdout_intrahour_shape = build_intrahour_production_shape(
        train_rec,
        "Export_kWh",
        method=intrahour_shape_method,
        quantile=shape_quantile,
    )
    holdout_model = train_performance_model(
        rec_intervals=train_rec,
        weather_df=train_weather,
        capacity_kw=capacity_kw,
        fallback_ratio=fallback_ratio,
        performance_ratio_upper_bound=max_performance_ratio,
        latitude=latitude,
        longitude=longitude,
        timezone_name=timezone_name,
        daily_active_capacity=daily_active_capacity,
        use_energy_weighting=use_performance_model_energy_weighting,
        exclude_suppressed_actuals=actual_quality_filter_enabled,
    )
    holdout_interval_forecast = build_interval_forecast(
        weather_df=holdout_weather,
        intrahour_shape=holdout_intrahour_shape,
        capacity_kw=capacity_kw,
        model=holdout_model,
        latitude=latitude,
        longitude=longitude,
        timezone_name=timezone_name,
        min_solar_elevation=min_solar_elevation,
        forecast_source="holdout",
        daily_active_capacity=daily_active_capacity,
        peak_hourly_kwh_quantile=peak_hourly_kwh_quantile,
    )
    holdout_hourly = build_hourly_backtest(
        rec_interval_df=holdout_rec,
        interval_backtest_forecast=holdout_interval_forecast,
        total_capacity_kw=capacity_kw,
    )
    if actual_quality_filter_enabled and not holdout_hourly.empty:
        holdout_hourly["ActualQualityExpected_kWh"] = holdout_hourly[
            ["Forecast_kWh", "BaseForecast_kWh"]
        ].max(axis=1)
        holdout_hourly = add_solar_actual_quality_flags(
            holdout_hourly,
            actual_kwh_col="Actual_kWh",
            expected_kwh_col="ActualQualityExpected_kWh",
            actual_to_expected_ratio_threshold=DEFAULT_ACTUAL_QUALITY_FORECAST_RATIO_THRESHOLD,
            actual_read_count_col="ActualReadCount",
        )

    scorecard = calculate_solar_backtest_diagnostic_metrics(
        holdout_hourly,
        daylight_threshold_mw=daylight_threshold_mw,
    )
    scorecard.insert(0, "Evaluation", "temporal_holdout")
    scorecard.insert(1, "Status", "completed")
    scorecard.insert(2, "Reason", "")
    scorecard.insert(3, "HoldoutDays", holdout_days)
    scorecard.insert(4, "HoldoutStart", holdout_start)
    scorecard.insert(5, "TrainRows", len(train_rec))
    scorecard.insert(6, "HoldoutRows", len(holdout_rec))
    return scorecard, holdout_hourly


# =============================================================================
# Main Forecaster
# =============================================================================


def run_forecaster(args: argparse.Namespace) -> None:
    """
    Main function to run the solar forecasting process.
    """
    if not (0 < args.performance_ratio <= args.performance_ratio_upper_bound):
        raise ValueError(
            "--performance-ratio must be greater than 0 and less than or equal to the ratio upper bound."
        )
    if not (1.0 <= args.performance_ratio_upper_bound <= 1.5):
        raise ValueError("--performance-ratio-upper-bound must be between 1.0 and 1.5.")
    if not (0 <= args.array_tilt_degrees <= 90):
        raise ValueError("--array-tilt-degrees must be between 0 and 90.")
    if not (0 <= args.array_azimuth_degrees <= 360):
        raise ValueError("--array-azimuth-degrees must be between 0 and 360.")
    if args.rec_history_months <= 0:
        raise ValueError("--rec-history-months must be greater than 0.")
    if not (0 < args.forecast_days <= 16):
        raise ValueError("--forecast-days must be between 1 and 16.")
    if not (0 <= args.historical_days <= 365):
        raise ValueError("--historical-days must be between 0 and 365.")
    if bool(args.rec_history_start) != bool(args.rec_history_end):
        raise ValueError(
            "--rec-history-start and --rec-history-end must be provided together."
        )
    if args.rec_history_start and args.rec_history_start > args.rec_history_end:
        raise ValueError(
            "--rec-history-start must be earlier than or equal to --rec-history-end."
        )
    if not (-10 <= args.min_solar_elevation <= 20):
        raise ValueError("--min-solar-elevation must be between -10 and 20 degrees.")
    if args.weather_clusters < 0:
        raise ValueError("--weather-clusters must be greater than or equal to 0.")
    if args.weather_locations_per_request <= 0:
        raise ValueError("--weather-locations-per-request must be greater than 0.")
    if args.max_weather_api_calls <= 0:
        raise ValueError("--max-weather-api-calls must be greater than 0.")
    if args.same_day_correction_min_intervals <= 0:
        raise ValueError("--same-day-correction-min-intervals must be greater than 0.")
    if args.same_day_correction_min_forecast_kwh < 0:
        raise ValueError(
            "--same-day-correction-min-forecast-kwh must be greater than or equal to 0."
        )
    if not (
        0 < args.same_day_correction_lower_bound <= args.same_day_correction_upper_bound
    ):
        raise ValueError(
            "--same-day-correction-lower-bound must be greater than 0 and less than or equal to "
            "--same-day-correction-upper-bound."
        )
    if args.same_day_correction_half_life_hours <= 0:
        raise ValueError(
            "--same-day-correction-half-life-hours must be greater than 0."
        )
    if not (0 <= args.same_day_correction_weather_similarity_floor <= 1):
        raise ValueError(
            "--same-day-correction-weather-similarity-floor must be between 0 and 1."
        )
    if args.daily_shape_method not in {"mean", "median", "upper-quantile"}:
        raise ValueError(
            "--daily-shape-method must be one of: mean, median, upper-quantile."
        )
    if args.intrahour_shape_method not in {"mean", "median", "upper-quantile"}:
        raise ValueError(
            "--intrahour-shape-method must be one of: mean, median, upper-quantile."
        )
    if not (0 < args.shape_quantile < 1):
        raise ValueError("--shape-quantile must be greater than 0 and less than 1.")
    if args.residual_calibration_min_rows <= 0:
        raise ValueError("--residual-calibration-min-rows must be greater than 0.")
    if args.residual_calibration_min_validation_rows <= 0:
        raise ValueError(
            "--residual-calibration-min-validation-rows must be greater than 0."
        )
    if not (0 <= args.residual_calibration_validation_fraction < 0.5):
        raise ValueError(
            "--residual-calibration-validation-fraction must be at least 0 and less than 0.5."
        )
    if args.residual_calibration_min_forecast_kwh < 0:
        raise ValueError(
            "--residual-calibration-min-forecast-kwh must be greater than or equal to 0."
        )
    if args.residual_calibration_energy_weight_power < 0:
        raise ValueError(
            "--residual-calibration-energy-weight-power must be greater than or equal to 0."
        )
    customer_segments = parse_customer_segments(args.customer_segments)
    if args.segment_forecasts and not customer_segments:
        raise ValueError(
            "--customer-segments must include at least one segment when --segment-forecasts is enabled."
        )
    if not (
        0
        < args.residual_calibration_lower_bound
        <= args.residual_calibration_upper_bound
    ):
        raise ValueError(
            "--residual-calibration-lower-bound must be greater than 0 and less than or equal to "
            "--residual-calibration-upper-bound."
        )
    if args.seasonal_calibration_prior_mwh < 0:
        raise ValueError(
            "--seasonal-calibration-prior-mwh must be greater than or equal to 0."
        )
    if not (
        0
        < args.seasonal_calibration_lower_bound
        <= args.seasonal_calibration_upper_bound
    ):
        raise ValueError(
            "--seasonal-calibration-lower-bound must be greater than 0 and less than or equal to "
            "--seasonal-calibration-upper-bound."
        )
    if not (0 <= args.calibration_min_daylight_row_coverage <= 1):
        raise ValueError(
            "--calibration-min-daylight-row-coverage must be between 0 and 1."
        )
    if args.calibration_min_day_actual_forecast_ratio < 0:
        raise ValueError(
            "--calibration-min-day-actual-forecast-ratio must be greater than or equal to 0."
        )
    if args.calibration_min_day_forecast_mwh < 0:
        raise ValueError(
            "--calibration-min-day-forecast-mwh must be greater than or equal to 0."
        )
    if args.regime_calibration_min_rows <= 0:
        raise ValueError("--regime-calibration-min-rows must be greater than 0.")
    if args.regime_calibration_min_forecast_mwh < 0:
        raise ValueError(
            "--regime-calibration-min-forecast-mwh must be greater than or equal to 0."
        )
    if args.regime_calibration_prior_mwh < 0:
        raise ValueError(
            "--regime-calibration-prior-mwh must be greater than or equal to 0."
        )
    if not (
        0 < args.regime_calibration_lower_bound <= args.regime_calibration_upper_bound
    ):
        raise ValueError(
            "--regime-calibration-lower-bound must be greater than 0 and less than or equal to "
            "--regime-calibration-upper-bound."
        )
    if not (0 < args.peak_calibration_quantile < 1):
        raise ValueError(
            "--peak-calibration-quantile must be greater than 0 and less than 1."
        )
    if args.peak_calibration_min_rows <= 0:
        raise ValueError("--peak-calibration-min-rows must be greater than 0.")
    if args.peak_calibration_min_forecast_mwh < 0:
        raise ValueError(
            "--peak-calibration-min-forecast-mwh must be greater than or equal to 0."
        )
    if not (0 < args.peak_calibration_lower_bound <= 1):
        raise ValueError(
            "--peak-calibration-lower-bound must be greater than 0 and less than or equal to 1."
        )
    if args.peak_calibration_target_capture <= 0:
        raise ValueError("--peak-calibration-target-capture must be greater than 0.")
    ZoneInfo(args.timezone)

    engine: Optional[Engine] = None
    try:
        parquet_root = Path(args.parquet_root)
        residual_calibration_model = identity_residual_calibration_model(
            lower_bound=args.residual_calibration_lower_bound,
            upper_bound=args.residual_calibration_upper_bound,
        )
        engine = connect(
            driver=args.driver,
            server=args.dest_server,
            database=args.dest_db,
            username=args.dest_user,
            password=args.dest_pass,
        )

        sites: Optional[pd.DataFrame] = None
        preloaded_export_intervals: Optional[pd.DataFrame] = None
        rec_segment_interval_df = pd.DataFrame()
        calibration_weather_source = pd.DataFrame()
        inferred_weather_source = pd.DataFrame()
        weather_source = pd.DataFrame()
        inferred_start_timestamp: Optional[pd.Timestamp] = None
        inferred_end_timestamp: Optional[pd.Timestamp] = None

        if args.production_source == "rec-parquet":
            sites = load_forecast_eligible_solar_sites(engine)
            total_capacity_kw = float(sites["SolarCECkW"].sum())

            if args.rec_history_start and args.rec_history_end:
                rec_start_date = args.rec_history_start
                rec_end_date = args.rec_history_end
            else:
                rec_start_date, rec_end_date = get_default_rec_history_window(
                    parquet_root,
                    args.rec_history_months,
                )

            rec_segment_interval_df = load_rec_interval_data(
                parquet_root=parquet_root,
                sites=sites,
                start_date=rec_start_date,
                end_date=rec_end_date,
                net_meter_export_source=args.net_meter_export_source,
                latitude=args.latitude,
                longitude=args.longitude,
                timezone_name=args.timezone,
                min_solar_elevation=args.min_solar_elevation,
                group_by_customer_segment=True,
            )
            rec_interval_df = aggregate_segment_export_to_total(rec_segment_interval_df)
            rec_interval_df = add_hour_ending_column(rec_interval_df)
            rec_segment_interval_df = add_hour_ending_column(rec_segment_interval_df)
            preloaded_export_intervals = rec_interval_df
            rec_interval_df[
                [
                    "IntervalStartDT",
                    "HE",
                    "Export_kWh",
                    "Export_kW",
                    "ExportSource",
                    "SolarElevationDeg",
                ]
            ].to_csv(args.rec_actual_15min_output, index=False)
            if args.segment_forecasts:
                rec_segment_interval_df[
                    [
                        "CustomerSegment",
                        "IntervalStartDT",
                        "HE",
                        "Export_kWh",
                        "Export_kW",
                        "ExportSource",
                        "SolarElevationDeg",
                    ]
                ].to_csv(args.segment_rec_actual_15min_output, index=False)

            rec_hourly = resample_actual_export_to_hourly(rec_interval_df)
            rec_hourly[
                ["IntervalStartDT", "HE", "Export_MW", "Export_kWh", "Export_kW"]
            ].to_csv(
                args.rec_actual_hourly_output,
                index=False,
            )
            if args.segment_forecasts:
                rec_segment_hourly = resample_actual_export_to_hourly(
                    rec_segment_interval_df
                )
                rec_segment_hourly[
                    [
                        "CustomerSegment",
                        "IntervalStartDT",
                        "HE",
                        "Export_MW",
                        "Export_kWh",
                        "Export_kW",
                    ]
                ].to_csv(args.segment_rec_actual_hourly_output, index=False)

            intrahour_shape = build_intrahour_production_shape(
                rec_interval_df,
                "Export_kWh",
                method=args.intrahour_shape_method,
                quantile=args.shape_quantile,
            )
            average_daily_shape = build_average_daily_shape(
                rec_interval_df,
                "Export_kW",
                method=args.daily_shape_method,
                quantile=args.shape_quantile,
            )
            average_daily_shape.to_csv(args.load_shape_output, index=False)
            logging.info(
                "Built solar export shapes using daily method %s and intra-hour method %s",
                args.daily_shape_method,
                args.intrahour_shape_method,
            )

            if args.use_capacity_weighted_weather:
                sites_with_coords = sites.dropna(subset=["Latitude", "Longitude"])
                if sites_with_coords.empty:
                    logging.warning(
                        "Capacity-weighted solar weather requested, but no forecast-eligible sites have coordinates. "
                        "Falling back to representative Roseville weather."
                    )
                    args.use_capacity_weighted_weather = False
                    weather_sites_df = build_system_weather_site(
                        args.latitude, args.longitude
                    )
                elif args.weather_clusters > 0:
                    weather_location_budget = (
                        args.max_weather_api_calls * args.weather_locations_per_request
                    )
                    requested_weather_samples = min(
                        args.weather_clusters, len(sites_with_coords)
                    )
                    weather_sample_count = min(
                        requested_weather_samples, weather_location_budget
                    )
                    if requested_weather_samples > weather_location_budget:
                        logging.warning(
                            "Requested %s representative weather locations, but the Open-Meteo call budget "
                            "allows %s locations (%s calls x %s locations/request).",
                            requested_weather_samples,
                            weather_location_budget,
                            args.max_weather_api_calls,
                            args.weather_locations_per_request,
                        )
                    logging.info(
                        "Sampling %s representative solar sites from all %s forecast-eligible sites with coordinates; "
                        "expected Open-Meteo calls per weather fetch: %s",
                        weather_sample_count,
                        len(sites_with_coords),
                        int(
                            math.ceil(
                                weather_sample_count
                                / args.weather_locations_per_request
                            )
                        ),
                    )
                    sites, weather_sites_df = build_weather_clusters(
                        sites, n_clusters=weather_sample_count
                    )
                else:
                    logging.info(
                        "Weather site sampling disabled; using single representative Roseville weather."
                    )
                    args.use_capacity_weighted_weather = False
                    weather_sites_df = build_system_weather_site(
                        args.latitude, args.longitude
                    )
            else:
                logging.info(
                    "Using single-point weather forecast for representative lat/lon."
                )
                weather_sites_df = build_system_weather_site(
                    args.latitude, args.longitude
                )

            calibration_weather_source = fetch_open_meteo_hourly_weather(
                weather_sites_df,
                rec_start_date,
                rec_end_date,
                use_forecast=False,
                timezone_name=args.timezone,
                array_tilt_degrees=args.array_tilt_degrees,
                array_azimuth_degrees=args.array_azimuth_degrees,
                weather_locations_per_request=args.weather_locations_per_request,
                cache_dir=args.weather_cache_dir,
            )
            calibration_weather = weather_for_sites(
                calibration_weather_source,
                sites,
                args.use_capacity_weighted_weather,
            )

            model = train_performance_model(
                rec_intervals=rec_interval_df,
                weather_df=calibration_weather,
                capacity_kw=total_capacity_kw,
                fallback_ratio=args.performance_ratio,
                performance_ratio_upper_bound=args.performance_ratio_upper_bound,
                sites=sites,
                latitude=args.latitude,
                longitude=args.longitude,
                timezone_name=args.timezone,
                array_tilt_degrees=args.array_tilt_degrees,
                array_azimuth_degrees=args.array_azimuth_degrees,
            )
            calibration_interval_forecast = pd.DataFrame()
            calibration_backtest_hourly = pd.DataFrame()
            if args.backtest or args.residual_calibration:
                logging.info(
                    "Building historical base forecast for residual calibration/backtest"
                )
                calibration_interval_forecast = build_interval_forecast(
                    weather_df=calibration_weather,
                    intrahour_shape=intrahour_shape,
                    capacity_kw=total_capacity_kw,
                    model=model,
                    sites=sites,
                    latitude=args.latitude,
                    longitude=args.longitude,
                    timezone_name=args.timezone,
                    min_solar_elevation=args.min_solar_elevation,
                    array_tilt_degrees=args.array_tilt_degrees,
                    array_azimuth_degrees=args.array_azimuth_degrees,
                    forecast_source="backtest",
                )
                calibration_interval_forecast = add_hour_ending_column(
                    calibration_interval_forecast
                )
                calibration_backtest_hourly = build_hourly_backtest(
                    rec_interval_df=rec_interval_df,
                    interval_backtest_forecast=calibration_interval_forecast,
                    total_capacity_kw=total_capacity_kw,
                )

            if args.residual_calibration:
                residual_calibration_model = train_residual_calibration_model(
                    backtest_df=calibration_backtest_hourly,
                    total_capacity_kw=total_capacity_kw,
                    latitude=args.latitude,
                    longitude=args.longitude,
                    timezone_name=args.timezone,
                    array_tilt_degrees=args.array_tilt_degrees,
                    array_azimuth_degrees=args.array_azimuth_degrees,
                    performance_ratio_upper_bound=args.performance_ratio_upper_bound,
                    lower_bound=args.residual_calibration_lower_bound,
                    upper_bound=args.residual_calibration_upper_bound,
                    min_training_forecast_kwh=args.residual_calibration_min_forecast_kwh,
                    min_training_rows=args.residual_calibration_min_rows,
                    validation_fraction=args.residual_calibration_validation_fraction,
                    min_validation_rows=args.residual_calibration_min_validation_rows,
                    use_energy_weighting=args.residual_calibration_energy_weighting,
                    energy_weight_power=args.residual_calibration_energy_weight_power,
                    use_seasonal_calibration=args.seasonal_calibration,
                    seasonal_prior_mwh=args.seasonal_calibration_prior_mwh,
                    seasonal_lower_bound=args.seasonal_calibration_lower_bound,
                    seasonal_upper_bound=args.seasonal_calibration_upper_bound,
                    use_quality_filter=args.calibration_quality_filter,
                    quality_min_row_coverage=args.calibration_min_daylight_row_coverage,
                    quality_min_actual_forecast_ratio=args.calibration_min_day_actual_forecast_ratio,
                    quality_min_day_forecast_mwh=args.calibration_min_day_forecast_mwh,
                    use_regime_calibration=args.regime_calibration,
                    regime_min_rows=args.regime_calibration_min_rows,
                    regime_min_forecast_mwh=args.regime_calibration_min_forecast_mwh,
                    regime_prior_mwh=args.regime_calibration_prior_mwh,
                    regime_lower_bound=args.regime_calibration_lower_bound,
                    regime_upper_bound=args.regime_calibration_upper_bound,
                    use_peak_calibration=args.peak_calibration,
                    peak_quantile=args.peak_calibration_quantile,
                    peak_min_rows=args.peak_calibration_min_rows,
                    peak_min_forecast_mwh=args.peak_calibration_min_forecast_mwh,
                    peak_lower_bound=args.peak_calibration_lower_bound,
                    peak_target_capture=args.peak_calibration_target_capture,
                )
            else:
                logging.info("Residual calibration disabled.")

            if args.backtest:
                logging.info("Running hourly backtest against REC/NET actual export")
                if args.residual_calibration:
                    interval_backtest_forecast = apply_residual_calibration(
                        interval_forecast=calibration_interval_forecast,
                        calibration_model=residual_calibration_model,
                        total_capacity_kw=total_capacity_kw,
                        latitude=args.latitude,
                        longitude=args.longitude,
                        timezone_name=args.timezone,
                        array_tilt_degrees=args.array_tilt_degrees,
                        array_azimuth_degrees=args.array_azimuth_degrees,
                        performance_ratio_upper_bound=args.performance_ratio_upper_bound,
                    )
                    interval_backtest_forecast["ForecastSource"] = "backtest_calibrated"
                else:
                    interval_backtest_forecast = calibration_interval_forecast

                backtest_hourly = build_hourly_backtest(
                    rec_interval_df=rec_interval_df,
                    interval_backtest_forecast=interval_backtest_forecast,
                    total_capacity_kw=total_capacity_kw,
                )
                if args.residual_calibration and args.peak_calibration:
                    peak_tuned = tune_peak_calibration_from_hourly_backtest(
                        calibration_model=residual_calibration_model,
                        hourly_backtest=backtest_hourly,
                        total_capacity_kw=total_capacity_kw,
                        quantile=args.peak_calibration_quantile,
                        min_rows=args.peak_calibration_min_rows,
                        min_forecast_mwh=args.peak_calibration_min_forecast_mwh,
                        lower_bound=args.peak_calibration_lower_bound,
                        target_capture=args.peak_calibration_target_capture,
                        label="total",
                    )
                    if peak_tuned:
                        interval_backtest_forecast = apply_residual_calibration(
                            interval_forecast=calibration_interval_forecast,
                            calibration_model=residual_calibration_model,
                            total_capacity_kw=total_capacity_kw,
                            latitude=args.latitude,
                            longitude=args.longitude,
                            timezone_name=args.timezone,
                            array_tilt_degrees=args.array_tilt_degrees,
                            array_azimuth_degrees=args.array_azimuth_degrees,
                            performance_ratio_upper_bound=args.performance_ratio_upper_bound,
                        )
                        interval_backtest_forecast["ForecastSource"] = (
                            "backtest_calibrated"
                        )
                        backtest_hourly = build_hourly_backtest(
                            rec_interval_df=rec_interval_df,
                            interval_backtest_forecast=interval_backtest_forecast,
                            total_capacity_kw=total_capacity_kw,
                        )
                backtest_summary = calculate_backtest_summary(backtest_hourly)
                backtest_hourly[
                    [
                        "IntervalStartDT",
                        "HE",
                        "Actual_MW",
                        "Actual_kWh",
                        "BaseForecast_MW",
                        "BaseForecast_kWh",
                        "Forecast_MW",
                        "Forecast_kWh",
                        "BaseError_MW",
                        "BaseError_kWh",
                        "Error_MW",
                        "Error_kWh",
                        "BaseAbsError_MW",
                        "BaseAbsError_kWh",
                        "AbsError_MW",
                        "AbsError_kWh",
                        "BaseAPE",
                        "APE",
                        "BaseCapacityFactor",
                        "CapacityFactor",
                        "ActiveCapacity_kW",
                        *FORECAST_DIAGNOSTIC_COLUMNS,
                        "PerformanceRatio",
                        "ResidualCalibrationFactor",
                        "SeasonalCalibrationFactor",
                        "RegimeCalibrationFactor",
                        "PeakCalibrationFactor",
                        "TotalCalibrationFactor",
                        "SameDayCorrectionFactor",
                        "BacktestForecast",
                        "ForecastSource",
                    ]
                ].to_csv(args.backtest_hourly_output, index=False)
                backtest_summary.to_csv(args.backtest_summary_output, index=False)
                diagnostics_output = getattr(
                    args, "solar_backtest_diagnostics_output", None
                )
                if diagnostics_output:
                    diagnostics = calculate_solar_backtest_diagnostic_metrics(
                        backtest_hourly,
                        daylight_threshold_mw=getattr(
                            args,
                            "solar_backtest_daylight_threshold_mw",
                            DEFAULT_SOLAR_BACKTEST_DAYLIGHT_THRESHOLD_MW,
                        ),
                    )
                    Path(diagnostics_output).parent.mkdir(parents=True, exist_ok=True)
                    diagnostics.to_csv(diagnostics_output, index=False)
                top_errors_output = getattr(
                    args, "solar_backtest_top_errors_output", None
                )
                if top_errors_output:
                    top_errors = build_solar_backtest_top_errors(
                        backtest_hourly,
                        top_n=getattr(
                            args,
                            "solar_backtest_top_error_count",
                            DEFAULT_SOLAR_BACKTEST_TOP_ERROR_COUNT,
                        ),
                        daylight_threshold_mw=getattr(
                            args,
                            "solar_backtest_daylight_threshold_mw",
                            DEFAULT_SOLAR_BACKTEST_DAYLIGHT_THRESHOLD_MW,
                        ),
                    )
                    Path(top_errors_output).parent.mkdir(parents=True, exist_ok=True)
                    top_errors.to_csv(top_errors_output, index=False)
                holdout_output = getattr(args, "solar_backtest_holdout_output", None)
                if holdout_output:
                    holdout_scorecard, holdout_hourly = (
                        build_solar_temporal_holdout_backtest(
                            rec_interval_df=rec_interval_df,
                            weather_df=calibration_weather_source,
                            capacity_kw=total_capacity_kw,
                            fallback_ratio=args.performance_ratio,
                            latitude=args.latitude,
                            longitude=args.longitude,
                            timezone_name=args.timezone,
                            min_solar_elevation=args.min_solar_elevation,
                            daily_active_capacity=None,
                            max_performance_ratio=args.performance_ratio_upper_bound,
                            use_performance_model_energy_weighting=True,
                            intrahour_shape_method=args.intrahour_shape_method,
                            shape_quantile=args.shape_quantile,
                            peak_hourly_kwh_quantile=DEFAULT_PEAK_HOURLY_KWH_QUANTILE,
                            holdout_days=getattr(
                                args,
                                "solar_backtest_holdout_days",
                                DEFAULT_SOLAR_BACKTEST_HOLDOUT_DAYS,
                            ),
                            residual_calibration_enabled=args.residual_calibration,
                            residual_lower_bound=args.residual_calibration_lower_bound,
                            residual_upper_bound=args.residual_calibration_upper_bound,
                            residual_min_forecast_kwh=args.residual_calibration_min_forecast_kwh,
                            residual_min_training_rows=args.residual_calibration_min_rows,
                            residual_energy_weighting=args.residual_calibration_energy_weighting,
                            seasonal_calibration_enabled=args.seasonal_calibration,
                            seasonal_prior_mwh=args.seasonal_calibration_prior_mwh,
                            seasonal_lower_bound=args.seasonal_calibration_lower_bound,
                            seasonal_upper_bound=args.seasonal_calibration_upper_bound,
                            daylight_threshold_mw=getattr(
                                args,
                                "solar_backtest_daylight_threshold_mw",
                                DEFAULT_SOLAR_BACKTEST_DAYLIGHT_THRESHOLD_MW,
                            ),
                        )
                    )
                    Path(holdout_output).parent.mkdir(parents=True, exist_ok=True)
                    holdout_scorecard.to_csv(holdout_output, index=False)
                    holdout_hourly_output = getattr(
                        args, "solar_backtest_holdout_hourly_output", None
                    )
                    if holdout_hourly_output and not holdout_hourly.empty:
                        Path(holdout_hourly_output).parent.mkdir(
                            parents=True, exist_ok=True
                        )
                        holdout_hourly.to_csv(holdout_hourly_output, index=False)
                summary_row = backtest_summary.iloc[0]
                logging.info(
                    "Backtest saved to %s; base WMAPE %.2f%% -> calibrated WMAPE %.2f%%, "
                    "bias %.2f MWh, RMSE %.2f MW",
                    args.backtest_hourly_output,
                    (
                        summary_row["BaseWMAPE"] * 100
                        if pd.notna(summary_row["BaseWMAPE"])
                        else float("nan")
                    ),
                    (
                        summary_row["WMAPE"] * 100
                        if pd.notna(summary_row["WMAPE"])
                        else float("nan")
                    ),
                    summary_row["Bias_MWh"],
                    summary_row["RMSE_MW"],
                )
            logging.info(
                "Using parquet export production shape from %s to %s",
                rec_start_date,
                rec_end_date,
            )

        else:
            total_capacity_kw = get_total_forecast_eligible_capacity(engine)
            prod_interval_df = load_production_interval_data(engine)
            if prod_interval_df.empty:
                raise ValueError(
                    "No historical production data found. Cannot create production shape."
                )

            prod_interval_df["IntervalStartDT"] = pd.to_datetime(
                prod_interval_df["IntervalStartDT"]
            )
            prod_interval_df["IntervalEnergy_kWh"] = (
                prod_interval_df["IntervalValue"] * INTERVAL_HOURS
            )
            intrahour_shape = build_intrahour_production_shape(
                prod_interval_df,
                "IntervalEnergy_kWh",
                method=args.intrahour_shape_method,
                quantile=args.shape_quantile,
            )
            average_daily_shape = build_average_daily_shape(
                prod_interval_df,
                "IntervalValue",
                method=args.daily_shape_method,
                quantile=args.shape_quantile,
            )
            average_daily_shape.to_csv(args.load_shape_output, index=False)
            model = PerformanceModel(
                None,
                args.performance_ratio,
                PERFORMANCE_FEATURE_COLUMNS,
                args.performance_ratio_upper_bound,
            )

            logging.info("Using representative DB production shape")
            weather_sites_df = build_system_weather_site(args.latitude, args.longitude)

        today = current_local_timestamp(args.timezone).date()
        forecast_start_date = today - timedelta(days=args.historical_days)
        forecast_end_date = today + timedelta(days=args.forecast_days - 1)

        inferred_interval_forecast = pd.DataFrame()
        if (
            args.infer_missing_history
            and preloaded_export_intervals is not None
            and not preloaded_export_intervals.empty
        ):
            last_actual_interval = pd.to_datetime(
                preloaded_export_intervals["IntervalStartDT"]
            ).max()
            inferred_start_timestamp = last_actual_interval.floor("h") + pd.Timedelta(
                hours=1
            )
            inferred_end_timestamp = pd.Timestamp(forecast_start_date) - pd.Timedelta(
                minutes=15
            )
            if inferred_start_timestamp <= inferred_end_timestamp:
                logging.info(
                    "Inferring missing historical solar generation from %s through %s",
                    inferred_start_timestamp,
                    inferred_end_timestamp,
                )
                inferred_weather_source = fetch_hourly_weather_for_date_range(
                    weather_sites_df,
                    inferred_start_timestamp.date(),
                    inferred_end_timestamp.date(),
                    timezone_name=args.timezone,
                    array_tilt_degrees=args.array_tilt_degrees,
                    array_azimuth_degrees=args.array_azimuth_degrees,
                    weather_locations_per_request=args.weather_locations_per_request,
                    cache_dir=args.weather_cache_dir,
                    reusable_weather_frames=[calibration_weather_source],
                )
                inferred_weather_df = weather_for_sites(
                    inferred_weather_source,
                    sites,
                    args.use_capacity_weighted_weather,
                )
                inferred_interval_forecast = build_interval_forecast(
                    weather_df=inferred_weather_df,
                    intrahour_shape=intrahour_shape,
                    capacity_kw=total_capacity_kw,
                    model=model,
                    sites=sites,
                    latitude=args.latitude,
                    longitude=args.longitude,
                    timezone_name=args.timezone,
                    min_solar_elevation=args.min_solar_elevation,
                    array_tilt_degrees=args.array_tilt_degrees,
                    array_azimuth_degrees=args.array_azimuth_degrees,
                    forecast_source="inferred_historical",
                )
                if args.residual_calibration:
                    inferred_interval_forecast = apply_residual_calibration(
                        interval_forecast=inferred_interval_forecast,
                        calibration_model=residual_calibration_model,
                        total_capacity_kw=total_capacity_kw,
                        latitude=args.latitude,
                        longitude=args.longitude,
                        timezone_name=args.timezone,
                        array_tilt_degrees=args.array_tilt_degrees,
                        array_azimuth_degrees=args.array_azimuth_degrees,
                        performance_ratio_upper_bound=args.performance_ratio_upper_bound,
                    )
                inferred_interval_forecast = inferred_interval_forecast[
                    (
                        inferred_interval_forecast["IntervalStartDT"]
                        >= inferred_start_timestamp
                    )
                    & (
                        inferred_interval_forecast["IntervalStartDT"]
                        <= inferred_end_timestamp
                    )
                ].copy()
                inferred_interval_forecast = add_hour_ending_column(
                    inferred_interval_forecast
                )
                logging.info(
                    "Built %s inferred historical 15-minute forecast rows",
                    len(inferred_interval_forecast),
                )

        weather_source = fetch_hourly_weather_for_date_range(
            weather_sites_df,
            forecast_start_date,
            forecast_end_date,
            timezone_name=args.timezone,
            array_tilt_degrees=args.array_tilt_degrees,
            array_azimuth_degrees=args.array_azimuth_degrees,
            weather_locations_per_request=args.weather_locations_per_request,
            cache_dir=args.weather_cache_dir,
            reusable_weather_frames=[
                calibration_weather_source,
                inferred_weather_source,
            ],
        )
        weather_df = weather_for_sites(
            weather_source, sites, args.use_capacity_weighted_weather
        )

        logging.info(
            "Running forecast for %s to %s with capacity %.2f kW",
            forecast_start_date,
            forecast_end_date,
            total_capacity_kw,
        )
        interval_forecast = build_interval_forecast(
            weather_df=weather_df,
            intrahour_shape=intrahour_shape,
            capacity_kw=total_capacity_kw,
            model=model,
            sites=sites,
            latitude=args.latitude,
            longitude=args.longitude,
            timezone_name=args.timezone,
            min_solar_elevation=args.min_solar_elevation,
            array_tilt_degrees=args.array_tilt_degrees,
            array_azimuth_degrees=args.array_azimuth_degrees,
            forecast_source="forecast",
        )
        interval_forecast["ForecastSource"] = np.where(
            interval_forecast["IntervalStartDT"].dt.date < today,
            "historical_forecast",
            "forecast",
        )
        if args.residual_calibration:
            interval_forecast = apply_residual_calibration(
                interval_forecast=interval_forecast,
                calibration_model=residual_calibration_model,
                total_capacity_kw=total_capacity_kw,
                latitude=args.latitude,
                longitude=args.longitude,
                timezone_name=args.timezone,
                array_tilt_degrees=args.array_tilt_degrees,
                array_azimuth_degrees=args.array_azimuth_degrees,
                performance_ratio_upper_bound=args.performance_ratio_upper_bound,
            )
        if args.same_day_correction:
            if sites is not None:
                same_day_actuals = load_same_day_export_actuals(
                    parquet_root=parquet_root,
                    sites=sites,
                    forecast_start_date=forecast_start_date,
                    forecast_end_date=forecast_end_date,
                    net_meter_export_source=args.net_meter_export_source,
                    latitude=args.latitude,
                    longitude=args.longitude,
                    timezone_name=args.timezone,
                    min_solar_elevation=args.min_solar_elevation,
                    preloaded_intervals=preloaded_export_intervals,
                )
                interval_forecast = apply_same_day_actual_correction(
                    interval_forecast=interval_forecast,
                    same_day_actuals=same_day_actuals,
                    timezone_name=args.timezone,
                    min_observed_intervals=args.same_day_correction_min_intervals,
                    min_observed_forecast_kwh=args.same_day_correction_min_forecast_kwh,
                    lower_bound=args.same_day_correction_lower_bound,
                    upper_bound=args.same_day_correction_upper_bound,
                    half_life_hours=args.same_day_correction_half_life_hours,
                    weather_similarity_floor=args.same_day_correction_weather_similarity_floor,
                )
            else:
                logging.info(
                    "Same-day correction skipped; REC/NET parquet actuals are not the production source"
                )
        interval_forecast = add_hour_ending_column(interval_forecast)
        if not inferred_interval_forecast.empty:
            interval_forecast = (
                pd.concat(
                    [inferred_interval_forecast, interval_forecast], ignore_index=True
                )
                .drop_duplicates(subset=["IntervalStartDT"], keep="last")
                .sort_values("IntervalStartDT")
            )

        interval_forecast[
            [
                "IntervalStartDT",
                "HE",
                "BaseForecast_kW",
                "BaseForecast_kWh",
                "Forecast_kW",
                "Forecast_kWh",
                "ActiveCapacity_kW",
                *FORECAST_DIAGNOSTIC_COLUMNS,
                "PerformanceRatio",
                "ResidualCalibrationFactor",
                "SeasonalCalibrationFactor",
                "RegimeCalibrationFactor",
                "PeakCalibrationFactor",
                "TotalCalibrationFactor",
                "SameDayCorrectionFactor",
                "ForecastSource",
            ]
        ].to_csv(
            args.output_15min,
            index=False,
        )

        hourly_forecast = resample_interval_forecast_to_hourly(
            interval_forecast, total_capacity_kw
        )
        hourly_forecast[
            [
                "IntervalStartDT",
                "HE",
                "BaseForecast_MW",
                "BaseForecast_kWh",
                "Forecast_MW",
                "Forecast_kWh",
                "BaseCapacityFactor",
                "CapacityFactor",
                "ActiveCapacity_kW",
                *FORECAST_DIAGNOSTIC_COLUMNS,
                "PerformanceRatio",
                "ResidualCalibrationFactor",
                "SeasonalCalibrationFactor",
                "RegimeCalibrationFactor",
                "PeakCalibrationFactor",
                "TotalCalibrationFactor",
                "SameDayCorrectionFactor",
                "ForecastSource",
            ]
        ].to_csv(
            args.output_hourly,
            index=False,
        )

        if (
            args.segment_forecasts
            and args.production_source == "rec-parquet"
            and sites is not None
            and not rec_segment_interval_df.empty
        ):
            logging.info(
                "Building separate customer-segment forecasts for %s",
                ", ".join(
                    CUSTOMER_SEGMENT_LABELS.get(segment, segment)
                    for segment in customer_segments
                ),
            )
            segment_interval_outputs = []
            segment_hourly_backtests = []
            segment_backtest_summaries = []
            segment_shape_outputs = []

            for segment in customer_segments:
                segment_label = CUSTOMER_SEGMENT_LABELS.get(segment, segment)
                segment_sites = sites[sites["CustomerSegment"].eq(segment)].copy()
                segment_rec_intervals = rec_segment_interval_df[
                    rec_segment_interval_df["CustomerSegment"].eq(segment)
                ].copy()
                segment_capacity_kw = float(
                    pd.to_numeric(segment_sites["SolarCECkW"], errors="coerce").sum()
                )

                if (
                    segment_sites.empty
                    or segment_rec_intervals.empty
                    or segment_capacity_kw <= 0
                ):
                    logging.warning(
                        "Skipping %s segment forecast; sites=%s intervals=%s capacity=%.2f kW",
                        segment_label,
                        len(segment_sites),
                        len(segment_rec_intervals),
                        segment_capacity_kw,
                    )
                    continue

                logging.info(
                    "Training %s segment model with %s sites, %.2f kW, %s actual intervals",
                    segment_label,
                    len(segment_sites),
                    segment_capacity_kw,
                    f"{len(segment_rec_intervals):,}",
                )
                segment_intrahour_shape = build_intrahour_production_shape(
                    segment_rec_intervals,
                    "Export_kWh",
                    method=args.intrahour_shape_method,
                    quantile=args.shape_quantile,
                )
                segment_daily_shape = build_average_daily_shape(
                    segment_rec_intervals,
                    "Export_kW",
                    method=args.daily_shape_method,
                    quantile=args.shape_quantile,
                )
                segment_daily_shape.insert(0, "CustomerSegment", segment)
                segment_shape_outputs.append(segment_daily_shape)

                segment_calibration_weather = weather_for_sites(
                    calibration_weather_source,
                    segment_sites,
                    args.use_capacity_weighted_weather,
                )
                segment_model = train_performance_model(
                    rec_intervals=segment_rec_intervals,
                    weather_df=segment_calibration_weather,
                    capacity_kw=segment_capacity_kw,
                    fallback_ratio=args.performance_ratio,
                    performance_ratio_upper_bound=args.performance_ratio_upper_bound,
                    sites=segment_sites,
                    latitude=args.latitude,
                    longitude=args.longitude,
                    timezone_name=args.timezone,
                    array_tilt_degrees=args.array_tilt_degrees,
                    array_azimuth_degrees=args.array_azimuth_degrees,
                )

                segment_residual_model = identity_residual_calibration_model(
                    lower_bound=args.residual_calibration_lower_bound,
                    upper_bound=args.residual_calibration_upper_bound,
                )
                segment_calibration_interval_forecast = pd.DataFrame()
                segment_calibration_backtest_hourly = pd.DataFrame()
                if args.backtest or args.residual_calibration:
                    segment_calibration_interval_forecast = build_interval_forecast(
                        weather_df=segment_calibration_weather,
                        intrahour_shape=segment_intrahour_shape,
                        capacity_kw=segment_capacity_kw,
                        model=segment_model,
                        sites=segment_sites,
                        latitude=args.latitude,
                        longitude=args.longitude,
                        timezone_name=args.timezone,
                        min_solar_elevation=args.min_solar_elevation,
                        array_tilt_degrees=args.array_tilt_degrees,
                        array_azimuth_degrees=args.array_azimuth_degrees,
                        forecast_source="backtest",
                    )
                    segment_calibration_interval_forecast["CustomerSegment"] = segment
                    segment_calibration_interval_forecast = add_hour_ending_column(
                        segment_calibration_interval_forecast
                    )
                    segment_calibration_backtest_hourly = build_hourly_backtest(
                        rec_interval_df=segment_rec_intervals,
                        interval_backtest_forecast=segment_calibration_interval_forecast,
                        total_capacity_kw=segment_capacity_kw,
                    )

                if args.residual_calibration:
                    segment_residual_model = train_residual_calibration_model(
                        backtest_df=segment_calibration_backtest_hourly,
                        total_capacity_kw=segment_capacity_kw,
                        latitude=args.latitude,
                        longitude=args.longitude,
                        timezone_name=args.timezone,
                        array_tilt_degrees=args.array_tilt_degrees,
                        array_azimuth_degrees=args.array_azimuth_degrees,
                        performance_ratio_upper_bound=args.performance_ratio_upper_bound,
                        lower_bound=args.residual_calibration_lower_bound,
                        upper_bound=args.residual_calibration_upper_bound,
                        min_training_forecast_kwh=args.residual_calibration_min_forecast_kwh,
                        min_training_rows=args.residual_calibration_min_rows,
                        validation_fraction=args.residual_calibration_validation_fraction,
                        min_validation_rows=args.residual_calibration_min_validation_rows,
                        use_energy_weighting=args.residual_calibration_energy_weighting,
                        energy_weight_power=args.residual_calibration_energy_weight_power,
                        use_seasonal_calibration=args.seasonal_calibration,
                        seasonal_prior_mwh=args.seasonal_calibration_prior_mwh,
                        seasonal_lower_bound=args.seasonal_calibration_lower_bound,
                        seasonal_upper_bound=args.seasonal_calibration_upper_bound,
                        use_quality_filter=args.calibration_quality_filter,
                        quality_min_row_coverage=args.calibration_min_daylight_row_coverage,
                        quality_min_actual_forecast_ratio=args.calibration_min_day_actual_forecast_ratio,
                        quality_min_day_forecast_mwh=args.calibration_min_day_forecast_mwh,
                        use_regime_calibration=args.regime_calibration,
                        regime_min_rows=args.regime_calibration_min_rows,
                        regime_min_forecast_mwh=args.regime_calibration_min_forecast_mwh,
                        regime_prior_mwh=args.regime_calibration_prior_mwh,
                        regime_lower_bound=args.regime_calibration_lower_bound,
                        regime_upper_bound=args.regime_calibration_upper_bound,
                        use_peak_calibration=args.peak_calibration,
                        peak_quantile=args.peak_calibration_quantile,
                        peak_min_rows=args.peak_calibration_min_rows,
                        peak_min_forecast_mwh=args.peak_calibration_min_forecast_mwh,
                        peak_lower_bound=args.peak_calibration_lower_bound,
                        peak_target_capture=args.peak_calibration_target_capture,
                    )

                if args.backtest and not segment_calibration_interval_forecast.empty:
                    if args.residual_calibration:
                        segment_backtest_interval_forecast = apply_residual_calibration(
                            interval_forecast=segment_calibration_interval_forecast,
                            calibration_model=segment_residual_model,
                            total_capacity_kw=segment_capacity_kw,
                            latitude=args.latitude,
                            longitude=args.longitude,
                            timezone_name=args.timezone,
                            array_tilt_degrees=args.array_tilt_degrees,
                            array_azimuth_degrees=args.array_azimuth_degrees,
                            performance_ratio_upper_bound=args.performance_ratio_upper_bound,
                        )
                        segment_backtest_interval_forecast["ForecastSource"] = (
                            "backtest_calibrated"
                        )
                    else:
                        segment_backtest_interval_forecast = (
                            segment_calibration_interval_forecast
                        )
                    segment_backtest_interval_forecast["CustomerSegment"] = segment

                    segment_backtest_hourly = build_hourly_backtest(
                        rec_interval_df=segment_rec_intervals,
                        interval_backtest_forecast=segment_backtest_interval_forecast,
                        total_capacity_kw=segment_capacity_kw,
                    )
                    segment_backtest_hourly["CustomerSegment"] = segment
                    if args.residual_calibration and args.peak_calibration:
                        peak_tuned = tune_peak_calibration_from_hourly_backtest(
                            calibration_model=segment_residual_model,
                            hourly_backtest=segment_backtest_hourly,
                            total_capacity_kw=segment_capacity_kw,
                            quantile=args.peak_calibration_quantile,
                            min_rows=args.peak_calibration_min_rows,
                            min_forecast_mwh=args.peak_calibration_min_forecast_mwh,
                            lower_bound=args.peak_calibration_lower_bound,
                            target_capture=args.peak_calibration_target_capture,
                            label=segment,
                        )
                        if peak_tuned:
                            segment_backtest_interval_forecast = apply_residual_calibration(
                                interval_forecast=segment_calibration_interval_forecast,
                                calibration_model=segment_residual_model,
                                total_capacity_kw=segment_capacity_kw,
                                latitude=args.latitude,
                                longitude=args.longitude,
                                timezone_name=args.timezone,
                                array_tilt_degrees=args.array_tilt_degrees,
                                array_azimuth_degrees=args.array_azimuth_degrees,
                                performance_ratio_upper_bound=args.performance_ratio_upper_bound,
                            )
                            segment_backtest_interval_forecast["ForecastSource"] = (
                                "backtest_calibrated"
                            )
                            segment_backtest_interval_forecast["CustomerSegment"] = (
                                segment
                            )
                            segment_backtest_hourly = build_hourly_backtest(
                                rec_interval_df=segment_rec_intervals,
                                interval_backtest_forecast=segment_backtest_interval_forecast,
                                total_capacity_kw=segment_capacity_kw,
                            )
                            segment_backtest_hourly["CustomerSegment"] = segment
                    segment_summary = calculate_backtest_summary(
                        segment_backtest_hourly
                    )
                    segment_summary.insert(0, "CustomerSegment", segment)
                    segment_hourly_backtests.append(segment_backtest_hourly)
                    segment_backtest_summaries.append(segment_summary)

                segment_forecast_parts = []
                if (
                    inferred_start_timestamp is not None
                    and inferred_end_timestamp is not None
                    and not inferred_weather_source.empty
                    and inferred_start_timestamp <= inferred_end_timestamp
                ):
                    segment_inferred_weather = weather_for_sites(
                        inferred_weather_source,
                        segment_sites,
                        args.use_capacity_weighted_weather,
                    )
                    segment_inferred_forecast = build_interval_forecast(
                        weather_df=segment_inferred_weather,
                        intrahour_shape=segment_intrahour_shape,
                        capacity_kw=segment_capacity_kw,
                        model=segment_model,
                        sites=segment_sites,
                        latitude=args.latitude,
                        longitude=args.longitude,
                        timezone_name=args.timezone,
                        min_solar_elevation=args.min_solar_elevation,
                        array_tilt_degrees=args.array_tilt_degrees,
                        array_azimuth_degrees=args.array_azimuth_degrees,
                        forecast_source="inferred_historical",
                    )
                    segment_inferred_forecast["CustomerSegment"] = segment
                    if args.residual_calibration:
                        segment_inferred_forecast = apply_residual_calibration(
                            interval_forecast=segment_inferred_forecast,
                            calibration_model=segment_residual_model,
                            total_capacity_kw=segment_capacity_kw,
                            latitude=args.latitude,
                            longitude=args.longitude,
                            timezone_name=args.timezone,
                            array_tilt_degrees=args.array_tilt_degrees,
                            array_azimuth_degrees=args.array_azimuth_degrees,
                            performance_ratio_upper_bound=args.performance_ratio_upper_bound,
                        )
                    segment_inferred_forecast["CustomerSegment"] = segment
                    segment_inferred_forecast = segment_inferred_forecast[
                        (
                            segment_inferred_forecast["IntervalStartDT"]
                            >= inferred_start_timestamp
                        )
                        & (
                            segment_inferred_forecast["IntervalStartDT"]
                            <= inferred_end_timestamp
                        )
                    ].copy()
                    segment_forecast_parts.append(
                        add_hour_ending_column(segment_inferred_forecast)
                    )

                segment_weather = weather_for_sites(
                    weather_source,
                    segment_sites,
                    args.use_capacity_weighted_weather,
                )
                segment_interval_forecast = build_interval_forecast(
                    weather_df=segment_weather,
                    intrahour_shape=segment_intrahour_shape,
                    capacity_kw=segment_capacity_kw,
                    model=segment_model,
                    sites=segment_sites,
                    latitude=args.latitude,
                    longitude=args.longitude,
                    timezone_name=args.timezone,
                    min_solar_elevation=args.min_solar_elevation,
                    array_tilt_degrees=args.array_tilt_degrees,
                    array_azimuth_degrees=args.array_azimuth_degrees,
                    forecast_source="forecast",
                )
                segment_interval_forecast["CustomerSegment"] = segment
                segment_interval_forecast["ForecastSource"] = np.where(
                    segment_interval_forecast["IntervalStartDT"].dt.date < today,
                    "historical_forecast",
                    "forecast",
                )
                if args.residual_calibration:
                    segment_interval_forecast = apply_residual_calibration(
                        interval_forecast=segment_interval_forecast,
                        calibration_model=segment_residual_model,
                        total_capacity_kw=segment_capacity_kw,
                        latitude=args.latitude,
                        longitude=args.longitude,
                        timezone_name=args.timezone,
                        array_tilt_degrees=args.array_tilt_degrees,
                        array_azimuth_degrees=args.array_azimuth_degrees,
                        performance_ratio_upper_bound=args.performance_ratio_upper_bound,
                    )
                    segment_interval_forecast["CustomerSegment"] = segment
                if args.same_day_correction:
                    segment_same_day_actuals = load_same_day_export_actuals(
                        parquet_root=parquet_root,
                        sites=segment_sites,
                        forecast_start_date=forecast_start_date,
                        forecast_end_date=forecast_end_date,
                        net_meter_export_source=args.net_meter_export_source,
                        latitude=args.latitude,
                        longitude=args.longitude,
                        timezone_name=args.timezone,
                        min_solar_elevation=args.min_solar_elevation,
                    )
                    segment_interval_forecast = apply_same_day_actual_correction(
                        interval_forecast=segment_interval_forecast,
                        same_day_actuals=segment_same_day_actuals,
                        timezone_name=args.timezone,
                        min_observed_intervals=args.same_day_correction_min_intervals,
                        min_observed_forecast_kwh=args.same_day_correction_min_forecast_kwh,
                        lower_bound=args.same_day_correction_lower_bound,
                        upper_bound=args.same_day_correction_upper_bound,
                        half_life_hours=args.same_day_correction_half_life_hours,
                        weather_similarity_floor=args.same_day_correction_weather_similarity_floor,
                    )
                    segment_interval_forecast["CustomerSegment"] = segment

                segment_forecast_parts.append(
                    add_hour_ending_column(segment_interval_forecast)
                )
                segment_full_interval_forecast = (
                    pd.concat(segment_forecast_parts, ignore_index=True)
                    .drop_duplicates(
                        subset=["CustomerSegment", "IntervalStartDT"], keep="last"
                    )
                    .sort_values(["CustomerSegment", "IntervalStartDT"])
                )
                segment_interval_outputs.append(segment_full_interval_forecast)

            if segment_interval_outputs:
                segmented_interval_forecast = pd.concat(
                    segment_interval_outputs, ignore_index=True
                )
                segmented_interval_forecast[
                    [
                        "CustomerSegment",
                        "IntervalStartDT",
                        "HE",
                        "BaseForecast_kW",
                        "BaseForecast_kWh",
                        "Forecast_kW",
                        "Forecast_kWh",
                        "ActiveCapacity_kW",
                        *FORECAST_DIAGNOSTIC_COLUMNS,
                        "PerformanceRatio",
                        "ResidualCalibrationFactor",
                        "SeasonalCalibrationFactor",
                        "RegimeCalibrationFactor",
                        "PeakCalibrationFactor",
                        "TotalCalibrationFactor",
                        "SameDayCorrectionFactor",
                        "ForecastSource",
                    ]
                ].to_csv(args.segment_output_15min, index=False)

                segmented_hourly_forecast = resample_interval_forecast_to_hourly(
                    segmented_interval_forecast,
                    total_capacity_kw,
                )
                segmented_hourly_forecast[
                    [
                        "CustomerSegment",
                        "IntervalStartDT",
                        "HE",
                        "BaseForecast_MW",
                        "BaseForecast_kWh",
                        "Forecast_MW",
                        "Forecast_kWh",
                        "BaseCapacityFactor",
                        "CapacityFactor",
                        "ActiveCapacity_kW",
                        *FORECAST_DIAGNOSTIC_COLUMNS,
                        "PerformanceRatio",
                        "ResidualCalibrationFactor",
                        "SeasonalCalibrationFactor",
                        "RegimeCalibrationFactor",
                        "PeakCalibrationFactor",
                        "TotalCalibrationFactor",
                        "SameDayCorrectionFactor",
                        "ForecastSource",
                    ]
                ].to_csv(args.segment_output_hourly, index=False)
                logging.info(
                    "Segmented 15-minute forecast saved to %s",
                    args.segment_output_15min,
                )
                logging.info(
                    "Segmented hourly forecast saved to %s", args.segment_output_hourly
                )

            if segment_hourly_backtests:
                segmented_backtest_hourly = pd.concat(
                    segment_hourly_backtests, ignore_index=True
                )
                segmented_backtest_hourly[
                    [
                        "CustomerSegment",
                        "IntervalStartDT",
                        "HE",
                        "Actual_MW",
                        "Actual_kWh",
                        "BaseForecast_MW",
                        "BaseForecast_kWh",
                        "Forecast_MW",
                        "Forecast_kWh",
                        "BaseError_MW",
                        "BaseError_kWh",
                        "Error_MW",
                        "Error_kWh",
                        "BaseAbsError_MW",
                        "BaseAbsError_kWh",
                        "AbsError_MW",
                        "AbsError_kWh",
                        "BaseAPE",
                        "APE",
                        "BaseCapacityFactor",
                        "CapacityFactor",
                        "ActiveCapacity_kW",
                        *FORECAST_DIAGNOSTIC_COLUMNS,
                        "PerformanceRatio",
                        "ResidualCalibrationFactor",
                        "SeasonalCalibrationFactor",
                        "RegimeCalibrationFactor",
                        "PeakCalibrationFactor",
                        "TotalCalibrationFactor",
                        "SameDayCorrectionFactor",
                        "BacktestForecast",
                        "ForecastSource",
                    ]
                ].to_csv(args.segment_backtest_hourly_output, index=False)
                pd.concat(segment_backtest_summaries, ignore_index=True).to_csv(
                    args.segment_backtest_summary_output,
                    index=False,
                )
                logging.info(
                    "Segmented backtest saved to %s",
                    args.segment_backtest_hourly_output,
                )

            if segment_shape_outputs:
                pd.concat(segment_shape_outputs, ignore_index=True).to_csv(
                    args.segment_load_shape_output,
                    index=False,
                )
                logging.info(
                    "Segmented load shapes saved to %s", args.segment_load_shape_output
                )

        logging.info("15-minute forecast saved to %s", args.output_15min)
        logging.info("Hourly forecast saved to %s", args.output_hourly)

        print("\n\n=== Roseville Hourly Forecast (first 20 rows) ===")
        print(
            hourly_forecast.head(20).to_string(
                index=False, float_format="{:.2f}".format
            )
        )

    except Exception as exc:
        logging.exception("Forecaster failed")
        raise

    finally:
        if engine is not None:
            engine.dispose()


def parse_args(argv: Optional[list[str]] = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Model and forecast system-wide solar generation.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )

    # Connection arguments
    parser.add_argument(
        "--driver", default="ODBC Driver 17 for SQL Server", help="ODBC driver to use."
    )
    parser.add_argument(
        "--timeout", type=int, default=30, help="ODBC connection timeout in seconds."
    )

    # Destination arguments
    dest_group = parser.add_argument_group("Destination Connection")
    dest_group.add_argument(
        "--dest-server", default="re-lrs-db-1", help="Destination server name."
    )
    dest_group.add_argument(
        "--dest-db", default="Forecast", help="Destination database name."
    )
    dest_group.add_argument(
        "--dest-user",
        default=None,
        help="Destination username (for SQL authentication).",
    )
    dest_group.add_argument(
        "--dest-pass",
        default=None,
        help="Destination password (for SQL authentication).",
    )

    # Forecaster options
    parser.add_argument(
        "--production-source",
        choices=["rec-parquet", "db-representative"],
        default="rec-parquet",
        help="Historical production source used to build the interval shape and calibration.",
    )
    parser.add_argument(
        "--use-capacity-weighted-weather",
        action="store_true",
        default=True,
        help=(
            "Use all forecast-eligible site coordinates to choose representative weather sample points, then "
            "capacity-weight those samples back to the full fleet."
        ),
    )
    parser.add_argument(
        "--weather-clusters",
        type=int,
        default=DEFAULT_WEATHER_CLUSTERS,
        help=(
            "Number of representative solar-site weather samples to select from all forecast-eligible coordinates. "
            "Requested locations are capped by --max-weather-api-calls * --weather-locations-per-request; "
            "0 uses the single representative Roseville point."
        ),
    )
    parser.add_argument(
        "--weather-locations-per-request",
        type=int,
        default=DEFAULT_WEATHER_LOCATIONS_PER_REQUEST,
        help="Maximum number of sampled weather locations sent in one Open-Meteo request.",
    )
    parser.add_argument(
        "--max-weather-api-calls",
        type=int,
        default=DEFAULT_MAX_WEATHER_API_CALLS,
        help=(
            "Maximum Open-Meteo requests allowed for each historical or forecast weather fetch. "
            "Increase this with --weather-clusters when you want more sampled locations."
        ),
    )
    parser.add_argument(
        "--parquet-root",
        default=str(DEFAULT_PARQUET_ROOT),
        help="Root folder containing COM and RES interval parquet files and the parquet index cache.",
    )
    parser.add_argument(
        "--weather-cache-dir",
        default=str(DEFAULT_SOLAR_WEATHER_CACHE_DIR),
        help="Solar Open-Meteo cache directory.",
    )
    parser.add_argument(
        "--rec-history-months",
        type=int,
        default=12,
        help=(
            "Most recent available export parquet months used when explicit history dates are not provided. "
            "Use longer windows (12-24 months) with active-capacity normalization for stronger seasonality learning."
        ),
    )
    parser.add_argument(
        "--net-meter-export-source",
        choices=["net", "rec"],
        default="net",
        help=(
            "For AMI_NET/AMI_NET_D sites, use negative NEM NET intervals as export "
            "or direct REC channel rows."
        ),
    )
    parser.add_argument(
        "--segment-forecasts",
        dest="segment_forecasts",
        action="store_true",
        default=True,
        help="Train and write separate customer-segment forecasts in addition to the total forecast.",
    )
    parser.add_argument(
        "--no-segment-forecasts",
        dest="segment_forecasts",
        action="store_false",
        help="Skip NEM/Solar 2.0 segmented forecast outputs.",
    )
    parser.add_argument(
        "--customer-segments",
        default=f"{CUSTOMER_SEGMENT_NEM},{CUSTOMER_SEGMENT_SOLAR_20}",
        help="Comma-separated customer segments to model separately: NEM, SOLAR_2_0, OTHER.",
    )
    parser.add_argument(
        "--latitude",
        type=float,
        default=ROSEVILLE_LATITUDE,
        help="Representative system latitude.",
    )
    parser.add_argument(
        "--longitude",
        type=float,
        default=ROSEVILLE_LONGITUDE,
        help="Representative system longitude.",
    )
    parser.add_argument(
        "--timezone",
        default="America/Los_Angeles",
        help="Local timezone for interval timestamps.",
    )
    parser.add_argument(
        "--min-solar-elevation",
        type=float,
        default=0.0,
        help=(
            "Minimum solar elevation angle in degrees for treating interval export as solar. "
            "Use 0 for daylight only; use 1-3 to remove low-sun/twilight noise."
        ),
    )
    parser.add_argument(
        "--rec-history-start",
        type=date.fromisoformat,
        default=None,
        help="Export calibration start date in YYYY-MM-DD format.",
    )
    parser.add_argument(
        "--rec-history-end",
        type=date.fromisoformat,
        default=None,
        help="Export calibration end date in YYYY-MM-DD format.",
    )
    parser.add_argument(
        "--historical-days",
        type=int,
        default=30,
        help="Number of historical days to include in the forecast output.",
    )
    parser.add_argument(
        "--forecast-days",
        type=int,
        default=16,
        help="Number of forecast days to produce (max 16).",
    )
    parser.add_argument(
        "--performance-ratio",
        type=float,
        default=DEFAULT_PERFORMANCE_RATIO,
        help="Fallback performance ratio used when REC/weather calibration is unavailable.",
    )
    parser.add_argument(
        "--performance-ratio-upper-bound",
        type=float,
        default=DEFAULT_PERFORMANCE_RATIO_UPPER_BOUND,
        help="Upper bound for learned performance ratios; values above 1 allow brief cloud-edge peak enhancement.",
    )
    parser.add_argument(
        "--array-tilt-degrees",
        type=float,
        default=DEFAULT_ARRAY_TILT_DEGREES,
        help="Representative fixed-array tilt used for plane-of-array irradiance features.",
    )
    parser.add_argument(
        "--array-azimuth-degrees",
        type=float,
        default=DEFAULT_ARRAY_AZIMUTH_DEGREES,
        help="Representative array azimuth, degrees clockwise from north; 180 is south-facing.",
    )
    parser.add_argument(
        "--daily-shape-method",
        choices=["mean", "median", "upper-quantile"],
        default=DEFAULT_DAILY_SHAPE_METHOD,
        help="Method used for the dashboard/display daily solar export shape.",
    )
    parser.add_argument(
        "--intrahour-shape-method",
        choices=["mean", "median", "upper-quantile"],
        default=DEFAULT_INTRAHOUR_SHAPE_METHOD,
        help="Method used to split hourly forecast energy into 15-minute intervals.",
    )
    parser.add_argument(
        "--shape-quantile",
        type=float,
        default=DEFAULT_SHAPE_QUANTILE,
        help="Quantile used when a shape method is upper-quantile.",
    )
    parser.add_argument(
        "--same-day-correction",
        dest="same_day_correction",
        action="store_true",
        default=True,
        help="Scale remaining same-day forecast intervals using completed actual export intervals.",
    )
    parser.add_argument(
        "--no-same-day-correction",
        dest="same_day_correction",
        action="store_false",
        help="Disable same-day actual-vs-forecast correction.",
    )
    parser.add_argument(
        "--same-day-correction-min-intervals",
        type=int,
        default=4,
        help="Minimum observed daylight 15-minute intervals required before applying same-day correction.",
    )
    parser.add_argument(
        "--same-day-correction-min-forecast-kwh",
        type=float,
        default=100.0,
        help="Minimum observed forecast kWh required before applying same-day correction.",
    )
    parser.add_argument(
        "--same-day-correction-lower-bound",
        type=float,
        default=0.25,
        help="Lowest allowed same-day correction factor.",
    )
    parser.add_argument(
        "--same-day-correction-upper-bound",
        type=float,
        default=1.75,
        help="Highest allowed same-day correction factor.",
    )
    parser.add_argument(
        "--same-day-correction-half-life-hours",
        type=float,
        default=4.0,
        help="Half-life used to blend same-day correction factors back toward 1.0 with forecast lead time.",
    )
    parser.add_argument(
        "--same-day-correction-weather-similarity-floor",
        type=float,
        default=0.15,
        help="Minimum weather-similarity weight used by same-day correction when remaining weather differs from observed intervals.",
    )
    parser.add_argument(
        "--residual-calibration",
        dest="residual_calibration",
        action="store_true",
        default=True,
        help="Learn and apply a bounded actual-vs-forecast residual correction model.",
    )
    parser.add_argument(
        "--no-residual-calibration",
        dest="residual_calibration",
        action="store_false",
        help="Disable learned residual forecast calibration.",
    )
    parser.add_argument(
        "--residual-calibration-min-rows",
        type=int,
        default=96,
        help="Minimum daylight hourly backtest rows required to train the residual calibration model.",
    )
    parser.add_argument(
        "--residual-calibration-validation-fraction",
        type=float,
        default=0.20,
        help="Most recent fraction of residual-calibration rows held out for time-based validation reporting.",
    )
    parser.add_argument(
        "--residual-calibration-min-validation-rows",
        type=int,
        default=48,
        help="Minimum recent holdout rows required for residual-calibration validation reporting.",
    )
    parser.add_argument(
        "--residual-calibration-min-forecast-kwh",
        type=float,
        default=25.0,
        help="Minimum hourly forecast kWh for a row to be used in residual calibration training.",
    )
    parser.add_argument(
        "--residual-calibration-energy-weighting",
        dest="residual_calibration_energy_weighting",
        action="store_true",
        default=True,
        help="Weight residual calibration training toward higher-energy forecast hours.",
    )
    parser.add_argument(
        "--no-residual-calibration-energy-weighting",
        dest="residual_calibration_energy_weighting",
        action="store_false",
        help="Fit residual calibration without hourly energy weights.",
    )
    parser.add_argument(
        "--residual-calibration-energy-weight-power",
        type=float,
        default=DEFAULT_RESIDUAL_CALIBRATION_ENERGY_WEIGHT_POWER,
        help=(
            "Exponent applied to hourly forecast kWh when residual calibration energy weighting is enabled. "
            "Values above 1 emphasize high-output peak hours."
        ),
    )
    parser.add_argument(
        "--residual-calibration-lower-bound",
        type=float,
        default=0.25,
        help="Lowest allowed learned residual correction factor.",
    )
    parser.add_argument(
        "--residual-calibration-upper-bound",
        type=float,
        default=1.75,
        help="Highest allowed learned residual correction factor.",
    )
    parser.add_argument(
        "--seasonal-calibration",
        dest="seasonal_calibration",
        action="store_true",
        default=True,
        help="Apply a bounded month-level correction after residual calibration.",
    )
    parser.add_argument(
        "--no-seasonal-calibration",
        dest="seasonal_calibration",
        action="store_false",
        help="Disable month-level seasonal correction.",
    )
    parser.add_argument(
        "--seasonal-calibration-prior-mwh",
        type=float,
        default=500.0,
        help="Prior MWh used to shrink month-level calibration factors toward the aggregate factor.",
    )
    parser.add_argument(
        "--seasonal-calibration-lower-bound",
        type=float,
        default=0.85,
        help="Lowest allowed month-level seasonal correction factor.",
    )
    parser.add_argument(
        "--seasonal-calibration-upper-bound",
        type=float,
        default=1.15,
        help="Highest allowed month-level seasonal correction factor.",
    )
    parser.add_argument(
        "--calibration-quality-filter",
        dest="calibration_quality_filter",
        action="store_true",
        default=True,
        help="Exclude likely incomplete actual-feed days from residual calibration training.",
    )
    parser.add_argument(
        "--no-calibration-quality-filter",
        dest="calibration_quality_filter",
        action="store_false",
        help="Use all daylight residual calibration rows, including suspect actual-feed days.",
    )
    parser.add_argument(
        "--calibration-min-daylight-row-coverage",
        type=float,
        default=DEFAULT_CALIBRATION_DAYLIGHT_ROW_COVERAGE_MIN,
        help="Minimum observed actual row coverage required for a daylight calibration day.",
    )
    parser.add_argument(
        "--calibration-min-day-actual-forecast-ratio",
        type=float,
        default=DEFAULT_CALIBRATION_DAY_ACTUAL_FORECAST_RATIO_MIN,
        help="Minimum actual/forecast daily energy ratio for material daylight calibration days.",
    )
    parser.add_argument(
        "--calibration-min-day-forecast-mwh",
        type=float,
        default=DEFAULT_CALIBRATION_DAY_FORECAST_MWH_MIN,
        help="Daily forecast MWh threshold before the actual/forecast ratio quality gate is applied.",
    )
    parser.add_argument(
        "--regime-calibration",
        dest="regime_calibration",
        action="store_true",
        default=True,
        help="Apply bounded clear-sky/cloud/elevation regime correction after residual and seasonal calibration.",
    )
    parser.add_argument(
        "--no-regime-calibration",
        dest="regime_calibration",
        action="store_false",
        help="Disable weather-regime residual correction.",
    )
    parser.add_argument(
        "--regime-calibration-min-rows",
        type=int,
        default=DEFAULT_REGIME_CALIBRATION_MIN_ROWS,
        help="Minimum hourly rows required before learning a weather-regime correction factor.",
    )
    parser.add_argument(
        "--regime-calibration-min-forecast-mwh",
        type=float,
        default=DEFAULT_REGIME_CALIBRATION_MIN_FORECAST_MWH,
        help="Minimum forecast MWh required before learning a weather-regime correction factor.",
    )
    parser.add_argument(
        "--regime-calibration-prior-mwh",
        type=float,
        default=DEFAULT_REGIME_CALIBRATION_PRIOR_MWH,
        help="Shrinkage prior MWh that pulls sparse weather-regime factors toward 1.0.",
    )
    parser.add_argument(
        "--regime-calibration-lower-bound",
        type=float,
        default=DEFAULT_REGIME_CALIBRATION_LOWER_BOUND,
        help="Lowest allowed weather-regime correction factor.",
    )
    parser.add_argument(
        "--regime-calibration-upper-bound",
        type=float,
        default=DEFAULT_REGIME_CALIBRATION_UPPER_BOUND,
        help="Highest allowed weather-regime correction factor.",
    )
    parser.add_argument(
        "--peak-calibration",
        dest="peak_calibration",
        action="store_true",
        default=True,
        help="Apply an upper-tail peak envelope correction after residual, seasonal, and regime calibration.",
    )
    parser.add_argument(
        "--no-peak-calibration",
        dest="peak_calibration",
        action="store_false",
        help="Disable upper-tail peak envelope correction.",
    )
    parser.add_argument(
        "--peak-calibration-quantile",
        type=float,
        default=DEFAULT_PEAK_CALIBRATION_QUANTILE,
        help="Forecast capacity-factor quantile that defines the upper-tail peak calibration band.",
    )
    parser.add_argument(
        "--peak-calibration-min-rows",
        type=int,
        default=DEFAULT_PEAK_CALIBRATION_MIN_ROWS,
        help="Minimum upper-tail hourly rows required before learning a peak calibration factor.",
    )
    parser.add_argument(
        "--peak-calibration-min-forecast-mwh",
        type=float,
        default=DEFAULT_PEAK_CALIBRATION_MIN_FORECAST_MWH,
        help="Minimum upper-tail forecast MWh required before learning a peak calibration factor.",
    )
    parser.add_argument(
        "--peak-calibration-lower-bound",
        type=float,
        default=DEFAULT_PEAK_CALIBRATION_LOWER_BOUND,
        help="Lowest allowed upper-tail peak calibration factor.",
    )
    parser.add_argument(
        "--peak-calibration-target-capture",
        type=float,
        default=DEFAULT_PEAK_CALIBRATION_TARGET_CAPTURE,
        help="Target forecast peak divided by actual peak when learning the upper-tail peak factor.",
    )
    parser.add_argument(
        "--backtest",
        dest="backtest",
        action="store_true",
        default=True,
        help="Build an hourly historical forecast backtest against REC/NET actual export.",
    )
    parser.add_argument(
        "--no-backtest",
        dest="backtest",
        action="store_false",
        help="Skip historical actual-vs-forecast backtest output.",
    )
    parser.add_argument(
        "--infer-missing-history",
        dest="infer_missing_history",
        action="store_true",
        default=True,
        help="Infer historical solar generation between the last actual interval and the forecast output start.",
    )
    parser.add_argument(
        "--no-infer-missing-history",
        dest="infer_missing_history",
        action="store_false",
        help="Do not append inferred historical rows for missing actual history.",
    )
    parser.add_argument(
        "--output-15min",
        default="forecast_outputs/roseville_solar_forecast.csv",
        help="15-minute forecast CSV path.",
    )
    parser.add_argument(
        "--output-hourly",
        default="forecast_outputs/roseville_solar_forecast_hourly.csv",
        help="Hourly forecast CSV path.",
    )
    parser.add_argument(
        "--segment-output-15min",
        default="forecast_outputs/roseville_solar_forecast_by_segment.csv",
        help="Segmented 15-minute forecast CSV path.",
    )
    parser.add_argument(
        "--segment-output-hourly",
        default="forecast_outputs/roseville_solar_forecast_hourly_by_segment.csv",
        help="Segmented hourly forecast CSV path.",
    )
    parser.add_argument(
        "--backtest-hourly-output",
        default="forecast_outputs/roseville_solar_backtest_hourly.csv",
        help="Hourly historical actual-vs-forecast backtest CSV path.",
    )
    parser.add_argument(
        "--backtest-summary-output",
        default="forecast_outputs/roseville_solar_backtest_summary.csv",
        help="Historical actual-vs-forecast backtest summary CSV path.",
    )
    parser.add_argument(
        "--solar-backtest-diagnostics-output",
        default="forecast_outputs/roseville_solar_backtest_diagnostics.csv",
        help="Slice-level solar backtest diagnostics CSV path.",
    )
    parser.add_argument(
        "--solar-backtest-top-errors-output",
        default="forecast_outputs/roseville_solar_backtest_top_errors.csv",
        help="Ranked top solar backtest underforecast/overforecast hours CSV path.",
    )
    parser.add_argument(
        "--solar-backtest-holdout-output",
        default="forecast_outputs/roseville_solar_backtest_holdout_scorecard.csv",
        help="Temporal holdout solar backtest scorecard CSV path.",
    )
    parser.add_argument(
        "--solar-backtest-holdout-hourly-output",
        default="forecast_outputs/roseville_solar_backtest_holdout_hourly.csv",
        help="Hourly temporal holdout solar backtest CSV path.",
    )
    parser.add_argument(
        "--solar-backtest-holdout-days",
        type=int,
        default=DEFAULT_SOLAR_BACKTEST_HOLDOUT_DAYS,
        help="Number of final historical days held out for out-of-sample solar scoring; use 0 to disable.",
    )
    parser.add_argument(
        "--solar-backtest-daylight-threshold-mw",
        type=float,
        default=DEFAULT_SOLAR_BACKTEST_DAYLIGHT_THRESHOLD_MW,
        help="Actual or forecast MW threshold used for daylight/active solar diagnostics.",
    )
    parser.add_argument(
        "--solar-backtest-top-error-count",
        type=int,
        default=DEFAULT_SOLAR_BACKTEST_TOP_ERROR_COUNT,
        help="Number of underforecast and overforecast hours to write to the top-error diagnostics.",
    )
    parser.add_argument(
        "--segment-backtest-hourly-output",
        default="forecast_outputs/roseville_solar_backtest_hourly_by_segment.csv",
        help="Segmented hourly backtest CSV path.",
    )
    parser.add_argument(
        "--segment-backtest-summary-output",
        default="forecast_outputs/roseville_solar_backtest_summary_by_segment.csv",
        help="Segmented backtest summary CSV path.",
    )
    parser.add_argument(
        "--rec-actual-15min-output",
        default="forecast_outputs/roseville_solar_rec_actual_15min.csv",
        help="Historical export actual 15-minute CSV path.",
    )
    parser.add_argument(
        "--rec-actual-hourly-output",
        default="forecast_outputs/roseville_solar_rec_actual_hourly.csv",
        help="Historical export actual hourly CSV path.",
    )
    parser.add_argument(
        "--segment-rec-actual-15min-output",
        default="forecast_outputs/roseville_solar_rec_actual_15min_by_segment.csv",
        help="Segmented historical export actual 15-minute CSV path.",
    )
    parser.add_argument(
        "--segment-rec-actual-hourly-output",
        default="forecast_outputs/roseville_solar_rec_actual_hourly_by_segment.csv",
        help="Segmented historical export actual hourly CSV path.",
    )
    parser.add_argument(
        "--load-shape-output",
        default="forecast_outputs/roseville_solar_load_shape.csv",
        help="Average daily load shape CSV path.",
    )
    parser.add_argument(
        "--segment-load-shape-output",
        default="forecast_outputs/roseville_solar_load_shape_by_segment.csv",
        help="Segmented average daily load shape CSV path.",
    )
    parser.add_argument("--verbose", action="store_true", help="Enable debug logging.")

    return parser.parse_args(argv)


def main(argv: Optional[list[str]] = None) -> int:
    args = parse_args(argv)
    configure_logging(args.verbose)

    try:
        run_forecaster(args)
        return 0
    except Exception:
        return 1


if __name__ == "__main__":
    sys.exit(main())
