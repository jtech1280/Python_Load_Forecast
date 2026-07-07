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
2.  Loads active solar site data (capacity, location) from the
    `Forecasting.ForecastSolarSite` table.
3.  Groups sites into geographic clusters to reduce weather API calls.
4.  Loads REC channel export and negative NET interval export data from parquet
    files for active solar sites.
5.  Trains a weather/time performance-ratio model from historical export and
    hourly irradiance/cloud-cover weather.
6.  Optionally trains residual and seasonal calibration factors from the solar
    backtest.
7.  Fetches hourly GHI and cloud-cover weather data from the Open-Meteo API,
    using local cache/retry handling.
8.  Builds hourly export energy from GHI, system capacity, and the learned
    performance model.
9.  Splits each hourly forecast into 15-minute intervals using the intra-hour
    historical shape.
10. Optionally corrects remaining same-day intervals using completed actual
    export intervals already observed today.
11. Saves the final forecast, actuals, backtest, and production shapes to CSV
    files.

Requirements
------------
pip install pyodbc pandas requests scikit-learn SQLAlchemy pyarrow
"""

from __future__ import annotations

import argparse
import hashlib
import logging
import math
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

from db_utils import connect, read_sql

# Suppress only the single InsecureRequestWarning from urllib3 needed for this script
urllib3.disable_warnings(urllib3.exceptions.InsecureRequestWarning)


# =============================================================================
# Logging
# =============================================================================

LOG_FORMAT = "%(asctime)s | %(levelname)-8s | %(message)s"
INTERVAL_HOURS = 0.25
DEFAULT_PERFORMANCE_RATIO = 0.75
DEFAULT_PARQUET_ROOT = Path(r"C:\PY_LRS")
DEFAULT_SOLAR_WEATHER_CACHE_DIR = Path("weather_cache") / "solar_weather"
DEFAULT_FORECAST_WEATHER_CACHE_MAX_AGE_HOURS = 6.0
ROSEVILLE_LATITUDE = 38.7522
ROSEVILLE_LONGITUDE = -121.2880
DEFAULT_DAILY_SHAPE_METHOD = "upper-quantile"
DEFAULT_INTRAHOUR_SHAPE_METHOD = "median"
DEFAULT_SHAPE_QUANTILE = 0.75
DEFAULT_MAX_PERFORMANCE_RATIO = 1.10
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
ACTUAL_QUALITY_OK = "OK"
ACTUAL_QUALITY_AMI_SUPPRESSED = "AMI_SUPPRESSED_ACTUAL"
# Fallback air temperature (deg C) used when weather is missing. Roughly the
# 25 deg C PV reference so the temperature feature stays derating-neutral.
DEFAULT_TEMPERATURE_C = 25.0
NET_METER_TYPES = {"AMI_NET", "AMI_NET_D"}
HOURLY_WEATHER_VARIABLES = [
    "shortwave_radiation",
    "direct_radiation",
    "diffuse_radiation",
    "temperature_2m",
    "wind_speed_10m",
    "cloud_cover",
    "cloud_cover_low",
    "cloud_cover_mid",
    "cloud_cover_high",
]
WEATHER_OUTPUT_COLUMNS = [
    "GHI_kWh_per_m2",
    "WeatherGHI_Wm2",
    "DirectRadiation_Wm2",
    "DiffuseRadiation_Wm2",
    "Temperature_C",
    "WindSpeed_ms",
    "CloudCoverPct",
    "CloudCoverLowPct",
    "CloudCoverMidPct",
    "CloudCoverHighPct",
]
PERFORMANCE_FEATURE_COLUMNS = [
    "GHI_kWh_per_m2",
    "WeatherGHI_Wm2",
    "DirectRadiation_Wm2",
    "DiffuseRadiation_Wm2",
    "ClearSkyGHI_Wm2",
    "ClearSkyIndex",
    "Temperature_C",
    "WindSpeed_ms",
    "CloudCoverPct",
    "CloudCoverLowPct",
    "CloudCoverMidPct",
    "CloudCoverHighPct",
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
MAX_SITES_PER_WEATHER_REQUEST = 50
DEFAULT_WEATHER_CLUSTERS = 10
OPEN_METEO_TIMEOUT_SECONDS = 60
OPEN_METEO_MAX_ATTEMPTS = 4
OPEN_METEO_RETRY_BACKOFF_SECONDS = 3.0


@dataclass
class PerformanceModel:
    estimator: Optional[GradientBoostingRegressor]
    fallback_ratio: float
    feature_columns: list[str]
    upper_bound: float = DEFAULT_MAX_PERFORMANCE_RATIO


@dataclass
class ResidualCalibrationModel:
    estimator: Optional[GradientBoostingRegressor]
    fallback_factor: float
    feature_columns: list[str]
    lower_bound: float
    upper_bound: float
    seasonal_factors: dict[int, float] = field(default_factory=dict)
    seasonal_default_factor: float = 1.0


def configure_logging(verbose: bool = False) -> None:
    logging.basicConfig(
        level=logging.DEBUG if verbose else logging.INFO,
        format=LOG_FORMAT,
        datefmt="%Y-%m-%d %H:%M:%S",
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
            if status_code is not None and 400 <= status_code < 500 and status_code != 429:
                raise
        except (requests.RequestException, ValueError) as exc:
            last_error = exc

        if attempt >= OPEN_METEO_MAX_ATTEMPTS:
            break
        sleep_seconds = OPEN_METEO_RETRY_BACKOFF_SECONDS * attempt
        logging.warning(
            "Open-Meteo %s weather request failed for %s to %s on attempt %s/%s: %s. "
            "Retrying in %.1f seconds.",
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
    cols = [col for col in ["SolarSiteKey", "Latitude", "Longitude"] if col in sites.columns]
    if not cols:
        return "no_site_columns"
    work = sites[cols].copy()
    sort_cols = [col for col in ["SolarSiteKey", "Latitude", "Longitude"] if col in work.columns]
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
    stem = "_".join(stem_parts)
    return root / f"{stem}.csv"


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
        valid = out[timestamp_col].notna()
        in_range = valid & out[timestamp_col].ge(start_date) & out[timestamp_col].le(end_date)
    else:
        out[timestamp_col] = pd.to_datetime(out[timestamp_col], errors="coerce")
        valid = out[timestamp_col].notna()
        in_range = valid & out[timestamp_col].dt.date.ge(start_date) & out[timestamp_col].dt.date.le(end_date)
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


def _archive_solar_forecast_weather(df: pd.DataFrame, cache_dir: str | Path | None, source_path: Path) -> None:
    if df is None or df.empty:
        return
    archive_dir = _solar_weather_cache_root(cache_dir) / "forecast_weather_runs"
    timestamp = pd.Timestamp.now(tz="UTC").strftime("%Y%m%d_%H%M%S")
    archive_path = archive_dir / f"{source_path.stem}_{timestamp}.csv"
    _write_solar_weather_cache(df, archive_path)


# =============================================================================
# Data Loading
# =============================================================================

def get_total_capacity(conn: Engine) -> float:
    """
    Calculates the total active solar capacity from the database.
    """
    logging.info("Calculating total active solar capacity")
    sql = "SELECT SUM(SolarCECkW) as TotalKw FROM Forecasting.ForecastSolarSite WHERE IsActive = 1"
    total_capacity = pd.read_sql(sql, conn).iloc[0]['TotalKw']
    if pd.isna(total_capacity) or float(total_capacity) <= 0:
        raise ValueError("No positive active solar capacity found.")
    total_capacity = float(total_capacity)
    logging.info(f"Total active solar capacity: {total_capacity:,.2f} kW")
    return total_capacity


def load_active_solar_sites(conn: Engine) -> pd.DataFrame:
    """
    Load active solar sites used to match REC service point parquet rows.
    """
    logging.info("Loading active solar sites")
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
    WHERE IsActive = 1
      AND LocationNumber IS NOT NULL
      AND SolarCECkW IS NOT NULL
      AND SolarCECkW > 0;
    """
    df = read_sql(conn, sql)
    if df.empty:
        raise ValueError("No active solar sites found.")

    df["LocationNumber"] = df["LocationNumber"].astype("Int64").astype(str)
    df["SolarCECkW"] = pd.to_numeric(df["SolarCECkW"], errors="coerce")
    df = df.dropna(subset=["LocationNumber", "SolarCECkW"])
    logging.info(
        "Loaded %s active solar sites totaling %s kW",
        len(df),
        f"{df['SolarCECkW'].sum():,.2f}",
    )
    return df


def build_daily_active_capacity(
    sites: pd.DataFrame,
    start_date: date,
    end_date: date,
) -> pd.DataFrame:
    """
    Build a daily active solar capacity (kW) series from site interconnection dates.

    Each site contributes its ``SolarCECkW`` from its ``InterconnectionDate`` onward.
    Sites without a usable interconnection date are treated as active for the whole
    window, matching the legacy assumption that all currently active sites always
    existed. The returned frame has one row per calendar day with columns ``Date``
    and ``ActiveCapacity_kW``. An empty frame is returned when capacity/date info is
    unavailable so callers can fall back to a single current-capacity scalar.
    """
    empty = pd.DataFrame(columns=["Date", "ActiveCapacity_kW"])
    if sites is None or sites.empty or "SolarCECkW" not in sites.columns:
        return empty
    if start_date > end_date:
        return empty

    work = sites.copy()
    work["SolarCECkW"] = pd.to_numeric(work["SolarCECkW"], errors="coerce").fillna(0.0)
    total_capacity = float(work["SolarCECkW"].sum())
    if total_capacity <= 0:
        return empty

    if "InterconnectionDate" in work.columns:
        interconnect = pd.to_datetime(work["InterconnectionDate"], errors="coerce")
    else:
        interconnect = pd.Series(pd.NaT, index=work.index)
    days = pd.date_range(start_date, end_date, freq="D")
    dated_mask = interconnect.notna()
    undated_capacity = float(work.loc[~dated_mask, "SolarCECkW"].sum())

    if not bool(dated_mask.any()):
        active_values = np.full(len(days), total_capacity, dtype="float64")
    else:
        dated = pd.DataFrame(
            {
                "InterconnectDay": interconnect[dated_mask].dt.normalize().to_numpy(),
                "SolarCECkW": work.loc[dated_mask, "SolarCECkW"].to_numpy(),
            }
        )
        cumulative = dated.groupby("InterconnectDay")["SolarCECkW"].sum().sort_index().cumsum()
        active_dated = (
            cumulative.reindex(cumulative.index.union(days)).ffill().reindex(days).fillna(0.0)
        )
        active_values = active_dated.to_numpy(dtype="float64") + undated_capacity

    out = pd.DataFrame({"Date": days.date, "ActiveCapacity_kW": active_values})
    active_start = float(out["ActiveCapacity_kW"].iloc[0])
    active_end = float(out["ActiveCapacity_kW"].iloc[-1])
    growth_pct = ((active_end - active_start) / active_start * 100.0) if active_start > 0 else float("nan")
    logging.info(
        "Built daily active solar capacity %s to %s: %.0f -> %.0f kW (%.1f%% growth); "
        "%s of %s sites lack an interconnection date and are treated as always active",
        start_date,
        end_date,
        active_start,
        active_end,
        growth_pct,
        int((~dated_mask).sum()),
        len(work),
    )
    return out


def _resolve_row_capacity(
    timestamps: pd.Series,
    daily_active_capacity: Optional[pd.DataFrame],
    fallback_capacity_kw: float,
) -> pd.Series:
    """
    Map interval timestamps to the active capacity (kW) for their calendar day.

    Falls back to ``fallback_capacity_kw`` for days outside the supplied series or
    when no daily capacity frame is provided.
    """
    timestamps = pd.to_datetime(timestamps)
    fallback = float(fallback_capacity_kw)
    if daily_active_capacity is None or daily_active_capacity.empty:
        return pd.Series(fallback, index=timestamps.index)
    lookup = (
        daily_active_capacity.dropna(subset=["Date"])
        .drop_duplicates(subset=["Date"], keep="last")
        .set_index("Date")["ActiveCapacity_kW"]
    )
    mapped = timestamps.dt.date.map(lookup)
    return pd.to_numeric(mapped, errors="coerce").fillna(fallback)


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


def calculate_solar_elevation_degrees(
    timestamps: pd.Series,
    latitude: float,
    longitude: float,
    timezone_name: str,
) -> pd.Series:
    """
    Approximate solar elevation for naive local timestamps using NOAA equations.
    """
    local_timestamps = pd.to_datetime(timestamps) + pd.Timedelta(minutes=INTERVAL_HOURS * 60 / 2)
    timezone = ZoneInfo(timezone_name)

    day_of_year = local_timestamps.dt.dayofyear.astype(float)
    local_hour = (
        local_timestamps.dt.hour
        + local_timestamps.dt.minute / 60.0
        + local_timestamps.dt.second / 3600.0
    )
    utc_offset_hours = local_timestamps.apply(
        lambda value: value.to_pydatetime().replace(tzinfo=timezone).utcoffset().total_seconds() / 3600.0
    )

    fractional_year = 2.0 * math.pi / 365.0 * (day_of_year - 1.0 + (local_hour - 12.0) / 24.0)
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
    return pd.Series(np.degrees(elevation_radians), index=timestamps.index)


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


def load_rec_file_catalog(parquet_root: Path, channels: Optional[set[str]] = None) -> pd.DataFrame:
    """
    Load the cached interval parquet catalog and keep requested channel files.
    """
    selected_channels = channels or {"REC"}
    catalog_path = parquet_root / "_shape_analysis_cache" / "spid_file_index" / "file_catalog.parquet"
    csv_index_path = parquet_root / "_interval_parquet_index.csv"

    if catalog_path.exists():
        catalog = pd.read_parquet(catalog_path)
    elif csv_index_path.exists():
        catalog = pd.read_csv(csv_index_path)
        catalog = catalog.reset_index(names="FileID")
    else:
        raise FileNotFoundError(
            f"Could not find parquet file catalog at {catalog_path} or {csv_index_path}."
        )

    rec_catalog = catalog[catalog["channel"].isin(selected_channels)].copy()
    if rec_catalog.empty:
        raise ValueError(f"No {sorted(selected_channels)} parquet files found under {parquet_root}.")

    rec_catalog["month"] = pd.to_numeric(rec_catalog["month"], errors="coerce").astype("Int64")
    return rec_catalog.dropna(subset=["month", "filepath", "FileID"])


def get_available_rec_date_range(parquet_root: Path) -> tuple[date, date]:
    """
    Return the approximate available REC date range based on catalog months.
    """
    rec_catalog = load_rec_file_catalog(parquet_root)
    first_month = int(rec_catalog["month"].min())
    last_month = int(rec_catalog["month"].max())
    first_date = date(first_month // 100, first_month % 100, 1)
    last_month_start = pd.Timestamp(year=last_month // 100, month=last_month % 100, day=1)
    last_date = last_month_start + pd.offsets.MonthEnd(0)
    return first_date, last_date.date()


def load_spid_file_lookup(parquet_root: Path) -> pd.DataFrame:
    """
    Load cached ServicePointID-to-FileID mappings.
    """
    lookup_root = parquet_root / "_shape_analysis_cache" / "spid_file_index" / "lookup"
    if not lookup_root.exists():
        raise FileNotFoundError(f"Could not find service point lookup cache at {lookup_root}.")

    lookup_files = sorted(lookup_root.glob("bucket=*/*.parquet"))
    if not lookup_files:
        raise FileNotFoundError(f"No lookup parquet files found under {lookup_root}.")

    logging.info("Loading service point file lookup from %s parquet parts", len(lookup_files))
    lookup = pd.concat(
        [pd.read_parquet(path, columns=["SPID", "SPID_BASE", "FileID", "RowCount"]) for path in lookup_files],
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
) -> pd.DataFrame:
    """
    Aggregate parquet rows for active solar locations into 15-minute system export.

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
    rec_catalog = rec_catalog[rec_catalog["month"].astype(int).isin(wanted_months)].copy()
    if rec_catalog.empty:
        raise ValueError(f"No export parquet files found for months {sorted(wanted_months)}.")

    sites = sites.copy()
    sites["MeterType"] = sites["MeterType"].fillna("").astype(str).str.strip().str.upper()
    net_meter_locations = set(
        sites.loc[sites["MeterType"].isin(NET_METER_TYPES), "LocationNumber"].astype(str)
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
            rec_catalog["channel"].eq("NET")
            & rec_catalog["nem_status"].eq("NEM")
        ]
        net_file_ids = set(net_files["FileID"].astype(int))
        net_lookup = lookup[
            lookup["FileID"].astype(int).isin(net_file_ids)
            & lookup["SPID_BASE"].isin(net_meter_locations)
        ].copy()
        net_lookup["ExportSource"] = "NET_NEGATIVE"
        source_lookups.append(net_lookup)

    lookup = pd.concat(source_lookups, ignore_index=True) if source_lookups else pd.DataFrame()
    if lookup.empty:
        raise ValueError("No active solar service points found in export parquet lookup for requested dates.")

    lookup["FileID"] = lookup["FileID"].astype(int)
    dedupe_meta = rec_catalog[["FileID", "month", "modified_time_ns"]].copy()
    dedupe_meta["FileID"] = dedupe_meta["FileID"].astype(int)
    lookup = lookup.merge(dedupe_meta, on="FileID", how="left")
    lookup["month"] = pd.to_numeric(lookup["month"], errors="coerce").astype("Int64")
    lookup["modified_time_ns"] = pd.to_numeric(lookup["modified_time_ns"], errors="coerce").fillna(0)

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
        "Matched %s export service points across %s active solar locations and %s files",
        service_points,
        matched_locations,
        lookup["FileID"].nunique(),
    )

    filepath_by_id = rec_catalog.set_index("FileID")["filepath"].to_dict()
    source_by_id = lookup.groupby("FileID")["ExportSource"].first().to_dict()
    spids_by_file = lookup.groupby("FileID")["SPID"].apply(list).to_dict()
    interval_start_min = pd.Timestamp(start_date)
    interval_start_max = pd.Timestamp(end_date) + pd.Timedelta(days=1)

    pieces = []
    total_rows = 0
    total_duplicate_interval_rows = 0
    for index, (file_id, service_points_in_file) in enumerate(spids_by_file.items(), start=1):
        path = filepath_by_id.get(int(file_id))
        if not path:
            continue

        if index % 50 == 0:
            logging.info("Processed %s / %s export parquet files", index, len(spids_by_file))

        df = pd.read_parquet(
            path,
            columns=["ServicePointID", "ReadingValue_kWh", "EndTimePST"],
            filters=[("ServicePointID", "in", service_points_in_file)],
        )
        if df.empty:
            continue

        df["IntervalStartDT"] = pd.to_datetime(df["EndTimePST"], errors="coerce") - pd.Timedelta(minutes=15)
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

        duplicate_interval_rows = int(df.duplicated(["ServicePointID", "IntervalStartDT"]).sum())
        if duplicate_interval_rows:
            total_duplicate_interval_rows += duplicate_interval_rows
            df = df.drop_duplicates(["ServicePointID", "IntervalStartDT"], keep="last")

        total_rows += len(df)
        interval_piece = df.groupby("IntervalStartDT", as_index=False)["Export_kWh"].sum()
        interval_piece["ExportSource"] = export_source
        pieces.append(interval_piece)

    if not pieces:
        raise ValueError("No export interval rows found for requested dates after filtering.")

    intervals = (
        pd.concat(pieces, ignore_index=True)
        .groupby("IntervalStartDT", as_index=False)
        .agg(
            Export_kWh=("Export_kWh", "sum"),
            ExportSource=("ExportSource", lambda values: "+".join(sorted(set(values)))),
        )
        .sort_values("IntervalStartDT")
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
        "Aggregated %s unique export rows into %s system intervals; total export %.2f MWh, peak %.2f MW",
        f"{total_rows:,}",
        f"{len(intervals):,}",
        intervals["Export_kWh"].sum() / 1000.0,
        intervals["Export_kW"].max() / 1000.0,
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


def build_production_shape(interval_df: pd.DataFrame, energy_col: str) -> pd.DataFrame:
    """
    Build a normalized 15-minute daily energy shape from interval energy data.

    Kept for compatibility with older callers; new runs write the richer
    average daily shape from ``build_average_daily_shape``.
    """
    shape_df = interval_df.copy()
    shape_df["IntervalStartDT"] = pd.to_datetime(shape_df["IntervalStartDT"])
    shape_df["date"] = shape_df["IntervalStartDT"].dt.date
    shape_df["time"] = shape_df["IntervalStartDT"].dt.time
    daily_totals = shape_df.groupby("date")[energy_col].sum().reset_index(name="TotalDailyProduction_kWh")
    shape_df = pd.merge(shape_df, daily_totals, on="date")
    shape_df = shape_df[shape_df["TotalDailyProduction_kWh"] > 0]
    shape_df["NormalizedProduction"] = shape_df[energy_col] / shape_df["TotalDailyProduction_kWh"]
    production_shape = shape_df.groupby("time")["NormalizedProduction"].mean().reset_index()
    production_shape.rename(columns={"NormalizedProduction": "ProductionCoefficient"}, inplace=True)
    coefficient_sum = production_shape["ProductionCoefficient"].sum()
    if coefficient_sum <= 0:
        raise ValueError("Production shape has no positive production coefficients.")
    production_shape["ProductionCoefficient"] = production_shape["ProductionCoefficient"] / coefficient_sum
    return production_shape


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
    shape_df[power_col] = pd.to_numeric(shape_df[power_col], errors="coerce").fillna(0.0)

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
                normalized_shape = shape_df.groupby("time", as_index=False)["NormalizedPower"].median()
                reference_peak_kw = valid_daily_peaks["DailyPeak_kW"].median()
            elif method == "upper-quantile":
                normalized_shape = shape_df.groupby("time", as_index=False)["NormalizedPower"].quantile(quantile)
                reference_peak_kw = valid_daily_peaks["DailyPeak_kW"].quantile(quantile)
            else:
                raise ValueError(f"Unsupported daily shape method: {method!r}")

            average_shape = complete_shape.merge(normalized_shape, on="time", how="left")
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
    shape_df[energy_col] = pd.to_numeric(shape_df[energy_col], errors="coerce").fillna(0.0)

    hourly_totals = (
        shape_df.groupby(["date", "hour"], as_index=False)[energy_col]
        .sum()
        .rename(columns={energy_col: "TotalHourlyProduction_kWh"})
    )
    shape_df = shape_df.merge(hourly_totals, on=["date", "hour"], how="left")
    shape_df = shape_df[shape_df["TotalHourlyProduction_kWh"] > 0].copy()
    shape_df["IntraHourCoefficient"] = shape_df[energy_col] / shape_df["TotalHourlyProduction_kWh"]

    if method == "mean":
        observed_shape = (
            shape_df.groupby(["hour", "minute"], as_index=False)["IntraHourCoefficient"]
            .mean()
        )
    elif method == "median":
        observed_shape = (
            shape_df.groupby(["hour", "minute"], as_index=False)["IntraHourCoefficient"]
            .median()
        )
    elif method == "upper-quantile":
        observed_shape = (
            shape_df.groupby(["hour", "minute"], as_index=False)["IntraHourCoefficient"]
            .quantile(quantile)
        )
    else:
        raise ValueError(f"Unsupported intra-hour shape method: {method!r}")
    complete_shape = pd.MultiIndex.from_product(
        [range(24), [0, 15, 30, 45]],
        names=["hour", "minute"],
    ).to_frame(index=False)
    complete_shape = complete_shape.merge(observed_shape, on=["hour", "minute"], how="left")

    normalized_groups = []
    for hour, group in complete_shape.groupby("hour", sort=True):
        group = group.copy()
        coefficient_sum = group["IntraHourCoefficient"].sum(skipna=True)
        if pd.isna(coefficient_sum) or coefficient_sum <= 0:
            group["IntraHourCoefficient"] = 0.25
        else:
            group["IntraHourCoefficient"] = group["IntraHourCoefficient"].fillna(0.0) / coefficient_sum
        normalized_groups.append(group)

    intrahour_shape = pd.concat(normalized_groups, ignore_index=True)
    intrahour_shape["ShapeMethod"] = method
    intrahour_shape["ShapeQuantile"] = quantile if method == "upper-quantile" else pd.NA
    return intrahour_shape


def get_default_rec_history_window(parquet_root: Path, history_months: int) -> tuple[date, date]:
    """
    Pick the most recent complete REC parquet months for calibration.
    """
    _, available_end = get_available_rec_date_range(parquet_root)
    history_start_month = add_months(date(available_end.year, available_end.month, 1), -(history_months - 1))
    return history_start_month, available_end


# =============================================================================
# Weather Data
# =============================================================================

def shortwave_radiation_to_kwh_per_m2(values: pd.Series, unit: str) -> pd.Series:
    """
    Convert Open-Meteo shortwave_radiation_sum values to kWh/m^2.
    """
    normalized_unit = (unit or "").replace("\u00b2", "2").replace("^2", "2").strip().lower()
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
    normalized_unit = (unit or "").replace("\u00b2", "2").replace("^2", "2").strip().lower()
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


def fetch_historical_weather(
    sites: pd.DataFrame,
    start_date: date,
    end_date: date,
    cache_dir: str | Path | None = DEFAULT_SOLAR_WEATHER_CACHE_DIR,
    use_cache: bool = True,
) -> pd.DataFrame:
    """
    Fetch historical GHI from Open-Meteo for a list of sites.
    """
    return fetch_open_meteo_weather(
        sites,
        start_date,
        end_date,
        use_forecast=False,
        cache_dir=cache_dir,
        use_cache=use_cache,
    )


def fetch_forecast_weather(
    sites: pd.DataFrame,
    start_date: date,
    end_date: date,
    cache_dir: str | Path | None = DEFAULT_SOLAR_WEATHER_CACHE_DIR,
    forecast_cache_max_age_hours: float = DEFAULT_FORECAST_WEATHER_CACHE_MAX_AGE_HOURS,
    use_cache: bool = True,
) -> pd.DataFrame:
    """
    Fetch forecast GHI from Open-Meteo for a list of sites.
    """
    return fetch_open_meteo_weather(
        sites,
        start_date,
        end_date,
        use_forecast=True,
        cache_dir=cache_dir,
        forecast_cache_max_age_hours=forecast_cache_max_age_hours,
        use_cache=use_cache,
    )


def fetch_open_meteo_weather(
    sites: pd.DataFrame,
    start_date: date,
    end_date: date,
    use_forecast: bool,
    cache_dir: str | Path | None = DEFAULT_SOLAR_WEATHER_CACHE_DIR,
    forecast_cache_max_age_hours: float = DEFAULT_FORECAST_WEATHER_CACHE_MAX_AGE_HOURS,
    use_cache: bool = True,
) -> pd.DataFrame:
    """
    Fetch daily GHI from Open-Meteo archive or forecast API.
    The Open-Meteo forecast API is limited to 16 days.
    """
    source_name = "forecast" if use_forecast else "historical"
    logging.info("Fetching %s weather data from %s to %s", source_name, start_date, end_date)
    cache_path = _solar_weather_cache_path(
        cache_dir=cache_dir,
        kind="daily",
        source_name=source_name,
        start_date=start_date,
        end_date=end_date,
        sites=sites,
        timezone_name="auto",
    )
    cached = pd.DataFrame()
    if use_cache:
        cached = _read_solar_weather_cache(
            cache_path,
            start_date=start_date,
            end_date=end_date,
            timestamp_col="date",
        )
        if not cached.empty and (
            not use_forecast or _solar_weather_cache_is_fresh(cache_path, forecast_cache_max_age_hours)
        ):
            logging.info("Using cached %s daily solar weather: %s", source_name, cache_path)
            return cached

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
                "Open-Meteo %s daily solar weather refresh failed; using stale cache %s. Details: %s",
                source_name,
                cache_path,
                exc,
            )
            return cached
        raise

    # If only one site is requested, the API returns a dict, not a list of dicts.
    # Wrap it in a list to handle this case.
    if isinstance(results, dict):
        results = [results]

    all_weather_data = []
    for i, site_weather in enumerate(results):
        site_id = sites.iloc[i]["SolarSiteKey"]
        temp_df = pd.DataFrame(site_weather["daily"])
        temp_df["SolarSiteKey"] = site_id
        radiation_unit = site_weather.get("daily_units", {}).get("shortwave_radiation_sum", "")
        temp_df["GHI_kWh_per_m2"] = shortwave_radiation_to_kwh_per_m2(
            temp_df["shortwave_radiation_sum"],
            radiation_unit,
        )
        all_weather_data.append(temp_df)

    weather_df = pd.concat(all_weather_data, ignore_index=True)
    weather_df.rename(columns={"time": "date"}, inplace=True)
    weather_df["date"] = pd.to_datetime(weather_df["date"]).dt.date
    if use_cache:
        _write_solar_weather_cache(weather_df, cache_path)
        if use_forecast:
            _archive_solar_forecast_weather(weather_df, cache_dir, cache_path)

    logging.info("Fetched %s weather data points", len(weather_df))
    return weather_df


def fetch_open_meteo_hourly_weather(
    sites: pd.DataFrame,
    start_date: date,
    end_date: date,
    use_forecast: bool,
    timezone_name: str,
    cache_dir: str | Path | None = DEFAULT_SOLAR_WEATHER_CACHE_DIR,
    forecast_cache_max_age_hours: float = DEFAULT_FORECAST_WEATHER_CACHE_MAX_AGE_HOURS,
    use_cache: bool = True,
) -> pd.DataFrame:
    """
    Fetch hourly GHI and cloud cover from Open-Meteo archive or forecast API.
    """
    source_name = "forecast" if use_forecast else "historical"
    logging.info("Fetching hourly %s weather data from %s to %s", source_name, start_date, end_date)
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
            not use_forecast or _solar_weather_cache_is_fresh(cache_path, forecast_cache_max_age_hours)
        ):
            logging.info("Using cached %s hourly solar weather: %s", source_name, cache_path)
            return cached

    url = (
        "https://api.open-meteo.com/v1/forecast"
        if use_forecast
        else "https://archive-api.open-meteo.com/v1/archive"
    )
    all_weather_data = []
    site_batches = [
        sites.iloc[i : i + MAX_SITES_PER_WEATHER_REQUEST]
        for i in range(0, len(sites), MAX_SITES_PER_WEATHER_REQUEST)
    ]

    try:
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
            }

            results = _open_meteo_get_json(
                url,
                params,
                source_name=source_name,
                start_date=start_date,
                end_date=end_date,
            )
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
                temp_df["IntervalStartDT"] = pd.to_datetime(temp_df["IntervalStartDT"], errors="coerce")

                hourly_units = site_weather.get("hourly_units", {})
                radiation_unit = hourly_units.get("shortwave_radiation", "W/m\u00b2")
                temp_df["WeatherGHI_Wm2"] = pd.to_numeric(temp_df.get("shortwave_radiation"), errors="coerce")
                temp_df["GHI_kWh_per_m2"] = hourly_irradiance_to_kwh_per_m2(
                    temp_df["shortwave_radiation"],
                    radiation_unit,
                )
                temp_df["DirectRadiation_Wm2"] = pd.to_numeric(temp_df.get("direct_radiation"), errors="coerce")
                temp_df["DiffuseRadiation_Wm2"] = pd.to_numeric(temp_df.get("diffuse_radiation"), errors="coerce")
                temp_df["Temperature_C"] = pd.to_numeric(temp_df.get("temperature_2m"), errors="coerce")
                temp_df["WindSpeed_ms"] = pd.to_numeric(temp_df.get("wind_speed_10m"), errors="coerce")

                cloud_columns = {
                    "cloud_cover": "CloudCoverPct",
                    "cloud_cover_low": "CloudCoverLowPct",
                    "cloud_cover_mid": "CloudCoverMidPct",
                    "cloud_cover_high": "CloudCoverHighPct",
                }
                for source_col, target_col in cloud_columns.items():
                    if source_col in temp_df.columns:
                        temp_df[target_col] = pd.to_numeric(temp_df[source_col], errors="coerce")
                    else:
                        temp_df[target_col] = pd.NA

                temp_df["date"] = temp_df["IntervalStartDT"].dt.date
                keep_columns = [
                    "SolarSiteKey",
                    "IntervalStartDT",
                    "date",
                    "GHI_kWh_per_m2",
                    "WeatherGHI_Wm2",
                    "DirectRadiation_Wm2",
                    "DiffuseRadiation_Wm2",
                    "Temperature_C",
                    "WindSpeed_ms",
                    "CloudCoverPct",
                    "CloudCoverLowPct",
                    "CloudCoverMidPct",
                    "CloudCoverHighPct",
                ]
                temp_df = temp_df[keep_columns].dropna(subset=["IntervalStartDT", "GHI_kWh_per_m2"])
                temp_df = temp_df[
                    (temp_df["date"] >= start_date)
                    & (temp_df["date"] <= end_date)
                ]
                all_weather_data.append(temp_df)

            if len(site_batches) > 1 and batch_num < len(site_batches):
                time.sleep(1)
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

    if not all_weather_data:
        raise ValueError(f"No hourly weather data returned from Open-Meteo for {start_date} to {end_date}.")

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
    cache_dir: str | Path | None = DEFAULT_SOLAR_WEATHER_CACHE_DIR,
    forecast_cache_max_age_hours: float = DEFAULT_FORECAST_WEATHER_CACHE_MAX_AGE_HOURS,
    use_cache: bool = True,
) -> pd.DataFrame:
    """
    Fetch archive data for past dates and forecast data for today/future dates.
    """
    today = date.today()
    frames = []

    archive_end = min(end_date, today - timedelta(days=1))
    if start_date <= archive_end:
        frames.append(fetch_historical_weather(sites, start_date, archive_end, cache_dir=cache_dir, use_cache=use_cache))

    forecast_start = max(start_date, today)
    if forecast_start <= end_date:
        frames.append(
            fetch_forecast_weather(
                sites,
                forecast_start,
                end_date,
                cache_dir=cache_dir,
                forecast_cache_max_age_hours=forecast_cache_max_age_hours,
                use_cache=use_cache,
            )
        )

    if not frames:
        frames.append(fetch_historical_weather(sites, start_date, end_date, cache_dir=cache_dir, use_cache=use_cache))

    return pd.concat(frames, ignore_index=True)


def fetch_hourly_weather_for_date_range(
    sites: pd.DataFrame,
    start_date: date,
    end_date: date,
    timezone_name: str,
    cache_dir: str | Path | None = DEFAULT_SOLAR_WEATHER_CACHE_DIR,
    forecast_cache_max_age_hours: float = DEFAULT_FORECAST_WEATHER_CACHE_MAX_AGE_HOURS,
    use_cache: bool = True,
) -> pd.DataFrame:
    """
    Fetch hourly archive data for past dates and hourly forecast data for today/future dates.
    """
    today = current_local_timestamp(timezone_name).date()
    frames = []

    archive_end = min(end_date, today - timedelta(days=1))
    if start_date <= archive_end:
        frames.append(
            fetch_open_meteo_hourly_weather(
                sites,
                start_date,
                archive_end,
                False,
                timezone_name,
                cache_dir=cache_dir,
                use_cache=use_cache,
            )
        )

    forecast_start = max(start_date, today)
    if forecast_start <= end_date:
        frames.append(
            fetch_open_meteo_hourly_weather(
                sites,
                forecast_start,
                end_date,
                True,
                timezone_name,
                cache_dir=cache_dir,
                forecast_cache_max_age_hours=forecast_cache_max_age_hours,
                use_cache=use_cache,
            )
        )

    if not frames:
        frames.append(
            fetch_open_meteo_hourly_weather(
                sites,
                start_date,
                end_date,
                False,
                timezone_name,
                cache_dir=cache_dir,
                use_cache=use_cache,
            )
        )

    return pd.concat(frames, ignore_index=True)


# =============================================================================
# Forecast Model
# =============================================================================

def build_system_weather_site(latitude: float, longitude: float) -> pd.DataFrame:
    """
    Use a representative Roseville point for system-wide weather.
    """
    return pd.DataFrame([{
        "SolarSiteKey": 1,
        "LocationNumber": "Roseville, CA",
        "Latitude": latitude,
        "Longitude": longitude,
    }])


def build_weather_clusters(sites: pd.DataFrame, n_clusters: int) -> tuple[pd.DataFrame, pd.DataFrame]:
    """
    Group sites into geographic clusters for weather forecasting using K-Means.
    """
    sites_with_coords = sites.dropna(subset=["Latitude", "Longitude"]).copy()
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
    sites_with_coords["WeatherCluster"] = kmeans.fit_predict(sites_with_coords[["Latitude", "Longitude"]])
    out_sites = sites.merge(sites_with_coords[["SolarSiteKey", "WeatherCluster"]], on="SolarSiteKey", how="left")

    cluster_centers = pd.DataFrame(kmeans.cluster_centers_, columns=["Latitude", "Longitude"])
    cluster_centers["SolarSiteKey"] = cluster_centers.index
    cluster_centers["LocationNumber"] = [f"Weather Cluster {i}" for i in range(n_clusters)]
    return out_sites, cluster_centers


def weighted_weather_average(group: pd.DataFrame, weather_columns: list[str]) -> pd.Series:
    """
    Capacity-weight weather columns for one timestamp.
    """
    weights = pd.to_numeric(group["SolarCECkW"], errors="coerce")
    valid_weight = weights.notna() & weights.gt(0)
    values = {}
    for column in weather_columns:
        series = pd.to_numeric(group[column], errors="coerce")
        valid = valid_weight & series.notna()
        if valid.any():
            values[column] = float((series[valid] * weights[valid]).sum() / weights[valid].sum())
        else:
            values[column] = pd.NA
    return pd.Series(values)


def aggregate_capacity_weighted_weather(weather_df: pd.DataFrame, sites: pd.DataFrame) -> pd.DataFrame:
    """
    Convert per-site or per-cluster weather rows to one capacity-weighted system series.
    """
    if weather_df.empty:
        return weather_df

    if "WeatherCluster" in sites.columns:
        weight_lookup = (
            sites.dropna(subset=["WeatherCluster"])
            .assign(SolarCECkW=lambda frame: pd.to_numeric(frame["SolarCECkW"], errors="coerce"))
            .groupby("WeatherCluster", as_index=False)["SolarCECkW"]
            .sum()
        )
        site_weather = weather_df.rename(columns={"SolarSiteKey": "WeatherCluster"}).merge(
            weight_lookup,
            on="WeatherCluster",
            how="left",
        )
    else:
        site_weather = weather_df.merge(
            sites[["SolarSiteKey", "SolarCECkW"]],
            on="SolarSiteKey",
            how="left",
        )

    keep_columns = ["IntervalStartDT", "SolarCECkW", *[col for col in WEATHER_OUTPUT_COLUMNS if col in site_weather.columns]]
    site_weather = site_weather[keep_columns].dropna(subset=["IntervalStartDT"]).copy()
    if site_weather.empty:
        return pd.DataFrame(columns=["IntervalStartDT", *WEATHER_OUTPUT_COLUMNS])

    site_weather["IntervalStartDT"] = pd.to_datetime(site_weather["IntervalStartDT"], errors="coerce")
    site_weather = site_weather.dropna(subset=["IntervalStartDT"])
    if site_weather.empty:
        return pd.DataFrame(columns=["IntervalStartDT", *WEATHER_OUTPUT_COLUMNS])

    weights = pd.to_numeric(site_weather.get("SolarCECkW"), errors="coerce").clip(lower=0.0)
    site_weather["_WeatherWeight"] = weights.fillna(0.0)
    out = (
        site_weather[["IntervalStartDT"]]
        .drop_duplicates()
        .sort_values("IntervalStartDT")
        .reset_index(drop=True)
    )

    for column in WEATHER_OUTPUT_COLUMNS:
        if column not in site_weather.columns:
            out[column] = np.nan
            continue

        values = pd.to_numeric(site_weather[column], errors="coerce")
        valid = values.notna() & site_weather["_WeatherWeight"].gt(0.0)
        weighted = site_weather.loc[valid, ["IntervalStartDT", "_WeatherWeight"]].copy()
        weighted["_WeightedValue"] = values.loc[valid] * weighted["_WeatherWeight"]
        numerator = weighted.groupby("IntervalStartDT", sort=False)["_WeightedValue"].sum()
        denominator = weighted.groupby("IntervalStartDT", sort=False)["_WeatherWeight"].sum()
        averaged = (numerator / denominator).replace([np.inf, -np.inf], np.nan)
        out = out.merge(
            averaged.rename(column).reset_index(),
            on="IntervalStartDT",
            how="left",
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


def add_performance_features(
    df: pd.DataFrame,
    latitude: float,
    longitude: float,
    timezone_name: str,
) -> pd.DataFrame:
    """
    Add weather/time features used by the performance-ratio model.
    """
    out = df.copy()
    out["IntervalStartDT"] = pd.to_datetime(out["IntervalStartDT"])
    for column in WEATHER_OUTPUT_COLUMNS:
        if column not in out.columns:
            out[column] = np.nan
        out[column] = pd.to_numeric(out[column], errors="coerce")

    out["GHI_kWh_per_m2"] = out["GHI_kWh_per_m2"].clip(lower=0)
    out["WeatherGHI_Wm2"] = out["WeatherGHI_Wm2"].fillna(out["GHI_kWh_per_m2"] * 1000.0)
    out["DirectRadiation_Wm2"] = out["DirectRadiation_Wm2"].clip(lower=0.0).fillna(0.0)
    out["DiffuseRadiation_Wm2"] = out["DiffuseRadiation_Wm2"].clip(lower=0.0).fillna(0.0)
    out["Temperature_C"] = out["Temperature_C"].fillna(DEFAULT_TEMPERATURE_C)
    out["WindSpeed_ms"] = out["WindSpeed_ms"].clip(lower=0.0).fillna(0.0)
    for column in ["CloudCoverPct", "CloudCoverLowPct", "CloudCoverMidPct", "CloudCoverHighPct"]:
        out[column] = out[column].clip(lower=0, upper=100).fillna(0.0)

    out["hour"] = out["IntervalStartDT"].dt.hour
    day_of_year = out["IntervalStartDT"].dt.dayofyear
    out["HourSin"] = np.sin(2.0 * np.pi * out["hour"] / 24.0)
    out["HourCos"] = np.cos(2.0 * np.pi * out["hour"] / 24.0)
    out["DayOfYearSin"] = np.sin(2.0 * np.pi * day_of_year / 366.0)
    out["DayOfYearCos"] = np.cos(2.0 * np.pi * day_of_year / 366.0)
    out["SolarElevationDeg"] = calculate_solar_elevation_degrees(
        out["IntervalStartDT"],
        latitude=latitude,
        longitude=longitude,
        timezone_name=timezone_name,
    )
    cos_zenith = np.sin(np.radians(out["SolarElevationDeg"].clip(lower=0.0)))
    clear_sky_ghi = pd.Series(0.0, index=out.index, dtype="float64")
    positive_sun_mask = cos_zenith > 0
    if positive_sun_mask.any():
        clear_sky_ghi.loc[positive_sun_mask] = (
            1098.0
            * cos_zenith.loc[positive_sun_mask]
            * np.exp(-0.059 / cos_zenith.loc[positive_sun_mask])
        )
    out["ClearSkyGHI_Wm2"] = clear_sky_ghi.clip(lower=0.0)
    clear_sky_denom = out["ClearSkyGHI_Wm2"].replace(0.0, np.nan)
    out["ClearSkyIndex"] = (
        out["WeatherGHI_Wm2"] / clear_sky_denom
    ).replace([np.inf, -np.inf], np.nan).clip(lower=0.0, upper=1.25).fillna(0.0)
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
        series.astype(str)
        .str.strip()
        .str.lower()
        .isin({"1", "true", "t", "yes", "y"})
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
    min_expected_kwh: float = DEFAULT_ACTUAL_QUALITY_MIN_EXPECTED_KWH,
    min_ghi_kwh_m2: float = DEFAULT_ACTUAL_QUALITY_MIN_GHI_KWH_M2,
    min_clear_sky_index: float = DEFAULT_ACTUAL_QUALITY_MIN_CLEAR_SKY_INDEX,
    min_bad_hours_per_day: int = DEFAULT_ACTUAL_QUALITY_MIN_BAD_HOURS_PER_DAY,
) -> pd.DataFrame:
    """
    Flag AMI-suppressed actuals that are physically inconsistent with clear-sky solar.

    The rule is intentionally narrow: only high-expected, high-irradiance rows
    with actuals near zero are flagged, and repeated bad hours mark the full
    operating day so partial AMI outages do not leak into calibration.
    """
    out = df.copy()
    if out.empty:
        out["ActualQualityFlag"] = pd.Series(dtype="object")
        out["SolarBacktestExcluded"] = pd.Series(dtype="bool")
        out["ActualToExpectedRatio"] = pd.Series(dtype="float64")
        out["ActualQualityExpected_kWh"] = pd.Series(dtype="float64")
        out["ActualQualitySuspiciousHour"] = pd.Series(dtype="bool")
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

    expected_denom = expected_kwh.where(expected_kwh > 0)
    ratio = (actual_kwh / expected_denom).replace([np.inf, -np.inf], np.nan)
    high_expected = expected_kwh >= min_expected_kwh
    high_irradiance = ghi >= min_ghi_kwh_m2
    clear_sky = clear_sky_index >= min_clear_sky_index
    suspicious_hour = (
        high_expected
        & high_irradiance
        & clear_sky
        & ratio.notna()
        & (ratio <= actual_to_expected_ratio_threshold)
    )

    dates = out["IntervalStartDT"].dt.date
    bad_hour_counts = suspicious_hour.groupby(dates).sum()
    bad_dates = set(bad_hour_counts[bad_hour_counts >= max(1, int(min_bad_hours_per_day))].index)
    suppressed_day = dates.isin(bad_dates)
    quality_excluded = suspicious_hour | suppressed_day

    existing_excluded = _actual_quality_exclusion_mask(out)
    out["SolarBacktestExcluded"] = existing_excluded | quality_excluded
    if "ActualQualityFlag" not in out.columns:
        out["ActualQualityFlag"] = ACTUAL_QUALITY_OK
    out["ActualQualityFlag"] = out["ActualQualityFlag"].fillna(ACTUAL_QUALITY_OK).astype(str)
    out.loc[quality_excluded, "ActualQualityFlag"] = ACTUAL_QUALITY_AMI_SUPPRESSED
    out["ActualToExpectedRatio"] = ratio
    out["ActualQualityExpected_kWh"] = expected_kwh
    out["ActualQualitySuspiciousHour"] = suspicious_hour.fillna(False).astype(bool)
    return out


def calibrate_performance_ratio(
    rec_intervals: pd.DataFrame,
    weather_df: pd.DataFrame,
    capacity_kw: float,
    fallback_ratio: float,
) -> float:
    """
    Calibrate an export performance ratio from REC export kWh and historical GHI.
    """
    daily_export = rec_intervals.copy()
    daily_export["date"] = pd.to_datetime(daily_export["IntervalStartDT"]).dt.date
    daily_export = daily_export.groupby("date", as_index=False)["Export_kWh"].sum()

    weather_for_calibration = weather_df.copy()
    if "date" not in weather_for_calibration.columns:
        weather_for_calibration["date"] = pd.to_datetime(weather_for_calibration["IntervalStartDT"]).dt.date
    daily_weather = (
        weather_for_calibration.groupby("date", as_index=False)["GHI_kWh_per_m2"]
        .sum()
    )

    calibration = daily_export.merge(daily_weather, on="date", how="inner")
    calibration["ModeledAvailable_kWh"] = calibration["GHI_kWh_per_m2"] * capacity_kw
    calibration = calibration[
        (calibration["Export_kWh"] > 0)
        & (calibration["ModeledAvailable_kWh"] > 0)
    ].copy()

    if calibration.empty:
        logging.warning(
            "No overlapping REC/weather calibration days found; using fallback performance ratio %.3f",
            fallback_ratio,
        )
        return fallback_ratio

    calibration["PerformanceRatio"] = (
        calibration["Export_kWh"] / calibration["ModeledAvailable_kWh"]
    ).clip(lower=0.0, upper=1.0)
    calibrated_ratio = float(calibration["PerformanceRatio"].median())
    logging.info(
        "Calibrated solar export performance ratio %.3f from %s days",
        calibrated_ratio,
        len(calibration),
    )
    return calibrated_ratio


def train_performance_model(
    rec_intervals: pd.DataFrame,
    weather_df: pd.DataFrame,
    capacity_kw: float,
    fallback_ratio: float,
    latitude: float,
    longitude: float,
    timezone_name: str,
    daily_active_capacity: Optional[pd.DataFrame] = None,
    max_performance_ratio: float = DEFAULT_MAX_PERFORMANCE_RATIO,
    min_training_available_kwh: float = 25.0,
    min_training_rows: int = 24,
    use_energy_weighting: bool = True,
    exclude_suppressed_actuals: bool = True,
) -> PerformanceModel:
    """
    Train a bounded model that predicts export performance ratio from weather and seasonality.
    """
    hourly_export = (
        rec_intervals.copy()
        .set_index("IntervalStartDT")
        .resample("h")
        .agg(Export_kWh=("Export_kWh", "sum"))
        .reset_index()
    )
    hourly_weather = add_performance_features(
        aggregate_weather_to_hourly(weather_df),
        latitude=latitude,
        longitude=longitude,
        timezone_name=timezone_name,
    )

    training_data = pd.merge(hourly_export, hourly_weather, on="IntervalStartDT", how="inner")
    training_data["CapacityForRow_kW"] = _resolve_row_capacity(
        training_data["IntervalStartDT"],
        daily_active_capacity,
        capacity_kw,
    )
    if daily_active_capacity is not None and not daily_active_capacity.empty:
        logging.info(
            "Normalizing performance-ratio training by daily active capacity (%.0f-%.0f kW)",
            float(daily_active_capacity["ActiveCapacity_kW"].min()),
            float(daily_active_capacity["ActiveCapacity_kW"].max()),
        )
    training_data["ModeledAvailable_kWh"] = (
        training_data["GHI_kWh_per_m2"] * training_data["CapacityForRow_kW"]
    )
    training_data = training_data[
        (training_data["ModeledAvailable_kWh"] >= min_training_available_kwh)
        & training_data["Export_kWh"].notna()
    ].copy()
    if exclude_suppressed_actuals and not training_data.empty:
        training_data = add_solar_actual_quality_flags(
            training_data,
            actual_kwh_col="Export_kWh",
            expected_kwh_col="ModeledAvailable_kWh",
            actual_to_expected_ratio_threshold=DEFAULT_ACTUAL_QUALITY_AVAILABLE_RATIO_THRESHOLD,
        )
        excluded_rows = int(_actual_quality_exclusion_mask(training_data).sum())
        if excluded_rows:
            excluded_days = training_data.loc[
                _actual_quality_exclusion_mask(training_data),
                "IntervalStartDT",
            ].dt.date.nunique()
            logging.info(
                "Excluded %s AMI-suppressed daylight rows across %s days from performance-ratio training.",
                excluded_rows,
                excluded_days,
            )
            training_data = training_data[~_actual_quality_exclusion_mask(training_data)].copy()

    if training_data.empty:
        logging.warning(
            "No usable model training rows found; using fallback performance ratio %.3f",
            fallback_ratio,
        )
        return PerformanceModel(None, fallback_ratio, PERFORMANCE_FEATURE_COLUMNS, max_performance_ratio)

    training_data["PerformanceRatio"] = (
        training_data["Export_kWh"] / training_data["ModeledAvailable_kWh"]
    ).clip(lower=0.0, upper=max_performance_ratio)
    learned_fallback = float(training_data["PerformanceRatio"].median())
    if pd.isna(learned_fallback) or learned_fallback <= 0:
        learned_fallback = fallback_ratio

    if len(training_data) < min_training_rows:
        logging.warning(
            "Only %s usable model training rows found; using median performance ratio %.3f",
            len(training_data),
            learned_fallback,
        )
        return PerformanceModel(None, learned_fallback, PERFORMANCE_FEATURE_COLUMNS, max_performance_ratio)

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
        sample_weight = training_data["Export_kWh"].clip(lower=min_training_available_kwh)
    model.fit(X, y, sample_weight=sample_weight)

    fitted = pd.Series(model.predict(X), index=training_data.index).clip(0.0, max_performance_ratio)
    fitted_kwh = training_data["ModeledAvailable_kWh"] * fitted
    actual_kwh_sum = training_data["Export_kWh"].sum()
    wmape = (
        (fitted_kwh - training_data["Export_kWh"]).abs().sum() / actual_kwh_sum
        if actual_kwh_sum > 0
        else np.nan
    )
    logging.info(
        "Trained performance model on %s hourly daylight rows; median ratio %.3f, "
        "energy weighting %s, in-sample daylight WMAPE %.2f%%",
        len(training_data),
        learned_fallback,
        "enabled" if use_energy_weighting else "disabled",
        wmape * 100 if pd.notna(wmape) else float("nan"),
    )
    return PerformanceModel(model, learned_fallback, PERFORMANCE_FEATURE_COLUMNS, max_performance_ratio)


def predict_performance_ratio(model: PerformanceModel, feature_df: pd.DataFrame) -> pd.Series:
    """
    Predict bounded performance ratio for weather feature rows.
    """
    if model.estimator is None:
        return pd.Series(model.fallback_ratio, index=feature_df.index)

    ratio = pd.Series(
        model.estimator.predict(feature_df[model.feature_columns].fillna(0.0)),
        index=feature_df.index,
    )
    return ratio.clip(lower=0.0, upper=model.upper_bound).fillna(model.fallback_ratio)


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
) -> pd.DataFrame:
    """
    Add weather, time, and base-forecast features used by the residual calibrator.
    """
    out = add_performance_features(
        df,
        latitude=latitude,
        longitude=longitude,
        timezone_name=timezone_name,
    )

    for column in ["Forecast_kWh", "Forecast_kW", "Forecast_MW", "CapacityFactor", "PerformanceRatio"]:
        if column not in out.columns:
            out[column] = np.nan
        out[column] = pd.to_numeric(out[column], errors="coerce")

    if not out["Forecast_MW"].isna().all():
        out["Forecast_kW"] = out["Forecast_kW"].fillna(out["Forecast_MW"] * 1000.0)
    if out["Forecast_kW"].isna().any() and not out["Forecast_kWh"].isna().all():
        out["Forecast_kW"] = out["Forecast_kW"].fillna(out["Forecast_kWh"] / INTERVAL_HOURS)
    if out["Forecast_MW"].isna().any() and not out["Forecast_kW"].isna().all():
        out["Forecast_MW"] = out["Forecast_MW"].fillna(out["Forecast_kW"] / 1000.0)
    if out["Forecast_kWh"].isna().any() and not out["Forecast_kW"].isna().all():
        out["Forecast_kWh"] = out["Forecast_kWh"].fillna(out["Forecast_kW"] * INTERVAL_HOURS)

    if total_capacity_kw > 0:
        out["CapacityFactor"] = out["CapacityFactor"].fillna(out["Forecast_kW"] / total_capacity_kw)
    out["CapacityFactor"] = out["CapacityFactor"].clip(lower=0.0)
    out["PerformanceRatio"] = out["PerformanceRatio"].fillna(0.0).clip(lower=0.0, upper=1.0)
    return out


def predict_residual_calibration_factor_from_estimator(
    estimator: GradientBoostingRegressor,
    feature_df: pd.DataFrame,
    feature_columns: list[str],
    total_capacity_kw: float,
    latitude: float,
    longitude: float,
    timezone_name: str,
    lower_bound: float,
    upper_bound: float,
    fallback_factor: float,
) -> pd.Series:
    """
    Predict residual factors from an estimator before the model dataclass is built.
    """
    if feature_df.empty:
        return pd.Series(dtype="float64")

    calibration_features = add_residual_calibration_features(
        feature_df,
        total_capacity_kw=total_capacity_kw,
        latitude=latitude,
        longitude=longitude,
        timezone_name=timezone_name,
    )
    factors = pd.Series(
        estimator.predict(calibration_features[feature_columns].fillna(0.0)),
        index=feature_df.index,
    )
    return factors.clip(lower_bound, upper_bound).fillna(fallback_factor)


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
    calibration_data["IntervalStartDT"] = pd.to_datetime(calibration_data["IntervalStartDT"])
    calibration_data["Forecast_kWh"] = pd.to_numeric(calibration_data["Forecast_kWh"], errors="coerce")
    calibration_data["Actual_kWh"] = pd.to_numeric(calibration_data["Actual_kWh"], errors="coerce")
    calibration_data["ResidualCalibrationFactor"] = residual_factors.reindex(calibration_data.index).fillna(1.0)
    calibration_data["ResidualForecast_kWh"] = (
        calibration_data["Forecast_kWh"] * calibration_data["ResidualCalibrationFactor"]
    )
    calibration_data = calibration_data.dropna(subset=["Actual_kWh", "ResidualForecast_kWh"])
    calibration_data = calibration_data[
        (calibration_data["Actual_kWh"] >= 0)
        & (calibration_data["ResidualForecast_kWh"] > 0)
    ].copy()
    excluded_mask = _actual_quality_exclusion_mask(calibration_data)
    if bool(excluded_mask.any()):
        logging.info(
            "Excluded %s AMI-suppressed daylight rows from seasonal calibration factors.",
            int(excluded_mask.sum()),
        )
        calibration_data = calibration_data[~excluded_mask].copy()
    if calibration_data.empty:
        logging.info("Seasonal calibration skipped; no positive residual forecast rows are available.")
        return {}, 1.0

    aggregate_factor = calibration_data["Actual_kWh"].sum() / calibration_data["ResidualForecast_kWh"].sum()
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
        factor = (
            (row["Actual_kWh"] + prior_kwh * aggregate_factor)
            / (row["ResidualForecast_kWh"] + prior_kwh)
        )
        seasonal_factors[int(row["Month"])] = float(np.clip(factor, lower_bound, upper_bound))

    logging.info(
        "Built seasonal calibration factors: %s",
        ", ".join(f"{month}={factor:.3f}" for month, factor in sorted(seasonal_factors.items())),
    )
    return seasonal_factors, aggregate_factor


def train_residual_calibration_model(
    backtest_df: pd.DataFrame,
    total_capacity_kw: float,
    latitude: float,
    longitude: float,
    timezone_name: str,
    lower_bound: float,
    upper_bound: float,
    min_training_forecast_kwh: float,
    min_training_rows: int,
    use_energy_weighting: bool,
    use_seasonal_calibration: bool,
    seasonal_prior_mwh: float,
    seasonal_lower_bound: float,
    seasonal_upper_bound: float,
) -> ResidualCalibrationModel:
    """
    Train a bounded model that corrects repeatable actual-vs-forecast residual bias.
    """
    model = identity_residual_calibration_model(lower_bound=lower_bound, upper_bound=upper_bound)
    if backtest_df.empty:
        logging.info("Residual calibration skipped; no backtest rows are available.")
        return model

    training_data = backtest_df.copy()
    training_data = training_data.dropna(subset=["Actual_kWh", "Forecast_kWh"])
    training_data = training_data[
        (training_data["Actual_kWh"] >= 0)
        & (training_data["Forecast_kWh"] >= min_training_forecast_kwh)
    ].copy()
    excluded_mask = _actual_quality_exclusion_mask(training_data)
    if bool(excluded_mask.any()):
        logging.info(
            "Excluded %s AMI-suppressed daylight rows from residual calibration training.",
            int(excluded_mask.sum()),
        )
        training_data = training_data[~excluded_mask].copy()
    if training_data.empty:
        logging.info("Residual calibration skipped; no daylight forecast rows met the training threshold.")
        return model

    aggregate_factor = training_data["Actual_kWh"].sum() / training_data["Forecast_kWh"].sum()
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

    feature_data = add_residual_calibration_features(
        training_data,
        total_capacity_kw=total_capacity_kw,
        latitude=latitude,
        longitude=longitude,
        timezone_name=timezone_name,
    )
    estimator = GradientBoostingRegressor(
        n_estimators=120,
        learning_rate=0.04,
        max_depth=2,
        min_samples_leaf=12,
        subsample=0.85,
        random_state=84,
    )
    X = feature_data[CALIBRATION_FEATURE_COLUMNS].fillna(0.0)
    y = training_data["ResidualCalibrationFactor"]
    sample_weight = None
    if use_energy_weighting:
        sample_weight = training_data["Forecast_kWh"].clip(lower=min_training_forecast_kwh)
    estimator.fit(X, y, sample_weight=sample_weight)

    fitted_factor = pd.Series(estimator.predict(X), index=training_data.index).clip(lower_bound, upper_bound)
    fitted_kwh = training_data["Forecast_kWh"] * fitted_factor
    seasonal_factors, seasonal_default_factor = build_seasonal_calibration_factors(
        backtest_df=backtest_df,
        residual_factors=predict_residual_calibration_factor_from_estimator(
            estimator=estimator,
            feature_df=backtest_df,
            feature_columns=CALIBRATION_FEATURE_COLUMNS,
            total_capacity_kw=total_capacity_kw,
            latitude=latitude,
            longitude=longitude,
            timezone_name=timezone_name,
            lower_bound=lower_bound,
            upper_bound=upper_bound,
            fallback_factor=aggregate_factor,
        ),
        use_seasonal_calibration=use_seasonal_calibration,
        prior_mwh=seasonal_prior_mwh,
        lower_bound=seasonal_lower_bound,
        upper_bound=seasonal_upper_bound,
    )
    if use_seasonal_calibration and seasonal_factors:
        fitted_month_factor = training_data["IntervalStartDT"].dt.month.map(seasonal_factors).fillna(
            seasonal_default_factor
        )
        fitted_kwh = fitted_kwh * fitted_month_factor

    actual_kwh_sum = training_data["Actual_kWh"].sum()
    calibrated_wmape = (
        (fitted_kwh - training_data["Actual_kWh"]).abs().sum() / actual_kwh_sum
        if actual_kwh_sum > 0
        else np.nan
    )
    base_wmape = (
        (training_data["Forecast_kWh"] - training_data["Actual_kWh"]).abs().sum() / actual_kwh_sum
        if actual_kwh_sum > 0
        else np.nan
    )
    logging.info(
        "Trained residual calibration on %s daylight rows; aggregate factor %.3f, "
        "in-sample daylight WMAPE %.2f%% -> %.2f%%",
        len(training_data),
        aggregate_factor,
        base_wmape * 100 if pd.notna(base_wmape) else float("nan"),
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
    )


def predict_residual_calibration_factor(
    model: ResidualCalibrationModel,
    feature_df: pd.DataFrame,
    total_capacity_kw: float,
    latitude: float,
    longitude: float,
    timezone_name: str,
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
    )
    factors = pd.Series(
        model.estimator.predict(calibration_features[model.feature_columns].fillna(0.0)),
        index=feature_df.index,
    )
    return factors.clip(lower=model.lower_bound, upper=model.upper_bound).fillna(model.fallback_factor)


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


def apply_residual_calibration(
    interval_forecast: pd.DataFrame,
    calibration_model: ResidualCalibrationModel,
    total_capacity_kw: float,
    latitude: float,
    longitude: float,
    timezone_name: str,
) -> pd.DataFrame:
    """
    Apply learned residual calibration while preserving the base forecast columns.
    """
    out = interval_forecast.copy()
    if out.empty:
        return out

    out["BaseForecast_kWh"] = pd.to_numeric(out["Forecast_kWh"], errors="coerce").fillna(0.0)
    out["BaseForecast_kW"] = pd.to_numeric(out["Forecast_kW"], errors="coerce").fillna(0.0)
    factors = predict_residual_calibration_factor(
        calibration_model,
        out,
        total_capacity_kw=total_capacity_kw,
        latitude=latitude,
        longitude=longitude,
        timezone_name=timezone_name,
    )
    out["ResidualCalibrationFactor"] = factors.reindex(out.index).fillna(calibration_model.fallback_factor)
    out["SeasonalCalibrationFactor"] = lookup_seasonal_calibration_factor(
        calibration_model,
        out["IntervalStartDT"],
    )
    out["TotalCalibrationFactor"] = out["ResidualCalibrationFactor"] * out["SeasonalCalibrationFactor"]
    active_mask = out["BaseForecast_kWh"] > 0
    out.loc[active_mask, "Forecast_kWh"] = (
        out.loc[active_mask, "BaseForecast_kWh"] * out.loc[active_mask, "TotalCalibrationFactor"]
    )
    out.loc[active_mask, "Forecast_kW"] = (
        out.loc[active_mask, "BaseForecast_kW"] * out.loc[active_mask, "TotalCalibrationFactor"]
    )
    out.loc[~active_mask, "ResidualCalibrationFactor"] = 1.0
    out.loc[~active_mask, "SeasonalCalibrationFactor"] = 1.0
    out.loc[~active_mask, "TotalCalibrationFactor"] = 1.0
    out.loc[~active_mask, "Forecast_kWh"] = 0.0
    out.loc[~active_mask, "Forecast_kW"] = 0.0
    return out


def build_interval_forecast(
    weather_df: pd.DataFrame,
    intrahour_shape: pd.DataFrame,
    capacity_kw: float,
    model: PerformanceModel,
    latitude: float,
    longitude: float,
    timezone_name: str,
    min_solar_elevation: float,
    forecast_source: str = "forecast",
    daily_active_capacity: Optional[pd.DataFrame] = None,
    peak_hourly_kwh_quantile: float = DEFAULT_PEAK_HOURLY_KWH_QUANTILE,
) -> pd.DataFrame:
    """
    Build 15-minute kW forecast from hourly GHI and intra-hour interval shape.
    """
    forecast_df = add_performance_features(
        aggregate_weather_to_hourly(weather_df),
        latitude=latitude,
        longitude=longitude,
        timezone_name=timezone_name,
    )
    forecast_df = forecast_df.dropna(subset=["GHI_kWh_per_m2"])
    forecast_df["PerformanceRatio"] = predict_performance_ratio(model, forecast_df)
    forecast_df["CapacityForDay_kW"] = _resolve_row_capacity(
        forecast_df["IntervalStartDT"],
        daily_active_capacity,
        capacity_kw,
    )
    forecast_df["Hourly_kWh"] = (
        forecast_df["GHI_kWh_per_m2"] * forecast_df["CapacityForDay_kW"] * forecast_df["PerformanceRatio"]
    )

    intrahour_shape = intrahour_shape.copy()
    interval_forecast = forecast_df.merge(intrahour_shape, on="hour", how="left")

    peak_threshold = float(interval_forecast["Hourly_kWh"].quantile(peak_hourly_kwh_quantile))
    if pd.notna(peak_threshold) and peak_threshold > 0:
        peak_mask = (
            interval_forecast["Hourly_kWh"] >= peak_threshold
        ) & (interval_forecast["SolarElevationDeg"] >= 25.0)
        if bool(peak_mask.any()):
            peak_shape = {0: 0.20, 15: 0.24, 30: 0.27, 45: 0.29}
            base = interval_forecast.loc[peak_mask, "IntraHourCoefficient"].fillna(0.25)
            boosted = 0.6 * base + 0.4 * interval_forecast.loc[peak_mask, "minute"].map(peak_shape).fillna(0.25)
            interval_forecast.loc[peak_mask, "IntraHourCoefficient"] = boosted

    interval_forecast["IntervalStartDT"] = (
        interval_forecast["IntervalStartDT"]
        + pd.to_timedelta(interval_forecast["minute"], unit="m")
    )
    interval_forecast["Forecast_kWh"] = (
        interval_forecast["Hourly_kWh"] * interval_forecast["IntraHourCoefficient"]
    )
    interval_forecast["Forecast_kW"] = interval_forecast["Forecast_kWh"] / INTERVAL_HOURS
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
    interval_forecast["TotalCalibrationFactor"] = 1.0
    interval_forecast["SameDayCorrectionFactor"] = 1.0
    return interval_forecast[
        [
            "IntervalStartDT",
            "Forecast_kWh",
            "Forecast_kW",
            "BaseForecast_kWh",
            "BaseForecast_kW",
            "ResidualCalibrationFactor",
            "SeasonalCalibrationFactor",
            "TotalCalibrationFactor",
            "SolarElevationDeg",
            "GHI_kWh_per_m2",
            "WeatherGHI_Wm2",
            "DirectRadiation_Wm2",
            "DiffuseRadiation_Wm2",
            "Temperature_C",
            "WindSpeed_ms",
            "ClearSkyGHI_Wm2",
            "ClearSkyIndex",
            "CloudCoverPct",
            "CloudCoverLowPct",
            "CloudCoverMidPct",
            "CloudCoverHighPct",
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
        logging.info("Same-day correction skipped; %s is outside the forecast window", local_today)
        return pd.DataFrame()

    complete_interval_cutoff = local_now.floor("15min") - pd.Timedelta(minutes=15)
    if complete_interval_cutoff.date() < local_today:
        logging.info("Same-day correction skipped; no completed intervals are available yet for %s", local_today)
        return pd.DataFrame()

    if preloaded_intervals is not None and not preloaded_intervals.empty:
        preloaded = preloaded_intervals.copy()
        preloaded["IntervalStartDT"] = pd.to_datetime(preloaded["IntervalStartDT"])
        preloaded_today = preloaded[preloaded["IntervalStartDT"].dt.date == local_today].copy()
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
            logging.info("Same-day correction skipped; no same-day export actuals found: %s", exc)
            return pd.DataFrame()

    actuals = actuals.copy()
    actuals["IntervalStartDT"] = pd.to_datetime(actuals["IntervalStartDT"])
    actuals = actuals[
        (actuals["IntervalStartDT"].dt.date == local_today)
        & (actuals["IntervalStartDT"] <= complete_interval_cutoff)
    ].copy()
    if actuals.empty:
        logging.info("Same-day correction skipped; no completed same-day export intervals found")
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
) -> pd.DataFrame:
    """
    Scale remaining same-day forecast intervals using observed actual-vs-forecast performance.
    """
    out = interval_forecast.copy()
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

    comparison = actuals[["IntervalStartDT", "Export_kWh"]].merge(
        out[["IntervalStartDT", "Forecast_kWh"]],
        on="IntervalStartDT",
        how="inner",
    )
    comparison = comparison[
        comparison["Forecast_kWh"].notna()
        & (comparison["Forecast_kWh"] > 0)
    ].copy()

    observed_intervals = len(comparison)
    observed_forecast_kwh = comparison["Forecast_kWh"].sum()
    observed_actual_kwh = comparison["Export_kWh"].sum()
    if observed_intervals < min_observed_intervals or observed_forecast_kwh < min_observed_forecast_kwh:
        logging.info(
            "Same-day correction skipped; observed %s daylight intervals and %.2f forecast kWh",
            observed_intervals,
            observed_forecast_kwh,
        )
        return out

    raw_factor = observed_actual_kwh / observed_forecast_kwh if observed_forecast_kwh > 0 else 1.0
    correction_factor = float(np.clip(raw_factor, lower_bound, upper_bound))
    last_observed_interval = actuals["IntervalStartDT"].max()
    future_same_day_mask = (
        (out["IntervalStartDT"].dt.date == local_today)
        & (out["IntervalStartDT"] > last_observed_interval)
    )
    intervals_to_correct = int(future_same_day_mask.sum())
    if intervals_to_correct == 0:
        logging.info("Same-day correction calculated %.3f but no remaining same-day intervals need correction", correction_factor)
        return out

    out.loc[future_same_day_mask, "Forecast_kWh"] *= correction_factor
    out.loc[future_same_day_mask, "Forecast_kW"] *= correction_factor
    out.loc[future_same_day_mask, "SameDayCorrectionFactor"] = correction_factor
    logging.info(
        "Applied same-day correction factor %.3f to %s remaining intervals "
        "(actual %.2f kWh / forecast %.2f kWh over %s observed daylight intervals; raw %.3f)",
        correction_factor,
        intervals_to_correct,
        observed_actual_kwh,
        observed_forecast_kwh,
        observed_intervals,
        raw_factor,
    )
    return out


def resample_interval_forecast_to_hourly(interval_forecast: pd.DataFrame, total_capacity_kw: float) -> pd.DataFrame:
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
    if "TotalCalibrationFactor" not in interval_forecast.columns:
        interval_forecast["TotalCalibrationFactor"] = (
            interval_forecast["ResidualCalibrationFactor"] * interval_forecast["SeasonalCalibrationFactor"]
        )
    if "ForecastSource" not in interval_forecast.columns:
        interval_forecast["ForecastSource"] = "forecast"
    if "PerformanceRatio" not in interval_forecast.columns:
        interval_forecast["PerformanceRatio"] = np.nan

    hourly_forecast = interval_forecast.set_index("IntervalStartDT").resample("h").agg(
        Forecast_kWh=("Forecast_kWh", "sum"),
        Forecast_kW=("Forecast_kW", "mean"),
        BaseForecast_kWh=("BaseForecast_kWh", "sum"),
        BaseForecast_kW=("BaseForecast_kW", "mean"),
        WeatherGHI_Wm2=("WeatherGHI_Wm2", "mean"),
        GHI_kWh_per_m2=("GHI_kWh_per_m2", "mean"),
        DirectRadiation_Wm2=("DirectRadiation_Wm2", "mean"),
        DiffuseRadiation_Wm2=("DiffuseRadiation_Wm2", "mean"),
        Temperature_C=("Temperature_C", "mean"),
        WindSpeed_ms=("WindSpeed_ms", "mean"),
        ClearSkyGHI_Wm2=("ClearSkyGHI_Wm2", "mean"),
        ClearSkyIndex=("ClearSkyIndex", "mean"),
        CloudCoverPct=("CloudCoverPct", "mean"),
        CloudCoverLowPct=("CloudCoverLowPct", "mean"),
        CloudCoverMidPct=("CloudCoverMidPct", "mean"),
        CloudCoverHighPct=("CloudCoverHighPct", "mean"),
        PerformanceRatio=("PerformanceRatio", "mean"),
        ResidualCalibrationFactor=("ResidualCalibrationFactor", "mean"),
        SeasonalCalibrationFactor=("SeasonalCalibrationFactor", "mean"),
        TotalCalibrationFactor=("TotalCalibrationFactor", "mean"),
        SameDayCorrectionFactor=("SameDayCorrectionFactor", "max"),
        ForecastSource=("ForecastSource", "last"),
    )
    hourly_forecast["Forecast_MW"] = hourly_forecast["Forecast_kW"] / 1000.0
    hourly_forecast["BaseForecast_MW"] = hourly_forecast["BaseForecast_kW"] / 1000.0
    if total_capacity_kw > 0:
        hourly_forecast["CapacityFactor"] = hourly_forecast["Forecast_kW"] / total_capacity_kw
        hourly_forecast["BaseCapacityFactor"] = hourly_forecast["BaseForecast_kW"] / total_capacity_kw
    else:
        hourly_forecast["CapacityFactor"] = 0.0
        hourly_forecast["BaseCapacityFactor"] = 0.0
    hourly_forecast.reset_index(inplace=True)
    return add_hour_ending_column(hourly_forecast)


def resample_actual_export_to_hourly(rec_interval_df: pd.DataFrame) -> pd.DataFrame:
    """
    Resample 15-minute actual export rows to hourly actual output.
    """
    rec_hourly = rec_interval_df.set_index("IntervalStartDT").resample("h").agg(
        Export_kWh=("Export_kWh", "sum"),
        Export_kW=("Export_kW", "mean"),
    )
    rec_hourly["Export_kW"] = rec_hourly["Export_kW"].fillna(rec_hourly["Export_kWh"])
    rec_hourly["Export_MW"] = rec_hourly["Export_kW"] / 1000.0
    rec_hourly.reset_index(inplace=True)
    return add_hour_ending_column(rec_hourly)


def build_hourly_backtest(
    rec_interval_df: pd.DataFrame,
    interval_backtest_forecast: pd.DataFrame,
    total_capacity_kw: float,
    apply_actual_quality_filter: bool = True,
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
    backtest = actual_hourly[["IntervalStartDT", "HE", "Actual_MW", "Actual_kWh", "Actual_kW"]].merge(
        forecast_hourly[
            [
                "IntervalStartDT",
                "Forecast_MW",
                "Forecast_kWh",
                "BaseForecast_MW",
                "BaseForecast_kWh",
                "CapacityFactor",
                "BaseCapacityFactor",
                "WeatherGHI_Wm2",
                "GHI_kWh_per_m2",
                "DirectRadiation_Wm2",
                "DiffuseRadiation_Wm2",
                "Temperature_C",
                "WindSpeed_ms",
                "ClearSkyGHI_Wm2",
                "ClearSkyIndex",
                "CloudCoverPct",
                "CloudCoverLowPct",
                "CloudCoverMidPct",
                "CloudCoverHighPct",
                "PerformanceRatio",
                "ResidualCalibrationFactor",
                "SeasonalCalibrationFactor",
                "TotalCalibrationFactor",
                "SameDayCorrectionFactor",
                "ForecastSource",
            ]
        ],
        on="IntervalStartDT",
        how="inner",
    )
    backtest["Error_MW"] = backtest["Forecast_MW"] - backtest["Actual_MW"]
    backtest["Error_kWh"] = backtest["Forecast_kWh"] - backtest["Actual_kWh"]
    backtest["AbsError_MW"] = backtest["Error_MW"].abs()
    backtest["AbsError_kWh"] = backtest["Error_kWh"].abs()
    backtest["BaseError_MW"] = backtest["BaseForecast_MW"] - backtest["Actual_MW"]
    backtest["BaseError_kWh"] = backtest["BaseForecast_kWh"] - backtest["Actual_kWh"]
    backtest["BaseAbsError_MW"] = backtest["BaseError_MW"].abs()
    backtest["BaseAbsError_kWh"] = backtest["BaseError_kWh"].abs()
    backtest["ActualQualityExpected_kWh"] = backtest[["Forecast_kWh", "BaseForecast_kWh"]].max(axis=1)
    if apply_actual_quality_filter:
        backtest = add_solar_actual_quality_flags(
            backtest,
            actual_kwh_col="Actual_kWh",
            expected_kwh_col="ActualQualityExpected_kWh",
            actual_to_expected_ratio_threshold=DEFAULT_ACTUAL_QUALITY_FORECAST_RATIO_THRESHOLD,
        )
    else:
        backtest["ActualQualityFlag"] = ACTUAL_QUALITY_OK
        backtest["SolarBacktestExcluded"] = False
        backtest["ActualToExpectedRatio"] = (
            backtest["Actual_kWh"] / backtest["ActualQualityExpected_kWh"].where(
                backtest["ActualQualityExpected_kWh"] > 0
            )
        ).replace([np.inf, -np.inf], np.nan)
        backtest["ActualQualitySuspiciousHour"] = False
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
            [{
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
                "RawWMAPE": pd.NA,
                "RawBaseWMAPE": pd.NA,
            }]
        )

    raw_backtest_df = backtest_df.copy()
    excluded_mask = _actual_quality_exclusion_mask(raw_backtest_df)
    excluded_df = raw_backtest_df[excluded_mask].copy()
    backtest_df = raw_backtest_df[~excluded_mask].copy()

    def wmape_for(data: pd.DataFrame, forecast_col: str, actual_col: str = "Actual_kWh") -> object:
        actual_sum = data[actual_col].sum()
        if actual_sum <= 0:
            return pd.NA
        return (data[forecast_col] - data[actual_col]).abs().sum() / actual_sum

    raw_actual_mwh = raw_backtest_df["Actual_kWh"].sum() / 1000.0
    raw_forecast_mwh = raw_backtest_df["Forecast_kWh"].sum() / 1000.0
    raw_base_forecast_mwh = raw_backtest_df["BaseForecast_kWh"].sum() / 1000.0
    excluded_actual_mwh = excluded_df["Actual_kWh"].sum() / 1000.0 if not excluded_df.empty else 0.0
    excluded_forecast_mwh = excluded_df["Forecast_kWh"].sum() / 1000.0 if not excluded_df.empty else 0.0
    raw_wmape = wmape_for(raw_backtest_df, "Forecast_kWh")
    raw_base_wmape = wmape_for(raw_backtest_df, "BaseForecast_kWh")

    if backtest_df.empty:
        return pd.DataFrame(
            [{
                "Start": raw_backtest_df["IntervalStartDT"].min(),
                "End": raw_backtest_df["IntervalStartDT"].max(),
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
                "RawIntervals": len(raw_backtest_df),
                "ExcludedIntervals": int(excluded_mask.sum()),
                "ExcludedActual_MWh": excluded_actual_mwh,
                "ExcludedForecast_MWh": excluded_forecast_mwh,
                "RawActual_MWh": raw_actual_mwh,
                "RawForecast_MWh": raw_forecast_mwh,
                "RawBaseForecast_MWh": raw_base_forecast_mwh,
                "RawWMAPE": raw_wmape,
                "RawBaseWMAPE": raw_base_wmape,
            }]
        )

    actual_mwh = backtest_df["Actual_kWh"].sum() / 1000.0
    forecast_mwh = backtest_df["Forecast_kWh"].sum() / 1000.0
    base_forecast_mwh = backtest_df["BaseForecast_kWh"].sum() / 1000.0
    bias_mwh = forecast_mwh - actual_mwh
    base_bias_mwh = base_forecast_mwh - actual_mwh
    rmse_mw = math.sqrt(float((backtest_df["Error_MW"] ** 2).mean()))
    base_rmse_mw = math.sqrt(float((backtest_df["BaseError_MW"] ** 2).mean()))
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
    return pd.DataFrame(
        [{
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
            "MAPE": backtest_df["APE"].dropna().mean(),
            "BaseMAPE": backtest_df["BaseAPE"].dropna().mean(),
            "ActualPeak_MW": backtest_df["Actual_MW"].max(),
            "ForecastPeak_MW": backtest_df["Forecast_MW"].max(),
            "BaseForecastPeak_MW": backtest_df["BaseForecast_MW"].max(),
            "RawIntervals": len(raw_backtest_df),
            "ExcludedIntervals": int(excluded_mask.sum()),
            "ExcludedActual_MWh": excluded_actual_mwh,
            "ExcludedForecast_MWh": excluded_forecast_mwh,
            "RawActual_MWh": raw_actual_mwh,
            "RawForecast_MWh": raw_forecast_mwh,
            "RawBaseForecast_MWh": raw_base_forecast_mwh,
            "RawWMAPE": raw_wmape,
            "RawBaseWMAPE": raw_base_wmape,
        }]
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
    abs_error_mw = error_mw.abs()
    base_abs_error_mw = base_error_mw.abs()
    actual_kwh_sum = float(actual_kwh.sum())
    forecast_kwh_sum = float(forecast_kwh.sum())
    base_forecast_kwh_sum = float(base_forecast_kwh.sum())
    bias_kwh = forecast_kwh_sum - actual_kwh_sum
    base_bias_kwh = base_forecast_kwh_sum - actual_kwh_sum
    wmape = abs(error_kwh).sum() / actual_kwh_sum if actual_kwh_sum > 0 else np.nan
    base_wmape = abs(base_error_kwh).sum() / actual_kwh_sum if actual_kwh_sum > 0 else np.nan
    wmape_improvement = (
        (base_wmape - wmape) / base_wmape
        if pd.notna(base_wmape) and base_wmape > 0 and pd.notna(wmape)
        else np.nan
    )
    positive_actual = actual_kwh > 0
    mape = (abs(error_kwh[positive_actual]) / actual_kwh[positive_actual]).mean() if positive_actual.any() else np.nan
    base_mape = (
        (abs(base_error_kwh[positive_actual]) / actual_kwh[positive_actual]).mean()
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
            "Bias_MWh": bias_kwh / 1000.0,
            "BaseBias_MWh": base_bias_kwh / 1000.0,
            "BiasPct": bias_kwh / actual_kwh_sum if actual_kwh_sum > 0 else np.nan,
            "BaseBiasPct": base_bias_kwh / actual_kwh_sum if actual_kwh_sum > 0 else np.nan,
            "MAE_MW": float(abs_error_mw.mean()),
            "BaseMAE_MW": float(base_abs_error_mw.mean()),
            "RMSE_MW": math.sqrt(float((error_mw ** 2).mean())),
            "BaseRMSE_MW": math.sqrt(float((base_error_mw ** 2).mean())),
            "WMAPE_PCT": wmape * 100.0 if pd.notna(wmape) else np.nan,
            "BaseWMAPE_PCT": base_wmape * 100.0 if pd.notna(base_wmape) else np.nan,
            "WMAPEImprovementPct": wmape_improvement * 100.0 if pd.notna(wmape_improvement) else np.nan,
            "MAPE_PCT": mape * 100.0 if pd.notna(mape) else np.nan,
            "BaseMAPE_PCT": base_mape * 100.0 if pd.notna(base_mape) else np.nan,
            "Underforecast_Rate_PCT": float((error_mw < 0).mean() * 100.0),
            "P90_AbsError_MW": float(abs_error_mw.quantile(0.90)),
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
    active_mask = daylight_mask | (work["Forecast_MW"].fillna(0.0) > daylight_threshold_mw)
    peak_hour_mask = daylight_mask & work["Hour"].between(11, 15, inclusive="both")
    actual_peak = work.loc[daylight_mask, "Actual_MW"].max() if daylight_mask.any() else np.nan
    if pd.notna(actual_peak) and actual_peak > 0:
        high_output_mask = daylight_mask & (work["Actual_MW"] >= actual_peak * 0.75)
    else:
        high_output_mask = pd.Series(False, index=work.index)

    rows.append(_solar_metric_row(work, "Overall", "all", "all"))
    rows.append(_solar_metric_row(work[daylight_mask], "DaylightActual", "Actual_MW", f">{daylight_threshold_mw:g}"))
    rows.append(_solar_metric_row(work[active_mask], "ActiveSolar", "ActualOrForecast_MW", f">{daylight_threshold_mw:g}"))
    rows.append(_solar_metric_row(work[peak_hour_mask], "PeakSolarHours11to15", "Hour", "11-15"))
    rows.append(_solar_metric_row(work[high_output_mask], "HighOutputActualGE75PctPeak", "ActualPeakShare", ">=0.75"))

    if "CloudCoverPct" in work.columns:
        work["CloudCoverBucket"] = pd.cut(
            work["CloudCoverPct"],
            bins=[-np.inf, 20.0, 50.0, 80.0, np.inf],
            labels=["0-20", "20-50", "50-80", "80-100"],
        )
    if "ClearSkyIndex" in work.columns:
        work["ClearSkyIndexBucket"] = pd.cut(
            work["ClearSkyIndex"],
            bins=[-np.inf, 0.25, 0.50, 0.75, 1.00, np.inf],
            labels=["0-0.25", "0.25-0.50", "0.50-0.75", "0.75-1.00", ">1.00"],
        )
    if "GHI_kWh_per_m2" in work.columns:
        work["GHIBucket"] = pd.cut(
            work["GHI_kWh_per_m2"],
            bins=[-np.inf, 0.10, 0.30, 0.50, 0.70, np.inf],
            labels=["0-0.10", "0.10-0.30", "0.30-0.50", "0.50-0.70", ">0.70"],
        )

    _append_grouped_solar_metrics(rows, work, "Month", "Month")
    _append_grouped_solar_metrics(rows, work, "Hour", "Hour", daylight_mask)
    _append_grouped_solar_metrics(rows, work, "CloudCover", "CloudCoverBucket", active_mask)
    _append_grouped_solar_metrics(rows, work, "ClearSkyIndex", "ClearSkyIndexBucket", active_mask)
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
    for column in ["Actual_MW", "Forecast_MW", "BaseForecast_MW", "Error_MW", "AbsError_MW"]:
        if column not in work.columns:
            work[column] = np.nan
        work[column] = pd.to_numeric(work[column], errors="coerce")
    work["Error_MW"] = work["Error_MW"].fillna(work["Forecast_MW"] - work["Actual_MW"])
    work["AbsError_MW"] = work["AbsError_MW"].fillna(work["Error_MW"].abs())
    work["Underforecast_MW"] = work["Actual_MW"] - work["Forecast_MW"]
    work["Overforecast_MW"] = work["Forecast_MW"] - work["Actual_MW"]
    active_mask = (
        (work["Actual_MW"].fillna(0.0) > daylight_threshold_mw)
        | (work["Forecast_MW"].fillna(0.0) > daylight_threshold_mw)
    )
    work = work[active_mask].copy()
    if work.empty:
        return pd.DataFrame()

    review_columns = [
        "IntervalStartDT",
        "HE",
        "Actual_MW",
        "Forecast_MW",
        "BaseForecast_MW",
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
        "TotalCalibrationFactor",
        "ActualQualityFlag",
        "SolarBacktestExcluded",
        "ActualToExpectedRatio",
        "ActualQualityExpected_kWh",
        "ActualQualitySuspiciousHour",
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
    return pd.concat([under, over], ignore_index=True)[["ErrorType", "Rank", *review_columns]]


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
    status_columns = ["Evaluation", "Status", "Reason", "HoldoutDays", "HoldoutStart", "TrainRows", "HoldoutRows"]

    def status_frame(status: str, reason: str, holdout_start: object = pd.NaT) -> pd.DataFrame:
        return pd.DataFrame(
            [{
                "Evaluation": "temporal_holdout",
                "Status": status,
                "Reason": reason,
                "HoldoutDays": holdout_days,
                "HoldoutStart": holdout_start,
                "TrainRows": 0,
                "HoldoutRows": 0,
            }],
            columns=status_columns,
        )

    if holdout_days <= 0:
        return status_frame("skipped", "holdout_days <= 0"), pd.DataFrame()
    if rec_interval_df.empty or weather_df.empty:
        return status_frame("skipped", "missing REC intervals or weather rows"), pd.DataFrame()

    rec = rec_interval_df.copy()
    weather = weather_df.copy()
    rec["IntervalStartDT"] = pd.to_datetime(rec["IntervalStartDT"])
    weather["IntervalStartDT"] = pd.to_datetime(weather["IntervalStartDT"])
    max_timestamp = rec["IntervalStartDT"].max()
    if pd.isna(max_timestamp):
        return status_frame("skipped", "REC intervals have no valid timestamps"), pd.DataFrame()

    holdout_start = max_timestamp.normalize() - pd.Timedelta(days=holdout_days - 1)
    train_rec = rec[rec["IntervalStartDT"] < holdout_start].copy()
    holdout_rec = rec[rec["IntervalStartDT"] >= holdout_start].copy()
    train_weather = weather[weather["IntervalStartDT"] < holdout_start].copy()
    holdout_weather = weather[weather["IntervalStartDT"] >= holdout_start].copy()
    if train_rec.empty or holdout_rec.empty or train_weather.empty or holdout_weather.empty:
        return status_frame("skipped", "insufficient train or holdout rows", holdout_start), pd.DataFrame()

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
        latitude=latitude,
        longitude=longitude,
        timezone_name=timezone_name,
        daily_active_capacity=daily_active_capacity,
        max_performance_ratio=max_performance_ratio,
        use_energy_weighting=use_performance_model_energy_weighting,
        exclude_suppressed_actuals=actual_quality_filter_enabled,
    )
    train_interval_forecast = build_interval_forecast(
        weather_df=train_weather,
        intrahour_shape=holdout_intrahour_shape,
        capacity_kw=capacity_kw,
        model=holdout_model,
        latitude=latitude,
        longitude=longitude,
        timezone_name=timezone_name,
        min_solar_elevation=min_solar_elevation,
        forecast_source="holdout_train",
        daily_active_capacity=daily_active_capacity,
        peak_hourly_kwh_quantile=peak_hourly_kwh_quantile,
    )
    train_backtest = build_hourly_backtest(
        rec_interval_df=train_rec,
        interval_backtest_forecast=train_interval_forecast,
        total_capacity_kw=capacity_kw,
        apply_actual_quality_filter=actual_quality_filter_enabled,
    )
    residual_model = identity_residual_calibration_model(
        lower_bound=residual_lower_bound,
        upper_bound=residual_upper_bound,
    )
    if residual_calibration_enabled:
        residual_model = train_residual_calibration_model(
            backtest_df=train_backtest,
            total_capacity_kw=capacity_kw,
            latitude=latitude,
            longitude=longitude,
            timezone_name=timezone_name,
            lower_bound=residual_lower_bound,
            upper_bound=residual_upper_bound,
            min_training_forecast_kwh=residual_min_forecast_kwh,
            min_training_rows=residual_min_training_rows,
            use_energy_weighting=residual_energy_weighting,
            use_seasonal_calibration=seasonal_calibration_enabled,
            seasonal_prior_mwh=seasonal_prior_mwh,
            seasonal_lower_bound=seasonal_lower_bound,
            seasonal_upper_bound=seasonal_upper_bound,
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
    if residual_calibration_enabled:
        holdout_interval_forecast = apply_residual_calibration(
            interval_forecast=holdout_interval_forecast,
            calibration_model=residual_model,
            total_capacity_kw=capacity_kw,
            latitude=latitude,
            longitude=longitude,
            timezone_name=timezone_name,
        )
        holdout_interval_forecast["ForecastSource"] = "holdout_calibrated"

    holdout_hourly = build_hourly_backtest(
        rec_interval_df=holdout_rec,
        interval_backtest_forecast=holdout_interval_forecast,
        total_capacity_kw=capacity_kw,
        apply_actual_quality_filter=actual_quality_filter_enabled,
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
            "--performance-ratio must be greater than 0 and less than or equal to "
            "--performance-ratio-upper-bound."
        )
    if not (1.0 <= args.performance_ratio_upper_bound <= 1.50):
        raise ValueError("--performance-ratio-upper-bound must be between 1.0 and 1.5.")
    if not (0.0 < args.peak_hourly_kwh_quantile < 1.0):
        raise ValueError("--peak-hourly-kwh-quantile must be greater than 0 and less than 1.")
    if args.rec_history_months <= 0:
        raise ValueError("--rec-history-months must be greater than 0.")
    if not (0 < args.forecast_days <= 16):
        raise ValueError("--forecast-days must be between 1 and 16.")
    if not (0 <= args.historical_days <= 365):
        raise ValueError("--historical-days must be between 0 and 365.")
    if bool(args.rec_history_start) != bool(args.rec_history_end):
        raise ValueError("--rec-history-start and --rec-history-end must be provided together.")
    if args.rec_history_start and args.rec_history_start > args.rec_history_end:
        raise ValueError("--rec-history-start must be earlier than or equal to --rec-history-end.")
    if not (-10 <= args.min_solar_elevation <= 20):
        raise ValueError("--min-solar-elevation must be between -10 and 20 degrees.")
    if args.weather_clusters < 0:
        raise ValueError("--weather-clusters must be greater than or equal to 0.")
    if args.same_day_correction_min_intervals <= 0:
        raise ValueError("--same-day-correction-min-intervals must be greater than 0.")
    if args.same_day_correction_min_forecast_kwh < 0:
        raise ValueError("--same-day-correction-min-forecast-kwh must be greater than or equal to 0.")
    if args.forecast_weather_cache_max_age_hours < 0:
        raise ValueError("--forecast-weather-cache-max-age-hours must be greater than or equal to 0.")
    if not (0 < args.same_day_correction_lower_bound <= args.same_day_correction_upper_bound):
        raise ValueError(
            "--same-day-correction-lower-bound must be greater than 0 and less than or equal to "
            "--same-day-correction-upper-bound."
        )
    if args.daily_shape_method not in {"mean", "median", "upper-quantile"}:
        raise ValueError("--daily-shape-method must be one of: mean, median, upper-quantile.")
    if args.intrahour_shape_method not in {"mean", "median", "upper-quantile"}:
        raise ValueError("--intrahour-shape-method must be one of: mean, median, upper-quantile.")
    if not (0 < args.shape_quantile < 1):
        raise ValueError("--shape-quantile must be greater than 0 and less than 1.")
    if args.residual_calibration_min_rows <= 0:
        raise ValueError("--residual-calibration-min-rows must be greater than 0.")
    if args.residual_calibration_min_forecast_kwh < 0:
        raise ValueError("--residual-calibration-min-forecast-kwh must be greater than or equal to 0.")
    if args.solar_backtest_daylight_threshold_mw < 0:
        raise ValueError("--solar-backtest-daylight-threshold-mw must be greater than or equal to 0.")
    if args.solar_backtest_top_error_count < 0:
        raise ValueError("--solar-backtest-top-error-count must be greater than or equal to 0.")
    if args.solar_backtest_holdout_days < 0:
        raise ValueError("--solar-backtest-holdout-days must be greater than or equal to 0.")
    if not (0 < args.residual_calibration_lower_bound <= args.residual_calibration_upper_bound):
        raise ValueError(
            "--residual-calibration-lower-bound must be greater than 0 and less than or equal to "
            "--residual-calibration-upper-bound."
        )
    if args.seasonal_calibration_prior_mwh < 0:
        raise ValueError("--seasonal-calibration-prior-mwh must be greater than or equal to 0.")
    if not (0 < args.seasonal_calibration_lower_bound <= args.seasonal_calibration_upper_bound):
        raise ValueError(
            "--seasonal-calibration-lower-bound must be greater than 0 and less than or equal to "
            "--seasonal-calibration-upper-bound."
        )
    ZoneInfo(args.timezone)
    weather_cache_dir = None if args.no_weather_cache else Path(args.weather_cache_dir)

    engine: Optional[Engine] = None
    try:
        parquet_root = Path(args.parquet_root)
        residual_calibration_model = identity_residual_calibration_model(
            lower_bound=args.residual_calibration_lower_bound,
            upper_bound=args.residual_calibration_upper_bound,
        )
        daily_active_capacity = pd.DataFrame(columns=["Date", "ActiveCapacity_kW"])
        engine = connect(
            driver=args.driver,
            server=args.dest_server,
            database=args.dest_db,
            username=args.dest_user,
            password=args.dest_pass,
        )

        sites: Optional[pd.DataFrame] = None
        preloaded_export_intervals: Optional[pd.DataFrame] = None
        weather_sites_df = build_system_weather_site(args.latitude, args.longitude)

        if args.production_source == "rec-parquet":
            sites = load_active_solar_sites(engine)
            total_capacity_kw = float(sites["SolarCECkW"].sum())

            if args.rec_history_start and args.rec_history_end:
                rec_start_date = args.rec_history_start
                rec_end_date = args.rec_history_end
            else:
                rec_start_date, rec_end_date = get_default_rec_history_window(
                    parquet_root,
                    args.rec_history_months,
                )

            if args.capacity_normalized_training:
                daily_active_capacity = build_daily_active_capacity(
                    sites,
                    rec_start_date,
                    rec_end_date,
                )
                if daily_active_capacity.empty:
                    logging.info(
                        "Capacity-normalized training requested but no usable interconnection "
                        "dates were found; falling back to flat current capacity."
                    )

            rec_interval_df = load_rec_interval_data(
                parquet_root=parquet_root,
                sites=sites,
                start_date=rec_start_date,
                end_date=rec_end_date,
                net_meter_export_source=args.net_meter_export_source,
                latitude=args.latitude,
                longitude=args.longitude,
                timezone_name=args.timezone,
                min_solar_elevation=args.min_solar_elevation,
            )
            rec_interval_df = add_hour_ending_column(rec_interval_df)
            preloaded_export_intervals = rec_interval_df
            rec_interval_df[
                ["IntervalStartDT", "HE", "Export_kWh", "Export_kW", "ExportSource", "SolarElevationDeg"]
            ].to_csv(args.rec_actual_15min_output, index=False)

            rec_hourly = resample_actual_export_to_hourly(rec_interval_df)
            rec_hourly[["IntervalStartDT", "HE", "Export_MW", "Export_kWh", "Export_kW"]].to_csv(
                args.rec_actual_hourly_output,
                index=False,
            )

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
                        "Capacity-weighted solar weather requested, but no active sites have coordinates. "
                        "Falling back to representative Roseville weather."
                    )
                    args.use_capacity_weighted_weather = False
                    weather_sites_df = build_system_weather_site(args.latitude, args.longitude)
                elif args.weather_clusters > 0 and args.weather_clusters < len(sites_with_coords):
                    logging.info(
                        "Clustering %s sites into %s weather forecast zones",
                        len(sites_with_coords),
                        args.weather_clusters,
                    )
                    sites, weather_sites_df = build_weather_clusters(sites, n_clusters=args.weather_clusters)
                else:
                    logging.info(
                        "Using capacity-weighted average weather across %s active sites with coordinates.",
                        len(sites_with_coords),
                    )
                    weather_sites_df = sites_with_coords
            else:
                logging.info("Using single-point weather forecast for representative lat/lon.")
                weather_sites_df = build_system_weather_site(args.latitude, args.longitude)

            calibration_weather = fetch_open_meteo_hourly_weather(
                weather_sites_df,
                rec_start_date,
                rec_end_date,
                use_forecast=False,
                timezone_name=args.timezone,
                cache_dir=weather_cache_dir,
                use_cache=not args.no_weather_cache,
            )
            if args.use_capacity_weighted_weather:
                calibration_weather = aggregate_capacity_weighted_weather(calibration_weather, sites)

            model = train_performance_model(
                rec_intervals=rec_interval_df,
                weather_df=calibration_weather,
                capacity_kw=total_capacity_kw,
                fallback_ratio=args.performance_ratio,
                latitude=args.latitude,
                longitude=args.longitude,
                timezone_name=args.timezone,
                daily_active_capacity=daily_active_capacity,
                max_performance_ratio=args.performance_ratio_upper_bound,
                use_energy_weighting=args.performance_model_energy_weighting,
                exclude_suppressed_actuals=args.actual_quality_filter,
            )
            calibration_interval_forecast = pd.DataFrame()
            calibration_backtest_hourly = pd.DataFrame()
            if args.backtest or args.residual_calibration:
                logging.info("Building historical base forecast for residual calibration/backtest")
                calibration_interval_forecast = build_interval_forecast(
                    weather_df=calibration_weather,
                    intrahour_shape=intrahour_shape,
                    capacity_kw=total_capacity_kw,
                    model=model,
                    latitude=args.latitude,
                    longitude=args.longitude,
                    timezone_name=args.timezone,
                    min_solar_elevation=args.min_solar_elevation,
                    forecast_source="backtest",
                    daily_active_capacity=daily_active_capacity,
                    peak_hourly_kwh_quantile=args.peak_hourly_kwh_quantile,
                )
                calibration_interval_forecast = add_hour_ending_column(calibration_interval_forecast)
                calibration_backtest_hourly = build_hourly_backtest(
                    rec_interval_df=rec_interval_df,
                    interval_backtest_forecast=calibration_interval_forecast,
                    total_capacity_kw=total_capacity_kw,
                    apply_actual_quality_filter=args.actual_quality_filter,
                )

            if args.residual_calibration:
                residual_calibration_model = train_residual_calibration_model(
                    backtest_df=calibration_backtest_hourly,
                    total_capacity_kw=total_capacity_kw,
                    latitude=args.latitude,
                    longitude=args.longitude,
                    timezone_name=args.timezone,
                    lower_bound=args.residual_calibration_lower_bound,
                    upper_bound=args.residual_calibration_upper_bound,
                    min_training_forecast_kwh=args.residual_calibration_min_forecast_kwh,
                    min_training_rows=args.residual_calibration_min_rows,
                    use_energy_weighting=args.residual_calibration_energy_weighting,
                    use_seasonal_calibration=args.seasonal_calibration,
                    seasonal_prior_mwh=args.seasonal_calibration_prior_mwh,
                    seasonal_lower_bound=args.seasonal_calibration_lower_bound,
                    seasonal_upper_bound=args.seasonal_calibration_upper_bound,
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
                    )
                    interval_backtest_forecast["ForecastSource"] = "backtest_calibrated"
                else:
                    interval_backtest_forecast = calibration_interval_forecast

                backtest_hourly = build_hourly_backtest(
                    rec_interval_df=rec_interval_df,
                    interval_backtest_forecast=interval_backtest_forecast,
                    total_capacity_kw=total_capacity_kw,
                    apply_actual_quality_filter=args.actual_quality_filter,
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
                        "WeatherGHI_Wm2",
                        "GHI_kWh_per_m2",
                        "DirectRadiation_Wm2",
                        "DiffuseRadiation_Wm2",
                        "Temperature_C",
                        "WindSpeed_ms",
                        "ClearSkyGHI_Wm2",
                        "ClearSkyIndex",
                        "CloudCoverPct",
                        "CloudCoverLowPct",
                        "CloudCoverMidPct",
                        "CloudCoverHighPct",
                        "PerformanceRatio",
                        "ResidualCalibrationFactor",
                        "SeasonalCalibrationFactor",
                        "TotalCalibrationFactor",
                        "SameDayCorrectionFactor",
                        "ActualQualityFlag",
                        "SolarBacktestExcluded",
                        "ActualToExpectedRatio",
                        "ActualQualityExpected_kWh",
                        "ActualQualitySuspiciousHour",
                        "BacktestForecast",
                        "ForecastSource",
                    ]
                ].to_csv(args.backtest_hourly_output, index=False)
                backtest_summary.to_csv(args.backtest_summary_output, index=False)
                diagnostics = calculate_solar_backtest_diagnostic_metrics(
                    backtest_hourly,
                    daylight_threshold_mw=args.solar_backtest_daylight_threshold_mw,
                )
                Path(args.solar_backtest_diagnostics_output).parent.mkdir(parents=True, exist_ok=True)
                diagnostics.to_csv(args.solar_backtest_diagnostics_output, index=False)
                top_errors = build_solar_backtest_top_errors(
                    backtest_hourly,
                    top_n=args.solar_backtest_top_error_count,
                    daylight_threshold_mw=args.solar_backtest_daylight_threshold_mw,
                )
                Path(args.solar_backtest_top_errors_output).parent.mkdir(parents=True, exist_ok=True)
                top_errors.to_csv(args.solar_backtest_top_errors_output, index=False)
                if args.solar_backtest_holdout_days > 0:
                    logging.info(
                        "Running %s-day temporal solar holdout backtest",
                        args.solar_backtest_holdout_days,
                    )
                    holdout_scorecard, holdout_hourly = build_solar_temporal_holdout_backtest(
                        rec_interval_df=rec_interval_df,
                        weather_df=calibration_weather,
                        capacity_kw=total_capacity_kw,
                        fallback_ratio=args.performance_ratio,
                        latitude=args.latitude,
                        longitude=args.longitude,
                        timezone_name=args.timezone,
                        min_solar_elevation=args.min_solar_elevation,
                        daily_active_capacity=daily_active_capacity,
                        max_performance_ratio=args.performance_ratio_upper_bound,
                        use_performance_model_energy_weighting=args.performance_model_energy_weighting,
                        intrahour_shape_method=args.intrahour_shape_method,
                        shape_quantile=args.shape_quantile,
                        peak_hourly_kwh_quantile=args.peak_hourly_kwh_quantile,
                        holdout_days=args.solar_backtest_holdout_days,
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
                        actual_quality_filter_enabled=args.actual_quality_filter,
                        daylight_threshold_mw=args.solar_backtest_daylight_threshold_mw,
                    )
                    Path(args.solar_backtest_holdout_output).parent.mkdir(parents=True, exist_ok=True)
                    holdout_scorecard.to_csv(args.solar_backtest_holdout_output, index=False)
                    if not holdout_hourly.empty:
                        Path(args.solar_backtest_holdout_hourly_output).parent.mkdir(parents=True, exist_ok=True)
                        holdout_hourly.to_csv(args.solar_backtest_holdout_hourly_output, index=False)
                summary_row = backtest_summary.iloc[0]
                logging.info(
                    "Backtest saved to %s; diagnostics saved to %s; base WMAPE %.2f%% -> "
                    "calibrated WMAPE %.2f%%, bias %.2f MWh, RMSE %.2f MW",
                    args.backtest_hourly_output,
                    args.solar_backtest_diagnostics_output,
                    summary_row["BaseWMAPE"] * 100 if pd.notna(summary_row["BaseWMAPE"]) else float("nan"),
                    summary_row["WMAPE"] * 100 if pd.notna(summary_row["WMAPE"]) else float("nan"),
                    summary_row["Bias_MWh"],
                    summary_row["RMSE_MW"],
                )
            logging.info("Using parquet export production shape from %s to %s", rec_start_date, rec_end_date)

        else:
            total_capacity_kw = get_total_capacity(engine)
            prod_interval_df = load_production_interval_data(engine)
            if prod_interval_df.empty:
                raise ValueError("No historical production data found. Cannot create production shape.")

            prod_interval_df["IntervalStartDT"] = pd.to_datetime(prod_interval_df["IntervalStartDT"])
            prod_interval_df["IntervalEnergy_kWh"] = prod_interval_df["IntervalValue"] * INTERVAL_HOURS
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
        if args.forecast_start:
            forecast_start_date = args.forecast_start
            forecast_end_date = forecast_start_date + timedelta(days=args.forecast_days - 1)
        else:
            forecast_start_date = today - timedelta(days=args.historical_days)
            forecast_end_date = today + timedelta(days=args.forecast_days - 1)

        inferred_interval_forecast = pd.DataFrame()
        if (
            args.infer_missing_history
            and preloaded_export_intervals is not None
            and not preloaded_export_intervals.empty
        ):
            last_actual_interval = pd.to_datetime(preloaded_export_intervals["IntervalStartDT"]).max()
            inferred_start_timestamp = last_actual_interval.floor("h") + pd.Timedelta(hours=1)
            inferred_end_timestamp = pd.Timestamp(forecast_start_date) - pd.Timedelta(minutes=15)
            if inferred_start_timestamp <= inferred_end_timestamp:
                logging.info(
                    "Inferring missing historical solar generation from %s through %s",
                    inferred_start_timestamp,
                    inferred_end_timestamp,
                )
                inferred_weather_df = fetch_hourly_weather_for_date_range(
                    weather_sites_df,
                    inferred_start_timestamp.date(),
                    inferred_end_timestamp.date(),
                    timezone_name=args.timezone,
                    cache_dir=weather_cache_dir,
                    forecast_cache_max_age_hours=args.forecast_weather_cache_max_age_hours,
                    use_cache=not args.no_weather_cache,
                )
                if args.use_capacity_weighted_weather and sites is not None and "SolarSiteKey" in inferred_weather_df.columns:
                    inferred_weather_df = aggregate_capacity_weighted_weather(inferred_weather_df, sites)
                inferred_interval_forecast = build_interval_forecast(
                    weather_df=inferred_weather_df,
                    intrahour_shape=intrahour_shape,
                    capacity_kw=total_capacity_kw,
                    model=model,
                    latitude=args.latitude,
                    longitude=args.longitude,
                    timezone_name=args.timezone,
                    min_solar_elevation=args.min_solar_elevation,
                    forecast_source="inferred_historical",
                    daily_active_capacity=daily_active_capacity,
                    peak_hourly_kwh_quantile=args.peak_hourly_kwh_quantile,
                )
                if args.residual_calibration:
                    inferred_interval_forecast = apply_residual_calibration(
                        interval_forecast=inferred_interval_forecast,
                        calibration_model=residual_calibration_model,
                        total_capacity_kw=total_capacity_kw,
                        latitude=args.latitude,
                        longitude=args.longitude,
                        timezone_name=args.timezone,
                    )
                inferred_interval_forecast = inferred_interval_forecast[
                    (inferred_interval_forecast["IntervalStartDT"] >= inferred_start_timestamp)
                    & (inferred_interval_forecast["IntervalStartDT"] <= inferred_end_timestamp)
                ].copy()
                inferred_interval_forecast = add_hour_ending_column(inferred_interval_forecast)
                logging.info("Built %s inferred historical 15-minute forecast rows", len(inferred_interval_forecast))

        weather_df = fetch_hourly_weather_for_date_range(
            weather_sites_df,
            forecast_start_date,
            forecast_end_date,
            timezone_name=args.timezone,
            cache_dir=weather_cache_dir,
            forecast_cache_max_age_hours=args.forecast_weather_cache_max_age_hours,
            use_cache=not args.no_weather_cache,
        )
        if args.use_capacity_weighted_weather and sites is not None and "SolarSiteKey" in weather_df.columns:
            weather_df = aggregate_capacity_weighted_weather(weather_df, sites)

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
            latitude=args.latitude,
            longitude=args.longitude,
            timezone_name=args.timezone,
            min_solar_elevation=args.min_solar_elevation,
            forecast_source="forecast",
            daily_active_capacity=daily_active_capacity,
            peak_hourly_kwh_quantile=args.peak_hourly_kwh_quantile,
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
                )
            else:
                logging.info("Same-day correction skipped; REC/NET parquet actuals are not the production source")
        interval_forecast = add_hour_ending_column(interval_forecast)
        if not inferred_interval_forecast.empty:
            interval_forecast = (
                pd.concat([inferred_interval_forecast, interval_forecast], ignore_index=True)
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
                "SolarElevationDeg",
                "WeatherGHI_Wm2",
                "GHI_kWh_per_m2",
                "DirectRadiation_Wm2",
                "DiffuseRadiation_Wm2",
                "Temperature_C",
                "WindSpeed_ms",
                "ClearSkyGHI_Wm2",
                "ClearSkyIndex",
                "CloudCoverPct",
                "CloudCoverLowPct",
                "CloudCoverMidPct",
                "CloudCoverHighPct",
                "PerformanceRatio",
                "ResidualCalibrationFactor",
                "SeasonalCalibrationFactor",
                "TotalCalibrationFactor",
                "SameDayCorrectionFactor",
                "ForecastSource",
            ]
        ].to_csv(
            args.output_15min,
            index=False,
        )

        hourly_forecast = resample_interval_forecast_to_hourly(interval_forecast, total_capacity_kw)
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
                "WeatherGHI_Wm2",
                "GHI_kWh_per_m2",
                "DirectRadiation_Wm2",
                "DiffuseRadiation_Wm2",
                "Temperature_C",
                "WindSpeed_ms",
                "ClearSkyGHI_Wm2",
                "ClearSkyIndex",
                "CloudCoverPct",
                "CloudCoverLowPct",
                "CloudCoverMidPct",
                "CloudCoverHighPct",
                "PerformanceRatio",
                "ResidualCalibrationFactor",
                "SeasonalCalibrationFactor",
                "TotalCalibrationFactor",
                "SameDayCorrectionFactor",
                "ForecastSource",
            ]
        ].to_csv(
            args.output_hourly,
            index=False,
        )

        logging.info("15-minute forecast saved to %s", args.output_15min)
        logging.info("Hourly forecast saved to %s", args.output_hourly)

        print("\n\n=== Roseville Hourly Forecast (first 20 rows) ===")
        print(hourly_forecast.head(20).to_string(index=False, float_format='{:.2f}'.format))

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
    parser.add_argument("--driver", default="ODBC Driver 17 for SQL Server", help="ODBC driver to use.")
    parser.add_argument("--timeout", type=int, default=30, help="ODBC connection timeout in seconds.")

    # Destination arguments
    dest_group = parser.add_argument_group("Destination Connection")
    dest_group.add_argument("--dest-server", default="re-lrs-db-1", help="Destination server name.")
    dest_group.add_argument("--dest-db", default="Forecast", help="Destination database name.")
    dest_group.add_argument("--dest-user", default=None, help="Destination username (for SQL authentication).")
    dest_group.add_argument("--dest-pass", default=None, help="Destination password (for SQL authentication).")

    # Forecaster options
    parser.add_argument(
        "--production-source",
        choices=["rec-parquet", "db-representative"],
        default="rec-parquet",
        help="Historical production source used to build the interval shape and calibration.",
    )
    parser.add_argument(
        "--use-capacity-weighted-weather",
        dest="use_capacity_weighted_weather",
        action="store_true",
        default=True,
        help="Use a capacity-weighted average of weather from active sites instead of a single point.",
    )
    parser.add_argument(
        "--no-capacity-weighted-weather",
        dest="use_capacity_weighted_weather",
        action="store_false",
        help="Use only the representative Roseville latitude/longitude weather point.",
    )
    parser.add_argument(
        "--weather-clusters",
        type=int,
        default=DEFAULT_WEATHER_CLUSTERS,
        help="Number of geographic clusters to use for weather forecasting (0 to disable). Reduces API calls.",
    )
    parser.add_argument(
        "--parquet-root",
        default=str(DEFAULT_PARQUET_ROOT),
        help="Root folder containing COM and RES interval parquet files and the parquet index cache.",
    )
    parser.add_argument(
        "--rec-history-months",
        type=int,
        default=18,
        help=(
            "Most recent available export parquet months used when explicit history dates are "
            "not provided. Use >=12 to cover a full annual cycle so the model learns seasonality "
            "and every monthly calibration factor is populated; pair with capacity-normalized "
            "training so fleet growth over the window does not bias the performance ratio."
        ),
    )
    parser.add_argument(
        "--capacity-normalized-training",
        dest="capacity_normalized_training",
        action="store_true",
        default=True,
        help=(
            "Normalize performance-ratio training and the historical backtest by the daily "
            "active capacity implied by site interconnection dates, so longer histories with "
            "fleet growth remain comparable."
        ),
    )
    parser.add_argument(
        "--no-capacity-normalized-training",
        dest="capacity_normalized_training",
        action="store_false",
        help="Train on a single flat current-capacity scalar (legacy behavior).",
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
    parser.add_argument("--latitude", type=float, default=ROSEVILLE_LATITUDE, help="Representative system latitude.")
    parser.add_argument("--longitude", type=float, default=ROSEVILLE_LONGITUDE, help="Representative system longitude.")
    parser.add_argument("--timezone", default="America/Los_Angeles", help="Local timezone for interval timestamps.")
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
        "--forecast-start",
        type=date.fromisoformat,
        default=None,
        help="Forecast start date in YYYY-MM-DD format. Defaults to today.",
    )
    parser.add_argument(
        "--historical-days",
        type=int,
        default=30,
        help="Number of historical days to include in the forecast output when --forecast-start is omitted.",
    )
    parser.add_argument("--forecast-days", type=int, default=16, help="Number of forecast days to produce (max 16).")
    parser.add_argument(
        "--weather-cache-dir",
        default=str(DEFAULT_SOLAR_WEATHER_CACHE_DIR),
        help="Directory for cached Open-Meteo solar weather responses.",
    )
    parser.add_argument(
        "--forecast-weather-cache-max-age-hours",
        type=float,
        default=DEFAULT_FORECAST_WEATHER_CACHE_MAX_AGE_HOURS,
        help="Freshness window for reusing cached forecast weather before trying Open-Meteo.",
    )
    parser.add_argument(
        "--no-weather-cache",
        action="store_true",
        help="Disable Open-Meteo solar weather cache reads and writes.",
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
        default=DEFAULT_MAX_PERFORMANCE_RATIO,
        help=(
            "Upper bound for learned/predicted performance ratio. Values above 1.0 allow "
            "limited cloud-edge enhancement and improve peak capture."
        ),
    )
    parser.add_argument(
        "--performance-model-energy-weighting",
        dest="performance_model_energy_weighting",
        action="store_true",
        default=True,
        help="Weight performance-ratio training toward higher actual-export daylight hours.",
    )
    parser.add_argument(
        "--no-performance-model-energy-weighting",
        dest="performance_model_energy_weighting",
        action="store_false",
        help="Fit the performance-ratio model without actual-export energy weights.",
    )
    parser.add_argument(
        "--actual-quality-filter",
        dest="actual_quality_filter",
        action="store_true",
        default=True,
        help=(
            "Exclude AMI-suppressed clear-sky actual rows from solar model training and "
            "headline backtest scoring."
        ),
    )
    parser.add_argument(
        "--no-actual-quality-filter",
        dest="actual_quality_filter",
        action="store_false",
        help="Keep AMI-suppressed actual rows in model training and headline backtest scoring.",
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
        "--peak-hourly-kwh-quantile",
        type=float,
        default=DEFAULT_PEAK_HOURLY_KWH_QUANTILE,
        help=(
            "Hourly-kWh quantile used to identify peak-production hours where interval splits "
            "are slightly boosted toward later-quarter peak shape."
        ),
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
    parser.add_argument("--output-15min", default="forecast_outputs/roseville_solar_forecast.csv", help="15-minute forecast CSV path.")
    parser.add_argument("--output-hourly", default="forecast_outputs/roseville_solar_forecast_hourly.csv", help="Hourly forecast CSV path.")
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
        "--load-shape-output",
        default="forecast_outputs/roseville_solar_load_shape.csv",
        help="Average daily load shape CSV path.",
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
