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
3.  Loads REC channel export and negative NET interval export data from parquet
    files for active solar sites.
4.  Calculates normalized 15-minute weights within each hour from historical
    export intervals.
5.  Fetches hourly GHI and cloud-cover weather data from the Open-Meteo API.
6.  Builds hourly export energy from GHI, system capacity, and calibrated
    performance ratio.
7.  Splits each hourly forecast into 15-minute intervals using the intra-hour
    historical shape.
8.  Optionally corrects remaining same-day intervals using completed actual
    export intervals already observed today.
9.  Saves the final forecast and export actuals to CSV files.

Requirements
------------
pip install pyodbc pandas requests scikit-learn SQLAlchemy pyarrow
"""

from __future__ import annotations

import argparse
import logging
import math
import sys
from datetime import date, datetime, timedelta
from pathlib import Path
from typing import Optional
from zoneinfo import ZoneInfo

import numpy as np
import pandas as pd
import requests
import urllib3
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
ROSEVILLE_LATITUDE = 38.7522
ROSEVILLE_LONGITUDE = -121.2880
NET_METER_TYPES = {"AMI_NET", "AMI_NET_D"}
HOURLY_WEATHER_VARIABLES = [
    "shortwave_radiation",
    "cloud_cover",
    "cloud_cover_low",
    "cloud_cover_mid",
    "cloud_cover_high",
]


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


def build_production_shape(interval_df: pd.DataFrame, energy_col: str) -> pd.DataFrame:
    """
    Build a normalized 15-minute daily energy shape from interval energy data.
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


def build_intrahour_production_shape(interval_df: pd.DataFrame, energy_col: str) -> pd.DataFrame:
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

    observed_shape = (
        shape_df.groupby(["hour", "minute"], as_index=False)["IntraHourCoefficient"]
        .mean()
    )
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

    return pd.concat(normalized_groups, ignore_index=True)


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
    logging.info("Fetching %s weather data from %s to %s", source_name, start_date, end_date)

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

    response = requests.get(url, params=params, verify=False, timeout=60)
    response.raise_for_status()  # Raise an exception for bad status codes
    results = response.json()

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

    logging.info("Fetched %s weather data points", len(weather_df))
    return weather_df


def fetch_open_meteo_hourly_weather(
    sites: pd.DataFrame,
    start_date: date,
    end_date: date,
    use_forecast: bool,
    timezone_name: str,
) -> pd.DataFrame:
    """
    Fetch hourly GHI and cloud cover from Open-Meteo archive or forecast API.
    """
    source_name = "forecast" if use_forecast else "historical"
    logging.info("Fetching hourly %s weather data from %s to %s", source_name, start_date, end_date)

    url = (
        "https://api.open-meteo.com/v1/forecast"
        if use_forecast
        else "https://archive-api.open-meteo.com/v1/archive"
    )
    params = {
        "latitude": ",".join(sites["Latitude"].astype(str).tolist()),
        "longitude": ",".join(sites["Longitude"].astype(str).tolist()),
        "start_date": start_date.isoformat(),
        "end_date": end_date.isoformat(),
        "hourly": ",".join(HOURLY_WEATHER_VARIABLES),
        "timezone": timezone_name,
    }

    response = requests.get(url, params=params, verify=False, timeout=60)
    response.raise_for_status()
    results = response.json()
    if isinstance(results, dict):
        results = [results]

    all_weather_data = []
    for i, site_weather in enumerate(results):
        hourly = site_weather.get("hourly")
        if not hourly:
            continue

        site_id = sites.iloc[i]["SolarSiteKey"]
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

    if not all_weather_data:
        raise ValueError(f"No hourly weather data returned from Open-Meteo for {start_date} to {end_date}.")

    weather_df = pd.concat(all_weather_data, ignore_index=True)
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
) -> pd.DataFrame:
    """
    Fetch hourly archive data for past dates and hourly forecast data for today/future dates.
    """
    today = current_local_timestamp(timezone_name).date()
    frames = []

    archive_end = min(end_date, today - timedelta(days=1))
    if start_date <= archive_end:
        frames.append(fetch_open_meteo_hourly_weather(sites, start_date, archive_end, False, timezone_name))

    forecast_start = max(start_date, today)
    if forecast_start <= end_date:
        frames.append(fetch_open_meteo_hourly_weather(sites, forecast_start, end_date, True, timezone_name))

    if not frames:
        frames.append(fetch_open_meteo_hourly_weather(sites, start_date, end_date, False, timezone_name))

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


def build_interval_forecast(
    weather_df: pd.DataFrame,
    intrahour_shape: pd.DataFrame,
    capacity_kw: float,
    performance_ratio: float,
    latitude: float,
    longitude: float,
    timezone_name: str,
    min_solar_elevation: float,
) -> pd.DataFrame:
    """
    Build 15-minute kW forecast from hourly GHI and intra-hour interval shape.
    """
    forecast_df = weather_df.copy()
    forecast_df["IntervalStartDT"] = pd.to_datetime(forecast_df["IntervalStartDT"])
    forecast_df["GHI_kWh_per_m2"] = pd.to_numeric(
        forecast_df["GHI_kWh_per_m2"],
        errors="coerce",
    ).clip(lower=0)

    forecast_df = (
        forecast_df.groupby("IntervalStartDT", as_index=False)
        .agg(
            GHI_kWh_per_m2=("GHI_kWh_per_m2", "mean"),
            WeatherGHI_Wm2=("WeatherGHI_Wm2", "mean"),
            CloudCoverPct=("CloudCoverPct", "mean"),
            CloudCoverLowPct=("CloudCoverLowPct", "mean"),
            CloudCoverMidPct=("CloudCoverMidPct", "mean"),
            CloudCoverHighPct=("CloudCoverHighPct", "mean"),
        )
        .dropna(subset=["GHI_kWh_per_m2"])
    )
    forecast_df["date"] = forecast_df["IntervalStartDT"].dt.date
    forecast_df["hour"] = forecast_df["IntervalStartDT"].dt.hour
    forecast_df["Hourly_kWh"] = forecast_df["GHI_kWh_per_m2"] * capacity_kw * performance_ratio

    intrahour_shape = intrahour_shape.copy()
    interval_forecast = forecast_df.merge(intrahour_shape, on="hour", how="left")
    interval_forecast["IntervalStartDT"] = (
        interval_forecast["IntervalStartDT"]
        + pd.to_timedelta(interval_forecast["minute"], unit="m")
    )
    interval_forecast["Forecast_kWh"] = (
        interval_forecast["Hourly_kWh"] * interval_forecast["IntraHourCoefficient"]
    )
    interval_forecast["Forecast_kW"] = interval_forecast["Forecast_kWh"] / INTERVAL_HOURS
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
    interval_forecast["SameDayCorrectionFactor"] = 1.0
    return interval_forecast[
        [
            "IntervalStartDT",
            "Forecast_kWh",
            "Forecast_kW",
            "SolarElevationDeg",
            "GHI_kWh_per_m2",
            "WeatherGHI_Wm2",
            "CloudCoverPct",
            "CloudCoverLowPct",
            "CloudCoverMidPct",
            "CloudCoverHighPct",
            "SameDayCorrectionFactor",
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


# =============================================================================
# Main Forecaster
# =============================================================================

def run_forecaster(args: argparse.Namespace) -> None:
    """
    Main function to run the solar forecasting process.
    """
    if not (0 < args.performance_ratio <= 1):
        raise ValueError("--performance-ratio must be greater than 0 and less than or equal to 1.")
    if args.rec_history_months <= 0:
        raise ValueError("--rec-history-months must be greater than 0.")
    if not (0 < args.forecast_days <= 16):
        raise ValueError("--forecast-days must be between 1 and 16.")
    if bool(args.rec_history_start) != bool(args.rec_history_end):
        raise ValueError("--rec-history-start and --rec-history-end must be provided together.")
    if args.rec_history_start and args.rec_history_start > args.rec_history_end:
        raise ValueError("--rec-history-start must be earlier than or equal to --rec-history-end.")
    if not (-10 <= args.min_solar_elevation <= 20):
        raise ValueError("--min-solar-elevation must be between -10 and 20 degrees.")
    if args.same_day_correction_min_intervals <= 0:
        raise ValueError("--same-day-correction-min-intervals must be greater than 0.")
    if args.same_day_correction_min_forecast_kwh < 0:
        raise ValueError("--same-day-correction-min-forecast-kwh must be greater than or equal to 0.")
    if not (0 < args.same_day_correction_lower_bound <= args.same_day_correction_upper_bound):
        raise ValueError(
            "--same-day-correction-lower-bound must be greater than 0 and less than or equal to "
            "--same-day-correction-upper-bound."
        )
    ZoneInfo(args.timezone)

    engine: Optional[Engine] = None
    try:
        parquet_root = Path(args.parquet_root)
        engine = connect(
            driver=args.driver,
            server=args.dest_server,
            database=args.dest_db,
            username=args.dest_user,
            password=args.dest_pass,
        )

        weather_site_df = build_system_weather_site(args.latitude, args.longitude)
        sites: Optional[pd.DataFrame] = None
        preloaded_export_intervals: Optional[pd.DataFrame] = None

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

            rec_hourly = rec_interval_df.set_index("IntervalStartDT").resample("h").agg(
                Export_kWh=("Export_kWh", "sum"),
                Export_kW=("Export_kW", "mean"),
            )
            rec_hourly["Export_MW"] = rec_hourly["Export_kW"] / 1000.0
            rec_hourly.reset_index(inplace=True)
            rec_hourly = add_hour_ending_column(rec_hourly)
            rec_hourly[["IntervalStartDT", "HE", "Export_MW", "Export_kWh", "Export_kW"]].to_csv(
                args.rec_actual_hourly_output,
                index=False,
            )

            intrahour_shape = build_intrahour_production_shape(rec_interval_df, "Export_kWh")
            calibration_weather = fetch_open_meteo_hourly_weather(
                weather_site_df,
                rec_start_date,
                rec_end_date,
                use_forecast=False,
                timezone_name=args.timezone,
            )
            performance_ratio = calibrate_performance_ratio(
                rec_intervals=rec_interval_df,
                weather_df=calibration_weather,
                capacity_kw=total_capacity_kw,
                fallback_ratio=args.performance_ratio,
            )
            logging.info("Using parquet export production shape from %s to %s", rec_start_date, rec_end_date)

        else:
            total_capacity_kw = get_total_capacity(engine)
            prod_interval_df = load_production_interval_data(engine)
            if prod_interval_df.empty:
                raise ValueError("No historical production data found. Cannot create production shape.")

            prod_interval_df["IntervalStartDT"] = pd.to_datetime(prod_interval_df["IntervalStartDT"])
            prod_interval_df["IntervalEnergy_kWh"] = prod_interval_df["IntervalValue"] * INTERVAL_HOURS
            intrahour_shape = build_intrahour_production_shape(prod_interval_df, "IntervalEnergy_kWh")
            performance_ratio = args.performance_ratio
            logging.info("Using representative DB production shape")

        forecast_start_date = args.forecast_start or current_local_timestamp(args.timezone).date()
        forecast_end_date = forecast_start_date + timedelta(days=args.forecast_days - 1)
        weather_df = fetch_hourly_weather_for_date_range(
            weather_site_df,
            forecast_start_date,
            forecast_end_date,
            timezone_name=args.timezone,
        )

        logging.info(
            "Running forecast for %s to %s with capacity %.2f kW and performance ratio %.3f",
            forecast_start_date,
            forecast_end_date,
            total_capacity_kw,
            performance_ratio,
        )
        interval_forecast = build_interval_forecast(
            weather_df=weather_df,
            intrahour_shape=intrahour_shape,
            capacity_kw=total_capacity_kw,
            performance_ratio=performance_ratio,
            latitude=args.latitude,
            longitude=args.longitude,
            timezone_name=args.timezone,
            min_solar_elevation=args.min_solar_elevation,
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

        interval_forecast[
            [
                "IntervalStartDT",
                "HE",
                "Forecast_kW",
                "Forecast_kWh",
                "SolarElevationDeg",
                "WeatherGHI_Wm2",
                "GHI_kWh_per_m2",
                "CloudCoverPct",
                "CloudCoverLowPct",
                "CloudCoverMidPct",
                "CloudCoverHighPct",
                "SameDayCorrectionFactor",
            ]
        ].to_csv(
            args.output_15min,
            index=False,
        )

        logging.info("Resampling forecast to hourly and converting to MW")
        hourly_forecast = interval_forecast.set_index("IntervalStartDT").resample("h").agg(
            Forecast_kWh=("Forecast_kWh", "sum"),
            Forecast_kW=("Forecast_kW", "mean"),
            WeatherGHI_Wm2=("WeatherGHI_Wm2", "mean"),
            GHI_kWh_per_m2=("GHI_kWh_per_m2", "mean"),
            CloudCoverPct=("CloudCoverPct", "mean"),
            CloudCoverLowPct=("CloudCoverLowPct", "mean"),
            CloudCoverMidPct=("CloudCoverMidPct", "mean"),
            CloudCoverHighPct=("CloudCoverHighPct", "mean"),
            SameDayCorrectionFactor=("SameDayCorrectionFactor", "max"),
        )
        hourly_forecast["Forecast_MW"] = hourly_forecast["Forecast_kW"] / 1000.0
        hourly_forecast.reset_index(inplace=True)
        hourly_forecast = add_hour_ending_column(hourly_forecast)
        hourly_forecast[
            [
                "IntervalStartDT",
                "HE",
                "Forecast_MW",
                "Forecast_kWh",
                "WeatherGHI_Wm2",
                "GHI_kWh_per_m2",
                "CloudCoverPct",
                "CloudCoverLowPct",
                "CloudCoverMidPct",
                "CloudCoverHighPct",
                "SameDayCorrectionFactor",
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
        "--parquet-root",
        default=str(DEFAULT_PARQUET_ROOT),
        help="Root folder containing COM and RES interval parquet files and the parquet index cache.",
    )
    parser.add_argument(
        "--rec-history-months",
        type=int,
        default=3,
        help="Most recent available export parquet months used when explicit history dates are not provided.",
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
    parser.add_argument("--forecast-days", type=int, default=16, help="Number of forecast days to produce (max 16).")
    parser.add_argument(
        "--performance-ratio",
        type=float,
        default=DEFAULT_PERFORMANCE_RATIO,
        help="Fallback performance ratio used when REC/weather calibration is unavailable.",
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
    parser.add_argument("--output-15min", default="forecast_outputs/roseville_solar_forecast.csv", help="15-minute forecast CSV path.")
    parser.add_argument("--output-hourly", default="forecast_outputs/roseville_solar_forecast_hourly.csv", help="Hourly forecast CSV path.")
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