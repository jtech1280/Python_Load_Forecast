from __future__ import annotations

import os
from pathlib import Path
import datetime as dt
import time
import warnings
import pandas as pd
import numpy as np
import requests
from zoneinfo import ZoneInfo

from forecasting.utils.output_archive import save_distinct_snapshot

OPENMETEO_RENAME = {
    "temperature_2m": "TempF",
    "relative_humidity_2m": "HumidityPct",
    "cloud_cover": "CloudCoverPct",
    "wind_speed_10m": "WindSpeedMph",
    "precipitation": "PrecipIn",
    "shortwave_radiation": "GHI_Wm2",
    "is_day": "IsDay",
}

def _today_local(tz_name: str) -> dt.date:
    return dt.datetime.now(ZoneInfo(tz_name)).date()

def _historical_end(config: dict) -> dt.date:
    hist_end = config["openmeteo"]["historical_end"]
    if hist_end:
        return dt.date.fromisoformat(str(hist_end))
    return _today_local(config["project"]["timezone"]) - dt.timedelta(days=1)

def _standard_params(config: dict) -> dict:
    return {
        "latitude": config["openmeteo"]["latitude"],
        "longitude": config["openmeteo"]["longitude"],
        "hourly": ",".join(config["openmeteo"]["hourly_vars"]),
        "timezone": config["openmeteo"]["timezone"],
        "temperature_unit": config["openmeteo"]["temperature_unit"],
        "wind_speed_unit": config["openmeteo"]["wind_speed_unit"],
        "precipitation_unit": config["openmeteo"]["precipitation_unit"],
    }

def _normalize_hourly(payload: dict, captured_at_utc: pd.Timestamp) -> pd.DataFrame:
    hourly = payload.get("hourly", {})
    times = hourly.get("time", [])
    if not times:
        raise ValueError(f"Open-Meteo payload missing 'hourly.time'. Keys: {list(payload.keys())}")

    out = pd.DataFrame({"ValidTimeUTC": pd.to_datetime(times)})
    for key, new_name in OPENMETEO_RENAME.items():
        if key in hourly:
            out[new_name] = hourly[key]

    out["CapturedAtUTC"] = pd.to_datetime(captured_at_utc)
    out.sort_values("ValidTimeUTC", inplace=True)
    return out

def _fetch_json(url: str, params: dict, verify: bool, timeout_seconds: int = 30, retries: int = 1, backoff_seconds: float = 2.0) -> dict:
    last_exc: Exception | None = None
    for attempt in range(max(1, int(retries))):
        try:
            r = requests.get(url, params=params, timeout=int(timeout_seconds), verify=verify)
            r.raise_for_status()
            return r.json()
        except Exception as exc:
            last_exc = exc
            if attempt < max(1, int(retries)) - 1:
                time.sleep(float(backoff_seconds) * (attempt + 1))
    if last_exc is not None:
        raise last_exc
    raise RuntimeError("Weather request failed without an exception")

def _weather_cache_dir(config: dict) -> Path:
    cache_dir = str(config.get("openmeteo", {}).get("cache_dir") or "weather_cache")
    path = Path(cache_dir)
    if not path.is_absolute():
        path = Path.cwd() / path
    return path

def _weather_cache_path(config: dict, stem: str) -> Path:
    return _weather_cache_dir(config) / f"{stem}.csv"

def _read_weather_cache(path: Path, config: dict, start: dt.date | None = None, end: dt.date | None = None) -> pd.DataFrame:
    if not path.exists():
        return pd.DataFrame()
    try:
        out = pd.read_csv(path)
    except Exception:
        return pd.DataFrame()
    if "DT" not in out.columns:
        return pd.DataFrame()

    tz_local = ZoneInfo(config["project"]["timezone"])
    out["DT"] = pd.to_datetime(out["DT"], errors="coerce", utc=True).dt.tz_convert(tz_local)
    out = out.dropna(subset=["DT"]).copy()
    if start is not None:
        out = out[out["DT"].dt.date >= start]
    if end is not None:
        out = out[out["DT"].dt.date <= end]

    cols = ["DT", "TempF", "HumidityPct", "CloudCoverPct", "WindSpeedMph", "PrecipIn", "GHI_Wm2", "IsDay"]
    out = out[[col for col in cols if col in out.columns]].sort_values("DT").drop_duplicates(subset=["DT"], keep="last")
    return out.reset_index(drop=True)

def _write_weather_cache(df: pd.DataFrame, path: Path) -> None:
    if df is None or df.empty:
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    cols = ["DT", "TempF", "HumidityPct", "CloudCoverPct", "WindSpeedMph", "PrecipIn", "GHI_Wm2", "IsDay"]
    df[[col for col in cols if col in df.columns]].to_csv(path, index=False)


def _archive_forecast_weather(df: pd.DataFrame, config: dict) -> Path | None:
    archive_dir = _weather_cache_dir(config) / "forecast_weather_runs"
    hash_cols = ["DT", "TempF", "HumidityPct", "CloudCoverPct", "WindSpeedMph", "PrecipIn", "GHI_Wm2", "IsDay"]
    return save_distinct_snapshot(
        df,
        archive_dir=archive_dir,
        stem="forecast_weather",
        hash_columns=hash_cols,
        metadata={"Source": "open_meteo_forecast"},
    )

def _forecast_results_weather_fallback(config: dict, start: dt.date | None = None, end: dt.date | None = None) -> pd.DataFrame:
    """Recover normalized weather from the latest forecast display export.

    This is intentionally a last-resort operational fallback for Open-Meteo outages. It is less
    authoritative than a fresh archive/forecast response, but it prevents a transient 504 from
    blocking model runs when a recent successful export is present.
    """
    output_dir = Path(str(config.get("project", {}).get("output_dir") or "forecast_outputs"))
    if not output_dir.is_absolute():
        output_dir = Path.cwd() / output_dir
    path = output_dir / "forecast_results.csv"
    if not path.exists():
        return pd.DataFrame()

    try:
        header = pd.read_csv(path, nrows=0)
        wanted = [
            "DT",
            "Temperature",
            "Humidity_Norm",
            "CloudCover_Norm",
            "WindSpeed_Mph",
            "PrecipIn",
            "Solar_Irradiance",
        ]
        usecols = [col for col in wanted if col in header.columns]
        if "DT" not in usecols or "Temperature" not in usecols:
            return pd.DataFrame()
        src = pd.read_csv(path, usecols=usecols)
    except Exception:
        return pd.DataFrame()

    tz_local = ZoneInfo(config["project"]["timezone"])
    out = pd.DataFrame()
    out["DT"] = pd.to_datetime(src["DT"], errors="coerce", utc=True).dt.tz_convert(tz_local)
    out["TempF"] = pd.to_numeric(src.get("Temperature"), errors="coerce")
    out["HumidityPct"] = pd.to_numeric(src.get("Humidity_Norm"), errors="coerce") * 100.0
    out["CloudCoverPct"] = pd.to_numeric(src.get("CloudCover_Norm"), errors="coerce") * 100.0
    out["WindSpeedMph"] = pd.to_numeric(src.get("WindSpeed_Mph"), errors="coerce")
    out["PrecipIn"] = pd.to_numeric(src.get("PrecipIn"), errors="coerce")
    out["GHI_Wm2"] = pd.to_numeric(src.get("Solar_Irradiance"), errors="coerce")
    out = out.dropna(subset=["DT", "TempF"]).copy()
    if start is not None:
        out = out[out["DT"].dt.date >= start]
    if end is not None:
        out = out[out["DT"].dt.date <= end]
    return out.sort_values("DT").drop_duplicates(subset=["DT"], keep="last").reset_index(drop=True)

def fetch_historical_weather(config: dict) -> pd.DataFrame:
    params = _standard_params(config)
    start = dt.date.fromisoformat(config["openmeteo"]["historical_start"])
    end = _historical_end(config)
    url = config["openmeteo"]["historical_url"]
    verify = bool(config["openmeteo"]["ssl_verify"])
    cache_stem = f"historical_weather_{start.isoformat()}_{end.isoformat()}"
    cache_path = _weather_cache_path(config, cache_stem)
    latest_cache_path = _weather_cache_path(config, "historical_weather_latest")
    exact_cached = _read_weather_cache(cache_path, config, start=start, end=end)
    if not exact_cached.empty:
        return exact_cached

    try:
        payload = _fetch_json(
            url,
            {**params, "start_date": start.isoformat(), "end_date": end.isoformat()},
            verify,
            timeout_seconds=90,
            retries=4,
            backoff_seconds=6.0,
        )
        df = _finalize_weather_frame(_normalize_hourly(payload, pd.Timestamp.now("UTC")), config)
        _write_weather_cache(df, cache_path)
        _write_weather_cache(df, latest_cache_path)
        return df
    except Exception as exc:
        for path in [cache_path, latest_cache_path]:
            cached = _read_weather_cache(path, config, start=start, end=end)
            if not cached.empty:
                warnings.warn(f"Historical weather API failed ({exc}); using cached weather: {path}", RuntimeWarning)
                return cached

        fallback = _forecast_results_weather_fallback(config, start=start, end=end)
        if not fallback.empty:
            _write_weather_cache(fallback, cache_path)
            _write_weather_cache(fallback, latest_cache_path)
            warnings.warn(
                f"Historical weather API failed ({exc}); rebuilt weather from forecast_outputs/forecast_results.csv",
                RuntimeWarning,
            )
            return fallback
        raise

def fetch_forecast_weather(config: dict) -> pd.DataFrame:
    params = _standard_params(config)
    params.update({"forecast_days": int(config["openmeteo"]["forecast_days"])})
    past_hours = config.get("openmeteo", {}).get("forecast_past_hours", 0)
    try:
        past_hours_i = int(past_hours or 0)
    except Exception:
        past_hours_i = 0
    if past_hours_i > 0:
        params.update({"past_hours": past_hours_i})
    url = config["openmeteo"]["forecast_url"]
    verify = bool(config["openmeteo"]["ssl_verify"])
    latest_cache_path = _weather_cache_path(config, "forecast_weather_latest")

    try:
        payload = _fetch_json(url, params, verify, timeout_seconds=90, retries=4, backoff_seconds=6.0)
        df = _finalize_weather_frame(_normalize_hourly(payload, pd.Timestamp.now("UTC")), config)
        _write_weather_cache(df, latest_cache_path)
        snapshot_path = _archive_forecast_weather(df, config)
        df.attrs["weather_source"] = "open_meteo_forecast"
        df.attrs["weather_snapshot_path"] = str(snapshot_path) if snapshot_path is not None else ""
        df.attrs["weather_cache_path"] = str(latest_cache_path)
        if snapshot_path is not None:
            print(f"Archived Open-Meteo forecast weather snapshot: {snapshot_path}")
        return df
    except Exception as exc:
        cached = _read_weather_cache(latest_cache_path, config)
        if not cached.empty:
            cached.attrs["weather_source"] = "forecast_weather_latest_cache"
            cached.attrs["weather_snapshot_path"] = ""
            cached.attrs["weather_cache_path"] = str(latest_cache_path)
            warnings.warn(f"Forecast weather API failed ({exc}); using cached weather: {latest_cache_path}", RuntimeWarning)
            return cached

        today = _today_local(config["project"]["timezone"])
        fallback = _forecast_results_weather_fallback(
            config,
            start=today - dt.timedelta(days=2),
            end=today + dt.timedelta(days=int(config["openmeteo"]["forecast_days"]) + 1),
        )
        if not fallback.empty:
            _write_weather_cache(fallback, latest_cache_path)
            fallback.attrs["weather_source"] = "forecast_results_fallback"
            fallback.attrs["weather_snapshot_path"] = ""
            fallback.attrs["weather_cache_path"] = str(latest_cache_path)
            warnings.warn(
                f"Forecast weather API failed ({exc}); rebuilt weather from forecast_outputs/forecast_results.csv",
                RuntimeWarning,
            )
            return fallback
        raise


def fetch_previous_run_weather(
    config: dict,
    start_dt: pd.Timestamp,
    end_dt: pd.Timestamp,
    max_previous_days: int = 7,
) -> pd.DataFrame:
    """Load fixed-lead archived forecast weather for replay realism checks."""
    max_days = max(1, min(7, int(max_previous_days or 7)))
    params = _standard_params(config)
    previous_hourly = []
    for lead_day in range(1, max_days + 1):
        for variable in config["openmeteo"]["hourly_vars"]:
            if variable == "is_day":
                continue
            previous_hourly.append(f"{variable}_previous_day{lead_day}")
    params.update({
        "start_date": pd.Timestamp(start_dt).date().isoformat(),
        "end_date": pd.Timestamp(end_dt).date().isoformat(),
        "hourly": ",".join(previous_hourly),
    })
    url = str(config.get("openmeteo", {}).get("previous_runs_url") or "https://previous-runs-api.open-meteo.com/v1/forecast")
    verify = bool(config["openmeteo"]["ssl_verify"])
    payload = _fetch_json(url, params, verify, timeout_seconds=90, retries=3, backoff_seconds=4.0)
    hourly = payload.get("hourly", {}) or {}
    times = hourly.get("time", [])
    if not times:
        return pd.DataFrame()

    frames = []
    for lead_day in range(1, max_days + 1):
        data = {"ValidTimeUTC": pd.to_datetime(times)}
        for key, new_name in OPENMETEO_RENAME.items():
            previous_key = f"{key}_previous_day{lead_day}"
            if previous_key in hourly:
                data[new_name] = hourly[previous_key]
        if "TempF" not in data:
            continue
        frame = _finalize_weather_frame(pd.DataFrame(data), config)
        if frame.empty:
            continue
        frame["Previous_Run_Lead_Days"] = int(lead_day)
        frames.append(frame)
    return pd.concat(frames, ignore_index=True, sort=False) if frames else pd.DataFrame()

# forecasting/data/weather_loader.py

from zoneinfo import ZoneInfo
import numpy as np
import pandas as pd

def _finalize_weather_frame(df: pd.DataFrame, config: dict) -> pd.DataFrame:
    """
    Normalize Open-Meteo hourly payload to a clean, tz-aware local frame.
    - Open-Meteo returns tz-naive 'time' strings, interpreted in the API `timezone` parameter.
    - We localize to that API timezone, then convert to the project's local timezone.
    """
    tz_local = ZoneInfo(config["project"]["timezone"])
    tz_api_name = str((config.get("openmeteo", {}) or {}).get("timezone") or "GMT")
    shift_hours = int((config.get("quality", {}) or {}).get("weather_timestamp_shift_hours", 0) or 0)
    max_gap = int((config.get("quality", {}) or {}).get("max_interpolation_gap_hours", 0) or 0)

    out = df.copy()

    # 1) Ensure datetime and localize to the API timezone (strings are tz-naive)
    out["DT_API"] = pd.to_datetime(out["ValidTimeUTC"], errors="coerce")
    tz_api = "UTC" if tz_api_name.upper() in {"UTC", "GMT"} else ZoneInfo(tz_api_name)
    # Open-Meteo returns wall-clock times without offsets. When requesting a non-UTC timezone,
    # DST transitions can create "nonexistent" (spring-forward) or "ambiguous" (fall-back) hours.
    out["DT_API"] = out["DT_API"].dt.tz_localize(
        tz_api,
        ambiguous="NaT",
        nonexistent="NaT",
    )

    # 2) Convert to local timezone
    out["DT"] = out["DT_API"].dt.tz_convert(tz_local)
    out = out.dropna(subset=["DT"]).copy()
    if shift_hours:
        out["DT"] = out["DT"] + pd.to_timedelta(shift_hours, unit="h")

    # 3) Clean/standardize fields
    min_f = float(config["quality"]["valid_temp_min_f"])
    max_f = float(config["quality"]["valid_temp_max_f"])

    # numeric coercions
    for col in ["TempF", "HumidityPct", "CloudCoverPct", "WindSpeedMph", "PrecipIn", "GHI_Wm2"]:
        if col in out.columns:
            out[col] = pd.to_numeric(out[col], errors="coerce")

    # temperature range filter
    out.loc[(out["TempF"] < min_f) | (out["TempF"] > max_f), "TempF"] = np.nan

    # Keep canonical set of columns for downstream merge
    cols = ["DT", "TempF", "HumidityPct", "CloudCoverPct", "WindSpeedMph", "PrecipIn", "GHI_Wm2", "IsDay"]
    out = out[[col for col in cols if col in out.columns]].sort_values("DT").drop_duplicates(subset=["DT"], keep="last").reset_index(drop=True)

    # Limited interpolation (ported from v11.6) so small, isolated invalid/missing values
    # don't cause hours to be dropped later by model-frame filtering. Keep this conservative.
    if max_gap > 0 and "TempF" in out.columns:
        work = out.set_index("DT")
        for col in ["TempF", "HumidityPct", "CloudCoverPct", "WindSpeedMph", "PrecipIn", "GHI_Wm2"]:
            if col in work.columns:
                work[col] = pd.to_numeric(work[col], errors="coerce").interpolate(
                    method="time",
                    limit=max_gap,
                    limit_direction="both",
                )
        out = work.reset_index()

    return out
