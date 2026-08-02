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
    "wind_direction_10m": "WindDirectionDeg",
    "precipitation": "PrecipIn",
    "shortwave_radiation": "GHI_Wm2",
    "is_day": "IsDay",
}
WEATHER_CACHE_COLS = [
    "DT",
    "TempF",
    "HumidityPct",
    "CloudCoverPct",
    "WindSpeedMph",
    "WindDirectionDeg",
    "PrecipIn",
    "GHI_Wm2",
    "IsDay",
]
_TRUSTSTORE_INJECTED = False
_TRUSTSTORE_WARNING_EMITTED = False

def _today_local(tz_name: str) -> dt.date:
    return dt.datetime.now(ZoneInfo(tz_name)).date()

def _now_local(config: dict) -> pd.Timestamp:
    return pd.Timestamp.now(tz=ZoneInfo(config["project"]["timezone"]))

def _parse_local_time(value: object, default: str) -> dt.time:
    text = str(value or default).strip()
    try:
        parts = text.split(":")
        hour = int(parts[0])
        minute = int(parts[1]) if len(parts) > 1 else 0
        second = int(parts[2]) if len(parts) > 2 else 0
        return dt.time(hour=hour, minute=minute, second=second)
    except Exception:
        fallback = str(default).split(":")
        return dt.time(hour=int(fallback[0]), minute=int(fallback[1]) if len(fallback) > 1 else 0)

def _forecast_import_policy(config: dict) -> dict:
    cfg = ((config.get("openmeteo", {}) or {}).get("forecast_import_policy", {}) or {})
    return cfg if bool(cfg.get("enabled", False)) else {}

def _localize_date_time(local_date: dt.date, local_time: dt.time, tz: ZoneInfo) -> pd.Timestamp:
    return pd.Timestamp.combine(local_date, local_time).tz_localize(
        tz,
        ambiguous="NaT",
        nonexistent="shift_forward",
    )

def _forecast_import_window(now_local: pd.Timestamp, policy: dict) -> tuple[pd.Timestamp, pd.Timestamp]:
    tz = now_local.tzinfo
    if tz is None:
        raise ValueError("now_local must be timezone-aware")
    start_time = _parse_local_time(policy.get("import_window_start_local"), "05:30")
    end_time = _parse_local_time(policy.get("import_window_end_local"), "08:00")
    local_date = now_local.date()
    start = _localize_date_time(local_date, start_time, tz)
    end = _localize_date_time(local_date, end_time, tz)
    if pd.isna(start) or pd.isna(end):
        raise ValueError("Forecast weather import window could not be localized")
    if end <= start:
        if now_local.time() <= end_time:
            start = start - pd.Timedelta(days=1)
        else:
            end = end + pd.Timedelta(days=1)
    return start, end

def _inside_forecast_import_window(now_local: pd.Timestamp, policy: dict) -> bool:
    start, end = _forecast_import_window(now_local, policy)
    return bool(start <= now_local <= end)

def _daily_forecast_weather_cache_path(config: dict, local_date: dt.date, policy: dict) -> Path:
    stem = str(policy.get("daily_cache_stem") or "forecast_weather_morning").strip() or "forecast_weather_morning"
    return _weather_cache_path(config, f"{stem}_{local_date.isoformat()}")

def _cache_mtime_local(path: Path, tz: ZoneInfo) -> pd.Timestamp | None:
    if not path.exists():
        return None
    try:
        return pd.Timestamp.fromtimestamp(path.stat().st_mtime, tz=dt.timezone.utc).tz_convert(tz)
    except OSError:
        return None

def _cache_was_written_in_import_window(path: Path, now_local: pd.Timestamp, policy: dict) -> bool:
    mtime = _cache_mtime_local(path, now_local.tzinfo)
    if mtime is None:
        return False
    start, end = _forecast_import_window(now_local, policy)
    return bool(start <= mtime <= end)

def _mark_forecast_weather_source(
    df: pd.DataFrame,
    *,
    source: str,
    snapshot_path: str | Path | None = "",
    cache_path: str | Path | None = "",
    daily_cache_path: str | Path | None = "",
    import_window: tuple[pd.Timestamp, pd.Timestamp] | None = None,
) -> pd.DataFrame:
    df.attrs["weather_source"] = str(source or "")
    df.attrs["weather_snapshot_path"] = str(snapshot_path) if snapshot_path else ""
    df.attrs["weather_cache_path"] = str(cache_path) if cache_path else ""
    df.attrs["weather_daily_cache_path"] = str(daily_cache_path) if daily_cache_path else ""
    if import_window is not None:
        df.attrs["weather_import_window_local"] = f"{import_window[0]} to {import_window[1]}"
    return df

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

def _fetch_json(url: str, params: dict, verify: bool | str, timeout_seconds: int = 30, retries: int = 1, backoff_seconds: float = 2.0) -> dict:
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

def _ssl_verify(config: dict) -> bool | str:
    openmeteo_cfg = config.get("openmeteo", {}) or {}
    override = os.getenv("OPENMETEO_SSL_VERIFY")
    if override is not None:
        return str(override).strip().lower() not in {"0", "false", "no", "off"}
    ca_bundle = os.getenv("OPENMETEO_CA_BUNDLE") or str(openmeteo_cfg.get("ssl_ca_bundle") or "").strip()
    if ca_bundle:
        return ca_bundle
    return bool(openmeteo_cfg.get("ssl_verify", True))

def _weather_request_verify(config: dict) -> bool | str:
    verify = _ssl_verify(config)
    if verify is True:
        _inject_os_truststore(config)
    return verify

def _inject_os_truststore(config: dict) -> None:
    global _TRUSTSTORE_INJECTED, _TRUSTSTORE_WARNING_EMITTED
    if _TRUSTSTORE_INJECTED:
        return
    openmeteo_cfg = config.get("openmeteo", {}) or {}
    if not bool(openmeteo_cfg.get("ssl_use_os_truststore", True)):
        return
    try:
        import truststore

        truststore.inject_into_ssl()
        _TRUSTSTORE_INJECTED = True
    except Exception as exc:
        if not _TRUSTSTORE_WARNING_EMITTED:
            warnings.warn(
                "Open-Meteo SSL OS trust-store injection failed "
                f"({exc}); falling back to Python/Certifi CA validation. "
                "Install truststore or set OPENMETEO_CA_BUNDLE to a PEM CA bundle if weather API TLS fails.",
                RuntimeWarning,
            )
            _TRUSTSTORE_WARNING_EMITTED = True

def _weather_cache_dir(config: dict) -> Path:
    cache_dir = str(config.get("openmeteo", {}).get("cache_dir") or "weather_cache")
    path = Path(cache_dir)
    if not path.is_absolute():
        path = Path.cwd() / path
    return path

def _weather_cache_path(config: dict, stem: str) -> Path:
    return _weather_cache_dir(config) / f"{stem}.csv"

def _requested_normalized_weather_cols(config: dict) -> set[str]:
    hourly_vars = ((config.get("openmeteo", {}) or {}).get("hourly_vars") or [])
    return {OPENMETEO_RENAME[var] for var in hourly_vars if var in OPENMETEO_RENAME}

def _read_weather_cache(
    path: Path,
    config: dict,
    start: dt.date | None = None,
    end: dt.date | None = None,
    *,
    require_requested_cols: bool = False,
) -> pd.DataFrame:
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

    if require_requested_cols:
        missing = _requested_normalized_weather_cols(config).difference(out.columns)
        if missing:
            return pd.DataFrame()

    cols = WEATHER_CACHE_COLS
    out = out[[col for col in cols if col in out.columns]].sort_values("DT").drop_duplicates(subset=["DT"], keep="last")
    return out.reset_index(drop=True)

def _write_weather_cache(df: pd.DataFrame, path: Path) -> None:
    if df is None or df.empty:
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    cols = WEATHER_CACHE_COLS
    df[[col for col in cols if col in df.columns]].to_csv(path, index=False)


def _archive_forecast_weather(df: pd.DataFrame, config: dict) -> str | Path | None:
    try:
        from forecasting.data.output_sql_store import (
            archive_forecast_weather_snapshot,
            output_sql_enabled,
            output_sql_config,
        )

        if output_sql_enabled(config):
            snapshot_id = archive_forecast_weather_snapshot(
                config,
                df,
                source="open_meteo_forecast",
            )
            sql_cfg = output_sql_config(config)
            schema = str(sql_cfg.get("schema") or "Forecasting")
            table = str(sql_cfg.get("forecast_weather_archive_table") or "LoadForecastWeatherArchive")
            return f"sql:{schema}.{table}:{snapshot_id}" if snapshot_id else None
    except Exception as exc:
        warnings.warn(
            f"SQL forecast weather archive failed ({exc}); continuing without CSV archive.",
            RuntimeWarning,
        )
        return None

    archive_dir = _weather_cache_dir(config) / "forecast_weather_runs"
    hash_cols = WEATHER_CACHE_COLS
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
            "WindDirection_Deg",
            "WindDirectionDeg",
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
    wind_direction_src = src.get("WindDirectionDeg")
    if wind_direction_src is None:
        wind_direction_src = src.get("WindDirection_Deg")
    out["WindDirectionDeg"] = pd.to_numeric(wind_direction_src, errors="coerce")
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
    verify = _weather_request_verify(config)
    cache_stem = f"historical_weather_{start.isoformat()}_{end.isoformat()}"
    cache_path = _weather_cache_path(config, cache_stem)
    latest_cache_path = _weather_cache_path(config, "historical_weather_latest")
    exact_cached = _read_weather_cache(cache_path, config, start=start, end=end, require_requested_cols=True)
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
    verify = _weather_request_verify(config)
    latest_cache_path = _weather_cache_path(config, "forecast_weather_latest")
    policy = _forecast_import_policy(config)
    now_local = _now_local(config)
    daily_cache_path: Path | None = None
    import_window: tuple[pd.Timestamp, pd.Timestamp] | None = None

    if policy:
        import_window = _forecast_import_window(now_local, policy)
        daily_cache_path = _daily_forecast_weather_cache_path(config, now_local.date(), policy)
        daily_cached = _read_weather_cache(daily_cache_path, config, require_requested_cols=True)
        if not daily_cached.empty:
            return _mark_forecast_weather_source(
                daily_cached,
                source="forecast_weather_morning_daily_cache",
                cache_path=latest_cache_path,
                daily_cache_path=daily_cache_path,
                import_window=import_window,
            )

        if _cache_was_written_in_import_window(latest_cache_path, now_local, policy):
            latest_cached = _read_weather_cache(latest_cache_path, config, require_requested_cols=True)
            if not latest_cached.empty:
                _write_weather_cache(latest_cached, daily_cache_path)
                return _mark_forecast_weather_source(
                    latest_cached,
                    source="forecast_weather_latest_cache_adopted_as_morning",
                    cache_path=latest_cache_path,
                    daily_cache_path=daily_cache_path,
                    import_window=import_window,
                )

        if not _inside_forecast_import_window(now_local, policy):
            cached = _read_weather_cache(latest_cache_path, config, require_requested_cols=True)
            if not cached.empty:
                warnings.warn(
                    "Skipping Open-Meteo forecast import outside the configured morning window "
                    f"({import_window[0]} to {import_window[1]}); using cached weather: {latest_cache_path}",
                    RuntimeWarning,
                )
                return _mark_forecast_weather_source(
                    cached,
                    source="forecast_weather_latest_cache_outside_morning_window",
                    cache_path=latest_cache_path,
                    daily_cache_path=daily_cache_path,
                    import_window=import_window,
                )

            today = now_local.date()
            fallback = _forecast_results_weather_fallback(
                config,
                start=today - dt.timedelta(days=2),
                end=today + dt.timedelta(days=int(config["openmeteo"]["forecast_days"]) + 1),
            )
            if not fallback.empty:
                warnings.warn(
                    "Skipping Open-Meteo forecast import outside the configured morning window "
                    f"({import_window[0]} to {import_window[1]}); using forecast_outputs/forecast_results.csv fallback.",
                    RuntimeWarning,
                )
                return _mark_forecast_weather_source(
                    fallback,
                    source="forecast_results_fallback_outside_morning_window",
                    cache_path=latest_cache_path,
                    daily_cache_path=daily_cache_path,
                    import_window=import_window,
                )

            raise RuntimeError(
                "No morning forecast weather cache is available and the current local time "
                f"{now_local} is outside the configured Open-Meteo import window "
                f"({import_window[0]} to {import_window[1]}). Run the weather import near "
                f"{policy.get('target_local_time', '06:30')} local or disable openmeteo.forecast_import_policy."
            )

    try:
        payload = _fetch_json(url, params, verify, timeout_seconds=90, retries=4, backoff_seconds=6.0)
        df = _finalize_weather_frame(_normalize_hourly(payload, pd.Timestamp.now("UTC")), config)
        _write_weather_cache(df, latest_cache_path)
        if daily_cache_path is not None:
            _write_weather_cache(df, daily_cache_path)
        snapshot_path = _archive_forecast_weather(df, config)
        _mark_forecast_weather_source(
            df,
            source="open_meteo_forecast",
            snapshot_path=snapshot_path,
            cache_path=latest_cache_path,
            daily_cache_path=daily_cache_path,
            import_window=import_window,
        )
        if snapshot_path is not None:
            print(f"Archived Open-Meteo forecast weather snapshot: {snapshot_path}")
        return df
    except Exception as exc:
        cached = _read_weather_cache(latest_cache_path, config)
        if not cached.empty:
            _mark_forecast_weather_source(
                cached,
                source="forecast_weather_latest_cache",
                cache_path=latest_cache_path,
                daily_cache_path=daily_cache_path,
                import_window=import_window,
            )
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
            if daily_cache_path is not None:
                _write_weather_cache(fallback, daily_cache_path)
            _mark_forecast_weather_source(
                fallback,
                source="forecast_results_fallback",
                cache_path=latest_cache_path,
                daily_cache_path=daily_cache_path,
                import_window=import_window,
            )
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
    try:
        from forecasting.data.output_sql_store import load_archived_forecast_weather, output_sql_enabled

        if output_sql_enabled(config):
            archived = load_archived_forecast_weather(
                config,
                start_dt=start_dt,
                end_dt=end_dt,
                max_previous_days=max_days,
            )
            if not archived.empty:
                archived.attrs["weather_source"] = "forecast_weather_archive_sql"
                return archived
    except Exception as exc:
        warnings.warn(
            f"SQL forecast weather archive lookup failed ({exc}); falling back to Open-Meteo previous-runs API.",
            RuntimeWarning,
        )

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
    verify = _weather_request_verify(config)
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
    for col in ["TempF", "HumidityPct", "CloudCoverPct", "WindSpeedMph", "WindDirectionDeg", "PrecipIn", "GHI_Wm2"]:
        if col in out.columns:
            out[col] = pd.to_numeric(out[col], errors="coerce")
    if "WindDirectionDeg" in out.columns:
        out["WindDirectionDeg"] = out["WindDirectionDeg"] % 360.0

    # temperature range filter
    out.loc[(out["TempF"] < min_f) | (out["TempF"] > max_f), "TempF"] = np.nan

    # Keep canonical set of columns for downstream merge
    cols = WEATHER_CACHE_COLS
    out = out[[col for col in cols if col in out.columns]].sort_values("DT").drop_duplicates(subset=["DT"], keep="last").reset_index(drop=True)

    # Limited interpolation (ported from v11.6) so small, isolated invalid/missing values
    # don't cause hours to be dropped later by model-frame filtering. Keep this conservative.
    if max_gap > 0 and "TempF" in out.columns:
        work = out.set_index("DT")
        for col in ["TempF", "HumidityPct", "CloudCoverPct", "WindSpeedMph", "WindDirectionDeg", "PrecipIn", "GHI_Wm2"]:
            if col in work.columns:
                work[col] = pd.to_numeric(work[col], errors="coerce").interpolate(
                    method="time",
                    limit=max_gap,
                    limit_direction="both",
                )
        if "WindDirectionDeg" in work.columns:
            work["WindDirectionDeg"] = work["WindDirectionDeg"] % 360.0
        out = work.reset_index()

    return out
