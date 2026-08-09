from __future__ import annotations

import warnings
from urllib.parse import quote_plus
from zoneinfo import ZoneInfo

import numpy as np
import pandas as pd


def _make_sql_engine(dsn_name: str):
    try:
        from sqlalchemy import create_engine
    except Exception as exc:
        raise RuntimeError(
            "SQLAlchemy is required for SQL Server loading. Install with `pip install sqlalchemy pyodbc`."
        ) from exc
    odbc = f"DSN={dsn_name};Trusted_Connection=yes;"
    return create_engine(f"mssql+pyodbc:///?odbc_connect={quote_plus(odbc)}")


def load_local_station_weather(config: dict) -> pd.DataFrame:
    cfg = config.get("local_weather", {}) or {}
    if not bool(cfg.get("enabled", False)):
        return pd.DataFrame()

    dsn = str(cfg.get("dsn_name") or "PowerSupply")
    query = str(cfg.get("query") or "").strip()
    if not query:
        return pd.DataFrame()

    try:
        engine = _make_sql_engine(dsn)
        raw = pd.read_sql_query(query, engine)
    except Exception as exc:
        warnings.warn(
            f"Local weather query failed; continuing with Open-Meteo only. Details: {exc}",
            RuntimeWarning,
        )
        return pd.DataFrame()

    required = {"DT", "READING"}
    if raw.empty or not required.issubset(raw.columns):
        warnings.warn(
            "Local weather query returned no usable DT/READING columns.", RuntimeWarning
        )
        return pd.DataFrame()

    tz = ZoneInfo(config["project"]["timezone"])
    out = pd.DataFrame()
    out["DT"] = pd.to_datetime(raw["DT"], errors="coerce")
    out["DT"] = out["DT"].dt.tz_localize(tz, ambiguous="NaT", nonexistent="NaT")
    out["LocalStation_TempF"] = pd.to_numeric(raw["READING"], errors="coerce")
    if "STATIONID" in raw.columns:
        out["LocalWeather_StationID"] = raw["STATIONID"].astype(str)
    if "CONCEPTID" in raw.columns:
        out["LocalWeather_ConceptID"] = raw["CONCEPTID"].astype(str)

    quality = config.get("quality", {}) or {}
    min_f = float(quality.get("valid_temp_min_f", -30.0))
    max_f = float(quality.get("valid_temp_max_f", 130.0))
    out.loc[
        (out["LocalStation_TempF"] < min_f) | (out["LocalStation_TempF"] > max_f),
        "LocalStation_TempF",
    ] = np.nan

    out = out.dropna(subset=["DT", "LocalStation_TempF"]).copy()
    out["DT"] = out["DT"].dt.floor("h")
    return (
        out.sort_values("DT")
        .drop_duplicates(subset=["DT"], keep="last")
        .reset_index(drop=True)
    )


def build_temperature_bias_lookup(
    openmeteo_hist: pd.DataFrame, local_weather: pd.DataFrame, config: dict
) -> tuple[pd.DataFrame, pd.DataFrame]:
    cfg = (config.get("local_weather", {}) or {}).get(
        "temperature_calibration", {}
    ) or {}
    max_abs_delta = float(cfg.get("max_abs_delta_f", 25.0))
    min_count = int(cfg.get("min_count", 24))

    if (
        openmeteo_hist is None
        or openmeteo_hist.empty
        or local_weather is None
        or local_weather.empty
    ):
        return pd.DataFrame(), pd.DataFrame()

    left = openmeteo_hist[["DT", "TempF"]].copy()
    left["DT"] = pd.to_datetime(left["DT"], errors="coerce", utc=True)
    right = local_weather[["DT", "LocalStation_TempF"]].copy()
    right["DT"] = pd.to_datetime(right["DT"], errors="coerce", utc=True)
    merged = left.merge(right, on="DT", how="inner")
    merged["OpenMeteo_TempF"] = pd.to_numeric(merged["TempF"], errors="coerce")
    merged["LocalStation_TempF"] = pd.to_numeric(
        merged["LocalStation_TempF"], errors="coerce"
    )
    merged["LocalStation_TempDelta_F"] = (
        merged["LocalStation_TempF"] - merged["OpenMeteo_TempF"]
    )
    merged = merged.dropna(
        subset=[
            "DT",
            "OpenMeteo_TempF",
            "LocalStation_TempF",
            "LocalStation_TempDelta_F",
        ]
    ).copy()
    merged = merged[merged["LocalStation_TempDelta_F"].abs().le(max_abs_delta)].copy()
    if merged.empty:
        return pd.DataFrame(), pd.DataFrame()

    local_dt = pd.to_datetime(merged["DT"], utc=True).dt.tz_convert(
        config["project"]["timezone"]
    )
    merged["Month"] = local_dt.dt.month
    merged["Hour"] = local_dt.dt.hour

    global_delta = float(merged["LocalStation_TempDelta_F"].mean())
    by_month = (
        merged.groupby("Month")["LocalStation_TempDelta_F"]
        .agg(["count", "mean"])
        .reset_index()
    )
    by_month = by_month.rename(
        columns={"count": "Month_Count", "mean": "Month_Delta_F"}
    )
    by_month_hour = (
        merged.groupby(["Month", "Hour"])["LocalStation_TempDelta_F"]
        .agg(["count", "mean", "std"])
        .reset_index()
    )
    by_month_hour = by_month_hour.rename(
        columns={"count": "N", "mean": "Bias_Delta_F", "std": "Bias_Std_F"}
    )
    lookup = by_month_hour.merge(by_month, on="Month", how="left")
    lookup["Global_Delta_F"] = global_delta
    lookup["Bias_Source"] = np.where(lookup["N"].ge(min_count), "month_hour", "month")
    lookup["Applied_Delta_F"] = np.where(
        lookup["N"].ge(min_count),
        lookup["Bias_Delta_F"],
        lookup["Month_Delta_F"].fillna(global_delta),
    )
    lookup["Applied_Delta_F"] = pd.to_numeric(
        lookup["Applied_Delta_F"], errors="coerce"
    ).fillna(global_delta)
    return lookup, merged


def apply_temperature_bias_calibration(
    weather_df: pd.DataFrame, lookup: pd.DataFrame, config: dict
) -> pd.DataFrame:
    cfg = (config.get("local_weather", {}) or {}).get(
        "temperature_calibration", {}
    ) or {}
    if (
        weather_df is None
        or weather_df.empty
        or lookup is None
        or lookup.empty
        or "TempF" not in weather_df.columns
    ):
        return weather_df

    blend = float(cfg.get("blend", 1.0))
    cap_f = float(cfg.get("cap_f", 8.0))
    out = weather_df.copy()
    local_dt = pd.to_datetime(out["DT"], errors="coerce", utc=True).dt.tz_convert(
        config["project"]["timezone"]
    )
    out["Month"] = local_dt.dt.month
    out["Hour"] = local_dt.dt.hour
    lk = lookup[["Month", "Hour", "Applied_Delta_F", "Bias_Source"]].copy()
    out = out.merge(lk, on=["Month", "Hour"], how="left")
    global_delta = (
        float(pd.to_numeric(lookup["Global_Delta_F"], errors="coerce").dropna().iloc[0])
        if "Global_Delta_F" in lookup.columns
        and pd.to_numeric(lookup["Global_Delta_F"], errors="coerce").notna().any()
        else 0.0
    )
    out["TempF_OpenMeteo"] = pd.to_numeric(out["TempF"], errors="coerce")
    out["LocalStation_TempBias_F"] = (
        pd.to_numeric(out["Applied_Delta_F"], errors="coerce")
        .fillna(global_delta)
        .clip(-cap_f, cap_f)
        * blend
    )
    out["TempF"] = out["TempF_OpenMeteo"] + out["LocalStation_TempBias_F"]
    out["LocalWeather_TempCal_Source"] = out["Bias_Source"].fillna("global")
    return out.drop(columns=["Applied_Delta_F", "Bias_Source"], errors="ignore")


def merge_local_station_temperature(
    weather_df: pd.DataFrame, local_weather: pd.DataFrame
) -> pd.DataFrame:
    if (
        weather_df is None
        or weather_df.empty
        or local_weather is None
        or local_weather.empty
    ):
        return weather_df
    out = weather_df.copy()
    left_dt = pd.to_datetime(out["DT"], errors="coerce", utc=True)
    local = local_weather[["DT", "LocalStation_TempF"]].copy()
    local["DT"] = pd.to_datetime(local["DT"], errors="coerce", utc=True)
    out["__DT_KEY"] = left_dt
    local["__DT_KEY"] = local["DT"]
    out = out.merge(
        local[["__DT_KEY", "LocalStation_TempF"]], on="__DT_KEY", how="left"
    )
    out["LocalStation_TempDelta_F"] = pd.to_numeric(
        out["LocalStation_TempF"], errors="coerce"
    ) - pd.to_numeric(out.get("TempF_OpenMeteo", out.get("TempF")), errors="coerce")
    return out.drop(columns=["__DT_KEY"], errors="ignore")


def local_temperature_bias_summary(
    matched: pd.DataFrame, lookup: pd.DataFrame
) -> pd.DataFrame:
    if matched is None or matched.empty:
        return pd.DataFrame()
    rows = []
    delta = pd.to_numeric(matched["LocalStation_TempDelta_F"], errors="coerce").dropna()
    rows.append(
        {
            "Segment": "Overall",
            "N": int(delta.count()),
            "Mean_Delta_F": float(delta.mean()),
            "Median_Delta_F": float(delta.median()),
            "MAE_Delta_F": float(delta.abs().mean()),
            "P90_Abs_Delta_F": float(delta.abs().quantile(0.90)),
        }
    )
    by_month = (
        matched.groupby("Month")["LocalStation_TempDelta_F"]
        .agg(
            N="count",
            Mean_Delta_F="mean",
            Median_Delta_F="median",
        )
        .reset_index()
    )
    by_month["Segment"] = "Month " + by_month["Month"].astype(str)
    by_month["MAE_Delta_F"] = (
        matched.groupby("Month")["LocalStation_TempDelta_F"]
        .apply(lambda s: s.abs().mean())
        .to_numpy()
    )
    by_month["P90_Abs_Delta_F"] = (
        matched.groupby("Month")["LocalStation_TempDelta_F"]
        .apply(lambda s: s.abs().quantile(0.90))
        .to_numpy()
    )
    summary = pd.concat(
        [
            pd.DataFrame(rows),
            by_month[
                [
                    "Segment",
                    "N",
                    "Mean_Delta_F",
                    "Median_Delta_F",
                    "MAE_Delta_F",
                    "P90_Abs_Delta_F",
                ]
            ],
        ],
        ignore_index=True,
    )
    if lookup is not None and not lookup.empty:
        summary["Lookup_Rows"] = len(lookup)
    return summary


def apply_dynamic_temperature_calibration(
    fut_wx: pd.DataFrame, hist_wx: pd.DataFrame, config: dict
) -> pd.DataFrame:
    """
    Dynamically correct the future weather forecast temperature if the actual local station
    temperatures in the last 24 hours have been trending higher or lower than the Open-Meteo actuals.
    This directly models transient cooling/warming events like the Delta Breeze.
    """
    cfg = (config.get("local_weather", {}) or {}).get(
        "temperature_calibration", {}
    ) or {}
    if not bool(cfg.get("dynamic_enabled", True)):
        return fut_wx

    if fut_wx is None or fut_wx.empty or hist_wx is None or hist_wx.empty:
        return fut_wx

    if "LocalStation_TempF" not in hist_wx.columns:
        return fut_wx

    matched = hist_wx.dropna(subset=["LocalStation_TempF"]).copy()
    if matched.empty:
        return fut_wx

    recent_window_hours = int(cfg.get("dynamic_window_hours", 24))
    recent = matched.tail(recent_window_hours)

    temp_om = pd.to_numeric(
        recent.get("TempF_OpenMeteo", recent.get("TempF")), errors="coerce"
    )
    temp_local = pd.to_numeric(recent["LocalStation_TempF"], errors="coerce")
    recent_bias = (temp_local - temp_om).dropna()

    if recent_bias.empty:
        return fut_wx

    mean_bias = float(recent_bias.mean())

    cap_f = float(cfg.get("dynamic_cap_f", 6.0))
    blend = float(cfg.get("dynamic_blend", 0.80))
    applied_bias = np.clip(mean_bias, -cap_f, cap_f) * blend

    out = fut_wx.copy()
    out["DT"] = pd.to_datetime(out["DT"])
    min_dt = out["DT"].min()

    decay_hours = float(cfg.get("dynamic_decay_hours", 48.0))

    hours_ahead = (out["DT"] - min_dt).dt.total_seconds() / 3600.0
    decay_factor = np.exp(-hours_ahead / decay_hours)

    correction = applied_bias * decay_factor
    out["TempF"] = out["TempF"] + correction

    out["Dynamic_Weather_Correction_F"] = correction

    return out
