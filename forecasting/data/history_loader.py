from __future__ import annotations

import datetime as dt

import pandas as pd
import numpy as np
from urllib.parse import quote_plus
from zoneinfo import ZoneInfo

def _make_sql_engine(dsn_name: str):
    try:
        from sqlalchemy import create_engine
    except Exception as exc:
        raise RuntimeError("SQLAlchemy is required for SQL Server loading. Install with `pip install sqlalchemy pyodbc`.") from exc
    odbc = f"DSN={dsn_name};Trusted_Connection=yes;"
    return create_engine(f"mssql+pyodbc:///?odbc_connect={quote_plus(odbc)}")


def actuals_import_cutoff_dt(config: dict, now: pd.Timestamp | None = None) -> pd.Timestamp | None:
    """Return the latest hourly actual timestamp allowed by the configured import freeze."""
    cfg = config.get("actuals_import", {}) or {}
    if not bool(cfg.get("limit_to_prior_day_he24", False)):
        return None

    tz = ZoneInfo(config["project"]["timezone"])
    if now is None:
        now_local = pd.Timestamp.now(tz=tz)
    else:
        now_local = pd.Timestamp(now)
        if now_local.tzinfo is None:
            now_local = now_local.tz_localize(tz)
        else:
            now_local = now_local.tz_convert(tz)

    days_back = max(1, int(cfg.get("cutoff_days_back", 1) or 1))
    hour_ending = int(cfg.get("cutoff_hour_ending", 24) or 24)
    hour_ending = min(24, max(1, hour_ending))
    hour_beginning = hour_ending - 1
    cutoff_date = now_local.date() - dt.timedelta(days=days_back)
    cutoff = pd.Timestamp.combine(cutoff_date, dt.time(hour=hour_beginning)).tz_localize(
        tz,
        ambiguous="NaT",
        nonexistent="shift_forward",
    )
    if pd.isna(cutoff):
        return None
    return cutoff

def load_hourly_system_mwh(config: dict) -> pd.DataFrame:
    """
    Load historical system load (MWh) from METRIX_HRLY_SYSTEM_VW and convert to a TZ-aware hourly series.
    Handles DST fall-back (ambiguous) and spring-forward (nonexistent) hours robustly.
    Returns columns: DT (tz-aware), MWH (float)
    """
    tz = ZoneInfo(config["project"]["timezone"])
    engine = _make_sql_engine(config["sql"]["dsn_name"])
    query = config["sql"]["history_query"]

    df = pd.read_sql_query(query, engine)

    # Expected columns: YEAR, MONTH, DAY, Hr1..Hr24
    hour_cols = [c for c in df.columns if str(c).lower().startswith("hr")]
    long = df.melt(
        id_vars=["YEAR", "MONTH", "DAY"],
        value_vars=hour_cols,
        var_name="HourLabel",
        value_name="MWH"
    )

    # Hr1..Hr24 -> 0..23 using the model's official hourly DT label. Empirical
    # alignment checks against the 5-minute feed treat this DT as the completed-hour label.
    long["Hour"] = long["HourLabel"].str.extract(r"(\d+)").astype(int) - 1

    # Build naive local wall-clock timestamps
    long["DT"] = pd.to_datetime(dict(year=long["YEAR"], month=long["MONTH"], day=long["DAY"])) \
                  + pd.to_timedelta(long["Hour"], unit="h")

    # Localize with explicit DST handling:
    # - ambiguous (fall-back repeated hour): mark as NaT and drop it. (Source data typically has 24
    #   columns/day and does not contain both repeated hours, so "infer" is not reliable.)
    # - nonexistent (spring-forward missing hour): mark as NaT and drop it (avoid duplicate timestamps)
    long["DT"] = long["DT"].dt.tz_localize(
        tz,
        ambiguous="NaT",
        nonexistent="NaT",
    )

    # Drop any rows that became NaT due to ambiguity
    long = long.dropna(subset=["DT"]).copy()

    # Clean target
    long["MWH"] = pd.to_numeric(long["MWH"], errors="coerce").astype(float)

    # Optional: Filter out very old rows to match your training horizon
    train_start = pd.Timestamp(config["training"]["train_start_date"]).to_pydatetime().date()
    # Convert to TZ-aware boundary at local midnight
    train_start_dt = pd.Timestamp(train_start).tz_localize(tz)
    long = long[long["DT"] >= train_start_dt].copy()

    actual_cutoff = actuals_import_cutoff_dt(config)
    if actual_cutoff is not None:
        long = long[long["DT"] <= actual_cutoff].copy()
        long.attrs["actuals_import_cutoff_dt"] = str(actual_cutoff)

    long = long.sort_values("DT").reset_index(drop=True)
    return long[["DT", "MWH"]]
