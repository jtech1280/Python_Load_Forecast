from __future__ import annotations

import warnings
from urllib.parse import quote_plus
from zoneinfo import ZoneInfo

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


def load_five_min_system_load(config: dict) -> pd.DataFrame:
    cfg = config.get("five_min_load", {}) or {}
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
            f"5-minute load query failed; continuing without intraday load features. Details: {exc}",
            RuntimeWarning,
        )
        return pd.DataFrame()

    if raw.empty or "StartPST" not in raw.columns or "Quantity" not in raw.columns:
        warnings.warn(
            "5-minute load query returned no usable StartPST/Quantity columns.",
            RuntimeWarning,
        )
        return pd.DataFrame()

    tz = ZoneInfo(config["project"]["timezone"])
    out = pd.DataFrame()
    out["StartPST"] = pd.to_datetime(raw["StartPST"], errors="coerce")
    out["DT"] = out["StartPST"].dt.tz_localize(tz, ambiguous="NaT", nonexistent="NaT")
    out["FiveMin_Load_MW"] = pd.to_numeric(raw["Quantity"], errors="coerce")
    out.loc[out["FiveMin_Load_MW"] <= 0, "FiveMin_Load_MW"] = pd.NA
    for col in ["EndPST", "QualityName", "TimeFrameName", "Interval"]:
        if col in raw.columns:
            out[col] = raw[col]
    out = out.dropna(subset=["DT", "FiveMin_Load_MW"]).copy()
    out = (
        out.sort_values("DT")
        .drop_duplicates(subset=["DT"], keep="last")
        .reset_index(drop=True)
    )
    return out
