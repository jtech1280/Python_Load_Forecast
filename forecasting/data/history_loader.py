from __future__ import annotations

import datetime as dt

import pandas as pd
import numpy as np
from urllib.parse import quote_plus
from zoneinfo import ZoneInfo


def _as_bool(value: object, *, default: bool) -> bool:
    if value is None or value == "":
        return default
    if isinstance(value, bool):
        return value
    if isinstance(value, (int, float)):
        return bool(value)
    if isinstance(value, str):
        normalized = value.strip().lower()
        if normalized in {"1", "true", "yes", "y", "on"}:
            return True
        if normalized in {"0", "false", "no", "n", "off"}:
            return False
    return default


def _coerce_sql_config(sql_config: dict | str) -> dict:
    if isinstance(sql_config, str):
        return {"dsn_name": sql_config}
    return dict(sql_config or {})


def _sql_connection_attempts(sql_config: dict | str) -> list[tuple[str, str]]:
    cfg = _coerce_sql_config(sql_config)
    dsn_name = str(cfg.get("dsn_name") or "").strip()
    if not dsn_name:
        raise ValueError("sql.dsn_name is required for hourly system-load loading.")

    username = str(cfg.get("username") or cfg.get("user") or "").strip()
    password = str(cfg.get("password") or "")
    has_sql_auth = bool(username and password)
    trusted_connection = _as_bool(
        cfg.get("trusted_connection"),
        default=True,
    )
    sql_auth_fallback = _as_bool(cfg.get("sql_auth_fallback"), default=True)

    attempts: list[tuple[str, str]] = []
    if trusted_connection:
        attempts.append(
            (
                "trusted connection",
                f"DSN={dsn_name};Trusted_Connection=yes;",
            )
        )
    if has_sql_auth and (sql_auth_fallback or not trusted_connection):
        attempts.append(("SQL auth", f"DSN={dsn_name};UID={username};PWD={password};"))
    if not attempts:
        attempts.append(
            (
                "trusted connection",
                f"DSN={dsn_name};Trusted_Connection=yes;",
            )
        )
    return attempts


def _scrub_sql_error(exc: BaseException, secrets: list[str]) -> str:
    message = f"{type(exc).__name__}: {exc}"
    for secret in secrets:
        if secret:
            message = message.replace(secret, "***")
    return message


def _make_sql_engine_from_odbc(odbc: str):
    try:
        from sqlalchemy import create_engine
    except Exception as exc:
        raise RuntimeError(
            "SQLAlchemy is required for SQL Server loading. Install with `pip install sqlalchemy pyodbc`."
        ) from exc
    return create_engine(f"mssql+pyodbc:///?odbc_connect={quote_plus(odbc)}")


def _make_sql_engine(dsn_name: str):
    return _make_sql_engine_from_odbc(f"DSN={dsn_name};Trusted_Connection=yes;")


def _read_sql_query_with_auth_fallback(
    query: str, sql_config: dict | str
) -> pd.DataFrame:
    cfg = _coerce_sql_config(sql_config)
    password = str(cfg.get("password") or "")
    secrets = [password, quote_plus(password) if password else ""]
    errors: list[str] = []

    for label, odbc in _sql_connection_attempts(cfg):
        engine = None
        try:
            engine = _make_sql_engine_from_odbc(odbc)
            return pd.read_sql_query(query, engine)
        except Exception as exc:
            errors.append(f"{label}: {_scrub_sql_error(exc, secrets)}")
        finally:
            if engine is not None:
                try:
                    engine.dispose()
                except Exception:
                    pass

    raise RuntimeError(
        "Failed to load hourly system MWh using configured SQL connection attempts:\n"
        + "\n".join(errors)
    )


def actuals_import_cutoff_dt(
    config: dict, now: pd.Timestamp | None = None
) -> pd.Timestamp | None:
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
    cutoff = pd.Timestamp.combine(
        cutoff_date, dt.time(hour=hour_beginning)
    ).tz_localize(
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
    sql_config = config["sql"]
    query = config["sql"]["history_query"]

    df = _read_sql_query_with_auth_fallback(query, sql_config)

    # Expected columns: YEAR, MONTH, DAY, Hr1..Hr24
    hour_cols = [c for c in df.columns if str(c).lower().startswith("hr")]
    long = df.melt(
        id_vars=["YEAR", "MONTH", "DAY"],
        value_vars=hour_cols,
        var_name="HourLabel",
        value_name="MWH",
    )

    # Hr1..Hr24 -> 0..23 using the model's official hourly DT label. Empirical
    # alignment checks against the 5-minute feed treat this DT as the completed-hour label.
    long["Hour"] = long["HourLabel"].str.extract(r"(\d+)").astype(int) - 1

    # Build naive local wall-clock timestamps
    long["DT"] = pd.to_datetime(
        dict(year=long["YEAR"], month=long["MONTH"], day=long["DAY"])
    ) + pd.to_timedelta(long["Hour"], unit="h")

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
    train_start = (
        pd.Timestamp(config["training"]["train_start_date"]).to_pydatetime().date()
    )
    # Convert to TZ-aware boundary at local midnight
    train_start_dt = pd.Timestamp(train_start).tz_localize(tz)
    long = long[long["DT"] >= train_start_dt].copy()

    actual_cutoff = actuals_import_cutoff_dt(config)
    if actual_cutoff is not None:
        long = long[long["DT"] <= actual_cutoff].copy()
        long.attrs["actuals_import_cutoff_dt"] = str(actual_cutoff)

    long = long.sort_values("DT").reset_index(drop=True)
    return long[["DT", "MWH"]]
