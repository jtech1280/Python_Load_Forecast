from __future__ import annotations

import hashlib
import json
import math
import os
import re
import uuid
from urllib.parse import quote_plus

import pandas as pd
from sqlalchemy import text


DEFAULT_REPLAY_TABLES = {
    "rolling_origin_replay_summary": "LoadForecastReplaySummary",
    "rolling_origin_replay_results": "LoadForecastReplayResult",
    "rolling_origin_replay_origin_coverage": "LoadForecastReplayOriginCoverage",
    "rolling_origin_replay_scorecard": "LoadForecastReplayScorecard",
    "rolling_origin_replay_weather_realism_scorecard": "LoadForecastReplayWeatherRealismScorecard",
    "rolling_origin_replay_weather_input_error_by_lead": "LoadForecastReplayWeatherInputErrorByLead",
    "rolling_origin_replay_weather_input_sensitivity_scorecard": "LoadForecastReplayWeatherInputSensitivityScorecard",
    "rolling_origin_replay_weather_input_sensitivity_detail": "LoadForecastReplayWeatherInputSensitivityDetail",
    "rolling_origin_replay_stage_metrics": "LoadForecastReplayStageMetric",
    "rolling_origin_replay_origin_metrics_by_stage": "LoadForecastReplayOriginMetricByStage",
    "rolling_origin_replay_scored_season_metrics_by_stage": "LoadForecastReplayScoredSeasonMetricByStage",
    "rolling_origin_replay_origin_season_metrics_by_stage": "LoadForecastReplayOriginSeasonMetricByStage",
    "rolling_origin_replay_horizon_metrics_by_stage": "LoadForecastReplayHorizonMetricByStage",
    "rolling_origin_replay_peak_window_metrics_by_stage": "LoadForecastReplayPeakWindowMetricByStage",
    "rolling_origin_replay_hot_peak_metrics_by_stage": "LoadForecastReplayHotPeakMetricByStage",
    "rolling_origin_replay_shoulder_heat_metrics_by_stage": "LoadForecastReplayShoulderHeatMetricByStage",
    "rolling_origin_replay_cloud_solar_midday_metrics_by_stage": "LoadForecastReplayCloudSolarMiddayMetricByStage",
    "rolling_origin_replay_weekend_metrics_by_stage": "LoadForecastReplayWeekendMetricByStage",
    "rolling_origin_replay_holiday_metrics_by_stage": "LoadForecastReplayHolidayMetricByStage",
    "rolling_origin_replay_long_horizon_metrics_by_stage": "LoadForecastReplayLongHorizonMetricByStage",
    "rolling_origin_replay_daily_peak_miss_by_stage": "LoadForecastReplayDailyPeakMissByStage",
    "rolling_origin_replay_timing": "LoadForecastReplayTiming",
    "production_readiness_scorecard": "LoadForecastProductionReadinessScorecard",
}

DEFAULT_OUTPUT_SQL_CONFIG = {
    "enabled": True,
    "dashboard_read_enabled": False,
    "dsn_name": "Forecast_DB",
    "database": "",
    "schema": "Forecasting",
    "run_table": "LoadForecastRun",
    "forecast_table": "LoadForecastOutput",
    "backtest_table": "LoadForecastBacktest",
    "weather_table": "LoadForecastWeather",
    "forecast_weather_archive_table": "LoadForecastWeatherArchive",
    "replay_tables": DEFAULT_REPLAY_TABLES,
    "chunksize": 1000,
}

_IDENTIFIER_RE = re.compile(r"^[A-Za-z_][A-Za-z0-9_]{0,127}$")
_DATETIME_COLUMNS = {"DT", "ValidTimeUTC", "CapturedAtUTC", "FirstDT", "LastDT"}


def output_sql_config(config: dict) -> dict:
    cfg = dict(DEFAULT_OUTPUT_SQL_CONFIG)
    cfg["replay_tables"] = dict(DEFAULT_REPLAY_TABLES)
    raw = (config.get("output_sql", {}) or {}) if isinstance(config, dict) else {}
    replay_overrides = raw.get("replay_tables") if isinstance(raw, dict) else None
    cfg.update({k: v for k, v in raw.items() if k != "replay_tables"})
    if isinstance(replay_overrides, dict):
        cfg["replay_tables"].update(replay_overrides)
    return cfg


def output_sql_enabled(config: dict) -> bool:
    return bool(output_sql_config(config).get("enabled", False))


def output_sql_dashboard_read_enabled(config: dict) -> bool:
    override = os.environ.get("DASH_FROM_SQL")
    if override is not None:
        return override.strip().lower() in {"1", "true", "yes", "on"}
    cfg = output_sql_config(config)
    return bool(cfg.get("dashboard_read_enabled", False))


def _make_sql_engine(sql_cfg: dict):
    try:
        from sqlalchemy import create_engine
    except Exception as exc:
        raise RuntimeError("SQLAlchemy is required for SQL Server output persistence.") from exc

    dsn_name = str(sql_cfg.get("dsn_name") or "").strip()
    if not dsn_name:
        raise ValueError("output_sql.dsn_name is required when SQL output persistence is enabled.")

    parts = [f"DSN={dsn_name}", "Trusted_Connection=yes"]
    database = str(sql_cfg.get("database") or "").strip()
    if database:
        parts.append(f"DATABASE={database}")
    odbc = ";".join(parts) + ";"
    return create_engine(
        f"mssql+pyodbc:///?odbc_connect={quote_plus(odbc)}",
        fast_executemany=False,
    )


def _clean_identifier(value: str, label: str) -> str:
    out = str(value or "").strip()
    if not _IDENTIFIER_RE.match(out):
        raise ValueError(f"Invalid SQL {label} identifier: {value!r}")
    return out


def _q(identifier: str) -> str:
    return "[" + str(identifier).replace("]", "]]") + "]"


def _sql_string(value: str) -> str:
    return "N'" + str(value).replace("'", "''") + "'"


def _full_name(schema: str, table: str) -> str:
    return f"{_q(schema)}.{_q(table)}"


def _object_name(schema: str, table: str) -> str:
    return f"{schema}.{table}"


def _is_datetime_column(column: str) -> bool:
    name = str(column)
    return name in _DATETIME_COLUMNS or name.endswith("AtUTC") or name.endswith("_UTC") or name.endswith("_DT")


def _format_datetime_value(value):
    if value is None:
        return None
    try:
        if pd.isna(value):
            return None
    except (TypeError, ValueError):
        return None
    try:
        ts = pd.Timestamp(value)
    except Exception:
        return None
    if pd.isna(ts):
        return None
    return ts.isoformat()


def _normalize_object_value(value):
    if value is None:
        return None
    if isinstance(value, (dict, list, tuple)):
        return json.dumps(value, default=str, sort_keys=True)
    try:
        if pd.isna(value):
            return None
    except (TypeError, ValueError):
        pass
    return value


def _clean_frame_for_sql(df: pd.DataFrame) -> pd.DataFrame:
    if df is None or df.empty:
        return pd.DataFrame()

    out = df.copy()
    for col in list(out.columns):
        if _is_datetime_column(col):
            out[col] = out[col].map(_format_datetime_value)
        elif (
            pd.api.types.is_object_dtype(out[col])
            or pd.api.types.is_string_dtype(out[col])
            or isinstance(out[col].dtype, pd.CategoricalDtype)
        ):
            out[col] = out[col].map(_normalize_object_value)
    return out.where(pd.notna(out), None)


def _prepare_frame_for_sql(
    df: pd.DataFrame,
    run_id: str,
    inserted_at_utc,
) -> pd.DataFrame:
    out = _clean_frame_for_sql(df)
    if out.empty:
        return pd.DataFrame()

    out.insert(0, "InsertedAtUTC", inserted_at_utc)
    out.insert(0, "RunID", str(run_id))
    return out


def _infer_sql_type(column: str, series: pd.Series) -> str:
    if column == "RunID":
        return "UNIQUEIDENTIFIER"
    if column == "InsertedAtUTC":
        return "DATETIME2(7)"
    if _is_datetime_column(column):
        return "DATETIMEOFFSET(7)"
    if pd.api.types.is_bool_dtype(series):
        return "BIT"
    if pd.api.types.is_integer_dtype(series):
        return "BIGINT"
    if pd.api.types.is_float_dtype(series) or pd.api.types.is_numeric_dtype(series):
        return "FLOAT"
    return "NVARCHAR(MAX)"


def _frame_hash(df: pd.DataFrame) -> str:
    if df is None or df.empty:
        return ""
    work = df.copy()
    if "DT" in work.columns:
        work["DT"] = pd.to_datetime(work["DT"], errors="coerce", utc=True).dt.strftime("%Y-%m-%dT%H:%M:%SZ")
        work.sort_values("DT", inplace=True)
    payload = work.reset_index(drop=True).to_csv(index=False, lineterminator="\n").encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


def _dt_bounds(df: pd.DataFrame) -> tuple[str | None, str | None]:
    if df is None or df.empty or "DT" not in df.columns:
        return None, None
    dt = pd.to_datetime(df["DT"], errors="coerce", utc=True)
    valid = dt.dropna()
    if valid.empty:
        return None, None
    return valid.min().isoformat(), valid.max().isoformat()


def _table_exists(conn, schema: str, table: str) -> bool:
    value = conn.execute(
        text("SELECT CASE WHEN OBJECT_ID(:object_name, 'U') IS NULL THEN 0 ELSE 1 END"),
        {"object_name": _object_name(schema, table)},
    ).scalar()
    return bool(value)


def _ensure_schema(conn, schema: str) -> None:
    conn.execute(
        text(
            f"""
            IF SCHEMA_ID({_sql_string(schema)}) IS NULL
                EXEC(N'CREATE SCHEMA {_q(schema)}')
            """
        )
    )


def _ensure_run_table(conn, schema: str, table: str) -> None:
    full = _full_name(schema, table)
    conn.execute(
        text(
            f"""
            IF OBJECT_ID({_sql_string(_object_name(schema, table))}, N'U') IS NULL
            BEGIN
                CREATE TABLE {full} (
                    [RunID] UNIQUEIDENTIFIER NOT NULL CONSTRAINT [PK_{table}_RunID] PRIMARY KEY,
                    [RunStartedAtUTC] DATETIME2(7) NOT NULL,
                    [InsertedAtUTC] DATETIME2(7) NOT NULL CONSTRAINT [DF_{table}_InsertedAtUTC] DEFAULT SYSUTCDATETIME(),
                    [ProjectName] NVARCHAR(256) NULL,
                    [Source] NVARCHAR(128) NULL,
                    [ForecastRows] INT NULL,
                    [BacktestRows] INT NULL,
                    [WeatherRows] INT NULL,
                    [FirstOutputDT] DATETIMEOFFSET(7) NULL,
                    [LastOutputDT] DATETIMEOFFSET(7) NULL,
                    [ContentHash] NVARCHAR(64) NULL,
                    [MetadataJson] NVARCHAR(MAX) NULL
                )
            END
            """
        )
    )


def _existing_columns(conn, schema: str, table: str) -> set[str]:
    rows = conn.execute(
        text(
            """
            SELECT COLUMN_NAME
            FROM INFORMATION_SCHEMA.COLUMNS
            WHERE TABLE_SCHEMA = :schema_name
              AND TABLE_NAME = :table_name
            """
        ),
        {"schema_name": schema, "table_name": table},
    )
    return {str(row[0]) for row in rows}


def _ensure_data_table(conn, schema: str, table: str, df: pd.DataFrame) -> None:
    full = _full_name(schema, table)
    if not _table_exists(conn, schema, table):
        column_defs = [
            "[OutputRowID] BIGINT IDENTITY(1,1) NOT NULL CONSTRAINT "
            f"[PK_{table}_OutputRowID] PRIMARY KEY",
        ]
        for col in df.columns:
            sql_type = _infer_sql_type(col, df[col])
            nullable = "NOT NULL" if col in {"RunID", "InsertedAtUTC"} else "NULL"
            column_defs.append(f"{_q(col)} {sql_type} {nullable}")
        ddl = ",\n                    ".join(column_defs)
        conn.execute(
            text(
                f"""
                CREATE TABLE {full} (
                    {ddl}
                )
                """
            )
        )
    else:
        existing = _existing_columns(conn, schema, table)
        for col in df.columns:
            if col in existing:
                continue
            conn.execute(text(f"ALTER TABLE {full} ADD {_q(col)} {_infer_sql_type(col, df[col])} NULL"))

    _ensure_index(conn, schema, table, ["RunID"])
    if "DT" in df.columns:
        _ensure_index(conn, schema, table, ["RunID", "DT"])
    if "Replay_Origin_DT" in df.columns:
        _ensure_index(conn, schema, table, ["RunID", "Replay_Origin_DT"])


def _ensure_forecast_weather_archive_table(conn, schema: str, table: str) -> None:
    full = _full_name(schema, table)
    conn.execute(
        text(
            f"""
            IF OBJECT_ID({_sql_string(_object_name(schema, table))}, N'U') IS NULL
            BEGIN
                CREATE TABLE {full} (
                    [WeatherArchiveRowID] BIGINT IDENTITY(1,1) NOT NULL CONSTRAINT [PK_{table}_WeatherArchiveRowID] PRIMARY KEY,
                    [SnapshotID] UNIQUEIDENTIFIER NOT NULL,
                    [ArchivedAtUTC] DATETIME2(7) NOT NULL,
                    [Source] NVARCHAR(128) NULL,
                    [ContentHash] NVARCHAR(64) NOT NULL,
                    [FirstDT] DATETIMEOFFSET(7) NULL,
                    [LastDT] DATETIMEOFFSET(7) NULL,
                    [DT] DATETIMEOFFSET(7) NOT NULL,
                    [TempF] FLOAT NULL,
                    [HumidityPct] FLOAT NULL,
                    [CloudCoverPct] FLOAT NULL,
                    [WindSpeedMph] FLOAT NULL,
                    [PrecipIn] FLOAT NULL,
                    [GHI_Wm2] FLOAT NULL,
                    [IsDay] BIGINT NULL
                )
            END
            """
        )
    )
    _ensure_index(conn, schema, table, ["SnapshotID"])
    _ensure_index(conn, schema, table, ["ContentHash"])
    _ensure_index(conn, schema, table, ["ArchivedAtUTC"])
    _ensure_index(conn, schema, table, ["DT"])


def _ensure_index(conn, schema: str, table: str, columns: list[str]) -> None:
    suffix = "_".join(columns)
    raw_name = f"IX_{table}_{suffix}"
    index_name = re.sub(r"[^A-Za-z0-9_]", "_", raw_name)[:120]
    full = _full_name(schema, table)
    column_sql = ", ".join(_q(col) for col in columns)
    conn.execute(
        text(
            f"""
            IF NOT EXISTS (
                SELECT 1
                FROM sys.indexes
                WHERE name = :index_name
                  AND object_id = OBJECT_ID(:object_name)
            )
            BEGIN
                CREATE INDEX {_q(index_name)} ON {full} ({column_sql})
            END
            """
        ),
        {"index_name": index_name, "object_name": _object_name(schema, table)},
    )


def _insert_run_metadata(
    conn,
    schema: str,
    table: str,
    *,
    run_id: str,
    run_started_at_utc,
    inserted_at_utc,
    project_name: str,
    source: str,
    forecast_rows: int,
    backtest_rows: int,
    weather_rows: int,
    first_output_dt: str | None,
    last_output_dt: str | None,
    content_hash: str,
    metadata: dict,
) -> None:
    full = _full_name(schema, table)
    conn.execute(
        text(
            f"""
            INSERT INTO {full} (
                [RunID], [RunStartedAtUTC], [InsertedAtUTC], [ProjectName], [Source],
                [ForecastRows], [BacktestRows], [WeatherRows], [FirstOutputDT], [LastOutputDT],
                [ContentHash], [MetadataJson]
            )
            VALUES (
                :run_id, :run_started_at_utc, :inserted_at_utc, :project_name, :source,
                :forecast_rows, :backtest_rows, :weather_rows, :first_output_dt, :last_output_dt,
                :content_hash, :metadata_json
            )
            """
        ),
        {
            "run_id": run_id,
            "run_started_at_utc": run_started_at_utc,
            "inserted_at_utc": inserted_at_utc,
            "project_name": project_name,
            "source": source,
            "forecast_rows": int(forecast_rows),
            "backtest_rows": int(backtest_rows),
            "weather_rows": int(weather_rows),
            "first_output_dt": first_output_dt,
            "last_output_dt": last_output_dt,
            "content_hash": content_hash,
            "metadata_json": json.dumps(metadata or {}, default=str, sort_keys=True),
        },
    )


def _sql_param_value(value):
    if value is None:
        return None
    try:
        if pd.isna(value):
            return None
    except (TypeError, ValueError):
        pass
    if isinstance(value, pd.Timestamp):
        return _format_datetime_value(value)
    if hasattr(value, "item"):
        try:
            value = value.item()
        except Exception:
            pass
    if isinstance(value, float) and not math.isfinite(value):
        return None
    return value


def _append_frame(conn, schema: str, table: str, df: pd.DataFrame, chunksize: int) -> int:
    if df is None or df.empty:
        return 0

    columns = list(df.columns)
    if not columns:
        return 0
    params = [f"p{i}" for i in range(len(columns))]
    column_sql = ", ".join(_q(col) for col in columns)
    values_sql = ", ".join(f":{name}" for name in params)
    insert_sql = text(f"INSERT INTO {_full_name(schema, table)} ({column_sql}) VALUES ({values_sql})")
    records = df.to_dict(orient="records")
    chunk_size = max(1, int(chunksize or 1000))
    for start in range(0, len(records), chunk_size):
        batch = [
            {
                param_name: _sql_param_value(record.get(column))
                for param_name, column in zip(params, columns)
            }
            for record in records[start:start + chunk_size]
        ]
        if batch:
            conn.execute(insert_sql, batch)
    return int(len(df))


def _forecast_weather_archive_frame(
    df: pd.DataFrame,
    *,
    snapshot_id: str,
    archived_at_utc,
    source: str,
    content_hash: str,
    first_dt: str | None,
    last_dt: str | None,
) -> pd.DataFrame:
    if df is None or df.empty:
        return pd.DataFrame()
    cols = ["DT", "TempF", "HumidityPct", "CloudCoverPct", "WindSpeedMph", "PrecipIn", "GHI_Wm2", "IsDay"]
    out = df[[col for col in cols if col in df.columns]].copy()
    if "DT" not in out.columns or out.empty:
        return pd.DataFrame()
    out.insert(0, "LastDT", last_dt)
    out.insert(0, "FirstDT", first_dt)
    out.insert(0, "ContentHash", content_hash)
    out.insert(0, "Source", str(source or "")[:128])
    out.insert(0, "ArchivedAtUTC", archived_at_utc)
    out.insert(0, "SnapshotID", str(snapshot_id))
    return _clean_frame_for_sql(out)


def archive_forecast_weather_snapshot(
    config: dict,
    weather_df: pd.DataFrame,
    *,
    source: str = "open_meteo_forecast",
    archived_at_utc=None,
) -> str | None:
    sql_cfg = output_sql_config(config)
    if not bool(sql_cfg.get("enabled", False)):
        return None
    if weather_df is None or weather_df.empty:
        return None

    schema = _clean_identifier(sql_cfg.get("schema", "Forecasting"), "schema")
    table = _clean_identifier(sql_cfg.get("forecast_weather_archive_table", "LoadForecastWeatherArchive"), "table")
    chunksize = int(sql_cfg.get("chunksize") or 1000)
    cols = ["DT", "TempF", "HumidityPct", "CloudCoverPct", "WindSpeedMph", "PrecipIn", "GHI_Wm2", "IsDay"]
    canonical = weather_df[[col for col in cols if col in weather_df.columns]].copy()
    if "DT" not in canonical.columns or canonical.empty:
        return None

    content_hash = _frame_hash(canonical)
    if not content_hash:
        return None
    try:
        archived_at_missing = archived_at_utc is None or pd.isna(archived_at_utc)
    except (TypeError, ValueError):
        archived_at_missing = archived_at_utc is None
    if archived_at_missing:
        now = pd.Timestamp.now(tz="UTC").tz_localize(None).to_pydatetime()
    else:
        archived_ts = pd.Timestamp(archived_at_utc)
        if archived_ts.tzinfo is not None:
            archived_ts = archived_ts.tz_convert("UTC").tz_localize(None)
        now = archived_ts.to_pydatetime()
    first_dt, last_dt = _dt_bounds(canonical)

    engine = _make_sql_engine(sql_cfg)
    try:
        with engine.begin() as conn:
            _ensure_schema(conn, schema)
            _ensure_forecast_weather_archive_table(conn, schema, table)
            existing = conn.execute(
                text(
                    f"""
                    SELECT TOP (1) [SnapshotID]
                    FROM {_full_name(schema, table)}
                    WHERE [ContentHash] = :content_hash
                    ORDER BY [ArchivedAtUTC] DESC
                    """
                ),
                {"content_hash": content_hash},
            ).scalar()
            if existing is not None:
                return str(existing)

            snapshot_id = str(uuid.uuid4())
            archive_df = _forecast_weather_archive_frame(
                canonical,
                snapshot_id=snapshot_id,
                archived_at_utc=now,
                source=source,
                content_hash=content_hash,
                first_dt=first_dt,
                last_dt=last_dt,
            )
            if archive_df.empty:
                return None
            _append_frame(conn, schema, table, archive_df, chunksize)
            return snapshot_id
    finally:
        engine.dispose()


def load_archived_forecast_weather(
    config: dict,
    *,
    start_dt,
    end_dt,
    max_previous_days: int = 7,
) -> pd.DataFrame:
    sql_cfg = output_sql_config(config)
    if not bool(sql_cfg.get("enabled", False)):
        return pd.DataFrame()

    schema = _clean_identifier(sql_cfg.get("schema", "Forecasting"), "schema")
    table = _clean_identifier(sql_cfg.get("forecast_weather_archive_table", "LoadForecastWeatherArchive"), "table")
    max_days = max(1, int(max_previous_days or 7))
    start_ts = pd.Timestamp(start_dt)
    end_ts = pd.Timestamp(end_dt)
    min_archive_ts = start_ts - pd.Timedelta(days=max_days + 1)
    max_archive_ts = end_ts
    if min_archive_ts.tzinfo is not None:
        min_archive_ts = min_archive_ts.tz_convert("UTC").tz_localize(None)
    if max_archive_ts.tzinfo is not None:
        max_archive_ts = max_archive_ts.tz_convert("UTC").tz_localize(None)

    engine = _make_sql_engine(sql_cfg)
    try:
        with engine.connect() as conn:
            if not _table_exists(conn, schema, table):
                return pd.DataFrame()
            df = pd.read_sql_query(
                text(
                    f"""
                    SELECT
                        [SnapshotID], [ArchivedAtUTC], [Source], [ContentHash],
                        [DT], [TempF], [HumidityPct], [CloudCoverPct], [WindSpeedMph],
                        [PrecipIn], [GHI_Wm2], [IsDay]
                    FROM {_full_name(schema, table)}
                    WHERE [DT] >= :start_dt
                      AND [DT] <= :end_dt
                      AND [ArchivedAtUTC] >= :min_archive_dt
                      AND [ArchivedAtUTC] <= :max_archive_dt
                    ORDER BY [ArchivedAtUTC] DESC
                    """
                ),
                conn,
                params={
                    "start_dt": _format_datetime_value(start_ts),
                    "end_dt": _format_datetime_value(end_ts),
                    "min_archive_dt": _format_datetime_value(min_archive_ts),
                    "max_archive_dt": _format_datetime_value(max_archive_ts),
                },
            )
    finally:
        engine.dispose()

    if df.empty:
        return pd.DataFrame()

    tz_name = str(((config.get("project", {}) or {}).get("timezone")) or "UTC")
    dt_local = pd.to_datetime(df["DT"], errors="coerce", utc=True).dt.tz_convert(tz_name)
    archived_local = pd.to_datetime(df["ArchivedAtUTC"], errors="coerce", utc=True).dt.tz_convert(tz_name)
    df = df.assign(
        DT=dt_local,
        Previous_Run_Lead_Days=(dt_local.dt.normalize() - archived_local.dt.normalize()).dt.days + 1,
    )
    df = df[df["Previous_Run_Lead_Days"].between(1, max_days)].copy()
    if df.empty:
        return pd.DataFrame()
    df["Previous_Run_Lead_Days"] = df["Previous_Run_Lead_Days"].astype(int)
    df["_ArchivedAtUTC"] = pd.to_datetime(df["ArchivedAtUTC"], errors="coerce", utc=True)
    df.sort_values(["DT", "Previous_Run_Lead_Days", "_ArchivedAtUTC"], ascending=[True, True, False], inplace=True)
    df.drop_duplicates(subset=["DT", "Previous_Run_Lead_Days"], keep="first", inplace=True)
    cols = ["DT", "TempF", "HumidityPct", "CloudCoverPct", "WindSpeedMph", "PrecipIn", "GHI_Wm2", "IsDay", "Previous_Run_Lead_Days"]
    return df[[col for col in cols if col in df.columns]].sort_values(["DT", "Previous_Run_Lead_Days"]).reset_index(drop=True)


def _canonical_weather_frame(df: pd.DataFrame | None) -> pd.DataFrame:
    if df is None or df.empty:
        return pd.DataFrame()
    cols = ["DT", "TempF", "HumidityPct", "CloudCoverPct", "WindSpeedMph", "PrecipIn", "GHI_Wm2", "IsDay"]
    out = df[[col for col in cols if col in df.columns]].copy()
    return out if "DT" in out.columns else pd.DataFrame()


def load_latest_archived_forecast_weather_snapshot(
    config: dict,
    *,
    current_df: pd.DataFrame | None = None,
) -> pd.DataFrame:
    sql_cfg = output_sql_config(config)
    if not bool(sql_cfg.get("enabled", False)):
        return pd.DataFrame()

    schema = _clean_identifier(sql_cfg.get("schema", "Forecasting"), "schema")
    table = _clean_identifier(sql_cfg.get("forecast_weather_archive_table", "LoadForecastWeatherArchive"), "table")
    current_hash = _frame_hash(_canonical_weather_frame(current_df)) if current_df is not None else ""

    engine = _make_sql_engine(sql_cfg)
    try:
        with engine.connect() as conn:
            if not _table_exists(conn, schema, table):
                return pd.DataFrame()
            snapshot_id = conn.execute(
                text(
                    f"""
                    SELECT TOP (1) [SnapshotID]
                    FROM {_full_name(schema, table)}
                    WHERE (:current_hash = '' OR [ContentHash] <> :current_hash)
                    GROUP BY [SnapshotID], [ArchivedAtUTC]
                    ORDER BY [ArchivedAtUTC] DESC
                    """
                ),
                {"current_hash": current_hash},
            ).scalar()
            if snapshot_id is None:
                return pd.DataFrame()
            df = pd.read_sql_query(
                text(
                    f"""
                    SELECT [DT], [TempF], [HumidityPct], [CloudCoverPct], [WindSpeedMph],
                           [PrecipIn], [GHI_Wm2], [IsDay]
                    FROM {_full_name(schema, table)}
                    WHERE [SnapshotID] = :snapshot_id
                    ORDER BY [DT]
                    """
                ),
                conn,
                params={"snapshot_id": str(snapshot_id)},
            )
    finally:
        engine.dispose()

    if df.empty:
        return pd.DataFrame()
    tz_name = str(((config.get("project", {}) or {}).get("timezone")) or "UTC")
    df["DT"] = pd.to_datetime(df["DT"], errors="coerce", utc=True).dt.tz_convert(tz_name)
    return df.dropna(subset=["DT"]).reset_index(drop=True)


def _clean_table_map(value: dict | None) -> dict[str, str]:
    if not isinstance(value, dict):
        return {}
    out: dict[str, str] = {}
    for diagnostic_name, table_name in value.items():
        if table_name is None or str(table_name).strip() == "":
            continue
        out[str(diagnostic_name)] = _clean_identifier(table_name, "table")
    return out


def _one_row_frame_from_dict(value: dict) -> pd.DataFrame:
    row = {}
    for key, item in (value or {}).items():
        if isinstance(item, (dict, list, tuple, set)):
            row[str(key)] = json.dumps(item, default=str, sort_keys=True)
        else:
            row[str(key)] = item
    return pd.DataFrame([row]) if row else pd.DataFrame()


def _replay_diagnostic_frames(
    diagnostics: dict | None,
    replay_table_map: dict[str, str],
) -> dict[str, pd.DataFrame]:
    if not isinstance(diagnostics, dict) or not replay_table_map:
        return {}

    frames: dict[str, pd.DataFrame] = {}
    for diagnostic_name in replay_table_map:
        value = diagnostics.get(diagnostic_name)
        if isinstance(value, pd.DataFrame):
            if not value.empty:
                frames[diagnostic_name] = value
        elif diagnostic_name == "rolling_origin_replay_summary" and isinstance(value, dict):
            frame = _one_row_frame_from_dict(value)
            if not frame.empty:
                frames[diagnostic_name] = frame
    return frames


def persist_run_outputs(
    config: dict,
    *,
    forecast_df: pd.DataFrame | None,
    backtest_df: pd.DataFrame | None = None,
    weather_df: pd.DataFrame | None = None,
    replay_diagnostics: dict | None = None,
    source: str = "forecasting.main",
    metadata: dict | None = None,
    run_id: str | None = None,
) -> str | None:
    sql_cfg = output_sql_config(config)
    if not bool(sql_cfg.get("enabled", False)):
        return None

    schema = _clean_identifier(sql_cfg.get("schema", "Forecasting"), "schema")
    run_table = _clean_identifier(sql_cfg.get("run_table", "LoadForecastRun"), "table")
    forecast_table = _clean_identifier(sql_cfg.get("forecast_table", "LoadForecastOutput"), "table")
    backtest_table = _clean_identifier(sql_cfg.get("backtest_table", "LoadForecastBacktest"), "table")
    weather_table = _clean_identifier(sql_cfg.get("weather_table", "LoadForecastWeather"), "table")
    replay_table_map = _clean_table_map(sql_cfg.get("replay_tables"))
    chunksize = int(sql_cfg.get("chunksize") or 1000)

    rid = str(run_id or uuid.uuid4())
    now = pd.Timestamp.now(tz="UTC").tz_localize(None).to_pydatetime()
    first_dt, last_dt = _dt_bounds(forecast_df)
    project_name = str((config.get("project", {}) or {}).get("name") or "")
    content_hash = _frame_hash(forecast_df)
    replay_frames = _replay_diagnostic_frames(replay_diagnostics, replay_table_map)
    metadata_out = dict(metadata or {})
    if replay_frames:
        metadata_out["replay_sql_tables"] = {
            name: {"table": replay_table_map[name], "rows": int(len(frame))}
            for name, frame in replay_frames.items()
        }

    engine = _make_sql_engine(sql_cfg)
    try:
        with engine.begin() as conn:
            _ensure_schema(conn, schema)
            _ensure_run_table(conn, schema, run_table)

            prepared_forecast = _prepare_frame_for_sql(forecast_df, rid, now)
            prepared_backtest = _prepare_frame_for_sql(backtest_df, rid, now)
            prepared_weather = _prepare_frame_for_sql(weather_df, rid, now)
            prepared_replay = {
                name: _prepare_frame_for_sql(frame, rid, now)
                for name, frame in replay_frames.items()
            }

            for table, frame in [
                (forecast_table, prepared_forecast),
                (backtest_table, prepared_backtest),
                (weather_table, prepared_weather),
                *[
                    (replay_table_map[name], frame)
                    for name, frame in prepared_replay.items()
                ],
            ]:
                if not frame.empty:
                    _ensure_data_table(conn, schema, table, frame)

            _insert_run_metadata(
                conn,
                schema,
                run_table,
                run_id=rid,
                run_started_at_utc=now,
                inserted_at_utc=now,
                project_name=project_name,
                source=source,
                forecast_rows=len(forecast_df) if forecast_df is not None else 0,
                backtest_rows=len(backtest_df) if backtest_df is not None else 0,
                weather_rows=len(weather_df) if weather_df is not None else 0,
                first_output_dt=first_dt,
                last_output_dt=last_dt,
                content_hash=content_hash,
                metadata=metadata_out,
            )

            _append_frame(conn, schema, forecast_table, prepared_forecast, chunksize)
            _append_frame(conn, schema, backtest_table, prepared_backtest, chunksize)
            _append_frame(conn, schema, weather_table, prepared_weather, chunksize)
            for name, frame in prepared_replay.items():
                _append_frame(conn, schema, replay_table_map[name], frame, chunksize)
    finally:
        engine.dispose()

    return rid


def _read_run_frame(conn, schema: str, table: str, run_id: str) -> pd.DataFrame:
    if not _table_exists(conn, schema, table):
        return pd.DataFrame()
    full = _full_name(schema, table)
    order_sql = "ORDER BY [DT]" if "DT" in _existing_columns(conn, schema, table) else "ORDER BY [OutputRowID]"
    df = pd.read_sql_query(
        text(f"SELECT * FROM {full} WHERE [RunID] = :run_id {order_sql}"),
        conn,
        params={"run_id": run_id},
    )
    return df.drop(columns=[c for c in ["OutputRowID", "RunID", "InsertedAtUTC"] if c in df.columns])


def load_latest_run_outputs(config: dict) -> dict[str, object]:
    sql_cfg = output_sql_config(config)
    schema = _clean_identifier(sql_cfg.get("schema", "Forecasting"), "schema")
    run_table = _clean_identifier(sql_cfg.get("run_table", "LoadForecastRun"), "table")
    forecast_table = _clean_identifier(sql_cfg.get("forecast_table", "LoadForecastOutput"), "table")
    backtest_table = _clean_identifier(sql_cfg.get("backtest_table", "LoadForecastBacktest"), "table")
    weather_table = _clean_identifier(sql_cfg.get("weather_table", "LoadForecastWeather"), "table")
    replay_table_map = _clean_table_map(sql_cfg.get("replay_tables"))

    engine = _make_sql_engine(sql_cfg)
    try:
        with engine.connect() as conn:
            if not _table_exists(conn, schema, run_table):
                return {}
            run_full = _full_name(schema, run_table)
            run_id = conn.execute(
                text(
                    f"""
                    SELECT TOP (1) [RunID]
                    FROM {run_full}
                    ORDER BY [RunStartedAtUTC] DESC, [InsertedAtUTC] DESC
                    """
                )
            ).scalar()
            if run_id is None:
                return {}
            run_id = str(run_id)
            diagnostics = {
                name: _read_run_frame(conn, schema, table, run_id)
                for name, table in replay_table_map.items()
            }
            diagnostics = {name: frame for name, frame in diagnostics.items() if not frame.empty}
            return {
                "run_id": run_id,
                "forecast": _read_run_frame(conn, schema, forecast_table, run_id),
                "backtest": _read_run_frame(conn, schema, backtest_table, run_id),
                "weather": _read_run_frame(conn, schema, weather_table, run_id),
                "diagnostics": diagnostics,
            }
    finally:
        engine.dispose()