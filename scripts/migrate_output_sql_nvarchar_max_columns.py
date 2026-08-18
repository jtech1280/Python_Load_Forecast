"""Shrink NVARCHAR(MAX) text columns in the output SQL tables to a bounded NVARCHAR(n).

_infer_sql_type() used to give every non-numeric/bool/datetime column a blanket
NVARCHAR(MAX), and pyodbc's fast_executemany bulk-insert path (10-100x faster than
the row-by-row default for the multi-thousand-row tables this pipeline writes)
cannot bind to NVARCHAR(MAX)/LOB columns. _infer_sql_type() now bounds new columns
based on their observed length, but _ensure_data_table() never alters an existing
column's type -- only newly created tables/columns get the bounded size. Tables
created before that fix keep their NVARCHAR(MAX) columns until migrated, which is
what this script does: it sizes each column from the LONGEST VALUE ACTUALLY IN THAT
COLUMN today (via MAX(LEN(...))), with headroom, so nothing existing gets truncated.

Safe by default: dry run, only prints the planned ALTER COLUMN statements and the
observed max length behind each one. Pass --execute to actually apply them. Each
table's changes commit (or roll back) independently, so one table failing does not
affect any other table already migrated in the same run.

Usage:
    python scripts/migrate_output_sql_nvarchar_max_columns.py            # dry run
    python scripts/migrate_output_sql_nvarchar_max_columns.py --execute  # apply
"""

from __future__ import annotations

import argparse
import platform
from pathlib import Path
import sys

platform.machine = lambda: "AMD64"

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from sqlalchemy import text

from forecasting.config_utils import load_forecast_config
from forecasting.data.output_sql_store import (
    _make_sql_engine,
    _table_exists,
    nvarchar_bound_for_observed_length,
    output_sql_config,
)


def _all_tables(sql_cfg: dict) -> list[str]:
    """Every table this pipeline writes free-text columns to. The run table only
    has RunID/InsertedAtUTC, both fixed types, so it's excluded."""
    tables = [
        sql_cfg["forecast_table"],
        sql_cfg["backtest_table"],
        sql_cfg["weather_table"],
        sql_cfg["forecast_weather_archive_table"],
    ]
    tables.extend((sql_cfg.get("replay_tables") or {}).values())
    # dict.values() can repeat if config overrides collide; keep it deterministic.
    seen: set[str] = set()
    ordered = []
    for table in tables:
        if table not in seen:
            seen.add(table)
            ordered.append(table)
    return ordered


def _max_nvarchar_columns(conn, schema: str, table: str) -> list[str]:
    rows = conn.execute(
        text(
            """
            SELECT COLUMN_NAME
            FROM INFORMATION_SCHEMA.COLUMNS
            WHERE TABLE_SCHEMA = :schema
              AND TABLE_NAME = :table
              AND DATA_TYPE = 'nvarchar'
              AND CHARACTER_MAXIMUM_LENGTH = -1
            ORDER BY COLUMN_NAME
            """
        ),
        {"schema": schema, "table": table},
    ).fetchall()
    return [row[0] for row in rows]


def _observed_max_length(conn, schema: str, table: str, column: str) -> int:
    result = conn.execute(
        text(f"SELECT MAX(LEN([{column}])) FROM [{schema}].[{table}]")
    ).scalar()
    return int(result) if result is not None else 0


def _migrate_table(conn, schema: str, table: str, *, execute: bool) -> tuple[int, int, int]:
    """Returns (planned, applied, errors) for this one table."""
    columns = _max_nvarchar_columns(conn, schema, table)
    if not columns:
        return 0, 0, 0

    planned = applied = errors = 0
    for column in columns:
        try:
            observed_max = _observed_max_length(conn, schema, table, column)
        except Exception as exc:
            errors += 1
            print(f"  ERROR reading {schema}.{table}.{column}: {exc}", flush=True)
            continue

        bound = nvarchar_bound_for_observed_length(observed_max)
        if bound is None:
            print(
                f"  SKIP {schema}.{table}.{column}: longest value on file is "
                f"{observed_max} chars, too long to safely bound -- leaving as "
                "NVARCHAR(MAX)",
                flush=True,
            )
            continue

        planned += 1
        stmt = f"ALTER TABLE [{schema}].[{table}] ALTER COLUMN [{column}] NVARCHAR({bound}) NULL"
        if execute:
            try:
                conn.execute(text(stmt))
                applied += 1
                print(f"  APPLIED: {stmt}  (longest value on file: {observed_max} chars)", flush=True)
            except Exception as exc:
                errors += 1
                print(f"  ERROR applying {stmt}: {exc}", flush=True)
        else:
            print(f"  PLANNED: {stmt}  (longest value on file: {observed_max} chars)", flush=True)

    return planned, applied, errors


def main() -> None:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument(
        "--execute",
        action="store_true",
        help="Actually run the ALTER COLUMN statements. Without this flag, only "
        "prints what would change.",
    )
    args = parser.parse_args()

    config = load_forecast_config()
    sql_cfg = output_sql_config(config)
    schema = sql_cfg["schema"]
    engine = _make_sql_engine(sql_cfg)

    total_planned = total_applied = total_errors = 0
    tables_touched = 0
    with engine.connect() as inspect_conn:
        existing_tables = [
            table
            for table in _all_tables(sql_cfg)
            if _table_exists(inspect_conn, schema, table)
        ]

    for table in existing_tables:
        # One transaction per table: a failure on one table's columns doesn't
        # touch another table's already-applied (or still-pending) changes.
        with engine.begin() as conn:
            planned, applied, errors = _migrate_table(
                conn, schema, table, execute=args.execute
            )
        if planned or errors:
            tables_touched += 1
            print(f"{schema}.{table}: planned={planned} applied={applied} errors={errors}", flush=True)
        total_planned += planned
        total_applied += applied
        total_errors += errors

    mode = "Applied" if args.execute else "Dry run -- planned"
    print(
        f"\n{mode} changes to {total_planned} column(s) across {tables_touched} table(s). "
        f"{'Applied=' + str(total_applied) + ', ' if args.execute else ''}Errors={total_errors}.",
        flush=True,
    )
    if not args.execute and total_planned:
        print("Re-run with --execute to apply these changes.", flush=True)


if __name__ == "__main__":
    main()
