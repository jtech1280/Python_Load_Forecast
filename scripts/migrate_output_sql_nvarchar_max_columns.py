"""Shrink NVARCHAR(MAX) text columns in the output SQL tables to a bounded NVARCHAR(n),
without the transaction-log blowup a plain ALTER COLUMN causes on a large table.

_infer_sql_type() used to give every non-numeric/bool/datetime column a blanket
NVARCHAR(MAX), and pyodbc's fast_executemany bulk-insert path (10-100x faster than
the row-by-row default for the multi-thousand-row tables this pipeline writes)
cannot bind to NVARCHAR(MAX)/LOB columns. _infer_sql_type() now bounds new columns
based on their observed length, but _ensure_data_table() never alters an existing
column's type -- only newly created tables/columns get the bounded size. Tables
created before that fix keep their NVARCHAR(MAX) columns until migrated, which is
what this script does.

An earlier version of this script did that with a straight
`ALTER TABLE ... ALTER COLUMN ... NVARCHAR(n)`. On SQL Server that's a size-of-data
operation -- it rewrites every row of the table to change the column's physical
storage, fully logged, all inside one transaction -- and these output tables are
append-only across every run ever done, so on a table with a large accumulated row
count that blew the transaction log up past the size of the database itself.

This version avoids that op entirely:
  1. ADD a new bounded-width column (metadata-only in SQL Server 2012+: nullable,
     no default -- doesn't touch a single existing row, negligible log).
  2. Backfill it from the old column in small batches (--batch-rows, default 2000),
     each batch its own committed transaction, so the log is freed and reusable
     between batches instead of one giant open transaction holding it all.
  3. Verify the backfill is complete (COUNT of any row where the old value isn't
     NULL but the new column is still NULL -- must be 0) before touching anything
     else.
  4. DROP the old NVARCHAR(MAX) column (metadata-only -- SQL Server doesn't
     rewrite rows to drop a column) and rename the new one into its place, so the
     column name callers use doesn't change.

Safe by default: dry run, only prints what each table/column would do. Pass
--execute to actually apply it. Each column is fully migrated (steps 1-4) before
moving to the next, and each step commits independently, so this is safe to
interrupt (Ctrl-C) between columns -- whatever has committed stays committed, nothing
is left in a half-open giant transaction, and re-running the script picks up
wherever it left off (already-bounded columns are simply not NVARCHAR(MAX)
anymore, so _max_nvarchar_columns won't find them again).

Usage:
    python scripts/migrate_output_sql_nvarchar_max_columns.py                     # dry run
    python scripts/migrate_output_sql_nvarchar_max_columns.py --execute           # apply
    python scripts/migrate_output_sql_nvarchar_max_columns.py --execute --batch-rows 500
    python scripts/migrate_output_sql_nvarchar_max_columns.py --execute --batch-delay-seconds 0.2

Before running against a table you know is large, check your database's recovery
model and log space first (SIMPLE recovery reclaims space on each batch's commit
automatically; FULL/bulk-logged needs periodic log backups to do the same -- take
one before starting if you're not already backing up the log frequently).
"""

from __future__ import annotations

import argparse
import platform
import time
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

DEFAULT_BATCH_ROWS = 2000
DEFAULT_BATCH_DELAY_SECONDS = 0.0
_MIGRATION_COLUMN_SUFFIX = "__bounded_migration"


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
              AND COLUMN_NAME NOT LIKE :suffix_pattern
            ORDER BY COLUMN_NAME
            """
        ),
        {
            "schema": schema,
            "table": table,
            "suffix_pattern": f"%{_MIGRATION_COLUMN_SUFFIX}",
        },
    ).fetchall()
    return [row[0] for row in rows]


def _observed_max_length(conn, schema: str, table: str, column: str) -> int:
    result = conn.execute(
        text(f"SELECT MAX(LEN([{column}])) FROM [{schema}].[{table}]")
    ).scalar()
    return int(result) if result is not None else 0


def _unmigrated_row_count(conn, schema: str, table: str, column: str, tmp_col: str) -> int:
    result = conn.execute(
        text(
            f"SELECT COUNT(*) FROM [{schema}].[{table}] "
            f"WHERE [{tmp_col}] IS NULL AND [{column}] IS NOT NULL"
        )
    ).scalar()
    return int(result) if result is not None else 0


def _column_metadata(conn, schema: str, table: str, column: str) -> dict | None:
    row = conn.execute(
        text(
            """
            SELECT DATA_TYPE, CHARACTER_MAXIMUM_LENGTH
            FROM INFORMATION_SCHEMA.COLUMNS
            WHERE TABLE_SCHEMA = :schema
              AND TABLE_NAME = :table
              AND COLUMN_NAME = :column
            """
        ),
        {"schema": schema, "table": table, "column": column},
    ).mappings().first()
    return dict(row) if row else None


def _staging_column_is_usable(metadata: dict, bound: int) -> bool:
    if str(metadata.get("DATA_TYPE") or "").lower() != "nvarchar":
        return False
    length = metadata.get("CHARACTER_MAXIMUM_LENGTH")
    try:
        length_i = int(length)
    except Exception:
        return False
    return length_i == -1 or length_i >= int(bound)


def _migrate_column(
    engine,
    schema: str,
    table: str,
    column: str,
    bound: int,
    *,
    observed_max: int,
    batch_rows: int,
    batch_delay_seconds: float,
    execute: bool,
) -> bool:
    """Add-backfill-drop-rename one column. Returns True on success (or a fully
    planned dry run), False if any step errored."""
    full = f"[{schema}].[{table}]"
    tmp_col = f"{column}{_MIGRATION_COLUMN_SUFFIX}"
    add_stmt = f"ALTER TABLE {full} ADD [{tmp_col}] NVARCHAR({bound}) NULL"
    drop_stmt = f"ALTER TABLE {full} DROP COLUMN [{column}]"
    rename_stmt = f"EXEC sp_rename '{schema}.{table}.{tmp_col}', '{column}', 'COLUMN'"

    with engine.connect() as conn:
        tmp_metadata = _column_metadata(conn, schema, table, tmp_col)

    if not execute:
        staging_step = (
            f"reuse existing [{tmp_col}]"
            if tmp_metadata
            else f"add [{tmp_col}]"
        )
        print(
            f"  PLANNED {schema}.{table}.{column} -> NVARCHAR({bound}) "
            f"(longest value on file: {observed_max} chars): {staging_step}, backfill in "
            f"batches of {batch_rows}, drop old column, rename into place",
            flush=True,
        )
        return True

    try:
        if tmp_metadata:
            if not _staging_column_is_usable(tmp_metadata, bound):
                print(
                    f"  ERROR {schema}.{table}.{column}: existing staging column "
                    f"[{tmp_col}] has incompatible type {tmp_metadata}; not using it.",
                    flush=True,
                )
                return False
            print(
                f"  REUSING existing staging column [{tmp_col}] on {full}",
                flush=True,
            )
        else:
            with engine.begin() as conn:
                conn.execute(text(add_stmt))
            print(f"  ADDED [{tmp_col}] NVARCHAR({bound}) to {full}", flush=True)

        backfill_stmt = text(
            f"UPDATE TOP ({int(batch_rows)}) {full} "
            f"SET [{tmp_col}] = LEFT([{column}], {int(bound)}) "
            f"WHERE [{tmp_col}] IS NULL AND [{column}] IS NOT NULL"
        )
        total_backfilled = 0
        while True:
            with engine.begin() as conn:
                result = conn.execute(backfill_stmt)
                rowcount = result.rowcount if result.rowcount is not None else 0
            total_backfilled += max(rowcount, 0)
            if rowcount <= 0:
                break
            if batch_delay_seconds > 0:
                time.sleep(batch_delay_seconds)
        print(f"  BACKFILLED {total_backfilled} row(s) into [{tmp_col}]", flush=True)

        with engine.connect() as conn:
            remaining = _unmigrated_row_count(conn, schema, table, column, tmp_col)
        if remaining:
            print(
                f"  ERROR {schema}.{table}.{column}: {remaining} row(s) still "
                "unmigrated after backfill loop reported done -- not dropping the "
                "old column. Re-run the script to retry.",
                flush=True,
            )
            return False

        with engine.begin() as conn:
            conn.execute(text(drop_stmt))
        print(f"  DROPPED old [{column}] (NVARCHAR(MAX)) from {full}", flush=True)

        with engine.begin() as conn:
            conn.execute(text(rename_stmt))
        print(f"  RENAMED [{tmp_col}] -> [{column}]", flush=True)
        return True
    except Exception as exc:
        print(f"  ERROR migrating {schema}.{table}.{column}: {exc}", flush=True)
        return False


def _migrate_table(
    engine,
    schema: str,
    table: str,
    *,
    batch_rows: int,
    batch_delay_seconds: float,
    execute: bool,
) -> tuple[int, int, int]:
    """Returns (planned, applied, errors) for this one table."""
    with engine.connect() as conn:
        columns = _max_nvarchar_columns(conn, schema, table)
    if not columns:
        return 0, 0, 0

    planned = applied = errors = 0
    for column in columns:
        with engine.connect() as conn:
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
        ok = _migrate_column(
            engine,
            schema,
            table,
            column,
            bound,
            observed_max=observed_max,
            batch_rows=batch_rows,
            batch_delay_seconds=batch_delay_seconds,
            execute=execute,
        )
        if execute:
            if ok:
                applied += 1
            else:
                errors += 1

    return planned, applied, errors


def main() -> None:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument(
        "--execute",
        action="store_true",
        help="Actually run the migration. Without this flag, only prints what "
        "would change.",
    )
    parser.add_argument(
        "--batch-rows",
        type=int,
        default=DEFAULT_BATCH_ROWS,
        help=f"Rows backfilled per committed batch (default {DEFAULT_BATCH_ROWS}). "
        "Smaller batches use less log space per commit at the cost of more "
        "round trips.",
    )
    parser.add_argument(
        "--batch-delay-seconds",
        type=float,
        default=DEFAULT_BATCH_DELAY_SECONDS,
        help="Pause between backfill batches (default 0). A small delay (e.g. "
        "0.1-0.5) reduces sustained I/O/log pressure on a table other things "
        "might still be reading.",
    )
    parser.add_argument(
        "--table",
        action="append",
        dest="only_tables",
        help="Only migrate this table (repeatable). Default: every configured "
        "output table.",
    )
    args = parser.parse_args()

    config = load_forecast_config()
    sql_cfg = output_sql_config(config)
    schema = sql_cfg["schema"]
    engine = _make_sql_engine(sql_cfg)

    candidate_tables = args.only_tables if args.only_tables else _all_tables(sql_cfg)
    with engine.connect() as inspect_conn:
        existing_tables = [
            table for table in candidate_tables if _table_exists(inspect_conn, schema, table)
        ]

    total_planned = total_applied = total_errors = 0
    tables_touched = 0
    for table in existing_tables:
        planned, applied, errors = _migrate_table(
            engine,
            schema,
            table,
            batch_rows=args.batch_rows,
            batch_delay_seconds=args.batch_delay_seconds,
            execute=args.execute,
        )
        if planned or errors:
            tables_touched += 1
            print(
                f"{schema}.{table}: planned={planned} applied={applied} errors={errors}",
                flush=True,
            )
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
