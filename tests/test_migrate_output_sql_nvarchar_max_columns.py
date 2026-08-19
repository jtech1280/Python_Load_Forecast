from __future__ import annotations

import importlib
import sys
import unittest
from pathlib import Path
from unittest.mock import MagicMock, patch

SCRIPTS_DIR = Path(__file__).resolve().parents[1] / "scripts"
if str(SCRIPTS_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPTS_DIR))

migrate = importlib.import_module("migrate_output_sql_nvarchar_max_columns")


def _mock_engine():
    """An engine whose .begin()/.connect() both hand out the same mock connection,
    so a test can set conn.execute.side_effect as one linear call sequence matching
    the order _migrate_column actually issues statements in."""
    conn = MagicMock()
    cm = MagicMock()
    cm.__enter__ = MagicMock(return_value=conn)
    cm.__exit__ = MagicMock(return_value=False)
    engine = MagicMock()
    engine.begin = MagicMock(return_value=cm)
    engine.connect = MagicMock(return_value=cm)
    return engine, conn


def _rowcount(n: int) -> MagicMock:
    result = MagicMock()
    result.rowcount = n
    return result


def _scalar(value) -> MagicMock:
    result = MagicMock()
    result.scalar.return_value = value
    return result


class AllTablesTests(unittest.TestCase):
    def test_includes_core_and_replay_tables_deduplicated(self):
        sql_cfg = {
            "forecast_table": "LoadForecastOutput",
            "backtest_table": "LoadForecastBacktest",
            "weather_table": "LoadForecastWeather",
            "forecast_weather_archive_table": "LoadForecastWeatherArchive",
            "replay_tables": {
                "a": "LoadForecastReplaySummary",
                "b": "LoadForecastOutput",  # deliberate collision with forecast_table
            },
        }
        tables = migrate._all_tables(sql_cfg)
        self.assertEqual(
            tables,
            [
                "LoadForecastOutput",
                "LoadForecastBacktest",
                "LoadForecastWeather",
                "LoadForecastWeatherArchive",
                "LoadForecastReplaySummary",
            ],
        )


class MigrateColumnTests(unittest.TestCase):
    def test_dry_run_never_touches_the_engine(self):
        engine, conn = _mock_engine()
        ok = migrate._migrate_column(
            engine,
            "Forecasting",
            "LoadForecastOutput",
            "Source",
            100,
            observed_max=6,
            batch_rows=2000,
            batch_delay_seconds=0,
            execute=False,
        )
        self.assertTrue(ok)
        engine.begin.assert_not_called()
        engine.connect.assert_not_called()

    def test_execute_never_issues_a_bare_alter_column(self):
        """The whole point of this rewrite: no ALTER COLUMN (the fully-logged,
        size-of-data statement that blew up the transaction log) anywhere."""
        engine, conn = _mock_engine()
        conn.execute.side_effect = [
            MagicMock(),  # ADD
            _rowcount(0),  # backfill loop: nothing to do (empty table in this test)
            _scalar(0),  # verification: 0 unmigrated rows
            MagicMock(),  # DROP
            MagicMock(),  # RENAME
        ]
        ok = migrate._migrate_column(
            engine,
            "Forecasting",
            "LoadForecastOutput",
            "Source",
            100,
            observed_max=6,
            batch_rows=2000,
            batch_delay_seconds=0,
            execute=True,
        )
        self.assertTrue(ok)
        issued_sql = [str(call.args[0]) for call in conn.execute.call_args_list]
        self.assertTrue(
            all("ALTER COLUMN" not in sql for sql in issued_sql), issued_sql
        )
        self.assertIn("ADD [Source__bounded_migration] NVARCHAR(100)", issued_sql[0])
        self.assertIn("DROP COLUMN [Source]", issued_sql[3])
        self.assertIn(
            "sp_rename 'Forecasting.LoadForecastOutput.Source__bounded_migration', 'Source'",
            issued_sql[4],
        )

    def test_backfill_loops_until_rowcount_is_zero(self):
        engine, conn = _mock_engine()
        conn.execute.side_effect = [
            MagicMock(),  # ADD
            _rowcount(2000),  # batch 1: full batch, more to do
            _rowcount(2000),  # batch 2: full batch, more to do
            _rowcount(437),  # batch 3: partial batch -- still > 0, one more pass
            _rowcount(0),  # batch 4: nothing left, loop terminates
            _scalar(0),  # verification
            MagicMock(),  # DROP
            MagicMock(),  # RENAME
        ]
        ok = migrate._migrate_column(
            engine,
            "Forecasting",
            "LoadForecastOutput",
            "Source",
            100,
            observed_max=6,
            batch_rows=2000,
            batch_delay_seconds=0,
            execute=True,
        )
        self.assertTrue(ok)
        # 4 backfill batches (loop only stops on an exact 0), +1 ADD +1 verify
        # +1 DROP +1 RENAME = 8 total calls.
        self.assertEqual(conn.execute.call_count, 8)

    def test_failed_verification_skips_drop_and_rename(self):
        engine, conn = _mock_engine()
        conn.execute.side_effect = [
            MagicMock(),  # ADD
            _rowcount(0),  # backfill loop reports done
            _scalar(3),  # verification: still 3 rows unmigrated -- bug/race condition
        ]
        ok = migrate._migrate_column(
            engine,
            "Forecasting",
            "LoadForecastOutput",
            "Source",
            100,
            observed_max=6,
            batch_rows=2000,
            batch_delay_seconds=0,
            execute=True,
        )
        self.assertFalse(ok)
        issued_sql = [str(call.args[0]) for call in conn.execute.call_args_list]
        self.assertEqual(len(issued_sql), 3)
        self.assertTrue(all("DROP" not in sql and "sp_rename" not in sql for sql in issued_sql))

    def test_exception_during_backfill_returns_false_without_raising(self):
        engine, conn = _mock_engine()
        conn.execute.side_effect = [MagicMock(), RuntimeError("log full")]
        ok = migrate._migrate_column(
            engine,
            "Forecasting",
            "LoadForecastOutput",
            "Source",
            100,
            observed_max=6,
            batch_rows=2000,
            batch_delay_seconds=0,
            execute=True,
        )
        self.assertFalse(ok)


class MigrateTableTests(unittest.TestCase):
    def test_skips_columns_too_long_to_bound_without_calling_migrate_column(self):
        engine, _ = _mock_engine()
        with (
            patch.object(migrate, "_max_nvarchar_columns", return_value=["Notes"]),
            patch.object(migrate, "_observed_max_length", return_value=5000),
            patch.object(migrate, "_migrate_column") as mock_migrate_column,
        ):
            planned, applied, errors = migrate._migrate_table(
                engine,
                "Forecasting",
                "LoadForecastOutput",
                batch_rows=2000,
                batch_delay_seconds=0,
                execute=True,
            )
        self.assertEqual((planned, applied, errors), (0, 0, 0))
        mock_migrate_column.assert_not_called()

    def test_tallies_applied_and_errors_across_columns(self):
        engine, _ = _mock_engine()
        with (
            patch.object(
                migrate, "_max_nvarchar_columns", return_value=["A", "B", "C"]
            ),
            patch.object(migrate, "_observed_max_length", return_value=6),
            patch.object(
                migrate, "_migrate_column", side_effect=[True, False, True]
            ) as mock_migrate_column,
        ):
            planned, applied, errors = migrate._migrate_table(
                engine,
                "Forecasting",
                "LoadForecastOutput",
                batch_rows=2000,
                batch_delay_seconds=0,
                execute=True,
            )
        self.assertEqual((planned, applied, errors), (3, 2, 1))
        self.assertEqual(mock_migrate_column.call_count, 3)

    def test_no_max_columns_is_a_no_op(self):
        engine, _ = _mock_engine()
        with patch.object(migrate, "_max_nvarchar_columns", return_value=[]):
            result = migrate._migrate_table(
                engine,
                "Forecasting",
                "LoadForecastOutput",
                batch_rows=2000,
                batch_delay_seconds=0,
                execute=True,
            )
        self.assertEqual(result, (0, 0, 0))


if __name__ == "__main__":
    unittest.main()
