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


class MigrateTableTests(unittest.TestCase):
    def test_dry_run_plans_without_calling_alter(self):
        conn = MagicMock()
        with (
            patch.object(
                migrate, "_max_nvarchar_columns", return_value=["Source", "Notes"]
            ),
            patch.object(
                migrate, "_observed_max_length", side_effect=[6, 30]
            ),
        ):
            planned, applied, errors = migrate._migrate_table(
                conn, "Forecasting", "LoadForecastOutput", execute=False
            )

        self.assertEqual((planned, applied, errors), (2, 0, 0))
        conn.execute.assert_not_called()

    def test_execute_applies_alter_for_each_bounded_column(self):
        conn = MagicMock()
        with (
            patch.object(migrate, "_max_nvarchar_columns", return_value=["Source"]),
            patch.object(migrate, "_observed_max_length", return_value=6),
        ):
            planned, applied, errors = migrate._migrate_table(
                conn, "Forecasting", "LoadForecastOutput", execute=True
            )

        self.assertEqual((planned, applied, errors), (1, 1, 0))
        conn.execute.assert_called_once()
        (stmt_arg,), _ = conn.execute.call_args
        stmt_text = str(stmt_arg)
        self.assertIn("ALTER TABLE [Forecasting].[LoadForecastOutput]", stmt_text)
        self.assertIn("ALTER COLUMN [Source] NVARCHAR(100) NULL", stmt_text)

    def test_column_too_long_to_bound_is_skipped_not_altered(self):
        conn = MagicMock()
        with (
            patch.object(migrate, "_max_nvarchar_columns", return_value=["Notes"]),
            patch.object(migrate, "_observed_max_length", return_value=5000),
        ):
            planned, applied, errors = migrate._migrate_table(
                conn, "Forecasting", "LoadForecastOutput", execute=True
            )

        self.assertEqual((planned, applied, errors), (0, 0, 0))
        conn.execute.assert_not_called()

    def test_no_max_columns_is_a_no_op(self):
        conn = MagicMock()
        with patch.object(migrate, "_max_nvarchar_columns", return_value=[]):
            result = migrate._migrate_table(
                conn, "Forecasting", "LoadForecastOutput", execute=True
            )
        self.assertEqual(result, (0, 0, 0))
        conn.execute.assert_not_called()

    def test_alter_failure_on_one_column_does_not_abort_the_others(self):
        conn = MagicMock()
        conn.execute.side_effect = [RuntimeError("locked"), None]
        with (
            patch.object(
                migrate, "_max_nvarchar_columns", return_value=["A", "B"]
            ),
            patch.object(migrate, "_observed_max_length", side_effect=[6, 6]),
        ):
            planned, applied, errors = migrate._migrate_table(
                conn, "Forecasting", "LoadForecastOutput", execute=True
            )

        self.assertEqual((planned, applied, errors), (2, 1, 1))
        self.assertEqual(conn.execute.call_count, 2)


if __name__ == "__main__":
    unittest.main()
