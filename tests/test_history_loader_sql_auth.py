import unittest
from unittest.mock import Mock, patch

import pandas as pd

from forecasting.data import history_loader


class HistoryLoaderSqlAuthTests(unittest.TestCase):
    def test_connection_attempts_use_trusted_then_sql_auth_fallback(self):
        attempts = history_loader._sql_connection_attempts(
            {
                "dsn_name": "ForecastSource",
                "trusted_connection": True,
                "sql_auth_fallback": True,
                "username": "svc_forecast",
                "password": "secret",
            }
        )

        self.assertEqual(
            [label for label, _ in attempts], ["trusted connection", "SQL auth"]
        )
        self.assertIn("Trusted_Connection=yes", attempts[0][1])
        self.assertIn("UID=svc_forecast", attempts[1][1])
        self.assertIn("PWD=secret", attempts[1][1])

    def test_read_query_falls_back_to_sql_auth_and_disposes_engines(self):
        first_engine = Mock()
        second_engine = Mock()
        expected = pd.DataFrame(
            {"YEAR": [2026], "MONTH": [8], "DAY": [1], "Hr1": [1.0]}
        )

        with (
            patch.object(
                history_loader,
                "_make_sql_engine_from_odbc",
                side_effect=[first_engine, second_engine],
            ) as make_engine,
            patch.object(
                history_loader.pd,
                "read_sql_query",
                side_effect=[RuntimeError("trusted login failed"), expected],
            ) as read_sql_query,
        ):
            actual = history_loader._read_sql_query_with_auth_fallback(
                "select 1",
                {
                    "dsn_name": "ForecastSource",
                    "trusted_connection": True,
                    "sql_auth_fallback": True,
                    "username": "svc_forecast",
                    "password": "secret",
                },
            )

        self.assertIs(actual, expected)
        self.assertEqual(make_engine.call_count, 2)
        self.assertEqual(read_sql_query.call_count, 2)
        first_engine.dispose.assert_called_once()
        second_engine.dispose.assert_called_once()

    def test_fallback_error_scrubs_password(self):
        engine = Mock()

        with (
            patch.object(
                history_loader, "_make_sql_engine_from_odbc", return_value=engine
            ),
            patch.object(
                history_loader.pd,
                "read_sql_query",
                side_effect=RuntimeError("login failed for PWD=secret"),
            ),
        ):
            with self.assertRaisesRegex(RuntimeError, r"PWD=\*\*\*") as raised:
                history_loader._read_sql_query_with_auth_fallback(
                    "select 1",
                    {
                        "dsn_name": "ForecastSource",
                        "trusted_connection": False,
                        "username": "svc_forecast",
                        "password": "secret",
                    },
                )

        self.assertNotIn("secret", str(raised.exception))
        engine.dispose.assert_called_once()


if __name__ == "__main__":
    unittest.main()
