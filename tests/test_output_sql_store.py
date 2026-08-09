import os
import platform
import unittest
from unittest.mock import patch

with patch.object(platform, "machine", return_value="AMD64"):
    import pandas as pd
    from forecasting.data import output_sql_store


class OutputSqlStoreTests(unittest.TestCase):
    def test_output_sql_is_enabled_by_default(self):
        self.assertTrue(output_sql_store.output_sql_enabled({}))
        self.assertEqual(output_sql_store.output_sql_config({})["dsn_name"], "Forecast_DB")
        self.assertEqual(
            output_sql_store.output_sql_config({})["replay_tables"]["rolling_origin_replay_results"],
            "LoadForecastReplayResult",
        )
        self.assertEqual(
            output_sql_store.output_sql_config({})["forecast_weather_archive_table"],
            "LoadForecastWeatherArchive",
        )

    def test_replay_table_config_overrides_merge_with_defaults(self):
        cfg = output_sql_store.output_sql_config(
            {
                "output_sql": {
                    "replay_tables": {
                        "rolling_origin_replay_results": "CustomReplayResult",
                    }
                }
            }
        )

        self.assertEqual(cfg["replay_tables"]["rolling_origin_replay_results"], "CustomReplayResult")
        self.assertEqual(
            cfg["replay_tables"]["production_readiness_scorecard"],
            "LoadForecastProductionReadinessScorecard",
        )

    def test_calibration_search_tables_default_and_override_merge(self):
        default_cfg = output_sql_store.output_sql_config({})
        self.assertEqual(
            default_cfg["calibration_search_tables"]["calibration_search_summary"],
            "LoadForecastCalibrationSearchSummary",
        )
        self.assertEqual(
            default_cfg["calibration_search_tables"]["calibration_search_trials"],
            "LoadForecastCalibrationSearchTrial",
        )

        overridden_cfg = output_sql_store.output_sql_config(
            {
                "output_sql": {
                    "calibration_search_tables": {
                        "calibration_search_summary": "CustomCalibrationSummary",
                    }
                }
            }
        )
        self.assertEqual(
            overridden_cfg["calibration_search_tables"]["calibration_search_summary"],
            "CustomCalibrationSummary",
        )
        # Unrelated defaults (repeats/trials/final-holdout table names, and replay_tables)
        # survive the partial override untouched.
        self.assertEqual(
            overridden_cfg["calibration_search_tables"]["calibration_search_trials"],
            "LoadForecastCalibrationSearchTrial",
        )
        self.assertEqual(
            overridden_cfg["replay_tables"]["rolling_origin_replay_results"],
            "LoadForecastReplayResult",
        )

    def test_dashboard_read_env_override(self):
        with patch.dict(os.environ, {"DASH_FROM_SQL": "1"}):
            self.assertTrue(output_sql_store.output_sql_dashboard_read_enabled({}))

        with patch.dict(os.environ, {"DASH_FROM_SQL": "0"}):
            self.assertFalse(
                output_sql_store.output_sql_dashboard_read_enabled(
                    {"output_sql": {"dashboard_read_enabled": True}}
                )
            )

    def test_prepare_frame_adds_run_metadata_and_formats_datetime_offset(self):
        inserted_at = pd.Timestamp("2026-06-23T12:00:00Z").tz_localize(None).to_pydatetime()
        df = pd.DataFrame(
            {
                "DT": ["2026-06-23 05:00:00-07:00"],
                "Replay_Origin_DT": ["2026-06-20 00:00:00"],
                "Forecast": [123.4],
                "Reason": [pd.NA],
                "Payload": [{"source": "test"}],
            }
        )

        out = output_sql_store._prepare_frame_for_sql(df, "run-1", inserted_at)

        self.assertEqual(list(out.columns[:2]), ["RunID", "InsertedAtUTC"])
        self.assertEqual(out.loc[0, "RunID"], "run-1")
        self.assertEqual(out.loc[0, "DT"], "2026-06-23T05:00:00-07:00")
        self.assertEqual(out.loc[0, "Replay_Origin_DT"], "2026-06-20T00:00:00")
        self.assertIsNone(out.loc[0, "Reason"])
        self.assertEqual(out.loc[0, "Payload"], '{"source": "test"}')

    def test_infer_sql_types_for_core_output_columns(self):
        df = pd.DataFrame(
            {
                "DT": ["2026-06-23 05:00:00-07:00"],
                "Forecast": [123.4],
                "IsDay": [1],
                "Production_Risk_Code": ["normal"],
            }
        )

        self.assertEqual(output_sql_store._infer_sql_type("DT", df["DT"]), "DATETIMEOFFSET(7)")
        self.assertEqual(
            output_sql_store._infer_sql_type("Replay_Origin_DT", df["DT"]),
            "DATETIMEOFFSET(7)",
        )
        self.assertEqual(output_sql_store._infer_sql_type("Forecast", df["Forecast"]), "FLOAT")
        self.assertEqual(output_sql_store._infer_sql_type("IsDay", df["IsDay"]), "BIGINT")
        self.assertEqual(
            output_sql_store._infer_sql_type("Production_Risk_Code", df["Production_Risk_Code"]),
            "NVARCHAR(MAX)",
        )

    def test_replay_diagnostic_frames_extracts_summary_results_and_scorecard(self):
        replay_tables = output_sql_store.output_sql_config({})["replay_tables"]
        diagnostics = {
            "rolling_origin_replay_summary": {
                "row_count": 2,
                "horizon_buckets": ["Day1", "Days2to7"],
            },
            "rolling_origin_replay_results": pd.DataFrame(
                {
                    "DT": ["2026-06-23 05:00:00-07:00"],
                    "Replay_Origin_ID": ["origin_01"],
                    "Final_Backtest_Forecast_MWH": [101.5],
                }
            ),
            "production_readiness_scorecard": pd.DataFrame(
                {
                    "Test": ["Seasonal rolling origins"],
                    "Pass": [True],
                }
            ),
            "non_replay_debug_table": pd.DataFrame({"A": [1]}),
        }

        frames = output_sql_store._replay_diagnostic_frames(diagnostics, replay_tables)

        self.assertIn("rolling_origin_replay_summary", frames)
        self.assertIn("rolling_origin_replay_results", frames)
        self.assertIn("production_readiness_scorecard", frames)
        self.assertNotIn("non_replay_debug_table", frames)
        self.assertEqual(frames["rolling_origin_replay_summary"].loc[0, "row_count"], 2)
        self.assertEqual(
            frames["rolling_origin_replay_summary"].loc[0, "horizon_buckets"],
            '["Day1", "Days2to7"]',
        )

    def test_append_frame_uses_explicit_insert_batches(self):
        class FakeConn:
            def __init__(self):
                self.calls = []

            def execute(self, statement, params=None):
                self.calls.append((str(statement), params))

        conn = FakeConn()
        df = pd.DataFrame({"DT": ["2026-06-23T05:00:00-07:00"], "Value": [float("nan")]})

        with patch.object(pd.DataFrame, "to_sql", side_effect=AssertionError("to_sql should not be used")):
            count = output_sql_store._append_frame(conn, "Forecasting", "AnyTable", df, chunksize=1)

        self.assertEqual(count, 1)
        self.assertEqual(len(conn.calls), 1)
        self.assertIn("INSERT INTO [Forecasting].[AnyTable]", conn.calls[0][0])
        self.assertIsNone(conn.calls[0][1][0]["p1"])

    def test_forecast_weather_archive_frame_adds_snapshot_metadata(self):
        archived_at = pd.Timestamp("2026-06-23T12:00:00Z").tz_localize(None).to_pydatetime()
        df = pd.DataFrame(
            {
                "DT": ["2026-06-23 05:00:00-07:00"],
                "TempF": [91.2],
                "CloudCoverPct": [10.0],
                "Unused": ["x"],
            }
        )

        out = output_sql_store._forecast_weather_archive_frame(
            df,
            snapshot_id="snapshot-1",
            archived_at_utc=archived_at,
            source="open_meteo_forecast",
            content_hash="abc",
            first_dt="2026-06-23T12:00:00+00:00",
            last_dt="2026-06-24T12:00:00+00:00",
        )

        self.assertEqual(out.loc[0, "SnapshotID"], "snapshot-1")
        self.assertEqual(out.loc[0, "ContentHash"], "abc")
        self.assertEqual(out.loc[0, "DT"], "2026-06-23T05:00:00-07:00")
        self.assertNotIn("Unused", out.columns)


if __name__ == "__main__":
    unittest.main()
