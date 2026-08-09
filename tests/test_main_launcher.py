import os
import platform
from pathlib import Path
import runpy
import subprocess
import sys
import tempfile
import unittest
from contextlib import redirect_stdout
from datetime import datetime
from io import StringIO
from unittest.mock import patch

from forecasting import main as forecast_main
from forecasting.config_utils import load_forecast_config


class MainLauncherTests(unittest.TestCase):
    def test_no_arg_script_run_defaults_to_save_csv_only(self):
        with patch.object(sys, "argv", ["forecasting/main.py"]):
            with patch.dict(os.environ, {}, clear=True):
                with redirect_stdout(StringIO()):
                    self.assertEqual(
                        forecast_main._resolve_default_argv(None),
                        ["--save-csv"],
                    )

    def test_explicit_cli_args_are_preserved(self):
        with patch.object(sys, "argv", ["forecasting/main.py", "--save-csv"]):
            self.assertEqual(forecast_main._resolve_default_argv(None), ["--save-csv"])

    def test_explicit_dashboard_arg_is_preserved(self):
        with patch.object(
            sys, "argv", ["forecasting/main.py", "--save-csv", "--run-dashboard"]
        ):
            self.assertEqual(
                forecast_main._resolve_default_argv(None),
                ["--save-csv", "--run-dashboard"],
            )

    def test_default_script_args_can_be_disabled(self):
        with patch.object(sys, "argv", ["forecasting/main.py"]):
            with patch.dict(os.environ, {"FORECAST_DEFAULT_RUN_ARGS": "0"}):
                self.assertEqual(forecast_main._resolve_default_argv(None), [])

    @unittest.skipUnless(os.name == "nt", "Windows-only platform WMI guard")
    def test_windows_platform_wmi_probe_is_disabled_by_default(self):
        original_wmi_query = getattr(platform, "_wmi_query", None)
        original_uname_cache = getattr(platform, "_uname_cache", None)
        if original_wmi_query is None:
            self.skipTest("platform._wmi_query is not available")

        try:
            with patch.dict(os.environ, {}, clear=True):
                forecast_main._disable_windows_platform_wmi_probe()

            with self.assertRaises(OSError):
                platform._wmi_query("OS", "Version")
        finally:
            platform._wmi_query = original_wmi_query
            if hasattr(platform, "_uname_cache"):
                platform._uname_cache = original_uname_cache

    def test_direct_script_execution_adds_project_root_to_sys_path(self):
        main_path = Path("forecasting/main.py").resolve()
        project_root = str(main_path.parents[1])
        original_path = list(sys.path)
        try:
            sys.path[:] = [entry for entry in sys.path if entry != project_root]
            namespace = runpy.run_path(str(main_path), run_name="direct_main_probe")

            self.assertEqual(namespace["PROJECT_ROOT"], main_path.parents[1])
            self.assertIn(project_root, sys.path)
        finally:
            sys.path[:] = original_path

    def test_relative_output_dir_resolves_against_project_root(self):
        config = {"project": {"output_dir": "forecast_outputs"}}

        out = forecast_main._normalize_project_paths(config)

        self.assertEqual(
            Path(out["project"]["output_dir"]),
            forecast_main.PROJECT_ROOT / "forecast_outputs",
        )

    def test_config_env_override_expands_and_normalizes_paths(self):
        with tempfile.TemporaryDirectory() as tmp:
            temp_root = Path(tmp)
            parquet_root = temp_root / "PY_LRS"
            parquet_root.mkdir()
            config_path = temp_root / "config.yaml"
            config_path.write_text(
                "\n".join(
                    [
                        "project:",
                        "  output_dir: ${FORECAST_OUTPUT_DIR:-forecast_outputs}",
                        "openmeteo:",
                        "  cache_dir: ${FORECAST_WEATHER_CACHE_DIR:-weather_cache}",
                        "solar:",
                        "  parquet_root: ${FORECAST_SOLAR_PARQUET_ROOT}",
                        "  parquet_root_candidates:",
                        "    - ${FORECAST_DATA_ROOT}",
                        "    - D:/PY_LRS",
                    ]
                ),
                encoding="utf-8",
            )

            with patch.dict(
                os.environ,
                {
                    "FORECAST_CONFIG": "",
                    "FORECAST_OUTPUT_DIR": "server_outputs",
                    "FORECAST_WEATHER_CACHE_DIR": "server_weather_cache",
                    "FORECAST_SOLAR_PARQUET_ROOT": str(parquet_root),
                    "FORECAST_CONFIG_LOCAL": str(temp_root / "missing.local.yaml"),
                },
            ):
                config = load_forecast_config(config_path)

            self.assertEqual(
                Path(config["project"]["output_dir"]),
                forecast_main.PROJECT_ROOT / "server_outputs",
            )
            self.assertEqual(
                Path(config["openmeteo"]["cache_dir"]),
                forecast_main.PROJECT_ROOT / "server_weather_cache",
            )
            self.assertEqual(Path(config["solar"]["parquet_root"]), parquet_root)

    def test_solar_command_uses_configured_paths_and_connection(self):
        with tempfile.TemporaryDirectory() as tmp:
            output_dir = Path(tmp)
            config = {
                "solar": {
                    "driver": "ODBC Driver 18 for SQL Server",
                    "dest_server": "server-b",
                    "dest_db": "ForecastB",
                    "production_source": "rec-parquet",
                    "parquet_root": "D:/PY_LRS",
                    "weather_cache_dir": str(output_dir / "solar_weather"),
                }
            }

            cmd = forecast_main._build_solar_command(output_dir, config=config)
            arg_map = {
                cmd[i]: cmd[i + 1]
                for i in range(len(cmd) - 1)
                if cmd[i].startswith("--")
            }

            self.assertEqual(arg_map["--driver"], "ODBC Driver 18 for SQL Server")
            self.assertEqual(arg_map["--dest-server"], "server-b")
            self.assertEqual(arg_map["--dest-db"], "ForecastB")
            self.assertEqual(arg_map["--parquet-root"], "D:/PY_LRS")
            self.assertEqual(
                arg_map["--weather-cache-dir"], str(output_dir / "solar_weather")
            )
            self.assertEqual(
                arg_map["--output-hourly"],
                str(output_dir / "roseville_solar_forecast_hourly.csv"),
            )

    def test_solar_forecast_refresh_failure_can_use_existing_file(self):
        with tempfile.TemporaryDirectory() as tmp:
            output_dir = Path(tmp)
            hourly = output_dir / "roseville_solar_forecast_hourly.csv"
            hourly.write_text("IntervalStartDT,Forecast_MW\n", encoding="utf-8")

            with patch.object(
                forecast_main.subprocess,
                "run",
                side_effect=subprocess.CalledProcessError(
                    1, ["python", "solar_forecaster.py"]
                ),
            ):
                with redirect_stdout(StringIO()) as stdout:
                    forecast_main._run_solar_forecast(
                        output_dir, allow_stale_on_failure=True
                    )

            self.assertIn("continuing with existing", stdout.getvalue())

    def test_solar_forecast_refresh_skips_outside_weather_import_window_with_existing_file(
        self,
    ):
        with tempfile.TemporaryDirectory() as tmp:
            output_dir = Path(tmp)
            hourly = output_dir / "roseville_solar_forecast_hourly.csv"
            hourly.write_text("IntervalStartDT,Forecast_MW\n", encoding="utf-8")
            config = {
                "project": {"timezone": "America/Los_Angeles"},
                "openmeteo": {
                    "forecast_import_policy": {
                        "enabled": True,
                        "import_window_start_local": "06:00",
                        "import_window_end_local": "07:45",
                    }
                },
            }

            with (
                patch.object(
                    forecast_main,
                    "_weather_import_window_for_now",
                    return_value=(
                        datetime(2026, 8, 1, 13, 0),
                        datetime(2026, 8, 1, 6, 0),
                        datetime(2026, 8, 1, 7, 45),
                    ),
                ),
                patch.object(
                    forecast_main.subprocess,
                    "run",
                    side_effect=AssertionError("unexpected solar refresh"),
                ),
            ):
                with redirect_stdout(StringIO()) as stdout:
                    forecast_main._run_solar_forecast(output_dir, config=config)

            self.assertIn(
                "Skipping solar forecast refresh outside Open-Meteo morning import window",
                stdout.getvalue(),
            )

    def test_solar_forecast_refresh_raises_outside_weather_import_window_without_existing_file(
        self,
    ):
        with tempfile.TemporaryDirectory() as tmp:
            output_dir = Path(tmp)
            config = {
                "project": {"timezone": "America/Los_Angeles"},
                "openmeteo": {
                    "forecast_import_policy": {
                        "enabled": True,
                        "import_window_start_local": "06:00",
                        "import_window_end_local": "07:45",
                    }
                },
            }

            with patch.object(
                forecast_main,
                "_weather_import_window_for_now",
                return_value=(
                    datetime(2026, 8, 1, 13, 0),
                    datetime(2026, 8, 1, 6, 0),
                    datetime(2026, 8, 1, 7, 45),
                ),
            ):
                with self.assertRaisesRegex(
                    RuntimeError, "outside the configured morning window"
                ):
                    forecast_main._run_solar_forecast(output_dir, config=config)

    def test_solar_forecast_refresh_failure_raises_without_existing_file(self):
        with tempfile.TemporaryDirectory() as tmp:
            output_dir = Path(tmp)

            with patch.object(
                forecast_main.subprocess,
                "run",
                side_effect=subprocess.CalledProcessError(
                    1, ["python", "solar_forecaster.py"]
                ),
            ):
                with self.assertRaises(subprocess.CalledProcessError):
                    forecast_main._run_solar_forecast(
                        output_dir, allow_stale_on_failure=True
                    )

    def test_solar_forecast_replay_validation_requires_backtest_outputs(self):
        with tempfile.TemporaryDirectory() as tmp:
            output_dir = Path(tmp)

            with patch.object(
                forecast_main.subprocess,
                "run",
                return_value=subprocess.CompletedProcess([], 0),
            ):
                with self.assertRaisesRegex(
                    RuntimeError, "required solar backtest outputs"
                ):
                    forecast_main._run_solar_forecast(
                        output_dir, require_backtest_outputs=True
                    )

    def test_solar_forecast_replay_validation_accepts_fresh_backtest_outputs(self):
        required_names = [
            "roseville_solar_forecast_hourly.csv",
            "roseville_solar_backtest_hourly.csv",
            "roseville_solar_backtest_summary.csv",
            "roseville_solar_backtest_diagnostics.csv",
            "roseville_solar_backtest_top_errors.csv",
            "roseville_solar_backtest_holdout_scorecard.csv",
        ]

        with tempfile.TemporaryDirectory() as tmp:
            output_dir = Path(tmp)

            def fake_run(*_args, **_kwargs):
                for name in required_names:
                    (output_dir / name).write_text("ok\n", encoding="utf-8")
                return subprocess.CompletedProcess([], 0)

            with patch.object(forecast_main.subprocess, "run", side_effect=fake_run):
                forecast_main._run_solar_forecast(
                    output_dir, require_backtest_outputs=True
                )


if __name__ == "__main__":
    unittest.main()
