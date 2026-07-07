import os
import platform
from pathlib import Path
import runpy
import subprocess
import sys
import tempfile
import unittest
from contextlib import redirect_stdout
from io import StringIO
from unittest.mock import patch

from forecasting import main as forecast_main


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
        with patch.object(sys, "argv", ["forecasting/main.py", "--save-csv", "--run-dashboard"]):
            self.assertEqual(forecast_main._resolve_default_argv(None), ["--save-csv", "--run-dashboard"])

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

    def test_solar_forecast_refresh_failure_can_use_existing_file(self):
        with tempfile.TemporaryDirectory() as tmp:
            output_dir = Path(tmp)
            hourly = output_dir / "roseville_solar_forecast_hourly.csv"
            hourly.write_text("IntervalStartDT,Forecast_MW\n", encoding="utf-8")

            with patch.object(
                forecast_main.subprocess,
                "run",
                side_effect=subprocess.CalledProcessError(1, ["python", "solar_forecaster.py"]),
            ):
                with redirect_stdout(StringIO()) as stdout:
                    forecast_main._run_solar_forecast(output_dir, allow_stale_on_failure=True)

            self.assertIn("continuing with existing", stdout.getvalue())

    def test_solar_forecast_refresh_failure_raises_without_existing_file(self):
        with tempfile.TemporaryDirectory() as tmp:
            output_dir = Path(tmp)

            with patch.object(
                forecast_main.subprocess,
                "run",
                side_effect=subprocess.CalledProcessError(1, ["python", "solar_forecaster.py"]),
            ):
                with self.assertRaises(subprocess.CalledProcessError):
                    forecast_main._run_solar_forecast(output_dir, allow_stale_on_failure=True)

    def test_solar_forecast_replay_validation_requires_backtest_outputs(self):
        with tempfile.TemporaryDirectory() as tmp:
            output_dir = Path(tmp)

            with patch.object(forecast_main.subprocess, "run", return_value=subprocess.CompletedProcess([], 0)):
                with self.assertRaisesRegex(RuntimeError, "required solar backtest outputs"):
                    forecast_main._run_solar_forecast(output_dir, require_backtest_outputs=True)

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
                forecast_main._run_solar_forecast(output_dir, require_backtest_outputs=True)


if __name__ == "__main__":
    unittest.main()
