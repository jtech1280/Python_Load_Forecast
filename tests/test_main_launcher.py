import os
from pathlib import Path
import runpy
import sys
import unittest
from contextlib import redirect_stdout
from io import StringIO
from unittest.mock import patch

from forecasting import main as forecast_main


class MainLauncherTests(unittest.TestCase):
    def test_no_arg_script_run_defaults_to_save_csv_and_dashboard(self):
        with patch.object(sys, "argv", ["forecasting/main.py"]):
            with patch.dict(os.environ, {}, clear=True):
                with redirect_stdout(StringIO()):
                    self.assertEqual(
                        forecast_main._resolve_default_argv(None),
                        ["--save-csv", "--run-dashboard"],
                    )

    def test_explicit_cli_args_are_preserved(self):
        with patch.object(sys, "argv", ["forecasting/main.py", "--save-csv"]):
            self.assertEqual(forecast_main._resolve_default_argv(None), ["--save-csv"])

    def test_default_script_args_can_be_disabled(self):
        with patch.object(sys, "argv", ["forecasting/main.py"]):
            with patch.dict(os.environ, {"FORECAST_DEFAULT_RUN_ARGS": "0"}):
                self.assertEqual(forecast_main._resolve_default_argv(None), [])

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


if __name__ == "__main__":
    unittest.main()
