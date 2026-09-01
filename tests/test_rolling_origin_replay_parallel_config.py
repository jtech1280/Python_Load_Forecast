from __future__ import annotations

import unittest
from unittest.mock import patch

import pandas as pd

from forecasting.backtest.rolling_origin_replay import (
    _score_weather_scenarios,
    _serial_replay_required_for_catboost_gpu,
    _weather_scenario_max_workers,
    _worker_config_for_parallel_replay,
    run_rolling_origin_replay,
)


class WorkerConfigForParallelReplayTests(unittest.TestCase):
    """Regression tests for the CPU-thread-oversubscription fix: origin-level replay
    parallelism (multiprocessing.Pool) previously left each worker's hardware.cpu_threads
    at -1 ("all cores"), so N concurrent workers each independently asked LightGBM/CatBoost
    for every logical core, oversubscribing the CPU by ~N times.
    """

    def test_default_cpu_threads_are_divided_across_worker_processes(self):
        config = {"hardware": {"cpu_threads": -1, "use_gpu": True}}
        with patch("os.cpu_count", return_value=16):
            worker_config = _worker_config_for_parallel_replay(config, num_processes=4)

        self.assertEqual(worker_config["hardware"]["cpu_threads"], 4)
        # Unrelated hardware keys must survive the clone untouched.
        self.assertTrue(worker_config["hardware"]["use_gpu"])
        # The original config passed in must not be mutated.
        self.assertEqual(config["hardware"]["cpu_threads"], -1)

    def test_division_rounds_down_but_never_below_one(self):
        config = {"hardware": {"cpu_threads": -1}}
        with patch("os.cpu_count", return_value=6):
            worker_config = _worker_config_for_parallel_replay(config, num_processes=8)

        self.assertEqual(worker_config["hardware"]["cpu_threads"], 1)

    def test_explicit_positive_cpu_threads_is_left_untouched(self):
        """A user who already set a specific thread count is assumed to have already
        accounted for the replay pool's concurrency; don't second-guess them."""
        config = {"hardware": {"cpu_threads": 12}}
        with patch("os.cpu_count", return_value=16):
            worker_config = _worker_config_for_parallel_replay(config, num_processes=4)

        self.assertIs(worker_config, config)

    def test_single_process_returns_original_config_unchanged(self):
        config = {"hardware": {"cpu_threads": -1}}
        worker_config = _worker_config_for_parallel_replay(config, num_processes=1)
        self.assertIs(worker_config, config)

    def test_missing_hardware_section_defaults_to_dividing_all_cores(self):
        config = {}
        with patch("os.cpu_count", return_value=8):
            worker_config = _worker_config_for_parallel_replay(config, num_processes=2)

        self.assertEqual(worker_config["hardware"]["cpu_threads"], 4)
        self.assertEqual(config, {})


class CatBoostGpuReplayParallelSafetyTests(unittest.TestCase):
    def test_catboost_gpu_requires_serial_replay_by_default(self):
        config = {
            "hardware": {"use_gpu": True},
            "model": {
                "catboost": {
                    "enabled": True,
                    "task_type": "GPU",
                    "require_gpu": True,
                }
            },
        }

        self.assertTrue(
            _serial_replay_required_for_catboost_gpu(
                config, skip_catboost=False, parallel_cfg={}
            )
        )

    def test_require_gpu_overrides_stale_cpu_task_type_for_serial_replay_guard(self):
        config = {
            "hardware": {"use_gpu": True},
            "model": {
                "catboost": {
                    "enabled": True,
                    "task_type": "CPU",
                    "require_gpu": True,
                }
            },
        }

        self.assertTrue(
            _serial_replay_required_for_catboost_gpu(
                config, skip_catboost=False, parallel_cfg={}
            )
        )

    def test_cpu_catboost_does_not_force_serial_replay(self):
        config = {
            "hardware": {"use_gpu": True},
            "model": {
                "catboost": {
                    "enabled": True,
                    "task_type": "CPU",
                    "require_gpu": False,
                }
            },
        }

        self.assertFalse(
            _serial_replay_required_for_catboost_gpu(
                config, skip_catboost=False, parallel_cfg={}
            )
        )

    def test_skipped_catboost_does_not_force_serial_replay(self):
        config = {
            "hardware": {"use_gpu": True},
            "model": {"catboost": {"enabled": True, "task_type": "GPU"}},
        }

        self.assertFalse(
            _serial_replay_required_for_catboost_gpu(
                config, skip_catboost=True, parallel_cfg={}
            )
        )

    def test_explicit_opt_out_allows_parallel_catboost_gpu(self):
        config = {
            "hardware": {"use_gpu": True},
            "model": {"catboost": {"enabled": True, "task_type": "GPU"}},
        }

        self.assertFalse(
            _serial_replay_required_for_catboost_gpu(
                config,
                skip_catboost=False,
                parallel_cfg={"serial_when_catboost_gpu": False},
            )
        )


class RollingOriginReplayCatBoostGpuExecutionTests(unittest.TestCase):
    def test_catboost_gpu_guard_runs_replay_without_multiprocessing_pool(self):
        train_df = pd.DataFrame(
            {
                "DT": pd.date_range("2026-01-01", periods=48, freq="h"),
                "MWH": range(48),
            }
        )
        config = {
            "hardware": {"use_gpu": True},
            "model": {
                "catboost": {
                    "enabled": True,
                    "task_type": "GPU",
                    "require_gpu": True,
                }
            },
            "training": {
                "rolling_origin_replay": {
                    "parallel": {
                        "enabled": True,
                        "processes": 4,
                        "serial_when_catboost_gpu": True,
                    },
                    "skip_catboost": False,
                    "horizon_days": 1,
                    "calibration_days": 1,
                }
            },
        }
        fake_origins = [
            pd.Timestamp("2026-01-02 00:00:00"),
            pd.Timestamp("2026-01-03 00:00:00"),
        ]

        def fake_single_origin(args):
            origin_number = args[0]
            return pd.DataFrame({"origin_number": [origin_number]}), []

        with (
            patch(
                "forecasting.backtest.rolling_origin_replay._origin_candidates",
                return_value=fake_origins,
            ),
            patch(
                "forecasting.backtest.rolling_origin_replay._run_single_origin_replay",
                side_effect=fake_single_origin,
            ) as run_single,
            patch("forecasting.backtest.rolling_origin_replay.Pool") as pool_cls,
        ):
            result = run_rolling_origin_replay(train_df, features=["MWH"], config=config)

        pool_cls.assert_not_called()
        self.assertEqual(run_single.call_count, 2)
        self.assertEqual(result["origin_number"].tolist(), [1, 2])


class WeatherScenarioMaxWorkersTests(unittest.TestCase):
    def test_defaults_to_serial(self):
        self.assertEqual(_weather_scenario_max_workers({}, n_scenarios=4), 1)

    def test_configured_value_is_used(self):
        config = {
            "training": {
                "rolling_origin_replay": {"parallel": {"weather_scenario_workers": 4}}
            }
        }
        self.assertEqual(_weather_scenario_max_workers(config, n_scenarios=4), 4)

    def test_capped_at_scenario_count(self):
        config = {
            "training": {
                "rolling_origin_replay": {"parallel": {"weather_scenario_workers": 10}}
            }
        }
        self.assertEqual(_weather_scenario_max_workers(config, n_scenarios=4), 4)

    def test_single_scenario_is_always_serial(self):
        config = {
            "training": {
                "rolling_origin_replay": {"parallel": {"weather_scenario_workers": 8}}
            }
        }
        self.assertEqual(_weather_scenario_max_workers(config, n_scenarios=1), 1)

    def test_zero_scenarios_is_serial(self):
        self.assertEqual(_weather_scenario_max_workers({}, n_scenarios=0), 1)


class ScoreWeatherScenariosTests(unittest.TestCase):
    """Serial (default) and parallel (opted-in via config) execution of
    _score_weather_scenarios must produce identical results -- scenarios are
    independent, read-only inference against the same already-trained models."""

    def _scenario_defs(self) -> list[dict]:
        return [
            {"name": "warmer", "temperature_delta_f": 3.0},
            {"name": "cooler", "temperature_delta_f": -3.0},
            {"name": "cloudier_solar_loss", "cloud_cover_delta_norm": 0.30},
            {"name": "clearer_high_solar", "cloud_cover_delta_norm": -0.30},
        ]

    def _fake_raw_prediction_frame(self, *, future_frame, **kwargs):
        # Echo the scenario's marker column back as a distinguishable, deterministic
        # per-scenario result -- lets the test verify each call got the right scenario
        # frame, in any execution order.
        return pd.DataFrame({"Marker": future_frame["Marker"].tolist()})

    def _run(self, config: dict) -> dict[str, pd.DataFrame]:
        def fake_make_scenario_frame(base_future_frame, scenario):
            return pd.DataFrame({"Marker": [scenario["name"]] * 3})

        base_future_frame = pd.DataFrame({"Marker": ["base"] * 3})
        with (
            patch(
                "forecasting.backtest.rolling_origin_replay.make_weather_scenario_frame",
                side_effect=fake_make_scenario_frame,
            ),
            patch(
                "forecasting.backtest.rolling_origin_replay._raw_prediction_frame",
                side_effect=self._fake_raw_prediction_frame,
            ),
        ):
            return _score_weather_scenarios(
                self._scenario_defs(),
                base_future_frame,
                target=pd.DataFrame(),
                hist=pd.DataFrame(),
                features=[],
                ensemble_weights={},
                xgb_model=None,
                lgb_model=None,
                prophet_fit=None,
                prophet_features=[],
                catboost_model=None,
                origin_dt=pd.Timestamp("2026-07-01"),
                origin_number=1,
                config=config,
            )

    def test_serial_and_parallel_produce_identical_results(self):
        serial_result = self._run({})
        parallel_result = self._run(
            {
                "training": {
                    "rolling_origin_replay": {
                        "parallel": {"weather_scenario_workers": 4}
                    }
                }
            }
        )

        self.assertEqual(set(serial_result.keys()), {"warmer", "cooler", "cloudier_solar_loss", "clearer_high_solar"})
        self.assertEqual(set(serial_result.keys()), set(parallel_result.keys()))
        for name in serial_result:
            pd.testing.assert_frame_equal(serial_result[name], parallel_result[name])

    def test_empty_scenario_result_is_dropped(self):
        def fake_make_scenario_frame(base_future_frame, scenario):
            return pd.DataFrame({"Marker": [scenario["name"]]})

        with (
            patch(
                "forecasting.backtest.rolling_origin_replay.make_weather_scenario_frame",
                side_effect=fake_make_scenario_frame,
            ),
            patch(
                "forecasting.backtest.rolling_origin_replay._raw_prediction_frame",
                return_value=pd.DataFrame(),
            ),
        ):
            result = _score_weather_scenarios(
                self._scenario_defs(),
                pd.DataFrame({"Marker": ["base"]}),
                target=pd.DataFrame(),
                hist=pd.DataFrame(),
                features=[],
                ensemble_weights={},
                xgb_model=None,
                lgb_model=None,
                prophet_fit=None,
                prophet_features=[],
                catboost_model=None,
                origin_dt=pd.Timestamp("2026-07-01"),
                origin_number=1,
                config={},
            )
        self.assertEqual(result, {})


if __name__ == "__main__":
    unittest.main()
