from __future__ import annotations

import unittest
from unittest.mock import patch

from forecasting.backtest.rolling_origin_replay import (
    _worker_config_for_parallel_replay,
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


if __name__ == "__main__":
    unittest.main()
