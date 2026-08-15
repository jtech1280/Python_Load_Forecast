"""Run from the project root:  python profile_full_trial.py

Profiles scorecard_for() -- the exact call one Optuna trial makes -- across the real

search-pool origins (holdout_fraction=0.25 applied the same way run_multi_seed_search does).

Prints only function-level timing stats -- no load/weather data is printed or written.

"""

import cProfile

import pstats

import random

import sys

import time

from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

from forecasting.config_utils import load_forecast_config

from forecasting.tuning.calibration_search import load_raw_origin_bundles

from scripts.tune_calibration_optuna import scorecard_for, split_bundles

CACHE_DIR = Path("forecast_outputs/calibration_search_cache")

config = load_forecast_config(None)

bundles = load_raw_origin_bundles(CACHE_DIR)

if not bundles:
    raise SystemExit(f"No cached bundles found in {CACHE_DIR}")

pool_bundles, final_holdout = split_bundles(bundles, 0.2, seed=42 + 10_000)

search_bundles, repeat_holdout = split_bundles(pool_bundles, 0.25, seed=42)

print(f"pool={len(pool_bundles)} search={len(search_bundles)} repeat_holdout={len(repeat_holdout)}")

t0 = time.perf_counter()

scorecard_for(search_bundles, config)

t1 = time.perf_counter()

print(f"\nUnprofiled wall time for scorecard_for() over {len(search_bundles)} search origins: {t1 - t0:.2f}s\n")

pr = cProfile.Profile()

pr.enable()

scorecard_for(search_bundles, config)

pr.disable()

stats = pstats.Stats(pr)

stats.sort_stats("cumulative")

stats.print_stats(40)
