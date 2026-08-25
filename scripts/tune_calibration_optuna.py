from __future__ import annotations

"""Automated calibration-parameter search: replaces the historical hand-tuning workflow
(run replay, eyeball the scorecard, hand-edit one config.yaml number, repeat) with an Optuna
search driven by the same `rolling_origin_replay_scorecard` objective a human would eyeball.

Two-phase workflow, because building the raw per-origin forecast cache is the expensive,
training-dependent part while re-scoring calibration parameters against it is cheap (see
forecasting/tuning/calibration_search.py for why):

  1. Build the cache once (retrains XGB/LGB/CatBoost per origin -- slow; this is exactly the
     work a normal --rolling-origin-replay run already does, just cached instead of thrown
     away):
       python scripts/tune_calibration_optuna.py --build-cache \
           --cache-dir forecast_outputs/calibration_search_cache

  2. Run (and re-run) the search against the cache as many times as you like (fast -- no
     retraining, just re-applying the correction chain per trial):
       python scripts/tune_calibration_optuna.py \
           --cache-dir forecast_outputs/calibration_search_cache --n-trials 100

The search only ever varies calibration.* parameters (see
forecasting/tuning/optuna_tuning.py's V125_PARAM_SPACE) -- it never changes model/feature/
training config, so a cache built once stays valid across searches as long as you don't
change those settings in config.yaml between build and search.

IMPORTANT -- overfitting: an optimizer can overfit a backtest window just as easily as a
human tuning by hand, just faster, and a single search/holdout split can pass "by luck" on a
small holdout. So the search runs multiple independent repeats (--n-repeats, default 5) with
different random search/holdout splits of a shared origin pool:

  - Each repeat gets its own Optuna study and its own best params.
  - Cross-repeat agreement is the real signal: parameters that land in a similar place every
    repeat are ones the data actually pins down; parameters that swing wildly between repeats
    relative to their search range are flagged as "unstable" and should not be trusted.
  - The repeat whose params are closest to the per-parameter median across all repeats is
    picked as the recommended candidate -- an actual trial's real joint parameter
    combination, not a synthetic average that was never evaluated.
  - A separate final-holdout set (--final-holdout-fraction) is carved out up front and never
    touched by ANY repeat's search or its own per-repeat holdout. The recommended candidate
    is scored against it exactly once, at the end. That score -- not any repeat's search or
    per-repeat-holdout score -- is the number to trust before promoting a config.
"""

import argparse
import json
import os
import platform
import random
import subprocess
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from forecasting.backtest.rolling_origin_replay import build_search_scorecard
from forecasting.config_utils import load_forecast_config
from forecasting.tuning.calibration_search import (
    RawOriginBundle,
    build_raw_origin_bundle_cache,
    load_raw_origin_bundles,
    score_bundles,
)
from forecasting.tuning.optuna_tuning import (
    V125_PARAM_SPACE,
    apply_v125_params,
    optuna_not_installed_message,
    scorecard_objective,
    suggest_v125_params,
)

UNSTABLE_PARAM_THRESHOLD = 0.5
OVERFIT_WARNING_THRESHOLD = 0.15


def split_bundles(
    bundles: list[RawOriginBundle], holdout_fraction: float, seed: int
) -> tuple[list[RawOriginBundle], list[RawOriginBundle]]:
    """Randomly split cached origins into a search set and a held-out set.

    A random split (rather than a time-ordered one) is used because origins are already
    seasonally balanced by `_origin_candidates`; splitting by time could accidentally exclude
    a whole season from the search or the holdout. Always leaves at least one origin for
    search. Deterministic for a given `seed` so results are reproducible.
    """
    if holdout_fraction <= 0 or len(bundles) < 2:
        return sorted(bundles, key=lambda b: b.origin_number), []
    shuffled = sorted(bundles, key=lambda b: b.origin_number)
    random.Random(seed).shuffle(shuffled)
    n_holdout = min(max(1, round(len(shuffled) * holdout_fraction)), len(shuffled) - 1)
    holdout = sorted(shuffled[:n_holdout], key=lambda b: b.origin_number)
    search = sorted(shuffled[n_holdout:], key=lambda b: b.origin_number)
    return search, holdout


def scorecard_for(bundles: list[RawOriginBundle], config: dict) -> pd.DataFrame:
    """Score bundles and build just the scorecard table scorecard_objective() reads.

    Uses build_search_scorecard() rather than the full build_rolling_origin_replay_bundle()
    -- the latter also builds dozens of other diagnostic tables (peak-window bias, hot-peak
    candidates, daily-peak miss, weather-sensitivity detail, ...) meant for human-facing
    replay reports, none of which this search's objective ever reads. Profiling showed that
    difference costing more than half of a trial's wall time.
    """
    replay_df = score_bundles(bundles, config)
    if replay_df.empty:
        return pd.DataFrame()
    return build_search_scorecard(replay_df, config)


def build_cache(config: dict, cache_dir: Path, origin_limit: int | None) -> None:
    from forecasting.forecast.forecast_pipeline import run_pipeline

    print(
        "Running the full pipeline once to build train_df/features for the replay cache...",
        flush=True,
    )
    results = run_pipeline(config)
    train_df = results["historical_fit_df"]
    features = results.get("features", [])
    paths = build_raw_origin_bundle_cache(
        train_df,
        features,
        config,
        cache_dir=cache_dir,
        origin_limit=origin_limit,
    )
    if not paths:
        raise SystemExit(
            "No origins produced a usable raw forecast bundle; check "
            "training.rolling_origin_replay config (min_train_days, calibration_days, etc.)."
        )
    print(f"Cached {len(paths)} raw origin bundles to {cache_dir}", flush=True)


def _normalized_param_vector(params: dict[str, float]) -> list[float]:
    vec = []
    for name, _path, low, high in V125_PARAM_SPACE:
        span = (high - low) or 1.0
        vec.append((params.get(name, low) - low) / span)
    return vec


def stability_report(all_params: list[dict[str, float]]) -> dict[str, dict[str, float]]:
    """Per-parameter spread across repeats, normalized to each parameter's search range.

    A high `range_fraction_of_search_space` means repeats disagree a lot on that parameter
    relative to how wide its allowed range is -- it isn't well pinned down by this data. A
    low value means repeats consistently converge on a similar value regardless of which
    origins they searched against, which is the actual evidence a value is trustworthy.
    """
    report: dict[str, dict[str, float]] = {}
    for name, _path, low, high in V125_PARAM_SPACE:
        values = [p.get(name, low) for p in all_params]
        span = (high - low) or 1.0
        report[name] = {
            "median": float(np.median(values)),
            "min": float(min(values)),
            "max": float(max(values)),
            "range_fraction_of_search_space": float((max(values) - min(values)) / span),
        }
    return report


def pick_central_repeat(all_params: list[dict[str, float]]) -> int:
    """Index of the repeat whose params are closest (normalized Euclidean distance) to the
    per-parameter median across all repeats. This is an actual repeat's real, jointly-searched
    parameter combination -- not a synthetic per-parameter average that was never evaluated
    together and could easily be a combination the objective never actually validated.
    """
    vectors = [_normalized_param_vector(p) for p in all_params]
    medians = [float(np.median(col)) for col in zip(*vectors)]
    distances = [
        sum((v - m) ** 2 for v, m in zip(vec, medians)) ** 0.5 for vec in vectors
    ]
    return int(min(range(len(distances)), key=lambda i: distances[i]))


def _trial_progress_callback(repeat_label: str, n_trials: int):
    """Optuna study.optimize callback: prints one line per completed trial with a running
    average trial duration and an ETA, since optuna's own logging only prints once per
    study by default (or once per trial at INFO level, with no ETA) -- neither tells you
    whether a "still on repeat 1/5" run is on trial 2 or trial 45 of that repeat."""
    start = time.monotonic()

    def _callback(study, trial) -> None:
        completed = trial.number + 1
        elapsed = time.monotonic() - start
        avg_seconds = elapsed / completed
        remaining_trials = max(0, n_trials - completed)
        eta_minutes = avg_seconds * remaining_trials / 60.0
        duration = trial.duration.total_seconds() if trial.duration else float("nan")
        print(
            f"{repeat_label}: trial {completed}/{n_trials} done in {duration:.1f}s "
            f"(avg {avg_seconds:.1f}s/trial, ~{eta_minutes:.1f} min remaining this repeat) "
            f"value={trial.value:.4f} best={study.best_value:.4f}",
            flush=True,
        )

    return _callback


def _run_one_repeat(
    config: dict,
    search_bundles: list[RawOriginBundle],
    repeat_holdout_bundles: list[RawOriginBundle],
    n_trials: int,
    study_name: str,
    storage: str | None,
    objective_weights: dict[str, float] | None,
    verbose: bool = False,
    repeat_label: str = "",
) -> dict:
    import optuna

    def objective(trial: "optuna.Trial") -> float:
        trial_config = suggest_v125_params(trial, config)
        return scorecard_objective(
            scorecard_for(search_bundles, trial_config), objective_weights
        )

    study = optuna.create_study(
        direction="minimize",
        study_name=study_name,
        storage=storage,
        load_if_exists=bool(storage),
    )
    callbacks = [_trial_progress_callback(repeat_label, n_trials)] if verbose else None
    study.optimize(objective, n_trials=n_trials, callbacks=callbacks)

    best_params = dict(study.best_trial.params)
    search_score = float(study.best_value)
    best_config = apply_v125_params(best_params, config)

    repeat_holdout_score = None
    if repeat_holdout_bundles:
        repeat_holdout_score = scorecard_objective(
            scorecard_for(repeat_holdout_bundles, best_config), objective_weights
        )

    return {
        "search_origins": [b.origin_number for b in search_bundles],
        "repeat_holdout_origins": [b.origin_number for b in repeat_holdout_bundles],
        "best_params": best_params,
        "search_set_objective": search_score,
        "repeat_holdout_objective": repeat_holdout_score,
        "trials_dataframe": study.trials_dataframe(),
    }


def _run_repeat_worker(args: tuple) -> dict:
    """Top-level (picklable) entry point for one repeat, run in its own OS process.

    Repeats are independent -- different random search/holdout splits, separate Optuna
    studies, no shared state until run_multi_seed_search aggregates results afterward -- so
    this is safe to parallelize. Takes only picklable, lightweight arguments (a cache_dir
    path, not the loaded bundles themselves) and reloads bundles from disk in-process rather
    than receiving them via multiprocessing IPC, since pickling the cached origins' DataFrames
    across a process boundary would itself be expensive at this cache's scale. Loading from
    disk and re-deriving the deterministic seeded split is cheap by comparison.
    """
    (
        cache_dir_str,
        repeat_index,
        n_repeats,
        seed,
        final_holdout_fraction,
        holdout_fraction,
        n_trials,
        config,
        study_name,
        storage,
        objective_weights,
        verbose,
    ) = args
    bundles = load_raw_origin_bundles(Path(cache_dir_str))
    final_holdout_seed = seed + 10_000
    pool_bundles, _final_holdout_bundles = split_bundles(
        bundles, final_holdout_fraction, final_holdout_seed
    )
    repeat_seed = seed + repeat_index
    search_bundles, repeat_holdout_bundles = split_bundles(
        pool_bundles, holdout_fraction, repeat_seed
    )
    repeat_label = f"Repeat {repeat_index + 1}/{n_repeats} (seed={repeat_seed})"
    print(
        f"{repeat_label}: {len(search_bundles)} search origins, "
        f"{len(repeat_holdout_bundles)} repeat-holdout origins.",
        flush=True,
    )
    result = _run_one_repeat(
        config=config,
        search_bundles=search_bundles,
        repeat_holdout_bundles=repeat_holdout_bundles,
        n_trials=n_trials,
        study_name=f"{study_name}_r{repeat_index}_seed{repeat_seed}",
        storage=storage,
        objective_weights=objective_weights,
        verbose=verbose,
        repeat_label=repeat_label,
    )
    result["repeat_index"] = repeat_index
    result["seed"] = repeat_seed
    return result


def _system_available_memory_bytes() -> int | None:
    """Best-effort, stdlib-only available-memory query (no psutil dependency in this project).
    Returns None if the platform isn't recognized or the query fails, rather than guessing --
    callers should skip the memory-based cap entirely in that case instead of acting on a made
    up number."""
    system = platform.system()
    try:
        if system == "Linux":
            with open("/proc/meminfo") as f:
                for line in f:
                    if line.startswith("MemAvailable:"):
                        return int(line.split()[1]) * 1024
        elif system == "Windows":
            import ctypes

            class _MemoryStatusEx(ctypes.Structure):
                _fields_ = [
                    ("dwLength", ctypes.c_ulong),
                    ("dwMemoryLoad", ctypes.c_ulong),
                    ("ullTotalPhys", ctypes.c_ulonglong),
                    ("ullAvailPhys", ctypes.c_ulonglong),
                    ("ullTotalPageFile", ctypes.c_ulonglong),
                    ("ullAvailPageFile", ctypes.c_ulonglong),
                    ("ullTotalVirtual", ctypes.c_ulonglong),
                    ("ullAvailVirtual", ctypes.c_ulonglong),
                    ("ullAvailExtendedVirtual", ctypes.c_ulonglong),
                ]

            stat = _MemoryStatusEx()
            stat.dwLength = ctypes.sizeof(_MemoryStatusEx)
            if ctypes.windll.kernel32.GlobalMemoryStatusEx(ctypes.byref(stat)):
                return int(stat.ullAvailPhys)
        elif system == "Darwin":
            # macOS has no cheap "available" figure without extra tooling; hw.memsize is
            # total physical memory, so this is intentionally conservative (treats "total" as
            # the ceiling, understating what's actually free).
            out = subprocess.run(
                ["sysctl", "-n", "hw.memsize"], capture_output=True, text=True, timeout=5
            )
            if out.returncode == 0 and out.stdout.strip():
                return int(out.stdout.strip())
    except Exception:
        return None
    return None


def _safe_parallel_repeats(requested: int, n_repeats: int, cache_dir: Path) -> int:
    """Cap --parallel-repeats to something the machine can plausibly run without swapping
    itself into the ground: never more than the CPU count or n_repeats, and -- when available
    memory can be determined -- never more workers than fit in ~80% of it, estimated from the
    cache's on-disk size. Each worker independently loads the full cache (see
    _run_repeat_worker's docstring for why), so memory scales linearly with worker count."""
    cap = max(1, min(requested, n_repeats, os.cpu_count() or 1))
    if cap <= 1:
        return cap

    available = _system_available_memory_bytes()
    if available is None:
        print(
            f"Could not determine available system memory on this platform -- proceeding "
            f"with --parallel-repeats {cap} unchecked. Each worker loads the full cache "
            f"independently, so if this machine doesn't have room for {cap} copies of it in "
            f"memory at once, expect heavy swapping or a killed worker; re-run with a lower "
            f"--parallel-repeats if that happens.",
            flush=True,
        )
        return cap

    on_disk_bytes = sum(
        p.stat().st_size for p in Path(cache_dir).glob("origin_*.pkl")
    )
    # Unpickled pandas DataFrames (object-dtype columns, Python string/Timestamp overhead)
    # commonly run 2-4x their on-disk pickle size in memory; 3x is a deliberately rough,
    # conservative estimate, not a measurement.
    per_worker_bytes = on_disk_bytes * 3
    if per_worker_bytes <= 0:
        return cap

    budget_bytes = int(available * 0.8)  # leave headroom for the OS and everything else
    safe_workers = max(1, budget_bytes // per_worker_bytes)
    if safe_workers < cap:
        print(
            f"--parallel-repeats {cap} would need an estimated "
            f"{cap * per_worker_bytes / 1e9:.1f} GB (~{per_worker_bytes / 1e9:.1f} GB/worker, "
            f"a rough estimate from the cache's on-disk size), but only "
            f"~{available / 1e9:.1f} GB is currently available. Reducing to "
            f"--parallel-repeats {safe_workers} to leave headroom. Pass a lower "
            f"--parallel-repeats explicitly to silence this, or free up memory and try again.",
            flush=True,
        )
        cap = int(safe_workers)
    return cap


def run_multi_seed_search(
    config: dict,
    cache_dir: Path,
    n_trials: int,
    n_repeats: int,
    holdout_fraction: float,
    final_holdout_fraction: float,
    seed: int,
    output_dir: Path,
    study_name: str,
    storage: str | None,
    objective_weights: dict[str, float] | None,
    verbose: bool = False,
    parallel_repeats: int = 1,
) -> dict:
    try:
        import optuna

        optuna.logging.set_verbosity(optuna.logging.WARNING)
    except ImportError as exc:
        raise SystemExit(optuna_not_installed_message()) from exc

    bundles = load_raw_origin_bundles(cache_dir)
    if not bundles:
        raise SystemExit(
            f"No cached raw origin bundles found in {cache_dir}. Run with --build-cache first."
        )

    # Carve out a final validation set up front, seeded independently of the repeat seeds so
    # it never lines up with any repeat's own search/holdout split. Never used for any
    # selection decision -- only to report the recommended candidate's true generalization.
    final_holdout_seed = seed + 10_000
    pool_bundles, final_holdout_bundles = split_bundles(
        bundles, final_holdout_fraction, final_holdout_seed
    )
    print(
        f"Loaded {len(bundles)} cached origins: {len(pool_bundles)} in the search pool, "
        f"{len(final_holdout_bundles)} reserved as a final validation set never used during search.",
        flush=True,
    )

    n_workers = _safe_parallel_repeats(parallel_repeats, n_repeats, cache_dir)
    if n_workers > 1:
        import multiprocessing

        print(
            f"Running {n_repeats} repeats across {n_workers} parallel worker process(es)...",
            flush=True,
        )
        worker_args = [
            (
                str(cache_dir),
                i,
                n_repeats,
                seed,
                final_holdout_fraction,
                holdout_fraction,
                n_trials,
                config,
                study_name,
                storage,
                objective_weights,
                verbose,
            )
            for i in range(n_repeats)
        ]
        # Explicit "spawn" context (rather than whatever the platform default is) so behavior
        # matches Windows/macOS in dev too, not just Linux's cheaper fork() -- spawn is what
        # actually exercises pickling of _run_repeat_worker's arguments and re-import of this
        # module in the child process, the failure mode that would otherwise only show up for
        # users on Windows.
        ctx = multiprocessing.get_context("spawn")
        with ctx.Pool(processes=n_workers) as pool:
            repeats = pool.map(_run_repeat_worker, worker_args)
        repeats.sort(key=lambda r: r["repeat_index"])
    else:
        repeats = []
        for i in range(n_repeats):
            repeat_seed = seed + i
            search_bundles, repeat_holdout_bundles = split_bundles(
                pool_bundles, holdout_fraction, repeat_seed
            )
            repeat_label = f"Repeat {i + 1}/{n_repeats} (seed={repeat_seed})"
            print(
                f"{repeat_label}: {len(search_bundles)} search origins, "
                f"{len(repeat_holdout_bundles)} repeat-holdout origins.",
                flush=True,
            )
            result = _run_one_repeat(
                config=config,
                search_bundles=search_bundles,
                repeat_holdout_bundles=repeat_holdout_bundles,
                n_trials=n_trials,
                study_name=f"{study_name}_r{i}_seed{repeat_seed}",
                storage=storage,
                objective_weights=objective_weights,
                verbose=verbose,
                repeat_label=repeat_label,
            )
            result["repeat_index"] = i
            result["seed"] = repeat_seed
            repeats.append(result)

    output_dir.mkdir(parents=True, exist_ok=True)
    trials_frames = []
    for r in repeats:
        df = r["trials_dataframe"].copy()
        df["repeat_index"] = r["repeat_index"]
        df["seed"] = r["seed"]
        trials_frames.append(df)
    pd.concat(trials_frames, ignore_index=True, sort=False).to_csv(
        output_dir / "calibration_search_trials.csv", index=False
    )

    all_params = [r["best_params"] for r in repeats]
    stability = stability_report(all_params)
    central_idx = pick_central_repeat(all_params)
    recommended_params = repeats[central_idx]["best_params"]
    recommended_config = apply_v125_params(recommended_params, config)

    final_holdout_score = None
    if final_holdout_bundles:
        final_scorecard = scorecard_for(final_holdout_bundles, recommended_config)
        final_holdout_score = scorecard_objective(final_scorecard, objective_weights)
        final_scorecard.to_csv(
            output_dir / "calibration_search_final_holdout_scorecard.csv", index=False
        )

    unstable_params = [
        name
        for name, stats in stability.items()
        if stats["range_fraction_of_search_space"] > UNSTABLE_PARAM_THRESHOLD
    ]

    summary = {
        "n_repeats": n_repeats,
        "n_trials_per_repeat": n_trials,
        "repeats": [
            {
                "repeat_index": r["repeat_index"],
                "seed": r["seed"],
                "search_origins": r["search_origins"],
                "repeat_holdout_origins": r["repeat_holdout_origins"],
                "best_params": r["best_params"],
                "search_set_objective": r["search_set_objective"],
                "repeat_holdout_objective": r["repeat_holdout_objective"],
            }
            for r in repeats
        ],
        "parameter_stability": stability,
        "unstable_parameters": unstable_params,
        "recommended_repeat_index": central_idx,
        "recommended_params": recommended_params,
        "final_holdout_origins": [b.origin_number for b in final_holdout_bundles],
        "final_holdout_objective": final_holdout_score,
    }
    (output_dir / "calibration_search_best_params.json").write_text(
        json.dumps(summary, indent=2), encoding="utf-8"
    )
    print(json.dumps(summary, indent=2))

    if unstable_params:
        print(
            "WARNING: these parameters vary widely across repeats relative to their search "
            f"range and are not well pinned down by this data: {unstable_params}. Don't trust "
            "their specific recommended values; consider narrowing the search range or "
            "gathering more origins before promoting them.",
            flush=True,
        )
    search_scores = [r["search_set_objective"] for r in repeats]
    if final_holdout_score is not None and search_scores:
        mean_search = sum(search_scores) / len(search_scores)
        if (
            mean_search
            and (final_holdout_score - mean_search) / abs(mean_search)
            > OVERFIT_WARNING_THRESHOLD
        ):
            print(
                "WARNING: the final held-out objective is notably worse than the average "
                f"search-set objective across repeats (>{int(OVERFIT_WARNING_THRESHOLD * 100)}% relative gap). "
                "The recommended config may be overfit -- verify with a fresh replay before promoting it.",
                flush=True,
            )
    return summary


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Automated calibration-parameter search over the rolling-origin replay."
    )
    parser.add_argument(
        "--config",
        type=str,
        default=None,
        help="Path to config.yaml (default: forecasting/config.yaml)",
    )
    parser.add_argument(
        "--cache-dir",
        type=Path,
        default=Path("forecast_outputs/calibration_search_cache"),
    )
    parser.add_argument(
        "--build-cache",
        action="store_true",
        help="(Re)build the raw per-origin forecast cache (expensive: retrains models)",
    )
    parser.add_argument(
        "--origin-limit",
        type=int,
        default=None,
        help="Cap the number of origins used when building the cache (useful for a quick smoke test)",
    )
    parser.add_argument(
        "--n-trials", type=int, default=50, help="Optuna trials per repeat"
    )
    parser.add_argument(
        "--n-repeats",
        type=int,
        default=5,
        help="Independent search repeats with different search/holdout splits of the pool, to check "
        "whether the search consistently converges on similar parameters or is overfitting a particular split",
    )
    parser.add_argument(
        "--holdout-fraction",
        type=float,
        default=0.25,
        help="Fraction of the search pool held out per repeat, scored once per repeat",
    )
    parser.add_argument(
        "--final-holdout-fraction",
        type=float,
        default=0.2,
        help="Fraction of ALL cached origins reserved up front and never used by any repeat's search -- "
        "the recommended config's score against this is the number to trust",
    )
    parser.add_argument(
        "--seed", type=int, default=42, help="Base seed; repeat i uses seed+i"
    )
    parser.add_argument("--study-name", type=str, default="calibration_search")
    parser.add_argument(
        "--storage",
        type=str,
        default=None,
        help="Optional optuna storage URL (e.g. sqlite:///forecast_outputs/calibration_search.db) to persist/resume studies",
    )
    parser.add_argument("--output-dir", type=Path, default=Path("forecast_outputs"))
    parser.add_argument(
        "--objective-weights",
        type=str,
        default=None,
        help="JSON dict overriding scorecard_objective's default weights",
    )
    parser.add_argument(
        "--verbose",
        action="store_true",
        help="Print one line per completed Optuna trial (duration, running average, ETA for "
        "the current repeat) instead of only printing once per repeat",
    )
    parser.add_argument(
        "--parallel-repeats",
        type=int,
        default=1,
        help="Run this many repeats concurrently in separate worker processes (default 1 = "
        "sequential, unchanged behavior). Repeats are fully independent (different random "
        "search/holdout splits, separate Optuna studies), so this is safe up to min(this, "
        "--n-repeats). Each worker independently reloads the cache from disk, and memory use "
        "scales with how many you run at once -- start low if you're not sure your machine "
        "has the RAM for N copies of the cache in memory simultaneously.",
    )
    args = parser.parse_args()

    config = load_forecast_config(args.config)

    if args.build_cache:
        build_cache(config, args.cache_dir, args.origin_limit)
        return

    weights = json.loads(args.objective_weights) if args.objective_weights else None
    run_multi_seed_search(
        config=config,
        cache_dir=args.cache_dir,
        n_trials=args.n_trials,
        n_repeats=args.n_repeats,
        holdout_fraction=args.holdout_fraction,
        final_holdout_fraction=args.final_holdout_fraction,
        seed=args.seed,
        output_dir=args.output_dir,
        study_name=args.study_name,
        storage=args.storage,
        objective_weights=weights,
        verbose=args.verbose,
        parallel_repeats=args.parallel_repeats,
    )


if __name__ == "__main__":
    main()
