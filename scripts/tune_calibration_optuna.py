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

NOT wired into your normal production run: this never touches config.yaml, and the recommended
params are a suggestion, not an applied setting. Nothing here changes production forecast
behavior until you (or a follow-up tool) copy `recommended_params` into config.yaml's
`calibration.*` keys yourself.

SQL persistence (--save-sql / --no-save-sql, mirrors forecasting/main.py's flags -- defaults to
whatever output_sql.enabled resolves to in config.yaml): if enabled, writes the run summary,
per-repeat results, all trials, and the final-holdout scorecard to their own SQL tables
(output_sql.calibration_search_tables). These are separate tables from your production
forecast/backtest/weather/replay output tables -- a search run cannot collide with or overwrite
a production run's rows.
"""

import argparse
import json
import random
import sys
from pathlib import Path

import numpy as np
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from forecasting.backtest.rolling_origin_replay import build_rolling_origin_replay_bundle
from forecasting.config_utils import load_forecast_config
from forecasting.tuning.calibration_search import (
    RawOriginBundle,
    build_raw_origin_bundles,
    load_raw_origin_bundles,
    save_raw_origin_bundles,
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
    replay_df = score_bundles(bundles, config)
    if replay_df.empty:
        return pd.DataFrame()
    bundle = build_rolling_origin_replay_bundle(replay_df, config)
    return bundle.get("rolling_origin_replay_scorecard", pd.DataFrame())


def build_cache(config: dict, cache_dir: Path, origin_limit: int | None) -> None:
    from forecasting.forecast.forecast_pipeline import run_pipeline

    print("Running the full pipeline once to build train_df/features for the replay cache...", flush=True)
    results = run_pipeline(config)
    train_df = results["historical_fit_df"]
    features = results.get("features", [])
    bundles = build_raw_origin_bundles(train_df, features, config, origin_limit=origin_limit)
    if not bundles:
        raise SystemExit(
            "No origins produced a usable raw forecast bundle; check "
            "training.rolling_origin_replay config (min_train_days, calibration_days, etc.)."
        )
    paths = save_raw_origin_bundles(bundles, cache_dir)
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
    distances = [sum((v - m) ** 2 for v, m in zip(vec, medians)) ** 0.5 for vec in vectors]
    return int(min(range(len(distances)), key=lambda i: distances[i]))


def _run_one_repeat(
    config: dict,
    search_bundles: list[RawOriginBundle],
    repeat_holdout_bundles: list[RawOriginBundle],
    n_trials: int,
    study_name: str,
    storage: str | None,
    objective_weights: dict[str, float] | None,
) -> dict:
    import optuna

    def objective(trial: "optuna.Trial") -> float:
        trial_config = suggest_v125_params(trial, config)
        return scorecard_objective(scorecard_for(search_bundles, trial_config), objective_weights)

    study = optuna.create_study(
        direction="minimize",
        study_name=study_name,
        storage=storage,
        load_if_exists=bool(storage),
    )
    study.optimize(objective, n_trials=n_trials)

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
) -> dict:
    try:
        import optuna

        optuna.logging.set_verbosity(optuna.logging.WARNING)
    except ImportError as exc:
        raise SystemExit(optuna_not_installed_message()) from exc

    bundles = load_raw_origin_bundles(cache_dir)
    if not bundles:
        raise SystemExit(f"No cached raw origin bundles found in {cache_dir}. Run with --build-cache first.")

    # Carve out a final validation set up front, seeded independently of the repeat seeds so
    # it never lines up with any repeat's own search/holdout split. Never used for any
    # selection decision -- only to report the recommended candidate's true generalization.
    final_holdout_seed = seed + 10_000
    pool_bundles, final_holdout_bundles = split_bundles(bundles, final_holdout_fraction, final_holdout_seed)
    print(
        f"Loaded {len(bundles)} cached origins: {len(pool_bundles)} in the search pool, "
        f"{len(final_holdout_bundles)} reserved as a final validation set never used during search.",
        flush=True,
    )

    repeats: list[dict] = []
    for i in range(n_repeats):
        repeat_seed = seed + i
        search_bundles, repeat_holdout_bundles = split_bundles(pool_bundles, holdout_fraction, repeat_seed)
        print(
            f"Repeat {i + 1}/{n_repeats} (seed={repeat_seed}): "
            f"{len(search_bundles)} search origins, {len(repeat_holdout_bundles)} repeat-holdout origins.",
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
    combined_trials_df = pd.concat(trials_frames, ignore_index=True, sort=False)
    combined_trials_df.to_csv(output_dir / "calibration_search_trials.csv", index=False)

    all_params = [r["best_params"] for r in repeats]
    stability = stability_report(all_params)
    central_idx = pick_central_repeat(all_params)
    recommended_params = repeats[central_idx]["best_params"]
    recommended_config = apply_v125_params(recommended_params, config)

    final_holdout_score = None
    final_scorecard = pd.DataFrame()
    if final_holdout_bundles:
        final_scorecard = scorecard_for(final_holdout_bundles, recommended_config)
        final_holdout_score = scorecard_objective(final_scorecard, objective_weights)
        final_scorecard.to_csv(output_dir / "calibration_search_final_holdout_scorecard.csv", index=False)

    unstable_params = [
        name for name, stats in stability.items() if stats["range_fraction_of_search_space"] > UNSTABLE_PARAM_THRESHOLD
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
    (output_dir / "calibration_search_best_params.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
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
        if mean_search and (final_holdout_score - mean_search) / abs(mean_search) > OVERFIT_WARNING_THRESHOLD:
            print(
                "WARNING: the final held-out objective is notably worse than the average "
                f"search-set objective across repeats (>{int(OVERFIT_WARNING_THRESHOLD * 100)}% relative gap). "
                "The recommended config may be overfit -- verify with a fresh replay before promoting it.",
                flush=True,
            )

    from forecasting.data.output_sql_store import output_sql_enabled, persist_calibration_search_outputs

    if output_sql_enabled(config):
        sql_run_id = persist_calibration_search_outputs(
            config,
            summary=summary,
            trials_df=combined_trials_df,
            final_holdout_scorecard=final_scorecard,
        )
        if sql_run_id:
            print(f"Persisted calibration search outputs to SQL Server RunID: {sql_run_id}", flush=True)

    return summary


def main() -> None:
    parser = argparse.ArgumentParser(description="Automated calibration-parameter search over the rolling-origin replay.")
    parser.add_argument("--config", type=str, default=None, help="Path to config.yaml (default: forecasting/config.yaml)")
    parser.add_argument("--cache-dir", type=Path, default=Path("forecast_outputs/calibration_search_cache"))
    parser.add_argument("--build-cache", action="store_true", help="(Re)build the raw per-origin forecast cache (expensive: retrains models)")
    parser.add_argument("--origin-limit", type=int, default=None, help="Cap the number of origins used when building the cache (useful for a quick smoke test)")
    parser.add_argument("--n-trials", type=int, default=50, help="Optuna trials per repeat")
    parser.add_argument(
        "--n-repeats", type=int, default=5,
        help="Independent search repeats with different search/holdout splits of the pool, to check "
        "whether the search consistently converges on similar parameters or is overfitting a particular split",
    )
    parser.add_argument("--holdout-fraction", type=float, default=0.25, help="Fraction of the search pool held out per repeat, scored once per repeat")
    parser.add_argument(
        "--final-holdout-fraction", type=float, default=0.2,
        help="Fraction of ALL cached origins reserved up front and never used by any repeat's search -- "
        "the recommended config's score against this is the number to trust",
    )
    parser.add_argument("--seed", type=int, default=42, help="Base seed; repeat i uses seed+i")
    parser.add_argument("--study-name", type=str, default="calibration_search")
    parser.add_argument(
        "--storage", type=str, default=None,
        help="Optional optuna storage URL (e.g. sqlite:///forecast_outputs/calibration_search.db) to persist/resume studies",
    )
    parser.add_argument("--output-dir", type=Path, default=Path("forecast_outputs"))
    parser.add_argument("--objective-weights", type=str, default=None, help="JSON dict overriding scorecard_objective's default weights")
    parser.add_argument(
        "--save-sql", action="store_true",
        help="Persist the search summary/repeats/trials/final-holdout-scorecard to SQL Server "
        "(output_sql.calibration_search_tables) -- separate tables from your production forecast/replay output",
    )
    parser.add_argument("--no-save-sql", action="store_true", help="Skip SQL Server persistence for this run")
    args = parser.parse_args()

    config = load_forecast_config(args.config)
    if args.save_sql:
        config.setdefault("output_sql", {})["enabled"] = True
    if args.no_save_sql:
        config.setdefault("output_sql", {})["enabled"] = False

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
    )


if __name__ == "__main__":
    main()
