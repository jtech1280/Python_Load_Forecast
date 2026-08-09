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
human tuning by hand, just faster. Origins are split into a search set (what the optimizer
sees) and a held-out set (scored once, at the end, with the winning config). The held-out
score is the number to trust before promoting a config, not the search-set score. If the two
diverge a lot, the script prints a warning -- treat that "win" with suspicion and consider a
larger origin cache or a different split before trusting it.
"""

import argparse
import json
import random
import sys
from pathlib import Path

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
    apply_v125_params,
    optuna_not_installed_message,
    scorecard_objective,
    suggest_v125_params,
)


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


def run_search(
    config: dict,
    cache_dir: Path,
    n_trials: int,
    holdout_fraction: float,
    seed: int,
    output_dir: Path,
    study_name: str,
    storage: str | None,
    objective_weights: dict[str, float] | None,
) -> dict:
    try:
        import optuna
    except ImportError as exc:
        raise SystemExit(optuna_not_installed_message()) from exc

    bundles = load_raw_origin_bundles(cache_dir)
    if not bundles:
        raise SystemExit(f"No cached raw origin bundles found in {cache_dir}. Run with --build-cache first.")
    search_bundles, holdout_bundles = split_bundles(bundles, holdout_fraction, seed)
    print(
        f"Loaded {len(bundles)} cached origins: {len(search_bundles)} for search, "
        f"{len(holdout_bundles)} held out (seed={seed}).",
        flush=True,
    )

    def objective(trial: "optuna.Trial") -> float:
        trial_config = suggest_v125_params(trial, config)
        return scorecard_objective(scorecard_for(search_bundles, trial_config), objective_weights)

    optuna.logging.set_verbosity(optuna.logging.WARNING)
    study = optuna.create_study(
        direction="minimize",
        study_name=study_name,
        storage=storage,
        load_if_exists=bool(storage),
    )
    study.optimize(objective, n_trials=n_trials)

    output_dir.mkdir(parents=True, exist_ok=True)
    study.trials_dataframe().to_csv(output_dir / "calibration_search_trials.csv", index=False)

    best_config = apply_v125_params(study.best_trial.params, config)
    search_score = float(study.best_value)

    holdout_score = None
    if holdout_bundles:
        holdout_scorecard = scorecard_for(holdout_bundles, best_config)
        holdout_score = scorecard_objective(holdout_scorecard, objective_weights)
        holdout_scorecard.to_csv(output_dir / "calibration_search_holdout_scorecard.csv", index=False)

    summary = {
        "n_trials": n_trials,
        "search_origins": [b.origin_number for b in search_bundles],
        "holdout_origins": [b.origin_number for b in holdout_bundles],
        "best_params": study.best_trial.params,
        "search_set_objective": search_score,
        "holdout_set_objective": holdout_score,
        "objective_gap_holdout_minus_search": None if holdout_score is None else holdout_score - search_score,
    }
    (output_dir / "calibration_search_best_params.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(json.dumps(summary, indent=2))

    if holdout_score is not None and search_score and (holdout_score - search_score) / abs(search_score) > 0.15:
        print(
            "WARNING: held-out objective is notably worse than the search-set objective "
            "(>15% relative gap). This config may be overfit to the search origins -- "
            "verify with a fresh replay before promoting it.",
            flush=True,
        )
    return summary


def main() -> None:
    parser = argparse.ArgumentParser(description="Automated calibration-parameter search over the rolling-origin replay.")
    parser.add_argument("--config", type=str, default=None, help="Path to config.yaml (default: forecasting/config.yaml)")
    parser.add_argument("--cache-dir", type=Path, default=Path("forecast_outputs/calibration_search_cache"))
    parser.add_argument("--build-cache", action="store_true", help="(Re)build the raw per-origin forecast cache (expensive: retrains models)")
    parser.add_argument("--origin-limit", type=int, default=None, help="Cap the number of origins used when building the cache (useful for a quick smoke test)")
    parser.add_argument("--n-trials", type=int, default=50)
    parser.add_argument("--holdout-fraction", type=float, default=0.25, help="Fraction of cached origins held out from search, scored once at the end")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--study-name", type=str, default="calibration_search")
    parser.add_argument(
        "--storage", type=str, default=None,
        help="Optional optuna storage URL (e.g. sqlite:///forecast_outputs/calibration_search.db) to persist/resume a study",
    )
    parser.add_argument("--output-dir", type=Path, default=Path("forecast_outputs"))
    parser.add_argument("--objective-weights", type=str, default=None, help="JSON dict overriding scorecard_objective's default weights")
    args = parser.parse_args()

    config = load_forecast_config(args.config)

    if args.build_cache:
        build_cache(config, args.cache_dir, args.origin_limit)
        return

    weights = json.loads(args.objective_weights) if args.objective_weights else None
    run_search(
        config=config,
        cache_dir=args.cache_dir,
        n_trials=args.n_trials,
        holdout_fraction=args.holdout_fraction,
        seed=args.seed,
        output_dir=args.output_dir,
        study_name=args.study_name,
        storage=args.storage,
        objective_weights=weights,
    )


if __name__ == "__main__":
    main()
