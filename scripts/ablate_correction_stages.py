from __future__ import annotations

"""Ablation harness for the post-training correction/calibration chain.

The raw XGB/LGB/CatBoost forecast goes through ~15 independently-configurable
correction stages (targeted-residual-meta, seasonal calibration, heat-peak
calibration, warm-ramp, cloud-solar-shape, peak-risk, recent-residual, the stage
selector, focused-scorecard-guard, focused-shape-residual-learner,
weather-robustness-hedge, operational-residual-learner, daily-peak-shadow,
hot-ramp-peak-capture, heat-persistence-peak-capture) before it becomes
Final_Forecast_MWH. Each one was added to fix a specific historical miss, and
none has ever been removed. This script answers the question that's actually
been missing: does each stage still earn its keep on held-out replay data, or
is it dead weight (or actively harmful) now that later stages exist alongside
it?

It does NOT retrain anything per stage. Like scripts/tune_calibration_optuna.py,
it reuses forecasting/tuning/calibration_search.py's split of the rolling-origin
replay into an expensive, training-dependent half (raw per-origin XGB/LGB/CatBoost
forecasts) and a cheap, calibration-dependent half (the correction chain itself,
apply_origin_correction_chain -- a pure function of a raw forecast bundle plus
config). Build the raw-forecast cache once, then this script re-scores baseline
plus one variant per stage (each variant = the full production config with
exactly that one stage's `enabled` flag forced to False) against the same cache,
cheaply, in a single process.

Two-phase workflow, same as tune_calibration_optuna.py:

  1. Build the cache once (expensive -- retrains XGB/LGB/CatBoost per origin):
       python scripts/ablate_correction_stages.py --build-cache \
           --cache-dir forecast_outputs/ablation_cache

  2. Run the ablation against the cache (cheap -- no retraining):
       python scripts/ablate_correction_stages.py \
           --cache-dir forecast_outputs/ablation_cache

Output, per stage: the same named "gates" build_production_readiness_scorecard
reports for a normal replay run (Hot peak days, Peak window hours 14-18, Day 1
only, Days 2-3, Days 4-7, Cloud/solar midday, Shoulder heat transition, Seasonal
rolling origins pooled) -- Before (baseline, stage on) vs After (stage forced
off) MAE/bias/pass, so "did removing this stage move a gate" uses the exact
thresholds already in use, not a new ad hoc metric.

NOT covered: the "Last 45 days" gate comes from a separate, non-cached backtest
(forecasting/backtest/rolling_backtest.py) that has no cheap-rescore split --
ablating it would mean full retraining per stage variant. Its row is included
in the output for completeness but will always show N=0/Pass=False; don't read
anything into it.

Per-origin breakdown for the two peak-relevant gates (Hot peak days, Peak window
hours 14-18): a pooled MAE delta can look like an improvement while actually
being carried by a few origins and hurting most others -- exactly what happened
to two different post-hoc correction prototypes tried earlier (see
forecasting/model/xgb_model.py's hot_peak_scope_mask commit history and the
session notes on the isotonic/GBM pooled-correction attempts). So for those two
gates specifically, this also reports how many origins got better vs worse vs
unchanged when the stage was removed, not just the pooled average -- a stage
that "helps" 3 origins by a lot while quietly hurting 10 others is a prune
candidate for redesign, not a keeper, even if its pooled MAE delta looks
positive.

This script only ever reads the cache and scores in-memory DataFrames. It never
writes to a database, never touches config.yaml, and never retrains a model in
its non-cache-build mode -- safe to run repeatedly.
"""

import argparse
import copy
import json
import sys
import time
import traceback
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from forecasting.config_utils import load_forecast_config
from forecasting.diagnostics.forecast_diagnostics import (
    build_production_readiness_scorecard,
)
from forecasting.tuning.calibration_search import (
    RawOriginBundle,
    build_raw_origin_bundles,
    load_raw_origin_bundles,
    save_raw_origin_bundles,
    score_bundles,
)

# (short name, dotted path to the stage's master `enabled` flag in config.yaml,
# one-line description). Paths verified against config.yaml's actual nesting and
# each module's own _cfg()/_guard_cfg() fallback resolution order, cross-checked
# against tests/test_calibration_search.py's _minimal_config -- not guessed.
STAGES: list[tuple[str, str, str]] = [
    (
        "targeted_residual_meta",
        "calibration.targeted_residual_meta.enabled",
        "Bias/solar-cloud meta correction applied first, before any other stage.",
    ),
    (
        "seasonal_calibration",
        "calibration.seasonal_enabled",
        "Learned seasonal/level residual-calibration lookup.",
    ),
    (
        "heat_peak_calibration",
        "calibration.heat_peak_enabled",
        "Heat-peak-hours residual lookup correction.",
    ),
    (
        "warm_ramp_correction",
        "calibration.warm_ramp_enabled",
        "Warm-day ramp-hours residual correction.",
    ),
    (
        "cloud_solar_shape_correction",
        "calibration.cloud_solar_shape_enabled",
        "Midday cloud/BTM-solar shape correction.",
    ),
    (
        "peak_risk_correction",
        "calibration.peak_risk.enabled",
        "Peak-risk point correction.",
    ),
    (
        "recent_residual_correction",
        "calibration.recent_residual.enabled",
        "Recent-level/AR residual correction (this is what feeds the Last-45 gate in production).",
    ),
    (
        "stage_selector",
        "calibration.stage_selector.enabled",
        "Operational stage selector that can choose among candidate forecast columns.",
    ),
    (
        "focused_scorecard_guard",
        "calibration.stage_selector.focused_scorecard_guard.enabled",
        "Rule-based guard that overrides the forecast in specific scorecard-failure patterns.",
    ),
    (
        "focused_shape_residual_learner",
        "calibration.stage_selector.focused_shape_residual_learner.enabled",
        "Learned residual-shape correction targeted at specific failure patterns.",
    ),
    (
        "weather_robustness_hedge",
        "calibration.weather_robustness_hedge.enabled",
        "Hedges the forecast against weather-forecast uncertainty.",
    ),
    (
        "operational_residual_learner",
        "calibration.operational_residual_learner.enabled",
        "Second-pass learned residual correction (incl. the hot-peak structural/broad shadow candidates).",
    ),
    (
        "daily_peak_shadow_model",
        "calibration.stage_selector.daily_peak_shadow_model.enabled",
        "Separate daily-peak-magnitude/timing model. Shadow-only in production (shadow_mode: true) "
        "as of this writing, so disabling it should be a near-total no-op on Final_Forecast_MWH; a "
        "material gate delta here would be a bug worth chasing on its own.",
    ),
    (
        "hot_ramp_peak_capture",
        "calibration.hot_ramp_peak_capture.enabled",
        "Guarded peak-anchor correction for clear extreme heat-ramp events.",
    ),
    (
        "heat_persistence_peak_capture",
        "calibration.heat_persistence_peak_capture.enabled",
        "Heat-persistence-anchored peak correction (the allow_anchorless_fallback stage revisited "
        "earlier this session).",
    ),
]

HOT_PEAK_TEST_NAME = "Hot peak days"
PEAK_WINDOW_TEST_NAME = "Peak window hours 14-18"
PER_ORIGIN_GATES = {HOT_PEAK_TEST_NAME, PEAK_WINDOW_TEST_NAME}


def _get_path(config: dict, dotted_path: str, default: Any = None) -> Any:
    cur: Any = config
    for key in dotted_path.split("."):
        if not isinstance(cur, dict) or key not in cur:
            return default
        cur = cur[key]
    return cur


def _set_path(config: dict, dotted_path: str, value: Any) -> dict:
    """Deep-copies config and sets dotted_path (creating intermediate dicts as
    needed) to value. Never mutates the input."""
    out = copy.deepcopy(config)
    keys = dotted_path.split(".")
    cur = out
    for key in keys[:-1]:
        nxt = cur.get(key)
        if not isinstance(nxt, dict):
            nxt = {}
            cur[key] = nxt
        cur = nxt
    cur[keys[-1]] = value
    return out


def _print_progress(label: str, advance: int = 0, total: int | None = None) -> None:
    """A plain-print progress callback for run_pipeline(). Without passing *some*
    callback here, run_pipeline's internal _progress() calls are all no-ops --
    _build_cache used to call run_pipeline(config) with no callback at all, which
    meant this entire step (data loading, training every model) printed nothing,
    start to finish. That's indistinguishable from a genuine hang: if this step
    ever takes far longer than expected again, this is what tells you it's still
    working (and roughly where), instead of leaving you to guess whether to kill it."""
    stamp = time.strftime("%H:%M:%S")
    print(f"[{stamp}] {label}", flush=True)


def build_cache(config: dict, cache_dir: Path, origin_limit: int | None) -> None:
    from forecasting.forecast.forecast_pipeline import run_pipeline

    print(
        "Running the full pipeline once to build train_df/features for the raw-forecast cache "
        "(this step trains the production XGB/LGB/CatBoost models on the full history and was "
        "previously silent end-to-end -- now prints each pipeline stage as it starts)...",
        flush=True,
    )
    results = run_pipeline(config, progress_callback=_print_progress)
    train_df = results["historical_fit_df"]
    features = results.get("features", [])
    print(
        f"[{time.strftime('%H:%M:%S')}] Pipeline done: {len(train_df)} training rows "
        f"({train_df['DT'].min()} to {train_df['DT'].max()}), {len(features)} features. "
        f"Starting per-origin replay-bundle build (prints per origin as it goes)...",
        flush=True,
    )
    bundles = build_raw_origin_bundles(
        train_df, features, config, origin_limit=origin_limit
    )
    if not bundles:
        raise SystemExit(
            "No origins produced a usable raw forecast bundle; check "
            "training.rolling_origin_replay config (min_train_days, calibration_days, etc.)."
        )
    paths = save_raw_origin_bundles(bundles, cache_dir)
    print(f"Cached {len(paths)} raw origin bundles to {cache_dir}", flush=True)


def gate_scorecard(bundles: list[RawOriginBundle], config: dict) -> pd.DataFrame:
    """The same named gates a normal replay run reports (build_production_readiness_scorecard),
    scored against a cached raw-forecast bundle set instead of a fresh replay run. recent_df is
    None -- see the module docstring on why the 'Last 45 days' row isn't meaningful here."""
    replay_df = score_bundles(bundles, config)
    return build_production_readiness_scorecard(None, replay_df, config=config)


def _origin_mae_by_gate(replay_df: pd.DataFrame, test_name: str) -> pd.Series:
    """Per-Replay_Origin_ID MAE for the same row mask build_production_readiness_scorecard
    uses for `test_name`, so a pooled gate delta can be checked against its per-origin spread."""
    if replay_df is None or replay_df.empty or "Replay_Origin_ID" not in replay_df.columns:
        return pd.Series(dtype=float)
    hour = pd.to_numeric(replay_df.get("Hour", pd.Series(np.nan, index=replay_df.index)), errors="coerce")
    temp = pd.to_numeric(
        replay_df.get("Temperature_DailyMax", pd.Series(np.nan, index=replay_df.index)), errors="coerce"
    )
    if test_name == HOT_PEAK_TEST_NAME:
        mask = hour.between(16, 20) & temp.ge(90.0)
    elif test_name == PEAK_WINDOW_TEST_NAME:
        mask = hour.between(14, 18)
    else:
        return pd.Series(dtype=float)
    sliced = replay_df.loc[mask]
    if sliced.empty or "Final_Backtest_Forecast_MWH" not in sliced.columns:
        return pd.Series(dtype=float)
    err = pd.to_numeric(sliced["Actual_MWH"], errors="coerce") - pd.to_numeric(
        sliced["Final_Backtest_Forecast_MWH"], errors="coerce"
    )
    return err.abs().groupby(sliced["Replay_Origin_ID"]).mean()


def _per_origin_summary(baseline_replay: pd.DataFrame, variant_replay: pd.DataFrame, test_name: str) -> dict:
    """N origins whose per-origin MAE got better/worse/unchanged when the stage was removed,
    plus the single largest degradation -- catches a pooled 'improvement' that's actually
    concentrated in a few origins while quietly hurting most others."""
    before = _origin_mae_by_gate(baseline_replay, test_name)
    after = _origin_mae_by_gate(variant_replay, test_name)
    common = before.index.intersection(after.index)
    if len(common) == 0:
        return {
            "N_Origins": 0,
            "N_Improved": 0,
            "N_Worsened": 0,
            "N_Unchanged": 0,
            "Max_Single_Origin_Worsening_MWH": np.nan,
            "Max_Single_Origin_Improvement_MWH": np.nan,
        }
    delta = after.loc[common] - before.loc[common]  # positive = worse after removing the stage
    tol = 0.05
    return {
        "N_Origins": int(len(common)),
        "N_Improved": int((delta < -tol).sum()),
        "N_Worsened": int((delta > tol).sum()),
        "N_Unchanged": int((delta.abs() <= tol).sum()),
        "Max_Single_Origin_Worsening_MWH": float(delta.max()),
        "Max_Single_Origin_Improvement_MWH": float(delta.min()),
    }


def run_ablation(
    bundles: list[RawOriginBundle], base_config: dict, stage_filter: set[str] | None
) -> pd.DataFrame:
    baseline_replay = score_bundles(bundles, base_config)
    baseline_gates = build_production_readiness_scorecard(None, baseline_replay, config=base_config)
    baseline_by_test = baseline_gates.set_index("Test")

    rows: list[dict] = []
    stages = [s for s in STAGES if not stage_filter or s[0] in stage_filter]
    for name, path, description in stages:
        current_value = _get_path(base_config, path, default=True)
        variant_config = _set_path(base_config, path, False)
        try:
            variant_replay = score_bundles(bundles, variant_config)
            variant_gates = build_production_readiness_scorecard(None, variant_replay, config=variant_config)
            variant_by_test = variant_gates.set_index("Test")
            error = None
        except Exception as exc:
            # A stage that can't be safely disabled without the pipeline crashing is itself
            # an important ablation finding -- e.g. peak_risk_correction's Adjusted_Forecast_MWH
            # column being read downstream by apply_operational_stage_selector with no fallback
            # when peak_risk is off. Record it and keep going instead of losing every other
            # stage's results to one crash.
            variant_replay = pd.DataFrame()
            variant_by_test = pd.DataFrame(columns=baseline_by_test.columns).set_index(
                pd.Index([], name="Test")
            )
            error = f"{type(exc).__name__}: {exc}"
            print(f"ERROR scoring stage '{name}' with it disabled: {error}", flush=True)
            print(f"--- full traceback for '{name}' ---", flush=True)
            traceback.print_exc()
            print(f"--- end traceback for '{name}' ---", flush=True)

        for test_name in baseline_by_test.index:
            before = baseline_by_test.loc[test_name]
            after = variant_by_test.loc[test_name] if test_name in variant_by_test.index else None
            row = {
                "Stage": name,
                "Stage_Currently_Enabled": bool(current_value) if current_value is not None else True,
                "Stage_Description": description,
                "Config_Path": path,
                "Test": test_name,
                "Error": error,
                "N": int(before.get("N", 0)),
                "Before_MAE_MWH": before.get("MAE_MWH"),
                "After_MAE_MWH": after.get("MAE_MWH") if after is not None else np.nan,
                "Delta_MAE_MWH": (
                    float(after.get("MAE_MWH")) - float(before.get("MAE_MWH"))
                    if after is not None
                    and pd.notna(after.get("MAE_MWH"))
                    and pd.notna(before.get("MAE_MWH"))
                    else np.nan
                ),
                "Before_Bias_MWH": before.get("Bias_MWH"),
                "After_Bias_MWH": after.get("Bias_MWH") if after is not None else np.nan,
                "Before_Pass": bool(before.get("Pass", False)),
                "After_Pass": bool(after.get("Pass", False)) if after is not None else False,
            }
            if test_name in PER_ORIGIN_GATES:
                row.update(_per_origin_summary(baseline_replay, variant_replay, test_name))
            rows.append(row)
        status = f"ERROR ({error})" if error else "ok"
        print(f"Scored stage '{name}' (enabled={current_value}) -> forced False [{status}]", flush=True)

    return pd.DataFrame(rows)


def print_summary(df: pd.DataFrame) -> None:
    if df.empty:
        print("No rows produced.")
        return
    errored = df[df["Error"].notna()]["Stage"].unique().tolist()
    if errored:
        print(
            "\n=== Stages that could NOT be safely disabled (pipeline raised when this "
            "stage was turned off) ==="
        )
        for stage_name in errored:
            msg = df[df["Stage"] == stage_name]["Error"].iloc[0]
            print(f"  {stage_name}: {msg}")
        print(
            "These stages have a hidden dependency from some other, unrelated part of the "
            "pipeline reading a column they produce, with no fallback when they're off -- "
            "a config toggle that looks independent isn't. Worth fixing before trusting "
            "these stages' gate deltas at all."
        )
    peak_rows = df[df["Test"].isin(PER_ORIGIN_GATES) & df["Error"].isna()].copy()
    if peak_rows.empty:
        return
    print("\n=== Hot-peak / peak-window impact of removing each stage (sorted, most load-bearing first) ===")
    peak_rows = peak_rows.sort_values("Delta_MAE_MWH", ascending=False)
    cols = [
        "Stage",
        "Test",
        "Before_MAE_MWH",
        "After_MAE_MWH",
        "Delta_MAE_MWH",
        "N_Improved",
        "N_Worsened",
        "N_Unchanged",
        "Max_Single_Origin_Worsening_MWH",
    ]
    print(peak_rows[cols].to_string(index=False, float_format=lambda x: f"{x:.3f}"))
    print(
        "\nReading this: Delta_MAE_MWH > 0 means removing the stage made that gate's MAE worse "
        "-- the stage is earning its keep. Delta_MAE_MWH <= 0 (or near zero) with N_Worsened "
        "close to N_Improved means the stage isn't doing much on this gate pooled, but check "
        "N_Worsened/N_Improved before calling it a prune candidate -- a stage can look pooled-neutral "
        "while helping some origins and hurting others by similar amounts."
    )


def main() -> int:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument("--config", type=str, default=None, help="Path to config.yaml")
    parser.add_argument(
        "--cache-dir", type=Path, default=Path("forecast_outputs/ablation_cache")
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
        "--stages",
        action="append",
        dest="stages",
        default=None,
        help="Only ablate this stage name (repeatable). Default: all stages. "
        f"Valid names: {', '.join(s[0] for s in STAGES)}",
    )
    parser.add_argument(
        "--baseline-only",
        action="store_true",
        help="Skip the per-stage ablation entirely and just print/save the gate scorecard for "
        "this cache+config as-is (no stage forced off). Useful for comparing two different "
        "caches -- e.g. one built with model.asymmetric_loss.enabled: true vs the normal one --  "
        "under the same correction-chain config, without ablating anything.",
    )
    parser.add_argument(
        "--output-csv",
        type=Path,
        default=Path("forecast_outputs/ablation_results.csv"),
    )
    args = parser.parse_args()

    config = load_forecast_config(args.config)

    if args.build_cache:
        build_cache(config, args.cache_dir, args.origin_limit)
        return 0

    if not args.cache_dir.exists():
        raise SystemExit(
            f"No cache found at {args.cache_dir}. Run with --build-cache first "
            "(expensive, one-time) before running the ablation."
        )

    bundles = load_raw_origin_bundles(args.cache_dir)
    print(f"Loaded {len(bundles)} cached raw origin bundles from {args.cache_dir}", flush=True)

    if args.baseline_only:
        gates = gate_scorecard(bundles, config)
        args.output_csv.parent.mkdir(parents=True, exist_ok=True)
        gates.to_csv(args.output_csv, index=False)
        print(f"\nSaved baseline gate scorecard to {args.output_csv}\n")
        cols = ["Test", "N", "MAE_MWH", "Bias_MWH", "Pass"]
        print(gates[[c for c in cols if c in gates.columns]].to_string(
            index=False, float_format=lambda x: f"{x:.3f}"
        ))
        return 0

    stage_filter = set(args.stages) if args.stages else None
    if stage_filter:
        unknown = stage_filter - {s[0] for s in STAGES}
        if unknown:
            raise SystemExit(f"Unknown stage name(s): {sorted(unknown)}")

    results = run_ablation(bundles, config, stage_filter)
    args.output_csv.parent.mkdir(parents=True, exist_ok=True)
    results.to_csv(args.output_csv, index=False)
    print(f"\nSaved full results to {args.output_csv}")
    print_summary(results)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
