from __future__ import annotations

"""Compares per-origin signed bias on a peak-relevant gate between two raw-forecast
caches built by ablate_correction_stages.py --build-cache.

Motivation: ablate_correction_stages.py's pooled Bias_MWH already showed the
record_breaking_heat prototype cache running a more positive (further from zero)
bias than the feature-off baseline cache on most gates, even though its pooled MAE
was better on Hot peak days and Peak window hours 14-18. A pooled bias shift can
mean the feature genuinely nudges every origin's forecast the same direction, or it
can mean a handful of origins swung hard while most stayed flat -- those have very
different implications for whether the feature is safe to enable. This answers
which one it is, origin by origin, the same way ablate_correction_stages.py's
_per_origin_summary already does for MAE deltas within a single cache -- this just
does it for signed bias across two different caches.

By default both caches are scored through the SAME config's correction chain
(--config, default config.yaml). That's correct ONLY when the thing you're testing
is baked in at TRAINING time -- e.g. record_breaking_heat's Temp_Excess_Over_
Climatology_F being present in one cache's raw per-origin forecasts and absent in
the other, which --build-cache already locked in before this script ever runs.

It is WRONG when the thing you're testing is a correction-chain (scoring-time)
parameter instead -- e.g. heat_persistence_peak_capture's moderate_min_maxtemp_f,
or anything else read inside apply_origin_correction_chain rather than during raw
bundle training. For those, pass --config-b pointing at the OTHER cache's own
config file, so cache-a is scored with --config and cache-b with --config-b. If
you don't, both caches silently get scored with whatever a single --config (or
plain config.yaml if you passed neither) says for that parameter, and any
"difference" you see is almost certainly just retraining noise between the two
--build-cache runs, not the thing you meant to test -- this happened for real: an
early heat_persistence_peak_capture moderate-tier validation ran exactly this
comparison with only --config (no --config-b) while the correction-chain
parameter under test lived in --config-b's file, and reported gate-by-gate
"improvement" that later turned out to be unfalsifiable, because both caches were
actually scored identically the whole time.

Rule of thumb: if what differs between cache-a and cache-b is a training/feature
config, one shared --config is fine (and cache-b's own knob doesn't matter to
scoring). If what differs is a calibration.* correction-chain knob, you need both
--config and --config-b, matching whatever --config each cache was BUILT with.

Origins are matched by Replay_Origin_ID (origin_NN), which is stable across cache
builds run against the same training window and --origin-limit.

Usage (training-time feature, one shared config):
    python scripts/compare_origin_bias_across_caches.py \\
        --cache-a forecast_outputs/record_breaking_cache \\
        --cache-b forecast_outputs/ablation_cache \\
        --test-name "Hot peak days"

Usage (correction-chain parameter, each cache needs its own matching config):
    python scripts/compare_origin_bias_across_caches.py \\
        --cache-a forecast_outputs/heat_persistence_moderate_full \\
        --config forecasting/config_heat_persistence_full_moderate.yaml \\
        --cache-b forecast_outputs/heat_persistence_baseline_full \\
        --config-b forecasting/config_heat_persistence_full_baseline.yaml \\
        --test-name "Hot peak days"
"""

import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from forecasting.config_utils import load_forecast_config
from forecasting.tuning.calibration_search import load_raw_origin_bundles, score_bundles

HOT_PEAK_TEST_NAME = "Hot peak days"
PEAK_WINDOW_TEST_NAME = "Peak window hours 14-18"


def _gate_mask(replay_df: pd.DataFrame, test_name: str) -> pd.Series:
    hour = pd.to_numeric(replay_df.get("Hour", pd.Series(np.nan, index=replay_df.index)), errors="coerce")
    temp = pd.to_numeric(
        replay_df.get("Temperature_DailyMax", pd.Series(np.nan, index=replay_df.index)), errors="coerce"
    )
    if test_name == HOT_PEAK_TEST_NAME:
        return hour.between(16, 20) & temp.ge(90.0)
    if test_name == PEAK_WINDOW_TEST_NAME:
        return hour.between(14, 18)
    raise ValueError(f"Unsupported test_name: {test_name!r}")


def _origin_signed_bias(replay_df: pd.DataFrame, test_name: str) -> pd.Series:
    """Signed mean error per Replay_Origin_ID (Actual - Forecast), matching the sign
    convention build_production_readiness_scorecard uses for Bias_MWH -- positive means
    under-forecasting (actual ran higher than predicted), negative means over-forecasting."""
    if replay_df is None or replay_df.empty or "Replay_Origin_ID" not in replay_df.columns:
        return pd.Series(dtype=float)
    sliced = replay_df.loc[_gate_mask(replay_df, test_name)]
    if sliced.empty or "Final_Backtest_Forecast_MWH" not in sliced.columns:
        return pd.Series(dtype=float)
    err = pd.to_numeric(sliced["Actual_MWH"], errors="coerce") - pd.to_numeric(
        sliced["Final_Backtest_Forecast_MWH"], errors="coerce"
    )
    return err.groupby(sliced["Replay_Origin_ID"]).mean()


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--cache-a", required=True, help="e.g. forecast_outputs/record_breaking_cache")
    parser.add_argument("--cache-b", required=True, help="e.g. forecast_outputs/ablation_cache")
    parser.add_argument("--label-a", default=None, help="Column label for --cache-a (default: its dir name)")
    parser.add_argument("--label-b", default=None, help="Column label for --cache-b (default: its dir name)")
    parser.add_argument("--test-name", default=HOT_PEAK_TEST_NAME, choices=[HOT_PEAK_TEST_NAME, PEAK_WINDOW_TEST_NAME])
    parser.add_argument("--config", default=None, help="Path to config.yaml used to score --cache-a (default: FORECAST_CONFIG or config.yaml)")
    parser.add_argument(
        "--config-b",
        default=None,
        help=(
            "Path to config.yaml used to score --cache-b. Omit to score cache-b with the SAME "
            "config as cache-a (--config) -- correct only for a training-time-feature "
            "comparison. For a correction-chain (scoring-time) parameter, pass this explicitly "
            "pointing at cache-b's own config, or the comparison will silently score both "
            "caches identically -- see the module docstring."
        ),
    )
    args = parser.parse_args()

    config_a = load_forecast_config(args.config)
    config_b = load_forecast_config(args.config_b) if args.config_b else config_a
    if args.config_b is None:
        print(
            "NOTE: --config-b not given -- scoring both caches with the same config "
            f"({args.config or 'FORECAST_CONFIG or config.yaml'}). This is only correct if "
            "what differs between the two caches was baked in at training time, not a "
            "correction-chain parameter read at scoring time. See the module docstring."
        )
    label_a = args.label_a or Path(args.cache_a).name
    label_b = args.label_b or Path(args.cache_b).name

    bundles_a = load_raw_origin_bundles(args.cache_a)
    bundles_b = load_raw_origin_bundles(args.cache_b)
    if not bundles_a:
        raise SystemExit(f"No cached bundles found in {args.cache_a}")
    if not bundles_b:
        raise SystemExit(f"No cached bundles found in {args.cache_b}")
    print(f"Loaded {len(bundles_a)} bundles from {args.cache_a} ({label_a})")
    print(f"Loaded {len(bundles_b)} bundles from {args.cache_b} ({label_b})")

    replay_a = score_bundles(bundles_a, config_a)
    replay_b = score_bundles(bundles_b, config_b)

    bias_a = _origin_signed_bias(replay_a, args.test_name)
    bias_b = _origin_signed_bias(replay_b, args.test_name)

    common = bias_a.index.intersection(bias_b.index)
    only_a = bias_a.index.difference(bias_b.index)
    only_b = bias_b.index.difference(bias_a.index)
    if len(only_a) or len(only_b):
        print(
            f"WARNING: {len(only_a)} origin(s) only in {label_a}, {len(only_b)} only in {label_b} "
            "-- excluded from the comparison below (caches likely built with different "
            "--origin-limit or a different training window)."
        )
    if len(common) == 0:
        raise SystemExit("No overlapping Replay_Origin_ID values between the two caches; nothing to compare.")

    out = pd.DataFrame(
        {
            f"Bias_{label_a}_MWH": bias_a.loc[common],
            f"Bias_{label_b}_MWH": bias_b.loc[common],
        }
    )
    out["Delta_MWH"] = out[f"Bias_{label_a}_MWH"] - out[f"Bias_{label_b}_MWH"]
    out = out.sort_values("Delta_MWH", key=lambda s: s.abs(), ascending=False)

    print(f"\n=== Per-origin signed bias on '{args.test_name}': {label_a} vs {label_b} ===")
    print(out.to_string(float_format=lambda x: f"{x:.3f}"))

    tol = 0.05
    same_direction = int(((out["Delta_MWH"] > tol) | (out["Delta_MWH"] < -tol)).sum())
    print(
        f"\n{same_direction}/{len(out)} origins moved by more than {tol} MWH between the two caches. "
        f"Mean delta: {out['Delta_MWH'].mean():.3f} MWH, std: {out['Delta_MWH'].std():.3f} MWH. "
        "If the delta is broadly similar in sign and magnitude across origins, the pooled bias shift "
        "is a genuine systematic effect of the feature. If it's a wide spread with a few large outliers "
        "driving most of the pooled number, that's a handful of origins, not a broad pattern -- check "
        "which calendar dates those origins are before trusting the pooled bias change."
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
