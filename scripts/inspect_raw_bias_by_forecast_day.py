from __future__ import annotations

"""Breaks each origin's raw-ensemble bias down by Forecast_Day (1 = hours 1-24
ahead, 2 = hours 25-48 ahead, etc.) instead of pooling across the whole horizon,
to test a specific mechanism: this pipeline's multi-day-ahead forecasts are
recursive (recursive_forecast() in rolling_origin_replay.py) -- each day's own
forecast feeds the next day's lag/rolling load features (MWH_Lag24,
MWH_Rolling168, ...) rather than using real actuals beyond day 1. If the raw
model underforecasts even slightly on day 1 of a sustained hot spell, that
under-forecast could compound forward into day 2, day 3, etc.

Motivation: inspect_raw_vs_corrected_hot_day_bias.py (Peak window hours 14-18,
27-origin cache) showed 2026 cool-season origins (Jan-May) with normal,
mixed-sign bias, but 2026 hot-season origins (Jul 6/7/20/21/27/28/29) climbing
steadily -- 2.7, 4.5, 10.1, 9.2, 14.5, 14.6, 15.6 MWH -- as each origin's
horizon reaches deeper into the 2026-07-24..08-10 heat streak. That pattern is
consistent with either (a) a genuine escalating heat-persistence effect the
raw model underweights, compounding via the recursive lag features, or (b) 2026
simply having a stronger heat response than history that's visible regardless
of horizon depth. Comparable-length historical streaks (2022/2023/2024) showed
much smaller raw bias despite similar streak lengths, which doesn't fit a pure
"long streaks are structurally hard" explanation.

If bias grows monotonically with Forecast_Day *within* origin_25/26/27
specifically (day 1 much smaller than day 7+), that's direct evidence of
compounding. If it's already large on day 1 and roughly flat across the
horizon, the model is simply wrong about this event's heat response from the
start, and the recursive chain isn't amplifying anything -- a different fix is
needed either way, but this tells us which.

Usage:
    python scripts/inspect_raw_bias_by_forecast_day.py \\
        --cache-dir forecast_outputs/heat_persistence_baseline_full \\
        --config forecasting/config.yaml \\
        --test-name "Peak window hours 14-18"
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


def _gate_mask(df: pd.DataFrame, test_name: str) -> pd.Series:
    hour = pd.to_numeric(df.get("Hour", pd.Series(np.nan, index=df.index)), errors="coerce")
    temp = pd.to_numeric(
        df.get("Temperature_DailyMax", pd.Series(np.nan, index=df.index)), errors="coerce"
    )
    if test_name == HOT_PEAK_TEST_NAME:
        return hour.between(16, 20) & temp.ge(90.0)
    if test_name == PEAK_WINDOW_TEST_NAME:
        return hour.between(14, 18)
    raise ValueError(f"Unsupported test_name: {test_name!r}")


def bias_by_origin_and_forecast_day(gate_rows: pd.DataFrame, raw_col: str) -> pd.DataFrame:
    """Pivoted table: rows=Replay_Origin_ID, columns=Forecast_Day, values=mean
    signed bias (Actual - raw_col) at that horizon depth for that origin."""
    actual = pd.to_numeric(gate_rows["Actual_MWH"], errors="coerce")
    pred = pd.to_numeric(gate_rows[raw_col], errors="coerce")
    err = actual - pred
    forecast_day = pd.to_numeric(gate_rows["Forecast_Day"], errors="coerce")
    grouped = pd.DataFrame(
        {
            "Replay_Origin_ID": gate_rows["Replay_Origin_ID"],
            "Forecast_Day": forecast_day,
            "Err": err,
        }
    ).dropna(subset=["Forecast_Day"])
    pivot = grouped.pivot_table(
        index="Replay_Origin_ID", columns="Forecast_Day", values="Err", aggfunc="mean"
    )
    pivot.columns = [f"Day{int(c)}" for c in pivot.columns]
    return pivot


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--cache-dir", required=True)
    parser.add_argument("--config", default=None, help="Path to config.yaml (default: FORECAST_CONFIG or config.yaml)")
    parser.add_argument(
        "--test-name", default=PEAK_WINDOW_TEST_NAME, choices=[HOT_PEAK_TEST_NAME, PEAK_WINDOW_TEST_NAME]
    )
    parser.add_argument(
        "--origins",
        nargs="+",
        default=None,
        help="Only print these Replay_Origin_ID values (e.g. origin_25 origin_26 origin_27). Default: all origins.",
    )
    args = parser.parse_args()

    config = load_forecast_config(args.config)
    bundles = load_raw_origin_bundles(args.cache_dir)
    if not bundles:
        raise SystemExit(f"No cached bundles found in {args.cache_dir}")
    print(f"Loaded {len(bundles)} bundles from {args.cache_dir}")

    replay = score_bundles(bundles, config)
    if replay.empty:
        raise SystemExit("score_bundles produced no rows.")

    gate_rows = replay.loc[_gate_mask(replay, args.test_name)]
    if gate_rows.empty:
        print(f"No rows matched the {args.test_name!r} mask.")
        return 0
    if "Forecast_Day" not in gate_rows.columns:
        raise SystemExit("Forecast_Day column not present in scored replay -- cannot break down by horizon depth.")
    if "Replay_Origin_ID" not in gate_rows.columns:
        raise SystemExit("Replay_Origin_ID column not present in scored replay -- cannot group by origin.")

    pivot = bias_by_origin_and_forecast_day(gate_rows, "Raw_Forecast_MWH")
    if args.origins:
        missing = [o for o in args.origins if o not in pivot.index]
        if missing:
            print(f"Warning: origins not found in this cache: {missing}")
        pivot = pivot.loc[[o for o in args.origins if o in pivot.index]]

    print(f"\n=== {args.test_name}: raw bias (Actual - Raw_Forecast_MWH) by origin x Forecast_Day ===")
    print(
        "(Rising left-to-right within a row = bias compounding deeper into that origin's "
        "horizon; flat = the model is already wrong on day 1 and the recursive forecast "
        "isn't amplifying it further)"
    )
    print(pivot.to_string(float_format=lambda x: f"{x:.2f}", na_rep="--"))

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
