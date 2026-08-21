from __future__ import annotations

"""Breaks down the "Day 1 only" gate by hour, origin, and season between two
raw-forecast caches, to find WHERE an aggregate Day-1 MAE change is actually coming
from -- e.g. the extended-training-history cache (forecast_outputs/
extended_history_cache) vs the standard one (forecast_outputs/ablation_cache), which
showed Day 1 only flipping from failing (3.63 MAE) to passing (3.37 MAE) pooled, with
no visibility yet into whether that's a broad, even improvement or concentrated in a
few hours/origins/seasons -- the same question that mattered for every other pooled
number checked this session.

Both caches are scored against the SAME (real, current) config, so only the
underlying raw forecasts differ -- exactly like scripts/ablate_correction_stages.py's
--baseline-only, just sliced down to Forecast_Day == 1 with breakdowns instead of one
pooled row.

Usage:
    python scripts/compare_day1_accuracy.py \
        --cache-a forecast_outputs/ablation_cache --label-a standard \
        --cache-b forecast_outputs/extended_history_cache --label-b extended \
        --output-csv forecast_outputs/day1_comparison.csv
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


def _day1_slice(replay_df: pd.DataFrame) -> pd.DataFrame:
    day = pd.to_numeric(replay_df.get("Forecast_Day"), errors="coerce")
    return replay_df.loc[day.eq(1)].copy()


def _mae_bias(df: pd.DataFrame) -> tuple[float, float, int]:
    if df.empty or "Final_Backtest_Forecast_MWH" not in df.columns:
        return float("nan"), float("nan"), 0
    err = pd.to_numeric(df["Actual_MWH"], errors="coerce") - pd.to_numeric(
        df["Final_Backtest_Forecast_MWH"], errors="coerce"
    )
    return float(err.abs().mean()), float(err.mean()), int(err.notna().sum())


def _breakdown_by(df: pd.DataFrame, group_col: str) -> pd.DataFrame | None:
    """Returns None (not an empty DataFrame) when group_col isn't present, so the
    caller can distinguish "nothing to group by" from "grouped but empty" instead
    of crashing on set_index against a column that was never there."""
    if group_col not in df.columns:
        return None
    rows = []
    for key, group in df.groupby(group_col):
        mae, bias, n = _mae_bias(group)
        rows.append({group_col: key, "N": n, "MAE_MWH": mae, "Bias_MWH": bias})
    return pd.DataFrame(rows, columns=[group_col, "N", "MAE_MWH", "Bias_MWH"])


def compare(day1_a: pd.DataFrame, day1_b: pd.DataFrame, group_col: str, label_a: str, label_b: str) -> pd.DataFrame:
    a = _breakdown_by(day1_a, group_col)
    b = _breakdown_by(day1_b, group_col)
    if a is None or b is None:
        return pd.DataFrame()
    joined = a.set_index(group_col).join(
        b.set_index(group_col), lsuffix=f"_{label_a}", rsuffix=f"_{label_b}", how="outer"
    )
    mae_a_col = f"MAE_MWH_{label_a}"
    mae_b_col = f"MAE_MWH_{label_b}"
    joined["Delta_MAE_MWH"] = joined[mae_b_col] - joined[mae_a_col]
    return joined.reset_index().sort_values("Delta_MAE_MWH")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--config", type=str, default=None)
    parser.add_argument("--cache-a", type=Path, required=True)
    parser.add_argument("--label-a", type=str, default="a")
    parser.add_argument("--cache-b", type=Path, required=True)
    parser.add_argument("--label-b", type=str, default="b")
    parser.add_argument("--output-csv", type=Path, default=Path("forecast_outputs/day1_comparison.csv"))
    args = parser.parse_args()

    config = load_forecast_config(args.config)

    bundles_a = load_raw_origin_bundles(args.cache_a)
    bundles_b = load_raw_origin_bundles(args.cache_b)
    print(f"Loaded {len(bundles_a)} bundles from {args.cache_a} ({args.label_a})", flush=True)
    print(f"Loaded {len(bundles_b)} bundles from {args.cache_b} ({args.label_b})", flush=True)

    replay_a = score_bundles(bundles_a, config)
    replay_b = score_bundles(bundles_b, config)
    day1_a = _day1_slice(replay_a)
    day1_b = _day1_slice(replay_b)

    mae_a, bias_a, n_a = _mae_bias(day1_a)
    mae_b, bias_b, n_b = _mae_bias(day1_b)
    print(f"\n=== Overall Day 1 only ===")
    print(f"{args.label_a}: N={n_a} MAE={mae_a:.3f} Bias={bias_a:.3f}")
    print(f"{args.label_b}: N={n_b} MAE={mae_b:.3f} Bias={bias_b:.3f}")
    print(f"Delta MAE ({args.label_b} - {args.label_a}): {mae_b - mae_a:+.3f}")

    all_rows = []
    for group_col in ["Hour", "Replay_Origin_ID", "Season", "DailyMaxTempBin"]:
        breakdown = compare(day1_a, day1_b, group_col, args.label_a, args.label_b)
        if breakdown.empty:
            continue
        breakdown.insert(0, "Group_By", group_col)
        breakdown = breakdown.rename(columns={group_col: "Group_Value"})
        all_rows.append(breakdown)
        print(f"\n=== Day 1 only, by {group_col} (sorted by Delta MAE, most-improved first) ===")
        print(
            breakdown[["Group_Value", f"N_{args.label_a}", f"N_{args.label_b}", f"MAE_MWH_{args.label_a}", f"MAE_MWH_{args.label_b}", "Delta_MAE_MWH"]]
            .to_string(index=False, float_format=lambda x: f"{x:.3f}")
        )

    if all_rows:
        combined = pd.concat(all_rows, ignore_index=True, sort=False)
        args.output_csv.parent.mkdir(parents=True, exist_ok=True)
        combined.to_csv(args.output_csv, index=False)
        print(f"\nSaved full breakdown to {args.output_csv}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
