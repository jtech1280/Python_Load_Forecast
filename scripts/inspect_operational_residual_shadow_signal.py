from __future__ import annotations

"""Checks whether operational_residual_learner's applied correction on Hot peak
days is small because it's capped, or because its own underlying residual
model isn't estimating a large correction in the first place.

Motivation: raising calibration.operational_residual_learner.capped_full_
shadow_cap_mwh from 1.0 to 6.0 (same cache, scored twice) moved Hot peak days
MAE by only -0.035 MWH -- real but far short of the ~2.3 MWH needed to clear
the 6.0 MWH threshold (raw bias on the worst origins runs 15-30 MWH). A 6x
looser cap barely engaging means one of two very different things:
  1. The cap is still binding (the uncapped "full shadow" signal is large, but
     even 6.0 MWH doesn't cover it) -- raising the cap further is worth trying.
  2. ORL's own residual model isn't predicting a large correction here at all
     (the uncapped signal itself is small) -- raising the cap wouldn't help
     regardless of how high it goes; the residual MODEL, not the cap, is the
     limiting factor.

Auto_Residual_Full_Shadow_Correction_MWH is the uncapped signal computed
before any capped_full_shadow/hot_peak_only/shadow_only scoping is applied
(operational_residual_learner.py, out["Auto_Residual_Full_Shadow_Correction_MWH"]
= full_corr) -- so it's exactly the number needed to tell these apart.

Usage:
    python scripts/inspect_operational_residual_shadow_signal.py \\
        --cache-dir forecast_outputs/heat_persistence_baseline_full \\
        --config forecasting/config.yaml \\
        --origins origin_25 origin_26 origin_27
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

SHADOW_COLS = [
    "Auto_Residual_Correction_MWH",
    "Auto_Residual_Full_Shadow_Correction_MWH",
    "Auto_Residual_Structural_HotPeak_Correction_MWH",
    "Auto_Residual_Broad_HotPeak_Shadow_Correction_MWH",
]


def _hot_peak_mask(df: pd.DataFrame) -> pd.Series:
    hour = pd.to_numeric(df.get("Hour", pd.Series(np.nan, index=df.index)), errors="coerce")
    temp = pd.to_numeric(
        df.get("Temperature_DailyMax", pd.Series(np.nan, index=df.index)), errors="coerce"
    )
    return hour.between(16, 20) & temp.ge(90.0)


def summarize_shadow_signal(gate_rows: pd.DataFrame) -> pd.DataFrame:
    actual = pd.to_numeric(gate_rows["Actual_MWH"], errors="coerce")
    raw = pd.to_numeric(gate_rows["Raw_Forecast_MWH"], errors="coerce")
    final = pd.to_numeric(gate_rows["Final_Backtest_Forecast_MWH"], errors="coerce")
    raw_err = actual - raw
    final_err = actual - final

    rows = []
    for origin_id, group in gate_rows.groupby("Replay_Origin_ID"):
        idx = group.index
        row = {
            "Replay_Origin_ID": origin_id,
            "N": len(group),
            "Raw_Bias_MWH": float(raw_err.loc[idx].mean()),
            "Final_Bias_MWH": float(final_err.loc[idx].mean()),
        }
        for col in SHADOW_COLS:
            if col in group.columns:
                values = pd.to_numeric(group[col], errors="coerce")
                row[f"{col}_mean"] = float(values.mean())
                row[f"{col}_max"] = float(values.max())
        rows.append(row)
    return pd.DataFrame(rows).sort_values("Raw_Bias_MWH", key=lambda s: s.abs(), ascending=False)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--cache-dir", required=True)
    parser.add_argument("--config", default=None, help="Path to config.yaml (default: FORECAST_CONFIG or config.yaml)")
    parser.add_argument("--origins", nargs="+", default=None)
    args = parser.parse_args()

    config = load_forecast_config(args.config)
    bundles = load_raw_origin_bundles(args.cache_dir)
    if not bundles:
        raise SystemExit(f"No cached bundles found in {args.cache_dir}")
    print(f"Loaded {len(bundles)} bundles from {args.cache_dir}")

    replay = score_bundles(bundles, config)
    if replay.empty:
        raise SystemExit("score_bundles produced no rows.")

    gate_rows = replay.loc[_hot_peak_mask(replay)]
    if gate_rows.empty:
        print("No rows matched the Hot peak days mask.")
        return 0
    if args.origins:
        gate_rows = gate_rows[gate_rows["Replay_Origin_ID"].isin(args.origins)]
        if gate_rows.empty:
            print(f"No Hot peak days rows found for origins: {args.origins}")
            return 0

    missing = [c for c in SHADOW_COLS if c not in gate_rows.columns]
    if missing:
        print(f"Warning: these Auto_Residual_* columns are not present in this cache/config: {missing}")

    summary = summarize_shadow_signal(gate_rows)
    print("\n=== Hot peak days: raw/final bias vs operational_residual_learner shadow signal (uncapped), by origin ===")
    print(
        "(*_mean/_max are in MWH. Full_Shadow_Correction is the uncapped signal before "
        "capped_full_shadow/hot_peak_only/shadow_only scoping is applied -- compare its "
        "magnitude to Raw_Bias_MWH to see whether the model is estimating anywhere near "
        "the correction actually needed, independent of any cap.)"
    )
    print(summary.to_string(index=False, float_format=lambda x: f"{x:.3f}"))

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
