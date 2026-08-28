from __future__ import annotations

"""Compares the RAW ensemble forecast (Raw_Forecast_MWH, before any correction-chain
stage runs) against the FINAL corrected forecast (Final_Backtest_Forecast_MWH) on
Hot peak days / Peak window rows, plus each individual benchmark model
(XGB/LGB/CatBoost), to answer a specific question: is Hot peak days' large bias
(more than half its MAE) already present in the raw model before any correction, or
is it something the correction chain introduces/fails to remove?

Motivation: this session tested four correction-chain candidates (record_breaking_
heat, heat_persistence_peak_capture's moderate tier, daily_peak_shadow_model
promotion, peak_risk_correction) against Hot peak days and none meaningfully closed
the gap to its 6.0 MWH MAE threshold (current ~8.3-8.4). Before trying another
correction-chain patch, this checks whether the ceiling is structural -- baked into
the raw XGB/LGB/CatBoost ensemble's training -- rather than something a post-hoc
correction stage could plausibly fix at all.

Uses the SAME cache's bundles scored once (no separate raw-only load needed --
apply_origin_correction_chain adds new columns on top of the raw ones rather than
overwriting them, so Raw_Forecast_MWH and Final_Backtest_Forecast_MWH are both
present, perfectly row-aligned, in a single score_bundles() call).

Usage:
    python scripts/inspect_raw_vs_corrected_hot_day_bias.py \\
        --cache-dir forecast_outputs/heat_persistence_baseline_full \\
        --config forecasting/config.yaml
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

FORECAST_COLS = [
    ("Raw_Forecast_MWH", "Raw ensemble (pre-correction)"),
    ("XGB_Pred_MWH", "XGBoost only"),
    ("LGB_Pred_MWH", "LightGBM only"),
    ("CatBoost_Pred_MWH", "CatBoost only"),
    ("Final_Backtest_Forecast_MWH", "Final (post-correction)"),
]


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


def _pooled_metrics(df: pd.DataFrame, col: str) -> tuple[float, float, int] | None:
    if col not in df.columns or "Actual_MWH" not in df.columns:
        return None
    actual = pd.to_numeric(df["Actual_MWH"], errors="coerce")
    pred = pd.to_numeric(df[col], errors="coerce")
    err = actual - pred
    valid = err.notna()
    n = int(valid.sum())
    if n == 0:
        return None
    return float(err[valid].abs().mean()), float(err[valid].mean()), n


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--cache-dir", required=True)
    parser.add_argument("--config", default=None, help="Path to config.yaml (default: FORECAST_CONFIG or config.yaml)")
    parser.add_argument(
        "--test-name", default=HOT_PEAK_TEST_NAME, choices=[HOT_PEAK_TEST_NAME, PEAK_WINDOW_TEST_NAME]
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

    print(f"\n=== {args.test_name}: raw model vs final corrected forecast ({len(gate_rows)} rows) ===")
    pooled_rows = []
    for col, label in FORECAST_COLS:
        metrics = _pooled_metrics(gate_rows, col)
        if metrics is None:
            continue
        mae, bias, n = metrics
        pooled_rows.append({"Column": col, "Label": label, "N": n, "MAE_MWH": mae, "Bias_MWH": bias})
    pooled_df = pd.DataFrame(pooled_rows)
    print(pooled_df.to_string(index=False, float_format=lambda x: f"{x:.3f}"))

    if "Replay_Origin_ID" in gate_rows.columns:
        raw_col = "Raw_Forecast_MWH"
        final_col = "Final_Backtest_Forecast_MWH"
        if raw_col in gate_rows.columns and final_col in gate_rows.columns:
            actual = pd.to_numeric(gate_rows["Actual_MWH"], errors="coerce")
            raw_err = actual - pd.to_numeric(gate_rows[raw_col], errors="coerce")
            final_err = actual - pd.to_numeric(gate_rows[final_col], errors="coerce")
            per_origin = pd.DataFrame(
                {
                    "Raw_Bias_MWH": raw_err.groupby(gate_rows["Replay_Origin_ID"]).mean(),
                    "Final_Bias_MWH": final_err.groupby(gate_rows["Replay_Origin_ID"]).mean(),
                }
            )
            per_origin["Correction_Chain_Delta_MWH"] = (
                per_origin["Final_Bias_MWH"].abs() - per_origin["Raw_Bias_MWH"].abs()
            )
            per_origin = per_origin.sort_values("Raw_Bias_MWH", key=lambda s: s.abs(), ascending=False)
            print(f"\n=== Per-origin: raw bias vs final bias ({args.test_name}) ===")
            print(
                "(Correction_Chain_Delta_MWH = |Final_Bias| - |Raw_Bias|; negative means "
                "the correction chain shrank the raw model's bias, positive means it grew it)"
            )
            print(per_origin.to_string(float_format=lambda x: f"{x:.3f}"))

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
