from __future__ import annotations

"""Breaks the "Shoulder heat transition" gate down by origin and cross-references each
contributing origin against whether heat_persistence_peak_capture (including its
moderate, sub-100F tier) fired anywhere else in that origin's 16-day horizon.

Motivation: a real full rolling-origin-replay run showed this gate regressing (MAE
4.529->4.803, bias magnitude 0.912->1.106) after enabling heat_persistence_peak_
capture's moderate tier, even though a same-day, confound-controlled ablation-cache A/B
predicted it would stay flat. The gate's own row mask (season Spring/Fall, temp 75-93F)
can never itself qualify for the persistence tier (which needs temp>=95F), so any real
causal link would have to be indirect -- e.g. a correction on a hot day earlier in an
origin's horizon shifting a same-origin, state-carrying stage (recent-residual/AR level
correction) that then touches cooler days later in that same horizon. This checks
whether that's plausible: are the origins driving the Shoulder-transition MAE the same
ones where the persistence correction fired elsewhere, or is the regression spread
across origins the correction never touched at all (pointing to ordinary retraining
noise instead)?

Reuses forecast_diagnostics.py's exact Shoulder heat transition mask -- season Spring/
Fall, hour 12-22, Temperature_DailyMax 75-93F -- rather than re-deriving it, so this
never silently drifts from the gate's real definition.

Usage:
    python scripts/inspect_shoulder_transition_regression.py \\
        --csv forecast_outputs/backtest_results.csv
"""

import argparse
import sys
from pathlib import Path

import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

FORECAST_COL = "Final_Backtest_Forecast_MWH"


def _shoulder_transition_mask(df: pd.DataFrame) -> pd.Series:
    season = df.get("Season", pd.Series("", index=df.index)).astype(str)
    hour = pd.to_numeric(df.get("Hour"), errors="coerce")
    temp = pd.to_numeric(df.get("Temperature_DailyMax"), errors="coerce")
    return season.isin(["Spring", "Fall"]) & hour.between(12, 22) & temp.between(75.0, 93.0)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--csv", default="forecast_outputs/backtest_results.csv")
    args = parser.parse_args()

    path = Path(args.csv)
    if not path.exists():
        raise SystemExit(f"No CSV found at {path}")
    df = pd.read_csv(path)
    for required in ("Replay_Origin_ID", "Actual_MWH", FORECAST_COL):
        if required not in df.columns:
            raise SystemExit(f"{path} is missing required column {required!r}")

    mask = _shoulder_transition_mask(df)
    gate_rows = df.loc[mask].copy()
    if gate_rows.empty:
        print("No rows matched the Shoulder heat transition mask in this CSV.")
        return 0

    err = pd.to_numeric(gate_rows["Actual_MWH"], errors="coerce") - pd.to_numeric(
        gate_rows[FORECAST_COL], errors="coerce"
    )
    gate_rows["_AbsError"] = err.abs()
    gate_rows["_SignedError"] = err

    per_origin = gate_rows.groupby("Replay_Origin_ID").agg(
        N=("_AbsError", "size"),
        MAE_MWH=("_AbsError", "mean"),
        Bias_MWH=("_SignedError", "mean"),
    )

    has_persistence_col = "Heat_Persistence_Peak_Correction_MWH" in df.columns
    has_scope_col = "Heat_Persistence_Peak_Scope_Flag" in df.columns
    has_strong_col = "Heat_Persistence_Peak_Strong_Flag" in df.columns
    if has_persistence_col:
        correction = pd.to_numeric(
            df["Heat_Persistence_Peak_Correction_MWH"], errors="coerce"
        ).fillna(0.0)
        applied = correction.ne(0.0)
        strong = (
            pd.to_numeric(df["Heat_Persistence_Peak_Strong_Flag"], errors="coerce").fillna(0).eq(1)
            if has_strong_col
            else pd.Series(False, index=df.index)
        )
        moderate_applied = applied & ~strong
        strong_applied = applied & strong
        touched_moderate = moderate_applied.groupby(df["Replay_Origin_ID"]).any()
        touched_strong = strong_applied.groupby(df["Replay_Origin_ID"]).any()
        per_origin["Moderate_Tier_Fired_Elsewhere_In_Horizon"] = per_origin.index.map(
            touched_moderate
        ).fillna(False)
        per_origin["Strong_Tier_Fired_Elsewhere_In_Horizon"] = per_origin.index.map(
            touched_strong
        ).fillna(False)
    else:
        print(
            "NOTE: Heat_Persistence_Peak_Correction_MWH not present in this CSV -- "
            "can't check whether the persistence stage fired elsewhere in each origin's "
            "horizon. Re-run against the full backtest_results.csv from a real "
            "rolling-origin-replay run (not the ablation-cache shortcut) to get this."
        )

    per_origin = per_origin.sort_values("MAE_MWH", ascending=False)
    print(f"=== Shoulder heat transition, per-origin breakdown ({path}) ===")
    print(per_origin.to_string(float_format=lambda x: f"{x:.3f}"))

    if has_persistence_col:
        n_touched = int(
            (
                per_origin["Moderate_Tier_Fired_Elsewhere_In_Horizon"]
                | per_origin["Strong_Tier_Fired_Elsewhere_In_Horizon"]
            ).sum()
        )
        print(
            f"\n{n_touched}/{len(per_origin)} origins contributing to this gate also had "
            "heat_persistence_peak_capture fire (moderate or strong tier) somewhere else "
            "in their horizon. If that's most/all of them, an indirect same-origin "
            "carryover (e.g. a state-carrying stage like recent-residual/AR correction) "
            "is plausible. If it's few or none, this regression is more likely ordinary "
            "retraining noise unrelated to the persistence tier."
        )

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
