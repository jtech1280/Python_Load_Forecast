from __future__ import annotations

"""Describes the cloud-cover/solar-loss distribution behind the "Cloud/solar midday" gate
(build_production_readiness_scorecard's hour.between(10,16) & (CloudCover_Norm>=0.60 |
BTM_Solar_Loss_From_ClearSky_MW>=1.25) slice), to answer a question a bare row count can't:
is the near-zero ablation delta for cloud_solar_shape_correction (see
scripts/ablate_correction_stages.py) because the stage genuinely doesn't matter, or because
this particular replay window didn't have enough distinctly cloudy midday events to exercise
it?

A row count alone can't distinguish "1795 rows, robustly cloudy, spread across many origins"
from "1795 rows, mostly just barely over the 0.60/1.25 threshold, or concentrated in 2-3
origins" -- the same pooled-vs-per-origin distinction that already mattered for the hot-peak
correction prototypes and the ablation harness itself.

Reuses the same raw-forecast cache as ablate_correction_stages.py (built via
scripts/ablate_correction_stages.py --build-cache) -- no retraining, just scores the cached
bundles once against the current config to get real CloudCover_Norm/solar-loss values.

Usage:
    python scripts/describe_cloud_solar_midday_coverage.py --cache-dir forecast_outputs/ablation_cache
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

MIN_CLOUD = 0.60
MIN_LOSS = 1.25


def describe(replay_df: pd.DataFrame) -> None:
    hour = pd.to_numeric(replay_df.get("Hour"), errors="coerce")
    cloud = pd.to_numeric(replay_df.get("CloudCover_Norm"), errors="coerce")
    loss = pd.to_numeric(replay_df.get("BTM_Solar_Loss_From_ClearSky_MW"), errors="coerce")

    candidate_window = hour.between(10, 16)
    n_candidate = int(candidate_window.sum())
    print(f"Rows in the hour-10-16 candidate window: {n_candidate}")
    if n_candidate == 0:
        print("No rows in the candidate window at all -- nothing to describe.")
        return

    cloud_missing = candidate_window & cloud.isna()
    loss_missing = candidate_window & loss.isna()
    print(
        f"  CloudCover_Norm missing/NaN in candidate window: {int(cloud_missing.sum())} "
        f"({100 * cloud_missing.sum() / n_candidate:.1f}%)"
    )
    print(
        f"  BTM_Solar_Loss_From_ClearSky_MW missing/NaN in candidate window: "
        f"{int(loss_missing.sum())} ({100 * loss_missing.sum() / n_candidate:.1f}%)"
    )

    qualifying = candidate_window & (cloud.ge(MIN_CLOUD) | loss.ge(MIN_LOSS))
    n_qualifying = int(qualifying.sum())
    print(
        f"\nRows qualifying for the gate (cloud>={MIN_CLOUD} or loss>={MIN_LOSS}): "
        f"{n_qualifying} ({100 * n_qualifying / n_candidate:.1f}% of the candidate window)"
    )
    if n_qualifying == 0:
        print("No qualifying rows -- the gate's N in the scorecard should be 0. Check for a mismatch.")
        return

    cloud_q = cloud[qualifying]
    loss_q = loss[qualifying]
    print("\nCloudCover_Norm among qualifying rows:")
    print(
        cloud_q.describe(percentiles=[0.1, 0.25, 0.5, 0.75, 0.9]).to_string()
    )
    print("\nBTM_Solar_Loss_From_ClearSky_MW among qualifying rows:")
    print(loss_q.describe(percentiles=[0.1, 0.25, 0.5, 0.75, 0.9]).to_string())

    # How many rows are only barely over the threshold vs. robustly over it -- a stage that's
    # mostly seeing marginal cases would plausibly have a smaller measurable effect than one
    # seeing deeply cloudy/high-loss conditions, independent of whether it "works."
    barely_cloud = qualifying & cloud.between(MIN_CLOUD, MIN_CLOUD + 0.10)
    barely_loss = qualifying & loss.between(MIN_LOSS, MIN_LOSS + 0.5)
    only_barely = qualifying & (
        (cloud.ge(MIN_CLOUD) & cloud.lt(MIN_CLOUD + 0.10) & ~loss.ge(MIN_LOSS))
        | (loss.ge(MIN_LOSS) & loss.lt(MIN_LOSS + 0.5) & ~cloud.ge(MIN_CLOUD))
    )
    print(
        f"\nQualifying rows within 0.10 of the cloud threshold: {int(barely_cloud.sum())} "
        f"({100 * barely_cloud.sum() / n_qualifying:.1f}%)"
    )
    print(
        f"Qualifying rows within 0.5 MW of the loss threshold: {int(barely_loss.sum())} "
        f"({100 * barely_loss.sum() / n_qualifying:.1f}%)"
    )
    print(
        f"Qualifying rows that ONLY just barely clear one threshold (not robustly over "
        f"either): {int(only_barely.sum())} ({100 * only_barely.sum() / n_qualifying:.1f}%)"
    )

    if "Replay_Origin_ID" in replay_df.columns:
        by_origin = replay_df.loc[qualifying, "Replay_Origin_ID"].value_counts()
        n_origins_total = replay_df["Replay_Origin_ID"].nunique()
        print(
            f"\nQualifying rows are spread across {len(by_origin)} of {n_origins_total} total "
            f"origins."
        )
        print("Rows per origin (top 10):")
        print(by_origin.head(10).to_string())
        top3_share = by_origin.sort_values(ascending=False).head(3).sum() / n_qualifying
        print(
            f"\nTop 3 origins account for {100 * top3_share:.1f}% of all qualifying rows -- "
            "if this is high, the near-zero ablation delta could be concentrated-event noise "
            "rather than a genuine null result across the season."
        )


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--config", type=str, default=None)
    parser.add_argument("--cache-dir", type=Path, default=Path("forecast_outputs/ablation_cache"))
    args = parser.parse_args()

    if not args.cache_dir.exists():
        raise SystemExit(
            f"No cache found at {args.cache_dir}. Build it first with "
            "scripts/ablate_correction_stages.py --build-cache."
        )

    config = load_forecast_config(args.config)
    bundles = load_raw_origin_bundles(args.cache_dir)
    print(f"Loaded {len(bundles)} cached raw origin bundles from {args.cache_dir}\n", flush=True)

    replay_df = score_bundles(bundles, config)
    if replay_df.empty:
        raise SystemExit("score_bundles returned no rows -- nothing to describe.")

    describe(replay_df)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
