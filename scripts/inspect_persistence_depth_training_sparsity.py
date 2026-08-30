from __future__ import annotations

"""Checks whether the raw model's growing bias deeper into a sustained >=95F
streak (see inspect_raw_bias_by_forecast_day.py) could be a training-data
sparsity problem rather than a purely learned-response-weakness one: even
though the 2026 event's streak depth is in-range (max historical
ConsecutiveVeryHotDays95 was 18, same as 2026 -- see
inspect_feature_extrapolation_2026_event.py), "in range" isn't the same as
"well represented." If only one or two independent historical streaks ever
reached day 13+ of a hot spell, the model's late-streak response is learned
from a handful of correlated rows (all from the same one or two events), not
from many independent examples -- a much weaker basis than the raw row count
alone would suggest.

Reports two views of the same thing:
  1. Row-count histogram: how many total historical days fall in each
     ConsecutiveVeryHotDays95 depth bucket (this is what a naive row-count
     check would show, but rows within one streak aren't independent
     evidence).
  2. Independent-episode count: how many separate >=95F streaks (from
     inspect_historical_heat_streaks.find_streaks) ever reached each bucket's
     depth at all -- the more informative number for "how many genuinely
     different events has the model seen at this persistence depth."

Uses only the local historical-weather archive -- no SQL/load data needed.

Usage:
    python scripts/inspect_persistence_depth_training_sparsity.py \\
        --config forecasting/config.yaml
"""

import argparse
import importlib
import sys
from pathlib import Path

import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))
SCRIPTS_DIR = Path(__file__).resolve().parent
if str(SCRIPTS_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPTS_DIR))

from forecasting.config_utils import load_forecast_config
from forecasting.data.weather_loader import fetch_historical_weather

inspect_extrapolation = importlib.import_module("inspect_feature_extrapolation_2026_event")
inspect_streaks = importlib.import_module("inspect_historical_heat_streaks")

DEPTH_BUCKETS = [(1, 3), (4, 7), (8, 12), (13, 18), (19, 999)]


def bucket_label(lo: int, hi: int) -> str:
    return f"{lo}-{hi}" if hi < 999 else f"{lo}+"


def row_count_histogram(daily: pd.DataFrame, feature: str = "ConsecutiveVeryHotDays95") -> pd.DataFrame:
    depth = daily[feature].fillna(0.0)
    rows = [{"Bucket": "0 (not hot)", "N_Days": int((depth == 0).sum())}]
    for lo, hi in DEPTH_BUCKETS:
        n = int(depth.between(lo, hi).sum())
        rows.append({"Bucket": bucket_label(lo, hi), "N_Days": n})
    return pd.DataFrame(rows)


def independent_episode_histogram(streaks: pd.DataFrame) -> pd.DataFrame:
    if streaks.empty:
        return pd.DataFrame(columns=["Bucket", "N_Independent_Streaks_Reaching_Depth"])
    rows = []
    for lo, hi in DEPTH_BUCKETS:
        n = int((streaks["Length_Days"] >= lo).sum())
        rows.append({"Bucket": f"{lo}+", "N_Independent_Streaks_Reaching_Depth": n})
    return pd.DataFrame(rows)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--config", default=None, help="Path to config.yaml (default: FORECAST_CONFIG or config.yaml)")
    parser.add_argument("--threshold", type=float, default=95.0)
    args = parser.parse_args()

    config = load_forecast_config(args.config)
    print("Fetching historical weather (uses the same local cache the real pipeline does)...")
    weather_df = fetch_historical_weather(config)
    if weather_df is None or weather_df.empty:
        raise SystemExit("fetch_historical_weather returned no rows.")

    daily = inspect_extrapolation.build_daily_features(weather_df)
    if daily.empty:
        raise SystemExit("No daily feature rows could be built from the historical archive.")
    print(
        f"Built {len(daily)} daily feature rows spanning {daily['Date'].min().date()} "
        f"to {daily['Date'].max().date()}."
    )

    print(f"\n=== Row-count histogram: days by ConsecutiveVeryHotDays{int(args.threshold)} depth ===")
    print(row_count_histogram(daily).to_string(index=False))

    streaks = inspect_streaks.find_streaks(daily, args.threshold)
    print(
        f"\n=== Independent >={args.threshold:.0f}F streaks ever reaching each depth "
        f"({len(streaks)} streaks total in the archive) ==="
    )
    episode_hist = independent_episode_histogram(streaks)
    print(episode_hist.to_string(index=False))

    deep = streaks[streaks["Length_Days"] >= 13].sort_values("Start")
    print("\n=== Streaks reaching 13+ days (the depth the 2026 event's late days sit at) ===")
    if deep.empty:
        print("  none")
    else:
        print(deep.to_string(index=False))

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
