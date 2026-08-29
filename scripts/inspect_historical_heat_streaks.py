from __future__ import annotations

"""Lists the longest consecutive-day daily-max-temperature streaks in the full
historical training archive, at one or more thresholds, to test a specific
hypothesis: is the 2026-07-27..29 event (a 6-7 day plateau of 95-98F) genuinely
rare/unprecedented in the training window, or has the model seen comparable
multi-day heat events before and still misses this one?

Motivation: inspect_raw_vs_corrected_hot_day_bias.py showed all three base models
(XGB/LGB/CatBoost) independently under-forecasting Hot peak days by a similar large
margin, and traced almost all of that pooled bias to origins 25/26/27 (Jul 27-29)
specifically -- a raw-model bias of ~15-16 MWH that the entire correction chain,
capped by design, can only shave a fraction of a MWH off. The base model's
BASE_FEATURES already include multi-day persistence signal (PriorDay_DailyMaxTemp,
DailyMaxTemp_3DayMean, ConsecutiveHotDays90, ConsecutiveVeryHotDays95,
ConsecutiveExtremeHotDays100, HeatPersistenceStress90/95) at several thresholds, not
just the 100F-locked one the correction chain uses -- so this isn't a missing-feature
problem. If a 6-7 day streak at this level is one of very few (or the only) example
in ~6 years of training data, the model may simply lack enough comparable episodes to
have learned the true load impact, regardless of what features are available to it.

Usage:
    python scripts/inspect_historical_heat_streaks.py \\
        --config forecasting/config.yaml \\
        --thresholds 90 95 100 \\
        --min-length 4
"""

import argparse
import sys
from pathlib import Path

import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from forecasting.config_utils import load_forecast_config
from forecasting.data.weather_loader import fetch_historical_weather
from forecasting.features.record_breaking_heat import build_daily_max_temp_reference


def find_streaks(daily: pd.DataFrame, threshold: float) -> pd.DataFrame:
    """One row per consecutive-day run where Temperature_DailyMax >= threshold,
    sorted longest first. `daily` must have Date (datetime-like) and
    Temperature_DailyMax columns; gaps in the calendar (missing dates) break a
    streak, same as the project's own consecutive-day counters."""
    if daily is None or daily.empty:
        return pd.DataFrame(columns=["Start", "End", "Length_Days", "Max_Temp_F"])

    d = daily.sort_values("Date").reset_index(drop=True).copy()
    d["Date"] = pd.to_datetime(d["Date"])
    expected_next = d["Date"].shift(1) + pd.Timedelta(days=1)
    calendar_continuous = d["Date"].eq(expected_next).fillna(False)
    hot = d["Temperature_DailyMax"].ge(threshold)
    # A new streak starts whenever this day isn't hot, or the calendar isn't
    # contiguous with the previous row (a data gap must not silently bridge two
    # otherwise-unrelated hot spells into one streak).
    new_streak = (~hot) | (~calendar_continuous)
    streak_id = new_streak.cumsum()

    streaks = []
    for _, group in d[hot].groupby(streak_id[hot]):
        streaks.append(
            {
                "Start": group["Date"].min().date(),
                "End": group["Date"].max().date(),
                "Length_Days": int(len(group)),
                "Max_Temp_F": float(group["Temperature_DailyMax"].max()),
            }
        )
    out = pd.DataFrame(streaks)
    if out.empty:
        return out
    return out.sort_values("Length_Days", ascending=False).reset_index(drop=True)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--config", default=None, help="Path to config.yaml (default: FORECAST_CONFIG or config.yaml)")
    parser.add_argument("--thresholds", nargs="+", type=float, default=[90.0, 95.0, 100.0])
    parser.add_argument("--min-length", type=int, default=4, help="Only print streaks at least this many days long")
    parser.add_argument("--top", type=int, default=15, help="Max streaks to print per threshold")
    args = parser.parse_args()

    config = load_forecast_config(args.config)
    print("Fetching historical weather (uses the same local cache the real pipeline does)...")
    weather_df = fetch_historical_weather(config)
    if weather_df is None or weather_df.empty:
        raise SystemExit("fetch_historical_weather returned no rows.")

    daily = build_daily_max_temp_reference(weather_df)
    if daily.empty:
        raise SystemExit("No daily-max-temperature rows could be built from the historical archive.")
    print(
        f"Built {len(daily)} daily rows spanning {daily['Date'].min().date()} to "
        f"{daily['Date'].max().date()}."
    )

    for threshold in sorted(args.thresholds):
        streaks = find_streaks(daily, threshold)
        qualifying = streaks[streaks["Length_Days"] >= args.min_length] if not streaks.empty else streaks
        print(f"\n=== Streaks >= {threshold:.0f}F, {args.min_length}+ days ({len(qualifying)} found) ===")
        if qualifying.empty:
            print("  none")
            continue
        print(qualifying.head(args.top).to_string(index=False))

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
