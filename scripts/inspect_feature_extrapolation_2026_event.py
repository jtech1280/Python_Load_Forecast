from __future__ import annotations

"""Checks whether the 2026-07-27..29 event sets a new all-time high, in the
training history available to the model, on any of the heat-persistence
features the raw XGB/LGB/CatBoost ensemble actually trains on (Temperature_
DailyMax, DailyMaxTemp_3DayMean, ConsecutiveHotDays90/95, ConsecutiveExtreme
HotDays100).

Motivation: this session ruled out "the model has no persistence features"
(BASE_FEATURES already has them at three thresholds) and "this kind of
multi-day streak has never happened before" (the 2026 event is tied, not
uniquely extreme, in raw streak *length* -- see inspect_historical_heat_
streaks.py). It also found the max_weight sample-weight cap was neutralizing
recency weighting for extreme rows, and fixed it, but a controlled retrain
sweep showed the fix only closes a small fraction of the ~13-15 MWH raw bias
on this event -- not a monotonic, reliable lever.

The remaining, untested hypothesis is structural rather than a weighting
problem: XGB/LGB/CatBoost are tree ensembles, and a tree's leaf can only ever
predict a value within the range of training targets that reached that leaf --
it cannot extrapolate past the highest feature value (or feature combination)
it was ever trained on. If the 2026 event pushes DailyMaxTemp_3DayMean, or the
exact combination of Temperature_DailyMax with an already-long
ConsecutiveVeryHotDays95 run, past anything the model has seen, the under-
forecast would be a structural ceiling, not something sample weighting could
ever fix, however it's tuned.

Uses only the local historical-weather archive (build_daily_max_temp_reference
/ add_heat_persistence_features) -- no SQL/load data or GPU needed, so this is
fast and safe to run repeatedly.

Usage:
    python scripts/inspect_feature_extrapolation_2026_event.py \\
        --config forecasting/config.yaml \\
        --event-start 2026-07-27 --event-end 2026-07-29
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
from forecasting.features.weather_features import add_heat_persistence_features

FEATURE_COLS = [
    "Temperature_DailyMax",
    "DailyMaxTemp_3DayMean",
    "ConsecutiveHotDays90",
    "ConsecutiveVeryHotDays95",
    "ConsecutiveExtremeHotDays100",
]


def build_daily_features(weather_df: pd.DataFrame) -> pd.DataFrame:
    """One row per date with Temperature_DailyMax/Min/Mean and the
    heat-persistence features the production model trains on, built the same
    way add_heat_persistence_features expects (DT + the three daily temp
    columns), from an hourly TempF weather archive."""
    dt = pd.to_datetime(weather_df["DT"], errors="coerce")
    temp = pd.to_numeric(weather_df["TempF"], errors="coerce")
    daily = (
        pd.DataFrame({"Date": dt.dt.date, "TempF": temp})
        .dropna(subset=["Date"])
        .groupby("Date")["TempF"]
        .agg(
            Temperature_DailyMax="max",
            Temperature_DailyMin="min",
            Temperature_DailyMean="mean",
        )
        .reset_index()
    )
    daily["Date"] = pd.to_datetime(daily["Date"])
    daily["DT"] = daily["Date"]
    daily = daily.sort_values("Date").reset_index(drop=True)
    return add_heat_persistence_features(daily)


def find_new_records(
    daily: pd.DataFrame, event_start, event_end
) -> pd.DataFrame:
    """For each feature in FEATURE_COLS, compare the event window's values to
    the historical max seen strictly before the event started. Returns one row
    per (Date, feature) inside the event window."""
    before = daily[daily["Date"] < event_start]
    event = daily[(daily["Date"] >= event_start) & (daily["Date"] <= event_end)]

    rows = []
    for col in FEATURE_COLS:
        if col not in daily.columns:
            continue
        hist_max = float(before[col].max()) if not before.empty else float("nan")
        hist_p99 = float(before[col].quantile(0.99)) if not before.empty else float("nan")
        for _, day_row in event.iterrows():
            value = float(day_row[col])
            rows.append(
                {
                    "Date": day_row["Date"].date(),
                    "Feature": col,
                    "Event_Value": value,
                    "Historical_Max_Before_Event": hist_max,
                    "Historical_P99_Before_Event": hist_p99,
                    "New_Record": value > hist_max,
                }
            )
    return pd.DataFrame(rows)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--config", default=None, help="Path to config.yaml (default: FORECAST_CONFIG or config.yaml)")
    parser.add_argument("--event-start", default="2026-07-27")
    parser.add_argument("--event-end", default="2026-07-29")
    parser.add_argument("--top", type=int, default=5, help="Also print this many all-time hottest days for context")
    args = parser.parse_args()

    config = load_forecast_config(args.config)
    print("Fetching historical weather (uses the same local cache the real pipeline does)...")
    weather_df = fetch_historical_weather(config)
    if weather_df is None or weather_df.empty:
        raise SystemExit("fetch_historical_weather returned no rows.")

    daily = build_daily_features(weather_df)
    if daily.empty:
        raise SystemExit("No daily feature rows could be built from the historical archive.")
    print(
        f"Built {len(daily)} daily feature rows spanning {daily['Date'].min().date()} "
        f"to {daily['Date'].max().date()}."
    )

    event_start = pd.Timestamp(args.event_start)
    event_end = pd.Timestamp(args.event_end)
    records = find_new_records(daily, event_start, event_end)
    if records.empty:
        print("No rows found in the event window -- check --event-start/--event-end against the archive's date range.")
        return 0

    print(f"\n=== {args.event_start} to {args.event_end}: event feature values vs. pre-event history ===")
    print(
        "(New_Record=True means the event's value on that day exceeds anything the model "
        "would have seen in training up to that point -- a tree ensemble cannot extrapolate "
        "past that, regardless of sample weighting)"
    )
    print(records.to_string(index=False, float_format=lambda x: f"{x:.3f}"))

    print(f"\n=== Top {args.top} all-time hottest days (any year, for context) ===")
    hottest = daily.sort_values("Temperature_DailyMax", ascending=False).head(args.top)
    print(
        hottest[["Date", "Temperature_DailyMax", "DailyMaxTemp_3DayMean", "ConsecutiveVeryHotDays95"]]
        .assign(Date=lambda d: d["Date"].dt.date)
        .to_string(index=False, float_format=lambda x: f"{x:.3f}")
    )

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
