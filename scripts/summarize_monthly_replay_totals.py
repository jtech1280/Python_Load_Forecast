from __future__ import annotations

"""Sums forecasted vs. actual MWH over a calendar month from a rolling-origin
replay's results CSV, to answer "how much energy did we forecast vs. actually
use" as a real monthly total rather than an hourly mean/bias.

Replay origins have overlapping horizons -- a given calendar hour can appear in
several origins' rows (once per origin whose horizon reaches it), each with its
own forecast for that hour at a different lead time. Naively summing every row
in the month would double- or triple-count those hours. This script reports
both:
  - A deduplicated total: one row per calendar hour, keeping the forecast with
    the smallest Forecast_Day (closest to real-time) -- a literal, non-inflated
    monthly total comparable to what was actually "the" forecast for that hour.
  - The naive all-rows total, for transparency, clearly labeled as summing
    every origin's view of the month (useful only as an internally consistent
    forecast-vs-actual comparison over the same row population, not a literal
    once-per-hour total).

Usage:
    python scripts/summarize_monthly_replay_totals.py \\
        --results-path forecast_outputs/rolling_origin_replay_results.csv \\
        --year 2026 --month 8
"""

import argparse
from pathlib import Path

import pandas as pd


def deduplicate_to_nearest_horizon(df: pd.DataFrame) -> pd.DataFrame:
    """One row per DT, keeping the smallest Forecast_Day (closest-to-real-time
    forecast) when a calendar hour is covered by more than one origin."""
    if df.empty:
        return df
    ordered = df.sort_values(["DT", "Forecast_Day"], kind="stable")
    return ordered.drop_duplicates(subset="DT", keep="first").reset_index(drop=True)


def monthly_totals(df: pd.DataFrame, forecast_col: str) -> dict:
    actual = pd.to_numeric(df["Actual_MWH"], errors="coerce")
    forecast = pd.to_numeric(df[forecast_col], errors="coerce")
    valid = actual.notna() & forecast.notna()
    actual_sum = float(actual[valid].sum())
    forecast_sum = float(forecast[valid].sum())
    return {
        "N_Hours": int(valid.sum()),
        "Actual_Sum_MWH": actual_sum,
        "Forecast_Sum_MWH": forecast_sum,
        "Diff_MWH": actual_sum - forecast_sum,
        "Diff_PCT": (
            (actual_sum - forecast_sum) / actual_sum * 100.0 if actual_sum else float("nan")
        ),
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--results-path", default="forecast_outputs/rolling_origin_replay_results.csv")
    parser.add_argument("--year", type=int, default=2026)
    parser.add_argument("--month", type=int, default=8)
    parser.add_argument(
        "--forecast-col",
        default="Final_Backtest_Forecast_MWH",
        help="Which forecast column to sum (default: Final_Backtest_Forecast_MWH, the fully-corrected replay forecast).",
    )
    args = parser.parse_args()

    path = Path(args.results_path)
    if not path.exists():
        raise SystemExit(f"Results file not found: {path}")

    df = pd.read_csv(path, low_memory=False)
    if "DT" not in df.columns:
        raise SystemExit(f"{path} has no DT column -- is this a rolling-origin replay results CSV?")
    # Parse as-is (don't force a UTC conversion first) so a tz-aware local timestamp like
    # "2026-08-05 10:00:00-07:00" keeps its local wall-clock date/hour when the tz label is
    # dropped, instead of shifting by the UTC offset and potentially crossing a day/month
    # boundary.
    parsed = pd.to_datetime(df["DT"], errors="coerce")
    if getattr(parsed.dt, "tz", None) is not None:
        parsed = parsed.dt.tz_localize(None)
    df["DT"] = parsed

    n_unparsed = int(df["DT"].isna().sum())
    if n_unparsed:
        print(f"Warning: {n_unparsed} of {len(df)} DT values failed to parse and were dropped.")
    if df["DT"].notna().any():
        print(f"DT range in file: {df['DT'].min()} to {df['DT'].max()}")

    month_df = df[(df["DT"].dt.year == args.year) & (df["DT"].dt.month == args.month)]
    if month_df.empty:
        raise SystemExit(
            f"No rows found for {args.year}-{args.month:02d} in {path}. "
            "Check the replay's fixed_origins/horizon actually covers this month "
            "(see the DT range printed above)."
        )
    if args.forecast_col not in month_df.columns:
        raise SystemExit(f"Column {args.forecast_col!r} not found. Available: {sorted(month_df.columns)}")

    print(f"Loaded {len(month_df)} rows for {args.year}-{args.month:02d} from {path}")
    if "Replay_Origin_ID" in month_df.columns:
        origins = sorted(month_df["Replay_Origin_ID"].unique())
        print(f"Origins whose horizon reaches this month: {origins}")

    dedup = deduplicate_to_nearest_horizon(month_df)
    dedup_totals = monthly_totals(dedup, args.forecast_col)
    print(
        f"\n=== Deduplicated (one row per hour, nearest-horizon forecast) "
        f"totals for {args.year}-{args.month:02d} ==="
    )
    for key, value in dedup_totals.items():
        print(f"  {key}: {value:,.2f}" if isinstance(value, float) else f"  {key}: {value}")

    all_rows_totals = monthly_totals(month_df, args.forecast_col)
    print(
        f"\n=== All-rows (every origin's view, hours covered by multiple origins "
        f"counted once per origin) totals for {args.year}-{args.month:02d} ==="
    )
    for key, value in all_rows_totals.items():
        print(f"  {key}: {value:,.2f}" if isinstance(value, float) else f"  {key}: {value}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
