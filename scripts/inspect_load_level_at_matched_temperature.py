from __future__ import annotations

"""Checks whether actual system load, at temperatures matching the 2026-07-27..29
event, has been trending upward year over year -- i.e. whether the raw model's
~14-16 MWH under-forecast on this event reflects genuine load growth (more
customers, AC, electrification) that a model trained mostly on 2020-2025
temperature-load relationships would systematically miss on the most recent
data, independent of weather.

Motivation: inspect_feature_extrapolation_2026_event.py ruled out the
structural "tree model can't extrapolate past training range" hypothesis --
95.5-98.1F with a 4-7 day ConsecutiveVeryHotDays95 streak is unremarkable
against this archive's history (117.5F max, streaks up to 18 days, seen as
recently as 2024). The event is not an out-of-domain temperature or
persistence pattern. So if the raw model still misses it this badly, the
next candidate is that LOAD at this temperature is higher now than the
temperature-load relationship the model learned from older training years --
which sample weighting could only partially compensate for (as the
recency_end_weight sweep confirmed) since it reweights rows without changing
what relationship the model can express.

Needs actual SQL Server load history (load_hourly_system_mwh) -- run this on
the same machine you build caches on, not in a sandbox without DB access.

Usage:
    python scripts/inspect_load_level_at_matched_temperature.py \\
        --config forecasting/config.yaml \\
        --event-start 2026-07-27 --event-end 2026-07-29 \\
        --temp-band-low 94 --temp-band-high 99
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
from forecasting.data.history_loader import load_hourly_system_mwh
from forecasting.data.weather_loader import fetch_historical_weather
from forecasting.features.record_breaking_heat import build_daily_max_temp_reference


def build_matched_scope(
    load_df: pd.DataFrame,
    daily_max: pd.DataFrame,
    hot_peak_hours: set[int],
    temp_band_low: float,
    temp_band_high: float,
) -> pd.DataFrame:
    """Hourly load rows restricted to hot-peak hours on days whose
    Temperature_DailyMax falls in [temp_band_low, temp_band_high]."""
    df = load_df.copy()
    df["Date"] = pd.to_datetime(df["DT"].dt.tz_localize(None).dt.date)
    df["Hour"] = df["DT"].dt.hour
    df["Year"] = df["DT"].dt.year

    merged = df.merge(daily_max[["Date", "Temperature_DailyMax"]], on="Date", how="inner")
    scope = merged["Hour"].isin(hot_peak_hours) & merged["Temperature_DailyMax"].between(
        temp_band_low, temp_band_high
    )
    return merged.loc[scope].reset_index(drop=True)


def yearly_summary(scoped: pd.DataFrame) -> pd.DataFrame:
    if scoped.empty:
        return pd.DataFrame(columns=["Year", "N_Hours", "Mean_MWH", "Median_MWH", "Max_MWH"])
    grouped = scoped.groupby("Year")["MWH"].agg(
        N_Hours="count", Mean_MWH="mean", Median_MWH="median", Max_MWH="max"
    )
    return grouped.reset_index().sort_values("Year")


def fit_trend_and_predict(summary: pd.DataFrame, target_year: int) -> tuple[float, float] | None:
    """Fits a linear year->Mean_MWH trend on years strictly before target_year
    and returns (predicted_mean_for_target_year, slope_mwh_per_year), or None
    if there isn't enough prior history to fit."""
    prior = summary[summary["Year"] < target_year]
    if len(prior) < 2:
        return None
    slope, intercept = np.polyfit(prior["Year"], prior["Mean_MWH"], 1)
    predicted = slope * target_year + intercept
    return float(predicted), float(slope)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--config", default=None, help="Path to config.yaml (default: FORECAST_CONFIG or config.yaml)")
    parser.add_argument("--event-start", default="2026-07-27")
    parser.add_argument("--event-end", default="2026-07-29")
    parser.add_argument("--temp-band-low", type=float, default=94.0)
    parser.add_argument("--temp-band-high", type=float, default=99.0)
    parser.add_argument("--hot-peak-hours", nargs="+", type=int, default=[16, 17, 18, 19, 20])
    args = parser.parse_args()

    config = load_forecast_config(args.config)
    print("Loading hourly system load from SQL Server...")
    load_df = load_hourly_system_mwh(config)
    print(f"Loaded {len(load_df)} hourly load rows.")

    print("Fetching historical weather (uses the same local cache the real pipeline does)...")
    weather_df = fetch_historical_weather(config)
    daily_max = build_daily_max_temp_reference(weather_df)
    if daily_max.empty:
        raise SystemExit("No daily max temperature rows could be built from the historical archive.")

    hot_peak_hours = set(args.hot_peak_hours)
    scoped = build_matched_scope(load_df, daily_max, hot_peak_hours, args.temp_band_low, args.temp_band_high)
    if scoped.empty:
        print(
            f"No hours matched hour in {sorted(hot_peak_hours)} and "
            f"Temperature_DailyMax in [{args.temp_band_low}, {args.temp_band_high}]."
        )
        return 0

    summary = yearly_summary(scoped)
    print(
        f"\n=== Actual load at hour {sorted(hot_peak_hours)} on days with "
        f"Temperature_DailyMax in [{args.temp_band_low}, {args.temp_band_high}]F, by year ==="
    )
    print(summary.to_string(index=False, float_format=lambda x: f"{x:.2f}"))

    event_start = pd.Timestamp(args.event_start)
    event_end = pd.Timestamp(args.event_end)
    event_scope = scoped[(scoped["Date"] >= event_start) & (scoped["Date"] <= event_end)]
    if not event_scope.empty:
        event_mean = float(event_scope["MWH"].mean())
        print(
            f"\n=== Event window ({args.event_start} to {args.event_end}) actual mean MWH "
            f"at these hours/temps: {event_mean:.2f} (N={len(event_scope)}) ==="
        )
        target_year = event_start.year
        fit = fit_trend_and_predict(summary, target_year)
        if fit is not None:
            predicted, slope = fit
            print(
                f"Linear year-over-year trend fit on years before {target_year}: "
                f"slope={slope:+.2f} MWH/year, predicted {target_year} mean at matched "
                f"temp/hours={predicted:.2f} MWH."
            )
            print(
                f"Event actual ({event_mean:.2f}) vs trend-predicted ({predicted:.2f}): "
                f"{event_mean - predicted:+.2f} MWH"
            )
            print(
                "(If this residual is small, the event is in line with an ongoing load-growth "
                "trend the model just hasn't been told about explicitly. If it's still large, "
                "something beyond a smooth year-over-year trend is going on.)"
            )
        else:
            print("Not enough prior years with matched-temperature hours to fit a trend.")
    else:
        print(
            f"\nNo hours in the event window ({args.event_start} to {args.event_end}) fell "
            f"inside the matched temperature band -- widen --temp-band-low/--temp-band-high."
        )

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
