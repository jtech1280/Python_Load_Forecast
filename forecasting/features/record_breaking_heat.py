from __future__ import annotations

""""Record-breaking-ness" heat feature: how far today's daily-max temperature sits
above (or below) the typical top-of-distribution temperature for this time of year,
based on a genuinely large historical reference -- not a repeat of the version
prototyped earlier this project, which computed its reference percentile from only
the ~27-origin rolling-origin replay sample and came out zero for most origins
because that sample is far too sparse to estimate a 90th/95th percentile from.

This version builds the reference from the full historical weather archive already
being fetched for training (`hist_wx` in forecast_pipeline.py) -- thousands of daily
observations, not 27 -- and, critically, only ever looks at YEARS STRICTLY BEFORE the
row being featurized: a training row from summer 2021 is compared against summer
2020's temperatures only, never summer 2022's, so a hot day doesn't get quietly
un-extreme-ified by hotter days that hadn't happened yet at that point in time. The
same leakage-safety discipline used throughout this project's origin-available
correction chain applies here too.

Two failure modes this is designed to fail loudly on rather than silently degrade:
  - Too few prior years for a given date (e.g. the first calendar year of the
    training window has zero prior years to compare against) -> NaN, not zero. A
    silent zero is exactly what made the first attempt look degenerate without
    anyone noticing until the per-origin breakdown was checked by hand.
  - A percentile computed from a suspiciously thin sample (few distinct prior years,
    even if the raw row count looks large because of the day-of-year window) -> also
    NaN, gated by min_reference_years, and the actual count is exposed as its own
    column (Temp_Climatology_Reference_Years) so this is auditable rather than
    another number nobody checks.
"""

import numpy as np
import pandas as pd

DEFAULT_PERCENTILE = 95.0
DEFAULT_DAY_OF_YEAR_WINDOW_DAYS = 10
DEFAULT_MIN_REFERENCE_YEARS = 3


def _cfg(config: dict | None) -> dict:
    raw = config or {}
    return (raw.get("features", {}) or {}).get("record_breaking_heat", {}) or {}


def build_daily_max_temp_reference(weather_df: pd.DataFrame) -> pd.DataFrame:
    """Collapse an hourly weather frame (DT, TempF) to one row per date with that
    date's max temperature. This is the raw material the climatology lookup is
    built from -- deliberately built from the full historical archive, not from
    any single origin's short calibration window."""
    if weather_df is None or weather_df.empty or "TempF" not in weather_df.columns:
        return pd.DataFrame(columns=["Date", "Year", "DOY", "Temperature_DailyMax"])
    dt = pd.to_datetime(weather_df["DT"], errors="coerce")
    temp = pd.to_numeric(weather_df["TempF"], errors="coerce")
    daily = (
        pd.DataFrame({"Date": dt.dt.date, "TempF": temp})
        .dropna(subset=["Date"])
        .groupby("Date", as_index=False)["TempF"]
        .max()
        .rename(columns={"TempF": "Temperature_DailyMax"})
    )
    daily["Date"] = pd.to_datetime(daily["Date"])
    daily["Year"] = daily["Date"].dt.year
    daily["DOY"] = daily["Date"].dt.dayofyear
    return daily.dropna(subset=["Temperature_DailyMax"]).reset_index(drop=True)


def _circular_doy_distance(a: np.ndarray, b: int, year_length: int = 366) -> np.ndarray:
    diff = np.abs(a - b)
    return np.minimum(diff, year_length - diff)


def _trailing_percentile_for_date(
    reference: pd.DataFrame,
    target_year: int,
    target_doy: int,
    *,
    percentile: float,
    window_days: int,
    min_reference_years: int,
) -> tuple[float, int]:
    """percentile of Temperature_DailyMax among rows from years strictly before
    target_year, within window_days of target_doy (circularly, so a window
    around Dec 30 correctly pulls in early-January rows from the prior year).
    Returns (nan, 0) if fewer than min_reference_years distinct years qualify."""
    prior = reference[reference["Year"] < target_year]
    if prior.empty:
        return float("nan"), 0
    distance = _circular_doy_distance(prior["DOY"].to_numpy(), target_doy)
    within_window = prior.loc[distance <= window_days]
    n_years = int(within_window["Year"].nunique())
    if n_years < min_reference_years:
        return float("nan"), n_years
    value = float(np.percentile(within_window["Temperature_DailyMax"].to_numpy(), percentile))
    return value, n_years


def compute_climatology_lookup(
    reference: pd.DataFrame,
    dates: pd.Series,
    config: dict | None = None,
) -> pd.DataFrame:
    """One row per distinct date in `dates`, with that date's trailing climatology
    percentile and how many distinct prior years supported it. `dates` is typically
    the union of every date appearing in the training and forecast frames -- future
    forecast dates are always after everything in `reference`, so they simply draw
    on the full available history with no leakage question."""
    cfg = _cfg(config)
    percentile = float(cfg.get("percentile", DEFAULT_PERCENTILE))
    window_days = int(cfg.get("day_of_year_window_days", DEFAULT_DAY_OF_YEAR_WINDOW_DAYS))
    min_reference_years = int(cfg.get("min_reference_years", DEFAULT_MIN_REFERENCE_YEARS))

    unique_dates = pd.to_datetime(pd.Series(dates).dropna().unique())
    rows = []
    for date in unique_dates:
        value, n_years = _trailing_percentile_for_date(
            reference,
            target_year=int(date.year),
            target_doy=int(date.dayofyear),
            percentile=percentile,
            window_days=window_days,
            min_reference_years=min_reference_years,
        )
        rows.append(
            {
                "Date": pd.Timestamp(date.date()),
                "Climatology_Temp_PXX_F": value,
                "Temp_Climatology_Reference_Years": n_years,
            }
        )
    return pd.DataFrame(
        rows, columns=["Date", "Climatology_Temp_PXX_F", "Temp_Climatology_Reference_Years"]
    )


def add_record_breaking_heat_features(
    df: pd.DataFrame, climatology_lookup: pd.DataFrame | None, config: dict | None = None
) -> pd.DataFrame:
    """Merges the precomputed climatology lookup onto `df` by date and derives
    Temp_Excess_Over_Climatology_F = Temperature_DailyMax - that date's trailing
    Pxx reference. Disabled (config off, or no lookup available) is a true no-op --
    the column is simply never added, so DEFAULT_FEATURES can list it unconditionally
    and _available_features() will just skip it when absent, same as every other
    optional feature in this project."""
    out = df.copy()
    cfg = _cfg(config)
    if not bool(cfg.get("enabled", False)):
        return out
    if climatology_lookup is None or climatology_lookup.empty:
        return out
    if "DT" not in out.columns or "Temperature_DailyMax" not in out.columns:
        return out

    out["Date"] = pd.to_datetime(out["DT"], errors="coerce").dt.normalize().dt.tz_localize(None)
    lookup = climatology_lookup.copy()
    lookup["Date"] = pd.to_datetime(lookup["Date"]).dt.normalize()
    out = out.merge(lookup, on="Date", how="left")
    out.drop(columns=["Date"], inplace=True)

    daily_max = pd.to_numeric(out["Temperature_DailyMax"], errors="coerce")
    reference = pd.to_numeric(out["Climatology_Temp_PXX_F"], errors="coerce")
    out["Temp_Excess_Over_Climatology_F"] = daily_max - reference
    return out
