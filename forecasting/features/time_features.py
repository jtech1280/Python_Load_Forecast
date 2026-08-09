from __future__ import annotations

import math
import pandas as pd
import numpy as np


def _nth_weekday(year: int, month: int, weekday: int, n: int) -> pd.Timestamp:
    """Return nth weekday in a month. Monday=0."""
    first = pd.Timestamp(year=year, month=month, day=1)
    offset = (weekday - first.dayofweek) % 7
    return first + pd.Timedelta(days=offset + 7 * (n - 1))


def _last_weekday(year: int, month: int, weekday: int) -> pd.Timestamp:
    """Return last weekday in a month. Monday=0."""
    last = pd.Timestamp(year=year, month=month, day=1) + pd.offsets.MonthEnd(0)
    offset = (last.dayofweek - weekday) % 7
    return last - pd.Timedelta(days=offset)


def _observed_fixed_holiday(year: int, month: int, day: int) -> pd.Timestamp:
    """Observed utility holiday for fixed-date holidays."""
    d = pd.Timestamp(year=year, month=month, day=day)
    if d.dayofweek == 5:  # Saturday observed Friday
        return d - pd.Timedelta(days=1)
    if d.dayofweek == 6:  # Sunday observed Monday
        return d + pd.Timedelta(days=1)
    return d


def roseville_holidays(year: int) -> set[pd.Timestamp]:
    """
    Roseville Electric-style holiday set used for calendar/load behavior features.
    Includes: New Year's Day, MLK Day, Presidents' Day, Memorial Day,
    Independence Day, Labor Day, Columbus Day, Veterans Day, Thanksgiving Day,
    and Christmas Day.
    """
    return {
        _observed_fixed_holiday(year, 1, 1),  # New Year's Day
        _nth_weekday(year, 1, 0, 3),  # Martin Luther King Jr. Day
        _nth_weekday(year, 2, 0, 3),  # Presidents' Day
        _last_weekday(year, 5, 0),  # Memorial Day
        _observed_fixed_holiday(year, 7, 4),  # Independence Day
        _nth_weekday(year, 9, 0, 1),  # Labor Day
        _nth_weekday(year, 10, 0, 2),  # Columbus Day
        _observed_fixed_holiday(year, 11, 11),  # Veterans Day
        _nth_weekday(year, 11, 3, 4),  # Thanksgiving Day
        _observed_fixed_holiday(year, 12, 25),  # Christmas Day
    }


def _holiday_frame(dates: pd.Series) -> pd.DataFrame:
    local_dt = pd.to_datetime(dates, errors="coerce")
    if getattr(local_dt.dt, "tz", None) is not None:
        # Compare local calendar dates. Holiday lookup timestamps are timezone-naive,
        # while forecast/replay DT values are local tz-aware.
        local_dt = local_dt.dt.tz_localize(None)
    d = local_dt.dt.normalize()
    years = sorted(set(d.dt.year.dropna().astype(int).tolist()))
    holiday_dates = set()
    for y in years:
        holiday_dates.update(roseville_holidays(y - 1))
        holiday_dates.update(roseville_holidays(y))
        holiday_dates.update(roseville_holidays(y + 1))

    holiday_norm = {pd.Timestamp(x).normalize() for x in holiday_dates}
    is_holiday = d.isin(holiday_norm)
    is_pre = (d + pd.Timedelta(days=1)).isin(holiday_norm)
    is_post = (d - pd.Timedelta(days=1)).isin(holiday_norm)
    return pd.DataFrame(
        {
            "IsHoliday": is_holiday.astype(int),
            "IsPreHoliday": is_pre.astype(int),
            "IsPostHoliday": is_post.astype(int),
            "IsHolidayAdjacent": (is_holiday | is_pre | is_post).astype(int),
        },
        index=dates.index,
    )


def _hour_group(hour: int) -> str:
    if 0 <= hour <= 5:
        return "Overnight"
    if 6 <= hour <= 9:
        return "Morning"
    if 10 <= hour <= 15:
        return "Midday"
    if 16 <= hour <= 20:
        return "Peak"
    return "LateEvening"


def add_time_features(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    out["Year"] = out["DT"].dt.year
    out["Month"] = out["DT"].dt.month
    out["Day"] = out["DT"].dt.day
    out["DayOfYear"] = out["DT"].dt.dayofyear
    out["WeekOfYear"] = out["DT"].dt.isocalendar().week.astype(int)
    out["Hour"] = out["DT"].dt.hour
    out["DOW"] = out["DT"].dt.dayofweek
    out["IsWeekend"] = (out["DOW"] >= 5).astype(int)
    out["IsMonday"] = (out["DOW"] == 0).astype(int)
    out["IsFriday"] = (out["DOW"] == 4).astype(int)

    hol = _holiday_frame(out["DT"])
    for c in hol.columns:
        out[c] = hol[c].values
    out["IsBusinessDay"] = ((out["IsWeekend"] == 0) & (out["IsHoliday"] == 0)).astype(
        int
    )

    # Cyclical encodings let the model understand wrap-around boundaries.
    out["HourSin"] = np.sin(2.0 * math.pi * out["Hour"] / 24.0)
    out["HourCos"] = np.cos(2.0 * math.pi * out["Hour"] / 24.0)
    out["DOWSin"] = np.sin(2.0 * math.pi * out["DOW"] / 7.0)
    out["DOWCos"] = np.cos(2.0 * math.pi * out["DOW"] / 7.0)
    out["MonthSin"] = np.sin(2.0 * math.pi * out["Month"] / 12.0)
    out["MonthCos"] = np.cos(2.0 * math.pi * out["Month"] / 12.0)
    out["DayOfYearSin"] = np.sin(2.0 * math.pi * out["DayOfYear"] / 366.0)
    out["DayOfYearCos"] = np.cos(2.0 * math.pi * out["DayOfYear"] / 366.0)

    out["Season"] = out["Month"].map(_month_to_season)
    out["HourGroup"] = out["Hour"].map(_hour_group)

    # Roseville TOU/load-behavior flags. Hour follows the model's official completed-hour label.
    out["IsSummerSeason"] = out["Month"].between(6, 9).astype(int)
    out["IsWinterSeason"] = (~out["Month"].between(6, 9)).astype(int)
    offpeak = (
        out["IsWeekend"].eq(1)
        | out["IsHoliday"].eq(1)
        | (out["Hour"] < 7)
        | (out["Hour"] >= 22)
    )
    superpeak = out["IsBusinessDay"].eq(1) & out["Hour"].between(16, 18)
    onpeak = out["IsBusinessDay"].eq(1) & ~offpeak & ~superpeak
    out["IsOffPeak"] = offpeak.astype(int)
    out["IsOnPeak"] = onpeak.astype(int)
    out["IsSuperPeak"] = superpeak.astype(int)
    out["IsLikelySystemPeakHour"] = (
        out["Hour"].between(16, 20) & out["IsBusinessDay"].eq(1)
    ).astype(int)
    return out


def _month_to_season(m: int) -> str:
    if m in (12, 1, 2):
        return "Winter"
    if m in (3, 4, 5):
        return "Spring"
    if m in (6, 7, 8, 9):
        return "Summer"
    if m in (10, 11):
        return "Fall"
    return "Unknown"
