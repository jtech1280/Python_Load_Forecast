from __future__ import annotations

import pandas as pd
import numpy as np

LAG_FEATURES = [
    "MWH_Lag1",
    "MWH_Lag2",
    "MWH_Lag3",
    "MWH_Lag24",
    "MWH_Lag48",
    "MWH_Lag72",
    "MWH_Lag168",
    "MWH_Rolling3",
    "MWH_Rolling6",
    "MWH_Rolling12",
    "MWH_Rolling24",
    "MWH_Rolling48",
    "MWH_Rolling168",
    "MWH_Rolling24Std",
    "MWH_SameHour7DayMean",
    "Load_Decay_1Hr_MWH",
    "Load_Decay_2Hr_MWH",
    "Lag1_Minus_SameHourYesterday_MWH",
    "Lag1_Minus_SameHour7DayMean_MWH",
    "Lag24_Minus_SameHour7DayMean_MWH",
    "PeakWindow_Lag1_Minus_SameHour7DayMean_MWH",
    "PeakWindow_Lag24_Minus_SameHour7DayMean_MWH",
    "PeakWindow16to18_Lag1_Minus_SameHour7DayMean_MWH",
    "PeakWindow16to18_Lag24_Minus_SameHour7DayMean_MWH",
    "HotPeak_Lag1_Minus_SameHourYesterday_MWH",
    "HotPeak_Lag1_Minus_SameHour7DayMean_MWH",
    "HotPeak_Lag24_Minus_SameHour7DayMean_MWH",
    "ClearHotPeak_Lag1_Minus_SameHourYesterday_MWH",
    "ClearHotPeak_Lag1_Minus_SameHour7DayMean_MWH",
    "ClearHotPeak_Lag24_Minus_SameHour7DayMean_MWH",
    "ClearPeak16to18_Lag24_Minus_SameHour7DayMean_MWH",
    "OvercastPeak16to18_Lag24_Minus_SameHour7DayMean_MWH",
    "ClearHotPeak16to18_Lag24_Minus_SameHour7DayMean_MWH",
    "OvercastCoolPeak16to18_Lag1_Minus_SameHour7DayMean_MWH",
    "OvercastCoolPeak16to18_Lag24_Minus_SameHour7DayMean_MWH",
    "PostPeak_LoadDecay_1Hr_MWH",
    "PostPeak_LoadDecay_2Hr_MWH",
    "PostPeak_LoadDecay_VsSameHourYesterday_MWH",
    "PostPeak_LoadDecay_VsSameHour7DayMean_MWH",
    "ClearHotEvening_LoadDecay_Vs7Day_MWH",
    "DeltaBreeze_PostPeak_LoadDecay_Signal",
]


def _num(df: pd.DataFrame, col: str, default: float = 0.0) -> pd.Series:
    if col in df.columns:
        return pd.to_numeric(df[col], errors="coerce")
    return pd.Series(default, index=df.index, dtype=float)


def add_load_decay_shape_features(df: pd.DataFrame) -> pd.DataFrame:
    """Add recursive-safe post-peak load decay features derived from lagged load only."""
    out = df.copy()
    lag1 = _num(out, "MWH_Lag1")
    lag2 = _num(out, "MWH_Lag2")
    lag3 = _num(out, "MWH_Lag3")
    lag24 = _num(out, "MWH_Lag24")
    same_hour_7day = _num(out, "MWH_SameHour7DayMean")
    hour = _num(out, "Hour", np.nan)
    post_peak = _num(out, "IsPostPeakEvening18to23", np.nan)
    if post_peak.isna().all():
        post_peak = hour.between(18, 23).astype(float)
    post_peak = post_peak.fillna(0.0).clip(lower=0.0, upper=1.0)
    peak_window = (
        _num(out, "IsPeakWindow14to18", 0.0).fillna(0.0).clip(lower=0.0, upper=1.0)
    )
    peak_window_16_18 = (
        _num(out, "IsPeakWindow16to18", 0.0).fillna(0.0).clip(lower=0.0, upper=1.0)
    )
    hot_peak = (
        _num(out, "IsHotPeakWindow16to20", 0.0).fillna(0.0).clip(lower=0.0, upper=1.0)
    )
    clear_hot_peak = (
        _num(out, "ClearHotPeakWindow16to20", 0.0)
        .fillna(0.0)
        .clip(lower=0.0, upper=1.0)
    )
    clear_peak_16_18 = (
        _num(out, "ClearPeakWindow16to18", 0.0).fillna(0.0).clip(lower=0.0, upper=1.0)
    )
    overcast_peak_16_18 = (
        _num(out, "OvercastPeakWindow16to18", 0.0)
        .fillna(0.0)
        .clip(lower=0.0, upper=1.0)
    )
    clear_hot_peak_16_18 = (
        _num(out, "ClearHotPeakWindow16to18", 0.0)
        .fillna(0.0)
        .clip(lower=0.0, upper=1.0)
    )
    overcast_cool_peak_16_18 = (
        _num(out, "OvercastCoolPeakWindow16to18", 0.0)
        .fillna(0.0)
        .clip(lower=0.0, upper=1.0)
    )
    clear_hot = (
        _num(out, "ClearHotEvening_Flag", 0.0).fillna(0.0).clip(lower=0.0, upper=1.0)
    )
    delta_breeze_flag = (
        _num(out, "DeltaBreeze_Cooling_Flag", 0.0)
        .fillna(0.0)
        .clip(lower=0.0, upper=1.0)
        .combine(
            _num(out, "DeltaBreeze_Westerly_Flow_Flag", 0.0)
            .fillna(0.0)
            .clip(lower=0.0, upper=1.0),
            max,
        )
        .combine(
            _num(out, "DeltaBreeze_EveningWindRamp_Flag", 0.0)
            .fillna(0.0)
            .clip(lower=0.0, upper=1.0),
            max,
        )
    )

    # Positive values mean the most recent load memory is already decaying / running below
    # the historical same-hour reference before the current prediction is made.
    out["Load_Decay_1Hr_MWH"] = lag2 - lag1
    out["Load_Decay_2Hr_MWH"] = lag3 - lag1
    out["Lag1_Minus_SameHourYesterday_MWH"] = lag1 - lag24
    out["Lag1_Minus_SameHour7DayMean_MWH"] = lag1 - same_hour_7day
    out["Lag24_Minus_SameHour7DayMean_MWH"] = lag24 - same_hour_7day
    out["PeakWindow_Lag1_Minus_SameHour7DayMean_MWH"] = (
        peak_window * out["Lag1_Minus_SameHour7DayMean_MWH"]
    )
    out["PeakWindow_Lag24_Minus_SameHour7DayMean_MWH"] = (
        peak_window * out["Lag24_Minus_SameHour7DayMean_MWH"]
    )
    out["PeakWindow16to18_Lag1_Minus_SameHour7DayMean_MWH"] = (
        peak_window_16_18 * out["Lag1_Minus_SameHour7DayMean_MWH"]
    )
    out["PeakWindow16to18_Lag24_Minus_SameHour7DayMean_MWH"] = (
        peak_window_16_18 * out["Lag24_Minus_SameHour7DayMean_MWH"]
    )
    out["HotPeak_Lag1_Minus_SameHourYesterday_MWH"] = (
        hot_peak * out["Lag1_Minus_SameHourYesterday_MWH"]
    )
    out["HotPeak_Lag1_Minus_SameHour7DayMean_MWH"] = (
        hot_peak * out["Lag1_Minus_SameHour7DayMean_MWH"]
    )
    out["HotPeak_Lag24_Minus_SameHour7DayMean_MWH"] = (
        hot_peak * out["Lag24_Minus_SameHour7DayMean_MWH"]
    )
    out["ClearHotPeak_Lag1_Minus_SameHourYesterday_MWH"] = (
        clear_hot_peak * out["Lag1_Minus_SameHourYesterday_MWH"]
    )
    out["ClearHotPeak_Lag1_Minus_SameHour7DayMean_MWH"] = (
        clear_hot_peak * out["Lag1_Minus_SameHour7DayMean_MWH"]
    )
    out["ClearHotPeak_Lag24_Minus_SameHour7DayMean_MWH"] = (
        clear_hot_peak * out["Lag24_Minus_SameHour7DayMean_MWH"]
    )
    out["ClearPeak16to18_Lag24_Minus_SameHour7DayMean_MWH"] = (
        clear_peak_16_18 * out["Lag24_Minus_SameHour7DayMean_MWH"]
    )
    out["OvercastPeak16to18_Lag24_Minus_SameHour7DayMean_MWH"] = (
        overcast_peak_16_18 * out["Lag24_Minus_SameHour7DayMean_MWH"]
    )
    out["ClearHotPeak16to18_Lag24_Minus_SameHour7DayMean_MWH"] = (
        clear_hot_peak_16_18 * out["Lag24_Minus_SameHour7DayMean_MWH"]
    )
    out["OvercastCoolPeak16to18_Lag1_Minus_SameHour7DayMean_MWH"] = (
        overcast_cool_peak_16_18 * out["Lag1_Minus_SameHour7DayMean_MWH"]
    )
    out["OvercastCoolPeak16to18_Lag24_Minus_SameHour7DayMean_MWH"] = (
        overcast_cool_peak_16_18 * out["Lag24_Minus_SameHour7DayMean_MWH"]
    )
    out["PostPeak_LoadDecay_1Hr_MWH"] = post_peak * out["Load_Decay_1Hr_MWH"]
    out["PostPeak_LoadDecay_2Hr_MWH"] = post_peak * out["Load_Decay_2Hr_MWH"]
    out["PostPeak_LoadDecay_VsSameHourYesterday_MWH"] = post_peak * (lag24 - lag1)
    out["PostPeak_LoadDecay_VsSameHour7DayMean_MWH"] = post_peak * (
        same_hour_7day - lag1
    )
    out["ClearHotEvening_LoadDecay_Vs7Day_MWH"] = (
        clear_hot * out["PostPeak_LoadDecay_VsSameHour7DayMean_MWH"]
    )
    out["DeltaBreeze_PostPeak_LoadDecay_Signal"] = delta_breeze_flag * (
        out["PostPeak_LoadDecay_1Hr_MWH"].clip(lower=0.0).fillna(0.0)
        + out["PostPeak_LoadDecay_VsSameHour7DayMean_MWH"].clip(lower=0.0).fillna(0.0)
    )
    return out


def add_basic_lags(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy().sort_values("DT").reset_index(drop=True)

    dt_index = pd.DatetimeIndex(out["DT"])
    y_by_dt = pd.Series(out["MWH"].astype(float).to_numpy(), index=dt_index)
    y_by_dt = y_by_dt[~y_by_dt.index.duplicated(keep="last")]

    # Compute against a complete, gapless hourly grid rather than plain positional
    # .shift() on `out` directly. A missing hour anywhere in history -- DST
    # spring-forward guarantees at least one a year, and missing/estimated source
    # data can add more -- would otherwise silently shift every later row's "N
    # hours/days ago" lookup by one position instead of producing a real NaN at
    # just the affected row, quietly corrupting same-hour/rolling features for the
    # entire remainder of the dataset.
    full_index = pd.date_range(y_by_dt.index.min(), y_by_dt.index.max(), freq="h")
    y_full = y_by_dt.reindex(full_index)

    # Shift first so training does not leak the current target into rolling statistics.
    shifted_full = y_full.shift(1)
    lag_cols = {f"MWH_Lag{lag}": y_full.shift(lag) for lag in [1, 2, 3, 24, 48, 72, 168]}

    for window in [3, 6, 12, 24, 48, 168]:
        min_periods = max(2, min(window, int(window * 0.5)))
        lag_cols[f"MWH_Rolling{window}"] = shifted_full.rolling(
            window=window, min_periods=min_periods
        ).mean()

    lag_cols["MWH_Rolling24Std"] = (
        shifted_full.rolling(window=24, min_periods=12).std().fillna(0.0)
    )
    same_hour_lags = [y_full.shift(24 * i) for i in range(1, 8)]
    lag_cols["MWH_SameHour7DayMean"] = pd.concat(same_hour_lags, axis=1).mean(axis=1)

    lag_frame = pd.DataFrame(lag_cols).reindex(dt_index).reset_index(drop=True)
    out = pd.concat([out, lag_frame], axis=1)
    out = add_load_decay_shape_features(out)
    return out
