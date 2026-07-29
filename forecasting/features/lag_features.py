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
    clear_hot = _num(out, "ClearHotEvening_Flag", 0.0).fillna(0.0).clip(lower=0.0, upper=1.0)
    delta_breeze_flag = (
        _num(out, "DeltaBreeze_Cooling_Flag", 0.0).fillna(0.0).clip(lower=0.0, upper=1.0)
        .combine(_num(out, "DeltaBreeze_Westerly_Flow_Flag", 0.0).fillna(0.0).clip(lower=0.0, upper=1.0), max)
        .combine(_num(out, "DeltaBreeze_EveningWindRamp_Flag", 0.0).fillna(0.0).clip(lower=0.0, upper=1.0), max)
    )

    # Positive values mean the most recent load memory is already decaying / running below
    # the historical same-hour reference before the current prediction is made.
    out["Load_Decay_1Hr_MWH"] = lag2 - lag1
    out["Load_Decay_2Hr_MWH"] = lag3 - lag1
    out["Lag1_Minus_SameHourYesterday_MWH"] = lag1 - lag24
    out["Lag1_Minus_SameHour7DayMean_MWH"] = lag1 - same_hour_7day
    out["PostPeak_LoadDecay_1Hr_MWH"] = post_peak * out["Load_Decay_1Hr_MWH"]
    out["PostPeak_LoadDecay_2Hr_MWH"] = post_peak * out["Load_Decay_2Hr_MWH"]
    out["PostPeak_LoadDecay_VsSameHourYesterday_MWH"] = post_peak * (lag24 - lag1)
    out["PostPeak_LoadDecay_VsSameHour7DayMean_MWH"] = post_peak * (same_hour_7day - lag1)
    out["ClearHotEvening_LoadDecay_Vs7Day_MWH"] = clear_hot * out["PostPeak_LoadDecay_VsSameHour7DayMean_MWH"]
    out["DeltaBreeze_PostPeak_LoadDecay_Signal"] = (
        delta_breeze_flag
        * (
            out["PostPeak_LoadDecay_1Hr_MWH"].clip(lower=0.0).fillna(0.0)
            + out["PostPeak_LoadDecay_VsSameHour7DayMean_MWH"].clip(lower=0.0).fillna(0.0)
        )
    )
    return out


def add_basic_lags(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy().sort_values("DT")
    y = out["MWH"].astype(float)

    # Shift first so training does not leak the current target into rolling statistics.
    shifted = y.shift(1)
    for lag in [1, 2, 3, 24, 48, 72, 168]:
        out[f"MWH_Lag{lag}"] = y.shift(lag)

    for window in [3, 6, 12, 24, 48, 168]:
        min_periods = max(2, min(window, int(window * 0.5)))
        out[f"MWH_Rolling{window}"] = shifted.rolling(window=window, min_periods=min_periods).mean()

    out["MWH_Rolling24Std"] = shifted.rolling(window=24, min_periods=12).std().fillna(0.0)
    same_hour_lags = [y.shift(24 * i) for i in range(1, 8)]
    out["MWH_SameHour7DayMean"] = pd.concat(same_hour_lags, axis=1).mean(axis=1)
    out = add_load_decay_shape_features(out)
    return out
