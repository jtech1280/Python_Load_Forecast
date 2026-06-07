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
]


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
    return out
