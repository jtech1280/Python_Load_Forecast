from __future__ import annotations

import pandas as pd
from .time_features import add_time_features
from .weather_features import add_weather_features
from .solar_features import add_solar_features
from .lag_features import add_basic_lags
from .intraday_load_features import merge_intraday_load_features, zero_intraday_load_features

def build_training_frame(
    load_df: pd.DataFrame,
    weather_df: pd.DataFrame,
    btm_monthly_df: pd.DataFrame,
    intraday_load_features: pd.DataFrame | None = None,
) -> pd.DataFrame:
    df = load_df.merge(weather_df, on="DT", how="left")
    df = add_time_features(df)
    df = add_weather_features(df)
    df = add_solar_features(df, btm_monthly_df)
    df = merge_intraday_load_features(df, intraday_load_features, allow_carry_forward=False)
    df = add_basic_lags(df)
    df = df.dropna(subset=["MWH", "Temperature"])  # require target + temp
    return df

def build_forecast_frame(
    weather_df: pd.DataFrame,
    btm_monthly_df: pd.DataFrame,
    intraday_load_features: pd.DataFrame | None = None,
    max_intraday_carry_forward_hours: int = 24,
    use_future_intraday_load_features: bool = True,
) -> pd.DataFrame:
    df = weather_df.copy()
    df = add_time_features(df)
    df = add_weather_features(df)
    df = add_solar_features(df, btm_monthly_df)
    if use_future_intraday_load_features:
        df = merge_intraday_load_features(
            df,
            intraday_load_features,
            allow_carry_forward=True,
            max_carry_forward_hours=max_intraday_carry_forward_hours,
        )
    else:
        df = zero_intraday_load_features(df)
    # No MWH column yet (future), lag features added during recursion using historical seed
    return df
