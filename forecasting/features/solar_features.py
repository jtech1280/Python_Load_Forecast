from __future__ import annotations

import numpy as np
import pandas as pd


def solar_hour_shape_from_hour(hour_values):
    hour = pd.Series(hour_values).astype(float)
    shape = np.sin(np.pi * (hour - 6.0) / 12.0)
    return np.clip(shape, 0.0, 1.0)


def _solar_season_factor(dt: pd.Series) -> pd.Series:
    """Simple Sacramento-area seasonal daylight/solar availability proxy, 0.65 winter to 1.0 summer."""
    doy = pd.to_datetime(dt).dt.dayofyear.astype(float)
    # Peak near June 21 (day 172), trough near Dec 21.
    return pd.Series(0.825 + 0.175 * np.cos(2.0 * np.pi * (doy - 172.0) / 365.25), index=dt.index).clip(0.60, 1.05)


def add_solar_features(df: pd.DataFrame, btm_monthly_df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    btm = btm_monthly_df.copy()
    btm["PeriodStart"] = pd.to_datetime(btm["DT"]).dt.to_period("M").dt.to_timestamp()
    out["PeriodStart"] = out["DT"].dt.tz_localize(None).dt.to_period("M").dt.to_timestamp()
    out = out.merge(
        btm[["PeriodStart", "Nameplate_MW", "Capacity_Ratio_To_Current", "Impact_Cap_MW"]],
        on="PeriodStart",
        how="left",
    )
    out.drop(columns=["PeriodStart"], inplace=True)

    out["Nameplate_MW"] = out["Nameplate_MW"].ffill().bfill().fillna(0.0)
    out["Capacity_Ratio_To_Current"] = out["Capacity_Ratio_To_Current"].ffill().bfill().fillna(0.0)
    out["Impact_Cap_MW"] = out["Impact_Cap_MW"].ffill().bfill().fillna(0.0)

    out["Solar_Irradiance"] = pd.to_numeric(out.get("GHI_Wm2"), errors="coerce").fillna(0.0).clip(lower=0.0)
    out["Solar_Hour_Shape"] = solar_hour_shape_from_hour(out["DT"].dt.hour)
    out["Solar_Season_Factor"] = _solar_season_factor(out["DT"])

    # Existing production proxy retained for compatibility. It is intentionally conservative.
    out["BTM_Solar_Proxy_MW"] = (
        out["Impact_Cap_MW"]
        * (out["Solar_Irradiance"].clip(lower=0.0) / 950.0)
        * out["Solar_Hour_Shape"]
    ).clip(lower=0.0)

    # V12 cloud/clear-sky features. These help the model learn that cloudy midday periods increase
    # system load by reducing behind-the-meter PV output.
    out["ClearSky_GHI_Proxy_Wm2"] = (1000.0 * out["Solar_Hour_Shape"] * out["Solar_Season_Factor"]).clip(lower=0.0)
    denom = out["ClearSky_GHI_Proxy_Wm2"].replace(0.0, np.nan)
    out["ClearSky_Index"] = (out["Solar_Irradiance"] / denom).replace([np.inf, -np.inf], np.nan).clip(0.0, 1.35).fillna(0.0)
    out["BTM_ClearSky_Proxy_MW"] = (
        out["Impact_Cap_MW"]
        * (out["ClearSky_GHI_Proxy_Wm2"] / 950.0)
        * out["Solar_Hour_Shape"]
    ).clip(lower=0.0)
    out["BTM_Solar_Cloud_Adjusted_MW"] = out["BTM_Solar_Proxy_MW"]
    out["BTM_Solar_Loss_From_ClearSky_MW"] = (out["BTM_ClearSky_Proxy_MW"] - out["BTM_Solar_Cloud_Adjusted_MW"]).clip(lower=0.0)

    cloud = pd.to_numeric(out.get("CloudCover_Norm", 0.0), errors="coerce").fillna(0.0)
    if cloud.max(skipna=True) > 1.5:
        cloud = cloud / 100.0
    cloud = cloud.clip(0.0, 1.0)
    out["Cloud_x_Solar_Hour"] = cloud * out["Solar_Hour_Shape"]
    out["Cloud_x_ClearSky_GHI"] = cloud * out["ClearSky_GHI_Proxy_Wm2"]

    out["Date"] = out["DT"].dt.date
    out["Daily_BTM_Solar_Proxy_Total_MWh"] = out.groupby("Date")["BTM_Solar_Proxy_MW"].transform("sum")
    out["Daily_BTM_Solar_Proxy_Max_MW"] = out.groupby("Date")["BTM_Solar_Proxy_MW"].transform("max")
    out["Daily_BTM_ClearSky_Max_MW"] = out.groupby("Date")["BTM_ClearSky_Proxy_MW"].transform("max")
    out["Daily_BTM_Solar_Loss_MWh"] = out.groupby("Date")["BTM_Solar_Loss_From_ClearSky_MW"].transform("sum")
    out["Daily_BTM_Solar_Loss_Max_MW"] = out.groupby("Date")["BTM_Solar_Loss_From_ClearSky_MW"].transform("max")

    out["BTM_x_GHI"] = out["BTM_Solar_Proxy_MW"] * out["Solar_Irradiance"]
    out["BTM_x_Cloud"] = out["BTM_Solar_Proxy_MW"] * cloud
    out["Solar_Midday_Flag"] = out["DT"].dt.hour.between(10, 15).astype(int)
    out["Solar_Evening_Ramp_Flag"] = out["DT"].dt.hour.between(16, 20).astype(int)
    out["BTM_Midday_Impact"] = out["BTM_Solar_Proxy_MW"] * out["Solar_Midday_Flag"]
    out["BTM_Evening_Ramp_Impact"] = out["BTM_Solar_Proxy_MW"] * out["Solar_Evening_Ramp_Flag"]
    out["Midday_Overcast_Solar_Loss_MW"] = out["BTM_Solar_Loss_From_ClearSky_MW"] * out["Solar_Midday_Flag"] * (cloud >= 0.60).astype(int)

    out = out.sort_values("DT").reset_index(drop=True)
    btm_diff_1 = out["BTM_Solar_Proxy_MW"].diff()
    btm_diff_2 = out["BTM_Solar_Proxy_MW"].diff(2)
    out["Solar_Ramp_Down_1hr"] = (-btm_diff_1).clip(lower=0.0).fillna(0.0)
    out["Solar_Ramp_Down_2hr"] = (-btm_diff_2).clip(lower=0.0).fillna(0.0)
    out["Solar_Ramp_Up_1hr"] = btm_diff_1.clip(lower=0.0).fillna(0.0)
    return out
