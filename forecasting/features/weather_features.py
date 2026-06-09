from __future__ import annotations

import pandas as pd
import numpy as np

TEMP_BIN_EDGES = [-999, 45, 55, 65, 75, 80, 85, 90, 95, 100, 999]
DAILY_MAX_BIN_EDGES = [-999, 65, 75, 85, 90, 95, 100, 105, 999]
BASE_TEMP = 65.0


def _heat_index_f(temp_f: pd.Series, rh_pct: pd.Series) -> pd.Series:
    """NOAA-style heat index approximation; falls back to temperature below hot/humid ranges."""
    t = temp_f.astype(float)
    rh = rh_pct.astype(float).clip(lower=0, upper=100)
    hi = (
        -42.379 + 2.04901523 * t + 10.14333127 * rh
        - 0.22475541 * t * rh - 0.00683783 * t * t
        - 0.05481717 * rh * rh + 0.00122874 * t * t * rh
        + 0.00085282 * t * rh * rh - 0.00000199 * t * t * rh * rh
    )
    return np.where((t >= 80) & (rh >= 40), hi, t)


def _consecutive_true_count(flag: pd.Series) -> pd.Series:
    values = flag.fillna(False).astype(bool)
    groups = values.ne(values.shift(fill_value=False)).cumsum()
    counts = values.groupby(groups).cumsum()
    return counts.where(values, 0).astype(float)


def add_heat_persistence_features(df: pd.DataFrame) -> pd.DataFrame:
    """Add daily heat-stress fields derived only from weather/calendar inputs."""
    out = df.copy()
    if "DT" not in out.columns:
        return out
    if "Date" not in out.columns:
        out["Date"] = pd.to_datetime(out["DT"], errors="coerce").dt.date

    required = {"Temperature_DailyMax", "Temperature_DailyMin", "Temperature_DailyMean"}
    if not required.issubset(out.columns):
        return out

    daily = (
        out[["Date", "Temperature_DailyMax", "Temperature_DailyMin", "Temperature_DailyMean"]]
        .dropna(subset=["Date"])
        .drop_duplicates(subset=["Date"])
        .sort_values("Date")
        .reset_index(drop=True)
    )
    if daily.empty:
        return out

    max_temp = pd.to_numeric(daily["Temperature_DailyMax"], errors="coerce")
    min_temp = pd.to_numeric(daily["Temperature_DailyMin"], errors="coerce")
    mean_temp = pd.to_numeric(daily["Temperature_DailyMean"], errors="coerce")

    daily["PriorDay_DailyMaxTemp"] = max_temp.shift(1).fillna(max_temp)
    daily["PriorDay_DailyMinTemp"] = min_temp.shift(1).fillna(min_temp)
    daily["DailyMaxTemp_Ramp_1Day"] = (max_temp - daily["PriorDay_DailyMaxTemp"]).fillna(0.0)
    daily["DailyMinTemp_Ramp_1Day"] = (min_temp - daily["PriorDay_DailyMinTemp"]).fillna(0.0)
    daily["DailyMaxTemp_2DayMean"] = max_temp.rolling(2, min_periods=1).mean()
    daily["DailyMaxTemp_3DayMean"] = max_temp.rolling(3, min_periods=1).mean()
    daily["DailyMinTemp_2DayMean"] = min_temp.rolling(2, min_periods=1).mean()
    daily["DailyMinTemp_3DayMean"] = min_temp.rolling(3, min_periods=1).mean()
    daily["DailyMeanTemp_3DayMean"] = mean_temp.rolling(3, min_periods=1).mean()
    daily["ConsecutiveHotDays90"] = _consecutive_true_count(max_temp.ge(90.0))
    daily["ConsecutiveVeryHotDays95"] = _consecutive_true_count(max_temp.ge(95.0))
    daily["ConsecutiveExtremeHotDays100"] = _consecutive_true_count(max_temp.ge(100.0))
    daily["OvernightHeatStress"] = (min_temp - 70.0).clip(lower=0.0)

    lookup = daily.set_index("Date")
    for col in [
        "PriorDay_DailyMaxTemp",
        "PriorDay_DailyMinTemp",
        "DailyMaxTemp_Ramp_1Day",
        "DailyMinTemp_Ramp_1Day",
        "DailyMaxTemp_2DayMean",
        "DailyMaxTemp_3DayMean",
        "DailyMinTemp_2DayMean",
        "DailyMinTemp_3DayMean",
        "DailyMeanTemp_3DayMean",
        "ConsecutiveHotDays90",
        "ConsecutiveVeryHotDays95",
        "ConsecutiveExtremeHotDays100",
        "OvernightHeatStress",
    ]:
        out[col] = out["Date"].map(lookup[col]).astype(float)

    peak_hour = pd.to_numeric(out.get("IsLikelySystemPeakHour", 0), errors="coerce").fillna(0.0)
    out["HeatPersistenceStress90"] = out["ConsecutiveHotDays90"].fillna(0.0) * peak_hour
    out["HeatPersistenceStress95"] = out["ConsecutiveVeryHotDays95"].fillna(0.0) * peak_hour
    out["DailyMax3DayMean_x_PeakHour"] = out["DailyMaxTemp_3DayMean"].fillna(0.0) * peak_hour
    out["OvernightHeatStress_x_PeakHour"] = out["OvernightHeatStress"].fillna(0.0) * peak_hour
    return out


def add_weather_features(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    out["Temperature"] = pd.to_numeric(out.get("TempF"), errors="coerce")
    out["HumidityPct"] = pd.to_numeric(out.get("HumidityPct"), errors="coerce")
    out["CloudCoverPct"] = pd.to_numeric(out.get("CloudCoverPct"), errors="coerce")
    out["WindSpeed_Mph"] = pd.to_numeric(out.get("WindSpeedMph"), errors="coerce").fillna(0.0)
    out["PrecipIn"] = pd.to_numeric(out.get("PrecipIn"), errors="coerce").fillna(0.0)
    out["GHI_Wm2"] = pd.to_numeric(out.get("GHI_Wm2"), errors="coerce").fillna(0.0)

    out["Humidity_Norm"] = (out["HumidityPct"] / 100.0).clip(0, 1).fillna(0.0)
    out["CloudCover_Norm"] = (out["CloudCoverPct"] / 100.0).clip(0, 1).fillna(0.0)
    out["Is_Raining"] = (out["PrecipIn"].fillna(0.0) > 0).astype(int)

    out["CDD"] = (out["Temperature"] - BASE_TEMP).clip(lower=0.0)
    out["HDD"] = (BASE_TEMP - out["Temperature"]).clip(lower=0.0)
    out["Temp_Squared"] = out["Temperature"] ** 2
    out["CDD_Squared"] = out["CDD"] ** 2
    out["HDD_Squared"] = out["HDD"] ** 2

    out["Extreme_Heat_80"] = (out["Temperature"] >= 80.0).astype(int)
    out["Extreme_Heat_85"] = (out["Temperature"] >= 85.0).astype(int)
    out["Extreme_Heat_90"] = (out["Temperature"] >= 90.0).astype(int)
    out["Extreme_Heat_95"] = (out["Temperature"] >= 95.0).astype(int)
    out["Extreme_Heat_100"] = (out["Temperature"] >= 100.0).astype(int)

    out["Temp_Bin"] = pd.cut(out["Temperature"], bins=TEMP_BIN_EDGES, labels=False, include_lowest=True).astype("float")

    out["Date"] = out["DT"].dt.date
    out["Temperature_DailyMax"] = out.groupby("Date")["Temperature"].transform("max")
    out["Temperature_DailyMin"] = out.groupby("Date")["Temperature"].transform("min")
    out["Temperature_DailyMean"] = out.groupby("Date")["Temperature"].transform("mean")
    out["Daily_CDD"] = (out["Temperature_DailyMean"] - BASE_TEMP).clip(lower=0.0)
    out["Daily_HDD"] = (BASE_TEMP - out["Temperature_DailyMean"]).clip(lower=0.0)
    out["DailyMaxTempBin"] = pd.cut(out["Temperature_DailyMax"], bins=DAILY_MAX_BIN_EDGES, labels=False, include_lowest=True).astype("float")

    out["HeatIndexF"] = _heat_index_f(out["Temperature"], out["HumidityPct"].fillna(0.0)).astype(float)
    out["HeatIndex_CDD"] = (out["HeatIndexF"] - BASE_TEMP).clip(lower=0.0)
    out["Cooling_Stress"] = out["CDD"] * out["IsLikelySystemPeakHour"]
    out["DailyMax_x_PeakHour"] = out["Temperature_DailyMax"] * out["IsLikelySystemPeakHour"]
    peak_window_14_18 = out["Hour"].between(14, 18).astype(float)
    hot_peak_16_20 = (out["Hour"].between(16, 20) & out["Temperature_DailyMax"].ge(90.0)).astype(float)
    non_business_hot_peak = (
        hot_peak_16_20.eq(1.0)
        & pd.to_numeric(out.get("IsLikelySystemPeakHour", 0), errors="coerce").fillna(0).eq(0)
    ).astype(float)
    daily_max_excess_90 = (out["Temperature_DailyMax"] - 90.0).clip(lower=0.0)
    daily_max_excess_95 = (out["Temperature_DailyMax"] - 95.0).clip(lower=0.0)

    # Scorecard-aligned heat interactions. The existing IsLikelySystemPeakHour
    # excludes weekends/holidays and starts at HE16; the production gates score
    # HE14-18 peak windows and HE16-20 hot days regardless of business-day status.
    out["IsPeakWindow14to18"] = peak_window_14_18
    out["IsHotPeakWindow16to20"] = hot_peak_16_20
    out["DailyMaxTempExcess90"] = daily_max_excess_90
    out["DailyMaxTempExcess95"] = daily_max_excess_95
    out["DailyMax_x_PeakWindow14to18"] = out["Temperature_DailyMax"] * peak_window_14_18
    out["CDD_x_PeakWindow14to18"] = out["CDD"] * peak_window_14_18
    out["CDD_x_HotPeakWindow16to20"] = out["CDD"] * hot_peak_16_20
    out["DailyMaxExcess90_x_PeakWindow14to18"] = daily_max_excess_90 * peak_window_14_18
    out["DailyMaxExcess90_x_HotPeakWindow16to20"] = daily_max_excess_90 * hot_peak_16_20
    out["DailyMaxExcess95_x_HotPeakWindow16to20"] = daily_max_excess_95 * hot_peak_16_20
    out["HeatIndexCDD_x_HotPeakWindow16to20"] = out["HeatIndex_CDD"] * hot_peak_16_20
    out["NonBusinessHotPeakWindow16to20"] = non_business_hot_peak
    out["DailyMaxExcess90_x_NonBusinessHotPeak"] = daily_max_excess_90 * non_business_hot_peak
    for h in range(14, 21):
        out[f"DailyMaxExcess90_x_HE{h}"] = daily_max_excess_90 * out["Hour"].eq(h).astype(float)
    # Weather interactions that help capture cloudy/humid/rainy load shape shifts.
    out["Humidity_x_Temp"] = out["Humidity_Norm"] * out["Temperature"]
    out["Wind_x_Temp"] = out["WindSpeed_Mph"] * out["Temperature"]
    out["Cloud_x_GHI"] = out["CloudCover_Norm"] * out["GHI_Wm2"]
    out["Rain_x_IsWeekend"] = out["Is_Raining"] * out["IsWeekend"]
    out["Hot_Humid_Stress"] = out["CDD"] * out["Humidity_Norm"]
    out = add_heat_persistence_features(out)
    return out
