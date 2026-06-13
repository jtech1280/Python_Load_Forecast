from __future__ import annotations

import numpy as np
import pandas as pd


def _local_datetime_series(values) -> pd.Series:
    try:
        return pd.to_datetime(values, errors="coerce")
    except ValueError:
        cleaned = pd.Series(values).astype(str).str.strip().str.replace(r"(?:[+-]\d{2}:?\d{2}|Z)$", "", regex=True)
        return pd.to_datetime(cleaned, errors="coerce")


def _hour_group(hour: int) -> str:
    if 0 <= int(hour) <= 5:
        return "Overnight"
    if 6 <= int(hour) <= 9:
        return "Morning"
    if 10 <= int(hour) <= 15:
        return "Midday"
    if 16 <= int(hour) <= 20:
        return "Peak"
    return "LateEvening"


def _bucket_cloud(values: pd.Series) -> pd.Series:
    cloud = pd.to_numeric(values, errors="coerce")
    if cloud.dropna().empty:
        return pd.Series(np.nan, index=values.index, dtype="object")
    if cloud.max(skipna=True) <= 1.5:
        bins = [-0.001, .20, .40, .60, .80, 1.001]
    else:
        bins = [-0.001, 20, 40, 60, 80, 100.001]
    return pd.cut(cloud, bins=bins, labels=["Clear/Low", "Some Clouds", "Partly Cloudy", "Mostly Cloudy", "Overcast"], include_lowest=True).astype("object")


def _bucket_loss(values: pd.Series) -> pd.Series:
    loss = pd.to_numeric(values, errors="coerce").fillna(0.0)
    out = pd.Series("None", index=loss.index, dtype="object")
    positive = loss[loss > 0]
    if len(positive) >= 10 and positive.nunique() >= 4:
        q1, q2, q3 = positive.quantile([0.25, 0.50, 0.75]).tolist()
        out.loc[(loss > 0) & (loss <= q1)] = "Low"
        out.loc[(loss > q1) & (loss <= q2)] = "Medium"
        out.loc[(loss > q2) & (loss <= q3)] = "High"
        out.loc[loss > q3] = "Extreme"
    elif len(positive):
        med = positive.median()
        out.loc[(loss > 0) & (loss <= med)] = "Low/Medium"
        out.loc[loss > med] = "High"
    return out


def _prep(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    out["DT"] = _local_datetime_series(out["DT"])
    if "Hour" not in out.columns:
        out["Hour"] = out["DT"].dt.hour
    if "HourGroup" not in out.columns:
        out["HourGroup"] = out["Hour"].map(_hour_group)
    if "DailyMaxTempBin" not in out.columns:
        if "DailyMaxTempBucket" in out.columns:
            out["DailyMaxTempBin"] = out["DailyMaxTempBucket"]
        elif "Temperature_DailyMax" in out.columns:
            temp_max = pd.to_numeric(out["Temperature_DailyMax"], errors="coerce")
            out["DailyMaxTempBin"] = pd.cut(
                temp_max,
                bins=[-np.inf, 70.0, 80.0, 88.0, 94.0, 98.0, np.inf],
                labels=[0, 1, 2, 3, 4, 5],
                include_lowest=True,
            ).astype("float")
        else:
            out["DailyMaxTempBin"] = np.nan
    if "CloudCoverBucket" not in out.columns and "CloudCover_Norm" in out.columns:
        out["CloudCoverBucket"] = _bucket_cloud(out["CloudCover_Norm"])
    if "SolarLossBucket" not in out.columns:
        loss_col = "BTM_Solar_Loss_From_ClearSky_MW" if "BTM_Solar_Loss_From_ClearSky_MW" in out.columns else "Midday_Overcast_Solar_Loss_MW"
        if loss_col in out.columns:
            out["SolarLossBucket"] = _bucket_loss(out[loss_col])
    return out


def build_residual_band_lookup(backtest_df: pd.DataFrame, shrink_floor_mwh: float = 4.0) -> dict:
    if backtest_df is None or backtest_df.empty:
        return {}
    work = _prep(backtest_df)
    if "Residual_MWH" in work.columns:
        residual = pd.to_numeric(work["Residual_MWH"], errors="coerce")
    else:
        residual = pd.to_numeric(work["Actual_MWH"], errors="coerce") - pd.to_numeric(work["Raw_Forecast_MWH"], errors="coerce")
    work["AbsResidual"] = residual.abs()
    work = work.dropna(subset=["AbsResidual"])
    if work.empty:
        return {}

    levels = []
    for keys in [
        ["Hour", "CloudCoverBucket", "SolarLossBucket"],
        ["Hour", "SolarLossBucket"],
        ["HourGroup", "SolarLossBucket"],
        ["HourGroup", "DailyMaxTempBin"],
        ["HourGroup"],
        ["Hour"],
    ]:
        if not all(k in work.columns for k in keys):
            continue
        grp = work.groupby(keys, dropna=False)["AbsResidual"].agg(
            p50=lambda s: float(np.nanquantile(s, 0.50)),
            p80=lambda s: float(np.nanquantile(s, 0.80)),
            p90=lambda s: float(np.nanquantile(s, 0.90)),
            mean="mean",
            count="count",
        ).reset_index()
        # V12.7 uses conditional residual bands as the primary source instead of forcing a broad
        # percentage band to dominate every hour.  Higher-risk buckets can still be widened below.
        grp["band_mwh"] = np.maximum(float(shrink_floor_mwh), grp["p80"].astype(float))
        levels.append({"keys": keys, "lookup": grp[keys + ["band_mwh", "count", "p80", "p90", "mean"]]})

    return {
        "ordered_levels": levels,
        "global_band_mwh": max(float(shrink_floor_mwh), float(work["AbsResidual"].quantile(0.80))),
        "global_p90_band_mwh": max(float(shrink_floor_mwh), float(work["AbsResidual"].quantile(0.90))),
    }


def _band_risk_multiplier(out: pd.DataFrame) -> pd.Series:
    hour = pd.to_numeric(out.get("Hour", pd.Series(np.nan, index=out.index)), errors="coerce").fillna(-1).astype(int)
    hg = out.get("HourGroup", pd.Series("", index=out.index)).astype(str)
    temp_max = pd.to_numeric(out.get("Temperature_DailyMax", pd.Series(np.nan, index=out.index)), errors="coerce")
    cloud = out.get("CloudCoverBucket", pd.Series("", index=out.index)).astype(str)
    loss = out.get("SolarLossBucket", pd.Series("", index=out.index)).astype(str)
    event_cls = out.get("CloudSolarEventClass", pd.Series("", index=out.index)).astype(str)
    mult = pd.Series(1.0, index=out.index, dtype=float)
    mult.loc[hg.isin(["Overnight", "Morning"])] *= 0.78

    # Keep targeted widening for volatile midday/peak regimes. These hours are where solar/cloud
    # and peak-load uncertainty matter operationally, so they should not be constrained by the
    # quieter overnight residual distribution.
    mult.loc[hour.between(12, 13)] *= 1.00
    mult.loc[hour.eq(14)] *= 1.35
    mult.loc[hour.eq(15)] *= 1.45
    mult.loc[hour.eq(16)] *= 1.55
    mult.loc[hour.eq(17)] *= 1.45
    mult.loc[hour.eq(18)] *= 1.25
    mult.loc[hg.eq("Peak")] *= 1.12
    mult.loc[cloud.isin(["Mostly Cloudy", "Overcast"]) & loss.isin(["High", "Extreme"])] *= 1.35
    mult.loc[event_cls.isin(["weekday_core_highimpact_solar_loss", "weekday_core_solar_loss", "weekday_core_hour14_solar_loss"])] *= 1.20
    mult.loc[hour.between(16, 18) & cloud.isin(["Mostly Cloudy", "Overcast"])] *= 1.15

    # The 2026-06-12 diagnostic run exposed severe undercoverage in the hottest
    # morning/midday bucket. Do not let the normal morning dampener suppress bands
    # when the forecasted daily max is already in the extreme heat regime.
    hot = temp_max.ge(95.0)
    extreme = temp_max.ge(105.0)
    ultra = temp_max.ge(112.0)
    mult.loc[hot & hg.eq("Overnight")] *= 1.55
    mult.loc[hot & hg.eq("Morning")] *= 3.25
    mult.loc[hot & hg.eq("Midday")] *= 2.05
    mult.loc[hot & hg.eq("Peak")] *= 1.25
    mult.loc[extreme & hg.eq("Morning")] *= 1.25
    mult.loc[extreme & hg.eq("Midday")] *= 1.20
    mult.loc[ultra & hg.isin(["Morning", "Midday", "Peak"])] *= 1.10
    return mult.clip(0.65, 5.00)


def _forecast_day_index(out: pd.DataFrame) -> pd.Series:
    if "Forecast_Day" in out.columns:
        day = pd.to_numeric(out["Forecast_Day"], errors="coerce")
        if day.notna().any():
            return day.fillna(999).astype(int)
    dt = pd.to_datetime(out["DT"], errors="coerce")
    if dt.dropna().empty:
        return pd.Series(999, index=out.index, dtype=int)
    first_day = dt.min().normalize()
    return ((dt.dt.normalize() - first_day).dt.days + 1).fillna(999).astype(int)


def _append_reason(reason: pd.Series, mask: pd.Series, token: str) -> pd.Series:
    if not mask.any():
        return reason
    empty = mask & reason.eq("none")
    reason.loc[empty] = token
    add = mask & ~reason.eq(token) & ~reason.eq("none") & ~reason.astype(str).str.contains(token, regex=False, na=False)
    reason.loc[add] = reason.loc[add].astype(str) + "+" + token
    return reason


def _weather_input_risk_multiplier(out: pd.DataFrame, cfg: dict | None) -> tuple[pd.Series, pd.Series]:
    cfg = cfg or {}
    mult = pd.Series(1.0, index=out.index, dtype=float)
    reason = pd.Series("none", index=out.index, dtype="object")
    risk_class = pd.Series("none", index=out.index, dtype="object")
    if not bool(cfg.get("enabled", False)):
        out["Weather_Input_Risk_Class"] = risk_class
        return mult, reason

    day = _forecast_day_index(out)
    max_day = int(cfg.get("max_day", 7))
    in_scope = day.between(1, max_day)
    if not in_scope.any():
        out["Weather_Input_Risk_Class"] = risk_class
        return mult, reason

    d1 = in_scope & day.eq(1)
    d23 = in_scope & day.between(2, 3)
    d47 = in_scope & day.between(4, min(max_day, 7))
    d8p = in_scope & day.between(8, max_day)
    mult.loc[d1] *= float(cfg.get("day1_multiplier", 1.20))
    mult.loc[d23] *= float(cfg.get("days2to3_multiplier", 1.40))
    mult.loc[d47] *= float(cfg.get("days4to7_multiplier", 1.60))
    mult.loc[d8p] *= float(cfg.get("days8to16_multiplier", cfg.get("days4to7_multiplier", 1.60)))
    reason.loc[d1] = "weather_day1"
    reason.loc[d23] = "weather_days2to3"
    reason.loc[d47] = "weather_days4to7"
    reason.loc[d8p] = "weather_days8to16"
    risk_class.loc[d1] = "day1_weather_error"
    risk_class.loc[d23] = "days2to3_weather_error"
    risk_class.loc[d47] = "days4to7_weather_error"
    risk_class.loc[d8p] = "days8to16_weather_error"

    hour = pd.to_numeric(out.get("Hour", pd.Series(np.nan, index=out.index)), errors="coerce").fillna(-1).astype(int)
    temp_max = pd.to_numeric(out.get("Temperature_DailyMax", pd.Series(np.nan, index=out.index)), errors="coerce")
    season = out.get("Season", pd.Series("", index=out.index)).astype(str)
    cloud = out.get("CloudCoverBucket", pd.Series("", index=out.index)).astype(str)
    loss = out.get("SolarLossBucket", pd.Series("", index=out.index)).astype(str)
    event_cls = out.get("CloudSolarEventClass", pd.Series("", index=out.index)).astype(str)

    cloudy_solar = in_scope & hour.between(10, 16) & (
        cloud.isin(["Mostly Cloudy", "Overcast"])
        | loss.isin(["High", "Extreme"])
        | event_cls.str.contains("solar_loss", case=False, na=False)
    )
    hot_peak = in_scope & hour.between(16, 20) & temp_max.ge(90.0)
    shoulder_heat = in_scope & season.isin(["Spring", "Fall"]) & hour.between(12, 22) & temp_max.between(75.0, 93.0)
    high_temp = in_scope & temp_max.ge(float(cfg.get("high_temp_min_maxtemp_f", 95.0)))

    mult.loc[cloudy_solar] *= float(cfg.get("cloudy_solar_multiplier", 1.15))
    mult.loc[hot_peak] *= float(cfg.get("hot_peak_multiplier", 1.10))
    mult.loc[shoulder_heat] *= float(cfg.get("shoulder_heat_multiplier", 1.0))
    mult.loc[high_temp] *= float(cfg.get("high_temp_multiplier", 1.0))
    reason = _append_reason(reason, cloudy_solar, "cloud_solar")
    reason = _append_reason(reason, hot_peak, "hot_peak")
    reason = _append_reason(reason, shoulder_heat, "shoulder_heat")
    reason = _append_reason(reason, high_temp, "high_temp")

    # Assign the most actionable class for operators; the full reason keeps all matched flags.
    risk_class.loc[cloudy_solar] = "cloudy_solar_loss_weather_error"
    risk_class.loc[shoulder_heat] = "shoulder_heat_weather_error"
    risk_class.loc[hot_peak] = "hot_peak_weather_error"
    risk_class.loc[high_temp & ~hot_peak] = "high_temp_weather_error"
    risk_class.loc[(d47 & (cloudy_solar | hot_peak | shoulder_heat | high_temp))] = (
        "days4to7_" + risk_class.loc[(d47 & (cloudy_solar | hot_peak | shoulder_heat | high_temp))].astype(str)
    )
    risk_class.loc[(d8p & (cloudy_solar | hot_peak | shoulder_heat | high_temp))] = (
        "days8to16_" + risk_class.loc[(d8p & (cloudy_solar | hot_peak | shoulder_heat | high_temp))].astype(str)
    )

    cap = float(cfg.get("cap_multiplier", 2.25))
    out["Weather_Input_Risk_Class"] = risk_class
    return mult.clip(lower=1.0, upper=cap), reason


def _apply_production_caution_labels(out: pd.DataFrame, forecast_day: pd.Series) -> pd.DataFrame:
    hour = pd.to_numeric(out.get("Hour", pd.Series(np.nan, index=out.index)), errors="coerce").fillna(-1).astype(int)
    temp_max = pd.to_numeric(out.get("Temperature_DailyMax", pd.Series(np.nan, index=out.index)), errors="coerce")
    risk_class = out.get("Weather_Input_Risk_Class", pd.Series("none", index=out.index)).astype(str)
    risk_mult = pd.to_numeric(out.get("Weather_Input_Risk_Multiplier", pd.Series(1.0, index=out.index)), errors="coerce").fillna(1.0)
    cloud = out.get("CloudCoverBucket", pd.Series("", index=out.index)).astype(str)
    loss = out.get("SolarLossBucket", pd.Series("", index=out.index)).astype(str)
    event_cls = out.get("CloudSolarEventClass", pd.Series("", index=out.index)).astype(str)

    reason = pd.Series("none", index=out.index, dtype="object")
    hot_peak = hour.between(16, 20) & temp_max.ge(90.0)
    peak_window = hour.between(14, 18)
    cloudy_solar = hour.between(10, 16) & (
        cloud.isin(["Mostly Cloudy", "Overcast"])
        | loss.isin(["High", "Extreme"])
        | event_cls.str.contains("solar_loss", case=False, na=False)
    )
    weather_risk = risk_mult.gt(1.0) | ~risk_class.eq("none")
    long_horizon = forecast_day.between(8, 16)
    recent_corr = pd.to_numeric(
        out.get("Recent_Level_Correction_MWH", pd.Series(0.0, index=out.index)),
        errors="coerce",
    ).fillna(0.0).abs()
    recent_bias_risk = recent_corr.ge(4.0)

    reason = _append_reason(reason, peak_window, "peak_window_data_limited")
    reason = _append_reason(reason, hot_peak, "hot_peak_data_limited")
    reason = _append_reason(reason, cloudy_solar, "cloudy_solar_midday_risk")
    reason = _append_reason(reason, weather_risk, "weather_input_risk")
    reason = _append_reason(reason, long_horizon, "days8to16_low_confidence")
    reason = _append_reason(reason, recent_bias_risk, "recent_bias_risk")

    label = pd.Series("Normal", index=out.index, dtype="object")
    label.loc[weather_risk] = "Weather risk"
    label.loc[cloudy_solar] = "Caution: solar/cloud"
    label.loc[peak_window] = "Caution: peak window"
    label.loc[hot_peak] = "Caution: hot peak"
    label.loc[recent_bias_risk] = "Caution: recent bias"
    label.loc[long_horizon] = "Low confidence"
    label.loc[long_horizon & (peak_window | hot_peak | cloudy_solar | weather_risk)] = "Low confidence caution"

    risk_code = pd.Series("NORMAL", index=out.index, dtype="object")
    risk_code.loc[recent_bias_risk] = "RECENT_BIAS_RISK"
    risk_code.loc[long_horizon] = "LONG_HORIZON_RISK"
    risk_code.loc[weather_risk] = "WEATHER_INPUT_RISK"
    risk_code.loc[peak_window] = "PEAK_WINDOW_RISK"
    risk_code.loc[cloudy_solar] = "SOLAR_CLOUD_RISK"
    risk_code.loc[hot_peak] = "HOT_PEAK_RISK"
    risk_code.loc[long_horizon & (weather_risk | peak_window | hot_peak | cloudy_solar)] = (
        risk_code.loc[long_horizon & (weather_risk | peak_window | hot_peak | cloudy_solar)].astype(str)
        + "+LONG_HORIZON_RISK"
    )

    out["Production_Caution_Flag"] = (~reason.eq("none")).astype(int)
    out["Production_Caution_Reason"] = reason
    out["Production_Confidence_Label"] = label
    out["Production_Risk_Code"] = risk_code
    return out


def apply_bands(
    df: pd.DataFrame,
    percent_band: float,
    floor_mwh: float,
    residual_lookup: dict | None = None,
    band_scale: float = 1.0,
    weather_input_risk: dict | None = None,
) -> pd.DataFrame:
    out = _prep(df)
    base = out["Calibrated_Forecast_MWH"].astype(float)
    scale = max(0.10, float(band_scale or 1.0))
    effective_floor = float(floor_mwh) * scale
    pct_band = np.maximum(float(floor_mwh), np.abs(base) * float(percent_band))
    out["Band_Method"] = "percent_floor"
    out["Band"] = pct_band * scale

    if residual_lookup and residual_lookup.get("ordered_levels"):
        # Start from the conditional global residual band; do not force the percent band to dominate.
        out["Band"] = np.maximum(float(floor_mwh), float(residual_lookup.get("global_band_mwh", floor_mwh)))
        unresolved = pd.Series(True, index=out.index)
        for level in residual_lookup["ordered_levels"]:
            keys = level["keys"]
            lookup = level["lookup"]
            if lookup is None or lookup.empty or not all(k in out.columns for k in keys):
                continue
            tmp = out.loc[unresolved, keys].reset_index().merge(lookup[keys + ["band_mwh"]], on=keys, how="left")
            matched = tmp["band_mwh"].notna()
            if matched.any():
                idx = tmp.loc[matched, "index"]
                out.loc[idx, "Band"] = np.maximum(float(floor_mwh), tmp.loc[matched, "band_mwh"].to_numpy(dtype=float))
                out.loc[idx, "Band_Method"] = "+".join(keys)
                unresolved.loc[idx] = False
        out["Band"] = out["Band"].astype(float) * _band_risk_multiplier(out) * scale
        # Keep an absolute lower bound and a light percent guard for very high-load hours.
        out["Band"] = np.maximum(
            out["Band"].astype(float),
            np.maximum(effective_floor, np.abs(base) * float(percent_band) * 0.65 * scale),
        )

    weather_mult, weather_reason = _weather_input_risk_multiplier(out, weather_input_risk)
    out["Weather_Input_Risk_Multiplier"] = weather_mult
    out["Weather_Input_Risk_Reason"] = weather_reason
    out["Band"] = out["Band"].astype(float) * weather_mult
    forecast_day = _forecast_day_index(out)
    out["Operational_Horizon_Label"] = "Informational"
    out.loc[forecast_day.eq(1), "Operational_Horizon_Label"] = "Day1"
    out.loc[forecast_day.between(2, 3), "Operational_Horizon_Label"] = "Days2to3"
    out.loc[forecast_day.between(4, 7), "Operational_Horizon_Label"] = "Days4to7"
    out.loc[forecast_day.between(8, 16), "Operational_Horizon_Label"] = "Days8to16_low_confidence"
    out = _apply_production_caution_labels(out, forecast_day)

    out["Upper_Band"] = base + out["Band"].astype(float)
    out["Lower_Band"] = np.maximum(0.0, base - out["Band"].astype(float))
    # The conditional band is trained from an 80th percentile absolute residual,
    # so expose it as an operational central 80% P10/P90 interval.
    out["P10_Forecast_MWH"] = out["Lower_Band"]
    out["P50_Forecast_MWH"] = base.clip(lower=0.0)
    out["P90_Forecast_MWH"] = out["Upper_Band"]
    out["Forecast_Low_MWH"] = out["P10_Forecast_MWH"]
    out["Forecast_Expected_MWH"] = out["P50_Forecast_MWH"]
    out["Forecast_High_MWH"] = out["P90_Forecast_MWH"]
    out["Quantile_Method"] = np.where(
        out["Band_Method"].astype(str).eq("percent_floor"),
        "percent_floor_central80",
        "conditional_residual_central80",
    )
    out.loc[weather_mult.gt(1.0), "Quantile_Method"] = out.loc[weather_mult.gt(1.0), "Quantile_Method"].astype(str) + "+weather_input_risk"
    return out
