from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from forecasting.features.weather_features import add_delta_breeze_weather_shape_features, add_heat_persistence_features


def _cfg(config: dict | None) -> dict:
    return ((config or {}).get("bands", {}) or {}).get("weather_scenarios", {}) or {}


def scenario_definitions(config: dict | None) -> list[dict[str, Any]]:
    cfg = _cfg(config)
    if not bool(cfg.get("enabled", True)):
        return []
    return list(
        cfg.get(
            "scenarios",
            [
                {"name": "warmer", "temperature_delta_f": 3.0},
                {"name": "cooler", "temperature_delta_f": -3.0},
                {"name": "cloudier_solar_loss", "cloud_cover_delta_norm": 0.30, "ghi_multiplier": 0.65},
                {"name": "clearer_high_solar", "cloud_cover_delta_norm": -0.30, "ghi_multiplier": 1.15},
            ],
        )
        or []
    )


def _daily_max_bin(values: pd.Series) -> pd.Series:
    return pd.cut(
        pd.to_numeric(values, errors="coerce"),
        bins=[-999, 65, 75, 85, 90, 95, 100, 105, 999],
        labels=False,
        include_lowest=True,
    ).astype(float)


def _recompute_weather_temperature(out: pd.DataFrame) -> pd.DataFrame:
    temp = pd.to_numeric(out.get("Temperature"), errors="coerce")
    humidity = pd.to_numeric(out.get("Humidity_Norm"), errors="coerce").fillna(0.0)
    out["CDD"] = (temp - 65.0).clip(lower=0.0)
    out["HDD"] = (65.0 - temp).clip(lower=0.0)
    out["Temp_Squared"] = temp ** 2
    out["CDD_Squared"] = out["CDD"] ** 2
    out["HDD_Squared"] = out["HDD"] ** 2
    for threshold in [80, 85, 90, 95, 100]:
        out[f"Extreme_Heat_{threshold}"] = temp.ge(float(threshold)).astype(int)
    out["Temp_Bin"] = pd.cut(
        temp,
        bins=[-999, 45, 55, 65, 75, 80, 85, 90, 95, 100, 999],
        labels=False,
        include_lowest=True,
    ).astype(float)
    if "Date" not in out.columns:
        out["Date"] = pd.to_datetime(out["DT"]).dt.date
    out["Temperature_DailyMax"] = out.groupby("Date")["Temperature"].transform("max")
    out["Temperature_DailyMin"] = out.groupby("Date")["Temperature"].transform("min")
    out["Temperature_DailyMean"] = out.groupby("Date")["Temperature"].transform("mean")
    out["Daily_CDD"] = (out["Temperature_DailyMean"] - 65.0).clip(lower=0.0)
    out["Daily_HDD"] = (65.0 - out["Temperature_DailyMean"]).clip(lower=0.0)
    out["DailyMaxTempBin"] = _daily_max_bin(out["Temperature_DailyMax"])
    out["HeatIndexF"] = temp
    out["HeatIndex_CDD"] = (out["HeatIndexF"] - 65.0).clip(lower=0.0)
    if "IsLikelySystemPeakHour" in out.columns:
        out["Cooling_Stress"] = out["CDD"] * pd.to_numeric(out["IsLikelySystemPeakHour"], errors="coerce").fillna(0.0)
        out["DailyMax_x_PeakHour"] = out["Temperature_DailyMax"] * pd.to_numeric(out["IsLikelySystemPeakHour"], errors="coerce").fillna(0.0)
    out["Humidity_x_Temp"] = humidity * temp
    if "WindSpeed_Mph" in out.columns:
        out["Wind_x_Temp"] = pd.to_numeric(out["WindSpeed_Mph"], errors="coerce").fillna(0.0) * temp
    out = add_delta_breeze_weather_shape_features(out)
    out = add_heat_persistence_features(out)
    return out


def _recompute_solar_cloud(out: pd.DataFrame) -> pd.DataFrame:
    cloud = pd.to_numeric(out.get("CloudCover_Norm"), errors="coerce").fillna(0.0).clip(0.0, 1.0)
    ghi = pd.to_numeric(out.get("GHI_Wm2", out.get("Solar_Irradiance")), errors="coerce").fillna(0.0).clip(lower=0.0)
    if "GHI_Wm2" in out.columns:
        out["GHI_Wm2"] = ghi
    out["CloudCover_Norm"] = cloud
    if "CloudCoverPct" in out.columns:
        out["CloudCoverPct"] = cloud * 100.0
    out["Solar_Irradiance"] = ghi
    if "Solar_Hour_Shape" in out.columns and "Impact_Cap_MW" in out.columns:
        out["BTM_Solar_Proxy_MW"] = (
            pd.to_numeric(out["Impact_Cap_MW"], errors="coerce").fillna(0.0)
            * (ghi / 950.0)
            * pd.to_numeric(out["Solar_Hour_Shape"], errors="coerce").fillna(0.0)
        ).clip(lower=0.0)
    if "BTM_ClearSky_Proxy_MW" in out.columns:
        out["BTM_Solar_Cloud_Adjusted_MW"] = pd.to_numeric(out.get("BTM_Solar_Proxy_MW"), errors="coerce").fillna(0.0)
        out["BTM_Solar_Loss_From_ClearSky_MW"] = (
            pd.to_numeric(out["BTM_ClearSky_Proxy_MW"], errors="coerce").fillna(0.0)
            - out["BTM_Solar_Cloud_Adjusted_MW"]
        ).clip(lower=0.0)
    if "ClearSky_GHI_Proxy_Wm2" in out.columns:
        denom = pd.to_numeric(out["ClearSky_GHI_Proxy_Wm2"], errors="coerce").replace(0.0, np.nan)
        out["ClearSky_Index"] = (ghi / denom).replace([np.inf, -np.inf], np.nan).clip(0.0, 1.35).fillna(0.0)
        out["Cloud_x_ClearSky_GHI"] = cloud * pd.to_numeric(out["ClearSky_GHI_Proxy_Wm2"], errors="coerce").fillna(0.0)
    if "Solar_Hour_Shape" in out.columns:
        out["Cloud_x_Solar_Hour"] = cloud * pd.to_numeric(out["Solar_Hour_Shape"], errors="coerce").fillna(0.0)
    out["Cloud_x_GHI"] = cloud * ghi
    if "BTM_Solar_Proxy_MW" in out.columns:
        out["BTM_x_GHI"] = pd.to_numeric(out["BTM_Solar_Proxy_MW"], errors="coerce").fillna(0.0) * ghi
        out["BTM_x_Cloud"] = pd.to_numeric(out["BTM_Solar_Proxy_MW"], errors="coerce").fillna(0.0) * cloud
        if "Solar_Midday_Flag" in out.columns:
            out["BTM_Midday_Impact"] = out["BTM_Solar_Proxy_MW"] * pd.to_numeric(out["Solar_Midday_Flag"], errors="coerce").fillna(0.0)
        if "Solar_Evening_Ramp_Flag" in out.columns:
            out["BTM_Evening_Ramp_Impact"] = out["BTM_Solar_Proxy_MW"] * pd.to_numeric(out["Solar_Evening_Ramp_Flag"], errors="coerce").fillna(0.0)
    if "Date" not in out.columns:
        out["Date"] = pd.to_datetime(out["DT"]).dt.date
    if "BTM_Solar_Proxy_MW" in out.columns:
        out["Daily_BTM_Solar_Proxy_Total_MWh"] = out.groupby("Date")["BTM_Solar_Proxy_MW"].transform("sum")
        out["Daily_BTM_Solar_Proxy_Max_MW"] = out.groupby("Date")["BTM_Solar_Proxy_MW"].transform("max")
        btm_diff_1 = out["BTM_Solar_Proxy_MW"].diff()
        btm_diff_2 = out["BTM_Solar_Proxy_MW"].diff(2)
        out["Solar_Ramp_Down_1hr"] = (-btm_diff_1).clip(lower=0.0).fillna(0.0)
        out["Solar_Ramp_Down_2hr"] = (-btm_diff_2).clip(lower=0.0).fillna(0.0)
        out["Solar_Ramp_Up_1hr"] = btm_diff_1.clip(lower=0.0).fillna(0.0)
    if "BTM_Solar_Loss_From_ClearSky_MW" in out.columns:
        out["Daily_BTM_Solar_Loss_MWh"] = out.groupby("Date")["BTM_Solar_Loss_From_ClearSky_MW"].transform("sum")
        out["Daily_BTM_Solar_Loss_Max_MW"] = out.groupby("Date")["BTM_Solar_Loss_From_ClearSky_MW"].transform("max")
        midday = pd.to_numeric(out.get("Solar_Midday_Flag", 0), errors="coerce").fillna(0.0)
        out["Midday_Overcast_Solar_Loss_MW"] = out["BTM_Solar_Loss_From_ClearSky_MW"] * midday * cloud.ge(0.60).astype(int)
    return out


def make_weather_scenario_frame(base_future_frame: pd.DataFrame, scenario: dict[str, Any]) -> pd.DataFrame:
    out = base_future_frame.copy().sort_values("DT").reset_index(drop=True)
    temp_delta = float(scenario.get("temperature_delta_f", 0.0) or 0.0)
    if temp_delta:
        for col in ["Temperature", "TempF"]:
            if col in out.columns:
                out[col] = pd.to_numeric(out[col], errors="coerce") + temp_delta
        out = _recompute_weather_temperature(out)

    cloud_delta = float(scenario.get("cloud_cover_delta_norm", 0.0) or 0.0)
    if cloud_delta and "CloudCover_Norm" in out.columns:
        out["CloudCover_Norm"] = (pd.to_numeric(out["CloudCover_Norm"], errors="coerce").fillna(0.0) + cloud_delta).clip(0.0, 1.0)

    ghi_multiplier = float(scenario.get("ghi_multiplier", 1.0) or 1.0)
    if ghi_multiplier != 1.0:
        for col in ["GHI_Wm2", "Solar_Irradiance"]:
            if col in out.columns:
                out[col] = (pd.to_numeric(out[col], errors="coerce").fillna(0.0) * ghi_multiplier).clip(lower=0.0)

    if cloud_delta or ghi_multiplier != 1.0:
        out = _recompute_solar_cloud(out)

    out = add_delta_breeze_weather_shape_features(out)
    out["Weather_Scenario_Name"] = str(scenario.get("name", "scenario"))
    return out


def scenario_column_name(name: str) -> str:
    safe = "".join(ch if ch.isalnum() else "_" for ch in str(name).strip().lower()).strip("_")
    return f"WeatherScenario_{safe}_P50_MWH"


def add_scenario_summary_columns(future_df: pd.DataFrame, scenario_columns: list[str]) -> pd.DataFrame:
    out = future_df.copy()
    cols = [col for col in scenario_columns if col in out.columns]
    if not cols:
        out["WeatherScenario_Spread_MWH"] = 0.0
        out["WeatherScenario_MaxAbsDelta_MWH"] = 0.0
        out["WeatherScenario_Cap_Applied"] = 0
        return out
    scen = out[cols].apply(pd.to_numeric, errors="coerce")
    base_col = "Final_Forecast_MWH" if "Final_Forecast_MWH" in out.columns else "Calibrated_Forecast_MWH"
    base = pd.to_numeric(out.get(base_col, pd.Series(np.nan, index=out.index)), errors="coerce")
    deltas = scen.subtract(base, axis=0)
    out["WeatherScenario_Min_P50_MWH"] = scen.min(axis=1)
    out["WeatherScenario_Max_P50_MWH"] = scen.max(axis=1)
    out["WeatherScenario_Spread_MWH"] = out["WeatherScenario_Max_P50_MWH"] - out["WeatherScenario_Min_P50_MWH"]
    out["WeatherScenario_HalfSpread_MWH"] = out["WeatherScenario_Spread_MWH"] / 2.0
    out["WeatherScenario_MaxAbsDelta_MWH"] = deltas.abs().max(axis=1).fillna(0.0)
    if "WeatherScenario_Cap_Applied" not in out.columns:
        out["WeatherScenario_Cap_Applied"] = 0
    return out


def _forecast_day_index(out: pd.DataFrame) -> pd.Series:
    if "Forecast_Day" in out.columns:
        day = pd.to_numeric(out["Forecast_Day"], errors="coerce")
        if day.notna().any():
            return day.fillna(999).astype(int)
    dt = pd.to_datetime(out.get("DT", pd.Series(pd.NaT, index=out.index)), errors="coerce")
    if dt.dropna().empty:
        return pd.Series(999, index=out.index, dtype=int)
    first_day = dt.min().normalize()
    return ((dt.dt.normalize() - first_day).dt.days + 1).fillna(999).astype(int)


def _horizon_bucket(day: pd.Series) -> pd.Series:
    day_num = pd.to_numeric(day, errors="coerce")
    return pd.Series(
        np.select(
            [
                day_num.eq(1),
                day_num.between(2, 3),
                day_num.between(4, 7),
                day_num.between(8, 16),
            ],
            ["Day1", "Days2to3", "Days4to7", "Days8to16"],
            default="Informational",
        ),
        index=day.index,
        dtype="object",
    )


def _scenario_delta_cap_series(df: pd.DataFrame, config: dict | None) -> pd.Series:
    cfg = _cfg(config)
    cap_cfg = cfg.get("delta_caps", {}) or {}
    default_cap = float(cap_cfg.get("default_mwh", cfg.get("scenario_delta_cap_mwh", 18.0)))
    hot_peak_cap = float(cap_cfg.get("hot_peak_mwh", default_cap))
    cloud_solar_cap = float(cap_cfg.get("cloud_solar_mwh", default_cap))
    long_horizon_cap = float(cap_cfg.get("days8to16_mwh", default_cap))

    cap = pd.Series(default_cap, index=df.index, dtype=float)
    hour = pd.to_numeric(df.get("Hour", pd.Series(np.nan, index=df.index)), errors="coerce")
    temp_max = pd.to_numeric(df.get("Temperature_DailyMax", pd.Series(np.nan, index=df.index)), errors="coerce")
    cloud = pd.to_numeric(df.get("CloudCover_Norm", pd.Series(np.nan, index=df.index)), errors="coerce")
    loss = pd.to_numeric(df.get("BTM_Solar_Loss_From_ClearSky_MW", pd.Series(np.nan, index=df.index)), errors="coerce")
    day = _forecast_day_index(df)

    hot_peak = hour.between(16, 20) & temp_max.ge(float(cap_cfg.get("hot_peak_min_maxtemp_f", 90.0)))
    cloud_solar = hour.between(10, 16) & (cloud.ge(0.60) | loss.ge(float(cap_cfg.get("cloud_solar_min_loss_mw", 1.25))))
    long_horizon = day.between(8, 16)
    cap.loc[hot_peak] = hot_peak_cap
    cap.loc[cloud_solar] = cloud_solar_cap
    cap.loc[long_horizon] = np.minimum(cap.loc[long_horizon], long_horizon_cap)
    return cap.clip(lower=0.0)


def apply_weather_scenario_delta_caps(
    future_df: pd.DataFrame,
    scenario_columns: list[str],
    config: dict | None,
    base_col: str | None = None,
) -> pd.DataFrame:
    """Bound scenario P50 deltas so stress tests inform risk bands without dominating them."""
    out = future_df.copy()
    cfg = _cfg(config)
    cols = [col for col in scenario_columns if col in out.columns]
    if not bool(cfg.get("cap_scenario_deltas", True)) or not cols:
        out["WeatherScenario_Cap_Applied"] = 0
        return out

    base_name = base_col or ("Final_Forecast_MWH" if "Final_Forecast_MWH" in out.columns else "Calibrated_Forecast_MWH")
    if base_name not in out.columns:
        out["WeatherScenario_Cap_Applied"] = 0
        return out
    base = pd.to_numeric(out[base_name], errors="coerce")
    cap = _scenario_delta_cap_series(out, config)
    any_capped = pd.Series(False, index=out.index, dtype=bool)
    for col in cols:
        scenario = pd.to_numeric(out[col], errors="coerce")
        delta = scenario - base
        capped_delta = delta.clip(lower=-cap, upper=cap)
        any_capped |= delta.notna() & capped_delta.notna() & (delta.sub(capped_delta).abs() > 1e-9)
        out[col] = (base + capped_delta).clip(lower=0.0)
    out["WeatherScenario_Cap_Applied"] = any_capped.astype(int)
    return out


def build_weather_stress_summary(
    future_df: pd.DataFrame,
    scenario_columns: list[str],
    base_col: str | None = None,
) -> pd.DataFrame:
    out = future_df.copy()
    cols = [col for col in scenario_columns if col in out.columns]
    base_name = base_col or ("Final_Forecast_MWH" if "Final_Forecast_MWH" in out.columns else "Calibrated_Forecast_MWH")
    if not cols or base_name not in out.columns:
        return pd.DataFrame()

    base = pd.to_numeric(out[base_name], errors="coerce")
    day = _forecast_day_index(out)
    hour = pd.to_numeric(out.get("Hour", pd.Series(np.nan, index=out.index)), errors="coerce")
    temp_max = pd.to_numeric(out.get("Temperature_DailyMax", pd.Series(np.nan, index=out.index)), errors="coerce")
    cloud = pd.to_numeric(out.get("CloudCover_Norm", pd.Series(np.nan, index=out.index)), errors="coerce")
    loss = pd.to_numeric(out.get("BTM_Solar_Loss_From_ClearSky_MW", pd.Series(np.nan, index=out.index)), errors="coerce")

    slices: list[tuple[str, pd.Series]] = [
        ("Overall", pd.Series(True, index=out.index, dtype=bool)),
        ("PeakWindow_HE14to18", hour.between(14, 18)),
        ("HotPeak_HE16to20_90FPlus", hour.between(16, 20) & temp_max.ge(90.0)),
        ("CloudSolarMidday", hour.between(10, 16) & (cloud.ge(0.60) | loss.ge(1.25))),
        ("LongHorizon_Days8to16", day.between(8, 16)),
    ]
    horizon = _horizon_bucket(day)
    for label in ["Day1", "Days2to3", "Days4to7", "Days8to16"]:
        slices.append((f"Horizon_{label}", horizon.eq(label)))

    rows: list[dict[str, Any]] = []
    cap_applied = pd.to_numeric(out.get("WeatherScenario_Cap_Applied", pd.Series(0, index=out.index)), errors="coerce").fillna(0).gt(0)
    for col in cols:
        scenario = col.removeprefix("WeatherScenario_").removesuffix("_P50_MWH")
        delta = pd.to_numeric(out[col], errors="coerce") - base
        for slice_name, mask in slices:
            valid = mask & delta.notna()
            if not valid.any():
                continue
            values = delta.loc[valid]
            rows.append(
                {
                    "Scenario": scenario,
                    "Slice": slice_name,
                    "N": int(valid.sum()),
                    "MeanDelta_MWH": float(values.mean()),
                    "MeanAbsDelta_MWH": float(values.abs().mean()),
                    "P90AbsDelta_MWH": float(values.abs().quantile(0.90)),
                    "MaxAbsDelta_MWH": float(values.abs().max()),
                    "CapHitRows": int((valid & cap_applied).sum()),
                }
            )
    return pd.DataFrame(rows)


def latest_validation_detail_path(output_dir: str | Path) -> Path | None:
    path = Path(output_dir)
    files = sorted(path.glob("weather_interval_coverage_validation_detail_*.csv"), key=lambda p: p.stat().st_mtime, reverse=True)
    production_files = [p for p in files if "smoke" not in p.stem.lower()]
    return (production_files or files)[0] if files else None


def build_conformal_lookup(detail_path: str | Path, quantile: float = 0.80) -> dict[str, Any]:
    path = Path(detail_path)
    if not path.exists():
        return {}
    detail = pd.read_csv(path, low_memory=False)
    if "Policy" in detail.columns:
        detail = detail[detail["Policy"].astype(str).eq("current_weather_risk")].copy()
    if detail.empty or "AbsError_MWH" not in detail.columns:
        return {}
    detail["AbsError_MWH"] = pd.to_numeric(detail["AbsError_MWH"], errors="coerce")
    detail = detail.dropna(subset=["AbsError_MWH"])
    if detail.empty:
        return {}

    lookup: dict[str, Any] = {
        "source_path": str(path),
        "quantile": float(quantile),
        "global": float(detail["AbsError_MWH"].quantile(float(quantile))),
        "risk_class": {},
        "horizon": {},
        "risk_horizon": {},
    }
    if "Weather_Input_Risk_Class" in detail.columns:
        lookup["risk_class"] = detail.groupby("Weather_Input_Risk_Class", dropna=False)["AbsError_MWH"].quantile(float(quantile)).astype(float).to_dict()
    if "Replay_Horizon_Bucket" in detail.columns:
        lookup["horizon"] = detail.groupby("Replay_Horizon_Bucket", dropna=False)["AbsError_MWH"].quantile(float(quantile)).astype(float).to_dict()
    if {"Weather_Input_Risk_Class", "Replay_Horizon_Bucket"}.issubset(detail.columns):
        rh = detail.groupby(["Weather_Input_Risk_Class", "Replay_Horizon_Bucket"], dropna=False)["AbsError_MWH"].quantile(float(quantile))
        lookup["risk_horizon"] = {"|".join(str(part) for part in key): float(value) for key, value in rh.items()}
    return lookup


def apply_conformal_weather_bands(df: pd.DataFrame, config: dict | None, output_dir: str | Path) -> pd.DataFrame:
    cfg = (((config or {}).get("bands", {}) or {}).get("conformal_weather", {}) or {})
    out = df.copy()
    out["Pre_Conformal_Band_MWH"] = pd.to_numeric(out.get("Band", np.nan), errors="coerce")
    if not bool(cfg.get("enabled", True)):
        out["Conformal_Weather_Band_MWH"] = np.nan
        out["Conformal_Weather_Source"] = "disabled"
        return out

    detail_path = latest_validation_detail_path(output_dir)
    if detail_path is None:
        out["Conformal_Weather_Band_MWH"] = np.nan
        out["Conformal_Weather_Source"] = "missing_validation_detail"
        return out

    lookup = build_conformal_lookup(detail_path, quantile=float(cfg.get("quantile", 0.80)))
    if not lookup:
        out["Conformal_Weather_Band_MWH"] = np.nan
        out["Conformal_Weather_Source"] = "empty_lookup"
        return out

    risk = out.get("Weather_Input_Risk_Class", pd.Series("none", index=out.index)).astype(str)
    horizon = out.get("Replay_Horizon_Bucket", pd.Series("", index=out.index)).astype(str)
    if "Forecast_Day" in out.columns:
        day = pd.to_numeric(out["Forecast_Day"], errors="coerce")
        horizon = pd.Series("Days8to16", index=out.index, dtype="object")
        horizon.loc[day.eq(1)] = "Day1"
        horizon.loc[day.between(2, 7)] = "Days2to7"

    conformal = pd.Series(float(lookup.get("global", np.nan)), index=out.index, dtype=float)
    source = pd.Series("global", index=out.index, dtype="object")
    for idx in out.index:
        key = f"{risk.loc[idx]}|{horizon.loc[idx]}"
        if key in lookup.get("risk_horizon", {}):
            conformal.loc[idx] = float(lookup["risk_horizon"][key])
            source.loc[idx] = "risk_horizon"
        elif risk.loc[idx] in lookup.get("risk_class", {}):
            conformal.loc[idx] = float(lookup["risk_class"][risk.loc[idx]])
            source.loc[idx] = "risk_class"
        elif horizon.loc[idx] in lookup.get("horizon", {}):
            conformal.loc[idx] = float(lookup["horizon"][horizon.loc[idx]])
            source.loc[idx] = "horizon"

    base_band = out["Pre_Conformal_Band_MWH"] / pd.to_numeric(out.get("Weather_Input_Risk_Multiplier", 1.0), errors="coerce").replace(0.0, np.nan).fillna(1.0)
    scenario_values = out.get("WeatherScenario_HalfSpread_MWH")
    if scenario_values is None:
        scenario_band = pd.Series(0.0, index=out.index, dtype=float)
    else:
        scenario_band = pd.to_numeric(scenario_values, errors="coerce").fillna(0.0)
    scenario_band = scenario_band * float(cfg.get("scenario_spread_multiplier", 1.0))
    safety = float(cfg.get("safety_multiplier", 1.0))
    safety_series = pd.Series(safety, index=out.index, dtype=float)
    horizon_multipliers = cfg.get("horizon_safety_multipliers", {}) or {}
    for horizon_name, multiplier in horizon_multipliers.items():
        safety_series.loc[horizon.eq(str(horizon_name))] *= float(multiplier)
    normal_multiplier = float(cfg.get("normal_only_safety_multiplier", 1.0))
    caution_reason = out.get("Production_Caution_Reason", pd.Series("none", index=out.index)).astype(str)
    high_risk = (
        caution_reason.str.contains("hot_peak|peak_window|cloudy_solar|days8to16", case=False, regex=True, na=False)
        | risk.str.contains("hot_peak|cloudy_solar|shoulder_heat|high_temp|days8to16", case=False, regex=True, na=False)
    )
    normal = ~high_risk
    safety_series.loc[normal] *= normal_multiplier
    safety_series.loc[high_risk] = safety
    candidate = np.maximum.reduce([
        base_band.to_numpy(dtype=float),
        (conformal * safety_series).to_numpy(dtype=float),
        scenario_band.to_numpy(dtype=float),
    ])
    current = out["Pre_Conformal_Band_MWH"].to_numpy(dtype=float)
    eligible = (
        pd.to_numeric(out.get("Weather_Input_Risk_Multiplier", 1.0), errors="coerce").fillna(1.0).gt(1.0)
        | ~risk.eq("none")
    ).to_numpy()
    candidate = np.where(eligible, candidate, current)
    min_fraction = float(cfg.get("min_fraction_of_multiplier_band", 0.0))
    if min_fraction > 0:
        candidate = np.maximum(candidate, current * min_fraction)
    if bool(cfg.get("allow_narrowing", True)):
        out["Band"] = candidate
    else:
        out["Band"] = np.maximum(current, candidate)
    out["Conformal_Weather_Band_MWH"] = conformal
    out["Conformal_Weather_Source"] = source
    base = pd.to_numeric(out["Calibrated_Forecast_MWH"], errors="coerce")
    out["Upper_Band"] = base + out["Band"].astype(float)
    out["Lower_Band"] = np.maximum(0.0, base - out["Band"].astype(float))
    out["P10_Forecast_MWH"] = out["Lower_Band"]
    out["P50_Forecast_MWH"] = base.clip(lower=0.0)
    out["P90_Forecast_MWH"] = out["Upper_Band"]
    out["Forecast_Low_MWH"] = out["P10_Forecast_MWH"]
    out["Forecast_Expected_MWH"] = out["P50_Forecast_MWH"]
    out["Forecast_High_MWH"] = out["P90_Forecast_MWH"]
    out["Quantile_Method"] = out.get("Quantile_Method", "conditional_residual_central80").astype(str) + "+conformal_weather"
    return out
