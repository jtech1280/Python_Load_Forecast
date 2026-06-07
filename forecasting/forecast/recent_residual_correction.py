from __future__ import annotations

import math
from typing import Any

import numpy as np
import pandas as pd


def _as_num(s: pd.Series | Any) -> pd.Series:
    return pd.to_numeric(s, errors="coerce")


def _hour_group(hour: int) -> str:
    h = int(hour)
    if 0 <= h <= 5:
        return "Overnight"
    if 6 <= h <= 9:
        return "Morning"
    if 10 <= h <= 15:
        return "Midday"
    if 16 <= h <= 20:
        return "Peak"
    return "LateEvening"


def _cfg(config: dict | None) -> dict:
    return ((config or {}).get("calibration", {}) or {}).get("recent_residual", {}) or {}


def _season_from_month(month: int) -> str:
    if int(month) in (12, 1, 2):
        return "Winter"
    if int(month) in (3, 4, 5):
        return "Spring"
    if int(month) in (6, 7, 8, 9):
        return "Summer"
    return "Fall"


def _recent_hot_peak_scale(row: pd.Series, recent_cfg: dict) -> float:
    hot_cfg = recent_cfg.get("hot_peak", {}) or {}
    hours = [int(hour) for hour in hot_cfg.get("hours", [16, 17, 18, 19, 20])]
    hour = int(row.get("Hour", pd.to_datetime(row.get("DT")).hour))
    daily_max = pd.to_numeric(pd.Series([row.get("Temperature_DailyMax")]), errors="coerce").iloc[0]
    if hour not in hours or not np.isfinite(daily_max) or daily_max < float(hot_cfg.get("min_maxtemp_f", 90.0)):
        return 1.0
    season = row.get("Season")
    if pd.isna(season):
        season = _season_from_month(pd.to_datetime(row.get("DT")).month)
    return float((hot_cfg.get("season_scales", {}) or {}).get(str(season), hot_cfg.get("default_scale", 1.0)))


def _forecast_day(row: pd.Series, horizon_index: int | None = None) -> int:
    day = pd.to_numeric(pd.Series([row.get("Forecast_Day")]), errors="coerce").iloc[0]
    if np.isfinite(day):
        return max(1, int(day))
    h = max(1, int(horizon_index or 1))
    return int(math.ceil(h / 24.0))


def _recent_horizon_regime_scale(row: pd.Series, recent_cfg: dict, horizon_index: int | None = None) -> float:
    """Dampen recent residual carryover where replay shows unstable transfer by lead/regime."""
    scale_cfg = recent_cfg.get("horizon_regime_scales", {}) or {}
    forecast_day = _forecast_day(row, horizon_index)
    scale = 1.0

    day_scales = scale_cfg.get("forecast_day_scales", {}) or {}
    if 2 <= forecast_day <= 3:
        scale *= float(day_scales.get("days2to3", 1.0))
    elif 4 <= forecast_day <= 7:
        scale *= float(day_scales.get("days4to7", 1.0))
    elif forecast_day >= 8:
        scale *= float(day_scales.get("days8plus", 1.0))

    hot_clear_cfg = scale_cfg.get("summer_clear_hot_days2to7", {}) or {}
    if bool(hot_clear_cfg.get("enabled", False)) and 2 <= forecast_day <= 7:
        season = row.get("Season")
        if pd.isna(season):
            season = _season_from_month(pd.to_datetime(row.get("DT")).month)
        daily_max = pd.to_numeric(pd.Series([row.get("Temperature_DailyMax")]), errors="coerce").iloc[0]
        cloud = pd.to_numeric(pd.Series([row.get("CloudCover_Norm")]), errors="coerce").iloc[0]
        if np.isfinite(cloud) and cloud > 1.5:
            cloud = cloud / 100.0
        min_temp = float(hot_clear_cfg.get("min_maxtemp_f", 90.0))
        max_cloud = float(hot_clear_cfg.get("max_cloud_cover_norm", 0.20))
        if str(season) == "Summer" and np.isfinite(daily_max) and daily_max >= min_temp and np.isfinite(cloud) and cloud <= max_cloud:
            scale *= float(hot_clear_cfg.get("scale", 1.0))

    return float(np.clip(scale, 0.0, 2.0))


def _temp_bucket_from_values(values: pd.Series) -> pd.Series:
    temp = pd.to_numeric(values, errors="coerce")
    return pd.cut(
        temp,
        bins=[-999, 65, 75, 85, 90, 95, 100, 105, 999],
        labels=["<65", "65-75", "75-85", "85-90", "90-95", "95-100", "100-105", "105+"],
        include_lowest=True,
    ).astype("object")


def _cloud_bucket_from_values(values: pd.Series) -> pd.Series:
    cloud = pd.to_numeric(values, errors="coerce")
    if cloud.dropna().empty:
        return pd.Series(np.nan, index=values.index, dtype="object")
    # Open-Meteo may be 0-100 or normalized 0-1 depending on the upstream loader.
    if cloud.max(skipna=True) <= 1.5:
        bins = [-0.001, 0.20, 0.40, 0.60, 0.80, 1.001]
    else:
        bins = [-0.001, 20, 40, 60, 80, 100.001]
    return pd.cut(
        cloud,
        bins=bins,
        labels=["Clear/Low", "Some Clouds", "Partly Cloudy", "Mostly Cloudy", "Overcast"],
        include_lowest=True,
    ).astype("object")


def _solar_bucket_from_values(values: pd.Series) -> pd.Series:
    solar = pd.to_numeric(values, errors="coerce").fillna(0.0)
    out = pd.Series("None", index=values.index, dtype="object")
    positive = solar[solar > 0]
    if len(positive) >= 10 and positive.nunique() >= 4:
        q1, q2, q3 = positive.quantile([0.25, 0.50, 0.75]).tolist()
        out.loc[(solar > 0) & (solar <= q1)] = "Low"
        out.loc[(solar > q1) & (solar <= q2)] = "Medium-Low"
        out.loc[(solar > q2) & (solar <= q3)] = "Medium-High"
        out.loc[solar > q3] = "High"
    elif len(positive):
        med = positive.median()
        out.loc[(solar > 0) & (solar <= med)] = "Low/Medium"
        out.loc[solar > med] = "High"
    return out


def _solar_loss_bucket_from_values(values: pd.Series) -> pd.Series:
    loss = pd.to_numeric(values, errors="coerce").fillna(0.0)
    out = pd.Series("None", index=values.index, dtype="object")
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


def _add_weather_residual_buckets(df: pd.DataFrame) -> pd.DataFrame:
    """Add reusable weather/solar buckets for recent residual profiles.

    These buckets make the online correction less blunt. For example, a recent positive
    residual during overcast/solar-loss periods should transfer more strongly to similar
    future overcast/solar-loss hours than to clear/high-solar hours.
    """
    out = df.copy()
    if "DailyMaxTempBucket" not in out.columns:
        if "DailyMaxTempBin" in out.columns:
            out["DailyMaxTempBucket"] = out["DailyMaxTempBin"].astype("object")
        elif "Temperature_DailyMax" in out.columns:
            out["DailyMaxTempBucket"] = _temp_bucket_from_values(out["Temperature_DailyMax"])
        elif "Temperature" in out.columns:
            out["DailyMaxTempBucket"] = _temp_bucket_from_values(out["Temperature"])
        else:
            out["DailyMaxTempBucket"] = np.nan

    if "CloudCoverBucket" not in out.columns:
        if "CloudCover_Norm" in out.columns:
            out["CloudCoverBucket"] = _cloud_bucket_from_values(out["CloudCover_Norm"])
        else:
            out["CloudCoverBucket"] = np.nan

    if "BTMSolarBucket" not in out.columns:
        if "BTM_Solar_Proxy_MW" in out.columns:
            out["BTMSolarBucket"] = _solar_bucket_from_values(out["BTM_Solar_Proxy_MW"])
        else:
            out["BTMSolarBucket"] = np.nan

    if "SolarLossBucket" not in out.columns:
        if "BTM_Solar_Loss_From_ClearSky_MW" in out.columns:
            out["SolarLossBucket"] = _solar_loss_bucket_from_values(out["BTM_Solar_Loss_From_ClearSky_MW"])
        elif "Midday_Overcast_Solar_Loss_MW" in out.columns:
            out["SolarLossBucket"] = _solar_loss_bucket_from_values(out["Midday_Overcast_Solar_Loss_MW"])
        else:
            out["SolarLossBucket"] = np.nan
    return out


def _prep_residual_frame(backtest_df: pd.DataFrame) -> pd.DataFrame:
    if backtest_df is None or backtest_df.empty:
        return pd.DataFrame()
    out = backtest_df.copy()
    out["DT"] = pd.to_datetime(out["DT"], errors="coerce")
    out = out.dropna(subset=["DT"]).sort_values("DT").reset_index(drop=True)
    if "Residual_MWH" in out.columns:
        out["Residual_MWH"] = _as_num(out["Residual_MWH"])
    elif {"Actual_MWH", "Raw_Forecast_MWH"}.issubset(out.columns):
        out["Residual_MWH"] = _as_num(out["Actual_MWH"]) - _as_num(out["Raw_Forecast_MWH"])
    else:
        return pd.DataFrame()
    out = out.dropna(subset=["Residual_MWH"])
    out["Hour"] = _as_num(out.get("Hour", out["DT"].dt.hour)).fillna(out["DT"].dt.hour).astype(int)
    out["HourGroup"] = out.get("HourGroup", out["Hour"].map(_hour_group))
    return _add_weather_residual_buckets(out)


def _clipper(cap: float):
    def _clip(x: float) -> float:
        return float(np.clip(x, -cap, cap)) if np.isfinite(x) else 0.0
    return _clip


def _group_lookup(work: pd.DataFrame, keys: list[str], cap: float, min_count: int) -> dict[str, float]:
    if not all(k in work.columns for k in keys):
        return {}
    g = work.dropna(subset=keys).groupby(keys, dropna=False)["Residual_MWH"].agg(["mean", "count"]).reset_index()
    g = g[g["count"] >= int(min_count)].copy()
    clip = _clipper(cap)
    if g.empty:
        return {}
    out: dict[str, float] = {}
    for _, row in g.iterrows():
        key = "|".join(str(row[k]) for k in keys)
        out[key] = clip(float(row["mean"]))
    return out


def build_recent_residual_profile(backtest_df: pd.DataFrame, config: dict | None = None) -> dict:
    """Build a compact, weather-aware correction profile from leakage-safe residuals.

    Residual convention is Actual - Forecast. Positive means the model is low.
    V12.4 still favors recent residuals, but now also learns correction levels by
    temperature bucket, cloud bucket, BTM solar bucket, and solar-loss bucket.
    """
    c = _cfg(config)
    if not bool(c.get("enabled", True)):
        return {"enabled": False}

    work = _prep_residual_frame(backtest_df)
    if work.empty:
        return {"enabled": True, "empty": True}

    recent_hours = int(c.get("recent_hours", 48))
    same_hour_days = int(c.get("same_hour_days", 7))
    min_bucket_count = int(c.get("min_weather_bucket_count", 6))
    max_dt = work["DT"].max()
    recent = work[work["DT"] >= max_dt - pd.Timedelta(hours=max(1, recent_hours))].copy()
    if recent.empty:
        recent = work.tail(min(len(work), max(24, recent_hours))).copy()

    same_hour_start = max_dt - pd.Timedelta(days=max(1, same_hour_days))
    same_hour = work[work["DT"] >= same_hour_start].copy()
    if same_hour.empty:
        same_hour = work.copy()

    cap = float(c.get("cap_mwh", 10.0))
    clip = _clipper(cap)

    hour_lookup = same_hour.groupby("Hour")["Residual_MWH"].mean().to_dict()
    hourgroup_lookup = same_hour.groupby("HourGroup")["Residual_MWH"].mean().to_dict()

    return {
        "enabled": True,
        "last_dt": str(max_dt),
        "last24_mean": clip(work.tail(min(24, len(work)))["Residual_MWH"].mean()),
        "recent_mean": clip(recent["Residual_MWH"].mean()),
        "global_mean": clip(work["Residual_MWH"].mean()),
        "same_hour_mean": {int(k): clip(v) for k, v in hour_lookup.items()},
        "hourgroup_mean": {str(k): clip(v) for k, v in hourgroup_lookup.items()},
        "temp_hourgroup_mean": _group_lookup(same_hour, ["DailyMaxTempBucket", "HourGroup"], cap, min_bucket_count),
        "cloud_hourgroup_mean": _group_lookup(same_hour, ["CloudCoverBucket", "HourGroup"], cap, min_bucket_count),
        "solar_hourgroup_mean": _group_lookup(same_hour, ["BTMSolarBucket", "HourGroup"], cap, min_bucket_count),
        "solar_loss_hourgroup_mean": _group_lookup(same_hour, ["SolarLossBucket", "HourGroup"], cap, min_bucket_count),
        "temp_cloud_hourgroup_mean": _group_lookup(same_hour, ["DailyMaxTempBucket", "CloudCoverBucket", "HourGroup"], cap, min_bucket_count),
        "metadata": {
            "n_backtest_hours": int(len(work)),
            "recent_hours_used": int(len(recent)),
            "same_hour_days": int(same_hour_days),
            "min_weather_bucket_count": int(min_bucket_count),
            "bias_mwh": float(work["Residual_MWH"].mean()),
            "mae_mwh": float(work["Residual_MWH"].abs().mean()),
            "underforecast_rate_pct": float((work["Residual_MWH"] > 0).mean() * 100.0),
        },
    }


def _row_with_buckets(row: pd.Series) -> pd.Series:
    tmp = pd.DataFrame([row.to_dict()])
    tmp = _add_weather_residual_buckets(tmp)
    return tmp.iloc[0]


def _weighted_recent_correction(row: pd.Series, profile: dict, config: dict | None, horizon_index: int | None = None) -> tuple[float, str]:
    c = _cfg(config)
    if not profile or not profile.get("enabled", True) or profile.get("empty"):
        return 0.0, "disabled_or_empty"

    weights = c.get("weights", {}) or {}
    w_recent = float(weights.get("recent_mean", 0.35))
    w_last24 = float(weights.get("last24_mean", 0.20))
    w_same = float(weights.get("same_hour", 0.16))
    w_hourgroup = float(weights.get("hourgroup", 0.06))
    w_global = float(weights.get("global", 0.03))
    w_temp_hg = float(weights.get("temp_hourgroup", 0.08))
    w_cloud_hg = float(weights.get("cloud_hourgroup", 0.05))
    w_solar_hg = float(weights.get("solar_hourgroup", 0.03))
    w_loss_hg = float(weights.get("solar_loss_hourgroup", 0.03))
    w_temp_cloud_hg = float(weights.get("temp_cloud_hourgroup", 0.01))

    row = _row_with_buckets(row)
    hour = int(row.get("Hour", pd.to_datetime(row.get("DT")).hour))
    hourgroup = str(row.get("HourGroup", _hour_group(hour)))
    temp_bucket = str(row.get("DailyMaxTempBucket")) if pd.notna(row.get("DailyMaxTempBucket")) else None
    cloud_bucket = str(row.get("CloudCoverBucket")) if pd.notna(row.get("CloudCoverBucket")) else None
    solar_bucket = str(row.get("BTMSolarBucket")) if pd.notna(row.get("BTMSolarBucket")) else None
    loss_bucket = str(row.get("SolarLossBucket")) if pd.notna(row.get("SolarLossBucket")) else None

    vals: list[tuple[str, float, float]] = []

    def add(name: str, value: Any, weight: float):
        try:
            v = float(value)
        except Exception:
            return
        if np.isfinite(v) and weight > 0:
            vals.append((name, v, float(weight)))

    def lookup_key(parts: list[str | None]) -> str | None:
        if any(p is None or p == "nan" for p in parts):
            return None
        return "|".join(str(p) for p in parts)

    add("recent_mean", profile.get("recent_mean"), w_recent)
    add("last24_mean", profile.get("last24_mean"), w_last24)
    add("same_hour", (profile.get("same_hour_mean", {}) or {}).get(hour), w_same)
    add("hourgroup", (profile.get("hourgroup_mean", {}) or {}).get(hourgroup), w_hourgroup)
    add("global", profile.get("global_mean"), w_global)
    add("temp_hourgroup", (profile.get("temp_hourgroup_mean", {}) or {}).get(lookup_key([temp_bucket, hourgroup])), w_temp_hg)
    add("cloud_hourgroup", (profile.get("cloud_hourgroup_mean", {}) or {}).get(lookup_key([cloud_bucket, hourgroup])), w_cloud_hg)
    add("solar_hourgroup", (profile.get("solar_hourgroup_mean", {}) or {}).get(lookup_key([solar_bucket, hourgroup])), w_solar_hg)
    add("solar_loss_hourgroup", (profile.get("solar_loss_hourgroup_mean", {}) or {}).get(lookup_key([loss_bucket, hourgroup])), w_loss_hg)
    add("temp_cloud_hourgroup", (profile.get("temp_cloud_hourgroup_mean", {}) or {}).get(lookup_key([temp_bucket, cloud_bucket, hourgroup])), w_temp_cloud_hg)

    if not vals:
        return 0.0, "no_match"

    numerator = sum(v * w for _, v, w in vals)
    denominator = sum(w for _, _, w in vals)
    raw = numerator / denominator if denominator else 0.0

    blend = float(c.get("blend", 0.85))
    cap = float(c.get("cap_mwh", 10.0))
    decay_hours = float(c.get("decay_hours", 96.0))
    min_decay = float(c.get("min_decay", 0.25))
    h = max(1, int(horizon_index or 1))
    decay = max(min_decay, math.exp(-(h - 1) / max(1.0, decay_hours)))
    if h <= 24:
        decay *= float(c.get("day1_scale", 1.0))
    correction_scale = _recent_hot_peak_scale(row, c) * _recent_horizon_regime_scale(row, c, horizon_index)
    correction = float(np.clip(raw * blend * decay * correction_scale, -cap, cap))
    return correction, "+".join(name for name, _, _ in vals)


def apply_recent_residual_correction(
    future_df: pd.DataFrame,
    profile: dict | None,
    config: dict | None = None,
    base_col: str = "Calibrated_Forecast_MWH",
) -> pd.DataFrame:
    """Add recent online level correction to future forecasts.

    V12.4 makes this correction weather-aware, which helps avoid applying the same
    upward shift to sunny/high-solar hours that was learned from cloudy/solar-loss hours.
    """
    out = future_df.copy().sort_values("DT").reset_index(drop=True)
    if base_col not in out.columns:
        base_col = "Raw_Forecast_MWH"
    if "Hour" not in out.columns:
        out["Hour"] = pd.to_datetime(out["DT"]).dt.hour.astype(int)
    if "HourGroup" not in out.columns:
        out["HourGroup"] = out["Hour"].map(_hour_group)
    out = _add_weather_residual_buckets(out)

    corrections = []
    sources = []
    for i, row in out.iterrows():
        corr, source = _weighted_recent_correction(row, profile or {}, config, horizon_index=i + 1)
        corrections.append(corr)
        sources.append(source)

    out["Recent_Level_Correction_MWH"] = corrections
    out["Recent_Correction_Source"] = sources
    out["Pre_Recent_Forecast_MWH"] = pd.to_numeric(out[base_col], errors="coerce")
    out["Recent_Corrected_Forecast_MWH"] = (out["Pre_Recent_Forecast_MWH"] + out["Recent_Level_Correction_MWH"]).clip(lower=0.0)
    out["Final_Forecast_MWH"] = out["Recent_Corrected_Forecast_MWH"]
    # Preserve the existing dashboard/output contract: Calibrated_Forecast_MWH is the final production forecast.
    out["Calibrated_Forecast_MWH"] = out["Final_Forecast_MWH"]
    return out


def simulate_recent_residual_correction_backtest(
    backtest_df: pd.DataFrame,
    config: dict | None = None,
    base_col: str = "Raw_Forecast_MWH",
) -> pd.DataFrame:
    """Simulate the recent correction over a backtest using only prior residuals for each row.

    This implementation is intentionally lightweight because it runs every normal forecast run.
    It uses only residuals from earlier holdout hours and matches future rows to prior rows by
    hour, hour group, temperature bucket, cloud bucket, solar bucket, and solar-loss bucket.
    """
    c = _cfg(config)
    out = backtest_df.copy().sort_values("DT").reset_index(drop=True)
    if out.empty or not bool(c.get("enabled", True)) or not {"Actual_MWH", base_col}.issubset(out.columns):
        out["Recent_Level_Correction_MWH"] = 0.0
        out["Recent_Corrected_Forecast_MWH"] = pd.to_numeric(out.get(base_col, out.get("Raw_Forecast_MWH", 0.0)), errors="coerce")
        out["Final_Backtest_Forecast_MWH"] = out["Recent_Corrected_Forecast_MWH"]
        return out

    out["DT"] = pd.to_datetime(out["DT"], errors="coerce")
    out["Hour"] = _as_num(out.get("Hour", out["DT"].dt.hour)).fillna(out["DT"].dt.hour).astype(int)
    out["HourGroup"] = out.get("HourGroup", out["Hour"].map(_hour_group))
    out = _add_weather_residual_buckets(out)
    out["_RecentBasisResidual"] = _as_num(out["Actual_MWH"]) - _as_num(out[base_col])

    weights = c.get("weights", {}) or {}
    w_recent = float(weights.get("recent_mean", 0.35))
    w_last24 = float(weights.get("last24_mean", 0.20))
    w_same = float(weights.get("same_hour", 0.16))
    w_hourgroup = float(weights.get("hourgroup", 0.06))
    w_global = float(weights.get("global", 0.03))
    w_temp_hg = float(weights.get("temp_hourgroup", 0.08))
    w_cloud_hg = float(weights.get("cloud_hourgroup", 0.05))
    w_solar_hg = float(weights.get("solar_hourgroup", 0.03))
    w_loss_hg = float(weights.get("solar_loss_hourgroup", 0.03))
    w_temp_cloud_hg = float(weights.get("temp_cloud_hourgroup", 0.01))

    recent_hours = int(c.get("recent_hours", 48))
    same_hour_days = int(c.get("same_hour_days", 7))
    cap = float(c.get("cap_mwh", 10.0))
    blend = float(c.get("blend", 0.85))

    corrections: list[float] = []
    sources: list[str] = []

    def add(vals: list[tuple[str, float, float]], name: str, series: pd.Series, weight: float):
        if weight <= 0 or series.empty:
            return
        v = pd.to_numeric(series, errors="coerce").mean()
        if np.isfinite(v):
            vals.append((name, float(np.clip(v, -cap, cap)), float(weight)))

    for i, row in out.iterrows():
        hist = out.iloc[:i]
        hist = hist[pd.to_numeric(hist["_RecentBasisResidual"], errors="coerce").notna()]
        if hist.empty:
            corrections.append(0.0)
            sources.append("insufficient_prior_residuals")
            continue

        vals: list[tuple[str, float, float]] = []
        same_window_start = row["DT"] - pd.Timedelta(days=max(1, same_hour_days))
        same_window = hist[hist["DT"] >= same_window_start]
        if same_window.empty:
            same_window = hist

        add(vals, "recent_mean", hist.tail(max(1, recent_hours))["_RecentBasisResidual"], w_recent)
        add(vals, "last24_mean", hist.tail(min(24, len(hist)))["_RecentBasisResidual"], w_last24)
        add(vals, "same_hour", same_window.loc[same_window["Hour"].eq(row["Hour"]), "_RecentBasisResidual"], w_same)
        add(vals, "hourgroup", same_window.loc[same_window["HourGroup"].eq(row["HourGroup"]), "_RecentBasisResidual"], w_hourgroup)
        add(vals, "global", hist["_RecentBasisResidual"], w_global)
        if pd.notna(row.get("DailyMaxTempBucket")):
            add(vals, "temp_hourgroup", same_window.loc[
                same_window["DailyMaxTempBucket"].eq(row.get("DailyMaxTempBucket")) & same_window["HourGroup"].eq(row["HourGroup"]), "_RecentBasisResidual"
            ], w_temp_hg)
        if pd.notna(row.get("CloudCoverBucket")):
            add(vals, "cloud_hourgroup", same_window.loc[
                same_window["CloudCoverBucket"].eq(row.get("CloudCoverBucket")) & same_window["HourGroup"].eq(row["HourGroup"]), "_RecentBasisResidual"
            ], w_cloud_hg)
        if pd.notna(row.get("BTMSolarBucket")):
            add(vals, "solar_hourgroup", same_window.loc[
                same_window["BTMSolarBucket"].eq(row.get("BTMSolarBucket")) & same_window["HourGroup"].eq(row["HourGroup"]), "_RecentBasisResidual"
            ], w_solar_hg)
        if pd.notna(row.get("SolarLossBucket")):
            add(vals, "solar_loss_hourgroup", same_window.loc[
                same_window["SolarLossBucket"].eq(row.get("SolarLossBucket")) & same_window["HourGroup"].eq(row["HourGroup"]), "_RecentBasisResidual"
            ], w_loss_hg)
        if pd.notna(row.get("DailyMaxTempBucket")) and pd.notna(row.get("CloudCoverBucket")):
            add(vals, "temp_cloud_hourgroup", same_window.loc[
                same_window["DailyMaxTempBucket"].eq(row.get("DailyMaxTempBucket"))
                & same_window["CloudCoverBucket"].eq(row.get("CloudCoverBucket"))
                & same_window["HourGroup"].eq(row["HourGroup"]), "_RecentBasisResidual"
            ], w_temp_cloud_hg)

        if not vals:
            corrections.append(0.0)
            sources.append("no_match")
            continue
        raw = sum(v * w for _, v, w in vals) / sum(w for _, _, w in vals)
        correction_scale = _recent_hot_peak_scale(row, c) * _recent_horizon_regime_scale(row, c)
        corrections.append(float(np.clip(raw * blend * correction_scale, -cap, cap)))
        sources.append("+".join(name for name, _, _ in vals))

    out["Recent_Level_Correction_MWH"] = corrections
    out["Recent_Correction_Source"] = sources
    out["Pre_Recent_Forecast_MWH"] = _as_num(out[base_col])
    out["Recent_Corrected_Forecast_MWH"] = (out["Pre_Recent_Forecast_MWH"] + out["Recent_Level_Correction_MWH"]).clip(lower=0.0)
    out["Final_Backtest_Forecast_MWH"] = out["Recent_Corrected_Forecast_MWH"]
    out["Final_Forecast_MWH"] = out["Recent_Corrected_Forecast_MWH"]
    out["Recent_Corrected_Residual_MWH"] = _as_num(out["Actual_MWH"]) - _as_num(out["Recent_Corrected_Forecast_MWH"])
    out["Recent_Corrected_AbsError_MWH"] = out["Recent_Corrected_Residual_MWH"].abs()
    out["Recent_Corrected_APE"] = np.where(
        _as_num(out["Actual_MWH"]).abs() > 1e-9,
        out["Recent_Corrected_AbsError_MWH"] / _as_num(out["Actual_MWH"]).abs() * 100.0,
        np.nan,
    )
    return out.drop(columns=["_RecentBasisResidual"], errors="ignore")
