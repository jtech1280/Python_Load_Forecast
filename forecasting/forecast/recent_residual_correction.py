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
    return ((config or {}).get("calibration", {}) or {}).get(
        "recent_residual", {}
    ) or {}


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
    daily_max = pd.to_numeric(
        pd.Series([row.get("Temperature_DailyMax")]), errors="coerce"
    ).iloc[0]
    if (
        hour not in hours
        or not np.isfinite(daily_max)
        or daily_max < float(hot_cfg.get("min_maxtemp_f", 90.0))
    ):
        return 1.0
    season = row.get("Season")
    if pd.isna(season):
        season = _season_from_month(pd.to_datetime(row.get("DT")).month)
    return float(
        (hot_cfg.get("season_scales", {}) or {}).get(
            str(season), hot_cfg.get("default_scale", 1.0)
        )
    )


def _forecast_day(row: pd.Series, horizon_index: int | None = None) -> int:
    day = pd.to_numeric(pd.Series([row.get("Forecast_Day")]), errors="coerce").iloc[0]
    if np.isfinite(day):
        return max(1, int(day))
    h = max(1, int(horizon_index or 1))
    return int(math.ceil(h / 24.0))


def _recent_horizon_regime_scale(
    row: pd.Series, recent_cfg: dict, horizon_index: int | None = None
) -> float:
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
        daily_max = pd.to_numeric(
            pd.Series([row.get("Temperature_DailyMax")]), errors="coerce"
        ).iloc[0]
        cloud = pd.to_numeric(
            pd.Series([row.get("CloudCover_Norm")]), errors="coerce"
        ).iloc[0]
        if np.isfinite(cloud) and cloud > 1.5:
            cloud = cloud / 100.0
        min_temp = float(hot_clear_cfg.get("min_maxtemp_f", 90.0))
        max_cloud = float(hot_clear_cfg.get("max_cloud_cover_norm", 0.20))
        if (
            str(season) == "Summer"
            and np.isfinite(daily_max)
            and daily_max >= min_temp
            and np.isfinite(cloud)
            and cloud <= max_cloud
        ):
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
        labels=[
            "Clear/Low",
            "Some Clouds",
            "Partly Cloudy",
            "Mostly Cloudy",
            "Overcast",
        ],
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
            out["DailyMaxTempBucket"] = _temp_bucket_from_values(
                out["Temperature_DailyMax"]
            )
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
            out["SolarLossBucket"] = _solar_loss_bucket_from_values(
                out["BTM_Solar_Loss_From_ClearSky_MW"]
            )
        elif "Midday_Overcast_Solar_Loss_MW" in out.columns:
            out["SolarLossBucket"] = _solar_loss_bucket_from_values(
                out["Midday_Overcast_Solar_Loss_MW"]
            )
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
        out["Residual_MWH"] = _as_num(out["Actual_MWH"]) - _as_num(
            out["Raw_Forecast_MWH"]
        )
    else:
        return pd.DataFrame()
    out = out.dropna(subset=["Residual_MWH"])
    out["Hour"] = (
        _as_num(out.get("Hour", out["DT"].dt.hour))
        .fillna(out["DT"].dt.hour)
        .astype(int)
    )
    out["HourGroup"] = out.get("HourGroup", out["Hour"].map(_hour_group))
    return _add_weather_residual_buckets(out)


def _clipper(cap: float):
    def _clip(x: float) -> float:
        return float(np.clip(x, -cap, cap)) if np.isfinite(x) else 0.0

    return _clip


def _ar_config(recent_cfg: dict) -> dict:
    return recent_cfg.get("ar_residual", {}) or {}


def _ar_cap(recent_cfg: dict, ar_cfg: dict) -> float:
    recent_cap = abs(float(recent_cfg.get("cap_mwh", 10.0)))
    default_cap = min(recent_cap, 4.0)
    return abs(float(ar_cfg.get("cap_mwh", default_cap)))


def _build_ar_residual_state(
    work: pd.DataFrame, recent_cfg: dict, residual_col: str = "Residual_MWH"
) -> dict:
    """Estimate a conservative AR(1) residual carryover from pre-origin residuals."""
    ar_cfg = _ar_config(recent_cfg)
    if not bool(ar_cfg.get("enabled", False)):
        return {"enabled": False}
    if (
        work is None
        or work.empty
        or "DT" not in work.columns
        or residual_col not in work.columns
    ):
        return {"enabled": True, "empty": True}

    frame = work.copy()
    frame["DT"] = pd.to_datetime(frame["DT"], errors="coerce")
    frame[residual_col] = _as_num(frame[residual_col])
    frame = (
        frame.dropna(subset=["DT", residual_col])
        .sort_values("DT")
        .reset_index(drop=True)
    )
    if frame.empty:
        return {"enabled": True, "empty": True}
    if "Hour" not in frame.columns:
        frame["Hour"] = frame["DT"].dt.hour.astype(int)
    else:
        frame["Hour"] = _as_num(frame["Hour"]).fillna(frame["DT"].dt.hour).astype(int)

    max_dt = frame["DT"].max()
    lookback_hours = int(
        ar_cfg.get("lookback_hours", ar_cfg.get("phi_window_hours", 168))
    )
    recent = frame[
        frame["DT"] >= max_dt - pd.Timedelta(hours=max(1, lookback_hours))
    ].copy()
    if recent.empty:
        recent = frame.copy()

    lagged = recent[["DT", residual_col]].copy()
    lagged["prev_residual"] = lagged[residual_col].shift(1)
    lagged["gap_hours"] = lagged["DT"].diff().dt.total_seconds() / 3600.0
    max_gap_hours = float(ar_cfg.get("max_lag_gap_hours", 1.5))
    pairs = lagged[
        lagged["prev_residual"].notna()
        & lagged[residual_col].notna()
        & lagged["gap_hours"].gt(0.0)
        & lagged["gap_hours"].le(max_gap_hours)
    ]

    min_pairs = int(ar_cfg.get("min_pairs", ar_cfg.get("min_observations", 24)))
    n_pairs = int(len(pairs))
    phi_raw = 0.0
    if n_pairs >= min_pairs:
        prev = pd.to_numeric(pairs["prev_residual"], errors="coerce").to_numpy(
            dtype=float
        )
        curr = pd.to_numeric(pairs[residual_col], errors="coerce").to_numpy(dtype=float)
        denom = float(np.dot(prev, prev))
        if denom > 1e-9:
            phi_raw = float(np.dot(prev, curr) / denom)

    if not bool(ar_cfg.get("allow_negative_phi", False)):
        phi_raw = max(0.0, phi_raw)
    shrink_k = max(0.0, float(ar_cfg.get("phi_shrink_k", 72.0)))
    shrink = n_pairs / (n_pairs + shrink_k) if n_pairs > 0 else 0.0
    phi_cap = abs(float(ar_cfg.get("phi_cap", 0.40)))
    phi = float(np.clip(phi_raw * shrink, -phi_cap, phi_cap))

    same_hour_days = int(
        ar_cfg.get("same_hour_days", recent_cfg.get("same_hour_days", 7))
    )
    same_start = max_dt - pd.Timedelta(days=max(1, same_hour_days))
    same_hour = frame[frame["DT"] >= same_start].copy()
    if same_hour.empty:
        same_hour = frame.copy()
    clip = _clipper(_ar_cap(recent_cfg, ar_cfg))
    same_lookup = same_hour.groupby("Hour")[residual_col].mean().to_dict()

    latest_residual = float(frame[residual_col].iloc[-1])
    return {
        "enabled": True,
        "empty": False,
        "lookback_hours": int(lookback_hours),
        "n_pairs": n_pairs,
        "phi_raw": float(phi_raw),
        "phi_shrink": float(shrink),
        "phi": phi,
        "latest_residual": latest_residual,
        "same_hour_mean": {int(k): clip(v) for k, v in same_lookup.items()},
        "same_hour_days": int(same_hour_days),
    }


def _ar_forecast_day_scale(
    row: pd.Series, ar_cfg: dict, horizon_index: int | None = None
) -> float:
    scales = ar_cfg.get("forecast_day_scales", ar_cfg.get("horizon_scales", {})) or {}
    forecast_day = _forecast_day(row, horizon_index)
    if forecast_day <= 1:
        return float(scales.get("day1", 1.0))
    if 2 <= forecast_day <= 3:
        return float(scales.get("days2to3", 0.45))
    if 4 <= forecast_day <= 7:
        return float(scales.get("days4to7", 0.15))
    return float(scales.get("days8plus", 0.0))


def _ar_residual_correction(
    row: pd.Series,
    profile: dict,
    recent_cfg: dict,
    horizon_index: int | None = None,
) -> tuple[float, float, float, str]:
    ar_cfg = _ar_config(recent_cfg)
    state = (profile or {}).get("ar_residual", {}) or {}
    if (
        not bool(ar_cfg.get("enabled", False))
        or not bool(state.get("enabled", False))
        or state.get("empty")
    ):
        return 0.0, np.nan, np.nan, "ar_disabled_or_empty"

    phi = float(state.get("phi", 0.0) or 0.0)
    latest_residual = float(state.get("latest_residual", 0.0) or 0.0)
    if (
        not np.isfinite(phi)
        or not np.isfinite(latest_residual)
        or abs(phi) <= 1e-12
        or abs(latest_residual) <= 1e-12
    ):
        return 0.0, phi, latest_residual, "ar1_zero"

    raw = phi * latest_residual
    source = "ar1_latest_residual"
    hour = int(row.get("Hour", pd.to_datetime(row.get("DT")).hour))
    same_lookup = state.get("same_hour_mean", {}) or {}
    same_hour_value = same_lookup.get(hour, same_lookup.get(str(hour)))
    same_hour_blend = float(
        np.clip(float(ar_cfg.get("same_hour_blend", 0.0)), 0.0, 1.0)
    )
    if same_hour_blend > 0.0 and same_hour_value is not None:
        same_hour_value = float(same_hour_value)
        if np.isfinite(same_hour_value):
            raw = (1.0 - same_hour_blend) * raw + same_hour_blend * same_hour_value
            source = "ar1_latest_residual+same_hour"

    blend = float(ar_cfg.get("blend", ar_cfg.get("correction_blend", 0.35)))
    h = max(1, int(horizon_index or 1))
    decay_hours = float(ar_cfg.get("decay_hours", 36.0))
    min_decay = float(ar_cfg.get("min_decay", 0.0))
    decay = max(min_decay, math.exp(-(h - 1) / max(1.0, decay_hours)))
    horizon_scale = _ar_forecast_day_scale(row, ar_cfg, horizon_index)
    cap = _ar_cap(recent_cfg, ar_cfg)
    correction = float(np.clip(raw * blend * decay * horizon_scale, -cap, cap))
    if abs(correction) <= 1e-12:
        source = "ar1_zero"
    return correction, phi, latest_residual, source


def _origin_day_config(recent_cfg: dict) -> dict:
    return recent_cfg.get("origin_day_state", {}) or {}


def _origin_day_cap(recent_cfg: dict, origin_cfg: dict) -> float:
    recent_cap = abs(float(recent_cfg.get("cap_mwh", 10.0)))
    default_cap = min(recent_cap, 2.0)
    return abs(float(origin_cfg.get("cap_mwh", default_cap)))


def _date_from_dt(dt: pd.Series) -> pd.Series:
    try:
        return pd.to_datetime(dt, errors="coerce").dt.normalize()
    except Exception:
        return (
            pd.to_datetime(dt, errors="coerce", utc=True)
            .dt.tz_convert(None)
            .dt.normalize()
        )


def _latest_same_sign_run_hours(
    residuals: pd.Series, target: float, threshold: float
) -> int:
    if not np.isfinite(target) or abs(target) <= 1e-12:
        return 0
    sign = np.sign(target)
    run = 0
    for value in pd.to_numeric(residuals, errors="coerce").dropna().iloc[::-1]:
        v = float(value)
        if abs(v) < threshold or np.sign(v) != sign:
            break
        run += 1
    return int(run)


def _build_origin_day_state(
    work: pd.DataFrame, recent_cfg: dict, residual_col: str = "Residual_MWH"
) -> dict:
    """Estimate a shrunk day-level residual state from pre-origin residuals."""
    origin_cfg = _origin_day_config(recent_cfg)
    if not bool(origin_cfg.get("enabled", False)):
        return {"enabled": False}
    if (
        work is None
        or work.empty
        or "DT" not in work.columns
        or residual_col not in work.columns
    ):
        return {
            "enabled": True,
            "empty": True,
            "source": "origin_day_disabled_or_empty",
        }

    frame = work.copy()
    frame["DT"] = pd.to_datetime(frame["DT"], errors="coerce")
    frame[residual_col] = _as_num(frame[residual_col])
    frame = (
        frame.dropna(subset=["DT", residual_col])
        .sort_values("DT")
        .reset_index(drop=True)
    )
    if frame.empty:
        return {
            "enabled": True,
            "empty": True,
            "source": "origin_day_disabled_or_empty",
        }
    if "Hour" not in frame.columns:
        frame["Hour"] = frame["DT"].dt.hour.astype(int)
    else:
        frame["Hour"] = _as_num(frame["Hour"]).fillna(frame["DT"].dt.hour).astype(int)
    if "HourGroup" not in frame.columns:
        frame["HourGroup"] = frame["Hour"].map(_hour_group)
    else:
        frame["HourGroup"] = frame["HourGroup"].fillna(frame["Hour"].map(_hour_group))

    max_dt = frame["DT"].max()
    lookback_days = int(origin_cfg.get("lookback_days", 7))
    recent = frame[
        frame["DT"] >= max_dt - pd.Timedelta(days=max(1, lookback_days))
    ].copy()
    if recent.empty:
        recent = frame.copy()
    recent["Date"] = _date_from_dt(recent["DT"])

    day_stats = (
        recent.groupby("Date", dropna=True)[residual_col]
        .agg(["mean", "count"])
        .reset_index()
    )
    min_day_hours = int(origin_cfg.get("min_day_hours", 12))
    eligible_days = day_stats[day_stats["count"] >= min_day_hours].copy()
    if eligible_days.empty:
        eligible_days = day_stats.copy()
    eligible_days = eligible_days.sort_values("Date").tail(max(1, lookback_days))

    n_days = int(len(eligible_days))
    n_hours = int(eligible_days["count"].sum()) if not eligible_days.empty else 0
    min_days = int(origin_cfg.get("min_days", 2))
    min_total_hours = int(origin_cfg.get("min_total_hours", 36))
    if n_days < min_days or n_hours < min_total_hours:
        return {
            "enabled": True,
            "empty": True,
            "source": "origin_day_insufficient_history",
            "n_days": n_days,
            "n_hours": n_hours,
        }

    latest_day_mean = float(eligible_days["mean"].iloc[-1])
    recent_day_mean = float(
        np.average(
            eligible_days["mean"], weights=np.sqrt(eligible_days["count"].clip(lower=1))
        )
    )
    latest_weight = float(
        np.clip(float(origin_cfg.get("latest_day_weight", 0.65)), 0.0, 1.0)
    )
    state_raw = (
        latest_weight * latest_day_mean + (1.0 - latest_weight) * recent_day_mean
    )

    source = "origin_day_state"
    if bool(origin_cfg.get("require_latest_recent_same_sign", True)):
        if (
            np.isfinite(latest_day_mean)
            and np.isfinite(recent_day_mean)
            and latest_day_mean * recent_day_mean < 0.0
        ):
            state_raw = 0.0
            source = "origin_day_sign_mismatch"

    shrink_days = max(0.0, float(origin_cfg.get("shrink_days", 6.0)))
    shrink_hours = max(0.0, float(origin_cfg.get("shrink_hours", 96.0)))
    day_shrink = n_days / (n_days + shrink_days) if n_days > 0 else 0.0
    hour_shrink = n_hours / (n_hours + shrink_hours) if n_hours > 0 else 0.0
    shrink = float(day_shrink * hour_shrink)
    state_cap = abs(
        float(
            origin_cfg.get(
                "state_cap_mwh", _origin_day_cap(recent_cfg, origin_cfg) * 2.0
            )
        )
    )
    state_mwh = float(np.clip(state_raw * shrink, -state_cap, state_cap))

    clip = _clipper(_origin_day_cap(recent_cfg, origin_cfg))
    same_hour_lookup = recent.groupby("Hour")[residual_col].mean().to_dict()
    hourgroup_lookup = recent.groupby("HourGroup")[residual_col].mean().to_dict()
    run_threshold = abs(float(origin_cfg.get("same_sign_threshold_mwh", 0.25)))
    same_sign_run_hours = _latest_same_sign_run_hours(
        frame[residual_col], state_raw, run_threshold
    )

    return {
        "enabled": True,
        "empty": False,
        "source": source,
        "lookback_days": int(lookback_days),
        "n_days": n_days,
        "n_hours": n_hours,
        "latest_day_mean": float(latest_day_mean),
        "recent_day_mean": float(recent_day_mean),
        "state_raw_mwh": float(state_raw),
        "state_shrink": shrink,
        "state_mwh": state_mwh,
        "same_sign_run_hours": int(same_sign_run_hours),
        "same_hour_mean": {int(k): clip(v) for k, v in same_hour_lookup.items()},
        "hourgroup_mean": {str(k): clip(v) for k, v in hourgroup_lookup.items()},
    }


def _origin_day_bucket(forecast_day: int) -> str:
    if forecast_day <= 1:
        return "day1"
    if 2 <= forecast_day <= 3:
        return "days2to3"
    if 4 <= forecast_day <= 7:
        return "days4to7"
    return "days8plus"


def _origin_day_forecast_day_scale(
    row: pd.Series, origin_cfg: dict, horizon_index: int | None = None
) -> float:
    scales = origin_cfg.get("forecast_day_scales", {}) or {}
    bucket = _origin_day_bucket(_forecast_day(row, horizon_index))
    defaults = {"day1": 0.35, "days2to3": 0.45, "days4to7": 0.25, "days8plus": 0.08}
    return float(scales.get(bucket, defaults[bucket]))


def _origin_day_cap_for_row(
    row: pd.Series,
    recent_cfg: dict,
    origin_cfg: dict,
    horizon_index: int | None = None,
) -> float:
    cap = _origin_day_cap(recent_cfg, origin_cfg)
    cap_by_day = origin_cfg.get("cap_by_forecast_day", {}) or {}
    bucket = _origin_day_bucket(_forecast_day(row, horizon_index))
    if bucket in cap_by_day:
        cap = min(cap, abs(float(cap_by_day[bucket])))
    return float(cap)


def _same_signed_component(base: float, value: Any, origin_cfg: dict) -> float | None:
    try:
        v = float(value)
    except Exception:
        return None
    if not np.isfinite(v):
        return None
    if (
        bool(origin_cfg.get("require_component_same_sign", True))
        and abs(base) > 1e-12
        and abs(v) > 1e-12
    ):
        if np.sign(base) != np.sign(v):
            return None
    return v


def _origin_day_residual_correction(
    row: pd.Series,
    profile: dict,
    recent_cfg: dict,
    horizon_index: int | None = None,
) -> tuple[float, float, float, str]:
    origin_cfg = _origin_day_config(recent_cfg)
    state = (profile or {}).get("origin_day_state", {}) or {}
    if (
        not bool(origin_cfg.get("enabled", False))
        or not bool(state.get("enabled", False))
        or state.get("empty")
    ):
        return (
            0.0,
            np.nan,
            np.nan,
            str(state.get("source", "origin_day_disabled_or_empty")),
        )

    state_mwh = float(state.get("state_mwh", 0.0) or 0.0)
    latest_day_mean = float(state.get("latest_day_mean", np.nan))
    min_abs_state = abs(float(origin_cfg.get("min_abs_state_mwh", 0.50)))
    if not np.isfinite(state_mwh) or abs(state_mwh) < min_abs_state:
        return 0.0, state_mwh, latest_day_mean, "origin_day_zero"

    raw = state_mwh
    source_parts = [str(state.get("source", "origin_day_state"))]
    hour = int(row.get("Hour", pd.to_datetime(row.get("DT")).hour))
    hourgroup = str(row.get("HourGroup", _hour_group(hour)))

    hourgroup_blend = float(
        np.clip(float(origin_cfg.get("hourgroup_blend", 0.15)), 0.0, 1.0)
    )
    hourgroup_value = _same_signed_component(
        raw, (state.get("hourgroup_mean", {}) or {}).get(hourgroup), origin_cfg
    )
    if hourgroup_blend > 0.0 and hourgroup_value is not None:
        raw = (1.0 - hourgroup_blend) * raw + hourgroup_blend * hourgroup_value
        source_parts.append("hourgroup")

    same_hour_blend = float(
        np.clip(float(origin_cfg.get("same_hour_blend", 0.10)), 0.0, 1.0)
    )
    same_lookup = state.get("same_hour_mean", {}) or {}
    same_hour_value = _same_signed_component(
        raw, same_lookup.get(hour, same_lookup.get(str(hour))), origin_cfg
    )
    if same_hour_blend > 0.0 and same_hour_value is not None:
        raw = (1.0 - same_hour_blend) * raw + same_hour_blend * same_hour_value
        source_parts.append("same_hour")

    same_sign_run = int(state.get("same_sign_run_hours", 0) or 0)
    min_run = int(origin_cfg.get("min_same_sign_run_hours", 4))
    run_scale = 1.0
    if min_run > 0 and same_sign_run < min_run:
        run_scale = float(
            np.clip(float(origin_cfg.get("short_run_scale", 0.50)), 0.0, 1.0)
        )
        source_parts.append("short_run_scaled")

    forecast_day = _forecast_day(row, horizon_index)
    decay_days = float(origin_cfg.get("decay_days", 5.0))
    min_decay = float(origin_cfg.get("min_decay", 0.0))
    decay = max(min_decay, math.exp(-(forecast_day - 1) / max(1.0, decay_days)))
    scale = _origin_day_forecast_day_scale(row, origin_cfg, horizon_index)
    blend = float(origin_cfg.get("blend", 0.45))
    cap = _origin_day_cap_for_row(row, recent_cfg, origin_cfg, horizon_index)
    correction = float(np.clip(raw * blend * decay * scale * run_scale, -cap, cap))
    if abs(correction) <= 1e-12:
        return 0.0, state_mwh, latest_day_mean, "origin_day_zero"
    return correction, state_mwh, latest_day_mean, "+".join(source_parts)


def _combine_recent_and_ar_corrections(
    base_correction: float,
    base_source: str,
    row: pd.Series,
    profile: dict,
    recent_cfg: dict,
    horizon_index: int | None = None,
) -> tuple[float, str, float, float, float, str, float, float, float, str]:
    ar_correction, ar_phi, ar_latest, ar_source = _ar_residual_correction(
        row, profile, recent_cfg, horizon_index
    )
    cap = float(recent_cfg.get("cap_mwh", 10.0))
    total_after_ar = float(np.clip(base_correction + ar_correction, -cap, cap))
    ar_applied = float(total_after_ar - base_correction)
    origin_correction, origin_state, origin_latest_day, origin_source = (
        _origin_day_residual_correction(
            row,
            profile,
            recent_cfg,
            horizon_index,
        )
    )
    total = float(np.clip(total_after_ar + origin_correction, -cap, cap))
    origin_applied = float(total - total_after_ar)

    source_parts: list[str] = []
    if (
        base_source
        not in {"no_match", "disabled_or_empty", "insufficient_prior_residuals"}
        or abs(base_correction) > 1e-12
    ):
        source_parts.append(base_source)
    if abs(ar_applied) > 1e-12:
        source_parts.append(ar_source)
    if abs(origin_applied) > 1e-12:
        source_parts.append(origin_source)
    source = "+".join(source_parts) if source_parts else base_source
    return (
        total,
        source,
        ar_applied,
        ar_phi,
        ar_latest,
        ar_source,
        origin_applied,
        origin_state,
        origin_latest_day,
        origin_source,
    )


def _group_lookup(
    work: pd.DataFrame, keys: list[str], cap: float, min_count: int
) -> dict[str, float]:
    if not all(k in work.columns for k in keys):
        return {}
    g = (
        work.dropna(subset=keys)
        .groupby(keys, dropna=False)["Residual_MWH"]
        .agg(["mean", "count"])
        .reset_index()
    )
    g = g[g["count"] >= int(min_count)].copy()
    clip = _clipper(cap)
    if g.empty:
        return {}
    out: dict[str, float] = {}
    for _, row in g.iterrows():
        key = "|".join(str(row[k]) for k in keys)
        out[key] = clip(float(row["mean"]))
    return out


def build_recent_residual_profile(
    backtest_df: pd.DataFrame, config: dict | None = None
) -> dict:
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
    recent = work[
        work["DT"] >= max_dt - pd.Timedelta(hours=max(1, recent_hours))
    ].copy()
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
        "temp_hourgroup_mean": _group_lookup(
            same_hour, ["DailyMaxTempBucket", "HourGroup"], cap, min_bucket_count
        ),
        "cloud_hourgroup_mean": _group_lookup(
            same_hour, ["CloudCoverBucket", "HourGroup"], cap, min_bucket_count
        ),
        "solar_hourgroup_mean": _group_lookup(
            same_hour, ["BTMSolarBucket", "HourGroup"], cap, min_bucket_count
        ),
        "solar_loss_hourgroup_mean": _group_lookup(
            same_hour, ["SolarLossBucket", "HourGroup"], cap, min_bucket_count
        ),
        "temp_cloud_hourgroup_mean": _group_lookup(
            same_hour,
            ["DailyMaxTempBucket", "CloudCoverBucket", "HourGroup"],
            cap,
            min_bucket_count,
        ),
        "ar_residual": _build_ar_residual_state(work, c),
        "origin_day_state": _build_origin_day_state(work, c),
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


def _weighted_recent_correction(
    row: pd.Series,
    profile: dict,
    config: dict | None,
    horizon_index: int | None = None,
) -> tuple[float, str, float, float, float, str, float, float, float, str]:
    c = _cfg(config)
    if not profile or not profile.get("enabled", True) or profile.get("empty"):
        return (
            0.0,
            "disabled_or_empty",
            0.0,
            np.nan,
            np.nan,
            "ar_disabled_or_empty",
            0.0,
            np.nan,
            np.nan,
            "origin_day_disabled_or_empty",
        )

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
    temp_bucket = (
        str(row.get("DailyMaxTempBucket"))
        if pd.notna(row.get("DailyMaxTempBucket"))
        else None
    )
    cloud_bucket = (
        str(row.get("CloudCoverBucket"))
        if pd.notna(row.get("CloudCoverBucket"))
        else None
    )
    solar_bucket = (
        str(row.get("BTMSolarBucket")) if pd.notna(row.get("BTMSolarBucket")) else None
    )
    loss_bucket = (
        str(row.get("SolarLossBucket"))
        if pd.notna(row.get("SolarLossBucket"))
        else None
    )

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
    add(
        "hourgroup",
        (profile.get("hourgroup_mean", {}) or {}).get(hourgroup),
        w_hourgroup,
    )
    add("global", profile.get("global_mean"), w_global)
    add(
        "temp_hourgroup",
        (profile.get("temp_hourgroup_mean", {}) or {}).get(
            lookup_key([temp_bucket, hourgroup])
        ),
        w_temp_hg,
    )
    add(
        "cloud_hourgroup",
        (profile.get("cloud_hourgroup_mean", {}) or {}).get(
            lookup_key([cloud_bucket, hourgroup])
        ),
        w_cloud_hg,
    )
    add(
        "solar_hourgroup",
        (profile.get("solar_hourgroup_mean", {}) or {}).get(
            lookup_key([solar_bucket, hourgroup])
        ),
        w_solar_hg,
    )
    add(
        "solar_loss_hourgroup",
        (profile.get("solar_loss_hourgroup_mean", {}) or {}).get(
            lookup_key([loss_bucket, hourgroup])
        ),
        w_loss_hg,
    )
    add(
        "temp_cloud_hourgroup",
        (profile.get("temp_cloud_hourgroup_mean", {}) or {}).get(
            lookup_key([temp_bucket, cloud_bucket, hourgroup])
        ),
        w_temp_cloud_hg,
    )

    if vals:
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
        correction_scale = _recent_hot_peak_scale(
            row, c
        ) * _recent_horizon_regime_scale(row, c, horizon_index)
        base_correction = float(
            np.clip(raw * blend * decay * correction_scale, -cap, cap)
        )
        base_source = "+".join(name for name, _, _ in vals)
    else:
        base_correction = 0.0
        base_source = "no_match"

    return _combine_recent_and_ar_corrections(
        base_correction, base_source, row, profile, c, horizon_index
    )


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
    ar_corrections = []
    ar_phis = []
    ar_latest_residuals = []
    ar_sources = []
    origin_day_corrections = []
    origin_day_states = []
    origin_day_latest_days = []
    origin_day_sources = []
    for i, row in out.iterrows():
        (
            corr,
            source,
            ar_corr,
            ar_phi,
            ar_latest,
            ar_source,
            origin_day_corr,
            origin_day_state,
            origin_day_latest,
            origin_day_source,
        ) = _weighted_recent_correction(row, profile or {}, config, horizon_index=i + 1)
        corrections.append(corr)
        sources.append(source)
        ar_corrections.append(ar_corr)
        ar_phis.append(ar_phi)
        ar_latest_residuals.append(ar_latest)
        ar_sources.append(ar_source)
        origin_day_corrections.append(origin_day_corr)
        origin_day_states.append(origin_day_state)
        origin_day_latest_days.append(origin_day_latest)
        origin_day_sources.append(origin_day_source)

    out["Recent_Level_Correction_MWH"] = corrections
    out["Recent_Correction_Source"] = sources
    out["AR_Residual_Correction_MWH"] = ar_corrections
    out["AR_Residual_Phi"] = ar_phis
    out["AR_Residual_Latest_MWH"] = ar_latest_residuals
    out["AR_Residual_Source"] = ar_sources
    out["OriginDay_State_Correction_MWH"] = origin_day_corrections
    out["OriginDay_State_MWH"] = origin_day_states
    out["OriginDay_Latest_Day_MWH"] = origin_day_latest_days
    out["OriginDay_State_Source"] = origin_day_sources
    out["Pre_Recent_Forecast_MWH"] = pd.to_numeric(out[base_col], errors="coerce")
    out["Recent_Corrected_Forecast_MWH"] = (
        out["Pre_Recent_Forecast_MWH"] + out["Recent_Level_Correction_MWH"]
    ).clip(lower=0.0)
    out["Final_Forecast_MWH"] = out["Recent_Corrected_Forecast_MWH"]
    # Preserve the existing dashboard/output contract: Calibrated_Forecast_MWH is the final production forecast.
    out["Calibrated_Forecast_MWH"] = out["Final_Forecast_MWH"]
    return out


def _factorize_series(s: pd.Series) -> tuple[np.ndarray, int]:
    codes, uniques = pd.factorize(s, sort=False, use_na_sentinel=True)
    return codes, len(uniques)


def _combo_codes(*parts: pd.Series) -> tuple[np.ndarray, int]:
    mask = pd.Series(True, index=parts[0].index)
    for p in parts:
        mask = mask & p.notna()
    joined = parts[0].astype(str)
    for p in parts[1:]:
        joined = joined + "\x00" + p.astype(str)
    combo = joined.where(mask, other=np.nan)
    return _factorize_series(combo)


def _range_source_means(
    valid_mask: np.ndarray,
    resid_valid: np.ndarray,
    row_codes: np.ndarray,
    n_categories: int,
    left_idx: np.ndarray,
    right_idx: np.ndarray,
    cap: float,
) -> np.ndarray:
    """Causal (prior-rows-only), category-grouped mean of resid_valid over [left_idx, right_idx)
    positions of the valid-residual subsequence, evaluated per row against that row's own
    category code. NaN where the row's own category is missing (code < 0) or the range has
    no matching observations."""
    codes_valid = row_codes[valid_mask]
    n_valid = len(resid_valid)
    n_cat = max(1, n_categories)
    contrib_sum = np.zeros((n_valid, n_cat))
    contrib_cnt = np.zeros((n_valid, n_cat))
    ok = codes_valid >= 0
    idx = np.nonzero(ok)[0]
    contrib_sum[idx, codes_valid[ok]] = resid_valid[ok]
    contrib_cnt[idx, codes_valid[ok]] = 1.0
    csum = np.vstack([np.zeros((1, n_cat)), np.cumsum(contrib_sum, axis=0)])
    ccnt = np.vstack([np.zeros((1, n_cat)), np.cumsum(contrib_cnt, axis=0)])

    k = np.clip(row_codes, 0, n_cat - 1)
    total = csum[right_idx, k] - csum[left_idx, k]
    count = ccnt[right_idx, k] - ccnt[left_idx, k]
    safe_count = np.where(count > 0, count, 1.0)
    mean = np.where((count > 0) & (row_codes >= 0), total / safe_count, np.nan)
    return np.clip(mean, -cap, cap)


def simulate_recent_residual_correction_backtest(
    backtest_df: pd.DataFrame,
    config: dict | None = None,
    base_col: str = "Raw_Forecast_MWH",
) -> pd.DataFrame:
    """Simulate the recent correction over a backtest using only prior residuals for each row.

    Vectorized: for every row this used to slice a growing DataFrame (out.iloc[:i]) and run
    ~8 boolean-mask filters against it, which is O(n^2)-ish in pandas overhead and became the
    dominant cost once this ran on multi-hundred-day extended-lookback frames (Optuna search,
    hot_ramp/heat_persistence lookup basis). It now computes the same causal, per-source means
    via prefix-sum range queries over a factorized category index, with no per-row DataFrame
    slicing. The AR-residual / origin-day-state carryover (both disabled by default) still need
    a growing history and fall back to the original per-row loop only when either is enabled.
    """
    c = _cfg(config)
    out = backtest_df.copy().sort_values("DT").reset_index(drop=True)
    if (
        out.empty
        or not bool(c.get("enabled", True))
        or not {"Actual_MWH", base_col}.issubset(out.columns)
    ):
        out["Recent_Level_Correction_MWH"] = 0.0
        out["Recent_Correction_Source"] = "disabled_or_empty"
        out["AR_Residual_Correction_MWH"] = 0.0
        out["AR_Residual_Phi"] = np.nan
        out["AR_Residual_Latest_MWH"] = np.nan
        out["AR_Residual_Source"] = "ar_disabled_or_empty"
        out["OriginDay_State_Correction_MWH"] = 0.0
        out["OriginDay_State_MWH"] = np.nan
        out["OriginDay_Latest_Day_MWH"] = np.nan
        out["OriginDay_State_Source"] = "origin_day_disabled_or_empty"
        out["Recent_Corrected_Forecast_MWH"] = pd.to_numeric(
            out.get(base_col, out.get("Raw_Forecast_MWH", 0.0)), errors="coerce"
        )
        out["Final_Backtest_Forecast_MWH"] = out["Recent_Corrected_Forecast_MWH"]
        return out

    out["DT"] = pd.to_datetime(out["DT"], errors="coerce")
    out["Hour"] = (
        _as_num(out.get("Hour", out["DT"].dt.hour))
        .fillna(out["DT"].dt.hour)
        .astype(int)
    )
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

    n = len(out)
    valid_mask = out["_RecentBasisResidual"].notna().to_numpy()
    resid_valid = out.loc[valid_mask, "_RecentBasisResidual"].to_numpy(dtype=float)
    valid_dt = out.loc[valid_mask, "DT"].to_numpy()

    cumsum_valid = np.cumsum(valid_mask.astype(np.int64))
    pos_i = np.concatenate([[0], cumsum_valid[:-1]]) if n else np.array([], dtype=np.int64)

    thresholds = (out["DT"] - pd.Timedelta(days=max(1, same_hour_days))).to_numpy()
    if len(valid_dt):
        window_left_raw = np.searchsorted(valid_dt, thresholds, side="left")
    else:
        window_left_raw = np.zeros(n, dtype=np.int64)
    window_left_raw = np.minimum(window_left_raw, pos_i)
    window_empty = window_left_raw >= pos_i
    window_left = np.where(window_empty, 0, window_left_raw)
    window_right = pos_i

    zero_codes = np.zeros(n, dtype=np.int64)
    hour_codes, n_hour = _factorize_series(out["Hour"])
    hourgroup_codes, n_hourgroup = _factorize_series(out["HourGroup"])
    temp_hg_codes, n_temp_hg = _combo_codes(out["DailyMaxTempBucket"], out["HourGroup"])
    cloud_hg_codes, n_cloud_hg = _combo_codes(out["CloudCoverBucket"], out["HourGroup"])
    solar_hg_codes, n_solar_hg = _combo_codes(out["BTMSolarBucket"], out["HourGroup"])
    loss_hg_codes, n_loss_hg = _combo_codes(out["SolarLossBucket"], out["HourGroup"])
    temp_cloud_hg_codes, n_temp_cloud_hg = _combo_codes(
        out["DailyMaxTempBucket"], out["CloudCoverBucket"], out["HourGroup"]
    )

    recent_left = np.maximum(0, pos_i - recent_hours)
    last24_left = np.maximum(0, pos_i - 24)

    sources_spec = [
        ("recent_mean", zero_codes, 1, recent_left, pos_i, w_recent),
        ("last24_mean", zero_codes, 1, last24_left, pos_i, w_last24),
        ("same_hour", hour_codes, n_hour, window_left, window_right, w_same),
        ("hourgroup", hourgroup_codes, n_hourgroup, window_left, window_right, w_hourgroup),
        ("global", zero_codes, 1, np.zeros(n, dtype=np.int64), pos_i, w_global),
        (
            "temp_hourgroup",
            temp_hg_codes,
            n_temp_hg,
            window_left,
            window_right,
            w_temp_hg,
        ),
        (
            "cloud_hourgroup",
            cloud_hg_codes,
            n_cloud_hg,
            window_left,
            window_right,
            w_cloud_hg,
        ),
        (
            "solar_hourgroup",
            solar_hg_codes,
            n_solar_hg,
            window_left,
            window_right,
            w_solar_hg,
        ),
        (
            "solar_loss_hourgroup",
            loss_hg_codes,
            n_loss_hg,
            window_left,
            window_right,
            w_loss_hg,
        ),
        (
            "temp_cloud_hourgroup",
            temp_cloud_hg_codes,
            n_temp_cloud_hg,
            window_left,
            window_right,
            w_temp_cloud_hg,
        ),
    ]

    values = np.full((n, len(sources_spec)), np.nan)
    weights_arr = np.array([spec[5] for spec in sources_spec], dtype=float)
    for col, (name, codes, n_cat, left, right, weight) in enumerate(sources_spec):
        if weight <= 0:
            continue
        values[:, col] = _range_source_means(
            valid_mask, resid_valid, codes, n_cat, left, right, cap
        )

    present = np.isfinite(values) & (weights_arr[None, :] > 0)
    weighted_vals = np.where(present, values * weights_arr[None, :], 0.0)
    denom = np.where(present, weights_arr[None, :], 0.0).sum(axis=1)
    numer = weighted_vals.sum(axis=1)
    has_match = denom > 0
    raw = np.where(has_match, numer / np.where(has_match, denom, 1.0), 0.0)

    names = [spec[0] for spec in sources_spec]
    joined_names = np.empty(n, dtype=object)
    for i in range(n):
        joined_names[i] = "+".join(
            name for name, is_present in zip(names, present[i]) if is_present
        )

    correction_scale = np.ones(n)
    matched_idx = np.nonzero(has_match)[0]
    if matched_idx.size:
        scale_frame = out.iloc[matched_idx]
        computed_scale = scale_frame.apply(
            lambda row: _recent_hot_peak_scale(row, c) * _recent_horizon_regime_scale(row, c),
            axis=1,
        )
        correction_scale[matched_idx] = computed_scale.to_numpy()

    base_correction = np.where(
        has_match, np.clip(raw * blend * correction_scale, -cap, cap), 0.0
    )
    base_source = np.where(
        pos_i == 0,
        "insufficient_prior_residuals",
        np.where(has_match, joined_names, "no_match"),
    )

    ar_cfg = _ar_config(c)
    origin_cfg = _origin_day_config(c)
    ar_enabled = bool(ar_cfg.get("enabled", False))
    origin_enabled = bool(origin_cfg.get("enabled", False))

    if not ar_enabled and not origin_enabled:
        corrections = base_correction
        sources = base_source
        ar_corrections = np.zeros(n)
        ar_phis = np.full(n, np.nan)
        ar_latest_residuals = np.full(n, np.nan)
        ar_sources = np.full(n, "ar_disabled_or_empty", dtype=object)
        origin_day_corrections = np.zeros(n)
        origin_day_states = np.full(n, np.nan)
        origin_day_latest_days = np.full(n, np.nan)
        origin_day_sources = np.full(n, "origin_day_disabled_or_empty", dtype=object)
    else:
        corrections = np.empty(n)
        sources = np.empty(n, dtype=object)
        ar_corrections = np.empty(n)
        ar_phis = np.empty(n)
        ar_latest_residuals = np.empty(n)
        ar_sources = np.empty(n, dtype=object)
        origin_day_corrections = np.empty(n)
        origin_day_states = np.empty(n)
        origin_day_latest_days = np.empty(n)
        origin_day_sources = np.empty(n, dtype=object)
        for i in range(n):
            if pos_i[i] == 0:
                corrections[i] = 0.0
                sources[i] = "insufficient_prior_residuals"
                ar_corrections[i] = 0.0
                ar_phis[i] = np.nan
                ar_latest_residuals[i] = np.nan
                ar_sources[i] = "ar_disabled_or_empty"
                origin_day_corrections[i] = 0.0
                origin_day_states[i] = np.nan
                origin_day_latest_days[i] = np.nan
                origin_day_sources[i] = "origin_day_disabled_or_empty"
                continue
            hist = out.iloc[:i]
            hist = hist[hist["_RecentBasisResidual"].notna()]
            row = out.iloc[i]
            residual_state_profile = {
                "enabled": True,
                "ar_residual": _build_ar_residual_state(
                    hist, c, residual_col="_RecentBasisResidual"
                ),
                "origin_day_state": _build_origin_day_state(
                    hist, c, residual_col="_RecentBasisResidual"
                ),
            }
            (
                correction,
                source,
                ar_corr,
                ar_phi,
                ar_latest,
                ar_source,
                origin_day_corr,
                origin_day_state,
                origin_day_latest,
                origin_day_source,
            ) = _combine_recent_and_ar_corrections(
                base_correction[i],
                base_source[i],
                row,
                residual_state_profile,
                c,
                horizon_index=1,
            )
            corrections[i] = correction
            sources[i] = source
            ar_corrections[i] = ar_corr
            ar_phis[i] = ar_phi
            ar_latest_residuals[i] = ar_latest
            ar_sources[i] = ar_source
            origin_day_corrections[i] = origin_day_corr
            origin_day_states[i] = origin_day_state
            origin_day_latest_days[i] = origin_day_latest
            origin_day_sources[i] = origin_day_source

    out["Recent_Level_Correction_MWH"] = corrections
    out["Recent_Correction_Source"] = sources
    out["AR_Residual_Correction_MWH"] = ar_corrections
    out["AR_Residual_Phi"] = ar_phis
    out["AR_Residual_Latest_MWH"] = ar_latest_residuals
    out["AR_Residual_Source"] = ar_sources
    out["OriginDay_State_Correction_MWH"] = origin_day_corrections
    out["OriginDay_State_MWH"] = origin_day_states
    out["OriginDay_Latest_Day_MWH"] = origin_day_latest_days
    out["OriginDay_State_Source"] = origin_day_sources
    out["Pre_Recent_Forecast_MWH"] = _as_num(out[base_col])
    out["Recent_Corrected_Forecast_MWH"] = (
        out["Pre_Recent_Forecast_MWH"] + out["Recent_Level_Correction_MWH"]
    ).clip(lower=0.0)
    out["Final_Backtest_Forecast_MWH"] = out["Recent_Corrected_Forecast_MWH"]
    out["Final_Forecast_MWH"] = out["Recent_Corrected_Forecast_MWH"]
    out["Recent_Corrected_Residual_MWH"] = _as_num(out["Actual_MWH"]) - _as_num(
        out["Recent_Corrected_Forecast_MWH"]
    )
    out["Recent_Corrected_AbsError_MWH"] = out["Recent_Corrected_Residual_MWH"].abs()
    out["Recent_Corrected_APE"] = np.where(
        _as_num(out["Actual_MWH"]).abs() > 1e-9,
        out["Recent_Corrected_AbsError_MWH"] / _as_num(out["Actual_MWH"]).abs() * 100.0,
        np.nan,
    )
    return out.drop(columns=["_RecentBasisResidual"], errors="ignore")
