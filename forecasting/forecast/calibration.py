from __future__ import annotations

import pandas as pd
import numpy as np


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


def _season_from_month(m: int) -> str:
    if m in (12, 1, 2):
        return "Winter"
    if m in (3, 4, 5):
        return "Spring"
    if m in (6, 7, 8, 9):
        return "Summer"
    if m in (10, 11):
        return "Fall"
    return "Unknown"


def _prep_context(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    out["DT"] = pd.to_datetime(out["DT"])
    if "Month" not in out.columns:
        out["Month"] = out["DT"].dt.month
    if "Hour" not in out.columns:
        out["Hour"] = out["DT"].dt.hour
    if "Season" not in out.columns:
        out["Season"] = out["Month"].map(_season_from_month)
    if "HourGroup" not in out.columns:
        out["HourGroup"] = out["Hour"].map(_hour_group)
    if "IsWeekend" not in out.columns:
        out["IsWeekend"] = out["DT"].dt.dayofweek.ge(5).astype(int)
    else:
        out["IsWeekend"] = pd.to_numeric(out["IsWeekend"], errors="coerce").fillna(0).astype(int)
    if "DailyMaxTempBin" not in out.columns:
        if "Temperature_DailyMax" in out.columns:
            bins = [-999, 65, 75, 85, 90, 95, 100, 105, 999]
            out["DailyMaxTempBin"] = pd.cut(out["Temperature_DailyMax"], bins=bins, labels=False, include_lowest=True).astype(float)
        elif "Temperature" in out.columns:
            out["Date"] = out["DT"].dt.date
            mx = out.groupby("Date")["Temperature"].transform("max")
            out["DailyMaxTempBin"] = pd.cut(mx, bins=[-999, 65, 75, 85, 90, 95, 100, 105, 999], labels=False, include_lowest=True).astype(float)
        else:
            out["DailyMaxTempBin"] = np.nan

    cloud = pd.to_numeric(out.get("CloudCover_Norm", pd.Series(np.nan, index=out.index)), errors="coerce")
    if cloud.notna().any() and cloud.max(skipna=True) > 1.5:
        cloud = cloud / 100.0
    out["CloudCoverBin"] = pd.cut(
        cloud.clip(0, 1), bins=[-0.001, 0.20, 0.40, 0.60, 0.80, 1.001],
        labels=False, include_lowest=True,
    ).astype(float)
    if "CloudCoverBucket" not in out.columns:
        out["CloudCoverBucket"] = pd.cut(
            cloud.clip(0, 1), bins=[-0.001, 0.20, 0.40, 0.60, 0.80, 1.001],
            labels=["Clear/Low", "Some Clouds", "Partly Cloudy", "Mostly Cloudy", "Overcast"], include_lowest=True,
        ).astype("object")

    solar = pd.to_numeric(out.get("BTM_Solar_Proxy_MW", pd.Series(0.0, index=out.index)), errors="coerce").fillna(0.0)
    out["BTMSolarProxyBin"] = pd.cut(
        solar, bins=[-0.001, 1, 5, 10, 15, 20, 30, 9999],
        labels=False, include_lowest=True,
    ).astype(float)
    if "SolarLossBucket" not in out.columns:
        loss_col = "BTM_Solar_Loss_From_ClearSky_MW" if "BTM_Solar_Loss_From_ClearSky_MW" in out.columns else "Midday_Overcast_Solar_Loss_MW"
        if loss_col in out.columns:
            loss = pd.to_numeric(out[loss_col], errors="coerce").fillna(0.0)
            out["SolarLossBucket"] = "None"
            pos = loss[loss > 0]
            if len(pos) >= 10 and pos.nunique() >= 4:
                q1, q2, q3 = pos.quantile([0.25, 0.50, 0.75]).tolist()
                out.loc[(loss > 0) & (loss <= q1), "SolarLossBucket"] = "Low"
                out.loc[(loss > q1) & (loss <= q2), "SolarLossBucket"] = "Medium"
                out.loc[(loss > q2) & (loss <= q3), "SolarLossBucket"] = "High"
                out.loc[loss > q3, "SolarLossBucket"] = "Extreme"
            elif len(pos):
                med = pos.median()
                out.loc[(loss > 0) & (loss <= med), "SolarLossBucket"] = "Low/Medium"
                out.loc[loss > med, "SolarLossBucket"] = "High"
        else:
            out["SolarLossBucket"] = np.nan
    return out


def _residual_work(backtest_df: pd.DataFrame) -> pd.DataFrame:
    work = _prep_context(backtest_df)
    if "Residual_MWH" in work.columns:
        work["Residual"] = pd.to_numeric(work["Residual_MWH"], errors="coerce")
    else:
        work["Residual"] = pd.to_numeric(work["Actual_MWH"], errors="coerce") - pd.to_numeric(work["Raw_Forecast_MWH"], errors="coerce")
    return work.dropna(subset=["Residual"]).copy()


def _hot_peak_season_scale(out: pd.DataFrame, hot_peak_cfg: dict | None) -> pd.Series:
    cfg = hot_peak_cfg or {}
    scale = pd.Series(float(cfg.get("default_scale", 1.0)), index=out.index, dtype=float)
    season_scales = cfg.get("season_scales", {}) or {}
    if not season_scales or "Temperature_DailyMax" not in out.columns:
        return scale
    hours = [int(hour) for hour in cfg.get("hours", [16, 17, 18, 19, 20])]
    hot_peak = (
        out["Hour"].isin(hours)
        & pd.to_numeric(out["Temperature_DailyMax"], errors="coerce").ge(float(cfg.get("min_maxtemp_f", 90.0)))
    )
    for season, season_scale in season_scales.items():
        scale.loc[hot_peak & out["Season"].astype(str).eq(str(season))] = float(season_scale)
    return scale


def _make_lookup(work: pd.DataFrame, keys: list[str], blend: float, cap_mwh: float, shrink_k: float) -> pd.DataFrame:
    if keys:
        grp = work.groupby(keys, dropna=False)["Residual"].agg(["mean", "count", "std"]).reset_index()
    else:
        grp = pd.DataFrame({
            "mean": [work["Residual"].mean()],
            "count": [int(work["Residual"].count())],
            "std": [work["Residual"].std()],
        })
    grp["shrink_weight"] = grp["count"] / (grp["count"] + float(shrink_k))
    grp["correction"] = (grp["mean"] * grp["shrink_weight"] * float(blend)).clip(-float(cap_mwh), float(cap_mwh))
    grp["abs_correction"] = grp["correction"].abs()
    return grp[keys + ["correction", "mean", "count", "std", "shrink_weight", "abs_correction"]]


def build_learned_residual_lookups(backtest_df: pd.DataFrame, blend: float, cap_mwh: float, shrink_k: float) -> dict:
    """Build multi-level learned residual corrections from a true holdout backtest.

    V12 changes the application strategy from most-specific-first to weighted fallback. The prior approach
    could let a sparse, tiny correction suppress broad and obvious bias. Weighted fallback allows broad
    hour/temp/cloud bias to keep influencing the final adjustment.
    """
    if backtest_df is None or backtest_df.empty:
        return {}

    work = _residual_work(backtest_df)
    if work.empty:
        return {}

    specs = [
        ("global", []),
        ("hour", ["Hour"]),
        ("season_hour", ["Season", "Hour"]),
        ("season_hourgroup_maxtemp", ["Season", "HourGroup", "DailyMaxTempBin"]),
        ("season_hour_maxtemp", ["Season", "Hour", "DailyMaxTempBin"]),
        ("season_month_hour_maxtemp", ["Season", "Month", "Hour", "DailyMaxTempBin"]),
        ("season_cloud_hourgroup", ["Season", "CloudCoverBin", "HourGroup"]),
        ("season_solar_hourgroup", ["Season", "BTMSolarProxyBin", "HourGroup"]),
        ("temp_cloud_hourgroup", ["DailyMaxTempBin", "CloudCoverBin", "HourGroup"]),
    ]
    return {
        "application_strategy": "weighted_fallback",
        "ordered_levels": [
            {"name": name, "keys": keys, "lookup": _make_lookup(work, keys, blend, cap_mwh, shrink_k)}
            for name, keys in specs
        ],
        "metadata": {
            "mean_residual": float(work["Residual"].mean()),
            "mae": float(work["Residual"].abs().mean()),
            "rmse": float(np.sqrt(np.mean(np.square(work["Residual"])))) if len(work) else np.nan,
            "underforecast_rate_pct": float((work["Residual"] > 0).mean() * 100.0),
        },
    }


def _default_level_weights() -> dict[str, float]:
    return {
        "global": 0.05,
        "hour": 0.08,
        "season_hour": 0.12,
        "season_hourgroup_maxtemp": 0.18,
        "season_hour_maxtemp": 0.20,
        "season_month_hour_maxtemp": 0.18,
        "season_cloud_hourgroup": 0.09,
        "season_solar_hourgroup": 0.05,
        "temp_cloud_hourgroup": 0.05,
    }


def apply_learned_calibration(
    future_df: pd.DataFrame,
    lookup_bundle: dict | None,
    level_weights: dict[str, float] | None = None,
    cap_mwh: float | None = None,
    base_col: str = "Raw_Forecast_MWH",
    hot_peak_cfg: dict | None = None,
) -> pd.DataFrame:
    out = _prep_context(future_df)
    if base_col not in out.columns:
        base_col = "Raw_Forecast_MWH"
    out["Residual_Cal_MWH"] = 0.0
    out["Calibration_Level"] = "none"
    out["Calibration_Matched_Levels"] = ""

    if not lookup_bundle or not lookup_bundle.get("ordered_levels"):
        out["Residual_Calibrated_Forecast_MWH"] = out[base_col].astype(float).clip(lower=0.0)
        out["Calibrated_Forecast_MWH"] = out["Residual_Calibrated_Forecast_MWH"]
        return out

    weights = _default_level_weights()
    if level_weights:
        weights.update({str(k): float(v) for k, v in level_weights.items()})

    weighted_sum = pd.Series(0.0, index=out.index)
    weight_sum = pd.Series(0.0, index=out.index)
    matched_names = {idx: [] for idx in out.index}

    for level in lookup_bundle["ordered_levels"]:
        name = str(level.get("name", "level"))
        keys = list(level.get("keys", []))
        lookup = level.get("lookup")
        lw = float(weights.get(name, 0.0))
        if lw <= 0 or lookup is None or lookup.empty or not all(k in out.columns for k in keys):
            continue

        if keys:
            tmp = out[keys].reset_index().merge(lookup[keys + ["correction", "count"]], on=keys, how="left")
        else:
            # Global lookup: same correction applies to every row.
            correction = float(lookup["correction"].iloc[0]) if "correction" in lookup else 0.0
            count = float(lookup["count"].iloc[0]) if "count" in lookup else np.nan
            tmp = pd.DataFrame({"index": out.index, "correction": correction, "count": count})

        matched = tmp["correction"].notna()
        if not matched.any():
            continue
        idx = tmp.loc[matched, "index"]
        corr = pd.to_numeric(tmp.loc[matched, "correction"], errors="coerce").fillna(0.0).to_numpy(dtype=float)
        # Let higher-count lookups have a bit more influence without letting them dominate.
        cnt = pd.to_numeric(tmp.loc[matched, "count"], errors="coerce").fillna(0.0).clip(lower=0.0).to_numpy(dtype=float)
        count_factor = np.sqrt(cnt / (cnt + 24.0))
        eff_w = lw * np.where(np.isfinite(count_factor), count_factor, 1.0)
        weighted_sum.loc[idx] += corr * eff_w
        weight_sum.loc[idx] += eff_w
        for ix in idx:
            matched_names[ix].append(name)

    mask = weight_sum > 0
    out.loc[mask, "Residual_Cal_MWH"] = (weighted_sum.loc[mask] / weight_sum.loc[mask]).astype(float)
    if cap_mwh is not None:
        out["Residual_Cal_MWH"] = out["Residual_Cal_MWH"].clip(-float(cap_mwh), float(cap_mwh))
    out["Residual_Cal_MWH"] = out["Residual_Cal_MWH"] * _hot_peak_season_scale(out, hot_peak_cfg)
    out.loc[mask, "Calibration_Level"] = "weighted_fallback"
    out["Calibration_Matched_Levels"] = ["+".join(matched_names[idx]) for idx in out.index]
    out["Residual_Calibrated_Forecast_MWH"] = (out[base_col].astype(float) + out["Residual_Cal_MWH"].astype(float)).clip(lower=0.0)
    out["Calibrated_Forecast_MWH"] = out["Residual_Calibrated_Forecast_MWH"]
    return out


def build_heat_peak_lookup(backtest_df: pd.DataFrame, min_maxtemp_f: float, hours: list[int], blend: float, cap_mwh: float, shrink_k: float) -> pd.DataFrame | None:
    if backtest_df is None or backtest_df.empty:
        return None
    work = _residual_work(backtest_df)
    if "Temperature_DailyMax" not in work.columns:
        return None
    mask = (pd.to_numeric(work["Temperature_DailyMax"], errors="coerce") >= float(min_maxtemp_f)) & (work["Hour"].isin([int(h) for h in hours]))
    work = work.loc[mask].dropna(subset=["Residual"])
    if work.empty:
        return None
    return _make_lookup(work, ["Hour", "DailyMaxTempBin"], blend, cap_mwh, shrink_k)


def apply_heat_peak_calibration(future_df: pd.DataFrame, heat_lookup: pd.DataFrame | None, min_maxtemp_f: float, hours: list[int]) -> pd.DataFrame:
    out = _prep_context(future_df)
    out["Heat_Peak_Cal_MWH"] = 0.0
    out["Heat_Adjusted_Forecast_MWH"] = out.get("Calibrated_Forecast_MWH", out["Raw_Forecast_MWH"]).astype(float)
    if heat_lookup is None or heat_lookup.empty or "Temperature_DailyMax" not in out.columns:
        out["Calibrated_Forecast_MWH"] = out["Heat_Adjusted_Forecast_MWH"]
        return out
    mask = (pd.to_numeric(out["Temperature_DailyMax"], errors="coerce") >= float(min_maxtemp_f)) & (out["Hour"].isin([int(h) for h in hours]))
    tmp = out.loc[mask, ["Hour", "DailyMaxTempBin"]].reset_index().merge(
        heat_lookup[["Hour", "DailyMaxTempBin", "correction"]], on=["Hour", "DailyMaxTempBin"], how="left"
    )
    matched = tmp["correction"].notna()
    if matched.any():
        idx = tmp.loc[matched, "index"]
        out.loc[idx, "Heat_Peak_Cal_MWH"] = tmp.loc[matched, "correction"].to_numpy(dtype=float)
        out.loc[idx, "Calibration_Level"] = out.loc[idx, "Calibration_Level"].astype(str) + "+heat_peak"
    out["Heat_Adjusted_Forecast_MWH"] = (
        out.get("Calibrated_Forecast_MWH", out["Raw_Forecast_MWH"]).astype(float) + out["Heat_Peak_Cal_MWH"].astype(float)
    ).clip(lower=0.0)
    out["Calibrated_Forecast_MWH"] = out["Heat_Adjusted_Forecast_MWH"]
    return out


def build_warm_ramp_lookup(
    backtest_df: pd.DataFrame,
    min_maxtemp_f: float = 75.0,
    max_maxtemp_f: float = 93.0,
    hours: list[int] | None = None,
    blend: float = 0.70,
    cap_mwh: float = 12.0,
    shrink_k: float = 15.0,
) -> dict | None:
    """Learn spring/warm shoulder-season ramp residuals.

    This targets the pattern found in the uploaded diagnostics: 75-85F and 85-90F spring days were
    consistently low in peak/late-evening hours.
    """
    if backtest_df is None or backtest_df.empty:
        return None
    hours = [int(h) for h in (hours or list(range(12, 23)))]
    work = _residual_work(backtest_df)
    if "Temperature_DailyMax" not in work.columns:
        return None
    temp = pd.to_numeric(work["Temperature_DailyMax"], errors="coerce")
    mask = (
        work["Season"].isin(["Spring", "Fall"])
        & temp.ge(float(min_maxtemp_f))
        & temp.le(float(max_maxtemp_f))
        & work["Hour"].isin(hours)
    )
    work = work.loc[mask].dropna(subset=["Residual"])
    if work.empty:
        return None
    specs = [
        ("warm_season_hour_temp_cloud", ["Season", "Hour", "DailyMaxTempBin", "CloudCoverBin"]),
        ("warm_weekend_hour_temp", ["IsWeekend", "Hour", "DailyMaxTempBin"]),
        ("warm_hour_temp_cloud", ["Hour", "DailyMaxTempBin", "CloudCoverBin"]),
        ("warm_season_hour_temp", ["Season", "Hour", "DailyMaxTempBin"]),
        ("warm_season_hourgroup_temp", ["Season", "HourGroup", "DailyMaxTempBin"]),
        ("warm_hour_temp", ["Hour", "DailyMaxTempBin"]),
        ("warm_hourgroup_temp", ["HourGroup", "DailyMaxTempBin"]),
    ]
    return {
        "ordered_levels": [{"name": name, "keys": keys, "lookup": _make_lookup(work, keys, blend, cap_mwh, shrink_k)} for name, keys in specs],
        "metadata": {
            "n": int(len(work)),
            "mean_residual": float(work["Residual"].mean()),
            "mae": float(work["Residual"].abs().mean()),
            "min_maxtemp_f": float(min_maxtemp_f),
            "max_maxtemp_f": float(max_maxtemp_f),
            "hours": hours,
        },
    }


def apply_warm_ramp_correction(
    future_df: pd.DataFrame,
    warm_lookup: dict | None,
    min_maxtemp_f: float = 75.0,
    max_maxtemp_f: float = 93.0,
    hours: list[int] | None = None,
    cap_mwh: float = 12.0,
) -> pd.DataFrame:
    """Apply warm shoulder-season ramp correction using weighted fallback.

    V12.5 changed this from most-specific-first to weighted fallback so a sparse
    low-count lookup cannot suppress the broad warm-evening bias seen in V12.4.
    """
    out = _prep_context(future_df)
    out["Warm_Ramp_Cal_MWH"] = 0.0
    out["Warm_Ramp_Adjusted_Forecast_MWH"] = out.get("Calibrated_Forecast_MWH", out["Raw_Forecast_MWH"]).astype(float)
    out["Warm_Ramp_Correction_Source"] = "none"
    if not warm_lookup or not warm_lookup.get("ordered_levels") or "Temperature_DailyMax" not in out.columns:
        out["Calibrated_Forecast_MWH"] = out["Warm_Ramp_Adjusted_Forecast_MWH"]
        return out
    hours = [int(h) for h in (hours or list(range(12, 23)))]
    temp = pd.to_numeric(out["Temperature_DailyMax"], errors="coerce")
    eligible = (
        out["Season"].isin(["Spring", "Fall"])
        & temp.ge(float(min_maxtemp_f))
        & temp.le(float(max_maxtemp_f))
        & out["Hour"].isin(hours)
    )
    if not eligible.any():
        out["Calibrated_Forecast_MWH"] = out["Warm_Ramp_Adjusted_Forecast_MWH"]
        return out

    weights = {
        "warm_season_hour_temp_cloud": 0.24,
        "warm_weekend_hour_temp": 0.16,
        "warm_hour_temp_cloud": 0.14,
        "warm_season_hour_temp": 0.18,
        "warm_season_hourgroup_temp": 0.12,
        "warm_hour_temp": 0.10,
        "warm_hourgroup_temp": 0.06,
    }
    weighted_sum = pd.Series(0.0, index=out.index)
    weight_sum = pd.Series(0.0, index=out.index)
    matched_names = {idx: [] for idx in out.index}

    for level in warm_lookup["ordered_levels"]:
        name = str(level.get("name", "level"))
        keys = level.get("keys", [])
        lookup = level.get("lookup")
        lw = float(weights.get(name, 0.0))
        if lw <= 0 or lookup is None or lookup.empty or not all(k in out.columns for k in keys):
            continue
        tmp = out.loc[eligible, keys].reset_index().merge(lookup[keys + ["correction", "count"]], on=keys, how="left")
        matched = tmp["correction"].notna()
        if matched.any():
            idx = tmp.loc[matched, "index"]
            corr = pd.to_numeric(tmp.loc[matched, "correction"], errors="coerce").fillna(0.0).to_numpy(dtype=float)
            cnt = pd.to_numeric(tmp.loc[matched, "count"], errors="coerce").fillna(0.0).clip(lower=0.0).to_numpy(dtype=float)
            count_factor = np.sqrt(cnt / (cnt + 10.0))
            eff_w = lw * np.where(np.isfinite(count_factor), count_factor, 1.0)
            weighted_sum.loc[idx] += corr * eff_w
            weight_sum.loc[idx] += eff_w
            for ix in idx:
                matched_names[ix].append(name)

    mask = eligible & (weight_sum > 0)
    out.loc[mask, "Warm_Ramp_Cal_MWH"] = (weighted_sum.loc[mask] / weight_sum.loc[mask]).clip(-float(cap_mwh), float(cap_mwh))
    out.loc[mask, "Warm_Ramp_Correction_Source"] = ["+".join(matched_names[idx]) for idx in out.loc[mask].index]
    out["Warm_Ramp_Adjusted_Forecast_MWH"] = (
        out.get("Calibrated_Forecast_MWH", out["Raw_Forecast_MWH"]).astype(float) + out["Warm_Ramp_Cal_MWH"].astype(float)
    ).clip(lower=0.0)
    out["Calibrated_Forecast_MWH"] = out["Warm_Ramp_Adjusted_Forecast_MWH"]
    return out


# Backward-compatible wrappers used by earlier versions.
def build_learned_seasonal_lookup(backtest_df: pd.DataFrame, blend: float, cap_mwh: float, shrink_k: float):
    bundle = build_learned_residual_lookups(backtest_df, blend, cap_mwh, shrink_k)
    if not bundle:
        return None
    for level in bundle.get("ordered_levels", []):
        if level["name"] == "season_hour":
            return level["lookup"][["Season", "Hour", "correction"]]
    return None


def apply_seasonal_calibration(future_df: pd.DataFrame, lookup: pd.DataFrame) -> pd.DataFrame:
    if lookup is None or lookup.empty:
        future_df["Seasonal_Cal_MWH"] = 0.0
        future_df["Calibrated_Forecast_MWH"] = future_df["Raw_Forecast_MWH"].astype(float).clip(lower=0.0)
        return future_df
    out = future_df.merge(lookup, on=["Season", "Hour"], how="left")
    out["Seasonal_Cal_MWH"] = out["correction"].fillna(0.0)
    out["Calibrated_Forecast_MWH"] = (out["Raw_Forecast_MWH"].astype(float) + out["Seasonal_Cal_MWH"]).clip(lower=0.0)
    return out
