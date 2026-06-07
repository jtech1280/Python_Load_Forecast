from __future__ import annotations

import numpy as np
import pandas as pd


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


def _as_num(x):
    return pd.to_numeric(x, errors="coerce")


def _temp_bin(values: pd.Series) -> pd.Series:
    return pd.cut(_as_num(values), bins=[-999, 65, 75, 85, 90, 95, 100, 105, 999], labels=False, include_lowest=True).astype(float)


def _cloud_bucket(values: pd.Series) -> pd.Series:
    cloud = _as_num(values)
    if cloud.dropna().empty:
        return pd.Series(np.nan, index=values.index, dtype="object")
    if cloud.max(skipna=True) <= 1.5:
        bins = [-0.001, .20, .40, .60, .80, 1.001]
    else:
        bins = [-0.001, 20, 40, 60, 80, 100.001]
    return pd.cut(cloud, bins=bins, labels=["Clear/Low", "Some Clouds", "Partly Cloudy", "Mostly Cloudy", "Overcast"], include_lowest=True).astype("object")


def _solar_loss_bucket(values: pd.Series) -> pd.Series:
    loss = _as_num(values).fillna(0.0)
    out = pd.Series("None", index=loss.index, dtype="object")
    pos = loss[loss > 0]
    if len(pos) >= 10 and pos.nunique() >= 4:
        q1, q2, q3 = pos.quantile([.25, .50, .75]).tolist()
        out.loc[(loss > 0) & (loss <= q1)] = "Low"
        out.loc[(loss > q1) & (loss <= q2)] = "Medium"
        out.loc[(loss > q2) & (loss <= q3)] = "High"
        out.loc[loss > q3] = "Extreme"
    elif len(pos):
        med = pos.median()
        out.loc[(loss > 0) & (loss <= med)] = "Low/Medium"
        out.loc[loss > med] = "High"
    return out


def _forecast_level_bucket(values: pd.Series) -> pd.Series:
    """Fixed Roseville-scale buckets for the pre-correction forecast level.

    The V12.6 metrics showed overcast solar-loss hours were not homogeneous: the same cloud/loss
    bucket could underforecast high-load weekdays and overforecast lower-load/weekend days.  A simple
    forecast-level bucket gives the cloud/solar lookup one more clue without using actual load.
    """
    v = _as_num(values)
    return pd.cut(
        v,
        bins=[-999, 105, 115, 125, 135, 150, 170, 9999],
        labels=["<=105", "105-115", "115-125", "125-135", "135-150", "150-170", "170+"],
        include_lowest=True,
    ).astype("object")


def _prep(df: pd.DataFrame, base_col: str | None = None) -> pd.DataFrame:
    out = df.copy()
    out["DT"] = pd.to_datetime(out["DT"], errors="coerce")
    out = out.dropna(subset=["DT"]).copy()
    out["Month"] = _as_num(out.get("Month", out["DT"].dt.month)).fillna(out["DT"].dt.month).astype(int)
    out["Hour"] = _as_num(out.get("Hour", out["DT"].dt.hour)).fillna(out["DT"].dt.hour).astype(int)
    out["Season"] = out.get("Season", out["Month"].map(_season_from_month))
    out["HourGroup"] = out.get("HourGroup", out["Hour"].map(_hour_group))
    if "IsWeekend" not in out.columns:
        out["IsWeekend"] = out["DT"].dt.dayofweek.ge(5).astype(int)
    else:
        out["IsWeekend"] = _as_num(out["IsWeekend"]).fillna(0).astype(int)

    if "DailyMaxTempBin" not in out.columns:
        if "Temperature_DailyMax" in out.columns:
            out["DailyMaxTempBin"] = _temp_bin(out["Temperature_DailyMax"])
        else:
            out["DailyMaxTempBin"] = np.nan
    if "CloudCoverBucket" not in out.columns:
        out["CloudCoverBucket"] = _cloud_bucket(out["CloudCover_Norm"]) if "CloudCover_Norm" in out.columns else np.nan
    if "SolarLossBucket" not in out.columns:
        col = "BTM_Solar_Loss_From_ClearSky_MW" if "BTM_Solar_Loss_From_ClearSky_MW" in out.columns else "Midday_Overcast_Solar_Loss_MW"
        out["SolarLossBucket"] = _solar_loss_bucket(out[col]) if col in out.columns else np.nan

    if "ClearSky_Index" not in out.columns:
        out["ClearSky_Index"] = np.nan
    if base_col is None or base_col not in out.columns:
        if "Calibrated_Forecast_MWH" in out.columns:
            base_col = "Calibrated_Forecast_MWH"
        elif "Raw_Forecast_MWH" in out.columns:
            base_col = "Raw_Forecast_MWH"
    if base_col and base_col in out.columns:
        out["CloudSolarBaseBucket"] = _forecast_level_bucket(out[base_col])
    elif "CloudSolarBaseBucket" not in out.columns:
        out["CloudSolarBaseBucket"] = np.nan
    out["CloudSolarEventClass"] = _cloud_solar_event_class(out)
    out["CloudSolarEventMultiplier"] = _cloud_solar_event_multiplier(out)
    return out


def _base_bucket_rank(bucket: pd.Series) -> pd.Series:
    """Ordinal rank for CloudSolarBaseBucket; higher means higher pre-correction load."""
    order = {"<=105": 0, "105-115": 1, "115-125": 2, "125-135": 3, "135-150": 4, "150-170": 5, "170+": 6}
    return bucket.astype("object").map(order).astype(float)


def _cloud_solar_event_class(out: pd.DataFrame) -> pd.Series:
    hour = _as_num(out.get("Hour", pd.Series(np.nan, index=out.index))).fillna(-1).astype(int)
    weekend = _as_num(out.get("IsWeekend", pd.Series(0, index=out.index))).fillna(0).astype(int).eq(1)
    cloud = out.get("CloudCoverBucket", pd.Series(np.nan, index=out.index)).astype("object")
    loss_bucket = out.get("SolarLossBucket", pd.Series(np.nan, index=out.index)).astype("object")
    loss_col = "BTM_Solar_Loss_From_ClearSky_MW" if "BTM_Solar_Loss_From_ClearSky_MW" in out.columns else "Midday_Overcast_Solar_Loss_MW"
    loss = _as_num(out[loss_col]).fillna(0.0) if loss_col in out.columns else pd.Series(0.0, index=out.index)
    csi = _as_num(out.get("ClearSky_Index", pd.Series(np.nan, index=out.index)))
    temp = _as_num(out.get("Temperature_DailyMax", pd.Series(np.nan, index=out.index)))
    base_bucket = out.get("CloudSolarBaseBucket", pd.Series(np.nan, index=out.index)).astype("object")
    base_rank = _base_bucket_rank(base_bucket).fillna(2.0)

    cls = pd.Series("other", index=out.index, dtype="object")
    solar_event = cloud.isin(["Mostly Cloudy", "Overcast"]) | loss_bucket.isin(["High", "Extreme"]) | loss.ge(1.25)
    cls.loc[solar_event] = "solar_loss_general"

    # V12.8: weekend solar-loss behavior is not homogeneous.  Low-load/cool weekend cases were
    # responsible for the largest V12.7 overforecast tail, while higher-load weekend cases still
    # occasionally needed some uplift.  Split these before the core-hour labels are applied.
    weekend_low_load = weekend & (base_rank.le(2) | temp.lt(68))
    weekend_high_load = weekend & base_rank.ge(3) & temp.ge(68)
    cls.loc[solar_event & weekend] = "weekend_solar_loss_dampen"
    cls.loc[solar_event & weekend_low_load] = "weekend_lowload_solar_loss_dampen"
    cls.loc[solar_event & weekend_high_load] = "weekend_highload_solar_loss"
    cls.loc[solar_event & hour.between(12, 14) & weekend_low_load] = "weekend_core_lowload_solar_loss_dampen"
    cls.loc[solar_event & hour.between(12, 14) & weekend_high_load] = "weekend_core_highload_solar_loss"
    cls.loc[solar_event & hour.between(12, 14) & weekend & ~(weekend_low_load | weekend_high_load)] = "weekend_core_solar_loss_dampen"

    # V12.8: split true weekday core solar-loss events from general/cloudy days.  The V12.7
    # validation showed the largest remaining underforecasts were weekday noon/13:00 overcast,
    # high-loss hours.  Hour 14 behaved less reliably, so it is kept in a separate transition class.
    weekday_core = (
        solar_event & (~weekend) & hour.between(12, 13)
        & (loss_bucket.isin(["High", "Extreme"]) | loss.ge(8.0))
        & (cloud.eq("Overcast") | csi.le(0.55) | csi.isna())
    )
    weekday_core_highimpact = weekday_core & loss.ge(10.0) & cloud.eq("Overcast") & base_rank.between(2, 3)
    cls.loc[weekday_core] = "weekday_core_solar_loss"
    cls.loc[weekday_core_highimpact] = "weekday_core_highimpact_solar_loss"
    cls.loc[
        solar_event & (~weekend) & hour.eq(14)
        & (loss_bucket.isin(["High", "Extreme"]) | loss.ge(6.0))
        & (cloud.isin(["Mostly Cloudy", "Overcast"]) | csi.le(0.60) | csi.isna())
    ] = "weekday_core_hour14_solar_loss"
    cls.loc[solar_event & (~weekend) & hour.between(10, 11)] = "weekday_morning_solar_loss"
    cls.loc[solar_event & (~weekend) & hour.between(15, 16)] = "weekday_late_midday_solar_loss"
    cls.loc[solar_event & temp.ge(75) & hour.between(16, 20)] = "warm_cloud_peak_transition"
    return cls


def _cloud_solar_event_multiplier(out: pd.DataFrame) -> pd.Series:
    cls = out.get("CloudSolarEventClass", pd.Series("other", index=out.index)).astype("object")
    hour = _as_num(out.get("Hour", pd.Series(np.nan, index=out.index))).fillna(-1).astype(int)
    csi = _as_num(out.get("ClearSky_Index", pd.Series(np.nan, index=out.index)))
    temp = _as_num(out.get("Temperature_DailyMax", pd.Series(np.nan, index=out.index)))
    base_bucket = out.get("CloudSolarBaseBucket", pd.Series(np.nan, index=out.index)).astype("object")
    base_rank = _base_bucket_rank(base_bucket).fillna(2.0)
    loss_col = "BTM_Solar_Loss_From_ClearSky_MW" if "BTM_Solar_Loss_From_ClearSky_MW" in out.columns else "Midday_Overcast_Solar_Loss_MW"
    loss = _as_num(out[loss_col]).fillna(0.0) if loss_col in out.columns else pd.Series(0.0, index=out.index)

    mult = pd.Series(1.0, index=out.index, dtype=float)
    mult.loc[cls.eq("weekday_core_solar_loss")] = 1.18
    mult.loc[cls.eq("weekday_core_highimpact_solar_loss")] = 1.25
    mult.loc[cls.eq("weekday_core_hour14_solar_loss")] = 0.86
    mult.loc[cls.eq("weekday_morning_solar_loss")] = 0.82
    mult.loc[cls.eq("weekday_late_midday_solar_loss")] = 0.76
    mult.loc[cls.eq("weekend_solar_loss_dampen")] = 0.45
    mult.loc[cls.eq("weekend_core_solar_loss_dampen")] = 0.35
    mult.loc[cls.eq("weekend_lowload_solar_loss_dampen")] = 0.24
    mult.loc[cls.eq("weekend_core_lowload_solar_loss_dampen")] = 0.16
    mult.loc[cls.eq("weekend_highload_solar_loss")] = 0.58
    mult.loc[cls.eq("weekend_core_highload_solar_loss")] = 0.52
    mult.loc[cls.eq("warm_cloud_peak_transition")] = 0.92

    # If clear-sky index is not extremely low, the BTM loss estimate is less conclusive; dampen positive uplifts.
    mult.loc[csi.gt(0.60) & loss.lt(10.0)] *= 0.70
    # Strengthen only the noon/13:00 high-impact weekday core; hour 14 had mixed sign in V12.7.
    mult.loc[cls.eq("weekday_core_highimpact_solar_loss") & hour.between(12, 13) & csi.le(0.45) & loss.ge(10.0)] *= 1.05
    # Low-load/cool weekend days were the largest V12.7 overforecast tail.
    mult.loc[cls.str.contains("weekend_core_lowload", na=False) & temp.lt(65) & base_rank.le(2)] *= 0.75
    return mult.clip(0.08, 1.65)

def _residual_work(backtest_df: pd.DataFrame, residual_col: str = "Residual_MWH", forecast_col: str | None = None) -> pd.DataFrame:
    work = _prep(backtest_df, base_col=forecast_col)
    if residual_col in work.columns:
        work["Residual"] = _as_num(work[residual_col])
    elif forecast_col and {"Actual_MWH", forecast_col}.issubset(work.columns):
        work["Residual"] = _as_num(work["Actual_MWH"]) - _as_num(work[forecast_col])
    elif "Residual_MWH" in work.columns:
        work["Residual"] = _as_num(work["Residual_MWH"])
    else:
        work["Residual"] = _as_num(work["Actual_MWH"]) - _as_num(work["Raw_Forecast_MWH"])
    return work.dropna(subset=["Residual"]).copy()


def _make_lookup(work: pd.DataFrame, keys: list[str], blend: float, cap_mwh: float, shrink_k: float) -> pd.DataFrame:
    grp = work.groupby(keys, dropna=False)["Residual"].agg(["mean", "count", "std"]).reset_index()
    grp["shrink_weight"] = grp["count"] / (grp["count"] + float(shrink_k))
    grp["correction"] = (grp["mean"] * grp["shrink_weight"] * float(blend)).clip(-float(cap_mwh), float(cap_mwh))
    grp["abs_correction"] = grp["correction"].abs()
    return grp[keys + ["correction", "mean", "count", "std", "shrink_weight", "abs_correction"]]


def _cloud_solar_mask(work: pd.DataFrame, hours: list[int], min_loss_mw: float) -> pd.Series:
    loss_col = "BTM_Solar_Loss_From_ClearSky_MW" if "BTM_Solar_Loss_From_ClearSky_MW" in work.columns else "Midday_Overcast_Solar_Loss_MW"
    loss = _as_num(work[loss_col]).fillna(0.0) if loss_col in work.columns else pd.Series(0.0, index=work.index)
    cloud = work.get("CloudCoverBucket", pd.Series(np.nan, index=work.index)).astype("object")
    loss_bucket = work.get("SolarLossBucket", pd.Series(np.nan, index=work.index)).astype("object")
    return (
        work["Hour"].isin([int(h) for h in hours])
        & (
            cloud.isin(["Mostly Cloudy", "Overcast"])
            | loss_bucket.isin(["High", "Extreme"])
            | loss.ge(float(min_loss_mw))
        )
    )


def build_cloud_solar_shape_lookup(
    backtest_df: pd.DataFrame,
    hours: list[int] | None = None,
    blend: float = 0.80,
    cap_mwh: float = 16.0,
    shrink_k: float = 10.0,
    min_loss_mw: float = 1.25,
    residual_col: str = "Residual_MWH",
    forecast_col: str | None = None,
) -> dict | None:
    """Learn hour-shaped corrections for cloudy/overcast BTM solar-loss events.

    V12.7 deliberately learns this correction from the residual left *after* the broad residual,
    heat, and warm-ramp corrections when the pipeline provides that frame.  This avoids double-counting
    the same upward bias that earlier correction layers already handled.
    """
    if backtest_df is None or backtest_df.empty:
        return None
    hours = [int(h) for h in (hours or [10, 11, 12, 13, 14, 15, 16])]
    work = _residual_work(backtest_df, residual_col=residual_col, forecast_col=forecast_col)
    if work.empty:
        return None
    event = work.loc[_cloud_solar_mask(work, hours, min_loss_mw)].copy()
    if event.empty:
        return None
    specs = [
        ("event_class_hour_loss", ["CloudSolarEventClass", "Hour", "SolarLossBucket"]),
        ("weekday_hour_cloud_loss", ["IsWeekend", "Hour", "CloudCoverBucket", "SolarLossBucket"]),
        ("event_class_hour_temp", ["CloudSolarEventClass", "Hour", "DailyMaxTempBin"]),
        ("hour_cloud_loss", ["Hour", "CloudCoverBucket", "SolarLossBucket"]),
        ("season_hour_cloud_loss", ["Season", "Hour", "CloudCoverBucket", "SolarLossBucket"]),
        ("hour_base_loss", ["Hour", "CloudSolarBaseBucket", "SolarLossBucket"]),
        ("hour_temp_loss", ["Hour", "DailyMaxTempBin", "SolarLossBucket"]),
        ("hour_loss", ["Hour", "SolarLossBucket"]),
        ("hour_cloud", ["Hour", "CloudCoverBucket"]),
    ]
    return {
        "ordered_levels": [{"name": name, "keys": keys, "lookup": _make_lookup(event, keys, blend, cap_mwh, shrink_k)} for name, keys in specs],
        "metadata": {
            "n": int(len(event)),
            "mean_residual": float(event["Residual"].mean()),
            "mae": float(event["Residual"].abs().mean()),
            "hours": hours,
            "min_loss_mw": float(min_loss_mw),
            "residual_col": residual_col,
            "forecast_col": forecast_col,
        },
    }


def _default_weights() -> dict[str, float]:
    return {
        "event_class_hour_loss": 0.24,
        "weekday_hour_cloud_loss": 0.20,
        "event_class_hour_temp": 0.14,
        "hour_cloud_loss": 0.14,
        "season_hour_cloud_loss": 0.10,
        "hour_base_loss": 0.08,
        "hour_temp_loss": 0.05,
        "hour_loss": 0.03,
        "hour_cloud": 0.02,
    }


def apply_cloud_solar_shape_correction(
    future_df: pd.DataFrame,
    lookup_bundle: dict | None,
    hours: list[int] | None = None,
    cap_mwh: float = 16.0,
    min_loss_mw: float = 1.25,
    level_weights: dict[str, float] | None = None,
    use_event_multiplier: bool = True,
) -> pd.DataFrame:
    base_col = "Calibrated_Forecast_MWH" if "Calibrated_Forecast_MWH" in future_df.columns else "Raw_Forecast_MWH"
    out = _prep(future_df, base_col=base_col)
    out["Cloud_Solar_Shape_Cal_MWH"] = 0.0
    out["Cloud_Solar_Shape_Raw_Cal_MWH"] = 0.0
    out["Cloud_Solar_Adjusted_Forecast_MWH"] = _as_num(out[base_col]).clip(lower=0.0)
    out["Cloud_Solar_Correction_Source"] = "none"
    if not lookup_bundle or not lookup_bundle.get("ordered_levels"):
        out["Calibrated_Forecast_MWH"] = out["Cloud_Solar_Adjusted_Forecast_MWH"]
        return out
    hours = [int(h) for h in (hours or [10, 11, 12, 13, 14, 15, 16])]
    eligible = _cloud_solar_mask(out, hours, min_loss_mw)
    if not eligible.any():
        out["Calibrated_Forecast_MWH"] = out["Cloud_Solar_Adjusted_Forecast_MWH"]
        return out

    weights = _default_weights()
    if level_weights:
        weights.update({str(k): float(v) for k, v in level_weights.items()})
    weighted_sum = pd.Series(0.0, index=out.index)
    weight_sum = pd.Series(0.0, index=out.index)
    matched_names = {idx: [] for idx in out.index}

    for level in lookup_bundle.get("ordered_levels", []):
        name = str(level.get("name", "level"))
        keys = list(level.get("keys", []))
        lk = level.get("lookup")
        lw = float(weights.get(name, 0.0))
        if lw <= 0 or lk is None or lk.empty or not all(k in out.columns for k in keys):
            continue
        tmp = out.loc[eligible, keys].reset_index().merge(lk[keys + ["correction", "count"]], on=keys, how="left")
        matched = tmp["correction"].notna()
        if matched.any():
            idx = tmp.loc[matched, "index"]
            corr = _as_num(tmp.loc[matched, "correction"]).fillna(0.0).to_numpy(dtype=float)
            cnt = _as_num(tmp.loc[matched, "count"]).fillna(0.0).clip(lower=0.0).to_numpy(dtype=float)
            count_factor = np.sqrt(cnt / (cnt + 14.0))
            eff_w = lw * np.where(np.isfinite(count_factor), count_factor, 1.0)
            weighted_sum.loc[idx] += corr * eff_w
            weight_sum.loc[idx] += eff_w
            for ix in idx:
                matched_names[ix].append(name)

    mask = eligible & (weight_sum > 0)
    raw_corr = pd.Series(0.0, index=out.index)
    raw_corr.loc[mask] = (weighted_sum.loc[mask] / weight_sum.loc[mask]).clip(-float(cap_mwh), float(cap_mwh))

    # V12.8 high-impact floor: the V12.7 metrics showed true weekday noon/13:00 overcast,
    # high BTM-loss hours were still undercorrected.  Apply a small, capped minimum positive
    # uplift only to that narrow class; do not touch hour 14 or weekend/low-load events.
    cls = out.get("CloudSolarEventClass", pd.Series("other", index=out.index)).astype("object")
    hour = _as_num(out.get("Hour", pd.Series(np.nan, index=out.index))).fillna(-1).astype(int)
    loss_col = "BTM_Solar_Loss_From_ClearSky_MW" if "BTM_Solar_Loss_From_ClearSky_MW" in out.columns else "Midday_Overcast_Solar_Loss_MW"
    loss = _as_num(out[loss_col]).fillna(0.0) if loss_col in out.columns else pd.Series(0.0, index=out.index)
    highimpact = mask & cls.eq("weekday_core_highimpact_solar_loss") & hour.between(12, 13) & loss.ge(10.0)
    if highimpact.any():
        floor = (1.35 + 0.20 * loss.loc[highimpact]).clip(lower=3.00, upper=min(float(cap_mwh), 4.75))
        raw_corr.loc[highimpact] = np.maximum(raw_corr.loc[highimpact].astype(float), floor.astype(float))

    out.loc[mask, "Cloud_Solar_Shape_Raw_Cal_MWH"] = raw_corr.loc[mask]

    if use_event_multiplier:
        mult = _as_num(out.get("CloudSolarEventMultiplier", pd.Series(1.0, index=out.index))).fillna(1.0)
        # Damp only positive uplift in likely overcorrection cases; allow learned negative corrections to remain.
        pos = raw_corr > 0
        corrected = raw_corr.copy()
        corrected.loc[pos] = raw_corr.loc[pos] * mult.loc[pos]
        out.loc[mask, "Cloud_Solar_Shape_Cal_MWH"] = corrected.loc[mask].clip(-float(cap_mwh), float(cap_mwh))
    else:
        out.loc[mask, "Cloud_Solar_Shape_Cal_MWH"] = raw_corr.loc[mask]

    out.loc[mask, "Cloud_Solar_Correction_Source"] = ["+".join(matched_names[idx]) for idx in out.loc[mask].index]
    out["Cloud_Solar_Adjusted_Forecast_MWH"] = (_as_num(out[base_col]) + _as_num(out["Cloud_Solar_Shape_Cal_MWH"])).clip(lower=0.0)
    out["Calibrated_Forecast_MWH"] = out["Cloud_Solar_Adjusted_Forecast_MWH"]
    return out


def cloud_solar_lookup_debug_table(lookup_bundle: dict | None) -> pd.DataFrame:
    if not lookup_bundle:
        return pd.DataFrame()
    frames = []
    for i, level in enumerate(lookup_bundle.get("ordered_levels", []), start=1):
        lk = level.get("lookup")
        if isinstance(lk, pd.DataFrame) and not lk.empty:
            tmp = lk.copy()
            tmp.insert(0, "LookupLevel", i)
            tmp.insert(1, "CalibrationLevel", level.get("name"))
            tmp.insert(2, "Keys", "+".join(level.get("keys", [])))
            frames.append(tmp)
    meta = lookup_bundle.get("metadata", {}) if isinstance(lookup_bundle, dict) else {}
    if meta:
        frames.append(pd.DataFrame([{"LookupLevel": 0, "CalibrationLevel": "metadata", "Keys": k, "correction": v} for k, v in meta.items()]))
    return pd.concat(frames, ignore_index=True, sort=False) if frames else pd.DataFrame()
