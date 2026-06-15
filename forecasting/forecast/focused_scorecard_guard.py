from __future__ import annotations

"""Focused post-selector guard for replay-identified hot/peak scorecard regimes."""

import numpy as np
import pandas as pd


def _as_num(values) -> pd.Series:
    return pd.to_numeric(values, errors="coerce")


def _local_datetime_series(values, index: pd.Index | None = None) -> pd.Series:
    raw = values if isinstance(values, pd.Series) else pd.Series(values, index=index)
    try:
        return pd.to_datetime(raw, errors="coerce")
    except ValueError:
        # Exported forecast CSVs can contain both -08:00 and -07:00 offsets.
        # Guard rules are local-hour rules, so preserve the local clock for
        # fallback month/hour/day extraction.
        cleaned = raw.astype(str).str.strip().str.replace(r"(?:[+-]\d{2}:?\d{2}|Z)$", "", regex=True)
        return pd.to_datetime(cleaned, errors="coerce")


def _forecast_anchor_mask(df: pd.DataFrame, preferred_col: str | None = None) -> pd.Series:
    candidates = [preferred_col, "Final_Forecast_MWH", "Forecast", "Raw_Forecast_MWH", "Stage_Selected_Forecast_MWH"]
    seen: set[str] = set()
    for col in candidates:
        if not col or col in seen or col not in df.columns:
            continue
        seen.add(col)
        values = _as_num(df[col])
        if values.notna().any():
            return values.notna()
    return pd.Series(True, index=df.index, dtype=bool)


def _guard_cfg(config: dict | None) -> dict:
    raw = config or {}
    calibration = raw.get("calibration", {}) or {}
    stage_selector = calibration.get("stage_selector", {}) or {}
    return stage_selector.get("focused_scorecard_guard", {}) or {}


def _month_series(df: pd.DataFrame) -> pd.Series:
    if "Month" in df.columns:
        month = _as_num(df["Month"])
        if month.notna().any():
            return month
    if "DT" in df.columns:
        return _local_datetime_series(df["DT"]).dt.month.astype(float)
    return pd.Series(np.nan, index=df.index, dtype=float)


def _forecast_day_series(df: pd.DataFrame, anchor_col: str | None = None) -> pd.Series:
    if "Forecast_Day" in df.columns:
        day = _as_num(df["Forecast_Day"])
        if day.notna().any():
            return day
    if "DT" in df.columns:
        dt = _local_datetime_series(df["DT"])
        if dt.notna().any():
            anchor = _forecast_anchor_mask(df, anchor_col) & dt.notna()
            first_day = (dt[anchor].min() if anchor.any() else dt.min()).normalize()
            return ((dt.dt.normalize() - first_day).dt.days + 1).astype(float)
    return pd.Series(np.nan, index=df.index, dtype=float)


def _hour_series(df: pd.DataFrame) -> pd.Series:
    if "Hour" in df.columns:
        hour = _as_num(df["Hour"])
        if hour.notna().any():
            return hour
    if "DT" in df.columns:
        return _local_datetime_series(df["DT"]).dt.hour.astype(float)
    return pd.Series(np.nan, index=df.index, dtype=float)


def _bool_series(df: pd.DataFrame, col: str) -> pd.Series:
    if col not in df.columns:
        return pd.Series(False, index=df.index, dtype=bool)
    values = df[col]
    if values.dtype == object:
        return values.astype(str).str.lower().isin({"true", "1", "yes"})
    return _as_num(values).fillna(0).ne(0)


def _optional_num_series(df: pd.DataFrame, *cols: str) -> pd.Series:
    for col in cols:
        if col in df.columns:
            values = _as_num(df[col])
            if values.notna().any():
                return values
    return pd.Series(np.nan, index=df.index, dtype=float)


def _list_mask(values: pd.Series, allowed) -> pd.Series:
    if allowed is None:
        return pd.Series(True, index=values.index, dtype=bool)
    allowed_set = {int(x) for x in allowed}
    return values.round().astype("Int64").isin(allowed_set).fillna(False)


def _rule_mask(
    df: pd.DataFrame,
    rule: dict,
    *,
    forecast: pd.Series,
    month: pd.Series,
    hour: pd.Series,
    forecast_day: pd.Series,
    daily_max: pd.Series,
    cloud_cover: pd.Series,
    solar_loss: pd.Series,
    is_holiday: pd.Series,
    is_weekend: pd.Series,
) -> pd.Series:
    mask = pd.Series(True, index=df.index, dtype=bool)
    mask &= _list_mask(month, rule.get("months"))
    mask &= _list_mask(hour, rule.get("hours"))

    if "min_forecast_day" in rule:
        mask &= forecast_day.ge(float(rule["min_forecast_day"]))
    if "max_forecast_day" in rule:
        mask &= forecast_day.le(float(rule["max_forecast_day"]))
    if "min_maxtemp_f" in rule:
        mask &= daily_max.ge(float(rule["min_maxtemp_f"]))
    if "max_maxtemp_f" in rule:
        # Exclusive upper bound keeps adjacent tuned temperature regimes from overlapping.
        mask &= daily_max.lt(float(rule["max_maxtemp_f"]))
    if "min_forecast_mwh" in rule:
        mask &= forecast.ge(float(rule["min_forecast_mwh"]))
    if "max_forecast_mwh" in rule:
        mask &= forecast.lt(float(rule["max_forecast_mwh"]))
    if "holiday" in rule:
        mask &= is_holiday.eq(bool(rule["holiday"]))
    if "weekend" in rule:
        mask &= is_weekend.eq(bool(rule["weekend"]))
    if "min_cloud_cover_norm" in rule:
        mask &= cloud_cover.ge(float(rule["min_cloud_cover_norm"]))
    if "max_cloud_cover_norm" in rule:
        mask &= cloud_cover.le(float(rule["max_cloud_cover_norm"]))
    if "min_solar_loss_mw" in rule:
        mask &= solar_loss.ge(float(rule["min_solar_loss_mw"]))
    if "max_solar_loss_mw" in rule:
        mask &= solar_loss.le(float(rule["max_solar_loss_mw"]))

    return mask.fillna(False)


def apply_focused_scorecard_guard(
    df: pd.DataFrame,
    config: dict | None,
    *,
    forecast_col: str,
    also_update_cols: tuple[str, ...] = ("Stage_Selected_Forecast_MWH",),
) -> pd.DataFrame:
    """Apply bounded residual guards for the remaining hot/peak scorecard slices.

    The rules intentionally use only production-available inputs: month, hour,
    forecast lead day, forecasted daily max temperature, holiday flag, and current
    point-forecast level. They run after stage selection/weather hedge and write
    explicit diagnostics so replay can attribute the change.
    """
    out = df.copy()
    out["Focused_Scorecard_Guard_MWH"] = 0.0
    out["Focused_Scorecard_Guard_Source"] = "none"

    cfg = _guard_cfg(config)
    if out.empty or not bool(cfg.get("enabled", False)) or forecast_col not in out.columns:
        return out

    rules = cfg.get("rules", []) or []
    if not rules:
        return out

    forecast = _as_num(out[forecast_col])
    month = _month_series(out)
    hour = _hour_series(out)
    forecast_day = _forecast_day_series(out, anchor_col=forecast_col)
    daily_max = _as_num(out.get("Temperature_DailyMax", pd.Series(np.nan, index=out.index)))
    cloud_cover = _optional_num_series(out, "CloudCover_Norm")
    solar_loss = _optional_num_series(
        out,
        "BTM_Solar_Loss_From_ClearSky_MW",
        "Midday_Overcast_Solar_Loss_MW",
    )
    is_holiday = _bool_series(out, "IsHoliday")
    is_weekend = _bool_series(out, "IsWeekend")

    total_adjustment = pd.Series(0.0, index=out.index, dtype=float)
    source = pd.Series("none", index=out.index, dtype="object")
    cap = abs(float(cfg.get("total_cap_mwh", 20.0)))

    for raw_rule in rules:
        rule = raw_rule or {}
        if not bool(rule.get("enabled", True)):
            continue
        adjustment = float(rule.get("adjustment_mwh", 0.0) or 0.0)
        if abs(adjustment) < 1e-9:
            continue
        mask = _rule_mask(
            out,
            rule,
            forecast=forecast,
            month=month,
            hour=hour,
            forecast_day=forecast_day,
            daily_max=daily_max,
            cloud_cover=cloud_cover,
            solar_loss=solar_loss,
            is_holiday=is_holiday,
            is_weekend=is_weekend,
        )
        if not mask.any():
            continue
        name = str(rule.get("name", "focused_scorecard_guard")).strip() or "focused_scorecard_guard"
        total_adjustment.loc[mask] += adjustment
        prior = source.loc[mask].astype(str)
        source.loc[mask] = np.where(prior.eq("none"), name, prior + "+" + name)

    total_adjustment = total_adjustment.clip(lower=-cap, upper=cap)
    if not total_adjustment.ne(0.0).any():
        return out

    out["Focused_Scorecard_Guard_MWH"] = total_adjustment
    out["Focused_Scorecard_Guard_Source"] = source
    out[forecast_col] = (forecast + total_adjustment).clip(lower=0.0)

    if forecast_col == "Final_Backtest_Forecast_MWH" and "Final_Forecast_MWH" in out.columns:
        out["Final_Forecast_MWH"] = out[forecast_col]
    for col in also_update_cols:
        if col in out.columns and col != forecast_col:
            out[col] = (_as_num(out[col]) + total_adjustment).clip(lower=0.0)
    return out
