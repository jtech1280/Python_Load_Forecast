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


def _season_series(df: pd.DataFrame, month: pd.Series) -> pd.Series:
    if "Season" in df.columns:
        season = df["Season"].where(pd.notna(df["Season"]), "").astype(str).str.strip()
        season = season.mask(season.str.lower().isin({"nan", "nat", "none"}), "")
        if season.ne("").any():
            return season

    season_from_month = pd.Series("", index=df.index, dtype="object")
    month_int = month.round().astype("Int64")
    season_from_month.loc[month_int.isin([12, 1, 2]).fillna(False)] = "Winter"
    season_from_month.loc[month_int.isin([3, 4, 5]).fillna(False)] = "Spring"
    season_from_month.loc[month_int.isin([6, 7, 8, 9]).fillna(False)] = "Summer"
    season_from_month.loc[month_int.isin([10, 11]).fillna(False)] = "Fall"
    return season_from_month


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


def _first_rule_value(rule: dict, *keys: str):
    for key in keys:
        if key in rule and rule[key] is not None:
            return rule[key]
    return None


def _rule_mask(
    df: pd.DataFrame,
    rule: dict,
    *,
    forecast: pd.Series,
    raw_minus_samehour_7day: pd.Series,
    raw_minus_samehour_yesterday: pd.Series,
    month: pd.Series,
    season: pd.Series,
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
    if rule.get("seasons") is not None:
        allowed_seasons = {str(x).strip() for x in rule.get("seasons", [])}
        mask &= season.astype(str).str.strip().isin(allowed_seasons)
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
    min_raw_7day = _first_rule_value(
        rule,
        "min_raw_minus_samehour_7day_mean_mwh",
        "min_raw_minus_rolling7_samehour_mwh",
    )
    if min_raw_7day is not None:
        mask &= raw_minus_samehour_7day.ge(float(min_raw_7day))
    max_raw_7day = _first_rule_value(
        rule,
        "max_raw_minus_samehour_7day_mean_mwh",
        "max_raw_minus_rolling7_samehour_mwh",
    )
    if max_raw_7day is not None:
        mask &= raw_minus_samehour_7day.le(float(max_raw_7day))
    min_raw_yesterday = _first_rule_value(
        rule,
        "min_raw_minus_samehour_yesterday_mwh",
        "min_raw_minus_lag24_mwh",
    )
    if min_raw_yesterday is not None:
        mask &= raw_minus_samehour_yesterday.ge(float(min_raw_yesterday))
    max_raw_yesterday = _first_rule_value(
        rule,
        "max_raw_minus_samehour_yesterday_mwh",
        "max_raw_minus_lag24_mwh",
    )
    if max_raw_yesterday is not None:
        mask &= raw_minus_samehour_yesterday.le(float(max_raw_yesterday))
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
    forecast lead day, forecasted daily max temperature, holiday flag, current
    point-forecast level, and recursive load-state features. They run after stage
    selection/weather hedge and write explicit diagnostics so replay can attribute
    the change.
    """
    out = df.copy()
    forecast = (
        _as_num(out[forecast_col])
        if forecast_col in out.columns
        else pd.Series(np.nan, index=out.index, dtype=float)
    )
    out["Pre_Focused_Guard_Forecast_MWH"] = forecast
    out["Post_Focused_Guard_Forecast_MWH"] = forecast
    out["Focused_Guard_Applied_Flag"] = 0
    out["Focused_Scorecard_Guard_MWH"] = 0.0
    out["Focused_Scorecard_Guard_Source"] = "none"

    cfg = _guard_cfg(config)
    if out.empty or not bool(cfg.get("enabled", False)) or forecast_col not in out.columns:
        return out

    rules = cfg.get("rules", []) or []
    if not rules:
        return out

    month = _month_series(out)
    season = _season_series(out, month)
    hour = _hour_series(out)
    forecast_day = _forecast_day_series(out, anchor_col=forecast_col)
    daily_max = _as_num(out.get("Temperature_DailyMax", pd.Series(np.nan, index=out.index)))
    cloud_cover = _optional_num_series(out, "CloudCover_Norm")
    solar_loss = _optional_num_series(
        out,
        "BTM_Solar_Loss_From_ClearSky_MW",
        "Midday_Overcast_Solar_Loss_MW",
    )
    raw_forecast = _optional_num_series(out, "Raw_Forecast_MWH")
    samehour_7day = _optional_num_series(
        out,
        "MWH_SameHour7DayMean",
        "Baseline_Rolling7DaySameHourAvg_MWH",
    )
    samehour_yesterday = _optional_num_series(
        out,
        "MWH_Lag24",
        "Baseline_SameHourYesterday_MWH",
    )
    raw_minus_samehour_7day = raw_forecast - samehour_7day
    raw_minus_samehour_yesterday = raw_forecast - samehour_yesterday
    out["Raw_Minus_SameHour7DayMean_MWH"] = raw_minus_samehour_7day
    out["Raw_Minus_SameHourYesterday_MWH"] = raw_minus_samehour_yesterday
    is_holiday = _bool_series(out, "IsHoliday")
    is_weekend = _bool_series(out, "IsWeekend")

    total_adjustment = pd.Series(0.0, index=out.index, dtype=float)
    source = pd.Series("none", index=out.index, dtype="object")
    cap = abs(float(cfg.get("total_cap_mwh", 20.0)))
    dynamic_cap = pd.Series(cap, index=out.index, dtype=float)

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
            raw_minus_samehour_7day=raw_minus_samehour_7day,
            raw_minus_samehour_yesterday=raw_minus_samehour_yesterday,
            month=month,
            season=season,
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
        rule_cap = rule.get("max_total_cap_mwh")
        if rule_cap is not None:
            dynamic_cap.loc[mask] = np.maximum(dynamic_cap.loc[mask], abs(float(rule_cap)))
        prior = source.loc[mask].astype(str)
        source.loc[mask] = np.where(prior.eq("none"), name, prior + "+" + name)

    total_adjustment = total_adjustment.where(total_adjustment.ge(-dynamic_cap), -dynamic_cap)
    total_adjustment = total_adjustment.where(total_adjustment.le(dynamic_cap), dynamic_cap)
    if not total_adjustment.ne(0.0).any():
        return out

    out["Focused_Scorecard_Guard_MWH"] = total_adjustment
    out["Focused_Scorecard_Guard_Source"] = source
    post_guard = (forecast + total_adjustment).clip(lower=0.0)
    out["Post_Focused_Guard_Forecast_MWH"] = post_guard
    out["Focused_Guard_Applied_Flag"] = total_adjustment.ne(0.0).astype(int)
    out[forecast_col] = post_guard

    if forecast_col == "Final_Backtest_Forecast_MWH" and "Final_Forecast_MWH" in out.columns:
        out["Final_Forecast_MWH"] = out[forecast_col]
    for col in also_update_cols:
        if col in out.columns and col != forecast_col:
            out[col] = (_as_num(out[col]) + total_adjustment).clip(lower=0.0)
    return out
