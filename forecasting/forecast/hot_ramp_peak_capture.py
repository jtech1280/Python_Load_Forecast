from __future__ import annotations

"""Shadow peak-anchor capture stages.

These stages are intentionally narrow. They target hot ramp and heat
persistence peak hours with bounded peak-anchor uplifts and leave production
forecasts unchanged while ``shadow_mode`` is true.
"""

from typing import Any

import numpy as np
import pandas as pd

HOT_RAMP_PEAK_COLUMNS = [
    "Hot_Ramp_Peak_Model_Version",
    "Hot_Ramp_Peak_Shadow_Mode",
    "Hot_Ramp_Peak_Base_Forecast_MWH",
    "Hot_Ramp_Peak_Correction_MWH",
    "Hot_Ramp_Peak_Shadow_Forecast_MWH",
    "Hot_Ramp_Peak_Correction_Applied_Flag",
    "Hot_Ramp_Peak_Source",
    "Hot_Ramp_Peak_Evaluation_Mode",
    "Hot_Ramp_Peak_Scope_Flag",
    "Hot_Ramp_Peak_Strong_Ramp_Flag",
    "Hot_Ramp_Peak_Base_DailyPeak_MWH",
    "Hot_Ramp_Peak_Predicted_Target_MWH",
    "Hot_Ramp_Peak_Predicted_PeakHour",
    "Hot_Ramp_Peak_Base_PeakHour",
    "Hot_Ramp_Peak_Timing_Selected_PeakHour",
    "Hot_Ramp_Peak_Timing_Source",
    "Hot_Ramp_Peak_Timing_Override_Flag",
    "Hot_Ramp_Peak_Timing_Block_Source",
    "Hot_Ramp_Peak_Timing_Confidence_MWH",
    "Hot_Ramp_Peak_Learned_Residual_MWH",
    "Hot_Ramp_Peak_Recent_Anchor_MWH",
    "Hot_Ramp_Peak_SameHour7_Anchor_MWH",
    "Hot_Ramp_Peak_Lag24_Anchor_MWH",
    "Hot_Ramp_Peak_WarmerScenario_Delta_MWH",
    "Hot_Ramp_Peak_Residual_MWH",
    "Hot_Ramp_Peak_AbsError_MWH",
    "Hot_Ramp_Peak_Delta_AbsError_MWH",
    "Hot_Ramp_Peak_Base_High_Guard_Applied",  # New diagnostic column
    "Hot_Ramp_Peak_Warmer_Delta_High_Guard_Applied",  # New diagnostic column
]

HEAT_PERSISTENCE_PEAK_COLUMNS = [
    "Heat_Persistence_Peak_Model_Version",
    "Heat_Persistence_Peak_Shadow_Mode",
    "Heat_Persistence_Peak_Base_Forecast_MWH",
    "Heat_Persistence_Peak_Correction_MWH",
    "Heat_Persistence_Peak_Shadow_Forecast_MWH",
    "Heat_Persistence_Peak_Correction_Applied_Flag",
    "Heat_Persistence_Peak_Source",
    "Heat_Persistence_Peak_Evaluation_Mode",
    "Heat_Persistence_Peak_Scope_Flag",
    "Heat_Persistence_Peak_Strong_Flag",
    "Heat_Persistence_Peak_Base_DailyPeak_MWH",
    "Heat_Persistence_Peak_Predicted_Target_MWH",
    "Heat_Persistence_Peak_Predicted_PeakHour",
    "Heat_Persistence_Peak_Learned_Residual_MWH",
    "Heat_Persistence_Peak_Recent_Anchor_MWH",
    "Heat_Persistence_Peak_SameHour7_Anchor_MWH",
    "Heat_Persistence_Peak_Lag24_Anchor_MWH",
    "Heat_Persistence_Peak_WarmerScenario_Delta_MWH",
    "Heat_Persistence_Peak_ConsecutiveExtremeHotDays100",
    "Heat_Persistence_Peak_DailyMaxTemp3DayMean_F",
    "Heat_Persistence_Peak_Residual_MWH",
    "Heat_Persistence_Peak_AbsError_MWH",
    "Heat_Persistence_Peak_Delta_AbsError_MWH",
]


def _cfg(config: dict | None) -> dict:
    raw = config or {}
    if "hot_ramp_peak_capture" in raw:
        return raw.get("hot_ramp_peak_capture", {}) or {}
    return (raw.get("calibration", {}) or {}).get("hot_ramp_peak_capture", {}) or {}


def _heat_persistence_cfg(config: dict | None) -> dict:
    raw = config or {}
    if "heat_persistence_peak_capture" in raw:
        return raw.get("heat_persistence_peak_capture", {}) or {}
    return (raw.get("calibration", {}) or {}).get(
        "heat_persistence_peak_capture", {}
    ) or {}


def _as_num(
    value: Any, index: pd.Index | None = None, default: float = np.nan
) -> pd.Series:
    if isinstance(value, pd.Series):
        raw = value
    else:
        raw = pd.Series(default, index=index)
    return pd.to_numeric(raw, errors="coerce")


def _optional_num(
    values: pd.DataFrame, *cols: str, default: float = np.nan
) -> pd.Series:
    for col in cols:
        if col in values.columns:
            return _as_num(values[col], values.index).fillna(default)
    return pd.Series(default, index=values.index, dtype=float)


def _local_datetime(values: pd.DataFrame) -> pd.Series:
    raw = values.get("DT", pd.Series(pd.NaT, index=values.index))
    try:
        return pd.to_datetime(raw, errors="coerce")
    except ValueError:
        cleaned = (
            raw.astype(str)
            .str.strip()
            .str.replace(r"(?:[+-]\d{2}:?\d{2}|Z)$", "", regex=True)
        )
        return pd.to_datetime(cleaned, errors="coerce")


def _date(values: pd.DataFrame, dt: pd.Series | None = None) -> pd.Series:
    if "Date" in values.columns:
        raw = values["Date"]
        if raw.notna().any():
            return raw.astype(str)
    dt = dt if dt is not None else _local_datetime(values)
    return dt.dt.date.astype(str)


def _hour(values: pd.DataFrame, dt: pd.Series | None = None) -> pd.Series:
    dt = dt if dt is not None else _local_datetime(values)
    hour = _as_num(
        values.get("Hour", pd.Series(np.nan, index=values.index)), values.index
    )
    return hour.where(hour.notna(), dt.dt.hour).fillna(0.0)


def _month(values: pd.DataFrame, dt: pd.Series | None = None) -> pd.Series:
    dt = dt if dt is not None else _local_datetime(values)
    month = _as_num(
        values.get("Month", pd.Series(np.nan, index=values.index)), values.index
    )
    return month.where(month.notna(), dt.dt.month).fillna(1.0)


def _forecast_day(values: pd.DataFrame, dt: pd.Series | None = None) -> pd.Series:
    if "Forecast_Day" in values.columns:
        day = _as_num(values["Forecast_Day"], values.index)
        if day.notna().any():
            return day
    dt = dt if dt is not None else _local_datetime(values)
    valid = dt.notna()
    out = pd.Series(np.nan, index=values.index, dtype=float)
    if valid.any():
        first_day = dt.loc[valid].min().normalize()
        out.loc[valid] = (dt.loc[valid].dt.normalize() - first_day).dt.days.astype(
            float
        ) + 1.0
    return out


def _cloud_norm(values: pd.DataFrame) -> pd.Series:
    cloud = _optional_num(values, "CloudCover_Norm", default=np.nan)
    if cloud.notna().any() and cloud.max(skipna=True) > 1.5:
        cloud = cloud / 100.0
    return cloud.clip(0.0, 1.0)


def _base_forecast(values: pd.DataFrame, forecast_col: str) -> pd.Series:
    for col in [
        forecast_col,
        "Final_Backtest_Forecast_MWH",
        "Final_Forecast_MWH",
        "Stage_Selected_Forecast_MWH",
        "Auto_Residual_Adjusted_Forecast_MWH",
        "Recent_Corrected_Forecast_MWH",
        "Calibrated_Forecast_MWH",
        "Raw_Forecast_MWH",
        "Forecast",
    ]:
        if col in values.columns:
            return _as_num(values[col], values.index)
    return pd.Series(np.nan, index=values.index, dtype=float)


def _daily_max(values: pd.DataFrame) -> pd.Series:
    return _optional_num(values, "Temperature_DailyMax", "Temperature", default=np.nan)


def _daily_ramp(values: pd.DataFrame, dt: pd.Series | None = None) -> pd.Series:
    configured = _optional_num(values, "DailyMaxTemp_Ramp_1Day", default=np.nan)
    if configured.notna().any():
        return configured

    daily_max = _daily_max(values)
    prior = _optional_num(values, "PriorDay_DailyMaxTemp", default=np.nan)
    derived = daily_max - prior
    if derived.notna().any():
        return derived

    dt = dt if dt is not None else _local_datetime(values)
    work = pd.DataFrame(
        {
            "_Date": _date(values, dt=dt),
            "_DailyMax": daily_max,
        },
        index=values.index,
    )
    daily = work.groupby("_Date", dropna=False)["_DailyMax"].max().reset_index()
    daily["_Date_dt"] = pd.to_datetime(daily["_Date"], errors="coerce")
    daily = daily.sort_values("_Date_dt")
    daily["_Ramp"] = daily["_DailyMax"].diff()
    ramp_map = daily.set_index("_Date")["_Ramp"].to_dict()
    return work["_Date"].map(ramp_map).reindex(values.index).astype(float)


def _consecutive_extreme_days100(
    values: pd.DataFrame, dt: pd.Series | None = None
) -> pd.Series:
    configured = _optional_num(values, "ConsecutiveExtremeHotDays100", default=np.nan)
    if configured.notna().any():
        return configured

    dt = dt if dt is not None else _local_datetime(values)
    date = _date(values, dt=dt)
    work = pd.DataFrame(
        {"_Date": date, "_DailyMax": _daily_max(values)}, index=values.index
    )
    daily = work.groupby("_Date", dropna=False)["_DailyMax"].max().reset_index()
    daily["_Date_dt"] = pd.to_datetime(daily["_Date"], errors="coerce")
    daily = daily.sort_values("_Date_dt")

    counts: list[int] = []
    running = 0
    for value in pd.to_numeric(daily["_DailyMax"], errors="coerce"):
        if pd.notna(value) and float(value) >= 100.0:
            running += 1
        else:
            running = 0
        counts.append(running)
    daily["_ConsecutiveExtremeHotDays100"] = counts
    count_map = daily.set_index("_Date")["_ConsecutiveExtremeHotDays100"].to_dict()
    return work["_Date"].map(count_map).reindex(values.index).astype(float)


def _dailymax_3day_mean(values: pd.DataFrame, dt: pd.Series | None = None) -> pd.Series:
    configured = _optional_num(values, "DailyMaxTemp_3DayMean", default=np.nan)
    if configured.notna().any():
        return configured

    dt = dt if dt is not None else _local_datetime(values)
    date = _date(values, dt=dt)
    work = pd.DataFrame(
        {"_Date": date, "_DailyMax": _daily_max(values)}, index=values.index
    )
    daily = work.groupby("_Date", dropna=False)["_DailyMax"].max().reset_index()
    daily["_Date_dt"] = pd.to_datetime(daily["_Date"], errors="coerce")
    daily = daily.sort_values("_Date_dt")
    daily["_DailyMax3DayMean"] = (
        pd.to_numeric(daily["_DailyMax"], errors="coerce")
        .rolling(3, min_periods=1)
        .mean()
    )
    mean_map = daily.set_index("_Date")["_DailyMax3DayMean"].to_dict()
    return work["_Date"].map(mean_map).reindex(values.index).astype(float)


def _horizon_bucket(day: float) -> str:
    if not np.isfinite(day):
        return "Unknown"
    d = int(day)
    if d <= 1:
        return "Day1"
    if d <= 7:
        return "Days2to7"
    if d <= 16:
        return "Days8to16"
    return "Day17Plus"


def _temp_bucket(temp: float) -> str:
    if not np.isfinite(temp):
        return "Unknown"
    if temp < 95:
        return "90-95"
    if temp < 100:
        return "95-100"
    if temp < 105:
        return "100-105"
    return "105+"


def _cloud_bucket(cloud: float) -> str:
    if not np.isfinite(cloud):
        return "Unknown"
    if cloud <= 0.20:
        return "Clear/Low"
    if cloud <= 0.40:
        return "Some Clouds"
    if cloud <= 0.60:
        return "Partly Cloudy"
    if cloud <= 0.80:
        return "Mostly Cloudy"
    return "Overcast"


def _as_list(value: Any) -> list[Any]:
    if value is None:
        return []
    if isinstance(value, (list, tuple, set)):
        return list(value)
    return [value]


def _bounded_mask(
    mask: pd.Series,
    series: pd.Series,
    rule: dict,
    *,
    min_key: str,
    max_key: str,
    max_exclusive_key: str | None = None,
) -> pd.Series:
    lower = rule.get(min_key)
    if lower is not None:
        mask &= series.ge(float(lower)).fillna(False)
    exclusive_key = max_exclusive_key or f"{max_key}_exclusive"
    upper_exclusive = rule.get(exclusive_key)
    if upper_exclusive is not None:
        mask &= series.lt(float(upper_exclusive)).fillna(False)
    else:
        upper = rule.get(max_key)
        if upper is not None:
            mask &= series.le(float(upper)).fillna(False)
    return mask


def _targeted_missing_slice_mask(
    values: pd.DataFrame,
    rule: dict,
    *,
    hour: pd.Series,
    month: pd.Series,
    forecast_day: pd.Series,
    daily_max: pd.Series,
    cloud: pd.Series,
    base: pd.Series,
    raw_minus_samehour7: pd.Series,
    existing_correction: pd.Series,
) -> pd.Series:
    has_forecast_day_filter = (
        rule.get("min_forecast_day") is not None
        or rule.get("max_forecast_day") is not None
    )
    if (
        has_forecast_day_filter
        and bool(rule.get("require_explicit_forecast_day", True))
        and "Forecast_Day" not in values.columns
    ):
        return pd.Series(False, index=values.index, dtype=bool)

    mask = pd.Series(True, index=values.index, dtype=bool)

    months = _as_list(rule.get("months", rule.get("month")))
    if months:
        mask &= month.astype("Int64").isin([int(m) for m in months]).fillna(False)

    hours = _as_list(rule.get("hours"))
    if hours:
        mask &= hour.astype("Int64").isin([int(h) for h in hours]).fillna(False)

    mask = _bounded_mask(
        mask, forecast_day, rule, min_key="min_forecast_day", max_key="max_forecast_day"
    )
    mask = _bounded_mask(
        mask,
        daily_max,
        rule,
        min_key="min_maxtemp_f",
        max_key="max_maxtemp_f",
        max_exclusive_key="max_maxtemp_f_exclusive",
    )
    mask = _bounded_mask(
        mask,
        cloud,
        rule,
        min_key="min_cloud_cover_norm",
        max_key="max_cloud_cover_norm",
    )
    mask = _bounded_mask(
        mask,
        raw_minus_samehour7,
        rule,
        min_key="min_raw_minus_samehour7_mwh",
        max_key="max_raw_minus_samehour7_mwh",
        max_exclusive_key="max_raw_minus_samehour7_mwh_exclusive",
    )
    mask = _bounded_mask(
        mask,
        base,
        rule,
        min_key="min_base_forecast_mwh",
        max_key="max_base_forecast_mwh",
    )

    if bool(rule.get("only_if_existing_correction_zero", True)):
        mask &= existing_correction.le(
            float(rule.get("existing_correction_epsilon_mwh", 1e-9))
        ).fillna(False)
    return mask.fillna(False)


def _enabled_targeted_missing_rules(cfg: dict) -> list[dict[str, Any]]:
    rules = (
        cfg.get("targeted_missing_slices", []) or cfg.get("targeted_slices", []) or []
    )
    enabled: list[dict[str, Any]] = []
    for raw_rule in rules:
        rule = raw_rule or {}
        if not bool(rule.get("enabled", True)):
            continue
        if float(rule.get("correction_mwh", 0.0) or 0.0) <= 0.0:
            continue
        enabled.append(rule)
    return enabled


def _apply_hot_ramp_targeted_missing_slices(
    out: pd.DataFrame,
    cfg: dict,
    *,
    targeted_rules: list[dict[str, Any]],
    row_hour: pd.Series,
    row_month: pd.Series,
    forecast_day: pd.Series,
    daily_max: pd.Series,
    cloud: pd.Series,
    base: pd.Series,
    same7: pd.Series,
    warmer_delta: pd.Series,
    correction: pd.Series,
    source: pd.Series,
    cooling: pd.Series,
    guard_cfg: dict,
) -> tuple[pd.Series, pd.Series]:
    if not targeted_rules:
        return correction, source

    raw_minus_samehour7 = _optional_num(out, "Raw_Forecast_MWH", default=np.nan) - same7
    for rule in targeted_rules:
        rule_correction = float(rule.get("correction_mwh", 0.0) or 0.0)
        capped_correction = float(
            min(rule_correction, float(rule.get("cap_mwh", rule_correction)))
        )
        if capped_correction <= 0.0:
            continue

        rule_mask = (
            _targeted_missing_slice_mask(
                out,
                rule,
                hour=row_hour,
                month=row_month,
                forecast_day=forecast_day,
                daily_max=daily_max,
                cloud=cloud,
                base=base,
                raw_minus_samehour7=raw_minus_samehour7,
                existing_correction=correction,
            )
            & base.notna()
        )

        if bool(rule.get("respect_cooling_guard", True)):
            blocked = rule_mask & cooling
            if blocked.any():
                source.loc[blocked & correction.le(1e-9)] = str(
                    rule.get("cooling_blocked_source")
                    or cfg.get("targeted_slice_cooling_blocked_source")
                    or guard_cfg.get("blocked_source")
                    or "hot_ramp_peak_targeted_slice_cooling_blocked"
                )
            rule_mask &= ~cooling
        if not rule_mask.any():
            continue

        guarded_correction = pd.Series(capped_correction, index=out.index, dtype=float)

        base_forecast_threshold_mwh = float(
            rule.get(
                "base_forecast_threshold_mwh",
                cfg.get("base_forecast_threshold_mwh", np.inf),
            )
        )
        default_guard_cap = float(cfg.get("cap_mwh", capped_correction))
        max_correction_mwh_if_base_high = float(
            rule.get(
                "max_correction_mwh_if_base_high",
                cfg.get("max_correction_mwh_if_base_high", default_guard_cap),
            )
        )
        base_high_guard = rule_mask & base.gt(base_forecast_threshold_mwh).fillna(False)
        if base_high_guard.any():
            guarded_correction.loc[base_high_guard] = np.minimum(
                guarded_correction.loc[base_high_guard],
                max_correction_mwh_if_base_high,
            )

        warmer_delta_threshold_mwh = float(
            rule.get(
                "warmer_delta_threshold_mwh",
                cfg.get("warmer_delta_threshold_mwh", np.inf),
            )
        )
        max_correction_mwh_if_warmer_delta_high = float(
            rule.get(
                "max_correction_mwh_if_warmer_delta_high",
                cfg.get("max_correction_mwh_if_warmer_delta_high", default_guard_cap),
            )
        )
        warmer_delta_high_guard = rule_mask & warmer_delta.gt(
            warmer_delta_threshold_mwh
        ).fillna(False)
        if warmer_delta_high_guard.any():
            guarded_correction.loc[warmer_delta_high_guard] = np.minimum(
                guarded_correction.loc[warmer_delta_high_guard],
                max_correction_mwh_if_warmer_delta_high,
            )

        apply_mask = (
            rule_mask & guarded_correction.gt(0.0) & correction.lt(guarded_correction)
        )
        if not apply_mask.any():
            continue

        correction.loc[apply_mask] = guarded_correction.loc[apply_mask]
        source.loc[apply_mask] = str(
            rule.get(
                "source",
                cfg.get("targeted_slice_source", "hot_ramp_peak_targeted_slice"),
            )
        )
        out.loc[apply_mask, "Hot_Ramp_Peak_Scope_Flag"] = 1
        out.loc[apply_mask, "Hot_Ramp_Peak_Predicted_Target_MWH"] = (
            base.loc[apply_mask] + guarded_correction.loc[apply_mask]
        )
        out.loc[apply_mask, "Hot_Ramp_Peak_Predicted_PeakHour"] = row_hour.loc[
            apply_mask
        ]
        out.loc[apply_mask, "Hot_Ramp_Peak_Base_PeakHour"] = row_hour.loc[apply_mask]
        out.loc[apply_mask, "Hot_Ramp_Peak_Timing_Selected_PeakHour"] = row_hour.loc[
            apply_mask
        ]
        out.loc[apply_mask, "Hot_Ramp_Peak_Timing_Source"] = "targeted_slice_row_hour"
        out.loc[apply_mask, "Hot_Ramp_Peak_Timing_Override_Flag"] = 0
        out.loc[apply_mask, "Hot_Ramp_Peak_Timing_Block_Source"] = ""
        out.loc[apply_mask, "Hot_Ramp_Peak_Timing_Confidence_MWH"] = np.nan
        out.loc[apply_mask, "Hot_Ramp_Peak_Learned_Residual_MWH"] = (
            guarded_correction.loc[apply_mask]
        )
        out.loc[apply_mask, "Hot_Ramp_Peak_Base_High_Guard_Applied"] = (
            base_high_guard.loc[apply_mask].astype(int)
        )
        out.loc[apply_mask, "Hot_Ramp_Peak_Warmer_Delta_High_Guard_Applied"] = (
            warmer_delta_high_guard.loc[apply_mask].astype(int)
        )

    return correction, source


def _finalize_hot_ramp_peak_capture(
    out: pd.DataFrame,
    *,
    base: pd.Series,
    correction: pd.Series,
    source: pd.Series,
    shadow_mode: bool,
    forecast_col: str,
    also_update_cols: tuple[str, ...],
) -> pd.DataFrame:
    correction = correction.where(base.notna(), 0.0)
    adjusted = (base + correction).clip(lower=0.0)
    source.loc[base.isna()] = "invalid_base_forecast"
    out["Hot_Ramp_Peak_Correction_MWH"] = correction
    out["Hot_Ramp_Peak_Shadow_Forecast_MWH"] = adjusted
    out["Hot_Ramp_Peak_Correction_Applied_Flag"] = correction.ne(0.0).astype(int)
    out["Hot_Ramp_Peak_Source"] = source

    actual = _optional_num(out, "Actual_MWH", "Actual", default=np.nan)
    base_residual = actual - base
    residual = actual - adjusted
    out["Hot_Ramp_Peak_Residual_MWH"] = residual
    out["Hot_Ramp_Peak_AbsError_MWH"] = residual.abs()
    out["Hot_Ramp_Peak_Delta_AbsError_MWH"] = residual.abs() - base_residual.abs()

    if not shadow_mode and forecast_col in out.columns:
        out[forecast_col] = adjusted
        if (
            forecast_col == "Final_Backtest_Forecast_MWH"
            and "Final_Forecast_MWH" in out.columns
        ):
            out["Final_Forecast_MWH"] = adjusted
        for col in also_update_cols:
            if col in out.columns and col != forecast_col:
                out[col] = adjusted
        if (
            forecast_col in {"Final_Backtest_Forecast_MWH", "Final_Forecast_MWH"}
            and actual.notna().any()
        ):
            out["Final_Residual_MWH"] = actual - _as_num(out[forecast_col], out.index)
            out["Final_AbsError_MWH"] = out["Final_Residual_MWH"].abs()
            out["Final_APE"] = np.where(
                actual.abs() > 1e-9,
                out["Final_AbsError_MWH"] / actual.abs() * 100.0,
                np.nan,
            )
    return out


def _persistence_bucket(consecutive_extreme: float) -> str:
    if not np.isfinite(consecutive_extreme):
        return "Unknown"
    if consecutive_extreme >= 5:
        return "ExtremeConsec5Plus"
    if consecutive_extreme >= 4:
        return "ExtremeConsec4"
    if consecutive_extreme >= 3:
        return "ExtremeConsec3"
    if consecutive_extreme >= 2:
        return "ExtremeConsec2"
    return "NoPersistence"


def _hour_weights(
    hours: pd.Series, center_hour: float, spread_hours: float
) -> pd.Series:
    h = _as_num(hours, hours.index)
    if spread_hours <= 0:
        raw = h.eq(round(center_hour)).astype(float)
    else:
        raw = np.exp(-0.5 * ((h - float(center_hour)) / float(spread_hours)) ** 2)
    weights = pd.Series(raw, index=hours.index, dtype=float).fillna(0.0)
    max_weight = weights.max(skipna=True)
    if not np.isfinite(max_weight) or max_weight <= 0.0:
        return pd.Series(0.0, index=hours.index, dtype=float)
    return weights / max_weight


_TIMING_SOURCE_COLUMNS = {
    "xgb_component": "XGB_Pred_MWH",
    "raw_xgb_lgb": "Raw_Forecast_MWH",
    "raw_forecast": "Raw_Forecast_MWH",
    "recent_corrected": "Recent_Corrected_Forecast_MWH",
    "focused_shape": "Focused_Shape_Adjusted_Forecast_MWH",
    "daily_peak_shadow": "Daily_Peak_Shadow_Adjusted_Forecast_MWH",
}


def _timing_source_specs(selector_cfg: dict) -> list[tuple[str, str]]:
    raw_sources = selector_cfg.get(
        "timing_sources", selector_cfg.get("sources", ["xgb_component"])
    )
    specs: list[tuple[str, str]] = []
    for raw in _as_list(raw_sources):
        if isinstance(raw, str):
            source = raw
            column = _TIMING_SOURCE_COLUMNS.get(raw, raw)
        else:
            item = raw or {}
            column = item.get("column")
            source = item.get("source") or item.get("name") or column
        if source and column:
            specs.append((str(source), str(column)))
    return specs


def _timing_peak_record(
    values: pd.DataFrame,
    *,
    source: str,
    column: str,
    candidate_idx: pd.Index,
    row_hour: pd.Series,
    allowed_hours: set[int],
) -> dict[str, Any] | None:
    if column not in values.columns:
        return None

    forecast = _as_num(values[column], values.index).loc[candidate_idx]
    hour = _as_num(row_hour, row_hour.index).loc[candidate_idx]
    valid = forecast.notna() & hour.notna()
    if allowed_hours:
        valid &= hour.round().astype("Int64").isin(allowed_hours).fillna(False)
    if not valid.any():
        return None

    ranked = forecast.loc[valid].sort_values(ascending=False)
    if ranked.empty:
        return None

    peak_idx = ranked.index[0]
    top_value = float(ranked.iloc[0])
    if len(ranked) > 1:
        margin = top_value - float(ranked.iloc[1])
    else:
        margin = np.inf
    peak_hour = float(hour.loc[peak_idx])
    return {
        "source": source,
        "column": column,
        "hour": peak_hour,
        "margin": margin,
    }


def _base_timing_selection(
    base_peak_hour: float,
    *,
    source: str,
    block_source: str = "",
    confidence: float = np.nan,
) -> dict[str, Any]:
    return {
        "selected_peak_hour": float(base_peak_hour),
        "source": source,
        "override": 0,
        "block_source": block_source,
        "confidence": confidence,
    }


def _select_hot_ramp_timing_hour(
    values: pd.DataFrame,
    cfg: dict,
    *,
    candidate_idx: pd.Index,
    row_hour: pd.Series,
    day_strong: bool,
    day_temp: float,
    day_ramp: float,
    base_peak_hour: float,
) -> dict[str, Any]:
    selector_cfg = cfg.get("peak_timing_selector", {}) or {}
    if not bool(selector_cfg.get("enabled", False)):
        return _base_timing_selection(
            base_peak_hour,
            source=str(
                selector_cfg.get("disabled_source", "base_timing_selector_disabled")
            ),
            block_source="selector_disabled",
        )

    if bool(selector_cfg.get("block_on_strong_hot_ramp", True)):
        block_min_temp = selector_cfg.get("strong_hot_ramp_min_maxtemp_f")
        block_min_ramp = selector_cfg.get("strong_hot_ramp_min_dailymax_ramp_1day_f")
        if block_min_temp is not None and block_min_ramp is not None:
            block = day_temp >= float(block_min_temp) and day_ramp >= float(
                block_min_ramp
            )
        else:
            block = day_strong
        if block:
            return _base_timing_selection(
                base_peak_hour,
                source=str(
                    selector_cfg.get(
                        "strong_hot_ramp_fallback_source",
                        "base_strong_hot_ramp_fallback",
                    )
                ),
                block_source=str(
                    selector_cfg.get(
                        "strong_hot_ramp_block_source", "strong_hot_ramp_timing_blocked"
                    )
                ),
            )

    fallback_hours = [int(h) for h in cfg.get("hours", [16, 17, 18, 19, 20])]
    allowed_hours = {
        int(h) for h in _as_list(selector_cfg.get("allowed_hours", fallback_hours))
    }
    min_margin = float(selector_cfg.get("min_peak_margin_mwh", 1.0))
    records: list[dict[str, Any]] = []
    for source_name, column in _timing_source_specs(selector_cfg):
        record = _timing_peak_record(
            values,
            source=source_name,
            column=column,
            candidate_idx=candidate_idx,
            row_hour=row_hour,
            allowed_hours=allowed_hours,
        )
        if record is None or float(record["margin"]) < min_margin:
            continue
        records.append(record)

    if not records:
        return _base_timing_selection(
            base_peak_hour,
            source=str(
                selector_cfg.get("no_candidate_source", "base_no_timing_candidate")
            ),
            block_source="no_confident_timing_candidate",
        )

    priority_values = _as_list(
        selector_cfg.get("source_priority", [record["source"] for record in records])
    )
    source_priority = {
        str(source_name): pos for pos, source_name in enumerate(priority_values)
    }
    consensus_required = max(1, int(selector_cfg.get("consensus_required", 1)))
    max_spread = float(selector_cfg.get("max_consensus_hour_spread", 0.0))
    required_source = selector_cfg.get("required_source")
    required_source = (
        str(required_source) if required_source not in {None, ""} else None
    )

    clusters: list[list[dict[str, Any]]] = []
    for record in records:
        cluster = [
            item
            for item in records
            if abs(float(item["hour"]) - float(record["hour"])) <= max_spread
        ]
        if len(cluster) < consensus_required:
            continue
        if required_source is not None and not any(
            str(item["source"]) == required_source for item in cluster
        ):
            continue
        clusters.append(cluster)

    if not clusters:
        confidence = max(
            (float(record["margin"]) for record in records), default=np.nan
        )
        return _base_timing_selection(
            base_peak_hour,
            source=str(
                selector_cfg.get("no_consensus_source", "base_no_timing_consensus")
            ),
            block_source="no_timing_consensus",
            confidence=confidence,
        )

    def cluster_key(cluster: list[dict[str, Any]]) -> tuple[int, float, int]:
        confidence = min(float(item["margin"]) for item in cluster)
        best_priority = min(
            source_priority.get(str(item["source"]), 9999) for item in cluster
        )
        return (len(cluster), confidence, -best_priority)

    selected_cluster = max(clusters, key=cluster_key)
    selected_record = min(
        selected_cluster,
        key=lambda item: source_priority.get(str(item["source"]), 9999),
    )
    selected_hour = float(selected_record["hour"])
    confidence = min(float(item["margin"]) for item in selected_cluster)
    override = int(
        np.isfinite(selected_hour) and abs(selected_hour - float(base_peak_hour)) > 1e-9
    )
    source_suffix = "consensus" if override else "aligned"
    return {
        "selected_peak_hour": selected_hour,
        "source": f"{selected_record['source']}_{source_suffix}",
        "override": override,
        "block_source": "",
        "confidence": confidence,
    }


def _timed_peak_correction(
    *,
    base: pd.Series,
    row_hour: pd.Series,
    adjustment_idx: pd.Index,
    base_peak: float,
    raw_correction: float,
    selected_peak_hour: float,
    spread_hours: float,
    selector_cfg: dict,
    timing_override: bool,
) -> pd.Series:
    weights = _hour_weights(
        row_hour.loc[adjustment_idx], selected_peak_hour, spread_hours=spread_hours
    )
    day_correction = raw_correction * weights
    if not (
        timing_override
        and bool(selector_cfg.get("target_selected_hour_to_daily_peak", True))
    ):
        return day_correction

    selected_mask = (
        row_hour.loc[adjustment_idx].round().eq(round(selected_peak_hour)).fillna(False)
    )
    selected_idx = selected_mask[selected_mask].index
    if len(selected_idx) == 0:
        return day_correction

    target_peak = base_peak + raw_correction
    selected_base = float(base.loc[selected_idx].max())
    selected_hour_correction = max(raw_correction, target_peak - selected_base)
    max_extra = selector_cfg.get("max_selected_hour_extra_correction_mwh")
    if max_extra is not None:
        selected_hour_correction = min(
            selected_hour_correction, raw_correction + float(max_extra)
        )
    if selected_hour_correction <= raw_correction + 1e-9:
        return day_correction

    day_correction = selected_hour_correction * weights
    adjusted = base.loc[adjustment_idx] + day_correction
    non_selected = ~selected_mask
    over_target = non_selected & adjusted.gt(target_peak).fillna(False)
    if over_target.any() and bool(
        selector_cfg.get("cap_nonselected_hours_to_target", True)
    ):
        day_correction.loc[over_target] = (target_peak - base.loc[over_target]).clip(
            lower=0.0
        )
    return day_correction.clip(lower=0.0)


def _cooling_underway_guard(values: pd.DataFrame, cfg: dict) -> pd.Series:
    guard_cfg = (
        cfg.get("cooling_underway_guard", {}) or cfg.get("cooling_guard", {}) or {}
    )
    if not bool(guard_cfg.get("enabled", False)):
        return pd.Series(False, index=values.index, dtype=bool)

    masks: list[pd.Series] = []

    def add_min(series: pd.Series, key: str) -> None:
        if key in guard_cfg:
            masks.append(series.ge(float(guard_cfg[key])).fillna(False))

    daily_max = _daily_max(values)
    temp = _optional_num(values, "Temperature", default=np.nan)
    drop = _optional_num(values, "Temperature_Drop_From_DailyMax_F", default=np.nan)
    if drop.isna().all():
        drop = daily_max - temp
    add_min(drop, "min_drop_from_dailymax_f")
    add_min(
        _optional_num(values, "TempDrop_Next1Hr_F", default=np.nan),
        "min_forecast_drop_next1hr_f",
    )
    add_min(
        _optional_num(values, "TempDrop_Next2Hr_F", default=np.nan),
        "min_forecast_drop_next2hr_f",
    )
    add_min(
        _optional_num(values, "TempDrop_Next3Hr_F", default=np.nan),
        "min_forecast_drop_next3hr_f",
    )
    add_min(
        _optional_num(values, "DeltaBreeze_Cooling_Signal", default=0.0),
        "min_delta_breeze_cooling_signal",
    )
    add_min(
        _optional_num(values, "DeltaBreeze_PostPeak_LoadDecay_Signal", default=0.0),
        "min_delta_breeze_post_peak_load_decay_signal",
    )
    add_min(
        _optional_num(values, "PostPeak_LoadDecay_1Hr_MWH", default=np.nan),
        "min_post_peak_load_decay_1hr_mwh",
    )
    add_min(
        _optional_num(values, "PostPeak_LoadDecay_2Hr_MWH", default=np.nan),
        "min_post_peak_load_decay_2hr_mwh",
    )

    if not masks:
        return pd.Series(False, index=values.index, dtype=bool)

    mode = str(guard_cfg.get("mode", "any") or "any").lower()
    guard = masks[0].copy()
    for mask in masks[1:]:
        guard = guard & mask if mode in {"all", "and"} else guard | mask

    hour = _hour(values)
    min_hour = guard_cfg.get("min_hour")
    if min_hour is not None:
        guard &= hour.ge(float(min_hour)).fillna(False)
    max_hour = guard_cfg.get("max_hour")
    if max_hour is not None:
        guard &= hour.le(float(max_hour)).fillna(False)
    max_cloud = guard_cfg.get("max_cloud_cover_norm")
    if max_cloud is not None:
        guard &= _cloud_norm(values).le(float(max_cloud)).fillna(False)
    return guard.fillna(False)


def _scope_mask(
    values: pd.DataFrame, cfg: dict, *, enforce_forecast_day: bool = True
) -> tuple[pd.Series, pd.Series]:
    dt = _local_datetime(values)
    hour = _hour(values, dt=dt).astype(int)
    forecast_day = _forecast_day(values, dt=dt)
    daily_max = _daily_max(values)
    ramp = _daily_ramp(values, dt=dt)
    cloud = _cloud_norm(values)

    hours = [int(h) for h in cfg.get("hours", [16, 17, 18, 19, 20])]
    mask = (
        hour.isin(hours)
        & daily_max.ge(float(cfg.get("min_maxtemp_f", 100.0))).fillna(False)
        & ramp.ge(float(cfg.get("min_dailymax_ramp_1day_f", 2.0))).fillna(False)
    )
    if enforce_forecast_day:
        min_day = cfg.get("min_forecast_day", 1)
        max_day = cfg.get("max_forecast_day", 7)
        if min_day is not None:
            mask &= forecast_day.ge(float(min_day)).fillna(False)
        if max_day is not None:
            mask &= forecast_day.le(float(max_day)).fillna(False)
    max_cloud = cfg.get("max_cloud_cover_norm", 0.40)
    if max_cloud is not None:
        mask &= cloud.le(float(max_cloud)).fillna(False)

    cooling = _cooling_underway_guard(values, cfg)
    return mask.fillna(False), cooling.fillna(False)


def _daily_training_frame(
    values: pd.DataFrame, cfg: dict, forecast_col: str
) -> pd.DataFrame:
    if values is None or values.empty:
        return pd.DataFrame()
    work = values.copy()
    dt = _local_datetime(work)
    work["_Date"] = _date(work, dt=dt)
    work["_Hour"] = _hour(work, dt=dt).astype(int)
    work["_Month"] = _month(work, dt=dt)
    work["_ForecastDay"] = _forecast_day(work, dt=dt)
    work["_Base"] = _base_forecast(work, forecast_col=forecast_col)
    work["_Actual"] = _optional_num(work, "Actual_MWH", "Actual", default=np.nan)
    work["_DailyMax"] = _daily_max(work)
    work["_Ramp1D"] = _daily_ramp(work, dt=dt)
    work["_Cloud"] = _cloud_norm(work)
    work["_Same7"] = _optional_num(
        work,
        "MWH_SameHour7DayMean",
        "Baseline_Rolling7DaySameHourAvg_MWH",
        default=np.nan,
    )
    work["_Lag24"] = _optional_num(
        work, "MWH_Lag24", "Baseline_SameHourYesterday_MWH", default=np.nan
    )

    scope, cooling = _scope_mask(
        work,
        cfg,
        enforce_forecast_day=bool(cfg.get("train_enforce_forecast_day", False)),
    )
    work["_Scope"] = scope & ~cooling

    rows: list[dict[str, Any]] = []
    for date, group in work.groupby("_Date", dropna=False):
        candidates = group[group["_Scope"]].copy()
        if candidates.empty:
            continue
        base_values = _as_num(candidates["_Base"], candidates.index).dropna()
        actual_values = _as_num(candidates["_Actual"], candidates.index).dropna()
        if base_values.empty or actual_values.empty:
            continue
        base_peak_idx = base_values.idxmax()
        actual_peak_idx = actual_values.idxmax()
        base_row = candidates.loc[base_peak_idx]
        actual_row = candidates.loc[actual_peak_idx]
        base_peak = float(base_row["_Base"])
        actual_peak = float(actual_row["_Actual"])
        if not (np.isfinite(base_peak) and np.isfinite(actual_peak)):
            continue
        daily_max = float(candidates["_DailyMax"].max())
        ramp = float(candidates["_Ramp1D"].max())
        cloud = float(candidates["_Cloud"].max())
        forecast_day = float(base_row["_ForecastDay"])
        same7 = float(base_row["_Same7"]) if pd.notna(base_row["_Same7"]) else np.nan
        lag24 = float(base_row["_Lag24"]) if pd.notna(base_row["_Lag24"]) else np.nan
        rows.append(
            {
                "Date": str(date),
                "Forecast_Day": forecast_day,
                "HorizonBucket": _horizon_bucket(forecast_day),
                "Month": (
                    int(base_row["_Month"]) if pd.notna(base_row["_Month"]) else np.nan
                ),
                "DailyMaxTempBucket": _temp_bucket(daily_max),
                "CloudCoverBucket": _cloud_bucket(cloud),
                "Temperature_DailyMax": daily_max,
                "DailyMaxTemp_Ramp_1Day": ramp,
                "CloudCover_Max": cloud,
                "Base_PeakHour": float(base_row["_Hour"]),
                "Actual_PeakHour": float(actual_row["_Hour"]),
                "Base_DailyPeak_MWH": base_peak,
                "Actual_DailyPeak_MWH": actual_peak,
                "Target_PeakResidual_MWH": actual_peak - base_peak,
                "SameHour7_AtBasePeak_MWH": same7,
                "SameHour7_TargetResidual_MWH": (
                    actual_peak - same7 if np.isfinite(same7) else np.nan
                ),
                "Lag24_AtBasePeak_MWH": lag24,
                "Lag24_TargetResidual_MWH": (
                    actual_peak - lag24 if np.isfinite(lag24) else np.nan
                ),
                "Lag24_RampSlope_MWH_Per_F": (
                    (actual_peak - lag24) / max(ramp, 1.0)
                    if np.isfinite(lag24)
                    else np.nan
                ),
            }
        )
    return pd.DataFrame(rows)


def _persistence_scope_mask(
    values: pd.DataFrame,
    cfg: dict,
    *,
    enforce_forecast_day: bool = True,
) -> tuple[pd.Series, pd.Series]:
    dt = _local_datetime(values)
    hour = _hour(values, dt=dt).astype(int)
    forecast_day = _forecast_day(values, dt=dt)
    daily_max = _daily_max(values)
    dailymax_3day = _dailymax_3day_mean(values, dt=dt)
    consecutive_extreme = _consecutive_extreme_days100(values, dt=dt)
    ramp = _daily_ramp(values, dt=dt)
    cloud = _cloud_norm(values)

    hours = [int(h) for h in cfg.get("hours", [16, 17, 18, 19, 20])]
    mask = (
        hour.isin(hours)
        & daily_max.ge(float(cfg.get("min_maxtemp_f", 100.0))).fillna(False)
        & consecutive_extreme.ge(
            float(cfg.get("min_consecutive_extreme_days100", 3.0))
        ).fillna(False)
    )
    min_3day = cfg.get("min_dailymax_3day_mean_f", 100.0)
    if min_3day is not None:
        mask &= dailymax_3day.ge(float(min_3day)).fillna(False)
    min_ramp = cfg.get("min_dailymax_ramp_1day_f")
    if min_ramp is not None:
        mask &= ramp.ge(float(min_ramp)).fillna(False)
    max_ramp = cfg.get("max_dailymax_ramp_1day_f", 2.0)
    if max_ramp is not None:
        mask &= ramp.le(float(max_ramp)).fillna(False)

    if enforce_forecast_day:
        min_day = cfg.get("min_forecast_day", 1)
        max_day = cfg.get("max_forecast_day", 7)
        if min_day is not None:
            mask &= forecast_day.ge(float(min_day)).fillna(False)
        if max_day is not None:
            mask &= forecast_day.le(float(max_day)).fillna(False)
    max_cloud = cfg.get("max_cloud_cover_norm", 0.40)
    if max_cloud is not None:
        mask &= cloud.le(float(max_cloud)).fillna(False)

    cooling = _cooling_underway_guard(values, cfg)
    return mask.fillna(False), cooling.fillna(False)


def _daily_persistence_training_frame(
    values: pd.DataFrame, cfg: dict, forecast_col: str
) -> pd.DataFrame:
    if values is None or values.empty:
        return pd.DataFrame()
    work = values.copy()
    dt = _local_datetime(work)
    work["_Date"] = _date(work, dt=dt)
    work["_Hour"] = _hour(work, dt=dt).astype(int)
    work["_Month"] = _month(work, dt=dt)
    work["_ForecastDay"] = _forecast_day(work, dt=dt)
    work["_Base"] = _base_forecast(work, forecast_col=forecast_col)
    work["_Actual"] = _optional_num(work, "Actual_MWH", "Actual", default=np.nan)
    work["_DailyMax"] = _daily_max(work)
    work["_Ramp1D"] = _daily_ramp(work, dt=dt)
    work["_DailyMax3DayMean"] = _dailymax_3day_mean(work, dt=dt)
    work["_ConsecutiveExtreme"] = _consecutive_extreme_days100(work, dt=dt)
    work["_Cloud"] = _cloud_norm(work)
    work["_Same7"] = _optional_num(
        work,
        "MWH_SameHour7DayMean",
        "Baseline_Rolling7DaySameHourAvg_MWH",
        default=np.nan,
    )
    work["_Lag24"] = _optional_num(
        work, "MWH_Lag24", "Baseline_SameHourYesterday_MWH", default=np.nan
    )

    scope, cooling = _persistence_scope_mask(
        work,
        cfg,
        enforce_forecast_day=bool(cfg.get("train_enforce_forecast_day", False)),
    )
    work["_Scope"] = scope & ~cooling

    rows: list[dict[str, Any]] = []
    for date, group in work.groupby("_Date", dropna=False):
        candidates = group[group["_Scope"]].copy()
        if candidates.empty:
            continue
        base_values = _as_num(candidates["_Base"], candidates.index).dropna()
        actual_values = _as_num(candidates["_Actual"], candidates.index).dropna()
        if base_values.empty or actual_values.empty:
            continue
        base_peak_idx = base_values.idxmax()
        actual_peak_idx = actual_values.idxmax()
        base_row = candidates.loc[base_peak_idx]
        actual_row = candidates.loc[actual_peak_idx]
        base_peak = float(base_row["_Base"])
        actual_peak = float(actual_row["_Actual"])
        if not (np.isfinite(base_peak) and np.isfinite(actual_peak)):
            continue
        daily_max = float(candidates["_DailyMax"].max())
        ramp = float(candidates["_Ramp1D"].max())
        dailymax_3day = float(candidates["_DailyMax3DayMean"].max())
        consecutive_extreme = float(candidates["_ConsecutiveExtreme"].max())
        cloud = float(candidates["_Cloud"].max())
        forecast_day = float(base_row["_ForecastDay"])
        same7 = float(base_row["_Same7"]) if pd.notna(base_row["_Same7"]) else np.nan
        lag24 = float(base_row["_Lag24"]) if pd.notna(base_row["_Lag24"]) else np.nan
        rows.append(
            {
                "Date": str(date),
                "Forecast_Day": forecast_day,
                "HorizonBucket": _horizon_bucket(forecast_day),
                "Month": (
                    int(base_row["_Month"]) if pd.notna(base_row["_Month"]) else np.nan
                ),
                "DailyMaxTempBucket": _temp_bucket(daily_max),
                "CloudCoverBucket": _cloud_bucket(cloud),
                "PersistenceBucket": _persistence_bucket(consecutive_extreme),
                "Temperature_DailyMax": daily_max,
                "DailyMaxTemp_Ramp_1Day": ramp,
                "DailyMaxTemp_3DayMean": dailymax_3day,
                "ConsecutiveExtremeHotDays100": consecutive_extreme,
                "CloudCover_Max": cloud,
                "Base_PeakHour": float(base_row["_Hour"]),
                "Actual_PeakHour": float(actual_row["_Hour"]),
                "Base_DailyPeak_MWH": base_peak,
                "Actual_DailyPeak_MWH": actual_peak,
                "Target_PeakResidual_MWH": actual_peak - base_peak,
                "SameHour7_AtBasePeak_MWH": same7,
                "SameHour7_TargetResidual_MWH": (
                    actual_peak - same7 if np.isfinite(same7) else np.nan
                ),
                "Lag24_AtBasePeak_MWH": lag24,
                "Lag24_TargetResidual_MWH": (
                    actual_peak - lag24 if np.isfinite(lag24) else np.nan
                ),
            }
        )
    return pd.DataFrame(rows)


def _mean_if_valid(series: pd.Series, default: float = 0.0) -> float:
    values = _as_num(series).replace([np.inf, -np.inf], np.nan).dropna()
    return float(values.mean()) if not values.empty else float(default)


def _lookup_records(
    daily: pd.DataFrame, keys: list[str], min_days: int
) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    if daily.empty or not all(k in daily.columns for k in keys):
        return rows
    for key_values, group in daily.groupby(keys, dropna=False):
        if not isinstance(key_values, tuple):
            key_values = (key_values,)
        if len(group) < int(min_days):
            continue
        row = {key: value for key, value in zip(keys, key_values)}
        row.update(
            {
                "N": int(len(group)),
                "PeakResidual_MWH": _mean_if_valid(group["Target_PeakResidual_MWH"]),
                "SameHour7Residual_MWH": _mean_if_valid(
                    group["SameHour7_TargetResidual_MWH"]
                ),
                "Lag24Residual_MWH": _mean_if_valid(group["Lag24_TargetResidual_MWH"]),
                "Lag24RampSlope_MWH_Per_F": _mean_if_valid(
                    group["Lag24_RampSlope_MWH_Per_F"]
                    if "Lag24_RampSlope_MWH_Per_F" in group.columns
                    else pd.Series(np.nan, index=group.index)
                ),
            }
        )
        rows.append(row)
    return rows


def _hot_ramp_anchorless_shadow_artifact(cfg: dict, forecast_col: str) -> dict:
    return {
        "daily_training_frame": pd.DataFrame(),
        "lookups": {},
        "metadata": {
            "model_version": str(
                cfg.get("model_version", "hot_ramp_peak_capture_shadow_v1")
            ),
            "training_days": 0,
            "training_start_date": "",
            "training_end_date": "",
            "target_peak_residual_mean_mwh": 0.0,
            "target_peak_residual_mae_mwh": 0.0,
            "recent_hot_peak_anchor_mwh": np.nan,
            "recent_hot_peak_anchor_days": 0,
            "recent_hot_peak_anchor_max_temp_f": np.nan,
            "global_peak_residual_mwh": 0.0,
            "global_samehour7_residual_mwh": 0.0,
            "global_lag24_residual_mwh": 0.0,
            "global_lag24_ramp_slope_mwh_per_f": 0.0,
            "forecast_col": forecast_col,
            "shadow_mode": True,
            "anchorless_shadow_fallback": True,
        },
    }


def _heat_persistence_anchorless_shadow_artifact(cfg: dict, forecast_col: str) -> dict:
    return {
        "daily_training_frame": pd.DataFrame(),
        "lookups": {},
        "metadata": {
            "model_version": str(
                cfg.get("model_version", "heat_persistence_peak_capture_shadow_v1")
            ),
            "training_days": 0,
            "training_start_date": "",
            "training_end_date": "",
            "target_peak_residual_mean_mwh": 0.0,
            "target_peak_residual_mae_mwh": 0.0,
            "recent_heat_persistence_peak_anchor_mwh": np.nan,
            "recent_heat_persistence_anchor_days": 0,
            "recent_heat_persistence_anchor_max_temp_f": np.nan,
            "global_peak_residual_mwh": 0.0,
            "global_samehour7_residual_mwh": 0.0,
            "global_lag24_residual_mwh": 0.0,
            "forecast_col": forecast_col,
            "shadow_mode": True,
            "anchorless_shadow_fallback": True,
        },
    }


def build_hot_ramp_peak_capture_artifact(
    backtest_df: pd.DataFrame,
    config: dict | None,
    *,
    forecast_col: str = "Final_Backtest_Forecast_MWH",
) -> dict | None:
    cfg = _cfg(config)
    if backtest_df is None or backtest_df.empty or not bool(cfg.get("enabled", False)):
        return None
    daily = _daily_training_frame(backtest_df, cfg, forecast_col=forecast_col)
    if daily.empty:
        return None

    min_days = int(cfg.get("min_training_days", 5))
    if len(daily) < min_days:
        return None

    recent_days = int(cfg.get("recent_anchor_days", 3))
    recent = daily.sort_values("Date").tail(max(1, recent_days))
    slope_cap = float(cfg.get("lag24_ramp_slope_cap_mwh_per_f", 4.0))
    global_slope_value = _mean_if_valid(daily["Lag24_RampSlope_MWH_Per_F"])
    global_slope_value = float(np.clip(global_slope_value, -slope_cap, slope_cap))

    lookup_levels = [
        ["HorizonBucket", "Month", "DailyMaxTempBucket", "CloudCoverBucket"],
        ["HorizonBucket", "DailyMaxTempBucket", "CloudCoverBucket"],
        ["HorizonBucket", "DailyMaxTempBucket"],
        ["DailyMaxTempBucket", "CloudCoverBucket"],
        ["DailyMaxTempBucket"],
        ["HorizonBucket"],
    ]
    lookups = {
        "+".join(keys): _lookup_records(
            daily, keys, min_days=int(cfg.get("min_lookup_days", 3))
        )
        for keys in lookup_levels
    }

    return {
        "daily_training_frame": daily,
        "lookups": lookups,
        "metadata": {
            "model_version": str(
                cfg.get("model_version", "hot_ramp_peak_capture_shadow_v1")
            ),
            "training_days": int(len(daily)),
            "training_start_date": str(daily["Date"].min()),
            "training_end_date": str(daily["Date"].max()),
            "target_peak_residual_mean_mwh": float(
                daily["Target_PeakResidual_MWH"].mean()
            ),
            "target_peak_residual_mae_mwh": float(
                daily["Target_PeakResidual_MWH"].abs().mean()
            ),
            "recent_hot_peak_anchor_mwh": float(recent["Actual_DailyPeak_MWH"].max()),
            "recent_hot_peak_anchor_days": int(len(recent)),
            "recent_hot_peak_anchor_max_temp_f": float(
                recent["Temperature_DailyMax"].max()
            ),
            "global_peak_residual_mwh": _mean_if_valid(
                daily["Target_PeakResidual_MWH"]
            ),
            "global_samehour7_residual_mwh": _mean_if_valid(
                daily["SameHour7_TargetResidual_MWH"]
            ),
            "global_lag24_residual_mwh": _mean_if_valid(
                daily["Lag24_TargetResidual_MWH"]
            ),
            "global_lag24_ramp_slope_mwh_per_f": global_slope_value,
            "forecast_col": forecast_col,
            "shadow_mode": bool(cfg.get("shadow_mode", True)),
        },
    }


def build_heat_persistence_peak_capture_artifact(
    backtest_df: pd.DataFrame,
    config: dict | None,
    *,
    forecast_col: str = "Final_Backtest_Forecast_MWH",
) -> dict | None:
    cfg = _heat_persistence_cfg(config)
    if backtest_df is None or backtest_df.empty or not bool(cfg.get("enabled", False)):
        return None
    daily = _daily_persistence_training_frame(
        backtest_df, cfg, forecast_col=forecast_col
    )
    if daily.empty:
        return None

    min_days = int(cfg.get("min_training_days", 3))
    if len(daily) < min_days:
        return None

    recent_days = int(cfg.get("recent_anchor_days", 3))
    recent = daily.sort_values("Date").tail(max(1, recent_days))
    lookup_levels = [
        [
            "HorizonBucket",
            "Month",
            "DailyMaxTempBucket",
            "CloudCoverBucket",
            "PersistenceBucket",
        ],
        [
            "HorizonBucket",
            "DailyMaxTempBucket",
            "CloudCoverBucket",
            "PersistenceBucket",
        ],
        ["DailyMaxTempBucket", "CloudCoverBucket", "PersistenceBucket"],
        ["DailyMaxTempBucket", "PersistenceBucket"],
        ["PersistenceBucket"],
        ["HorizonBucket"],
    ]
    lookups = {
        "+".join(keys): _lookup_records(
            daily, keys, min_days=int(cfg.get("min_lookup_days", 2))
        )
        for keys in lookup_levels
    }

    return {
        "daily_training_frame": daily,
        "lookups": lookups,
        "metadata": {
            "model_version": str(
                cfg.get("model_version", "heat_persistence_peak_capture_shadow_v1")
            ),
            "training_days": int(len(daily)),
            "training_start_date": str(daily["Date"].min()),
            "training_end_date": str(daily["Date"].max()),
            "target_peak_residual_mean_mwh": float(
                daily["Target_PeakResidual_MWH"].mean()
            ),
            "target_peak_residual_mae_mwh": float(
                daily["Target_PeakResidual_MWH"].abs().mean()
            ),
            "recent_heat_persistence_peak_anchor_mwh": float(
                recent["Actual_DailyPeak_MWH"].max()
            ),
            "recent_heat_persistence_anchor_days": int(len(recent)),
            "recent_heat_persistence_anchor_max_temp_f": float(
                recent["Temperature_DailyMax"].max()
            ),
            "global_peak_residual_mwh": _mean_if_valid(
                daily["Target_PeakResidual_MWH"]
            ),
            "global_samehour7_residual_mwh": _mean_if_valid(
                daily["SameHour7_TargetResidual_MWH"]
            ),
            "global_lag24_residual_mwh": _mean_if_valid(
                daily["Lag24_TargetResidual_MWH"]
            ),
            "forecast_col": forecast_col,
            "shadow_mode": bool(cfg.get("shadow_mode", True)),
        },
    }


def _lookup_row(artifact: dict, attrs: dict[str, Any]) -> dict[str, Any] | None:
    for level_name, rows in (artifact.get("lookups", {}) or {}).items():
        keys = level_name.split("+")
        for row in rows:
            if all(str(row.get(key)) == str(attrs.get(key)) for key in keys):
                return row
    return None


def _init_columns(
    out: pd.DataFrame,
    base: pd.Series,
    *,
    cfg: dict,
    source: str,
    evaluation_mode: str,
    shadow_mode: bool,
) -> pd.DataFrame:
    out["Hot_Ramp_Peak_Model_Version"] = str(
        cfg.get("model_version", "hot_ramp_peak_capture_shadow_v1")
    )
    out["Hot_Ramp_Peak_Shadow_Mode"] = int(bool(shadow_mode))
    out["Hot_Ramp_Peak_Base_Forecast_MWH"] = base
    out["Hot_Ramp_Peak_Correction_MWH"] = 0.0
    out["Hot_Ramp_Peak_Shadow_Forecast_MWH"] = base
    out["Hot_Ramp_Peak_Correction_Applied_Flag"] = 0
    out["Hot_Ramp_Peak_Source"] = source
    out["Hot_Ramp_Peak_Evaluation_Mode"] = evaluation_mode
    out["Hot_Ramp_Peak_Scope_Flag"] = 0
    out["Hot_Ramp_Peak_Strong_Ramp_Flag"] = 0
    out["Hot_Ramp_Peak_Base_DailyPeak_MWH"] = np.nan
    out["Hot_Ramp_Peak_Predicted_Target_MWH"] = np.nan
    out["Hot_Ramp_Peak_Predicted_PeakHour"] = np.nan
    out["Hot_Ramp_Peak_Base_PeakHour"] = np.nan
    out["Hot_Ramp_Peak_Timing_Selected_PeakHour"] = np.nan
    out["Hot_Ramp_Peak_Timing_Source"] = ""
    out["Hot_Ramp_Peak_Timing_Override_Flag"] = 0
    out["Hot_Ramp_Peak_Timing_Block_Source"] = ""
    out["Hot_Ramp_Peak_Timing_Confidence_MWH"] = np.nan
    out["Hot_Ramp_Peak_Learned_Residual_MWH"] = np.nan
    out["Hot_Ramp_Peak_Recent_Anchor_MWH"] = np.nan
    out["Hot_Ramp_Peak_SameHour7_Anchor_MWH"] = np.nan
    out["Hot_Ramp_Peak_Lag24_Anchor_MWH"] = np.nan
    out["Hot_Ramp_Peak_WarmerScenario_Delta_MWH"] = 0.0
    out["Hot_Ramp_Peak_Base_High_Guard_Applied"] = 0  # Initialize new column
    out["Hot_Ramp_Peak_Warmer_Delta_High_Guard_Applied"] = 0  # Initialize new column
    actual = _optional_num(out, "Actual_MWH", "Actual", default=np.nan)
    residual = actual - base
    out["Hot_Ramp_Peak_Residual_MWH"] = residual
    out["Hot_Ramp_Peak_AbsError_MWH"] = residual.abs()
    out["Hot_Ramp_Peak_Delta_AbsError_MWH"] = 0.0
    return out


def _init_heat_persistence_columns(
    out: pd.DataFrame,
    base: pd.Series,
    *,
    cfg: dict,
    source: str,
    evaluation_mode: str,
    shadow_mode: bool,
) -> pd.DataFrame:
    out["Heat_Persistence_Peak_Model_Version"] = str(
        cfg.get("model_version", "heat_persistence_peak_capture_shadow_v1")
    )
    out["Heat_Persistence_Peak_Shadow_Mode"] = int(bool(shadow_mode))
    out["Heat_Persistence_Peak_Base_Forecast_MWH"] = base
    out["Heat_Persistence_Peak_Correction_MWH"] = 0.0
    out["Heat_Persistence_Peak_Shadow_Forecast_MWH"] = base
    out["Heat_Persistence_Peak_Correction_Applied_Flag"] = 0
    out["Heat_Persistence_Peak_Source"] = source
    out["Heat_Persistence_Peak_Evaluation_Mode"] = evaluation_mode
    out["Heat_Persistence_Peak_Scope_Flag"] = 0
    out["Heat_Persistence_Peak_Strong_Flag"] = 0
    out["Heat_Persistence_Peak_Base_DailyPeak_MWH"] = np.nan
    out["Heat_Persistence_Peak_Predicted_Target_MWH"] = np.nan
    out["Heat_Persistence_Peak_Predicted_PeakHour"] = np.nan
    out["Heat_Persistence_Peak_Learned_Residual_MWH"] = np.nan
    out["Heat_Persistence_Peak_Recent_Anchor_MWH"] = np.nan
    out["Heat_Persistence_Peak_SameHour7_Anchor_MWH"] = np.nan
    out["Heat_Persistence_Peak_Lag24_Anchor_MWH"] = np.nan
    out["Heat_Persistence_Peak_WarmerScenario_Delta_MWH"] = 0.0
    out["Heat_Persistence_Peak_ConsecutiveExtremeHotDays100"] = np.nan
    out["Heat_Persistence_Peak_DailyMaxTemp3DayMean_F"] = np.nan
    actual = _optional_num(out, "Actual_MWH", "Actual", default=np.nan)
    residual = actual - base
    out["Heat_Persistence_Peak_Residual_MWH"] = residual
    out["Heat_Persistence_Peak_AbsError_MWH"] = residual.abs()
    out["Heat_Persistence_Peak_Delta_AbsError_MWH"] = 0.0
    return out


def apply_hot_ramp_peak_capture(
    df: pd.DataFrame,
    artifact: dict | None,
    config: dict | None,
    *,
    forecast_col: str = "Final_Forecast_MWH",
    also_update_cols: tuple[str, ...] = (),
    force_shadow: bool | None = None,
    evaluation_mode: str = "shadow",
) -> pd.DataFrame:
    out = df.copy()
    cfg = _cfg(config)
    shadow_mode = (
        bool(cfg.get("shadow_mode", True))
        if force_shadow is None
        else bool(force_shadow)
    )
    base = _base_forecast(out, forecast_col=forecast_col)
    out = _init_columns(
        out,
        base,
        cfg=cfg,
        source="disabled",
        evaluation_mode=evaluation_mode,
        shadow_mode=shadow_mode,
    )
    if out.empty or not bool(cfg.get("enabled", False)):
        return out

    dt = _local_datetime(out)
    row_date = _date(out, dt=dt)
    row_hour = _hour(out, dt=dt)
    row_month = _month(out, dt=dt)
    forecast_day = _forecast_day(out, dt=dt)
    daily_max = _daily_max(out)
    ramp = _daily_ramp(out, dt=dt)
    cloud = _cloud_norm(out)
    same7 = _optional_num(
        out,
        "MWH_SameHour7DayMean",
        "Baseline_Rolling7DaySameHourAvg_MWH",
        default=np.nan,
    )
    lag24 = _optional_num(
        out, "MWH_Lag24", "Baseline_SameHourYesterday_MWH", default=np.nan
    )
    warmer = _optional_num(out, "WeatherScenario_warmer_P50_MWH", default=np.nan)
    warmer_delta = (warmer - base).clip(lower=0.0).fillna(0.0)

    scope, cooling = _scope_mask(out, cfg)
    guard_cfg = (
        cfg.get("cooling_underway_guard", {}) or cfg.get("cooling_guard", {}) or {}
    )
    enabled_targeted_rules = _enabled_targeted_missing_rules(cfg)
    if not artifact or not artifact.get("metadata"):
        if shadow_mode and bool(cfg.get("allow_anchorless_shadow_fallback", False)):
            artifact = _hot_ramp_anchorless_shadow_artifact(cfg, forecast_col)
        elif enabled_targeted_rules:
            correction = pd.Series(0.0, index=out.index, dtype=float)
            source = pd.Series("insufficient_history", index=out.index, dtype="object")
            correction, source = _apply_hot_ramp_targeted_missing_slices(
                out,
                cfg,
                targeted_rules=enabled_targeted_rules,
                row_hour=row_hour,
                row_month=row_month,
                forecast_day=forecast_day,
                daily_max=daily_max,
                cloud=cloud,
                base=base,
                same7=same7,
                warmer_delta=warmer_delta,
                correction=correction,
                source=source,
                cooling=cooling,
                guard_cfg=guard_cfg,
            )
            return _finalize_hot_ramp_peak_capture(
                out,
                base=base,
                correction=correction,
                source=source,
                shadow_mode=shadow_mode,
                forecast_col=forecast_col,
                also_update_cols=also_update_cols,
            )
        else:
            out["Hot_Ramp_Peak_Source"] = "insufficient_history"
            return out

    strong = (
        scope
        & daily_max.ge(
            float(cfg.get("strong_ramp_min_maxtemp_f", cfg.get("min_maxtemp_f", 100.0)))
        ).fillna(False)
        & ramp.ge(float(cfg.get("strong_ramp_min_dailymax_ramp_1day_f", 3.0))).fillna(
            False
        )
    )
    out.loc[scope, "Hot_Ramp_Peak_Scope_Flag"] = 1
    out.loc[strong, "Hot_Ramp_Peak_Strong_Ramp_Flag"] = 1

    correction = pd.Series(0.0, index=out.index, dtype=float)
    source = pd.Series("out_of_scope", index=out.index, dtype="object")
    source.loc[scope & cooling] = str(
        cfg.get("cooling_blocked_source")
        or guard_cfg.get("blocked_source")
        or "hot_ramp_peak_cooling_underway_blocked"
    )

    hours = [int(h) for h in cfg.get("hours", [16, 17, 18, 19, 20])]
    spread_hours = float(cfg.get("spread_hours", 1.0))
    cap = float(cfg.get("cap_mwh", 9.0))
    # Matches config.yaml's hot_ramp_peak_capture.strong_ramp_cap_mwh (not strong_cap_mwh,
    # which is the key name used by the sibling heat_persistence_peak_capture section).
    strong_cap = float(cfg.get("strong_ramp_cap_mwh", cap))
    min_abs = float(cfg.get("min_abs_correction_mwh", 0.25))
    floor = float(cfg.get("ramp_floor_mwh", 0.0))
    strong_floor = float(cfg.get("strong_ramp_floor_mwh", 4.0))
    warmer_fraction = float(cfg.get("warmer_scenario_fraction", 0.25))
    metadata = artifact.get("metadata", {}) or {}
    anchorless_fallback = bool(metadata.get("anchorless_shadow_fallback", False))
    block_negative_learned = bool(cfg.get("block_negative_learned_residual", True))
    min_learned_residual = float(cfg.get("min_learned_residual_mwh", 0.0))
    anchor_support_guard = bool(cfg.get("anchor_support_guard_enabled", True))
    anchor_min_learned = float(cfg.get("anchor_min_learned_residual_mwh", strong_floor))
    anchor_min_temp = cfg.get("anchor_min_maxtemp_f", 105.0)
    floor_requires_anchor_support = bool(cfg.get("floor_requires_anchor_support", True))

    # New guardrail config values
    base_forecast_threshold_mwh = float(cfg.get("base_forecast_threshold_mwh", np.inf))
    max_correction_mwh_if_base_high = float(
        cfg.get("max_correction_mwh_if_base_high", cap)
    )
    warmer_delta_threshold_mwh = float(cfg.get("warmer_delta_threshold_mwh", np.inf))
    max_correction_mwh_if_warmer_delta_high = float(
        cfg.get("max_correction_mwh_if_warmer_delta_high", cap)
    )

    valid_scope = scope & ~cooling & base.notna()
    for date, idx in out.groupby(row_date, dropna=False).groups.items():
        day_index = pd.Index(idx)
        day_scope = valid_scope.loc[day_index]
        if not day_scope.any():
            continue
        candidate_idx = day_scope[day_scope].index
        base_candidates = base.loc[candidate_idx].dropna()
        if base_candidates.empty:
            continue
        peak_idx = base_candidates.idxmax()
        base_peak = float(base.loc[peak_idx])
        base_peak_hour = float(row_hour.loc[peak_idx])
        day_temp = float(daily_max.loc[candidate_idx].max())
        day_ramp = float(ramp.loc[candidate_idx].max())
        day_cloud = float(cloud.loc[candidate_idx].max())
        day_forecast = float(forecast_day.loc[peak_idx])
        day_month = (
            int(row_month.loc[peak_idx])
            if pd.notna(row_month.loc[peak_idx])
            else np.nan
        )
        attrs = {
            "HorizonBucket": _horizon_bucket(day_forecast),
            "Month": day_month,
            "DailyMaxTempBucket": _temp_bucket(day_temp),
            "CloudCoverBucket": _cloud_bucket(day_cloud),
        }
        lookup = _lookup_row(artifact, attrs) or {}
        learned_residual = float(
            lookup.get(
                "PeakResidual_MWH", metadata.get("global_peak_residual_mwh", 0.0)
            )
        )
        same7_residual = float(
            lookup.get(
                "SameHour7Residual_MWH",
                metadata.get("global_samehour7_residual_mwh", learned_residual),
            )
        )
        lag24_residual = float(
            lookup.get(
                "Lag24Residual_MWH",
                metadata.get("global_lag24_residual_mwh", learned_residual),
            )
        )
        lag24_slope = float(
            lookup.get(
                "Lag24RampSlope_MWH_Per_F",
                metadata.get("global_lag24_ramp_slope_mwh_per_f", 0.0),
            )
        )
        lag24_slope = float(
            np.clip(
                lag24_slope,
                -float(cfg.get("lag24_ramp_slope_cap_mwh_per_f", 4.0)),
                float(cfg.get("lag24_ramp_slope_cap_mwh_per_f", 4.0)),
            )
        )

        if (
            block_negative_learned
            and not anchorless_fallback
            and learned_residual < min_learned_residual
        ):
            source.loc[candidate_idx] = str(
                cfg.get(
                    "negative_learned_source", "hot_ramp_peak_negative_learned_residual"
                )
            )
            continue

        anchor_supported = (
            anchorless_fallback
            or not anchor_support_guard
            or learned_residual >= anchor_min_learned
            or (anchor_min_temp is not None and day_temp >= float(anchor_min_temp))
        )

        targets = [base_peak + max(learned_residual, 0.0)]
        historical_targets: list[float] = []
        same7_peak = (
            float(same7.loc[peak_idx]) if pd.notna(same7.loc[peak_idx]) else np.nan
        )
        lag24_peak = (
            float(lag24.loc[peak_idx]) if pd.notna(lag24.loc[peak_idx]) else np.nan
        )
        if np.isfinite(same7_peak):
            historical_targets.append(same7_peak + same7_residual)
        if np.isfinite(lag24_peak):
            historical_targets.append(lag24_peak + lag24_residual)
            historical_targets.append(lag24_peak + lag24_slope * max(day_ramp, 0.0))
        recent_anchor = float(metadata.get("recent_hot_peak_anchor_mwh", np.nan))
        if np.isfinite(recent_anchor):
            historical_targets.append(recent_anchor)
        if anchor_supported:
            targets.extend(historical_targets)
        day_warmer_delta = (
            float(warmer_delta.loc[candidate_idx].max()) if len(candidate_idx) else 0.0
        )
        if day_warmer_delta > 0.0:
            targets.append(base_peak + day_warmer_delta * warmer_fraction)

        valid_targets = [t for t in targets if np.isfinite(t)]
        if not valid_targets:
            source.loc[candidate_idx] = "hot_ramp_peak_no_anchor"
            continue
        target = max(valid_targets)
        raw_correction = target - base_peak
        day_strong = bool(strong.loc[candidate_idx].any())
        day_floor = strong_floor if day_strong else floor
        if (
            floor_requires_anchor_support
            and day_floor > 0.0
            and not anchor_supported
            and not anchorless_fallback
            and raw_correction < day_floor
        ):
            source.loc[candidate_idx] = str(
                cfg.get(
                    "insufficient_anchor_support_source",
                    "hot_ramp_peak_insufficient_anchor_support",
                )
            )
            continue
        if day_floor > 0.0 and raw_correction > -1e-9:
            raw_correction = max(raw_correction, day_floor)
        day_cap = strong_cap if day_strong else cap
        raw_correction = float(np.clip(raw_correction, 0.0, day_cap))

        # Apply new guardrails
        base_high_guard_applied = 0
        if base_peak > base_forecast_threshold_mwh:
            raw_correction = min(raw_correction, max_correction_mwh_if_base_high)
            base_high_guard_applied = 1

        warmer_delta_high_guard_applied = 0
        if day_warmer_delta > warmer_delta_threshold_mwh:
            raw_correction = min(
                raw_correction, max_correction_mwh_if_warmer_delta_high
            )
            warmer_delta_high_guard_applied = 1

        if raw_correction < min_abs:
            source.loc[candidate_idx] = str(
                cfg.get("below_min_source", "hot_ramp_peak_below_min_correction")
            )
            continue

        adjustment_idx = day_index[
            row_hour.loc[day_index].astype("Int64").isin(hours).fillna(False)
        ]
        adjustment_idx = pd.Index(
            [i for i in adjustment_idx if bool(valid_scope.loc[i])]
        )
        if len(adjustment_idx) == 0:
            source.loc[candidate_idx] = "hot_ramp_peak_no_adjustment_rows"
            continue
        timing = _select_hot_ramp_timing_hour(
            out,
            cfg,
            candidate_idx=candidate_idx,
            row_hour=row_hour,
            day_strong=day_strong,
            day_temp=day_temp,
            day_ramp=day_ramp,
            base_peak_hour=base_peak_hour,
        )
        selected_peak_hour = float(timing["selected_peak_hour"])
        selector_cfg = cfg.get("peak_timing_selector", {}) or {}
        correction.loc[adjustment_idx] = _timed_peak_correction(
            base=base,
            row_hour=row_hour,
            adjustment_idx=adjustment_idx,
            base_peak=base_peak,
            raw_correction=raw_correction,
            selected_peak_hour=selected_peak_hour,
            spread_hours=spread_hours,
            selector_cfg=selector_cfg,
            timing_override=bool(timing["override"]),
        )
        source.loc[adjustment_idx] = str(cfg.get("source", "hot_ramp_peak_shadow"))
        out.loc[day_index, "Hot_Ramp_Peak_Base_DailyPeak_MWH"] = base_peak
        out.loc[day_index, "Hot_Ramp_Peak_Predicted_Target_MWH"] = (
            base_peak + raw_correction
        )
        out.loc[day_index, "Hot_Ramp_Peak_Predicted_PeakHour"] = selected_peak_hour
        out.loc[day_index, "Hot_Ramp_Peak_Base_PeakHour"] = base_peak_hour
        out.loc[day_index, "Hot_Ramp_Peak_Timing_Selected_PeakHour"] = (
            selected_peak_hour
        )
        out.loc[day_index, "Hot_Ramp_Peak_Timing_Source"] = str(timing["source"])
        out.loc[day_index, "Hot_Ramp_Peak_Timing_Override_Flag"] = int(
            timing["override"]
        )
        out.loc[day_index, "Hot_Ramp_Peak_Timing_Block_Source"] = str(
            timing["block_source"]
        )
        out.loc[day_index, "Hot_Ramp_Peak_Timing_Confidence_MWH"] = float(
            timing["confidence"]
        )
        out.loc[day_index, "Hot_Ramp_Peak_Learned_Residual_MWH"] = learned_residual
        out.loc[day_index, "Hot_Ramp_Peak_Recent_Anchor_MWH"] = recent_anchor
        out.loc[day_index, "Hot_Ramp_Peak_SameHour7_Anchor_MWH"] = (
            same7_peak + same7_residual if np.isfinite(same7_peak) else np.nan
        )
        out.loc[day_index, "Hot_Ramp_Peak_Lag24_Anchor_MWH"] = (
            max(
                lag24_peak + lag24_residual,
                lag24_peak + lag24_slope * max(day_ramp, 0.0),
            )
            if np.isfinite(lag24_peak)
            else np.nan
        )
        out.loc[day_index, "Hot_Ramp_Peak_WarmerScenario_Delta_MWH"] = day_warmer_delta
        out.loc[day_index, "Hot_Ramp_Peak_Base_High_Guard_Applied"] = (
            base_high_guard_applied
        )
        out.loc[day_index, "Hot_Ramp_Peak_Warmer_Delta_High_Guard_Applied"] = (
            warmer_delta_high_guard_applied
        )

    correction, source = _apply_hot_ramp_targeted_missing_slices(
        out,
        cfg,
        targeted_rules=enabled_targeted_rules,
        row_hour=row_hour,
        row_month=row_month,
        forecast_day=forecast_day,
        daily_max=daily_max,
        cloud=cloud,
        base=base,
        same7=same7,
        warmer_delta=warmer_delta,
        correction=correction,
        source=source,
        cooling=cooling,
        guard_cfg=guard_cfg,
    )

    return _finalize_hot_ramp_peak_capture(
        out,
        base=base,
        correction=correction,
        source=source,
        shadow_mode=shadow_mode,
        forecast_col=forecast_col,
        also_update_cols=also_update_cols,
    )


def apply_heat_persistence_peak_capture(
    df: pd.DataFrame,
    artifact: dict | None,
    config: dict | None,
    *,
    forecast_col: str = "Final_Forecast_MWH",
    also_update_cols: tuple[str, ...] = (),
    force_shadow: bool | None = None,
    evaluation_mode: str = "shadow",
) -> pd.DataFrame:
    out = df.copy()
    cfg = _heat_persistence_cfg(config)
    shadow_mode = (
        bool(cfg.get("shadow_mode", True))
        if force_shadow is None
        else bool(force_shadow)
    )
    base = _base_forecast(out, forecast_col=forecast_col)
    out = _init_heat_persistence_columns(
        out,
        base,
        cfg=cfg,
        source="disabled",
        evaluation_mode=evaluation_mode,
        shadow_mode=shadow_mode,
    )
    if out.empty or not bool(cfg.get("enabled", False)):
        return out
    if not artifact or not artifact.get("metadata"):
        if shadow_mode and bool(cfg.get("allow_anchorless_shadow_fallback", False)):
            artifact = _heat_persistence_anchorless_shadow_artifact(cfg, forecast_col)
        else:
            out["Heat_Persistence_Peak_Source"] = "insufficient_history"
            return out

    dt = _local_datetime(out)
    row_date = _date(out, dt=dt)
    row_hour = _hour(out, dt=dt)
    row_month = _month(out, dt=dt)
    forecast_day = _forecast_day(out, dt=dt)
    daily_max = _daily_max(out)
    dailymax_3day = _dailymax_3day_mean(out, dt=dt)
    consecutive_extreme = _consecutive_extreme_days100(out, dt=dt)
    cloud = _cloud_norm(out)
    same7 = _optional_num(
        out,
        "MWH_SameHour7DayMean",
        "Baseline_Rolling7DaySameHourAvg_MWH",
        default=np.nan,
    )
    lag24 = _optional_num(
        out, "MWH_Lag24", "Baseline_SameHourYesterday_MWH", default=np.nan
    )
    warmer = _optional_num(out, "WeatherScenario_warmer_P50_MWH", default=np.nan)
    warmer_delta = (warmer - base).clip(lower=0.0).fillna(0.0)

    scope, cooling = _persistence_scope_mask(out, cfg)
    guard_cfg = (
        cfg.get("cooling_underway_guard", {}) or cfg.get("cooling_guard", {}) or {}
    )
    strong = (
        scope
        & daily_max.ge(
            float(cfg.get("strong_min_maxtemp_f", cfg.get("min_maxtemp_f", 100.0)))
        ).fillna(False)
        & consecutive_extreme.ge(
            float(
                cfg.get(
                    "strong_min_consecutive_extreme_days100",
                    cfg.get("min_consecutive_extreme_days100", 3.0),
                )
            )
        ).fillna(False)
    )
    out.loc[scope, "Heat_Persistence_Peak_Scope_Flag"] = 1
    out.loc[strong, "Heat_Persistence_Peak_Strong_Flag"] = 1
    out["Heat_Persistence_Peak_ConsecutiveExtremeHotDays100"] = consecutive_extreme
    out["Heat_Persistence_Peak_DailyMaxTemp3DayMean_F"] = dailymax_3day

    correction = pd.Series(0.0, index=out.index, dtype=float)
    source = pd.Series("out_of_scope", index=out.index, dtype="object")
    source.loc[scope & cooling] = str(
        cfg.get("cooling_blocked_source")
        or guard_cfg.get("blocked_source")
        or "heat_persistence_peak_cooling_underway_blocked"
    )

    hours = [int(h) for h in cfg.get("hours", [16, 17, 18, 19, 20])]
    spread_hours = float(cfg.get("spread_hours", 1.0))
    cap = float(cfg.get("cap_mwh", 9.0))
    strong_cap = float(cfg.get("strong_cap_mwh", cap))
    min_abs = float(cfg.get("min_abs_correction_mwh", 0.25))
    floor = float(cfg.get("persistence_floor_mwh", 0.0))
    strong_floor = float(cfg.get("strong_persistence_floor_mwh", 9.0))
    warmer_fraction = float(cfg.get("warmer_scenario_fraction", 0.25))
    floor_without_anchor = bool(cfg.get("floor_applies_without_positive_anchor", True))
    metadata = artifact.get("metadata", {}) or {}

    valid_scope = scope & ~cooling & base.notna()
    for date, idx in out.groupby(row_date, dropna=False).groups.items():
        day_index = pd.Index(idx)
        day_scope = valid_scope.loc[day_index]
        if not day_scope.any():
            continue
        candidate_idx = day_scope[day_scope].index
        base_candidates = base.loc[candidate_idx].dropna()
        if base_candidates.empty:
            continue
        peak_idx = base_candidates.idxmax()
        base_peak = float(base.loc[peak_idx])
        peak_hour = float(row_hour.loc[peak_idx])
        day_temp = float(daily_max.loc[candidate_idx].max())
        day_cloud = float(cloud.loc[candidate_idx].max())
        day_consecutive = float(consecutive_extreme.loc[candidate_idx].max())
        day_forecast = float(forecast_day.loc[peak_idx])
        day_month = (
            int(row_month.loc[peak_idx])
            if pd.notna(row_month.loc[peak_idx])
            else np.nan
        )
        attrs = {
            "HorizonBucket": _horizon_bucket(day_forecast),
            "Month": day_month,
            "DailyMaxTempBucket": _temp_bucket(day_temp),
            "CloudCoverBucket": _cloud_bucket(day_cloud),
            "PersistenceBucket": _persistence_bucket(day_consecutive),
        }
        lookup = _lookup_row(artifact, attrs) or {}
        learned_residual = float(
            lookup.get(
                "PeakResidual_MWH", metadata.get("global_peak_residual_mwh", 0.0)
            )
        )
        same7_residual = float(
            lookup.get(
                "SameHour7Residual_MWH",
                metadata.get("global_samehour7_residual_mwh", learned_residual),
            )
        )
        lag24_residual = float(
            lookup.get(
                "Lag24Residual_MWH",
                metadata.get("global_lag24_residual_mwh", learned_residual),
            )
        )

        targets = [base_peak + learned_residual]
        same7_peak = (
            float(same7.loc[peak_idx]) if pd.notna(same7.loc[peak_idx]) else np.nan
        )
        lag24_peak = (
            float(lag24.loc[peak_idx]) if pd.notna(lag24.loc[peak_idx]) else np.nan
        )
        if np.isfinite(same7_peak):
            targets.append(same7_peak + same7_residual)
        if np.isfinite(lag24_peak):
            targets.append(lag24_peak + lag24_residual)
        recent_anchor = float(
            metadata.get("recent_heat_persistence_peak_anchor_mwh", np.nan)
        )
        if np.isfinite(recent_anchor):
            targets.append(recent_anchor)
        day_warmer_delta = (
            float(warmer_delta.loc[candidate_idx].max()) if len(candidate_idx) else 0.0
        )
        if day_warmer_delta > 0.0:
            targets.append(base_peak + day_warmer_delta * warmer_fraction)

        valid_targets = [t for t in targets if np.isfinite(t)]
        target = max(valid_targets) if valid_targets else base_peak
        raw_correction = target - base_peak
        # `targets` always has at least [base_peak + learned_residual], which is finite even
        # when nothing actually anchors a positive correction (learned_residual defaults to
        # 0.0), so `valid_targets` can never be empty. The real "no positive anchor" signal is
        # whether any target implies load above the base forecast.
        has_positive_anchor = raw_correction > 1e-9
        if not has_positive_anchor and not floor_without_anchor:
            source.loc[candidate_idx] = "heat_persistence_peak_no_anchor"
            continue
        day_strong = bool(strong.loc[candidate_idx].any())
        day_floor = strong_floor if day_strong else floor
        if day_floor > 0.0 and (floor_without_anchor or raw_correction > -1e-9):
            raw_correction = max(raw_correction, day_floor)
        day_cap = strong_cap if day_strong else cap
        raw_correction = float(np.clip(raw_correction, 0.0, day_cap))
        if raw_correction < min_abs:
            source.loc[candidate_idx] = str(
                cfg.get(
                    "below_min_source", "heat_persistence_peak_below_min_correction"
                )
            )
            continue

        adjustment_idx = day_index[
            row_hour.loc[day_index].astype("Int64").isin(hours).fillna(False)
        ]
        adjustment_idx = pd.Index(
            [i for i in adjustment_idx if bool(valid_scope.loc[i])]
        )
        if len(adjustment_idx) == 0:
            source.loc[candidate_idx] = "heat_persistence_peak_no_adjustment_rows"
            continue
        weights = _hour_weights(
            row_hour.loc[adjustment_idx], peak_hour, spread_hours=spread_hours
        )
        correction.loc[adjustment_idx] = raw_correction * weights
        source.loc[adjustment_idx] = str(
            cfg.get("source", "heat_persistence_peak_shadow")
        )
        out.loc[day_index, "Heat_Persistence_Peak_Base_DailyPeak_MWH"] = base_peak
        out.loc[day_index, "Heat_Persistence_Peak_Predicted_Target_MWH"] = (
            base_peak + raw_correction
        )
        out.loc[day_index, "Heat_Persistence_Peak_Predicted_PeakHour"] = peak_hour
        out.loc[day_index, "Heat_Persistence_Peak_Learned_Residual_MWH"] = (
            learned_residual
        )
        out.loc[day_index, "Heat_Persistence_Peak_Recent_Anchor_MWH"] = recent_anchor
        out.loc[day_index, "Heat_Persistence_Peak_SameHour7_Anchor_MWH"] = (
            same7_peak + same7_residual if np.isfinite(same7_peak) else np.nan
        )
        out.loc[day_index, "Heat_Persistence_Peak_Lag24_Anchor_MWH"] = (
            lag24_peak + lag24_residual if np.isfinite(lag24_peak) else np.nan
        )
        out.loc[day_index, "Heat_Persistence_Peak_WarmerScenario_Delta_MWH"] = (
            day_warmer_delta
        )

    correction = correction.where(base.notna(), 0.0)
    adjusted = (base + correction).clip(lower=0.0)
    source.loc[base.isna()] = "invalid_base_forecast"
    out["Heat_Persistence_Peak_Correction_MWH"] = correction
    out["Heat_Persistence_Peak_Shadow_Forecast_MWH"] = adjusted
    out["Heat_Persistence_Peak_Correction_Applied_Flag"] = correction.ne(0.0).astype(
        int
    )
    out["Heat_Persistence_Peak_Source"] = source

    actual = _optional_num(out, "Actual_MWH", "Actual", default=np.nan)
    base_residual = actual - base
    residual = actual - adjusted
    out["Heat_Persistence_Peak_Residual_MWH"] = residual
    out["Heat_Persistence_Peak_AbsError_MWH"] = residual.abs()
    out["Heat_Persistence_Peak_Delta_AbsError_MWH"] = (
        residual.abs() - base_residual.abs()
    )

    if not shadow_mode and forecast_col in out.columns:
        out[forecast_col] = adjusted
        if (
            forecast_col == "Final_Backtest_Forecast_MWH"
            and "Final_Forecast_MWH" in out.columns
        ):
            out["Final_Forecast_MWH"] = adjusted
        for col in also_update_cols:
            if col in out.columns and col != forecast_col:
                out[col] = adjusted
        if (
            forecast_col in {"Final_Backtest_Forecast_MWH", "Final_Forecast_MWH"}
            and actual.notna().any()
        ):
            out["Final_Residual_MWH"] = actual - _as_num(out[forecast_col], out.index)
            out["Final_AbsError_MWH"] = out["Final_Residual_MWH"].abs()
            out["Final_APE"] = np.where(
                actual.abs() > 1e-9,
                out["Final_AbsError_MWH"] / actual.abs() * 100.0,
                np.nan,
            )
    return out


def hot_ramp_peak_capture_summary(
    backtest_df: pd.DataFrame | None, artifact: dict | None, config: dict | None
) -> dict:
    cfg = _cfg(config)
    summary: dict[str, Any] = {
        "enabled": bool(cfg.get("enabled", False)),
        "shadow_mode": bool(cfg.get("shadow_mode", True)),
        "model_version": str(
            cfg.get("model_version", "hot_ramp_peak_capture_shadow_v1")
        ),
    }
    if artifact and artifact.get("metadata"):
        summary["artifact"] = dict(artifact.get("metadata") or {})
    if (
        backtest_df is None
        or backtest_df.empty
        or "Hot_Ramp_Peak_Shadow_Forecast_MWH" not in backtest_df.columns
    ):
        summary["evaluation_rows"] = 0
        return summary

    actual = _optional_num(backtest_df, "Actual_MWH", "Actual", default=np.nan)
    base = _optional_num(backtest_df, "Hot_Ramp_Peak_Base_Forecast_MWH", default=np.nan)
    shadow = _optional_num(
        backtest_df, "Hot_Ramp_Peak_Shadow_Forecast_MWH", default=np.nan
    )
    applied = _optional_num(
        backtest_df, "Hot_Ramp_Peak_Correction_Applied_Flag", default=0.0
    ).eq(1)
    valid = actual.notna() & base.notna() & shadow.notna() & applied
    summary["evaluation_rows"] = int(valid.sum())
    if not valid.any():
        return summary
    base_abs = (actual[valid] - base[valid]).abs()
    shadow_abs = (actual[valid] - shadow[valid]).abs()
    delta = shadow_abs - base_abs
    residual = actual[valid] - shadow[valid]
    summary["baseline_mae_mwh"] = float(base_abs.mean())
    summary["hot_ramp_peak_shadow_mae_mwh"] = float(shadow_abs.mean())
    summary["delta_mae_mwh"] = float(delta.mean())
    summary["bias_mwh"] = float(residual.mean())
    summary["underforecast_rate_pct"] = float((residual > 0.0).mean() * 100.0)
    summary["improved_rows"] = int((delta < 0.0).sum())
    summary["worsened_rows"] = int((delta > 0.0).sum())
    summary["mean_correction_mwh"] = float(
        _optional_num(
            backtest_df.loc[valid], "Hot_Ramp_Peak_Correction_MWH", default=0.0
        ).mean()
    )
    summary["source_counts"] = {
        str(k): int(v)
        for k, v in backtest_df["Hot_Ramp_Peak_Source"]
        .value_counts(dropna=False)
        .head(20)
        .items()
    }
    return summary


def heat_persistence_peak_capture_summary(
    backtest_df: pd.DataFrame | None,
    artifact: dict | None,
    config: dict | None,
) -> dict:
    cfg = _heat_persistence_cfg(config)
    summary: dict[str, Any] = {
        "enabled": bool(cfg.get("enabled", False)),
        "shadow_mode": bool(cfg.get("shadow_mode", True)),
        "model_version": str(
            cfg.get("model_version", "heat_persistence_peak_capture_shadow_v1")
        ),
    }
    if artifact and artifact.get("metadata"):
        summary["artifact"] = dict(artifact.get("metadata") or {})
    if (
        backtest_df is None
        or backtest_df.empty
        or "Heat_Persistence_Peak_Shadow_Forecast_MWH" not in backtest_df.columns
    ):
        summary["evaluation_rows"] = 0
        return summary

    actual = _optional_num(backtest_df, "Actual_MWH", "Actual", default=np.nan)
    base = _optional_num(
        backtest_df, "Heat_Persistence_Peak_Base_Forecast_MWH", default=np.nan
    )
    shadow = _optional_num(
        backtest_df, "Heat_Persistence_Peak_Shadow_Forecast_MWH", default=np.nan
    )
    applied = _optional_num(
        backtest_df, "Heat_Persistence_Peak_Correction_Applied_Flag", default=0.0
    ).eq(1)
    valid = actual.notna() & base.notna() & shadow.notna() & applied
    summary["evaluation_rows"] = int(valid.sum())
    if not valid.any():
        return summary
    base_abs = (actual[valid] - base[valid]).abs()
    shadow_abs = (actual[valid] - shadow[valid]).abs()
    delta = shadow_abs - base_abs
    residual = actual[valid] - shadow[valid]
    summary["baseline_mae_mwh"] = float(base_abs.mean())
    summary["heat_persistence_peak_shadow_mae_mwh"] = float(shadow_abs.mean())
    summary["delta_mae_mwh"] = float(delta.mean())
    summary["bias_mwh"] = float(residual.mean())
    summary["underforecast_rate_pct"] = float((residual > 0.0).mean() * 100.0)
    summary["improved_rows"] = int((delta < 0.0).sum())
    summary["worsened_rows"] = int((delta > 0.0).sum())
    summary["mean_correction_mwh"] = float(
        _optional_num(
            backtest_df.loc[valid], "Heat_Persistence_Peak_Correction_MWH", default=0.0
        ).mean()
    )
    summary["source_counts"] = {
        str(k): int(v)
        for k, v in backtest_df["Heat_Persistence_Peak_Source"]
        .value_counts(dropna=False)
        .head(20)
        .items()
    }
    return summary
