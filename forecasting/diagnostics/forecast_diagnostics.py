from __future__ import annotations

import json
import math
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from forecasting.forecast.uncertainty_bands import (
    _band_risk_multiplier,
    _hot_bucket_band_floor,
    _prep as _prep_band_inputs,
)


RAW_STAGE = "raw_xgb_lgb_production"
FINAL_STAGE = "final_corrected_production"


def _as_num(s: pd.Series | Any) -> pd.Series:
    return pd.to_numeric(s, errors="coerce")


def _safe_rmse(residual: pd.Series) -> float:
    r = _as_num(residual).dropna()
    return float(np.sqrt(np.mean(np.square(r)))) if len(r) else np.nan


def _safe_mape(actual: pd.Series, forecast: pd.Series) -> float:
    a = _as_num(actual).astype(float)
    f = _as_num(forecast).astype(float)
    mask = a.abs() > 1e-9
    if not mask.any():
        return np.nan
    return float((np.abs((a[mask] - f[mask]) / a[mask])).mean() * 100.0)


def _season_from_month(m: int) -> str:
    m = int(m)
    if m in (12, 1, 2):
        return "Winter"
    if m in (3, 4, 5):
        return "Spring"
    if m in (6, 7, 8, 9):
        return "Summer"
    return "Fall"


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


def _temp_bucket(values: pd.Series) -> pd.Series:
    temp = _as_num(values)
    return pd.cut(
        temp,
        bins=[-999, 65, 75, 85, 90, 95, 100, 105, 999],
        labels=["<65", "65-75", "75-85", "85-90", "90-95", "95-100", "100-105", "105+"],
        include_lowest=True,
    ).astype("object")


def _add_readable_bins(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()

    if "Temperature_DailyMax" in out.columns:
        temp = _as_num(out["Temperature_DailyMax"])
    elif "Temperature" in out.columns:
        temp = _as_num(out["Temperature"])
    else:
        temp = pd.Series(np.nan, index=out.index)
    if "DailyMaxTempBucket" not in out.columns:
        out["DailyMaxTempBucket"] = _temp_bucket(temp)

    if "CloudCoverBucket" not in out.columns:
        if "CloudCover_Norm" in out.columns:
            cloud = _as_num(out["CloudCover_Norm"])
            if cloud.max(skipna=True) <= 1.5:
                bins = [-0.001, 0.20, 0.40, 0.60, 0.80, 1.001]
            else:
                bins = [-0.001, 20, 40, 60, 80, 100.001]
            out["CloudCoverBucket"] = pd.cut(
                cloud,
                bins=bins,
                labels=["Clear/Low", "Some Clouds", "Partly Cloudy", "Mostly Cloudy", "Overcast"],
                include_lowest=True,
            ).astype("object")
        else:
            out["CloudCoverBucket"] = np.nan

    if "BTMSolarBucket" not in out.columns:
        if "BTM_Solar_Proxy_MW" in out.columns:
            solar = _as_num(out["BTM_Solar_Proxy_MW"]).fillna(0.0)
            positive = solar[solar > 0]
            out["BTMSolarBucket"] = "None"
            if len(positive) >= 10 and positive.nunique() >= 4:
                q1, q2, q3 = positive.quantile([0.25, 0.50, 0.75]).tolist()
                out.loc[(solar > 0) & (solar <= q1), "BTMSolarBucket"] = "Low"
                out.loc[(solar > q1) & (solar <= q2), "BTMSolarBucket"] = "Medium-Low"
                out.loc[(solar > q2) & (solar <= q3), "BTMSolarBucket"] = "Medium-High"
                out.loc[solar > q3, "BTMSolarBucket"] = "High"
            elif len(positive):
                med = positive.median()
                out.loc[(solar > 0) & (solar <= med), "BTMSolarBucket"] = "Low/Medium"
                out.loc[solar > med, "BTMSolarBucket"] = "High"
        else:
            out["BTMSolarBucket"] = np.nan

    if "SolarLossBucket" not in out.columns:
        loss_col = "BTM_Solar_Loss_From_ClearSky_MW" if "BTM_Solar_Loss_From_ClearSky_MW" in out.columns else "Midday_Overcast_Solar_Loss_MW"
        if loss_col in out.columns:
            loss = _as_num(out[loss_col]).fillna(0.0)
            positive = loss[loss > 0]
            out["SolarLossBucket"] = "None"
            if len(positive) >= 10 and positive.nunique() >= 4:
                q1, q2, q3 = positive.quantile([0.25, 0.50, 0.75]).tolist()
                out.loc[(loss > 0) & (loss <= q1), "SolarLossBucket"] = "Low"
                out.loc[(loss > q1) & (loss <= q2), "SolarLossBucket"] = "Medium"
                out.loc[(loss > q2) & (loss <= q3), "SolarLossBucket"] = "High"
                out.loc[loss > q3, "SolarLossBucket"] = "Extreme"
            elif len(positive):
                med = positive.median()
                out.loc[(loss > 0) & (loss <= med), "SolarLossBucket"] = "Low/Medium"
                out.loc[loss > med, "SolarLossBucket"] = "High"
        else:
            out["SolarLossBucket"] = np.nan

    return out


def _add_baselines(out: pd.DataFrame) -> pd.DataFrame:
    """Add simple diagnostic baselines for skill-score comparisons.

    These baselines are not production forecasts for the first few holdout hours because they
    use observed holdout values after they become available. They are diagnostic comparators.
    """
    out = out.sort_values("DT").reset_index(drop=True).copy()
    actual = _as_num(out["Actual_MWH"])
    out["Baseline_SameHourYesterday_MWH"] = actual.shift(24)
    out["Baseline_SameHour7DaysAgo_MWH"] = actual.shift(168)
    # Rolling same-hour average using prior observations only.
    out["Baseline_Rolling7DaySameHourAvg_MWH"] = (
        out.groupby("Hour", dropna=False)["Actual_MWH"]
        .transform(lambda s: _as_num(s).shift(1).rolling(window=7, min_periods=2).mean())
    )
    return out


def prep_backtest(backtest_df: pd.DataFrame) -> pd.DataFrame:
    """Normalize the backtest frame into a diagnostics-friendly schema."""
    if backtest_df is None or backtest_df.empty:
        return pd.DataFrame()

    out = backtest_df.copy()
    out["DT"] = pd.to_datetime(out["DT"], errors="coerce")
    out = out.dropna(subset=["DT"]).sort_values("DT").reset_index(drop=True)

    if "Actual_MWH" not in out.columns and "Actual" in out.columns:
        out["Actual_MWH"] = out["Actual"]
    if "Raw_Forecast_MWH" not in out.columns and "Forecast" in out.columns:
        out["Raw_Forecast_MWH"] = out["Forecast"]

    out["Actual_MWH"] = _as_num(out.get("Actual_MWH", np.nan))
    out["Raw_Forecast_MWH"] = _as_num(out.get("Raw_Forecast_MWH", np.nan))
    out["Residual_MWH"] = _as_num(out.get("Residual_MWH", out["Actual_MWH"] - out["Raw_Forecast_MWH"]))
    out["AbsError_MWH"] = _as_num(out.get("AbsError_MWH", out["Residual_MWH"].abs()))
    out["APE"] = _as_num(out.get("APE", np.where(out["Actual_MWH"].abs() > 1e-9, out["AbsError_MWH"] / out["Actual_MWH"].abs() * 100.0, np.nan)))

    out["Date"] = out["DT"].dt.date.astype(str)
    if "Forecast_Lead_Hour" in out.columns:
        default_lead = pd.Series(np.arange(1, len(out) + 1, dtype=int), index=out.index)
        out["Forecast_Lead_Hour"] = _as_num(out["Forecast_Lead_Hour"]).fillna(default_lead).astype(int)
    else:
        out["Forecast_Lead_Hour"] = np.arange(1, len(out) + 1, dtype=int)
    if "Forecast_Day" in out.columns:
        out["Forecast_Day"] = _as_num(out["Forecast_Day"]).fillna(((out["Forecast_Lead_Hour"] - 1) // 24 + 1)).astype(int)
    else:
        out["Forecast_Day"] = ((out["Forecast_Lead_Hour"] - 1) // 24 + 1).astype(int)
    out["Hour"] = _as_num(out.get("Hour", out["DT"].dt.hour)).fillna(out["DT"].dt.hour).astype(int)
    out["Month"] = _as_num(out.get("Month", out["DT"].dt.month)).fillna(out["DT"].dt.month).astype(int)
    out["DOW"] = _as_num(out.get("DOW", out["DT"].dt.dayofweek)).fillna(out["DT"].dt.dayofweek).astype(int)
    out["Season"] = out.get("Season", out["Month"].map(_season_from_month))
    out["HourGroup"] = out.get("HourGroup", out["Hour"].map(_hour_group))
    out["IsWeekend"] = _as_num(out.get("IsWeekend", out["DOW"].isin([5, 6]).astype(int))).fillna(0).astype(int)
    if "IsHoliday" not in out.columns:
        out["IsHoliday"] = 0

    # Final corrected stage is the V12 production backtest simulation when available.
    if "Final_Backtest_Forecast_MWH" not in out.columns:
        if "Final_Forecast_MWH" in out.columns:
            out["Final_Backtest_Forecast_MWH"] = out["Final_Forecast_MWH"]
        elif "Recent_Corrected_Forecast_MWH" in out.columns:
            out["Final_Backtest_Forecast_MWH"] = out["Recent_Corrected_Forecast_MWH"]

    out["Underforecast_MWH"] = out["Residual_MWH"].clip(lower=0.0)
    out["Overforecast_MWH"] = (-out["Residual_MWH"]).clip(lower=0.0)
    out["Forecast_Error_MWH"] = out["Raw_Forecast_MWH"] - out["Actual_MWH"]
    out["Underforecast_Flag"] = (out["Residual_MWH"] > 0).astype(int)
    out["Overforecast_Flag"] = (out["Residual_MWH"] < 0).astype(int)

    out = _add_readable_bins(out)
    out = _add_baselines(out)
    return out


def _available_stage_columns(df: pd.DataFrame) -> dict[str, str]:
    candidates = {
        "xgb_component": "XGB_Pred_MWH",
        "lgb_component": "LGB_Pred_MWH",
        "prophet_benchmark": "Prophet_Pred_MWH",
        "catboost_benchmark": "CatBoost_Pred_MWH",
        RAW_STAGE: "Raw_Forecast_MWH",
        "targeted_residual_meta_adjusted": "Targeted_Meta_Adjusted_Forecast_MWH",
        "residual_calibrated": "Residual_Calibrated_Forecast_MWH",
        "heat_adjusted": "Heat_Adjusted_Forecast_MWH",
        "warm_ramp_adjusted": "Warm_Ramp_Adjusted_Forecast_MWH",
        "cloud_solar_adjusted": "Cloud_Solar_Adjusted_Forecast_MWH",
        "peak_risk_adjusted": "Peak_Risk_Adjusted_Forecast_MWH",
        "recent_corrected_simulation": "Recent_Corrected_Forecast_MWH",
        "stage_selected_production": "Stage_Selected_Forecast_MWH",
        FINAL_STAGE: "Final_Backtest_Forecast_MWH",
        "baseline_same_hour_yesterday": "Baseline_SameHourYesterday_MWH",
        "baseline_same_hour_7_days_ago": "Baseline_SameHour7DaysAgo_MWH",
        "baseline_rolling_7day_same_hour_avg": "Baseline_Rolling7DaySameHourAvg_MWH",
    }
    return {name: col for name, col in candidates.items() if col in df.columns and _as_num(df[col]).notna().any()}


def _metric_dict(actual: pd.Series, forecast: pd.Series, label: str | None = None, col: str | None = None) -> dict[str, Any]:
    a = _as_num(actual)
    f = _as_num(forecast)
    mask = a.notna() & f.notna()
    if not mask.any():
        return {}
    residual = a[mask] - f[mask]
    abs_error = residual.abs()
    peak_idx = a[mask].idxmax()
    out = {
        "N": int(mask.sum()),
        "Actual_Mean_MWH": float(a[mask].mean()),
        "Forecast_Mean_MWH": float(f[mask].mean()),
        "Bias_MWH": float(residual.mean()),
        "MAE_MWH": float(abs_error.mean()),
        "RMSE_MWH": _safe_rmse(residual),
        "MAPE_PCT": _safe_mape(a[mask], f[mask]),
        "Underforecast_Rate_PCT": float((residual > 0).mean() * 100.0),
        "P90_AbsError_MWH": float(abs_error.quantile(0.90)),
        "Max_Underforecast_MWH": float(residual.max()),
        "Max_Overforecast_MWH": float((-residual).max()),
        "Actual_Peak_MWH": float(a[mask].max()),
        "Forecast_At_Actual_Peak_MWH": float(f.loc[peak_idx]),
        "Underforecast_At_Actual_Peak_MWH": float(a.loc[peak_idx] - f.loc[peak_idx]),
    }
    if label is not None:
        out["Stage"] = label
    if col is not None:
        out["ForecastColumn"] = col
    return out


def metrics_summary(df: pd.DataFrame) -> dict[str, Any]:
    if df is None or df.empty:
        return {}
    stages = _available_stage_columns(df)
    raw = _metric_dict(df["Actual_MWH"], df[stages.get(RAW_STAGE, "Raw_Forecast_MWH")]) if RAW_STAGE in stages or "Raw_Forecast_MWH" in df else {}
    final_col = stages.get(FINAL_STAGE) or stages.get("recent_corrected_simulation") or stages.get(RAW_STAGE)
    final = _metric_dict(df["Actual_MWH"], df[final_col]) if final_col else {}

    # Preserve legacy top-level fields as raw metrics, but add explicit raw/final blocks.
    out: dict[str, Any] = {
        "row_count": int(len(df)),
        "start_dt": str(pd.to_datetime(df["DT"].min())) if "DT" in df else None,
        "end_dt": str(pd.to_datetime(df["DT"].max())) if "DT" in df else None,
        "primary_metric_stage": FINAL_STAGE,
        "primary_metric_column": final_col,
        "raw_model": raw,
        "final_corrected_model": final,
    }
    for k, v in raw.items():
        if k not in {"Stage", "ForecastColumn", "N"}:
            out[k] = v
    out["legacy_top_level_metrics_are_raw"] = True
    if final:
        out["Final_MAE_MWH"] = final.get("MAE_MWH")
        out["Final_RMSE_MWH"] = final.get("RMSE_MWH")
        out["Final_MAPE_PCT"] = final.get("MAPE_PCT")
        out["Final_Bias_MWH"] = final.get("Bias_MWH")
        out["Final_Underforecast_Rate_PCT"] = final.get("Underforecast_Rate_PCT")
        out["Final_Underforecast_At_Actual_Peak_MWH"] = final.get("Underforecast_At_Actual_Peak_MWH")
    if raw and final:
        out["Final_MAE_Improvement_vs_Raw_MWH"] = raw.get("MAE_MWH", np.nan) - final.get("MAE_MWH", np.nan)
        out["Final_RMSE_Improvement_vs_Raw_MWH"] = raw.get("RMSE_MWH", np.nan) - final.get("RMSE_MWH", np.nan)
        out["Final_Bias_Abs_Improvement_vs_Raw_MWH"] = abs(raw.get("Bias_MWH", np.nan)) - abs(final.get("Bias_MWH", np.nan))
    return out


def _segment_metrics(group: pd.DataFrame, forecast_col: str = "Raw_Forecast_MWH") -> pd.Series:
    actual = _as_num(group["Actual_MWH"])
    forecast = _as_num(group[forecast_col])
    m = _metric_dict(actual, forecast)
    return pd.Series({k: v for k, v in m.items() if k not in {"Stage", "ForecastColumn"}})


def build_metrics_by_group(df: pd.DataFrame, keys: list[str], min_count: int = 1, forecast_col: str = "Raw_Forecast_MWH") -> pd.DataFrame:
    """Fast grouped accuracy metrics for a single forecast column."""
    if df is None or df.empty or forecast_col not in df.columns or not all(k in df.columns for k in keys):
        return pd.DataFrame()

    work = df[keys + ["Actual_MWH", forecast_col]].copy()
    work["Actual_MWH"] = _as_num(work["Actual_MWH"])
    work["Forecast_MWH"] = _as_num(work[forecast_col])
    work = work.dropna(subset=["Actual_MWH", "Forecast_MWH"])
    if work.empty:
        return pd.DataFrame()

    work["Residual_MWH"] = work["Actual_MWH"] - work["Forecast_MWH"]
    work["AbsError_MWH"] = work["Residual_MWH"].abs()
    work["SqResidual_MWH"] = work["Residual_MWH"] ** 2
    work["APE"] = np.where(work["Actual_MWH"].abs() > 1e-9, work["AbsError_MWH"] / work["Actual_MWH"].abs() * 100.0, np.nan)
    work["Underforecast_Flag"] = (work["Residual_MWH"] > 0).astype(float)
    work["OverResidual_MWH"] = (-work["Residual_MWH"]).clip(lower=0.0)

    gb = work.groupby(keys, dropna=False)
    out = gb.agg(
        N=("Residual_MWH", "size"),
        Actual_Mean_MWH=("Actual_MWH", "mean"),
        Forecast_Mean_MWH=("Forecast_MWH", "mean"),
        Bias_MWH=("Residual_MWH", "mean"),
        MAE_MWH=("AbsError_MWH", "mean"),
        MAPE_PCT=("APE", "mean"),
        Underforecast_Rate_PCT=("Underforecast_Flag", lambda s: float(s.mean() * 100.0)),
        P90_AbsError_MWH=("AbsError_MWH", lambda s: float(s.quantile(0.90))),
        Max_Underforecast_MWH=("Residual_MWH", "max"),
        Max_Overforecast_MWH=("OverResidual_MWH", "max"),
        Mean_SqResidual_MWH=("SqResidual_MWH", "mean"),
    ).reset_index()
    out["RMSE_MWH"] = np.sqrt(out.pop("Mean_SqResidual_MWH"))
    out = out[out["N"] >= int(min_count)].copy()
    sort_cols = [c for c in ["MAE_MWH", "N"] if c in out.columns]
    return out.sort_values(sort_cols, ascending=[False, False][:len(sort_cols)]).reset_index(drop=True) if sort_cols else out.reset_index(drop=True)

def build_metrics_by_group_by_stage(df: pd.DataFrame, keys: list[str], min_count: int = 1, stages: dict[str, str] | None = None) -> pd.DataFrame:
    if df is None or df.empty or not all(k in df.columns for k in keys):
        return pd.DataFrame()
    stages = stages or _available_stage_columns(df)
    frames = []
    for stage, col in stages.items():
        seg = build_metrics_by_group(df, keys, min_count=min_count, forecast_col=col)
        if not seg.empty:
            seg.insert(0, "Stage", stage)
            seg.insert(1, "ForecastColumn", col)
            frames.append(seg)
    return pd.concat(frames, ignore_index=True, sort=False) if frames else pd.DataFrame()


def build_backtest_metrics_by_segment(df: pd.DataFrame, min_count: int = 6, forecast_col: str = "Raw_Forecast_MWH") -> pd.DataFrame:
    specs = {
        "Season_Hour": ["Season", "Hour"],
        "Season_HourGroup": ["Season", "HourGroup"],
        "Month_Hour": ["Month", "Hour"],
        "DailyMaxTempBucket_HourGroup": ["DailyMaxTempBucket", "HourGroup"],
        "CloudCoverBucket_HourGroup": ["CloudCoverBucket", "HourGroup"],
        "BTMSolarBucket_HourGroup": ["BTMSolarBucket", "HourGroup"],
        "SolarLossBucket_HourGroup": ["SolarLossBucket", "HourGroup"],
        "CloudSolarEventClass_HourGroup": ["CloudSolarEventClass", "HourGroup"],
        "CloudSolarEventClass_Hour": ["CloudSolarEventClass", "Hour"],
        "Temp_Cloud_HourGroup": ["DailyMaxTempBucket", "CloudCoverBucket", "HourGroup"],
        "Forecast_Day_HourGroup": ["Forecast_Day", "HourGroup"],
        "Weekend_Hour": ["IsWeekend", "Hour"],
        "Holiday_HourGroup": ["IsHoliday", "HourGroup"],
    }
    frames = []
    for segment_name, keys in specs.items():
        seg = build_metrics_by_group(df, keys, min_count=min_count, forecast_col=forecast_col)
        if not seg.empty:
            seg.insert(0, "Segment", segment_name)
            seg.insert(1, "SegmentKeys", "+".join(keys))
            frames.append(seg)
    return pd.concat(frames, ignore_index=True, sort=False) if frames else pd.DataFrame()


def build_backtest_metrics_by_segment_by_stage(df: pd.DataFrame, min_count: int = 6) -> pd.DataFrame:
    stages = _available_stage_columns(df)
    preferred = {k: v for k, v in stages.items() if k in [RAW_STAGE, "targeted_residual_meta_adjusted", "residual_calibrated", "heat_adjusted", "warm_ramp_adjusted", "cloud_solar_adjusted", "peak_risk_adjusted", "recent_corrected_simulation", FINAL_STAGE, "prophet_benchmark", "catboost_benchmark", "baseline_same_hour_yesterday", "baseline_rolling_7day_same_hour_avg"]}
    frames = []
    for stage, col in preferred.items():
        seg = build_backtest_metrics_by_segment(df, min_count=min_count, forecast_col=col)
        if not seg.empty:
            seg.insert(0, "Stage", stage)
            seg.insert(1, "ForecastColumn", col)
            frames.append(seg)
    return pd.concat(frames, ignore_index=True, sort=False) if frames else pd.DataFrame()


def build_daily_peak_miss_table(df: pd.DataFrame, forecast_col: str = "Raw_Forecast_MWH", stage: str | None = None) -> pd.DataFrame:
    if df is None or df.empty or forecast_col not in df.columns:
        return pd.DataFrame()
    rows = []
    for date, g in df.groupby("Date", dropna=False):
        g = g.dropna(subset=["Actual_MWH", forecast_col])
        if g.empty:
            continue
        actual_peak_idx = g["Actual_MWH"].idxmax()
        forecast_peak_idx = g[forecast_col].idxmax()
        actual_peak = g.loc[actual_peak_idx]
        forecast_peak = g.loc[forecast_peak_idx]
        timing_error = (pd.to_datetime(forecast_peak["DT"]) - pd.to_datetime(actual_peak["DT"])).total_seconds() / 3600.0
        rows.append({
            "Stage": stage or RAW_STAGE,
            "ForecastColumn": forecast_col,
            "Date": date,
            "Season": actual_peak.get("Season"),
            "Actual_Peak_DT": actual_peak["DT"],
            "Actual_Peak_Hour": int(actual_peak["Hour"]),
            "Actual_Peak_MWH": float(actual_peak["Actual_MWH"]),
            "Forecast_At_Actual_Peak_MWH": float(actual_peak[forecast_col]),
            "Underforecast_At_Actual_Peak_MWH": float(actual_peak["Actual_MWH"] - actual_peak[forecast_col]),
            "Forecast_Peak_DT": forecast_peak["DT"],
            "Forecast_Peak_Hour": int(forecast_peak["Hour"]),
            "Forecast_Peak_MWH": float(forecast_peak[forecast_col]),
            "Daily_Peak_Timing_Error_Hours": float(timing_error),
            "Daily_Energy_Actual_MWH": float(g["Actual_MWH"].sum()),
            "Daily_Energy_Forecast_MWH": float(g[forecast_col].sum()),
            "Daily_Energy_Error_MWH": float(g[forecast_col].sum() - g["Actual_MWH"].sum()),
            "Daily_MAE_MWH": float((g["Actual_MWH"] - g[forecast_col]).abs().mean()),
            "Daily_MAPE_PCT": _safe_mape(g["Actual_MWH"], g[forecast_col]),
            "DailyMaxTempBucket": actual_peak.get("DailyMaxTempBucket"),
            "Temperature_DailyMax": float(actual_peak["Temperature_DailyMax"]) if "Temperature_DailyMax" in actual_peak and pd.notna(actual_peak["Temperature_DailyMax"]) else np.nan,
            "CloudCoverBucket": actual_peak.get("CloudCoverBucket"),
            "BTMSolarBucket": actual_peak.get("BTMSolarBucket"),
            "SolarLossBucket": actual_peak.get("SolarLossBucket"),
        })
    out = pd.DataFrame(rows)
    if not out.empty:
        out.sort_values(["Underforecast_At_Actual_Peak_MWH", "Daily_MAE_MWH"], ascending=[False, False], inplace=True)
    return out.reset_index(drop=True)


def build_daily_peak_miss_by_stage(df: pd.DataFrame) -> pd.DataFrame:
    frames = []
    for stage, col in _available_stage_columns(df).items():
        if stage.startswith("baseline_") or stage in {"xgb_component", "lgb_component"}:
            continue
        tab = build_daily_peak_miss_table(df, forecast_col=col, stage=stage)
        if not tab.empty:
            frames.append(tab)
    return pd.concat(frames, ignore_index=True, sort=False) if frames else pd.DataFrame()


def build_top_error_tables(df: pd.DataFrame, n: int = 100, forecast_col: str = "Raw_Forecast_MWH", stage: str | None = None) -> tuple[pd.DataFrame, pd.DataFrame]:
    if df is None or df.empty or forecast_col not in df.columns:
        return pd.DataFrame(), pd.DataFrame()
    work = df.copy()
    work["Stage"] = stage or RAW_STAGE
    work["Stage_Forecast_MWH"] = _as_num(work[forecast_col])
    work["Stage_Residual_MWH"] = _as_num(work["Actual_MWH"]) - work["Stage_Forecast_MWH"]
    work["Stage_AbsError_MWH"] = work["Stage_Residual_MWH"].abs()
    work["Stage_APE"] = np.where(_as_num(work["Actual_MWH"]).abs() > 1e-9, work["Stage_AbsError_MWH"] / _as_num(work["Actual_MWH"]).abs() * 100.0, np.nan)
    keep = [
        "Stage", "ForecastColumn", "DT", "Date", "Forecast_Lead_Hour", "Forecast_Day", "Season", "Month", "Hour", "HourGroup",
        "Actual_MWH", "Stage_Forecast_MWH", "Raw_Forecast_MWH", "Residual_Calibrated_Forecast_MWH", "Warm_Ramp_Adjusted_Forecast_MWH", "Cloud_Solar_Adjusted_Forecast_MWH", "Peak_Risk_Adjusted_Forecast_MWH", "Recent_Corrected_Forecast_MWH", "XGB_Pred_MWH", "LGB_Pred_MWH", "CatBoost_Pred_MWH", "Prophet_Pred_MWH",
        "Stage_Residual_MWH", "Stage_AbsError_MWH", "Stage_APE", "Temperature", "Temperature_DailyMax", "DailyMaxTempBucket",
        "CloudCover_Norm", "CloudCoverBucket", "BTM_Solar_Proxy_MW", "BTMSolarBucket", "SolarLossBucket",
        "BTM_Solar_Loss_From_ClearSky_MW", "Cloud_Solar_Shape_Cal_MWH", "Cloud_Solar_Shape_Raw_Cal_MWH", "CloudSolarEventClass", "CloudSolarEventMultiplier", "CloudSolarBaseBucket", "Humidity_Norm", "WindSpeed_Mph", "PrecipIn", "IsWeekend", "IsHoliday",
    ]
    work["ForecastColumn"] = forecast_col
    keep = [c for c in keep if c in work.columns]
    under = work[work["Stage_Residual_MWH"] > 0].sort_values("Stage_Residual_MWH", ascending=False)[keep].head(int(n)).copy()
    over = work[work["Stage_Residual_MWH"] < 0].assign(Stage_Overforecast_MWH=lambda x: -x["Stage_Residual_MWH"]).sort_values("Stage_Overforecast_MWH", ascending=False)
    keep_over = keep + (["Stage_Overforecast_MWH"] if "Stage_Overforecast_MWH" in over.columns else [])
    over = over[[c for c in keep_over if c in over.columns]].head(int(n)).copy()
    return under.reset_index(drop=True), over.reset_index(drop=True)


def build_top_error_tables_by_stage(df: pd.DataFrame, n: int = 100) -> tuple[pd.DataFrame, pd.DataFrame]:
    under_frames = []
    over_frames = []
    for stage, col in _available_stage_columns(df).items():
        if stage in {"xgb_component", "lgb_component"} or stage.startswith("baseline_"):
            continue
        u, o = build_top_error_tables(df, n=n, forecast_col=col, stage=stage)
        if not u.empty:
            under_frames.append(u)
        if not o.empty:
            over_frames.append(o)
    return (
        pd.concat(under_frames, ignore_index=True, sort=False) if under_frames else pd.DataFrame(),
        pd.concat(over_frames, ignore_index=True, sort=False) if over_frames else pd.DataFrame(),
    )


def _flatten_lookup_table(bundle: dict | None, source_name: str) -> pd.DataFrame:
    if not bundle:
        return pd.DataFrame()
    frames = []
    for level in bundle.get("ordered_levels", []):
        lookup = level.get("lookup")
        if isinstance(lookup, pd.DataFrame) and not lookup.empty:
            tmp = lookup.copy()
            tmp.insert(0, "LookupSource", source_name)
            tmp.insert(1, "CalibrationLevel", level.get("name", "+".join(level.get("keys", []))))
            tmp.insert(2, "Keys", "+".join(level.get("keys", [])))
            frames.append(tmp)
    if not frames:
        return pd.DataFrame()
    return pd.concat(frames, ignore_index=True, sort=False)


def build_calibration_debug_table(calibration_lookup_bundle: dict | None, heat_peak_lookup: pd.DataFrame | None, warm_ramp_lookup: dict | None = None) -> pd.DataFrame:
    frames = [_flatten_lookup_table(calibration_lookup_bundle, "residual_calibration")]
    if heat_peak_lookup is not None and isinstance(heat_peak_lookup, pd.DataFrame) and not heat_peak_lookup.empty:
        h = heat_peak_lookup.copy()
        h.insert(0, "LookupSource", "heat_peak_calibration")
        h.insert(1, "CalibrationLevel", "heat_peak_hour_maxtemp")
        h.insert(2, "Keys", "Hour+DailyMaxTempBin")
        frames.append(h)
    if warm_ramp_lookup:
        frames.append(_flatten_lookup_table(warm_ramp_lookup, "warm_ramp_calibration"))
    frames = [f for f in frames if f is not None and not f.empty]
    return pd.concat(frames, ignore_index=True, sort=False) if frames else pd.DataFrame()


def build_band_debug_table(residual_band_lookup: dict | None) -> pd.DataFrame:
    if not residual_band_lookup:
        return pd.DataFrame()
    frames = []
    for i, level in enumerate(residual_band_lookup.get("ordered_levels", []), start=1):
        lookup = level.get("lookup")
        if isinstance(lookup, pd.DataFrame) and not lookup.empty:
            tmp = lookup.copy()
            tmp.insert(0, "BandLevel", i)
            tmp.insert(1, "Keys", "+".join(level.get("keys", [])))
            tmp["Global_Band_MWH"] = residual_band_lookup.get("global_band_mwh", np.nan)
            frames.append(tmp)
    return pd.concat(frames, ignore_index=True, sort=False) if frames else pd.DataFrame()


def feature_importance_table(model: Any, features: list[str], model_name: str) -> pd.DataFrame:
    if model is None or not features:
        return pd.DataFrame()
    values = None
    try:
        values = np.asarray(getattr(model, "feature_importances_"), dtype=float)
    except Exception:
        values = None
    if values is None or values.size == 0:
        return pd.DataFrame()
    n = min(len(features), len(values))
    out = pd.DataFrame({
        "Model": model_name,
        "Feature": list(features)[:n],
        "Importance": values[:n],
    })
    total = out["Importance"].sum()
    out["Importance_Pct"] = out["Importance"] / total * 100.0 if total > 0 else np.nan
    return out.sort_values("Importance", ascending=False).reset_index(drop=True)


def prophet_regressor_table(prophet_model: Any | None, prophet_features: list[str] | None) -> pd.DataFrame:
    if prophet_model is None or not prophet_features:
        return pd.DataFrame()
    extra = getattr(prophet_model, "extra_regressors", {}) or {}
    rows = []
    for feature in prophet_features:
        meta = extra.get(feature, {}) if isinstance(extra, dict) else {}
        rows.append({
            "Model": "prophet",
            "Feature": feature,
            "PriorScale": meta.get("prior_scale") if isinstance(meta, dict) else np.nan,
            "Standardize": meta.get("standardize") if isinstance(meta, dict) else np.nan,
            "Mode": meta.get("mode") if isinstance(meta, dict) else np.nan,
        })
    return pd.DataFrame(rows)


def _component_metric_rows(df: pd.DataFrame, component_cols: dict[str, str]) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    if df is None or df.empty or "Actual_MWH" not in df.columns:
        return rows
    actual = _as_num(df["Actual_MWH"])
    for name, col in component_cols.items():
        if col not in df.columns:
            continue
        pred = _as_num(df[col])
        row = _metric_dict(actual, pred, label=name, col=col)
        if row:
            row["Model"] = row.pop("Stage")
            rows.append(row)
    return rows


def build_forecast_stage_metrics(df: pd.DataFrame) -> pd.DataFrame:
    """Compare raw model, benchmark models, baselines, and V12 corrected stages."""
    rows = _component_metric_rows(df, _available_stage_columns(df))
    if not rows:
        return pd.DataFrame()
    out = pd.DataFrame(rows)
    raw_mae = out.loc[out["Model"].eq(RAW_STAGE), "MAE_MWH"]
    raw_bias = out.loc[out["Model"].eq(RAW_STAGE), "Bias_MWH"]
    final_mae = out.loc[out["Model"].eq(FINAL_STAGE), "MAE_MWH"]
    if not raw_mae.empty:
        out["MAE_Improvement_vs_Raw_MWH"] = float(raw_mae.iloc[0]) - out["MAE_MWH"]
        out["Skill_vs_Raw_PCT"] = np.where(float(raw_mae.iloc[0]) > 0, (float(raw_mae.iloc[0]) - out["MAE_MWH"]) / float(raw_mae.iloc[0]) * 100.0, np.nan)
    if not raw_bias.empty:
        out["Bias_Abs_Improvement_vs_Raw_MWH"] = abs(float(raw_bias.iloc[0])) - out["Bias_MWH"].abs()
    if not final_mae.empty:
        out["MAE_Delta_vs_Final_MWH"] = out["MAE_MWH"] - float(final_mae.iloc[0])
    return out.sort_values("MAE_MWH").reset_index(drop=True)


def build_model_component_metrics(df: pd.DataFrame) -> pd.DataFrame:
    rows = _component_metric_rows(df, _available_stage_columns(df))
    return pd.DataFrame(rows).sort_values("MAE_MWH").reset_index(drop=True) if rows else pd.DataFrame()


# Ordered correction chain as applied by the pipeline. Used to attribute the MARGINAL
# (incremental) value of each stage rather than only its standalone metric.
_STAGE_CHAIN_ORDER = [
    ("Raw_Forecast_MWH", "Raw XGB+LGB"),
    ("Residual_Calibrated_Forecast_MWH", "+Residual"),
    ("Targeted_Meta_Adjusted_Forecast_MWH", "+TargetedMeta"),
    ("Warm_Ramp_Adjusted_Forecast_MWH", "+WarmRamp"),
    ("Cloud_Solar_Adjusted_Forecast_MWH", "+CloudSolar"),
    ("Peak_Risk_Adjusted_Forecast_MWH", "+PeakRisk"),
    ("Recent_Corrected_Forecast_MWH", "+Recent"),
    ("Stage_Selected_Forecast_MWH", "+StageSelector"),
    ("Final_Backtest_Forecast_MWH", "Final"),
]


def build_stage_marginal_contributions(df: pd.DataFrame, slices: dict[str, pd.DataFrame] | None = None) -> pd.DataFrame:
    """Attribute the marginal (incremental) MAE/bias change of each correction stage,
    overall and per slice. A positive Marginal_dMAE means the stage *worsened* MAE on
    that slice -- a signal that a heavily-tuned layer may not earn its place and should
    be re-examined or gated off on the production-weather scorecard.

    Note: this attributes the *realized-weather* chain, since the intermediate stage
    columns are only recomputed on realized weather; the production-weather degradation
    is captured separately by the weather-realism scorecard's Final column.
    """
    frames = {"Overall": df}
    if slices:
        frames.update(slices)

    rows = []
    for slice_name, frame in frames.items():
        if frame is None or frame.empty or "Actual_MWH" not in frame.columns:
            continue
        actual = _as_num(frame["Actual_MWH"])
        present = [(col, lbl) for col, lbl in _STAGE_CHAIN_ORDER if col in frame.columns and _as_num(frame[col]).notna().any()]
        prev_mae = None
        for col, lbl in present:
            resid = actual - _as_num(frame[col])
            mask = resid.notna()
            if not mask.any():
                continue
            mae = float(resid[mask].abs().mean())
            bias = float(resid[mask].mean())
            marginal = np.nan if prev_mae is None else mae - prev_mae
            rows.append({
                "Slice": slice_name,
                "Stage": lbl,
                "Column": col,
                "N": int(mask.sum()),
                "MAE_MWH": mae,
                "Bias_MWH": bias,
                "Marginal_dMAE_MWH": marginal,
                "Worsens_Slice": bool(np.isfinite(marginal) and marginal > 0.0),
            })
            prev_mae = mae
    return pd.DataFrame(rows)


def _scorecard_metric_row(
    df: pd.DataFrame,
    test: str,
    purpose: str,
    basis: str,
    target: str,
    gate: str,
    forecast_col: str = "Final_Backtest_Forecast_MWH",
) -> dict[str, Any]:
    if df is None or df.empty or forecast_col not in df.columns or "Actual_MWH" not in df.columns:
        return {
            "Test": test,
            "Purpose": purpose,
            "Basis": basis,
            "N": 0,
            "Target": target,
            "Gate": gate,
            "Pass": False,
        }
    metrics = _metric_dict(df["Actual_MWH"], df[forecast_col], col=forecast_col)
    row = {
        "Test": test,
        "Purpose": purpose,
        "Basis": basis,
        "ForecastColumn": forecast_col,
        "Target": target,
        "Gate": gate,
    }
    row.update({k: v for k, v in metrics.items() if k not in {"Stage", "ForecastColumn"}})

    mae = float(row.get("MAE_MWH", np.nan))
    mape = float(row.get("MAPE_PCT", np.nan))
    bias = float(row.get("Bias_MWH", np.nan))
    passes = True
    if "mae<=" in gate:
        limit = float(gate.split("mae<=", 1)[1].split(";")[0])
        passes = passes and np.isfinite(mae) and mae <= limit
    if "mape<=" in gate:
        limit = float(gate.split("mape<=", 1)[1].split(";")[0])
        passes = passes and np.isfinite(mape) and mape <= limit
    if "abs_bias<=" in gate:
        limit = float(gate.split("abs_bias<=", 1)[1].split(";")[0])
        passes = passes and np.isfinite(bias) and abs(bias) <= limit
    row["Pass"] = bool(passes)
    return row


def build_production_readiness_scorecard(recent_df: pd.DataFrame, replay_df: pd.DataFrame) -> pd.DataFrame:
    """Official production scorecard: rolling-origin replay is primary, recent backtest is context."""
    recent_col = "Final_Backtest_Forecast_MWH"
    replay_col = "Final_Backtest_Forecast_MWH"
    rows: list[dict[str, Any]] = []
    rows.append(_scorecard_metric_row(recent_df, "Last 45 days", "Recent behavior", "recent_45_day_backtest", "Recent MAE <= 3.0 MWh", "mae<=3.0", recent_col))
    rows.append(_scorecard_metric_row(replay_df, "Seasonal rolling origins", "General robustness", "seasonal_rolling_origin_replay", "MAE <= 4.5, MAPE <= 3.5%, abs bias <= 0.75", "mae<=4.5;mape<=3.5;abs_bias<=0.75", replay_col))

    if replay_df is None or replay_df.empty:
        return pd.DataFrame(rows)
    day = _as_num(replay_df.get("Forecast_Day", pd.Series(np.nan, index=replay_df.index)))
    hour = _as_num(replay_df.get("Hour", pd.Series(np.nan, index=replay_df.index)))
    temp = _as_num(replay_df.get("Temperature_DailyMax", pd.Series(np.nan, index=replay_df.index)))
    cloud = _as_num(replay_df.get("CloudCover_Norm", pd.Series(np.nan, index=replay_df.index)))
    loss = _as_num(replay_df.get("BTM_Solar_Loss_From_ClearSky_MW", pd.Series(np.nan, index=replay_df.index)))
    season = replay_df.get("Season", pd.Series("", index=replay_df.index)).astype(str)
    rows.extend(
        [
            _scorecard_metric_row(replay_df[day.eq(1)], "Day 1 only", "Near-term operational forecast", "seasonal_rolling_origin_replay", "MAE <= 3.5 MWh", "mae<=3.5", replay_col),
            _scorecard_metric_row(replay_df[day.between(2, 3)], "Days 2-3", "Short weather forecast horizon", "seasonal_rolling_origin_replay", "MAE <= 5.0 MWh", "mae<=5.0", replay_col),
            _scorecard_metric_row(replay_df[day.between(4, 7)], "Days 4-7", "Weather uncertainty horizon", "seasonal_rolling_origin_replay", "MAE <= 5.0 MWh", "mae<=5.0", replay_col),
            _scorecard_metric_row(replay_df[hour.between(16, 20) & temp.ge(90.0)], "Hot peak days", "Operational risk", "seasonal_rolling_origin_replay", "MAE <= 6.0 MWh", "mae<=6.0", replay_col),
            _scorecard_metric_row(replay_df[hour.between(10, 16) & (cloud.ge(0.60) | loss.ge(1.25))], "Cloud/solar midday", "BTM solar/cloud risk", "seasonal_rolling_origin_replay", "MAE <= 7.0 MWh", "mae<=7.0", replay_col),
            _scorecard_metric_row(replay_df[season.isin(["Spring", "Fall"]) & hour.between(12, 22) & temp.between(75.0, 93.0)], "Shoulder heat transition", "Spring/fall load-response risk", "seasonal_rolling_origin_replay", "MAE <= 7.0 MWh", "mae<=7.0", replay_col),
            _scorecard_metric_row(replay_df[hour.between(14, 18)], "Peak window hours 14-18", "Peak planning risk", "seasonal_rolling_origin_replay", "MAE <= 5.5 MWh", "mae<=5.5", replay_col),
        ]
    )
    return pd.DataFrame(rows)


def build_recent_profile_debug_table(profile: dict | None) -> pd.DataFrame:
    if not profile:
        return pd.DataFrame()
    rows = []
    scalar_keys = ["recent_mean", "last24_mean", "global_mean"]
    lookup_keys = [
        "same_hour_mean", "hourgroup_mean", "temp_hourgroup_mean", "cloud_hourgroup_mean",
        "solar_hourgroup_mean", "solar_loss_hourgroup_mean", "temp_cloud_hourgroup_mean",
    ]
    for key in scalar_keys:
        if key in profile:
            rows.append({"Level": key, "Key": "all", "Correction_MWH": profile.get(key)})
    for level in lookup_keys:
        for k, v in (profile.get(level, {}) or {}).items():
            rows.append({"Level": level, "Key": k, "Correction_MWH": v})
    if profile.get("metadata"):
        for k, v in profile.get("metadata", {}).items():
            rows.append({"Level": "metadata", "Key": k, "Correction_MWH": v})
    return pd.DataFrame(rows)


def _diagnostic_band_for_row(
    row: pd.Series,
    forecast: float,
    residual_band_lookup: dict | None,
    percent_band: float,
    floor_mwh: float,
    band_scale: float = 1.0,
    hot_bucket_band_floor: dict | None = None,
) -> tuple[float, str]:
    band = max(float(floor_mwh), abs(float(forecast)) * float(percent_band)) if np.isfinite(forecast) else float(floor_mwh)
    method = "percent_or_floor"
    if residual_band_lookup and residual_band_lookup.get("ordered_levels"):
        band = max(band, float(residual_band_lookup.get("global_band_mwh", floor_mwh)))
        for level in residual_band_lookup.get("ordered_levels", []):
            keys = level.get("keys", [])
            lookup = level.get("lookup")
            if lookup is None or not isinstance(lookup, pd.DataFrame) or lookup.empty or not all(k in row.index for k in keys):
                continue
            mask = pd.Series(True, index=lookup.index)
            for k in keys:
                mask &= lookup[k].astype("object").fillna("__NA__").eq(row.get(k) if pd.notna(row.get(k)) else "__NA__")
            m = lookup.loc[mask]
            if not m.empty and "band_mwh" in m.columns:
                try:
                    band = max(band, float(m.iloc[0]["band_mwh"]))
                    method = "+".join(keys)
                    break
                except Exception:
                    pass
    scale = max(0.10, float(band_scale or 1.0))
    band_inputs = _prep_band_inputs(row.to_frame().T)
    mult = float(_band_risk_multiplier(band_inputs).iloc[0])
    band_value = band * scale * mult
    hot_floor = _hot_bucket_band_floor(band_inputs, hot_bucket_band_floor)
    if hot_floor.notna().any() and float(hot_floor.iloc[0]) > band_value:
        band_value = float(hot_floor.iloc[0])
        method = f"{method}+hot_bucket_floor"
    return band_value, method


def build_band_coverage_by_stage(df: pd.DataFrame, residual_band_lookup: dict | None, config: dict | None = None) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    if df is None or df.empty or "Actual_MWH" not in df.columns:
        return pd.DataFrame(), pd.DataFrame(), pd.DataFrame(), pd.DataFrame()
    bands_cfg = (config or {}).get("bands", {}) if isinstance(config, dict) else {}
    percent_band = float(bands_cfg.get("default_percent_band", 0.08))
    floor_mwh = float(bands_cfg.get("band_floor_mwh", 5.0))
    band_scale = float(bands_cfg.get("band_scale", 1.0))
    stages = {k: v for k, v in _available_stage_columns(df).items() if k in [RAW_STAGE, "cloud_solar_adjusted", "peak_risk_adjusted", "recent_corrected_simulation", FINAL_STAGE, "prophet_benchmark"]}
    rows = []
    for stage, col in stages.items():
        for idx, row in df.iterrows():
            actual = float(row["Actual_MWH"]) if pd.notna(row.get("Actual_MWH")) else np.nan
            forecast = float(row[col]) if col in row.index and pd.notna(row.get(col)) else np.nan
            if not np.isfinite(actual) or not np.isfinite(forecast):
                continue
            band, method = _diagnostic_band_for_row(
                row,
                forecast,
                residual_band_lookup,
                percent_band,
                floor_mwh,
                band_scale,
                bands_cfg.get("hot_bucket_band_floor", {}),
            )
            residual = actual - forecast
            rows.append({
                "Stage": stage,
                "ForecastColumn": col,
                "DT": row.get("DT"),
                "Hour": row.get("Hour"),
                "HourGroup": row.get("HourGroup"),
                "Forecast_Day": row.get("Forecast_Day"),
                "DailyMaxTempBucket": row.get("DailyMaxTempBucket"),
                "CloudCoverBucket": row.get("CloudCoverBucket"),
                "BTMSolarBucket": row.get("BTMSolarBucket"),
                "SolarLossBucket": row.get("SolarLossBucket"),
                "Actual_MWH": actual,
                "Forecast_MWH": forecast,
                "Residual_MWH": residual,
                "AbsError_MWH": abs(residual),
                "Band_MWH": band,
                "Band_Method": method,
                "Inside_Band": int(abs(residual) <= band),
                "Band_Miss_MWH": max(0.0, abs(residual) - band),
            })
    detail = pd.DataFrame(rows)
    if detail.empty:
        return pd.DataFrame(), pd.DataFrame(), pd.DataFrame(), pd.DataFrame()

    def summarize(g: pd.DataFrame) -> pd.Series:
        return pd.Series({
            "N": int(len(g)),
            "Coverage_PCT": float(g["Inside_Band"].mean() * 100.0),
            "Avg_Band_MWH": float(g["Band_MWH"].mean()),
            "Avg_AbsError_MWH": float(g["AbsError_MWH"].mean()),
            "P90_AbsError_MWH": float(g["AbsError_MWH"].quantile(0.90)),
            "Avg_Band_Miss_MWH": float(g["Band_Miss_MWH"].mean()),
            "Max_Band_Miss_MWH": float(g["Band_Miss_MWH"].max()),
        })

    summary = detail.groupby(["Stage", "ForecastColumn"], dropna=False).apply(summarize, include_groups=False).reset_index()
    by_hour = detail.groupby(["Stage", "Hour"], dropna=False).apply(summarize, include_groups=False).reset_index()
    by_temp = detail.groupby(["Stage", "DailyMaxTempBucket", "HourGroup"], dropna=False).apply(summarize, include_groups=False).reset_index()
    return summary, by_hour, by_temp, detail


def build_diagnostics_bundle(
    backtest_df: pd.DataFrame,
    forecast_display_df: pd.DataFrame | None = None,
    features: list[str] | None = None,
    xgb_model: Any | None = None,
    lgb_model: Any | None = None,
    prophet_model: Any | None = None,
    prophet_features: list[str] | None = None,
    catboost_model: Any | None = None,
    calibration_lookup_bundle: dict | None = None,
    heat_peak_lookup: pd.DataFrame | None = None,
    warm_ramp_lookup: dict | None = None,
    cloud_solar_shape_lookup: dict | None = None,
    recent_residual_profile: dict | None = None,
    residual_band_lookup: dict | None = None,
    config: dict | None = None,
) -> dict[str, Any]:
    """
    Build diagnostics tables that explain where the forecast is missing and which stage fixed it.

    Residual convention:
      Residual_MWH = Actual_MWH - Forecast_MWH
      Positive residual = underforecast.
      Negative residual = overforecast.
    """
    diag_cfg = (config or {}).get("diagnostics", {}) if isinstance(config, dict) else {}
    top_n = int(diag_cfg.get("top_n", 100))
    min_segment_count = int(diag_cfg.get("min_segment_count", 6))

    bt = prep_backtest(backtest_df)
    under, over = build_top_error_tables(bt, n=top_n, forecast_col="Raw_Forecast_MWH", stage=RAW_STAGE)
    under_stage, over_stage = build_top_error_tables_by_stage(bt, n=top_n)

    xgb_imp = feature_importance_table(xgb_model, features or [], "xgb")
    lgb_imp = feature_importance_table(lgb_model, features or [], "lgb")
    cat_imp = feature_importance_table(catboost_model, features or [], "catboost")
    feature_importance = pd.concat([xgb_imp, lgb_imp, cat_imp], ignore_index=True, sort=False) if not xgb_imp.empty or not lgb_imp.empty or not cat_imp.empty else pd.DataFrame()
    prophet_regs = prophet_regressor_table(prophet_model, prophet_features or [])
    try:
        from forecasting.forecast.event_shape_corrections import cloud_solar_lookup_debug_table
        cloud_solar_debug = cloud_solar_lookup_debug_table(cloud_solar_shape_lookup)
    except Exception:
        cloud_solar_debug = pd.DataFrame()

    band_summary, band_by_hour, band_by_temp, band_detail = build_band_coverage_by_stage(bt, residual_band_lookup, config=config)

    bundle: dict[str, Any] = {
        "diagnostics_summary": metrics_summary(bt),
        "backtest_enriched": bt,
        # Legacy/raw tables preserved for compatibility.
        "backtest_metrics_by_segment": build_backtest_metrics_by_segment(bt, min_count=min_segment_count, forecast_col="Raw_Forecast_MWH"),
        "error_by_hour": build_metrics_by_group(bt, ["Hour"], min_count=1, forecast_col="Raw_Forecast_MWH"),
        "seasonal_error_by_hour": build_metrics_by_group(bt, ["Season", "Hour"], min_count=min_segment_count, forecast_col="Raw_Forecast_MWH"),
        "seasonal_error_by_month_hour": build_metrics_by_group(bt, ["Season", "Month", "Hour"], min_count=min_segment_count, forecast_col="Raw_Forecast_MWH"),
        "seasonal_error_by_max_temp_bin": build_metrics_by_group(bt, ["Season", "DailyMaxTempBucket", "HourGroup"], min_count=min_segment_count, forecast_col="Raw_Forecast_MWH"),
        "seasonal_error_by_cloud_bin": build_metrics_by_group(bt, ["Season", "CloudCoverBucket", "HourGroup"], min_count=min_segment_count, forecast_col="Raw_Forecast_MWH"),
        "seasonal_error_by_solar_bin": build_metrics_by_group(bt, ["Season", "BTMSolarBucket", "HourGroup"], min_count=min_segment_count, forecast_col="Raw_Forecast_MWH"),
        "error_by_temp_cloud_hourgroup": build_metrics_by_group(bt, ["DailyMaxTempBucket", "CloudCoverBucket", "HourGroup"], min_count=min_segment_count, forecast_col="Raw_Forecast_MWH"),
        "daily_peak_miss_table": build_daily_peak_miss_table(bt, forecast_col="Raw_Forecast_MWH", stage=RAW_STAGE),
        "top_100_underforecast_hours": under,
        "top_100_overforecast_hours": over,
        # V12.4 stage-aware diagnostics.
        "model_component_metrics": build_model_component_metrics(bt),
        "forecast_stage_metrics": build_forecast_stage_metrics(bt),
        "backtest_metrics_by_segment_by_stage": build_backtest_metrics_by_segment_by_stage(bt, min_count=min_segment_count),
        "error_by_hour_by_stage": build_metrics_by_group_by_stage(bt, ["Hour"], min_count=1),
        "error_by_forecast_lead_hour_by_stage": build_metrics_by_group_by_stage(bt, ["Forecast_Lead_Hour"], min_count=1),
        "error_by_forecast_day_by_stage": build_metrics_by_group_by_stage(bt, ["Forecast_Day"], min_count=1),
        "seasonal_error_by_max_temp_bin_by_stage": build_metrics_by_group_by_stage(bt, ["Season", "DailyMaxTempBucket", "HourGroup"], min_count=min_segment_count),
        "seasonal_error_by_cloud_bin_by_stage": build_metrics_by_group_by_stage(bt, ["Season", "CloudCoverBucket", "HourGroup"], min_count=min_segment_count),
        "seasonal_error_by_solar_loss_bin_by_stage": build_metrics_by_group_by_stage(bt, ["Season", "SolarLossBucket", "HourGroup"], min_count=min_segment_count),
        "cloud_solar_event_error_by_stage": build_metrics_by_group_by_stage(bt, ["CloudSolarEventClass", "HourGroup"], min_count=min_segment_count),
        "cloud_solar_event_hour_error_by_stage": build_metrics_by_group_by_stage(bt, ["CloudSolarEventClass", "Hour"], min_count=min_segment_count),
        "daily_peak_miss_by_stage": build_daily_peak_miss_by_stage(bt),
        "stage_marginal_contributions": build_stage_marginal_contributions(bt),
        "top_100_underforecast_hours_by_stage": under_stage,
        "top_100_overforecast_hours_by_stage": over_stage,
        "band_coverage_summary": band_summary,
        "band_coverage_by_hour": band_by_hour,
        "band_coverage_by_temp_bucket": band_by_temp,
        "band_coverage_detail": band_detail,
        "recent_residual_profile_debug": build_recent_profile_debug_table(recent_residual_profile),
        "calibration_lookup_debug": build_calibration_debug_table(calibration_lookup_bundle, heat_peak_lookup, warm_ramp_lookup),
        "cloud_solar_shape_lookup_debug": cloud_solar_debug,
        "band_lookup_debug": build_band_debug_table(residual_band_lookup),
        "feature_importance": feature_importance,
        "prophet_regressors": prophet_regs,
        "forecast_display_for_review": forecast_display_df.copy() if isinstance(forecast_display_df, pd.DataFrame) else pd.DataFrame(),
    }
    return bundle


def _json_clean(obj: Any) -> Any:
    if isinstance(obj, dict):
        return {str(k): _json_clean(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [_json_clean(v) for v in obj]
    if isinstance(obj, (np.integer,)):
        return int(obj)
    if isinstance(obj, (np.floating,)):
        return None if not np.isfinite(obj) else float(obj)
    if isinstance(obj, float):
        return None if not math.isfinite(obj) else obj
    try:
        if pd.isna(obj) and not isinstance(obj, (pd.DataFrame, pd.Series)):
            return None
    except Exception:
        pass
    return obj


def export_diagnostics_bundle(bundle: dict[str, Any], output_dir: str | Path) -> dict[str, str]:
    """Write diagnostics tables to CSV and summary metrics to JSON."""
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    written: dict[str, str] = {}

    for name, value in (bundle or {}).items():
        if isinstance(value, pd.DataFrame):
            file_path = output_path / f"{name}.csv"
            if value.empty:
                if file_path.exists():
                    file_path.unlink()
                continue
            value.to_csv(file_path, index=False)
            written[name] = str(file_path)
        elif isinstance(value, dict):
            file_path = output_path / f"{name}.json"
            file_path.write_text(json.dumps(_json_clean(value), indent=2), encoding="utf-8")
            written[name] = str(file_path)

    manifest = output_path / "diagnostics_manifest.json"
    manifest.write_text(json.dumps(written, indent=2), encoding="utf-8")
    written["diagnostics_manifest"] = str(manifest)
    return written
