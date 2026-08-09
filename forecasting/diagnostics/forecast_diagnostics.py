from __future__ import annotations

import json
import math
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from forecasting.forecast.anomaly_exclusions import drop_excluded_intervals
from forecasting.forecast.focused_scorecard_guard import (
    build_focused_scorecard_rule_audit,
)
from forecasting.forecast.uncertainty_bands import (
    _band_risk_multiplier,
    _hot_bucket_band_floor,
    _prep as _prep_band_inputs,
)

RAW_STAGE = "raw_xgb_lgb_production"
FINAL_STAGE = "final_corrected_production"
HE18_20_CODE_HOURS = [17, 18, 19]
PEAK_WINDOW_14_18_HOURS = [14, 15, 16, 17, 18]
PEAK_WINDOW_14_20_HOURS = [14, 15, 16, 17, 18, 19, 20]
EXTREME_HEAT_MIN_DAILY_MAX_F = 105.0


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
                labels=[
                    "Clear/Low",
                    "Some Clouds",
                    "Partly Cloudy",
                    "Mostly Cloudy",
                    "Overcast",
                ],
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
        loss_col = (
            "BTM_Solar_Loss_From_ClearSky_MW"
            if "BTM_Solar_Loss_From_ClearSky_MW" in out.columns
            else "Midday_Overcast_Solar_Loss_MW"
        )
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
    out["Baseline_Rolling7DaySameHourAvg_MWH"] = out.groupby("Hour", dropna=False)[
        "Actual_MWH"
    ].transform(lambda s: _as_num(s).shift(1).rolling(window=7, min_periods=2).mean())
    return out


def _local_datetime_series(values) -> pd.Series:
    raw = values if isinstance(values, pd.Series) else pd.Series(values)
    try:
        return pd.to_datetime(raw, errors="coerce")
    except ValueError:
        # Exported CSVs can contain both standard-time and daylight-time offsets.
        # Diagnostics bucket by local clock, so strip offsets instead of converting to UTC.
        cleaned = (
            raw.astype(str)
            .str.strip()
            .str.replace(r"(?:[+-]\d{2}:?\d{2}|Z)$", "", regex=True)
        )
        return pd.to_datetime(cleaned, errors="coerce")


def prep_backtest(backtest_df: pd.DataFrame) -> pd.DataFrame:
    """Normalize the backtest frame into a diagnostics-friendly schema."""
    if backtest_df is None or backtest_df.empty:
        return pd.DataFrame()

    out = backtest_df.copy()
    out["DT"] = _local_datetime_series(out["DT"])
    out = out.dropna(subset=["DT"]).sort_values("DT").reset_index(drop=True)

    if "Actual_MWH" not in out.columns and "Actual" in out.columns:
        out["Actual_MWH"] = out["Actual"]
    if "Raw_Forecast_MWH" not in out.columns and "Forecast" in out.columns:
        out["Raw_Forecast_MWH"] = out["Forecast"]

    out["Actual_MWH"] = _as_num(out.get("Actual_MWH", np.nan))
    out["Raw_Forecast_MWH"] = _as_num(out.get("Raw_Forecast_MWH", np.nan))
    out["Residual_MWH"] = _as_num(
        out.get("Residual_MWH", out["Actual_MWH"] - out["Raw_Forecast_MWH"])
    )
    out["AbsError_MWH"] = _as_num(out.get("AbsError_MWH", out["Residual_MWH"].abs()))
    out["APE"] = _as_num(
        out.get(
            "APE",
            np.where(
                out["Actual_MWH"].abs() > 1e-9,
                out["AbsError_MWH"] / out["Actual_MWH"].abs() * 100.0,
                np.nan,
            ),
        )
    )

    out["Date"] = out["DT"].dt.date.astype(str)
    if "Forecast_Lead_Hour" in out.columns:
        default_lead = pd.Series(np.arange(1, len(out) + 1, dtype=int), index=out.index)
        out["Forecast_Lead_Hour"] = (
            _as_num(out["Forecast_Lead_Hour"]).fillna(default_lead).astype(int)
        )
    else:
        out["Forecast_Lead_Hour"] = np.arange(1, len(out) + 1, dtype=int)
    if "Forecast_Day" in out.columns:
        out["Forecast_Day"] = (
            _as_num(out["Forecast_Day"])
            .fillna(((out["Forecast_Lead_Hour"] - 1) // 24 + 1))
            .astype(int)
        )
    else:
        out["Forecast_Day"] = ((out["Forecast_Lead_Hour"] - 1) // 24 + 1).astype(int)
    out["Hour"] = (
        _as_num(out.get("Hour", out["DT"].dt.hour))
        .fillna(out["DT"].dt.hour)
        .astype(int)
    )
    out["Month"] = (
        _as_num(out.get("Month", out["DT"].dt.month))
        .fillna(out["DT"].dt.month)
        .astype(int)
    )
    out["DOW"] = (
        _as_num(out.get("DOW", out["DT"].dt.dayofweek))
        .fillna(out["DT"].dt.dayofweek)
        .astype(int)
    )
    out["Season"] = out.get("Season", out["Month"].map(_season_from_month))
    out["HourGroup"] = out.get("HourGroup", out["Hour"].map(_hour_group))
    out["IsWeekend"] = (
        _as_num(out.get("IsWeekend", out["DOW"].isin([5, 6]).astype(int)))
        .fillna(0)
        .astype(int)
    )
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
        "focused_shape_shadow": "Focused_Shape_Adjusted_Forecast_MWH",
        FINAL_STAGE: "Final_Backtest_Forecast_MWH",
        "auto_residual_shadow": "Auto_Residual_Adjusted_Forecast_MWH",
        "daily_peak_shadow": "Daily_Peak_Shadow_Adjusted_Forecast_MWH",
        "heat_analog_shadow": "Heat_Analog_Shadow_Forecast_MWH",
        "auto_residual_full_shadow": "Auto_Residual_Full_Shadow_Adjusted_Forecast_MWH",
        "auto_residual_structural_hot_peak_shadow": "Auto_Residual_Structural_HotPeak_Adjusted_Forecast_MWH",
        "auto_residual_broad_hot_peak_shadow": "Auto_Residual_Broad_HotPeak_Shadow_Adjusted_Forecast_MWH",
        "hot_ramp_peak_shadow": "Hot_Ramp_Peak_Shadow_Forecast_MWH",
        "heat_persistence_peak_shadow": "Heat_Persistence_Peak_Shadow_Forecast_MWH",
        "baseline_same_hour_yesterday": "Baseline_SameHourYesterday_MWH",
        "baseline_same_hour_7_days_ago": "Baseline_SameHour7DaysAgo_MWH",
        "baseline_rolling_7day_same_hour_avg": "Baseline_Rolling7DaySameHourAvg_MWH",
    }
    return {
        name: col
        for name, col in candidates.items()
        if col in df.columns and _as_num(df[col]).notna().any()
    }


def _metric_dict(
    actual: pd.Series,
    forecast: pd.Series,
    label: str | None = None,
    col: str | None = None,
) -> dict[str, Any]:
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


_SHADOW_STAGE_NAMES = {
    "focused_shape_shadow",
    "auto_residual_shadow",
    "auto_residual_full_shadow",
    "auto_residual_structural_hot_peak_shadow",
    "auto_residual_broad_hot_peak_shadow",
    "daily_peak_shadow",
    "hot_ramp_peak_shadow",
    "heat_persistence_peak_shadow",
    "heat_analog_shadow",
}


def _stage_leader_audit(df: pd.DataFrame, final_col: str | None) -> dict[str, Any]:
    if df is None or df.empty or "Actual_MWH" not in df.columns:
        return {}
    stages = _available_stage_columns(df)
    if not stages:
        return {}

    rows: list[dict[str, Any]] = []
    for stage, col in stages.items():
        if stage.startswith("baseline_"):
            continue
        metrics = _metric_dict(df["Actual_MWH"], df[col], label=stage, col=col)
        if metrics:
            rows.append(metrics)
    if not rows:
        return {}

    final_row = next(
        (
            r
            for r in rows
            if r.get("ForecastColumn") == final_col or r.get("Stage") == FINAL_STAGE
        ),
        None,
    )
    best = min(rows, key=lambda r: float(r.get("MAE_MWH", np.inf)))
    out: dict[str, Any] = {
        "best_stage_by_mae": {
            "Stage": best.get("Stage"),
            "ForecastColumn": best.get("ForecastColumn"),
            "N": best.get("N"),
            "MAE_MWH": best.get("MAE_MWH"),
            "Bias_MWH": best.get("Bias_MWH"),
            "MAPE_PCT": best.get("MAPE_PCT"),
        },
    }

    if final_row:
        final_mae = float(final_row.get("MAE_MWH", np.nan))
        best_mae = float(best.get("MAE_MWH", np.nan))
        improvement = final_mae - best_mae
        out["Best_Stage_MAE_Improvement_vs_Final_MWH"] = improvement
        out["Final_Is_Best_Stage_By_MAE"] = bool(
            np.isfinite(improvement) and improvement <= 1e-9
        )

    shadow_rows = [r for r in rows if r.get("Stage") in _SHADOW_STAGE_NAMES]
    if shadow_rows:
        best_shadow = min(shadow_rows, key=lambda r: float(r.get("MAE_MWH", np.inf)))
        out["best_shadow_stage_by_mae"] = {
            "Stage": best_shadow.get("Stage"),
            "ForecastColumn": best_shadow.get("ForecastColumn"),
            "N": best_shadow.get("N"),
            "MAE_MWH": best_shadow.get("MAE_MWH"),
            "Bias_MWH": best_shadow.get("Bias_MWH"),
            "MAPE_PCT": best_shadow.get("MAPE_PCT"),
        }
        if final_row:
            final_mae = float(final_row.get("MAE_MWH", np.nan))
            shadow_mae = float(best_shadow.get("MAE_MWH", np.nan))
            shadow_improvement = final_mae - shadow_mae
            out["Best_Shadow_MAE_Improvement_vs_Final_MWH"] = shadow_improvement
            out["Best_Shadow_Beats_Final"] = bool(
                np.isfinite(shadow_improvement) and shadow_improvement > 1e-9
            )

    focused = next((r for r in rows if r.get("Stage") == "focused_shape_shadow"), None)
    if focused and final_row:
        final_mae = float(final_row.get("MAE_MWH", np.nan))
        focused_mae = float(focused.get("MAE_MWH", np.nan))
        focused_improvement = final_mae - focused_mae
        out["Focused_Shape_Shadow_MAE_MWH"] = focused.get("MAE_MWH")
        out["Focused_Shape_Shadow_MAE_Improvement_vs_Final_MWH"] = focused_improvement
        out["Focused_Shape_Shadow_Beats_Final"] = bool(
            np.isfinite(focused_improvement) and focused_improvement > 1e-9
        )
    return out


def metrics_summary(df: pd.DataFrame) -> dict[str, Any]:
    if df is None or df.empty:
        return {}
    stages = _available_stage_columns(df)
    raw = (
        _metric_dict(df["Actual_MWH"], df[stages.get(RAW_STAGE, "Raw_Forecast_MWH")])
        if RAW_STAGE in stages or "Raw_Forecast_MWH" in df
        else {}
    )
    final_col = (
        stages.get(FINAL_STAGE)
        or stages.get("recent_corrected_simulation")
        or stages.get(RAW_STAGE)
    )
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
        out["Final_Underforecast_At_Actual_Peak_MWH"] = final.get(
            "Underforecast_At_Actual_Peak_MWH"
        )
    if raw and final:
        out["Final_MAE_Improvement_vs_Raw_MWH"] = raw.get(
            "MAE_MWH", np.nan
        ) - final.get("MAE_MWH", np.nan)
        out["Final_RMSE_Improvement_vs_Raw_MWH"] = raw.get(
            "RMSE_MWH", np.nan
        ) - final.get("RMSE_MWH", np.nan)
        out["Final_Bias_Abs_Improvement_vs_Raw_MWH"] = abs(
            raw.get("Bias_MWH", np.nan)
        ) - abs(final.get("Bias_MWH", np.nan))
    out.update(_stage_leader_audit(df, final_col))
    return out


def _segment_metrics(
    group: pd.DataFrame, forecast_col: str = "Raw_Forecast_MWH"
) -> pd.Series:
    actual = _as_num(group["Actual_MWH"])
    forecast = _as_num(group[forecast_col])
    m = _metric_dict(actual, forecast)
    return pd.Series(
        {k: v for k, v in m.items() if k not in {"Stage", "ForecastColumn"}}
    )


def build_metrics_by_group(
    df: pd.DataFrame,
    keys: list[str],
    min_count: int = 1,
    forecast_col: str = "Raw_Forecast_MWH",
) -> pd.DataFrame:
    """Fast grouped accuracy metrics for a single forecast column."""
    if (
        df is None
        or df.empty
        or forecast_col not in df.columns
        or not all(k in df.columns for k in keys)
    ):
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
    work["APE"] = np.where(
        work["Actual_MWH"].abs() > 1e-9,
        work["AbsError_MWH"] / work["Actual_MWH"].abs() * 100.0,
        np.nan,
    )
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
        Underforecast_Rate_PCT=(
            "Underforecast_Flag",
            lambda s: float(s.mean() * 100.0),
        ),
        P90_AbsError_MWH=("AbsError_MWH", lambda s: float(s.quantile(0.90))),
        Max_Underforecast_MWH=("Residual_MWH", "max"),
        Max_Overforecast_MWH=("OverResidual_MWH", "max"),
        Mean_SqResidual_MWH=("SqResidual_MWH", "mean"),
    ).reset_index()
    out["RMSE_MWH"] = np.sqrt(out.pop("Mean_SqResidual_MWH"))
    out = out[out["N"] >= int(min_count)].copy()
    sort_cols = [c for c in ["MAE_MWH", "N"] if c in out.columns]
    return (
        out.sort_values(
            sort_cols, ascending=[False, False][: len(sort_cols)]
        ).reset_index(drop=True)
        if sort_cols
        else out.reset_index(drop=True)
    )


def build_metrics_by_group_by_stage(
    df: pd.DataFrame,
    keys: list[str],
    min_count: int = 1,
    stages: dict[str, str] | None = None,
) -> pd.DataFrame:
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
    return (
        pd.concat(frames, ignore_index=True, sort=False) if frames else pd.DataFrame()
    )


def build_backtest_metrics_by_segment(
    df: pd.DataFrame, min_count: int = 6, forecast_col: str = "Raw_Forecast_MWH"
) -> pd.DataFrame:
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
        seg = build_metrics_by_group(
            df, keys, min_count=min_count, forecast_col=forecast_col
        )
        if not seg.empty:
            seg.insert(0, "Segment", segment_name)
            seg.insert(1, "SegmentKeys", "+".join(keys))
            frames.append(seg)
    return (
        pd.concat(frames, ignore_index=True, sort=False) if frames else pd.DataFrame()
    )


def build_backtest_metrics_by_segment_by_stage(
    df: pd.DataFrame, min_count: int = 6
) -> pd.DataFrame:
    stages = _available_stage_columns(df)
    preferred = {
        k: v
        for k, v in stages.items()
        if k
        in [
            RAW_STAGE,
            "targeted_residual_meta_adjusted",
            "residual_calibrated",
            "heat_adjusted",
            "warm_ramp_adjusted",
            "cloud_solar_adjusted",
            "peak_risk_adjusted",
            "recent_corrected_simulation",
            FINAL_STAGE,
            "auto_residual_shadow",
            "auto_residual_broad_hot_peak_shadow",
            "daily_peak_shadow",
            "hot_ramp_peak_shadow",
            "heat_persistence_peak_shadow",
            "heat_analog_shadow",
            "prophet_benchmark",
            "catboost_benchmark",
            "baseline_same_hour_yesterday",
            "baseline_rolling_7day_same_hour_avg",
        ]
    }
    frames = []
    for stage, col in preferred.items():
        seg = build_backtest_metrics_by_segment(
            df, min_count=min_count, forecast_col=col
        )
        if not seg.empty:
            seg.insert(0, "Stage", stage)
            seg.insert(1, "ForecastColumn", col)
            frames.append(seg)
    return (
        pd.concat(frames, ignore_index=True, sort=False) if frames else pd.DataFrame()
    )


def build_daily_peak_miss_table(
    df: pd.DataFrame,
    forecast_col: str = "Raw_Forecast_MWH",
    stage: str | None = None,
    peak_hours: list[int] | set[int] | None = None,
    peak_window_name: str | None = None,
) -> pd.DataFrame:
    if df is None or df.empty or forecast_col not in df.columns:
        return pd.DataFrame()
    work = df.copy()
    if "Date" not in work.columns:
        if "DT" not in work.columns:
            return pd.DataFrame()
        work = prep_backtest(work)
        if forecast_col not in work.columns:
            return pd.DataFrame()
    hour_set = {int(h) for h in peak_hours} if peak_hours is not None else set()
    if hour_set:
        if "Hour" not in work.columns:
            return pd.DataFrame()
        hour_mask = _as_num(work["Hour"]).astype("Int64").isin(hour_set).fillna(False)
        work = work.loc[hour_mask].copy()
        if work.empty:
            return pd.DataFrame()
    rows = []
    for date, g in work.groupby("Date", dropna=False):
        g = g.dropna(subset=["Actual_MWH", forecast_col])
        if g.empty:
            continue
        actual_peak_idx = g["Actual_MWH"].idxmax()
        forecast_peak_idx = g[forecast_col].idxmax()
        actual_peak = g.loc[actual_peak_idx]
        forecast_peak = g.loc[forecast_peak_idx]
        timing_error = (
            pd.to_datetime(forecast_peak["DT"]) - pd.to_datetime(actual_peak["DT"])
        ).total_seconds() / 3600.0
        rows.append(
            {
                "Stage": stage or RAW_STAGE,
                "ForecastColumn": forecast_col,
                "Date": date,
                "PeakWindowName": peak_window_name
                or ("custom_hours" if hour_set else "all_hours"),
                "PeakHours": (
                    ",".join(str(h) for h in sorted(hour_set)) if hour_set else "all"
                ),
                "Season": actual_peak.get("Season"),
                "Actual_Peak_DT": actual_peak["DT"],
                "Actual_Peak_Hour": int(actual_peak["Hour"]),
                "Actual_Peak_MWH": float(actual_peak["Actual_MWH"]),
                "Forecast_At_Actual_Peak_MWH": float(actual_peak[forecast_col]),
                "Underforecast_At_Actual_Peak_MWH": float(
                    actual_peak["Actual_MWH"] - actual_peak[forecast_col]
                ),
                "Forecast_Peak_DT": forecast_peak["DT"],
                "Forecast_Peak_Hour": int(forecast_peak["Hour"]),
                "Forecast_Peak_MWH": float(forecast_peak[forecast_col]),
                "Daily_Peak_Timing_Error_Hours": float(timing_error),
                "Daily_Energy_Actual_MWH": float(g["Actual_MWH"].sum()),
                "Daily_Energy_Forecast_MWH": float(g[forecast_col].sum()),
                "Daily_Energy_Error_MWH": float(
                    g[forecast_col].sum() - g["Actual_MWH"].sum()
                ),
                "Daily_MAE_MWH": float(
                    (g["Actual_MWH"] - g[forecast_col]).abs().mean()
                ),
                "Daily_MAPE_PCT": _safe_mape(g["Actual_MWH"], g[forecast_col]),
                "DailyMaxTempBucket": actual_peak.get("DailyMaxTempBucket"),
                "Temperature_DailyMax": (
                    float(actual_peak["Temperature_DailyMax"])
                    if "Temperature_DailyMax" in actual_peak
                    and pd.notna(actual_peak["Temperature_DailyMax"])
                    else np.nan
                ),
                "CloudCoverBucket": actual_peak.get("CloudCoverBucket"),
                "BTMSolarBucket": actual_peak.get("BTMSolarBucket"),
                "SolarLossBucket": actual_peak.get("SolarLossBucket"),
            }
        )
    out = pd.DataFrame(rows)
    if not out.empty:
        out.sort_values(
            ["Underforecast_At_Actual_Peak_MWH", "Daily_MAE_MWH"],
            ascending=[False, False],
            inplace=True,
        )
    return out.reset_index(drop=True)


def build_daily_peak_miss_by_stage(
    df: pd.DataFrame,
    peak_hours: list[int] | set[int] | None = None,
    peak_window_name: str | None = None,
) -> pd.DataFrame:
    frames = []
    for stage, col in _available_stage_columns(df).items():
        if stage.startswith("baseline_") or stage in {"xgb_component", "lgb_component"}:
            continue
        tab = build_daily_peak_miss_table(
            df,
            forecast_col=col,
            stage=stage,
            peak_hours=peak_hours,
            peak_window_name=peak_window_name,
        )
        if not tab.empty:
            frames.append(tab)
    return (
        pd.concat(frames, ignore_index=True, sort=False) if frames else pd.DataFrame()
    )


def build_daily_peak_window_miss_by_stage(
    df: pd.DataFrame,
    peak_hours: list[int] | set[int] | None = None,
    peak_window_name: str = "HE18to20_CodeHours17to19",
) -> pd.DataFrame:
    """Daily peak timing/magnitude metrics inside a specific operational peak window."""
    if df is None or df.empty:
        return pd.DataFrame()
    peak_hours = peak_hours or HE18_20_CODE_HOURS
    if "Replay_Origin_ID" not in df.columns:
        return build_daily_peak_miss_by_stage(
            df, peak_hours=peak_hours, peak_window_name=peak_window_name
        )

    frames: list[pd.DataFrame] = []
    for origin_id, group in df.groupby("Replay_Origin_ID", dropna=False):
        peak = build_daily_peak_miss_by_stage(
            group, peak_hours=peak_hours, peak_window_name=peak_window_name
        )
        if peak.empty:
            continue
        peak.insert(0, "Replay_Origin_ID", origin_id)
        peak.insert(
            1,
            "Replay_Origin_DT",
            (
                group["Replay_Origin_DT"].iloc[0]
                if "Replay_Origin_DT" in group.columns
                else pd.NaT
            ),
        )
        frames.append(peak)
    return (
        pd.concat(frames, ignore_index=True, sort=False) if frames else pd.DataFrame()
    )


def _default_forecast_col(df: pd.DataFrame) -> str | None:
    for stage in [
        FINAL_STAGE,
        "stage_selected_production",
        "recent_corrected_simulation",
        RAW_STAGE,
    ]:
        col = _available_stage_columns(df).get(stage)
        if col:
            return col
    return None


def _hours_label(hours: list[int] | set[int]) -> str:
    return ",".join(str(int(h)) for h in sorted({int(h) for h in hours}))


def _diagnostic_slice_metric_row(
    df: pd.DataFrame,
    mask: pd.Series,
    *,
    slice_name: str,
    slice_group: str,
    slice_value: str,
    purpose: str,
    hour_definition: str,
    forecast_col: str,
    gate: str = "diagnostic_only",
    gate_mae_mwh: float | None = None,
) -> dict[str, Any]:
    mask = pd.Series(mask, index=df.index).fillna(False).astype(bool)
    subset = df.loc[mask].copy()
    row: dict[str, Any] = {
        "Slice": slice_name,
        "SliceGroup": slice_group,
        "SliceValue": slice_value,
        "Purpose": purpose,
        "HourDefinition": hour_definition,
        "ForecastColumn": forecast_col,
        "Gate": gate,
        "Gate_MAE_MWH": gate_mae_mwh if gate_mae_mwh is not None else np.nan,
        "Pass": np.nan,
        "N": 0,
    }
    if (
        subset.empty
        or forecast_col not in subset.columns
        or "Actual_MWH" not in subset.columns
    ):
        return row
    metrics = _metric_dict(subset["Actual_MWH"], subset[forecast_col], col=forecast_col)
    if not metrics:
        return row
    row.update(
        {k: v for k, v in metrics.items() if k not in {"Stage", "ForecastColumn"}}
    )
    if gate_mae_mwh is not None:
        mae = float(row.get("MAE_MWH", np.nan))
        row["Pass"] = bool(np.isfinite(mae) and mae <= float(gate_mae_mwh))
    return row


def _forecast_day_bucket(values: pd.Series) -> pd.Series:
    day = _as_num(values)
    return pd.cut(
        day,
        bins=[0, 1, 3, 7, 16, 999],
        labels=["Day1", "Days2-3", "Days4-7", "Days8-16", "Day17+"],
        include_lowest=True,
    ).astype("object")


def _peak_dailymax_bias_band(values: pd.Series) -> pd.Series:
    temp = _as_num(values)
    return pd.cut(
        temp,
        bins=[-999, 75, 85, 90, 92.5, 95, 98, 100, 105, 999],
        labels=[
            "<75",
            "75-85",
            "85-90",
            "90-92.5",
            "92.5-95",
            "95-98",
            "98-100",
            "100-105",
            "105+",
        ],
        include_lowest=True,
        right=False,
    ).astype("object")


def _peak_cloud_bias_band(values: pd.Series) -> pd.Series:
    cloud = _as_num(values)
    if cloud.notna().any() and cloud.max(skipna=True) > 1.5:
        cloud = cloud / 100.0
    return pd.cut(
        cloud,
        bins=[-0.001, 0.10, 0.35, 0.60, 1.001],
        labels=["Clear/Low", "Some/Partly", "Cloudy", "Overcast"],
        include_lowest=True,
    ).astype("object")


def _lag_anchor_state_band(values: pd.Series) -> pd.Series:
    delta = _as_num(values)
    return pd.cut(
        delta,
        bins=[-999, -10, 10, 25, 50, 999],
        labels=["<-10", "-10..10", "10..25", "25..50", ">50"],
        include_lowest=True,
    ).astype("object")


def build_peak_window_bias_scorecard(
    df: pd.DataFrame,
    forecast_col: str | None = None,
    min_count: int = 5,
) -> pd.DataFrame:
    """Rank HE14-18 residual-bias slices by net residual contribution."""
    if df is None or df.empty:
        return pd.DataFrame()
    work = prep_backtest(df)
    forecast_col = forecast_col or _default_forecast_col(work)
    if (
        not forecast_col
        or forecast_col not in work.columns
        or "Actual_MWH" not in work.columns
    ):
        return pd.DataFrame()

    hour = _as_num(work.get("Hour", pd.Series(np.nan, index=work.index))).astype(
        "Int64"
    )
    work = work.loc[hour.isin(PEAK_WINDOW_14_18_HOURS).fillna(False)].copy()
    if work.empty:
        return pd.DataFrame()
    hour = _as_num(work.get("Hour", pd.Series(np.nan, index=work.index))).astype(
        "Int64"
    )

    daily_max = _as_num(
        work.get(
            "Temperature_DailyMax",
            work.get("Temperature", pd.Series(np.nan, index=work.index)),
        )
    )
    cloud = _as_num(work.get("CloudCover_Norm", pd.Series(np.nan, index=work.index)))
    raw = _as_num(work.get("Raw_Forecast_MWH", pd.Series(np.nan, index=work.index)))
    same7 = _as_num(
        work.get("MWH_SameHour7DayMean", pd.Series(np.nan, index=work.index))
    )
    lag24_minus_7 = _as_num(
        work.get(
            "Lag24_Minus_SameHour7DayMean_MWH", pd.Series(np.nan, index=work.index)
        )
    )
    raw_minus_7 = raw - same7

    work["Forecast_Day_Bucket"] = _forecast_day_bucket(work["Forecast_Day"])
    work["Peak_Hour_Band"] = np.where(hour.isin([14, 15]), "HE14-15", "HE16-18")
    work["DailyMaxTempBiasBand"] = _peak_dailymax_bias_band(daily_max)
    work["CloudCoverBiasBand"] = _peak_cloud_bias_band(cloud)
    work["LagAnchorState"] = _lag_anchor_state_band(raw_minus_7)
    work["RawMinusSameHour7_MWH"] = raw_minus_7
    work["Lag24MinusSameHour7_MWH"] = lag24_minus_7
    work["Forecast_MWH"] = _as_num(work[forecast_col])
    work["Actual_MWH"] = _as_num(work["Actual_MWH"])
    work["Residual_MWH"] = work["Actual_MWH"] - work["Forecast_MWH"]
    work["AbsError_MWH"] = work["Residual_MWH"].abs()
    work["SqResidual_MWH"] = work["Residual_MWH"] ** 2
    work["Underforecast_Flag"] = work["Residual_MWH"].gt(0.0).astype(float)
    work["Positive_Residual_MWH"] = work["Residual_MWH"].clip(lower=0.0)
    work["Negative_Residual_MWH"] = work["Residual_MWH"].clip(upper=0.0)
    work["Raw_Residual_MWH"] = work["Actual_MWH"] - raw

    keys = [
        "Month",
        "Forecast_Day_Bucket",
        "Peak_Hour_Band",
        "DailyMaxTempBiasBand",
        "CloudCoverBiasBand",
        "LagAnchorState",
    ]
    work = work.dropna(subset=["Actual_MWH", "Forecast_MWH"])
    if work.empty:
        return pd.DataFrame()

    grouped = work.groupby(keys, dropna=False, observed=True)
    out = grouped.agg(
        N=("Residual_MWH", "size"),
        Residual_Sum_MWH=("Residual_MWH", "sum"),
        Positive_Residual_Sum_MWH=("Positive_Residual_MWH", "sum"),
        Negative_Residual_Sum_MWH=("Negative_Residual_MWH", "sum"),
        Bias_MWH=("Residual_MWH", "mean"),
        MAE_MWH=("AbsError_MWH", "mean"),
        Mean_SqResidual_MWH=("SqResidual_MWH", "mean"),
        Underforecast_Rate_PCT=(
            "Underforecast_Flag",
            lambda s: float(s.mean() * 100.0),
        ),
        P90_AbsError_MWH=("AbsError_MWH", lambda s: float(s.quantile(0.90))),
        Max_Underforecast_MWH=("Residual_MWH", "max"),
        Max_Overforecast_MWH=("Residual_MWH", lambda s: float((-s).max())),
        Actual_Mean_MWH=("Actual_MWH", "mean"),
        Forecast_Mean_MWH=("Forecast_MWH", "mean"),
        Raw_Bias_MWH=("Raw_Residual_MWH", "mean"),
        Raw_Residual_Sum_MWH=("Raw_Residual_MWH", "sum"),
        Mean_RawMinusSameHour7_MWH=("RawMinusSameHour7_MWH", "mean"),
        Mean_Lag24MinusSameHour7_MWH=("Lag24MinusSameHour7_MWH", "mean"),
    ).reset_index()
    out["RMSE_MWH"] = np.sqrt(out.pop("Mean_SqResidual_MWH"))
    out = out[out["N"] >= int(min_count)].copy()
    if out.empty:
        return out
    out.insert(0, "ForecastColumn", forecast_col)
    out.insert(1, "RankBasis", "Residual_Sum_MWH_desc")
    out["Abs_Residual_Sum_MWH"] = out["Residual_Sum_MWH"].abs()
    out["Bias_Delta_vs_Raw_MWH"] = out["Bias_MWH"] - out["Raw_Bias_MWH"]
    return out.sort_values(
        ["Residual_Sum_MWH", "N"], ascending=[False, False]
    ).reset_index(drop=True)


def build_peak_window_expansion_scorecard(
    df: pd.DataFrame,
    forecast_col: str | None = None,
) -> pd.DataFrame:
    """Compare the current 14-18 peak window against wider late-evening candidates."""
    if df is None or df.empty:
        return pd.DataFrame()
    work = prep_backtest(df)
    forecast_col = forecast_col or _default_forecast_col(work)
    if not forecast_col or forecast_col not in work.columns:
        return pd.DataFrame()

    hour = _as_num(work.get("Hour", pd.Series(np.nan, index=work.index))).astype(
        "Int64"
    )
    rows = [
        _diagnostic_slice_metric_row(
            work,
            hour.isin(PEAK_WINDOW_14_18_HOURS),
            slice_name="PeakWindowHours14to18",
            slice_group="peak_window",
            slice_value="current_gate",
            purpose="Current production peak-window gate",
            hour_definition=f"code Hour {_hours_label(PEAK_WINDOW_14_18_HOURS)}",
            forecast_col=forecast_col,
            gate="mae<=5.5",
            gate_mae_mwh=5.5,
        ),
        _diagnostic_slice_metric_row(
            work,
            hour.isin(PEAK_WINDOW_14_20_HOURS),
            slice_name="PeakWindowHours14to20",
            slice_group="peak_window",
            slice_value="candidate_expansion",
            purpose="A/B candidate before changing peak-window training weights",
            hour_definition=f"code Hour {_hours_label(PEAK_WINDOW_14_20_HOURS)}",
            forecast_col=forecast_col,
        ),
        _diagnostic_slice_metric_row(
            work,
            hour.isin([19, 20]),
            slice_name="LatePeakHours19to20",
            slice_group="peak_window",
            slice_value="late_extension_only",
            purpose="Isolate incremental HE20/HE21 late-peak behavior",
            hour_definition="code Hour 19-20",
            forecast_col=forecast_col,
        ),
        _diagnostic_slice_metric_row(
            work,
            hour.isin(HE18_20_CODE_HOURS),
            slice_name="HE18to20_CodeHours17to19",
            slice_group="peak_window",
            slice_value="solar_shifted_daily_peak",
            purpose="Daily peak timing/magnitude focus for later net-load peaks",
            hour_definition="operational HE18-20 / code Hour 17-19",
            forecast_col=forecast_col,
        ),
    ]
    return pd.DataFrame(rows)


def build_peak_window_14to20_metrics_by_stage(
    df: pd.DataFrame, min_count: int = 1
) -> pd.DataFrame:
    if df is None or df.empty:
        return pd.DataFrame()
    work = prep_backtest(df)
    hour = _as_num(work.get("Hour", pd.Series(np.nan, index=work.index))).astype(
        "Int64"
    )
    subset = work.loc[hour.isin(PEAK_WINDOW_14_20_HOURS)].copy()
    keys = [
        k
        for k in ["Replay_Horizon_Bucket", "Forecast_Day", "Hour"]
        if k in subset.columns
    ]
    return (
        build_metrics_by_group_by_stage(subset, keys, min_count=min_count)
        if keys
        else pd.DataFrame()
    )


def build_extreme_heat_peak_scorecard(
    df: pd.DataFrame,
    forecast_col: str | None = None,
    min_daily_max_f: float = EXTREME_HEAT_MIN_DAILY_MAX_F,
) -> pd.DataFrame:
    """Dedicated 105F+ hot-peak scorecard. Rows are diagnostic until a gate is calibrated."""
    if df is None or df.empty:
        return pd.DataFrame()
    work = prep_backtest(df)
    forecast_col = forecast_col or _default_forecast_col(work)
    if not forecast_col or forecast_col not in work.columns:
        return pd.DataFrame()

    hour = _as_num(work.get("Hour", pd.Series(np.nan, index=work.index))).astype(
        "Int64"
    )
    daily_max = _as_num(
        work.get("Temperature_DailyMax", pd.Series(np.nan, index=work.index))
    )
    hot = daily_max.ge(float(min_daily_max_f))
    rows = [
        _diagnostic_slice_metric_row(
            work,
            hot & hour.isin(PEAK_WINDOW_14_20_HOURS),
            slice_name="ExtremeHeat105PlusPeakWindowHours14to20",
            slice_group="extreme_heat_peak",
            slice_value="105f_plus_14to20",
            purpose="Sparse extreme-heat peak risk",
            hour_definition=f"Temperature_DailyMax >= {float(min_daily_max_f):.1f}F; code Hour {_hours_label(PEAK_WINDOW_14_20_HOURS)}",
            forecast_col=forecast_col,
        ),
        _diagnostic_slice_metric_row(
            work,
            hot & hour.between(16, 20),
            slice_name="ExtremeHeat105PlusHotPeakHours16to20",
            slice_group="extreme_heat_peak",
            slice_value="105f_plus_hot_peak_gate_hours",
            purpose="105F+ subset of current hot-peak operating hours",
            hour_definition=f"Temperature_DailyMax >= {float(min_daily_max_f):.1f}F; code Hour 16-20",
            forecast_col=forecast_col,
        ),
        _diagnostic_slice_metric_row(
            work,
            hot & hour.isin(HE18_20_CODE_HOURS),
            slice_name="ExtremeHeat105PlusHE18to20CodeHours17to19",
            slice_group="extreme_heat_peak",
            slice_value="105f_plus_he18_20",
            purpose="105F+ daily peak timing/magnitude focus window",
            hour_definition=f"Temperature_DailyMax >= {float(min_daily_max_f):.1f}F; operational HE18-20 / code Hour 17-19",
            forecast_col=forecast_col,
        ),
    ]
    return pd.DataFrame(rows)


def build_extreme_heat_peak_metrics_by_stage(
    df: pd.DataFrame,
    min_daily_max_f: float = EXTREME_HEAT_MIN_DAILY_MAX_F,
    min_count: int = 1,
) -> pd.DataFrame:
    if df is None or df.empty:
        return pd.DataFrame()
    work = prep_backtest(df)
    hour = _as_num(work.get("Hour", pd.Series(np.nan, index=work.index))).astype(
        "Int64"
    )
    daily_max = _as_num(
        work.get("Temperature_DailyMax", pd.Series(np.nan, index=work.index))
    )
    subset = work.loc[
        daily_max.ge(float(min_daily_max_f)) & hour.isin(PEAK_WINDOW_14_20_HOURS)
    ].copy()
    keys = [
        k
        for k in ["Replay_Horizon_Bucket", "Forecast_Day", "Hour"]
        if k in subset.columns
    ]
    return (
        build_metrics_by_group_by_stage(subset, keys, min_count=min_count)
        if keys
        else pd.DataFrame()
    )


def _heat_analog_shadow_cfg(config: dict | None) -> dict[str, Any]:
    if not isinstance(config, dict):
        return {}
    diag = config.get("diagnostics", {}) or {}
    if isinstance(diag.get("heat_analog_shadow"), dict):
        return diag.get("heat_analog_shadow", {}) or {}
    return {}


def apply_multisummer_heat_analog_shadow(
    df: pd.DataFrame,
    config: dict | None = None,
    base_col: str | None = None,
) -> pd.DataFrame:
    """Add a non-production, prior-origin multi-summer analog forecast for 105F+ peak rows."""
    if df is None or df.empty:
        return pd.DataFrame() if df is None else df.copy()
    required = {"DT", "Actual_MWH", "Hour", "Temperature_DailyMax"}
    if not required.issubset(df.columns):
        return df.copy()

    out = prep_backtest(df)
    base_col = base_col or _default_forecast_col(out)
    if not base_col or base_col not in out.columns:
        return out

    cfg = _heat_analog_shadow_cfg(config)
    min_daily_max_f = float(cfg.get("min_daily_max_f", EXTREME_HEAT_MIN_DAILY_MAX_F))
    temp_window_f = float(cfg.get("temp_window_f", 3.0))
    blend = float(cfg.get("blend", 0.50))
    cap_mwh = abs(float(cfg.get("cap_mwh", 12.0)))
    min_analog_count = max(1, int(cfg.get("min_analog_count", 3)))
    peak_hours = [int(h) for h in cfg.get("peak_hours", PEAK_WINDOW_14_20_HOURS)]
    summer_months = [int(m) for m in cfg.get("months", [6, 7, 8, 9])]

    out["Heat_Analog_Model_Version"] = "multi_summer_v1"
    out["Heat_Analog_Shadow_Mode"] = "shadow_only"
    out["Heat_Analog_Production_Scope"] = "not_production"
    out["Heat_Analog_Base_Forecast_MWH"] = _as_num(out[base_col])
    out["Heat_Analog_Shadow_Correction_MWH"] = 0.0
    out["Heat_Analog_Shadow_Forecast_MWH"] = out["Heat_Analog_Base_Forecast_MWH"]
    out["Heat_Analog_Shadow_Applied_Flag"] = 0
    out["Heat_Analog_Scope_Flag"] = 0
    out["Heat_Analog_Source"] = "out_of_scope"
    out["Heat_Analog_Temp_Window_F"] = temp_window_f
    out["Heat_Analog_MinDailyMax_F"] = min_daily_max_f
    out["Heat_Analog_PeakHours"] = _hours_label(peak_hours)
    out["Heat_Analog_MinAnalogCount"] = min_analog_count
    out["Heat_Analog_Count_SameHour_PreOrigin"] = 0
    out["Heat_Analog_Count_HE14_20_PreOrigin"] = 0
    out["Heat_Analog_Actual_Mean_SameHour_PreOrigin_MWH"] = np.nan
    out["Heat_Analog_Actual_P90_SameHour_PreOrigin_MWH"] = np.nan
    out["Heat_Analog_Actual_Mean_HE14_20_PreOrigin_MWH"] = np.nan
    out["Heat_Analog_Actual_P90_HE14_20_PreOrigin_MWH"] = np.nan

    hour = _as_num(out["Hour"]).astype("Int64")
    month = _as_num(out.get("Month", pd.Series(np.nan, index=out.index))).astype(
        "Int64"
    )
    daily_max = _as_num(out["Temperature_DailyMax"])
    scope = (
        daily_max.ge(min_daily_max_f)
        & hour.isin(peak_hours)
        & month.isin(summer_months)
    )
    scope = scope.fillna(False).astype(bool)
    out.loc[scope, "Heat_Analog_Scope_Flag"] = 1
    out.loc[scope, "Heat_Analog_Source"] = "no_pre_origin_match"
    if not scope.any():
        actual = _as_num(out["Actual_MWH"])
        shadow = _as_num(out["Heat_Analog_Shadow_Forecast_MWH"])
        out["Heat_Analog_Shadow_Residual_MWH"] = actual - shadow
        out["Heat_Analog_Shadow_AbsError_MWH"] = out[
            "Heat_Analog_Shadow_Residual_MWH"
        ].abs()
        out["Heat_Analog_Delta_AbsError_MWH"] = (
            out["Heat_Analog_Shadow_AbsError_MWH"]
            - (actual - out["Heat_Analog_Base_Forecast_MWH"]).abs()
        )
        return out

    source = out.drop_duplicates(subset=["DT"], keep="first").copy()
    source["_DT_UTC"] = pd.to_datetime(source["DT"], errors="coerce", utc=True)
    source["_Month_Num"] = _as_num(source["Month"])
    source["_Hour_Num"] = _as_num(source["Hour"])
    source["_DailyMax_Num"] = _as_num(source["Temperature_DailyMax"])
    source["_Actual_Num"] = _as_num(source["Actual_MWH"])
    source = source.dropna(
        subset=["_DT_UTC", "_Month_Num", "_Hour_Num", "_DailyMax_Num", "_Actual_Num"]
    )

    out["_HeatAnalog_Ref_UTC"] = (
        pd.to_datetime(out["Replay_Origin_DT"], errors="coerce", utc=True)
        if "Replay_Origin_DT" in out.columns
        else pd.to_datetime(out["DT"], errors="coerce", utc=True)
    )
    out["_HeatAnalog_Hour_Num"] = _as_num(out["Hour"])
    out["_HeatAnalog_Month_Num"] = _as_num(out["Month"])
    out["_HeatAnalog_DailyMax_Num"] = _as_num(out["Temperature_DailyMax"])

    for idx in out.index[scope]:
        ref_dt = out.at[idx, "_HeatAnalog_Ref_UTC"]
        row_hour = out.at[idx, "_HeatAnalog_Hour_Num"]
        row_month = out.at[idx, "_HeatAnalog_Month_Num"]
        row_temp = out.at[idx, "_HeatAnalog_DailyMax_Num"]
        base_forecast = out.at[idx, "Heat_Analog_Base_Forecast_MWH"]
        if (
            pd.isna(ref_dt)
            or not np.isfinite(row_hour)
            or not np.isfinite(row_month)
            or not np.isfinite(row_temp)
            or not np.isfinite(base_forecast)
        ):
            continue

        prior = source[source["_DT_UTC"].lt(ref_dt)]
        if prior.empty:
            continue
        temp_match = prior["_DailyMax_Num"].ge(min_daily_max_f) & prior[
            "_DailyMax_Num"
        ].between(
            float(row_temp) - temp_window_f,
            float(row_temp) + temp_window_f,
        )
        hour_match = prior["_Hour_Num"].eq(float(row_hour))
        same_hour_temp = prior.loc[temp_match & hour_match].copy()
        candidates = [
            (
                "same_month_multi_summer",
                same_hour_temp.loc[same_hour_temp["_Month_Num"].eq(float(row_month))],
            ),
            (
                "summer_months_multi_summer",
                same_hour_temp.loc[same_hour_temp["_Month_Num"].isin(summer_months)],
            ),
            ("all_prior_same_hour_temp", same_hour_temp),
        ]

        chosen_source = None
        same_values = pd.Series(dtype=float)
        for source_name, candidate in candidates:
            values = candidate["_Actual_Num"].dropna()
            if len(values) >= min_analog_count:
                chosen_source = source_name
                same_values = values
                break
        if chosen_source is None:
            continue

        peak_values = prior.loc[
            temp_match
            & prior["_Month_Num"].isin(summer_months)
            & prior["_Hour_Num"].isin(PEAK_WINDOW_14_20_HOURS),
            "_Actual_Num",
        ].dropna()
        analog_mean = float(same_values.mean())
        raw_correction = (analog_mean - float(base_forecast)) * blend
        correction = (
            float(np.clip(raw_correction, -cap_mwh, cap_mwh))
            if cap_mwh > 0
            else float(raw_correction)
        )

        out.at[idx, "Heat_Analog_Source"] = chosen_source
        out.at[idx, "Heat_Analog_Shadow_Correction_MWH"] = correction
        out.at[idx, "Heat_Analog_Shadow_Forecast_MWH"] = (
            float(base_forecast) + correction
        )
        out.at[idx, "Heat_Analog_Shadow_Applied_Flag"] = int(abs(correction) > 1e-9)
        out.at[idx, "Heat_Analog_Count_SameHour_PreOrigin"] = int(len(same_values))
        out.at[idx, "Heat_Analog_Actual_Mean_SameHour_PreOrigin_MWH"] = analog_mean
        out.at[idx, "Heat_Analog_Actual_P90_SameHour_PreOrigin_MWH"] = float(
            same_values.quantile(0.90)
        )
        out.at[idx, "Heat_Analog_Count_HE14_20_PreOrigin"] = int(len(peak_values))
        out.at[idx, "Heat_Analog_Actual_Mean_HE14_20_PreOrigin_MWH"] = (
            float(peak_values.mean()) if len(peak_values) else np.nan
        )
        out.at[idx, "Heat_Analog_Actual_P90_HE14_20_PreOrigin_MWH"] = (
            float(peak_values.quantile(0.90)) if len(peak_values) else np.nan
        )

    actual = _as_num(out["Actual_MWH"])
    base = _as_num(out["Heat_Analog_Base_Forecast_MWH"])
    shadow = _as_num(out["Heat_Analog_Shadow_Forecast_MWH"])
    out["Heat_Analog_Base_Residual_MWH"] = actual - base
    out["Heat_Analog_Base_AbsError_MWH"] = out["Heat_Analog_Base_Residual_MWH"].abs()
    out["Heat_Analog_Shadow_Residual_MWH"] = actual - shadow
    out["Heat_Analog_Shadow_AbsError_MWH"] = out[
        "Heat_Analog_Shadow_Residual_MWH"
    ].abs()
    out["Heat_Analog_Delta_AbsError_MWH"] = (
        out["Heat_Analog_Shadow_AbsError_MWH"] - out["Heat_Analog_Base_AbsError_MWH"]
    )
    out["Actual_Minus_Heat_Analog_SameHour_Mean_MWH"] = actual - _as_num(
        out["Heat_Analog_Actual_Mean_SameHour_PreOrigin_MWH"]
    )
    return out.drop(
        columns=[
            "_HeatAnalog_Ref_UTC",
            "_HeatAnalog_Hour_Num",
            "_HeatAnalog_Month_Num",
            "_HeatAnalog_DailyMax_Num",
        ],
        errors="ignore",
    )


def build_heat_analog_shadow_detail(
    df: pd.DataFrame, config: dict | None = None
) -> pd.DataFrame:
    if df is None or df.empty:
        return pd.DataFrame()
    work = (
        df.copy()
        if "Heat_Analog_Shadow_Forecast_MWH" in df.columns
        else apply_multisummer_heat_analog_shadow(df, config=config)
    )
    if work.empty or "Heat_Analog_Scope_Flag" not in work.columns:
        return pd.DataFrame()
    scope = _as_num(work["Heat_Analog_Scope_Flag"]).fillna(0).eq(1)
    detail = work.loc[scope].copy()
    if detail.empty:
        return pd.DataFrame()
    preferred_cols = [
        "DT",
        "Replay_Origin_ID",
        "Replay_Origin_DT",
        "Replay_Origin_Year",
        "Replay_Origin_Month",
        "Forecast_Lead_Hour",
        "Forecast_Day",
        "Replay_Horizon_Bucket",
        "Season",
        "Month",
        "Hour",
        "Actual_MWH",
        "Temperature",
        "Temperature_DailyMax",
        "DailyMaxTempBucket",
        "CloudCover_Norm",
        "Raw_Forecast_MWH",
        "Final_Backtest_Forecast_MWH",
        "Heat_Analog_Base_Forecast_MWH",
        "Heat_Analog_Shadow_Forecast_MWH",
        "Heat_Analog_Shadow_Correction_MWH",
        "Heat_Analog_Shadow_Applied_Flag",
        "Heat_Analog_Scope_Flag",
        "Heat_Analog_Source",
        "Heat_Analog_Temp_Window_F",
        "Heat_Analog_MinDailyMax_F",
        "Heat_Analog_PeakHours",
        "Heat_Analog_MinAnalogCount",
        "Heat_Analog_Count_SameHour_PreOrigin",
        "Heat_Analog_Actual_Mean_SameHour_PreOrigin_MWH",
        "Heat_Analog_Actual_P90_SameHour_PreOrigin_MWH",
        "Heat_Analog_Count_HE14_20_PreOrigin",
        "Heat_Analog_Actual_Mean_HE14_20_PreOrigin_MWH",
        "Heat_Analog_Actual_P90_HE14_20_PreOrigin_MWH",
        "Heat_Analog_Base_Residual_MWH",
        "Heat_Analog_Base_AbsError_MWH",
        "Heat_Analog_Shadow_Residual_MWH",
        "Heat_Analog_Shadow_AbsError_MWH",
        "Heat_Analog_Delta_AbsError_MWH",
        "Actual_Minus_Heat_Analog_SameHour_Mean_MWH",
    ]
    ordered = [col for col in preferred_cols if col in detail.columns]
    extras = [col for col in detail.columns if col not in ordered]
    sort_cols = [
        col for col in ["Replay_Origin_ID", "DT", "Hour"] if col in detail.columns
    ]
    detail = detail[ordered + extras]
    return (
        detail.sort_values(sort_cols, kind="stable").reset_index(drop=True)
        if sort_cols
        else detail.reset_index(drop=True)
    )


def build_heat_analog_shadow_metrics(
    df: pd.DataFrame, config: dict | None = None, min_count: int = 1
) -> pd.DataFrame:
    if df is None or df.empty:
        return pd.DataFrame()
    work = (
        df.copy()
        if "Heat_Analog_Shadow_Forecast_MWH" in df.columns
        else apply_multisummer_heat_analog_shadow(df, config=config)
    )
    if work.empty or "Heat_Analog_Scope_Flag" not in work.columns:
        return pd.DataFrame()
    scope = _as_num(work["Heat_Analog_Scope_Flag"]).fillna(0).eq(1)
    subset = work.loc[scope].copy()
    if subset.empty:
        return pd.DataFrame()
    keys = [
        k for k in ["Heat_Analog_Source", "Forecast_Day", "Hour"] if k in subset.columns
    ]
    return (
        build_metrics_by_group_by_stage(subset, keys, min_count=min_count)
        if keys
        else pd.DataFrame()
    )


def _daily_peak_shadow_cfg(config: dict | None) -> dict:
    if not isinstance(config, dict):
        return {}
    if isinstance(config.get("daily_peak_shadow_model"), dict):
        return config.get("daily_peak_shadow_model", {}) or {}
    cal = config.get("calibration", {}) or {}
    if isinstance(cal.get("daily_peak_shadow_model"), dict):
        return cal.get("daily_peak_shadow_model", {}) or {}
    stage_selector = cal.get("stage_selector", {}) or {}
    return stage_selector.get("daily_peak_shadow_model", {}) or {}


def _daily_peak_comparison_rows(
    df: pd.DataFrame,
    *,
    base_col: str,
    shadow_col: str,
    peak_hours: set[int],
) -> pd.DataFrame:
    if (
        df is None
        or df.empty
        or base_col not in df.columns
        or shadow_col not in df.columns
    ):
        return pd.DataFrame()
    required = {"Actual_MWH", "DT", "Date", "Hour"}
    if not required.issubset(df.columns):
        return pd.DataFrame()

    group_keys = ["Date"]
    if "Replay_Origin_ID" in df.columns:
        group_keys.insert(0, "Replay_Origin_ID")

    rows: list[dict[str, Any]] = []
    for keys, group in df.groupby(group_keys, dropna=False):
        scored = group.dropna(subset=["Actual_MWH", base_col, shadow_col]).copy()
        if scored.empty:
            continue
        if peak_hours:
            peak_mask = (
                _as_num(scored["Hour"]).astype("Int64").isin(peak_hours).fillna(False)
            )
            if peak_mask.any():
                scored = scored.loc[peak_mask].copy()
        if scored.empty:
            continue

        actual = _as_num(scored["Actual_MWH"]).astype(float)
        base = _as_num(scored[base_col]).astype(float)
        shadow = _as_num(scored[shadow_col]).astype(float)
        valid = actual.notna() & base.notna() & shadow.notna()
        if not valid.any():
            continue
        scored = scored.loc[valid]
        actual = actual.loc[valid]
        base = base.loc[valid]
        shadow = shadow.loc[valid]

        actual_peak_idx = actual.idxmax()
        base_peak_idx = base.idxmax()
        shadow_peak_idx = shadow.idxmax()
        actual_peak_dt = pd.to_datetime(scored.loc[actual_peak_idx, "DT"])
        base_peak_dt = pd.to_datetime(scored.loc[base_peak_idx, "DT"])
        shadow_peak_dt = pd.to_datetime(scored.loc[shadow_peak_idx, "DT"])
        forecast_day = np.nan
        if "Forecast_Day" in scored.columns:
            day_values = _as_num(scored["Forecast_Day"]).dropna()
            if not day_values.empty:
                forecast_day = float(day_values.median())

        row: dict[str, Any] = {
            "Date": scored.loc[actual_peak_idx, "Date"],
            "Forecast_Day": forecast_day,
            "Actual_Peak_MWH": float(actual.max()),
            "Base_AtActualPeak_MWH": float(base.loc[actual_peak_idx]),
            "Shadow_AtActualPeak_MWH": float(shadow.loc[actual_peak_idx]),
            "Base_ForecastPeak_MWH": float(base.max()),
            "Shadow_ForecastPeak_MWH": float(shadow.max()),
            "Base_PeakAtActual_AbsError_MWH": float(
                abs(actual.loc[actual_peak_idx] - base.loc[actual_peak_idx])
            ),
            "Shadow_PeakAtActual_AbsError_MWH": float(
                abs(actual.loc[actual_peak_idx] - shadow.loc[actual_peak_idx])
            ),
            "Base_ForecastPeak_AbsError_MWH": float(abs(actual.max() - base.max())),
            "Shadow_ForecastPeak_AbsError_MWH": float(abs(actual.max() - shadow.max())),
            "Base_Bias_AtActualPeak_MWH": float(
                actual.loc[actual_peak_idx] - base.loc[actual_peak_idx]
            ),
            "Shadow_Bias_AtActualPeak_MWH": float(
                actual.loc[actual_peak_idx] - shadow.loc[actual_peak_idx]
            ),
            "Base_DailyPeak_Timing_Error_Hours": float(
                (base_peak_dt - actual_peak_dt).total_seconds() / 3600.0
            ),
            "Shadow_DailyPeak_Timing_Error_Hours": float(
                (shadow_peak_dt - actual_peak_dt).total_seconds() / 3600.0
            ),
            "Base_DailyHourly_MAE_MWH": float((actual - base).abs().mean()),
            "Shadow_DailyHourly_MAE_MWH": float((actual - shadow).abs().mean()),
            "Daily_Peak_Correction_Applied_Flag": int(
                _as_num(
                    scored.get(
                        "Daily_Peak_Correction_Applied_Flag",
                        pd.Series(0, index=scored.index),
                    )
                )
                .fillna(0)
                .eq(1)
                .any()
            ),
            "MaxAbs_DailyPeakCorrection_MWH": float((shadow - base).abs().max()),
        }
        if "Replay_Origin_ID" in scored.columns:
            row["Replay_Origin_ID"] = scored["Replay_Origin_ID"].iloc[0]
        if "Replay_Origin_DT" in scored.columns:
            row["Replay_Origin_DT"] = scored["Replay_Origin_DT"].iloc[0]
        row["Delta_PeakAtActual_MAE_MWH"] = (
            row["Shadow_PeakAtActual_AbsError_MWH"]
            - row["Base_PeakAtActual_AbsError_MWH"]
        )
        row["Delta_ForecastPeak_MAE_MWH"] = (
            row["Shadow_ForecastPeak_AbsError_MWH"]
            - row["Base_ForecastPeak_AbsError_MWH"]
        )
        row["Delta_DailyHourly_MAE_MWH"] = (
            row["Shadow_DailyHourly_MAE_MWH"] - row["Base_DailyHourly_MAE_MWH"]
        )
        rows.append(row)
    return pd.DataFrame(rows)


def build_daily_peak_shadow_window_scorecard(
    df: pd.DataFrame, config: dict | None = None
) -> pd.DataFrame:
    """Summarize daily-peak shadow performance by the configured forecast-day window.

    This is a promotion diagnostic for the shadow daily-peak layer. It compares the
    shadow candidate to the production final forecast and shows whether benefit is
    concentrated inside the configured application window or leaking into weak
    exact-day horizons.
    """
    if df is None or df.empty or "Actual_MWH" not in df.columns:
        return pd.DataFrame()
    shadow_col = "Daily_Peak_Shadow_Adjusted_Forecast_MWH"
    if shadow_col not in df.columns:
        return pd.DataFrame()
    base_col = (
        "Daily_Peak_Base_Forecast_MWH"
        if "Daily_Peak_Base_Forecast_MWH" in df.columns
        else "Final_Backtest_Forecast_MWH"
    )
    if base_col not in df.columns:
        return pd.DataFrame()

    work = prep_backtest(df)
    if work.empty or base_col not in work.columns or shadow_col not in work.columns:
        return pd.DataFrame()

    cfg = _daily_peak_shadow_cfg(config)
    min_day = cfg.get("min_application_forecast_day")
    max_day = cfg.get("max_application_forecast_day")
    try:
        min_day_int = int(min_day) if min_day is not None else None
    except (TypeError, ValueError):
        min_day_int = None
    try:
        max_day_int = int(max_day) if max_day is not None else None
    except (TypeError, ValueError):
        max_day_int = None

    peak_hours = {
        int(h) for h in cfg.get("peak_hours", [14, 15, 16, 17, 18, 19, 20, 21])
    }
    day = _as_num(work.get("Forecast_Day", pd.Series(np.nan, index=work.index)))
    correction = (_as_num(work[shadow_col]) - _as_num(work[base_col])).abs()
    applied_mask = _as_num(
        work.get(
            "Daily_Peak_Correction_Applied_Flag", pd.Series(np.nan, index=work.index)
        )
    ).eq(1)
    if not applied_mask.any():
        applied_mask = correction.gt(1e-9)

    daily = _daily_peak_comparison_rows(
        work,
        base_col=base_col,
        shadow_col=shadow_col,
        peak_hours=peak_hours,
    )

    slices: list[tuple[str, str, pd.Series]] = [
        ("all_rows", "all", pd.Series(True, index=work.index))
    ]
    if min_day_int is not None and max_day_int is not None:
        slices.extend(
            [
                (
                    f"configured_window_days_{min_day_int}_to_{max_day_int}",
                    "configured_window",
                    day.between(min_day_int, max_day_int),
                ),
                (
                    f"below_configured_window_before_day_{min_day_int}",
                    "outside_window_low",
                    day.lt(min_day_int),
                ),
                (
                    f"above_configured_window_after_day_{max_day_int}",
                    "outside_window_high",
                    day.gt(max_day_int),
                ),
            ]
        )
    for forecast_day in sorted(day.dropna().astype(int).unique().tolist()):
        slices.append(
            (f"forecast_day_{forecast_day}", "exact_forecast_day", day.eq(forecast_day))
        )

    rows: list[dict[str, Any]] = []
    for slice_name, slice_type, mask in slices:
        mask = pd.Series(mask, index=work.index).fillna(False).astype(bool)
        subset = work.loc[mask].copy()
        if subset.empty:
            continue

        base_metrics = _metric_dict(
            subset["Actual_MWH"], subset[base_col], col=base_col
        )
        shadow_metrics = _metric_dict(
            subset["Actual_MWH"], subset[shadow_col], col=shadow_col
        )
        if not base_metrics or not shadow_metrics:
            continue

        daily_subset = pd.DataFrame()
        if not daily.empty and "Forecast_Day" in daily.columns:
            daily_day = _as_num(daily["Forecast_Day"])
            if slice_name == "all_rows":
                daily_subset = daily
            elif (
                slice_type == "configured_window"
                and min_day_int is not None
                and max_day_int is not None
            ):
                daily_subset = daily.loc[daily_day.between(min_day_int, max_day_int)]
            elif slice_type == "outside_window_low" and min_day_int is not None:
                daily_subset = daily.loc[daily_day.lt(min_day_int)]
            elif slice_type == "outside_window_high" and max_day_int is not None:
                daily_subset = daily.loc[daily_day.gt(max_day_int)]
            elif slice_type == "exact_forecast_day":
                try:
                    exact_day = int(slice_name.rsplit("_", 1)[-1])
                    daily_subset = daily.loc[daily_day.eq(exact_day)]
                except ValueError:
                    daily_subset = pd.DataFrame()

        row: dict[str, Any] = {
            "Slice": slice_name,
            "SliceType": slice_type,
            "BaseForecastColumn": base_col,
            "ShadowForecastColumn": shadow_col,
            "Configured_MinApplicationForecastDay": min_day_int,
            "Configured_MaxApplicationForecastDay": max_day_int,
            "N_HourlyRows": int(len(subset)),
            "Applied_HourlyRows": int(applied_mask.loc[subset.index].sum()),
            "MaxAbs_Correction_MWH": (
                float(correction.loc[subset.index].max())
                if correction.loc[subset.index].notna().any()
                else np.nan
            ),
            "Base_Hourly_MAE_MWH": base_metrics.get("MAE_MWH"),
            "Shadow_Hourly_MAE_MWH": shadow_metrics.get("MAE_MWH"),
            "Delta_Hourly_MAE_MWH": shadow_metrics.get("MAE_MWH", np.nan)
            - base_metrics.get("MAE_MWH", np.nan),
            "Base_Hourly_Bias_MWH": base_metrics.get("Bias_MWH"),
            "Shadow_Hourly_Bias_MWH": shadow_metrics.get("Bias_MWH"),
            "Delta_Hourly_Bias_MWH": shadow_metrics.get("Bias_MWH", np.nan)
            - base_metrics.get("Bias_MWH", np.nan),
            "Base_Hourly_P90_AbsError_MWH": base_metrics.get("P90_AbsError_MWH"),
            "Shadow_Hourly_P90_AbsError_MWH": shadow_metrics.get("P90_AbsError_MWH"),
            "Delta_Hourly_P90_AbsError_MWH": shadow_metrics.get(
                "P90_AbsError_MWH", np.nan
            )
            - base_metrics.get("P90_AbsError_MWH", np.nan),
        }
        if not daily_subset.empty:
            row.update(
                {
                    "N_DailyPeakDays": int(len(daily_subset)),
                    "Applied_DailyPeakDays": int(
                        _as_num(daily_subset["Daily_Peak_Correction_Applied_Flag"])
                        .fillna(0)
                        .eq(1)
                        .sum()
                    ),
                    "Base_PeakAtActual_MAE_MWH": float(
                        daily_subset["Base_PeakAtActual_AbsError_MWH"].mean()
                    ),
                    "Shadow_PeakAtActual_MAE_MWH": float(
                        daily_subset["Shadow_PeakAtActual_AbsError_MWH"].mean()
                    ),
                    "Delta_PeakAtActual_MAE_MWH": float(
                        daily_subset["Delta_PeakAtActual_MAE_MWH"].mean()
                    ),
                    "Base_ForecastPeak_MAE_MWH": float(
                        daily_subset["Base_ForecastPeak_AbsError_MWH"].mean()
                    ),
                    "Shadow_ForecastPeak_MAE_MWH": float(
                        daily_subset["Shadow_ForecastPeak_AbsError_MWH"].mean()
                    ),
                    "Delta_ForecastPeak_MAE_MWH": float(
                        daily_subset["Delta_ForecastPeak_MAE_MWH"].mean()
                    ),
                    "Base_Bias_AtActualPeak_MWH": float(
                        daily_subset["Base_Bias_AtActualPeak_MWH"].mean()
                    ),
                    "Shadow_Bias_AtActualPeak_MWH": float(
                        daily_subset["Shadow_Bias_AtActualPeak_MWH"].mean()
                    ),
                    "Delta_Bias_AtActualPeak_MWH": float(
                        daily_subset["Shadow_Bias_AtActualPeak_MWH"].mean()
                        - daily_subset["Base_Bias_AtActualPeak_MWH"].mean()
                    ),
                    "Base_DailyPeak_Timing_MAE_Hours": float(
                        daily_subset["Base_DailyPeak_Timing_Error_Hours"].abs().mean()
                    ),
                    "Shadow_DailyPeak_Timing_MAE_Hours": float(
                        daily_subset["Shadow_DailyPeak_Timing_Error_Hours"].abs().mean()
                    ),
                    "Delta_DailyPeak_Timing_MAE_Hours": float(
                        daily_subset["Shadow_DailyPeak_Timing_Error_Hours"].abs().mean()
                        - daily_subset["Base_DailyPeak_Timing_Error_Hours"].abs().mean()
                    ),
                    "Improved_PeakAtActual_Days": int(
                        daily_subset["Delta_PeakAtActual_MAE_MWH"].lt(0.0).sum()
                    ),
                    "Worsened_PeakAtActual_Days": int(
                        daily_subset["Delta_PeakAtActual_MAE_MWH"].gt(0.0).sum()
                    ),
                }
            )
        rows.append(row)

    out = pd.DataFrame(rows)
    if out.empty:
        return out
    return out.sort_values(["SliceType", "Slice"], kind="stable").reset_index(drop=True)


def build_top_error_tables(
    df: pd.DataFrame,
    n: int = 100,
    forecast_col: str = "Raw_Forecast_MWH",
    stage: str | None = None,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    if df is None or df.empty or forecast_col not in df.columns:
        return pd.DataFrame(), pd.DataFrame()
    work = df.copy()
    work["Stage"] = stage or RAW_STAGE
    work["Stage_Forecast_MWH"] = _as_num(work[forecast_col])
    work["Stage_Residual_MWH"] = (
        _as_num(work["Actual_MWH"]) - work["Stage_Forecast_MWH"]
    )
    work["Stage_AbsError_MWH"] = work["Stage_Residual_MWH"].abs()
    work["Stage_APE"] = np.where(
        _as_num(work["Actual_MWH"]).abs() > 1e-9,
        work["Stage_AbsError_MWH"] / _as_num(work["Actual_MWH"]).abs() * 100.0,
        np.nan,
    )
    keep = [
        "Stage",
        "ForecastColumn",
        "DT",
        "Date",
        "Forecast_Lead_Hour",
        "Forecast_Day",
        "Season",
        "Month",
        "Hour",
        "HourGroup",
        "Actual_MWH",
        "Stage_Forecast_MWH",
        "Raw_Forecast_MWH",
        "Residual_Calibrated_Forecast_MWH",
        "Warm_Ramp_Adjusted_Forecast_MWH",
        "Cloud_Solar_Adjusted_Forecast_MWH",
        "Peak_Risk_Adjusted_Forecast_MWH",
        "Recent_Corrected_Forecast_MWH",
        "XGB_Pred_MWH",
        "LGB_Pred_MWH",
        "CatBoost_Pred_MWH",
        "Prophet_Pred_MWH",
        "Stage_Residual_MWH",
        "Stage_AbsError_MWH",
        "Stage_APE",
        "Temperature",
        "Temperature_DailyMax",
        "DailyMaxTempBucket",
        "CloudCover_Norm",
        "CloudCoverBucket",
        "BTM_Solar_Proxy_MW",
        "BTMSolarBucket",
        "SolarLossBucket",
        "BTM_Solar_Loss_From_ClearSky_MW",
        "Cloud_Solar_Shape_Cal_MWH",
        "Cloud_Solar_Shape_Raw_Cal_MWH",
        "CloudSolarEventClass",
        "CloudSolarEventMultiplier",
        "CloudSolarBaseBucket",
        "Humidity_Norm",
        "WindSpeed_Mph",
        "WindDirection_Deg",
        "WindDirection_Available_Flag",
        "Westerly_Flow_Mph",
        "Westerly_Flow_Flag",
        "WindRamp_1Hr_Mph",
        "WindRamp_3Hr_Mph",
        "WindRamp_Next1Hr_Mph",
        "WindRamp_Next3Hr_Mph",
        "WesterlyFlow_Ramp_1Hr_Mph",
        "WesterlyFlow_Ramp_3Hr_Mph",
        "WesterlyFlow_Next1Hr_Ramp_Mph",
        "WesterlyFlow_Next3Hr_Ramp_Mph",
        "Temperature_Drop_From_DailyMax_F",
        "TempDrop_1Hr_F",
        "TempDrop_2Hr_F",
        "TempDrop_3Hr_F",
        "TempDrop_Next1Hr_F",
        "TempDrop_Next2Hr_F",
        "TempDrop_Next3Hr_F",
        "IsPostPeakEvening18to23",
        "ClearHotEvening_Flag",
        "ClearVeryHotEvening_Flag",
        "ClearHotEvening_x_TempDropFromDailyMax",
        "ClearHotEvening_x_ForecastDropNext3Hr",
        "ClearHotEvening_x_WesterlyFlow",
        "ClearHotEvening_x_WesterlyFlowRamp",
        "DeltaBreeze_Westerly_Flow_Flag",
        "DeltaBreeze_EveningWindRamp_Flag",
        "DeltaBreeze_Cooling_Flag",
        "DeltaBreeze_Cooling_Signal",
        "DeltaBreeze_CoolingNoDirection_Signal",
        "DeltaBreeze_ClearHotEvening_Signal",
        "Load_Decay_1Hr_MWH",
        "Load_Decay_2Hr_MWH",
        "Lag1_Minus_SameHourYesterday_MWH",
        "Lag1_Minus_SameHour7DayMean_MWH",
        "PostPeak_LoadDecay_1Hr_MWH",
        "PostPeak_LoadDecay_2Hr_MWH",
        "PostPeak_LoadDecay_VsSameHourYesterday_MWH",
        "PostPeak_LoadDecay_VsSameHour7DayMean_MWH",
        "ClearHotEvening_LoadDecay_Vs7Day_MWH",
        "DeltaBreeze_PostPeak_LoadDecay_Signal",
        "PrecipIn",
        "IsWeekend",
        "IsHoliday",
    ]
    work["ForecastColumn"] = forecast_col
    keep = [c for c in keep if c in work.columns]
    under = (
        work[work["Stage_Residual_MWH"] > 0]
        .sort_values("Stage_Residual_MWH", ascending=False)[keep]
        .head(int(n))
        .copy()
    )
    over = (
        work[work["Stage_Residual_MWH"] < 0]
        .assign(Stage_Overforecast_MWH=lambda x: -x["Stage_Residual_MWH"])
        .sort_values("Stage_Overforecast_MWH", ascending=False)
    )
    keep_over = keep + (
        ["Stage_Overforecast_MWH"] if "Stage_Overforecast_MWH" in over.columns else []
    )
    over = over[[c for c in keep_over if c in over.columns]].head(int(n)).copy()
    return under.reset_index(drop=True), over.reset_index(drop=True)


def build_top_error_tables_by_stage(
    df: pd.DataFrame, n: int = 100
) -> tuple[pd.DataFrame, pd.DataFrame]:
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
        (
            pd.concat(under_frames, ignore_index=True, sort=False)
            if under_frames
            else pd.DataFrame()
        ),
        (
            pd.concat(over_frames, ignore_index=True, sort=False)
            if over_frames
            else pd.DataFrame()
        ),
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
            tmp.insert(
                1,
                "CalibrationLevel",
                level.get("name", "+".join(level.get("keys", []))),
            )
            tmp.insert(2, "Keys", "+".join(level.get("keys", [])))
            frames.append(tmp)
    if not frames:
        return pd.DataFrame()
    return pd.concat(frames, ignore_index=True, sort=False)


def build_calibration_debug_table(
    calibration_lookup_bundle: dict | None,
    heat_peak_lookup: pd.DataFrame | None,
    warm_ramp_lookup: dict | None = None,
) -> pd.DataFrame:
    frames = [_flatten_lookup_table(calibration_lookup_bundle, "residual_calibration")]
    if (
        heat_peak_lookup is not None
        and isinstance(heat_peak_lookup, pd.DataFrame)
        and not heat_peak_lookup.empty
    ):
        h = heat_peak_lookup.copy()
        h.insert(0, "LookupSource", "heat_peak_calibration")
        h.insert(1, "CalibrationLevel", "heat_peak_hour_maxtemp")
        h.insert(2, "Keys", "Hour+DailyMaxTempBin")
        frames.append(h)
    if warm_ramp_lookup:
        frames.append(_flatten_lookup_table(warm_ramp_lookup, "warm_ramp_calibration"))
    frames = [f for f in frames if f is not None and not f.empty]
    return (
        pd.concat(frames, ignore_index=True, sort=False) if frames else pd.DataFrame()
    )


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
    return (
        pd.concat(frames, ignore_index=True, sort=False) if frames else pd.DataFrame()
    )


def feature_importance_table(
    model: Any, features: list[str], model_name: str
) -> pd.DataFrame:
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
    out = pd.DataFrame(
        {
            "Model": model_name,
            "Feature": list(features)[:n],
            "Importance": values[:n],
        }
    )
    total = out["Importance"].sum()
    out["Importance_Pct"] = out["Importance"] / total * 100.0 if total > 0 else np.nan
    return out.sort_values("Importance", ascending=False).reset_index(drop=True)


def prophet_regressor_table(
    prophet_model: Any | None, prophet_features: list[str] | None
) -> pd.DataFrame:
    if prophet_model is None or not prophet_features:
        return pd.DataFrame()
    extra = getattr(prophet_model, "extra_regressors", {}) or {}
    rows = []
    for feature in prophet_features:
        meta = extra.get(feature, {}) if isinstance(extra, dict) else {}
        rows.append(
            {
                "Model": "prophet",
                "Feature": feature,
                "PriorScale": (
                    meta.get("prior_scale") if isinstance(meta, dict) else np.nan
                ),
                "Standardize": (
                    meta.get("standardize") if isinstance(meta, dict) else np.nan
                ),
                "Mode": meta.get("mode") if isinstance(meta, dict) else np.nan,
            }
        )
    return pd.DataFrame(rows)


def _component_metric_rows(
    df: pd.DataFrame, component_cols: dict[str, str]
) -> list[dict[str, Any]]:
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
        out["Skill_vs_Raw_PCT"] = np.where(
            float(raw_mae.iloc[0]) > 0,
            (float(raw_mae.iloc[0]) - out["MAE_MWH"]) / float(raw_mae.iloc[0]) * 100.0,
            np.nan,
        )
    if not raw_bias.empty:
        out["Bias_Abs_Improvement_vs_Raw_MWH"] = (
            abs(float(raw_bias.iloc[0])) - out["Bias_MWH"].abs()
        )
    if not final_mae.empty:
        out["MAE_Delta_vs_Final_MWH"] = out["MAE_MWH"] - float(final_mae.iloc[0])
    return out.sort_values("MAE_MWH").reset_index(drop=True)


def _cloud_solar_gate_mask(df: pd.DataFrame) -> pd.Series:
    hour = _as_num(df.get("Hour", pd.Series(np.nan, index=df.index)))
    cloud = _as_num(df.get("CloudCover_Norm", pd.Series(np.nan, index=df.index)))
    if cloud.notna().any() and cloud.max(skipna=True) > 1.5:
        cloud = cloud / 100.0
    loss_candidates = [
        _as_num(df[col])
        for col in ["BTM_Solar_Loss_From_ClearSky_MW", "Midday_Overcast_Solar_Loss_MW"]
        if col in df.columns
    ]
    if loss_candidates:
        solar_loss = pd.concat(loss_candidates, axis=1).max(axis=1)
        solar_gate = solar_loss.ge(1.25)
    else:
        solar_gate = pd.Series(True, index=df.index)
    return (hour.between(10, 16) & cloud.ge(0.60) & solar_gate).fillna(False)


def _shadow_promotion_slice_gate(
    df: pd.DataFrame,
    final_col: str,
    candidate_col: str,
    mask: pd.Series,
    *,
    require_improvement: bool,
) -> dict[str, Any]:
    mask = pd.Series(mask, index=df.index).fillna(False).astype(bool)
    subset = df.loc[mask]
    if subset.empty:
        return {
            "N": 0,
            "Final_MAE_MWH": np.nan,
            "Candidate_MAE_MWH": np.nan,
            "MAE_Delta_vs_Final_MWH": np.nan,
            "Pass": True,
        }
    final = _metric_dict(subset["Actual_MWH"], subset[final_col], col=final_col)
    candidate = _metric_dict(
        subset["Actual_MWH"], subset[candidate_col], col=candidate_col
    )
    if not final or not candidate:
        return {
            "N": 0,
            "Final_MAE_MWH": np.nan,
            "Candidate_MAE_MWH": np.nan,
            "MAE_Delta_vs_Final_MWH": np.nan,
            "Pass": True,
        }
    final_mae = float(final.get("MAE_MWH", np.nan))
    candidate_mae = float(candidate.get("MAE_MWH", np.nan))
    delta = candidate_mae - final_mae
    tolerance = 1e-9
    passes = delta < -tolerance if require_improvement else delta <= tolerance
    return {
        "N": int(candidate.get("N", 0) or 0),
        "Final_MAE_MWH": final_mae,
        "Candidate_MAE_MWH": candidate_mae,
        "MAE_Delta_vs_Final_MWH": delta,
        "Pass": bool(np.isfinite(delta) and passes),
    }


def build_shadow_stage_promotion_audit(df: pd.DataFrame) -> pd.DataFrame:
    """Compare shadow candidates against final and gate them on bias/slice safety."""
    if df is None or df.empty or "Actual_MWH" not in df.columns:
        return pd.DataFrame()
    work = prep_backtest(df) if "DT" in df.columns else df.copy()
    stages = _available_stage_columns(work)
    final_col = (
        stages.get(FINAL_STAGE)
        or stages.get("recent_corrected_simulation")
        or stages.get(RAW_STAGE)
    )
    if not final_col:
        return pd.DataFrame()
    final = _metric_dict(
        work["Actual_MWH"], work[final_col], label=FINAL_STAGE, col=final_col
    )
    if not final:
        return pd.DataFrame()

    rows = []
    final_mae = float(final.get("MAE_MWH", np.nan))
    final_bias = float(final.get("Bias_MWH", np.nan))
    hour = _as_num(work.get("Hour", pd.Series(np.nan, index=work.index)))
    forecast_day = _as_num(
        work.get("Forecast_Day", pd.Series(np.nan, index=work.index))
    )
    daily_max = _as_num(
        work.get(
            "Temperature_DailyMax",
            work.get("Temperature", pd.Series(np.nan, index=work.index)),
        )
    )
    slice_masks = {
        "HotPeak": (hour.between(16, 20) & daily_max.ge(90.0)).fillna(False),
        "Peak14to18": hour.isin(PEAK_WINDOW_14_18_HOURS).fillna(False),
        "Day1": forecast_day.eq(1).fillna(False),
        "Days2to7": forecast_day.between(2, 7).fillna(False),
        "CloudSolar": _cloud_solar_gate_mask(work),
    }
    for stage, col in stages.items():
        if stage not in _SHADOW_STAGE_NAMES or col not in work.columns:
            continue
        candidate = _metric_dict(work["Actual_MWH"], work[col], label=stage, col=col)
        if not candidate:
            continue
        candidate_mae = float(candidate.get("MAE_MWH", np.nan))
        candidate_bias = float(candidate.get("Bias_MWH", np.nan))
        improvement = final_mae - candidate_mae
        seasonal_abs_bias = abs(candidate_bias)
        seasonal_bias_pass = bool(
            np.isfinite(seasonal_abs_bias) and seasonal_abs_bias <= 0.75
        )
        hot_peak_gate = _shadow_promotion_slice_gate(
            work, final_col, col, slice_masks["HotPeak"], require_improvement=False
        )
        peak_window_gate = _shadow_promotion_slice_gate(
            work, final_col, col, slice_masks["Peak14to18"], require_improvement=True
        )
        day1_gate = _shadow_promotion_slice_gate(
            work, final_col, col, slice_masks["Day1"], require_improvement=False
        )
        days2to7_gate = _shadow_promotion_slice_gate(
            work, final_col, col, slice_masks["Days2to7"], require_improvement=False
        )
        cloud_solar_gate = _shadow_promotion_slice_gate(
            work, final_col, col, slice_masks["CloudSolar"], require_improvement=False
        )
        gate_failures = []
        if not seasonal_bias_pass:
            gate_failures.append("Seasonal_Abs_Bias")
        for name, gate in [
            ("HotPeak_MAE", hot_peak_gate),
            ("Peak14to18_MAE", peak_window_gate),
            ("Day1_NoRegression", day1_gate),
            ("Days2to7_NoRegression", days2to7_gate),
            ("CloudSolar_NoRegression", cloud_solar_gate),
        ]:
            if not bool(gate["Pass"]):
                gate_failures.append(name)
        promotion_gate_pass = not gate_failures
        beats_final = bool(np.isfinite(improvement) and improvement > 1e-9)
        rows.append(
            {
                "Stage": stage,
                "ForecastColumn": col,
                "N": candidate.get("N"),
                "Candidate_MAE_MWH": candidate.get("MAE_MWH"),
                "Final_MAE_MWH": final.get("MAE_MWH"),
                "Candidate_MAE_Improvement_vs_Final_MWH": improvement,
                "Candidate_Bias_MWH": candidate.get("Bias_MWH"),
                "Final_Bias_MWH": final_bias,
                "Candidate_MAPE_PCT": candidate.get("MAPE_PCT"),
                "Final_MAPE_PCT": final.get("MAPE_PCT"),
                "Candidate_Underforecast_Rate_PCT": candidate.get(
                    "Underforecast_Rate_PCT"
                ),
                "Final_Underforecast_Rate_PCT": final.get("Underforecast_Rate_PCT"),
                "Seasonal_Abs_Bias_Gate_MWH": 0.75,
                "Candidate_Seasonal_Abs_Bias_MWH": seasonal_abs_bias,
                "Seasonal_Abs_Bias_Gate_Pass": seasonal_bias_pass,
                "HotPeak_N": hot_peak_gate["N"],
                "HotPeak_Final_MAE_MWH": hot_peak_gate["Final_MAE_MWH"],
                "HotPeak_Candidate_MAE_MWH": hot_peak_gate["Candidate_MAE_MWH"],
                "HotPeak_MAE_Delta_vs_Final_MWH": hot_peak_gate[
                    "MAE_Delta_vs_Final_MWH"
                ],
                "HotPeak_MAE_Gate_Pass": hot_peak_gate["Pass"],
                "Peak14to18_N": peak_window_gate["N"],
                "Peak14to18_Final_MAE_MWH": peak_window_gate["Final_MAE_MWH"],
                "Peak14to18_Candidate_MAE_MWH": peak_window_gate["Candidate_MAE_MWH"],
                "Peak14to18_MAE_Delta_vs_Final_MWH": peak_window_gate[
                    "MAE_Delta_vs_Final_MWH"
                ],
                "Peak14to18_MAE_Gate_Pass": peak_window_gate["Pass"],
                "Day1_N": day1_gate["N"],
                "Day1_MAE_Delta_vs_Final_MWH": day1_gate["MAE_Delta_vs_Final_MWH"],
                "Day1_NoRegression_Gate_Pass": day1_gate["Pass"],
                "Days2to7_N": days2to7_gate["N"],
                "Days2to7_MAE_Delta_vs_Final_MWH": days2to7_gate[
                    "MAE_Delta_vs_Final_MWH"
                ],
                "Days2to7_NoRegression_Gate_Pass": days2to7_gate["Pass"],
                "CloudSolar_N": cloud_solar_gate["N"],
                "CloudSolar_MAE_Delta_vs_Final_MWH": cloud_solar_gate[
                    "MAE_Delta_vs_Final_MWH"
                ],
                "CloudSolar_NoRegression_Gate_Pass": cloud_solar_gate["Pass"],
                "Promotion_Gate_Pass": promotion_gate_pass,
                "Promotion_Gate_Failures": (
                    ";".join(gate_failures) if gate_failures else "none"
                ),
                "Meets_Promotion_Rule": bool(beats_final and promotion_gate_pass),
                "Beats_Final": beats_final,
            }
        )
    out = pd.DataFrame(rows)
    if out.empty:
        return out
    return out.sort_values(
        ["Meets_Promotion_Rule", "Candidate_MAE_Improvement_vs_Final_MWH"],
        ascending=[False, False],
    ).reset_index(drop=True)


def build_hot_peak_shadow_candidate_scorecard(
    df: pd.DataFrame,
    group_cols: list[str] | None = None,
    min_count: int = 1,
) -> pd.DataFrame:
    """Compare shadow candidates against final inside hot-peak promotion slices."""
    if df is None or df.empty or "Actual_MWH" not in df.columns:
        return pd.DataFrame()

    work = _add_readable_bins(df.copy())
    if "Hour" not in work.columns and "DT" in work.columns:
        work["Hour"] = _local_datetime_series(work["DT"]).dt.hour
    if "Month" not in work.columns and "DT" in work.columns:
        work["Month"] = _local_datetime_series(work["DT"]).dt.month
    group_cols = group_cols or [
        "Replay_Horizon_Bucket",
        "Month",
        "DailyMaxTempBucket",
        "CloudCoverBucket",
    ]
    available_group_cols = [c for c in group_cols if c in work.columns]
    if not available_group_cols:
        return pd.DataFrame()

    hour = _as_num(work.get("Hour", pd.Series(np.nan, index=work.index)))
    daily_max = _as_num(
        work.get("Temperature_DailyMax", pd.Series(np.nan, index=work.index))
    )
    hot_peak = hour.between(16, 20) & daily_max.ge(90.0)
    work = work.loc[hot_peak.fillna(False)].copy()
    if work.empty:
        return pd.DataFrame()

    stages = _available_stage_columns(work)
    final_col = (
        stages.get(FINAL_STAGE)
        or stages.get("recent_corrected_simulation")
        or stages.get(RAW_STAGE)
    )
    if not final_col or final_col not in work.columns:
        return pd.DataFrame()

    candidate_stages = {
        stage: col
        for stage, col in stages.items()
        if stage in _SHADOW_STAGE_NAMES and col in work.columns
    }
    if not candidate_stages:
        return pd.DataFrame()

    rows: list[dict[str, Any]] = []
    for keys, group in work.groupby(available_group_cols, dropna=False):
        if not isinstance(keys, tuple):
            keys = (keys,)
        if len(group) < int(min_count):
            continue
        final = _metric_dict(
            group["Actual_MWH"], group[final_col], label=FINAL_STAGE, col=final_col
        )
        if not final:
            continue
        final_mae = float(final.get("MAE_MWH", np.nan))
        final_bias = float(final.get("Bias_MWH", np.nan))
        for stage, col in candidate_stages.items():
            candidate = _metric_dict(
                group["Actual_MWH"], group[col], label=stage, col=col
            )
            if not candidate:
                continue
            candidate_mae = float(candidate.get("MAE_MWH", np.nan))
            improvement = final_mae - candidate_mae
            row = {col_name: key for col_name, key in zip(available_group_cols, keys)}
            row.update(
                {
                    "Stage": stage,
                    "ForecastColumn": col,
                    "N": candidate.get("N"),
                    "Candidate_MAE_MWH": candidate.get("MAE_MWH"),
                    "Final_MAE_MWH": final.get("MAE_MWH"),
                    "Candidate_MAE_Improvement_vs_Final_MWH": improvement,
                    "Candidate_Bias_MWH": candidate.get("Bias_MWH"),
                    "Final_Bias_MWH": final_bias,
                    "Candidate_Bias_Abs_Improvement_vs_Final_MWH": abs(final_bias)
                    - abs(float(candidate.get("Bias_MWH", np.nan))),
                    "Candidate_MAPE_PCT": candidate.get("MAPE_PCT"),
                    "Final_MAPE_PCT": final.get("MAPE_PCT"),
                    "Candidate_Underforecast_Rate_PCT": candidate.get(
                        "Underforecast_Rate_PCT"
                    ),
                    "Final_Underforecast_Rate_PCT": final.get("Underforecast_Rate_PCT"),
                    "Candidate_P90_AbsError_MWH": candidate.get("P90_AbsError_MWH"),
                    "Final_P90_AbsError_MWH": final.get("P90_AbsError_MWH"),
                    "Beats_Final": bool(
                        np.isfinite(improvement) and improvement > 1e-9
                    ),
                    "Promote_Slice_Candidate": bool(
                        int(candidate.get("N", 0) or 0) >= int(min_count)
                        and np.isfinite(improvement)
                        and improvement > 1e-9
                    ),
                }
            )
            rows.append(row)

    out = pd.DataFrame(rows)
    if out.empty:
        return out
    return out.sort_values(
        ["Promote_Slice_Candidate", "Candidate_MAE_Improvement_vs_Final_MWH", "N"],
        ascending=[False, False, False],
    ).reset_index(drop=True)


def _dailymax_ramp_1day_for_scorecard(
    work: pd.DataFrame, daily_max: pd.Series
) -> pd.Series:
    if "DailyMaxTemp_Ramp_1Day" in work.columns:
        ramp = _as_num(work["DailyMaxTemp_Ramp_1Day"])
        if ramp.notna().any():
            return ramp
    if "PriorDay_DailyMaxTemp" in work.columns:
        prior = _as_num(work["PriorDay_DailyMaxTemp"])
        ramp = daily_max - prior
        if ramp.notna().any():
            return ramp
    if "Date" in work.columns:
        date = work["Date"].astype(str)
    elif "DT" in work.columns:
        date = _local_datetime_series(work["DT"]).dt.date.astype(str)
    else:
        return pd.Series(np.nan, index=work.index)
    daily = (
        pd.DataFrame({"Date": date, "DailyMax": daily_max}, index=work.index)
        .groupby("Date", dropna=False)["DailyMax"]
        .max()
        .reset_index()
    )
    daily["_DateDT"] = pd.to_datetime(daily["Date"], errors="coerce")
    daily = daily.sort_values("_DateDT")
    daily["_Ramp"] = daily["DailyMax"].diff()
    return date.map(daily.set_index("Date")["_Ramp"]).reindex(work.index).astype(float)


def _consecutive_extreme_days100_for_scorecard(
    work: pd.DataFrame, daily_max: pd.Series
) -> pd.Series:
    if "ConsecutiveExtremeHotDays100" in work.columns:
        consecutive = _as_num(work["ConsecutiveExtremeHotDays100"])
        if consecutive.notna().any():
            return consecutive
    if "Date" in work.columns:
        date = work["Date"].astype(str)
    elif "DT" in work.columns:
        date = _local_datetime_series(work["DT"]).dt.date.astype(str)
    else:
        return pd.Series(np.nan, index=work.index)

    group_cols = ["Date"]
    source = pd.DataFrame({"Date": date, "DailyMax": daily_max}, index=work.index)
    if "Replay_Origin_ID" in work.columns:
        source["Replay_Origin_ID"] = work["Replay_Origin_ID"].astype(str)
        group_cols = ["Replay_Origin_ID", "Date"]

    daily = source.groupby(group_cols, dropna=False)["DailyMax"].max().reset_index()
    daily["_DateDT"] = pd.to_datetime(daily["Date"], errors="coerce")
    daily = daily.sort_values(
        (["Replay_Origin_ID"] if "Replay_Origin_ID" in daily.columns else [])
        + ["_DateDT"]
    )

    if "Replay_Origin_ID" in daily.columns:
        counts = []
        for _, group in daily.groupby("Replay_Origin_ID", dropna=False, sort=False):
            running = 0
            for value in pd.to_numeric(group["DailyMax"], errors="coerce"):
                running = (
                    running + 1 if pd.notna(value) and float(value) >= 100.0 else 0
                )
                counts.append(running)
        daily["_ConsecutiveExtremeHotDays100"] = counts
        key = pd.MultiIndex.from_frame(daily[["Replay_Origin_ID", "Date"]])
        values = pd.Series(daily["_ConsecutiveExtremeHotDays100"].to_numpy(), index=key)
        lookup_key = pd.MultiIndex.from_arrays(
            [source["Replay_Origin_ID"], source["Date"]]
        )
        return pd.Series(
            values.reindex(lookup_key).to_numpy(), index=work.index, dtype=float
        )

    running = 0
    counts = []
    for value in pd.to_numeric(daily["DailyMax"], errors="coerce"):
        running = running + 1 if pd.notna(value) and float(value) >= 100.0 else 0
        counts.append(running)
    daily["_ConsecutiveExtremeHotDays100"] = counts
    return (
        date.map(daily.set_index("Date")["_ConsecutiveExtremeHotDays100"])
        .reindex(work.index)
        .astype(float)
    )


def build_hot_ramp_peak_candidate_scorecard(
    df: pd.DataFrame,
    group_cols: list[str] | None = None,
    min_count: int = 1,
) -> pd.DataFrame:
    """Compare shadow candidates against final inside the hot-ramp peak stratum."""
    if df is None or df.empty or "Actual_MWH" not in df.columns:
        return pd.DataFrame()

    work = _add_readable_bins(df.copy())
    if "Hour" not in work.columns and "DT" in work.columns:
        work["Hour"] = _local_datetime_series(work["DT"]).dt.hour
    if "Month" not in work.columns and "DT" in work.columns:
        work["Month"] = _local_datetime_series(work["DT"]).dt.month

    hour = _as_num(work.get("Hour", pd.Series(np.nan, index=work.index)))
    daily_max = _as_num(
        work.get("Temperature_DailyMax", pd.Series(np.nan, index=work.index))
    )
    ramp = _dailymax_ramp_1day_for_scorecard(work, daily_max)
    hot_ramp_peak = hour.between(16, 20) & daily_max.ge(100.0) & ramp.ge(2.0)
    work = work.loc[hot_ramp_peak.fillna(False)].copy()
    if work.empty:
        return pd.DataFrame()

    out = build_hot_peak_shadow_candidate_scorecard(
        work,
        group_cols=group_cols
        or ["Replay_Horizon_Bucket", "Month", "DailyMaxTempBucket", "CloudCoverBucket"],
        min_count=min_count,
    )
    if out.empty:
        return out
    out.insert(0, "Target_Slice", "hot_ramp_peak_100f_ramp2_he16to20")
    return out


def build_heat_persistence_peak_candidate_scorecard(
    df: pd.DataFrame,
    group_cols: list[str] | None = None,
    min_count: int = 1,
) -> pd.DataFrame:
    """Compare shadow candidates against final inside the heat-persistence peak stratum."""
    if df is None or df.empty or "Actual_MWH" not in df.columns:
        return pd.DataFrame()

    work = _add_readable_bins(df.copy())
    if "Hour" not in work.columns and "DT" in work.columns:
        work["Hour"] = _local_datetime_series(work["DT"]).dt.hour
    if "Month" not in work.columns and "DT" in work.columns:
        work["Month"] = _local_datetime_series(work["DT"]).dt.month

    hour = _as_num(work.get("Hour", pd.Series(np.nan, index=work.index)))
    daily_max = _as_num(
        work.get("Temperature_DailyMax", pd.Series(np.nan, index=work.index))
    )
    consecutive = _consecutive_extreme_days100_for_scorecard(work, daily_max)
    heat_persistence_peak = (
        hour.between(16, 20) & daily_max.ge(100.0) & consecutive.ge(3.0)
    )
    work = work.loc[heat_persistence_peak.fillna(False)].copy()
    if work.empty:
        return pd.DataFrame()

    out = build_hot_peak_shadow_candidate_scorecard(
        work,
        group_cols=group_cols
        or ["Replay_Horizon_Bucket", "Month", "DailyMaxTempBucket", "CloudCoverBucket"],
        min_count=min_count,
    )
    if out.empty:
        return out
    out.insert(0, "Target_Slice", "heat_persistence_peak_100f_consec3_he16to20")
    return out


def _feature_mean(df: pd.DataFrame, col: str) -> float:
    if col not in df.columns or df.empty:
        return np.nan
    values = _as_num(df[col])
    return float(values.mean()) if values.notna().any() else np.nan


def build_delta_breeze_shape_metrics_by_stage(
    df: pd.DataFrame, min_count: int = 1
) -> pd.DataFrame:
    """Stage accuracy on explicit evening-cooling / Delta Breeze diagnostic slices."""
    if df is None or df.empty or "Actual_MWH" not in df.columns:
        return pd.DataFrame()
    work = df.copy()
    if "Hour" not in work.columns and "DT" in work.columns:
        work["Hour"] = _local_datetime_series(work["DT"]).dt.hour
    stages = _available_stage_columns(work)
    if not stages:
        return pd.DataFrame()

    hour = _as_num(work.get("Hour", pd.Series(np.nan, index=work.index)))
    daily_max = _as_num(
        work.get("Temperature_DailyMax", pd.Series(np.nan, index=work.index))
    )
    cloud = _as_num(work.get("CloudCover_Norm", pd.Series(np.nan, index=work.index)))
    wind_direction_available = (
        _as_num(
            work.get("WindDirection_Available_Flag", pd.Series(0.0, index=work.index))
        )
        .fillna(0.0)
        .gt(0)
    )
    westerly_flag = (
        _as_num(work.get("Westerly_Flow_Flag", pd.Series(0.0, index=work.index)))
        .fillna(0.0)
        .gt(0)
    )
    clear_hot = (
        _as_num(work.get("ClearHotEvening_Flag", pd.Series(0.0, index=work.index)))
        .fillna(0.0)
        .gt(0)
    )
    cooling_from_max = _as_num(
        work.get(
            "Temperature_Drop_From_DailyMax_F", pd.Series(np.nan, index=work.index)
        )
    ).ge(5.0)
    forecast_cooling = _as_num(
        work.get("TempDrop_Next3Hr_F", pd.Series(np.nan, index=work.index))
    ).ge(6.0)
    delta_cooling = (
        _as_num(work.get("DeltaBreeze_Cooling_Flag", pd.Series(0.0, index=work.index)))
        .fillna(0.0)
        .gt(0)
    )
    delta_westerly = (
        _as_num(
            work.get("DeltaBreeze_Westerly_Flow_Flag", pd.Series(0.0, index=work.index))
        )
        .fillna(0.0)
        .gt(0)
    )
    delta_wind_ramp = (
        _as_num(
            work.get(
                "DeltaBreeze_EveningWindRamp_Flag", pd.Series(0.0, index=work.index)
            )
        )
        .fillna(0.0)
        .gt(0)
    )
    post_peak = (
        _as_num(
            work.get("IsPostPeakEvening18to23", pd.Series(np.nan, index=work.index))
        )
        .fillna(0.0)
        .gt(0)
    )
    if not post_peak.any():
        post_peak = hour.between(18, 23)

    masks: list[tuple[str, pd.Series]] = [
        ("post_peak_evening_18_23", post_peak),
        ("clear_hot_evening_95plus", clear_hot),
        (
            "clear_hot_evening_wind_direction_available",
            clear_hot & wind_direction_available,
        ),
        ("clear_hot_evening_westerly", clear_hot & westerly_flag),
        ("clear_hot_evening_cooling_from_max_5f", clear_hot & cooling_from_max),
        ("clear_hot_evening_forecast_cooling_next3hr_6f", clear_hot & forecast_cooling),
        ("delta_breeze_westerly_cooling", delta_westerly & delta_cooling),
        ("delta_breeze_westerly_ramp", delta_westerly & delta_wind_ramp),
        ("delta_breeze_cooling_no_direction", clear_hot & delta_cooling),
        (
            "hot_clear_evening_proxy_without_direction",
            hour.between(18, 23) & daily_max.ge(95.0) & cloud.le(0.20),
        ),
    ]
    feature_cols = [
        "Temperature_Drop_From_DailyMax_F",
        "TempDrop_1Hr_F",
        "TempDrop_2Hr_F",
        "TempDrop_3Hr_F",
        "TempDrop_Next1Hr_F",
        "TempDrop_Next2Hr_F",
        "TempDrop_Next3Hr_F",
        "WindSpeed_Mph",
        "WindDirection_Available_Flag",
        "Westerly_Flow_Mph",
        "Westerly_Flow_Flag",
        "WindRamp_1Hr_Mph",
        "WindRamp_3Hr_Mph",
        "WesterlyFlow_Ramp_1Hr_Mph",
        "WesterlyFlow_Ramp_3Hr_Mph",
        "ClearHotEvening_Flag",
        "DeltaBreeze_Westerly_Flow_Flag",
        "DeltaBreeze_EveningWindRamp_Flag",
        "DeltaBreeze_Cooling_Flag",
        "PostPeak_LoadDecay_1Hr_MWH",
        "PostPeak_LoadDecay_2Hr_MWH",
        "PostPeak_LoadDecay_VsSameHourYesterday_MWH",
        "PostPeak_LoadDecay_VsSameHour7DayMean_MWH",
        "DeltaBreeze_PostPeak_LoadDecay_Signal",
    ]

    rows: list[dict[str, Any]] = []
    for slice_name, mask in masks:
        mask = pd.Series(mask, index=work.index).fillna(False).astype(bool)
        subset = work.loc[mask].copy()
        if len(subset) < int(min_count):
            continue
        for stage, forecast_col in stages.items():
            if stage.startswith("baseline_") or forecast_col not in subset.columns:
                continue
            metrics = _metric_dict(
                subset["Actual_MWH"],
                subset[forecast_col],
                label=stage,
                col=forecast_col,
            )
            if not metrics:
                continue
            row = {"Slice": slice_name}
            row.update(metrics)
            for feature_col in feature_cols:
                row[f"Mean_{feature_col}"] = _feature_mean(subset, feature_col)
            rows.append(row)
    out = pd.DataFrame(rows)
    if out.empty:
        return out
    sort_cols = [c for c in ["Slice", "MAE_MWH", "Stage"] if c in out.columns]
    return out.sort_values(
        sort_cols, ascending=[True, False, True][: len(sort_cols)]
    ).reset_index(drop=True)


def build_model_component_metrics(df: pd.DataFrame) -> pd.DataFrame:
    rows = _component_metric_rows(df, _available_stage_columns(df))
    return (
        pd.DataFrame(rows).sort_values("MAE_MWH").reset_index(drop=True)
        if rows
        else pd.DataFrame()
    )


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
    ("Daily_Peak_Shadow_Adjusted_Forecast_MWH", "+DailyPeakShadow"),
]


def build_stage_marginal_contributions(
    df: pd.DataFrame, slices: dict[str, pd.DataFrame] | None = None
) -> pd.DataFrame:
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
        present = [
            (col, lbl)
            for col, lbl in _STAGE_CHAIN_ORDER
            if col in frame.columns and _as_num(frame[col]).notna().any()
        ]
        prev_mae = None
        for col, lbl in present:
            resid = actual - _as_num(frame[col])
            mask = resid.notna()
            if not mask.any():
                continue
            mae = float(resid[mask].abs().mean())
            bias = float(resid[mask].mean())
            marginal = np.nan if prev_mae is None else mae - prev_mae
            rows.append(
                {
                    "Slice": slice_name,
                    "Stage": lbl,
                    "Column": col,
                    "N": int(mask.sum()),
                    "MAE_MWH": mae,
                    "Bias_MWH": bias,
                    "Marginal_dMAE_MWH": marginal,
                    "Worsens_Slice": bool(np.isfinite(marginal) and marginal > 0.0),
                }
            )
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
    if (
        df is None
        or df.empty
        or forecast_col not in df.columns
        or "Actual_MWH" not in df.columns
    ):
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
    row.update(
        {k: v for k, v in metrics.items() if k not in {"Stage", "ForecastColumn"}}
    )

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


def build_production_readiness_scorecard(
    recent_df: pd.DataFrame,
    replay_df: pd.DataFrame,
    config: dict | None = None,
) -> pd.DataFrame:
    """Official production scorecard: rolling-origin replay is primary, recent backtest is context."""
    if config is not None:
        recent_df = (
            drop_excluded_intervals(recent_df, config)
            if recent_df is not None
            else recent_df
        )
        replay_df = (
            drop_excluded_intervals(replay_df, config)
            if replay_df is not None
            else replay_df
        )

    recent_col = "Final_Backtest_Forecast_MWH"
    replay_col = "Final_Backtest_Forecast_MWH"
    rows: list[dict[str, Any]] = []
    rows.append(
        _scorecard_metric_row(
            recent_df,
            "Last 45 days",
            "Recent behavior",
            "recent_45_day_backtest",
            "Recent MAE <= 3.0 MWh",
            "mae<=3.0",
            recent_col,
        )
    )
    rows.append(
        _scorecard_metric_row(
            replay_df,
            "Seasonal rolling origins",
            "General robustness",
            "seasonal_rolling_origin_replay",
            "MAE <= 4.5, MAPE <= 3.5%, abs bias <= 0.75",
            "mae<=4.5;mape<=3.5;abs_bias<=0.75",
            replay_col,
        )
    )

    if replay_df is None or replay_df.empty:
        return pd.DataFrame(rows)
    day = _as_num(
        replay_df.get("Forecast_Day", pd.Series(np.nan, index=replay_df.index))
    )
    hour = _as_num(replay_df.get("Hour", pd.Series(np.nan, index=replay_df.index)))
    temp = _as_num(
        replay_df.get("Temperature_DailyMax", pd.Series(np.nan, index=replay_df.index))
    )
    cloud = _as_num(
        replay_df.get("CloudCover_Norm", pd.Series(np.nan, index=replay_df.index))
    )
    loss = _as_num(
        replay_df.get(
            "BTM_Solar_Loss_From_ClearSky_MW", pd.Series(np.nan, index=replay_df.index)
        )
    )
    season = replay_df.get("Season", pd.Series("", index=replay_df.index)).astype(str)
    rows.extend(
        [
            _scorecard_metric_row(
                replay_df[day.eq(1)],
                "Day 1 only",
                "Near-term operational forecast",
                "seasonal_rolling_origin_replay",
                "MAE <= 3.5 MWh",
                "mae<=3.5",
                replay_col,
            ),
            _scorecard_metric_row(
                replay_df[day.between(2, 3)],
                "Days 2-3",
                "Short weather forecast horizon",
                "seasonal_rolling_origin_replay",
                "MAE <= 5.0 MWh",
                "mae<=5.0",
                replay_col,
            ),
            _scorecard_metric_row(
                replay_df[day.between(4, 7)],
                "Days 4-7",
                "Weather uncertainty horizon",
                "seasonal_rolling_origin_replay",
                "MAE <= 5.0 MWh",
                "mae<=5.0",
                replay_col,
            ),
            _scorecard_metric_row(
                replay_df[hour.between(16, 20) & temp.ge(90.0)],
                "Hot peak days",
                "Operational risk",
                "seasonal_rolling_origin_replay",
                "MAE <= 6.0 MWh",
                "mae<=6.0",
                replay_col,
            ),
            _scorecard_metric_row(
                replay_df[hour.between(10, 16) & (cloud.ge(0.60) | loss.ge(1.25))],
                "Cloud/solar midday",
                "BTM solar/cloud risk",
                "seasonal_rolling_origin_replay",
                "MAE <= 7.0 MWh",
                "mae<=7.0",
                replay_col,
            ),
            _scorecard_metric_row(
                replay_df[
                    season.isin(["Spring", "Fall"])
                    & hour.between(12, 22)
                    & temp.between(75.0, 93.0)
                ],
                "Shoulder heat transition",
                "Spring/fall load-response risk",
                "seasonal_rolling_origin_replay",
                "MAE <= 7.0 MWh",
                "mae<=7.0",
                replay_col,
            ),
            _scorecard_metric_row(
                replay_df[hour.between(14, 18)],
                "Peak window hours 14-18",
                "Peak planning risk",
                "seasonal_rolling_origin_replay",
                "MAE <= 5.5 MWh",
                "mae<=5.5",
                replay_col,
            ),
        ]
    )
    return pd.DataFrame(rows)


def build_recent_profile_debug_table(profile: dict | None) -> pd.DataFrame:
    if not profile:
        return pd.DataFrame()
    rows = []
    scalar_keys = ["recent_mean", "last24_mean", "global_mean"]
    lookup_keys = [
        "same_hour_mean",
        "hourgroup_mean",
        "temp_hourgroup_mean",
        "cloud_hourgroup_mean",
        "solar_hourgroup_mean",
        "solar_loss_hourgroup_mean",
        "temp_cloud_hourgroup_mean",
    ]
    for key in scalar_keys:
        if key in profile:
            rows.append(
                {"Level": key, "Key": "all", "Correction_MWH": profile.get(key)}
            )
    for level in lookup_keys:
        for k, v in (profile.get(level, {}) or {}).items():
            rows.append({"Level": level, "Key": k, "Correction_MWH": v})
    for state_key in ["ar_residual", "origin_day_state"]:
        state = profile.get(state_key, {}) or {}
        if not isinstance(state, dict):
            continue
        for k, v in state.items():
            if isinstance(v, dict):
                continue
            rows.append({"Level": state_key, "Key": k, "Correction_MWH": v})
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
    band = (
        max(float(floor_mwh), abs(float(forecast)) * float(percent_band))
        if np.isfinite(forecast)
        else float(floor_mwh)
    )
    method = "percent_or_floor"
    if residual_band_lookup and residual_band_lookup.get("ordered_levels"):
        band = max(band, float(residual_band_lookup.get("global_band_mwh", floor_mwh)))
        for level in residual_band_lookup.get("ordered_levels", []):
            keys = level.get("keys", [])
            lookup = level.get("lookup")
            if (
                lookup is None
                or not isinstance(lookup, pd.DataFrame)
                or lookup.empty
                or not all(k in row.index for k in keys)
            ):
                continue
            mask = pd.Series(True, index=lookup.index)
            for k in keys:
                mask &= (
                    lookup[k]
                    .astype("object")
                    .fillna("__NA__")
                    .eq(row.get(k) if pd.notna(row.get(k)) else "__NA__")
                )
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


def build_band_coverage_by_stage(
    df: pd.DataFrame, residual_band_lookup: dict | None, config: dict | None = None
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    if df is None or df.empty or "Actual_MWH" not in df.columns:
        return pd.DataFrame(), pd.DataFrame(), pd.DataFrame(), pd.DataFrame()
    bands_cfg = (config or {}).get("bands", {}) if isinstance(config, dict) else {}
    percent_band = float(bands_cfg.get("default_percent_band", 0.08))
    floor_mwh = float(bands_cfg.get("band_floor_mwh", 5.0))
    band_scale = float(bands_cfg.get("band_scale", 1.0))
    stages = {
        k: v
        for k, v in _available_stage_columns(df).items()
        if k
        in [
            RAW_STAGE,
            "cloud_solar_adjusted",
            "peak_risk_adjusted",
            "recent_corrected_simulation",
            FINAL_STAGE,
            "prophet_benchmark",
        ]
    }
    rows = []
    for stage, col in stages.items():
        for idx, row in df.iterrows():
            actual = (
                float(row["Actual_MWH"]) if pd.notna(row.get("Actual_MWH")) else np.nan
            )
            forecast = (
                float(row[col])
                if col in row.index and pd.notna(row.get(col))
                else np.nan
            )
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
            rows.append(
                {
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
                }
            )
    detail = pd.DataFrame(rows)
    if detail.empty:
        return pd.DataFrame(), pd.DataFrame(), pd.DataFrame(), pd.DataFrame()

    def summarize(g: pd.DataFrame) -> pd.Series:
        return pd.Series(
            {
                "N": int(len(g)),
                "Coverage_PCT": float(g["Inside_Band"].mean() * 100.0),
                "Avg_Band_MWH": float(g["Band_MWH"].mean()),
                "Avg_AbsError_MWH": float(g["AbsError_MWH"].mean()),
                "P90_AbsError_MWH": float(g["AbsError_MWH"].quantile(0.90)),
                "Avg_Band_Miss_MWH": float(g["Band_Miss_MWH"].mean()),
                "Max_Band_Miss_MWH": float(g["Band_Miss_MWH"].max()),
            }
        )

    summary = (
        detail.groupby(["Stage", "ForecastColumn"], dropna=False)
        .apply(summarize, include_groups=False)
        .reset_index()
    )
    by_hour = (
        detail.groupby(["Stage", "Hour"], dropna=False)
        .apply(summarize, include_groups=False)
        .reset_index()
    )
    by_temp = (
        detail.groupby(["Stage", "DailyMaxTempBucket", "HourGroup"], dropna=False)
        .apply(summarize, include_groups=False)
        .reset_index()
    )
    return summary, by_hour, by_temp, detail


def _future_point_forecast_col(df: pd.DataFrame) -> str | None:
    for col in [
        "Forecast_Expected_MWH",
        "Forecast",
        "Final_Forecast_MWH",
        "Stage_Selected_Forecast_MWH",
        "Final_Backtest_Forecast_MWH",
    ]:
        if col in df.columns and _as_num(df[col]).notna().any():
            return col
    return None


def _optional_series(
    df: pd.DataFrame, col: str, default: float | str = np.nan
) -> pd.Series:
    if col in df.columns:
        return df[col]
    return pd.Series(default, index=df.index)


def build_future_peak_muting_audit(
    forecast_df: pd.DataFrame | None,
    backtest_df: pd.DataFrame | None = None,
    config: dict | None = None,
) -> pd.DataFrame:
    """Flag future hot-peak days where the point forecast lags hotter/stress scenarios.

    This is an operational forward-looking audit. Replay scorecards validate realized-weather
    accuracy, while this table checks whether the current P50 is being held down during hot
    weather-risk peaks and merely exporting the risk into the upper band.
    """
    if forecast_df is None or forecast_df.empty:
        return pd.DataFrame()

    cfg = ((config or {}).get("diagnostics", {}) or {}).get(
        "future_peak_muting_audit", {}
    ) or {}
    if not bool(cfg.get("enabled", True)):
        return pd.DataFrame()

    work = forecast_df.copy()
    if "DT" not in work.columns:
        return pd.DataFrame()

    point_col = _future_point_forecast_col(work)
    if point_col is None:
        return pd.DataFrame()

    point = _as_num(work[point_col])
    work = work.loc[point.notna()].copy()
    if work.empty:
        return pd.DataFrame()

    raw_dt = work["DT"].copy()
    dt = _local_datetime_series(work["DT"])
    work = work.loc[dt.notna()].copy()
    if work.empty:
        return pd.DataFrame()
    dt = _local_datetime_series(work["DT"])
    work["_DT_Label"] = raw_dt.loc[work.index].astype(str)
    work["_DT"] = dt
    work["_Date"] = work["_DT"].dt.date.astype(str)
    work["_Hour"] = (
        _as_num(work["Hour"]).fillna(work["_DT"].dt.hour).astype(int)
        if "Hour" in work.columns
        else work["_DT"].dt.hour.astype(int)
    )
    work["_PointForecast_MWH"] = _as_num(work[point_col])
    if "Forecast_Day" in work.columns:
        work["_Forecast_Day"] = _as_num(work["Forecast_Day"])
    else:
        first_day = work["_DT"].dt.normalize().min()
        work["_Forecast_Day"] = (work["_DT"].dt.normalize() - first_day).dt.days + 1

    if "Temperature_DailyMax" in work.columns:
        daily_max = _as_num(work["Temperature_DailyMax"])
    elif "Temperature" in work.columns:
        daily_max = _as_num(work["Temperature"])
    else:
        daily_max = pd.Series(np.nan, index=work.index)
    work["_DailyMaxTemp_F"] = daily_max
    work["_Temperature_F"] = _as_num(_optional_series(work, "Temperature"))

    peak_hours = [int(h) for h in cfg.get("peak_hours", PEAK_WINDOW_14_20_HOURS)]
    hot_min_daily_max = float(cfg.get("hot_min_daily_max_f", 95.0))
    scenario_gap_warn = float(cfg.get("scenario_gap_warn_mwh", 8.0))
    hotter_day_min_temp_increase = float(cfg.get("hotter_day_min_temp_increase_f", 1.0))
    hotter_day_peak_drop_warn = float(cfg.get("hotter_day_peak_drop_warn_mwh", 2.0))
    analog_min_daily_max = float(
        cfg.get("analog_min_daily_max_f", EXTREME_HEAT_MIN_DAILY_MAX_F)
    )

    scenario_cols = [
        col
        for col in [
            "WeatherScenario_warmer_P50_MWH",
            "WeatherScenario_hot_stress_5f_P50_MWH",
            "WeatherScenario_cloudier_solar_loss_P50_MWH",
            "WeatherScenario_severe_cloud_solar_loss_P50_MWH",
            "WeatherScenario_clearer_high_solar_P50_MWH",
            "WeatherScenario_cooler_P50_MWH",
        ]
        if col in work.columns
    ]
    if "WeatherScenario_Max_P50_MWH" in work.columns:
        work["_MaxScenario_P50_MWH"] = _as_num(work["WeatherScenario_Max_P50_MWH"])
    elif scenario_cols:
        work["_MaxScenario_P50_MWH"] = pd.concat(
            [_as_num(work[col]) for col in scenario_cols], axis=1
        ).max(axis=1, skipna=True)
    else:
        work["_MaxScenario_P50_MWH"] = np.nan
    work["_HotStress_P50_MWH"] = _as_num(
        _optional_series(work, "WeatherScenario_hot_stress_5f_P50_MWH")
    )
    work["_Warmer_P50_MWH"] = _as_num(
        _optional_series(work, "WeatherScenario_warmer_P50_MWH")
    )

    analog_count = 0
    analog_mean = np.nan
    analog_p90 = np.nan
    analog_max = np.nan
    if (
        backtest_df is not None
        and isinstance(backtest_df, pd.DataFrame)
        and not backtest_df.empty
    ):
        bt = prep_backtest(backtest_df)
        if not bt.empty and "Actual_MWH" in bt.columns:
            bt_hour = _as_num(bt.get("Hour", pd.Series(np.nan, index=bt.index))).astype(
                "Int64"
            )
            bt_temp = _as_num(
                bt.get("Temperature_DailyMax", pd.Series(np.nan, index=bt.index))
            )
            analog_values = _as_num(
                bt.loc[
                    bt_temp.ge(analog_min_daily_max) & bt_hour.isin(peak_hours),
                    "Actual_MWH",
                ]
            ).dropna()
            analog_count = int(len(analog_values))
            if analog_count:
                analog_mean = float(analog_values.mean())
                analog_p90 = float(analog_values.quantile(0.90))
                analog_max = float(analog_values.max())

    rows: list[dict[str, Any]] = []
    for date, g in work.sort_values("_DT").groupby("_Date", dropna=False):
        g = g.dropna(subset=["_PointForecast_MWH"])
        if g.empty:
            continue
        peak_idx = g["_PointForecast_MWH"].idxmax()
        peak = g.loc[peak_idx]
        peak_window = g.loc[g["_Hour"].isin(peak_hours)].copy()
        if peak_window.empty:
            peak_window = g

        max_gap_series = _as_num(peak_window["_MaxScenario_P50_MWH"]) - _as_num(
            peak_window["_PointForecast_MWH"]
        )
        hot_gap_series = _as_num(peak_window["_HotStress_P50_MWH"]) - _as_num(
            peak_window["_PointForecast_MWH"]
        )
        warmer_gap_series = _as_num(peak_window["_Warmer_P50_MWH"]) - _as_num(
            peak_window["_PointForecast_MWH"]
        )

        max_gap_idx = (
            max_gap_series.idxmax() if max_gap_series.notna().any() else peak_idx
        )
        hot_gap_idx = (
            hot_gap_series.idxmax() if hot_gap_series.notna().any() else peak_idx
        )
        max_gap_row = peak_window.loc[max_gap_idx]
        hot_gap_row = peak_window.loc[hot_gap_idx]

        daily_peak_hot_gap = (
            float(peak.get("_HotStress_P50_MWH", np.nan) - peak["_PointForecast_MWH"])
            if pd.notna(peak.get("_HotStress_P50_MWH", np.nan))
            else np.nan
        )
        daily_peak_warmer_gap = (
            float(peak.get("_Warmer_P50_MWH", np.nan) - peak["_PointForecast_MWH"])
            if pd.notna(peak.get("_Warmer_P50_MWH", np.nan))
            else np.nan
        )
        daily_peak_max_gap = (
            float(peak.get("_MaxScenario_P50_MWH", np.nan) - peak["_PointForecast_MWH"])
            if pd.notna(peak.get("_MaxScenario_P50_MWH", np.nan))
            else np.nan
        )

        rows.append(
            {
                "Date": date,
                "Forecast_Day": float(peak.get("_Forecast_Day", np.nan)),
                "PointForecastColumn": point_col,
                "Forecast_Peak_DT": peak.get("_DT_Label"),
                "Forecast_Peak_Hour": int(peak["_Hour"]),
                "Forecast_Peak_MWH": float(peak["_PointForecast_MWH"]),
                "Temperature_At_Forecast_Peak_F": (
                    float(peak.get("_Temperature_F", np.nan))
                    if pd.notna(peak.get("_Temperature_F", np.nan))
                    else np.nan
                ),
                "Temperature_DailyMax_F": (
                    float(peak_window["_DailyMaxTemp_F"].max(skipna=True))
                    if peak_window["_DailyMaxTemp_F"].notna().any()
                    else np.nan
                ),
                "Peak_In_Configured_Window": bool(
                    int(peak["_Hour"]) in set(peak_hours)
                ),
                "DailyPeak_WarmerScenario_Gap_MWH": daily_peak_warmer_gap,
                "DailyPeak_HotStressScenario_Gap_MWH": daily_peak_hot_gap,
                "DailyPeak_MaxScenario_Gap_MWH": daily_peak_max_gap,
                "MaxPeakWindowScenarioGap_DT": max_gap_row.get("_DT_Label"),
                "MaxPeakWindowScenarioGap_Hour": int(max_gap_row["_Hour"]),
                "MaxPeakWindowScenarioGap_MWH": (
                    float(max_gap_series.loc[max_gap_idx])
                    if max_gap_series.notna().any()
                    else np.nan
                ),
                "MaxPeakWindowScenarioGap_Point_MWH": float(
                    max_gap_row["_PointForecast_MWH"]
                ),
                "MaxPeakWindowScenarioGap_Scenario_MWH": (
                    float(max_gap_row["_MaxScenario_P50_MWH"])
                    if pd.notna(max_gap_row.get("_MaxScenario_P50_MWH", np.nan))
                    else np.nan
                ),
                "MaxPeakWindowHotStressGap_DT": hot_gap_row.get("_DT_Label"),
                "MaxPeakWindowHotStressGap_Hour": int(hot_gap_row["_Hour"]),
                "MaxPeakWindowHotStressGap_MWH": (
                    float(hot_gap_series.loc[hot_gap_idx])
                    if hot_gap_series.notna().any()
                    else np.nan
                ),
                "MaxPeakWindowWarmerGap_MWH": (
                    float(warmer_gap_series.max(skipna=True))
                    if warmer_gap_series.notna().any()
                    else np.nan
                ),
                "Forecast_High_MWH": (
                    float(peak.get("Forecast_High_MWH", np.nan))
                    if "Forecast_High_MWH" in peak.index
                    and pd.notna(peak.get("Forecast_High_MWH"))
                    else np.nan
                ),
                "Band_MWH": (
                    float(peak.get("Band", np.nan))
                    if "Band" in peak.index and pd.notna(peak.get("Band"))
                    else np.nan
                ),
                "Weather_Robustness_Hedge_MWH": (
                    float(peak.get("Weather_Robustness_Hedge_MWH", np.nan))
                    if "Weather_Robustness_Hedge_MWH" in peak.index
                    and pd.notna(peak.get("Weather_Robustness_Hedge_MWH"))
                    else np.nan
                ),
                "Peak_Risk_Cal_MWH": (
                    float(peak.get("Peak_Risk_Cal_MWH", np.nan))
                    if "Peak_Risk_Cal_MWH" in peak.index
                    and pd.notna(peak.get("Peak_Risk_Cal_MWH"))
                    else np.nan
                ),
                "Stage_Selector_Reason": peak.get("Stage_Selector_Reason", ""),
                "Production_Confidence_Label": peak.get(
                    "Production_Confidence_Label", ""
                ),
                "Production_Risk_Code": peak.get("Production_Risk_Code", ""),
                "Analog_MinDailyMax_F": analog_min_daily_max,
                "Analog_PeakHours": _hours_label(peak_hours),
                "Analog_Actual_Count": analog_count,
                "Analog_Actual_Mean_MWH": analog_mean,
                "Analog_Actual_P90_MWH": analog_p90,
                "Analog_Actual_Max_MWH": analog_max,
                "Forecast_Delta_vs_Analog_P90_MWH": (
                    float(peak["_PointForecast_MWH"] - analog_p90)
                    if np.isfinite(analog_p90)
                    else np.nan
                ),
                "Forecast_Delta_vs_Analog_Max_MWH": (
                    float(peak["_PointForecast_MWH"] - analog_max)
                    if np.isfinite(analog_max)
                    else np.nan
                ),
            }
        )

    out = pd.DataFrame(rows)
    if out.empty:
        return out

    out.sort_values(["Date"], inplace=True)
    out["Temperature_DailyMax_Delta_From_Previous_Day_F"] = _as_num(
        out["Temperature_DailyMax_F"]
    ).diff()
    out["Forecast_Peak_Delta_From_Previous_Day_MWH"] = _as_num(
        out["Forecast_Peak_MWH"]
    ).diff()
    muted_flags = []
    muted_reasons = []
    for _, row in out.iterrows():
        reasons = []
        hot_enough = (
            pd.notna(row.get("Temperature_DailyMax_F"))
            and float(row["Temperature_DailyMax_F"]) >= hot_min_daily_max
        )
        if hot_enough:
            if (
                pd.notna(row.get("DailyPeak_MaxScenario_Gap_MWH"))
                and float(row["DailyPeak_MaxScenario_Gap_MWH"]) >= scenario_gap_warn
            ):
                reasons.append("daily_peak_below_max_scenario")
            if (
                pd.notna(row.get("MaxPeakWindowScenarioGap_MWH"))
                and float(row["MaxPeakWindowScenarioGap_MWH"]) >= scenario_gap_warn
            ):
                reasons.append("peak_window_below_max_scenario")
            if (
                pd.notna(row.get("Temperature_DailyMax_Delta_From_Previous_Day_F"))
                and pd.notna(row.get("Forecast_Peak_Delta_From_Previous_Day_MWH"))
                and float(row["Temperature_DailyMax_Delta_From_Previous_Day_F"])
                >= hotter_day_min_temp_increase
                and float(row["Forecast_Peak_Delta_From_Previous_Day_MWH"])
                <= -hotter_day_peak_drop_warn
            ):
                reasons.append("hotter_day_lower_peak")
        muted_flags.append(bool(reasons))
        muted_reasons.append("+".join(reasons) if reasons else "none")
    out["Muted_Peak_Flag"] = muted_flags
    out["Muted_Peak_Reasons"] = muted_reasons
    return out.reset_index(drop=True)


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

    bt = apply_multisummer_heat_analog_shadow(prep_backtest(backtest_df), config=config)
    under, over = build_top_error_tables(
        bt, n=top_n, forecast_col="Raw_Forecast_MWH", stage=RAW_STAGE
    )
    under_stage, over_stage = build_top_error_tables_by_stage(bt, n=top_n)

    xgb_imp = feature_importance_table(xgb_model, features or [], "xgb")
    lgb_imp = feature_importance_table(lgb_model, features or [], "lgb")
    cat_imp = feature_importance_table(catboost_model, features or [], "catboost")
    feature_importance = (
        pd.concat([xgb_imp, lgb_imp, cat_imp], ignore_index=True, sort=False)
        if not xgb_imp.empty or not lgb_imp.empty or not cat_imp.empty
        else pd.DataFrame()
    )
    prophet_regs = prophet_regressor_table(prophet_model, prophet_features or [])
    try:
        from forecasting.forecast.event_shape_corrections import (
            cloud_solar_lookup_debug_table,
        )

        cloud_solar_debug = cloud_solar_lookup_debug_table(cloud_solar_shape_lookup)
    except Exception:
        cloud_solar_debug = pd.DataFrame()

    band_summary, band_by_hour, band_by_temp, band_detail = (
        build_band_coverage_by_stage(bt, residual_band_lookup, config=config)
    )

    bundle: dict[str, Any] = {
        "diagnostics_summary": metrics_summary(bt),
        "backtest_enriched": bt,
        # Legacy/raw tables preserved for compatibility.
        "backtest_metrics_by_segment": build_backtest_metrics_by_segment(
            bt, min_count=min_segment_count, forecast_col="Raw_Forecast_MWH"
        ),
        "error_by_hour": build_metrics_by_group(
            bt, ["Hour"], min_count=1, forecast_col="Raw_Forecast_MWH"
        ),
        "seasonal_error_by_hour": build_metrics_by_group(
            bt,
            ["Season", "Hour"],
            min_count=min_segment_count,
            forecast_col="Raw_Forecast_MWH",
        ),
        "seasonal_error_by_month_hour": build_metrics_by_group(
            bt,
            ["Season", "Month", "Hour"],
            min_count=min_segment_count,
            forecast_col="Raw_Forecast_MWH",
        ),
        "seasonal_error_by_max_temp_bin": build_metrics_by_group(
            bt,
            ["Season", "DailyMaxTempBucket", "HourGroup"],
            min_count=min_segment_count,
            forecast_col="Raw_Forecast_MWH",
        ),
        "seasonal_error_by_cloud_bin": build_metrics_by_group(
            bt,
            ["Season", "CloudCoverBucket", "HourGroup"],
            min_count=min_segment_count,
            forecast_col="Raw_Forecast_MWH",
        ),
        "seasonal_error_by_solar_bin": build_metrics_by_group(
            bt,
            ["Season", "BTMSolarBucket", "HourGroup"],
            min_count=min_segment_count,
            forecast_col="Raw_Forecast_MWH",
        ),
        "error_by_temp_cloud_hourgroup": build_metrics_by_group(
            bt,
            ["DailyMaxTempBucket", "CloudCoverBucket", "HourGroup"],
            min_count=min_segment_count,
            forecast_col="Raw_Forecast_MWH",
        ),
        "daily_peak_miss_table": build_daily_peak_miss_table(
            bt, forecast_col="Raw_Forecast_MWH", stage=RAW_STAGE
        ),
        "top_100_underforecast_hours": under,
        "top_100_overforecast_hours": over,
        # V12.4 stage-aware diagnostics.
        "model_component_metrics": build_model_component_metrics(bt),
        "forecast_stage_metrics": build_forecast_stage_metrics(bt),
        "shadow_stage_promotion_audit": build_shadow_stage_promotion_audit(bt),
        "hot_peak_shadow_candidate_scorecard": build_hot_peak_shadow_candidate_scorecard(
            bt, min_count=1
        ),
        "hot_ramp_peak_candidate_scorecard": build_hot_ramp_peak_candidate_scorecard(
            bt, min_count=1
        ),
        "heat_persistence_peak_candidate_scorecard": build_heat_persistence_peak_candidate_scorecard(
            bt, min_count=1
        ),
        "delta_breeze_shape_metrics_by_stage": build_delta_breeze_shape_metrics_by_stage(
            bt, min_count=min_segment_count
        ),
        "peak_window_bias_scorecard": build_peak_window_bias_scorecard(
            bt, forecast_col="Final_Backtest_Forecast_MWH", min_count=5
        ),
        "peak_window_expansion_scorecard": build_peak_window_expansion_scorecard(bt),
        "peak_window_14to20_metrics_by_stage": build_peak_window_14to20_metrics_by_stage(
            bt, min_count=1
        ),
        "extreme_heat_peak_scorecard": build_extreme_heat_peak_scorecard(bt),
        "extreme_heat_peak_metrics_by_stage": build_extreme_heat_peak_metrics_by_stage(
            bt, min_count=1
        ),
        "daily_peak_he18_20_miss_by_stage": build_daily_peak_window_miss_by_stage(bt),
        "heat_analog_shadow_metrics": build_heat_analog_shadow_metrics(
            bt, config=config, min_count=1
        ),
        "heat_analog_shadow_detail": build_heat_analog_shadow_detail(bt, config=config),
        "daily_peak_shadow_window_scorecard": build_daily_peak_shadow_window_scorecard(
            bt, config=config
        ),
        "backtest_metrics_by_segment_by_stage": build_backtest_metrics_by_segment_by_stage(
            bt, min_count=min_segment_count
        ),
        "error_by_hour_by_stage": build_metrics_by_group_by_stage(
            bt, ["Hour"], min_count=1
        ),
        "error_by_forecast_lead_hour_by_stage": build_metrics_by_group_by_stage(
            bt, ["Forecast_Lead_Hour"], min_count=1
        ),
        "error_by_forecast_day_by_stage": build_metrics_by_group_by_stage(
            bt, ["Forecast_Day"], min_count=1
        ),
        "seasonal_error_by_max_temp_bin_by_stage": build_metrics_by_group_by_stage(
            bt,
            ["Season", "DailyMaxTempBucket", "HourGroup"],
            min_count=min_segment_count,
        ),
        "seasonal_error_by_cloud_bin_by_stage": build_metrics_by_group_by_stage(
            bt, ["Season", "CloudCoverBucket", "HourGroup"], min_count=min_segment_count
        ),
        "seasonal_error_by_solar_loss_bin_by_stage": build_metrics_by_group_by_stage(
            bt, ["Season", "SolarLossBucket", "HourGroup"], min_count=min_segment_count
        ),
        "cloud_solar_event_error_by_stage": build_metrics_by_group_by_stage(
            bt, ["CloudSolarEventClass", "HourGroup"], min_count=min_segment_count
        ),
        "cloud_solar_event_hour_error_by_stage": build_metrics_by_group_by_stage(
            bt, ["CloudSolarEventClass", "Hour"], min_count=min_segment_count
        ),
        "daily_peak_miss_by_stage": build_daily_peak_miss_by_stage(bt),
        "stage_marginal_contributions": build_stage_marginal_contributions(bt),
        "focused_guard_rule_audit": build_focused_scorecard_rule_audit(
            bt,
            config,
            forecast_col="Pre_Focused_Guard_Forecast_MWH",
        ),
        "top_100_underforecast_hours_by_stage": under_stage,
        "top_100_overforecast_hours_by_stage": over_stage,
        "band_coverage_summary": band_summary,
        "band_coverage_by_hour": band_by_hour,
        "band_coverage_by_temp_bucket": band_by_temp,
        "band_coverage_detail": band_detail,
        "future_peak_muting_audit": build_future_peak_muting_audit(
            forecast_display_df, backtest_df=bt, config=config
        ),
        "recent_residual_profile_debug": build_recent_profile_debug_table(
            recent_residual_profile
        ),
        "calibration_lookup_debug": build_calibration_debug_table(
            calibration_lookup_bundle, heat_peak_lookup, warm_ramp_lookup
        ),
        "cloud_solar_shape_lookup_debug": cloud_solar_debug,
        "band_lookup_debug": build_band_debug_table(residual_band_lookup),
        "feature_importance": feature_importance,
        "prophet_regressors": prophet_regs,
        "forecast_display_for_review": (
            forecast_display_df.copy()
            if isinstance(forecast_display_df, pd.DataFrame)
            else pd.DataFrame()
        ),
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


def export_diagnostics_bundle(
    bundle: dict[str, Any], output_dir: str | Path
) -> dict[str, str]:
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
            file_path.write_text(
                json.dumps(_json_clean(value), indent=2), encoding="utf-8"
            )
            written[name] = str(file_path)

    manifest = output_path / "diagnostics_manifest.json"
    manifest.write_text(json.dumps(written, indent=2), encoding="utf-8")
    written["diagnostics_manifest"] = str(manifest)
    return written
