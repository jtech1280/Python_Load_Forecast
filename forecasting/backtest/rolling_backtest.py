from __future__ import annotations

import time

import pandas as pd
import numpy as np
from forecasting.forecast.recursive_engine import recursive_forecast
from forecasting.model.trainers import train_tree_models
from forecasting.model.prophet_model import train_prophet, DEFAULT_PROPHET_REGRESSORS, prophet_enabled
from forecasting.model.catboost_model import train_catboost, catboost_enabled


PRED_COLS = [
    "DT",
    "Raw_Forecast_MWH",
    "XGB_Pred_MWH",
    "LGB_Pred_MWH",
    "CatBoost_Pred_MWH",
    "Prophet_Pred_MWH",
    "Prophet_Lower_MWH",
    "Prophet_Upper_MWH",
    "MWH_Lag24",
    "MWH_SameHour7DayMean",
    "Load_Decay_1Hr_MWH",
    "Load_Decay_2Hr_MWH",
    "Lag1_Minus_SameHourYesterday_MWH",
    "Lag1_Minus_SameHour7DayMean_MWH",
    "Lag24_Minus_SameHour7DayMean_MWH",
    "PeakWindow_Lag1_Minus_SameHour7DayMean_MWH",
    "PeakWindow_Lag24_Minus_SameHour7DayMean_MWH",
    "PeakWindow16to18_Lag1_Minus_SameHour7DayMean_MWH",
    "PeakWindow16to18_Lag24_Minus_SameHour7DayMean_MWH",
    "HotPeak_Lag1_Minus_SameHourYesterday_MWH",
    "HotPeak_Lag1_Minus_SameHour7DayMean_MWH",
    "HotPeak_Lag24_Minus_SameHour7DayMean_MWH",
    "ClearHotPeak_Lag1_Minus_SameHourYesterday_MWH",
    "ClearHotPeak_Lag1_Minus_SameHour7DayMean_MWH",
    "ClearHotPeak_Lag24_Minus_SameHour7DayMean_MWH",
    "ClearPeak16to18_Lag24_Minus_SameHour7DayMean_MWH",
    "OvercastPeak16to18_Lag24_Minus_SameHour7DayMean_MWH",
    "ClearHotPeak16to18_Lag24_Minus_SameHour7DayMean_MWH",
    "OvercastCoolPeak16to18_Lag1_Minus_SameHour7DayMean_MWH",
    "OvercastCoolPeak16to18_Lag24_Minus_SameHour7DayMean_MWH",
    "PostPeak_LoadDecay_1Hr_MWH",
    "PostPeak_LoadDecay_2Hr_MWH",
    "PostPeak_LoadDecay_VsSameHourYesterday_MWH",
    "PostPeak_LoadDecay_VsSameHour7DayMean_MWH",
    "ClearHotEvening_LoadDecay_Vs7Day_MWH",
    "DeltaBreeze_PostPeak_LoadDecay_Signal",
]


def _safe_mape(actual: pd.Series, forecast: pd.Series) -> float:
    a = pd.to_numeric(actual, errors="coerce").astype(float)
    f = pd.to_numeric(forecast, errors="coerce").astype(float)
    mask = a.abs() > 1e-9
    return float((np.abs((a[mask] - f[mask]) / a[mask])).mean() * 100.0) if mask.any() else np.nan


def _record_timing(
    timing_rows: list[dict] | None,
    *,
    stage: str,
    started: float,
    status: str = "completed",
    rows: int | None = None,
    detail: str = "",
) -> None:
    if timing_rows is None:
        return
    timing_rows.append({
        "Stage": str(stage),
        "Status": str(status),
        "Elapsed_Sec": round(float(time.perf_counter() - started), 3),
        "Rows": np.nan if rows is None else int(rows),
        "Detail": str(detail or ""),
    })


def run_rolling_backtest(
    train_df: pd.DataFrame,
    features: list[str],
    ensemble_weights: dict[str, float],
    backtest_days: int,
    config: dict | None = None,
    skip_catboost: bool = False,
    skip_prophet: bool = False,
    collect_timing: bool = False,
) -> pd.DataFrame:
    """
    Hold out the most recent N days, train only on data before the cutoff, then forecast recursively.
    This avoids the calibration leakage that happens when the backtest model has already seen the test window.
    """
    train_df = train_df.copy().sort_values("DT").reset_index(drop=True)
    cutoff_dt = train_df["DT"].max() - pd.Timedelta(days=int(backtest_days))
    hist = train_df[train_df["DT"] < cutoff_dt].copy()
    test = train_df[train_df["DT"] >= cutoff_dt].copy()

    if hist.empty or test.empty:
        return pd.DataFrame()

    timing_rows: list[dict] | None = [] if collect_timing else None
    started = time.perf_counter()
    bt_xgb, bt_lgb, bt_features = train_tree_models(hist, features, config=config, stage_name="backtest holdout")
    _record_timing(timing_rows, stage="tree training", started=started, rows=len(hist), detail=f"features={len(bt_features)}")

    # Prophet is trained/predicted for benchmarking. Production Raw_Forecast_MWH uses the weights passed in;
    # V12 passes prophet=0 unless blend_into_production is explicitly enabled.
    started = time.perf_counter()
    if skip_prophet and prophet_enabled(config):
        bt_prophet_fit = None
        _record_timing(timing_rows, stage="Prophet training", started=started, status="skipped", detail="skip_prophet=true")
    elif prophet_enabled(config):
        bt_prophet_fit = train_prophet(hist, DEFAULT_PROPHET_REGRESSORS, config=config)
        _record_timing(
            timing_rows,
            stage="Prophet training",
            started=started,
            rows=len(hist),
            detail=(
                f"regressors={len(bt_prophet_fit.regressors) if bt_prophet_fit is not None else 0};"
                f"fit_rows={bt_prophet_fit.train_rows if bt_prophet_fit is not None else 0};"
                f"source_rows={bt_prophet_fit.source_rows if bt_prophet_fit is not None else 0}"
            ),
        )
    else:
        bt_prophet_fit = None
        _record_timing(timing_rows, stage="Prophet training", started=started, status="skipped", detail="prophet_enabled=false")
    bt_prophet_features = bt_prophet_fit.regressors if bt_prophet_fit is not None else []

    started = time.perf_counter()
    if skip_catboost:
        bt_catboost = None
        _record_timing(timing_rows, stage="CatBoost training", started=started, status="skipped", detail="skip_catboost=true")
    else:
        bt_catboost, _ = train_catboost(hist, bt_features, config=config) if catboost_enabled(config) else (None, bt_features)
        _record_timing(
            timing_rows,
            stage="CatBoost training",
            started=started,
            rows=len(hist) if catboost_enabled(config) else None,
            status="completed" if catboost_enabled(config) else "skipped",
            detail="" if catboost_enabled(config) else "catboost_enabled=false",
        )

    seed = hist[["DT", "MWH"]].copy()
    future_frame = test.drop(columns=["MWH"]).copy()
    started = time.perf_counter()
    fut_preds = recursive_forecast(
        future_frame, seed, bt_xgb, bt_lgb, bt_features, ensemble_weights,
        prophet_fit=bt_prophet_fit, prophet_features=bt_prophet_features, catboost_model=bt_catboost,
    )
    _record_timing(timing_rows, stage="recursive forecast", started=started, rows=len(fut_preds))

    context_cols = [
        "DT", "MWH", "Season", "Month", "Hour", "HourGroup", "DOW", "IsWeekend", "IsHoliday",
        "IsLikelySystemPeakHour", "Temperature", "Temperature_DailyMax", "DailyMaxTempBin",
        "IsPeakWindow14to18", "IsPeakWindow16to18", "IsHotPeakWindow16to20",
        "ClearHotPeakWindow16to20", "ClearHotPeakWindow16to18",
        "OvercastHotPeakWindow16to20", "CloudCover_x_PeakWindow14to18",
        "CloudCover_x_PeakWindow16to18", "CloudCover_x_HotPeakWindow16to20",
        "ClearPeakWindow14to18", "OvercastPeakWindow14to18",
        "ClearPeakWindow16to18", "OvercastPeakWindow16to18",
        "PeakWindowDailyMaxBelow75", "PeakWindowDailyMax75to85", "PeakWindowDailyMax85to90",
        "ClearPeakDailyMaxBelow75", "ClearPeakDailyMax75to85", "ClearPeakDailyMax85to90",
        "OvercastPeakDailyMaxBelow75", "OvercastPeakDailyMax75to85", "OvercastPeakDailyMax85to90",
        "PeakHE16to18DailyMaxBelow75", "OvercastPeakHE16to18DailyMaxBelow75",
        "OvercastCoolPeakWindow16to18", "OvercastPeakHE16to18DailyMax90to92_5",
        "ClearPeakHE16to18DailyMax85to90", "ClearPeakHE16to18DailyMax95to98",
        "ClearPeakHE16to18DailyMax98to100", "ClearPeakHE16to18DailyMax100to105",
        "ClearPeakHE16to18DailyMax105Plus",
        "HotPeakDailyMax90to92_5", "HotPeakDailyMax92_5to95", "HotPeakDailyMax95to98",
        "HotPeakDailyMax98to100", "HotPeakDailyMax100to105", "HotPeakDailyMax105Plus",
        "ClearHotPeakDailyMax90to92_5", "ClearHotPeakDailyMax92_5to95",
        "ClearHotPeakDailyMax95to98", "ClearHotPeakDailyMax98to100", "ClearHotPeakDailyMax100to105",
        "ClearHotPeakDailyMax105Plus",
        "OvercastHotPeakDailyMax90to92_5", "OvercastHotPeakDailyMax92_5to95",
        "OvercastHotPeakDailyMax95to98", "OvercastHotPeakDailyMax98to100",
        "OvercastHotPeakDailyMax100to105", "OvercastHotPeakDailyMax105Plus",
        "ClearHotPeak_x_DailyMaxExcess90", "ClearHotPeak_x_DailyMaxExcess95",
        "ClearHotPeak_x_CDD", "OvercastHotPeak_x_DailyMaxExcess90",
        "OvercastHotPeak_x_DailyMaxExcess95", "OvercastHotPeak_x_CDD",
        "Month_x_HotPeak", "MonthSin_x_HotPeak", "MonthCos_x_HotPeak",
        "Month_x_OvercastPeakWindow14to18", "MonthSin_x_OvercastPeakWindow14to18",
        "MonthCos_x_OvercastPeakWindow14to18",
        "Month_x_OvercastPeakHE16to18DailyMaxBelow75",
        "Month_x_OvercastPeakHE16to18DailyMax90to92_5",
        "Month_x_ClearPeakHE16to18DailyMax85to90",
        "Month_x_ClearPeakHE16to18DailyMax95to98",
        "Month_x_ClearPeakHE16to18DailyMax98to100",
        "Month_x_ClearPeakHE16to18DailyMax100to105",
        "Month_x_ClearPeakHE16to18DailyMax105Plus",
        "Month_x_ClearHotPeak", "MonthSin_x_ClearHotPeak", "MonthCos_x_ClearHotPeak",
        "Month_x_OvercastHotPeak", "MonthSin_x_OvercastHotPeak", "MonthCos_x_OvercastHotPeak",
        "ClearHotPeak_x_HE16", "ClearHotPeak_x_HE17",
        "ClearHotPeak_x_HE18", "ClearHotPeak_x_HE19", "ClearHotPeak_x_HE20",
        "BTM_Solar_Proxy_MW", "BTM_Solar_Loss_From_ClearSky_MW", "Midday_Overcast_Solar_Loss_MW", "ClearSky_Index", "CloudCover_Norm", "Humidity_Norm", "WindSpeed_Mph", "WindDirection_Deg", "WindDirection_Available_Flag", "Westerly_Flow_Mph", "Westerly_Flow_Flag", "WindRamp_1Hr_Mph", "WindRamp_3Hr_Mph", "WindRamp_Next1Hr_Mph", "WindRamp_Next3Hr_Mph", "WesterlyFlow_Ramp_1Hr_Mph", "WesterlyFlow_Ramp_3Hr_Mph", "WesterlyFlow_Next1Hr_Ramp_Mph", "WesterlyFlow_Next3Hr_Ramp_Mph", "Temperature_Drop_From_DailyMax_F", "TempDrop_1Hr_F", "TempDrop_2Hr_F", "TempDrop_3Hr_F", "TempDrop_Next1Hr_F", "TempDrop_Next2Hr_F", "TempDrop_Next3Hr_F", "IsPostPeakEvening18to23", "ClearHotEvening_Flag", "ClearVeryHotEvening_Flag", "ClearHotEvening_x_TempDropFromDailyMax", "ClearHotEvening_x_ForecastDropNext3Hr", "ClearHotEvening_x_WesterlyFlow", "ClearHotEvening_x_WesterlyFlowRamp", "DeltaBreeze_Westerly_Flow_Flag", "DeltaBreeze_EveningWindRamp_Flag", "DeltaBreeze_Cooling_Flag", "DeltaBreeze_Cooling_Signal", "DeltaBreeze_CoolingNoDirection_Signal", "DeltaBreeze_ClearHotEvening_Signal", "PrecipIn",
    ]
    context_cols = [c for c in context_cols if c in test.columns]
    pred_cols = [c for c in PRED_COLS if c in fut_preds.columns]
    out = test[context_cols].merge(fut_preds[pred_cols], on="DT", how="left")
    out.rename(columns={"MWH": "Actual_MWH"}, inplace=True)
    out["Residual_MWH"] = out["Actual_MWH"].astype(float) - out["Raw_Forecast_MWH"].astype(float)
    out["AbsError_MWH"] = out["Residual_MWH"].abs()
    out["APE"] = np.where(out["Actual_MWH"].abs() > 1e-9, out["AbsError_MWH"] / out["Actual_MWH"].abs() * 100.0, np.nan)
    out.attrs["metrics"] = {
        "MAE_MWH": float(out["AbsError_MWH"].mean()),
        "RMSE_MWH": float(np.sqrt(np.mean(np.square(out["Residual_MWH"])))) if len(out) else np.nan,
        "MAPE_PCT": _safe_mape(out["Actual_MWH"], out["Raw_Forecast_MWH"]),
        "Peak_Error_MWH": float(out.loc[out["Actual_MWH"].idxmax(), "Raw_Forecast_MWH"] - out["Actual_MWH"].max()) if len(out) else np.nan,
    }
    if timing_rows is not None:
        out.attrs["rolling_backtest_timing"] = timing_rows
    return out
