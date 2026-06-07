from __future__ import annotations

import pandas as pd
import numpy as np
from forecasting.forecast.recursive_engine import recursive_forecast
from forecasting.model.trainers import train_tree_models
from forecasting.model.prophet_model import train_prophet, DEFAULT_PROPHET_REGRESSORS, prophet_enabled
from forecasting.model.catboost_model import train_catboost, catboost_enabled


def _safe_mape(actual: pd.Series, forecast: pd.Series) -> float:
    a = pd.to_numeric(actual, errors="coerce").astype(float)
    f = pd.to_numeric(forecast, errors="coerce").astype(float)
    mask = a.abs() > 1e-9
    return float((np.abs((a[mask] - f[mask]) / a[mask])).mean() * 100.0) if mask.any() else np.nan


def run_rolling_backtest(
    train_df: pd.DataFrame,
    features: list[str],
    ensemble_weights: dict[str, float],
    backtest_days: int,
    config: dict | None = None,
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

    bt_xgb, bt_lgb, bt_features = train_tree_models(hist, features, config=config, stage_name="backtest holdout")
    # Prophet is trained/predicted for benchmarking. Production Raw_Forecast_MWH uses the weights passed in;
    # V12 passes prophet=0 unless blend_into_production is explicitly enabled.
    bt_prophet_fit = train_prophet(hist, DEFAULT_PROPHET_REGRESSORS, config=config) if prophet_enabled(config) else None
    bt_prophet_features = bt_prophet_fit.regressors if bt_prophet_fit is not None else []
    bt_catboost, _ = train_catboost(hist, bt_features, config=config) if catboost_enabled(config) else (None, bt_features)

    seed = hist[["DT", "MWH"]].copy()
    future_frame = test.drop(columns=["MWH"]).copy()
    fut_preds = recursive_forecast(
        future_frame, seed, bt_xgb, bt_lgb, bt_features, ensemble_weights,
        prophet_fit=bt_prophet_fit, prophet_features=bt_prophet_features, catboost_model=bt_catboost,
    )

    context_cols = [
        "DT", "MWH", "Season", "Month", "Hour", "HourGroup", "DOW", "IsWeekend", "IsHoliday",
        "IsLikelySystemPeakHour", "Temperature", "Temperature_DailyMax", "DailyMaxTempBin",
        "BTM_Solar_Proxy_MW", "BTM_Solar_Loss_From_ClearSky_MW", "Midday_Overcast_Solar_Loss_MW", "ClearSky_Index", "CloudCover_Norm", "Humidity_Norm", "WindSpeed_Mph", "PrecipIn",
    ]
    context_cols = [c for c in context_cols if c in test.columns]
    pred_cols = ["DT", "Raw_Forecast_MWH", "XGB_Pred_MWH", "LGB_Pred_MWH", "CatBoost_Pred_MWH", "Prophet_Pred_MWH", "Prophet_Lower_MWH", "Prophet_Upper_MWH"]
    pred_cols = [c for c in pred_cols if c in fut_preds.columns]
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
    return out
