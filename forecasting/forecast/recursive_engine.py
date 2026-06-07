from __future__ import annotations

import numpy as np
import pandas as pd

from forecasting.model.ensemble import blend_predictions
from forecasting.model.prophet_model import predict_prophet


def _last(values: list[float], n: int) -> float:
    return float(values[-n]) if len(values) >= n else np.nan


def _rolling(values: list[float], window: int, min_periods: int | None = None, std: bool = False) -> float:
    min_periods = min_periods or max(2, int(window * 0.5))
    if len(values) < min_periods:
        return np.nan
    arr = np.asarray(values[-window:], dtype=float)
    if arr.size == 0:
        return np.nan
    return float(np.nanstd(arr, ddof=1)) if std and arr.size > 1 else float(np.nanmean(arr))


def _same_hour_7day_mean(values: list[float]) -> float:
    idxs = [24 * i for i in range(1, 8)]
    vals = [values[-i] for i in idxs if len(values) >= i]
    return float(np.nanmean(vals)) if vals else np.nan


def _prepare_x(row: pd.Series, features: list[str]) -> pd.DataFrame:
    x = row.reindex(features)
    x = pd.to_numeric(x, errors="coerce").replace([np.inf, -np.inf], np.nan).fillna(0.0)
    return x.astype(float).to_frame().T


def recursive_forecast(
    future_frame: pd.DataFrame,
    historical_seed: pd.DataFrame,
    xgb_model,
    lgb_model,
    features: list[str],
    ensemble_weights: dict[str, float],
    prophet_fit=None,
    prophet_features: list[str] | None = None,
    catboost_model=None,
) -> pd.DataFrame:
    """Recursively predict MWH for each future hour using prior actuals + prior forecasts as lag memory.

    XGBoost/LightGBM use recursive lag features. Prophet is optional and contributes a non-recursive
    trend/seasonality/regressor prediction that is blended with the tree model predictions.
    """
    fut = future_frame.copy().sort_values("DT").reset_index(drop=True)
    hist = historical_seed[["DT", "MWH"]].copy().sort_values("DT")
    base_series = pd.to_numeric(hist["MWH"], errors="coerce").dropna().astype(float).tolist()

    prophet_components = pd.DataFrame(index=fut.index)
    if prophet_fit is not None:
        try:
            prophet_components = predict_prophet(prophet_fit, fut, prophet_features).reset_index(drop=True)
        except Exception as exc:
            # Prophet should not break the whole forecast if the tree ensemble is healthy.
            print(f"WARNING: Prophet prediction failed; continuing with XGB/LGB only. Details: {exc}")
            prophet_components = pd.DataFrame(index=fut.index)

    preds = []
    xgb_preds = []
    lgb_preds = []
    prophet_preds = []
    catboost_preds = []

    for i in range(len(fut)):
        row = fut.iloc[i].copy()

        row["MWH_Lag1"] = _last(base_series, 1)
        row["MWH_Lag2"] = _last(base_series, 2)
        row["MWH_Lag3"] = _last(base_series, 3)
        row["MWH_Lag24"] = _last(base_series, 24)
        row["MWH_Lag48"] = _last(base_series, 48)
        row["MWH_Lag72"] = _last(base_series, 72)
        row["MWH_Lag168"] = _last(base_series, 168)
        row["MWH_Rolling3"] = _rolling(base_series, 3, min_periods=2)
        row["MWH_Rolling6"] = _rolling(base_series, 6, min_periods=3)
        row["MWH_Rolling12"] = _rolling(base_series, 12, min_periods=6)
        row["MWH_Rolling24"] = _rolling(base_series, 24, min_periods=12)
        row["MWH_Rolling48"] = _rolling(base_series, 48, min_periods=24)
        row["MWH_Rolling168"] = _rolling(base_series, 168, min_periods=84)
        row["MWH_Rolling24Std"] = _rolling(base_series, 24, min_periods=12, std=True)
        row["MWH_SameHour7DayMean"] = _same_hour_7day_mean(base_series)

        X_row = _prepare_x(row, features)
        xgb_p = float(xgb_model.predict(X_row)[0])
        lgb_p = float(lgb_model.predict(X_row)[0])
        prop_p = np.nan
        if "Prophet_Pred_MWH" in prophet_components.columns and i < len(prophet_components):
            prop_p = float(prophet_components.loc[i, "Prophet_Pred_MWH"])

        cat_p = np.nan
        if catboost_model is not None:
            try:
                cat_p = float(catboost_model.predict(X_row)[0])
            except Exception as exc:
                if i == 0:
                    print(f"WARNING: CatBoost prediction failed; continuing without CatBoost benchmark. Details: {exc}")
                cat_p = np.nan

        pred_arr = blend_predictions(
            [xgb_p], [lgb_p], ensemble_weights,
            prophet_pred=[prop_p] if np.isfinite(prop_p) else None,
            catboost_pred=[cat_p] if np.isfinite(cat_p) else None,
        )
        pred = float(pred_arr[0]) if len(pred_arr) and np.isfinite(pred_arr[0]) else np.nanmean([xgb_p, lgb_p])
        pred = max(0.0, pred)

        xgb_preds.append(max(0.0, xgb_p))
        lgb_preds.append(max(0.0, lgb_p))
        prophet_preds.append(max(0.0, prop_p) if np.isfinite(prop_p) else np.nan)
        catboost_preds.append(max(0.0, cat_p) if np.isfinite(cat_p) else np.nan)
        preds.append(pred)
        base_series.append(pred)

    fut["XGB_Pred_MWH"] = xgb_preds
    fut["LGB_Pred_MWH"] = lgb_preds
    if catboost_model is not None:
        fut["CatBoost_Pred_MWH"] = catboost_preds
    if prophet_fit is not None:
        fut["Prophet_Pred_MWH"] = prophet_preds
        for col in ["Prophet_Lower_MWH", "Prophet_Upper_MWH"]:
            if col in prophet_components.columns:
                fut[col] = prophet_components[col].to_numpy()
    fut["Raw_Forecast_MWH"] = preds
    return fut
