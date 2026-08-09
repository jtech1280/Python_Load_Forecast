from __future__ import annotations

import numpy as np
import pandas as pd

from forecasting.model.ensemble import blend_predictions
from forecasting.model.prophet_model import predict_prophet
from forecasting.utils.device_utils import prepare_for_prediction


LOAD_DECAY_SHAPE_FEATURES = [
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

RECURSIVE_LAG_OUTPUT_FEATURES = [
    "MWH_Lag1",
    "MWH_Lag2",
    "MWH_Lag3",
    "MWH_Rolling3",
    "MWH_Rolling6",
    "MWH_Rolling12",
    "MWH_Rolling24",
    "MWH_Rolling24Std",
]


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


def _row_float(row: pd.Series, col: str, default: float = 0.0) -> float:
    try:
        value = pd.to_numeric(pd.Series([row.get(col, default)]), errors="coerce").iloc[0]
        return float(value) if np.isfinite(value) else float(default)
    except Exception:
        return float(default)


def _load_decay_shape_values(row: pd.Series) -> dict[str, float]:
    lag1 = _row_float(row, "MWH_Lag1", np.nan)
    lag2 = _row_float(row, "MWH_Lag2", np.nan)
    lag3 = _row_float(row, "MWH_Lag3", np.nan)
    lag24 = _row_float(row, "MWH_Lag24", np.nan)
    same_hour_7day = _row_float(row, "MWH_SameHour7DayMean", np.nan)
    hour = _row_float(row, "Hour", np.nan)
    post_peak = _row_float(row, "IsPostPeakEvening18to23", np.nan)
    if not np.isfinite(post_peak):
        post_peak = 1.0 if np.isfinite(hour) and 18 <= int(hour) <= 23 else 0.0
    post_peak = float(np.clip(post_peak, 0.0, 1.0))
    peak_window = _row_float(row, "IsPeakWindow14to18", np.nan)
    if not np.isfinite(peak_window):
        peak_window = 1.0 if np.isfinite(hour) and 14 <= int(hour) <= 18 else 0.0
    peak_window = float(np.clip(peak_window, 0.0, 1.0))
    peak_window_16_18 = _row_float(row, "IsPeakWindow16to18", np.nan)
    if not np.isfinite(peak_window_16_18):
        peak_window_16_18 = 1.0 if np.isfinite(hour) and 16 <= int(hour) <= 18 else 0.0
    peak_window_16_18 = float(np.clip(peak_window_16_18, 0.0, 1.0))
    hot_peak = _row_float(row, "IsHotPeakWindow16to20", np.nan)
    daily_max = _row_float(row, "Temperature_DailyMax", np.nan)
    if not np.isfinite(hot_peak):
        hot_peak = 1.0 if (
            np.isfinite(hour)
            and np.isfinite(daily_max)
            and 16 <= int(hour) <= 20
            and daily_max >= 90.0
        ) else 0.0
    hot_peak = float(np.clip(hot_peak, 0.0, 1.0))
    clear_hot_peak = _row_float(row, "ClearHotPeakWindow16to20", np.nan)
    if not np.isfinite(clear_hot_peak):
        cloud = _row_float(row, "CloudCover_Norm", np.nan)
        clear_hot_peak = hot_peak if np.isfinite(cloud) and cloud <= 0.10 else 0.0
    clear_hot_peak = float(np.clip(clear_hot_peak, 0.0, 1.0))
    cloud = _row_float(row, "CloudCover_Norm", np.nan)
    clear_peak_16_18 = _row_float(row, "ClearPeakWindow16to18", np.nan)
    if not np.isfinite(clear_peak_16_18):
        clear_peak_16_18 = peak_window_16_18 if np.isfinite(cloud) and cloud <= 0.10 else 0.0
    clear_peak_16_18 = float(np.clip(clear_peak_16_18, 0.0, 1.0))
    overcast_peak_16_18 = _row_float(row, "OvercastPeakWindow16to18", np.nan)
    if not np.isfinite(overcast_peak_16_18):
        overcast_peak_16_18 = peak_window_16_18 if np.isfinite(cloud) and cloud >= 0.60 else 0.0
    overcast_peak_16_18 = float(np.clip(overcast_peak_16_18, 0.0, 1.0))
    clear_hot_peak_16_18 = _row_float(row, "ClearHotPeakWindow16to18", np.nan)
    if not np.isfinite(clear_hot_peak_16_18):
        clear_hot_peak_16_18 = clear_peak_16_18 if np.isfinite(daily_max) and daily_max >= 90.0 else 0.0
    clear_hot_peak_16_18 = float(np.clip(clear_hot_peak_16_18, 0.0, 1.0))
    overcast_cool_peak_16_18 = _row_float(row, "OvercastCoolPeakWindow16to18", np.nan)
    if not np.isfinite(overcast_cool_peak_16_18):
        overcast_cool_peak_16_18 = overcast_peak_16_18 if np.isfinite(daily_max) and daily_max < 75.0 else 0.0
    overcast_cool_peak_16_18 = float(np.clip(overcast_cool_peak_16_18, 0.0, 1.0))
    clear_hot = float(np.clip(_row_float(row, "ClearHotEvening_Flag", 0.0), 0.0, 1.0))
    delta_flag = max(
        float(np.clip(_row_float(row, "DeltaBreeze_Cooling_Flag", 0.0), 0.0, 1.0)),
        float(np.clip(_row_float(row, "DeltaBreeze_Westerly_Flow_Flag", 0.0), 0.0, 1.0)),
        float(np.clip(_row_float(row, "DeltaBreeze_EveningWindRamp_Flag", 0.0), 0.0, 1.0)),
    )

    load_decay_1 = lag2 - lag1 if np.isfinite(lag2) and np.isfinite(lag1) else np.nan
    load_decay_2 = lag3 - lag1 if np.isfinite(lag3) and np.isfinite(lag1) else np.nan
    lag1_minus_yesterday = lag1 - lag24 if np.isfinite(lag1) and np.isfinite(lag24) else np.nan
    lag1_minus_7day = lag1 - same_hour_7day if np.isfinite(lag1) and np.isfinite(same_hour_7day) else np.nan
    lag24_minus_7day = lag24 - same_hour_7day if np.isfinite(lag24) and np.isfinite(same_hour_7day) else np.nan
    vs_yesterday = lag24 - lag1 if np.isfinite(lag24) and np.isfinite(lag1) else np.nan
    vs_7day = same_hour_7day - lag1 if np.isfinite(same_hour_7day) and np.isfinite(lag1) else np.nan
    post_decay_1 = post_peak * load_decay_1 if np.isfinite(load_decay_1) else np.nan
    post_decay_2 = post_peak * load_decay_2 if np.isfinite(load_decay_2) else np.nan
    post_vs_yesterday = post_peak * vs_yesterday if np.isfinite(vs_yesterday) else np.nan
    post_vs_7day = post_peak * vs_7day if np.isfinite(vs_7day) else np.nan
    clear_hot_vs_7day = clear_hot * post_vs_7day if np.isfinite(post_vs_7day) else np.nan
    delta_signal = delta_flag * (
        max(0.0, post_decay_1) if np.isfinite(post_decay_1) else 0.0
    )
    if np.isfinite(post_vs_7day):
        delta_signal += delta_flag * max(0.0, post_vs_7day)

    return {
        "Load_Decay_1Hr_MWH": load_decay_1,
        "Load_Decay_2Hr_MWH": load_decay_2,
        "Lag1_Minus_SameHourYesterday_MWH": lag1_minus_yesterday,
        "Lag1_Minus_SameHour7DayMean_MWH": lag1_minus_7day,
        "Lag24_Minus_SameHour7DayMean_MWH": lag24_minus_7day,
        "PeakWindow_Lag1_Minus_SameHour7DayMean_MWH": peak_window * lag1_minus_7day if np.isfinite(lag1_minus_7day) else np.nan,
        "PeakWindow_Lag24_Minus_SameHour7DayMean_MWH": peak_window * lag24_minus_7day if np.isfinite(lag24_minus_7day) else np.nan,
        "PeakWindow16to18_Lag1_Minus_SameHour7DayMean_MWH": peak_window_16_18 * lag1_minus_7day if np.isfinite(lag1_minus_7day) else np.nan,
        "PeakWindow16to18_Lag24_Minus_SameHour7DayMean_MWH": peak_window_16_18 * lag24_minus_7day if np.isfinite(lag24_minus_7day) else np.nan,
        "HotPeak_Lag1_Minus_SameHourYesterday_MWH": hot_peak * lag1_minus_yesterday if np.isfinite(lag1_minus_yesterday) else np.nan,
        "HotPeak_Lag1_Minus_SameHour7DayMean_MWH": hot_peak * lag1_minus_7day if np.isfinite(lag1_minus_7day) else np.nan,
        "HotPeak_Lag24_Minus_SameHour7DayMean_MWH": hot_peak * lag24_minus_7day if np.isfinite(lag24_minus_7day) else np.nan,
        "ClearHotPeak_Lag1_Minus_SameHourYesterday_MWH": clear_hot_peak * lag1_minus_yesterday if np.isfinite(lag1_minus_yesterday) else np.nan,
        "ClearHotPeak_Lag1_Minus_SameHour7DayMean_MWH": clear_hot_peak * lag1_minus_7day if np.isfinite(lag1_minus_7day) else np.nan,
        "ClearHotPeak_Lag24_Minus_SameHour7DayMean_MWH": clear_hot_peak * lag24_minus_7day if np.isfinite(lag24_minus_7day) else np.nan,
        "ClearPeak16to18_Lag24_Minus_SameHour7DayMean_MWH": clear_peak_16_18 * lag24_minus_7day if np.isfinite(lag24_minus_7day) else np.nan,
        "OvercastPeak16to18_Lag24_Minus_SameHour7DayMean_MWH": overcast_peak_16_18 * lag24_minus_7day if np.isfinite(lag24_minus_7day) else np.nan,
        "ClearHotPeak16to18_Lag24_Minus_SameHour7DayMean_MWH": clear_hot_peak_16_18 * lag24_minus_7day if np.isfinite(lag24_minus_7day) else np.nan,
        "OvercastCoolPeak16to18_Lag1_Minus_SameHour7DayMean_MWH": overcast_cool_peak_16_18 * lag1_minus_7day if np.isfinite(lag1_minus_7day) else np.nan,
        "OvercastCoolPeak16to18_Lag24_Minus_SameHour7DayMean_MWH": overcast_cool_peak_16_18 * lag24_minus_7day if np.isfinite(lag24_minus_7day) else np.nan,
        "PostPeak_LoadDecay_1Hr_MWH": post_decay_1,
        "PostPeak_LoadDecay_2Hr_MWH": post_decay_2,
        "PostPeak_LoadDecay_VsSameHourYesterday_MWH": post_vs_yesterday,
        "PostPeak_LoadDecay_VsSameHour7DayMean_MWH": post_vs_7day,
        "ClearHotEvening_LoadDecay_Vs7Day_MWH": clear_hot_vs_7day,
        "DeltaBreeze_PostPeak_LoadDecay_Signal": delta_signal,
    }


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
    lag24_values = []
    same_hour_7day_values = []
    lag_output_values = {col: [] for col in RECURSIVE_LAG_OUTPUT_FEATURES}
    load_decay_shape_values = {col: [] for col in LOAD_DECAY_SHAPE_FEATURES}

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
        lag24_values.append(row["MWH_Lag24"])
        same_hour_7day_values.append(row["MWH_SameHour7DayMean"])
        for col in RECURSIVE_LAG_OUTPUT_FEATURES:
            lag_output_values[col].append(row[col])
        for col, value in _load_decay_shape_values(row).items():
            row[col] = value
            load_decay_shape_values[col].append(value)

        X_row = _prepare_x(row, features)
        
        # Prepare data for GPU/CPU consistency with trained models
        X_row_xgb = prepare_for_prediction(X_row, xgb_model)
        X_row_lgb = prepare_for_prediction(X_row, lgb_model)
        
        xgb_p = float(xgb_model.predict(X_row_xgb)[0])
        lgb_p = float(lgb_model.predict(X_row_lgb)[0])
        prop_p = np.nan
        if "Prophet_Pred_MWH" in prophet_components.columns and i < len(prophet_components):
            prop_p = float(prophet_components.loc[i, "Prophet_Pred_MWH"])

        cat_p = np.nan
        if catboost_model is not None:
            try:
                X_row_cat = prepare_for_prediction(X_row, catboost_model)
                cat_p = float(catboost_model.predict(X_row_cat)[0])
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
    fut["MWH_Lag24"] = lag24_values
    fut["MWH_SameHour7DayMean"] = same_hour_7day_values
    for col, values in lag_output_values.items():
        fut[col] = values
    for col, values in load_decay_shape_values.items():
        fut[col] = values
    fut["Raw_Forecast_MWH"] = preds
    return fut
