from __future__ import annotations

from multiprocessing import Pool, cpu_count
from pathlib import Path
import time
from typing import Any

import numpy as np
import pandas as pd

from forecasting.backtest.rolling_backtest import run_rolling_backtest
from forecasting.diagnostics.forecast_diagnostics import (
    apply_multisummer_heat_analog_shadow,
    build_daily_peak_miss_by_stage,
    build_daily_peak_shadow_window_scorecard,
    build_daily_peak_window_miss_by_stage,
    build_delta_breeze_shape_metrics_by_stage,
    build_extreme_heat_peak_metrics_by_stage,
    build_extreme_heat_peak_scorecard,
    build_forecast_stage_metrics,
    build_heat_persistence_peak_candidate_scorecard,
    build_hot_ramp_peak_candidate_scorecard,
    build_hot_peak_shadow_candidate_scorecard,
    build_heat_analog_shadow_detail,
    build_heat_analog_shadow_metrics,
    build_metrics_by_group_by_stage,
    build_peak_window_bias_scorecard,
    build_peak_window_14to20_metrics_by_stage,
    build_peak_window_expansion_scorecard,
    metrics_summary,
    prep_backtest,
)
from forecasting.forecast.forecast_pipeline import (
    _production_ensemble_weights,
    apply_origin_available_correction_chain,
    build_correction_artifacts,
)
from forecasting.forecast.focused_scorecard_guard import (
    apply_focused_scorecard_guard,
    build_focused_scorecard_rule_audit,
)
from forecasting.forecast.focused_shape_residual_learner import (
    apply_focused_shape_residual_learner,
    focused_shape_residual_summary,
)
from forecasting.forecast.anomaly_exclusions import excluded_interval_mask
from forecasting.forecast.weather_scenarios import (
    add_scenario_summary_columns,
    apply_weather_scenario_delta_caps,
    make_weather_scenario_frame,
    scenario_column_name,
    scenario_definitions,
)
from forecasting.forecast.weather_robustness_hedge import apply_weather_robustness_hedge
from forecasting.forecast.operational_residual_learner import apply_operational_residual_learner
from forecasting.forecast.daily_peak_shadow_model import (
    apply_daily_peak_shadow_model,
    daily_peak_shadow_summary,
)
from forecasting.forecast.hot_ramp_peak_capture import (
    HEAT_PERSISTENCE_PEAK_COLUMNS,
    HOT_RAMP_PEAK_COLUMNS,
    apply_heat_persistence_peak_capture,
    apply_hot_ramp_peak_capture,
    heat_persistence_peak_capture_summary,
    hot_ramp_peak_capture_summary,
)
from forecasting.forecast.recursive_engine import recursive_forecast
from forecasting.data.weather_loader import fetch_previous_run_weather
from forecasting.features.feature_builder import build_forecast_frame
from forecasting.features.intraday_load_features import zero_intraday_load_features
from forecasting.model.catboost_model import catboost_enabled, train_catboost
from forecasting.model.prophet_model import DEFAULT_PROPHET_REGRESSORS, prophet_enabled, train_prophet
from forecasting.model.trainers import train_tree_models


CONTEXT_COLS = [
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
    "PriorDay_DailyMaxTemp", "PriorDay_DailyMinTemp", "DailyMaxTemp_Ramp_1Day", "DailyMinTemp_Ramp_1Day",
    "DailyMaxTemp_2DayMean", "DailyMaxTemp_3DayMean", "DailyMinTemp_2DayMean", "DailyMinTemp_3DayMean",
    "ConsecutiveHotDays90", "ConsecutiveVeryHotDays95", "ConsecutiveExtremeHotDays100",
    "HeatPersistenceStress90", "HeatPersistenceStress95", "DailyMax3DayMean_x_PeakHour",
    "OvernightHeatStress", "OvernightHeatStress_x_PeakHour",
    "BTM_Solar_Proxy_MW", "BTM_Solar_Loss_From_ClearSky_MW", "Midday_Overcast_Solar_Loss_MW",
    "ClearSky_Index", "CloudCover_Norm", "Humidity_Norm", "WindSpeed_Mph", "WindDirection_Deg",
    "WindDirection_Available_Flag", "Westerly_Flow_Mph", "Westerly_Flow_Flag",
    "WindRamp_1Hr_Mph", "WindRamp_3Hr_Mph", "WindRamp_Next1Hr_Mph", "WindRamp_Next3Hr_Mph",
    "WesterlyFlow_Ramp_1Hr_Mph", "WesterlyFlow_Ramp_3Hr_Mph",
    "WesterlyFlow_Next1Hr_Ramp_Mph", "WesterlyFlow_Next3Hr_Ramp_Mph",
    "Temperature_Drop_From_DailyMax_F", "TempDrop_1Hr_F", "TempDrop_2Hr_F", "TempDrop_3Hr_F",
    "TempDrop_Next1Hr_F", "TempDrop_Next2Hr_F", "TempDrop_Next3Hr_F",
    "IsPostPeakEvening18to23", "ClearHotEvening_Flag", "ClearVeryHotEvening_Flag",
    "ClearHotEvening_x_TempDropFromDailyMax", "ClearHotEvening_x_ForecastDropNext3Hr",
    "ClearHotEvening_x_WesterlyFlow", "ClearHotEvening_x_WesterlyFlowRamp",
    "DeltaBreeze_Westerly_Flow_Flag", "DeltaBreeze_EveningWindRamp_Flag",
    "DeltaBreeze_Cooling_Flag", "DeltaBreeze_Cooling_Signal",
    "DeltaBreeze_CoolingNoDirection_Signal", "DeltaBreeze_ClearHotEvening_Signal",
    "PrecipIn",
]
PRED_COLS = [
    "DT", "Raw_Forecast_MWH", "XGB_Pred_MWH", "LGB_Pred_MWH", "CatBoost_Pred_MWH",
    "Prophet_Pred_MWH", "Prophet_Lower_MWH", "Prophet_Upper_MWH",
    "MWH_Lag24", "MWH_SameHour7DayMean",
    "Load_Decay_1Hr_MWH", "Load_Decay_2Hr_MWH",
    "Lag1_Minus_SameHourYesterday_MWH", "Lag1_Minus_SameHour7DayMean_MWH",
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
    "PostPeak_LoadDecay_1Hr_MWH", "PostPeak_LoadDecay_2Hr_MWH",
    "PostPeak_LoadDecay_VsSameHourYesterday_MWH", "PostPeak_LoadDecay_VsSameHour7DayMean_MWH",
    "ClearHotEvening_LoadDecay_Vs7Day_MWH", "DeltaBreeze_PostPeak_LoadDecay_Signal",
]
BTM_REPLAY_COLS = ["DT", "Nameplate_MW", "Capacity_Ratio_To_Current", "Impact_Cap_MW"]
WEATHER_FRAME_COLS = ["DT", "TempF", "HumidityPct", "CloudCoverPct", "WindSpeedMph", "WindDirectionDeg", "PrecipIn", "GHI_Wm2", "IsDay"]
WEATHER_REALISM_PREFIX_COLS = [
    "Raw_Forecast_MWH", "XGB_Pred_MWH", "LGB_Pred_MWH", "CatBoost_Pred_MWH",
    "Prophet_Pred_MWH", "Residual_MWH", "Final_Backtest_Forecast_MWH", "Final_Residual_MWH",
    "Final_AbsError_MWH", "Final_APE", "Temperature", "Temperature_DailyMax",
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
    "CloudCover_Norm", "Humidity_Norm", "WindSpeed_Mph", "WindDirection_Deg",
    "WindDirection_Available_Flag", "Westerly_Flow_Mph", "Westerly_Flow_Flag",
    "WindRamp_1Hr_Mph", "WindRamp_3Hr_Mph", "WindRamp_Next1Hr_Mph", "WindRamp_Next3Hr_Mph",
    "WesterlyFlow_Ramp_1Hr_Mph", "WesterlyFlow_Ramp_3Hr_Mph",
    "WesterlyFlow_Next1Hr_Ramp_Mph", "WesterlyFlow_Next3Hr_Ramp_Mph",
    "Temperature_Drop_From_DailyMax_F", "TempDrop_1Hr_F", "TempDrop_2Hr_F", "TempDrop_3Hr_F",
    "TempDrop_Next1Hr_F", "TempDrop_Next2Hr_F", "TempDrop_Next3Hr_F",
    "IsPostPeakEvening18to23", "ClearHotEvening_Flag", "ClearVeryHotEvening_Flag",
    "ClearHotEvening_x_TempDropFromDailyMax", "ClearHotEvening_x_ForecastDropNext3Hr",
    "ClearHotEvening_x_WesterlyFlow", "ClearHotEvening_x_WesterlyFlowRamp",
    "DeltaBreeze_Westerly_Flow_Flag", "DeltaBreeze_EveningWindRamp_Flag",
    "DeltaBreeze_Cooling_Flag", "DeltaBreeze_Cooling_Signal",
    "DeltaBreeze_CoolingNoDirection_Signal", "DeltaBreeze_ClearHotEvening_Signal",
    "Load_Decay_1Hr_MWH", "Load_Decay_2Hr_MWH",
    "Lag1_Minus_SameHourYesterday_MWH", "Lag1_Minus_SameHour7DayMean_MWH",
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
    "PostPeak_LoadDecay_1Hr_MWH", "PostPeak_LoadDecay_2Hr_MWH",
    "PostPeak_LoadDecay_VsSameHourYesterday_MWH", "PostPeak_LoadDecay_VsSameHour7DayMean_MWH",
    "ClearHotEvening_LoadDecay_Vs7Day_MWH", "DeltaBreeze_PostPeak_LoadDecay_Signal",
    "BTM_Solar_Proxy_MW", "BTM_Solar_Loss_From_ClearSky_MW",
    "Midday_Overcast_Solar_Loss_MW", "Forecast_Weather_Lead_Days",
    "Weather_Robustness_Hedge_MWH", "Weather_Robustness_Hedge_Source",
    "Weather_Robustness_Jensen_MWH", "Weather_Robustness_Upper_MWH",
    "Weather_Robustness_Warmer_Delta_MWH", "Weather_Robustness_Temp_Sigma_F",
    "Weather_Robustness_Temp_Bias_Damping", "Weather_Robustness_Gate",
    "Pre_Focused_Guard_Forecast_MWH", "Post_Focused_Guard_Forecast_MWH",
    "Focused_Guard_Applied_Flag", "Focused_Scorecard_Guard_MWH",
    "Focused_Scorecard_Guard_Source", "Raw_Minus_SameHour7DayMean_MWH",
    "Raw_Minus_SameHourYesterday_MWH",
    "Focused_Shape_Model_Version", "Focused_Shape_Shadow_Mode",
    "Focused_Shape_Base_Forecast_MWH", "Focused_Shape_Correction_MWH",
    "Focused_Shape_Adjusted_Forecast_MWH", "Focused_Shape_Correction_Applied_Flag",
    "Focused_Shape_Source", "Focused_Shape_Evaluation_Mode",
    "Focused_Shape_Residual_MWH", "Focused_Shape_AbsError_MWH",
    "Focused_Shape_Delta_AbsError_MWH", "Focused_Shape_RuleUnion_Flag",
    "Focused_Shape_Scope_Flag",
    "Auto_Residual_Model_Version", "Auto_Residual_Shadow_Mode",
    "Auto_Residual_Production_Scope",
    "Auto_Residual_Base_Forecast_MWH", "Auto_Residual_Correction_MWH",
    "Auto_Residual_Adjusted_Forecast_MWH", "Auto_Residual_Correction_Applied_Flag",
    "Auto_Residual_Source", "Auto_Residual_Evaluation_Mode",
    "Auto_Residual_Residual_MWH", "Auto_Residual_AbsError_MWH",
    "Auto_Residual_Delta_AbsError_MWH",
    "Auto_Residual_Full_Shadow_Correction_MWH",
    "Auto_Residual_Full_Shadow_Adjusted_Forecast_MWH",
    "Auto_Residual_Full_Shadow_Correction_Applied_Flag",
    "Auto_Residual_Full_Shadow_Source",
    "Auto_Residual_Full_Shadow_Residual_MWH",
    "Auto_Residual_Full_Shadow_AbsError_MWH",
    "Auto_Residual_Full_Shadow_Delta_AbsError_MWH",
    "Auto_Residual_Structural_HotPeak_Correction_MWH",
    "Auto_Residual_Structural_HotPeak_Adjusted_Forecast_MWH",
    "Auto_Residual_Structural_HotPeak_Correction_Applied_Flag",
    "Auto_Residual_Structural_HotPeak_Source",
    "Auto_Residual_Structural_HotPeak_Residual_MWH",
    "Auto_Residual_Structural_HotPeak_AbsError_MWH",
    "Auto_Residual_Structural_HotPeak_Delta_AbsError_MWH",
    "Auto_Residual_Broad_HotPeak_Shadow_Correction_MWH",
    "Auto_Residual_Broad_HotPeak_Shadow_Adjusted_Forecast_MWH",
    "Auto_Residual_Broad_HotPeak_Shadow_Correction_Applied_Flag",
    "Auto_Residual_Broad_HotPeak_Shadow_Source",
    "Auto_Residual_Broad_HotPeak_Shadow_Residual_MWH",
    "Auto_Residual_Broad_HotPeak_Shadow_AbsError_MWH",
    "Auto_Residual_Broad_HotPeak_Shadow_Delta_AbsError_MWH",
    "Daily_Peak_Model_Version", "Daily_Peak_Shadow_Mode",
    "Daily_Peak_Base_Forecast_MWH", "Daily_Peak_Correction_MWH",
    "Daily_Peak_Shadow_Adjusted_Forecast_MWH", "Daily_Peak_Correction_Applied_Flag",
    "Daily_Peak_Source", "Daily_Peak_Evaluation_Mode",
    "Daily_Peak_Base_DailyPeak_MWH", "Daily_Peak_Predicted_Residual_MWH",
    "Daily_Peak_Predicted_DailyPeak_MWH", "Daily_Peak_Base_PeakHour",
    "Daily_Peak_Predicted_PeakHour", "Daily_Peak_Timing_Shift_Hours",
    "Daily_Peak_Residual_MWH", "Daily_Peak_AbsError_MWH",
    "Daily_Peak_Delta_AbsError_MWH",
    *HOT_RAMP_PEAK_COLUMNS,
    *HEAT_PERSISTENCE_PEAK_COLUMNS,
]


def _replay_cfg(config: dict | None) -> dict[str, Any]:
    return ((config or {}).get("training", {}) or {}).get("rolling_origin_replay", {}) or {}


def _as_int(value: Any, default: int, minimum: int = 1) -> int:
    try:
        return max(minimum, int(value))
    except Exception:
        return max(minimum, int(default))


def _local_datetime_series(values, index: pd.Index | None = None) -> pd.Series:
    raw = values if isinstance(values, pd.Series) else pd.Series(values, index=index)
    try:
        return pd.to_datetime(raw, errors="coerce")
    except ValueError:
        cleaned = raw.astype(str).str.strip().str.replace(r"(?:[+-]\d{2}:?\d{2}|Z)$", "", regex=True)
        return pd.to_datetime(cleaned, errors="coerce")


def _season_from_month(month: int) -> str:
    m = int(month)
    if m in (12, 1, 2):
        return "Winter"
    if m in (3, 4, 5):
        return "Spring"
    if m in (6, 7, 8, 9):
        return "Summer"
    return "Fall"


def _balanced_seasonal_origins(candidates: list[pd.Timestamp], config: dict) -> list[pd.Timestamp]:
    cfg = _replay_cfg(config)
    max_origins = _as_int(cfg.get("max_origins"), 12)
    per_season = _as_int(cfg.get("origins_per_season"), max(1, max_origins // 4))

    chosen: list[pd.Timestamp] = []
    for season in ["Winter", "Spring", "Summer", "Fall"]:
        season_candidates = [origin for origin in candidates if _season_from_month(origin.month) == season]
        chosen.extend(season_candidates[:per_season])
    if len(chosen) < max_origins:
        chosen.extend(origin for origin in candidates if origin not in chosen)
    return chosen[:max_origins]


def _fixed_origin_values(cfg: dict[str, Any]) -> list[str]:
    values: list[str] = []
    raw = cfg.get("fixed_origins") or cfg.get("origin_dates") or []
    if isinstance(raw, str):
        values.extend(part.strip() for part in raw.replace(";", ",").split(",") if part.strip())
    elif isinstance(raw, (list, tuple)):
        values.extend(str(part).strip() for part in raw if str(part).strip())

    file_path = str(cfg.get("fixed_origins_file") or cfg.get("origin_dates_file") or "").strip()
    if file_path:
        path = Path(file_path)
        if not path.exists():
            raise FileNotFoundError(f"Fixed replay origins file not found: {path}")
        values.extend(
            line.strip().lstrip("\ufeff")
            for line in path.read_text(encoding="utf-8").splitlines()
            if line.strip().lstrip("\ufeff") and not line.strip().lstrip("\ufeff").startswith("#")
        )
    return values


def _parse_fixed_origins(cfg: dict[str, Any], first_dt: pd.Timestamp) -> list[pd.Timestamp]:
    raw_values = _fixed_origin_values(cfg)
    if not raw_values:
        return []

    parsed: list[pd.Timestamp] = []
    target_tz = first_dt.tz
    for raw in raw_values:
        raw_text = str(raw).strip()
        ts = pd.Timestamp(raw)
        if pd.isna(ts):
            continue
        if ts.tz is None and target_tz is not None:
            ts = ts.tz_localize(target_tz)
        elif ts.tz is not None and target_tz is not None:
            ts = ts.tz_convert(target_tz)
        has_explicit_time = any(sep in raw_text for sep in (" ", "T")) and len(raw_text) > 10
        parsed.append(ts if has_explicit_time else ts.normalize())

    return list(dict.fromkeys(parsed))


def _origin_candidates(train_df: pd.DataFrame, config: dict) -> list[pd.Timestamp]:
    cfg = _replay_cfg(config)
    horizon_days = _as_int(cfg.get("horizon_days"), 16)
    origin_step_days = _as_int(cfg.get("origin_step_days"), 28)
    max_origins = _as_int(cfg.get("max_origins"), 12)
    min_train_days = _as_int(cfg.get("min_train_days"), 365)
    calibration_days = _as_int(cfg.get("calibration_days"), 45)

    dt = pd.to_datetime(train_df["DT"], errors="coerce").dropna()
    if dt.empty:
        return []
    first_dt = dt.min()
    latest_origin = dt.max().normalize() - pd.Timedelta(days=horizon_days - 1)
    earliest_origin = first_dt.normalize() + pd.Timedelta(days=min_train_days + calibration_days)

    fixed_origins = _parse_fixed_origins(cfg, first_dt)
    if fixed_origins:
        valid_fixed: list[pd.Timestamp] = []
        for origin in fixed_origins:
            horizon_end = origin + pd.Timedelta(days=horizon_days)
            if origin < earliest_origin or origin > latest_origin:
                continue
            if ((dt >= origin) & (dt < horizon_end)).sum() >= 24:
                valid_fixed.append(origin)
        return valid_fixed[:max_origins]

    origins: list[pd.Timestamp] = []
    origin = latest_origin
    while origin >= earliest_origin:
        horizon_end = origin + pd.Timedelta(days=horizon_days)
        if ((dt >= origin) & (dt < horizon_end)).sum() >= 24:
            origins.append(origin)
        origin -= pd.Timedelta(days=origin_step_days)

    selection = str(cfg.get("origin_selection", "seasonal_balanced")).strip().lower()
    if selection == "seasonal_balanced":
        origins = _balanced_seasonal_origins(origins, config)
    else:
        origins = origins[:max_origins]
    return list(reversed(origins))


def _horizon_bucket(day: int) -> str:
    if int(day) <= 1:
        return "Day1"
    if int(day) <= 7:
        return "Days2to7"
    if int(day) <= 16:
        return "Days8to16"
    return "Days17Plus"


def _weather_realism_cfg(config: dict | None) -> dict[str, Any]:
    return (_replay_cfg(config).get("weather_realism", {}) or {}) if isinstance(config, dict) else {}


def _add_origin_metadata(out: pd.DataFrame, origin_dt: pd.Timestamp, origin_number: int) -> pd.DataFrame:
    out = out.sort_values("DT").reset_index(drop=True).copy()
    out["Replay_Origin_ID"] = f"origin_{origin_number:02d}"
    out["Replay_Origin_DT"] = origin_dt
    out["Replay_Origin_Year"] = int(origin_dt.year)
    out["Replay_Origin_Month"] = int(origin_dt.month)
    out["Replay_Origin_Season"] = _season_from_month(origin_dt.month)
    out["Forecast_Lead_Hour"] = np.arange(1, len(out) + 1, dtype=int)
    out["Forecast_Day"] = ((out["Forecast_Lead_Hour"] - 1) // 24 + 1).astype(int)
    out["Replay_Horizon_Bucket"] = out["Forecast_Day"].map(_horizon_bucket)
    return out


def _append_timing_row(
    timing_rows: list[dict[str, Any]] | None,
    *,
    origin_number: int,
    origin_dt: pd.Timestamp,
    stage: str,
    elapsed_sec: float,
    status: str = "completed",
    rows: int | None = None,
    scenario_count: int | None = None,
    detail: str = "",
) -> None:
    if timing_rows is None:
        return
    timing_rows.append({
        "Replay_Origin_ID": f"origin_{origin_number:02d}",
        "Replay_Origin_Number": int(origin_number),
        "Replay_Origin_DT": origin_dt,
        "Replay_Origin_Year": int(origin_dt.year),
        "Replay_Origin_Month": int(origin_dt.month),
        "Replay_Origin_Season": _season_from_month(origin_dt.month),
        "Stage": str(stage),
        "Status": str(status),
        "Elapsed_Sec": round(float(elapsed_sec), 3),
        "Rows": np.nan if rows is None else int(rows),
        "Scenario_Count": np.nan if scenario_count is None else int(scenario_count),
        "Detail": str(detail or ""),
    })


def _log_and_record_timing(
    timing_rows: list[dict[str, Any]] | None,
    *,
    origin_number: int,
    origin_dt: pd.Timestamp,
    stage: str,
    started: float,
    log_timing: bool,
    status: str = "completed",
    rows: int | None = None,
    scenario_count: int | None = None,
    detail: str = "",
) -> float:
    elapsed = time.perf_counter() - started
    if log_timing:
        suffix = f" completed in {elapsed:.1f}s" if status == "completed" else f" {status} in {elapsed:.1f}s"
        print(f"Rolling-origin replay origin {origin_number}: {stage}{suffix}", flush=True)
    _append_timing_row(
        timing_rows,
        origin_number=origin_number,
        origin_dt=origin_dt,
        stage=stage,
        elapsed_sec=elapsed,
        status=status,
        rows=rows,
        scenario_count=scenario_count,
        detail=detail,
    )
    return elapsed


def _btm_from_feature_frame(frame: pd.DataFrame) -> pd.DataFrame:
    cols = [col for col in BTM_REPLAY_COLS if col in frame.columns]
    if len(cols) < len(BTM_REPLAY_COLS):
        return pd.DataFrame(columns=BTM_REPLAY_COLS)
    out = frame[cols].copy().sort_values("DT")
    out["Replay_PeriodStart"] = pd.to_datetime(out["DT"]).dt.tz_localize(None).dt.to_period("M").dt.to_timestamp()
    out = out.drop_duplicates(subset=["Replay_PeriodStart"], keep="last").drop(columns=["Replay_PeriodStart"])
    return out.reset_index(drop=True)


def _previous_run_future_frame(target: pd.DataFrame, config: dict, origin_dt: pd.Timestamp) -> pd.DataFrame:
    cfg = _weather_realism_cfg(config)
    if not bool(cfg.get("enabled", False)):
        return pd.DataFrame()
    max_previous_days = _as_int(cfg.get("max_previous_days"), 7)
    # Hard ceiling on how far back the previous-runs weather provider can supply a
    # fixed-lead forecast. Open-Meteo's previous-runs API historically supports ~7 days;
    # raise provider_max_days only if your provider/plan returns longer leads (otherwise
    # the extra leads simply return empty and are skipped). Days beyond this remain
    # OUTSIDE the production-weather-validated window (see scorecard summary flag).
    provider_max_days = _as_int(cfg.get("provider_max_days"), 7)
    last_eligible = min(int(max_previous_days), int(provider_max_days))
    target_dt = pd.to_datetime(target["DT"], errors="coerce")
    lead_days = (target_dt.dt.normalize() - pd.Timestamp(origin_dt).normalize()).dt.days + 1
    eligible = target.loc[lead_days.between(1, last_eligible)].copy()
    if eligible.empty:
        return pd.DataFrame()

    try:
        previous = fetch_previous_run_weather(
            config,
            start_dt=eligible["DT"].min(),
            end_dt=eligible["DT"].max(),
            max_previous_days=last_eligible,
        )
    except Exception as exc:
        print(f"WARNING: previous-run weather realism fetch failed for replay origin {origin_dt}. Details: {exc}")
        return pd.DataFrame()
    if previous.empty:
        return pd.DataFrame()

    selector = eligible[["DT"]].copy()
    selector["Forecast_Weather_Lead_Days"] = (
        pd.to_datetime(selector["DT"], errors="coerce").dt.normalize() - pd.Timestamp(origin_dt).normalize()
    ).dt.days + 1
    previous = previous.rename(columns={"Previous_Run_Lead_Days": "Forecast_Weather_Lead_Days"})
    weather_cols = [col for col in WEATHER_FRAME_COLS if col in previous.columns]
    selected_weather = selector.merge(
        previous[weather_cols + ["Forecast_Weather_Lead_Days"]],
        on=["DT", "Forecast_Weather_Lead_Days"],
        how="inner",
    )
    if selected_weather.empty:
        return pd.DataFrame()
    btm = _btm_from_feature_frame(target)
    if btm.empty:
        return pd.DataFrame()
    future = build_forecast_frame(selected_weather[weather_cols], btm)
    future = future.merge(
        selected_weather[["DT", "Forecast_Weather_Lead_Days"]],
        on="DT",
        how="left",
    )
    return future.sort_values("DT").reset_index(drop=True)


def _raw_prediction_frame(
    target: pd.DataFrame,
    future_frame: pd.DataFrame,
    hist: pd.DataFrame,
    features: list[str],
    ensemble_weights: dict[str, float],
    xgb_model,
    lgb_model,
    prophet_fit,
    prophet_features: list[str],
    catboost_model,
    origin_dt: pd.Timestamp,
    origin_number: int,
    config: dict | None = None,
) -> pd.DataFrame:
    if target.empty or future_frame.empty:
        return pd.DataFrame()
    target_actual = target[["DT", "MWH"]].copy()
    future = future_frame[future_frame["DT"].isin(target_actual["DT"])].copy()
    if future.empty:
        return pd.DataFrame()
    five_min_cfg = ((config or {}).get("five_min_load", {}) or {})
    if not bool(five_min_cfg.get("future_model_features_enabled", False)):
        future = zero_intraday_load_features(future)
    raw = recursive_forecast(
        future_frame=future.drop(columns=["MWH"], errors="ignore"),
        historical_seed=hist[["DT", "MWH"]].copy(),
        xgb_model=xgb_model,
        lgb_model=lgb_model,
        features=features,
        ensemble_weights=ensemble_weights,
        prophet_fit=prophet_fit,
        prophet_features=prophet_features,
        catboost_model=catboost_model,
    )

    context_cols = [c for c in CONTEXT_COLS if c in future.columns and c != "MWH"]
    for extra in ["Forecast_Weather_Lead_Days"]:
        if extra in future.columns:
            context_cols.append(extra)
    pred_cols = [c for c in PRED_COLS if c in raw.columns]
    out = target_actual.rename(columns={"MWH": "Actual_MWH"}).merge(
        future[context_cols],
        on="DT",
        how="left",
    ).merge(raw[pred_cols], on="DT", how="left")
    out["Residual_MWH"] = pd.to_numeric(out["Actual_MWH"], errors="coerce") - pd.to_numeric(out["Raw_Forecast_MWH"], errors="coerce")
    out["AbsError_MWH"] = out["Residual_MWH"].abs()
    out["APE"] = np.where(
        pd.to_numeric(out["Actual_MWH"], errors="coerce").abs() > 1e-9,
        out["AbsError_MWH"] / pd.to_numeric(out["Actual_MWH"], errors="coerce").abs() * 100.0,
        np.nan,
    )
    return _add_origin_metadata(out, origin_dt, origin_number)


def _origin_raw_forecasts(
    train_df: pd.DataFrame,
    features: list[str],
    config: dict,
    origin_dt: pd.Timestamp,
    horizon_days: int,
    origin_number: int,
    timing_rows: list[dict[str, Any]] | None = None,
) -> tuple[pd.DataFrame, pd.DataFrame, dict[str, pd.DataFrame], dict[str, pd.DataFrame]]:
    hist = train_df[train_df["DT"] < origin_dt].copy()
    target = train_df[(train_df["DT"] >= origin_dt) & (train_df["DT"] < origin_dt + pd.Timedelta(days=horizon_days))].copy()
    if hist.empty or target.empty:
        return pd.DataFrame(), pd.DataFrame(), {}, {}

    replay_cfg = _replay_cfg(config)
    log_timing = bool(replay_cfg.get("log_timing", True))
    skip_prophet = bool(replay_cfg.get("skip_prophet", False))
    skip_catboost = bool(replay_cfg.get("skip_catboost", False))

    def _log_stage(
        stage: str,
        started: float,
        *,
        status: str = "completed",
        rows: int | None = None,
        scenario_count: int | None = None,
        detail: str = "",
    ) -> None:
        _log_and_record_timing(
            timing_rows,
            origin_number=origin_number,
            origin_dt=origin_dt,
            stage=stage,
            started=started,
            log_timing=log_timing,
            status=status,
            rows=rows,
            scenario_count=scenario_count,
            detail=detail,
        )

    stage_name = f"rolling-origin replay {origin_number} @ {origin_dt}"
    started = time.perf_counter()
    xgb_model, lgb_model, trained_features = train_tree_models(hist, features, config=config, stage_name=stage_name)
    _log_stage("tree training", started, rows=len(hist), detail=f"features={len(trained_features)}")

    started = time.perf_counter()
    if skip_prophet and prophet_enabled(config):
        prophet_fit = None
        _log_stage("Prophet training", started, status="skipped", detail="skip_prophet=true")
    else:
        prophet_fit = train_prophet(hist, DEFAULT_PROPHET_REGRESSORS, config=config) if prophet_enabled(config) else None
        _log_stage(
            "Prophet training",
            started,
            rows=len(hist),
            detail=(
                f"regressors={len(prophet_fit.regressors) if prophet_fit is not None else 0};"
                f"fit_rows={prophet_fit.train_rows if prophet_fit is not None else 0};"
                f"source_rows={prophet_fit.source_rows if prophet_fit is not None else 0}"
            ),
        )
    prophet_features = prophet_fit.regressors if prophet_fit is not None else []

    started = time.perf_counter()
    if skip_catboost and catboost_enabled(config):
        catboost_model = None
        _log_stage("CatBoost training", started, status="skipped", detail="skip_catboost=true")
    else:
        catboost_model, _ = train_catboost(hist, trained_features, config=config) if catboost_enabled(config) else (None, trained_features)
        _log_stage("CatBoost training", started, rows=len(hist))

    ensemble_weights = _production_ensemble_weights(config)
    started = time.perf_counter()
    realized = _raw_prediction_frame(
        target=target,
        future_frame=target.drop(columns=["MWH"]).copy(),
        hist=hist,
        features=trained_features,
        ensemble_weights=ensemble_weights,
        xgb_model=xgb_model,
        lgb_model=lgb_model,
        prophet_fit=prophet_fit,
        prophet_features=prophet_features,
        catboost_model=catboost_model,
        origin_dt=origin_dt,
        origin_number=origin_number,
        config=config,
    )
    _log_stage("realized base forecast", started, rows=len(realized))
    scenario_defs = scenario_definitions(config)
    realized_scenarios: dict[str, pd.DataFrame] = {}
    hedge_cfg = ((config.get("calibration", {}) or {}).get("weather_robustness_hedge", {}) or {})
    if bool(hedge_cfg.get("apply_to_realized_replay", True)) and bool(hedge_cfg.get("enabled", True)):
        started = time.perf_counter()
        realized_future = target.drop(columns=["MWH"]).copy()
        for scenario in scenario_defs:
            name = str(scenario.get("name", "scenario"))
            scenario_frame = make_weather_scenario_frame(realized_future, scenario)
            scenario_raw = _raw_prediction_frame(
                target=target,
                future_frame=scenario_frame,
                hist=hist,
                features=trained_features,
                ensemble_weights=ensemble_weights,
                xgb_model=xgb_model,
                lgb_model=lgb_model,
                prophet_fit=prophet_fit,
                prophet_features=prophet_features,
                catboost_model=catboost_model,
                origin_dt=origin_dt,
                origin_number=origin_number,
                config=config,
            )
            if not scenario_raw.empty:
                realized_scenarios[name] = scenario_raw
        _log_stage(
            "realized weather scenarios",
            started,
            rows=sum(len(frame) for frame in realized_scenarios.values()),
            scenario_count=len(realized_scenarios),
        )
    else:
        started = time.perf_counter()
        _log_stage("realized weather scenarios", started, status="skipped", scenario_count=0, detail="weather hedge disabled")
    started = time.perf_counter()
    weather_realism_future = _previous_run_future_frame(target, config, origin_dt)
    _log_stage("previous-run weather fetch/frame", started, rows=len(weather_realism_future))
    started = time.perf_counter()
    weather_realism = _raw_prediction_frame(
        target=target,
        future_frame=weather_realism_future,
        hist=hist,
        features=trained_features,
        ensemble_weights=ensemble_weights,
        xgb_model=xgb_model,
        lgb_model=lgb_model,
        prophet_fit=prophet_fit,
        prophet_features=prophet_features,
        catboost_model=catboost_model,
        origin_dt=origin_dt,
        origin_number=origin_number,
        config=config,
    )
    _log_stage("previous-run weather forecast", started, rows=len(weather_realism))
    weather_scenarios: dict[str, pd.DataFrame] = {}
    if not weather_realism_future.empty:
        started = time.perf_counter()
        for scenario in scenario_defs:
            name = str(scenario.get("name", "scenario"))
            scenario_frame = make_weather_scenario_frame(weather_realism_future, scenario)
            scenario_raw = _raw_prediction_frame(
                target=target,
                future_frame=scenario_frame,
                hist=hist,
                features=trained_features,
                ensemble_weights=ensemble_weights,
                xgb_model=xgb_model,
                lgb_model=lgb_model,
                prophet_fit=prophet_fit,
                prophet_features=prophet_features,
                catboost_model=catboost_model,
                origin_dt=origin_dt,
                origin_number=origin_number,
                config=config,
            )
            if not scenario_raw.empty:
                weather_scenarios[name] = scenario_raw
        _log_stage(
            "previous-run weather scenarios",
            started,
            rows=sum(len(frame) for frame in weather_scenarios.values()),
            scenario_count=len(weather_scenarios),
        )
    else:
        started = time.perf_counter()
        _log_stage("previous-run weather scenarios", started, status="skipped", scenario_count=0, detail="no previous-run weather frame")
    return realized, weather_realism, realized_scenarios, weather_scenarios


def _merge_weather_realism(corrected: pd.DataFrame, realism: pd.DataFrame) -> pd.DataFrame:
    if corrected.empty or realism.empty:
        return corrected
    scenario_cols = [col for col in realism.columns if col.startswith("WeatherScenario_")]
    cols = ["DT", "Replay_Origin_ID"] + [col for col in WEATHER_REALISM_PREFIX_COLS if col in realism.columns] + scenario_cols
    suffix = realism[cols].copy().rename(
        columns={col: f"WeatherRealism_{col}" for col in cols if col not in {"DT", "Replay_Origin_ID"}}
    )
    return corrected.merge(suffix, on=["DT", "Replay_Origin_ID"], how="left")


def _recompute_final_errors(df: pd.DataFrame, forecast_col: str = "Final_Backtest_Forecast_MWH") -> pd.DataFrame:
    out = df.copy()
    if "Actual_MWH" not in out.columns or forecast_col not in out.columns:
        return out
    actual = pd.to_numeric(out["Actual_MWH"], errors="coerce")
    forecast = pd.to_numeric(out[forecast_col], errors="coerce")
    out["Final_Residual_MWH"] = actual - forecast
    out["Final_AbsError_MWH"] = out["Final_Residual_MWH"].abs()
    out["Final_APE"] = np.where(
        actual.abs() > 1e-9,
        out["Final_AbsError_MWH"] / actual.abs() * 100.0,
        np.nan,
    )
    return out


def _apply_weather_scenarios_and_hedge(
    corrected: pd.DataFrame,
    raw_scenarios: dict[str, pd.DataFrame],
    config: dict,
    artifacts: dict | None,
    *,
    apply_hedge: bool,
    also_update_stage: bool = False,
) -> pd.DataFrame:
    out = corrected.copy()
    scenario_columns: list[str] = []
    for scenario_name, raw_scenario in raw_scenarios.items():
        corrected_scenario = apply_origin_available_correction_chain(raw_scenario, config, artifacts)
        col = scenario_column_name(scenario_name)
        scenario_values = corrected_scenario[["DT", "Replay_Origin_ID", "Final_Backtest_Forecast_MWH"]].copy()
        scenario_values.rename(columns={"Final_Backtest_Forecast_MWH": col}, inplace=True)
        out = out.merge(
            scenario_values,
            on=["DT", "Replay_Origin_ID"],
            how="left",
        )
        scenario_columns.append(col)
    out = apply_weather_scenario_delta_caps(
        out,
        scenario_columns,
        config=config,
        base_col="Final_Backtest_Forecast_MWH",
    )
    out = add_scenario_summary_columns(out, scenario_columns)
    if apply_hedge:
        out = apply_weather_robustness_hedge(
            out,
            config=config,
            base_col="Final_Backtest_Forecast_MWH",
            also_update_cols=("Stage_Selected_Forecast_MWH",) if also_update_stage else (),
        )
        out = _recompute_final_errors(out, forecast_col="Final_Backtest_Forecast_MWH")
    return out


def _apply_replay_focused_guard(
    df: pd.DataFrame,
    config: dict,
    *,
    also_update_stage: bool,
) -> pd.DataFrame:
    out = apply_focused_scorecard_guard(
        df,
        config=config,
        forecast_col="Final_Backtest_Forecast_MWH",
        also_update_cols=("Stage_Selected_Forecast_MWH",) if also_update_stage else (),
    )
    return _recompute_final_errors(out, forecast_col="Final_Backtest_Forecast_MWH")


def apply_origin_correction_chain(
    raw_origin: pd.DataFrame,
    raw_weather_realism: pd.DataFrame,
    raw_realized_scenarios: dict[str, pd.DataFrame],
    raw_weather_scenarios: dict[str, pd.DataFrame],
    config: dict,
    artifacts: dict,
    *,
    apply_primary_weather_hedge: bool = True,
) -> pd.DataFrame:
    """Apply the full post-training correction/calibration chain to one origin's already-computed
    raw forecast bundle. Pure transform over `artifacts` + `config`: no XGB/LGB/CatBoost/Prophet
    training happens here (that's `_origin_raw_forecasts`, called by the caller beforehand).

    Split out of `_run_single_origin_replay` so calibration-parameter search (see
    `forecasting/tuning/optuna_tuning.py` and `scripts/tune_calibration_optuna.py`) can re-run just
    this chain against a cached raw forecast bundle for many trial configs, instead of repeating the
    expensive per-origin model training on every trial. Note `artifacts` itself is config-dependent
    (built by `build_correction_artifacts`) and must be rebuilt per trial alongside this call.
    """
    corrected = apply_origin_available_correction_chain(raw_origin, config, artifacts)
    if apply_primary_weather_hedge and raw_realized_scenarios:
        corrected = _apply_weather_scenarios_and_hedge(
            corrected,
            raw_realized_scenarios,
            config,
            artifacts,
            apply_hedge=True,
            also_update_stage=True,
        )
    corrected = _apply_replay_focused_guard(corrected, config, also_update_stage=True)
    focused_shape_base_col = (
        "Pre_Focused_Guard_Forecast_MWH"
        if "Pre_Focused_Guard_Forecast_MWH" in corrected.columns
        else "Final_Backtest_Forecast_MWH"
    )
    corrected = apply_focused_shape_residual_learner(
        corrected,
        artifacts.get("focused_shape_residual_artifact"),
        config,
        forecast_col=focused_shape_base_col,
        also_update_cols=("Final_Backtest_Forecast_MWH", "Stage_Selected_Forecast_MWH"),
        update_forecast_col=focused_shape_base_col != "Pre_Focused_Guard_Forecast_MWH",
        evaluation_mode="origin_available_shadow",
    )
    corrected = apply_operational_residual_learner(
        corrected,
        artifacts.get("operational_residual_artifact"),
        config,
        forecast_col="Final_Backtest_Forecast_MWH",
        also_update_cols=("Stage_Selected_Forecast_MWH",),
        evaluation_mode="origin_available_shadow",
    )
    corrected = apply_daily_peak_shadow_model(
        corrected,
        artifacts.get("daily_peak_shadow_artifact"),
        config,
        forecast_col="Final_Backtest_Forecast_MWH",
        also_update_cols=("Stage_Selected_Forecast_MWH",),
        evaluation_mode="origin_available_shadow",
    )
    corrected = apply_hot_ramp_peak_capture(
        corrected,
        artifacts.get("hot_ramp_peak_capture_artifact"),
        config,
        forecast_col="Final_Backtest_Forecast_MWH",
        also_update_cols=("Stage_Selected_Forecast_MWH",),
        evaluation_mode="origin_available_shadow",
    )
    corrected = apply_heat_persistence_peak_capture(
        corrected,
        artifacts.get("heat_persistence_peak_capture_artifact"),
        config,
        forecast_col="Final_Backtest_Forecast_MWH",
        also_update_cols=("Stage_Selected_Forecast_MWH",),
        evaluation_mode="origin_available_shadow",
    )
    if not raw_weather_realism.empty:
        corrected_weather_realism = apply_origin_available_correction_chain(raw_weather_realism, config, artifacts)
        # V12.9: apply the same weather-uncertainty peak hedge the production pipeline applies.
        corrected_weather_realism = _apply_weather_scenarios_and_hedge(
            corrected_weather_realism,
            raw_weather_scenarios,
            config,
            artifacts,
            apply_hedge=True,
            also_update_stage=False,
        )
        corrected_weather_realism = _apply_replay_focused_guard(
            corrected_weather_realism,
            config,
            also_update_stage=False,
        )
        focused_shape_base_col = (
            "Pre_Focused_Guard_Forecast_MWH"
            if "Pre_Focused_Guard_Forecast_MWH" in corrected_weather_realism.columns
            else "Final_Backtest_Forecast_MWH"
        )
        corrected_weather_realism = apply_focused_shape_residual_learner(
            corrected_weather_realism,
            artifacts.get("focused_shape_residual_artifact"),
            config,
            forecast_col=focused_shape_base_col,
            also_update_cols=("Final_Backtest_Forecast_MWH",),
            update_forecast_col=focused_shape_base_col != "Pre_Focused_Guard_Forecast_MWH",
            evaluation_mode="weather_realism_origin_available_shadow",
        )
        corrected_weather_realism = apply_operational_residual_learner(
            corrected_weather_realism,
            artifacts.get("operational_residual_artifact"),
            config,
            forecast_col="Final_Backtest_Forecast_MWH",
            evaluation_mode="weather_realism_origin_available_shadow",
        )
        corrected_weather_realism = apply_daily_peak_shadow_model(
            corrected_weather_realism,
            artifacts.get("daily_peak_shadow_artifact"),
            config,
            forecast_col="Final_Backtest_Forecast_MWH",
            evaluation_mode="weather_realism_origin_available_shadow",
        )
        corrected_weather_realism = apply_hot_ramp_peak_capture(
            corrected_weather_realism,
            artifacts.get("hot_ramp_peak_capture_artifact"),
            config,
            forecast_col="Final_Backtest_Forecast_MWH",
            evaluation_mode="weather_realism_origin_available_shadow",
        )
        corrected_weather_realism = apply_heat_persistence_peak_capture(
            corrected_weather_realism,
            artifacts.get("heat_persistence_peak_capture_artifact"),
            config,
            forecast_col="Final_Backtest_Forecast_MWH",
            evaluation_mode="weather_realism_origin_available_shadow",
        )
        corrected = _merge_weather_realism(corrected, corrected_weather_realism)
    return corrected


def _run_single_origin_replay(args: tuple) -> tuple[pd.DataFrame | None, list[dict[str, Any]]]:
    """Run the replay process for a single origin. For use with multiprocessing."""
    (
        origin_number,
        origin_dt,
        work,
        features,
        config,
        horizon_days,
        calibration_days,
        log_timing,
        skip_catboost,
        skip_calibration_prophet,
        apply_primary_weather_hedge,
    ) = args

    timing_rows: list[dict[str, Any]] = []
    origin_started = time.perf_counter()
    print(
        "Rolling-origin replay origin "
        f"{origin_number}: {origin_dt} "
        f"({_season_from_month(origin_dt.month)})",
        flush=True,
    )
    pre_origin = work[work["DT"] < origin_dt].copy()
    stage_started = time.perf_counter()
    raw_calibration = run_rolling_backtest(
        train_df=pre_origin,
        features=features,
        ensemble_weights=_production_ensemble_weights(config),
        backtest_days=calibration_days,
        config=config,
        skip_catboost=skip_catboost,
        skip_prophet=skip_calibration_prophet,
        collect_timing=True,
    )
    calibration_detail = ";".join(
        part for part in [
            "skip_catboost=true" if skip_catboost else "",
            "skip_prophet=true" if skip_calibration_prophet else "",
        ]
        if part
    )
    _log_and_record_timing(
        timing_rows,
        origin_number=origin_number,
        origin_dt=origin_dt,
        stage="calibration backtest",
        started=stage_started,
        log_timing=log_timing,
        rows=len(raw_calibration),
        detail=calibration_detail,
    )
    calibration_timing_rows = raw_calibration.attrs.get("rolling_backtest_timing", []) if hasattr(raw_calibration, "attrs") else []
    for timing in calibration_timing_rows:
        rows_value = timing.get("Rows")
        rows = None if pd.isna(rows_value) else int(rows_value)
        _append_timing_row(
            timing_rows,
            origin_number=origin_number,
            origin_dt=origin_dt,
            stage=f"calibration backtest: {timing.get('Stage', '')}",
            elapsed_sec=float(timing.get("Elapsed_Sec", 0.0) or 0.0),
            status=str(timing.get("Status", "completed")),
            rows=rows,
            detail=str(timing.get("Detail", "") or ""),
        )
    raw_calibration.attrs = {}
    stage_started = time.perf_counter()
    artifacts = build_correction_artifacts(raw_calibration, config)
    _log_and_record_timing(
        timing_rows,
        origin_number=origin_number,
        origin_dt=origin_dt,
        stage="correction artifacts",
        started=stage_started,
        log_timing=log_timing,
        rows=len(raw_calibration),
    )
    stage_started = time.perf_counter()
    raw_origin, raw_weather_realism, raw_realized_scenarios, raw_weather_scenarios = _origin_raw_forecasts(
        work,
        features,
        config,
        origin_dt,
        horizon_days,
        origin_number,
        timing_rows=timing_rows,
    )
    _log_and_record_timing(
        timing_rows,
        origin_number=origin_number,
        origin_dt=origin_dt,
        stage="raw forecast bundle",
        started=stage_started,
        log_timing=log_timing,
        rows=len(raw_origin),
    )
    if raw_origin.empty:
        return None, timing_rows

    stage_started = time.perf_counter()
    corrected = apply_origin_correction_chain(
        raw_origin,
        raw_weather_realism,
        raw_realized_scenarios,
        raw_weather_scenarios,
        config,
        artifacts,
        apply_primary_weather_hedge=apply_primary_weather_hedge,
    )
    _log_and_record_timing(
        timing_rows,
        origin_number=origin_number,
        origin_dt=origin_dt,
        stage="correction/scenario merge",
        started=stage_started,
        log_timing=log_timing,
        rows=len(corrected),
        scenario_count=len(raw_realized_scenarios) + len(raw_weather_scenarios),
    )
    corrected["Replay_Calibration_Days"] = calibration_days
    corrected["Replay_Calibration_Start_DT"] = raw_calibration["DT"].min() if not raw_calibration.empty else pd.NaT
    corrected["Replay_Calibration_End_DT"] = raw_calibration["DT"].max() if not raw_calibration.empty else pd.NaT

    elapsed = time.perf_counter() - origin_started
    _append_timing_row(
        timing_rows,
        origin_number=origin_number,
        origin_dt=origin_dt,
        stage="origin total",
        elapsed_sec=elapsed,
        rows=len(corrected),
    )
    print(
        "Rolling-origin replay completed origin "
        f"{origin_number}: rows={len(corrected)} "
        f"elapsed={elapsed:.1f}s",
        flush=True,
    )
    return corrected, timing_rows


def run_rolling_origin_replay(train_df: pd.DataFrame, features: list[str], config: dict) -> pd.DataFrame:
    """Replay multi-origin production horizons with pre-origin correction residual windows."""
    if train_df is None or train_df.empty:
        return pd.DataFrame()

    cfg = _replay_cfg(config)
    horizon_days = _as_int(cfg.get("horizon_days"), 16)
    calibration_days = _as_int(cfg.get("calibration_days"), 45)
    log_timing = bool(cfg.get("log_timing", True))
    skip_catboost = bool(cfg.get("skip_catboost", False))
    skip_calibration_prophet = bool(cfg.get("skip_calibration_prophet", False))
    hedge_cfg = ((config.get("calibration", {}) or {}).get("weather_robustness_hedge", {}) or {})
    apply_primary_weather_hedge = bool(hedge_cfg.get("apply_to_realized_replay", True))
    work = train_df.copy().sort_values("DT").reset_index(drop=True)
    origins = _origin_candidates(work, config)

    parallel_cfg = (cfg.get("parallel", {}) or {}) if isinstance(cfg, dict) else {}
    parallel_enabled = bool(parallel_cfg.get("enabled", True))

    pool_args = [
        (
            origin_number,
            origin_dt,
            work,
            features,
            config,
            horizon_days,
            calibration_days,
            log_timing,
            skip_catboost,
            skip_calibration_prophet,
            apply_primary_weather_hedge,
        )
        for origin_number, origin_dt in enumerate(origins, start=1)
    ]

    if not parallel_enabled or len(origins) <= 1:
        print(f"Running {len(origins)} rolling-origin replays sequentially.", flush=True)
        results = [_run_single_origin_replay(arg) for arg in pool_args]
    else:
        num_processes = parallel_cfg.get("processes")
        if not isinstance(num_processes, int) or num_processes <= 0:
            try:
                num_processes = max(1, cpu_count() // 2)
            except NotImplementedError:
                num_processes = 2
        num_processes = min(num_processes, len(origins))
        print(
            f"Running {len(origins)} rolling-origin replays in parallel on {num_processes} processes...",
            flush=True,
        )
        with Pool(processes=num_processes) as pool:
            results = pool.map(_run_single_origin_replay, pool_args)

    frames = [res[0] for res in results if res is not None and res[0] is not None and not res[0].empty]
    all_timing_rows = [row for res in results if res is not None and res[1] for row in res[1]]

    result = pd.concat(frames, ignore_index=True, sort=False) if frames else pd.DataFrame()
    if all_timing_rows:
        result.attrs["rolling_origin_replay_timing"] = list(all_timing_rows)
    return result


def _daily_peak_by_origin(df: pd.DataFrame) -> pd.DataFrame:
    frames: list[pd.DataFrame] = []
    if df.empty or "Replay_Origin_ID" not in df.columns:
        return pd.DataFrame()
    for origin_id, group in df.groupby("Replay_Origin_ID", dropna=False):
        peak = build_daily_peak_miss_by_stage(group)
        if peak.empty:
            continue
        peak.insert(0, "Replay_Origin_ID", origin_id)
        peak.insert(1, "Replay_Origin_DT", group["Replay_Origin_DT"].iloc[0] if "Replay_Origin_DT" in group.columns else pd.NaT)
        frames.append(peak)
    return pd.concat(frames, ignore_index=True, sort=False) if frames else pd.DataFrame()


def _june_hot_origin_diagnostics(bt: pd.DataFrame) -> pd.DataFrame:
    """Detailed audit rows for recurring June hot/peak replay misses."""
    if bt is None or bt.empty:
        return pd.DataFrame()
    required = {"DT", "Replay_Origin_ID", "Actual_MWH", "Month", "Hour", "Temperature_DailyMax"}
    if not required.issubset(bt.columns):
        return pd.DataFrame()

    work = bt.copy()
    month = pd.to_numeric(work["Month"], errors="coerce")
    hour = pd.to_numeric(work["Hour"], errors="coerce")
    daily_max = pd.to_numeric(work["Temperature_DailyMax"], errors="coerce")
    mask = month.eq(6) & daily_max.ge(90.0) & hour.between(14, 20)
    if not mask.any():
        return pd.DataFrame()

    diag = work.loc[mask].copy()
    actual = pd.to_numeric(diag["Actual_MWH"], errors="coerce")
    forecast_cols = {
        "Raw": "Raw_Forecast_MWH",
        "XGB": "XGB_Pred_MWH",
        "LGB": "LGB_Pred_MWH",
        "CatBoost": "CatBoost_Pred_MWH",
        "Targeted_Meta": "Targeted_Meta_Adjusted_Forecast_MWH",
        "Residual_Calibrated": "Residual_Calibrated_Forecast_MWH",
        "Cloud_Solar": "Cloud_Solar_Adjusted_Forecast_MWH",
        "Peak_Risk": "Peak_Risk_Adjusted_Forecast_MWH",
        "Recent_Corrected": "Recent_Corrected_Forecast_MWH",
        "Stage_Selected": "Stage_Selected_Forecast_MWH",
        "Pre_Focused_Guard": "Pre_Focused_Guard_Forecast_MWH",
        "Post_Focused_Guard": "Post_Focused_Guard_Forecast_MWH",
        "Final_Backtest": "Final_Backtest_Forecast_MWH",
        "Final": "Final_Forecast_MWH",
    }
    for label, col in forecast_cols.items():
        if col in diag.columns:
            diag[f"{label}_Residual_MWH"] = actual - pd.to_numeric(diag[col], errors="coerce")

    stage_pairs = [
        ("Targeted_Meta_Delta_From_Raw_MWH", "Targeted_Meta_Adjusted_Forecast_MWH", "Raw_Forecast_MWH"),
        ("Residual_Cal_Delta_From_Targeted_MWH", "Residual_Calibrated_Forecast_MWH", "Targeted_Meta_Adjusted_Forecast_MWH"),
        ("Cloud_Solar_Delta_From_Residual_Cal_MWH", "Cloud_Solar_Adjusted_Forecast_MWH", "Residual_Calibrated_Forecast_MWH"),
        ("Peak_Risk_Delta_From_Cloud_Solar_MWH", "Peak_Risk_Adjusted_Forecast_MWH", "Cloud_Solar_Adjusted_Forecast_MWH"),
        ("Recent_Delta_From_Peak_Risk_MWH", "Recent_Corrected_Forecast_MWH", "Peak_Risk_Adjusted_Forecast_MWH"),
        ("Stage_Selected_Delta_From_Recent_MWH", "Stage_Selected_Forecast_MWH", "Recent_Corrected_Forecast_MWH"),
        ("Focused_Guard_Delta_From_Pre_Guard_MWH", "Post_Focused_Guard_Forecast_MWH", "Pre_Focused_Guard_Forecast_MWH"),
        ("Final_Delta_From_Raw_MWH", "Final_Forecast_MWH", "Raw_Forecast_MWH"),
    ]
    for out_col, hi_col, lo_col in stage_pairs:
        if hi_col in diag.columns and lo_col in diag.columns:
            diag[out_col] = pd.to_numeric(diag[hi_col], errors="coerce") - pd.to_numeric(diag[lo_col], errors="coerce")

    source = work.drop_duplicates(subset=["DT"], keep="first").copy()
    source["_DT_UTC"] = pd.to_datetime(source["DT"], errors="coerce", utc=True)
    source["_Month_Num"] = pd.to_numeric(source["Month"], errors="coerce")
    source["_Hour_Num"] = pd.to_numeric(source["Hour"], errors="coerce")
    source["_DailyMax_Num"] = pd.to_numeric(source["Temperature_DailyMax"], errors="coerce")
    source["_Actual_Num"] = pd.to_numeric(source["Actual_MWH"], errors="coerce")
    diag["_Replay_Origin_UTC"] = pd.to_datetime(diag.get("Replay_Origin_DT"), errors="coerce", utc=True)
    diag["_Hour_Num"] = pd.to_numeric(diag["Hour"], errors="coerce")
    diag["_DailyMax_Num"] = pd.to_numeric(diag["Temperature_DailyMax"], errors="coerce")

    analog_rows: list[dict[str, float | int]] = []
    temp_window_f = 3.0
    for _, row in diag.iterrows():
        origin_dt = row["_Replay_Origin_UTC"]
        temp = row["_DailyMax_Num"]
        row_hour = row["_Hour_Num"]
        prior = source[source["_DT_UTC"].lt(origin_dt)] if pd.notna(origin_dt) else source.iloc[0:0]
        if pd.notna(temp):
            prior = prior[
                prior["_Month_Num"].eq(6)
                & prior["_DailyMax_Num"].between(float(temp) - temp_window_f, float(temp) + temp_window_f)
            ]
        else:
            prior = prior.iloc[0:0]
        same_hour = prior[prior["_Hour_Num"].eq(row_hour)] if pd.notna(row_hour) else prior.iloc[0:0]
        peak_window = prior[prior["_Hour_Num"].between(14, 20)]

        same_values = same_hour["_Actual_Num"].dropna()
        peak_values = peak_window["_Actual_Num"].dropna()
        analog_rows.append({
            "Analog_Temp_Window_F": temp_window_f,
            "Analog_Count_SameHour_PreOrigin": int(len(same_values)),
            "Analog_Actual_Mean_SameHour_PreOrigin_MWH": float(same_values.mean()) if len(same_values) else np.nan,
            "Analog_Actual_P90_SameHour_PreOrigin_MWH": float(same_values.quantile(0.9)) if len(same_values) else np.nan,
            "Analog_Count_HE14_20_PreOrigin": int(len(peak_values)),
            "Analog_Actual_Mean_HE14_20_PreOrigin_MWH": float(peak_values.mean()) if len(peak_values) else np.nan,
            "Analog_Actual_P90_HE14_20_PreOrigin_MWH": float(peak_values.quantile(0.9)) if len(peak_values) else np.nan,
        })
    analog = pd.DataFrame(analog_rows, index=diag.index)
    diag = pd.concat(
        [diag.drop(columns=["_Replay_Origin_UTC", "_Hour_Num", "_DailyMax_Num"], errors="ignore"), analog],
        axis=1,
    )
    if "Analog_Actual_Mean_SameHour_PreOrigin_MWH" in diag.columns:
        diag["Actual_Minus_Analog_SameHour_Mean_MWH"] = (
            pd.to_numeric(diag["Actual_MWH"], errors="coerce")
            - pd.to_numeric(diag["Analog_Actual_Mean_SameHour_PreOrigin_MWH"], errors="coerce")
        )

    preferred_cols = [
        "DT", "Replay_Origin_ID", "Replay_Origin_DT", "Replay_Origin_Year", "Replay_Origin_Month",
        "Forecast_Lead_Hour", "Forecast_Day", "Replay_Horizon_Bucket", "Season", "Month", "Hour",
        "Actual_MWH", "Temperature", "Temperature_DailyMax", "CloudCover_Norm",
        "BTM_Solar_Proxy_MW", "BTM_Solar_Loss_From_ClearSky_MW", "Midday_Overcast_Solar_Loss_MW",
        "Raw_Forecast_MWH", "XGB_Pred_MWH", "LGB_Pred_MWH", "CatBoost_Pred_MWH",
        "Targeted_Meta_Adjusted_Forecast_MWH", "Residual_Calibrated_Forecast_MWH",
        "Cloud_Solar_Adjusted_Forecast_MWH", "Peak_Risk_Adjusted_Forecast_MWH",
        "Recent_Corrected_Forecast_MWH", "Stage_Selected_Forecast_MWH",
        "Pre_Focused_Guard_Forecast_MWH", "Post_Focused_Guard_Forecast_MWH",
        "Focused_Guard_Applied_Flag", "Focused_Scorecard_Guard_MWH",
        "Focused_Scorecard_Guard_Source", "Final_Backtest_Forecast_MWH", "Final_Forecast_MWH",
        "Focused_Shape_Model_Version", "Focused_Shape_Shadow_Mode",
        "Focused_Shape_Base_Forecast_MWH", "Focused_Shape_Correction_MWH",
        "Focused_Shape_Adjusted_Forecast_MWH", "Focused_Shape_Correction_Applied_Flag",
        "Focused_Shape_Source", "Focused_Shape_Evaluation_Mode",
        "Focused_Shape_Residual_MWH", "Focused_Shape_AbsError_MWH",
        "Focused_Shape_Delta_AbsError_MWH", "Focused_Shape_RuleUnion_Flag",
        "Focused_Shape_Scope_Flag",
        "Auto_Residual_Model_Version", "Auto_Residual_Shadow_Mode",
        "Auto_Residual_Production_Scope",
        "Auto_Residual_Base_Forecast_MWH", "Auto_Residual_Correction_MWH",
        "Auto_Residual_Adjusted_Forecast_MWH", "Auto_Residual_Correction_Applied_Flag",
        "Auto_Residual_Source", "Auto_Residual_Evaluation_Mode",
        "Auto_Residual_Residual_MWH", "Auto_Residual_AbsError_MWH",
        "Auto_Residual_Delta_AbsError_MWH",
        "Auto_Residual_Full_Shadow_Correction_MWH",
        "Auto_Residual_Full_Shadow_Adjusted_Forecast_MWH",
        "Auto_Residual_Full_Shadow_Correction_Applied_Flag",
        "Auto_Residual_Full_Shadow_Source",
        "Auto_Residual_Full_Shadow_Residual_MWH",
        "Auto_Residual_Full_Shadow_AbsError_MWH",
        "Auto_Residual_Full_Shadow_Delta_AbsError_MWH",
        "Auto_Residual_Structural_HotPeak_Correction_MWH",
        "Auto_Residual_Structural_HotPeak_Adjusted_Forecast_MWH",
        "Auto_Residual_Structural_HotPeak_Correction_Applied_Flag",
        "Auto_Residual_Structural_HotPeak_Source",
        "Auto_Residual_Structural_HotPeak_Residual_MWH",
        "Auto_Residual_Structural_HotPeak_AbsError_MWH",
        "Auto_Residual_Structural_HotPeak_Delta_AbsError_MWH",
        "Auto_Residual_Broad_HotPeak_Shadow_Correction_MWH",
        "Auto_Residual_Broad_HotPeak_Shadow_Adjusted_Forecast_MWH",
        "Auto_Residual_Broad_HotPeak_Shadow_Correction_Applied_Flag",
        "Auto_Residual_Broad_HotPeak_Shadow_Source",
        "Auto_Residual_Broad_HotPeak_Shadow_Residual_MWH",
        "Auto_Residual_Broad_HotPeak_Shadow_AbsError_MWH",
        "Auto_Residual_Broad_HotPeak_Shadow_Delta_AbsError_MWH",
        "Daily_Peak_Model_Version", "Daily_Peak_Shadow_Mode",
        "Daily_Peak_Base_Forecast_MWH", "Daily_Peak_Correction_MWH",
        "Daily_Peak_Shadow_Adjusted_Forecast_MWH", "Daily_Peak_Correction_Applied_Flag",
        "Daily_Peak_Source", "Daily_Peak_Evaluation_Mode",
        "Daily_Peak_Base_DailyPeak_MWH", "Daily_Peak_Predicted_Residual_MWH",
        "Daily_Peak_Predicted_DailyPeak_MWH", "Daily_Peak_Base_PeakHour",
        "Daily_Peak_Predicted_PeakHour", "Daily_Peak_Timing_Shift_Hours",
        "Daily_Peak_Residual_MWH", "Daily_Peak_AbsError_MWH",
        "Daily_Peak_Delta_AbsError_MWH",
        *HOT_RAMP_PEAK_COLUMNS,
        *HEAT_PERSISTENCE_PEAK_COLUMNS,
        "Raw_Residual_MWH", "XGB_Residual_MWH", "LGB_Residual_MWH", "CatBoost_Residual_MWH",
        "Stage_Selected_Residual_MWH", "Pre_Focused_Guard_Residual_MWH",
        "Post_Focused_Guard_Residual_MWH", "Final_Residual_MWH", "Final_AbsError_MWH",
        "Targeted_Meta_Delta_From_Raw_MWH", "Residual_Cal_Delta_From_Targeted_MWH",
        "Cloud_Solar_Delta_From_Residual_Cal_MWH", "Peak_Risk_Delta_From_Cloud_Solar_MWH",
        "Recent_Delta_From_Peak_Risk_MWH", "Stage_Selected_Delta_From_Recent_MWH",
        "Focused_Guard_Delta_From_Pre_Guard_MWH", "Final_Delta_From_Raw_MWH",
        "Stage_Selector_Source", "Stage_Selector_Reason", "Targeted_Meta_Source",
        "Cloud_Solar_Correction_Source", "Peak_Risk_Source", "Recent_Correction_Source",
        "Analog_Temp_Window_F", "Analog_Count_SameHour_PreOrigin",
        "Analog_Actual_Mean_SameHour_PreOrigin_MWH", "Analog_Actual_P90_SameHour_PreOrigin_MWH",
        "Analog_Count_HE14_20_PreOrigin", "Analog_Actual_Mean_HE14_20_PreOrigin_MWH",
        "Analog_Actual_P90_HE14_20_PreOrigin_MWH", "Actual_Minus_Analog_SameHour_Mean_MWH",
    ]
    ordered = [col for col in preferred_cols if col in diag.columns]
    extras = [col for col in diag.columns if col not in ordered]
    return diag[ordered + extras].sort_values(
        ["Replay_Origin_ID", "DT", "Hour"],
        kind="stable",
    ).reset_index(drop=True)


def _origin_coverage(df: pd.DataFrame) -> pd.DataFrame:
    if df.empty or "Replay_Origin_ID" not in df.columns:
        return pd.DataFrame()

    rows = []
    for origin_id, group in df.groupby("Replay_Origin_ID", dropna=False):
        rows.append({
            "Replay_Origin_ID": origin_id,
            "Replay_Origin_DT": group["Replay_Origin_DT"].iloc[0] if "Replay_Origin_DT" in group.columns else pd.NaT,
            "Replay_Origin_Year": group["Replay_Origin_Year"].iloc[0] if "Replay_Origin_Year" in group.columns else np.nan,
            "Replay_Origin_Month": group["Replay_Origin_Month"].iloc[0] if "Replay_Origin_Month" in group.columns else np.nan,
            "Replay_Origin_Season": group["Replay_Origin_Season"].iloc[0] if "Replay_Origin_Season" in group.columns else np.nan,
            "Scored_Start_DT": group["DT"].min() if "DT" in group.columns else pd.NaT,
            "Scored_End_DT": group["DT"].max() if "DT" in group.columns else pd.NaT,
            "Rows": int(len(group)),
            "Forecast_Days": int(pd.to_numeric(group.get("Forecast_Day"), errors="coerce").nunique()) if "Forecast_Day" in group.columns else np.nan,
            "Horizon_Buckets": "|".join(sorted(str(x) for x in group.get("Replay_Horizon_Bucket", pd.Series(dtype=object)).dropna().unique())),
            "Replay_Calibration_Days": group["Replay_Calibration_Days"].iloc[0] if "Replay_Calibration_Days" in group.columns else np.nan,
            "Replay_Calibration_Start_DT": group["Replay_Calibration_Start_DT"].iloc[0] if "Replay_Calibration_Start_DT" in group.columns else pd.NaT,
            "Replay_Calibration_End_DT": group["Replay_Calibration_End_DT"].iloc[0] if "Replay_Calibration_End_DT" in group.columns else pd.NaT,
        })
    return pd.DataFrame(rows).sort_values("Replay_Origin_DT").reset_index(drop=True)


def _ensure_origin_context(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    if "Replay_Origin_DT" not in out.columns:
        return out
    origin_dt = _local_datetime_series(out["Replay_Origin_DT"])
    if "Replay_Origin_Year" not in out.columns:
        out["Replay_Origin_Year"] = origin_dt.dt.year
    if "Replay_Origin_Month" not in out.columns:
        out["Replay_Origin_Month"] = origin_dt.dt.month
    if "Replay_Origin_Season" not in out.columns:
        out["Replay_Origin_Season"] = origin_dt.dt.month.map(
            lambda month: _season_from_month(month) if pd.notna(month) else np.nan
        )
    return out


def _scorecard_slice(df: pd.DataFrame, slice_name: str, slice_group: str, slice_value: str) -> pd.DataFrame:
    metrics = build_forecast_stage_metrics(df)
    if metrics.empty:
        return metrics
    metrics.insert(0, "Slice", slice_name)
    metrics.insert(1, "SliceGroup", slice_group)
    metrics.insert(2, "SliceValue", slice_value)
    return metrics


def _scorecard_by_values(df: pd.DataFrame, key: str, prefix: str) -> list[pd.DataFrame]:
    frames = []
    if key not in df.columns:
        return frames
    for value, group in df.groupby(key, dropna=False):
        value_label = "<missing>" if pd.isna(value) else str(value)
        frames.append(_scorecard_slice(group, f"{prefix}:{value_label}", key, value_label))
    return frames


def _dailymax_ramp_1day(bt: pd.DataFrame, daily_max_temp: pd.Series) -> pd.Series:
    if "DailyMaxTemp_Ramp_1Day" in bt.columns:
        ramp = pd.to_numeric(bt["DailyMaxTemp_Ramp_1Day"], errors="coerce")
        if ramp.notna().any():
            return ramp
    if "PriorDay_DailyMaxTemp" in bt.columns:
        prior = pd.to_numeric(bt["PriorDay_DailyMaxTemp"], errors="coerce")
        ramp = daily_max_temp - prior
        if ramp.notna().any():
            return ramp
    if "Date" not in bt.columns:
        return pd.Series(np.nan, index=bt.index)
    daily = (
        pd.DataFrame({"Date": bt["Date"].astype(str), "DailyMax": daily_max_temp}, index=bt.index)
        .groupby("Date", dropna=False)["DailyMax"]
        .max()
        .reset_index()
    )
    daily["_DateDT"] = pd.to_datetime(daily["Date"], errors="coerce")
    daily = daily.sort_values("_DateDT")
    daily["_Ramp"] = daily["DailyMax"].diff()
    return bt["Date"].astype(str).map(daily.set_index("Date")["_Ramp"]).reindex(bt.index).astype(float)


def _consecutive_extreme_days100(bt: pd.DataFrame, daily_max_temp: pd.Series) -> pd.Series:
    if "ConsecutiveExtremeHotDays100" in bt.columns:
        consecutive = pd.to_numeric(bt["ConsecutiveExtremeHotDays100"], errors="coerce")
        if consecutive.notna().any():
            return consecutive
    if "Date" not in bt.columns:
        return pd.Series(np.nan, index=bt.index)

    source = pd.DataFrame({"Date": bt["Date"].astype(str), "DailyMax": daily_max_temp}, index=bt.index)
    group_cols = ["Date"]
    if "Replay_Origin_ID" in bt.columns:
        source["Replay_Origin_ID"] = bt["Replay_Origin_ID"].astype(str)
        group_cols = ["Replay_Origin_ID", "Date"]

    daily = source.groupby(group_cols, dropna=False)["DailyMax"].max().reset_index()
    daily["_DateDT"] = pd.to_datetime(daily["Date"], errors="coerce")
    daily = daily.sort_values((["Replay_Origin_ID"] if "Replay_Origin_ID" in daily.columns else []) + ["_DateDT"])

    if "Replay_Origin_ID" in daily.columns:
        counts = []
        for _, group in daily.groupby("Replay_Origin_ID", dropna=False, sort=False):
            running = 0
            for value in pd.to_numeric(group["DailyMax"], errors="coerce"):
                running = running + 1 if pd.notna(value) and float(value) >= 100.0 else 0
                counts.append(running)
        daily["_ConsecutiveExtremeHotDays100"] = counts
        values = pd.Series(
            daily["_ConsecutiveExtremeHotDays100"].to_numpy(),
            index=pd.MultiIndex.from_frame(daily[["Replay_Origin_ID", "Date"]]),
        )
        lookup_key = pd.MultiIndex.from_arrays([source["Replay_Origin_ID"], source["Date"]])
        return pd.Series(values.reindex(lookup_key).to_numpy(), index=bt.index, dtype=float)

    running = 0
    counts = []
    for value in pd.to_numeric(daily["DailyMax"], errors="coerce"):
        running = running + 1 if pd.notna(value) and float(value) >= 100.0 else 0
        counts.append(running)
    daily["_ConsecutiveExtremeHotDays100"] = counts
    return bt["Date"].astype(str).map(daily.set_index("Date")["_ConsecutiveExtremeHotDays100"]).reindex(bt.index).astype(float)


def _event_slices(bt: pd.DataFrame) -> dict[str, pd.DataFrame]:
    daily_max_temp = (
        pd.to_numeric(bt["Temperature_DailyMax"], errors="coerce")
        if "Temperature_DailyMax" in bt.columns
        else pd.Series(np.nan, index=bt.index)
    )
    dailymax_ramp_1day = _dailymax_ramp_1day(bt, daily_max_temp)
    consecutive_extreme = _consecutive_extreme_days100(bt, daily_max_temp)
    hour = pd.to_numeric(bt.get("Hour"), errors="coerce")
    forecast_day = pd.to_numeric(bt.get("Forecast_Day"), errors="coerce")
    return {
        "PeakWindowHours14to18": bt[hour.between(14, 18)].copy(),
        "PeakWindowHours14to20": bt[hour.between(14, 20)].copy(),
        "LatePeakHours19to20": bt[hour.between(19, 20)].copy(),
        "HE18to20CodeHours17to19": bt[hour.between(17, 19)].copy(),
        "HotPeakDailyMax90Plus": bt[bt["HourGroup"].eq("Peak") & daily_max_temp.ge(90.0)].copy(),
        "HotRampPeak100PlusRamp2HE16to20": bt[
            hour.between(16, 20)
            & daily_max_temp.ge(100.0)
            & dailymax_ramp_1day.ge(2.0)
        ].copy(),
        "HeatPersistencePeak100PlusConsec3HE16to20": bt[
            hour.between(16, 20)
            & daily_max_temp.ge(100.0)
            & consecutive_extreme.ge(3.0)
        ].copy(),
        "ExtremeHeat105PlusPeakWindowHours14to20": bt[hour.between(14, 20) & daily_max_temp.ge(105.0)].copy(),
        "ShoulderSeasonHeatTransition": bt[
            bt["Season"].isin(["Spring", "Fall"])
            & hour.between(12, 22)
            & daily_max_temp.ge(75.0)
            & daily_max_temp.le(93.0)
        ].copy(),
        "CloudSolarMidday": bt[
            bt["HourGroup"].eq("Midday")
            & (
                bt["CloudCoverBucket"].isin(["Mostly Cloudy", "Overcast"])
                | bt["SolarLossBucket"].isin(["High", "Extreme"])
            )
        ].copy(),
        "WeekendHours": bt[pd.to_numeric(bt.get("IsWeekend"), errors="coerce").fillna(0).eq(1)].copy(),
        "HolidayHours": bt[pd.to_numeric(bt.get("IsHoliday"), errors="coerce").fillna(0).eq(1)].copy(),
        "LongHorizonDays8to16": bt[forecast_day.between(8, 16)].copy(),
    }


def _seasonal_scorecard(bt: pd.DataFrame, event_slices: dict[str, pd.DataFrame]) -> pd.DataFrame:
    frames = [
        _scorecard_slice(bt, "Overall", "all", "all"),
    ]
    slice_values = {
        "PeakWindowHours14to18": "peak_window",
        "PeakWindowHours14to20": "peak_window_14to20_candidate",
        "LatePeakHours19to20": "late_peak_19to20",
        "HE18to20CodeHours17to19": "he18to20_code17to19",
        "HotPeakDailyMax90Plus": "hot_peak",
        "HotRampPeak100PlusRamp2HE16to20": "hot_ramp_peak_100f_ramp2_he16to20",
        "HeatPersistencePeak100PlusConsec3HE16to20": "heat_persistence_peak_100f_consec3_he16to20",
        "ExtremeHeat105PlusPeakWindowHours14to20": "extreme_heat_105f_plus_peak",
        "ShoulderSeasonHeatTransition": "shoulder_season_heat_transition",
        "CloudSolarMidday": "cloud_solar_midday",
        "WeekendHours": "weekend",
        "HolidayHours": "holiday",
        "LongHorizonDays8to16": "days8to16",
    }
    frames.extend(
        _scorecard_slice(frame, name, "event_slice", slice_values.get(name, name))
        for name, frame in event_slices.items()
    )
    frames.extend(_scorecard_by_values(bt, "Season", "ScoredSeason"))
    frames.extend(_scorecard_by_values(bt, "Replay_Origin_Season", "OriginSeason"))
    frames.extend(_scorecard_by_values(bt, "Replay_Horizon_Bucket", "Horizon"))
    frames = [frame for frame in frames if frame is not None and not frame.empty]
    return pd.concat(frames, ignore_index=True, sort=False) if frames else pd.DataFrame()


def _weather_realism_metric_frame(bt: pd.DataFrame, use_previous_run_weather: bool) -> pd.DataFrame:
    required = "WeatherRealism_Final_Backtest_Forecast_MWH"
    if bt.empty or required not in bt.columns:
        return pd.DataFrame()
    eligible = bt[pd.to_numeric(bt[required], errors="coerce").notna()].copy()
    if eligible.empty:
        return pd.DataFrame()
    if not use_previous_run_weather:
        eligible["WeatherInputBasis"] = "realized_historical_weather"
        return eligible

    mappings = {
        "WeatherRealism_Raw_Forecast_MWH": "Raw_Forecast_MWH",
        "WeatherRealism_XGB_Pred_MWH": "XGB_Pred_MWH",
        "WeatherRealism_LGB_Pred_MWH": "LGB_Pred_MWH",
        "WeatherRealism_CatBoost_Pred_MWH": "CatBoost_Pred_MWH",
        "WeatherRealism_Prophet_Pred_MWH": "Prophet_Pred_MWH",
        "WeatherRealism_Residual_MWH": "Residual_MWH",
        "WeatherRealism_Final_Backtest_Forecast_MWH": "Final_Backtest_Forecast_MWH",
        "WeatherRealism_Final_Residual_MWH": "Final_Residual_MWH",
        "WeatherRealism_Final_AbsError_MWH": "Final_AbsError_MWH",
        "WeatherRealism_Final_APE": "Final_APE",
        "WeatherRealism_Temperature": "Temperature",
        "WeatherRealism_Temperature_DailyMax": "Temperature_DailyMax",
        "WeatherRealism_CloudCover_Norm": "CloudCover_Norm",
        "WeatherRealism_BTM_Solar_Proxy_MW": "BTM_Solar_Proxy_MW",
        "WeatherRealism_BTM_Solar_Loss_From_ClearSky_MW": "BTM_Solar_Loss_From_ClearSky_MW",
        "WeatherRealism_Midday_Overcast_Solar_Loss_MW": "Midday_Overcast_Solar_Loss_MW",
        "WeatherRealism_Weather_Robustness_Hedge_MWH": "Weather_Robustness_Hedge_MWH",
        "WeatherRealism_Weather_Robustness_Hedge_Source": "Weather_Robustness_Hedge_Source",
        "WeatherRealism_Weather_Robustness_Jensen_MWH": "Weather_Robustness_Jensen_MWH",
        "WeatherRealism_Weather_Robustness_Upper_MWH": "Weather_Robustness_Upper_MWH",
        "WeatherRealism_Weather_Robustness_Warmer_Delta_MWH": "Weather_Robustness_Warmer_Delta_MWH",
        "WeatherRealism_Weather_Robustness_Temp_Sigma_F": "Weather_Robustness_Temp_Sigma_F",
        "WeatherRealism_Weather_Robustness_Temp_Bias_Damping": "Weather_Robustness_Temp_Bias_Damping",
        "WeatherRealism_Weather_Robustness_Gate": "Weather_Robustness_Gate",
        "WeatherRealism_Pre_Focused_Guard_Forecast_MWH": "Pre_Focused_Guard_Forecast_MWH",
        "WeatherRealism_Post_Focused_Guard_Forecast_MWH": "Post_Focused_Guard_Forecast_MWH",
        "WeatherRealism_Focused_Guard_Applied_Flag": "Focused_Guard_Applied_Flag",
        "WeatherRealism_Focused_Scorecard_Guard_MWH": "Focused_Scorecard_Guard_MWH",
        "WeatherRealism_Focused_Scorecard_Guard_Source": "Focused_Scorecard_Guard_Source",
    }
    for source, target in mappings.items():
        if source in eligible.columns:
            eligible[target] = eligible[source]
    # Only the component models and the Final (stage-selected) forecast are genuinely
    # recomputed on forecast weather in the realism path. The intermediate correction
    # stages below are NOT recomputed, so reporting their (realized-weather) values under
    # the production-weather basis is misleading -- null them so the scorecard omits them.
    # Weather-independent baselines are left intact.
    not_recomputed_stage_cols = [
        "Targeted_Meta_Adjusted_Forecast_MWH",
        "Residual_Calibrated_Forecast_MWH",
        "Heat_Adjusted_Forecast_MWH",
        "Warm_Ramp_Adjusted_Forecast_MWH",
        "Cloud_Solar_Adjusted_Forecast_MWH",
        "Peak_Risk_Adjusted_Forecast_MWH",
        "Recent_Corrected_Forecast_MWH",
        "Stage_Selected_Forecast_MWH",
    ]
    for col in not_recomputed_stage_cols:
        if col in eligible.columns:
            eligible[col] = np.nan
    eligible["WeatherInputBasis"] = "previous_run_fixed_lead_weather"
    return prep_backtest(eligible)


def _weather_realism_scorecard(bt: pd.DataFrame) -> pd.DataFrame:
    frames = []
    for use_previous in [False, True]:
        frame = _weather_realism_metric_frame(bt, use_previous)
        if frame.empty:
            continue
        scorecard = _seasonal_scorecard(frame, _event_slices(frame))
        if scorecard.empty:
            continue
        scorecard.insert(0, "WeatherInputBasis", frame["WeatherInputBasis"].iloc[0])
        frames.append(scorecard)
    return pd.concat(frames, ignore_index=True, sort=False) if frames else pd.DataFrame()


def _weather_input_error_by_lead(bt: pd.DataFrame) -> pd.DataFrame:
    specs = {
        "Temperature": ("Temperature", "WeatherRealism_Temperature"),
        "Temperature_DailyMax": ("Temperature_DailyMax", "WeatherRealism_Temperature_DailyMax"),
        "CloudCover_Norm": ("CloudCover_Norm", "WeatherRealism_CloudCover_Norm"),
        "BTM_Solar_Proxy_MW": ("BTM_Solar_Proxy_MW", "WeatherRealism_BTM_Solar_Proxy_MW"),
        "BTM_Solar_Loss_From_ClearSky_MW": (
            "BTM_Solar_Loss_From_ClearSky_MW",
            "WeatherRealism_BTM_Solar_Loss_From_ClearSky_MW",
        ),
    }
    group_cols = ["Weather_Variable", "Forecast_Weather_Lead_Days", "Replay_Horizon_Bucket"]
    frames = []
    for variable, (realized_col, previous_col) in specs.items():
        lead_col = "WeatherRealism_Forecast_Weather_Lead_Days"
        if not {realized_col, previous_col, lead_col}.issubset(bt.columns):
            continue
        work = bt[[realized_col, previous_col, lead_col, "Replay_Horizon_Bucket"]].copy()
        work["Realized_Value"] = pd.to_numeric(work[realized_col], errors="coerce")
        work["Previous_Run_Value"] = pd.to_numeric(work[previous_col], errors="coerce")
        work["Forecast_Weather_Lead_Days"] = pd.to_numeric(work[lead_col], errors="coerce")
        work = work.dropna(subset=["Realized_Value", "Previous_Run_Value", "Forecast_Weather_Lead_Days"])
        if work.empty:
            continue
        work["Weather_Variable"] = variable
        work["Weather_Error"] = work["Previous_Run_Value"] - work["Realized_Value"]
        work["Abs_Weather_Error"] = work["Weather_Error"].abs()
        frames.append(work[group_cols + ["Realized_Value", "Previous_Run_Value", "Weather_Error", "Abs_Weather_Error"]])
    if not frames:
        return pd.DataFrame()
    detail = pd.concat(frames, ignore_index=True, sort=False)
    return detail.groupby(group_cols, dropna=False).agg(
        N=("Weather_Error", "size"),
        Realized_Mean=("Realized_Value", "mean"),
        Previous_Run_Mean=("Previous_Run_Value", "mean"),
        Bias_PreviousMinusRealized=("Weather_Error", "mean"),
        MAE_PreviousVsRealized=("Abs_Weather_Error", "mean"),
        P90_AbsError=("Abs_Weather_Error", lambda values: float(values.quantile(0.90))),
    ).reset_index()


def _bucket_abs(values: pd.Series, bins: list[float], labels: list[str]) -> pd.Series:
    v = pd.to_numeric(values, errors="coerce").abs()
    return pd.cut(v, bins=bins, labels=labels, include_lowest=True).astype("object")


def _weather_event_label(row: pd.Series) -> str:
    forecast_day = pd.to_numeric(pd.Series([row.get("Forecast_Day")]), errors="coerce").iloc[0]
    hour = pd.to_numeric(pd.Series([row.get("Hour")]), errors="coerce").iloc[0]
    daily_max = pd.to_numeric(pd.Series([row.get("Temperature_DailyMax")]), errors="coerce").iloc[0]
    solar_loss = pd.to_numeric(pd.Series([row.get("BTM_Solar_Loss_From_ClearSky_MW")]), errors="coerce").iloc[0]
    cloud = pd.to_numeric(pd.Series([row.get("CloudCover_Norm")]), errors="coerce").iloc[0]
    if np.isfinite(cloud) and cloud > 1.5:
        cloud = cloud / 100.0
    season = str(row.get("Season"))
    hour_group = str(row.get("HourGroup"))
    if np.isfinite(forecast_day) and forecast_day >= 8:
        return "long_horizon_days8to16"
    if hour_group == "Peak" and np.isfinite(daily_max) and daily_max >= 90.0:
        return "hot_peak"
    if hour_group == "Midday" and (
        (np.isfinite(solar_loss) and solar_loss >= 1.25) or (np.isfinite(cloud) and cloud >= 0.60)
    ):
        return "cloudy_solar_loss_midday"
    if season in {"Spring", "Fall"} and np.isfinite(hour) and 12 <= hour <= 22 and np.isfinite(daily_max) and 75.0 <= daily_max <= 93.0:
        return "shoulder_heat_transition"
    if int(pd.to_numeric(pd.Series([row.get("IsHoliday", 0)]), errors="coerce").fillna(0).iloc[0]) == 1:
        return "holiday"
    if int(pd.to_numeric(pd.Series([row.get("IsWeekend", 0)]), errors="coerce").fillna(0).iloc[0]) == 1:
        return "weekend"
    return "normal"


def _weather_error_driver(row: pd.Series) -> str:
    temp = abs(float(row.get("Weather_DailyMaxTemp_Error_F", np.nan)))
    cloud = abs(float(row.get("Weather_CloudCover_Error_Norm", np.nan)))
    solar = abs(float(row.get("Weather_Solar_Loss_Error_MW", np.nan)))
    triggers = []
    if np.isfinite(temp) and temp >= 5.0:
        triggers.append(("temp", temp / 5.0))
    if np.isfinite(cloud) and cloud >= 0.35:
        triggers.append(("cloud", cloud / 0.35))
    if np.isfinite(solar) and solar >= 3.0:
        triggers.append(("solar_loss", solar / 3.0))
    if len(triggers) >= 2:
        return "multi_weather_error"
    if len(triggers) == 1:
        return triggers[0][0]

    medium = []
    if np.isfinite(temp) and temp >= 2.0:
        medium.append(("temp_moderate", temp / 2.0))
    if np.isfinite(cloud) and cloud >= 0.20:
        medium.append(("cloud_moderate", cloud / 0.20))
    if np.isfinite(solar) and solar >= 1.25:
        medium.append(("solar_loss_moderate", solar / 1.25))
    if not medium:
        return "low_weather_error"
    return max(medium, key=lambda item: item[1])[0]


def _add_weather_input_sensitivity_columns(bt: pd.DataFrame) -> pd.DataFrame:
    out = bt.copy()
    required = {"WeatherRealism_Final_Backtest_Forecast_MWH", "Final_Backtest_Forecast_MWH"}
    if not required.issubset(out.columns):
        return out

    pairs = {
        "Weather_Temp_Error_F": ("WeatherRealism_Temperature", "Temperature"),
        "Weather_DailyMaxTemp_Error_F": ("WeatherRealism_Temperature_DailyMax", "Temperature_DailyMax"),
        "Weather_CloudCover_Error_Norm": ("WeatherRealism_CloudCover_Norm", "CloudCover_Norm"),
        "Weather_Solar_Proxy_Error_MW": ("WeatherRealism_BTM_Solar_Proxy_MW", "BTM_Solar_Proxy_MW"),
        "Weather_Solar_Loss_Error_MW": ("WeatherRealism_BTM_Solar_Loss_From_ClearSky_MW", "BTM_Solar_Loss_From_ClearSky_MW"),
        "Weather_Midday_Solar_Loss_Error_MW": ("WeatherRealism_Midday_Overcast_Solar_Loss_MW", "Midday_Overcast_Solar_Loss_MW"),
    }
    for out_col, (previous_col, realized_col) in pairs.items():
        if {previous_col, realized_col}.issubset(out.columns):
            out[out_col] = pd.to_numeric(out[previous_col], errors="coerce") - pd.to_numeric(out[realized_col], errors="coerce")

    prev_forecast = pd.to_numeric(out["WeatherRealism_Final_Backtest_Forecast_MWH"], errors="coerce")
    realized_forecast = pd.to_numeric(out["Final_Backtest_Forecast_MWH"], errors="coerce")
    actual = pd.to_numeric(out.get("Actual_MWH"), errors="coerce")
    out["Weather_Input_Forecast_Delta_MWH"] = prev_forecast - realized_forecast
    out["Weather_Input_Residual_Delta_MWH"] = (actual - prev_forecast) - (actual - realized_forecast)
    out["Weather_Input_AbsError_Delta_MWH"] = (actual - prev_forecast).abs() - (actual - realized_forecast).abs()
    out["Weather_Input_PctAbsError_Delta"] = np.where(
        actual.abs() > 1e-9,
        out["Weather_Input_AbsError_Delta_MWH"] / actual.abs() * 100.0,
        np.nan,
    )

    lead = pd.to_numeric(out.get("WeatherRealism_Forecast_Weather_Lead_Days"), errors="coerce")
    out["Weather_Forecast_Lead_Bucket"] = pd.cut(
        lead,
        bins=[-np.inf, 1, 3, 7, np.inf],
        labels=["day1", "days2to3", "days4to7", "days8plus"],
        include_lowest=True,
    ).astype("object")
    if "Weather_DailyMaxTemp_Error_F" in out.columns:
        out["Weather_DailyMaxTemp_AbsError_Bucket"] = _bucket_abs(
            out["Weather_DailyMaxTemp_Error_F"],
            [-np.inf, 2.0, 5.0, 8.0, np.inf],
            ["0-2F", "2-5F", "5-8F", "8F+"],
        )
    if "Weather_CloudCover_Error_Norm" in out.columns:
        out["Weather_CloudCover_AbsError_Bucket"] = _bucket_abs(
            out["Weather_CloudCover_Error_Norm"],
            [-np.inf, 0.20, 0.35, 0.60, np.inf],
            ["0-20pp", "20-35pp", "35-60pp", "60pp+"],
        )
    if "Weather_Solar_Loss_Error_MW" in out.columns:
        out["Weather_SolarLoss_AbsError_Bucket"] = _bucket_abs(
            out["Weather_Solar_Loss_Error_MW"],
            [-np.inf, 1.25, 3.0, 6.0, np.inf],
            ["0-1.25MW", "1.25-3MW", "3-6MW", "6MW+"],
        )

    eligible = pd.to_numeric(out["WeatherRealism_Final_Backtest_Forecast_MWH"], errors="coerce").notna()
    out.loc[eligible, "Weather_Input_Event_Slice"] = out.loc[eligible].apply(_weather_event_label, axis=1)
    out.loc[eligible, "Weather_Input_Risk_Class"] = out.loc[eligible].apply(_weather_error_driver, axis=1)
    return out


def _weather_input_sensitivity_detail(bt: pd.DataFrame) -> pd.DataFrame:
    work = _add_weather_input_sensitivity_columns(bt)
    required = "WeatherRealism_Final_Backtest_Forecast_MWH"
    if required not in work.columns:
        return pd.DataFrame()
    detail = work[pd.to_numeric(work[required], errors="coerce").notna()].copy()
    if detail.empty:
        return pd.DataFrame()
    cols = [
        "DT",
        "Replay_Origin_ID",
        "Replay_Origin_DT",
        "Replay_Origin_Season",
        "Season",
        "Forecast_Day",
        "Forecast_Lead_Hour",
        "WeatherRealism_Forecast_Weather_Lead_Days",
        "Replay_Horizon_Bucket",
        "Hour",
        "HourGroup",
        "Weather_Forecast_Lead_Bucket",
        "Weather_Input_Event_Slice",
        "Weather_Input_Risk_Class",
        "Actual_MWH",
        "Final_Backtest_Forecast_MWH",
        "WeatherRealism_Final_Backtest_Forecast_MWH",
        "Final_AbsError_MWH",
        "WeatherRealism_Final_AbsError_MWH",
        "Weather_Input_Forecast_Delta_MWH",
        "Weather_Input_Residual_Delta_MWH",
        "Weather_Input_AbsError_Delta_MWH",
        "Weather_Input_PctAbsError_Delta",
        "Temperature_DailyMax",
        "WeatherRealism_Temperature_DailyMax",
        "Weather_DailyMaxTemp_Error_F",
        "Weather_DailyMaxTemp_AbsError_Bucket",
        "CloudCover_Norm",
        "WeatherRealism_CloudCover_Norm",
        "Weather_CloudCover_Error_Norm",
        "Weather_CloudCover_AbsError_Bucket",
        "BTM_Solar_Loss_From_ClearSky_MW",
        "WeatherRealism_BTM_Solar_Loss_From_ClearSky_MW",
        "Weather_Solar_Loss_Error_MW",
        "Weather_SolarLoss_AbsError_Bucket",
    ]
    return detail[[c for c in cols if c in detail.columns]].copy()


def _weather_sensitivity_metrics(group: pd.DataFrame) -> pd.Series:
    actual = pd.to_numeric(group["Actual_MWH"], errors="coerce")
    realized_forecast = pd.to_numeric(group["Final_Backtest_Forecast_MWH"], errors="coerce")
    previous_forecast = pd.to_numeric(group["WeatherRealism_Final_Backtest_Forecast_MWH"], errors="coerce")
    mask = actual.notna() & realized_forecast.notna() & previous_forecast.notna()
    if not mask.any():
        return pd.Series(dtype=float)
    actual = actual[mask]
    realized_forecast = realized_forecast[mask]
    previous_forecast = previous_forecast[mask]
    realized_resid = actual - realized_forecast
    previous_resid = actual - previous_forecast
    realized_abs = realized_resid.abs()
    previous_abs = previous_resid.abs()
    abs_delta = previous_abs - realized_abs
    out = {
        "N": int(mask.sum()),
        "RealizedWeather_MAE_MWH": float(realized_abs.mean()),
        "PreviousRunWeather_MAE_MWH": float(previous_abs.mean()),
        "WeatherInput_MAE_Delta_MWH": float(abs_delta.mean()),
        "WeatherInput_MAPE_Delta_PCT": float(((previous_abs - realized_abs) / actual.abs() * 100.0).replace([np.inf, -np.inf], np.nan).mean()),
        "PreviousRunWeather_Bias_MWH": float(previous_resid.mean()),
        "RealizedWeather_Bias_MWH": float(realized_resid.mean()),
        "P90_WeatherInput_AbsError_Delta_MWH": float(abs_delta.quantile(0.90)),
        "WeatherInput_Harm_Rate_PCT": float((abs_delta > 0).mean() * 100.0),
        "Forecast_Delta_Bias_MWH": float((previous_forecast - realized_forecast).mean()),
        "P90_Abs_Forecast_Delta_MWH": float((previous_forecast - realized_forecast).abs().quantile(0.90)),
    }
    for col in ["Weather_DailyMaxTemp_Error_F", "Weather_CloudCover_Error_Norm", "Weather_Solar_Loss_Error_MW"]:
        if col in group.columns:
            values = pd.to_numeric(group.loc[mask, col], errors="coerce")
            out[f"Mean_{col}"] = float(values.mean())
            out[f"MAE_{col}"] = float(values.abs().mean())
            out[f"P90_Abs_{col}"] = float(values.abs().quantile(0.90))
    return pd.Series(out)


def _weather_input_sensitivity_scorecard(bt: pd.DataFrame) -> pd.DataFrame:
    detail = _weather_input_sensitivity_detail(bt)
    if detail.empty:
        return pd.DataFrame()
    frames = []

    def add_slice(name: str, group: str, value: str, frame: pd.DataFrame):
        if frame.empty:
            return
        metrics = _weather_sensitivity_metrics(frame)
        if metrics.empty:
            return
        row = metrics.to_frame().T
        row.insert(0, "Slice", name)
        row.insert(1, "SliceGroup", group)
        row.insert(2, "SliceValue", value)
        frames.append(row)

    add_slice("Overall", "all", "all", detail)
    for col, prefix in [
        ("Weather_Forecast_Lead_Bucket", "WeatherLead"),
        ("Replay_Horizon_Bucket", "Horizon"),
        ("Forecast_Day", "ForecastDay"),
        ("Season", "ScoredSeason"),
        ("Replay_Origin_Season", "OriginSeason"),
        ("HourGroup", "HourGroup"),
        ("Weather_Input_Event_Slice", "Event"),
        ("Weather_Input_Risk_Class", "RiskClass"),
        ("Weather_DailyMaxTemp_AbsError_Bucket", "DailyMaxTempError"),
        ("Weather_CloudCover_AbsError_Bucket", "CloudCoverError"),
        ("Weather_SolarLoss_AbsError_Bucket", "SolarLossError"),
    ]:
        if col not in detail.columns:
            continue
        for value, group_df in detail.groupby(col, dropna=False):
            label = "<missing>" if pd.isna(value) else str(value)
            add_slice(f"{prefix}:{label}", col, label, group_df)

    for keys, prefix in [
        (["Weather_Forecast_Lead_Bucket", "Weather_Input_Event_Slice"], "LeadEvent"),
        (["Replay_Horizon_Bucket", "Weather_Input_Risk_Class"], "HorizonRiskClass"),
        (["Season", "Weather_Input_Risk_Class"], "SeasonRiskClass"),
    ]:
        if not all(k in detail.columns for k in keys):
            continue
        for values, group_df in detail.groupby(keys, dropna=False):
            values_tuple = values if isinstance(values, tuple) else (values,)
            label = "|".join("<missing>" if pd.isna(v) else str(v) for v in values_tuple)
            add_slice(f"{prefix}:{label}", "+".join(keys), label, group_df)

    return pd.concat(frames, ignore_index=True, sort=False) if frames else pd.DataFrame()


def _scorecard_summary(bt: pd.DataFrame, coverage: pd.DataFrame, config: dict) -> dict[str, Any]:
    cfg = _replay_cfg(config)
    min_per_season = _as_int(cfg.get("scorecard_min_origins_per_season"), 2)
    seasons = ["Winter", "Spring", "Summer", "Fall"]
    season_counts = (
        coverage["Replay_Origin_Season"].value_counts(dropna=False).to_dict()
        if not coverage.empty and "Replay_Origin_Season" in coverage.columns
        else {}
    )
    by_season = {season: int(season_counts.get(season, 0)) for season in seasons}
    return {
        "origin_selection": str(cfg.get("origin_selection", "seasonal_balanced")),
        "origins_per_season_target": _as_int(cfg.get("origins_per_season"), 3),
        "scorecard_min_origins_per_season": min_per_season,
        "scorecard_ready": bool(all(count >= min_per_season for count in by_season.values())),
        "origin_count_by_season": by_season,
        "scored_seasons": sorted(str(x) for x in bt.get("Season", pd.Series(dtype=object)).dropna().unique()),
    }


def build_rolling_origin_replay_bundle(replay_df: pd.DataFrame, config: dict) -> dict[str, Any]:
    """Build focused replay diagnostics for horizon, peak, weather, and solar risk slices."""
    if replay_df is None or replay_df.empty:
        return {
            "rolling_origin_replay_summary": {
                "row_count": 0,
                "origin_count": 0,
                "weather_input_basis": "historical weather features; archived forecast weather not configured",
            }
        }

    timing_source = getattr(replay_df, "attrs", {}).get("rolling_origin_replay_timing")
    timing_df = timing_source.copy() if isinstance(timing_source, pd.DataFrame) else pd.DataFrame(timing_source or [])
    replay_work = replay_df.copy(deep=False)
    replay_work.attrs = {}
    bt = _add_weather_input_sensitivity_columns(_ensure_origin_context(prep_backtest(replay_work)))
    # Pure exclusion: configured anomalous intervals (e.g. DER dispatch hours) are not valid
    # scoring targets, so drop them from every replay scorecard (peak window, hot peak, daily
    # peak miss, etc.) to avoid spurious "misses".
    _excluded_mask = excluded_interval_mask(bt, config)
    excluded_interval_rows = int(_excluded_mask.sum())
    if excluded_interval_rows:
        bt = bt.loc[~_excluded_mask].copy()
    bt = apply_multisummer_heat_analog_shadow(bt, config=config)
    coverage = _origin_coverage(bt)
    event_slices = _event_slices(bt)
    peak_window = event_slices["PeakWindowHours14to18"]
    peak_window_14to20 = event_slices["PeakWindowHours14to20"]
    extreme_heat_peak = event_slices["ExtremeHeat105PlusPeakWindowHours14to20"]
    hot_peak = event_slices["HotPeakDailyMax90Plus"]
    hot_ramp_peak = event_slices["HotRampPeak100PlusRamp2HE16to20"]
    heat_persistence_peak = event_slices["HeatPersistencePeak100PlusConsec3HE16to20"]
    shoulder_heat = event_slices["ShoulderSeasonHeatTransition"]
    cloud_solar_midday = event_slices["CloudSolarMidday"]
    weekend = event_slices["WeekendHours"]
    holiday = event_slices["HolidayHours"]
    long_horizon = event_slices["LongHorizonDays8to16"]
    weather_realism_count = int(pd.to_numeric(
        bt.get("WeatherRealism_Final_Backtest_Forecast_MWH", pd.Series(np.nan, index=bt.index)),
        errors="coerce",
    ).notna().sum())

    summary = metrics_summary(bt)
    summary.update({
        "origin_count": int(bt["Replay_Origin_ID"].nunique()) if "Replay_Origin_ID" in bt.columns else 0,
        "horizon_buckets": sorted(str(x) for x in bt.get("Replay_Horizon_Bucket", pd.Series(dtype=object)).dropna().unique()),
        "weather_input_basis": (
            "historical weather features primary; previous-run fixed-lead forecast weather comparison available for Days1to7"
            if weather_realism_count
            else "historical weather features; previous-run forecast weather comparison unavailable"
        ),
        "weather_realism_rows": weather_realism_count,
        "weather_realism_max_previous_days": int(_weather_realism_cfg(config).get("max_previous_days", 7)),
        # Leads beyond the realism window are NOT validated against operational forecast
        # weather; their point forecasts (and the weather-uncertainty hedge sizing at those
        # leads) rest on extrapolation, not measured forecast-weather error.
        "weather_realism_validated_horizon_days": "1-{}".format(
            min(int(_weather_realism_cfg(config).get("max_previous_days", 7)),
                int(_weather_realism_cfg(config).get("provider_max_days", 7)))
        ),
        "weather_realism_unvalidated_horizon_days": "{}-{}".format(
            min(int(_weather_realism_cfg(config).get("max_previous_days", 7)),
                int(_weather_realism_cfg(config).get("provider_max_days", 7))) + 1,
            int(((config.get("forecast", {}) or {}).get("horizons", {}) or {}).get("full_days", 16)),
        ),
        "recent_residual_basis": "pre-origin correction window only",
        "excluded_interval_rows": excluded_interval_rows,
    })
    summary.update(_scorecard_summary(bt, coverage, config))
    bundle = {
        "rolling_origin_replay_summary": summary,
        "rolling_origin_replay_results": bt,
        "rolling_origin_replay_origin_coverage": coverage,
        "rolling_origin_replay_scorecard": _seasonal_scorecard(bt, event_slices),
        "rolling_origin_replay_weather_realism_scorecard": _weather_realism_scorecard(bt),
        "rolling_origin_replay_weather_input_error_by_lead": _weather_input_error_by_lead(bt),
        "rolling_origin_replay_weather_input_sensitivity_scorecard": _weather_input_sensitivity_scorecard(bt),
        "rolling_origin_replay_weather_input_sensitivity_detail": _weather_input_sensitivity_detail(bt),
        "rolling_origin_replay_stage_metrics": build_forecast_stage_metrics(bt),
        "rolling_origin_replay_origin_metrics_by_stage": build_metrics_by_group_by_stage(
            bt, ["Replay_Origin_ID", "Replay_Origin_DT", "Replay_Origin_Year", "Replay_Origin_Season", "Replay_Origin_Month"], min_count=1,
        ),
        "rolling_origin_replay_scored_season_metrics_by_stage": build_metrics_by_group_by_stage(
            bt, ["Season", "Replay_Horizon_Bucket"], min_count=1,
        ),
        "rolling_origin_replay_origin_season_metrics_by_stage": build_metrics_by_group_by_stage(
            bt, ["Replay_Origin_Season", "Replay_Horizon_Bucket"], min_count=1,
        ),
        "rolling_origin_replay_horizon_metrics_by_stage": build_metrics_by_group_by_stage(
            bt, ["Replay_Horizon_Bucket"], min_count=1,
        ),
        "rolling_origin_replay_peak_window_metrics_by_stage": build_metrics_by_group_by_stage(
            peak_window, ["Replay_Horizon_Bucket", "Hour"], min_count=1,
        ),
        "rolling_origin_replay_peak_window_bias_scorecard": build_peak_window_bias_scorecard(
            bt,
            forecast_col="Final_Backtest_Forecast_MWH",
            min_count=5,
        ),
        "rolling_origin_replay_peak_window_expansion_scorecard": build_peak_window_expansion_scorecard(bt),
        "rolling_origin_replay_peak_window_14to20_metrics_by_stage": build_peak_window_14to20_metrics_by_stage(
            peak_window_14to20,
            min_count=1,
        ),
        "rolling_origin_replay_extreme_heat_peak_scorecard": build_extreme_heat_peak_scorecard(bt),
        "rolling_origin_replay_extreme_heat_peak_metrics_by_stage": build_extreme_heat_peak_metrics_by_stage(
            extreme_heat_peak,
            min_count=1,
        ),
        "rolling_origin_replay_hot_peak_metrics_by_stage": build_metrics_by_group_by_stage(
            hot_peak, ["Replay_Horizon_Bucket", "DailyMaxTempBucket", "Hour"], min_count=1,
        ),
        "rolling_origin_replay_hot_peak_candidate_metrics_by_stage": build_metrics_by_group_by_stage(
            hot_peak, ["Replay_Horizon_Bucket", "Month", "DailyMaxTempBucket", "CloudCoverBucket"], min_count=1,
        ),
        "rolling_origin_replay_hot_peak_candidate_scorecard": build_hot_peak_shadow_candidate_scorecard(
            hot_peak,
            group_cols=["Replay_Horizon_Bucket", "Month", "DailyMaxTempBucket", "CloudCoverBucket"],
            min_count=1,
        ),
        "rolling_origin_replay_hot_ramp_peak_metrics_by_stage": build_metrics_by_group_by_stage(
            hot_ramp_peak, ["Replay_Horizon_Bucket", "Month", "DailyMaxTempBucket", "CloudCoverBucket"], min_count=1,
        ),
        "rolling_origin_replay_hot_ramp_peak_candidate_scorecard": build_hot_ramp_peak_candidate_scorecard(
            hot_ramp_peak,
            group_cols=["Replay_Horizon_Bucket", "Month", "DailyMaxTempBucket", "CloudCoverBucket"],
            min_count=1,
        ),
        "rolling_origin_replay_heat_persistence_peak_metrics_by_stage": build_metrics_by_group_by_stage(
            heat_persistence_peak,
            ["Replay_Horizon_Bucket", "Month", "DailyMaxTempBucket", "CloudCoverBucket"],
            min_count=1,
        ),
        "rolling_origin_replay_heat_persistence_peak_candidate_scorecard": build_heat_persistence_peak_candidate_scorecard(
            heat_persistence_peak,
            group_cols=["Replay_Horizon_Bucket", "Month", "DailyMaxTempBucket", "CloudCoverBucket"],
            min_count=1,
        ),
        "rolling_origin_replay_shoulder_heat_metrics_by_stage": build_metrics_by_group_by_stage(
            shoulder_heat, ["Replay_Horizon_Bucket", "Season", "DailyMaxTempBucket", "HourGroup"], min_count=1,
        ),
        "rolling_origin_replay_cloud_solar_midday_metrics_by_stage": build_metrics_by_group_by_stage(
            cloud_solar_midday, ["Replay_Horizon_Bucket", "CloudCoverBucket", "SolarLossBucket", "Hour"], min_count=1,
        ),
        "rolling_origin_replay_weekend_metrics_by_stage": build_metrics_by_group_by_stage(
            weekend, ["Replay_Horizon_Bucket", "Season", "HourGroup"], min_count=1,
        ),
        "rolling_origin_replay_holiday_metrics_by_stage": build_metrics_by_group_by_stage(
            holiday, ["Replay_Horizon_Bucket", "Season", "HourGroup"], min_count=1,
        ),
        "rolling_origin_replay_long_horizon_metrics_by_stage": build_metrics_by_group_by_stage(
            long_horizon, ["Season", "HourGroup", "DailyMaxTempBucket"], min_count=1,
        ),
        "rolling_origin_replay_delta_breeze_shape_metrics_by_stage": build_delta_breeze_shape_metrics_by_stage(bt, min_count=1),
        "rolling_origin_replay_daily_peak_shadow_window_scorecard": build_daily_peak_shadow_window_scorecard(bt, config=config),
        "rolling_origin_replay_daily_peak_miss_by_stage": _daily_peak_by_origin(bt),
        "rolling_origin_replay_daily_peak_he18_20_miss_by_stage": build_daily_peak_window_miss_by_stage(bt),
        "rolling_origin_replay_heat_analog_shadow_metrics": build_heat_analog_shadow_metrics(bt, config=config, min_count=1),
        "rolling_origin_replay_heat_analog_shadow_detail": build_heat_analog_shadow_detail(bt, config=config),
        "rolling_origin_replay_focused_guard_rule_audit": build_focused_scorecard_rule_audit(
            bt,
            config,
            forecast_col="Pre_Focused_Guard_Forecast_MWH",
        ),
        "focused_shape_residual_summary": focused_shape_residual_summary(
            bt,
            None,
            config,
        ),
        "daily_peak_shadow_summary": daily_peak_shadow_summary(
            bt,
            None,
            config,
        ),
        "hot_ramp_peak_capture_summary": hot_ramp_peak_capture_summary(
            bt,
            None,
            config,
        ),
        "heat_persistence_peak_capture_summary": heat_persistence_peak_capture_summary(
            bt,
            None,
            config,
        ),
        "june_hot_origin_diagnostics": _june_hot_origin_diagnostics(bt),
    }
    if not timing_df.empty:
        bundle["rolling_origin_replay_timing"] = timing_df
    return bundle
