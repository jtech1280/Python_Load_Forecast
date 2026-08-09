from __future__ import annotations

import unittest

import numpy as np
import pandas as pd

from forecasting.backtest.rolling_backtest import (
    PRED_COLS as ROLLING_BACKTEST_PRED_COLS,
)
from forecasting.backtest.rolling_origin_replay import (
    PRED_COLS as ROLLING_REPLAY_PRED_COLS,
)
from forecasting.data.weather_loader import _finalize_weather_frame, _normalize_hourly
from forecasting.diagnostics.forecast_diagnostics import (
    build_delta_breeze_shape_metrics_by_stage,
    build_peak_window_bias_scorecard,
)
from forecasting.features.lag_features import add_basic_lags
from forecasting.features.time_features import add_time_features
from forecasting.features.weather_features import add_weather_features
from forecasting.forecast.recursive_engine import LOAD_DECAY_SHAPE_FEATURES
from forecasting.model.prophet_model import DEFAULT_PROPHET_REGRESSORS
from forecasting.model.xgb_model import DEFAULT_FEATURES


class DeltaBreezeFeatureTests(unittest.TestCase):
    def test_weather_features_capture_clear_hot_westerly_evening_cooling_shape(self):
        df = pd.DataFrame(
            {
                "DT": pd.date_range(
                    "2026-07-28 16:00", periods=8, freq="h", tz="America/Los_Angeles"
                ),
                "TempF": [99.0, 100.0, 97.0, 93.0, 88.0, 84.0, 82.0, 80.0],
                "HumidityPct": [20.0] * 8,
                "CloudCoverPct": [10.0] * 8,
                "WindSpeedMph": [3.0, 4.0, 5.0, 8.0, 10.0, 12.0, 9.0, 8.0],
                "WindDirectionDeg": [270.0] * 8,
                "PrecipIn": [0.0] * 8,
                "GHI_Wm2": [700.0, 500.0, 250.0, 50.0, 0.0, 0.0, 0.0, 0.0],
            }
        )
        out = add_weather_features(add_time_features(df))
        hour17 = out.loc[out["Hour"].eq(17)].iloc[0]
        hour20 = out.loc[out["Hour"].eq(20)].iloc[0]

        self.assertEqual(hour17["IsPeakWindow16to18"], 1.0)
        self.assertEqual(hour17["ClearPeakWindow16to18"], 1.0)
        self.assertEqual(hour17["ClearHotPeakWindow16to18"], 1.0)
        self.assertEqual(hour17["ClearPeakHE16to18DailyMax100to105"], 1.0)
        self.assertAlmostEqual(hour17["CloudCover_x_PeakWindow16to18"], 0.10)
        self.assertEqual(hour20["WindDirection_Deg"], 270.0)
        self.assertAlmostEqual(hour20["Westerly_Flow_Mph"], 10.0)
        self.assertEqual(hour20["Westerly_Flow_Flag"], 1.0)
        self.assertEqual(hour20["ClearHotEvening_Flag"], 1.0)
        self.assertEqual(hour20["ClearHotPeakWindow16to20"], 1.0)
        self.assertEqual(hour20["OvercastHotPeakWindow16to20"], 0.0)
        self.assertAlmostEqual(hour20["CloudCover_x_HotPeakWindow16to20"], 0.10)
        self.assertEqual(hour20["ClearHotPeakDailyMax100to105"], 1.0)
        self.assertAlmostEqual(hour20["ClearHotPeak_x_DailyMaxExcess90"], 10.0)
        self.assertAlmostEqual(hour20["Month_x_HotPeak"], 7.0)
        self.assertAlmostEqual(hour20["Month_x_ClearHotPeak"], 7.0)
        self.assertEqual(hour20["ClearHotPeak_x_HE20"], 1.0)
        self.assertEqual(hour20["DeltaBreeze_Westerly_Flow_Flag"], 1.0)
        self.assertEqual(hour20["DeltaBreeze_Cooling_Flag"], 1.0)
        self.assertAlmostEqual(hour20["Temperature_Drop_From_DailyMax_F"], 12.0)
        self.assertAlmostEqual(hour20["TempDrop_3Hr_F"], 12.0)
        self.assertGreater(hour20["DeltaBreeze_Cooling_Signal"], 0.0)

    def test_non_westerly_direction_does_not_trigger_directional_delta_breeze_flag(
        self,
    ):
        df = pd.DataFrame(
            {
                "DT": pd.date_range(
                    "2026-07-28 18:00", periods=4, freq="h", tz="America/Los_Angeles"
                ),
                "TempF": [100.0, 94.0, 88.0, 84.0],
                "HumidityPct": [20.0] * 4,
                "CloudCoverPct": [5.0] * 4,
                "WindSpeedMph": [10.0] * 4,
                "WindDirectionDeg": [90.0] * 4,
                "PrecipIn": [0.0] * 4,
                "GHI_Wm2": [100.0, 0.0, 0.0, 0.0],
            }
        )
        out = add_weather_features(add_time_features(df))

        self.assertTrue((out["Westerly_Flow_Mph"] == 0.0).all())
        self.assertTrue((out["Westerly_Flow_Flag"] == 0.0).all())
        self.assertTrue((out["DeltaBreeze_Westerly_Flow_Flag"] == 0.0).all())
        self.assertGreater(out["DeltaBreeze_CoolingNoDirection_Signal"].max(), 0.0)

    def test_weather_features_capture_overcast_mild_hot_peak_slice(self):
        df = pd.DataFrame(
            {
                "DT": pd.date_range(
                    "2026-07-20 16:00", periods=5, freq="h", tz="America/Los_Angeles"
                ),
                "TempF": [91.0, 91.5, 91.0, 90.5, 90.0],
                "HumidityPct": [50.0] * 5,
                "CloudCoverPct": [80.0] * 5,
                "WindSpeedMph": [3.0] * 5,
                "WindDirectionDeg": [180.0] * 5,
                "PrecipIn": [0.0] * 5,
                "GHI_Wm2": [250.0, 150.0, 50.0, 0.0, 0.0],
            }
        )
        out = add_weather_features(add_time_features(df))
        row = out.loc[out["Hour"].eq(17)].iloc[0]

        self.assertEqual(row["OvercastHotPeakWindow16to20"], 1.0)
        self.assertEqual(row["OvercastHotPeakDailyMax90to92_5"], 1.0)
        self.assertEqual(row["ClearHotPeakWindow16to20"], 0.0)
        self.assertAlmostEqual(row["CloudCover_x_HotPeakWindow16to20"], 0.80)
        self.assertAlmostEqual(row["Month_x_OvercastHotPeak"], 7.0)
        self.assertEqual(row["IsPeakWindow16to18"], 1.0)
        self.assertEqual(row["OvercastPeakWindow16to18"], 1.0)
        self.assertEqual(row["OvercastPeakHE16to18DailyMax90to92_5"], 1.0)
        self.assertAlmostEqual(row["Month_x_OvercastPeakHE16to18DailyMax90to92_5"], 7.0)

    def test_weather_features_capture_non_hot_overcast_peak_window_slices(self):
        df = pd.DataFrame(
            {
                "DT": list(
                    pd.date_range(
                        "2026-11-10 14:00",
                        periods=5,
                        freq="h",
                        tz="America/Los_Angeles",
                    )
                )
                + list(
                    pd.date_range(
                        "2026-11-11 14:00",
                        periods=5,
                        freq="h",
                        tz="America/Los_Angeles",
                    )
                ),
                "TempF": [70.0, 71.0, 72.0, 71.0, 70.0, 80.0, 81.0, 82.0, 81.0, 80.0],
                "HumidityPct": [60.0] * 10,
                "CloudCoverPct": [80.0] * 10,
                "WindSpeedMph": [4.0] * 10,
                "WindDirectionDeg": [180.0] * 10,
                "PrecipIn": [0.0] * 10,
                "GHI_Wm2": [200.0, 150.0, 100.0, 50.0, 0.0] * 2,
            }
        )
        out = add_weather_features(add_time_features(df))
        below75 = out.loc[
            out["DT"].eq(pd.Timestamp("2026-11-10 15:00", tz="America/Los_Angeles"))
        ].iloc[0]
        below75_he16 = out.loc[
            out["DT"].eq(pd.Timestamp("2026-11-10 16:00", tz="America/Los_Angeles"))
        ].iloc[0]
        band75to85 = out.loc[
            out["DT"].eq(pd.Timestamp("2026-11-11 15:00", tz="America/Los_Angeles"))
        ].iloc[0]

        self.assertEqual(below75["OvercastPeakWindow14to18"], 1.0)
        self.assertEqual(below75["PeakWindowDailyMaxBelow75"], 1.0)
        self.assertEqual(below75["OvercastPeakDailyMaxBelow75"], 1.0)
        self.assertAlmostEqual(below75["Month_x_OvercastPeakWindow14to18"], 11.0)
        self.assertEqual(below75_he16["IsPeakWindow16to18"], 1.0)
        self.assertEqual(below75_he16["OvercastPeakWindow16to18"], 1.0)
        self.assertEqual(below75_he16["OvercastCoolPeakWindow16to18"], 1.0)
        self.assertEqual(below75_he16["OvercastPeakHE16to18DailyMaxBelow75"], 1.0)
        self.assertAlmostEqual(
            below75_he16["Month_x_OvercastPeakHE16to18DailyMaxBelow75"], 11.0
        )
        self.assertEqual(band75to85["PeakWindowDailyMax75to85"], 1.0)
        self.assertEqual(band75to85["OvercastPeakDailyMax75to85"], 1.0)

    def test_post_peak_load_decay_features_use_lagged_load_only(self):
        dt_index = pd.date_range(
            "2026-07-01 00:00", periods=72, freq="h", tz="America/Los_Angeles"
        )
        mwh = np.full(72, 250.0)
        target_idx = 68  # 2026-07-03 20:00 local
        mwh[target_idx - 3] = 280.0
        mwh[target_idx - 2] = 270.0
        mwh[target_idx - 1] = 260.0
        mwh[target_idx - 24] = 275.0
        df = pd.DataFrame(
            {
                "DT": dt_index,
                "MWH": mwh,
                "TempF": [
                    100.0 if ts.hour == 17 else 85.0 if ts.hour >= 20 else 95.0
                    for ts in dt_index
                ],
                "HumidityPct": [20.0] * len(dt_index),
                "CloudCoverPct": [10.0] * len(dt_index),
                "WindSpeedMph": [8.0] * len(dt_index),
                "WindDirectionDeg": [270.0] * len(dt_index),
                "PrecipIn": [0.0] * len(dt_index),
                "GHI_Wm2": [0.0] * len(dt_index),
            }
        )
        out = add_basic_lags(add_weather_features(add_time_features(df)))
        row = out.loc[out["DT"].eq(dt_index[target_idx])].iloc[0]
        he17 = out.loc[
            out["DT"].eq(pd.Timestamp("2026-07-03 17:00", tz="America/Los_Angeles"))
        ].iloc[0]

        self.assertAlmostEqual(row["Load_Decay_1Hr_MWH"], 10.0)
        self.assertAlmostEqual(row["Load_Decay_2Hr_MWH"], 20.0)
        self.assertEqual(row["ClearHotPeakWindow16to20"], 1.0)
        self.assertAlmostEqual(
            row["Lag24_Minus_SameHour7DayMean_MWH"],
            row["MWH_Lag24"] - row["MWH_SameHour7DayMean"],
        )
        self.assertAlmostEqual(
            row["HotPeak_Lag1_Minus_SameHour7DayMean_MWH"],
            row["Lag1_Minus_SameHour7DayMean_MWH"],
        )
        self.assertAlmostEqual(
            row["ClearHotPeak_Lag24_Minus_SameHour7DayMean_MWH"],
            row["Lag24_Minus_SameHour7DayMean_MWH"],
        )
        self.assertEqual(he17["IsPeakWindow16to18"], 1.0)
        self.assertEqual(he17["ClearHotPeakWindow16to18"], 1.0)
        self.assertAlmostEqual(
            he17["PeakWindow16to18_Lag24_Minus_SameHour7DayMean_MWH"],
            he17["Lag24_Minus_SameHour7DayMean_MWH"],
        )
        self.assertAlmostEqual(
            he17["ClearHotPeak16to18_Lag24_Minus_SameHour7DayMean_MWH"],
            he17["Lag24_Minus_SameHour7DayMean_MWH"],
        )
        self.assertAlmostEqual(row["PostPeak_LoadDecay_VsSameHourYesterday_MWH"], 15.0)
        self.assertAlmostEqual(
            row["PostPeak_LoadDecay_VsSameHour7DayMean_MWH"],
            row["MWH_SameHour7DayMean"] - row["MWH_Lag1"],
        )
        self.assertGreater(row["DeltaBreeze_PostPeak_LoadDecay_Signal"], 0.0)

    def test_weather_loader_normalizes_wind_direction(self):
        payload = {
            "hourly": {
                "time": ["2026-07-28T20:00"],
                "temperature_2m": [88.0],
                "relative_humidity_2m": [20.0],
                "cloud_cover": [10.0],
                "wind_speed_10m": [12.0],
                "wind_direction_10m": [270.0],
                "precipitation": [0.0],
                "shortwave_radiation": [0.0],
                "is_day": [0],
            }
        }
        config = {
            "project": {"timezone": "America/Los_Angeles"},
            "openmeteo": {"timezone": "America/Los_Angeles"},
            "quality": {
                "valid_temp_min_f": -50.0,
                "valid_temp_max_f": 130.0,
                "weather_timestamp_shift_hours": 0,
                "max_interpolation_gap_hours": 0,
            },
        }

        out = _finalize_weather_frame(
            _normalize_hourly(payload, pd.Timestamp("2026-07-28T12:00:00Z")), config
        )

        self.assertIn("WindDirectionDeg", out.columns)
        self.assertEqual(out.loc[0, "WindDirectionDeg"], 270.0)

    def test_delta_breeze_shape_diagnostics_score_stages_on_shape_slices(self):
        df = pd.DataFrame(
            {
                "DT": pd.date_range(
                    "2026-07-28 18:00", periods=3, freq="h", tz="America/Los_Angeles"
                ),
                "Actual_MWH": [280.0, 260.0, 240.0],
                "Raw_Forecast_MWH": [285.0, 275.0, 260.0],
                "Final_Backtest_Forecast_MWH": [282.0, 266.0, 246.0],
                "Hour": [18, 19, 20],
                "Temperature_DailyMax": [100.0, 100.0, 100.0],
                "CloudCover_Norm": [0.05, 0.05, 0.05],
                "WindDirection_Available_Flag": [1.0, 1.0, 1.0],
                "Westerly_Flow_Flag": [1.0, 1.0, 1.0],
                "Westerly_Flow_Mph": [6.0, 8.0, 10.0],
                "ClearHotEvening_Flag": [1.0, 1.0, 1.0],
                "Temperature_Drop_From_DailyMax_F": [3.0, 7.0, 12.0],
                "TempDrop_Next3Hr_F": [10.0, 8.0, 0.0],
                "DeltaBreeze_Cooling_Flag": [1.0, 1.0, 1.0],
                "DeltaBreeze_Westerly_Flow_Flag": [1.0, 1.0, 1.0],
                "PostPeak_LoadDecay_VsSameHour7DayMean_MWH": [0.0, 8.0, 15.0],
            }
        )

        out = build_delta_breeze_shape_metrics_by_stage(df, min_count=1)

        self.assertFalse(out.empty)
        self.assertIn("clear_hot_evening_westerly", set(out["Slice"]))
        self.assertIn("raw_xgb_lgb_production", set(out["Stage"]))
        self.assertIn("final_corrected_production", set(out["Stage"]))
        self.assertIn("Mean_Westerly_Flow_Mph", out.columns)

    def test_peak_window_bias_scorecard_ranks_by_residual_sum(self):
        rows = []
        for hour in [14, 15, 16, 17, 18]:
            rows.append(
                {
                    "DT": pd.Timestamp("2026-11-10") + pd.Timedelta(hours=hour),
                    "Month": 11,
                    "Forecast_Day": 10,
                    "Hour": hour,
                    "Actual_MWH": 210.0,
                    "Final_Backtest_Forecast_MWH": 200.0,
                    "Raw_Forecast_MWH": 201.0,
                    "MWH_SameHour7DayMean": 200.0,
                    "Lag24_Minus_SameHour7DayMean_MWH": 2.0,
                    "Temperature_DailyMax": 72.0,
                    "CloudCover_Norm": 0.80,
                }
            )
        for hour in [14, 15, 16]:
            rows.append(
                {
                    "DT": pd.Timestamp("2026-06-10") + pd.Timedelta(hours=hour),
                    "Month": 6,
                    "Forecast_Day": 10,
                    "Hour": hour,
                    "Actual_MWH": 190.0,
                    "Final_Backtest_Forecast_MWH": 200.0,
                    "Raw_Forecast_MWH": 199.0,
                    "MWH_SameHour7DayMean": 210.0,
                    "Lag24_Minus_SameHour7DayMean_MWH": -12.0,
                    "Temperature_DailyMax": 88.0,
                    "CloudCover_Norm": 0.05,
                }
            )

        scorecard = build_peak_window_bias_scorecard(pd.DataFrame(rows), min_count=2)

        self.assertFalse(scorecard.empty)
        top = scorecard.iloc[0]
        self.assertEqual(top["Month"], 11)
        self.assertEqual(top["Forecast_Day_Bucket"], "Days8-16")
        self.assertEqual(top["Peak_Hour_Band"], "HE16-18")
        self.assertEqual(top["DailyMaxTempBiasBand"], "<75")
        self.assertEqual(top["CloudCoverBiasBand"], "Overcast")
        self.assertEqual(top["LagAnchorState"], "-10..10")
        self.assertAlmostEqual(top["Residual_Sum_MWH"], 30.0)
        self.assertEqual(top["RankBasis"], "Residual_Sum_MWH_desc")

    def test_model_and_replay_feature_lists_include_delta_breeze_features(self):
        tree_required = {
            "WindDirection_Deg",
            "Westerly_Flow_Mph",
            "TempDrop_Next3Hr_F",
            "ClearHotEvening_x_WesterlyFlow",
            "DeltaBreeze_Cooling_Signal",
            "ClearHotPeakWindow16to20",
            "ClearHotPeakWindow16to18",
            "OvercastHotPeakWindow16to20",
            "CloudCover_x_HotPeakWindow16to20",
            "IsPeakWindow16to18",
            "CloudCover_x_PeakWindow16to18",
            "OvercastPeakWindow14to18",
            "OvercastPeakWindow16to18",
            "PeakWindowDailyMaxBelow75",
            "OvercastPeakDailyMaxBelow75",
            "OvercastPeakDailyMax75to85",
            "OvercastCoolPeakWindow16to18",
            "OvercastPeakHE16to18DailyMaxBelow75",
            "OvercastPeakHE16to18DailyMax90to92_5",
            "ClearPeakHE16to18DailyMax85to90",
            "ClearPeakHE16to18DailyMax100to105",
            "ClearPeakHE16to18DailyMax105Plus",
            "Month_x_OvercastPeakWindow14to18",
            "Month_x_OvercastPeakHE16to18DailyMaxBelow75",
            "Month_x_ClearPeakHE16to18DailyMax100to105",
            "ClearHotPeakDailyMax98to100",
            "ClearHotPeakDailyMax105Plus",
            "OvercastHotPeakDailyMax90to92_5",
            "HotPeakDailyMax105Plus",
            "ClearHotPeak_x_DailyMaxExcess90",
            "OvercastHotPeak_x_DailyMaxExcess90",
            "Month_x_HotPeak",
            "Month_x_OvercastHotPeak",
            "ClearHotPeak_x_HE18",
            "Lag24_Minus_SameHour7DayMean_MWH",
            "PeakWindow_Lag24_Minus_SameHour7DayMean_MWH",
            "PeakWindow16to18_Lag24_Minus_SameHour7DayMean_MWH",
            "ClearHotPeak_Lag24_Minus_SameHour7DayMean_MWH",
            "ClearHotPeak16to18_Lag24_Minus_SameHour7DayMean_MWH",
            "OvercastCoolPeak16to18_Lag24_Minus_SameHour7DayMean_MWH",
            "PostPeak_LoadDecay_VsSameHour7DayMean_MWH",
            "DeltaBreeze_PostPeak_LoadDecay_Signal",
        }
        prophet_excluded = tree_required - {
            "WindDirection_Deg",
            "PostPeak_LoadDecay_VsSameHour7DayMean_MWH",
            "DeltaBreeze_PostPeak_LoadDecay_Signal",
        }
        pred_required = {
            "Load_Decay_1Hr_MWH",
            "Lag24_Minus_SameHour7DayMean_MWH",
            "HotPeak_Lag1_Minus_SameHour7DayMean_MWH",
            "PeakWindow_Lag24_Minus_SameHour7DayMean_MWH",
            "PeakWindow16to18_Lag24_Minus_SameHour7DayMean_MWH",
            "ClearHotPeak_Lag24_Minus_SameHour7DayMean_MWH",
            "ClearHotPeak16to18_Lag24_Minus_SameHour7DayMean_MWH",
            "OvercastCoolPeak16to18_Lag24_Minus_SameHour7DayMean_MWH",
            "PostPeak_LoadDecay_VsSameHourYesterday_MWH",
            "PostPeak_LoadDecay_VsSameHour7DayMean_MWH",
            "DeltaBreeze_PostPeak_LoadDecay_Signal",
        }

        self.assertTrue(tree_required.issubset(set(DEFAULT_FEATURES)))
        self.assertFalse(prophet_excluded.intersection(set(DEFAULT_PROPHET_REGRESSORS)))
        self.assertTrue(pred_required.issubset(set(LOAD_DECAY_SHAPE_FEATURES)))
        self.assertTrue(pred_required.issubset(set(ROLLING_BACKTEST_PRED_COLS)))
        self.assertTrue(pred_required.issubset(set(ROLLING_REPLAY_PRED_COLS)))


if __name__ == "__main__":
    unittest.main()
