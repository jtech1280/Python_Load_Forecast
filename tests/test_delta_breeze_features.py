from __future__ import annotations

import unittest

import numpy as np
import pandas as pd

from forecasting.backtest.rolling_backtest import PRED_COLS as ROLLING_BACKTEST_PRED_COLS
from forecasting.backtest.rolling_origin_replay import PRED_COLS as ROLLING_REPLAY_PRED_COLS
from forecasting.data.weather_loader import _finalize_weather_frame, _normalize_hourly
from forecasting.diagnostics.forecast_diagnostics import build_delta_breeze_shape_metrics_by_stage
from forecasting.features.lag_features import add_basic_lags
from forecasting.features.time_features import add_time_features
from forecasting.features.weather_features import add_weather_features
from forecasting.model.prophet_model import DEFAULT_PROPHET_REGRESSORS
from forecasting.model.xgb_model import DEFAULT_FEATURES


class DeltaBreezeFeatureTests(unittest.TestCase):
    def test_weather_features_capture_clear_hot_westerly_evening_cooling_shape(self):
        df = pd.DataFrame(
            {
                "DT": pd.date_range("2026-07-28 16:00", periods=8, freq="h", tz="America/Los_Angeles"),
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
        hour20 = out.loc[out["Hour"].eq(20)].iloc[0]

        self.assertEqual(hour20["WindDirection_Deg"], 270.0)
        self.assertAlmostEqual(hour20["Westerly_Flow_Mph"], 10.0)
        self.assertEqual(hour20["Westerly_Flow_Flag"], 1.0)
        self.assertEqual(hour20["ClearHotEvening_Flag"], 1.0)
        self.assertEqual(hour20["DeltaBreeze_Westerly_Flow_Flag"], 1.0)
        self.assertEqual(hour20["DeltaBreeze_Cooling_Flag"], 1.0)
        self.assertAlmostEqual(hour20["Temperature_Drop_From_DailyMax_F"], 12.0)
        self.assertAlmostEqual(hour20["TempDrop_3Hr_F"], 12.0)
        self.assertGreater(hour20["DeltaBreeze_Cooling_Signal"], 0.0)

    def test_non_westerly_direction_does_not_trigger_directional_delta_breeze_flag(self):
        df = pd.DataFrame(
            {
                "DT": pd.date_range("2026-07-28 18:00", periods=4, freq="h", tz="America/Los_Angeles"),
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

    def test_post_peak_load_decay_features_use_lagged_load_only(self):
        dt_index = pd.date_range("2026-07-01 00:00", periods=72, freq="h", tz="America/Los_Angeles")
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
                "TempF": [100.0 if ts.hour == 17 else 85.0 if ts.hour >= 20 else 95.0 for ts in dt_index],
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

        self.assertAlmostEqual(row["Load_Decay_1Hr_MWH"], 10.0)
        self.assertAlmostEqual(row["Load_Decay_2Hr_MWH"], 20.0)
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

        out = _finalize_weather_frame(_normalize_hourly(payload, pd.Timestamp("2026-07-28T12:00:00Z")), config)

        self.assertIn("WindDirectionDeg", out.columns)
        self.assertEqual(out.loc[0, "WindDirectionDeg"], 270.0)

    def test_delta_breeze_shape_diagnostics_score_stages_on_shape_slices(self):
        df = pd.DataFrame(
            {
                "DT": pd.date_range("2026-07-28 18:00", periods=3, freq="h", tz="America/Los_Angeles"),
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

    def test_model_and_replay_feature_lists_include_delta_breeze_features(self):
        tree_required = {
            "WindDirection_Deg",
            "Westerly_Flow_Mph",
            "TempDrop_Next3Hr_F",
            "ClearHotEvening_x_WesterlyFlow",
            "DeltaBreeze_Cooling_Signal",
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
            "PostPeak_LoadDecay_VsSameHourYesterday_MWH",
            "PostPeak_LoadDecay_VsSameHour7DayMean_MWH",
            "DeltaBreeze_PostPeak_LoadDecay_Signal",
        }

        self.assertTrue(tree_required.issubset(set(DEFAULT_FEATURES)))
        self.assertFalse(prophet_excluded.intersection(set(DEFAULT_PROPHET_REGRESSORS)))
        self.assertTrue(pred_required.issubset(set(ROLLING_BACKTEST_PRED_COLS)))
        self.assertTrue(pred_required.issubset(set(ROLLING_REPLAY_PRED_COLS)))


if __name__ == "__main__":
    unittest.main()
