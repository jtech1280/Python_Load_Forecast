import unittest

from forecasting.main import _disable_windows_platform_wmi_probe

_disable_windows_platform_wmi_probe()

import numpy as np
import pandas as pd

from forecasting.features.solar_features import add_solar_features
from forecasting.backtest.rolling_backtest import PRED_COLS as ROLLING_BACKTEST_PRED_COLS
from forecasting.backtest.rolling_origin_replay import PRED_COLS as ROLLING_REPLAY_PRED_COLS, _apply_replay_focused_guard
from forecasting.forecast.focused_scorecard_guard import apply_focused_scorecard_guard
from forecasting.diagnostics.forecast_diagnostics import _diagnostic_band_for_row, prep_backtest
from forecasting.forecast.forecast_pipeline import apply_operational_stage_selector
from forecasting.forecast.peak_risk_correction import apply_peak_risk_correction
from forecasting.forecast.recursive_engine import recursive_forecast
from forecasting.forecast.uncertainty_bands import _band_risk_multiplier, _prep, apply_bands
from forecasting.forecast.weather_robustness_hedge import apply_weather_robustness_hedge
from forecasting.model.ensemble import blend_predictions
from forecasting.data.local_weather_loader import apply_dynamic_temperature_calibration


class ForecastControlTests(unittest.TestCase):
    def test_weather_robustness_hedge_derives_forecast_day_from_dt(self):
        df = pd.DataFrame(
            {
                "DT": pd.date_range("2026-06-15 17:00", periods=2, freq="h"),
                "Final_Forecast_MWH": [300.0, 295.0],
                "Stage_Selected_Forecast_MWH": [300.0, 295.0],
                "WeatherScenario_warmer_P50_MWH": [318.0, 310.0],
                "WeatherScenario_cooler_P50_MWH": [292.0, 289.0],
                "Temperature_DailyMax": [105.0, 105.0],
            }
        )

        out = apply_weather_robustness_hedge(
            df,
            config={
                "weather_robustness_hedge": {
                    "enabled": True,
                    "hours": [17, 18],
                    "min_maxtemp_f": 90.0,
                    "min_forecast_day": 1,
                    "max_forecast_day": 16,
                    "cap_mwh": 6.0,
                    "upper_scenario_blend": 0.10,
                }
            },
        )

        self.assertTrue((out["Weather_Robustness_Gate"] == 1).all())
        self.assertTrue((out["Weather_Robustness_Hedge_MWH"] > 0.0).all())
        self.assertGreater(out.loc[0, "Final_Forecast_MWH"], df.loc[0, "Final_Forecast_MWH"])

    def test_weather_robustness_hedge_handles_mixed_offset_export_timestamps(self):
        df = pd.DataFrame(
            {
                "DT": [
                    "2020-01-01 00:00:00-08:00",
                    "2026-03-08 17:00:00-08:00",
                    "2026-03-08 18:00:00-07:00",
                ],
                "Final_Forecast_MWH": [np.nan, 300.0, 295.0],
                "Stage_Selected_Forecast_MWH": [np.nan, 300.0, 295.0],
                "WeatherScenario_warmer_P50_MWH": [np.nan, 318.0, 310.0],
                "WeatherScenario_cooler_P50_MWH": [np.nan, 292.0, 289.0],
                "Temperature_DailyMax": [56.0, 105.0, 105.0],
            }
        )

        out = apply_weather_robustness_hedge(
            df,
            config={
                "weather_robustness_hedge": {
                    "enabled": True,
                    "hours": [17, 18],
                    "min_maxtemp_f": 90.0,
                    "min_forecast_day": 1,
                    "max_forecast_day": 16,
                    "cap_mwh": 6.0,
                    "upper_scenario_blend": 0.10,
                }
            },
        )

        self.assertEqual(out.loc[0, "Weather_Robustness_Gate"], 0)
        self.assertTrue((out.loc[1:, "Weather_Robustness_Gate"] == 1).all())
        self.assertTrue((out.loc[1:, "Weather_Robustness_Hedge_MWH"] > 0.0).all())

    def test_weather_robustness_hedge_applies_lower_capped_ramp_path(self):
        df = pd.DataFrame(
            {
                "DT": [pd.Timestamp("2026-06-12 12:00")],
                "Forecast_Day": [16],
                "Final_Forecast_MWH": [220.0],
                "Stage_Selected_Forecast_MWH": [220.0],
                "WeatherScenario_warmer_P50_MWH": [240.0],
                "WeatherScenario_cooler_P50_MWH": [210.0],
                "Temperature_DailyMax": [102.3],
            }
        )

        out = apply_weather_robustness_hedge(
            df,
            config={
                "weather_robustness_hedge": {
                    "enabled": True,
                    "hours": [17, 18],
                    "min_maxtemp_f": 90.0,
                    "min_forecast_day": 1,
                    "max_forecast_day": 16,
                    "ramp_hours": [10, 11, 12, 13, 14, 15],
                    "ramp_min_maxtemp_f": 100.0,
                    "ramp_min_forecast_day": 8,
                    "ramp_max_forecast_day": 16,
                    "ramp_cap_mwh": 2.5,
                    "ramp_max_fraction_of_warmer_delta": 0.25,
                    "cap_mwh": 4.0,
                    "upper_scenario_blend": 0.10,
                }
            },
        )

        self.assertEqual(out.loc[0, "Weather_Robustness_Gate"], 1)
        self.assertGreater(out.loc[0, "Weather_Robustness_Hedge_MWH"], 0.0)
        self.assertLessEqual(out.loc[0, "Weather_Robustness_Hedge_MWH"], 2.5)
        self.assertEqual(out.loc[0, "Weather_Robustness_Hedge_Source"], "weather_uncertainty_ramp_hedge")

    def test_extreme_heat_morning_midday_bands_are_widened(self):
        df = pd.DataFrame(
            {
                "DT": [
                    "2026-03-08 08:00:00-08:00",
                    "2026-03-08 12:00:00-07:00",
                    "2026-03-08 08:00:00-07:00",
                ],
                "Temperature_DailyMax": [105.0, 105.0, 78.0],
                "CloudCover_Norm": [0.1, 0.1, 0.1],
                "BTM_Solar_Loss_From_ClearSky_MW": [0.0, 0.0, 0.0],
            }
        )
        prepped = _prep(df)

        mult = _band_risk_multiplier(prepped)

        self.assertGreater(mult.iloc[0], mult.iloc[2])
        self.assertGreater(mult.iloc[1], 1.0)

    def test_hot_overnight_band_floor_applies_to_production_and_diagnostics(self):
        floor_cfg = {
            "enabled": True,
            "rules": [
                {
                    "name": "hot_100_plus_overnight_min_band",
                    "min_daily_max_temp_f": 100.0,
                    "hour_groups": ["Overnight"],
                    "min_band_mwh": 7.5,
                },
                {
                    "name": "hot_100_plus_early_overnight_min_band",
                    "min_daily_max_temp_f": 100.0,
                    "hours": [0, 1],
                    "min_band_mwh": 13.0,
                },
                {
                    "name": "hot_100_plus_late_evening_min_band",
                    "min_daily_max_temp_f": 100.0,
                    "hour_groups": ["LateEvening"],
                    "min_band_mwh": 15.0,
                }
            ],
        }
        df = pd.DataFrame(
            {
                "DT": [
                    pd.Timestamp("2026-06-12 01:00"),
                    pd.Timestamp("2026-06-12 02:00"),
                    pd.Timestamp("2026-06-12 23:00"),
                ],
                "Calibrated_Forecast_MWH": [170.0, 190.0, 205.0],
                "Temperature_DailyMax": [102.0, 102.0, 102.0],
            }
        )

        out = apply_bands(
            df,
            percent_band=0.01,
            floor_mwh=4.0,
            band_scale=0.55,
            hot_bucket_band_floor=floor_cfg,
        )
        diagnostic_band, diagnostic_method = _diagnostic_band_for_row(
            out.iloc[1],
            forecast=190.0,
            residual_band_lookup=None,
            percent_band=0.01,
            floor_mwh=4.0,
            band_scale=0.55,
            hot_bucket_band_floor=floor_cfg,
        )

        self.assertEqual(out["Band"].tolist(), [13.0, 7.5, 15.0])
        self.assertTrue(out["Band_Method"].astype(str).str.contains("hot_bucket_floor").all())
        self.assertEqual(diagnostic_band, 7.5)
        self.assertIn("hot_bucket_floor", diagnostic_method)

    def test_focused_guard_applies_june_extreme_heat_rule(self):
        df = pd.DataFrame(
            {
                "DT": [pd.Timestamp("2026-06-26 12:00")],
                "Final_Forecast_MWH": [280.0],
                "Stage_Selected_Forecast_MWH": [280.0],
                "Temperature_DailyMax": [115.0],
                "IsHoliday": [0],
            }
        )
        config = {
            "calibration": {
                "stage_selector": {
                    "focused_scorecard_guard": {
                        "enabled": True,
                        "total_cap_mwh": 18.0,
                        "rules": [
                            {
                                "name": "june_extreme_heat_midday_ramp_up",
                                "adjustment_mwh": 10.0,
                                "months": [6],
                                "hours": [10, 11, 12, 13, 14, 15],
                                "min_forecast_day": 1,
                                "max_forecast_day": 16,
                                "min_maxtemp_f": 105.0,
                            }
                        ],
                    }
                }
            }
        }

        out = apply_focused_scorecard_guard(df, config, forecast_col="Final_Forecast_MWH")

        self.assertEqual(out.loc[0, "Focused_Scorecard_Guard_MWH"], 10.0)
        self.assertEqual(out.loc[0, "Final_Forecast_MWH"], 290.0)
        self.assertEqual(out.loc[0, "Stage_Selected_Forecast_MWH"], 290.0)
        self.assertEqual(out.loc[0, "Pre_Focused_Guard_Forecast_MWH"], 280.0)
        self.assertEqual(out.loc[0, "Post_Focused_Guard_Forecast_MWH"], 290.0)
        self.assertEqual(out.loc[0, "Focused_Guard_Applied_Flag"], 1)

    def test_focused_guard_applies_june_100_to_105_long_hot_ramp_rule(self):
        df = pd.DataFrame(
            {
                "DT": [
                    pd.Timestamp("2026-06-12 12:00"),
                    pd.Timestamp("2026-06-12 13:00"),
                    pd.Timestamp("2026-06-12 15:00"),
                    pd.Timestamp("2026-06-12 17:00"),
                ],
                "Forecast_Day": [16, 16, 16, 16],
                "Final_Forecast_MWH": [226.0, 252.0, 286.0, 312.0],
                "Stage_Selected_Forecast_MWH": [226.0, 252.0, 286.0, 312.0],
                "Temperature_DailyMax": [102.3, 102.3, 102.3, 102.3],
                "IsHoliday": [0, 0, 0, 0],
            }
        )
        config = {
            "calibration": {
                "stage_selector": {
                    "focused_scorecard_guard": {
                        "enabled": True,
                        "total_cap_mwh": 18.0,
                        "rules": [
                            {
                                "name": "june_long_hot_100_105_core_ramp_up",
                                "adjustment_mwh": 10.0,
                                "months": [6],
                                "hours": [10, 11, 12, 13],
                                "min_forecast_day": 8,
                                "max_forecast_day": 16,
                                "min_maxtemp_f": 100.0,
                                "max_maxtemp_f": 105.0,
                                "min_forecast_mwh": 230.0,
                            },
                            {
                                "name": "june_long_hot_100_105_peak_ramp_up",
                                "adjustment_mwh": 8.0,
                                "months": [6],
                                "hours": [14, 15],
                                "min_forecast_day": 8,
                                "max_forecast_day": 16,
                                "min_maxtemp_f": 100.0,
                                "max_maxtemp_f": 105.0,
                                "min_forecast_mwh": 265.0,
                            },
                            {
                                "name": "june_long_hot_100_105_peak_finish_up",
                                "adjustment_mwh": 5.5,
                                "months": [6],
                                "hours": [16, 17, 18],
                                "min_forecast_day": 8,
                                "max_forecast_day": 16,
                                "min_maxtemp_f": 100.0,
                                "max_maxtemp_f": 105.0,
                            },
                        ],
                    }
                }
            }
        }

        out = apply_focused_scorecard_guard(df, config, forecast_col="Final_Forecast_MWH")

        self.assertEqual(out["Focused_Scorecard_Guard_MWH"].tolist(), [0.0, 10.0, 8.0, 5.5])
        self.assertEqual(out["Final_Forecast_MWH"].tolist(), [226.0, 262.0, 294.0, 317.5])

    def test_focused_guard_rule_can_extend_total_cap_for_narrow_pattern(self):
        df = pd.DataFrame(
            {
                "DT": [pd.Timestamp("2026-08-26 14:00"), pd.Timestamp("2026-08-26 15:00")],
                "Forecast_Day": [10, 10],
                "Final_Forecast_MWH": [250.0, 250.0],
                "Stage_Selected_Forecast_MWH": [250.0, 250.0],
                "Temperature_DailyMax": [96.8, 96.8],
                "CloudCover_Norm": [0.10, 0.10],
                "BTM_Solar_Loss_From_ClearSky_MW": [0.0, 0.0],
                "IsHoliday": [0, 0],
            }
        )
        config = {
            "calibration": {
                "stage_selector": {
                    "focused_scorecard_guard": {
                        "enabled": True,
                        "total_cap_mwh": 30.0,
                        "rules": [
                            {
                                "name": "normal_cap_backoff",
                                "adjustment_mwh": -30.0,
                                "months": [8],
                                "hours": [14, 15],
                                "min_forecast_day": 8,
                                "max_forecast_day": 12,
                                "min_maxtemp_f": 95.0,
                                "max_maxtemp_f": 99.0,
                            },
                            {
                                "name": "extended_cap_backoff",
                                "adjustment_mwh": -30.0,
                                "months": [8],
                                "hours": [14],
                                "min_forecast_day": 8,
                                "max_forecast_day": 12,
                                "min_maxtemp_f": 95.0,
                                "max_maxtemp_f": 99.0,
                                "max_total_cap_mwh": 60.0,
                            },
                        ],
                    }
                }
            }
        }

        out = apply_focused_scorecard_guard(df, config, forecast_col="Final_Forecast_MWH")

        self.assertEqual(out["Focused_Scorecard_Guard_MWH"].tolist(), [-60.0, -30.0])
        self.assertEqual(out["Final_Forecast_MWH"].tolist(), [190.0, 220.0])
        self.assertEqual(out["Stage_Selected_Forecast_MWH"].tolist(), [190.0, 220.0])

    def test_focused_guard_applies_weather_shape_and_weekend_filters(self):
        df = pd.DataFrame(
            {
                "DT": [
                    pd.Timestamp("2026-06-08 14:00"),
                    pd.Timestamp("2026-06-03 16:00"),
                    pd.Timestamp("2026-06-06 16:00"),
                ],
                "Forecast_Day": [12, 13, 10],
                "Final_Forecast_MWH": [150.0, 224.0, 188.0],
                "Stage_Selected_Forecast_MWH": [150.0, 224.0, 188.0],
                "Temperature_DailyMax": [83.8, 93.3, 86.3],
                "CloudCover_Norm": [0.93, 0.0, 0.1],
                "BTM_Solar_Loss_From_ClearSky_MW": [0.6, 0.0, 2.2],
                "IsWeekend": [0, 0, 1],
                "IsHoliday": [0, 0, 0],
            }
        )
        config = {
            "calibration": {
                "stage_selector": {
                    "focused_scorecard_guard": {
                        "enabled": True,
                        "total_cap_mwh": 18.0,
                        "rules": [
                            {
                                "name": "june_long_cloudy_mild_peak_window_up",
                                "adjustment_mwh": 8.0,
                                "months": [6],
                                "hours": [14, 15],
                                "min_forecast_day": 8,
                                "max_forecast_day": 16,
                                "min_maxtemp_f": 80.0,
                                "max_maxtemp_f": 90.0,
                                "min_cloud_cover_norm": 0.80,
                                "min_solar_loss_mw": 0.50,
                                "weekend": False,
                                "holiday": False,
                            },
                            {
                                "name": "june_long_clear_hot_peak_window_up",
                                "adjustment_mwh": 4.0,
                                "months": [6],
                                "hours": [14, 15, 16, 17],
                                "min_forecast_day": 8,
                                "max_forecast_day": 16,
                                "min_maxtemp_f": 90.0,
                                "max_maxtemp_f": 95.0,
                                "max_cloud_cover_norm": 0.20,
                                "max_solar_loss_mw": 0.50,
                                "weekend": False,
                                "holiday": False,
                            },
                            {
                                "name": "june_long_mild_clear_weekend_peak_down",
                                "adjustment_mwh": -8.0,
                                "months": [6],
                                "hours": [14, 15, 16, 17, 18],
                                "min_forecast_day": 8,
                                "max_forecast_day": 16,
                                "min_maxtemp_f": 80.0,
                                "max_maxtemp_f": 90.0,
                                "max_cloud_cover_norm": 0.30,
                                "weekend": True,
                                "holiday": False,
                            },
                        ],
                    }
                }
            }
        }

        out = apply_focused_scorecard_guard(df, config, forecast_col="Final_Forecast_MWH")

        self.assertEqual(out["Focused_Scorecard_Guard_MWH"].tolist(), [8.0, 4.0, -8.0])
        self.assertEqual(out["Final_Forecast_MWH"].tolist(), [158.0, 228.0, 180.0])

    def test_focused_guard_can_gate_on_raw_minus_same_hour_load_state(self):
        df = pd.DataFrame(
            {
                "DT": [
                    pd.Timestamp("2026-06-25 15:00"),
                    pd.Timestamp("2026-06-25 16:00"),
                    pd.Timestamp("2026-07-01 17:00"),
                ],
                "Final_Forecast_MWH": [220.0, 220.0, 220.0],
                "Stage_Selected_Forecast_MWH": [220.0, 220.0, 220.0],
                "Raw_Forecast_MWH": [230.0, 225.0, 220.0],
                "MWH_SameHour7DayMean": [210.0, 215.0, 200.0],
                "MWH_Lag24": [225.0, 220.0, 225.0],
                "Temperature_DailyMax": [93.0, 93.0, 94.0],
                "CloudCover_Norm": [0.0, 0.0, 0.0],
            }
        )
        config = {
            "calibration": {
                "stage_selector": {
                    "focused_scorecard_guard": {
                        "enabled": True,
                        "total_cap_mwh": 30.0,
                        "rules": [
                            {
                                "name": "june_july_clear_hot_raw_level_backoff",
                                "adjustment_mwh": -5.0,
                                "months": [6, 7],
                                "hours": [13, 14, 15, 16, 17, 18],
                                "min_maxtemp_f": 90.0,
                                "max_maxtemp_f": 100.0,
                                "max_cloud_cover_norm": 0.20,
                                "min_raw_minus_samehour_7day_mean_mwh": 15.0,
                                "min_raw_minus_samehour_yesterday_mwh": 0.0,
                            }
                        ],
                    }
                }
            }
        }

        out = apply_focused_scorecard_guard(df, config, forecast_col="Final_Forecast_MWH")

        self.assertEqual(out["Focused_Scorecard_Guard_MWH"].tolist(), [-5.0, 0.0, 0.0])
        self.assertEqual(out["Final_Forecast_MWH"].tolist(), [215.0, 220.0, 220.0])
        self.assertEqual(out["Stage_Selected_Forecast_MWH"].tolist(), [215.0, 220.0, 220.0])
        self.assertEqual(out["Raw_Minus_SameHour7DayMean_MWH"].tolist(), [20.0, 10.0, 20.0])
        self.assertEqual(out["Raw_Minus_SameHourYesterday_MWH"].tolist(), [5.0, 5.0, -5.0])

    def test_focused_guard_can_gate_on_max_raw_minus_same_hour_load_state(self):
        df = pd.DataFrame(
            {
                "DT": [pd.Timestamp("2026-07-06 16:00"), pd.Timestamp("2026-07-29 16:00")],
                "Final_Forecast_MWH": [250.0, 250.0],
                "Stage_Selected_Forecast_MWH": [250.0, 250.0],
                "Raw_Forecast_MWH": [240.0, 248.0],
                "MWH_SameHour7DayMean": [235.0, 220.0],
                "MWH_Lag24": [230.0, 230.0],
                "Temperature_DailyMax": [97.0, 97.0],
                "CloudCover_Norm": [0.0, 0.0],
            }
        )
        config = {
            "calibration": {
                "stage_selector": {
                    "focused_scorecard_guard": {
                        "enabled": True,
                        "total_cap_mwh": 30.0,
                        "rules": [
                            {
                                "name": "july_low_state_backoff",
                                "adjustment_mwh": -10.0,
                                "months": [7],
                                "hours": [16],
                                "min_maxtemp_f": 95.0,
                                "max_maxtemp_f": 98.0,
                                "max_raw_minus_samehour_7day_mean_mwh": 12.0,
                            }
                        ],
                    }
                }
            }
        }

        out = apply_focused_scorecard_guard(df, config, forecast_col="Final_Forecast_MWH")

        self.assertEqual(out["Focused_Scorecard_Guard_MWH"].tolist(), [-10.0, 0.0])
        self.assertEqual(out["Raw_Minus_SameHour7DayMean_MWH"].tolist(), [5.0, 28.0])
        self.assertEqual(out["Final_Forecast_MWH"].tolist(), [240.0, 250.0])

    def test_recursive_forecast_exposes_load_state_lags_for_guards(self):
        class ConstantModel:
            def __init__(self, value):
                self.value = value

            def predict(self, X):
                return np.full(len(X), self.value, dtype=float)

        hist = pd.DataFrame(
            {
                "DT": pd.date_range("2026-06-01 00:00", periods=168, freq="h"),
                "MWH": np.arange(168, dtype=float),
            }
        )
        future = pd.DataFrame({"DT": [pd.Timestamp("2026-06-08 00:00")]})

        out = recursive_forecast(
            future_frame=future,
            historical_seed=hist,
            xgb_model=ConstantModel(200.0),
            lgb_model=ConstantModel(200.0),
            features=["MWH_Lag24", "MWH_SameHour7DayMean"],
            ensemble_weights={"xgb": 0.5, "lgb": 0.5},
        )

        self.assertIn("MWH_Lag24", out.columns)
        self.assertIn("MWH_SameHour7DayMean", out.columns)
        self.assertEqual(out.loc[0, "MWH_Lag24"], 144.0)
        self.assertEqual(out.loc[0, "MWH_SameHour7DayMean"], 72.0)

    def test_backtest_prediction_whitelists_retain_load_state_lags(self):
        required = {"MWH_Lag24", "MWH_SameHour7DayMean"}

        self.assertTrue(required.issubset(set(ROLLING_BACKTEST_PRED_COLS)))
        self.assertTrue(required.issubset(set(ROLLING_REPLAY_PRED_COLS)))

    def test_focused_guard_handles_mixed_offset_export_timestamps(self):
        df = pd.DataFrame(
            {
                "DT": [
                    "2020-01-01 00:00:00-08:00",
                    "2026-03-08 01:00:00-08:00",
                    "2026-03-08 03:00:00-07:00",
                ],
                "Final_Forecast_MWH": [np.nan, 280.0, 285.0],
                "Stage_Selected_Forecast_MWH": [np.nan, 280.0, 285.0],
                "Temperature_DailyMax": [56.0, 105.0, 105.0],
                "IsHoliday": [0, 0, 0],
            }
        )
        config = {
            "calibration": {
                "stage_selector": {
                    "focused_scorecard_guard": {
                        "enabled": True,
                        "total_cap_mwh": 18.0,
                        "rules": [
                            {
                                "name": "dst_mixed_offset_local_hour_guard",
                                "adjustment_mwh": 4.0,
                                "months": [3],
                                "hours": [1, 3],
                                "min_forecast_day": 1,
                                "max_forecast_day": 16,
                                "min_maxtemp_f": 100.0,
                            }
                        ],
                    }
                }
            }
        }

        out = apply_focused_scorecard_guard(df, config, forecast_col="Final_Forecast_MWH")

        self.assertEqual(out.loc[0, "Focused_Scorecard_Guard_MWH"], 0.0)
        self.assertTrue((out.loc[1:, "Focused_Scorecard_Guard_MWH"] == 4.0).all())

    def test_prep_backtest_handles_mixed_offset_export_timestamps(self):
        df = pd.DataFrame(
            {
                "DT": [
                    "2026-03-08 01:00:00-08:00",
                    "2026-03-08 03:00:00-07:00",
                ],
                "Actual_MWH": [100.0, 110.0],
                "Raw_Forecast_MWH": [99.0, 108.0],
            }
        )

        out = prep_backtest(df)

        self.assertEqual(out["Hour"].tolist(), [1, 3])
        self.assertEqual(out["Residual_MWH"].tolist(), [1.0, 2.0])

    def test_replay_focused_guard_applies_without_weather_hedge(self):
        df = pd.DataFrame(
            {
                "DT": [pd.Timestamp("2025-06-26 19:00")],
                "Actual_MWH": [243.0],
                "Final_Backtest_Forecast_MWH": [215.0],
                "Final_Forecast_MWH": [215.0],
                "Stage_Selected_Forecast_MWH": [215.0],
                "Forecast_Day": [5],
                "Month": [6],
                "Hour": [19],
                "Temperature_DailyMax": [90.1],
                "IsHoliday": [0],
            }
        )
        config = {
            "calibration": {
                "stage_selector": {
                    "focused_scorecard_guard": {
                        "enabled": True,
                        "total_cap_mwh": 40.0,
                        "rules": [
                            {
                                "name": "june_days4to7_mild_hot_evening_recovery_up",
                                "adjustment_mwh": 40.0,
                                "months": [6],
                                "hours": [19],
                                "min_forecast_day": 4,
                                "max_forecast_day": 7,
                                "min_maxtemp_f": 90.0,
                                "max_maxtemp_f": 95.0,
                            }
                        ],
                    }
                }
            }
        }

        out = _apply_replay_focused_guard(df, config, also_update_stage=True)

        self.assertEqual(out.loc[0, "Pre_Focused_Guard_Forecast_MWH"], 215.0)
        self.assertEqual(out.loc[0, "Post_Focused_Guard_Forecast_MWH"], 255.0)
        self.assertEqual(out.loc[0, "Final_Backtest_Forecast_MWH"], 255.0)
        self.assertEqual(out.loc[0, "Final_Forecast_MWH"], 255.0)
        self.assertEqual(out.loc[0, "Stage_Selected_Forecast_MWH"], 255.0)
        self.assertEqual(out.loc[0, "Final_Residual_MWH"], -12.0)

    def test_solar_features_use_weather_proxy_when_solar_forecast_missing(self):
        df = pd.DataFrame(
            {
                "DT": [pd.Timestamp("2026-06-15 12:00")],
                "GHI_Wm2": [900.0],
                "CloudCover_Norm": [0.10],
            }
        )
        btm = pd.DataFrame(
            {
                "DT": [pd.Timestamp("2026-06-01")],
                "Nameplate_MW": [25.0],
                "Capacity_Ratio_To_Current": [1.0],
                "Impact_Cap_MW": [20.0],
            }
        )

        out = add_solar_features(df, btm)

        self.assertGreater(out.loc[0, "BTM_Solar_Proxy_MW"], 0.0)
        self.assertGreater(out.loc[0, "BTM_ClearSky_Proxy_MW"], out.loc[0, "BTM_Solar_Proxy_MW"])
        self.assertGreater(out.loc[0, "BTM_Solar_Loss_From_ClearSky_MW"], 0.0)

    def test_peak_risk_uses_tree_gap_without_prophet(self):
        df = pd.DataFrame(
            {
                "DT": [pd.Timestamp("2026-06-15 17:00")],
                "Hour": [17],
                "Temperature_DailyMax": [100.0],
                "Forecast_Day": [3],
                "Calibrated_Forecast_MWH": [100.0],
                "XGB_Pred_MWH": [112.0],
                "LGB_Pred_MWH": [111.0],
                "CatBoost_Pred_MWH": [105.0],
            }
        )
        config = {
            "calibration": {
                "peak_risk": {
                    "enabled": True,
                    "hours": [17],
                    "min_maxtemp_f": 90.0,
                    "prophet_gap_threshold_mwh": 99.0,
                    "catboost_gap_threshold_mwh": 99.0,
                    "tree_gap_threshold_mwh": 5.0,
                    "tree_gap_signal_strength": 0.50,
                    "tree_gap_model_cols": ["XGB_Pred_MWH", "LGB_Pred_MWH", "CatBoost_Pred_MWH"],
                    "blend": 1.0,
                    "cap_mwh": 10.0,
                }
            }
        }

        out = apply_peak_risk_correction(df, config)

        self.assertAlmostEqual(out.loc[0, "Peak_Risk_Cal_MWH"], 3.5)
        self.assertEqual(out.loc[0, "Peak_Risk_Source"], "tree_peak_gap")
        self.assertAlmostEqual(out.loc[0, "Peak_Risk_Adjusted_Forecast_MWH"], 103.5)

    def test_stage_selector_conditional_override_respects_hour_filter(self):
        df = pd.DataFrame(
            {
                "DT": [pd.Timestamp("2026-07-15 12:00"), pd.Timestamp("2026-07-15 16:00")],
                "Forecast_Day": [9, 9],
                "Season": ["Summer", "Summer"],
                "Hour": [12, 16],
                "Temperature_DailyMax": [104.0, 104.0],
                "DailyMaxTempBucket": [6, 6],
                "Raw_Forecast_MWH": [250.0, 252.0],
                "Residual_Calibrated_Forecast_MWH": [265.0, 267.0],
                "Final_Forecast_MWH": [280.0, 282.0],
            }
        )
        config = {
            "calibration": {
                "stage_selector": {
                    "enabled": True,
                    "conditional_stage_overrides": [
                        {
                            "name": "summer_high_temp_raw_override",
                            "enabled": True,
                            "seasons": ["Summer"],
                            "hours": [9, 10, 11, 12, 13],
                            "min_daily_max_temp_bucket": 6,
                            "min_forecast_day": 1,
                            "max_forecast_day": 16,
                            "stage": "raw",
                        }
                    ],
                }
            }
        }

        out = apply_operational_stage_selector(df, config, forecast_col="Final_Forecast_MWH")

        self.assertEqual(out.loc[0, "Stage_Selected_Forecast_MWH"], 250.0)
        self.assertEqual(out.loc[0, "Stage_Selector_Source"], "Raw_Forecast_MWH")
        self.assertIn("conditional_stage_override:summer_high_temp_raw_override", out.loc[0, "Stage_Selector_Reason"])
        self.assertEqual(out.loc[1, "Stage_Selected_Forecast_MWH"], 282.0)
        self.assertEqual(out.loc[1, "Stage_Selector_Source"], "Final_Forecast_MWH")
        self.assertNotIn("conditional_stage_override", out.loc[1, "Stage_Selector_Reason"])

    def test_blend_predictions_pads_short_optional_components(self):
        blended = blend_predictions(
            [10.0, 20.0, 30.0],
            [12.0, 22.0, 32.0],
            {"xgb": 0.5, "lgb": 0.3, "catboost": 0.2},
            catboost_pred=[14.0],
        )

        self.assertTrue(np.allclose(blended[0], 11.4))
        self.assertTrue(np.allclose(blended[1:], [20.75, 30.75]))

    def test_apply_dynamic_temperature_calibration_adjusts_temperatures_with_decay(self):
        hist_wx = pd.DataFrame(
            {
                "DT": pd.date_range("2026-06-15 00:00", periods=24, freq="h"),
                "TempF": [90.0] * 24,
                "LocalStation_TempF": [85.0] * 24, # Cooler by 5 degrees consistently
            }
        )
        fut_wx = pd.DataFrame(
            {
                "DT": pd.date_range("2026-06-16 00:00", periods=24, freq="h"),
                "TempF": [95.0] * 24,
            }
        )
        config = {
            "local_weather": {
                "temperature_calibration": {
                    "dynamic_enabled": True,
                    "dynamic_window_hours": 24,
                    "dynamic_cap_f": 6.0,
                    "dynamic_blend": 0.80, # Expected bias: -5.0 * 0.80 = -4.0
                    "dynamic_decay_hours": 24.0,
                }
            }
        }
        
        out = apply_dynamic_temperature_calibration(fut_wx, hist_wx, config)
        
        self.assertIn("Dynamic_Weather_Correction_F", out.columns)
        # Verify the first hour (hour 0) has approx -4.0 degrees correction
        self.assertAlmostEqual(out.loc[0, "Dynamic_Weather_Correction_F"], -4.0, places=2)
        # Verify the 24th hour has decayed towards 0 (factor of exp(-23/24) = ~0.38 -> -4 * 0.38 = ~-1.5)
        self.assertTrue(-4.0 < out.loc[23, "Dynamic_Weather_Correction_F"] < -1.0)
        self.assertAlmostEqual(out.loc[0, "TempF"], 91.0, places=2)


if __name__ == "__main__":
    unittest.main()
