import unittest

import numpy as np
import pandas as pd

from forecasting.forecast.focused_scorecard_guard import apply_focused_scorecard_guard
from forecasting.diagnostics.forecast_diagnostics import _diagnostic_band_for_row
from forecasting.forecast.uncertainty_bands import _band_risk_multiplier, _prep, apply_bands
from forecasting.forecast.weather_robustness_hedge import apply_weather_robustness_hedge
from forecasting.model.ensemble import blend_predictions


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

    def test_blend_predictions_pads_short_optional_components(self):
        blended = blend_predictions(
            [10.0, 20.0, 30.0],
            [12.0, 22.0, 32.0],
            {"xgb": 0.5, "lgb": 0.3, "catboost": 0.2},
            catboost_pred=[14.0],
        )

        self.assertTrue(np.allclose(blended[0], 11.4))
        self.assertTrue(np.allclose(blended[1:], [20.75, 30.75]))


if __name__ == "__main__":
    unittest.main()
