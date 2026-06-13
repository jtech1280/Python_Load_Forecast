import unittest

import numpy as np
import pandas as pd

from forecasting.forecast.focused_scorecard_guard import apply_focused_scorecard_guard
from forecasting.forecast.uncertainty_bands import _band_risk_multiplier, _prep
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
