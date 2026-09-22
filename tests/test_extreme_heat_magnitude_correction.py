from __future__ import annotations

import unittest

import pandas as pd

from forecasting.forecast.extreme_heat_magnitude_correction import (
    apply_extreme_heat_magnitude_correction,
)


def _df(hour, daily_max, streak_depth, forecast=500.0):
    return pd.DataFrame(
        {
            "Hour": hour,
            "Temperature_DailyMax": daily_max,
            "ConsecutiveVeryHotDays95": streak_depth,
            "Final_Backtest_Forecast_MWH": forecast,
            "Stage_Selected_Forecast_MWH": forecast,
        }
    )


def _config(**overrides):
    cfg = {
        "enabled": True,
        "min_maxtemp_f": 98.0,
        "max_consecutive_very_hot_days95": 4.0,
        "escalation_mwh_per_degree": 4.0,
        "max_correction_mwh": 30.0,
        "hours": [16, 17, 18, 19, 20],
    }
    cfg.update(overrides)
    return {"calibration": {"extreme_heat_magnitude_correction": cfg}}


class DisabledByDefaultTests(unittest.TestCase):
    def test_disabled_config_is_a_no_op(self):
        df = _df(hour=[18], daily_max=[103.8], streak_depth=[1.0])
        config = {"calibration": {"extreme_heat_magnitude_correction": {"enabled": False}}}
        out = apply_extreme_heat_magnitude_correction(df, config)
        self.assertEqual(out["Final_Backtest_Forecast_MWH"].iloc[0], 500.0)
        self.assertNotIn("Extreme_Heat_Magnitude_Correction_MWH", out.columns)

    def test_missing_config_section_is_a_no_op(self):
        df = _df(hour=[18], daily_max=[103.8], streak_depth=[1.0])
        out = apply_extreme_heat_magnitude_correction(df, {})
        self.assertEqual(out["Final_Backtest_Forecast_MWH"].iloc[0], 500.0)


class MagnitudeLogicTests(unittest.TestCase):
    def test_no_correction_below_onset_temp(self):
        df = _df(hour=[18, 18], daily_max=[97.9, 90.0], streak_depth=[1.0, 1.0])
        out = apply_extreme_heat_magnitude_correction(df, _config())
        self.assertTrue((out["Final_Backtest_Forecast_MWH"] == 500.0).all())
        self.assertTrue((out["Extreme_Heat_Magnitude_Scope_Flag"] == 0).all())

    def test_correction_scales_linearly_past_onset(self):
        df = _df(hour=[18, 18], daily_max=[100.7, 103.8], streak_depth=[1.0, 2.0])
        out = apply_extreme_heat_magnitude_correction(df, _config())
        # (100.7-98)*4.0=10.8, (103.8-98)*4.0=23.2
        self.assertAlmostEqual(out["Extreme_Heat_Magnitude_Correction_MWH"].iloc[0], 10.8, places=6)
        self.assertAlmostEqual(out["Extreme_Heat_Magnitude_Correction_MWH"].iloc[1], 23.2, places=6)
        self.assertAlmostEqual(out["Final_Backtest_Forecast_MWH"].iloc[0], 510.8, places=6)

    def test_correction_clipped_at_max(self):
        df = _df(hour=[18], daily_max=[115.0], streak_depth=[1.0])
        out = apply_extreme_heat_magnitude_correction(df, _config())
        # (115-98)*4.0 = 68, clipped to 30
        self.assertAlmostEqual(out["Extreme_Heat_Magnitude_Correction_MWH"].iloc[0], 30.0)
        self.assertAlmostEqual(out["Final_Backtest_Forecast_MWH"].iloc[0], 530.0)

    def test_deep_streak_is_excluded_even_if_extremely_hot(self):
        """A day deep in a sustained streak that's also extremely hot stays owned by
        escalating_heat_persistence_correction -- this rule must not also fire and
        double-correct it."""
        df = _df(hour=[18], daily_max=[105.1], streak_depth=[11.0])
        out = apply_extreme_heat_magnitude_correction(df, _config())
        self.assertEqual(out["Final_Backtest_Forecast_MWH"].iloc[0], 500.0)
        self.assertEqual(out["Extreme_Heat_Magnitude_Scope_Flag"].iloc[0], 0)

    def test_shallow_streak_at_the_boundary_still_qualifies(self):
        df = _df(hour=[18], daily_max=[99.0], streak_depth=[4.0])
        out = apply_extreme_heat_magnitude_correction(df, _config())
        self.assertEqual(out["Extreme_Heat_Magnitude_Scope_Flag"].iloc[0], 1)

    def test_outside_hot_peak_hours_is_untouched(self):
        df = _df(hour=[10], daily_max=[103.8], streak_depth=[1.0])
        out = apply_extreme_heat_magnitude_correction(df, _config())
        self.assertEqual(out["Final_Backtest_Forecast_MWH"].iloc[0], 500.0)

    def test_missing_streak_depth_column_treated_as_zero_and_still_qualifies(self):
        df = pd.DataFrame(
            {
                "Hour": [18],
                "Temperature_DailyMax": [103.8],
                "Final_Backtest_Forecast_MWH": [500.0],
            }
        )
        out = apply_extreme_heat_magnitude_correction(df, _config())
        self.assertGreater(out["Final_Backtest_Forecast_MWH"].iloc[0], 500.0)

    def test_also_update_cols_receive_the_same_correction(self):
        df = _df(hour=[18], daily_max=[100.7], streak_depth=[1.0])
        out = apply_extreme_heat_magnitude_correction(
            df, _config(), also_update_cols=("Stage_Selected_Forecast_MWH",)
        )
        self.assertAlmostEqual(out["Stage_Selected_Forecast_MWH"].iloc[0], 510.8, places=6)
        self.assertAlmostEqual(out["Final_Backtest_Forecast_MWH"].iloc[0], 510.8, places=6)

    def test_empty_dataframe_returns_empty(self):
        df = pd.DataFrame(columns=["Hour", "Temperature_DailyMax", "Final_Backtest_Forecast_MWH"])
        out = apply_extreme_heat_magnitude_correction(df, _config())
        self.assertTrue(out.empty)


if __name__ == "__main__":
    unittest.main()
