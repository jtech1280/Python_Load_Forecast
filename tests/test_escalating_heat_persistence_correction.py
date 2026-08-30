from __future__ import annotations

import unittest

import numpy as np
import pandas as pd

from forecasting.forecast.escalating_heat_persistence_correction import (
    apply_escalating_heat_persistence_correction,
)


def _df(hour, daily_max, streak_depth, forecast=500.0, n=None):
    if n is None:
        n = len(hour) if hasattr(hour, "__len__") else 1
    return pd.DataFrame(
        {
            "Hour": hour,
            "Temperature_DailyMax": daily_max,
            "ConsecutiveVeryHotDays95": streak_depth,
            "Final_Backtest_Forecast_MWH": forecast,
            "Stage_Selected_Forecast_MWH": forecast,
        }
    )


class DisabledByDefaultTests(unittest.TestCase):
    def test_disabled_config_is_a_no_op(self):
        df = _df(hour=[18], daily_max=[98.0], streak_depth=[15.0])
        config = {"calibration": {"escalating_heat_persistence_correction": {"enabled": False}}}
        out = apply_escalating_heat_persistence_correction(df, config)
        self.assertEqual(out["Final_Backtest_Forecast_MWH"].iloc[0], 500.0)
        self.assertNotIn("Escalating_Heat_Persistence_Correction_MWH", out.columns)

    def test_missing_config_section_is_a_no_op(self):
        df = _df(hour=[18], daily_max=[98.0], streak_depth=[15.0])
        out = apply_escalating_heat_persistence_correction(df, {})
        self.assertEqual(out["Final_Backtest_Forecast_MWH"].iloc[0], 500.0)


class EscalationLogicTests(unittest.TestCase):
    def _config(self, **overrides):
        cfg = {
            "enabled": True,
            "min_consecutive_very_hot_days95": 5.0,
            "escalation_mwh_per_day": 1.5,
            "max_correction_mwh": 20.0,
            "hours": [16, 17, 18, 19, 20],
            "min_maxtemp_f": 90.0,
        }
        cfg.update(overrides)
        return {"calibration": {"escalating_heat_persistence_correction": cfg}}

    def test_no_correction_at_or_below_onset_depth(self):
        df = _df(hour=[18, 18], daily_max=[98.0, 98.0], streak_depth=[5.0, 3.0])
        out = apply_escalating_heat_persistence_correction(df, self._config())
        self.assertTrue((out["Final_Backtest_Forecast_MWH"] == 500.0).all())
        self.assertTrue((out["Escalating_Heat_Persistence_Scope_Flag"] == 0).all())

    def test_correction_scales_linearly_past_onset(self):
        df = _df(hour=[18, 18], daily_max=[98.0, 98.0], streak_depth=[7.0, 10.0])
        out = apply_escalating_heat_persistence_correction(df, self._config())
        # (7-5)*1.5=3.0, (10-5)*1.5=7.5
        self.assertAlmostEqual(out["Final_Backtest_Forecast_MWH"].iloc[0], 503.0)
        self.assertAlmostEqual(out["Final_Backtest_Forecast_MWH"].iloc[1], 507.5)
        self.assertAlmostEqual(out["Escalating_Heat_Persistence_Correction_MWH"].iloc[0], 3.0)

    def test_correction_clipped_at_max(self):
        df = _df(hour=[18], daily_max=[98.0], streak_depth=[30.0])
        out = apply_escalating_heat_persistence_correction(df, self._config())
        self.assertAlmostEqual(out["Final_Backtest_Forecast_MWH"].iloc[0], 520.0)
        self.assertAlmostEqual(out["Escalating_Heat_Persistence_Correction_MWH"].iloc[0], 20.0)

    def test_outside_hot_peak_hours_is_untouched(self):
        df = _df(hour=[10], daily_max=[98.0], streak_depth=[15.0])
        out = apply_escalating_heat_persistence_correction(df, self._config())
        self.assertEqual(out["Final_Backtest_Forecast_MWH"].iloc[0], 500.0)

    def test_below_min_maxtemp_is_untouched(self):
        df = _df(hour=[18], daily_max=[80.0], streak_depth=[15.0])
        out = apply_escalating_heat_persistence_correction(df, self._config())
        self.assertEqual(out["Final_Backtest_Forecast_MWH"].iloc[0], 500.0)

    def test_missing_streak_depth_column_treated_as_zero(self):
        df = pd.DataFrame(
            {
                "Hour": [18],
                "Temperature_DailyMax": [98.0],
                "Final_Backtest_Forecast_MWH": [500.0],
            }
        )
        out = apply_escalating_heat_persistence_correction(df, self._config())
        self.assertEqual(out["Final_Backtest_Forecast_MWH"].iloc[0], 500.0)

    def test_also_update_cols_receive_the_same_correction(self):
        df = _df(hour=[18], daily_max=[98.0], streak_depth=[9.0])
        out = apply_escalating_heat_persistence_correction(
            df, self._config(), also_update_cols=("Stage_Selected_Forecast_MWH",)
        )
        # (9-5)*1.5 = 6.0
        self.assertAlmostEqual(out["Stage_Selected_Forecast_MWH"].iloc[0], 506.0)
        self.assertAlmostEqual(out["Final_Backtest_Forecast_MWH"].iloc[0], 506.0)

    def test_empty_dataframe_returns_empty(self):
        df = pd.DataFrame(columns=["Hour", "Temperature_DailyMax", "Final_Backtest_Forecast_MWH"])
        out = apply_escalating_heat_persistence_correction(df, self._config())
        self.assertTrue(out.empty)


if __name__ == "__main__":
    unittest.main()
