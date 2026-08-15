from __future__ import annotations

import unittest

import pandas as pd

from forecasting.forecast.hot_ramp_peak_capture import (
    apply_heat_persistence_peak_capture,
    apply_hot_ramp_peak_capture,
)


class HotRampStrongCapTests(unittest.TestCase):
    def test_strong_ramp_cap_mwh_is_the_key_actually_honored(self):
        """Regression test: apply_hot_ramp_peak_capture must read the strong-day cap from
        config.yaml's `strong_ramp_cap_mwh` key, not the unrelated `strong_cap_mwh` key
        (that name belongs to the sibling heat_persistence_peak_capture section). A decoy
        `strong_cap_mwh` is included below specifically so a regression back to the wrong
        key would be caught here rather than passing silently.
        """
        dt = pd.date_range("2026-07-15 16:00", periods=5, freq="h")
        df = pd.DataFrame(
            {
                "DT": dt,
                "Raw_Forecast_MWH": [500.0, 510.0, 520.0, 515.0, 505.0],
                "Temperature_DailyMax": [105.0] * 5,
                "DailyMaxTemp_Ramp_1Day": [5.0] * 5,
                "CloudCover_Norm": [0.0] * 5,
                "Forecast_Day": [2] * 5,
                "WeatherScenario_warmer_P50_MWH": [700.0] * 5,
            }
        )
        cfg = {
            "enabled": True,
            "shadow_mode": True,
            "hours": [16, 17, 18, 19, 20],
            "min_maxtemp_f": 100.0,
            "min_dailymax_ramp_1day_f": 2.0,
            "strong_ramp_min_maxtemp_f": 100.0,
            "strong_ramp_min_dailymax_ramp_1day_f": 3.0,
            "min_forecast_day": 1,
            "max_forecast_day": 7,
            "max_cloud_cover_norm": 0.40,
            "cap_mwh": 9.0,
            "strong_ramp_cap_mwh": 12.0,
            "strong_cap_mwh": 999.0,  # decoy wrong key; must NOT be picked up
            "warmer_scenario_fraction": 1.0,
            "min_abs_correction_mwh": 0.1,
            "ramp_floor_mwh": 0.0,
            "strong_ramp_floor_mwh": 0.0,
            "anchor_min_maxtemp_f": 105.0,
            "anchor_support_guard_enabled": True,
            "allow_anchorless_shadow_fallback": False,
        }
        artifact = {"metadata": {"global_peak_residual_mwh": 0.0}, "lookups": {}}
        out = apply_hot_ramp_peak_capture(
            df,
            artifact,
            {"hot_ramp_peak_capture": cfg},
            forecast_col="Raw_Forecast_MWH",
            evaluation_mode="shadow",
        )
        peak_correction = out.loc[
            out["Raw_Forecast_MWH"].idxmax(), "Hot_Ramp_Peak_Correction_MWH"
        ]
        self.assertAlmostEqual(float(peak_correction), 12.0, places=6)
        self.assertTrue((out["Hot_Ramp_Peak_Correction_MWH"] <= 12.0 + 1e-9).all())


class HotRampTimingAlignmentTests(unittest.TestCase):
    def test_timing_override_caps_nonselected_rows_with_subset_index(self):
        dt = pd.date_range("2026-07-22 15:00", periods=7, freq="h")
        base_by_hour = {
            15: 295.0,
            16: 310.0,
            17: 300.0,
            18: 320.0,
            19: 319.0,
            20: 318.0,
            21: 296.0,
        }
        timing_by_hour = {
            15: 295.0,
            16: 312.0,
            17: 340.0,
            18: 322.0,
            19: 319.0,
            20: 318.0,
            21: 296.0,
        }
        df = pd.DataFrame(
            [
                {
                    "DT": stamp,
                    "Hour": stamp.hour,
                    "Month": 7,
                    "Forecast_Day": 2,
                    "Final_Forecast_MWH": base_by_hour[stamp.hour],
                    "Raw_Forecast_MWH": timing_by_hour[stamp.hour],
                    "XGB_Pred_MWH": timing_by_hour[stamp.hour],
                    "Temperature_DailyMax": 102.0,
                    "DailyMaxTemp_Ramp_1Day": 2.5,
                    "CloudCover_Norm": 0.05,
                    "MWH_SameHour7DayMean": base_by_hour[stamp.hour] - 1.0,
                    "MWH_Lag24": base_by_hour[stamp.hour] - 2.0,
                }
                for stamp in dt
            ]
        )
        cfg = {
            "enabled": True,
            "shadow_mode": False,
            "hours": [16, 17, 18, 19, 20],
            "min_maxtemp_f": 100.0,
            "min_dailymax_ramp_1day_f": 2.0,
            "strong_ramp_min_dailymax_ramp_1day_f": 10.0,
            "max_cloud_cover_norm": 0.40,
            "cap_mwh": 40.0,
            "strong_ramp_cap_mwh": 40.0,
            "min_abs_correction_mwh": 0.1,
            "ramp_floor_mwh": 0.0,
            "strong_ramp_floor_mwh": 0.0,
            "anchor_support_guard_enabled": False,
            "spread_hours": 1.0,
            "peak_timing_selector": {
                "enabled": True,
                "required_source": "xgb_component",
                "timing_sources": [
                    {"source": "xgb_component", "column": "XGB_Pred_MWH"},
                    {"source": "raw_xgb_lgb", "column": "Raw_Forecast_MWH"},
                ],
                "source_priority": ["xgb_component", "raw_xgb_lgb"],
                "consensus_required": 2,
                "max_consensus_hour_spread": 0.0,
                "min_peak_margin_mwh": 1.0,
                "allowed_hours": [16, 17, 18, 19, 20],
                "block_on_strong_hot_ramp": False,
                "target_selected_hour_to_daily_peak": True,
                "cap_nonselected_hours_to_target": True,
            },
        }
        artifact = {
            "lookups": {},
            "metadata": {
                "global_peak_residual_mwh": 10.0,
                "global_samehour7_residual_mwh": 10.0,
                "global_lag24_residual_mwh": 10.0,
                "global_lag24_ramp_slope_mwh_per_f": 0.0,
            },
        }

        out = apply_hot_ramp_peak_capture(
            df,
            artifact,
            {"hot_ramp_peak_capture": cfg},
            forecast_col="Final_Forecast_MWH",
        )

        selected = out.loc[out["Hour"].eq(17)].iloc[0]
        base_peak = out.loc[out["Hour"].eq(18)].iloc[0]
        self.assertEqual(selected["Hot_Ramp_Peak_Timing_Override_Flag"], 1)
        self.assertEqual(selected["Hot_Ramp_Peak_Timing_Selected_PeakHour"], 17.0)
        self.assertGreater(
            selected["Hot_Ramp_Peak_Correction_MWH"],
            base_peak["Hot_Ramp_Peak_Correction_MWH"],
        )
        self.assertAlmostEqual(base_peak["Hot_Ramp_Peak_Correction_MWH"], 10.0)
        self.assertLessEqual(base_peak["Final_Forecast_MWH"], 330.0 + 1e-9)


class HeatPersistenceNoAnchorGuardTests(unittest.TestCase):
    def _no_anchor_frame_and_config(
        self, floor_applies_without_positive_anchor: bool
    ) -> tuple[pd.DataFrame, dict]:
        dt = pd.date_range("2026-07-15 16:00", periods=5, freq="h")
        df = pd.DataFrame(
            {
                "DT": dt,
                "Raw_Forecast_MWH": [500.0, 510.0, 520.0, 515.0, 505.0],
                "Temperature_DailyMax": [101.0] * 5,
                "CloudCover_Norm": [0.0] * 5,
                "Forecast_Day": [2] * 5,
                "ConsecutiveExtremeHotDays100": [3.0] * 5,
                "DailyMaxTemp_Ramp_1Day": [0.0] * 5,
            }
        )
        cfg = {
            "enabled": True,
            "shadow_mode": True,
            "hours": [16, 17, 18, 19, 20],
            "min_maxtemp_f": 100.0,
            "min_consecutive_extreme_days100": 3.0,
            "min_forecast_day": 1,
            "max_forecast_day": 16,
            "max_cloud_cover_norm": 0.40,
            "cap_mwh": 9.0,
            "min_abs_correction_mwh": 0.1,
            "persistence_floor_mwh": 3.0,
            "strong_persistence_floor_mwh": 3.0,
            "floor_applies_without_positive_anchor": floor_applies_without_positive_anchor,
        }
        return df, cfg

    def test_no_anchor_row_is_skipped_when_floor_requires_anchor_support(self):
        """Regression test: previously `valid_targets` was always non-empty (seeded
        unconditionally with base_peak + learned_residual, which defaults to +0.0), so the
        `heat_persistence_peak_no_anchor` branch could never fire regardless of
        floor_applies_without_positive_anchor. With no real anchor signal (no same-hour-7,
        lag24, recent-anchor, or warmer-scenario data) and the flag set False, the row must
        now be skipped with zero correction rather than silently floored.
        """
        df, cfg = self._no_anchor_frame_and_config(
            floor_applies_without_positive_anchor=False
        )
        artifact = {"metadata": {"global_peak_residual_mwh": 0.0}, "lookups": {}}
        out = apply_heat_persistence_peak_capture(
            df,
            artifact,
            {"heat_persistence_peak_capture": cfg},
            forecast_col="Raw_Forecast_MWH",
            evaluation_mode="shadow",
        )
        self.assertTrue((out["Heat_Persistence_Peak_Correction_MWH"] == 0.0).all())
        self.assertTrue(
            (
                out["Heat_Persistence_Peak_Source"] == "heat_persistence_peak_no_anchor"
            ).all()
        )

    def test_no_anchor_row_still_floored_when_flag_allows_it(self):
        """With floor_applies_without_positive_anchor left True (current production config),
        behavior is unchanged: a no-anchor row still gets floored, matching pre-fix behavior.
        """
        df, cfg = self._no_anchor_frame_and_config(
            floor_applies_without_positive_anchor=True
        )
        artifact = {"metadata": {"global_peak_residual_mwh": 0.0}, "lookups": {}}
        out = apply_heat_persistence_peak_capture(
            df,
            artifact,
            {"heat_persistence_peak_capture": cfg},
            forecast_col="Raw_Forecast_MWH",
            evaluation_mode="shadow",
        )
        peak_correction = out.loc[
            out["Raw_Forecast_MWH"].idxmax(), "Heat_Persistence_Peak_Correction_MWH"
        ]
        self.assertAlmostEqual(float(peak_correction), 3.0, places=6)


if __name__ == "__main__":
    unittest.main()
