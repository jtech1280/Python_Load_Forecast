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
