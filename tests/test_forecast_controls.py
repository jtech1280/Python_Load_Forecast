import unittest
from pathlib import Path

import yaml

from forecasting.main import _disable_windows_platform_wmi_probe

_disable_windows_platform_wmi_probe()

import numpy as np
import pandas as pd

from forecasting.features.solar_features import add_solar_features
from forecasting.backtest.rolling_backtest import (
    PRED_COLS as ROLLING_BACKTEST_PRED_COLS,
)
from forecasting.backtest.rolling_origin_replay import (
    PRED_COLS as ROLLING_REPLAY_PRED_COLS,
    _apply_replay_focused_guard,
)
from forecasting.forecast.focused_scorecard_guard import (
    apply_focused_scorecard_guard,
    build_focused_scorecard_rule_audit,
)
from forecasting.forecast.focused_shape_residual_learner import (
    apply_focused_shape_residual_learner,
    build_focused_shape_residual_learner,
)
from forecasting.diagnostics.forecast_diagnostics import (
    _diagnostic_band_for_row,
    apply_multisummer_heat_analog_shadow,
    build_daily_peak_window_miss_by_stage,
    build_daily_peak_shadow_window_scorecard,
    build_extreme_heat_peak_metrics_by_stage,
    build_extreme_heat_peak_scorecard,
    build_future_peak_muting_audit,
    build_heat_persistence_peak_candidate_scorecard,
    build_hot_ramp_peak_candidate_scorecard,
    build_hot_peak_shadow_candidate_scorecard,
    build_peak_window_expansion_scorecard,
    build_shadow_stage_promotion_audit,
    metrics_summary,
    prep_backtest,
)
from forecasting.forecast.forecast_pipeline import (
    apply_operational_stage_selector,
    build_display_df,
)
from forecasting.forecast.peak_risk_correction import (
    apply_day1_live_ramp_override,
    apply_hot_ramp_scenario_override,
    apply_multiday_live_heat_anchor_override,
    apply_peak_risk_correction,
)
from forecasting.forecast.recursive_engine import recursive_forecast
from forecasting.forecast.recent_residual_correction import (
    apply_recent_residual_correction,
    build_recent_residual_profile,
    simulate_recent_residual_correction_backtest,
)
from forecasting.forecast.uncertainty_bands import (
    _band_risk_multiplier,
    _prep,
    apply_bands,
)
from forecasting.forecast.weather_robustness_hedge import apply_weather_robustness_hedge
from forecasting.forecast.operational_residual_learner import (
    apply_operational_residual_learner,
    build_operational_residual_learner,
    simulate_operational_residual_learner_backtest,
)
from forecasting.forecast.daily_peak_shadow_model import (
    apply_daily_peak_shadow_model,
    build_daily_peak_shadow_model,
    daily_peak_shadow_summary,
)
from forecasting.forecast.hot_ramp_peak_capture import (
    apply_heat_persistence_peak_capture,
    apply_hot_ramp_peak_capture,
    build_heat_persistence_peak_capture_artifact,
    build_hot_ramp_peak_capture_artifact,
)
from forecasting.model.ensemble import blend_predictions
from forecasting.data.local_weather_loader import apply_dynamic_temperature_calibration


class ForecastControlTests(unittest.TestCase):
    def test_config_targeted_missing_slices_are_bounded_production_rules(self):
        config_path = (
            Path(__file__).resolve().parents[1] / "forecasting" / "config.yaml"
        )
        with config_path.open("r", encoding="utf-8") as handle:
            config = yaml.safe_load(handle) or {}

        rules = (
            config.get("calibration", {})
            .get("hot_ramp_peak_capture", {})
            .get("targeted_missing_slices", [])
        )
        enabled = [
            str(rule.get("name", ""))
            for rule in rules
            if bool(rule.get("enabled", True))
        ]
        expected_enabled = [
            "july_long_98_100_clear_moderate_state_peak_capture",
            "july_long_100_105_clear_high_state_peak_capture",
            "july_days2to3_90_92_clear_neutral_state_peak_capture",
            "june_days4to7_92_95_clear_moderate_state_peak_capture",
            "july_days4to7_90_92_overcast_deep_low_state_peak_capture",
            "july_days4to7_90_92_overcast_deep_low_state_he14_15_peak_capture",
            "july_days2to3_85_90_clear_deep_low_state_peak_capture",
            "july_days2to3_90_92_clear_negative_state_peak_capture",
            "july_days4to7_95_98_clear_neutral_state_peak_capture",
            "july_days4to7_95_98_clear_mild_negative_state_peak_capture",
            "july_days4to7_95_100_clear_high_state_peak_capture",
            "july_long_98_100_clear_negative_state_peak_capture",
            "july_long_90_92_clear_negative_state_peak_capture",
        ]

        self.assertTrue(rules)
        self.assertEqual(enabled, expected_enabled)
        disabled = [
            str(rule.get("name", ""))
            for rule in rules
            if not bool(rule.get("enabled", True))
        ]
        self.assertIn("july_long_95_98_clear_positive_state_peak_capture", disabled)
        self.assertIn("june_long_100_105_clear_very_high_state_peak_capture", disabled)
        rule_by_name = {str(rule.get("name", "")): rule for rule in rules}
        july_98_100 = rule_by_name["july_long_98_100_clear_moderate_state_peak_capture"]
        self.assertEqual([int(hour) for hour in july_98_100.get("hours", [])], [17, 18])
        self.assertGreaterEqual(
            float(july_98_100.get("min_raw_minus_samehour7_mwh", 0.0)), 20.0
        )
        self.assertLessEqual(
            float(july_98_100.get("max_raw_minus_samehour7_mwh_exclusive", 999.0)),
            25.0,
        )
        july_100_105 = rule_by_name["july_long_100_105_clear_high_state_peak_capture"]
        self.assertEqual(
            [int(hour) for hour in july_100_105.get("hours", [])], [16, 17]
        )
        self.assertGreaterEqual(float(july_100_105.get("min_forecast_day", 0.0)), 15.0)
        self.assertLessEqual(float(july_100_105.get("max_forecast_day", 999.0)), 16.0)
        july_95_98_neutral = rule_by_name[
            "july_days4to7_95_98_clear_neutral_state_peak_capture"
        ]
        self.assertEqual(float(july_95_98_neutral.get("correction_mwh", 0.0)), 7.0)
        july_overcast_he14_15 = rule_by_name[
            "july_days4to7_90_92_overcast_deep_low_state_he14_15_peak_capture"
        ]
        self.assertEqual(
            [int(hour) for hour in july_overcast_he14_15.get("hours", [])], [14, 15]
        )
        self.assertEqual(float(july_overcast_he14_15.get("correction_mwh", 0.0)), 6.0)
        july_95_98_mild_negative = rule_by_name[
            "july_days4to7_95_98_clear_mild_negative_state_peak_capture"
        ]
        self.assertGreaterEqual(
            float(july_95_98_mild_negative.get("min_raw_minus_samehour7_mwh", 0.0)),
            -15.0,
        )
        self.assertLessEqual(
            float(
                july_95_98_mild_negative.get(
                    "max_raw_minus_samehour7_mwh_exclusive", 999.0
                )
            ),
            -10.0,
        )
        self.assertLessEqual(
            float(july_95_98_mild_negative.get("correction_mwh", 0.0)), 4.0
        )
        for rule in rules:
            if not bool(rule.get("enabled", True)):
                continue
            months = {int(month) for month in rule.get("months", [])}
            hours = {int(hour) for hour in rule.get("hours", [])}
            self.assertTrue(months.issubset({6, 7}))
            self.assertTrue(hours.issubset(set(range(14, 21))))
            self.assertIn("min_forecast_day", rule)
            self.assertLessEqual(float(rule.get("max_forecast_day", 999.0)), 16.0)
            self.assertTrue(
                "min_maxtemp_f" in rule or "max_maxtemp_f_exclusive" in rule
            )
            self.assertTrue(
                "min_raw_minus_samehour7_mwh" in rule
                or "max_raw_minus_samehour7_mwh_exclusive" in rule
            )
            self.assertLessEqual(float(rule.get("correction_mwh", 0.0)), 14.0)

    def test_config_keeps_current_learned_candidates_guarded_for_production(self):
        config_path = (
            Path(__file__).resolve().parents[1] / "forecasting" / "config.yaml"
        )
        with config_path.open("r", encoding="utf-8") as handle:
            config = yaml.safe_load(handle) or {}
        cal = config.get("calibration", {})
        stage_selector = cal.get("stage_selector", {})

        focused_shape = stage_selector.get("focused_shape_residual_learner", {})
        self.assertFalse(focused_shape.get("shadow_mode", True))
        self.assertTrue(
            focused_shape.get("apply_correction_to_reference_forecast", False)
        )
        focused_shape_guard = focused_shape.get("promotion_delta_guard", {})
        self.assertTrue(focused_shape_guard.get("enabled", False))
        self.assertLessEqual(
            float(
                focused_shape_guard.get(
                    "risk_slice_max_abs_delta_vs_reference_mwh", 999.0
                )
            ),
            5.0,
        )
        self.assertFalse(
            (focused_shape.get("scope", {}) or {}).get(
                "use_focused_guard_rule_union", True
            )
        )
        self.assertTrue(
            stage_selector.get("daily_peak_shadow_model", {}).get("shadow_mode", False)
        )
        self.assertTrue(
            cal.get("heat_persistence_peak_capture", {}).get("shadow_mode", False)
        )

        orl = cal.get("operational_residual_learner", {})
        self.assertEqual(orl.get("production_scope"), "capped_full_shadow")
        self.assertNotEqual(orl.get("production_scope"), "all")
        self.assertLessEqual(float(orl.get("capped_full_shadow_cap_mwh", 999.0)), 1.0)
        hot_peak = orl.get("hot_peak", {})
        self.assertTrue(hot_peak.get("structural_shadow", {}).get("enabled", False))
        self.assertTrue(hot_peak.get("broad_shadow", {}).get("enabled", False))
        hot_ramp_peak = cal.get("hot_ramp_peak_capture", {})
        timing_selector = hot_ramp_peak.get("peak_timing_selector", {})
        self.assertTrue(timing_selector.get("enabled", False))
        self.assertEqual(timing_selector.get("required_source"), "xgb_component")
        self.assertGreaterEqual(int(timing_selector.get("consensus_required", 0)), 2)
        self.assertTrue(timing_selector.get("block_on_strong_hot_ramp", False))

    def test_daily_peak_shadow_model_keeps_final_forecast_and_improves_peak_candidate(
        self,
    ):
        rows = []
        for day in range(10):
            date = pd.Timestamp("2026-07-01") + pd.Timedelta(days=day)
            for hour in range(14, 22):
                dt = date + pd.Timedelta(hours=hour)
                base = 100.0 + (hour - 14) * 2.0
                actual = base
                if hour == 20:
                    actual = 126.0
                elif hour == 21:
                    actual = 118.0
                rows.append(
                    {
                        "DT": dt,
                        "Hour": hour,
                        "Month": dt.month,
                        "DOW": dt.dayofweek,
                        "Forecast_Day": 1,
                        "Actual_MWH": actual,
                        "Final_Backtest_Forecast_MWH": base,
                        "Final_Forecast_MWH": base,
                        "Raw_Forecast_MWH": base - 1.0,
                        "XGB_Pred_MWH": base - 2.0,
                        "LGB_Pred_MWH": base,
                        "CatBoost_Pred_MWH": base + 1.0,
                        "Prophet_Pred_MWH": base - 3.0,
                        "MWH_SameHour7DayMean": base - 5.0,
                        "MWH_Lag24": base - 4.0,
                        "Temperature_DailyMax": 101.0,
                        "Temperature": 101.0 - max(0, hour - 17) * 1.5,
                        "CloudCover_Norm": 0.0,
                        "TempDrop_Next3Hr_F": max(0, hour - 17) * 2.0,
                        "DeltaBreeze_Cooling_Flag": 1 if hour >= 18 else 0,
                        "DeltaBreeze_Westerly_Flow_Flag": 1 if hour >= 18 else 0,
                        "Westerly_Flow_Mph": 8.0 if hour >= 18 else 1.0,
                    }
                )
        df = pd.DataFrame(rows)
        config = {
            "daily_peak_shadow_model": {
                "enabled": True,
                "shadow_mode": True,
                "min_days": 5,
                "min_peak_window_rows_per_day": 4,
                "peak_hours": [14, 15, 16, 17, 18, 19, 20, 21],
                "adjustment_hours": [14, 15, 16, 17, 18, 19, 20, 21],
                "target_clip_mwh": 35.0,
                "blend": 0.75,
                "cap_mwh": 12.0,
                "min_abs_correction_mwh": 0.1,
                "spread_hours": 1.0,
                "timing_model_enabled": True,
                "timing_blend": 1.0,
                "max_timing_shift_hours": 2.0,
                "learning_rate": 0.1,
                "max_iter": 30,
                "max_leaf_nodes": 8,
                "min_samples_leaf": 2,
                "timing_min_samples_leaf": 2,
                "l2_regularization": 0.0,
                "random_state": 42,
            }
        }

        artifact = build_daily_peak_shadow_model(
            df.iloc[:72],
            config,
            forecast_col="Final_Backtest_Forecast_MWH",
        )
        self.assertIsNotNone(artifact)
        future = (
            df.iloc[72:]
            .rename(columns={"Final_Backtest_Forecast_MWH": "Final_Forecast_MWH"})
            .copy()
        )
        future = future.loc[:, ~future.columns.duplicated()].copy()
        original_final = future["Final_Forecast_MWH"].copy()

        out = apply_daily_peak_shadow_model(
            future,
            artifact,
            config,
            forecast_col="Final_Forecast_MWH",
            evaluation_mode="unit_shadow",
        )
        summary = daily_peak_shadow_summary(out, artifact, config)

        self.assertTrue((out["Final_Forecast_MWH"] == original_final).all())
        self.assertGreater(out["Daily_Peak_Correction_Applied_Flag"].sum(), 0)
        self.assertGreater(
            out["Daily_Peak_Shadow_Adjusted_Forecast_MWH"].max(),
            original_final.max(),
        )
        self.assertLess(summary["delta_daily_peak_mae_mwh"], 0.0)
        self.assertEqual(out["Daily_Peak_Shadow_Mode"].iloc[0], 1)

    def test_daily_peak_shadow_model_can_gate_application_by_forecast_day(self):
        class ConstantPeakResidualModel:
            def predict(self, x):
                return np.full(len(x), 4.0)

        rows = []
        for date, forecast_day in [
            (pd.Timestamp("2026-07-01"), 1),
            (pd.Timestamp("2026-07-03"), 3),
            (pd.Timestamp("2026-07-06"), 6),
            (pd.Timestamp("2026-07-08"), 8),
        ]:
            for hour in range(14, 22):
                base = 100.0 + (10.0 if hour == 18 else abs(hour - 18))
                rows.append(
                    {
                        "DT": date + pd.Timedelta(hours=hour),
                        "Hour": hour,
                        "Month": date.month,
                        "DOW": date.dayofweek,
                        "Forecast_Day": forecast_day,
                        "Final_Forecast_MWH": base,
                        "Raw_Forecast_MWH": base,
                        "Temperature_DailyMax": 100.0,
                        "Temperature": 96.0,
                        "CloudCover_Norm": 0.0,
                    }
                )
        future = pd.DataFrame(rows)
        config = {
            "daily_peak_shadow_model": {
                "enabled": True,
                "shadow_mode": True,
                "min_application_forecast_day": 2,
                "max_application_forecast_day": 5,
                "peak_hours": [14, 15, 16, 17, 18, 19, 20, 21],
                "adjustment_hours": [16, 17, 18, 19, 20],
                "blend": 1.0,
                "cap_mwh": 10.0,
                "min_abs_correction_mwh": 0.0,
                "spread_hours": 0.0,
                "timing_model_enabled": False,
            }
        }
        artifact = {
            "residual_model": ConstantPeakResidualModel(),
            "timing_model": None,
            "fill_values": pd.Series(dtype=float),
            "feature_columns": ["Base_DailyPeak_MWH", "Forecast_Day"],
        }

        out = apply_daily_peak_shadow_model(
            future,
            artifact,
            config,
            forecast_col="Final_Forecast_MWH",
        )

        day_1 = out[out["Forecast_Day"].eq(1)]
        day_3 = out[out["Forecast_Day"].eq(3)]
        day_6 = out[out["Forecast_Day"].eq(6)]
        day_8 = out[out["Forecast_Day"].eq(8)]
        self.assertEqual(day_3["Daily_Peak_Correction_Applied_Flag"].sum(), 1)
        self.assertAlmostEqual(
            day_3.loc[day_3["Hour"].eq(18), "Daily_Peak_Correction_MWH"].iloc[0],
            4.0,
        )
        for gated_out in [day_1, day_6, day_8]:
            self.assertEqual(gated_out["Daily_Peak_Correction_Applied_Flag"].sum(), 0)
            self.assertTrue((gated_out["Daily_Peak_Correction_MWH"] == 0.0).all())
            self.assertEqual(
                gated_out["Daily_Peak_Source"].unique().tolist(),
                ["horizon_out_of_scope"],
            )

    def test_daily_peak_shadow_window_scorecard_summarizes_configured_forecast_days(
        self,
    ):
        rows = []
        for date, forecast_day in [
            (pd.Timestamp("2026-07-01"), 1),
            (pd.Timestamp("2026-07-03"), 3),
            (pd.Timestamp("2026-07-06"), 6),
        ]:
            for hour in [16, 17, 18, 19]:
                actual = 120.0 if hour == 18 else 100.0 + hour
                base = actual - 4.0 if forecast_day == 3 and hour == 18 else actual
                shadow = actual - 1.0 if forecast_day == 3 and hour == 18 else base
                rows.append(
                    {
                        "DT": date + pd.Timedelta(hours=hour),
                        "Forecast_Day": forecast_day,
                        "Hour": hour,
                        "Actual_MWH": actual,
                        "Raw_Forecast_MWH": base,
                        "Final_Backtest_Forecast_MWH": base,
                        "Daily_Peak_Shadow_Adjusted_Forecast_MWH": shadow,
                        "Daily_Peak_Correction_Applied_Flag": int(
                            forecast_day == 3 and hour == 18
                        ),
                    }
                )
        config = {
            "calibration": {
                "stage_selector": {
                    "daily_peak_shadow_model": {
                        "min_application_forecast_day": 2,
                        "max_application_forecast_day": 5,
                        "peak_hours": [16, 17, 18, 19],
                    }
                }
            }
        }

        scorecard = build_daily_peak_shadow_window_scorecard(
            pd.DataFrame(rows), config=config
        )

        configured = scorecard[
            scorecard["Slice"].eq("configured_window_days_2_to_5")
        ].iloc[0]
        day_1 = scorecard[scorecard["Slice"].eq("forecast_day_1")].iloc[0]
        day_3 = scorecard[scorecard["Slice"].eq("forecast_day_3")].iloc[0]
        day_6 = scorecard[scorecard["Slice"].eq("forecast_day_6")].iloc[0]
        self.assertEqual(configured["N_HourlyRows"], 4)
        self.assertEqual(configured["Applied_HourlyRows"], 1)
        self.assertAlmostEqual(configured["Delta_PeakAtActual_MAE_MWH"], -3.0)
        self.assertAlmostEqual(day_3["Delta_PeakAtActual_MAE_MWH"], -3.0)
        self.assertEqual(day_1["Applied_HourlyRows"], 0)
        self.assertEqual(day_6["Applied_HourlyRows"], 0)
        self.assertAlmostEqual(day_1["Delta_PeakAtActual_MAE_MWH"], 0.0)
        self.assertAlmostEqual(day_6["Delta_PeakAtActual_MAE_MWH"], 0.0)

    def test_peak_window_expansion_extreme_heat_and_he18_20_daily_peak_metrics(self):
        rows = []
        for forecast_day, date, daily_max in [
            (1, pd.Timestamp("2026-07-01"), 106.0),
            (2, pd.Timestamp("2026-07-02"), 104.0),
        ]:
            for hour in range(14, 21):
                actual = 300.0 + hour + (10.0 if hour == 19 else 0.0)
                final = actual - (4.0 if hour in {19, 20} else 2.0)
                rows.append(
                    {
                        "DT": date + pd.Timedelta(hours=hour),
                        "Forecast_Day": forecast_day,
                        "Replay_Horizon_Bucket": "Days1to3",
                        "Hour": hour,
                        "Month": date.month,
                        "Actual_MWH": actual,
                        "Raw_Forecast_MWH": final - 1.0,
                        "Final_Backtest_Forecast_MWH": final,
                        "Temperature_DailyMax": daily_max,
                        "Temperature": daily_max,
                        "CloudCover_Norm": 0.0,
                    }
                )
        df = pd.DataFrame(rows)

        expansion = build_peak_window_expansion_scorecard(df)
        current = expansion[expansion["Slice"].eq("PeakWindowHours14to18")].iloc[0]
        expanded = expansion[expansion["Slice"].eq("PeakWindowHours14to20")].iloc[0]
        late = expansion[expansion["Slice"].eq("LatePeakHours19to20")].iloc[0]
        self.assertEqual(current["N"], 10)
        self.assertEqual(expanded["N"], 14)
        self.assertEqual(late["N"], 4)

        extreme_scorecard = build_extreme_heat_peak_scorecard(df)
        extreme = extreme_scorecard[
            extreme_scorecard["Slice"].eq("ExtremeHeat105PlusPeakWindowHours14to20")
        ].iloc[0]
        self.assertEqual(extreme["N"], 7)

        extreme_metrics = build_extreme_heat_peak_metrics_by_stage(df)
        final_extreme = extreme_metrics[
            extreme_metrics["Stage"].eq("final_corrected_production")
        ]
        self.assertEqual(set(final_extreme["Forecast_Day"].astype(int)), {1})
        self.assertTrue(
            {17, 18, 19, 20}.issubset(set(final_extreme["Hour"].astype(int)))
        )

        he18_20 = build_daily_peak_window_miss_by_stage(df)
        final_peaks = he18_20[he18_20["Stage"].eq("final_corrected_production")]
        self.assertEqual(
            set(final_peaks["PeakWindowName"]), {"HE18to20_CodeHours17to19"}
        )
        self.assertEqual(set(final_peaks["Actual_Peak_Hour"].astype(int)), {19})

    def test_future_peak_muting_audit_flags_hotter_day_lower_peak_and_scenario_gap(
        self,
    ):
        rows = []
        for date, daily_max, peak_value, hot_stress_peak in [
            (pd.Timestamp("2026-08-07"), 108.0, 346.0, 356.0),
            (pd.Timestamp("2026-08-08"), 109.5, 342.0, 354.0),
        ]:
            stress_gap = hot_stress_peak - peak_value
            for hour in range(14, 21):
                point = peak_value - abs(hour - 18) * 12.0
                hot_stress = point + stress_gap
                rows.append(
                    {
                        "DT": date + pd.Timedelta(hours=hour),
                        "Forecast_Expected_MWH": point,
                        "Forecast": point,
                        "Temperature": daily_max - abs(hour - 17) * 0.5,
                        "Temperature_DailyMax": daily_max,
                        "WeatherScenario_warmer_P50_MWH": hot_stress - 2.0,
                        "WeatherScenario_hot_stress_5f_P50_MWH": hot_stress,
                        "WeatherScenario_cooler_P50_MWH": point - 10.0,
                        "Weather_Robustness_Hedge_MWH": 0.0,
                    }
                )

        backtest = pd.DataFrame(
            {
                "DT": pd.to_datetime(["2026-07-15 17:00", "2026-07-15 18:00"]),
                "Hour": [17, 18],
                "Actual_MWH": [341.0, 340.0],
                "Raw_Forecast_MWH": [338.0, 337.0],
                "Final_Backtest_Forecast_MWH": [341.0, 340.0],
                "Temperature_DailyMax": [105.0, 105.0],
            }
        )

        audit = build_future_peak_muting_audit(pd.DataFrame(rows), backtest_df=backtest)

        aug8 = audit[audit["Date"].eq("2026-08-08")].iloc[0]
        self.assertTrue(bool(aug8["Muted_Peak_Flag"]))
        self.assertIn("hotter_day_lower_peak", aug8["Muted_Peak_Reasons"])
        self.assertIn("daily_peak_below_max_scenario", aug8["Muted_Peak_Reasons"])
        self.assertEqual(aug8["Analog_Actual_Count"], 2)
        self.assertAlmostEqual(aug8["MaxPeakWindowHotStressGap_MWH"], 12.0)

    def test_multisummer_heat_analog_shadow_is_prior_origin_and_non_production(self):
        rows = []
        for dt, actual, final in [
            (pd.Timestamp("2024-07-10 17:00"), 330.0, 320.0),
            (pd.Timestamp("2025-07-10 17:00"), 350.0, 330.0),
            (pd.Timestamp("2026-07-05 17:00"), 500.0, 300.0),
            (pd.Timestamp("2026-07-10 17:00"), 335.0, 300.0),
        ]:
            rows.append(
                {
                    "DT": dt,
                    "Replay_Origin_ID": "origin_2026_07_01",
                    "Replay_Origin_DT": pd.Timestamp("2026-07-01"),
                    "Forecast_Day": 1,
                    "Replay_Horizon_Bucket": "Days1to3",
                    "Hour": dt.hour,
                    "Month": dt.month,
                    "Actual_MWH": actual,
                    "Raw_Forecast_MWH": final - 2.0,
                    "Final_Backtest_Forecast_MWH": final,
                    "Temperature_DailyMax": 106.0,
                    "Temperature": 106.0,
                    "CloudCover_Norm": 0.0,
                }
            )
        config = {
            "diagnostics": {
                "heat_analog_shadow": {
                    "min_analog_count": 2,
                    "blend": 0.5,
                    "cap_mwh": 100.0,
                }
            }
        }

        out = apply_multisummer_heat_analog_shadow(pd.DataFrame(rows), config=config)
        target = out[out["DT"].eq(pd.Timestamp("2026-07-10 17:00"))].iloc[0]

        self.assertEqual(target["Final_Backtest_Forecast_MWH"], 300.0)
        self.assertEqual(target["Heat_Analog_Count_SameHour_PreOrigin"], 2)
        self.assertEqual(target["Heat_Analog_Source"], "same_month_multi_summer")
        self.assertAlmostEqual(
            target["Heat_Analog_Actual_Mean_SameHour_PreOrigin_MWH"], 340.0
        )
        self.assertAlmostEqual(target["Heat_Analog_Shadow_Correction_MWH"], 20.0)
        self.assertAlmostEqual(target["Heat_Analog_Shadow_Forecast_MWH"], 320.0)

        audit = build_shadow_stage_promotion_audit(out)
        self.assertIn("heat_analog_shadow", set(audit["Stage"]))

    def test_hot_ramp_peak_capture_shadow_targets_strong_ramp_peak(self):
        rows = []
        shape = {16: 100.0, 17: 104.0, 18: 108.0, 19: 106.0, 20: 102.0}
        for day in range(6):
            date = pd.Timestamp("2026-07-01") + pd.Timedelta(days=day)
            for hour, base in shape.items():
                actual = base + (6.0 if hour == 18 else 2.0)
                rows.append(
                    {
                        "DT": date + pd.Timedelta(hours=hour),
                        "Hour": hour,
                        "Month": 7,
                        "Forecast_Day": 1,
                        "Actual_MWH": actual,
                        "Final_Backtest_Forecast_MWH": base,
                        "Final_Forecast_MWH": base,
                        "Raw_Forecast_MWH": base - 1.0,
                        "Temperature": 103.0,
                        "Temperature_DailyMax": 103.0,
                        "DailyMaxTemp_Ramp_1Day": 3.5,
                        "CloudCover_Norm": 0.05,
                        "MWH_SameHour7DayMean": base - 2.0,
                        "MWH_Lag24": base - 3.0,
                    }
                )
        config = {
            "calibration": {
                "hot_ramp_peak_capture": {
                    "enabled": True,
                    "shadow_mode": True,
                    "min_training_days": 3,
                    "min_lookup_days": 1,
                    "train_enforce_forecast_day": False,
                    "min_maxtemp_f": 100.0,
                    "min_dailymax_ramp_1day_f": 2.0,
                    "strong_ramp_min_dailymax_ramp_1day_f": 3.0,
                    "hours": [16, 17, 18, 19, 20],
                    "max_cloud_cover_norm": 0.40,
                    "spread_hours": 1.0,
                    "strong_ramp_floor_mwh": 4.0,
                    "cap_mwh": 9.0,
                    "strong_ramp_cap_mwh": 10.0,
                }
            }
        }
        artifact = build_hot_ramp_peak_capture_artifact(
            pd.DataFrame(rows),
            config,
            forecast_col="Final_Backtest_Forecast_MWH",
        )
        future = []
        for hour, base in shape.items():
            future.append(
                {
                    "DT": pd.Timestamp("2026-07-10") + pd.Timedelta(hours=hour),
                    "Hour": hour,
                    "Month": 7,
                    "Forecast_Day": 1,
                    "Final_Forecast_MWH": base + 2.0,
                    "Stage_Selected_Forecast_MWH": base + 2.0,
                    "Temperature": 104.0,
                    "Temperature_DailyMax": 104.0,
                    "DailyMaxTemp_Ramp_1Day": 3.5,
                    "CloudCover_Norm": 0.05,
                    "MWH_SameHour7DayMean": base,
                    "MWH_Lag24": base - 1.0,
                }
            )

        out = apply_hot_ramp_peak_capture(
            pd.DataFrame(future),
            artifact,
            config,
            forecast_col="Final_Forecast_MWH",
            also_update_cols=("Stage_Selected_Forecast_MWH",),
            evaluation_mode="unit_test_shadow",
        )

        peak = out.loc[out["Hour"].eq(18)].iloc[0]
        self.assertIsNotNone(artifact)
        self.assertEqual(peak["Hot_Ramp_Peak_Source"], "hot_ramp_peak_shadow")
        self.assertEqual(peak["Hot_Ramp_Peak_Shadow_Mode"], 1)
        self.assertEqual(peak["Hot_Ramp_Peak_Scope_Flag"], 1)
        self.assertEqual(peak["Hot_Ramp_Peak_Strong_Ramp_Flag"], 1)
        self.assertGreaterEqual(peak["Hot_Ramp_Peak_Correction_MWH"], 4.0)
        self.assertLessEqual(peak["Hot_Ramp_Peak_Correction_MWH"], 10.0)
        self.assertEqual(peak["Final_Forecast_MWH"], 110.0)
        self.assertGreater(
            peak["Hot_Ramp_Peak_Shadow_Forecast_MWH"], peak["Final_Forecast_MWH"]
        )
        self.assertEqual(
            out["Hot_Ramp_Peak_Correction_MWH"].idxmax(),
            out.loc[out["Hour"].eq(18)].index[0],
        )

    def test_hot_ramp_peak_capture_cooling_guard_blocks_shadow_lift(self):
        train = pd.DataFrame(
            {
                "DT": pd.date_range("2026-07-01 18:00", periods=5, freq="24h"),
                "Hour": 18,
                "Month": 7,
                "Actual_MWH": 116.0,
                "Final_Backtest_Forecast_MWH": 110.0,
                "Final_Forecast_MWH": 110.0,
                "Raw_Forecast_MWH": 109.0,
                "Temperature": 103.0,
                "Temperature_DailyMax": 103.0,
                "DailyMaxTemp_Ramp_1Day": 3.0,
                "CloudCover_Norm": 0.0,
            }
        )
        config = {
            "calibration": {
                "hot_ramp_peak_capture": {
                    "enabled": True,
                    "shadow_mode": True,
                    "min_training_days": 3,
                    "min_lookup_days": 1,
                    "train_enforce_forecast_day": False,
                    "hours": [18],
                    "min_maxtemp_f": 100.0,
                    "min_dailymax_ramp_1day_f": 2.0,
                    "cooling_underway_guard": {
                        "enabled": True,
                        "mode": "any",
                        "min_hour": 18,
                        "min_drop_from_dailymax_f": 5.0,
                        "min_forecast_drop_next3hr_f": 6.0,
                        "blocked_source": "hot_ramp_peak_cooling_underway_blocked",
                    },
                }
            }
        }
        artifact = build_hot_ramp_peak_capture_artifact(
            train,
            config,
            forecast_col="Final_Backtest_Forecast_MWH",
        )
        future = pd.DataFrame(
            {
                "DT": [pd.Timestamp("2026-07-10 18:00")],
                "Hour": [18],
                "Month": [7],
                "Forecast_Day": [1],
                "Final_Forecast_MWH": [110.0],
                "Temperature": [96.0],
                "Temperature_DailyMax": [103.0],
                "DailyMaxTemp_Ramp_1Day": [3.0],
                "CloudCover_Norm": [0.0],
                "TempDrop_Next3Hr_F": [7.0],
            }
        )

        out = apply_hot_ramp_peak_capture(
            future, artifact, config, forecast_col="Final_Forecast_MWH"
        )

        self.assertEqual(out["Hot_Ramp_Peak_Correction_Applied_Flag"].sum(), 0)
        self.assertEqual(
            out["Hot_Ramp_Peak_Source"].iloc[0],
            "hot_ramp_peak_cooling_underway_blocked",
        )
        self.assertEqual(out["Final_Forecast_MWH"].iloc[0], 110.0)

    def test_hot_ramp_peak_capture_blocks_negative_learned_residual(self):
        train = pd.DataFrame(
            {
                "DT": pd.date_range("2026-07-01 18:00", periods=5, freq="24h"),
                "Hour": 18,
                "Month": 7,
                "Actual_MWH": 100.0,
                "Final_Backtest_Forecast_MWH": 104.0,
                "Final_Forecast_MWH": 104.0,
                "Raw_Forecast_MWH": 104.0,
                "Temperature": 102.0,
                "Temperature_DailyMax": 102.0,
                "DailyMaxTemp_Ramp_1Day": 3.5,
                "CloudCover_Norm": 0.0,
                "MWH_SameHour7DayMean": 95.0,
                "MWH_Lag24": 95.0,
            }
        )
        config = {
            "calibration": {
                "hot_ramp_peak_capture": {
                    "enabled": True,
                    "shadow_mode": True,
                    "min_training_days": 3,
                    "min_lookup_days": 1,
                    "train_enforce_forecast_day": False,
                    "hours": [18],
                    "min_maxtemp_f": 100.0,
                    "min_dailymax_ramp_1day_f": 2.0,
                    "strong_ramp_floor_mwh": 4.0,
                    "strong_ramp_cap_mwh": 10.0,
                }
            }
        }
        artifact = build_hot_ramp_peak_capture_artifact(
            train,
            config,
            forecast_col="Final_Backtest_Forecast_MWH",
        )
        future = pd.DataFrame(
            {
                "DT": [pd.Timestamp("2026-07-10 18:00")],
                "Hour": [18],
                "Month": [7],
                "Forecast_Day": [1],
                "Final_Forecast_MWH": [100.0],
                "Temperature": [102.0],
                "Temperature_DailyMax": [102.0],
                "DailyMaxTemp_Ramp_1Day": [3.5],
                "CloudCover_Norm": [0.0],
                "MWH_SameHour7DayMean": [130.0],
                "MWH_Lag24": [130.0],
            }
        )

        out = apply_hot_ramp_peak_capture(
            future, artifact, config, forecast_col="Final_Forecast_MWH"
        )

        self.assertEqual(out["Hot_Ramp_Peak_Correction_Applied_Flag"].sum(), 0)
        self.assertEqual(
            out["Hot_Ramp_Peak_Source"].iloc[0],
            "hot_ramp_peak_negative_learned_residual",
        )
        self.assertEqual(out["Hot_Ramp_Peak_Shadow_Forecast_MWH"].iloc[0], 100.0)

    def test_hot_ramp_peak_capture_requires_anchor_support_for_strong_floor(self):
        train = pd.DataFrame(
            {
                "DT": pd.date_range("2026-07-01 18:00", periods=5, freq="24h"),
                "Hour": 18,
                "Month": 7,
                "Actual_MWH": 103.0,
                "Final_Backtest_Forecast_MWH": 100.0,
                "Final_Forecast_MWH": 100.0,
                "Raw_Forecast_MWH": 100.0,
                "Temperature": 102.0,
                "Temperature_DailyMax": 102.0,
                "DailyMaxTemp_Ramp_1Day": 3.5,
                "CloudCover_Norm": 0.0,
                "MWH_SameHour7DayMean": 98.0,
                "MWH_Lag24": 98.0,
            }
        )
        config = {
            "calibration": {
                "hot_ramp_peak_capture": {
                    "enabled": True,
                    "shadow_mode": True,
                    "min_training_days": 3,
                    "min_lookup_days": 1,
                    "train_enforce_forecast_day": False,
                    "hours": [18],
                    "min_maxtemp_f": 100.0,
                    "min_dailymax_ramp_1day_f": 2.0,
                    "strong_ramp_floor_mwh": 4.0,
                    "strong_ramp_cap_mwh": 10.0,
                    "anchor_min_learned_residual_mwh": 4.0,
                    "anchor_min_maxtemp_f": 105.0,
                    "floor_requires_anchor_support": True,
                }
            }
        }
        artifact = build_hot_ramp_peak_capture_artifact(
            train,
            config,
            forecast_col="Final_Backtest_Forecast_MWH",
        )
        future = pd.DataFrame(
            {
                "DT": [pd.Timestamp("2026-07-10 18:00")],
                "Hour": [18],
                "Month": [7],
                "Forecast_Day": [1],
                "Final_Forecast_MWH": [100.0],
                "Temperature": [102.0],
                "Temperature_DailyMax": [102.0],
                "DailyMaxTemp_Ramp_1Day": [3.5],
                "CloudCover_Norm": [0.0],
                "MWH_SameHour7DayMean": [130.0],
                "MWH_Lag24": [130.0],
            }
        )

        out = apply_hot_ramp_peak_capture(
            future, artifact, config, forecast_col="Final_Forecast_MWH"
        )

        self.assertEqual(out["Hot_Ramp_Peak_Correction_Applied_Flag"].sum(), 0)
        self.assertEqual(
            out["Hot_Ramp_Peak_Source"].iloc[0],
            "hot_ramp_peak_insufficient_anchor_support",
        )
        self.assertEqual(out["Hot_Ramp_Peak_Shadow_Forecast_MWH"].iloc[0], 100.0)

    def test_hot_ramp_peak_capture_anchorless_shadow_fallback_uses_strong_floor(self):
        shape = {16: 300.0, 17: 310.0, 18: 320.0, 19: 315.0, 20: 300.0}
        future = pd.DataFrame(
            [
                {
                    "DT": pd.Timestamp("2026-07-21") + pd.Timedelta(hours=hour),
                    "Hour": hour,
                    "Month": 7,
                    "Forecast_Day": 2,
                    "Final_Forecast_MWH": base,
                    "Temperature": 101.0,
                    "Temperature_DailyMax": 101.0,
                    "DailyMaxTemp_Ramp_1Day": 3.2,
                    "CloudCover_Norm": 0.0,
                }
                for hour, base in shape.items()
            ]
        )
        config = {
            "calibration": {
                "hot_ramp_peak_capture": {
                    "enabled": True,
                    "shadow_mode": True,
                    "allow_anchorless_shadow_fallback": True,
                    "min_maxtemp_f": 100.0,
                    "min_dailymax_ramp_1day_f": 2.0,
                    "strong_ramp_min_dailymax_ramp_1day_f": 3.0,
                    "hours": [16, 17, 18, 19, 20],
                    "max_cloud_cover_norm": 0.40,
                    "spread_hours": 1.0,
                    "strong_ramp_floor_mwh": 4.0,
                    "cap_mwh": 9.0,
                    "strong_ramp_cap_mwh": 10.0,
                }
            }
        }

        out = apply_hot_ramp_peak_capture(
            pd.DataFrame(future), None, config, forecast_col="Final_Forecast_MWH"
        )

        peak = out.loc[out["Hour"].eq(18)].iloc[0]
        self.assertEqual(peak["Hot_Ramp_Peak_Source"], "hot_ramp_peak_shadow")
        self.assertEqual(peak["Hot_Ramp_Peak_Scope_Flag"], 1)
        self.assertEqual(peak["Hot_Ramp_Peak_Strong_Ramp_Flag"], 1)
        self.assertAlmostEqual(peak["Hot_Ramp_Peak_Correction_MWH"], 4.0)
        self.assertEqual(peak["Final_Forecast_MWH"], 320.0)
        self.assertAlmostEqual(peak["Hot_Ramp_Peak_Shadow_Forecast_MWH"], 324.0)

        prod_config = {
            "calibration": {
                "hot_ramp_peak_capture": {
                    **config["calibration"]["hot_ramp_peak_capture"],
                    "shadow_mode": False,
                }
            }
        }
        blocked = apply_hot_ramp_peak_capture(
            pd.DataFrame(future),
            None,
            prod_config,
            forecast_col="Final_Forecast_MWH",
        )
        self.assertEqual(blocked["Hot_Ramp_Peak_Correction_Applied_Flag"].sum(), 0)
        self.assertEqual(
            blocked["Hot_Ramp_Peak_Source"].iloc[0], "insufficient_history"
        )

    def test_hot_ramp_peak_capture_timing_selector_uses_xgb_consensus_when_confident(
        self,
    ):
        shape = {16: 300.0, 17: 312.0, 18: 316.0, 19: 311.0, 20: 304.0}
        future = pd.DataFrame(
            [
                {
                    "DT": pd.Timestamp("2026-07-22") + pd.Timedelta(hours=hour),
                    "Hour": hour,
                    "Month": 7,
                    "Forecast_Day": 2,
                    "Final_Forecast_MWH": base,
                    "Stage_Selected_Forecast_MWH": base,
                    "Raw_Forecast_MWH": {
                        16: 303.0,
                        17: 330.0,
                        18: 322.0,
                        19: 314.0,
                        20: 304.0,
                    }[hour],
                    "XGB_Pred_MWH": {
                        16: 302.0,
                        17: 331.0,
                        18: 321.0,
                        19: 314.0,
                        20: 304.0,
                    }[hour],
                    "Temperature": 102.0,
                    "Temperature_DailyMax": 102.0,
                    "DailyMaxTemp_Ramp_1Day": 2.5,
                    "CloudCover_Norm": 0.05,
                    "MWH_SameHour7DayMean": base - 1.0,
                    "MWH_Lag24": base - 2.0,
                }
                for hour, base in shape.items()
            ]
        )
        config = {
            "calibration": {
                "hot_ramp_peak_capture": {
                    "enabled": True,
                    "shadow_mode": False,
                    "min_maxtemp_f": 100.0,
                    "min_dailymax_ramp_1day_f": 2.0,
                    "strong_ramp_min_dailymax_ramp_1day_f": 3.0,
                    "hours": [16, 17, 18, 19, 20],
                    "max_cloud_cover_norm": 0.40,
                    "spread_hours": 1.0,
                    "cap_mwh": 9.0,
                    "strong_cap_mwh": 10.0,
                    "peak_timing_selector": {
                        "enabled": True,
                        "required_source": "xgb_component",
                        "timing_sources": [
                            {"source": "xgb_component", "column": "XGB_Pred_MWH"},
                            {"source": "raw_xgb_lgb", "column": "Raw_Forecast_MWH"},
                        ],
                        "source_priority": ["xgb_component", "raw_xgb_lgb"],
                        "consensus_required": 2,
                        "max_consensus_hour_spread": 1.0,
                        "min_peak_margin_mwh": 1.0,
                        "allowed_hours": [16, 17, 18, 19, 20],
                        "block_on_strong_hot_ramp": True,
                        "strong_hot_ramp_min_maxtemp_f": 100.0,
                        "strong_hot_ramp_min_dailymax_ramp_1day_f": 3.0,
                        "target_selected_hour_to_daily_peak": True,
                        "cap_nonselected_hours_to_target": True,
                        "max_selected_hour_extra_correction_mwh": 4.0,
                    },
                }
            }
        }
        artifact = {
            "lookups": {},
            "metadata": {
                "global_peak_residual_mwh": 6.0,
                "global_samehour7_residual_mwh": 6.0,
                "global_lag24_residual_mwh": 6.0,
                "global_lag24_ramp_slope_mwh_per_f": 0.0,
            },
        }

        out = apply_hot_ramp_peak_capture(
            future,
            artifact,
            config,
            forecast_col="Final_Forecast_MWH",
            also_update_cols=("Stage_Selected_Forecast_MWH",),
        )

        selected = out.loc[out["Hour"].eq(17)].iloc[0]
        self.assertEqual(selected["Hot_Ramp_Peak_Base_PeakHour"], 18.0)
        self.assertEqual(selected["Hot_Ramp_Peak_Predicted_PeakHour"], 17.0)
        self.assertEqual(selected["Hot_Ramp_Peak_Timing_Selected_PeakHour"], 17.0)
        self.assertEqual(
            selected["Hot_Ramp_Peak_Timing_Source"], "xgb_component_consensus"
        )
        self.assertEqual(selected["Hot_Ramp_Peak_Timing_Override_Flag"], 1)
        self.assertEqual(selected["Hot_Ramp_Peak_Timing_Block_Source"], "")
        self.assertGreaterEqual(selected["Hot_Ramp_Peak_Timing_Confidence_MWH"], 1.0)
        self.assertEqual(
            out["Hot_Ramp_Peak_Correction_MWH"].idxmax(),
            out.loc[out["Hour"].eq(17)].index[0],
        )
        self.assertEqual(
            out["Final_Forecast_MWH"].idxmax(), out.loc[out["Hour"].eq(17)].index[0]
        )

    def test_hot_ramp_peak_capture_timing_selector_blocks_strong_hot_ramp_timing(self):
        shape = {16: 300.0, 17: 312.0, 18: 316.0, 19: 311.0, 20: 304.0}
        future = pd.DataFrame(
            [
                {
                    "DT": pd.Timestamp("2026-07-22") + pd.Timedelta(hours=hour),
                    "Hour": hour,
                    "Month": 7,
                    "Forecast_Day": 2,
                    "Final_Forecast_MWH": base,
                    "Raw_Forecast_MWH": {
                        16: 303.0,
                        17: 330.0,
                        18: 322.0,
                        19: 314.0,
                        20: 304.0,
                    }[hour],
                    "XGB_Pred_MWH": {
                        16: 302.0,
                        17: 331.0,
                        18: 321.0,
                        19: 314.0,
                        20: 304.0,
                    }[hour],
                    "Temperature": 103.0,
                    "Temperature_DailyMax": 103.0,
                    "DailyMaxTemp_Ramp_1Day": 3.5,
                    "CloudCover_Norm": 0.05,
                    "MWH_SameHour7DayMean": base - 1.0,
                    "MWH_Lag24": base - 2.0,
                }
                for hour, base in shape.items()
            ]
        )
        config = {
            "calibration": {
                "hot_ramp_peak_capture": {
                    "enabled": True,
                    "shadow_mode": False,
                    "min_maxtemp_f": 100.0,
                    "min_dailymax_ramp_1day_f": 2.0,
                    "strong_ramp_min_dailymax_ramp_1day_f": 3.0,
                    "hours": [16, 17, 18, 19, 20],
                    "max_cloud_cover_norm": 0.40,
                    "spread_hours": 1.0,
                    "cap_mwh": 9.0,
                    "strong_cap_mwh": 10.0,
                    "peak_timing_selector": {
                        "enabled": True,
                        "required_source": "xgb_component",
                        "timing_sources": [
                            {"source": "xgb_component", "column": "XGB_Pred_MWH"},
                            {"source": "raw_xgb_lgb", "column": "Raw_Forecast_MWH"},
                        ],
                        "source_priority": ["xgb_component", "raw_xgb_lgb"],
                        "consensus_required": 2,
                        "max_consensus_hour_spread": 1.0,
                        "min_peak_margin_mwh": 1.0,
                        "allowed_hours": [16, 17, 18, 19, 20],
                        "block_on_strong_hot_ramp": True,
                        "strong_hot_ramp_min_maxtemp_f": 100.0,
                        "strong_hot_ramp_min_dailymax_ramp_1day_f": 3.0,
                    },
                }
            }
        }
        artifact = {
            "lookups": {},
            "metadata": {
                "global_peak_residual_mwh": 6.0,
                "global_samehour7_residual_mwh": 6.0,
                "global_lag24_residual_mwh": 6.0,
                "global_lag24_ramp_slope_mwh_per_f": 0.0,
            },
        }

        out = apply_hot_ramp_peak_capture(
            future, artifact, config, forecast_col="Final_Forecast_MWH"
        )

        peak = out.loc[out["Hour"].eq(18)].iloc[0]
        self.assertEqual(peak["Hot_Ramp_Peak_Base_PeakHour"], 18.0)
        self.assertEqual(peak["Hot_Ramp_Peak_Predicted_PeakHour"], 18.0)
        self.assertEqual(peak["Hot_Ramp_Peak_Timing_Selected_PeakHour"], 18.0)
        self.assertEqual(
            peak["Hot_Ramp_Peak_Timing_Source"], "base_strong_hot_ramp_fallback"
        )
        self.assertEqual(peak["Hot_Ramp_Peak_Timing_Override_Flag"], 0)
        self.assertEqual(
            peak["Hot_Ramp_Peak_Timing_Block_Source"], "strong_hot_ramp_timing_blocked"
        )
        self.assertEqual(
            out["Hot_Ramp_Peak_Correction_MWH"].idxmax(),
            out.loc[out["Hour"].eq(18)].index[0],
        )

    def test_hot_ramp_peak_capture_targeted_slice_can_run_without_artifact(self):
        config = {
            "calibration": {
                "hot_ramp_peak_capture": {
                    "enabled": True,
                    "shadow_mode": False,
                    "min_maxtemp_f": 100.0,
                    "min_dailymax_ramp_1day_f": 2.0,
                    "max_forecast_day": 7,
                    "hours": [16, 17, 18, 19, 20],
                    "targeted_missing_slices": [
                        {
                            "source": "unit_targeted_no_artifact_july_long_98_100_clear",
                            "months": [7],
                            "min_forecast_day": 8,
                            "max_forecast_day": 16,
                            "hours": [16, 17, 18, 19, 20],
                            "min_maxtemp_f": 98.0,
                            "max_maxtemp_f_exclusive": 100.0,
                            "max_cloud_cover_norm": 0.10,
                            "min_raw_minus_samehour7_mwh": 10.0,
                            "max_raw_minus_samehour7_mwh_exclusive": 25.0,
                            "correction_mwh": 6.0,
                        }
                    ],
                }
            }
        }
        future = pd.DataFrame(
            [
                {
                    "DT": pd.Timestamp("2026-07-20 17:00"),
                    "Hour": 17,
                    "Month": 7,
                    "Forecast_Day": 10,
                    "Final_Forecast_MWH": 300.0,
                    "Stage_Selected_Forecast_MWH": 300.0,
                    "Raw_Forecast_MWH": 315.0,
                    "MWH_SameHour7DayMean": 300.0,
                    "Temperature": 96.0,
                    "Temperature_DailyMax": 99.0,
                    "DailyMaxTemp_Ramp_1Day": 1.0,
                    "CloudCover_Norm": 0.05,
                },
                {
                    "DT": pd.Timestamp("2026-07-20 18:00"),
                    "Hour": 18,
                    "Month": 7,
                    "Forecast_Day": 10,
                    "Final_Forecast_MWH": 305.0,
                    "Stage_Selected_Forecast_MWH": 305.0,
                    "Raw_Forecast_MWH": 335.0,
                    "MWH_SameHour7DayMean": 300.0,
                    "Temperature": 96.0,
                    "Temperature_DailyMax": 99.0,
                    "DailyMaxTemp_Ramp_1Day": 1.0,
                    "CloudCover_Norm": 0.05,
                },
            ]
        )

        out = apply_hot_ramp_peak_capture(
            future,
            None,
            config,
            forecast_col="Final_Forecast_MWH",
            also_update_cols=("Stage_Selected_Forecast_MWH",),
        )

        matched = out.iloc[0]
        adjacent = out.iloc[1]
        self.assertEqual(
            matched["Hot_Ramp_Peak_Source"],
            "unit_targeted_no_artifact_july_long_98_100_clear",
        )
        self.assertEqual(matched["Hot_Ramp_Peak_Correction_Applied_Flag"], 1)
        self.assertAlmostEqual(matched["Hot_Ramp_Peak_Correction_MWH"], 6.0)
        self.assertAlmostEqual(matched["Final_Forecast_MWH"], 306.0)
        self.assertAlmostEqual(matched["Stage_Selected_Forecast_MWH"], 306.0)
        self.assertEqual(matched["Hot_Ramp_Peak_Scope_Flag"], 1)
        self.assertEqual(adjacent["Hot_Ramp_Peak_Source"], "insufficient_history")
        self.assertEqual(adjacent["Hot_Ramp_Peak_Correction_Applied_Flag"], 0)
        self.assertAlmostEqual(adjacent["Final_Forecast_MWH"], 305.0)

    def test_hot_ramp_peak_capture_targeted_slice_respects_high_guards(self):
        config = {
            "calibration": {
                "hot_ramp_peak_capture": {
                    "enabled": True,
                    "shadow_mode": False,
                    "base_forecast_threshold_mwh": 300.0,
                    "max_correction_mwh_if_base_high": 2.0,
                    "warmer_delta_threshold_mwh": 10.0,
                    "max_correction_mwh_if_warmer_delta_high": 1.0,
                    "targeted_missing_slices": [
                        {
                            "source": "unit_targeted_guarded_slice",
                            "months": [7],
                            "min_forecast_day": 8,
                            "max_forecast_day": 16,
                            "hours": [16, 17, 18, 19, 20],
                            "min_maxtemp_f": 98.0,
                            "max_maxtemp_f_exclusive": 100.0,
                            "max_cloud_cover_norm": 0.10,
                            "min_raw_minus_samehour7_mwh": 10.0,
                            "max_raw_minus_samehour7_mwh_exclusive": 25.0,
                            "correction_mwh": 6.0,
                        }
                    ],
                }
            }
        }
        future = pd.DataFrame(
            [
                {
                    "DT": pd.Timestamp("2026-07-20 17:00"),
                    "Hour": 17,
                    "Month": 7,
                    "Forecast_Day": 10,
                    "Final_Forecast_MWH": 305.0,
                    "Raw_Forecast_MWH": 320.0,
                    "MWH_SameHour7DayMean": 305.0,
                    "WeatherScenario_warmer_P50_MWH": 320.0,
                    "Temperature": 96.0,
                    "Temperature_DailyMax": 99.0,
                    "CloudCover_Norm": 0.05,
                }
            ]
        )

        out = apply_hot_ramp_peak_capture(
            future, None, config, forecast_col="Final_Forecast_MWH"
        )

        row = out.iloc[0]
        self.assertEqual(row["Hot_Ramp_Peak_Source"], "unit_targeted_guarded_slice")
        self.assertEqual(row["Hot_Ramp_Peak_Correction_Applied_Flag"], 1)
        self.assertAlmostEqual(row["Hot_Ramp_Peak_Correction_MWH"], 1.0)
        self.assertAlmostEqual(row["Final_Forecast_MWH"], 306.0)
        self.assertAlmostEqual(row["Hot_Ramp_Peak_Predicted_Target_MWH"], 306.0)
        self.assertAlmostEqual(row["Hot_Ramp_Peak_Learned_Residual_MWH"], 1.0)
        self.assertEqual(row["Hot_Ramp_Peak_Base_High_Guard_Applied"], 1)
        self.assertEqual(row["Hot_Ramp_Peak_Warmer_Delta_High_Guard_Applied"], 1)

    def test_hot_ramp_peak_capture_targets_configured_missing_slice_only(self):
        config = {
            "calibration": {
                "hot_ramp_peak_capture": {
                    "enabled": True,
                    "shadow_mode": False,
                    "min_maxtemp_f": 100.0,
                    "min_dailymax_ramp_1day_f": 2.0,
                    "max_forecast_day": 7,
                    "hours": [16, 17, 18, 19, 20],
                    "targeted_missing_slices": [
                        {
                            "source": "unit_targeted_july_long_98_100_clear",
                            "months": [7],
                            "min_forecast_day": 8,
                            "max_forecast_day": 16,
                            "hours": [16, 17, 18, 19, 20],
                            "min_maxtemp_f": 98.0,
                            "max_maxtemp_f_exclusive": 100.0,
                            "max_cloud_cover_norm": 0.10,
                            "min_raw_minus_samehour7_mwh": 10.0,
                            "max_raw_minus_samehour7_mwh_exclusive": 25.0,
                            "correction_mwh": 6.0,
                        }
                    ],
                }
            }
        }
        artifact = {
            "lookups": {},
            "metadata": {
                "global_peak_residual_mwh": 0.0,
                "global_samehour7_residual_mwh": 0.0,
                "global_lag24_residual_mwh": 0.0,
                "global_lag24_ramp_slope_mwh_per_f": 0.0,
            },
        }
        future = pd.DataFrame(
            [
                {
                    "DT": pd.Timestamp("2026-07-20 17:00"),
                    "Hour": 17,
                    "Month": 7,
                    "Forecast_Day": 10,
                    "Final_Forecast_MWH": 300.0,
                    "Stage_Selected_Forecast_MWH": 300.0,
                    "Raw_Forecast_MWH": 315.0,
                    "MWH_SameHour7DayMean": 300.0,
                    "Temperature": 96.0,
                    "Temperature_DailyMax": 99.0,
                    "DailyMaxTemp_Ramp_1Day": 1.0,
                    "CloudCover_Norm": 0.05,
                },
                {
                    "DT": pd.Timestamp("2026-07-20 18:00"),
                    "Hour": 18,
                    "Month": 7,
                    "Forecast_Day": 10,
                    "Final_Forecast_MWH": 305.0,
                    "Stage_Selected_Forecast_MWH": 305.0,
                    "Raw_Forecast_MWH": 335.0,
                    "MWH_SameHour7DayMean": 300.0,
                    "Temperature": 96.0,
                    "Temperature_DailyMax": 99.0,
                    "DailyMaxTemp_Ramp_1Day": 1.0,
                    "CloudCover_Norm": 0.05,
                },
            ]
        )

        out = apply_hot_ramp_peak_capture(
            future,
            artifact,
            config,
            forecast_col="Final_Forecast_MWH",
            also_update_cols=("Stage_Selected_Forecast_MWH",),
        )

        matched = out.iloc[0]
        adjacent = out.iloc[1]
        self.assertEqual(
            matched["Hot_Ramp_Peak_Source"], "unit_targeted_july_long_98_100_clear"
        )
        self.assertEqual(matched["Hot_Ramp_Peak_Correction_Applied_Flag"], 1)
        self.assertAlmostEqual(matched["Hot_Ramp_Peak_Correction_MWH"], 6.0)
        self.assertAlmostEqual(matched["Final_Forecast_MWH"], 306.0)
        self.assertEqual(matched["Hot_Ramp_Peak_Scope_Flag"], 1)
        self.assertEqual(adjacent["Hot_Ramp_Peak_Source"], "out_of_scope")
        self.assertEqual(adjacent["Hot_Ramp_Peak_Correction_Applied_Flag"], 0)
        self.assertAlmostEqual(adjacent["Final_Forecast_MWH"], 305.0)

        no_lead = apply_hot_ramp_peak_capture(
            future.drop(columns=["Forecast_Day"]),
            artifact,
            config,
            forecast_col="Final_Forecast_MWH",
            also_update_cols=("Stage_Selected_Forecast_MWH",),
        )
        self.assertEqual(no_lead["Hot_Ramp_Peak_Correction_Applied_Flag"].sum(), 0)
        self.assertEqual(no_lead["Hot_Ramp_Peak_Source"].iloc[0], "out_of_scope")

    def test_hot_ramp_peak_candidate_scorecard_uses_stratified_event_slice(self):
        rows = []
        for hour, actual, final, shadow in [
            (18, 110.0, 100.0, 108.0),
            (19, 106.0, 100.0, 105.0),
            (15, 100.0, 99.0, 130.0),
        ]:
            rows.append(
                {
                    "DT": pd.Timestamp("2026-07-10") + pd.Timedelta(hours=hour),
                    "Replay_Horizon_Bucket": "Day1",
                    "Hour": hour,
                    "Month": 7,
                    "Actual_MWH": actual,
                    "Raw_Forecast_MWH": final - 1.0,
                    "Final_Backtest_Forecast_MWH": final,
                    "Hot_Ramp_Peak_Shadow_Forecast_MWH": shadow,
                    "Temperature_DailyMax": 104.0,
                    "DailyMaxTemp_Ramp_1Day": 3.0,
                    "CloudCover_Norm": 0.05,
                    "DailyMaxTempBucket": "100-105",
                    "CloudCoverBucket": "Clear/Low",
                }
            )

        scorecard = build_hot_ramp_peak_candidate_scorecard(
            pd.DataFrame(rows),
            group_cols=[
                "Replay_Horizon_Bucket",
                "Month",
                "DailyMaxTempBucket",
                "CloudCoverBucket",
            ],
            min_count=2,
        )

        hot_ramp = scorecard[scorecard["Stage"].eq("hot_ramp_peak_shadow")].iloc[0]
        self.assertEqual(hot_ramp["Target_Slice"], "hot_ramp_peak_100f_ramp2_he16to20")
        self.assertEqual(hot_ramp["N"], 2)
        self.assertAlmostEqual(hot_ramp["Candidate_MAE_MWH"], 1.5)
        self.assertAlmostEqual(hot_ramp["Final_MAE_MWH"], 8.0)
        self.assertTrue(bool(hot_ramp["Promote_Slice_Candidate"]))

    def test_heat_persistence_peak_capture_shadow_targets_low_ramp_consecutive_heat_peak(
        self,
    ):
        rows = []
        shape = {16: 332.0, 17: 340.0, 18: 335.0, 19: 325.0, 20: 305.0}
        for day in range(5):
            date = pd.Timestamp("2026-08-01") + pd.Timedelta(days=day)
            consecutive = min(day + 1, 5)
            for hour, base in shape.items():
                rows.append(
                    {
                        "DT": date + pd.Timedelta(hours=hour),
                        "Hour": hour,
                        "Month": 8,
                        "Forecast_Day": 1,
                        "Actual_MWH": base + (13.0 if hour == 17 else 6.0),
                        "Final_Backtest_Forecast_MWH": base,
                        "Final_Forecast_MWH": base,
                        "Raw_Forecast_MWH": base - 1.0,
                        "Temperature": 104.0,
                        "Temperature_DailyMax": 104.0,
                        "DailyMaxTemp_Ramp_1Day": 1.4,
                        "DailyMaxTemp_3DayMean": 102.5,
                        "ConsecutiveExtremeHotDays100": consecutive,
                        "CloudCover_Norm": 0.0,
                        "MWH_SameHour7DayMean": base - 5.0,
                        "MWH_Lag24": base - 4.0,
                    }
                )
        config = {
            "calibration": {
                "heat_persistence_peak_capture": {
                    "enabled": True,
                    "shadow_mode": True,
                    "min_training_days": 2,
                    "min_lookup_days": 1,
                    "train_enforce_forecast_day": False,
                    "min_maxtemp_f": 100.0,
                    "min_consecutive_extreme_days100": 3.0,
                    "min_dailymax_3day_mean_f": 100.0,
                    "max_dailymax_ramp_1day_f": 2.0,
                    "strong_min_consecutive_extreme_days100": 3.0,
                    "hours": [16, 17, 18, 19, 20],
                    "max_cloud_cover_norm": 0.40,
                    "spread_hours": 1.0,
                    "strong_persistence_floor_mwh": 9.0,
                    "cap_mwh": 9.0,
                    "strong_cap_mwh": 9.0,
                    "floor_applies_without_positive_anchor": True,
                }
            }
        }
        artifact = build_heat_persistence_peak_capture_artifact(
            pd.DataFrame(rows),
            config,
            forecast_col="Final_Backtest_Forecast_MWH",
        )
        future = []
        for hour, base in shape.items():
            future.append(
                {
                    "DT": pd.Timestamp("2026-08-10") + pd.Timedelta(hours=hour),
                    "Hour": hour,
                    "Month": 8,
                    "Forecast_Day": 1,
                    "Final_Forecast_MWH": base,
                    "Stage_Selected_Forecast_MWH": base,
                    "Temperature": 104.0,
                    "Temperature_DailyMax": 104.0,
                    "DailyMaxTemp_Ramp_1Day": 1.4,
                    "DailyMaxTemp_3DayMean": 103.0,
                    "ConsecutiveExtremeHotDays100": 3.0,
                    "CloudCover_Norm": 0.0,
                    "MWH_SameHour7DayMean": base - 2.0,
                    "MWH_Lag24": base - 1.0,
                }
            )

        out = apply_heat_persistence_peak_capture(
            pd.DataFrame(future),
            artifact,
            config,
            forecast_col="Final_Forecast_MWH",
            also_update_cols=("Stage_Selected_Forecast_MWH",),
            evaluation_mode="unit_test_shadow",
        )

        peak = out.loc[out["Hour"].eq(17)].iloc[0]
        self.assertIsNotNone(artifact)
        self.assertEqual(
            peak["Heat_Persistence_Peak_Source"], "heat_persistence_peak_shadow"
        )
        self.assertEqual(peak["Heat_Persistence_Peak_Shadow_Mode"], 1)
        self.assertEqual(peak["Heat_Persistence_Peak_Scope_Flag"], 1)
        self.assertEqual(peak["Heat_Persistence_Peak_Strong_Flag"], 1)
        self.assertAlmostEqual(peak["Heat_Persistence_Peak_Correction_MWH"], 9.0)
        self.assertEqual(peak["Final_Forecast_MWH"], 340.0)
        self.assertAlmostEqual(peak["Heat_Persistence_Peak_Shadow_Forecast_MWH"], 349.0)
        self.assertEqual(
            out["Heat_Persistence_Peak_Correction_MWH"].idxmax(),
            out.loc[out["Hour"].eq(17)].index[0],
        )

    def test_heat_persistence_peak_capture_production_mode_updates_final_columns(self):
        shape = {16: 332.0, 17: 340.0, 18: 335.0, 19: 325.0, 20: 305.0}
        future = pd.DataFrame(
            [
                {
                    "DT": pd.Timestamp("2026-08-10") + pd.Timedelta(hours=hour),
                    "Hour": hour,
                    "Month": 8,
                    "Forecast_Day": 6,
                    "Actual_MWH": base + (12.0 if hour == 17 else 4.0),
                    "Final_Forecast_MWH": base,
                    "Stage_Selected_Forecast_MWH": base,
                    "Temperature": 104.0,
                    "Temperature_DailyMax": 104.0,
                    "DailyMaxTemp_Ramp_1Day": 1.4,
                    "DailyMaxTemp_3DayMean": 103.0,
                    "ConsecutiveExtremeHotDays100": 3.0,
                    "CloudCover_Norm": 0.0,
                }
                for hour, base in shape.items()
            ]
        )
        artifact = {
            "lookups": {},
            "metadata": {
                "global_peak_residual_mwh": 9.0,
                "global_samehour7_residual_mwh": 0.0,
                "global_lag24_residual_mwh": 0.0,
            },
        }
        config = {
            "calibration": {
                "heat_persistence_peak_capture": {
                    "enabled": True,
                    "shadow_mode": False,
                    "source": "unit_heat_persistence_peak_prod",
                    "min_maxtemp_f": 100.0,
                    "min_consecutive_extreme_days100": 3.0,
                    "min_dailymax_3day_mean_f": 100.0,
                    "max_dailymax_ramp_1day_f": 2.0,
                    "strong_min_consecutive_extreme_days100": 3.0,
                    "hours": [16, 17, 18, 19, 20],
                    "max_cloud_cover_norm": 0.40,
                    "spread_hours": 1.0,
                    "cap_mwh": 9.0,
                    "strong_cap_mwh": 9.0,
                    "floor_applies_without_positive_anchor": True,
                }
            }
        }

        out = apply_heat_persistence_peak_capture(
            future,
            artifact,
            config,
            forecast_col="Final_Forecast_MWH",
            also_update_cols=("Stage_Selected_Forecast_MWH",),
            evaluation_mode="unit_test_prod",
        )

        peak = out.loc[out["Hour"].eq(17)].iloc[0]
        self.assertEqual(peak["Heat_Persistence_Peak_Shadow_Mode"], 0)
        self.assertEqual(
            peak["Heat_Persistence_Peak_Source"], "unit_heat_persistence_peak_prod"
        )
        self.assertAlmostEqual(peak["Heat_Persistence_Peak_Correction_MWH"], 9.0)
        self.assertAlmostEqual(peak["Heat_Persistence_Peak_Shadow_Forecast_MWH"], 349.0)
        self.assertAlmostEqual(peak["Final_Forecast_MWH"], 349.0)
        self.assertAlmostEqual(peak["Stage_Selected_Forecast_MWH"], 349.0)
        self.assertAlmostEqual(peak["Final_Residual_MWH"], 3.0)

    def test_heat_persistence_peak_capture_anchorless_shadow_fallback_uses_floor(self):
        shape = {16: 332.0, 17: 340.0, 18: 335.0, 19: 325.0, 20: 305.0}
        future = pd.DataFrame(
            [
                {
                    "DT": pd.Timestamp("2026-08-03") + pd.Timedelta(hours=hour),
                    "Hour": hour,
                    "Month": 8,
                    "Forecast_Day": 6,
                    "Final_Forecast_MWH": base,
                    "Temperature": 104.0,
                    "Temperature_DailyMax": 104.0,
                    "DailyMaxTemp_Ramp_1Day": 1.0,
                    "DailyMaxTemp_3DayMean": 103.0,
                    "ConsecutiveExtremeHotDays100": 3.0,
                    "CloudCover_Norm": 0.0,
                }
                for hour, base in shape.items()
            ]
        )
        config = {
            "calibration": {
                "heat_persistence_peak_capture": {
                    "enabled": True,
                    "shadow_mode": True,
                    "allow_anchorless_shadow_fallback": True,
                    "min_maxtemp_f": 100.0,
                    "min_consecutive_extreme_days100": 3.0,
                    "min_dailymax_3day_mean_f": 100.0,
                    "max_dailymax_ramp_1day_f": 2.0,
                    "strong_min_consecutive_extreme_days100": 3.0,
                    "hours": [16, 17, 18, 19, 20],
                    "max_cloud_cover_norm": 0.40,
                    "spread_hours": 1.0,
                    "strong_persistence_floor_mwh": 9.0,
                    "cap_mwh": 9.0,
                    "strong_cap_mwh": 9.0,
                    "floor_applies_without_positive_anchor": True,
                }
            }
        }

        out = apply_heat_persistence_peak_capture(
            pd.DataFrame(future),
            None,
            config,
            forecast_col="Final_Forecast_MWH",
        )

        peak = out.loc[out["Hour"].eq(17)].iloc[0]
        self.assertEqual(
            peak["Heat_Persistence_Peak_Source"], "heat_persistence_peak_shadow"
        )
        self.assertEqual(peak["Heat_Persistence_Peak_Scope_Flag"], 1)
        self.assertEqual(peak["Heat_Persistence_Peak_Strong_Flag"], 1)
        self.assertAlmostEqual(peak["Heat_Persistence_Peak_Correction_MWH"], 9.0)
        self.assertEqual(peak["Final_Forecast_MWH"], 340.0)
        self.assertAlmostEqual(peak["Heat_Persistence_Peak_Shadow_Forecast_MWH"], 349.0)

    def test_heat_persistence_peak_capture_anchorless_production_fallback_updates_final(
        self,
    ):
        shape = {16: 332.0, 17: 340.0, 18: 335.0, 19: 325.0, 20: 305.0}
        future = pd.DataFrame(
            [
                {
                    "DT": pd.Timestamp("2026-08-03") + pd.Timedelta(hours=hour),
                    "Hour": hour,
                    "Month": 8,
                    "Forecast_Day": 6,
                    "Actual_MWH": base + (12.0 if hour == 17 else 4.0),
                    "Final_Forecast_MWH": base,
                    "Stage_Selected_Forecast_MWH": base,
                    "Temperature": 104.0,
                    "Temperature_DailyMax": 104.0,
                    "DailyMaxTemp_Ramp_1Day": 1.0,
                    "DailyMaxTemp_3DayMean": 103.0,
                    "ConsecutiveExtremeHotDays100": 3.0,
                    "CloudCover_Norm": 0.0,
                }
                for hour, base in shape.items()
            ]
        )
        config = {
            "calibration": {
                "heat_persistence_peak_capture": {
                    "enabled": True,
                    "shadow_mode": False,
                    "source": "unit_heat_persistence_peak_prod",
                    "allow_anchorless_fallback": True,
                    "allow_anchorless_shadow_fallback": True,
                    "min_maxtemp_f": 100.0,
                    "min_consecutive_extreme_days100": 3.0,
                    "min_dailymax_3day_mean_f": 100.0,
                    "max_dailymax_ramp_1day_f": 2.0,
                    "strong_min_consecutive_extreme_days100": 3.0,
                    "hours": [16, 17, 18, 19, 20],
                    "max_cloud_cover_norm": 0.40,
                    "spread_hours": 1.0,
                    "strong_persistence_floor_mwh": 9.0,
                    "cap_mwh": 9.0,
                    "strong_cap_mwh": 9.0,
                    "floor_applies_without_positive_anchor": True,
                }
            }
        }

        out = apply_heat_persistence_peak_capture(
            future,
            None,
            config,
            forecast_col="Final_Forecast_MWH",
            also_update_cols=("Stage_Selected_Forecast_MWH",),
        )

        peak = out.loc[out["Hour"].eq(17)].iloc[0]
        self.assertEqual(peak["Heat_Persistence_Peak_Shadow_Mode"], 0)
        self.assertEqual(
            peak["Heat_Persistence_Peak_Source"], "unit_heat_persistence_peak_prod"
        )
        self.assertAlmostEqual(peak["Heat_Persistence_Peak_Correction_MWH"], 9.0)
        self.assertAlmostEqual(peak["Final_Forecast_MWH"], 349.0)
        self.assertAlmostEqual(peak["Stage_Selected_Forecast_MWH"], 349.0)

    def test_heat_persistence_peak_candidate_scorecard_uses_consecutive_heat_stratum(
        self,
    ):
        rows = []
        for hour, actual, final, shadow, consecutive in [
            (17, 113.0, 100.0, 109.0, 3.0),
            (18, 110.0, 101.0, 109.0, 3.0),
            (19, 100.0, 98.0, 130.0, 2.0),
        ]:
            rows.append(
                {
                    "DT": pd.Timestamp("2026-08-03") + pd.Timedelta(hours=hour),
                    "Replay_Horizon_Bucket": "Day1",
                    "Hour": hour,
                    "Month": 8,
                    "Actual_MWH": actual,
                    "Raw_Forecast_MWH": final - 1.0,
                    "Final_Backtest_Forecast_MWH": final,
                    "Heat_Persistence_Peak_Shadow_Forecast_MWH": shadow,
                    "Temperature_DailyMax": 104.0,
                    "ConsecutiveExtremeHotDays100": consecutive,
                    "CloudCover_Norm": 0.0,
                    "DailyMaxTempBucket": "100-105",
                    "CloudCoverBucket": "Clear/Low",
                }
            )

        scorecard = build_heat_persistence_peak_candidate_scorecard(
            pd.DataFrame(rows),
            group_cols=[
                "Replay_Horizon_Bucket",
                "Month",
                "DailyMaxTempBucket",
                "CloudCoverBucket",
            ],
            min_count=2,
        )

        persistence = scorecard[
            scorecard["Stage"].eq("heat_persistence_peak_shadow")
        ].iloc[0]
        self.assertEqual(
            persistence["Target_Slice"], "heat_persistence_peak_100f_consec3_he16to20"
        )
        self.assertEqual(persistence["N"], 2)
        self.assertAlmostEqual(persistence["Candidate_MAE_MWH"], 2.5)
        self.assertAlmostEqual(persistence["Final_MAE_MWH"], 11.0)
        self.assertTrue(bool(persistence["Promote_Slice_Candidate"]))

    def test_operational_residual_learner_shadow_keeps_final_forecast(self):
        dt = pd.date_range("2026-07-01", periods=72, freq="h")
        train = pd.DataFrame(
            {
                "DT": dt,
                "Hour": dt.hour,
                "Month": dt.month,
                "Actual_MWH": 108.0,
                "Final_Backtest_Forecast_MWH": 100.0,
                "Final_Residual_MWH": 8.0,
                "Raw_Forecast_MWH": 98.0,
                "XGB_Pred_MWH": 99.0,
                "LGB_Pred_MWH": 100.0,
                "CatBoost_Pred_MWH": 97.0,
                "Prophet_Pred_MWH": 110.0,
                "Temperature_DailyMax": 101.0,
                "Temperature": 98.0,
                "CloudCover_Norm": 0.0,
            }
        )
        config = {
            "operational_residual_learner": {
                "enabled": True,
                "shadow_mode": True,
                "min_rows": 24,
                "min_samples_leaf": 2,
                "max_iter": 20,
                "blend": 0.5,
                "cap_mwh": 6.0,
                "total_cap_mwh": 6.0,
                "hot_peak": {"enabled": False},
            }
        }
        artifact = build_operational_residual_learner(
            train,
            config,
            forecast_col="Final_Backtest_Forecast_MWH",
        )
        future = (
            train.tail(2)
            .rename(columns={"Final_Backtest_Forecast_MWH": "Final_Forecast_MWH"})
            .copy()
        )
        original = future["Final_Forecast_MWH"].copy()

        out = apply_operational_residual_learner(
            future,
            artifact,
            config,
            forecast_col="Final_Forecast_MWH",
        )

        self.assertTrue((out["Final_Forecast_MWH"] == original).all())
        self.assertTrue((out["Auto_Residual_Correction_MWH"] > 0.0).all())
        self.assertTrue((out["Auto_Residual_Adjusted_Forecast_MWH"] > original).all())
        self.assertTrue((out["Auto_Residual_Shadow_Mode"] == 1).all())

    def test_operational_residual_learner_hot_peak_gate_requires_low_forecast_signal(
        self,
    ):
        dt = pd.date_range("2026-07-01", periods=96, freq="h")
        train = pd.DataFrame(
            {
                "DT": dt,
                "Hour": dt.hour,
                "Month": dt.month,
                "Actual_MWH": 108.0,
                "Final_Backtest_Forecast_MWH": 100.0,
                "Final_Residual_MWH": 8.0,
                "Raw_Forecast_MWH": 98.0,
                "MWH_SameHour7DayMean": 80.0,
                "XGB_Pred_MWH": 99.0,
                "LGB_Pred_MWH": 100.0,
                "CatBoost_Pred_MWH": 97.0,
                "Prophet_Pred_MWH": 110.0,
                "Temperature_DailyMax": 101.0,
                "Temperature": 98.0,
                "CloudCover_Norm": 0.0,
            }
        )
        config = {
            "operational_residual_learner": {
                "enabled": True,
                "shadow_mode": True,
                "min_rows": 24,
                "min_samples_leaf": 2,
                "max_iter": 20,
                "blend": 0.5,
                "cap_mwh": 6.0,
                "total_cap_mwh": 6.0,
                "hot_peak": {
                    "enabled": True,
                    "min_rows": 2,
                    "min_samples_leaf": 2,
                    "hours": [17],
                    "min_maxtemp_f": 90.0,
                    "blend": 0.5,
                    "cap_mwh": 6.0,
                    "positive_gate": {
                        "enabled": True,
                        "min_raw_minus_samehour_7day_mean_mwh": 20.0,
                        "max_raw_minus_samehour_7day_mean_mwh": 35.0,
                        "min_raw_minus_samehour_yesterday_mwh": 20.0,
                        "max_final_minus_raw_forecast_mwh": 14.0,
                        "max_cloud_cover_norm": 0.20,
                        "allow_negative_correction": False,
                    },
                },
            }
        }
        artifact = build_operational_residual_learner(
            train,
            config,
            forecast_col="Final_Backtest_Forecast_MWH",
        )
        future = pd.DataFrame(
            {
                "DT": pd.date_range("2026-07-05 17:00", periods=6, freq="24h"),
                "Hour": [17, 17, 17, 17, 17, 17],
                "Month": [7, 7, 7, 7, 7, 7],
                "Final_Forecast_MWH": [100.0, 100.0, 100.0, 100.0, 100.0, 116.0],
                "Raw_Forecast_MWH": [98.0, 98.0, 98.0, 98.0, 98.0, 98.0],
                "MWH_SameHour7DayMean": [95.0, 70.0, 55.0, 70.0, 70.0, 70.0],
                "MWH_Lag24": [75.0, 75.0, 75.0, 85.0, 75.0, 75.0],
                "Temperature_DailyMax": [101.0, 101.0, 101.0, 101.0, 101.0, 101.0],
                "Temperature": [98.0, 98.0, 98.0, 98.0, 98.0, 98.0],
                "CloudCover_Norm": [0.0, 0.0, 0.0, 0.0, 0.60, 0.0],
            }
        )

        out = apply_operational_residual_learner(
            future,
            artifact,
            config,
            forecast_col="Final_Forecast_MWH",
        )

        self.assertEqual(out.loc[0, "Auto_Residual_Correction_MWH"], 0.0)
        self.assertEqual(
            out.loc[0, "Auto_Residual_Source"], "hot_peak_low_forecast_gate_blocked"
        )
        self.assertGreater(out.loc[1, "Auto_Residual_Correction_MWH"], 0.0)
        self.assertEqual(out.loc[1, "Auto_Residual_Source"], "global+hot_peak")
        self.assertEqual(out.loc[2, "Auto_Residual_Correction_MWH"], 0.0)
        self.assertEqual(
            out.loc[2, "Auto_Residual_Source"], "hot_peak_low_forecast_gate_blocked"
        )
        self.assertEqual(out.loc[3, "Auto_Residual_Correction_MWH"], 0.0)
        self.assertEqual(
            out.loc[3, "Auto_Residual_Source"], "hot_peak_low_forecast_gate_blocked"
        )
        self.assertEqual(out.loc[4, "Auto_Residual_Correction_MWH"], 0.0)
        self.assertEqual(
            out.loc[4, "Auto_Residual_Source"], "hot_peak_low_forecast_gate_blocked"
        )
        self.assertEqual(out.loc[5, "Auto_Residual_Correction_MWH"], 0.0)
        self.assertEqual(
            out.loc[5, "Auto_Residual_Source"], "hot_peak_low_forecast_gate_blocked"
        )

    def test_operational_residual_learner_hot_peak_gate_uses_live_ramp_evidence(self):
        dt = pd.date_range("2026-07-01", periods=96, freq="h")
        train = pd.DataFrame(
            {
                "DT": dt,
                "Hour": dt.hour,
                "Month": dt.month,
                "Actual_MWH": 108.0,
                "Final_Backtest_Forecast_MWH": 100.0,
                "Final_Residual_MWH": 8.0,
                "Raw_Forecast_MWH": 98.0,
                "MWH_SameHour7DayMean": 70.0,
                "MWH_Lag24": 95.0,
                "XGB_Pred_MWH": 99.0,
                "LGB_Pred_MWH": 100.0,
                "CatBoost_Pred_MWH": 97.0,
                "Prophet_Pred_MWH": 110.0,
                "Temperature_DailyMax": 101.0,
                "Temperature": 98.0,
                "CloudCover_Norm": 0.0,
            }
        )
        config = {
            "operational_residual_learner": {
                "enabled": True,
                "shadow_mode": True,
                "min_rows": 24,
                "min_samples_leaf": 2,
                "max_iter": 20,
                "blend": 0.5,
                "cap_mwh": 6.0,
                "total_cap_mwh": 6.0,
                "hot_peak": {
                    "enabled": True,
                    "min_rows": 2,
                    "min_samples_leaf": 2,
                    "hours": [17],
                    "min_maxtemp_f": 90.0,
                    "blend": 0.5,
                    "cap_mwh": 6.0,
                    "positive_gate": {
                        "enabled": True,
                        "min_raw_minus_samehour_7day_mean_mwh": 20.0,
                        "max_raw_minus_samehour_7day_mean_mwh": 35.0,
                        "min_raw_minus_samehour_yesterday_mwh": 20.0,
                        "max_final_minus_raw_forecast_mwh": 14.0,
                        "max_cloud_cover_norm": 0.20,
                        "allow_negative_correction": False,
                        "live_ramp_gate": {
                            "enabled": True,
                            "source": "global+hot_peak_live_ramp_gate",
                            "min_forecast_day": 1,
                            "max_forecast_day": 1,
                            "hours": [17],
                            "min_lag1_mwh": 110.0,
                            "min_lag1_minus_base_forecast_mwh": 10.0,
                            "min_lag1_minus_samehour_yesterday_mwh": 20.0,
                            "min_lag1_minus_samehour_7day_mean_mwh": 35.0,
                            "min_live_ramp_1hr_mwh": 5.0,
                            "max_cloud_cover_norm": 0.20,
                        },
                    },
                },
            }
        }
        artifact = build_operational_residual_learner(
            train,
            config,
            forecast_col="Final_Backtest_Forecast_MWH",
        )
        future = pd.DataFrame(
            {
                "DT": pd.date_range("2026-07-05 17:00", periods=2, freq="24h"),
                "Hour": [17, 17],
                "Month": [7, 7],
                "Forecast_Day": [1, 1],
                "Final_Forecast_MWH": [100.0, 100.0],
                "Raw_Forecast_MWH": [98.0, 98.0],
                "MWH_Lag1": [116.0, 116.0],
                "MWH_Lag2": [108.0, 115.0],
                "MWH_SameHour7DayMean": [70.0, 70.0],
                "MWH_Lag24": [95.0, 95.0],
                "Temperature_DailyMax": [101.0, 101.0],
                "Temperature": [98.0, 98.0],
                "CloudCover_Norm": [0.0, 0.0],
            }
        )

        out = apply_operational_residual_learner(
            future,
            artifact,
            config,
            forecast_col="Final_Forecast_MWH",
        )

        self.assertGreater(out.loc[0, "Auto_Residual_Correction_MWH"], 0.0)
        self.assertEqual(
            out.loc[0, "Auto_Residual_Source"], "global+hot_peak_live_ramp_gate"
        )
        self.assertEqual(out.loc[1, "Auto_Residual_Correction_MWH"], 0.0)
        self.assertEqual(
            out.loc[1, "Auto_Residual_Source"], "hot_peak_low_forecast_gate_blocked"
        )

    def test_operational_residual_learner_structural_hot_peak_shadow_skips_raw_yesterday_gate(
        self,
    ):
        dt = pd.date_range("2026-07-01", periods=96, freq="h")
        train = pd.DataFrame(
            {
                "DT": dt,
                "Hour": dt.hour,
                "Month": dt.month,
                "Actual_MWH": 110.0,
                "Final_Backtest_Forecast_MWH": 100.0,
                "Final_Residual_MWH": 10.0,
                "Raw_Forecast_MWH": 99.0,
                "MWH_SameHour7DayMean": 96.0,
                "MWH_Lag24": 102.0,
                "XGB_Pred_MWH": 99.0,
                "LGB_Pred_MWH": 100.0,
                "CatBoost_Pred_MWH": 98.0,
                "Prophet_Pred_MWH": 108.0,
                "Temperature_DailyMax": 96.0,
                "Temperature": 93.0,
                "CloudCover_Norm": 0.0,
                "DailyMaxTemp_Ramp_1Day": 4.0,
                "OvernightHeatStress": 8.0,
                "HeatPersistenceStress90": 12.0,
            }
        )
        config = {
            "operational_residual_learner": {
                "enabled": True,
                "shadow_mode": False,
                "production_scope": "hot_peak_only",
                "min_rows": 24,
                "min_samples_leaf": 2,
                "max_iter": 20,
                "blend": 0.5,
                "cap_mwh": 6.0,
                "total_cap_mwh": 6.0,
                "hot_peak": {
                    "enabled": True,
                    "min_rows": 2,
                    "min_samples_leaf": 2,
                    "hours": [17],
                    "min_maxtemp_f": 90.0,
                    "blend": 0.5,
                    "cap_mwh": 6.0,
                    "structural_shadow": {
                        "enabled": True,
                        "source": "global+hot_peak_structural_residual",
                        "min_rows": 2,
                        "min_samples_leaf": 2,
                        "hours": [17],
                        "min_maxtemp_f": 90.0,
                        "min_forecast_day": 1,
                        "max_forecast_day": 16,
                        "blend": 0.5,
                        "cap_mwh": 5.0,
                        "total_cap_mwh": 5.0,
                        "min_positive_correction_mwh": 1.0,
                        "allow_negative_correction": False,
                    },
                    "positive_gate": {
                        "enabled": True,
                        "min_raw_minus_samehour_7day_mean_mwh": 20.0,
                        "min_raw_minus_samehour_yesterday_mwh": 20.0,
                        "allow_negative_correction": False,
                    },
                },
            }
        }
        artifact = build_operational_residual_learner(
            train,
            config,
            forecast_col="Final_Backtest_Forecast_MWH",
        )
        future = pd.DataFrame(
            {
                "DT": [pd.Timestamp("2026-07-10 17:00")],
                "Hour": [17],
                "Month": [7],
                "Forecast_Day": [8],
                "Final_Forecast_MWH": [100.0],
                "Raw_Forecast_MWH": [99.0],
                "MWH_SameHour7DayMean": [96.0],
                "MWH_Lag24": [102.0],
                "Temperature_DailyMax": [96.0],
                "Temperature": [93.0],
                "CloudCover_Norm": [0.0],
                "DailyMaxTemp_Ramp_1Day": [4.0],
                "OvernightHeatStress": [8.0],
                "HeatPersistenceStress90": [12.0],
            }
        )

        out = apply_operational_residual_learner(
            future,
            artifact,
            config,
            forecast_col="Final_Forecast_MWH",
        )

        self.assertEqual(out.loc[0, "Auto_Residual_Correction_MWH"], 0.0)
        self.assertEqual(
            out.loc[0, "Auto_Residual_Source"], "hot_peak_low_forecast_gate_blocked"
        )
        self.assertGreater(
            out.loc[0, "Auto_Residual_Structural_HotPeak_Correction_MWH"], 0.0
        )
        self.assertLessEqual(
            out.loc[0, "Auto_Residual_Structural_HotPeak_Correction_MWH"], 5.0
        )
        self.assertEqual(
            out.loc[0, "Auto_Residual_Structural_HotPeak_Source"],
            "global+hot_peak_structural_residual",
        )
        self.assertEqual(out.loc[0, "Final_Forecast_MWH"], 100.0)

    def test_operational_residual_learner_broad_hot_peak_shadow_uses_looser_gate_only_in_shadow(
        self,
    ):
        dt = pd.date_range("2026-07-01", periods=96, freq="h")
        train = pd.DataFrame(
            {
                "DT": dt,
                "Hour": dt.hour,
                "Month": dt.month,
                "Actual_MWH": 110.0,
                "Final_Backtest_Forecast_MWH": 100.0,
                "Final_Residual_MWH": 10.0,
                "Raw_Forecast_MWH": 99.0,
                "MWH_SameHour7DayMean": 96.0,
                "MWH_Lag24": 102.0,
                "XGB_Pred_MWH": 99.0,
                "LGB_Pred_MWH": 100.0,
                "CatBoost_Pred_MWH": 98.0,
                "Prophet_Pred_MWH": 108.0,
                "Temperature_DailyMax": 96.0,
                "Temperature": 93.0,
                "CloudCover_Norm": 0.0,
            }
        )
        config = {
            "operational_residual_learner": {
                "enabled": True,
                "shadow_mode": False,
                "production_scope": "hot_peak_only",
                "min_rows": 24,
                "min_samples_leaf": 2,
                "max_iter": 20,
                "blend": 0.5,
                "cap_mwh": 6.0,
                "total_cap_mwh": 6.0,
                "hot_peak": {
                    "enabled": True,
                    "min_rows": 2,
                    "min_samples_leaf": 2,
                    "hours": [17],
                    "min_maxtemp_f": 90.0,
                    "blend": 0.5,
                    "cap_mwh": 6.0,
                    "broad_shadow": {
                        "enabled": True,
                        "source": "global+hot_peak_broad_shadow",
                        "hours": [17],
                        "min_maxtemp_f": 90.0,
                        "blend": 0.5,
                        "cap_mwh": 6.0,
                        "total_cap_mwh": 6.0,
                        "min_abs_correction_mwh": 0.1,
                        "allow_negative_correction": False,
                        "positive_gate": {
                            "enabled": True,
                            "min_raw_minus_samehour_7day_mean_mwh": 0.0,
                            "min_raw_minus_samehour_yesterday_mwh": -10.0,
                            "max_final_minus_raw_forecast_mwh": 14.0,
                            "allow_negative_correction": False,
                            "blocked_source": "hot_peak_broad_low_forecast_gate_blocked",
                        },
                    },
                    "positive_gate": {
                        "enabled": True,
                        "min_raw_minus_samehour_7day_mean_mwh": 20.0,
                        "min_raw_minus_samehour_yesterday_mwh": 20.0,
                        "allow_negative_correction": False,
                        "blocked_source": "hot_peak_low_forecast_gate_blocked",
                    },
                },
            }
        }
        artifact = build_operational_residual_learner(
            train,
            config,
            forecast_col="Final_Backtest_Forecast_MWH",
        )
        future = pd.DataFrame(
            {
                "DT": [pd.Timestamp("2026-07-10 17:00")],
                "Hour": [17],
                "Month": [7],
                "Forecast_Day": [8],
                "Final_Forecast_MWH": [100.0],
                "Raw_Forecast_MWH": [99.0],
                "MWH_SameHour7DayMean": [96.0],
                "MWH_Lag24": [102.0],
                "Temperature_DailyMax": [96.0],
                "Temperature": [93.0],
                "CloudCover_Norm": [0.0],
            }
        )

        out = apply_operational_residual_learner(
            future,
            artifact,
            config,
            forecast_col="Final_Forecast_MWH",
        )

        self.assertEqual(out.loc[0, "Auto_Residual_Correction_MWH"], 0.0)
        self.assertEqual(
            out.loc[0, "Auto_Residual_Source"], "hot_peak_low_forecast_gate_blocked"
        )
        self.assertGreater(
            out.loc[0, "Auto_Residual_Broad_HotPeak_Shadow_Correction_MWH"], 0.0
        )
        self.assertEqual(
            out.loc[0, "Auto_Residual_Broad_HotPeak_Shadow_Source"],
            "global+hot_peak_broad_shadow",
        )
        self.assertGreater(
            out.loc[0, "Auto_Residual_Broad_HotPeak_Shadow_Adjusted_Forecast_MWH"],
            100.0,
        )
        self.assertEqual(out.loc[0, "Final_Forecast_MWH"], 100.0)

    def test_operational_residual_learner_blocks_hot_peak_lift_when_cooling_underway(
        self,
    ):
        dt = pd.date_range("2026-07-01", periods=96, freq="h")
        train = pd.DataFrame(
            {
                "DT": dt,
                "Hour": dt.hour,
                "Month": dt.month,
                "Actual_MWH": 108.0,
                "Final_Backtest_Forecast_MWH": 100.0,
                "Final_Residual_MWH": 8.0,
                "Raw_Forecast_MWH": 98.0,
                "MWH_SameHour7DayMean": 70.0,
                "MWH_Lag24": 75.0,
                "XGB_Pred_MWH": 99.0,
                "LGB_Pred_MWH": 100.0,
                "CatBoost_Pred_MWH": 97.0,
                "Prophet_Pred_MWH": 110.0,
                "Temperature_DailyMax": 101.0,
                "Temperature": 98.0,
                "CloudCover_Norm": 0.0,
            }
        )
        config = {
            "operational_residual_learner": {
                "enabled": True,
                "shadow_mode": True,
                "min_rows": 24,
                "min_samples_leaf": 2,
                "max_iter": 20,
                "blend": 0.5,
                "cap_mwh": 6.0,
                "total_cap_mwh": 6.0,
                "hot_peak": {
                    "enabled": True,
                    "min_rows": 2,
                    "min_samples_leaf": 2,
                    "hours": [17],
                    "min_maxtemp_f": 90.0,
                    "blend": 0.5,
                    "cap_mwh": 6.0,
                    "positive_gate": {
                        "enabled": True,
                        "min_raw_minus_samehour_7day_mean_mwh": 20.0,
                        "max_raw_minus_samehour_7day_mean_mwh": 35.0,
                        "min_raw_minus_samehour_yesterday_mwh": 20.0,
                        "max_final_minus_raw_forecast_mwh": 14.0,
                        "max_cloud_cover_norm": 0.20,
                        "allow_negative_correction": False,
                        "cooling_underway_guard": {
                            "enabled": True,
                            "mode": "all",
                            "min_drop_from_dailymax_f": 5.0,
                            "min_forecast_drop_next3hr_f": 6.0,
                            "cap_positive_correction_mwh": 0.0,
                            "blocked_source": "hot_peak_cooling_underway_guard_blocked",
                        },
                    },
                },
            }
        }
        artifact = build_operational_residual_learner(
            train,
            config,
            forecast_col="Final_Backtest_Forecast_MWH",
        )
        future = pd.DataFrame(
            {
                "DT": pd.date_range("2026-07-05 17:00", periods=3, freq="24h"),
                "Hour": [17, 17, 17],
                "Month": [7, 7, 7],
                "Final_Forecast_MWH": [100.0, 100.0, 100.0],
                "Raw_Forecast_MWH": [98.0, 98.0, 98.0],
                "MWH_SameHour7DayMean": [70.0, 70.0, 70.0],
                "MWH_Lag24": [75.0, 75.0, 75.0],
                "Temperature_DailyMax": [101.0, 101.0, 101.0],
                "Temperature": [95.0, 97.0, 95.0],
                "Temperature_Drop_From_DailyMax_F": [6.0, 4.0, 6.0],
                "TempDrop_Next3Hr_F": [7.0, 7.0, 3.0],
                "CloudCover_Norm": [0.0, 0.0, 0.0],
            }
        )

        out = apply_operational_residual_learner(
            future,
            artifact,
            config,
            forecast_col="Final_Forecast_MWH",
        )

        self.assertEqual(out.loc[0, "Auto_Residual_Correction_MWH"], 0.0)
        self.assertEqual(
            out.loc[0, "Auto_Residual_Source"],
            "hot_peak_cooling_underway_guard_blocked",
        )
        self.assertGreater(out.loc[1, "Auto_Residual_Correction_MWH"], 0.0)
        self.assertEqual(out.loc[1, "Auto_Residual_Source"], "global+hot_peak")
        self.assertGreater(out.loc[2, "Auto_Residual_Correction_MWH"], 0.0)
        self.assertEqual(out.loc[2, "Auto_Residual_Source"], "global+hot_peak")

    def test_operational_residual_learner_hot_peak_only_scope_updates_only_gated_hot_rows(
        self,
    ):
        dt = pd.date_range("2026-07-01", periods=96, freq="h")
        train = pd.DataFrame(
            {
                "DT": dt,
                "Hour": dt.hour,
                "Month": dt.month,
                "Actual_MWH": 108.0,
                "Final_Backtest_Forecast_MWH": 100.0,
                "Final_Residual_MWH": 8.0,
                "Raw_Forecast_MWH": 98.0,
                "MWH_SameHour7DayMean": 70.0,
                "MWH_Lag24": 75.0,
                "XGB_Pred_MWH": 99.0,
                "LGB_Pred_MWH": 100.0,
                "CatBoost_Pred_MWH": 97.0,
                "Prophet_Pred_MWH": 110.0,
                "Temperature_DailyMax": 101.0,
                "Temperature": 98.0,
                "CloudCover_Norm": 0.0,
            }
        )
        config = {
            "operational_residual_learner": {
                "enabled": True,
                "shadow_mode": False,
                "production_scope": "hot_peak_only",
                "min_rows": 24,
                "min_samples_leaf": 2,
                "max_iter": 20,
                "blend": 0.5,
                "cap_mwh": 6.0,
                "total_cap_mwh": 6.0,
                "hot_peak": {
                    "enabled": True,
                    "min_rows": 2,
                    "min_samples_leaf": 2,
                    "hours": [17],
                    "min_maxtemp_f": 90.0,
                    "blend": 0.5,
                    "cap_mwh": 6.0,
                    "positive_gate": {
                        "enabled": True,
                        "min_raw_minus_samehour_7day_mean_mwh": 20.0,
                        "max_raw_minus_samehour_7day_mean_mwh": 35.0,
                        "min_raw_minus_samehour_yesterday_mwh": 20.0,
                        "max_final_minus_raw_forecast_mwh": 14.0,
                        "max_cloud_cover_norm": 0.20,
                        "allow_negative_correction": False,
                    },
                },
            }
        }
        artifact = build_operational_residual_learner(
            train,
            config,
            forecast_col="Final_Backtest_Forecast_MWH",
        )
        future = pd.DataFrame(
            {
                "DT": pd.to_datetime(["2026-07-05 10:00", "2026-07-05 17:00"]),
                "Hour": [10, 17],
                "Month": [7, 7],
                "Final_Forecast_MWH": [100.0, 100.0],
                "Stage_Selected_Forecast_MWH": [100.0, 100.0],
                "Raw_Forecast_MWH": [98.0, 98.0],
                "MWH_SameHour7DayMean": [70.0, 70.0],
                "MWH_Lag24": [75.0, 75.0],
                "Temperature_DailyMax": [101.0, 101.0],
                "Temperature": [98.0, 98.0],
                "CloudCover_Norm": [0.0, 0.0],
            }
        )

        out = apply_operational_residual_learner(
            future,
            artifact,
            config,
            forecast_col="Final_Forecast_MWH",
            also_update_cols=("Stage_Selected_Forecast_MWH",),
        )

        self.assertEqual(out.loc[0, "Auto_Residual_Production_Scope"], "hot_peak_only")
        self.assertEqual(out.loc[1, "Auto_Residual_Production_Scope"], "hot_peak_only")
        self.assertTrue((out["Auto_Residual_Shadow_Mode"] == 0).all())
        self.assertGreater(out.loc[0, "Auto_Residual_Full_Shadow_Correction_MWH"], 0.0)
        self.assertEqual(out.loc[0, "Auto_Residual_Correction_MWH"], 0.0)
        self.assertEqual(out.loc[0, "Auto_Residual_Source"], "global_shadow_only")
        self.assertEqual(out.loc[0, "Final_Forecast_MWH"], 100.0)
        self.assertEqual(out.loc[0, "Stage_Selected_Forecast_MWH"], 100.0)
        self.assertGreater(out.loc[1, "Auto_Residual_Correction_MWH"], 0.0)
        self.assertEqual(out.loc[1, "Auto_Residual_Source"], "global+hot_peak")
        self.assertGreater(out.loc[1, "Final_Forecast_MWH"], 100.0)
        self.assertEqual(
            out.loc[1, "Stage_Selected_Forecast_MWH"], out.loc[1, "Final_Forecast_MWH"]
        )

    def test_operational_residual_learner_capped_full_scope_promotes_small_global_signal(
        self,
    ):
        dt = pd.date_range("2026-07-01", periods=72, freq="h")
        train = pd.DataFrame(
            {
                "DT": dt,
                "Hour": dt.hour,
                "Month": dt.month,
                "Actual_MWH": 108.0,
                "Final_Backtest_Forecast_MWH": 100.0,
                "Final_Residual_MWH": 8.0,
                "Raw_Forecast_MWH": 98.0,
                "XGB_Pred_MWH": 99.0,
                "LGB_Pred_MWH": 100.0,
                "CatBoost_Pred_MWH": 97.0,
                "Prophet_Pred_MWH": 110.0,
                "Temperature_DailyMax": 82.0,
                "Temperature": 76.0,
                "CloudCover_Norm": 0.0,
            }
        )
        config = {
            "operational_residual_learner": {
                "enabled": True,
                "shadow_mode": False,
                "production_scope": "capped_full_shadow",
                "capped_full_shadow_cap_mwh": 1.0,
                "min_rows": 24,
                "min_samples_leaf": 2,
                "max_iter": 20,
                "blend": 1.0,
                "cap_mwh": 6.0,
                "total_cap_mwh": 6.0,
                "hot_peak": {"enabled": False},
            }
        }
        artifact = build_operational_residual_learner(
            train,
            config,
            forecast_col="Final_Backtest_Forecast_MWH",
        )
        future = pd.DataFrame(
            {
                "DT": pd.to_datetime(["2026-07-05 10:00", "2026-07-05 11:00"]),
                "Hour": [10, 11],
                "Month": [7, 7],
                "Final_Forecast_MWH": [100.0, 100.0],
                "Stage_Selected_Forecast_MWH": [100.0, 100.0],
                "Raw_Forecast_MWH": [98.0, 98.0],
                "Temperature_DailyMax": [82.0, 82.0],
                "Temperature": [76.0, 76.0],
                "CloudCover_Norm": [0.0, 0.0],
            }
        )

        out = apply_operational_residual_learner(
            future,
            artifact,
            config,
            forecast_col="Final_Forecast_MWH",
            also_update_cols=("Stage_Selected_Forecast_MWH",),
        )

        self.assertEqual(
            out.loc[0, "Auto_Residual_Production_Scope"], "capped_full_shadow"
        )
        self.assertGreater(out.loc[0, "Auto_Residual_Full_Shadow_Correction_MWH"], 1.0)
        self.assertAlmostEqual(out.loc[0, "Auto_Residual_Correction_MWH"], 1.0)
        self.assertEqual(
            out.loc[0, "Auto_Residual_Source"], "global+capped_full_shadow_cap"
        )
        self.assertAlmostEqual(out.loc[0, "Final_Forecast_MWH"], 101.0)
        self.assertEqual(
            out.loc[0, "Stage_Selected_Forecast_MWH"], out.loc[0, "Final_Forecast_MWH"]
        )

    def test_operational_residual_learner_walk_forward_marks_insufficient_history(self):
        dt = pd.date_range("2026-07-01", periods=96, freq="h")
        df = pd.DataFrame(
            {
                "DT": dt,
                "Hour": dt.hour,
                "Month": dt.month,
                "Actual_MWH": 105.0,
                "Final_Backtest_Forecast_MWH": 100.0,
                "Final_Residual_MWH": 5.0,
                "Raw_Forecast_MWH": 98.0,
                "Prophet_Pred_MWH": 108.0,
                "Temperature_DailyMax": 95.0,
                "CloudCover_Norm": 0.0,
            }
        )
        config = {
            "operational_residual_learner": {
                "enabled": True,
                "shadow_mode": True,
                "min_rows": 24,
                "backtest_min_train_rows": 48,
                "min_samples_leaf": 2,
                "max_iter": 20,
                "blend": 0.5,
                "cap_mwh": 6.0,
                "total_cap_mwh": 6.0,
                "hot_peak": {"enabled": False},
            }
        }

        out = simulate_operational_residual_learner_backtest(
            df,
            config,
            forecast_col="Final_Backtest_Forecast_MWH",
        )

        self.assertEqual(
            out.loc[0, "Auto_Residual_Source"], "insufficient_walk_forward_history"
        )
        self.assertTrue((out["Final_Backtest_Forecast_MWH"] == 100.0).all())
        self.assertTrue((out["Auto_Residual_Adjusted_Forecast_MWH"] >= 100.0).all())

    def test_hot_peak_shadow_candidate_scorecard_flags_only_slices_that_beat_final(
        self,
    ):
        df = pd.DataFrame(
            {
                "DT": pd.to_datetime(["2026-07-10 17:00", "2026-08-10 17:00"]),
                "Actual_MWH": [110.0, 100.0],
                "Final_Backtest_Forecast_MWH": [100.0, 100.0],
                "Auto_Residual_Broad_HotPeak_Shadow_Adjusted_Forecast_MWH": [
                    108.0,
                    110.0,
                ],
                "Replay_Horizon_Bucket": ["Days8to16", "Days8to16"],
                "Month": [7, 8],
                "Hour": [17, 17],
                "Temperature_DailyMax": [96.0, 96.0],
                "CloudCover_Norm": [0.10, 0.10],
            }
        )

        scorecard = build_hot_peak_shadow_candidate_scorecard(df, min_count=1)

        july = scorecard[
            scorecard["Stage"].eq("auto_residual_broad_hot_peak_shadow")
            & scorecard["Month"].eq(7)
        ].iloc[0]
        august = scorecard[
            scorecard["Stage"].eq("auto_residual_broad_hot_peak_shadow")
            & scorecard["Month"].eq(8)
        ].iloc[0]
        self.assertTrue(bool(july["Promote_Slice_Candidate"]))
        self.assertGreater(july["Candidate_MAE_Improvement_vs_Final_MWH"], 0.0)
        self.assertFalse(bool(august["Promote_Slice_Candidate"]))
        self.assertLess(august["Candidate_MAE_Improvement_vs_Final_MWH"], 0.0)

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
        self.assertGreater(
            out.loc[0, "Final_Forecast_MWH"], df.loc[0, "Final_Forecast_MWH"]
        )

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
        self.assertEqual(
            out.loc[0, "Weather_Robustness_Hedge_Source"],
            "weather_uncertainty_ramp_hedge",
        )

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
                },
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
        self.assertTrue(
            out["Band_Method"].astype(str).str.contains("hot_bucket_floor").all()
        )
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

        out = apply_focused_scorecard_guard(
            df, config, forecast_col="Final_Forecast_MWH"
        )

        self.assertEqual(out.loc[0, "Focused_Scorecard_Guard_MWH"], 10.0)
        self.assertEqual(out.loc[0, "Final_Forecast_MWH"], 290.0)
        self.assertEqual(out.loc[0, "Stage_Selected_Forecast_MWH"], 290.0)
        self.assertEqual(out.loc[0, "Pre_Focused_Guard_Forecast_MWH"], 280.0)
        self.assertEqual(out.loc[0, "Post_Focused_Guard_Forecast_MWH"], 290.0)
        self.assertEqual(out.loc[0, "Focused_Guard_Applied_Flag"], 1)

    def test_focused_guard_does_not_infer_horizon_for_actual_backtest_rows(self):
        df = pd.DataFrame(
            {
                "DT": [pd.Timestamp("2026-06-26 12:00")],
                "Actual_MWH": [286.0],
                "Final_Backtest_Forecast_MWH": [280.0],
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

        out = apply_focused_scorecard_guard(
            df, config, forecast_col="Final_Backtest_Forecast_MWH"
        )

        self.assertEqual(out.loc[0, "Focused_Scorecard_Guard_MWH"], 0.0)
        self.assertEqual(out.loc[0, "Final_Backtest_Forecast_MWH"], 280.0)
        self.assertEqual(out.loc[0, "Stage_Selected_Forecast_MWH"], 280.0)
        self.assertEqual(out.loc[0, "Focused_Guard_Applied_Flag"], 0)

    def test_focused_guard_allows_explicit_no_horizon_backtest_rules(self):
        df = pd.DataFrame(
            {
                "DT": [pd.Timestamp("2026-06-16 14:00")],
                "Actual_MWH": [210.0],
                "Final_Backtest_Forecast_MWH": [220.0],
                "Stage_Selected_Forecast_MWH": [220.0],
                "Temperature_DailyMax": [91.0],
                "CloudCover_Norm": [0.10],
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
                                "name": "safe_no_horizon_shape_rule",
                                "adjustment_mwh": -3.0,
                                "allow_without_forecast_day": True,
                                "months": [6],
                                "hours": [14],
                                "min_maxtemp_f": 85.0,
                                "max_maxtemp_f": 93.0,
                                "max_cloud_cover_norm": 0.20,
                            },
                            {
                                "name": "blocked_horizon_rule",
                                "adjustment_mwh": 10.0,
                                "allow_without_forecast_day": True,
                                "months": [6],
                                "hours": [14],
                                "min_forecast_day": 1,
                                "max_forecast_day": 16,
                                "min_maxtemp_f": 85.0,
                            },
                        ],
                    }
                }
            }
        }

        out = apply_focused_scorecard_guard(
            df, config, forecast_col="Final_Backtest_Forecast_MWH"
        )

        self.assertEqual(out.loc[0, "Focused_Scorecard_Guard_MWH"], -3.0)
        self.assertEqual(out.loc[0, "Final_Backtest_Forecast_MWH"], 217.0)
        self.assertEqual(out.loc[0, "Stage_Selected_Forecast_MWH"], 217.0)
        self.assertEqual(
            out.loc[0, "Focused_Scorecard_Guard_Source"], "safe_no_horizon_shape_rule"
        )
        self.assertEqual(out.loc[0, "Focused_Guard_Applied_Flag"], 1)

    def test_focused_guard_explicit_forecast_day_gate_ignores_no_horizon_backtest_rows(
        self,
    ):
        config = {
            "calibration": {
                "stage_selector": {
                    "focused_scorecard_guard": {
                        "enabled": True,
                        "total_cap_mwh": 18.0,
                        "rules": [
                            {
                                "name": "near_term_only_when_horizon_is_explicit",
                                "adjustment_mwh": 8.0,
                                "allow_without_forecast_day": True,
                                "months": [7],
                                "hours": [14],
                                "min_explicit_forecast_day": 1,
                                "max_explicit_forecast_day": 3,
                                "min_maxtemp_f": 103.0,
                            }
                        ],
                    }
                }
            }
        }
        no_horizon = pd.DataFrame(
            {
                "DT": [pd.Timestamp("2026-07-14 14:00")],
                "Actual_MWH": [290.0],
                "Final_Backtest_Forecast_MWH": [280.0],
                "Stage_Selected_Forecast_MWH": [280.0],
                "Temperature_DailyMax": [104.0],
                "IsHoliday": [0],
            }
        )
        explicit_horizon = pd.DataFrame(
            {
                "DT": [
                    pd.Timestamp("2026-07-14 14:00"),
                    pd.Timestamp("2026-07-14 14:00"),
                ],
                "Forecast_Day": [2, 8],
                "Final_Forecast_MWH": [280.0, 280.0],
                "Stage_Selected_Forecast_MWH": [280.0, 280.0],
                "Temperature_DailyMax": [104.0, 104.0],
                "IsHoliday": [0, 0],
            }
        )

        no_horizon_out = apply_focused_scorecard_guard(
            no_horizon,
            config,
            forecast_col="Final_Backtest_Forecast_MWH",
        )
        explicit_out = apply_focused_scorecard_guard(
            explicit_horizon,
            config,
            forecast_col="Final_Forecast_MWH",
        )

        self.assertEqual(no_horizon_out["Focused_Scorecard_Guard_MWH"].tolist(), [8.0])
        self.assertEqual(
            no_horizon_out["Final_Backtest_Forecast_MWH"].tolist(), [288.0]
        )
        self.assertEqual(
            explicit_out["Focused_Scorecard_Guard_MWH"].tolist(), [8.0, 0.0]
        )
        self.assertEqual(explicit_out["Final_Forecast_MWH"].tolist(), [288.0, 280.0])

    def test_focused_guard_can_gate_on_prior_focused_stack(self):
        df = pd.DataFrame(
            {
                "DT": [
                    pd.Timestamp("2026-07-18 16:00"),
                    pd.Timestamp("2026-07-18 16:00"),
                ],
                "Forecast_Day": [12, 12],
                "Final_Forecast_MWH": [240.0, 190.0],
                "Stage_Selected_Forecast_MWH": [240.0, 190.0],
                "Temperature_DailyMax": [95.0, 95.0],
                "IsHoliday": [0, 0],
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
                                "name": "prior_recovery",
                                "adjustment_mwh": 8.0,
                                "months": [7],
                                "hours": [16],
                                "min_forecast_day": 8,
                                "max_forecast_day": 16,
                                "min_maxtemp_f": 95.0,
                                "min_forecast_mwh": 220.0,
                            },
                            {
                                "name": "stack_limited_ramp",
                                "adjustment_mwh": 2.0,
                                "months": [7],
                                "hours": [16],
                                "min_forecast_day": 8,
                                "max_forecast_day": 16,
                                "min_maxtemp_f": 95.0,
                                "max_prior_total_adjustment_mwh": 4.0,
                            },
                        ],
                    }
                }
            }
        }

        out = apply_focused_scorecard_guard(
            df, config, forecast_col="Final_Forecast_MWH"
        )

        self.assertEqual(out["Focused_Scorecard_Guard_MWH"].tolist(), [8.0, 2.0])
        self.assertEqual(out["Final_Forecast_MWH"].tolist(), [248.0, 192.0])
        self.assertEqual(
            out["Focused_Scorecard_Guard_Source"].tolist(),
            ["prior_recovery", "stack_limited_ramp"],
        )

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

        out = apply_focused_scorecard_guard(
            df, config, forecast_col="Final_Forecast_MWH"
        )

        self.assertEqual(
            out["Focused_Scorecard_Guard_MWH"].tolist(), [0.0, 10.0, 8.0, 5.5]
        )
        self.assertEqual(
            out["Final_Forecast_MWH"].tolist(), [226.0, 262.0, 294.0, 317.5]
        )

    def test_focused_guard_rule_can_extend_total_cap_for_narrow_pattern(self):
        df = pd.DataFrame(
            {
                "DT": [
                    pd.Timestamp("2026-08-26 14:00"),
                    pd.Timestamp("2026-08-26 15:00"),
                ],
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

        out = apply_focused_scorecard_guard(
            df, config, forecast_col="Final_Forecast_MWH"
        )

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

        out = apply_focused_scorecard_guard(
            df, config, forecast_col="Final_Forecast_MWH"
        )

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

        out = apply_focused_scorecard_guard(
            df, config, forecast_col="Final_Forecast_MWH"
        )

        self.assertEqual(out["Focused_Scorecard_Guard_MWH"].tolist(), [-5.0, 0.0, 0.0])
        self.assertEqual(out["Final_Forecast_MWH"].tolist(), [215.0, 220.0, 220.0])
        self.assertEqual(
            out["Stage_Selected_Forecast_MWH"].tolist(), [215.0, 220.0, 220.0]
        )
        self.assertEqual(
            out["Raw_Minus_SameHour7DayMean_MWH"].tolist(), [20.0, 10.0, 20.0]
        )
        self.assertEqual(
            out["Raw_Minus_SameHourYesterday_MWH"].tolist(), [5.0, 5.0, -5.0]
        )

    def test_focused_guard_can_gate_on_max_raw_minus_same_hour_load_state(self):
        df = pd.DataFrame(
            {
                "DT": [
                    pd.Timestamp("2026-07-06 16:00"),
                    pd.Timestamp("2026-07-29 16:00"),
                ],
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

        out = apply_focused_scorecard_guard(
            df, config, forecast_col="Final_Forecast_MWH"
        )

        self.assertEqual(out["Focused_Scorecard_Guard_MWH"].tolist(), [-10.0, 0.0])
        self.assertEqual(out["Raw_Minus_SameHour7DayMean_MWH"].tolist(), [5.0, 28.0])
        self.assertEqual(out["Final_Forecast_MWH"].tolist(), [240.0, 250.0])

    def test_focused_guard_rule_audit_scores_rule_only_delta(self):
        df = pd.DataFrame(
            {
                "DT": [
                    pd.Timestamp("2026-07-29 17:00"),
                    pd.Timestamp("2026-07-29 18:00"),
                ],
                "Actual_MWH": [110.0, 90.0],
                "Final_Backtest_Forecast_MWH": [105.0, 105.0],
                "Pre_Focused_Guard_Forecast_MWH": [100.0, 100.0],
                "Post_Focused_Guard_Forecast_MWH": [105.0, 105.0],
                "Focused_Scorecard_Guard_MWH": [5.0, 5.0],
                "Focused_Scorecard_Guard_Source": ["test_lift", "test_lift"],
                "Raw_Forecast_MWH": [100.0, 100.0],
                "MWH_SameHour7DayMean": [90.0, 90.0],
                "MWH_Lag24": [95.0, 95.0],
                "Temperature_DailyMax": [100.0, 100.0],
                "CloudCover_Norm": [0.0, 0.0],
            }
        )
        config = {
            "calibration": {
                "stage_selector": {
                    "focused_scorecard_guard": {
                        "enabled": True,
                        "rules": [
                            {
                                "name": "test_lift",
                                "adjustment_mwh": 5.0,
                                "allow_without_forecast_day": True,
                                "months": [7],
                                "hours": [17, 18],
                                "min_maxtemp_f": 95.0,
                            }
                        ],
                    }
                }
            }
        }

        audit = build_focused_scorecard_rule_audit(df, config)

        self.assertEqual(len(audit), 1)
        self.assertEqual(audit.loc[0, "RuleName"], "test_lift")
        self.assertEqual(audit.loc[0, "ScoredRows"], 2)
        self.assertAlmostEqual(audit.loc[0, "Baseline_MAE_MWH"], 10.0)
        self.assertAlmostEqual(audit.loc[0, "RuleOnly_MAE_MWH"], 10.0)
        self.assertAlmostEqual(audit.loc[0, "CurrentStack_MAE_OnRows_MWH"], 10.0)
        self.assertEqual(audit.loc[0, "RuleHealth_Status"], "pass")
        self.assertEqual(audit.loc[0, "RuleHealth_FailReasons"], "")

    def test_focused_shape_residual_learner_shadow_does_not_change_forecast(self):
        dt = pd.date_range("2026-07-01 14:00", periods=48, freq="h")
        train = pd.DataFrame(
            {
                "DT": dt,
                "Actual_MWH": 110.0,
                "Final_Backtest_Forecast_MWH": 100.0,
                "Raw_Forecast_MWH": 98.0,
                "MWH_SameHour7DayMean": 90.0,
                "MWH_Lag24": 92.0,
                "Temperature_DailyMax": 98.0,
                "Temperature": 95.0,
                "CloudCover_Norm": 0.0,
                "IsHoliday": 0,
                "IsWeekend": 0,
            }
        )
        config = {
            "calibration": {
                "stage_selector": {
                    "focused_shape_residual_learner": {
                        "enabled": True,
                        "shadow_mode": True,
                        "min_rows": 12,
                        "min_samples_leaf": 4,
                        "max_iter": 20,
                        "blend": 1.0,
                        "cap_mwh": 12.0,
                        "min_abs_correction_mwh": 0.0,
                        "scope": {
                            "use_focused_guard_rule_union": False,
                            "require_scope_for_application": True,
                            "include_hot_peak": True,
                            "hot_peak_min_maxtemp_f": 90.0,
                            "hot_peak_hours": [14, 15, 16, 17, 18, 19, 20, 21],
                            "include_cloud_solar": False,
                            "include_delta_breeze": False,
                            "include_long_horizon_heat": False,
                        },
                    },
                }
            }
        }
        artifact = build_focused_shape_residual_learner(
            train,
            config,
            forecast_col="Final_Backtest_Forecast_MWH",
        )
        future = pd.DataFrame(
            {
                "DT": [pd.Timestamp("2026-07-03 17:00")],
                "Final_Backtest_Forecast_MWH": [100.0],
                "Raw_Forecast_MWH": [98.0],
                "MWH_SameHour7DayMean": [90.0],
                "MWH_Lag24": [92.0],
                "Temperature_DailyMax": [98.0],
                "Temperature": [95.0],
                "CloudCover_Norm": [0.0],
                "IsHoliday": [0],
                "IsWeekend": [0],
            }
        )

        out = apply_focused_shape_residual_learner(
            future,
            artifact,
            config,
            forecast_col="Final_Backtest_Forecast_MWH",
        )

        self.assertIsNotNone(artifact)
        self.assertEqual(out.loc[0, "Final_Backtest_Forecast_MWH"], 100.0)
        self.assertEqual(out.loc[0, "Focused_Shape_Correction_Applied_Flag"], 1)
        self.assertGreater(out.loc[0, "Focused_Shape_Adjusted_Forecast_MWH"], 100.0)
        self.assertEqual(out.loc[0, "Focused_Shape_Shadow_Mode"], 1)

    def test_focused_shape_residual_learner_promotes_when_shadow_disabled(self):
        dt = pd.date_range("2026-07-01 14:00", periods=48, freq="h")
        train = pd.DataFrame(
            {
                "DT": dt,
                "Actual_MWH": 110.0,
                "Pre_Focused_Guard_Forecast_MWH": 100.0,
                "Final_Backtest_Forecast_MWH": 105.0,
                "Final_Forecast_MWH": 105.0,
                "Stage_Selected_Forecast_MWH": 105.0,
                "Raw_Forecast_MWH": 98.0,
                "MWH_SameHour7DayMean": 90.0,
                "MWH_Lag24": 92.0,
                "Temperature_DailyMax": 98.0,
                "Temperature": 95.0,
                "CloudCover_Norm": 0.0,
                "IsHoliday": 0,
                "IsWeekend": 0,
            }
        )
        config = {
            "calibration": {
                "stage_selector": {
                    "focused_shape_residual_learner": {
                        "enabled": True,
                        "shadow_mode": False,
                        "min_rows": 12,
                        "min_samples_leaf": 4,
                        "max_iter": 20,
                        "blend": 1.0,
                        "cap_mwh": 12.0,
                        "promotion_delta_guard": {
                            "enabled": True,
                            "reference_col": "current_final",
                            "max_abs_delta_vs_reference_mwh": 2.0,
                        },
                        "min_abs_correction_mwh": 0.0,
                        "scope": {
                            "use_focused_guard_rule_union": False,
                            "require_scope_for_application": True,
                            "include_hot_peak": True,
                            "hot_peak_min_maxtemp_f": 90.0,
                            "hot_peak_hours": [14, 15, 16, 17, 18, 19, 20, 21],
                            "include_cloud_solar": False,
                            "include_delta_breeze": False,
                            "include_long_horizon_heat": False,
                        },
                    },
                }
            }
        }
        artifact = build_focused_shape_residual_learner(
            train,
            config,
            forecast_col="Pre_Focused_Guard_Forecast_MWH",
        )
        future = pd.DataFrame(
            {
                "DT": [pd.Timestamp("2026-07-03 17:00")],
                "Actual_MWH": [110.0],
                "Pre_Focused_Guard_Forecast_MWH": [100.0],
                "Final_Backtest_Forecast_MWH": [105.0],
                "Final_Forecast_MWH": [105.0],
                "Stage_Selected_Forecast_MWH": [105.0],
                "Raw_Forecast_MWH": [98.0],
                "MWH_SameHour7DayMean": [90.0],
                "MWH_Lag24": [92.0],
                "Temperature_DailyMax": [98.0],
                "Temperature": [95.0],
                "CloudCover_Norm": [0.0],
                "IsHoliday": [0],
                "IsWeekend": [0],
            }
        )

        out = apply_focused_shape_residual_learner(
            future,
            artifact,
            config,
            forecast_col="Pre_Focused_Guard_Forecast_MWH",
            also_update_cols=(
                "Final_Backtest_Forecast_MWH",
                "Stage_Selected_Forecast_MWH",
            ),
            update_forecast_col=False,
        )

        self.assertIsNotNone(artifact)
        self.assertEqual(out.loc[0, "Pre_Focused_Guard_Forecast_MWH"], 100.0)
        self.assertGreater(out.loc[0, "Final_Backtest_Forecast_MWH"], 105.0)
        self.assertLessEqual(out.loc[0, "Final_Backtest_Forecast_MWH"], 107.0)
        self.assertEqual(
            out.loc[0, "Final_Forecast_MWH"], out.loc[0, "Final_Backtest_Forecast_MWH"]
        )
        self.assertEqual(
            out.loc[0, "Stage_Selected_Forecast_MWH"],
            out.loc[0, "Final_Backtest_Forecast_MWH"],
        )
        self.assertEqual(out.loc[0, "Focused_Shape_Shadow_Mode"], 0)
        self.assertIn("focused_shape_production", out.loc[0, "Focused_Shape_Source"])
        self.assertIn("promotion_delta_guard", out.loc[0, "Focused_Shape_Source"])
        self.assertAlmostEqual(
            out.loc[0, "Final_Residual_MWH"],
            110.0 - out.loc[0, "Final_Backtest_Forecast_MWH"],
        )

    def test_focused_shape_residual_learner_can_apply_additively_to_reference(self):
        dt = pd.date_range("2026-07-01 14:00", periods=48, freq="h")
        train = pd.DataFrame(
            {
                "DT": dt,
                "Actual_MWH": 110.0,
                "Pre_Focused_Guard_Forecast_MWH": 100.0,
                "Final_Backtest_Forecast_MWH": 105.0,
                "Final_Forecast_MWH": 105.0,
                "Stage_Selected_Forecast_MWH": 105.0,
                "Raw_Forecast_MWH": 98.0,
                "MWH_SameHour7DayMean": 90.0,
                "MWH_Lag24": 92.0,
                "Temperature_DailyMax": 98.0,
                "Temperature": 95.0,
                "CloudCover_Norm": 0.0,
                "IsHoliday": 0,
                "IsWeekend": 0,
            }
        )
        config = {
            "calibration": {
                "stage_selector": {
                    "focused_shape_residual_learner": {
                        "enabled": True,
                        "shadow_mode": False,
                        "apply_correction_to_reference_forecast": True,
                        "min_rows": 12,
                        "min_samples_leaf": 4,
                        "max_iter": 20,
                        "blend": 1.0,
                        "cap_mwh": 12.0,
                        "promotion_delta_guard": {
                            "enabled": False,
                            "reference_col": "current_final",
                        },
                        "min_abs_correction_mwh": 0.0,
                        "scope": {
                            "use_focused_guard_rule_union": False,
                            "require_scope_for_application": True,
                            "include_hot_peak": True,
                            "hot_peak_min_maxtemp_f": 90.0,
                            "hot_peak_hours": [14, 15, 16, 17, 18, 19, 20, 21],
                            "include_cloud_solar": False,
                            "include_delta_breeze": False,
                            "include_long_horizon_heat": False,
                        },
                    },
                }
            }
        }
        artifact = build_focused_shape_residual_learner(
            train,
            config,
            forecast_col="Pre_Focused_Guard_Forecast_MWH",
        )
        future = train.head(1).copy()

        out = apply_focused_shape_residual_learner(
            future,
            artifact,
            config,
            forecast_col="Pre_Focused_Guard_Forecast_MWH",
            also_update_cols=(
                "Final_Backtest_Forecast_MWH",
                "Stage_Selected_Forecast_MWH",
            ),
            update_forecast_col=False,
        )

        self.assertIsNotNone(artifact)
        self.assertEqual(out.loc[0, "Focused_Shape_Base_Forecast_MWH"], 105.0)
        self.assertGreater(out.loc[0, "Final_Backtest_Forecast_MWH"], 112.0)
        self.assertEqual(
            out.loc[0, "Final_Forecast_MWH"], out.loc[0, "Final_Backtest_Forecast_MWH"]
        )

    def test_metrics_summary_flags_focused_shape_shadow_beating_final(self):
        df = pd.DataFrame(
            {
                "DT": pd.date_range("2026-07-01", periods=2, freq="h"),
                "Actual_MWH": [100.0, 100.0],
                "Raw_Forecast_MWH": [90.0, 110.0],
                "Final_Backtest_Forecast_MWH": [95.0, 105.0],
                "Focused_Shape_Adjusted_Forecast_MWH": [99.0, 101.0],
            }
        )

        summary = metrics_summary(df)
        audit = build_shadow_stage_promotion_audit(df)

        self.assertTrue(summary["Focused_Shape_Shadow_Beats_Final"])
        self.assertAlmostEqual(summary["Focused_Shape_Shadow_MAE_MWH"], 1.0)
        self.assertAlmostEqual(
            summary["Focused_Shape_Shadow_MAE_Improvement_vs_Final_MWH"], 4.0
        )
        self.assertEqual(audit.loc[0, "Stage"], "focused_shape_shadow")
        self.assertTrue(audit.loc[0, "Beats_Final"])

    def test_shadow_promotion_audit_requires_bias_and_slice_gates(self):
        df = pd.DataFrame(
            {
                "DT": [
                    pd.Timestamp("2026-07-01 17:00"),
                    pd.Timestamp("2026-07-02 14:00"),
                    pd.Timestamp("2026-07-08 00:00"),
                ],
                "Actual_MWH": [100.0, 100.0, 100.0],
                "Final_Backtest_Forecast_MWH": [95.0, 90.0, 90.0],
                "Focused_Shape_Adjusted_Forecast_MWH": [100.0, 100.0, 100.0],
                "Auto_Residual_Full_Shadow_Adjusted_Forecast_MWH": [94.0, 100.0, 106.0],
                "Raw_Forecast_MWH": [95.0, 90.0, 90.0],
                "Forecast_Day": [1, 2, 8],
                "Hour": [17, 14, 0],
                "Month": [7, 7, 7],
                "Temperature_DailyMax": [95.0, 82.0, 82.0],
                "CloudCover_Norm": [0.10, 0.80, 0.10],
                "BTM_Solar_Loss_From_ClearSky_MW": [0.0, 2.0, 0.0],
            }
        )

        audit = build_shadow_stage_promotion_audit(df)
        focused = audit.loc[audit["Stage"].eq("focused_shape_shadow")].iloc[0]
        full_shadow = audit.loc[audit["Stage"].eq("auto_residual_full_shadow")].iloc[0]

        self.assertTrue(focused["Meets_Promotion_Rule"])
        self.assertEqual(focused["Promotion_Gate_Failures"], "none")
        self.assertTrue(full_shadow["Beats_Final"])
        self.assertFalse(full_shadow["Meets_Promotion_Rule"])
        self.assertIn("HotPeak_MAE", full_shadow["Promotion_Gate_Failures"])
        self.assertIn("Day1_NoRegression", full_shadow["Promotion_Gate_Failures"])

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
            features=[
                "MWH_Lag24",
                "MWH_SameHour7DayMean",
            ],
            ensemble_weights={"xgb": 0.5, "lgb": 0.5},
        )

        self.assertIn("MWH_Lag24", out.columns)
        self.assertIn("MWH_SameHour7DayMean", out.columns)
        self.assertEqual(out.loc[0, "MWH_Lag24"], 144.0)
        self.assertEqual(out.loc[0, "MWH_SameHour7DayMean"], 72.0)

    def test_backtest_prediction_whitelists_retain_load_state_lags(self):
        required = {
            "MWH_Lag24",
            "MWH_SameHour7DayMean",
        }

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

        out = apply_focused_scorecard_guard(
            df, config, forecast_col="Final_Forecast_MWH"
        )

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
        self.assertGreater(
            out.loc[0, "BTM_ClearSky_Proxy_MW"], out.loc[0, "BTM_Solar_Proxy_MW"]
        )
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
                    "tree_gap_model_cols": [
                        "XGB_Pred_MWH",
                        "LGB_Pred_MWH",
                        "CatBoost_Pred_MWH",
                    ],
                    "blend": 1.0,
                    "cap_mwh": 10.0,
                }
            }
        }

        out = apply_peak_risk_correction(df, config)

        self.assertAlmostEqual(out.loc[0, "Peak_Risk_Cal_MWH"], 3.5)
        self.assertEqual(out.loc[0, "Peak_Risk_Source"], "tree_peak_gap")
        self.assertAlmostEqual(out.loc[0, "Peak_Risk_Adjusted_Forecast_MWH"], 103.5)

    def test_peak_risk_positive_guard_blocks_cooling_overforecast_risk(self):
        df = pd.DataFrame(
            {
                "DT": [pd.Timestamp("2026-07-29 17:00")],
                "Hour": [17],
                "Temperature_DailyMax": [100.7],
                "Calibrated_Forecast_MWH": [309.1],
                "Raw_Forecast_MWH": [308.6],
                "MWH_SameHour7DayMean": [269.0],
                "TempDrop_Next3Hr_F": [9.8],
                "Prophet_Pred_MWH": [346.7],
            }
        )
        config = {
            "calibration": {
                "peak_risk": {
                    "enabled": True,
                    "hours": [17],
                    "min_maxtemp_f": 78.0,
                    "prophet_gap_threshold_mwh": 6.0,
                    "catboost_gap_threshold_mwh": 99.0,
                    "blend": 0.55,
                    "cap_mwh": 10.0,
                    "positive_guard": {
                        "enabled": True,
                        "overforecast_risk": {
                            "enabled": True,
                            "hours": [17],
                            "min_maxtemp_f": 95.0,
                            "min_raw_minus_samehour_7day_mean_mwh": 30.0,
                            "min_forecast_drop_next3hr_f": 6.0,
                            "max_positive_correction_mwh": 0.0,
                            "blocked_source": "peak_risk_overforecast_guard_blocked",
                        },
                    },
                }
            }
        }

        out = apply_peak_risk_correction(df, config)

        self.assertAlmostEqual(out.loc[0, "Peak_Risk_Cal_MWH"], 0.0)
        self.assertEqual(
            out.loc[0, "Peak_Risk_Source"], "peak_risk_overforecast_guard_blocked"
        )
        self.assertAlmostEqual(out.loc[0, "Peak_Risk_Adjusted_Forecast_MWH"], 309.1)

    def test_peak_risk_positive_guard_preserves_plausibly_low_hot_peak_uplift(self):
        df = pd.DataFrame(
            {
                "DT": [
                    pd.Timestamp("2026-07-21 17:00"),
                    pd.Timestamp("2026-07-21 18:00"),
                ],
                "Hour": [17, 18],
                "Temperature_DailyMax": [105.0, 105.0],
                "Calibrated_Forecast_MWH": [316.0, 318.0],
                "Raw_Forecast_MWH": [316.0, 318.0],
                "Raw_Minus_SameHour7DayMean_MWH": [26.0, 38.0],
                "TempDrop_Next3Hr_F": [8.0, 4.0],
                "Prophet_Pred_MWH": [340.0, 342.0],
            }
        )
        config = {
            "calibration": {
                "peak_risk": {
                    "enabled": True,
                    "hours": [17, 18],
                    "min_maxtemp_f": 78.0,
                    "prophet_gap_threshold_mwh": 6.0,
                    "catboost_gap_threshold_mwh": 99.0,
                    "blend": 1.0,
                    "cap_mwh": 10.0,
                    "positive_guard": {
                        "enabled": True,
                        "overforecast_risk": {
                            "enabled": True,
                            "hours": [17, 18],
                            "min_maxtemp_f": 95.0,
                            "min_raw_minus_samehour_7day_mean_mwh": 30.0,
                            "min_forecast_drop_next3hr_f": 6.0,
                            "max_positive_correction_mwh": 0.0,
                            "blocked_source": "peak_risk_overforecast_guard_blocked",
                        },
                    },
                }
            }
        }

        out = apply_peak_risk_correction(df, config)

        self.assertAlmostEqual(out.loc[0, "Peak_Risk_Cal_MWH"], 10.0)
        self.assertEqual(out.loc[0, "Peak_Risk_Source"], "prophet_peak_gap")
        self.assertAlmostEqual(out.loc[1, "Peak_Risk_Cal_MWH"], 10.0)
        self.assertEqual(out.loc[1, "Peak_Risk_Source"], "prophet_peak_gap")

    def test_peak_risk_hot_ramp_override_bypasses_cooling_guard_when_persistent(self):
        df = pd.DataFrame(
            {
                "DT": [
                    pd.Timestamp("2026-08-02 17:00"),
                    pd.Timestamp("2026-08-02 18:00"),
                ],
                "Hour": [17, 18],
                "Forecast_Day": [2, 2],
                "Temperature_DailyMax": [103.6, 103.6],
                "Calibrated_Forecast_MWH": [320.0, 322.0],
                "Raw_Forecast_MWH": [319.0, 321.0],
                "Raw_Minus_SameHour7DayMean_MWH": [33.0, 33.0],
                "TempDrop_Next3Hr_F": [7.2, 12.5],
                "Prophet_Pred_MWH": [344.0, 346.0],
                "ConsecutiveVeryHotDays95": [2.0, 1.0],
                "ConsecutiveExtremeHotDays100": [2.0, 1.0],
                "DailyMaxTemp_Ramp_1Day": [1.5, 1.5],
            }
        )
        config = {
            "calibration": {
                "hot_ramp_override": {
                    "enabled": True,
                    "hours": [17, 18],
                    "min_maxtemp_f": 100.0,
                    "max_maxtemp_f": 109.9,
                    "min_consecutive_very_hot_days_95": 2,
                    "min_consecutive_extreme_hot_days_100": 2,
                    "guard_bypass_enabled": True,
                    "peak_risk_protected_cap_mwh": 8.0,
                    "scenario_lift_enabled": False,
                },
                "peak_risk": {
                    "enabled": True,
                    "hours": [17, 18],
                    "min_maxtemp_f": 78.0,
                    "prophet_gap_threshold_mwh": 6.0,
                    "catboost_gap_threshold_mwh": 99.0,
                    "blend": 1.0,
                    "cap_mwh": 10.0,
                    "positive_guard": {
                        "enabled": True,
                        "overforecast_risk": {
                            "enabled": True,
                            "hours": [17, 18],
                            "min_maxtemp_f": 95.0,
                            "min_raw_minus_samehour_7day_mean_mwh": 30.0,
                            "min_forecast_drop_next3hr_f": 6.0,
                            "max_positive_correction_mwh": 0.0,
                            "blocked_source": "peak_risk_overforecast_guard_blocked",
                        },
                    },
                },
            }
        }

        out = apply_peak_risk_correction(df, config)

        self.assertAlmostEqual(out.loc[0, "Peak_Risk_Cal_MWH"], 8.0)
        self.assertIn("hot_ramp_guard_bypass", out.loc[0, "Peak_Risk_Source"])
        self.assertAlmostEqual(
            out.loc[0, "Hot_Ramp_Override_PeakRisk_Protected_MWH"], 8.0
        )
        self.assertEqual(out.loc[0, "Hot_Ramp_Override_Gate"], 1)
        self.assertAlmostEqual(out.loc[1, "Peak_Risk_Cal_MWH"], 0.0)
        self.assertEqual(
            out.loc[1, "Peak_Risk_Source"], "peak_risk_overforecast_guard_blocked"
        )
        self.assertEqual(out.loc[1, "Hot_Ramp_Override_Gate"], 0)

    def test_hot_ramp_scenario_override_lifts_toward_hot_stress_gap_with_cap(self):
        df = pd.DataFrame(
            {
                "DT": [
                    pd.Timestamp("2026-08-08 17:00"),
                    pd.Timestamp("2026-08-08 18:00"),
                    pd.Timestamp("2026-08-08 16:00"),
                    pd.Timestamp("2026-08-08 17:00"),
                    pd.Timestamp("2026-08-09 17:00"),
                ],
                "Hour": [17, 18, 16, 17, 17],
                "Forecast_Day": [6, 6, 6, 6, 7],
                "Final_Forecast_MWH": [334.0, 333.0, 325.0, 334.0, 334.0],
                "Stage_Selected_Forecast_MWH": [334.0, 333.0, 325.0, 334.0, 334.0],
                "Calibrated_Forecast_MWH": [334.0, 333.0, 325.0, 334.0, 334.0],
                "WeatherScenario_hot_stress_5f_P50_MWH": [
                    354.0,
                    344.0,
                    345.0,
                    354.0,
                    354.0,
                ],
                "Temperature_DailyMax": [108.4, 108.4, 108.4, 110.2, 108.4],
                "ConsecutiveVeryHotDays95": [5.0, 5.0, 5.0, 5.0, 5.0],
                "ConsecutiveExtremeHotDays100": [5.0, 5.0, 5.0, 5.0, 5.0],
                "Hot_Ramp_Override_Cal_MWH": [0.0, 0.0, 0.0, 0.0, 6.5],
                "Hot_Ramp_Override_Source": [
                    "none",
                    "none",
                    "none",
                    "none",
                    "hot_ramp_guard_bypass",
                ],
            }
        )
        config = {
            "calibration": {
                "hot_ramp_override": {
                    "enabled": True,
                    "hours": [17, 18, 19, 20],
                    "min_maxtemp_f": 100.0,
                    "max_maxtemp_f": 109.9,
                    "min_consecutive_very_hot_days_95": 2,
                    "min_consecutive_extreme_hot_days_100": 2,
                    "scenario_lift_enabled": True,
                    "scenario_gap_blend": 0.40,
                    "scenario_lift_cap_mwh": 6.0,
                    "total_cap_mwh": 8.0,
                }
            }
        }

        out = apply_hot_ramp_scenario_override(
            df,
            config=config,
            base_col="Final_Forecast_MWH",
            also_update_cols=("Stage_Selected_Forecast_MWH", "Calibrated_Forecast_MWH"),
        )

        self.assertAlmostEqual(out.loc[0, "Hot_Ramp_Override_Scenario_Lift_MWH"], 6.0)
        self.assertAlmostEqual(out.loc[0, "Final_Forecast_MWH"], 340.0)
        self.assertAlmostEqual(out.loc[0, "Stage_Selected_Forecast_MWH"], 340.0)
        self.assertAlmostEqual(out.loc[0, "Calibrated_Forecast_MWH"], 340.0)
        self.assertEqual(
            out.loc[0, "Hot_Ramp_Override_Source"], "hot_ramp_hot_stress_scenario_lift"
        )
        self.assertAlmostEqual(out.loc[1, "Hot_Ramp_Override_Scenario_Lift_MWH"], 4.4)
        self.assertEqual(out.loc[2, "Hot_Ramp_Override_Gate"], 0)
        self.assertEqual(out.loc[3, "Hot_Ramp_Override_Gate"], 0)
        self.assertAlmostEqual(out.loc[4, "Hot_Ramp_Override_Scenario_Lift_MWH"], 1.5)
        self.assertAlmostEqual(out.loc[4, "Hot_Ramp_Override_Cal_MWH"], 8.0)
        self.assertAlmostEqual(out.loc[4, "Final_Forecast_MWH"], 335.5)

    def test_day1_live_ramp_override_carries_observed_delta_to_hot_peak(self):
        history_rows = []
        for hour, yesterday, today in [
            (5, 129.97, 149.76),
            (6, 126.85, 152.40),
            (7, 132.93, 160.42),
        ]:
            history_rows.append(
                {
                    "DT": pd.Timestamp("2026-08-02") + pd.Timedelta(hours=hour),
                    "MWH": yesterday,
                }
            )
            history_rows.append(
                {
                    "DT": pd.Timestamp("2026-08-03") + pd.Timedelta(hours=hour),
                    "MWH": today,
                }
            )
        history = pd.DataFrame(history_rows)
        future = pd.DataFrame(
            {
                "DT": [
                    pd.Timestamp("2026-08-03 15:00"),
                    pd.Timestamp("2026-08-03 18:00"),
                    pd.Timestamp("2026-08-04 18:00"),
                ],
                "Hour": [15, 18, 18],
                "Forecast_Day": [1, 1, 2],
                "Final_Forecast_MWH": [327.0, 346.5, 338.0],
                "Stage_Selected_Forecast_MWH": [327.0, 346.5, 338.0],
                "Calibrated_Forecast_MWH": [327.0, 346.5, 338.0],
                "Temperature_DailyMax": [105.0, 105.0, 105.0],
                "MWH_Lag24": [300.7, 335.7, 335.7],
            }
        )
        config = {
            "calibration": {
                "day1_live_ramp_override": {
                    "enabled": True,
                    "hours": [16, 17, 18, 19, 20],
                    "min_maxtemp_f": 100.0,
                    "min_latest_actual_hour": 6,
                    "min_pair_hours": 3,
                    "observation_tail_hours": 3,
                    "observed_delta_stat": "latest",
                    "min_observed_delta_mwh": 15.0,
                    "carry_fraction": 0.90,
                    "cap_mwh": 16.0,
                }
            }
        }

        out = apply_day1_live_ramp_override(
            future,
            history,
            config=config,
            base_col="Final_Forecast_MWH",
            also_update_cols=("Stage_Selected_Forecast_MWH", "Calibrated_Forecast_MWH"),
        )

        observed_delta = 160.42 - 132.93
        target = 335.7 + observed_delta * 0.90
        expected_lift = target - 346.5
        self.assertEqual(out.loc[0, "Day1_Live_Ramp_Gate"], 0)
        self.assertEqual(out.loc[1, "Day1_Live_Ramp_Gate"], 1)
        self.assertAlmostEqual(
            out.loc[1, "Day1_Live_Ramp_Observed_Delta_MWH"], observed_delta
        )
        self.assertAlmostEqual(out.loc[1, "Day1_Live_Ramp_Target_MWH"], target)
        self.assertAlmostEqual(out.loc[1, "Day1_Live_Ramp_Cal_MWH"], expected_lift)
        self.assertAlmostEqual(out.loc[1, "Final_Forecast_MWH"], target)
        self.assertAlmostEqual(out.loc[1, "Calibrated_Forecast_MWH"], target)
        self.assertEqual(out.loc[1, "Day1_Live_Ramp_Source"], "day1_live_same_day_ramp")
        self.assertEqual(out.loc[2, "Day1_Live_Ramp_Gate"], 0)

    def test_multiday_live_heat_anchor_lifts_muted_following_hot_day(self):
        future = pd.DataFrame(
            {
                "DT": [
                    pd.Timestamp("2026-08-03 18:00"),
                    pd.Timestamp("2026-08-04 18:00"),
                    pd.Timestamp("2026-08-08 18:00"),
                ],
                "Hour": [18, 18, 18],
                "Forecast_Day": [1, 2, 6],
                "Final_Forecast_MWH": [360.0, 338.0, 342.0],
                "Stage_Selected_Forecast_MWH": [360.0, 338.0, 342.0],
                "Calibrated_Forecast_MWH": [360.0, 338.0, 342.0],
                "Temperature_DailyMax": [105.0, 104.0, 108.0],
                "ConsecutiveExtremeHotDays100": [3.0, 4.0, 8.0],
                "Day1_Live_Ramp_Cal_MWH": [16.0, 0.0, 0.0],
                "Day1_Live_Ramp_Observed_Delta_MWH": [31.3, 31.3, 31.3],
            }
        )
        config = {
            "calibration": {
                "multiday_live_heat_anchor_override": {
                    "enabled": True,
                    "hours": [16, 17, 18, 19, 20],
                    "anchor_hours": [16, 17, 18, 19],
                    "min_forecast_day": 2,
                    "max_forecast_day": 7,
                    "min_maxtemp_f": 102.0,
                    "min_anchor_observed_delta_mwh": 20.0,
                    "min_anchor_live_ramp_mwh": 8.0,
                    "min_consecutive_extreme_hot_days_100": 2,
                    "anchor_discount_mwh": 7.0,
                    "lead_decay_mwh_per_day": 1.0,
                    "temp_shortfall_discount_mwh_per_f": 2.0,
                    "temp_excess_credit_mwh_per_f": 1.0,
                    "weekend_discount_mwh": 12.0,
                    "sunday_extra_discount_mwh": 2.0,
                    "hour_target_offsets_mwh": {18: 0.0},
                    "cap_mwh": 12.0,
                    "max_target_mwh": 365.0,
                }
            }
        }

        out = apply_multiday_live_heat_anchor_override(
            future,
            config=config,
            base_col="Final_Forecast_MWH",
            also_update_cols=("Stage_Selected_Forecast_MWH", "Calibrated_Forecast_MWH"),
        )

        self.assertEqual(out.loc[0, "MultiDay_Heat_Anchor_Gate"], 0)
        self.assertEqual(out.loc[1, "MultiDay_Heat_Anchor_Gate"], 1)
        self.assertAlmostEqual(out.loc[1, "MultiDay_Heat_Anchor_Target_MWH"], 350.0)
        self.assertAlmostEqual(out.loc[1, "MultiDay_Heat_Anchor_Cal_MWH"], 12.0)
        self.assertAlmostEqual(out.loc[1, "Final_Forecast_MWH"], 350.0)
        self.assertAlmostEqual(out.loc[1, "Calibrated_Forecast_MWH"], 350.0)
        self.assertEqual(
            out.loc[1, "MultiDay_Heat_Anchor_Source"], "multiday_live_heat_anchor"
        )
        self.assertEqual(out.loc[2, "MultiDay_Heat_Anchor_Gate"], 1)
        self.assertAlmostEqual(out.loc[2, "MultiDay_Heat_Anchor_Cal_MWH"], 0.0)
        self.assertEqual(
            out.loc[2, "MultiDay_Heat_Anchor_Source"], "already_at_heat_anchor_target"
        )

    def test_display_df_exports_heat_ramp_review_columns(self):
        train = pd.DataFrame(
            {
                "DT": [pd.Timestamp("2026-08-02 17:00")],
                "MWH": [333.0],
                "Temperature": [103.5],
                "Temperature_DailyMax": [103.6],
                "DailyMaxTemp_Ramp_1Day": [1.6],
                "ConsecutiveExtremeHotDays100": [2.0],
            }
        )
        future = pd.DataFrame(
            {
                "DT": [pd.Timestamp("2026-08-03 17:00")],
                "Final_Forecast_MWH": [342.0],
                "Calibrated_Forecast_MWH": [342.0],
                "Temperature": [105.1],
                "Temperature_DailyMax": [105.3],
                "DailyMaxTemp_Ramp_1Day": [1.7],
                "ConsecutiveExtremeHotDays100": [3.0],
                "Hot_Ramp_Override_Gate": [1],
                "Hot_Ramp_Override_Scenario_Lift_MWH": [6.0],
                "Day1_Live_Ramp_Gate": [1],
                "Day1_Live_Ramp_Cal_MWH": [14.0],
                "Day1_Live_Ramp_Target_MWH": [360.0],
                "MultiDay_Heat_Anchor_Gate": [1],
                "MultiDay_Heat_Anchor_Cal_MWH": [10.0],
                "MultiDay_Heat_Anchor_Target_MWH": [352.0],
                "MWH_Lag24": [335.7],
            }
        )

        out = build_display_df(train, future)

        self.assertIn("DailyMaxTemp_Ramp_1Day", out.columns)
        self.assertIn("ConsecutiveExtremeHotDays100", out.columns)
        self.assertIn("Hot_Ramp_Override_Scenario_Lift_MWH", out.columns)
        self.assertIn("Day1_Live_Ramp_Cal_MWH", out.columns)
        self.assertIn("MultiDay_Heat_Anchor_Cal_MWH", out.columns)
        self.assertIn("MWH_Lag24", out.columns)
        self.assertEqual(
            out.loc[out["Actual"].notna(), "DailyMaxTemp_Ramp_1Day"].iloc[0], 1.6
        )
        self.assertEqual(
            out.loc[
                out["Forecast"].notna(), "Hot_Ramp_Override_Scenario_Lift_MWH"
            ].iloc[0],
            6.0,
        )
        self.assertEqual(
            out.loc[out["Forecast"].notna(), "Day1_Live_Ramp_Target_MWH"].iloc[0], 360.0
        )
        self.assertEqual(
            out.loc[out["Forecast"].notna(), "MultiDay_Heat_Anchor_Target_MWH"].iloc[0],
            352.0,
        )

    def test_stage_selector_conditional_override_respects_hour_filter(self):
        df = pd.DataFrame(
            {
                "DT": [
                    pd.Timestamp("2026-07-15 12:00"),
                    pd.Timestamp("2026-07-15 16:00"),
                ],
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

        out = apply_operational_stage_selector(
            df, config, forecast_col="Final_Forecast_MWH"
        )

        self.assertEqual(out.loc[0, "Stage_Selected_Forecast_MWH"], 250.0)
        self.assertEqual(out.loc[0, "Stage_Selector_Source"], "Raw_Forecast_MWH")
        self.assertIn(
            "conditional_stage_override:summer_high_temp_raw_override",
            out.loc[0, "Stage_Selector_Reason"],
        )
        self.assertEqual(out.loc[1, "Stage_Selected_Forecast_MWH"], 282.0)
        self.assertEqual(out.loc[1, "Stage_Selector_Source"], "Final_Forecast_MWH")
        self.assertNotIn(
            "conditional_stage_override", out.loc[1, "Stage_Selector_Reason"]
        )

    def test_stage_selector_day1_summer_hot_peak_override_promotes_peak_risk(self):
        df = pd.DataFrame(
            {
                "DT": [
                    pd.Timestamp("2026-08-02 18:00"),
                    pd.Timestamp("2026-08-02 15:00"),
                    pd.Timestamp("2026-08-03 18:00"),
                    pd.Timestamp("2026-08-02 19:00"),
                ],
                "Forecast_Day": [1, 1, 2, 1],
                "Season": ["Summer", "Summer", "Summer", "Summer"],
                "Hour": [18, 15, 18, 19],
                "Temperature_DailyMax": [105.0, 105.0, 105.0, 105.0],
                "Targeted_Meta_Adjusted_Forecast_MWH": [322.0, 295.0, 330.0, 333.0],
                "Peak_Risk_Adjusted_Forecast_MWH": [331.0, 302.0, 340.0, 331.0],
                "Final_Forecast_MWH": [324.0, 297.0, 332.0, 332.0],
            }
        )
        config = {
            "calibration": {
                "stage_selector": {
                    "enabled": True,
                    "day1_stage": "targeted_meta",
                    "days2to3_stage": "targeted_meta",
                    "hot_peak_final_stack_override": False,
                    "conditional_stage_overrides": [
                        {
                            "name": "day1_summer_hot_peak_peak_risk_override",
                            "enabled": True,
                            "seasons": ["Summer"],
                            "hours": [16, 17, 18, 19, 20],
                            "min_maxtemp_f": 103.0,
                            "min_forecast_day": 1,
                            "max_forecast_day": 1,
                            "stage": "peak_risk",
                            "only_if_stage_above_current": True,
                        }
                    ],
                }
            }
        }

        out = apply_operational_stage_selector(
            df, config, forecast_col="Final_Forecast_MWH"
        )

        self.assertEqual(out.loc[0, "Stage_Selected_Forecast_MWH"], 331.0)
        self.assertEqual(
            out.loc[0, "Stage_Selector_Source"], "Peak_Risk_Adjusted_Forecast_MWH"
        )
        self.assertIn(
            "conditional_stage_override:day1_summer_hot_peak_peak_risk_override",
            out.loc[0, "Stage_Selector_Reason"],
        )
        self.assertEqual(out.loc[1, "Stage_Selected_Forecast_MWH"], 295.0)
        self.assertNotIn(
            "day1_summer_hot_peak_peak_risk_override",
            out.loc[1, "Stage_Selector_Reason"],
        )
        self.assertEqual(out.loc[2, "Stage_Selected_Forecast_MWH"], 330.0)
        self.assertNotIn(
            "day1_summer_hot_peak_peak_risk_override",
            out.loc[2, "Stage_Selector_Reason"],
        )
        self.assertEqual(out.loc[3, "Stage_Selected_Forecast_MWH"], 333.0)
        self.assertEqual(
            out.loc[3, "Stage_Selector_Source"], "Targeted_Meta_Adjusted_Forecast_MWH"
        )
        self.assertNotIn(
            "day1_summer_hot_peak_peak_risk_override",
            out.loc[3, "Stage_Selector_Reason"],
        )

    def test_stage_selector_day1_cloud_solar_override_uses_peak_risk_only_for_cloud_slice(
        self,
    ):
        df = pd.DataFrame(
            {
                "DT": [
                    pd.Timestamp("2026-07-15 12:00"),
                    pd.Timestamp("2026-07-15 13:00"),
                    pd.Timestamp("2026-07-16 12:00"),
                ],
                "Forecast_Day": [1, 1, 2],
                "Hour": [12, 13, 12],
                "Temperature_DailyMax": [95.0, 95.0, 95.0],
                "CloudCover_Norm": [0.80, 0.10, 0.80],
                "BTM_Solar_Loss_From_ClearSky_MW": [3.0, 0.0, 3.0],
                "Targeted_Meta_Adjusted_Forecast_MWH": [190.0, 191.0, 192.0],
                "Peak_Risk_Adjusted_Forecast_MWH": [205.0, 206.0, 207.0],
                "Final_Forecast_MWH": [210.0, 211.0, 212.0],
            }
        )
        config = {
            "calibration": {
                "stage_selector": {
                    "enabled": True,
                    "day1_stage": "targeted_meta",
                    "days2to3_stage": "targeted_meta",
                    "cloud_solar_stage_override": {
                        "enabled": True,
                        "hours": [10, 11, 12, 13, 14, 15, 16],
                        "min_forecast_day": 1,
                        "max_forecast_day": 1,
                        "min_cloud_cover_norm": 0.60,
                        "min_solar_loss_mw": 1.25,
                        "stage": "peak_risk",
                    },
                }
            }
        }

        out = apply_operational_stage_selector(
            df, config, forecast_col="Final_Forecast_MWH"
        )

        self.assertEqual(out.loc[0, "Stage_Selected_Forecast_MWH"], 205.0)
        self.assertEqual(
            out.loc[0, "Stage_Selector_Source"], "Peak_Risk_Adjusted_Forecast_MWH"
        )
        self.assertIn("cloud_solar_stage_override", out.loc[0, "Stage_Selector_Reason"])
        self.assertEqual(out.loc[1, "Stage_Selected_Forecast_MWH"], 191.0)
        self.assertEqual(
            out.loc[1, "Stage_Selector_Source"], "Targeted_Meta_Adjusted_Forecast_MWH"
        )
        self.assertNotIn(
            "cloud_solar_stage_override", out.loc[1, "Stage_Selector_Reason"]
        )
        self.assertEqual(out.loc[2, "Stage_Selected_Forecast_MWH"], 192.0)
        self.assertEqual(
            out.loc[2, "Stage_Selector_Source"], "Targeted_Meta_Adjusted_Forecast_MWH"
        )
        self.assertNotIn(
            "cloud_solar_stage_override", out.loc[2, "Stage_Selector_Reason"]
        )

    def test_long_horizon_peak_month_correction_can_scale_specific_month_days(self):
        df = pd.DataFrame(
            {
                "DT": [
                    pd.Timestamp("2026-09-08 14:00"),
                    pd.Timestamp("2026-09-11 14:00"),
                    pd.Timestamp("2026-07-08 14:00"),
                ],
                "Forecast_Day": [8, 11, 8],
                "Month": [9, 9, 7],
                "Hour": [14, 14, 14],
                "Temperature_DailyMax": [82.0, 82.0, 100.0],
                "Final_Forecast_MWH": [100.0, 100.0, 100.0],
            }
        )
        config = {
            "calibration": {
                "stage_selector": {
                    "enabled": True,
                    "long_horizon_peak_hot_month_correction": {
                        "enabled": True,
                        "min_forecast_day": 8,
                        "max_forecast_day": 16,
                        "peak_hours": [14],
                        "hot_hours": [16],
                        "peak_month_offsets_mwh": {"7": 7.76, "9": -6.44},
                        "peak_month_forecast_day_scales": {"9": {"8": 0.0}},
                    },
                }
            }
        }

        out = apply_operational_stage_selector(
            df, config, forecast_col="Final_Forecast_MWH"
        )

        self.assertEqual(
            out["Long_Horizon_Peak_Month_Correction_MWH"].tolist(), [0.0, -6.44, 7.76]
        )
        self.assertEqual(
            out["Stage_Selected_Forecast_MWH"].tolist(), [100.0, 93.56, 107.76]
        )

    def test_blend_predictions_pads_short_optional_components(self):
        blended = blend_predictions(
            [10.0, 20.0, 30.0],
            [12.0, 22.0, 32.0],
            {"xgb": 0.5, "lgb": 0.3, "catboost": 0.2},
            catboost_pred=[14.0],
        )

        self.assertTrue(np.allclose(blended[0], 11.4))
        self.assertTrue(np.allclose(blended[1:], [20.75, 30.75]))

    def test_recent_residual_profile_estimates_shrunk_ar_phi(self):
        residuals = np.arange(1.0, 49.0)
        df = pd.DataFrame(
            {
                "DT": pd.date_range(
                    "2026-07-01 00:00", periods=len(residuals), freq="h"
                ),
                "Actual_MWH": 100.0 + residuals,
                "Raw_Forecast_MWH": 100.0,
            }
        )
        config = {
            "calibration": {
                "recent_residual": {
                    "enabled": True,
                    "cap_mwh": 10.0,
                    "ar_residual": {
                        "enabled": True,
                        "lookback_hours": 72,
                        "min_pairs": 10,
                        "phi_shrink_k": 24.0,
                        "phi_cap": 0.40,
                    },
                }
            }
        }

        profile = build_recent_residual_profile(df, config)

        ar = profile["ar_residual"]
        self.assertEqual(ar["n_pairs"], 47)
        self.assertAlmostEqual(ar["phi"], 0.40, places=6)
        self.assertEqual(ar["latest_residual"], 48.0)

    def test_ar_recent_residual_correction_caps_and_decays_by_lead(self):
        future = pd.DataFrame(
            {
                "DT": pd.date_range("2026-07-03 00:00", periods=2, freq="h"),
                "Calibrated_Forecast_MWH": [100.0, 100.0],
            }
        )
        profile = {
            "enabled": True,
            "ar_residual": {
                "enabled": True,
                "phi": 0.50,
                "latest_residual": 8.0,
                "same_hour_mean": {},
            },
        }
        config = {
            "calibration": {
                "recent_residual": {
                    "enabled": True,
                    "cap_mwh": 10.0,
                    "weights": {
                        "recent_mean": 0.0,
                        "last24_mean": 0.0,
                        "same_hour": 0.0,
                        "hourgroup": 0.0,
                        "global": 0.0,
                    },
                    "ar_residual": {
                        "enabled": True,
                        "blend": 1.0,
                        "same_hour_blend": 0.0,
                        "cap_mwh": 3.0,
                        "decay_hours": 1.0,
                        "min_decay": 0.0,
                        "forecast_day_scales": {
                            "day1": 1.0,
                            "days2to3": 1.0,
                            "days4to7": 1.0,
                            "days8plus": 1.0,
                        },
                    },
                }
            }
        }

        out = apply_recent_residual_correction(future, profile, config)

        self.assertAlmostEqual(out.loc[0, "AR_Residual_Correction_MWH"], 3.0, places=6)
        self.assertAlmostEqual(
            out.loc[1, "AR_Residual_Correction_MWH"], 4.0 * np.exp(-1.0), places=6
        )
        self.assertAlmostEqual(out.loc[0, "Recent_Level_Correction_MWH"], 3.0, places=6)
        self.assertIn("ar1_latest_residual", out.loc[0, "Recent_Correction_Source"])

    def test_ar_recent_residual_correction_blends_same_hour_residual(self):
        future = pd.DataFrame(
            {
                "DT": [pd.Timestamp("2026-07-03 00:00")],
                "Calibrated_Forecast_MWH": [100.0],
            }
        )
        profile = {
            "enabled": True,
            "ar_residual": {
                "enabled": True,
                "phi": 0.50,
                "latest_residual": 8.0,
                "same_hour_mean": {0: 2.0},
            },
        }
        config = {
            "calibration": {
                "recent_residual": {
                    "enabled": True,
                    "cap_mwh": 10.0,
                    "weights": {
                        "recent_mean": 0.0,
                        "last24_mean": 0.0,
                        "same_hour": 0.0,
                        "hourgroup": 0.0,
                        "global": 0.0,
                    },
                    "ar_residual": {
                        "enabled": True,
                        "blend": 1.0,
                        "same_hour_blend": 0.50,
                        "cap_mwh": 10.0,
                        "decay_hours": 24.0,
                        "min_decay": 0.0,
                        "forecast_day_scales": {"day1": 1.0},
                    },
                }
            }
        }

        out = apply_recent_residual_correction(future, profile, config)

        self.assertAlmostEqual(out.loc[0, "AR_Residual_Correction_MWH"], 3.0, places=6)
        self.assertEqual(
            out.loc[0, "AR_Residual_Source"], "ar1_latest_residual+same_hour"
        )

    def test_recent_residual_backtest_ar_uses_only_prior_rows(self):
        residuals = np.arange(1.0, 8.0)
        df = pd.DataFrame(
            {
                "DT": pd.date_range(
                    "2026-07-01 00:00", periods=len(residuals), freq="h"
                ),
                "Actual_MWH": 100.0 + residuals,
                "Raw_Forecast_MWH": 100.0,
            }
        )
        config = {
            "calibration": {
                "recent_residual": {
                    "enabled": True,
                    "cap_mwh": 10.0,
                    "weights": {
                        "recent_mean": 0.0,
                        "last24_mean": 0.0,
                        "same_hour": 0.0,
                        "hourgroup": 0.0,
                        "global": 0.0,
                    },
                    "ar_residual": {
                        "enabled": True,
                        "lookback_hours": 24,
                        "min_pairs": 2,
                        "phi_shrink_k": 0.0,
                        "phi_cap": 1.0,
                        "blend": 1.0,
                        "same_hour_blend": 0.0,
                        "cap_mwh": 10.0,
                        "decay_hours": 24.0,
                        "forecast_day_scales": {"day1": 1.0},
                    },
                }
            }
        }

        out = simulate_recent_residual_correction_backtest(df, config)

        self.assertEqual(out.loc[0, "AR_Residual_Correction_MWH"], 0.0)
        self.assertEqual(out.loc[1, "AR_Residual_Correction_MWH"], 0.0)
        self.assertAlmostEqual(out.loc[3, "AR_Residual_Correction_MWH"], 3.0, places=6)
        self.assertEqual(out.loc[3, "AR_Residual_Latest_MWH"], 3.0)

    def test_origin_day_state_profile_estimates_shrunk_day_level_state(self):
        residuals = [1.0] * 24 + [3.0] * 24 + [5.0] * 24
        df = pd.DataFrame(
            {
                "DT": pd.date_range(
                    "2026-07-01 00:00", periods=len(residuals), freq="h"
                ),
                "Actual_MWH": 100.0 + np.array(residuals),
                "Raw_Forecast_MWH": 100.0,
            }
        )
        config = {
            "calibration": {
                "recent_residual": {
                    "enabled": True,
                    "cap_mwh": 10.0,
                    "origin_day_state": {
                        "enabled": True,
                        "lookback_days": 7,
                        "min_day_hours": 12,
                        "min_days": 2,
                        "min_total_hours": 24,
                        "latest_day_weight": 0.50,
                        "shrink_days": 0.0,
                        "shrink_hours": 0.0,
                        "state_cap_mwh": 10.0,
                    },
                }
            }
        }

        profile = build_recent_residual_profile(df, config)

        state = profile["origin_day_state"]
        self.assertEqual(state["n_days"], 3)
        self.assertEqual(state["n_hours"], 72)
        self.assertAlmostEqual(state["latest_day_mean"], 5.0, places=6)
        self.assertAlmostEqual(state["recent_day_mean"], 3.0, places=6)
        self.assertAlmostEqual(state["state_mwh"], 4.0, places=6)

    def test_origin_day_state_correction_caps_and_decays_by_forecast_day(self):
        future = pd.DataFrame(
            {
                "DT": pd.date_range("2026-07-04 00:00", periods=2, freq="24h"),
                "Forecast_Day": [1, 2],
                "Calibrated_Forecast_MWH": [100.0, 100.0],
            }
        )
        profile = {
            "enabled": True,
            "origin_day_state": {
                "enabled": True,
                "state_mwh": 8.0,
                "latest_day_mean": 8.0,
                "same_sign_run_hours": 24,
                "same_hour_mean": {},
                "hourgroup_mean": {},
            },
        }
        config = {
            "calibration": {
                "recent_residual": {
                    "enabled": True,
                    "cap_mwh": 10.0,
                    "weights": {
                        "recent_mean": 0.0,
                        "last24_mean": 0.0,
                        "same_hour": 0.0,
                        "hourgroup": 0.0,
                        "global": 0.0,
                    },
                    "origin_day_state": {
                        "enabled": True,
                        "blend": 1.0,
                        "hourgroup_blend": 0.0,
                        "same_hour_blend": 0.0,
                        "cap_mwh": 3.0,
                        "decay_days": 1.0,
                        "forecast_day_scales": {
                            "day1": 1.0,
                            "days2to3": 1.0,
                            "days4to7": 1.0,
                            "days8plus": 1.0,
                        },
                        "cap_by_forecast_day": {"day1": 2.0, "days2to3": 3.0},
                    },
                }
            }
        }

        out = apply_recent_residual_correction(future, profile, config)

        self.assertAlmostEqual(
            out.loc[0, "OriginDay_State_Correction_MWH"], 2.0, places=6
        )
        self.assertAlmostEqual(
            out.loc[1, "OriginDay_State_Correction_MWH"], 8.0 * np.exp(-1.0), places=6
        )
        self.assertIn("origin_day_state", out.loc[0, "Recent_Correction_Source"])

    def test_origin_day_state_ignores_opposite_signed_hour_component(self):
        future = pd.DataFrame(
            {
                "DT": [pd.Timestamp("2026-07-04 10:00")],
                "Hour": [10],
                "HourGroup": ["Midday"],
                "Calibrated_Forecast_MWH": [100.0],
            }
        )
        profile = {
            "enabled": True,
            "origin_day_state": {
                "enabled": True,
                "state_mwh": 4.0,
                "latest_day_mean": 4.0,
                "same_sign_run_hours": 24,
                "same_hour_mean": {10: -10.0},
                "hourgroup_mean": {"Midday": 2.0},
            },
        }
        config = {
            "calibration": {
                "recent_residual": {
                    "enabled": True,
                    "cap_mwh": 10.0,
                    "weights": {
                        "recent_mean": 0.0,
                        "last24_mean": 0.0,
                        "same_hour": 0.0,
                        "hourgroup": 0.0,
                        "global": 0.0,
                    },
                    "origin_day_state": {
                        "enabled": True,
                        "blend": 1.0,
                        "hourgroup_blend": 0.25,
                        "same_hour_blend": 0.50,
                        "require_component_same_sign": True,
                        "cap_mwh": 10.0,
                        "forecast_day_scales": {"day1": 1.0},
                    },
                }
            }
        }

        out = apply_recent_residual_correction(future, profile, config)

        self.assertAlmostEqual(
            out.loc[0, "OriginDay_State_Correction_MWH"], 3.5, places=6
        )
        self.assertEqual(
            out.loc[0, "OriginDay_State_Source"], "origin_day_state+hourgroup"
        )

    def test_recent_residual_backtest_origin_day_state_uses_only_prior_rows(self):
        residuals = [5.0] * 24 + [1.0]
        df = pd.DataFrame(
            {
                "DT": pd.date_range(
                    "2026-07-01 00:00", periods=len(residuals), freq="h"
                ),
                "Actual_MWH": 100.0 + np.array(residuals),
                "Raw_Forecast_MWH": 100.0,
            }
        )
        config = {
            "calibration": {
                "recent_residual": {
                    "enabled": True,
                    "cap_mwh": 10.0,
                    "weights": {
                        "recent_mean": 0.0,
                        "last24_mean": 0.0,
                        "same_hour": 0.0,
                        "hourgroup": 0.0,
                        "global": 0.0,
                    },
                    "origin_day_state": {
                        "enabled": True,
                        "lookback_days": 7,
                        "min_day_hours": 12,
                        "min_days": 1,
                        "min_total_hours": 12,
                        "latest_day_weight": 1.0,
                        "shrink_days": 0.0,
                        "shrink_hours": 0.0,
                        "blend": 1.0,
                        "hourgroup_blend": 0.0,
                        "same_hour_blend": 0.0,
                        "cap_mwh": 10.0,
                        "forecast_day_scales": {"day1": 1.0},
                    },
                }
            }
        }

        out = simulate_recent_residual_correction_backtest(df, config)

        self.assertEqual(out.loc[0, "OriginDay_State_Correction_MWH"], 0.0)
        self.assertAlmostEqual(
            out.loc[24, "OriginDay_State_Correction_MWH"], 5.0, places=6
        )
        self.assertAlmostEqual(out.loc[24, "OriginDay_Latest_Day_MWH"], 5.0, places=6)

    def test_apply_dynamic_temperature_calibration_adjusts_temperatures_with_decay(
        self,
    ):
        hist_wx = pd.DataFrame(
            {
                "DT": pd.date_range("2026-06-15 00:00", periods=24, freq="h"),
                "TempF": [90.0] * 24,
                "LocalStation_TempF": [85.0] * 24,  # Cooler by 5 degrees consistently
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
                    "dynamic_blend": 0.80,  # Expected bias: -5.0 * 0.80 = -4.0
                    "dynamic_decay_hours": 24.0,
                }
            }
        }

        out = apply_dynamic_temperature_calibration(fut_wx, hist_wx, config)

        self.assertIn("Dynamic_Weather_Correction_F", out.columns)
        # Verify the first hour (hour 0) has approx -4.0 degrees correction
        self.assertAlmostEqual(
            out.loc[0, "Dynamic_Weather_Correction_F"], -4.0, places=2
        )
        # Verify the 24th hour has decayed towards 0 (factor of exp(-23/24) = ~0.38 -> -4 * 0.38 = ~-1.5)
        self.assertTrue(-4.0 < out.loc[23, "Dynamic_Weather_Correction_F"] < -1.0)
        self.assertAlmostEqual(out.loc[0, "TempF"], 91.0, places=2)


if __name__ == "__main__":
    unittest.main()
