import unittest

from forecasting.main import _disable_windows_platform_wmi_probe, load_config

_disable_windows_platform_wmi_probe()

import pandas as pd

from forecasting.diagnostics.forecast_diagnostics import (
    build_production_readiness_scorecard,
)
from forecasting.forecast.anomaly_exclusions import (
    drop_excluded_intervals,
    excluded_interval_mask,
)
from forecasting.forecast.focused_scorecard_guard import apply_focused_scorecard_guard

DER_CONFIG = {
    "anomaly_exclusions": {
        "enabled": True,
        "events": [
            {
                "name": "2026-07-15 DER dispatch",
                "date": "2026-07-15",
                "he_start": 18,
                "he_end": 21,
            },
        ],
    }
}


def _july_day_frame(day: str, tz: str | None = "America/Los_Angeles") -> pd.DataFrame:
    return pd.DataFrame(
        {
            "DT": pd.date_range(f"{day} 00:00", periods=24, freq="h", tz=tz),
            "Actual_MWH": range(24),
        }
    )


class AnomalyExclusionMaskTests(unittest.TestCase):
    def test_he_range_maps_to_completed_hour_label(self):
        df = _july_day_frame("2026-07-15")
        mask = excluded_interval_mask(df, DER_CONFIG)
        self.assertEqual(sorted(df.loc[mask, "DT"].dt.hour.tolist()), [18, 19, 20, 21])

    def test_drop_removes_only_excluded_rows(self):
        df = _july_day_frame("2026-07-15")
        out = drop_excluded_intervals(df, DER_CONFIG)
        remaining = out["DT"].dt.hour.tolist()
        self.assertEqual(len(out), 20)
        self.assertNotIn(18, remaining)
        self.assertNotIn(19, remaining)
        self.assertNotIn(20, remaining)
        self.assertNotIn(21, remaining)
        self.assertIn(17, remaining)
        self.assertIn(16, remaining)
        self.assertIn(15, remaining)

    def test_other_day_untouched(self):
        df = _july_day_frame("2026-07-14")
        self.assertFalse(excluded_interval_mask(df, DER_CONFIG).any())

    def test_disabled_flag_excludes_nothing(self):
        df = _july_day_frame("2026-07-15")
        cfg = {
            "anomaly_exclusions": {
                "enabled": False,
                "events": DER_CONFIG["anomaly_exclusions"]["events"],
            }
        }
        self.assertFalse(excluded_interval_mask(df, cfg).any())

    def test_no_events_or_config_excludes_nothing(self):
        df = _july_day_frame("2026-07-15")
        self.assertFalse(excluded_interval_mask(df, {}).any())
        self.assertFalse(excluded_interval_mask(df, None).any())
        self.assertFalse(
            excluded_interval_mask(df, {"anomaly_exclusions": {"events": []}}).any()
        )

    def test_whole_day_exclusion_when_he_range_omitted(self):
        df = _july_day_frame("2026-07-15")
        cfg = {
            "anomaly_exclusions": {"enabled": True, "events": [{"date": "2026-07-15"}]}
        }
        self.assertTrue(excluded_interval_mask(df, cfg).all())

    def test_missing_dt_column_is_safe(self):
        df = pd.DataFrame({"Actual_MWH": [1, 2, 3]})
        mask = excluded_interval_mask(df, DER_CONFIG)
        self.assertEqual(len(mask), 3)
        self.assertFalse(mask.any())

    def test_empty_frame_is_safe(self):
        self.assertTrue(drop_excluded_intervals(pd.DataFrame(), DER_CONFIG).empty)

    def test_tz_naive_dt_supported(self):
        df = pd.DataFrame(
            {"DT": pd.date_range("2026-07-15 16:00", periods=6, freq="h")}
        )  # 16..21
        mask = excluded_interval_mask(df, DER_CONFIG)
        self.assertEqual(sorted(df.loc[mask, "DT"].dt.hour.tolist()), [18, 19, 20, 21])

    def test_mixed_offset_export_timestamps_are_treated_as_local_wall_clock(self):
        df = pd.DataFrame(
            {
                "DT": [
                    "2026-07-15 17:00:00-07:00",
                    "2026-07-15 20:00:00-07:00",
                    "2026-01-15 17:00:00-08:00",
                ]
            }
        )
        mask = excluded_interval_mask(df, DER_CONFIG)
        self.assertEqual(mask.tolist(), [False, True, False])


class ProductionConfigExclusionTests(unittest.TestCase):
    def test_production_config_excludes_july15_der(self):
        cfg = load_config()
        events = (cfg.get("anomaly_exclusions", {}) or {}).get("events", []) or []
        july15 = [e for e in events if str(e.get("date")) == "2026-07-15"]
        self.assertTrue(july15, "expected a July 15 DER exclusion event in config.yaml")
        self.assertEqual(int(july15[0]["he_start"]), 18)
        self.assertEqual(int(july15[0]["he_end"]), 21)

    def test_production_config_drops_july15_peak_hours(self):
        cfg = load_config()
        df = _july_day_frame("2026-07-15")
        remaining = drop_excluded_intervals(df, cfg)["DT"].dt.hour.tolist()
        for hour_label in (18, 19, 20, 21):
            self.assertNotIn(hour_label, remaining)
        self.assertIn(17, remaining)
        self.assertIn(16, remaining)
        self.assertIn(15, remaining)

    def test_production_readiness_scorecard_drops_july15_der_hours(self):
        cfg = load_config()
        dt = pd.date_range(
            "2026-07-15 00:00", periods=24, freq="h", tz="America/Los_Angeles"
        )
        frame = pd.DataFrame(
            {
                "DT": dt,
                "Actual_MWH": 100.0,
                "Final_Backtest_Forecast_MWH": 100.0,
                "Forecast_Day": 1,
                "Hour": dt.hour,
                "Temperature_DailyMax": 95.0,
                "CloudCover_Norm": 0.0,
                "BTM_Solar_Loss_From_ClearSky_MW": 0.0,
                "Season": "Summer",
            }
        )

        scorecard = build_production_readiness_scorecard(frame, frame, config=cfg)
        recent_n = int(scorecard.loc[scorecard["Test"].eq("Last 45 days"), "N"].iloc[0])
        replay_n = int(
            scorecard.loc[scorecard["Test"].eq("Seasonal rolling origins"), "N"].iloc[0]
        )

        self.assertEqual(recent_n, 20)
        self.assertEqual(replay_n, 20)


class JulyEveningPeakGuardTests(unittest.TestCase):
    GUARD_CONFIG = {
        "calibration": {
            "stage_selector": {
                "focused_scorecard_guard": {
                    "enabled": True,
                    "total_cap_mwh": 30.0,
                    "rules": [
                        {
                            "name": "july_recent_100_plus_clear_hot_evening_lift",
                            "adjustment_mwh": 8.0,
                            "allow_without_forecast_day": True,
                            "months": [7],
                            "hours": [17, 18, 19, 20],
                            "min_maxtemp_f": 100.0,
                            "max_cloud_cover_norm": 0.25,
                            "holiday": False,
                        }
                    ],
                }
            }
        }
    }
    POST_PEAK_BACKOFF_CONFIG = {
        "calibration": {
            "stage_selector": {
                "focused_scorecard_guard": {
                    "enabled": True,
                    "total_cap_mwh": 30.0,
                    "rules": [
                        {
                            "name": "july_recent_95_100_clear_post_peak_decay_backoff",
                            "adjustment_mwh": -3.0,
                            "allow_without_forecast_day": True,
                            "months": [7],
                            "hours": [18, 19, 20, 21, 22, 23],
                            "min_maxtemp_f": 95.0,
                            "max_maxtemp_f": 100.0,
                            "max_cloud_cover_norm": 0.20,
                            "holiday": False,
                        }
                    ],
                }
            }
        }
    }

    def test_july_100plus_clear_evening_hours_get_lift(self):
        df = pd.DataFrame(
            {
                "DT": pd.to_datetime(
                    [
                        "2026-07-14 16:00",
                        "2026-07-14 17:00",
                        "2026-07-14 18:00",
                        "2026-07-14 19:00",
                        "2026-07-14 20:00",
                    ]
                ),
                "Final_Forecast_MWH": [314.0, 318.0, 320.0, 311.0, 299.0],
                "Stage_Selected_Forecast_MWH": [314.0, 318.0, 320.0, 311.0, 299.0],
                "Temperature_DailyMax": [103.8] * 5,
                "CloudCover_Norm": [0.0] * 5,
                "IsHoliday": [0] * 5,
            }
        )
        out = apply_focused_scorecard_guard(
            df, self.GUARD_CONFIG, forecast_col="Final_Forecast_MWH"
        )
        guard = out["Focused_Scorecard_Guard_MWH"].tolist()
        # Hour 16 is covered by the sibling pre-peak ramp rule, not this evening lift.
        self.assertEqual(guard[0], 0.0)
        self.assertEqual(guard[1:], [8.0, 8.0, 8.0, 8.0])
        self.assertEqual(out.loc[1, "Final_Forecast_MWH"], 326.0)
        self.assertTrue(
            out.loc[1:, "Focused_Scorecard_Guard_Source"]
            .str.contains("july_recent_100_plus_clear_hot_evening_lift")
            .all()
        )

    def test_july_cloudy_sub100_afternoon_is_not_lifted(self):
        # July 13-style: a cloudy sub-100F afternoon over-forecast must NOT be lifted.
        df = pd.DataFrame(
            {
                "DT": pd.to_datetime(
                    ["2026-07-13 17:00", "2026-07-13 18:00", "2026-07-13 19:00"]
                ),
                "Final_Forecast_MWH": [277.0, 270.0, 266.0],
                "Stage_Selected_Forecast_MWH": [277.0, 270.0, 266.0],
                "Temperature_DailyMax": [99.1, 99.1, 99.1],
                "CloudCover_Norm": [1.0, 0.98, 0.98],
                "IsHoliday": [0, 0, 0],
            }
        )
        out = apply_focused_scorecard_guard(
            df, self.GUARD_CONFIG, forecast_col="Final_Forecast_MWH"
        )
        self.assertTrue((out["Focused_Scorecard_Guard_MWH"] == 0.0).all())

    def test_july_clear_95_100_post_peak_decay_backoff_applies_narrowly(self):
        df = pd.DataFrame(
            {
                "DT": pd.to_datetime(
                    [
                        "2026-07-27 17:00",
                        "2026-07-27 18:00",
                        "2026-07-27 19:00",
                        "2026-07-27 20:00",
                        "2026-07-27 21:00",
                        "2026-07-27 22:00",
                        "2026-07-27 23:00",
                        "2026-07-27 18:00",
                        "2026-07-27 19:00",
                    ]
                ),
                "Final_Forecast_MWH": [260.0] * 9,
                "Stage_Selected_Forecast_MWH": [260.0] * 9,
                "Temperature_DailyMax": [
                    96.8,
                    96.8,
                    96.8,
                    96.8,
                    96.8,
                    96.8,
                    96.8,
                    96.8,
                    100.0,
                ],
                "CloudCover_Norm": [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.25, 0.0],
                "IsHoliday": [0] * 9,
            }
        )
        out = apply_focused_scorecard_guard(
            df,
            self.POST_PEAK_BACKOFF_CONFIG,
            forecast_col="Final_Forecast_MWH",
        )

        self.assertEqual(
            out["Focused_Scorecard_Guard_MWH"].tolist(),
            [0.0, -3.0, -3.0, -3.0, -3.0, -3.0, -3.0, 0.0, 0.0],
        )
        self.assertEqual(out.loc[1, "Final_Forecast_MWH"], 257.0)
        self.assertEqual(out.loc[7, "Final_Forecast_MWH"], 260.0)
        self.assertEqual(out.loc[8, "Final_Forecast_MWH"], 260.0)
        self.assertTrue(
            out.loc[1:6, "Focused_Scorecard_Guard_Source"]
            .str.contains("july_recent_95_100_clear_post_peak_decay_backoff")
            .all()
        )
        self.assertEqual(out.loc[0, "Focused_Scorecard_Guard_Source"], "none")
        self.assertEqual(out.loc[7, "Focused_Scorecard_Guard_Source"], "none")
        self.assertEqual(out.loc[8, "Focused_Scorecard_Guard_Source"], "none")

    def test_production_config_defines_july_evening_lift_rule(self):
        cfg = load_config()
        rules = cfg["calibration"]["stage_selector"]["focused_scorecard_guard"]["rules"]
        rule = next(
            (
                r
                for r in rules
                if r.get("name") == "july_recent_100_plus_clear_hot_evening_lift"
            ),
            None,
        )
        self.assertIsNotNone(
            rule, "expected the July evening peak-lift guard rule in config.yaml"
        )
        self.assertEqual(rule["months"], [7])
        self.assertEqual(rule["hours"], [17, 18, 19, 20])
        self.assertGreaterEqual(float(rule["adjustment_mwh"]), 8.0)
        self.assertEqual(float(rule["min_maxtemp_f"]), 100.0)
        self.assertTrue(bool(rule.get("allow_without_forecast_day")))

    def test_production_config_defines_july_post_peak_decay_backoff_rule(self):
        cfg = load_config()
        rules = cfg["calibration"]["stage_selector"]["focused_scorecard_guard"]["rules"]
        rule = next(
            (
                r
                for r in rules
                if r.get("name") == "july_recent_95_100_clear_post_peak_decay_backoff"
            ),
            None,
        )
        self.assertIsNotNone(
            rule, "expected the July post-peak decay backoff guard rule in config.yaml"
        )
        self.assertEqual(rule["months"], [7])
        self.assertEqual(rule["hours"], [18, 19, 20, 21, 22, 23])
        self.assertEqual(float(rule["adjustment_mwh"]), -3.0)
        self.assertEqual(float(rule["min_maxtemp_f"]), 95.0)
        self.assertEqual(float(rule["max_maxtemp_f"]), 100.0)
        self.assertLessEqual(float(rule["max_cloud_cover_norm"]), 0.20)
        self.assertTrue(bool(rule.get("allow_without_forecast_day")))

    def test_production_config_stacks_post_peak_backoff_with_late_evening_lift(self):
        cfg = load_config()
        df = pd.DataFrame(
            {
                "DT": pd.to_datetime(
                    [
                        "2026-07-27 18:00",
                        "2026-07-27 19:00",
                        "2026-07-27 20:00",
                        "2026-07-27 21:00",
                        "2026-07-27 22:00",
                        "2026-07-27 23:00",
                    ]
                ),
                "Final_Forecast_MWH": [260.0] * 6,
                "Stage_Selected_Forecast_MWH": [260.0] * 6,
                "Temperature_DailyMax": [96.8] * 6,
                "CloudCover_Norm": [0.0] * 6,
                "IsHoliday": [0] * 6,
                "IsWeekend": [0] * 6,
            }
        )

        out = apply_focused_scorecard_guard(df, cfg, forecast_col="Final_Forecast_MWH")

        self.assertEqual(
            out["Focused_Scorecard_Guard_MWH"].tolist(),
            [-3.0, -3.0, -3.0, -3.0, 0.0, 0.0],
        )
        self.assertEqual(
            out["Final_Forecast_MWH"].tolist(),
            [257.0, 257.0, 257.0, 257.0, 260.0, 260.0],
        )
        self.assertTrue(
            out.loc[0:3, "Focused_Scorecard_Guard_Source"]
            .str.contains("july_recent_95_100_clear_post_peak_decay_backoff")
            .all()
        )
        self.assertTrue(
            out.loc[4:5, "Focused_Scorecard_Guard_Source"]
            .str.contains(
                "july_recent_95_100_clear_post_peak_decay_backoff\\+july_clear_warm_late_evening_lift"
            )
            .all()
        )


if __name__ == "__main__":
    unittest.main()
