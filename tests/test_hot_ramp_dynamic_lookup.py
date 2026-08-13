from __future__ import annotations

import unittest

import numpy as np
import pandas as pd

from forecasting.forecast.forecast_pipeline import rare_event_artifact_lookback_days
from forecasting.forecast.hot_ramp_peak_capture import (
    _persistence_scope_mask,
    _scope_mask,
    build_hot_ramp_peak_capture_artifact,
    build_heat_persistence_peak_capture_artifact,
)


def _single_row_frame(*, hour: int, daily_max: float, ramp: float) -> pd.DataFrame:
    dt = pd.Timestamp(f"2026-08-01 {hour:02d}:00", tz="America/Los_Angeles")
    return pd.DataFrame(
        {
            "DT": [dt],
            "Temperature_DailyMax": [daily_max],
            "DailyMaxTemp_Ramp_1Day": [ramp],
            "CloudCover_Norm": [0.1],
            "ConsecutiveExtremeHotDays100": [1.0],
        }
    )


class ScopeMaskTrainingOverrideTests(unittest.TestCase):
    """train_min_* config keys only widen the lookup's training pool; apply-time gating
    (training=False, the default used everywhere except _daily_training_frame) must be
    byte-identical to before this change."""

    def test_hot_ramp_training_override_admits_a_row_apply_time_would_reject(self):
        row = _single_row_frame(hour=17, daily_max=97.0, ramp=0.0)
        cfg = {
            "min_maxtemp_f": 100.0,
            "min_dailymax_ramp_1day_f": 2.0,
            "train_min_maxtemp_f": 95.0,
            "train_min_dailymax_ramp_1day_f": 0.0,
        }

        apply_mask, _ = _scope_mask(row, cfg, training=False)
        train_mask, _ = _scope_mask(row, cfg, training=True)

        self.assertFalse(bool(apply_mask.iloc[0]))
        self.assertTrue(bool(train_mask.iloc[0]))

    def test_hot_ramp_training_mode_is_a_noop_when_train_min_unset(self):
        row = _single_row_frame(hour=17, daily_max=97.0, ramp=0.0)
        cfg = {"min_maxtemp_f": 100.0, "min_dailymax_ramp_1day_f": 2.0}

        apply_mask, _ = _scope_mask(row, cfg, training=False)
        train_mask, _ = _scope_mask(row, cfg, training=True)

        self.assertFalse(bool(apply_mask.iloc[0]))
        self.assertFalse(bool(train_mask.iloc[0]))

    def test_heat_persistence_training_override_admits_a_row_apply_time_would_reject(
        self,
    ):
        row = _single_row_frame(hour=17, daily_max=97.0, ramp=0.0)
        row["ConsecutiveExtremeHotDays100"] = 1.0
        cfg = {
            "min_maxtemp_f": 100.0,
            "min_consecutive_extreme_days100": 3.0,
            "min_dailymax_3day_mean_f": 100.0,
            "train_min_maxtemp_f": 95.0,
            "train_min_consecutive_extreme_days100": 1.0,
            "train_min_dailymax_3day_mean_f": 95.0,
        }

        apply_mask, _ = _persistence_scope_mask(row, cfg, training=False)
        train_mask, _ = _persistence_scope_mask(row, cfg, training=True)

        self.assertFalse(bool(apply_mask.iloc[0]))
        self.assertTrue(bool(train_mask.iloc[0]))


def _sparse_hot_but_not_ramping_history(n_days: int = 20) -> pd.DataFrame:
    """Every day is hot (97F) but never ramps >=2F day-over-day (flat at 97F), so the
    apply-time-strict scope (min_maxtemp_f=100) never matches at all -- mirrors the
    diagnosed production failure mode where build_hot_ramp_peak_capture_artifact
    returns None for most origins."""
    rows = []
    for day in range(n_days):
        dt = pd.Timestamp("2026-07-01", tz="America/Los_Angeles") + pd.Timedelta(
            days=day
        )
        for hour in [15, 16, 17, 18, 19]:
            rows.append(
                {
                    "DT": dt + pd.Timedelta(hours=hour),
                    "Actual_MWH": 300.0 + hour,
                    "Final_Backtest_Forecast_MWH": 290.0 + hour,
                    "Temperature_DailyMax": 97.0,
                    "DailyMaxTemp_Ramp_1Day": 0.0,
                    "CloudCover_Norm": 0.1,
                }
            )
    return pd.DataFrame(rows)


class ArtifactStarvationRegressionTests(unittest.TestCase):
    """Regression test for the exact failure mode diagnosed in the 20260809_205652
    replay: a training window that is hot every day but never satisfies the joint
    min_maxtemp_f + min_dailymax_ramp_1day_f gate produces an empty daily_training_frame,
    so build_hot_ramp_peak_capture_artifact returns None and the whole origin falls back
    to the static targeted_missing_slices rules. train_min_* overrides should let the
    artifact build successfully from the same data.
    """

    def test_hot_ramp_artifact_is_none_without_training_overrides(self):
        history = _sparse_hot_but_not_ramping_history()
        cfg = {
            "hot_ramp_peak_capture": {
                "enabled": True,
                "min_maxtemp_f": 100.0,
                "min_dailymax_ramp_1day_f": 2.0,
                "min_training_days": 1,
                "min_lookup_days": 1,
                "hours": [16, 17, 18, 19, 20],
            }
        }

        artifact = build_hot_ramp_peak_capture_artifact(
            history, cfg, forecast_col="Final_Backtest_Forecast_MWH"
        )

        self.assertIsNone(artifact)

    def test_hot_ramp_artifact_builds_with_training_overrides(self):
        history = _sparse_hot_but_not_ramping_history()
        cfg = {
            "hot_ramp_peak_capture": {
                "enabled": True,
                "min_maxtemp_f": 100.0,
                "min_dailymax_ramp_1day_f": 2.0,
                "train_min_maxtemp_f": 95.0,
                "train_min_dailymax_ramp_1day_f": 0.0,
                "min_training_days": 1,
                "min_lookup_days": 1,
                "hours": [16, 17, 18, 19, 20],
            }
        }

        artifact = build_hot_ramp_peak_capture_artifact(
            history, cfg, forecast_col="Final_Backtest_Forecast_MWH"
        )

        self.assertIsNotNone(artifact)
        self.assertGreater(artifact["metadata"]["training_days"], 0)

    def test_heat_persistence_artifact_builds_with_training_overrides(self):
        history = _sparse_hot_but_not_ramping_history()
        for col_default, value in [("ConsecutiveExtremeHotDays100", 1.0)]:
            history[col_default] = value
        cfg = {
            "heat_persistence_peak_capture": {
                "enabled": True,
                "min_maxtemp_f": 100.0,
                "min_consecutive_extreme_days100": 3.0,
                "min_dailymax_3day_mean_f": 100.0,
                "max_dailymax_ramp_1day_f": 2.0,
                "train_min_maxtemp_f": 95.0,
                "train_min_consecutive_extreme_days100": 1.0,
                "train_min_dailymax_3day_mean_f": 95.0,
                "min_training_days": 1,
                "min_lookup_days": 1,
                "hours": [16, 17, 18, 19, 20],
            }
        }

        without_overrides = dict(cfg)
        without_overrides["heat_persistence_peak_capture"] = {
            k: v
            for k, v in cfg["heat_persistence_peak_capture"].items()
            if not k.startswith("train_min")
        }
        starved = build_heat_persistence_peak_capture_artifact(
            history, without_overrides, forecast_col="Final_Backtest_Forecast_MWH"
        )
        self.assertIsNone(starved)

        artifact = build_heat_persistence_peak_capture_artifact(
            history, cfg, forecast_col="Final_Backtest_Forecast_MWH"
        )
        self.assertIsNotNone(artifact)
        self.assertGreater(artifact["metadata"]["training_days"], 0)


class RareEventArtifactLookbackDaysTests(unittest.TestCase):
    def test_default_unset_returns_none(self):
        self.assertIsNone(rare_event_artifact_lookback_days({}, base_backtest_days=45))

    def test_value_not_greater_than_base_returns_none(self):
        config = {"calibration": {"rare_event_artifact_lookback_days": 45}}
        self.assertIsNone(
            rare_event_artifact_lookback_days(config, base_backtest_days=45)
        )

    def test_value_greater_than_base_is_returned(self):
        config = {"calibration": {"rare_event_artifact_lookback_days": 120}}
        self.assertEqual(
            rare_event_artifact_lookback_days(config, base_backtest_days=45), 120
        )

    def test_non_numeric_value_returns_none(self):
        config = {"calibration": {"rare_event_artifact_lookback_days": "oops"}}
        self.assertIsNone(
            rare_event_artifact_lookback_days(config, base_backtest_days=45)
        )

    def test_config_key_selects_an_independent_value(self):
        """calibration_search.py passes config_key="rare_event_artifact_lookback_days_search"
        so it can carry its own value independently of the live/replay default -- the two
        keys must not interfere with each other."""
        config = {
            "calibration": {
                "rare_event_artifact_lookback_days": None,
                "rare_event_artifact_lookback_days_search": 730,
            }
        }
        self.assertIsNone(
            rare_event_artifact_lookback_days(config, base_backtest_days=45)
        )
        self.assertEqual(
            rare_event_artifact_lookback_days(
                config,
                base_backtest_days=45,
                config_key="rare_event_artifact_lookback_days_search",
            ),
            730,
        )


if __name__ == "__main__":
    unittest.main()
