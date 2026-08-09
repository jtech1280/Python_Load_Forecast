from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

import pandas as pd

from forecasting.tuning.calibration_search import (
    RawOriginBundle,
    load_raw_origin_bundles,
    save_raw_origin_bundles,
    score_bundles,
)


def _minimal_bundle(origin_number: int = 1) -> RawOriginBundle:
    dt = pd.date_range("2026-06-01 00:00", periods=72, freq="h")
    base = pd.DataFrame(
        {
            "DT": dt,
            "Actual_MWH": 520.0,
            "Raw_Forecast_MWH": 500.0,
            "XGB_Pred_MWH": 500.0,
            "LGB_Pred_MWH": 500.0,
            "Hour": dt.hour.astype(float),
            "DOW": dt.dayofweek.astype(float),
            "Month": dt.month.astype(float),
            "Season": "Summer",
            "Temperature": 90.0,
            "Temperature_DailyMax": 95.0,
            "CloudCover_Norm": 0.1,
            "IsWeekend": 0,
            "IsHoliday": 0,
            "Forecast_Day": ((dt - dt[0]).days + 1).astype(float),
        }
    )
    return RawOriginBundle(
        origin_number=origin_number,
        origin_dt=dt[0],
        calibration_days=3,
        raw_calibration=base.copy(),
        raw_origin=base.copy(),
        raw_weather_realism=pd.DataFrame(),
        raw_realized_scenarios={},
        raw_weather_scenarios={},
    )


def _minimal_config(cap_mwh: float) -> dict:
    """A config with every optional correction stage disabled except the seasonal
    learned-calibration lookup, so score_bundles exercises real production code
    (build_correction_artifacts + apply_origin_correction_chain) without needing a
    fully realistic replay dataset."""
    return {
        "calibration": {
            "targeted_residual_meta": {"enabled": False},
            "seasonal_enabled": True,
            "cap_mwh": cap_mwh,
            "heat_peak_enabled": False,
            "warm_ramp_enabled": False,
            "cloud_solar_shape_enabled": False,
            "recent_residual": {"enabled": False},
            "stage_selector": {},
            "operational_residual_learner": {"enabled": False},
            "daily_peak_shadow_model": {"enabled": False},
            "hot_ramp_peak_capture": {"enabled": False},
            "heat_persistence_peak_capture": {"enabled": False},
            "weather_robustness_hedge": {"enabled": False},
        },
        "focused_shape_residual_learner": {"enabled": False},
        "operational_residual_learner": {"enabled": False},
        "daily_peak_shadow_model": {"enabled": False},
        "hot_ramp_peak_capture": {"enabled": False},
        "heat_persistence_peak_capture": {"enabled": False},
    }


class ScoreBundlesTests(unittest.TestCase):
    def test_score_bundles_stamps_calibration_metadata(self):
        bundle = _minimal_bundle()
        out = score_bundles([bundle], _minimal_config(cap_mwh=10.0))
        self.assertFalse(out.empty)
        self.assertTrue((out["Replay_Calibration_Days"] == 3).all())
        self.assertEqual(
            out["Replay_Calibration_Start_DT"].iloc[0],
            bundle.raw_calibration["DT"].min(),
        )
        self.assertEqual(
            out["Replay_Calibration_End_DT"].iloc[0], bundle.raw_calibration["DT"].max()
        )
        self.assertIn("Final_Backtest_Forecast_MWH", out.columns)

    def test_score_bundles_is_sensitive_to_calibration_cap(self):
        """The whole point of this module: re-scoring the same cached raw bundle under a
        different calibration.cap_mwh must actually change the corrected forecast, proving
        an Optuna trial that only varies calibration parameters gets a real signal without
        retraining XGB/LGB/CatBoost."""
        bundle = _minimal_bundle()
        low = score_bundles([bundle], _minimal_config(cap_mwh=0.5))
        high = score_bundles([bundle], _minimal_config(cap_mwh=22.0))
        low_mean = low["Final_Backtest_Forecast_MWH"].mean()
        high_mean = high["Final_Backtest_Forecast_MWH"].mean()
        # Bundle has a systematic +20 MWH Actual-vs-Raw residual; a higher cap should let
        # more of it through the learned-calibration correction.
        self.assertGreater(high_mean, low_mean)

    def test_score_bundles_empty_bundle_list_returns_empty_frame(self):
        out = score_bundles([], _minimal_config(cap_mwh=10.0))
        self.assertTrue(out.empty)


class RawOriginBundlePersistenceTests(unittest.TestCase):
    def test_save_and_load_round_trip(self):
        bundles = [_minimal_bundle(1), _minimal_bundle(2)]
        with tempfile.TemporaryDirectory() as tmp:
            cache_dir = Path(tmp) / "cache"
            paths = save_raw_origin_bundles(bundles, cache_dir)
            self.assertEqual(len(paths), 2)
            for path in paths:
                self.assertTrue(path.exists())

            loaded = load_raw_origin_bundles(cache_dir)
            self.assertEqual(len(loaded), 2)
            self.assertEqual([b.origin_number for b in loaded], [1, 2])
            pd.testing.assert_frame_equal(
                loaded[0].raw_calibration, bundles[0].raw_calibration
            )
            self.assertEqual(loaded[0].calibration_days, bundles[0].calibration_days)

    def test_load_missing_cache_dir_returns_empty_list(self):
        self.assertEqual(
            load_raw_origin_bundles("/nonexistent/path/does/not/exist"), []
        )


if __name__ == "__main__":
    unittest.main()
