from __future__ import annotations

import importlib
import sys
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import numpy as np
import pandas as pd
import yaml

SCRIPTS_DIR = Path(__file__).resolve().parents[1] / "scripts"
if str(SCRIPTS_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPTS_DIR))

ablate = importlib.import_module("ablate_correction_stages")

from forecasting.tuning.calibration_search import save_raw_origin_bundles

from forecasting.tuning.calibration_search import RawOriginBundle


def _synthetic_bundle(origin_number: int, hot: bool, rng: np.random.Generator) -> RawOriginBundle:
    """A 72-hour origin bundle with enough shape (varying hour/temp, non-constant
    actual/raw forecast) to exercise the Hot-peak-days (hour 16-20 & dailymax>=90)
    and Peak-window (hour 14-18) gate slices, and to give the correction chain real
    residual signal to act on rather than a degenerate constant series.
    """
    dt = pd.date_range(f"2026-06-{1 + origin_number:02d} 00:00", periods=72, freq="h")
    hour = dt.hour.astype(float)
    daily_max = 96.0 if hot else 78.0
    temp = daily_max - 10 * np.abs(hour - 16) / 16.0
    actual = 500 + 6 * np.clip(temp - 65, 0, None) + rng.normal(0, 3, len(dt))
    raw = actual - rng.normal(3, 2, len(dt))  # raw model mildly under-forecasts
    base = pd.DataFrame(
        {
            "DT": dt,
            "Actual_MWH": actual,
            "Raw_Forecast_MWH": raw,
            "XGB_Pred_MWH": raw,
            "LGB_Pred_MWH": raw,
            "Hour": hour,
            "DOW": dt.dayofweek.astype(float),
            "Month": dt.month.astype(float),
            "Season": "Summer",
            "Temperature": temp,
            "Temperature_DailyMax": daily_max,
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


def _config(**overrides) -> dict:
    base = {
        "calibration": {
            "targeted_residual_meta": {"enabled": False},
            "seasonal_enabled": True,
            "cap_mwh": 10.0,
            "heat_peak_enabled": False,
            "warm_ramp_enabled": False,
            "cloud_solar_shape_enabled": False,
            "recent_residual": {"enabled": False},
            "stage_selector": {},
            "operational_residual_learner": {"enabled": False},
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
    for path, value in overrides.items():
        cur = base
        keys = path.split(".")
        for key in keys[:-1]:
            cur = cur.setdefault(key, {})
        cur[keys[-1]] = value
    return base


class PathHelperTests(unittest.TestCase):
    def test_set_path_does_not_mutate_input(self):
        original = {"calibration": {"peak_risk": {"enabled": True}}}
        snapshot = {"calibration": {"peak_risk": {"enabled": True}}}
        ablate._set_path(original, "calibration.peak_risk.enabled", False)
        self.assertEqual(original, snapshot)

    def test_set_path_creates_missing_intermediate_dicts(self):
        out = ablate._set_path({}, "calibration.stage_selector.focused_scorecard_guard.enabled", False)
        self.assertEqual(
            ablate._get_path(out, "calibration.stage_selector.focused_scorecard_guard.enabled"),
            False,
        )

    def test_get_path_returns_default_when_missing(self):
        self.assertEqual(ablate._get_path({}, "calibration.peak_risk.enabled", "fallback"), "fallback")

    def test_every_declared_stage_path_is_reachable(self):
        # Every path in STAGES must resolve to something under `_config()`'s known top-level
        # keys once set -- a typo'd path would silently create a dead branch nothing reads.
        for name, path, _desc in ablate.STAGES:
            cfg = ablate._set_path(_config(), path, False)
            self.assertEqual(
                ablate._get_path(cfg, path), False, f"stage '{name}' path '{path}' did not round-trip"
            )


class BuildCachePipelineInputsTests(unittest.TestCase):
    def test_build_cache_reuses_cached_pipeline_inputs(self):
        config = {
            "calibration": {},
            "training": {"rolling_origin_replay": {"parallel": {"enabled": False}}},
        }
        train_df = pd.DataFrame(
            {
                "DT": pd.date_range("2026-07-01", periods=4, freq="h"),
                "Actual_MWH": [1.0, 2.0, 3.0, 4.0],
            }
        )
        with tempfile.TemporaryDirectory() as tmp:
            cache_dir = Path(tmp) / "cache"
            with (
                patch(
                    "forecasting.forecast.forecast_pipeline.run_pipeline",
                    return_value={"historical_fit_df": train_df, "features": ["Hour"]},
                ) as mock_pipeline,
                patch.object(
                    ablate,
                    "build_raw_origin_bundle_cache",
                    return_value=[cache_dir / "origin_001_20260701.pkl"],
                ) as mock_build,
            ):
                ablate.build_cache(config, cache_dir, origin_limit=None)
                ablate.build_cache(config, cache_dir, origin_limit=None)

        self.assertEqual(mock_pipeline.call_count, 1)
        self.assertEqual(mock_build.call_count, 2)


class RunAblationSmokeTests(unittest.TestCase):
    def setUp(self):
        rng = np.random.default_rng(13)
        self.bundles = [
            _synthetic_bundle(1, hot=True, rng=rng),
            _synthetic_bundle(2, hot=True, rng=rng),
            _synthetic_bundle(3, hot=False, rng=rng),
            _synthetic_bundle(4, hot=False, rng=rng),
        ]
        self.config = _config(**{"calibration.peak_risk.enabled": True})

    def test_gate_scorecard_produces_hot_peak_and_peak_window_rows(self):
        gates = ablate.gate_scorecard(self.bundles, self.config)
        self.assertIn(ablate.HOT_PEAK_TEST_NAME, gates["Test"].tolist())
        self.assertIn(ablate.PEAK_WINDOW_TEST_NAME, gates["Test"].tolist())
        hot_row = gates.set_index("Test").loc[ablate.HOT_PEAK_TEST_NAME]
        self.assertGreater(int(hot_row["N"]), 0)

    def test_run_ablation_single_stage_produces_expected_columns(self):
        results = ablate.run_ablation(self.bundles, self.config, {"seasonal_calibration"})
        self.assertFalse(results.empty)
        self.assertEqual(set(results["Stage"]), {"seasonal_calibration"})
        for col in ["Before_MAE_MWH", "After_MAE_MWH", "Delta_MAE_MWH", "Before_Pass", "After_Pass", "Error"]:
            self.assertIn(col, results.columns)
        self.assertTrue(results["Error"].isna().all())
        peak_window_row = results[results["Test"] == ablate.PEAK_WINDOW_TEST_NAME].iloc[0]
        self.assertTrue(np.isfinite(peak_window_row["Before_MAE_MWH"]))
        self.assertTrue(np.isfinite(peak_window_row["After_MAE_MWH"]))
        # seasonal_calibration was enabled=True in this test config -- disabling it should
        # actually move the peak-window MAE, not be a silent no-op.
        self.assertNotAlmostEqual(
            float(peak_window_row["Before_MAE_MWH"]), float(peak_window_row["After_MAE_MWH"]), places=6
        )

    def test_per_origin_summary_present_only_for_peak_gates(self):
        results = ablate.run_ablation(self.bundles, self.config, {"seasonal_calibration"})
        peak_rows = results[results["Test"].isin(ablate.PER_ORIGIN_GATES)]
        other_rows = results[~results["Test"].isin(ablate.PER_ORIGIN_GATES)]
        for col in ["N_Improved", "N_Worsened", "N_Unchanged"]:
            self.assertTrue(peak_rows[col].notna().all())
            self.assertTrue(other_rows[col].isna().all())

    def test_a_stage_that_crashes_when_disabled_is_captured_not_raised(self):
        """Regression test for a real bug this harness surfaced: disabling
        calibration.peak_risk.enabled crashes apply_operational_stage_selector
        downstream (it unconditionally reads Peak_Risk_Adjusted_Forecast_MWH with
        no fallback when peak_risk was skipped). The harness must record this as
        a per-stage error, not let one stage's crash lose every other stage's
        results or blow up the whole ablation run.
        """
        results = ablate.run_ablation(self.bundles, self.config, {"peak_risk_correction"})
        self.assertFalse(results.empty)
        self.assertTrue(results["Error"].notna().all())
        self.assertTrue((results["Error"].str.contains("Peak_Risk_Adjusted_Forecast_MWH")).all())
        self.assertTrue(results["After_MAE_MWH"].isna().all())
        peak_window_row = results[results["Test"] == ablate.PEAK_WINDOW_TEST_NAME].iloc[0]
        # The baseline (stage enabled) side must still be a real number -- only the
        # crashed "disabled" side should be NaN.
        self.assertTrue(np.isfinite(peak_window_row["Before_MAE_MWH"]))
        self.assertTrue(np.isnan(peak_window_row["After_MAE_MWH"]))

    def test_all_stages_run_end_to_end_without_error(self):
        results = ablate.run_ablation(self.bundles, self.config, None)
        self.assertEqual(set(results["Stage"]), {s[0] for s in ablate.STAGES})


class BaselineOnlyCliTests(unittest.TestCase):
    """--baseline-only exists to compare two caches (e.g. standard training vs a cache built
    with model.asymmetric_loss.enabled: true) under the same config, without ablating any
    correction stage -- confirms the CLI wiring produces the same numbers gate_scorecard()
    would, not a parallel/duplicated computation that could drift from it."""

    def test_baseline_only_writes_gate_scorecard_matching_gate_scorecard_function(self):
        rng = np.random.default_rng(3)
        bundles = [
            _synthetic_bundle(1, hot=True, rng=rng),
            _synthetic_bundle(2, hot=False, rng=rng),
        ]
        config = _config()

        with tempfile.TemporaryDirectory() as tmp:
            tmp_path = Path(tmp)
            cache_dir = tmp_path / "cache"
            save_raw_origin_bundles(bundles, cache_dir)
            config_path = tmp_path / "config.yaml"
            config_path.write_text(yaml.safe_dump(config), encoding="utf-8")
            output_csv = tmp_path / "baseline.csv"

            argv = [
                "ablate_correction_stages.py",
                "--config",
                str(config_path),
                "--cache-dir",
                str(cache_dir),
                "--baseline-only",
                "--output-csv",
                str(output_csv),
            ]
            with patch.object(sys, "argv", argv):
                exit_code = ablate.main()
            self.assertEqual(exit_code, 0)

            written = pd.read_csv(output_csv)
            expected = ablate.gate_scorecard(bundles, config)
            self.assertEqual(
                written[["Test", "N"]].to_dict("records"),
                expected[["Test", "N"]].to_dict("records"),
            )


if __name__ == "__main__":
    unittest.main()
