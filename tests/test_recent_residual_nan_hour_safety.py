from __future__ import annotations

"""Regression tests for a real bug found via scripts/ablate_correction_stages.py against
production replay data: with calibration.cloud_solar_shape_enabled: false, the pipeline
crashed with `ValueError: cannot convert float NaN to integer` inside
_weighted_recent_correction.

Root cause: apply_cloud_solar_shape_correction (event_shape_corrections.py's _prep()) backfills
NaN Hour values from DT as a side effect of running, and every later correction stage in the
chain inherits that repaired column. When cloud_solar_shape_enabled is false, that repair never
happens, so a row with a legitimately NaN Hour (present in real weather-realism scenario data)
flows all the way to `hour = int(row.get("Hour", pd.to_datetime(row.get("DT")).hour))` --
row.get()'s default only fires when the key is *missing*, not when its value is NaN, so
int(nan) raises. Fixed at two levels: apply_recent_residual_correction now backfills Hour the
same NaN-aware way _prep() does before entering its per-row loop, and the four duplicated
`hour = int(row.get("Hour", ...))` sites (_recent_hot_peak_scale, _ar_residual_correction,
_origin_day_residual_correction, _weighted_recent_correction) are hardened directly, since two
of them (_ar_residual_correction, _origin_day_residual_correction) are also reachable from
simulate_recent_residual_correction_backtest's fallback loop, a second entry point the
apply_recent_residual_correction-level fix alone would not have covered.
"""

import unittest

import numpy as np
import pandas as pd

from forecasting.forecast.recent_residual_correction import (
    _ar_residual_correction,
    _origin_day_residual_correction,
    _recent_hot_peak_scale,
    _weighted_recent_correction,
    apply_recent_residual_correction,
)


def _row(hour, dt: str) -> pd.Series:
    return pd.Series({"Hour": hour, "DT": pd.Timestamp(dt), "DailyMaxTempBucket": np.nan})


class WeightedRecentCorrectionNanHourTests(unittest.TestCase):
    def test_nan_hour_falls_back_to_dt_derived_hour_instead_of_crashing(self):
        profile = {"enabled": True, "global_mean": 5.0}
        # DT is 14:00 -- should behave identically to an explicit Hour=14 row.
        row_nan_hour = _row(np.nan, "2026-07-15 14:00:00")
        row_explicit_hour = _row(14, "2026-07-15 14:00:00")

        result_nan = _weighted_recent_correction(row_nan_hour, profile, {}, horizon_index=1)
        result_explicit = _weighted_recent_correction(
            row_explicit_hour, profile, {}, horizon_index=1
        )
        self.assertEqual(result_nan, result_explicit)

    def test_valid_hour_value_is_unaffected(self):
        profile = {"enabled": True, "same_hour_mean": {9: 3.0}}
        row = _row(9, "2026-07-15 09:00:00")
        corr, source, *_ = _weighted_recent_correction(row, profile, {}, horizon_index=1)
        self.assertFalse(np.isnan(corr))
        self.assertNotEqual(source, "disabled_or_empty")

    def test_missing_hour_key_still_falls_back_to_dt(self):
        # No "Hour" key at all (the pre-existing, already-working case) must keep working.
        row = pd.Series({"DT": pd.Timestamp("2026-07-15 11:00:00")})
        profile = {"enabled": True, "global_mean": 2.0}
        corr, *_ = _weighted_recent_correction(row, profile, {}, horizon_index=1)
        self.assertFalse(np.isnan(corr))


class SiblingHelperNanHourTests(unittest.TestCase):
    """The same fragile `int(row.get("Hour", ...))` pattern was duplicated in three other
    helpers reachable from a different entry point (simulate_recent_residual_correction_backtest's
    AR/origin-day fallback loop). Each must independently survive a NaN Hour with a valid DT."""

    def test_recent_hot_peak_scale_survives_nan_hour(self):
        row = _row(np.nan, "2026-07-15 17:00:00")
        try:
            _recent_hot_peak_scale(row, {})
        except ValueError as exc:
            self.fail(f"_recent_hot_peak_scale raised on NaN Hour: {exc}")

    def test_ar_residual_correction_survives_nan_hour(self):
        row = _row(np.nan, "2026-07-15 17:00:00")
        try:
            _ar_residual_correction(row, {}, {}, horizon_index=1)
        except ValueError as exc:
            self.fail(f"_ar_residual_correction raised on NaN Hour: {exc}")

    def test_origin_day_residual_correction_survives_nan_hour(self):
        row = _row(np.nan, "2026-07-15 17:00:00")
        try:
            _origin_day_residual_correction(row, {}, {})
        except ValueError as exc:
            self.fail(f"_origin_day_residual_correction raised on NaN Hour: {exc}")


class ApplyRecentResidualCorrectionIntegrationTests(unittest.TestCase):
    def test_reproduces_and_fixes_the_reported_crash(self):
        """Reproduces the exact reported shape: a future_df whose Hour column already exists
        (so the old `if "Hour" not in out.columns` guard did nothing) but has a NaN entry for
        one row -- the situation cloud_solar_shape_enabled: false left unrepaired upstream."""
        dt = pd.date_range("2026-07-15 00:00", periods=5, freq="h")
        future_df = pd.DataFrame(
            {
                "DT": dt,
                "Hour": [0.0, 1.0, np.nan, 3.0, 4.0],
                "Calibrated_Forecast_MWH": [100.0, 101.0, 102.0, 103.0, 104.0],
            }
        )
        profile = {"enabled": True, "global_mean": 1.5}
        try:
            out = apply_recent_residual_correction(future_df, profile, {})
        except ValueError as exc:
            self.fail(f"apply_recent_residual_correction raised on NaN Hour: {exc}")
        self.assertFalse(out["Hour"].isna().any())
        # DT for the NaN row is 02:00 -- confirm it was actually derived, not just dropped/zeroed.
        repaired_row = out.loc[out["DT"] == pd.Timestamp("2026-07-15 02:00:00")]
        self.assertEqual(int(repaired_row["Hour"].iloc[0]), 2)

    def test_existing_valid_hour_values_are_preserved(self):
        dt = pd.date_range("2026-07-15 00:00", periods=3, freq="h")
        future_df = pd.DataFrame(
            {
                "DT": dt,
                "Hour": [0.0, 1.0, 2.0],
                "Calibrated_Forecast_MWH": [100.0, 101.0, 102.0],
            }
        )
        out = apply_recent_residual_correction(future_df, {"enabled": True}, {})
        self.assertEqual(list(out["Hour"]), [0, 1, 2])


if __name__ == "__main__":
    unittest.main()
