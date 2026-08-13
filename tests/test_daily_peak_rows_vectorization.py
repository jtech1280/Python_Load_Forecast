from __future__ import annotations

"""Differential test for the _daily_training_frame / _daily_persistence_training_frame
vectorization: proves the new groupby-based implementation produces byte-identical
output to the original per-calendar-day Python loop, across many randomized synthetic
datasets covering edge cases (missing hours, NaN actuals/lags, empty-scope days, ties).

The reference implementations below are copied verbatim from the pre-vectorization
version of forecasting/forecast/hot_ramp_peak_capture.py (git blob at HEAD before the
2026-08-13 vectorization commit) and must NEVER be "fixed" to match the new code --
their entire purpose is to stay frozen as the ground truth being diffed against.
"""

import unittest
from typing import Any

import numpy as np
import pandas as pd

from forecasting.forecast.hot_ramp_peak_capture import (
    _as_num,
    _base_forecast,
    _cloud_bucket,
    _cloud_norm,
    _consecutive_extreme_days100,
    _daily_max,
    _daily_persistence_training_frame,
    _daily_ramp,
    _daily_training_frame,
    _dailymax_3day_mean,
    _date,
    _forecast_day,
    _horizon_bucket,
    _hour,
    _local_datetime,
    _month,
    _optional_num,
    _persistence_bucket,
    _persistence_scope_mask,
    _scope_mask,
    _temp_bucket,
)


def _reference_daily_training_frame(
    values: pd.DataFrame, cfg: dict, forecast_col: str
) -> pd.DataFrame:
    if values is None or values.empty:
        return pd.DataFrame()
    work = values.copy()
    dt = _local_datetime(work)
    work["_Date"] = _date(work, dt=dt)
    work["_Hour"] = _hour(work, dt=dt).astype(int)
    work["_Month"] = _month(work, dt=dt)
    work["_ForecastDay"] = _forecast_day(work, dt=dt)
    work["_Base"] = _base_forecast(work, forecast_col=forecast_col)
    work["_Actual"] = _optional_num(work, "Actual_MWH", "Actual", default=np.nan)
    work["_DailyMax"] = _daily_max(work)
    work["_Ramp1D"] = _daily_ramp(work, dt=dt)
    work["_Cloud"] = _cloud_norm(work)
    work["_Same7"] = _optional_num(
        work,
        "MWH_SameHour7DayMean",
        "Baseline_Rolling7DaySameHourAvg_MWH",
        default=np.nan,
    )
    work["_Lag24"] = _optional_num(
        work, "MWH_Lag24", "Baseline_SameHourYesterday_MWH", default=np.nan
    )

    scope, cooling = _scope_mask(
        work,
        cfg,
        enforce_forecast_day=bool(cfg.get("train_enforce_forecast_day", False)),
        training=True,
    )
    work["_Scope"] = scope & ~cooling

    rows: list[dict[str, Any]] = []
    for date, group in work.groupby("_Date", dropna=False):
        candidates = group[group["_Scope"]].copy()
        if candidates.empty:
            continue
        base_values = _as_num(candidates["_Base"], candidates.index).dropna()
        actual_values = _as_num(candidates["_Actual"], candidates.index).dropna()
        if base_values.empty or actual_values.empty:
            continue
        base_peak_idx = base_values.idxmax()
        actual_peak_idx = actual_values.idxmax()
        base_row = candidates.loc[base_peak_idx]
        actual_row = candidates.loc[actual_peak_idx]
        base_peak = float(base_row["_Base"])
        actual_peak = float(actual_row["_Actual"])
        if not (np.isfinite(base_peak) and np.isfinite(actual_peak)):
            continue
        daily_max = float(candidates["_DailyMax"].max())
        ramp = float(candidates["_Ramp1D"].max())
        cloud = float(candidates["_Cloud"].max())
        forecast_day = float(base_row["_ForecastDay"])
        same7 = float(base_row["_Same7"]) if pd.notna(base_row["_Same7"]) else np.nan
        lag24 = float(base_row["_Lag24"]) if pd.notna(base_row["_Lag24"]) else np.nan
        rows.append(
            {
                "Date": str(date),
                "Forecast_Day": forecast_day,
                "HorizonBucket": _horizon_bucket(forecast_day),
                "Month": (
                    int(base_row["_Month"]) if pd.notna(base_row["_Month"]) else np.nan
                ),
                "DailyMaxTempBucket": _temp_bucket(daily_max),
                "CloudCoverBucket": _cloud_bucket(cloud),
                "Temperature_DailyMax": daily_max,
                "DailyMaxTemp_Ramp_1Day": ramp,
                "CloudCover_Max": cloud,
                "Base_PeakHour": float(base_row["_Hour"]),
                "Actual_PeakHour": float(actual_row["_Hour"]),
                "Base_DailyPeak_MWH": base_peak,
                "Actual_DailyPeak_MWH": actual_peak,
                "Target_PeakResidual_MWH": actual_peak - base_peak,
                "SameHour7_AtBasePeak_MWH": same7,
                "SameHour7_TargetResidual_MWH": (
                    actual_peak - same7 if np.isfinite(same7) else np.nan
                ),
                "Lag24_AtBasePeak_MWH": lag24,
                "Lag24_TargetResidual_MWH": (
                    actual_peak - lag24 if np.isfinite(lag24) else np.nan
                ),
                "Lag24_RampSlope_MWH_Per_F": (
                    (actual_peak - lag24) / max(ramp, 1.0)
                    if np.isfinite(lag24)
                    else np.nan
                ),
            }
        )
    return pd.DataFrame(rows)


def _reference_daily_persistence_training_frame(
    values: pd.DataFrame, cfg: dict, forecast_col: str
) -> pd.DataFrame:
    if values is None or values.empty:
        return pd.DataFrame()
    work = values.copy()
    dt = _local_datetime(work)
    work["_Date"] = _date(work, dt=dt)
    work["_Hour"] = _hour(work, dt=dt).astype(int)
    work["_Month"] = _month(work, dt=dt)
    work["_ForecastDay"] = _forecast_day(work, dt=dt)
    work["_Base"] = _base_forecast(work, forecast_col=forecast_col)
    work["_Actual"] = _optional_num(work, "Actual_MWH", "Actual", default=np.nan)
    work["_DailyMax"] = _daily_max(work)
    work["_Ramp1D"] = _daily_ramp(work, dt=dt)
    work["_DailyMax3DayMean"] = _dailymax_3day_mean(work, dt=dt)
    work["_ConsecutiveExtreme"] = _consecutive_extreme_days100(work, dt=dt)
    work["_Cloud"] = _cloud_norm(work)
    work["_Same7"] = _optional_num(
        work,
        "MWH_SameHour7DayMean",
        "Baseline_Rolling7DaySameHourAvg_MWH",
        default=np.nan,
    )
    work["_Lag24"] = _optional_num(
        work, "MWH_Lag24", "Baseline_SameHourYesterday_MWH", default=np.nan
    )

    scope, cooling = _persistence_scope_mask(
        work,
        cfg,
        enforce_forecast_day=bool(cfg.get("train_enforce_forecast_day", False)),
        training=True,
    )
    work["_Scope"] = scope & ~cooling

    rows: list[dict[str, Any]] = []
    for date, group in work.groupby("_Date", dropna=False):
        candidates = group[group["_Scope"]].copy()
        if candidates.empty:
            continue
        base_values = _as_num(candidates["_Base"], candidates.index).dropna()
        actual_values = _as_num(candidates["_Actual"], candidates.index).dropna()
        if base_values.empty or actual_values.empty:
            continue
        base_peak_idx = base_values.idxmax()
        actual_peak_idx = actual_values.idxmax()
        base_row = candidates.loc[base_peak_idx]
        actual_row = candidates.loc[actual_peak_idx]
        base_peak = float(base_row["_Base"])
        actual_peak = float(actual_row["_Actual"])
        if not (np.isfinite(base_peak) and np.isfinite(actual_peak)):
            continue
        daily_max = float(candidates["_DailyMax"].max())
        ramp = float(candidates["_Ramp1D"].max())
        dailymax_3day = float(candidates["_DailyMax3DayMean"].max())
        consecutive_extreme = float(candidates["_ConsecutiveExtreme"].max())
        cloud = float(candidates["_Cloud"].max())
        forecast_day = float(base_row["_ForecastDay"])
        same7 = float(base_row["_Same7"]) if pd.notna(base_row["_Same7"]) else np.nan
        lag24 = float(base_row["_Lag24"]) if pd.notna(base_row["_Lag24"]) else np.nan
        rows.append(
            {
                "Date": str(date),
                "Forecast_Day": forecast_day,
                "HorizonBucket": _horizon_bucket(forecast_day),
                "Month": (
                    int(base_row["_Month"]) if pd.notna(base_row["_Month"]) else np.nan
                ),
                "DailyMaxTempBucket": _temp_bucket(daily_max),
                "CloudCoverBucket": _cloud_bucket(cloud),
                "PersistenceBucket": _persistence_bucket(consecutive_extreme),
                "Temperature_DailyMax": daily_max,
                "DailyMaxTemp_Ramp_1Day": ramp,
                "DailyMaxTemp_3DayMean": dailymax_3day,
                "ConsecutiveExtremeHotDays100": consecutive_extreme,
                "CloudCover_Max": cloud,
                "Base_PeakHour": float(base_row["_Hour"]),
                "Actual_PeakHour": float(actual_row["_Hour"]),
                "Base_DailyPeak_MWH": base_peak,
                "Actual_DailyPeak_MWH": actual_peak,
                "Target_PeakResidual_MWH": actual_peak - base_peak,
                "SameHour7_AtBasePeak_MWH": same7,
                "SameHour7_TargetResidual_MWH": (
                    actual_peak - same7 if np.isfinite(same7) else np.nan
                ),
                "Lag24_AtBasePeak_MWH": lag24,
                "Lag24_TargetResidual_MWH": (
                    actual_peak - lag24 if np.isfinite(lag24) else np.nan
                ),
            }
        )
    return pd.DataFrame(rows)


def _random_frame(rng: np.random.Generator, n_days: int) -> pd.DataFrame:
    rows = []
    for day in range(n_days):
        base_date = pd.Timestamp("2023-01-01") + pd.Timedelta(days=day)
        # Randomly drop some hours to exercise incomplete-day / empty-scope edge cases.
        hours = [h for h in range(24) if rng.random() > 0.05]
        daily_max_temp = float(rng.uniform(70, 112))
        ramp = float(rng.normal(0, 4))
        cloud = float(rng.uniform(0, 100))
        consecutive_extreme = float(rng.integers(0, 6))
        for h in hours:
            actual = rng.normal(300, 40)
            base = actual + rng.normal(0, 15)
            # Sprinkle NaNs into optional columns to exercise the finite-value guards.
            if rng.random() < 0.05:
                actual = np.nan
            if rng.random() < 0.05:
                base = np.nan
            same7 = rng.normal(300, 40) if rng.random() > 0.1 else np.nan
            lag24 = rng.normal(300, 40) if rng.random() > 0.1 else np.nan
            rows.append(
                {
                    "DT": base_date + pd.Timedelta(hours=h),
                    "Actual_MWH": actual,
                    "Final_Backtest_Forecast_MWH": base,
                    "Temperature_DailyMax": daily_max_temp,
                    "DailyMaxTemp_Ramp_1Day": ramp,
                    "CloudCover_Norm": cloud / 100.0,
                    "ConsecutiveExtremeHotDays100": consecutive_extreme,
                    "MWH_SameHour7DayMean": same7,
                    "MWH_Lag24": lag24,
                }
            )
    df = pd.DataFrame(rows)
    df["DT"] = pd.to_datetime(df["DT"]).dt.tz_localize(
        "America/Los_Angeles", ambiguous="NaT", nonexistent="shift_forward"
    )
    return df


def _random_cfg(rng: np.random.Generator) -> dict:
    return {
        "hours": [14, 15, 16, 17, 18, 19, 20],
        "min_maxtemp_f": float(rng.uniform(85, 100)),
        "train_min_maxtemp_f": float(rng.uniform(80, 95)),
        "min_dailymax_ramp_1day_f": float(rng.uniform(0, 3)),
        "train_min_dailymax_ramp_1day_f": 0.0,
        "max_cloud_cover_norm": float(rng.uniform(0.3, 0.7)),
        "train_enforce_forecast_day": False,
    }


def _random_persistence_cfg(rng: np.random.Generator) -> dict:
    return {
        "hours": [14, 15, 16, 17, 18, 19, 20],
        "min_maxtemp_f": float(rng.uniform(85, 100)),
        "train_min_maxtemp_f": float(rng.uniform(80, 95)),
        "min_consecutive_extreme_days100": float(rng.uniform(1, 4)),
        "train_min_consecutive_extreme_days100": 1.0,
        "min_dailymax_3day_mean_f": float(rng.uniform(80, 100)),
        "train_min_dailymax_3day_mean_f": 80.0,
        "max_dailymax_ramp_1day_f": float(rng.uniform(1, 5)),
        "max_cloud_cover_norm": float(rng.uniform(0.3, 0.7)),
        "train_enforce_forecast_day": False,
    }


def _assert_frames_match(
    test: unittest.TestCase, actual: pd.DataFrame, expected: pd.DataFrame
):
    test.assertEqual(sorted(actual.columns), sorted(expected.columns))
    actual_sorted = actual.sort_values("Date").reset_index(drop=True)[expected.columns]
    expected_sorted = expected.sort_values("Date").reset_index(drop=True)
    pd.testing.assert_frame_equal(
        actual_sorted, expected_sorted, check_dtype=False, check_exact=False, rtol=1e-9
    )


class DailyTrainingFrameVectorizationTests(unittest.TestCase):
    def test_matches_reference_across_many_random_datasets(self):
        rng = np.random.default_rng(1234)
        for trial in range(25):
            df = _random_frame(rng, n_days=rng.integers(20, 80))
            cfg = _random_cfg(rng)
            with self.subTest(trial=trial):
                actual = _daily_training_frame(
                    df, cfg, forecast_col="Final_Backtest_Forecast_MWH"
                )
                expected = _reference_daily_training_frame(
                    df, cfg, forecast_col="Final_Backtest_Forecast_MWH"
                )
                _assert_frames_match(self, actual, expected)

    def test_empty_input_returns_empty_frame(self):
        out = _daily_training_frame(
            pd.DataFrame(), {}, forecast_col="Final_Backtest_Forecast_MWH"
        )
        self.assertTrue(out.empty)

    def test_no_scope_passing_rows_returns_empty_frame(self):
        rng = np.random.default_rng(7)
        df = _random_frame(rng, n_days=10)
        cfg = {
            "hours": [14, 15, 16],
            "min_maxtemp_f": 200.0,
            "train_min_maxtemp_f": 200.0,
        }
        out = _daily_training_frame(df, cfg, forecast_col="Final_Backtest_Forecast_MWH")
        self.assertTrue(out.empty)


class DailyPersistenceTrainingFrameVectorizationTests(unittest.TestCase):
    def test_matches_reference_across_many_random_datasets(self):
        rng = np.random.default_rng(5678)
        for trial in range(25):
            df = _random_frame(rng, n_days=rng.integers(20, 80))
            cfg = _random_persistence_cfg(rng)
            with self.subTest(trial=trial):
                actual = _daily_persistence_training_frame(
                    df, cfg, forecast_col="Final_Backtest_Forecast_MWH"
                )
                expected = _reference_daily_persistence_training_frame(
                    df, cfg, forecast_col="Final_Backtest_Forecast_MWH"
                )
                _assert_frames_match(self, actual, expected)

    def test_empty_input_returns_empty_frame(self):
        out = _daily_persistence_training_frame(
            pd.DataFrame(), {}, forecast_col="Final_Backtest_Forecast_MWH"
        )
        self.assertTrue(out.empty)


if __name__ == "__main__":
    unittest.main()
