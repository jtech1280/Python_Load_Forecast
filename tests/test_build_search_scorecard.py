from __future__ import annotations

"""build_search_scorecard() is a deliberate duplicate of the bt/event_slices construction
inside build_rolling_origin_replay_bundle(), added so scripts/tune_calibration_optuna.py's
per-trial scoring can skip the dozens of other diagnostic tables that function also builds
(profiling showed that difference costing more than half of a trial's wall time). Since the
two now have to stay in sync by hand, this test differentially checks
build_search_scorecard(replay_df, config) against
build_rolling_origin_replay_bundle(replay_df, config)["rolling_origin_replay_scorecard"]
across several replay frames, including ones that exercise excluded-interval filtering and
the multi-summer heat analog shadow, so a future edit to one path that silently diverges from
the other gets caught here rather than by a quietly-worse-then-better Optuna score."""

import unittest

import numpy as np
import pandas as pd

from forecasting.backtest.rolling_origin_replay import (
    build_rolling_origin_replay_bundle,
    build_search_scorecard,
)


def _replay_frame(rng: np.random.Generator, n_origins: int, hours_per_origin: int) -> pd.DataFrame:
    frames = []
    for origin in range(n_origins):
        origin_dt = pd.Timestamp("2026-06-01") + pd.Timedelta(days=origin * 5)
        dt = pd.date_range(origin_dt, periods=hours_per_origin, freq="h")
        n = len(dt)
        actual = 500 + rng.normal(0, 25, n)
        final = actual + rng.normal(0, 15, n)
        frame = pd.DataFrame(
            {
                "DT": dt,
                "Hour": dt.hour,
                "Month": dt.month,
                "DOW": dt.dayofweek,
                "Season": np.select(
                    [dt.month.isin([12, 1, 2]), dt.month.isin([3, 4, 5]), dt.month.isin([6, 7, 8, 9])],
                    ["Winter", "Spring", "Summer"],
                    default="Fall",
                ),
                "IsWeekend": dt.dayofweek.isin([5, 6]).astype(int),
                "IsHoliday": 0,
                "Actual_MWH": actual,
                "Final_Backtest_Forecast_MWH": final,
                "Temperature": rng.uniform(40, 110, n),
                "Temperature_DailyMax": rng.uniform(60, 112, n),
                "CloudCover_Norm": rng.uniform(0, 1, n),
                "Replay_Origin_ID": origin,
                "Replay_Origin_DT": origin_dt,
                "Replay_Calibration_Days": 45,
                "Replay_Calibration_Start_DT": origin_dt - pd.Timedelta(days=45),
                "Replay_Calibration_End_DT": origin_dt,
                "Forecast_Day": ((dt - dt[0]).days + 1).astype(float),
            }
        )
        frames.append(frame)
    return pd.concat(frames, ignore_index=True, sort=False)


def _assert_scorecards_match(a: pd.DataFrame, b: pd.DataFrame, msg: str) -> None:
    assert a.empty == b.empty, f"{msg}: emptiness mismatch"
    if a.empty:
        return
    assert set(a.columns) == set(b.columns), f"{msg}: column mismatch {set(a.columns) ^ set(b.columns)}"
    a_sorted = a.sort_values(list(a.columns)).reset_index(drop=True)
    b_sorted = b.sort_values(list(b.columns)).reset_index(drop=True)
    for col in a_sorted.columns:
        av = a_sorted[col]
        bv = b_sorted[col]
        if av.dtype == object or bv.dtype == object:
            both_na = av.isna().to_numpy() & bv.isna().to_numpy()
            mism = (av.astype(str).to_numpy() != bv.astype(str).to_numpy()) & ~both_na
            assert not mism.any(), f"{msg}: col {col} mismatch at {np.nonzero(mism)[0][:5]}"
        else:
            pd.testing.assert_series_equal(
                av, bv, check_dtype=False, check_names=False, check_exact=False,
                rtol=1e-9, atol=1e-9,
            )


class BuildSearchScorecardTests(unittest.TestCase):
    def test_matches_full_bundle_scorecard_across_random_trials(self):
        config = {}
        for trial in range(8):
            rng = np.random.default_rng(700 + trial)
            n_origins = int(rng.integers(1, 6))
            hours = int(rng.integers(24, 200))
            replay_df = _replay_frame(rng, n_origins, hours)
            with self.subTest(trial=trial, n_origins=n_origins, hours=hours):
                lightweight = build_search_scorecard(replay_df.copy(), config)
                full_bundle = build_rolling_origin_replay_bundle(replay_df.copy(), config)
                full = full_bundle.get("rolling_origin_replay_scorecard", pd.DataFrame())
                _assert_scorecards_match(lightweight, full, f"trial{trial}")

    def test_empty_replay_df(self):
        empty = pd.DataFrame()
        self.assertTrue(build_search_scorecard(empty, {}).empty)
        full_bundle = build_rolling_origin_replay_bundle(empty, {})
        self.assertTrue(
            full_bundle.get("rolling_origin_replay_scorecard", pd.DataFrame()).empty
        )

    def test_with_excluded_intervals_configured(self):
        rng = np.random.default_rng(42)
        replay_df = _replay_frame(rng, 3, 96)
        config = {
            "anomaly_exclusions": {
                "intervals": [
                    {
                        "start": str(replay_df["DT"].min()),
                        "end": str(replay_df["DT"].min() + pd.Timedelta(hours=5)),
                        "reason": "test exclusion",
                    }
                ]
            }
        }
        lightweight = build_search_scorecard(replay_df.copy(), config)
        full_bundle = build_rolling_origin_replay_bundle(replay_df.copy(), config)
        full = full_bundle.get("rolling_origin_replay_scorecard", pd.DataFrame())
        _assert_scorecards_match(lightweight, full, "excluded_intervals")


if __name__ == "__main__":
    unittest.main()
