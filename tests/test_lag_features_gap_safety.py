from __future__ import annotations

import random
import unittest

import numpy as np
import pandas as pd

from forecasting.features.lag_features import add_basic_lags

LAG_COLS = [
    "MWH_Lag1",
    "MWH_Lag2",
    "MWH_Lag3",
    "MWH_Lag24",
    "MWH_Lag48",
    "MWH_Lag72",
    "MWH_Lag168",
    "MWH_Rolling3",
    "MWH_Rolling6",
    "MWH_Rolling12",
    "MWH_Rolling24",
    "MWH_Rolling48",
    "MWH_Rolling168",
    "MWH_Rolling24Std",
    "MWH_SameHour7DayMean",
]


def _old_add_basic_lags(df: pd.DataFrame) -> pd.DataFrame:
    """Frozen verbatim copy of add_basic_lags as of commit 91256d8 (pre-gap-fix):
    plain positional .shift() with no reindex to a gapless hourly grid. Kept only
    as a differential-testing reference for the fix in this file's sibling tests.
    """
    out = df.copy().sort_values("DT")
    y = out["MWH"].astype(float)

    shifted = y.shift(1)
    for lag in [1, 2, 3, 24, 48, 72, 168]:
        out[f"MWH_Lag{lag}"] = y.shift(lag)

    for window in [3, 6, 12, 24, 48, 168]:
        min_periods = max(2, min(window, int(window * 0.5)))
        out[f"MWH_Rolling{window}"] = shifted.rolling(
            window=window, min_periods=min_periods
        ).mean()

    out["MWH_Rolling24Std"] = (
        shifted.rolling(window=24, min_periods=12).std().fillna(0.0)
    )
    same_hour_lags = [y.shift(24 * i) for i in range(1, 8)]
    out["MWH_SameHour7DayMean"] = pd.concat(same_hour_lags, axis=1).mean(axis=1)
    return out


def _gapless_frame(n_hours: int, seed: int) -> pd.DataFrame:
    rng = random.Random(seed)
    start = pd.Timestamp("2024-01-01", tz="America/Los_Angeles")
    dts = pd.date_range(start, periods=n_hours, freq="h")
    mwh = [100.0 + 20.0 * rng.random() for _ in range(n_hours)]
    return pd.DataFrame({"DT": dts, "MWH": mwh})


class GaplessDifferentialTests(unittest.TestCase):
    def test_matches_old_positional_implementation_on_gapless_data(self):
        for seed in range(8):
            with self.subTest(seed=seed):
                df = _gapless_frame(24 * 30, seed=seed)
                old = _old_add_basic_lags(df)
                new = add_basic_lags(df)
                for col in LAG_COLS:
                    a = old[col].to_numpy()
                    b = new[col].to_numpy()
                    np.testing.assert_allclose(a, b, equal_nan=True, atol=1e-9)


class GapCorrectnessTests(unittest.TestCase):
    """A single missing hour only ever needs to disturb lag/rolling lookups whose
    window spans it -- at most 168 hours (7 days, the deepest lookback here) after
    the gap. Once every row's full lookback window lies entirely past the gap, a
    *uniform* one-row shift preserves relative elapsed time between any two such
    rows, so plain positional .shift() happens to self-correct there too. The real
    difference between old and new is inside that 7-day window: old silently
    substitutes the wrong hour's value; new correctly drops just the missing
    lookback (partial same-hour mean / rolling min_periods) instead.
    """

    N_HOURS = 24 * 40
    GAP_AT = 24 * 5 + 10  # some hour in the middle of day 6

    def _frame_with_gap(self) -> pd.DataFrame:
        df = _gapless_frame(self.N_HOURS, seed=1)
        return df.drop(index=self.GAP_AT).reset_index(drop=True)

    def test_gap_does_not_misalign_lookups_once_past_the_deepest_window(self):
        gapped = self._frame_with_gap()
        full = _gapless_frame(self.N_HOURS, seed=1)

        new_gapped = add_basic_lags(gapped)
        new_full = add_basic_lags(full)

        # Merge on DT so row positions line up correctly even though `gapped` has
        # one fewer row than `full` from this point on.
        merged = new_full.merge(
            new_gapped, on="DT", suffixes=("_full", "_gapped"), how="inner"
        )
        # Rows more than 7 days (168h, the deepest lookback used here) after the
        # gap: no lag/rolling window for these rows can overlap the missing hour
        # anymore, so the fixed implementation must exactly reproduce the
        # gapless-run ground truth.
        cutoff = full["DT"].iloc[self.GAP_AT] + pd.Timedelta(hours=168)
        settled = merged[merged["DT"] > cutoff]
        self.assertGreater(len(settled), 24 * 10)
        for col in LAG_COLS:
            np.testing.assert_allclose(
                settled[f"{col}_full"].to_numpy(),
                settled[f"{col}_gapped"].to_numpy(),
                equal_nan=True,
                atol=1e-9,
                err_msg=col,
            )

    def test_old_implementation_silently_misaligns_within_the_disturbed_window(self):
        """Confirms the bug this fix addresses is real: within the 7 days right
        after the gap, the old positional-shift implementation silently disagrees
        with the gapless-run ground truth (using the wrong hour's value) rather
        than honestly reflecting the missing data point.
        """
        gapped = self._frame_with_gap()
        full = _gapless_frame(self.N_HOURS, seed=1)

        old_gapped = _old_add_basic_lags(gapped)
        new_full = add_basic_lags(full)

        merged = new_full.merge(
            old_gapped, on="DT", suffixes=("_full", "_old_gapped"), how="inner"
        )
        gap_time = full["DT"].iloc[self.GAP_AT]
        disturbed = merged[
            (merged["DT"] > gap_time)
            & (merged["DT"] <= gap_time + pd.Timedelta(hours=168))
        ]
        self.assertGreater(len(disturbed), 0)

        mismatched = ~np.isclose(
            disturbed["MWH_SameHour7DayMean_full"].to_numpy(),
            disturbed["MWH_SameHour7DayMean_old_gapped"].to_numpy(),
            equal_nan=True,
            atol=1e-9,
        )
        self.assertTrue(
            mismatched.any(),
            "expected the old positional-shift implementation to disagree with "
            "ground truth somewhere in the disturbed window -- if this fails, the "
            "synthetic gap scenario stopped reproducing the bug",
        )

    def test_duplicate_timestamp_does_not_crash(self):
        df = _gapless_frame(24 * 10, seed=2)
        dup_row = df.iloc[[5]].copy()
        with_dup = pd.concat([df, dup_row], ignore_index=True)
        result = add_basic_lags(with_dup)
        self.assertEqual(len(result), len(with_dup))
        self.assertFalse(result["MWH_SameHour7DayMean"].isna().all())


if __name__ == "__main__":
    unittest.main()
