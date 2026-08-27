from __future__ import annotations

import tempfile
import unittest
import warnings
from pathlib import Path

import pandas as pd

from forecasting.utils.output_archive import _read_manifest, save_distinct_snapshot


def _frame() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "DT": pd.date_range("2026-07-01", periods=3, freq="h"),
            "Final_Forecast_MWH": [500.0, 510.0, 520.0],
        }
    )


class SaveDistinctSnapshotTests(unittest.TestCase):
    def test_no_pandas_deprecation_warning_from_the_timestamp_call(self):
        """Regression test: pd.Timestamp.utcnow() is deprecated (Pandas4Warning) in
        favor of pd.Timestamp.now("UTC") -- this must not emit that warning."""
        with tempfile.TemporaryDirectory() as tmp:
            with warnings.catch_warnings():
                warnings.simplefilter("error")
                save_distinct_snapshot(_frame(), Path(tmp), "forecast_results")

    def test_manifest_created_at_is_parseable_and_tz_aware(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = save_distinct_snapshot(_frame(), Path(tmp), "forecast_results")
            self.assertIsNotNone(path)
            manifest = _read_manifest(Path(tmp))
            self.assertEqual(len(manifest), 1)
            parsed = pd.to_datetime(manifest["CreatedAtUTC"], errors="coerce", utc=True)
            self.assertTrue(parsed.notna().all())

    def test_identical_content_reuses_the_existing_snapshot(self):
        """Exercises the manifest-sort-by-CreatedAtUTC path on a second call -- if the
        CreatedAtUTC format ever became unparseable, this dedup lookup is what would
        break first."""
        with tempfile.TemporaryDirectory() as tmp:
            first = save_distinct_snapshot(_frame(), Path(tmp), "forecast_results")
            second = save_distinct_snapshot(_frame(), Path(tmp), "forecast_results")
            self.assertEqual(first, second)
            manifest = _read_manifest(Path(tmp))
            self.assertEqual(len(manifest), 1)


if __name__ == "__main__":
    unittest.main()
