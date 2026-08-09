import tempfile
import unittest
from pathlib import Path

import pandas as pd

from forecasting.solar.solar_forecaster import (
    load_rec_file_catalog,
    load_spid_file_lookup,
)


class SolarIndexCacheTests(unittest.TestCase):
    def _write_interval_parquet(
        self, root: Path, relative: str, spids: list[str]
    ) -> None:
        path = root / relative
        path.parent.mkdir(parents=True, exist_ok=True)
        pd.DataFrame(
            {
                "ServicePointID": spids,
                "ReadingValue_kWh": [1.0] * len(spids),
                "EndTimePST": ["2024-01-01 00:15"] * len(spids),
            }
        ).to_parquet(path, index=False)

    def test_missing_catalog_and_lookup_are_rebuilt_from_interval_tree(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            self._write_interval_parquet(
                root,
                "RES/REC/Monthly_202401/servicepoint_batch_0001.parquet",
                ["3000001_1", "3000001_1", "3000002_1"],
            )
            self._write_interval_parquet(
                root,
                "RES/NEM/NET/Monthly_202401/servicepoint_batch_0002.parquet",
                ["3000003_1", "3000004_1"],
            )
            self._write_interval_parquet(
                root,
                "COM/NEM/GS1/REC/Monthly_202402/servicepoint_batch_0003.parquet",
                ["4000001_1"],
            )

            catalog = load_rec_file_catalog(root, {"REC", "NET"})
            self.assertTrue((root / "_interval_parquet_index.csv").exists())
            self.assertTrue(
                (
                    root / "_shape_analysis_cache/spid_file_index/file_catalog.parquet"
                ).exists()
            )
            self.assertEqual(set(catalog["channel"]), {"REC", "NET"})
            self.assertEqual(set(catalog["nem_status"]), {"NEM", "Non-NEM"})
            self.assertIn("GS-1", set(catalog["rate_group"]))
            self.assertIn("RES", set(catalog["rate_group"]))

            lookup = load_spid_file_lookup(root)
            self.assertTrue(
                (root / "_shape_analysis_cache/spid_file_index/lookup").exists()
            )
            self.assertEqual(
                set(lookup.columns), {"SPID", "SPID_BASE", "FileID", "RowCount"}
            )
            row = lookup.loc[lookup["SPID"].eq("3000001_1")].iloc[0]
            self.assertEqual(row["SPID_BASE"], "3000001")
            self.assertEqual(int(row["RowCount"]), 2)

    def test_legacy_csv_index_is_normalized_and_aggregate_files_are_ignored(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            interval_path = (
                root / "RES/REC/Monthly_202401/servicepoint_batch_0001.parquet"
            )
            self._write_interval_parquet(
                root,
                "RES/REC/Monthly_202401/servicepoint_batch_0001.parquet",
                ["3000001_1", "3000002_1"],
            )
            aggregate_path = (
                root / "DEL_vs_System/AMI_REC_by_RateGroup_Observed.parquet"
            )
            aggregate_path.parent.mkdir(parents=True, exist_ok=True)
            pd.DataFrame(
                {"Name": ["GS-1"], "Date": ["2024-01-01"], "Value": [1.0]}
            ).to_parquet(
                aggregate_path,
                index=False,
            )
            pd.DataFrame(
                [
                    {
                        "filepath": str(interval_path),
                        "filename": interval_path.name,
                        "relative_path": str(interval_path.relative_to(root)),
                        "folder": str(interval_path.parent),
                        "segment": "RES",
                        "channel": "REC",
                        "rate_group": "RES",
                        "nem_status": "Non-NEM",
                        "month": 202401,
                        "size_bytes": interval_path.stat().st_size,
                    },
                    {
                        "filepath": str(aggregate_path),
                        "filename": aggregate_path.name,
                        "relative_path": str(aggregate_path.relative_to(root)),
                        "folder": str(aggregate_path.parent),
                        "segment": "DEL_vs_System",
                        "channel": "REC",
                        "rate_group": "",
                        "nem_status": "",
                        "month": 202401,
                        "size_bytes": aggregate_path.stat().st_size,
                    },
                ]
            ).to_csv(root / "_interval_parquet_index.csv", index=False)

            catalog = load_rec_file_catalog(root, {"REC"})
            self.assertEqual(len(catalog), 1)
            self.assertIn("modified_time_ns", catalog.columns)
            self.assertGreater(int(catalog["modified_time_ns"].iloc[0]), 0)
            self.assertEqual(catalog["filepath"].iloc[0], str(interval_path))

            lookup = load_spid_file_lookup(root)
            self.assertEqual(set(lookup["SPID"]), {"3000001_1", "3000002_1"})


if __name__ == "__main__":
    unittest.main()
