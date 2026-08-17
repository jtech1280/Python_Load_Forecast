import unittest

import numpy as np
import pandas as pd

from forecasting.diagnostics.training_weight_diagnostics import (
    build_training_weight_diagnostic,
)


class TrainingWeightDiagnosticTests(unittest.TestCase):
    def _synthetic_train_df(self) -> pd.DataFrame:
        rows = []
        # 100 cool rows (DailyMax 70F), non-hot, non-peak hours.
        for i in range(100):
            rows.append(
                {
                    "MWH": 100.0,
                    "Temperature_DailyMax": 70.0,
                    "Hour": 3,
                    "IsLikelySystemPeakHour": 0,
                }
            )
        # 5 hot-peak-scope rows (DailyMax 106F, hour 17 -> in hot_peak_hours).
        for i in range(5):
            rows.append(
                {
                    "MWH": 300.0,
                    "Temperature_DailyMax": 106.0,
                    "Hour": 17,
                    "IsLikelySystemPeakHour": 1,
                }
            )
        # 3 hot-but-not-peak-hour rows (DailyMax 106F, hour 3).
        for i in range(3):
            rows.append(
                {
                    "MWH": 150.0,
                    "Temperature_DailyMax": 106.0,
                    "Hour": 3,
                    "IsLikelySystemPeakHour": 0,
                }
            )
        return pd.DataFrame(rows)

    def test_buckets_isolate_sparse_hot_scope_rows(self):
        df = self._synthetic_train_df()
        report = build_training_weight_diagnostic(df, config=None)

        total_rows = len(df)

        below_75_all = report[
            (report["Temperature_DailyMax_Bucket"] == "<75")
            & (report["Scope"] == "AllHours")
        ].iloc[0]
        self.assertEqual(below_75_all["N_Rows"], 100)
        self.assertAlmostEqual(
            below_75_all["Pct_Of_Total_Rows"], 100.0 * 100 / total_rows
        )

        hot_105_all = report[
            (report["Temperature_DailyMax_Bucket"] == "105+")
            & (report["Scope"] == "AllHours")
        ].iloc[0]
        self.assertEqual(hot_105_all["N_Rows"], 8)

        hot_105_scope = report[
            (report["Temperature_DailyMax_Bucket"] == "105+")
            & (report["Scope"] == "HotPeakHours16to20")
        ].iloc[0]
        self.assertEqual(hot_105_scope["N_Rows"], 5)
        # The hot-peak-scope rows should carry more than 1x weight (peak/hot upweighting).
        self.assertGreater(hot_105_scope["Mean_Sample_Weight"], 1.0)

        # Sparse hot rows should be a tiny fraction of raw rows despite upweighting.
        self.assertLess(hot_105_all["Pct_Of_Total_Rows"], 10.0)

    def test_report_covers_every_bucket_and_scope_combination(self):
        df = self._synthetic_train_df()
        report = build_training_weight_diagnostic(df, config=None)
        # 9 temperature buckets x 2 scopes (AllHours, HotPeakHours16to20).
        self.assertEqual(len(report), 18)
        self.assertTrue((report["N_Rows"] >= 0).all())

    def test_respects_configured_sample_weight_thresholds(self):
        df = self._synthetic_train_df()
        config = {
            "model": {
                "sample_weight": {
                    "hot_day_min_f": 95.0,
                    "hot_peak_hours": [17],
                    "hot_peak_weight": 5.0,
                    "non_business_hot_peak_weight": 1.0,
                    "peak_q90_weight": 1.0,
                    "peak_q95_weight": 1.0,
                    "recency_end_weight": 1.0,
                }
            }
        }
        report = build_training_weight_diagnostic(df, config=config)
        hot_105_scope = report[
            (report["Temperature_DailyMax_Bucket"] == "105+")
            & (report["Scope"] == "HotPeakHours16to20")
        ].iloc[0]
        self.assertAlmostEqual(hot_105_scope["Mean_Sample_Weight"], 5.0, places=6)


if __name__ == "__main__":
    unittest.main()
