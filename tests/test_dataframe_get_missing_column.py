from __future__ import annotations

import unittest

import pandas as pd

from forecasting.forecast.calibration import _prep_context
from forecasting.forecast.weather_scenarios import _recompute_solar_cloud


class DataFrameGetMissingColumnRegressionTests(unittest.TestCase):
    """Regression tests for `df.get("Col", <scalar>)` used where the caller expects a
    per-row Series. pandas' DataFrame.get returns the bare scalar default (not a Series)
    when the column is absent, so any `.fillna`/`.notna`/`.replace`/`.gt` chained onto it
    used to raise AttributeError instead of treating the missing column as "all default".
    Each test drops the specific optional column each fixed call site reads.
    """

    def test_prep_context_survives_missing_cloud_and_solar_columns(self):
        dt = pd.date_range("2026-07-01", periods=30, freq="h")
        df = pd.DataFrame({"DT": dt, "Temperature": 90.0})
        # No CloudCover_Norm, no BTM_Solar_Proxy_MW.
        out = _prep_context(df)
        self.assertIn("BTMSolarProxyBin", out.columns)
        self.assertIn("CloudCoverBin", out.columns)
        self.assertTrue((out["BTMSolarProxyBin"] == 0.0).all())

    def test_recompute_solar_cloud_survives_missing_midday_flag(self):
        dt = pd.date_range("2026-07-01", periods=10, freq="h")
        df = pd.DataFrame(
            {
                "DT": dt,
                "BTM_Solar_Loss_From_ClearSky_MW": 5.0,
                # No Solar_Midday_Flag, no CloudCover_Norm.
            }
        )
        out = _recompute_solar_cloud(df)
        self.assertIn("Midday_Overcast_Solar_Loss_MW", out.columns)
        self.assertTrue((out["Midday_Overcast_Solar_Loss_MW"] == 0.0).all())


if __name__ == "__main__":
    unittest.main()
