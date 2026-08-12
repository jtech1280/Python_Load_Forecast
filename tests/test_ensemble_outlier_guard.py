from __future__ import annotations

import unittest
from unittest.mock import patch
from zoneinfo import ZoneInfo

import pandas as pd

from forecasting.data.weather_loader import (
    _apply_ensemble_outlier_guard,
    _fetch_ensemble_temperature_stats,
)


def _config(
    *, enabled: bool = True, min_deviation_f: float = 10.0, min_members: int = 5
) -> dict:
    return {
        "project": {"timezone": "America/Los_Angeles"},
        "openmeteo": {
            "latitude": 38.7521,
            "longitude": -121.2880,
            "hourly_vars": ["temperature_2m"],
            "timezone": "America/Los_Angeles",
            "temperature_unit": "fahrenheit",
            "wind_speed_unit": "mph",
            "precipitation_unit": "inch",
            "forecast_days": 16,
            "ssl_verify": True,
            "ssl_use_os_truststore": False,
            "ensemble_outlier_guard": {
                "enabled": enabled,
                "min_deviation_f": min_deviation_f,
                "min_members": min_members,
            },
        },
    }


def _ensemble_payload(hours: list[str], member_temps: list[list[float]]) -> dict:
    """member_temps[i] is the list of member values for hours[i]."""
    n_members = len(member_temps[0])
    hourly = {"time": hours}
    for m in range(n_members):
        hourly[f"temperature_2m_member{m:02d}"] = [row[m] for row in member_temps]
    return {"hourly": hourly}


class FetchEnsembleTemperatureStatsTests(unittest.TestCase):
    def test_parses_dynamic_member_columns_into_mean_std_count(self):
        config = _config()
        payload = _ensemble_payload(
            ["2026-08-19T00:00", "2026-08-19T01:00"],
            [[100.0, 102.0, 104.0], [90.0, 92.0, 94.0]],
        )
        with patch("forecasting.data.weather_loader._fetch_json", return_value=payload):
            out = _fetch_ensemble_temperature_stats(config)

        self.assertEqual(len(out), 2)
        self.assertAlmostEqual(out.loc[0, "TempF_Ensemble_Mean"], 102.0)
        self.assertEqual(out.loc[0, "TempF_Ensemble_Member_Count"], 3)
        self.assertAlmostEqual(out.loc[1, "TempF_Ensemble_Mean"], 92.0)
        self.assertIsInstance(out["DT"].dtype, pd.DatetimeTZDtype)

    def test_empty_payload_returns_empty_frame(self):
        config = _config()
        with patch(
            "forecasting.data.weather_loader._fetch_json", return_value={"hourly": {}}
        ):
            out = _fetch_ensemble_temperature_stats(config)
        self.assertTrue(out.empty)


class ApplyEnsembleOutlierGuardTests(unittest.TestCase):
    def _point_frame(self, temps: list[float]) -> pd.DataFrame:
        tz = ZoneInfo("America/Los_Angeles")
        dt = pd.date_range("2026-08-19 00:00", periods=len(temps), freq="h", tz=tz)
        return pd.DataFrame({"DT": dt, "TempF": temps})

    def test_disabled_by_default_leaves_temps_unchanged(self):
        df = self._point_frame([79.0])
        config = _config(enabled=False)
        with patch(
            "forecasting.data.weather_loader._fetch_ensemble_temperature_stats"
        ) as mock_fetch:
            out = _apply_ensemble_outlier_guard(df, config)

        mock_fetch.assert_not_called()
        self.assertEqual(out.loc[0, "TempF"], 79.0)
        self.assertFalse(out.loc[0, "TempF_Outlier_Corrected"])

    def test_flyer_beyond_threshold_is_replaced_with_ensemble_mean(self):
        """Regression test for the diagnosed incident: a point forecast of 79F when the
        trustworthy ensemble mean is ~92F (deviation 13F > the 10F default threshold)
        must be corrected."""
        df = self._point_frame([79.0])
        config = _config(enabled=True, min_deviation_f=10.0, min_members=5)
        ensemble = pd.DataFrame(
            {
                "DT": df["DT"],
                "TempF_Ensemble_Mean": [92.0],
                "TempF_Ensemble_Std": [2.0],
                "TempF_Ensemble_Member_Count": [31],
            }
        )
        with patch(
            "forecasting.data.weather_loader._fetch_ensemble_temperature_stats",
            return_value=ensemble,
        ):
            out = _apply_ensemble_outlier_guard(df, config)

        self.assertAlmostEqual(out.loc[0, "TempF"], 92.0)
        self.assertTrue(out.loc[0, "TempF_Outlier_Corrected"])

    def test_small_deviation_under_threshold_is_left_alone(self):
        df = self._point_frame([90.0])
        config = _config(enabled=True, min_deviation_f=10.0, min_members=5)
        ensemble = pd.DataFrame(
            {
                "DT": df["DT"],
                "TempF_Ensemble_Mean": [92.0],
                "TempF_Ensemble_Std": [1.5],
                "TempF_Ensemble_Member_Count": [31],
            }
        )
        with patch(
            "forecasting.data.weather_loader._fetch_ensemble_temperature_stats",
            return_value=ensemble,
        ):
            out = _apply_ensemble_outlier_guard(df, config)

        self.assertAlmostEqual(out.loc[0, "TempF"], 90.0)
        self.assertFalse(out.loc[0, "TempF_Outlier_Corrected"])

    def test_untrustworthy_low_member_count_is_not_corrected(self):
        """A large deviation should still be ignored if too few ensemble members are
        present to trust the mean -- avoids overcorrecting off a degraded response."""
        df = self._point_frame([79.0])
        config = _config(enabled=True, min_deviation_f=10.0, min_members=5)
        ensemble = pd.DataFrame(
            {
                "DT": df["DT"],
                "TempF_Ensemble_Mean": [92.0],
                "TempF_Ensemble_Std": [2.0],
                "TempF_Ensemble_Member_Count": [2],
            }
        )
        with patch(
            "forecasting.data.weather_loader._fetch_ensemble_temperature_stats",
            return_value=ensemble,
        ):
            out = _apply_ensemble_outlier_guard(df, config)

        self.assertAlmostEqual(out.loc[0, "TempF"], 79.0)
        self.assertFalse(out.loc[0, "TempF_Outlier_Corrected"])

    def test_ensemble_fetch_failure_falls_back_to_point_forecast_unmodified(self):
        df = self._point_frame([79.0])
        config = _config(enabled=True)
        with patch(
            "forecasting.data.weather_loader._fetch_ensemble_temperature_stats",
            side_effect=RuntimeError("network error"),
        ):
            with self.assertWarns(RuntimeWarning):
                out = _apply_ensemble_outlier_guard(df, config)

        self.assertAlmostEqual(out.loc[0, "TempF"], 79.0)
        self.assertFalse(out.loc[0, "TempF_Outlier_Corrected"])


if __name__ == "__main__":
    unittest.main()
