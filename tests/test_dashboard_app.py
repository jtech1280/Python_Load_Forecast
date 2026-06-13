import math
import unittest

import numpy as np
import pandas as pd

from forecasting.dashboard import app as legacy_dashboard_app
from forecasting.dashboard import dashboard_app
from forecasting.dashboard import run_from_outputs
from forecasting.dashboard.layout import make_layout


def _display_frame(hours=32):
    dt = pd.date_range("2026-01-01 00:00", periods=hours, freq="h")
    actual = [100.0 + i if i <= 5 else np.nan for i in range(hours)]
    forecast = [np.nan if i <= 5 else 110.0 + i for i in range(hours)]
    return pd.DataFrame({"DT": dt, "Actual": actual, "Forecast": forecast})


def _walk_components(component):
    yield component
    children = getattr(component, "children", None)
    if children is None:
        return
    if isinstance(children, (list, tuple)):
        for child in children:
            yield from _walk_components(child)
    else:
        yield from _walk_components(children)


def _component_ids(component):
    return [getattr(item, "id", None) for item in _walk_components(component)]


def _component_text(component):
    return [item for item in _walk_components(component) if isinstance(item, str)]


class DashboardAppHelperTests(unittest.TestCase):
    def test_fmt_num_formats_numeric_values_and_blanks_invalids(self):
        self.assertEqual(dashboard_app._fmt_num(1234.567, decimals=2), "1,234.57")
        self.assertEqual(dashboard_app._fmt_num(np.nan), "")
        self.assertEqual(dashboard_app._fmt_num("not-a-number"), "")

    def test_hour_ending_helpers_preserve_utility_he_convention(self):
        self.assertEqual(
            dashboard_app._format_hour_ending(pd.Timestamp("2026-01-01 17:00")),
            "01/01/2026 HE18",
        )
        self.assertEqual(
            dashboard_app._format_hour_ending(pd.Timestamp("2026-01-01 23:00")),
            "01/01/2026 HE24",
        )

    def test_dashboard_dt_parser_localizes_naive_and_converts_offset_timestamps(self):
        naive = dashboard_app._coerce_dashboard_dt(
            pd.Series(["2026-01-01 17:00:00"]),
            "America/Los_Angeles",
        )
        aware = dashboard_app._coerce_dashboard_dt(
            pd.Series(["2026-06-16 17:00:00-07:00", "2026-01-01 17:00:00-08:00"]),
            "America/Los_Angeles",
        )

        self.assertEqual(naive.iloc[0].hour, 17)
        self.assertEqual(str(naive.iloc[0]), "2026-01-01 17:00:00-08:00")
        self.assertEqual(aware.iloc[0].hour, 17)
        self.assertEqual(aware.iloc[1].hour, 17)

    def test_slice_display_anchors_window_on_latest_actual(self):
        display = _display_frame()

        sliced = dashboard_app._slice_display(display, horizon_days=1, history_hours=2)
        future = dashboard_app._future_only(display, horizon_days=1)

        self.assertEqual(sliced["DT"].iloc[0], pd.Timestamp("2026-01-01 03:00"))
        self.assertEqual(sliced["DT"].iloc[-1], pd.Timestamp("2026-01-02 05:00"))
        self.assertEqual(len(future), 24)
        self.assertTrue((future["DT"] > pd.Timestamp("2026-01-01 05:00")).all())

    def test_display_confidence_band_falls_back_to_upper_lower_and_caps_display_width(self):
        df = pd.DataFrame(
            {
                "Forecast": [100.0] * 5,
                "Upper_Band": [130.0] * 5,
                "Lower_Band": [70.0] * 5,
                "Operational_Horizon_Label": ["Day1"] * 5,
            }
        )

        upper, lower, band = dashboard_app._display_confidence_band(df)

        self.assertTrue(np.allclose(band.to_numpy(), [18.0] * 5))
        self.assertTrue(np.allclose(upper.to_numpy(), [118.0] * 5))
        self.assertTrue(np.allclose(lower.to_numpy(), [82.0] * 5))

    def test_forecast_series_controls_only_offer_present_numeric_series(self):
        df = pd.DataFrame(
            {
                "Forecast": [100.0],
                "Actual": [np.nan],
                "Raw_Forecast_MWH": [99.0],
                "Forecast_Change_From_Prior_Run_MWH": [np.nan],
            }
        )

        options, default = dashboard_app.forecast_series_controls(df)
        values = {option["value"] for option in options}

        self.assertIn("Forecast", values)
        self.assertIn("Raw", values)
        self.assertIn("band", values)
        self.assertNotIn("Actual", values)
        self.assertNotIn("ForecastChange", values)
        self.assertEqual(default, ["Forecast", "band"])

    def test_forecast_series_defaults_include_future_component_forecasts(self):
        df = pd.DataFrame(
            {
                "Forecast": [100.0],
                "Actual": [95.0],
                "Previous_Forecast_MWH": [94.0],
                "XGB_Pred_MWH": [99.0],
                "LGB_Pred_MWH": [101.0],
                "Prophet_Pred_MWH": [98.0],
                "Calibrated_Forecast_MWH": [100.5],
            }
        )

        options, default = dashboard_app.forecast_series_controls(df)
        labels = {option["value"]: option["label"] for option in options}

        self.assertIn("Calibrated", labels)
        self.assertEqual(labels["Calibrated"], "Calibrated Forecast")
        self.assertEqual(labels["Previous"], "Historical Forecast")
        self.assertEqual(default, ["Forecast", "Actual", "Previous", "XGB", "LGB", "Prophet", "Calibrated", "band"])

    def test_make_forecast_graph_respects_selected_series_and_handoff(self):
        dt = pd.date_range("2026-01-01 00:00", periods=4, freq="h")
        df = pd.DataFrame(
            {
                "DT": dt,
                "Actual": [100.0, 101.0, np.nan, np.nan],
                "Forecast": [np.nan, np.nan, 103.0, 104.0],
                "Upper_Band": [np.nan, np.nan, 108.0, 109.0],
                "Lower_Band": [np.nan, np.nan, 98.0, 99.0],
                "Raw_Forecast_MWH": [99.0, 100.0, 102.0, 103.0],
            }
        )

        fig = dashboard_app._make_forecast_graph(df, weather_variable="none", selected_series=["Actual", "Forecast"])
        names = [trace.name for trace in fig.data]

        self.assertEqual(names, ["Actual", "Published Forecast", "Actual-Forecast Handoff"])
        self.assertEqual(pd.Timestamp(fig.data[0].x[0]), pd.Timestamp("2026-01-01 01:00"))
        self.assertNotIn("Smoothed Forecast Band", names)
        self.assertNotIn("Raw XGB+LGB", names)

    def test_make_forecast_graph_adds_prior_run_weather_for_selected_weather(self):
        dt = pd.date_range("2026-01-01 00:00", periods=2, freq="h")
        df = pd.DataFrame(
            {
                "DT": dt,
                "Actual": [100.0, 101.0],
                "Forecast": [102.0, 103.0],
                "CloudCover_Norm": [0.40, 0.50],
                "Prior_Run_CloudCover_Norm": [0.35, 0.45],
            }
        )

        fig = dashboard_app._make_forecast_graph(
            df,
            weather_variable="CloudCover_Norm",
            selected_series=["Actual", "Forecast"],
        )
        names = [trace.name for trace in fig.data]

        self.assertIn("CloudCover_Norm", names)
        self.assertIn("Prior Run Cloud", names)
        prior_trace = next(trace for trace in fig.data if trace.name == "Prior Run Cloud")
        self.assertEqual(list(prior_trace.y), [0.35, 0.45])

    def test_make_forecast_graph_adds_actual_and_forecast_temperature_lines(self):
        dt = pd.date_range("2026-01-01 00:00", periods=4, freq="h")
        df = pd.DataFrame(
            {
                "DT": dt,
                "Actual": [100.0, 101.0, np.nan, np.nan],
                "Forecast": [np.nan, np.nan, 103.0, 104.0],
                "Temperature": [70.0, 72.0, 74.0, 76.0],
                "Forecast_Run_Temperature_F": [68.0, 71.0, 74.0, 76.0],
            }
        )

        fig = dashboard_app._make_forecast_graph(
            df,
            weather_variable="Temperature",
            selected_series=["Actual", "Forecast"],
        )
        names = [trace.name for trace in fig.data]

        self.assertIn("Actual Temperature", names)
        self.assertIn("Forecast Temperature", names)
        actual_temp = next(trace for trace in fig.data if trace.name == "Actual Temperature")
        forecast_temp = next(trace for trace in fig.data if trace.name == "Forecast Temperature")
        self.assertEqual(list(actual_temp.y[:2]), [70.0, 72.0])
        self.assertTrue(np.isnan(actual_temp.y[2]))
        self.assertEqual(list(forecast_temp.y), [68.0, 71.0, 74.0, 76.0])

    def test_attach_previous_forecast_history_uses_last_duplicate_and_residual_columns(self):
        dt = pd.date_range("2026-01-01 00:00", periods=2, freq="h")
        display = pd.DataFrame({"DT": dt, "Forecast": [110.0, 120.0]})
        backtest = pd.DataFrame(
            {
                "DT": [dt[0], dt[0], dt[1]],
                "Final_Backtest_Forecast_MWH": [90.0, 91.0, 119.0],
                "XGB_Pred_MWH": [89.0, 92.0, 118.0],
                "LGB_Pred_MWH": [88.0, 93.0, 117.0],
                "Prophet_Pred_MWH": [87.0, 94.0, 116.0],
                "Calibrated_Forecast_MWH": [86.0, 95.0, 115.0],
                "Final_Residual_MWH": [-1.0, 2.0, -3.0],
                "Final_AbsError_MWH": [1.0, 2.0, 3.0],
            }
        )

        out = dashboard_app._attach_previous_forecast_history(display, backtest)

        self.assertEqual(out.loc[0, "Previous_Forecast_MWH"], 91.0)
        self.assertEqual(out.loc[0, "Previous_Forecast_Miss_MWH"], 2.0)
        self.assertEqual(out.loc[1, "Previous_Forecast_AbsMiss_MWH"], 3.0)
        self.assertEqual(out.loc[0, "XGB_Pred_MWH"], 92.0)
        self.assertEqual(out.loc[0, "LGB_Pred_MWH"], 93.0)
        self.assertEqual(out.loc[0, "Prophet_Pred_MWH"], 94.0)
        self.assertEqual(out.loc[0, "Calibrated_Forecast_MWH"], 95.0)

    def test_attach_forecast_run_weather_adds_current_forecast_weather_columns(self):
        dt = pd.date_range("2026-01-01 00:00", periods=2, freq="h")
        display = pd.DataFrame({"DT": dt, "Actual": [100.0, 101.0], "Temperature": [70.0, 80.0]})
        forecast_weather = pd.DataFrame(
            {
                "DT": dt,
                "TempF": [69.0, 77.0],
                "HumidityPct": [45.0, 65.0],
                "CloudCoverPct": [20.0, 60.0],
            }
        )

        out = dashboard_app._attach_forecast_run_weather(display, forecast_weather)

        self.assertEqual(out["Forecast_Run_Temperature_F"].tolist(), [69.0, 77.0])
        self.assertEqual(out["Forecast_Run_Humidity_Norm"].tolist(), [0.45, 0.65])
        self.assertEqual(out["Forecast_Run_CloudCover_Norm"].tolist(), [0.20, 0.60])

    def test_attach_prior_run_comparison_adds_forecast_and_weather_deltas(self):
        dt = pd.date_range("2026-01-01 00:00", periods=2, freq="h")
        display = pd.DataFrame(
            {
                "DT": dt,
                "Forecast": [110.0, 120.0],
                "Temperature": [70.0, 80.0],
                "CloudCover_Norm": [0.40, 0.70],
                "Humidity_Norm": [0.50, 0.60],
                "Solar_Irradiance": [200.0, 500.0],
            }
        )
        previous_forecast = pd.DataFrame(
            {
                "DT": dt,
                "Forecast": [100.0, 125.0],
                "CloudCover_Norm": [0.20, 0.60],
                "Solar_Irradiance": [150.0, 450.0],
            }
        )
        previous_weather = pd.DataFrame({"DT": dt, "TempF": [69.0, 77.0], "HumidityPct": [45.0, 65.0]})

        out = dashboard_app._attach_prior_run_comparison(display, previous_forecast, previous_weather)

        self.assertEqual(out["Prior_Run_Forecast_MWH"].tolist(), [100.0, 125.0])
        self.assertEqual(out["Forecast_Change_From_Prior_Run_MWH"].tolist(), [10.0, -5.0])
        self.assertEqual(out["Prior_Run_Temperature_F"].tolist(), [69.0, 77.0])
        self.assertEqual(out["Temperature_Change_From_Prior_Run_F"].tolist(), [1.0, 3.0])
        self.assertEqual(out["Prior_Run_CloudCover_Norm"].tolist(), [0.20, 0.60])
        self.assertEqual(out["Prior_Run_Humidity_Norm"].tolist(), [0.45, 0.65])
        self.assertEqual(out["Prior_Run_Solar_Irradiance"].tolist(), [150.0, 450.0])

    def test_table_data_filters_to_forecast_rows_and_formats_operator_columns(self):
        dt = pd.date_range("2026-01-01 00:00", periods=2, freq="h")
        df = pd.DataFrame(
            {
                "DT": dt,
                "Forecast": [np.nan, 101.23],
                "Actual": [100.0, np.nan],
                "Raw_Forecast_MWH": [99.0, 100.8],
                "Forecast_Low_MWH": [np.nan, 95.0],
                "Forecast_Expected_MWH": [np.nan, 101.0],
                "Forecast_High_MWH": [np.nan, 107.0],
                "Production_Risk_Code": ["NORMAL", "CAUTION"],
                "Scenario_Cap": [False, True],
                "WeatherScenario_Cap_Applied": [False, True],
            }
        )

        rows, columns = dashboard_app._table_data(df)
        column_ids = [column["id"] for column in columns]

        self.assertEqual(len(rows), 1)
        self.assertIn("Raw XGB+LGB", column_ids)
        self.assertIn("Low", column_ids)
        self.assertEqual(rows[0]["DT"], "01/01/2026 HE02")
        self.assertEqual(rows[0]["Forecast"], "101.2")
        self.assertEqual(rows[0]["Raw XGB+LGB"], "100.8")
        self.assertEqual(rows[0]["Scenario Cap"], "Yes")

    def test_temp_sensitivity_table_data_defaults_to_baseline(self):
        frame = pd.DataFrame(
            {
                "DT": [pd.Timestamp("2026-01-01 14:00")],
                "baseline": [125.12],
                "plus_1": [127.34],
            }
        )

        rows, columns = dashboard_app._temp_sensitivity_table_data(frame, selected_steps=None)

        self.assertEqual([column["id"] for column in columns], ["Date", "Time", "Baseline"])
        self.assertEqual(rows[0], {"Date": "2026-01-01", "Time": "14:00", "Baseline": "125.1"})

    def test_residual_validation_prefers_final_residual_and_computes_statistics(self):
        residuals = pd.Series([1.0, 2.0, 3.0, 4.0])
        backtest = pd.DataFrame(
            {
                "Final_Residual_MWH": residuals,
                "Recent_Corrected_Residual_MWH": [-10.0, -10.0, -10.0, -10.0],
                "Residual_MWH": [10.0, 10.0, 10.0, 10.0],
            }
        )

        chosen = dashboard_app._residual_for_validation(backtest)

        self.assertEqual(chosen.tolist(), residuals.tolist())
        self.assertAlmostEqual(dashboard_app._acf_at_lag(chosen, 1), 1.0)
        self.assertAlmostEqual(dashboard_app._durbin_watson(pd.Series([1.0, 2.0, 3.0])), 2.0 / 14.0)
        self.assertTrue(math.isnan(dashboard_app._acf_at_lag(pd.Series([1.0]), 1)))

    def test_create_dashboard_app_constructs_layout_with_series_controls(self):
        dt = pd.date_range("2026-01-01 00:00", periods=4, freq="h")
        display = pd.DataFrame(
            {
                "DT": dt,
                "Actual": [100.0, 101.0, np.nan, np.nan],
                "Forecast": [np.nan, np.nan, 103.0, 104.0],
                "Upper_Band": [np.nan, np.nan, 108.0, 109.0],
                "Lower_Band": [np.nan, np.nan, 98.0, 99.0],
            }
        )

        app = dashboard_app.create_dashboard_app(
            historical_fit_df=pd.DataFrame(),
            future_results={"display": display},
            backtest_results=pd.DataFrame(),
            config={"project": {"timezone": "America/Los_Angeles"}},
        )

        self.assertIsNotNone(app.layout)
        self.assertGreaterEqual(len(app.callback_map), 4)
        ids = _component_ids(app.layout)
        callback_input_ids = {
            dep["id"]
            for meta in app.callback_map.values()
            for dep in meta.get("inputs", [])
        }

        self.assertTrue(any(isinstance(item, dict) and item.get("type") == "series-group" for item in ids))
        self.assertTrue(any("series-group" in str(item) for item in callback_input_ids))

    def test_layout_hides_series_controls_for_legacy_callers(self):
        layout = make_layout()

        ids = _component_ids(layout)
        text = _component_text(layout)

        self.assertFalse(any(isinstance(item, dict) and item.get("type") == "series-group" for item in ids))
        self.assertNotIn("No forecast series available.", text)

    def test_legacy_app_does_not_render_empty_series_message(self):
        dt = pd.date_range("2026-01-01 00:00", periods=2, freq="h")
        display = pd.DataFrame({"DT": dt, "Actual": [np.nan, np.nan], "Forecast": [100.0, 101.0]})

        app = legacy_dashboard_app.create_dashboard_app(
            historical_fit_df=pd.DataFrame(),
            future_results={"display": display},
            backtest_results=pd.DataFrame(),
            config={"project": {"timezone": "America/Los_Angeles"}},
        )

        self.assertNotIn("No forecast series available.", _component_text(app.layout))

    def test_run_from_outputs_uses_series_dashboard_app(self):
        self.assertIs(run_from_outputs.create_dashboard_app, dashboard_app.create_dashboard_app)

    def test_legacy_app_module_uses_series_dashboard_app(self):
        self.assertIs(legacy_dashboard_app.create_dashboard_app, dashboard_app.create_dashboard_app)


if __name__ == "__main__":
    unittest.main()
