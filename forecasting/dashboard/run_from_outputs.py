from __future__ import annotations

import sys
from pathlib import Path

# Add project root to sys.path to support executing this script directly
project_root = str(Path(__file__).resolve().parents[2])
if project_root not in sys.path:
    sys.path.insert(0, project_root)

import os

from forecasting.main import _disable_windows_platform_wmi_probe

_disable_windows_platform_wmi_probe()

import pandas as pd

from forecasting.dashboard.dashboard_app import create_dashboard_app
from forecasting.config_utils import load_forecast_config
from forecasting.utils.output_archive import load_latest_distinct_snapshot


def _forecast_weather_frame_from_output(df: pd.DataFrame) -> pd.DataFrame:
    if df is None or df.empty or "DT" not in df.columns:
        return pd.DataFrame()
    out = pd.DataFrame({"DT": df["DT"]})
    mapping = {
        "Temperature": "TempF",
        "Humidity_Norm": "HumidityPct",
        "CloudCover_Norm": "CloudCoverPct",
        "WindSpeed_Mph": "WindSpeedMph",
        "WindDirection_Deg": "WindDirectionDeg",
        "PrecipIn": "PrecipIn",
        "Solar_Irradiance": "GHI_Wm2",
    }
    for src, dst in mapping.items():
        if src in df.columns:
            values = pd.to_numeric(df[src], errors="coerce")
            if src in {"Humidity_Norm", "CloudCover_Norm"}:
                values = values * 100.0
            out[dst] = values
    return out.dropna(subset=["DT"]).copy()


def _load_latest_sql_outputs(cfg: dict) -> dict:
    try:
        from forecasting.data.output_sql_store import (
            load_latest_run_outputs,
            output_sql_dashboard_read_enabled,
        )
    except Exception as exc:
        print(f"SQL output support is unavailable; using CSV outputs. Details: {exc}", flush=True)
        return {}

    if not output_sql_dashboard_read_enabled(cfg):
        return {}

    try:
        bundle = load_latest_run_outputs(cfg)
    except Exception as exc:
        print(f"Could not load latest SQL forecast outputs; using CSV outputs. Details: {exc}", flush=True)
        return {}

    forecast = bundle.get("forecast")
    backtest = bundle.get("backtest")
    if not isinstance(forecast, pd.DataFrame) or forecast.empty:
        print("SQL output loading is enabled but no forecast rows were found; using CSV outputs.", flush=True)
        return {}
    if not isinstance(backtest, pd.DataFrame) or backtest.empty:
        print("SQL output loading is enabled but no backtest rows were found; using CSV outputs.", flush=True)
        return {}

    print(f"Loaded latest forecast outputs from SQL Server RunID: {bundle.get('run_id')}", flush=True)
    return bundle


def _load_previous_sql_weather_snapshot(cfg: dict, current_weather: pd.DataFrame) -> pd.DataFrame:
    try:
        from forecasting.data.output_sql_store import (
            load_latest_archived_forecast_weather_snapshot,
            output_sql_enabled,
        )
    except Exception as exc:
        print(f"SQL weather archive support is unavailable; using CSV weather archive. Details: {exc}", flush=True)
        return pd.DataFrame()

    if not output_sql_enabled(cfg):
        return pd.DataFrame()

    try:
        return load_latest_archived_forecast_weather_snapshot(cfg, current_df=current_weather)
    except Exception as exc:
        print(f"Could not load SQL weather archive snapshot; using CSV weather archive. Details: {exc}", flush=True)
        return pd.DataFrame()


def main():
    here = Path(__file__).resolve().parents[1]  # forecasting/
    cfg = load_forecast_config()

    out_dir = Path(cfg.get("project", {}).get("output_dir", "forecast_outputs"))
    if not out_dir.is_absolute():
        out_dir = here.parent / out_dir

    forecast_csv = out_dir / "forecast_results.csv"
    backtest_csv = out_dir / "backtest_results.csv"

    sql_bundle = _load_latest_sql_outputs(cfg)
    if sql_bundle:
        fut = sql_bundle["forecast"]
        bt = sql_bundle["backtest"]
    else:
        if not forecast_csv.exists():
            raise SystemExit(f"Missing {forecast_csv}. Run the pipeline with --save-csv first.")
        if not backtest_csv.exists():
            raise SystemExit(f"Missing {backtest_csv}. Run the pipeline with --save-csv first.")
        fut = pd.read_csv(forecast_csv, low_memory=False)
        bt = pd.read_csv(backtest_csv, low_memory=False)

    previous_forecast = load_latest_distinct_snapshot(
        out_dir / "forecast_runs",
        current_df=fut,
        hash_columns=[
            "DT",
            "Forecast",
            "Forecast_Low_MWH",
            "Forecast_Expected_MWH",
            "Forecast_High_MWH",
            "P10_Forecast_MWH",
            "P90_Forecast_MWH",
            "Production_Risk_Code",
            "Temperature",
            "Temperature_DailyMax",
            "CloudCover_Norm",
            "Solar_Irradiance",
            "WeatherScenario_Spread_MWH",
            "WeatherScenario_MaxAbsDelta_MWH",
            "WeatherScenario_Cap_Applied",
        ],
    )
    cache_dir = Path(str(cfg.get("openmeteo", {}).get("cache_dir") or "weather_cache"))
    if not cache_dir.is_absolute():
        cache_dir = here.parent / cache_dir
    current_weather_for_snapshot = _forecast_weather_frame_from_output(fut)
    previous_weather = _load_previous_sql_weather_snapshot(cfg, current_weather_for_snapshot)
    if previous_weather.empty:
        previous_weather = load_latest_distinct_snapshot(
            cache_dir / "forecast_weather_runs",
            current_df=current_weather_for_snapshot,
            hash_columns=["DT", "TempF", "HumidityPct", "CloudCoverPct", "WindSpeedMph", "WindDirectionDeg", "PrecipIn", "GHI_Wm2", "IsDay"],
        )
    diagnostics = {}
    current_mtime = max([p.stat().st_mtime for p in [forecast_csv, backtest_csv] if p.exists()] or [0.0])
    stale_scorecards = {}
    for name in [
        "forecast_weather_used",
        "production_readiness_scorecard",
        "band_coverage_summary",
        "daily_peak_miss_by_stage",
        "backtest_metrics_by_segment_by_stage",
        "delta_breeze_shape_metrics_by_stage",
        "top_100_underforecast_hours_by_stage",
        "stage_marginal_contributions",
    ]:
        path = out_dir / f"{name}.csv"
        if path.exists():
            diagnostics[name] = pd.read_csv(path, low_memory=False)
            if name.endswith("scorecard") or name == "production_readiness_scorecard":
                stale_scorecards[name] = path.stat().st_mtime < current_mtime
    if sql_bundle and isinstance(sql_bundle.get("weather"), pd.DataFrame) and not sql_bundle["weather"].empty:
        diagnostics["forecast_weather_used"] = sql_bundle["weather"]
    if sql_bundle and isinstance(sql_bundle.get("diagnostics"), dict):
        for name, frame in sql_bundle["diagnostics"].items():
            if isinstance(frame, pd.DataFrame) and not frame.empty:
                diagnostics[name] = frame
    diagnostics["_stale_scorecards"] = stale_scorecards

    app = create_dashboard_app(
        historical_fit_df=pd.DataFrame(),  # optional for baseline UI; comparable/sensitivity need full history
        future_results={
            "display": fut,
            "current_weather_snapshot": diagnostics.get("forecast_weather_used", pd.DataFrame()),
            "previous_forecast_snapshot": previous_forecast,
            "previous_weather_snapshot": previous_weather,
        },
        backtest_results=bt,
        config=cfg,
        diagnostics_results=diagnostics,
    )
    port = int(os.environ.get("DASH_PORT", "8051"))
    print(
        "Dashboard server starting from saved outputs at http://127.0.0.1:"
        f"{port}. This process will keep running until stopped.",
        flush=True,
    )
    app.run(host="0.0.0.0", port=port, debug=False)


if __name__ == "__main__":
    main()
