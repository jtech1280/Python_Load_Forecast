import argparse
import json
from pathlib import Path
import yaml


def load_config():
    here = Path(__file__).resolve().parent
    cfg_path = here / "config.yaml"
    with open(cfg_path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def main():
    parser = argparse.ArgumentParser(description="Roseville System Load Forecast V12.8 (targeted solar-loss refinement + risk bands + max CPU/GPU)")
    parser.add_argument("--run-dashboard", action="store_true", help="Launch Dash dashboard after forecast")
    parser.add_argument("--horizon-days", type=int, default=None, help="Override forecast horizon days (default from config)")
    parser.add_argument("--save-csv", action="store_true", help="Export forecast/backtest CSVs to output directory")
    parser.add_argument("--skip-diagnostics", action="store_true", help="Skip detailed tuning diagnostics export")
    parser.add_argument("--disable-prophet", action="store_true", help="Skip Prophet benchmark training/prediction even if enabled in config.yaml")
    parser.add_argument("--disable-catboost", action="store_true", help="Skip CatBoost benchmark training/prediction even if enabled in config.yaml")
    parser.add_argument("--disable-five-min-load", action="store_true", help="Disable PowerSupply 5-minute load features for A/B replay testing")
    parser.add_argument("--disable-local-weather", action="store_true", help="Disable local PowerSupply weather station diagnostics/calibration")
    parser.add_argument("--use-local-weather-calibration", action="store_true", help="Apply Berry Sub historical temperature bias calibration to Open-Meteo temperature")
    parser.add_argument("--blend-catboost", action="store_true", help="Allow CatBoost to affect production Raw_Forecast_MWH using config ensemble weight")
    parser.add_argument("--cpu-only", action="store_true", help="Disable GPU attempts and force CPU training")
    parser.add_argument("--use-gpu", action="store_true", help="Force-enable XGBoost GPU attempts even if config.yaml disables them")
    parser.add_argument("--use-lgb-gpu", action="store_true", help="Also attempt LightGBM GPU training; falls back to CPU if configured")
    parser.add_argument("--gpu-priority", action="store_true", help="Use all supported GPU model backends and train tree models sequentially to prioritize GPU work")
    parser.add_argument("--cpu-threads", type=int, default=None, help="CPU threads for OpenMP/BLAS/XGB/LGB. Use -1 for all visible logical cores.")
    parser.add_argument("--safe-performance", action="store_true", help="Disable parallel tree-model training; each model still uses configured CPU threads")
    parser.add_argument("--force-parallel-cpu-training", action="store_true", help="Allow XGB CPU and LGB CPU to train concurrently; can oversubscribe CPUs")
    parser.add_argument("--rolling-origin-replay", action="store_true", help="Run opt-in multi-origin production-horizon replay diagnostics")
    parser.add_argument("--replay-max-origins", type=int, default=None, help="Override rolling replay origin count")
    parser.add_argument("--replay-origin-step-days", type=int, default=None, help="Override days between rolling replay origins")
    parser.add_argument("--replay-origins-per-season", type=int, default=None, help="Override scorecard origins selected per season")
    parser.add_argument("--replay-fixed-origins", type=str, default=None, help="Comma-separated fixed rolling replay origin dates")
    parser.add_argument("--replay-fixed-origins-file", type=str, default=None, help="Text file containing one fixed rolling replay origin date per line")
    parser.add_argument("--train-start-date", type=str, default=None, help="Override training and historical weather start date, e.g. 2018-01-01")
    args = parser.parse_args()

    config = load_config()

    if args.disable_prophet:
        config.setdefault("model", {}).setdefault("prophet", {})["enabled"] = False
        config.setdefault("model", {}).setdefault("ensemble_weights", {})["prophet"] = 0.0

    if args.disable_catboost:
        config.setdefault("model", {}).setdefault("catboost", {})["enabled"] = False
        config.setdefault("model", {}).setdefault("ensemble_weights", {})["catboost"] = 0.0

    if args.disable_five_min_load:
        config.setdefault("five_min_load", {})["enabled"] = False

    if args.disable_local_weather:
        config.setdefault("local_weather", {})["enabled"] = False

    if args.use_local_weather_calibration:
        config.setdefault("local_weather", {})["enabled"] = True
        config.setdefault("local_weather", {}).setdefault("temperature_calibration", {})["enabled"] = True

    if args.blend_catboost:
        config.setdefault("model", {}).setdefault("catboost", {})["blend_into_production"] = True
        config.setdefault("model", {}).setdefault("ensemble_weights", {})["catboost"] = float(config.get("model", {}).get("ensemble_weights", {}).get("catboost", 0.20) or 0.20)

    if args.cpu_threads is not None:
        config.setdefault("hardware", {})["cpu_threads"] = int(args.cpu_threads)
        config.setdefault("model", {}).setdefault("xgb", {})["n_jobs"] = int(args.cpu_threads)
        config.setdefault("model", {}).setdefault("lgb", {})["n_jobs"] = int(args.cpu_threads)

    if args.safe_performance:
        config.setdefault("hardware", {})["performance_mode"] = "safe"
        config.setdefault("hardware", {})["parallel_tree_training"] = False

    if args.force_parallel_cpu_training:
        config.setdefault("hardware", {})["force_parallel_cpu_training"] = True
        config.setdefault("hardware", {})["parallel_tree_training"] = True

    if args.cpu_only:
        config.setdefault("hardware", {})["use_gpu"] = False
        config.setdefault("hardware", {})["use_lgb_gpu"] = False
        config.setdefault("hardware", {})["parallel_tree_training"] = False
        config.setdefault("model", {}).setdefault("xgb", {})["device"] = "cpu"
        config.setdefault("model", {}).setdefault("xgb", {})["tree_method"] = "hist"
        config.setdefault("model", {}).setdefault("lgb", {})["use_gpu"] = False
    elif args.use_gpu:
        config.setdefault("hardware", {})["use_gpu"] = True
        config.setdefault("model", {}).setdefault("xgb", {})["device"] = "cuda"
        config.setdefault("model", {}).setdefault("xgb", {})["tree_method"] = "hist"

    if args.use_lgb_gpu:
        config.setdefault("hardware", {})["use_lgb_gpu"] = True
        config.setdefault("model", {}).setdefault("lgb", {})["use_gpu"] = True

    if args.gpu_priority:
        config.setdefault("hardware", {})["performance_mode"] = "gpu_priority"
        config.setdefault("hardware", {})["use_gpu"] = True
        config.setdefault("hardware", {})["use_lgb_gpu"] = True
        config.setdefault("hardware", {})["parallel_tree_training"] = False
        config.setdefault("model", {}).setdefault("xgb", {})["device"] = "cuda"
        config.setdefault("model", {}).setdefault("xgb", {})["tree_method"] = "hist"
        config.setdefault("model", {}).setdefault("lgb", {})["use_gpu"] = True
        config.setdefault("model", {}).setdefault("catboost", {})["task_type"] = "GPU"

    if args.rolling_origin_replay:
        config.setdefault("training", {}).setdefault("rolling_origin_replay", {})["enabled"] = True
    if args.replay_max_origins is not None:
        config.setdefault("training", {}).setdefault("rolling_origin_replay", {})["max_origins"] = int(args.replay_max_origins)
    if args.replay_origin_step_days is not None:
        config.setdefault("training", {}).setdefault("rolling_origin_replay", {})["origin_step_days"] = int(args.replay_origin_step_days)
    if args.replay_origins_per_season is not None:
        config.setdefault("training", {}).setdefault("rolling_origin_replay", {})["origins_per_season"] = int(args.replay_origins_per_season)
    if args.replay_fixed_origins:
        config.setdefault("training", {}).setdefault("rolling_origin_replay", {})["fixed_origins"] = str(args.replay_fixed_origins)
    if args.replay_fixed_origins_file:
        config.setdefault("training", {}).setdefault("rolling_origin_replay", {})["fixed_origins_file"] = str(args.replay_fixed_origins_file)
    if args.train_start_date:
        # Keep load-history and Open-Meteo history aligned so older load rows are not dropped
        # later by missing weather features.
        config.setdefault("training", {})["train_start_date"] = str(args.train_start_date)
        config.setdefault("openmeteo", {})["historical_start"] = str(args.train_start_date)

    output_dir = Path(config["project"]["output_dir"])
    output_dir.mkdir(parents=True, exist_ok=True)

    # Apply thread env before importing NumPy / XGBoost / LightGBM-heavy modules.
    from forecasting.utils.performance import apply_runtime_thread_settings, write_runtime_performance_info

    perf_info = apply_runtime_thread_settings(config)
    write_runtime_performance_info(config, perf_info)
    print(
        "Runtime performance: "
        f"threads={perf_info.get('resolved_cpu_threads')}, "
        f"xgb_gpu={perf_info.get('xgb_gpu_requested')}, "
        f"lgb_gpu={perf_info.get('lgb_gpu_requested')}, "
        f"parallel_tree_training={perf_info.get('parallel_tree_training')}"
    )

    # Delayed imports keep runtime thread settings effective.
    from forecasting.forecast.forecast_pipeline import run_pipeline
    from forecasting.diagnostics import export_diagnostics_bundle
    from forecasting.diagnostics.forecast_diagnostics import build_production_readiness_scorecard
    from forecasting.model.xgb_model import write_xgb_training_info, get_last_xgb_training_info
    from forecasting.model.lgb_model import write_lgb_training_info, get_last_lgb_training_info
    from forecasting.model.catboost_model import write_catboost_training_info, get_last_catboost_training_info
    from forecasting.utils.output_archive import save_distinct_snapshot

    results = run_pipeline(config, override_horizon_days=args.horizon_days)

    replay_cfg = ((config.get("training", {}) or {}).get("rolling_origin_replay", {}) or {})
    if results is not None and bool(replay_cfg.get("enabled", False)):
        from forecasting.backtest.rolling_origin_replay import (
            build_rolling_origin_replay_bundle,
            run_rolling_origin_replay,
        )

        replay_df = run_rolling_origin_replay(
            train_df=results["historical_fit_df"],
            features=results.get("features", []),
            config=config,
        )
        results.setdefault("diagnostics", {}).update(build_rolling_origin_replay_bundle(replay_df, config))
        results.setdefault("diagnostics", {})["production_readiness_scorecard"] = build_production_readiness_scorecard(
            results.get("backtest"),
            replay_df,
        )
        print(
            "Rolling-origin replay: "
            f"origins={replay_df['Replay_Origin_ID'].nunique() if not replay_df.empty else 0}, "
            f"rows={len(replay_df)}"
        )

    if args.save_csv and results is not None:
        forecast_csv = output_dir / "forecast_results.csv"
        backtest_csv = output_dir / "backtest_results.csv"
        metrics_json = output_dir / "backtest_metrics.json"
        metrics_raw_json = output_dir / "backtest_metrics_raw.json"
        metrics_final_json = output_dir / "backtest_metrics_final.json"
        features_txt = output_dir / "model_features.txt"
        prophet_features_txt = output_dir / "prophet_regressor_features.txt"
        catboost_features_txt = output_dir / "catboost_feature_list.txt"

        results["future"]["display"].to_csv(forecast_csv, index=False)
        forecast_snapshot = save_distinct_snapshot(
            results["future"]["display"],
            archive_dir=output_dir / "forecast_runs",
            stem="forecast_results",
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
            metadata={"Source": "production_forecast"},
        )
        results["backtest"].to_csv(backtest_csv, index=False)
        diag_summary = (results.get("diagnostics", {}) or {}).get("diagnostics_summary", {}) or {}
        raw_metrics = diag_summary.get("raw_model", results.get("backtest_metrics", {}))
        final_metrics = diag_summary.get("final_corrected_model", {})
        # V12.8: backtest_metrics.json defaults to final corrected metrics to avoid judging
        # production performance by raw XGB/LGB. Raw metrics are preserved separately.
        metrics_json.write_text(json.dumps(final_metrics or results.get("backtest_metrics", {}), indent=2, default=str), encoding="utf-8")
        metrics_raw_json.write_text(json.dumps(raw_metrics, indent=2, default=str), encoding="utf-8")
        metrics_final_json.write_text(json.dumps(final_metrics, indent=2, default=str), encoding="utf-8")
        features_txt.write_text("\n".join(results.get("features", [])), encoding="utf-8")
        prophet_features_txt.write_text("\n".join(results.get("prophet_features", [])), encoding="utf-8")
        catboost_features_txt.write_text("\n".join(results.get("catboost_features", [])), encoding="utf-8")

        print(f"Saved forecast to: {forecast_csv}")
        if forecast_snapshot is not None:
            print(f"Archived distinct forecast output snapshot: {forecast_snapshot}")
        print(f"Saved backtest to: {backtest_csv}")
        print(f"Saved metrics to: {metrics_json}")
        print(f"Saved tree feature list to: {features_txt}")
        write_xgb_training_info(config)
        write_lgb_training_info(config)
        write_catboost_training_info(config)
        write_runtime_performance_info(config, perf_info)
        print(f"Saved Prophet regressor list to: {prophet_features_txt}")
        print(f"XGBoost backend: {get_last_xgb_training_info().get('selected_backend')}")
        print(f"LightGBM backend: {get_last_lgb_training_info().get('selected_backend')}")
        print(f"CatBoost backend: {get_last_catboost_training_info().get('selected_backend')}")

        diagnostics_enabled = bool(config.get("diagnostics", {}).get("enabled", True))
        if diagnostics_enabled and not args.skip_diagnostics:
            written = export_diagnostics_bundle(results.get("diagnostics", {}), output_dir)
            print(f"Saved diagnostics files: {len(written)}")
            print(f"Saved diagnostics manifest to: {written.get('diagnostics_manifest')}")

    if args.run_dashboard and results is not None:
        from forecasting.dashboard.app import create_dashboard_app
        app = create_dashboard_app(
            historical_fit_df=results["historical_fit_df"],
            future_results=results["future"],
            backtest_results=results["backtest"],
            config=config,
            diagnostics_results=results.get("diagnostics", {}),
        )
        app.run(host="0.0.0.0", port=8050, debug=False)


if __name__ == "__main__":
    main()
