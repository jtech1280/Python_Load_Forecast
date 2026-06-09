from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import yaml
from sklearn.ensemble import HistGradientBoostingRegressor

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))


def _load_config() -> dict[str, Any]:
    with open(Path("forecasting") / "config.yaml", "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def _as_num(frame: pd.DataFrame, col: str, default: float = np.nan) -> pd.Series:
    if col not in frame.columns:
        return pd.Series(default, index=frame.index, dtype=float)
    return pd.to_numeric(frame[col], errors="coerce")


def _latest_replay(output_dir: Path) -> Path:
    files = sorted(
        output_dir.glob("rolling_origin_replay_results_*.csv"),
        key=lambda p: p.stat().st_mtime,
        reverse=True,
    )
    for path in files:
        if "smoke" not in path.stem.lower() and "scenario_path_full" in path.stem.lower():
            return path
    if files:
        return files[0]
    raise FileNotFoundError("No rolling_origin_replay_results_*.csv files found.")


def _feature_frame(frame: pd.DataFrame) -> pd.DataFrame:
    dt = pd.to_datetime(frame["DT"], errors="coerce", utc=True)
    hour = _as_num(frame, "Hour").fillna(dt.dt.hour)
    day = _as_num(frame, "Forecast_Day").fillna(999.0)
    daily_max = _as_num(frame, "Temperature_DailyMax")
    cloud = _as_num(frame, "CloudCover_Norm")
    if cloud.notna().any() and cloud.max(skipna=True) > 1.5:
        cloud = cloud / 100.0
    solar_loss = _as_num(frame, "BTM_Solar_Loss_From_ClearSky_MW", 0.0).fillna(0.0)

    return pd.DataFrame(
        {
            "Selected": _as_num(frame, "Final_Backtest_Forecast_MWH"),
            "Raw": _as_num(frame, "Raw_Forecast_MWH"),
            "XGB": _as_num(frame, "XGB_Pred_MWH"),
            "LGB": _as_num(frame, "LGB_Pred_MWH"),
            "CatBoost": _as_num(frame, "CatBoost_Pred_MWH"),
            "Prophet": _as_num(frame, "Prophet_Pred_MWH"),
            "TargetedMeta": _as_num(frame, "Targeted_Meta_Adjusted_Forecast_MWH"),
            "PeakRisk": _as_num(frame, "Peak_Risk_Adjusted_Forecast_MWH"),
            "ResidualCalibrated": _as_num(frame, "Residual_Calibrated_Forecast_MWH"),
            "WarmRamp": _as_num(frame, "Warm_Ramp_Adjusted_Forecast_MWH"),
            "CloudSolar": _as_num(frame, "Cloud_Solar_Adjusted_Forecast_MWH"),
            "RecentCorrected": _as_num(frame, "Recent_Corrected_Forecast_MWH"),
            "SelectedMinusRaw": _as_num(frame, "Final_Backtest_Forecast_MWH") - _as_num(frame, "Raw_Forecast_MWH"),
            "SelectedMinusRecent": _as_num(frame, "Final_Backtest_Forecast_MWH") - _as_num(frame, "Recent_Corrected_Forecast_MWH"),
            "SelectedMinusPeak": _as_num(frame, "Final_Backtest_Forecast_MWH") - _as_num(frame, "Peak_Risk_Adjusted_Forecast_MWH"),
            "PeakMonthCorrection": _as_num(frame, "Long_Horizon_Peak_Month_Correction_MWH", 0.0).fillna(0.0),
            "HotMonthCorrection": _as_num(frame, "Long_Horizon_Hot_Month_Correction_MWH", 0.0).fillna(0.0),
            "WeatherHedge": _as_num(frame, "Weather_Robustness_Hedge_MWH", 0.0).fillna(0.0),
            "WeatherWarmerDelta": _as_num(frame, "Weather_Robustness_Warmer_Delta_MWH", 0.0).fillna(0.0),
            "Hour": hour,
            "HourSin": np.sin(2.0 * np.pi * hour / 24.0),
            "HourCos": np.cos(2.0 * np.pi * hour / 24.0),
            "ForecastDay": day,
            "Day1": day.eq(1).astype(float),
            "Days2to7": day.between(2, 7).astype(float),
            "Days8Plus": day.ge(8).astype(float),
            "Temperature": _as_num(frame, "Temperature"),
            "DailyMaxTemp": daily_max,
            "TempOver90": (daily_max - 90.0).clip(lower=0.0),
            "TempOver95": (daily_max - 95.0).clip(lower=0.0),
            "CloudCover": cloud,
            "SolarLoss": solar_loss,
            "Humidity": _as_num(frame, "Humidity_Norm"),
            "Weekend": _as_num(frame, "IsWeekend", 0.0).fillna(0.0),
            "Holiday": _as_num(frame, "IsHoliday", 0.0).fillna(0.0),
            "HotPeak": (hour.between(16, 20) & daily_max.ge(90.0)).astype(float),
            "PeakWindow": hour.between(14, 18).astype(float),
        },
        index=frame.index,
    )


def _metrics(frame: pd.DataFrame, forecast: pd.Series, prefix: str) -> dict[str, float]:
    actual = _as_num(frame, "Actual_MWH")
    hour = _as_num(frame, "Hour")
    temp = _as_num(frame, "Temperature_DailyMax")
    day = _as_num(frame, "Forecast_Day")
    cloud = _as_num(frame, "CloudCover_Norm")
    if cloud.notna().any() and cloud.max(skipna=True) > 1.5:
        cloud = cloud / 100.0
    loss = _as_num(frame, "BTM_Solar_Loss_From_ClearSky_MW")
    season = frame.get("Season", pd.Series("", index=frame.index)).astype(str)
    masks = {
        "overall": pd.Series(True, index=frame.index),
        "day1": day.eq(1),
        "days2to3": day.between(2, 3),
        "days4to7": day.between(4, 7),
        "hot_peak": hour.between(16, 20) & temp.ge(90.0),
        "cloud_solar_midday": hour.between(10, 16) & (cloud.ge(0.60) | loss.ge(1.25)),
        "shoulder_heat": season.isin(["Spring", "Fall"]) & hour.between(12, 22) & temp.between(75.0, 93.0),
        "peak_window_14_18": hour.between(14, 18),
    }
    out: dict[str, float] = {}
    for name, mask in masks.items():
        if not mask.any():
            continue
        residual = actual.loc[mask] - forecast.loc[mask]
        out[f"{prefix}_{name}_n"] = int(mask.sum())
        out[f"{prefix}_{name}_mae_mwh"] = float(residual.abs().mean())
        out[f"{prefix}_{name}_bias_mwh"] = float(residual.mean())
        out[f"{prefix}_{name}_p90_abs_error_mwh"] = float(residual.abs().quantile(0.90))
    out[f"{prefix}_scorecard_pass"] = bool(
        out.get(f"{prefix}_overall_mae_mwh", np.inf) <= 4.5
        and abs(out.get(f"{prefix}_overall_bias_mwh", np.inf)) <= 0.75
        and out.get(f"{prefix}_day1_mae_mwh", np.inf) <= 3.5
        and out.get(f"{prefix}_days2to3_mae_mwh", np.inf) <= 5.0
        and out.get(f"{prefix}_days4to7_mae_mwh", np.inf) <= 5.0
        and out.get(f"{prefix}_hot_peak_mae_mwh", np.inf) <= 6.0
        and out.get(f"{prefix}_cloud_solar_midday_mae_mwh", np.inf) <= 7.0
        and out.get(f"{prefix}_shoulder_heat_mae_mwh", np.inf) <= 7.0
        and out.get(f"{prefix}_peak_window_14_18_mae_mwh", np.inf) <= 5.5
    )
    return out


def _event_mask(replay: pd.DataFrame, scope: str, min_day: int, exclude_holidays: bool) -> pd.Series:
    hour = _as_num(replay, "Hour")
    temp = _as_num(replay, "Temperature_DailyMax")
    day = _as_num(replay, "Forecast_Day")
    hot = hour.between(16, 20) & temp.ge(90.0)
    peak = hour.between(14, 18)
    if scope == "hot_peak":
        eligible = hot
    elif scope == "peak_window":
        eligible = peak
    else:
        eligible = hot | peak
    eligible &= day.ge(int(min_day))
    if exclude_holidays:
        eligible &= ~_as_num(replay, "IsHoliday", 0.0).fillna(0.0).ne(0.0)
    return eligible.fillna(False)


def _evaluate_model_guard(
    replay: pd.DataFrame,
    blend: float,
    cap: float,
    loss: str,
    scope: str,
    min_day: int,
    max_iter: int,
    max_leaf_nodes: int,
    min_samples_leaf: int,
    l2_regularization: float,
) -> pd.Series:
    out = pd.Series(0.0, index=replay.index, dtype=float)
    eligible = _event_mask(replay, scope=scope, min_day=min_day, exclude_holidays=True)
    origins = sorted(pd.to_datetime(replay["Replay_Origin_DT"], errors="coerce", utc=True).dropna().unique())
    replay_origin = pd.to_datetime(replay["Replay_Origin_DT"], errors="coerce", utc=True)

    for origin in origins:
        train = replay[replay_origin.lt(origin) & eligible].copy()
        test = replay[replay_origin.eq(origin) & eligible].copy()
        if len(train) < 80 or test.empty:
            continue
        target = _as_num(train, "Actual_MWH") - _as_num(train, "Final_Backtest_Forecast_MWH")
        x_train = _feature_frame(train)
        valid = target.notna() & x_train["Selected"].notna()
        if int(valid.sum()) < 80:
            continue
        fill = x_train.median(numeric_only=True).fillna(0.0)
        model = HistGradientBoostingRegressor(
            loss=loss,
            learning_rate=0.03,
            max_iter=max_iter,
            max_leaf_nodes=max_leaf_nodes,
            min_samples_leaf=min_samples_leaf,
            l2_regularization=l2_regularization,
            random_state=42,
        )
        model.fit(x_train.loc[valid].fillna(fill).fillna(0.0), target.loc[valid])
        pred = pd.Series(model.predict(_feature_frame(test).fillna(fill).fillna(0.0)), index=test.index)
        out.loc[test.index] = (pred * float(blend)).clip(-float(cap), float(cap))
    return out


def _evaluate_scenario_midpoint_guard(
    replay: pd.DataFrame,
    blend: float,
    cap: float,
    scope: str,
    min_day: int,
    upper_blend: float,
    upper_cap: float,
    low_forecast_threshold: float,
) -> pd.Series:
    warmer = _as_num(replay, "WeatherScenario_warmer_P50_MWH")
    cooler = _as_num(replay, "WeatherScenario_cooler_P50_MWH")
    base = _as_num(replay, "Final_Backtest_Forecast_MWH")
    midpoint_delta = ((warmer + cooler) / 2.0 - base).fillna(0.0)
    eligible = _event_mask(replay, scope=scope, min_day=min_day, exclude_holidays=True)
    correction = (midpoint_delta * float(blend)).clip(-float(cap), float(cap)).where(eligible, 0.0)
    if upper_blend > 0:
        hour = _as_num(replay, "Hour")
        temp = _as_num(replay, "Temperature_DailyMax")
        upper_delta = (warmer - base).clip(lower=0.0).fillna(0.0)
        upper_mask = eligible & hour.between(16, 20) & temp.ge(90.0) & base.le(float(low_forecast_threshold))
        correction = correction + (
            upper_delta * float(upper_blend)
        ).clip(0.0, float(upper_cap)).where(upper_mask, 0.0)
    return correction


def main() -> None:
    parser = argparse.ArgumentParser(description="Evaluate a guarded peak-window residual model against saved replay origins.")
    parser.add_argument("--replay-path", default=None)
    parser.add_argument("--label", default=None)
    args = parser.parse_args()

    output_dir = Path("forecast_outputs")
    replay_path = Path(args.replay_path) if args.replay_path else _latest_replay(output_dir)
    label = args.label or replay_path.stem.removeprefix("rolling_origin_replay_results_")
    _load_config()  # Validate config readability; this evaluation uses saved replay columns as the baseline.
    replay = pd.read_csv(replay_path, low_memory=False)
    replay["Replay_Origin_DT"] = pd.to_datetime(replay["Replay_Origin_DT"], errors="coerce", utc=True)

    base_forecast = _as_num(replay, "Final_Backtest_Forecast_MWH")
    rows: list[dict[str, Any]] = []
    base = {
        "candidate": "current_saved_replay",
        "blend": 0.0,
        "cap_mwh": 0.0,
        "loss": "none",
        "scope": "none",
        "min_day": 0,
        "correction_rows": 0,
        "mean_nonzero_correction_mwh": 0.0,
    }
    base.update(_metrics(replay, base_forecast, "metric"))
    rows.append(base)

    model_shapes = [
        ("absolute_error", 80, 24, 4, 10.0),
        ("absolute_error", 120, 16, 6, 5.0),
        ("squared_error", 80, 24, 4, 10.0),
        ("squared_error", 120, 16, 6, 5.0),
    ]
    for loss, max_iter, min_samples_leaf, max_leaf_nodes, l2 in model_shapes:
        for scope in ["peak_hot", "hot_peak", "peak_window"]:
            for min_day in [1, 4, 8]:
                raw_correction = None
                for blend in [0.05, 0.10, 0.15, 0.25, 0.35, 0.50]:
                    for cap in [1.0, 2.0, 3.0, 4.0, 5.0]:
                        if raw_correction is None:
                            raw_correction = _evaluate_model_guard(
                                replay,
                                blend=1.0,
                                cap=999.0,
                                loss=loss,
                                scope=scope,
                                min_day=min_day,
                                max_iter=max_iter,
                                max_leaf_nodes=max_leaf_nodes,
                                min_samples_leaf=min_samples_leaf,
                                l2_regularization=l2,
                            )
                        correction = (raw_correction * float(blend)).clip(-float(cap), float(cap))
                        candidate_forecast = base_forecast + correction
                        nonzero = correction.abs().gt(1e-9)
                        row = {
                            "candidate": "walk_forward_hgbr_residual_guard",
                            "blend": blend,
                            "cap_mwh": cap,
                            "loss": loss,
                            "scope": scope,
                            "min_day": min_day,
                            "max_iter": max_iter,
                            "min_samples_leaf": min_samples_leaf,
                            "max_leaf_nodes": max_leaf_nodes,
                            "l2_regularization": l2,
                            "correction_rows": int(nonzero.sum()),
                            "mean_nonzero_correction_mwh": float(correction.loc[nonzero].mean()) if nonzero.any() else 0.0,
                        }
                        row.update(_metrics(replay, candidate_forecast, "metric"))
                        rows.append(row)

    for scope in ["peak_hot", "hot_peak", "peak_window"]:
        for min_day in [4, 8]:
            for blend in [0.25, 0.50, 0.75, 1.0]:
                for cap in [2.0, 3.0, 4.0, 5.0, 6.0]:
                    for upper_blend, upper_cap, low_forecast_threshold in [
                        (0.0, 0.0, 0.0),
                        (0.05, 2.0, 220.0),
                        (0.08, 3.0, 230.0),
                        (0.10, 4.0, 230.0),
                    ]:
                        correction = _evaluate_scenario_midpoint_guard(
                            replay,
                            blend=blend,
                            cap=cap,
                            scope=scope,
                            min_day=min_day,
                            upper_blend=upper_blend,
                            upper_cap=upper_cap,
                            low_forecast_threshold=low_forecast_threshold,
                        )
                        candidate_forecast = base_forecast + correction
                        nonzero = correction.abs().gt(1e-9)
                        row = {
                            "candidate": "scenario_midpoint_guard",
                            "blend": blend,
                            "cap_mwh": cap,
                            "loss": "none",
                            "scope": scope,
                            "min_day": min_day,
                            "upper_blend": upper_blend,
                            "upper_cap_mwh": upper_cap,
                            "low_forecast_threshold_mwh": low_forecast_threshold,
                            "correction_rows": int(nonzero.sum()),
                            "mean_nonzero_correction_mwh": float(correction.loc[nonzero].mean()) if nonzero.any() else 0.0,
                        }
                        row.update(_metrics(replay, candidate_forecast, "metric"))
                        rows.append(row)

    detail = pd.DataFrame(rows)
    detail["hot_peak_gate_excess_mwh"] = (detail["metric_hot_peak_mae_mwh"] - 6.0).clip(lower=0.0)
    detail["peak_window_gate_excess_mwh"] = (detail["metric_peak_window_14_18_mae_mwh"] - 5.5).clip(lower=0.0)
    detail["combined_gate_excess_mwh"] = detail["hot_peak_gate_excess_mwh"] + detail["peak_window_gate_excess_mwh"]
    detail.sort_values(
        ["combined_gate_excess_mwh", "metric_hot_peak_mae_mwh", "metric_peak_window_14_18_mae_mwh", "metric_overall_mae_mwh"],
        inplace=True,
    )
    out_path = output_dir / f"peak_window_residual_guard_evaluation_{label}.csv"
    summary_path = output_dir / f"peak_window_residual_guard_evaluation_{label}.json"
    detail.to_csv(out_path, index=False)

    current = detail[detail["candidate"].eq("current_saved_replay")].iloc[0].to_dict()
    best_combined = detail.iloc[0].to_dict()
    best_peak = detail.sort_values(["metric_peak_window_14_18_mae_mwh", "metric_hot_peak_mae_mwh"]).iloc[0].to_dict()
    best_hot = detail.sort_values(["metric_hot_peak_mae_mwh", "metric_peak_window_14_18_mae_mwh"]).iloc[0].to_dict()
    summary = {
        "replay_path": str(replay_path),
        "detail_path": str(out_path),
        "current_saved_replay": current,
        "best_combined_gate_excess": best_combined,
        "best_by_peak_window": best_peak,
        "best_by_hot_peak": best_hot,
        "recommendation": (
            "do_not_enable_peak_window_residual_guard"
            if not bool(best_combined.get("metric_scorecard_pass", False))
            else "review_candidate_before_enabling"
        ),
    }
    summary_path.write_text(json.dumps(summary, indent=2, default=str), encoding="utf-8")
    print(f"Wrote {out_path}")
    print(f"Wrote {summary_path}")
    print(detail.head(10).to_string(index=False))


if __name__ == "__main__":
    main()
