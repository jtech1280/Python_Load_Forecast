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

from forecasting.forecast.forecast_pipeline import apply_operational_stage_selector


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
            "PeakRisk": _as_num(frame, "Peak_Risk_Adjusted_Forecast_MWH"),
            "ResidualCalibrated": _as_num(frame, "Residual_Calibrated_Forecast_MWH"),
            "RecentCorrected": _as_num(frame, "Recent_Corrected_Forecast_MWH"),
            "Hour": hour,
            "HourSin": np.sin(2.0 * np.pi * hour / 24.0),
            "HourCos": np.cos(2.0 * np.pi * hour / 24.0),
            "ForecastDay": day,
            "Day1": day.eq(1).astype(float),
            "Days2to7": day.between(2, 7).astype(float),
            "Days8Plus": day.ge(8).astype(float),
            "Temperature": _as_num(frame, "Temperature"),
            "DailyMaxTemp": daily_max,
            "CloudCover": cloud,
            "SolarLoss": solar_loss,
            "Humidity": _as_num(frame, "Humidity_Norm"),
            "Weekend": _as_num(frame, "IsWeekend", 0.0).fillna(0.0),
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
    masks = {
        "overall": pd.Series(True, index=frame.index),
        "day1": day.eq(1),
        "days2to7": day.between(2, 7),
        "hot_peak": hour.between(16, 20) & temp.ge(90.0),
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
    return out


def _evaluate_model_guard(replay: pd.DataFrame, blend: float, cap: float, loss: str) -> pd.Series:
    out = pd.Series(0.0, index=replay.index, dtype=float)
    hour = _as_num(replay, "Hour")
    temp = _as_num(replay, "Temperature_DailyMax")
    eligible = hour.between(14, 18) | (hour.between(16, 20) & temp.ge(90.0))
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
            max_iter=60,
            max_leaf_nodes=4,
            min_samples_leaf=30,
            l2_regularization=20.0,
            random_state=42,
        )
        model.fit(x_train.loc[valid].fillna(fill).fillna(0.0), target.loc[valid])
        pred = pd.Series(model.predict(_feature_frame(test).fillna(fill).fillna(0.0)), index=test.index)
        out.loc[test.index] = (pred * float(blend)).clip(-float(cap), float(cap))
    return out


def main() -> None:
    parser = argparse.ArgumentParser(description="Evaluate a guarded peak-window residual model against saved replay origins.")
    parser.add_argument("--replay-path", default=None)
    parser.add_argument("--label", default=None)
    args = parser.parse_args()

    output_dir = Path("forecast_outputs")
    replay_path = Path(args.replay_path) if args.replay_path else _latest_replay(output_dir)
    label = args.label or replay_path.stem.removeprefix("rolling_origin_replay_results_")
    config = _load_config()
    replay = pd.read_csv(replay_path, low_memory=False)
    replay["Replay_Origin_DT"] = pd.to_datetime(replay["Replay_Origin_DT"], errors="coerce", utc=True)
    replay = apply_operational_stage_selector(replay, config=config, forecast_col="Final_Backtest_Forecast_MWH")

    base_forecast = _as_num(replay, "Final_Backtest_Forecast_MWH")
    rows: list[dict[str, Any]] = []
    base = {
        "candidate": "current_stage_selector",
        "blend": 0.0,
        "cap_mwh": 0.0,
        "loss": "none",
        "correction_rows": 0,
        "mean_nonzero_correction_mwh": 0.0,
    }
    base.update(_metrics(replay, base_forecast, "metric"))
    rows.append(base)

    for loss in ["absolute_error", "squared_error"]:
        for blend in [0.05, 0.10, 0.15, 0.25, 0.35, 0.50]:
            for cap in [1.0, 2.0, 3.0, 4.0, 5.0]:
                correction = _evaluate_model_guard(replay, blend=blend, cap=cap, loss=loss)
                candidate_forecast = base_forecast + correction
                nonzero = correction.abs().gt(1e-9)
                row = {
                    "candidate": "walk_forward_hgbr_residual_guard",
                    "blend": blend,
                    "cap_mwh": cap,
                    "loss": loss,
                    "correction_rows": int(nonzero.sum()),
                    "mean_nonzero_correction_mwh": float(correction.loc[nonzero].mean()) if nonzero.any() else 0.0,
                }
                row.update(_metrics(replay, candidate_forecast, "metric"))
                rows.append(row)

    detail = pd.DataFrame(rows).sort_values(
        ["metric_peak_window_14_18_mae_mwh", "metric_hot_peak_mae_mwh", "metric_overall_mae_mwh"]
    )
    out_path = output_dir / f"peak_window_residual_guard_evaluation_{label}.csv"
    summary_path = output_dir / f"peak_window_residual_guard_evaluation_{label}.json"
    detail.to_csv(out_path, index=False)

    current = detail[detail["candidate"].eq("current_stage_selector")].iloc[0].to_dict()
    best_peak = detail.iloc[0].to_dict()
    best_hot = detail.sort_values(["metric_hot_peak_mae_mwh", "metric_peak_window_14_18_mae_mwh"]).iloc[0].to_dict()
    summary = {
        "replay_path": str(replay_path),
        "detail_path": str(out_path),
        "current_stage_selector": current,
        "best_by_peak_window": best_peak,
        "best_by_hot_peak": best_hot,
        "recommendation": (
            "do_not_enable_peak_window_residual_guard"
            if best_peak.get("metric_peak_window_14_18_mae_mwh", np.inf) >= current.get("metric_peak_window_14_18_mae_mwh", np.inf)
            and best_hot.get("metric_hot_peak_mae_mwh", np.inf) >= current.get("metric_hot_peak_mae_mwh", np.inf)
            else "review_candidate_before_enabling"
        ),
    }
    summary_path.write_text(json.dumps(summary, indent=2, default=str), encoding="utf-8")
    print(f"Wrote {out_path}")
    print(f"Wrote {summary_path}")
    print(detail.head(10).to_string(index=False))


if __name__ == "__main__":
    main()
