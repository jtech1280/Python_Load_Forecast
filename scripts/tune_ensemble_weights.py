from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

import numpy as np
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from forecasting.model.ensemble import blend_predictions


COMPONENT_COLUMNS = {
    "xgb": "XGB_Pred_MWH",
    "lgb": "LGB_Pred_MWH",
    "catboost": "CatBoost_Pred_MWH",
}


def _metrics(actual: np.ndarray, pred: np.ndarray) -> dict[str, float]:
    mask = np.isfinite(actual) & np.isfinite(pred)
    if not mask.any():
        return {}
    actual_valid = actual[mask]
    pred_valid = pred[mask]
    residual = actual_valid - pred_valid
    peak_idx = int(np.nanargmax(actual_valid))
    return {
        "N": int(mask.sum()),
        "MAE_MWH": float(np.mean(np.abs(residual))),
        "RMSE_MWH": float(np.sqrt(np.mean(residual ** 2))),
        "Bias_MWH": float(np.mean(residual)),
        "Underforecast_Rate_PCT": float(np.mean(residual > 0.0) * 100.0),
        "P90_AbsError_MWH": float(np.nanquantile(np.abs(residual), 0.90)),
        "Underforecast_At_Actual_Peak_MWH": float(actual_valid[peak_idx] - pred_valid[peak_idx]),
    }


def _weight_grid(step: float) -> list[dict[str, float]]:
    values = np.arange(0.0, 1.0 + (step / 2.0), step)
    rows: list[dict[str, float]] = []
    for xgb_weight in values:
        for lgb_weight in values:
            catboost_weight = 1.0 - float(xgb_weight) - float(lgb_weight)
            if catboost_weight < -1e-9:
                continue
            rows.append({
                "xgb": round(float(xgb_weight), 10),
                "lgb": round(float(lgb_weight), 10),
                "catboost": round(float(catboost_weight), 10),
            })
    return rows


def tune_weights(
    backtest_path: Path,
    output_dir: Path,
    step: float,
    max_peak_under_degradation: float,
) -> tuple[pd.DataFrame, dict]:
    backtest = pd.read_csv(backtest_path, low_memory=False)
    actual = pd.to_numeric(backtest["Actual_MWH"], errors="coerce").to_numpy(dtype=float)
    components = {
        name: pd.to_numeric(backtest[column], errors="coerce").to_numpy(dtype=float)
        for name, column in COMPONENT_COLUMNS.items()
        if column in backtest.columns
    }
    missing = sorted(set(COMPONENT_COLUMNS) - set(components))
    if missing:
        raise SystemExit(f"Missing component columns in {backtest_path}: {', '.join(missing)}")

    baseline_weights = {"xgb": 0.50, "lgb": 0.30, "catboost": 0.0}
    baseline_pred = blend_predictions(
        components["xgb"],
        components["lgb"],
        baseline_weights,
        catboost_pred=components["catboost"],
    )
    baseline = _metrics(actual, baseline_pred)
    max_peak_under = baseline["Underforecast_At_Actual_Peak_MWH"] + float(max_peak_under_degradation)

    rows = []
    for weights in _weight_grid(step):
        pred = blend_predictions(
            components["xgb"],
            components["lgb"],
            weights,
            catboost_pred=components["catboost"],
        )
        metrics = _metrics(actual, pred)
        if not metrics:
            continue
        rows.append({**weights, **metrics})

    grid = pd.DataFrame(rows).sort_values(["MAE_MWH", "RMSE_MWH"]).reset_index(drop=True)
    constrained = grid[grid["Underforecast_At_Actual_Peak_MWH"] <= max_peak_under].copy()
    recommendation_row = (constrained if not constrained.empty else grid).iloc[0]
    recommendation = {
        "source": str(backtest_path),
        "step": float(step),
        "selection_rule": "min_mae_with_actual_peak_underforecast_not_worse_than_baseline",
        "max_peak_under_degradation_mwh": float(max_peak_under_degradation),
        "baseline_weights": baseline_weights,
        "baseline_metrics": baseline,
        "recommended_weights": {
            "xgb": float(recommendation_row["xgb"]),
            "lgb": float(recommendation_row["lgb"]),
            "catboost": float(recommendation_row["catboost"]),
            "prophet": 0.0,
        },
        "recommended_metrics": {
            key: float(recommendation_row[key])
            for key in [
                "MAE_MWH",
                "RMSE_MWH",
                "Bias_MWH",
                "Underforecast_Rate_PCT",
                "P90_AbsError_MWH",
                "Underforecast_At_Actual_Peak_MWH",
            ]
        },
    }

    output_dir.mkdir(parents=True, exist_ok=True)
    grid.to_csv(output_dir / "ensemble_weight_grid.csv", index=False)
    (output_dir / "ensemble_weight_recommendation.json").write_text(
        json.dumps(recommendation, indent=2),
        encoding="utf-8",
    )
    return grid, recommendation


def main() -> None:
    parser = argparse.ArgumentParser(description="Grid-tune XGB/LGB/CatBoost ensemble weights from backtest diagnostics.")
    parser.add_argument("--backtest", type=Path, default=Path("forecast_outputs/backtest_enriched.csv"))
    parser.add_argument("--output-dir", type=Path, default=Path("forecast_outputs"))
    parser.add_argument("--step", type=float, default=0.05)
    parser.add_argument(
        "--max-peak-under-degradation",
        type=float,
        default=0.0,
        help="Allowed increase in underforecast at the actual peak versus the baseline XGB/LGB blend.",
    )
    args = parser.parse_args()

    _grid, recommendation = tune_weights(
        backtest_path=args.backtest,
        output_dir=args.output_dir,
        step=args.step,
        max_peak_under_degradation=args.max_peak_under_degradation,
    )
    print(json.dumps(recommendation, indent=2))


if __name__ == "__main__":
    main()
