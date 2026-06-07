from __future__ import annotations

"""Optional V12.5 Optuna tuning harness.

This module is intentionally not run by the normal forecast path. Use it for controlled experiments
once a representative multi-season backtest configuration is available. It keeps imports lazy so the
main forecast does not require optuna to be installed.
"""

from copy import deepcopy
from typing import Any

import numpy as np
import pandas as pd


FINAL_STAGE_NAMES = {
    "final_corrected",
    "final_corrected_production",
    "recent_corrected_simulation",
}


def _final_rows(metrics: pd.DataFrame, stage_col: str) -> pd.DataFrame:
    if not isinstance(metrics, pd.DataFrame) or metrics.empty or stage_col not in metrics.columns:
        return pd.DataFrame()
    return metrics[metrics[stage_col].astype(str).isin(FINAL_STAGE_NAMES)].copy()


def _objective_score(stage_metrics: pd.DataFrame, segment_metrics: pd.DataFrame, weights: dict[str, float] | None = None) -> float:
    weights = weights or {}
    w_overall = float(weights.get("overall_mae", 0.40))
    w_peak = float(weights.get("peak_mae", 0.25))
    w_peak_miss = float(weights.get("daily_peak_miss", 0.20))
    w_warm_bias = float(weights.get("warm_peak_bias", 0.10))
    w_cloud = float(weights.get("cloudy_midday_mae", 0.05))

    final = _final_rows(stage_metrics, "Model")
    overall_mae = float(final["MAE_MWH"].iloc[0]) if not final.empty and "MAE_MWH" in final else 1e6

    peak_mae = overall_mae
    warm_bias = 0.0
    cloud_mae = overall_mae
    if isinstance(segment_metrics, pd.DataFrame) and not segment_metrics.empty:
        final_seg = segment_metrics[segment_metrics.get("Stage", "").eq("final_corrected")].copy() if "Stage" in segment_metrics.columns else segment_metrics.copy()
        if "HourGroup" in final_seg.columns:
            peak = final_seg[final_seg["HourGroup"].astype(str).eq("Peak")]
            if not peak.empty and "MAE_MWH" in peak:
                peak_mae = float(peak["MAE_MWH"].mean())
        if {"DailyMaxTempBucket", "HourGroup", "Bias_MWH"}.issubset(final_seg.columns):
            warm = final_seg[final_seg["DailyMaxTempBucket"].astype(str).isin(["75-85", "85-90"]) & final_seg["HourGroup"].astype(str).eq("Peak")]
            if not warm.empty:
                warm_bias = float(np.abs(warm["Bias_MWH"]).mean())
        if {"CloudCoverBucket", "HourGroup", "MAE_MWH"}.issubset(final_seg.columns):
            cloudy = final_seg[final_seg["CloudCoverBucket"].astype(str).isin(["Mostly Cloudy", "Overcast"]) & final_seg["HourGroup"].astype(str).eq("Midday")]
            if not cloudy.empty:
                cloud_mae = float(cloudy["MAE_MWH"].mean())

    # daily peak miss term is filled by caller when available; use peak MAE fallback here.
    peak_miss = peak_mae
    return w_overall * overall_mae + w_peak * peak_mae + w_peak_miss * peak_miss + w_warm_bias * warm_bias + w_cloud * cloud_mae


def scorecard_objective(scorecard: pd.DataFrame, weights: dict[str, float] | None = None) -> float:
    """Objective for the season-balanced replay scorecard.

    It intentionally uses only slices emitted by ``rolling_origin_replay_scorecard.csv`` so correction
    tuning can compare runs without rebuilding ad hoc segment tables.
    """
    weights = weights or {}
    final = _final_rows(scorecard, "Model")
    if final.empty or "Slice" not in final.columns:
        return 1e6

    def _metric(slice_name: str, column: str, default: float) -> float:
        rows = final[final["Slice"].astype(str).eq(slice_name)]
        if rows.empty or column not in rows or pd.to_numeric(rows[column], errors="coerce").dropna().empty:
            return default
        return float(pd.to_numeric(rows[column], errors="coerce").dropna().mean())

    overall_mae = _metric("Overall", "MAE_MWH", 1e6)
    peak_mae = _metric("PeakWindowHours14to18", "MAE_MWH", overall_mae)
    peak_miss = abs(_metric("PeakWindowHours14to18", "Underforecast_At_Actual_Peak_MWH", peak_mae))
    warm_bias = abs(_metric("HotPeakDailyMax90Plus", "Bias_MWH", 0.0))
    cloud_mae = _metric("CloudSolarMidday", "MAE_MWH", overall_mae)
    return (
        float(weights.get("overall_mae", 0.40)) * overall_mae
        + float(weights.get("peak_mae", 0.25)) * peak_mae
        + float(weights.get("daily_peak_miss", 0.20)) * peak_miss
        + float(weights.get("warm_peak_bias", 0.10)) * warm_bias
        + float(weights.get("cloudy_midday_mae", 0.05)) * cloud_mae
    )


def suggest_v125_params(trial: Any, base_config: dict) -> dict:
    cfg = deepcopy(base_config)
    cal = cfg.setdefault("calibration", {})
    rr = cal.setdefault("recent_residual", {})
    rr_weights = rr.setdefault("weights", {})

    cal["cloud_solar_shape_blend"] = trial.suggest_float("cloud_solar_shape_blend", 0.55, 0.95)
    cal["cloud_solar_shape_cap_mwh"] = trial.suggest_float("cloud_solar_shape_cap_mwh", 8.0, 24.0)
    cal["warm_ramp_blend"] = trial.suggest_float("warm_ramp_blend", 0.55, 0.98)
    cal["warm_ramp_cap_mwh"] = trial.suggest_float("warm_ramp_cap_mwh", 8.0, 22.0)
    rr["blend"] = trial.suggest_float("recent_blend", 0.60, 0.98)
    rr["cap_mwh"] = trial.suggest_float("recent_cap_mwh", 6.0, 16.0)
    rr_weights["solar_loss_hourgroup"] = trial.suggest_float("w_solar_loss_hourgroup", 0.03, 0.16)
    rr_weights["temp_cloud_hourgroup"] = trial.suggest_float("w_temp_cloud_hourgroup", 0.02, 0.12)
    rr_weights["same_hour"] = trial.suggest_float("w_same_hour", 0.08, 0.25)
    return cfg


def optuna_not_installed_message() -> str:
    return "Install optuna with: python -m pip install optuna"
