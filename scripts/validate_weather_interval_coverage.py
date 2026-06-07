from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import yaml

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from forecasting.forecast.uncertainty_bands import apply_bands, build_residual_band_lookup


def _load_config() -> dict[str, Any]:
    with open(Path("forecasting") / "config.yaml", "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def _latest_replay_label(output_dir: Path) -> str:
    files = sorted(
        output_dir.glob("rolling_origin_replay_results_*.csv"),
        key=lambda p: p.stat().st_mtime,
        reverse=True,
    )
    for path in files:
        label = path.stem.removeprefix("rolling_origin_replay_results_")
        if label and label != "recent_guard_candidate":
            return label
    raise FileNotFoundError("No labeled rolling_origin_replay_results_*.csv file found.")


def _hour_group(hour: int) -> str:
    h = int(hour)
    if 0 <= h <= 5:
        return "Overnight"
    if 6 <= h <= 9:
        return "Morning"
    if 10 <= h <= 15:
        return "Midday"
    if 16 <= h <= 20:
        return "Peak"
    return "LateEvening"


def _add_daily_max_bin(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    out["DailyMaxTempBin"] = pd.cut(
        pd.to_numeric(out["Temperature_DailyMax"], errors="coerce"),
        bins=[-999, 65, 75, 85, 90, 95, 100, 105, 999],
        labels=False,
        include_lowest=True,
    ).astype(float)
    return out


def _read_replay(path: Path) -> pd.DataFrame:
    df = pd.read_csv(path, low_memory=False)
    for col in ["DT", "Replay_Origin_DT", "Replay_Calibration_Start_DT", "Replay_Calibration_End_DT"]:
        if col in df.columns:
            df[col] = pd.to_datetime(df[col], errors="coerce", utc=True)
    return df


def _make_band_basis(replay: pd.DataFrame) -> pd.DataFrame:
    basis = replay.copy()
    basis["Residual_MWH"] = pd.to_numeric(basis["Final_Residual_MWH"], errors="coerce")
    return basis.dropna(subset=["Residual_MWH", "Actual_MWH", "Final_Backtest_Forecast_MWH"]).copy()


def _make_operational_weather_frame(replay: pd.DataFrame) -> pd.DataFrame:
    required = ["WeatherRealism_Final_Backtest_Forecast_MWH", "Actual_MWH"]
    work = replay.dropna(subset=[c for c in required if c in replay.columns]).copy()
    mappings = {
        "WeatherRealism_Temperature": "Temperature",
        "WeatherRealism_Temperature_DailyMax": "Temperature_DailyMax",
        "WeatherRealism_CloudCover_Norm": "CloudCover_Norm",
        "WeatherRealism_BTM_Solar_Proxy_MW": "BTM_Solar_Proxy_MW",
        "WeatherRealism_BTM_Solar_Loss_From_ClearSky_MW": "BTM_Solar_Loss_From_ClearSky_MW",
        "WeatherRealism_Midday_Overcast_Solar_Loss_MW": "Midday_Overcast_Solar_Loss_MW",
    }
    for source, target in mappings.items():
        if source in work.columns:
            work[target] = pd.to_numeric(work[source], errors="coerce")
    for source in [c for c in work.columns if c.startswith("WeatherRealism_WeatherScenario_")]:
        target = source.removeprefix("WeatherRealism_")
        work[target] = pd.to_numeric(work[source], errors="coerce")
    work["Calibrated_Forecast_MWH"] = pd.to_numeric(work["WeatherRealism_Final_Backtest_Forecast_MWH"], errors="coerce")
    work["Hour"] = pd.to_numeric(work.get("Hour"), errors="coerce").fillna(pd.to_datetime(work["DT"]).dt.hour).astype(int)
    work["HourGroup"] = work["Hour"].map(_hour_group)
    work["Month"] = pd.to_datetime(work["DT"]).dt.month.astype(int)
    work["Season"] = work.get("Season", np.nan)
    work["CloudSolarEventClass"] = ""
    work = work.drop(columns=["CloudCoverBucket", "SolarLossBucket", "DailyMaxTempBin"], errors="ignore")
    return _add_daily_max_bin(work)


def _prior_weather_cfg(current: dict[str, Any]) -> dict[str, Any]:
    prior = dict(current or {})
    prior.update(
        {
            "enabled": True,
            "max_day": 7,
            "day1_multiplier": 1.25,
            "days2to3_multiplier": 1.45,
            "days4to7_multiplier": 1.65,
            "cloudy_solar_multiplier": 1.20,
            "hot_peak_multiplier": 1.10,
            "shoulder_heat_multiplier": 1.0,
            "high_temp_multiplier": 1.0,
            "cap_multiplier": 2.25,
        }
    )
    return prior


def _apply_policy(frame: pd.DataFrame, lookup: dict, bands_cfg: dict[str, Any], policy: str) -> pd.DataFrame:
    weather_cfg = dict(bands_cfg.get("weather_input_risk", {}) or {})
    if policy == "no_weather_risk":
        weather_cfg["enabled"] = False
    elif policy == "prior_weather_risk":
        weather_cfg = _prior_weather_cfg(weather_cfg)
    elif policy != "current_weather_risk":
        raise ValueError(f"Unknown policy: {policy}")

    out = apply_bands(
        frame,
        percent_band=float(bands_cfg.get("default_percent_band", 0.055)),
        floor_mwh=float(bands_cfg.get("band_floor_mwh", 4.0)),
        residual_lookup=lookup,
        band_scale=float(bands_cfg.get("band_scale", 1.0)),
        weather_input_risk=weather_cfg,
    )
    actual = pd.to_numeric(out["Actual_MWH"], errors="coerce")
    out["Interval_Covered"] = actual.between(out["Lower_Band"], out["Upper_Band"])
    out["AbsError_MWH"] = (actual - pd.to_numeric(out["Calibrated_Forecast_MWH"], errors="coerce")).abs()
    out["Band_Miss_MWH"] = np.maximum(out["AbsError_MWH"] - pd.to_numeric(out["Band"], errors="coerce"), 0.0)
    out["Policy"] = policy
    return out


def _scenario_halfspread(frame: pd.DataFrame) -> pd.Series:
    if "WeatherScenario_HalfSpread_MWH" in frame.columns:
        return pd.to_numeric(frame["WeatherScenario_HalfSpread_MWH"], errors="coerce").fillna(0.0)
    scenario_cols = [
        c
        for c in frame.columns
        if c.startswith("WeatherScenario_")
        and c.endswith("_P50_MWH")
        and c not in {"WeatherScenario_Min_P50_MWH", "WeatherScenario_Max_P50_MWH"}
    ]
    if not scenario_cols:
        return pd.Series(0.0, index=frame.index, dtype=float)
    scen = frame[scenario_cols].apply(pd.to_numeric, errors="coerce")
    return ((scen.max(axis=1) - scen.min(axis=1)) / 2.0).fillna(0.0)


def _conformal_quantile(
    calibration: pd.DataFrame,
    risk_class: str,
    horizon_bucket: str,
    quantile: float,
    min_group_rows: int,
    min_global_rows: int,
) -> tuple[float, str]:
    candidates = [
        (
            calibration[
                calibration["Weather_Input_Risk_Class"].astype(str).eq(str(risk_class))
                & calibration["Replay_Horizon_Bucket"].astype(str).eq(str(horizon_bucket))
            ],
            "risk_horizon",
            min_group_rows,
        ),
        (
            calibration[calibration["Weather_Input_Risk_Class"].astype(str).eq(str(risk_class))],
            "risk_class",
            min_group_rows,
        ),
        (
            calibration[calibration["Replay_Horizon_Bucket"].astype(str).eq(str(horizon_bucket))],
            "horizon",
            min_group_rows,
        ),
        (calibration, "global", min_global_rows),
    ]
    for sample, source, min_rows in candidates:
        errors = pd.to_numeric(sample.get("AbsError_MWH"), errors="coerce").dropna()
        if len(errors) >= int(min_rows):
            return float(errors.quantile(float(quantile))), source
    return np.nan, "insufficient_prior"


def _apply_walk_forward_conformal_weather_policy(current_detail: pd.DataFrame, bands_cfg: dict[str, Any]) -> pd.DataFrame:
    cfg = dict((bands_cfg.get("conformal_weather", {}) or {}))
    out = current_detail.copy()
    out["Policy"] = "current_weather_risk_conformal_walkforward"
    out["Pre_Conformal_Band_MWH"] = pd.to_numeric(out["Band"], errors="coerce")
    out["WeatherScenario_HalfSpread_MWH"] = _scenario_halfspread(out)
    out["Conformal_Weather_Band_MWH"] = np.nan
    out["Conformal_Weather_Source"] = "disabled"
    if not bool(cfg.get("enabled", True)):
        return out

    quantile = float(cfg.get("quantile", 0.80))
    safety = float(cfg.get("safety_multiplier", 1.0))
    scenario_mult = float(cfg.get("scenario_spread_multiplier", 1.0))
    min_fraction = float(cfg.get("min_fraction_of_multiplier_band", 0.0))
    min_group_rows = int(cfg.get("min_group_rows", 20))
    min_global_rows = int(cfg.get("min_global_rows", 50))
    allow_narrowing = bool(cfg.get("allow_narrowing", True))

    origin_col = "Replay_Origin_DT"
    if origin_col not in out.columns:
        out["Conformal_Weather_Source"] = "missing_origin"
        return out

    out[origin_col] = pd.to_datetime(out[origin_col], errors="coerce", utc=True)
    out["Weather_Input_Risk_Class"] = out.get("Weather_Input_Risk_Class", "none").astype(str)
    out["Replay_Horizon_Bucket"] = out.get("Replay_Horizon_Bucket", "").astype(str)
    out = out.sort_values([origin_col, "DT"]).copy()
    origins = [origin for origin in out[origin_col].dropna().sort_values().unique()]

    for origin in origins:
        origin_mask = out[origin_col].eq(origin)
        calibration = out[out[origin_col].lt(origin)].copy()
        if calibration.empty:
            out.loc[origin_mask, "Conformal_Weather_Source"] = "insufficient_prior"
            continue

        for idx, row in out.loc[origin_mask].iterrows():
            q, source = _conformal_quantile(
                calibration,
                str(row.get("Weather_Input_Risk_Class", "none")),
                str(row.get("Replay_Horizon_Bucket", "")),
                quantile,
                min_group_rows,
                min_global_rows,
            )
            out.loc[idx, "Conformal_Weather_Band_MWH"] = q
            out.loc[idx, "Conformal_Weather_Source"] = source

    current = out["Pre_Conformal_Band_MWH"].astype(float)
    multiplier = pd.to_numeric(out.get("Weather_Input_Risk_Multiplier", 1.0), errors="coerce").replace(0.0, np.nan).fillna(1.0)
    base_band = current / multiplier
    conformal = pd.to_numeric(out["Conformal_Weather_Band_MWH"], errors="coerce")
    scenario_band = pd.to_numeric(out["WeatherScenario_HalfSpread_MWH"], errors="coerce").fillna(0.0) * scenario_mult
    candidate = np.maximum.reduce(
        [
            base_band.to_numpy(dtype=float),
            (conformal.fillna(-np.inf) * safety).to_numpy(dtype=float),
            scenario_band.to_numpy(dtype=float),
        ]
    )
    eligible = multiplier.gt(1.0) | ~out["Weather_Input_Risk_Class"].astype(str).eq("none")
    candidate = np.where(eligible.to_numpy(), candidate, current.to_numpy(dtype=float))
    no_conformal = out["Conformal_Weather_Source"].astype(str).eq("insufficient_prior")
    candidate = np.where(no_conformal.to_numpy(), current.to_numpy(dtype=float), candidate)
    if min_fraction > 0:
        candidate = np.maximum(candidate, current.to_numpy(dtype=float) * min_fraction)
    out["Band"] = candidate if allow_narrowing else np.maximum(current.to_numpy(dtype=float), candidate)

    base = pd.to_numeric(out["Calibrated_Forecast_MWH"], errors="coerce")
    actual = pd.to_numeric(out["Actual_MWH"], errors="coerce")
    out["Upper_Band"] = base + out["Band"].astype(float)
    out["Lower_Band"] = np.maximum(0.0, base - out["Band"].astype(float))
    out["P10_Forecast_MWH"] = out["Lower_Band"]
    out["P50_Forecast_MWH"] = base.clip(lower=0.0)
    out["P90_Forecast_MWH"] = out["Upper_Band"]
    out["Interval_Covered"] = actual.between(out["Lower_Band"], out["Upper_Band"])
    out["Band_Miss_MWH"] = np.maximum(out["AbsError_MWH"] - pd.to_numeric(out["Band"], errors="coerce"), 0.0)
    out["Quantile_Method"] = out.get("Quantile_Method", "conditional_residual_central80").astype(str) + "+walkforward_conformal_weather"
    return out


def _event_slice(row: pd.Series) -> str:
    hour_group = str(row.get("HourGroup"))
    season = str(row.get("Season"))
    hour = pd.to_numeric(pd.Series([row.get("Hour")]), errors="coerce").iloc[0]
    day = pd.to_numeric(pd.Series([row.get("Forecast_Day")]), errors="coerce").iloc[0]
    temp = pd.to_numeric(pd.Series([row.get("Temperature_DailyMax")]), errors="coerce").iloc[0]
    cloud = pd.to_numeric(pd.Series([row.get("CloudCover_Norm")]), errors="coerce").iloc[0]
    loss = pd.to_numeric(pd.Series([row.get("BTM_Solar_Loss_From_ClearSky_MW")]), errors="coerce").iloc[0]
    if np.isfinite(day) and day >= 8:
        return "long_horizon_days8to16"
    if hour_group == "Peak" and np.isfinite(temp) and temp >= 90:
        return "hot_peak"
    if hour_group == "Midday" and ((np.isfinite(cloud) and cloud >= 0.60) or (np.isfinite(loss) and loss >= 1.25)):
        return "cloudy_solar_loss_midday"
    if season in {"Spring", "Fall"} and np.isfinite(hour) and 12 <= hour <= 22 and np.isfinite(temp) and 75 <= temp <= 93:
        return "shoulder_heat_transition"
    return "normal"


def _coverage_metrics(group: pd.DataFrame) -> pd.Series:
    covered = group["Interval_Covered"].astype(bool)

    def mean_col(col: str) -> float:
        if col not in group.columns:
            return np.nan
        return float(pd.to_numeric(group[col], errors="coerce").mean())

    return pd.Series(
        {
            "N": int(len(group)),
            "Coverage_PCT": float(covered.mean() * 100.0),
            "Avg_Band_MWH": float(pd.to_numeric(group["Band"], errors="coerce").mean()),
            "Avg_Pre_Conformal_Band_MWH": mean_col("Pre_Conformal_Band_MWH"),
            "Avg_Conformal_Weather_Band_MWH": mean_col("Conformal_Weather_Band_MWH"),
            "Avg_WeatherScenario_HalfSpread_MWH": mean_col("WeatherScenario_HalfSpread_MWH"),
            "Avg_AbsError_MWH": float(pd.to_numeric(group["AbsError_MWH"], errors="coerce").mean()),
            "P90_AbsError_MWH": float(pd.to_numeric(group["AbsError_MWH"], errors="coerce").quantile(0.90)),
            "Avg_Band_Miss_MWH": float(pd.to_numeric(group["Band_Miss_MWH"], errors="coerce").mean()),
            "P90_Band_Miss_MWH": float(pd.to_numeric(group["Band_Miss_MWH"], errors="coerce").quantile(0.90)),
            "Avg_Weather_Risk_Multiplier": float(pd.to_numeric(group["Weather_Input_Risk_Multiplier"], errors="coerce").mean()),
        }
    )


def _scorecard(detail: pd.DataFrame) -> pd.DataFrame:
    frames = []

    def add(name: str, group_cols: list[str]) -> None:
        if group_cols and not all(c in detail.columns for c in group_cols):
            return
        if not group_cols:
            out = detail.groupby("Policy", dropna=False).apply(_coverage_metrics, include_groups=False).reset_index()
            out.insert(1, "Slice", "Overall")
            frames.append(out)
            return
        out = detail.groupby(["Policy"] + group_cols, dropna=False).apply(_coverage_metrics, include_groups=False).reset_index()
        slice_values = out[group_cols].astype("string").fillna("nan").apply(lambda row: "|".join(row.tolist()), axis=1)
        out["Slice"] = name + ":" + slice_values
        frames.append(out.drop(columns=group_cols))

    add("Overall", [])
    add("Horizon", ["Replay_Horizon_Bucket"])
    add("ForecastDay", ["Forecast_Day"])
    add("WeatherLead", ["Weather_Forecast_Lead_Bucket"])
    add("RiskClass", ["Weather_Input_Risk_Class"])
    add("Event", ["Interval_Event_Slice"])
    add("Season", ["Season"])
    add("ConformalSource", ["Conformal_Weather_Source"])
    return pd.concat(frames, ignore_index=True, sort=False) if frames else pd.DataFrame()


def main() -> None:
    parser = argparse.ArgumentParser(description="Validate weather-realism interval coverage from a saved replay.")
    parser.add_argument("--label", default=None, help="Replay label, e.g. recent_guard_full_20260524_210025")
    parser.add_argument("--replay-path", default=None, help="Explicit rolling_origin_replay_results CSV path.")
    parser.add_argument("--output-label", default=None, help="Output label when --replay-path is used.")
    args = parser.parse_args()

    output_dir = Path("forecast_outputs")
    if args.replay_path:
        replay_path = Path(args.replay_path)
        if not replay_path.exists():
            raise FileNotFoundError(f"Replay path not found: {replay_path}")
        label = args.output_label or replay_path.stem.removeprefix("rolling_origin_replay_results").strip("_") or "adhoc"
    else:
        label = args.label or _latest_replay_label(output_dir)
        replay_path = output_dir / f"rolling_origin_replay_results_{label}.csv"
    config = _load_config()
    bands_cfg = config.get("bands", {}) or {}
    replay = _read_replay(replay_path)

    lookup = build_residual_band_lookup(
        _make_band_basis(replay),
        shrink_floor_mwh=float(bands_cfg.get("band_floor_mwh", 4.0)),
    )
    operational = _make_operational_weather_frame(replay)
    base_detail = pd.concat(
        [_apply_policy(operational, lookup, bands_cfg, policy) for policy in ["no_weather_risk", "prior_weather_risk", "current_weather_risk"]],
        ignore_index=True,
        sort=False,
    )
    base_detail["Weather_Forecast_Lead_Bucket"] = pd.cut(
        pd.to_numeric(base_detail.get("WeatherRealism_Forecast_Weather_Lead_Days"), errors="coerce"),
        bins=[-np.inf, 1, 3, 7, np.inf],
        labels=["day1", "days2to3", "days4to7", "days8plus"],
        include_lowest=True,
    ).astype("object")
    base_detail["Interval_Event_Slice"] = base_detail.apply(_event_slice, axis=1)
    current = base_detail[base_detail["Policy"].eq("current_weather_risk")].copy()
    conformal = _apply_walk_forward_conformal_weather_policy(current, bands_cfg)
    detail = pd.concat([base_detail, conformal], ignore_index=True, sort=False)

    scorecard = _scorecard(detail)
    detail_path = output_dir / f"weather_interval_coverage_validation_detail_{label}.csv"
    scorecard_path = output_dir / f"weather_interval_coverage_validation_scorecard_{label}.csv"
    summary_path = output_dir / f"weather_interval_coverage_validation_summary_{label}.json"
    detail.to_csv(detail_path, index=False)
    scorecard.to_csv(scorecard_path, index=False)

    summary_rows = scorecard[(scorecard["Slice"].eq("Overall"))].copy()
    summary = {
        "label": label,
        "replay_path": str(replay_path),
        "detail_path": str(detail_path),
        "scorecard_path": str(scorecard_path),
        "overall": summary_rows.to_dict(orient="records"),
    }
    summary_path.write_text(json.dumps(summary, indent=2, default=str), encoding="utf-8")
    print(f"Wrote {scorecard_path}")
    print(f"Wrote {detail_path}")
    print(f"Wrote {summary_path}")
    print(scorecard[scorecard["Slice"].isin(["Overall", "Horizon:Days2to7", "Event:hot_peak", "Event:shoulder_heat_transition", "Event:cloudy_solar_loss_midday"])].to_string(index=False))


if __name__ == "__main__":
    main()
