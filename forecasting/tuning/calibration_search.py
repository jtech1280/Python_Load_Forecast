from __future__ import annotations

"""Cache the expensive, config-training-independent half of the rolling-origin replay so
calibration-parameter search (forecasting/tuning/optuna_tuning.py) can score many trial
configs without repeating XGB/LGB/CatBoost/Prophet training on every trial.

Architecture (see forecasting/backtest/rolling_origin_replay.py for the production path):
  per origin: [expensive, training-dependent]  calibration backtest + _origin_raw_forecasts
              [cheap, calibration-dependent]    build_correction_artifacts + apply_origin_correction_chain

`build_raw_origin_bundles` runs only the expensive half once and returns picklable bundles.
`save_raw_origin_bundles`/`load_raw_origin_bundles` persist them to disk. `score_bundles`
runs only the cheap half against a (possibly different) config and returns a replay
DataFrame shaped like `run_rolling_origin_replay`'s output, suitable for
`build_rolling_origin_replay_bundle` / `scorecard_objective`.

This intentionally reaches into rolling_origin_replay.py's private helpers
(_origin_candidates, _origin_raw_forecasts, _replay_cfg, _as_int) to stay byte-for-byte
consistent with the production origin-selection and raw-forecast logic rather than
reimplementing it. If that module's internals move, update the imports here.
"""

import pickle
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import pandas as pd

from forecasting.backtest.rolling_backtest import run_rolling_backtest
from forecasting.backtest.rolling_origin_replay import (
    _as_int,
    _origin_candidates,
    _origin_raw_forecasts,
    _replay_cfg,
    apply_origin_correction_chain,
)
from forecasting.forecast.forecast_pipeline import (
    build_correction_artifacts,
    rare_event_artifact_lookback_days,
    _production_ensemble_weights,
)


@dataclass
class RawOriginBundle:
    """Everything `apply_origin_correction_chain` needs for one origin, captured before any
    calibration-parameter-dependent processing runs."""

    origin_number: int
    origin_dt: pd.Timestamp
    calibration_days: int
    raw_calibration: pd.DataFrame
    raw_origin: pd.DataFrame
    raw_weather_realism: pd.DataFrame
    raw_realized_scenarios: dict[str, pd.DataFrame]
    raw_weather_scenarios: dict[str, pd.DataFrame]
    # Optional longer-history raw backtest for hot_ramp_peak_capture/
    # heat_persistence_peak_capture's walk-forward lookup artifacts (see
    # calibration.rare_event_artifact_lookback_days). None when unset -- matches
    # build_correction_artifacts' own default (no extra backtest, no behavior change).
    # Bundles pickled before this field existed won't have it; score_bundles() reads it
    # with getattr(..., None) so old caches still load, they just won't benefit from the
    # extended lookback until the cache is rebuilt.
    extended_lookback_raw: pd.DataFrame | None = None


def build_raw_origin_bundles(
    train_df: pd.DataFrame,
    features: list[str],
    config: dict,
    *,
    origin_limit: int | None = None,
) -> list[RawOriginBundle]:
    """Run calibration-window training + per-origin raw forecasting once per origin.

    Everything here depends on `config`'s `model.*`/`training.*` (and feature-building)
    settings, NOT on `calibration.*` settings. Build the cache once against the base config
    before starting a calibration-only search; do not rebuild it per trial, and do not reuse
    a cache across configs that change model/feature/training settings.
    """
    if train_df is None or train_df.empty:
        return []

    cfg = _replay_cfg(config)
    horizon_days = _as_int(cfg.get("horizon_days"), 16)
    calibration_days = _as_int(cfg.get("calibration_days"), 45)
    skip_catboost = bool(cfg.get("skip_catboost", False))
    skip_calibration_prophet = bool(cfg.get("skip_calibration_prophet", False))

    work = train_df.copy().sort_values("DT").reset_index(drop=True)
    origins = _origin_candidates(work, config)
    if origin_limit is not None:
        origins = origins[: int(origin_limit)]

    extended_lookback_days = rare_event_artifact_lookback_days(config, calibration_days)

    bundles: list[RawOriginBundle] = []
    for origin_number, origin_dt in enumerate(origins, start=1):
        print(
            f"[calibration_search] building raw bundle for origin {origin_number}/{len(origins)}: {origin_dt}",
            flush=True,
        )
        pre_origin = work[work["DT"] < origin_dt].copy()
        raw_calibration = run_rolling_backtest(
            train_df=pre_origin,
            features=features,
            ensemble_weights=_production_ensemble_weights(config),
            backtest_days=calibration_days,
            config=config,
            skip_catboost=skip_catboost,
            skip_prophet=skip_calibration_prophet,
        )
        raw_calibration.attrs = {}
        extended_lookback_raw = None
        if extended_lookback_days is not None:
            print(
                f"[calibration_search] building rare-event lookback ({extended_lookback_days}d) "
                f"for origin {origin_number}/{len(origins)}: {origin_dt}",
                flush=True,
            )
            extended_lookback_raw = run_rolling_backtest(
                train_df=pre_origin,
                features=features,
                ensemble_weights=_production_ensemble_weights(config),
                backtest_days=extended_lookback_days,
                config=config,
                skip_catboost=True,
                skip_prophet=True,
            )
            extended_lookback_raw.attrs = {}
        (
            raw_origin,
            raw_weather_realism,
            raw_realized_scenarios,
            raw_weather_scenarios,
        ) = _origin_raw_forecasts(
            work,
            features,
            config,
            origin_dt,
            horizon_days,
            origin_number,
        )
        if raw_origin.empty:
            continue
        bundles.append(
            RawOriginBundle(
                origin_number=origin_number,
                origin_dt=origin_dt,
                calibration_days=calibration_days,
                raw_calibration=raw_calibration,
                raw_origin=raw_origin,
                raw_weather_realism=raw_weather_realism,
                raw_realized_scenarios=raw_realized_scenarios,
                raw_weather_scenarios=raw_weather_scenarios,
                extended_lookback_raw=extended_lookback_raw,
            )
        )
    return bundles


def _bundle_path(cache_dir: Path, bundle: RawOriginBundle) -> Path:
    date_tag = pd.Timestamp(bundle.origin_dt).strftime("%Y%m%d")
    return cache_dir / f"origin_{bundle.origin_number:03d}_{date_tag}.pkl"


def save_raw_origin_bundles(
    bundles: list[RawOriginBundle], cache_dir: str | Path
) -> list[Path]:
    cache_dir = Path(cache_dir)
    cache_dir.mkdir(parents=True, exist_ok=True)
    paths = []
    for bundle in bundles:
        path = _bundle_path(cache_dir, bundle)
        with path.open("wb") as fh:
            pickle.dump(bundle, fh, protocol=pickle.HIGHEST_PROTOCOL)
        paths.append(path)
    return paths


def load_raw_origin_bundles(cache_dir: str | Path) -> list[RawOriginBundle]:
    cache_dir = Path(cache_dir)
    if not cache_dir.exists():
        return []
    bundles: list[RawOriginBundle] = []
    for path in sorted(cache_dir.glob("origin_*.pkl")):
        with path.open("rb") as fh:
            bundles.append(pickle.load(fh))
    bundles.sort(key=lambda b: b.origin_number)
    return bundles


def score_bundles(bundles: list[RawOriginBundle], config: dict) -> pd.DataFrame:
    """Re-run only the calibration/correction chain for `config` against cached raw bundles.

    Equivalent in shape to `run_rolling_origin_replay`'s output (one row per DT x origin, same
    `Replay_Calibration_*` columns), but skips all model training. Safe to call once per Optuna
    trial.
    """
    frames: list[pd.DataFrame] = []
    for bundle in bundles:
        artifacts = build_correction_artifacts(
            bundle.raw_calibration,
            config,
            extended_lookback_df=getattr(bundle, "extended_lookback_raw", None),
        )
        corrected = apply_origin_correction_chain(
            bundle.raw_origin,
            bundle.raw_weather_realism,
            bundle.raw_realized_scenarios,
            bundle.raw_weather_scenarios,
            config,
            artifacts,
        )
        if corrected.empty:
            continue
        corrected["Replay_Calibration_Days"] = bundle.calibration_days
        corrected["Replay_Calibration_Start_DT"] = (
            bundle.raw_calibration["DT"].min()
            if not bundle.raw_calibration.empty
            else pd.NaT
        )
        corrected["Replay_Calibration_End_DT"] = (
            bundle.raw_calibration["DT"].max()
            if not bundle.raw_calibration.empty
            else pd.NaT
        )
        frames.append(corrected)
    return (
        pd.concat(frames, ignore_index=True, sort=False) if frames else pd.DataFrame()
    )
