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
(_origin_candidates, _origin_raw_forecasts, _replay_cfg, _as_int,
_worker_config_for_parallel_replay) to stay byte-for-byte consistent with the production
origin-selection and raw-forecast logic, and to reuse its already-proven per-origin
multiprocessing pattern for build_raw_origin_bundles's parallel path, rather than
reimplementing either. If that module's internals move, update the imports here.
"""

import os
import pickle
from dataclasses import dataclass
from multiprocessing import Pool, cpu_count
from pathlib import Path
from typing import Any

import pandas as pd

from forecasting.backtest.rolling_backtest import run_rolling_backtest
from forecasting.backtest.rolling_origin_replay import (
    _as_int,
    _origin_candidates,
    _origin_raw_forecasts,
    _replay_cfg,
    _serial_replay_required_for_catboost_gpu,
    _worker_config_for_parallel_replay,
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
    # calibration.rare_event_artifact_lookback_days_search -- a dedicated key so this
    # tool's cache-once-reuse-many-trials use case can carry its own value independently
    # of the live/replay default). None when unset -- matches
    # build_correction_artifacts' own default (no extra backtest, no behavior change).
    # Bundles pickled before this field existed won't have it; score_bundles() reads it
    # with getattr(..., None) so old caches still load, they just won't benefit from the
    # extended lookback until the cache is rebuilt.
    extended_lookback_raw: pd.DataFrame | None = None


def _build_one_origin_bundle(args: tuple) -> RawOriginBundle | None:
    """Build one origin's raw bundle. Module-level (picklable) so it can run either inline
    (sequential path) or as a multiprocessing.Pool worker (parallel path) -- see
    build_raw_origin_bundles. Mirrors rolling_origin_replay.py's _run_single_origin_replay:
    same idea (per-origin training is independent, so it's safe to parallelize across
    origins), applied here instead of to the production replay path.
    """
    (
        origin_number,
        origin_dt,
        n_origins,
        work,
        features,
        config,
        horizon_days,
        calibration_days,
        extended_lookback_days,
        skip_catboost,
        skip_calibration_prophet,
    ) = args
    print(
        f"[calibration_search] building raw bundle for origin {origin_number}/{n_origins}: {origin_dt}",
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
            f"for origin {origin_number}/{n_origins}: {origin_dt}",
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
        return None
    return RawOriginBundle(
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


def _raw_origin_pool_args(
    train_df: pd.DataFrame,
    features: list[str],
    config: dict,
    *,
    origin_limit: int | None = None,
) -> tuple[list[tuple], bool, int]:
    if train_df is None or train_df.empty:
        return [], False, 1

    cfg = _replay_cfg(config)
    horizon_days = _as_int(cfg.get("horizon_days"), 16)
    calibration_days = _as_int(cfg.get("calibration_days"), 45)
    skip_catboost = bool(cfg.get("skip_catboost", False))
    skip_calibration_prophet = bool(cfg.get("skip_calibration_prophet", False))

    work = train_df.copy().sort_values("DT").reset_index(drop=True)
    origins = _origin_candidates(work, config)
    if origin_limit is not None:
        origins = origins[: int(origin_limit)]

    extended_lookback_days = rare_event_artifact_lookback_days(
        config,
        calibration_days,
        config_key="rare_event_artifact_lookback_days_search",
    )

    parallel_cfg = (cfg.get("parallel", {}) or {}) if isinstance(cfg, dict) else {}
    parallel_enabled = bool(parallel_cfg.get("enabled", True)) and len(origins) > 1
    num_processes = parallel_cfg.get("processes")
    if not isinstance(num_processes, int) or num_processes <= 0:
        try:
            num_processes = max(1, cpu_count() // 2)
        except NotImplementedError:
            num_processes = 2
    num_processes = min(num_processes, max(1, len(origins)))

    if (
        parallel_enabled
        and num_processes > 1
        and _serial_replay_required_for_catboost_gpu(
            config, skip_catboost, parallel_cfg
        )
    ):
        print(
            "CatBoost GPU is enabled; building raw origin bundles sequentially "
            "to avoid native CatBoost GPU memory aborts from concurrent workers.",
            flush=True,
        )
        parallel_enabled = False
        num_processes = 1

    run_config = config
    if parallel_enabled and num_processes > 1:
        run_config = _worker_config_for_parallel_replay(config, num_processes)

    pool_args = [
        (
            origin_number,
            origin_dt,
            len(origins),
            work,
            features,
            run_config,
            horizon_days,
            calibration_days,
            extended_lookback_days,
            skip_catboost,
            skip_calibration_prophet,
        )
        for origin_number, origin_dt in enumerate(origins, start=1)
    ]
    return pool_args, parallel_enabled, num_processes


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

    Each origin's three training passes (calibration backtest, rare-event extended lookback,
    origin raw/weather-realism/scenario forecasts) are independent of every other origin, the
    same property that already makes rolling_origin_replay.py's production replay path safe to
    parallelize across origins. This reuses that same training.rolling_origin_replay.parallel.*
    config (enabled/processes) and CPU-thread-division safeguard, so a cache build picks up
    whatever parallelism you already have configured for replay -- e.g. the gpu_ram_part
    setting that lets concurrent CatBoost GPU training share VRAM safely applies here too.
    """
    pool_args, parallel_enabled, num_processes = _raw_origin_pool_args(
        train_df, features, config, origin_limit=origin_limit
    )
    if not pool_args:
        return []

    if not parallel_enabled or num_processes <= 1:
        results = [_build_one_origin_bundle(arg) for arg in pool_args]
    else:
        print(
            f"[calibration_search] building {len(pool_args)} raw origin bundles in parallel "
            f"on {num_processes} processes...",
            flush=True,
        )
        with Pool(processes=num_processes) as pool:
            results = pool.map(_build_one_origin_bundle, pool_args)

    return [bundle for bundle in results if bundle is not None]


def _bundle_path(cache_dir: Path, bundle: RawOriginBundle) -> Path:
    date_tag = pd.Timestamp(bundle.origin_dt).strftime("%Y%m%d")
    return cache_dir / f"origin_{bundle.origin_number:03d}_{date_tag}.pkl"


def _save_one_raw_origin_bundle(bundle: RawOriginBundle, cache_dir: str | Path) -> Path:
    cache_dir = Path(cache_dir)
    cache_dir.mkdir(parents=True, exist_ok=True)
    path = _bundle_path(cache_dir, bundle)
    tmp_path = path.with_name(f"{path.name}.tmp.{os.getpid()}")
    with tmp_path.open("wb") as fh:
        pickle.dump(bundle, fh, protocol=pickle.HIGHEST_PROTOCOL)
    tmp_path.replace(path)
    return path


def _build_and_save_one_origin_bundle(args: tuple) -> str | None:
    origin_args, cache_dir = args
    bundle = _build_one_origin_bundle(origin_args)
    if bundle is None:
        return None
    path = _save_one_raw_origin_bundle(bundle, cache_dir)
    print(f"[calibration_search] saved raw origin bundle: {path}", flush=True)
    return str(path)


def build_raw_origin_bundle_cache(
    train_df: pd.DataFrame,
    features: list[str],
    config: dict,
    cache_dir: str | Path,
    *,
    origin_limit: int | None = None,
) -> list[Path]:
    """Build and persist raw origin bundles, writing each bundle as soon as it completes.

    This is the preferred path for long cache-build CLI runs. Returning full
    RawOriginBundle objects through a multiprocessing queue can move hundreds of
    MB of DataFrames through the parent process before anything is written to
    disk. Here each worker writes its own bundle and returns only the path, so a
    late failure or pool issue does not discard hours of completed origins.
    """
    pool_args, parallel_enabled, num_processes = _raw_origin_pool_args(
        train_df, features, config, origin_limit=origin_limit
    )
    if not pool_args:
        return []

    cache_dir = Path(cache_dir)
    cache_dir.mkdir(parents=True, exist_ok=True)
    worker_args = [(arg, str(cache_dir)) for arg in pool_args]
    paths: list[Path] = []

    if not parallel_enabled or num_processes <= 1:
        for arg in worker_args:
            path = _build_and_save_one_origin_bundle(arg)
            if path is not None:
                paths.append(Path(path))
    else:
        print(
            f"[calibration_search] building {len(pool_args)} raw origin bundles in parallel "
            f"on {num_processes} processes and writing each bundle as it completes...",
            flush=True,
        )
        with Pool(processes=num_processes) as pool:
            for path in pool.imap_unordered(
                _build_and_save_one_origin_bundle, worker_args, chunksize=1
            ):
                if path is not None:
                    paths.append(Path(path))
                    print(
                        f"[calibration_search] collected saved bundle {len(paths)}/{len(pool_args)}",
                        flush=True,
                    )

    return sorted(paths)


def save_raw_origin_bundles(
    bundles: list[RawOriginBundle], cache_dir: str | Path
) -> list[Path]:
    paths = []
    for bundle in bundles:
        paths.append(_save_one_raw_origin_bundle(bundle, cache_dir))
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
