from __future__ import annotations

"""Manual, rule-based correction for sustained heat streaks that go deeper than
any learned correction-chain stage reliably compensates for.

Every reactive/learned mechanism tested against the 2026-07-27..29 event
(heat_persistence_peak_capture's moderate tier, operational_residual_learner's
residual model -- even after extending its training window from 45 to 730
days) estimates at most a few MWH of correction on the worst rows, while the
raw ensemble's actual bias runs 15-30 MWH deep into the streak.
scripts/inspect_raw_bias_by_forecast_day.py showed that bias tracks
ConsecutiveVeryHotDays95 depth directly and identically regardless of forecast
lead time -- a relationship clean enough to state explicitly as a rule instead
of hoping a model infers it from a handful of historical exemplars of streaks
this deep.

This stage adds a correction that grows linearly with each day past a
configured onset depth, capped at a safety ceiling so an unprecedented future
streak doesn't extrapolate the rule into an ever-growing correction.
"""

import numpy as np
import pandas as pd


def _cfg(config: dict | None) -> dict:
    return ((config or {}).get("calibration", {}) or {}).get(
        "escalating_heat_persistence_correction", {}
    ) or {}


def _scope_mask(df: pd.DataFrame, cfg: dict) -> pd.Series:
    hour = pd.to_numeric(
        df.get("Hour", pd.Series(np.nan, index=df.index)), errors="coerce"
    )
    daily_max = pd.to_numeric(
        df.get("Temperature_DailyMax", pd.Series(np.nan, index=df.index)),
        errors="coerce",
    )
    hours = {int(h) for h in cfg.get("hours", [16, 17, 18, 19, 20])}
    min_maxtemp_f = float(cfg.get("min_maxtemp_f", 90.0))
    return hour.astype("Int64").isin(hours) & daily_max.ge(min_maxtemp_f)


def apply_escalating_heat_persistence_correction(
    df: pd.DataFrame,
    config: dict | None,
    *,
    forecast_col: str = "Final_Backtest_Forecast_MWH",
    also_update_cols: tuple[str, ...] = (),
) -> pd.DataFrame:
    out = df.copy()
    cfg = _cfg(config)
    if not bool(cfg.get("enabled", False)) or out.empty or forecast_col not in out.columns:
        return out

    onset_days = float(cfg.get("min_consecutive_very_hot_days95", 5.0))
    escalation_mwh_per_day = float(cfg.get("escalation_mwh_per_day", 1.5))
    max_correction_mwh = float(cfg.get("max_correction_mwh", 20.0))

    streak_depth = pd.to_numeric(
        out.get("ConsecutiveVeryHotDays95", pd.Series(np.nan, index=out.index)),
        errors="coerce",
    ).fillna(0.0)
    scope = _scope_mask(out, cfg) & streak_depth.gt(onset_days)

    correction = pd.Series(0.0, index=out.index)
    correction.loc[scope] = (
        (streak_depth.loc[scope] - onset_days) * escalation_mwh_per_day
    ).clip(upper=max_correction_mwh)

    out["Escalating_Heat_Persistence_Correction_MWH"] = correction
    out["Escalating_Heat_Persistence_Scope_Flag"] = scope.astype(int)

    forecast = pd.to_numeric(out[forecast_col], errors="coerce")
    out[forecast_col] = forecast + correction
    for col in also_update_cols:
        if col in out.columns:
            existing = pd.to_numeric(out[col], errors="coerce")
            out[col] = existing + correction

    return out
