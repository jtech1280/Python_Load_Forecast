from __future__ import annotations

"""Manual, rule-based correction for sudden, isolated extreme-heat spikes that
escalating_heat_persistence_correction structurally cannot reach.

Motivation: a real ~2-week production run (Ada server, 2026-09-22) surfaced a
distinct failure mode from the one this project spent a full session
diagnosing and fixing. The 2026-07-27..08-10 event was a long, gradually
building streak -- escalating_heat_persistence_correction (keyed on
ConsecutiveVeryHotDays95 depth) closed most of its raw bias. But the worst
post-correction miss in that same production run was 2026-09-09..10 (100.7F
and 103.8F, 29.9 MWH miss at the single worst hour) -- a sudden, isolated 1-2
day spike (Sep 7 was only 89F) that never builds any streak depth at all
(ConsecutiveVeryHotDays95 peaked at 2, well under the persistence rule's
4-day onset), so that rule never fires for it regardless of tuning.

This applies a second, independent rule: once Temperature_DailyMax crosses a
high absolute threshold, add a correction that grows with degrees above that
threshold, capped at a safety ceiling. It's deliberately scoped to *low*
persistence depth (<= the companion rule's onset) so the two rules partition
the space rather than stack unpredictably on a day that is both deep in a
streak and extremely hot -- the persistence rule already owns that case.
"""

import numpy as np
import pandas as pd


def _cfg(config: dict | None) -> dict:
    return ((config or {}).get("calibration", {}) or {}).get(
        "extreme_heat_magnitude_correction", {}
    ) or {}


def _scope_mask(df: pd.DataFrame, cfg: dict) -> pd.Series:
    hour = pd.to_numeric(
        df.get("Hour", pd.Series(np.nan, index=df.index)), errors="coerce"
    )
    daily_max = pd.to_numeric(
        df.get("Temperature_DailyMax", pd.Series(np.nan, index=df.index)),
        errors="coerce",
    )
    streak_depth = pd.to_numeric(
        df.get("ConsecutiveVeryHotDays95", pd.Series(np.nan, index=df.index)),
        errors="coerce",
    ).fillna(0.0)
    hours = {int(h) for h in cfg.get("hours", [16, 17, 18, 19, 20])}
    min_maxtemp_f = float(cfg.get("min_maxtemp_f", 98.0))
    max_streak_depth = float(cfg.get("max_consecutive_very_hot_days95", 4.0))
    return (
        hour.astype("Int64").isin(hours)
        & daily_max.ge(min_maxtemp_f)
        & streak_depth.le(max_streak_depth)
    )


def apply_extreme_heat_magnitude_correction(
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

    min_maxtemp_f = float(cfg.get("min_maxtemp_f", 98.0))
    escalation_mwh_per_degree = float(cfg.get("escalation_mwh_per_degree", 4.0))
    max_correction_mwh = float(cfg.get("max_correction_mwh", 30.0))

    daily_max = pd.to_numeric(
        out.get("Temperature_DailyMax", pd.Series(np.nan, index=out.index)),
        errors="coerce",
    ).fillna(0.0)
    scope = _scope_mask(out, cfg)

    correction = pd.Series(0.0, index=out.index)
    correction.loc[scope] = (
        (daily_max.loc[scope] - min_maxtemp_f) * escalation_mwh_per_degree
    ).clip(upper=max_correction_mwh)

    out["Extreme_Heat_Magnitude_Correction_MWH"] = correction
    out["Extreme_Heat_Magnitude_Scope_Flag"] = scope.astype(int)

    forecast = pd.to_numeric(out[forecast_col], errors="coerce")
    out[forecast_col] = forecast + correction
    for col in also_update_cols:
        if col in out.columns:
            existing = pd.to_numeric(out[col], errors="coerce")
            out[col] = existing + correction

    return out
