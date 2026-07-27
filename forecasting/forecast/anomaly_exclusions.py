from __future__ import annotations

"""Pure exclusion of anomalous load intervals (e.g. DER dispatch, telemetry gaps).

Some historical hours do not reflect the natural weather->load relationship the model
is trying to learn. The clearest example is a demand-response / DER dispatch, where the
*metered* load is deliberately pushed away from native demand for a few peak hours. Using
those hours as residual/calibration targets teaches the correction chain the wrong level,
and scoring against them produces spurious peak "misses".

This module implements *pure exclusion*: the configured intervals are dropped from the
residual-learning inputs and from the replay scorecards. It does not attempt to repair the
underlying actual (that would need the dispatched-MW telemetry). Events are described the
way operators describe them - a local calendar date plus an inclusive hour-ENDING (HE)
range. HE ``h`` maps to the project's hour-beginning ``Hour`` value ``h - 1`` (HE 17 is the
16:00-17:00 interval, i.e. ``Hour == 16``).
"""

import pandas as pd


def _exclusion_events(config: dict | None) -> list[dict]:
    cfg = ((config or {}).get("anomaly_exclusions", {}) or {})
    if not bool(cfg.get("enabled", True)):
        return []
    events = cfg.get("events", []) or []
    return [ev for ev in events if isinstance(ev, dict) and ev.get("date") is not None]


def _hour_beginning_bounds(event: dict) -> tuple[int, int]:
    """Return the inclusive hour-beginning bounds for an event's HE range.

    A missing ``he_start``/``he_end`` means the whole day (hour-beginning 0..23).
    """
    he_start = event.get("he_start")
    he_end = event.get("he_end")
    if he_start is None and he_end is None:
        return 0, 23
    hb_start = int(he_start) - 1 if he_start is not None else 0
    hb_end = int(he_end) - 1 if he_end is not None else 23
    lo, hi = sorted((hb_start, hb_end))
    return max(0, lo), min(23, hi)


def excluded_interval_mask(df: pd.DataFrame, config: dict | None, dt_col: str = "DT") -> pd.Series:
    """Boolean mask (aligned to ``df.index``) of rows inside a configured exclusion window."""
    mask = pd.Series(False, index=df.index if df is not None else None, dtype=bool)
    events = _exclusion_events(config)
    if df is None or df.empty or not events or dt_col not in df.columns:
        return mask

    raw_dt = df[dt_col]
    try:
        dt = pd.to_datetime(raw_dt, errors="coerce")
    except ValueError:
        # CSV exports can contain mixed DST offsets. Exclusion windows are configured
        # as local wall-clock dates/hours, so strip the offset rather than converting
        # to UTC and shifting the local hour.
        cleaned = raw_dt.astype(str).str.strip().str.replace(r"(?:[+-]\d{2}:?\d{2}|Z)$", "", regex=True)
        dt = pd.to_datetime(cleaned, errors="coerce")
    if getattr(dt.dt, "tz", None) is not None:
        # Exclusion windows are local calendar dates/hours; forecast/replay DT is local tz-aware.
        dt = dt.dt.tz_localize(None)
    day = dt.dt.normalize()
    hour = dt.dt.hour

    for event in events:
        try:
            event_day = pd.Timestamp(event["date"]).normalize()
        except (ValueError, TypeError):
            continue
        lo, hi = _hour_beginning_bounds(event)
        mask |= day.eq(event_day) & hour.between(lo, hi)

    return mask.fillna(False)


def drop_excluded_intervals(df: pd.DataFrame, config: dict | None, dt_col: str = "DT") -> pd.DataFrame:
    """Return ``df`` without the rows that fall inside a configured exclusion window."""
    if df is None or df.empty:
        return df
    mask = excluded_interval_mask(df, config, dt_col=dt_col)
    if not mask.any():
        return df
    return df.loc[~mask].copy()
