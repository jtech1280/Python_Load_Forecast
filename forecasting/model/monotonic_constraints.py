from __future__ import annotations

from typing import Any

# Features where load should never decrease as the feature value increases. These are
# heat/cooling-demand signals for a summer-peaking, AC-driven system: as they rise, air
# conditioning load rises too. Constraining them keeps tree splits from learning
# non-monotonic noise in the sparsely-populated extreme-heat bins that drive summer
# peak underforecast, and keeps predictions sane when extrapolating past the hottest
# day seen in training.
DEFAULT_INCREASING_FEATURES = [
    "Temperature", "Temperature_DailyMax", "Temperature_DailyMean",
    "CDD", "Daily_CDD", "Cooling_Stress",
    "Temp_Squared", "CDD_Squared",
    "Extreme_Heat_80", "Extreme_Heat_85", "Extreme_Heat_90", "Extreme_Heat_95", "Extreme_Heat_100",
    "HeatIndexF", "HeatIndex_CDD",
    "DailyMaxTempExcess90", "DailyMaxTempExcess95",
    "DailyMax_x_PeakHour", "DailyMax_x_PeakWindow14to18",
    "CDD_x_PeakWindow14to18", "CDD_x_HotPeakWindow16to20",
    "DailyMaxExcess90_x_PeakWindow14to18", "DailyMaxExcess90_x_HotPeakWindow16to20",
    "DailyMaxExcess95_x_HotPeakWindow16to20", "HeatIndexCDD_x_HotPeakWindow16to20",
    "Hot_Humid_Stress", "Humidity_x_Temp",
    "ConsecutiveHotDays90", "ConsecutiveVeryHotDays95", "ConsecutiveExtremeHotDays100",
    "HeatPersistenceStress90", "HeatPersistenceStress95", "DailyMax3DayMean_x_PeakHour",
    "OvernightHeatStress", "OvernightHeatStress_x_PeakHour",
]

DEFAULT_DECREASING_FEATURES: list[str] = []


def _cfg(config: dict | None, *keys, default=None):
    cur = config or {}
    for k in keys:
        if not isinstance(cur, dict) or k not in cur:
            return default
        cur = cur[k]
    return cur


def _as_bool(value, default: bool = False) -> bool:
    if value is None:
        return default
    if isinstance(value, bool):
        return value
    if isinstance(value, (int, float)):
        return bool(value)
    if isinstance(value, str):
        return value.strip().lower() in {"1", "true", "yes", "y", "on"}
    return default


def monotonic_constraints_enabled(config: dict | None) -> bool:
    return _as_bool(_cfg(config, "model", "monotonic_constraints", "enabled", default=True), default=True)


def build_monotone_vector(features: list[str], config: dict | None) -> tuple[int, ...] | None:
    """Build a per-feature monotonic direction vector aligned to ``features`` order.

    Returns ``None`` when constraints are disabled or no configured/default feature is
    present in ``features``, so callers can skip passing the parameter entirely.
    """
    if not monotonic_constraints_enabled(config):
        return None

    mc_cfg = _cfg(config, "model", "monotonic_constraints", default={}) or {}
    increasing = set(mc_cfg.get("increasing_features", DEFAULT_INCREASING_FEATURES) or [])
    decreasing = set(mc_cfg.get("decreasing_features", DEFAULT_DECREASING_FEATURES) or [])

    overlap = increasing & decreasing
    if overlap:
        raise ValueError(
            f"Features listed in both increasing_features and decreasing_features: {sorted(overlap)}"
        )
    if not increasing and not decreasing:
        return None

    vector = tuple(1 if f in increasing else (-1 if f in decreasing else 0) for f in features)
    return vector if any(vector) else None


def xgb_monotone_param(features: list[str], config: dict | None) -> Any:
    return build_monotone_vector(features, config)


def lgb_monotone_param(features: list[str], config: dict | None) -> list[int] | None:
    vector = build_monotone_vector(features, config)
    return list(vector) if vector is not None else None


def catboost_monotone_param(features: list[str], config: dict | None) -> list[int] | None:
    vector = build_monotone_vector(features, config)
    return list(vector) if vector is not None else None
