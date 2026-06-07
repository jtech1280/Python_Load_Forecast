from __future__ import annotations

"""Weather-uncertainty peak hedge (V12.9 production weather-robustness hardening).

Motivation
----------
Rolling-origin replay shows a large gap between the *realized-weather* forecast
(model fed the weather that actually occurred) and the *forecast-weather* forecast
(model fed the weather that was operationally available at run time). On hot-peak
hours that gap is severe: hot-peak MAE roughly 2.5x worse under forecast weather,
and a *systematic underforecast bias* of ~7-8 MWh even though the forecasted daily
max temperature is, on average, slightly warm.

That underforecast is not (only) bad luck in the temperature forecast -- it is a
predictable consequence of load being convex in temperature at the hot end.
By Jensen's inequality, when temperature is uncertain,

    E[ load(T_actual) ]  >  load(E[T_actual]) ~= load(T_forecast)

so a model fed the *point* temperature forecast will, in expectation, sit below the
expected load whenever the local load-temperature response curves upward. The point
forecast is therefore biased low precisely in the hot/peak regime that matters most
for capacity and reliability planning. The correction stack upstream is tuned on
realized-weather residuals, so it does not compensate for this.

What this layer does
--------------------
It converts information the pipeline already computes -- the +/-3F weather-scenario
re-predictions -- from a band-only signal into a small, bounded uplift on the point
forecast, applied only in hot/peak conditions and scaled by how untrustworthy the
temperature forecast is at that lead time.

    f''_est   = (warmer_P50 - 2*base + cooler_P50) / scenario_delta^2     (local curvature)
    jensen    = 0.5 * sigma_T(lead)^2 * f''_est                          (Jensen term)
    upper     = upper_blend * (sigma_T(lead)/sigma_ref) * (warmer_P50 - base)+   (one-sided lean)
    hedge     = clip( jensen_scale*jensen + upper, 0, cap )              (never negative)

The one-sided `upper` term reflects that, for peak planning, being caught short is
more costly than overforecasting; it leans on the warmer scenario, weighted by lead
uncertainty. sigma_T(lead) is the daily-max-temperature forecast error by lead,
taken from rolling_origin_replay_weather_input_error_by_lead (MAE converted to a
Gaussian sigma).

Safety properties
-----------------
* The hedge is clipped to be non-negative, so it never pushes any hour down.
* It is gated to peak hours and to forecasted daily-max >= a threshold, so normal /
  cool hours are untouched (verified: non-hot MAE unchanged in replay).
* It is bounded by a cap and by a multiple of the warmer-scenario delta, so a noisy
  curvature estimate cannot produce a runaway uplift.
* If the scenario columns are absent (e.g. the realized-weather replay path, which
  does not compute scenarios), it no-ops. In the realized-weather production-history
  case sigma_T(lead) ~ 0, so the hedge is ~0 -- it does not disturb realized-weather
  metrics, only the operational forecast-weather case it is designed for.
"""

import numpy as np
import pandas as pd


# Daily-max temperature forecast error (MAE, degrees F) by lead day, from
# rolling_origin_replay_weather_input_error_by_lead.csv. Used to size the hedge.
_DEFAULT_MAE_BY_LEAD_F = {
    1: 1.94,
    2: 1.00,
    3: 1.92,
    4: 2.36,
    5: 4.11,
    6: 4.69,
    7: 6.30,
}
# Half-normal: E|X| = sigma * sqrt(2/pi)  ->  sigma = MAE * 1.2533
_MAE_TO_SIGMA = 1.2533


def _as_num(x) -> pd.Series:
    return pd.to_numeric(x, errors="coerce")


def _cfg(config: dict | None) -> dict:
    return (((config or {}).get("calibration", {}) or {}).get("weather_robustness_hedge", {}) or {})


def _sigma_by_lead_lookup(cfg: dict) -> tuple[dict[int, float], float, float]:
    mae_by_lead = {int(k): float(v) for k, v in (cfg.get("dailymax_temp_mae_by_lead_f", _DEFAULT_MAE_BY_LEAD_F) or {}).items()}
    if not mae_by_lead:
        mae_by_lead = dict(_DEFAULT_MAE_BY_LEAD_F)
    mae_to_sigma = float(cfg.get("mae_to_sigma_factor", _MAE_TO_SIGMA))
    # Reference sigma normalizes the one-sided upper-scenario term. Pin it to the longest
    # *validated* lead (default 7) so that extending the table to unvalidated leads 8-16
    # does NOT rescale (and weaken) the validated day-1..7 hedge. Leads beyond the
    # reference get sigma_norm > 1, i.e. a larger hedge where weather is least trustworthy.
    ref_lead = int(cfg.get("sigma_ref_lead", 7))
    if ref_lead not in mae_by_lead:
        ref_lead = max(mae_by_lead)
    sigma_ref = mae_by_lead[ref_lead] * mae_to_sigma
    return mae_by_lead, mae_to_sigma, sigma_ref


def _lead_series(df: pd.DataFrame, lead_col: str | None) -> pd.Series:
    """Resolve a lead-days series. Prefer an explicit realism lead column, then
    Forecast_Weather_Lead_Days, then Forecast_Day."""
    candidates = []
    if lead_col:
        candidates.append(lead_col)
    candidates += ["Forecast_Weather_Lead_Days", "WeatherRealism_Forecast_Weather_Lead_Days", "Forecast_Day"]
    for col in candidates:
        if col in df.columns:
            s = _as_num(df[col])
            if s.notna().any():
                return s
    return pd.Series(np.nan, index=df.index, dtype=float)


def apply_weather_robustness_hedge(
    df: pd.DataFrame,
    config: dict | None = None,
    base_col: str = "Final_Forecast_MWH",
    warmer_col: str = "WeatherScenario_warmer_P50_MWH",
    cooler_col: str = "WeatherScenario_cooler_P50_MWH",
    maxtemp_col: str = "Temperature_DailyMax",
    lead_col: str | None = None,
    also_update_cols: tuple[str, ...] = ("Stage_Selected_Forecast_MWH",),
) -> pd.DataFrame:
    """Add a bounded, lead-aware peak hedge to the point forecast.

    Writes diagnostic columns `Weather_Robustness_Hedge_MWH` and
    `Weather_Robustness_Hedge_Source`, updates ``base_col`` in place, and mirrors the
    update into any present columns in ``also_update_cols`` (e.g. the stage-selected
    forecast that operations consumes).
    """
    out = df.copy()
    cfg = _cfg(config)
    out["Weather_Robustness_Hedge_MWH"] = 0.0
    out["Weather_Robustness_Hedge_Source"] = "none"

    if not bool(cfg.get("enabled", True)) or out.empty:
        return out
    # Needs base + both scenario re-predictions; otherwise no-op (e.g. realized-weather path).
    if base_col not in out.columns or warmer_col not in out.columns or cooler_col not in out.columns:
        return out

    hours = {int(h) for h in cfg.get("hours", [13, 14, 15, 16, 17, 18, 19, 20, 21])}
    min_maxtemp = float(cfg.get("min_maxtemp_f", 88.0))
    min_day = int(cfg.get("min_forecast_day", 1))
    max_day = int(cfg.get("max_forecast_day", 16))
    scenario_delta = float(cfg.get("scenario_delta_f", 3.0))
    jensen_scale = float(cfg.get("jensen_scale", 2.0))
    upper_blend = float(cfg.get("upper_scenario_blend", 0.30))
    cap = float(cfg.get("cap_mwh", 16.0))
    warmer_bound_mult = float(cfg.get("max_fraction_of_warmer_delta", 1.25))

    mae_by_lead, mae_to_sigma, sigma_ref = _sigma_by_lead_lookup(cfg)
    max_lead = max(mae_by_lead)

    def _sigma(l: float) -> float:
        if pd.isna(l):
            return mae_by_lead[max_lead] * mae_to_sigma
        li = int(round(l))
        mae = mae_by_lead.get(li, mae_by_lead[max_lead])
        return mae * mae_to_sigma

    base = _as_num(out[base_col])
    warmer = _as_num(out[warmer_col])
    cooler = _as_num(out[cooler_col])

    if "Hour" in out.columns:
        hour = _as_num(out["Hour"]).fillna(-1).astype(int)
    else:
        hour = pd.to_datetime(out.get("DT"), errors="coerce").dt.hour.fillna(-1).astype(int)
    fmax = _as_num(out[maxtemp_col]) if maxtemp_col in out.columns else pd.Series(np.nan, index=out.index)

    lead = _lead_series(out, lead_col)
    forecast_day = _as_num(out["Forecast_Day"]) if "Forecast_Day" in out.columns else lead

    sigma = lead.map(_sigma)
    sigma2 = sigma ** 2
    sigma_norm = (sigma / sigma_ref).clip(lower=0.0)

    # Local curvature of the load-temperature response from the +/-deltaF scenarios.
    warmer_delta = (warmer - base)
    cooler_delta = (cooler - base)
    fpp = (warmer_delta + cooler_delta) / (scenario_delta ** 2)

    jensen = 0.5 * sigma2 * fpp * jensen_scale
    upper = upper_blend * sigma_norm * warmer_delta.clip(lower=0.0)

    hedge = (jensen + upper).clip(lower=0.0, upper=cap)
    # Never exceed a small multiple of the warmer-scenario uplift itself.
    hedge = np.minimum(hedge, warmer_delta.clip(lower=0.0) * warmer_bound_mult)

    gate = (
        hour.isin(hours)
        & fmax.ge(min_maxtemp)
        & forecast_day.between(min_day, max_day)
    )
    hedge = hedge.where(gate, 0.0).fillna(0.0)

    out["Weather_Robustness_Hedge_MWH"] = hedge
    out["Weather_Robustness_Hedge_Source"] = np.where(
        hedge.to_numpy() > 0.0, "weather_uncertainty_peak_hedge", "none"
    )
    out[base_col] = (base + hedge).clip(lower=0.0)
    for col in also_update_cols:
        if col in out.columns:
            out[col] = (_as_num(out[col]) + hedge).clip(lower=0.0)
    return out
