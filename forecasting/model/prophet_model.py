from __future__ import annotations

from dataclasses import dataclass
from typing import Any
import warnings

import numpy as np
import pandas as pd

try:  # Prophet is intentionally optional because it is a heavier dependency than XGBoost/LightGBM.
    from prophet import Prophet
except Exception:  # pragma: no cover - exercised only when dependency is absent
    Prophet = None  # type: ignore[assignment]

from forecasting.features.time_features import roseville_holidays


DEFAULT_PROPHET_REGRESSORS = [
    # Weather response known for the forecast horizon. Keep one representative per
    # highly collinear heat cluster so Prophet's standardized regressors stay stable.
    "Temperature_DailyMax",
    "CDD", "HDD", "Cooling_Stress",
    "Extreme_Heat_85", "Extreme_Heat_95", "Extreme_Heat_100",
    "DailyMaxTemp_Ramp_1Day", "DailyMinTemp_Ramp_1Day",
    "ConsecutiveVeryHotDays95", "ConsecutiveExtremeHotDays100",
    "OvernightHeatStress",
    "Humidity_Norm", "CloudCover_Norm", "WindSpeed_Mph", "PrecipIn", "Is_Raining",
    "Wind_x_Temp", "Rain_x_IsWeekend", "Hot_Humid_Stress",
    # Time/calendar/load-shape fields known in advance
    "Hour", "DOW", "Month", "DayOfYear", "WeekOfYear",
    "HourSin", "HourCos", "DOWSin", "DOWCos", "MonthSin", "MonthCos", "DayOfYearSin", "DayOfYearCos",
    "IsWeekend", "IsBusinessDay", "IsHoliday", "IsPreHoliday", "IsPostHoliday", "IsHolidayAdjacent",
    "IsMonday", "IsFriday", "IsSummerSeason", "IsWinterSeason", "IsOffPeak", "IsOnPeak", "IsSuperPeak", "IsLikelySystemPeakHour",
    # Solar / BTM fields known or proxied for the forecast horizon. Restore the broader
    # family for the controlled replay; the heat family remains pruned above.
    "Nameplate_MW", "Capacity_Ratio_To_Current", "Impact_Cap_MW", "Solar_Irradiance",
    "BTM_Solar_Proxy_MW", "Daily_BTM_Solar_Proxy_Total_MWh", "Daily_BTM_Solar_Proxy_Max_MW",
    "BTM_x_GHI",
    "BTM_x_Cloud", "Solar_Midday_Flag", "Solar_Evening_Ramp_Flag", "BTM_Evening_Ramp_Impact",
    "Solar_Hour_Shape", "Cloud_x_Solar_Hour", "Solar_Season_Factor", "ClearSky_Index",
    "ClearSky_GHI_Proxy_Wm2", "BTM_ClearSky_Proxy_MW",
    "BTM_Solar_Cloud_Adjusted_MW", "BTM_Solar_Loss_From_ClearSky_MW",
    "Cloud_x_GHI", "Cloud_x_ClearSky_GHI", "Daily_BTM_ClearSky_Max_MW",
    "Daily_BTM_Solar_Loss_MWh", "Daily_BTM_Solar_Loss_Max_MW", "Midday_Overcast_Solar_Loss_MW",
    "BTM_Midday_Impact", "Solar_Ramp_Down_1hr", "Solar_Ramp_Down_2hr", "Solar_Ramp_Up_1hr",
    "Humidity_x_Temp",
]


def _to_prophet_naive_datetime(values) -> pd.Series:
    """Return timezone-naive timestamps for Prophet.

    Prophet rejects timezone-aware `ds` values. For utility load forecasting, the
    desired seasonality is usually local wall-clock time, so this strips timezone
    metadata without converting the clock hour. Invalid values become NaT.
    """
    ser = pd.Series(values).copy()

    def _strip_one(v):
        if pd.isna(v):
            return pd.NaT
        try:
            ts = pd.Timestamp(v)
        except Exception:
            return pd.NaT
        if ts.tzinfo is not None:
            try:
                return ts.tz_localize(None)
            except Exception:
                try:
                    return ts.tz_convert(None)
                except Exception:
                    return pd.NaT
        return ts

    return pd.to_datetime(ser.map(_strip_one), errors="coerce")


@dataclass(frozen=True)
class ProphetFitResult:
    model: Any
    regressors: list[str]
    fill_values: dict[str, float]


def _cfg(config: dict | None, *keys, default=None):
    cur = config or {}
    for k in keys:
        if not isinstance(cur, dict) or k not in cur:
            return default
        cur = cur[k]
    return cur


def prophet_enabled(config: dict | None) -> bool:
    return bool(_cfg(config, "model", "prophet", "enabled", default=False))


def _available_regressors(df: pd.DataFrame, regressors: list[str]) -> list[str]:
    return [c for c in regressors if c in df.columns]


def _clean_regressor_frame(df: pd.DataFrame, regressors: list[str], fill_values: dict[str, float] | None = None) -> tuple[pd.DataFrame, dict[str, float]]:
    out = pd.DataFrame(index=df.index)
    fills: dict[str, float] = {} if fill_values is None else dict(fill_values)

    for col in regressors:
        if col in df.columns:
            s = pd.to_numeric(df[col], errors="coerce").replace([np.inf, -np.inf], np.nan)
        else:
            s = pd.Series(np.nan, index=df.index, dtype=float)

        if col not in fills:
            med = float(s.median()) if s.notna().any() else 0.0
            fills[col] = med if np.isfinite(med) else 0.0
        out[col] = s.fillna(fills[col]).astype(float)

    return out, fills


def _remove_constant_regressors(df: pd.DataFrame, regressors: list[str]) -> list[str]:
    keep: list[str] = []
    for col in regressors:
        if col not in df.columns:
            continue
        s = pd.to_numeric(df[col], errors="coerce").replace([np.inf, -np.inf], np.nan)
        if s.nunique(dropna=True) > 1:
            keep.append(col)
    return keep


def _make_roseville_holiday_df(start_year: int, end_year: int) -> pd.DataFrame:
    rows = []
    for year in range(int(start_year), int(end_year) + 1):
        for d in sorted(roseville_holidays(year)):
            rows.append({"holiday": "Roseville_Utility_Holiday", "ds": pd.Timestamp(d).normalize()})
    return pd.DataFrame(rows).drop_duplicates().reset_index(drop=True)


def make_prophet_model(config: dict | None, train_df: pd.DataFrame):
    if Prophet is None:
        raise RuntimeError(
            "Prophet is enabled in config.yaml but the Python package is not installed. "
            "Install it with: python -m pip install prophet"
        )

    p = _cfg(config, "model", "prophet", default={}) or {}
    dt = _to_prophet_naive_datetime(train_df["DT"])
    start_year = int(dt.min().year) - 1
    end_year = int(dt.max().year) + int(p.get("holiday_years_forward", 3))
    holidays = _make_roseville_holiday_df(start_year, end_year) if bool(p.get("add_roseville_holidays", True)) else None

    model = Prophet(
        growth=str(p.get("growth", "linear")),
        yearly_seasonality=p.get("yearly_seasonality", True),
        weekly_seasonality=p.get("weekly_seasonality", True),
        daily_seasonality=p.get("daily_seasonality", True),
        seasonality_mode=str(p.get("seasonality_mode", "multiplicative")),
        changepoint_prior_scale=float(p.get("changepoint_prior_scale", 0.08)),
        seasonality_prior_scale=float(p.get("seasonality_prior_scale", 6.0)),
        holidays_prior_scale=float(p.get("holidays_prior_scale", 5.0)),
        interval_width=float(p.get("interval_width", 0.80)),
        holidays=holidays,
    )

    if bool(p.get("add_monthly_seasonality", True)):
        model.add_seasonality(
            name="monthly",
            period=float(p.get("monthly_period_days", 30.5)),
            fourier_order=int(p.get("monthly_fourier_order", 5)),
        )
    if bool(p.get("add_annual_peak_seasonality", True)):
        model.add_seasonality(
            name="annual_peak_shape",
            period=float(p.get("annual_peak_period_days", 365.25)),
            fourier_order=int(p.get("annual_peak_fourier_order", 12)),
        )
    return model


def train_prophet(
    df: pd.DataFrame,
    regressors: list[str] | None = None,
    config: dict | None = None,
) -> ProphetFitResult | None:
    """Train an optional Prophet model with known future regressors.

    The tree models remain the primary short-term accuracy engines. Prophet adds a smoother trend/seasonality
    signal and can help stabilize odd periods where tree models overreact to lag/weather noise.
    """
    if not prophet_enabled(config):
        return None

    work = df.copy().sort_values("DT").reset_index(drop=True)
    work["DT"] = _to_prophet_naive_datetime(work["DT"])
    work["MWH"] = pd.to_numeric(work["MWH"], errors="coerce")
    work = work.dropna(subset=["DT", "MWH"])
    if len(work) < int(_cfg(config, "model", "prophet", "min_train_rows", default=24 * 60)):
        return None

    reg_candidates = _available_regressors(work, regressors or DEFAULT_PROPHET_REGRESSORS)
    reg_candidates = _remove_constant_regressors(work, reg_candidates)
    reg_frame, fill_values = _clean_regressor_frame(work, reg_candidates)

    prophet_df = pd.DataFrame({"ds": work["DT"], "y": work["MWH"].astype(float)})
    for col in reg_candidates:
        prophet_df[col] = reg_frame[col].values

    try:
        model = make_prophet_model(config, work)
    except Exception as exc:
        warnings.warn(
            "Prophet benchmark setup failed and will be skipped. "
            "The XGB/LGB production forecast can still run. "
            f"Reason: {exc}",
            RuntimeWarning,
        )
        return None

    p = _cfg(config, "model", "prophet", default={}) or {}
    reg_prior_scale = p.get("regressor_prior_scale", None)
    for col in reg_candidates:
        kwargs = {"standardize": bool(p.get("standardize_regressors", True))}
        if reg_prior_scale is not None:
            kwargs["prior_scale"] = float(reg_prior_scale)
        model.add_regressor(col, **kwargs)

    try:
        model.fit(prophet_df)
    except Exception as exc:
        warnings.warn(
            "Prophet benchmark training failed and will be skipped. "
            "The XGB/LGB production forecast can still run. "
            f"Reason: {exc}",
            RuntimeWarning,
        )
        return None

    setattr(model, "_forecasting_regressors", reg_candidates)
    setattr(model, "_forecasting_regressor_fill_values", fill_values)
    return ProphetFitResult(model=model, regressors=reg_candidates, fill_values=fill_values)


def predict_prophet(model_or_fit: Any | ProphetFitResult | None, df: pd.DataFrame, regressors: list[str] | None = None) -> pd.DataFrame:
    """Return Prophet yhat/yhat_lower/yhat_upper aligned to df rows."""
    if model_or_fit is None:
        return pd.DataFrame(index=df.index)

    if isinstance(model_or_fit, ProphetFitResult):
        model = model_or_fit.model
        reg_list = list(model_or_fit.regressors)
        fill_values = dict(model_or_fit.fill_values)
    else:
        model = model_or_fit
        reg_list = list(regressors or getattr(model, "_forecasting_regressors", []))
        fill_values = dict(getattr(model, "_forecasting_regressor_fill_values", {}))

    future = pd.DataFrame({"ds": _to_prophet_naive_datetime(df["DT"])}, index=df.index)
    reg_frame, _ = _clean_regressor_frame(df, reg_list, fill_values=fill_values)
    for col in reg_list:
        future[col] = reg_frame[col].values

    out = pd.DataFrame(index=df.index)
    out["Prophet_Pred_MWH"] = np.nan
    out["Prophet_Lower_MWH"] = np.nan
    out["Prophet_Upper_MWH"] = np.nan

    valid_mask = future["ds"].notna()
    if not valid_mask.any():
        return out

    try:
        fcst = model.predict(future.loc[valid_mask].reset_index(drop=True))
    except Exception as exc:
        warnings.warn(
            "Prophet benchmark prediction failed and will be skipped. "
            "The XGB/LGB production forecast can still run. "
            f"Reason: {exc}",
            RuntimeWarning,
        )
        return out

    valid_index = future.index[valid_mask]
    out.loc[valid_index, "Prophet_Pred_MWH"] = pd.to_numeric(fcst.get("yhat"), errors="coerce").clip(lower=0.0).to_numpy()
    if "yhat_lower" in fcst.columns:
        out.loc[valid_index, "Prophet_Lower_MWH"] = pd.to_numeric(fcst["yhat_lower"], errors="coerce").clip(lower=0.0).to_numpy()
    if "yhat_upper" in fcst.columns:
        out.loc[valid_index, "Prophet_Upper_MWH"] = pd.to_numeric(fcst["yhat_upper"], errors="coerce").clip(lower=0.0).to_numpy()
    return out
