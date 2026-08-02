from __future__ import annotations

import numpy as np
import pandas as pd
from zoneinfo import ZoneInfo


INTRADAY_LOAD_FEATURES = [
    "FiveMin_Load_Available",
    "FiveMin_Data_Age_Hours",
    "FiveMin_PrevHour_Avg_MW",
    "FiveMin_PrevHour_Max_MW",
    "FiveMin_PrevHour_Min_MW",
    "FiveMin_PrevHour_Last_MW",
    "FiveMin_PrevHour_Range_MW",
    "FiveMin_PrevHour_Ramp_MW",
    "FiveMin_PrevHour_Count",
    "FiveMin_Ramp_15Min_MW",
    "FiveMin_Ramp_30Min_MW",
    "FiveMin_Ramp_60Min_MW",
]


def build_hourly_load_from_five_min(
    five_min_df: pd.DataFrame,
    *,
    timezone: str = "America/Los_Angeles",
    min_intervals_per_hour: int = 10,
) -> pd.DataFrame:
    """Aggregate completed 5-minute MW samples into hourly MWh-equivalent load.

    The source values are MW interval readings. For an hourly load series, the average MW
    across a completed hour is the MWh-equivalent value for that hour. The official hourly
    load feed is aligned to the completed hour label, so a 14:00-15:00 five-minute block is
    exported with DT=15:00.
    """
    if five_min_df is None or five_min_df.empty or not {"DT", "FiveMin_Load_MW"}.issubset(five_min_df.columns):
        return pd.DataFrame()

    tz = ZoneInfo(str(timezone))
    work = five_min_df[["DT", "FiveMin_Load_MW"]].copy()
    work["DT"] = pd.to_datetime(work["DT"], errors="coerce", utc=True).dt.tz_convert(tz)
    work["FiveMin_Load_MW"] = pd.to_numeric(work["FiveMin_Load_MW"], errors="coerce")
    work = work.dropna(subset=["DT", "FiveMin_Load_MW"]).sort_values("DT")
    if work.empty:
        return pd.DataFrame()

    work["HourDT"] = work["DT"].dt.floor("h")
    grouped = work.groupby("HourDT")
    hourly = grouped["FiveMin_Load_MW"].agg(
        MWH="mean",
        FiveMin_Interval_Count="count",
        FiveMin_Hourly_Min_MW="min",
        FiveMin_Hourly_Max_MW="max",
        FiveMin_Hourly_Last_MW="last",
    ).reset_index()
    last_start = grouped["DT"].max().reset_index(name="FiveMin_Last_Interval_Start")
    hourly = hourly.merge(last_start, on="HourDT", how="left")
    hourly["FiveMin_Hourly_Range_MW"] = hourly["FiveMin_Hourly_Max_MW"] - hourly["FiveMin_Hourly_Min_MW"]
    hourly["FiveMin_Hour_End"] = hourly["HourDT"] + pd.Timedelta(hours=1)
    complete = hourly["FiveMin_Last_Interval_Start"] >= (hourly["HourDT"] + pd.Timedelta(minutes=55))
    enough = pd.to_numeric(hourly["FiveMin_Interval_Count"], errors="coerce").ge(int(min_intervals_per_hour))
    hourly = hourly[complete & enough].copy()
    if hourly.empty:
        return pd.DataFrame()

    hourly["DT"] = hourly["FiveMin_Hour_End"]
    hourly["Load_Source"] = "five_min_completed_hour"
    keep = [
        "DT",
        "MWH",
        "Load_Source",
        "FiveMin_Interval_Count",
        "FiveMin_Hourly_Min_MW",
        "FiveMin_Hourly_Max_MW",
        "FiveMin_Hourly_Last_MW",
        "FiveMin_Hourly_Range_MW",
        "FiveMin_Last_Interval_Start",
        "FiveMin_Hour_End",
    ]
    return hourly[keep].sort_values("DT").reset_index(drop=True)


def append_recent_five_min_hourly_load(
    load_df: pd.DataFrame,
    five_min_hourly: pd.DataFrame,
    *,
    replace_overlap_hours: int = 0,
) -> pd.DataFrame:
    """Append completed 5-minute-derived hourly rows beyond the official hourly feed."""
    if load_df is None or load_df.empty or five_min_hourly is None or five_min_hourly.empty:
        return load_df

    base = load_df.copy()
    if "Load_Source" not in base.columns:
        base["Load_Source"] = "hourly_history"
    base["DT"] = pd.to_datetime(base["DT"], errors="coerce")
    recent = five_min_hourly.copy()
    recent["DT"] = pd.to_datetime(recent["DT"], errors="coerce")
    base = base.dropna(subset=["DT"])
    recent = recent.dropna(subset=["DT", "MWH"])
    if base.empty or recent.empty:
        return base

    latest_official = base["DT"].max()
    overlap = max(0, int(replace_overlap_hours or 0))
    cutoff = latest_official - pd.Timedelta(hours=overlap)
    recent = recent[recent["DT"] > cutoff].copy()
    if recent.empty:
        return base.sort_values("DT").reset_index(drop=True)

    combined = pd.concat([base, recent], ignore_index=True, sort=False)
    combined.sort_values(["DT", "Load_Source"], inplace=True)
    # Recent 5-minute rows are appended after official rows, so keep='last' lets a
    # configured overlap replace stale official hours.
    combined = combined.drop_duplicates(subset=["DT"], keep="last")
    return combined.sort_values("DT").reset_index(drop=True)


def build_intraday_load_feature_frame(five_min_df: pd.DataFrame) -> pd.DataFrame:
    if five_min_df is None or five_min_df.empty or not {"DT", "FiveMin_Load_MW"}.issubset(five_min_df.columns):
        return pd.DataFrame()

    work = five_min_df[["DT", "FiveMin_Load_MW"]].copy()
    work["DT"] = pd.to_datetime(work["DT"], errors="coerce", utc=True)
    work["FiveMin_Load_MW"] = pd.to_numeric(work["FiveMin_Load_MW"], errors="coerce")
    work = work.dropna(subset=["DT", "FiveMin_Load_MW"]).sort_values("DT")
    if work.empty:
        return pd.DataFrame()

    work["HourDT"] = work["DT"].dt.floor("h")
    grouped = work.groupby("HourDT")
    hourly = grouped["FiveMin_Load_MW"].agg(
        FiveMin_PrevHour_Avg_MW="mean",
        FiveMin_PrevHour_Max_MW="max",
        FiveMin_PrevHour_Min_MW="min",
        FiveMin_PrevHour_Last_MW="last",
        FiveMin_PrevHour_Count="count",
    ).reset_index()
    hourly["FiveMin_PrevHour_Range_MW"] = hourly["FiveMin_PrevHour_Max_MW"] - hourly["FiveMin_PrevHour_Min_MW"]
    first_load = grouped["FiveMin_Load_MW"].first().reset_index(name="FiveMin_PrevHour_First_MW")
    hourly = hourly.merge(first_load, on="HourDT", how="left")
    hourly["FiveMin_PrevHour_Ramp_MW"] = hourly["FiveMin_PrevHour_Last_MW"] - hourly["FiveMin_PrevHour_First_MW"]
    hourly["DT"] = hourly["HourDT"] + pd.Timedelta(hours=2)

    tail = work.set_index("DT")["FiveMin_Load_MW"].sort_index()
    ramp_rows = []
    for hour_start in hourly["HourDT"]:
        feature_asof = hour_start + pd.Timedelta(hours=1)
        asof = tail[tail.index < feature_asof]
        if asof.empty:
            ramp_rows.append((np.nan, np.nan, np.nan))
            continue
        last = float(asof.iloc[-1])
        def _lag_delta(minutes: int) -> float:
            cutoff = feature_asof - pd.Timedelta(minutes=minutes)
            prior = asof[asof.index <= cutoff]
            if prior.empty:
                return np.nan
            return last - float(prior.iloc[-1])
        ramp_rows.append((_lag_delta(15), _lag_delta(30), _lag_delta(60)))
    ramps = pd.DataFrame(ramp_rows, columns=["FiveMin_Ramp_15Min_MW", "FiveMin_Ramp_30Min_MW", "FiveMin_Ramp_60Min_MW"])
    hourly = pd.concat([hourly.reset_index(drop=True), ramps], axis=1)
    hourly["FiveMin_Load_Available"] = 1.0
    hourly["FiveMin_Data_Age_Hours"] = 0.0

    keep = ["DT"] + INTRADAY_LOAD_FEATURES
    return hourly[[c for c in keep if c in hourly.columns]].sort_values("DT").reset_index(drop=True)


def merge_intraday_load_features(
    frame: pd.DataFrame,
    intraday_features: pd.DataFrame,
    *,
    allow_carry_forward: bool = False,
    max_carry_forward_hours: int = 24,
) -> pd.DataFrame:
    out = frame.copy()
    for col in INTRADAY_LOAD_FEATURES:
        if col not in out.columns:
            out[col] = 0.0

    if intraday_features is None or intraday_features.empty or "DT" not in out.columns:
        return out

    left = out.copy()
    left["__DT_KEY"] = pd.to_datetime(left["DT"], errors="coerce", utc=True)
    feats = intraday_features.copy()
    feats["__DT_KEY"] = pd.to_datetime(feats["DT"], errors="coerce", utc=True)
    feats = feats.dropna(subset=["__DT_KEY"]).sort_values("__DT_KEY")
    if feats.empty:
        out["FiveMin_Load_Available"] = 0.0
        return out

    feature_cols = [c for c in INTRADAY_LOAD_FEATURES if c in feats.columns]
    if allow_carry_forward:
        left_sorted = left.drop(columns=feature_cols, errors="ignore").reset_index().sort_values("__DT_KEY")
        right = feats[["__DT_KEY"] + feature_cols].rename(columns={"__DT_KEY": "__FEATURE_DT"})
        right_sorted = right.sort_values("__FEATURE_DT")
        merged = pd.merge_asof(
            left_sorted,
            right_sorted,
            left_on="__DT_KEY",
            right_on="__FEATURE_DT",
            direction="backward",
        ).sort_values("index").set_index("index")
        age = (merged["__DT_KEY"] - merged["__FEATURE_DT"]).dt.total_seconds() / 3600.0
        valid = age.notna() & age.le(float(max_carry_forward_hours))
        for col in feature_cols:
            out.loc[merged.index, col] = np.where(valid, pd.to_numeric(merged[col], errors="coerce"), 0.0)
        out["FiveMin_Data_Age_Hours"] = np.where(valid, age, float(max_carry_forward_hours) + 1.0)
        out["FiveMin_Load_Available"] = valid.astype(float).to_numpy()
    else:
        exact = left.merge(feats[["__DT_KEY"] + feature_cols], on="__DT_KEY", how="left", suffixes=("", "__five"))
        for col in feature_cols:
            src = f"{col}__five" if f"{col}__five" in exact.columns else col
            out[col] = pd.to_numeric(exact[src], errors="coerce").fillna(0.0).to_numpy()
        out["FiveMin_Load_Available"] = pd.to_numeric(out["FiveMin_Load_Available"], errors="coerce").fillna(0.0)
        out["FiveMin_Data_Age_Hours"] = pd.to_numeric(out["FiveMin_Data_Age_Hours"], errors="coerce").fillna(float(max_carry_forward_hours) + 1.0)

    for col in INTRADAY_LOAD_FEATURES:
        out[col] = pd.to_numeric(out[col], errors="coerce").replace([np.inf, -np.inf], np.nan).fillna(0.0)
    return out


def zero_intraday_load_features(frame: pd.DataFrame, *, unavailable_age_hours: float = 999.0) -> pd.DataFrame:
    """Return a copy with intraday feature columns set to operationally unavailable."""
    if frame is None or frame.empty:
        return frame
    out = frame.copy()
    for col in INTRADAY_LOAD_FEATURES:
        out[col] = 0.0
    if "FiveMin_Data_Age_Hours" in out.columns:
        out["FiveMin_Data_Age_Hours"] = float(unavailable_age_hours)
    if "FiveMin_Load_Available" in out.columns:
        out["FiveMin_Load_Available"] = 0.0
    return out
