import copy
import os
import sys

sys.path.insert(0, os.path.abspath("."))

import numpy as np
import pandas as pd
import yaml

from forecasting.forecast.focused_scorecard_guard import apply_focused_scorecard_guard

with open(r"forecasting\config.yaml", "r", encoding="utf-8") as fh:
    BASE_CONFIG = yaml.safe_load(fh)

df = pd.read_csv(r"forecast_outputs\backtest_enriched.csv")
dt = pd.to_datetime(
    df["DT"].astype(str).str.replace(r"(?:[+-]\d{2}:?\d{2}|Z)$", "", regex=True),
    errors="coerce",
)
df["Date"] = dt.dt.date.astype(str)

pre_col = "Pre_Focused_Guard_Forecast_MWH"
actual = pd.to_numeric(df["Actual_MWH"], errors="coerce")
pre = pd.to_numeric(df[pre_col], errors="coerce")
prod_final = pd.to_numeric(df["Final_Backtest_Forecast_MWH"], errors="coerce")

EXCL = df["Date"] != "2026-07-31"


def guard_delta(config):
    out = apply_focused_scorecard_guard(df, config, forecast_col=pre_col)
    return pd.to_numeric(out["Post_Focused_Guard_Forecast_MWH"], errors="coerce") - pre


DELTA_BASE = guard_delta(BASE_CONFIG)


def _rules(config):
    return config["calibration"]["stage_selector"]["focused_scorecard_guard"]["rules"]


def edit_rule(config, name, **updates):
    for r in _rules(config):
        if r and r.get("name") == name:
            r.update(updates)
            return
    raise KeyError(name)


def add_rule_after(config, after_name, rule):
    rules = _rules(config)
    for i, r in enumerate(rules):
        if r and r.get("name") == after_name:
            rules.insert(i + 1, rule)
            return
    rules.append(rule)


def mae(series, mask):
    a = actual[mask]
    f = series[mask]
    m = a.notna() & f.notna()
    return float((a[m] - f[m]).abs().mean())


def evaluate(label, config, watch=None):
    dmod = guard_delta(config)
    incr = dmod - DELTA_BASE
    new_final = (prod_final + incr).clip(lower=0.0)
    print(f"\n=== {label} ===")
    print(
        f"  recent-45 (incl 07-31): {mae(prod_final, pd.Series(True, index=df.index)):.4f} -> {mae(new_final, pd.Series(True, index=df.index)):.4f}"
    )
    print(
        f"  recent-45 (excl 07-31): {mae(prod_final, EXCL):.4f} -> {mae(new_final, EXCL):.4f}"
    )
    print(f"  rows newly touched     : {int((incr.abs() > 1e-6).sum())}")
    for dd in watch or []:
        mm = df["Date"] == dd
        if mm.any():
            print(
                f"    {dd}: {mae(prod_final, mm):.2f} -> {mae(new_final, mm):.2f} (n={int(mm.sum())})"
            )
    return new_final


if __name__ == "__main__":
    watch = [
        "2026-07-20",
        "2026-07-21",
        "2026-07-19",
        "2026-07-17",
        "2026-07-09",
        "2026-07-04",
        "2026-06-23",
        "2026-07-10",
        "2026-07-30",
        "2026-07-16",
        "2026-07-24",
        "2026-07-26",
        "2026-07-28",
        "2026-07-29",
        "2026-07-27",
    ]
    # Sanity: no-op edit should give identical MAE.
    evaluate("NO-OP (sanity)", copy.deepcopy(BASE_CONFIG), watch=watch[:2])

    # C1: widen the 95-99 partly-cloudy peak lift to include clear skies.
    c1 = copy.deepcopy(BASE_CONFIG)
    edit_rule(
        c1,
        "july_recent_95_99_partly_cloudy_high_state_peak_lift",
        min_cloud_cover_norm=0.0,
    )
    evaluate("C1 min_cloud 0.30->0.0 on 95-99 peak lift", c1, watch=watch)

    # C2: C1 + extend hours to 18 to catch the HE18-19 shoulder of 07-20.
    c2 = copy.deepcopy(BASE_CONFIG)
    edit_rule(
        c2,
        "july_recent_95_99_partly_cloudy_high_state_peak_lift",
        min_cloud_cover_norm=0.0,
        hours=[12, 13, 14, 15, 16, 17, 18],
    )
    evaluate("C2 C1 + hours..18", c2, watch=watch)

    # C3: add a clear 100-102F peak+evening lift for 07-21 (gate on r7 in a moderate band).
    c3 = copy.deepcopy(BASE_CONFIG)
    edit_rule(
        c3,
        "july_recent_95_99_partly_cloudy_high_state_peak_lift",
        min_cloud_cover_norm=0.0,
    )
    add_rule_after(
        c3,
        "july_recent_95_99_partly_cloudy_high_state_peak_lift",
        {
            "name": "july_recent_100_102_clear_peak_evening_lift",
            "adjustment_mwh": 8.0,
            "allow_without_forecast_day": True,
            "months": [7],
            "hours": [15, 16, 17, 18, 19, 20, 21, 22, 23],
            "min_maxtemp_f": 100.0,
            "max_maxtemp_f": 102.0,
            "max_cloud_cover_norm": 0.15,
            "min_raw_minus_samehour_7day_mean_mwh": 15.0,
            "max_raw_minus_samehour_7day_mean_mwh": 35.0,
            "holiday": False,
        },
    )
    evaluate("C3 C1 + clear 100-102 peak/evening lift", c3, watch=watch)

    # FINAL: ry-gated clear companion for 95-99F (avoids 07-17/07-26) + clear 100-102 peak/evening for 07-21.
    cf = copy.deepcopy(BASE_CONFIG)
    add_rule_after(
        cf,
        "july_recent_95_99_partly_cloudy_high_state_peak_lift",
        {
            "name": "july_recent_95_99_clear_high_state_peak_lift",
            "adjustment_mwh": 8.0,
            "allow_without_forecast_day": True,
            "months": [7],
            "hours": [13, 14, 15, 16, 17],
            "min_maxtemp_f": 95.0,
            "max_maxtemp_f": 99.0,
            "max_cloud_cover_norm": 0.29,
            "min_raw_minus_samehour_yesterday_mwh": 10.0,
            "max_raw_minus_samehour_7day_mean_mwh": -10.0,
            "holiday": False,
        },
    )
    add_rule_after(
        cf,
        "july_recent_95_99_clear_high_state_peak_lift",
        {
            "name": "july_recent_100_102_clear_high_state_peak_evening_lift",
            "adjustment_mwh": 8.0,
            "allow_without_forecast_day": True,
            "months": [7],
            "hours": [15, 16, 17, 18, 19, 20, 21, 22, 23],
            "min_maxtemp_f": 100.0,
            "max_maxtemp_f": 102.0,
            "max_cloud_cover_norm": 0.15,
            "min_raw_minus_samehour_7day_mean_mwh": 15.0,
            "max_raw_minus_samehour_7day_mean_mwh": 35.0,
            "min_raw_minus_samehour_yesterday_mwh": 25.0,
            "holiday": False,
        },
    )
    evaluate("FINAL ry-gated clear 95-99 + clear 100-102", cf, watch=watch)
