from __future__ import annotations

import pandas as pd

from forecasting.model.xgb_model import build_sample_weights

TEMP_BUCKET_EDGES: list[tuple[float, float, str]] = [
    (float("-inf"), 75.0, "<75"),
    (75.0, 85.0, "75-85"),
    (85.0, 90.0, "85-90"),
    (90.0, 92.5, "90-92.5"),
    (92.5, 95.0, "92.5-95"),
    (95.0, 98.0, "95-98"),
    (98.0, 100.0, "98-100"),
    (100.0, 105.0, "100-105"),
    (105.0, float("inf"), "105+"),
]


def _temp_bucket_labels(daily_max: pd.Series) -> pd.Series:
    labels = pd.Series("unknown", index=daily_max.index, dtype="object")
    for lo, hi, label in TEMP_BUCKET_EDGES:
        mask = daily_max.ge(lo) & daily_max.lt(hi)
        labels.loc[mask] = label
    labels.loc[daily_max.isna()] = "unknown"
    return labels


def build_training_weight_diagnostic(
    train_df: pd.DataFrame, config: dict | None
) -> pd.DataFrame:
    """
    Report, per Temperature_DailyMax bucket, how many training rows exist and how
    much total sample weight (from build_sample_weights) they carry -- both for all
    rows and restricted to the hot-peak-hour scope those weights are tuned against.

    Answers: is the raw model's hot-day under-forecast bias a training-data sparsity
    problem (few rows even after upweighting), or something else?
    """
    df = train_df.reset_index(drop=True)
    weights = build_sample_weights(df, config)

    sw_cfg = ((config or {}).get("model", {}) or {}).get("sample_weight", {}) or {}
    hot_min = float(sw_cfg.get("hot_day_min_f", 90.0))
    hot_hours = {int(h) for h in sw_cfg.get("hot_peak_hours", [16, 17, 18, 19, 20])}

    daily_max = pd.to_numeric(df.get("Temperature_DailyMax"), errors="coerce")
    hour = pd.to_numeric(df.get("Hour"), errors="coerce")
    hot_peak_scope = daily_max.ge(hot_min) & hour.astype("Int64").isin(hot_hours)

    bucket = _temp_bucket_labels(daily_max)

    out = pd.DataFrame(
        {
            "Temperature_DailyMax_Bucket": bucket,
            "SampleWeight": weights,
            "IsHotPeakScope": hot_peak_scope.fillna(False),
        }
    )

    total_rows = len(out)
    total_weight = float(out["SampleWeight"].sum())

    rows = []
    for _, _, label in TEMP_BUCKET_EDGES:
        for scope_only in (False, True):
            slice_df = out[out["Temperature_DailyMax_Bucket"] == label]
            if scope_only:
                slice_df = slice_df[slice_df["IsHotPeakScope"]]
            n = len(slice_df)
            weight_sum = float(slice_df["SampleWeight"].sum())
            rows.append(
                {
                    "Temperature_DailyMax_Bucket": label,
                    "Scope": "HotPeakHours16to20" if scope_only else "AllHours",
                    "N_Rows": n,
                    "Pct_Of_Total_Rows": (100.0 * n / total_rows) if total_rows else 0.0,
                    "Mean_Sample_Weight": (weight_sum / n) if n else 0.0,
                    "Total_Sample_Weight": weight_sum,
                    "Pct_Of_Total_Weight": (
                        100.0 * weight_sum / total_weight if total_weight else 0.0
                    ),
                }
            )

    report = pd.DataFrame(rows)
    report["Weighted_Share_vs_RawShare_Ratio"] = report["Pct_Of_Total_Weight"] / report[
        "Pct_Of_Total_Rows"
    ].replace(0.0, pd.NA)
    return report
