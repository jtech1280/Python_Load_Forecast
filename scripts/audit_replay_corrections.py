from __future__ import annotations

import argparse
import json
import re
from pathlib import Path
from typing import Iterable

import numpy as np
import pandas as pd


FOCUSED_REQUIRED = {
    "Actual_MWH",
    "Pre_Focused_Guard_Forecast_MWH",
    "Post_Focused_Guard_Forecast_MWH",
    "Focused_Scorecard_Guard_MWH",
    "Focused_Scorecard_Guard_Source",
}


def _label_from_path(path: Path) -> str:
    match = re.search(r"_(\d{8}_\d{6})_", path.name)
    return match.group(1) if match else path.stem


def _latest_replay_path(output_dir: Path) -> Path:
    candidates = sorted(
        output_dir.glob("rolling_origin_replay_results_*.csv"),
        key=lambda p: p.stat().st_mtime,
        reverse=True,
    )
    if candidates:
        return candidates[0]
    current = output_dir / "rolling_origin_replay_results.csv"
    if current.exists():
        return current
    raise FileNotFoundError(f"No rolling-origin replay results CSV found in {output_dir}")


def _as_num(values) -> pd.Series:
    return pd.to_numeric(values, errors="coerce")


def _err(actual: pd.Series, pred: pd.Series) -> pd.Series:
    return _as_num(actual) - _as_num(pred)


def _metrics(actual: pd.Series, pred: pd.Series) -> dict[str, float]:
    err = _err(actual, pred)
    return {
        "N": int(err.notna().sum()),
        "MAE_MWH": float(err.abs().mean()),
        "RMSE_MWH": float(np.sqrt((err**2).mean())),
        "Bias_MWH": float(err.mean()),
        "P90_AbsError_MWH": float(err.abs().quantile(0.90)),
        "P99_AbsError_MWH": float(err.abs().quantile(0.99)),
    }


def _metric_delta_row(
    *,
    actual: pd.Series,
    before: pd.Series,
    after: pd.Series,
    base: dict,
) -> dict:
    before_m = _metrics(actual, before)
    after_m = _metrics(actual, after)
    row = dict(base)
    row.update(
        {
            "Before_MAE_MWH": before_m["MAE_MWH"],
            "After_MAE_MWH": after_m["MAE_MWH"],
            "Delta_MAE_MWH": after_m["MAE_MWH"] - before_m["MAE_MWH"],
            "Before_RMSE_MWH": before_m["RMSE_MWH"],
            "After_RMSE_MWH": after_m["RMSE_MWH"],
            "Delta_RMSE_MWH": after_m["RMSE_MWH"] - before_m["RMSE_MWH"],
            "Before_Bias_MWH": before_m["Bias_MWH"],
            "After_Bias_MWH": after_m["Bias_MWH"],
            "Delta_Bias_MWH": after_m["Bias_MWH"] - before_m["Bias_MWH"],
            "Before_P90_AbsError_MWH": before_m["P90_AbsError_MWH"],
            "After_P90_AbsError_MWH": after_m["P90_AbsError_MWH"],
            "Delta_P90_AbsError_MWH": after_m["P90_AbsError_MWH"] - before_m["P90_AbsError_MWH"],
            "Before_P99_AbsError_MWH": before_m["P99_AbsError_MWH"],
            "After_P99_AbsError_MWH": after_m["P99_AbsError_MWH"],
            "Delta_P99_AbsError_MWH": after_m["P99_AbsError_MWH"] - before_m["P99_AbsError_MWH"],
        }
    )
    return row


def _group_value(row: pd.Series, group_cols: list[str]) -> str:
    if not group_cols:
        return "all"
    return "|".join(str(row.get(col)) for col in group_cols)


def focused_guard_source_audit(df: pd.DataFrame) -> pd.DataFrame:
    if not FOCUSED_REQUIRED.issubset(df.columns):
        return pd.DataFrame()
    adjustment = _as_num(df["Focused_Scorecard_Guard_MWH"]).fillna(0.0)
    active = adjustment.abs().gt(1e-9)
    if not active.any():
        return pd.DataFrame()

    rows = []
    work = df.loc[active].copy()
    for source, group in work.groupby("Focused_Scorecard_Guard_Source", dropna=False):
        idx = group.index
        adj = adjustment.loc[idx]
        rows.append(
            _metric_delta_row(
                actual=df.loc[idx, "Actual_MWH"],
                before=df.loc[idx, "Pre_Focused_Guard_Forecast_MWH"],
                after=df.loc[idx, "Post_Focused_Guard_Forecast_MWH"],
                base={
                    "Audit": "focused_guard_source",
                    "Source": str(source),
                    "N": int(len(idx)),
                    "Mean_Adjustment_MWH": float(adj.mean()),
                    "MeanAbs_Adjustment_MWH": float(adj.abs().mean()),
                    "Min_Adjustment_MWH": float(adj.min()),
                    "Max_Adjustment_MWH": float(adj.max()),
                },
            )
        )
    return pd.DataFrame(rows).sort_values(["Delta_MAE_MWH", "N"], ascending=[False, False]).reset_index(drop=True)


def focused_guard_rule_audit(df: pd.DataFrame) -> pd.DataFrame:
    if not FOCUSED_REQUIRED.issubset(df.columns):
        return pd.DataFrame()
    adjustment = _as_num(df["Focused_Scorecard_Guard_MWH"]).fillna(0.0)
    active = adjustment.abs().gt(1e-9)
    if not active.any():
        return pd.DataFrame()

    sources = df.loc[active, "Focused_Scorecard_Guard_Source"].fillna("none").astype(str)
    rule_names = sorted({part for source in sources for part in source.split("+") if part and part != "none"})
    rows = []
    for rule_name in rule_names:
        mask = active & df["Focused_Scorecard_Guard_Source"].fillna("").astype(str).str.split("+").map(lambda parts: rule_name in parts)
        idx = df.index[mask]
        if len(idx) == 0:
            continue
        adj = adjustment.loc[idx]
        rows.append(
            _metric_delta_row(
                actual=df.loc[idx, "Actual_MWH"],
                before=df.loc[idx, "Pre_Focused_Guard_Forecast_MWH"],
                after=df.loc[idx, "Post_Focused_Guard_Forecast_MWH"],
                base={
                    "Audit": "focused_guard_rule",
                    "Rule": rule_name,
                    "N": int(len(idx)),
                    "Mean_Total_Adjustment_MWH": float(adj.mean()),
                    "MeanAbs_Total_Adjustment_MWH": float(adj.abs().mean()),
                },
            )
        )
    return pd.DataFrame(rows).sort_values(["Delta_MAE_MWH", "N"], ascending=[False, False]).reset_index(drop=True)


def _long_horizon_group_rows(
    df: pd.DataFrame,
    *,
    component: str,
    correction: pd.Series,
    group_cols: list[str],
) -> list[dict]:
    active = correction.abs().gt(1e-9)
    if not active.any():
        return []
    if "Pre_Focused_Guard_Forecast_MWH" in df.columns:
        with_correction = _as_num(df["Pre_Focused_Guard_Forecast_MWH"])
    else:
        with_correction = _as_num(df["Final_Backtest_Forecast_MWH"])
    without_correction = with_correction - correction
    rows = []
    grouped = df.loc[active].groupby(group_cols, dropna=False) if group_cols else [("all", df.loc[active])]
    for key, group in grouped:
        idx = group.index
        corr = correction.loc[idx]
        value = key if not isinstance(key, tuple) else "|".join(str(x) for x in key)
        rows.append(
            _metric_delta_row(
                actual=df.loc[idx, "Actual_MWH"],
                before=without_correction.loc[idx],
                after=with_correction.loc[idx],
                base={
                    "Audit": "long_horizon_correction",
                    "Component": component,
                    "Group": "+".join(group_cols) if group_cols else "all",
                    "Value": str(value),
                    "N": int(len(idx)),
                    "Mean_Adjustment_MWH": float(corr.mean()),
                    "MeanAbs_Adjustment_MWH": float(corr.abs().mean()),
                },
            )
        )
    return rows


def long_horizon_audit(df: pd.DataFrame) -> pd.DataFrame:
    if "Actual_MWH" not in df.columns:
        return pd.DataFrame()
    peak = _as_num(df.get("Long_Horizon_Peak_Month_Correction_MWH", pd.Series(0.0, index=df.index))).fillna(0.0)
    hot = _as_num(df.get("Long_Horizon_Hot_Month_Correction_MWH", pd.Series(0.0, index=df.index))).fillna(0.0)
    groups = [
        [],
        [col for col in ["Month"] if col in df.columns],
        [col for col in ["Month", "Hour"] if col in df.columns],
        [col for col in ["Month", "Forecast_Day"] if col in df.columns],
        [col for col in ["Season"] if col in df.columns],
    ]
    rows: list[dict] = []
    for component, correction in [
        ("peak_month", peak),
        ("hot_month", hot),
        ("combined_peak_hot", peak + hot),
    ]:
        for group_cols in groups:
            rows.extend(_long_horizon_group_rows(df, component=component, correction=correction, group_cols=group_cols))
    if not rows:
        return pd.DataFrame()
    return pd.DataFrame(rows).sort_values(["Delta_MAE_MWH", "N"], ascending=[False, False]).reset_index(drop=True)


def stage_reason_audit(df: pd.DataFrame) -> pd.DataFrame:
    required = {"Actual_MWH", "Final_Backtest_Forecast_MWH", "Stage_Selector_Reason"}
    if not required.issubset(df.columns):
        return pd.DataFrame()
    rows = []
    for reason, group in df.groupby("Stage_Selector_Reason", dropna=False):
        metrics = _metrics(group["Actual_MWH"], group["Final_Backtest_Forecast_MWH"])
        rows.append(
            {
                "Audit": "stage_selector_reason",
                "Stage_Selector_Reason": str(reason),
                **metrics,
            }
        )
    return pd.DataFrame(rows).sort_values(["MAE_MWH", "N"], ascending=[False, False]).reset_index(drop=True)


def correction_activity_audit(df: pd.DataFrame, correction_cols: Iterable[str]) -> pd.DataFrame:
    if not {"Actual_MWH", "Final_Backtest_Forecast_MWH"}.issubset(df.columns):
        return pd.DataFrame()
    rows = []
    for col in correction_cols:
        if col not in df.columns:
            continue
        values = _as_num(df[col]).fillna(0.0)
        active = values.abs().gt(1e-9)
        if not active.any():
            continue
        metrics = _metrics(df.loc[active, "Actual_MWH"], df.loc[active, "Final_Backtest_Forecast_MWH"])
        rows.append(
            {
                "Audit": "correction_activity",
                "CorrectionColumn": col,
                "ActiveRows": int(active.sum()),
                "ActivePct": float(active.mean() * 100.0),
                "Mean_Adjustment_MWH": float(values.loc[active].mean()),
                "MeanAbs_Adjustment_MWH": float(values.loc[active].abs().mean()),
                **metrics,
            }
        )
    return pd.DataFrame(rows).sort_values(["MAE_MWH", "ActiveRows"], ascending=[False, False]).reset_index(drop=True)


def _read_replay(path: Path) -> pd.DataFrame:
    df = pd.read_csv(path)
    if "DT" in df.columns:
        try:
            df["DT"] = pd.to_datetime(df["DT"], errors="coerce", utc=True)
        except Exception:
            pass
    if "Month" not in df.columns and "DT" in df.columns:
        df["Month"] = pd.to_datetime(df["DT"], errors="coerce", utc=True).dt.month
    return df


def _save(df: pd.DataFrame, path: Path) -> None:
    if df is None or df.empty:
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(path, index=False)


def main() -> int:
    parser = argparse.ArgumentParser(description="Audit replay correction and guard impacts from saved rolling-origin replay results.")
    parser.add_argument("--replay-path", default=None, help="Explicit rolling_origin_replay_results CSV path.")
    parser.add_argument("--output-dir", default="forecast_outputs/replay_runs", help="Replay output directory.")
    parser.add_argument("--report-dir", default=None, help="Directory for audit CSVs. Defaults to output-dir.")
    parser.add_argument("--top-n", type=int, default=15)
    parser.add_argument("--no-save", action="store_true")
    args = parser.parse_args()

    output_dir = Path(args.output_dir)
    replay_path = Path(args.replay_path) if args.replay_path else _latest_replay_path(output_dir)
    report_dir = Path(args.report_dir) if args.report_dir else output_dir
    label = _label_from_path(replay_path)
    replay = _read_replay(replay_path)

    correction_cols = [
        "Focused_Scorecard_Guard_MWH",
        "Long_Horizon_Peak_Month_Correction_MWH",
        "Long_Horizon_Hot_Month_Correction_MWH",
        "Recent_Level_Correction_MWH",
        "Peak_Risk_Cal_MWH",
        "OriginDay_State_Correction_MWH",
        "AR_Residual_Correction_MWH",
    ]
    reports = {
        "focused_sources": focused_guard_source_audit(replay),
        "focused_rules": focused_guard_rule_audit(replay),
        "long_horizon": long_horizon_audit(replay),
        "stage_reasons": stage_reason_audit(replay),
        "correction_activity": correction_activity_audit(replay, correction_cols),
    }

    summary = {
        "replay_path": str(replay_path),
        "label": label,
        "rows": int(len(replay)),
        "reports": {name: int(len(df)) for name, df in reports.items()},
    }
    print(json.dumps(summary, indent=2))

    for name, df in reports.items():
        if df.empty:
            print(f"\n{name}: no rows")
            continue
        print(f"\n{name}: top harmful by Delta_MAE_MWH or MAE_MWH")
        sort_col = "Delta_MAE_MWH" if "Delta_MAE_MWH" in df.columns else "MAE_MWH"
        print(df.sort_values(sort_col, ascending=False).head(args.top_n).to_string(index=False, float_format=lambda x: f"{x:.4f}"))

    if not args.no_save:
        for name, df in reports.items():
            _save(df, report_dir / f"correction_audit_{name}_{label}.csv")
        (report_dir / f"correction_audit_summary_{label}.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
        print(f"\nSaved audit reports to: {report_dir}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
