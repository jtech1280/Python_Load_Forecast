from __future__ import annotations

import argparse
import json
from pathlib import Path

import pandas as pd


KEY_TESTS = [
    "Last 45 days",
    "Seasonal rolling origins",
    "Day 1 only",
    "Days 2-3",
    "Days 4-7",
    "Hot peak days",
    "Cloud/solar midday",
    "Shoulder heat transition",
    "Peak window hours 14-18",
]


def _scorecard_path(output_dir: Path, label: str | None) -> Path:
    if label:
        labeled = output_dir / f"production_readiness_scorecard_{label}.csv"
        if labeled.exists():
            return labeled
    current = output_dir / "production_readiness_scorecard.csv"
    if current.exists():
        return current
    raise FileNotFoundError("No production_readiness_scorecard CSV found.")


def _num(row: pd.Series, col: str) -> float | None:
    value = pd.to_numeric(pd.Series([row.get(col)]), errors="coerce").iloc[0]
    return None if pd.isna(value) else float(value)


def main() -> int:
    parser = argparse.ArgumentParser(description="Print the key replay scorecard rows.")
    parser.add_argument("--output-dir", default="forecast_outputs")
    parser.add_argument("--label", default=None)
    parser.add_argument("--json-out", default=None)
    args = parser.parse_args()

    output_dir = Path(args.output_dir)
    path = _scorecard_path(output_dir, args.label)
    df = pd.read_csv(path)

    rows = []
    for test in KEY_TESTS:
        match = df[df["Test"].astype(str).eq(test)]
        if match.empty:
            continue
        row = match.iloc[0]
        rows.append(
            {
                "Test": test,
                "Pass": bool(row.get("Pass")),
                "N": int(_num(row, "N") or 0),
                "MAE_MWH": _num(row, "MAE_MWH"),
                "MAPE_PCT": _num(row, "MAPE_PCT"),
                "Bias_MWH": _num(row, "Bias_MWH"),
                "P90_AbsError_MWH": _num(row, "P90_AbsError_MWH"),
                "Max_Underforecast_MWH": _num(row, "Max_Underforecast_MWH"),
                "Underforecast_At_Actual_Peak_MWH": _num(row, "Underforecast_At_Actual_Peak_MWH"),
            }
        )

    summary = pd.DataFrame(rows)
    print(f"Scorecard: {path}")
    if summary.empty:
        print("No key scorecard rows found.")
    else:
        print(summary.to_string(index=False, float_format=lambda x: f"{x:.4f}"))

    backend_files = [
        "xgb_training_backend",
        "lgb_training_backend",
        "catboost_training_backend",
        "runtime_performance",
    ]
    print("\nBackends:")
    for stem in backend_files:
        candidates = []
        if args.label:
            candidates.append(output_dir / f"{stem}_{args.label}.json")
        candidates.append(output_dir / f"{stem}.json")
        found = next((p for p in candidates if p.exists()), None)
        if not found:
            continue
        try:
            data = json.loads(found.read_text(encoding="utf-8"))
        except Exception:
            continue
        selected = data.get("selected_backend")
        if selected is not None:
            print(f"{stem}: {selected}")
        elif stem == "runtime_performance":
            print(
                "runtime_performance: "
                f"threads={data.get('resolved_cpu_threads')}, "
                f"xgb_gpu={data.get('xgb_gpu_requested')}, "
                f"lgb_gpu={data.get('lgb_gpu_requested')}, "
                f"parallel={data.get('parallel_tree_training')}"
            )

    if args.json_out:
        out_path = Path(args.json_out)
        out_path.write_text(json.dumps(rows, indent=2), encoding="utf-8")
        print(f"\nSaved summary JSON: {out_path}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
