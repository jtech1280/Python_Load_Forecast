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
    """Convert column to numeric, return None if NaN (optimized)."""
    try:
        value = float(row.get(col))
        return value if pd.notna(value) else None
    except (TypeError, ValueError):
        # Fall back to pandas conversion if direct conversion fails
        value = pd.to_numeric(pd.Series([row.get(col)]), errors="coerce").iloc[0]
        return None if pd.isna(value) else float(value)


def _summarize_rows(df: pd.DataFrame) -> list[dict]:
    # Pre-convert Test column to string for faster comparison
    df = df.copy()
    df["Test"] = df["Test"].astype(str)

    rows = []
    for test in KEY_TESTS:
        match = df[df["Test"].eq(test)]
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
                "Underforecast_At_Actual_Peak_MWH": _num(
                    row, "Underforecast_At_Actual_Peak_MWH"
                ),
            }
        )
    return rows


def _print_comparison(current_rows: list[dict], compare_rows: list[dict], compare_path: Path) -> None:
    current_by_test = {r["Test"]: r for r in current_rows}
    compare_by_test = {r["Test"]: r for r in compare_rows}
    tests_in_order = [t for t in KEY_TESTS if t in current_by_test or t in compare_by_test]

    diff_rows = []
    for test in tests_in_order:
        cur = current_by_test.get(test)
        cmp = compare_by_test.get(test)
        cur_mae = cur["MAE_MWH"] if cur else None
        cmp_mae = cmp["MAE_MWH"] if cmp else None
        cur_bias = cur["Bias_MWH"] if cur else None
        cmp_bias = cmp["Bias_MWH"] if cmp else None
        diff_rows.append(
            {
                "Test": test,
                "Pass_current": cur["Pass"] if cur else None,
                "Pass_compare": cmp["Pass"] if cmp else None,
                "MAE_current": cur_mae,
                "MAE_compare": cmp_mae,
                "Delta_MAE": (
                    cur_mae - cmp_mae if cur_mae is not None and cmp_mae is not None else None
                ),
                "Bias_current": cur_bias,
                "Bias_compare": cmp_bias,
                "Delta_Bias": (
                    cur_bias - cmp_bias
                    if cur_bias is not None and cmp_bias is not None
                    else None
                ),
            }
        )
    print(f"\nComparison against: {compare_path}")
    print(
        "(Delta = current - compare; negative Delta_MAE means current is better, "
        "Delta_Bias moving toward 0 means current is less biased)"
    )
    print(
        pd.DataFrame(diff_rows).to_string(
            index=False, float_format=lambda x: f"{x:.4f}", na_rep="--"
        )
    )


def main() -> int:
    parser = argparse.ArgumentParser(description="Print the key replay scorecard rows.")
    parser.add_argument("--output-dir", default="forecast_outputs")
    parser.add_argument("--label", default=None)
    parser.add_argument("--json-out", default=None)
    parser.add_argument(
        "--compare-label",
        default=None,
        help=(
            "Label of a second scorecard to diff the current one against, e.g. "
            "'before_moderate_tier' for production_readiness_scorecard_before_moderate_tier.csv "
            "in --output-dir. Ignored if --compare-path is also given."
        ),
    )
    parser.add_argument(
        "--compare-path",
        default=None,
        help="Explicit path to a second scorecard CSV to diff the current one against.",
    )
    args = parser.parse_args()

    output_dir = Path(args.output_dir)
    path = _scorecard_path(output_dir, args.label)
    df = pd.read_csv(path)
    rows = _summarize_rows(df)

    summary = pd.DataFrame(rows)
    print(f"Scorecard: {path}")
    if summary.empty:
        print("No key scorecard rows found.")
    else:
        print(summary.to_string(index=False, float_format=lambda x: f"{x:.4f}"))

    compare_path_arg = args.compare_path
    if compare_path_arg is None and args.compare_label:
        compare_path_arg = str(_scorecard_path(output_dir, args.compare_label))
    if compare_path_arg:
        compare_path = Path(compare_path_arg)
        if not compare_path.exists():
            raise FileNotFoundError(f"--compare scorecard not found: {compare_path}")
        compare_rows = _summarize_rows(pd.read_csv(compare_path))
        _print_comparison(rows, compare_rows, compare_path)

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
