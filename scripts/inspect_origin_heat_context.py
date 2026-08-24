from __future__ import annotations

"""Prints the weather context (daily max temp trend into the origin day, consecutive
100F+ day count, and -- when present in the cache's raw_origin frame -- the climatology
reference/excess columns) for specific origins in a raw-forecast cache built by
ablate_correction_stages.py --build-cache.

Note on the climatology columns specifically: raw_origin/raw_calibration are curated
forecast/actuals/context frames (see rolling_origin_replay.py's _origin_raw_forecasts and
run_rolling_backtest), not a dump of every training input feature. Temp_Excess_Over_
Climatology_F can be genuinely absent from them even when record_breaking_heat was
enabled and the feature was used to train the underlying model -- so their absence here
is NOT proof the feature was off for this cache. If you need to know whether it was
actually enabled, check features.record_breaking_heat.enabled in the config that was
active when --build-cache ran for this cache dir.

Motivation: comparing per-origin bias between two caches (see
compare_origin_bias_across_caches.py) can surface a handful of origins where a feature
change moved bias the "wrong" way. Before reading anything into that, it's worth knowing
whether those origins are actually distinct events or the same multi-day heat wave
counted three times, and whether they look meaningfully more extreme (longer ramp,
more consecutive 100F+ days, a thinner climatology reference) than the origins where the
feature helped -- this prints exactly that context, origin by origin.

Usage:
    python scripts/inspect_origin_heat_context.py \\
        --cache-dir forecast_outputs/record_breaking_cache \\
        --origins 21 22 23 24 25 26 27 \\
        --lookback-days 5
"""

import argparse
import sys
from pathlib import Path

import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from forecasting.tuning.calibration_search import load_raw_origin_bundles

CLIMATOLOGY_COLS = [
    "Climatology_Temp_PXX_F",
    "Temp_Climatology_Reference_Years",
    "Temp_Excess_Over_Climatology_F",
]


def _daily_max(df: pd.DataFrame) -> pd.Series:
    """Date -> Temperature_DailyMax, one row per calendar date (a raw bundle frame is
    hourly, but Temperature_DailyMax is already constant within a date)."""
    if df is None or df.empty or "DT" not in df.columns or "Temperature_DailyMax" not in df.columns:
        return pd.Series(dtype=float)
    dt = pd.to_datetime(df["DT"], errors="coerce")
    temp = pd.to_numeric(df["Temperature_DailyMax"], errors="coerce")
    return pd.Series(temp.to_numpy(), index=dt.dt.date).groupby(level=0).max().sort_index()


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--cache-dir", required=True)
    parser.add_argument("--origins", nargs="+", type=int, required=True, help="Origin numbers, e.g. 21 22 25 26 27")
    parser.add_argument("--lookback-days", type=int, default=5, help="Trailing days of daily-max temp to print from raw_calibration")
    args = parser.parse_args()

    bundles = {b.origin_number: b for b in load_raw_origin_bundles(args.cache_dir)}
    if not bundles:
        raise SystemExit(f"No cached bundles found in {args.cache_dir}")

    missing = [n for n in args.origins if n not in bundles]
    if missing:
        print(f"WARNING: origin(s) not found in this cache: {missing}")

    for n in args.origins:
        bundle = bundles.get(n)
        if bundle is None:
            continue
        print(f"\n=== origin_{n:02d} ({bundle.origin_dt}) ===")

        origin_daily = _daily_max(bundle.raw_origin)
        origin_date = pd.Timestamp(bundle.origin_dt).date()
        origin_temp = origin_daily.get(origin_date)
        print(f"Origin-day Temperature_DailyMax: {origin_temp}")

        cal_daily = _daily_max(bundle.raw_calibration)
        trailing = cal_daily[cal_daily.index < origin_date].tail(args.lookback_days)
        if not trailing.empty:
            print(f"Trailing {len(trailing)} day(s) of daily-max temp leading into the origin:")
            for date, temp in trailing.items():
                print(f"  {date}: {temp:.1f} F")
        else:
            print("No trailing calibration-window daily-max temps available.")

        consec_col = "ConsecutiveExtremeHotDays100"
        if consec_col in bundle.raw_origin.columns:
            origin_rows = bundle.raw_origin[
                pd.to_datetime(bundle.raw_origin["DT"], errors="coerce").dt.date.eq(origin_date)
            ]
            if not origin_rows.empty:
                print(f"{consec_col} on origin day: {origin_rows[consec_col].iloc[0]}")

        available_climatology = [c for c in CLIMATOLOGY_COLS if c in bundle.raw_origin.columns]
        if available_climatology:
            origin_rows = bundle.raw_origin[
                pd.to_datetime(bundle.raw_origin["DT"], errors="coerce").dt.date.eq(origin_date)
            ]
            if not origin_rows.empty:
                for col in available_climatology:
                    print(f"{col} on origin day: {origin_rows[col].iloc[0]}")
        else:
            print(
                "record_breaking_heat climatology columns not present in raw_origin -- this is "
                "NOT evidence the feature was off. raw_origin/raw_calibration are curated "
                "forecast/actuals/context frames (see rolling_origin_replay.py's "
                "_origin_raw_forecasts and run_rolling_backtest), not a dump of every training "
                "input feature, so a model-input column like Temp_Excess_Over_Climatology_F can "
                "be absent here even when it was used to train the underlying model. To check "
                "whether the feature was actually enabled for this cache, look at "
                "features.record_breaking_heat.enabled in whatever config was active when "
                "--build-cache ran for this cache dir -- that's the source of truth, not this "
                "column check."
            )

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
