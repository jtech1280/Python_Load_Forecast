from __future__ import annotations

"""Differential test for the simulate_recent_residual_correction_backtest vectorization:
proves the new prefix-sum/range-query implementation produces the same output as the
original per-row `for i, row in out.iterrows(): hist = out.iloc[:i]` loop, across many
randomized synthetic datasets covering edge cases (missing hours, NaN residuals, missing
weather buckets, duplicate timestamps, zero weights, AR-residual/origin-day-state enabled).

The reference implementation below is copied verbatim from the pre-vectorization version of
forecasting/forecast/recent_residual_correction.py (git blob at HEAD before the 2026-08-13
vectorization commit) and must NEVER be "fixed" to match the new code -- its entire purpose
is to stay frozen as the ground truth being diffed against. All helper functions it calls
(_cfg, _add_weather_residual_buckets, _build_ar_residual_state, _combine_recent_and_ar_
corrections, etc.) are unchanged by the vectorization, so they're imported live from the
real module rather than duplicated here.
"""

import unittest

import numpy as np
import pandas as pd

from forecasting.forecast.recent_residual_correction import (
    _add_weather_residual_buckets,
    _as_num,
    _build_ar_residual_state,
    _build_origin_day_state,
    _cfg,
    _combine_recent_and_ar_corrections,
    _hour_group,
    _recent_hot_peak_scale,
    _recent_horizon_regime_scale,
    simulate_recent_residual_correction_backtest,
)


def _reference_simulate_recent_residual_correction_backtest(
    backtest_df: pd.DataFrame,
    config: dict | None = None,
    base_col: str = "Raw_Forecast_MWH",
) -> pd.DataFrame:
    c = _cfg(config)
    out = backtest_df.copy().sort_values("DT").reset_index(drop=True)
    if (
        out.empty
        or not bool(c.get("enabled", True))
        or not {"Actual_MWH", base_col}.issubset(out.columns)
    ):
        out["Recent_Level_Correction_MWH"] = 0.0
        out["Recent_Correction_Source"] = "disabled_or_empty"
        out["AR_Residual_Correction_MWH"] = 0.0
        out["AR_Residual_Phi"] = np.nan
        out["AR_Residual_Latest_MWH"] = np.nan
        out["AR_Residual_Source"] = "ar_disabled_or_empty"
        out["OriginDay_State_Correction_MWH"] = 0.0
        out["OriginDay_State_MWH"] = np.nan
        out["OriginDay_Latest_Day_MWH"] = np.nan
        out["OriginDay_State_Source"] = "origin_day_disabled_or_empty"
        out["Recent_Corrected_Forecast_MWH"] = pd.to_numeric(
            out.get(base_col, out.get("Raw_Forecast_MWH", 0.0)), errors="coerce"
        )
        out["Final_Backtest_Forecast_MWH"] = out["Recent_Corrected_Forecast_MWH"]
        return out

    out["DT"] = pd.to_datetime(out["DT"], errors="coerce")
    out["Hour"] = (
        _as_num(out.get("Hour", out["DT"].dt.hour))
        .fillna(out["DT"].dt.hour)
        .astype(int)
    )
    out["HourGroup"] = out.get("HourGroup", out["Hour"].map(_hour_group))
    out = _add_weather_residual_buckets(out)
    out["_RecentBasisResidual"] = _as_num(out["Actual_MWH"]) - _as_num(out[base_col])

    weights = c.get("weights", {}) or {}
    w_recent = float(weights.get("recent_mean", 0.35))
    w_last24 = float(weights.get("last24_mean", 0.20))
    w_same = float(weights.get("same_hour", 0.16))
    w_hourgroup = float(weights.get("hourgroup", 0.06))
    w_global = float(weights.get("global", 0.03))
    w_temp_hg = float(weights.get("temp_hourgroup", 0.08))
    w_cloud_hg = float(weights.get("cloud_hourgroup", 0.05))
    w_solar_hg = float(weights.get("solar_hourgroup", 0.03))
    w_loss_hg = float(weights.get("solar_loss_hourgroup", 0.03))
    w_temp_cloud_hg = float(weights.get("temp_cloud_hourgroup", 0.01))

    recent_hours = int(c.get("recent_hours", 48))
    same_hour_days = int(c.get("same_hour_days", 7))
    cap = float(c.get("cap_mwh", 10.0))
    blend = float(c.get("blend", 0.85))

    corrections: list[float] = []
    sources: list[str] = []
    ar_corrections: list[float] = []
    ar_phis: list[float] = []
    ar_latest_residuals: list[float] = []
    ar_sources: list[str] = []
    origin_day_corrections: list[float] = []
    origin_day_states: list[float] = []
    origin_day_latest_days: list[float] = []
    origin_day_sources: list[str] = []

    def add(
        vals: list[tuple[str, float, float]],
        name: str,
        series: pd.Series,
        weight: float,
    ):
        if weight <= 0 or series.empty:
            return
        v = pd.to_numeric(series, errors="coerce").mean()
        if np.isfinite(v):
            vals.append((name, float(np.clip(v, -cap, cap)), float(weight)))

    for i, row in out.iterrows():
        hist = out.iloc[:i]
        hist = hist[
            pd.to_numeric(hist["_RecentBasisResidual"], errors="coerce").notna()
        ]
        if hist.empty:
            corrections.append(0.0)
            sources.append("insufficient_prior_residuals")
            ar_corrections.append(0.0)
            ar_phis.append(np.nan)
            ar_latest_residuals.append(np.nan)
            ar_sources.append("ar_disabled_or_empty")
            origin_day_corrections.append(0.0)
            origin_day_states.append(np.nan)
            origin_day_latest_days.append(np.nan)
            origin_day_sources.append("origin_day_disabled_or_empty")
            continue

        vals: list[tuple[str, float, float]] = []
        same_window_start = row["DT"] - pd.Timedelta(days=max(1, same_hour_days))
        same_window = hist[hist["DT"] >= same_window_start]
        if same_window.empty:
            same_window = hist

        add(
            vals,
            "recent_mean",
            hist.tail(max(1, recent_hours))["_RecentBasisResidual"],
            w_recent,
        )
        add(
            vals,
            "last24_mean",
            hist.tail(min(24, len(hist)))["_RecentBasisResidual"],
            w_last24,
        )
        add(
            vals,
            "same_hour",
            same_window.loc[
                same_window["Hour"].eq(row["Hour"]), "_RecentBasisResidual"
            ],
            w_same,
        )
        add(
            vals,
            "hourgroup",
            same_window.loc[
                same_window["HourGroup"].eq(row["HourGroup"]), "_RecentBasisResidual"
            ],
            w_hourgroup,
        )
        add(vals, "global", hist["_RecentBasisResidual"], w_global)
        if pd.notna(row.get("DailyMaxTempBucket")):
            add(
                vals,
                "temp_hourgroup",
                same_window.loc[
                    same_window["DailyMaxTempBucket"].eq(row.get("DailyMaxTempBucket"))
                    & same_window["HourGroup"].eq(row["HourGroup"]),
                    "_RecentBasisResidual",
                ],
                w_temp_hg,
            )
        if pd.notna(row.get("CloudCoverBucket")):
            add(
                vals,
                "cloud_hourgroup",
                same_window.loc[
                    same_window["CloudCoverBucket"].eq(row.get("CloudCoverBucket"))
                    & same_window["HourGroup"].eq(row["HourGroup"]),
                    "_RecentBasisResidual",
                ],
                w_cloud_hg,
            )
        if pd.notna(row.get("BTMSolarBucket")):
            add(
                vals,
                "solar_hourgroup",
                same_window.loc[
                    same_window["BTMSolarBucket"].eq(row.get("BTMSolarBucket"))
                    & same_window["HourGroup"].eq(row["HourGroup"]),
                    "_RecentBasisResidual",
                ],
                w_solar_hg,
            )
        if pd.notna(row.get("SolarLossBucket")):
            add(
                vals,
                "solar_loss_hourgroup",
                same_window.loc[
                    same_window["SolarLossBucket"].eq(row.get("SolarLossBucket"))
                    & same_window["HourGroup"].eq(row["HourGroup"]),
                    "_RecentBasisResidual",
                ],
                w_loss_hg,
            )
        if pd.notna(row.get("DailyMaxTempBucket")) and pd.notna(
            row.get("CloudCoverBucket")
        ):
            add(
                vals,
                "temp_cloud_hourgroup",
                same_window.loc[
                    same_window["DailyMaxTempBucket"].eq(row.get("DailyMaxTempBucket"))
                    & same_window["CloudCoverBucket"].eq(row.get("CloudCoverBucket"))
                    & same_window["HourGroup"].eq(row["HourGroup"]),
                    "_RecentBasisResidual",
                ],
                w_temp_cloud_hg,
            )

        if vals:
            raw = sum(v * w for _, v, w in vals) / sum(w for _, _, w in vals)
            correction_scale = _recent_hot_peak_scale(
                row, c
            ) * _recent_horizon_regime_scale(row, c)
            base_correction = float(np.clip(raw * blend * correction_scale, -cap, cap))
            base_source = "+".join(name for name, _, _ in vals)
        else:
            base_correction = 0.0
            base_source = "no_match"

        residual_state_profile = {
            "enabled": True,
            "ar_residual": _build_ar_residual_state(
                hist, c, residual_col="_RecentBasisResidual"
            ),
            "origin_day_state": _build_origin_day_state(
                hist, c, residual_col="_RecentBasisResidual"
            ),
        }
        (
            correction,
            source,
            ar_corr,
            ar_phi,
            ar_latest,
            ar_source,
            origin_day_corr,
            origin_day_state,
            origin_day_latest,
            origin_day_source,
        ) = _combine_recent_and_ar_corrections(
            base_correction,
            base_source,
            row,
            residual_state_profile,
            c,
            horizon_index=1,
        )
        corrections.append(correction)
        sources.append(source)
        ar_corrections.append(ar_corr)
        ar_phis.append(ar_phi)
        ar_latest_residuals.append(ar_latest)
        ar_sources.append(ar_source)
        origin_day_corrections.append(origin_day_corr)
        origin_day_states.append(origin_day_state)
        origin_day_latest_days.append(origin_day_latest)
        origin_day_sources.append(origin_day_source)

    out["Recent_Level_Correction_MWH"] = corrections
    out["Recent_Correction_Source"] = sources
    out["AR_Residual_Correction_MWH"] = ar_corrections
    out["AR_Residual_Phi"] = ar_phis
    out["AR_Residual_Latest_MWH"] = ar_latest_residuals
    out["AR_Residual_Source"] = ar_sources
    out["OriginDay_State_Correction_MWH"] = origin_day_corrections
    out["OriginDay_State_MWH"] = origin_day_states
    out["OriginDay_Latest_Day_MWH"] = origin_day_latest_days
    out["OriginDay_State_Source"] = origin_day_sources
    out["Pre_Recent_Forecast_MWH"] = _as_num(out[base_col])
    out["Recent_Corrected_Forecast_MWH"] = (
        out["Pre_Recent_Forecast_MWH"] + out["Recent_Level_Correction_MWH"]
    ).clip(lower=0.0)
    out["Final_Backtest_Forecast_MWH"] = out["Recent_Corrected_Forecast_MWH"]
    out["Final_Forecast_MWH"] = out["Recent_Corrected_Forecast_MWH"]
    out["Recent_Corrected_Residual_MWH"] = _as_num(out["Actual_MWH"]) - _as_num(
        out["Recent_Corrected_Forecast_MWH"]
    )
    out["Recent_Corrected_AbsError_MWH"] = out["Recent_Corrected_Residual_MWH"].abs()
    out["Recent_Corrected_APE"] = np.where(
        _as_num(out["Actual_MWH"]).abs() > 1e-9,
        out["Recent_Corrected_AbsError_MWH"] / _as_num(out["Actual_MWH"]).abs() * 100.0,
        np.nan,
    )
    return out.drop(columns=["_RecentBasisResidual"], errors="ignore")


def _random_frame(rng: np.random.Generator, n_hours: int, start: str = "2024-01-01") -> pd.DataFrame:
    dt_full = pd.date_range(start, periods=n_hours, freq="h")
    keep_mask = rng.random(n_hours) > 0.03
    keep_mask[0] = True
    dt = dt_full[keep_mask]
    n = len(dt)
    actual = 500 + rng.normal(0, 25, n)
    base = 500 + rng.normal(0, 25, n)
    nan_mask = rng.random(n) < 0.05
    actual = np.where(nan_mask, np.nan, actual)
    df = pd.DataFrame(
        {
            "DT": dt,
            "Hour": dt.hour,
            "Actual_MWH": actual,
            "Raw_Forecast_MWH": base,
            "Temperature_DailyMax": rng.uniform(40, 108, n),
            "CloudCover_Norm": rng.uniform(0, 1, n),
            "Forecast_Day": rng.integers(1, 16, n).astype(float),
        }
    )
    if rng.random() < 0.3:
        drop_col = rng.choice(["Temperature_DailyMax", "CloudCover_Norm"])
        mask = rng.random(n) < 0.4
        df.loc[mask, drop_col] = np.nan
    if rng.random() < 0.3:
        df["Season"] = rng.choice(["Winter", "Spring", "Summer", "Fall"], n)
    return df


def _random_cfg(rng: np.random.Generator, ar_enabled: bool, origin_enabled: bool) -> dict:
    weights = {
        "recent_mean": round(float(rng.uniform(0, 0.5)), 3),
        "last24_mean": round(float(rng.uniform(0, 0.4)), 3),
        "same_hour": round(float(rng.uniform(0, 0.3)), 3),
        "hourgroup": round(float(rng.uniform(0, 0.2)), 3),
        "global": round(float(rng.uniform(0, 0.1)), 3),
        "temp_hourgroup": round(float(rng.uniform(0, 0.2)), 3),
        "cloud_hourgroup": round(float(rng.uniform(0, 0.2)), 3),
        "solar_hourgroup": round(float(rng.uniform(0, 0.1)), 3),
        "solar_loss_hourgroup": round(float(rng.uniform(0, 0.1)), 3),
        "temp_cloud_hourgroup": round(float(rng.uniform(0, 0.05)), 3),
    }
    if rng.random() < 0.3:
        k = rng.choice(list(weights))
        weights[k] = 0.0
    return {
        "calibration": {
            "recent_residual": {
                "enabled": True,
                "recent_hours": int(rng.integers(4, 72)),
                "same_hour_days": int(rng.integers(1, 14)),
                "cap_mwh": float(rng.uniform(3, 15)),
                "blend": float(rng.uniform(0.3, 1.0)),
                "weights": weights,
                "hot_peak": {
                    "hours": [16, 17, 18, 19, 20],
                    "min_maxtemp_f": 90.0,
                    "season_scales": {"Summer": 0.0},
                    "default_scale": float(rng.uniform(0, 1)),
                },
                "horizon_regime_scales": {
                    "forecast_day_scales": {
                        "days2to3": float(rng.uniform(0, 1)),
                        "days4to7": 1.0,
                        "days8plus": 1.0,
                    },
                    "summer_clear_hot_days2to7": {
                        "enabled": bool(rng.random() < 0.5),
                        "min_maxtemp_f": 90.0,
                        "max_cloud_cover_norm": 0.2,
                        "scale": 0.0,
                    },
                },
                "ar_residual": {
                    "enabled": ar_enabled,
                    "lookback_hours": int(rng.integers(24, 200)),
                    "min_pairs": int(rng.integers(4, 30)),
                    "same_hour_blend": float(rng.uniform(0, 0.5)),
                    "blend": float(rng.uniform(0.1, 0.6)),
                },
                "origin_day_state": {
                    "enabled": origin_enabled,
                    "lookback_days": int(rng.integers(2, 10)),
                    "min_days": int(rng.integers(1, 4)),
                    "min_total_hours": int(rng.integers(10, 50)),
                },
            }
        }
    }


def _assert_frames_match(a: pd.DataFrame, b: pd.DataFrame, msg: str) -> None:
    assert set(a.columns) == set(b.columns), f"{msg}: column mismatch {set(a.columns) ^ set(b.columns)}"
    for col in a.columns:
        if col == "_RecentBasisResidual":
            continue
        av = a[col].reset_index(drop=True)
        bv = b[col].reset_index(drop=True)
        if av.dtype == object or bv.dtype == object:
            both_na = av.isna().to_numpy() & bv.isna().to_numpy()
            av_arr = av.astype(str).to_numpy()
            bv_arr = bv.astype(str).to_numpy()
            mism = (av_arr != bv_arr) & ~both_na
            assert not mism.any(), (
                f"{msg}: col {col} mismatch at rows {np.nonzero(mism)[0][:5].tolist()}\n"
                f"ref={av_arr[mism][:5].tolist()}\nnew={bv_arr[mism][:5].tolist()}"
            )
        else:
            pd.testing.assert_series_equal(
                av, bv, check_dtype=False, check_names=False, check_exact=False,
                rtol=1e-7, atol=1e-9,
            )


class RecentResidualBacktestVectorizationTests(unittest.TestCase):
    def test_matches_reference_across_random_trials(self):
        n_trials = 30
        for trial in range(n_trials):
            rng = np.random.default_rng(3000 + trial)
            n_hours = int(rng.integers(20, 400))
            ar_enabled = bool(rng.random() < 0.25)
            origin_enabled = bool(rng.random() < 0.25)
            df = _random_frame(rng, n_hours)
            cfg = _random_cfg(rng, ar_enabled, origin_enabled)
            label = f"trial{trial} n_hours={n_hours} ar={ar_enabled} origin={origin_enabled}"
            with self.subTest(label):
                ref_out = _reference_simulate_recent_residual_correction_backtest(
                    df.copy(), config=cfg
                )
                new_out = simulate_recent_residual_correction_backtest(df.copy(), config=cfg)
                _assert_frames_match(ref_out, new_out, label)

    def test_empty_dataframe_with_columns(self):
        df = pd.DataFrame(columns=["DT", "Actual_MWH", "Raw_Forecast_MWH"])
        ref_out = _reference_simulate_recent_residual_correction_backtest(df.copy(), config={})
        new_out = simulate_recent_residual_correction_backtest(df.copy(), config={})
        _assert_frames_match(ref_out, new_out, "empty_df")

    def test_disabled_config(self):
        df = _random_frame(np.random.default_rng(5), 30)
        cfg = {"calibration": {"recent_residual": {"enabled": False}}}
        ref_out = _reference_simulate_recent_residual_correction_backtest(df.copy(), config=cfg)
        new_out = simulate_recent_residual_correction_backtest(df.copy(), config=cfg)
        _assert_frames_match(ref_out, new_out, "disabled")

    def test_single_row(self):
        df = _random_frame(np.random.default_rng(7), 1)
        cfg = _random_cfg(np.random.default_rng(8), False, False)
        ref_out = _reference_simulate_recent_residual_correction_backtest(df.copy(), config=cfg)
        new_out = simulate_recent_residual_correction_backtest(df.copy(), config=cfg)
        _assert_frames_match(ref_out, new_out, "single_row")

    def test_all_nan_actual(self):
        df = _random_frame(np.random.default_rng(9), 60)
        df["Actual_MWH"] = np.nan
        cfg = _random_cfg(np.random.default_rng(10), True, True)
        ref_out = _reference_simulate_recent_residual_correction_backtest(df.copy(), config=cfg)
        new_out = simulate_recent_residual_correction_backtest(df.copy(), config=cfg)
        _assert_frames_match(ref_out, new_out, "all_nan_actual")

    def test_all_zero_weights(self):
        cfg = _random_cfg(np.random.default_rng(11), False, False)
        for k in cfg["calibration"]["recent_residual"]["weights"]:
            cfg["calibration"]["recent_residual"]["weights"][k] = 0.0
        df = _random_frame(np.random.default_rng(12), 80)
        ref_out = _reference_simulate_recent_residual_correction_backtest(df.copy(), config=cfg)
        new_out = simulate_recent_residual_correction_backtest(df.copy(), config=cfg)
        _assert_frames_match(ref_out, new_out, "all_zero_weights")

    def test_duplicate_timestamps(self):
        rng = np.random.default_rng(13)
        df = _random_frame(rng, 100)
        dup_positions = rng.choice(len(df) - 1, size=5, replace=False)
        df = pd.concat([df, df.iloc[dup_positions]], ignore_index=True)
        cfg = _random_cfg(np.random.default_rng(14), True, False)
        ref_out = _reference_simulate_recent_residual_correction_backtest(df.copy(), config=cfg)
        new_out = simulate_recent_residual_correction_backtest(df.copy(), config=cfg)
        _assert_frames_match(ref_out, new_out, "duplicate_dt")

    def test_tiny_and_huge_windows(self):
        tiny_cfg = _random_cfg(np.random.default_rng(15), False, True)
        tiny_cfg["calibration"]["recent_residual"]["recent_hours"] = 1
        tiny_cfg["calibration"]["recent_residual"]["same_hour_days"] = 1
        df1 = _random_frame(np.random.default_rng(16), 120)
        ref_out = _reference_simulate_recent_residual_correction_backtest(df1.copy(), config=tiny_cfg)
        new_out = simulate_recent_residual_correction_backtest(df1.copy(), config=tiny_cfg)
        _assert_frames_match(ref_out, new_out, "tiny_windows")

        huge_cfg = _random_cfg(np.random.default_rng(17), True, False)
        huge_cfg["calibration"]["recent_residual"]["recent_hours"] = 10000
        huge_cfg["calibration"]["recent_residual"]["same_hour_days"] = 365
        df2 = _random_frame(np.random.default_rng(18), 150)
        ref_out2 = _reference_simulate_recent_residual_correction_backtest(df2.copy(), config=huge_cfg)
        new_out2 = simulate_recent_residual_correction_backtest(df2.copy(), config=huge_cfg)
        _assert_frames_match(ref_out2, new_out2, "huge_windows")


if __name__ == "__main__":
    unittest.main()
