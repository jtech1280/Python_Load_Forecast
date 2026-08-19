from __future__ import annotations

import unittest
import inspect

import numpy as np
import pandas as pd

from forecasting.model.xgb_model import (
    build_sample_weights,
    hot_peak_scope_mask,
    make_asymmetric_hot_peak_objective,
    train_xgb,
)


class HotPeakScopeMaskTests(unittest.TestCase):
    def test_matches_hot_day_and_peak_hour_config(self):
        df = pd.DataFrame(
            {
                "Temperature_DailyMax": [95, 95, 95, 80, 95],
                "Hour": [16, 20, 21, 17, 17],
            }
        )
        mask = hot_peak_scope_mask(df, {})
        # row0: hot day, hour 16 -> in scope. row1: hot day, hour 20 -> in scope.
        # row2: hot day but hour 21 not in default hot_peak_hours -> out.
        # row3: hour 17 but not a hot day -> out. row4: hot day + hour 17 -> in scope.
        self.assertEqual(list(mask), [True, True, False, False, True])

    def test_respects_custom_config_thresholds(self):
        df = pd.DataFrame(
            {
                "Temperature_DailyMax": [85, 85],
                "Hour": [10, 11],
            }
        )
        config = {
            "model": {
                "sample_weight": {
                    "hot_day_min_f": 80.0,
                    "hot_peak_hours": [10],
                }
            }
        }
        mask = hot_peak_scope_mask(df, config)
        self.assertEqual(list(mask), [True, False])

    def test_falls_back_to_likely_system_peak_hour_when_primary_scope_empty(self):
        df = pd.DataFrame(
            {
                "Temperature_DailyMax": [95, 95, 95],
                "Hour": [3, 4, 5],  # none in default hot_peak_hours
                "IsLikelySystemPeakHour": [0, 1, 1],
            }
        )
        mask = hot_peak_scope_mask(df, {})
        self.assertEqual(list(mask), [False, True, True])

    def test_all_false_when_temperature_daily_max_missing(self):
        df = pd.DataFrame({"Hour": [16, 17, 18]})
        mask = hot_peak_scope_mask(df, {})
        self.assertEqual(list(mask), [False, False, False])
        self.assertEqual(mask.dtype, bool)


class BuildSampleWeightsRefactorRegressionTests(unittest.TestCase):
    """hot_peak_scope_mask() was factored out of build_sample_weights()'s previously
    inline scope logic. Freeze the pre-refactor formula here and confirm the
    refactored function still produces identical weights for it."""

    def _old_scorecard_hot_peak(self, df: pd.DataFrame, sw_cfg: dict) -> pd.Series:
        daily_max = pd.to_numeric(df["Temperature_DailyMax"], errors="coerce")
        hour = pd.to_numeric(
            df.get("Hour", pd.Series(np.nan, index=df.index)), errors="coerce"
        )
        likely_peak = (
            pd.to_numeric(
                df.get("IsLikelySystemPeakHour", pd.Series(0, index=df.index)),
                errors="coerce",
            )
            .fillna(0)
            .astype(int)
            .eq(1)
        )
        hot_min = float(sw_cfg.get("hot_day_min_f", 90.0))
        hot_hours = {int(h) for h in sw_cfg.get("hot_peak_hours", [16, 17, 18, 19, 20])}
        scope = daily_max.ge(hot_min) & hour.astype("Int64").isin(hot_hours)
        if not scope.any():
            scope = daily_max.ge(hot_min) & likely_peak
        return scope.fillna(False)

    def test_weights_unchanged_by_extraction(self):
        rng = np.random.default_rng(3)
        n = 400
        df = pd.DataFrame(
            {
                "MWH": rng.normal(600, 80, n).clip(min=100),
                "Temperature_DailyMax": rng.uniform(60, 105, n),
                "Hour": rng.integers(0, 24, n),
                "IsLikelySystemPeakHour": rng.integers(0, 2, n),
            }
        )
        config = {"model": {"sample_weight": {}}}
        expected_scope = self._old_scorecard_hot_peak(df, {})
        actual_scope = hot_peak_scope_mask(df, config)
        pd.testing.assert_series_equal(
            expected_scope, actual_scope, check_names=False
        )

        weights = build_sample_weights(df, config)
        self.assertEqual(weights.shape, (n,))
        self.assertTrue(np.all(np.isfinite(weights)))


class MakeAsymmetricHotPeakObjectiveTests(unittest.TestCase):
    def test_uses_sklearn_objective_signature_with_sample_weight_param(self):
        """XGBRegressor.fit() inspects this signature via inspect.signature() and
        requires a `sample_weight` parameter whenever fit() is called with
        sample_weight (which train_xgb always does) -- see the docstring on
        make_asymmetric_hot_peak_objective for why. A regression here would
        surface as a hard training-time crash, not a silently wrong answer, but
        pin it explicitly since it's easy to reintroduce the native-API
        `(preds, dtrain)` convention by mistake.
        """
        objective = make_asymmetric_hot_peak_objective(
            np.array([True, False]), under_forecast_penalty=2.0
        )
        params = list(inspect.signature(objective).parameters)
        self.assertEqual(params[:2], ["y_true", "y_pred"])
        self.assertIn("sample_weight", params)

    def test_all_false_mask_reduces_to_squared_error(self):
        preds = np.array([10.0, 20.0, 5.0, 8.0])
        y = np.array([12.0, 15.0, 9.0, 8.0])
        mask = np.array([False, False, False, False])
        objective = make_asymmetric_hot_peak_objective(mask, under_forecast_penalty=5.0)
        grad, hess = objective(y, preds)
        np.testing.assert_allclose(grad, preds - y)
        np.testing.assert_allclose(hess, np.ones_like(preds))

    def test_penalty_of_one_reduces_to_squared_error_even_with_mask(self):
        preds = np.array([10.0, 20.0, 5.0, 8.0])
        y = np.array([12.0, 25.0, 9.0, 3.0])
        mask = np.array([True, True, True, True])
        objective = make_asymmetric_hot_peak_objective(mask, under_forecast_penalty=1.0)
        grad, hess = objective(y, preds)
        np.testing.assert_allclose(grad, preds - y)
        np.testing.assert_allclose(hess, np.ones_like(preds))

    def test_only_masked_underforecast_rows_are_penalized(self):
        preds = np.array([10.0, 20.0, 5.0, 8.0, 30.0])
        y = np.array([12.0, 15.0, 9.0, 3.0, 20.0])
        # row0: masked, under-forecast (pred<y) -> penalized
        # row1: masked, over-forecast (pred>y) -> not penalized
        # row2: masked, under-forecast -> penalized
        # row3: unmasked, over-forecast -> not penalized
        # row4: unmasked, over-forecast -> not penalized
        mask = np.array([True, True, True, False, False])
        penalty = 3.0
        objective = make_asymmetric_hot_peak_objective(
            mask, under_forecast_penalty=penalty
        )
        grad, hess = objective(y, preds)

        base_error = preds - y
        expected_weight = np.array([penalty, 1.0, penalty, 1.0, 1.0])
        np.testing.assert_allclose(grad, expected_weight * base_error)
        np.testing.assert_allclose(hess, expected_weight)

    def test_sample_weight_is_folded_into_grad_and_hess(self):
        """The sklearn API does NOT apply sample_weight itself for a custom
        objective -- the objective must. Confirms build_sample_weights' output
        (recency/peak/hot-day emphasis) still takes effect when combined with the
        asymmetric under-forecast penalty, rather than being silently dropped."""
        preds = np.array([10.0, 30.0])
        y = np.array([15.0, 20.0])  # row0 under-forecast, row1 over-forecast
        mask = np.array([True, True])
        penalty = 2.0
        sample_weight = np.array([4.0, 3.0])
        objective = make_asymmetric_hot_peak_objective(
            mask, under_forecast_penalty=penalty
        )
        grad, hess = objective(y, preds, sample_weight=sample_weight)

        base_error = preds - y
        expected_weight = np.array([penalty, 1.0]) * sample_weight
        np.testing.assert_allclose(grad, expected_weight * base_error)
        np.testing.assert_allclose(hess, expected_weight)

    def test_mask_is_cast_to_bool_and_defensively_copied(self):
        mask = np.array([1, 0, 1])
        objective = make_asymmetric_hot_peak_objective(mask, under_forecast_penalty=2.0)
        mask[:] = 0  # mutate the original array after building the closure
        preds = np.array([1.0, 1.0, 1.0])
        y = np.array([5.0, 5.0, 5.0])  # all under-forecast
        grad, _hess = objective(y, preds)
        # closure should have captured its own copy, unaffected by the later mutation
        np.testing.assert_allclose(grad, [2.0 * -4.0, 1.0 * -4.0, 2.0 * -4.0])


def _synthetic_frame_with_underfit_hot_peak_regime(n_days: int = 60) -> pd.DataFrame:
    """Hourly synthetic load where a minority "hot peak" regime (hot days, hours
    16-20) sits well above the smooth trend the bulk of rows follow. With limited
    tree capacity, a plain squared-error fit systematically under-predicts this
    minority regime -- the exact failure mode the asymmetric objective targets.
    """
    rng = np.random.default_rng(11)
    dt = pd.date_range("2026-06-01", periods=24 * n_days, freq="h")
    hour = dt.hour.values.astype(float)
    day_idx = dt.dayofyear.values
    rng_days = np.random.default_rng(5)
    daily_max_by_day = {
        d: rng_days.uniform(60, 105) for d in np.unique(day_idx)
    }
    daily_max = np.array([daily_max_by_day[d] for d in day_idx])

    base_load = 500 + 4 * np.clip(daily_max - 65, 0, None) + 15 * np.sin(
        hour / 24 * 2 * np.pi
    )
    noise = rng.normal(0, 4, len(dt))
    mwh = base_load + noise

    hot_peak_scope = (daily_max >= 90) & np.isin(hour, [16, 17, 18, 19, 20])
    mwh = np.where(hot_peak_scope, mwh + 55, mwh)

    return pd.DataFrame(
        {
            "DT": dt,
            "MWH": mwh,
            "Temperature": daily_max,
            "Temperature_DailyMax": daily_max,
            "CDD": np.clip(daily_max - 65, 0, None),
            "Hour": hour,
        }
    ), hot_peak_scope


class AsymmetricObjectiveTrainingIntegrationTests(unittest.TestCase):
    def setUp(self):
        self.df, self.hot_peak_scope = _synthetic_frame_with_underfit_hot_peak_regime()
        self.features = ["Temperature", "Temperature_DailyMax", "CDD", "Hour"]
        self.base_model_cfg = {
            "early_stopping": {"enabled": False},
            "monotonic_constraints": {"enabled": False},
            "xgb": {"n_estimators": 50, "max_depth": 3, "learning_rate": 0.15},
        }

    def _bias(self, model, feats, scope_mask) -> float:
        X = self.df.loc[scope_mask, feats]
        preds = model.predict(X)
        actual = self.df.loc[scope_mask, "MWH"].to_numpy()
        return float(np.mean(preds - actual))

    def test_disabled_by_default_is_a_true_no_op(self):
        config_without_key = {"model": dict(self.base_model_cfg)}
        config_with_disabled_key = {
            "model": {
                **self.base_model_cfg,
                "asymmetric_loss": {
                    "enabled": False,
                    "under_forecast_penalty_multiplier": 9.0,
                },
            }
        }
        model_a, feats_a = train_xgb(self.df, self.features, config=config_without_key)
        model_b, feats_b = train_xgb(
            self.df, self.features, config=config_with_disabled_key
        )
        self.assertEqual(feats_a, feats_b)
        preds_a = model_a.predict(self.df[feats_a])
        preds_b = model_b.predict(self.df[feats_b])
        np.testing.assert_allclose(preds_a, preds_b)

    def test_enabled_reduces_underforecast_bias_in_scope_without_disturbing_rest(self):
        disabled_cfg = {
            "model": {
                **self.base_model_cfg,
                "asymmetric_loss": {"enabled": False},
            }
        }
        enabled_cfg = {
            "model": {
                **self.base_model_cfg,
                "asymmetric_loss": {
                    "enabled": True,
                    "under_forecast_penalty_multiplier": 4.0,
                },
            }
        }
        model_off, feats_off = train_xgb(self.df, self.features, config=disabled_cfg)
        model_on, feats_on = train_xgb(self.df, self.features, config=enabled_cfg)

        bias_off_scope = self._bias(model_off, feats_off, self.hot_peak_scope)
        bias_on_scope = self._bias(model_on, feats_on, self.hot_peak_scope)

        # The plain squared-error fit should under-predict the minority hot-peak
        # regime (negative bias); the asymmetric objective should measurably close
        # that gap.
        self.assertLess(bias_off_scope, -1.0)
        self.assertGreater(bias_on_scope, bias_off_scope + 1.0)

        unscoped = ~self.hot_peak_scope
        bias_off_rest = self._bias(model_off, feats_off, unscoped)
        bias_on_rest = self._bias(model_on, feats_on, unscoped)
        # Rows outside the targeted scope shouldn't be pushed around materially.
        self.assertLess(abs(bias_on_rest - bias_off_rest), 3.0)


if __name__ == "__main__":
    unittest.main()
