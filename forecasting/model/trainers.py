from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor
from typing import Any

import pandas as pd

from forecasting.model.xgb_model import train_xgb
from forecasting.model.lgb_model import train_lgb
from forecasting.utils.performance import parallel_tree_training_enabled


def train_tree_models(
    df: pd.DataFrame,
    features: list[str],
    config: dict | None = None,
    stage_name: str = "training",
):
    """Train XGBoost and LightGBM with an adaptive parallel strategy.

    When XGBoost is expected to use CUDA and LightGBM is expected to use CPU, the two
    jobs can run at the same time and use both the GPU and CPU. If GPU is disabled or
    LightGBM GPU is also enabled, the safer default is sequential training with each
    model allowed to use all configured CPU threads.
    """
    cfg = config or {}
    if parallel_tree_training_enabled(cfg):
        print(f"Training XGBoost and LightGBM concurrently for {stage_name} (XGB GPU + LGB CPU performance mode)...")
        with ThreadPoolExecutor(max_workers=2, thread_name_prefix="tree-train") as ex:
            xgb_future = ex.submit(train_xgb, df, features, cfg)
            lgb_future = ex.submit(train_lgb, df, features, cfg)
            xgb_model, xgb_features = xgb_future.result()
            lgb_model, _ = lgb_future.result()
        return xgb_model, lgb_model, xgb_features

    print(f"Training XGBoost and LightGBM sequentially for {stage_name} (all configured threads per model)...")
    xgb_model, xgb_features = train_xgb(df, features, config=cfg)
    lgb_model, _ = train_lgb(df, xgb_features, config=cfg)
    return xgb_model, lgb_model, xgb_features
