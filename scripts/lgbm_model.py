"""
lightgbm_model.py
__________________
Model: LightGBM baseline.
Uses engineered lag features based on EDA:
  - Short-range lags (1–3) for autoregressive dependency
  - Daily-cycle lags (144, 288) for 24h periodicity
  - Weekly-cycle lag (1008) for strong 7-day correlation
  - Rolling mean/std (144) for short-term level/volatility shifts
Features are derived from the same windowed arrays as GRU, ensuring
identical data splits/scaling across all models. Only the input
representation differs.
"""
import sys
import os
import numpy as np
import pandas as pd
import lightgbm as lgb
 
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from model_utils import compute_metrics, run_timed, save_json
 
PROJECT_ROOT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..")
TABLES_DIR = os.path.join(PROJECT_ROOT, "tables")
SHORT_LAGS = [1, 2, 3]
DAILY_LAGS = [144]
ROLLING_WINDOW = 144
 
 
def extract_features(X):
    """
    X: array of shape (n_samples, 144) -- each row is the p=144 window
    immediately preceding the target. Since p=144 exactly equals one
    day, lag-144 (1 day ago relative to the TARGET) is just X[:, 0]
    (the oldest, first point in the window). Short lags (1,2,3 steps
    before target) are the LAST few points in the window: X[:, -1],
    X[:, -2], X[:, -3].
    """
    n = X.shape[0]
    feats = {}
    for lag in SHORT_LAGS:
        feats[f"lag_{lag}"] = X[:, -lag]
    feats["lag_144"] = X[:, 0]      # oldest point in window = 1 day before target
    feats["rolling_mean_144"] = X.mean(axis=1)
    feats["rolling_std_144"] = X.std(axis=1)
    feats["rolling_min_144"] = X.min(axis=1)
    feats["rolling_max_144"] = X.max(axis=1)
    feats["last_value"] = X[:, -1]  # == lag_1, kept for robustness
    return pd.DataFrame(feats)
 
 
def train_lightgbm(X_train_feats, y_train, X_val_feats, y_val):
    train_set = lgb.Dataset(X_train_feats, label=y_train)
    val_set = lgb.Dataset(X_val_feats, label=y_val, reference=train_set)
 
    params = {
        "objective": "regression",
        "metric": "rmse",
        "num_leaves": 31,
        "learning_rate": 0.05,
        "verbose": -1,
        "seed": 42,
    }
    model = lgb.train(
        params, train_set, num_boost_round=500,
        valid_sets=[val_set],
        callbacks=[lgb.early_stopping(stopping_rounds=20, verbose=False)],
    )
    return model
 
 
def run_for_area(data_dir, area_id):
    npz = np.load(os.path.join(data_dir, f"windows_{area_id}.npz"))
    X_train_full, y_train_full = npz["X_train"], npz["y_train"]
    X_test, y_test = npz["X_test"], npz["y_test"]
    scaler_min, scaler_max = npz["scaler_min"], npz["scaler_max"]
    train_scaled = npz["train_scaled"]
    test_index = pd.to_datetime(npz["test_index"])
 
    n_val = int(len(X_train_full) * 0.1)
    X_train, y_train = X_train_full[:-n_val], y_train_full[:-n_val]
    X_val, y_val = X_train_full[-n_val:], y_train_full[-n_val:]
 
    train_feats = extract_features(X_train)
    val_feats = extract_features(X_val)
    test_feats = extract_features(X_test)
 
    print(f"\n=== LightGBM - Square {area_id} ===")
    print(f"Features: {list(train_feats.columns)}")
    print(f"Train: {train_feats.shape}, Val: {val_feats.shape}, Test: {test_feats.shape}")
 
    def _train():
        return train_lightgbm(train_feats, y_train, val_feats, y_val)
 
    model, train_timing = run_timed(_train, n_runs=3)  # cheap enough for 3 runs
 
    def _infer():
        return model.predict(test_feats, num_iteration=model.best_iteration)
 
    preds, infer_timing = run_timed(_infer, n_runs=3)
 
    metrics, y_true, y_pred = compute_metrics(y_test, preds, scaler_min, scaler_max,
                                               train_scaled=train_scaled)
    print(f"Metrics: {metrics}")
    print(f"Best iteration: {model.best_iteration}")
    print(f"Train time (mean of 3 runs): {train_timing['mean_seconds']:.3f}s "
          f"(+/- {train_timing['std_seconds']:.3f}s)")
    print(f"Inference time (mean of 3 runs): {infer_timing['mean_seconds']:.4f}s")
 
    feature_importance = dict(zip(train_feats.columns,
                                   [int(x) for x in model.feature_importance()]))
 
    result = {
        "model": "LightGBM",
        "features": list(train_feats.columns),
        "feature_importance": feature_importance,
        "best_iteration": int(model.best_iteration),
        "area_id": area_id,
        "metrics": metrics,
        "train_timing": train_timing,
        "inference_timing": infer_timing,
    }
    save_json(result, os.path.join(TABLES_DIR, f"lightgbm_{area_id}_result.json"))
 
    np.savez(
        os.path.join(data_dir, f"lightgbm_preds_{area_id}.npz"),
        y_true=y_true, y_pred=y_pred, test_index=test_index.values.astype("datetime64[ns]"),
    )
    return result
 
 
if __name__ == "__main__":
    data_dir, area_id = sys.argv[1], sys.argv[2]
    run_for_area(data_dir, area_id)
 