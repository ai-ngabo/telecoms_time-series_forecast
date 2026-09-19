"""
phase4_utils.py
__________________
Shared utilities for models (SARIMA, GRU, LightGBM, etc.),
ensuring consistent timing and metric evaluation:
  - time_block(): context manager for wall-clock timing (perf_counter)
  - compute_metrics(): MAE, MAPE, RMSE on inverse-transformed traffic data
  - run_timed(): repeat a callable N times (default 3) and report mean/std
"""
import time
import json
import os
from contextlib import contextmanager
import numpy as np
 
 
@contextmanager
def time_block():
    start = time.perf_counter()
    result = {}
    yield result
    result["seconds"] = time.perf_counter() - start
 
 
def inverse_scale(values, scaler_min, scaler_max):
    """Manually invert MinMax scaling using the stored min/max (data_min_,
    data_max_) rather than re-fitting a scaler object, since we persist
    windows as plain .npz arrays rather than pickled sklearn objects."""
    values = np.asarray(values, dtype="float64")
    return values * (scaler_max - scaler_min) + scaler_min
 
 
def compute_metrics(y_true_scaled, y_pred_scaled, scaler_min, scaler_max,
                     train_scaled=None, seasonal_period=144, eps=1e-6):
    y_true = inverse_scale(y_true_scaled, scaler_min, scaler_max)
    y_pred = inverse_scale(y_pred_scaled, scaler_min, scaler_max)
 
    mae = float(np.mean(np.abs(y_true - y_pred)))
    rmse = float(np.sqrt(np.mean((y_true - y_pred) ** 2)))
    denom = np.where(np.abs(y_true) < eps, eps, np.abs(y_true))
    mape = float(np.mean(np.abs((y_true - y_pred) / denom)) * 100)
 
    metrics = {"MAE": mae, "MAPE": mape, "RMSE": rmse}
 
    if train_scaled is not None:
        train_original = inverse_scale(train_scaled, scaler_min, scaler_max)
        naive_errors = np.abs(train_original[seasonal_period:] - train_original[:-seasonal_period])
        naive_mae = float(np.mean(naive_errors))
        metrics["MASE"] = float(mae / naive_mae) if naive_mae > eps else float("nan")
 
    return metrics, y_true, y_pred
 
 
def run_timed(fn, n_runs=3, *args, **kwargs):
    """Run `fn` n_runs times, returning (result_of_last_run, timing_stats)."""
    times = []
    result = None
    for _ in range(n_runs):
        with time_block() as t:
            result = fn(*args, **kwargs)
        times.append(t["seconds"])
    timing_stats = {
        "n_runs": n_runs,
        "mean_seconds": float(np.mean(times)),
        "std_seconds": float(np.std(times)),
        "all_runs_seconds": times,
    }
    return result, timing_stats
 
 
def save_json(obj, path):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(obj, f, indent=2)
 