"""
sarima.py
__________________
Model: SARIMA baseline.
Native seasonal SARIMA with m=144 (daily cycle at 10-min resolution)
was computationally infeasible on CPU hardware. Instead, we apply
explicit seasonal differencing (series[t] - series[t-144]) and fit a
fast non-seasonal ARMA(p,q) via auto_arima. Forecasts are reconstructed
by adding back series[t-144], ensuring no leakage. This approach is
equivalent in spirit to SARIMA(p,0,q)(0,1,0)[144] but tractable in practice.
"""
import sys
import os
import numpy as np
import pandas as pd
import warnings
warnings.filterwarnings("ignore")
 
from statsmodels.tsa.statespace.sarimax import SARIMAX
import pmdarima as pm
 
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from model_utils import time_block, compute_metrics, run_timed, save_json
 
PROJECT_ROOT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..")
TABLES_DIR = os.path.join(PROJECT_ROOT, "tables")
SEASONAL_LAG = 144  # daily period at 10-min resolution
 
 
def seasonal_difference(series, lag=SEASONAL_LAG):
    return series[lag:] - series[:-lag]
 
 
def search_order(diff_series):
    """Fast non-seasonal order search on the small-state-dimension
    differenced series."""
    model = pm.auto_arima(
        diff_series,
        start_p=0, max_p=4, start_q=0, max_q=4,
        d=0,  # already stationary by construction (seasonally differenced)
        seasonal=False,
        stepwise=True, suppress_warnings=True, trace=True,
        error_action="ignore",
    )
    return model.order
 
 
def fit_arma(diff_series, order):
    model = SARIMAX(diff_series, order=order,
                     enforce_stationarity=True, enforce_invertibility=True)
    return model.fit(disp=False, maxiter=100)
 
 
def one_step_ahead_forecast(fitted, full_series, order, n_train_diff, test_len, lag=SEASONAL_LAG):
    """
    Walk-forward one-step-ahead forecasting on the ORIGINAL (non-differenced)
    scale. At each test step t:
      1. Forecast the next differenced value from the ARMA model's state.
      2. Reconstruct the original-scale prediction by adding series[t-lag]
      3. Append the TRUE observed differenced value (computed from the
         real series, not the prediction) to advance the ARMA model's
         state before the next step
    """
    current_fit = fitted
    preds_original_scale = []
    n_train = n_train_diff + lag  # differenced series is `lag` shorter than original
    for i in range(test_len):
        t = n_train + i
        diff_pred = current_fit.forecast(steps=1)[0]
        original_pred = diff_pred + full_series[t - lag]
        preds_original_scale.append(original_pred)
 
        true_diff = full_series[t] - full_series[t - lag]
        current_fit = current_fit.append([true_diff], refit=False)
    return np.array(preds_original_scale)
 
 
def run_for_area(data_dir, area_id):
    npz = np.load(os.path.join(data_dir, f"windows_{area_id}.npz"))
    train_scaled = npz["train_scaled"]
    test_scaled = npz["test_scaled"]
    scaler_min, scaler_max = npz["scaler_min"], npz["scaler_max"]
    test_index = pd.to_datetime(npz["test_index"])
 
    full_series = np.concatenate([train_scaled, test_scaled])
    train_diff = seasonal_difference(train_scaled, SEASONAL_LAG)
 
    print(f"\n=== SARIMA (seasonally-differenced ARMA) - Square {area_id} ===")
    print(f"Train length: {len(train_scaled)}, differenced: {len(train_diff)}")
    print("Searching non-seasonal ARMA order on differenced series...")
    order = search_order(train_diff)
    print(f"Selected order: {order}")
 
    def _train():
        return fit_arma(train_diff, order)
 
    fitted, train_timing = run_timed(_train, n_runs=3)
 
    def _infer():
        return one_step_ahead_forecast(
            fitted, full_series, order, len(train_diff), len(test_scaled), SEASONAL_LAG
        )
 
    preds, infer_timing = run_timed(_infer, n_runs=3)
 
    metrics, y_true, y_pred = compute_metrics(test_scaled, preds, scaler_min, scaler_max,
                                               train_scaled=train_scaled)
    print(f"Metrics: {metrics}")
    print(f"Train time (mean of 3 fits): {train_timing['mean_seconds']:.3f}s "
          f"(+/- {train_timing['std_seconds']:.3f}s)")
    print(f"Inference time (mean of 3 runs, full 1008-step test set): "
          f"{infer_timing['mean_seconds']:.2f}s")
 
    result = {
        "model": "SARIMA",
        "implementation_note": "seasonal differencing at lag=144 + non-seasonal ARMA "
                                "(native seasonal state-space was intractable on CPU; "
                                "see module docstring)",
        "area_id": area_id,
        "arma_order": list(order),
        "seasonal_lag": SEASONAL_LAG,
        "metrics": metrics,
        "train_timing": train_timing,
        "inference_timing": infer_timing,
    }
    save_json(result, os.path.join(TABLES_DIR, f"sarima_{area_id}_result.json"))
 
    np.savez(
        os.path.join(data_dir, f"sarima_preds_{area_id}.npz"),
        y_true=y_true, y_pred=y_pred, test_index=test_index.values.astype("datetime64[ns]"),
    )
    return result
 
 
if __name__ == "__main__":
    data_dir, area_id = sys.argv[1], sys.argv[2]
    run_for_area(data_dir, area_id)
 