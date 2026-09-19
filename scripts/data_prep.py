"""
data_prep.py
___________________
Shared preprocessing for models (SARIMA, GRU, LightGBM, etc.),
ensuring identical train/test splits and scaling:
  - Train ≤ 2013-12-15, Test = 2013-12-16–22 (temporal, no shuffle)
  - Window length p=144 (24h) from Phase 2 analysis
  - MinMax scaling fitted on train only (avoids leakage)
  - One-step-ahead framing: predict x(t+1) from past p values
Outputs cached windowed arrays for top-traffic areas so all models
load consistent data directly.
"""

import sys
import os
import json
import numpy as np
import pandas as pd
from sklearn.preprocessing import MinMaxScaler

PROJECT_ROOT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..")
TABLES_DIR = os.path.join(PROJECT_ROOT, "tables")
DATA_DIR = os.path.join(PROJECT_ROOT, "processed_data")

WINDOW_P = 144  # 24h, from ACF/PACF evidence (Phase 2)
TRAIN_END = "2013-12-15 23:50:00"       
TEST_START = "2013-12-16 00:00:00"
TEST_END = "2013-12-22 23:50:00"         


def get_top3_ids(matrix_path):
    """Reuse the Phase 2 top-3 result if cached; otherwise recompute."""
    top3_csv = os.path.join(TABLES_DIR, "top3_areas.csv")
    if os.path.exists(top3_csv):
        top3 = pd.read_csv(top3_csv, index_col=0)
        return [str(i) for i in top3.index.tolist()]
    df = pd.read_parquet(matrix_path)
    total = df.sum(axis=0)
    return total.sort_values(ascending=False).head(3).index.tolist()


def make_windows(series_scaled, p):
    """
    Build sliding windows for one-step-ahead forecasting:
    X[i] = series[i : i+p]  ->  y[i] = series[i+p]
    """
    X, y = [], []
    for i in range(len(series_scaled) - p):
        X.append(series_scaled[i:i + p])
        y.append(series_scaled[i + p])
    return np.array(X, dtype="float32"), np.array(y, dtype="float32")


def prepare_area(df, area_id, p=WINDOW_P):
    series = df[area_id].astype("float32")

    train_raw = series.loc[:TRAIN_END].values.reshape(-1, 1)
    test_raw = series.loc[TEST_START:TEST_END].values.reshape(-1, 1)

    # Scaler fit on TRAIN ONLY
    scaler = MinMaxScaler(feature_range=(0, 1))
    train_scaled = scaler.fit_transform(train_raw).flatten()
    test_scaled = scaler.transform(test_raw).flatten()

    test_context = np.concatenate([train_scaled[-p:], test_scaled])

    X_train, y_train = make_windows(train_scaled, p)
    X_test, y_test = make_windows(test_context, p)

    return {
        "area_id": area_id,
        "X_train": X_train, "y_train": y_train,
        "X_test": X_test, "y_test": y_test,
        "train_scaled": train_scaled,     # train sequential scaled series for SARIMA
        "test_scaled": test_scaled,       # test sequential scaled series  for SARIMA
        "scaler": scaler,
        "train_index": series.loc[:TRAIN_END].index[p:],
        "test_index": series.loc[TEST_START:TEST_END].index,
    }


def main(matrix_path):
    os.makedirs(DATA_DIR, exist_ok=True)
    df = pd.read_parquet(matrix_path)
    df.index = pd.to_datetime(df.index)

    top3_ids = get_top3_ids(matrix_path)
    print(f"Top-3 areas: {top3_ids}")
    print(f"Window length p = {WINDOW_P} (24h)")
    print(f"Train: start -> {TRAIN_END}")
    print(f"Test:  {TEST_START} -> {TEST_END}")

    summary = {}
    for area_id in top3_ids:
        prepared = prepare_area(df, area_id)
        out_path = os.path.join(DATA_DIR, f"windows_{area_id}.npz")
        np.savez(
            out_path,
            X_train=prepared["X_train"], y_train=prepared["y_train"],
            X_test=prepared["X_test"], y_test=prepared["y_test"],
            train_scaled=prepared["train_scaled"], test_scaled=prepared["test_scaled"],
            scaler_min=prepared["scaler"].data_min_,
            scaler_max=prepared["scaler"].data_max_,
            train_index=prepared["train_index"].values.astype("datetime64[ns]"),
            test_index=prepared["test_index"].values.astype("datetime64[ns]"),
        )
        summary[area_id] = {
            "X_train_shape": list(prepared["X_train"].shape),
            "X_test_shape": list(prepared["X_test"].shape),
            "saved_to": out_path,
        }
        print(f"  [{area_id}] X_train={prepared['X_train'].shape} "
              f"X_test={prepared['X_test'].shape}  -> {out_path}")

    with open(os.path.join(TABLES_DIR, "data_prep_summary.json"), "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)
    print("\nDone. All 3 areas' windowed train/test arrays cached in processed_data/.")


if __name__ == "__main__":
    main(sys.argv[1])