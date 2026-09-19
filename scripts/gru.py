"""""
gru.py
___________________
Model: GRU baseline.
Architecture: single GRU layer (64 units) + dense output for one-step prediction.
Chosen as a faster, efficient alternative to LSTM given CPU-only hardware limits.
Training: Adam optimiser, MSE loss, early stopping on a validation slice
from the end of the training period (temporal order preserved, test set untouched).
"""
import sys
import os
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
 
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from model_utils import compute_metrics, run_timed, save_json
 
PROJECT_ROOT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..")
TABLES_DIR = os.path.join(PROJECT_ROOT, "tables")
 
HIDDEN_UNITS = 64
N_LAYERS = 1
MAX_EPOCHS = 30 
PATIENCE = 5
BATCH_SIZE = 256
LEARNING_RATE = 1e-3
VAL_FRACTION = 0.1       # last 10% of TRAIN (temporally) held out for early stopping
 
torch.manual_seed(42) 
 
 
class GRUForecaster(nn.Module):
    def __init__(self, hidden_size=HIDDEN_UNITS, n_layers=N_LAYERS):
        super().__init__()
        self.gru = nn.GRU(input_size=1, hidden_size=hidden_size,
                           num_layers=n_layers, batch_first=True)
        self.fc = nn.Linear(hidden_size, 1)
 
    def forward(self, x):
        # x: (batch, seq_len) -> (batch, seq_len, 1) for GRU's expected input shape
        x = x.unsqueeze(-1)
        out, _ = self.gru(x)
        last_hidden = out[:, -1, :]  # final timestep's hidden state
        return self.fc(last_hidden).squeeze(-1)
 
 
def train_gru(X_train, y_train, X_val, y_val):
    model = GRUForecaster()
    optimizer = torch.optim.Adam(model.parameters(), lr=LEARNING_RATE)
    loss_fn = nn.MSELoss()
 
    X_train_t = torch.tensor(X_train, dtype=torch.float32)
    y_train_t = torch.tensor(y_train, dtype=torch.float32)
    X_val_t = torch.tensor(X_val, dtype=torch.float32)
    y_val_t = torch.tensor(y_val, dtype=torch.float32)
 
    n_samples = X_train_t.shape[0]
    best_val_loss = float("inf")
    best_state = None
    epochs_without_improvement = 0
 
    for epoch in range(MAX_EPOCHS):
        model.train()
        perm = torch.randperm(n_samples)
        for i in range(0, n_samples, BATCH_SIZE):
            idx = perm[i:i + BATCH_SIZE]
            xb, yb = X_train_t[idx], y_train_t[idx]
            optimizer.zero_grad()
            pred = model(xb)
            loss = loss_fn(pred, yb)
            loss.backward()
            optimizer.step()
 
        model.eval()
        with torch.no_grad():
            val_pred = model(X_val_t)
            val_loss = loss_fn(val_pred, y_val_t).item()
 
        if val_loss < best_val_loss - 1e-6:
            best_val_loss = val_loss
            best_state = {k: v.clone() for k, v in model.state_dict().items()}
            epochs_without_improvement = 0
        else:
            epochs_without_improvement += 1
            if epochs_without_improvement >= PATIENCE:
                print(f"  Early stopping at epoch {epoch+1} (best val_loss={best_val_loss:.6f})")
                break
 
    model.load_state_dict(best_state)
    return model, {"final_epoch": epoch + 1, "best_val_loss": best_val_loss}
 
 
def run_for_area(data_dir, area_id):
    npz = np.load(os.path.join(data_dir, f"windows_{area_id}.npz"))
    X_train_full, y_train_full = npz["X_train"], npz["y_train"]
    X_test, y_test = npz["X_test"], npz["y_test"]
    scaler_min, scaler_max = npz["scaler_min"], npz["scaler_max"]
    train_scaled = npz["train_scaled"]
    test_index = pd.to_datetime(npz["test_index"])

    n_val = int(len(X_train_full) * VAL_FRACTION)
    X_train, y_train = X_train_full[:-n_val], y_train_full[:-n_val]
    X_val, y_val = X_train_full[-n_val:], y_train_full[-n_val:]
 
    print(f"\n=== GRU - Square {area_id} ===")
    print(f"Train: {X_train.shape}, Val: {X_val.shape}, Test: {X_test.shape}")
 
    def _train():
        return train_gru(X_train, y_train, X_val, y_val)

    (model, train_info), train_timing = run_timed(_train, n_runs=1)
 
    def _infer():
        model.eval()
        with torch.no_grad():
            preds = model(torch.tensor(X_test, dtype=torch.float32)).numpy()
        return preds
 
    preds, infer_timing = run_timed(_infer, n_runs=3)
 
    metrics, y_true, y_pred = compute_metrics(y_test, preds, scaler_min, scaler_max,
                                               train_scaled=train_scaled)
    print(f"Metrics: {metrics}")
    print(f"Train time (single timed run): {train_timing['mean_seconds']:.2f}s")
    print(f"Inference time (mean of 3 runs, full test set): {infer_timing['mean_seconds']:.4f}s")
 
    result = {
        "model": "GRU",
        "architecture": {"hidden_units": HIDDEN_UNITS, "n_layers": N_LAYERS,
                          "window_p": X_train.shape[1]},
        "training_config": {"max_epochs": MAX_EPOCHS, "patience": PATIENCE,
                             "batch_size": BATCH_SIZE, "learning_rate": LEARNING_RATE,
                             "actual_epochs_run": train_info["final_epoch"],
                             "best_val_loss": train_info["best_val_loss"]},
        "area_id": area_id,
        "metrics": metrics,
        "train_timing": train_timing,
        "inference_timing": infer_timing,
    }
    save_json(result, os.path.join(TABLES_DIR, f"gru_{area_id}_result.json"))
 
    np.savez(
        os.path.join(data_dir, f"gru_preds_{area_id}.npz"),
        y_true=y_true, y_pred=y_pred, test_index=test_index.values.astype("datetime64[ns]"),
    )
    return result
 
 
if __name__ == "__main__":
    data_dir, area_id = sys.argv[1], sys.argv[2]
    run_for_area(data_dir, area_id)
 