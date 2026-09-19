"""
tune_gru.py
 ______________
 Hyperparameter tuning for the GRU model. 
 Uses a small grid (hidden_units × learning_rate, 4 configs) to
 balance evidence with CPU-only runtime limits. Each config runs with a reduced
 epoch ceiling (TUNING_MAX_EPOCHS) to allow relative ranking without requiring
 full convergence.

 Evaluation: validation split only, Square 5161, consistent with tune_lightgbm.py
 and the brief’s “one selected area” allowance.

CLI Usage:
   python scripts/tune_gru.py processed_data 5161
"""

import sys
import os
import itertools
import numpy as np
import pandas as pd
import torch
import torch.nn as nn

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from model_utils import inverse_scale, save_json

PROJECT_ROOT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..")
TABLES_DIR = os.path.join(PROJECT_ROOT, "tables")

GRID = {
    "hidden_units": [32, 64],
    "learning_rate": [1e-3, 5e-4],
}
TUNING_MAX_EPOCHS = 15   # reduced from the full run's 30
TUNING_PATIENCE = 4
BATCH_SIZE = 256

torch.manual_seed(42)


class GRUForecaster(nn.Module):
    def __init__(self, hidden_size):
        super().__init__()
        self.gru = nn.GRU(input_size=1, hidden_size=hidden_size, batch_first=True)
        self.fc = nn.Linear(hidden_size, 1)

    def forward(self, x):
        x = x.unsqueeze(-1)
        out, _ = self.gru(x)
        return self.fc(out[:, -1, :]).squeeze(-1)


def train_one_config(X_train, y_train, X_val, y_val, hidden_units, learning_rate):
    model = GRUForecaster(hidden_units)
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
    loss_fn = nn.MSELoss()

    X_train_t = torch.tensor(X_train, dtype=torch.float32)
    y_train_t = torch.tensor(y_train, dtype=torch.float32)
    X_val_t = torch.tensor(X_val, dtype=torch.float32)
    y_val_t = torch.tensor(y_val, dtype=torch.float32)

    n = X_train_t.shape[0]
    best_val_loss = float("inf")
    best_state = None
    no_improve = 0

    for epoch in range(TUNING_MAX_EPOCHS):
        model.train()
        perm = torch.randperm(n)
        for i in range(0, n, BATCH_SIZE):
            idx = perm[i:i + BATCH_SIZE]
            optimizer.zero_grad()
            loss = loss_fn(model(X_train_t[idx]), y_train_t[idx])
            loss.backward()
            optimizer.step()

        model.eval()
        with torch.no_grad():
            val_loss = loss_fn(model(X_val_t), y_val_t).item()

        if val_loss < best_val_loss - 1e-6:
            best_val_loss = val_loss
            best_state = {k: v.clone() for k, v in model.state_dict().items()}
            no_improve = 0
        else:
            no_improve += 1
            if no_improve >= TUNING_PATIENCE:
                break

    model.load_state_dict(best_state)
    return model, epoch + 1


def main(data_dir, area_id):
    npz = np.load(os.path.join(data_dir, f"windows_{area_id}.npz"))
    X_train_full, y_train_full = npz["X_train"], npz["y_train"]
    scaler_min, scaler_max = npz["scaler_min"], npz["scaler_max"]

    n_val = int(len(X_train_full) * 0.1)
    X_train, y_train = X_train_full[:-n_val], y_train_full[:-n_val]
    X_val, y_val = X_train_full[-n_val:], y_train_full[-n_val:]
    y_val_original = inverse_scale(y_val, scaler_min, scaler_max)

    print(f"=== GRU hyperparameter tuning - Square {area_id} ===")
    print(f"Grid: {GRID}  (reduced epoch ceiling: {TUNING_MAX_EPOCHS})")
    print(f"Train: {X_train.shape}, Val: {X_val.shape}\n")

    log = []
    for hidden_units, learning_rate in itertools.product(GRID["hidden_units"], GRID["learning_rate"]):
        model, epochs_run = train_one_config(X_train, y_train, X_val, y_val, hidden_units, learning_rate)
        model.eval()
        with torch.no_grad():
            val_pred_scaled = model(torch.tensor(X_val, dtype=torch.float32)).numpy()
        val_pred_original = inverse_scale(val_pred_scaled, scaler_min, scaler_max)
        val_rmse = float(np.sqrt(np.mean((y_val_original - val_pred_original) ** 2)))

        entry = {"hidden_units": hidden_units, "learning_rate": learning_rate,
                  "epochs_run": epochs_run, "val_RMSE": val_rmse}
        log.append(entry)
        print(f"  hidden_units={hidden_units:3d}  learning_rate={learning_rate:.4f}  "
              f"epochs_run={epochs_run:2d}  val_RMSE={val_rmse:.3f}")

    log_df = pd.DataFrame(log).sort_values("val_RMSE")
    log_path = os.path.join(TABLES_DIR, "gru_tuning_log.csv")
    log_df.to_csv(log_path, index=False)

    best = log_df.iloc[0]
    print(f"\nBest config: hidden_units={int(best['hidden_units'])}, "
          f"learning_rate={best['learning_rate']}, val_RMSE={best['val_RMSE']:.3f}")

    reasoning = (
        f"Grid search over {len(log)} (hidden_units x learning_rate) combinations, "
        f"each trained with a reduced {TUNING_MAX_EPOCHS}-epoch ceiling (sufficient for "
        f"relative ranking, not full convergence) on the validation split of Square {area_id}. "
        f"The best configuration (hidden_units={int(best['hidden_units'])}, "
        f"learning_rate={best['learning_rate']}) achieved val_RMSE={best['val_RMSE']:.3f}. "
    )
    default_row = log_df[(log_df.hidden_units == 64) & (log_df.learning_rate == 1e-3)]
    if not default_row.empty:
        default_rmse = float(default_row.iloc[0]["val_RMSE"])
        diff_pct = (default_rmse - best["val_RMSE"]) / default_rmse * 100
        reasoning += (
            f"The default used in the main Phase 4 results (hidden_units=64, "
            f"learning_rate=0.001) achieved val_RMSE={default_rmse:.3f}, "
            f"{diff_pct:.1f}% {'worse' if diff_pct > 0 else 'better'} than the best found. "
        )
        if abs(diff_pct) < 3:
            reasoning += "This difference is small enough that the default (64 units) was retained, consistent with the Section 3 efficiency-trade-off justification for GRU over LSTM."
        else:
            reasoning += "The best configuration should replace the default in the final results."

    summary = {
        "model": "GRU", "area_id": area_id, "grid": GRID,
        "tuning_max_epochs": TUNING_MAX_EPOCHS,
        "n_configs_tried": len(log), "best_config": best.to_dict(),
        "reasoning": reasoning,
    }
    save_json(summary, os.path.join(TABLES_DIR, "gru_tuning_summary.json"))
    print(f"\n{reasoning}")
    print(f"\nLog saved to: {log_path}")


if __name__ == "__main__":
    main(sys.argv[1], sys.argv[2])