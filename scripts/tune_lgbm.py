"""
tune_lightgbm.py
_________________
Hyperparameter tuning for the LightGBM model. Performs grid search over
num_leaves × learning_rate, evaluated only on the validation split (Square 5161)
to avoid test leakage. This aligns with the brief’s allowance to show tuning
evidence from one selected area.

 Outputs:
   - tables/lightgbm_tuning_log.csv (all configs + validation RMSE)
   - tables/lightgbm_tuning_summary.json (final selection + reasoning)

 Usage:
   python scripts/tune_lgbm.py processed_data 5161

"""
import sys
import os
import json
import itertools
import numpy as np
import pandas as pd
import lightgbm as lgb

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from model_utils import inverse_scale, save_json

PROJECT_ROOT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..")
TABLES_DIR = os.path.join(PROJECT_ROOT, "tables")

SHORT_LAGS = [1, 2, 3]

# The grid  (9 combinations) to keep this a genuinely
# fast, re-runnable tuning step rather than an exhaustive search; num_leaves
# controls model complexity/overfitting risk, learning_rate controls how
# aggressively boosting rounds correct residuals.
GRID = {
    "num_leaves": [15, 31, 63],
    "learning_rate": [0.01, 0.05, 0.1],
}


def extract_features(X):
    feats = {}
    for lag in SHORT_LAGS:
        feats[f"lag_{lag}"] = X[:, -lag]
    feats["lag_144"] = X[:, 0]
    feats["rolling_mean_144"] = X.mean(axis=1)
    feats["rolling_std_144"] = X.std(axis=1)
    feats["rolling_min_144"] = X.min(axis=1)
    feats["rolling_max_144"] = X.max(axis=1)
    feats["last_value"] = X[:, -1]
    return pd.DataFrame(feats)


def run_one_config(train_feats, y_train, val_feats, y_val, num_leaves, learning_rate):
    params = {
        "objective": "regression", "metric": "rmse",
        "num_leaves": num_leaves, "learning_rate": learning_rate,
        "verbose": -1, "seed": 42,
    }
    train_set = lgb.Dataset(train_feats, label=y_train)
    val_set = lgb.Dataset(val_feats, label=y_val, reference=train_set)
    model = lgb.train(
        params, train_set, num_boost_round=500, valid_sets=[val_set],
        callbacks=[lgb.early_stopping(stopping_rounds=20, verbose=False)],
    )
    val_pred_scaled = model.predict(val_feats, num_iteration=model.best_iteration)
    return model, val_pred_scaled


def main(data_dir, area_id):
    npz = np.load(os.path.join(data_dir, f"windows_{area_id}.npz"))
    X_train_full, y_train_full = npz["X_train"], npz["y_train"]
    scaler_min, scaler_max = npz["scaler_min"], npz["scaler_max"]

    n_val = int(len(X_train_full) * 0.1)
    X_train, y_train = X_train_full[:-n_val], y_train_full[:-n_val]
    X_val, y_val = X_train_full[-n_val:], y_train_full[-n_val:]

    train_feats = extract_features(X_train)
    val_feats = extract_features(X_val)
    y_val_original = inverse_scale(y_val, scaler_min, scaler_max)

    print(f"=== LightGBM hyperparameter tuning - Square {area_id} ===")
    print(f"Grid: {GRID}")
    print(f"Train: {train_feats.shape}, Val: {val_feats.shape}\n")

    log = []
    for num_leaves, learning_rate in itertools.product(GRID["num_leaves"], GRID["learning_rate"]):
        model, val_pred_scaled = run_one_config(
            train_feats, y_train, val_feats, y_val, num_leaves, learning_rate
        )
        val_pred_original = inverse_scale(val_pred_scaled, scaler_min, scaler_max)
        val_rmse = float(np.sqrt(np.mean((y_val_original - val_pred_original) ** 2)))

        entry = {
            "num_leaves": num_leaves, "learning_rate": learning_rate,
            "best_iteration": int(model.best_iteration), "val_RMSE": val_rmse,
        }
        log.append(entry)
        print(f"  num_leaves={num_leaves:3d}  learning_rate={learning_rate:.2f}  "
              f"best_iter={model.best_iteration:4d}  val_RMSE={val_rmse:.3f}")

    log_df = pd.DataFrame(log).sort_values("val_RMSE")
    log_path = os.path.join(TABLES_DIR, "lightgbm_tuning_log.csv")
    log_df.to_csv(log_path, index=False)

    best = log_df.iloc[0]
    print(f"\nBest config: num_leaves={int(best['num_leaves'])}, "
          f"learning_rate={best['learning_rate']}, val_RMSE={best['val_RMSE']:.3f}")

    reasoning = (
        f"Grid search over {len(log)} (num_leaves x learning_rate) combinations on the "
        f"validation split of Square {area_id}. The best configuration "
        f"(num_leaves={int(best['num_leaves'])}, learning_rate={best['learning_rate']}) "
        f"achieved the lowest validation RMSE ({best['val_RMSE']:.3f}). "
    )
    # Compare against the default used in the main Phase 4 run (num_leaves=31, lr=0.05)
    default_row = log_df[(log_df.num_leaves == 31) & (log_df.learning_rate == 0.05)]
    if not default_row.empty:
        default_rmse = float(default_row.iloc[0]["val_RMSE"])
        diff_pct = (default_rmse - best["val_RMSE"]) / default_rmse * 100
        reasoning += (
            f"The default configuration used in the main Phase 4 results "
            f"(num_leaves=31, learning_rate=0.05) achieved val_RMSE={default_rmse:.3f}, "
            f"{diff_pct:.1f}% {'worse' if diff_pct > 0 else 'better'} than the best found. "
        )
        if abs(diff_pct) < 2:
            reasoning += "This difference is small enough that the default was retained for the final results."
        else:
            reasoning += "The best configuration should replace the default in the final results."

    summary = {
        "model": "LightGBM", "area_id": area_id, "grid": GRID,
        "n_configs_tried": len(log), "best_config": best.to_dict(),
        "reasoning": reasoning,
    }
    save_json(summary, os.path.join(TABLES_DIR, "lightgbm_tuning_summary.json"))
    print(f"\n{reasoning}")
    print(f"\nLog saved to: {log_path}")


if __name__ == "__main__":
    main(sys.argv[1], sys.argv[2])