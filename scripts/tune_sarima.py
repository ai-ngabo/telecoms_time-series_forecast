"""
tune_sarima.py
_______________
Hyperparameter tuning for the SARIMA model, aligned with the documented
experiment format used in tune_gru.py and tune_lightgbm.py. Unlike GRU/LightGBM,
SARIMA tuning relies on pmdarima’s auto_arima() stepwise search over (p,q)
combinations, selecting by AIC. This script captures that search trace
(candidate AIC + fit time) into structured CSV/JSON for proper documentation,
instead of leaving it as console output.

Outputs:
  - CSV/JSON logs of auto_arima search results

CLI Usage:
   python scripts/tune_sarima.py processed_data 5161

"""
import sys
import os
import re
import io
import contextlib
import numpy as np
import pandas as pd
import warnings
warnings.filterwarnings("ignore")
import pmdarima as pm

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from model_utils import save_json

PROJECT_ROOT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..")
TABLES_DIR = os.path.join(PROJECT_ROOT, "tables")
SEASONAL_LAG = 144

TRACE_PATTERN = re.compile(
    r"ARIMA\((\d+),(\d+),(\d+)\)\((\d+),(\d+),(\d+)\)\[(\d+)\]\s*(intercept)?\s*:\s*"
    r"AIC=([\-\d\.]+|inf),\s*Time=([\d\.]+)\s*sec"
)


def seasonal_difference(series, lag=SEASONAL_LAG):
    return series[lag:] - series[:-lag]


def main(data_dir, area_id):
    npz = np.load(os.path.join(data_dir, f"windows_{area_id}.npz"))
    train_scaled = npz["train_scaled"]
    train_diff = seasonal_difference(train_scaled, SEASONAL_LAG)

    print(f"=== SARIMA order search (documented) - Square {area_id} ===")
    print(f"Searching non-seasonal ARMA order on seasonally-differenced series "
          f"({len(train_diff)} points)...\n")

    buf = io.StringIO()
    with contextlib.redirect_stdout(buf):
        model = pm.auto_arima(
            train_diff,
            start_p=0, max_p=4, start_q=0, max_q=4,
            d=0, seasonal=False,
            stepwise=True, suppress_warnings=True, trace=True,
            error_action="ignore",
        )
    trace_text = buf.getvalue()
    print(trace_text)

    rows = []
    for line in trace_text.splitlines():
        m = TRACE_PATTERN.search(line)
        if m:
            p, d, q, P, D, Q, m_period, intercept, aic, fit_time = m.groups()
            rows.append({
                "order": f"({p},{d},{q})",
                "intercept": bool(intercept),
                "AIC": float(aic) if aic != "inf" else float("inf"),
                "fit_time_sec": float(fit_time),
            })

    log_df = pd.DataFrame(rows).sort_values("AIC")
    log_path = os.path.join(TABLES_DIR, "sarima_order_search_log.csv")
    log_df.to_csv(log_path, index=False)

    best = log_df.iloc[0]
    print(f"\nBest order: {best['order']}, AIC={best['AIC']:.3f}, "
          f"fit_time={best['fit_time_sec']:.2f}s")
    print(f"Total candidates evaluated: {len(log_df)}")

    reasoning = (
        f"pmdarima's stepwise auto_arima search evaluated {len(log_df)} candidate "
        f"non-seasonal ARMA(p,q) orders (searched on the seasonally-differenced series, "
        f"see sarima.py's module docstring for why native seasonal SARIMAX was "
        f"computationally intractable) on Square {area_id}'s training data, selecting "
        f"by AIC (Akaike Information Criterion - balances model fit against complexity, "
        f"penalizing unnecessary extra parameters). "
        f"The selected order {best['order']} achieved the lowest AIC "
        f"({best['AIC']:.3f}) among all candidates tried, and was used for the final "
        f"Phase 4 SARIMA results on this area."
    )

    summary = {
        "model": "SARIMA", "area_id": area_id,
        "search_method": "pmdarima auto_arima (stepwise, AIC-selected)",
        "n_candidates_evaluated": len(log_df),
        "best_order": best["order"], "best_AIC": float(best["AIC"]),
        "reasoning": reasoning,
    }
    save_json(summary, os.path.join(TABLES_DIR, "sarima_order_search_summary.json"))
    print(f"\n{reasoning}")
    print(f"\nLog saved to: {log_path}")


if __name__ == "__main__":
    main(sys.argv[1], sys.argv[2])