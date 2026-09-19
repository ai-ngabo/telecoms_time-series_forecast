"""
exp_data_analysis.py
__________________

Section 2 (Exploratory Data Analysis) for the MI2 traffic forecasting project.

Run from the project root:
    python scripts/exp_data_analysis.py processed_data/full_wide_matrix_novdec.parquet

Outputs (saved under figures/ and tables/, created relative to the project root):

1. figures/fig1_traffic_distribution.png  
   - Histogram (linear + log scale) of total internet traffic per area 
     across Nov–Dec, with summary statistics.

2. tables/top3_areas.csv  
   - IDs of the 3 areas with the highest total traffic.

3. figures/fig2_two_week_series.png  
   - Time series for the first two weeks, showing: top-3 areas, Square 4159, Square 4556.

4. figures/fig3_acf_pacf_top_area.png  
   - ACF/PACF plots for the top-traffic area’s full-period series.

5. figures/fig4_stl_decomposition_top_area.png  
   - STL decomposition (trend, seasonal, residual) of the top-traffic area.

6. tables/adf_test_top_area.json  
   - Augmented Dickey-Fuller stationarity test results for the top area’s series, 
     plus comparison on its residual component.
"""

import sys
import os
import json
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from statsmodels.graphics.tsaplots import plot_acf, plot_pacf
from statsmodels.tsa.seasonal import STL
from statsmodels.tsa.stattools import adfuller

PROJECT_ROOT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..")
FIG_DIR = os.path.join(PROJECT_ROOT, "figures")
TABLE_DIR = os.path.join(PROJECT_ROOT, "tables")

REFERENCE_SQUARES = ["4159", "4556"]   #square IDs of two reference areas with low traffic
DAILY_STEPS = 144   # 10-min steps in 24h
WEEKLY_STEPS = DAILY_STEPS * 7  # 10-min steps in 7 days


def ensure_dirs():
    os.makedirs(FIG_DIR, exist_ok=True)
    os.makedirs(TABLE_DIR, exist_ok=True)


def fig1_distribution(df):
    total_traffic = df.sum(axis=0)  # one value per square_id, summed over full period
    stats = {
        "mean": float(total_traffic.mean()),
        "median": float(total_traffic.median()),
        "std": float(total_traffic.std()),
        "skew": float(total_traffic.skew()),
        "min": float(total_traffic.min()),
        "max": float(total_traffic.max()),
    }

    fig, axes = plt.subplots(1, 2, figsize=(12, 4.5))
    axes[0].hist(total_traffic, bins=80, color="steelblue", edgecolor="none")
    axes[0].set_title("Total internet traffic per area\n(linear scale)")
    axes[0].set_xlabel("Total traffic (Nov 1 - Dec 31) 2013")
    axes[0].set_ylabel("Number of areas")

    axes[1].hist(total_traffic, bins=80, color="darkorange", edgecolor="none")
    axes[1].set_yscale("log")
    axes[1].set_title("Total internet traffic per area\n(log-scale y-axis)")
    axes[1].set_xlabel("Total traffic (Nov 1 - Dec 31) 2013")
    axes[1].set_ylabel("Number of areas (log)")

    fig.suptitle("Figure 1: Distribution of total internet traffic across 10,000 areas", y=1.03)
    fig.tight_layout()
    out_path = os.path.join(FIG_DIR, "fig1_traffic_distribution.png")
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)

    with open(os.path.join(TABLE_DIR, "traffic_distribution_stats.json"), "w") as f:
        json.dump(stats, f, indent=2)

    print(f"[fig1] saved to {out_path}")
    print(f"[fig1] stats: {stats}")
    return total_traffic, stats


def identify_top3(total_traffic):
    top3 = total_traffic.sort_values(ascending=False).head(3)
    top3_ids = top3.index.tolist()
    top3.to_csv(os.path.join(TABLE_DIR, "top3_areas.csv"), header=["total_traffic"])
    print(f"[top3] {top3_ids}")
    print(top3)
    return top3_ids


def fig2_two_week_series(df, top3_ids):
    cols_to_plot = top3_ids + REFERENCE_SQUARES
    labels = [f"Square {c} (top traffic)" for c in top3_ids] + \
             [f"Square {c} (reference)" for c in REFERENCE_SQUARES]

    start = pd.Timestamp("2013-11-01 00:00:00")
    end = start + pd.Timedelta(days=14)
    window = df.loc[start:end - pd.Timedelta(minutes=10)]

    fig, axes = plt.subplots(len(cols_to_plot), 1, figsize=(13, 2.2 * len(cols_to_plot)), sharex=True)
    for ax, col, label in zip(axes, cols_to_plot, labels):
        ax.plot(window.index, window[col], linewidth=0.8, color="teal")
        ax.set_ylabel("Internet\ntraffic", fontsize=9)
        ax.set_title(label, fontsize=10, loc="left")
        ax.grid(alpha=0.3)
    axes[-1].set_xlabel("Time")
    fig.suptitle("Figure 2: First two weeks (Nov 1-14, 2013) - top-3 areas + reference squares", y=1.01)
    fig.tight_layout()
    out_path = os.path.join(FIG_DIR, "fig2_two_week_series.png")
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"[fig2] saved to {out_path}  (window: {window.index.min()} -> {window.index.max()})")


def fig3_acf_pacf(df, top_id, max_lag=WEEKLY_STEPS):
    series = df[top_id]
    fig, axes = plt.subplots(2, 1, figsize=(12, 7))
    plot_acf(series, lags=max_lag, ax=axes[0])
    axes[0].set_title(f"ACF - Square {top_id} (up to {max_lag} lags = 7 days)")
    axes[0].axvline(DAILY_STEPS, color="red", linestyle="--", linewidth=0.8, label="24h lag")
    axes[0].axvline(DAILY_STEPS * 2, color="red", linestyle="--", linewidth=0.8)
    axes[0].axvline(DAILY_STEPS * 7, color="green", linestyle="--", linewidth=0.8, label="7-day lag")
    axes[0].legend()

    plot_pacf(series, lags=min(max_lag, DAILY_STEPS * 2), ax=axes[1], method="ywm")
    axes[1].set_title(f"PACF - Square {top_id} (up to {min(max_lag, DAILY_STEPS*2)} lags)")
    axes[1].axvline(DAILY_STEPS, color="red", linestyle="--", linewidth=0.8)

    fig.suptitle(f"Figure 3: ACF/PACF for the top-traffic area (Square {top_id})", y=1.02)
    fig.tight_layout()
    out_path = os.path.join(FIG_DIR, "fig3_acf_pacf_top_area.png")
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"[fig3] saved to {out_path}")


def fig4_stl_and_adf(df, top_id):
    series = df[top_id]
    stl = STL(series, period=DAILY_STEPS, robust=True)
    result = stl.fit()

    fig, axes = plt.subplots(4, 1, figsize=(13, 9), sharex=True)
    axes[0].plot(series.index, series.values, linewidth=0.6, color="black")
    axes[0].set_title("Observed")
    axes[1].plot(series.index, result.trend, linewidth=0.8, color="steelblue")
    axes[1].set_title("Trend")
    axes[2].plot(series.index, result.seasonal, linewidth=0.6, color="darkorange")
    axes[2].set_title("Seasonal (24h period)")
    axes[3].plot(series.index, result.resid, linewidth=0.4, color="gray")
    axes[3].set_title("Residual")
    axes[3].set_xlabel("Time")

    fig.suptitle(f"Figure 4: STL decomposition - Square {top_id} (period=144 steps = 24h)", y=1.01)
    fig.tight_layout()
    out_path = os.path.join(FIG_DIR, "fig4_stl_decomposition_top_area.png")
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"[fig4] saved to {out_path}")

    # ADF tests: on the raw series, and on the residual (post-decomposition)
    adf_raw = adfuller(series.dropna(), autolag="AIC")
    adf_resid = adfuller(result.resid.dropna(), autolag="AIC")

    def pack(res):
        return {
            "adf_statistic": float(res[0]),
            "p_value": float(res[1]),
            "n_lags_used": int(res[2]),
            "n_obs": int(res[3]),
            "critical_values": {k: float(v) for k, v in res[4].items()},
            "stationary_at_5pct": bool(res[0] < res[4]["5%"]),
        }

    adf_result = {
        "raw_series": pack(adf_raw),
        "stl_residual": pack(adf_resid),
    }
    with open(os.path.join(TABLE_DIR, "adf_test_top_area.json"), "w") as f:
        json.dump(adf_result, f, indent=2)
    print(f"[adf] raw series stationary at 5%: {adf_result['raw_series']['stationary_at_5pct']}")
    print(f"[adf] STL residual stationary at 5%: {adf_result['stl_residual']['stationary_at_5pct']}")
    return adf_result


def main(matrix_path):
    ensure_dirs()
    df = pd.read_parquet(matrix_path)
    df.index = pd.to_datetime(df.index)
    print(f"Loaded matrix: {df.shape}, {df.index.min()} -> {df.index.max()}")

    total_traffic, dist_stats = fig1_distribution(df)
    top3_ids = identify_top3(total_traffic)
    fig2_two_week_series(df, top3_ids)
    top_id = top3_ids[0]
    fig3_acf_pacf(df, top_id)
    adf_result = fig4_stl_and_adf(df, top_id)

    print("\n=== PHASE 2 SUMMARY ===")
    print(f"Top-3 areas by total traffic: {top3_ids}")
    print(f"Distribution skew: {dist_stats['skew']:.2f}")
    print(f"Top area ({top_id}) raw series stationary (ADF, 5%): "
          f"{adf_result['raw_series']['stationary_at_5pct']}")


if __name__ == "__main__":
    main(sys.argv[1])