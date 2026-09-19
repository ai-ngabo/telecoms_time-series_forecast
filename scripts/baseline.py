"""
phase1_baseline.py
__________________

Baseline measurement for Section 1: a naive load of a single raw MI2 
day-file using pandas defaults (no column filtering, dtype control, 
or chunking). This represents the memory footprint of a first-pass 
implementation.

Run as a standalone process to ensure peak RSS measurements are not 
affected by objects from other stages.
"""

import sys
import os
import json
import pandas as pd
from memory_profile import PeakMemoryMonitor

TABLES_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "tables")

COLS = ["square_id", "time_ms", "country_code",
        "sms_in", "sms_out", "call_in", "call_out", "internet"]


def naive_load(path):
    df = pd.read_csv(path, sep="\t", header=None, names=COLS)
    return df


def main(path):
    with PeakMemoryMonitor() as mon:
        df = naive_load(path)
        # I've decided to work with the "internet" column for the rest of the analysis, so
        # force a sum to ensure the column is actually loaded into memory.
        _ = df["internet"].sum()

    result = {
        "stage": "baseline_naive_load",
        "rows": int(len(df)),
        "columns": list(df.columns),
        "dtypes": {c: str(t) for c, t in df.dtypes.items()},
        "pandas_reported_memory_mb": float(df.memory_usage(deep=True).sum() / 1e6),
        "baseline_rss_mb": mon.baseline_mb,
        "peak_rss_mb": mon.peak_mb,
        "delta_rss_mb": mon.delta_mb,
    }
    mon.report("BASELINE naive load")
    print(json.dumps(result, indent=2))

    os.makedirs(TABLES_DIR, exist_ok=True)
    out_path = os.path.join(TABLES_DIR, "baseline_result.json")
    with open(out_path, "w") as f:
        json.dump(result, f, indent=2)
    print(f"\nSaved to: {os.path.abspath(out_path)}")


if __name__ == "__main__":
    main(sys.argv[1])