"""
phase1_optimized.py
__________________

"After" pipeline for Section 1: optimized load of a single raw MI2 day-file 
using four techniques:

1. Column selection  
   - Only `square_id`, `time_interval_ms`, and `internet` are read.  
   - Unused fields (sms/call in/out) are dropped at read time via `usecols`.

2. Chunked loading  
   - File is streamed in fixed-size chunks (`chunksize=...`) instead of 
     being fully materialized in memory.

3. Dtype downcasting  
   - `square_id`: int64 -> uint16 (max value 10,000 fits in 16 bits).  
   - `internet`: float64 -> float32 (single precision sufficient for traffic volumes).

4. Aggregation  
   - Raw rows exist per `(square_id, time_interval_ms, country_code)`.  
   - Internet traffic is summed across country codes per chunk, reducing 
     ~4.8M raw rows/day to ~1.44M `(square_id, time_interval_ms)` rows/day.

Output:  
- One long-format Parquet file per day (`square_id`, `time_interval_ms`, `internet`).  
- Downstream phases pivot these into the wide (time × area) matrix as needed.  
- Long-format Parquet is chosen for persistence because columnar compression 
  is efficient and schema remains stable/extensible.  
- The wide pivot is derived later in memory at low cost.
"""
import sys
import os
import json
import pandas as pd
from memory_profile import PeakMemoryMonitor

# directories for output tables and processed data
TABLES_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "tables")
DATA_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "processed_data")

COL_NUM = [0, 1, 7]   # square_id, time_interval_ms, internet
COL_NAMES = ["square_id", "time_interval_ms", "internet"]
DTYPES = {"square_id": "int32", "internet": "float32"}  
CHUNK_SIZE = 500_000


def optimized_load(path, chunksize=CHUNK_SIZE):
    partials = []
    reader = pd.read_csv(
        path, sep="\t", header=None,
        usecols=COL_NUM, names=COL_NAMES,
        dtype=DTYPES,
        chunksize=chunksize,
    )
    for chunk in reader:
        # Missing internet_traffic means "no internet activity recorded
        # in this 10-min window for this square/country", per dataset
        # treat as 0 before aggregating.
        chunk["internet"] = chunk["internet"].fillna(0).astype("float32")
        # Collapse country_code dimension: sum traffic per (square, time).
        agg = chunk.groupby(["square_id", "time_interval_ms"], as_index=False)["internet"].sum()
        partials.append(agg)

    # Sums are associative: re-summing the partial per-chunk aggregates
    # gives the same result as aggregating the whole file at once
    combined = pd.concat(partials, ignore_index=True)
    final = combined.groupby(["square_id", "time_interval_ms"], as_index=False)["internet"].sum()

    # Downcast further after aggregation (groupby can upcast to float64/int64).
    final["square_id"] = final["square_id"].astype("uint16")
    final["internet"] = final["internet"].astype("float32")
    return final


def main(path):
    with PeakMemoryMonitor() as mon:
        df = optimized_load(path)
        _ = df["internet"].sum()

    os.makedirs(DATA_DIR, exist_ok=True)
    out_path = os.path.join(DATA_DIR, "2013-11-01_long_sample.parquet")
    df.to_parquet(out_path, index=False, engine="pyarrow", compression="snappy")

    result = {
        "stage": "optimized_chunked_load",
        "rows_out": int(len(df)),
        "columns": list(df.columns),
        "dtypes": {c: str(t) for c, t in df.dtypes.items()},
        "pandas_reported_memory_mb": float(df.memory_usage(deep=True).sum() / 1e6),
        "baseline_rss_mb": mon.baseline_mb,
        "peak_rss_mb": mon.peak_mb,
        "delta_rss_mb": mon.delta_mb,
        "parquet_file_size_mb": os.path.getsize(out_path) / 1e6,
    }
    mon.report("OPTIMIZED chunked load")
    print(json.dumps(result, indent=2))

    os.makedirs(TABLES_DIR, exist_ok=True)
    out_json = os.path.join(TABLES_DIR, "optimized_baseline_result.json")
    with open(out_json, "w") as f:
        json.dump(result, f, indent=2)
    print(f"\nSaved to: {os.path.abspath(out_json)}")


if __name__ == "__main__":
    main(sys.argv[1])