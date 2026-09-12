"""
batch_pipeline.py
_________________

Builds an optimized dataset from daily telecom raw text files 
(streamed in chunks to avoid memory overload). 

Per file:
- Reads only [square_id, time_interval_ms, internet] columns instead of all 8 columns
- Downcasts dtypes for efficiency (square_id: int32 -> uint16, internet: float32)
- Aggregates internet traffic per (square_id, time_interval_ms)
- Outputs one Parquet file per day (full dataset)

After all days are processed:
- Concatenates daily Parquets
- Reindexes to full grid (10,000 squares × N timesteps), filling missing combos with 0
- Pivots to wide matrix (timesteps × areas) for EDA & forecasting
- Saves as compressed Parquet (~335 MB in memory vs ~19 GB raw data)

Purpose: Efficiently transform large raw telecom logs into a compact, 
analysis-ready dataset for time-series forecasting.
"""

import sys
import glob
import os
import re
import pandas as pd
import numpy as np

COL_NUM = [0, 1, 7]
COL_NAMES = ["square_id", "time_interval_ms", "internet"]
DTYPES = {"square_id": "int32", "internet": "float32"}
CHUNK_SIZE = 500_000


def process_file(path, chunksize=CHUNK_SIZE):
    partials = []
    reader = pd.read_csv(
        path, sep="\t", header=None,
        usecols=COL_NUM, names=COL_NAMES,
        dtype=DTYPES, chunksize=chunksize,
    )
    for chunk in reader:
        chunk["internet"] = chunk["internet"].fillna(0).astype("float32")
        agg = chunk.groupby(["square_id", "time_interval_ms"], as_index=False)["internet"].sum()
        partials.append(agg)
    combined = pd.concat(partials, ignore_index=True)
    final = combined.groupby(["square_id", "time_interval_ms"], as_index=False)["internet"].sum()
    final["square_id"] = final["square_id"].astype("uint16")
    final["internet"] = final["internet"].astype("float32")
    return final


def discover_files(raw_root):
    """
    Recursively find every raw day-file under raw_root, regardless of
    which month-subfolder it lives in, de-duplicated and sorted by the
    date embedded in the filename (YYYY-MM-DD).
    """
    pattern = os.path.join(raw_root, "**", "sms-call-internet-mi-*.txt")
    files = glob.glob(pattern, recursive=True)
    files = sorted(set(files), key=lambda p: _extract_date(p) or p)
    return files


def _extract_date(path):
    m = re.search(r"(\d{4}-\d{2}-\d{2})", os.path.basename(path))
    return m.group(1) if m else None


def run_all(raw_root, out_dir, skip_dates=None):
    """
    skip_dates: optional list of 'YYYY-MM-DD' strings to exclude. 
    Use this to omit specific day-files (e.g., the single 2014-01 file) 
    that fall outside the intended training or test window.
    """
    os.makedirs(out_dir, exist_ok=True)
    skip_dates = set(skip_dates or [])

    files = discover_files(raw_root)
    if not files:
        raise FileNotFoundError(
            f"No files matching 'sms-call-internet-mi-*.txt' found under {raw_root}. "
            f"Check the path and that filenames follow the expected pattern."
        )

    print(f"Found {len(files)} raw day-files under {raw_root}:")
    for p in files:
        print(f"   {_extract_date(p)}  <-  {p}")

    for path in files:
        date_str = _extract_date(path) or os.path.basename(path)
        if date_str in skip_dates:
            print(f"  [excluded by config] {date_str}")
            continue
        out_path = os.path.join(out_dir, f"{date_str}_long.parquet")
        if os.path.exists(out_path):
            print(f"  [skip] {date_str} already processed")
            continue
        print(f"  [processing] {date_str} ...")
        df = process_file(path)
        df.to_parquet(out_path, index=False, engine="pyarrow", compression="snappy")
        print(f"    -> {len(df):,} rows written")
    print("Done. Now call combine_and_pivot(out_dir, ...) to build the final wide matrix.")


def combine_and_pivot(long_dir, out_wide_path):
    """
    Concatenate all daily Parquet files, reindex to the full 
    (square_id × time_ms) grid, fill missing values with 0, 
    and pivot into a wide (time × area) matrix saved as Parquet.
    """

    files = sorted(glob.glob(os.path.join(long_dir, "*_long.parquet")))
    if not files:
        raise FileNotFoundError(f"No long-format parquet files found in {long_dir}")

    dfs = [pd.read_parquet(f) for f in files]
    long_df = pd.concat(dfs, ignore_index=True)
    long_df = long_df.groupby(["square_id", "time_interval_ms"], as_index=False)["internet"].sum()

    all_squares = np.arange(1, 10001, dtype="uint16")
    all_times = np.sort(long_df["time_interval_ms"].unique())
    full_index = pd.MultiIndex.from_product(
        [all_squares, all_times], names=["square_id", "time_ms"]
    )

    long_df = long_df.set_index(["square_id", "time_ms"]).reindex(full_index, fill_value=0.0)
    long_df = long_df.reset_index()
    long_df["internet"] = long_df["internet"].astype("float32")

    wide = long_df.pivot(index="time_ms", columns="square_id", values="internet")
    wide.index = pd.to_datetime(wide.index, unit="ms")
    wide = wide.sort_index()
    wide.columns = wide.columns.astype(str) 

    wide.to_parquet(out_wide_path, engine="pyarrow", compression="snappy")
    print(f"Wide matrix shape: {wide.shape}  ->  saved to {out_wide_path}")
    print(f"File size on disk: {os.path.getsize(out_wide_path) / 1e6:.1f} MB")
    return wide


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("raw_root", help="Root folder containing month-subfolders of raw .txt files (e.g. telecom_italia/)")
    parser.add_argument("out_dir", help="Output folder for processed Parquet files (e.g. data_processed/)")
    parser.add_argument("--skip-date", action="append", default=[],
                         help="A YYYY-MM-DD date to exclude from processing (repeatable). "
                              "E.g. --skip-date 2014-01-01 if you decide the January file "
                              "falls outside your train/test window.")
    args = parser.parse_args()

    daily_dir = os.path.join(args.out_dir, "daily_long")
    run_all(args.raw_root, daily_dir, skip_dates=args.skip_date)
    combine_and_pivot(daily_dir, os.path.join(args.out_dir, "full_extended_matrix.parquet"))