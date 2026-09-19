"""
 spatial_analysis.py
 __________________
    Section 3 (Spatial Analysis) for the MI2 traffic forecasting project.
    Generates Figure 6 (spatial EDA): a choropleth map of total Internet traffic
    across Milan’s grid. Uses milano-grid.geojson to overlay polygon boundaries,
    highlights the top-3 busiest areas, and annotates Milan’s Duomo for reference.

    Requires: telecom_italia/data_milano_grid/milano-grid.geojson
    (Downloadable from Harvard Dataverse; separate from activity data)

Usage:
   python scripts/spatial_analysis.py processed_data/full_wide_matrix.parquet \
          telecom_italia/data_milano_grid/milano-grid.geojson
"""

import sys
import os
import json
import numpy as np
import pandas as pd
import geopandas as gpd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
from math import radians, sin, cos, sqrt, atan2

PROJECT_ROOT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..")
FIG_DIR = os.path.join(PROJECT_ROOT, "figures")
TABLES_DIR = os.path.join(PROJECT_ROOT, "tables")

# Milan's Duomo - used as a fixed, well-known reference point for the
# city centre, to test whether high-traffic areas cluster around it.
DUOMO_LON, DUOMO_LAT = 9.1900, 45.4642


def haversine_m(lon1, lat1, lon2, lat2):
    """Great-circle distance in metres between two (lon, lat) points."""
    R = 6371000
    dlat = radians(lat2 - lat1)
    dlon = radians(lon2 - lon1)
    a = sin(dlat / 2) ** 2 + cos(radians(lat1)) * cos(radians(lat2)) * sin(dlon / 2) ** 2
    return 2 * R * atan2(sqrt(a), sqrt(1 - a))


def load_total_traffic(matrix_path):
    df = pd.read_parquet(matrix_path)
    total = df.sum(axis=0)
    total.index = total.index.astype(int)
    return total


def build_map(matrix_path, geojson_path):
    os.makedirs(FIG_DIR, exist_ok=True)
    os.makedirs(TABLES_DIR, exist_ok=True)

    total_traffic = load_total_traffic(matrix_path)
    top3_ids = total_traffic.sort_values(ascending=False).head(3).index.tolist()
    print(f"Top-3 areas: {top3_ids}")

    gdf = gpd.read_file(geojson_path)
    if "cellId" not in gdf.columns:
        raise ValueError(
            f"Expected a 'cellId' column in {geojson_path}, found: {list(gdf.columns)}. "
            f"This script assumes the Harvard Dataverse Milano Grid file "
            f"(doi:10.7910/DVN/QJWLFU) format."
        )

    gdf["total_traffic"] = gdf["cellId"].map(total_traffic)
    n_matched = gdf["total_traffic"].notna().sum()
    print(f"Matched {n_matched}/{len(gdf)} grid cells to traffic data.")
    if n_matched != len(gdf):
        print("WARNING: not all grid cells matched - check that cellId "
              "in the GeoJSON aligns with square_id in the traffic data.")

    # Compute each top-3 area's distance from the Duomo - this is the

    top3_gdf = gdf[gdf["cellId"].isin(top3_ids)].copy()
    top3_gdf["centroid"] = top3_gdf.geometry.centroid
    distances = {}
    for _, row in top3_gdf.iterrows():
        d = haversine_m(DUOMO_LON, DUOMO_LAT, row["centroid"].x, row["centroid"].y)
        distances[int(row["cellId"])] = round(d)
        print(f"  Square {int(row['cellId'])}: {d:.0f} m from the Duomo")

    with open(os.path.join(TABLES_DIR, "spatial_top3_distances.json"), "w", encoding="utf-8") as f:
        json.dump({"top3_ids": top3_ids, "distance_from_duomo_m": distances}, f, indent=2)

    #Building the figure
    fig, ax = plt.subplots(figsize=(10, 10))
    gdf.plot(
        column="total_traffic", ax=ax, cmap="inferno",
        norm=mcolors.LogNorm(vmin=gdf["total_traffic"].min() + 1, vmax=gdf["total_traffic"].max()),
        edgecolor="none", legend=True,
        legend_kwds={"label": "Total internet traffic (log scale)", "shrink": 0.7},
    )
    ax.scatter([DUOMO_LON], [DUOMO_LAT], marker="*", s=350, color="cyan",
               edgecolor="black", linewidth=0.8, zorder=5, label="Duomo di Milano (city centre)")

    offsets = [(-70, -35), (70, -10), (-70, 45)]
    for (cid, (x, y)), offset in zip(
        {int(r["cellId"]): (r["centroid"].x, r["centroid"].y) for _, r in top3_gdf.iterrows()}.items(),
        offsets,
    ):
        ax.scatter([x], [y], marker="o", s=80, facecolor="none", edgecolor="lime", linewidth=2, zorder=6)
        ax.annotate(f"Square {cid}", (x, y), xytext=offset, textcoords="offset points",
                     fontsize=10, color="white",
                     bbox=dict(boxstyle="round,pad=0.3", fc="black", alpha=0.75),
                     arrowprops=dict(arrowstyle="->", color="lime", lw=1.3))

    # Inset zoom on the tight top-3 cluster
    axins = ax.inset_axes([0.03, 0.03, 0.32, 0.32])
    gdf.plot(column="total_traffic", ax=axins, cmap="inferno",
              norm=mcolors.LogNorm(vmin=gdf["total_traffic"].min() + 1, vmax=gdf["total_traffic"].max()),
              edgecolor="grey", linewidth=0.1)
    axins.scatter([DUOMO_LON], [DUOMO_LAT], marker="*", s=250, color="cyan", edgecolor="black", linewidth=0.8, zorder=5)
    for cid, (x, y) in {int(r["cellId"]): (r["centroid"].x, r["centroid"].y) for _, r in top3_gdf.iterrows()}.items():
        axins.scatter([x], [y], marker="o", s=100, facecolor="none", edgecolor="lime", linewidth=2.5, zorder=6)
    pad = 0.01
    axins.set_xlim(min(x for x, y in [(r["centroid"].x, r["centroid"].y) for _, r in top3_gdf.iterrows()]) - pad,
                     max(x for x, y in [(r["centroid"].x, r["centroid"].y) for _, r in top3_gdf.iterrows()]) + pad)
    axins.set_ylim(min(y for x, y in [(r["centroid"].x, r["centroid"].y) for _, r in top3_gdf.iterrows()]) - pad,
                     max(y for x, y in [(r["centroid"].x, r["centroid"].y) for _, r in top3_gdf.iterrows()]) + pad)
    axins.set_xticks([]); axins.set_yticks([])
    axins.set_title("Zoom: city centre", fontsize=9, color="white", backgroundcolor="black")
    for spine in axins.spines.values():
        spine.set_edgecolor("lime"); spine.set_linewidth(1.5)

    ax.set_title("Figure 6: Total Internet traffic across the Milan grid (Nov-Dec 2013)", fontsize=13)
    ax.set_xlabel("Longitude"); ax.set_ylabel("Latitude")
    ax.legend(loc="upper right", fontsize=9)
    ax.set_aspect("equal")
    plt.tight_layout()

    out_path = os.path.join(FIG_DIR, "fig6_spatial_traffic_map.png")
    plt.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"\nSaved: {out_path}")


if __name__ == "__main__":
    matrix_path, geojson_path = sys.argv[1], sys.argv[2]
    build_map(matrix_path, geojson_path)