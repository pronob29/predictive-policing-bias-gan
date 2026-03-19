"""
scripts/01_preprocess.py
========================
Entry point: preprocess raw crime data for Baltimore and Chicago.

Usage
-----
    python scripts/01_preprocess.py

Outputs
-------
    data/processed/baltimore_2017.csv
    data/processed/baltimore_2018.csv
    data/processed/baltimore_2019.csv
    data/processed/baltimore_combined.csv
    data/processed/baltimore_neighborhoods.csv
    data/processed/chicago_2022.csv
    data/processed/chicago_neighborhoods.csv
"""

import os
import sys
import yaml
import pandas as pd

# Allow imports from project root
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.data.preprocess import preprocess_baltimore, preprocess_chicago

# ── Config ────────────────────────────────────────────────────────────────────
with open("configs/config.yaml") as f:
    cfg = yaml.safe_load(f)

RAW  = cfg["paths"]["data_raw"]
PROC = cfg["paths"]["data_processed"]
os.makedirs(PROC, exist_ok=True)

# ── Baltimore ─────────────────────────────────────────────────────────────────
print("\n" + "="*55)
print("STEP 1: Baltimore Part 1 Crime Data")
print("="*55)

balt_raw = os.path.join(RAW, cfg["data"]["baltimore"]["raw_file"])
balt_data = preprocess_baltimore(
    raw_path=balt_raw,
    years=tuple(cfg["data"]["baltimore"]["years"]),
    bbox=cfg["data"]["baltimore"]["bbox"],
)

for yr, df in balt_data.items():
    if yr == "combined":
        out = os.path.join(PROC, "baltimore_combined.csv")
    else:
        out = os.path.join(PROC, f"baltimore_{yr}.csv")
    df.to_csv(out, index=False)
    print(f"  Saved: {out}  ({len(df):,} rows)")

# Neighbourhood summary
nbhd_df = (balt_data["combined"]
           .groupby("Neighborhood")
           .agg(
               N_Crimes=("Latitude", "count"),
               Pct_Black=("Pct_Black", "mean"),
               Pct_White=("Pct_White", "mean"),
               Median_Income=("Median_Income", "mean"),
               Poverty_Rate=("Poverty_Rate", "mean"),
               Race_Category=("Race_Category", lambda x: x.mode()[0]),
           )
           .reset_index()
           .sort_values("N_Crimes", ascending=False))
nbhd_out = os.path.join(PROC, "baltimore_neighborhoods.csv")
nbhd_df.to_csv(nbhd_out, index=False)
print(f"  Saved: {nbhd_out}  ({len(nbhd_df)} neighbourhoods)")

# ── Chicago ───────────────────────────────────────────────────────────────────
print("\n" + "="*55)
print("STEP 2: Chicago Crime Data")
print("="*55)

chi_raw = os.path.join(RAW, cfg["data"]["chicago"]["raw_file"])
chi_df  = preprocess_chicago(chi_raw, year=cfg["data"]["chicago"]["year"])

chi_out = os.path.join(PROC, "chicago_2022.csv")
chi_df.to_csv(chi_out, index=False)
print(f"  Saved: {chi_out}  ({len(chi_df):,} rows)")

chi_nbhd = (chi_df
            .groupby("Neighborhood")
            .agg(
                N_Crimes=("Latitude", "count"),
                Pct_Black=("Pct_Black", "mean"),
                Pct_White=("Pct_White", "mean"),
                Median_Income=("Median_Income", "mean"),
                Poverty_Rate=("Poverty_Rate", "mean"),
                Race_Category=("Race_Category", lambda x: x.mode()[0]),
            )
            .reset_index()
            .sort_values("N_Crimes", ascending=False))
chi_nbhd_out = os.path.join(PROC, "chicago_neighborhoods.csv")
chi_nbhd.to_csv(chi_nbhd_out, index=False)
print(f"  Saved: {chi_nbhd_out}  ({len(chi_nbhd)} neighbourhoods)")

print("\n✓ Preprocessing complete.")
