"""
scripts/04_visualize.py
=======================
Entry point: generate all 9 publication-quality figures.

Usage
-----
    python scripts/04_visualize.py

Outputs (PDF + PNG at 300 DPI)
-------------------------------
    results/figures/fig1_monthly_detection_rate.pdf/.png
    results/figures/fig2_multiyear_DIR.pdf/.png
    results/figures/fig3_parity_gap.pdf/.png
    results/figures/fig4_cross_city_DIR.pdf/.png
    results/figures/fig5_sensitivity_analysis.pdf/.png
    results/figures/fig6_ctgan_debiasing.pdf/.png
    results/figures/fig7_socioeconomic_scatter.pdf/.png
    results/figures/fig8_DIR_heatmap.pdf/.png
    results/figures/fig9_gini_trend.pdf/.png
"""

import os
import sys
import yaml
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.visualization.figures import generate_all_figures

# ── Config ────────────────────────────────────────────────────────────────────
with open("configs/config.yaml") as f:
    cfg = yaml.safe_load(f)

PROC    = cfg["paths"]["data_processed"]
RES     = cfg["paths"]["results"]
FIG_DIR = os.path.join(RES, "figures")
os.makedirs(FIG_DIR, exist_ok=True)

# ── Load all result files ─────────────────────────────────────────────────────
print("Loading results...")

sim     = pd.read_csv(os.path.join(RES, "simulation_results.csv"))
monthly = pd.read_csv(os.path.join(RES, "metrics", "monthly_bias_metrics.csv"))
annual  = pd.read_csv(os.path.join(RES, "metrics", "annual_bias_metrics.csv"))
sens    = pd.read_csv(os.path.join(RES, "sensitivity", "sensitivity_results.csv"))
debias  = pd.read_csv(os.path.join(RES, "debiasing", "debiasing_comparison.csv"))
nbhd    = pd.read_csv(os.path.join(RES, "socioeconomic", "neighborhood_stats.csv"))

# ── Generate ──────────────────────────────────────────────────────────────────
print("\n" + "="*55)
print("Generating Figures")
print("="*55)

generate_all_figures(
    sim=sim,
    monthly=monthly,
    annual=annual,
    sens=sens,
    debias=debias,
    nbhd=nbhd,
    out_dir=FIG_DIR,
)

print("\n✓ All figures saved to:", FIG_DIR)
