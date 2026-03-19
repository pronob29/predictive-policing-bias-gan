"""
scripts/02_simulate.py
======================
Entry point: run GAN-based predictive policing simulations.

Runs the full 11-month simulation (Feb–Dec) for:
  - Baltimore: 2017, 2018, 2019
  - Chicago: 2022
  Each city/year is simulated in both "detected" and "reported" modes.

Usage
-----
    python scripts/02_simulate.py

Output
------
    results/simulation_results.csv   (264 rows)
"""

import os
import sys
import yaml
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.simulation.policing_sim import run_simulation, feet_to_degrees

# ── Config ────────────────────────────────────────────────────────────────────
with open("configs/config.yaml") as f:
    cfg = yaml.safe_load(f)

PROC    = cfg["paths"]["data_processed"]
RES     = cfg["paths"]["results"]
SIM_CFG = cfg["simulation"]
TR_CFG  = cfg["training"]
os.makedirs(RES, exist_ok=True)

import torch; import numpy as np
torch.manual_seed(TR_CFG["seed"]); np.random.seed(TR_CFG["seed"])

# ── Load Processed Data ───────────────────────────────────────────────────────
print("\n" + "="*55)
print("Loading processed data...")
print("="*55)

balt = {}
for yr in cfg["data"]["baltimore"]["years"]:
    path = os.path.join(PROC, f"baltimore_{yr}.csv")
    balt[yr] = pd.read_csv(path)
    print(f"  Baltimore {yr}: {len(balt[yr]):,} crimes")

chi = pd.read_csv(os.path.join(PROC, "chicago_2022.csv"))
chi["Year"] = 2022
print(f"  Chicago 2022:  {len(chi):,} crimes")

# ── Run Simulations ───────────────────────────────────────────────────────────
all_results = []

print("\n" + "="*55)
print("Running GAN Simulations")
print("="*55)

for yr in cfg["data"]["baltimore"]["years"]:
    df = balt[yr]
    for mode in SIM_CFG["modes"]:
        res = run_simulation(
            df=df, city="Baltimore", year=yr, mode=mode,
            n_officers=SIM_CFG["n_officers"],
            radius_ft=SIM_CFG["detection_radius_ft"],
            reporting_prob=SIM_CFG["reporting_probability"],
            gan_epochs=TR_CFG["epochs"],
            gan_batch_size=TR_CFG["batch_size"],
            gan_lr=TR_CFG["learning_rate"],
        )
        all_results.append(res)

for mode in SIM_CFG["modes"]:
    res = run_simulation(
        df=chi, city="Chicago", year=2022, mode=mode,
        n_officers=SIM_CFG["n_officers"],
        radius_ft=SIM_CFG["detection_radius_ft"],
        reporting_prob=SIM_CFG["reporting_probability"],
        gan_epochs=TR_CFG["epochs_chicago"],
        gan_batch_size=TR_CFG["batch_size"],
        gan_lr=TR_CFG["learning_rate"],
    )
    all_results.append(res)

# ── Save ──────────────────────────────────────────────────────────────────────
sim_df = pd.concat([r for r in all_results if not r.empty], ignore_index=True)
out    = os.path.join(RES, "simulation_results.csv")
sim_df.to_csv(out, index=False)
print(f"\n✓ Simulation complete — {len(sim_df)} rows → {out}")
