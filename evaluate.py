"""
scripts/03_evaluate.py
======================
Entry point: compute all bias metrics, sensitivity analysis, CTGAN debiasing,
and socioeconomic regression from simulation outputs.

Usage
-----
    python scripts/03_evaluate.py

Outputs
-------
    results/metrics/monthly_bias_metrics.csv
    results/metrics/annual_bias_metrics.csv
    results/sensitivity/sensitivity_results.csv
    results/debiasing/debiasing_comparison.csv
    results/socioeconomic/correlation_results.csv
    results/socioeconomic/neighborhood_stats.csv
    results/socioeconomic/regression_results.csv
"""

import os
import sys
import yaml
import numpy as np
import pandas as pd
from scipy import stats

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.metrics.bias_metrics import compute_monthly_metrics, compute_annual_metrics
from src.simulation.policing_sim import run_simulation, feet_to_degrees
from src.models.gan import train_gan, generate_patrol_locations
from src.simulation.policing_sim import detect_crimes

# ── Config ────────────────────────────────────────────────────────────────────
with open("configs/config.yaml") as f:
    cfg = yaml.safe_load(f)

PROC    = cfg["paths"]["data_processed"]
RES     = cfg["paths"]["results"]
SIM_CFG = cfg["simulation"]
TR_CFG  = cfg["training"]
SENS    = cfg["sensitivity"]
DEB     = cfg["debiasing"]

for sub in ["metrics", "sensitivity", "debiasing", "socioeconomic"]:
    os.makedirs(os.path.join(RES, sub), exist_ok=True)

import torch; import numpy as np
torch.manual_seed(TR_CFG["seed"]); np.random.seed(TR_CFG["seed"])

# ── Load simulation results ───────────────────────────────────────────────────
print("\n" + "="*55)
print("STEP 1: Bias Metrics")
print("="*55)

sim = pd.read_csv(os.path.join(RES, "simulation_results.csv"))
monthly = compute_monthly_metrics(sim)
annual  = compute_annual_metrics(monthly)

monthly.to_csv(os.path.join(RES, "metrics", "monthly_bias_metrics.csv"), index=False)
annual.to_csv( os.path.join(RES, "metrics", "annual_bias_metrics.csv"),  index=False)
print(f"  Monthly metrics: {len(monthly)} rows")
print(f"  Annual metrics:  {len(annual)} rows")
print(annual[["City","Year","Mode","Avg_DIR","Std_DIR","Months_DIR_Above_1"]].to_string(index=False))

# ── Sensitivity Analysis ──────────────────────────────────────────────────────
print("\n" + "="*55)
print("STEP 2: Sensitivity Analysis")
print("="*55)

df_2019 = pd.read_csv(os.path.join(PROC, "baltimore_2019.csv"))
df_2019["Year"] = 2019
sens_records = []

def _dir_from_sim(res: pd.DataFrame) -> float:
    if res.empty: return np.nan
    rates = {
        race: (grp["N_Detected"].sum() / max(grp["N_Total"].sum(), 1))
        for race, grp in res.groupby("Race_Category")
    }
    return rates.get("Black", 0) / max(rates.get("White", 0.001), 0.001)

print("  Varying detection radius...")
for r_ft in SENS["radii_ft"]:
    res = run_simulation(df_2019, "Baltimore", 2019, mode="detected",
                         n_officers=SIM_CFG["n_officers"], radius_ft=r_ft,
                         reporting_prob=SIM_CFG["reporting_probability"],
                         gan_epochs=TR_CFG["epochs_sensitivity"], verbose=False)
    dir_val = _dir_from_sim(res)
    sens_records.append({"Parameter": "Radius_ft", "Value": r_ft, "DIR_Black_White": dir_val})
    print(f"    radius={r_ft}ft → DIR={dir_val:.3f}")

print("  Varying officer count...")
for n_off in SENS["officer_counts"]:
    res = run_simulation(df_2019, "Baltimore", 2019, mode="detected",
                         n_officers=n_off,
                         radius_ft=SIM_CFG["detection_radius_ft"],
                         reporting_prob=SIM_CFG["reporting_probability"],
                         gan_epochs=TR_CFG["epochs_sensitivity"], verbose=False)
    dir_val = _dir_from_sim(res)
    sens_records.append({"Parameter": "N_Officers", "Value": n_off, "DIR_Black_White": dir_val})
    print(f"    officers={n_off} → DIR={dir_val:.3f}")

print("  Varying reporting probability...")
for rp in SENS["reporting_probs"]:
    res = run_simulation(df_2019, "Baltimore", 2019, mode="reported",
                         n_officers=SIM_CFG["n_officers"],
                         radius_ft=SIM_CFG["detection_radius_ft"],
                         reporting_prob=rp,
                         gan_epochs=TR_CFG["epochs_sensitivity"], verbose=False)
    dir_val = _dir_from_sim(res)
    sens_records.append({"Parameter": "Reporting_Prob", "Value": rp, "DIR_Black_White": dir_val})
    print(f"    reporting_prob={rp:.2f} → DIR={dir_val:.3f}")

sens_df = pd.DataFrame(sens_records)
sens_df.to_csv(os.path.join(RES, "sensitivity", "sensitivity_results.csv"), index=False)
print(f"  ✓ Sensitivity results: {len(sens_df)} configurations")

# ── CTGAN Debiasing ───────────────────────────────────────────────────────────
print("\n" + "="*55)
print("STEP 3: CTGAN Debiasing")
print("="*55)

try:
    from ctgan import CTGAN
    has_ctgan = True
except ImportError:
    os.system("pip install ctgan --break-system-packages -q")
    try:
        from ctgan import CTGAN
        has_ctgan = True
    except ImportError:
        has_ctgan = False
        print("  WARNING: ctgan unavailable — skipping debiasing step")

if has_ctgan:
    df_jan = df_2019[df_2019["Month"] == 1].copy()
    spc    = DEB["samples_per_class"]

    balanced = pd.concat([
        sub.sample(min(len(sub), spc), random_state=42)
        for _, sub in df_jan.groupby("Race_Category")
    ], ignore_index=True)

    ctgan = CTGAN(epochs=DEB["ctgan_epochs"], verbose=False)
    ctgan.fit(balanced[["Latitude", "Longitude", "Race_Category"]], ["Race_Category"])
    synth = ctgan.sample(DEB["synthetic_sample_size"])
    print(f"  CTGAN synthetic samples: {len(synth)}")

    df_feb = df_2019[df_2019["Month"] == 2].copy()
    crime_coords = df_feb[["Latitude", "Longitude"]].values
    r_deg = feet_to_degrees(SIM_CFG["detection_radius_ft"])

    G_b, m_b, s_b = train_gan(df_jan[["Latitude","Longitude"]].values, epochs=200)
    G_d, m_d, s_d = train_gan(synth[["Latitude","Longitude"]].values, epochs=200)

    pl_b  = generate_patrol_locations(G_b, m_b, s_b, n_officers=SIM_CFG["n_officers"])
    pl_d  = generate_patrol_locations(G_d, m_d, s_d, n_officers=SIM_CFG["n_officers"])

    det_b = detect_crimes(crime_coords, pl_b, r_deg)
    det_d = detect_crimes(crime_coords, pl_d, r_deg)

    def _rates(det, df):
        return {
            race: det[df["Race_Category"] == race].mean()
            for race in ["Black", "White", "Neither"]
        }

    rb, rd = _rates(det_b, df_feb), _rates(det_d, df_feb)

    debias_df = pd.DataFrame([
        {"Condition":    "Biased (raw training)",
         "DIR":          round(rb["Black"] / max(rb["White"], 1e-6), 4),
         "Pct_Det_Black": round(rb["Black"], 4),
         "Pct_Det_White": round(rb["White"], 4),
         "Parity_Gap":   round(rb["Black"] - rb["White"], 4)},
        {"Condition":    "Debiased (CTGAN balanced)",
         "DIR":          round(rd["Black"] / max(rd["White"], 1e-6), 4),
         "Pct_Det_Black": round(rd["Black"], 4),
         "Pct_Det_White": round(rd["White"], 4),
         "Parity_Gap":   round(rd["Black"] - rd["White"], 4)},
    ])
    debias_df.to_csv(os.path.join(RES, "debiasing", "debiasing_comparison.csv"), index=False)
    print(debias_df.to_string(index=False))

# ── Socioeconomic Analysis ────────────────────────────────────────────────────
print("\n" + "="*55)
print("STEP 4: Socioeconomic Analysis")
print("="*55)

combined = pd.read_csv(os.path.join(PROC, "baltimore_combined.csv"))
nbhd = combined.groupby("Neighborhood").agg(
    Crime_Rate=("Latitude", "count"),
    Pct_Black=("Pct_Black", "mean"),
    Pct_White=("Pct_White", "mean"),
    Median_Income=("Median_Income", "mean"),
    Poverty_Rate=("Poverty_Rate", "mean"),
    Race_Category=("Race_Category", lambda x: x.mode()[0]),
).reset_index()

# Attach average detection rate per racial group from simulation
balt_det = annual[(annual["City"]=="Baltimore") & (annual["Mode"]=="detected")]
race_det_map = {
    "Black":   float(balt_det["Avg_Det_Rate_Black"].mean()),
    "White":   float(balt_det["Avg_Det_Rate_White"].mean()),
    "Neither": float(balt_det["Avg_Det_Rate_Neither"].mean()),
}
nbhd["Det_Rate"] = nbhd["Race_Category"].map(race_det_map).fillna(0)
nbhd.to_csv(os.path.join(RES, "socioeconomic", "neighborhood_stats.csv"), index=False)

# Correlations
corr_records = []
target = nbhd["Det_Rate"]
for var in ["Pct_Black", "Pct_White", "Median_Income", "Poverty_Rate"]:
    x    = nbhd[var]
    mask = x.notna() & target.notna()
    if mask.sum() < 5: continue
    rp, pp = stats.pearsonr(x[mask], target[mask])
    rs, ps = stats.spearmanr(x[mask], target[mask])
    corr_records.append({"Variable": var, "n": int(mask.sum()),
                          "Pearson_r": round(rp,4), "Pearson_p": round(pp,6),
                          "Spearman_r": round(rs,4), "Spearman_p": round(ps,6)})

corr_df = pd.DataFrame(corr_records)
corr_df.to_csv(os.path.join(RES, "socioeconomic", "correlation_results.csv"), index=False)
print("\n  Correlations with Detection Rate:")
print(corr_df.to_string(index=False))

# OLS regression
X_cols = ["Pct_Black", "Median_Income", "Poverty_Rate"]
sub    = nbhd[X_cols + ["Det_Rate"]].dropna()
if len(sub) >= 5:
    X      = np.column_stack([np.ones(len(sub))] + [sub[c].values for c in X_cols])
    y      = sub["Det_Rate"].values
    coeffs, _, _, _ = np.linalg.lstsq(X, y, rcond=None)
    y_pred = X @ coeffs
    r2     = 1 - ((y - y_pred)**2).sum() / max(((y - y.mean())**2).sum(), 1e-10)
    reg_df = pd.DataFrame({"Variable": ["Intercept"] + X_cols,
                           "Coefficient": [round(c, 8) for c in coeffs]})
    reg_df.to_csv(os.path.join(RES, "socioeconomic", "regression_results.csv"), index=False)
    print(f"\n  OLS R² = {r2:.4f}")
    print(reg_df.to_string(index=False))

print("\n✓ Evaluation complete.")
