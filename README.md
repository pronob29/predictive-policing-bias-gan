# Unmasking Algorithmic Bias in Predictive Policing
### A GAN-Based Simulation Framework

Replication package for the paper submitted to [Conference Name].

---

## Overview

This repository contains the full experimental pipeline for a study on racial bias in GAN-driven predictive policing systems. The framework trains a Generative Adversarial Network on historical crime data to simulate police deployment, then measures how the resulting detection patterns vary across neighbourhoods of different racial compositions.

**Datasets used:**
| Dataset | Source | Records |
|---|---|---|
| Baltimore Part 1 Crime (2017–2019) | Baltimore City Open Data | 145,178 |
| Chicago Crimes (2022) | City of Chicago Data Portal | 233,690 |
| US Census ACS 5-Year Estimates | US Census Bureau | Neighbourhood-level |

---

## Project Structure

```
.
├── configs/
│   └── config.yaml                  # All hyperparameters and paths
│
├── data/
│   ├── raw/                         # Original downloaded files (see Data below)
│   │   ├── baltimore_part1_crime.csv
│   │   └── chicago_crimes_2022.csv
│   └── processed/                   # Cleaned, feature-enriched outputs
│       ├── baltimore_2017.csv        # 51,511 crimes
│       ├── baltimore_2018.csv        # 47,856 crimes
│       ├── baltimore_2019.csv        # 45,811 crimes
│       ├── baltimore_combined.csv    # 145,178 crimes (all years)
│       ├── baltimore_neighborhoods.csv
│       ├── chicago_2022.csv          # 233,690 crimes
│       └── chicago_neighborhoods.csv
│
├── src/                             # Importable source library
│   ├── data/
│   │   └── preprocess.py            # Baltimore & Chicago preprocessing
│   ├── models/
│   │   └── gan.py                   # Generator, Discriminator, train_gan
│   ├── simulation/
│   │   └── policing_sim.py          # Noisy-OR detection, run_simulation
│   ├── metrics/
│   │   └── bias_metrics.py          # DIR, Gini, parity gap, bias score
│   └── visualization/
│       └── figures.py               # All 9 paper figures
│
├── preprocess.py                    # Step 1 — raw → processed data
├── simulate.py                      # Step 2 — GAN training + simulation
├── evaluate.py                      # Step 3 — metrics, sensitivity, debiasing, regression
├── visualize.py                     # Step 4 — generate all figures
│
├── results/
│   ├── simulation_results.csv       # 264-row monthly output table
│   ├── figures/                     # Fig 1–9 (PDF + PNG, 300 DPI)
│   ├── metrics/
│   │   ├── annual_bias_metrics.csv
│   │   └── monthly_bias_metrics.csv
│   ├── sensitivity/
│   │   └── sensitivity_results.csv
│   ├── debiasing/
│   │   └── debiasing_comparison.csv
│   └── socioeconomic/
│       ├── correlation_results.csv
│       ├── neighborhood_stats.csv
│       └── regression_results.csv
│
├── requirements.txt
└── .gitignore
```

---

## Quickstart

### 1. Install dependencies

```bash
pip install -r requirements.txt
```

### 2. Download raw data

| File | URL |
|---|---|
| Baltimore Part 1 Crime | https://data.baltimorecity.gov → save as `data/raw/baltimore_part1_crime.csv` |
| Chicago Crimes 2022 | https://data.cityofchicago.org → save as `data/raw/chicago_crimes_2022.csv` |

> **Note:** Processed data files are already included in `data/processed/` for direct replication without re-downloading.

### 3. Run the pipeline

```bash
# Step 1 — Preprocess (skip if using provided processed files)
python preprocess.py

# Step 2 — Run GAN simulations  (~15–30 min on CPU)
python simulate.py

# Step 3 — Compute metrics, sensitivity, debiasing, regression
python evaluate.py

# Step 4 — Generate all paper figures
python visualize.py
```

All hyperparameters are in `configs/config.yaml` — no hardcoded values in scripts.

---

## Model Architecture

### Generator
```
noise(100) → Linear(256) → BatchNorm → LeakyReLU(0.2)
           → Linear(512) → BatchNorm → LeakyReLU(0.2)
           → Linear(256) → BatchNorm → LeakyReLU(0.2)
           → Linear(2)   → Tanh
```

### Discriminator
```
(lat, lon) → Linear(512) → LeakyReLU(0.2) → Dropout(0.3)
           → Linear(256) → LeakyReLU(0.2) → Dropout(0.3)
           → Linear(128) → LeakyReLU(0.2)
           → Linear(1)   → Sigmoid
```

**Training:** Adam optimiser, lr=0.0002, β=(0.5, 0.999), 200 epochs per month.

### Noisy-OR Detection Model

```
P(detected | n_nearby officers) = 1 − (1 − p_base)^n_nearby
```

where `p_base = 0.85` and detection radius = 700 ft (matching paper).
Undetected crimes are independently reported with probability 52.1% (Pew Research, 2019).

---

## Key Results

### Disparate Impact Ratio  (DIR = Black detection rate / White detection rate)

| City | Year | Mode | Avg. DIR | SD | Months DIR > 1 |
|---|---|---|---|---|---|
| Baltimore | 2017 | detected | 1.02 | 0.42 | 5/11 |
| Baltimore | 2018 | detected | 0.41 | 0.77 | 2/11 |
| Baltimore | 2019 | detected | **16.72** | 12.39 | **10/11** |
| Baltimore | 2019 | reported | 0.78 | 0.31 | 4/11 |
| Chicago | 2022 | detected | **13.12** | 9.41 | **11/11** |
| Chicago | 2022 | reported | 2.52 | 1.71 | 10/11 |

### Socioeconomic Correlates of Detection Rate (n = 279 neighbourhoods)

| Variable | Pearson r | Spearman r | p-value |
|---|---|---|---|
| % Black population | −0.384 | −0.708 | < 0.001 |
| % White population | +0.328 | +0.699 | < 0.001 |
| Median income | +0.137 | +0.477 | 0.022 |
| Poverty rate | −0.214 | −0.581 | < 0.001 |

> OLS R² = 0.295 — poverty rate is the dominant predictor after controlling for race.

### Sensitivity Analysis

DIR ranges from 0.03–1.59 across 14 parameter configurations (radius 400–1500 ft, officers 30–120, reporting 30–80%). The bias direction reverses as the detection radius grows beyond the natural clustering radius of Black-majority neighbourhoods.

### CTGAN Debiasing

Race-balanced synthetic training via CTGAN substantially redistributes simulated patrol presence, reducing White-neighbourhood detection rate from 7.1% to 0.4%.

---

## Bias Metrics

| Metric | Formula | Interpretation |
|---|---|---|
| **DIR** | det_rate_Black / det_rate_White | >1 = over-policing of Black areas |
| **Parity Gap** | det_rate_Black − det_rate_White | pp difference |
| **Gini** | Gini(det_rate_Black, White, Neither) | 0 = equal, 1 = fully unequal |
| **Bias Score** | (DIR − 1) × \|Parity Gap\| | composite magnitude |

---

## Figures

| Figure | Description |
|---|---|
| Fig 1 | Monthly detection rate by racial composition — Baltimore 2019 |
| Fig 2 | Multi-year Disparate Impact Ratio with ±1 SD — Baltimore |
| Fig 3 | Monthly Demographic Parity Gap, multi-year overlay |
| Fig 4 | Cross-city DIR comparison — Baltimore vs Chicago |
| Fig 5 | Sensitivity analysis — 3-panel (radius / officers / reporting) |
| Fig 6 | CTGAN debiasing — biased vs debiased detection rates and DIR |
| Fig 7 | Socioeconomic scatter matrix — 5 predictors vs detection rate |
| Fig 8 | DIR heatmap — month × year (detected and reported modes) |
| Fig 9 | Gini coefficient trend across all simulation steps |

---

## Citation

If you use this code or data, please cite:

```bibtex
@inproceedings{author2025gan,
  title     = {Unmasking Algorithmic Bias in Predictive Policing:
               A GAN-Based Simulation Framework},
  author    = {Author(s)},
  booktitle = {Proceedings of [Conference]},
  year      = {2025}
}
```

---

## License

Code is released under the **MIT License**.
Data files are subject to the original open-data licences of Baltimore City and the City of Chicago.
