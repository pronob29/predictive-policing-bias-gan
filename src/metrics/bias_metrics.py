"""
src/metrics/bias_metrics.py
===========================
Fairness and bias quantification metrics for predictive policing analysis.

Metrics implemented:
  - Disparate Impact Ratio (DIR)     — ratio of detection rates across groups
  - Demographic Parity Gap           — absolute difference in detection rates
  - Gini Coefficient                 — inequality across all racial groups
  - Bias Amplification Score         — composite (DIR - 1) × parity_gap
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Optional


# ── Individual Metric Functions ───────────────────────────────────────────────

def disparate_impact_ratio(
    rate_privileged: float,
    rate_disadvantaged: float,
    epsilon: float = 1e-6,
) -> float:
    """
    Disparate Impact Ratio (DIR).

    DIR = detection_rate(disadvantaged) / detection_rate(privileged)

    DIR > 1  →  disadvantaged group over-policed relative to privileged group.
    DIR = 1  →  demographic parity (no disparity).
    DIR < 1  →  disadvantaged group under-policed.

    Parameters
    ----------
    rate_privileged    : detection rate of the reference (privileged) group
    rate_disadvantaged : detection rate of the group of interest
    epsilon            : small value to avoid division by zero

    Returns
    -------
    float
    """
    return rate_disadvantaged / max(rate_privileged, epsilon)


def demographic_parity_gap(
    rate_a: float,
    rate_b: float,
) -> float:
    """
    Demographic Parity Gap.

    Gap = rate_A − rate_B  (percentage points)

    Positive values indicate group A is detected more frequently than group B.
    """
    return rate_a - rate_b


def gini_coefficient(rates: List[float]) -> float:
    """
    Gini Coefficient of detection rate inequality across racial groups.

    0  →  perfect equality (all groups detected at the same rate).
    1  →  maximum inequality (all detections concentrated in one group).

    Parameters
    ----------
    rates : list of detection rates (one per racial group)

    Returns
    -------
    float in [0, 1]
    """
    arr = np.sort(np.array(rates, dtype=float))
    n   = len(arr)
    if n == 0 or arr.sum() == 0:
        return 0.0
    index = np.arange(1, n + 1)
    return float((2 * index - n - 1) @ arr / (n * arr.sum()))


def bias_amplification_score(dir_value: float, parity_gap: float) -> float:
    """
    Composite bias amplification score.

    Score = (DIR − 1) × |parity_gap|

    Captures both the direction and magnitude of bias in a single scalar.
    """
    return (dir_value - 1.0) * abs(parity_gap)


# ── DataFrame-Level Metric Computation ────────────────────────────────────────

def compute_monthly_metrics(sim_results: pd.DataFrame) -> pd.DataFrame:
    """
    Compute per-month bias metrics from simulation results.

    Parameters
    ----------
    sim_results : DataFrame output from run_simulation()
                  Columns: City, Year, Month, Mode, Race_Category,
                           N_Total, N_Detected, Detection_Rate, …

    Returns
    -------
    monthly_metrics : DataFrame indexed by (City, Year, Month, Mode)
    """
    records = []

    for (city, year, month, mode), grp in sim_results.groupby(
            ["City", "Year", "Month", "Mode"]):

        row: Dict = {"City": city, "Year": year, "Month": month, "Mode": mode}

        # Extract per-race detection rates
        race_rates: Dict[str, float] = {}
        for race in ["Black", "White", "Neither"]:
            sub = grp[grp["Race_Category"] == race]
            if len(sub):
                nt = sub["N_Total"].values[0]
                nd = sub["N_Detected"].values[0]
                nr = sub["N_Reported"].values[0]
                rate = nd / max(nt, 1)
                row[f"N_Total_{race}"]    = nt
                row[f"N_Detected_{race}"] = nd
                row[f"N_Reported_{race}"] = nr
                row[f"Det_Rate_{race}"]   = round(rate, 6)
                race_rates[race] = rate
            else:
                row[f"N_Total_{race}"]    = 0
                row[f"N_Detected_{race}"] = 0
                row[f"N_Reported_{race}"] = 0
                row[f"Det_Rate_{race}"]   = 0.0
                race_rates[race] = 0.0

        pb = race_rates.get("Black", 0.0)
        pw = race_rates.get("White", 0.0)
        pn = race_rates.get("Neither", 0.0)

        row["DIR"]              = round(disparate_impact_ratio(pw, pb), 6)
        row["Parity_Gap"]       = round(demographic_parity_gap(pb, pw), 6)
        row["Gini"]             = round(gini_coefficient([pb, pw, pn]), 6)
        row["Bias_Score"]       = round(bias_amplification_score(row["DIR"], row["Parity_Gap"]), 8)

        records.append(row)

    return pd.DataFrame(records)


def compute_annual_metrics(monthly_metrics: pd.DataFrame) -> pd.DataFrame:
    """
    Aggregate monthly bias metrics to annual summaries.

    Parameters
    ----------
    monthly_metrics : output of compute_monthly_metrics()

    Returns
    -------
    annual_metrics : DataFrame indexed by (City, Year, Mode)
    """
    records = []

    for (city, year, mode), grp in monthly_metrics.groupby(["City", "Year", "Mode"]):
        records.append({
            "City":                 city,
            "Year":                 year,
            "Mode":                 mode,
            "Avg_DIR":              round(grp["DIR"].mean(), 4),
            "Std_DIR":              round(grp["DIR"].std(), 4),
            "Max_DIR":              round(grp["DIR"].max(), 4),
            "Avg_Parity_Gap":       round(grp["Parity_Gap"].mean(), 4),
            "Avg_Gini":             round(grp["Gini"].mean(), 4),
            "Avg_Bias_Score":       round(grp["Bias_Score"].mean(), 8),
            "Months_DIR_Above_1":   int((grp["DIR"] > 1).sum()),
            "Avg_Det_Rate_Black":   round(grp["Det_Rate_Black"].mean(), 4),
            "Avg_Det_Rate_White":   round(grp["Det_Rate_White"].mean(), 4),
            "Avg_Det_Rate_Neither": round(grp["Det_Rate_Neither"].mean(), 4),
            "N_Months":             len(grp),
        })

    return pd.DataFrame(records)
