"""
src/simulation/policing_sim.py
==============================
Predictive policing simulation engine.

Implements the Noisy-OR detection and citizen reporting model from the paper:
  - Officers are deployed to GAN-generated patrol locations.
  - Crimes within the detection radius are detected with Noisy-OR probability.
  - Undetected crimes may be independently reported by citizens.
  - The simulation iterates month-by-month (February through December),
    training the GAN on the previous month's crime data each step.
"""

import numpy as np
import pandas as pd
from typing import Optional
from scipy.spatial.distance import cdist

from src.models.gan import train_gan, generate_patrol_locations

# ── Physical Constants ────────────────────────────────────────────────────────
FEET_TO_KM  = 0.0003048
KM_TO_DEG   = 1.0 / 111.0          # approximate degrees per km


def feet_to_degrees(radius_ft: float) -> float:
    """Convert a radius in feet to approximate decimal degrees."""
    return radius_ft * FEET_TO_KM * KM_TO_DEG


# ── Core Detection Model ──────────────────────────────────────────────────────

def detect_crimes(
    crime_coords: np.ndarray,
    patrol_coords: np.ndarray,
    radius_deg: float,
    p_base: float = 0.85,
    noisy_or_clip: int = 10,
) -> np.ndarray:
    """
    Noisy-OR crime detection model.

    For each crime, counts how many officers are within `radius_deg`.
    Detection probability follows: P(det) = 1 - (1 - p_base)^n_nearby.

    Parameters
    ----------
    crime_coords   : (N, 2) array of (lat, lon)
    patrol_coords  : (M, 2) array of officer positions
    radius_deg     : detection radius in decimal degrees
    p_base         : base detection probability per officer
    noisy_or_clip  : maximum officer count used in the Noisy-OR formula

    Returns
    -------
    detected : (N,) boolean array
    """
    if len(patrol_coords) == 0 or len(crime_coords) == 0:
        return np.zeros(len(crime_coords), dtype=bool)

    dists    = cdist(crime_coords, patrol_coords, metric="euclidean")
    n_nearby = (dists <= radius_deg).sum(axis=1)
    p_detect = 1.0 - (1.0 - p_base) ** np.clip(n_nearby, 0, noisy_or_clip)
    return np.random.random(len(crime_coords)) < p_detect


def apply_citizen_reporting(
    n_undetected: int,
    reporting_prob: float = 0.521,
) -> np.ndarray:
    """
    Independent citizen crime reporting (Pew Research: 52.1% reporting rate).

    Parameters
    ----------
    n_undetected   : number of crimes not detected by patrol
    reporting_prob : probability each undetected crime is reported

    Returns
    -------
    reported_flags : (n_undetected,) boolean array
    """
    return np.random.random(n_undetected) < reporting_prob


# ── Main Simulation Loop ──────────────────────────────────────────────────────

def run_simulation(
    df: pd.DataFrame,
    city: str,
    year: int,
    mode: str = "detected",
    n_officers: int = 60,
    radius_ft: float = 700.0,
    reporting_prob: float = 0.521,
    gan_epochs: int = 200,
    gan_batch_size: int = 64,
    gan_lr: float = 0.0002,
    verbose: bool = True,
) -> pd.DataFrame:
    """
    Run the 11-month predictive policing simulation (February through December).

    The GAN is trained each month on crimes known from the previous month
    (either those detected by patrol, or those reported by citizens, depending
    on `mode`). The trained Generator proposes officer deployment locations for
    the current month.

    Parameters
    ----------
    df             : processed crime DataFrame for a single city/year
    city           : city label for output rows
    year           : year label for output rows
    mode           : "detected" — train GAN on patrol-detected crimes;
                     "reported" — train GAN on citizen-reported crimes
    n_officers     : number of officers deployed each month
    radius_ft      : patrol detection radius in feet
    reporting_prob : citizen reporting probability (Noisy-OR)
    gan_epochs     : GAN training epochs per month
    gan_batch_size : GAN mini-batch size
    gan_lr         : GAN Adam learning rate
    verbose        : print per-month progress

    Returns
    -------
    results : DataFrame with monthly detection statistics per racial group
    """
    radius_deg = feet_to_degrees(radius_ft)
    races      = ["Black", "White", "Neither"]
    results    = []

    if verbose:
        print(f"\n{'─'*55}")
        print(f"  {city} {year} | mode={mode} | officers={n_officers} "
              f"| radius={radius_ft:.0f}ft")
        print(f"{'─'*55}")

    # January seed — all crimes are "known" to initialise the GAN
    df_jan = df[(df["Year"] == year) & (df["Month"] == 1)].copy()
    if df_jan.empty:
        print(f"  WARNING: No January data for {city} {year}. Skipping.")
        return pd.DataFrame()

    df_jan["Detected"] = True
    df_jan["Reported"] = True
    prev = df_jan.copy()

    # February through December
    for month in range(2, 13):
        df_month = df[(df["Year"] == year) & (df["Month"] == month)].copy()
        if df_month.empty:
            if verbose:
                print(f"  Month {month:2d}: no data — skipped")
            continue

        # Select training data for this month's GAN
        train_col = "Detected" if mode == "detected" else "Reported"
        train_df  = prev[prev[train_col] == True]
        if len(train_df) < 10:
            train_df = prev       # fallback: use all previous month's data

        if verbose:
            print(f"  Month {month:2d}: {len(df_month):6,} crimes | "
                  f"train={len(train_df):5,}", end="", flush=True)

        # Train GAN → generate patrol locations
        G, mean, std = train_gan(
            train_df[["Latitude", "Longitude"]].values,
            epochs=gan_epochs,
            batch_size=min(gan_batch_size, len(train_df)),
            lr=gan_lr,
        )
        if G is None:
            if verbose: print(" | GAN failed — skipped")
            continue

        patrol_locs  = generate_patrol_locations(G, mean, std, n_officers=n_officers)
        crime_coords = df_month[["Latitude", "Longitude"]].values

        # Detect crimes
        detected_mask = detect_crimes(crime_coords, patrol_locs, radius_deg)

        # Citizen reporting for undetected crimes
        reported_mask = detected_mask.copy()
        ud_idx        = np.where(~detected_mask)[0]
        rep_flags     = apply_citizen_reporting(len(ud_idx), reporting_prob)
        reported_mask[ud_idx[rep_flags]] = True

        df_month = df_month.copy()
        df_month["Detected"] = detected_mask
        df_month["Reported"] = reported_mask

        # Aggregate by racial group
        for race in races:
            sub = df_month[df_month["Race_Category"] == race]
            n_total    = len(sub)
            n_detected = int(sub["Detected"].sum())
            n_reported = int(sub["Reported"].sum())
            results.append({
                "City":           city,
                "Year":           year,
                "Month":          month,
                "Mode":           mode,
                "Race_Category":  race,
                "N_Total":        n_total,
                "N_Detected":     n_detected,
                "N_Reported":     n_reported,
                "Detection_Rate": round(n_detected / max(n_total, 1), 6),
                "Reporting_Rate": round(n_reported / max(n_total, 1), 6),
                "N_Officers":     n_officers,
                "Radius_ft":      radius_ft,
                "Reporting_Prob": reporting_prob,
            })

        if verbose:
            pb = df_month[df_month["Race_Category"]=="Black"]["Detected"].mean()
            pw = df_month[df_month["Race_Category"]=="White"]["Detected"].mean()
            print(f" | det={detected_mask.mean():.1%} "
                  f"| B={pb:.1%}  W={pw:.1%}")

        prev = df_month.copy()

    return pd.DataFrame(results)
