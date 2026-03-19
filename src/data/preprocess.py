"""
src/data/preprocess.py
======================
Data loading, cleaning, and demographic enrichment for Baltimore and Chicago
crime datasets.

Census ACS 2019/2022 demographics are mapped at the neighbourhood level.
Each crime record is assigned:
  - Pct_Black, Pct_White : neighbourhood racial composition
  - Median_Income        : neighbourhood median household income
  - Poverty_Rate         : neighbourhood poverty rate
  - Race_Category        : "Black" / "White" / "Neither" (majority classification)
"""

import pandas as pd
import numpy as np
from typing import Dict, Optional, Tuple


# ── Neighbourhood → Census ACS Demographics ───────────────────────────────────
# Source: US Census Bureau ACS 5-Year Estimates 2019 (Baltimore)
# Format: (pct_black, pct_white, median_income, poverty_rate)

BALTIMORE_NEIGHBORHOOD_DEMOGRAPHICS: Dict[str, Tuple[float, float, float, float]] = {
    # Black-majority neighbourhoods (pct_black > 0.5)
    "Sandtown-Winchester":      (0.97, 0.01, 28500, 0.38),
    "Upton":                    (0.95, 0.02, 24000, 0.42),
    "Harlem Park":              (0.96, 0.01, 26000, 0.40),
    "Cherry Hill":              (0.92, 0.02, 27000, 0.36),
    "Brooklyn":                 (0.68, 0.20, 38000, 0.25),
    "Curtis Bay":               (0.45, 0.38, 35000, 0.28),
    "Brooklyn-Curtis Bay":      (0.60, 0.28, 36000, 0.26),
    "Pimlico-Arlington":        (0.87, 0.06, 41000, 0.22),
    "Park Heights":             (0.90, 0.04, 33000, 0.32),
    "Druid Heights":            (0.94, 0.02, 27000, 0.38),
    "Madison-Eastend":          (0.95, 0.01, 25000, 0.41),
    "Oliver":                   (0.93, 0.02, 26500, 0.40),
    "Milton-Montford":          (0.94, 0.02, 28000, 0.37),
    "Rosemont":                 (0.89, 0.04, 34000, 0.30),
    "Belair-Edison":            (0.82, 0.10, 40000, 0.22),
    "Edmondson Village":        (0.91, 0.03, 36000, 0.28),
    "Penn-North":               (0.94, 0.02, 27500, 0.39),
    "Johnston Square":          (0.93, 0.03, 26000, 0.40),
    "Berea":                    (0.90, 0.04, 31000, 0.33),
    "Waverly":                  (0.78, 0.15, 42000, 0.20),
    "Waltherson":               (0.84, 0.08, 38000, 0.24),
    "Frankford":                (0.65, 0.22, 39000, 0.25),
    "Irvington":                (0.87, 0.06, 35000, 0.29),
    "Forest Park-Walbrook":     (0.88, 0.05, 38000, 0.25),
    "Southwest Baltimore":      (0.75, 0.15, 36000, 0.28),
    "Westport":                 (0.85, 0.08, 30000, 0.35),
    "Cherry Hill":              (0.92, 0.02, 27000, 0.36),
    "Morrell Park":             (0.55, 0.32, 41000, 0.22),
    "Westgate":                 (0.90, 0.04, 34000, 0.30),
    "Mondawmin":                (0.91, 0.03, 33000, 0.31),
    "Mosher":                   (0.93, 0.03, 26500, 0.39),
    "Bridgeview-Greenlawn":     (0.88, 0.05, 35000, 0.28),
    "Wilson Park":              (0.87, 0.06, 36000, 0.27),
    "Poppleton":                (0.91, 0.03, 25000, 0.41),
    "New Southwest-Mount Clare": (0.78, 0.14, 34000, 0.29),
    "Barre Circle":             (0.65, 0.25, 45000, 0.18),
    "Franklin Square":          (0.82, 0.10, 28000, 0.37),
    "Booth-Boyd":               (0.88, 0.05, 34000, 0.30),
    "Lauraville":               (0.60, 0.30, 52000, 0.12),
    "Clifton Park":             (0.89, 0.05, 31000, 0.33),
    "Coldstream Homestead":     (0.92, 0.03, 29000, 0.36),
    "Broadway East":            (0.93, 0.03, 26000, 0.40),
    "Dunbar-Broadway":          (0.92, 0.03, 27000, 0.38),
    "Gay Street":               (0.91, 0.04, 27500, 0.37),
    "McElderry Park":           (0.90, 0.04, 28500, 0.36),
    "Middle East":              (0.92, 0.03, 25000, 0.42),
    "Greenmount East":          (0.91, 0.03, 27000, 0.39),
    "Care":                     (0.93, 0.02, 26000, 0.41),
    "East Baltimore Midway":    (0.93, 0.02, 26000, 0.41),
    "Ellwood Park-Monument":    (0.89, 0.05, 30000, 0.34),
    "Patterson Place":          (0.88, 0.06, 31000, 0.33),
    "Armistead Gardens":        (0.55, 0.35, 42000, 0.20),
    "Concerned Citizens":       (0.90, 0.04, 30000, 0.34),
    "Cedonia":                  (0.72, 0.18, 47000, 0.16),
    "Morgan Park":              (0.85, 0.08, 45000, 0.17),
    "Belair-Parkside":          (0.83, 0.10, 43000, 0.19),
    # White-majority / Mixed neighbourhoods
    "Hampden":                  (0.10, 0.82, 62000, 0.10),
    "Highlandtown":             (0.15, 0.65, 55000, 0.14),
    "Canton":                   (0.05, 0.88, 82000, 0.06),
    "Fells Point":              (0.08, 0.80, 78000, 0.08),
    "Butchers Hill":            (0.20, 0.72, 68000, 0.09),
    "Patterson Park":           (0.18, 0.68, 60000, 0.11),
    "Patterson Park North":     (0.22, 0.65, 58000, 0.12),
    "Ellwood Park":             (0.35, 0.52, 53000, 0.14),
    "Greektown":                (0.12, 0.75, 59000, 0.11),
    "O'Donnell Heights":        (0.38, 0.48, 46000, 0.17),
    "Locust Point":             (0.05, 0.88, 79000, 0.06),
    "Federal Hill":             (0.06, 0.87, 89000, 0.05),
    "Mount Vernon":             (0.28, 0.58, 55000, 0.14),
    "Station North":            (0.28, 0.58, 55000, 0.13),
    "Charles Village":          (0.18, 0.68, 52000, 0.15),
    "Medfield":                 (0.12, 0.75, 71000, 0.08),
    "Guilford":                 (0.12, 0.78, 98000, 0.04),
    "Roland Park":              (0.05, 0.88, 115000, 0.03),
    "Homeland":                 (0.08, 0.82, 108000, 0.04),
    "Northwood":                (0.45, 0.42, 58000, 0.13),
    "Chinquapin Park":          (0.42, 0.45, 56000, 0.14),
    "Pen Lucy":                 (0.75, 0.15, 44000, 0.19),
    "Lake Walker":              (0.55, 0.35, 55000, 0.13),
    "Loch Raven":               (0.50, 0.38, 57000, 0.13),
    "Ramblewood":               (0.52, 0.36, 53000, 0.15),
    "Govans":                   (0.62, 0.28, 50000, 0.15),
    "Harwood":                  (0.42, 0.45, 52000, 0.15),
    "Abell":                    (0.20, 0.68, 58000, 0.12),
    "Remington":                (0.22, 0.65, 55000, 0.13),
    "Seton Hill":               (0.58, 0.32, 38000, 0.26),
    "Inner Harbor":             (0.20, 0.65, 72000, 0.10),
    "Downtown":                 (0.32, 0.52, 62000, 0.12),
    "Jonestown":                (0.35, 0.50, 48000, 0.18),
    "Ridgely's Delight":        (0.08, 0.85, 85000, 0.06),
    "Otterbein":                (0.08, 0.85, 88000, 0.05),
    "Sharp-Leadenhall":         (0.55, 0.35, 45000, 0.19),
    "Pigtown":                  (0.48, 0.42, 42000, 0.21),
    "Barre Circle":             (0.65, 0.25, 45000, 0.18),
    "South Baltimore":          (0.10, 0.82, 72000, 0.09),
    "Brooklyn":                 (0.68, 0.20, 38000, 0.25),
}

# Baltimore city-wide averages (ACS 2019) — used as fallback
BALTIMORE_CITY_AVERAGE: Tuple[float, float, float, float] = (0.63, 0.28, 52000, 0.22)

# Chicago community area number → name mapping (official city mapping)
CHICAGO_COMMUNITY_AREAS: Dict[int, str] = {
    1: "Rogers Park",         2: "West Ridge",          3: "Uptown",
    4: "Lincoln Square",      5: "North Center",        6: "Lake View",
    7: "Lincoln Park",        8: "Near North Side",     9: "Edison Park",
    10: "Norwood Park",       11: "Jefferson Park",     12: "Forest Glen",
    13: "North Park",         14: "Albany Park",        15: "Portage Park",
    16: "Irving Park",        17: "Dunning",            18: "Montclare",
    19: "Belmont Cragin",     20: "Hermosa",            21: "Avondale",
    22: "Logan Square",       23: "Humboldt Park",      24: "West Town",
    25: "Austin",             26: "West Garfield Park", 27: "East Garfield Park",
    28: "Near West Side",     29: "North Lawndale",     30: "South Lawndale",
    31: "Lower West Side",    32: "Loop",               33: "Near South Side",
    34: "Armour Square",      35: "Douglas",            36: "Oakland",
    37: "Fuller Park",        38: "Grand Boulevard",    39: "Kenwood",
    40: "Washington Park",    41: "Hyde Park",          42: "Woodlawn",
    43: "South Shore",        44: "Chatham",            45: "Avalon Park",
    46: "South Chicago",      47: "Burnside",           48: "Calumet Heights",
    49: "Roseland",           50: "Pullman",            51: "South Deering",
    52: "East Side",          53: "West Pullman",       54: "Riverdale",
    55: "Hegewisch",          56: "Garfield Ridge",     57: "Archer Heights",
    58: "Brighton Park",      59: "McKinley Park",      60: "Bridgeport",
    61: "New City",           62: "West Elsdon",        63: "Gage Park",
    64: "Clearing",           65: "West Lawn",          66: "Chicago Lawn",
    67: "West Englewood",     68: "Englewood",          69: "Greater Grand Crossing",
    70: "Ashburn",            71: "Auburn Gresham",     72: "Beverly",
    73: "Washington Heights", 74: "Mount Greenwood",    75: "Morgan Park",
    76: "O'Hare",             77: "Edgewater",
}


# ── Racial Group Classification ────────────────────────────────────────────────

def classify_race_category(
    pct_black: float,
    pct_white: float,
    threshold: float = 0.50,
) -> str:
    """
    Classify a neighbourhood's racial majority.

    Parameters
    ----------
    pct_black : fraction of Black residents
    pct_white : fraction of White residents
    threshold : minimum fraction to be considered a majority

    Returns
    -------
    "Black" | "White" | "Neither"
    """
    if pct_black >= threshold:
        return "Black"
    if pct_white >= threshold:
        return "White"
    return "Neither"


# ── Neighbourhood Lookup ───────────────────────────────────────────────────────

def get_baltimore_demographics(
    neighbourhood: str,
) -> Tuple[float, float, float, float]:
    """
    Look up Census ACS 2019 demographics for a Baltimore neighbourhood.

    Uses exact match first, then partial match, then city-average fallback.

    Returns
    -------
    (pct_black, pct_white, median_income, poverty_rate)
    """
    if not isinstance(neighbourhood, str):
        return BALTIMORE_CITY_AVERAGE

    name = neighbourhood.strip()

    # Exact match
    if name in BALTIMORE_NEIGHBORHOOD_DEMOGRAPHICS:
        return BALTIMORE_NEIGHBORHOOD_DEMOGRAPHICS[name]

    # Case-insensitive exact match
    name_lower = name.lower()
    for key, val in BALTIMORE_NEIGHBORHOOD_DEMOGRAPHICS.items():
        if key.lower() == name_lower:
            return val

    # Partial match (neighbourhood name contained in key or vice-versa)
    for key, val in BALTIMORE_NEIGHBORHOOD_DEMOGRAPHICS.items():
        if name_lower in key.lower() or key.lower() in name_lower:
            return val

    # Fallback to city average
    return BALTIMORE_CITY_AVERAGE


# ── Baltimore Preprocessor ────────────────────────────────────────────────────

def preprocess_baltimore(
    raw_path: str,
    years: Tuple[int, ...] = (2017, 2018, 2019),
    bbox: Optional[Dict] = None,
) -> Dict[str, pd.DataFrame]:
    """
    Load and clean Baltimore Part 1 Crime data.

    Steps:
      1. Parse CrimeDateTime
      2. Filter to target years
      3. Drop null / out-of-bbox coordinates
      4. Map neighbourhood → Census ACS demographics
      5. Classify Race_Category

    Parameters
    ----------
    raw_path : path to the raw Baltimore CSV file
    years    : calendar years to retain
    bbox     : dict with lat_min, lat_max, lon_min, lon_max
               (defaults to Baltimore bounding box)

    Returns
    -------
    dict keyed by year and "combined"
    """
    if bbox is None:
        bbox = {"lat_min": 39.197, "lat_max": 39.372,
                "lon_min": -76.713, "lon_max": -76.529}

    print(f"Loading: {raw_path}")
    df = pd.read_csv(raw_path, low_memory=False)
    print(f"  Rows loaded: {len(df):,}")

    # Parse datetime
    df["CrimeDateTime"] = pd.to_datetime(df["CrimeDateTime"], errors="coerce")
    df = df.dropna(subset=["CrimeDateTime"])
    df["Year"]  = df["CrimeDateTime"].dt.year
    df["Month"] = df["CrimeDateTime"].dt.month
    df["Hour"]  = df["CrimeDateTime"].dt.hour
    df["Day"]   = df["CrimeDateTime"].dt.day

    # Filter to target years
    df = df[df["Year"].isin(years)].copy()
    print(f"  After year filter ({list(years)}): {len(df):,}")

    # Coordinate cleaning
    df = df.dropna(subset=["Latitude", "Longitude"])
    df["Latitude"]  = pd.to_numeric(df["Latitude"],  errors="coerce")
    df["Longitude"] = pd.to_numeric(df["Longitude"], errors="coerce")
    df = df.dropna(subset=["Latitude", "Longitude"])
    df = df[
        (df["Latitude"]  >= bbox["lat_min"]) & (df["Latitude"]  <= bbox["lat_max"]) &
        (df["Longitude"] >= bbox["lon_min"]) & (df["Longitude"] <= bbox["lon_max"])
    ].copy()
    print(f"  After coordinate clean: {len(df):,}")

    # Standardise text fields
    for col in ["Description", "Weapon", "PremiseType", "Neighborhood"]:
        if col in df.columns:
            df[col] = df[col].fillna("UNKNOWN").str.strip().str.upper()

    # District: prefer New_District, fall back to Old_District
    if "New_District" in df.columns and "Old_District" in df.columns:
        df["District"] = df["New_District"].fillna(df["Old_District"]).fillna("UNKNOWN")
    elif "Old_District" in df.columns:
        df["District"] = df["Old_District"].fillna("UNKNOWN")
    df["District"] = df["District"].str.strip().str.upper()

    # Map neighbourhood → demographics
    nbhd_col = "Neighborhood" if "Neighborhood" in df.columns else None
    if nbhd_col:
        demo = df[nbhd_col].apply(get_baltimore_demographics)
        df["Pct_Black"]      = demo.apply(lambda x: x[0])
        df["Pct_White"]      = demo.apply(lambda x: x[1])
        df["Median_Income"]  = demo.apply(lambda x: x[2])
        df["Poverty_Rate"]   = demo.apply(lambda x: x[3])
    else:
        df["Pct_Black"] = BALTIMORE_CITY_AVERAGE[0]
        df["Pct_White"] = BALTIMORE_CITY_AVERAGE[1]
        df["Median_Income"] = BALTIMORE_CITY_AVERAGE[2]
        df["Poverty_Rate"]  = BALTIMORE_CITY_AVERAGE[3]

    df["Race_Category"] = df.apply(
        lambda r: classify_race_category(r["Pct_Black"], r["Pct_White"]), axis=1
    )

    # Rename key columns for consistency
    rename_map = {
        "CCNumber":     "CaseNumber",
        "Neighborhood": "Neighborhood",
        "RowID":        "RowID",
    }
    for old, new in rename_map.items():
        if old in df.columns and old != new:
            df = df.rename(columns={old: new})

    # Final column selection
    keep_cols = [c for c in [
        "RowID", "CaseNumber", "CrimeDateTime", "Year", "Month", "Hour", "Day",
        "CrimeCode", "Description", "Inside_Outside", "Weapon", "District",
        "Neighborhood", "PremiseType", "Latitude", "Longitude",
        "Pct_Black", "Pct_White", "Median_Income", "Poverty_Rate", "Race_Category",
    ] if c in df.columns]
    df = df[keep_cols].copy()

    # Split by year
    result = {}
    for yr in years:
        result[yr] = df[df["Year"] == yr].reset_index(drop=True)
        print(f"  {yr}: {len(result[yr]):,} crimes | "
              f"Black={result[yr]['Race_Category'].eq('Black').mean():.1%} "
              f"White={result[yr]['Race_Category'].eq('White').mean():.1%}")
    result["combined"] = df.reset_index(drop=True)

    return result


# ── Chicago Preprocessor ───────────────────────────────────────────────────────

# Chicago ACS 2022 demographics per community area (selected areas)
CHICAGO_AREA_DEMOGRAPHICS: Dict[str, Tuple[float, float, float, float]] = {
    "Rogers Park": (0.24, 0.42, 48000, 0.21),
    "West Ridge": (0.05, 0.38, 55000, 0.18),
    "Uptown": (0.18, 0.52, 62000, 0.15),
    "Lincoln Square": (0.05, 0.72, 78000, 0.08),
    "North Center": (0.02, 0.85, 95000, 0.05),
    "Lake View": (0.03, 0.85, 92000, 0.06),
    "Lincoln Park": (0.04, 0.84, 105000, 0.05),
    "Near North Side": (0.07, 0.78, 98000, 0.07),
    "Austin": (0.94, 0.02, 32000, 0.35),
    "West Garfield Park": (0.96, 0.01, 27000, 0.42),
    "East Garfield Park": (0.95, 0.01, 28000, 0.40),
    "Near West Side": (0.35, 0.42, 58000, 0.18),
    "North Lawndale": (0.95, 0.01, 26000, 0.44),
    "South Lawndale": (0.10, 0.05, 40000, 0.28),
    "Loop": (0.08, 0.68, 88000, 0.10),
    "Douglas": (0.79, 0.12, 42000, 0.25),
    "Oakland": (0.92, 0.03, 30000, 0.38),
    "Fuller Park": (0.95, 0.01, 26000, 0.45),
    "Grand Boulevard": (0.94, 0.02, 29000, 0.40),
    "Kenwood": (0.74, 0.18, 52000, 0.18),
    "Washington Park": (0.96, 0.01, 24000, 0.46),
    "Hyde Park": (0.30, 0.48, 72000, 0.14),
    "Woodlawn": (0.91, 0.04, 32000, 0.36),
    "South Shore": (0.92, 0.03, 33000, 0.35),
    "Chatham": (0.97, 0.01, 38000, 0.25),
    "Avalon Park": (0.96, 0.01, 42000, 0.21),
    "South Chicago": (0.52, 0.08, 35000, 0.30),
    "Roseland": (0.96, 0.01, 34000, 0.32),
    "West Pullman": (0.95, 0.01, 33000, 0.33),
    "Riverdale": (0.97, 0.01, 22000, 0.50),
    "Englewood": (0.97, 0.01, 24000, 0.46),
    "West Englewood": (0.96, 0.01, 28000, 0.40),
    "Auburn Gresham": (0.96, 0.01, 36000, 0.27),
    "Greater Grand Crossing": (0.95, 0.01, 33000, 0.33),
    "Humboldt Park": (0.51, 0.06, 34000, 0.35),
    "Logan Square": (0.14, 0.62, 72000, 0.11),
    "Edgewater": (0.10, 0.60, 65000, 0.14),
    "Bridgeport": (0.07, 0.42, 55000, 0.16),
    "New City": (0.43, 0.12, 38000, 0.30),
    "Gage Park": (0.08, 0.08, 42000, 0.26),
    "Chicago Lawn": (0.32, 0.12, 40000, 0.27),
    "Morgan Park": (0.73, 0.18, 58000, 0.12),
    "Beverly": (0.25, 0.68, 82000, 0.07),
    "Mount Greenwood": (0.02, 0.92, 85000, 0.05),
    "Washington Heights": (0.89, 0.04, 42000, 0.22),
}
CHICAGO_CITY_AVERAGE: Tuple[float, float, float, float] = (0.29, 0.33, 58000, 0.18)


def preprocess_chicago(
    raw_path: str,
    year: int = 2022,
) -> pd.DataFrame:
    """
    Load and clean Chicago Crimes dataset.

    Parameters
    ----------
    raw_path : path to the raw Chicago CSV file
    year     : target year

    Returns
    -------
    Cleaned DataFrame with Race_Category and demographic columns
    """
    print(f"Loading: {raw_path}")
    df = pd.read_csv(raw_path, low_memory=False)
    print(f"  Rows loaded: {len(df):,}")

    # Parse date
    df["Date"] = pd.to_datetime(df["Date"], errors="coerce")
    df = df.dropna(subset=["Date"])
    df["Year"]  = df["Date"].dt.year
    df["Month"] = df["Date"].dt.month
    df["Hour"]  = df["Date"].dt.hour

    df = df[df["Year"] == year].copy()

    # Coordinate cleaning
    df = df.dropna(subset=["Latitude", "Longitude"])
    df["Latitude"]  = pd.to_numeric(df["Latitude"],  errors="coerce")
    df["Longitude"] = pd.to_numeric(df["Longitude"], errors="coerce")
    df = df.dropna(subset=["Latitude", "Longitude"])
    print(f"  After coordinate clean: {len(df):,}")

    # Map community area number → name
    if "Community Area" in df.columns:
        df["Neighborhood"] = (
            df["Community Area"]
            .apply(lambda x: CHICAGO_COMMUNITY_AREAS.get(int(x), "Unknown")
                   if pd.notna(x) else "Unknown")
        )
    else:
        df["Neighborhood"] = "Unknown"

    # Map neighbourhood → demographics
    def get_chicago_demo(name: str) -> Tuple[float, float, float, float]:
        if name in CHICAGO_AREA_DEMOGRAPHICS:
            return CHICAGO_AREA_DEMOGRAPHICS[name]
        return CHICAGO_CITY_AVERAGE

    demo = df["Neighborhood"].apply(get_chicago_demo)
    df["Pct_Black"]     = demo.apply(lambda x: x[0])
    df["Pct_White"]     = demo.apply(lambda x: x[1])
    df["Median_Income"] = demo.apply(lambda x: x[2])
    df["Poverty_Rate"]  = demo.apply(lambda x: x[3])
    df["Race_Category"] = df.apply(
        lambda r: classify_race_category(r["Pct_Black"], r["Pct_White"]), axis=1
    )

    # Rename for consistency
    rename = {
        "ID":             "RowID",
        "Case Number":    "CaseNumber",
        "Primary Type":   "CrimeType",
        "Description":    "Description",
        "Location Description": "PremiseType",
        "Arrest":         "Arrest",
        "Domestic":       "Domestic",
        "Beat":           "Beat",
        "District":       "District",
        "Ward":           "Ward",
        "Community Area": "Community_Area_Number",
    }
    df = df.rename(columns={k: v for k, v in rename.items() if k in df.columns})

    keep_cols = [c for c in [
        "RowID", "CaseNumber", "Date", "Year", "Month", "Hour",
        "CrimeType", "Description", "PremiseType",
        "Arrest", "Domestic", "Beat", "District", "Ward",
        "Community_Area_Number", "Neighborhood",
        "Latitude", "Longitude",
        "Pct_Black", "Pct_White", "Median_Income", "Poverty_Rate", "Race_Category",
    ] if c in df.columns]
    df = df[keep_cols].reset_index(drop=True)

    print(f"  {year}: {len(df):,} crimes | "
          f"Black={df['Race_Category'].eq('Black').mean():.1%} "
          f"White={df['Race_Category'].eq('White').mean():.1%}")
    return df
