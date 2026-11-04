#!/usr/bin/env python3

from importlib import import_module as _load_module
C = _load_module("15_pipeline_config").from_environment()

### Join the observations
import pandas as pd
import numpy as np
import os

# === File paths ===
# Read the Reach-enriched matching output.
sites_txt = str(C.tables()["reaches"])
wells_txt = str(C.groundwater_dem)
out_txt = str(C.tables()["joined"])
STUDY_START = pd.Timestamp('2023-01-01')
STUDY_END = pd.Timestamp('2025-12-31')

# === Helper functions ===
def normalize_colname(name: str) -> str:
    # Trim whitespace, lowercase names, and replace slashes by underscores.
    return str(name).strip().lower().replace("/", "_")

def find_col(df, targets):
    """Find the actual column corresponding to a normalized name."""
    norm_map = {normalize_colname(c): c for c in df.columns}
    for t in targets:
        key = normalize_colname(t)
        if key in norm_map:
            return norm_map[key]
    return None

def parse_dates_to_day(s: pd.Series) -> pd.Series:
    # Convert to datetime and normalize to calendar days.
    dt = pd.to_datetime(s, errors='coerce', utc=True)
    return dt.dt.tz_convert(None).dt.normalize()

def main():
    # 1. === Read Sites (SWOT river observations) ===
    print("Reading the sites table...")
    df_sites = pd.read_csv(sites_txt, sep="\t", dtype=str, low_memory=False)
    
    # Normalize site_no.
    df_sites["site_no"] = df_sites["site_no"].astype(str).str.strip()
    
    # Extract the date from shp_file when a separate date column is unavailable.
    if 'date' in df_sites.columns:
        df_sites['date'] = parse_dates_to_day(df_sites['date'])
    elif 'shp_file' in df_sites.columns:
        df_sites['date'] = parse_dates_to_day(df_sites['shp_file'].str.extract(r'(\d{8})')[0])
    
    # Exclude invalid rows.
    df_sites = df_sites.dropna(subset=['site_no', 'date'])
    df_sites = df_sites[df_sites['date'].between(STUDY_START, STUDY_END)].copy()
    print(f"Sites loaded: {len(df_sites)} rows")

    # 2. === Read Wells (groundwater observations) ===
    print("Reading the wells table...")
    df_wells = pd.read_csv(wells_txt, sep="\t", dtype=str, low_memory=False)
    
    # Resolve required columns; the original site_no values already match in both sources.
    col_site = find_col(df_wells, ['site_no'])
    col_date = find_col(df_wells, ['date', 'times']) 

    if not all([col_site, col_date]):
        raise ValueError("Wells input lacks site_no or date/times; inspect its header.")

    df_wells['site_no'] = df_wells[col_site].astype(str).str.strip()
    df_wells['date'] = parse_dates_to_day(df_wells[col_date])
    
    # Retain the required source fields.
    cols_to_keep = ['site_no', 'date']
    expected_extra_cols = [
        'wells_lon',
        'wells_lat',
        'Value_EGM2008_m',
        'GWL_Value_m',
        'Well_Elevation_EGM2008',
        'AquiferType',
        'country_province',
        'Depth to Water Below Land Surface in meter',
        'Well_Elevation',
    ]
    
    for extra in expected_extra_cols:
        actual_col = find_col(df_wells, [extra])
        if actual_col and actual_col not in cols_to_keep:
            cols_to_keep.append(actual_col)
            
    df_wells = df_wells[cols_to_keep].dropna(subset=['site_no', 'date'])
    df_wells = df_wells[df_wells['date'].between(STUDY_START, STUDY_END)].copy()

    # Retain wells present in sites, including every available date for those wells.
    site_no_set = set(df_sites['site_no'].dropna().unique())
    print(f"Distinct site_no values in sites: {len(site_no_set)}")
    df_wells = df_wells[df_wells['site_no'].isin(site_no_set)].copy()
    
    # Remove duplicate well-date keys to avoid expanding the join.
    df_wells = df_wells.drop_duplicates(subset=['site_no', 'date'], keep='first')
    
    print(f"Wells records with site_no present in sites: {len(df_wells)} rows")
    print(f"Distinct retained well identifiers: {df_wells['site_no'].nunique()}")

    # 3. === Restricted outer join ===
    # Include:
    # 1) existing site_no + date records from the sites table;
    # 2) other dates for the same wells, with missing river-side fields.
    print("Joining observations with a restricted outer merge...")
    df_merged = pd.merge(
        df_sites,
        df_wells,
        on=['site_no', 'date'],
        how='outer',
        suffixes=("", "_wells"),
    )

    # 4. === Consolidate overlapping column names ===
    # Overlapping fields receive a _wells suffix on the groundwater side.
    # Fill missing original fields from the suffixed fields, then drop the suffix columns.
    duplicate_suffix = "_wells"
    duplicate_cols = [col for col in df_merged.columns if col.endswith(duplicate_suffix)]
    for duplicate_col in duplicate_cols:
        base_col = duplicate_col[:-len(duplicate_suffix)]
        if base_col in df_merged.columns:
            df_merged[base_col] = df_merged[base_col].combine_first(df_merged[duplicate_col])
            df_merged = df_merged.drop(columns=[duplicate_col])
        else:
            df_merged = df_merged.rename(columns={duplicate_col: base_col})

    # Sort records.
    df_merged = df_merged.sort_values(by=['site_no', 'date'])

    # 5. === Save ===
    print(f"Join complete; total rows: {len(df_merged)}")
    print(f"Distinct joined wells: {df_merged['site_no'].nunique()}")
    
    # Create the output directory if needed.
    os.makedirs(os.path.dirname(out_txt), exist_ok=True)
    
    print(f"Saving to: {out_txt}")
    df_merged.to_csv(out_txt, sep="\t", index=False)
    print("Processing complete.")

if __name__ == "__main__":
    main()
