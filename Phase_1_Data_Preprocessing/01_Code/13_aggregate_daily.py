#!/usr/bin/env python3

from importlib import import_module as _load_module
C = _load_module("15_pipeline_config").from_environment()

# Aggregate each well-Reach-Node-day to one value to avoid weighting overlapping granules twice.
from pathlib import Path

import pandas as pd

WORK_DIR = C.tables()["directory"]
INPUT_FILE = C.tables()["top1"]
OUTPUT_FILE = C.tables()["daily"]

GROUP_KEYS = ["site_no", "reach_id", "node_id", "number_match_SG", "date"]
MEAN_COLUMNS = {
    "river_lon", "river_lat", "river_width", "river_wse",
    "xtrk_dist", "distance_adjusted_m", "distance_m",
    "distance_prior_bank_m", "distance_prior_center_m",
    "distance_observed_bank_m", "distance_observed_center_m",
    "Value_EGM2008_m", "GWL_Value_m",
    "Depth to Water Below Land Surface in meter",
    "precipitation_mm", "evapotranspiration_mm", "estimated_wse",
}
# Use conservative quality summaries: larger uncertainty/quality code/dark fraction and fewer good pixels.
MAX_COLUMNS = {"river_wse_u", "river_node_q", "dark_frac"}
MIN_COLUMNS = {"n_good_pix"}

def main():
    if not INPUT_FILE.exists():
        raise FileNotFoundError(f"Previous-stage output not found: {INPUT_FILE}")

    print(f"Reading selected daily groundwater-river data: {INPUT_FILE}")
    df = pd.read_csv(INPUT_FILE, sep="\t", low_memory=False)
    original_columns = df.columns.tolist()
    missing_columns = sorted(set(GROUP_KEYS).difference(df.columns))
    if missing_columns:
        raise KeyError(f"Input lacks daily aggregation keys: {missing_columns}")

    df["date"] = pd.to_datetime(df["date"], errors="coerce")
    invalid_date_count = int(df["date"].isna().sum())
    if invalid_date_count:
        raise ValueError(f"Unparseable dates are not silently discarded: {invalid_date_count:,} records")

    # Restrict to the study period. January 2026 climate supplies only the final December 2025 daily endpoint;
    # do not include 2026 observations in the daily/monthly analysis.
    STUDY_START = pd.Timestamp("2023-01-01")
    STUDY_END = pd.Timestamp("2025-12-31")
    study_period_mask = df["date"].between(STUDY_START, STUDY_END)
    out_of_period_count = int((~study_period_mask).sum())
    df = df.loc[study_period_mask].copy()

    original_row_count = len(df)
    exact_duplicate_count = int(df.duplicated().sum())
    df = df.drop_duplicates().copy()

    same_day_mask = df.duplicated(GROUP_KEYS, keep=False)
    same_day_rows = df.loc[same_day_mask].copy()
    single_rows = df.loc[~same_day_mask].copy()
    same_day_group_count = (
        same_day_rows.groupby(GROUP_KEYS, dropna=False, sort=False).ngroups
        if len(same_day_rows) else 0
    )

    aggregation = {}
    for column in original_columns:
        if column in GROUP_KEYS:
            continue
        if column in MEAN_COLUMNS:
            aggregation[column] = "mean"
        elif column in MAX_COLUMNS:
            aggregation[column] = "max"
        elif column in MIN_COLUMNS:
            aggregation[column] = "min"
        else:
            aggregation[column] = "first"

    if len(same_day_rows):
        daily_groups = (
            same_day_rows.groupby(GROUP_KEYS, as_index=False, dropna=False, sort=False)
                         .agg(aggregation)
        )
        daily = pd.concat([single_rows, daily_groups], ignore_index=True, sort=False)
    else:
        daily = single_rows

    # Keep estimated_wse identical to the aggregated river_wse.
    if "estimated_wse" in daily.columns:
        daily["estimated_wse"] = daily["river_wse"]
    daily = daily[original_columns]
    daily = daily.sort_values(
        ["site_no", "number_match_SG", "date", "node_id"], kind="stable"
    ).reset_index(drop=True)

    remaining_duplicate_count = int(daily.duplicated(GROUP_KEYS).sum())
    if remaining_duplicate_count:
        raise AssertionError(f"Duplicate keys remain after daily aggregation: {remaining_duplicate_count:,} records")

    daily.to_csv(OUTPUT_FILE, sep="\t", index=False)
    print(f"Input rows: {original_row_count:,}")
    print(f"Rows outside the study period (excluded): {out_of_period_count:,}")
    print(f"Exact duplicate rows removed within the study period: {exact_duplicate_count:,}")
    print(f"Nonidentical same-day groups to aggregate: {same_day_group_count:,} groups / {len(same_day_rows):,} rows")
    print(f"Unique daily records: {len(daily):,} rows (reduced by {original_row_count - len(daily):,})")
    print(f"Saved to: {OUTPUT_FILE}")


if __name__ == "__main__":
    main()
