#!/usr/bin/env python3

from importlib import import_module as _load_module
C = _load_module("15_pipeline_config").from_environment()

### Select fixed well-river pairs
import pandas as pd

# === Path configuration ===
input_csv = str(C.tables()["climate"])
sorted_first_output_path = str(C.tables()["selected"])

def main():
    print(f"Reading input data: {input_csv}")
    df = pd.read_csv(input_csv, sep="\t")
    print(f"Data loaded: {len(df)} rows.")

    # Exclude missing reach_id/node_id from the candidate-ranking statistics.
    valid_df = df.dropna(subset=['reach_id', 'node_id'])

    # 1. Use static prior-bank distance and count distinct observation dates.
    # Overlapping same-day granules are not independent temporal observations.
    valid_df = valid_df.copy()
    valid_df['date'] = pd.to_datetime(valid_df['date'], errors='coerce')
    valid_df['distance_prior_bank_m'] = pd.to_numeric(
        valid_df['distance_prior_bank_m'], errors='coerce'
    )
    valid_df = valid_df.dropna(subset=['date', 'distance_prior_bank_m'])
    print("Calculating static prior distances and distinct observation-day counts...")
    node_stats = valid_df.groupby(['site_no', 'reach_id', 'node_id']).agg(
        prior_bank_distance_m=('distance_prior_bank_m', 'median'),
        obs_day_count=('date', 'nunique'),
    ).reset_index()

    # 2. Select one fixed Node within each well-Reach combination.
    print("Selecting fixed Nodes by observation-day count, then distance...")
    node_stats = node_stats.sort_values(
        by=['site_no', 'reach_id', 'obs_day_count', 'prior_bank_distance_m', 'node_id'],
        ascending=[True, True, False, True, True]
    )
    best_nodes = node_stats.groupby(['site_no', 'reach_id']).head(1).copy()

    # 3. Rank the selected Reaches within each well.
    print("Assigning Reach priority ranks (number_match_SG)...")
    best_nodes = best_nodes.sort_values(
        by=['site_no', 'obs_day_count', 'prior_bank_distance_m', 'reach_id', 'node_id'],
        ascending=[True, False, True, True, True]
    )
    best_nodes['number_match_SG'] = best_nodes.groupby('site_no').cumcount() + 1

    # ================= Fill early missing pair identifiers =================
    print("Assigning the best pair to early records lacking Reach and Node IDs...")
    
    # A. Identify the highest-ranked Reach/Node combination for each well.
    top1_nodes = best_nodes[best_nodes['number_match_SG'] == 1][['site_no', 'reach_id', 'node_id']]
    top1_nodes = top1_nodes.rename(columns={'reach_id': 'best_reach_id', 'node_id': 'best_node_id'})

    # B. Join this fallback pair by site_no, including early unmatched records.
    df = pd.merge(df, top1_nodes, on='site_no', how='left')

    # C. Fill missing reach_id and node_id values.
    df['reach_id'] = df['reach_id'].fillna(df['best_reach_id'])
    df['node_id'] = df['node_id'].fillna(df['best_node_id'])

    # Remove helper columns.
    df = df.drop(columns=['best_reach_id', 'best_node_id'])
    # ====================================================

    # 4. Filter the candidate observations.
    print("Retaining the selected fixed-Node time series...")
    # Early unmatched records now have the selected pair identifiers.
    # Join to best_nodes to exclude rejected Nodes while retaining early well records.
    best_nodes_subset = best_nodes[['site_no', 'reach_id', 'node_id', 'number_match_SG']]
    df_final = pd.merge(df, best_nodes_subset, on=['site_no', 'reach_id', 'node_id'], how='inner')

    # Restore the complete well calendar; rejected-Node dates must not remove groundwater observations.
    helper = vars(_load_module("17_preserve_groundwater_dates"))
    top1 = best_nodes_subset.loc[best_nodes_subset['number_match_SG'].eq(1)]
    df_final = helper['restore_top1_groundwater_dates'](df, df_final, top1)

    # 5. Generate estimated_wse and sort by well, Reach rank and observation date.
    df_final['estimated_wse'] = df_final['river_wse']
    df_final = df_final.sort_values(by=['site_no', 'number_match_SG', 'shp_file']).reset_index(drop=True)

    # 6. Save results.
    print(f"Extraction complete: {len(df_final)} time-series records.")
    print(f"Saving selected records to: {sorted_first_output_path}")
    df_final.to_csv(sorted_first_output_path, index=False, sep="\t")
    
    print("\nPair selection and missing-identifier assignment complete.")

if __name__ == "__main__":
    main()