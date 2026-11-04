#!/usr/bin/env python3

from importlib import import_module as _load_module
C = _load_module("15_pipeline_config").from_environment()

import pandas as pd

# File path
file_path = str(C.tables()["selected"])

def check_number_match():
    print(f"Reading file: {file_path} ...")
    # The source is tab-separated; change sep only if its format differs.
    df = pd.read_csv(file_path, sep='\t', low_memory=False)
    
    # Check that the ranking column exists.
    if 'number_match_SG' not in df.columns:
        print("Error: column 'number_match_SG' was not found.")
        print("Available columns:", df.columns.tolist())
        return

    # Extract distinct nonmissing ranks.
    unique_values = df['number_match_SG'].dropna().unique()
    print("-" * 40)
    print(f"Distinct number_match_SG values: {unique_values}")
    
    # Multiple ranks are allowed here; the following cell retains only rank 1.
    print("Record counts by candidate rank:")
    print(df['number_match_SG'].value_counts(dropna=False))
    numeric_rank = pd.to_numeric(df['number_match_SG'], errors='coerce')
    if numeric_rank.notna().all() and numeric_rank.ge(1).all() and numeric_rank.eq(1).any():
        print("Candidate ranks passed; the next cell retains number_match_SG == 1.")
    else:
        raise ValueError("number_match_SG is missing, nonpositive, or contains no rank 1.")

if __name__ == "__main__":
    check_number_match()