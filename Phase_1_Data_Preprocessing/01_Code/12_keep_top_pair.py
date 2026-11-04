#!/usr/bin/env python3

from importlib import import_module as _load_module
C = _load_module("15_pipeline_config").from_environment()

### Retain records with number_match_SG == 1
import os

import pandas as pd


input_file = str(C.tables()["selected"])
output_file = str(C.tables()["top1"])


def filter_number_match_sg1():
    print(f"Reading file: {input_file}")
    df = pd.read_csv(input_file, sep='	', low_memory=False)

    if 'number_match_SG' not in df.columns:
        raise ValueError(
            "Column 'number_match_SG' was not found. Available columns: "
            f"{df.columns.tolist()}"
        )

    number_match_sg = pd.to_numeric(df['number_match_SG'], errors='coerce')
    df_sg1 = df[number_match_sg == 1].copy()

    os.makedirs(os.path.dirname(output_file), exist_ok=True)
    df_sg1.to_csv(output_file, sep='	', index=False)

    print(f"Input rows: {len(df)}")
    print(f"Rows with number_match_SG == 1: {len(df_sg1)}")
    if 'site_no' in df_sg1.columns:
        print(f"Distinct selected wells: {df_sg1['site_no'].nunique()}")
    print(f"Saved to: {output_file}")


if __name__ == '__main__':
    filter_number_match_sg1()
