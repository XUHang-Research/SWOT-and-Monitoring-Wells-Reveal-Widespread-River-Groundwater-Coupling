
from importlib import import_module as _load_module
C = _load_module("15_pipeline_config").from_environment()

import os
import time
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import numpy as np
import pandas as pd
import xarray as xr


# ================= Configuration =================
CODE_DIR = Path(__file__).resolve().parent
HERE = C.tables()["directory"]
INPUT_FILE = str(C.tables()["joined"])
PRE_FOLDER = str(C.precipitation_daily)
ET_FOLDER = str(C.evapotranspiration_daily)
OUTPUT_FILE = str(C.tables()["climate"])

LON_COL = "wells_lon"
LAT_COL = "wells_lat"
DATE_COL = "date"

PRE_VAR = "tp"
ET_VAR = "e"

PRE_OUT_COL = "precipitation_mm"
ET_OUT_COL = "evapotranspiration_mm"

# ERA5-Land monthly files are large. Keep this moderate unless memory is ample.
NUM_WORKERS = C.workers["climate_join"]
# ================================================


def build_month_file(folder, prefix, year, month):
    return os.path.join(folder, f"{prefix}_{year}_{month:02d}.nc")


def normalize_longitudes(lons, ds):
    lon_min = float(ds["longitude"].min())
    lon_max = float(ds["longitude"].max())
    lons = np.asarray(lons, dtype="float64")

    if lon_min >= 0 and lon_max > 180:
        return np.mod(lons, 360.0)
    if lon_min < 0 and lon_max <= 180:
        return ((lons + 180.0) % 360.0) - 180.0
    return lons


def extract_dataset_values(nc_path, var_name, group_df):
    values = np.full(len(group_df), np.nan, dtype="float64")
    if not os.path.exists(nc_path):
        return values, f"missing: {os.path.basename(nc_path)}"

    try:
        with xr.open_dataset(nc_path) as ds:
            if var_name not in ds.data_vars:
                return values, f"missing variable {var_name}: {os.path.basename(nc_path)}"

            time_dim = "valid_time" if "valid_time" in ds.dims else "time"
            lons = normalize_longitudes(group_df[LON_COL].values, ds)

            lat_idx = ds.indexes["latitude"].get_indexer(
                group_df[LAT_COL].to_numpy(), method="nearest"
            )
            lon_idx = ds.indexes["longitude"].get_indexer(lons, method="nearest")
            time_idx = ds.indexes[time_dim].get_indexer(
                group_df["_date_dt"].to_numpy(), method="nearest"
            )
            cube = np.asarray(ds[var_name].load().values)
            values = np.asarray(cube[time_idx, lat_idx, lon_idx], dtype="float64")
            return values, None

    except Exception as exc:
        return values, f"failed {os.path.basename(nc_path)}: {exc}"


def process_month_group(args):
    (year, month), group_df = args

    pre_path = build_month_file(
        PRE_FOLDER, "ERA5_Land_Precip_Daily", year, month
    )
    et_path = build_month_file(
        ET_FOLDER, "ERA5_Land_ET_Daily", year, month
    )

    pre_values, pre_msg = extract_dataset_values(pre_path, PRE_VAR, group_df)
    et_values, et_msg = extract_dataset_values(et_path, ET_VAR, group_df)

    messages = []
    if pre_msg:
        messages.append(f"PRE {year}-{month:02d}: {pre_msg}")
    if et_msg:
        messages.append(f"ET {year}-{month:02d}: {et_msg}")

    return group_df["_query_index"].to_numpy(), pre_values, et_values, messages


def prepare_tasks(df):
    valid = df[[LON_COL, LAT_COL, DATE_COL]].copy()
    valid["_row_index"] = df.index.to_numpy()
    valid[LON_COL] = pd.to_numeric(valid[LON_COL], errors="coerce")
    valid[LAT_COL] = pd.to_numeric(valid[LAT_COL], errors="coerce")
    valid["_date_dt"] = pd.to_datetime(valid[DATE_COL], errors="coerce")
    valid = valid.dropna(subset=[LON_COL, LAT_COL, "_date_dt"]).copy()

    key_columns = [LON_COL, LAT_COL, "_date_dt"]
    keys = pd.MultiIndex.from_frame(valid[key_columns])
    query_codes, unique_keys = pd.factorize(keys, sort=False)
    queries = unique_keys.to_frame(index=False)
    queries.columns = key_columns
    queries["_query_index"] = np.arange(len(queries), dtype="int64")
    queries["_year"] = queries["_date_dt"].dt.year.astype("int16")
    queries["_month"] = queries["_date_dt"].dt.month.astype("int8")

    tasks = []
    for (year, month), group in queries.groupby(["_year", "_month"], sort=True):
        tasks.append(((int(year), int(month)), group.copy()))
    return (
        tasks,
        valid["_row_index"].to_numpy(dtype="int64"),
        query_codes.astype("int64", copy=False),
        len(valid),
        len(queries),
    )


def main():
    start_time = time.time()

    print("1. Reading input result table...")
    df = pd.read_csv(INPUT_FILE, sep="\t", low_memory=False)
    print(f"   Rows: {len(df):,}")

    missing_cols = [col for col in [LON_COL, LAT_COL, DATE_COL] if col not in df.columns]
    if missing_cols:
        raise ValueError(f"Input file is missing required columns: {missing_cols}")

    print("2. Preparing monthly extraction tasks...")
    tasks, valid_indices, query_codes, valid_rows, unique_queries = prepare_tasks(df)
    print(f"   Valid rows with lon/lat/date: {valid_rows:,}")
    print(f"   Unique well-coordinate + date queries: {unique_queries:,}")
    print(f"   Monthly tasks: {len(tasks)}")
    print(f"   Workers: {NUM_WORKERS}")

    df[PRE_OUT_COL] = np.nan
    df[ET_OUT_COL] = np.nan

    print("3. Extracting PRE and ET values from daily NetCDF files...")
    warning_count = 0
    query_pre = np.full(unique_queries, np.nan, dtype="float64")
    query_et = np.full(unique_queries, np.nan, dtype="float64")
    with ProcessPoolExecutor(max_workers=NUM_WORKERS) as executor:
        for i, (indices, pre_values, et_values, messages) in enumerate(
            executor.map(process_month_group, tasks), start=1
        ):
            query_pre[indices] = pre_values
            query_et[indices] = et_values

            if messages:
                warning_count += len(messages)
                for msg in messages:
                    print(f"   Warning: {msg}")

            print(f"   [{i}/{len(tasks)}] filled {len(indices):,} rows")

    df.loc[valid_indices, PRE_OUT_COL] = query_pre[query_codes]
    df.loc[valid_indices, ET_OUT_COL] = query_et[query_codes]

    print("4. Saving output table...")
    os.makedirs(os.path.dirname(OUTPUT_FILE), exist_ok=True)
    df.to_csv(OUTPUT_FILE, sep="\t", index=False)

    elapsed = time.time() - start_time
    print("Done.")
    print(f"   Output: {OUTPUT_FILE}")
    print(f"   Warnings: {warning_count}")
    print(f"   Elapsed: {elapsed:.2f} seconds")

    preview_cols = [LON_COL, LAT_COL, DATE_COL, PRE_OUT_COL, ET_OUT_COL]
    print(df[preview_cols].head().to_string(index=False))


if __name__ == "__main__":
    main()
