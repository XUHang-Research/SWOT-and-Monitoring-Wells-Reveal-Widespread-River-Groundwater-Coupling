
from importlib import import_module as _load_module
C = _load_module("15_pipeline_config").from_environment()

import glob
import calendar
import os
import re
import tempfile
import time
import zipfile
from concurrent.futures import ProcessPoolExecutor

import numpy as np
import xarray as xr


# ================= Configuration =================
# 1. Input folder: these .nc files are zip archives containing data_0.nc.
input_folder = str(C.inputs["precipitation_hourly"])

# 2. Output folder
output_folder = str(C.precipitation_daily)

# 3. Parallel workers
# Each output month needs both the current and following monthly archive so
# that the last day's accumulation can be read at next-month 00:00 UTC.
# Keep this conservative because every worker may extract two large archives.
num_workers = C.workers["climate_conversion"]

# 4. Unit conversion
# ERA5-Land total precipitation is usually in m of water equivalent. Daily
# precipitation is often used in mm.
convert_to_mm = True

# 5. Variable selection
# Set this to a fixed variable name if needed, for example "tp".
target_variable = "tp"

# 6. xarray backend
# Use None for automatic detection after extracting the real NetCDF member.
xarray_engine = None

# 7. Existing daily files were produced by summing cumulative snapshots and
# must be replaced when this corrected script is run.
overwrite_existing = True
# ================================================


VARIABLE_CANDIDATES = (
    "tp",
    "precipitation",
    "total_precipitation",
    "pre",
    "PRE",
)

DAILY_AGGREGATION = (
    "ERA5-Land accumulated forecast value at next-day 00:00 UTC; "
    "timestamp shifted back one day"
)


def find_time_dim(ds):
    for name in ("valid_time", "time"):
        if name in ds.dims or name in ds.coords:
            return name

    for coord_name in ds.coords:
        if "time" in coord_name.lower():
            return coord_name

    raise ValueError(f"Could not find a time dimension. Dims: {list(ds.dims)}")


def find_variable(ds):
    if target_variable is not None:
        if target_variable not in ds.data_vars:
            raise ValueError(
                f"target_variable={target_variable!r} not found. "
                f"Available variables: {list(ds.data_vars)}"
            )
        return target_variable

    for name in VARIABLE_CANDIDATES:
        if name in ds.data_vars:
            return name

    data_vars = list(ds.data_vars)
    if len(data_vars) == 1:
        return data_vars[0]

    raise ValueError(
        "Could not auto-detect precipitation variable. "
        f"Available variables: {data_vars}. Please set target_variable."
    )


def is_zip_file(path):
    return zipfile.is_zipfile(path)


def extract_first_netcdf(zip_path, temp_dir):
    with zipfile.ZipFile(zip_path) as zf:
        nc_members = [
            item for item in zf.namelist()
            if item.lower().endswith((".nc", ".nc4", ".cdf"))
        ]
        if not nc_members:
            raise ValueError("Zip archive contains no NetCDF member.")

        member = nc_members[0]
        return zf.extract(member, path=temp_dir)


def open_dataset_safely(path):
    if xarray_engine:
        return xr.open_dataset(path, engine=xarray_engine)
    return xr.open_dataset(path)


def output_is_corrected(path):
    try:
        with open_dataset_safely(path) as ds:
            return (
                target_variable in ds.data_vars
                and ds[target_variable].attrs.get("daily_aggregation")
                == DAILY_AGGREGATION
            )
    except Exception:
        return False


def build_output_name(file_name):
    save_name = file_name
    if save_name.endswith(".zip"):
        save_name = save_name[:-4]

    if "Hourly" in save_name:
        save_name = save_name.replace("Hourly", "Daily")
    else:
        save_name = f"Daily_{save_name}"

    if not save_name.endswith(".nc"):
        save_name = f"{save_name}.nc"

    return save_name


def parse_file_month(path):
    match = re.search(r"_(\d{4})_(\d{2})(?:\.[^.]+)+$", os.path.basename(path))
    if not match:
        raise ValueError(f"Could not parse YYYY_MM from input file: {path}")

    year, month = map(int, match.groups())
    if not 1 <= month <= 12:
        raise ValueError(f"Invalid month in input file: {path}")
    return year, month


def next_month(year, month):
    if month == 12:
        return year + 1, 1
    return year, month + 1


def prepare_dataset(ds):
    if "expver" in ds.dims:
        return ds.sel(expver=1).combine_first(ds.sel(expver=5))
    return ds


def midnight_accumulations(ds, var_name, time_dim):
    da = ds[var_name]
    step_type = str(da.attrs.get("GRIB_stepType", "")).lower()
    if step_type != "accum":
        raise ValueError(
            f"{var_name!r} must be an accumulated ERA5-Land forecast field; "
            f"GRIB_stepType={step_type or 'missing'!r}"
        )

    return da.where(da[time_dim].dt.hour == 0, drop=True)


def build_daily_accumulation(current_ds, following_ds, year, month):
    current_time = find_time_dim(current_ds)
    following_time = find_time_dim(following_ds)
    current_var = find_variable(current_ds)
    following_var = find_variable(following_ds)
    if current_var != following_var:
        raise ValueError(
            f"Variable changed between consecutive files: "
            f"{current_var!r} versus {following_var!r}"
        )

    current_midnight = midnight_accumulations(
        current_ds, current_var, current_time
    )
    following_midnight = midnight_accumulations(
        following_ds, following_var, following_time
    )
    if following_time != current_time:
        following_midnight = following_midnight.rename({following_time: current_time})

    next_year, next_month_number = next_month(year, month)
    expected_endpoint = f"{next_year:04d}-{next_month_number:02d}-01T00:00:00"
    next_endpoint = following_midnight.sel({current_time: [expected_endpoint]})

    endpoints = xr.concat(
        [current_midnight, next_endpoint],
        dim=current_time,
        coords="minimal",
        compat="override",
        join="exact",
        combine_attrs="override",
    )

    # In the standard ERA5-Land forecast convention, D+1 00:00 stores the
    # completed accumulation for day D. Shift endpoint timestamps back to D.
    daily_da = endpoints.assign_coords(
        {current_time: endpoints[current_time] - np.timedelta64(1, "D")}
    )
    daily_da = daily_da.where(
        (daily_da[current_time].dt.year == year)
        & (daily_da[current_time].dt.month == month),
        drop=True,
    )
    daily_da = daily_da.sortby(current_time)

    expected_days = calendar.monthrange(year, month)[1]
    if daily_da.sizes[current_time] != expected_days:
        raise ValueError(
            f"Expected {expected_days} daily endpoints for {year}-{month:02d}, "
            f"found {daily_da.sizes[current_time]}"
        )

    source_attrs = dict(current_ds[current_var].attrs)
    if convert_to_mm:
        daily_da = daily_da * 1000
        source_attrs["units"] = "mm"
    daily_da.attrs.update(source_attrs)
    daily_da.attrs["daily_aggregation"] = DAILY_AGGREGATION
    return daily_da, current_var


def process_month(job):
    (year, month), f_path, following_path = job
    file_name = os.path.basename(f_path)
    save_name = build_output_name(file_name)
    save_path = os.path.join(output_folder, save_name)
    temp_save_path = f"{save_path}.tmp.{os.getpid()}"

    try:
        if os.path.exists(save_path):
            if output_is_corrected(save_path):
                return f"SKIP already corrected: {save_name}"
            if not overwrite_existing:
                return f"SKIP existing unverified output: {save_name}"

        with tempfile.TemporaryDirectory(prefix="pre_unzip_current_") as current_temp, \
             tempfile.TemporaryDirectory(prefix="pre_unzip_next_") as following_temp:
            current_data_path = f_path
            if is_zip_file(f_path):
                current_data_path = extract_first_netcdf(f_path, current_temp)

            following_data_path = following_path
            if is_zip_file(following_path):
                following_data_path = extract_first_netcdf(
                    following_path, following_temp
                )

            with open_dataset_safely(current_data_path) as current_raw, \
                 open_dataset_safely(following_data_path) as following_raw:
                current_ds = prepare_dataset(current_raw)
                following_ds = prepare_dataset(following_raw)
                daily_da, var_name = build_daily_accumulation(
                    current_ds, following_ds, year, month
                )

                daily_ds = daily_da.to_dataset(name=var_name)
                daily_ds.attrs.update(current_ds.attrs)

                daily_ds.to_netcdf(
                    temp_save_path,
                    encoding={var_name: {"zlib": True, "complevel": 5}},
                )
                os.replace(temp_save_path, save_path)

        return f"DONE: {save_name}"

    except Exception as exc:
        return f"FAILED: {file_name} -> {exc}"

    finally:
        if os.path.exists(temp_save_path):
            os.remove(temp_save_path)


def main():
    os.makedirs(output_folder, exist_ok=True)

    nc_files = sorted(glob.glob(os.path.join(input_folder, "*.nc")))
    zip_files = sorted(glob.glob(os.path.join(input_folder, "*.zip")))
    input_files = sorted(set(nc_files + zip_files))
    files_by_month = {}
    for path in input_files:
        key = parse_file_month(path)
        if key in files_by_month:
            raise ValueError(f"Multiple input files found for {key}: {path}")
        files_by_month[key] = path

    jobs = []
    missing_following = []
    for key, path in sorted(files_by_month.items()):
        following_key = next_month(*key)
        following_path = files_by_month.get(following_key)
        if following_path is None:
            missing_following.append((key, path))
            continue
        jobs.append((key, path, following_path))

    total_files = len(input_files)
    total_jobs = len(jobs)

    print(
        f"Found {total_files} files; {total_jobs} complete output months can be "
        f"built with {num_workers} workers."
    )
    for (year, month), path in missing_following:
        save_path = os.path.join(
            output_folder, build_output_name(os.path.basename(path))
        )
        stale_message = ""
        if (
            overwrite_existing
            and os.path.exists(save_path)
            and not output_is_corrected(save_path)
        ):
            invalid_path = f"{save_path}.invalid_hourly_sum"
            suffix = 1
            while os.path.exists(invalid_path):
                invalid_path = f"{save_path}.invalid_hourly_sum.{suffix}"
                suffix += 1
            os.replace(save_path, invalid_path)
            stale_message = (
                f" Legacy output moved to {os.path.basename(invalid_path)}."
            )
        print(
            f"SKIP {os.path.basename(path)}: next-month 00:00 endpoint is "
            f"unavailable for {year}-{month:02d}.{stale_message}"
        )
    print("-" * 50)

    start_time = time.time()

    with ProcessPoolExecutor(max_workers=num_workers) as executor:
        results = executor.map(process_month, jobs)
        for idx, result in enumerate(results, start=1):
            print(f"[{idx}/{total_jobs}] {result}")

    duration = time.time() - start_time
    print("-" * 50)
    print(f"All done. Total time: {duration:.2f} seconds")


if __name__ == "__main__":
    main()
