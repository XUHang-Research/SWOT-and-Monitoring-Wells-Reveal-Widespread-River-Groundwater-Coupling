#!/usr/bin/env python3
"""Build monthly GW-SW tables with complete-calendar-month ERA5-Land climate.

The Phase-1 daily pair table contains rows only on dates where either a well or
SWOT has an observation.  Therefore precipitation and evapotranspiration must
not be averaged from those rows.  This module instead sums every corrected
daily ERA5-Land accumulation in the calendar month at each well location.
"""

from __future__ import annotations

from importlib import import_module as _load_module
C = _load_module("15_pipeline_config").from_environment()


import calendar
import json
from pathlib import Path

import numpy as np
import pandas as pd
import xarray as xr


PRE_FOLDER = C.precipitation_daily
ET_FOLDER = C.evapotranspiration_daily
PRE_TEMPLATE = "ERA5_Land_Precip_Daily_{year:04d}_{month:02d}.nc"
ET_TEMPLATE = "ERA5_Land_ET_Daily_{year:04d}_{month:02d}.nc"
EXPECTED_DAILY_AGGREGATION = (
    "ERA5-Land accumulated forecast value at next-day 00:00 UTC; "
    "timestamp shifted back one day"
)
STUDY_START = pd.Timestamp("2023-01-01")
STUDY_END = pd.Timestamp("2025-12-31")
STUDY_MONTHS = pd.period_range(STUDY_START, STUDY_END, freq="M")
COASTAL_FALLBACK_GRID_RADIUS = 3
COASTAL_FALLBACK_MAX_DISTANCE_DEGREES = 0.25

GROUP_KEYS = ["site_no", "node_id", "year_month"]
COUNT_COLUMNS = [
    "river_wse_obs_days",
    "groundwater_obs_days",
    "paired_obs_days",
    "precipitation_days",
    "evapotranspiration_days",
]


def _normalize_longitudes(values: np.ndarray, longitude_index: pd.Index) -> np.ndarray:
    values = np.asarray(values, dtype=float)
    if float(longitude_index.min()) >= 0:
        return values % 360.0
    return ((values + 180.0) % 360.0) - 180.0


def _nearest_indices(index: pd.Index, values: np.ndarray, coordinate: str) -> np.ndarray:
    values = np.asarray(values, dtype=float)
    if not np.isfinite(values).all():
        raise ValueError(f"{coordinate} contains missing or non-finite station coordinates")
    indices = index.get_indexer(values, method="nearest")
    if np.any(indices < 0):
        raise ValueError(f"Could not match every station {coordinate} to the ERA5-Land grid")
    matched = np.asarray(index, dtype=float)[indices]
    difference = np.abs(matched - values)
    if coordinate == "longitude":
        difference = np.minimum(difference, 360.0 - difference)
    grid = np.sort(np.unique(np.asarray(index, dtype=float)))
    spacing = float(np.median(np.diff(grid))) if len(grid) > 1 else 0.0
    tolerance = spacing / 2.0 + 1e-8
    if np.any(difference > tolerance):
        raise ValueError(
            f"Nearest ERA5-Land {coordinate} exceeds half-grid tolerance: "
            f"maximum difference={difference.max():.6f}, tolerance={tolerance:.6f}"
        )
    return indices


def _fill_coastal_masked_points(
    monthly_total: np.ndarray,
    point_total: np.ndarray,
    lat_index: np.ndarray,
    lon_index: np.ndarray,
    station_lon: np.ndarray,
    station_lat: np.ndarray,
    longitude: pd.Index,
    latitude: pd.Index,
) -> tuple[np.ndarray, dict]:
    """Replace masked coastal cells with the nearest valid ERA5-Land cell.

    A valid coastal well can round to an ocean-masked 0.1-degree cell.  Search
    a tightly bounded neighbourhood and fail loudly instead of retaining a
    silent NaN.  Distance is locally approximated in degrees with longitude
    scaled by cos(latitude), which is sufficient for this sub-grid search.
    """
    result = np.asarray(point_total, dtype=float).copy()
    missing_positions = np.flatnonzero(~np.isfinite(result))
    if not len(missing_positions):
        return result, {
            "coastal_land_fallback_sites": 0,
            "max_fallback_distance_degrees": 0.0,
        }

    lon_values = np.asarray(longitude, dtype=float)
    lat_values = np.asarray(latitude, dtype=float)
    selected_distances: list[float] = []
    unresolved: list[int] = []

    for position in missing_positions:
        center_y = int(lat_index[position])
        center_x = int(lon_index[position])
        best: tuple[float, int, int] | None = None

        for delta_y in range(-COASTAL_FALLBACK_GRID_RADIUS, COASTAL_FALLBACK_GRID_RADIUS + 1):
            candidate_y = center_y + delta_y
            if not 0 <= candidate_y < len(lat_values):
                continue
            for delta_x in range(-COASTAL_FALLBACK_GRID_RADIUS, COASTAL_FALLBACK_GRID_RADIUS + 1):
                candidate_x = (center_x + delta_x) % len(lon_values)
                if not np.isfinite(monthly_total[candidate_y, candidate_x]):
                    continue
                longitude_difference = abs(lon_values[candidate_x] - station_lon[position])
                longitude_difference = min(longitude_difference, 360.0 - longitude_difference)
                distance = float(np.hypot(
                    lat_values[candidate_y] - station_lat[position],
                    longitude_difference * np.cos(np.deg2rad(station_lat[position])),
                ))
                candidate = (distance, candidate_y, candidate_x)
                if best is None or candidate < best:
                    best = candidate

        if best is None or best[0] > COASTAL_FALLBACK_MAX_DISTANCE_DEGREES:
            unresolved.append(int(position))
            continue
        distance, selected_y, selected_x = best
        result[position] = monthly_total[selected_y, selected_x]
        selected_distances.append(distance)

    if unresolved:
        sample = [
            {
                "longitude": float(station_lon[position]),
                "latitude": float(station_lat[position]),
            }
            for position in unresolved[:10]
        ]
        raise ValueError(
            "No valid ERA5-Land cell within the bounded coastal fallback for "
            f"{len(unresolved)} station points; sample={sample}"
        )
    if not np.isfinite(result).all():
        raise ValueError("Coastal ERA5-Land fallback unexpectedly retained non-finite values")

    return result, {
        "coastal_land_fallback_sites": int(len(selected_distances)),
        "max_fallback_distance_degrees": float(max(selected_distances, default=0.0)),
        "coastal_fallback_rule": (
            f"nearest finite cell within {COASTAL_FALLBACK_GRID_RADIUS} grid cells "
            f"and {COASTAL_FALLBACK_MAX_DISTANCE_DEGREES} approximate degrees"
        ),
    }


def _complete_month_values(
    path: Path,
    variable: str,
    year: int,
    month: int,
    station_lon: np.ndarray,
    station_lat: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, dict]:
    """Return complete-month totals and valid-day counts at station points."""
    n_station = len(station_lon)
    if not path.is_file():
        return (
            np.full(n_station, np.nan),
            np.zeros(n_station, dtype=np.int16),
            {"status": "missing", "path": str(path)},
        )

    with xr.open_dataset(path) as dataset:
        if variable not in dataset.data_vars:
            raise KeyError(f"{path} does not contain required variable {variable!r}")
        data = dataset[variable]
        if data.attrs.get("daily_aggregation") != EXPECTED_DAILY_AGGREGATION:
            raise ValueError(f"{path} was not produced by the corrected daily accumulation method")
        if str(data.attrs.get("units", "")).lower() != "mm":
            raise ValueError(f"{path} has units={data.attrs.get('units')!r}, expected 'mm'")

        time_name = "valid_time" if "valid_time" in dataset.coords else "time"
        dates = pd.DatetimeIndex(dataset[time_name].values).normalize()
        expected_dates = pd.date_range(
            f"{year:04d}-{month:02d}-01",
            periods=calendar.monthrange(year, month)[1],
            freq="D",
        )
        if not dates.equals(expected_dates):
            raise ValueError(
                f"{path} is not a complete calendar month: "
                f"found {len(dates)} dates from {dates.min()} to {dates.max()}, "
                f"expected {len(expected_dates)}"
            )

        longitude = dataset.indexes["longitude"]
        latitude = dataset.indexes["latitude"]
        normalized_lon = _normalize_longitudes(station_lon, longitude)
        lon_index = _nearest_indices(longitude, normalized_lon, "longitude")
        lat_index = _nearest_indices(latitude, station_lat, "latitude")

        # Summing the complete daily cube first gives a small 2-D monthly field
        # and avoids slow random NetCDF reads for thousands of station points.
        monthly_total = (
            data.transpose(time_name, "latitude", "longitude")
            .sum(dim=time_name, skipna=False)
            .load()
            .values
        )
        point_total = np.asarray(monthly_total[lat_index, lon_index], dtype=float)
        point_total, fallback_diagnostics = _fill_coastal_masked_points(
            monthly_total=monthly_total,
            point_total=point_total,
            lat_index=lat_index,
            lon_index=lon_index,
            station_lon=normalized_lon,
            station_lat=np.asarray(station_lat, dtype=float),
            longitude=longitude,
            latitude=latitude,
        )
        point_days = np.full(n_station, len(expected_dates), dtype=np.int16)
        diagnostics = {
            "status": "complete",
            "path": str(path),
            "days": len(expected_dates),
            "units": "mm/month",
            "aggregation": "sum of every corrected daily accumulation in the calendar month",
            **fallback_diagnostics,
        }
        return point_total, point_days, diagnostics


def _build_station_climate(monthly: pd.DataFrame) -> tuple[pd.DataFrame, list[dict]]:
    station_month = monthly[["site_no", "year_month", "wells_lon", "wells_lat"]].copy()
    coordinate_counts = station_month.groupby(
        ["site_no", "year_month"], dropna=False
    )[["wells_lon", "wells_lat"]].nunique(dropna=False)
    if (coordinate_counts > 1).any(axis=None):
        raise ValueError("A site-month has more than one coordinate pair")
    station_month = station_month.drop_duplicates(["site_no", "year_month"])

    station_month["precipitation_mm"] = np.nan
    station_month["evapotranspiration_mm"] = np.nan
    station_month["precipitation_days"] = 0
    station_month["evapotranspiration_days"] = 0
    diagnostics: list[dict] = []

    for period, index in station_month.groupby("year_month", sort=True).groups.items():
        year, month = int(period.year), int(period.month)
        row_index = np.asarray(list(index), dtype=int)
        subset = station_month.loc[row_index]
        lon = pd.to_numeric(subset["wells_lon"], errors="coerce").to_numpy(float)
        lat = pd.to_numeric(subset["wells_lat"], errors="coerce").to_numpy(float)

        pre, pre_days, pre_diag = _complete_month_values(
            PRE_FOLDER / PRE_TEMPLATE.format(year=year, month=month),
            "tp", year, month, lon, lat,
        )
        et, et_days, et_diag = _complete_month_values(
            ET_FOLDER / ET_TEMPLATE.format(year=year, month=month),
            "e", year, month, lon, lat,
        )
        station_month.loc[row_index, "precipitation_mm"] = pre
        station_month.loc[row_index, "evapotranspiration_mm"] = et
        station_month.loc[row_index, "precipitation_days"] = pre_days
        station_month.loc[row_index, "evapotranspiration_days"] = et_days
        diagnostics.append({
            "year_month": str(period),
            "n_sites": int(len(row_index)),
            "precipitation": pre_diag,
            "evapotranspiration": et_diag,
        })
        print(
            f"  climate {period}: sites={len(row_index):,}; "
            f"PRE={pre_diag['status']}; ET={et_diag['status']}",
            flush=True,
        )
    return station_month, diagnostics


def _station_climate_with_reference(
    monthly: pd.DataFrame,
    output_file: Path,
) -> tuple[pd.DataFrame, list[dict]]:
    """Reuse the freshly built 5-km station-month climate for nested subsets."""
    if output_file.parent.name == "distance5000":
        return _build_station_climate(monthly)

    reference = output_file.parent.parent / "distance5000" / "6matched_results_monthly.txt"
    reference_qc = reference.with_suffix(".climate_qc.json")
    required = {
        "site_no", "date", "precipitation_mm", "evapotranspiration_mm",
        "precipitation_days", "evapotranspiration_days",
    }
    if not reference.is_file() or not reference_qc.is_file():
        print("  fresh 5-km climate reference unavailable; reading daily NetCDF directly", flush=True)
        return _build_station_climate(monthly)
    qc = json.loads(reference_qc.read_text(encoding="utf-8"))
    definition = qc.get("climate_definition", {})
    if "complete-calendar-month sum" not in definition.get("precipitation_mm", ""):
        print("  5-km climate reference is not a verified complete-month product", flush=True)
        return _build_station_climate(monthly)

    header = set(pd.read_csv(reference, sep="\t", nrows=0).columns)
    if not required.issubset(header):
        print("  5-km climate reference lacks required climate/count columns", flush=True)
        return _build_station_climate(monthly)

    ref = pd.read_csv(reference, sep="\t", usecols=sorted(required), low_memory=False)
    ref["date"] = pd.to_datetime(ref["date"], errors="coerce")
    ref["year_month"] = ref["date"].dt.to_period("M")
    ref = ref.drop(columns="date").drop_duplicates(["site_no", "year_month"])
    if ref.duplicated(["site_no", "year_month"]).any():
        raise ValueError("5-km climate reference contains duplicate site-month keys")

    target = monthly[["site_no", "year_month", "wells_lon", "wells_lat"]].drop_duplicates(
        ["site_no", "year_month"]
    )
    joined = target.merge(
        ref,
        on=["site_no", "year_month"],
        how="left",
        indicator=True,
        validate="one_to_one",
    )
    missing = joined["_merge"].ne("both")
    diagnostics: list[dict] = [{
        "status": "reused_5km_station_month_climate",
        "reference": str(reference),
        "rows_reused": int((~missing).sum()),
        "rows_missing_from_reference": int(missing.sum()),
    }]
    reused = joined.loc[~missing].drop(columns="_merge")
    if missing.any():
        computed, computed_diagnostics = _build_station_climate(
            joined.loc[missing, ["site_no", "year_month", "wells_lon", "wells_lat"]]
        )
        reused = pd.concat([reused, computed], ignore_index=True, sort=False)
        diagnostics.extend(computed_diagnostics)
    print(
        f"  reused complete-month climate from 5 km: {(~missing).sum():,}/{len(joined):,} site-months",
        flush=True,
    )
    return reused, diagnostics


def build_monthly_table(input_file: Path | str, output_file: Path | str) -> dict:
    input_file = Path(input_file)
    output_file = Path(output_file)
    if not input_file.is_file():
        raise FileNotFoundError(input_file)

    print(f"Reading the final daily paired table: {input_file}", flush=True)
    daily = pd.read_csv(input_file, sep="\t", low_memory=False)
    required = {
        "site_no", "node_id", "date", "wells_lon", "wells_lat",
        "river_wse", "Value_EGM2008_m", "number_match_SG",
        "precipitation_mm", "evapotranspiration_mm",
        "slope2", "p_wid_var", "d_x_area",
    }
    missing = sorted(required.difference(daily.columns))
    if missing:
        raise KeyError(f"Daily input is missing required columns: {missing}")

    daily["number_match_SG"] = pd.to_numeric(daily["number_match_SG"], errors="coerce")
    daily = daily.loc[daily["number_match_SG"].eq(1)].copy()
    daily["date"] = pd.to_datetime(daily["date"], errors="coerce")
    if daily["date"].isna().any():
        raise ValueError(f"Daily input contains {int(daily['date'].isna().sum()):,} invalid dates")
    daily = daily.loc[daily["date"].between(STUDY_START, STUDY_END)].copy()
    if daily.empty:
        raise ValueError(
            f"Daily input has no records in study period {STUDY_START.date()} to {STUDY_END.date()}"
        )
    duplicate_keys = ["site_no", "node_id", "date"]
    if daily.duplicated(duplicate_keys).any():
        raise ValueError(
            f"Daily input contains {int(daily.duplicated(duplicate_keys).sum()):,} "
            "duplicate site-node-date rows"
        )

    numeric_mean_columns = {
        "river_wse", "estimated_wse", "Value_EGM2008_m", "GWL_Value_m",
        "Depth to Water Below Land Surface in meter", "slope2", "p_wid_var", "d_x_area",
    }
    for column in numeric_mean_columns:
        if column in daily:
            daily[column] = pd.to_numeric(daily[column], errors="coerce")
    daily["year_month"] = daily["date"].dt.to_period("M")
    daily["_river_valid"] = daily["river_wse"].notna().astype(np.int16)
    daily["_groundwater_valid"] = daily["Value_EGM2008_m"].notna().astype(np.int16)
    daily["_paired_valid"] = (
        daily["river_wse"].notna() & daily["Value_EGM2008_m"].notna()
    ).astype(np.int16)

    excluded = {
        *GROUP_KEYS, "date", "precipitation_mm", "evapotranspiration_mm",
        "_river_valid", "_groundwater_valid", "_paired_valid",
    }
    aggregation = {
        column: ("mean" if column in numeric_mean_columns else "first")
        for column in daily.columns
        if column not in excluded
    }
    monthly = daily.groupby(GROUP_KEYS, as_index=False, dropna=False).agg(aggregation)
    counts = daily.groupby(GROUP_KEYS, as_index=False, dropna=False).agg(
        river_wse_obs_days=("_river_valid", "sum"),
        groundwater_obs_days=("_groundwater_valid", "sum"),
        paired_obs_days=("_paired_valid", "sum"),
    )
    monthly = monthly.merge(counts, on=GROUP_KEYS, how="left", validate="one_to_one")

    # Climate descriptors such as P/ET must describe the fixed study window,
    # not only months in which a well or SWOT happened to be observed.  Build a
    # complete 36-month calendar for every selected Top-1 site before extracting
    # ERA5-Land.  Observation fields remain missing in newly inserted months and
    # their observation-day counts are explicitly zero.
    site_nodes = daily[["site_no", "node_id"]].drop_duplicates()
    if site_nodes.duplicated("site_no").any():
        sample = site_nodes.loc[
            site_nodes.duplicated("site_no", keep=False), "site_no"
        ].astype(str).unique()[:10].tolist()
        raise ValueError(
            "Top-1 monthly construction requires one node_id per site_no; "
            f"sample sites with multiple nodes={sample}"
        )

    # Carry only genuinely static pair/site fields into climate-only months.
    # This keeps every calendar row traceable while avoiding fabrication of
    # river or groundwater observations.
    static_candidates = [
        "wells_lon", "wells_lat", "shp_file", "reach_id",
        "river_p_lon", "river_p_lat", "river_p_width",
        "distance_adjusted_m", "distance_m", "distance_prior_bank_m",
        "distance_prior_center_m", "slope2", "p_wid_var", "d_x_area",
        "reach_source_count", "reach_valid_slope2_count",
        "reach_valid_p_wid_var_count", "reach_valid_d_x_area_count",
        "reach_attribute_matched", "Well_Elevation_EGM2008",
        "AquiferType", "Well_Elevation", "number_match_SG",
    ]
    static_columns = [column for column in static_candidates if column in daily]
    site_static = (
        daily.groupby(["site_no", "node_id"], as_index=False, dropna=False)
        [static_columns].first()
    )
    calendar = pd.DataFrame({"year_month": STUDY_MONTHS})
    full_calendar = site_static.merge(calendar, how="cross")
    monthly = full_calendar.merge(
        monthly,
        on=GROUP_KEYS,
        how="left",
        suffixes=("_calendar", ""),
        validate="one_to_one",
    )
    for column in static_columns:
        calendar_column = f"{column}_calendar"
        if calendar_column not in monthly:
            continue
        if column in monthly:
            monthly[column] = monthly[column].combine_first(monthly[calendar_column])
        else:
            monthly[column] = monthly[calendar_column]
        monthly = monthly.drop(columns=calendar_column)
    for column in ("river_wse_obs_days", "groundwater_obs_days", "paired_obs_days"):
        monthly[column] = monthly[column].fillna(0)

    expected_rows = len(site_nodes) * len(STUDY_MONTHS)
    if len(monthly) != expected_rows or monthly.duplicated(GROUP_KEYS).any():
        raise ValueError(
            "Failed to construct the complete site-month calendar: "
            f"rows={len(monthly):,}, expected={expected_rows:,}"
        )

    station_climate, diagnostics = _station_climate_with_reference(monthly, output_file)
    monthly = monthly.merge(
        station_climate[[
            "site_no", "year_month", "precipitation_mm", "evapotranspiration_mm",
            "precipitation_days", "evapotranspiration_days",
        ]],
        on=["site_no", "year_month"],
        how="left",
        validate="many_to_one",
    )
    monthly["date"] = monthly["year_month"].dt.to_timestamp()
    monthly = monthly.drop(columns="year_month")
    monthly = monthly.sort_values(["site_no", "node_id", "date"]).reset_index(drop=True)
    for column in COUNT_COLUMNS:
        monthly[column] = pd.to_numeric(monthly[column], errors="raise").astype("int16")

    temporary = output_file.with_suffix(output_file.suffix + ".tmp")
    monthly.to_csv(temporary, sep="\t", index=False)
    temporary.replace(output_file)

    qc = {
        "input_file": str(input_file),
        "output_file": str(output_file),
        "monthly_rows": int(len(monthly)),
        "sites": int(monthly["site_no"].nunique()),
        "date_min": str(monthly["date"].min().date()),
        "date_max": str(monthly["date"].max().date()),
        "study_period": {
            "start": str(STUDY_START.date()),
            "end": str(STUDY_END.date()),
            "note": "2026-01 hourly climate is used only as the next-day endpoint for 2025-12-31",
        },
        "climate_definition": {
            "precipitation_mm": "complete-calendar-month sum of corrected ERA5-Land daily tp, mm/month",
            "evapotranspiration_mm": "complete-calendar-month sum of positive-loss ERA5-Land daily e, mm/month",
            "site_window": (
                "fixed 2023-01 through 2025-12 natural-month calendar for "
                "every selected site, independent of river/GW observation availability"
            ),
        },
        "calendar_coverage": {
            "expected_months_per_site": int(len(STUDY_MONTHS)),
            "minimum_months_per_site": int(monthly.groupby("site_no")["date"].nunique().min()),
            "maximum_months_per_site": int(monthly.groupby("site_no")["date"].nunique().max()),
            "sites_with_complete_window": int(
                monthly.groupby("site_no")["date"].nunique().eq(len(STUDY_MONTHS)).sum()
            ),
            "duplicate_site_month_rows": int(monthly.duplicated(["site_no", "date"]).sum()),
        },
        "count_columns": {
            "river_wse_obs_days": "unique daily rows with finite river_wse",
            "groundwater_obs_days": "unique daily rows with finite Value_EGM2008_m",
            "paired_obs_days": "unique daily rows with both finite river_wse and Value_EGM2008_m",
            "precipitation_days": "calendar days represented in precipitation_mm",
            "evapotranspiration_days": "calendar days represented in evapotranspiration_mm",
        },
        "months": diagnostics,
    }
    qc_path = output_file.with_suffix(".climate_qc.json")
    qc_path.write_text(json.dumps(qc, indent=2, ensure_ascii=False), encoding="utf-8")
    print(
        f"Monthly table complete: {len(monthly):,} rows, {monthly['site_no'].nunique():,} sites; "
        f"output {output_file}",
        flush=True,
    )
    return qc


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("input_file", type=Path)
    parser.add_argument("output_file", type=Path)
    args = parser.parse_args()
    build_monthly_table(args.input_file, args.output_file)
