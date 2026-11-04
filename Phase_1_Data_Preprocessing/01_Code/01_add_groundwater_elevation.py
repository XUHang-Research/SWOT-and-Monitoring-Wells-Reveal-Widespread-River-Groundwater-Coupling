#!/usr/bin/env python3
"""Fill missing EGM2008 well elevations from Copernicus DEM GLO-30.

The Copernicus DEM tile metadata identifies its vertical CRS as the EGM2008
geoid. Existing well elevations and groundwater fields are never overwritten.
Missing depth/water counterparts are completed independently of elevation filling.
Reference-point elevations and unconverted source datums are handled explicitly.
The table is processed in two streaming passes so memory use stays bounded.
"""

from __future__ import annotations

from importlib import import_module as _load_module
C = _load_module("15_pipeline_config").from_environment()


import json
import math
import os
from collections import defaultdict
from pathlib import Path

import numpy as np
import pandas as pd
import rasterio


INPUT_FILE = C.inputs["groundwater_raw"]
METADATA_FILE = C.inputs["well_metadata"]
DEM_FILE = C.inputs["dem_vrt"]
OUTPUT_FILE = C.groundwater_dem
QC_FILE = C.groundwater_dem.with_suffix(".qc.json")

CHUNK_SIZE = 500_000
SITE_COLUMN = "site_no"
LON_COLUMN = "wells_lon"
LAT_COLUMN = "wells_lat"
ELEVATION_COLUMN = "Well_Elevation_EGM2008"
ELEVATION_STATUS_COLUMN = "Elevation_Record_Status"
DEPTH_COLUMN = "Depth to Water Below Land Surface in meter"
WATER_LEVEL_COLUMN = "Value_EGM2008_m"
FILL_FLAG_COLUMN = "DEM_Fill_Flag"
GROUND_BASIS_COLUMN = "Complement_Ground_Elevation_m"
GROUND_DATUM_COLUMN = "Complement_Ground_Datum"
COMPLEMENT_STATUS_COLUMN = "Complement_Status"

# Copernicus DEM GLO-30 metadata evidence from a representative local tile.
DEM_VERTICAL_DATUM = "EGM2008 geoid (EPSG:3855)"
DEM_VALID_MIN_M = -500.0
DEM_VALID_MAX_M = 9000.0
FORMULA_TOLERANCE_M = 0.05


def numeric(series: pd.Series) -> pd.Series:
    return pd.to_numeric(series, errors="coerce")


def load_elevation_policies() -> dict[str, str]:
    """Identify source elevations explicitly unsuitable as surveyed ground."""
    metadata = pd.read_excel(METADATA_FILE, dtype={SITE_COLUMN: "string"})
    if metadata[SITE_COLUMN].duplicated().any():
        raise ValueError("Metadata must contain one unambiguous record per site")
    policies = {}
    for _, row in metadata.iterrows():
        source = str(row.get("Well_Elevation_Source", "")).lower()
        method = str(row.get("Well_Elevation_Method", "")).lower()
        confidence = str(row.get("Well_Elevation_Confidence", "")).upper()
        if "reference point" in source or "reference elevation" in source:
            policy = "REFERENCE_POINT"
        elif "proxy" in method or confidence == "LOW":
            policy = "LOW_CONFIDENCE_PROXY"
        elif "unresolved" in method:
            policy = "UNRESOLVED_GROUND"
        else:
            policy = "SOURCE_GROUND"
        policies[str(row[SITE_COLUMN])] = policy
    return policies


def complete_counterparts(chunk, dem_elevation, policies):
    """Preserve observations; derive only with a usable, compatible ground basis.

    Copernicus surface heights are estimates, not surveyed bare-earth heights.
    They may replace a reference-point height ONLY as the derivation basis;
    the source Well_Elevation columns remain untouched.
    """
    index = chunk.index
    def column(name):
        return numeric(chunk[name]) if name in chunk else pd.Series(np.nan, index=index)

    elevation = column(ELEVATION_COLUMN)
    depth = column(DEPTH_COLUMN)
    water = column(WATER_LEVEL_COLUMN)
    finite_depth, finite_water = np.isfinite(depth), np.isfinite(water)
    need_depth, need_water = ~finite_depth & finite_water, ~finite_water & finite_depth
    policy = chunk[SITE_COLUMN].map(policies).fillna("SOURCE_GROUND")
    trusted_ground = policy.eq("SOURCE_GROUND")
    dem = chunk[SITE_COLUMN].map(dem_elevation).astype(float)
    source_status = chunk.get("Vertical_conversion_status", pd.Series("", index=index)).fillna("").astype(str)
    original_datum = source_status.str.startswith("SKIPPED")
    ground = elevation.where(trusted_ground & ~original_datum, dem)
    ground_datum = pd.Series("EGM2008", index=index, dtype="string")
    use_dem = ~trusted_ground | original_datum | chunk[FILL_FLAG_COLUMN].fillna("").str.contains("E", regex=False)
    status = pd.Series("", index=index, dtype="string")
    status.loc[~finite_depth & ~finite_water] = "BOTH_WATER_AND_DEPTH_MISSING"
    status.loc[need_depth | need_water] = "MISSING_USABLE_GROUND"

    # A pass-through EGM2008-named field must never be subtracted from EGM2008 DEM.
    # Original-datum subtraction is allowed only for a trusted source ground,
    # a preserved original water elevation and a verified pass-through value.
    original_ground, original_water = column("Well_Elevation"), column("GWL_Value_m")
    native_ok = (original_datum & trusted_ground & np.isfinite(original_ground)
                 & np.isfinite(original_water)
                 & np.isclose(water, original_water, atol=1e-8, rtol=0))
    native_depth = need_depth & original_datum
    ground.loc[native_depth] = original_ground.loc[native_depth].where(native_ok.loc[native_depth])
    source_datum = chunk.get("Elevation datum", pd.Series("UNKNOWN", index=index)).fillna("UNKNOWN").astype(str)
    ground_datum.loc[native_depth] = "SOURCE:" + source_datum.loc[native_depth]
    status.loc[native_depth] = "UNRESOLVED_SOURCE_DATUM_OR_GROUND"

    derive_depth = need_depth & np.isfinite(ground) & (~original_datum | native_ok)
    derive_water = need_water & np.isfinite(ground)
    chunk.loc[derive_depth, DEPTH_COLUMN] = ground.loc[derive_depth] - water.loc[derive_depth]
    chunk.loc[derive_water, WATER_LEVEL_COLUMN] = ground.loc[derive_water] - depth.loc[derive_water]
    status.loc[derive_depth] = "DEPTH_FROM_SOURCE_GROUND_WATER"
    status.loc[derive_depth & use_dem] = "DEPTH_FROM_COP30_WATER_ESTIMATE"
    status.loc[derive_depth & native_ok] = "DEPTH_FROM_ORIGINAL_DATUM_GROUND_WATER"
    status.loc[derive_water] = "WATER_FROM_SOURCE_GROUND_DEPTH"
    status.loc[derive_water & use_dem] = "WATER_FROM_COP30_DEPTH_ESTIMATE"
    chunk[FILL_FLAG_COLUMN] = update_flag(chunk[FILL_FLAG_COLUMN], derive_depth, "D")
    chunk[FILL_FLAG_COLUMN] = update_flag(chunk[FILL_FLAG_COLUMN], derive_water, "V")
    relevant = need_depth | need_water
    chunk[GROUND_BASIS_COLUMN] = ground.where(relevant)
    chunk[GROUND_DATUM_COLUMN] = ground_datum.where(relevant, "")
    chunk[COMPLEMENT_STATUS_COLUMN] = status
    return derive_depth, derive_water


def first_pass_site_inventory() -> tuple[dict[str, tuple[float, float]], dict[str, float], dict]:
    """Collect one coordinate and the existing EGM2008 elevation per site."""
    coordinates: dict[str, tuple[float, float]] = {}
    elevation_sum: defaultdict[str, float] = defaultdict(float)
    elevation_count: defaultdict[str, int] = defaultdict(int)
    elevation_min: dict[str, float] = {}
    elevation_max: dict[str, float] = {}
    row_count = 0
    needs_counterpart = set()
    needs_water = set()

    usecols = [SITE_COLUMN, LON_COLUMN, LAT_COLUMN, ELEVATION_COLUMN,
               DEPTH_COLUMN, WATER_LEVEL_COLUMN]
    for chunk_number, chunk in enumerate(
        pd.read_csv(
            INPUT_FILE,
            sep="\t",
            usecols=usecols,
            dtype={SITE_COLUMN: "string"},
            chunksize=CHUNK_SIZE,
            low_memory=False,
            encoding="utf-8-sig",
        ),
        start=1,
    ):
        row_count += len(chunk)
        chunk[LON_COLUMN] = numeric(chunk[LON_COLUMN])
        chunk[LAT_COLUMN] = numeric(chunk[LAT_COLUMN])
        chunk[ELEVATION_COLUMN] = numeric(chunk[ELEVATION_COLUMN])
        has_depth = np.isfinite(numeric(chunk[DEPTH_COLUMN]))
        has_water = np.isfinite(numeric(chunk[WATER_LEVEL_COLUMN]))
        needs_counterpart.update(chunk.loc[has_depth ^ has_water, SITE_COLUMN].dropna().astype(str))
        needs_water.update(chunk.loc[has_depth & ~has_water, SITE_COLUMN].dropna().astype(str))

        coord_rows = chunk.dropna(subset=[SITE_COLUMN, LON_COLUMN, LAT_COLUMN])
        coord_rows = coord_rows[
            coord_rows[LON_COLUMN].between(-180, 180)
            & coord_rows[LAT_COLUMN].between(-90, 90)
        ].drop_duplicates(SITE_COLUMN)
        for row in coord_rows.itertuples(index=False):
            site = str(getattr(row, SITE_COLUMN))
            if site not in coordinates:
                coordinates[site] = (
                    float(getattr(row, LON_COLUMN)),
                    float(getattr(row, LAT_COLUMN)),
                )

        valid_elevation = chunk.loc[
            chunk[SITE_COLUMN].notna() & np.isfinite(chunk[ELEVATION_COLUMN])
        ]
        if not valid_elevation.empty:
            grouped = valid_elevation.groupby(SITE_COLUMN, sort=False)[ELEVATION_COLUMN].agg(
                ["sum", "count", "min", "max"]
            )
            for site_value, values in grouped.iterrows():
                site = str(site_value)
                elevation_sum[site] += float(values["sum"])
                elevation_count[site] += int(values["count"])
                value_min = float(values["min"])
                value_max = float(values["max"])
                elevation_min[site] = min(elevation_min.get(site, value_min), value_min)
                elevation_max[site] = max(elevation_max.get(site, value_max), value_max)

        print(
            f"Pass 1 chunk {chunk_number}: {row_count:,} rows; "
            f"{len(coordinates):,} sites with coordinates; "
            f"{len(elevation_count):,} sites with existing EGM2008 elevation",
            flush=True,
        )

    existing_elevation = {
        site: elevation_sum[site] / elevation_count[site]
        for site in elevation_count
        if elevation_count[site] > 0
    }
    variable_sites = {
        site: elevation_max[site] - elevation_min[site]
        for site in elevation_count
        if elevation_max[site] - elevation_min[site] > 0.05
    }
    inventory = {
        "input_rows": row_count,
        "sites_needing_counterpart": sorted(needs_counterpart),
        "sites_needing_water": sorted(needs_water),
        "sites_with_coordinates": len(coordinates),
        "sites_with_existing_elevation": len(existing_elevation),
        "sites_without_existing_elevation": len(set(coordinates).difference(existing_elevation)),
        "sites_with_existing_elevation_range_gt_0p05m": len(variable_sites),
        "largest_existing_site_elevation_ranges_m": [
            {"site_no": site, "range_m": round(value, 6)}
            for site, value in sorted(variable_sites.items(), key=lambda item: item[1], reverse=True)[:20]
        ],
    }
    return coordinates, existing_elevation, inventory


def sample_dem(coordinates: dict[str, tuple[float, float]]) -> dict[str, float]:
    """Sample the VRT once per site, ordered geographically for faster tile IO."""
    ordered = sorted(
        coordinates.items(),
        key=lambda item: (math.floor(item[1][1]), math.floor(item[1][0]), item[0]),
    )
    sampled: dict[str, float] = {}
    with rasterio.open(DEM_FILE) as dataset:
        if dataset.crs is None or not dataset.crs.is_geographic:
            raise ValueError(f"Expected a geographic lon/lat DEM, found CRS={dataset.crs}")
        for start in range(0, len(ordered), 20_000):
            batch = ordered[start : start + 20_000]
            values = dataset.sample([xy for _, xy in batch], indexes=1, masked=True)
            for (site, _), raw in zip(batch, values):
                if np.ma.is_masked(raw) or len(raw) == 0:
                    continue
                value = float(raw[0])
                if np.isfinite(value) and DEM_VALID_MIN_M <= value <= DEM_VALID_MAX_M:
                    sampled[site] = value
            print(
                f"DEM sample: {min(start + len(batch), len(ordered)):,}/{len(ordered):,} sites; "
                f"valid={len(sampled):,}",
                flush=True,
            )
    return sampled


def update_flag(flag: pd.Series, mask: pd.Series, code: str) -> pd.Series:
    current = flag.fillna("").astype("string")
    current.loc[mask] = current.loc[mask].fillna("") + code
    return current


def second_pass_write(
    existing_elevation: dict[str, float], dem_elevation: dict[str, float],
    policies: dict[str, str],
) -> dict:
    """Fill fields without overwriting observations and atomically write output."""
    temporary = OUTPUT_FILE.with_suffix(OUTPUT_FILE.suffix + ".tmp")
    totals = defaultdict(int)
    depth_min = np.inf
    depth_max = -np.inf
    maximum_formula_residual = 0.0
    first_chunk = True

    for chunk_number, chunk in enumerate(
        pd.read_csv(
            INPUT_FILE,
            sep="\t",
            dtype={SITE_COLUMN: "string"},
            chunksize=CHUNK_SIZE,
            low_memory=False,
            encoding="utf-8-sig",
        ),
        start=1,
    ):
        totals["rows"] += len(chunk)
        for column in (ELEVATION_COLUMN, DEPTH_COLUMN, WATER_LEVEL_COLUMN):
            chunk[column] = numeric(chunk[column])
        if ELEVATION_STATUS_COLUMN not in chunk:
            chunk[ELEVATION_STATUS_COLUMN] = pd.Series(pd.NA, index=chunk.index, dtype="string")
        else:
            chunk[ELEVATION_STATUS_COLUMN] = chunk[ELEVATION_STATUS_COLUMN].astype("string")
        if FILL_FLAG_COLUMN not in chunk:
            chunk[FILL_FLAG_COLUMN] = pd.Series("", index=chunk.index, dtype="string")

        site_existing = chunk[SITE_COLUMN].map(existing_elevation)
        site_dem = chunk[SITE_COLUMN].map(dem_elevation)
        # NaN and +/-infinity are both invalid elevations and are eligible for
        # safe site propagation or DEM filling.
        elevation_missing = ~np.isfinite(chunk[ELEVATION_COLUMN])
        fill_from_site = elevation_missing & site_existing.notna()
        fill_from_dem = elevation_missing & ~fill_from_site & site_dem.notna()
        chunk.loc[fill_from_site, ELEVATION_COLUMN] = site_existing.loc[fill_from_site]
        chunk.loc[fill_from_dem, ELEVATION_COLUMN] = site_dem.loc[fill_from_dem]
        chunk.loc[fill_from_site, ELEVATION_STATUS_COLUMN] = "SITE_EGM2008_PROPAGATED"
        chunk.loc[fill_from_dem, ELEVATION_STATUS_COLUMN] = "DEM_COP30_EGM2008"
        chunk[FILL_FLAG_COLUMN] = update_flag(chunk[FILL_FLAG_COLUMN], fill_from_site, "S")
        chunk[FILL_FLAG_COLUMN] = update_flag(chunk[FILL_FLAG_COLUMN], fill_from_dem, "E")
        totals["elevation_rows_propagated_from_site"] += int(fill_from_site.sum())
        totals["elevation_rows_filled_from_dem"] += int(fill_from_dem.sum())

        can_fill_depth, can_fill_water_level = complete_counterparts(
            chunk, dem_elevation, policies)
        totals["depth_rows_derived"] += int(can_fill_depth.sum())
        totals["water_level_rows_derived"] += int(can_fill_water_level.sum())
        for label, count in chunk[COMPLEMENT_STATUS_COLUMN].value_counts().items():
            if label:
                totals[f"complement_status_{label}"] += int(count)

        depth = chunk[DEPTH_COLUMN]
        finite_depth = depth[np.isfinite(depth)]
        if not finite_depth.empty:
            depth_min = min(depth_min, float(finite_depth.min()))
            depth_max = max(depth_max, float(finite_depth.max()))
        totals["depth_finite_rows"] += int(finite_depth.size)
        totals["depth_negative_rows"] += int((finite_depth < 0).sum())
        totals["depth_below_minus_5m_rows"] += int((finite_depth < -5).sum())
        totals["depth_above_500m_rows"] += int((finite_depth > 500).sum())
        totals["depth_above_1000m_rows"] += int((finite_depth > 1000).sum())

        formula_ground = chunk[GROUND_BASIS_COLUMN].where(
            can_fill_depth | can_fill_water_level, chunk[ELEVATION_COLUMN])
        complete = (np.isfinite(formula_ground) & np.isfinite(chunk[WATER_LEVEL_COLUMN])
                    & np.isfinite(chunk[DEPTH_COLUMN]))
        if complete.any():
            residual = (
                formula_ground.loc[complete]
                - chunk.loc[complete, WATER_LEVEL_COLUMN]
                - chunk.loc[complete, DEPTH_COLUMN]
            ).abs()
            totals["formula_complete_rows"] += int(residual.size)
            totals["formula_residual_above_0p05m_rows"] += int(
                (residual > FORMULA_TOLERANCE_M).sum()
            )
            maximum_formula_residual = max(maximum_formula_residual, float(residual.max()))

        chunk.to_csv(
            temporary,
            sep="\t",
            index=False,
            mode="w" if first_chunk else "a",
            header=first_chunk,
            encoding="utf-8-sig",
        )
        first_chunk = False
        print(
            f"Pass 2 chunk {chunk_number}: {totals['rows']:,} rows written; "
            f"DEM elevation rows={totals['elevation_rows_filled_from_dem']:,}; "
            f"derived depth={totals['depth_rows_derived']:,}; "
            f"derived water level={totals['water_level_rows_derived']:,}",
            flush=True,
        )

    os.replace(temporary, OUTPUT_FILE)
    result = dict(totals)
    result["depth_min_m"] = None if not np.isfinite(depth_min) else depth_min
    result["depth_max_m"] = None if not np.isfinite(depth_max) else depth_max
    result["maximum_formula_residual_m"] = maximum_formula_residual
    result["output_size_bytes"] = OUTPUT_FILE.stat().st_size
    return result


def main() -> None:
    for required in (INPUT_FILE, METADATA_FILE, DEM_FILE):
        if not required.is_file():
            raise FileNotFoundError(required)

    print(f"Input:  {INPUT_FILE}", flush=True)
    print(f"DEM:    {DEM_FILE}", flush=True)
    print(f"Output: {OUTPUT_FILE}", flush=True)
    coordinates, existing_elevation, inventory = first_pass_site_inventory()
    policies = load_elevation_policies()
    missing_sites = set(coordinates).difference(existing_elevation)
    fallback_sites = {site for site in inventory["sites_needing_counterpart"]
                      if policies.get(site, "SOURCE_GROUND") != "SOURCE_GROUND"}
    # A missing water elevation can always be estimated from DEM and genuine
    # DBLS, even when the existing source elevation has not been converted.
    fallback_sites.update(inventory["sites_needing_water"])
    sample_sites = (missing_sites | fallback_sites).intersection(coordinates)
    dem_elevation = sample_dem({site: coordinates[site] for site in sample_sites})

    dem_fill_sites = missing_sites.intersection(dem_elevation)

    write_stats = second_pass_write(existing_elevation, dem_elevation, policies)
    report = {
        "method": "Preserve observations; fill missing elevation and independently complete depth/water using compatible ground basis",
        "dem_vertical_datum": DEM_VERTICAL_DATUM,
        "input_file": str(INPUT_FILE),
        "metadata_file": str(METADATA_FILE),
        "dem_file": str(DEM_FILE),
        "output_file": str(OUTPUT_FILE),
        "fill_flag_codes": {
            "S": "missing row elevation propagated from an existing EGM2008 elevation of the same site",
            "E": "missing site elevation filled from Copernicus DEM GLO-30",
            "D": "missing depth derived using Complement_Ground_Elevation_m; datum/method recorded",
            "V": "missing EGM2008 water derived using Complement_Ground_Elevation_m minus depth",
        },
        "inventory": inventory,
        "dem_sampling": {
            "sites_sampled_valid": len(dem_elevation),
            "sites_sampled_for_derivation_basis": len(fallback_sites.intersection(coordinates)),
            "sites_without_existing_elevation": len(missing_sites),
            "sites_fillable_from_dem": len(dem_fill_sites),
            "sites_still_missing_dem_or_coordinates": len(missing_sites.difference(dem_fill_sites)),
        },
        "existing_elevation_vs_dem_qc": {
            "performed": False,
            "reason": "Existing elevations preserved; reference/proxy elevations are not used as surveyed ground",
        },
        "row_level_output_qc": write_stats,
        "anomaly_interpretation": {
            "negative_depth": "flag for review; may be physically valid artesian head and was not deleted",
            "depth_above_500m": "flag for review; was not deleted",
            "formula_tolerance_m": FORMULA_TOLERANCE_M,
            "existing_elevations": "preserved and not compared against DEM in this fill-only run",
            "dem_depth": "Copernicus surface-height estimate, not surveyed bare-earth depth; source reference elevation is preserved separately",
            "unconverted_water": "Never subtract EGM2008 DEM from original-datum water. Unresolved ground/datum remains missing and is explicitly reported.",
        },
    }
    QC_FILE.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(report, ensure_ascii=False, indent=2), flush=True)
    print(f"QC: {QC_FILE}", flush=True)


if __name__ == "__main__":
    main()
