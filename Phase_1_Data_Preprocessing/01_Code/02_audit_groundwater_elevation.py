#!/usr/bin/env python3
"""Post-write audit for dataset_all_2023_DEM.txt, separated by fill provenance."""

from __future__ import annotations

from importlib import import_module as _load_module
C = _load_module("15_pipeline_config").from_environment()


import json
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np
import pandas as pd


DATA = C.groundwater_dem
RAW = C.inputs["groundwater_raw"]
QC = C.groundwater_dem.with_suffix(".qc.json")
CHUNK_SIZE = 500_000
SITE = "site_no"
DATE = "date"
ELEV = "Well_Elevation_EGM2008"
DEPTH = "Depth to Water Below Land Surface in meter"
WATER = "Value_EGM2008_m"
FLAG = "DEM_Fill_Flag"
BASIS = "Complement_Ground_Elevation_m"
DATUM = "Complement_Ground_Datum"
METHOD = "Complement_Status"
USECOLS = [SITE, DATE, ELEV, DEPTH, WATER, FLAG, BASIS, DATUM, METHOD,
           "Well_Elevation", "GWL_Value_m", "Vertical_conversion_status"]


def complement_violations(chunk):
    """Check recorded derivations without repeating the fill algorithm."""
    values = {c: pd.to_numeric(chunk[c], errors="coerce") for c in
              (ELEV, DEPTH, WATER, BASIS, "Well_Elevation", "GWL_Value_m")}
    flag = chunk[FLAG].fillna("")
    method = chunk[METHOD].fillna("")
    derived = flag.str.contains("D|V", regex=True)
    finite = {c: np.isfinite(v) for c, v in values.items()}
    residual = (values[BASIS] - values[WATER] - values[DEPTH]).abs()
    skipped = chunk["Vertical_conversion_status"].fillna("").str.startswith("SKIPPED")
    native = method.eq("DEPTH_FROM_ORIGINAL_DATUM_GROUND_WATER")
    result = {
        "derived_missing_finite_basis_or_counterpart": int((derived & ~(
            finite[BASIS] & finite[DEPTH] & finite[WATER])).sum()),
        "derived_formula_residual_gt_1e_8m": int((derived & residual.gt(1e-8)).sum()),
        "derivable_counterpart_still_missing": int((finite[BASIS] & (
            finite[DEPTH] ^ finite[WATER])).sum()),
        "unconverted_water_mixed_with_EGM2008_DEM": int((
            skipped & method.str.startswith("DEPTH_FROM_") & chunk[DATUM].eq("EGM2008")).sum()),
        "derived_method_flag_mismatch": int((
            flag.str.contains("D", regex=False).ne(method.str.startswith("DEPTH_FROM_"))
            | flag.str.contains("V", regex=False).ne(method.str.startswith("WATER_FROM_"))).sum()),
        "original_datum_basis_mismatch": int((native & ~(
            skipped & finite["Well_Elevation"] & finite["GWL_Value_m"]
            & np.isclose(values[BASIS], values["Well_Elevation"], atol=1e-8, rtol=0)
            & np.isclose(values[WATER], values["GWL_Value_m"], atol=1e-8, rtol=0))).sum()),
    }
    missing_one = finite[DEPTH] ^ finite[WATER]
    declared_unresolved = method.isin([
        "MISSING_USABLE_GROUND", "UNRESOLVED_SOURCE_DATUM_OR_GROUND"])
    result["missing_counterpart_without_explanation"] = int((missing_one & ~declared_unresolved).sum())
    return result


def update_top(frame, current, column, largest, n=20):
    if frame.empty:
        return current
    columns = list(dict.fromkeys([SITE, DATE, FLAG, ELEV, BASIS, DATUM, METHOD, WATER, DEPTH, column]))
    candidate = frame[columns].copy()
    candidate = candidate.replace([np.inf, -np.inf], np.nan).dropna(subset=[column])
    merged = pd.concat([current, candidate], ignore_index=True)
    if merged.empty:
        return merged
    return merged.nlargest(n, column) if largest else merged.nsmallest(n, column)


def main():
    totals = Counter()
    flags = Counter()
    by_country = defaultdict(Counter)
    top_low_depth = pd.DataFrame()
    top_high_depth = pd.DataFrame()
    top_residual = pd.DataFrame()
    violations = Counter()
    methods = Counter()
    raw_iterator = pd.read_csv(
        RAW, sep="\t", usecols=[SITE, DATE, ELEV, DEPTH, WATER],
        dtype={SITE: "string", DATE: "string"}, chunksize=CHUNK_SIZE,
        low_memory=False, encoding="utf-8-sig")

    for chunk_number, chunk in enumerate(
        pd.read_csv(
            DATA,
            sep="\t",
            usecols=USECOLS,
            dtype={SITE: "string", FLAG: "string", DATE: "string"},
            chunksize=CHUNK_SIZE,
            low_memory=False,
            encoding="utf-8-sig",
        ),
        start=1,
    ):
        raw = next(raw_iterator, None)
        if raw is None or len(raw) != len(chunk):
            raise ValueError("DEM output row count differs from raw input")
        if not raw[[SITE, DATE]].reset_index(drop=True).equals(chunk[[SITE, DATE]].reset_index(drop=True)):
            raise ValueError("DEM output changed raw site/date order")
        for column in (ELEV, DEPTH, WATER, BASIS):
            chunk[column] = pd.to_numeric(chunk[column], errors="coerce")
            if column != BASIS:
                source = pd.to_numeric(raw[column], errors="coerce")
                observed = np.isfinite(source)
                violations[f"existing_{column}_changed"] += int((observed & ~np.isclose(
                    source, chunk[column], rtol=0, atol=1e-8, equal_nan=True)).sum())
        violations.update(complement_violations(chunk))
        methods.update(chunk[METHOD].fillna("").value_counts().to_dict())
        flag = chunk[FLAG].fillna("")
        flags.update(flag.value_counts().to_dict())
        totals["rows"] += len(chunk)
        totals["missing_elevation_rows"] += int((~np.isfinite(chunk[ELEV])).sum())
        totals["missing_depth_rows"] += int(chunk[DEPTH].isna().sum())
        totals["missing_water_level_rows"] += int(chunk[WATER].isna().sum())
        for column, label in ((ELEV, "elevation"), (DEPTH, "depth"), (WATER, "water_level")):
            totals[f"nonfinite_{label}_rows"] += int(np.isinf(chunk[column]).sum())

        finite_depth = np.isfinite(chunk[DEPTH])
        negative = finite_depth & chunk[DEPTH].lt(0)
        below_minus5 = finite_depth & chunk[DEPTH].lt(-5)
        above500 = finite_depth & chunk[DEPTH].gt(500)
        above1000 = finite_depth & chunk[DEPTH].gt(1000)
        totals["depth_finite_rows"] += int(finite_depth.sum())
        totals["depth_negative_rows"] += int(negative.sum())
        totals["depth_below_minus_5m_rows"] += int(below_minus5.sum())
        totals["depth_above_500m_rows"] += int(above500.sum())
        totals["depth_above_1000m_rows"] += int(above1000.sum())

        dem_rows = flag.str.contains("E", regex=False)
        derived_depth = flag.str.contains("D", regex=False)
        derived_water = flag.str.contains("V", regex=False)
        for prefix, mask in (
            ("dem", dem_rows),
            ("derived_depth", derived_depth),
            ("derived_water", derived_water),
            ("original_or_unchanged", flag.eq("")),
        ):
            totals[f"{prefix}_rows"] += int(mask.sum())
            totals[f"{prefix}_negative_depth_rows"] += int((mask & negative).sum())
            totals[f"{prefix}_depth_below_minus_5m_rows"] += int((mask & below_minus5).sum())
            totals[f"{prefix}_depth_above_500m_rows"] += int((mask & above500).sum())
            totals[f"{prefix}_depth_above_1000m_rows"] += int((mask & above1000).sum())

        formula_ground = chunk[BASIS].where(derived_depth | derived_water, chunk[ELEV])
        values = np.column_stack([formula_ground, chunk[WATER], chunk[DEPTH]])
        complete = np.isfinite(values).all(axis=1)
        residual = pd.Series(np.nan, index=chunk.index, name="formula_residual_m")
        residual.loc[complete] = np.abs(values[complete, 0] - values[complete, 1] - values[complete, 2])
        totals["formula_complete_finite_rows"] += int(complete.sum())
        totals["formula_residual_above_0p05m_rows"] += int((residual > 0.05).sum())
        totals["formula_residual_above_1m_rows"] += int((residual > 1).sum())
        totals["formula_residual_above_10m_rows"] += int((residual > 10).sum())
        totals["dem_formula_residual_above_0p05m_rows"] += int((dem_rows & residual.gt(0.05)).sum())
        totals["derived_depth_formula_residual_above_0p05m_rows"] += int(
            (derived_depth & residual.gt(0.05)).sum()
        )
        totals["derived_water_formula_residual_above_0p05m_rows"] += int(
            (derived_water & residual.gt(0.05)).sum()
        )
        totals["dem_elevation_only_formula_residual_above_0p05m_rows"] += int(
            (flag.eq("E") & residual.gt(0.05)).sum()
        )

        country = chunk[SITE].fillna("UNKNOWN").str.split("+", n=1).str[0].str.split("_", n=1).str[0]
        audit_frame = pd.DataFrame(
            {"country": country, "rows": 1, "negative": negative, "below_minus5": below_minus5,
             "above500": above500, "above1000": above1000, "residual_gt_0p05": residual.gt(0.05)}
        )
        grouped = audit_frame.groupby("country", sort=False).sum()
        for name, values_country in grouped.iterrows():
            for key, value in values_country.items():
                by_country[str(name)][key] += int(value)

        extended = chunk.copy()
        extended["formula_residual_m"] = residual
        top_low_depth = update_top(extended, top_low_depth, DEPTH, largest=False)
        top_high_depth = update_top(extended, top_high_depth, DEPTH, largest=True)
        top_residual = update_top(extended, top_residual, "formula_residual_m", largest=True)
        print(f"Audit chunk {chunk_number}: {totals['rows']:,} rows", flush=True)

    if next(raw_iterator, None) is not None:
        raise ValueError("DEM output has fewer rows than raw input")
    report = json.loads(QC.read_text(encoding="utf-8"))
    report["post_write_audit"] = {
        "status": "failed" if any(violations.values()) else "passed",
        "violations": dict(violations),
        "complement_status_counts": dict(methods),
        "totals": dict(totals),
        "fill_flag_counts": dict(flags),
        "country_counts": {country: dict(counts) for country, counts in sorted(by_country.items())},
        "lowest_depth_rows": top_low_depth.replace({np.nan: None}).to_dict("records"),
        "highest_depth_rows": top_high_depth.replace({np.nan: None}).to_dict("records"),
        "largest_formula_residual_rows": top_residual.replace({np.nan: None}).to_dict("records"),
        "interpretation": "Flags only; no rows were deleted or clipped. Negative depth may include artesian conditions.",
    }
    # Replace the preliminary value that could include infinity with the audited finite maximum.
    if not top_residual.empty:
        report["row_level_output_qc"]["maximum_formula_residual_m"] = float(
            top_residual["formula_residual_m"].max()
        )
    QC.write_text(json.dumps(report, ensure_ascii=False, indent=2, allow_nan=False), encoding="utf-8")
    print(json.dumps(report["post_write_audit"]["totals"], ensure_ascii=False, indent=2), flush=True)
    print(f"Updated QC: {QC}", flush=True)
    if any(violations.values()):
        raise ValueError(f"DEM complementary-field audit failed: {dict(violations)}")


if __name__ == "__main__":
    main()
