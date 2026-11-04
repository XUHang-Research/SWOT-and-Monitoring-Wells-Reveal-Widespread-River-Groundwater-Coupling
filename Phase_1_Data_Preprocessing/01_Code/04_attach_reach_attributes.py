#!/usr/bin/env python3
"""Append selected SWOT RiverSP Reach attributes to the existing Node matches.

The Node matching result already contains ``reach_id`` and a dated Node
shapefile name.  This script uses those two fields to find the corresponding
daily Reach shapefile and appends three Reach-only attributes:

* slope2: enhanced water-surface slope relative to the geoid (m/m);
* p_wid_var: prior-database spatial reach-width variability (m^2);
* d_x_area: change in channel cross-sectional area from the prior value (m^2).

Daily merged Reach shapefiles can contain more than one row for the same
``reach_id``.  A direct merge would therefore multiply Node rows.  To keep one
output row per input Node match, valid values are collapsed to their median for
each (date, reach_id).  ``reach_source_count`` records how many Reach rows were
available, so ambiguous daily matches remain auditable.

This file only defines the processing workflow.  It is invoked by the configured Phase 1 Python entry point.
"""

from __future__ import annotations

from importlib import import_module as _load_module
C = _load_module("15_pipeline_config").from_environment()


import argparse
import os
import re
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import numpy as np
import pandas as pd
import pyogrio


CODE_DIR = Path(__file__).resolve().parent
WORK_DIR = C.tables(5000)["directory"]
DEFAULT_NODE_RESULTS = C.tables(5000)["nodes"]
DEFAULT_REACH_DIR = C.inputs["swot_merged_root"] / "Reach"
DEFAULT_OUTPUT = C.tables(5000)["reaches"]

DATE_PATTERN = re.compile(r"(20\d{6})")
FILL_LIMIT = -1.0e10
TARGET_COLUMNS = ["slope2", "p_wid_var", "d_x_area"]
REACH_COLUMNS = ["reach_id", *TARGET_COLUMNS]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Attach slope2, p_wid_var and d_x_area from dated SWOT Reach "
            "shapefiles to an existing Node-match table."
        )
    )
    parser.add_argument(
        "--node-results",
        type=Path,
        default=DEFAULT_NODE_RESULTS,
        help="TXT produced by 03_match_swot_nodes.py.",
    )
    parser.add_argument(
        "--reach-dir",
        type=Path,
        default=DEFAULT_REACH_DIR,
        help="Directory containing Version-D-only merged_reach_YYYYMMDD.fgb files.",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=DEFAULT_OUTPUT,
        help="Output tab-separated TXT file.",
    )
    parser.add_argument(
        "--workers",
        type=int,
        default=C.workers["reach"],
        help="Number of Reach shapefiles read in parallel (default: up to 16).",
    )
    return parser.parse_args()


def normalize_reach_id(value: object) -> str | None:
    """Return a stable string ID without converting long IDs through float."""
    if pd.isna(value):
        return None
    text = str(value).strip()
    if not text or text.lower() in {"nan", "none", "no_data", "<na>"}:
        return None
    match = re.fullmatch(r"([+-]?\d+)\.0+", text)
    return match.group(1) if match else text


def date_from_name(name: str) -> str | None:
    """Extract an ISO date from a merged Node/Reach filename."""
    match = DATE_PATTERN.search(name)
    if match is None:
        return None
    parsed = pd.to_datetime(match.group(1), format="%Y%m%d", errors="coerce")
    return None if pd.isna(parsed) else parsed.strftime("%Y-%m-%d")


def build_reach_file_map(reach_dir: Path) -> dict[str, Path]:
    """Index exactly one merged Reach FlatGeobuf file per calendar date."""
    file_map: dict[str, Path] = {}
    duplicate_dates: dict[str, list[Path]] = {}
    for path in sorted(reach_dir.glob("merged_reach_*.fgb")):
        date = date_from_name(path.name)
        if date is None:
            continue
        if date in file_map:
            duplicate_dates.setdefault(date, [file_map[date]]).append(path)
        else:
            file_map[date] = path
    if duplicate_dates:
        examples = "; ".join(
            f"{date}: {[p.name for p in paths]}"
            for date, paths in list(duplicate_dates.items())[:5]
        )
        raise ValueError(f"Multiple merged Reach FlatGeobuf files for the same date: {examples}")
    return file_map


def clean_numeric(series: pd.Series) -> pd.Series:
    """Convert SWOT numeric fields and mask the -999999999999 fill value."""
    values = pd.to_numeric(series, errors="coerce")
    values = values.mask(values <= FILL_LIMIT)
    return values.replace([np.inf, -np.inf], np.nan)


def read_one_date(task: tuple[str, str, tuple[str, ...]]) -> tuple[str, pd.DataFrame, str | None]:
    """Read one Reach FlatGeobuf and make a unique daily lookup for requested IDs."""
    date, shp_path_text, wanted_ids = task
    shp_path = Path(shp_path_text)
    empty_columns = [
        "date",
        "_reach_id_key",
        *TARGET_COLUMNS,
        "reach_source_count",
        "reach_valid_slope2_count",
        "reach_valid_p_wid_var_count",
        "reach_valid_d_x_area_count",
    ]
    try:
        info = pyogrio.read_info(shp_path)
        available = set(info["fields"])
        missing = sorted(set(REACH_COLUMNS).difference(available))
        if missing:
            raise KeyError(f"Missing fields: {missing}")

        reaches = pyogrio.read_dataframe(
            shp_path,
            columns=REACH_COLUMNS,
            read_geometry=False,
            use_arrow=True,
        )
        reaches["_reach_id_key"] = reaches["reach_id"].map(normalize_reach_id)
        reaches = reaches[reaches["_reach_id_key"].isin(wanted_ids)].copy()
        if reaches.empty:
            return date, pd.DataFrame(columns=empty_columns), None

        for column in TARGET_COLUMNS:
            reaches[column] = clean_numeric(reaches[column])

        # Median is robust to repeated continent/pass records and ignores NaN.
        daily = (
            reaches.groupby("_reach_id_key", as_index=False, sort=False, dropna=False)
            .agg(
                slope2=("slope2", "median"),
                p_wid_var=("p_wid_var", "median"),
                d_x_area=("d_x_area", "median"),
                reach_source_count=("_reach_id_key", "size"),
                reach_valid_slope2_count=("slope2", "count"),
                reach_valid_p_wid_var_count=("p_wid_var", "count"),
                reach_valid_d_x_area_count=("d_x_area", "count"),
            )
        )
        daily.insert(0, "date", date)
        return date, daily[empty_columns], None
    except Exception as exc:
        message = f"{shp_path.name}: {type(exc).__name__}: {exc}"
        return date, pd.DataFrame(columns=empty_columns), message


def main() -> None:
    args = parse_args()
    if not args.node_results.exists():
        raise FileNotFoundError(f"Node matching results not found: {args.node_results}")
    if not args.reach_dir.is_dir():
        raise NotADirectoryError(f"Reach directory not found: {args.reach_dir}")
    if args.workers < 1:
        raise ValueError("--workers must be at least 1")

    print(f"Reading Node matching results: {args.node_results}")
    nodes = pd.read_csv(
        args.node_results,
        sep="\t",
        dtype="string",
        encoding="utf-8-sig",
        low_memory=False,
    )
    required_node_columns = {"reach_id", "shp_file"}
    missing_node_columns = sorted(required_node_columns.difference(nodes.columns))
    if missing_node_columns:
        raise KeyError(f"Node matching results lack fields: {missing_node_columns}")
    existing_targets = sorted(set(TARGET_COLUMNS).intersection(nodes.columns))
    if existing_targets:
        raise ValueError(f"Target fields already exist in the Node table; refusing to overwrite: {existing_targets}")

    original_columns = nodes.columns.tolist()
    original_row_count = len(nodes)
    nodes["_input_row_order"] = np.arange(original_row_count, dtype=np.int64)
    nodes["_reach_id_key"] = nodes["reach_id"].map(normalize_reach_id)
    date_digits = nodes["shp_file"].str.extract(r"(20\d{6})", expand=False)
    extracted_dates = (
        pd.to_datetime(date_digits, format="%Y%m%d", errors="coerce")
        .dt.strftime("%Y-%m-%d")
        .astype("string")
    )

    if "date" in nodes.columns:
        supplied_dates = pd.to_datetime(nodes["date"], errors="coerce").dt.strftime("%Y-%m-%d")
        disagreement = supplied_dates.notna() & extracted_dates.notna() & (supplied_dates != extracted_dates)
        if disagreement.any():
            raise ValueError(
                f"Dates disagree between date and shp_file: {int(disagreement.sum()):,} rows"
            )
        nodes["date"] = supplied_dates.astype("string").fillna(extracted_dates)
    else:
        nodes["date"] = extracted_dates

    invalid_key = nodes["_reach_id_key"].isna() | nodes["date"].isna()
    if invalid_key.any():
        raise ValueError(
            "Unparseable reach_id or shp_file date: "
            f"{int(invalid_key.sum()):,} rows; these are not silently discarded."
        )

    requested_pairs = nodes[["date", "_reach_id_key"]].drop_duplicates()
    ids_by_date = {
        str(date): tuple(group["_reach_id_key"].dropna().unique().tolist())
        for date, group in requested_pairs.groupby("date", sort=True)
    }
    reach_file_map = build_reach_file_map(args.reach_dir)
    missing_file_dates = sorted(set(ids_by_date).difference(reach_file_map))
    if missing_file_dates:
        preview = ", ".join(missing_file_dates[:10])
        suffix = " ..." if len(missing_file_dates) > 10 else ""
        print(
            f"Warning: {len(missing_file_dates):,} dates lack Reach FlatGeobuf files; "
            f"their attributes will be missing: {preview}{suffix}"
        )

    tasks = [
        (date, str(reach_file_map[date]), ids_by_date[date])
        for date in sorted(set(ids_by_date).intersection(reach_file_map))
    ]
    print(
        f"Node rows: {original_row_count:,}; unique date + reach_id: "
        f"{len(requested_pairs):,}; Reach files to read: {len(tasks):,}"
    )

    lookups: list[pd.DataFrame] = []
    errors: list[str] = []
    if args.workers == 1:
        results = map(read_one_date, tasks)
        for _, lookup, error in results:
            if not lookup.empty:
                lookups.append(lookup)
            if error:
                errors.append(error)
    else:
        with ProcessPoolExecutor(max_workers=args.workers) as executor:
            for _, lookup, error in executor.map(read_one_date, tasks, chunksize=1):
                if not lookup.empty:
                    lookups.append(lookup)
                if error:
                    errors.append(error)

    if errors:
        preview = "\n".join(errors[:10])
        raise RuntimeError(
            f"Failed to read {len(errors):,} Reach files; first 10 errors:\n{preview}"
        )

    lookup_columns = [
        "date",
        "_reach_id_key",
        *TARGET_COLUMNS,
        "reach_source_count",
        "reach_valid_slope2_count",
        "reach_valid_p_wid_var_count",
        "reach_valid_d_x_area_count",
    ]
    if lookups:
        reach_lookup = pd.concat(lookups, ignore_index=True)[lookup_columns]
    else:
        reach_lookup = pd.DataFrame(columns=lookup_columns)

    duplicate_lookup = reach_lookup.duplicated(["date", "_reach_id_key"]).sum()
    if duplicate_lookup:
        raise AssertionError(f"Duplicate keys remain in the daily Reach lookup: {duplicate_lookup:,} rows")

    matched = nodes.merge(
        reach_lookup,
        on=["date", "_reach_id_key"],
        how="left",
        sort=False,
        validate="many_to_one",
    )
    if len(matched) != original_row_count:
        raise AssertionError(
            f"Row count changed during the join: {original_row_count:,} -> {len(matched):,}"
        )
    matched = matched.sort_values("_input_row_order", kind="stable")

    count_columns = [
        "reach_source_count",
        "reach_valid_slope2_count",
        "reach_valid_p_wid_var_count",
        "reach_valid_d_x_area_count",
    ]
    for column in count_columns:
        matched[column] = matched[column].fillna(0).astype("Int64")
    matched["reach_attribute_matched"] = matched["reach_source_count"] > 0

    appended_columns = [
        *([] if "date" in original_columns else ["date"]),
        *TARGET_COLUMNS,
        "reach_source_count",
        "reach_valid_slope2_count",
        "reach_valid_p_wid_var_count",
        "reach_valid_d_x_area_count",
        "reach_attribute_matched",
    ]
    output_columns = [*original_columns, *appended_columns]
    args.output.parent.mkdir(parents=True, exist_ok=True)
    temporary_output = args.output.with_suffix(args.output.suffix + ".tmp")
    matched[output_columns].to_csv(
        temporary_output,
        sep="\t",
        index=False,
        encoding="utf-8-sig",
    )
    temporary_output.replace(args.output)

    matched_rows = int(matched["reach_attribute_matched"].sum())
    multi_source_pairs = int((reach_lookup["reach_source_count"] > 1).sum())
    print(f"Node rows with matched Reach attributes: {matched_rows:,} / {original_row_count:,}")
    print(f"Date + reach_id pairs with multiple Reach sources: {multi_source_pairs:,}")
    for column in TARGET_COLUMNS:
        print(f"Nonmissing {column} rows: {int(matched[column].notna().sum()):,}")
    print(f"Output: {args.output}")


if __name__ == "__main__":
    main()
