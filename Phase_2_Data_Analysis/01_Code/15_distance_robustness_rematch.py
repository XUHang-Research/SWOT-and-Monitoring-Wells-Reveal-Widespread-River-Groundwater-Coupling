#!/usr/bin/env python3
"""Build Phase-1-equivalent monthly inputs after rematching each distance.

For every 0.5/1/2/5-km threshold, the preferred Reach/Node is selected only
after applying that threshold.  The selected well series then follows Phase-1
Cells 26, 30 and 32: the complete well calendar is retained independently of
the threshold-specific Top-1 river calendar, exact duplicates are removed
before same-day averaging, and unique daily values are averaged to months.

The 5-km ranking source and the threshold-specific series sources are scanned
in site-sized blocks, avoiding an in-memory four-threshold expansion.  The
compact monthly result is cached and shared by all Phase-2 distance experiments.
"""

from __future__ import annotations

from collections.abc import Iterator
import gc
import json
import os
from pathlib import Path

import pandas as pd


THRESHOLDS_M = (500, 1000, 2000, 5000)
ALGORITHM_VERSION = 7
DISTANCE_COLUMN = "distance_prior_bank_m"
GROUP_COLUMNS = ["site_no", "reach_id", "node_id"]
VALUE_COLUMNS = ["river_wse", "Value_EGM2008_m"]
READ_COLUMNS = GROUP_COLUMNS + ["date", DISTANCE_COLUMN] + VALUE_COLUMNS


def _file_signature(path: Path) -> dict[str, object]:
    stat = path.stat()
    return {
        "path": str(path.resolve()),
        "size_bytes": stat.st_size,
        "mtime_ns": stat.st_mtime_ns,
    }


def _source_signature(
    path: Path,
    series_sources: dict[int, Path] | None = None,
) -> dict[str, object]:
    signature = {
        **_file_signature(path),
        "algorithm_version": ALGORITHM_VERSION,
        "thresholds_m": list(THRESHOLDS_M),
        "distance_column": DISTANCE_COLUMN,
        "ranking_observation_unit": "unique_date",
    }
    if series_sources is not None:
        signature["series_sources"] = {
            str(int(threshold)): _file_signature(Path(source))
            for threshold, source in sorted(series_sources.items())
        }
    return signature


def _candidate_dtypes() -> dict[str, str]:
    return {
        "site_no": "string",
        "reach_id": "string",
        "node_id": "string",
        "date": "string",
        DISTANCE_COLUMN: "float64",
        "river_wse": "float64",
        "Value_EGM2008_m": "float64",
    }


def _iter_site_blocks(path: Path) -> Iterator[tuple[str, pd.DataFrame]]:
    """Stream a site-grouped candidate table without splitting a well."""
    chunksize = int(os.environ.get("GRL_REMATCH_CHUNKSIZE", "100000"))
    carry = None
    finished_sites: set[str] = set()

    for chunk in pd.read_csv(
        path,
        sep="\t",
        usecols=READ_COLUMNS,
        dtype=_candidate_dtypes(),
        chunksize=chunksize,
        low_memory=False,
    ):
        chunk["date"] = pd.to_datetime(chunk["date"], errors="coerce").dt.normalize()
        chunk = chunk.dropna(subset=["site_no", "date"])
        if chunk.empty:
            continue
        if carry is not None:
            chunk = pd.concat([carry, chunk], ignore_index=True)

        last_site = str(chunk["site_no"].iloc[-1])
        complete = chunk.loc[chunk["site_no"].ne(last_site)]
        carry = chunk.loc[chunk["site_no"].eq(last_site)].copy()

        for site_no, block in complete.groupby("site_no", sort=False):
            site_key = str(site_no)
            if site_key in finished_sites:
                raise ValueError(
                    "candidate table is not grouped by site_no; streaming "
                    f"rematch would split site {site_key!r}"
                )
            finished_sites.add(site_key)
            yield site_key, block

    if carry is not None and not carry.empty:
        for site_no, block in carry.groupby("site_no", sort=False):
            site_key = str(site_no)
            if site_key in finished_sites:
                raise ValueError(
                    "candidate table is not grouped by site_no; streaming "
                    f"rematch would split site {site_key!r}"
                )
            yield site_key, block


def _selection_table(path: Path) -> tuple[pd.DataFrame, int]:
    """Select one threshold-specific Top-1 Reach/Node for every well."""
    selected_rows = []
    candidate_group_count = 0
    for site_index, (_, block) in enumerate(_iter_site_blocks(path), start=1):
        valid = block.dropna(subset=GROUP_COLUMNS + [DISTANCE_COLUMN])
        valid = valid.loc[valid[DISTANCE_COLUMN].le(max(THRESHOLDS_M))]
        if valid.empty:
            continue
        stats = valid.groupby(GROUP_COLUMNS, as_index=False, sort=False).agg(
            prior_bank_distance_m=(DISTANCE_COLUMN, "median"),
            prior_bank_distance_min_m=(DISTANCE_COLUMN, "min"),
            prior_bank_distance_max_m=(DISTANCE_COLUMN, "max"),
            obs_day_count=("date", "nunique"),
        )
        candidate_group_count += len(stats)
        spread = (
            stats["prior_bank_distance_max_m"]
            - stats["prior_bank_distance_min_m"]
        ).abs()
        if bool((spread > 1e-6).any()):
            count = int((spread > 1e-6).sum())
            raise ValueError(
                f"{count:,} site/Reach/Node groups have a non-static "
                f"{DISTANCE_COLUMN}; rematching was stopped."
            )

        for threshold_m in THRESHOLDS_M:
            eligible = stats.loc[
                stats["prior_bank_distance_m"].le(threshold_m)
            ].copy()
            if eligible.empty:
                continue
            # Cell 26: best Node within Reach, then best Reach within well.
            eligible = eligible.sort_values(
                GROUP_COLUMNS[:2]
                + ["obs_day_count", "prior_bank_distance_m", "node_id"],
                ascending=[True, True, False, True, True],
                kind="stable",
            )
            best_node = eligible.groupby(
                GROUP_COLUMNS[:2], sort=False, as_index=False
            ).head(1)
            top1 = best_node.sort_values(
                ["obs_day_count", "prior_bank_distance_m", "reach_id", "node_id"],
                ascending=[False, True, True, True],
                kind="stable",
            ).iloc[0]
            selected_rows.append(
                {
                    "threshold_m": threshold_m,
                    "site_no": top1["site_no"],
                    "reach_id": top1["reach_id"],
                    "node_id": top1["node_id"],
                    "obs_day_count": int(top1["obs_day_count"]),
                    "prior_bank_distance_m": float(top1["prior_bank_distance_m"]),
                }
            )
        if site_index % 500 == 0:
            print(
                f"  ranking pass: {site_index:,} sites, "
                f"{candidate_group_count:,} candidate groups",
                flush=True,
            )
            gc.collect()

    if not selected_rows:
        raise ValueError(f"no valid Reach/Node candidates found in {path}")
    selection = pd.DataFrame(selected_rows)
    selection = selection[
        ["threshold_m"] + GROUP_COLUMNS + ["obs_day_count", "prior_bank_distance_m"]
    ]
    return selection, candidate_group_count


def _monthly_for_site(block: pd.DataFrame, chosen: pd.DataFrame) -> list[pd.DataFrame]:
    """Reproduce Cells 26, 30 and 32 for one well at each threshold."""
    parts = []
    # Groundwater is a property of the well-date, independent of the river
    # candidate. Keep the complete series even on rejected-Node-only dates.
    groundwater = block[['date', 'Value_EGM2008_m']].dropna(
        subset=['Value_EGM2008_m']).drop_duplicates()
    if groundwater.duplicated('date').any():
        raise ValueError('Conflicting groundwater values within a well-date')
    for row in chosen.itertuples(index=False):
        selected_reach = row.reach_id
        selected_node = row.node_id
        work = block.copy()
        work["reach_id"] = work["reach_id"].fillna(selected_reach)
        work["node_id"] = work["node_id"].fillna(selected_node)
        work = work.loc[
            work["reach_id"].eq(selected_reach)
            & work["node_id"].eq(selected_node),
            GROUP_COLUMNS + ["date"] + VALUE_COLUMNS,
        ]
        if work.empty:
            continue

        work = work.drop_duplicates()
        river = work.groupby('date', as_index=False, sort=False)[['river_wse']].mean()
        daily = river.merge(groundwater, on='date', how='outer', validate='one_to_one')
        daily["date"] = daily["date"].dt.to_period("M").dt.to_timestamp()
        monthly = daily.groupby("date", as_index=False, sort=False)[VALUE_COLUMNS].mean()
        monthly.insert(0, "node_id", selected_node)
        monthly.insert(0, "reach_id", selected_reach)
        monthly.insert(0, "site_no", row.site_no)
        monthly.insert(0, "threshold_m", int(row.threshold_m))
        parts.append(monthly)
    return parts


def build_strict_monthly(
    candidate_file: Path,
    series_sources: dict[int, Path] | None = None,
) -> tuple[pd.DataFrame, dict[str, object]]:
    """Return monthly GW-SW series after independent rematching per threshold."""
    candidate_file = Path(candidate_file)
    if not candidate_file.is_file():
        raise FileNotFoundError(candidate_file)
    if series_sources is None:
        series_sources = {threshold: candidate_file for threshold in THRESHOLDS_M}
    series_sources = {
        int(threshold): Path(source) for threshold, source in series_sources.items()
    }
    missing_thresholds = sorted(set(THRESHOLDS_M).difference(series_sources))
    if missing_thresholds:
        raise KeyError(f"missing series source for thresholds: {missing_thresholds}")
    for source in set(series_sources.values()):
        if not source.is_file():
            raise FileNotFoundError(source)

    selection, candidate_group_count = _selection_table(candidate_file)
    batches = []
    thresholds_by_source: dict[Path, list[int]] = {}
    for threshold, source in series_sources.items():
        thresholds_by_source.setdefault(source, []).append(threshold)

    for source, source_thresholds in thresholds_by_source.items():
        source_selection = selection.loc[
            selection["threshold_m"].isin(source_thresholds)
        ]
        by_site = {
            str(site_no): group.copy()
            for site_no, group in source_selection.groupby("site_no", sort=False)
        }
        pending = []
        print(
            f"  series pass: thresholds={sorted(source_thresholds)} "
            f"source={source}",
            flush=True,
        )
        for index, (site_no, block) in enumerate(_iter_site_blocks(source), start=1):
            chosen = by_site.get(site_no)
            if chosen is not None:
                pending.extend(_monthly_for_site(block, chosen))
            if index % 500 == 0:
                if pending:
                    batches.append(pd.concat(pending, ignore_index=True))
                    pending.clear()
                print(f"    series pass: {index:,} sites", flush=True)
                gc.collect()
        if pending:
            batches.append(pd.concat(pending, ignore_index=True))
    if not batches:
        raise ValueError("threshold rematch produced no monthly rows")

    monthly = (
        pd.concat(batches, ignore_index=True)
        .sort_values(["threshold_m", "site_no", "date"], kind="stable")
        .reset_index(drop=True)
    )
    monthly = monthly.merge(
        selection,
        on=["threshold_m"] + GROUP_COLUMNS,
        how="left",
        validate="many_to_one",
    )
    monthly["country"] = monthly["site_no"].astype("string").str.split(
        "+", n=1
    ).str[0].str.split("_", n=1).str[0]

    provenance = _source_signature(candidate_file, series_sources)
    provenance.update(
        {
            "strict_threshold_specific_rematch": True,
            "candidate_group_count": int(candidate_group_count),
            "selection_count_by_threshold": {
                str(int(k)): int(v)
                for k, v in selection.groupby("threshold_m").size().items()
            },
            "streaming_passes": {
                "selection_source": 1,
                "each_unique_series_source": 1,
            },
            "selection_source": str(candidate_file.resolve()),
            "series_source_by_threshold": {
                str(threshold): str(source.resolve())
                for threshold, source in sorted(series_sources.items())
            },
            "daily_aggregation": "mean per site/Reach/Node/date",
            "monthly_aggregation": "mean of unique daily values",
            "phase1_missing_identifier_fill": (
                "complete source well series outer-joined to threshold-specific Top-1 river calendar"
            ),
            "exact_duplicate_handling": "drop before same-day averaging",
            "ranking": [
                "maximum unique observation dates",
                "minimum distance_prior_bank_m",
                "deterministic Reach/Node ID tie-break",
            ],
        }
    )
    return monthly, provenance


def load_or_build_strict_monthly(
    candidate_file: Path,
    cache_file: Path,
    series_sources: dict[int, Path] | None = None,
) -> tuple[pd.DataFrame, dict[str, object]]:
    """Load a valid cache, otherwise perform the out-of-core rematching."""
    # The submission archive intentionally omits the pre-ranking candidates.
    # Its optional archived-input mode verifies both supplied cache files and
    # recomputes downstream statistics without claiming to rerun pair selection.
    archive_manifest = os.environ.get("GRL_ARCHIVED_REMATCH_MANIFEST")
    if archive_manifest:
        import hashlib
        manifest_path = Path(archive_manifest)
        record = json.loads(manifest_path.read_text())
        archive_root = manifest_path.parent
        verified = {}
        for kind in ("table", "metadata"):
            item = record[kind]
            path = archive_root / item["path"]
            if hashlib.sha256(path.read_bytes()).hexdigest() != item["sha256"]:
                raise ValueError(f"Archived rematch {kind} checksum mismatch: {path}")
            verified[kind] = path
        metadata = json.loads(verified["metadata"].read_text())
        if metadata["algorithm_version"] != ALGORITHM_VERSION:
            raise ValueError("Archived rematch algorithm version mismatch")
        if metadata["thresholds_m"] != list(THRESHOLDS_M):
            raise ValueError("Archived rematch thresholds mismatch")
        return pd.read_parquet(verified["table"]), metadata
    candidate_file = Path(candidate_file)
    cache_file = Path(cache_file)
    metadata_file = cache_file.with_suffix(cache_file.suffix + ".json")
    if series_sources is None:
        series_sources = {threshold: candidate_file for threshold in THRESHOLDS_M}
    series_sources = {
        int(threshold): Path(source) for threshold, source in series_sources.items()
    }
    expected = _source_signature(candidate_file, series_sources)

    if cache_file.is_file() and metadata_file.is_file():
        with metadata_file.open("r", encoding="utf-8") as stream:
            metadata = json.load(stream)
        if all(metadata.get(key) == value for key, value in expected.items()):
            return pd.read_parquet(cache_file), metadata

    monthly, metadata = build_strict_monthly(candidate_file, series_sources)
    cache_file.parent.mkdir(parents=True, exist_ok=True)
    monthly.to_parquet(cache_file, index=False)
    with metadata_file.open("w", encoding="utf-8") as stream:
        json.dump(metadata, stream, ensure_ascii=False, indent=2)
    return monthly, metadata
