#!/usr/bin/env python3
"""Run configured Phase 1 scripts without notebooks or site-specific wrappers."""
from __future__ import annotations
import argparse
from importlib import import_module
import json
import os
from pathlib import Path
import subprocess
import sys
import time

sys.dont_write_bytecode = True

CODE = Path(__file__).resolve().parent
DOWNSTREAM = (
    "08_join_groundwater_river.py",
    "09_merge_daily_climate.py",
    "10_select_fixed_pairs.py",
    "11_validate_pair_ranks.py",
    "12_keep_top_pair.py",
    "13_aggregate_daily.py",
    "14_build_monthly.py",
)

def command_plan(config, stage, distances):
    plan = []
    def add(script, distance=5000, *arguments):
        if script == "14_build_monthly.py" and not arguments:
            arguments = (config.tables(distance)["daily"], config.tables(distance)["monthly"])
        plan.append({"script": script, "distance_m": distance, "arguments": list(map(str, arguments))})
    if stage in {"all", "prepare"}:
        add("01_add_groundwater_elevation.py")
        add("02_audit_groundwater_elevation.py")
        add("03_match_swot_nodes.py", 5000, "--radius", 5000, "--workers", config.workers["matching"])
        add("04_attach_reach_attributes.py", 5000, "--workers", config.workers["reach"])
        for table in ["nodes", "reaches"]:
            add("05_derive_nested_candidates.py", 5000,
                "--input", config.tables(5000)[table],
                "--output-1000", config.tables(1000)[table],
                "--output-3000", config.tables(3000)[table])
        add("06_precipitation_hourly_to_daily.py")
        add("07_evapotranspiration_hourly_to_daily.py")
    if stage in {"all", "downstream"}:
        for distance in sorted(set(distances), reverse=True):
            for script in DOWNSTREAM:
                add(script, distance)
    # Preserve the verified production sequence's final ordered monthly pass.
    if stage in {"all", "monthly"}:
        for distance in sorted(set(distances), reverse=True):
            add("14_build_monthly.py", distance)
    return plan

def preflight(config, stage, distances):
    required = []
    if stage in {"all", "prepare"}:
        required.extend(config.inputs.values())
        swot = config.inputs["swot_merged_root"]
        required.extend([swot/"Node", swot/"Reach", swot/"merge_granule_selection.csv",
                         swot/"merge_granule_selection_summary.json"])
    if stage == "downstream":
        required.append(config.groundwater_dem)
        required.extend(config.tables(d)["reaches"] for d in distances)
    if stage == "monthly":
        required.extend(config.tables(d)["daily"] for d in distances)
    if stage in {"downstream", "monthly"}:
        required.extend([config.precipitation_daily, config.evapotranspiration_daily])
    missing = [str(path) for path in required if not path.exists()]
    if missing:
        raise FileNotFoundError("Missing required inputs:\n" + "\n".join(missing))

def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", required=True, type=Path, help="JSON configuration; relative paths use its parent directory.")
    parser.add_argument("--stage", choices=["all", "prepare", "downstream", "monthly"], default="all")
    parser.add_argument("--distances", nargs="+", type=int, choices=[1000, 3000, 5000], default=[1000, 3000, 5000])
    parser.add_argument("--dry-run", action="store_true", help="Print the configured command plan without reading observations or creating outputs.")
    args = parser.parse_args()
    try:
        config = import_module("15_pipeline_config").load_config(args.config)
        plan = command_plan(config, args.stage, args.distances)
        for i, job in enumerate(plan, 1):
            print(f"{i:02d}. distance={job['distance_m']} m; {job['script']} {' '.join(job['arguments'])}", flush=True)
        if args.dry_run:
            print(f"Workspace (not created): {config.workspace}")
            return
        preflight(config, args.stage, args.distances)
    except (ValueError, FileNotFoundError, json.JSONDecodeError) as error:
        parser.error(str(error))
    config.groundwater_dem.parent.mkdir(parents=True, exist_ok=True)
    for distance in [1000, 3000, 5000]:
        config.tables(distance)["directory"].mkdir(parents=True, exist_ok=True)
    logs = config.workspace / "logs"
    logs.mkdir(parents=True, exist_ok=True)
    env = os.environ.copy()
    env.update(GRL_PHASE1_CONFIG=str(config.source), PYTHONDONTWRITEBYTECODE="1", PYTHONUNBUFFERED="1",
               OPENBLAS_NUM_THREADS="1", OMP_NUM_THREADS="1", MKL_NUM_THREADS="1")
    completed = []
    receipt = {"status": "running", "config": str(config.source), "stage": args.stage,
               "distances_m": sorted(set(args.distances), reverse=True), "commands": plan, "completed": completed}
    receipt_path = logs / f"phase1_{args.stage}_receipt.json"
    try:
        for i, job in enumerate(plan, 1):
            env["GRL_PHASE1_DISTANCE"] = str(job["distance_m"])
            logpath = logs / f"{i:02d}_distance{job['distance_m']}_{job['script']}.log"
            started = time.monotonic()
            print(f"Running {job['script']}; log: {logpath}", flush=True)
            with logpath.open("w") as log:
                subprocess.run([sys.executable, "-u", str(CODE/job["script"]), *job["arguments"]],
                               env=env, cwd=CODE, stdout=log, stderr=subprocess.STDOUT, check=True)
            # Source climate converters report worker failures in text. Preserve
            # the full production runner's guard even if the process exits zero.
            if "FAILED:" in logpath.read_text(errors="replace"):
                raise RuntimeError(f"Worker failure reported in {logpath}")
            completed.append({**job, "elapsed_seconds": time.monotonic()-started, "log": str(logpath)})
            receipt_path.write_text(json.dumps(receipt, indent=2)+"\n")
        receipt["status"] = "passed"
    except Exception as error:
        receipt.update(status="failed", error=str(error))
        raise
    finally:
        receipt_path.write_text(json.dumps(receipt, indent=2)+"\n")
    print(f"Phase 1 completed; receipt: {receipt_path}", flush=True)

if __name__ == "__main__":
    main()
