"""Paths and execution resources for the standalone Phase 1 workflow.

Scientific thresholds are defined in the numbered calculation modules. This
configuration controls external locations, output locations and process counts.
"""
from __future__ import annotations
from dataclasses import dataclass
import json
import os
from pathlib import Path

DISTANCES = (1000, 3000, 5000)
ARCHIVE = Path(__file__).resolve().parents[2]
INTERNAL_RUNS = ARCHIVE / "Phase_1_Data_Preprocessing" / "02_Reproduction"
INPUT_KEYS = (
    "groundwater_raw", "well_metadata", "dem_vrt", "sword_nodes",
    "swot_merged_root", "precipitation_hourly", "evapotranspiration_hourly",
)
DEFAULT_WORKERS = {"matching": 80, "reach": 16, "climate_conversion": 4, "climate_join": 8}

@dataclass(frozen=True)
class Configuration:
    source: Path
    inputs: dict
    workspace: Path
    workers: dict
    distance: int

    @property
    def groundwater_dem(self):
        return self.workspace / "groundwater" / "dataset_all_2023_DEM.txt"

    @property
    def precipitation_daily(self):
        return self.workspace / "climate" / "PRE_daily"

    @property
    def evapotranspiration_daily(self):
        return self.workspace / "climate" / "ET_daily"

    def tables(self, distance=None):
        d = self.distance if distance is None else int(distance)
        if d not in DISTANCES:
            raise ValueError(f"Unsupported distance: {d}")
        folder = self.workspace / f"distance{d}"
        return {
            "directory": folder,
            "nodes": folder / f"1matched_results_all_country_nodeid_distance{d}_minwidth50m——20260321_v5.txt",
            "reaches": folder / f"1matched_results_all_country_nodeid_distance{d}_minwidth50m_Reach.txt",
            "joined": folder / f"2results_all_country_nodeid_distance{d}_withwidth50.txt",
            "climate": folder / f"3results_all_country_nodeid_distance{d}_withwidth50_pre_et.txt",
            "selected": folder / "4matched_day_wells_shpsite_with_estimated_wse_fixed_node_alltimes.txt",
            "top1": folder / "4matched_day_wells_shpsite_with_estimated_wse_fixed_node_alltimes_SG1.txt",
            "daily": folder / "5matched_results_daily.txt",
            "monthly": folder / "6matched_results_monthly.txt",
        }

def load_config(filename, distance=5000):
    source = Path(filename).expanduser().resolve()
    data = json.loads(source.read_text(encoding="utf-8"))
    if not isinstance(data, dict):
        raise ValueError("Configuration must be a JSON object.")
    allowed = {"inputs", "workspace", "workers"}
    if set(data) - allowed:
        raise ValueError(f"Unknown configuration keys: {sorted(set(data) - allowed)}")
    provided = data.get("inputs", {})
    if not isinstance(provided, dict) or set(provided) != set(INPUT_KEYS):
        raise ValueError("inputs must define exactly: " + ", ".join(INPUT_KEYS))
    def resolve(value):
        if not isinstance(value, str) or not value.strip():
            raise ValueError("Every configured path must be a nonempty string.")
        path = Path(value).expanduser()
        return (source.parent / path).resolve() if not path.is_absolute() else path.resolve()
    inputs = {name: resolve(provided[name]) for name in INPUT_KEYS}
    work = resolve(data.get("workspace"))
    if (work == ARCHIVE or ARCHIVE in work.parents) and INTERNAL_RUNS not in work.parents:
        raise ValueError("An internal workspace must be a child of Phase_1_Data_Preprocessing/02_Reproduction.")
    for name, path in inputs.items():
        if work == path or work in path.parents or path in work.parents:
            raise ValueError(f"workspace and input location {name} must not contain one another.")
    configured_workers = data.get("workers", {})
    if not isinstance(configured_workers, dict) or set(configured_workers) - set(DEFAULT_WORKERS):
        raise ValueError("Unknown worker setting.")
    workers = {**DEFAULT_WORKERS, **configured_workers}
    if any(type(v) is not int or v < 1 for v in workers.values()):
        raise ValueError("Worker counts must be positive integers.")
    if int(distance) not in DISTANCES:
        raise ValueError(f"Unsupported distance: {distance}")
    return Configuration(source, inputs, work, workers, int(distance))

def from_environment():
    filename = os.environ.get("GRL_PHASE1_CONFIG")
    if not filename:
        raise RuntimeError("Run 00_run_phase1.py --config FILE; see the Phase 1 readme.")
    return load_config(filename, int(os.environ.get("GRL_PHASE1_DISTANCE", "5000")))
