#!/usr/bin/env python3
"""Reproduce selected analyses or figures without overwriting archived products."""
from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
import time

ROOT = Path(__file__).resolve().parents[2]
P1 = "Phase_1_Data_Preprocessing"
P2 = "Phase_2_Data_Analysis"
P3 = "Phase_3_Data_Visualization"
INTERNAL_RUNS = ROOT / P1 / "02_Reproduction"
STEPS = ["01_coupling_metrics.py", "02_event_scale_daily.py",
         "03_mechanism_covariates.py", "04_robustness_stats.py",
         "05_event_permutation_null.py", "06_main_numbers.py",
         "07_tf_comparison.py", "08_extract_example_series.py"]
FIGURES = {"1": "01_fig1_honest_metric.py", "2": "02_fig2_hero_map.py",
           "3": "03_fig3_regional.py", "S1": "04_fig_s1_composition.py",
           "S2": "05_fig_s2_temporal_criteria.py", "S3": "06_fig_s3_robustness.py",
           "S4": "07_fig_s4_predict.py", "S5": "08_fig_s5_test_noise.py",
           "S6": "09_fig_s6_cycle_floor.py", "S7": "10_fig_s7_signature.py",
           "S8": "11_fig_s8_at_sensitivity.py", "S9": "12_fig_s9_s10_criteria.py",
           "S10": "12_fig_s9_s10_criteria.py"}
PAIRED_INPUT_FIGURES = {"S2", "S6", "S9", "S10"}


def digest(path):
    h = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for chunk in iter(lambda: stream.read(8 * 1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def snapshot_id():
    """Identify the code, analysis inputs and artwork copied to a workspace."""
    h = hashlib.sha256()
    for folder in (P2, P3):
        for path in sorted((ROOT / folder).rglob("*")):
            if not path.is_file() or path.suffix in {".md", ".pyc"} or "__pycache__" in path.parts:
                continue
            h.update(str(path.relative_to(ROOT)).encode("utf-8"))
            h.update(b"\0")
            h.update(digest(path).encode("ascii"))
    return h.hexdigest()


def workspace(output, phase1_data=None):
    output = Path(output).resolve()
    if phase1_data and (output == phase1_data or phase1_data in output.parents):
        raise ValueError("Keep the reproduction workspace outside the external input directory.")
    if output == ROOT or output in ROOT.parents:
        raise ValueError("The output must be a separate working directory.")
    if ROOT in output.parents and INTERNAL_RUNS not in output.parents:
        raise ValueError("An internal output must be a child of Phase_1_Data_Preprocessing/02_Reproduction.")
    marker = output / "workspace.json"
    package_id = snapshot_id()
    if output.exists() and any(output.iterdir()) and not marker.exists():
        raise ValueError(f"Refusing to use a nonempty unrelated directory: {output}")
    if marker.exists():
        saved = json.loads(marker.read_text())
        if saved["package_id"] != package_id:
            raise ValueError("This workspace belongs to a different input snapshot.")
        if saved.get("external_phase1_data") != (str(phase1_data) if phase1_data else None):
            raise ValueError("Use a new workspace when changing the external Phase 1 inputs.")
        return output
    output.mkdir(parents=True, exist_ok=True)
    # Optional external tables are linked only in the disposable workspace.
    # The distributed Phase 1 directory contains code, never paired data.
    if phase1_data:
        (output / P1).symlink_to(phase1_data, target_is_directory=True)
    else:
        (output / P1).mkdir()
    for folder in [P2, P3]:
        shutil.copytree(ROOT / folder, output / folder,
                        ignore=shutil.ignore_patterns("__pycache__", "*.pyc"))
    marker.write_text(json.dumps({"package_id": package_id,
                                 "archive": str(ROOT),
                                 "external_phase1_data": str(phase1_data) if phase1_data else None}, indent=2) + "\n")
    return output


def environment(work):
    env = os.environ.copy()
    font = env.get("GRL_FONT_FAMILY", "Arial")
    for key in list(env):
        if key.startswith("GRL_"):
            del env[key]
    env.update(PYTHONDONTWRITEBYTECODE="1", PYTHONUNBUFFERED="1",
               MPLBACKEND="Agg", OPENBLAS_NUM_THREADS="1", OMP_NUM_THREADS="1",
               MKL_NUM_THREADS="1", NUMEXPR_NUM_THREADS="1",
               MPLCONFIGDIR=str(work / ".matplotlib"), GRL_FONT_FAMILY=font,
               GRL_NATURAL_EARTH_DIR=str(ROOT / P3 / "02_Resources/natural_earth"),
               GRL_ARCHIVED_REMATCH_MANIFEST=str(ROOT / P2 / "shared_cache/rematch_input.json"),
               GRL_MISSING_DEPTH_POLICY="complete_cases")
    # Avoid inheriting a machine-specific PROJ installation from the calling shell.
    import pyproj
    proj = Path(pyproj.__file__).parent / "proj_dir/share/proj"
    if not (proj / "proj.db").is_file():
        proj = Path(pyproj.datadir.get_data_dir())
    env.pop("PROJ_LIB", None)
    env["PROJ_DATA"] = str(proj)
    return env


def run(script, work, env, logname):
    logs = work / "logs"
    logs.mkdir(exist_ok=True)
    started = time.monotonic()
    print(f"Running {script.name}; log: {logs / logname}", flush=True)
    with (logs / logname).open("w") as log:
        subprocess.run([sys.executable, "-u", str(script)], cwd=script.parent,
                       env=env, stdout=log, stderr=subprocess.STDOUT, check=True)
    print(f"Completed {script.name} in {time.monotonic() - started:.1f} s", flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="command", required=True)
    figures = sub.add_parser("figures", help="Render figures using archived analysis tables")
    figures.add_argument("--only", nargs="+", choices=list(FIGURES))
    analysis = sub.add_parser("phase2", help="Rebuild summaries, or reanalyse external paired tables")
    analysis.add_argument("--distances", nargs="+", type=int, choices=[1000, 3000, 5000], default=[1000])
    analysis.add_argument("--steps", nargs="+", type=int, choices=range(1, 9))
    analysis.add_argument("--rebuild-rematch", action="store_true",
                          help="Use the supplied Phase 1 pre-ranking candidates instead of the archived sensitivity cache.")
    for command in [figures, analysis]:
        command.add_argument("--output", type=Path, default=ROOT.parent / "GRL_reproduced")
        command.add_argument("--phase1-data", type=Path,
                             help="Optional external directory containing distance*/5matched_results_daily.txt and 6matched_results_monthly.txt")
    args = parser.parse_args()
    external = args.phase1_data.resolve() if args.phase1_data else None
    if args.command == "phase2" and args.rebuild_rematch and external is None:
        parser.error("--rebuild-rematch requires --phase1-data.")
    if external is not None and not external.is_dir():
        parser.error(f"External Phase 1 data directory does not exist: {external}")
    if args.command == "figures":
        chosen = args.only or [key for key in FIGURES if external or key not in PAIRED_INPUT_FIGURES]
        require_pairs = bool(set(chosen) & PAIRED_INPUT_FIGURES)
        required_distances = [1000] if require_pairs else []
    else:
        chosen = sorted(set(args.steps or (range(1, 9) if external else [6])))
        require_pairs = any(step != 6 for step in chosen)
        required_distances = sorted(set(args.distances)) if require_pairs else []
    if require_pairs and external is None:
        parser.error("Phase 1 data are intentionally excluded. Supply --phase1-data PATH for "
                     "S2/S6/S9/S10 or Phase 2 steps other than 6; see README.md.")
    for distance in required_distances:
        for name in ("5matched_results_daily.txt", "6matched_results_monthly.txt"):
            path = external / f"distance{distance}" / name
            if not path.is_file():
                parser.error(f"Missing external paired input: {path}")
    work = workspace(args.output, external)
    env = environment(work)
    receipt = {"command": args.command, "external_phase1_data": str(external) if external else None}
    if args.command == "phase2" and args.rebuild_rematch:
        env.pop("GRL_ARCHIVED_REMATCH_MANIFEST", None)
        env["GRL_ROBUSTNESS_CANDIDATE_FILE"] = str(
            external / "distance5000/3results_all_country_nodeid_distance5000_withwidth50_pre_et.txt")
        for threshold, source_distance in ((500, 1000), (1000, 1000), (2000, 3000), (5000, 5000)):
            path = external / f"distance{source_distance}/3results_all_country_nodeid_distance{source_distance}_withwidth50_pre_et.txt"
            if not path.is_file():
                parser.error(f"Missing rematching candidate input: {path}")
            env[f"GRL_ROBUSTNESS_SERIES_FILE_{threshold}"] = str(path)
        receipt["rematch_mode"] = "rebuild_from_phase1_candidates"
    if args.command == "figures":
        # S2 precedes S8. S9 and S10 share a renderer and run only once.
        selected = {FIGURES[k] for k in chosen}
        rendered = [key for key, name in FIGURES.items() if name in selected]
        preserved = [key for key in FIGURES if key not in rendered]
        print("Figures to regenerate: " + ", ".join(rendered), flush=True)
        if preserved:
            print("Reference artwork retained without regeneration: " + ", ".join(preserved), flush=True)
        for name in sorted(selected):
            run(work / P3 / "01_Code" / name, work, env, name + ".log")
        receipt.update(regenerated_figures=rendered, reference_only_figures=preserved)
    else:
        print("Phase 2 steps: " + ", ".join(map(str, chosen)) +
              "; unselected dependencies use archived reference results.", flush=True)
        for distance in sorted(set(args.distances)):
            inputs = work / P1 / f"distance{distance}"
            env.update(GRL_MONTHLY_FILE=str(inputs / "6matched_results_monthly.txt"),
                       GRL_DAILY_FILE=str(inputs / "5matched_results_daily.txt"),
                       GRL_OUTPUT_DIR=str(work / P2 / f"distance{distance}_data"),
                       GRL_MAX_DISTANCE_M=str(distance),
                       GRL_DISTANCE_COLUMN="distance_prior_bank_m",
                       GRL_ROBUSTNESS_CACHE_FILE=str(work / P2 / "shared_cache/distance_robustness_monthly_strict.parquet"))
            for step in chosen:
                name = STEPS[step - 1]
                run(work / P2 / "01_Code" / name, work, env, f"distance{distance}_{name}.log")
        receipt.update(steps=chosen, distances_m=sorted(set(args.distances)))
    (work / (args.command + "_reproduction_receipt.json")).write_text(json.dumps(receipt, indent=2) + "\n")
    print(f"Reproduction outputs: {work}", flush=True)


if __name__ == "__main__":
    main()
