#!/usr/bin/env python3
"""Run the GRL derived-data pipeline for isolated 1/3/5 km experiments."""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import subprocess
import sys
from time import time

from importlib import import_module as _import_module
_harmonic_model = _import_module("10_harmonic_model")
harmonic_criterion = _harmonic_model.harmonic_criterion


HERE = Path(__file__).resolve().parent
PHASE2 = HERE.parent
COMPLETE_PROCESS = PHASE2.parent
PHASE1 = COMPLETE_PROCESS / "Phase_1_Data_Preprocessing"
STEPS = tuple(f"{n:02d}_{name}.py" for n, name in [
    (1, "coupling_metrics"),
    (2, "event_scale_daily"),
    (3, "mechanism_covariates"),
    (4, "robustness_stats"),
    (5, "event_permutation_null"),
    (6, "main_numbers"),
    (7, "tf_comparison"),
    (8, "extract_example_series"),
])


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--distances",
        type=int,
        nargs="+",
        choices=(1000, 3000, 5000),
        default=(1000, 3000, 5000),
    )
    parser.add_argument("--start-step", type=int, choices=range(1, 9), default=1)
    parser.add_argument("--end-step", type=int, choices=range(1, 9), default=8)
    return parser.parse_args()


def run_one(distance_m: int, start_step: int, end_step: int) -> None:
    base = PHASE1 / f"distance{distance_m}"
    monthly = base / "6matched_results_monthly.txt"
    daily = base / "5matched_results_daily.txt"
    output = PHASE2 / f"distance{distance_m}_data"
    robustness_candidates = (
        PHASE1
        / "distance5000"
        / "3results_all_country_nodeid_distance5000_withwidth50_pre_et.txt"
    )
    robustness_series = {
        500: PHASE1 / "distance1000" / "3results_all_country_nodeid_distance1000_withwidth50_pre_et.txt",
        1000: PHASE1 / "distance1000" / "3results_all_country_nodeid_distance1000_withwidth50_pre_et.txt",
        2000: PHASE1 / "distance3000" / "3results_all_country_nodeid_distance3000_withwidth50_pre_et.txt",
        5000: robustness_candidates,
    }
    robustness_cache = PHASE2 / "shared_cache" / "distance_robustness_monthly_strict.parquet"

    for path in (monthly, daily, robustness_candidates, *set(robustness_series.values())):
        if not path.is_file():
            raise FileNotFoundError(path)
    output.mkdir(parents=True, exist_ok=True)

    env = os.environ.copy()
    # The full pipeline keeps the main daily population and reports finite-depth
    # subsets for depth-dependent statistics. A caller may explicitly require
    # strict completeness; the isolated mechanism script also defaults strict.
    env.setdefault('GRL_MISSING_DEPTH_POLICY', 'complete_cases')
    if env['GRL_MISSING_DEPTH_POLICY'] not in {'error', 'complete_cases'}:
        raise ValueError(f"Unknown missing-depth policy: {env['GRL_MISSING_DEPTH_POLICY']}")
    env.update(
        PYTHONUNBUFFERED="1",
        GRL_MONTHLY_FILE=str(monthly),
        GRL_DAILY_FILE=str(daily),
        GRL_OUTPUT_DIR=str(output),
        GRL_DISTANCE_COLUMN="distance_prior_bank_m",
        GRL_MAX_DISTANCE_M=str(distance_m),
        # Each threshold is applied to the unranked 5-km candidate universe;
        # the Top-1 Reach/Node is then selected independently at that threshold.
        GRL_ROBUSTNESS_CANDIDATE_FILE=str(robustness_candidates),
        GRL_ROBUSTNESS_CACHE_FILE=str(robustness_cache),
        GRL_ROBUSTNESS_SERIES_FILE_500=str(robustness_series[500]),
        GRL_ROBUSTNESS_SERIES_FILE_1000=str(robustness_series[1000]),
        GRL_ROBUSTNESS_SERIES_FILE_2000=str(robustness_series[2000]),
        GRL_ROBUSTNESS_SERIES_FILE_5000=str(robustness_series[5000]),
    )

    started = time()
    print("\n" + "=" * 78, flush=True)
    print(f"distance={distance_m} m | monthly={monthly}", flush=True)
    print(f"daily={daily} | output={output}", flush=True)
    print(
        "distance field=distance_prior_bank_m | robustness=strict 0.5/1/2/5-km rematch",
        flush=True,
    )
    print(f"harmonic eligibility={harmonic_criterion()}", flush=True)
    print(f"missing-depth policy={env['GRL_MISSING_DEPTH_POLICY']}", flush=True)
    for step_number, script_name in enumerate(STEPS, start=1):
        if not start_step <= step_number <= end_step:
            continue
        print(f"\n--- step {step_number}: {script_name} ---", flush=True)
        subprocess.run([sys.executable, str(HERE / script_name)], cwd=HERE, env=env, check=True)

    # A partial rerun must not mint a fresh full-pipeline manifest over a mix of
    # old and new products. Only a genuine step-1-through-step-8 run may do so.
    if start_step == 1 and end_step == 8:
        products = {
            path.name: {"size_bytes": path.stat().st_size}
            for path in sorted(output.iterdir())
            if path.is_file() and path.name != "pipeline_manifest.json"
        }
        manifest = {
            "distance_m": distance_m,
            "distance_column": "distance_prior_bank_m",
            "monthly_input": str(monthly.relative_to(COMPLETE_PROCESS)),
            "daily_input": str(daily.relative_to(COMPLETE_PROCESS)),
            "distance_robustness_candidate_parent": str(
                robustness_candidates.relative_to(COMPLETE_PROCESS)
            ),
            "distance_robustness_cache": str(robustness_cache.relative_to(COMPLETE_PROCESS)),
            "distance_robustness_method": "threshold_then_rerank_top1",
            "harmonic_eligibility": harmonic_criterion(),
            "missing_depth_policy": env['GRL_MISSING_DEPTH_POLICY'],
            "distance_robustness_series_sources": {
                str(threshold): str(path.relative_to(COMPLETE_PROCESS))
                for threshold, path in sorted(robustness_series.items())
            },
            "elapsed_seconds": round(time() - started, 3),
            "products": products,
        }
        with (output / "pipeline_manifest.json").open("w", encoding="utf-8") as stream:
            json.dump(manifest, stream, indent=2, ensure_ascii=False)
        print(f"wrote {output / 'pipeline_manifest.json'}", flush=True)


def main() -> None:
    args = parse_args()
    if args.start_step > args.end_step:
        raise ValueError("--start-step cannot exceed --end-step")
    for distance_m in args.distances:
        run_one(distance_m, args.start_step, args.end_step)


if __name__ == "__main__":
    main()
