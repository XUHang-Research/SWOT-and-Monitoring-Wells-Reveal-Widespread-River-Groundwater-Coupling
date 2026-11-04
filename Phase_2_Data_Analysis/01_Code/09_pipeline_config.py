"""Environment-configurable paths and distance rules for the GRL pipeline.

Defaults preserve the original ``raw/`` -> ``output/`` layout.  A caller can
run the same source scripts against an isolated distance experiment by setting
the ``GRL_*`` environment variables below.
"""

from __future__ import annotations

import os
from pathlib import Path


CODE_DIR = Path(__file__).resolve().parent
SOURCE_DIR = CODE_DIR.parent
COMPLETE_PROCESS = SOURCE_DIR.parent
PHASE1_1000 = COMPLETE_PROCESS / "Phase_1_Data_Preprocessing" / "distance1000"
PHASE1_3000 = COMPLETE_PROCESS / "Phase_1_Data_Preprocessing" / "distance3000"
PHASE1_5000 = COMPLETE_PROCESS / "Phase_1_Data_Preprocessing" / "distance5000"
RAW = Path(os.environ.get("GRL_RAW_DIR", PHASE1_1000))
OUT = Path(os.environ.get("GRL_OUTPUT_DIR", SOURCE_DIR / "distance1000_data"))
MON = Path(os.environ.get("GRL_MONTHLY_FILE", RAW / "6matched_results_monthly.txt"))
DAY = Path(
    os.environ.get(
        "GRL_DAILY_FILE",
        RAW / "5matched_results_daily.txt",
    )
)
ROBUSTNESS_CANDIDATES = Path(
    os.environ.get(
        "GRL_ROBUSTNESS_CANDIDATE_FILE",
        PHASE1_5000 / "3results_all_country_nodeid_distance5000_withwidth50_pre_et.txt",
    )
)
ROBUSTNESS_SERIES_CANDIDATES = {
    500: Path(os.environ.get(
        "GRL_ROBUSTNESS_SERIES_FILE_500",
        PHASE1_1000 / "3results_all_country_nodeid_distance1000_withwidth50_pre_et.txt",
    )),
    1000: Path(os.environ.get(
        "GRL_ROBUSTNESS_SERIES_FILE_1000",
        PHASE1_1000 / "3results_all_country_nodeid_distance1000_withwidth50_pre_et.txt",
    )),
    2000: Path(os.environ.get(
        "GRL_ROBUSTNESS_SERIES_FILE_2000",
        PHASE1_3000 / "3results_all_country_nodeid_distance3000_withwidth50_pre_et.txt",
    )),
    5000: Path(os.environ.get(
        "GRL_ROBUSTNESS_SERIES_FILE_5000",
        PHASE1_5000 / "3results_all_country_nodeid_distance5000_withwidth50_pre_et.txt",
    )),
}
ROBUSTNESS_CACHE = Path(
    os.environ.get(
        "GRL_ROBUSTNESS_CACHE_FILE",
        SOURCE_DIR / "shared_cache" / "distance_robustness_monthly_strict.parquet",
    )
)
DISTANCE_COLUMN = os.environ.get("GRL_DISTANCE_COLUMN", "distance_prior_bank_m")
MAX_DISTANCE_M = float(os.environ.get("GRL_MAX_DISTANCE_M", "1000"))
MAX_DISTANCE_KM = MAX_DISTANCE_M / 1000.0

# Main harmonic eligibility screen. The model contains six coefficients:
# intercept, linear trend, and annual/semiannual sine-cosine pairs. These
# settings are shared by every Phase-2 harmonic fit and can be overridden only
# explicitly for a controlled sensitivity experiment.
HARMONIC_MIN_POINTS = int(os.environ.get("GRL_HARMONIC_MIN_POINTS", "18"))
HARMONIC_MIN_SPAN_YEARS = float(
    os.environ.get("GRL_HARMONIC_MIN_SPAN_YEARS", "1.5")
)
HARMONIC_MAX_CONDITION_NUMBER = float(
    os.environ.get("GRL_HARMONIC_MAX_CONDITION_NUMBER", "10000")
)
HARMONIC_N_PARAMETERS = 6

OUT.mkdir(parents=True, exist_ok=True)
