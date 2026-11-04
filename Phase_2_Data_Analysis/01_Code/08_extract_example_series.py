#!/usr/bin/env python3
"""Step 8 — freeze the two example paired series shown in the manuscript.

Extracts the full daily records of the two illustrative pairs from the raw
daily matched file (chunked, memory-safe) and freezes them to a small parquet:

  Germany_Bavaria+7785  — high sub-seasonal co-variability (SWOT overpasses ride the well's ups
                          and downs at every timescale)
  France+BSS002DDJL     — seasonally similar but weak sub-seasonal co-variability (shares only the
                          broad seasonal shape)

ALL rows of the two sites are kept: the dense DAILY well record plus the sparse
SWOT overpasses (river_wse is NaN between overpasses). No distance filter is
applied here — distance_m is populated only on SWOT-overpass rows, and both
sites already passed the <= 1 km screen in the analysis; re-filtering would
wrongly drop the dense well-only rows.

Inputs : raw/4matched_day_wells_shpsite_with_estimated_wse_fixed_node_alltimes_SG1.txt
Outputs: output/12_fig2_example_series.parquet (one row per site-day; columns
         site_no, reach_id, date, river_wse, river_wse_u, Value_EGM2008_m)

Requires: numpy, pandas, pyarrow.
"""
import gc
from pathlib import Path

import pandas as pd

from importlib import import_module as _import_module
_pipeline_config = _import_module("09_pipeline_config")
DAY = _pipeline_config.DAY
OUT = _pipeline_config.OUT

SITES = ["Germany_Bavaria+7785", "France+BSS002DDJL"]
COLS = ["site_no", "reach_id", "date", "river_wse", "river_wse_u", "Value_EGM2008_m"]

print(f"extract all rows of {SITES} from the daily file (chunked) ...")
parts = []
for ch in pd.read_csv(DAY, sep="\t", usecols=COLS, chunksize=500000, low_memory=False):
    ch = ch[ch.site_no.isin(SITES)]
    if len(ch):
        parts.append(ch.copy())
    del ch
    gc.collect()
ex = pd.concat(parts, ignore_index=True)
for c in ["reach_id", "river_wse", "river_wse_u", "Value_EGM2008_m"]:
    ex[c] = pd.to_numeric(ex[c], errors="coerce")
ex["date"] = pd.to_datetime(ex["date"], errors="coerce")
ex = ex.dropna(subset=["date"]).sort_values(["site_no", "date"]).reset_index(drop=True)
ex = ex[COLS]  # usecols preserves file order; restore the documented column order

for sid in SITES:
    g = ex[ex.site_no == sid]
    print(f"  {sid}: {len(g)} rows | well obs {g.Value_EGM2008_m.notna().sum()} "
          f"| SWOT overpasses {g.river_wse.notna().sum()} "
          f"| {g.date.min().date()} .. {g.date.max().date()}")

ex.to_parquet(OUT / "12_fig2_example_series.parquet")
print(f"wrote output/12_fig2_example_series.parquet ({len(ex)} rows)")
