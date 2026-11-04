#!/usr/bin/env python3
"""Figure S6 — schematic record-length precision limitation for annual-band phase.

The standard error of an annual-band phase estimate scales approximately as
1/sqrt(number of observed annual cycles). The maximum paired-record span is
computed directly from the current 1-km monthly well--Reach table rather than
being entered as a fixed value. Records shorter than this maximum occupy the
shaded low-power region: annual-band phase lags are therefore reported with
wide uncertainty and the paper's inference does not rest on them.

Input  : Phase_1_Data_Preprocessing/distance1000/6matched_results_monthly.txt.
         The curve is analytic (SE ∝ 1/sqrt(cycles)); the marker is the maximum
         elapsed span of an individual pairwise-finite well--Reach record.
Output : fig_s6_cycle_floor.png (600 dpi) and fig_s6_cycle_floor.pdf.

Requires: numpy, pandas, matplotlib.
"""
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib as mpl
import matplotlib.pyplot as plt
from importlib import import_module as _import_module
_grl_figure_style = _import_module("13_grl_figure_style")
GRL_TEXT_PT = _grl_figure_style.SUPPLEMENT_TEXT_PT
GRL_PANEL_PT = _grl_figure_style.SUPPLEMENT_PANEL_PT
configure_grl_matplotlib = _grl_figure_style.configure_grl_supplementary_matplotlib
grl_figsize = _grl_figure_style.grl_figsize

# ---------------------------------------------------------------- plot style
configure_grl_matplotlib()

HERE = Path(__file__).resolve().parent.parent
ROOT = HERE.parent
MONTHLY = (
    ROOT / "Phase_1_Data_Preprocessing" / "distance1000"
    / "6matched_results_monthly.txt"
)
OK = dict(recharge="#D55E00", curve="#222222")
PERIOD_DAYS = 365.25
DISTANCE_COLUMN = "distance_prior_bank_m"


def maximum_paired_span_years(path: Path) -> tuple[float, int]:
    """Return the maximum within-pair elapsed span in the current 1-km table."""
    required = {
        "site_no", "reach_id", "date", "river_wse", "Value_EGM2008_m",
        DISTANCE_COLUMN, "number_match_SG",
    }
    header = set(pd.read_csv(path, sep="\t", nrows=0).columns)
    missing = sorted(required.difference(header))
    if missing:
        raise KeyError(f"{path} is missing required Fig. S6 fields: {missing}")

    frame = pd.read_csv(path, sep="\t", usecols=sorted(required), low_memory=False)
    for column in ("river_wse", "Value_EGM2008_m", DISTANCE_COLUMN, "number_match_SG"):
        frame[column] = pd.to_numeric(frame[column], errors="coerce")
    frame["date"] = pd.to_datetime(frame.date, errors="coerce")
    frame = frame[
        frame[DISTANCE_COLUMN].le(1000)
        & (frame.number_match_SG.eq(1) | frame.number_match_SG.isna())
    ].dropna(subset=["site_no", "reach_id", "date", "river_wse", "Value_EGM2008_m"])
    if frame.empty:
        raise ValueError(f"{path} contains no pairwise-finite records within 1 km")

    support = (frame.groupby(["site_no", "reach_id"], sort=False).date
                    .agg(first="min", last="max", n="size"))
    support = support[support.n.ge(2)].copy()
    if support.empty:
        raise ValueError(f"{path} contains no paired record with at least two dates")
    support["span_years"] = (
        (support["last"] - support["first"]).dt.total_seconds()
        / (PERIOD_DAYS * 24 * 60 * 60)
    )
    return float(support.span_years.max()), int(len(support))


SWOT_YEARS, N_PAIRED_RECORDS = maximum_paired_span_years(MONTHLY)
cycle_note = "<3 annual cycles" if SWOT_YEARS < 3 else f"{SWOT_YEARS:.2f} annual cycles"
print(f"input data: {MONTHLY}")
print(f"maximum paired-record span = {SWOT_YEARS:.4f} years "
      f"across {N_PAIRED_RECORDS:,} well--Reach records")

f = plt.figure(figsize=grl_figsize(85, 67), layout="constrained")
ax = f.subplots(1, 1)
yrs = np.linspace(1, 5, 100)
se = 1.0 / np.sqrt(np.maximum(yrs, 0.1))  # phase SE scales ~1/sqrt(n cycles)
ax.plot(yrs, se, color=OK["curve"], lw=1.2)
ax.axvspan(0, SWOT_YEARS, color=OK["recharge"], alpha=0.15)
ax.axvline(SWOT_YEARS, color=OK["recharge"], lw=1.0, ls="--")
ax.text(SWOT_YEARS + 0.05, se.max() * 0.85,
        f"Max. paired span\n{SWOT_YEARS:.2f} yr",
        fontsize=GRL_TEXT_PT, color=OK["recharge"])
ax.set_xlabel("Record length (annual cycles)")
ax.set_ylabel("Relative annual-band phase SE\n(schematic)")
ax.set_title("Schematic cycle-count scaling", fontsize=GRL_TEXT_PT)

f.savefig(HERE / "fig_s6_cycle_floor.pdf")
f.savefig(HERE / "fig_s6_cycle_floor.png", dpi=600)
plt.close(f)
print("saved fig_s6_cycle_floor.png/.pdf")
