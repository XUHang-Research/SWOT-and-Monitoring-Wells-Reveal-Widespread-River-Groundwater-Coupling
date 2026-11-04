#!/usr/bin/env python3
"""Figure S7 — why raw seasonal correlation is not evidence of hydraulic coupling: the R_A–τ
signature test.

(a) Per-reach amplitude ratio R_A (well/river) against phase lag τ in the R_A–τ
    plane. A local diffusive (seepage) response is damped and delayed
    (R_A ≤ 1, τ ≥ 0, shaded quadrant); the observed population spreads across all
    four quadrants, with large fractions amplified (R_A > 1) or leading (τ ≤ 0).
(b) Uncertainty-propagated incompatible and lead-only fractions for all,
    development, and geographic-holdout Reaches, with values printed beside
    their hierarchical 95% intervals.
(c) Robustness of the all and geographic-holdout incompatible fractions to the
    well–Reach match-distance screen.

Inputs : data/08_reach_at_uncertainty.parquet — one row per Reach; circularly
         aggregated R_A/tau point estimates, phase concentration, paired-residual
         block-bootstrap intervals and threshold-classification probabilities.
         data/08_main_numbers.yml — frozen headline numbers; section
         diffusive_falsification (threshold and distance robustness grids).
         data/08_robustness.json — reach-level bootstrap CIs.
Output : fig_s7_signature.png (600 dpi) and fig_s7_signature.pdf.

Requires: numpy, pandas, pyarrow, matplotlib, pyyaml.
"""
import json
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from importlib import import_module as _import_module
_grl_figure_style = _import_module("13_grl_figure_style")
GRL_TEXT_PT = _grl_figure_style.SUPPLEMENT_TEXT_PT
GRL_PANEL_PT = _grl_figure_style.SUPPLEMENT_PANEL_PT
configure_grl_matplotlib = _grl_figure_style.configure_grl_supplementary_matplotlib
grl_figsize = _grl_figure_style.grl_figsize

# ---------------------------------------------------------------- plot style
configure_grl_matplotlib()

HERE = Path(__file__).resolve().parent.parent
DATA = HERE.parent / "Phase_2_Data_Analysis" / "distance1000_data"
C = dict(dev="#0072B2", hol="#D55E00", neu="#4D4D4D", compat="#DCE6F0",
         amp="#E69F00", lead="#CC79A7", black="#000000")

def panel(fig, ax, lab, dx=-0.02, dy=0.005):
    bb = ax.get_position()
    fig.text(bb.x0 + dx, bb.y1 + dy, lab, fontsize=GRL_PANEL_PT,
             fontweight="bold", va="bottom", ha="right")


# ---------------------------------------------------------------- raw input data
RC = pd.read_parquet(DATA / "08_reach_at_uncertainty.parquet").rename(
    columns={"tau_days": "tau"})
ROB = json.load(open(DATA / "08_robustness.json"))
print(f"=== raw input: 08_reach_at_uncertainty.parquet — {len(RC)} Reach rows x {RC.shape[1]} cols ===")
print(RC[["reach_id", "A", "A_ci_low", "A_ci_high", "tau", "tau_ci_low_days",
          "tau_ci_high_days", "phase_concentration_R", "split"]].head(8).to_string())
print("=== raw input: 08_robustness.json — sections:", ", ".join(ROB.keys()), "\n")

RC = RC[np.isfinite(RC.A) | np.isfinite(RC.tau)]
print(f"reach table: {len(RC)} reaches with finite circular A or tau")

f = plt.figure(figsize=grl_figsize(170, 118))
group_handles = [
    Line2D([], [], color=C["neu"], marker="o", ls="none", ms=4, label="All reaches"),
    Line2D([], [], color=C["dev"], marker="o", ls="none", ms=4, label="Development group"),
    Line2D([], [], color=C["hol"], marker="^", ls="none", ms=4, label="Geographic testing group"),
]
f.legend(handles=group_handles, loc="upper center", bbox_to_anchor=(0.5, 0.995),
         ncol=3, frameon=False, handletextpad=0.3, columnspacing=1.6,
         borderaxespad=0)

# (a) R_A–τ plane; the legacy input column A stores the annual amplitude ratio.
ax = f.add_axes([0.085, 0.115, 0.39, 0.79])
r = RC.dropna(subset=["A", "tau"]).copy()
ALO, AHI, TLO, THI = 0.005, 50, -185, 185
# Plot every Reach at its actual value; never move outliers onto an axis edge.
outside = ~r.A.between(ALO, AHI) | ~r.tau.between(TLO, THI)
if outside.any():
    raise ValueError(f"Fig. S7a: expand the axes to include {int(outside.sum())} Reaches")
ax.axhspan(ALO, 1.0, xmin=(0 - TLO) / (THI - TLO), color=C["compat"], alpha=0.35, zorder=0)
ax.text(90, 0.018, "diffusive response\n(damped, delayed)", fontsize=GRL_TEXT_PT, ha="center",
        color="#40608A", zorder=1)
uncertain = r.p_amplified_or_lead.between(0.05, 0.95, inclusive="neither")
for sp, cc, mk, ss in [("dev", C["dev"], "o", 4), ("holdout", C["hol"], "^", 6)]:
    m = r.split == sp
    ma = m & uncertain
    mc = m & ~uncertain
    ax.scatter(r.tau[ma], r.A[ma], s=ss, c=cc, marker=mk, alpha=0.18, lw=0, zorder=2)
    ax.scatter(r.tau[mc], r.A[mc], s=ss, c=cc, marker=mk, alpha=0.68, lw=0, zorder=3,
               label=f"Development group\n(n={int(m.sum())})" if sp == "dev" else f"Geographic testing group\n(n={int(m.sum())})")
ax.axhline(1.0, color="0.55", lw=0.5, ls="--", zorder=2)
ax.axvline(0, color="0.55", lw=0.5, ls="--", zorder=2)
ax.set_yscale("log")
ax.set_xlim(TLO, THI)
ax.set_ylim(ALO, AHI)
ax.set_xlabel("Phase lag τ (days)")
ax.set_ylabel(r"Amplitude ratio $R_A$ (well/river)")
q = dict(
    UL=float(((r.A > 1) & (r.tau <= 0)).mean()), UR=float(((r.A > 1) & (r.tau > 0)).mean()),
    LL=float(((r.A <= 1) & (r.tau <= 0)).mean()), LR=float(((r.A <= 1) & (r.tau > 0)).mean()))
ax.text(0.025, 0.90, f"{q['UL']*100:.0f}%", transform=ax.transAxes,
        fontsize=GRL_TEXT_PT, fontweight="bold")
ax.text(0.975, 0.90, f"{q['UR']*100:.0f}%", transform=ax.transAxes,
        fontsize=GRL_TEXT_PT, fontweight="bold", ha="right")
ax.text(0.025, 0.085, f"{q['LL']*100:.0f}%", transform=ax.transAxes,
        fontsize=GRL_TEXT_PT, fontweight="bold")
ax.text(0.975, 0.085, f"{q['LR']*100:.0f}%", transform=ax.transAxes,
        fontsize=GRL_TEXT_PT, fontweight="bold", ha="right")
ax.text(-175, 1.35, "amplified", fontsize=GRL_TEXT_PT, color=C["amp"], style="normal")
ax.text(-88, 0.55, "leads river", fontsize=GRL_TEXT_PT, color=C["lead"], style="normal", ha="center")
panel(f, ax, "a", dx=-0.045, dy=0.005)
print(f"quadrant fractions (n={len(r)}): " + ", ".join(f"{k}={v*100:.0f}%" for k, v in q.items())
      + f"; all points shown at actual values within [{ALO}, {AHI}]")

# (b) Fractions with both Reach sampling and single-Reach R_A/tau uncertainty
# propagated through a hierarchical bootstrap. The six rows are symmetric
# across ALL/development/holdout, and annotations report point [95% interval].
axb = f.add_axes([0.63, 0.57, 0.35, 0.335])
atu = ROB["single_reach_at_uncertainty"]
rows = [
    ("ALL", "Amp. or lead", atu["ALL"]["uncertainty_propagated_amplified_or_lead_fraction"], C["neu"], "o"),
    ("DEV", "Amp. or lead", atu["DEV"]["uncertainty_propagated_amplified_or_lead_fraction"], C["dev"], "o"),
    ("HOLDOUT", "Amp. or lead", atu["HOLDOUT"]["uncertainty_propagated_amplified_or_lead_fraction"], C["hol"], "^"),
    ("ALL", "Lead only", atu["ALL"]["uncertainty_propagated_lead_fraction"], C["neu"], "o"),
    ("DEV", "Lead only", atu["DEV"]["uncertainty_propagated_lead_fraction"], C["dev"], "o"),
    ("HOLDOUT", "Lead only", atu["HOLDOUT"]["uncertainty_propagated_lead_fraction"], C["hol"], "^"),
]
ylabs = []
for i, (splt, metric_label, c95, cc, mk) in enumerate(rows):
    lo, mid, hi = c95
    y = len(rows) - i - (0.35 if i >= 3 else 0.0)
    axb.plot([lo, hi], [y, y], color=cc, lw=0.9, zorder=2)
    axb.scatter([mid], [y], s=12, c=cc, marker=mk, zorder=3)
    axb.annotate(
        f"{mid:.0%} [{lo:.0%}–{hi:.0%}]", (hi + 0.012, y),
        xytext=(0, 0), textcoords="offset points",
        va="center", fontsize=GRL_TEXT_PT, color=cc, linespacing=1.0,
    )
    ylabs.append((y, metric_label))
axb.set_yticks([5.0, 1.65], ["Amplified\nor leading", "Leading\nonly"])
axb.tick_params(axis="y", length=0)
axb.axhline(3.325, color="0.88", lw=0.5, zorder=0)
axb.set_xlim(0.20, 1.00)
axb.set_xticks([0.2, 0.4, 0.6, 0.8, 1.0])
axb.set_xlabel("Fraction of reaches", fontsize=GRL_TEXT_PT)
axb.set_title("Classification (95% CI)", loc="center", color="#111111")
panel(f, axb, "b", dx=-0.035)

# (c) Match-distance sensitivity, moved from the former Fig. S3c so that all
# R_A/tau physical-screen results are presented together.
axc = f.add_axes([0.63, 0.105, 0.35, 0.34])
dist = ROB["distance_robustness"]
keys = ["<= 0.5 km", "<= 1 km", "<= 2 km", "<= 5 km"]
x_distance = np.array([0.5, 1.0, 2.0, 5.0])
holdout = np.array([dist[key]["amplified_or_lead_holdout"] for key in keys])
all_reaches = np.array([dist[key]["amplified_or_lead_all"] for key in keys])
axc.plot(x_distance, holdout, "-^", color=C["hol"], ms=4,
         label="Geographic testing group")
axc.plot(x_distance, all_reaches, "-o", color=C["neu"], ms=4, label="All reaches")
axc.axhline(0.5, color=C["black"], ls=":", lw=0.7)
axc.set_ylim(0.4, 0.65)
axc.set_xticks(x_distance, ["0.5", "1", "2", "5"])
axc.set_xlabel("Maximum prior bank distance (km)", fontsize=GRL_TEXT_PT)
axc.set_ylabel("Fraction of reaches\n($R_A$–$\\tau$ incompatible)", fontsize=GRL_TEXT_PT)
axc.set_title("Match-distance sensitivity", loc="center", color="#111111")
panel(f, axc, "c", dx=-0.035, dy=0.020)

f.savefig(HERE / "fig_s7_signature.pdf")
f.savefig(HERE / "fig_s7_signature.png", dpi=600)
plt.close(f)
print("saved fig_s7_signature.png/.pdf | A/tau plane, propagated fractions, "
      "and match-distance sensitivity")
