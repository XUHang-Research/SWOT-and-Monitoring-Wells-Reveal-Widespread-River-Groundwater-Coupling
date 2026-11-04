#!/usr/bin/env python3
"""Figure S3 — residual correlation statistical validation (1x2).

(a) Event-scale observed values for the complete independently daily-eligible
    population versus both the observation-order circular-shift null and the
    strict irregular-time OU null evaluated at the exact observed dates.
(b) Same-date observed, seasonal-harmonic, and harmonic-residual correlations
    over the complete independently daily-eligible three-correlation
    population, with Reach-bootstrap intervals shown separately for all,
    development, and holdout Reaches. Monthly-harmonic eligibility is not
    imposed. The three correlations are not additive.

Input  : data/08_robustness.json — precomputed robustness statistics
         (bootstrap CIs, distance screen, complete-daily same-date correlation
         components, circular-shift permutation null).
Output : fig_s3_robustness.png (600 dpi) and fig_s3_robustness.pdf.

Requires: numpy, matplotlib.
"""
import json
from pathlib import Path

import numpy as np
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
DATA = HERE.parent / "Phase_2_Data_Analysis" / "distance1000_data"
OK = dict(river="#0072B2", recharge="#D55E00", holdout="#D55E00", detect="#009E73",
          grey="#999999", curve="#222222", orange="#E69F00")
# Restrained blue/grey/terracotta marks used in panels a and d.
BAR = dict(observed="#4C78A8", null="#C3C7CC", time="#E69F00",
           seasonal="#C98262", deseasonalized="#4C78A8")


def panel(fig, ax, label, column_x):
    """Place panel letters on fixed figure columns, independent of tick labels."""
    bb = ax.get_position()
    fig.text(column_x, bb.y1 + 0.012, label, fontsize=GRL_PANEL_PT,
             fontweight="bold", va="bottom", ha="left")


# ---------------------------------------------------------------- raw input data
rob = json.load(open(DATA / "08_robustness.json"))
print("=== raw input: 08_robustness.json ===")
print("top-level keys:", ", ".join(rob.keys()))
print(json.dumps({k: rob[k] for k in ["bootstrap_ci_by_reach", "event_permutation_null"]},
                 indent=2)[:1200], "...\n")

# Blocking consumer contract for panel b. The three daily correlations must be
# summarized from the complete independent daily table, not from the narrower
# monthly-harmonic x daily-event intersection used by Fig. S10.
same_sample = rob.get("same_sample_correlations")
if not isinstance(same_sample, dict):
    raise KeyError("08_robustness.json lacks same_sample_correlations")
sample_definition = same_sample.get("sample_definition", {})
if sample_definition.get("source_table") != "08_event_metrics_all_sites.parquet":
    raise ValueError(
        "Fig. S3b requires complete-daily same_sample_correlations from "
        "08_event_metrics_all_sites.parquet; rerun Phase 2 step 4"
    )
for split in ("ALL", "DEV", "HOLDOUT"):
    block = same_sample.get(split)
    if not isinstance(block, dict):
        raise KeyError(f"same_sample_correlations lacks {split}")
    for key in ("n_site_pairs", "n_reach", "metric_medians", "paired_comparisons"):
        if key not in block:
            raise KeyError(f"same_sample_correlations[{split!r}] lacks {key}")
if int(same_sample["ALL"]["n_site_pairs"]) != (
        int(same_sample["DEV"]["n_site_pairs"])
        + int(same_sample["HOLDOUT"]["n_site_pairs"])):
    raise ValueError("complete-daily ALL site count is not DEV + HOLDOUT")
if int(same_sample["ALL"]["n_reach"]) != (
        int(same_sample["DEV"]["n_reach"])
        + int(same_sample["HOLDOUT"]["n_reach"])):
    raise ValueError("complete-daily ALL Reach count is not DEV + HOLDOUT")

f = plt.figure(figsize=grl_figsize(170, 88))
gs = f.add_gridspec(
    1, 2, left=0.08, right=0.985, bottom=0.19, top=0.90,
    width_ratios=[0.92, 1.08], wspace=0.56,
)
ax_null = f.add_subplot(gs[0, 0])
ax_same = f.add_subplot(gs[0, 1])

# (b) Complete-daily same-date comparison. All values and intervals are
# produced in Phase 2 from 08_event_metrics_all_sites.parquet; the renderer
# deliberately does not bootstrap a second, potentially divergent set of
# statistics. No monthly-harmonic eligibility gate is applied, and the three
# correlations are not additive components.
c = ax_same
same_date_metrics = [
    ("raw_r_same_sample_median", "Observed\nvalues"),
    ("seasonal_harmonic_r_same_sample_median", "Seasonal\nharmonics"),
    ("event_r_median", "Residuals"),
]
split_style = {
    "ALL": (OK["grey"], "o", "All reaches"),
    "DEV": (OK["river"], "o", "Development group"),
    "HOLDOUT": (OK["holdout"], "^", "Geographic testing group"),
}
x = np.arange(len(same_date_metrics), dtype=float)
offsets = {"ALL": -0.10, "DEV": 0.0, "HOLDOUT": 0.10}
for split, (colour, marker, label) in split_style.items():
    block = same_sample[split]
    metric_medians = block["metric_medians"]
    legend_label = f"{label} (n={block['n_reach']})"
    for xi, (key, _) in enumerate(same_date_metrics):
        lo, md, hi = map(float, metric_medians[key])
        xx = xi + offsets[split]
        c.plot([xx, xx], [lo, hi], color=colour, lw=1.1, zorder=2)
        c.scatter(xx, md, s=18, color=colour, marker=marker, zorder=3,
                  edgecolor="white", linewidth=0.35,
                  label=legend_label if xi == 0 else None)
c.set_xticks(x)
c.set_xticklabels([label for _, label in same_date_metrics], fontsize=GRL_TEXT_PT)
c.set_xlim(-0.45, len(same_date_metrics) - 0.55)
c.set_ylabel("Correlation r")
c.set_ylim(-0.16, 0.82)
c.set_title("Same-date correlations", fontsize=GRL_TEXT_PT)
panel(f, c, "b", column_x=0.525)
# The full group names fit in the empty band below all correlation intervals.
c.legend(frameon=False, fontsize=GRL_TEXT_PT, loc="lower left", ncol=1,
         bbox_to_anchor=(0.01, 0.045), handletextpad=0.25,
         labelspacing=0.2, borderaxespad=0.1)

# Keep the paired contrast available for the caption, without a text block in b.
dev_delta = same_sample["DEV"]["paired_comparisons"][
    "seasonal_harmonic_minus_event"
]["median_paired_difference"]
hold_delta = same_sample["HOLDOUT"]["paired_comparisons"][
    "seasonal_harmonic_minus_event"
]["median_paired_difference"]
def delta_text(values):
    lo, md, hi = map(float, values)
    return f"{md:+.2f} [{lo:+.2f}, {hi:+.2f}]"
print("Caption: paired median delta-r; development", delta_text(dev_delta),
      "; geographic testing", delta_text(hold_delta))

# (a) event-scale nulls — first panel: cited first from the main text
d = ax_null
# Panel a deliberately uses the complete daily population in this preview.
# Observed summaries and BOTH nulls must come from the same ALL record.
ep = rob["event_permutation_null"]["ALL"]
groups = ["Median residual\ncorrelation r", "Fraction of\nreaches with\nr ≥ 0.3"]
obs = [ep["obs_event_r_median"], ep["obs_frac_ge0p3"]]
null = [ep["null_event_r_median"], ep["null_frac_ge0p3"]]
time_null = [ep["time_aware_null_event_r_median"],
             ep["time_aware_null_frac_ge0p3"]]
x = np.arange(2)
w = 0.25
d.scatter(x - w, obs, s=30, color=BAR["observed"], marker="o", label="Observed", zorder=3)
d.scatter(x, null, s=25, color="0.55", marker="s", label="Index-shift null", zorder=3)
d.scatter(x + w, time_null, s=30, color=BAR["time"], marker="^", label="Time-aware OU null", zorder=3)
d.axhline(0, color=OK["curve"], lw=0.6)
# Explicit margins: observed points and p-labels need clear space below the
# legend, otherwise the annotation and the legend box collide under autoscaling
d.set_ylim(-0.08, 0.82)
d.set_xlim(-0.75, 1.55)
d.text(0 - w, max(obs[0], 0) + 0.03,
       rf"$p_{{\mathrm{{OU}}}}={ep['time_aware_p_median']:.4f}$",
       ha="center", fontsize=GRL_TEXT_PT)
d.text(1 - w, obs[1] + 0.03,
       rf"$p_{{\mathrm{{OU}}}}={ep['time_aware_p_frac']:.4f}$",
       ha="center", fontsize=GRL_TEXT_PT)
d.set_xticks(x)
d.set_xticklabels(groups, fontsize=GRL_TEXT_PT)
d.set_ylabel("Value")
d.set_title(f"All reaches (n = {ep['n_reach']})", fontsize=GRL_TEXT_PT)
panel(f, d, "a", column_x=0.012)
d.legend(frameon=False, fontsize=GRL_TEXT_PT, loc="upper left")

f.savefig(HERE / "fig_s3_robustness.pdf")
f.savefig(HERE / "fig_s3_robustness.png", dpi=600)
plt.close(f)
print(f"saved fig_s3_robustness.png/.pdf | time-aware p_median="
      f"{ep['time_aware_p_median']}, p_frac={ep['time_aware_p_frac']}")
