#!/usr/bin/env python3
"""Figure S1 — global sample composition and the fraction with r >= 0.3.

(a) Number of reaches by continent: the sample is dominated by Germany + China
    (the development regions), which motivates the geographic rest-of-world holdout.
(b) Fraction of reaches with synchronous residual correlation r >= 0.3 for all,
    development, and geographic-holdout Reaches, with reach-level bootstrap
    95% CIs, against population-specific circular-shift permutation nulls.

Inputs : data/08_event_metrics_all_sites.parquet — independently daily-eligible
         pair table; columns used: reach_id, country, event_r (residual correlation r).
         data/08_main_numbers.yml — frozen headline numbers; section correlation
         (r>=0.3 fraction CIs and the permutation null).
Output : fig_s1_composition.png (600 dpi) and fig_s1_composition.pdf.

Requires: numpy, pandas, pyarrow, matplotlib, pyyaml.
"""
from pathlib import Path

import numpy as np
import pandas as pd
import yaml
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

# ---------------------------------------------------------------- raw input data
S = pd.read_parquet(DATA / "08_event_metrics_all_sites.parquet")
YML = yaml.safe_load(open(DATA / "08_main_numbers.yml"))
print(f"=== raw input: 08_event_metrics_all_sites.parquet — {len(S)} rows x {S.shape[1]} cols ===")
print(S[["site_no", "reach_id", "country", "event_r"]].head(8).to_string())
print("\n=== raw input: 08_main_numbers.yml — sections:", ", ".join(YML.keys()), "\n")

RC = S.groupby("reach_id").agg(
    event_r=("event_r", "median"),
    country=("country", "first")).reset_index().dropna(subset=["event_r"])

# monitoring network / country -> continent (NGWMN and USGS are United States networks)
CONTINENT = {
    "China": "China", "Germany": "Europe", "France": "Europe", "Czechia": "Europe",
    "Sweden": "Europe", "Norway": "Europe", "NGWMN": "North America", "USGS": "North America",
    "Canada": "North America", "Australia": "Australia",
    "ZAF": "Africa", "Brazil": "South America"}
RC["region"] = RC.country.map(CONTINENT).fillna("Other")

fig = plt.figure(figsize=grl_figsize(170, 70))

# (a) composition by continent
axa = fig.add_axes([0.14, 0.24, 0.33, 0.65])
comp = RC.region.value_counts()
major = comp[comp >= 5].index.tolist()
other_n = int(comp[comp < 5].sum())
order = ["China", "Europe", "North America", "Australia", "Africa", "South America"]
comp = comp.reindex([o for o in order if o in major])
if other_n:
    comp = pd.concat([comp, pd.Series({"Other": other_n})])
yy = np.arange(len(comp))[::-1]
axa.barh(yy, comp.values, color="#4D4D4D", alpha=0.85, height=0.66)
for y, v in zip(yy, comp.values):
    axa.text(v + 4, y, str(int(v)), va="center", fontsize=GRL_TEXT_PT, color="0.35")
axa.set_yticks(yy)
axa.set_yticklabels(comp.index, fontsize=GRL_TEXT_PT)
axa.set_xlabel("Number of reaches", fontsize=GRL_TEXT_PT)
axa.set_xlim(0, comp.max() * 1.2)
de_cn = int(RC.country.isin(["Germany", "China"]).sum())
fig.text(axa.get_position().x0 - 0.055, axa.get_position().y1 + 0.03, "a",
         fontsize=GRL_PANEL_PT, fontweight="bold")
print(f"composition: {dict(comp)}; Germany+China = {de_cn/len(RC)*100:.0f}% of {len(RC)} reaches")

# (b) fraction with r >= 0.3 versus permutation null
axb = fig.add_axes([0.64, 0.24, 0.315, 0.65])
ci = YML["correlation"]["event_r_frac_ge_0p3"]
null = YML["correlation"]["event_permutation_null"]
rows = [
    ("all reaches", ci["ALL"]["value"], ci["ALL"]["ci95"],
     null["ALL"]["null_frac_ge0p3"], null["ALL"]["p_frac"], "#4D4D4D", "o"),
    ("Development group", ci["DEV"]["value"], ci["DEV"]["ci95"],
     null["DEV"]["null_frac_ge0p3"], null["DEV"]["p_frac"], "#0072B2", "o"),
    ("Geographic testing group", ci["HOLDOUT"]["value"], ci["HOLDOUT"]["ci95"],
     null["HOLDOUT"]["null_frac_ge0p3"], null["HOLDOUT"]["p_frac"], "#D55E00", "^"),
]
for i, (lab, v, c95, nullf, pnull, cc, mk) in enumerate(rows):
    y = len(rows) - i
    axb.plot([nullf, v], [y, y], color="0.85", lw=0.55, zorder=1)
    axb.plot([c95[0], c95[2]], [y, y], color=cc, lw=1.1, zorder=3)
    axb.scatter([v], [y], s=20, c=cc, marker=mk, zorder=4)
    axb.text(c95[2] + 0.015, y, f"{v*100:.0f}%", va="center", fontsize=GRL_TEXT_PT, color=cc)
    # A hollow diamond pairs each population with its own shift-null estimate.
    # Permutation p values are retained in the accompanying caption notes.
    axb.scatter([nullf], [y], s=20, marker="D", facecolor="white",
                edgecolor="0.45", linewidth=0.7, zorder=4)
    axb.annotate(f"{nullf:.0%}", (nullf, y), xytext=(2, -11),
                 textcoords="offset points", ha="left", va="top",
                 fontsize=GRL_TEXT_PT, color="0.4")
axb.set_yticks([3, 2, 1])
axb.set_yticklabels(["All", "Development\ngroup", "Geographic\ntesting group"], fontsize=GRL_TEXT_PT)
axb.set_xlim(0, 0.62)
axb.set_ylim(0.3, 3.6)
axb.set_xlabel("Fraction of reaches with\nresidual correlation r ≥ 0.3", fontsize=GRL_TEXT_PT)
axb.legend(handles=[
    Line2D([], [], color="0.3", marker="o", lw=1.1, markersize=3.5,
           label="Observed (95% CI)"),
    Line2D([], [], color="0.45", marker="D", mfc="white", lw=0,
           markersize=3.5, label="Shift null"),
], loc="lower right", bbox_to_anchor=(1.0, 1.015), ncol=2,
    handlelength=1.3, handletextpad=0.35, columnspacing=1.0, borderaxespad=0)
fig.text(axb.get_position().x0 - 0.05, axb.get_position().y1 + 0.03, "b",
         fontsize=GRL_PANEL_PT, fontweight="bold")

fig.savefig(HERE / "fig_s1_composition.pdf")
# 602 dpi avoids a sub-600 effective DPI after mm-to-pixel rounding.
fig.savefig(HERE / "fig_s1_composition.png", dpi=602)
plt.close(fig)
print(f"saved fig_s1_composition.png/.pdf | r>=0.3: all {ci['ALL']['value']*100:.0f}%, "
      f"development {ci['DEV']['value']*100:.0f}%, "
      f"holdout {ci['HOLDOUT']['value']*100:.0f}% | null: "
      f"all {null['ALL']['null_frac_ge0p3']*100:.0f}%, "
      f"development {null['DEV']['null_frac_ge0p3']*100:.0f}%, "
      f"holdout {null['HOLDOUT']['null_frac_ge0p3']*100:.0f}%")
