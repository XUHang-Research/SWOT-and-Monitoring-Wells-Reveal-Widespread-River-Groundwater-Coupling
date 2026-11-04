#!/usr/bin/env python3
"""Figure S8 — sensitivity of the R_A–tau incompatibility screen.

(a) All-Reach fraction under a 3 x 3 grid of amplitude-ratio and
    phase-lag thresholds. The outlined circle is the primary R_A > 1.2 or tau <= 0
    definition.
(b) Independently rematched all-Reach fraction under the loose/main/strict
    temporal-support gates crossed with 0.5/1/2/5-km match-distance screens.
    Circle labels give the fraction; labels below give the Reach denominator.
    Circle area is constant and has no quantitative meaning.

Inputs : Phase-2 08_main_numbers.yml and the machine-readable statistics
         archived by Fig. S2.
Output : fig_s8_at_sensitivity.png (600 dpi) and .pdf.
"""
import json
from pathlib import Path

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
import yaml

from importlib import import_module as _import_module
_grl_figure_style = _import_module("13_grl_figure_style")
GRL_TEXT_PT = _grl_figure_style.SUPPLEMENT_TEXT_PT
GRL_PANEL_PT = _grl_figure_style.SUPPLEMENT_PANEL_PT
configure_grl_matplotlib = _grl_figure_style.configure_grl_supplementary_matplotlib
grl_figsize = _grl_figure_style.grl_figsize

configure_grl_matplotlib()

HERE = Path(__file__).resolve().parent.parent
DATA = HERE.parent / "Phase_2_Data_Analysis" / "distance1000_data"
MAIN_PATH = DATA / "08_main_numbers.yml"
TEMPORAL_PATH = HERE / "fig_s2_temporal_criteria_stats.json"

if not TEMPORAL_PATH.is_file():
    raise FileNotFoundError(
        "Fig. S8b requires fig_s2_temporal_criteria_stats.json; "
        "run 05_fig_s2_temporal_criteria.py first"
    )

main = yaml.safe_load(MAIN_PATH.read_text(encoding="utf-8"))
temporal = json.loads(TEMPORAL_PATH.read_text(encoding="utf-8"))
dfa = main["diffusive_falsification"]
population_n_reach = int(dfa["single_reach_at_uncertainty"]["ALL"]["n_reach"])

# Bundled, unmodified Scientific Colour Maps 8 palette; see colormaps/LICENSE.txt.
cmap = mpl.colors.ListedColormap(
    np.loadtxt(Path(__file__).resolve().parent / "colormaps" / "lajolla.txt")[::-1],
    name="scm.lajolla_r",
)

# Include the ALL-grid minimum (about 39%) without clipping the colour scale.
vmin, vmax = 0.35, 0.65


def cell_text_colour(value):
    """Choose readable black/white text from the actual cell luminance."""
    rgb = np.asarray(cmap(mpl.colors.Normalize(vmin, vmax)(value))[:3])
    linear = np.where(rgb <= 0.04045, rgb / 12.92, ((rgb + 0.055) / 1.055) ** 2.4)
    luminance = float(linear @ np.array([0.2126, 0.7152, 0.0722]))
    return "black" if luminance > 0.179 else "white"


def panel(fig, ax, label, dx=-0.035, dy=0.012):
    box = ax.get_position()
    fig.text(box.x0 + dx, box.y1 + dy, label, fontsize=GRL_PANEL_PT,
             fontweight="bold", ha="right", va="bottom")


# Panel a: operational R_A/tau threshold choices in the complete monthly sample.
# Preserve legacy A-prefixed lookup keys in the archived statistics.
threshold_grid = dfa["threshold_robustness"]["ALL"]
a_keys = ["A>1.1", "A>1.2", "A>1.5"]
tau_keys = ["tau<-15", "tau<=0", "tau<+15"]
threshold_values = np.array([
    [threshold_grid[f"{a_key} | {tau_key}"] for tau_key in tau_keys]
    for a_key in a_keys
], dtype=float)

# Panel b: jointly rematched temporal-support and match-distance choices.
joint = temporal["distance_by_temporal_criterion"]
criteria = ["loose", "main", "strict"]
distances = sorted(int(value) for value in joint)
joint_values = np.array([
    [joint[str(distance)][criterion]["amplified_or_leading_fraction"]
     for distance in distances]
    for criterion in criteria
], dtype=float)
joint_n = np.array([
    [joint[str(distance)][criterion]["n_reach"] for distance in distances]
    for criterion in criteria
], dtype=int)

# The main-gate row must reproduce the archived one-dimensional ALL-distance
# result. This guards the panel split without introducing a second estimate.
distance_archive = dfa["distance_robustness"]
for column, distance in enumerate(distances):
    key = f"<= {distance / 1000:g} km"
    expected = distance_archive[key]
    if joint_n[1, column] != int(expected["n_reach_all"]):
        raise ValueError(f"Fig. S8b main-gate Reach count disagrees at {key}")
    if not np.isclose(
        joint_values[1, column],
        float(expected["amplified_or_lead_all"]),
        atol=5e-4,
    ):
        raise ValueError(f"Fig. S8b main-gate fraction disagrees at {key}")

fig = plt.figure(figsize=grl_figsize(170, 84))
gs = fig.add_gridspec(
    1, 2, left=0.125, right=0.855, bottom=0.20, top=0.84,
    width_ratios=[0.90, 1.22], wspace=0.48,
)
ax_a = fig.add_subplot(gs[0, 0])
ax_b = fig.add_subplot(gs[0, 1])

circle_area = 760  # points^2, fixed across both panels; colour alone encodes fraction
xa, ya = np.meshgrid(np.arange(3), np.arange(3))
image_a = ax_a.scatter(
    xa.ravel(), ya.ravel(), c=threshold_values.ravel(), s=circle_area,
    cmap=cmap, vmin=vmin, vmax=vmax, edgecolors="0.85", linewidths=0.35,
)
for row in range(threshold_values.shape[0]):
    for column in range(threshold_values.shape[1]):
        value = threshold_values[row, column]
        colour = cell_text_colour(value)
        ax_a.text(column, row, f"{value:.0%}", ha="center", va="center",
                  fontsize=GRL_TEXT_PT, color=colour)
ax_a.scatter([1], [1], s=circle_area * 1.17, facecolors="none",
             edgecolors="0.35", linewidths=0.65)
ax_a.set_xticks(range(3), ["τ < −15", "τ ≤ 0", "τ < +15"])
ax_a.set_yticks(range(3), [r"$R_A > 1.1$", r"$R_A > 1.2$", r"$R_A > 1.5$"])
ax_a.set_xlim(-0.5, 2.5)
ax_a.set_ylim(2.6, -0.6)
ax_a.tick_params(length=0)
ax_a.set_xlabel("Phase-lag criterion (days)")
ax_a.set_ylabel("Amplitude-ratio criterion")
ax_a.set_title(
    "Physical-threshold sensitivity (all)",
    fontsize=GRL_TEXT_PT,
)
panel(fig, ax_a, "a")

xb, yb = np.meshgrid(np.arange(len(distances)), np.arange(len(criteria)))
image_b = ax_b.scatter(
    xb.ravel(), yb.ravel(), c=joint_values.ravel(), s=circle_area,
    cmap=cmap, vmin=vmin, vmax=vmax, edgecolors="0.85", linewidths=0.35,
)
for row in range(joint_values.shape[0]):
    for column in range(joint_values.shape[1]):
        value = joint_values[row, column]
        colour = cell_text_colour(value)
        ax_b.text(
            column, row, f"{value:.0%}",
            ha="center", va="center", fontsize=GRL_TEXT_PT, color=colour,
        )
        ax_b.text(column, row + 0.40, f"n={joint_n[row, column]:,}",
                  ha="center", va="center", fontsize=GRL_TEXT_PT, color="0.40")
ax_b.set_xticks(
    range(len(distances)), [f"{distance / 1000:g}" for distance in distances]
)
ax_b.set_yticks(range(len(criteria)), [criterion.capitalize() for criterion in criteria])
ax_b.set_xlim(-0.5, len(distances) - 0.5)
ax_b.set_ylim(len(criteria) - 0.4, -0.6)
ax_b.tick_params(length=0)
ax_b.set_xlabel("Maximum prior bank distance (km)")
ax_b.set_ylabel("Temporal-support gate")
ax_b.set_title("Temporal–distance sensitivity (all)", fontsize=GRL_TEXT_PT)
panel(fig, ax_b, "b")

for axes in (ax_a, ax_b):
    for spine in axes.spines.values():
        spine.set_visible(False)

# Leave room for the R_A–tau label inside the fixed-width export.
cax = fig.add_axes([0.868, 0.27, 0.012, 0.54])
cbar = fig.colorbar(image_b, cax=cax)
cbar.solids.set_rasterized(False)
cbar.ax.yaxis.set_major_formatter(mpl.ticker.PercentFormatter(xmax=1, decimals=0))
cbar.set_label("Fraction of reaches\n($R_A$–$\\tau$ incompatible)")
cbar.ax.tick_params(labelsize=GRL_TEXT_PT, width=0.5, length=2)
cbar.outline.set_linewidth(0.5)

fig.savefig(HERE / "fig_s8_at_sensitivity.pdf")
fig.savefig(HERE / "fig_s8_at_sensitivity.png", dpi=600)
plt.close(fig)

print(
    "saved fig_s8_at_sensitivity.png/.pdf | threshold range "
    f"[{threshold_values.min():.3f}, {threshold_values.max():.3f}], "
    f"joint range [{joint_values.min():.3f}, {joint_values.max():.3f}]"
)
