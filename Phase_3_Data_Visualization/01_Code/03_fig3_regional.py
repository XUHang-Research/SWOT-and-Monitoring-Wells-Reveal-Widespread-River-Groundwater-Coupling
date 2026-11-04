#!/usr/bin/env python3
"""Figure 3 — regional residual correlations with gradient-coloured reach points.

Within-region-normalized symmetric violins and density-constrained points show
the complete reach-level distribution. Point colour uses the complete scm.vik_r
palette on a common -1 to +1 scale, centred at zero (warm negative, blue positive).
Capped 0.9-pt intervals and symbols show regional medians and reach-bootstrap 95%
confidence intervals. Development and geographic-testing group summaries use
a color-vision-aware blue/orange pair plus
redundant circle/triangle markers.

Input  : Phase_2_Data_Analysis/distance1000_data/08_event_metrics_all_sites.parquet
         — one row per independently
         daily-eligible well site x SWOT Reach pair; columns used: reach_id,
         country, event_r (residual correlation r). Monthly-harmonic eligibility is
         deliberately not imposed on this standalone regional residual correlation
         analysis.
         Regions are countries with at least MIN_REACH analyzed reaches; the USGS
         and NGWMN monitoring networks are merged into "USA".
Output : fig3_regional.png (600 dpi), fig3_regional.pdf, and
         fig3_regional.svg in the Phase-3 figure directory.

Requires: numpy, pandas, pyarrow, scipy, matplotlib.
"""
import zlib
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from matplotlib.colors import ListedColormap, Normalize
from matplotlib.cm import ScalarMappable
from scipy.stats import gaussian_kde
from importlib import import_module as _import_module
_grl_figure_style = _import_module("13_grl_figure_style")
GRL_TEXT_PT = _grl_figure_style.GRL_TEXT_PT
configure_grl_matplotlib = _grl_figure_style.configure_grl_matplotlib
grl_figsize = _grl_figure_style.grl_figsize

# ---------------------------------------------------------------- plot style
configure_grl_matplotlib()

HERE = Path(__file__).resolve().parent.parent
DATA = HERE.parent / "Phase_2_Data_Analysis" / "distance1000_data"
# Exact Scientific Colour Maps 8 table; bundled with its MIT licence.
POINT_CMAP = ListedColormap(
    np.loadtxt(HERE / "01_Code" / "colormaps" / "vik.txt")[::-1], name="scm.vik_r"
)
POINT_NORM = Normalize(vmin=-1.0, vmax=1.0)
C = {
    "development": "#0072B2",  # Okabe-Ito blue
    "holdout": "#D55E00",      # Okabe-Ito vermilion/orange
    "cloud": "#F2F3F4",
    "cloud_edge": "#B8BEC4",
    "reach": "#677078",
    "zero": "#B7BCC1",
    "screen": "#4E5963",
}
MIN_REACH = 15                 # a country enters the panel with >= 15 analyzed reaches
DEV = {"Germany", "China"}     # development regions; all others are geographic holdout
X_LIMITS = (-0.68, 1.02)


def boot_ci(x, n=10000, seed=0):
    """Median and vectorized bootstrap 95% confidence interval."""
    x = np.asarray(x, float)
    rng = np.random.default_rng(seed)
    samples = np.median(x[rng.integers(0, len(x), (n, len(x)))], axis=1)
    return (
        float(np.median(x)),
        float(np.percentile(samples, 2.5)),
        float(np.percentile(samples, 97.5)),
    )


def violin_profile(values, width=0.245):
    """Return a within-region-normalized KDE profile for one violin."""
    values = np.asarray(values, float)
    grid = np.linspace(
        max(-1.0, values.min() - 0.12),
        min(1.0, values.max() + 0.12),
        320,
    )
    if len(values) > 1 and not np.allclose(values, values[0]):
        kde = gaussian_kde(values, bw_method="scott")
        density = kde(grid)
    else:
        density = np.exp(-0.5 * ((grid - values[0]) / 0.025) ** 2)
    keep = density >= density.max() * 0.01
    grid = grid[keep]
    density = density[keep] / density[keep].max() * width
    return grid, density


def draw_violin(ax, values, y, rng, width=0.245):
    """Draw a symmetric violin and return density-constrained point offsets."""
    grid, density = violin_profile(values, width=width)
    ax.fill_between(
        grid, y - density, y + density,
        facecolor=C["cloud"], edgecolor="none", zorder=1,
    )
    ax.plot(grid, y + density, color=C["cloud_edge"], lw=0.45, zorder=2)
    ax.plot(grid, y - density, color=C["cloud_edge"], lw=0.45, zorder=2)

    # Deterministic sina-style spread constrained by the local KDE envelope.
    local_width = np.interp(values, grid, density, left=0.018, right=0.018)
    local_width = np.maximum(local_width * 0.76, 0.014)
    return y + rng.uniform(-1.0, 1.0, len(values)) * local_width


# ---------------------------------------------------------------- raw input data
M = pd.read_parquet(DATA / "08_event_metrics_all_sites.parquet")
print(f"=== raw input: 08_event_metrics_all_sites.parquet — {len(M)} rows x {M.shape[1]} cols ===")
print("columns:", ", ".join(M.columns))
print(M[["site_no", "reach_id", "country", "event_r"]].head(8).to_string(), "\n")

M["country"] = M["country"].replace({"USGS": "USA", "NGWMN": "USA"})
R = (
    M.groupby("reach_id")
    .agg(sub_seasonal_r=("event_r", "median"), country=("country", "first"))
    .reset_index()
    .dropna(subset=["sub_seasonal_r"])
)

counts = R.country.value_counts()
regions = [country for country in counts.index if counts[country] >= MIN_REACH]
rows = []
for country in regions:
    values = R.loc[R.country == country, "sub_seasonal_r"].to_numpy(float)
    median, low, high = boot_ci(values, seed=zlib.crc32(country.encode()) % 1000)
    rows.append({
        "region": country,
        "median": median,
        "low": low,
        "high": high,
        "values": values,
        "n": len(values),
        "development": country in DEV,
    })
rows.sort(key=lambda item: item["median"])

# ---------------------------------------------------------------- figure
# The adopted layout uses a 57.375 x 49.6 mm data panel and a dedicated
# right column for the full-height vertical colour key.
fig = plt.figure(figsize=grl_figsize(105, 70))
ax = fig.add_axes([25.075 / 105, 10 / 70, 57.375 / 105, 49.6 / 70])
rng = np.random.default_rng(12)
for y, row in enumerate(rows):
    y_points = draw_violin(ax, row["values"], y, rng)
    ax.scatter(
        row["values"], y_points, s=3.2,
        c=row["values"], cmap=POINT_CMAP, norm=POINT_NORM,
        alpha=0.9, edgecolors="#899198", linewidth=0.10, zorder=3,
    )

    color = C["development"] if row["development"] else C["holdout"]
    marker = "o" if row["development"] else "^"
    ax.errorbar(
        [row["median"]], [y],
        xerr=[[row["median"] - row["low"]], [row["high"] - row["median"]]],
        fmt="none", ecolor=color, elinewidth=0.9,
        capsize=2.0, capthick=0.9, zorder=5,
    )
    ax.scatter(
        [row["median"]], [y], s=18, marker=marker,
        color=color, edgecolor="white", linewidth=0.45, zorder=6,
    )
    print(
        f"  {row['region']:<10s} n = {row['n']:4d}   "
        f"median r = {row['median']:+.3f} "
        f"[{row['low']:+.3f}, {row['high']:+.3f}]  "
        f"({'development' if row['development'] else 'holdout'})"
    )

# Neutral zero reference and the operational coupling screen used in the study.
ax.axvline(0, color=C["zero"], lw=0.55, ls=":", zorder=0)
ax.axvline(0.3, color=C["screen"], lw=0.65, ls=(0, (3, 2)), zorder=0)
ax.text(
    0.315, len(rows) - 0.40, "r = 0.3",
    fontsize=GRL_TEXT_PT, color=C["screen"], ha="left", va="top",
)

ax.set_yticks(range(len(rows)))
ax.set_yticklabels(
    [row["region"] for row in rows], fontsize=GRL_TEXT_PT,
)
ax.tick_params(axis="y", length=2.5, width=0.55, pad=4, direction="out")
ax.set_xlim(*X_LIMITS)
ax.set_ylim(-0.62, len(rows) - 0.38)
ax.set_xticks([-0.5, 0.0, 0.5, 1.0])
ax.set_xlabel("Residual correlation r", fontsize=GRL_TEXT_PT)
ax.spines["left"].set_visible(True)
ax.spines["left"].set_linewidth(0.55)

legend_handles = [
    Line2D(
        [0], [0], color=C["development"], lw=0.9, marker="o",
        markerfacecolor=C["development"], markeredgecolor="white",
        markeredgewidth=0.45, markersize=4.0, label="Development group",
    ),
    Line2D(
        [0], [0], color=C["holdout"], lw=0.9, marker="^",
        markerfacecolor=C["holdout"], markeredgecolor="white",
        markeredgewidth=0.45, markersize=4.2, label="Geographic testing group",
    ),
]
fig.legend(
    handles=legend_handles, loc="upper center", bbox_to_anchor=(0.5, 0.975),
    ncol=2, fontsize=GRL_TEXT_PT, frameon=False, handlelength=1.5,
    handletextpad=0.35, columnspacing=1.2, borderaxespad=0,
)

# Right-side key: 1.5 mm wide, full data-panel height with aligned ends.
cax = fig.add_axes([89 / 105, 10 / 70, 1.5 / 105, 49.6 / 70])
colourbar = fig.colorbar(ScalarMappable(norm=POINT_NORM, cmap=POINT_CMAP),
                       cax=cax, orientation="vertical", ticks=[-1, 0, 1])
colourbar.outline.set_linewidth(0.4)
colourbar.ax.tick_params(labelsize=GRL_TEXT_PT, length=2.0, width=0.4, pad=1.5)
colourbar.solids.set_rasterized(False)
colourbar.set_label("Residual correlation r", fontsize=GRL_TEXT_PT, labelpad=3)

metadata = {"Creator": "Matplotlib; GRL Fig. 3 regional residual correlation distributions"}
fig.savefig(HERE / "fig3_regional.pdf", metadata=metadata)
fig.savefig(HERE / "fig3_regional.svg", metadata=metadata)
fig.savefig(HERE / "fig3_regional.png", dpi=600, metadata=metadata)
plt.close(fig)
print(
    f"\nsaved fig3_regional.png/.pdf/.svg | {len(rows)} regions, "
    f"{sum(row['n'] for row in rows)} reaches "
    f"(countries with >= {MIN_REACH} reaches)"
)
