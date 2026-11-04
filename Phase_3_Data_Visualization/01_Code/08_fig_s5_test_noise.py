#!/usr/bin/env python3
"""Figure S5 — the paired test and river WSE residual variability.

(a) The five-continent sample (four major regions plus two Brazilian reaches):
    development reaches (Germany + China, blue) and
    the geographic rest-of-world holdout (orange) on a Robinson map. Median
    well–reach match distance 0.75 km; all pairs <= 1 km.
(b) Schematic of the local diffusive (seepage) expectation: the groundwater
    response to a sinusoidal river forcing is damped (R_A <= 1) and delayed
    (τ >= 0, shaded envelope); amplified or leading responses are incompatible
    with that idealized local model.
(c) Per-reach river WSE residual variability: the median Node-level
    robust residual scale after removing annual and semiannual harmonics within
    each Node. It contains unresolved hydrologic
    variability and measurement error, and is not pure SWOT measurement noise.

Inputs : data/08_global_coupling_metrics.parquet — pair table; columns used:
         reach_id, lon, lat, split, sigma_swot_emp (legacy internal name for
         per-reach river WSE residual variability, m).
         data/08_reach_at_uncertainty.parquet — circular Reach-level R_A/tau.
         data/08_main_numbers.yml — frozen numbers; section dataset (residual-
         variability p5–p95 fold over all matched reaches).
Output : fig_s5_test_noise.png (600 dpi) and fig_s5_test_noise.pdf.

R_A is the dimensionless well/river amplitude ratio; the input column A is
retained for compatibility. Individual harmonic amplitudes retain symbol A.

Requires: numpy, pandas, pyarrow, matplotlib, pyyaml, cartopy.
Cartopy downloads Natural Earth coastlines on first run.
"""
from pathlib import Path

import numpy as np
import pandas as pd
import yaml
import matplotlib as mpl
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import cartopy.crs as ccrs
import cartopy.feature as cfeature
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
         amp="#E69F00", lead="#CC79A7", land="#EDEDED", coast="#999999", black="#000000")


def panel(fig, ax, lab, dx=-0.02, dy=0.005):
    bb = ax.get_position()
    fig.text(bb.x0 + dx, bb.y1 + dy, lab, fontsize=GRL_PANEL_PT,
             fontweight="bold", va="bottom", ha="right")


# ---------------------------------------------------------------- raw input data
S = pd.read_parquet(DATA / "08_global_coupling_metrics.parquet")
AT = pd.read_parquet(DATA / "08_reach_at_uncertainty.parquet")
YML = yaml.safe_load(open(DATA / "08_main_numbers.yml"))
print(f"=== raw input: 08_global_coupling_metrics.parquet — {len(S)} rows x {S.shape[1]} cols ===")
print(S[["site_no", "reach_id", "country", "lon", "lat", "split", "sigma_swot_emp"]].head(8).to_string())
print("\n=== raw input: 08_main_numbers.yml — sections:", ", ".join(YML.keys()), "\n")

# reach table: one row per SWOT reach, reaches with a finite amplitude ratio or lag
RC = S.groupby("reach_id").agg(
    sigma=("sigma_swot_emp", "median"), lon=("lon", "median"),
    lat=("lat", "median"), split=("split", "first")).reset_index()
RC = RC.merge(
    AT[["reach_id", "A", "tau_days"]].rename(columns={"tau_days": "tau"}),
    on="reach_id", how="left")
RC = RC[np.isfinite(RC.A) | np.isfinite(RC.tau)]
print(f"reach table: {len(RC)} analysis reaches")

f = plt.figure(figsize=grl_figsize(170, 140))

# (a) sample map
ax = f.add_axes([0.08, 0.52, 0.84, 0.45], projection=ccrs.Robinson())
ax.set_extent([-180, 180, -60, 75], crs=ccrs.PlateCarree())
ax.add_feature(cfeature.LAND.with_scale("110m"), facecolor=C["land"], edgecolor="none", zorder=0)
ax.coastlines("110m", linewidth=0.3, color=C["coast"], zorder=1)
grid = ax.gridlines(crs=ccrs.PlateCarree(), draw_labels=True, linewidth=0.25,
                    color="#B8B8B8", alpha=0.65, linestyle=":", zorder=1,
                    x_inline=False, y_inline=False)
grid.top_labels = False
grid.right_labels = True
grid.xlocator = mticker.FixedLocator([-120, -60, 0, 60, 120])
grid.ylocator = mticker.FixedLocator([-30, 0, 30, 60])
grid.xlabel_style = {"size": GRL_TEXT_PT, "color": "0.35"}
grid.ylabel_style = {"size": GRL_TEXT_PT, "color": "0.35"}
ax.spines["geo"].set_visible(False)
pc = ccrs.PlateCarree()
d = RC[RC.split == "dev"]
h = RC[RC.split == "holdout"]
ax.scatter(d.lon, d.lat, s=2.5, c=C["dev"], marker="o", alpha=0.75, lw=0, transform=pc,
           zorder=3, label=f"Development group (n = {len(d)})")
ax.scatter(h.lon, h.lat, s=4.5, c=C["hol"], marker="^", alpha=0.85, lw=0, transform=pc,
           zorder=4, label=f"Geographic testing group (n = {len(h)})")
map_handles, map_labels = ax.get_legend_handles_labels()
f.legend(map_handles, map_labels, loc="upper center", bbox_to_anchor=(0.5, 0.49),
         ncol=2, fontsize=GRL_TEXT_PT, frameon=False, handletextpad=0.2,
         columnspacing=1.4, borderaxespad=0, markerscale=2.2)
median_match_distance_km = float(YML["dataset"]["median_match_distance_km"])
f.text(0.5, 0.430, f"Median match distance {median_match_distance_km:.2f} km; all pairs ≤1 km",
       fontsize=GRL_TEXT_PT, ha="center", color="0.35")
f.text(0.025, 0.98, "a", fontsize=GRL_PANEL_PT, fontweight="bold", va="top")

# (b) schematic — diffusive expectation
ax = f.add_axes([0.06, 0.06, 0.40, 0.32])
ax.set_axis_off()
t = np.linspace(0, 24, 500)
w = 2 * np.pi / 12
riv = np.sin(w * t)
lo = np.full_like(t, np.inf)
hi = np.full_like(t, -np.inf)
for Aa in (0.15, 0.4, 0.7, 1.0):
    for lag in (0, 1, 2, 3):
        y = Aa * np.sin(w * (t - lag))
        lo = np.minimum(lo, y)
        hi = np.maximum(hi, y)
ax.fill_between(t, lo, hi, color=C["compat"], alpha=0.45, zorder=1)
ax.plot(t, riv, color=C["black"], lw=0.9, zorder=3)
ax.plot(t, 0.5 * np.sin(w * (t - 2)), color="#5B7FA6", lw=0.7, zorder=3)
ax.plot(t, 1.6 * np.sin(w * t), color=C["amp"], lw=0.8, ls="--", zorder=2)
ax.plot(t, 0.8 * np.sin(w * (t + 2.5)), color=C["lead"], lw=0.8, ls="--", zorder=2)
ax.set_xlim(0, 33)
ax.set_ylim(-2.25, 2.35)
ax.text(24.5, 0.1, "River", fontsize=GRL_TEXT_PT, va="center")
ax.text(24.5, -0.65, "Damped,\ndelayed", fontsize=GRL_TEXT_PT,
        va="center", color="#5B7FA6")
ax.annotate("Amplified", xy=(15, 1.6), xytext=(17.2, 1.9),
            fontsize=GRL_TEXT_PT, color=C["amp"], va="center",
            arrowprops=dict(arrowstyle="-", color=C["amp"], lw=0.5))
ax.text(24.5, 0.85, "Leads river", fontsize=GRL_TEXT_PT, va="center", color=C["lead"])
ax.set_title("Seepage response (schematic)", fontsize=GRL_TEXT_PT,
             color="#111111", loc="center")
ax.annotate("", xy=(14, -2.0), xytext=(12, -2.0),
            arrowprops=dict(arrowstyle="-|>", lw=0.7, color="0.3"))
ax.text(11.6, -2.0, "Months", fontsize=GRL_TEXT_PT, ha="right", va="center", color="0.3")
panel(f, ax, "b", dx=-0.035)

# (c) per-reach river WSE residual variability (global analysis reaches)
ax = f.add_axes([0.60, 0.10, 0.36, 0.28])
g = RC.sigma.dropna()
g = g[g > 0]
bins = np.geomspace(g.min(), g.max(), 34)
ax.hist(g, bins=bins, density=True, color=C["neu"], alpha=0.4, lw=0,
        label=f"global analysis reaches (n={len(g)})")
ax.set_xscale("log")
fold_g = float(np.percentile(g, 95) / np.percentile(g, 5))
fold_all = YML["dataset"].get("wse_residual_variability_p5_p95_fold")
n_all = YML["dataset"].get("n_reaches_wse_residual_variability")
if fold_all is None:  # backward compatibility with previously generated YAML
    fold_all = YML["dataset"]["sigma_swot_p5_p95_fold"]
if n_all is None:
    n_all = YML["dataset"]["n_reaches_sigma_swot_empirical"]
# Definitions and p5–p95 spans are retained in the companion caption notes.
ax.set_xlabel("River WSE residual variability (m)")
ax.set_ylabel("Density")
ax.set_title(f"Analysis reaches (n = {len(g)})", fontsize=GRL_TEXT_PT)
panel(f, ax, "c", dx=-0.045)

f.savefig(HERE / "fig_s5_test_noise.pdf")
f.savefig(HERE / "fig_s5_test_noise.png", dpi=600)
plt.close(f)
print(f"saved fig_s5_test_noise.png/.pdf | residual-variability p5–p95 fold: {fold_g:.0f}x (analysis), "
      f"{fold_all:.0f}x (all {n_all} matched reaches)")
