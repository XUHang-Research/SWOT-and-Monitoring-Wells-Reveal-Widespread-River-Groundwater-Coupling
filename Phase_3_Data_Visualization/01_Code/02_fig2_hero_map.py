#!/usr/bin/env python3
"""Figure 2 — global map of river–groundwater synchronous coupling.

Top    : global Robinson map of river–groundwater coupling, defined as the residual
         correlation r between SWOT river stage and the co-located well's groundwater
         level (both deseasonalized). One-sided sequential blues encode r;
         grey marks reaches with r < 0.3. This is an effect-size screen, not a
         per-Reach statistical-detection threshold or proof of hydraulic exchange.
         The palette is deliberately NOT diverging: a red/blue map would read as
         gaining/losing (flux direction), which these data do not measure.
Below  : a 2 x 2 grid of PlateCarree maps (North America, Europe, China, Australia).
         Lettered locator boxes on the overview identify each region. All five
         maps share the vertical colour scale in a separate right-hand column.
         Regional n and median r summarize all reaches inside the displayed window;
         records outside these windows remain in the global overview.
         Headings are centered above each map. Panel a has no outer frame and
         omits the polar margins; its two threshold-key entries sit below it.

Input  : data/08_event_metrics_all_sites.parquet — one row per independently
         daily-eligible well site x SWOT Reach pair; columns used: reach_id,
         country, lon, lat, event_r (residual correlation r). Monthly-harmonic
         eligibility is deliberately not imposed on this standalone
         residual correlation map.
Output : fig2_hero_map.png (600 dpi) and fig2_hero_map.pdf in the Phase-3 root.

Requires: numpy, pandas, pyarrow, matplotlib, cartopy.
Cartopy downloads Natural Earth coastlines (110m, 50m) on first run.
"""
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import matplotlib.ticker as mticker
from matplotlib.lines import Line2D
from matplotlib.colors import Normalize, LinearSegmentedColormap
import cartopy.crs as ccrs
import cartopy.feature as cfeature
from importlib import import_module as _import_module
_grl_figure_style = _import_module("13_grl_figure_style")
GRL_TEXT_PT = _grl_figure_style.GRL_TEXT_PT
GRL_PANEL_PT = _grl_figure_style.GRL_PANEL_PT
configure_grl_matplotlib = _grl_figure_style.configure_grl_matplotlib
grl_figsize = _grl_figure_style.grl_figsize

# ---------------------------------------------------------------- plot style
configure_grl_matplotlib()

HERE = Path(__file__).resolve().parent.parent
DATA = HERE.parent / "Phase_2_Data_Analysis" / "distance1000_data"

GREY = "#B6B6B6"   # reaches with r < 0.3
LAND = "#EDEDED"
COAST = "#999999"
BOX = "#777777"    # locator boxes / regional frames
MARKER_AREA_SCALE = 0.75  # fraction of the original unscaled marker areas; no edge
BLUES = LinearSegmentedColormap.from_list(
    "coupling", ["#CFE0EF", "#9CC3E0", "#56B4E9", "#2171B5", "#08306B"])
NORM = Normalize(0.3, 0.85)
PC = ccrs.PlateCarree()

# ---------------------------------------------------------------- raw input data
S = pd.read_parquet(DATA / "08_event_metrics_all_sites.parquet")
print(f"=== raw input: 08_event_metrics_all_sites.parquet — {len(S)} rows x {S.shape[1]} cols ===")
print("columns:", ", ".join(S.columns))
print(S[["site_no", "reach_id", "country", "lon", "lat", "event_r"]].head(8).to_string(), "\n")

# aggregate the pair table to one row per SWOT reach (median across paired wells)
RC = S.groupby("reach_id").agg(
    event_r=("event_r", "median"),
    country=("country", "first"),
    lon=("lon", "median"), lat=("lat", "median")).reset_index().dropna(subset=["event_r"])

# monitoring network / country -> continent for the regional insets
# (NGWMN and USGS are United States monitoring networks)
CONTINENT = {
    "China": "China", "Germany": "Europe", "France": "Europe", "Czechia": "Europe",
    "Sweden": "Europe", "Norway": "Europe", "NGWMN": "North America", "USGS": "North America",
    "Canada": "North America", "Australia": "Australia",
    "ZAF": "Africa", "Brazil": "South America"}
RC["region"] = RC.country.map(CONTINENT).fillna("Other")

# Geographic windows include all North American, Chinese and Australian points.
# The French overseas record in South America belongs on the overview rather
# than in the European geographic window. Country/network assignments do not change.
# Padding gives every regional PlateCarree frame the same 1.65 aspect ratio.
REGIONS = {
    "North America": ("b", [-140, -64, 20, 20 + 76 / 1.65]),
    "Europe":        ("c", [-13.45, 33.45, 40, 40 + 46.9 / 1.65]),
    "China":         ("d", [76, 136, 16.8, 16.8 + 60 / 1.65]),
    "Australia":     ("e", [110.4, 156.6, -40, -12]),
}
# Sparse regional coordinates remain legible at the shared 9-pt font size.
REGION_GRATICULES = {
    "North America": ([-120, -100, -80], [30, 45]),
    "China": ([90, 110, 130], [20, 35, 50]),
    "Europe": ([0, 10, 20], [45, 55, 65]),
    "Australia": ([120, 135, 150], [-35, -25, -15]),
}


def draw_reaches(ax, df, s_det, s_nod):
    """Scatter one marker per reach; grey circles denote values below 0.3."""
    nod = df[df.event_r < 0.3]
    det = df[df.event_r >= 0.3]
    # Keep all reaches as filled circles, matching the original hero-map
    # convention: grey encodes the below-threshold context population and hue
    # carries continuous r only for reaches at or above the operational screen.
    ax.scatter(nod.lon, nod.lat, c=GREY, s=s_nod * MARKER_AREA_SCALE, marker="o",
               alpha=0.75, edgecolors="none", linewidths=0, transform=PC, zorder=2)
    sc = ax.scatter(det.lon, det.lat, c=det.event_r.clip(0.3, 0.85), cmap=BLUES, norm=NORM,
                    s=s_det * MARKER_AREA_SCALE, alpha=0.95, edgecolors="none", lw=0, transform=PC, zorder=3)
    return sc



def panel_heading(fig, slot, ax, letter, heading):
    """Center each heading above its map; keep the bold letter in the left margin."""
    box = slot.get_position(fig)
    map_box = ax.get_position()
    top = map_box.y1 + 0.012
    fig.text(box.x0 - 0.045, top, letter, fontsize=GRL_PANEL_PT,
             fontweight="bold", ha="left", va="bottom")
    fig.text((map_box.x0 + map_box.x1) / 2, top, heading, fontsize=GRL_TEXT_PT,
             ha="center", va="bottom", color="#111111", linespacing=1.15)


def main():
    fig = plt.figure(figsize=grl_figsize(170, 185))
    # A dedicated side column keeps the shared colourbar out of every map.
    outer = fig.add_gridspec(
        1, 2, left=0.065, right=0.985, bottom=0.060, top=0.940,
        width_ratios=[1, 0.13], wspace=0.12,
    )
    maps = outer[0, 0].subgridspec(
        3, 2, height_ratios=[1.6, 1, 1], hspace=0.42, wspace=0.25,
    )

    # ---- global overview, spanning both regional columns ----
    hero = fig.add_subplot(maps[0, :], projection=ccrs.Robinson())
    # Restore the frameless overview and its original latitude window.
    hero.set_extent([-180, 180, -60, 78], crs=PC)
    hero.add_feature(cfeature.LAND.with_scale("110m"), facecolor=LAND, edgecolor="none", zorder=0)
    hero.coastlines("110m", linewidth=0.3, color=COAST, zorder=1)
    grid = hero.gridlines(crs=PC, draw_labels=False, linewidth=0.25,
                          color="#B8B8B8", alpha=0.65, linestyle=":", zorder=1)
    grid.xlocator = mticker.FixedLocator(np.arange(-120, 181, 60))
    grid.ylocator = mticker.FixedLocator(np.arange(-60, 61, 30))
    hero.spines["geo"].set_visible(False)
    sc = draw_reaches(hero, RC, s_det=7, s_nod=4)

    frac = float((RC.event_r >= 0.3).mean())
    panel_heading(fig, maps[0, :], hero, "a", f"Global (n = {len(RC)}) | median r = {RC.event_r.median():.2f}")
    hero_box = hero.get_position()
    fig.legend(
        handles=[
            Line2D([], [], marker="o", color=GREY, ms=3.3 * np.sqrt(MARKER_AREA_SCALE), lw=0,
                   markeredgecolor="none", markeredgewidth=0, label="r < 0.3"),
            Line2D([], [], marker="o", color=BLUES(NORM(0.6)), ms=3.3 * np.sqrt(MARKER_AREA_SCALE), lw=0,
                   markeredgecolor="none", markeredgewidth=0, label=f"r ≥ 0.3 at {frac*100:.0f}% of reaches"),
        ],
        loc="upper center", ncol=2,
        bbox_to_anchor=((hero_box.x0 + hero_box.x1) / 2, hero_box.y0 - 0.018),
        borderaxespad=0, handlelength=0.7, handletextpad=0.5, columnspacing=1.8,
        fontsize=GRL_TEXT_PT, frameon=False,
    )

    # ---- elongated shared colourbar in its dedicated right-hand column ----
    side = outer[0, 1].get_position(fig)
    cax = fig.add_axes([side.x0, 0.140, 0.012, 0.72])
    cb = fig.colorbar(sc, cax=cax, orientation="vertical", extend="max", extendfrac=0.015)
    cb.set_label("Residual correlation r", fontsize=GRL_TEXT_PT,
                 labelpad=7)
    cb.ax.tick_params(labelsize=GRL_TEXT_PT, length=2)
    cb.set_ticks([0.3, 0.5, 0.7, 0.85])
    cb.outline.set_linewidth(0.5)

    # ---- regional grid: North America / Europe, then China / Australia ----
    for index, (region, (letter, ext)) in enumerate(REGIONS.items()):
        slot = maps[1 + index // 2, index % 2]
        iax = fig.add_subplot(slot, projection=PC)
        iax.set_extent(ext, crs=PC)
        iax.add_feature(cfeature.LAND.with_scale("50m"), facecolor=LAND, edgecolor="none", zorder=0)
        iax.coastlines("50m", linewidth=0.25, color=COAST, zorder=1)
        igrid = iax.gridlines(crs=PC, draw_labels=True, linewidth=0.2,
                             color="#B8B8B8", alpha=0.6, linestyle=":",
                             x_inline=False, y_inline=False, zorder=1)
        igrid.top_labels = False
        igrid.left_labels = True
        igrid.right_labels = False
        igrid.xlocator = mticker.FixedLocator(REGION_GRATICULES[region][0])
        igrid.ylocator = mticker.FixedLocator(REGION_GRATICULES[region][1])
        igrid.xpadding = 3
        igrid.ypadding = 3
        igrid.xlabel_style = {"size": GRL_TEXT_PT, "color": "0.35"}
        igrid.ylabel_style = {"size": GRL_TEXT_PT, "color": "0.35"}
        for sp in iax.spines.values():
            sp.set_edgecolor(BOX)
            sp.set_linewidth(0.5)
        sub = RC[RC.region == region]
        visible = sub[sub.lon.between(ext[0], ext[1]) & sub.lat.between(ext[2], ext[3])]
        draw_reaches(iax, sub, s_det=13, s_nod=9)
        panel_heading(fig, slot, iax, letter, f"{region} (n = {len(visible)}) | median r = {visible.event_r.median():.2f}")
        print(f"  panel {letter} {region:<14s} displayed = {len(visible):4d}; "
              f"country/network group = {len(sub):4d}; displayed median r = {visible.event_r.median():.2f}")

        # Lettered locator windows replace long leader lines across the figure.
        lon0, lon1, lat0, lat1 = ext
        hero.add_patch(mpatches.Rectangle((lon0, lat0), lon1 - lon0, lat1 - lat0, transform=PC,
                                          fill=False, edgecolor=BOX, lw=0.5, zorder=6))
        hero.annotate(letter, (lon0, lat1), xycoords=PC._as_mpl_transform(hero),
                      xytext=(-2, 3), textcoords="offset points",
                      fontsize=GRL_PANEL_PT, fontweight="bold", color="0.25",
                      ha="right", va="bottom", zorder=7)

    fig.savefig(HERE / "fig2_hero_map.pdf")
    fig.savefig(HERE / "fig2_hero_map.png", dpi=600)
    plt.close(fig)
    print(f"\nsaved fig2_hero_map.png/.pdf | {len(RC)} reaches, "
          f"{frac*100:.0f}% with r >= 0.3")


if __name__ == "__main__":
    main()
