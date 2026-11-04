#!/usr/bin/env python3
"""Figure 1 — synchronous coupling from paired records and a
same-date comparison of observed, seasonal-harmonic, and residual correlation.

(a, b) Two real paired records on dual axes: the dense daily well record (ochre
       line) with the sparse SWOT overpasses (teal dots) on top. (a) has a
       high residual correlation — the SWOT dots ride the well's ups and downs.
       (b) shares only the broad seasonal shape — the dots follow the season but
       not the residual fluctuations. Both y-axes span the same range in metres,
       each centred on its own series, so vertical distance means the same
       amplitude on both axes and the two records are directly comparable.
(c)    Population summary over the complete independently daily-eligible
       population for which all three correlations are finite: observed values,
       seasonal-harmonic reconstructions, and harmonic residuals, all evaluated
       on exactly the same SWOT dates. No monthly-harmonic eligibility screen is
       imposed. The three correlations are complementary diagnostics, not
       additive parts.

Inputs : data/08_event_metrics_all_sites.parquet — one row per independently
         daily-eligible (well site x SWOT reach) pair; columns used: site_no,
         reach_id, split, raw_r_same_sample,
         seasonal_harmonic_r_same_sample, event_r. ``event_r`` is the
         harmonic-residual correlation. This table supplies both the example
         metric lookup and the complete daily population in panel c.
         data/08_robustness.json — precomputed Reach-bootstrap confidence
         intervals for the same complete daily three-correlation population.
         data/12_fig2_example_series.parquet — daily time series for the two
         example pairs; columns used: site_no, date, river_wse (SWOT water-surface
         elevation, m), Value_EGM2008_m (groundwater level, m).
Output : fig1_honest_metric.png (600 dpi) and fig1_honest_metric.pdf.

Requires: numpy, pandas, pyarrow, matplotlib.
"""
import json
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
from matplotlib.ticker import FormatStrFormatter, MaxNLocator
from matplotlib.lines import Line2D
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
C = dict(river="#007F86", groundwater="#B67C5B", neu="#536166",
         violin="#D8E4E5", violin_edge="#8FA9AC", summary="#253F46")


def panel(fig, ax, lab, dx=-0.02, dy=0.006):
    """Bold panel letter anchored to the axes' figure position."""
    bb = ax.get_position()
    fig.text(bb.x0 + dx, bb.y1 + dy, lab, fontsize=GRL_PANEL_PT,
             fontweight="bold", va="bottom", ha="right")


# ---------------------------------------------------------------- raw input data
EVENT = pd.read_parquet(DATA / "08_event_metrics_all_sites.parquet")
EX = pd.read_parquet(DATA / "12_fig2_example_series.parquet")
ROB = json.load(open(DATA / "08_robustness.json"))
required_example_columns = {
    "site_no", "reach_id", "date", "river_wse", "Value_EGM2008_m",
}
missing_example_columns = sorted(required_example_columns.difference(EX.columns))
if missing_example_columns:
    raise KeyError(
        "12_fig2_example_series.parquet is missing fields: "
        f"{missing_example_columns}"
    )
EX["date"] = pd.to_datetime(EX.date)
print(f"=== raw input: 08_event_metrics_all_sites.parquet — "
      f"{len(EVENT)} rows x {EVENT.shape[1]} cols ===")
print("columns:", ", ".join(EVENT.columns))
metric_preview = [
    "site_no", "reach_id", "country", "raw_r_same_sample",
    "seasonal_harmonic_r_same_sample", "event_r",
]
print(EVENT[[c for c in metric_preview if c in EVENT]].head(8).to_string())
print(f"\n=== raw input: 12_fig2_example_series.parquet — {len(EX)} rows x {EX.shape[1]} cols ===")
print("columns:", ", ".join(EX.columns))
print(EX.head(8).to_string(), "\n")

DAILY_METRIC_COLUMNS = (
    "raw_r_same_sample",
    "seasonal_harmonic_r_same_sample",
    "event_r",
)
required_event_columns = {"site_no", "reach_id", "split", *DAILY_METRIC_COLUMNS}
missing_event_columns = sorted(required_event_columns.difference(EVENT.columns))
if missing_event_columns:
    raise KeyError(
        "08_event_metrics_all_sites.parquet is missing fields: "
        f"{missing_event_columns}"
    )
if EVENT.site_no.duplicated().any():
    raise ValueError(
        "08_event_metrics_all_sites.parquet must contain exactly one row per "
        f"site_no; found {int(EVENT.site_no.duplicated().sum())} duplicates"
    )
if not EVENT.split.isin(["dev", "holdout"]).all():
    raise ValueError("daily event table contains invalid or missing split labels")
if not EVENT.groupby("reach_id").split.nunique(dropna=False).eq(1).all():
    raise ValueError("a daily-eligible Reach crosses the frozen development/holdout split")

same_sample = ROB.get("same_sample_correlations")
if not isinstance(same_sample, dict):
    raise KeyError("08_robustness.json lacks same_sample_correlations")
sample_definition = same_sample.get("sample_definition", {})
if sample_definition.get("source_table") != "08_event_metrics_all_sites.parquet":
    raise ValueError(
        "Fig. 1 requires same_sample_correlations generated from the complete "
        "08_event_metrics_all_sites.parquet population; rerun Phase 2 step 4"
    )


def main():
    f = plt.figure(figsize=grl_figsize(170, 108))

    # One shared key in its own top margin, outside both data panels.
    f.legend(handles=[
        Line2D([], [], linestyle="none", marker="o", markersize=3.4,
               markerfacecolor=C["river"], markeredgecolor="white",
               markeredgewidth=0.3, label="SWOT overpass"),
        Line2D([], [], color=C["groundwater"], lw=1.0,
               label="Groundwater level, daily"),
    ], loc="upper center", bbox_to_anchor=(0.305, 0.985), ncol=2,
        fontsize=GRL_TEXT_PT, frameon=False, handlelength=1.8,
        handletextpad=0.5, columnspacing=1.2, borderaxespad=0)

    # (a, b) two real paired records on dual axes, stacked on the left
    # (site id, axes rectangle, panel letter)
    exdef = [("Germany_Bavaria+7785", [0.105, 0.59, 0.400, 0.27], "a"),
             ("France+BSS002DDJL", [0.105, 0.14, 0.400, 0.27], "b")]
    for sid, rect, plab in exdef:
        g = EX[EX.site_no == sid].sort_values("date")
        if g.empty:
            raise ValueError(f"example site {sid!r} is absent from the series table")
        gw = g.dropna(subset=["Value_EGM2008_m"])   # dense daily well record
        gs = g.dropna(subset=["river_wse"])         # sparse SWOT overpasses
        metric_row = EVENT[EVENT.site_no == sid]
        if len(metric_row) != 1:
            raise ValueError(
                f"example site {sid!r} has {len(metric_row)} rows in the complete "
                "daily event-metric table; expected exactly one"
            )
        met = metric_row.iloc[0]
        series_reaches = pd.to_numeric(g.reach_id, errors="coerce")
        metric_reach = pd.to_numeric(pd.Series([met.reach_id]), errors="coerce").iloc[0]
        if (series_reaches.isna().any() or series_reaches.nunique() != 1
                or not np.isclose(series_reaches.iloc[0], metric_reach,
                                  rtol=0.0, atol=0.0)):
            raise ValueError(
                f"example site {sid!r} has a missing, non-unique, or mismatched "
                "reach_id between its series and daily metric row"
            )
        axr = f.add_axes(rect)
        axg = axr.twinx()
        axg.plot(gw.date, gw.Value_EGM2008_m, "-", color=C["groundwater"],
                 lw=1.0, zorder=2)
        axr.scatter(gs.date, gs.river_wse, s=10, color=C["river"],
                    edgecolors="white", linewidths=0.2, zorder=4)
        axr.set_zorder(axg.get_zorder() + 1)
        axr.patch.set_visible(False)  # SWOT dots drawn over the well line
        # same y-span in metres on both axes, each centred on its own series
        sw = gs.river_wse.values
        wl = gw.Value_EGM2008_m.values
        span = 1.18 * max(np.nanmax(sw) - np.nanmin(sw), np.nanmax(wl) - np.nanmin(wl))
        swc = (np.nanmin(sw) + np.nanmax(sw)) / 2.0
        wlc = (np.nanmin(wl) + np.nanmax(wl)) / 2.0
        axr.set_ylim(swc - span / 2, swc + span / 2)
        axg.set_ylim(wlc - span / 2, wlc + span / 2)
        axr.tick_params(axis="y", labelsize=GRL_TEXT_PT, length=2.5)
        axg.tick_params(axis="y", labelsize=GRL_TEXT_PT, length=2.5)
        axr.set_ylabel("River WSE (m)", fontsize=GRL_TEXT_PT)
        axg.set_ylabel("Groundwater level (m)", fontsize=GRL_TEXT_PT, rotation=90)
        # pin y-axis labels to identical axes-fraction coords in (a) and (b) so they
        # line up vertically across the two panels; uniform 1-decimal tick format
        # Reserve space for 9-pt tick labels and single-line vertical axis labels.
        axr.yaxis.set_label_coords(-0.180, 0.5)
        axg.yaxis.set_label_coords(1.180, 0.5)
        axr.yaxis.set_major_formatter(FormatStrFormatter("%.1f"))
        axg.yaxis.set_major_formatter(FormatStrFormatter("%.1f"))
        axr.yaxis.set_major_locator(MaxNLocator(nbins=4, steps=[1, 2, 2.5, 5, 10]))
        axg.yaxis.set_major_locator(MaxNLocator(nbins=4, steps=[1, 2, 2.5, 5, 10]))
        # Colored spines link each axis to its series; all text remains dark.
        axr.spines["top"].set_visible(False)
        axr.spines["right"].set_visible(False)
        axr.spines["left"].set_visible(True)
        axr.spines["left"].set_color(C["river"])
        axg.spines["top"].set_visible(False)
        axg.spines["left"].set_visible(False)
        axg.spines["right"].set_visible(True)
        axg.spines["right"].set_color(C["groundwater"])
        axr.set_title("Seasonal harmonic r = "
                      f"{met.seasonal_harmonic_r_same_sample:.2f}; "
                      f"residual r = {met.event_r:.2f}",
                      fontsize=GRL_TEXT_PT, color="#111111", pad=16)
        axr.set_xlim(g.date.min(), g.date.max())
        axr.xaxis.set_major_locator(mdates.MonthLocator((1, 7)))
        axr.xaxis.set_major_formatter(mdates.DateFormatter("%b\n%Y"))
        axr.tick_params(axis="x", labelsize=GRL_TEXT_PT, length=2.5)
        panel(f, axr, plab, dx=-0.080, dy=0.047)
        print(f"  example {sid}: seasonal harmonic r = "
              f"{met.seasonal_harmonic_r_same_sample:.2f}, residual correlation r = {met.event_r:.2f}")

    # (c) Complete independently daily-eligible population. Filter all three
    # same-date correlations jointly before Reach aggregation so every Reach
    # median contains exactly the same constituent daily site pairs. Monthly
    # harmonic eligibility is deliberately irrelevant here. These correlations
    # are not an additive variance/covariance decomposition.
    # Keep a dedicated gutter between the right groundwater axis in panels a/b
    # and this panel's y label.
    ax = f.add_axes([0.705, 0.14, 0.280, 0.72])
    metrics = [
        ("raw_r_same_sample", "raw_r_same_sample_median", "Observed\nvalues"),
        ("seasonal_harmonic_r_same_sample", "seasonal_harmonic_r_same_sample_median",
         "Seasonal\nharmonics"),
        ("event_r", "event_r_median", "Residuals"),
    ]
    metric_columns = [m[0] for m in metrics]
    Em = EVENT.dropna(subset=metric_columns).copy()
    if Em.empty:
        raise ValueError("complete daily three-correlation population is empty")
    rcE = Em.groupby("reach_id")[metric_columns].median().reset_index()
    same_sample_all = same_sample["ALL"]
    ci_all = same_sample_all["metric_medians"]
    if len(Em) != int(same_sample_all["n_site_pairs"]):
        raise AssertionError(
            "Fig. 1 complete-daily site-pair count does not match robustness JSON: "
            f"{len(Em)} != {same_sample_all['n_site_pairs']}"
        )
    if len(rcE) != int(same_sample_all["n_reach"]):
        raise AssertionError(
            "Fig. 1 complete-daily Reach count does not match robustness JSON: "
            f"{len(rcE)} != {same_sample_all['n_reach']}"
        )
    meds = {}
    for x0, (col, ci_key, _) in enumerate(metrics):
        vals = rcE[col].dropna().values
        vp = ax.violinplot([vals], positions=[x0], widths=0.7, showextrema=False)
        for b in vp["bodies"]:
            b.set_facecolor(C["violin"])
            b.set_alpha(0.80)
            b.set_edgecolor(C["violin_edge"])
            b.set_linewidth(0.6)
        lo, m, hi = map(float, ci_all[ci_key])
        observed_median = float(np.median(vals))
        if not np.isclose(observed_median, m, atol=5.1e-4, rtol=0):
            raise AssertionError(
                f"Fig. 1 complete-daily median mismatch for {col}: "
                f"pair table={observed_median:.6f}, robustness JSON={m:.6f}"
            )
        ax.plot([x0, x0], [lo, hi], color=C["summary"], lw=1.1, zorder=4)
        ax.scatter([x0], [m], s=18, c=C["summary"], zorder=5)
        ax.text(x0, 1.06, f"{m:.2f}", ha="center", fontsize=GRL_TEXT_PT, fontweight="bold")
        meds[col] = m
    # thin per-reach connecting lines (random subsample for legibility)
    ss = rcE.sample(min(150, len(rcE)), random_state=1)
    ax.plot(np.arange(len(metrics)), ss[metric_columns].to_numpy().T,
            color=C["neu"], alpha=0.06, lw=0.3, zorder=1)
    ax.axhline(0, color="0.6", lw=0.5, zorder=0)
    ax.set_xticks(np.arange(len(metrics)))
    ax.set_xticklabels([m[2] for m in metrics], fontsize=GRL_TEXT_PT)
    ax.set_xlim(-0.55, len(metrics) - 0.45)
    ax.set_ylabel("Correlation r")
    ax.set_ylim(-0.85, 1.18)
    ax.set_title(f"Same-date correlations\n{len(rcE)} reaches", fontsize=GRL_TEXT_PT,
                 pad=8, linespacing=1.5)
    panel(f, ax, "c", dx=-0.065, dy=0.047)

    f.savefig(HERE / "fig1_honest_metric.pdf")
    f.savefig(HERE / "fig1_honest_metric.png", dpi=600)
    plt.close(f)
    print("\nsaved fig1_honest_metric.png/.pdf | same-date Reach medians: "
          f"observed {meds['raw_r_same_sample']:.2f}, "
          f"seasonal harmonic {meds['seasonal_harmonic_r_same_sample']:.2f}, "
          f"residual correlation r {meds['event_r']:.2f} ({len(rcE)} reaches; not additive)")


if __name__ == "__main__":
    main()
