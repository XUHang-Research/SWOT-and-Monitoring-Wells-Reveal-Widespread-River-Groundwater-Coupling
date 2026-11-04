#!/usr/bin/env python3
"""Figures S9 and S10 — frequency-response ordering and
disagreement between annual-gain and residual-correlation criteria.

S9(a): The low-pass test with all four fitted amplitudes propagated through a
    paired residual block bootstrap at the actual monthly dates, alongside an
    equal-gain sampling-calendar null.
S9(b): Per-Reach classification uncertainty.
S9(c): Low-pass ordering across joint-amplitude-SNR tertiles.
S10: Per-reach annual power gain TF1 against the residual correlation
         r: the operational gain band (shaded) and the r criterion
    (r >= 0.3, dashed) label similar numbers of reaches connected but disagree
    on which reaches those are.

Inputs : data/6matched_results_monthly.txt — RAW matched monthly SWOT x well
         series (tab-separated; one row per site-month). Columns used: site_no,
         reach_id, date, river_wse (SWOT water-surface elevation, m),
         Value_EGM2008_m (groundwater level, m), distance_prior_bank_m,
         number_match_SG.
         The per-reach transfer functions are recomputed from this raw table by
         harmonic regression (annual + semiannual).
         data/08_global_coupling_metrics.parquet — monthly-harmonic-eligible
         pair table; columns used: site_no, reach_id, split.
         data/08_event_metrics_all_sites.parquet — independently daily-eligible
         pair table; columns used: site_no, reach_id, split, event_r. S10
         pairs the monthly TF and daily table on (site_no, reach_id) and requires
         only the two quantities being compared to be finite, yielding the exact
         TF1 x residual correlation intersection for the current input; S9
         remains independent of residual correlation r.
         data/12_tf_comparison.json — precomputed agreement/low-pass statistics
         (κ, agreement fraction, low-pass fractions with CIs).
         data/13_tf_lowpass_uncertainty.json/.parquet — four-amplitude
         uncertainty propagation, equal-gain null, and joint amplitude SNR.
Outputs: fig_s9_frequency_response.pdf/.png and
         fig_s10_gain_correlation.pdf/.png.

Requires: numpy, pandas, pyarrow, matplotlib.
"""
import json
from pathlib import Path
import sys

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
ANALYSIS_DIR = HERE.parent / "Phase_2_Data_Analysis" / "01_Code"
sys.path.insert(0, str(ANALYSIS_DIR))
_harmonic_model = _import_module("10_harmonic_model")
fit_harmonic = _harmonic_model.fit_harmonic
harmonic_criterion = _harmonic_model.harmonic_criterion

DATA = HERE.parent / "Phase_2_Data_Analysis" / "distance1000_data"
MONTHLY = (
    HERE.parent
    / "Phase_1_Data_Preprocessing"
    / "distance1000"
    / "6matched_results_monthly.txt"
)
DISTANCE_COLUMN = "distance_prior_bank_m"
C = dict(dev="#0072B2", hol="#D55E00", neu="#4D4D4D", black="#000000")


def panel(fig, ax, lab, dx=-0.02, dy=0.006):
    bb = ax.get_position()
    fig.text(bb.x0 + dx, bb.y1 + dy, lab, fontsize=GRL_PANEL_PT,
             fontweight="bold", va="bottom", ha="right")


# ---------------------------------------------------------------- raw input data
TFJ = json.load(open(DATA / "12_tf_comparison.json"))
TFU = json.load(open(DATA / "13_tf_lowpass_uncertainty.json"))
TFUR = pd.read_parquet(DATA / "13_tf_lowpass_uncertainty.parquet")
S = pd.read_parquet(DATA / "08_global_coupling_metrics.parquet")
EVENT = pd.read_parquet(DATA / "08_event_metrics_all_sites.parquet")
mon = pd.read_csv(MONTHLY, sep="\t", low_memory=False)
tf_rho_record = TFJ["A1_continuous"].get(
    "spearman_log10TF_annual_vs_covariability",
    TFJ["A1_continuous"].get("spearman_log10TF_annual_vs_coupling"),
)
if tf_rho_record is None:
    raise KeyError("12_tf_comparison.json is missing the annual TF-versus-r Spearman record")
print(f"=== raw input: 6matched_results_monthly.txt — {len(mon)} rows x {mon.shape[1]} cols ===")
print("columns:", ", ".join(mon.columns[:12]), "...")
print(mon[["site_no", "reach_id", "date", "river_wse", "Value_EGM2008_m", DISTANCE_COLUMN]]
      .head(6).to_string())
print(f"\n=== raw input: 08_global_coupling_metrics.parquet — {len(S)} rows ===")
print(f"=== raw input: 08_event_metrics_all_sites.parquet — {len(EVENT)} rows ===")
print("=== raw input: 12_tf_comparison.json — sections:", ", ".join(TFJ.keys()), "\n")
print("=== raw input: 13_tf_lowpass_uncertainty.json/.parquet — "
      f"{len(TFUR)} Reach rows ===\n")

required_monthly = {"site_no", "reach_id", "split"}
missing_monthly = sorted(required_monthly.difference(S.columns))
if missing_monthly:
    raise KeyError(
        f"08_global_coupling_metrics.parquet is missing fields: {missing_monthly}"
    )
required_daily = {
    "site_no",
    "reach_id",
    "event_r",
    "split",
}
missing_daily = sorted(required_daily.difference(EVENT.columns))
if missing_daily:
    raise KeyError(
        f"08_event_metrics_all_sites.parquet is missing fields: {missing_daily}"
    )
if S.site_no.duplicated().any():
    raise ValueError(
        f"08_global_coupling_metrics.parquet has {int(S.site_no.duplicated().sum())} "
        "duplicate site_no rows"
    )
if EVENT.site_no.duplicated().any():
    raise ValueError(
        f"08_event_metrics_all_sites.parquet has "
        f"{int(EVENT.site_no.duplicated().sum())} duplicate site_no rows"
    )
if not EVENT.split.isin(["dev", "holdout"]).all():
    raise ValueError("daily event population contains invalid or missing split labels")
if (EVENT.groupby("reach_id")["split"].nunique(dropna=False) > 1).any():
    raise ValueError("split is not constant within at least one daily event Reach")

# clean the raw monthly table exactly as in the archived analysis:
# numeric coercion, valid dates, match distance <= 1 km, unique station-gauge match
for c in ["river_wse", "Value_EGM2008_m", DISTANCE_COLUMN, "number_match_SG"]:
    if c in mon:
        mon[c] = pd.to_numeric(mon[c], errors="coerce")
mon["date"] = pd.to_datetime(mon["date"], errors="coerce")
m = mon.dropna(subset=["river_wse", "Value_EGM2008_m", "date"])
m = m[m[DISTANCE_COLUMN] <= 1000]
if "number_match_SG" in m:
    m = m[(m.number_match_SG == 1) | (m.number_match_SG.isna())]
del mon
print(f"cleaned monthly series: {len(m)} rows, {m.site_no.nunique()} sites")


def h2(t, x):
    """Annual + semiannual harmonic amplitudes by least squares.
    Returns (annual amplitude, semiannual amplitude) or None when the shared
    main-analysis eligibility criterion is not satisfied."""
    result = fit_harmonic(t, x)
    if result is None:
        return None
    return result['amplitudes'][0], result['amplitudes'][1]


# Per-site transfer functions TF = (well amplitude / river amplitude)^2 in the
# annual (TF1) and semiannual (TF2) bands.  A1 and A2 deliberately use different
# denominators below: A1 fixes finite TF1 x finite event_r at site level; A2
# uses every harmonic-eligible TF site independently of event_r.
rows = []
for sid, g in m.groupby("site_no"):
    g = g.sort_values("date")
    ti = pd.DatetimeIndex(g.date)
    tt = ti.year.values + (ti.dayofyear.values - 1) / 365.25
    hr = h2(tt, g.river_wse.values)
    hg = h2(tt, g.Value_EGM2008_m.values)
    if hr is None or hg is None or hr[0] <= 0 or hr[1] <= 0:
        continue
    rows.append(dict(site_no=sid, reach_id=g.reach_id.iloc[0],
                     TF1=(hg[0] / hr[0]) ** 2, TF2=(hg[1] / hr[1]) ** 2))
F = pd.DataFrame(rows)
if F.empty or F.site_no.duplicated().any():
    raise ValueError("TF refit is empty or has duplicate site_no rows")
if set(F.site_no) != set(S.site_no):
    raise ValueError(
        "TF refit and archived pair table have different site populations; "
        "rerun/repair Phase 2 before drawing Fig. S9"
    )

monthly_tf = F.merge(
    S[["site_no", "reach_id", "split"]],
    on=["site_no", "reach_id"], how="inner", validate="one_to_one",
)
if len(monthly_tf) != len(F):
    raise ValueError(
        "TF refit and monthly archive disagree on (site_no, reach_id): "
        f"only {len(monthly_tf)} of {len(F)} pairs match"
    )
if not monthly_tf.split.isin(["dev", "holdout"]).all():
    raise ValueError("monthly TF population contains invalid or missing split labels")
if (monthly_tf.groupby("reach_id")["split"].nunique(dropna=False) > 1).any():
    raise ValueError("split is not constant within at least one TF Reach")

# The only cross-scale join in this figure: pair the complete independently
# daily-eligible table with the monthly TF table on both identifiers. Keep both
# split labels until their equality has been checked explicitly.
site = monthly_tf.merge(
    EVENT[["site_no", "reach_id", "event_r", "split"]],
    on=["site_no", "reach_id"], how="inner", validate="one_to_one",
    suffixes=("_monthly", "_daily"),
)
split_mismatch = site.split_monthly.ne(site.split_daily)
if split_mismatch.any():
    raise ValueError(
        f"{int(split_mismatch.sum())} cross-scale sites disagree on the frozen split"
    )
site["split"] = site.pop("split_monthly")
site = site.drop(columns="split_daily")

# Part 2 (formerly d): finite monthly TF1 x finite daily residual correlation population,
# followed by Reach medians. No unused third metric is allowed to narrow this
# direct two-variable comparison. Fig. 1c/S1d instead use the complete daily
# population and never impose monthly eligibility.
a1_finite_columns = ["TF1", "event_r"]
a1_site = site.dropna(subset=a1_finite_columns).copy()
if not np.isfinite(a1_site[a1_finite_columns].to_numpy(dtype=float)).all():
    raise ValueError("Fig. S10 common population contains non-finite metric values")
if not a1_site.split.isin(["dev", "holdout"]).all():
    raise ValueError("Fig. S10 common population contains invalid or missing split labels")

# Exact key-set contract: the plotted sites must equal, not merely have the same
# size as, the intersection of finite monthly TF1 pairs and finite daily residual correlation
# pairs. This prevents an unnoticed third-variable filter or identifier drift.
monthly_finite = monthly_tf.loc[
    np.isfinite(monthly_tf.TF1), ["site_no", "reach_id"]
]
daily_finite = EVENT.loc[
    np.isfinite(pd.to_numeric(EVENT.event_r, errors="coerce")),
    ["site_no", "reach_id"],
]
expected_a1_keys = set(map(tuple, monthly_finite.to_numpy())) & set(
    map(tuple, daily_finite.to_numpy())
)
A1_EXPECTED_N_SITE = len({site for site, reach in expected_a1_keys})
A1_EXPECTED_N_REACH = len({reach for site, reach in expected_a1_keys})
actual_a1_keys = set(map(tuple, a1_site[["site_no", "reach_id"]].to_numpy()))
if actual_a1_keys != expected_a1_keys:
    raise ValueError(
        "Fig. S10 key set is not the exact finite TF1 x finite residual correlation intersection: "
        f"missing={len(expected_a1_keys - actual_a1_keys)}, "
        f"extra={len(actual_a1_keys - expected_a1_keys)}"
    )
T = (a1_site.groupby("reach_id")
       .agg(TF1=("TF1", "median"), event_r=("event_r", "median"),
            split=("split", "first"), n_site_common=("site_no", "size"))
       .reset_index())

# A2: complete TF population, with no residual correlation requirement.
T_full = (monthly_tf.dropna(subset=["TF1", "TF2"])
            .groupby("reach_id")
            .agg(TF1=("TF1", "median"), TF2=("TF2", "median"),
                 split=("split", "first"), n_site_TF=("site_no", "size"))
            .reset_index())

# Blocking figure/data contract: all plotted A1 values and displayed A2
# summaries must reproduce the Phase-2 JSON generated from the same rules.
den = TFJ.get("denominators")
if den is None:
    raise KeyError("12_tf_comparison.json lacks explicit denominators; rerun Phase 2 step 7")
expected_counts = {
    "A1_common_site": (len(a1_site), len(T)),
    "A2_full_TF": (len(monthly_tf), len(T_full)),
}
for key, (n_site, n_reach) in expected_counts.items():
    rec = den.get(key, {})
    if int(rec.get("n_site", -1)) != n_site or int(rec.get("n_reach", -1)) != n_reach:
        raise ValueError(
            f"Fig. S9 {key} denominator {n_site} sites/{n_reach} reaches does not "
            f"match 12_tf_comparison.json ({rec.get('n_site')}/{rec.get('n_reach')})"
        )

# Current-input cross-scale contract. Derive the denominator from the exact
# independently reconstructed key intersection, not a previous input's count.
if len(a1_site) != A1_EXPECTED_N_SITE or len(T) != A1_EXPECTED_N_REACH:
    raise ValueError(
        "Fig. S10 must use the current monthly-TF x daily-event common sample: "
        f"got {len(a1_site)} sites/{len(T)} reaches, expected "
        f"{A1_EXPECTED_N_SITE}/{A1_EXPECTED_N_REACH}"
    )
a1_contract = den.get("A1_common_site", {})
if (int(a1_contract.get("n_site", -1)) != A1_EXPECTED_N_SITE
        or int(a1_contract.get("n_reach", -1)) != A1_EXPECTED_N_REACH):
    raise ValueError(
        "12_tf_comparison.json does not preserve the current Fig. S10 "
        f"{A1_EXPECTED_N_SITE}-site/{A1_EXPECTED_N_REACH}-Reach contract"
    )
required_a1_contract = {
    "monthly_source_table": "08_global_coupling_metrics.parquet",
    "daily_source_table": "08_event_metrics_all_sites.parquet",
    "intersection_keys": ["site_no", "reach_id"],
    "site_filter": "finite TF1 and finite event_r",
    "reach_aggregation": "median after site-level intersection",
}
contract_mismatches = {
    key: (a1_contract.get(key), expected)
    for key, expected in required_a1_contract.items()
    if a1_contract.get(key) != expected
}
if contract_mismatches:
    raise ValueError(
        "12_tf_comparison.json has a stale Fig. S10 data contract: "
        f"{contract_mismatches}; rerun Phase 2 step 7"
    )

band_check = TFJ["A1_agreement"]["TF_band_0.250_4.000"]
conn_tf = T.TF1.between(0.25, 4.0).to_numpy()
conn_r = T.event_r.ge(0.3).to_numpy()
agreement = float(np.mean(conn_tf == conn_r))
p_tf = float(conn_tf.mean())
p_r = float(conn_r.mean())
chance = p_tf * p_r + (1 - p_tf) * (1 - p_r)
kappa = float((agreement - chance) / (1 - chance)) if chance < 1 else np.nan
rho = float(T.TF1.clip(lower=1e-9).rank().corr(T.event_r.rank()))
checks = {
    "agreement": (agreement, band_check["agreement"][1]),
    "kappa": (kappa, band_check["kappa"][1]),
    "rho": (rho, tf_rho_record["rho"]),
}
for label, (actual, expected) in checks.items():
    if not np.isclose(actual, expected, atol=1e-12, rtol=1e-12):
        raise ValueError(
            f"Fig. S10 A1 {label}={actual:.12g} does not reproduce JSON={expected:.12g}"
        )

lp_json = TFJ["A2_lowpass"]
for key, sub in [("ALL", T_full), ("DEV", T_full[T_full.split == "dev"]),
                 ("HOLDOUT", T_full[T_full.split == "holdout"])]:
    if key not in lp_json:
        continue
    actual_n = int(len(sub))
    actual_frac = float((sub.TF2 < sub.TF1).mean())
    expected_n = int(lp_json[key]["n_reach"])
    expected_frac = float(lp_json[key]["frac_TF2_lt_TF1"][1])
    if actual_n != expected_n or not np.isclose(
            actual_frac, expected_frac, atol=1e-12, rtol=1e-12):
        raise ValueError(
            f"Fig. S9 A2 {key} does not reproduce JSON: "
            f"n={actual_n}/{expected_n}, fraction={actual_frac:.12g}/{expected_frac:.12g}"
        )

print(f"A1 common population: {len(a1_site)} sites, {len(T)} reaches")
print(f"A2 full-TF population: {len(monthly_tf)} sites, {len(T_full)} reaches")
print("Figs. S9/S10 denominators and displayed statistics reproduce 12_tf_comparison.json")
print("harmonic eligibility:", harmonic_criterion())

if int(TFU.get("provenance", {}).get("n_site", -1)) != len(monthly_tf):
    raise ValueError("TF uncertainty site denominator does not match full TF population")
if int(TFU.get("provenance", {}).get("n_reach", -1)) != len(T_full):
    raise ValueError("TF uncertainty Reach denominator does not match full TF population")
if len(TFUR) != len(T_full) or set(TFUR.reach_id) != set(T_full.reach_id):
    raise ValueError("TF uncertainty Reach keys do not match the full TF population")

# Formal figures: separate the two scientific questions, preserving all data contracts
# above and all archived point estimates, nulls and bootstrap intervals.
from matplotlib.lines import Line2D
from matplotlib.patches import Patch

band = TFJ["A1_agreement"]["TF_band_0.250_4.000"]
lp = TFJ["A2_lowpass"]
unc = TFU["summary"]
frequency = lp["frequency_response"]
well_ratio = frequency["median_well_A2_over_A1"][1]
river_ratio = frequency["median_river_A2_over_A1"][1]
null_all = unc["ALL"]["equal_gain_sampling_null_fraction"]

# Figure S9: frequency ordering, uncertainty and SNR diagnostics.
f = plt.figure(figsize=grl_figsize(170, 130))
axb = f.add_axes([0.15, 0.54, 0.34, 0.29])
axc = f.add_axes([0.60, 0.54, 0.37, 0.29])
axe = f.add_axes([0.15, 0.14, 0.82, 0.24])
labs, vals, los, his, cols = [], [], [], [], []
for key, col in [("ALL", C["neu"]), ("DEV", C["dev"]), ("HOLDOUT", C["hol"])]:
    bounds = unc[key]["uncertainty_propagated_fraction_TF2_lt_TF1"]
    labs.append({"ALL": "All", "DEV": "Development\ngroup",
                 "HOLDOUT": "Geographic\ntesting group"}[key])
    los.append(bounds[0]); vals.append(bounds[1]); his.append(bounds[2]); cols.append(col)
y = np.arange(len(labs))[::-1]
for i, key in enumerate(("ALL", "DEV", "HOLDOUT")):
    axb.errorbar(vals[i], y[i], xerr=[[vals[i]-los[i]], [his[i]-vals[i]]],
                 fmt="^" if key == "HOLDOUT" else "o", color=cols[i],
                 ms=4, lw=0.9, capsize=2, capthick=0.7, zorder=3)
    axb.text(his[i]+0.010, y[i], f"{vals[i]:.0%}", va="center", color=cols[i])
axb.axvline(0.5, color="0.65", lw=0.55, ls=":", zorder=0)
axb.axvspan(null_all[0], null_all[2], color="#E69F00", alpha=0.16, lw=0, zorder=0)
axb.axvline(null_all[1], color="#E69F00", lw=0.8, ls="--", zorder=1)
axb.set_yticks(y, labs)
axb.set_ylim(-0.6, 2.6)
axb.set_xlim(0.40, 0.82)
axb.set_xlabel("Fraction of reaches\nwith TF$_2$ < TF$_1$")
axb.set_title("Frequency ordering")
f.legend(handles=[Line2D([], [], color="0.35", marker="o", ms=3.5, lw=0.8,
                           label="Propagated (95% CI)"),
                     Patch(facecolor="#E69F00", alpha=0.25, edgecolor="none",
                           label="Equal-gain null (95% interval)")],
            loc="upper center", bbox_to_anchor=(0.53, 0.95), ncol=2,
            frameon=False, handletextpad=0.4, labelspacing=0.25, borderaxespad=0)
panel(f, axb, "a", dx=-0.125, dy=0.018)

# Class fractions are connected in support order; each line is one population.
# These are categorical summaries, not a time series or a fitted response curve.
class_keys = [
    ("robust_lowpass_fraction_p_ge_0p95", "Robust\nlow-pass"),
    ("uncertain_fraction", "Uncertain"),
    ("robust_opposite_fraction_p_le_0p05", "Robust\nopposite"),
]
class_values = np.array([[unc[k][key] for key, _ in class_keys]
                         for k in ("ALL", "DEV", "HOLDOUT")])
group_styles = [
    ("ALL", "All reaches", C["neu"], "o", "--"),
    ("DEV", "Development group", C["dev"], "o", "-"),
    ("HOLDOUT", "Geographic testing group", C["hol"], "^", "-."),
]
for row_index, (key, label, colour, marker, linestyle) in enumerate(group_styles):
    # Pale connectors show category order; opaque markers carry the estimates.
    axc.plot(range(3), class_values[row_index],
             color=mpl.colors.to_rgba(colour, 0.45), marker=marker,
             markerfacecolor=colour, markeredgecolor="white", markeredgewidth=0.35,
             ls=linestyle, ms=4.6, lw=0.65, label=label, zorder=3)
axc.set_xlim(-0.2, 2.2)
axc.set_ylim(-0.025, 1.025)
axc.set_xticks(range(3), [label for _, label in class_keys])
axc.set_yticks([0, 0.25, 0.5, 0.75, 1.0])
axc.yaxis.set_major_formatter(mpl.ticker.PercentFormatter(xmax=1, decimals=0))
axc.set_ylabel("Fraction of reaches")
axc.set_title("Classification uncertainty")
axc.grid(axis="y", color="0.90", lw=0.5, zorder=0)
handles, labels = axc.get_legend_handles_labels()
f.legend(handles, labels, loc="upper center", bbox_to_anchor=(0.53, 0.995),
          ncol=3, handletextpad=0.35, columnspacing=1.1, borderaxespad=0,
          frameon=False)
panel(f, axc, "b", dx=-0.045, dy=0.018)

snr_rows = unc["ALL"]["joint_amplitude_snr_tertiles"]
snr_colours = ["#5B9AC2", "#2878AA", "#08519C"]
for index, (row, colour) in enumerate(zip(snr_rows, snr_colours)):
    lo_s, md_s, hi_s = row["uncertainty_propagated_fraction_TF2_lt_TF1"]
    axe.errorbar(index, md_s, yerr=[[md_s-lo_s], [hi_s-md_s]], fmt="o",
                 color=colour, ms=4, lw=0.9, capsize=2, capthick=0.7, zorder=3)
    axe.annotate(f"{md_s:.0%}", (index, md_s), xytext=(7, 0),
                  textcoords="offset points", va="center", color=colour)
axe.axhline(0.5, color="0.6", lw=0.55, ls=":", zorder=0)
axe.set_xticks(range(3), [f"{name} (n = {row['n_reach']})"
                        for name, row in zip(["Low", "Middle", "High"], snr_rows)])
axe.set_xlim(-0.45, 2.45)
axe.set_ylim(0.50, 0.82)
axe.set_ylabel("Fraction of reaches\nwith TF$_2$ < TF$_1$")
axe.set_xlabel("Joint amplitude SNR tertile")
axe.set_title("SNR sensitivity")
panel(f, axe, "c", dx=-0.125, dy=0.018)
f.savefig(HERE / "fig_s9_frequency_response.pdf")
f.savefig(HERE / "fig_s9_frequency_response.png", dpi=600)
plt.close(f)

# Figure S10: exact finite monthly TF1 × finite daily-r intersection, unchanged.
# Keep a 60-mm data panel and a compact, title-free summary strip.
f2 = plt.figure(figsize=grl_figsize(120, 92))
ax = f2.add_axes([0.14, 15 / 92, 0.82, 60 / 92])
lo, hi = 0.25, 4.0
d = T.dropna(subset=["TF1", "event_r"])
TFLO, TFHI, RLO, RHI = 1e-4, 2e3, -0.75, 1.0
# Retain the full gain range rather than projecting extreme values onto edges.
outside = ~d.TF1.between(TFLO, TFHI) | ~d.event_r.between(RLO, RHI)
if outside.any():
    raise ValueError(f"Fig. S10: expand the axes to include {int(outside.sum())} Reaches")
ax.axvspan(lo, hi, color="#DCE6F0", alpha=0.65, zorder=0)
ax.axhline(0.3, color="0.45", lw=0.65, ls="--", zorder=2)
for sp, col, mk, area in [("dev", C["dev"], "o", 7), ("holdout", C["hol"], "^", 9)]:
    g = d[d.split == sp]
    ax.scatter(g.TF1, g.event_r, s=area, color=col, marker=mk,
                alpha=0.55, lw=0, zorder=3,
                label="Development group" if sp == "dev" else "Geographic testing group")
ax.set_xscale("log")
ax.set_xlim(TFLO, TFHI)
ax.set_ylim(RLO, RHI)
ax.set_xlabel(r"Annual power gain $\mathrm{TF}_1$")
ax.set_ylabel("Residual correlation r")
ax.text(np.sqrt(lo * hi), 0.03, "Gain band", transform=ax.get_xaxis_transform(),
         ha="center", va="bottom", color="#40608A")
ax.text(0.035, 0.33, "r ≥ 0.3", transform=ax.get_yaxis_transform(), color="0.3")
handles, labels = ax.get_legend_handles_labels()
f2.legend(handles, labels, loc="upper left", bbox_to_anchor=(0.14, 0.98),
           ncol=1, frameon=False, handletextpad=0.2, labelspacing=0.5,
           markerscale=1.6, borderaxespad=0)
# A dedicated top-right strip avoids covering any observations.
f2.text(0.96, 0.98,
         f"Agreement {band['agreement'][1]:.0%}\n"
         f"$\\kappa$ = {band['kappa'][1]:.2f}\n"
         f"Spearman $\\rho$ = {tf_rho_record['rho']:+.2f}",
         ha="right", va="top", linespacing=1.25)
f2.savefig(HERE / "fig_s10_gain_correlation.pdf")
f2.savefig(HERE / "fig_s10_gain_correlation.png", dpi=600)
plt.close(f2)
print(f"saved Figs. S9 and S10 | common sample = {len(T)} reaches; "
      f"full TF sample = {len(T_full)} reaches; κ = {kappa:.2f}; "
      f"agreement = {agreement:.0%}; rho = {rho:+.2f}")
