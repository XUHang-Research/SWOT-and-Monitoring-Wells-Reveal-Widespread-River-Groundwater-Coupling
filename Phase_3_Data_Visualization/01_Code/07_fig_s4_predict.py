#!/usr/bin/env python3
"""Figure S4 — weak associations do not yield transferable prediction of coupling.

The figure separates three kinds of static descriptors that were previously
described too broadly as "Reach properties":

* Reach attributes: river width, Reach slope2 and prior width variance;
* well attribute: groundwater depth below land surface;
* well–Reach pairing attribute: prior well-to-bank distance.

Panel (a) evaluates a five-descriptor linear model on the predefined
rest-of-world holdout after training on Germany + China. Panels (b–d) use the
same holdout Reach sample to show the weak univariate associations that can
coexist with poor multivariate out-of-region prediction.

Input  : ../Phase_2_Data_Analysis/distance1000_data/
         08_event_covariates_all_sites.parquet — independently daily-eligible
         residual correlation sites plus the five static descriptors; monthly-harmonic
         eligibility is not imposed.
Output : fig_s4_predict.png (600 dpi) and fig_s4_predict.pdf.

Requires: numpy, pandas, pyarrow, matplotlib, scipy.
"""
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats as sps
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
# Okabe–Ito colours: accessible in common colour-vision deficiencies.
C = dict(hold="#D55E00", trend="#0072B2", reach="#009E73",
         neutral="#595959", pale="#BDBDBD", black="#000000")
RNG_SEED = 20250311
N_BOOT = 10_000

# ---------------------------------------------------------------- raw input data
M = pd.read_parquet(DATA / "08_event_covariates_all_sites.parquet")
print(f"=== raw input: 08_event_covariates_all_sites.parquet — {len(M)} rows x {M.shape[1]} cols ===")
required = {"site_no", "reach_id", "country", "event_r", "river_width", "depth_m",
            "slope2", "p_wid_var", "distance_prior_bank_m", "split"}
missing = sorted(required.difference(M.columns))
if missing:
    raise KeyError(f"08_mechanism.parquet is missing required Fig. S4 fields: {missing}")

# One independent plotting/model row per Reach.  First freeze the finite
# residual correlation sites so every predictor median is computed only from wells that also
# contribute to that Reach's response median.  Grouping all wells first would
# allow a multi-well Reach's predictors and event_r to come from different
# constituent site populations.
event_site = M[np.isfinite(pd.to_numeric(M.event_r, errors="coerce"))].copy()
if event_site.empty:
    raise ValueError("finite event_r site population is empty")
if event_site.site_no.duplicated().any():
    raise ValueError(
        f"08_event_covariates_all_sites.parquet has "
        f"{int(event_site.site_no.duplicated().sum())} "
        "duplicate finite-event site rows"
    )
if (event_site.groupby("reach_id")["split"].nunique(dropna=True) > 1).any():
    raise ValueError("split is not constant within at least one finite-event Reach")
# Freeze finite-depth sites before aggregation so predictors and response
# share the same wells when a source depth is unresolved.
event_site = event_site.dropna(subset=['depth_m']).copy()
R = (event_site.groupby("reach_id")
       .agg(event_r=("event_r", "median"),
            width=("river_width", "median"),
            slope2=("slope2", "median"),
            p_wid_var=("p_wid_var", "median"),
            depth=("depth_m", "median"),
            distance_prior_bank_m=("distance_prior_bank_m", "median"),
            split=("split", "first"))
       .reset_index()
       .dropna(subset=["event_r"]))
if len(R) != event_site.reach_id.nunique():
    raise RuntimeError("finite-event Reach aggregation changed the Reach denominator")
print(f"fixed finite-event population: {len(event_site)} sites, {len(R)} Reaches")
feat = ["width", "slope2", "p_wid_var", "depth", "distance_prior_bank_m"]
dev = R[R.split == "dev"].dropna(subset=feat + ["event_r"]).copy()
hold = R[R.split == "holdout"].dropna(subset=feat + ["event_r"]).copy()
if dev.empty or hold.empty:
    raise ValueError(f"complete-case model sample is empty: dev={len(dev)}, holdout={len(hold)}")
print(f"Reach-level complete cases: development={len(dev)}, holdout={len(hold)}")


def raw_features(df):
    """Scientifically interpretable transforms before dev-only scaling."""
    return np.column_stack([
        np.log10(df.width.clip(lower=1e-6)),
        df.slope2.to_numpy(float),
        np.log10(1.0 + df.p_wid_var.clip(lower=0.0)),
        df.depth.to_numpy(float),
        df.distance_prior_bank_m.to_numpy(float) / 1000.0,
    ])


feature_names = ["log10(width)", "slope2", "log10(1+p_wid_var)",
                 "depth", "bank_distance_km"]


def fit_predict(train, test):
    """Fit the specified OLS model, scaling with training Reaches only."""
    xtr = raw_features(train)
    xte = raw_features(test)
    mu = xtr.mean(axis=0)
    sd = xtr.std(axis=0)
    if np.any(~np.isfinite(sd)) or np.any(sd <= 0):
        raise ValueError("invalid/non-varying predictor in a development sample")
    atr = np.column_stack([np.ones(len(xtr)), (xtr - mu) / sd])
    ate = np.column_stack([np.ones(len(xte)), (xte - mu) / sd])
    beta, *_ = np.linalg.lstsq(atr, train.event_r.to_numpy(float), rcond=None)
    return ate @ beta, beta


def prediction_scores(actual, predicted):
    """Out-of-sample R2 and rank correlation."""
    actual = np.asarray(actual, float)
    predicted = np.asarray(predicted, float)
    ss_res = float(np.sum((actual - predicted) ** 2))
    ss_tot = float(np.sum((actual - actual.mean()) ** 2))
    r2 = 1 - ss_res / ss_tot
    rho, p = sps.spearmanr(predicted, actual)
    return float(r2), float(rho), float(p)


pred, beta = fit_predict(dev, hold)
act = hold.event_r.to_numpy(float)
r2_oos, rho_pred, p_pred = prediction_scores(act, pred)
dpred, _ = fit_predict(dev, dev)
r2_in, _, _ = prediction_scores(dev.event_r.to_numpy(float), dpred)
print("OLS residual correlation r ~ 1 + " + " + ".join(feature_names) +
      f" | trained on development (n={len(dev)}), tested on holdout (n={len(hold)})")
print("standardized development coefficients: " + ", ".join(
    f"{name}={coef:+.3f}" for name, coef in zip(feature_names, beta[1:])))
print(f"out-of-sample R2={r2_oos:.3f}; Spearman rho={rho_pred:+.3f} (p={p_pred:.3f}); "
      f"development in-sample R2={r2_in:.3f}")


def bootstrap_prediction_ci(train, test, n_boot=N_BOOT, seed=RNG_SEED):
    """Reach bootstrap of the entire train-and-test prediction procedure."""
    rng = np.random.default_rng(seed)
    scores = np.full((n_boot, 2), np.nan)
    for i in range(n_boot):
        tr = train.iloc[rng.integers(0, len(train), len(train))]
        te = test.iloc[rng.integers(0, len(test), len(test))]
        try:
            pb, _ = fit_predict(tr, te)
            scores[i, :2] = prediction_scores(te.event_r.to_numpy(float), pb)[:2]
        except (ValueError, np.linalg.LinAlgError):
            continue
    return np.nanquantile(scores, [0.025, 0.975], axis=0)


def spearman_with_ci(x, y, n_boot=N_BOOT, seed=RNG_SEED):
    """Reach-bootstrap CI for a univariate Spearman association."""
    x = np.asarray(x, float)
    y = np.asarray(y, float)
    keep = np.isfinite(x) & np.isfinite(y)
    x, y = x[keep], y[keep]
    rho, p = sps.spearmanr(x, y)
    rng = np.random.default_rng(seed)
    boot = np.full(n_boot, np.nan)
    for i in range(n_boot):
        idx = rng.integers(0, len(x), len(x))
        boot[i] = sps.spearmanr(x[idx], y[idx]).statistic
    lo, hi = np.nanquantile(boot, [0.025, 0.975])
    return float(rho), float(p), float(lo), float(hi), int(len(x))


def quantile_summary(x, y, n_bins=8):
    """Equal-count bins; return median x/y and interquartile y range."""
    d = pd.DataFrame({"x": np.asarray(x, float), "y": np.asarray(y, float)}).dropna()
    d["bin"] = pd.qcut(d.x, q=n_bins, duplicates="drop")
    return (d.groupby("bin", observed=True)
             .agg(x=("x", "median"), y=("y", "median"),
                  y25=("y", lambda z: z.quantile(0.25)),
                  y75=("y", lambda z: z.quantile(0.75)))
             .reset_index(drop=True))


pred_ci = bootstrap_prediction_ci(dev, hold)
print(f"bootstrap 95% CI: R2 [{pred_ci[0, 0]:.3f}, {pred_ci[1, 0]:.3f}], "
      f"prediction rho [{pred_ci[0, 1]:.3f}, {pred_ci[1, 1]:.3f}]")

# Every right-side statistic uses exactly the same complete-case holdout
# Reaches as panel a. This makes the five descriptors directly comparable.
assoc = {
    "depth": spearman_with_ci(hold.depth, hold.event_r, seed=RNG_SEED + 1),
    "distance": spearman_with_ci(hold.distance_prior_bank_m, hold.event_r,
                                  seed=RNG_SEED + 2),
    "width": spearman_with_ci(hold.width, hold.event_r, seed=RNG_SEED + 3),
    "slope2": spearman_with_ci(hold.slope2, hold.event_r, seed=RNG_SEED + 4),
    "p_wid_var": spearman_with_ci(hold.p_wid_var, hold.event_r, seed=RNG_SEED + 5),
}
for name, (rho, p, lo_ci, hi_ci, n) in assoc.items():
    print(f"holdout {name}: rho={rho:+.3f}, 95% CI [{lo_ci:+.3f}, {hi_ci:+.3f}], "
          f"p={p:.3g}, n={n}")

# ---------------------------------------------------------------- figure
fig = plt.figure(figsize=grl_figsize(170, 120))
gs = fig.add_gridspec(
    3, 2, width_ratios=[1.18, 1.15],
    left=0.105, right=0.970, bottom=0.140, top=0.945,
    wspace=0.40, hspace=0.78,
)
ax_a = fig.add_subplot(gs[:, 0])
ax_b = fig.add_subplot(gs[0, 1])
ax_c = fig.add_subplot(gs[1, 1])
ax_d = fig.add_subplot(gs[2, 1])

# (a) Five-descriptor transfer to the predefined holdout.
lo, hi = -0.65, 1.02
ax_a.plot([lo, hi], [lo, hi], ls="--", color=C["black"], lw=0.8,
          zorder=1, label="Perfect prediction (1:1)")
ax_a.axhline(float(act.mean()), color=C["neutral"], lw=0.8, ls=":", zorder=1)
ax_a.text(hi - 0.02, float(act.mean()) + 0.02, "Geographic\ntesting group\nmean",
          ha="right", va="bottom", fontsize=GRL_TEXT_PT, color=C["neutral"])
ax_a.scatter(pred, act, s=10, color=C["hold"], alpha=0.58, lw=0, zorder=3,
             rasterized=True)
ax_a.set_xlim(lo, hi)
ax_a.set_ylim(lo, hi)
ax_a.set_aspect("equal", adjustable="box")
ax_a.set_xlabel("Predicted residual correlation r")
ax_a.set_ylabel("Observed residual correlation r")
ax_a.text(
    1.0, 1.055,
    f"$R^2_{{OOS}}$ = {r2_oos:.2f} "
    f"[{pred_ci[0, 0]:.2f}, {pred_ci[1, 0]:.2f}]\n"
    f"Spearman $\\rho$ = {rho_pred:.2f} ($p$ = {p_pred:.2f})",
    transform=ax_a.transAxes, fontsize=GRL_TEXT_PT, ha="right", va="bottom",
)
ax_a.legend(loc="upper center", bbox_to_anchor=(0.5, -0.19),
            fontsize=GRL_TEXT_PT, handletextpad=0.4, borderaxespad=0.25, frameon=False)


def plot_descriptor(ax, x, y, stat, xlabel, symlog=False):
    """Raw holdout Reaches plus an equal-count-bin median/IQR summary."""
    rho, p, lo_ci, hi_ci, n = stat
    ax.scatter(x, y, s=6, color=C["hold"], alpha=0.25, lw=0,
               zorder=1, rasterized=True)
    q = quantile_summary(x, y)
    yerr = np.vstack([q.y - q.y25, q.y75 - q.y])
    ax.errorbar(q.x, q.y, yerr=yerr, fmt="o-", color=C["trend"],
                mec="white", mew=0.35, ms=3.1, lw=0.9, capsize=1.5,
                zorder=3)
    ax.axhline(0, color=C["pale"], lw=0.55, ls=":", zorder=0)
    if symlog:
        # Preserve legitimate negative/artesian values and very deep wells.
        ax.set_xscale("symlog", linthresh=3, linscale=0.75, base=10)
        ax.set_xticks([-30, 0, 10, 100, 400])
        ax.set_xticklabels(["−30", "0", "10", "100", "400"])
    ax.set_ylim(-0.68, 1.02)
    ax.set_yticks([-0.5, 0, 0.5, 1.0])
    ax.set_xlabel(xlabel, labelpad=1.5)
    ax.set_ylabel("Residual\ncorrelation r", labelpad=1.5)
    # Match panel a: statistics above the data, aligned with the right edge.
    ax.text(1.0, 1.055,
            f"$\\rho$ = {rho:+.2f} [{lo_ci:+.2f}, {hi_ci:+.2f}]",
            transform=ax.transAxes, ha="right", va="bottom", fontsize=GRL_TEXT_PT,
            clip_on=False)


# (b) Well property and (c) well–Reach pairing property.
plot_descriptor(ax_b, hold.depth, hold.event_r, assoc["depth"],
                "Depth below land surface (m)", symlog=True)
plot_descriptor(ax_c, hold.distance_prior_bank_m, hold.event_r, assoc["distance"],
                "Prior bank distance (m)")
ax_c.set_xlim(0, 1000)
ax_c.set_xticks([0, 250, 500, 750, 1000])

# (d) The three actual Reach attributes, shown on a common effect-size axis.
reach_vars = ["width", "slope2", "p_wid_var"]
reach_labels = ["River width", "Reach slope", "Prior width\nvariance"]
ypos = np.arange(len(reach_vars))[::-1]
for y0, var in zip(ypos, reach_vars):
    rho, p, lo_ci, hi_ci, n = assoc[var]
    ax_d.plot([lo_ci, hi_ci], [y0, y0], color=C["reach"], lw=1.6,
              solid_capstyle="round")
    ax_d.plot(rho, y0, "o", color=C["reach"], mec="white", mew=0.45,
              ms=4.4, zorder=3)
ax_d.axvline(0, color=C["neutral"], lw=0.65, ls=":", zorder=0)
ax_d.set_yticks(ypos)
ax_d.set_yticklabels(reach_labels)
ax_d.set_xlim(-0.23, 0.28)
ax_d.set_xticks([-0.2, -0.1, 0, 0.1, 0.2])
ax_d.set_ylim(-0.55, 2.55)
ax_d.set_xlabel("Spearman $\\rho$ (95% CI)", labelpad=1.5)

# Panel letters are placed in the complete allocated panel margins, not on data.
for ax, letter in zip([ax_a, ax_b, ax_c, ax_d], "abcd"):
    y_letter = 1.015 if ax is ax_a else 1.055
    ax.text(-0.115 if ax is ax_a else -0.16, y_letter, letter,
            transform=ax.transAxes, fontsize=GRL_PANEL_PT, fontweight="bold",
            fontstyle="normal", ha="left", va="bottom", clip_on=False)

fig.savefig(HERE / "fig_s4_predict.pdf")
fig.savefig(HERE / "fig_s4_predict.png", dpi=600)
plt.close(fig)
print(f"saved fig_s4_predict.png/.pdf | R2_oos={r2_oos:.3f}, "
      f"rho_pred={rho_pred:+.3f}, n_holdout={len(hold)}")
