#!/usr/bin/env python3
"""Step 5 — two autocorrelation-preserving nulls for sub-seasonal co-variability.

Tests whether the observed harmonic-residual co-variability is above
chance given autocorrelation: per site, circularly shift the groundwater
residual relative to the river residual and recompute the correlation B times;
aggregate to a reach-median null and compare to the observed reach-median (and
the fraction >= 0.3). A second, time-aware null fits a continuous-time AR(1)
(Ornstein-Uhlenbeck correlation) to each marginal residual series and simulates
independent river and groundwater surrogates at the exact observed dates,
then residualizes each surrogate with the same six-column harmonic model
as the observed series before standardizing and correlating. The
resampling units are frozen to the complete
independent daily event-r sites, Reach IDs, and Reach-level dev/holdout split
before the daily table is read.  A separate fixed-seed Reach bootstrap gives
confidence intervals for the two observed statistics.
Memory-safe chunked daily read (<= 1 km).

Inputs : Phase-1 5matched_results_daily.txt
         output/08_event_metrics_all_sites.parquet (standalone daily sample)
Outputs: merges an 'event_permutation_null' section into
         output/08_robustness.json (from step 4)

Requires: numpy, pandas.
"""
import collections
import gc
import json
import numpy as np
import pandas as pd

from importlib import import_module as _import_module
_pipeline_config = _import_module("09_pipeline_config")
DAY = _pipeline_config.DAY
DISTANCE_COLUMN = _pipeline_config.DISTANCE_COLUMN
MAX_DISTANCE_KM = _pipeline_config.MAX_DISTANCE_KM
MAX_DISTANCE_M = _pipeline_config.MAX_DISTANCE_M
OUT = _pipeline_config.OUT
_harmonic_model = _import_module("10_harmonic_model")
harmonic_design = _harmonic_model.harmonic_design
residualize_harmonic_draws = _harmonic_model.residualize_harmonic_draws
B = 10_000
RNG = np.random.default_rng(0)
TIME_AWARE_RNG = np.random.default_rng(20260812)
REPRODUCTION_TOL = 1e-10
OBS_BOOTSTRAP_N = 10_000
OBS_BOOTSTRAP_SEED = 20260808


def yfrac(d):
    d = pd.DatetimeIndex(d)
    return d.year.values + (d.dayofyear.values - 1) / 365.25


def hres(t, x, fr=(1., 2.)):
    M = harmonic_design(t, frequencies=fr)
    b, *_ = np.linalg.lstsq(M, x, rcond=None)
    return x - M @ b


def fit_ou_timescale_days(t_years, residual):
    """Estimate an irregular-time OU correlation scale by profile likelihood.

    The residual variance is profiled out. A logarithmic grid is deliberately
    used instead of a fragile optimizer because some short series have nearly
    flat likelihood surfaces. The bounds are reported in the output contract.
    """
    t_days = np.asarray(t_years, float) * 365.25
    x = np.asarray(residual, float)
    order = np.argsort(t_days)
    t_days, x = t_days[order], x[order]
    dt = np.diff(t_days)
    if len(x) < 3 or not np.all(dt > 0):
        return None
    x = (x - x.mean()) / x.std()
    grid = np.geomspace(0.5, 730.5, 240)
    scores = np.full(len(grid), np.inf)
    for index, tau in enumerate(grid):
        phi = np.exp(-dt / tau)
        innovation = x[1:] - phi * x[:-1]
        variance_factor = np.maximum(1.0 - phi * phi, 1e-12)
        sigma2 = np.mean(np.square(innovation) / variance_factor)
        if sigma2 <= 0 or not np.isfinite(sigma2):
            continue
        scores[index] = (
            len(innovation) * np.log(sigma2)
            + np.sum(np.log(variance_factor))
        )
    if not np.isfinite(scores).any():
        return None
    return float(grid[int(np.nanargmin(scores))])


def simulate_ou_at_dates(t_years, tau_days, n_draw, rng):
    """Simulate standardized OU surrogates at the exact irregular dates."""
    t_days = np.asarray(t_years, float) * 365.25
    order = np.argsort(t_days)
    inverse = np.argsort(order)
    sorted_days = t_days[order]
    dt = np.diff(sorted_days)
    draws = np.empty((int(n_draw), len(t_days)), dtype=float)
    draws[:, 0] = rng.standard_normal(int(n_draw))
    for index, gap in enumerate(dt, start=1):
        phi = float(np.exp(-gap / tau_days))
        draws[:, index] = (
            phi * draws[:, index - 1]
            + np.sqrt(max(1.0 - phi * phi, 0.0))
            * rng.standard_normal(int(n_draw))
        )
    draws = draws[:, inverse]
    draws -= draws.mean(axis=1, keepdims=True)
    scale = draws.std(axis=1, keepdims=True)
    return np.divide(draws, scale, out=np.zeros_like(draws), where=scale > 0)


# Freeze the complete independent daily population first.  The null is
# recomputed from the daily records for exactly these sites; monthly harmonic
# eligibility is deliberately irrelevant to this standalone event-r result.
event_file = OUT / "08_event_metrics_all_sites.parquet"
event_population = pd.read_parquet(
    event_file,
    columns=["site_no", "reach_id", "split", "event_r"],
).dropna(subset=["event_r"]).copy()
if event_population["site_no"].duplicated().any():
    duplicates = int(event_population["site_no"].duplicated(keep=False).sum())
    raise ValueError(
        f"independent daily sample has {duplicates:,} duplicated site_no rows"
    )
if not event_population["split"].isin(["dev", "holdout"]).all():
    raise ValueError(
        "independent daily sample contains an invalid or missing frozen split"
    )
if not event_population.groupby("reach_id").split.nunique(dropna=False).eq(1).all():
    raise ValueError("a Reach appears in multiple independent daily splits")

event_population = event_population.set_index("site_no", verify_integrity=True)
event_sites = frozenset(event_population.index)
expected_reaches = int(event_population["reach_id"].nunique())
expected_dev_reaches = int(
    event_population.loc[
        event_population["split"].eq("dev"), "reach_id"
    ].nunique()
)
expected_holdout_reaches = int(
    event_population.loc[
        event_population["split"].eq("holdout"), "reach_id"
    ].nunique()
)
print(
    f"freeze independent daily event_r sample: {len(event_population):,} sites, "
    f"{expected_reaches:,} reaches, {expected_dev_reaches:,} development reaches, "
    f"{expected_holdout_reaches:,} holdout reaches",
    flush=True,
)

print(
    f"accumulate daily paired (<={MAX_DISTANCE_KM:g} km) for independent "
    "event-r sites ...",
    flush=True,
)
use = ['site_no', 'reach_id', 'river_wse', 'Value_EGM2008_m', DISTANCE_COLUMN, 'date']
site_pair = collections.defaultdict(list); site_meta = {}
for ch in pd.read_csv(DAY, sep='\t', usecols=use, chunksize=500000, low_memory=False):
    ch[DISTANCE_COLUMN] = pd.to_numeric(ch[DISTANCE_COLUMN], errors='coerce')
    ch['rw'] = pd.to_numeric(ch['river_wse'], errors='coerce')
    ch['gw'] = pd.to_numeric(ch['Value_EGM2008_m'], errors='coerce')
    ch['t'] = yfrac(pd.to_datetime(ch['date'], errors='coerce'))
    ch = ch[
        (ch[DISTANCE_COLUMN] <= MAX_DISTANCE_M)
        & np.isfinite(ch.t)
        & ch["site_no"].isin(event_sites)
    ]
    for sid, g in ch.dropna(subset=['rw', 'gw']).groupby('site_no'):
        site_pair[sid].extend(zip(g.t.tolist(), g.rw.tolist(), g.gw.tolist()))
        if sid not in site_meta:
            frozen = event_population.loc[sid]
            site_meta[sid] = (
                frozen["reach_id"], frozen["split"], float(frozen["event_r"])
            )
    del ch; gc.collect()
print(f"  paired sites {len(site_pair)}", flush=True)

print(f"per-site observed event_r + circular-shift null (B={B}) ...", flush=True)
recs = []          # (site_no, reach_id, frozen split, recomputed obs, archived obs)
null_mat = []      # per-site array of B null event_r
time_null_mat = [] # per-site array of B irregular-time OU null event_r
reach_ids = []
ou_records = []
for sid, vals in site_pair.items():
    if len(vals) < 24:
        continue
    a = np.array(vals); t, rw, gw = a[:, 0], a[:, 1], a[:, 2]
    m = np.isfinite(t) & np.isfinite(rw) & np.isfinite(gw)
    if m.sum() < 24:
        continue
    t, rw, gw = t[m], rw[m], gw[m]
    rr = hres(t, rw); gr = hres(t, gw)
    if rr.std() == 0 or gr.std() == 0:
        continue
    order = np.argsort(t)
    t, rr, gr = t[order], rr[order], gr[order]
    n = len(rr)
    obs = float(np.corrcoef(rr, gr)[0, 1])
    rr_z = (rr - rr.mean()) / rr.std(); gr_z = (gr - gr.mean()) / gr.std()
    offs = RNG.integers(1, n, B)
    nulls = np.array([float(np.dot(rr_z, np.roll(gr_z, k)) / n) for k in offs])
    tau_r = fit_ou_timescale_days(t, rr)
    tau_g = fit_ou_timescale_days(t, gr)
    if tau_r is None or tau_g is None:
        continue
    river_surrogate = simulate_ou_at_dates(t, tau_r, B, TIME_AWARE_RNG)
    gw_surrogate = simulate_ou_at_dates(t, tau_g, B, TIME_AWARE_RNG)
    river_surrogate = residualize_harmonic_draws(t, river_surrogate)
    gw_surrogate = residualize_harmonic_draws(t, gw_surrogate)
    time_nulls = np.mean(river_surrogate * gw_surrogate, axis=1)
    rid, split, archived_obs = site_meta[sid]
    recs.append((sid, rid, split, obs, archived_obs))
    null_mat.append(nulls); time_null_mat.append(time_nulls); reach_ids.append(rid)
    ou_records.append((sid, tau_r, tau_g))
null_mat = np.array(null_mat)  # [n_site, B]
time_null_mat = np.array(time_null_mat)  # [n_site, B]
obs_df = pd.DataFrame(
    recs,
    columns=['site_no', 'reach_id', 'split', 'event_r', 'archived_event_r'],
)
print(f"  sites with event_r {len(obs_df)}", flush=True)

# Blocking sample and numerical reproduction checks: the observed arm of the
# permutation test must be exactly the archived independent daily population.
observed_sites = set(obs_df["site_no"])
missing_sites = sorted(event_sites.difference(observed_sites))
unexpected_sites = sorted(observed_sites.difference(event_sites))
if missing_sites or unexpected_sites:
    raise RuntimeError(
        "permutation sample differs from independent daily sites: "
        f"missing={len(missing_sites):,}, unexpected={len(unexpected_sites):,}"
    )
max_abs_event_r_difference = float(
    (obs_df["event_r"] - obs_df["archived_event_r"]).abs().max()
)
if (
    not np.isfinite(max_abs_event_r_difference)
    or max_abs_event_r_difference > REPRODUCTION_TOL
):
    raise RuntimeError(
        "daily observed event_r does not reproduce the all-sites table: "
        f"max absolute difference={max_abs_event_r_difference:.3e}"
    )
observed_reaches = int(obs_df["reach_id"].nunique())
observed_dev_reaches = int(
    obs_df.loc[obs_df["split"].eq("dev"), "reach_id"].nunique()
)
observed_holdout_reaches = int(
    obs_df.loc[obs_df["split"].eq("holdout"), "reach_id"].nunique()
)
if (
    observed_reaches != expected_reaches
    or observed_dev_reaches != expected_dev_reaches
    or observed_holdout_reaches != expected_holdout_reaches
):
    raise RuntimeError(
        "permutation Reach sample differs from independent daily population: "
        f"ALL {observed_reaches} != {expected_reaches}; "
        f"DEV {observed_dev_reaches} != {expected_dev_reaches}; "
        f"HOLDOUT {observed_holdout_reaches} != {expected_holdout_reaches}"
    )
print(
    "  independent daily sample and observed event_r reproduced exactly "
    f"(max |difference|={max_abs_event_r_difference:.3e})",
    flush=True,
)


def reach_median(site_vals, rids):
    s = pd.DataFrame({'reach_id': rids, 'v': site_vals})
    return s.groupby('reach_id').v.median()


def summarize(mask, label, bootstrap_seed):
    idx = np.where(mask.values)[0]
    rids = [reach_ids[i] for i in idx]
    obs_rm = reach_median(obs_df.event_r.values[idx], rids)
    obs_med = float(obs_rm.median()); obs_frac = float((obs_rm >= 0.3).mean())

    # Reach is both the inferential and resampling unit.  Use a separate RNG
    # so adding these intervals cannot alter the circular-shift null draws.
    obs_values = obs_rm.to_numpy(dtype=float)
    bootstrap_rng = np.random.default_rng(bootstrap_seed)
    bootstrap_idx = bootstrap_rng.integers(
        0, len(obs_values), size=(OBS_BOOTSTRAP_N, len(obs_values)))
    bootstrap_values = obs_values[bootstrap_idx]
    bootstrap_medians = np.median(bootstrap_values, axis=1)
    bootstrap_fractions = np.mean(bootstrap_values >= 0.3, axis=1)
    obs_median_ci95 = [
        float(np.percentile(bootstrap_medians, 2.5)),
        float(obs_med),
        float(np.percentile(bootstrap_medians, 97.5)),
    ]
    obs_fraction_ci95 = [
        float(np.percentile(bootstrap_fractions, 2.5)),
        float(obs_frac),
        float(np.percentile(bootstrap_fractions, 97.5)),
    ]

    # null: per permutation b, reach-median across sites, then median over reaches
    null_med = np.empty(B); null_frac = np.empty(B)
    time_null_med = np.empty(B); time_null_frac = np.empty(B)
    sub = null_mat[idx]
    time_sub = time_null_mat[idx]
    for b in range(B):
        rm = reach_median(sub[:, b], rids)
        null_med[b] = rm.median(); null_frac[b] = (rm >= 0.3).mean()
        time_rm = reach_median(time_sub[:, b], rids)
        time_null_med[b] = time_rm.median()
        time_null_frac[b] = (time_rm >= 0.3).mean()
    p_med = (1 + int(np.sum(null_med >= obs_med))) / (1 + B)
    p_frac = (1 + int(np.sum(null_frac >= obs_frac))) / (1 + B)
    p_time_med = (1 + int(np.sum(time_null_med >= obs_med))) / (1 + B)
    p_time_frac = (1 + int(np.sum(time_null_frac >= obs_frac))) / (1 + B)
    return dict(label=label, n_site=int(len(idx)), n_reach=int(len(obs_rm)),
                obs_event_r_median=float(obs_med),
                obs_event_r_median_ci95=obs_median_ci95,
                null_event_r_median=float(np.median(null_med)),
                null_event_r_median_p95=float(np.percentile(null_med, 95)),
                p_median=float(p_med),
                obs_frac_ge0p3=float(obs_frac),
                obs_frac_ge0p3_ci95=obs_fraction_ci95,
                null_frac_ge0p3=float(np.median(null_frac)),
                p_frac=float(p_frac),
                time_aware_null_event_r_median=float(np.median(time_null_med)),
                time_aware_null_event_r_median_p95=float(np.percentile(time_null_med, 95)),
                time_aware_p_median=float(p_time_med),
                time_aware_null_frac_ge0p3=float(np.median(time_null_frac)),
                time_aware_null_frac_ge0p3_p95=float(np.percentile(time_null_frac, 95)),
                time_aware_p_frac=float(p_time_frac),
                observed_bootstrap=dict(
                    unit='Reach',
                    n_replicates=OBS_BOOTSTRAP_N,
                    seed=int(bootstrap_seed),
                    interval='percentile 95% CI',
                    interval_format='[2.5th percentile, point estimate, 97.5th percentile]',
                ))


ou_df = pd.DataFrame(ou_records, columns=["site_no", "river_tau_days", "gw_tau_days"])
res = dict(B=B,
           time_aware_null=dict(
               model="independent continuous-time AR(1)/OU marginal residual surrogates",
               surrogate_processing="same six-column harmonic residualization as observed event_r; standardize after projection",
               refit_harmonics_for_each_surrogate=True,
               dates="exact pairwise-finite observed dates; no interpolation",
               autocorrelation="phi(delta_t)=exp(-delta_t/tau_c)",
               tau_estimation="profile likelihood over a 0.5-730.5 day logarithmic grid",
               cross_series_null="river and groundwater innovations simulated independently",
               n_site=int(len(ou_df)),
               river_tau_days_median=float(ou_df.river_tau_days.median()),
               groundwater_tau_days_median=float(ou_df.gw_tau_days.median()),
               note=("time-aware OU is the primary strict null; observation-order "
                     "circular shift is retained as a simpler sensitivity null"),
           ),
           sample_definition=dict(
               source=str(event_file),
               unit=("independent daily event-r site_no, aggregated to its "
                     "frozen daily-population reach_id"),
               monthly_harmonic_eligibility_required=False,
               n_site=int(len(obs_df)),
               n_reach_all=observed_reaches,
               n_reach_dev=observed_dev_reaches,
               n_reach_holdout=observed_holdout_reaches,
               split=("frozen Reach-level split from "
                      "08_event_metrics_all_sites.parquet"),
               max_abs_observed_event_r_difference=max_abs_event_r_difference,
           ),
           ALL=summarize(
               pd.Series([True] * len(obs_df)), 'ALL', OBS_BOOTSTRAP_SEED),
           DEV=summarize(
               obs_df.split == 'dev', 'DEV', OBS_BOOTSTRAP_SEED + 2),
           HOLDOUT=summarize(
               obs_df.split == 'holdout', 'HOLDOUT', OBS_BOOTSTRAP_SEED + 1))
print("\n=== event-scale observation-order and time-aware nulls ===")
print(json.dumps(res, indent=2))

rob = json.load(open(OUT / "08_robustness.json"))
rob['event_permutation_null'] = res
json.dump(rob, open(OUT / "08_robustness.json", 'w'), indent=2)
print("\nmerged into output/08_robustness.json")
