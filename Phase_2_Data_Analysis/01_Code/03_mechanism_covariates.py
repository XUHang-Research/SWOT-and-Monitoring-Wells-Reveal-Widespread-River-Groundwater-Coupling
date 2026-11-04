#!/usr/bin/env python3
"""Step 3 — mechanism covariates and regional descriptive analysis.

Extends the pair table with per-site water-table depth, the recharge-vs-river
annual phase offset, pair-specific prior bank distance, Reach slope2 and prior
width variance, a river-regime index (seasonal amplitude / subseasonal WSE
variability), and a
region label (the two US monitoring networks are combined into 'USA').
Then runs the descriptive regional analysis: regional effect sizes with a
fixed-seed label-permutation test, within-region sign tests, and region-level
and reach-level Spearman correlations of sub-seasonal co-variability against candidate controls
(aridity, depth, regime, phase offset, width), a leave-one-region-out check of
whether region-median co-variability is predictable from region-median descriptors, and
the Australia deep-dive. With only a small number of confounded country-regions
and a ~3-yr record this analysis is descriptive/exploratory, as the caveats
field records.

Inputs : raw/6matched_results_monthly.txt
         output/08_global_coupling_metrics.parquet (final table from step 2)
         output/08_event_metrics_all_sites.parquet (standalone daily event-r
         population from step 2)
Outputs: output/08_mechanism.parquet (pair table + depth_m,
         rech_river_phase_off_mo, slope2, p_wid_var,
         distance_prior_bank_m, region, river_regime),
         output/08_event_covariates_all_sites.parquet (complete independent
         daily event-r population plus static well/Reach/pair descriptors),
         output/08_mechanism_summary.json

Requires: numpy, pandas, pyarrow, scipy.
"""
import gc
import json
import os

import numpy as np
import pandas as pd
from scipy.stats import binomtest, kruskal, rankdata, spearmanr, tiecorrect

from importlib import import_module as _import_module
_harmonic_model = _import_module("10_harmonic_model")
fit_harmonic = _harmonic_model.fit_harmonic
_pipeline_config = _import_module("09_pipeline_config")
DISTANCE_COLUMN = _pipeline_config.DISTANCE_COLUMN
HARMONIC_MIN_POINTS = _pipeline_config.HARMONIC_MIN_POINTS
MAX_DISTANCE_KM = _pipeline_config.MAX_DISTANCE_KM
MAX_DISTANCE_M = _pipeline_config.MAX_DISTANCE_M
MON = _pipeline_config.MON
OUT = _pipeline_config.OUT
DEPTHCOL = "Depth to Water Below Land Surface in meter"
REACH_COVARIATES = ["slope2", "p_wid_var"]
N_REGION_PERMUTATIONS = 10_000
REGION_PERMUTATION_SEED = 20260807
MIN_REGION_REACHES = 15
CLIMATE_START_MONTH = pd.Period("2023-01", freq="M")
CLIMATE_END_MONTH = pd.Period("2025-12", freq="M")
CLIMATE_MONTHS = pd.period_range(
    CLIMATE_START_MONTH, CLIMATE_END_MONTH, freq="M")


def yfrac(d): d = pd.DatetimeIndex(d); return d.year.values + (d.dayofyear.values - 1) / 365.25
def ann_phase(t, x):
    result = fit_harmonic(t, x)
    if result is None:
        return np.nan, np.nan
    return result['phase'], result['amp']


def complete_window_aridity(frame):
    """Match step 1's fixed 36-month P/|ET| aridity definition."""
    required = {
        'site_no', 'date', 'precipitation_mm', 'evapotranspiration_mm',
        'precipitation_days', 'evapotranspiration_days',
    }
    missing = sorted(required.difference(frame.columns))
    if missing:
        raise KeyError(f"monthly input is missing required climate fields: {missing}")
    climate = frame[list(required)].copy()
    climate['date'] = pd.to_datetime(climate['date'], errors='coerce')
    if climate.date.isna().any():
        raise ValueError("monthly climate input contains invalid dates")
    climate['year_month'] = climate.date.dt.to_period('M')
    outside = ~climate.year_month.between(
        CLIMATE_START_MONTH, CLIMATE_END_MONTH)
    if outside.any():
        raise ValueError(
            f"monthly climate input contains {int(outside.sum())} rows outside "
            "the fixed 2023-01 to 2025-12 window"
        )
    if climate.duplicated(['site_no', 'year_month']).any():
        raise ValueError("monthly climate input contains duplicate site-month rows")
    for column in (
        'precipitation_mm', 'evapotranspiration_mm',
        'precipitation_days', 'evapotranspiration_days',
    ):
        climate[column] = pd.to_numeric(climate[column], errors='coerce')
    if not np.isfinite(
        climate[['precipitation_mm', 'evapotranspiration_mm']].to_numpy(float)
    ).all():
        raise ValueError("monthly climate input contains missing/non-finite P or ET")
    expected_days = climate.date.dt.days_in_month
    if not climate.precipitation_days.eq(expected_days).all():
        raise ValueError("monthly climate input contains incomplete precipitation months")
    if not climate.evapotranspiration_days.eq(expected_days).all():
        raise ValueError("monthly climate input contains incomplete ET months")
    coverage = climate.groupby('site_no').year_month.agg(['nunique', 'min', 'max'])
    complete = (
        coverage['nunique'].eq(len(CLIMATE_MONTHS))
        & coverage['min'].eq(CLIMATE_START_MONTH)
        & coverage['max'].eq(CLIMATE_END_MONTH)
    )
    if not complete.all():
        raise ValueError(
            f"{int((~complete).sum())} sites lack the complete 36-month "
            "climate window"
        )
    site = climate.groupby('site_no', as_index=False).agg(
        aridity_precipitation_sum_mm=('precipitation_mm', 'sum'),
        aridity_evapotranspiration_abs_sum_mm=(
            'evapotranspiration_mm', lambda values: values.abs().sum()),
        aridity_n_months=('year_month', 'nunique'),
        aridity_start_month=('year_month', 'min'),
        aridity_end_month=('year_month', 'max'),
    )
    denominator = site.aridity_evapotranspiration_abs_sum_mm
    site['aridity'] = np.divide(
        site.aridity_precipitation_sum_mm,
        denominator,
        out=np.full(len(site), np.nan),
        where=denominator.gt(0),
    )
    site['aridity_start_month'] = site.aridity_start_month.astype(str)
    site['aridity_end_month'] = site.aridity_end_month.astype(str)
    return site


def ols_fit_predict(Xtr, ytr, Xte):
    """OLS with intercept: fit on (Xtr, ytr), predict at Xte. Plain lstsq."""
    A = np.column_stack([np.ones(len(Xtr)), np.asarray(Xtr, float)])
    b, *_ = np.linalg.lstsq(A, np.asarray(ytr, float), rcond=None)
    return np.column_stack([np.ones(len(Xte)), np.asarray(Xte, float)]) @ b


def regional_effect_sizes(frame, kw_stat):
    """Return rank-based and raw-scale regional effects plus label nulls.

    ``eta_squared_H = H / (n - 1)`` is the rank-based effect reported alongside
    Kruskal-Wallis. ``raw_anova_eta_squared`` is the literal between-region
    share of squared deviations on the event-r scale. Labels are permuted while
    the observed region sizes remain fixed.
    """
    z = frame[['region', 'event_r']].dropna().copy()
    y = z.event_r.to_numpy(float)
    codes = pd.Categorical(z.region).codes
    n = len(y)
    counts = np.bincount(codes).astype(float)
    grand = float(y.mean())
    ss_total = float(np.sum((y - grand) ** 2))

    def raw_eta(group_codes):
        sums = np.bincount(group_codes, weights=y, minlength=len(counts))
        means = sums / counts
        return float(np.sum(counts * (means - grand) ** 2) / ss_total)

    ranks = rankdata(y)
    tie_correction = float(tiecorrect(ranks))

    def eta_h(group_codes):
        rank_sums = np.bincount(group_codes, weights=ranks, minlength=len(counts))
        h = (12.0 / (n * (n + 1.0)) * np.sum(rank_sums ** 2 / counts)
             - 3.0 * (n + 1.0)) / tie_correction
        return float(h / (n - 1.0))

    observed_eta_h = float(kw_stat / (n - 1.0))
    observed_raw_eta = raw_eta(codes)
    if not np.isclose(eta_h(codes), observed_eta_h, rtol=1e-10, atol=1e-12):
        raise RuntimeError("rank-based eta-squared does not reproduce scipy Kruskal-Wallis H")

    rng = np.random.default_rng(REGION_PERMUTATION_SEED)
    null_eta_h = np.empty(N_REGION_PERMUTATIONS)
    null_raw_eta = np.empty(N_REGION_PERMUTATIONS)
    for i in range(N_REGION_PERMUTATIONS):
        permuted = rng.permutation(codes)
        null_eta_h[i] = eta_h(permuted)
        null_raw_eta[i] = raw_eta(permuted)

    return dict(
        n_regions=int(len(counts)), n_reaches=int(n),
        kruskal_eta_squared_H=observed_eta_h,
        kruskal_eta_squared_H_definition='H / (n - 1); rank-based effect size',
        raw_anova_eta_squared=observed_raw_eta,
        raw_anova_eta_squared_definition='SS_between / SS_total on reach-level event_r',
        label_permutation=dict(
            n_permutations=N_REGION_PERMUTATIONS,
            seed=REGION_PERMUTATION_SEED,
            permutation_unit='region labels across reaches; observed region sizes preserved',
            alternative='one-sided: permuted effect >= observed effect',
            p_kruskal_eta_squared_H=float(
                (1 + np.sum(null_eta_h >= observed_eta_h)) / (N_REGION_PERMUTATIONS + 1)),
            p_raw_anova_eta_squared=float(
                (1 + np.sum(null_raw_eta >= observed_raw_eta)) / (N_REGION_PERMUTATIONS + 1)),
        ),
    )


def within_region_sign_test(frame, variable):
    """Spearman association in each retained region plus an exact sign test."""
    rows = []
    for region, group in frame.groupby('region', sort=True):
        finite = group[[variable, 'event_r']].dropna()
        if len(finite) < 4:
            continue
        rho, p_value = spearmanr(finite[variable], finite.event_r)
        rows.append(dict(region=region, n=int(len(finite)), rho=float(rho),
                         p_value=float(p_value), sign=int(np.sign(rho))))
    nonzero = [row for row in rows if row['sign'] != 0]
    n_positive = sum(row['sign'] > 0 for row in nonzero)
    sign_p = (float(binomtest(n_positive, len(nonzero), p=0.5,
                              alternative='two-sided').pvalue)
              if nonzero else np.nan)
    return dict(variable=variable, region_results=rows,
                n_regions=int(len(nonzero)), n_positive=int(n_positive),
                n_negative=int(len(nonzero) - n_positive),
                test='exact two-sided binomial sign test against p=0.5; zero correlations excluded',
                p_value=sign_p)


print(f"extract hydrogeomorphic covariates (monthly, <={MAX_DISTANCE_KM:g} km) ...")
mon = pd.read_csv(MON, sep='\t', low_memory=False)
required = {
    'site_no', 'reach_id', 'date', 'river_wse', 'Value_EGM2008_m',
    DISTANCE_COLUMN, 'precipitation_mm', 'evapotranspiration_mm', DEPTHCOL,
    *REACH_COVARIATES,
}
missing = sorted(required.difference(mon.columns))
if missing:
    raise KeyError(f"monthly input is missing required mechanism fields: {missing}")
for c in ['river_wse', 'Value_EGM2008_m', DISTANCE_COLUMN, 'precipitation_mm',
          'evapotranspiration_mm', DEPTHCOL, *REACH_COVARIATES]:
    if c in mon: mon[c] = pd.to_numeric(mon[c], errors='coerce')
mon['date'] = pd.to_datetime(mon['date'], errors='coerce')
SITE_CLIMATE = complete_window_aridity(mon)
base = mon.loc[mon[DISTANCE_COLUMN].le(MAX_DISTANCE_M)].copy()

# slope2 and p_wid_var are Reach attributes; distance is specific to the
# well-site x Reach pair. Median aggregation prevents repeated monthly SWOT
# observations from weighting otherwise static descriptors multiple times.
P = (base.groupby(['site_no', 'reach_id'], as_index=False, dropna=False)
         .agg(slope2=('slope2', 'median'),
              p_wid_var=('p_wid_var', 'median'),
              distance_prior_bank_m=(DISTANCE_COLUMN, 'median')))

# Static groundwater depth for the independent daily event-r population.  It
# intentionally uses all available monthly base rows and does not require a
# paired water-level month or a valid monthly harmonic fit.
STATIC_SITE = (base.groupby('site_no', as_index=False, dropna=False)
                  .agg(depth_m=(DEPTHCOL, 'median')))

# Recharge/river phase and river amplitude do not require a finite groundwater
# value.  Requiring one here would reintroduce the monthly river-GW intersection
# into descriptors intended for the standalone daily event-r population.
m = base.dropna(subset=[
    'river_wse', 'precipitation_mm', 'evapotranspiration_mm', 'date'
]).copy()
m['R'] = m['precipitation_mm'] - m['evapotranspiration_mm']
ext = []
for sid, g in m.groupby('site_no'):
    g = g.sort_values('date')
    if g.shape[0] < HARMONIC_MIN_POINTS: continue
    t = yfrac(g.date)
    rp, river_amp = ann_phase(t, g.river_wse.values)
    cp, _ = ann_phase(t, g.R.values)
    off = np.nan
    if np.isfinite(rp) and np.isfinite(cp): off = abs(((cp - rp + np.pi) % (2 * np.pi)) - np.pi) / np.pi * 6  # months 0..6
    depth_values = g[DEPTHCOL].dropna()
    depth = float(depth_values.median()) if len(depth_values) else np.nan
    ext.append(dict(
        site_no=sid,
        depth_m=depth,
        rech_river_phase_off_mo=off,
        river_annual_amplitude_m=river_amp,
    ))
SITE_HARMONIC = pd.DataFrame(ext)
del mon, base, m
gc.collect()
print(
    f"  per-site harmonic descriptors for {len(SITE_HARMONIC)} sites; "
    f"phase offset available "
    f"{SITE_HARMONIC.rech_river_phase_off_mo.notna().mean():.2f}")
print(f"  pair descriptors for {len(P)} site x Reach pairs; "
      f"slope2={P.slope2.notna().mean():.2f}, p_wid_var={P.p_wid_var.notna().mean():.2f}, "
      f"prior bank distance={P.distance_prior_bank_m.notna().mean():.2f}")

S = (pd.read_parquet(OUT / "08_global_coupling_metrics.parquet")
       .merge(SITE_HARMONIC, on='site_no', how='left', validate='many_to_one')
       .merge(P, on=['site_no', 'reach_id'], how='left', validate='one_to_one'))
S['region'] = S.country.map(lambda c: 'USA' if c in ('USGS', 'NGWMN') else c)
# River regime: seasonal amplitude / sub-seasonal WSE residual variability.
# sigma_swot_emp is retained only as the legacy internal column name.
S['river_regime'] = S.ampR / S.sigma_swot_emp.where(S.sigma_swot_emp > 1e-9)

# Independent daily event-r table.  Keep this population separate from S:
# merging event_r onto S would silently restore the monthly-harmonic gate.
EVENT = pd.read_parquet(OUT / "08_event_metrics_all_sites.parquet")
event_required = {
    'site_no', 'reach_id', 'country', 'lon', 'lat', 'river_width',
    'event_r', 'split',
}
event_missing = sorted(event_required.difference(EVENT.columns))
if event_missing:
    raise KeyError(
        f"08_event_metrics_all_sites.parquet is missing fields: {event_missing}"
    )
if EVENT.site_no.duplicated().any():
    raise ValueError("independent daily event-r table contains duplicate sites")
if not EVENT.groupby('reach_id').split.nunique(dropna=False).eq(1).all():
    raise ValueError("a Reach appears in multiple independent daily splits")
EVENT['region'] = EVENT.country.map(
    lambda c: 'USA' if c in ('USGS', 'NGWMN') else c)
if not EVENT.groupby('reach_id').region.nunique(dropna=False).eq(1).all():
    raise ValueError("a Reach appears in multiple daily event-r regions")

# Static descriptors for Fig. S4.  Depth is from every available base month;
# P contains static Reach/pair fields from every base row.  Neither merge
# imposes monthly-harmonic eligibility on the daily event-r sites.
EVENT_COV = (EVENT.merge(
                 STATIC_SITE, on='site_no', how='left', validate='one_to_one')
             .merge(
                 SITE_HARMONIC[[
                     'site_no', 'rech_river_phase_off_mo',
                     'river_annual_amplitude_m',
                 ]],
                 on='site_no', how='left', validate='one_to_one')
             .merge(P, on=['site_no', 'reach_id'], how='left',
                    validate='one_to_one')
             .merge(SITE_CLIMATE, on='site_no', how='left',
                    validate='one_to_one'))
if EVENT_COV.aridity.isna().any() or not EVENT_COV.aridity_n_months.eq(
        len(CLIMATE_MONTHS)).all():
    raise ValueError(
        "independent daily event-r sites did not receive complete-window aridity"
    )

# Prefer the monthly river amplitude computed beside the recharge phase.  A
# small number of daily-event sites lack the monthly harmonic support required
# for that fit; their amplitude falls back to the already validated same-date
# daily harmonic fit from step 2.  The source remains auditable per row.
monthly_amplitude = EVENT_COV['river_annual_amplitude_m']
daily_amplitude = EVENT_COV['river_annual_amplitude_m_same_sample']
EVENT_COV['river_annual_amplitude_for_regime_m'] = (
    monthly_amplitude.combine_first(daily_amplitude))
EVENT_COV['river_regime_amplitude_source'] = np.where(
    monthly_amplitude.notna(), 'monthly_river_climate_fit',
    'same_date_daily_event_fit')
EVENT_COV['river_regime'] = (
    EVENT_COV.river_annual_amplitude_for_regime_m
    / EVENT_COV.sigma_swot_emp.where(EVENT_COV.sigma_swot_emp > 1e-9)
)

required_event_covariates = [
    'event_r', 'river_width', 'depth_m', 'slope2', 'p_wid_var',
    'distance_prior_bank_m', 'aridity', 'sigma_swot_emp',
    'sigma_swot_emp_n_nodes', 'river_annual_amplitude_for_regime_m',
    'river_regime', 'region', 'lon', 'lat',
]
absent_event_covariates = sorted(
    set(required_event_covariates).difference(EVENT_COV.columns))
if absent_event_covariates:
    raise KeyError(
        "EVENT_COV is missing required covariate columns: "
        f"{absent_event_covariates}")
depth_missing_policy = os.environ.get('GRL_MISSING_DEPTH_POLICY', 'error')
if depth_missing_policy not in {'error', 'complete_cases'}:
    raise ValueError(f'Unknown missing-depth policy: {depth_missing_policy}')
# The default remains strict. An unresolved method choice cannot silently
# activate complete-case depth statistics in a full rerun.
missing_event_covariates = {
    column: int(EVENT_COV[column].isna().sum())
    for column in required_event_covariates
    if EVENT_COV[column].isna().any() and (column != 'depth_m' or depth_missing_policy == 'error')
}
if missing_event_covariates:
    raise ValueError(
        "complete daily EVENT_COV lacks required covariates: "
        f"{missing_event_covariates}")
EVENT_COV['depth_covariate_status'] = np.where(
    np.isfinite(pd.to_numeric(EVENT_COV.depth_m, errors='coerce')),
    'AVAILABLE', 'UNAVAILABLE_IN_PHASE1_MONTHLY_INPUT')
if np.isinf(pd.to_numeric(EVENT_COV.depth_m, errors='coerce')).any():
    raise ValueError('Infinite depth covariate is invalid')
if EVENT_COV.site_no.duplicated().any():
    raise ValueError("EVENT_COV contains duplicate site_no rows")

# Pure regional event-r branch: one row per Reach from the complete independent
# daily population.  Only countries with the pre-specified minimum Reach
# support enter cross-region inference and the regional figure.
def aggregate_event_descriptors(frame):
    """Aggregate one fixed constituent-site population to Reach medians."""
    return (frame.groupby('reach_id', as_index=False)
                .agg(event_r=('event_r', 'median'),
                     width=('river_width', 'median'),
                     aridity=('aridity', 'median'), depth=('depth_m', 'median'),
                     phase_off=('rech_river_phase_off_mo', 'median'),
                     regime=('river_regime', 'median'),
                     region=('region', 'first'), split=('split', 'first'),
                     lon=('lon', 'median'), lat=('lat', 'median'),
                     n_site_pairs=('site_no', 'size')))

event_rc = aggregate_event_descriptors(EVENT_COV)
depth_sites = EVENT_COV[EVENT_COV.depth_covariate_status.eq('AVAILABLE')].copy()
depth_rc = aggregate_event_descriptors(depth_sites)
event_keep = event_rc.region.value_counts()
event_keep = event_keep[event_keep >= MIN_REGION_REACHES].index
event_r_regional = event_rc[event_rc.region.isin(event_keep)].copy()
event_reg = event_r_regional.groupby('region').agg(
   n=('event_r', 'size'), event_r=('event_r', 'median'),
   aridity=('aridity', 'median'), depth=('depth', 'median'), phase_off=('phase_off', 'median'),
   regime=('regime', 'median'), width=('width', 'median')).reset_index()

# Main regional statistics retain the complete daily event population.
# Depth-dependent statistics use finite-depth sites before Reach aggregation.
# Phase-offset correlations remain pairwise-finite because some sites lack
# monthly harmonic support.
r = event_r_regional
reg = event_reg
depth_r = depth_rc[depth_rc.region.isin(event_keep)].copy()
depth_reg = depth_r.groupby('region', as_index=False).agg(
    n=('event_r','size'), event_r=('event_r','median'), depth=('depth','median'))
def sp(a, b):
    m = np.isfinite(a) & np.isfinite(b)
    if m.sum() < 4: return (np.nan, np.nan)
    r_, p_ = spearmanr(a[m], b[m]); return (float(r_), float(p_))
region_corr = {c: sp(reg[c].values, reg.event_r.values) for c in ['aridity', 'depth', 'phase_off', 'regime', 'width']}
region_corr['depth'] = sp(depth_reg.depth.values, depth_reg.event_r.values)
kw_groups = [
    g.event_r.dropna().to_numpy()
    for _, g in event_r_regional.groupby('region')
]
kw_groups = [g for g in kw_groups if len(g)]
kw_stat, kw_p = kruskal(*kw_groups) if len(kw_groups) >= 2 else (np.nan, np.nan)
regional_effect = regional_effect_sizes(event_r_regional, kw_stat)
within_region_tests = {
    variable: within_region_sign_test(depth_r if variable == 'depth' else r, variable)
    for variable in ['width', 'depth']
}

# ---- reach-level Spearman (within full sample) + leave-one-region-out CV of a connectivity model ----
reach_corr = {c: sp(r[c].values, r.event_r.values) for c in ['aridity', 'depth', 'phase_off', 'regime', 'width']}
reach_corr['depth'] = sp(depth_r.depth.values, depth_r.event_r.values)
# connectivity index from pre-specified signs: +aridity(wetter couples), -depth(deeper decouples), +regime(seasonal couples), -phase_off
feats = ['aridity', 'depth', 'regime']   # pre-specified connectivity controls with data coverage
rr = depth_r.dropna(subset=feats + ['event_r']).copy()
for f in feats: rr[f + '_z'] = (rr[f] - rr[f].mean()) / rr[f].std()
Z = [f + '_z' for f in feats]
# Leave one region out and predict its median event_r. Medians are used
# consistently for both descriptors and event_r because the regional table and
# figures summarize skewed Reach distributions with robust regional medians.
loro = []
regions = rr.region.unique()
for hold in regions:
    tr = rr[rr.region != hold]; te = rr[rr.region == hold]
    gtr = tr.groupby('region')[Z + ['event_r']].median()
    if len(gtr) < 3: continue
    pred = ols_fit_predict(
        gtr[Z].values,
        gtr.event_r.values,
        te[Z].median().values.reshape(1, -1),
    )[0]
    loro.append(dict(region=hold, obs=float(te.event_r.median()), pred=float(pred), n=len(te)))
loro = pd.DataFrame(loro)
loro_r = sp(loro.obs.values, loro.pred.values) if len(loro) >= 4 else (np.nan, np.nan)

# ---- Australia deep-dive ----
au = r[r.region == 'Australia']; oth = r[r.region != 'Australia']
aus = dict(n=len(au), event_r=float(au.event_r.median()), river_clocked=float((au.event_r >= 0.3).mean()),
   aridity=float(au.aridity.median()), depth_m=float(au.depth.median()),
   regime=float(au.regime.median()), phase_off_mo=float(au.phase_off.median()),
   vs_rest=dict(event_r=float(oth.event_r.median()), aridity=float(oth.aridity.median()),
      depth_m=float(oth.depth.median()), regime=float(oth.regime.median())))
event_au = event_rc[event_rc.region == 'Australia']
event_oth = event_rc[event_rc.region != 'Australia']
event_aus = dict(
    n=int(len(event_au)),
    event_r=float(event_au.event_r.median()),
    river_clocked=float((event_au.event_r >= 0.3).mean()),
    vs_rest=dict(
        n=int(len(event_oth)),
        event_r=float(event_oth.event_r.median()),
        river_clocked=float((event_oth.event_r >= 0.3).mean()),
    ),
)

out = dict(
  sample_definitions=dict(
     regional_event_r=(
        'complete independent daily event-r population from '
        '08_event_metrics_all_sites.parquet; median across eligible sites '
        'within Reach; cross-region tests retain countries with at least '
        f'{MIN_REGION_REACHES} eligible Reaches'),
     regional_event_r_n_site=int(len(EVENT)),
     regional_event_r_n_reach_all=int(len(event_rc)),
     regional_event_r_n_reach_retained=int(len(event_r_regional)),
     mechanism_covariates=(
        'complete independent daily event-r EVENT_COV population; all static '
        'descriptor correlations, within-region tests, LORO, and Australia '
        'mechanism comparisons are formed after Reach aggregation from this '
        'population, without monthly river-GW eligibility; depth-dependent tests use finite-depth sites before Reach aggregation'),
     mechanism_covariates_n_site=int(len(EVENT_COV)),
     mechanism_covariates_n_reach_all=int(len(event_rc)),
     mechanism_covariates_n_reach_retained=int(len(r)),
     depth_missing_policy=depth_missing_policy,
     depth_complete_case_rule='Filter finite site depth before Reach aggregation; response and predictors use those same wells. Main daily population is preserved.',
     depth_complete_case_n_site=int(len(depth_sites)),
     depth_missing_site_ids=sorted(EVENT_COV.loc[EVENT_COV.depth_covariate_status.ne('AVAILABLE'), 'site_no'].astype(str)),
     depth_complete_case_n_reach_all=int(len(depth_rc)),
     depth_complete_case_n_reach_retained=int(len(depth_r)),
     descriptor_finite_reaches={
        column: int(r[column].notna().sum())
        for column in ['width', 'aridity', 'depth', 'phase_off', 'regime']}),
  heterogeneity=dict(test="Kruskal-Wallis across retained regions",
     statistic=float(kw_stat), p_value=float(kw_p),
     n_regions=len(kw_groups), n_reaches=int(len(event_r_regional)),
     sample='complete independent daily event-r population'),
  regional_effect_size=regional_effect,
  within_region_sign_tests=within_region_tests,
  region_level_table=event_reg.to_dict(orient='records'),
  region_level_mechanism_table=reg.to_dict(orient='records'),
  region_level_depth_complete_case_table=depth_reg.to_dict(orient='records'),
  region_level_spearman_vs_eventr=region_corr,
  reach_level_spearman_vs_eventr=reach_corr,
  leave_one_region_out=dict(
     aggregation='regional medians for training descriptors, training target, test descriptors, and observed target',
     predictions=loro.to_dict(orient='records'),
     obs_vs_pred_spearman=loro_r, n_regions=len(loro)),
  australia=aus,
  australia_event_r_full_daily=event_aus,
  caveats=(f"regional event-r and mechanism analyses retain {len(event_reg)} "
           "regions from the same complete daily event-r mother population; "
           "depth-dependent tests use the explicitly reported finite-depth site subset; "
           "phase-offset correlations use their explicitly reported finite-"
           "Reach subset because some sites lack monthly harmonic support; "
           "reach-level event_r is noisy over a ~3-yr record; region=country "
           "bundles climate+geology+anthropogenic"))
S.to_parquet(OUT / "08_mechanism.parquet", index=False)
EVENT_COV.to_parquet(OUT / "08_event_covariates_all_sites.parquet", index=False)
json.dump(out, open(OUT / "08_mechanism_summary.json", 'w'), indent=2, default=str)
print(json.dumps(out, indent=2, default=str))
print("\nwrote output/08_mechanism.parquet + "
      "output/08_event_covariates_all_sites.parquet + "
      "output/08_mechanism_summary.json")
