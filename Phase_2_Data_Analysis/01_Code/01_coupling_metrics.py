#!/usr/bin/env python3
"""Step 1 — per-site river-groundwater co-variability and harmonic metrics.

From the raw monthly matched SWOT x well series: fixed-window (2023-2025)
per-site aridity computed before any water-level availability screen, harmonic amplitude
ratio A and phase lag tau (annual band), seasonal correlation, deseasonalized
(monthly-climatology) correlation, seasonal commonality partition (unique river
vs unique recharge variance), and an aridity proxy. From the raw daily matched
file (chunked, memory-safe): per-reach SWOT reported height uncertainty.
Assigns the pre-specified development (Germany + China) / sealed rest-of-world
holdout split at REACH level (a reach is 'dev' iff a majority of its sites are
in a development country, applied to all its sites, so no reach appears in both
splits).

Inputs : raw/6matched_results_monthly.txt
         raw/4matched_day_wells_shpsite_with_estimated_wse_fixed_node_alltimes_SG1.txt
Outputs: output/08_global_coupling_metrics.parquet (one row per well site x SWOT
         reach pair; extended by step 2), output/08_summary.json,
         output/08_holdout.json

Requires: numpy, pandas, pyarrow, scipy.
"""
import collections
import gc
import json
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import spearmanr

from importlib import import_module as _import_module
_pipeline_config = _import_module("09_pipeline_config")
DAY = _pipeline_config.DAY
DISTANCE_COLUMN = _pipeline_config.DISTANCE_COLUMN
MAX_DISTANCE_KM = _pipeline_config.MAX_DISTANCE_KM
MAX_DISTANCE_M = _pipeline_config.MAX_DISTANCE_M
MON = _pipeline_config.MON
OUT = _pipeline_config.OUT
_harmonic_model = _import_module("10_harmonic_model")
fit_harmonic = _harmonic_model.fit_harmonic
harmonic_criterion = _harmonic_model.harmonic_criterion
harmonic_design = _harmonic_model.harmonic_design
_pipeline_config = _import_module("09_pipeline_config")
HARMONIC_MIN_POINTS = _pipeline_config.HARMONIC_MIN_POINTS
_phase_statistics = _import_module("13_phase_statistics")
aggregate_reach_at = _phase_statistics.aggregate_reach_at
circular_mean_days = _phase_statistics.circular_mean_days
DEV_COUNTRIES = {"Germany", "China"}  # pre-specified: explore on the mass, confirm on the independent rest-of-world
CLIMATE_START_MONTH = pd.Period("2023-01", freq="M")
CLIMATE_END_MONTH = pd.Period("2025-12", freq="M")
CLIMATE_MONTHS = pd.period_range(CLIMATE_START_MONTH, CLIMATE_END_MONTH, freq="M")


def country(s): return str(s).split('+')[0].split('_')[0]
def yfrac(d): d = pd.DatetimeIndex(d); return d.year.values + (d.dayofyear.values - 1) / 365.25
def harm(t, x, fr=(1., 2.)):
    return fit_harmonic(t, x, frequencies=fr)
def seasonal_recon(t, x, fr=(1., 2.)):
    """Fitted annual+semiannual (no trend/intercept) reconstruction of x at times t; for commonality predictors."""
    result = fit_harmonic(t, x, frequencies=fr)
    if result is None:
        return None
    seasonal_design = harmonic_design(np.asarray(t, float), frequencies=fr)
    seasonal_design[:, :2] = 0.0  # drop intercept and trend
    return seasonal_design @ result['beta']
def r2_of(y, X):
    X = np.column_stack([np.ones(len(y))] + X); m = np.all(np.isfinite(X), 1) & np.isfinite(y)
    if m.sum() < 6: return np.nan
    y2 = y[m]; X2 = X[m]; b, *_ = np.linalg.lstsq(X2, y2, rcond=None)
    return 1 - np.var(y2 - X2 @ b) / np.var(y2) if np.var(y2) > 0 else np.nan


def complete_window_aridity(frame):
    """Return one P/|ET| value per site from all 36 natural months.

    This function intentionally runs before the pairwise-finite water-level
    screen.  It fails loudly on duplicate, missing, incomplete, or out-of-window
    site-month climate records so observation availability cannot determine the
    climate denominator.
    """
    required = {
        'site_no', 'date', 'precipitation_mm', 'evapotranspiration_mm',
        'precipitation_days', 'evapotranspiration_days',
    }
    missing = sorted(required.difference(frame.columns))
    if missing:
        raise KeyError(f"monthly input is missing required climate fields: {missing}")

    climate = frame[list(required)].copy()
    climate['date'] = pd.to_datetime(climate['date'], errors='coerce')
    if climate['date'].isna().any():
        raise ValueError("monthly climate input contains invalid dates")
    climate['year_month'] = climate['date'].dt.to_period('M')
    outside = ~climate['year_month'].between(CLIMATE_START_MONTH, CLIMATE_END_MONTH)
    if outside.any():
        raise ValueError(
            f"monthly climate input contains {int(outside.sum())} rows outside "
            "the fixed 2023-01 to 2025-12 window"
        )
    if climate.duplicated(['site_no', 'year_month']).any():
        raise ValueError(
            "monthly climate input contains duplicate site-month rows before "
            "water-level screening"
        )

    for column in (
        'precipitation_mm', 'evapotranspiration_mm',
        'precipitation_days', 'evapotranspiration_days',
    ):
        climate[column] = pd.to_numeric(climate[column], errors='coerce')
    if not np.isfinite(
        climate[['precipitation_mm', 'evapotranspiration_mm']].to_numpy(float)
    ).all():
        raise ValueError("monthly climate input contains missing/non-finite P or ET")
    expected_days = climate['date'].dt.days_in_month
    if not climate['precipitation_days'].eq(expected_days).all():
        raise ValueError("monthly climate input contains incomplete precipitation months")
    if not climate['evapotranspiration_days'].eq(expected_days).all():
        raise ValueError("monthly climate input contains incomplete ET months")

    coverage = climate.groupby('site_no')['year_month'].agg(['nunique', 'min', 'max'])
    complete = (
        coverage['nunique'].eq(len(CLIMATE_MONTHS))
        & coverage['min'].eq(CLIMATE_START_MONTH)
        & coverage['max'].eq(CLIMATE_END_MONTH)
    )
    if not complete.all():
        sample = coverage.loc[~complete].head(10).reset_index().to_dict(orient='records')
        raise ValueError(
            f"{int((~complete).sum())} sites lack the complete 36-month climate "
            f"window; sample={sample}"
        )

    site = climate.groupby('site_no', as_index=False).agg(
        aridity_precipitation_sum_mm=('precipitation_mm', 'sum'),
        aridity_evapotranspiration_abs_sum_mm=(
            'evapotranspiration_mm', lambda values: values.abs().sum()),
        aridity_n_months=('year_month', 'nunique'),
        aridity_start_month=('year_month', 'min'),
        aridity_end_month=('year_month', 'max'),
    )
    denominator = site['aridity_evapotranspiration_abs_sum_mm']
    site['aridity'] = np.divide(
        site['aridity_precipitation_sum_mm'], denominator,
        out=np.full(len(site), np.nan), where=denominator.gt(0),
    )
    site['aridity_start_month'] = site['aridity_start_month'].astype(str)
    site['aridity_end_month'] = site['aridity_end_month'].astype(str)
    return site


print("1) monthly load + screen ...")
mon = pd.read_csv(MON, sep='\t', low_memory=False)
for c in ['river_wse', 'Value_EGM2008_m', DISTANCE_COLUMN, 'precipitation_mm', 'evapotranspiration_mm', 'river_width', 'number_match_SG']:
    if c in mon: mon[c] = pd.to_numeric(mon[c], errors='coerce')
mon['date'] = pd.to_datetime(mon['date'], errors='coerce')
mon['month'] = mon['date'].dt.month
mon['country'] = mon.site_no.map(country)
# Compute fixed-window climate before selecting pairwise-finite river/GW months.
site_climate = complete_window_aridity(mon)
print(
    f"   fixed climate window: {len(site_climate)} sites x "
    f"{len(CLIMATE_MONTHS)} complete natural months"
)
m = mon.dropna(subset=['river_wse', 'Value_EGM2008_m', 'date']).copy()
m = m[m[DISTANCE_COLUMN] <= MAX_DISTANCE_M]
if 'number_match_SG' in m: m = m[(m.number_match_SG == 1) | (m.number_match_SG.isna())]
# ET is stored POSITIVE in this file (98.7% > 0), so the net recharge proxy is R = P - ET (NOT P + ET).
_etpos = (mon['evapotranspiration_mm'] > 0).mean()
print(f"   ET positive fraction = {_etpos:.3f} -> using R = P - ET")
m['R'] = m['precipitation_mm'] - m['evapotranspiration_mm']
print(f"   screened monthly rows {len(m)}, sites {m.site_no.nunique()}, reaches {m.reach_id.nunique()}")

print("2) per-site metrics (harmonic A/tau, seasonal/deseason r, commonality) ...")
rows = []
for sid, g in m.groupby('site_no'):
    g = g.sort_values('date')
    if g.shape[0] < HARMONIC_MIN_POINTS: continue
    t = yfrac(g.date)
    hr = harm(t, g.river_wse.values); hg = harm(t, g.Value_EGM2008_m.values)
    if hr is None or hg is None: continue
    A = hg['amp'] / hr['amp'] if hr['amp'] > 0 else np.nan
    dph = (hg['phase'] - hr['phase'] + np.pi) % (2 * np.pi) - np.pi   # GW minus river annual phase (rad), in [-pi,pi)
    tau_days = dph / (2 * np.pi) * 365.25
    # seasonal corr
    sr = np.corrcoef(g.river_wse, g.Value_EGM2008_m)[0, 1] if g.river_wse.std() > 0 and g.Value_EGM2008_m.std() > 0 else np.nan
    # deseasonalized: remove month-of-year climatology
    gg = g.copy()
    gg['rA'] = gg.river_wse - gg.groupby('month').river_wse.transform('mean')
    gg['gA'] = gg.Value_EGM2008_m - gg.groupby('month').Value_EGM2008_m.transform('mean')
    dr = np.corrcoef(gg.rA, gg.gA)[0, 1] if gg.rA.std() > 0 and gg.gA.std() > 0 else np.nan
    # commonality: GW ~ river SEASONAL recon + recharge SEASONAL recon (annual+semiannual)
    y = g.Value_EGM2008_m.values
    riv_rec = seasonal_recon(t, g.river_wse.values); rch_rec = seasonal_recon(t, g.R.values)
    if riv_rec is None or rch_rec is None:
        uniq_r = uniq_R = r2b = np.nan
    else:
        r2b = r2_of(y, [riv_rec, rch_rec]); r2r = r2_of(y, [riv_rec]); r2R = r2_of(y, [rch_rec])
        uniq_r = r2b - r2R if np.isfinite(r2b) and np.isfinite(r2R) else np.nan
        uniq_R = r2b - r2r if np.isfinite(r2b) and np.isfinite(r2r) else np.nan
    rows.append(dict(site_no=sid, reach_id=g.reach_id.iloc[0], country=g.country.iloc[0],
        lon=g.wells_lon.iloc[0] if 'wells_lon' in g else np.nan, lat=g.wells_lat.iloc[0] if 'wells_lat' in g else np.nan,
        dist_km=g[DISTANCE_COLUMN].median() / 1000, river_width=g.river_width.median() if 'river_width' in g else np.nan,
        aquifer=str(g.AquiferType.iloc[0]) if 'AquiferType' in g else 'NA',
        n_mon=g.shape[0], ampR=hr['amp'], ampG=hg['amp'], A=A, tau_days=tau_days, gw_peak=hg['phase'], riv_peak=hr['phase'],
        seasonal_r=sr, deseason_r=dr, r2_both=r2b, uniq_river=uniq_r, uniq_rech=uniq_R))
S = pd.DataFrame(rows)
S = S.merge(site_climate, on='site_no', how='left', validate='one_to_one')
if S['aridity'].isna().any() or not S['aridity_n_months'].eq(len(CLIMATE_MONTHS)).all():
    raise ValueError("eligible water-level sites did not receive complete-window aridity")
S['is_dev'] = S.country.isin(DEV_COUNTRIES)
# seal the holdout at REACH level: a reach is 'dev' iff a majority of its sites are dev-country,
# applied to ALL its sites, so no reach_id appears in both splits.
reach_dev = S.groupby('reach_id')['is_dev'].mean() >= 0.5
S['split'] = np.where(S.reach_id.map(reach_dev), 'dev', 'holdout')
print(f"   sites with metrics {len(S)} | dev {(S.split=='dev').sum()} holdout {(S.split=='holdout').sum()}")
del mon, m; gc.collect()

print("3) daily chunked: per-reach sigma_SWOT (river_wse_u) + screened site list ...")
use = ['site_no', 'reach_id', 'river_wse', 'river_wse_u', 'Value_EGM2008_m', DISTANCE_COLUMN, 'river_width', 'date']
wse_u = collections.defaultdict(list); paired = collections.Counter()
for ch in pd.read_csv(DAY, sep='\t', usecols=use, chunksize=400000, low_memory=False):
    ch['river_wse_u'] = pd.to_numeric(ch['river_wse_u'], errors='coerce')
    ch[DISTANCE_COLUMN] = pd.to_numeric(ch[DISTANCE_COLUMN], errors='coerce')
    ch['rw'] = pd.to_numeric(ch['river_wse'], errors='coerce'); ch['gw'] = pd.to_numeric(ch['Value_EGM2008_m'], errors='coerce')
    ch = ch[ch[DISTANCE_COLUMN] <= MAX_DISTANCE_M]
    for rid, g in ch.dropna(subset=['river_wse_u']).groupby('reach_id'):
        wse_u[rid].extend(g.river_wse_u.tolist())
    for sid, g in ch.dropna(subset=['rw', 'gw']).groupby('site_no'):
        paired[sid] += len(g)
    del ch; gc.collect()
sig = pd.Series({r: np.nanmedian(v) for r, v in wse_u.items()}, name='sigma_swot')
S = S.merge(sig.rename_axis('reach_id').reset_index(), on='reach_id', how='left')
print(f"   reaches with sigma_swot {sig.notna().sum()} | sigma med {sig.median():.3f} m | IQR fold {sig.quantile(.75)/sig.quantile(.25):.1f}x | max/min {sig.max()/sig.min():.0f}x")
del wse_u; gc.collect()

S.to_parquet(OUT / "08_global_coupling_metrics.parquet", index=False)

# ---- headline summary (reach-level: median per reach) ----
def reach_agg(df):
    base = df.groupby('reach_id').agg(seasonal_r=('seasonal_r', 'median'),
        deseason_r=('deseason_r', 'median'), uniq_river=('uniq_river', 'median'), uniq_rech=('uniq_rech', 'median'),
        aridity=('aridity', 'median'), width=('river_width', 'median'),
        split=('split', 'first')).reset_index()
    at = aggregate_reach_at(df).rename(columns={'tau_days': 'tau'})
    return base.merge(
        at[['reach_id', 'A', 'tau', 'phase_concentration_R', 'n_phase_sites']],
        on='reach_id', how='left')
def phase_conc(reach):
    _, concentration = circular_mean_days(reach.tau.to_numpy())
    return concentration
def boot_frac(x, fn, n=10_000, seed=0):
    rng = np.random.default_rng(seed); x = np.asarray(x)
    vals = [fn(x[rng.integers(0, len(x), len(x))]) for _ in range(n)]
    return float(np.nanpercentile(vals, 2.5)), float(np.nanpercentile(vals, 97.5))

def block(df, name):
    rc = reach_agg(df); rc = rc[np.isfinite(rc.A) | np.isfinite(rc.tau)]
    al = ((rc.A > 1.2) | (rc.tau <= 0))
    climate = rc[['A', 'aridity']].dropna()
    climate_result = spearmanr(climate.A, climate.aridity) if len(climate) >= 3 else None
    out = dict(n_reach=int(len(rc)),
        amplified_or_lead_frac=float(al.mean()),
        seasonal_r_median=float(rc.seasonal_r.median()),
        seasonal_r_frac_gt0p5=float((rc.seasonal_r > 0.5).mean()),
        deseason_r_median=float(rc.deseason_r.median()),
        deseason_r_frac_ge0p3=float((rc.deseason_r >= 0.3).mean()),
        phase_concentration_r=phase_conc(rc),
        reach_phase_concentration_R_median=float(rc.phase_concentration_R.median()),
        uniq_river_median=float(rc.uniq_river.median()),
        uniq_rech_median=float(rc.uniq_rech.median()),
        A_vs_aridity_spearman=dict(
            rho=float(climate_result.statistic) if climate_result is not None else np.nan,
            p_value=float(climate_result.pvalue) if climate_result is not None else np.nan,
            n_reach=int(len(climate))))
    d = (rc.uniq_rech - rc.uniq_river).dropna()
    if len(d) > 10: out['rech_minus_river_uniq_CI'] = list(boot_frac(d.values, np.nanmedian))
    return out

summary = dict(scope=(f"global SWOT x in-situ well, 2023-2025, datum-free relative metrics, "
                      f"<={MAX_DISTANCE_KM:g}km by {DISTANCE_COLUMN}, reach-level"),
    harmonic_eligibility=harmonic_criterion(),
    distance_definition=dict(column=DISTANCE_COLUMN, maximum_m=MAX_DISTANCE_M),
    aridity_definition=dict(
        metric='sum(precipitation_mm) / sum(abs(evapotranspiration_mm))',
        temporal_window='fixed 2023-01 through 2025-12 natural months',
        months_per_site=int(len(CLIMATE_MONTHS)),
        computed_before_water_level_pairwise_screen=True,
        climate_month_completeness_required=True),
    dev_countries=sorted(DEV_COUNTRIES),
    sigma_swot=dict(n_reach=int(sig.notna().sum()), median_m=float(sig.median()),
        iqr_fold=float(sig.quantile(.75) / sig.quantile(.25)), maxmin_fold=float(sig.max() / sig.min())),
    ALL=block(S, 'all'), DEV=block(S[S.split == 'dev'], 'dev'), HOLDOUT=block(S[S.split == 'holdout'], 'holdout'))
json.dump(summary, open(OUT / "08_summary.json", 'w'), indent=2)
json.dump(dict(dev_countries=sorted(DEV_COUNTRIES),
    dev_sites=int((S.split == 'dev').sum()), holdout_sites=int((S.split == 'holdout').sum()),
    holdout_countries=sorted(S[S.split == 'holdout'].country.unique().tolist())), open(OUT / "08_holdout.json", 'w'), indent=2)
print("\n=== SUMMARY ==="); print(json.dumps(summary, indent=2))
print("\nwrote output/08_global_coupling_metrics.parquet, 08_summary.json, 08_holdout.json")
