#!/usr/bin/env python3
"""Step 2 — WSE residual variability + synchronous sub-seasonal co-variability.

From the raw daily matched file (chunked, memory-safe): (a) per-reach
sigma_swot_emp (legacy internal column name), the residual scale (MAD) of river_wse about its
annual+semiannual harmonic, after collapsing cross-well duplicates to one
Reach/Node/date river observation. This statistic contains both unresolved
sub-seasonal hydrologic variability and measurement error; it is not a pure
SWOT measurement-noise estimate. (b) Per-site correlations computed on one
exactly shared set of pairwise-finite daily dates: raw observations
(``raw_r_same_sample``), annual+semiannual periodic reconstructions without
intercept/trend (``seasonal_harmonic_r_same_sample``), and full-model residuals
(``event_r``; >= 24 paired observations). Both are merged into the pair table
from step 1.

Inputs : Phase-1 5matched_results_daily.txt
         output/08_global_coupling_metrics.parquet (from step 1)
Outputs: output/08_global_coupling_metrics.parquet (adds
         sigma_swot_emp, the three same-sample correlations, n_obs, and
         harmonic-fit diagnostics — this is the final pair table), updates
         output/08_summary.json, and
         output/08_event_metrics_all_sites.parquet, the standalone daily
         event-r population with map metadata and a frozen Reach-level
         development/geographic-holdout split

Requires: numpy, pandas, pyarrow.
"""
import collections
import gc
import json

import numpy as np
import pandas as pd

from importlib import import_module as _import_module
_harmonic_model = _import_module("10_harmonic_model")
fit_harmonic = _harmonic_model.fit_harmonic
harmonic_criterion = _harmonic_model.harmonic_criterion
_event_site_metadata = _import_module("11_event_site_metadata")
EventSiteMetadata = _event_site_metadata.EventSiteMetadata
_pipeline_config = _import_module("09_pipeline_config")
DAY = _pipeline_config.DAY
DISTANCE_COLUMN = _pipeline_config.DISTANCE_COLUMN
HARMONIC_MIN_POINTS = _pipeline_config.HARMONIC_MIN_POINTS
MAX_DISTANCE_KM = _pipeline_config.MAX_DISTANCE_KM
MAX_DISTANCE_M = _pipeline_config.MAX_DISTANCE_M
OUT = _pipeline_config.OUT

SIGMA_NUMERICAL_ZERO_M = 1e-9
EVENT_MIN_PAIRED_DAYS = 24
ANNUAL_AMPLITUDE_SENSITIVITY_M = 0.01
PERIODIC_SD_FRACTION_SENSITIVITY = 0.10
DEV_COUNTRIES = {"Germany", "China"}

# Step-2 columns copied from the all-daily-site table into the monthly-eligible
# pair table. Keeping the list explicit makes reruns idempotent: all stale
# versions (including pandas merge suffixes) are removed before the new merge.
DAILY_PAIR_METRIC_COLUMNS = (
    'raw_r_same_sample',
    'seasonal_harmonic_r_same_sample',
    'event_r',
    'n_obs',
    'span_years_same_sample',
    'harmonic_rank_same_sample',
    'condition_number_same_sample',
    'river_annual_amplitude_m_same_sample',
    'groundwater_annual_amplitude_m_same_sample',
    'river_semiannual_amplitude_m_same_sample',
    'groundwater_semiannual_amplitude_m_same_sample',
    'river_periodic_sd_m_same_sample',
    'groundwater_periodic_sd_m_same_sample',
    'river_raw_sd_m_same_sample',
    'groundwater_raw_sd_m_same_sample',
    'river_residual_sd_m_same_sample',
    'groundwater_residual_sd_m_same_sample',
    'river_harmonic_r2_same_sample',
    'groundwater_harmonic_r2_same_sample',
)


def yfrac(d): d = pd.DatetimeIndex(d); return d.year.values + (d.dayofyear.values - 1) / 365.25
def country(s): return str(s).split('+')[0].split('_')[0]
def harm_resid(t, x, fr=(1., 2.)):
    result = fit_harmonic(t, x, frequencies=fr)
    if result is None:
        return None, None
    r = result['residual']
    return r, 1.4826 * np.median(np.abs(r - np.median(r)))


def finite_correlation(x, y):
    """Pearson r for two already aligned vectors, or NaN if not identifiable."""
    x = np.asarray(x, dtype=float)
    y = np.asarray(y, dtype=float)
    if len(x) != len(y) or len(x) < 2:
        return np.nan
    if not (np.isfinite(x).all() and np.isfinite(y).all()):
        return np.nan
    if np.std(x) <= 0 or np.std(y) <= 0:
        return np.nan
    value = float(np.corrcoef(x, y)[0, 1])
    return value if np.isfinite(value) else np.nan


print(f"accumulate daily (<={MAX_DISTANCE_KM:g} km) per-reach river_wse and per-site paired river+gw ...")
use = ['site_no', 'reach_id', 'node_id', 'river_wse', 'Value_EGM2008_m',
       DISTANCE_COLUMN, 'river_width', 'wells_lon', 'wells_lat', 'date']
river_frames = []  # screened river rows; globally de-duplicated after chunked read
site_pair = collections.defaultdict(list)  # site -> list[(t, rw, gw)]; carries reach_id,width
metadata_accumulator = EventSiteMetadata()
for ch in pd.read_csv(DAY, sep='\t', usecols=use, chunksize=500000, low_memory=False):
    ch[DISTANCE_COLUMN] = pd.to_numeric(ch[DISTANCE_COLUMN], errors='coerce')
    ch['rw'] = pd.to_numeric(ch['river_wse'], errors='coerce')
    ch['gw'] = pd.to_numeric(ch['Value_EGM2008_m'], errors='coerce')
    ch['obs_date'] = pd.to_datetime(ch['date'], errors='coerce').dt.normalize()
    ch['t'] = yfrac(ch['obs_date'])
    ch = ch[(ch[DISTANCE_COLUMN] <= MAX_DISTANCE_M) & np.isfinite(ch.t)]
    river_frames.append(
        ch.dropna(subset=['reach_id', 'obs_date', 'rw'])[
            ['reach_id', 'node_id', 'obs_date', 't', 'rw']
        ].copy()
    )
    for sid, g in ch.dropna(subset=['rw', 'gw']).groupby('site_no'):
        site_pair[sid].extend(zip(g.t.tolist(), g.rw.tolist(), g.gw.tolist()))
        metadata_accumulator.update(sid, g)
    del ch; gc.collect()

# The same complete paired-day population supplies both event_r and metadata.
site_meta = metadata_accumulator.finalize()
del metadata_accumulator

# The daily table is a well-river pair table.  The same physical SWOT
# Reach/Node/date observation can therefore occur once per nearby well.  Keep
# one river value per physical observation before fitting a Reach-level
# variability scale. ``dropna=False`` retains the small legacy subset without node_id as
# one Reach/date observation rather than silently dropping those reaches.
river_rows = pd.concat(river_frames, ignore_index=True)
raw_river_row_count = len(river_rows)
raw_counts = river_rows.groupby('reach_id').size()
river_obs = (
    river_rows.groupby(
        ['reach_id', 'node_id', 'obs_date'],
        as_index=False,
        dropna=False,
        sort=False,
    )
    .agg(t=('t', 'first'), rw=('rw', 'mean'))
)
unique_counts = river_obs.groupby('reach_id').size()
affected_reaches = int(
    raw_counts.sub(unique_counts, fill_value=0).gt(0).sum()
)
print(
    f"  river observations {raw_river_row_count:,} pair rows -> "
    f"{len(river_obs):,} unique Reach/Node/date rows "
    f"({affected_reaches:,} reaches de-weighted) | paired sites {len(site_pair)}"
)
del river_frames, river_rows, raw_counts, unique_counts; gc.collect()

print("per-reach sub-seasonal WSE residual variability (median Node harmonic-residual MAD) ...")
# Absolute WSE differs systematically along a Reach because upstream and
# downstream Nodes have different water-surface elevations. Fitting one
# harmonic to all Nodes would count that longitudinal spatial gradient as
# temporal residual variability. Fit each physical Node time series first and
# use the median valid Node scale as the Reach descriptor.
node_scales = []
for (rid, nid), g in river_obs.groupby(
        ['reach_id', 'node_id'], sort=False, dropna=False):
    if len(g) < HARMONIC_MIN_POINTS:
        continue
    _, scale = harm_resid(g.t.to_numpy(), g.rw.to_numpy())
    if scale is not None and np.isfinite(scale):
        node_scales.append((rid, nid, float(scale)))
node_scale = pd.DataFrame(node_scales, columns=['reach_id', 'node_id', 'scale'])
if node_scale.empty:
    sig = pd.Series(dtype=float, name='sigma_swot_emp')
    sig_n_nodes = pd.Series(dtype='int64', name='sigma_swot_emp_n_nodes')
else:
    sig = node_scale.groupby('reach_id', sort=False).scale.median()
    sig_n_nodes = node_scale.groupby('reach_id', sort=False).size()
sig = sig.mask(sig.abs() <= SIGMA_NUMERICAL_ZERO_M, 0.0)
sig_pos = sig[sig > 0]
iqr_fold = sig_pos.quantile(.75) / sig_pos.quantile(.25) if len(sig_pos) else np.nan
p5_p95_fold = sig_pos.quantile(.95) / sig_pos.quantile(.05) if len(sig_pos) else np.nan
maxmin_fold = sig_pos.max() / sig_pos.min() if len(sig_pos) else np.nan
print(f"  reaches with sub-seasonal WSE residual variability {len(sig)} "
      f"(internal column sigma_swot_emp; zero={int((sig == 0).sum())}) | "
      f"median {sig.median():.3f} m | positive-only IQR fold {iqr_fold:.1f}x | "
      f"p5-p95 fold {p5_p95_fold:.0f}x | max/min {maxmin_fold:.0f}x")
if len(sig_n_nodes):
    print(f"  contributing valid Nodes per Reach: median {sig_n_nodes.median():.0f}, "
          f"maximum {sig_n_nodes.max():.0f}")
del river_obs; gc.collect()

print("same-date raw, seasonal-harmonic, and residual river-GW correlations per site (>=24 paired obs) ...")
ev = []
for sid, vals in site_pair.items():
    if len(vals) < EVENT_MIN_PAIRED_DAYS:
        continue
    a = np.asarray(vals, dtype=float)
    t, rw, gw = a[:, 0], a[:, 1], a[:, 2]

    # Apply one unconditional common-finite mask before either fit. Equal
    # residual lengths after independent filtering would not prove that the
    # retained dates were identical if non-finite values occurred at different
    # positions in the river and groundwater series.
    common = np.isfinite(t) & np.isfinite(rw) & np.isfinite(gw)
    if int(common.sum()) < EVENT_MIN_PAIRED_DAYS:
        continue
    t, rw, gw = t[common], rw[common], gw[common]
    if pd.Index(t).duplicated().any():
        raise ValueError(
            f"site {sid} contains repeated daily dates after the Phase-1 "
            "Top-1 screen; same-date correlations require one value per date"
        )
    river_fit = fit_harmonic(t, rw)
    groundwater_fit = fit_harmonic(t, gw)
    if river_fit is None or groundwater_fit is None:
        continue
    if river_fit['n'] != groundwater_fit['n'] or river_fit['n'] != len(t):
        raise RuntimeError(f"same-sample harmonic alignment failed for site {sid}")

    raw_r = finite_correlation(rw, gw)
    periodic_r = finite_correlation(
        river_fit['periodic_fitted'], groundwater_fit['periodic_fitted'])
    event_r = finite_correlation(
        river_fit['residual'], groundwater_fit['residual'])
    # Preserve the existing event-r population: a zero/near-zero periodic
    # component can make periodic_r unavailable without invalidating event_r.
    if not np.isfinite(event_r):
        continue

    ev.append(dict(
        site_no=sid,
        reach_id=site_meta[sid][0],
        river_width=site_meta[sid][1],
        country=country(sid),
        lon=site_meta[sid][2],
        lat=site_meta[sid][3],
        raw_r_same_sample=raw_r,
        seasonal_harmonic_r_same_sample=periodic_r,
        event_r=event_r,
        n_obs=int(len(t)),
        span_years_same_sample=float(river_fit['span_years']),
        harmonic_rank_same_sample=int(river_fit['rank']),
        condition_number_same_sample=float(river_fit['condition_number']),
        river_annual_amplitude_m_same_sample=float(river_fit['amplitudes'][0]),
        groundwater_annual_amplitude_m_same_sample=float(groundwater_fit['amplitudes'][0]),
        river_semiannual_amplitude_m_same_sample=float(river_fit['amplitudes'][1]),
        groundwater_semiannual_amplitude_m_same_sample=float(groundwater_fit['amplitudes'][1]),
        river_periodic_sd_m_same_sample=float(np.std(river_fit['periodic_fitted'])),
        groundwater_periodic_sd_m_same_sample=float(np.std(groundwater_fit['periodic_fitted'])),
        river_raw_sd_m_same_sample=float(np.std(rw)),
        groundwater_raw_sd_m_same_sample=float(np.std(gw)),
        river_residual_sd_m_same_sample=float(np.std(river_fit['residual'])),
        groundwater_residual_sd_m_same_sample=float(np.std(groundwater_fit['residual'])),
        river_harmonic_r2_same_sample=float(river_fit['r2']),
        groundwater_harmonic_r2_same_sample=float(groundwater_fit['r2']),
    ))
E = pd.DataFrame(ev)
if E.empty:
    raise ValueError("no sites passed the independent daily event-r criteria")
if E.site_no.duplicated().any():
    raise ValueError("independent daily event-r population contains duplicate site_no rows")
if E[['country', 'lon', 'lat']].isna().any().any():
    raise ValueError("independent daily event-r population has missing map metadata")

# Freeze the development/geographic-holdout assignment on this standalone
# daily population.  As in step 1, a Reach is development iff at least half of
# its eligible sites come from the pre-specified Germany + China development
# countries; the resulting Reach label is then applied to every site on it.
E['is_dev'] = E.country.isin(DEV_COUNTRIES)
reach_dev = E.groupby('reach_id', sort=False).is_dev.mean().ge(0.5)
E['split'] = np.where(E.reach_id.map(reach_dev), 'dev', 'holdout')
if not E.groupby('reach_id').split.nunique(dropna=False).eq(1).all():
    raise AssertionError("a Reach appears in multiple daily event-r splits")

# Attach the Reach-level WSE residual scale to the *complete* independent daily
# event population before it is frozen.  Downstream regional/mechanism analyses
# must not have to recover these fields from the narrower monthly-harmonic table
# S, because doing so would silently restore that eligibility gate.
E = E.merge(
    sig.rename('sigma_swot_emp').rename_axis('reach_id').reset_index(),
    on='reach_id', how='left', validate='many_to_one')
E = E.merge(
    sig_n_nodes.rename('sigma_swot_emp_n_nodes').rename_axis('reach_id').reset_index(),
    on='reach_id', how='left', validate='many_to_one')
if E[['sigma_swot_emp', 'sigma_swot_emp_n_nodes']].isna().any().any():
    missing_scale = int(E['sigma_swot_emp'].isna().sum())
    missing_nodes = int(E['sigma_swot_emp_n_nodes'].isna().sum())
    raise ValueError(
        "complete daily event-r population lacks Reach WSE residual diagnostics: "
        f"sigma missing={missing_scale}, n_nodes missing={missing_nodes}")
event_reach_count = int(E.reach_id.nunique())
event_holdout_reach_count = int(
    E.loc[E.split.eq('holdout'), 'reach_id'].nunique()
)
print(
    f"  sites with event_r {len(E)} | Reaches {event_reach_count} "
    f"(holdout {event_holdout_reach_count}) | "
    f"same-date raw/harmonic/event medians "
    f"{E.raw_r_same_sample.median():.3f}/"
    f"{E.seasonal_harmonic_r_same_sample.median():.3f}/"
    f"{E.event_r.median():.3f} | "
    f"event frac>=0.3 {(E.event_r>=0.3).mean():.2f} | "
    f"event frac<=0 {(E.event_r<=0).mean():.2f}"
)
E.to_parquet(OUT / "08_event_metrics_all_sites.parquet", index=False)
del site_pair; gc.collect()

# merge into metrics
S = pd.read_parquet(OUT / "08_global_coupling_metrics.parquet")
derived_names = (
    'sigma_swot_emp',
    'sigma_swot_emp_n_nodes',
    *DAILY_PAIR_METRIC_COLUMNS,
)
stale = [c for c in S.columns if any(c == name or c.startswith(name + '_') for name in derived_names)]
S = S.drop(columns=stale, errors='ignore')
if S.site_no.duplicated().any():
    raise ValueError("monthly-harmonic pair table contains duplicate site_no rows")

# A site identifier is not, by itself, a sufficient relational key for a
# hydrologic pair: silently attaching a daily metric to a different Reach would
# contaminate both the site intersection and the subsequent Reach median.  Make
# any disagreement explicit, then merge on the full site-Reach key.
site_reach_check = S[['site_no', 'reach_id']].merge(
    E[['site_no', 'reach_id']],
    on='site_no', how='inner', validate='one_to_one',
    suffixes=('_monthly', '_daily'),
)
reach_equal = (
    site_reach_check.reach_id_monthly.eq(site_reach_check.reach_id_daily)
    | (
        site_reach_check.reach_id_monthly.isna()
        & site_reach_check.reach_id_daily.isna()
    )
)
if not reach_equal.all():
    raise ValueError(
        "monthly and daily tables disagree on reach_id for "
        f"{int((~reach_equal).sum())} shared sites"
    )

S = S.merge(
    sig.rename('sigma_swot_emp').rename_axis('reach_id').reset_index(),
    on='reach_id', how='left', validate='many_to_one',
)
S = S.merge(
    sig_n_nodes.rename('sigma_swot_emp_n_nodes').rename_axis('reach_id').reset_index(),
    on='reach_id', how='left', validate='many_to_one',
)
S = S.merge(
    E[['site_no', 'reach_id', *DAILY_PAIR_METRIC_COLUMNS]],
    on=['site_no', 'reach_id'], how='left', validate='one_to_one',
)
S.to_parquet(OUT / "08_global_coupling_metrics.parquet", index=False)

# update summary
summ = json.load(open(OUT / "08_summary.json"))
summ.pop('sigma_swot_empirical', None)  # remove the former, misleading semantic name
summ['wse_residual_variability'] = dict(
    internal_column='sigma_swot_emp',
    definition=('median across valid Nodes of 1.4826 x MAD of Node-level WSE '
                'residuals after annual+semiannual harmonic removal'),
    interpretation=('sub-seasonal hydrologic variability plus measurement error; '
                    'not a pure SWOT measurement-noise estimate; Node-specific '
                    'fitting removes longitudinal between-Node WSE offsets'),
    n_reach=int(sig.notna().sum()), median_m=float(sig.median()),
    n_zero=int((sig == 0).sum()),
    numerical_zero_threshold_m=SIGMA_NUMERICAL_ZERO_M,
    iqr_fold=float(iqr_fold),
    p5_p95_fold=float(p5_p95_fold),
    maxmin_fold=float(maxmin_fold))
summ['harmonic_eligibility'] = harmonic_criterion()
# Remove the obsolete monthly-raw versus daily-residual cross-sampling summary.
# Fig. 1 is supported by ``same_sample_correlation`` below, whose three metrics
# use identical dates in the complete independent daily population.
summ.pop('event_scale', None)

# Same-date daily comparison. All three correlations are computed at the site
# level from exactly the same pairwise-finite daily dates.  Start from E, the
# complete independently daily-eligible population, rather than S, the monthly
# harmonic population. Filter the three-metric finite intersection before
# Reach aggregation so each Reach median uses the same constituent sites for
# all three metrics without imposing monthly eligibility.
same_sample_metrics = [
    'raw_r_same_sample',
    'seasonal_harmonic_r_same_sample',
    'event_r',
]
same_sample = E.dropna(subset=same_sample_metrics).copy()


def same_sample_block(df):
    rc = df.groupby('reach_id').agg(
        raw_r_same_sample=('raw_r_same_sample', 'median'),
        seasonal_harmonic_r_same_sample=(
            'seasonal_harmonic_r_same_sample', 'median'),
        event_r=('event_r', 'median'),
    ).reset_index()
    harmonic_minus_event = (
        rc.seasonal_harmonic_r_same_sample - rc.event_r)
    raw_minus_event = rc.raw_r_same_sample - rc.event_r
    return dict(
        n_site=int(df.site_no.nunique()),
        n_reach=int(len(rc)),
        raw_r_same_sample_median=float(rc.raw_r_same_sample.median()),
        seasonal_harmonic_r_same_sample_median=float(
            rc.seasonal_harmonic_r_same_sample.median()),
        event_r_median=float(rc.event_r.median()),
        seasonal_harmonic_minus_event_difference_of_population_medians=float(
            rc.seasonal_harmonic_r_same_sample.median()
            - rc.event_r.median()),
        seasonal_harmonic_minus_event_median_paired_reach_difference=float(
            harmonic_minus_event.median()),
        raw_minus_event_difference_of_population_medians=float(
            rc.raw_r_same_sample.median() - rc.event_r.median()),
        raw_minus_event_median_paired_reach_difference=float(
            raw_minus_event.median()),
        fraction_reaches_seasonal_harmonic_gt_event=float(
            (harmonic_minus_event > 0).mean()),
        fraction_reaches_raw_gt_event=float((raw_minus_event > 0).mean()),
    )


river_periodic_fraction = (
    same_sample.river_periodic_sd_m_same_sample
    / same_sample.river_raw_sd_m_same_sample
)
groundwater_periodic_fraction = (
    same_sample.groundwater_periodic_sd_m_same_sample
    / same_sample.groundwater_raw_sd_m_same_sample
)
low_annual_amplitude = (
    same_sample.river_annual_amplitude_m_same_sample
    < ANNUAL_AMPLITUDE_SENSITIVITY_M
) | (
    same_sample.groundwater_annual_amplitude_m_same_sample
    < ANNUAL_AMPLITUDE_SENSITIVITY_M
)
low_periodic_fraction = (
    river_periodic_fraction < PERIODIC_SD_FRACTION_SENSITIVITY
) | (
    groundwater_periodic_fraction < PERIODIC_SD_FRACTION_SENSITIVITY
)

summ['same_sample_correlation'] = dict(
    source_table='08_event_metrics_all_sites.parquet',
    monthly_harmonic_eligibility_required=False,
    sample_definition=(
        'complete independently daily-eligible site-Reach population with finite '
        'raw_r_same_sample, seasonal_harmonic_r_same_sample, and event_r; all '
        'three are calculated from exactly the same pairwise-finite daily dates '
        'at each site, then median-aggregated by Reach; monthly-harmonic '
        'eligibility is not imposed'),
    minimum_paired_daily_observations=EVENT_MIN_PAIRED_DAYS,
    metric_definitions=dict(
        raw_r_same_sample=(
            'Pearson correlation of raw river stage and groundwater level on '
            'the shared daily dates'),
        seasonal_harmonic_r_same_sample=(
            'Pearson correlation of the fitted annual+semiannual periodic '
            'components on the shared daily dates; intercept and trend excluded'),
        event_r=(
            'Pearson correlation of residuals after removing intercept, trend, '
            'annual, and semiannual terms on the shared daily dates'),
    ),
    difference_definitions=dict(
        difference_of_population_medians=(
            'median over Reach values of the first metric minus the separate '
            'median over Reach values of event_r'),
        median_paired_reach_difference=(
            'median over Reaches of the within-Reach first-metric minus event_r '
            'difference'),
        interpretation=(
            'descriptive correlation contrasts, not an additive partition of '
            'variance attributable to seasonality'),
    ),
    identifiability_sensitivity=dict(
        policy='diagnostic only; neither threshold excludes the main analysis',
        population=(
            'three-metric finite intersection within the complete independent '
            'daily population'),
        annual_amplitude_threshold_m=ANNUAL_AMPLITUDE_SENSITIVITY_M,
        n_site_any_series_below_annual_amplitude_threshold=int(
            low_annual_amplitude.sum()),
        periodic_sd_over_raw_sd_threshold=PERIODIC_SD_FRACTION_SENSITIVITY,
        n_site_any_series_below_periodic_sd_fraction_threshold=int(
            low_periodic_fraction.sum()),
        n_site_evaluated=int(len(same_sample)),
    ),
    ALL=same_sample_block(same_sample),
    DEV=same_sample_block(same_sample[same_sample.split == 'dev']),
    HOLDOUT=same_sample_block(same_sample[same_sample.split == 'holdout']),
)
json.dump(summ, open(OUT / "08_summary.json", 'w'), indent=2)
print("\n=== wse_residual_variability + same_sample_correlation ===")
print(json.dumps({
    'wse_residual_variability': summ['wse_residual_variability'],
    'same_sample_correlation': summ['same_sample_correlation'],
}, indent=2))
print("merged into output/08_global_coupling_metrics.parquet; wrote "
      "08_event_metrics_all_sites.parquet; updated output/08_summary.json")
