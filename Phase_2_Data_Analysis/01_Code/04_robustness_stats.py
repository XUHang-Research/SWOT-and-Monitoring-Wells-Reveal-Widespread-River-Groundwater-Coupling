#!/usr/bin/env python3
"""Step 4 — statistical robustness of the headline numbers.

Reach-level bootstrap 95% CIs with metric-specific monthly, daily, and
same-sample denominators; single-Reach A/tau
uncertainty from a paired calendar-block wild bootstrap; circular Reach phase
aggregation; the A/tau
threshold-robustness grid; distance robustness (Reach/Node independently
rematched inside each 0.5/1/2/5-km threshold before A and tau are recomputed);
difference-aware paired Reach bootstraps for raw, seasonal-harmonic, and
event-residual correlations calculated on identical daily observations;
deseasonalization-method robustness
(three alternative deseasonalizations of the seasonal->sub-seasonal collapse);
and a synthetic harmonic sign test (known lag/lead recovered with the correct
sign). No re-analysis of the headline pipeline — this only quantifies
uncertainty/robustness around the existing numbers.

Inputs : output/08_global_coupling_metrics.parquet (monthly-harmonic table)
         output/08_event_metrics_all_sites.parquet (standalone daily event-r)
         Phase-1 5-km pre-ranking candidate table
Outputs: output/08_robustness.json (extended by step 5)

Requires: numpy, pandas, pyarrow, dask.
"""
import json

import numpy as np
import pandas as pd

from importlib import import_module as _import_module
_harmonic_model = _import_module("10_harmonic_model")
fit_harmonic = _harmonic_model.fit_harmonic
harmonic_criterion = _harmonic_model.harmonic_criterion
_pipeline_config = _import_module("09_pipeline_config")
DISTANCE_COLUMN = _pipeline_config.DISTANCE_COLUMN
HARMONIC_MIN_POINTS = _pipeline_config.HARMONIC_MIN_POINTS
MAX_DISTANCE_M = _pipeline_config.MAX_DISTANCE_M
MON = _pipeline_config.MON
OUT = _pipeline_config.OUT
ROBUSTNESS_CACHE = _pipeline_config.ROBUSTNESS_CACHE
ROBUSTNESS_CANDIDATES = _pipeline_config.ROBUSTNESS_CANDIDATES
ROBUSTNESS_SERIES_CANDIDATES = _pipeline_config.ROBUSTNESS_SERIES_CANDIDATES
_distance_robustness_rematch = _import_module("15_distance_robustness_rematch")
THRESHOLDS_M = _distance_robustness_rematch.THRESHOLDS_M
VALUE_COLUMNS = _distance_robustness_rematch.VALUE_COLUMNS
load_or_build_strict_monthly = _distance_robustness_rematch.load_or_build_strict_monthly
_phase_statistics = _import_module("13_phase_statistics")
aggregate_reach_at = _phase_statistics.aggregate_reach_at
build_reach_at_uncertainty = _phase_statistics.build_reach_at_uncertainty
circular_mean_days = _phase_statistics.circular_mean_days
summarize_reach_at_uncertainty = _phase_statistics.summarize_reach_at_uncertainty
DEV = {"Germany", "China"}
RNG = np.random.default_rng(0)
SAME_SAMPLE_BOOTSTRAP_N = 10_000
EVENT_BOOTSTRAP_SEED = 20260808
EVENT_BOOTSTRAP_N = 10_000
EVENT_BOOTSTRAP_SEEDS = {
    'ALL': EVENT_BOOTSTRAP_SEED,
    'DEV': EVENT_BOOTSTRAP_SEED + 2,
    'HOLDOUT': EVENT_BOOTSTRAP_SEED + 1,
}
# The three-correlation comparison now uses the complete daily population.
# Reusing the standalone event-r seeds makes its event-r intervals exactly
# reproducible across the two output blocks, rather than merely statistically
# equivalent under different random draws.
SAME_SAMPLE_BOOTSTRAP_SEEDS = EVENT_BOOTSTRAP_SEEDS.copy()
SAME_SAMPLE_COLUMNS = (
    'raw_r_same_sample',
    'seasonal_harmonic_r_same_sample',
    'event_r',
)


def country(s): return str(s).split('+')[0].split('_')[0]
def yfrac(d):
    d = pd.DatetimeIndex(d)
    return d.year.values + (d.dayofyear.values - 1) / 365.25


def harm(t, x, fr=(1., 2.)):
    """annual+semiannual+trend fit -> amplitude (annual), phase (annual rad). Matches step 1."""
    return fit_harmonic(t, x, frequencies=fr)


def reach_table(S):
    """Reach table with median A and equal-weight circular-mean tau."""
    base = S.groupby('reach_id').agg(
        seasonal_r=('seasonal_r', 'median'), event_r=('event_r', 'median'),
        split=('split', 'first')).reset_index()
    phase = aggregate_reach_at(S).rename(columns={'tau_days': 'tau'})
    rc = base.merge(
        phase[['reach_id', 'A', 'tau', 'phase_concentration_R', 'n_phase_sites']],
        on='reach_id', how='left')
    return rc[np.isfinite(rc.A) | np.isfinite(rc.tau)]


def seasonal_reach_table(S):
    """Reach medians for every finite monthly seasonal correlation."""
    finite = S.dropna(subset=['seasonal_r']).copy()
    if finite.empty:
        raise ValueError('monthly seasonal-r population is empty')
    if finite.site_no.duplicated().any():
        raise ValueError('monthly seasonal-r population has duplicate sites')
    if not finite.groupby('reach_id').split.nunique(dropna=False).eq(1).all():
        raise ValueError('a monthly seasonal-r Reach crosses frozen splits')
    reach = finite.groupby('reach_id').agg(
        seasonal_r=('seasonal_r', 'median'),
        split=('split', 'first'),
    ).reset_index()
    return finite, reach


def same_sample_reach_table(event):
    """Fix the three-correlation population before Reach aggregation.

    The raw, seasonal-harmonic reconstruction, and event-residual
    correlations are all calculated from the same daily observations in step
    2.  Requiring all three site values here prevents a zero-variance or
    otherwise undefined correlation from silently changing the wells
    represented within a Reach for just one metric.
    """
    required = ('site_no', 'reach_id', 'split', *SAME_SAMPLE_COLUMNS)
    missing = [column for column in required if column not in event]
    if missing:
        raise KeyError(
            'same-sample correlation columns are missing from '
            f'08_event_metrics_all_sites.parquet: {missing}'
        )
    paired = event.dropna(subset=list(SAME_SAMPLE_COLUMNS)).copy()
    if paired.empty:
        raise ValueError('no site pairs have all three same-sample correlations')
    if paired.duplicated(['site_no', 'reach_id']).any():
        raise ValueError('duplicate site_no x reach_id rows in same-sample population')
    if paired.duplicated(['site_no']).any():
        raise ValueError('a Top-1 site is assigned to multiple same-sample Reaches')
    split_counts = paired.groupby('reach_id').split.nunique(dropna=False)
    if not split_counts.eq(1).all():
        raise ValueError('a Reach appears in multiple development/holdout splits')
    reach = paired.groupby('reach_id', as_index=False).agg(
        raw_r_same_sample=('raw_r_same_sample', 'median'),
        seasonal_harmonic_r_same_sample=(
            'seasonal_harmonic_r_same_sample', 'median'),
        event_r=('event_r', 'median'),
        split=('split', 'first'),
        n_site_pairs=('site_no', 'size'),
    )
    return paired, reach


def _same_sample_ci(values, bootstrap_indices, reducer=np.median):
    """Return [lower, point, upper] using one synchronized Reach bootstrap."""
    values = np.asarray(values, dtype=float)
    draws = reducer(values[bootstrap_indices], axis=1)
    return [
        float(np.percentile(draws, 2.5)),
        float(reducer(values)),
        float(np.percentile(draws, 97.5)),
    ]


def same_sample_block(event, seed):
    """Summarize one split on a fixed site population and paired Reach draws."""
    paired, reach = same_sample_reach_table(event)
    n_reach = len(reach)
    rng = np.random.default_rng(seed)
    bootstrap_indices = rng.integers(
        0, n_reach, size=(SAME_SAMPLE_BOOTSTRAP_N, n_reach))

    metric_medians = {
        f'{column}_median': _same_sample_ci(
            reach[column].to_numpy(), bootstrap_indices)
        for column in SAME_SAMPLE_COLUMNS
    }
    comparisons = {}
    for name, left_name, right_name in (
        (
            'seasonal_harmonic_minus_event',
            'seasonal_harmonic_r_same_sample',
            'event_r',
        ),
        ('raw_minus_event', 'raw_r_same_sample', 'event_r'),
        (
            'seasonal_harmonic_minus_raw',
            'seasonal_harmonic_r_same_sample',
            'raw_r_same_sample',
        ),
    ):
        left = reach[left_name].to_numpy(dtype=float)
        right = reach[right_name].to_numpy(dtype=float)
        paired_difference = left - right
        median_paired_difference = _same_sample_ci(
            paired_difference, bootstrap_indices)

        # This is deliberately separate from median(left - right).  Both use
        # identical resampled Reach indices so the second interval is also a
        # paired, rather than independent, bootstrap interval.
        difference_draws = (
            np.median(left[bootstrap_indices], axis=1)
            - np.median(right[bootstrap_indices], axis=1)
        )
        difference_of_medians = [
            float(np.percentile(difference_draws, 2.5)),
            float(np.median(left) - np.median(right)),
            float(np.percentile(difference_draws, 97.5)),
        ]
        comparisons[name] = {
            'left_metric': left_name,
            'right_metric': right_name,
            'median_paired_difference': median_paired_difference,
            'difference_of_medians': difference_of_medians,
            'fraction_reach_paired_difference_positive': _same_sample_ci(
                (paired_difference > 0).astype(float),
                bootstrap_indices,
                reducer=np.mean,
            ),
        }

    return {
        'n_site_pairs': int(len(paired)),
        'n_reach': int(n_reach),
        'bootstrap_seed': int(seed),
        'metric_medians': metric_medians,
        'event_r_frac_ge0p3': _same_sample_ci(
            (reach.event_r.to_numpy(dtype=float) >= 0.3).astype(float),
            bootstrap_indices,
            reducer=np.mean,
        ),
        'paired_comparisons': comparisons,
    }


def independent_event_reach_table(event):
    """Reach medians for the standalone daily event-r population."""
    required = {'site_no', 'reach_id', 'split', 'event_r'}
    missing = sorted(required.difference(event.columns))
    if missing:
        raise KeyError(
            f'08_event_metrics_all_sites.parquet is missing fields: {missing}'
        )
    finite = event.dropna(subset=['event_r']).copy()
    if finite.empty:
        raise ValueError('standalone daily event-r population is empty')
    if finite.site_no.duplicated().any():
        raise ValueError('standalone daily event-r population has duplicate sites')
    if not finite.split.isin(['dev', 'holdout']).all():
        raise ValueError('standalone daily event-r population has invalid split labels')
    if not finite.groupby('reach_id').split.nunique(dropna=False).eq(1).all():
        raise ValueError('a standalone daily event-r Reach crosses frozen splits')
    reach = finite.groupby('reach_id', as_index=False).agg(
        event_r=('event_r', 'median'),
        split=('split', 'first'),
        n_site_pairs=('site_no', 'size'),
    )
    return finite, reach


def independent_event_block(event, seed):
    """Fixed-seed Reach bootstrap for standalone event-r summaries."""
    finite, reach = independent_event_reach_table(event)
    values = reach.event_r.to_numpy(dtype=float)
    rng = np.random.default_rng(seed)
    indices = rng.integers(
        0, len(values), size=(EVENT_BOOTSTRAP_N, len(values)))
    draws = values[indices]

    def interval(samples, point):
        return [
            float(np.percentile(samples, 2.5)),
            float(point),
            float(np.percentile(samples, 97.5)),
        ]

    return {
        'n_site_pairs': int(len(finite)),
        'n_reach': int(len(reach)),
        'bootstrap_seed': int(seed),
        'event_r_median': interval(
            np.median(draws, axis=1), np.median(values)),
        'event_r_frac_ge0p3': interval(
            np.mean(draws >= 0.3, axis=1), np.mean(values >= 0.3)),
    }


def boot_ci(arr, fn, n=10000):
    arr = np.asarray(arr)
    vals = [fn(arr[RNG.integers(0, len(arr), len(arr))]) for _ in range(n)]
    return [float(np.nanpercentile(vals, 2.5)),
            float(fn(arr)),
            float(np.nanpercentile(vals, 97.5))]


def validate_phase1_anchor(rematched, reference_file, threshold_m, tol=1e-9):
    """Require observation-supported Phase-1 months to reproduce row-for-row.

    Phase 1 now carries a fixed 36-month climate calendar per site.  Months
    inserted only to complete that climate window have neither river nor GW
    observations and are intentionally outside the strict rematching product,
    which reconstructs observation series rather than climate support.
    """
    columns = ['site_no', 'reach_id', 'node_id', 'date', *VALUE_COLUMNS]
    reference = pd.read_csv(reference_file, sep='\t', usecols=columns, low_memory=False)
    reference['date'] = pd.to_datetime(reference['date'], errors='coerce')
    rebuilt = rematched.loc[rematched.threshold_m.eq(threshold_m), columns].copy()
    rebuilt['date'] = pd.to_datetime(rebuilt['date'], errors='coerce')
    reference = reference.loc[reference[VALUE_COLUMNS].notna().any(axis=1)].copy()
    rebuilt = rebuilt.loc[rebuilt[VALUE_COLUMNS].notna().any(axis=1)].copy()

    keys = ['site_no', 'date']
    if reference.duplicated(keys).any() or rebuilt.duplicated(keys).any():
        raise ValueError(f"{threshold_m:g}-m anchor contains duplicate site/month keys")
    joined = reference.merge(
        rebuilt, on=keys, how='outer', suffixes=('_phase1', '_rematch'),
        indicator=True, validate='one_to_one')
    if not joined['_merge'].eq('both').all():
        counts = joined['_merge'].value_counts().to_dict()
        raise ValueError(
            f"{threshold_m:g}-m rematch does not reproduce Phase-1 rows: {counts}")

    for column in ['reach_id', 'node_id']:
        left = pd.to_numeric(joined[f'{column}_phase1'], errors='coerce')
        right = pd.to_numeric(joined[f'{column}_rematch'], errors='coerce')
        equal = left.eq(right) | (left.isna() & right.isna())
        if not bool(equal.all()):
            raise ValueError(
                f"{threshold_m:g}-m rematch changes {column} for "
                f"{int((~equal).sum()):,} site/month rows")

    diagnostics = {
        'threshold_m': int(threshold_m),
        'comparison_population': (
            'site-month rows with at least one finite river/GW value; '
            'climate-only calendar rows excluded'),
        'n_monthly_rows': int(len(joined)),
    }
    for column in VALUE_COLUMNS:
        left = pd.to_numeric(joined[f'{column}_phase1'], errors='coerce')
        right = pd.to_numeric(joined[f'{column}_rematch'], errors='coerce')
        if not left.isna().equals(right.isna()):
            raise ValueError(
                f"{threshold_m:g}-m rematch changes the missing-value pattern of {column}")
        finite = left.notna() & right.notna()
        max_diff = float((left[finite] - right[finite]).abs().max()) if finite.any() else 0.0
        if max_diff > tol:
            raise ValueError(
                f"{threshold_m:g}-m rematch does not reproduce {column}: "
                f"max absolute difference={max_diff:.3e}")
        diagnostics[f'{column}_max_abs_difference'] = max_diff
    return diagnostics


def main():
    out = {'harmonic_eligibility': harmonic_criterion()}
    S = pd.read_parquet(OUT / "08_global_coupling_metrics.parquet")
    EVENT = pd.read_parquet(OUT / "08_event_metrics_all_sites.parquet")
    independent_event_reach_table(EVENT)  # blocking schema/split validation

    # ---------- (0) single-Reach A/tau uncertainty and circular point phase ----------
    print("0) single-Reach A/tau calendar-block wild bootstrap ...", flush=True)
    reach_uncertainty = build_reach_at_uncertainty(
        MON,
        S,
        distance_column=DISTANCE_COLUMN,
        maximum_distance_m=MAX_DISTANCE_M,
    )
    reach_uncertainty.to_parquet(OUT / "08_reach_at_uncertainty.parquet", index=False)
    out['single_reach_at_uncertainty'] = summarize_reach_at_uncertainty(reach_uncertainty)
    print(
        f"   wrote {len(reach_uncertainty):,} Reaches to "
        "08_reach_at_uncertainty.parquet",
        flush=True,
    )
    # Refresh only the A/tau-dependent headline fields. This also upgrades an
    # existing pre-correction summary without requiring another daily-file pass.
    summary_path = OUT / "08_summary.json"
    summary = json.load(open(summary_path))
    for label, subset in (
        ('ALL', reach_uncertainty),
        ('DEV', reach_uncertainty[reach_uncertainty.split == 'dev']),
        ('HOLDOUT', reach_uncertainty[reach_uncertainty.split == 'holdout']),
    ):
        _, population_R = circular_mean_days(subset.tau_days.to_numpy())
        summary[label]['n_reach'] = int(len(subset))
        summary[label]['amplified_or_lead_frac'] = float(
            ((subset.A > 1.2) | (subset.tau_days <= 0)).mean())
        summary[label]['phase_concentration_r'] = population_R
        summary[label]['reach_phase_concentration_R_median'] = float(
            subset.phase_concentration_R.median())
    json.dump(summary, open(summary_path, 'w'), indent=2)

    # ---------- (1) bootstrap CIs by reach on headline fractions ----------
    print("1) bootstrap CIs by reach ...", flush=True)
    ci = {
        'sample_definition': {
            'A_tau': (
                'monthly-harmonic-eligible 08_global_coupling_metrics; '
                'median A and equal-weight circular-mean tau within Reach'),
            'seasonal_r': (
                'every 08_global_coupling_metrics site with finite monthly '
                'seasonal_r, then median seasonal_r within Reach; daily '
                'event-r eligibility is not imposed'),
            'event_r': (
                'complete independently daily-eligible '
                '08_event_metrics_all_sites population; monthly harmonic '
                'eligibility is not imposed; median event_r within Reach'),
            'event_bootstrap_unit': 'Reach',
            'event_bootstrap_replicates': EVENT_BOOTSTRAP_N,
            'event_bootstrap_seed': EVENT_BOOTSTRAP_SEEDS,
            'interval_format': (
                '[2.5th percentile, point estimate, 97.5th percentile]'),
        }
    }
    groups = (
        ('ALL', S, EVENT, EVENT_BOOTSTRAP_SEEDS['ALL']),
        ('DEV', S[S.split == 'dev'], EVENT[EVENT.split == 'dev'],
         EVENT_BOOTSTRAP_SEEDS['DEV']),
        ('HOLDOUT', S[S.split == 'holdout'],
         EVENT[EVENT.split == 'holdout'], EVENT_BOOTSTRAP_SEEDS['HOLDOUT']),
    )
    for label, sub, event_sub, event_seed in groups:
        rc = reach_table(sub)
        amp_lead = ((rc.A > 1.2) | (rc.tau <= 0)).values.astype(float)
        lead = (rc.tau <= 0).values.astype(float)
        seasonal_sites, seasonal_reach = seasonal_reach_table(sub)
        seasonal = seasonal_reach.seasonal_r.to_numpy()
        event_block = independent_event_block(event_sub, event_seed)
        ci[label] = dict(
            denominator={
                'A_tau_n_reach': int(len(rc)),
                'seasonal_r_n_site_pairs': int(len(seasonal_sites)),
                'seasonal_r_n_reach': int(len(seasonal_reach)),
                'event_r_n_site_pairs': event_block['n_site_pairs'],
                'event_r_n_reach': event_block['n_reach'],
            },
            # Backward-compatible A/tau denominator name.
            n_reach=int(len(rc)),
            n_site_seasonal_monthly=int(len(seasonal_sites)),
            n_reach_seasonal_monthly=int(len(seasonal_reach)),
            n_site_event_independent=event_block['n_site_pairs'],
            n_reach_event_independent=event_block['n_reach'],
            amplified_or_lead_frac=boot_ci(amp_lead, np.mean),
            lead_frac=boot_ci(lead, np.mean),
            seasonal_r_median=boot_ci(seasonal, np.median),
            event_r_bootstrap_seed=event_block['bootstrap_seed'],
            event_r_median=event_block['event_r_median'],
            event_r_frac_ge0p3=event_block['event_r_frac_ge0p3'],
        )
    out['bootstrap_ci_by_reach'] = ci

    # ---------- (1b) three correlations on identical daily observations ----------
    print("1b) complete-daily same-date correlation comparison ...", flush=True)
    same_sample = {
        'sample_definition': {
            'source_table': '08_event_metrics_all_sites.parquet',
            'monthly_harmonic_eligibility_required': False,
            'site_filter': (
                'complete independently daily-eligible site_no x reach_id rows '
                'with finite raw_r_same_sample, '
                'seasonal_harmonic_r_same_sample, and event_r; no monthly '
                'harmonic eligibility filter is imposed'
            ),
            'observation_dates': (
                'all three correlations use the identical pairwise-finite daily '
                'river-well observations selected in step 2'
            ),
            'seasonal_harmonic_definition': (
                'Pearson correlation between fitted annual+semiannual '
                'reconstructions; intercept and linear trend excluded'
            ),
            'reach_aggregation': (
                'median across the fixed site-pair intersection within each Reach'
            ),
            'bootstrap_unit': 'Reach; synchronized paired resampling across metrics',
            'bootstrap_replicates': SAME_SAMPLE_BOOTSTRAP_N,
            # Retained for schema compatibility; ALL is the base population.
            'bootstrap_seed_base': SAME_SAMPLE_BOOTSTRAP_SEEDS['ALL'],
            'bootstrap_seed_by_split': SAME_SAMPLE_BOOTSTRAP_SEEDS,
            'interval_format': '[2.5th percentile, point estimate, 97.5th percentile]',
            'paired_difference_definition': (
                'median_paired_difference is median_i(left_i - right_i); '
                'difference_of_medians is median_i(left_i) - median_i(right_i). '
                'They are not algebraically interchangeable and neither is an '
                'additive fraction of correlation explained by seasonality.'
            ),
        }
    }
    for label, subset in (
        ('ALL', EVENT),
        ('DEV', EVENT[EVENT.split == 'dev']),
        ('HOLDOUT', EVENT[EVENT.split == 'holdout']),
    ):
        block = same_sample_block(
            subset, seed=SAME_SAMPLE_BOOTSTRAP_SEEDS[label])

        # The current daily contract has finite values for all three metrics at
        # every event-r-eligible site.  Enforce that fact so Fig. 1/S1d cannot
        # silently become a narrower subset while continuing to be described
        # as the complete daily population.
        expected = ci[label]
        expected_sites = expected['n_site_event_independent']
        expected_reaches = expected['n_reach_event_independent']
        if (block['n_site_pairs'], block['n_reach']) != (
                expected_sites, expected_reaches):
            raise ValueError(
                f'{label} complete-daily three-metric population contracted: '
                f"{block['n_site_pairs']} sites/{block['n_reach']} Reaches != "
                f'{expected_sites} sites/{expected_reaches} Reaches in the '
                'standalone event-r population'
            )

        # Identical population, Reach ordering, and seeds must give byte-for-
        # byte identical full-precision event-r intervals in both robustness blocks.
        if block['metric_medians']['event_r_median'] != expected['event_r_median']:
            raise AssertionError(
                f'{label} event-r median CI differs between the full-daily '
                'three-metric and standalone summaries'
            )
        if block['event_r_frac_ge0p3'] != expected['event_r_frac_ge0p3']:
            raise AssertionError(
                f'{label} event-r fraction CI differs between the full-daily '
                'three-metric and standalone summaries'
            )
        same_sample[label] = block
    require_reach_sum = (
        same_sample['DEV']['n_reach'] + same_sample['HOLDOUT']['n_reach']
    )
    if require_reach_sum != same_sample['ALL']['n_reach']:
        raise AssertionError(
            'same-sample DEV and HOLDOUT Reach counts do not sum to ALL: '
            f"{require_reach_sum} != {same_sample['ALL']['n_reach']}"
        )
    require_pair_sum = (
        same_sample['DEV']['n_site_pairs']
        + same_sample['HOLDOUT']['n_site_pairs']
    )
    if require_pair_sum != same_sample['ALL']['n_site_pairs']:
        raise AssertionError(
            'same-sample DEV and HOLDOUT site-pair counts do not sum to ALL: '
            f"{require_pair_sum} != {same_sample['ALL']['n_site_pairs']}"
        )
    out['same_sample_correlations'] = same_sample

    # ---------- (2) threshold-robustness grid ----------
    print("2) threshold grid ...", flush=True)
    grid = {}
    tau_conds = {'tau<-15': lambda r: r.tau < -15,
                 'tau<=0': lambda r: r.tau <= 0,
                 'tau<+15': lambda r: r.tau < 15}
    for label, sub in [('ALL', S), ('HOLDOUT', S[S.split == 'holdout'])]:
        rc = reach_table(sub)
        g = {}
        for aT in (1.1, 1.2, 1.5):
            for tname, tcond in tau_conds.items():
                frac = float(((rc.A > aT) | tcond(rc)).mean())
                g[f'A>{aT} | {tname}'] = float(frac)
        grid[label] = g
    out['threshold_grid'] = grid

    # ---------- (3) distance robustness (strict threshold-specific rematch) ----------
    print("3) distance robustness (threshold, then independently rematch Top-1) ...", flush=True)
    rob_mon, rematch_provenance = load_or_build_strict_monthly(
        ROBUSTNESS_CANDIDATES,
        ROBUSTNESS_CACHE,
        ROBUSTNESS_SERIES_CANDIDATES,
    )
    phase1_anchor_validation = None
    if int(MAX_DISTANCE_M) in THRESHOLDS_M:
        phase1_anchor_validation = validate_phase1_anchor(
            rob_mon, MON, int(MAX_DISTANCE_M))
        print(
            f"   Phase-1 {MAX_DISTANCE_M:g}-m anchor reproduced exactly: "
            f"{phase1_anchor_validation['n_monthly_rows']:,} monthly rows",
            flush=True,
        )
    for c in ['river_wse', 'Value_EGM2008_m', 'threshold_m']:
        rob_mon[c] = pd.to_numeric(rob_mon[c], errors='coerce')
    rob_mon['date'] = pd.to_datetime(rob_mon['date'], errors='coerce')
    if 'country' not in rob_mon:
        rob_mon['country'] = rob_mon.site_no.map(country)
    rob_base = rob_mon.dropna(subset=['river_wse', 'Value_EGM2008_m', 'date']).copy()

    def metrics_at(thr_km):
        threshold_m = int(round(thr_km * 1000))
        d = rob_base[rob_base['threshold_m'].eq(threshold_m)]
        rows = []
        for sid, g in d.groupby('site_no'):
            if g.shape[0] < HARMONIC_MIN_POINTS:
                continue
            t = yfrac(g.date.values)
            hr = harm(t, g.river_wse.values); hg = harm(t, g.Value_EGM2008_m.values)
            if hr is None or hg is None or hr['amp'] <= 0:
                continue
            A = hg['amp'] / hr['amp']
            dph = (hg['phase'] - hr['phase'] + np.pi) % (2 * np.pi) - np.pi
            rows.append(dict(reach_id=g.reach_id.iloc[0], country=g.country.iloc[0],
                             A=A, tau=dph / (2 * np.pi) * 365.25))
        df = pd.DataFrame(rows)
        if df.empty:
            return dict(n_reach_all=0, n_reach_holdout=0,
                        amplified_or_lead_all=None,
                        amplified_or_lead_holdout=None)
        df['is_dev'] = df.country.isin(DEV)
        reach_dev = df.groupby('reach_id')['is_dev'].mean() >= 0.5
        df['split'] = np.where(df.reach_id.map(reach_dev), 'dev', 'holdout')
        rc = aggregate_reach_at(df, A_col='A', tau_col='tau')
        rc = rc.rename(columns={'tau_days': 'tau'})
        rc = rc[np.isfinite(rc.A) | np.isfinite(rc.tau)]
        h = rc[rc.split == 'holdout']
        return dict(n_reach_all=int(len(rc)), n_reach_holdout=int(len(h)),
                    amplified_or_lead_all=float(((rc.A > 1.2) | (rc.tau <= 0)).mean()),
                    amplified_or_lead_holdout=float(((h.A > 1.2) | (h.tau <= 0)).mean()))
    out['distance_robustness'] = {
        f'<= {threshold_m / 1000:g} km': metrics_at(threshold_m / 1000)
        for threshold_m in THRESHOLDS_M
    }
    out['distance_robustness_definition'] = dict(
        distance_column=DISTANCE_COLUMN,
        source_file=str(ROBUSTNESS_CANDIDATES),
        cache_file=str(ROBUSTNESS_CACHE),
        source_maximum_km=5,
        nested_cumulative_screens_km=[v / 1000 for v in THRESHOLDS_M],
        strict_threshold_specific_rematch=True,
        phase1_anchor_validation=phase1_anchor_validation,
        operation_order=[
            'apply threshold to unranked 5-km candidate universe',
            'count unique observation dates per site/Reach/Node',
            'select best Node within Reach',
            'select Top-1 Reach within well',
            'rebuild the selected pair from the nearest enclosing Phase-1 candidate table',
            'fill early missing Reach/Node identifiers from the threshold-specific Top-1',
            'drop exact duplicate rows',
            'aggregate overlapping granules to daily means',
            'aggregate daily values to monthly means',
            'fit annual+semiannual harmonic A and tau',
        ],
        cache_provenance=rematch_provenance,
    )
    del rob_mon

    # ---------- (4) deseasonalization-method robustness ----------
    print("4) deseasonalization-method robustness ...", flush=True)
    mon = pd.read_csv(MON, sep='\t', low_memory=False)
    for c in ['river_wse', 'Value_EGM2008_m', DISTANCE_COLUMN, 'number_match_SG']:
        if c in mon:
            mon[c] = pd.to_numeric(mon[c], errors='coerce')
    mon['date'] = pd.to_datetime(mon['date'], errors='coerce')
    mon['country'] = mon.site_no.map(country)
    base = mon.dropna(subset=['river_wse', 'Value_EGM2008_m', 'date']).copy()
    if 'number_match_SG' in base:
        base = base[(base.number_match_SG == 1) | (base.number_match_SG.isna())]
    base['month'] = base['date'].dt.month
    rows_clim, rows_harm, rows_dtcl = [], [], []
    for sid, g in base[base[DISTANCE_COLUMN] <= MAX_DISTANCE_M].groupby('site_no'):
        if g.shape[0] < HARMONIC_MIN_POINTS:
            continue
        g = g.sort_values('date')
        rw = g.river_wse.values; gw = g.Value_EGM2008_m.values
        t = yfrac(g.date.values)
        if rw.std() == 0 or gw.std() == 0:
            continue
        river_fit = harm(t, rw)
        gw_fit = harm(t, gw)
        if river_fit is None or gw_fit is None:
            continue
        sr = np.corrcoef(rw, gw)[0, 1]
        # (a) monthly climatology removal
        rA = rw - g.groupby('month').river_wse.transform('mean').values
        gA = gw - g.groupby('month').Value_EGM2008_m.transform('mean').values
        if rA.std() > 0 and gA.std() > 0:
            rows_clim.append((sid, g.reach_id.iloc[0], sr, np.corrcoef(rA, gA)[0, 1]))
        # (b) annual+semiannual harmonic residual
        rr, gr = river_fit['residual'], gw_fit['residual']
        if rr.std() > 0 and gr.std() > 0:
            rows_harm.append((sid, g.reach_id.iloc[0], sr, np.corrcoef(rr, gr)[0, 1]))
        # (c) linear detrend + monthly climatology
        rd = rA - np.polyval(np.polyfit(t, rA, 1), t)
        gd = gA - np.polyval(np.polyfit(t, gA, 1), t)
        if rd.std() > 0 and gd.std() > 0:
            rows_dtcl.append((sid, g.reach_id.iloc[0], sr, np.corrcoef(rd, gd)[0, 1]))

    method_rows = {
        'monthly_climatology_removal': rows_clim,
        'annual+semiannual_harmonic_residual': rows_harm,
        'detrend+monthly_climatology': rows_dtcl,
    }
    method_frames = {
        name: pd.DataFrame(
            rows,
            columns=['site_no', 'reach_id', 'seasonal_r', 'deseason_r'],
        )
        for name, rows in method_rows.items()
    }

    # Compare methods on one fixed site x Reach population.  Filtering before
    # Reach aggregation prevents a method with a zero-variance residual from
    # silently changing both the wells represented within a Reach and the
    # Reach denominator of its point estimate.
    common_keys = None
    for df in method_frames.values():
        keys = pd.MultiIndex.from_frame(df[['site_no', 'reach_id']]).unique()
        common_keys = keys if common_keys is None else common_keys.intersection(keys)
    if common_keys is None or len(common_keys) == 0:
        raise ValueError('no common site x Reach pairs across deseasonalization methods')
    for name, df in method_frames.items():
        keys = pd.MultiIndex.from_frame(df[['site_no', 'reach_id']])
        method_frames[name] = df.loc[keys.isin(common_keys)].copy()

    common_reach_counts = {
        name: int(df.reach_id.nunique()) for name, df in method_frames.items()
    }
    if len(set(common_reach_counts.values())) != 1:
        raise AssertionError(
            f'deseasonalization common-intersection Reach counts differ: {common_reach_counts}'
        )
    print(
        '   fixed intersection: '
        f'{len(common_keys)} site x Reach pairs, '
        f'{next(iter(common_reach_counts.values()))} reaches',
        flush=True,
    )

    def collapse(df, name):
        rc = df.groupby('reach_id').agg(seasonal_r=('seasonal_r', 'median'),
                                        deseason_r=('deseason_r', 'median'))
        return dict(method=name, n_reach=int(len(rc)),
                    seasonal_r_median=float(rc.seasonal_r.median()),
                    deseason_r_median=float(rc.deseason_r.median()))
    out['deseason_methods'] = [
        collapse(method_frames[name], name) for name in method_rows
    ]
    out['deseason_methods_sample_definition'] = dict(
        unit='site_no x reach_id pair, then median aggregation by Reach',
        rule='fixed intersection of pairs with finite residual correlation under all three methods',
        n_pair=int(len(common_keys)),
        n_reach=int(next(iter(common_reach_counts.values()))),
    )

    # ---------- (5) synthetic harmonic sign test ----------
    print("5) synthetic harmonic sign test ...", flush=True)
    tt = np.arange(0, 3, 1 / 24)  # 3 yr, ~biweekly
    river = np.cos(2 * np.pi * (tt - 0.0))
    # well lags river by 30 days, attenuated 0.5  -> expect A~0.5, tau~+30
    lag = 30 / 365.25
    well_lag = 0.5 * np.cos(2 * np.pi * (tt - lag))
    hr = harm(tt, river); hl = harm(tt, well_lag)
    A_lag = hl['amp'] / hr['amp']
    tau_lag = (((hl['phase'] - hr['phase'] + np.pi) % (2 * np.pi)) - np.pi) / (2 * np.pi) * 365.25
    # well leads river by 20 days -> expect tau~-20
    lead = -20 / 365.25
    well_lead = 0.8 * np.cos(2 * np.pi * (tt - lead))
    hd = harm(tt, well_lead)
    tau_lead = (((hd['phase'] - hr['phase'] + np.pi) % (2 * np.pi)) - np.pi) / (2 * np.pi) * 365.25
    passed = bool(0.45 < A_lag < 0.55 and 25 < tau_lag < 35 and -25 < tau_lead < -15)
    out['synthetic_harmonic_test'] = dict(
        A_lag30=float(A_lag), tau_lag30_days=float(tau_lag),
        tau_lead20_days=float(tau_lead), passed=passed)

    json.dump(out, open(OUT / "08_robustness.json", 'w'), indent=2)
    print("\n=== 08_robustness.json ===")
    print(json.dumps(out, indent=2))


if __name__ == '__main__':
    main()
