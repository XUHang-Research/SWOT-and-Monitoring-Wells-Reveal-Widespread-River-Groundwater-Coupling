#!/usr/bin/env python3
"""Step 6 — single reproducibility export of every headline number.

Loads only the finalized tables produced by steps 1–5 and writes
08_main_numbers.yml with each headline number, its denominator definition, and
(where applicable) a reach-level bootstrap 95% CI. It performs only transparent
final-table summaries (including explicitly named Spearman associations) so
every manuscript number can be traced to one machine-readable source.

Inputs : output/08_summary.json, output/08_robustness.json,
         output/08_mechanism_summary.json,
         output/08_global_coupling_metrics.parquet,
         output/08_event_covariates_all_sites.parquet
Outputs: output/08_main_numbers.yml

Requires: numpy, pandas, pyarrow, pyyaml.
"""
import json

import numpy as np
import pandas as pd
import yaml
from scipy.stats import spearmanr

from importlib import import_module as _import_module
_harmonic_model = _import_module("10_harmonic_model")
harmonic_criterion = _harmonic_model.harmonic_criterion
_pipeline_config = _import_module("09_pipeline_config")
DISTANCE_COLUMN = _pipeline_config.DISTANCE_COLUMN
MAX_DISTANCE_KM = _pipeline_config.MAX_DISTANCE_KM
OUT = _pipeline_config.OUT
summ = json.load(open(OUT / "08_summary.json"))
rob = json.load(open(OUT / "08_robustness.json"))
mech = json.load(open(OUT / "08_mechanism_summary.json"))
S = pd.read_parquet(OUT / "08_global_coupling_metrics.parquet")
EVENT_COV = pd.read_parquet(OUT / "08_event_covariates_all_sites.parquet")


def reach_spearman_with_aridity(table, metric):
    """Spearman association after fixing the metric/aridity site intersection."""
    pair = table[['site_no', 'reach_id', metric, 'aridity']].dropna().copy()
    reach = (pair.groupby('reach_id', as_index=False)
             .agg(metric=(metric, 'median'), aridity=('aridity', 'median')))
    if len(reach) < 3:
        rho = p_value = np.nan
    else:
        result = spearmanr(reach.metric, reach.aridity)
        rho, p_value = float(result.statistic), float(result.pvalue)
    return {
        'rho': rho,
        'p_value': p_value,
        'n_reach': int(len(reach)),
        'n_site_pairs': int(len(pair)),
    }


climate_associations = {
    'aridity_definition': summ['aridity_definition'],
    'analysis_unit': 'Reach',
    'aggregation': {
        'A_vs_aridity': (
            'monthly-harmonic-eligible 08_global_coupling_metrics population; '
            'finite A and complete-window aridity, median-aggregated by Reach'),
        'event_r_vs_aridity': (
            'complete independent daily event-r population in '
            '08_event_covariates_all_sites; complete-window aridity, '
            'median-aggregated by Reach'),
    },
    'A_vs_aridity': reach_spearman_with_aridity(S, 'A'),
    'event_r_vs_aridity': reach_spearman_with_aridity(EVENT_COV, 'event_r'),
}

# Reach-level co-variability versus water-table depth. Retain finite zero and
# negative values because they can represent land-surface crossings or artesian
# conditions rather than invalid observations.  Use the standalone daily
# population; monthly-harmonic eligibility is not imposed.
_rc = (EVENT_COV.dropna(subset=['event_r', 'depth_m'])
       .groupby('reach_id').agg(event_r=('event_r', 'median'), depth=('depth_m', 'median'))
       .reset_index().dropna(subset=['depth']))
depth_rho, _ = spearmanr(_rc.depth, _rc.event_r)

ci = rob['bootstrap_ci_by_reach']
same_sample = rob['same_sample_correlations']
event_null = rob['event_permutation_null']
residual_variability = summ.get('wse_residual_variability')
if residual_variability is None:
    # Backward-compatible reading of products created before the terminology
    # correction. The statistic itself is unchanged.
    residual_variability = summ['sigma_swot_empirical']


def interval_record(interval):
    """Expand a robustness [lower, point, upper] triple unambiguously."""
    return {
        'value': interval[1],
        'bootstrap_ci95': [interval[0], interval[2]],
    }


def independent_event_record(label, value_key, ci_key):
    """Export a standalone daily event statistic with legacy triple CI shape."""
    block = event_null[label]
    value = block[value_key]
    endpoints = block[ci_key]
    return {
        'value': value,
        'ci95': endpoints,
        'n_site_pairs': block['n_site'],
        'n_reach': block['n_reach'],
    }


def same_sample_export(label):
    """Copy one split's complete-daily same-date comparison into the headline file."""
    block = same_sample[label]
    metrics = block['metric_medians']
    comparisons = block['paired_comparisons']
    return {
        'n_site_pairs': block['n_site_pairs'],
        'n_reach': block['n_reach'],
        'bootstrap_seed': block['bootstrap_seed'],
        'metric_medians': {
            'raw_r_same_sample': interval_record(
                metrics['raw_r_same_sample_median']),
            'seasonal_harmonic_r_same_sample': interval_record(
                metrics['seasonal_harmonic_r_same_sample_median']),
            'event_r': interval_record(metrics['event_r_median']),
        },
        'event_r_fraction_ge_0p3': interval_record(
            block['event_r_frac_ge0p3']),
        'paired_comparisons': {
            name: {
                'left_metric': comparison['left_metric'],
                'right_metric': comparison['right_metric'],
                'median_paired_difference': interval_record(
                    comparison['median_paired_difference']),
                'difference_of_medians': interval_record(
                    comparison['difference_of_medians']),
                'fraction_reach_paired_difference_positive': interval_record(
                    comparison['fraction_reach_paired_difference_positive']),
            }
            for name, comparison in comparisons.items()
        },
    }

doc = {
    'provenance': {
        'scope': summ['scope'],
        'dev_countries': summ['dev_countries'],
        'denominator_note': (
            'all headline fractions are computed at Reach level. A/tau uses '
            'median A and equal-weight circular-mean tau across monthly-eligible '
            f'sites within {MAX_DISTANCE_KM:g} km; standalone event-r uses the '
            'complete independently daily-eligible population, as does the '
            'same-date raw/seasonal-harmonic/residual comparison. Only direct '
            'monthly-versus-daily paired tests use a frozen monthly-daily '
            'intersection. CIs are Reach-level bootstraps (n=10000).'),
        'distance_column': DISTANCE_COLUMN,
        'reach_phase_aggregation': 'equal-weight circular mean of site phase differences',
        'harmonic_eligibility': harmonic_criterion(),
        'generated_by': '06_main_numbers.py',
    },
    'dataset': {
        'n_pairs_with_metrics': int(S.site_no.nunique()),
        'n_reaches_with_metrics': int(S.reach_id.nunique()),
        'n_independent_daily_event_sites': int(
            event_null['sample_definition']['n_site']),
        'n_independent_daily_event_reaches': int(
            event_null['sample_definition']['n_reach_all']),
        'median_match_distance_km': float(S.dist_km.median()),
        'n_reaches_wse_residual_variability': residual_variability['n_reach'],
        'wse_residual_variability_p5_p95_fold': residual_variability['p5_p95_fold'],
        'analysis_window': '2023–2025 (SWOT science orbit, ~3 yr)',
    },
    'diffusive_falsification': {
        'thresholds': {'amplified_A_gt': 1.2, 'lead_tau_le_days': 0},
        'denominator': f'reach-aggregated pairs within {MAX_DISTANCE_KM:g} km with finite A or tau',
        'amplified_or_lead_frac': {
            'ALL': {'value': float(summ['ALL']['amplified_or_lead_frac']),
                    'ci95': ci['ALL']['amplified_or_lead_frac'], 'n_reach': ci['ALL']['n_reach']},
            'DEV': {'value': float(summ['DEV']['amplified_or_lead_frac']),
                    'ci95': ci['DEV']['amplified_or_lead_frac'], 'n_reach': ci['DEV']['n_reach']},
            'HOLDOUT': {'value': float(summ['HOLDOUT']['amplified_or_lead_frac']),
                        'ci95': ci['HOLDOUT']['amplified_or_lead_frac'], 'n_reach': ci['HOLDOUT']['n_reach']},
        },
        'lead_frac_tau_le_0': {
            'ALL': {'value': ci['ALL']['lead_frac'][1], 'ci95': ci['ALL']['lead_frac']},
            'HOLDOUT': {'value': ci['HOLDOUT']['lead_frac'][1], 'ci95': ci['HOLDOUT']['lead_frac']},
        },
        'threshold_robustness': rob['threshold_grid'],
        'distance_robustness': rob['distance_robustness'],
        'synthetic_sign_test': rob['synthetic_harmonic_test'],
        'single_reach_at_uncertainty': rob['single_reach_at_uncertainty'],
    },
    'correlation': {
        'denominator': (
            f'within {MAX_DISTANCE_KM:g} km; standalone event-r uses every '
            'independently daily-eligible site and Reach. The three same-date '
            'daily correlations use that same complete daily population; only '
            'direct monthly-versus-daily comparisons use an explicitly frozen '
            'intersection'),
        'seasonal_r_median': {
            'denominator': (
                'all monthly-harmonic-eligible site-Reach pairs with finite '
                'monthly seasonal_r, median-aggregated by Reach; daily event-r '
                'eligibility is not imposed'),
            'ALL': {'value': ci['ALL']['seasonal_r_median'][1],
                    'ci95': ci['ALL']['seasonal_r_median'],
                    'n_site_pairs': ci['ALL']['n_site_seasonal_monthly'],
                    'n_reach': ci['ALL']['n_reach_seasonal_monthly']},
            'HOLDOUT': {'value': ci['HOLDOUT']['seasonal_r_median'][1],
                        'ci95': ci['HOLDOUT']['seasonal_r_median'],
                        'n_site_pairs': ci['HOLDOUT']['n_site_seasonal_monthly'],
                        'n_reach': ci['HOLDOUT']['n_reach_seasonal_monthly']},
        },
        'event_r_median': {
            'denominator': (
                'complete independent daily event-r population; median across '
                'eligible sites within Reach'),
            'ALL': independent_event_record(
                'ALL', 'obs_event_r_median', 'obs_event_r_median_ci95'),
            'DEV': independent_event_record(
                'DEV', 'obs_event_r_median', 'obs_event_r_median_ci95'),
            'HOLDOUT': independent_event_record(
                'HOLDOUT', 'obs_event_r_median',
                'obs_event_r_median_ci95'),
        },
        'event_r_frac_ge_0p3': {
            'denominator': (
                'complete independent daily event-r population; threshold '
                'applied after site values are median-aggregated within Reach'),
            'ALL': independent_event_record(
                'ALL', 'obs_frac_ge0p3', 'obs_frac_ge0p3_ci95'),
            'DEV': independent_event_record(
                'DEV', 'obs_frac_ge0p3', 'obs_frac_ge0p3_ci95'),
            'HOLDOUT': independent_event_record(
                'HOLDOUT', 'obs_frac_ge0p3', 'obs_frac_ge0p3_ci95'),
        },
        'same_sample_daily_comparison': {
            'denominator': same_sample['sample_definition'],
            'interpretation': (
                'The three correlations use identical daily observations from '
                'the complete independently daily-eligible population; monthly '
                'harmonic eligibility is not imposed. '
                'Correlation is not additively decomposable: the paired '
                'differences quantify contrasts between metrics, not a '
                'fraction explained by seasonality.'
            ),
            'ALL': same_sample_export('ALL'),
            'DEV': same_sample_export('DEV'),
            'HOLDOUT': same_sample_export('HOLDOUT'),
        },
        'deseasonalization_method_robustness': rob['deseason_methods'],
        'event_permutation_null': event_null,
        'climate_associations': climate_associations,
    },
    'regional_exploratory': {
        'sample_definitions': mech['sample_definitions'],
        'kruskal_wallis_p': mech['heterogeneity']['p_value'],
        'n_regions': mech['heterogeneity']['n_regions'],
        'n_reaches': mech['heterogeneity']['n_reaches'],
        'regional_effect_size': mech['regional_effect_size'],
        'within_region_sign_tests': mech['within_region_sign_tests'],
        'aridity_between_region_spearman': mech['region_level_spearman_vs_eventr']['aridity'][0],
        'aridity_within_reach_spearman': mech['reach_level_spearman_vs_eventr']['aridity'][0],
        'leave_one_region_out_obs_vs_pred_spearman': mech['leave_one_region_out']['obs_vs_pred_spearman'][0],
        'covariability_depth_sample': (
            'Independent daily event-r sites with finite depth filtered before '
            'Reach aggregation; response and depth use the same wells'),
        'covariability_depth_spearman_rho': float(depth_rho),
        'covariability_depth_spearman_n': int(len(_rc)),
        'covariability_depth_n_site': int(EVENT_COV[['event_r','depth_m']].notna().all(axis=1).sum()),
        'covariability_depth_missing_site_ids': sorted(EVENT_COV.loc[EVENT_COV.depth_m.isna(),'site_no'].astype(str)),
        'australia_event_r_full_daily': mech['australia_event_r_full_daily'],
        'australia_mechanism_full_daily': mech['australia'],
        'caveat': mech['caveats'],
    },
}

with open(OUT / "08_main_numbers.yml", 'w') as f:
    yaml.safe_dump(doc, f, sort_keys=False, allow_unicode=True, default_flow_style=False)
print("wrote output/08_main_numbers.yml")
print(open(OUT / "08_main_numbers.yml").read())
