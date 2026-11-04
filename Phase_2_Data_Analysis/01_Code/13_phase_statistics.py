#!/usr/bin/env python3
"""Circular phase aggregation and Reach-specific A/tau uncertainty.

Phase lag is directional circular data on a one-year period.  It must not be
aggregated with ordinary linear medians near the -pi/+pi branch cut.  This
module provides one implementation used by the analysis and figure scripts.

Reach uncertainty is estimated with a leverage-adjusted paired residual
calendar-block wild bootstrap:
annual+semiannual harmonic models are fitted at fixed observation dates; the
all river and groundwater site residuals in each Reach receive the same Rademacher multiplier within
each three-calendar-month block, with a randomized block origin. Missing
months remain missing, dates remain fixed, and endpoints are not wrapped.
Both harmonics are refitted; and sites are resampled within each
Reach.  Circular intervals are unwrapped locally around the Reach point phase.
"""
from __future__ import annotations

import hashlib
from pathlib import Path

import numpy as np
import pandas as pd

from importlib import import_module as _import_module
_harmonic_model = _import_module("10_harmonic_model")
prepare_harmonic_data = _harmonic_model.prepare_harmonic_data
_calendar_bootstrap = _import_module("12_calendar_bootstrap")
calendar_block_weights = _calendar_bootstrap.calendar_block_weights
iter_reach_calendars = _calendar_bootstrap.iter_reach_calendars

PERIOD_DAYS = 365.25
DEFAULT_BOOTSTRAPS = 10_000
DEFAULT_BLOCK_MONTHS = 3
DEFAULT_SEED = 20250311


def wrap_radians(x):
    """Wrap radians to [-pi, pi)."""
    x = np.asarray(x, dtype=float)
    return (x + np.pi) % (2 * np.pi) - np.pi


def days_to_radians(days):
    return np.asarray(days, dtype=float) * (2 * np.pi / PERIOD_DAYS)


def radians_to_days(angle):
    return np.asarray(angle, dtype=float) * (PERIOD_DAYS / (2 * np.pi))


def circular_mean_radians(angle):
    """Return equal-weight circular mean and resultant length R."""
    angle = np.asarray(angle, dtype=float)
    angle = angle[np.isfinite(angle)]
    if not len(angle):
        return np.nan, np.nan
    z = np.mean(np.exp(1j * angle))
    return float(wrap_radians(np.angle(z))), float(np.abs(z))


def circular_mean_days(days):
    angle, concentration = circular_mean_radians(days_to_radians(days))
    return float(radians_to_days(angle)), concentration


def aggregate_reach_at(df, A_col="A", tau_col="tau_days", split_col="split"):
    """Aggregate site-level A/tau to Reach median A and circular-mean tau."""
    rows = []
    for reach_id, group in df.groupby("reach_id", sort=False):
        tau_days, concentration = circular_mean_days(
            pd.to_numeric(group[tau_col], errors="coerce").to_numpy()
        )
        A = pd.to_numeric(group[A_col], errors="coerce").median()
        row = {
            "reach_id": reach_id,
            "A": float(A) if np.isfinite(A) else np.nan,
            "tau_days": tau_days,
            "phase_concentration_R": concentration,
            "n_phase_sites": int(pd.to_numeric(group[tau_col], errors="coerce").notna().sum()),
        }
        if split_col in group:
            row["split"] = group[split_col].iloc[0]
        rows.append(row)
    return pd.DataFrame(rows)


def _stable_seed(value, base_seed=DEFAULT_SEED):
    digest = hashlib.blake2b(str(value).encode("utf-8"), digest_size=8).digest()
    return (int.from_bytes(digest, "little") ^ int(base_seed)) % (2**32)


def _yfrac(date):
    date = pd.DatetimeIndex(date)
    return date.year.to_numpy() + (date.dayofyear.to_numpy() - 1) / 365.25


def _site_residual_block_bootstrap(group, n_boot, block_months, seed, calendar_plan=None):
    """Paired calendar-block wild bootstrap with HC2 residual adjustment."""
    group = group.sort_values("date")
    t = _yfrac(group.date)
    river = pd.to_numeric(group.river_wse, errors="coerce").to_numpy(float)
    gw = pd.to_numeric(group.Value_EGM2008_m, errors="coerce").to_numpy(float)
    valid = np.isfinite(t) & np.isfinite(river) & np.isfinite(gw)
    t, river, gw = t[valid], river[valid], gw[valid]
    prepared = prepare_harmonic_data(t, river)
    if prepared is None:
        return None
    t, river, design, diagnostics = prepared
    pinv = np.linalg.pinv(design)
    beta_r = pinv @ river
    beta_g = pinv @ gw
    fit_r = design @ beta_r
    fit_g = design @ beta_g
    # HC2 adjustment compensates for fitted residual shrinkage. Both residual
    # series receive exactly the same multiplier at each original date.
    leverage = np.einsum("ij,ji->i", design, pinv)
    leverage_scale = np.sqrt(np.maximum(1.0 - leverage, np.finfo(float).eps))
    resid_r = (river - fit_r) / leverage_scale
    resid_g = (gw - fit_g) / leverage_scale
    resid_r -= resid_r.mean()
    resid_g -= resid_g.mean()

    dates = group.date.to_numpy()[valid]
    if calendar_plan is None:
        weights = calendar_block_weights(dates, n_boot, block_months, seed)
    else:
        if calendar_plan.n_boot != n_boot or calendar_plan.block_months != block_months:
            raise ValueError('shared calendar configuration differs from site fit')
        weights = calendar_plan.for_dates(dates)
    river_boot = fit_r[None, :] + weights * resid_r[None, :]
    gw_boot = fit_g[None, :] + weights * resid_g[None, :]
    beta_r_boot = river_boot @ pinv.T
    beta_g_boot = gw_boot @ pinv.T
    amp_r = np.hypot(beta_r_boot[:, 2], beta_r_boot[:, 3])
    amp_g = np.hypot(beta_g_boot[:, 2], beta_g_boot[:, 3])
    A_boot = np.divide(
        amp_g, amp_r,
        out=np.full(n_boot, np.nan),
        where=amp_r > np.finfo(float).eps,
    )
    phase_r = np.arctan2(beta_r_boot[:, 3], beta_r_boot[:, 2])
    phase_g = np.arctan2(beta_g_boot[:, 3], beta_g_boot[:, 2])
    phase_boot = wrap_radians(phase_g - phase_r)
    return {
        "A": A_boot,
        "phase": phase_boot,
        "n_mon": int(len(t)),
        "ampR": float(np.hypot(beta_r[2], beta_r[3])),
        "ampG": float(np.hypot(beta_g[2], beta_g[3])),
    }


def build_reach_at_uncertainty(
    monthly_file,
    pair_table,
    distance_column,
    maximum_distance_m,
    n_boot=DEFAULT_BOOTSTRAPS,
    block_months=DEFAULT_BLOCK_MONTHS,
    seed=DEFAULT_SEED,
):
    """Build one row per Reach with circular point estimates and bootstrap CIs."""
    monthly_file = Path(monthly_file)
    wanted = {
        "site_no", "reach_id", "date", "river_wse", "Value_EGM2008_m",
        distance_column, "number_match_SG",
    }
    monthly = pd.read_csv(
        monthly_file,
        sep="\t",
        usecols=lambda column: column in wanted,
        low_memory=False,
    )
    for column in ("river_wse", "Value_EGM2008_m", distance_column, "number_match_SG"):
        if column in monthly:
            monthly[column] = pd.to_numeric(monthly[column], errors="coerce")
    monthly["date"] = pd.to_datetime(monthly["date"], errors="coerce")
    monthly = monthly.dropna(subset=["river_wse", "Value_EGM2008_m", "date"])
    monthly = monthly[monthly[distance_column] <= maximum_distance_m]
    if "number_match_SG" in monthly:
        monthly = monthly[
            monthly.number_match_SG.eq(1) | monthly.number_match_SG.isna()
        ]

    eligible_sites = set(pair_table.site_no.astype(str))
    monthly = monthly[monthly.site_no.astype(str).isin(eligible_sites)]
    site_boot = {}
    for _, reach_monthly, plan in iter_reach_calendars(
            monthly, pair_table, n_boot, block_months, seed):
        for site_no, group in reach_monthly.groupby("site_no", sort=False):
            result = _site_residual_block_bootstrap(
                group, n_boot=int(n_boot), block_months=int(block_months),
                seed=seed, calendar_plan=plan)
            if result is not None:
                site_boot[str(site_no)] = result

    rows = []
    for reach_id, group in pair_table.groupby("reach_id", sort=False):
        group = group[group.site_no.astype(str).isin(site_boot)].copy()
        if group.empty:
            continue
        sites = group.site_no.astype(str).tolist()
        A_hat = float(pd.to_numeric(group.A, errors="coerce").median())
        phase_hat, concentration = circular_mean_radians(
            days_to_radians(pd.to_numeric(group.tau_days, errors="coerce"))
        )
        tau_hat = float(radians_to_days(phase_hat))

        rng = np.random.default_rng(_stable_seed(reach_id, seed + 1))
        A_reach = np.full(n_boot, np.nan)
        phase_reach = np.full(n_boot, np.nan)
        n_site = len(sites)
        for replicate in range(n_boot):
            chosen = rng.integers(0, n_site, n_site)
            # One synchronized time realization, including repeated sampled sites.
            A_values = np.array([
                site_boot[sites[j]]["A"][replicate] for j in chosen
            ])
            phase_values = np.array([
                site_boot[sites[j]]["phase"][replicate] for j in chosen
            ])
            A_reach[replicate] = np.nanmedian(A_values)
            phase_reach[replicate], _ = circular_mean_radians(phase_values)

        joint_finite = np.isfinite(A_reach) & np.isfinite(phase_reach)
        finite_A = A_reach[joint_finite]
        finite_phase = phase_reach[joint_finite]
        if not len(finite_A) or not len(finite_phase):
            continue
        phase_delta = wrap_radians(finite_phase - phase_hat)
        tau_delta = radians_to_days(phase_delta)
        tau_boot = radians_to_days(finite_phase)
        p_A = float(np.mean(finite_A > 1.2))
        p_tau = float(np.mean(tau_boot <= 0))
        p_violation = float(np.mean(
            (finite_A > 1.2) | (tau_boot <= 0)
        ))
        A_low, A_high = np.percentile(finite_A, [2.5, 97.5])
        tau_low_delta, tau_high_delta = np.percentile(tau_delta, [2.5, 97.5])
        rows.append({
            "reach_id": reach_id,
            "split": group.split.iloc[0],
            "A": A_hat,
            "A_ci_low": float(A_low),
            "A_ci_high": float(A_high),
            "tau_days": tau_hat,
            "tau_ci_low_days": float(tau_hat + tau_low_delta),
            "tau_ci_high_days": float(tau_hat + tau_high_delta),
            "tau_ci_width_days": float(tau_high_delta - tau_low_delta),
            "phase_concentration_R": concentration,
            "p_A_gt_1p2": p_A,
            "p_tau_le_0": p_tau,
            "p_amplified_or_lead": p_violation,
            "n_sites": int(n_site),
            "n_mon_median": float(np.median([site_boot[s]["n_mon"] for s in sites])),
            "ampR_median_m": float(np.median([site_boot[s]["ampR"] for s in sites])),
            "ampG_median_m": float(np.median([site_boot[s]["ampG"] for s in sites])),
            "bootstrap_replicates": int(n_boot),
            "residual_block_months": int(block_months),
            "residual_bootstrap_method": "paired calendar-block wild bootstrap",
            "residual_block_unit": "calendar_month",
            "temporal_dependence_unit": "Reach",
            "replicate_alignment": "same calendar weights and replicate across sites and Nodes",
            "phase_aggregation": "equal-weight circular mean",
            "tau_interval": "percentile interval locally unwrapped around point phase",
        })
    return pd.DataFrame(rows)


def summarize_reach_at_uncertainty(table):
    """Machine-readable uncertainty summary by frozen Reach split."""
    def propagated_fraction(probability, seed_value, n_boot=10000):
        probability = np.asarray(probability, dtype=float)
        probability = probability[np.isfinite(probability)]
        rng = np.random.default_rng(seed_value)
        estimates = np.empty(n_boot)
        for replicate in range(n_boot):
            sampled = probability[rng.integers(0, len(probability), len(probability))]
            estimates[replicate] = np.mean(rng.random(len(sampled)) < sampled)
        return [
            float(np.percentile(estimates, 2.5)),
            float(np.mean(probability)),
            float(np.percentile(estimates, 97.5)),
        ]

    output = {
        "method": (
            "HC2-adjusted paired calendar-block wild bootstrap at fixed dates; sites resampled "
            "within Reach; shared calendar weights and synchronized replicates across all "
            "sites/Nodes in each Reach; equal-weight circular-mean phase"
        ),
        "residual_block_months": int(table.residual_block_months.iloc[0]),
        "residual_block_unit": "calendar_month",
        "temporal_dependence_unit": "Reach",
        "replicate_alignment": "same replicate across sites, including repeated sampled sites",
        "multiplier": "Rademacher +/-1, shared by all river/GW site series in each Reach/calendar block",
        "block_origin": "uniform random monthly offset per replicate; no endpoint wrapping",
        "missing_months": "original observation dates and missingness retained; no interpolation",
        "thresholds": {"A": 1.2, "tau_days": 0.0, "classification_probability": 0.95},
    }
    for index, (label, subset) in enumerate((
        ("ALL", table),
        ("DEV", table[table.split.eq("dev")]),
        ("HOLDOUT", table[table.split.eq("holdout")]),
    )):
        p = subset.p_amplified_or_lead
        output[label] = {
            "n_reach": int(len(subset)),
            "point_amplified_or_lead_fraction": float(
                ((subset.A > 1.2) | (subset.tau_days <= 0)).mean()
            ),
            "robust_violation_fraction_p_ge_0p95": float((p >= 0.95).mean()),
            "robust_compatible_fraction_p_le_0p05": float((p <= 0.05).mean()),
            "uncertain_classification_fraction": float(((p > 0.05) & (p < 0.95)).mean()),
            "uncertainty_propagated_amplified_or_lead_fraction": propagated_fraction(
                subset.p_amplified_or_lead, DEFAULT_SEED + 100 + index),
            "uncertainty_propagated_lead_fraction": propagated_fraction(
                subset.p_tau_le_0, DEFAULT_SEED + 200 + index),
            "median_tau_ci_width_days": float(subset.tau_ci_width_days.median()),
            "median_A_ci_fold": float(
                (subset.A_ci_high / subset.A_ci_low.where(subset.A_ci_low > 0)).median()
            ),
            "n_phase_concentration_below_0p5": int((subset.phase_concentration_R < 0.5).sum()),
        }
    return output
