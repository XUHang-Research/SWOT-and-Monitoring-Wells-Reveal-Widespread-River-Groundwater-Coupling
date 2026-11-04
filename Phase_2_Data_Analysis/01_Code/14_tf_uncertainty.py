"""Uncertainty propagation for annual/semiannual transfer-function ordering.

The routines in this module keep every site's accepted monthly sampling dates
fixed, multiply paired HC2 residuals by shared Reach/calendar-block weights, refit all four
river/groundwater harmonic amplitudes, and then resample sites within Reaches.
They also construct an equal-gain null in which the groundwater/river gain is
identical at the annual and semiannual bands while retaining the observed
sampling calendar, fitted phases, baseline/trend, and paired residuals.
"""
from __future__ import annotations

import hashlib

import numpy as np
import pandas as pd

from importlib import import_module as _import_module
_harmonic_model = _import_module("10_harmonic_model")
prepare_harmonic_data = _harmonic_model.prepare_harmonic_data
_calendar_bootstrap = _import_module("12_calendar_bootstrap")
calendar_block_weights = _calendar_bootstrap.calendar_block_weights
iter_reach_calendars = _calendar_bootstrap.iter_reach_calendars


DEFAULT_BOOTSTRAPS = 10_000
DEFAULT_BLOCK_MONTHS = 3
DEFAULT_SEED = 20260812
EPS = np.finfo(float).tiny


def _stable_seed(value, base_seed=DEFAULT_SEED):
    digest = hashlib.blake2b(str(value).encode("utf-8"), digest_size=8).digest()
    return (int.from_bytes(digest, "little") ^ int(base_seed)) % (2**32)


def _year_fraction(date):
    date = pd.DatetimeIndex(date)
    return date.year.to_numpy() + (date.dayofyear.to_numpy() - 1) / 365.25


def _amplitudes(beta):
    """Return annual and semiannual amplitudes from rows of coefficients."""
    return (
        np.hypot(beta[..., 2], beta[..., 3]),
        np.hypot(beta[..., 4], beta[..., 5]),
    )


def _tf_and_log_order(amp_r1, amp_r2, amp_g1, amp_g2):
    amp_r1 = np.maximum(np.asarray(amp_r1, float), EPS)
    amp_r2 = np.maximum(np.asarray(amp_r2, float), EPS)
    amp_g1 = np.maximum(np.asarray(amp_g1, float), EPS)
    amp_g2 = np.maximum(np.asarray(amp_g2, float), EPS)
    tf1 = np.square(amp_g1 / amp_r1)
    tf2 = np.square(amp_g2 / amp_r2)
    # Numerically preferable to subtracting or dividing two potentially large
    # squared ratios. Positive values are exactly equivalent to TF2 < TF1.
    log_order = 2.0 * (
        np.log(amp_g1) - np.log(amp_r1)
        - np.log(amp_g2) + np.log(amp_r2)
    )
    return tf1, tf2, log_order


def bootstrap_site_tf(group, n_boot=DEFAULT_BOOTSTRAPS,
                      block_months=DEFAULT_BLOCK_MONTHS,
                      seed=DEFAULT_SEED, calendar_plan=None):
    """Refit both bands under observed and equal-gain residual bootstraps."""
    group = group.sort_values("date")
    t = _year_fraction(group.date)
    river = pd.to_numeric(group.river_wse, errors="coerce").to_numpy(float)
    gw = pd.to_numeric(group.Value_EGM2008_m, errors="coerce").to_numpy(float)
    valid = np.isfinite(t) & np.isfinite(river) & np.isfinite(gw)
    t, river, gw = t[valid], river[valid], gw[valid]
    prepared = prepare_harmonic_data(t, river)
    if prepared is None:
        return None
    t, river, design, diagnostics = prepared
    if len(gw) != len(river):
        return None

    pinv = np.linalg.pinv(design)
    beta_r = pinv @ river
    beta_g = pinv @ gw
    fit_r = design @ beta_r
    fit_g = design @ beta_g
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
            raise ValueError('shared calendar configuration differs from TF fit')
        weights = calendar_plan.for_dates(dates)
    river_boot = fit_r[None, :] + weights * resid_r[None, :]
    gw_boot = fit_g[None, :] + weights * resid_g[None, :]
    beta_r_boot = river_boot @ pinv.T
    beta_g_boot = gw_boot @ pinv.T
    amp_r1, amp_r2 = _amplitudes(beta_r_boot)
    amp_g1, amp_g2 = _amplitudes(beta_g_boot)
    tf1, tf2, log_order = _tf_and_log_order(
        amp_r1, amp_r2, amp_g1, amp_g2)

    point_r1, point_r2 = _amplitudes(beta_r)
    point_g1, point_g2 = _amplitudes(beta_g)

    # Equal-gain null: preserve the fitted groundwater annual gain and the
    # semiannual phase, but set the semiannual groundwater magnitude so its
    # gain relative to the river equals the annual gain.
    equal_beta_g = beta_g.copy()
    gain_annual = point_g1 / max(point_r1, EPS)
    target_g2 = gain_annual * point_r2
    phase_g2 = np.arctan2(beta_g[5], beta_g[4])
    equal_beta_g[4] = target_g2 * np.cos(phase_g2)
    equal_beta_g[5] = target_g2 * np.sin(phase_g2)
    equal_fit_g = design @ equal_beta_g
    equal_gw_boot = equal_fit_g[None, :] + weights * resid_g[None, :]
    equal_beta_g_boot = equal_gw_boot @ pinv.T
    null_g1, null_g2 = _amplitudes(equal_beta_g_boot)
    equal_gain_tf1, equal_gain_tf2, equal_gain_log_order = _tf_and_log_order(
        amp_r1, amp_r2, null_g1, null_g2)

    def amplitude_snr(point, draws):
        scale = float(np.std(draws, ddof=1))
        return float(point / scale) if scale > 0 and np.isfinite(scale) else np.nan

    return {
        "tf1": tf1,
        "tf2": tf2,
        "log_order": log_order,
        "equal_gain_log_order": equal_gain_log_order,
        "equal_gain_tf1": equal_gain_tf1,
        "equal_gain_tf2": equal_gain_tf2,
        "snr_r1": amplitude_snr(point_r1, amp_r1),
        "snr_r2": amplitude_snr(point_r2, amp_r2),
        "snr_g1": amplitude_snr(point_g1, amp_g1),
        "snr_g2": amplitude_snr(point_g2, amp_g2),
        "n_months": int(diagnostics["n"]),
        "span_years": float(diagnostics["span_years"]),
        "condition_number": float(diagnostics["condition_number"]),
    }


def build_tf_uncertainty(monthly, pair_table, n_boot=DEFAULT_BOOTSTRAPS,
                         block_months=DEFAULT_BLOCK_MONTHS,
                         seed=DEFAULT_SEED):
    """Return one uncertainty row per Reach and the site-fit count."""
    pair = pair_table[["site_no", "reach_id", "split", "TF1", "TF2"]].copy()
    pair["site_key"] = pair.site_no.astype(str)
    site_boot = {}
    for _, reach_monthly, plan in iter_reach_calendars(
            monthly, pair, n_boot, block_months, seed):
        for site_no, group in reach_monthly.groupby("site_no", sort=False):
            result = bootstrap_site_tf(
                group, n_boot=int(n_boot), block_months=int(block_months),
                seed=seed, calendar_plan=plan)
            if result is not None:
                site_boot[str(site_no)] = result

    rows = []
    for reach_id, group in pair.groupby("reach_id", sort=False):
        group = group[group.site_key.isin(site_boot)].copy()
        if group.empty:
            continue
        sites = group.site_key.tolist()
        n_site = len(sites)
        rng = np.random.default_rng(_stable_seed(reach_id, seed + 1))
        reach_log_order = np.full(int(n_boot), np.nan)
        reach_null_order = np.full(int(n_boot), np.nan)
        for replicate in range(int(n_boot)):
            chosen = rng.integers(0, n_site, n_site)
            # Preserve the shared temporal realization across every sampled site.
            tf1 = np.array([
                site_boot[sites[j]]["tf1"][replicate] for j in chosen
            ])
            tf2 = np.array([
                site_boot[sites[j]]["tf2"][replicate] for j in chosen
            ])
            null_tf1 = np.array([
                site_boot[sites[j]]["equal_gain_tf1"][replicate]
                for j in chosen
            ])
            null_tf2 = np.array([
                site_boot[sites[j]]["equal_gain_tf2"][replicate]
                for j in chosen
            ])
            med_tf1 = np.nanmedian(tf1)
            med_tf2 = np.nanmedian(tf2)
            if med_tf1 > 0 and med_tf2 > 0:
                reach_log_order[replicate] = np.log(med_tf1 / med_tf2)
            med_null_tf1 = np.nanmedian(null_tf1)
            med_null_tf2 = np.nanmedian(null_tf2)
            if med_null_tf1 > 0 and med_null_tf2 > 0:
                reach_null_order[replicate] = np.log(
                    med_null_tf1 / med_null_tf2)

        finite = np.isfinite(reach_log_order)
        finite_null = np.isfinite(reach_null_order)
        if not finite.any() or not finite_null.any():
            continue
        point_tf1 = float(pd.to_numeric(group.TF1, errors="coerce").median())
        point_tf2 = float(pd.to_numeric(group.TF2, errors="coerce").median())
        point_log_order = float(np.log(point_tf1 / point_tf2))
        snr_columns = ("snr_r1", "snr_r2", "snr_g1", "snr_g2")
        snr = {
            name: float(np.nanmedian([site_boot[s][name] for s in sites]))
            for name in snr_columns
        }
        joint_snr = float(np.nanmin(list(snr.values())))
        low, high = np.nanpercentile(reach_log_order[finite], [2.5, 97.5])
        rows.append({
            "reach_id": reach_id,
            "split": group.split.iloc[0],
            "n_sites": int(n_site),
            "TF1": point_tf1,
            "TF2": point_tf2,
            "log_TF1_over_TF2": point_log_order,
            "log_order_ci_low": float(low),
            "log_order_ci_high": float(high),
            "p_TF2_lt_TF1": float(np.mean(reach_log_order[finite] > 0)),
            "p_TF2_lt_TF1_equal_gain_null": float(
                np.mean(reach_null_order[finite_null] > 0)
            ),
            "joint_amplitude_snr": joint_snr,
            **snr,
            "n_months_median": float(np.median([
                site_boot[s]["n_months"] for s in sites
            ])),
            "span_years_median": float(np.median([
                site_boot[s]["span_years"] for s in sites
            ])),
            "condition_number_max": float(np.max([
                site_boot[s]["condition_number"] for s in sites
            ])),
            "bootstrap_replicates": int(n_boot),
            "residual_block_months": int(block_months),
            "residual_block_unit": "calendar_month",
            "residual_bootstrap_method": "paired calendar-block wild bootstrap",
            "temporal_dependence_unit": "Reach",
            "replicate_alignment": "same calendar weights and replicate across sites and Nodes",
        })
    return pd.DataFrame(rows), int(len(site_boot))


def _propagated_fraction(probability, seed, n_boot=10_000):
    probability = np.asarray(probability, float)
    probability = probability[np.isfinite(probability)]
    if not len(probability):
        return [np.nan, np.nan, np.nan], np.array([])
    rng = np.random.default_rng(seed)
    estimates = np.empty(int(n_boot))
    for replicate in range(int(n_boot)):
        sampled = probability[rng.integers(0, len(probability), len(probability))]
        estimates[replicate] = np.mean(rng.random(len(sampled)) < sampled)
    return [
        float(np.percentile(estimates, 2.5)),
        float(np.mean(probability)),
        float(np.percentile(estimates, 97.5)),
    ], estimates


def summarize_tf_uncertainty(table, seed=DEFAULT_SEED):
    """Summarize propagated ordering, equal-gain null, and SNR dependence."""
    output = {
        "method": (
            "paired HC2-adjusted calendar-block wild bootstrap at each site's "
            "fixed accepted monthly dates; all four band amplitudes refitted; "
            "sites resampled within Reach with shared weights and synchronized replicates"
        ),
        "ordering_statistic": (
            "log(TF1/TF2); positive is exactly equivalent to TF2<TF1"
        ),
        "equal_gain_null": (
            "groundwater semiannual fitted amplitude adjusted so annual and "
            "semiannual groundwater/river gains are equal; original dates, "
            "phases, baseline/trend, and paired residual blocks retained"
        ),
        "residual_block_months": int(table.residual_block_months.iloc[0]),
        "residual_block_unit": "calendar_month",
        "temporal_dependence_unit": "Reach",
        "endpoint_wrapping": False,
        "classification_probability": {
            "robust_lowpass": ">=0.95",
            "uncertain": "0.05<p<0.95",
            "robust_opposite": "<=0.05",
        },
    }
    for index, (label, subset) in enumerate((
        ("ALL", table),
        ("DEV", table[table.split.eq("dev")]),
        ("HOLDOUT", table[table.split.eq("holdout")]),
    )):
        propagated, _ = _propagated_fraction(
            subset.p_TF2_lt_TF1, seed + 100 + index)
        null_interval, null_draws = _propagated_fraction(
            subset.p_TF2_lt_TF1_equal_gain_null, seed + 200 + index)
        observed_point = float((subset.TF2 < subset.TF1).mean())
        p_equal_gain = (
            float((1 + np.sum(null_draws >= observed_point)) / (1 + len(null_draws)))
            if len(null_draws) else np.nan
        )
        probability = subset.p_TF2_lt_TF1
        block = {
            "n_reach": int(len(subset)),
            "point_fraction_TF2_lt_TF1": observed_point,
            "uncertainty_propagated_fraction_TF2_lt_TF1": propagated,
            "robust_lowpass_fraction_p_ge_0p95": float((probability >= 0.95).mean()),
            "uncertain_fraction": float(
                ((probability > 0.05) & (probability < 0.95)).mean()
            ),
            "robust_opposite_fraction_p_le_0p05": float((probability <= 0.05).mean()),
            "equal_gain_sampling_null_fraction": null_interval,
            "p_observed_point_vs_equal_gain_null": p_equal_gain,
            "median_joint_amplitude_snr": float(subset.joint_amplitude_snr.median()),
        }
        if len(subset) >= 3:
            ranked = subset.copy()
            ranked["snr_group"] = pd.qcut(
                ranked.joint_amplitude_snr.rank(method="first"),
                q=3, labels=["low", "middle", "high"],
            )
            block["joint_amplitude_snr_tertiles"] = []
            for group_label in ("low", "middle", "high"):
                group = ranked[ranked.snr_group.eq(group_label)]
                interval, _ = _propagated_fraction(
                    group.p_TF2_lt_TF1,
                    seed + 1000 + index * 10
                    + {"low": 0, "middle": 1, "high": 2}[group_label],
                )
                block["joint_amplitude_snr_tertiles"].append({
                    "tertile": group_label,
                    "n_reach": int(len(group)),
                    "snr_range": [float(group.joint_amplitude_snr.min()),
                                  float(group.joint_amplitude_snr.max())],
                    "point_fraction_TF2_lt_TF1": float(
                        (group.TF2 < group.TF1).mean()
                    ),
                    "uncertainty_propagated_fraction_TF2_lt_TF1": interval,
                })
        output[label] = block
    return output
