"""One auditable annual+semiannual harmonic implementation for the GRL pipeline."""

from __future__ import annotations

import numpy as np

from importlib import import_module as _import_module
_pipeline_config = _import_module("09_pipeline_config")
HARMONIC_MAX_CONDITION_NUMBER = _pipeline_config.HARMONIC_MAX_CONDITION_NUMBER
HARMONIC_MIN_POINTS = _pipeline_config.HARMONIC_MIN_POINTS
HARMONIC_MIN_SPAN_YEARS = _pipeline_config.HARMONIC_MIN_SPAN_YEARS
HARMONIC_N_PARAMETERS = _pipeline_config.HARMONIC_N_PARAMETERS


FREQUENCIES = (1.0, 2.0)


def harmonic_design(t, frequencies=FREQUENCIES):
    """Intercept + centered trend + sine/cosine pairs at each frequency."""
    t = np.asarray(t, dtype=float)
    columns = [np.ones(len(t)), t - t.mean()]
    for frequency in frequencies:
        columns.extend([
            np.cos(2 * np.pi * frequency * t),
            np.sin(2 * np.pi * frequency * t),
        ])
    return np.column_stack(columns)


def prepare_harmonic_data(
    t,
    x,
    frequencies=FREQUENCIES,
    min_points=HARMONIC_MIN_POINTS,
    min_span_years=HARMONIC_MIN_SPAN_YEARS,
    max_condition_number=HARMONIC_MAX_CONDITION_NUMBER,
):
    """Return screened ``(t, x, design, diagnostics)`` or ``None``.

    The screen is deliberately applied before least squares and checks sample
    size, elapsed support, full column rank, and numerical conditioning. It
    never truncates an accepted record: all pairwise-finite values are used.
    """
    t = np.asarray(t, dtype=float)
    x = np.asarray(x, dtype=float)
    finite = np.isfinite(t) & np.isfinite(x)
    t, x = t[finite], x[finite]
    n = int(len(x))
    span_years = float(np.ptp(t)) if n else np.nan
    if n < int(min_points) or not np.isfinite(span_years) or span_years < float(min_span_years):
        return None
    design = harmonic_design(t, frequencies=frequencies)
    rank = int(np.linalg.matrix_rank(design))
    condition_number = float(np.linalg.cond(design))
    expected_rank = int(design.shape[1])
    if (
        rank != expected_rank
        or expected_rank != HARMONIC_N_PARAMETERS
        or not np.isfinite(condition_number)
        or condition_number > float(max_condition_number)
    ):
        return None
    diagnostics = {
        "n": n,
        "span_years": span_years,
        "rank": rank,
        "condition_number": condition_number,
    }
    return t, x, design, diagnostics


def fit_harmonic(t, x, frequencies=FREQUENCIES):
    """Fit the shared model and return coefficients, bands, fit and residual."""
    prepared = prepare_harmonic_data(t, x, frequencies=frequencies)
    if prepared is None:
        return None
    t, x, design, diagnostics = prepared
    beta, *_ = np.linalg.lstsq(design, x, rcond=None)
    fitted = design @ beta
    # Keep the periodic reconstruction separate from the intercept and trend.
    # This makes it possible to compare the annual+semiannual co-variation of
    # two series on exactly the same observation dates as their residual
    # correlation, without conflating that comparison with temporal sampling.
    periodic_fitted = design[:, 2:] @ beta[2:]
    baseline_trend_fitted = design[:, :2] @ beta[:2]
    residual = x - fitted
    amplitudes = []
    phases = []
    for band in range(len(frequencies)):
        cosine = 2 + 2 * band
        sine = cosine + 1
        amplitudes.append(float(np.hypot(beta[cosine], beta[sine])))
        phases.append(float(np.arctan2(beta[sine], beta[cosine])))
    variance = float(np.var(x))
    return {
        **diagnostics,
        "t": t,
        "x": x,
        "design": design,
        "beta": beta,
        "fitted": fitted,
        "periodic_fitted": periodic_fitted,
        "baseline_trend_fitted": baseline_trend_fitted,
        "residual": residual,
        "amplitudes": tuple(amplitudes),
        "phases": tuple(phases),
        "amp": amplitudes[0],
        "phase": phases[0],
        "r2": 1 - float(np.var(residual)) / variance if variance > 0 else np.nan,
    }
def harmonic_criterion():
    """Machine-readable main-analysis criterion for manifests and outputs."""
    return {
        "minimum_pairwise_finite_observations": HARMONIC_MIN_POINTS,
        "minimum_elapsed_span_years": HARMONIC_MIN_SPAN_YEARS,
        "model_parameters": HARMONIC_N_PARAMETERS,
        "full_column_rank_required": True,
        "maximum_condition_number": HARMONIC_MAX_CONDITION_NUMBER,
        "accepted_record_policy": "use all pairwise-finite observations; do not truncate",
    }


def residualize_harmonic_draws(t, draws):
    """Project each simulated series through the observed six-column model.

    Rows are simulations and columns are the same paired dates used for the
    observed event-r. Standardize *after* projecting out baseline, trend and
    annual/semiannual terms, making row dot products residual correlations.
    """
    t = np.asarray(t, dtype=float)
    draws = np.asarray(draws, dtype=float)
    if draws.ndim != 2 or draws.shape[1] != len(t):
        raise ValueError('draws must have shape (simulations, observed dates)')
    if not np.isfinite(t).all() or not np.isfinite(draws).all():
        raise ValueError('OU projection requires finite dates and simulations')
    design = harmonic_design(t)
    if np.linalg.matrix_rank(design) != HARMONIC_N_PARAMETERS:
        raise ValueError('OU projection requires the full-rank observed design')
    beta, *_ = np.linalg.lstsq(design, draws.T, rcond=None)
    residual = draws - (design @ beta).T
    residual -= residual.mean(axis=1, keepdims=True)
    scale = residual.std(axis=1, keepdims=True)
    if not np.isfinite(scale).all() or np.any(scale <= 0):
        raise ValueError('OU simulation has no finite residual variance')
    return residual / scale
