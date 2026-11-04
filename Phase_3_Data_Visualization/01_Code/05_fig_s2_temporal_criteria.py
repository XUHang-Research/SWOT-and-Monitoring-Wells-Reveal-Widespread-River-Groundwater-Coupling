#!/usr/bin/env python3
"""Figure S2 — temporal-support sensitivity of the harmonic conclusions.

The annual + semiannual harmonic model has six coefficients (intercept, trend,
and sine/cosine pairs at 1 and 2 yr-1).  This figure compares three
pre-specified eligibility screens applied to the complete monthly records:

    loose sensitivity : >=12 paired months and >=1.0 yr span
    main analysis     : >=18 paired months and >=1.5 yr span
    strict sensitivity: >=24 paired months and >=2.0 yr span

Every accepted fit must additionally have a full-rank six-column design matrix
and condition number <= 1e4.  The eligibility screen never truncates a record:
all available observations are used in every accepted fit.

The residual correlation is independently recomputed and screened from the
complete daily paired records. Daily residual correlation eligibility is never intersected
with monthly harmonic eligibility.  Its established reliability floor of 24
paired days is kept for all three screens, while the elapsed-span screen varies
from 1.0 to 2.0 yr.  R_A/tau and transfer-function conclusions retain the monthly
screens above, so conclusion-specific denominators are reported explicitly.

Panels:
  (a) daily paired-record support and the strictest daily criterion passed;
  (b) retention of daily residual-correlation sites, Reaches, and networks vs loose;
  (c) monthly paired-record support and the strictest monthly criterion passed;
  (d) retention of monthly-harmonic sites, Reaches, and networks vs loose;
  (e) Reach-bootstrap 95% CIs for three manuscript conclusions at 1 km, with
      point estimates and interval ranges printed beside the error bars.

The temporal-by-distance matrix is still calculated and archived in the
machine-readable statistics file for rendering as Fig. S8b, where it belongs
with the R_A–tau physical-screen robustness tests.

Inputs:
  Phase-1 distance1000/6matched_results_monthly.txt
  Phase-1 distance1000/5matched_results_daily.txt
  Phase-2 distance1000_data/08_global_coupling_metrics.parquet
  Phase-2 distance1000_data/08_event_metrics_all_sites.parquet (reproduction check)
  Phase-2 shared_cache/distance_robustness_monthly_strict.parquet

Outputs:
  fig_s2_temporal_criteria.{pdf,svg,png}
  fig_s2_temporal_criteria_stats.json

Requires: numpy, pandas, pyarrow>=20, matplotlib; cmcrameri optional.
"""
from __future__ import annotations

import json
from pathlib import Path

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from importlib import import_module as _import_module
_grl_figure_style = _import_module("13_grl_figure_style")
GRL_TEXT_PT = _grl_figure_style.SUPPLEMENT_TEXT_PT
GRL_PANEL_PT = _grl_figure_style.SUPPLEMENT_PANEL_PT
configure_grl_matplotlib = _grl_figure_style.configure_grl_supplementary_matplotlib
grl_figsize = _grl_figure_style.grl_figsize

try:
    from cmcrameri import cm as scm
    MATRIX_CMAP = scm.oslo_r
    MATRIX_CMAP_NAME = "cmcrameri.oslo_r"
except ImportError:
    MATRIX_CMAP = mpl.colormaps["cividis"]
    MATRIX_CMAP_NAME = "matplotlib.cividis"


HERE = Path(__file__).resolve().parent.parent
ROOT = HERE.parent
MONTHLY = (
    ROOT / "Phase_1_Data_Preprocessing" / "distance1000"
    / "6matched_results_monthly.txt"
)
DAILY = (
    ROOT / "Phase_1_Data_Preprocessing" / "distance1000"
    / "5matched_results_daily.txt"
)
PAIR_TABLE = (
    ROOT / "Phase_2_Data_Analysis" / "distance1000_data"
    / "08_global_coupling_metrics.parquet"
)
EVENT_TABLE = (
    ROOT / "Phase_2_Data_Analysis" / "distance1000_data"
    / "08_event_metrics_all_sites.parquet"
)
DISTANCE_MONTHLY = (
    ROOT / "Phase_2_Data_Analysis" / "shared_cache"
    / "distance_robustness_monthly_strict.parquet"
)
DISTANCE_COLUMN = "distance_prior_bank_m"
CONDITION_MAX = 1.0e4
EVENT_MIN_PAIRED_DAYS = 24
PERIOD_DAYS = 365.25
N_BOOT = 10_000

CRITERIA = {
    "loose": {"n_min": 12, "span_min_years": 1.0, "color": "#E69F00", "marker": "o"},
    "main": {"n_min": 18, "span_min_years": 1.5, "color": "#0072B2", "marker": "s"},
    "strict": {"n_min": 24, "span_min_years": 2.0, "color": "#009E73", "marker": "^"},
}
DISPLAY = {"loose": "loose", "main": "main", "strict": "strict"}

configure_grl_matplotlib()


def country(site_no: str) -> str:
    """Match the analysis definition of country/network."""
    return str(site_no).split("+")[0].split("_")[0]


def yfrac(date) -> np.ndarray:
    date = pd.DatetimeIndex(date)
    return date.year.to_numpy() + (date.dayofyear.to_numpy() - 1) / PERIOD_DAYS


def design_matrix(t: np.ndarray) -> np.ndarray:
    t = np.asarray(t, float)
    return np.column_stack([
        np.ones(len(t)),
        t - t.mean(),
        np.cos(2 * np.pi * t),
        np.sin(2 * np.pi * t),
        np.cos(4 * np.pi * t),
        np.sin(4 * np.pi * t),
    ])


def fit_site(group: pd.DataFrame) -> dict:
    """Fit both records once and retain diagnostics for all three gates."""
    group = group.sort_values("date")
    t = yfrac(group.date)
    river = pd.to_numeric(group.river_wse, errors="coerce").to_numpy(float)
    gw = pd.to_numeric(group.Value_EGM2008_m, errors="coerce").to_numpy(float)
    valid = np.isfinite(t) & np.isfinite(river) & np.isfinite(gw)
    t, river, gw = t[valid], river[valid], gw[valid]
    n = int(len(t))
    span = float(np.ptp(t)) if n else np.nan
    base = {
        "site_no": group.site_no.iloc[0],
        "reach_id": group.reach_id.iloc[0],
        "country": group.country.iloc[0] if "country" in group else country(group.site_no.iloc[0]),
        "n": n,
        "span_years": span,
        "rank": 0,
        "condition_number": np.nan,
        "fit_valid": False,
        "A": np.nan,
        "tau_days": np.nan,
        "TF1": np.nan,
        "TF2": np.nan,
    }
    if n < 6:
        return base
    matrix = design_matrix(t)
    rank = int(np.linalg.matrix_rank(matrix))
    condition = float(np.linalg.cond(matrix))
    base.update(rank=rank, condition_number=condition)
    if rank != 6 or not np.isfinite(condition) or condition > CONDITION_MAX:
        return base
    beta_r, *_ = np.linalg.lstsq(matrix, river, rcond=None)
    beta_g, *_ = np.linalg.lstsq(matrix, gw, rcond=None)
    amp_r1 = float(np.hypot(beta_r[2], beta_r[3]))
    amp_g1 = float(np.hypot(beta_g[2], beta_g[3]))
    amp_r2 = float(np.hypot(beta_r[4], beta_r[5]))
    amp_g2 = float(np.hypot(beta_g[4], beta_g[5]))
    eps = np.finfo(float).eps
    if amp_r1 <= eps:
        return base
    phase_r = float(np.arctan2(beta_r[3], beta_r[2]))
    phase_g = float(np.arctan2(beta_g[3], beta_g[2]))
    dphase = (phase_g - phase_r + np.pi) % (2 * np.pi) - np.pi
    base.update(
        fit_valid=True,
        A=amp_g1 / amp_r1,
        tau_days=dphase / (2 * np.pi) * PERIOD_DAYS,
        TF1=(amp_g1 / amp_r1) ** 2,
        TF2=(amp_g2 / amp_r2) ** 2 if amp_r2 > eps else np.nan,
    )
    return base


def fit_daily_event_site(group: pd.DataFrame) -> dict:
    """Fit daily residuals and expose independently screened residual correlation values.

    The gates do not truncate an accepted record, so their residual correlation
    is identical when a site passes more than one gate. Eligibility is still
    evaluated independently from the daily support and numerical diagnostics.
    """
    t = yfrac(group.date)
    river = pd.to_numeric(group.river_wse, errors="coerce").to_numpy(float)
    gw = pd.to_numeric(group.Value_EGM2008_m, errors="coerce").to_numpy(float)
    valid = np.isfinite(t) & np.isfinite(river) & np.isfinite(gw)
    t, river, gw = t[valid], river[valid], gw[valid]
    n = int(len(t))
    span = float(np.ptp(t)) if n else np.nan
    base = {
        "site_no": group.site_no.iloc[0],
        "event_reach_id": group.reach_id.iloc[0],
        "event_n": n,
        "event_span_years": span,
        "event_rank": 0,
        "event_condition_number": np.nan,
        "event_fit_valid": False,
        "event_r_daily": np.nan,
    }
    for name in CRITERIA:
        base[f"event_eligible_{name}"] = False
        base[f"event_r_{name}"] = np.nan
    if n < 6:
        return base

    matrix = design_matrix(t)
    rank = int(np.linalg.matrix_rank(matrix))
    condition = float(np.linalg.cond(matrix))
    base.update(event_rank=rank, event_condition_number=condition)
    if rank != 6 or not np.isfinite(condition) or condition > CONDITION_MAX:
        return base

    beta_r, *_ = np.linalg.lstsq(matrix, river, rcond=None)
    beta_g, *_ = np.linalg.lstsq(matrix, gw, rcond=None)
    residual_r = river - matrix @ beta_r
    residual_g = gw - matrix @ beta_g
    if np.std(residual_r) <= 0 or np.std(residual_g) <= 0:
        return base
    event_r = float(np.corrcoef(residual_r, residual_g)[0, 1])
    if not np.isfinite(event_r):
        return base

    base.update(event_fit_valid=True, event_r_daily=event_r)
    for name, rule in CRITERIA.items():
        eligible = (
            n >= EVENT_MIN_PAIRED_DAYS
            and span >= rule["span_min_years"]
        )
        base[f"event_eligible_{name}"] = eligible
        if eligible:
            base[f"event_r_{name}"] = event_r
    return base


def criterion_mask(frame: pd.DataFrame, name: str) -> pd.Series:
    rule = CRITERIA[name]
    return (
        frame.fit_valid
        & frame.n.ge(rule["n_min"])
        & frame.span_years.ge(rule["span_min_years"])
    )


def circular_mean_days(values) -> float:
    values = pd.to_numeric(pd.Series(values), errors="coerce").dropna().to_numpy(float)
    if not len(values):
        return np.nan
    angle = values * (2 * np.pi / PERIOD_DAYS)
    return float(np.angle(np.mean(np.exp(1j * angle))) * PERIOD_DAYS / (2 * np.pi))


def finite_median(values) -> float:
    values = pd.to_numeric(pd.Series(values), errors="coerce").to_numpy(float)
    values = values[np.isfinite(values)]
    return float(np.median(values)) if len(values) else np.nan


def reach_metrics(frame: pd.DataFrame) -> pd.DataFrame:
    """Reach aggregation: median amplitudes/TF/r and circular mean phase."""
    rows = []
    for reach_id, group in frame.groupby("reach_id", sort=False):
        rows.append({
            "reach_id": reach_id,
            "country": group.country.iloc[0],
            "A": finite_median(group.A),
            "tau_days": circular_mean_days(group.tau_days),
            "TF1": finite_median(group.TF1),
            "TF2": finite_median(group.TF2),
            "event_r": finite_median(group.event_r)
                if "event_r" in group else np.nan,
        })
    return pd.DataFrame(rows)


def bootstrap_ci(values, statistic, seed: int) -> list[float]:
    values = np.asarray(values, float)
    values = values[np.isfinite(values)]
    if not len(values):
        return [np.nan, np.nan, np.nan]
    rng = np.random.default_rng(seed)
    draws = np.empty(N_BOOT, float)
    # Iteration avoids a potentially large N_BOOT x n_reach allocation.
    for i in range(N_BOOT):
        sample = values[rng.integers(0, len(values), len(values))]
        draws[i] = statistic(sample)
    return [
        float(np.percentile(draws, 2.5)),
        float(statistic(values)),
        float(np.percentile(draws, 97.5)),
    ]


def json_ready(value):
    if isinstance(value, dict):
        return {str(k): json_ready(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [json_ready(v) for v in value]
    if isinstance(value, (np.integer,)):
        return int(value)
    if isinstance(value, (np.floating,)):
        return None if not np.isfinite(value) else float(value)
    if isinstance(value, float) and not np.isfinite(value):
        return None
    return value


def panel_label(fig, ax, label, dx=-0.045, dy=0.012):
    box = ax.get_position()
    fig.text(box.x0 + dx, box.y1 + dy, label, fontsize=GRL_PANEL_PT, fontweight="bold",
             fontstyle="normal", ha="right", va="bottom")


def load_daily_event_metrics() -> tuple[pd.DataFrame, dict]:
    """Read the 1-km daily pairs in chunks and recompute residual correlation r by site."""
    wanted = [
        "site_no", "reach_id", "date", "river_wse", "Value_EGM2008_m",
        DISTANCE_COLUMN,
    ]
    frames = []
    input_rows = 0
    distance_rows = 0
    for chunk in pd.read_csv(
        DAILY, sep="\t", usecols=wanted, chunksize=500_000, low_memory=False,
    ):
        input_rows += len(chunk)
        for column in ("river_wse", "Value_EGM2008_m", DISTANCE_COLUMN):
            chunk[column] = pd.to_numeric(chunk[column], errors="coerce")
        chunk["date"] = pd.to_datetime(chunk.date, errors="coerce").dt.normalize()
        chunk = chunk[chunk[DISTANCE_COLUMN].le(1000)]
        distance_rows += len(chunk)
        chunk = chunk.dropna(
            subset=["site_no", "reach_id", "date", "river_wse", "Value_EGM2008_m"]
        )
        frames.append(
            chunk[["site_no", "reach_id", "date", "river_wse", "Value_EGM2008_m"]]
        )
    if not frames:
        raise RuntimeError(f"no readable daily records in {DAILY}")
    daily = pd.concat(frames, ignore_index=True)
    event = pd.DataFrame(
        fit_daily_event_site(group)
        for _, group in daily.groupby("site_no", sort=False)
    )
    event["country"] = event.site_no.map(country)
    diagnostics = {
        "input_rows": int(input_rows),
        "rows_within_1km": int(distance_rows),
        "pairwise_finite_rows_within_1km": int(len(daily)),
        "screened_sites": int(len(event)),
        "full_rank_sites": int(event.event_rank.eq(6).sum()),
        "numerically_valid_event_fits": int(event.event_fit_valid.sum()),
    }
    return event, diagnostics


def load_primary() -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, dict, dict]:
    wanted = {
        "site_no", "reach_id", "date", "river_wse", "Value_EGM2008_m",
        DISTANCE_COLUMN, "number_match_SG",
    }
    monthly = pd.read_csv(
        MONTHLY, sep="\t", usecols=lambda col: col in wanted, low_memory=False,
    )
    for column in ("river_wse", "Value_EGM2008_m", DISTANCE_COLUMN, "number_match_SG"):
        if column in monthly:
            monthly[column] = pd.to_numeric(monthly[column], errors="coerce")
    monthly["date"] = pd.to_datetime(monthly.date, errors="coerce")
    monthly = monthly.dropna(subset=["river_wse", "Value_EGM2008_m", "date"])
    monthly = monthly[monthly[DISTANCE_COLUMN].le(1000)]
    if "number_match_SG" in monthly:
        monthly = monthly[
            monthly.number_match_SG.eq(1) | monthly.number_match_SG.isna()
        ]
    monthly = monthly.copy()
    monthly["country"] = monthly.site_no.map(country)
    site = pd.DataFrame(fit_site(group) for _, group in monthly.groupby("site_no", sort=False))

    archive = pd.read_parquet(
        PAIR_TABLE, columns=["site_no", "reach_id", "A", "tau_days"],
    ).rename(columns={
        "reach_id": "reach_id_archive",
        "A": "A_archive",
        "tau_days": "tau_days_archive",
    })
    archive_duplicate_sites = int(archive.site_no.duplicated(keep=False).sum())
    if archive_duplicate_sites:
        raise RuntimeError(
            "archived main harmonic table contains duplicate site_no values: "
            f"{archive_duplicate_sites} rows"
        )
    main_recomputed = site.loc[
        criterion_mask(site, "main"), ["site_no", "reach_id", "A", "tau_days"]
    ].copy()
    harmonic_outer = main_recomputed.merge(
        archive, on="site_no", how="outer", indicator=True, validate="one_to_one",
    )
    missing_from_recompute = int(harmonic_outer._merge.eq("right_only").sum())
    new_in_recompute = int(harmonic_outer._merge.eq("left_only").sum())
    check = harmonic_outer[harmonic_outer._merge.eq("both")].copy()
    reach_id_mismatch = int(check.reach_id.ne(check.reach_id_archive).sum())
    max_A_difference = float((check.A - check.A_archive).abs().max())
    tau_difference = (
        (check.tau_days - check.tau_days_archive + PERIOD_DAYS / 2)
        % PERIOD_DAYS - PERIOD_DAYS / 2
    )
    max_tau_difference_days = float(tau_difference.abs().max())
    tolerance = 1e-6
    if (
        missing_from_recompute
        or new_in_recompute
        or reach_id_mismatch
        or len(check) != len(archive)
        or not np.isfinite(max_A_difference)
        or not np.isfinite(max_tau_difference_days)
        or max_A_difference > tolerance
        or max_tau_difference_days > tolerance
    ):
        raise RuntimeError(
            "harmonic refit does not reproduce the archived A/tau values: "
            f"missing={missing_from_recompute}, new={new_in_recompute}, "
            f"reach mismatch={reach_id_mismatch}, n={len(check)}, "
            f"max |dA|={max_A_difference:.3e}, "
            f"max circular |dtau|={max_tau_difference_days:.3e} d"
        )
    harmonic_reproduction = {
        "overlap_sites": int(len(check)),
        "archive_sites": int(len(archive)),
        "recomputed_main_sites": int(len(main_recomputed)),
        "archive_reaches": int(archive.reach_id_archive.nunique()),
        "recomputed_main_reaches": int(main_recomputed.reach_id.nunique()),
        "archive_duplicate_site_rows": archive_duplicate_sites,
        "missing_from_recompute": missing_from_recompute,
        "new_in_recompute": new_in_recompute,
        "reach_id_mismatch_sites": reach_id_mismatch,
        "maximum_absolute_A_difference": max_A_difference,
        "maximum_circular_tau_difference_days": max_tau_difference_days,
        "tolerance": tolerance,
        "passed": True,
    }

    event, daily_diagnostics = load_daily_event_metrics()
    event_archive = pd.read_parquet(
        EVENT_TABLE, columns=["site_no", "event_r", "n_obs"],
    ).rename(columns={"event_r": "event_r_archive", "n_obs": "n_obs_archive"})
    event_archive_duplicate_sites = int(
        event_archive.site_no.duplicated(keep=False).sum()
    )
    if event_archive_duplicate_sites:
        raise RuntimeError(
            "archived main residual correlation table contains duplicate site_no values: "
            f"{event_archive_duplicate_sites} rows"
        )
    main_event = event[event.event_eligible_main].merge(
        event_archive, on="site_no", how="outer", indicator=True,
        validate="one_to_one",
    )
    missing_from_recompute = int(main_event._merge.eq("right_only").sum())
    new_in_recompute = int(main_event._merge.eq("left_only").sum())
    event_overlap = main_event[main_event._merge.eq("both")].copy()
    max_event_difference = float(
        (event_overlap.event_r_main - event_overlap.event_r_archive).abs().max()
    )
    n_obs_mismatch = int(
        event_overlap.event_n.ne(event_overlap.n_obs_archive).sum()
    )
    event_tolerance = 1e-9
    if (
        missing_from_recompute
        or new_in_recompute
        or n_obs_mismatch
        or not np.isfinite(max_event_difference)
        or max_event_difference > event_tolerance
    ):
        raise RuntimeError(
            "daily main-gate residual correlation r does not reproduce the archived table: "
            f"missing={missing_from_recompute}, new={new_in_recompute}, "
            f"n_obs mismatch={n_obs_mismatch}, "
            f"max |dr|={max_event_difference:.3e}"
        )
    event_reproduction = {
        "overlap_sites": int(len(event_overlap)),
        "missing_from_recompute": missing_from_recompute,
        "new_in_recompute": new_in_recompute,
        "n_obs_mismatch_sites": n_obs_mismatch,
        "archive_duplicate_site_rows": event_archive_duplicate_sites,
        "maximum_absolute_event_r_difference": max_event_difference,
        "tolerance": event_tolerance,
        "passed": True,
    }
    reproduction = {
        "monthly_A_tau": harmonic_reproduction,
        "daily_event_r_main_gate": event_reproduction,
    }
    return monthly, site, event, reproduction, daily_diagnostics


def summarize_primary(
    site: pd.DataFrame,
    daily_event: pd.DataFrame,
) -> tuple[dict, dict[str, dict[str, pd.DataFrame]]]:
    """Summarize monthly and daily conclusions on independent populations.

    ``site`` contains the monthly harmonic fits used for A/tau and TF.  The
    residual correlation population is selected only from ``daily_event`` under the
    criterion-specific daily support gate.  In particular, no monthly-site
    eligibility mask is applied to residual correlation r before its Reach median is formed.
    """
    summaries = {}
    reaches = {}
    for j, name in enumerate(CRITERIA):
        monthly_selected = site[criterion_mask(site, name)].copy()
        monthly_reach = reach_metrics(monthly_selected)

        event_selected = daily_event[
            daily_event[f"event_eligible_{name}"]
            & daily_event[f"event_r_{name}"].notna()
        ].copy()
        event_selected["event_r"] = event_selected[f"event_r_{name}"]
        event_reach = (
            event_selected.groupby("event_reach_id", sort=False)
            .agg(event_r=("event_r", "median"), country=("country", "first"))
            .rename_axis("reach_id")
            .reset_index()
        )

        reaches[name] = {
            "monthly_harmonic": monthly_reach,
            "daily_event_r": event_reach,
        }
        event = event_reach.event_r.dropna().to_numpy(float)
        at = monthly_reach.dropna(subset=["A", "tau_days"])
        lowpass = monthly_reach.dropna(subset=["TF1", "TF2"])
        amp_lead = ((at.A > 1.2) | (at.tau_days <= 0)).to_numpy(float)
        lp = (lowpass.TF2 < lowpass.TF1).to_numpy(float)
        monthly_at_sites = monthly_selected.dropna(subset=["A", "tau_days"])
        monthly_tf_sites = monthly_selected.dropna(subset=["TF1", "TF2"])
        summaries[name] = {
            "criterion": {
                "minimum_paired_months": CRITERIA[name]["n_min"],
                "minimum_span_years": CRITERIA[name]["span_min_years"],
                "full_rank_required": True,
                "maximum_condition_number": CONDITION_MAX,
            },
            "daily_event_r_criterion": {
                "minimum_paired_days": EVENT_MIN_PAIRED_DAYS,
                "minimum_span_years": CRITERIA[name]["span_min_years"],
                "full_rank_required": True,
                "maximum_condition_number": CONDITION_MAX,
                "all_available_pairwise_finite_days_used": True,
                "independent_of_monthly_harmonic_eligibility": True,
            },
            # Keep the legacy scalar counts for the monthly harmonic population.
            # The daily and monthly retention panels use their explicit populations.
            "n_site": int(len(monthly_selected)),
            "n_reach": int(monthly_reach.reach_id.nunique()),
            "n_country_or_network": int(monthly_selected.country.nunique()),
            "panel_b_population": "daily_event_r",
            "panel_d_population": "monthly_harmonic",
            "monthly_harmonic_population": {
                "n_site": int(len(monthly_selected)),
                "n_reach": int(monthly_reach.reach_id.nunique()),
                "n_country_or_network": int(monthly_selected.country.nunique()),
            },
            "daily_event_r_population": {
                "n_site": int(len(event_selected)),
                "n_reach": int(event_reach.reach_id.nunique()),
                "n_country_or_network": int(event_selected.country.nunique()),
                "monthly_eligibility_intersection_applied": False,
            },
            "numerical_failures_after_time_screen": int((
                site.n.ge(CRITERIA[name]["n_min"])
                & site.span_years.ge(CRITERIA[name]["span_min_years"])
                & ~site.fit_valid
            ).sum()),
            "event_r_fraction_ge_0p3": bootstrap_ci(
                (event >= 0.3).astype(float), np.mean, 100 + j,
            ),
            "amplified_or_leading_fraction": bootstrap_ci(
                amp_lead, np.mean, 200 + j,
            ),
            "lowpass_fraction_TF2_lt_TF1": bootstrap_ci(
                lp, np.mean, 300 + j,
            ),
            "denominators": {
                "event_r_sites": int(len(event_selected)),
                "event_r_reaches": int(len(event)),
                "event_r_countries_or_networks": int(
                    event_selected.country.nunique()
                ),
                "A_tau_sites": int(len(monthly_at_sites)),
                "A_tau_reaches": int(len(at)),
                "A_tau_countries_or_networks": int(
                    monthly_at_sites.country.nunique()
                ),
                "two_band_TF_sites": int(len(monthly_tf_sites)),
                "two_band_TF_reaches": int(len(lowpass)),
                "two_band_TF_countries_or_networks": int(
                    monthly_tf_sites.country.nunique()
                ),
            },
        }
    return summaries, reaches


def distance_matrix() -> tuple[np.ndarray, np.ndarray, dict]:
    cached = pd.read_parquet(DISTANCE_MONTHLY)
    for column in ("threshold_m", "river_wse", "Value_EGM2008_m"):
        cached[column] = pd.to_numeric(cached[column], errors="coerce")
    cached["date"] = pd.to_datetime(cached.date, errors="coerce")
    cached = cached.dropna(subset=["threshold_m", "river_wse", "Value_EGM2008_m", "date"])
    if "country" not in cached:
        cached["country"] = cached.site_no.map(country)

    thresholds = np.array(sorted(cached.threshold_m.dropna().unique()), dtype=int)
    values = np.full((len(CRITERIA), len(thresholds)), np.nan)
    counts = np.zeros_like(values, dtype=int)
    detail = {}
    for col, threshold in enumerate(thresholds):
        subset = cached[cached.threshold_m.eq(threshold)].copy()
        fitted = pd.DataFrame(
            fit_site(group) for _, group in subset.groupby("site_no", sort=False)
        )
        detail[str(int(threshold))] = {}
        for row, name in enumerate(CRITERIA):
            selected = fitted[criterion_mask(fitted, name)].copy()
            reach = reach_metrics(selected)
            reach = reach.dropna(subset=["A", "tau_days"])
            mask = (reach.A > 1.2) | (reach.tau_days <= 0)
            values[row, col] = float(mask.mean()) if len(reach) else np.nan
            counts[row, col] = int(len(reach))
            detail[str(int(threshold))][name] = {
                "n_site": int(len(selected)),
                "n_reach": int(len(reach)),
                "amplified_or_leading_fraction": values[row, col],
            }
    return values, counts, detail


def plot_temporal_support(ax, records, *, daily, span_limit):
    """Display the existing eligibility masks without changing the fit or gate."""
    masks = {
        name: records[f"event_eligible_{name}"] if daily else criterion_mask(records, name)
        for name in CRITERIA
    }
    loose, main, strict = (masks[name] for name in CRITERIA)
    span_column = "event_span_years" if daily else "span_years"
    n_column = "event_n" if daily else "n"
    classes = [
        ("does not pass loose", ~loose, "#B3B3B3", ".", 8, 0.40),
        ("loose only", loose & ~main, CRITERIA["loose"]["color"], "o", 11, 0.70),
        ("main, not strict", main & ~strict, CRITERIA["main"]["color"], "s", 11, 0.70),
        ("passes strict", strict, CRITERIA["strict"]["color"], "^", 12, 0.70),
    ]
    for label, mask, color_, marker, size, alpha in classes:
        d = records[mask]
        ax.scatter(d[span_column], d[n_column], s=size, color=color_, marker=marker,
                   alpha=alpha, lw=0,
                   label=f"{label} ({len(d):,} sites)")
    upper = max(ax.get_ylim()[1], EVENT_MIN_PAIRED_DAYS * 1.1)
    for name, rule in CRITERIA.items():
        color_ = rule["color"]
        n_min = EVENT_MIN_PAIRED_DAYS if daily else rule["n_min"]
        ax.plot([rule["span_min_years"], rule["span_min_years"]],
                [n_min, upper], color=color_, lw=0.65, ls=":", zorder=0)
        if not daily:
            ax.plot([rule["span_min_years"], span_limit], [n_min, n_min],
                    color=color_, lw=0.65, ls=":", zorder=0)
    if daily:
        # All daily gates share one floor; do not imply 12/18/24 daily thresholds.
        ax.plot([CRITERIA["loose"]["span_min_years"], span_limit],
                [EVENT_MIN_PAIRED_DAYS] * 2, color="0.35", lw=0.65, ls=":", zorder=0)
    ax.set_xlim(0, span_limit)
    ax.set_ylim(0, upper)
    ax.set_xlabel("Paired-record span (years)")
    ax.set_ylabel("Paired daily observations" if daily else "Paired monthly observations")
    ax.set_title("Daily (≥24 paired days)" if daily else "Monthly", loc="center", pad=5)
    # The key inside panel a serves both temporal-support panels; counts remain in
    # the statistics archive and companion caption rather than in duplicate keys.


def plot_retention(ax, primary, *, daily):
    """Normalize each scale to its own loose population; labels are raw counts."""
    names = list(CRITERIA)
    x = np.arange(len(names))
    population = "daily_event_r_population" if daily else "monthly_harmonic_population"
    entities = [
        ("Sites", "n_site", "#000000", "o", (-18, 7)),
        ("Reaches", "n_reach", "#CC79A7", "s", (0, -13)),
        ("Countries/networks", "n_country_or_network", "#56B4E9", "^", (18, 8)),
    ]
    for label, key, color_, marker, text_offset in entities:
        counts_ = np.array([primary[name][population][key] for name in names], float)
        retention = counts_ / counts_[0] * 100
        ax.plot(x, retention, color=color_, marker=marker, ms=4.2,
                mfc="white" if key != "n_site" else color_, mew=0.8, label=label)
        for xi, yi, count_ in zip(x, retention, counts_.astype(int)):
            # The steeper monthly site curve otherwise crosses its own labels.
            offset = (-18, 11) if not daily and key == "n_site" and xi > 0 else text_offset
            if daily and key == "n_site" and xi == 1:
                offset = (14, 2)  # Fit between the site and country lines at 9 pt.
            if key == "n_site" and xi == 0:
                offset = (-12, 7)  # Keep the larger count label clear of the y spine.
            ax.annotate(f"{count_:,}", (xi, yi), xytext=offset,
                        textcoords="offset points", ha="center", va="bottom",
                        fontsize=GRL_TEXT_PT, color=color_)
    ax.set_xticks(x, [DISPLAY[name].capitalize() for name in names])
    ax.set_xlim(-0.40, 2.40)
    ax.set_ylabel("Retained relative to loose (%)")
    ax.set_ylim(0, 115)
    ax.set_title("Daily retention" if daily else "Monthly retention", loc="center", pad=5)
    ax.axhline(100, color="0.75", lw=0.5, ls=":", zorder=0)
    # The shared key inside panel b serves daily and monthly retention panels.


def make_figure(site, daily_event, primary):
    fig = plt.figure(figsize=grl_figsize(170, 190))
    # The top rows need ordinary axis-label margins; only the conclusion panel
    # needs the larger left margin for its long category labels.
    gs = fig.add_gridspec(
        2, 2, left=0.11, right=0.96, bottom=0.44, top=0.94,
        wspace=0.30, hspace=0.52,
    )
    ax_a = fig.add_subplot(gs[0, 0])
    ax_b = fig.add_subplot(gs[0, 1])
    ax_c = fig.add_subplot(gs[1, 0])
    ax_d = fig.add_subplot(gs[1, 1])
    ax_e = fig.add_axes([0.255, 0.065, 0.705, 0.27])
    span_limit = max(site.span_years.max(), daily_event.event_span_years.max()) * 1.05
    plot_temporal_support(ax_a, daily_event, daily=True, span_limit=span_limit)
    plot_retention(ax_b, primary, daily=True)
    plot_temporal_support(ax_c, site, daily=False, span_limit=span_limit)
    plot_retention(ax_d, primary, daily=False)
    class_handles = [
        mpl.lines.Line2D([], [], linestyle="none", marker=marker, color=color,
                         markersize=4, label=label)
        for label, marker, color in [
            ("Fails loose", ".", "#B3B3B3"),
            ("Loose only", "o", CRITERIA["loose"]["color"]),
            ("Main, not strict", "s", CRITERIA["main"]["color"]),
            ("Passes strict", "^", CRITERIA["strict"]["color"]),
        ]
    ]
    ax_a.legend(handles=class_handles, loc="upper left", ncol=1,
                handletextpad=0.2, labelspacing=0.25, borderaxespad=0.5,
                frameon=True, facecolor="white", edgecolor="none", framealpha=1)
    entity_handles, entity_labels = ax_b.get_legend_handles_labels()
    ax_b.legend(entity_handles, entity_labels, loc="lower left", ncol=1,
                handlelength=1.5, handletextpad=0.4, labelspacing=0.3, borderaxespad=0.6)

    # (e) Three conclusion-level fractions with unchanged Reach-bootstrap uncertainty.
    names = list(CRITERIA)
    metrics = [
        ("Daily r ≥ 0.3", "event_r_fraction_ge_0p3"),
        ("$R_A > 1.2$ or $\\tau$ ≤ 0", "amplified_or_leading_fraction"),
        ("TF$_2$ < TF$_1$", "lowpass_fraction_TF2_lt_TF1"),
    ]
    y_base = np.arange(len(metrics))[::-1]
    offsets = {"loose": 0.18, "main": 0.0, "strict": -0.18}
    for name in names:
        rule = CRITERIA[name]
        for y0, (_, key) in zip(y_base, metrics):
            lo, value, hi = primary[name][key]
            y = y0 + offsets[name]
            ax_e.plot([lo, hi], [y, y], color=rule["color"], lw=1.3)
            ax_e.scatter(value, y, s=23 if name == "main" else 17,
                         marker=rule["marker"], color=rule["color"],
                         edgecolor="white", linewidth=0.4, zorder=3)
            ax_e.text(
                hi + 0.012, y,
                f"{value:.0%} [{lo*100:.0f}–{hi:.0%}]",
                va="center", fontsize=GRL_TEXT_PT, color=rule["color"],
            )
    ax_e.axvline(0.5, color="0.35", lw=0.6, ls=":", zorder=0)
    ax_e.set_yticks(y_base, [label for label, _ in metrics])
    ax_e.set_xlim(0, 1)
    ax_e.set_xlabel("Fraction of reaches (95% CI)")
    handles = [mpl.lines.Line2D([], [], color=rule["color"], marker=rule["marker"],
                               linestyle="none", ms=4, label=DISPLAY[name])
               for name, rule in CRITERIA.items()]
    # Keep the key outside the data region: an in-axis lower-right legend masks
    # the strict frequency-ordering point and its bootstrap interval.
    ax_e.legend(handles=handles, loc="lower center", bbox_to_anchor=(0.5, 1.04),
                ncol=3, columnspacing=0.8, handletextpad=0.15,
                borderaxespad=0.0)

    panel_label(fig, ax_a, "a", dx=-0.047)
    panel_label(fig, ax_b, "b", dx=-0.043)
    panel_label(fig, ax_c, "c", dx=-0.047)
    panel_label(fig, ax_d, "d", dx=-0.043)
    panel_label(fig, ax_e, "e", dx=-0.172)

    metadata = {"Creator": "matplotlib", "Title": "Temporal-support sensitivity"}
    fig.savefig(HERE / "fig_s2_temporal_criteria.pdf", metadata=metadata)
    fig.savefig(HERE / "fig_s2_temporal_criteria.svg", metadata=metadata)
    # 602 dpi avoids a sub-600 effective DPI after mm-to-pixel rounding.
    fig.savefig(HERE / "fig_s2_temporal_criteria.png", dpi=602,
                metadata={"Software": "matplotlib"})
    plt.close(fig)


def main():
    print("1) load and fit the primary 1-km monthly and daily records ...", flush=True)
    monthly, site, daily_event, reproduction, daily_diagnostics = load_primary()
    print(
        f"   {len(monthly):,} paired monthly rows; {len(site):,} sites; "
        f"{site.reach_id.nunique():,} Reaches",
        flush=True,
    )
    print(
        "   archive reproduction passed: "
        f"monthly n={reproduction['monthly_A_tau']['overlap_sites']:,}, "
        f"max |dA|={reproduction['monthly_A_tau']['maximum_absolute_A_difference']:.2e}, "
        "max |dtau|="
        f"{reproduction['monthly_A_tau']['maximum_circular_tau_difference_days']:.2e} d; "
        f"daily residual correlation n={reproduction['daily_event_r_main_gate']['overlap_sites']:,}, "
        "max |dr|="
        f"{reproduction['daily_event_r_main_gate']['maximum_absolute_event_r_difference']:.2e}",
        flush=True,
    )

    print("2) summarize temporal criteria with Reach bootstrap CIs ...", flush=True)
    primary, reaches = summarize_primary(site, daily_event)
    for name in CRITERIA:
        s = primary[name]
        print(
            f"   {name}: monthly {s['n_site']:,} sites/"
            f"{s['n_reach']:,} Reaches; daily residual correlation r "
            f"{s['denominators']['event_r_sites']:,} sites/"
            f"{s['denominators']['event_r_reaches']:,} Reaches; "
            f"event>=0.3={s['event_r_fraction_ge_0p3'][1]:.3f}, "
            f"amp/lead={s['amplified_or_leading_fraction'][1]:.3f}, "
            f"low-pass={s['lowpass_fraction_TF2_lt_TF1'][1]:.3f}",
            flush=True,
        )

    print("3) recompute the temporal x distance robustness matrix ...", flush=True)
    matrix, matrix_n, distance_detail = distance_matrix()
    thresholds = np.array(sorted(int(v) for v in distance_detail), dtype=int)

    numerical = {
        "all_screened_sites": int(len(site)),
        "full_rank_sites": int(site["rank"].eq(6).sum()),
        "condition_number_le_1e4_sites": int(site.fit_valid.sum()),
        "condition_number_quantiles": {
            str(q): float(site.condition_number.quantile(q))
            for q in (0.5, 0.9, 0.95, 0.99)
        },
    }
    output = {
        "figure": "fig_s2_temporal_criteria",
        "purpose": (
            "Eligibility sensitivity for the six-parameter intercept + trend + "
            "annual + semiannual harmonic model; accepted records are never truncated."
        ),
        "inputs": {
            "primary_monthly": str(MONTHLY),
            "primary_daily": str(DAILY),
            "archived_pair_metrics": str(PAIR_TABLE),
            "archived_event_metrics_for_main_gate_reproduction": str(EVENT_TABLE),
            "strict_distance_rematch_cache": str(DISTANCE_MONTHLY),
        },
        "model": {
            "n_parameters": 6,
            "terms": ["intercept", "linear trend", "annual cosine", "annual sine",
                      "semiannual cosine", "semiannual sine"],
            "all_available_observations_used_after_screen": True,
            "full_rank_required": True,
            "maximum_condition_number": CONDITION_MAX,
            "daily_event_r_minimum_paired_days": EVENT_MIN_PAIRED_DAYS,
            "daily_event_r_definition": (
                "Pearson correlation of river and groundwater residuals after "
                "separate six-parameter harmonic fits"
            ),
            "variable_specific_populations": (
                "Daily residual correlation r is selected and reach-aggregated solely under "
                "the daily support gate. A/tau and TF use the corresponding "
                "monthly support gate; the populations are not intersected."
            ),
        },
        "bootstrap": {"unit": "Reach", "draws": N_BOOT, "interval": "percentile 95%"},
        "primary_1km": primary,
        "numerical_diagnostics": numerical,
        "daily_event_diagnostics": daily_diagnostics,
        "archived_reproduction": reproduction,
        "distance_by_temporal_criterion": distance_detail,
        "joint_matrix_colormap": MATRIX_CMAP_NAME,
        "joint_matrix_consumer": "fig_s8_at_sensitivity panel b",
    }
    stats_path = HERE / "fig_s2_temporal_criteria_stats.json"
    stats_path.write_text(
        json.dumps(json_ready(output), indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )

    print("4) render vector and 600-dpi outputs ...", flush=True)
    make_figure(site, daily_event, primary)
    print(
        "   saved fig_s2_temporal_criteria.pdf/.svg/.png and "
        "fig_s2_temporal_criteria_stats.json",
        flush=True,
    )


if __name__ == "__main__":
    main()
