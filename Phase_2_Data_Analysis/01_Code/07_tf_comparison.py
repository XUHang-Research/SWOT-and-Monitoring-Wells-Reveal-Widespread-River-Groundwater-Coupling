#!/usr/bin/env python3
"""Step 7 — transfer function benchmarked against the co-variability metric.

TF_f = (A_gw,f / A_riv,f)^2 at the annual and semiannual bands (the only two
bands the SWOT cadence resolves), the energy-ratio analogue of Clyne & Sawyer
(2022) Eq. 1 (ESD_gw/ESD_sw). Cross-tabulated against the r_sub >= 0.3
co-variability criterion (agreement, Cohen's kappa), plus the low-pass test
(TF_semiannual < TF_annual) that Clyne & Sawyer could not settle with n = 17.

Energy convention: for a fitted harmonic the spectral energy at f is
proportional to c_f^2 + s_f^2 = A_f^2, so a TF band of [0.5, 2] is an
AMPLITUDE-ratio band of [0.707, 1.414]; every band is reported with its
amplitude equivalent so the two are never confused.

The script re-fits the harmonics from the raw monthly table and BLOCKS unless
the refit reproduces the archived annual gains exactly. It also enforces
self-contained site/Reach population contracts for the complete daily event-r
table, the monthly-TF x daily-event A1 intersection, and the full monthly-TF A2
population. Test A1 fixes finite TF1 x finite event_r at *site* level and only
then takes Reach medians. Test A2 is a standalone transfer-function test and
therefore retains every eligible TF site, whether or not event_r is available.

All intervals are reported as [lo95, value, hi95] (reach-level bootstrap,
n = 10000), matching 08_main_numbers.yml.

Inputs : raw/6matched_results_monthly.txt
         output/08_global_coupling_metrics.parquet (final table from step 2)
         output/08_event_metrics_all_sites.parquet (independent daily table)
Outputs: output/12_tf_comparison.json

Requires: numpy, pandas, pyarrow, scipy.
"""
import json
import sys

import numpy as np
import pandas as pd
from scipy import stats as sps

from importlib import import_module as _import_module
_harmonic_model = _import_module("10_harmonic_model")
fit_harmonic = _harmonic_model.fit_harmonic
harmonic_criterion = _harmonic_model.harmonic_criterion
_pipeline_config = _import_module("09_pipeline_config")
DISTANCE_COLUMN = _pipeline_config.DISTANCE_COLUMN
MAX_DISTANCE_M = _pipeline_config.MAX_DISTANCE_M
MON = _pipeline_config.MON
OUT = _pipeline_config.OUT
_tf_uncertainty = _import_module("14_tf_uncertainty")
DEFAULT_BLOCK_MONTHS = _tf_uncertainty.DEFAULT_BLOCK_MONTHS
TF_UNCERTAINTY_BOOTSTRAPS = _tf_uncertainty.DEFAULT_BOOTSTRAPS
TF_UNCERTAINTY_SEED = _tf_uncertainty.DEFAULT_SEED
build_tf_uncertainty = _tf_uncertainty.build_tf_uncertainty
summarize_tf_uncertainty = _tf_uncertainty.summarize_tf_uncertainty

NBOOT = 10_000
DEV_COUNTRIES = {"China", "Germany"}
TOL = 1e-6
TF_RTOL = 1e-10
# Counts belong to an input version. Freeze each run's denominators from the
# independent upstream products, while retaining all exact-key/refit checks.
POPULATION_CONTRACTS = {distance: {} for distance in (1000, 3000, 5000)}
DISTANCE_CONTRACT_M = int(round(MAX_DISTANCE_M))
if DISTANCE_CONTRACT_M not in POPULATION_CONTRACTS:
    raise ValueError(
        f"no TF/event population contract for {MAX_DISTANCE_M:g} m"
    )
POPULATION_CONTRACT = POPULATION_CONTRACTS[DISTANCE_CONTRACT_M]


def die(msg):
    """Fail loudly (not an assert: asserts vanish under python -O)."""
    sys.exit(f"FATAL: {msg}")


def enforce_population_contract(name, table):
    """Require this run's unique-site and unique-Reach denominator."""
    expected = POPULATION_CONTRACT[name]
    observed = {
        "n_site": int(table.site_no.nunique()),
        "n_reach": int(table.reach_id.nunique()),
    }
    if observed != expected:
        die(
            f"{name} population changed: observed "
            f"{observed['n_site']} sites/{observed['n_reach']} Reaches, "
            f"expected {expected['n_site']} sites/{expected['n_reach']} Reaches"
        )
    return observed


# ----------------------------------------------------------------- helpers
def harm2(t, x):
    """Harmonic fit retaining BOTH bands.

    Input : t (float array, decimal years), x (float array, series values)
    Output: dict(A1, phi1, A2, phi2, n) or None if under-determined.
    Design matrix identical to the harm() fit of step 1.
    """
    result = fit_harmonic(t, x)
    if result is None:
        return None
    return dict(A1=result['amplitudes'][0], phi1=result['phases'][0],
                A2=result['amplitudes'][1], phi2=result['phases'][1],
                n=result['n'])


def yfrac(d):
    """Datetime index -> decimal years (identical to step 1)."""
    d = pd.DatetimeIndex(d)
    return d.year.values + (d.dayofyear.values - 1) / 365.25


def _boot_idx(n, seed):
    rng = np.random.default_rng(seed)
    return rng.integers(0, n, (NBOOT, n))


def ci_frac(mask, seed=0):
    """Reach-level bootstrap of a fraction. Input: bool array. Output: [lo95, value, hi95]."""
    x = np.asarray(mask, float)
    x = x[np.isfinite(x)]
    if len(x) == 0:
        return [float("nan")] * 3
    d = x[_boot_idx(len(x), seed)].mean(1)
    return [float(np.percentile(d, 2.5)), float(x.mean()), float(np.percentile(d, 97.5))]


def ci_stat(x, fn=np.median, seed=0):
    """Reach-level bootstrap of a statistic. Output: [lo95, value, hi95]."""
    x = np.asarray(x, float)
    x = x[np.isfinite(x)]
    if len(x) == 0:
        return [float("nan")] * 3
    d = fn(x[_boot_idx(len(x), seed)], axis=1)
    return [float(np.percentile(d, 2.5)), float(fn(x)), float(np.percentile(d, 97.5))]


def spearman(a, b):
    """Spearman rho + two-sided p over pairwise-finite entries. Output: (rho, p, n)."""
    a = np.asarray(a, float)
    b = np.asarray(b, float)
    m = np.isfinite(a) & np.isfinite(b)
    if m.sum() < 8:
        return float("nan"), float("nan"), int(m.sum())
    r, p = sps.spearmanr(a[m], b[m])
    return float(r), float(p), int(m.sum())


def crosstab_stats(conn_tf, conn_r):
    """Agreement and Cohen's kappa for two boolean classifications.

    Input : conn_tf, conn_r (bool arrays, same length). Output: dict of scalars.
    """
    a = float(np.sum(conn_tf & conn_r))
    b = float(np.sum(conn_tf & ~conn_r))
    c = float(np.sum(~conn_tf & conn_r))
    d = float(np.sum(~conn_tf & ~conn_r))
    n = a + b + c + d
    if n == 0:
        return dict(agreement=float("nan"), kappa=float("nan"))
    po = (a + d) / n
    pe = ((a + b) * (a + c) + (c + d) * (b + d)) / n ** 2
    kappa = (po - pe) / (1 - pe) if pe < 1 else float("nan")
    return dict(agreement=po, kappa=float(kappa),
                both=a / n, tf_only=b / n, r_only=c / n, neither=d / n)


def ci_crosstab(conn_tf, conn_r, seed=0):
    """Bootstrap CIs for every cross-tab statistic. Output: dict key -> [lo, value, hi]."""
    conn_tf = np.asarray(conn_tf, bool)
    conn_r = np.asarray(conn_r, bool)
    n = len(conn_tf)
    idx = _boot_idx(n, seed)
    keys = ["agreement", "kappa", "both", "tf_only", "r_only", "neither"]
    draws = {k: np.empty(NBOOT) for k in keys}
    for i in range(NBOOT):
        s = crosstab_stats(conn_tf[idx[i]], conn_r[idx[i]])
        for k in keys:
            draws[k][i] = s.get(k, np.nan)
    obs = crosstab_stats(conn_tf, conn_r)
    return {k: [float(np.nanpercentile(draws[k], 2.5)), float(obs[k]),
                float(np.nanpercentile(draws[k], 97.5))] for k in keys}


# ----------------------------------------------------------------- 1. load
print("1) load archived reach metrics + monthly series ...")
S = pd.read_parquet(OUT / "08_global_coupling_metrics.parquet")
EVENT = pd.read_parquet(OUT / "08_event_metrics_all_sites.parquet")
POPULATION_CONTRACT['daily_event_full'] = dict(
    n_site=int(EVENT.site_no.nunique()), n_reach=int(EVENT.reach_id.nunique()))

# Restore-well-calendar changes may change monthly eligibility. Freeze this
# run's expected populations from the independent Step 1/2 products; require
# the independent TF refit to reproduce their exact identities below.
monthly_expected = S.loc[np.isfinite(S.A), ['site_no', 'reach_id']]
common_expected = monthly_expected.merge(
    EVENT.loc[np.isfinite(EVENT.event_r), ['site_no', 'reach_id']],
    on=['site_no', 'reach_id'], how='inner', validate='one_to_one')
for name, population in [('A2_full_monthly_TF', monthly_expected),
                         ('A1_monthly_TF_x_daily_event', common_expected)]:
    POPULATION_CONTRACT[name] = dict(n_site=int(population.site_no.nunique()),
                                     n_reach=int(population.reach_id.nunique()))

mon = pd.read_csv(MON, sep="\t", low_memory=False)
for c in ["river_wse", "Value_EGM2008_m", DISTANCE_COLUMN, "river_width", "number_match_SG"]:
    if c in mon:
        mon[c] = pd.to_numeric(mon[c], errors="coerce")
mon["date"] = pd.to_datetime(mon["date"], errors="coerce")
m = mon.dropna(subset=["river_wse", "Value_EGM2008_m", "date"]).copy()
m = m[m[DISTANCE_COLUMN] <= MAX_DISTANCE_M]
if "number_match_SG" in m:
    m = m[(m.number_match_SG == 1) | (m.number_match_SG.isna())]
print(f"   screened monthly rows {len(m)}, sites {m.site_no.nunique()}, reaches {m.reach_id.nunique()}")
del mon

# ----------------------------------------------------------------- 2. two-frequency harmonic fit
print("2) harmonic refit retaining annual + semiannual ...")
rows = []
for sid, g in m.groupby("site_no"):
    g = g.sort_values("date")
    t = yfrac(g.date)
    hr = harm2(t, g.river_wse.values)
    hg = harm2(t, g.Value_EGM2008_m.values)
    if hr is None or hg is None or hr["A1"] <= 0 or hr["A2"] <= 0:
        continue
    rows.append(dict(site_no=sid, reach_id=g.reach_id.iloc[0],
                     A1_riv=hr["A1"], A1_gw=hg["A1"], A2_riv=hr["A2"], A2_gw=hg["A2"],
                     TF1=(hg["A1"] / hr["A1"]) ** 2, TF2=(hg["A2"] / hr["A2"]) ** 2,
                     river_A2_A1=hr["A2"] / hr["A1"],
                     well_A2_A1=hg["A2"] / hg["A1"] if hg["A1"] > 0 else np.nan,
                     A_annual=hg["A1"] / hr["A1"]))
F = pd.DataFrame(rows)
print(f"   sites with both-band gains: {len(F)}")

# --- BLOCKING reproduction checks against the archived pair table
if S.site_no.duplicated().any():
    die(f"archived pair table has {int(S.site_no.duplicated().sum())} duplicate site_no rows")
if F.site_no.duplicated().any():
    die(f"harmonic refit has {int(F.site_no.duplicated().sum())} duplicate site_no rows")
required_event = {"site_no", "reach_id", "event_r", "country", "split"}
missing_event = sorted(required_event.difference(EVENT.columns))
if missing_event:
    die(f"independent daily table is missing fields: {missing_event}")
if EVENT.site_no.duplicated().any():
    die(
        "independent daily table has "
        f"{int(EVENT.site_no.duplicated().sum())} duplicate site_no rows"
    )
if not np.isfinite(EVENT.event_r).all():
    die("independent daily table contains non-finite event_r rows")
if not EVENT.split.isin(["dev", "holdout"]).all():
    die("independent daily table contains invalid split labels")
if not EVENT.groupby("reach_id").split.nunique(dropna=False).eq(1).all():
    die("at least one independent daily Reach crosses frozen splits")
daily_observed = enforce_population_contract("daily_event_full", EVENT)

chk = F.merge(
    S[["site_no", "A", "ampR", "ampG"]],
    on="site_no", how="inner", validate="one_to_one",
).dropna(
    subset=["A", "ampR", "ampG", "A_annual"])
if len(chk) < 100:
    die(f"only {len(chk)} sites overlap the archive; cannot verify reproduction")
d_A = float(np.abs(chk.A_annual - chk.A).max())
d_R = float(np.abs(chk.A1_riv - chk.ampR).max())
d_G = float(np.abs(chk.A1_gw - chk.ampG).max())
d_TF = float(np.abs(chk.TF1 - chk.A ** 2).max())
d_TF_rel = float((np.abs(chk.TF1 - chk.A ** 2) /
                  np.maximum(np.abs(chk.A ** 2), 1.0)).max())
print(f"   reproduction (n={len(chk)}): max|A_new-A_arch|={d_A:.2e} max|ampR|={d_R:.2e} "
      f"max|ampG|={d_G:.2e} max|TF1-A^2|={d_TF:.2e} max relative TF error={d_TF_rel:.2e}")
for name, val in [("annual gain A", d_A), ("river annual amplitude ampR", d_R),
                  ("well annual amplitude ampG", d_G)]:
    if not np.isfinite(val) or val > TOL:
        die(f"{name} does not reproduce the archive (max deviation {val:.3e} > {TOL:g})")
if not np.allclose(chk.TF1, chk.A ** 2, atol=TOL, rtol=TF_RTOL):
    die(f"TF1 vs A^2 identity does not reproduce the archive (max absolute deviation "
        f"{d_TF:.3e}; max relative deviation {d_TF_rel:.3e})")
print("   reproduction checks PASSED")

# ----------------------------------------------------------------- 3. define the two analysis populations, then aggregate
print("3) define site populations, then aggregate to reach level (median) ...")
# The full-TF population should reproduce the archived harmonic-eligible site
# population exactly.  This is a blocking check because otherwise the A2
# denominator could silently include a different analysis universe.
f_sites = set(F.site_no)
s_sites = set(S.site_no)
if f_sites != s_sites:
    die("TF refit and archived pair table have different site populations: "
        f"TF-only={len(f_sites - s_sites)}, archive-only={len(s_sites - f_sites)}")

# The site sets alone cannot detect a changed Reach assignment. Verify the
# complete pair key before either A1 or A2 is aggregated.
tf_monthly_key = F[["site_no", "reach_id"]].merge(
    S[["site_no", "reach_id"]],
    on="site_no", how="inner", validate="one_to_one",
    suffixes=("_TF", "_monthly"),
)
tf_monthly_reach_mismatch = tf_monthly_key.reach_id_TF.ne(
    tf_monthly_key.reach_id_monthly)
if tf_monthly_reach_mismatch.any():
    die(
        f"{int(tf_monthly_reach_mismatch.sum())} sites disagree on reach_id "
        "between the TF refit and archived monthly table"
    )
if set(F[['site_no','reach_id']].itertuples(index=False, name=None)) != set(
        monthly_expected.itertuples(index=False, name=None)):
    die('TF refit is not the exact finite current monthly population')
a2_observed = enforce_population_contract("A2_full_monthly_TF", F)

meta = (S.groupby("reach_id")
          .agg(width=("river_width", "median"),
               sigma_swot_emp=("sigma_swot_emp", "median"),
               country=("country", "first"), split_arch=("split", "first"))
          .reset_index())
tf_full = (F.groupby("reach_id")
             .agg(TF1=("TF1", "median"), TF2=("TF2", "median"),
                  A1_riv=("A1_riv", "median"), A2_riv=("A2_riv", "median"),
                  river_A2_A1=("river_A2_A1", "median"),
                  well_A2_A1=("well_A2_A1", "median"),
                  n_site_TF=("site_no", "size"))
             .reset_index())
R_a2 = meta.merge(tf_full, on="reach_id", how="inner", validate="one_to_one")
if len(R_a2) != a2_observed["n_reach"]:
    die(
        "A2 Reach aggregation changed the contracted denominator: "
        f"{len(R_a2)} != {a2_observed['n_reach']}"
    )

# A1 is the only direct monthly-versus-daily comparison. Build it explicitly
# from the monthly TF refit and the independent daily E table at site-Reach
# level; do not read the compatibility copy of event_r stored in S. Merging
# before Reach aggregation guarantees that a multi-well Reach's TF and event-r
# medians use exactly the same constituent wells.
tf_event_key = F[["site_no", "reach_id"]].merge(
    EVENT[["site_no", "reach_id"]],
    on="site_no", how="inner", validate="one_to_one",
    suffixes=("_TF", "_daily"),
)
tf_event_reach_mismatch = tf_event_key.reach_id_TF.ne(
    tf_event_key.reach_id_daily)
if tf_event_reach_mismatch.any():
    die(
        f"{int(tf_event_reach_mismatch.sum())} shared sites disagree on "
        "reach_id between monthly TF and independent daily tables"
    )

monthly_split = S[["site_no", "reach_id", "split"]].rename(
    columns={"split": "split_monthly"})
daily_event = EVENT[[
    "site_no", "reach_id", "event_r", "country", "split",
]].rename(columns={"split": "split_daily"})
a1_site = (
    F[["site_no", "reach_id", "TF1"]]
    .merge(
        monthly_split, on=["site_no", "reach_id"], how="inner",
        validate="one_to_one",
    )
    .merge(
        daily_event, on=["site_no", "reach_id"], how="inner",
        validate="one_to_one",
    )
)
a1_site = a1_site[
    np.isfinite(a1_site.TF1) & np.isfinite(a1_site.event_r)
].copy()
if a1_site.empty:
    die("finite monthly TF1 x finite independent daily event_r population is empty")
split_mismatch = a1_site.split_monthly.ne(a1_site.split_daily)
if split_mismatch.any():
    die(
        f"{int(split_mismatch.sum())} A1 sites disagree between the frozen "
        "monthly and daily split labels"
    )
a1_site["split"] = a1_site.split_daily
expected_a1_keys = set(
    F.loc[np.isfinite(F.TF1), ["site_no", "reach_id"]]
    .itertuples(index=False, name=None)
).intersection(
    set(
        EVENT.loc[np.isfinite(EVENT.event_r), ["site_no", "reach_id"]]
        .itertuples(index=False, name=None)
    )
)
actual_a1_keys = set(
    a1_site[["site_no", "reach_id"]].itertuples(index=False, name=None)
)
if actual_a1_keys != expected_a1_keys:
    die(
        "A1 is not the exact finite monthly-TF x daily-event "
        "(site_no, reach_id) intersection: "
        f"missing={len(expected_a1_keys - actual_a1_keys)}, "
        f"extra={len(actual_a1_keys - expected_a1_keys)}"
    )
a1_observed = enforce_population_contract(
    "A1_monthly_TF_x_daily_event", a1_site)
if (a1_site.groupby("reach_id")["split"].nunique(dropna=False) > 1).any():
    die("split is not constant within at least one A1 common-site Reach")
both = (a1_site.groupby("reach_id")
          .agg(TF1=("TF1", "median"), event_r=("event_r", "median"),
               country=("country", "first"), split=("split", "first"),
               n_site_common=("site_no", "size"))
          .reset_index())

# split: reproduce the reach-majority rule used in step 1 and verify it against
# the archived per-pair table; never silently overwrite it.
reach_dev = S.assign(is_dev=S.country.isin(DEV_COUNTRIES)).groupby("reach_id")["is_dev"].mean() >= 0.5
derived = np.where(R_a2.reach_id.map(reach_dev), "dev", "holdout")
known = R_a2.split_arch.notna()
mismatch = int((R_a2.split_arch[known].values != derived[known.values]).sum())
if mismatch:
    die(f"{mismatch} reaches disagree between the archived split and the country rule; "
        "resolve before proceeding (the holdout must not be redefined)")
R_a2["split"] = np.where(known, R_a2.split_arch, derived)
if len(both) != a1_observed["n_reach"]:
    die(
        "A1 Reach aggregation changed the contracted denominator: "
        f"{len(both)} != {a1_observed['n_reach']}"
    )
print(f"   A1 common population: {len(a1_site)} sites, {len(both)} reaches")
print(f"   A2 full-TF population: {len(F)} sites, "
      f"{int(R_a2.TF1.notna().sum())} reaches | split verified (0 mismatches)")

print(
    "   population contracts PASSED: "
    f"daily {daily_observed['n_site']}/{daily_observed['n_reach']}, "
    f"A1 {a1_observed['n_site']}/{a1_observed['n_reach']}, "
    f"A2 {a2_observed['n_site']}/{a2_observed['n_reach']}"
)

out_tf = {"provenance": {
    "generated_by": "07_tf_comparison.py",
    "tf_definition": "TF_f = (A_gw,f / A_riv,f)^2, the energy-ratio analogue of Clyne & Sawyer 2022 Eq.1 (ESD_gw/ESD_sw)",
    "bands": "annual f=1/yr and semiannual f=2/yr; the only bands SWOT cadence resolves",
    "interval_convention": "[lo95, value, hi95], reach-level bootstrap n=10000",
    "harmonic_eligibility": harmonic_criterion(),
    "reproduction": {"max_abs_dev_annual_gain": d_A,
                     "max_abs_dev_TF1_vs_A_squared": d_TF,
                     "max_relative_dev_TF1_vs_A_squared": d_TF_rel,
                     "tf_allclose_atol": TOL, "tf_allclose_rtol": TF_RTOL},
    "population_contract": {
        "distance_m": DISTANCE_CONTRACT_M,
        "expected": POPULATION_CONTRACT,
        "observed": {
            "daily_event_full": daily_observed,
            "A1_monthly_TF_x_daily_event": a1_observed,
            "A2_full_monthly_TF": a2_observed,
        },
    },
    "denominator_policy": (
        "A1 alone uses the finite monthly TF1 x finite independent daily event_r "
        "site-level intersection before Reach medians; "
        "A2 uses the full harmonic-eligible TF population independently of event_r")},
    "denominators": {
        "A1_common_site": {
            "definition": (
                "finite monthly TF1 x finite independent daily event_r at site "
                "level, with identical reach_id required, then Reach median"),
            "monthly_source_table": "08_global_coupling_metrics.parquet",
            "daily_source_table": "08_event_metrics_all_sites.parquet",
            "intersection_keys": ["site_no", "reach_id"],
            "reach_id_integrity_check": "reach_id must agree before the (site_no, reach_id) merge",
            "site_filter": "finite TF1 and finite event_r",
            "reach_aggregation": "median after site-level intersection",
            "n_site": int(len(a1_site)), "n_reach": int(len(both)),
        },
        "A2_full_TF": {
            "definition": "all harmonic-eligible TF sites, then Reach median; event_r not required",
            "monthly_source_table": "08_global_coupling_metrics.parquet",
            "n_site": int(len(F)), "n_reach": int(R_a2.TF1.notna().sum()),
        },
    },
    # Retained for backward compatibility; the explicit denominator block above
    # should be used by new consumers.
    "n_reach_with_TF": int(R_a2.TF1.notna().sum())}

# ----------------------------------------------------------------- 4. Test A1
print("4) Test A1 - TF classification vs synchronous sub-seasonal r ...")
out_tf["A1_agreement"] = {
    "n_site": int(len(a1_site)), "n_reach": int(len(both)),
    "denominator": (
        "finite monthly TF1 x finite independent daily event_r sites, with "
        "identical reach_id required, followed by Reach medians"),
    "note":
    "TF bands are OUR operationalisation of Clyne's qualitative 'near 1' / 'approaching 0 or infinity'; "
    "each TF band is also given as its amplitude-ratio equivalent (sqrt of the TF bounds)"}
for k, (lo, hi) in enumerate([(0.5, 2.0), (1 / 3, 3.0), (0.25, 4.0)]):
    conn_tf = ((both.TF1 >= lo) & (both.TF1 <= hi)).values
    conn_r = (both.event_r >= 0.3).values
    key = f"TF_band_{lo:.3f}_{hi:.3f}"
    out_tf["A1_agreement"][key] = {
        "amplitude_ratio_equivalent": [float(np.sqrt(lo)), float(np.sqrt(hi))],
        "tf_connected_frac": ci_frac(conn_tf, seed=10 + k),
        "r_ge0p3_frac": ci_frac(conn_r, seed=20 + k),
        **{kk: vv for kk, vv in ci_crosstab(conn_tf, conn_r, seed=30 + k).items()},
        "median_event_r_where_TF_in_band_but_r_below_0p3":
            float(both.event_r[conn_tf & ~conn_r].median()) if (conn_tf & ~conn_r).any() else None,
        "median_TF1_where_r_ge_0p3_but_TF_outside_band":
            float(both.TF1[~conn_tf & conn_r].median()) if (~conn_tf & conn_r).any() else None}
    st = crosstab_stats(conn_tf, conn_r)
    print(f"   TF band [{lo:.2f},{hi:.2f}] (A in [{np.sqrt(lo):.2f},{np.sqrt(hi):.2f}]): "
          f"TF-conn {conn_tf.mean():.3f} vs r-detect {conn_r.mean():.3f} | "
          f"agreement {st['agreement']:.3f} kappa {st['kappa']:+.3f} | "
          f"TF-only {st['tf_only']:.3f} r-only {st['r_only']:.3f}")

rho, p, n = spearman(np.log10(both.TF1.clip(lower=1e-9)), both.event_r)
out_tf["A1_continuous"] = {"spearman_log10TF_annual_vs_covariability": {"rho": rho, "p": p, "n": n}}
print(f"   spearman(log10 TF_annual, r_sub) = {rho:+.3f} (p={p:.2g}, n={n})")

# ----------------------------------------------------------------- 5. Test A2
print("5) Test A2 - does TF decline from annual to semiannual (low-pass)? ...")
lp = R_a2.dropna(subset=["TF1", "TF2"]).copy()
out_tf["A2_lowpass"] = {"expectation":
    "low-pass filtering predicts TF_semiannual < TF_annual; Clyne & Sawyer (2022) reported this was "
    "'not consistently evident' in their transfer functions with n=17 sites",
    "interpretation_guide": "a fraction whose CI includes 0.5 means no systematic low-pass decline",
    "denominator": "full harmonic-eligible TF site population; event_r is not required",
    "n_site": int(len(F)), "n_reach": int(len(lp))}
for j, (lab, sub) in enumerate([("ALL", lp), ("DEV", lp[lp.split == "dev"]),
                                ("HOLDOUT", lp[lp.split == "holdout"])]):
    if len(sub) < 20:
        continue
    out_tf["A2_lowpass"][lab] = {
        "n_reach": int(len(sub)),
        "frac_TF2_lt_TF1": ci_frac((sub.TF2 < sub.TF1).values, seed=40 + j),
        "median_TF_annual": ci_stat(sub.TF1.values, seed=50 + j),
        "median_TF_semiannual": ci_stat(sub.TF2.values, seed=60 + j)}
    f_ = out_tf["A2_lowpass"][lab]["frac_TF2_lt_TF1"]
    print(f"   {lab:8s} n={len(sub):4d} frac(TF2<TF1)={f_[1]:.3f} [{f_[0]:.3f}-{f_[2]:.3f}] "
          f"| median TF1={sub.TF1.median():.3f} TF2={sub.TF2.median():.3f}")

# Frequency-response and measurement-noise checks cited in the manuscript.
# All calculations remain at Reach level: site ratios/amplitudes are first
# aggregated by the median, exactly as for TF1 and TF2 above.
paired = lp.dropna(subset=["well_A2_A1", "river_A2_A1"]).copy()
wilcoxon = sps.wilcoxon(
    paired.well_A2_A1, paired.river_A2_A1,
    alternative="less", zero_method="wilcox",
)
out_tf["A2_lowpass"]["frequency_response"] = {
    "n_reach": int(len(paired)),
    "ratio_definition": "semiannual amplitude / annual amplitude; site ratios aggregated to Reach medians",
    "median_well_A2_over_A1": ci_stat(paired.well_A2_A1.values, seed=70),
    "median_river_A2_over_A1": ci_stat(paired.river_A2_A1.values, seed=71),
    "median_paired_well_minus_river": ci_stat(
        (paired.well_A2_A1 - paired.river_A2_A1).values, seed=72),
    "fraction_well_ratio_lt_river_ratio": ci_frac(
        (paired.well_A2_A1 < paired.river_A2_A1).values, seed=73),
    "paired_wilcoxon_well_lt_river": {
        "statistic": float(wilcoxon.statistic), "p_value": float(wilcoxon.pvalue)},
}

noise = lp.dropna(subset=["sigma_swot_emp", "A1_riv", "A2_riv"]).copy()
noise = noise[noise.sigma_swot_emp > 0]
labels = ["low", "middle", "high"]
noise["noise_tertile"] = pd.qcut(noise.sigma_swot_emp, q=3, labels=labels)
tertiles = []
for j, label in enumerate(labels):
    sub = noise[noise.noise_tertile == label]
    tertiles.append({
        "tertile": label,
        "n_reach": int(len(sub)),
        "sigma_swot_emp_range_m": [float(sub.sigma_swot_emp.min()),
                                     float(sub.sigma_swot_emp.max())],
        "frac_TF2_lt_TF1": ci_frac((sub.TF2 < sub.TF1).values, seed=80 + j),
    })
noise["river_snr_annual"] = noise.A1_riv / noise.sigma_swot_emp
noise["river_snr_semiannual"] = noise.A2_riv / noise.sigma_swot_emp
snr2 = noise[(noise.river_snr_annual > 2) & (noise.river_snr_semiannual > 2)]
out_tf["A2_lowpass"]["noise_robustness"] = {
    "noise_definition": "per-Reach sub-seasonal river-WSE residual variability (legacy column sigma_swot_emp)",
    "noise_tertiles": tertiles,
    "river_signal_to_noise": {
        "n_reach": int(len(noise)),
        "median_annual": ci_stat(noise.river_snr_annual.values, seed=90),
        "median_semiannual": ci_stat(noise.river_snr_semiannual.values, seed=91),
    },
    "river_snr_gt_2_both_bands": {
        "threshold": 2.0, "n_reach": int(len(snr2)),
        "frac_TF2_lt_TF1": ci_frac((snr2.TF2 < snr2.TF1).values, seed=92),
    },
}
print(f"   frequency response: median well A2/A1={paired.well_A2_A1.median():.3f}, "
      f"river A2/A1={paired.river_A2_A1.median():.3f}")
print("   noise tertiles: " + ", ".join(
    f"{row['tertile']} n={row['n_reach']} low-pass={row['frac_TF2_lt_TF1'][1]:.3f}"
    for row in tertiles))
print(f"   river SNR>2 in both bands: n={len(snr2)}, "
      f"low-pass={(snr2.TF2 < snr2.TF1).mean():.3f}")

# ----------------------------------------------------------------- 6. uncertainty propagation and sampling-calendar null
print("6) propagate all four amplitude uncertainties at the actual monthly dates ...")
tf_site_uncertainty_population = F.merge(
    S[["site_no", "reach_id", "split"]],
    on=["site_no", "reach_id"], how="inner", validate="one_to_one",
)
tf_reach_uncertainty, n_uncertainty_sites = build_tf_uncertainty(
    monthly=m,
    pair_table=tf_site_uncertainty_population,
    n_boot=TF_UNCERTAINTY_BOOTSTRAPS,
    block_months=DEFAULT_BLOCK_MONTHS,
    seed=TF_UNCERTAINTY_SEED,
)
if n_uncertainty_sites != len(F):
    die(
        "TF uncertainty fit does not reproduce the full monthly site population: "
        f"{n_uncertainty_sites} != {len(F)}"
    )
if len(tf_reach_uncertainty) != len(R_a2):
    die(
        "TF uncertainty aggregation does not reproduce the full monthly Reach "
        f"population: {len(tf_reach_uncertainty)} != {len(R_a2)}"
    )
tf_uncertainty_summary = summarize_tf_uncertainty(
    tf_reach_uncertainty, seed=TF_UNCERTAINTY_SEED)
tf_uncertainty_output = {
    "provenance": {
        "generated_by": "07_tf_comparison.py via 14_tf_uncertainty.py",
        "monthly_source": str(MON),
        "point_estimate_source": "12_tf_comparison.json A2_lowpass",
        "n_site": int(n_uncertainty_sites),
        "n_reach": int(len(tf_reach_uncertainty)),
        "bootstrap_replicates_per_site_and_reach": int(TF_UNCERTAINTY_BOOTSTRAPS),
        "residual_block_months": int(DEFAULT_BLOCK_MONTHS),
        "residual_block_unit": "calendar_month",
        "temporal_dependence_unit": "Reach",
        "replicate_alignment": "shared calendar weights and synchronized replicate across sites/Nodes",
        "seed": int(TF_UNCERTAINTY_SEED),
        "sampling_calendar": "each site's exact pairwise-finite accepted monthly dates",
        "interpolation": False,
        "harmonic_eligibility": harmonic_criterion(),
    },
    "summary": tf_uncertainty_summary,
}
tf_reach_uncertainty.to_parquet(
    OUT / "13_tf_lowpass_uncertainty.parquet", index=False)
json.dump(
    tf_uncertainty_output,
    open(OUT / "13_tf_lowpass_uncertainty.json", "w"),
    indent=2,
    ensure_ascii=False,
)
all_uncertainty = tf_uncertainty_summary["ALL"]
print(
    "   ALL uncertainty-propagated low-pass fraction="
    f"{all_uncertainty['uncertainty_propagated_fraction_TF2_lt_TF1'][1]:.3f}; "
    "equal-gain null="
    f"{all_uncertainty['equal_gain_sampling_null_fraction'][1]:.3f}; "
    f"p={all_uncertainty['p_observed_point_vs_equal_gain_null']:.4f}"
)
print("   wrote output/13_tf_lowpass_uncertainty.json/.parquet")
del m

json.dump(out_tf, open(OUT / "12_tf_comparison.json", "w"), indent=2, ensure_ascii=False)
print("   wrote output/12_tf_comparison.json")
print("done")
