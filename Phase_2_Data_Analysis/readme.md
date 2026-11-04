# Phase 2: coupling metrics, uncertainty and robustness analyses

This phase turns the final well–river daily and monthly tables into scientific metrics, uncertainty estimates, null-model comparisons and manuscript summary numbers. It contains **17 Python scripts/modules (00–16)**, **15 reference products for each of the 1-, 3- and 5-km experiments**, and an archived monthly rematch cache for independent 0.5/1/2/5-km sensitivity.

The 45 distance-specific products record the reference analyses. The input/output catalogue below identifies the products and their producing scripts.

Full-input validation on 4 November 2026 reran steps 01–08 for all three distances using newly generated Phase 1 products and rebuilt the sensitivity cache from the pre-ranking candidates. All 21 distance-specific Parquet tables and the shared rematch table were byte-identical to the references. Scientific values in the 21 JSON/YAML result files and the cache metadata were exactly equal; changed source paths and modification times record the new run. Reference execution manifests are represented by new reproduction receipts rather than copied as calculation results.

## Directory guide

| Location | Contents and purpose |
|---|---|
| [01_Code/](01_Code/) | Production runner 00, analysis steps 01–08, shared modules 09–15 and workspace runner 16 |
| `distance1000_data/` | Main 1-km reference analysis products |
| `distance3000_data/` | Separate 3-km experiment using that threshold's independently selected pairs |
| `distance5000_data/` | Separate 5-km experiment |
| `shared_cache/` | Monthly records from independent 0.5/1/2/5-km rematching, with cache metadata |

## Scientific quantities and populations

| Quantity | Meaning and population |
|---|---|
| Monthly annual amplitude ratio `R_A` | Groundwater annual harmonic amplitude divided by river annual harmonic amplitude; stored in the legacy column `A` |
| Annual phase lag `tau_days` | Circular groundwater-minus-river annual phase lag in days; positive values indicate a later groundwater maximum |
| Daily residual correlation `event_r` | Pearson correlation of separately fitted river and groundwater harmonic residuals on exactly the same finite daily dates |
| `raw_r_same_sample` | Raw river–groundwater correlation on that same daily sample |
| `seasonal_harmonic_r_same_sample` | Correlation of annual + semiannual fitted periodic components on that same daily sample |
| `TF1` and `TF2` | Annual and semiannual power gains: squared groundwater/river harmonic amplitude ratios |
| `sigma_swot` | Reach-level median of product-reported river water-surface-elevation uncertainty (`river_wse_u`); metres |
| `sigma_swot_emp` | Reach-level river WSE residual variability: median across valid Nodes of 1.4826 × MAD of residuals from each Node's fitted harmonic model; metres. Includes unresolved hydrologic variability and measurement error, not pure instrumental noise |
| `aridity` | Full 2023–2025 precipitation sum divided by the absolute actual-ET sum, using all 36 calendar months |

The common harmonic design contains six coefficients: intercept, centered linear trend, and sine/cosine pairs at annual and semiannual frequencies. The main monthly screen requires at least 18 pairwise-finite months, at least 1.5 years of elapsed span, full column rank and condition number at most 10,000. Accepted fits use all available finite observations; the screen does not truncate records to 18 months.

The daily residual analysis is screened independently: at least 24 paired daily observations, at least 1.5 years of elapsed span and the corresponding fit-identifiability checks. A well can be daily eligible without being monthly eligible, or the reverse. Figs. 1–3 use the independent daily population rather than an intersection with the monthly table.

Germany and China define the development group. The geographic testing group is stored as `holdout`. Group assignment is frozen at Reach level so a Reach is not split between development and testing. The source label `country` can identify a monitoring network such as USGS or NGWMN; it is not a standardized country code.

At Reach level, ratios and correlations use the aggregation specified by each analysis, usually medians over eligible wells; annual phases use circular aggregation. The S10 comparison first intersects finite annual gain and daily residual correlation at the well–Reach level and only then aggregates to Reaches.

## Execution order and commands

The normal sequence for each distance is:

~~~text
01 monthly harmonic metrics + initial summaries
  → 02 independent daily metrics + daily fields on the monthly table
  → 03 covariates and descriptive mechanism analyses
  → 04 uncertainty and robustness
  → 05 circular-shift and irregular-time OU nulls
  → 06 headline-number YAML
  → 07 annual/semiannual gain comparisons and uncertainty
  → 08 illustrative daily records
~~~

Scripts 09–15 are imported helpers. Script 16 runs selected analyses or figures in a separate workspace and sets their paths.

After completing Phase 1, run all analysis steps from the submission root, including independent-distance rematching:

~~~bash
python Phase_2_Data_Analysis/01_Code/16_reproduce.py phase2 --distances 1000 3000 5000 --steps 1 2 3 4 5 6 7 8 --phase1-data /path/to/phase1_work --rebuild-rematch --output ../GRL_analysis_run
~~~

The Phase 1 workspace uses `distance1000/`, `distance3000/` and `distance5000/`, each containing `5matched_results_daily.txt` and `6matched_results_monthly.txt`. The selected distances determine which input folders are used.

The `--rebuild-rematch` option also requires the climate-enriched pre-ranking candidate table, `3results_all_country_nodeid_distance{distance}_withwidth50_pre_et.txt`, in each of those three folders. It rebuilds or validates the sensitivity cache against the supplied Phase 1 candidates. Without this option, the runner uses the bundled reference rematch cache even when new daily/monthly tables are supplied. Continue to Phase 3 with the same `--output ../GRL_analysis_run` and `--phase1-data` values so the figures use these newly calculated results.

To recalculate the headline summary from the saved analysis products:

~~~bash
python Phase_2_Data_Analysis/01_Code/16_reproduce.py phase2 --steps 6 --distances 1000 3000 5000 --output ../GRL_summary_run
~~~

Selecting only some steps retains saved products for unselected dependencies. Use a new output directory when changing the input snapshot. Outputs, logs and the execution receipt go to the selected workspace. The shared rematch cache retains its threshold-specific selected series; rebuilding it from candidates is handled separately by module 15.

## What each code file does

### 00 — Full-layout multi-distance runner

[00_run_all_distances.py](01_Code/00_run_all_distances.py) is the original production orchestrator. It sets distance-specific paths and environment variables, runs steps 01–08 and records `pipeline_manifest.json` with settings and products. It expects the full upstream Phase 1 layout, including pre-ranking candidate tables.

For workspace-based execution, use [16_reproduce.py](01_Code/16_reproduce.py). Script 00 documents and runs the original full-layout multi-distance sequence.

### 01 — Monthly coupling and harmonic metrics

[01_coupling_metrics.py](01_Code/01_coupling_metrics.py) reads the final monthly paired table and daily observations needed for reported SWOT-uncertainty aggregation. It computes annual harmonic amplitudes, `R_A`, circular phase lag, seasonal and monthly-climatology-adjusted correlations, seasonal commonality diagnostics and full-window aridity.

It assigns the monthly population's Reach-level development/testing split and writes `08_global_coupling_metrics.parquet`, `08_summary.json` and `08_holdout.json`. Later steps add daily diagnostics to this monthly-eligible table; they do not redefine it as the complete daily population.

### 02 — Independent daily residual correlations

[02_event_scale_daily.py](01_Code/02_event_scale_daily.py) reads the final daily table in chunks. For each eligible well–Reach pair, it computes raw, fitted-periodic and residual correlations on one common set of finite paired dates, with separate harmonic fits for river and groundwater.

It also deduplicates river observations across wells sharing them, estimates Node residual variability and aggregates that diagnostic to Reaches. Outputs are standalone `08_event_metrics_all_sites.parquet`, daily additions to `08_global_coupling_metrics.parquet` and updated `08_summary.json`. The standalone table supplies the map, regional distributions and null-model population.

### 03 — Mechanism covariates and regional descriptions

[03_mechanism_covariates.py](01_Code/03_mechanism_covariates.py) combines monthly records and both metric populations with groundwater depth, prior-bank distance, Reach slope, prior width variance, river regime and recharge–river seasonal phase offset. It keeps the daily-event covariate table separate from the monthly mechanism table.

It writes `08_mechanism.parquet`, `08_event_covariates_all_sites.parquet` and `08_mechanism_summary.json`. The summary includes regional effects, permutation/sign tests, Spearman associations and exploratory regional prediction diagnostics. These are descriptive associations, not estimates of hydraulic flux direction or causal effects. The archive runner uses the existing complete-case policy for unresolved groundwater depth.

### 04 — Uncertainty and robustness

[04_robustness_stats.py](01_Code/04_robustness_stats.py) reads finalized monthly and daily metrics, paired observations and threshold-specific monthly rematch records. It calculates Reach-bootstrap intervals, synchronized same-date correlation comparisons, single-Reach amplitude/phase uncertainty, physical-threshold grids, distance sensitivity, deseasonalization sensitivity and a synthetic phase-sign check.

It writes `08_reach_at_uncertainty.parquet` and `08_robustness.json`, which step 05 extends. Reach-specific amplitude/phase draws use actual calendar blocks through modules 12 and 13. The 0.5/1/2/5-km analysis reranks candidates separately inside each threshold when its cache is built; it is not a post-filter of primary 1-km pairs.

### 05 — Autocorrelation-preserving null models

[05_event_permutation_null.py](01_Code/05_event_permutation_null.py) freezes the complete daily-eligible well/Reach population and split from step 02, reads its daily observations and compares observed correlations with two nulls.

The first shifts groundwater residuals in observation order relative to river residuals. The second fits irregular-time Ornstein–Uhlenbeck correlation scales and generates independent river and groundwater surrogates at the actual observation dates. Those surrogates pass through the same six-column harmonic residualization used for observations before standardization and correlation.

The script appends observed, shift-null and OU-null summaries to `event_permutation_null` in `08_robustness.json`. Observed Reach-bootstrap confidence intervals and null-distribution intervals answer different questions.

### 06 — Traceable headline numbers

[06_main_numbers.py](01_Code/06_main_numbers.py) reads finalized tables and summaries from steps 01–05 and writes `08_main_numbers.yml`. It collects dataset counts, correlations, physical-screen summaries, uncertainty intervals and selected associations, with denominator definitions.

This step summarizes saved results; it does not refit harmonics or rerun null simulations.

### 07 — Frequency-response and gain comparisons

[07_tf_comparison.py](01_Code/07_tf_comparison.py) refits annual and semiannual harmonics from the monthly table, verifies that annual gains reproduce archived monthly metrics and computes power gains at both frequencies. It evaluates agreement between annual-gain bands and `r ≥ 0.3` on the exact finite monthly-gain × daily-correlation well–Reach intersection.

The low-pass ordering test, `TF2 < TF1`, instead uses the complete eligible monthly TF population. Module 14 propagates four-amplitude uncertainty and constructs an equal-gain null. Outputs are `12_tf_comparison.json` and `13_tf_lowpass_uncertainty.json/.parquet`, supporting S9 and S10. A power-gain band is the square of its amplitude-ratio band; TF1 from 0.25 to 4 corresponds to annual amplitude ratio from 0.5 to 2.

### 08 — Freeze the illustrative daily series

[08_extract_example_series.py](01_Code/08_extract_example_series.py) extracts the fixed illustrative wells `Germany_Bavaria+7785` and `France+BSS002DDJL` from the final daily table. It retains groundwater-only days as well as sparse SWOT observations and writes `12_fig2_example_series.parquet`.

The filename is historical: this table now supplies **Fig. 1a,b**, not the Fig. 2 map. These are selected examples, not an additional population-wide test.

### 09 — Analysis paths and shared eligibility settings

[09_pipeline_config.py](01_Code/09_pipeline_config.py) supplies daily/monthly input paths, output folders, rematching resources, distance rules and harmonic-eligibility settings. The workspace runner sets the relevant `GRL_*` environment variables for each workspace and distance.

The runner clears inherited `GRL_*` settings except the font preference before applying its own configuration. Controlled method changes require deliberate configuration/runner changes; a stale shell variable should not silently alter review reproduction.

### 10 — Shared harmonic model

[10_harmonic_model.py](01_Code/10_harmonic_model.py) builds the intercept/trend/annual/semiannual design, screens finite observations and matrix identifiability, fits the model and exposes amplitudes, phases, fitted values and residuals. It also residualizes simulated draws through the observed design matrix, aligning observation and null processing.

### 11 — Metadata aggregation across read chunks

[11_event_site_metadata.py](01_Code/11_event_site_metadata.py) accumulates daily-event site metadata across all paired rows, including records split across input chunks. It prevents a site's metadata summary from depending on a chunk boundary. It creates no independent output file.

### 12 — Actual-calendar bootstrap weights

[12_calendar_bootstrap.py](01_Code/12_calendar_bootstrap.py) provides stable seeds and randomized-origin, non-wrapping three-calendar-month block weights. Missing months remain missing. All site series within a Reach use the same Reach/calendar weight plan, preserving shared river-side perturbations instead of assigning unrelated weights to wells sharing a Node.

These blocks span calendar time, not three consecutive available observations. This module supplies weights; the amplitude/phase and TF modules perform refits and site resampling.

### 13 — Circular phase and amplitude/phase uncertainty

[13_phase_statistics.py](01_Code/13_phase_statistics.py) wraps phases, converts radians/days, computes circular Reach means and builds Reach-specific `R_A`/phase uncertainty. It uses leverage-adjusted paired residuals, shared calendar weights, harmonic refits and within-Reach site resampling. Circular intervals are unwrapped locally around the Reach point phase.

Its results feed `08_reach_at_uncertainty.parquet` and the corresponding robustness summaries. Ordinary linear medians are inappropriate for annual phases near the −π/+π boundary.

### 14 — Transfer-function uncertainty

[14_tf_uncertainty.py](01_Code/14_tf_uncertainty.py) propagates annual and semiannual amplitudes through the actual-date, shared-Reach block framework. It estimates ordering probabilities, robust/uncertain classifications, Reach-level intervals and signal-to-noise sensitivity.

Its equal-gain null constrains groundwater/river gain to be equal at the two frequencies while retaining the observed calendar, fitted phases, baseline/trend and paired residual structure. It is different from the independent-series OU null in step 05.

### 15 — Independent distance rematching and cache loading

[15_distance_robustness_rematch.py](01_Code/15_distance_robustness_rematch.py) selects the top Reach/Node only after applying each 0.5/1/2/5-km threshold. It reconstructs the retained groundwater calendar, unique daily values and monthly means following Phase 1.

For a new rematch calculation, it streams pre-ranking candidates and builds the shared cache. By default, the workspace runner uses `shared_cache/rematch_input.json` to verify and load the saved cache; manifest paths are relative to the manifest directory. The `phase2 --rebuild-rematch` option switches to the pre-ranking candidates in the supplied Phase 1 workspace. Supplying new daily/monthly tables without that option does not update the threshold-specific selections in the cache.

### 16 — Workspace runner for analysis and figures

[16_reproduce.py](01_Code/16_reproduce.py) provides the `phase2` and `figures` commands. It copies the analysis/figure products into a separate working directory, links the configured Phase 1 input directory when supplied, sets the analysis paths, map-resource path and verified rematch-cache manifest, then runs the requested steps.

It records per-step logs and a JSON receipt. A content fingerprint identifies the copied code and products so an existing workspace is not silently reused with a different package snapshot. The default output is `GRL_reproduced/` beside the submission directory. This runner controls execution and does not change scientific calculations.

## Reference product catalogue

Each `distance*_data/` directory contains the following 15 files. Prefixes such as `08_` and `12_` are retained output names, not execution-step numbers.

| File | Contents / row unit | Produced or updated by |
|---|---|---|
| `08_global_coupling_metrics.parquet` | Monthly-eligible well–Reach metrics, enriched with daily diagnostics where available | 01, 02 |
| `08_summary.json` | Dataset, monthly and daily summaries | 01, 02 |
| `08_holdout.json` | Monthly analysis development/geographic-testing split summary | 01 |
| `08_event_metrics_all_sites.parquet` | Complete independently daily-eligible well–Reach population | 02 |
| `08_mechanism.parquet` | Monthly metric population with mechanism covariates | 03 |
| `08_event_covariates_all_sites.parquet` | Independent daily population with static descriptors | 03 |
| `08_mechanism_summary.json` | Descriptive regional and covariate results | 03 |
| `08_reach_at_uncertainty.parquet` | Monthly-analysis Reach amplitude-ratio and circular-phase uncertainty | 04 via 13 |
| `08_robustness.json` | Bootstrap, threshold, distance, deseasonalization and null summaries | 04, 05 |
| `08_main_numbers.yml` | Headline values, denominators and intervals | 06 |
| `12_tf_comparison.json` | Gain-band agreement and frequency ordering | 07 |
| `13_tf_lowpass_uncertainty.parquet` | Reach low-pass probabilities, uncertainty and SNR diagnostics | 07 via 14 |
| `13_tf_lowpass_uncertainty.json` | Group propagated fractions, equal-gain null and SNR summaries | 07 via 14 |
| `12_fig2_example_series.parquet` | Two illustrative daily well–river records used in Fig. 1 | 08 |
| `pipeline_manifest.json` | Original settings, input references and product inventory | Production runner 00 |

The workspace runner writes a separate `phase2_reproduction_receipt.json` in the workspace. Original manifests copied with reference products describe their source run; they are not completion receipts for a new partial reproduction.

## Which products support which figures?

| Product group | Main figure uses |
|---|---|
| Independent daily metrics and example series | Figs. 1–3 and S1; daily eligibility check in S2; daily intersection for S10 |
| Robustness and headline summaries | Fig. 1c, S1, S3, S5, S7 and S8 |
| Daily covariates | S4 prediction and association panels |
| Monthly metrics and Reach amplitude/phase uncertainty | S2 validation, S5, S7, S9 and S10 |
| Independent-distance monthly cache | S2 temporal-by-distance calculations, subsequently displayed in S8b |
| TF comparisons and uncertainty | S9 and S10 |

The [Phase 3 guide](../Phase_3_Data_Visualization/readme.md) describes every panel and renderer. Some renderers compute statistics themselves; S4 fits the specified prediction model and runs its Reach bootstraps during rendering.

## Reproducibility and interpretation

Bootstrap and null routines use 10,000 draws in the archived main settings, with task-specific fixed seeds. Equal draw counts do not imply identical resampling units: Reach bootstraps, calendar residual bootstraps, within-Reach site resampling and surrogate nulls have different purposes.

Parquet is the authoritative saved tabular format. Many JSON/YAML interval triples are `[lower 2.5%, point estimate, upper 97.5%]`; read the named statistic to distinguish an observed-statistic CI, an uncertainty-propagated interval and a null-distribution range.

The `r ≥ 0.3` threshold is an effect-size screen, not a per-Reach significance test or proof of hydraulic exchange. Failure of a simplified damped-and-delayed response signature does not identify an alternative causal mechanism.

Historical source paths in result metadata document the original run; the workspace runner sets the paths for a new execution. The field reference below defines the main saved quantities.

## Analysis field reference

| Field | Meaning / units |
|---|---|
| `country` | Monitoring network/country label derived from the well identifier; USGS and NGWMN are separate labels |
| `split`, `is_dev` | Frozen analysis grouping: Germany/China development; other included networks geographic testing (`holdout`) |
| `ampR`, `ampG` | Annual river/groundwater harmonic amplitudes, metres |
| `A` | Legacy internal name for annual amplitude ratio R_A = ampG/ampR; dimensionless, not an amplitude in metres |
| `tau_days` / plotting `tau` | Annual circular phase lag, days; positive means the groundwater maximum occurs later |
| `n_mon` | Number of pairwise-finite monthly observations used for the accepted fit |
| `event_r` | Daily Pearson correlation after separate intercept, trend, annual and semiannual harmonic fits; dimensionless |
| `raw_r_same_sample`, `seasonal_harmonic_r_same_sample` | Raw and fitted-periodic correlations evaluated on exactly the same daily dates as `event_r` |
| `n_obs`, `span_years_same_sample` | Number and elapsed span of the accepted same-day paired record |
| `harmonic_rank_same_sample`, `condition_number_same_sample` | Design-matrix diagnostics for the daily harmonic fit |
| `*_amplitude_m_same_sample`, `*_sd_m_same_sample` | Harmonic amplitudes or standard deviations from the common daily sample, metres |
| `aridity` | Full 36-month precipitation sum divided by absolute actual-ET sum; dimensionless |
| `aridity_*_sum_mm`, `aridity_n_months`, `aridity_*_month` | Climate totals and support used in that ratio |
| `depth_m` | Derived mechanism-analysis depth covariate, metres |
| `sigma_swot` | Reach-level median of product-reported river water-surface-elevation uncertainty (`river_wse_u`); metres |
| `sigma_swot_emp` | Reach-level river WSE residual variability: median across valid Nodes of 1.4826 × MAD of residuals from each Node's fitted harmonic model; metres. Includes unresolved hydrologic variability and measurement error, not pure instrumental noise |
| `TF1`, `TF2` | Annual/semiannual power gains: squared groundwater/river amplitude ratio at the respective frequency |
| `A_ci_low`, `A_ci_high`, `tau_ci_*` | Reach-level paired-calendar-bootstrap intervals; ratio or days as appropriate |
| `phase_concentration_R` | Circular phase concentration; dimensionless |
| `p_amplified_or_lead`, `p_*` | Bootstrap classification probabilities; inspect the corresponding JSON definition for the specific event |
| `threshold_m` | Bank-distance threshold for the independently rematched monthly cache, metres |

Monthly harmonic eligibility requires at least 18 paired months, a span of at least 1.5 years, full rank for the six-parameter model, and condition number at most 10,000. Daily residual-correlation eligibility is assessed independently, with at least 24 paired days and a span of at least 1.5 years. Temporal-sensitivity definitions are documented in the S2 script.

`08_global_coupling_metrics.parquet` is a monthly-eligible well table; `08_event_metrics_all_sites.parquet` is an independently daily-eligible well table. `08_reach_at_uncertainty.parquet` and `13_tf_lowpass_uncertainty.parquet` contain Reach-level uncertainty results. Reach summaries are computed using the metric-specific aggregation in the supplied code; phase lags use circular methods rather than ordinary medians.

JSON bootstrap interval lists are generally `[lower 2.5%, point estimate, upper 97.5%]`; inspect the named section for whether the center is a fixed estimate or uncertainty-propagated summary. Null-model intervals are not automatically confidence intervals. Seeds and method details are retained in the code and JSON metadata.

## Rebuilding the sensitivity cache in a complete run

For a fresh raw-input reproduction, use the complete-run command above with `--rebuild-rematch` and point `--phase1-data` at the newly generated Phase 1 workspace. This selects its pre-ranking candidate tables and disables the archived-cache shortcut. Cache signatures are checked against these candidate sources. Run all eight analysis steps and then render figures in that same workspace.
