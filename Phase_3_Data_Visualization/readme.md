# Phase 3: figure-by-figure guide and rendering code

This directory contains the code and reference artwork for **Figs. 1–3 and S1–S10**: 13 figures, 28 artwork exports and the S2 statistics JSON. Scripts 01–12 are renderers; script 12 produces S9 and S10 together. Script 13 supplies the common typography and canvas settings.

Each figure has PDF and PNG versions. Fig. 3 and S2 also have SVG versions. The reference artwork is preserved, including the expanded S7a and S10 axes. This guide explains the scientific purpose, panels, inputs, calculations and outputs of every renderer.

Full-input validation on 4 November 2026 regenerated all 13 figures from freshly calculated Phase 1–2 outputs. All 13 PNGs were pixel-identical to the references, all 13 PDFs rendered identically at 150 dpi, and both SVG drawings were identical after excluding metadata and normalizing generated identifiers. The scientific values in the S2 statistics JSON were exactly equal. PDF font inspection confirmed embedded Arial, Arial Bold and Arial Italic fonts. This validation used Python 3.11.14, the versions in `requirements.txt`, Arial, and the existing Matplotlib `cividis` fallback for S2.

## Directory and input conventions

| Location | Purpose |
|---|---|
| [01_Code/](01_Code/) | Twelve rendering scripts and one common style module |
| `01_Code/colormaps/` | Bundled Scientific Colour Maps tables and their license |
| This directory's root | Reference artwork and `fig_s2_temporal_criteria_stats.json` |
| `../Phase_2_Data_Analysis/distance1000_data/` | Main figure-analysis tables and saved summary statistics |
| `../Phase_2_Data_Analysis/shared_cache/` | Independently rematched monthly records used in temporal–distance sensitivity |
| `02_Resources/natural_earth/` | Natural Earth public-domain basemaps for cartographic rendering |

Unless explicitly stated otherwise, the Phase 2 inputs below refer to the **1-km experiment**. The 3- and 5-km experiment folders are separate reference analyses, not automatically substituted when a figure is rendered.

The shorthand “daily population” means independently daily-eligible well–Reach pairs; “monthly population” means monthly-harmonic-eligible pairs. They are different populations. Plotted Reach counts are not monitoring-well counts. The field `A` is the annual amplitude ratio `R_A`, while individual harmonic amplitudes retain the symbol A and units of metres. `TF1` is annual power gain, the square of the annual amplitude ratio.

## Rendering commands

Use the shared workspace runner from the submission root. To render the results just calculated in Phase 2, reuse the same `--output` workspace and `--phase1-data` directory:

~~~bash
python Phase_2_Data_Analysis/01_Code/16_reproduce.py figures --phase1-data /path/to/phase1_work --output ../GRL_analysis_run
~~~

To select particular figures, add `--only` followed by their identifiers, for example:

~~~bash
python Phase_2_Data_Analysis/01_Code/16_reproduce.py figures --only 1 2 3 --phase1-data /path/to/phase1_work --output ../GRL_analysis_run
~~~

The `--phase1-data` argument identifies the Phase 1 output workspace described in its README. Individual figure inputs are listed below. The runner writes figures and logs to the separate output directory and records the selected renderers in `figures_reproduction_receipt.json`.

An existing workspace retains its computed Phase 2 products. A new workspace is initialized from the packaged reference products, so choosing a different output directory does not automatically carry over a preceding Phase 2 calculation.

S2 precedes S8 when both are selected, so S8 reads the updated temporal–distance statistics. S9 and S10 share a renderer: requesting either runs it once and produces both. Some renderers perform additional calculations, particularly S2, S4 and S9/S10; their roles are explained in the figure guide.

## Figure and script index

| Figure | Code | Main subject |
|---|---|---|
| 1 | [01_fig1_honest_metric.py](01_Code/01_fig1_honest_metric.py) | Paired examples and same-date correlation comparison |
| 2 | [02_fig2_hero_map.py](01_Code/02_fig2_hero_map.py) | Global and regional Reach-level residual-correlation maps |
| 3 | [03_fig3_regional.py](01_Code/03_fig3_regional.py) | Regional distributions and Reach-bootstrap median intervals |
| S1 | [04_fig_s1_composition.py](01_Code/04_fig_s1_composition.py) | Sample composition and observed/null threshold fractions |
| S2 | [05_fig_s2_temporal_criteria.py](01_Code/05_fig_s2_temporal_criteria.py) | Independent daily/monthly temporal-support sensitivity |
| S3 | [06_fig_s3_robustness.py](01_Code/06_fig_s3_robustness.py) | Shift/OU nulls and same-date correlations by group |
| S4 | [07_fig_s4_predict.py](01_Code/07_fig_s4_predict.py) | Geographic prediction and static-descriptor associations |
| S5 | [08_fig_s5_test_noise.py](01_Code/08_fig_s5_test_noise.py) | Study design, idealized response and river residual variability |
| S6 | [09_fig_s6_cycle_floor.py](01_Code/09_fig_s6_cycle_floor.py) | Schematic annual-phase precision versus record length |
| S7 | [10_fig_s7_signature.py](01_Code/10_fig_s7_signature.py) | Amplitude–phase signature, uncertainty and distance sensitivity |
| S8 | [11_fig_s8_at_sensitivity.py](01_Code/11_fig_s8_at_sensitivity.py) | Physical-threshold and temporal–distance sensitivity |
| S9 | [12_fig_s9_s10_criteria.py](01_Code/12_fig_s9_s10_criteria.py) | Frequency ordering, uncertainty and SNR |
| S10 | [12_fig_s9_s10_criteria.py](01_Code/12_fig_s9_s10_criteria.py) | Annual-gain versus residual-correlation criteria |

The following sections describe the purpose, panels and input/output files of each figure.

## Fig. 1 — Paired records and same-date correlations

**Code:** `01_fig1_honest_metric.py`. **Output:** `fig1_honest_metric.pdf/.png`.

The figure explains why a shared seasonal shape and a high residual correlation are different observations.

- **a:** Daily groundwater and sparse SWOT river levels for `Germany_Bavaria+7785`, the stronger residual-correlation example.
- **b:** The corresponding records for `France+BSS002DDJL`, an example of seasonal similarity with weaker residual agreement.
- **c:** Reach summaries of raw, fitted-periodic and residual correlations, using exactly the same daily dates and the complete daily population with all three correlations finite.

Panels a,b use two vertical axes with equal spans in metres, centered on their own records; their absolute vertical offsets should not be read as hydraulic-head differences. Panel c's three diagnostics are not additive variance components.

**Inputs:** `08_event_metrics_all_sites.parquet`, `08_robustness.json` and `12_fig2_example_series.parquet`. The latter's historical filename refers to Fig. 2, but its current consumer is Fig. 1. The renderer reads the frozen example series and existing same-sample bootstrap summaries; it does not reread the full daily observation table.

## Fig. 2 — Geographic distribution of residual correlation

**Code:** `02_fig2_hero_map.py`. **Output:** `fig2_hero_map.pdf/.png`.

Panel **a** is a global Robinson-projection map. Panels **b–e** show North America, Europe, China and Australia in a 2 × 2 regional arrangement, with a shared vertical color scale. Each marker represents one Reach after aggregation from daily-eligible wells. Headings report the displayed population size and median residual correlation.

Gray markers denote `r < 0.3`; sequential blues encode the higher residual correlations. Regional counts/medians use the displayed map windows, so their sum need not equal the global total. The threshold is an effect-size screen, not proof of hydraulic exchange, and marker color does not represent gaining/losing flux direction.

**Inputs:** `08_event_metrics_all_sites.parquet` and bundled Natural Earth land/coastline layers. The archive style setup points Cartopy to these local layers, avoiding a map download during rendering.

## Fig. 3 — Regional distributions

**Code:** `03_fig3_regional.py`. **Output:** `fig3_regional.pdf/.png/.svg`.

Symmetric violins and individual Reach points display regional residual-correlation distributions. Countries enter the comparison when they have at least 15 analyzed Reaches; USGS and NGWMN are combined as USA. Point colors follow residual correlation on the shared −1 to +1 `vik_r` scale. Symbols and capped intervals mark regional medians and 95% Reach-bootstrap intervals, with development/testing group distinctions.

Violin widths are normalized within each region; their area is not a comparison of regional sample sizes. The renderer recomputes the fixed-seed 10,000-draw median intervals from the saved daily metrics.

**Inputs:** `08_event_metrics_all_sites.parquet` and `01_Code/colormaps/vik.txt`. Monthly eligibility is not imposed.

## Fig. S1 — Sample composition and threshold fractions

**Code:** `04_fig_s1_composition.py`. **Output:** `fig_s1_composition.pdf/.png`.

- **a:** Number of independently daily-eligible Reaches by continent, describing the geographic balance of the sample.
- **b:** Observed fractions with `r ≥ 0.3` for all, development and geographic-testing Reaches, with Reach-bootstrap CIs and their population-specific circular-shift null comparisons.

Observed and null quantities use matching group definitions. A continent's number of sampled Reaches describes the monitoring sample, not global river coverage.

**Inputs:** `08_event_metrics_all_sites.parquet` and `08_main_numbers.yml`.

## Fig. S2 — Temporal-support sensitivity

**Code:** `05_fig_s2_temporal_criteria.py`. **Output:** `fig_s2_temporal_criteria.pdf/.png/.svg` and `fig_s2_temporal_criteria_stats.json`.

- **a:** Daily paired observation count versus elapsed span, classified by the strictest daily gate passed.
- **b:** Retained daily sites, Reaches and networks relative to the loose daily gate.
- **c:** Monthly paired observation count versus elapsed span, classified by the strictest monthly gate passed.
- **d:** Retained monthly sites, Reaches and networks relative to the loose monthly gate.
- **e:** Three conclusion-specific fractions with Reach-bootstrap intervals: daily `r ≥ 0.3`, monthly `R_A > 1.2 or tau ≤ 0`, and monthly `TF2 < TF1`.

| Gate | Monthly paired observations | Daily paired observations | Minimum elapsed span |
|---|---:|---:|---:|
| Loose | 12 | 24 | 1.0 year |
| Main | 18 | 24 | 1.5 years |
| Strict | 24 | 24 | 2.0 years |

All accepted fits must satisfy the model rank/conditioning checks. The gates select whole records; they do not truncate accepted time series. Daily and monthly populations remain independent. Panel e uses the appropriate denominator for each conclusion, rather than forcing the three conclusions onto a common intersection.

**Inputs:** Final 1-km daily/monthly tables; `08_global_coupling_metrics.parquet` and `08_event_metrics_all_sites.parquet` for reproduction checks; `shared_cache/distance_robustness_monthly_strict.parquet` for independent distance sensitivity. The renderer refits temporal-sensitivity metrics and validates the main gate against the reference metrics. Its JSON also stores the temporal-by-distance matrix displayed in S8b.

## Fig. S3 — Statistical validation

**Code:** `06_fig_s3_robustness.py`. **Output:** `fig_s3_robustness.pdf/.png`.

- **a:** All-Reach observed median residual correlation and fraction above `r ≥ 0.3`, compared with observation-order shift and irregular-time OU null summaries. The annotations report OU-null probabilities.
- **b:** Same-date raw, fitted-periodic and residual-correlation summaries with Reach-bootstrap intervals for all, development and geographic-testing Reaches.

The left panel uses the complete daily population, not only the geographic testing group. The right panel uses the complete daily population with all three correlations finite; no monthly-eligibility intersection is imposed. The renderer prints paired correlation contrasts for possible caption use without placing a paragraph of those contrasts inside the figure.

**Input:** `08_robustness.json`. The null simulations and uncertainty calculations are read from Phase 2 rather than rerun here.

## Fig. S4 — Transferability of static-descriptor prediction

**Code:** `07_fig_s4_predict.py`. **Output:** `fig_s4_predict.pdf/.png`.

- **a:** Predicted versus observed residual correlation in the geographic testing group. The five-descriptor linear model is trained on development Reaches; scaling parameters are estimated from development data only. Reported scores include out-of-sample R² and Spearman rho.
- **b:** Residual correlation versus depth below land surface, with individual Reaches and equal-count-bin median/IQR summaries.
- **c:** Residual correlation versus prior bank distance, using the same testing sample and descriptive bin summaries.
- **d:** Spearman associations for river width, Reach slope and prior width variance, with 95% Reach-bootstrap intervals.

All panels use the same complete-case testing Reaches after the constituent finite-event/depth population is fixed. The model uses groundwater depth below land surface, prior well-to-bank distance, and three river descriptors summarized at Reach level. Depth uses a symmetric-log axis so negative/artesian values remain visible. Bin IQRs describe spread, not confidence intervals for the trend.

**Input:** `08_event_covariates_all_sites.parquet`. Unlike a purely graphical renderer, this script fits the model and recomputes 10,000-draw prediction and association bootstraps. Its weak-prediction shape is retained; the renderer does not improve or regularize the fit for appearance.

## Fig. S5 — Study design and residual variability

**Code:** `08_fig_s5_test_noise.py`. **Output:** `fig_s5_test_noise.pdf/.png`.

- **a:** Geographic distribution of the monthly-analysis Reaches, distinguishing development and geographic-testing groups.
- **b:** An explicitly schematic damped-and-delayed groundwater response to sinusoidal river forcing under an idealized local diffusive model.
- **c:** Distribution of Reach-level river WSE residual variability, in metres, with density on the vertical axis.

The schematic is explanatory, not a fitted or simulated observation record. The residual-variability diagnostic mixes unresolved hydrologic fluctuations and observation error; neither the historical script name `test_noise` nor the field `sigma_swot_emp` makes it a pure SWOT-noise estimate.

**Inputs:** `08_global_coupling_metrics.parquet`, `08_reach_at_uncertainty.parquet`, `08_main_numbers.yml` and local basemaps. The map population differs from the independent daily population in Fig. 2.

## Fig. S6 — Schematic phase-precision limitation

**Code:** `09_fig_s6_cycle_floor.py`. **Output:** `fig_s6_cycle_floor.pdf/.png`.

This single-panel illustration shows relative annual-phase standard error proportional to `1 / sqrt(number of annual cycles)`. The curve is analytic and schematic. A marker is calculated from the maximum elapsed span of an individual pairwise-finite well–Reach record in the current monthly table.

The marker is an observed record-span quantity; the curve is not an empirical confidence interval or a power calculation for every well.

**Input:** `distance1000/6matched_results_monthly.txt` from the Phase 1 workspace. The paired-span marker is recalculated during rendering.

## Fig. S7 — Amplitude–phase signature

**Code:** `10_fig_s7_signature.py`. **Output:** `fig_s7_signature.pdf/.png`.

- **a:** Reach annual amplitude ratio `R_A` versus circular phase lag tau. The shaded damped/delayed region uses `R_A ≤ 1` and positive lag; the four quadrant percentages use the geometric boundary at `R_A = 1`. Point opacity reflects classification uncertainty.
- **b:** Uncertainty-propagated amplified-or-leading and leading-only fractions for all, development and geographic-testing Reaches, with hierarchical 95% intervals.
- **c:** All-Reach and geographic-testing incompatible fractions at independently rematched 0.5/1/2/5-km distance screens.

The operational incompatibility statistic uses `R_A > 1.2 or tau ≤ 0`, including an amplitude tolerance. It should not be confused with the `R_A = 1` quadrant divider in panel a. This is a check against a simplified local diffusive response, not a complete classification of all groundwater–river mechanisms.

**Inputs:** `08_reach_at_uncertainty.parquet` and `08_robustness.json`. Panel a uses an amplitude-ratio axis from 0.005 to 50 and phase from −185 to 185 days, plotting actual values without pushing outliers to the frame. The renderer stops if future data fall outside those limits.

## Fig. S8 — Physical and temporal–distance sensitivity

**Code:** `11_fig_s8_at_sensitivity.py`. **Output:** `fig_s8_at_sensitivity.pdf/.png`.

- **a:** All-Reach incompatibility fraction across a 3 × 3 grid of amplitude-ratio and phase-lag cutoffs. The outlined circle identifies the primary `R_A > 1.2 or tau ≤ 0` setting.
- **b:** All-Reach incompatibility fraction across loose/main/strict temporal gates and independently rematched 0.5/1/2/5-km distance screens.

Color and printed percentages encode fractions. Circle area is constant and has no quantitative meaning; denominator labels provide Reach counts. Both panels describe all eligible Reaches for the relevant setting.

**Inputs:** `08_main_numbers.yml`, saved `fig_s2_temporal_criteria_stats.json` and the bundled reversed `lajolla` color table. To update temporal sensitivity from changed observations, rerun S2 first; rendering S8 alone does not recompute that matrix.

## Fig. S9 — Frequency ordering and its uncertainty

**Code:** `12_fig_s9_s10_criteria.py`, shared with S10. **Output:** `fig_s9_frequency_response.pdf/.png`.

- **a:** Fractions satisfying `TF2 < TF1` after propagating all four river/groundwater harmonic amplitudes, compared with the equal-gain sampling-calendar null.
- **b:** Fractions of robust low-pass, uncertain and robust opposite classifications, connected by group-specific lines. Categories correspond to low-pass probabilities at least 0.95, between 0.05 and 0.95, and at most 0.05. These lines connect categories; they are not time series.
- **c:** Propagated low-pass fractions across joint-amplitude-SNR tertiles for all Reaches, with uncertainty intervals.

S9 uses the complete eligible monthly TF population without requiring a finite daily residual correlation. The equal-gain null compares two frequencies under matched gains; it is distinct from S3's independent-series OU null.

**Inputs:** Final 1-km monthly table; `08_global_coupling_metrics.parquet`; `12_tf_comparison.json`; `13_tf_lowpass_uncertainty.json/.parquet`. The shared script also loads the daily metric table for S10. It refits harmonics and checks population/statistic agreement against the archived TF products.

## Fig. S10 — Annual gain versus residual-correlation criteria

**Code:** `12_fig_s9_s10_criteria.py`, shared with S9. **Output:** `fig_s10_gain_correlation.pdf/.png`.

A Reach-level scatter compares annual power gain `TF1` on a logarithmic horizontal axis with daily residual correlation on the vertical axis. The shaded gain band is `0.25 ≤ TF1 ≤ 4`, equivalent to annual amplitude ratio from 0.5 to 2; the horizontal criterion is `r ≥ 0.3`. The upper-right summary gives classification agreement, Cohen's kappa and Spearman rho.

The population is the exact finite annual-TF × finite daily-correlation intersection joined by `site_no` and `reach_id` before Reach aggregation. It therefore differs from S9's full monthly population. Similar marginal fractions do not imply that the two criteria select the same Reaches.

**Inputs:** S9's shared inputs plus `08_event_metrics_all_sites.parquet`. The script verifies the intersection and comparison statistics. The axes span `TF1 = 10^-4 to 2,000` and `r = −0.75 to 1`; actual values are plotted without boundary clipping, and an out-of-range guard stops rendering for new data requiring wider limits.

## Shared style module and supporting resources

[13_grl_figure_style.py](01_Code/13_grl_figure_style.py) is imported by all renderers. It sets a common 9-point text profile and 10-point panel letters, normal-weight titles/labels, line/tick defaults and vector-export font settings. Text, numerals and plain `r` use Arial; mathematical letters use the configured matching family where available, with STIX fallback for unsupported symbols. The supplementary-style entry point is an alias of the same profile.

The module enforces its configured figure-size ranges: 50–85 mm or 105–170 mm width and at most 228 mm height. It also points Cartopy at included Natural Earth resources. These describe the implementation's settings, not a claim that every publication or display will use the same final scale.

Arial is not distributed. Install it to match reference typography, or select an installed alternative before the root command:

~~~bash
export GRL_FONT_FAMILY="DejaVu Sans"
~~~

Changing the family changes text layout and can prevent pixel-identical reproduction. PDF/PNG are supplied for viewing; SVG keeps editable text and can depend on the viewing system's installed fonts.

| Resource | Role |
|---|---|
| `colormaps/vik.txt` | Reversed diverging palette for Fig. 3 Reach points |
| `colormaps/lajolla.txt` | Reversed sequential palette for S8 circles |
| `colormaps/oslo.txt` | Retained palette resource; current visible S8 rendering uses lajolla |
| `colormaps/LICENSE.txt` | License for bundled color tables |
| `02_Resources/natural_earth/` | Land/coastline layers at 1:110 million and 1:50 million scales for Figs. 2 and S5 |

## Interpretation notes

S4 recomputes statistical bootstraps during rendering. Its execution time can therefore exceed that of figures that mainly display saved summaries.

The public figure labels use “residual correlation r” and “geographic testing group.” Historical internal names such as `event_r`, `holdout`, `A` and `12_fig2_example_series.parquet` remain for compatibility. These names do not imply extra analyses or different figure assignments.

Field units and denominator definitions are given in the [Phase 1 table reference](../Phase_1_Data_Preprocessing/readme.md) and [Phase 2 analysis guide](../Phase_2_Data_Analysis/readme.md).

The four Natural Earth layers are `ne_110m_land`, `ne_110m_coastline`, `ne_50m_land` and `ne_50m_coastline`, each with its SHP/SHX/DBF/PRJ/CPG components. These are public-domain basemaps; credit: Made with Natural Earth. See the [Natural Earth terms of use](https://www.naturalearthdata.com/about/terms-of-use/).
