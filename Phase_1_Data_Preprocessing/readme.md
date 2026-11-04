# Phase 1: groundwater–river pairing and climate preprocessing

Phase 1 now has **18 Python files, one configuration example, and no notebooks or shell wrappers**. All three matching distances share the same implementation. The source distribution contains no observation data.

Phase 1 converts externally supplied groundwater records, well metadata, merged SWOT RiverSP observations, SWORD prior geometry, elevation information and ERA5-Land fields into one selected well–Node–Reach time series per well at each distance threshold. Its two principal products are the final daily table and the final monthly table. Phase 2 fits the scientific metrics to those products; Phase 3 uses the resulting metrics and, for selected figures, the paired tables themselves.

This phase does not produce manuscript figures. Its role is to establish the observation calendar, spatial pairing, quality screens and climate quantities on which those figures depend. Raw-data download, provider-specific datum conversion and the separate SWOT granule-merging program precede the input boundary documented here.

Full-input validation on 4 November 2026 completed all 32 planned commands, including DEM supplementation and its independent audit. The final daily and monthly products for 1, 3 and 5 km were byte-identical to the reference workflow.

## Workflow at a glance

~~~text
Groundwater + metadata + DEM ── 01, 02 ── completed groundwater + audit
Metadata + SWORD + SWOT Node ── 03 ───── 5-km Node candidates
SWOT Reach attributes ───────── 04 ───── 5-km enriched candidates
5-km candidates ─────────────── 05 ───── nested 3-km and 1-km candidates
Hourly accumulated climate ─── 06, 07 ── corrected daily climate grids

For each requested distance, in order 5 km → 3 km → 1 km:
  08 join well and river calendars
  09 attach daily precipitation and actual ET
  10 select a fixed Node per well–Reach and rank Reaches
  11 inspect rank validity
  12 retain the highest-ranked Reach per well
  13 aggregate overlapping records to unique daily observations
  14 create monthly water-level means and full-calendar climate totals
~~~

The two calendars have different roles: groundwater-only days remain available for groundwater monthly means, while later daily correlations use only dates with both finite river and groundwater levels. The three distance branches select their own final pair; they are not produced by filtering an already selected 5-km pair.

## Entry point and configuration

The entry point is `01_Code/00_run_phase1.py`. Copy `config.example.json` to a location of your choice and edit its input paths, output workspace and worker counts. Relative paths are resolved against the configuration file's directory. Production modules contain no original host paths or fixed interpreter paths.

Required external inputs:

| Configuration key | Expected resource |
|---|---|
| `groundwater_raw` | Groundwater observation TXT, equivalent to dataset_all_2023.txt |
| `well_metadata` | Well metadata workbook, equivalent to metadata2023.xlsx |
| `dem_vrt` | Copernicus GLO-30 VRT, with its referenced elevation tiles available |
| `sword_nodes` | Prepared SWORD v17b Node geometry/attributes |
| `swot_merged_root` | Validated Version-D daily FlatGeobuf inventory, with Node/Reach subdirectories, merge_granule_selection.csv and merge_granule_selection_summary.json |
| `precipitation_hourly` | ERA5-Land accumulated precipitation source archives |
| `evapotranspiration_hourly` | ERA5-Land accumulated actual-evaporation source archives |

Source acquisition, earlier groundwater datum conversion, DEM mosaic creation and the separate SWOT merge program are outside this archive's input boundary.

The `workspace` must be separate from input locations and production code/results. All generated groundwater, climate, paired tables and logs are placed in the selected workspace. The runner uses the current Python interpreter for all stages. Use a fresh workspace for a fresh comparison; reusing one can overwrite intermediate products, and the original climate-cache reuse rules still apply.

Install the shared Python dependencies from [requirements.txt](../requirements.txt) using Python 3.11. The same dependency file covers all three phases.

## Commands

From the archive root, inspect the plan without reading observations or creating outputs:

~~~bash
python Phase_1_Data_Preprocessing/01_Code/00_run_phase1.py --config /path/to/phase1_config.json --dry-run
~~~

Run the configured workflow when the external inputs are ready:

~~~bash
python Phase_1_Data_Preprocessing/01_Code/00_run_phase1.py --config /path/to/phase1_config.json --stage all
~~~

Available stages:

| Stage | Operations / prerequisites |
|---|---|
| `all` | Prepare inputs, process requested distances, then perform the final ordered monthly pass |
| `prepare` | DEM completion/audit, one 5-km Node scan, Reach attachment, nested 3/1-km candidates, daily climate conversion |
| `downstream` | Join observations through monthly aggregation; requires prepared Reach-candidate tables, DEM-completed groundwater and corrected daily climate in the workspace |
| `monthly` | Rebuild monthly outputs from final daily tables and corrected daily climate |

`--distances 1000 3000 5000` is the default. Downstream execution sorts distances as 5000 → 3000 → 1000, preserving reuse of the 5-km full-calendar climate reference. For example:

~~~bash
python Phase_1_Data_Preprocessing/01_Code/00_run_phase1.py --config /path/to/phase1_config.json --stage downstream --distances 1000
~~~

Preparation always scans 5 km and derives the smaller candidates. The final monthly pass in `all` preserves the verified production sequence; it calls the same monthly module again, without maintaining a second implementation. Logs and execution receipts are under `WORKSPACE/logs/`. Worker failures reported by the climate converters stop the driver even if the subprocess returns zero.

## Numbered source files

| File(s) | Purpose |
|---|---|
| `00_run_phase1.py` | Unified execution plan, subprocess control and receipts |
| `01_add_groundwater_elevation.py` | Compatible elevation/depth/water-level completion |
| `02_audit_groundwater_elevation.py` | Independent preservation, formula and datum audit |
| `03_match_swot_nodes.py` | 5-km static prior-geometry matching and quality screens |
| `04_attach_reach_attributes.py` | Join date/Reach-specific attributes |
| `05_derive_nested_candidates.py` | Exact nested 3/1-km candidates from the 5-km universe |
| `06_precipitation_hourly_to_daily.py`, `07_evapotranspiration_hourly_to_daily.py` | Shared accumulated-climate conversion |
| `08_join_groundwater_river.py` | Restricted outer join preserving well dates |
| `09_merge_daily_climate.py` | Shared coordinate/date climate extraction |
| `10_select_fixed_pairs.py` | Fixed Node selection, Reach ranking, groundwater-date restoration |
| `11_validate_pair_ranks.py` | Candidate-rank validity check |
| `12_keep_top_pair.py` | Retain the top Reach/Node per well |
| `13_aggregate_daily.py` | Exact duplicate removal and same-day aggregation |
| `14_build_monthly.py` | Independent monthly level means and full-calendar climate |
| `15_pipeline_config.py` | Path/resource configuration and table-name contracts |
| `16_geodesic_candidate_search.py` | Conservative candidate-search radius |
| `17_preserve_groundwater_dates.py` | Shared groundwater-calendar preservation |

There are no distance-specific code copies. Scripts 15–17 are shared modules; numbering there is a browsing order, not a later processing stage.

## What each code file does

### 00 — Execution entry point

[00_run_phase1.py](01_Code/00_run_phase1.py) reads the JSON configuration, constructs the requested stage plan, checks that required inputs exist and runs each calculation in a subprocess using the current Python interpreter. It supplies configuration and distance settings to the calculation modules, keeps per-step logs and writes a JSON execution receipt. A nonzero child exit or a reported climate-worker `FAILED:` message stops the run. The `--dry-run` option prints the plan without reading observation tables or creating the workspace.

### 01 — Ground elevation and complementary water-level fields

[01_add_groundwater_elevation.py](01_Code/01_add_groundwater_elevation.py) reads the groundwater TXT, well metadata workbook and Copernicus DEM VRT. It fills missing eligible ground elevations and derives a missing depth or water-level counterpart only under the implemented datum and ground-reference rules. The relationship is water-level elevation = ground elevation − depth below land surface. Existing populated elevations and groundwater values are preserved.

The code treats ground elevations, reference-point elevations, uncertain elevation proxies and unresolved source datums separately. It records the ground basis, datum and completion status; it does not assume that every existing field named `Value_EGM2008_m` has a verified common datum. Outputs are `groundwater/dataset_all_2023_DEM.txt` and its `.qc.json` report.

### 02 — Independent elevation audit

[02_audit_groundwater_elevation.py](01_Code/02_audit_groundwater_elevation.py) compares the completed groundwater file with the raw input and checks preservation of original rows/values, permitted completions and the depth–elevation relationship. It adds `post_write_audit` to the QC JSON and raises an error for violations. Negative or unusually large depths are reported rather than automatically deleted or clipped.

### 03 — Spatial Node candidates and SWOT screening

[03_match_swot_nodes.py](01_Code/03_match_swot_nodes.py) reads well coordinates from the metadata workbook, SWORD v17b Node priors and the externally merged Version-D Node inventory. It validates the merge inventory, obtains a conservative spherical candidate set and then screens using WGS84 geodesic distances to the static prior Node geometry. It writes the 5-km Node-candidate table, including prior/observed coordinates, distance diagnostics, WSE, width and quality attributes.

The main distance is `max(geodesic well-to-prior-Node distance − prior width / 2, 0)`. It approximates bank distance; it is not a surveyed bank boundary. Wells absent from the configured metadata workbook do not acquire spatial candidates merely because the groundwater TXT contains their observations.

The implemented observation screens include:

| Quantity | Accepted values |
|---|---|
| Prior width | Greater than 50 m; applied to the product prior and joined SWORD prior |
| Observed width | Greater than 0 and less than 100,000 m |
| WSE | Greater than −1,000 m, with required finite fields |
| Water type / dam flag | River code `1` and `p_dam_id = 0` |
| Ice flags | Each of `ice_dyn_f` and `ice_clim_f` is 0 or −999 |
| Dark fraction | At least 0 and less than 0.5 |
| Absolute cross-track distance | 10,000–60,000 m, inclusive |
| Good pixels | At least 8 |
| Node quality | `node_q = 0`, or `node_q = 1` with `wse_u < 0.3 m` |
| Prior-geometry offset | At most 100 m between product prior and joined SWORD prior |

The accepted −999 ice flag means unavailable information, not a confirmed ice-free observation. The source code contains additional identifier, missing-value and inventory checks.

### 04 — Date- and Reach-specific river attributes

[04_attach_reach_attributes.py](01_Code/04_attach_reach_attributes.py) reads the 5-km Node candidates and the merged Reach inventory. It joins `slope2`, `p_wid_var` and `d_x_area` using the acquisition date and the Node record's corresponding `reach_id`. Thus the prior width variance is taken from the matching Reach product, not from another nearby Reach.

Multiple source rows for the same date/Reach are reduced to valid-value medians before joining, avoiding multiplication of Node observations. Support counts and match diagnostics accompany the enriched candidate table.

### 05 — Nested candidate universes

[05_derive_nested_candidates.py](01_Code/05_derive_nested_candidates.py) streams a 5-km candidate TSV and writes its exact 1-km and 3-km subsets using `distance_prior_bank_m`. The driver calls it once for Node candidates and once for Reach-enriched candidates. This step creates candidate universes; fixed-Node selection and Reach ranking occur later, independently at each threshold.

### 06 — Daily precipitation

[06_precipitation_hourly_to_daily.py](01_Code/06_precipitation_hourly_to_daily.py) reads the external ERA5-Land precipitation sources, including supported archives, and converts accumulated forecast values into daily precipitation. It uses the next day's 00:00 UTC accumulation, shifts the timestamp back to the day represented and converts metres to millimetres. It does not sum successive within-day cumulative values. Output NetCDF files follow `ERA5_Land_Precip_Daily_YYYY_MM.nc` under `climate/PRE_daily/`.

### 07 — Daily actual evapotranspiration

[07_evapotranspiration_hourly_to_daily.py](01_Code/07_evapotranspiration_hourly_to_daily.py) uses the same next-day-midnight convention for ERA5-Land actual evaporation. It converts metres to millimetres and reverses the source evaporation sign to express evaporative loss as positive. It does not replace actual ET with potential ET. Output files follow `ERA5_Land_ET_Daily_YYYY_MM.nc` under `climate/ET_daily/`. Both climate converters need the following month's endpoint to complete a month.

### 08 — Join groundwater and candidate river calendars

[08_join_groundwater_river.py](01_Code/08_join_groundwater_river.py) joins the distance-specific Reach candidates with the DEM-completed groundwater series within 2023–2025. Its restricted outer join retains well-only dates for wells in the candidate population, as well as river observation dates. The output is the `2results...withwidth50.txt` table; missing river values on groundwater-only dates are intentional.

Before joining, the current implementation reduces duplicate groundwater `site_no/date` keys by keeping the first input row. This is different from script 13's exact-record deduplication and daily river aggregation. If multiple conflicting source measurements exist for one well/day, their resolution must be considered upstream; this documentation revision does not change that behavior.

### 09 — Attach daily climate

[09_merge_daily_climate.py](01_Code/09_merge_daily_climate.py) reads the joined table and corrected daily grids, extracts climate by well coordinates and date, and adds `precipitation_mm` and `evapotranspiration_mm`. Its `3results...withwidth50_pre_et.txt` output is still a candidate table, not the final fixed-pair product. Monthly totals are subsequently rebuilt from the full daily climate calendar by script 14.

### 10 — Fixed Node selection and Reach ranking

[10_select_fixed_pairs.py](01_Code/10_select_fixed_pairs.py) first selects one Node within each well–Reach combination by descending number of distinct accepted observation dates, then ascending median prior-bank distance, then ascending Node ID. It then ranks those Reaches within each well using descending observation-day count, ascending distance, ascending Reach ID and ascending Node ID.

The resulting `number_match_SG` ranks therefore determine the final Reach as well as the Node. Before writing the selected-series table, the script calls module 17 to restore groundwater dates lost when rejected Nodes are removed. It retains all ranked Reaches at this stage; script 12 performs the final rank-1 selection. `estimated_wse` is retained as a compatibility field equal to observed `river_wse`.

### 11 — Candidate-rank diagnostics

[11_validate_pair_ranks.py](01_Code/11_validate_pair_ranks.py) reads the selected-series table and reports counts by `number_match_SG`. When the rank field is present, it checks that ranks are numeric, nonmissing and positive, with at least one rank-1 record. Multiple Reach ranks are expected at this point. This diagnostic writes to the execution log, not to a new data table; the subsequent script also requires the rank column before filtering.

### 12 — One final pair per well

[12_keep_top_pair.py](01_Code/12_keep_top_pair.py) keeps only `number_match_SG == 1` and writes `4matched_day_wells_shpsite_with_estimated_wse_fixed_node_alltimes_SG1.txt`. This is where the workflow reduces the ranked candidates to one selected Reach/Node per well at the current distance threshold.

### 13 — Unique daily observations

[13_aggregate_daily.py](01_Code/13_aggregate_daily.py) removes exact duplicate records, normalizes observation days and aggregates by well, Reach, Node, rank and date. Water levels, widths and specified continuous fields use means; `river_wse_u`, Node-quality code and dark fraction use conservative maxima; good-pixel count uses the minimum. Other metadata follow the source aggregation contract.

The output is `5matched_results_daily.txt`. River and groundwater values may be missing independently; a well-only day is not discarded because a river observation is absent. Width aggregation uses the complete daily group rather than separate read-chunk summaries.

### 14 — Monthly water levels and complete-month climate

[14_build_monthly.py](01_Code/14_build_monthly.py) reads the final daily table and corrected daily climate grids. River and groundwater monthly means use their respective finite daily observations, not only their common dates. Climate totals instead sum every calendar day in each study month at the well location, including months with no water-level observations. The implementation includes the documented bounded coastal-grid fallback.

Outputs are `6matched_results_monthly.txt` and `6matched_results_monthly.climate_qc.json`. The monthly product records river, groundwater and paired observation-day counts and the number of climate days contributing to each total. The final ordered monthly pass can reuse the 5-km climate reference while retaining threshold-specific water-level series.

### 15 — Configuration and path contracts

[15_pipeline_config.py](01_Code/15_pipeline_config.py) defines the required input keys, allowed distances, worker settings, workspace rules and output names. It resolves relative paths against the configuration file and rejects unknown settings or overlapping input/output locations. Calculation modules receive this configuration through the entry point's environment; they are not designed to be launched without that context.

### 16 — Conservative geodesic candidate search

[16_geodesic_candidate_search.py](01_Code/16_geodesic_candidate_search.py) supplies the conservative unit-sphere chord radius used by script 03. It uses the WGS84 minimum curvature radius to avoid excluding valid candidates before the precise ellipsoidal-distance check. It produces no standalone data file.

### 17 — Preserve the groundwater calendar

[17_preserve_groundwater_dates.py](01_Code/17_preserve_groundwater_dates.py) restores well dates that disappear when candidate Nodes are filtered. It retains groundwater and well metadata on those days, associates them with the selected top pair and leaves river observations missing where that selected Node was not observed. It is called by script 10 and does not create an independent output product.

## Preserved methods and output contract

Quality filters, study window, pair-ranking rules, missing-date preservation, aggregation and climate methods are unchanged. The primary distance remains `max(geodesic well-to-prior-Node distance - prior width/2, 0)`. Each well retains one Node and Reach after independent threshold-specific ranking.

Monthly river and groundwater means use their respective finite daily observations. Daily residual analyses later use common finite dates. Climate totals cover all calendar days in 2023–2025. Original final filenames and fields are retained:

~~~text
WORKSPACE/
  groundwater/dataset_all_2023_DEM.txt
  climate/PRE_daily/
  climate/ET_daily/
  distance1000/5matched_results_daily.txt
  distance1000/6matched_results_monthly.txt
  distance3000/...
  distance5000/...
  logs/
~~~

For subsequent analysis or figures, pass this workspace as `--phase1-data` to `Phase_2_Data_Analysis/01_Code/16_reproduce.py`. See the Phase 2 and Phase 3 guides for the corresponding commands.

## How these products support the analyses and figures

| Phase 1 product | Main downstream uses |
|---|---|
| Final daily paired table | Phase 2 daily residual correlations, shift/OU nulls, example records and river residual variability; direct daily-support calculations in Fig. S2 |
| Final monthly paired table | Phase 2 annual amplitude/phase, covariates and annual/semiannual gains; direct recalculation in S2, S6, S9 and S10 |
| Climate-enriched pre-ranking candidates | Independent threshold-specific rematching for the 0.5/1/2/5-km sensitivity, when rebuilding its cache from external candidates |
| QC JSON files and logs | Auditing elevation completion, calendar coverage and step completion; not manuscript figure inputs |

See [Phase 2](../Phase_2_Data_Analysis/readme.md) for metric definitions and [Phase 3](../Phase_3_Data_Visualization/readme.md) for the panel-by-panel figure guide. Final table field definitions follow below.

## Final daily and monthly table fields

| Field or field family | Meaning / units |
|---|---|
| `site_no` | Provider-prefixed well identifier; categorical, preserve exactly |
| `reach_id`, `node_id` | Selected SWORD/SWOT Reach and Node identifiers; categorical |
| `date` | Observation day in the daily table; calendar month label in the monthly table; study window 2023–2025 |
| `wells_lon`, `wells_lat` | Well longitude/latitude, decimal degrees |
| `river_lon`, `river_lat` | Observed SWOT Node position, decimal degrees |
| `river_p_lon`, `river_p_lat` | Static prior Node position, decimal degrees |
| `river_wse` | Observed river water-surface elevation, metres; input to river harmonic and residual calculations |
| `river_wse_u` | SWOT water-surface-elevation uncertainty, metres |
| `Value_EGM2008_m` | Existing groundwater-level elevation analysis field, metres; see datum note below |
| `GWL_Value_m` | Preserved source groundwater-value field, metres; source-specific interpretation |
| `Depth to Water Below Land Surface in meter` | Groundwater depth relative to land surface, metres; negative values are retained where present |
| `Well_Elevation_EGM2008`, `Well_Elevation` | Target/source well-elevation metadata, metres |
| `river_width`, `river_p_width` | Observed/prior river width, metres |
| `distance_prior_center_m` | Geodesic well-to-prior-Node distance, metres |
| `distance_prior_bank_m` | `max(distance_prior_center_m - river_p_width / 2, 0)`, metres; primary matching distance |
| `distance_adjusted_m`, `distance_m` | Compatibility aliases of prior bank and prior center distances, respectively |
| `distance_observed_*_m` | Distances based on observed Node geometry, metres; diagnostic rather than the primary match screen |
| `prior_geometry_offset_m` | Offset between product prior position and selected prior geometry, metres |
| `river_node_q` | Aggregated Node-quality field; consult the RiverSP product definition for native flags |
| `n_good_pix`, `dark_frac`, `xtrk_dist` | Good-pixel count, dark-water fraction, cross-track distance in metres; upstream observation quality fields |
| `slope2`, `p_wid_var`, `d_x_area` | Attributes attached by acquisition date and matching Reach ID; retain native RiverSP definitions and units |
| `reach_source_count`, `reach_valid_*_count`, `reach_attribute_matched` | Reach-attribute support and match diagnostics |
| `number_match_SG` | Upstream match-ranking field; final selected pair has rank 1 |
| `shp_file` | Historical source-file label; not a file dependency to open |
| `estimated_wse` | Legacy estimated river-level field; the main paired analyses use observed `river_wse` |
| `AquiferType`, `Country/Province` | Preserved source metadata categories; missing/unknown categories remain explicit |
| `precipitation_mm` | Daily precipitation or full-calendar monthly precipitation total, millimetres |
| `evapotranspiration_mm` | Daily actual ET loss or full-calendar monthly actual ET total, millimetres; not potential ET |
| `river_wse_obs_days`, `groundwater_obs_days`, `paired_obs_days` | Monthly counts of finite river days, groundwater days and same-date paired days |
| `precipitation_days`, `evapotranspiration_days` | Number of daily climate values contributing to the monthly total |

The upstream target datum is EGM2008, but the name `Value_EGM2008_m` does not prove successful datum conversion for every source record. The source final tables use this field name; source datum exceptions are not resolved by repackaging. Reuse for absolute hydraulic-head differences requires the original source-datum information, which is outside this release.

Blank TSV cells and Parquet null/NaN values mean missing data. Some native source flags or missing categories have special values; they must not be treated as quantitative observations. Repackaging does not introduce additional interpolation or alter the existing missing-value handling.

