# SWOT–groundwater analysis workflow

Code and derived results for the study of river–groundwater coupling using SWOT RiverSP water-surface elevations and groundwater monitoring records from 2023–2025.

## Package structure

~~~text
Zenodo_Submit/
├── README.md
├── LICENSE
├── requirements.txt
├── Phase_1_Data_Preprocessing/
├── Phase_2_Data_Analysis/
└── Phase_3_Data_Visualization/
~~~

| Phase | Purpose | Detailed guide |
|---|---|---|
| Phase 1 | Groundwater elevation completion, SWOT quality screening, well–river pairing and daily/monthly aggregation | [Phase 1 README](Phase_1_Data_Preprocessing/readme.md) |
| Phase 2 | Harmonic and residual-correlation metrics, covariates, uncertainty, null models and sensitivity analyses | [Phase 2 README](Phase_2_Data_Analysis/readme.md) |
| Phase 3 | Main and supporting figures, shared typography, color tables and map resources | [Phase 3 README](Phase_3_Data_Visualization/readme.md) |

Phase 1 contains preprocessing code and a configuration example. Phase 2 contains analysis code and derived results for the 1-, 3- and 5-km experiments. Phase 3 contains figure code and the reference artwork for Figs. 1–3 and S1–S10.

## Installation

Use Python 3.11 in a dedicated environment, then run from this directory:

~~~bash
python -m pip install -r requirements.txt
~~~

The requirements list the Python package versions from the scientific execution environment. Arial is installed separately. To use another installed font for figures:

~~~bash
export GRL_FONT_FAMILY="DejaVu Sans"
~~~

## Workflow

1. Configure and run `Phase_1_Data_Preprocessing/01_Code/00_run_phase1.py` to create the paired daily/monthly products.
2. Run the analysis steps described in the Phase 2 README.
3. Render the manuscript figures following the Phase 3 README.

The Phase 1 configuration defines the input locations and a separate output workspace. The shared Phase 2–3 entry point is `Phase_2_Data_Analysis/01_Code/16_reproduce.py`; its commands, parameters and examples are documented in the phase guides. Calculations write to a separate workspace so the distributed results are preserved.

The phase READMEs explain every code file, the input/output tables, metric definitions, execution order and each figure panel.

Full reproduction was verified on 4 November 2026 using Python 3.11.14, the listed dependency versions, Arial, and the configured external raw inputs. All six final Phase 1 tables and all 22 Phase 2 Parquet tables were byte-identical to the reference products. Scientific values in all 23 checked statistics files were exactly equal. All 13 PNG figures were pixel-identical, all 13 PDF renders were identical, and both SVG drawings were identical after normalizing generated identifiers and excluding metadata.

## License

Copyright (c) 2026 Hang Xu. Original software is licensed under the MIT License; original research outputs and standalone documentation are licensed under CC BY 4.0. Third-party materials and rights in underlying source observations or databases retain their respective terms. See [LICENSE](LICENSE) for the scope, attribution requirements and third-party notices.

For the Zenodo record, declare both MIT and CC-BY-4.0 and refer to the scopes in LICENSE.

