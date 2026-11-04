"""Keep well observations independently of the selected river Node's calendar."""

import numpy as np
import pandas as pd

WELL_COLUMNS = (
    'site_no', 'date', 'wells_lon', 'wells_lat', 'Value_EGM2008_m',
    'GWL_Value_m', 'Well_Elevation_EGM2008', 'AquiferType', 'country_province',
    'Depth to Water Below Land Surface in meter', 'Well_Elevation',
    'precipitation_mm', 'evapotranspiration_mm',
)


def restore_top1_groundwater_dates(candidates, retained, top1):
    """Append missing well-only dates to Top-1, preserving every river row.

    The unranked restricted outer join contains the complete source well
    record. A well date attached only to a rejected Node must survive as a
    well-only row. Copy only well/climate fields: river measurements, quality,
    geometry, and distances from the rejected Node must never be transferred.
    """
    keys = ['site_no', 'date']
    values = pd.to_numeric(candidates['Value_EGM2008_m'], errors='coerce')
    cols = [c for c in WELL_COLUMNS if c in candidates]
    wells = candidates.loc[np.isfinite(values), cols].drop_duplicates()
    if wells.duplicated(keys).any():
        raise ValueError('Conflicting well/climate fields within a source well-day')
    if top1['site_no'].duplicated().any():
        raise ValueError('Top-1 must identify exactly one Reach/Node per well')
    wells = wells.merge(top1, on='site_no', how='inner', validate='many_to_one')
    present = retained.loc[retained['number_match_SG'].eq(1), keys].drop_duplicates()
    missing = wells.merge(present.assign(_present=True), on=keys,
                          how='left', validate='one_to_one')
    missing = missing.loc[missing['_present'].isna()].drop(columns='_present')
    added = missing.reindex(columns=retained.columns)
    result = pd.concat([retained, added], ignore_index=True, sort=False)
    # Invariant: all source finite well days survive; no river observation is
    # manufactured on an appended date, so the daily paired sample is unchanged.
    if len(added) and added['river_wse'].notna().any():
        raise AssertionError('Restored groundwater dates contain invented river WSE')
    print(f'Restored Top-1 well-only dates: {len(added):,}; '
          f'wells: {added.site_no.nunique():,}', flush=True)
    return result
