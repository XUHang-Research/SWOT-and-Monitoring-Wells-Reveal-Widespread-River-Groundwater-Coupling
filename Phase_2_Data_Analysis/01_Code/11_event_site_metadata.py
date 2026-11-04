"""Aggregate daily-event metadata over all paired rows, across read chunks."""

import pandas as pd


class EventSiteMetadata:
    columns = ('river_width', 'wells_lon', 'wells_lat')

    def __init__(self):
        self.reaches = {}
        self.values = {}

    def update(self, site_no, paired_rows):
        if paired_rows.reach_id.nunique(dropna=False) != 1:
            raise ValueError(f'site {site_no} maps to multiple Reach IDs')
        reach = paired_rows.reach_id.iloc[0]
        if site_no in self.reaches and self.reaches[site_no] != reach:
            raise ValueError(f'site {site_no} maps to multiple Reach IDs across chunks')
        self.reaches[site_no] = reach
        numeric = paired_rows[list(self.columns)].apply(pd.to_numeric, errors='coerce')
        self.values.setdefault(site_no, []).append(numeric)

    def finalize(self):
        return {
            site_no: (self.reaches[site_no], *pd.concat(parts, ignore_index=True).median().tolist())
            for site_no, parts in self.values.items()
        }
