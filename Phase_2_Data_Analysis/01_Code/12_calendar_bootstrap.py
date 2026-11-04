"""Shared actual-calendar wild weights for all site series in a Reach."""
import hashlib
import numpy as np
import pandas as pd


def stable_seed(value, base_seed):
    digest = hashlib.blake2b(str(value).encode('utf-8'), digest_size=8).digest()
    return (int.from_bytes(digest, 'little') ^ int(base_seed)) % (2**32)


def calendar_block_weights(dates, n_boot, block_months, seed):
    """Random-origin, non-wrapping calendar blocks with +/-1 multipliers."""
    dates = pd.DatetimeIndex(dates)
    if dates.hasnans or not len(dates):
        raise ValueError('calendar block dates must be nonempty and finite')
    if int(block_months) != block_months or block_months < 1 or int(n_boot) != n_boot or n_boot < 1:
        raise ValueError('block_months and n_boot must be positive integers')
    month = dates.year.to_numpy() * 12 + dates.month.to_numpy() - 1
    if len(np.unique(month)) != len(month):
        raise ValueError('calendar bootstrap requires one paired observation per month')
    relative = month - month.min()
    rng = np.random.default_rng(seed)
    origin = rng.integers(0, int(block_months), size=(int(n_boot), 1))
    labels = (relative[None, :] + origin) // int(block_months)
    signs = 2 * rng.integers(0, 2, size=(int(n_boot), int(labels.max()) + 1)) - 1
    return np.take_along_axis(signs, labels, axis=1)


class ReachCalendarPlan:
    """One weight matrix per Reach; different site masks select its columns.

    Share weights across river/GW, wells and Nodes, retaining the empirical
    contemporaneous residual dependence within a Reach. Each site's own fitted
    mean, HC2 residuals and observation mask remain unchanged. Sites with the
    same Node and same calendar obtain identical river bootstrap series.
    Different calendars retain their respective fitted means/projections, but
    never draw independent weights for an overlapping calendar month.
    """
    def __init__(self, dates, n_boot, block_months, seed):
        dates = pd.DatetimeIndex(dates)
        if dates.hasnans:
            raise ValueError('Reach calendar contains missing dates')
        self.months = pd.PeriodIndex(dates, freq='M').unique().sort_values()
        self.n_boot = int(n_boot)
        self.block_months = int(block_months)
        self.weights = calendar_block_weights(
            self.months.to_timestamp(), n_boot, block_months, seed)

    def for_dates(self, dates):
        months = pd.PeriodIndex(pd.DatetimeIndex(dates), freq='M')
        if months.hasnans or months.duplicated().any():
            raise ValueError('site calendar must contain distinct finite months')
        index = self.months.get_indexer(months)
        if (index < 0).any():
            raise ValueError('site date is outside the shared Reach calendar')
        return self.weights[:, index]


def iter_reach_calendars(monthly, pair_table, n_boot, block_months, seed):
    """Validate site/Reach membership and construct only accepted calendars."""
    members = pair_table[['site_no', 'reach_id']].copy()
    members['site_no'] = members.site_no.astype(str)
    if members.site_no.duplicated().any() or members.reach_id.isna().any():
        raise ValueError('uncertainty requires one finite Reach per site')
    data = monthly.copy()
    data['site_no'] = data.site_no.astype(str)
    data = data[data.site_no.isin(members.site_no)].copy()
    expected = members.set_index('site_no').reach_id
    if not data.reach_id.eq(data.site_no.map(expected)).all():
        raise ValueError('monthly and uncertainty site/Reach membership differs')
    data['date'] = pd.to_datetime(data.date, errors='coerce')
    keep = data.date.notna()
    for column in ('river_wse', 'Value_EGM2008_m'):
        data[column] = pd.to_numeric(data[column], errors='coerce')
        keep &= np.isfinite(data[column])
    data = data.loc[keep]
    for reach_id, group in data.groupby('reach_id', sort=False):
        plan = ReachCalendarPlan(group.date, n_boot, block_months,
                                 stable_seed(reach_id, seed))
        yield reach_id, group, plan
