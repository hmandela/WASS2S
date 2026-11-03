"""Calendar contracts shared by the PRCP and TEMP native model paths."""
import numpy as np
import pandas as pd


def _lag_source_dates(targets, lag_months, init_month):
    months = [int(m) for m in lag_months]
    if len(months) != 3 or any(not 1 <= m <= 12 for m in months):
        raise ValueError("PCR OBS-LAG requires three valid consecutive season months")
    if any(months[i+1] != months[i] % 12 + 1 for i in (0, 1)):
        raise ValueError("PCR OBS-LAG season months must be consecutive and ordered")
    init_month = int(init_month)
    if not 1 <= init_month <= 12:
        raise ValueError("Initialization month must be in 1..12")
    dates = []
    for date in pd.DatetimeIndex(targets):
        init_year = date.year - int(date.month < init_month)
        initialization = pd.Timestamp(init_year, init_month, 1)
        centre_date = pd.Timestamp(init_year, months[1], 1)
        # Availability on initialization day is valid; a season still in
        # progress is not. NDJ can require stepping back two calendar years.
        while centre_date + pd.DateOffset(months=2) > initialization:
            centre_date -= pd.DateOffset(years=1)
        dates.append(centre_date)
    return pd.DatetimeIndex(dates)


def lagged_pcr_year_bounds(observations, lag_months, target_time, init_month):
    """Download interval in season-start years, including the operational lag."""
    targets = list(observations['T'].values) + [target_time]
    source_dates = _lag_source_dates(targets, lag_months, init_month)
    # WAS_Download_Reanalysis year_start/end refer to the FIRST season month;
    # its seasonal output is labelled with the actual CENTRE calendar date.
    start_years = source_dates.year - int(int(lag_months[1]) < int(lag_months[0]))
    return int(start_years.min()), int(start_years.max())


def align_lagged_pcr_fields(field, observations, lag_months, target_time, init_month):
    """Align raw seasonal predictors to target dates, using completed lag seasons.

    Source files label seasons by their centre month. For each target, select the
    latest lag season completed before initialization, by calendar (not row order).
    The target year is the valid year; initialization may lie in the previous year.
    The original predictor dates survive as ``lag_predictor_time`` metadata.
    """
    source = pd.DatetimeIndex(field['T'].values)
    keys = source.to_period('M')
    if keys.has_duplicates:
        raise ValueError("PCR OBS-LAG has duplicate source year/month labels")
    lookup = {key: i for i, key in enumerate(keys)}

    def select(targets):
        targets = pd.DatetimeIndex(targets)
        required = _lag_source_dates(targets, lag_months, init_month)
        rows = []
        for date, centre in zip(targets, required):
            key = centre.to_period('M')
            if key not in lookup:
                raise ValueError(f"PCR OBS-LAG missing completed predictor season {key} "
                                 f"for target {date.date()}")
            rows.append(lookup[key])
        selected = field.isel(T=rows).assign_coords(
            T=np.asarray(targets, dtype='datetime64[ns]'),
            lag_predictor_time=('T', source.values[rows]))
        selected.attrs = dict(field.attrs)
        selected.attrs['time_convention'] = 'target valid time; source in lag_predictor_time'
        return selected

    return select(observations['T'].values), select([target_time])
