"""Daily percentile indices with explicit calendar and reference resampling.

The in-base adjustment excludes the target year and duplicates each other
reference year in turn (Zhang et al., 2005). Annual frequencies, rather than
thresholds, are averaged. February 29 is scored against February 28 thresholds.
"""
import warnings

import numpy as np
import pandas as pd
import xarray as xr

from .was_scientific_utils import noleap_dayofyear


def _thresholds(annual, percentile, method):
    # shape: year, calendar_day; windows never bridge an absent calendar day.
    padded = np.concatenate((annual[:, -2:], annual, annual[:, :2]), axis=1)
    windows = np.lib.stride_tricks.sliding_window_view(padded, 5, axis=1)
    with warnings.catch_warnings():
        warnings.filterwarnings("ignore", message="All-NaN slice encountered")
        return np.nanpercentile(windows, percentile, axis=(0, 2), method=method)


def _annual_percentile_series(values, *, years, doys, leapday, selected,
                              output_years, reference_years, percentile,
                              hot, bootstrap, min_valid_fraction, method):
    values = np.asarray(values, dtype=float)
    values = np.where(np.isfinite(values), values, np.nan)
    annual = np.full((len(reference_years), 365), np.nan)
    for k, year in enumerate(reference_years):
        take = (years == year) & ~leapday
        annual[k, doys[take] - 1] = values[take]
    overall = _thresholds(annual, percentile, method)
    result = np.full(len(output_years), np.nan)
    coverage = np.zeros(len(output_years))

    def frequency(take, threshold):
        x, q = values[take], threshold[doys[take] - 1]
        valid = np.isfinite(x) & np.isfinite(q)
        frac = valid.sum() / len(x) if len(x) else 0.0
        if frac < min_valid_fraction or not valid.any():
            return np.nan, frac
        extreme = x[valid] > q[valid] if hot else x[valid] < q[valid]
        return 100.0 * extreme.mean(), frac

    for i, year in enumerate(output_years):
        take = (years == year) & selected
        if bootstrap and year in reference_years:
            remaining = annual[reference_years != year]
            estimates = [frequency(take, _thresholds(
                np.concatenate((remaining, remaining[k:k + 1]), axis=0),
                percentile, method)) for k in range(len(remaining))]
            # Every replacement is required; missing replicates are not discarded.
            result[i] = np.mean([v[0] for v in estimates])
            coverage[i] = min(v[1] for v in estimates)
        else:
            result[i], coverage[i] = frequency(take, overall)
    return result, coverage


def temperature_percentile_index(data, *, base_period, percentile=90,
                                 hot=True, season=None, in_base_bootstrap=True,
                                 min_valid_fraction=1.0,
                                 quantile_method="median_unbiased"):
    """Return annual percentage and valid-day fraction on a complete calendar.

    Gregorian daily data are required, with a ``time`` axis and arbitrary
    remaining dimensions (including station or a singleton grid). Year bounds
    are inclusive. The default annual coverage requires all requested days.
    ``median_unbiased`` is Hyndman-Fan type 8; ``linear`` reproduces type 7.
    The exhaustive reference-year resampling is deterministic, not a CI.
    """
    if not 0 < min_valid_fraction <= 1:
        raise ValueError("min_valid_fraction must be in (0, 1]")
    if not isinstance(base_period, slice) or base_period.start is None or base_period.stop is None:
        raise ValueError("base_period needs inclusive start and stop years")
    if season is not None and (not len(season) or any(int(m) != m or not 1 <= m <= 12 for m in season)):
        raise ValueError("season must contain calendar months in 1..12")
    start, stop = int(str(base_period.start)[:4]), int(str(base_period.stop)[:4])
    if stop < start:
        raise ValueError("base_period stop precedes start")
    dates = pd.DatetimeIndex(data.time.values)
    if dates.empty or dates.hasnans or dates.has_duplicates or not dates.equals(dates.normalize()):
        raise ValueError("Daily time must be nonempty, unique and labelled at midnight")
    dates_full = pd.date_range(f"{dates.min().year}-01-01", f"{dates.max().year}-12-31")
    data = data.sortby("time").reindex(time=dates_full)
    output_years = np.unique(dates_full.year)
    reference_years = output_years[(output_years >= start) & (output_years <= stop)]
    if not len(reference_years):
        raise ValueError("Empty base period")
    if in_base_bootstrap and len(reference_years) < 2:
        raise ValueError("In-base bootstrap requires at least two reference years")
    kwargs = dict(years=dates_full.year.to_numpy(), doys=noleap_dayofyear(dates_full),
                  leapday=np.asarray((dates_full.month == 2) & (dates_full.day == 29)),
                  selected=np.asarray(dates_full.month.isin(season) if season else np.ones(len(dates_full), bool)),
                  output_years=output_years, reference_years=reference_years,
                  percentile=percentile, hot=hot, bootstrap=in_base_bootstrap,
                  min_valid_fraction=min_valid_fraction, method=quantile_method)
    if data.chunks is not None:
        data = data.chunk(time=-1)
    result, coverage = xr.apply_ufunc(
        _annual_percentile_series, data,
        input_core_dims=[["time"]], output_core_dims=[["year"], ["year"]],
        vectorize=True, dask="parallelized", output_dtypes=[float, float],
        dask_gufunc_kwargs={"output_sizes": {"year": len(output_years)}}, kwargs=kwargs)
    coord = pd.to_datetime([f"{y}-01-01" for y in output_years])
    result = result.rename(year="time").assign_coords(time=coord)
    coverage = coverage.rename(year="time").assign_coords(time=coord)
    result = result.assign_coords(valid_day_fraction=coverage)
    result.attrs.update(in_base_bootstrap=int(in_base_bootstrap), quantile_method=quantile_method,
                        min_valid_fraction=min_valid_fraction, units="%",
                        reference="Zhang et al. 2005, DOI 10.1175/JCLI3366.1",
                        leap_day_threshold="February 28", calendar="proleptic_gregorian")
    return result.transpose("time", *[d for d in data.dims if d != "time"])
