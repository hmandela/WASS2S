"""Calendar and observation-availability rules for seasonal PBC.

The paper's weekly persistence predictors are adapted to preceding windows
of the *same length* as the target season. The default is four windows
shifted by one month; their number and monthly step are configurable. Each
window is compared with its own calendar-season climatology, using only
reference observations available before that issuance.  In particular a dry
season is never classified against the wet-season target's thresholds.

Target-event thresholds are a different object: WASS2S fixes those on the
requested observed climatology and reuses them for fitting and verification.
They may intentionally include years after a historical issuance; they define
the verification event, whereas predictor transformations are strictly as-of.
"""
from __future__ import annotations

import warnings

import numpy as np
import pandas as pd
import xarray as xr

__all__ = ["aggregate_pbc_target_seasons", "build_pbc_seasonal_features",
           "pbc_target_intervals"]

CDF_LEVELS = np.asarray([1.0 / 3.0, 2.0 / 3.0])


def _integer(value, name, minimum=0):
    try:
        integer = int(value)
    except (TypeError, ValueError, OverflowError) as exc:
        raise ValueError(f"{name} must be an integer >= {minimum}") from exc
    if isinstance(value, (bool, np.bool_)) or integer != value or integer < minimum:
        raise ValueError(f"{name} must be an integer >= {minimum}")
    return integer


def _season(months):
    months = [_integer(m, "season month", 1) for m in months]
    if not months or len(months) > 12 or any(m > 12 for m in months):
        raise ValueError("season_months must contain 1..12 calendar months")
    if any(months[j] != months[j - 1] % 12 + 1 for j in range(1, len(months))):
        raise ValueError("season_months must be consecutive calendar months in target order")
    return months


def _times(values, name):
    if isinstance(values, xr.DataArray):
        values = values.values
    if isinstance(values, (str, pd.Timestamp, np.datetime64)):
        values = [values]
    try:
        result = pd.DatetimeIndex(values)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"{name} must contain valid calendar dates") from exc
    if not len(result) or result.hasnans or result.has_duplicates:
        raise ValueError(f"{name} must be nonempty, finite and unique")
    if result.tz is not None:
        raise ValueError(f"{name} must contain timezone-naive calendar dates")
    return result


def _date(period):
    return period.to_timestamp().to_datetime64().astype("datetime64[ns]")


def pbc_target_intervals(target_times, season_months, initialization_month=None,
                         issuance_times=None):
    """Return physical season bounds and issuance dates, preserving T labels.

    A label must name a month within the season.  Thus JAS may be labelled
    July or August, and DJF labelled January 2026 starts in December 2025.
    Inferred issuance is the most recent requested initialization month on or
    before the target's first month. Explicit dates support unusual archives.
    Partial-season forecasts are rejected rather than scored as fully future.
    """
    months = _season(season_months)
    times = _times(target_times, "target_times")
    starts = []
    for t in times:
        if t.month not in months:
            raise ValueError("Each target T label must identify a month within season_months")
        starts.append(t.to_period("M") - months.index(t.month))
    ends = [start + len(months) - 1 for start in starts]
    if issuance_times is None:
        if initialization_month is None:
            raise ValueError("Provide initialization_month or explicit issuance_times")
        initialization_month = _integer(initialization_month, "initialization_month", 1)
        if initialization_month > 12:
            raise ValueError("initialization_month must lie in 1..12")
        issues = [pd.Period(year=s.year, month=initialization_month, freq="M") for s in starts]
        issues = [issue - 12 if issue > start else issue for issue, start in zip(issues, starts)]
    else:
        issues = _times(issuance_times, "issuance_times")
        if len(issues) != len(times):
            raise ValueError("issuance_times and target_times must have the same length")
        # Month starts are the operational convention; mid-month issuance is
        # conservatively treated as having only previous full months available.
        issues = list(issues.to_period("M"))
    if any(issue > start for issue, start in zip(issues, starts)):
        raise ValueError("PBC seasonal targets must be fully future at issuance")
    return xr.Dataset({
        "target_start": ("T", np.asarray([_date(s) for s in starts])),
        "target_end": ("T", np.asarray([_date(e) for e in ends])),
        "issuance_date": ("T", np.asarray([_date(i) for i in issues])),
    }, coords={"T": times.values})


def _monthly(observations, aggregation):
    if not isinstance(observations, xr.DataArray) or set(observations.dims) != {"T", "Y", "X"}:
        raise ValueError("monthly_observations must be a DataArray with exactly T,Y,X dimensions")
    if aggregation not in {"sum", "mean", "day_weighted_mean"}:
        raise ValueError("aggregation must be 'sum', 'mean' or 'day_weighted_mean'")
    for dim in ("T", "Y", "X"):
        if dim not in observations.coords or observations.get_index(dim).has_duplicates:
            raise ValueError("Monthly observation coordinates must be explicit and unique")
    times = _times(observations["T"], "monthly observation T")
    periods = times.to_period("M")
    if periods.has_duplicates or not periods.is_monotonic_increasing:
        raise ValueError("Monthly observations must be sorted with one value per calendar month")
    units = str(observations.attrs.get("units", "")).lower().replace(" ", "").replace("−", "-")
    if aggregation == "sum" and any(rate in units for rate in (
            "/day", "/hour", "/second", "/d", "/h", "/s", "day-1", "day^-1",
            "d-1", "d^-1", "s-1", "s^-1", "h-1", "h^-1")):
        raise ValueError("Convert monthly precipitation rates to monthly totals before seasonal summation")
    obs = observations.transpose("T", "Y", "X")
    return obs, periods


def _aggregate(values, lookup, start, length, aggregation, shape):
    positions = [lookup.get(start + j) for j in range(length)]
    if any(p is None for p in positions):
        return np.full(shape, np.nan)
    part = values[np.asarray(positions)]
    valid = np.isfinite(part).all(axis=0)
    if aggregation == "sum":
        result = part.sum(axis=0)
    elif aggregation == "day_weighted_mean":
        weights = np.asarray([(start + j).days_in_month for j in range(length)], dtype=float)
        result = np.einsum("t,tyx->yx", weights, part) / weights.sum()
    else:
        result = part.mean(axis=0)
    return np.where(valid, result, np.nan)


def aggregate_pbc_target_seasons(monthly_observations, target_times, *,
                                 season_months, aggregation="sum"):
    """Aggregate complete monthly seasons, preserving external seasonal labels.

    Missing constituent months or nonfinite cells produce missing seasonal
    observations. This never treats an incomplete season as a shorter sum.
    ``mean`` means an arithmetic mean of monthly means; choose
    ``day_weighted_mean`` for the mean over all days of the season (including
    leap-year February), matching day-weighted ERA5 seasonal products.
    """
    months = _season(season_months)
    times = _times(target_times, "target_times")
    obs, periods = _monthly(monthly_observations, aggregation)
    intervals = pbc_target_intervals(times, months, initialization_month=months[0])
    values = np.asarray(obs.values, dtype=float)
    lookup = dict(zip(periods, range(len(periods))))
    shape = (obs.sizes["Y"], obs.sizes["X"])
    output = np.asarray([_aggregate(values, lookup, pd.Period(pd.Timestamp(t), freq="M"),
                                   len(months), aggregation, shape)
                         for t in intervals.target_start.values])
    result = xr.DataArray(output, dims=("T", "Y", "X"),
                          coords={"T": times.values, "Y": obs.Y, "X": obs.X},
                          name=obs.name or "seasonal_observations", attrs=dict(obs.attrs))
    result.attrs.update(season_months=",".join(map(str, months)),
                        aggregation=aggregation,
                        missing_month_policy="any missing constituent gives a missing season")
    return result


def _quantiles(values, min_count):
    if len(values) == 0:
        raise ValueError("Internal empty climatology")
    values = np.asarray(values)
    count = np.isfinite(values).sum(axis=0)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", RuntimeWarning)
        q = np.nanquantile(values, CDF_LEVELS, axis=0)
    return np.where(count[None] >= min_count, q, np.nan), count


def _threshold_values(thresholds, obs):
    if not isinstance(thresholds, xr.DataArray):
        raise ValueError("target_thresholds must be a DataArray")
    dimension = next((d for d in ("cdf_threshold", "quantile") if d in thresholds.dims), None)
    if dimension is None or set(thresholds.dims) != {dimension, "Y", "X"} or thresholds.sizes[dimension] != 2:
        raise ValueError("target_thresholds must have dimensions quantile(2),Y,X or cdf_threshold(2),Y,X")
    for dim in ("Y", "X"):
        if not thresholds.get_index(dim).equals(obs.get_index(dim)):
            raise ValueError("target_thresholds and observations must have exactly matching grids")
    if dimension in thresholds.coords:
        levels = np.asarray(thresholds[dimension], dtype=float)
        if not (np.allclose(levels, CDF_LEVELS) or np.allclose(levels, [.33, .67])):
            raise ValueError("target_thresholds must be ordered lower then upper terciles")
    values = np.asarray(thresholds.transpose(dimension, "Y", "X"), dtype=float)
    if np.any(np.isfinite(values).all(axis=0) & (values[0] > values[1])):
        raise ValueError("Lower target threshold cannot exceed upper target threshold")
    return values


def build_pbc_seasonal_features(monthly_observations, target_times, *,
        initialization_month=None, season_months, target_thresholds=None,
        aggregation="sum", observation_lag_months=0,
        persistence_lags=4, persistence_step_months=1,
        climatology_years=(1991, 2020), climatology_min_samples=8,
        climatology_window_years=20, issuance_times=None):
    """Build as-of seasonal persistence indicators and rolling climatology.

    ``lag=1`` is the latest fully observed target-length window. Each further
    window ends ``persistence_step_months`` earlier. The default four windows
    for May issuance and a 3-month target are FMA, JFM, DJF and NDJ. Use
    ``persistence_lags=2, persistence_step_months=3`` for the original disjoint
    FMA/NDJ configuration. Extra observation latency moves all windows back.

    Lag thresholds use the requested climatology years, but only windows
    fully available by that sample's issuance, excluding the queried window
    itself.  No insufficient reference is replaced with a future climatology.
    The climatology feature uses the preceding ``climatology_window_years``
    matching target seasons and fixed target-event thresholds. Missing local
    indicators are filled with k/3, as in the article, but at least
    ``climatology_min_samples`` real observations per cell remain required.
    """
    months = _season(season_months)
    length = len(months)
    lag_months = _integer(observation_lag_months, "observation_lag_months")
    n_lags = _integer(persistence_lags, "persistence_lags", 1)
    step = _integer(persistence_step_months, "persistence_step_months", 1)
    min_count = _integer(climatology_min_samples, "climatology_min_samples", 1)
    window = _integer(climatology_window_years, "climatology_window_years", 1)
    if len(climatology_years) != 2:
        raise ValueError("climatology_years must contain inclusive start and end years")
    clim_start, clim_end = map(int, climatology_years)
    if clim_start > clim_end:
        raise ValueError("climatology_years bounds are reversed")
    times = _times(target_times, "target_times")
    if len(set(times.month)) != 1:
        raise ValueError("Each PBC call must use one common target-season label month")
    obs, periods = _monthly(monthly_observations, aggregation)
    intervals = pbc_target_intervals(times, months, initialization_month, issuance_times)
    starts = [pd.Period(pd.Timestamp(t), freq="M") for t in intervals.target_start.values]
    issues = [pd.Period(pd.Timestamp(t), freq="M") for t in intervals.issuance_date.values]
    values = np.asarray(obs.values, dtype=float)
    lookup = dict(zip(periods, range(len(periods))))
    shape = (obs.sizes["Y"], obs.sizes["X"])
    cache = {}

    def aggregate(start):
        if start not in cache:
            cache[start] = _aggregate(values, lookup, start, length, aggregation, shape)
        return cache[start]

    if target_thresholds is None:
        labels = pd.DatetimeIndex([pd.Timestamp(year=y, month=times[0].month, day=1)
                                   for y in range(clim_start, clim_end + 1)])
        reference_starts = [t.to_period("M") - months.index(t.month) for t in labels]
        target_q, target_count = _quantiles([aggregate(s) for s in reference_starts], min_count)
    else:
        target_q = _threshold_values(target_thresholds, obs)
        target_count = None
    n = len(times)
    lag_values = np.full((n, n_lags, *shape), np.nan)
    lag_indicators = np.full((2, n, n_lags, *shape), np.nan)
    lag_thresholds = np.full((2, n, n_lags, *shape), np.nan)
    lag_counts = np.zeros((n, n_lags, *shape), dtype=np.int64)
    climatology = np.full((2, n, *shape), np.nan)
    climatology_count = np.zeros((n, *shape), dtype=np.int64)
    lag_starts = np.empty((n, n_lags), dtype="datetime64[ns]")
    lag_ends = np.empty_like(lag_starts)
    known_ends = np.empty(n, dtype="datetime64[ns]")

    for i, (issue, target_start, label) in enumerate(zip(issues, starts, times)):
        known = issue - 1 - lag_months
        known_ends[i] = _date(known)
        for lag in range(n_lags):
            end = known - lag * step
            start = end - length + 1
            lag_starts[i, lag], lag_ends[i, lag] = _date(start), _date(end)
            current = aggregate(start)
            lag_values[i, lag] = current
            # Calendar reference years name the start of each lag window.
            reference_starts = [pd.Period(year=y, month=start.month, freq="M")
                                for y in range(clim_start, clim_end + 1)]
            reference_starts = [s for s in reference_starts
                                if s + length - 1 <= known and s != start]
            if reference_starts:
                q, count = _quantiles([aggregate(s) for s in reference_starts], min_count)
                lag_thresholds[:, i, lag] = q
                lag_counts[i, lag] = count
                valid = np.isfinite(current)[None] & np.isfinite(q)
                lag_indicators[:, i, lag] = np.where(valid, current[None] <= q, np.nan)
        # Use matching target seasons from preceding label-years, never a
        # not-yet-complete prior season when a long lead crosses New Year.
        historical_starts = [target_start - 12 * delta for delta in range(1, window + 1)]
        historical_starts = [s for s in historical_starts if s + length - 1 <= known]
        if historical_starts:
            historical = np.asarray([aggregate(s) for s in historical_starts])
            valid = np.isfinite(historical)
            count = valid.sum(axis=0)
            climatology_count[i] = count
            valid_q = np.isfinite(target_q)
            indicators = np.where(valid[None] & valid_q[:, None],
                                  historical[None] <= target_q[:, None], np.nan)
            filled = np.where(np.isfinite(indicators), indicators,
                              CDF_LEVELS[:, None, None, None])
            climatology[:, i] = np.where((count >= min_count)[None] & valid_q,
                                         filled.mean(axis=1), np.nan)

    coords = {"cdf_threshold": CDF_LEVELS, "T": times.values,
              "lag": np.arange(1, n_lags + 1), "Y": obs.Y, "X": obs.X}
    result = xr.Dataset({
        "lag_indicators": (("cdf_threshold", "T", "lag", "Y", "X"), lag_indicators),
        "climatology": (("cdf_threshold", "T", "Y", "X"), climatology),
        "lag_values": (("T", "lag", "Y", "X"), lag_values),
        "lag_thresholds": (("cdf_threshold", "T", "lag", "Y", "X"), lag_thresholds),
        "lag_reference_count": (("T", "lag", "Y", "X"), lag_counts),
        "climatology_reference_count": (("T", "Y", "X"), climatology_count),
        "target_thresholds": (("cdf_threshold", "Y", "X"), target_q),
        "lag_start": (("T", "lag"), lag_starts),
        "lag_end": (("T", "lag"), lag_ends),
        "known_end": ("T", known_ends),
    }, coords=coords)
    result.update(intervals)
    if target_count is not None:
        result["target_reference_count"] = (("Y", "X"), target_count)
    result.attrs.update(
        adaptation="seasonal PBC: configurable preceding observed target-length windows",
        season_months=",".join(map(str, months)), aggregation=aggregation,
        observation_lag_months=lag_months,
        persistence_lags=n_lags, persistence_step_months=step,
        persistence_windows_overlap=int(n_lags > 1 and step < length),
        persistence_window_length_months=length,
        climatology_start=clim_start, climatology_end=clim_end,
        climatology_min_samples=min_count, climatology_window_years=window,
        lag_threshold_reference="matching lag-calendar seasons from configured years; available at issuance; queried lag excluded",
        target_threshold_reference="fixed observed target-season climatology; identical fitting and verification events",
        category_definition="cumulative indicators: y <= q1; y <= q2",
        missing_month_policy="any missing constituent gives a missing seasonal observation",
        rolling_climatology_missing="k/3 imputation within previous-year slots; minimum real count enforced",
        endpoint_convention="inclusive calendar month labels")
    result.lag_values.attrs.update(obs.attrs)
    result.target_thresholds.attrs.update(obs.attrs)
    return result
