"""Calendar-safe seasonal sequence construction for TelNet.

The last fully observed seasonal state is followed by one-month shifts. For
example, a JFM state available on 1 April is followed by FMA, MAM, AMJ, MJJ,
JJA and JAS. Some first targets therefore contain already observed months;
``target_fully_future`` identifies these partial-season forecasts explicitly.
No observed value after ``known_end`` enters a predictor tensor.

All fields remain in their physical units. Standardisation/detrending must be
fitted by the estimator on its allowed training samples, never in this module.
"""
from __future__ import annotations

from typing import Iterable

import numpy as np
import pandas as pd
import xarray as xr


__all__ = ["build_telnet_samples", "purged_train_indices", "grouped_year_splits"]


def _positive_integer(value, name, minimum=1):
    if isinstance(value, (bool, np.bool_)) or int(value) != value or int(value) < minimum:
        raise ValueError(f"{name} must be an integer >= {minimum}")
    return int(value)


def _month_index(obj, name):
    if "T" not in obj.dims or "T" not in obj.coords:
        raise ValueError(f"{name} needs an explicit monthly T coordinate")
    try:
        years = np.asarray(obj["T"].dt.year, dtype=int)
        months = np.asarray(obj["T"].dt.month, dtype=int)
    except (AttributeError, TypeError, ValueError) as exc:
        raise ValueError(f"{name} T must contain calendar dates") from exc
    index = pd.PeriodIndex([pd.Period(year=int(y), month=int(m), freq="M")
                            for y, m in zip(years, months)], freq="M")
    if index.has_duplicates:
        raise ValueError(f"{name} must contain exactly one value per calendar month")
    if not index.is_monotonic_increasing:
        raise ValueError(f"{name} monthly dates must be sorted")
    if len(index) == 0:
        raise ValueError(f"{name} is empty")
    return index


def _observed_monthly(observations):
    if not isinstance(observations, xr.DataArray):
        raise TypeError("observations must be a DataArray with dimensions T,Y,X")
    if "M" in observations.dims:
        if observations.sizes["M"] != 1:
            raise ValueError("The observation member dimension M must be singleton")
        observations = observations.isel(M=0, drop=True)
    if set(observations.dims) != {"T", "Y", "X"}:
        raise ValueError("observations must have exactly T,Y,X dimensions")
    for dim in ("Y", "X"):
        if dim not in observations.coords or observations.get_index(dim).has_duplicates:
            raise ValueError("Observation spatial coordinates must be explicit and unique")
    index = _month_index(observations, "observations")
    return observations.transpose("T", "Y", "X"), index


def _covariate_monthly(covariates):
    if isinstance(covariates, xr.Dataset):
        if not covariates.data_vars:
            raise ValueError("At least one covariate is required")
        names = list(covariates.data_vars)
        if any(set(covariates[name].dims) != {"T"} for name in names):
            raise ValueError("Dataset covariates must each have only the monthly T dimension")
        arr = covariates[names].to_array(dim="feature").transpose("T", "feature")
    elif isinstance(covariates, xr.DataArray):
        arr = covariates
        if set(arr.dims) != {"T", "feature"}:
            raise ValueError("Covariates must have dimensions T,feature or be a Dataset of monthly indices")
        if "feature" not in arr.coords:
            arr = arr.assign_coords(feature=[f"feature_{i}" for i in range(arr.sizes["feature"])])
        arr = arr.transpose("T", "feature")
    else:
        raise TypeError("covariates must be a monthly DataArray or Dataset")
    if arr.sizes["feature"] < 1 or arr.get_index("feature").has_duplicates:
        raise ValueError("Covariate feature labels must be nonempty and unique")
    return arr, _month_index(arr, "covariates")


def _seasonal_values(values, lookup, start, length, operation, shape):
    positions = [lookup.get(start + j) for j in range(length)]
    if any(pos is None for pos in positions):
        return np.full(shape, np.nan, dtype=float)
    part = values[np.asarray(positions, dtype=int)]
    # A missing constituent month cannot silently become a shorter season.
    valid = np.isfinite(part).all(axis=0)
    out = part.sum(axis=0) if operation == "sum" else part.mean(axis=0)
    return np.where(valid, out, np.nan)


def _date(month):
    return month.to_timestamp().to_datetime64().astype("datetime64[ns]")


def build_telnet_samples(observations, covariates, *, history_length=4,
                         forecast_length=6, season_length=3, aggregation="sum",
                         observation_lag_months=0, initialization_months=None,
                         availability_end=None, issuance_dates=None):
    """Build raw lagged and future overlapping seasonal sequences.

    Parameters
    ----------
    observations : DataArray (T,Y,X)
        Monthly physical totals for ``aggregation='sum'`` (e.g. rainfall), or
        monthly means for ``'mean'`` (e.g. temperature). This function does not
        convert a daily rate into a monthly total.
    covariates : DataArray (T,feature) or Dataset
        Observed monthly climate indices. They are averaged over the same
        historical seasonal windows as the lagged predictand.
    history_length, forecast_length, season_length : int
        Counts of overlapping seasonal states, predicted states and months in
        each seasonal state, respectively.
    observation_lag_months : int
        Additional complete months unavailable after the month immediately
        before issuance. Zero means an issue on 1 April may use March.
    initialization_months : iterable[int], optional
        Months (1..12) of issuance to retain. None retains all months.
    availability_end : date-like, optional
        Last monthly observation that is known, not a submonthly availability
        timestamp. Predictor windows end no later than this month. Targets
        remain independent verification labels and can extend beyond it.
    issuance_dates : iterable[date-like], optional
        Explicit requested issuance months, including an operational future
        month. Dates are normalised to month starts, sorted, and must be unique.
        If omitted, complete-history issues through the last observed month
        plus one plus ``observation_lag_months`` are constructed.

    Returns
    -------
    Dataset
        ``history(sample,lag,Y,X)``, ``covariates(sample,lag,feature)``,
        ``targets(sample,lead,Y,X)`` and their calendar/provenance fields.
        ``history_month`` and ``target_month`` are zero-based month numbers of
        the seasonal centre used by the network's month embeddings.
        ``target_month_date`` is the WASS2S seasonal T (centre-month first day).
        ``known_end`` and interval ``*_start/end`` fields are month labels;
        interval endpoints are inclusive. Targets outside the archive are NaN.
        Target label windows overlap and must be purged before temporal CV.
    """
    history_length = _positive_integer(history_length, "history_length")
    forecast_length = _positive_integer(forecast_length, "forecast_length")
    season_length = _positive_integer(season_length, "season_length")
    lag_months = _positive_integer(observation_lag_months, "observation_lag_months", 0)
    aggregation = str(aggregation).lower()
    if aggregation not in {"sum", "mean"}:
        raise ValueError("aggregation must be sum or mean")
    obs, obs_months = _observed_monthly(observations)
    units = str(obs.attrs.get("units", "")).lower().replace(" ", "").replace("−", "-")
    if aggregation == "sum" and any(rate in units for rate in (
            "/day", "/hour", "/second", "/d", "/h", "/s", "day-1", "day^-1",
            "d-1", "d^-1", "s-1", "s^-1", "h-1", "h^-1")):
        raise ValueError("Monthly precipitation rates must first be converted to monthly totals; summing rate values is invalid")
    cov, cov_months = _covariate_monthly(covariates)
    cutoff = None if availability_end is None else pd.Period(pd.Timestamp(availability_end), freq="M")
    if initialization_months is None:
        init_months = set(range(1, 13))
    else:
        init_months = {_positive_integer(v, "initialization month") for v in initialization_months}
        if not init_months or not init_months <= set(range(1, 13)):
            raise ValueError("initialization_months must contain month numbers in 1..12")
    first_known_end = obs_months[0] + history_length + season_length - 2
    if issuance_dates is None:
        first = first_known_end + 1 + lag_months
        last = obs_months[-1] + 1 + lag_months
        if cutoff is not None:
            last = min(last, cutoff + 1 + lag_months)
        issues = pd.period_range(first, last, freq="M") if first <= last else pd.PeriodIndex([], freq="M")
    else:
        requested = ([issuance_dates] if isinstance(issuance_dates, (str, pd.Timestamp, np.datetime64))
                     else list(issuance_dates))
        issues = pd.PeriodIndex([pd.Period(pd.Timestamp(t), freq="M") for t in requested], freq="M")
        if issues.has_duplicates:
            raise ValueError("issuance_dates must identify unique issuance months")
        issues = issues.sort_values()
    issues = pd.PeriodIndex([t for t in issues if t.month in init_months], freq="M")
    if len(issues) == 0:
        raise ValueError("No TelNet samples remain for the requested history and issuance months")
    obs_values = np.asarray(obs.values, dtype=float)
    cov_values = np.asarray(cov.values, dtype=float)
    obs_lookup = {month: i for i, month in enumerate(obs_months)}
    cov_lookup = {month: i for i, month in enumerate(cov_months)}
    ny, nx, nf = obs.sizes["Y"], obs.sizes["X"], cov.sizes["feature"]
    history = np.full((len(issues), history_length, ny, nx), np.nan)
    cov_history = np.full((len(issues), history_length, nf), np.nan)
    targets = np.full((len(issues), forecast_length, ny, nx), np.nan)
    hs = np.empty((len(issues), history_length), dtype="datetime64[ns]")
    he = np.empty_like(hs); hm = np.empty((len(issues), history_length), dtype=np.int64)
    ts = np.empty((len(issues), forecast_length), dtype="datetime64[ns]")
    te = np.empty_like(ts); tmdate = np.empty_like(ts)
    tm = np.empty((len(issues), forecast_length), dtype=np.int64)
    known_ends = np.empty(len(issues), dtype="datetime64[ns]")
    fully_future = np.empty((len(issues), forecast_length), dtype=bool)
    centre_offset = (season_length - 1) // 2
    for i, issue in enumerate(issues):
        known = issue - 1 - lag_months
        if cutoff is not None:
            known = min(known, cutoff)
        known_ends[i] = _date(known)
        latest_start = known - season_length + 1
        for j in range(history_length):
            start = latest_start - history_length + 1 + j
            end = start + season_length - 1
            hs[i,j], he[i,j] = _date(start), _date(end)
            hm[i,j] = (start + centre_offset).month - 1
            history[i,j] = _seasonal_values(obs_values,obs_lookup,start,season_length,aggregation,(ny,nx))
            cov_history[i,j] = _seasonal_values(cov_values,cov_lookup,start,season_length,"mean",(nf,))
        for j in range(forecast_length):
            start = latest_start + j + 1
            end = start + season_length - 1
            ts[i,j], te[i,j] = _date(start), _date(end)
            tmdate[i,j] = _date(start + centre_offset)
            tm[i,j] = (start + centre_offset).month - 1
            fully_future[i,j] = start >= issue
            targets[i,j] = _seasonal_values(obs_values,obs_lookup,start,season_length,aggregation,(ny,nx))
    coords = {"sample": np.asarray([_date(t) for t in issues]),
              "lag": np.arange(-history_length + 1, 1),
              "lead": np.arange(1, forecast_length + 1),
              "Y": obs["Y"], "X": obs["X"], "feature": cov["feature"]}
    result = xr.Dataset({
        "history": (("sample","lag","Y","X"),history),
        "covariates": (("sample","lag","feature"),cov_history),
        "targets": (("sample","lead","Y","X"),targets),
        "history_start": (("sample","lag"),hs),
        "history_end": (("sample","lag"),he),
        "history_month": (("sample","lag"),hm),
        "target_start": (("sample","lead"),ts),
        "target_end": (("sample","lead"),te),
        "target_month": (("sample","lead"),tm),
        "target_month_date": (("sample","lead"),tmdate),
        "target_fully_future": (("sample","lead"),fully_future),
        "known_end": (("sample",),known_ends),
    }, coords=coords)
    result.attrs.update(season_length=season_length, history_length=history_length,
        forecast_length=forecast_length, aggregation=aggregation,
        observation_lag_months=lag_months,
        observation_availability_end="none" if cutoff is None else str(cutoff),
        temporal_definition="monthly overlapping seasonal windows; inclusive interval endpoints",
        covariate_aggregation="seasonal arithmetic mean",
        target_timestamp="first day of seasonal centre month",
        missing_month_policy="any missing constituent month gives a missing seasonal value",
        future_predictor_policy="observed inputs end at known_end; never at a forecast target date")
    result["history"].attrs.update(obs.attrs)
    result["targets"].attrs.update(obs.attrs)
    return result


def _positions(values, size, name):
    arr = np.asarray(values)
    if arr.ndim != 1 or not np.issubdtype(arr.dtype, np.integer):
        raise ValueError(f"{name} must be one-dimensional integer positions")
    if np.any((arr < 0) | (arr >= size)) or len(np.unique(arr)) != len(arr):
        raise ValueError(f"{name} must be unique valid sample positions")
    return arr.astype(int)


def purged_train_indices(samples, test_indices, candidate_indices=None, *,
                         embargo_months=0, purge_history=True, forward_only=False):
    """Remove samples sharing target months with verification windows.

    If ``purge_history`` is true, an observation used as an input in training
    may not belong to any held-out target interval either. The exclusion is
    based on actual calendar months, including cross-year seasons. It is more
    conservative than merely matching the central seasonal timestamp.
    ``forward_only`` restricts training to earlier issuance dates; non-forward
    blocked CV is a retrospective estimate, not historical real-time skill.
    """
    if "sample" not in samples.dims:
        raise ValueError("samples must be a TelNet Dataset with sample dimension")
    embargo = _positive_integer(embargo_months,"embargo_months",0)
    n = samples.sizes["sample"]
    te = _positions(test_indices,n,"test_indices")
    if not len(te):
        raise ValueError("At least one test sample is required")
    candidates = np.arange(n) if candidate_indices is None else _positions(candidate_indices,n,"candidate_indices")
    candidates = candidates[~np.isin(candidates,te)]
    tstart = samples["target_start"].values.astype("datetime64[M]").astype(np.int64).min(axis=1)
    tend = samples["target_end"].values.astype("datetime64[M]").astype(np.int64).max(axis=1)
    hstart = samples["history_start"].values.astype("datetime64[M]").astype(np.int64).min(axis=1)
    starts = np.minimum(hstart,tstart) if purge_history else tstart
    blocked_start = tstart[te] - embargo
    blocked_end = tend[te] + embargo
    overlap = ((starts[candidates,None] <= blocked_end[None,:]) &
               (tend[candidates,None] >= blocked_start[None,:])).any(axis=1)
    keep = ~overlap
    if forward_only:
        keep &= samples["sample"].values[candidates] < samples["sample"].values[te].min()
    return candidates[keep]


def grouped_year_splits(samples, n_splits=None, *, gap_years=0,
                        embargo_months=0, purge_history=True, forward_only=False):
    """Yield purged train/test positions with whole issuance years grouped.

    ``n_splits=None`` leaves out each year. Otherwise consecutive distinct
    years are partitioned into ``n_splits`` blocks, all dates in each block
    being held out together. Empty training partitions are not yielded.
    """
    gap_years = _positive_integer(gap_years,"gap_years",0)
    years = np.asarray(samples["sample"].dt.year,dtype=int)
    unique = np.unique(years)
    if len(unique) < 2:
        raise ValueError("At least two issuance years are required for grouped CV")
    if n_splits is None:
        blocks = [np.asarray([year]) for year in unique]
    else:
        n_splits = _positive_integer(n_splits,"n_splits",2)
        if n_splits > len(unique):
            raise ValueError("n_splits cannot exceed the number of issuance years")
        blocks = np.array_split(unique,n_splits)
    for block in blocks:
        test = np.where(np.isin(years,block))[0]
        distant = (np.abs(years[:,None] - block[None,:]) > gap_years).all(axis=1)
        train = purged_train_indices(samples,test,np.where(distant)[0],
            embargo_months=embargo_months,purge_history=purge_history,forward_only=forward_only)
        if len(train):
            yield train,test
