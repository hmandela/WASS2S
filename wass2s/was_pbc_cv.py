"""Chronological seasonal PBC on existing probabilistic forecast archives.

This module never refits the upstream forecasting model. Validation is therefore
conditional on that archive when it contains retrospective cross-validation
predictions. It is not a substitute for fully nested validation of both stages.
"""
import numpy as np
import pandas as pd
import xarray as xr

from .was_probability_reference import climatological_terciles


def _dates(da, label):
    if not isinstance(da, xr.DataArray) or 'T' not in da.dims:
        raise TypeError(f'{label} must be a DataArray with dimension T')
    dates = pd.DatetimeIndex(da['T'].values)
    if dates.has_duplicates or not dates.is_monotonic_increasing or dates.hasnans:
        raise ValueError(f'{label} T dates must be unique, sorted and finite')
    return dates


def _archive_provenance(prob, intervals, provenance):
    if provenance == 'retrospective_oof':
        return
    if provenance != 'asof_archive':
        raise ValueError('base_provenance must be retrospective_oof or asof_archive')
    if 'base_training_end' not in prob.coords or prob['base_training_end'].dims != ('T',):
        raise ValueError('asof_archive requires a base_training_end(T) coordinate for every forecast')
    end = pd.DatetimeIndex(prob['base_training_end'].values)
    issue = pd.DatetimeIndex(intervals['issuance_date'].values)
    if end.hasnans or np.any(end >= issue):
        raise ValueError('base_training_end must precede the corresponding issuance date')


def _training_indices(intervals, query_issue, training_years, observation_lag_months):
    # Month starts are used throughout: an end month is fully known only once
    # the following month starts, plus the explicitly configured publication lag.
    end = pd.DatetimeIndex(intervals['target_end'].values).to_period('M')
    issue = pd.Timestamp(query_issue)
    available = (end + 1 + int(observation_lag_months)).to_timestamp()
    lower = issue - pd.DateOffset(years=int(training_years))
    return np.flatnonzero((available <= issue) & (available > lower))


def _feature_slice(features, dates, name):
    return None if features is None else features[name].sel(T=dates)


def run_pbc_seasonal(hindcast_prob, forecast_prob, observations,
                     monthly_observations=None, *, initialization_month,
                     season_months, clim_year_start, clim_year_end,
                     forecast_year=None, aggregation='sum', alpha=1.,
                     min_samples=8, training_years=20,
                     observation_lag_months=0, persistence=True,
                     persistence_lags=4, persistence_step_months=1,
                     base_provenance='retrospective_oof'):
    """Return rolling PBC hindcasts and an operational corrected forecast.

    Inputs use PB/PN/PA probabilities, actual seasonal labels ``T`` and a common
    grid. The observed climatology defines fixed events for all years. PBC fits
    only on seasons fully observable by each issuance; the initial warm-up is
    retained as raw probabilities and explicitly excluded from score masks.

    Persistence uses observed target-length windows: by default four windows
    one month apart. A May issuance with zero extra latency therefore uses
    FMA, JFM, DJF and NDJ. Set ``persistence_lags=2`` and
    ``persistence_step_months=3`` to reproduce the initial two-trimester layout.
    These windows are predictors of each annual case, not extra training rows.

    ``asof_archive`` checks declared upstream training cutoffs. The default
    ``retrospective_oof`` labels validation as conditional, because historical
    out-of-fold base predictions may depend on later observations.
    """
    from .was_pbc import WAS_PBC, _probabilities
    from .was_pbc_data import (pbc_target_intervals, build_pbc_seasonal_features,
                               aggregate_pbc_target_seasons, _integer)

    if int(training_years) != training_years or training_years < 1:
        raise ValueError('training_years must be a positive integer')
    if int(observation_lag_months) != observation_lag_months or observation_lag_months < 0:
        raise ValueError('observation_lag_months must be a nonnegative integer')
    persistence_lags = _integer(persistence_lags, 'persistence_lags', 1)
    persistence_step_months = _integer(persistence_step_months, 'persistence_step_months', 1)
    if persistence and monthly_observations is None:
        raise ValueError('Seasonal Persistence++ requires monthly observations; use persistence=False for Debias++ only')
    hdates = _dates(hindcast_prob, 'hindcast_prob')
    fdates = _dates(forecast_prob, 'forecast_prob')
    odates = _dates(observations, 'observations')
    if not len(hdates) or not len(fdates):
        raise ValueError('Nonempty hindcast and forecast archives are required')
    if np.intersect1d(hdates.values, fdates.values).size:
        raise ValueError('Hindcast and operational forecast target dates must be distinct')
    if forecast_year is not None and np.any(fdates.year != int(forecast_year)):
        raise ValueError('Forecast target dates disagree with forecast_year; dates are never reassigned')
    months = np.unique(np.concatenate([hdates.month, fdates.month, odates.month]))
    if len(months) != 1 or hdates.year.has_duplicates or fdates.year.has_duplicates or odates.year.has_duplicates:
        raise ValueError('One common seasonal label month and one target per year are required')
    if set(observations.dims) != {'T', 'Y', 'X'}:
        raise ValueError('observations must have exactly T,Y,X dimensions')
    for prob in (hindcast_prob, forecast_prob):
        for dim in ('Y', 'X'):
            if not prob[dim].equals(observations[dim]):
                raise ValueError('PBC requires identical observation and forecast grids')
        declared = prob.attrs.get('tercile_reference')
        if declared is not None and declared != 'climatology':
            raise ValueError('PBC requires fixed climatological events; fold-specific base terciles cannot be relabelled')
        for key, requested in (('climatology_start', clim_year_start), ('climatology_end', clim_year_end)):
            if key in prob.attrs and int(prob.attrs[key]) != int(requested):
                raise ValueError('Base probability climatology disagrees with the requested PBC reference')
    try:
        truth = observations.sel(T=hdates)
    except KeyError as exc:
        raise ValueError('Every hindcast target date must exist in observations; no year-only realignment') from exc
    thresholds, reference = climatological_terciles(
        observations, clim_year_start, clim_year_end, min_count=min_samples)
    thresholds = thresholds.rename(quantile='cdf_threshold').assign_coords(cdf_threshold=[1/3, 2/3])
    hi = pbc_target_intervals(hdates, season_months, initialization_month=initialization_month)
    fi = pbc_target_intervals(fdates, season_months, initialization_month=initialization_month)
    pbc_target_intervals(odates, season_months, initialization_month=initialization_month)
    _archive_provenance(hindcast_prob, hi, base_provenance)
    _archive_provenance(forecast_prob, fi, base_provenance)
    base_h_end = hindcast_prob.coords.get('base_training_end')
    base_f_end = forecast_prob.coords.get('base_training_end')
    # Validate even warm-up distributions and remove partial category vectors.
    # Auxiliary per-archive coordinates are recorded under distinct names below.
    hindcast_prob = _probabilities(hindcast_prob).reset_coords(drop=True)
    forecast_prob = _probabilities(forecast_prob).reset_coords(drop=True)

    features = None
    if persistence:
        # A different observational product cannot silently supply persistence
        # and event labels. Compare the actual monthly-derived seasonal truth.
        monthly_truth = aggregate_pbc_target_seasons(monthly_observations, hdates,
            season_months=season_months, aggregation=aggregation)
        if not monthly_truth['Y'].equals(truth['Y']) or not monthly_truth['X'].equals(truth['X']):
            raise ValueError('Monthly and seasonal observation grids disagree')
        units_m = str(monthly_observations.attrs.get('units', '')).strip().lower()
        units_s = str(observations.attrs.get('units', '')).strip().lower()
        if units_m and units_s and units_m != units_s:
            raise ValueError('Monthly and seasonal observation units disagree; convert explicitly')
        observed = np.asarray(truth.transpose('T', 'Y', 'X').values)
        rebuilt = np.asarray(monthly_truth.transpose('T', 'Y', 'X').values)
        if np.any(np.isfinite(observed) & ~np.isfinite(rebuilt)) or not np.allclose(
                observed[np.isfinite(observed)], rebuilt[np.isfinite(observed)], rtol=1e-5, atol=1e-6):
            raise ValueError('Monthly aggregates disagree with seasonal verification observations')
        all_dates = hdates.append(fdates).sort_values()
        features = build_pbc_seasonal_features(monthly_observations, all_dates,
            initialization_month=initialization_month, season_months=season_months,
            target_thresholds=thresholds, aggregation=aggregation,
            observation_lag_months=observation_lag_months,
            climatology_years=(clim_year_start, clim_year_end),
            climatology_min_samples=min_samples,
            climatology_window_years=training_years,
            persistence_lags=persistence_lags,
            persistence_step_months=persistence_step_months)

    def fit_at(issue):
        idx = _training_indices(hi, issue, training_years, observation_lag_months)
        dates = hdates[idx]
        train = hindcast_prob.isel(T=idx)
        obs_train = truth.isel(T=idx)
        valid = (train.notnull().all('probability') & np.isfinite(obs_train)
                 & thresholds.notnull().all('cdf_threshold'))
        count = valid.sum('T').astype('int32')
        model = None
        if len(idx) >= min_samples:
            model = WAS_PBC(alpha=alpha, min_samples=min_samples, persistence=persistence)
            model.fit(train, obs_train, thresholds,
                lag_indicators=_feature_slice(features, dates, 'lag_indicators'),
                climatology=_feature_slice(features, dates, 'climatology'))
        return model, count, idx

    corrected, counts, eligible, last_available, persistence_fitted = [], [], [], [], []
    for i, date in enumerate(hdates):
        query = hindcast_prob.isel(T=slice(i, i+1))
        issue = pd.Timestamp(hi['issuance_date'].isel(T=i).item())
        model, count, idx = fit_at(issue)
        pred = query.copy() if model is None else model.predict(query,
            lag_indicators=_feature_slice(features, [date], 'lag_indicators'),
            climatology=_feature_slice(features, [date], 'climatology'))
        corrected.append(pred)
        counts.append(count.expand_dims(T=[date]))
        used = (xr.zeros_like(thresholds, dtype='int8') if model is None
                else model.state_['persistence_used'])
        persistence_fitted.append(used.expand_dims(T=[date]))
        eligible.append(int(len(idx) >= min_samples))
        last_available.append(hi['target_end'].isel(T=int(idx[-1])).values if len(idx) else np.datetime64('NaT'))
    future, future_counts, future_eligible = [], [], []
    final_model = None
    for i, date in enumerate(fdates):
        query = forecast_prob.isel(T=slice(i, i+1))
        issue = pd.Timestamp(fi['issuance_date'].isel(T=i).item())
        model, count, idx = fit_at(issue)
        if model is None:
            raise ValueError(f'Insufficient fully observable hindcast seasons for operational PBC at {issue.date()}: {len(idx)} < {min_samples}')
        final_model = model
        future.append(model.predict(query,
            lag_indicators=_feature_slice(features, [date], 'lag_indicators'),
            climatology=_feature_slice(features, [date], 'climatology')))
        future_counts.append(count.expand_dims(T=[date]))
        future_eligible.append(int(len(idx) >= min_samples))

    hp = xr.concat(corrected, dim='T').transpose('probability', 'T', 'Y', 'X')
    fp = xr.concat(future, dim='T').transpose('probability', 'T', 'Y', 'X').rename(T='forecast_time')
    count = xr.concat(counts, dim='T')
    eligible_cell = ((count >= min_samples) & np.isfinite(truth)
                     & hp.notnull().all('probability') & hindcast_prob.notnull().all('probability'))
    result = xr.Dataset({
        'hindcast_probability': hp, 'forecast_probability': fp,
        'raw_hindcast_probability': hindcast_prob,
        'raw_forecast_probability': forecast_prob.rename(T='forecast_time'),
        'training_count': count,
        'forecast_training_count': xr.concat(future_counts, dim='T').rename(T='forecast_time'),
        'score_eligible': xr.DataArray(eligible, dims='T', coords={'T': hdates}).astype('int8'),
        'score_eligible_cell': eligible_cell.astype('int8'),
        'persistence_fitted': xr.concat(persistence_fitted, dim='T'),
        'thresholds': thresholds, 'reference_probabilities': reference,
        'issuance_date': hi['issuance_date'],
        'forecast_issuance_date': fi['issuance_date'].rename(T='forecast_time'),
        'target_start': hi['target_start'], 'target_end': hi['target_end'],
        'last_training_target_end': xr.DataArray(np.array(last_available, dtype='datetime64[ns]'), dims='T', coords={'T': hdates}),
    })
    if base_h_end is not None:
        result['base_hindcast_training_end'] = base_h_end.reset_coords(drop=True)
    if base_f_end is not None:
        result['base_forecast_training_end'] = base_f_end.reset_coords(drop=True).rename(T='forecast_time')
    if features is not None:
        for name in ('lag_values', 'lag_indicators', 'lag_thresholds', 'lag_reference_count',
                     'climatology_reference_count', 'lag_start', 'lag_end', 'climatology', 'known_end'):
            result[name] = features[name].sel(T=hdates)
            result['forecast_' + name] = features[name].sel(T=fdates).rename(T='forecast_time')
    # Store the last operational fit for inspection, without any pickle payload.
    if final_model is not None:
        for name, value in final_model.state_.data_vars.items():
            if 'T' not in value.dims:
                result['fitted_' + name] = value
    scope = ('conditional_postprocessing_on_retrospective_oof' if base_provenance == 'retrospective_oof'
             else 'chronological_postprocessing_on_declared_asof_archive')
    metadata = dict(pbc_schema='update14-seasonal-pbc-sliding-2',
        pbc_method='seasonal_adaptation_of_Guan_et_al_2026',
        pbc_reference='doi:10.1038/s41586-026-11126-x',
        pbc_validation_scope=scope, pbc_base_provenance=base_provenance,
        pbc_training='expanding_or_rolling_fully_observable_previous_seasons',
        pbc_training_years=int(training_years), pbc_min_samples=int(min_samples),
        pbc_alpha=float(alpha), pbc_persistence_requested=int(bool(persistence)),
        pbc_persistence_lags=int(persistence_lags),
        pbc_persistence_step_months=int(persistence_step_months),
        pbc_persistence_windows_overlap=int(persistence_lags > 1 and persistence_step_months < len(season_months)),
        pbc_persistence_source='observed monthly fields aggregated into target-length windows',
        pbc_variant='two_branch_seasonal_pbc' if persistence else 'debias_only',
        pbc_season_months=','.join(map(str, season_months)),
        pbc_initialization_month=int(initialization_month), pbc_aggregation=aggregation,
        pbc_base_models_refitted=0, pbc_nested_pipeline_validation=0,
        pbc_base_event_check='declared reference metadata checked when present; caller must ensure missing metadata uses the same fixed events',
        pbc_observation_lag_months=int(observation_lag_months),
        tercile_reference='climatology', climatology_start=int(clim_year_start),
        climatology_end=int(clim_year_end), tercile_quantile_levels='1/3,2/3',
        category_definition='PB: y <= low; PN: low < y <= high; PA: y > high',
        pbc_warmup='raw fallback, excluded from paired verification',
        probability_calibration_effective='seasonal_pbc_native_cdf_correction')
    result.attrs.update(metadata)
    for name in ('hindcast_probability', 'forecast_probability'):
        result[name].attrs.update(metadata)
    return result
