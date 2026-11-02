"""Optional monthly multiplicative correction of DAILY discharge, v5.1.0.

The default seasonal-aggregate workflow is not replaced.  This module fits
monthly ratios on OBS/model common valid days, propagates FACTORS (not flows)
to HYBAS outlets, then multiplies the extracted daily series.  The existing
climatological HW/MODE catalogue is applied AFTER that multiplication.

Temporal protocols are explicit.  ``leave_month_out`` protects the factor for
one YYYY-MM, not an entire seasonal forecast validation.  Outer validation must
also protect climatological windows, neighbouring dependent dates and station
selection.  Use absolute ``excluded_periods`` and/or ``information_cutoff`` for
such folds, and ``window_protocol='refit_excluding'`` in the high-level builder.
Nothing is executed and no file is opened at import.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Mapping
import calendar
import copy
import warnings

import numpy as np
import pandas as pd
import xarray as xr


@dataclass(frozen=True)
class MonthlyBiasConfig:
    """Controls for monthly linear scaling of nonnegative daily discharge.

    temporal_mode
        ``in_sample``: all eligible calibration data; explicitly potentially
        overlapping. NOT an independent validation when overlap exists.
        ``leave_month_out``: for Jan 1991 use other Januaries only, excluding
        Jan 1991 at ALL donors. Future years remain allowed (retrospective CV).
        ``leave_year_out``: remove the target CALENDAR year at all donors;
        for a season spanning years prefer a single explicit absolute block.
        ``blocked``: fixed calibration minus ``excluded_periods`` everywhere.
        ``expanding``: dates strictly before the target month's start, further
        bounded by ``information_cutoff`` when supplied. Future data forbidden.
    weighting
        ``pooled_days`` is ratio of means on the SAME paired daily sample.
        ``equal_years`` averages the accepted monthly means with equal weights.
        Neither is the mean of instantaneous OBS/model ratios.
    window_source
        ``observations`` (default): apply transferred observed windows after
        correction. ``corrected_glofas``: detect a local climatological regime
        from the corrected daily GloFAS first, then aggregate. A corrected
        climatology is still NOT a forecast of next year's dates.
    window_protocol
        ``fixed_reference`` keeps the original observation-defined catalogue;
        its independence is NOT inferred from the factor validation mode.
        ``refit_excluding`` masks explicit outer exclusions / cutoff before
        regime detection at every donor and every GloFAS fallback outlet.
        A different window is not fitted per application month.
    missing_factor_policy
        nan (default), raw (explicit flagged fallback), or error. No global
        mean factor, epsilon denominator or unannounced clipping is used.
    """
    temporal_mode: str = 'leave_month_out'
    weighting: str = 'pooled_days'
    min_training_years: int = 10
    min_valid_fraction_month: float = 0.80
    min_model_mean: float = 0.01             # same units as Q, usually m3/s
    missing_values: tuple[float, ...] = (-99.0, -999.0)
    excluded_periods: tuple[tuple[str, str], ...] = ()  # inclusive absolute dates
    excluded_stations: tuple[str, ...] = ()
    information_cutoff: str | None = None    # use training dates STRICTLY before
    gap_days: int = 0
    window_source: str = 'observations'        # or corrected_glofas: re-detect AFTER correction
    window_protocol: str = 'fixed_reference'
    missing_factor_policy: str = 'nan'
    factor_bounds: tuple[float, float] | None = None  # reject, do NOT clip
    keep_daily: bool = False

    def validate(self) -> None:
        modes = {'in_sample', 'leave_month_out', 'leave_year_out', 'blocked', 'expanding'}
        if self.temporal_mode not in modes:
            raise ValueError(f'temporal_mode must be one of {sorted(modes)}')
        if self.weighting not in {'pooled_days', 'equal_years'}:
            raise ValueError('weighting: pooled_days or equal_years')
        if (isinstance(self.min_training_years, bool) or
                not isinstance(self.min_training_years, (int, np.integer)) or self.min_training_years < 2):
            raise ValueError('min_training_years must be an integer >= 2')
        if not 0 < self.min_valid_fraction_month <= 1:
            raise ValueError('min_valid_fraction_month must lie in (0,1]')
        if not np.isfinite(self.min_model_mean) or self.min_model_mean <= 0:
            raise ValueError('min_model_mean must be finite and positive')
        if isinstance(self.gap_days, bool) or not isinstance(self.gap_days, (int, np.integer)) or self.gap_days < 0:
            raise ValueError('gap_days must be an integer >= 0')
        if self.missing_factor_policy not in {'nan', 'raw', 'error'}:
            raise ValueError('missing_factor_policy: nan, raw or error')
        if self.window_source not in {'observations', 'corrected_glofas'}:
            raise ValueError('window_source: observations or corrected_glofas')
        if self.window_protocol not in {'fixed_reference', 'refit_excluding'}:
            raise ValueError('window_protocol: fixed_reference or refit_excluding')
        if self.temporal_mode == 'blocked' and not self.excluded_periods:
            raise ValueError('blocked requires explicit excluded_periods (inclusive absolute dates)')
        for pair in self.excluded_periods:
            if len(pair) != 2:
                raise ValueError('Every excluded period must be (start_date, end_date)')
            if _day(pair[0]) > _day(pair[1]):
                raise ValueError('Exclusion start must precede end')
        if isinstance(self.excluded_stations, str):
            raise ValueError('excluded_stations must be a sequence of station names, not one string')
        if self.information_cutoff is not None:
            _day(self.information_cutoff)
        if (self.window_protocol == 'refit_excluding' and
                not self.excluded_periods and self.information_cutoff is None):
            raise ValueError('refit_excluding requires outer excluded_periods or information_cutoff; '
                             'leave_month_out alone protects factors, NOT one global climatology')
        if self.factor_bounds is not None:
            lo, hi = self.factor_bounds
            if not np.isfinite([lo, hi]).all() or not 0 <= lo < hi:
                raise ValueError('factor_bounds must be finite and 0 <= lower < upper')


def _day(value: Any) -> pd.Timestamp:
    out = pd.Timestamp(value)
    if pd.isna(out) or out.tz is not None or out != out.normalize():
        raise ValueError('Use finite timezone-naive Gregorian day-labelled dates')
    return out


def reference_training_mask(index, config: MonthlyBiasConfig) -> np.ndarray:
    """Mask OUTER exclusions identically at every site; independent of a local anchor.

    This mask protects regime estimation when window_protocol='refit_excluding'.
    The calibration-period bounds constrain factor fitting separately. Here we
    deliberately do not restrict a climatology to an unrelated factor period.
    """
    t = pd.DatetimeIndex(index)
    mask = np.ones(len(t), dtype=bool)
    gap = pd.Timedelta(days=int(config.gap_days))
    for lo, hi in config.excluded_periods:
        mask &= ~((t >= _day(lo) - gap) & (t <= _day(hi) + gap))
    if config.information_cutoff is not None:
        mask &= t < _day(config.information_cutoff) - gap
    return mask


def _clean_series(series: pd.Series, missing_values) -> pd.Series:
    if not isinstance(series, pd.Series):
        raise TypeError('A pandas.Series with a DatetimeIndex is required')
    t = pd.DatetimeIndex(series.index)
    if (t.hasnans or not t.is_unique or t.tz is not None or
            not t.equals(t.normalize())):
        raise ValueError('Unique, timezone-naive, midnight-labelled daily dates are required')
    v = pd.to_numeric(series, errors='coerce').to_numpy(dtype=float, copy=True)
    v[~np.isfinite(v)] = np.nan
    for code in missing_values:
        v[v == float(code)] = np.nan
    v[v < 0] = np.nan
    return pd.Series(v, index=t, name=series.name).sort_index()


class MonthlyFactorEstimator:
    """Prepared common-day pairs for ONE station; cheap repeated fold fitting.

    Missing dates are not turned into zero. Each accepted calendar month must
    meet coverage against its actual number of Gregorian days, including Feb 29.
    """
    def __init__(self, observed: pd.Series, model: pd.Series, *, config=None,
                 calibration_period=(None, None), station=''):
        self.config = config or MonthlyBiasConfig()
        self.config.validate()
        self.station = str(station)
        o = _clean_series(observed, self.config.missing_values)
        g = _clean_series(model, self.config.missing_values)
        p = pd.concat([o.rename('obs'), g.rename('model')], axis=1).sort_index()
        mask = reference_training_mask(p.index, self.config)
        lo, hi = calibration_period
        if lo is not None:
            mask &= p.index >= _day(lo)
        if hi is not None:
            mask &= p.index <= _day(hi)
        if lo is not None and hi is not None and _day(lo) > _day(hi):
            raise ValueError('Reversed calibration period')
        self.pairs = p.loc[mask & np.isfinite(p.obs) & np.isfinite(p.model)]
        self._by_month = {m: self.pairs[self.pairs.index.month == m] for m in range(1, 13)}
        self.cache: dict[str, dict] = {}

    def fit(self, application_month) -> dict:
        period = pd.Period(application_month, freq='M')
        key = str(period)
        if key in self.cache:
            return dict(self.cache[key])
        cfg = self.config
        start, end = period.start_time, period.end_time.normalize()
        p = self._by_month[period.month]
        gap = pd.Timedelta(days=int(cfg.gap_days))
        if cfg.temporal_mode == 'leave_month_out':
            p = p.loc[~((p.index >= start - gap) & (p.index <= end + gap))]
        elif cfg.temporal_mode == 'leave_year_out':
            lo, hi = pd.Timestamp(period.year, 1, 1), pd.Timestamp(period.year, 12, 31)
            p = p.loc[~((p.index >= lo - gap) & (p.index <= hi + gap))]
        elif cfg.temporal_mode == 'expanding':
            p = p.loc[p.index < start - gap]
        # in_sample and blocked use only the globally eligible pairs prepared
        # above; blocked always has an explicit outer block.
        per_year = p.groupby(p.index.year).agg(
            n_days=('obs', 'size'), obs_sum=('obs', 'sum'), model_sum=('model', 'sum'))
        per_year['expected_days'] = [calendar.monthrange(int(y), period.month)[1] for y in per_year.index]
        per_year['coverage'] = per_year.n_days / per_year.expected_days
        good = per_year[per_year.coverage >= cfg.min_valid_fraction_month]
        p = p.loc[p.index.year.isin(good.index)]
        row = dict(station=self.station, application_month=key, calendar_month=period.month,
                   factor=np.nan, status='INSUFFICIENT_TRAINING_YEARS',
                   n_training_years=len(good), n_pairs=int(good.n_days.sum()),
                   training_years=','.join(map(str, good.index.tolist())),
                   training_start=p.index.min() if len(p) else pd.NaT,
                   training_end=p.index.max() if len(p) else pd.NaT,
                   obs_mean=np.nan, model_mean=np.nan,
                   overlap_days=int(((p.index >= start) & (p.index <= end)).sum()),
                   future_training_used=bool((p.index > end).any()),
                   temporal_mode=cfg.temporal_mode, weighting=cfg.weighting)
        if self.station in set(map(str, cfg.excluded_stations)):
            row.update(status='STATION_EXCLUDED', n_training_years=0, n_pairs=0,
                       training_years='', training_start=pd.NaT, training_end=pd.NaT,
                       overlap_days=0, future_training_used=False)
        elif len(good) >= cfg.min_training_years:
            if cfg.weighting == 'pooled_days':
                om = float(good.obs_sum.sum() / good.n_days.sum())
                gm = float(good.model_sum.sum() / good.n_days.sum())
            else:
                om = float((good.obs_sum / good.n_days).mean())
                gm = float((good.model_sum / good.n_days).mean())
            row.update(obs_mean=om, model_mean=gm)
            if gm < cfg.min_model_mean:
                row['status'] = 'MODEL_MEAN_TOO_SMALL'
            else:
                factor = om / gm
                if not np.isfinite(factor):
                    row['status'] = 'NONFINITE_FACTOR'
                elif cfg.factor_bounds is not None and not cfg.factor_bounds[0] <= factor <= cfg.factor_bounds[1]:
                    row['status'] = 'FACTOR_OUTSIDE_BOUNDS'
                else:
                    row.update(factor=float(factor), status='OK')
        self.cache[key] = row
        return dict(row)


def fit_monthly_factors(observed, model, *, application_months, config=None,
                        calibration_period=(None, None), station='') -> pd.DataFrame:
    """Fit explicit monthly folds at a gauge, independently of the HYBAS workflow."""
    est = MonthlyFactorEstimator(observed, model, config=config,
                                 calibration_period=calibration_period, station=station)
    months = sorted(set(str(pd.Period(m, freq='M')) for m in application_months))
    return pd.DataFrame([est.fit(m) for m in months])


def correct_monthly_outlets(daily: xr.DataArray, observations: Mapping[str, pd.Series],
        simulations: Mapping[str, pd.Series], station_metadata: pd.DataFrame,
        outlets: pd.DataFrame, *, config=None, propagation_config=None,
        calibration_period=(None, None), basins=None, graph=None, skip_outlets=()) -> dict:
    """Correct extracted Q(T,outlet) with fold-specific monthly regional factors.

    Only station vectors and the relatively small month-by-outlet factor table
    are materialized. With a Dask input, multiplying the daily series remains
    lazy; no dense geographic T,Y,X factor cube is constructed.

    Local fitting takes priority only for the SAME HYBAS and the SAME mapped
    river pixel. Ungauged targets receive positive donor factors averaged in log
    space. A legitimate zero factor is usable locally but not log-interpolated.
    Network/hybrid/IDW/top-kriging reuse the v5.1.0 candidate restrictions.
    """
    if __package__:
        from .highflow_predictand import PredictandConfig, _candidates, _weights
    else:
        from highflow_predictand import PredictandConfig, _candidates, _weights
    cfg = config or MonthlyBiasConfig(); cfg.validate()
    pcfg = propagation_config or PredictandConfig()
    pcfg.validate()
    if pcfg.propagation_method in {'network', 'hybrid'} and graph is None:
        if basins is None:
            raise ValueError('Network/hybrid transfer requires graph or basin topology; choose idw_log explicitly otherwise')
        if __package__:
            from .glofas_highflow_bias_correction import build_hybas_digraph
        else:
            from glofas_highflow_bias_correction import build_hybas_digraph
        graph = build_hybas_digraph(basins)
    if set(daily.dims) != {'T', 'outlet'}:
        raise ValueError('Monthly discharge correction expects extracted Q(T,outlet), not a geographic cube or members')
    t = pd.DatetimeIndex(daily['T'].values)
    if (not len(t) or t.hasnans or not t.is_unique or t.tz is not None or
            not t.equals(t.normalize()) or not t.is_monotonic_increasing):
        raise ValueError('Sorted unique day-labelled T is required')
    tab = outlets.copy().reset_index(drop=True); tab['outlet_id'] = tab.outlet_id.astype(str)
    if tab.outlet_id.duplicated().any():
        raise ValueError('Duplicate outlet_id')
    if list(map(str, daily.outlet.values)) != list(tab.outlet_id):
        raise ValueError('Daily outlet coordinates must match metadata in the same order')
    meta = station_metadata.copy()
    if 'station' not in meta:
        raise ValueError('station_metadata requires station')
    meta['station'] = meta.station.astype(str)
    if meta.station.duplicated().any():
        raise ValueError('Duplicate station metadata')
    excluded = set(map(str, cfg.excluded_stations))
    meta = meta[~meta.station.isin(excluded)]
    estimators = {name: MonthlyFactorEstimator(observations[name], simulations[name],
                  config=cfg, calibration_period=calibration_period, station=name)
                  for name in meta.station if name in observations and name in simulations}
    meta = meta[meta.station.isin(estimators)]
    periods = t.to_period('M'); unique = periods.unique()
    factors = np.full((len(unique), len(tab)), np.nan, dtype=float)
    rows, donors, fit_rows = [], [], []
    relation_cache = {}
    candidate_map = {str(target.outlet_id): _candidates(target, meta, pcfg, graph, relation_cache)
                     for _, target in tab.iterrows()}
    skip = set(map(str, skip_outlets))
    metadata = meta.set_index('station')
    for mi, period in enumerate(unique):
        fitted = {name: estimator.fit(str(period)) for name, estimator in estimators.items()}
        fit_rows.extend(fitted.values())
        for oi, target in tab.iterrows():
            # Index can be non-positional in user-provided metadata.
            oi = int(tab.index.get_loc(oi))
            oid = str(target.outlet_id)
            audit = dict(outlet_id=oid, HYBAS_ID=int(target.HYBAS_ID),
                         application_month=str(period), calendar_month=period.month,
                         factor=np.nan, status='NO_CALIBRATED_DONOR',
                         n_donors=0, donor_stations='', temporal_mode=cfg.temporal_mode,
                         training_overlap_days=0, future_training_used=False)
            if oid in skip:
                factors[mi, oi] = 1.
                audit.update(factor=1., status='NOT_APPLIED_GLOFAS_REFERENCE')
                rows.append(audit); continue
            choices = [c for c in candidate_map[oid] if fitted[c['station']]['status'] == 'OK']
            local = []
            for c in choices:
                d = metadata.loc[c['station']]
                if (int(d.HYBAS_ID) == int(target.HYBAS_ID) and
                    np.isfinite(target.get('matched_source_x', np.nan)) and
                    target.get('matched_source_x') == d.get('matched_source_x') and
                    target.get('matched_source_y') == d.get('matched_source_y')):
                    local.append(c)
            if local:
                use = local[:1]; weights = np.ones(1)
                factor = fitted[use[0]['station']]['factor']
                audit['status'] = 'LOCAL_MONTHLY_FACTOR'
            else:
                use = [c for c in choices if fitted[c['station']]['factor'] > 0][:pcfg.k_donors]
                weights = _weights(use, pcfg)
                factor = np.nan
                if use:
                    z = np.log([fitted[c['station']]['factor'] for c in use])
                    if pcfg.propagation_method == 'topkriging' and len(use) > 1:
                        if basins is None:
                            raise ValueError('Top-kriging requires real basin supports')
                        if __package__:
                            from .highflow_topkriging import (TopKrigingConfig, build_supports_from_hybas,
                                fit_point_variogram, topkriging_predict)
                        else:
                            from highflow_topkriging import (TopKrigingConfig, build_supports_from_hybas,
                                fit_point_variogram, topkriging_predict)
                        tk = pcfg.topkriging_config or TopKrigingConfig()
                        names = ['donor:' + c['station'] for c in use] + ['target:' + oid]
                        ids = [c['HYBAS_ID'] for c in use] + [int(target.HYBAS_ID)]
                        try:
                            support = build_supports_from_hybas(ids, basins, names=names, config=tk)
                            ss = [support[n] for n in names[:-1]]
                            vg = fit_point_variogram(ss, z, tk)
                            prediction = topkriging_predict(support[names[-1]], ss, z, vg, tk)
                            factor = float(np.exp(prediction['value']))
                            audit['log_factor_variance'] = float(prediction['variance'])
                            weights = np.asarray(prediction.get('weights', np.full(len(use), np.nan)))
                        except (ValueError, np.linalg.LinAlgError) as exc:
                            audit.update(status='TOPKRIGING_FAILED', error=str(exc))
                    else:
                        factor = float(np.exp(np.dot(weights, z)))
                    if np.isfinite(factor):
                        audit['status'] = 'REGIONAL_MONTHLY_FACTOR'
                elif choices:
                    audit['status'] = 'ZERO_FACTORS_NOT_LOG_TRANSFERABLE'
            if np.isfinite(factor) and cfg.factor_bounds is not None:
                if not cfg.factor_bounds[0] <= factor <= cfg.factor_bounds[1]:
                    factor = np.nan; audit['status'] = 'REGIONAL_FACTOR_OUTSIDE_BOUNDS'
            if not np.isfinite(factor):
                if cfg.missing_factor_policy == 'error':
                    raise ValueError(f'{oid}/{period}: {audit["status"]}')
                if cfg.missing_factor_policy == 'raw':
                    factor = 1.; audit['status'] += '__RAW_RETAINED'
            factors[mi, oi] = factor
            audit.update(factor=factor, n_donors=len(use),
                         donor_stations=';'.join(c['station'] for c in use),
                         training_overlap_days=max((fitted[c['station']]['overlap_days'] for c in use), default=0),
                         future_training_used=any(fitted[c['station']]['future_training_used'] for c in use))
            for c, weight in zip(use, weights):
                f = fitted[c['station']]
                donors.append(dict(c, outlet_id=oid, application_month=str(period),
                    donor_factor=f['factor'], weight=float(weight),
                    training_years=f['training_years'], n_pairs=f['n_pairs'],
                    overlap_days=f['overlap_days'], future_training_used=f['future_training_used']))
            rows.append(audit)
    month_field = xr.DataArray(factors, dims=('month_index', 'outlet'),
        coords={'month_index': np.arange(len(unique)), 'outlet': daily.outlet}, name='monthly_factor')
    indexer = xr.DataArray(unique.get_indexer(periods), dims='T', coords={'T': daily['T']})
    factor_daily = month_field.isel(month_index=indexer).drop_vars('month_index')
    # Match the input chunking when available. The small factor table has already
    # been computed; only the T,outlet broadcast/multiplication can remain lazy.
    if daily.chunks is not None:
        factor_daily = factor_daily.chunk({dim: daily.chunksizes[dim] for dim in ('T', 'outlet')})
    clean = daily.where(np.isfinite(daily) & (daily >= 0))
    for code in cfg.missing_values:
        clean = clean.where(clean != code)
    corrected = (clean * factor_daily).rename(daily.name or 'Q')
    corrected.attrs = dict(daily.attrs, correction='monthly_ratio', correction_scale='daily_by_calendar_month',
        temporal_mode=cfg.temporal_mode, window_protocol=cfg.window_protocol,
        missing_factor_policy=cfg.missing_factor_policy,
        validation_scope='factor_exclusion_only_unless_outer_calendar_protocol_is_protected')
    result = dict(corrected=corrected, factors=pd.DataFrame(rows), donors=pd.DataFrame(donors),
                  station_factors=pd.DataFrame(fit_rows), config=cfg,
                  application_months=tuple(map(str, unique)),
                  scope='daily discharge at extracted outlets; no routing or flow-conservation constraint')
    if len(result['factors']) and (result['factors'].training_overlap_days > 0).any():
        warnings.warn('Monthly calibration overlaps application months: in-sample fit, not independent validation.', RuntimeWarning)
    return result


def aggregate_corrected_like(daily, raw, *, kind, selector, raw_daily=None):
    """Apply the already selected HW/MODE windows after DAILY correction.

    Reuse precisely the raw aggregation's years, targets and physical outlets.
    A RAW comparator on common valid days is included in the diagnostics when
    raw_daily is supplied. The full-support raw product is not overwritten.
    """
    if __package__:
        from .highflow_outlet_aggregation import select_catalog, aggregate_series
    else:
        from highflow_outlet_aggregation import select_catalog, aggregate_series
    final = copy.copy(raw)
    final.values = xr.full_like(raw.values, np.nan)
    final.coverage = xr.full_like(raw.coverage, np.nan)
    final.expected_days = raw.expected_days.copy(deep=True)
    final.diagnostics = raw.diagnostics.copy()
    selected, _ = select_catalog(raw.windows, kind=kind, selector=selector)
    years = pd.DatetimeIndex(raw.values['T'].values).year.tolist()
    times = pd.DatetimeIndex(daily['T'].values)
    records = []
    for oi, oid in enumerate(raw.values.outlet.values):
        local = pd.Series(daily.sel(outlet=oid).values, index=times)
        baseline = (pd.Series(raw_daily.sel(outlet=oid).values, index=times)
                    if raw_daily is not None else None)
        for ki, target_id in enumerate(raw.values.target.values):
            cc = selected[(selected.outlet_id.astype(str) == str(oid)) & (selected.target_id == target_id)]
            if cc.empty:
                continue
            st = aggregate_series(local, cc.to_dict('records'), int(cc.anchor_doy.iloc[0]), years, raw.config)
            final.values.values[:, oi, ki] = st.value.values
            final.coverage.values[:, oi, ki] = st.coverage.values
            if baseline is not None:
                common = aggregate_series(baseline.where(local.notna()), cc.to_dict('records'),
                    int(cc.anchor_doy.iloc[0]), years, raw.config)
                st['raw_value_common_days'] = common.value.to_numpy()
            for r in st.to_dict('records'):
                r.update(outlet_id=str(oid), target_id=str(target_id))
                records.append(r)
    final.diagnostics = pd.DataFrame(records)
    return final


__all__ = ['MonthlyBiasConfig', 'MonthlyFactorEstimator', 'fit_monthly_factors',
           'correct_monthly_outlets', 'reference_training_mask']
