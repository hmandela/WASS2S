"""Probability and forecast contracts shared by all seasonal analogue engines.

Analogue members are observations from the allowed training library. Their
weighted empirical CDF is evaluated at independently declared climate events.
Residual dressing is a separate, explicit comparison method.
"""
import numpy as np
import pandas as pd
import xarray as xr
from .was_probability_reference import climatological_terciles
from .was_probability_calibration import _field, normalize_probability_calibration

NATIVE = ('analog', 'analog_ensemble', 'direct')


def field(data):
    result = _field(data, 'analogue predictand')
    years = np.asarray(result["T"].dt.year.values)
    if len(np.unique(years)) != len(years):
        raise ValueError('Analogues require one target season per year')
    return result


def event_reference(model, observations, start=None, end=None):
    if getattr(model, 'tercile_reference', 'climatology') == 'training':
        return None, None
    sample = getattr(model, '_analog_event_reference_sample', None)
    thresholds = getattr(model, '_analog_event_thresholds', None)
    if sample is not None and thresholds is not None:
        xr.align(thresholds, observations.isel(T=0, drop=True), join='exact')
        return sample, thresholds
    sample = observations if sample is None else field(sample)
    start = getattr(model, 'clim_year_start', None) if start is None else start
    end = getattr(model, 'clim_year_end', None) if end is None else end
    q, _ = climatological_terciles(sample, start, end)
    if start is not None:
        sample = sample.where((sample["T"].dt.year >= int(start)) & (sample["T"].dt.year <= int(end)), drop=True)
    return sample, q


def empirical_reference(sample, thresholds):
    """Observed event frequencies, including ties and dry point masses."""
    low, high = [thresholds.isel(quantile=i, drop=True) for i in range(2)]
    sample, low, high = xr.align(sample, low, high, join='exact')
    valid = np.isfinite(sample)
    count = valid.sum('T')
    mask = (count >= 3) & np.isfinite(low) & np.isfinite(high) & (low <= high)
    return xr.concat([((sample <= low) & valid).sum('T') / count,
                      ((sample > low) & (sample <= high) & valid).sum('T') / count,
                      ((sample > high) & valid).sum('T') / count],
                     dim=xr.IndexVariable('probability', ['PB', 'PN', 'PA'])).where(mask)


def probability_maps(model_class, observations, years, weights, reference,
                     shrinkage=0., thresholds=None):
    lam = float(shrinkage)
    if not np.isfinite(lam) or not 0 <= lam <= 1:
        raise ValueError('probability shrinkage must lie in [0,1]')
    years, weights = np.asarray(years, int), np.asarray(weights, float)
    if years.ndim != 1 or weights.shape != years.shape or len(np.unique(years)) != len(years):
        raise ValueError('Analogue years must be unique and match one weight per year')
    if np.any(~np.isfinite(weights)) or np.any(weights < 0) or (len(years) and weights.sum() <= 0):
        raise ValueError('Analogue weights must be finite, nonnegative and have positive sum')
    q = climatological_terciles(reference)[0] if thresholds is None else thresholds
    prior = empirical_reference(reference, q)
    if not len(years):
        return prior
    low, high = [q.isel(quantile=i, drop=True) for i in range(2)]
    arr = xr.concat([model_class._map_for_year(observations, int(y)) for y in years], dim='analog')
    arr, low, high, prior = xr.align(arr, low, high, prior, join='exact')
    w = xr.DataArray(weights / weights.sum(), dims='analog')
    valid = np.isfinite(arr)
    den = xr.where(valid, w, 0.).sum('analog')
    p = xr.concat([xr.where(valid & (arr <= low), w, 0.).sum('analog'),
                   xr.where(valid & (arr > low) & (arr <= high), w, 0.).sum('analog'),
                   xr.where(valid & (arr > high), w, 0.).sum('analog')],
                  dim=xr.IndexVariable('probability', ['PB', 'PN', 'PA'])) / den
    # Missing observations among selected analogues do not manufacture evidence.
    p = ((1-lam)*p + lam*prior).where((den > 0) & prior.notnull().all('probability'))
    return p / p.sum('probability', skipna=False)


def metadata(model, probability, mode='outer'):
    mode = normalize_probability_calibration(mode)
    probability.attrs.update(probability_calibration_requested=mode,
        probability_calibration=mode, probability_calibration_effective='native_analog_ensemble',
        residual_calibration_applicable=0,
        calibration='weighted training analogue observations; declared empirical climate events',
        calibration_strictly_nested=1,
        calibration_strictness_scope='native probabilities fitted without the outer verification targets; not a claim of independent inner tuning scores',
        tercile_reference=getattr(model, 'tercile_reference', 'climatology'),
        probability_prior='empirical event frequencies of the declared reference',
        tuning_inner_preprocessing=getattr(model, 'inner_preprocessing', 'strict'))
    return probability


def selection_reference(model, observations, selection, fixed_sample, fixed_q, start, end):
    train = np.asarray(selection['train_years'], int)
    years = np.asarray(selection['analog_years'], int)
    if not np.isin(years, train).all():
        raise ValueError('Cached analogues must belong to their training library')
    if fixed_sample is not None:
        return fixed_sample, fixed_q
    sample = observations.where(observations["T"].dt.year.isin(train), drop=True)
    q, _ = climatological_terciles(sample, start, end)
    if start is not None:
        sample = sample.where((sample["T"].dt.year >= int(start)) & (sample["T"].dt.year <= int(end)), drop=True)
    return sample, q


def native_compute_prob(model, observations, start, end, hindcast, **kwargs):
    p, h = field(observations), _field(hindcast, 'hindcast')
    xr.align(p.isel(T=0, drop=True), h.isel(T=0, drop=True), join='exact')
    external = kwargs.get('obs_for_terciles')
    if external is not None:
        q, _ = climatological_terciles(field(external), start, end)
        sample = field(external)
        if start is not None:
            sample = sample.where((sample["T"].dt.year >= int(start)) & (sample["T"].dt.year <= int(end)), drop=True)
        fixed, qfixed = (sample, q) if getattr(model, 'tercile_reference', 'climatology') != 'training' else (None, None)
    else:
        fixed, qfixed = model._fixed_tercile_reference(p, start, end)
    parts = []
    for t in h["T"].values:
        year = int(pd.Timestamp(t).year)
        sels = model._analog_member_cv_cache.get(year)
        if not sels:
            one = model._analog_cv_cache.get(year)
            sels = [] if one is None else [one]
        if not sels:
            if not bool(np.isfinite(h.sel(T=t)).any()):
                parts.append(xr.full_like(h.sel(T=t, drop=True), np.nan).expand_dims(
                    probability=['PB','PN','PA'], T=[t]))
                continue
            raise ValueError(f'No training analogue selections for {year}; run cross_validate/compute_model first')
        probs = []
        for sel in sels:
            if year in np.asarray(sel['train_years'], int):
                raise ValueError('Verification year appears in the cached analogue training library')
            sample, q = selection_reference(model, p, sel, fixed, qfixed, start, end)
            probs.append(model._analog_probability_maps(p, sel['analog_years'], sel['weights'],
                sample, sel['probability_shrinkage'], thresholds=q))
        prob = model._mean_probabilities(probs).where(np.isfinite(h.sel(T=t, drop=True)))
        parts.append(prob.expand_dims(T=[t]))
    out = xr.concat(parts, dim='T').transpose('probability','T','Y','X')
    return metadata(model, out, kwargs.get('probability_calibration', 'outer'))


def select_members(model, observations, train_positions, test_positions=None, *, source='hindcast', tune=None):
    p = field(observations)
    year = int(model.year_forecast) if source == 'forecast' else int(p["T"].dt.year.isel(T=int(test_positions[0])))
    selections, dets = [], []
    reuse = getattr(model, '_analog_reuse_selections', {}) or {}
    member_fixed = reuse.get('members') or ([reuse['single']] if reuse.get('single') else [])
    old_fixed = getattr(model, '_analog_fixed_selection', None)
    try:
        for member in model._available_target_members(source, year):
            model._analog_fixed_selection = next((sel for sel in member_fixed if sel.get('member') == member), None)
            sel = model.select_analogs(p, train_positions, test_positions,
                target_source=source, tune=tune, member=member)
            years = np.asarray(sel['analog_years'], int)
            if not np.isin(years, np.asarray(sel['train_years'], int)).all():
                raise ValueError('Selected analogue year is outside the permitted training library')
            if len(years) < model.min_analogs:
                if not model.fallback_to_climatology:
                    raise ValueError('No defensible analogue set found')
                # Make deterministic/probabilistic fallback decisions identical.
                sel = dict(sel, analog_years=np.array([], int), weights=np.array([], float),
                           distances=np.array([], float), fallback='climatology')
                sample = p.where(p["T"].dt.year.isin(sel['train_years']), drop=True)
                det = sample.mean('T', skipna=True)
            else:
                det = model._weighted_map([model._map_for_year(p, int(y)) for y in years], sel['weights'])
            selections.append(dict(sel)); dets.append(det)
    finally:
        model._analog_fixed_selection = old_fixed
    if not dets:
        raise ValueError('No model target members are available')
    stack = xr.concat(dets, dim='member_analog')
    aggregation = model.target_member_aggregation
    if aggregation not in ('mean','median'):
        raise ValueError('target_member_aggregation must be mean or median')
    det = getattr(stack, aggregation)('member_analog', skipna=True)
    model._analog_member_cv_cache[year] = selections
    model._analog_cv_cache[year] = selections[0]
    return det, selections


def compute_model(model, predictant, itrain, itest, **kwargs):
    p = field(predictant)
    positions = np.asarray(itest, int).ravel()
    if len(positions) != 1:
        raise ValueError('Analogue compute_model requires one held-out year')
    det, _ = select_members(model, p, itrain, positions, tune=kwargs.get('nested_tuning'))
    return det.expand_dims(T=p["T"].isel(T=positions)).transpose('T','Y','X')


def residual_forecast(model, p, h, det, start, end, **kwargs):
    from .was_crossfit import CrossFitConfig
    from .was_cross_validate import _training_probabilities
    h = _field(h, 'hindcast_det')
    xr.align(p.isel(T=0, drop=True), h.isel(T=0, drop=True), join='exact')
    obs, hind = xr.align(p, h, join='inner')
    if obs.sizes['T'] < 3:
        raise ValueError('Residual forecast needs at least three matching OOF hindcast dates')
    errors = obs - hind
    ref, q = model._fixed_tercile_reference(p, start, end)
    options = {k:kwargs[k] for k in ('probability_fit','probability_distribution_map','probability_fit_mode') if k in kwargs}
    options['tercile_reference'] = getattr(model,'tercile_reference','climatology')
    if q is not None:
        options.update(_probability_thresholds=q, _probability_reference=empirical_reference(ref,q))
    cfg = CrossFitConfig(climatology_start=start, climatology_end=end)
    prob, _, _, _ = _training_probabilities(model, det, errors, p, cfg, options)
    prob.attrs.update(probability_calibration='operational_oof',
        probability_calibration_requested=normalize_probability_calibration(kwargs.get('probability_calibration')),
        residual_source='supplied out-of-fold hindcast', residual_variance_ddof=1,
        probability_calibration_effective='operational_oof', calibration_strictly_nested=0,
        tercile_reference=options['tercile_reference'])
    return prob


def forecast(model, predictant, start, end, hindcast, **kwargs):
    p = field(predictant)
    if bool((p["T"].dt.year >= int(model.year_forecast)).any()):
        raise ValueError('Operational analogue training observations must precede year_forecast')
    if kwargs.get('obs_for_terciles') is not None:
        sample = field(kwargs['obs_for_terciles'])
        q, _ = climatological_terciles(sample, start, end)
        if start is not None:
            sample = sample.where((sample["T"].dt.year >= int(start)) & (sample["T"].dt.year <= int(end)), drop=True)
        model._analog_event_reference_sample, model._analog_event_thresholds = sample, q
    fixed, qfixed = model._fixed_tercile_reference(p, start, end)
    det, sels = select_members(model,p,np.arange(p.sizes['T']),source='forecast',
        tune=kwargs.get('nested_tuning',model.nested_tuning))
    date = kwargs.get('forecast_time', getattr(model, 'forecast_time', None))
    if date is None:
        dates = p.get_index('T')
        if len(set(zip(dates.month, dates.day))) != 1:
            raise ValueError('Specify forecast_time when observation season dates differ')
        date = pd.Timestamp(dates[0]).replace(year=int(model.year_forecast))
    date = pd.Timestamp(date)
    if date.year != int(model.year_forecast):
        raise ValueError('forecast_time year must equal year_forecast')
    det = det.expand_dims(T=[date]).transpose('T','Y','X')
    if model.probability_method in NATIVE:
        probs=[]
        for sel in sels:
            sample,q=selection_reference(model,p,sel,fixed,qfixed,start,end)
            probs.append(model._analog_probability_maps(p,sel['analog_years'],sel['weights'],sample,
                sel['probability_shrinkage'],thresholds=q))
        prob=model._mean_probabilities(probs).expand_dims(T=[date]).transpose('probability','T','Y','X')
        prob=metadata(model,prob,kwargs.get('probability_calibration','outer'))
    else:
        prob=residual_forecast(model,p,hindcast,det,start,end,**kwargs)
    prob=prob.where(np.isfinite(det))
    union=sorted({int(y) for sel in sels for y in sel['analog_years']})
    return np.asarray(union,int),det,prob


def region_masks(model, observations):
    """Align categorical masks without interpolation of their numeric labels."""
    rm=model.regional_masks
    template=observations.isel(T=0,drop=True)
    if isinstance(rm,dict):
        items=rm.items()
    elif isinstance(rm,xr.DataArray):
        labels=np.unique(np.asarray(rm.values)[np.isfinite(rm.values)])
        excluded=[rm.attrs.get(k) for k in ('background_label','nodata','_FillValue') if k in rm.attrs]
        items=[(str(v),rm==v) for v in labels if v not in excluded]
    else:
        raise TypeError('regional_masks must be a dict or a labelled DataArray')
    out={}; covered=xr.zeros_like(template,dtype=bool)
    for name,mask in items:
        if not isinstance(mask,xr.DataArray):
            mask=xr.DataArray(np.asarray(mask),dims=('Y','X'),coords={'Y':template.Y,'X':template.X})
        if set(mask.dims)!={'Y','X'}:
            raise ValueError('Each regional mask must have only Y,X dimensions')
        for dim in ('Y','X'):
            if mask.get_index(dim).has_duplicates:
                raise ValueError('Regional mask coordinates must be unique')
            if not mask.get_index(dim).equals(template.get_index(dim)):
                lower,upper=float(mask[dim].min()),float(mask[dim].max())
                mask=mask.sortby(dim).reindex({dim:template[dim]},method='nearest')
                mask=mask.where((template[dim]>=lower)&(template[dim]<=upper))
        mask=(np.isfinite(mask)&(mask!=0)).transpose('Y','X')
        if bool((covered&mask).any()):
            raise ValueError(f'Regional masks overlap in region {name!r}; provide disjoint regions')
        if bool(mask.any()):
            out[str(name)]=mask; covered=covered|mask
    if not out:
        raise ValueError('No non-empty regional mask was provided')
    return out


def _regional_child(model,name):
    from copy import copy
    child=copy(model)
    child._analog_cv_cache={}; child._analog_member_cv_cache={}
    reuse=(getattr(model,'_analog_reuse_selections',{}) or {}).get('regions') or {}
    record=reuse.get(name)
    child._analog_reuse_selections={} if record is None else {
        'members':record.get('member_selections') or [record], 'single':record}
    return child


def regional_compute_model(model,predictant,itrain,itest,**kwargs):
    p=field(predictant); masks=region_masks(model,p)
    pos=np.asarray(itest,int).ravel()
    if len(pos)!=1:
        raise ValueError('Regional analogue compute_model requires one held-out year')
    year=int(p['T'].dt.year.isel(T=int(pos[0])))
    # Baseline for uncovered cells is the available training climatology.
    result=p.isel(T=np.asarray(itrain,int)).mean('T',skipna=True).expand_dims(T=p['T'].isel(T=pos))
    records={}
    for name,mask in masks.items():
        child=_regional_child(model,name)
        pred=compute_model(child,p.where(mask),itrain,pos,**kwargs)
        sels=child._analog_member_cv_cache[year]
        records[name]=dict(sels[0],member_selections=sels)
        result=xr.where(mask,pred,result)
    model._regional_cv_cache[year]=records; model._regional_last=records
    return result.transpose('T','Y','X')


def regional_compute_prob(model,observations,start,end,hindcast,**kwargs):
    if model.probability_method not in NATIVE:
        from .was_probability_calibration import direct_hindcast_probabilities
        return direct_hindcast_probabilities(model,observations,start,end,hindcast,**kwargs)
    p=field(observations); h=_field(hindcast,'hindcast_det'); masks=region_masks(model,p)
    if kwargs.get('obs_for_terciles') is not None:
        ref=field(kwargs['obs_for_terciles']); q,_=climatological_terciles(ref,start,end)
        if start is not None:ref=ref.where((ref['T'].dt.year>=int(start))&(ref['T'].dt.year<=int(end)),drop=True)
        fixed,qfixed=(ref,q) if getattr(model,'tercile_reference','climatology')!='training' else (None,None)
    else:
        fixed,qfixed=model._fixed_tercile_reference(p,start,end)
    parts=[]
    for t in h['T'].values:
        year=int(pd.Timestamp(t).year); records=model._regional_cv_cache.get(year)
        if not records:
            if bool(np.isfinite(h.sel(T=t)).any()):
                raise ValueError(f'No regional analogue selections for {year}')
            parts.append(xr.full_like(h.sel(T=t,drop=True),np.nan).expand_dims(probability=['PB','PN','PA'],T=[t]));continue
        first=next(iter(records.values()))
        sample,q=selection_reference(model,p,first,fixed,qfixed,start,end)
        prob=empirical_reference(sample,q)
        for name,mask in masks.items():
            if name not in records:
                raise ValueError(f'Missing analogue selections for region {name!r}, year {year}')
            record=records[name]; child=_regional_child(model,name)
            child._analog_member_cv_cache[year]=record.get('member_selections') or [record]
            local=native_compute_prob(child,p,start,end,h.sel(T=[t]),**kwargs).isel(T=0,drop=True)
            prob=xr.where(mask,local,prob)
        parts.append(prob.where(np.isfinite(h.sel(T=t,drop=True))).expand_dims(T=[t]))
    out=xr.concat(parts,dim='T').transpose('probability','T','Y','X')
    return metadata(model,out,kwargs.get('probability_calibration','outer'))


def regional_forecast(model,predictant,start,end,hindcast,**kwargs):
    p=field(predictant); masks=region_masks(model,p)
    if bool((p['T'].dt.year>=int(model.year_forecast)).any()):
        raise ValueError('Operational analogue training observations must precede year_forecast')
    reference=kwargs.get('obs_for_terciles',p)
    q,_=climatological_terciles(reference,start,end)
    if start is not None:reference=reference.where((reference['T'].dt.year>=int(start))&(reference['T'].dt.year<=int(end)),drop=True)
    base_probability=empirical_reference(reference,q)
    result=None; probs=None; records={}; union=[]
    for name,mask in masks.items():
        child=_regional_child(model,name)
        child._analog_event_reference_sample=reference
        child._analog_event_thresholds=q
        region_kwargs=dict(kwargs,obs_for_terciles=reference)
        years,det,prob=forecast(child,p.where(mask),start,end,
            None if hindcast is None else hindcast.where(mask),**region_kwargs)
        if result is None:
            result=p.mean('T',skipna=True).expand_dims(T=det['T'])
            probs=base_probability.expand_dims(T=det['T'])
        result=xr.where(mask,det,result); probs=xr.where(mask,prob,probs)
        sels=child._analog_member_cv_cache[int(model.year_forecast)]
        records[name]=dict(sels[0],member_selections=sels); union.extend(years.tolist())
    model._regional_last=records
    model._regional_cv_cache[int(model.year_forecast)]=records
    probs=probs.transpose('probability','T','Y','X').where(np.isfinite(result))
    if model.probability_method in NATIVE:probs=metadata(model,probs,kwargs.get('probability_calibration','outer'))
    else:probs.attrs.update(probability_calibration_effective='operational_oof',residual_source='supplied out-of-fold hindcast')
    return np.asarray(sorted(set(union)),int),result.transpose('T','Y','X'),probs
