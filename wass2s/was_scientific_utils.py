"""Shared numerical contracts. No plotting, downloads or model imports."""
import numpy as np
import pandas as pd
import xarray as xr


def align_predictor_target(predictor, target, *, mode='exact'):
    """Validate temporal identity; never guess from equally sized arrays.

    ``season_year`` is an explicit opt-in for annual seasonal fields labelled
    at different months. The ordered, unique verification years must agree;
    cross-year seasons should carry their explicit verification-year labels.
    No missing-year intersection or positional shift is performed.
    """
    if mode not in {'exact', 'season_year'}:
        raise ValueError("time alignment must be 'exact' or 'season_year'")
    if isinstance(predictor, (list, tuple)):
        return type(predictor)(align_predictor_target(p, target, mode=mode) for p in predictor)
    if isinstance(predictor, dict):
        return {k: align_predictor_target(v, target, mode=mode) for k, v in predictor.items()}
    if predictor is None:
        return None
    for obj in (predictor, target):
        if 'T' not in obj.dims:
            raise ValueError('Predictor and target need a T dimension')
        index = obj.get_index('T')
        if index.has_duplicates or not index.is_monotonic_increasing:
            raise ValueError('T coordinates must be unique and sorted')
    if predictor.get_index('T').equals(target.get_index('T')):
        return predictor
    if mode == 'season_year':
        a, b = predictor['T'].dt.year.values, target['T'].dt.year.values
        if len(np.unique(a)) == len(a) and len(np.unique(b)) == len(b) and np.array_equal(a, b):
            out = predictor.assign_coords(T=target['T'])
            out.attrs = dict(predictor.attrs, time_alignment='explicit matching verification year')
            return out
    raise ValueError('Predictor/target T coordinates differ; align explicitly before fitting or choose time_alignment="season_year" for matching annual verification years')


def apply_physical_support(prediction, reference):
    """Apply explicit variable support in PHYSICAL units only.

    Never infer support from observed extrema (which can contain held-out y).
    Set reference.attrs['support_policy'] to 'nonnegative' or 'identity'.
    Recognised rainfall names default to nonnegative; unknown variables do not.
    """
    policy = reference.attrs.get("support_policy")
    if policy is None:
        name = str(getattr(reference, "name", "") or "").upper()
        standard = str(reference.attrs.get("standard_name", "")).lower()
        policy = "nonnegative" if name in {"PRCP", "PRECIP", "PRECIPITATION", "RAINFALL"} or standard.startswith("precipitation_") or standard == "lwe_thickness_of_precipitation_amount" else "identity"
    if policy == "nonnegative":
        return prediction.clip(min=0)
    if policy == "identity":
        return prediction
    raise ValueError(f"Unknown support_policy: {policy!r}")


def climate_cluster_map(data, n_clusters=10, random_state=42, min_valid=3):
    """One deterministic cluster per CELL using training climatology only."""
    from sklearn.cluster import KMeans
    data = data.transpose("T", "Y", "X")
    q = data.quantile([1/3, 2/3], "T", skipna=True)
    features = xr.concat([
        data.mean("T"), data.std("T"),
        q.isel(quantile=0, drop=True), q.isel(quantile=1, drop=True),
    ], dim="feature").transpose("Y", "X", "feature")
    values = np.asarray(features.values, dtype=float).reshape(-1, 4)
    valid = np.isfinite(values).all(axis=1) & (np.isfinite(data).sum("T").values.ravel() >= min_valid)
    labels = np.full(len(values), np.nan)
    effective = 0
    if valid.any():
        v = values[valid]
        scale = v.std(axis=0)
        v = (v-v.mean(axis=0))/np.where(scale > 0, scale, 1)
        effective = min(max(1, int(n_clusters)), len(v), len(np.unique(v, axis=0)))
        labels[valid] = 0 if effective == 1 else KMeans(n_clusters=effective, random_state=random_state, n_init=10).fit_predict(v)
    result = xr.DataArray(labels.reshape(data.sizes['Y'], data.sizes['X']), dims=('Y','X'), coords={'Y':data.Y, 'X':data.X}, name='cluster')
    result.attrs.update(features='mean,std,q33,q67', effective_clusters=effective,
                        training_times=','.join(str(t) for t in data['T'].values))
    return result


def noleap_dayofyear(time):
    """365-day calendar index; Feb 29 uses Feb 28's threshold.

    Works for pandas and xarray Gregorian/noleap coordinates. A 360-day
    calendar is intentionally rejected instead of silently reinterpreted.
    """
    if isinstance(time, xr.DataArray):
        calendar = getattr(time.dt, 'calendar', 'proleptic_gregorian')
        if calendar not in {'standard','gregorian','proleptic_gregorian','noleap','365_day'}:
            raise ValueError(f'Unsupported percentile calendar: {calendar}')
        return (time.dt.dayofyear - (time.dt.is_leap_year & (time.dt.month > 2)).astype(int)
                - ((time.dt.month == 2) & (time.dt.day == 29)).astype(int))
    index = pd.DatetimeIndex(time)
    return np.asarray(index.dayofyear - (index.is_leap_year & (index.month > 2)).astype(int)
                      - ((index.month == 2) & (index.day == 29)).astype(int))


def calendar_day_percentiles(data, quantile, time_dim='time'):
    """Five-day circular calendar windows, finite observations, leap-safe.

    Feb 29 is excluded from the reference sample. Missing days stay missing;
    they cannot join non-adjacent calendar days. This is ETCCDI-like and does
    not implement the in-base-period ETCCDI bootstrap adjustment.
    """
    data = data.sortby(time_dim)
    if data.get_index(time_dim).has_duplicates:
        raise ValueError('Duplicate daily dates in percentile reference')
    t = data[time_dim]
    data = data.where(~((t.dt.month == 2) & (t.dt.day == 29)), drop=True)
    if data.sizes[time_dim] == 0:
        raise ValueError('Empty percentile reference period')
    t = data[time_dim]
    data = data.assign_coords(_year=(time_dim,t.dt.year.values),
                              dayofyear=(time_dim,noleap_dayofyear(t).values))
    annual = data.set_index({time_dim:['_year','dayofyear']}).unstack(time_dim).reindex(dayofyear=np.arange(1,366))
    # Coordinate replacement keeps the circular padding unique and ordered.
    padded = xr.concat([annual.isel(dayofyear=slice(-2,None)), annual,
                        annual.isel(dayofyear=slice(0,2))], dim='dayofyear')
    padded = padded.assign_coords(dayofyear=np.arange(-1,368))
    windows = padded.rolling(dayofyear=5,center=True,min_periods=1).construct('_window').isel(dayofyear=slice(2,-2))
    windows = windows.assign_coords(dayofyear=np.arange(1,366))
    if hasattr(windows.data,'chunks'):
        windows = windows.chunk({'_year':-1,'_window':-1})
    return windows.quantile(quantile,dim=['_year','_window'],skipna=True,method='linear')


def prepend_available_observations(model, observations, init_month, initialization_year=None):
    """Complete yearly forcing only with observations BEFORE initialization.

    A whole calendar year is retained so missing future forcing is visible.
    With initialization_year=None, each calendar year is treated as an
    independent hindcast initialized in init_month (no cross-year hindcasts).
    Pass initialization_year for forecasts extending over New Year. The caller
    must additionally honour the actual observation release delay.
    """
    month = int(init_month)
    if month < 1 or month > 12:
        raise ValueError('init_month must be in 1..12')
    if isinstance(observations, xr.Dataset):
        if len(observations.data_vars) != 1:
            raise ValueError('Select one observational forcing variable')
        observations = next(iter(observations.data_vars.values()))
    if model.get_index("T").has_duplicates or observations.get_index("T").has_duplicates:
        raise ValueError("Duplicate forcing dates")
    for field in (model, observations):
        if field["T"].dt.calendar not in {"standard", "gregorian", "proleptic_gregorian"}:
            raise ValueError("Forcing completion currently requires a Gregorian calendar")
    target = {d: model[d] for d in ("Y", "X")
              if not observations.get_index(d).equals(model.get_index(d))}
    if target:
        observations = observations.interp(target)
    outputs = []
    model_years = np.unique(model["T"].dt.year.values)
    if initialization_year is not None:
        if np.min(model_years) < int(initialization_year):
            raise ValueError("Model contains dates before its initialization year")
        model_years = np.arange(int(initialization_year), int(np.max(model_years))+1)
    for year in model_years:
        times = pd.date_range(f'{int(year)}-01-01',f'{int(year)}-12-31',freq='D')
        field = model.reindex(T=times)
        cutoff_year = year if initialization_year is None else initialization_year
        cutoff = np.datetime64(f'{int(cutoff_year)}-{month:02d}-01')
        prior = observations.reindex(T=times).where(lambda a: a['T'] < cutoff)
        outputs.append(field.fillna(prior))
    out = xr.concat(outputs,dim='T').transpose('T','Y','X')
    out.attrs.update(model.attrs)
    out.attrs['observations_cutoff'] = f'before initialization month {month:02d}'
    out.attrs['post_initialization_missing'] = 'preserved'
    return out


def c3s_monthly_precipitation_total(data, init_month=None):
    """Integrate C3S monthly mean precipitation rate using valid-month days.

    forecastMonth is one-based: lead 1 is the initialization month. Require
    explicit units; do not silently treat an accumulation as a rate.
    """
    time_name = next((d for d in ('forecast_reference_time','indexing_time','time','T') if d in data.coords), None)
    if 'valid_time' in data.coords:
        valid = data['valid_time']
        if 'forecastMonth' not in valid.dims:
            raise ValueError('valid_time must resolve every forecastMonth')
        days = valid.dt.days_in_month
    elif time_name is not None and 'forecastMonth' in data.coords:
        from calendar import monthrange
        if data[time_name].dt.calendar not in {"standard", "gregorian", "proleptic_gregorian"}:
            raise ValueError("Non-Gregorian monthly rates require explicit valid_time")
        refs = np.atleast_1d(data[time_name].values)
        leads = np.asarray(data.forecastMonth.values)
        if np.any(leads < 1) or np.any(leads != leads.astype(int)):
            raise ValueError('C3S forecastMonth must contain positive integer leads')
        rows = []
        for ref in refs:
            year = int(ref.year) if hasattr(ref,'year') else pd.Timestamp(ref).year
            month = int(ref.month) if hasattr(ref,'month') else pd.Timestamp(ref).month
            row = []
            for lead in leads.astype(int):
                offset = year*12 + month-1 + lead-1
                yr, mo = divmod(offset,12)
                row.append(monthrange(yr,mo+1)[1])
            rows.append(row)
        if data[time_name].ndim == 0:
            days = xr.DataArray(rows[0],dims='forecastMonth',coords={'forecastMonth':data.forecastMonth})
        else:
            days = xr.DataArray(rows,dims=(time_name,'forecastMonth'),coords={time_name:data[time_name],'forecastMonth':data.forecastMonth})
    else:
        raise ValueError('C3S precipitation needs valid_time or initialization and forecastMonth')
    def integrate(a):
        units = str(a.attrs.get('units','')).lower().replace(' ','').replace('**','^')
        factors = {'ms^-1':1000*86400,'ms-1':1000*86400,'m/s':1000*86400,
                   'mms^-1':86400,'mms-1':86400,'mm/s':86400,
                   'kgm^-2s^-1':86400,'kgm-2s-1':86400,
                   'mm/day':1,'mmd^-1':1,'mmday^-1':1,'mmday-1':1,
                   'm/day':1000,'md^-1':1000,'mday^-1':1000,'mday-1':1000}
        if units not in factors:
            raise ValueError(f'Unsupported/missing monthly precipitation RATE units: {a.attrs.get("units")!r}')
        out = (a*days*factors[units]).sum('forecastMonth',skipna=False)
        out.attrs.update(a.attrs,units='mm',source_units=a.attrs['units'],aggregation='sum of valid-month integrated rates')
        return out
    if isinstance(data,xr.Dataset):
        return xr.Dataset({name:integrate(a) for name,a in data.data_vars.items()},attrs=data.attrs)
    return integrate(data)


def c3s_leads_from_offsets(offsets):
    """Convert WASS2S calendar-month offsets (0=initialization) to CDS leads.

    This preserves the season labels and NMME positional convention used by
    WAS_Download_Models. CDS forecastMonth 1 denotes the initialization month.
    """
    result = []
    for offset in offsets:
        value = float(offset)
        if not np.isfinite(value) or value < 0 or value != int(value):
            raise ValueError('Monthly offsets must be nonnegative integers')
        result.append(str(int(value)+1))
    if not result or len(set(result)) != len(result):
        raise ValueError('Monthly offsets must be nonempty and unique')
    return result
