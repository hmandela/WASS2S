"""Shared safeguards for the high-flow modules (no CLI and no implicit I/O).

Conventions
-----------
Discharge is a daily mean rate. Missing dates are missing observations, never
zero flow. The 365-day calendar is used ONLY to define climatological bounds;
aggregation always uses actual Gregorian dates, including 29 February.
"""
from __future__ import annotations
from typing import Any, Sequence
import re
import numpy as np
import pandas as pd
import xarray as xr

DAYS = 365

def area_to_km2(area: xr.DataArray, *, assumed_units: str | None = None) -> xr.DataArray:
    """Convert explicit area units. Never infer units from magnitude.

    An upstream area above 100,000 km² is entirely possible. A numerical
    threshold is therefore NOT a safe substitute for the units attribute.
    Set ``area.attrs['units']`` or pass ``assumed_units`` for stripped files.
    """
    original = str(area.attrs.get('units', '') or assumed_units or '')
    unit = original.strip().casefold().replace('²','2').replace('**','').replace('^','')
    unit = re.sub(r'[\s_]+', '', unit)
    scales = {'m2':1e-6,'squaremeters':1e-6,'squaremetres':1e-6,
              'squaremeter':1e-6,'squaremetre':1e-6,
              'km2':1.,'squarekilometers':1.,'squarekilometres':1.,
              'squarekilometer':1.,'squarekilometre':1.}
    if unit not in scales:
        raise ValueError(f"Unknown/missing upstream-area units {original!r}; set units='m2' or 'km2' explicitly.")
    out = area.astype(float) * scales[unit]
    out = out.where(np.isfinite(out) & (out > 0))
    out.attrs = dict(area.attrs, units='km2', original_units=original,
                     conversion_note='Exact-unit conversion; no magnitude inference')
    return out

def align_spatial_grid(field: xr.DataArray, template: xr.DataArray,
                       y: str = 'Y', x: str = 'X') -> xr.DataArray:
    """Align a labelled 2-D field to the template without interpolation.

    Reversed axes are reordered. Missing coordinates, duplicate labels, and
    genuinely shifted grids are rejected. A larger auxiliary grid may be
    subsetted, but a nearest-neighbour guess is never made.
    """
    ren = {}
    for target, candidates in ((y, ('Y','y','latitude','lat')), (x, ('X','x','longitude','lon'))):
        if target not in field.dims:
            found = next((d for d in candidates if d in field.dims),None)
            if found is None: raise ValueError(f'Cannot find horizontal dimension {target!r}.')
            ren[found] = target
    a = field.rename(ren)
    for d in list(a.dims):
        if d not in (y,x):
            if a.sizes[d]!=1: raise ValueError(f'Area/mask must be static: extra dimension {d}.')
            a=a.isel({d:0},drop=True)
    for d in (y,x):
        if d not in a.coords or d not in template.coords: raise ValueError(f'Missing labelled coordinate {d}.')
        if a[d].ndim!=1 or template[d].ndim!=1: raise ValueError('Only rectilinear grids are supported.')
        if not a.get_index(d).is_unique or not template.get_index(d).is_unique:
            raise ValueError(f'Duplicate coordinates on {d}.')
        if not template.get_index(d).isin(a.get_index(d)).all():
            raise ValueError(f'Auxiliary grid is not exactly aligned on {d}; regrid explicitly upstream.')
    a=a.sel({y:template[y],x:template[x]}).transpose(y,x)
    xr.align(a, template, join='exact', copy=False)
    return a

def clean_daily_series(series: pd.Series, *, negative_to_nan: bool=True,
                       missing_values: Sequence[float]=(-99.,-999.)) -> pd.Series:
    """Numeric QC, exact sentinels, duplicate-day averaging, daily reindexing.

    No interpolation is performed. Intraday timestamps are aggregated to their
    day; this assumes samples of a daily mean, NOT discharge volumes.
    """
    if not isinstance(series,pd.Series): raise TypeError('Expected a pandas.Series.')
    if not isinstance(series.index,pd.DatetimeIndex):
        raise TypeError('A Gregorian pandas.DatetimeIndex is required; convert model calendars explicitly.')
    if series.index.tz is not None: raise ValueError('Use timezone-naive daily hydrological dates.')
    v=pd.to_numeric(series,errors='coerce').to_numpy(dtype=float,na_value=np.nan,copy=True)
    v[~np.isfinite(v)]=np.nan
    for sentinel in missing_values: v[v==float(sentinel)]=np.nan
    if negative_to_nan:v[v<0]=np.nan
    valid=series.index.notna()
    q=pd.Series(v[valid],index=series.index[valid].normalize(),name=series.name)
    q=q.groupby(level=0).mean().sort_index()
    if q.empty:return q
    return q.reindex(pd.date_range(q.index.min(),q.index.max(),freq='D'))

def canonical_doy(index: pd.DatetimeIndex) -> np.ndarray:
    """365-day month/day coordinate, with 29 February mapped to day 59.

    This mapping is only for labels/masks. Expected annual counts and maxima
    must use ``window_dates`` so leap days cannot disappear from targets.
    """
    i=pd.DatetimeIndex(index)
    return np.asarray(i.dayofyear - (i.is_leap_year & (i.month>2)).astype(int)
                      - ((i.month==2)&(i.day==29)).astype(int),dtype=int)

def wrap_doy(value: float) -> int:
    """Round FIRST, wrap SECOND: the result is always an integer in 1..365."""
    if not np.isfinite(value): raise ValueError('Day of year must be finite.')
    return int(np.rint(value-1)) % DAYS + 1

def doy_to_md(doy: int) -> str:
    if not 1<=int(doy)<=365: raise ValueError(f'Day outside 1..365: {doy}')
    return (pd.Timestamp('2001-01-01')+pd.Timedelta(days=int(doy)-1)).strftime('%m-%d')

def hydrological_year_bounds(year: int, anchor_doy: int) -> tuple[pd.Timestamp,pd.Timestamp]:
    md=doy_to_md(anchor_doy)
    return pd.Timestamp(f'{int(year)}-{md}'),pd.Timestamp(f'{int(year)+1}-{md}')-pd.Timedelta(days=1)

def window_dates(year: int, anchor_doy: int, window: dict[str, Any]) -> pd.DatetimeIndex:
    """Actual daily dates of ONE seasonal occurrence.

    The year labels the hydrological year in which the WINDOW START occurs.
    A window crossing the hydrological anchor is kept whole and can extend
    into the following hydrological year; it is never split or dropped.
    ``start_pos`` and ``end_pos`` may wrap (end<start), or end may exceed 365.
    """
    if 'start_doy' in window and 'end_doy' in window:
        s,e=int(window['start_doy']),int(window['end_doy'])
    else:
        s=(int(anchor_doy)+int(window['start_pos'])-2)%365+1
        e=(int(anchor_doy)+int(window['end_pos'])-2)%365+1
    sm,em=doy_to_md(s),doy_to_md(e)
    sy=int(year)+int(s<int(anchor_doy))
    start=pd.Timestamp(f'{sy}-{sm}')
    end=pd.Timestamp(f'{sy+int(e<s)}-{em}')
    return pd.date_range(start,end,freq='D')

def union_window_dates(year: int, anchor_doy: int, windows: Sequence[dict]) -> pd.DatetimeIndex:
    out=pd.DatetimeIndex([])
    for w in windows:out=out.union(window_dates(year,anchor_doy,w))
    return out.sort_values()

def coverage_statistics(series: pd.Series, expected: pd.DatetimeIndex,
                        min_mean: float=.8,min_max: float=.9) -> dict:
    """Statistics on the expected calendar, not only the available rows.

    The accepted mean is the mean of available observations. Coverage thresholds
    limit missingness; they do not prove the missing values are unbiased.
    """
    q=series.reindex(expected); n=int(q.notna().sum()); N=len(expected)
    frac=n/N if N else np.nan
    mok=bool(N>0 and n>0 and frac>=min_mean); xok=bool(N>0 and n>0 and frac>=min_max)
    return dict(Q_MEAN=float(q.mean()) if mok else np.nan,
                Q_MAX=float(q.max()) if xok else np.nan,
                PEAK_DATE=q.idxmax() if xok else pd.NaT,
                N_VALID=n,N_EXPECTED=N,VALID_FRACTION=frac,MEAN_OK=mok,MAX_OK=xok)

def circular_bias_mean(values, weights=None, *, min_resultant=.1) -> float:
    """Circular mean of signed day shifts; antipodal mixtures are undefined."""
    a=np.asarray(values,dtype=float);w=np.ones(len(a)) if weights is None else np.asarray(weights,dtype=float)
    ok=np.isfinite(a)&np.isfinite(w)&(w>=0)
    if not ok.any() or w[ok].sum()<=0:return np.nan
    w=w[ok]/w[ok].sum();z=np.dot(w,np.exp(2j*np.pi*a[ok]/365.))
    if abs(z)<min_resultant:return np.nan
    return float((np.angle(z)*365/(2*np.pi)+182.5)%365-182.5)

def unwrap_biases(values, *, min_resultant=.1):
    """Unwrap near a common circular centre for linear spatial estimators."""
    a=np.asarray(values,dtype=float);centre=circular_bias_mean(a,min_resultant=min_resultant)
    if not np.isfinite(centre):raise ValueError('Ambiguous/antipodal timing biases; do not pool these modes.')
    return centre+(a-centre+182.5)%365-182.5

def window_day_set(window: dict, anchor_doy: int=1) -> set[int]:
    if 'start_doy' in window:
        s,e=int(window['start_doy']),int(window['end_doy'])
    else:
        s=(anchor_doy+int(window['start_pos'])-2)%365+1
        e=(anchor_doy+int(window['end_pos'])-2)%365+1
    return {((s-1+k)%365)+1 for k in range((e-s)%365+1)}

def match_windows(reference: Sequence[dict], candidates: Sequence[dict], *,
                  min_overlap=.2, reference_anchor=1, candidate_anchor=1):
    """One-to-one maximum-overlap assignment, independent of local mode IDs.

    Returns (reference index, candidate index, Jaccard overlap). Dummy zero
    assignments allow a mode to be absent; no substitution by HW1 is made.
    """
    if not reference or not candidates:return []
    from scipy.optimize import linear_sum_assignment
    scores=np.zeros((len(reference),len(candidates)))
    for i,a in enumerate(reference):
        A=window_day_set(a,reference_anchor)
        for j,b in enumerate(candidates):
            B=window_day_set(b,candidate_anchor)
            scores[i,j]=len(A&B)/len(A|B) if A|B else 0.
    n=len(reference)+len(candidates);mat=np.zeros((n,n))
    mat[:len(reference),:len(candidates)]=np.where(scores>=min_overlap,scores,0.)
    rows,cols=linear_sum_assignment(-mat)
    return [(int(i),int(j),float(scores[i,j])) for i,j in zip(rows,cols)
            if i<len(reference) and j<len(candidates) and scores[i,j]>=min_overlap and scores[i,j]>0]
