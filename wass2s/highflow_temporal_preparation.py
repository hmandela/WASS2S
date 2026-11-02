"""Optional preparation of accumulated runoff, without a downloader or CLI.

Use on RAW accumulated runoff from ONE forecast initialization. A flattened
already-processed archive cannot recover missing accumulations or initialization
metadata that an older downloader discarded. The historical downloader is not
monkey-patched by this module.
"""
from __future__ import annotations
import numpy as np
import pandas as pd
import xarray as xr


def prepare_runoff_intervals(accumulated: xr.DataArray, *, initialization,
        lead_dim='leadtime', output_units='m3 s-1', input_units=None,
        cell_area_m2: xr.DataArray | None=None, negative_tolerance_m=1e-10):
    """Deaccumulate ONE model initialization, retaining exact interval bounds.

    Accumulations must be in metres of water. Explicit input_units='m' is allowed
    for a file with missing unit metadata, not for relabelling a different unit.
    Leads are timedeltas or hours. Forecast lead 0 has no positive interval and
    is removed. The first nonzero lead is an interval since initialization; it
    is NOT silently made into one day. Missing adjacent cumulative values yield
    missing increments. Large negative increments become NaN, not zeros.

    output_units='m3 s-1': local grid-cell volume RATE, not routed discharge.
    output_units='mm': amount over interval; not a rate and not a daily total
    unless the interval is exactly a day. Members are preserved.
    """
    if not isinstance(accumulated,xr.DataArray):raise TypeError('Select one accumulated DataArray')
    da=accumulated.copy(deep=False)
    if lead_dim not in da.dims:raise ValueError(f'Missing {lead_dim} dimension')
    if set(da.dims)-{lead_dim,'Y','X','M'}:
        raise ValueError('Select ONE initialization and level; expected leadtime,Y,X and optional M')
    if output_units not in {'m3 s-1','mm'}:raise ValueError('output_units: m3 s-1 or mm')
    u=str(da.attrs.get('units','') or input_units or '').strip().lower()
    if u not in {'m','metre','meter','metres','meters'}:raise ValueError('Accumulated runoff must be in metres; do not infer units from magnitude')
    lt=da[lead_dim]
    if np.issubdtype(lt.dtype,np.timedelta64):secs=np.asarray(lt/np.timedelta64(1,'s'),float)
    else:secs=np.asarray(lt,float)*3600.
    if len(secs)==0 or not np.isfinite(secs).all() or (secs<0).any() or (np.diff(secs)<=0).any():
        raise ValueError('Leads must be distinct, increasing nonnegative durations')
    first=da.isel({lead_dim:slice(0,1)})
    increments=xr.concat([first,da.diff(lead_dim)],dim=lead_dim).assign_coords({lead_dim:da[lead_dim]})
    increments=increments.where(increments>=-float(negative_tolerance_m)).clip(min=0)
    prev=np.r_[0.,secs[:-1]];duration=secs-prev;keep=duration>0
    increments=increments.isel({lead_dim:np.flatnonzero(keep)})
    origin=pd.Timestamp(initialization)
    if origin.tzinfo is not None:raise ValueError('Use a timezone-naive initialization')
    begin=origin+pd.to_timedelta(prev[keep],unit='s');end=origin+pd.to_timedelta(secs[keep],unit='s')
    if output_units=='m3 s-1':
        if cell_area_m2 is None:
            if min(da.sizes.get('Y',0),da.sizes.get('X',0))<2:raise ValueError('Supply cell_area_m2 for a singleton grid')
            lat=np.asarray(da.Y,float);lon=np.asarray(da.X,float)
            dy=abs(np.diff(lat));dx=abs(np.diff(lon))
            if not np.allclose(dy,dy[0]) or not np.allclose(dx,dx[0]):raise ValueError('Irregular grid: supply actual cell_area_m2')
            north=np.deg2rad(np.minimum(lat+dy[0]/2,90));south=np.deg2rad(np.maximum(lat-dy[0]/2,-90))
            cell_area_m2=xr.DataArray(6371000.**2*np.deg2rad(dx[0])*(np.sin(north)-np.sin(south)),dims='Y',coords={'Y':da.Y})
        if set(cell_area_m2.dims)-{'Y','X'}:raise ValueError('Cell area must be Y or Y,X')
        for dim in cell_area_m2.dims:
            if not cell_area_m2.get_index(dim).equals(da.get_index(dim)):raise ValueError('Cell area labels must exactly match the source grid')
        if not (np.isfinite(cell_area_m2)&(cell_area_m2>0)).all():raise ValueError('Cell areas must be finite positive m2')
        dt=xr.DataArray(duration[keep],dims=lead_dim,coords={lead_dim:increments[lead_dim]})
        result=increments*cell_area_m2/dt
        quantity='local grid-cell runoff volume rate; not routed river discharge';support='interval_mean'
    else:
        result=increments*1000.;quantity='local runoff water depth accumulated over each interval';support='interval_amount'
    result=result.rename({lead_dim:'T'}).assign_coords(T=end,
        interval_start=('T',begin.values),interval_end=('T',end.values),
        initialization=origin.to_datetime64())
    result=result.transpose('T',*(['M'] if 'M' in result.dims else []),'Y','X')
    result.name=da.name or 'RUNOFF'
    result.attrs=dict(da.attrs,units=output_units,quantity=quantity,time_support=support,
        preparation='deaccumulated within one initialization; missing increments retained',
        negative_increment_policy='small rounding negatives clipped, larger negatives missing')
    return result


def intervals_to_daily(data: xr.DataArray) -> xr.DataArray:
    """Relabel genuinely 24-hour midnight intervals by their START day.

    This refuses subdaily, multiday, overlapping and undated intervals. It does
    not disaggregate them, and it never changes the numerical values.
    """
    if 'interval_start' not in data.coords or 'interval_end' not in data.coords:
        raise ValueError('interval_start/end coordinates required')
    a=pd.DatetimeIndex(data.interval_start.values);b=pd.DatetimeIndex(data.interval_end.values)
    if not np.all((b-a)==pd.Timedelta(days=1)) or not a.equals(a.normalize()):
        raise ValueError('Only exact midnight-to-midnight 24-hour intervals can be day-labelled')
    if not a.is_unique or not a.is_monotonic_increasing:raise ValueError('Overlapping or unordered intervals')
    out=data.assign_coords(T=a).drop_vars(['interval_start','interval_end'])
    out.attrs=dict(data.attrs,time_support='daily',date_label='represented day START, not accumulation endpoint')
    return out
