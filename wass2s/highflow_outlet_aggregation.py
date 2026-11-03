"""Common outlet/window aggregation for predictands AND predictors.

No download, CLI or calculation runs on import.  This module never detects a
new season from a predictor: it applies the supplied, source-independent window
catalogue.  Pixel indices are always specific to the input grid.

Public contract
---------------
* Daily fields: T,Y,X (optionally M). T labels the represented calendar day.
* Interval means: additionally pass start/end bounds; do not invent daily data
  by assigning a 90-day mean to its final date.
* Climatological boundaries use a 365-day month/day calendar, but integration
  uses the actual Gregorian dates, including 29 February.
* Outlet identifiers and physical coordinates are preserved. Shared pixels do
  not merge outlets. Nothing is spatially interpolated between river pixels.
"""
from __future__ import annotations
from dataclasses import dataclass, field
from typing import Any, Sequence
import json
import hashlib
import warnings
import numpy as np
import pandas as pd
import xarray as xr

if __package__:
    from .highflow_utils import window_dates, canonical_doy
else:
    from highflow_utils import window_dates, canonical_doy


@dataclass(frozen=True)
class AggregationConfig:
    """Explicit temporal semantics; coverage is based on EXPECTED duration.

    ``daily`` means that each T represents one complete day, even when some
    dates are absent from the file. A daily rate retains its units after mean.
    ``interval_mean`` requires bounds or interval_start/end coordinates.
    Incomplete accepted means remain available-duration means, not imputations.
    """
    statistic: str = 'mean'                 # mean, max, sum (daily amounts only)
    min_valid_fraction: float = 0.80
    time_support: str = 'daily'              # daily, interval_mean
    partial_interval_policy: str = 'reject' # reject, constant (explicit approximation)
    max_missing_run_days: int | None = None
    negative_as_missing: bool = False       # generic: temperature may be negative
    missing_values: tuple[float, ...] = (-99., -999.)
    xarray_layout: str = 'auto'              # auto, sparse T,Y,X, outlet T,outlet
    max_sparse_cells: int = 10_000_000      # all targets and members included
    max_sparse_bytes: int | None = 512 * 1024**2  # conservative formatting estimate
    # Only the OUTPUT index can be moved; extraction always uses true X/Y.
    sparse_coordinate_policy: str = 'raise'  # raise, increment
    coordinate_increment: float = 0.00001   # degrees, output X index only

    def validate(self):
        if self.sparse_coordinate_policy not in {'raise', 'increment'}:
            raise ValueError('sparse_coordinate_policy: raise or increment')
        if (not np.isfinite(self.coordinate_increment)
                or not 0 < self.coordinate_increment <= 0.001):
            raise ValueError('coordinate_increment must be in (0, 0.001] degrees')
        if self.statistic not in {'mean','max','sum'}: raise ValueError('statistic: mean, max or sum')
        if not 0 < self.min_valid_fraction <= 1: raise ValueError('min_valid_fraction must be in (0,1]')
        if self.time_support not in {'daily','interval_mean'}: raise ValueError('Unsupported time_support')
        if self.partial_interval_policy not in {'reject','constant'}: raise ValueError('Unsupported partial_interval_policy')
        if self.xarray_layout not in {'auto','sparse','outlet'}:
            raise ValueError('xarray_layout: auto, sparse or outlet')
        if isinstance(self.max_sparse_cells,bool) or int(self.max_sparse_cells)!=self.max_sparse_cells or self.max_sparse_cells < 1:
            raise ValueError('max_sparse_cells must be a positive integer')
        if self.max_sparse_bytes is not None and (isinstance(self.max_sparse_bytes,bool)
                or int(self.max_sparse_bytes)!=self.max_sparse_bytes or self.max_sparse_bytes < 1):
            raise ValueError('max_sparse_bytes must be a positive integer or None')
        if self.time_support == 'interval_mean' and self.statistic != 'mean':
            raise ValueError('Interval-mean inputs support mean only; prepare daily amounts/maxima explicitly')
        if self.max_missing_run_days is not None and self.max_missing_run_days < 0:
            raise ValueError('max_missing_run_days must be nonnegative or None')


@dataclass
class OutletAggregation:
    """Compact numerical result plus audit tables, before optional formatting."""
    values: xr.DataArray                    # T,[M],outlet,target
    coverage: xr.DataArray
    expected_days: xr.DataArray              # T,outlet,target
    diagnostics: pd.DataFrame
    outlets: pd.DataFrame
    windows: pd.DataFrame
    mapping: pd.DataFrame
    config: AggregationConfig
    variable: str


def as_field(data: xr.Dataset | xr.DataArray, variable: str | None = None) -> xr.DataArray:
    """Choose a physical field without mistaking UPAREA for discharge."""
    if isinstance(data, xr.Dataset):
        if variable is None:
            preferred = [v for v in ('Q','dis24','discharge','river_discharge') if v in data.data_vars]
            if len(preferred) == 1: variable = preferred[0]
            elif len(data.data_vars) == 1: variable = next(iter(data.data_vars))
            else: raise ValueError('Specify variable for a multi-variable Dataset')
        da = data[variable]
        da = da.assign_attrs(dict(data.attrs, **da.attrs))
    elif isinstance(data, xr.DataArray):
        if variable is not None and data.name not in (None, variable):
            raise ValueError(f'DataArray name {data.name!r} differs from requested {variable!r}')
        da = data
    else: raise TypeError('Expected xarray.DataArray or Dataset')
    aliases = {'time':'T','valid_time':'T','latitude':'Y','lat':'Y','longitude':'X','lon':'X','member':'M','number':'M'}
    ren = {k:v for k,v in aliases.items() if k in da.dims and v not in da.dims}
    da = da.rename(ren)
    if not {'T','Y','X'}.issubset(da.dims): raise ValueError(f'Expected T,Y,X; got {da.dims}')
    if set(da.dims)-{'T','Y','X','M'}:
        raise ValueError('Select initialization/lead/level first; only T,Y,X and optional M are accepted')
    for c in ('T','Y','X'):
        if c not in da.coords or da[c].dims != (c,): raise ValueError(f'Rectilinear coordinate {c} required')
        if not da.get_index(c).is_unique: raise ValueError(f'Duplicate {c} labels; do not mix overlapping forecast initializations')
    if not np.issubdtype(da["T"].dtype, np.datetime64):
        raise ValueError('Use decoded Gregorian valid dates. Convert other model calendars explicitly upstream')
    if np.isnat(da["T"].values).any(): raise ValueError('T contains NaT')
    if not np.isfinite(da.X.values).all() or not np.isfinite(da.Y.values).all():
        raise ValueError('Nonfinite horizontal coordinates')
    if np.any(np.abs(da.Y.values)>90): raise ValueError('Y must contain geographic latitude')
    return da.sortby('T')


def outlet_table(outlets, basins=None, *, level: int | None=None,
                 duplicate_policy: str = 'separate', warn_multiple: bool = True) -> pd.DataFrame:
    """Normalize outlets without modifying their physical coordinates or HYBAS_ID.

    ``HYBAS_ID`` identifies a polygon, NOT always a unique physical pour point:
    official coastal aggregates can have several points with the same basin ID.
    ``outlet_id`` is our independent, stable string key for each physical point.

    Policies
    --------
    Identical repeated points with compatible metadata are collapsed (warning +
    audit). Different points belonging to one basin are kept under ``separate``;
    ``error`` requires the caller to resolve them before proceeding. Existing
    unique outlet_id values are preserved. Automatically generated point keys
    use a coordinate hash, so they do not depend on input row order.

    Area safety
    -----------
    ``outlet_uparea_km2`` is an optional *point-specific* drainage area. For a
    multi-outlet/coastal-aggregate basin, polygon UP_AREA is not assumed to be
    the area of every pour point. Missing point areas yield an explicit matching
    status, not a fabricated area. Physical X/Y and HYBAS_ID remain unchanged.
    """
    if duplicate_policy not in {'separate', 'error'}:
        raise ValueError("duplicate_policy must be 'separate' or 'error'")
    if not isinstance(outlets, pd.DataFrame):
        raise TypeError('outlets must be a DataFrame or GeoDataFrame')
    if not outlets.columns.is_unique:
        raise ValueError('Duplicate outlet column names')
    if hasattr(outlets, 'to_crs') and hasattr(outlets, 'geometry'):
        if outlets.crs is None:
            raise ValueError('Declare the outlet CRS')
        geo = outlets.to_crs(4326)
        if (geo.geometry.isna().any() or geo.geometry.is_empty.any()
                or not geo.geometry.geom_type.eq('Point').all()):
            raise ValueError('Explicit, non-empty Point outlets are required; explode MultiPoints explicitly')
        tab = pd.DataFrame(geo.drop(columns=geo.geometry.name)).copy()
        tab['X'] = geo.geometry.x.to_numpy()
        tab['Y'] = geo.geometry.y.to_numpy()
    else:
        tab = outlets.copy()
        tab = tab.rename(columns={k: v for k, v in
            [('longitude','X'),('latitude','Y'),('lon','X'),('lat','Y')]
            if k in tab and v not in tab})
    names = [c for c in tab if str(c).upper() == 'HYBAS_ID']
    if len(names) != 1:
        raise ValueError('outlets must contain exactly one HYBAS_ID field')
    tab = tab.rename(columns={names[0]: 'HYBAS_ID'}).reset_index(drop=True)
    ids = pd.to_numeric(tab.HYBAS_ID, errors='coerce')
    if (not np.isfinite(ids).all() or not np.equal(ids, np.floor(ids)).all()
            or (ids <= 0).any() or (ids >= np.iinfo('int64').max).any()):
        raise ValueError('HYBAS_ID must be a finite positive integer')
    tab['HYBAS_ID'] = ids.astype('int64')
    if not {'X','Y'}.issubset(tab):
        raise ValueError('outlets require X,Y')
    for c in ('X','Y'):
        tab[c] = pd.to_numeric(tab[c], errors='coerce').astype(float)
        if not np.isfinite(tab[c]).all():
            raise ValueError(f'Invalid outlet {c}')
    if (np.abs(tab.Y) > 90).any() or (np.abs(tab.X) > 180).any():
        raise ValueError('Outlet coordinates must be WGS84 -180..180,-90..90')

    # Collapse only the SAME physical point of the SAME basin. Do not call
    # drop_duplicates(HYBAS_ID): that would delete genuine coastal outlets.
    ignored = {'FID', 'fid', 'OBJECTID', 'objectid', 'index_left', 'index_right',
               'duplicate_rows_removed', 'outlet_point_count',
               'outlet_identity_status', 'outlet_identity_method'}
    kept, audit = [], []
    has_key = 'outlet_id' in tab
    for (hid, x, y), group in tab.groupby(['HYBAS_ID','X','Y'], sort=False):
        row = group.iloc[0].copy()
        for c in group.columns:
            if c in ignored:
                continue
            nonmissing = group[c].dropna()
            if len(nonmissing):
                unique = nonmissing.drop_duplicates()
                if len(unique) > 1:
                    raise ValueError(
                        f'Conflicting duplicate outlet metadata: HYBAS_ID={hid}, '
                        f'X={x}, Y={y}, column={c}. Resolve explicitly; no point was moved.')
                row[c] = unique.iloc[0]
        previous = pd.to_numeric(group.get('duplicate_rows_removed',
                                  pd.Series(0, index=group.index)), errors='coerce').fillna(0)
        row['duplicate_rows_removed'] = int(previous.sum()) + len(group)-1
        kept.append(row)
        if len(group) > 1:
            audit.append(dict(HYBAS_ID=int(hid), X=x, Y=y,
                              action='IDENTICAL_POINT_DUPLICATES_REMOVED',
                              rows_removed=len(group)-1))
    tab = pd.DataFrame(kept, columns=list(tab.columns)+
        ([] if 'duplicate_rows_removed' in tab else ['duplicate_rows_removed'])) if kept else tab.copy()
    tab = tab.reset_index(drop=True)
    tab['HYBAS_ID'] = tab.HYBAS_ID.astype('int64')
    if audit:
        warnings.warn(f"Removed {sum(r['rows_removed'] for r in audit)} repeated identical "
                      "outlet row(s); HYBAS_ID and physical coordinates unchanged.",
                      RuntimeWarning, stacklevel=2)
    count = tab.groupby('HYBAS_ID')['HYBAS_ID'].transform('size')
    # Preserve the multi-outlet flag when a previously normalized catalogue is
    # subsetted to just one point (including at a later predictor extraction).
    known_multi = tab.get('multiple_outlet_basin', pd.Series(False, index=tab.index)).fillna(False).astype(bool)
    multiple = (count > 1) | known_multi
    if duplicate_policy == 'error' and (count > 1).any():
        detail = tab.loc[count > 1, ['HYBAS_ID','X','Y']].to_dict('records')
        raise ValueError(f'Duplicate HYBAS_ID outlets at distinct points: {detail}. '
                         "Use duplicate_policy='separate'; never increment HYBAS_ID.")
    tab['outlet_point_count'] = count.astype(int)
    physical_basin_ids = tab.HYBAS_ID.astype(str)
    if has_key:
        keys = tab.outlet_id.astype('string')
        if keys.isna().any() or keys.str.strip().eq('').any():
            raise ValueError('Explicit outlet_id must be nonempty')
        repeated_keys = keys.duplicated(keep=False)
        # A legacy key equal to the parent ID is upgraded; conflicting custom
        # keys are not silently reassigned to another physical location.
        if (repeated_keys & ~keys.eq(physical_basin_ids)).any():
            raise ValueError('A custom outlet_id refers to different physical points')
    else:
        keys = physical_basin_ids.astype('string')
    methods = []
    for i, row in tab.iterrows():
        if multiple.iloc[i] and str(keys.iloc[i]) == str(int(row.HYBAS_ID)):
            payload = '|'.join(float(v if v != 0 else 0).hex() for v in (row.X, row.Y))
            suffix = hashlib.sha256(payload.encode('ascii')).hexdigest()[:16]
            keys.iloc[i] = f'{int(row.HYBAS_ID)}__p{suffix}'
            methods.append('basin_and_coordinate_hash')
        else:
            methods.append('explicit' if has_key else 'single_basin')
    if keys.duplicated().any():
        raise ValueError('Nonunique outlet_id after point normalization')
    tab['outlet_id'] = keys.astype(str)
    tab['outlet_identity_method'] = methods
    if (count > 1).any() and warn_multiple:
        warnings.warn(f"{tab.loc[count > 1, 'HYBAS_ID'].nunique()} HYBAS_ID group(s) have "
                      "distinct physical outlets. They are kept with separate outlet_id values; "
                      "point-specific drainage areas are required for GloFAS matching.",
                      RuntimeWarning, stacklevel=2)

    if basins is not None:
        bn = [c for c in basins.columns if str(c).upper() == 'HYBAS_ID']
        if len(bn) != 1:
            raise ValueError('basins must contain exactly one HYBAS_ID field')
        b = pd.DataFrame(basins.rename(columns={bn[0]: 'HYBAS_ID'})).copy()
        bid = pd.to_numeric(b.HYBAS_ID, errors='coerce')
        if (not np.isfinite(bid).all() or not np.equal(bid,np.floor(bid)).all()):
            raise ValueError('Invalid basin HYBAS_ID')
        b['HYBAS_ID'] = bid.astype('int64')
        if b.HYBAS_ID.duplicated().any():
            raise ValueError('Duplicate basin HYBAS_ID; dissolve/resolve the polygon layer separately')
        cols = [c for c in ('HYBAS_ID','UP_AREA','SUB_AREA','NEXT_DOWN','MAIN_BAS','DIST_MAIN','COAST') if c in b]
        # Many physical points -> one basin. Topology remains keyed by HYBAS_ID.
        tab = tab.drop(columns=[c for c in cols if c != 'HYBAS_ID' and c in tab])
        tab = tab.merge(b[cols], on='HYBAS_ID', how='left',
                        validate='many_to_one', indicator=True, sort=False)
        if not tab['_merge'].eq('both').all():
            raise ValueError('Some outlets are absent from the supplied basin layer')
        tab = tab.drop(columns='_merge')
    coast = pd.to_numeric(tab.get('COAST', pd.Series(0, index=tab.index)), errors='coerce').eq(1)
    tab['multiple_outlet_basin'] = multiple.to_numpy() | coast.to_numpy()
    # A selected subset can contain only one point of a COAST=1 aggregate.
    # Keep the same coordinate-based identity convention even in that subset.
    for i in tab.index[tab.multiple_outlet_basin & tab.outlet_id.eq(tab.HYBAS_ID.astype(str))]:
        payload = '|'.join(float(v if v != 0 else 0).hex() for v in (tab.at[i,'X'],tab.at[i,'Y']))
        tab.at[i,'outlet_id'] = f"{int(tab.at[i,'HYBAS_ID'])}__p{hashlib.sha256(payload.encode('ascii')).hexdigest()[:16]}"
        tab.at[i,'outlet_identity_method'] = 'basin_and_coordinate_hash'
    tab['outlet_identity_status'] = np.where(tab.multiple_outlet_basin,
            'MULTIPLE_PHYSICAL_OUTLETS', 'UNIQUE_PHYSICAL_OUTLET')
    # Missing explicit areas stay missing. In particular, never assign the
    # polygon's aggregate area to every small independent coastal catchment.
    raw_area = tab.get('outlet_uparea_km2', pd.Series(np.nan, index=tab.index))
    point_area = pd.to_numeric(raw_area, errors='coerce')
    invalid_area = raw_area.notna() & (~np.isfinite(point_area) | (point_area <= 0))
    if invalid_area.any():
        raise ValueError('outlet_uparea_km2 must be positive finite km2 or missing')
    tab['outlet_uparea_km2'] = point_area
    parent_area = pd.to_numeric(tab.get('UP_AREA', pd.Series(np.nan, index=tab.index)), errors='coerce')
    parent_ok = ~tab.multiple_outlet_basin & np.isfinite(parent_area) & (parent_area > 0)
    tab['MATCH_UP_AREA_KM2'] = point_area.where(point_area.notna(), parent_area.where(parent_ok))
    tab['match_area_source'] = np.select([point_area.notna(), parent_ok],
        ['outlet_uparea_km2', 'HYBAS_UP_AREA'], default='missing_point_area')
    tab['outlet_match_status'] = np.where(tab.MATCH_UP_AREA_KM2.notna(), 'OK',
        np.where(tab.multiple_outlet_basin, 'OUTLET_AREA_REQUIRED', 'INVALID_TARGET_AREA'))
    if 'ELEV' not in tab:
        tab['ELEV'] = np.nan
    if level is not None:
        if isinstance(level,bool) or int(level) != level or not 1 <= int(level) <= 12:
            raise ValueError('hybas_level must be 1..12')
        if 'hybas_level' in tab and not tab.hybas_level.eq(int(level)).all():
            raise ValueError('Conflicting hybas levels')
        tab['hybas_level'] = int(level)
    # Decode the BASIN identity, never the internal compound outlet_id.
    hid_text = tab.HYBAS_ID.astype(str)
    if hid_text.str.fullmatch(r'[1-9][0-9]{9}').all() and len(tab):
        levels = hid_text.str[1:3].astype(int).unique()
        if len(levels) != 1:
            raise ValueError('Mixed HydroBASINS levels are not an outlet catalogue')
        if level is not None and levels[0] != int(level):
            raise ValueError('HYBAS_ID encodes another level')
        tab['hybas_level'] = int(levels[0])
    tab.attrs['duplicate_outlet_audit'] = audit
    return tab.reset_index(drop=True)


def _sparse_coordinate_index(tab, config):
    """Allocate a unique output Y/X pair; never edit the outlet table itself.

    Increment only repeated PAIRS, not every repeated latitude or longitude.
    Reserve all genuine pairs first to avoid moving one point onto another.
    Allocation order is stable by outlet_id; the DataFrame order is unchanged.
    """
    config.validate()
    index = tab[['outlet_id','HYBAS_ID','X','Y']].copy().reset_index(drop=True)
    index = index.rename(columns={'X':'X_ORIG','Y':'Y_ORIG'})
    index['X_INDEX'] = index.X_ORIG
    index['Y_INDEX'] = index.Y_ORIG
    pairs = list(zip(index.X_ORIG, index.Y_ORIG))
    if len(set(pairs)) < len(pairs):
        if config.sparse_coordinate_policy == 'raise':
            raise ValueError('Coincident outlets require xarray_layout="outlet" or '
                'sparse_coordinate_policy="increment"; physical coordinates are never moved')
        reserved, seen = set(pairs), set()
        for i in sorted(range(len(index)), key=lambda j: str(index.at[j,'outlet_id'])):
            x,y = pairs[i]
            if (x,y) not in seen:
                seen.add((x,y))
                continue
            found = False
            for n in range(1, len(index)+2):
                for sign in (1,-1):
                    xx = x + sign*n*config.coordinate_increment
                    if -180 <= xx <= 180 and (xx,y) not in reserved:
                        index.at[i,'X_INDEX'] = xx
                        reserved.add((xx,y))
                        found = True
                        break
                if found:
                    break
            if not found:
                raise ValueError('Unable to allocate a unique output X index')
    index['X_OFFSET'] = index.X_INDEX-index.X_ORIG
    index['Y_OFFSET'] = index.Y_INDEX-index.Y_ORIG
    index['coordinate_adjusted'] = (index.X_OFFSET != 0) | (index.Y_OFFSET != 0)
    return index


def _distance(lon1, lat1, lon2, lat2):
    a,b,c,d=np.deg2rad([lon1,lat1,lon2,lat2])
    h=np.sin((d-b)/2)**2+np.cos(b)*np.cos(d)*np.sin((c-a)/2)**2
    return float(6371.0088*2*np.arcsin(np.sqrt(np.clip(h,0,1))))


def extract_at_outlets(data, outlets: pd.DataFrame, *, variable=None, mapping=None,
                       max_distance_km=150.) -> tuple[xr.DataArray,pd.DataFrame]:
    """Vectorized extraction (not a Cartesian product) on this source's grid.

    With mapping, exact source_x/source_y labels are required. Without mapping,
    nearest coordinate labels are selected and their geodesic distance audited.
    Outlets outside the latitude/regional-longitude span are rejected. A global
    longitude axis is recognized from its wrap-around spacing.
    """
    da=as_field(data,variable);tab=outlets.copy()
    if 'outlet_id' not in tab: tab=outlet_table(tab)
    xv=np.asarray(da.X.values,float);yv=np.asarray(da.Y.values,float)
    rows=[]
    supplied=None
    if mapping is not None:
        supplied=mapping.copy();supplied['outlet_id']=supplied.outlet_id.astype(str)
        if supplied.outlet_id.duplicated().any(): raise ValueError('Mapping needs exactly one row per outlet')
        supplied=supplied.set_index('outlet_id')
    for _,r in tab.iterrows():
        status='OK';xi=yi=0;sx=sy=dist=np.nan
        if supplied is not None:
            if str(r.outlet_id) not in supplied.index: status='MISSING_MAPPING'
            else:
                m=supplied.loc[str(r.outlet_id)];status=str(m.get('status','OK'))
                sx=float(m.get('source_x',np.nan));sy=float(m.get('source_y',np.nan))
                if status=='OK':
                    xx=np.flatnonzero(xv==sx);yy=np.flatnonzero(yv==sy)
                    if len(xx)!=1 or len(yy)!=1:
                        raise ValueError('Mapping does not match the source grid exactly; recompute mapping for this grid')
                    xi=int(xx[0]);yi=int(yy[0]);dist=_distance(r.X,r.Y,sx,sy)
        else:
            delta=np.abs((xv-float(r.X)+180)%360-180)
            xi=int(np.argmin(delta));yi=int(np.argmin(abs(yv-float(r.Y))))
            sx,sy=float(xv[xi]),float(yv[yi]);dist=_distance(r.X,r.Y,sx,sy)
            # Regional domain test using a minimal circular arc covering X.
            xs=np.sort(np.mod(xv,360));gaps=np.diff(np.r_[xs,xs[0]+360])
            cut=int(np.argmax(gaps));origin=xs[(cut+1)%len(xs)]
            width=360-gaps[cut];loc=(float(r.X)-origin)%360
            global_x=len(xs)>2 and gaps.max()<=1.6*np.median(gaps)
            if (r.Y<yv.min() or r.Y>yv.max() or (not global_x and loc>width+1e-8)):
                status='OUTSIDE_SOURCE_GRID'
            if max_distance_km is not None and dist>max_distance_km: status='PIXEL_TOO_FAR'
        rows.append(dict(outlet_id=str(r.outlet_id),source_x=sx,source_y=sy,x_index=xi,y_index=yi,
                         distance_km=dist,status=status))
    audit=pd.DataFrame(rows)
    if audit.empty: raise ValueError('Empty outlet catalogue')
    ix=xr.DataArray(audit.x_index.to_numpy(int),dims='outlet',coords={'outlet':tab.outlet_id.values})
    iy=xr.DataArray(audit.y_index.to_numpy(int),dims='outlet',coords={'outlet':tab.outlet_id.values})
    sub=da.isel(X=ix,Y=iy).drop_vars(['X','Y'],errors='ignore')
    good=xr.DataArray(audit.status.eq('OK').to_numpy(),dims='outlet',coords={'outlet':tab.outlet_id.values})
    sub=sub.where(good)
    sub=sub.assign_coords(outlet_lon=('outlet',tab.X.values),outlet_lat=('outlet',tab.Y.values))
    sub.attrs=dict(da.attrs,extraction='source-specific outlet point extraction; no basin spatial mean')
    audit['shared_source_pixel']=audit.duplicated(['source_x','source_y'],keep=False)&audit.status.eq('OK')
    return sub.transpose('T',*(['M'] if 'M' in sub.dims else []),'outlet'),audit


def select_catalog(catalog: pd.DataFrame, *, kind='season', selector='main') -> tuple[pd.DataFrame,list[str]]:
    """Select reference objects, never replace an absent MODE2 by HW1.

    selector='all' forms a UNION, while 'each' returns one target per numbered
    window. Components of an ALL target remain separate rows for date masking.
    """
    if kind not in {'season','mode'}: raise ValueError('kind must be season or mode')
    prefix='HW' if kind=='season' else 'MODE'
    if isinstance(selector,bool): raise ValueError('Boolean is not a mode identifier')
    if not (selector in ('main','all','each') or isinstance(selector,(int,np.integer)) and selector>=1):
        raise ValueError('selector: main, all, each, or a positive integer')
    c=catalog.copy()
    if c.empty:
        c=pd.DataFrame(columns=['outlet_id','kind','window_id','number','is_main','anchor_doy','start_doy','end_doy'])
    c=c[c.kind==kind].copy()
    if len(c):
        required={'outlet_id','kind','window_id','number','is_main','anchor_doy','start_doy','end_doy'}
        if not required.issubset(c):raise ValueError(f'Window catalogue missing {sorted(required-set(c.columns))}')
        c['outlet_id']=c.outlet_id.astype(str)
        if c.duplicated(['outlet_id','kind','window_id']).any():raise ValueError('Duplicate native window identifier')
        for col in ('start_doy','end_doy','anchor_doy'):
            x=pd.to_numeric(c[col],errors='coerce')
            if not np.isfinite(x).all() or not (x==np.floor(x)).all() or not x.between(1,365).all():
                raise ValueError(f'{col} must contain integer climatological days 1..365')
            c[col]=x.astype(int)
        def as_bool(v):
            if isinstance(v,(bool,np.bool_)):return bool(v)
            if isinstance(v,str) and v.lower() in ('true','false'):return v.lower()=='true'
            if pd.isna(v):return False
            raise ValueError('is_main must be boolean, not an arbitrary truthy value')
        c['is_main']=c.is_main.map(as_bool)
    if selector=='each':
        c['target_id']=c.window_id
        names=sorted(c.window_id.unique(),key=lambda s:int(s[len(prefix):])) or [prefix+'1']
    elif selector=='all':c['target_id']=prefix+'_ALL';names=[prefix+'_ALL']
    elif selector=='main':
        c=c[c.is_main.fillna(False).astype(bool)].copy();c['target_id']=prefix+'_MAIN';names=[prefix+'_MAIN']
        if c.outlet_id.duplicated().any():raise ValueError('Multiple MAIN windows for an outlet')
    else:
        c=c[c.number==int(selector)].copy();c['target_id']=prefix+str(int(selector));names=[prefix+str(int(selector))]
    return c,names


def _max_missing_run(valid):
    padded=np.r_[True,valid,True];starts=np.flatnonzero(np.diff(padded.astype(int))==-1)
    ends=np.flatnonzero(np.diff(padded.astype(int))==1)
    return int(np.max(ends-starts)) if len(starts) else 0


def aggregate_series(series: pd.Series, components: Sequence[dict], anchor: int,
                     years: Sequence[int], cfg: AggregationConfig,
                     bounds: tuple[pd.DatetimeIndex,pd.DatetimeIndex] | None=None) -> pd.DataFrame:
    """Reference numerical engine used for both station and grid aggregates."""
    cfg.validate();q=series.sort_index();idx=pd.DatetimeIndex(q.index)
    if not idx.is_unique or idx.hasnans: raise ValueError('Unique finite valid times required')
    if idx.tz is not None: raise ValueError('Use timezone-naive Gregorian times')
    vals=pd.to_numeric(q,errors='coerce').to_numpy(dtype=float,copy=True)
    vals[~np.isfinite(vals)]=np.nan
    for v in cfg.missing_values:vals[vals==v]=np.nan
    if cfg.negative_as_missing:vals[vals<0]=np.nan
    q=pd.Series(vals,index=idx)
    if cfg.time_support=='daily' and not idx.equals(idx.normalize()):
        raise ValueError('Daily inputs must be day-labelled at midnight; use interval bounds otherwise')
    if cfg.time_support=='interval_mean':
        if bounds is None: raise ValueError('Interval means require interval_start and interval_end')
        bs,be=map(pd.DatetimeIndex,bounds)
        if len(bs)!=len(q) or len(be)!=len(q) or bs.hasnans or be.hasnans: raise ValueError('Invalid interval bounds')
        if (be<=bs).any() or not bs.is_monotonic_increasing or (bs[1:]<be[:-1]).any():
            raise ValueError('Intervals must be positive, sorted and non-overlapping')
    rows=[]
    for year in years:
        dates=pd.DatetimeIndex([])
        for w in components:dates=dates.union(window_dates(int(year),int(anchor),w))
        dates=dates.sort_values();N=len(dates)
        approximation=False;rejected=False
        if cfg.time_support=='daily':
            arr=q.reindex(dates).values;valid=np.isfinite(arr)
            n=float(valid.sum());coverage=n/N if N else np.nan
            good=N>0 and n>0 and coverage>=cfg.min_valid_fraction
            # Do not count the low-water gap between two ALL components as
            # consecutive missing high-water days.
            splits=np.flatnonzero(np.diff(dates.as_unit('ns').asi8)>86_400_000_000_000)+1 if N else []
            maxgap=max((_max_missing_run(v) for v in np.split(valid,splits)),default=0) if N else 0
            if cfg.max_missing_run_days is not None:good=good and maxgap<=cfg.max_missing_run_days
            if good:
                value={'mean':np.nanmean,'max':np.nanmax,'sum':np.nansum}[cfg.statistic](arr)
            else:value=np.nan
        else:
            # Normalize pandas2/3 datetime resolutions before nanosecond arithmetic.
            # Integrate the intersection of each input interval with the UNION
            # of all expected days. No double counting at HW/MODE boundaries.
            overlap=np.zeros(len(q),float)
            if N:
                cuts=np.flatnonzero(np.diff(dates.as_unit('ns').asi8)>86_400_000_000_000)+1
                for chunk in np.split(dates.as_unit('ns').asi8,cuts):
                    lo=chunk[0];hi=chunk[-1]+86_400_000_000_000
                    overlap+=np.maximum(0,np.minimum(be.as_unit('ns').asi8,hi)-np.maximum(bs.as_unit('ns').asi8,lo))/1e9
            seconds=(be.as_unit('ns').asi8-bs.as_unit('ns').asi8)/1e9
            partial=(overlap>0)&(overlap<seconds-1e-6)
            approximation=bool(partial.any())
            rejected=bool(approximation and cfg.partial_interval_policy=='reject')
            ok=np.isfinite(vals);n=float(overlap[ok].sum()/86400)
            coverage=n/N if N else np.nan
            good=N>0 and n>0 and coverage>=cfg.min_valid_fraction and not rejected
            value=float(np.dot(vals[ok],overlap[ok])/overlap[ok].sum()) if good else np.nan
            maxgap=np.nan
            if cfg.max_missing_run_days is not None:
                raise ValueError('max_missing_run_days is currently a daily-input control')
        rows.append(dict(year=int(year),value=float(value),coverage=float(coverage),n_valid_days=n,
                         n_expected_days=N,max_missing_run_days=maxgap,
                         start_date=dates[0] if N else pd.NaT,end_date=dates[-1] if N else pd.NaT,
                         status='PARTIAL_INTERVAL_REJECTED' if rejected else ('OK' if good else 'INSUFFICIENT_COVERAGE'),
                         interval_approximation=approximation))
    return pd.DataFrame(rows)


def aggregate_field(data, outlets, windows, *, variable=None, kind='season',selector='main',
                    config=None, mapping=None, years=None, max_distance_km=150.,
                    interval_start=None, interval_end=None) -> OutletAggregation:
    """Extract and aggregate any compatible xarray field on fixed windows.

    Only outlet time series are materialized, never the complete T,Y,X cube.
    Members are processed separately; they are not silently averaged together.
    """
    cfg=config or AggregationConfig();cfg.validate()
    if interval_start is not None or interval_end is not None:
        source_t=next((data[c] for c in ('T','time','valid_time') if c in data.coords),None)
        if source_t is not None and not pd.DatetimeIndex(source_t.values).is_monotonic_increasing:
            raise ValueError('Sort the field and its explicit interval bounds together before aggregation')
    tab=outlets.copy() if 'outlet_id' in outlets else outlet_table(outlets)
    da=as_field(data,variable)
    declared=da.attrs.get('time_support')
    if declared in {'interval_mean','interval_amount'} and cfg.time_support=='daily':
        raise ValueError('Input represents intervals, not daily samples; pass interval_mean or use intervals_to_daily')
    if declared=='interval_amount' and cfg.time_support=='interval_mean':
        raise ValueError('Interval amounts are not interval mean rates; convert explicitly')
    sub,mp=extract_at_outlets(da,tab,mapping=mapping,max_distance_km=max_distance_km)
    selected,targets=select_catalog(windows,kind=kind,selector=selector)
    t=pd.DatetimeIndex(sub["T"].values)
    if len(t)==0:raise ValueError('Empty time dimension')
    support_start=t.min();support_end=t.max()
    if cfg.time_support=='interval_mean':
        starts=interval_start if interval_start is not None else sub.coords.get('interval_start')
        ends=interval_end if interval_end is not None else sub.coords.get('interval_end')
        if starts is None or ends is None:raise ValueError('Interval bounds are missing')
        support_start=pd.Timestamp(np.asarray(starts).min())
        support_end=pd.Timestamp(np.asarray(ends).max())-pd.Timedelta(nanoseconds=1)
    if years is None:
        year_list=list(range(support_start.year-1,support_end.year+1))
        # Remove a wholly irrelevant leading/trailing annual label, but keep
        # any season overlapping the data extent so partial coverage is visible.
        used=[]
        for y in year_list:
            if any((d:=window_dates(y,int(r.anchor_doy),r.to_dict()))[-1]>=support_start
                   and d[0]<=support_end for _,r in selected.iterrows()): used.append(y)
        year_list=used or list(range(support_start.year,support_end.year+1))
    else:
        yy=np.asarray(list(years),float)
        if len(yy)==0 or not np.isfinite(yy).all() or not (yy==yy.astype(int)).all():raise ValueError('years must be nonempty integers')
        year_list=sorted(set(yy.astype(int).tolist()))
    dims=['T']+(['M'] if 'M' in sub.dims else [])+['outlet','target']
    coords={'T':pd.to_datetime([f'{y}-01-01' for y in year_list]),'outlet':tab.outlet_id.values,'target':targets}
    if 'M' in sub.dims:coords['M']=sub.M.values
    shape=tuple(len(coords[d]) for d in dims);values=np.full(shape,np.nan);cover=values.copy()
    expected=np.full((len(year_list),len(tab),len(targets)),np.nan)
    records=[]
    bs=interval_start if interval_start is not None else sub.coords.get('interval_start')
    be=interval_end if interval_end is not None else sub.coords.get('interval_end')
    bounds=None
    if cfg.time_support=='interval_mean':
        if bs is None or be is None:raise ValueError('Interval bounds are missing')
        # Explicit bounds correspond to the sorted T input.
        bounds=(pd.DatetimeIndex(np.asarray(bs)),pd.DatetimeIndex(np.asarray(be)))
    for oi,oid in enumerate(tab.outlet_id):
        local=sub.sel(outlet=oid).load()
        members=local.M.values if 'M' in local.dims else [None]
        for ti,target in enumerate(targets):
            c=selected[(selected.outlet_id.astype(str)==str(oid))&(selected.target_id==target)]
            if c.empty:continue
            if c.anchor_doy.nunique()!=1:raise ValueError('An ALL union needs one hydrological anchor per outlet')
            for mi,member in enumerate(members):
                one=local.sel(M=member) if member is not None else local
                q=pd.Series(one.values,index=t)
                stats=aggregate_series(q,c.to_dict('records'),int(c.anchor_doy.iloc[0]),year_list,cfg,bounds)
                ix=(slice(None),mi,oi,ti) if 'M' in sub.dims else (slice(None),oi,ti)
                values[ix]=stats.value.values;cover[ix]=stats.coverage.values
                expected[:,oi,ti]=stats.n_expected_days.values
                for r in stats.to_dict('records'):
                    r.update(outlet_id=str(oid),target_id=target)
                    if member is not None:r['M']=member.item() if hasattr(member,'item') else member
                    records.append(r)
    name=variable or str(da.name or 'VALUE')
    va=xr.DataArray(values,dims=dims,coords=coords,name=name,
                    attrs=dict(da.attrs,statistic=cfg.statistic,time_support=cfg.time_support,
                               aggregation='fixed climatological window; Gregorian expected duration',
                               annual_label='hydrological year containing window START; not physical Jan 1',
                               min_valid_fraction=float(cfg.min_valid_fraction)))
    co=xr.DataArray(cover,dims=dims,coords=coords,name='coverage',attrs={'units':'1'})
    ne=xr.DataArray(expected,dims=['T','outlet','target'],coords={d:coords[d] for d in ['T','outlet','target']},name='expected_days')
    return OutletAggregation(va,co,ne,pd.DataFrame(records),tab,windows.copy(),mp,cfg,name)


def _target_name(variable,target,statistic):
    if target.endswith(('_MAIN','_ALL')):
        family,select=target.split('_');return f'{variable}_{family}_{statistic.upper()}_{select}'
    return f'{target}_{variable}_{statistic.upper()}'


def _resolve_output_layout(result, output_format):
    """Decide ONCE before unstack allocates any dense Cartesian grid.

    'sparse' is a historical label: its numpy buffer includes all NaN holes;
    it is not a COO sparse array. Limits include all targets and members, plus
    an estimate for repeated dense coordinates and a 2x formatting work margin.
    They are budgets, not measurements of free RAM or guarantees for an entire
    Python session (raw/final results and caller objects may coexist).
    """
    cfg=result.config;cfg.validate();tab=result.outlets;v=result.values
    requested=cfg.xarray_layout
    index=None
    physical_pairs=tab[['Y','X']].drop_duplicates().shape[0]
    collision=physical_pairs<len(tab)
    reason='requested_outlet' if requested=='outlet' else 'within_budget'
    ny,nx=int(tab.Y.nunique()),int(tab.X.nunique())
    # Only construct displaced output indices when that policy was requested.
    # Physical positions and the precomputed source-pixel mapping never change.
    if requested!='outlet' and not (collision and cfg.sparse_coordinate_policy=='raise'):
        index=_sparse_coordinate_index(tab,cfg)
        ny,nx=int(index.Y_INDEX.nunique()),int(index.X_INDEX.nunique())
    total=int(v.sizes['T'])*int(v.sizes.get('M',1))*ny*nx*int(v.sizes['target'])
    numeric=total*int(v.dtype.itemsize)
    max_id_chars=max([len(str(i)) for i in tab.outlet_id]+[1])
    # Four numeric 2-D coords: HYBAS_ID, X_ORIG, Y_ORIG, ELEV + OUTLET_ID.
    coordinates=ny*nx*(4*8+max_id_chars*4)*max(1,int(v.sizes['target']))
    estimated=int(2*(numeric+coordinates)+(ny+nx)*8)
    reasons=[]
    if collision and cfg.sparse_coordinate_policy=='raise':reasons.append('coincident_coordinates')
    if total>cfg.max_sparse_cells:reasons.append('max_sparse_cells')
    if cfg.max_sparse_bytes is not None and estimated>cfg.max_sparse_bytes:reasons.append('max_sparse_bytes')
    effective='outlet' if requested=='outlet' else 'sparse'
    needs_xarray=output_format in {'both','xarray'}
    if not needs_xarray:
        effective='outlet';index=None;reason='dataframe_only_no_cartesian_allocation'
    elif requested=='auto' and reasons:
        effective='outlet';index=None;reason=','.join(reasons)
        warnings.warn(f'Cartesian T,Y,X would contain {total:,} values '
            f'(~{estimated/1024**2:.1f} MiB formatting estimate). '
            f'xarray_layout="auto" uses T,outlet ({reason}); '
            'outlet coordinates, matching and seasonal values are unchanged.',
            RuntimeWarning,stacklevel=3)
    elif requested=='sparse' and reasons:
        if 'coincident_coordinates' in reasons:
            raise ValueError('Coincident outlets require xarray_layout="outlet" or '
                'sparse_coordinate_policy="increment"; physical coordinates are never moved')
        raise MemoryError(f'Sparse T,Y,X would create {total:,} cells '
            f'(~{estimated/1024**2:.1f} MiB estimated formatting memory). '
            'Use xarray_layout="auto" or "outlet", or explicitly raise the budgets.')
    if effective=='outlet':index=None
    diagnostics=dict(requested_layout=requested,resolved_layout=effective,
        reason=reason,n_times=int(v.sizes['T']),n_outlets=int(v.sizes['outlet']),
        n_members=int(v.sizes.get('M',1)),n_targets=int(v.sizes['target']),
        n_unique_y=ny,n_unique_x=nx,physical_coordinate_collisions=int(len(tab)-physical_pairs),
        cartesian_cells=total,compact_cells=int(v.size),numeric_bytes=numeric,
        estimated_sparse_formatting_bytes=estimated,
        max_sparse_cells=int(cfg.max_sparse_cells),max_sparse_bytes=cfg.max_sparse_bytes)
    return effective,index,diagnostics


def format_aggregation(result: OutletAggregation, *, output_format='both') -> dict:
    """Format without modifying extraction coordinates or scientific values.

    Default 'auto': T,Y,X only when its complete Cartesian representation fits
    the configured budgets, otherwise T,outlet. DataFrames always use real X/Y.
    Explicit 'sparse' remains a strict request and raises before allocation if
    unsafe. This function can reformat result['aggregation'] without recomputing
    climatology, matching or yearly means.
    """
    if output_format not in {'both','xarray','dataframe'}:
        raise ValueError('output_format: both, xarray, dataframe')
    tab=result.outlets;v=result.values;dataset={};tables={}
    layout,coordinate_index,layout_diagnostics=_resolve_output_layout(result,output_format)
    for target in v.target.values:
        name=_target_name(result.variable,str(target),result.config.statistic)
        arr=v.sel(target=target,drop=True).rename(name)
        arr=arr.assign_coords(HYBAS_ID=('outlet',tab.HYBAS_ID.to_numpy('int64')),
                              X_ORIG=('outlet',tab.X.to_numpy()),Y_ORIG=('outlet',tab.Y.to_numpy()),
                              ELEV=('outlet',pd.to_numeric(tab.ELEV,errors='coerce').to_numpy()),
                              OUTLET_ID=('outlet',tab.outlet_id.astype(str).to_numpy()))
        arr.attrs=dict(arr.attrs,xarray_layout_requested=result.config.xarray_layout,
                       xarray_layout_resolved=layout)
        if output_format in {'both','xarray'} and layout=='sparse':
            spatial=arr.assign_coords(Y=('outlet',coordinate_index.Y_INDEX.values),
                                      X=('outlet',coordinate_index.X_INDEX.values))
            spatial=spatial.set_index(outlet=['Y','X']).unstack('outlet')
            spatial=spatial.transpose('T',*(['M'] if 'M' in spatial.dims else []),'Y','X')
            # Empty Cartesian cells have no outlet ID. Encode the absence as an
            # empty string explicitly; NetCDF cannot round-trip object NaNs as
            # missing strings reliably. Numeric values remain NaN in the holes.
            spatial=spatial.assign_coords(OUTLET_ID=spatial.OUTLET_ID.fillna('').astype(str))
            spatial.attrs = dict(spatial.attrs,
                coordinate_index_policy=result.config.sparse_coordinate_policy,
                coordinate_increment_degrees=float(result.config.coordinate_increment),
                physical_coordinate_fields='X_ORIG Y_ORIG',
                coordinates_adjusted_for_index=int(coordinate_index.coordinate_adjusted.sum()))
            if coordinate_index.coordinate_adjusted.any():
                spatial.X.attrs['comment'] = 'Index may be offset; X_ORIG contains true longitude'
                spatial.Y.attrs['comment'] = 'Index latitude; Y_ORIG contains true latitude'
            dataset[name]=spatial
        else:
            # Paired geographic coordinates, not independent grid dimensions.
            dataset[name]=arr.assign_coords(X=('outlet',tab.X.to_numpy()),Y=('outlet',tab.Y.to_numpy()))
        if output_format in {'both','dataframe'}:
            members=arr.M.values if 'M' in arr.dims else [None]
            for member in members:
                a=arr.sel(M=member) if member is not None else arr
                ids=[str(i) for i in tab.outlet_id]
                rows=[['LON',*tab.X.values],['LAT',*tab.Y.values],['DAILY/ELEV',*tab.ELEV.values]]
                for date,row in zip(pd.DatetimeIndex(a["T"].values),a.transpose('T','outlet').values):
                    rows.append([date.strftime('%Y%m%d'),*row])
                df=pd.DataFrame(rows,columns=['ID',*ids])
                df.attrs=dict(a.attrs,variable=name,member=str(member) if member is not None else '',
                              physical_coordinates='unmodified HYBAS outlets')
                key=name if member is None else f'{name}__M{member}'
                tables[key]=df
    out=dict(outlet_xarray=v,coverage=result.coverage,expected_days=result.expected_days,
             annual_table=result.diagnostics,outlets=tab,windows=result.windows,mapping=result.mapping,
             aggregation=result, coordinate_index=coordinate_index,
             resolved_xarray_layout=layout,layout_diagnostics=layout_diagnostics)
    if output_format in {'both','xarray'}:
        out['xarray']=next(iter(dataset.values())) if len(dataset)==1 else xr.Dataset(dataset)
    if output_format in {'both','dataframe'}:
        out['dataframe']=next(iter(tables.values())) if len(tables)==1 else tables
    return out
