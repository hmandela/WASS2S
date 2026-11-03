"""Compact PCRaster-LDD topology on a labelled native GloFAS grid.

Codes are geographic directions, not array-index offsets. All arrays are
normalized to increasing X/Y; 2 (south) therefore DECREASES the row index.
No interpolation or repair is ever performed. 5 denotes a terminal cell,
not necessarily the sea. A cropped path is UNKNOWN, never disconnected.

The network may be larger than the discharge domain. It is held as NumPy
integer arrays, not a Python object/NetworkX node per raster pixel. The reverse
CSR adjacency is built only when an upstream catchment is requested.
"""
from __future__ import annotations
from dataclasses import dataclass
from collections import OrderedDict
import hashlib
import re
import warnings
import numpy as np
import pandas as pd
import xarray as xr

if __package__:
    from .highflow_utils import area_to_km2
else:
    from highflow_utils import area_to_km2

# Row/column increments with north=increasing Y and east=increasing X.
LDD_OFFSETS = {1: (-1,-1), 2: (-1,0), 3: (-1,1), 4: (0,-1),
               5: (0,0), 6: (0,1), 7: (1,-1), 8: (1,0), 9: (1,1)}
PIT, OUTSIDE, MISSING, INVALID = -1, -2, -3, -4


def normalize_static(field, name=None):
    """Select a static variable, normalize aliases, sort without interpolation."""
    if isinstance(field, xr.Dataset):
        names = [name] if name else []
        names += [k for k, v in field.data_vars.items() if v.ndim >= 2]
        key = next((k for k in names if k in field), None)
        if key is None:
            raise ValueError('Select a two-dimensional static variable')
        if name is None and len([k for k,v in field.data_vars.items() if v.ndim>=2]) != 1:
            raise ValueError('Select the desired DataArray from the static Dataset')
        field = field[key].assign_attrs(dict(field.attrs, **field[key].attrs))
    if not isinstance(field, xr.DataArray):
        raise TypeError('Static maps must be xarray DataArray/Dataset')
    rename = {}
    for target, names in [('Y', ('Y','y','lat','latitude')), ('X', ('X','x','lon','longitude'))]:
        found = next((d for d in names if d in field.dims), None)
        if found is None:
            raise ValueError(f'Missing {target} dimension')
        if found != target:
            rename[found] = target
    a = field.rename(rename)
    for d in set(a.dims)-{'Y','X'}:
        if a.sizes[d] != 1:
            raise ValueError(f'Static field has non-singleton {d}')
        a = a.isel({d:0}, drop=True)
    for d in ('Y','X'):
        v = np.asarray(a[d].values, dtype=float)
        if v.ndim != 1 or not np.isfinite(v).all() or len(np.unique(v)) != len(v):
            raise ValueError(f'{d} must contain finite unique rectilinear coordinates')
        # Angular native grid only. Do not treat projected metres as degrees.
        if np.any(np.abs(v) > (90 if d == 'Y' else 360)):
            raise ValueError('Expected geographic longitude/latitude, not a projected CRS')
    return a.sortby('Y').sortby('X').transpose('Y','X')


def grid_fingerprint(field):
    a = normalize_static(field.isel(T=0, drop=True) if 'T' in field.dims else field)
    h = hashlib.sha256()
    for d in ('Y','X'):
        h.update(d.encode()); h.update(np.asarray(a[d], dtype='<f8').tobytes())
    return h.hexdigest()


def declared_version(field):
    """Only explicit GloFAS keys; static-catalogue revisions are NOT model versions."""
    for key in ('glofas_version', 'glofas_system_version', 'system_version'):
        value = field.attrs.get(key)
        if value:
            m = re.search(r'(?:version_|v)?(\d+)[_.](\d+)', str(value).lower())
            return f'{int(m[1])}.{int(m[2])}' if m else str(value)
    return None


def assert_versions(fields, version=None, strict=False):
    entries = {k: declared_version(v) for k,v in fields.items() if v is not None}
    if version is not None:
        dummy=xr.DataArray(0, attrs={'glofas_version':version})
        expected=declared_version(dummy)
        if any(v is not None and v != expected for v in entries.values()):
            raise ValueError(f'GloFAS auxiliary version conflict: expected {expected}, got {entries}')
    known = {v for v in entries.values() if v is not None}
    if len(known)>1:
        raise ValueError(f'Mixed GloFAS model versions: {entries}')
    if strict and any(v is None for v in entries.values()):
        raise ValueError(f'Missing GloFAS version metadata: {entries}')
    return dict(declared=entries, expected=version,
                status='KNOWN_COMPATIBLE' if entries and all(entries.values()) else 'UNKNOWN_METADATA')


def exact_subset(field, template, tolerance=0.0):
    """Native-grid alignment, accepting reversed axes but never shifted centres."""
    a=normalize_static(field)
    selectors={}
    for d in ('Y','X'):
        axis=a.get_index(d)
        if tolerance:
            step=float(np.min(np.abs(np.diff(axis)))) if len(axis)>1 else np.inf
            if tolerance<0 or tolerance>step*1e-4:
                raise ValueError('Coordinate tolerance must be tiny relative to a native cell (<1e-4 cell)')
            ix=axis.get_indexer(template[d].values,method='nearest',tolerance=tolerance)
        else:ix=axis.get_indexer(template[d].values)
        if (ix<0).any() or len(np.unique(ix))!=len(ix):
            raise ValueError(f'Static/discharge grid mismatch on {d}; interpolation is forbidden')
        selectors[d]=ix
    out=a.isel(selectors).assign_coords(Y=template.Y,X=template.X).transpose('Y','X')
    return out


@dataclass(frozen=True)
class LDDNetworkConfig:
    max_network_cells: int = 10_000_000
    max_trace_cells: int = 2_000_000
    path_cache_size: int = 128
    domain_complete: bool = False  # explicit user assertion, never inferred from a crop
    require_version_metadata: bool = False
    coordinate_tolerance: float = 0.0  # optional roundoff only, max 1e-4 native cell


class GloFASStaticNetwork:
    """Static, reusable topology and ancillary maps for one native domain.

    `uparea` must cover the LDD domain, which may exceed the Q subset. No field
    is silently extended. `channel_mask=None` is allowed but documented as an
    unverified channel mask. Full-global graphs are refused above the explicit
    memory cap; crop a *topological* domain, not just administrative boundaries.
    """
    def __init__(self, ldd, uparea, channel_mask=None, cell_area=None, *,
                 version=None, config=None):
        self.config=config or LDDNetworkConfig()
        if self.config.max_network_cells<1 or self.config.max_trace_cells<1 or self.config.path_cache_size<0:
            raise ValueError('Invalid network memory/trace limits')
        if not np.isfinite(self.config.coordinate_tolerance) or self.config.coordinate_tolerance < 0:
            raise ValueError('coordinate_tolerance must be finite and nonnegative')
        self.ldd=normalize_static(ldd)
        self.shape=(self.ldd.sizes['Y'],self.ldd.sizes['X']); self.size=int(np.prod(self.shape))
        if not self.size or self.size>self.config.max_network_cells:
            raise MemoryError(f'LDD domain has {self.size:,} cells; subset the static network explicitly')
        for d in ('Y','X'):
            v=self.ldd[d].values
            if len(v)>2 and not np.allclose(np.diff(v),np.diff(v)[0],rtol=1e-7,atol=1e-10):
                raise ValueError('LDD requires contiguous regularly spaced native coordinates')
        self.uparea=exact_subset(area_to_km2(normalize_static(uparea)),self.ldd,self.config.coordinate_tolerance).load()
        self.version_audit=assert_versions({'LDD':self.ldd,'UPAREA':normalize_static(uparea),
            'CHAN':normalize_static(channel_mask) if channel_mask is not None else None,
            'CELLAREA':normalize_static(cell_area) if cell_area is not None else None},
            version,self.config.require_version_metadata)
        self.version=version
        codes=np.asarray(self.ldd.values,dtype=float)
        nodata=~np.isfinite(codes) | np.isin(codes,[0,255])
        # Honour explicit source fill values; values like 16/32 are invalid D8,
        # not silently reinterpreted as missing.
        for key in ('_FillValue','missing_value'):
            for value in np.atleast_1d(self.ldd.attrs.get(key, [])):
                nodata |= codes==value
        if np.any(~nodata & (~np.isin(codes,list(LDD_OFFSETS)))):
            raise ValueError('Invalid PCRaster LDD codes; expected 1..9 (5=terminal). ESRI D8 is not accepted')
        self.valid=(~nodata).ravel()
        self.codes=np.where(nodata,0,codes).astype('uint8')
        self.channel_mask=None
        if channel_mask is not None:
            ch=np.asarray(exact_subset(channel_mask,self.ldd,self.config.coordinate_tolerance).values,dtype=float)
            if np.any(np.isfinite(ch)&~np.isin(ch,[0,1])):
                raise ValueError('CHAN must contain 0/1/missing values')
            self.channel_mask=(ch==1)&(~nodata)
        self.cell_area=None
        if cell_area is not None:
            ca=exact_subset(normalize_static(cell_area),self.ldd,self.config.coordinate_tolerance)
            self.cell_area=np.asarray(area_to_km2(ca).values,dtype=float).ravel()
        self.downstream=np.full(self.size,INVALID,dtype=np.int64)
        ny,nx=self.shape
        for code,(dy,dx) in LDD_OFFSETS.items():
            yy,xx=np.where(self.codes==code); src=yy*nx+xx
            if code==5:
                self.downstream[src]=PIT;continue
            ty,tx=yy+dy,xx+dx
            inside=(ty>=0)&(ty<ny)&(tx>=0)&(tx<nx)
            self.downstream[src[~inside]]=OUTSIDE
            dest=ty[inside]*nx+tx[inside]
            self.downstream[src[inside]]=np.where(self.valid[dest],dest,MISSING)
        self._check_cycles()
        self._cache=OrderedDict();self._reverse=None
        self.fingerprint=self._fingerprint()
        source=np.flatnonzero(self.downstream>=0)
        areas=np.asarray(self.uparea).ravel()
        decreases=np.isfinite(areas[source]) & np.isfinite(areas[self.downstream[source]]) & (areas[self.downstream[source]] < areas[source]*(1-1e-6))
        self.audit=dict(n_cells=self.size,n_valid=int(self.valid.sum()),
            n_uparea_decreases_along_ldd=int(decreases.sum()),
            n_terminal=int((self.downstream==PIT).sum()),
            n_boundary_exit=int((self.downstream==OUTSIDE).sum()),
            n_missing_downstream=int((self.downstream==MISSING).sum()),
            channel_mask='SUPPLIED' if self.channel_mask is not None else 'NOT_SUPPLIED',
            domain_complete_asserted=bool(self.config.domain_complete),
            version=self.version_audit, topology_fingerprint=self.fingerprint)

    def _fingerprint(self):
        h=hashlib.sha256(grid_fingerprint(self.ldd).encode())
        h.update(self.codes.tobytes());h.update(np.asarray(self.uparea,dtype='<f8').tobytes())
        if self.channel_mask is not None:h.update(self.channel_mask.tobytes())
        if self.cell_area is not None:h.update(self.cell_area.astype('<f8').tobytes())
        return h.hexdigest()

    def _check_cycles(self):
        """Kahn elimination with array storage: no per-pixel graph objects."""
        edges=self.downstream[self.downstream>=0]
        indegree=np.bincount(edges,minlength=self.size).astype(np.int64)
        queue=np.empty(int(self.valid.sum()),dtype=np.int64)
        first=np.flatnonzero(self.valid&(indegree==0));tail=len(first);queue[:tail]=first;head=0
        while head<tail:
            i=int(queue[head]);head+=1;j=int(self.downstream[i])
            if j>=0:
                indegree[j]-=1
                if indegree[j]==0:queue[tail]=j;tail+=1
        if tail!=int(self.valid.sum()):
            cells=np.flatnonzero(self.valid&(indegree>0))
            raise ValueError(f'LDD contains directed cycles ({len(cells)} cells); no automatic repair was performed')

    def pixel(self,x,y):
        """Return a native flat index, rejecting nearest-neighbour guesses."""
        opts=({'method':'nearest','tolerance':self.config.coordinate_tolerance} if self.config.coordinate_tolerance else {})
        ix=self.ldd.get_index('X').get_indexer([float(x)],**opts)[0]
        iy=self.ldd.get_index('Y').get_indexer([float(y)],**opts)[0]
        if ix<0 or iy<0:raise ValueError(f'Coordinates {(x,y)} are not native LDD centres')
        return int(iy*self.shape[1]+ix)

    def xy(self,pixel):
        i=int(pixel)
        if not 0<=i<self.size:raise IndexError('Pixel outside LDD domain')
        y,x=divmod(i,self.shape[1]);return float(self.ldd.X[x]),float(self.ldd.Y[y])

    def trace(self,pixel):
        i=int(pixel)
        if i in self._cache:
            self._cache.move_to_end(i);return self._cache[i]
        if not 0<=i<self.size or not self.valid[i]:
            return np.array([],dtype=np.int64),'INVALID_START'
        path=[];p=i
        while p>=0 and len(path)<self.config.max_trace_cells:
            path.append(p);p=int(self.downstream[p])
        status={PIT:'TERMINAL',OUTSIDE:'UNKNOWN_DOMAIN_TRUNCATED',MISSING:'UNKNOWN_MISSING_DOWNSTREAM'}.get(p,'TRACE_LIMIT')
        arr=np.asarray(path,dtype=np.int64);arr.setflags(write=False)
        answer=(arr,status)
        if self.config.path_cache_size:
            self._cache[i]=answer
            while len(self._cache)>self.config.path_cache_size:self._cache.popitem(last=False)
        return answer

    def relationship(self,a,b):
        a,b=int(a),int(b)
        if not (0<=a<self.size and 0<=b<self.size and self.valid[a] and self.valid[b]):return 'INVALID_NETWORK'
        if a==b:return 'SAME_PIXEL'
        pa,sa=self.trace(a);pb,sb=self.trace(b)
        if b in pa:return 'UPSTREAM'
        if a in pb:return 'DOWNSTREAM'
        if np.intersect1d(pa,pb,assume_unique=True).size:return 'LATERAL_BRANCH'
        if sa=='TERMINAL' and sb=='TERMINAL':return 'DISCONNECTED_CONFIRMED'
        return 'UNKNOWN_DOMAIN_TRUNCATED' if 'TRUNCATED' in sa+sb or 'LIMIT' in sa+sb else 'UNKNOWN_MISSING_DOWNSTREAM'

    def upstream(self,pixel):
        """Return contributing pixel IDs and conservative domain-completeness status."""
        pixel=int(pixel)
        if not 0<=pixel<self.size or not self.valid[pixel]:raise ValueError('Invalid outlet pixel')
        if self._reverse is None:
            src=np.flatnonzero(self.downstream>=0);dst=self.downstream[src]
            order=np.argsort(dst,kind='stable')
            offsets=np.r_[0,np.cumsum(np.bincount(dst,minlength=self.size),dtype=np.int64)]
            self._reverse=(src[order],offsets)
        src,offsets=self._reverse
        found=[pixel];head=0
        while head<len(found):
            p=found[head];head+=1
            found.extend(src[offsets[p]:offsets[p+1]].tolist())
            if len(found)>self.config.max_trace_cells:
                return np.asarray(found,dtype=np.int64),'UPSTREAM_LIMIT'
        arr=np.sort(np.asarray(found,dtype=np.int64));ny,nx=self.shape;y,x=np.divmod(arr,nx)
        boundary=np.any((y==0)|(y==ny-1)|(x==0)|(x==nx-1))
        # Unknown/masked neighbours could hide missing upstream contributions.
        hole=False
        for dy,dx in LDD_OFFSETS.values():
            yy,xx=y+dy,x+dx;inside=(yy>=0)&(yy<ny)&(xx>=0)&(xx<nx)
            if np.any(~self.valid[yy[inside]*nx+xx[inside]]):hole=True;break
        status='ASSERTED_COMPLETE' if self.config.domain_complete else (
            'UNKNOWN_BOUNDARY_CONTACT' if boundary or hole else 'INTERIOR_COMPLETE')
        return arr,status

    def catchment_overlap(self,pixel,reference_mask):
        """Area-weighted raster-centre IoU, only when model support is complete.

        The supplied independent reference mask must cover the same LDD domain;
        its own completeness is the caller's responsibility. It is not inferred
        from SUB_AREA or from the polygon containing a gauge.
        """
        if self.cell_area is None:raise ValueError('CELLAREA with explicit units is needed for basin overlap')
        ref=exact_subset(reference_mask,self.ldd)
        refv=np.asarray(ref,dtype=float)
        if not np.isfinite(refv).all() or not np.isin(refv,[0,1]).all():
            raise ValueError('Reference catchment mask must be complete boolean/0-1 data')
        ids,status=self.upstream(pixel)
        if status not in {'ASSERTED_COMPLETE','INTERIOR_COMPLETE'}:
            return dict(catchment_iou=np.nan,catchment_status=status)
        r=refv.ravel()==1; model=np.zeros(self.size,bool);model[ids]=True
        union=r|model
        if not np.isfinite(self.cell_area[union]).all():
            return dict(catchment_iou=np.nan,catchment_status='MISSING_CELLAREA')
        den=self.cell_area[union].sum()
        return dict(catchment_iou=float(self.cell_area[r&model].sum()/den) if den>0 else np.nan,
                    catchment_status=status,model_catchment_km2=float(self.cell_area[ids].sum()))

    def validate_discharge(self,q):
        assert_versions({'Q':q,'LDD':self.ldd,'UPAREA':self.uparea},self.version,
                        self.config.require_version_metadata)
        exact_subset(self.ldd,q,self.config.coordinate_tolerance)
        return grid_fingerprint(q)


def hybas_reference_masks(network, basins, hybas_ids, *, max_area_discrepancy=.05,
                          max_total_mask_cells=20_000_000):
    """Raster-centre masks of COMPLETE upstream HydroBASINS unions.

    `SUB_AREA` sum is checked against target `UP_AREA`; failure refuses the
    reference instead of substituting the local polygon. This is a necessary
    closure check, not proof of perfect basin delineation. A gauge is not
    assumed to be at the HYBAS outlet. Returned keys are 'hybas:ID', used only
    for outlet comparisons by the matcher, not for gauge masks.
    """
    from shapely.ops import unary_union
    from shapely import intersects_xy
    if basins.crs is None:raise ValueError('Reference basin CRS is required')
    b=basins.to_crs(4326).copy()
    need={'HYBAS_ID','NEXT_DOWN','UP_AREA','SUB_AREA','geometry'}
    if need-set(b):raise ValueError(f'Reference masks need {sorted(need)}')
    if b.HYBAS_ID.duplicated().any():raise ValueError('Duplicate basin polygon IDs')
    ids=list(dict.fromkeys(int(i) for i in hybas_ids))
    if len(ids)*network.size>max_total_mask_cells:raise MemoryError('Reference mask budget exceeded; build only ambiguous target masks')
    if not 0<=max_area_discrepancy<1:raise ValueError('Invalid area discrepancy threshold')
    byid=b.set_index('HYBAS_ID');reverse={}
    for _,r in b.iterrows():
        if pd.notna(r.NEXT_DOWN):reverse.setdefault(int(r.NEXT_DOWN),[]).append(int(r.HYBAS_ID))
    result={}
    for hid in ids:
        if hid not in byid.index:raise KeyError(hid)
        stack=[hid];seen=set()
        while stack:
            v=stack.pop()
            if v in seen:raise ValueError('HydroBASINS upstream topology contains a cycle')
            seen.add(v);stack.extend(reverse.get(v,[]))
        upstream=byid.loc[sorted(seen)]
        expected=float(byid.loc[hid,'UP_AREA']);subs=pd.to_numeric(upstream.SUB_AREA,errors='coerce')
        if expected<=0 or not np.isfinite(expected) or not np.isfinite(subs).all() or (subs<0).any():
            raise ValueError('Invalid HydroBASINS upstream/sub-basin areas')
        discrepancy=abs(subs.sum()-expected)/expected
        if discrepancy>max_area_discrepancy:
            raise ValueError(f'HYBAS {hid}: upstream polygon support is incomplete/inconsistent (area error {discrepancy:.1%})')
        polygon=unary_union(upstream.geometry.tolist())
        xmin,ymin,xmax,ymax=polygon.bounds
        # The entire reference must fit inside the outer cell EDGES, not merely
        # be clipped to the target domain and falsely counted as complete.
        x=network.ldd.X.values;y=network.ldd.Y.values
        if len(x)<2 or len(y)<2:raise ValueError('Reference-mask raster needs at least 2x2 cells')
        dx=(x[1]-x[0])/2;dy=(y[1]-y[0])/2
        if xmin<x[0]-dx or xmax>x[-1]+dx or ymin<y[0]-dy or ymax>y[-1]+dy:
            raise ValueError(f'HYBAS {hid}: reference basin extends outside the LDD domain')
        mask=np.zeros(network.shape,dtype=bool)
        for start in range(0,len(y),256):
            xx,yy=np.meshgrid(x,y[start:start+256]);mask[start:start+len(yy)]=intersects_xy(polygon,xx,yy)
        result['hybas:'+str(hid)]=xr.DataArray(mask,dims=['Y','X'],coords={'Y':y,'X':x},
            attrs=dict(HYBAS_ID=hid,support='upstream_polygon_union',
                       rasterization='cell-centre intersection; not exact fractional coverage',
                       upstream_polygon_count=len(seen),area_closure_error=float(discrepancy)))
    return result
