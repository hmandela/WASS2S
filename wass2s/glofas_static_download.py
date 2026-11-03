"""Static-map infrastructure behind WAS_Download_GloFAS_historical.

Normal WASS2S usage goes through the existing WAS_Download class. The standalone
function remains for v5.2 compatibility, but no second download workflow is
required. Sources are versioned and cached; crops retain native grid cells.
No I/O or network request is performed at import.
"""
from __future__ import annotations
from pathlib import Path
import hashlib
import json
import os
import shutil
import tempfile
from urllib.request import Request, urlopen
from urllib.parse import urlparse
import numpy as np
import xarray as xr

if __package__:
    from .glofas_static_network import normalize_static, declared_version
else:
    from glofas_static_network import normalize_static, declared_version

_JRC='https://jeodpp.jrc.ec.europa.eu/ftp/jrc-opendata/CEMS-GLOFAS/LISFLOOD_static_and_parameter_maps_for_GloFAS/'
_V5=_JRC+'v2.1.1_OS-LISFLOOD-v5.x/'
_ECMWF='https://confluence.ecmwf.int/download/attachments/242067380/'
# Pinned catalogue references, not a request to silently use the newest maps.
STATIC_MAP_URLS={
 'version_5_0':{
  'LDD':_V5+'Catchments_morphology_and_river_network/ldd_repaired.nc',
  'UPAREA':_V5+'Catchments_morphology_and_river_network/upArea_repaired_correctedmetadata_3000.nc',
  'CHAN':_V5+'Catchments_morphology_and_river_network/chan_Global_03min.nc',
  'CELLAREA':_V5+'Main/pixarea_Global_03min.nc'},
 'version_4_0':{
  'LDD':_ECMWF+'ldd_glofas_v4_0.nc?api=v2',
  'UPAREA':_ECMWF+'uparea_glofas_v4_0.nc?api=v2'},
}


def static_map_catalogue(version='version_5_0'):
    if version not in STATIC_MAP_URLS:raise ValueError('Only explicitly registered model versions are supported')
    return dict(STATIC_MAP_URLS[version])


def _sha(path):
    h=hashlib.sha256()
    with Path(path).open('rb') as f:
        for data in iter(lambda:f.read(1024**2),b''):h.update(data)
    return h.hexdigest()


def download_glofas_static_maps(directory,*,version='version_5_0',variables=('LDD','UPAREA','CHAN','CELLAREA'),
        area=None,force=False,url_overrides=None,timeout=120,max_download_bytes=2*1024**3):
    """Download explicit maps and optionally crop native centres to [N,W,S,E].

    The raw global file is downloaded once; this endpoint is not a subsetting
    service. Returned paths are static regional files when area is supplied.
    Sources and SHA256 are written to JSON; unknown/moved URLs fail visibly.
    CHAN/CELLAREA for v4 require explicit overrides, not cross-version maps.
    Network downloads were not executed in the offline v5.2 test environment.
    """
    root=Path(directory);root.mkdir(parents=True,exist_ok=True)
    urls=static_map_catalogue(version);urls.update(url_overrides or {})
    variables=tuple(variables)
    if set(variables)-set(urls):raise ValueError(f'No registered URLs for {set(variables)-set(urls)}; provide version-compatible explicit URLs')
    if area is not None:
        area=_static_area(area)
    outputs={}
    for name in variables:
        if name not in {'LDD','UPAREA','CHAN','CELLAREA'}:raise ValueError('Unknown static map')
        url=urls[name]
        if urlparse(url).scheme!='https':raise ValueError('Static downloads require HTTPS')
        raw=root/f'{name}_{version}_source.nc';manifest=raw.with_suffix('.json')
        reusable=False
        if raw.exists() and manifest.exists() and not force:
            info=json.loads(manifest.read_text())
            reusable=info.get('url')==url and info.get('sha256')==_sha(raw)
        if not reusable:
            fd,tmp=tempfile.mkstemp(prefix=name+'_',suffix='.part',dir=root);os.close(fd)
            try:
                with urlopen(Request(url,headers={'User-Agent':'WAS-Hydro/5.2'}),timeout=timeout) as r,open(tmp,'wb') as f:
                    length=r.headers.get('Content-Length')
                    if length is not None and int(length)>max_download_bytes:raise MemoryError('Source file exceeds download size budget')
                    count=0
                    while True:
                        block=r.read(1024**2)
                        if not block:break
                        count+=len(block)
                        if count>max_download_bytes:raise MemoryError('Download size budget exceeded')
                        f.write(block)
                with xr.open_dataset(tmp) as ds:
                    a=normalize_static(ds)
                    actual=declared_version(a)
                    expected=version.replace('version_','').replace('_','.')
                    if actual is not None and actual!=expected:raise ValueError('Downloaded map declares an incompatible GloFAS version')
                digest=_sha(tmp);os.replace(tmp,raw)
                manifest.write_text(json.dumps(dict(url=url,version=version,sha256=digest,bytes=count),indent=2))
            finally:
                Path(tmp).unlink(missing_ok=True)
        # Saving selected maps makes their version/source declaration explicit;
        # do not attach model version to an unknown manual file by inference.
        suffix=hashlib.sha256(json.dumps(area).encode()).hexdigest()[:10] if area else 'full'
        dest=root/f'{name}_{version}_{suffix}.nc'
        derived_manifest=dest.with_suffix('.json')
        source_hash=_sha(raw)
        reusable_derived=False
        if dest.exists() and derived_manifest.exists() and not force:
            dmeta=json.loads(derived_manifest.read_text())
            reusable_derived=dmeta.get('source_sha256')==source_hash and dmeta.get('sha256')==_sha(dest)
        if not reusable_derived:
            with xr.open_dataset(raw) as ds:
                a=normalize_static(ds)
                if area is not None:a=_crop_static(a,area)
                if a.size==0:raise ValueError('Static map crop is empty')
                a=a.rename(name).load()
                a.attrs.update(glofas_version=version,source_url=url,source_sha256=_sha(raw),
                               domain_note='Native crop; upstream-domain completeness has NOT been asserted')
                if name=='LDD':
                    original_convention=str(a.attrs.get('flow_direction_convention','')).lower()
                    if 'esri' in original_convention or (original_convention and 'pcraster' not in original_convention):
                        raise ValueError('LDD must use the PCRaster convention, not ESRI D8')
                    a.attrs['flow_direction_convention']='PCRaster LDD' 
                fd,tmp=tempfile.mkstemp(prefix=name+'_',suffix='.nc',dir=root);os.close(fd)
                try:a.to_netcdf(tmp);os.replace(tmp,dest)
                finally:Path(tmp).unlink(missing_ok=True)
            derived_manifest.write_text(json.dumps(dict(source_sha256=source_hash,sha256=_sha(dest),area=area,version=version),indent=2))
        outputs[name]=dest
    return outputs

# ---------------------------------------------------------------------------
# WASS2S integration: the public entry point is
# WAS_Download.WAS_Download_GloFAS_historical().  These functions are private
# infrastructure; users do not need to call a separate static downloader.
# ---------------------------------------------------------------------------

def _static_area(area):
    """Validate a native-grid crop [N, W, S, E], including dateline requests."""
    values = np.asarray(area, dtype=float)
    if (values.shape != (4,) or not np.isfinite(values).all()
            or not -90 <= values[2] < values[0] <= 90
            or not (-180 <= values[1] <= 180 and -180 <= values[3] <= 180)
            or values[1] == values[3]):
        raise ValueError('area must be finite [North, West, South, East] in degrees')
    return tuple(float(value) for value in values)


def _crop_static(array, area):
    """Select native centres only. Never interpolate or repair a drainage map."""
    north, west, south, east = _static_area(area)
    array = normalize_static(array)
    # Normalize the longitude convention by relabelling, without moving cells.
    if np.any(np.asarray(array.X) > 180):
        array = array.assign_coords(X=((array.X + 180) % 360) - 180).sortby('X')
        if not array.get_index('X').is_unique:
            raise ValueError('Duplicate longitude after normalization; resolve the cyclic meridian explicitly')
    array = array.sel(Y=slice(south, north))
    if west <= east:
        array = array.sel(X=slice(west, east))
    else:
        array = array.isel(X=np.flatnonzero((array.X.values >= west) | (array.X.values <= east)))
    if not array.sizes.get('Y') or not array.sizes.get('X'):
        raise ValueError('Requested static domain contains no native grid centre')
    return array


def _select_static_variable(dataset, name):
    """Use documented aliases; reject ambiguous files instead of picking a large layer."""
    aliases = {
        'UPAREA': ('UPAREA', 'UPAREA_KM2', 'uparea', 'upArea', 'ec_upArea', 'upstream_area'),
        'LDD': ('LDD', 'ldd', 'ldd_repaired', 'ec_ldd_repaired'),
        'CHAN': ('CHAN', 'chan', 'channel_mask', 'channel'),
        'CELLAREA': ('CELLAREA', 'pixarea', 'cell_area', 'cellarea'),
    }
    key = next((key for key in aliases[name] if key in dataset.data_vars), None)
    if key is None:
        spatial = [key for key, item in dataset.data_vars.items() if item.ndim >= 2]
        if len(spatial) != 1:
            raise ValueError(f'Cannot uniquely identify {name} in {list(dataset.data_vars)}')
        key = spatial[0]
    return normalize_static(dataset, key)


def _canonical_static(array, name, version):
    """Validate physical values and explicit units; preserve missing cells."""
    if __package__:
        from .highflow_utils import area_to_km2
    else:
        from highflow_utils import area_to_km2
    actual = declared_version(array)
    expected = declared_version(xr.DataArray(0, attrs={'system_version': version}))
    if actual is not None and actual != expected:
        raise ValueError(f'{name} declares GloFAS {actual}, requested {expected}')
    array = array.load().astype('float64')
    if name in ('UPAREA', 'CELLAREA'):
        # Conversion is based on the units string, never on the numerical scale.
        array = (area_to_km2(array) * 1_000_000.0).assign_attrs(
            dict(array.attrs, units='m2', source_units=array.attrs.get('units', '')))
        array = array.astype('float32')
    else:
        values = np.asarray(array)
        if name == 'LDD':
            # 0 and 255 are documented nodata representations, not directions.
            array = array.where(~array.isin([0, 255]))
            values = np.asarray(array)
            allowed = np.arange(1, 10)
        else:
            allowed = np.array([0, 1])
        if np.any(np.isfinite(values) & ~np.isin(values, allowed)):
            convention = 'PCRaster LDD 1..9 (5=terminal)' if name == 'LDD' else 'CHAN 0/1'
            raise ValueError(f'Invalid {name} values; expected {convention} or missing')
        array.attrs['units'] = '1'
        if name == 'LDD':
            convention = str(array.attrs.get('flow_direction_convention', '')).lower()
            if 'esri' in convention or (convention and 'pcraster' not in convention):
                raise ValueError('LDD must use the PCRaster convention, not ESRI D8')
            array.attrs['flow_direction_convention'] = 'PCRaster LDD'
    array = array.rename(name)
    array.attrs.update(
        system_version=version, WASS2S_dimensions='Y,X',
        source_version_status=array.attrs.get('source_version_status',
            'DECLARED_COMPATIBLE' if actual else 'USER_ASSERTED_VERSION'),
    )
    return array


def _write_static_atomic(dataset, destination):
    """Close a new NetCDF before atomically replacing its public destination."""
    import importlib.util
    destination = Path(destination)
    destination.parent.mkdir(parents=True, exist_ok=True)
    engine = ('netcdf4' if importlib.util.find_spec('netCDF4') else
              'h5netcdf' if importlib.util.find_spec('h5netcdf') else 'scipy')
    encoding = {}
    for name in dataset.data_vars:
        dtype, fill = ('int16', -9999) if name == 'LDD' else (
            ('int8', -1) if name == 'CHAN' else ('float32', -9999.0))
        encoding[name] = {'dtype': dtype, '_FillValue': fill}
        if engine != 'scipy':
            encoding[name].update(zlib=True, complevel=4)
    # Inherited packing/coordinates can conflict with an explicit encoding.
    clean = dataset.drop_encoding()
    for name in clean.data_vars:
        clean[name].attrs = {k: v for k, v in clean[name].attrs.items()
                             if k not in ('_FillValue', 'missing_value', 'scale_factor', 'add_offset')}
    fd, temp = tempfile.mkstemp(prefix=destination.name + '.', suffix='.part', dir=destination.parent)
    os.close(fd)
    try:
        clean.to_netcdf(temp, engine=engine, encoding=encoding)
        os.replace(temp, destination)
    finally:
        Path(temp).unlink(missing_ok=True)


def _cached_static(path, *, names, area, version):
    """Read only a versioned, same-domain public cache; legacy attributes supported."""
    import ast
    if not Path(path).is_file():
        return None
    with xr.open_dataset(path) as ds:
        actual = declared_version(ds)
        expected = declared_version(xr.DataArray(0, attrs={'system_version': version}))
        if actual != expected:
            raise ValueError(f'Existing static cache {path} has unknown/different version; use force_static_download=True')
        old_area = ds.attrs.get('spatial_subset_NWSE')
        try:
            bounds = tuple(map(float, ast.literal_eval(str(old_area))))
        except (ValueError, SyntaxError, TypeError):
            bounds = None
        if bounds != tuple(area):
            raise ValueError(f'Existing static cache {path} has another/unknown area; use another directory or force_static_download=True')
        if not set(names).issubset(ds.data_vars):
            return None
        return ds[list(names)].load()


def prepare_glofas_auxiliaries(directory, *, variables, version, area,
                               force=False, url_overrides=None, local_files=None,
                               max_retries=3, retry_delay=10, timeout=180):
    """Internal implementation of the unified WASS2S GloFAS download option.

    Returns (Dataset, paths). `Static_UPAREA_<version>.nc` is retained for
    backward compatibility. `Static_GLOFAS_<version>.nc` groups all requested
    auxiliaries. All fields have only Y,X dimensions. Missing values are never
    replaced by zero. User-supplied local files assert compatibility with the
    requested model version when the source has no declaration; that status
    is recorded, not called independently verified.
    """
    import time
    import warnings
    root = Path(directory)
    root.mkdir(parents=True, exist_ok=True)
    area = _static_area(area)
    names = tuple(dict.fromkeys(str(v).removeprefix('GLOFAS.').upper() for v in variables))
    allowed = {'UPAREA', 'LDD', 'CHAN', 'CELLAREA'}
    if not names or set(names) - allowed:
        raise ValueError(f'Unsupported static maps: {set(names) - allowed}')
    if int(max_retries) != max_retries or max_retries < 1 or retry_delay < 0:
        raise ValueError('Require max_retries >= 1 and retry_delay >= 0')
    urls = {str(k).removeprefix('GLOFAS.').upper(): v for k, v in (url_overrides or {}).items()}
    sources = {str(k).removeprefix('GLOFAS.').upper(): Path(v) for k, v in (local_files or {}).items()}
    if (set(urls) | set(sources)) - allowed:
        raise ValueError('Unknown auxiliary name in static_map_urls/static_map_files')
    if set(urls) & set(sources):
        raise ValueError('Declare a local file OR a URL for an auxiliary, not both')
    public = root / f'Static_GLOFAS_{version}.nc'
    legacy = root / f'Static_UPAREA_{version}.nc'
    # A new explicit local/remote source invalidates a public cache unless its
    # content provenance is checked below. Never ignore an explicit source.
    can_use_public = not force and not urls and not sources
    cached = _cached_static(public, names=names, area=area, version=version) if can_use_public else None
    if cached is not None:
        for name in names:
            cached[name] = _canonical_static(cached[name], name, version)
        if 'UPAREA' in names:
            km = (cached.UPAREA / 1e6).rename('UPAREA_KM2')
            km.attrs = dict(cached.UPAREA.attrs, units='km2', derived_from='UPAREA / 1e6')
            cached['UPAREA_KM2'] = km
            if not legacy.is_file():
                _write_static_atomic(cached[['UPAREA', 'UPAREA_KM2']], legacy)
        return cached, {'static': public, 'uparea': legacy if 'UPAREA' in names else None}

    legacy_ds = None
    if 'UPAREA' in names and not force and 'UPAREA' not in urls and 'UPAREA' not in sources:
        legacy_ds = _cached_static(legacy, names=['UPAREA', 'UPAREA_KM2'], area=area, version=version)
    arrays = {}
    raw_paths = {}
    for name in names:
        if name == 'UPAREA' and legacy_ds is not None:
            arr = legacy_ds.UPAREA
            origin = 'existing_versioned_uparea_cache'
        elif name in sources:
            path = sources[name]
            if not path.is_file():
                raise FileNotFoundError(path)
            with xr.open_dataset(path) as ds:
                arr = _crop_static(_select_static_variable(ds, name), area).load()
            arr.attrs.update(source_file=str(path), source_sha256=_sha(path))
            origin = 'explicit_local_file'
        else:
            # The helper remains compatible for existing advanced callers. Its
            # cache is now an internal detail of WAS_Download_GloFAS_historical.
            for attempt in range(int(max_retries)):
                try:
                    downloaded = download_glofas_static_maps(
                        root / '.glofas_static', version=version, variables=(name,),
                        area=list(area), force=force,
                        url_overrides={name: urls[name]} if name in urls else None,
                        timeout=timeout,
                    )
                    break
                except (OSError, TimeoutError) as exc:
                    status = getattr(exc, 'code', None)
                    if status in (400, 401, 403, 404, 405, 410, 422) or attempt + 1 == max_retries:
                        raise
                    time.sleep(retry_delay)
            path = downloaded[name]
            with xr.open_dataset(path) as ds:
                arr = _select_static_variable(ds, name).load()
            raw_paths[name] = str(path)
            origin = 'explicit_url' if name in urls else 'versioned_catalogue'
        arr = _canonical_static(arr, name, version)
        arr.attrs['source_selection'] = origin
        arrays[name] = arr
    # Merge by exact native labels. An outer join would corrupt the topology.
    reference = arrays.get('LDD', next(iter(arrays.values())))
    for name, arr in arrays.items():
        if any(not np.array_equal(arr[d].values, reference[d].values) for d in ('Y', 'X')):
            raise ValueError(f'{name} and auxiliary native grids differ; no interpolation is allowed')
    result = xr.Dataset(arrays)
    if 'UPAREA' in result:
        result['UPAREA_KM2'] = (result.UPAREA / 1e6).astype('float32')
        result.UPAREA_KM2.attrs = dict(result.UPAREA.attrs, units='km2', derived_from='UPAREA / 1e6')
    result.attrs.update(
        title='GloFAS static network fields for WASS2S',
        system_version=version, spatial_subset_NWSE=str(list(area)),
        crs='EPSG:4326', WASS2S_dimensions='Y,X',
        domain_note='Native crop; upstream-domain completeness has NOT been asserted',
        wass2s_hydro_integration='5.2.0',
    )
    # If the same public cache has extra compatible maps, preserve them when a
    # later request asks only for UPAREA. Never extend coordinates to retain them.
    if public.is_file() and not force and not urls and not sources:
        with xr.open_dataset(public) as old:
            old_known = [n for n in old.data_vars if n in allowed | {'UPAREA_KM2'}]
        old = _cached_static(public, names=old_known, area=area, version=version)
        if old is not None:
            for name in old.data_vars:
                if name not in result:
                    if any(not np.array_equal(old[d], result[d]) for d in ('Y', 'X')):
                        raise ValueError('Existing auxiliary grid differs from the requested native grid')
                    result[name] = old[name]
    _write_static_atomic(result, public)
    if 'UPAREA' in result:
        _write_static_atomic(result[['UPAREA', 'UPAREA_KM2']], legacy)
    report = {
        'system_version': version, 'area': list(area), 'variables': list(result.data_vars),
        'sha256': _sha(public), 'download_cache': raw_paths,
        'sources': {name: dict(source_file=result[name].attrs.get('source_file'),
                               source_url=result[name].attrs.get('source_url'),
                               source_sha256=result[name].attrs.get('source_sha256'),
                               source_version_status=result[name].attrs.get('source_version_status'))
                    for name in names},
        'upstream_domain_complete': False,
    }
    public.with_suffix('.json').write_text(json.dumps(report, indent=2), encoding='utf-8')
    return result, {'static': public, 'uparea': legacy if 'UPAREA' in result else None}
