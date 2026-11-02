#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
GloFAS high-flow bias correction and HydroBASINS-aware outlet matching.

This is a PURE IMPORTABLE MODULE: no argparse, no CLI, no main().

The module is designed to work with the high-water/high-flow regime detection
implemented in WAS_Hydro / highwater_season_predictands_modeaware.py.

Main scientific workflow
------------------------
1. Associate each observed gauging outlet with a HydroBASINS level-9 HYBAS_ID.
2. Use HydroBASINS UP_AREA (km2) and the GloFAS upstream-area raster (m2) to
   identify the *hydrologically plausible* GloFAS river pixel near the outlet.
3. Extract daily GloFAS discharge at the selected pixel.
4. Fit a bias correction ONLY on the observed climatological HW/MODE window.
5. Optionally propagate multiplicative correction factors through the
   HydroBASINS drainage network to ungauged HYBAS_ID outlets.
6. Apply the correction and keep a full audit table of every spatial match.

Why upstream area matters
-------------------------
A geographically nearest GloFAS pixel is not necessarily the correct river
pixel. Two nearby river cells can drain very different upstream areas. The
matching score used here therefore combines:

    area mismatch = |log(A_glofas / A_hybas)|

and a normalized geodesic distance to the HydroBASINS outlet. By default the
area term receives much more weight than distance.

HydroBASINS UP_AREA is documented in km2. GloFAS v4 upstream-area auxiliary
files store upstream area in m2. This module converts both to km2 internally.

Dependencies
------------
numpy, pandas, xarray, geopandas, scipy, networkx

Optional
--------
matplotlib for diagnostic plotting.
"""

from __future__ import annotations

from dataclasses import dataclass, asdict, replace
from typing import Any, Iterable, Optional, Literal
import math
import warnings

import numpy as np
import pandas as pd
import xarray as xr
import geopandas as gpd
from scipy.stats import rankdata
import networkx as nx
try:
    from .highflow_utils import (area_to_km2, clean_daily_series, window_dates,
        coverage_statistics, hydrological_year_bounds, match_windows, doy_to_md,
        wrap_doy, canonical_doy)
except ImportError:
    from highflow_utils import (area_to_km2, clean_daily_series, window_dates,
        coverage_statistics, hydrological_year_bounds, match_windows, doy_to_md,
        wrap_doy, canonical_doy)


# =============================================================================
# Configuration
# =============================================================================


@dataclass
class GloFASPixelMatchConfig:
    """Configuration for HYBAS outlet -> GloFAS river-pixel matching."""

    # Candidate search radius around the target outlet.
    search_radius_km: float = 40.0
    assumed_uparea_units: Optional[str] = None
    prefer_station_area: bool = True
    # A measured station catchment area takes precedence over a different L9
    # outlet area. Supply it as station_uparea_km2, with its provenance.
    accepted_match_statuses: tuple[str, ...] = ("OK", "LOW_CANDIDATE_SUPPORT")

    # Relative importance of hydrological area consistency vs geographic
    # proximity. Values are normalized internally, so they need not sum to 1.
    area_weight: float = 0.85
    distance_weight: float = 0.15

    # If the best candidate differs too much in drainage area, retain it in the
    # table but flag it as suspicious. Set None to disable the threshold.
    max_area_error_pct: Optional[float] = 35.0

    # Optional hard candidate filter. At very small headwater areas GloFAS may
    # not represent a perennial river at its native 0.05-degree resolution.
    min_glofas_uparea_km2: float = 0.0

    # Require a minimum number of finite candidate river pixels in the search
    # box. If fewer are available, the match is flagged LOW_CANDIDATE_SUPPORT.
    min_candidates: int = 1

    # Station-to-HydroBASINS association. In "auto" mode, an explicit
    # HydroBASINS outlet point layer is preferred; otherwise polygon containment
    # is used. This is safer for gauges located exactly on level-9 boundaries.
    station_hybas_assignment: Literal["auto", "nearest_outlet", "polygon"] = "auto"
    station_to_hybas_max_distance_km: float = 30.0

    # If an explicit HydroBASINS outlet point layer is supplied, it is preferred
    # over polygon centroids / station coordinates for GloFAS pixel matching.
    prefer_hybas_outlet_points: bool = True

    # Candidate-table diagnostics: number of top alternatives retained for each
    # target. 0 disables the alternatives table.
    retain_top_candidates: int = 5


@dataclass
class GloFASHighFlowBiasConfig:
    """Configuration for the first-stage gauged high-flow bias correction.

    This implementation intentionally starts with transparent, robust methods.
    More sophisticated distributional methods can be added without changing
    the pixel-matching architecture.
    """

    method: Literal["mean_ratio", "log_ratio", "eqm"] = "log_ratio"

    # Which observed window table is used.
    correction_scope: Literal["season", "mode"] = "season"
    selector: int | Literal["main"] = "main"

    # Minimum number of valid daily OBS/GloFAS pairs used for fitting.
    min_pairs: int = 60
    min_q: float = 0.01

    # How the correction is applied outside the selected high-flow window.
    # raw: retain uncorrected GloFAS; nan: mask it; corrected_all: apply the
    # fitted correction to every day despite fitting only on high-flow days.
    outside_highflow: Literal["raw", "nan", "corrected_all"] = "raw"

    # Empirical quantile mapping controls.
    eqm_n_quantiles: int = 101

    # Network regionalization of multiplicative factors.
    propagate_to_ungauged: bool = True
    propagation_method: str = "hybrid"  # network, hybrid, idw_log, topkriging
    topkriging_config: Any = None
    network_k_donors: int = 3
    network_power: float = 1.0
    network_area_penalty: float = 0.60
    network_max_area_ratio: Optional[float] = 6.0
    allow_lateral_donors: bool = False


# =============================================================================
# Basic helpers
# =============================================================================


def _find_name(names: Iterable[str], candidates: Iterable[str]) -> Optional[str]:
    lookup = {str(n).lower(): str(n) for n in names}
    for c in candidates:
        if str(c).lower() in lookup:
            return lookup[str(c).lower()]
    return None


def _haversine_km(lon1, lat1, lon2, lat2):
    """Vectorized great-circle distance in km."""
    r = 6371.0088
    lon1 = np.deg2rad(np.asarray(lon1, dtype=float))
    lat1 = np.deg2rad(np.asarray(lat1, dtype=float))
    lon2 = np.deg2rad(np.asarray(lon2, dtype=float))
    lat2 = np.deg2rad(np.asarray(lat2, dtype=float))
    dlon = lon2 - lon1
    dlat = lat2 - lat1
    a = np.sin(dlat / 2.0) ** 2 + np.cos(lat1) * np.cos(lat2) * np.sin(dlon / 2.0) ** 2
    a = np.clip(a, 0.0, 1.0)
    return 2.0 * r * np.arcsin(np.sqrt(a))


def _normalize_lon_for_grid(lon: float, grid_lon: np.ndarray) -> float:
    """Convert a longitude to the convention used by a GloFAS grid."""
    glon = np.asarray(grid_lon, dtype=float)
    if np.nanmax(glon) > 180.0 and lon < 0:
        return lon % 360.0
    if np.nanmin(glon) < 0.0 and lon > 180.0:
        return ((lon + 180.0) % 360.0) - 180.0
    return float(lon)


def _circular_monthday_mask(index: pd.DatetimeIndex, start_md: str, end_md: str) -> np.ndarray:
    """Boolean calendar mask for MM-DD windows, including cross-year windows."""
    sm, sd = map(int, str(start_md).split("-"))
    em, ed = map(int, str(end_md).split("-"))
    code = index.month * 100 + index.day
    s = sm * 100 + sd
    e = em * 100 + ed
    if s <= e:
        return np.asarray((code >= s) & (code <= e), dtype=bool)
    return np.asarray((code >= s) | (code <= e), dtype=bool)


def _pick_dataarray(obj: xr.DataArray | xr.Dataset, variable: Optional[str] = None) -> xr.DataArray:
    if isinstance(obj, xr.DataArray):
        return obj
    if not isinstance(obj, xr.Dataset):
        raise TypeError("Expected xarray.DataArray or xarray.Dataset.")
    if variable is not None:
        if variable not in obj:
            raise KeyError(f"Variable {variable!r} not found. Available: {list(obj.data_vars)}")
        return obj[variable]
    if len(obj.data_vars) != 1:
        raise ValueError(
            "Dataset has multiple data variables; provide variable=. "
            f"Available: {list(obj.data_vars)}"
        )
    return obj[next(iter(obj.data_vars))]


def _normalize_xy_names(da: xr.DataArray) -> tuple[xr.DataArray, str, str]:
    """Return DataArray and horizontal coordinate names.

    Supported conventions include X/Y, longitude/latitude, lon/lat.
    """
    xname = _find_name(da.coords, ["X", "longitude", "lon", "x"])
    yname = _find_name(da.coords, ["Y", "latitude", "lat", "y"])
    if xname is None or yname is None:
        xname = _find_name(da.dims, ["X", "longitude", "lon", "x"])
        yname = _find_name(da.dims, ["Y", "latitude", "lat", "y"])
    if xname is None or yname is None:
        raise ValueError(
            "Could not identify GloFAS horizontal coordinates. Expected X/Y, "
            "longitude/latitude, or lon/lat."
        )
    return da, xname, yname


def _infer_time_name(da: xr.DataArray) -> str:
    t = _find_name(da.coords, ["T", "time", "valid_time", "date"])
    if t is None:
        t = _find_name(da.dims, ["T", "time", "valid_time", "date"])
    if t is None:
        raise ValueError("Could not identify time dimension in GloFAS discharge.")
    return t


def _uparea_to_km2(uparea: xr.DataArray, assumed_units=None) -> xr.DataArray:
    """Exact unit conversion; never infer units from magnitude or substrings."""
    return area_to_km2(uparea, assumed_units=assumed_units)


# =============================================================================
# Station -> HydroBASINS association
# =============================================================================


def validate_glofas_auxiliary(discharge,uparea,*,discharge_variable=None,uparea_variable=None):
    """Check declared versions and exact horizontal labels before extraction.

    Missing version metadata cannot be inferred from values. Its status stays
    UNKNOWN; users must then verify product provenance themselves.
    """
    q=_pick_dataarray(discharge,discharge_variable);a=_pick_dataarray(uparea,uparea_variable)
    try:from .highflow_utils import align_spatial_grid
    except ImportError:from highflow_utils import align_spatial_grid
    q,x,y=_normalize_xy_names(q)
    align_spatial_grid(a,q,y,x)
    def version(obj,da):
        for attrs in (da.attrs,obj.attrs):
            for key in ('glofas_version','model_version','system_version'):
                if key in attrs:return str(attrs[key]).strip().lower().replace('version_','').replace('v','').replace('_','.')
        return None
    vq,va=version(discharge,q),version(uparea,a)
    if vq is not None and va is not None and vq!=va:
        raise ValueError(f'Declared GloFAS versions differ: discharge={vq}, upstream area={va}')
    return dict(grid_alignment='EXACT_LABELS',discharge_version=vq,area_version=va,
                version_check='MATCH' if vq is not None and va is not None else 'UNKNOWN_METADATA')


def associate_stations_to_hybas(station_metadata, hybas9, *, hybas_outlets=None,
    station_col="station", lon_col="X", lat_col="Y", id_col="HYBAS_ID",
    uparea_col="UP_AREA", next_down_col="NEXT_DOWN", main_bas_col="MAIN_BAS", config=None):
    """Auditable association, never treating a gauge and an L9 outlet as identical.

    Explicit IDs are honored row-by-row. Otherwise nearest OUTLET (not centroid)
    or polygon coverage is used. Ambiguous boundaries are not resolved at random.
    A documented station_uparea_km2 selects the station itself as the matching
    target; otherwise the L9 outlet area/position is a flagged proxy.
    """
    cfg=config or GloFASPixelMatchConfig()
    if cfg.station_hybas_assignment not in {"auto","nearest_outlet","polygon"}:
        raise ValueError("Invalid station_hybas_assignment")
    for c in [station_col,lon_col,lat_col]:
        if c not in station_metadata: raise ValueError(f"Missing station column {c}")
    if station_metadata[station_col].astype(str).duplicated().any():
        raise ValueError("Station identifiers must be unique")
    if hybas9.crs is None: raise ValueError("HydroBASINS CRS must be supplied explicitly")
    bas=hybas9.to_crs(4326).copy()
    if bas[id_col].duplicated().any(): raise ValueError("Duplicate HYBAS_ID")
    bas=bas.set_index(id_col,drop=False)
    outs=None
    if hybas_outlets is not None:
        if hybas_outlets.crs is None: raise ValueError("Outlet CRS must be supplied")
        outs=hybas_outlets.to_crs(4326).copy()
        if not outs.geometry.geom_type.eq('Point').all(): raise ValueError("Outlets must be Points")
        outs=outs.set_index(id_col,drop=False)
    from shapely.geometry import Point
    rows=[]
    for _,r in station_metadata.iterrows():
        rec=r.to_dict(); lon=float(r[lon_col]);lat=float(r[lat_col])
        rec.update(station_lon=lon,station_lat=lat,station_hybas_boundary_ambiguous=False)
        hid=r.get(id_col,np.nan); method='explicit_id'
        validxy=np.isfinite(lon+lat) and -90<=lat<=90
        if pd.isna(hid):
            if validxy and outs is not None and cfg.station_hybas_assignment in {'auto','nearest_outlet'}:
                d=_haversine_km(lon,lat,outs.geometry.x.values,outs.geometry.y.values)
                if np.isfinite(d).any(): hid=outs.iloc[int(np.nanargmin(d))][id_col]
                method='nearest_outlet'
            elif validxy and cfg.station_hybas_assignment!='nearest_outlet':
                hit=bas[bas.geometry.covers(Point(lon,lat))]
                if len(hit)==1:hid=hit.iloc[0][id_col]
                elif len(hit)>1:rec['station_hybas_boundary_ambiguous']=True
                method='polygon'
            else: method='unresolved'
        rec[id_col]=int(hid) if pd.notna(hid) else pd.NA
        rec['station_hybas_assignment']=method
        b=bas.loc[int(hid)] if pd.notna(hid) and int(hid) in bas.index else None
        for c in [uparea_col,next_down_col,main_bas_col,'SUB_AREA','DIST_MAIN','DIST_SINK','PFAF_ID','COAST']:
            if c in bas.columns:rec[c]=b[c] if b is not None else np.nan
        rec['HYBAS_UP_AREA_KM2']=float(rec.get(uparea_col,np.nan))
        local_outs = (outs.loc[outs[id_col].eq(int(hid))]
                      if outs is not None and pd.notna(hid) else None)
        o = None
        multi = bool(pd.notna(rec.get('COAST')) and rec.get('COAST') == 1)
        if local_outs is not None and len(local_outs):
            multi = multi or len(local_outs) > 1
            if 'multiple_outlet_basin' in local_outs:
                multi = multi or bool(local_outs.multiple_outlet_basin.fillna(False).any())
            if validxy:
                ds_out = _haversine_km(lon, lat, local_outs.geometry.x.values, local_outs.geometry.y.values)
                best = int(np.nanargmin(ds_out))
                # Equidistant different points are ambiguous for the outlet
                # proxy. A measured gauge area still permits gauge-based matching.
                tied = np.flatnonzero(np.isclose(ds_out, ds_out[best], rtol=0, atol=1e-9))
                if len(tied) == 1:
                    o = local_outs.iloc[best]
            elif len(local_outs) == 1:
                o = local_outs.iloc[0]
        rec['multiple_outlet_basin'] = multi
        rec['associated_outlet_id'] = str(o.get('outlet_id','')) if o is not None else ''
        olon=float(o.geometry.x) if o is not None else np.nan
        olat=float(o.geometry.y) if o is not None else np.nan
        distance=float(_haversine_km(lon,lat,olon,olat))
        rec.update(hybas_outlet_lon=olon,hybas_outlet_lat=olat,station_to_hybas_outlet_km=distance,
                   station_to_hybas_outlet_ok=bool(np.isfinite(distance) and distance<=cfg.station_to_hybas_max_distance_km))
        area=pd.to_numeric(pd.Series([r.get('station_uparea_km2',np.nan)]),errors='coerce').iloc[0]
        use_station=cfg.prefer_station_area and np.isfinite(area) and area>0
        use_outlet=not use_station and cfg.prefer_hybas_outlet_points and o is not None
        rec.update(target_lon=olon if use_outlet else lon,target_lat=olat if use_outlet else lat,
                   target_coordinate_source='hybas_outlet' if use_outlet else 'station',
                   MATCH_UP_AREA_KM2=float(area) if use_station else (np.nan if multi else rec['HYBAS_UP_AREA_KM2']),
                   area_source='station_area' if use_station else ('missing_station_area_multi_outlet' if multi else 'HYBAS_outlet_proxy'),
                   station_area_source=r.get('station_area_source','unspecified'))
        status='OK'
        if b is None: status='UNMATCHED_HYBAS'
        elif not validxy: status='INVALID_COORDINATES'
        elif rec['station_hybas_boundary_ambiguous']:status='AMBIGUOUS_HYBAS'
        elif not use_station and np.isfinite(distance) and distance>cfg.station_to_hybas_max_distance_km:status='OUTLET_TOO_FAR'
        elif multi and not use_station:status='MULTIPLE_OUTLETS_STATION_AREA_REQUIRED'
        elif not use_station and o is None:status='AREA_PROXY_WITHOUT_OUTLET'
        rec['station_hybas_status']=status
        rows.append(rec)
    return pd.DataFrame(rows)


# =============================================================================
# HYBAS outlet -> hydrologically plausible GloFAS river pixel
# =============================================================================


def match_hybas_outlets_to_glofas(
    targets: pd.DataFrame,
    glofas_uparea: xr.DataArray | xr.Dataset,
    *,
    variable: Optional[str] = None,
    id_col: str = "HYBAS_ID",
    lon_col: str = "target_lon",
    lat_col: str = "target_lat",
    expected_area_col: str = "HYBAS_UP_AREA_KM2",
    config: Optional[GloFASPixelMatchConfig] = None,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Find the hydrologically most plausible GloFAS river pixel per target.

    Returns
    -------
    matches : DataFrame
        One row per target, with chosen GloFAS coordinates, upstream area,
        distance, area mismatch, score, and quality flags.
    alternatives : DataFrame
        Top-N candidate pixels per target for auditing. Empty when disabled.
    """
    cfg = config or GloFASPixelMatchConfig()
    up = _pick_dataarray(glofas_uparea, variable)
    up, xname, yname = _normalize_xy_names(up)
    up = _uparea_to_km2(up, cfg.assumed_uparea_units)
    if cfg.search_radius_km<=0: raise ValueError("search_radius_km must be positive")
    if expected_area_col=="HYBAS_UP_AREA_KM2" and "MATCH_UP_AREA_KM2" in targets:
        expected_area_col="MATCH_UP_AREA_KM2"

    # Upstream area should be static 2D. Drop singleton non-horizontal dims.
    extra = [d for d in up.dims if d not in {xname, yname}]
    for d in extra:
        if up.sizes[d] != 1:
            raise ValueError(f"GloFAS upstream-area field must be 2D; unexpected dim {d}={up.sizes[d]}")
        up = up.isel({d: 0}, drop=True)

    up = up.transpose(yname,xname)
    xvals = np.asarray(up[xname].values, dtype=float)
    yvals = np.asarray(up[yname].values, dtype=float)
    if xvals.ndim != 1 or yvals.ndim != 1:
        raise ValueError("Only rectilinear 1-D GloFAS longitude/latitude grids are supported.")

    required = {lon_col, lat_col, expected_area_col}
    missing = required - set(targets.columns)
    if missing:
        raise ValueError(f"targets missing required columns: {sorted(missing)}")

    aw = max(float(cfg.area_weight), 0.0)
    dw = max(float(cfg.distance_weight), 0.0)
    if aw + dw <= 0:
        raise ValueError("At least one of area_weight or distance_weight must be positive.")
    aw, dw = aw / (aw + dw), dw / (aw + dw)

    rows: list[dict[str, Any]] = []
    alt_rows: list[dict[str, Any]] = []

    for ridx, row in targets.reset_index(drop=True).iterrows():
        lon = pd.to_numeric(pd.Series([row[lon_col]]), errors="coerce").iloc[0]
        lat = pd.to_numeric(pd.Series([row[lat_col]]), errors="coerce").iloc[0]
        area = pd.to_numeric(pd.Series([row[expected_area_col]]), errors="coerce").iloc[0]
        base = dict(row)
        base["target_row"] = ridx

        # Carry any upstream association failure through to the pixel match, so a
        # gauge whose HydroBASINS outlet is implausibly far (OUTLET_TOO_FAR) or
        # unmatched is never silently corrected via an on-grid outlet pixel.
        upstream_status = str(row.get("station_hybas_status", "OK"))
        if upstream_status not in {"OK", "AREA_PROXY_WITHOUT_OUTLET"}:
            base.update({
                "match_status": upstream_status,
                "glofas_lon": np.nan,
                "glofas_lat": np.nan,
                "glofas_uparea_km2": np.nan,
                "distance_km": np.nan,
                "area_error_pct": np.nan,
                "area_log_mismatch": np.nan,
                "match_score": np.nan,
                "candidate_count": 0,
            })
            rows.append(base)
            continue

        if not (np.isfinite(lon) and np.isfinite(lat) and np.isfinite(area) and area > 0):
            base.update({
                "match_status": "INVALID_TARGET",
                "glofas_lon": np.nan,
                "glofas_lat": np.nan,
                "glofas_uparea_km2": np.nan,
                "distance_km": np.nan,
                "area_error_pct": np.nan,
                "area_log_mismatch": np.nan,
                "match_score": np.nan,
                "candidate_count": 0,
            })
            rows.append(base)
            continue

        lon_grid = _normalize_lon_for_grid(float(lon), xvals)
        # Geodesic search box. Longitude-degree width expands with latitude.
        dlat = cfg.search_radius_km / 110.574
        coslat = max(math.cos(math.radians(float(lat))), 0.05)
        dlon = cfg.search_radius_km / (111.320 * coslat)

        ix = np.where((xvals >= lon_grid - dlon) & (xvals <= lon_grid + dlon))[0]
        iy = np.where((yvals >= float(lat) - dlat) & (yvals <= float(lat) + dlat))[0]
        if ix.size == 0 or iy.size == 0:
            base.update({
                "match_status": "OUTSIDE_GLOFAS_GRID",
                "glofas_lon": np.nan,
                "glofas_lat": np.nan,
                "glofas_uparea_km2": np.nan,
                "distance_km": np.nan,
                "area_error_pct": np.nan,
                "area_log_mismatch": np.nan,
                "match_score": np.nan,
                "candidate_count": 0,
            })
            rows.append(base)
            continue

        sub = up.isel({yname: iy, xname: ix})
        vals = np.asarray(sub.values, dtype=float)
        XX, YY = np.meshgrid(xvals[ix], yvals[iy])
        dist = _haversine_km(lon_grid, float(lat), XX, YY)

        valid = (
            np.isfinite(vals)
            & (vals > max(cfg.min_glofas_uparea_km2, 0.0))
            & np.isfinite(dist)
            & (dist <= cfg.search_radius_km)
        )
        if not np.any(valid):
            base.update({
                "match_status": "NO_RIVER_PIXEL_IN_RADIUS",
                "glofas_lon": np.nan,
                "glofas_lat": np.nan,
                "glofas_uparea_km2": np.nan,
                "distance_km": np.nan,
                "area_error_pct": np.nan,
                "area_log_mismatch": np.nan,
                "match_score": np.nan,
                "candidate_count": 0,
            })
            rows.append(base)
            continue

        cand_area = vals[valid]
        cand_lon = XX[valid]
        cand_lat = YY[valid]
        cand_dist = dist[valid]
        area_log = np.abs(np.log(cand_area / float(area)))
        area_pct = np.abs(cand_area - float(area)) / float(area) * 100.0
        distance_norm = cand_dist / max(cfg.search_radius_km, 1e-9)
        score = aw * area_log + dw * distance_norm
        order = np.argsort(score)
        best = int(order[0])

        candidate_count = int(len(order))
        status = "OK"
        if candidate_count < cfg.min_candidates:
            status = "LOW_CANDIDATE_SUPPORT"
        if cfg.max_area_error_pct is not None and area_pct[best] > cfg.max_area_error_pct:
            status = "AREA_MISMATCH"

        second_score = float(score[order[1]]) if candidate_count > 1 else np.nan
        score_margin = second_score - float(score[best]) if candidate_count > 1 else np.nan

        base.update({
            "match_status": status,
            "glofas_lon": float(cand_lon[best]),
            "glofas_lat": float(cand_lat[best]),
            "glofas_uparea_km2": float(cand_area[best]),
            "distance_km": float(cand_dist[best]),
            "area_error_pct": float(area_pct[best]),
            "area_log_mismatch": float(area_log[best]),
            "match_score": float(score[best]),
            "second_best_score": second_score,
            "score_margin": score_margin,
            "candidate_count": candidate_count,
            "glofas_x_index": int(ix[np.where(valid)[1][best]]) if valid.ndim == 2 else np.nan,
            "glofas_y_index": int(iy[np.where(valid)[0][best]]) if valid.ndim == 2 else np.nan,
            "glofas_x_coord_name": xname,
            "glofas_y_coord_name": yname,
        })
        rows.append(base)

        keep = min(max(int(cfg.retain_top_candidates), 0), candidate_count)
        for rank, j in enumerate(order[:keep], start=1):
            alt = {
                "target_row": ridx,
                id_col: row.get(id_col, np.nan),
                "candidate_rank": rank,
                "candidate_lon": float(cand_lon[j]),
                "candidate_lat": float(cand_lat[j]),
                "candidate_uparea_km2": float(cand_area[j]),
                "distance_km": float(cand_dist[j]),
                "area_error_pct": float(area_pct[j]),
                "area_log_mismatch": float(area_log[j]),
                "score": float(score[j]),
            }
            if "station" in row.index:
                alt["station"] = row["station"]
            alt_rows.append(alt)

    matches = pd.DataFrame(rows)
    alternatives = pd.DataFrame(alt_rows)
    return matches, alternatives


# =============================================================================
# Extract GloFAS discharge at selected pixels
# =============================================================================


def extract_glofas_discharge_at_matches(
    glofas_discharge: xr.DataArray | xr.Dataset,
    matches: pd.DataFrame,
    *,
    variable: Optional[str] = None,
    station_col: str = "station",
    id_col: str = "HYBAS_ID",
    valid_status: tuple[str, ...] = ("OK", "LOW_CANDIDATE_SUPPORT"),
) -> tuple[pd.DataFrame, xr.DataArray]:
    """Extract one daily GloFAS series for each matched outlet.

    Returns a wide pandas DataFrame (date index, station/HYBAS columns) and an
    xarray DataArray with dimensions (T, outlet).
    """
    da = _pick_dataarray(glofas_discharge, variable)
    da, xname, yname = _normalize_xy_names(da)
    tname = _infer_time_name(da)

    series = []
    labels = []
    meta_rows = []
    for _, r in matches.iterrows():
        if str(r.get("match_status")) not in valid_status:
            continue
        if not (np.isfinite(r.get("glofas_lon", np.nan)) and np.isfinite(r.get("glofas_lat", np.nan))):
            continue
        lon = _normalize_lon_for_grid(float(r["glofas_lon"]), np.asarray(da[xname].values))
        lat = float(r["glofas_lat"])
        try:
            q = da.sel({xname: lon, yname: lat})
        except KeyError as exc:
            raise ValueError("Matched auxiliary pixel is absent from discharge grid; check version/grid alignment") from exc
        # Keep only time after selecting horizontal coordinates. Any ensemble or
        # lead dimensions must be reduced by the caller before this function.
        extras = [d for d in q.dims if d != tname]
        if extras:
            raise ValueError(
                "GloFAS discharge still has non-time dimensions after pixel extraction: "
                f"{extras}. Select/reduce ensemble or lead dimensions first."
            )
        label = str(r[station_col]) if station_col in r and pd.notna(r[station_col]) else str(r[id_col])
        q = q.rename({tname: "T"}) if tname != "T" else q
        q = q.assign_coords(outlet=label).expand_dims("outlet")
        series.append(q)
        labels.append(label)
        meta_rows.append(r)

    if not series:
        raise ValueError("No valid GloFAS pixel matches were available for extraction.")
    if len(set(labels))!=len(labels): raise ValueError("Duplicate outlet labels in matches")
    out = xr.concat(series, dim="outlet", coords="different", compat="equals", join="exact").transpose("T", "outlet")
    out.name = "GloFAS"
    out.attrs.update({"long_name": "GloFAS discharge extracted at hydrologically matched river pixels"})
    wide = out.to_pandas()
    wide.index = pd.DatetimeIndex(wide.index)
    wide=pd.DataFrame({c:clean_daily_series(wide[c]) for c in wide})
    out=xr.DataArray(wide.values,coords={"T":wide.index,"outlet":wide.columns.astype(str)},dims=("T","outlet"),name="GloFAS",attrs=out.attrs)
    return wide, out


# =============================================================================
# Observed-window selection and gauged correction fitting
# =============================================================================


def select_highflow_windows(
    diagnostics: dict[str, Any],
    *,
    scope: Literal["season", "mode"] = "season",
    selector: int | Literal["main"] = "main",
) -> pd.DataFrame:
    """Select one observed climatological HW/MODE window per station."""
    if scope == "season":
        tab = diagnostics.get("highwater_seasons", pd.DataFrame()).copy()
        idname, mainname = "season_id", "is_main_season"
    elif scope == "mode":
        tab = diagnostics.get("highflow_modes", pd.DataFrame()).copy()
        idname, mainname = "mode_id", "is_main_mode"
    else:
        raise ValueError("scope must be 'season' or 'mode'.")
    if tab.empty:
        raise ValueError(f"No {scope} definitions found in diagnostics.")
    for flag in ("weak_fallback", "weak_season_fallback", "fallback_window"):
        if flag in tab: tab=tab[~tab[flag].fillna(False).astype(bool)]
    if selector == "main":
        if mainname not in tab.columns:
            raise KeyError(f"Missing diagnostic column {mainname!r}.")
        out = tab[tab[mainname].fillna(False).astype(bool)].copy()
    else:
        sel = int(selector)
        out = tab[pd.to_numeric(tab[idname], errors="coerce") == sel].copy()
    if out.empty:
        raise ValueError(f"No {scope} window matched selector={selector!r}.")
    # There should normally be exactly one selected window per station.
    if out["station"].duplicated().any(): raise ValueError("Ambiguous selected windows per station")
    out = out.sort_values(["station", idname])
    return out


def _fit_eqm(obs: np.ndarray, sim: np.ndarray, n_quantiles: int = 101) -> dict[str, Any]:
    probs = np.linspace(0.0, 1.0, max(int(n_quantiles), 5))
    sq = np.quantile(sim, probs)
    oq = np.quantile(obs, probs)
    # np.interp requires increasing x. Collapse duplicated simulated quantiles.
    keep = np.r_[True, np.diff(sq) > 0]
    sq = sq[keep]
    oq = oq[keep]
    if sq.size < 2:
        raise ValueError("Insufficient unique simulated quantiles for EQM.")
    return {"sim_quantiles": sq, "obs_quantiles": oq, "probs": probs[keep]}


def _apply_eqm(values: np.ndarray, model: dict[str, Any]) -> np.ndarray:
    x = np.asarray(values, dtype=float)
    sq = np.asarray(model["sim_quantiles"], dtype=float)
    oq = np.asarray(model["obs_quantiles"], dtype=float)
    # Linear interpolation within support; constant correction ratio extension
    # outside support prevents exploding extrapolation in this first version.
    y = np.interp(x, sq, oq)
    low_ratio = oq[0] / sq[0] if sq[0] > 0 else 1.0
    high_ratio = oq[-1] / sq[-1] if sq[-1] > 0 else 1.0
    y = np.where(x < sq[0], x * low_ratio, y)
    y = np.where(x > sq[-1], x * high_ratio, y)
    return np.maximum(y, 0.0)


def fit_gauged_highflow_corrections(observations,glofas_at_stations,windows,*,config=None,
                                  calibration_start=None,calibration_end=None):
    """Fit on paired dates only. Returned fit scores are IN-SAMPLE diagnostics."""
    cfg=config or GloFASHighFlowBiasConfig();rows=[];models={}
    if windows.station.duplicated().any():raise ValueError('One explicitly selected window per station required')
    wm=windows.set_index('station')
    for name,obs in observations.items():
        if name not in glofas_at_stations or name not in wm.index:continue
        o=clean_daily_series(obs);s=clean_daily_series(glofas_at_stations[name]);w=wm.loc[name]
        paired=pd.concat([o.rename('obs'),s.rename('sim')],axis=1).loc[calibration_start:calibration_end]
        m=_circular_monthday_mask(paired.index,w.start_month_day,w.end_month_day)
        pairs=paired.loc[m].dropna()
        rec=dict(station=name,method=cfg.method,scope=cfg.correction_scope,selector=cfg.selector,
                 start_month_day=w.start_month_day,end_month_day=w.end_month_day,
                 peak_month_day=w.get('peak_month_day',np.nan),evaluation_scope='in_sample_calibration')
        try:
            model=_fit_correction_model_arrays(pairs.obs.values,pairs.sim.values,cfg.method,cfg)
            model.update(start_month_day=w.start_month_day,end_month_day=w.end_month_day)
            models[name]=model
            if cfg.method=='log_ratio':pairs=pairs[(pairs.obs>cfg.min_q)&(pairs.sim>cfg.min_q)]
            pred=_apply_model_array(pairs.sim.values,model)
            metrics=_hydro_metrics(pairs.obs.values,pred);raw=_hydro_metrics(pairs.obs.values,pairs.sim.values)
            rec.update(fit_status='OK',n_pairs=model['n_pairs'],multiplicative_factor=model.get('factor',np.nan),
                       log_factor=model.get('log_factor',np.nan),calibration_start=str(pairs.index.min().date()),
                       calibration_end=str(pairs.index.max().date()),calibration_n_years=pairs.index.year.nunique(),
                       raw_mean_bias_pct=raw['pbias'],corrected_mean_bias_pct=metrics['pbias'],
                       raw_rmse=raw['rmse'],corrected_rmse=metrics['rmse'])
        except ValueError as exc:rec.update(fit_status='INSUFFICIENT_OR_INVALID_PAIRS',n_pairs=len(pairs),error=str(exc))
        rows.append(rec)
    return pd.DataFrame(rows),models


def _apply_model_array(values: np.ndarray, model: dict[str, Any]) -> np.ndarray:
    x = np.asarray(values, dtype=float)
    method = model["method"]
    if method in {"mean_ratio", "log_ratio"}:
        return np.maximum(x * float(model["factor"]), 0.0)
    if method == "eqm":
        return _apply_eqm(x, model)
    raise ValueError(f"Unknown model method: {method}")


def apply_gauged_highflow_corrections(
    glofas_at_stations: pd.DataFrame,
    models: dict[str, dict[str, Any]],
    *,
    outside_highflow: Literal["raw", "nan", "corrected_all"] = "raw",
) -> pd.DataFrame:
    """Apply station-specific high-flow corrections to extracted GloFAS series."""
    out = glofas_at_stations.copy().astype(float)
    idx = pd.DatetimeIndex(out.index)
    for station, model in models.items():
        if station not in out.columns:
            continue
        raw = out[station].to_numpy(dtype=float)
        corr = _apply_model_array(raw, model)
        mask = _circular_monthday_mask(idx, model["start_month_day"], model["end_month_day"])
        if outside_highflow == "raw":
            vals = raw.copy(); vals[mask] = corr[mask]
        elif outside_highflow == "nan":
            vals = np.full_like(raw, np.nan); vals[mask] = corr[mask]
        elif outside_highflow == "corrected_all":
            vals = corr
        else:
            raise ValueError("outside_highflow must be raw, nan, or corrected_all.")
        out[station] = vals
    return out


# =============================================================================
# HydroBASINS network regionalization for multiplicative factors
# =============================================================================


def build_hybas_digraph(
    basins: gpd.GeoDataFrame,
    *,
    id_col: str = "HYBAS_ID",
    next_down_col: str = "NEXT_DOWN",
    uparea_col: str = "UP_AREA",
    main_bas_col: str = "MAIN_BAS",
) -> nx.DiGraph:
    """Build a directed HydroBASINS graph (upstream -> immediate downstream)."""
    for c in [id_col, next_down_col]:
        if c not in basins.columns:
            raise ValueError(f"HydroBASINS layer must contain {c!r}.")
    b = basins.copy()
    if b[id_col].duplicated().any():raise ValueError("Duplicate HYBAS_ID in drainage network")
    if b.crs is None:
        raise ValueError("Supply HydroBASINS CRS explicitly")
    b = b.to_crs(4326).reset_index(drop=True)
    # representative_point() avoids the geographic-CRS centroid warning and is
    # sufficient only as a fallback when HydroBASINS DIST_MAIN is unavailable.
    reps = b.geometry.representative_point()
    ids = set(pd.to_numeric(b[id_col], errors="coerce").dropna().astype(int))
    G = nx.DiGraph()
    for _, r in b.iterrows():
        hid = int(r[id_col])
        G.add_node(
            hid,
            UP_AREA=float(r[uparea_col]) if uparea_col in b.columns and pd.notna(r[uparea_col]) else np.nan,
            MAIN_BAS=int(r[main_bas_col]) if main_bas_col in b.columns and pd.notna(r[main_bas_col]) else None,
        )
    for i, r in b.reset_index(drop=True).iterrows():
        hid = int(r[id_col])
        nd = int(r[next_down_col]) if pd.notna(r[next_down_col]) else 0
        if nd == 0 or nd not in ids:
            continue
        jidx = np.where(pd.to_numeric(b[id_col], errors="coerce").to_numpy() == nd)[0]
        if len(jidx) == 0:
            continue
        j = int(jidx[0])
        if "DIST_MAIN" in b.columns and pd.notna(r.get("DIST_MAIN")) and pd.notna(b.iloc[j].get("DIST_MAIN")):
            # Along a directed upstream->downstream edge, the difference of
            # DIST_MAIN values is an actual along-network distance in km.
            d = abs(float(r["DIST_MAIN"]) - float(b.iloc[j]["DIST_MAIN"]))
        else:
            d = float(_haversine_km(reps.iloc[i].x, reps.iloc[i].y, reps.iloc[j].x, reps.iloc[j].y))
        G.add_edge(hid, nd, distance_km=max(d, 1e-6))
    if not nx.is_directed_acyclic_graph(G):raise ValueError("Drainage network contains a directed cycle")
    missing=[int(v) for v in b[next_down_col] if pd.notna(v) and int(v)!=0 and int(v) not in ids]
    G.graph["missing_downstream_ids"]=sorted(set(missing))
    return G


def _directed_relation_and_distance(G: nx.DiGraph, donor: int, target: int) -> tuple[str, float]:
    if donor == target:
        return "same", 0.0
    try:
        d = nx.shortest_path_length(G, donor, target, weight="distance_km")
        return "upstream_donor", float(d)
    except nx.NetworkXNoPath:
        pass
    try:
        d = nx.shortest_path_length(G, target, donor, weight="distance_km")
        return "downstream_donor", float(d)
    except nx.NetworkXNoPath:
        pass
    UG = G.to_undirected(as_view=True)
    try:
        d = nx.shortest_path_length(UG, donor, target, weight="distance_km")
        return "lateral", float(d)
    except nx.NetworkXNoPath:
        return "disconnected", np.inf


def propagate_multiplicative_factors_network(
    basins: gpd.GeoDataFrame,
    gauged_factors: pd.DataFrame,
    *,
    id_col: str = "HYBAS_ID",
    factor_col: str = "multiplicative_factor",
    station_col: str = "station",
    config: Optional[GloFASHighFlowBiasConfig] = None,
) -> pd.DataFrame:
    """Regionalize positive gauged correction factors along HydroBASINS network.

    The interpolation is performed in log-factor space. Directly connected
    upstream/downstream donors are preferred; lateral donors are excluded by
    default. Donors with very different UP_AREA are penalized / optionally
    rejected.
    """
    cfg = config or GloFASHighFlowBiasConfig()
    G = build_hybas_digraph(basins)
    gf = gauged_factors.copy()
    gf[id_col] = pd.to_numeric(gf[id_col], errors="coerce")
    gf[factor_col] = pd.to_numeric(gf[factor_col], errors="coerce")
    gf = gf.dropna(subset=[id_col, factor_col])
    gf = gf[gf[factor_col] > 0]
    if gf.empty:
        raise ValueError("No positive gauged correction factors available for propagation.")
    # Multiple gauges mapped to one level-9 basin: geometric mean is the natural
    # aggregation for multiplicative factors.
    agg = gf.groupby(id_col).agg(
        log_factor=(factor_col, lambda x: float(np.mean(np.log(np.asarray(x, dtype=float))))),
        n_gauges=(factor_col, "size"),
    )
    station_names = gf.groupby(id_col)[station_col].apply(lambda x: ";".join(map(str, x))).to_dict() if station_col in gf else {}

    donors = [int(i) for i in agg.index if int(i) in G]
    out_rows = []
    for target in G.nodes:
        target_area = G.nodes[target].get("UP_AREA", np.nan)
        target_main = G.nodes[target].get("MAIN_BAS", None)
        if target in agg.index:
            out_rows.append({
                id_col: target,
                "log_factor": float(agg.loc[target, "log_factor"]),
                "multiplicative_factor": float(np.exp(agg.loc[target, "log_factor"])),
                "propagation_status": "GAUGED",
                "primary_donor_hybas": target,
                "primary_donor_station": station_names.get(target, ""),
                "n_donors_used": int(agg.loc[target, "n_gauges"]),
                "primary_relation": "same",
                "primary_network_distance_km": 0.0,
            })
            continue

        cand = []
        for donor in donors:
            donor_main = G.nodes[donor].get("MAIN_BAS", None)
            if target_main is not None and donor_main is not None and target_main != donor_main:
                continue
            relation, d = _directed_relation_and_distance(G, donor, target)
            if not np.isfinite(d) or relation == "disconnected":
                continue
            if relation == "lateral" and not cfg.allow_lateral_donors:
                continue
            donor_area = G.nodes[donor].get("UP_AREA", np.nan)
            penalty = 0.0
            ratio = np.nan
            if np.isfinite(target_area) and target_area > 0 and np.isfinite(donor_area) and donor_area > 0:
                ratio = max(target_area / donor_area, donor_area / target_area)
                if cfg.network_max_area_ratio is not None and ratio > cfg.network_max_area_ratio:
                    continue
                penalty = abs(math.log(target_area / donor_area))
            # Keep physical network distance as the dominant term, augmented by
            # a dimensionless catchment-scale penalty.
            d_eff = max(float(d), 1e-6) * (1.0 + cfg.network_area_penalty * penalty)
            # Slight relation preference: direct upstream is most useful,
            # downstream second, lateral last (if allowed).
            relation_multiplier = {"upstream_donor": 1.0, "downstream_donor": 1.15, "lateral": 1.5}.get(relation, 1.0)
            d_eff *= relation_multiplier
            cand.append((d_eff, donor, relation, d, ratio))

        if not cand:
            # No information is not a global geometric-mean correction.
            lf = np.nan
            out_rows.append({
                id_col: target,
                "log_factor": lf,
                "multiplicative_factor": float(np.exp(lf)),
                "propagation_status": "NO_ADMISSIBLE_DONOR",
                "primary_donor_hybas": np.nan,
                "primary_donor_station": "",
                "n_donors_used": 0,
                "primary_relation": "none",
                "primary_network_distance_km": np.nan,
            })
            continue
        cand.sort(key=lambda z: z[0])
        use = cand[: max(1, int(cfg.network_k_donors))]
        deff = np.asarray([u[0] for u in use], dtype=float)
        w = 1.0 / np.maximum(deff, 1e-9) ** float(cfg.network_power)
        lfs = np.asarray([float(agg.loc[u[1], "log_factor"]) for u in use])
        lf = float(np.dot(w, lfs) / w.sum())
        primary = use[0]
        out_rows.append({
            id_col: target,
            "log_factor": lf,
            "multiplicative_factor": float(np.exp(lf)),
            "propagation_status": "NETWORK",
            "primary_donor_hybas": primary[1],
            "primary_donor_station": station_names.get(primary[1], ""),
            "n_donors_used": len(use),
            "primary_relation": primary[2],
            "primary_network_distance_km": primary[3],
            "primary_area_ratio": primary[4],
        })
    return pd.DataFrame(out_rows)


# =============================================================================
# Apply propagated factors to all HYBAS outlets
# =============================================================================


def extract_all_hybas_glofas_outlets(
    glofas_discharge: xr.DataArray | xr.Dataset,
    outlet_matches: pd.DataFrame,
    *,
    variable: Optional[str] = None,
    id_col: str = "HYBAS_ID",
    valid_status: tuple[str, ...] = ("OK", "LOW_CANDIDATE_SUPPORT"),
) -> xr.DataArray:
    """Extract GloFAS discharge at every matched HYBAS outlet as (T, HYBAS_ID)."""
    da = _pick_dataarray(glofas_discharge, variable)
    da, xname, yname = _normalize_xy_names(da)
    tname = _infer_time_name(da)
    arrs = []
    ids = []
    for _, r in outlet_matches.iterrows():
        if str(r.get("match_status")) not in valid_status:
            continue
        if not pd.notna(r.get(id_col)):
            continue
        lon = _normalize_lon_for_grid(float(r["glofas_lon"]), np.asarray(da[xname].values))
        lat = float(r["glofas_lat"])
        try:
            q = da.sel({xname: lon, yname: lat})
        except KeyError as exc:
            raise ValueError("Matched auxiliary pixel is absent from discharge grid; check version/grid alignment") from exc
        extras = [d for d in q.dims if d != tname]
        if extras:
            raise ValueError(f"Reduce non-time GloFAS dimensions before outlet extraction: {extras}")
        hid = int(r[id_col])
        q = q.rename({tname: "T"}) if tname != "T" else q
        q = q.assign_coords(HYBAS_ID=hid).expand_dims("HYBAS_ID")
        arrs.append(q); ids.append(hid)
    if not arrs:
        raise ValueError("No matched HYBAS outlets could be extracted from GloFAS.")
    if len(set(ids))!=len(ids):raise ValueError("Duplicate HYBAS outlet identifiers")
    out = xr.concat(arrs, dim="HYBAS_ID", coords="different", compat="equals", join="exact").transpose("T", "HYBAS_ID")
    out.name = "GloFAS"
    return out


def apply_propagated_factors(
    q_outlets: xr.DataArray,
    propagated_factors: pd.DataFrame,
    *,
    id_col: str = "HYBAS_ID",
    factor_col: str = "multiplicative_factor",
) -> xr.DataArray:
    """Apply one positive multiplicative factor to each HYBAS outlet series."""
    if "HYBAS_ID" not in q_outlets.dims:
        raise ValueError("q_outlets must have a HYBAS_ID dimension.")
    pf = propagated_factors.set_index(id_col)[factor_col]
    ids = np.asarray(q_outlets["HYBAS_ID"].values)
    factors = np.asarray([pf.get(int(i), np.nan) for i in ids], dtype=float)
    f = xr.DataArray(factors, coords={"HYBAS_ID": ids}, dims=("HYBAS_ID",), name="bias_factor")
    out = q_outlets * f
    out.name = "GloFAS_corrected"
    out.attrs.update({
        "correction": "multiplicative factor regionalized in log space on HydroBASINS network",
        "outside_highflow_note": "This function applies factors to all dates; use calendar-window masks when high-flow-only application is required.",
    })
    return out


def outlet_xarray_to_picture(
    da: xr.DataArray,
    outlet_metadata: pd.DataFrame,
    *,
    id_col: str = "HYBAS_ID",
    lon_col: str = "target_lon",
    lat_col: str = "target_lat",
    elev_col: Optional[str] = None,
    date_format: str = "%Y%m%d",
) -> pd.DataFrame:
    """Export (T, HYBAS_ID) outlet discharge to a WASS2S-like DataFrame."""
    if not {"T", "HYBAS_ID"}.issubset(da.dims):
        raise ValueError("Expected DataArray dimensions T and HYBAS_ID.")
    meta = outlet_metadata.drop_duplicates(id_col).set_index(id_col)
    ids = [int(v) for v in da["HYBAS_ID"].values]
    names = [str(v) for v in ids]
    rows = []
    rows.append(["LON"] + [meta.loc[i, lon_col] if i in meta.index else np.nan for i in ids])
    rows.append(["LAT"] + [meta.loc[i, lat_col] if i in meta.index else np.nan for i in ids])
    rows.append(["DAILY/ELEV"] + [meta.loc[i, elev_col] if elev_col and elev_col in meta.columns and i in meta.index else np.nan for i in ids])
    vals = da.transpose("T", "HYBAS_ID").values
    times = pd.DatetimeIndex(da["T"].values)
    for ti, t in enumerate(times):
        rows.append([pd.Timestamp(t).strftime(date_format)] + vals[ti, :].tolist())
    return pd.DataFrame(rows, columns=["ID"] + names)



def make_hybas_outlet_targets(
    hybas9: gpd.GeoDataFrame,
    hybas_outlets: gpd.GeoDataFrame,
    *,
    id_col: str = "HYBAS_ID",
    uparea_col: str = "UP_AREA",
) -> pd.DataFrame:
    """Build the target table required to match *all* level-9 outlets to GloFAS."""
    if id_col not in hybas9.columns or uparea_col not in hybas9.columns:
        raise ValueError(f"hybas9 must contain {id_col!r} and {uparea_col!r}.")
    if id_col not in hybas_outlets.columns:
        raise ValueError(f"hybas_outlets must contain {id_col!r}.")
    outs = hybas_outlets.copy()
    if outs.crs is None:
        outs = outs.set_crs(4326)
    outs = outs.to_crs(4326)
    basattrs = hybas9.copy()
    basattrs[id_col] = pd.to_numeric(basattrs[id_col], errors="coerce").astype("Int64")
    outs[id_col] = pd.to_numeric(outs[id_col], errors="coerce").astype("Int64")
    cols = [id_col, uparea_col]
    for c in ["NEXT_DOWN", "MAIN_BAS", "SUB_AREA", "DIST_MAIN", "DIST_SINK", "PFAF_ID"]:
        if c in basattrs.columns:
            cols.append(c)
    attrs = basattrs[cols].drop_duplicates(id_col)
    out = pd.DataFrame({
        id_col: outs[id_col].values,
        "target_lon": outs.geometry.x.values,
        "target_lat": outs.geometry.y.values,
    }).merge(attrs, on=id_col, how="left")
    out["HYBAS_UP_AREA_KM2"] = pd.to_numeric(out[uparea_col], errors="coerce")
    out["target_coordinate_source"] = "hybas_outlet"
    return out


def apply_propagated_highflow_factors(
    q_outlets: xr.DataArray,
    propagated_factors: pd.DataFrame,
    donor_windows: pd.DataFrame,
    *,
    id_col: str = "HYBAS_ID",
    factor_col: str = "multiplicative_factor",
    donor_station_col: str = "primary_donor_station",
    outside_highflow: Literal["raw", "nan", "corrected_all"] = "raw",
) -> tuple[xr.DataArray, pd.DataFrame]:
    """Apply propagated factors using the primary donor's observed HW/MODE calendar.

    For ungauged basins, the high-flow calendar is inherited from the primary
    network donor. This choice is explicit and returned in an audit table; it is
    preferable to silently applying a high-flow-fitted factor all year.
    """
    if not {"T", "HYBAS_ID"}.issubset(q_outlets.dims):
        raise ValueError("q_outlets must have dimensions T and HYBAS_ID.")
    pf = propagated_factors.drop_duplicates(id_col).set_index(id_col)
    w = donor_windows.drop_duplicates("station").set_index("station")
    times = pd.DatetimeIndex(q_outlets["T"].values)
    out = q_outlets.copy(deep=True).astype(float)
    audit = []
    for hid in q_outlets["HYBAS_ID"].values:
        ih = int(hid)
        if ih not in pf.index:
            audit.append({id_col: ih, "application_status": "NO_FACTOR"})
            continue
        factor = float(pf.loc[ih, factor_col])
        if not np.isfinite(factor) or factor<=0:
            if outside_highflow=="nan":out.loc[{"HYBAS_ID":hid}]=np.nan
            audit.append({id_col:ih,"application_status":"NO_VALID_FACTOR"})
            continue
        donor_raw = str(pf.loc[ih, donor_station_col]) if donor_station_col in pf.columns else ""
        donor = donor_raw.split(";")[0].strip() if donor_raw else ""
        raw = q_outlets.sel(HYBAS_ID=hid).values.astype(float)
        corr = raw * factor
        if outside_highflow == "corrected_all":
            vals = corr
            start_md = end_md = np.nan
            status = "CORRECTED_ALL"
        elif donor in w.index:
            wr = w.loc[donor]
            if isinstance(wr, pd.DataFrame):
                wr = wr.iloc[0]
            start_md = wr["start_month_day"]
            end_md = wr["end_month_day"]
            mask = _circular_monthday_mask(times, start_md, end_md)
            if outside_highflow == "raw":
                vals = raw.copy(); vals[mask] = corr[mask]
            elif outside_highflow == "nan":
                vals = np.full_like(raw, np.nan); vals[mask] = corr[mask]
            else:
                raise ValueError("outside_highflow must be raw, nan, or corrected_all.")
            status = "DONOR_WINDOW"
        else:
            # No observed donor window exists (e.g., regional fallback). Do not
            # silently distort low flows. Under 'raw', retain raw GloFAS.
            start_md = end_md = np.nan
            if outside_highflow == "nan":
                vals = np.full_like(raw, np.nan)
            else:
                vals = raw.copy()
            status = "NO_DONOR_WINDOW"
        out.loc[{"HYBAS_ID": hid}] = vals
        audit.append({
            id_col: ih,
            "factor": factor,
            "donor_station": donor,
            "start_month_day": start_md,
            "end_month_day": end_md,
            "application_status": status,
        })
    out.name = "GloFAS_highflow_corrected"
    out.attrs.update({
        "correction": "HydroBASINS-network regionalized multiplicative high-flow correction",
        "window_transfer": "Observed HW/MODE calendar inherited from primary gauged donor for ungauged basins",
    })
    return out, pd.DataFrame(audit)

# =============================================================================
# High-level orchestration helper (module-level, WAS_Hydro wrapper can call it)
# =============================================================================


def prepare_glofas_hybas_matching(
    station_metadata: pd.DataFrame,
    hybas9: gpd.GeoDataFrame,
    glofas_uparea: xr.DataArray | xr.Dataset,
    *,
    hybas_outlets: Optional[gpd.GeoDataFrame] = None,
    uparea_variable: Optional[str] = None,
    match_config: Optional[GloFASPixelMatchConfig] = None,
) -> dict[str, Any]:
    """Convenience function combining station->HYBAS and HYBAS->GloFAS matching."""
    associated = associate_stations_to_hybas(
        station_metadata,
        hybas9,
        hybas_outlets=hybas_outlets,
        config=match_config,
    )
    matches, alternatives = match_hybas_outlets_to_glofas(
        associated,
        glofas_uparea,
        variable=uparea_variable,
        config=match_config,
    )
    return {
        "station_hybas": associated,
        "glofas_matches": matches,
        "glofas_candidates": alternatives,
        "match_config": match_config or GloFASPixelMatchConfig(),
    }

# =============================================================================
# Correction evaluation: temporal, spatial, and spatio-temporal CV
# =============================================================================


@dataclass
class GloFASCorrectionEvaluationConfig:
    """Configuration for independent evaluation of high-flow bias correction.

    ``window_validation="fixed_conditional"`` retains the supplied calendars
    and evaluates amplitude correction conditional on these calendars. Use an
    independent reference period for an independently fixed target definition.

    ``window_validation="refit"`` relearns windows within the outer folds. For
    a held-out station, its observed calendar is not used: timing is transferred
    from admissible training donors. Absolute test dates are purged across all
    donor anchors. ``temporal_strategy="expanding"`` additionally rejects future
    training dates. These tests evaluate target construction and correction;
    they do not turn reanalysis discharge into a seasonal forecast model.
    """

    methods: tuple[str, ...] = ("mean_ratio", "log_ratio", "eqm")

    # Temporal leave-one-hydrological-year-out CV at each gauged station.
    temporal_cv: bool = True

    # Leave-one-station-out CV of the spatial/regional transfer. Distributional
    # EQM is intentionally excluded from scalar-factor propagation.
    spatial_cv: bool = True

    # Most stringent test: remove a station AND the target hydrological year
    # from donor calibration. This can be substantially more expensive.
    spatiotemporal_cv: bool = False

    propagation_methods: tuple[str, ...] = ("network", "idw_log", "hybrid")

    # Minimum number of finite high-flow observations in a held-out year.
    min_test_pairs: int = 15

    # IDW in log-factor space (used only for spatial CV benchmarking).
    idw_k: int = 6
    idw_power: float = 2.0
    idw_max_distance_km: Optional[float] = None
    idw_same_main_basin: bool = True

    # Whether to keep point-level CV predictions. Useful for diagnostics but can
    # be large for regional applications.
    keep_daily_predictions: bool = True

    # Small denominator guard for percentage metrics.
    epsilon: float = 1e-12

    # fixed_conditional: evaluate amplitude correction with supplied calendars;
    # refit: learn calendars without held-out observations, including timing
    # regionalization for held-out stations. Neither is a forecast model.
    window_validation: str = "fixed_conditional"
    temporal_strategy: str = "leave_one_out"   # or "expanding"
    cv_anchor_month_day: str = "01-01"         # exogenous outer-CV anchor
    min_valid_fraction_mean: float = 0.80
    min_valid_fraction_max: float = 0.90
    purge_days: int = 0
    test_years: Optional[tuple[int, ...]] = None
    calibration_start: Optional[str] = None
    calibration_end: Optional[str] = None
    evaluation_start: Optional[str] = None
    evaluation_end: Optional[str] = None
    highflow_config: Any = None
    timing_transfer_config: Any = None
    topkriging_config: Any = None
    climatology_start: Optional[str] = None
    climatology_end: Optional[str] = None
    # Additional grouping prevents pseudo-independent duplicate-pixel gauges.
    exclude_same_pixel_donors: bool = True


def _hydro_year_labels(index: pd.DatetimeIndex, anchor_month_day: str) -> np.ndarray:
    """Return integer hydrological-year labels for arbitrary MM-DD anchors."""
    try:
        am, ad = map(int, str(anchor_month_day).split("-"))
    except Exception:
        am, ad = 1, 1
    code = index.month * 100 + index.day
    acode = am * 100 + ad
    years = index.year.to_numpy(dtype=int)
    return np.where(code >= acode, years, years - 1).astype(int)


def _rank_corr(a: np.ndarray, b: np.ndarray) -> float:
    if len(a) < 2:
        return np.nan
    ra = pd.Series(a).rank(method="average").to_numpy(dtype=float)
    rb = pd.Series(b).rank(method="average").to_numpy(dtype=float)
    if np.nanstd(ra) <= 0 or np.nanstd(rb) <= 0:
        return np.nan
    return float(np.corrcoef(ra, rb)[0, 1])


def _hydro_metrics(obs: np.ndarray, sim: np.ndarray, epsilon: float = 1e-12) -> dict[str, float]:
    """Standard deterministic hydrological metrics on paired finite values."""
    o = np.asarray(obs, dtype=float)
    s = np.asarray(sim, dtype=float)
    m = np.isfinite(o) & np.isfinite(s)
    o = o[m]; s = s[m]
    n = int(len(o))
    if n == 0:
        return dict(n=0,**{k:np.nan for k in ["bias","pbias","mae","rmse","pearson_r","spearman_r","kge","nse"]})

    err = s - o
    mean_o = float(np.mean(o))
    mean_s = float(np.mean(s))
    bias = float(np.mean(err))
    pbias = float(100.0 * np.sum(err) / np.sum(o)) if abs(float(np.sum(o))) > epsilon else np.nan
    mae = float(np.mean(np.abs(err)))
    rmse = float(np.sqrt(np.mean(err ** 2)))

    if n >= 2 and np.std(o) > epsilon and np.std(s) > epsilon:
        r = float(np.corrcoef(o, s)[0, 1])
        sr = _rank_corr(o, s)
    else:
        r = sr = np.nan

    if n >= 2 and np.std(o, ddof=1) > epsilon and np.isfinite(r):
        alpha = float(np.std(s, ddof=1) / np.std(o, ddof=1))
        beta = float(mean_s / mean_o) if abs(mean_o) > epsilon else np.nan
        kge = float(1.0 - np.sqrt((r - 1.0) ** 2 + (alpha - 1.0) ** 2 + (beta - 1.0) ** 2)) if np.isfinite(beta) else np.nan
        denom = float(np.sum((o - mean_o) ** 2))
        nse = float(1.0 - np.sum((s - o) ** 2) / denom) if denom > epsilon else np.nan
    else:
        kge = np.nan
    denom = float(np.sum((o-mean_o)**2))
    nse = float(1.0-np.sum((s-o)**2)/denom) if denom>epsilon else np.nan

    return {
        "n": n,
        "bias": bias,
        "pbias": pbias,
        "mae": mae,
        "rmse": rmse,
        "pearson_r": r,
        "spearman_r": sr,
        "kge": kge,
        "nse": nse,
    }


def _fit_correction_model_arrays(obs,sim,method,cfg):
    """Fit nonnegative paired discharges; zeros excluded ONLY for log ratios."""
    o=np.asarray(obs,float);s=np.asarray(sim,float)
    valid=np.isfinite(o)&np.isfinite(s)&(o>=0)&(s>=0)
    if method=='log_ratio':valid&=(o>cfg.min_q)&(s>cfg.min_q)
    o=o[valid];s=s[valid]
    if len(o)<cfg.min_pairs:raise ValueError(f'Insufficient training pairs: {len(o)} < {cfg.min_pairs}')
    if method=='mean_ratio':
        if s.sum()<=0:raise ValueError('Zero simulated training volume')
        factor=float(o.sum()/s.sum())
        return dict(method=method,factor=factor,n_pairs=len(o))
    if method=='log_ratio':
        lf=float(np.mean(np.log(o)-np.log(s)))
        return dict(method=method,factor=float(np.exp(lf)),log_factor=lf,n_pairs=len(o))
    if method=='eqm':return dict(method=method,n_pairs=len(o),**_fit_eqm(o,s,cfg.eqm_n_quantiles))
    raise ValueError(f'Unknown method {method}')


def _selected_station_arrays(station,observations,glofas_at_stations,windows,station_summary):
    if station not in observations or station not in glofas_at_stations:return None
    w=windows[windows.station.astype(str)==str(station)] if 'station' in windows else pd.DataFrame()
    if w.empty:return None
    if len(w)!=1:raise ValueError('Expected one selected window per station')
    o=clean_daily_series(observations[station]);s=clean_daily_series(glofas_at_stations[station])
    if o.empty or s.empty:return None
    a=max(o.index.min(),s.index.min());b=min(o.index.max(),s.index.max())
    if b<a:return None
    idx=pd.date_range(a,b,freq='D');ov=o.reindex(idx).to_numpy(float);sv=s.reindex(idx).to_numpy(float)
    wr=w.iloc[0];hf=_circular_monthday_mask(idx,wr.start_month_day,wr.end_month_day)
    ss=station_summary[station_summary.station.astype(str)==str(station)] if 'station' in station_summary else pd.DataFrame()
    anchor=str(ss.iloc[0].get('hydrological_anchor','01-01')) if not ss.empty else '01-01'
    return dict(index=idx,obs=ov,sim=sv,hf=hf,finite=hf&np.isfinite(ov)&np.isfinite(sv),
                hydro_year=_hydro_year_labels(idx,anchor),anchor=anchor,window=wr)


def _append_annual_prediction(rows,*,cv_type,station,method,propagation_method,hydro_year,
    dates,obs,raw,corrected,expected_dates=None,min_valid_fraction_mean=.8,min_valid_fraction_max=.9):
    """Fair annual scoring uses a complete expected calendar and identical days.

    An observed maximum from incomplete observations is only an observed-sample
    maximum. Coverage flags remain available even when the chosen threshold
    accepts it. No interpolation or leap-day removal is performed.
    """
    if expected_dates is None:
        raise ValueError('expected_dates is required: row count is not seasonal coverage')
    d=pd.DatetimeIndex(dates);expected=pd.DatetimeIndex(expected_dates)
    a=pd.DataFrame({'obs':obs,'raw':raw,'corrected':corrected},index=d).reindex(expected)
    valid=np.isfinite(a.to_numpy(float)).all(axis=1);a.loc[~valid,:]=np.nan
    stats={c:coverage_statistics(a[c],expected,min_valid_fraction_mean,min_valid_fraction_max) for c in a}
    rec=dict(cv_type=cv_type,station=station,method=method,propagation_method=propagation_method,
             hydro_year=int(hydro_year),n_days=int(valid.sum()),n_expected_days=len(expected),
             valid_fraction=float(valid.mean()) if len(valid) else 0.,
             expected_start=expected.min(),expected_end=expected.max(),
             mean_coverage_ok=stats['obs']['MEAN_OK'],max_coverage_ok=stats['obs']['MAX_OK'])
    for name,st in stats.items():
        rec[name+'_q_mean']=st['Q_MEAN'];rec[name+'_q_max']=st['Q_MAX'];rec[name+'_peak_date']=st['PEAK_DATE']
    for name in ('raw','corrected'):
        a1,a0=rec[name+'_peak_date'],rec['obs_peak_date']
        rec[name+'_peak_timing_error_days']=float((a1-a0).days) if pd.notna(a1) and pd.notna(a0) else np.nan
    rows.append(rec)


def _build_score_tables(daily_predictions,annual_predictions,epsilon):
    """Every corrected score carries its OWN raw baseline on the SAME support.

    RAW rows have reference_method naming the corrected method whose successful
    folds they use. Thus a method cannot gain skill just by failing difficult
    years. For a cross-method ranking use common_support_scores (also returned).
    """
    drows=[];prows=[];keys=['cv_type','station','method','propagation_method']
    def add_records(out,base,o,p,r):
        common=np.isfinite(o)&np.isfinite(p)&np.isfinite(r)
        scores=_hydro_metrics(o[common],p[common],epsilon);raw=_hydro_metrics(o[common],r[common],epsilon)
        rec=dict(base,reference_method=base['method'],**scores)
        rec.update({'raw_'+k:v for k,v in raw.items()});out.append(rec)
        rr=dict(rec);rr.update(raw);rr['method']='RAW';out.append(rr)
    if not daily_predictions.empty:
        for k,g in daily_predictions[daily_predictions.method!='RAW'].groupby(keys,dropna=False):
            base=dict(zip(keys,k));add_records(drows,base,g.obs.to_numpy(float),g.prediction.to_numpy(float),g.raw.to_numpy(float))
    if not annual_predictions.empty:
        for k,g in annual_predictions[annual_predictions.method!='RAW'].groupby(keys,dropna=False):
            for target,suffix in [('Q_HW_MEAN','q_mean'),('Q_HW_MAX','q_max')]:
                base=dict(zip(keys,k));base.update(target=target,n_years=int(g[['obs_'+suffix,'corrected_'+suffix,'raw_'+suffix]].dropna().shape[0]))
                add_records(prows,base,g['obs_'+suffix].to_numpy(float),g['corrected_'+suffix].to_numpy(float),g['raw_'+suffix].to_numpy(float))
                if suffix=='q_max':
                    for rec,name in [(prows[-2],'corrected'),(prows[-1],'raw')]:
                        err=g[name+'_peak_timing_error_days'].dropna().to_numpy(float)
                        rec.update(peak_timing_mae_days=float(np.abs(err).mean()) if len(err) else np.nan,
                                   peak_timing_bias_days=float(err.mean()) if len(err) else np.nan)
    return pd.DataFrame(drows),pd.DataFrame(prows)


def _add_skill_against_raw(scores):
    if scores.empty:return scores
    out=scores.copy()
    for m in ('rmse','mae'):
        if 'raw_'+m in out:
            den=out['raw_'+m].where(out['raw_'+m]>0)
            out[m+'_skill']=1-out[m]/den
    if 'raw_pbias' in out:out['abs_pbias_improvement']=out.raw_pbias.abs()-out.pbias.abs()
    for m in ('kge','nse'):
        if 'raw_'+m in out:out[m+'_gain']=out[m]-out['raw_'+m]
    return out


def _predict_idw_logfactor(
    target_row: pd.Series,
    donor_factors: pd.DataFrame,
    station_hybas: pd.DataFrame,
    eval_cfg: GloFASCorrectionEvaluationConfig,
) -> dict[str, Any]:
    """Predict a withheld station factor by geodesic IDW in log-factor space."""
    meta_cols = [c for c in ["station", "HYBAS_ID", "target_lon", "target_lat", "HYBAS_UP_AREA_KM2", "MAIN_BAS"] if c in station_hybas.columns]
    donors = donor_factors.merge(station_hybas[meta_cols].drop_duplicates("station"), on="station", how="left")
    donors = donors[np.isfinite(pd.to_numeric(donors["multiplicative_factor"], errors="coerce")) & (pd.to_numeric(donors["multiplicative_factor"], errors="coerce") > 0)].copy()
    if donors.empty:
        return {"factor": np.nan, "status": "NO_DONOR"}

    if eval_cfg.idw_same_main_basin and "MAIN_BAS" in donors.columns and pd.notna(target_row.get("MAIN_BAS")):
        same = donors[pd.to_numeric(donors["MAIN_BAS"], errors="coerce") == float(target_row["MAIN_BAS"])]
        donors = same
        if donors.empty: return {"factor": np.nan, "status": "NO_DONOR_IN_MAIN_BASIN"}

    lon = pd.to_numeric(donors["target_lon"], errors="coerce").to_numpy(dtype=float)
    lat = pd.to_numeric(donors["target_lat"], errors="coerce").to_numpy(dtype=float)
    d = _haversine_km(float(target_row["target_lon"]), float(target_row["target_lat"]), lon, lat)
    valid = np.isfinite(d)
    if eval_cfg.idw_max_distance_km is not None:
        valid &= d <= float(eval_cfg.idw_max_distance_km)
    donors = donors.loc[valid].copy(); d = d[valid]
    if donors.empty:
        return {"factor": np.nan, "status": "NO_DONOR_IN_RADIUS"}
    order = np.argsort(d)[: max(1, min(int(eval_cfg.idw_k), len(d)))]
    d = d[order]; donors = donors.iloc[order]
    if d[0] <= 1e-9:
        lf = float(np.log(float(donors.iloc[0]["multiplicative_factor"])))
        return {"factor": float(np.exp(lf)), "status": "IDW_EXACT", "primary_donor_station": str(donors.iloc[0]["station"]), "distance_km": float(d[0]), "n_donors": 1}
    weights = 1.0 / np.maximum(d, 1e-9) ** float(eval_cfg.idw_power)
    lfs = np.log(pd.to_numeric(donors["multiplicative_factor"], errors="coerce").to_numpy(dtype=float))
    lf = float(np.dot(weights, lfs) / np.sum(weights))
    return {
        "factor": float(np.exp(lf)),
        "status": "IDW",
        "primary_donor_station": str(donors.iloc[0]["station"]),
        "distance_km": float(d[0]),
        "n_donors": int(len(donors)),
    }


def _predict_spatial_factor(target_station,donor_factors,station_hybas,basins,propagation_method,bias_cfg,eval_cfg):
    """The same admissibility rules are explicit across all methods."""
    mm=station_hybas[station_hybas.station.astype(str)==str(target_station)]
    if mm.empty:return dict(factor=np.nan,status='TARGET_NOT_MAPPED')
    tr=mm.iloc[0]
    if propagation_method=='idw_log':
        # Apply the common area-ratio guard before the geometric benchmark.
        donors=donor_factors.copy()
        if bias_cfg.network_max_area_ratio is not None and tr.get('HYBAS_UP_AREA_KM2',0)>0:
            areas=station_hybas.set_index('station').HYBAS_UP_AREA_KM2
            a=pd.to_numeric(donors.station.map(areas),errors='coerce');ta=float(tr.HYBAS_UP_AREA_KM2)
            donors=donors[np.maximum(a/ta,ta/a)<=bias_cfg.network_max_area_ratio]
        return _predict_idw_logfactor(tr,donors,station_hybas,eval_cfg)
    if propagation_method=='topkriging':
        return _topkrige_logfactor(target_station,donor_factors,station_hybas,basins,bias_cfg,eval_cfg)
    if propagation_method not in {'network','hybrid'}:raise ValueError('Unsupported propagation method')
    cfg=replace(bias_cfg,network_area_penalty=0.) if propagation_method=='network' else bias_cfg
    G=getattr(eval_cfg,'_network_graph',None)
    if G is None:
        G=build_hybas_digraph(basins);eval_cfg._network_graph=G
    if pd.isna(tr.HYBAS_ID) or int(tr.HYBAS_ID) not in G:return dict(factor=np.nan,status='TARGET_NOT_MAPPED')
    target=int(tr.HYBAS_ID)
    donors=donor_factors.merge(station_hybas[['station','HYBAS_ID']],on='station').dropna(subset=['HYBAS_ID','multiplicative_factor'])
    donors=donors[np.isfinite(donors.multiplicative_factor)&(donors.multiplicative_factor>0)]
    items=[]
    cache=G.graph.setdefault('_paths',{})
    for hid,dd in donors.groupby('HYBAS_ID'):
        hid=int(hid)
        if hid not in G:continue
        tm,dm=G.nodes[target].get('MAIN_BAS'),G.nodes[hid].get('MAIN_BAS')
        if tm is not None and dm is not None and tm!=dm:continue
        if hid not in cache:
            cache[hid]=(nx.single_source_dijkstra_path_length(G,hid,weight='distance_km'),
                        nx.single_source_dijkstra_path_length(G.reverse(copy=False),hid,weight='distance_km'))
        forward,reverse=cache[hid]
        if hid==target:rel,d='same',0.
        elif target in forward:rel,d='upstream_donor',forward[target]
        elif target in reverse:rel,d='downstream_donor',reverse[target]
        elif cfg.allow_lateral_donors:
            rel,d=_directed_relation_and_distance(G,hid,target)
        else:continue
        if not np.isfinite(d):continue
        ta,da=G.nodes[target].get('UP_AREA',np.nan),G.nodes[hid].get('UP_AREA',np.nan)
        scale=abs(np.log(ta/da)) if ta>0 and da>0 else np.inf
        if cfg.network_max_area_ratio is not None and scale>np.log(cfg.network_max_area_ratio):continue
        effective=max(d,1e-6)*(1+cfg.network_area_penalty*(scale if np.isfinite(scale) else 0.))
        effective*={'same':1,'upstream_donor':1,'downstream_donor':1.15,'lateral':1.5}.get(rel,1)
        items.append((effective,float(np.log(dd.multiplicative_factor).mean()),str(dd.iloc[0].station),d,rel,hid))
    if not items:return dict(factor=np.nan,status='NO_ADMISSIBLE_DONOR')
    items=sorted(items)[:max(1,cfg.network_k_donors)]
    if items[0][4]=='same':items=items[:1]
    w=1/np.maximum([i[0] for i in items],1e-9)**cfg.network_power
    lf=float(np.dot(w,[i[1] for i in items])/w.sum())
    return dict(factor=float(np.exp(lf)),status='NETWORK',primary_donor_station=items[0][2],
                primary_donor_hybas=items[0][5],distance_km=items[0][3],n_donors=len(items))


def _fit_full_station_factors(
    observations: dict[str, pd.Series],
    glofas_at_stations: pd.DataFrame,
    windows: pd.DataFrame,
    method: str,
    cfg: GloFASHighFlowBiasConfig,
) -> pd.DataFrame:
    """Fit independent full-period station factors for spatial CV donors."""
    if method not in {"mean_ratio", "log_ratio"}:
        return pd.DataFrame(columns=["station", "multiplicative_factor"])
    ccfg = replace(cfg, method=method)
    fits, _ = fit_gauged_highflow_corrections(observations, glofas_at_stations, windows, config=ccfg)
    if fits.empty:
        return fits
    return fits[fits.get("fit_status", "") == "OK"].copy()


def temporal_highflow_cross_validation(observations,glofas_at_stations,windows,station_summary,*,bias_config=None,evaluation_config=None):
    return _run_highflow_cv('temporal',observations,glofas_at_stations,windows,station_summary,
                           pd.DataFrame(),None,bias_config,evaluation_config)


def spatial_highflow_cross_validation(observations,glofas_at_stations,windows,station_summary,station_hybas,basins,*,bias_config=None,evaluation_config=None):
    return _run_highflow_cv('spatial',observations,glofas_at_stations,windows,station_summary,
                           station_hybas,basins,bias_config,evaluation_config)


def spatiotemporal_highflow_cross_validation(observations,glofas_at_stations,windows,station_summary,station_hybas,basins,*,bias_config=None,evaluation_config=None):
    return _run_highflow_cv('spatiotemporal',observations,glofas_at_stations,windows,station_summary,
                           station_hybas,basins,bias_config,evaluation_config)


def evaluate_pixel_matches(matches: pd.DataFrame, raw_daily_scores: Optional[pd.DataFrame] = None) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Classify pixel-match quality and optionally join raw GloFAS performance."""
    d = matches.copy()
    area = pd.to_numeric(d.get("area_error_pct", np.nan), errors="coerce")
    dist = pd.to_numeric(d.get("distance_km", np.nan), errors="coerce")
    status = d.get("match_status", pd.Series("UNKNOWN", index=d.index)).astype(str)
    quality = np.full(len(d), "SUSPECT", dtype=object)
    quality[(area <= 35) & (dist <= 40)] = "MARGINAL"
    quality[(area <= 25) & (dist <= 20)] = "ACCEPTABLE"
    quality[(area <= 10) & (dist <= 10)] = "GOOD"
    quality[~status.isin(["OK", "LOW_CANDIDATE_SUPPORT", "AREA_MISMATCH"]).to_numpy()] = "INVALID"
    d["pixel_match_quality"] = quality

    if raw_daily_scores is not None and not raw_daily_scores.empty and "station" in d.columns:
        raw = raw_daily_scores[(raw_daily_scores["method"] == "RAW") & (raw_daily_scores["cv_type"] == "temporal")]
        cols = [c for c in ["station", "kge", "rmse", "pbias"] if c in raw.columns]
        if cols:
            rr = raw[cols].drop_duplicates("station").rename(columns={"kge": "raw_highflow_kge", "rmse": "raw_highflow_rmse", "pbias": "raw_highflow_pbias"})
            d = d.merge(rr, on="station", how="left")

    summary = d.groupby("pixel_match_quality", dropna=False).agg(
        n=("pixel_match_quality", "size"),
        median_area_error_pct=("area_error_pct", "median"),
        median_distance_km=("distance_km", "median"),
    ).reset_index()
    return d, summary


def build_method_ranking(daily_scores: pd.DataFrame, predictand_scores: pd.DataFrame) -> pd.DataFrame:
    """Aggregate station-level CV scores into a concise method comparison table."""
    rows: list[dict[str, Any]] = []
    if not daily_scores.empty:
        gcols = ["cv_type", "method", "propagation_method"]
        for keys, g in daily_scores.groupby(gcols, dropna=False):
            rows.append({
                "cv_type": keys[0], "target": "DAILY_HIGHFLOW", "method": keys[1], "propagation_method": keys[2],
                "n_stations": int(g["station"].nunique()),
                "median_kge": float(np.nanmedian(g["kge"])) if g["kge"].notna().any() else np.nan,
                "median_rmse": float(np.nanmedian(g["rmse"])) if g["rmse"].notna().any() else np.nan,
                "median_pbias_abs": float(np.nanmedian(np.abs(g["pbias"]))) if g["pbias"].notna().any() else np.nan,
                "median_rmse_skill": float(np.nanmedian(g["rmse_skill"])) if "rmse_skill" in g and g["rmse_skill"].notna().any() else np.nan,
                "median_kge_gain": float(np.nanmedian(g["kge_gain"])) if "kge_gain" in g and g["kge_gain"].notna().any() else np.nan,
            })
    if not predictand_scores.empty:
        gcols = ["cv_type", "target", "method", "propagation_method"]
        for keys, g in predictand_scores.groupby(gcols, dropna=False):
            rows.append({
                "cv_type": keys[0], "target": keys[1], "method": keys[2], "propagation_method": keys[3],
                "n_stations": int(g["station"].nunique()),
                "median_kge": float(np.nanmedian(g["kge"])) if g["kge"].notna().any() else np.nan,
                "median_rmse": float(np.nanmedian(g["rmse"])) if g["rmse"].notna().any() else np.nan,
                "median_pbias_abs": float(np.nanmedian(np.abs(g["pbias"]))) if g["pbias"].notna().any() else np.nan,
                "median_rmse_skill": float(np.nanmedian(g["rmse_skill"])) if "rmse_skill" in g and g["rmse_skill"].notna().any() else np.nan,
                "median_kge_gain": float(np.nanmedian(g["kge_gain"])) if "kge_gain" in g and g["kge_gain"].notna().any() else np.nan,
                "median_peak_timing_mae_days": float(np.nanmedian(g["peak_timing_mae_days"])) if "peak_timing_mae_days" in g and g["peak_timing_mae_days"].notna().any() else np.nan,
            })
    out = pd.DataFrame(rows)
    if not out.empty:
        # Positive skill/gain first; RAW remains an explicit baseline row.
        out = out.sort_values(["cv_type", "target", "median_rmse_skill", "median_kge"], ascending=[True, True, False, False], na_position="last").reset_index(drop=True)
    return out


def evaluate_glofas_highflow_correction(observations,glofas_at_stations,windows,station_summary,
    station_hybas,basins,*,pixel_matches=None,bias_config=None,evaluation_config=None,periods=None,highflow_config=None):
    """Evaluate an explicitly declared target-definition / correction protocol.

    ``window_validation='fixed_conditional'`` does NOT test an ungauged calendar.
    ``'refit'`` excludes target observations from spatial calendar estimation and
    held-out absolute dates from temporal training. Refitted cross-year windows
    use a conservative TWO-year purge. Neither protocol trains a seasonal
    forecasting model. All annual scores enforce expected-date completeness.
    """
    cfg=replace(evaluation_config or GloFASCorrectionEvaluationConfig())
    if highflow_config is not None:cfg.highflow_config=highflow_config
    if periods is not None:
        periods.validate()
        for a,b in [('calibration_start',periods.calibration[0]),('calibration_end',periods.calibration[1]),
                    ('climatology_start',periods.climatology[0]),('climatology_end',periods.climatology[1]),
                    ('evaluation_start',periods.application[0]),('evaluation_end',periods.application[1])]:
            if getattr(cfg,a) is not None and b is not None and pd.Timestamp(getattr(cfg,a))!=pd.Timestamp(b):
                raise ValueError(f'Conflicting {a} in periods and evaluation_config')
            if b is not None:setattr(cfg,a,b)
    _validate_evaluation_config(cfg)
    results={}
    if cfg.temporal_cv:results['temporal']=temporal_highflow_cross_validation(observations,glofas_at_stations,windows,station_summary,bias_config=bias_config,evaluation_config=cfg)
    if cfg.spatial_cv:results['spatial']=spatial_highflow_cross_validation(observations,glofas_at_stations,windows,station_summary,station_hybas,basins,bias_config=bias_config,evaluation_config=cfg)
    if cfg.spatiotemporal_cv:results['spatiotemporal']=spatiotemporal_highflow_cross_validation(observations,glofas_at_stations,windows,station_summary,station_hybas,basins,bias_config=bias_config,evaluation_config=cfg)
    def combine(k):
        parts=[r[k] for r in results.values() if k in r and not r[k].empty]
        return pd.concat(parts,ignore_index=True,sort=False) if parts else pd.DataFrame()
    daily=combine('daily_predictions');annual=combine('annual_predictions')
    ds=combine('daily_scores');ps=combine('predictand_scores');common=common_support_scores(daily,annual,cfg.epsilon)
    detail,summary=evaluate_pixel_matches(pixel_matches,ds) if pixel_matches is not None else (pd.DataFrame(),pd.DataFrame())
    out=dict(daily_scores=ds,predictand_scores=ps,annual_predictions=annual,
             daily_predictions=daily if cfg.keep_daily_predictions else pd.DataFrame(),
             common_support_daily_scores=common['daily_scores'],common_support_predictand_scores=common['predictand_scores'],
             temporal_folds=results.get('temporal',{}).get('folds',pd.DataFrame()),
             spatial_folds=results.get('spatial',{}).get('folds',pd.DataFrame()),
             spatiotemporal_folds=results.get('spatiotemporal',{}).get('folds',pd.DataFrame()),
             spatial_factor_cv=combine('factor_cv'),pixel_matching_detail=detail,pixel_matching_summary=summary,
             method_ranking=build_method_ranking(common['daily_scores'],common['predictand_scores']),
             evaluation_config=cfg,window_validation=cfg.window_validation,
             skipped_methods=[dict(cv_type=k,method='eqm',reason='Distributional map is not a scalar factor')
                              for k in results if k!='temporal' and 'eqm' in cfg.methods],
             validation_scope='Retrospective correction/transfer, not seasonal forecast skill',
             uncertainty_note='No mass-balance guarantee; observed-sample maxima need coverage review')
    if periods is not None:out['periods']=periods.describe()
    return out


# =============================================================================
# v4 validation engine: explicit dates, independent training and matched scores.
# =============================================================================

def _validate_evaluation_config(cfg):
    if cfg.window_validation not in {'fixed_conditional','refit'}:
        raise ValueError('window_validation must be fixed_conditional or refit')
    if cfg.temporal_strategy not in {'leave_one_out','expanding'}:
        raise ValueError('temporal_strategy must be leave_one_out or expanding')
    if set(cfg.methods)-{'mean_ratio','log_ratio','eqm'}:raise ValueError('Unsupported correction method')
    if set(cfg.propagation_methods)-{'network','hybrid','idw_log','topkriging'}:raise ValueError('Unsupported propagation method')
    if len(set(cfg.methods))!=len(cfg.methods):raise ValueError('Duplicate methods')
    if not 0<cfg.min_valid_fraction_mean<=1 or not 0<cfg.min_valid_fraction_max<=1:
        raise ValueError('Coverage fractions must be in (0,1]')
    if cfg.purge_days<0:raise ValueError('purge_days must be nonnegative')
    pd.Timestamp('2001-'+cfg.cv_anchor_month_day)  # disallow Feb 29 as fixed anchor


def _md_window(window):
    """Normalize a definition without changing its physical time window."""
    w=dict(window)
    for side in ('start','end'):
        if side+'_month_day' not in w or pd.isna(w[side+'_month_day']):
            w[side+'_month_day']=doy_to_md(int(w[side+'_doy']))
        w[side+'_doy']=int(pd.Timestamp('2001-'+str(w[side+'_month_day'])).dayofyear)
    return w


def _cv_purge_interval(year,anchor,expected_dates=None,*,refit=False,purge_days=0):
    """Absolute dates excluded from ALL gauges, irrespective of their anchors.

    A refitted window may wrap the outer-year anchor. Before its estimation we
    conservatively remove TWO consecutive outer years. This is more expensive
    in sample size, but guarantees that no possible evaluated window is used
    to choose its own boundaries. Fixed windows need only their actual span.
    """
    ad=int(pd.Timestamp('2001-'+anchor).dayofyear)
    a,b=hydrological_year_bounds(int(year),ad)
    if refit:_,b=hydrological_year_bounds(int(year)+1,ad)
    if expected_dates is not None and len(expected_dates):
        a=min(a,pd.DatetimeIndex(expected_dates).min());b=max(b,pd.DatetimeIndex(expected_dates).max())
    return a-pd.Timedelta(days=purge_days),b+pd.Timedelta(days=purge_days)


def _training_series(series,ecfg,purge=None,*,climatology=False):
    """Missing dates remain explicit. No interpolation is introduced in CV."""
    q=clean_daily_series(series).copy()
    a,b=(ecfg.climatology_start,ecfg.climatology_end) if climatology else (ecfg.calibration_start,ecfg.calibration_end)
    if a is not None:q.loc[q.index<pd.Timestamp(a)]=np.nan
    if b is not None:q.loc[q.index>pd.Timestamp(b)]=np.nan
    if purge is not None:
        a,b=purge
        drop=(q.index>=a)&(q.index<=b)
        if ecfg.temporal_strategy=='expanding':drop=q.index>=a
        q.loc[drop]=np.nan
    return q


def _choose_cv_window(q,bcfg,ecfg,cache,key):
    """Refit all preprocessing, modes and breakpoints on allowed data only."""
    if key in cache:return cache[key]
    try:
        from .highwater_season_predictands_modeaware import HighWaterConfig,fit_station_highwater_regime
        from .highflow_window_regionalization import _window_from_regime
    except ImportError:
        from highwater_season_predictands_modeaware import HighWaterConfig,fit_station_highwater_regime
        from highflow_window_regionalization import _window_from_regime
    hc=replace(ecfg.highflow_config or HighWaterConfig(),n_bootstrap=0,
               mode_specific_outputs=bcfg.correction_scope=='mode',allow_weak_season_fallback=False)
    regime=fit_station_highwater_regime(q,hc)
    chosen=_window_from_regime(regime,bcfg.correction_scope,bcfg.selector)
    if chosen is None:raise ValueError('Requested season/mode is absent in training climatology')
    collection=regime.get('mode_windows' if bcfg.correction_scope=='mode' else 'windows',[])
    answer=(_md_window(chosen),[_md_window(v) for v in collection])
    cache[key]=answer
    return answer


def _cv_window_definitions(target,stations,obs,sim,windows,station_hybas,bcfg,ecfg,kind,purge,cache):
    """For an ungauged target, its OBS is NEVER used in window estimation.

    The local model window is shifted by donor timing biases using the same
    guarded regionalizer used in production. The predictand evaluated is then
    discharge *within this predicted window*, not a forecast of annual onset.
    """
    if ecfg.window_validation=='fixed_conditional':
        out={str(r.station):_md_window(r) for _,r in windows.iterrows()}
        return out,dict(window_source='supplied_fixed_definition',window_validation='fixed_conditional')
    refs={};models={};all_models={}
    pkey=None if purge is None else tuple(str(v) for v in purge)
    for s in stations:
        if kind=='temporal' and s!=target:continue
        if kind=='temporal' or s!=target:
            try:
                qo=_training_series(obs[s],ecfg,purge,climatology=True)
                refs[s],_=_choose_cv_window(qo,bcfg,ecfg,cache,('obs',s,pkey))
            except (ValueError,RuntimeError):continue
        if kind!='temporal':
            try:
                qs=_training_series(sim[s],ecfg,purge,climatology=True)
                models[s],all_models[s]=_choose_cv_window(qs,bcfg,ecfg,cache,('sim',s,pkey))
            except (ValueError,RuntimeError):continue
    if kind=='temporal':
        return refs,dict(window_source='training_observations',window_validation='refit')
    if target not in models:raise ValueError('No model window at withheld station')
    try:
        from .highflow_window_regionalization import GridWindowConfig,_predict_timing_target,corrected_grid_windows
        from .highflow_topkriging import build_supports_from_hybas
    except ImportError:
        from highflow_window_regionalization import GridWindowConfig,_predict_timing_target,corrected_grid_windows
        from highflow_topkriging import build_supports_from_hybas
    tc=ecfg.timing_transfer_config or GridWindowConfig(scope=bcfg.correction_scope,selector=bcfg.selector)
    meta=station_hybas.set_index('station',drop=False);dr=[]
    for s,w in refs.items():
        if s==target or s not in all_models or s not in meta.index:continue
        paired=match_windows([w],all_models[s],min_overlap=tc.min_mode_overlap)
        if not paired:continue
        mw=all_models[s][paired[0][1]];r=meta.loc[s]
        dr.append(dict(station=s,status='OK',lon=r.target_lon,lat=r.target_lat,
             main_basin=r.get('MAIN_BAS',np.nan),uparea_km2=r.get('MATCH_UP_AREA_KM2',r.get('HYBAS_UP_AREA_KM2',np.nan)),
             obs_start_doy=w['start_doy'],obs_end_doy=w['end_doy'],
             model_start_doy=mw['start_doy'],model_end_doy=mw['end_doy'],
             bias_start_days=(w['start_doy']-mw['start_doy']+182.5)%365-182.5,
             bias_end_days=(w['end_doy']-mw['end_doy']+182.5)%365-182.5))
    if not dr:raise ValueError('No homologous timing donor without target observations')
    r=meta.loc[target];mw=models[target]
    tr=pd.Series(dict(status='OK',station=target,lon=r.target_lon,lat=r.target_lat,y_index=0,x_index=0,
                      model_start_doy=mw['start_doy'],model_end_doy=mw['end_doy']))
    gs=ts=None
    if tc.transfer_method=='topkriging' and not tc.allow_disc_approximation:
        allsup=getattr(ecfg,'_timing_supports',{})
        gs={s:v for s,v in allsup.items() if s!=target};ts=allsup.get(target)
    info=_predict_timing_target(tr,pd.DataFrame(dr),tc,r.get('HYBAS_UP_AREA_KM2'),r.get('MAIN_BAS'),gs,ts)
    merged=tr.to_dict();merged.update(info)
    cw=corrected_grid_windows(pd.DataFrame([merged]),tc).iloc[0]
    if not np.isfinite(cw.corrected_start_doy+cw.corrected_end_doy):
        raise ValueError('Window regionalization failed: '+str(info.get('transfer_status')))
    refs[target]=_md_window(dict(start_doy=int(cw.corrected_start_doy),end_doy=int(cw.corrected_end_doy)))
    return refs,dict(window_source='model_plus_training_donor_timing_bias',window_validation='refit',
                     timing_donors=info.get('donor_stations',''),timing_method=tc.transfer_method)


def _topkrige_logfactor(target_station,donor_factors,station_hybas,basins,bias_cfg,eval_cfg):
    """Transfer log-factors with actual upstream unions, refitting per fold.

    exp(predicted log factor) is a geometric/median-type factor, not an unbiased
    lognormal arithmetic mean. Its MSE is conditional on the covariance model.
    """
    try:
        from .highflow_topkriging import TopKrigingConfig,build_supports_from_hybas,fit_point_variogram,topkriging_predict
    except ImportError:
        from highflow_topkriging import TopKrigingConfig,build_supports_from_hybas,fit_point_variogram,topkriging_predict
    meta=station_hybas.set_index('station',drop=False)
    if target_station not in meta.index:return dict(factor=np.nan,status='TARGET_NOT_MAPPED')
    tr=meta.loc[target_station];d=donor_factors.merge(station_hybas,on='station',how='inner')
    d=d[np.isfinite(d.multiplicative_factor)&(d.multiplicative_factor>0)&d.HYBAS_ID.notna()]
    if eval_cfg.idw_same_main_basin and pd.notna(tr.get('MAIN_BAS')):
        d=d[d.MAIN_BAS==tr.MAIN_BAS]
    area=float(tr.get('HYBAS_UP_AREA_KM2',np.nan))
    if bias_cfg.network_max_area_ratio is not None and np.isfinite(area) and area>0:
        aa=pd.to_numeric(d.HYBAS_UP_AREA_KM2,errors='coerce');ratio=np.maximum(aa/area,area/aa)
        d=d[ratio<=bias_cfg.network_max_area_ratio]
    if d.empty:return dict(factor=np.nan,status='NO_ADMISSIBLE_DONOR')
    cfg=eval_cfg.topkriging_config or TopKrigingConfig()
    # The largest spatial matrix is bounded by max_donors.
    dist=_haversine_km(tr.target_lon,tr.target_lat,d.target_lon,d.target_lat)
    d=d.iloc[np.argsort(dist)[:cfg.max_donors]]
    names=d.station.astype(str).tolist()+[str(target_station)];ids=d.HYBAS_ID.astype(int).tolist()+[int(tr.HYBAS_ID)]
    supports=build_supports_from_hybas(ids,basins,names=names,config=cfg)
    donors=[supports[n] for n in names[:-1]];z=np.log(d.multiplicative_factor.values)
    vg=fit_point_variogram(donors,z,cfg);p=topkriging_predict(supports[str(target_station)],donors,z,vg,cfg)
    return dict(factor=float(np.exp(p['value'])),status='TOPKRIGING',log_factor_variance=p['variance'],
                n_donors=p['n_donors'],primary_donor_station=names[0],variogram_status=vg['status'],
                support_kind=supports[str(target_station)].support_kind)


def _run_highflow_cv(kind,observations,glofas_at_stations,windows,station_summary,station_hybas,basins,bcfg=None,ecfg=None):
    bcfg=bcfg or GloFASHighFlowBiasConfig();ecfg=replace(ecfg or GloFASCorrectionEvaluationConfig())
    _validate_evaluation_config(ecfg)
    obs={str(k):clean_daily_series(v) for k,v in observations.items()}
    sim={str(k):clean_daily_series(glofas_at_stations[k]) for k in glofas_at_stations}
    stations=sorted(set(obs)&set(sim));cache={};daily=[];annual=[];folds=[];factors=[]
    meta=station_hybas.copy()
    if not meta.empty:meta['station']=meta.station.astype(str)
    # Build geometric supports once, but NEVER fit a variogram outside folds.
    if ecfg.window_validation=='refit' and kind!='temporal' and ecfg.timing_transfer_config is not None:
        tc=ecfg.timing_transfer_config
        if tc.transfer_method=='topkriging' and not tc.allow_disc_approximation:
            try:from .highflow_topkriging import build_supports_from_hybas
            except ImportError:from highflow_topkriging import build_supports_from_hybas
            mm=meta.dropna(subset=['HYBAS_ID'])
            ecfg._timing_supports=build_supports_from_hybas(mm.HYBAS_ID.astype(int).tolist(),basins,
                                  names=mm.station.tolist(),config=tc.topkriging_config)
    wm={str(r.station):_md_window(r) for _,r in windows.iterrows()} if 'station' in windows else {}
    for target in stations:
        if obs[target].empty or sim[target].empty:continue
        idx=obs[target].index.union(sim[target].index)
        if idx.empty:continue
        idx=pd.date_range(idx.min(),idx.max(),freq='D')
        if ecfg.evaluation_start is not None:idx=idx[idx>=pd.Timestamp(ecfg.evaluation_start)]
        if ecfg.evaluation_end is not None:idx=idx[idx<=pd.Timestamp(ecfg.evaluation_end)]
        if idx.empty:continue
        if ecfg.window_validation=='fixed_conditional':
            if target not in wm:continue
            ss=station_summary[station_summary.station.astype(str)==target] if 'station' in station_summary else pd.DataFrame()
            anchor=str(ss.iloc[0].get('hydrological_anchor','01-01')) if not ss.empty else '01-01'
        else:anchor=ecfg.cv_anchor_month_day
        ad=int(pd.Timestamp('2001-'+anchor).dayofyear)
        years=sorted(set(_hydro_year_labels(idx,anchor)))
        if ecfg.test_years is not None:years=[v for v in years if v in ecfg.test_years]
        for year in years:
            fixed_expected=window_dates(year,ad,wm[target]) if ecfg.window_validation=='fixed_conditional' else None
            purge=None if kind=='spatial' else _cv_purge_interval(year,anchor,fixed_expected,
                   refit=ecfg.window_validation=='refit',purge_days=ecfg.purge_days)
            # Exclude alias gauges that would return the held-out observation via
            # a duplicate simulated pixel. Coordinates must be mapped, not jittered.
            pool=list(stations)
            if kind!='temporal' and ecfg.exclude_same_pixel_donors and not meta.empty:
                mr=meta[meta.station==target]
                if not mr.empty:
                    tr=mr.iloc[0]
                    xy=('glofas_lon','glofas_lat') if {'glofas_lon','glofas_lat'}.issubset(meta) else None
                    if xy:
                        same=meta[(meta[xy[0]]==tr[xy[0]])&(meta[xy[1]]==tr[xy[1]])].station.tolist()
                        pool=[s for s in pool if s==target or s not in same]
            basefold=dict(cv_type=kind,station=target,hydro_year=int(year),window_validation=ecfg.window_validation,
                 temporal_strategy=ecfg.temporal_strategy,purge_start=purge[0] if purge else pd.NaT,
                 purge_end=purge[1] if purge else pd.NaT)
            try:
                defs,window_info=_cv_window_definitions(target,pool,obs,sim,windows,meta,bcfg,ecfg,kind,purge,cache)
                if target not in defs:raise ValueError('No admissible test window')
                expected=window_dates(year,ad,defs[target])
                if not expected.isin(idx).any():continue
                # Keep complete expected window even when evaluation data are
                # truncated; missing days must count in coverage.
                test=pd.DataFrame({'obs':obs[target].reindex(expected),'raw':sim[target].reindex(expected)})
                test.loc[~test.index.isin(idx),:]=np.nan
                if np.isfinite(test.values).all(1).sum()<ecfg.min_test_pairs:
                    folds.append(dict(basefold,status='INSUFFICIENT_TEST_PAIRS',**window_info));continue
            except (ValueError,RuntimeError) as exc:
                folds.append(dict(basefold,status='WINDOW_UNAVAILABLE',error=str(exc)));continue
            methods=ecfg.methods if kind=='temporal' else tuple(m for m in ecfg.methods if m!='eqm')
            for method in methods:
                cfg=replace(bcfg,method=method)
                model=None;df=[];train_dates=[]
                training_stations=[target] if kind=='temporal' else [s for s in pool if s!=target]
                for s in training_stations:
                    if s not in defs:continue
                    qo=_training_series(obs[s],ecfg,purge);qs=_training_series(sim[s],ecfg,purge)
                    pair=pd.concat([qo.rename('obs'),qs.rename('sim')],axis=1).dropna()
                    w=defs[s];pair=pair.loc[_circular_monthday_mask(pair.index,w['start_month_day'],w['end_month_day'])]
                    try:
                        fitted=_fit_correction_model_arrays(pair.obs.values,pair.sim.values,method,cfg)
                    except ValueError:continue
                    train_dates.extend(pair.index.tolist())
                    if kind=='temporal':model=fitted
                    else:df.append(dict(station=s,multiplicative_factor=fitted['factor']))
                td=pd.DatetimeIndex(train_dates);leaked=int(((td>=purge[0])&(td<=purge[1])).sum()) if purge else 0
                if leaked:raise AssertionError('Internal CV leakage: held-out dates found in training')
                props=('none',) if kind=='temporal' else ecfg.propagation_methods
                for prop in props:
                    rec=dict(basefold,method=method,propagation_method=prop,**window_info,
                             train_date_min=td.min(),train_date_max=td.max(),n_training_pairs=len(td),
                             leaked_training_dates=leaked,expected_start=expected.min(),expected_end=expected.max())
                    info={}
                    if kind=='temporal':
                        if model is None:folds.append(dict(rec,status='INSUFFICIENT_TRAINING'));continue
                    else:
                        if not df:folds.append(dict(rec,status='NO_DONOR_FACTOR'));continue
                        try:info=_predict_spatial_factor(target,pd.DataFrame(df),meta,basins,prop,cfg,ecfg)
                        except (ValueError,np.linalg.LinAlgError) as exc:
                            folds.append(dict(rec,status='PROPAGATION_FAILED',error=str(exc)));continue
                        factor=info.get('factor',np.nan)
                        if not np.isfinite(factor) or factor<=0:
                            folds.append(dict(rec,status=info.get('status','NO_FACTOR')));continue
                        model=dict(method=method,factor=factor)
                        # Full-record target factor is diagnostic only; never
                        # enters training, donor selection or model fitting.
                        try:
                            ref=_fit_correction_model_arrays(test.obs.values,test.raw.values,method,cfg).get('factor',np.nan)
                        except ValueError:ref=np.nan
                        factors.append(dict(rec,observed_factor=ref,predicted_factor=factor,
                            log_factor_error=np.log(factor/ref) if ref>0 else np.nan,
                            reference_factor_scope='held_out_window_diagnostic_only',**info))
                    pred=_apply_model_array(test.raw.values,model)
                    good=np.isfinite(test.obs.values)&np.isfinite(test.raw.values)&np.isfinite(pred)
                    for d,o,r,p in zip(expected[good],test.obs.values[good],test.raw.values[good],pred[good]):
                        daily.append(dict(cv_type=kind,station=target,method=method,propagation_method=prop,
                                      hydro_year=int(year),date=d,obs=float(o),raw=float(r),prediction=float(p)))
                    _append_annual_prediction(annual,cv_type=kind,station=target,method=method,
                        propagation_method=prop,hydro_year=year,dates=expected,expected_dates=expected,
                        obs=test.obs.values,raw=test.raw.values,corrected=pred,
                        min_valid_fraction_mean=ecfg.min_valid_fraction_mean,min_valid_fraction_max=ecfg.min_valid_fraction_max)
                    folds.append(dict(rec,status='OK',factor=model.get('factor',np.nan),n_donors=len(df)))
    d=pd.DataFrame(daily);a=pd.DataFrame(annual);ds,ps=_build_score_tables(d,a,ecfg.epsilon)
    return dict(daily_predictions=d,annual_predictions=a,daily_scores=_add_skill_against_raw(ds),
                predictand_scores=_add_skill_against_raw(ps),folds=pd.DataFrame(folds),factor_cv=pd.DataFrame(factors),
                validation_scope=ecfg.window_validation)


def common_support_scores(daily,annual,epsilon=1e-12):
    """Intersect successful method/propagation folds before comparing methods.

    This cannot repair a method with no prediction at all: missing experiments
    remain visible in folds and are excluded from a ranking, not called winners.
    """
    def common(df,extra):
        if df.empty:return df
        keep=[]
        for _,g in df.groupby(['cv_type','station'],dropna=False):
            methods=list(g[['method','propagation_method']].drop_duplicates().itertuples(index=False,name=None))
            sets=[]
            for m,p in methods:
                gg=g[(g.method==m)&(g.propagation_method==p)]
                sets.append(set(gg[['hydro_year']+extra].itertuples(index=False,name=None)))
            shared=set.intersection(*sets) if sets else set()
            keep.append(g[[v in shared for v in g[['hydro_year']+extra].itertuples(index=False,name=None)]])
        return pd.concat(keep,ignore_index=True) if keep else df.iloc[:0]
    d=common(daily,['date']);a=common(annual,[])
    ds,ps=_build_score_tables(d,a,epsilon)
    return dict(daily_scores=_add_skill_against_raw(ds),predictand_scores=_add_skill_against_raw(ps))



def propagate_multiplicative_factors(basins,gauged_factors,station_metadata,*,hybas_outlets=None,config=None):
    """Production counterpart of the CV spatial-factor estimator.

    EQM is not regionalized as a scalar. Actual upstream unions are used for
    top-kriging; IDW/top-kriging require explicit outlet locations. This routine
    does not reconstruct mass-conserving incremental inflows.
    """
    cfg=config or GloFASHighFlowBiasConfig()
    if cfg.method not in {'mean_ratio','log_ratio'}:raise ValueError('Only scalar factors can be regionalized')
    if cfg.propagation_method not in {'network','hybrid','idw_log','topkriging'}:raise ValueError('Unknown propagation method')
    if cfg.propagation_method in {'network','hybrid'}:
        cc=replace(cfg,network_area_penalty=0.) if cfg.propagation_method=='network' else cfg
        out=propagate_multiplicative_factors_network(basins,gauged_factors,config=cc)
        out['propagation_method']=cfg.propagation_method
        return out
    if hybas_outlets is None:raise ValueError('Explicit HYBAS outlet points are required for this propagation method')
    targets=make_hybas_outlet_targets(basins,hybas_outlets)
    targets['station']=['HYBAS_'+str(int(v)) for v in targets.HYBAS_ID]
    if 'MAIN_BAS' not in targets and 'MAIN_BAS' in basins:
        targets=targets.merge(basins[['HYBAS_ID','MAIN_BAS']],on='HYBAS_ID',how='left')
    meta=pd.concat([station_metadata,targets],ignore_index=True,sort=False)
    ecfg=GloFASCorrectionEvaluationConfig(idw_k=cfg.network_k_donors,idw_power=cfg.network_power,
                                         topkriging_config=cfg.topkriging_config)
    donors=gauged_factors[['station','multiplicative_factor']].copy();rows=[]
    for _,r in targets.iterrows():
        try:info=_predict_spatial_factor(r.station,donors,meta,basins,cfg.propagation_method,cfg,ecfg)
        except (ValueError,np.linalg.LinAlgError) as exc:info=dict(factor=np.nan,status='PROPAGATION_FAILED',error=str(exc))
        factor=info.get('factor',np.nan)
        rows.append(dict(HYBAS_ID=int(r.HYBAS_ID),multiplicative_factor=factor,
                         log_factor=np.log(factor) if factor>0 else np.nan,propagation_status=info.get('status'),
                         primary_donor_station=info.get('primary_donor_station',''),n_donors_used=info.get('n_donors',0),
                         propagation_method=cfg.propagation_method,log_factor_variance=info.get('log_factor_variance',np.nan),
                         variogram_status=info.get('variogram_status',''),error=info.get('error','')))
    return pd.DataFrame(rows)


__all__ = [
    "GloFASPixelMatchConfig",
    "GloFASHighFlowBiasConfig",
    "GloFASCorrectionEvaluationConfig",
    "associate_stations_to_hybas",
    "validate_glofas_auxiliary",
    "match_hybas_outlets_to_glofas",
    "extract_glofas_discharge_at_matches",
    "select_highflow_windows",
    "fit_gauged_highflow_corrections",
    "apply_gauged_highflow_corrections",
    "build_hybas_digraph",
    "propagate_multiplicative_factors_network",
    "propagate_multiplicative_factors",
    "extract_all_hybas_glofas_outlets",
    "apply_propagated_factors",
    "make_hybas_outlet_targets",
    "apply_propagated_highflow_factors",
    "outlet_xarray_to_picture",
    "prepare_glofas_hybas_matching",
    "temporal_highflow_cross_validation",
    "spatial_highflow_cross_validation",
    "spatiotemporal_highflow_cross_validation",
    "evaluate_pixel_matches",
    "build_method_ranking",
    "evaluate_glofas_highflow_correction",
]
