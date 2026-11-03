"""Transfer the observed high-water calendar from gauged outlets to a model grid.

THE PROBLEM
-----------
High-water windows are estimated from observations, which exist only at
gauged outlets, and they differ from one outlet to the next.  Gridded model
discharge (GloFAS, or any other routed hydrological model) exists everywhere.
To compute high-flow predictands on the grid, every river pixel needs a
window.  Transplanting the nearest gauge's observed window is a benchmark, not a guaranteed solution:
a headwater pixel and the main-stem outlet it is nearest to can have flood
seasons weeks apart, because the window is controlled by local climate and by
basin response time, both of which vary along the network.

THE APPROACH
------------
The transferred quantity is the model's TIMING ERROR, not the complete local
calendar. Its spatial smoothness is a TESTABLE HYPOTHESIS, not a guarantee.

    1. LOCAL MODEL WINDOW.  The same detection algorithm used on the
       observations is run on the model's own climatology at every river
       pixel.  This yields a window everywhere that already respects local
       regime: headwaters peak earlier than the main stem, distinct
       tributaries keep distinct calendars, bimodality is preserved where
       the model produces it.

    2. TIMING BIAS AT GAUGES.  At each gauged pixel, the observed window and
       the model window are compared.  The signed circular differences
       (d_start, d_end), in days, are the model's timing error there.

    3. REGIONALISE THE BIAS, NOT THE WINDOW.  (d_start, d_end) are
       transferred to every river pixel by network- and scale-aware
       weighting.  Timing errors may be spatially smoother than the windows
       themselves, if the part of the window that varies strongly in
       space - the local regime - is already carried by step 1.

    4. CORRECTED WINDOW = local model window shifted by the regionalised
       bias.  At a gauged pixel this returns the observed window exactly, by
       construction.

WHY THIS RESPECTS OUTLET INFLUENCE
----------------------------------
A gauge influences a pixel through the weights of step 3, which combine
network relation (same main basin, upstream/downstream), drainage-area
similarity in log space, and geodesic distance.  The zone of influence of a
gauge is therefore hydrological, not a Voronoi cell: a gauge on a different
river does not contaminate a neighbouring catchment merely by being close,
and a gauge far downstream on the same stem still informs its headwaters.

MODEL-AGNOSTIC
--------------
Nothing here is specific to GloFAS.  Any gridded daily discharge with dims
(T, Y, X) works.  An upstream-area grid is used when available, both to
define the river mask and to weight by scale; without it, a river mask can
be derived from mean discharge and the scale term is dropped.

VALIDATION
----------
``cross_validate_grid_windows`` holds out each gauge in turn, predicts its
window from the others, and reports circular errors in days against two
baselines: the uncorrected model window, and direct transplantation of the
nearest gauge's observed window.  If the method does not beat both, it should
not be used - and the function says so plainly.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Any, Optional, Sequence

import numpy as np
import pandas as pd
import xarray as xr

if __package__:
    from .highwater_season_predictands_modeaware import (
        HighWaterConfig,
        fit_station_highwater_regime,
        compute_annual_predictands,
        _rotpos_to_doy,
        _doy_to_rotpos,
        _doy_to_monthday,
        _monthday_to_doy,
    )
    from .highflow_method_validation import (
        circular_day_difference,
        circular_day_mean,
        circular_day_mae,
    )
    from .highflow_topkriging import (
        TopKrigingConfig,
        build_supports_from_points,
        fit_point_variogram,
        topkriging_predict,
    )
else:  # pragma: no cover - flat layout
    from highwater_season_predictands_modeaware import (
        HighWaterConfig,
        fit_station_highwater_regime,
        compute_annual_predictands,
        _rotpos_to_doy,
        _doy_to_rotpos,
        _doy_to_monthday,
        _monthday_to_doy,
    )
    from highflow_method_validation import (
        circular_day_difference,
        circular_day_mean,
        circular_day_mae,
    )
    from highflow_topkriging import (
        TopKrigingConfig,
        build_supports_from_points,
        fit_point_variogram,
        topkriging_predict,
    )

if __package__:
    from .highflow_utils import (area_to_km2, align_spatial_grid, wrap_doy,
        circular_bias_mean, unwrap_biases, match_windows, window_day_set)
else:
    from highflow_utils import (area_to_km2, align_spatial_grid, wrap_doy,
        circular_bias_mean, unwrap_biases, match_windows, window_day_set)

__all__ = [
    "GridWindowConfig",
    "build_river_mask",
    "fit_gridded_local_windows",
    "gauge_window_bias",
    "propagate_window_bias",
    "corrected_grid_windows",
    "compute_grid_highflow_predictands",
    "cross_validate_grid_windows",
    "regionalize_highflow_windows",
]

DAYS = 365


# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

@dataclass
class GridWindowConfig:
    """Settings for transferring high-water calendars to a model grid.

    Defaults are deliberately conservative and are not universal constants;
    they should be checked with ``cross_validate_grid_windows``.
    """

    # --- river mask -------------------------------------------------------
    min_uparea_km2: float = 100.0
    """Pixels draining less than this are not treated as river pixels.
    Below roughly this scale a 0.05 deg routed model carries little skill."""

    min_mean_discharge: float = 0.0
    """Fallback river criterion when no upstream-area grid is supplied."""

    max_pixels: Optional[int] = None
    """Safety cap on the number of pixels fitted, for exploratory runs."""

    # --- bias transfer ----------------------------------------------------
    k_donors: int = 4
    """Number of gauges combined for each pixel."""

    idw_power: float = 2.0

    area_penalty: float = 0.60
    """Weight of the log drainage-area mismatch in the effective distance.
    Set to 0 to ignore scale and use distance alone."""

    max_area_ratio: float = 8.0
    """Gauges whose drainage area differs by more than this factor are not
    used as donors for a pixel."""

    same_basin_only: bool = True
    """Restrict donors to the pixel's own main basin when basin membership
    is available.  The bias of a model in one river system does not inform
    another."""

    cross_basin_multiplier: float = 3.0
    """Distance inflation applied to a donor outside the pixel's main basin,
    used only when ``same_basin_only`` is False."""

    max_shift_days: int = 60
    """Timing corrections larger than this are treated as implausible and
    clipped; a model whose flood season is two months off is not merely
    biased, and the pixel is flagged rather than silently patched."""

    min_duration_days: int = 20
    max_duration_days: int = 300

    # --- window detection at pixels ---------------------------------------
    transfer_method: str = "idw"
    """How the timing bias is carried to the pixels: 'idw' (hydrologically
    weighted inverse distance, the default) or 'topkriging' (Skoien & Bloschl,
    accounting for catchment support and nestedness).  Top-kriging needs
    catchment supports, which are built from HydroBASINS when available and
    otherwise approximated by equivalent-area discs."""

    scope: str = "season"
    """'season' uses high-water seasons; 'mode' uses chronological modes."""

    selector: str | int = "main"
    allow_disc_approximation: bool = False
    # Top-kriging requires provided upstream supports unless this approximation
    # is deliberately enabled. Equivalent-area discs do NOT recover topology.
    topkriging_config: Optional[TopKrigingConfig] = None
    no_donor_policy: str = "nan"  # "nan" or explicit "raw"; never global mean
    min_mode_overlap: float = 0.20
    min_circular_resultant: float = 0.10
    assumed_uparea_units: Optional[str] = None
    pixel_batch_size: int = 16

    """'main' (largest-volume window), 'envelope' (first onset to last
    cessation), or an integer window number."""


# ---------------------------------------------------------------------------
# Geometry helpers
# ---------------------------------------------------------------------------

def _haversine_km(lon1, lat1, lon2, lat2):
    r = 6371.0088
    p1 = np.radians(np.asarray(lat1, dtype=float))
    p2 = np.radians(np.asarray(lat2, dtype=float))
    dp = p2 - p1
    dl = np.radians(np.asarray(lon2, dtype=float) - np.asarray(lon1, dtype=float))
    a = np.sin(dp / 2.0) ** 2 + np.cos(p1) * np.cos(p2) * np.sin(dl / 2.0) ** 2
    return 2.0 * r * np.arcsin(np.sqrt(np.clip(a, 0.0, 1.0)))


def _uparea_to_km2(uparea: xr.DataArray) -> xr.DataArray:
    return area_to_km2(uparea)


def _dim_names(da: xr.DataArray) -> tuple[str, str, str]:
    """Explicit coordinate conventions; extra/ambiguous dimensions are rejected."""
    def choose(candidates):
        hit=[d for d in candidates if d in da.dims]
        if len(hit)!=1:raise ValueError(f'Cannot identify a unique dimension from {candidates}.')
        return hit[0]
    return choose(('T','time','valid_time')),choose(('Y','lat','latitude')),choose(('X','lon','longitude'))


# ---------------------------------------------------------------------------
# 1. River mask and local model windows
# ---------------------------------------------------------------------------

def build_river_mask(discharge: xr.DataArray, uparea: Optional[xr.DataArray]=None,
                     config: Optional[GridWindowConfig]=None):
    """Label-aligned river mask. Only 2-D reductions are materialised in memory."""
    cfg=config or GridWindowConfig();t,y,x=_dim_names(discharge)
    if set(discharge.dims)!={t,y,x}:raise ValueError('Select a member/variable before river-window fitting.')
    q=discharge.transpose(t,y,x)
    finite=(np.isfinite(q)&(q>=0)).sum(t).values
    times=pd.DatetimeIndex(q[t].values)
    if times.empty:raise ValueError('Empty discharge time axis.')
    n_expected=(times.max().normalize()-times.min().normalize()).days+1
    has_data=finite>.5*n_expected
    if uparea is not None:
        a=area_to_km2(uparea,assumed_units=cfg.assumed_uparea_units)
        a=align_spatial_grid(a,q,y,x);area=a.values
        return has_data&np.isfinite(area)&(area>=cfg.min_uparea_km2),area
    mean=q.where(q>=0).mean(t,skipna=True).values
    return has_data&np.isfinite(mean)&(mean>cfg.min_mean_discharge),None


def _window_from_regime(regime: dict, scope: str, selector) -> Optional[dict]:
    """Select the exact requested object. A missing MODE never becomes HW1."""
    scope=str(scope).lower()
    if scope not in {'mode','season'}:raise ValueError("scope must be 'mode' or 'season'.")
    wins=regime.get('mode_windows' if scope=='mode' else 'windows',[]) or []
    wins=[w for w in wins if not w.get('weak_season_fallback',False)]
    if not wins:return None
    if isinstance(selector,str) and selector.lower()=='envelope':
        first=min(wins,key=lambda w:w['start_pos']);last=max(wins,key=lambda w:w['end_pos'])
        return dict(start_doy=first['start_doy'],end_doy=last['end_doy'],start_pos=first['start_pos'],
                    end_pos=last['end_pos'],duration_days=last['end_pos']-first['start_pos']+1,window_id='envelope')
    if isinstance(selector,str) and selector.lower()=='main':
        key='is_main_mode' if scope=='mode' else 'is_main_season'
        selected=[w for w in wins if w.get(key,False)]
        if len(selected)>1:raise ValueError('Multiple main windows: invalid regime definition.')
        if not selected:return None
        w=selected[0]
    else:
        try:num=int(selector)
        except (TypeError,ValueError):raise ValueError('selector must be main, envelope or a positive integer.')
        if num<1:raise ValueError('selector must be positive.')
        key='mode_id' if scope=='mode' else 'season_id'
        w=next((a for a in wins if int(a.get(key,-1))==num),None)
        if w is None:return None
    out=dict(w);out['window_id']=w.get('mode_id' if scope=='mode' else 'season_id')
    return out


def fit_gridded_local_windows(
    discharge: xr.DataArray,
    river_mask: np.ndarray,
    highflow_config: HighWaterConfig,
    config: Optional[GridWindowConfig] = None,
    reference_start=None,
    reference_end=None,
    progress_every: int = 0,
) -> pd.DataFrame:
    """Run the high-water detection on the model climatology at every river pixel.

    This is the step that preserves local regime.  It is also the expensive
    one: one climatology plus one dynamic-programming segmentation per pixel.
    Cost is linear in the number of river pixels.

    Failures are recorded, never raised: a model grid always contains pixels
    with degenerate or flat regimes.
    """
    cfg = config or GridWindowConfig()
    t, y, x = _dim_names(discharge)
    times = pd.DatetimeIndex(discharge[t].values)
    lats = np.asarray(discharge[y].values, dtype=float)
    lons = np.asarray(discharge[x].values, dtype=float)

    # Force bootstrap off: it would multiply an already heavy loop for no
    # benefit, since pixel-level confidence intervals are not used here.
    from dataclasses import replace as _replace
    pixel_cfg = _replace(highflow_config, n_bootstrap=0, mode_specific_outputs=(cfg.scope == "mode"))

    idx = np.argwhere(river_mask)
    if cfg.max_pixels is not None and len(idx) > int(cfg.max_pixels):
        step = int(np.ceil(len(idx) / float(cfg.max_pixels)))
        idx = idx[::step]

    if cfg.pixel_batch_size < 1: raise ValueError('pixel_batch_size must be positive.')
    discharge = discharge.transpose(t,y,x)
    batch_values = None
    rows = []
    for n, (iy, ix) in enumerate(idx):
        rec = {
            "y_index": int(iy),
            "x_index": int(ix),
            "lat": float(lats[iy]),
            "lon": float(lons[ix]),
        }
        try:
            if n % cfg.pixel_batch_size == 0:
                block = idx[n:n+cfg.pixel_batch_size]
                batch_values = discharge.isel({y:xr.DataArray(block[:,0],dims='P'),
                                                x:xr.DataArray(block[:,1],dims='P')}).values
            s = pd.Series(batch_values[:, n % cfg.pixel_batch_size], index=times)
            regime = fit_station_highwater_regime(
                s, pixel_cfg,
                reference_start=reference_start,
                reference_end=reference_end,
            )
            w = _window_from_regime(regime, cfg.scope, cfg.selector)
            if w is None:
                raise ValueError("no high-water window detected")
            rec.update({
                "status": "OK",
                "anchor_doy": int(regime["anchor_doy"]),
                "modality": regime.get("modality"),
                "n_modes": int(regime.get("n_modes", 0)),
                "model_start_doy": w["start_doy"],
                "model_end_doy": w["end_doy"],
                "model_duration_days": w["duration_days"],
                "model_start_month_day": _doy_to_monthday(w["start_doy"]),
                "model_end_month_day": _doy_to_monthday(w["end_doy"]),
                "model_window_id": w.get("window_id"),
                "window_family": cfg.scope,
                "model_windows": [dict(v) for v in regime.get("mode_windows" if cfg.scope=="mode" else "windows",[])],

            })
        except Exception as exc:  # noqa: BLE001
            rec.update({"status": "FAILED", "error": str(exc)[:160]})
        rows.append(rec)
        if progress_every and (n + 1) % progress_every == 0:
            print(f"  ... {n + 1}/{len(idx)} pixels", flush=True)

    return pd.DataFrame(rows)


# ---------------------------------------------------------------------------
# 2. Timing bias at gauges
# ---------------------------------------------------------------------------

def gauge_window_bias(observed_windows: pd.DataFrame, pixel_windows: pd.DataFrame,
                      gauge_pixels: pd.DataFrame, min_overlap: float=.2) -> pd.DataFrame:
    """One-to-one homology by circular overlap, never by local chronological ID.

    All candidate windows at the model pixel are considered when available.
    Unmatched observed modes remain unmatched and cannot calibrate a transfer.
    A single manually supplied window remains supported by the same rule.
    """
    pw=pixel_windows[pixel_windows.status=='OK'].set_index(['y_index','x_index']);rows=[]
    for _,g in gauge_pixels.iterrows():
        key=(int(g.y_index),int(g.x_index));obs=observed_windows[observed_windows.station.astype(str)==str(g.station)]
        base={'station':str(g.station)}
        for col in ('uparea_km2','main_basin','lon','lat','HYBAS_ID'): 
            if col in g:base[col]=g[col]
        if key not in pw.index or obs.empty:
            rows.append(dict(base,status='NO_MODEL_WINDOW' if key not in pw.index else 'NO_OBS_WINDOW'));continue
        m=pw.loc[key]
        if isinstance(m,pd.DataFrame):raise ValueError('Duplicate pixel window keys.')
        candidates=m.get('model_windows',None)
        if not isinstance(candidates,list) or not candidates:
            candidates=[dict(start_doy=int(m.model_start_doy),end_doy=int(m.model_end_doy),
                             duration_days=int(m.model_duration_days),window_id=m.get('model_window_id',None))]
        references=obs.to_dict('records');pairs=match_windows(references,candidates,min_overlap=min_overlap)
        matched={i:(j,score) for i,j,score in pairs}
        for i,o in enumerate(references):
            if i not in matched:
                rows.append(dict(base,status='MODE_UNMATCHED',observed_window_id=o.get('mode_id',o.get('season_id',None))));continue
            j,score=matched[i];c=candidates[j];ms,me=int(c['start_doy']),int(c['end_doy'])
            rec=dict(base,status='OK',obs_start_doy=int(o['start_doy']),obs_end_doy=int(o['end_doy']),
                     model_start_doy=ms,model_end_doy=me,window_overlap=score,
                     observed_window_id=o.get('mode_id',o.get('season_id',None)),
                     model_window_id=c.get('mode_id',c.get('season_id',c.get('window_id',None))),
                     window_family=m.get('window_family','season'),
                     bias_start_days=float(circular_day_difference(o['start_doy'],ms)),
                     bias_end_days=float(circular_day_difference(o['end_doy'],me)),
                     obs_duration_days=int((o['end_doy']-o['start_doy'])%365+1),
                     model_duration_days=int((me-ms)%365+1))
            for label,value in [('obs_start',o['start_doy']),('obs_end',o['end_doy']),('model_start',ms),('model_end',me)]:
                rec[label+'_month_day']=_doy_to_monthday(value)
            rows.append(rec)
    return pd.DataFrame(rows)


# ---------------------------------------------------------------------------
# 3. Regionalise the bias
# ---------------------------------------------------------------------------

def _effective_distance(pix_lon,pix_lat,pix_area,pix_basin,donors,cfg):
    """Geodesic distance with explicit scale/basin admissibility, not routing."""
    d=np.maximum(_haversine_km(pix_lon,pix_lat,donors.lon.to_numpy(),donors.lat.to_numpy()),1e-6)
    if pix_area is not None and np.isfinite(pix_area) and pix_area>0 and 'uparea_km2' in donors:
        a=pd.to_numeric(donors.uparea_km2,errors='coerce').to_numpy(float)
        ratio=np.full(len(a),np.inf);ok=np.isfinite(a)&(a>0)
        ratio[ok]=np.abs(np.log(pix_area/a[ok]))
        d=d*(1+cfg.area_penalty*ratio)
        if cfg.max_area_ratio is not None:d=np.where(ratio<=np.log(cfg.max_area_ratio),d,np.inf)
    if pix_basin is not None and pd.notna(pix_basin):
        same=donors['main_basin'].to_numpy()==pix_basin if 'main_basin' in donors else np.zeros(len(d),bool)
        d=np.where(same,d,np.inf if cfg.same_basin_only else d*cfg.cross_basin_multiplier)
    return d


def propagate_window_bias(pixel_windows,gauge_bias,config=None,pixel_area=None,pixel_basin=None,
                          *,gauge_supports=None,target_supports=None):
    """Shared prediction path used by BOTH application and spatial validation.

    Explicit supports are keyed by station and by ``"iy:ix"`` respectively.
    Without supports, top-kriging refuses to guess a basin. Disc approximation
    requires ``allow_disc_approximation=True`` and is labelled in every output.
    """
    cfg=config or GridWindowConfig()
    if cfg.transfer_method not in {'idw','topkriging'}:raise ValueError('Unknown timing transfer method.')
    if cfg.no_donor_policy not in {'nan','raw'}:raise ValueError('no_donor_policy must be nan or raw.')
    donors=gauge_bias[gauge_bias.status=='OK'].copy() if 'status' in gauge_bias else pd.DataFrame()
    out=pixel_windows.copy();records=[]
    for _,p in out.iterrows():
        if p.get('status')!='OK':
            records.append(dict(bias_start_days=np.nan,bias_end_days=np.nan,transfer_status='NO_MODEL_WINDOW',donor_stations=''));continue
        iy,ix=int(p.get('y_index',0)),int(p.get('x_index',0))
        area=float(pixel_area[iy,ix]) if pixel_area is not None else p.get('uparea_km2',None)
        basin=pixel_basin[iy,ix] if pixel_basin is not None else p.get('main_basin',None)
        target_support=(target_supports or {}).get(f'{iy}:{ix}')
        rec=_predict_timing_target(p,donors,cfg,area,basin,gauge_supports,target_support)
        if rec['transfer_status'].startswith('NO_') and cfg.no_donor_policy=='raw':
            rec.update(bias_start_days=0.,bias_end_days=0.,transfer_status='RAW_NO_DONOR')
        records.append(rec)
    for key in sorted(set().union(*(r.keys() for r in records)) if records else []):
        out[key]=[r.get(key,np.nan) for r in records]
    if out.empty:
        for c in ('bias_start_days','bias_end_days','transfer_status','donor_stations'):out[c]=pd.Series(dtype=object)
    return out


# ---------------------------------------------------------------------------
# 4. Corrected windows
# ---------------------------------------------------------------------------

def corrected_grid_windows(transferred,config=None):
    """Apply audited circular shifts; shared guards in CV and production."""
    cfg=config or GridWindowConfig();out=transferred.copy();rows=[]
    if not 1<=cfg.min_duration_days<=cfg.max_duration_days<=365:raise ValueError('Invalid duration bounds.')
    for _,r in out.iterrows():
        rec=dict(corrected_start_doy=np.nan,corrected_end_doy=np.nan,corrected_duration_days=np.nan,window_flag='NO_WINDOW')
        bs,be=r.get('bias_start_days',np.nan),r.get('bias_end_days',np.nan)
        if r.get('status')=='OK' and np.isfinite(bs) and np.isfinite(be):
            ds=float(np.clip(bs,-cfg.max_shift_days,cfg.max_shift_days));de=float(np.clip(be,-cfg.max_shift_days,cfg.max_shift_days))
            s=wrap_doy(r.model_start_doy+ds);e=wrap_doy(r.model_end_doy+de);dur=(e-s)%365+1
            flag='CLIPPED_SHIFT' if (ds!=bs or de!=be) else 'OK'
            if not cfg.min_duration_days<=dur<=cfg.max_duration_days:
                shift=circular_bias_mean([ds,de],min_resultant=cfg.min_circular_resultant)
                if np.isfinite(shift):
                    s=wrap_doy(r.model_start_doy+shift);e=wrap_doy(r.model_end_doy+shift);dur=(e-s)%365+1;flag='DURATION_GUARD'
                else:dur=0
            if cfg.min_duration_days<=dur<=cfg.max_duration_days:
                rec.update(corrected_start_doy=s,corrected_end_doy=e,corrected_duration_days=dur,window_flag=flag)
            else:rec['window_flag']='INVALID_DURATION'
        rows.append(rec)
    for key in ('corrected_start_doy','corrected_end_doy','corrected_duration_days','window_flag'):
        out[key]=[r[key] for r in rows]
    for side in ('start','end'):
        out[f'corrected_{side}_month_day']=[_doy_to_monthday(int(v)) if pd.notna(v) else '' for v in out[f'corrected_{side}_doy']]
    return out


# ---------------------------------------------------------------------------
# 5. Predictands on the grid
# ---------------------------------------------------------------------------

def compute_grid_highflow_predictands(
    discharge: xr.DataArray,
    windows: pd.DataFrame,
    highflow_config: HighWaterConfig,
    use_corrected: bool = True,
    progress_every: int = 0,
) -> xr.Dataset:
    """Apply per-pixel windows to the gridded discharge.

    Returns a Dataset with ``Q_HW_MEAN`` and ``Q_HW_MAX`` on (T, Y, X), where
    T labels the hydrological year.  The per-pixel anchor from the model fit
    is reused, so cross-calendar-year windows are handled exactly as in the
    gauged workflow.

    Coverage rules from ``highflow_config`` apply unchanged: a pixel-year
    whose window is insufficiently observed yields NaN rather than a value
    computed on partial data.
    """
    t, y, x = _dim_names(discharge)
    times = pd.DatetimeIndex(discharge[t].values)
    lats = np.asarray(discharge[y].values, dtype=float)
    lons = np.asarray(discharge[x].values, dtype=float)
    discharge = discharge.transpose(t,y,x)

    s_col = "corrected_start_doy" if use_corrected else "model_start_doy"
    e_col = "corrected_end_doy" if use_corrected else "model_end_doy"

    usable = windows[
        (windows["status"] == "OK")
        & windows[s_col].notna()
        & windows[e_col].notna()
    ]

    per_pixel: dict[tuple[int, int], pd.DataFrame] = {}
    years: set[int] = set()
    for n, (_, r) in enumerate(usable.iterrows()):
        iy, ix = int(r["y_index"]), int(r["x_index"])
        anchor = int(r["anchor_doy"])
        s_pos = _doy_to_rotpos(int(r[s_col]), anchor)
        e_pos = _doy_to_rotpos(int(r[e_col]), anchor)
        if e_pos < s_pos:
            e_pos += 365  # one occurrence, labelled by the hydrological year of its start
        pseudo = {
            "anchor_doy": anchor,
            "windows": [{
                "season_id": 1,
                "start_pos": s_pos,
                "end_pos": e_pos,
                "duration_days": int(e_pos - s_pos + 1),
                "is_main_season": True,
            }],
            "mode_windows": [],
        }
        s = pd.Series(discharge.isel({y:iy,x:ix}).values, index=times)
        try:
            ann = compute_annual_predictands(s, pseudo, highflow_config)
        except Exception:  # noqa: BLE001
            continue
        per_pixel[(iy, ix)] = ann
        years.update(ann["hydro_year"].dropna().astype(int).tolist())
        if progress_every and (n + 1) % progress_every == 0:
            print(f"  ... {n + 1}/{len(usable)} pixels", flush=True)

    years_sorted = sorted(years)
    if not years_sorted:
        raise RuntimeError("No pixel produced usable high-flow predictands.")

    shape = (len(years_sorted), len(lats), len(lons))
    qmean = np.full(shape, np.nan)
    qmax = np.full(shape, np.nan)
    ypos = {yv: i for i, yv in enumerate(years_sorted)}

    for (iy, ix), ann in per_pixel.items():
        for _, row in ann.iterrows():
            hy = row.get("hydro_year")
            if not np.isfinite(hy) or int(hy) not in ypos:
                continue
            k = ypos[int(hy)]
            qmean[k, iy, ix] = row.get("HW1_Q_MEAN", np.nan)
            qmax[k, iy, ix] = row.get("HW1_Q_MAX", np.nan)

    T = pd.to_datetime([f"{v:04d}-01-01" for v in years_sorted])
    ds = xr.Dataset(
        {
            "Q_HW_MEAN": ((t, y, x), qmean),
            "Q_HW_MAX": ((t, y, x), qmax),
        },
        coords={t: T, y: lats, x: lons},
    )
    anchor_grid = np.full((len(lats),len(lons)),np.nan)
    start_grid = anchor_grid.copy();end_grid=anchor_grid.copy()
    for _,r in usable.iterrows():
        iy,ix=int(r.y_index),int(r.x_index)
        anchor_grid[iy,ix]=r.anchor_doy;start_grid[iy,ix]=r[s_col];end_grid[iy,ix]=r[e_col]
    ds['anchor_doy']=((y,x),anchor_grid)
    ds['window_start_doy']=((y,x),start_grid);ds['window_end_doy']=((y,x),end_grid)
    ds.attrs["description"] = "Gridded high-water discharge predictands"
    ds.attrs["window_source"] = (
        "model window corrected by regionalised gauge timing bias"
        if use_corrected else "uncorrected model window"
    )
    ds.attrs["T_semantics"] = "T labels the hydrological year; anchor is per pixel"
    return ds


# ---------------------------------------------------------------------------
# 6. Validation
# ---------------------------------------------------------------------------

def cross_validate_grid_windows(gauge_bias,config=None,*,gauge_supports=None):
    """Leave-one-STATION-out with the actual application estimator and guards.

    All modes of a withheld station are excluded together. All comparisons use
    the same successful folds. Exclusions remain visible in ``folds``. This
    evaluates spatial timing transfer conditional on the supplied climatologies,
    not a prospective forecast or an independent temporal regime validation.
    """
    cfg=config or GridWindowConfig();g=gauge_bias[gauge_bias.status=='OK'].reset_index(drop=True)
    if g.station.nunique()<3:return dict(folds=pd.DataFrame(),summary=pd.DataFrame(),verdict='INSUFFICIENT_GAUGES')
    rows=[]
    for i,target in g.iterrows():
        donors=g[g.station.astype(str)!=str(target.station)].copy()
        target_p=target.copy();target_p['status']='OK';target_p['y_index']=0;target_p['x_index']=0
        area=target.get('uparea_km2',None);basin=target.get('main_basin',None)
        target_support=(gauge_supports or {}).get(str(target.station))
        info=_predict_timing_target(target_p,donors,cfg,area,basin,gauge_supports,target_support)
        rec=dict(station=target.station,status=info['transfer_status'],transfer_method_used=cfg.transfer_method,
                 obs_start_doy=target.obs_start_doy,obs_end_doy=target.obs_end_doy)
        merged=target_p.to_dict();merged.update(info)
        corrected=corrected_grid_windows(pd.DataFrame([merged]),cfg).iloc[0]
        if not np.isfinite(corrected.corrected_start_doy):rows.append(rec);continue
        # Baseline uses the same admissible/homologous donor set reported by the estimator.
        named=str(info.get('donor_stations','')).split(';');allowed=donors[donors.station.astype(str).isin(named)]
        if allowed.empty:rows.append(rec);continue
        dist=_effective_distance(target.lon,target.lat,area,basin,allowed,cfg)
        nearest=allowed.iloc[int(np.argmin(dist))]
        for label,s,e in [('method',corrected.corrected_start_doy,corrected.corrected_end_doy),
                           ('model_raw',target.model_start_doy,target.model_end_doy),
                           ('nearest_obs',nearest.obs_start_doy,nearest.obs_end_doy)]:
            rec[f'err_start_{label}']=float(circular_day_difference(s,target.obs_start_doy))
            rec[f'err_end_{label}']=float(circular_day_difference(e,target.obs_end_doy))
        rec.update(status='OK',window_flag=corrected.window_flag,nearest_donor=nearest.station,
                   donor_stations=info.get('donor_stations',''),support_kind=info.get('support_kind',''),
                   variogram_start_status=info.get('variogram_start_status',''))
        rows.append(rec)
    folds=pd.DataFrame(rows);valid=folds[folds.status=='OK'];summary=[]
    for label in ('method','model_raw','nearest_obs'):
        if valid.empty:break
        a=valid[f'err_start_{label}'].to_numpy();b=valid[f'err_end_{label}'].to_numpy()
        summary.append(dict(approach=label,mae_start_days=float(np.abs(a).mean()),mae_end_days=float(np.abs(b).mean()),
                            mae_overall_days=float(np.abs(np.r_[a,b]).mean()),bias_start_days=float(a.mean()),bias_end_days=float(b.mean()),
                            n_folds=len(valid),n_excluded=len(folds)-len(valid)))
    table=pd.DataFrame(summary)
    if table.empty:return dict(folds=folds,summary=table,verdict='NO_FOLD')
    table=table.sort_values('mae_overall_days').reset_index(drop=True)
    winner=table.iloc[0].approach
    return dict(folds=folds,summary=table,verdict={'method':'METHOD_BEST','model_raw':'MODEL_RAW_BETTER','nearest_obs':'NEAREST_OBS_BETTER'}[winner],
                validation_scope='spatial transfer conditional on supplied reference climatologies')


# ---------------------------------------------------------------------------
# Orchestrator
# ---------------------------------------------------------------------------

def regionalize_highflow_windows(
    discharge: xr.DataArray,
    observed_windows: pd.DataFrame,
    gauge_pixels: pd.DataFrame,
    highflow_config: HighWaterConfig,
    uparea: Optional[xr.DataArray] = None,
    config: Optional[GridWindowConfig] = None,
    pixel_basin: Optional[np.ndarray] = None,
    reference_start=None,
    reference_end=None,
    compute_predictands: bool = True,
    progress_every: int = 0,
    gauge_supports=None,
    target_supports=None,
    periods=None,
) -> dict[str, Any]:
    """Run the whole window-regionalisation chain.

    Returns the pixel window table at every stage, the gauge timing biases,
    the cross-validation, and optionally the gridded predictands.
    """
    cfg = config or GridWindowConfig()
    if periods is not None:
        periods.validate(pd.DatetimeIndex(discharge[_dim_names(discharge)[0]].values))
        if reference_start is not None or reference_end is not None:
            raise ValueError('Use periods or reference_start/end, not both.')
        reference_start,reference_end=periods.climatology
    if isinstance(pixel_basin,xr.DataArray):
        _,yy,xx=_dim_names(discharge)
        pixel_basin=align_spatial_grid(pixel_basin,discharge,yy,xx).values
    mask, area = build_river_mask(discharge, uparea, cfg)
    pixel_windows = fit_gridded_local_windows(
        discharge, mask, highflow_config, cfg,
        reference_start=reference_start, reference_end=reference_end,
        progress_every=progress_every,
    )

    gp = gauge_pixels.copy()
    if area is not None and "uparea_km2" not in gp.columns:
        gp["uparea_km2"] = [
            float(area[int(r["y_index"]), int(r["x_index"])]) for _, r in gp.iterrows()
        ]

    bias = gauge_window_bias(observed_windows, pixel_windows, gp, min_overlap=cfg.min_mode_overlap)
    transferred = propagate_window_bias(
        pixel_windows, bias, cfg, pixel_area=area, pixel_basin=pixel_basin, gauge_supports=gauge_supports, target_supports=target_supports
    )
    corrected = corrected_grid_windows(transferred, cfg)
    cv = cross_validate_grid_windows(bias, cfg, gauge_supports=gauge_supports)

    out = {
        "river_mask": mask,
        "pixel_area_km2": area,
        "pixel_windows": corrected,
        "gauge_bias": bias,
        "cross_validation": cv,
        "config": cfg,
    }
    if periods is not None:
        out['periods']=periods.describe(pd.DatetimeIndex(discharge[_dim_names(discharge)[0]].values))
        tdim=_dim_names(discharge)[0]
        a,b=periods.application
        if a is not None or b is not None: discharge=discharge.sel({tdim:slice(a,b)})
    if compute_predictands:
        out["predictands"] = compute_grid_highflow_predictands(
            discharge, corrected, highflow_config,
            use_corrected=True, progress_every=progress_every,
        )
    return out


def _propagate_window_bias_topkriging(pixel_windows,donors,cfg,pixel_area=None,**kwargs):
    """Compatibility entry point; all paths now use the same estimator."""
    from dataclasses import replace
    return propagate_window_bias(pixel_windows,donors,replace(cfg,transfer_method='topkriging'),pixel_area,**kwargs)


def _predict_timing_target(target, donors, cfg, area=None, basin=None,
                          gauge_supports=None, target_support=None):
    missing=dict(bias_start_days=np.nan,bias_end_days=np.nan,transfer_status='NO_DONOR',donor_stations='')
    if donors.empty:return missing
    d=donors.copy()
    d=d[np.isfinite(pd.to_numeric(d.bias_start_days,errors='coerce')) & np.isfinite(pd.to_numeric(d.bias_end_days,errors='coerce'))]
    if d.empty:return missing
    # Cross-station MODE numbers are not physical labels. Restrict each donor
    # to the window most overlapping the target's local MODEL season.
    if {'model_start_doy','model_end_doy'}.issubset(target.index) and {'model_start_doy','model_end_doy'}.issubset(d):
        A=window_day_set(dict(start_doy=target.model_start_doy,end_doy=target.model_end_doy))
        overlaps=[]
        for _,r in d.iterrows():
            B=window_day_set(dict(start_doy=r.model_start_doy,end_doy=r.model_end_doy))
            overlaps.append(len(A&B)/len(A|B))
        d['_homology']=overlaps;d=d[d._homology>=cfg.min_mode_overlap]
        d=d.sort_values('_homology',ascending=False).drop_duplicates('station')
    d=d.reset_index(drop=True)
    if d.empty:return dict(missing,transfer_status='NO_HOMOLOGOUS_DONOR')
    distance=_effective_distance(target.lon,target.lat,area,basin,d,cfg)
    order=[int(i) for i in np.argsort(distance) if np.isfinite(distance[i])][:max(1,cfg.k_donors)]
    if not order:return dict(missing,transfer_status='NO_ADMISSIBLE_DONOR')
    chosen=d.iloc[order].reset_index(drop=True);distance=distance[order]
    if cfg.transfer_method=='idw':
        weights=1/np.maximum(distance,1e-9)**cfg.idw_power
        bs=circular_bias_mean(chosen.bias_start_days,weights,min_resultant=cfg.min_circular_resultant)
        be=circular_bias_mean(chosen.bias_end_days,weights,min_resultant=cfg.min_circular_resultant)
        if not np.isfinite(bs+be):return dict(missing,transfer_status='NO_UNAMBIGUOUS_CIRCULAR_MEAN')
        return dict(bias_start_days=bs,bias_end_days=be,transfer_status='TRANSFERRED_IDW',
                    donor_stations=';'.join(chosen.station.astype(str)),n_donors=len(chosen),
                    support_kind='outlet_points',transfer_method_used='idw')
    tk=cfg.topkriging_config or TopKrigingConfig(max_donors=cfg.k_donors)
    if target_support is not None and gauge_supports is not None:
        selected=[i for i,n in enumerate(chosen.station.astype(str)) if n in gauge_supports]
        chosen=chosen.iloc[selected].reset_index(drop=True)
        if chosen.empty:return dict(missing,transfer_status='NO_DONOR_SUPPORT')
        sup=[gauge_supports[n] for n in chosen.station.astype(str)];tsup=target_support
    elif cfg.allow_disc_approximation:
        if area is None or not np.isfinite(area) or area<=0 or 'uparea_km2' not in chosen:
            raise ValueError('Disc approximation requires target and donor upstream areas.')
        names=chosen.station.astype(str).tolist();key='__TARGET__'
        if key in names:raise ValueError('Reserved support name __TARGET__.')
        supports=build_supports_from_points(names+[key],list(chosen.lon)+[float(target.lon)],
                    list(chosen.lat)+[float(target.lat)],list(chosen.uparea_km2)+[area],tk)
        sup=[supports[n] for n in names];tsup=supports[key]
    else:
        raise ValueError('Top-kriging needs actual gauge_supports and target_supports. Set allow_disc_approximation=True only for an explicitly labelled approximation.')
    try:
        bs=unwrap_biases(chosen.bias_start_days,min_resultant=cfg.min_circular_resultant)
        be=unwrap_biases(chosen.bias_end_days,min_resultant=cfg.min_circular_resultant)
    except ValueError:return dict(missing,transfer_status='NO_UNAMBIGUOUS_CIRCULAR_MEAN')
    vgs=fit_point_variogram(sup,bs,tk);vge=fit_point_variogram(sup,be,tk)
    ps=topkriging_predict(tsup,sup,bs,vgs,tk);pe=topkriging_predict(tsup,sup,be,vge,tk)
    return dict(bias_start_days=(ps['value']+182.5)%365-182.5,bias_end_days=(pe['value']+182.5)%365-182.5,
                transfer_status='TRANSFERRED_TOPKRIGING',donor_stations=ps['donors'],n_donors=ps['n_donors'],
                variance_start_days2=ps['variance'],variance_end_days2=pe['variance'],
                variogram_start_status=vgs['status'],variogram_end_status=vge['status'],
                support_kind=tsup.support_kind,transfer_method_used='topkriging')
