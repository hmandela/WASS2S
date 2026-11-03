"""Seasonal high-water predictand construction, with or without observations.

This is the v5 MAIN workflow. The older daily bias-correction workflow remains
available for compatibility. The OPTIONAL monthly_ratio branch added to v5.1.0
uses a separate monthly fitter BEFORE seasonal aggregation; default is unchanged. The order is:
  reference windows -> outlet extraction -> yearly window aggregates -> bias
  correction of those aggregates -> regional predictand.

The ``glofas`` branch never needs an observation DataFrame and never pretends to
perform observation-based calibration. The ``auto`` branch switches to it only
when no observation object was supplied; per-outlet fallback is separately
controlled and recorded. Climatologies and fitted correction parameters are
static over the requested application period.
"""
from __future__ import annotations
from dataclasses import dataclass, replace
from typing import Any
import copy
import logging
import numpy as np
import pandas as pd
import xarray as xr

if __package__:
    from .highflow_outlet_aggregation import (AggregationConfig, OutletAggregation, as_field,
        outlet_table, extract_at_outlets, select_catalog, aggregate_series, aggregate_field,
        format_aggregation)
    from .highwater_season_predictands_modeaware import (HighWaterConfig,
        fit_station_highwater_regime, bootstrap_window_uncertainty)
    from .highflow_periods import ClimatologyPeriod
    from .highflow_utils import (align_spatial_grid, clean_daily_series, match_windows,
        window_dates, wrap_doy, doy_to_md, circular_bias_mean)
    from .glofas_highflow_bias_correction import (GloFASPixelMatchConfig,
        associate_stations_to_hybas, match_hybas_outlets_to_glofas,
        validate_glofas_auxiliary, build_hybas_digraph, _directed_relation_and_distance)
else:
    from highflow_outlet_aggregation import (AggregationConfig, OutletAggregation, as_field,
        outlet_table, extract_at_outlets, select_catalog, aggregate_series, aggregate_field,
        format_aggregation)
    from highwater_season_predictands_modeaware import (HighWaterConfig,
        fit_station_highwater_regime, bootstrap_window_uncertainty)
    from highflow_periods import ClimatologyPeriod
    from highflow_utils import (align_spatial_grid, clean_daily_series, match_windows,
        window_dates, wrap_doy, doy_to_md, circular_bias_mean)
    from glofas_highflow_bias_correction import (GloFASPixelMatchConfig,
        associate_stations_to_hybas, match_hybas_outlets_to_glofas,
        validate_glofas_auxiliary, build_hybas_digraph, _directed_relation_and_distance)

LOG=logging.getLogger(__name__)


@dataclass(frozen=True)
class PredictandConfig:
    """Configuration of the seasonal-aggregate workflow, not daily correction.

    reference_source:
      observations: observed calendars transferred to admissible HYBAS outlets;
      glofas: each outlet has its own GloFAS climatology, no OBS calibration;
      auto: observations when supplied, otherwise the complete glofas branch.

    fallback='glofas' is an EXPLICIT per-outlet opt-in. With the default 'nan',
    an outlet having no admissible observed calendar remains missing. Never use
    an unrelated global-average season/factor as an invisible fallback.
    """
    reference_source: str = 'auto'
    kind: str = 'season'                      # season, mode
    selector: int | str = 'main'              # main, all, each, 1, 2, ...
    correction: str = 'auto'                  # + monthly_ratio: DAILY correction before HW aggregation
    monthly_bias: Any = None                  # MonthlyBiasConfig, only for correction='monthly_ratio'
    min_training_years: int = 10
    min_q: float = 0.01                       # log_ratio only; mean_ratio keeps zeros
    eqm_n_quantiles: int = 31
    fallback: str = 'nan'                     # nan, glofas, error
    uncorrected_policy: str = 'nan'           # nan, raw, error (when OBS fit unavailable)
    window_transfer: str = 'donor'            # donor, circular
    propagation_method: str = 'hybrid'        # network, hybrid, idw_log, topkriging
    k_donors: int = 3
    distance_power: float = 1.0
    max_donor_distance_km: float | None = 500.
    max_area_ratio: float | None = 6.
    area_penalty: float = .6
    same_main_basin: bool = True
    allow_lateral_donors: bool = False
    min_window_overlap: float = .20
    topkriging_config: Any = None
    # The proxy is visibly flagged. Set False to require actual gauge area.
    allow_station_area_proxy: bool = True
    # Target cleaning occurs before matching, detection and aggregation.
    duplicate_outlet_policy: str = 'best_glofas'  # best_glofas, separate, error
    coastal_policy: str = 'exclude'             # exclude COAST=1, or keep
    missing_coast_policy: str = 'error'          # error, keep (warning + audit)
    multi_no_match_policy: str = 'drop'          # drop group, or error; never first

    def validate(self):
        if self.duplicate_outlet_policy not in {'best_glofas', 'separate', 'error'}:
            raise ValueError('duplicate_outlet_policy: best_glofas, separate or error')
        if self.coastal_policy not in {'exclude', 'keep'}:
            raise ValueError('coastal_policy: exclude or keep')
        if self.missing_coast_policy not in {'error', 'keep'}:
            raise ValueError('missing_coast_policy: error or keep')
        if self.multi_no_match_policy not in {'drop', 'error'}:
            raise ValueError('multi_no_match_policy: drop or error')
        if self.reference_source not in {'auto','observations','glofas'}:raise ValueError('Invalid reference_source')
        if self.correction not in {'auto','none','mean_ratio','log_ratio','eqm','monthly_ratio'}:raise ValueError('Invalid correction')
        if self.monthly_bias is not None and self.correction != 'monthly_ratio':
            raise ValueError('monthly_bias is only used with correction="monthly_ratio"')
        if self.fallback not in {'nan','glofas','error'}:raise ValueError('Invalid fallback')
        if self.uncorrected_policy not in {'nan','raw','error'}:raise ValueError('Invalid uncorrected_policy')
        if self.window_transfer not in {'donor','circular'}:raise ValueError('Invalid window_transfer')
        if self.propagation_method not in {'network','hybrid','idw_log','topkriging'}:raise ValueError('Invalid propagation_method')
        if self.min_training_years<2:raise ValueError('min_training_years must be >= 2')
        if self.k_donors<1 or self.distance_power<=0:raise ValueError('Invalid donor weighting controls')
        if self.max_area_ratio is not None and self.max_area_ratio<1:raise ValueError('max_area_ratio must be >= 1')
        if self.max_donor_distance_km is not None and self.max_donor_distance_km<=0:raise ValueError('Invalid donor distance')
        if not 0<self.min_window_overlap<=1:raise ValueError('min_window_overlap must be in (0,1]')
        if self.min_q<=0:raise ValueError('min_q must be > 0')
        select_catalog(pd.DataFrame(),kind=self.kind,selector=self.selector)


def _empty_catalog():
    return pd.DataFrame(columns=['outlet_id','kind','window_id','number','is_main','anchor_doy',
        'start_doy','end_doy','start_month_day','end_month_day','window_source','source_station'])


def catalog_from_regime(regime, identifier, source, source_station=''):
    """Represent native HW/MODE definitions without losing their distinction."""
    rows=[]
    for kind,key,idkey,mainkey,prefix in [('season','windows','season_id','is_main_season','HW'),
                                         ('mode','mode_windows','mode_id','is_main_mode','MODE')]:
        for w in regime.get(key,[]):
            if w.get('weak_season_fallback',False):continue
            r=dict(w)
            r.update(outlet_id=str(identifier),kind=kind,window_id=f'{prefix}{int(w[idkey])}',
                     number=int(w[idkey]),is_main=bool(w.get(mainkey,False)),
                     anchor_doy=int(regime['anchor_doy']),window_source=source,
                     source_station=str(source_station),source_window_id=f'{prefix}{int(w[idkey])}')
            rows.append(r)
    return pd.DataFrame(rows) if rows else _empty_catalog()


def _slice(q, period):
    a,b=period
    return q.loc[a:b] if (a is not None or b is not None) else q


def _greatcircle(a,b):
    la1,ph1,la2,ph2=np.deg2rad([float(a.X),float(a.Y),float(b.X),float(b.Y)])
    h=np.sin((ph2-ph1)/2)**2+np.cos(ph1)*np.cos(ph2)*np.sin((la2-la1)/2)**2
    return float(6371.0088*2*np.arcsin(np.sqrt(np.clip(h,0,1))))


def _candidates(target, metadata, cfg, graph, relation_cache):
    """Rank eligible station donors without using any application-year data."""
    rows=[]
    for _,d in metadata.iterrows():
        same=int(d.HYBAS_ID)==int(target.HYBAS_ID)
        if cfg.same_main_basin and pd.notna(target.get('MAIN_BAS')) and pd.notna(d.get('MAIN_BAS')):
            if int(target.MAIN_BAS)!=int(d.MAIN_BAS):continue
        da=float(d.get('MATCH_UP_AREA_KM2',d.get('UP_AREA',np.nan)))
        ta=float(target.get('MATCH_UP_AREA_KM2',target.get('UP_AREA',np.nan)));ratio=np.nan;penalty=0.
        if da>0 and ta>0 and np.isfinite(da+ta):
            ratio=max(da/ta,ta/da)
            if cfg.max_area_ratio is not None and ratio>cfg.max_area_ratio:continue
            penalty=abs(np.log(da/ta))
        relation='same' if same else 'spatial'
        multi = bool(target.get('multiple_outlet_basin',False)) or bool(d.get('multiple_outlet_basin',False))
        local_pixel = (same and np.isfinite(target.get('matched_source_x',np.nan))
                       and target.get('matched_source_x') == d.get('matched_source_x')
                       and target.get('matched_source_y') == d.get('matched_source_y'))
        if multi and cfg.propagation_method in {'network','hybrid','topkriging'}:
            # A coastal aggregate is not a single drainage support/tree node.
            # Only local calibration at the very same mapped pixel is accepted;
            # point-specific topology/supports must be supplied for broader use.
            if not local_pixel:
                continue
            relation,dist = 'same_mapped_pixel',_greatcircle(target,d)
        elif cfg.propagation_method in {'network','hybrid'}:
            key=(int(d.HYBAS_ID),int(target.HYBAS_ID))
            if key not in relation_cache:
                relation_cache[key]=_directed_relation_and_distance(graph,*key)
            relation,dist=relation_cache[key]
            if relation=='disconnected' or (relation=='lateral' and not cfg.allow_lateral_donors):continue
        else:dist=_greatcircle(target,d)
        if not np.isfinite(dist):continue
        if cfg.max_donor_distance_km is not None and dist>cfg.max_donor_distance_km:continue
        # For two gauges associated with the same polygon, actual coordinate
        # distance resolves ties; membership does not imply the same station.
        tie=_greatcircle(target,d)
        effective=max(dist if not same else tie,.05)
        if cfg.propagation_method in {'hybrid','topkriging'}:effective*=1+cfg.area_penalty*penalty
        rows.append(dict(station=str(d.station),HYBAS_ID=int(d.HYBAS_ID),distance_km=float(dist),
                         effective_distance=effective,relation=relation,area_ratio=ratio,same_hybas=same))
    rows.sort(key=lambda r:(not r['same_hybas'],r['effective_distance'],r['station']))
    return rows


def _weights(candidates,cfg):
    d=np.array([r['effective_distance'] for r in candidates],float)
    if len(d)==0:return d
    w=(np.min(d)/d)**cfg.distance_power
    return w/w.sum()


def _copy_or_average_windows(target, candidates, station_catalog, cfg):
    """Transfer a PRIMARY observed regime; optionally average matched bounds.

    Mode-count/chronology come from one identified reference donor, NOT from
    averaging local MODE1 labels. Other donors contribute only after one-to-one
    circular overlap matching to that donor's physical windows.
    """
    if not candidates:return _empty_catalog()
    first=candidates[0]['station']
    base=station_catalog[station_catalog.outlet_id==first].copy()
    base['outlet_id']=str(target.outlet_id)
    base['source_station']=first;base['window_source']='observations_transferred'
    base['window_transfer']='primary_donor'
    base['calendar_donor_count']=1
    if int(candidates[0]['HYBAS_ID'])==int(target.HYBAS_ID):
        base['window_source']=('observations_at_same_mapped_pixel'
            if candidates[0]['relation']=='same_mapped_pixel' else 'observations_at_associated_hybas')
    if cfg.window_transfer=='donor' or len(candidates)==1:return base
    use=candidates[:cfg.k_donors]
    for kind in ('season','mode'):
        indices=base.index[base.kind==kind].tolist()
        refs=base.loc[indices].to_dict('records')
        matches={}
        for c in use:
            candidates_rows=station_catalog[(station_catalog.outlet_id==c['station'])&(station_catalog.kind==kind)]
            records=candidates_rows.to_dict('records')
            matches[c['station']]=(records,{i:j for i,j,_ in match_windows(refs,records,min_overlap=cfg.min_window_overlap)})
        for i,idx in enumerate(indices):
            selected=[];starts=[];lengths=[]
            ref=base.loc[idx]
            for c in use:
                records,mp=matches[c['station']]
                if i not in mp:continue
                w=records[mp[i]];selected.append(c)
                starts.append(float(w['start_doy']))
                lengths.append((int(w['end_doy'])-int(w['start_doy']))%365+1)
            weights=_weights(selected,cfg)
            center=circular_bias_mean(np.asarray(starts)-float(ref.start_doy),weights)
            if not np.isfinite(center):continue
            sd=wrap_doy(float(ref.start_doy)+center)
            duration=max(1,min(365,int(np.rint(np.dot(weights,lengths)))))
            ed=(sd+duration-2)%365+1
            base.loc[idx,['start_doy','end_doy','duration_days']]=[sd,ed,duration]
            base.loc[idx,'start_month_day']=doy_to_md(sd);base.loc[idx,'end_month_day']=doy_to_md(ed)
            base.loc[idx,'window_transfer']='matched_circular_start_and_duration'
            base.loc[idx,'calendar_donor_count']=len(selected)
    # Reject newly overlapping windows within the same family: do not change
    # the number of physical seasons by averaging their edges across donors.
    for kind in ('season','mode'):
        rows=base[base.kind==kind].to_dict('records');seen=set()
        for r in rows:
            days={((int(r['start_doy'])-1+j)%365)+1 for j in range((int(r['end_doy'])-int(r['start_doy']))%365+1)}
            if days&seen:
                clean=_copy_or_average_windows(target,candidates[:1],station_catalog,replace(cfg,window_transfer='donor'))
                clean['window_transfer']='primary_donor_after_overlap_rejection'
                return clean
            seen|=days
    return base


def _fit_model(pairs,cfg,method):
    """One equally weighted annual aggregate per training sample."""
    valid=pairs[np.isfinite(pairs.observed)&np.isfinite(pairs.glofas)].copy()
    if method=='log_ratio':valid=valid[(valid.observed>cfg.min_q)&(valid.glofas>cfg.min_q)]
    if len(valid)<cfg.min_training_years:
        return dict(method=method,status='INSUFFICIENT_TRAINING_YEARS',n_years=len(valid))
    o=valid.observed.to_numpy();g=valid.glofas.to_numpy()
    if np.any(o<0) or np.any(g<0):raise ValueError('Discharge aggregates cannot be negative')
    model=dict(method=method,status='OK',n_years=len(valid),training_years=valid.year.astype(int).tolist())
    if method=='mean_ratio':
        if g.sum()<=0 or o.sum()<=0:return dict(model,status='ZERO_CALIBRATION_MEAN')
        model['factor']=float(o.sum()/g.sum())
    elif method=='log_ratio':model['factor']=float(np.exp(np.mean(np.log(o)-np.log(g))))
    elif method=='eqm':
        ps=np.linspace(0,1,min(cfg.eqm_n_quantiles,len(valid)))
        sx=np.quantile(g,ps);oy=np.quantile(o,ps)
        # Tied model quantiles are collapsed to their mean corresponding OBS
        # quantile. No random jitter and no unbounded tail extrapolation.
        unique,inv=np.unique(sx,return_inverse=True)
        vals=np.bincount(inv,weights=oy)/np.bincount(inv)
        model.update(sim_quantiles=unique,obs_quantiles=vals,tail_policy='clamped empirical range')
    else:raise ValueError('Unsupported correction method')
    return model


def _apply_model(values,model):
    if model['method'] in {'mean_ratio','log_ratio'}:return values*model['factor']
    out=np.interp(values,model['sim_quantiles'],model['obs_quantiles'])
    return np.where(np.isfinite(values),out,np.nan)


def _paired_aggregates(obs,sim,components,anchor,years,acfg,cal_period):
    """Compare OBS and GloFAS on the same valid DAYS inside each fixed window.

    An incomplete accepted season is never scaled to full volume. Whole
    seasonal occurrences outside calibration dates are excluded, not truncated
    and re-labelled as complete seasons.
    """
    idx=obs.index.union(sim.index);o=obs.reindex(idx);g=sim.reindex(idx)
    paired=np.isfinite(o)&np.isfinite(g)
    os=aggregate_series(o.where(paired),components,anchor,years,acfg)
    gs=aggregate_series(g.where(paired),components,anchor,years,acfg)
    out=os.rename(columns={'value':'observed'})
    out['glofas']=gs.value.values
    a,b=cal_period
    in_period=pd.Series(True,index=out.index)
    if a is not None:in_period &= out.start_date>=pd.Timestamp(a)
    if b is not None:in_period &= out.end_date<=pd.Timestamp(b)
    out['in_calibration_period']=in_period
    out['observed_all_years']=out['observed']
    out['glofas_all_years']=out['glofas']
    out.loc[~in_period,['observed','glofas']]=np.nan
    return out


def _make_glofas_mapping(targets,q,up,cfg):
    matches,alts=match_hybas_outlets_to_glofas(targets,up,config=cfg)
    matches['outlet_id']=targets.outlet_id.to_numpy()
    if len(alts):
        alts['outlet_id'] = alts.target_row.map(dict(enumerate(targets.outlet_id.astype(str))))
    mp=pd.DataFrame(dict(outlet_id=matches.outlet_id,source_x=matches.get('glofas_lon',np.nan),
                         source_y=matches.get('glofas_lat',np.nan),
                         status=np.where(matches.match_status.isin(cfg.accepted_match_statuses),'OK',matches.match_status)))
    return matches,alts,mp


def _model_keys_for_components(components,donor_catalog,cfg):
    """Return physically matching donor window IDs, not matching local numbers."""
    if len(components)==0 or donor_catalog.empty:return None
    rows=donor_catalog.to_dict('records')
    matched=match_windows(components,rows,min_overlap=cfg.min_window_overlap)
    if len(matched)!=len(components):return None
    return tuple(sorted(rows[j]['window_id'] for _,j,_ in matched))


def build_highflow_predictand(hydro, *, glofas_discharge, glofas_uparea=None,
        basins=None, outlets=None, hydrosheds=None, hybas_level=None,
        observations=None, observation_variable='Observation', station_attributes=None,
        variable=None, config=None, aggregation_config=None, highflow_config=None,
        match_config=None, periods=None, output_format='both', years=None, glofas_matching=None):
    """Build the seasonal target; observations are genuinely optional.

    No network access: pass ``hydrosheds=hydro.download_hydrosheds(...)`` or
    explicit basins/outlets. Any selected HydroBASINS level is accepted.

    The result includes a ``glofas_raw`` comparator, complete provenance, OBS
    calibration pairs, factors, source pixel alternatives and the common window
    catalogue. In the no-OBS branch, raw and final seasonal values are identical.
    """
    cfg=config or PredictandConfig();cfg.validate()
    matching_context = glofas_matching.context if glofas_matching is not None else None
    acfg=aggregation_config or AggregationConfig(negative_as_missing=True)
    acfg.validate()
    if acfg.time_support!='daily':raise ValueError('Predictand detection expects daily GloFAS; use aggregate_highflow_field for bounded predictor intervals')
    if acfg.statistic not in {'mean','max'}:raise ValueError('Discharge predictands support seasonal mean or maximum')
    # Enforce hydrological QC for the predictand even if a generic aggregation
    # config was passed for use with signed meteorological predictors elsewhere.
    acfg=replace(acfg,negative_as_missing=True)
    hw=copy.deepcopy(highflow_config or HighWaterConfig(n_bootstrap=0))
    hw.mode_specific_outputs=True
    pr=copy.deepcopy(periods or ClimatologyPeriod(require_minimum_years=hw.min_reference_years))
    pr.validate()
    hw.min_reference_years=max(hw.min_reference_years,int(pr.require_minimum_years))
    has_obs=observations is not None and not (isinstance(observations,(dict,pd.DataFrame)) and len(observations)==0)
    source=cfg.reference_source
    if source=='auto':source='observations' if has_obs else 'glofas'
    if source=='observations' and not has_obs:raise ValueError('reference_source="observations" requires observations; use "glofas" or "auto" without them')
    method=cfg.correction
    if method=='auto':method='mean_ratio' if source=='observations' else 'none'
    if source=='glofas' and method!='none':
        raise ValueError('The direct GloFAS branch has no observation-based correction; set correction="none" or "auto"')
    monthly_cfg=None
    if method=='monthly_ratio':
        if __package__:
            from .highflow_monthly_bias import MonthlyBiasConfig, reference_training_mask, correct_monthly_outlets
        else:
            from highflow_monthly_bias import MonthlyBiasConfig, reference_training_mask, correct_monthly_outlets
        monthly_cfg=cfg.monthly_bias or MonthlyBiasConfig()
        if not isinstance(monthly_cfg,MonthlyBiasConfig):raise TypeError('monthly_bias must be MonthlyBiasConfig')
        monthly_cfg.validate()
    def reference_series(series):
        if matching_context is not None:
            series = series.where(matching_context.mask(series.index))
        # Apply GLOBAL outer-fold exclusions before estimating any regime.
        # Dynamic leave-one-month-out exclusions concern only factor fitting.
        if monthly_cfg is not None and monthly_cfg.window_protocol=='refit_excluding':
            return series.where(reference_training_mask(series.index,monthly_cfg))
        return series
    if hydrosheds is not None:
        if basins is not None or outlets is not None:raise ValueError('Pass hydrosheds OR basins/outlets, not both')
        basins=hydrosheds['basins'];outlets=hydrosheds['pour_points']
        declared=hydrosheds.get('metadata',{}).get('level')
        if hybas_level is not None and declared is not None and int(hybas_level)!=int(declared):raise ValueError('Conflicting HYBAS level')
        hybas_level=declared if hybas_level is None else hybas_level
    if basins is None or outlets is None:raise ValueError('Explicit basin polygons and outlet points are required')
    q=as_field(glofas_discharge,variable)
    import re
    q_units=str(q.attrs.get('units','')).strip()
    canonical_units=re.sub(r'[\s_^*]','',q_units.lower()).replace('³','3').replace('−','-').replace('⁻','-').replace('¹','1')
    if q_units and canonical_units not in {'m3s-1','m3/s','m3sec-1','m3second-1','m3second^-1','cumecs'}:
        raise ValueError(f'GloFAS predictand requires discharge units, not {q_units!r}')
    if 'M' in q.dims:raise ValueError('Historical GloFAS predictand must be one daily series per pixel, not an ensemble')
    time=pd.DatetimeIndex(q["T"].values)
    if not time.equals(time.normalize()):raise ValueError('GloFAS input must be daily day-labelled data')
    if glofas_uparea is None and isinstance(glofas_discharge,xr.Dataset):
        key=next((k for k in ('UPAREA','UPAREA_KM2','uparea') if k in glofas_discharge),None)
        if key:glofas_uparea=glofas_discharge[key]
    if glofas_uparea is None:raise ValueError('Supply glofas_uparea; closest geographic pixel is not used as a silent fallback')
    if isinstance(glofas_uparea,xr.Dataset):
        key=next((k for k in ('UPAREA','UPAREA_KM2','uparea') if k in glofas_uparea),None)
        if key is None and len(glofas_uparea.data_vars)==1:key=next(iter(glofas_uparea.data_vars))
        if key is None:raise ValueError('Select the upstream-area DataArray explicitly')
        up=glofas_uparea[key].assign_attrs(dict(glofas_uparea.attrs,**glofas_uparea[key].attrs))
    else:up=glofas_uparea
    version=validate_glofas_auxiliary(q,up)
    up=align_spatial_grid(up,q)
    mc=match_config or (glofas_matching.physical_config if glofas_matching is not None else GloFASPixelMatchConfig())
    if __package__:
        from .highflow_outlet_selection import prepare_prediction_outlets, _canonical_columns
    else:
        from highflow_outlet_selection import prepare_prediction_outlets, _canonical_columns
    # Keep the entire supplied polygon topology. Only prediction targets are
    # filtered. This helper also retains all rejected candidate-point matches.
    basins=_canonical_columns(basins)  # copy; preserve every polygon/network node
    if glofas_matching is None:
        prepared=prepare_prediction_outlets(outlets,basins,up,config=cfg,
            match_config=mc,level=hybas_level)
    else:
        if __package__:
            from .glofas_matching_integration import matching_for_application
        else:
            from glofas_matching_integration import matching_for_application
        glofas_matching = matching_for_application(glofas_matching,q,up,basins,outlets,
            cfg,mc,station_attributes,monthly_cfg)
        prepared=glofas_matching.prepared_outlets

    tab=prepared.outlets
    matches,alternatives,pixelmap=prepared.matches,prepared.alternatives,prepared.pixel_mapping
    import geopandas as gpd
    outlet_geometry=gpd.GeoDataFrame(tab[['HYBAS_ID','outlet_id','multiple_outlet_basin','outlet_uparea_km2']].copy(),
        geometry=gpd.points_from_xy(tab.X,tab.Y),crs=4326)
    daily,extract_audit=extract_at_outlets(q,tab,mapping=pixelmap,max_distance_km=None)
    # Retain mapped pixels for unambiguous local donor identification. A shared
    # coastal polygon alone does not establish flow connectivity.
    tab['matched_source_x'] = matches.glofas_lon.to_numpy()
    tab['matched_source_y'] = matches.glofas_lat.to_numpy()
    qname=str(q.name or 'Q')
    # Fixed reference and application periods are separate. The raw cube is
    # retained unchanged; masking for calibration never edits the input.
    ref_start,ref_end=pr.climatology
    local_regimes={};station_regimes={};boot=[];failures=[]
    local_cache={};obs_series={};station_sim={};station_meta=pd.DataFrame();input_station_metadata=pd.DataFrame()
    station_cat=_empty_catalog();station_matches=pd.DataFrame();station_alts=pd.DataFrame()
    graph=None;relation_cache={};candidate_cache={}
    monthly_station_meta=pd.DataFrame();precorrected=None
    if source=='observations':
        if isinstance(observations,dict):
            st=copy.deepcopy(observations)
            if any(not isinstance(v,dict) or 'series' not in v for v in st.values()):
                raise ValueError('Observation dict: station -> {series, X, Y, ELEV(optional)}')
        else:
            st=hydro._prepare_highflow_observations(observations,variable=observation_variable,
                missing_values=hw.missing_values,negative_flow_as_nan=True)
        if monthly_cfg is not None:
            excluded=set(map(str,monthly_cfg.excluded_stations))
            unknown=excluded-set(map(str,st))
            if unknown:raise ValueError(f'Unknown excluded station names: {sorted(unknown)}')
            st={str(name):info for name,info in st.items() if str(name) not in excluded}
            if not st:raise ValueError('No observation stations remain after monthly excluded_stations')
        if matching_context is not None:
            st={str(name):info for name,info in st.items()
                if str(name) not in set(map(str,matching_context.excluded_stations))}
            if not st:raise ValueError('No observation station remains in the shared matching/training context')
        meta=[];cat=[]
        for name,info in st.items():
            name=str(name)
            obs_series[name]=clean_daily_series(info['series'],missing_values=hw.missing_values)
            meta.append(dict(station=name,X=info.get('X_ORIG',info.get('X',np.nan)),
                             Y=info.get('Y_ORIG',info.get('Y',np.nan)),ELEV=info.get('ELEV',np.nan)))
            try:
                reg=fit_station_highwater_regime(reference_series(obs_series[name]),hw,ref_start,ref_end)
                station_regimes[name]=reg
                cat.append(catalog_from_regime(reg,name,'observations',name))
                if hw.n_bootstrap:
                    for w in bootstrap_window_uncertainty(reference_series(obs_series[name]),reg,hw):
                        boot.append(dict(w,station=name,reference_source='observations'))
            except ValueError as exc:
                failures.append(dict(entity=name,stage='observed_climatology',status='DETECTION_FAILED',message=str(exc)))
        station_cat=pd.concat(cat,ignore_index=True) if cat else _empty_catalog()
        meta=pd.DataFrame(meta)
        input_station_metadata=meta.copy()
        if station_attributes is not None:
            attrs=station_attributes.copy();attrs['station']=attrs.station.astype(str)
            if attrs.station.duplicated().any():raise ValueError('Duplicate station_attributes rows')
            # Physical metadata come from observations; only additional mapping
            # attributes are accepted here, avoiding unexplained coordinate edits.
            meta=meta.merge(attrs.drop(columns=[c for c in ('X','Y','ELEV') if c in attrs]),on='station',how='left',validate='one_to_one')
        if glofas_matching is None:
            station_meta=associate_stations_to_hybas(meta,basins,hybas_outlets=outlet_geometry,config=mc)
            station_meta['outlet_id']=station_meta.station.astype(str)
            if not cfg.allow_station_area_proxy:
                station_meta.loc[station_meta.area_source!='station_area','station_hybas_status']='STATION_AREA_REQUIRED'
            station_matches,station_alts,sm=_make_glofas_mapping(station_meta,q,up,mc)
        else:
            names=meta.station.astype(str).tolist()
            known=glofas_matching.station_metadata.set_index('station',drop=False)
            if not set(names).issubset(known.index):
                raise ValueError('Observations contain stations absent from the frozen mapping')
            station_meta=known.loc[names].reset_index(drop=True).copy()
            # No new gauge position is silently accepted when reusing a mapping.
            verify=meta.set_index('station')
            for _,r in station_meta.iterrows():
                if r.X!=verify.loc[r.station,'X'] or r.Y!=verify.loc[r.station,'Y']:
                    raise ValueError('Gauge coordinates changed after matching')
            station_matches=glofas_matching.station_matches.set_index('station',drop=False).loc[names].reset_index(drop=True).copy()
            station_alts=glofas_matching.station_candidates.copy()
            if len(station_alts):station_alts=station_alts[station_alts.entity.isin(['station:'+x for x in names])].copy()
            sm=pd.DataFrame(dict(outlet_id=station_matches.outlet_id,
                source_x=station_matches.glofas_lon,source_y=station_matches.glofas_lat,
                status=np.where(station_matches.match_status.isin(mc.accepted_match_statuses),'OK',station_matches.match_status)))
        station_meta['matched_source_x'] = station_matches.glofas_lon.to_numpy()
        station_meta['matched_source_y'] = station_matches.glofas_lat.to_numpy()
        station_extract=station_meta[['outlet_id','X','Y']].copy()
        st_daily,_=extract_at_outlets(q,station_extract,mapping=sm,max_distance_km=None)
        for name in station_meta.station:
            station_sim[str(name)]=clean_daily_series(pd.Series(st_daily.sel(outlet=str(name)).values,index=time),missing_values=hw.missing_values)
        selected_st,_=select_catalog(station_cat,kind=cfg.kind,selector=cfg.selector)
        valid_st=set(selected_st.outlet_id.astype(str))
        valid_match=set(station_matches.loc[station_matches.match_status.isin(mc.accepted_match_statuses),'station'].astype(str))
        # Monthly-factor eligibility depends on paired data and geometry, NOT
        # on whether full-record observed regime detection succeeded.
        monthly_station_meta=station_meta[station_meta.station.isin(valid_match)].copy()
        station_meta=station_meta[station_meta.station.isin(valid_st&valid_match)].copy()
        if cfg.propagation_method in {'network','hybrid'}:
            graph=build_hybas_digraph(basins)

    learning_obs = ({name:series.where(matching_context.mask(series.index))
                     for name,series in obs_series.items()} if matching_context is not None else obs_series)
    if monthly_cfg is not None and monthly_cfg.window_source=='corrected_glofas':
        precorrected=correct_monthly_outlets(daily,learning_obs,station_sim,monthly_station_meta,tab,
            config=monthly_cfg,propagation_config=cfg,calibration_period=pr.calibration,
            basins=basins,graph=graph)

    def detect_local(oid):
        if oid not in local_cache:
            try:
                local_data=precorrected['corrected'] if precorrected is not None else daily
                series=reference_series(pd.Series(local_data.sel(outlet=oid).values,index=time))
                reg=fit_station_highwater_regime(series,hw,ref_start,ref_end)
                local_regimes[oid]=reg
                local_cache[oid]=catalog_from_regime(reg,oid,
                    'glofas_monthly_corrected' if precorrected is not None else 'glofas_direct')
                if hw.n_bootstrap:
                    for w in bootstrap_window_uncertainty(series,reg,hw):
                        boot.append(dict(w,outlet_id=oid,reference_source='glofas'))
            except ValueError as exc:
                local_cache[oid]=_empty_catalog()
                failures.append(dict(entity=oid,stage='glofas_climatology',status='DETECTION_FAILED',message=str(exc)))
        return local_cache[oid].copy()

    cats=[];provenance=[]
    for _,target in tab.iterrows():
        oid=str(target.outlet_id);match_status=pixelmap.set_index('outlet_id').loc[oid,'status']
        if match_status!='OK':
            provenance.append(dict(outlet_id=oid,window_source='none',window_status=str(match_status)))
            continue
        if source=='glofas' or precorrected is not None:c=detect_local(oid)
        else:
            candidates=_candidates(target,station_meta,cfg,graph,relation_cache)
            candidate_cache[oid]=candidates
            c=_copy_or_average_windows(target,candidates,station_cat,cfg)
            chosen,_=select_catalog(c,kind=cfg.kind,selector=cfg.selector)
            if chosen.empty:
                if cfg.fallback=='error':raise ValueError(f'No observed reference window for outlet {oid}')
                if cfg.fallback=='glofas':
                    c=detect_local(oid);c['window_source']='glofas_fallback'
        chosen,_=select_catalog(c,kind=cfg.kind,selector=cfg.selector)
        status='OK' if len(chosen) else 'NO_IDENTIFIABLE_SELECTED_WINDOW'
        provenance.append(dict(outlet_id=oid,window_source=str(c.window_source.iloc[0]) if len(c) else 'none',
                               window_status=status,source_station=str(c.source_station.iloc[0]) if len(c) else ''))
        if len(c):cats.append(c)
    catalog=pd.concat(cats,ignore_index=True) if cats else _empty_catalog()
    application=q.sel(T=slice(*pr.application))
    raw=aggregate_field(application,tab,catalog,config=acfg,mapping=pixelmap,
                        kind=cfg.kind,selector=cfg.selector,years=years,max_distance_km=None)
    if monthly_cfg is not None:
        return _finish_monthly_predictand(daily=daily,raw=raw,obs_series=obs_series,
            station_sim=station_sim,monthly_station_meta=monthly_station_meta,tab=tab,
            cfg=cfg,monthly_cfg=monthly_cfg,pr=pr,hw=hw,basins=basins,graph=graph,
            prepared=prepared,catalog=catalog,provenance=provenance,station_cat=station_cat,
            station_regimes=station_regimes,local_regimes=local_regimes,boot=boot,failures=failures,
            station_matches=station_matches,station_alts=station_alts,extract_audit=extract_audit,
            version=version,precorrected=precorrected,output_format=output_format,
            learning_obs=learning_obs, glofas_matching=glofas_matching)
    final=copy.copy(raw);final.values=raw.values.copy(deep=True);final.diagnostics=raw.diagnostics.copy()
    # Index available calibration years widely enough to include cross-year
    # windows. The expected dates decide actual eligibility, not year labels.
    selected,target_names=select_catalog(catalog,kind=cfg.kind,selector=cfg.selector)
    models={};pair_rows=[];factor_rows=[];donor_audit=[]
    observed_annual=[]
    def donor_model(name,keys):
        cachekey=(name,tuple(keys))
        if cachekey not in models:
            dc=station_cat[(station_cat.outlet_id==name)&station_cat.window_id.isin(keys)]
            if len(dc)!=len(keys):return None
            o=learning_obs[name];g=station_sim[name]
            years_fit=list(range(min(o.index.min().year,g.index.min().year)-1,max(o.index.max().year,g.index.max().year)+1))
            pair=_paired_aggregates(o,g,dc.to_dict('records'),int(dc.anchor_doy.iloc[0]),years_fit,acfg,pr.calibration)
            pair['station']=name;pair['window_key']='+'.join(keys)
            pair_rows.extend(pair.to_dict('records'))
            models[cachekey]=_fit_model(pair,cfg,method if method!='none' else 'mean_ratio')
        return models[cachekey]

    # OBS annual values, including validation years, remain separate from the
    # restricted, common-day calibration pairs. Never overwrite OBS with GloFAS.
    if source=='observations':
        ss,stnames=select_catalog(station_cat,kind=cfg.kind,selector=cfg.selector)
        for (name,tname),cc in ss.groupby(['outlet_id','target_id'],sort=False):
            qobs=_slice(obs_series[name],pr.application)
            yy=list(range(qobs.index.min().year-1,qobs.index.max().year+1)) if len(qobs) else []
            if not yy:continue
            stats=aggregate_series(qobs,cc.to_dict('records'),int(cc.anchor_doy.iloc[0]),yy,acfg)
            stats['station']=name;stats['target_id']=tname
            observed_annual.extend(stats.to_dict('records'))

    # Materialize all requested station fits, not just the first few donors
    # eventually selected for one target. This makes evaluation complete.
    if source=='observations':
        requested,_=select_catalog(station_cat,kind=cfg.kind,selector=cfg.selector)
        for (name,_),cc in requested.groupby(['outlet_id','target_id'],sort=False):
            if name in station_sim:
                donor_model(name,tuple(sorted(cc.window_id)))
    prov=pd.DataFrame(provenance).set_index('outlet_id')
    for oi,target in tab.iterrows():
        oid=str(target.outlet_id)
        for ti,tname in enumerate(target_names):
            cc=selected[(selected.outlet_id==oid)&(selected.target_id==tname)]
            origin=prov.loc[oid,'window_source']
            row=dict(outlet_id=oid,HYBAS_ID=int(target.HYBAS_ID),target_id=tname,
                     window_source=origin,method=method,correction_scale='seasonal_aggregate',
                     factor=np.nan,correction_status='NO_WINDOW',n_donors=0)
            if cc.empty:factor_rows.append(row);continue
            if method=='none' or str(origin).startswith('glofas'):
                row.update(method='none',factor=1.,correction_status='NOT_APPLIED_GLOFAS_REFERENCE' if str(origin).startswith('glofas') else 'DISABLED_BY_USER')
                factor_rows.append(row);continue
            candidate_list=candidate_cache.get(oid,[]);usable=[];fitted=[]
            for can in candidate_list:
                name=can['station']
                dc=station_cat[(station_cat.outlet_id==name)&(station_cat.kind==cfg.kind)]
                keys=_model_keys_for_components(cc.to_dict('records'),dc,cfg)
                if keys is None:continue
                model=donor_model(name,keys)
                if model is None or model['status']!='OK':continue
                if method=='eqm':
                    # EQM is used only at the SAME mapped pixel with identical
                    # calendar bounds. It is not turned into a scalar donor factor.
                    smrow=station_matches.set_index('station').loc[name]
                    omrow=matches.set_index('outlet_id').loc[oid]
                    identical=(smrow.glofas_lon==omrow.glofas_lon and smrow.glofas_lat==omrow.glofas_lat)
                    boundaries={(int(r.start_doy),int(r.end_doy)) for _,r in cc.iterrows()}
                    dsel=dc[dc.window_id.isin(keys)]
                    identical=identical and boundaries=={(int(r.start_doy),int(r.end_doy)) for _,r in dsel.iterrows()}
                    if not identical:continue
                usable.append(can);fitted.append(model)
                if len(usable)>=cfg.k_donors:break
            if not usable:
                row['correction_status']='NO_CALIBRATED_DONOR' if method!='eqm' else 'EQM_NOT_TRANSFERABLE_OR_INSUFFICIENT_DATA'
                if cfg.uncorrected_policy=='error':raise ValueError(f'{oid}/{tname}: {row["correction_status"]}')
                if cfg.uncorrected_policy=='raw':row['correction_status']+='__RAW_RETAINED'
                else:final.values.values[:,oi,ti]=np.nan
                factor_rows.append(row);continue
            if method=='eqm':
                final.values.values[:,oi,ti]=_apply_model(raw.values.values[:,oi,ti],fitted[0])
                row.update(correction_status='LOCAL_EQM',n_donors=1,donor_stations=usable[0]['station'],
                           n_training_years=fitted[0]['n_years'])
            else:
                weights=_weights(usable,cfg);z=np.log([m['factor'] for m in fitted])
                lf=float(np.dot(weights,z));variance=np.nan
                if cfg.propagation_method=='topkriging' and len(usable)>1 and not all(c['relation']=='same_mapped_pixel' for c in usable):
                    if __package__:
                        from .highflow_topkriging import TopKrigingConfig,build_supports_from_hybas,fit_point_variogram,topkriging_predict
                    else:
                        from highflow_topkriging import TopKrigingConfig,build_supports_from_hybas,fit_point_variogram,topkriging_predict
                    tk=cfg.topkriging_config or TopKrigingConfig()
                    names=[d['station'] for d in usable]+['TARGET_'+oid]
                    ids=[d['HYBAS_ID'] for d in usable]+[int(target.HYBAS_ID)]
                    try:
                        supports=build_supports_from_hybas(ids,basins,names=names,config=tk)
                        donor_supports=[supports[n] for n in names[:-1]]
                        vg=fit_point_variogram(donor_supports,z,tk)
                        pred=topkriging_predict(supports[names[-1]],donor_supports,z,vg,tk)
                        lf=float(pred['value']);variance=float(pred['variance'])
                    except (ValueError,np.linalg.LinAlgError) as exc:
                        # No hidden IDW substitution for failed top-kriging.
                        row.update(correction_status='TOPKRIGING_FAILED',error=str(exc))
                        if cfg.uncorrected_policy=='error':raise
                        if cfg.uncorrected_policy!='raw':final.values.values[:,oi,ti]=np.nan
                        factor_rows.append(row);continue
                factor=float(np.exp(lf))
                final.values.values[:,oi,ti]=raw.values.values[:,oi,ti]*factor
                row.update(factor=factor,correction_status='CALIBRATED_SEASONAL_FACTOR',n_donors=len(usable),
                           donor_stations=';'.join(c['station'] for c in usable),
                           n_training_years=min(m['n_years'] for m in fitted),log_factor_variance=variance,
                           factor_method=cfg.propagation_method)
                for c,w,m in zip(usable,weights,fitted):
                    donor_audit.append(dict(c,outlet_id=oid,target_id=tname,donor_factor=m['factor'],
                        weighting_value=float(w),weighting_role='spatial_IDW' if cfg.propagation_method!='topkriging' else 'selection_only_not_kriging_weight',
                        training_years=','.join(map(str,m['training_years']))))
            factor_rows.append(row)
    factors=pd.DataFrame(factor_rows)
    final.values.attrs=dict(raw.values.attrs,reference_source=source,correction=method,
                            correction_scale='seasonal_aggregate_not_daily',
                            reference_quality='GloFAS-based predictand, not independent observations')
    # Reflect corrected values in the long output, while retaining raw values.
    if len(final.diagnostics):
        final.diagnostics['raw_value']=final.diagnostics['value']
        arr=final.values.values
        lookup={(int(pd.Timestamp(t).year),str(o),str(k)):float(arr[yi,oi,ki])
                for yi,t in enumerate(final.values['T'].values)
                for oi,o in enumerate(final.values.outlet.values)
                for ki,k in enumerate(final.values.target.values)}
        final.diagnostics['value']=[lookup[(int(r.year),str(r.outlet_id),str(r.target_id))] for _,r in final.diagnostics.iterrows()]
        final.diagnostics=final.diagnostics.merge(factors[['outlet_id','target_id','correction_status','window_source']],on=['outlet_id','target_id'],how='left')
    formatted=format_aggregation(final,output_format=output_format)
    raw_formatted=format_aggregation(raw,output_format=output_format)
    formatted.update(glofas_raw=raw_formatted,windows=catalog,outlets=tab,
        outlet_cleaning=prepared.audit,
        provenance=prov.reset_index().merge(factors,on=['outlet_id','window_source'],how='left'),
        factors=factors,calibration_pairs=pd.DataFrame(pair_rows),observed_annual=pd.DataFrame(observed_annual),
        station_windows=station_cat,station_metadata=station_meta,donors=pd.DataFrame(donor_audit),failures=pd.DataFrame(failures),
        bootstrap=pd.DataFrame(boot),regimes={'observations':station_regimes,'glofas':local_regimes},
        mapping={'outlets':matches,'outlet_candidates':alternatives,
                 'outlet_identity':tab[['HYBAS_ID','outlet_id','X','Y','outlet_identity_status',
                     'duplicate_rows_removed','outlet_point_count','multiple_outlet_basin',
                     'MATCH_UP_AREA_KM2','match_area_source','outlet_match_status']].copy(),
                 'stations':station_matches,
                 'station_candidates':station_alts,'source_grid':extract_audit,'auxiliary':version},
        config=cfg,periods=pr,highflow_config=hw,
        resolved_reference_source=source,observation_validation_available=source=='observations',
        network_audit=dict(graph.graph) if graph is not None else {})
    if glofas_matching is not None:
        formatted['glofas_matching']=glofas_matching
        formatted['matching_provenance']=glofas_matching.context.as_dict()
        formatted['matching_network_audit']=glofas_matching.network.audit
    return formatted



def _finish_monthly_predictand(*, daily, raw, obs_series, station_sim, monthly_station_meta,
        tab, cfg, monthly_cfg, pr, hw, basins, graph, prepared, catalog, provenance,
        station_cat, station_regimes, local_regimes, boot, failures, station_matches,
        station_alts, extract_audit, version, precorrected, output_format, learning_obs=None, glofas_matching=None):
    """Monthly DAILY branch. No second seasonal-magnitude calibration is run."""
    if __package__:
        from .highflow_monthly_bias import correct_monthly_outlets, aggregate_corrected_like
    else:
        from highflow_monthly_bias import correct_monthly_outlets, aggregate_corrected_like
    prov=pd.DataFrame(provenance)
    skip=prov.loc[prov.window_source.isin(['glofas_direct','glofas_fallback']),'outlet_id'].tolist()
    application=daily.sel(T=slice(*pr.application))
    monthly=precorrected or correct_monthly_outlets(application,learning_obs if learning_obs is not None else obs_series,station_sim,
        monthly_station_meta,tab,config=monthly_cfg,propagation_config=cfg,
        calibration_period=pr.calibration,basins=basins,graph=graph,skip_outlets=skip)
    corrected=monthly['corrected'].sel(T=slice(*pr.application))
    final=aggregate_corrected_like(corrected,raw,kind=cfg.kind,selector=cfg.selector,raw_daily=application)
    final.values.attrs=dict(raw.values.attrs,correction='monthly_ratio',
        correction_scale='daily_by_calendar_month_before_seasonal_aggregation',
        temporal_mode=monthly_cfg.temporal_mode,window_source=monthly_cfg.window_source,
        window_protocol=monthly_cfg.window_protocol,
        validation_scope='conditional_calendar' if monthly_cfg.window_protocol=='fixed_reference' else 'explicit_outer_exclusions',
        reference_quality='Observation-adjusted GloFAS; not independent observed discharge')
    if len(final.diagnostics):
        keys=['year','outlet_id','target_id']
        base=raw.diagnostics[keys+['value','coverage']].rename(columns={'value':'raw_value','coverage':'raw_coverage'})
        final.diagnostics=final.diagnostics.merge(base,on=keys,how='left',validate='one_to_one')
        final.diagnostics=final.diagnostics.merge(prov[['outlet_id','window_source']],on='outlet_id',how='left',validate='many_to_one')
        final.diagnostics['correction_status']='MONTHLY_DAILY_THEN_AGGREGATED'
        final.diagnostics['temporal_mode']=monthly_cfg.temporal_mode
        final.diagnostics['window_protocol']=monthly_cfg.window_protocol
    observed_annual=[]
    selected,_=select_catalog(station_cat,kind=cfg.kind,selector=cfg.selector)
    for (name,target),cc in selected.groupby(['outlet_id','target_id'],sort=False):
        observed=_slice(obs_series[name],pr.application)
        if not len(observed):continue
        yy=pd.DatetimeIndex(raw.values['T'].values).year.tolist()
        stats=aggregate_series(observed,cc.to_dict('records'),int(cc.anchor_doy.iloc[0]),yy,raw.config)
        stats['station']=name;stats['target_id']=target
        observed_annual.extend(stats.to_dict('records'))
    # Keep a deliberately EMPTY scalar factor table: no unique seasonal factor
    # exists when 12 monthly multipliers were used. Actual factors are below.
    result=format_aggregation(final,output_format=output_format)
    result.update(glofas_raw=format_aggregation(raw,output_format=output_format),windows=catalog,
        outlets=tab,outlet_cleaning=prepared.audit,provenance=prov,
        factors=pd.DataFrame(columns=['outlet_id','target_id','factor']),
        monthly_factors=monthly['factors'],monthly_station_factors=monthly['station_factors'],
        monthly_donors=monthly['donors'],calibration_pairs=pd.DataFrame(),
        observed_annual=pd.DataFrame(observed_annual),station_windows=station_cat,
        station_metadata=monthly_station_meta,donors=monthly['donors'],failures=pd.DataFrame(failures),
        bootstrap=pd.DataFrame(boot),regimes={'observations':station_regimes,'glofas':local_regimes},
        mapping={'outlets':prepared.matches,'outlet_candidates':prepared.alternatives,
                 'outlet_identity':tab.copy(),'stations':station_matches,
                 'station_candidates':station_alts,'source_grid':extract_audit,'auxiliary':version},
        config=cfg,periods=pr,highflow_config=hw,monthly_bias_config=monthly_cfg,
        resolved_reference_source='observations',observation_validation_available=True,
        correction_scale='monthly_daily',network_audit=dict(graph.graph) if graph is not None else {},
        validation_scope='conditional_calendar' if monthly_cfg.window_protocol=='fixed_reference' else 'explicit_outer_exclusions',
        monthly_validation_note='LOMO protects monthly factors only. Seasonal evaluation requires one outer absolute block; '
            'future years are allowed except in expanding or with an information_cutoff. '
            'Excluded stations are absent from both regime estimation and calibration.')
    if monthly_cfg.keep_daily:
        result['daily_corrected']=corrected.assign_coords(
            X=('outlet',tab.X.to_numpy()),Y=('outlet',tab.Y.to_numpy()),
            HYBAS_ID=('outlet',tab.HYBAS_ID.to_numpy(dtype='int64')))
    if glofas_matching is not None:
        result['glofas_matching']=glofas_matching
        result['matching_provenance']=glofas_matching.context.as_dict()
        result['matching_network_audit']=glofas_matching.network.audit
    return result


def aggregate_with_reference(data, reference, *, variable=None, config=None,
                             kind=None,selector=None,years=None,output_format='both',**kwargs):
    """Reuse a built predictand's exact outlets/windows for C3S or another field.

    No correction and no season re-detection. The source has its own mapping;
    a GloFAS pixel's integer indexes are never reused on a C3S grid.
    """
    pcfg=reference.get('config',PredictandConfig())
    cfg=config or AggregationConfig()
    if config is None and reference.get('aggregation') is not None:
        # Reuse only spatial OUTPUT encoding, not the predictand's temporal or
        # discharge QC settings, on another physical variable/source grid.
        rc=reference['aggregation'].config
        # Do not expand a compact reference just because a one-year forecast is
        # smaller. A dense reference can still become compact for many members.
        layout=('outlet' if reference.get('resolved_xarray_layout')=='outlet'
                else rc.xarray_layout)
        cfg=replace(cfg, xarray_layout=layout,
                    max_sparse_cells=rc.max_sparse_cells,
                    max_sparse_bytes=rc.max_sparse_bytes,
                    sparse_coordinate_policy=rc.sparse_coordinate_policy,
                    coordinate_increment=rc.coordinate_increment)
    out=aggregate_field(data,reference['outlets'],reference['windows'],variable=variable,
        kind=kind or pcfg.kind,selector=pcfg.selector if selector is None else selector,
        config=cfg,years=years,**kwargs)
    return format_aggregation(out,output_format=output_format)
