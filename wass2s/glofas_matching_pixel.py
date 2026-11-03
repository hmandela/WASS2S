"""v5.2: CEMS-guided native river-pixel matching, with optional raw-flow regime.

This is an assisted procedure, NOT automatic CEMS certification. Physical
constraints are hard filters. Regime likeness ranks admissible candidates only.
A candidate is never labelled ACCEPTED_REVIEWED without an explicit decision.
The mapping is frozen for a training context; never switch pixels month by month.
"""
from __future__ import annotations
from dataclasses import dataclass, field, asdict, replace
from collections import OrderedDict
import copy
import hashlib
import json
import warnings
import numpy as np
import pandas as pd
import xarray as xr
from scipy.ndimage import uniform_filter1d
from scipy.signal import find_peaks

if __package__:
    from .glofas_static_network import GloFASStaticNetwork, grid_fingerprint, exact_subset
    from .glofas_training_context import MatchingTrainingContext
    from .highflow_utils import clean_daily_series, canonical_doy
    from .highflow_outlet_aggregation import as_field, outlet_table
    from .highflow_outlet_selection import prepare_prediction_outlets, _canonical_columns
    from .glofas_highflow_bias_correction import GloFASPixelMatchConfig, associate_stations_to_hybas, _haversine_km
else:
    from glofas_static_network import GloFASStaticNetwork, grid_fingerprint, exact_subset
    from glofas_training_context import MatchingTrainingContext
    from highflow_utils import clean_daily_series, canonical_doy
    from highflow_outlet_aggregation import as_field, outlet_table
    from highflow_outlet_selection import prepare_prediction_outlets, _canonical_columns
    from glofas_highflow_bias_correction import GloFASPixelMatchConfig, associate_stations_to_hybas, _haversine_km


@dataclass(frozen=True)
class RegimeMatchingConfig:
    mode: str = 'physical_regime'          # physical or physical_regime
    min_training_years: int = 5
    min_common_days: int = 365
    min_candidate_coverage: float = 0.90
    min_year_coverage: float = 0.60
    max_missing_climatology_days: int = 5
    smoothing_days: int = 15
    min_seasonality_cv: float = 0.03
    regime_weight: float = 0.50
    ambiguity_margin: float = 0.03
    require_channel_mask: bool = False
    require_reference: bool = False
    require_reviewed: bool = False
    max_candidates_per_target: int = 2000
    max_series_bytes: int = 256 * 1024**2
    series_cache_size: int = 128
    retain_regime_curves: int = 5
    outlet_reference: str = 'donors'        # donors or none
    network_conflict_policy: str = 'reject'   # flag or reject
    min_catchment_iou: float | None = None

    def validate(self):
        if self.mode not in {'physical','physical_regime'}:raise ValueError('Invalid matching mode')
        if self.min_training_years<2 or self.min_common_days<2:raise ValueError('Insufficient training constraints')
        if not 0<self.min_candidate_coverage<=1 or not 0<self.min_year_coverage<=1:raise ValueError('Invalid coverage')
        if not 0<=self.regime_weight<=1 or self.ambiguity_margin<0:raise ValueError('Invalid scoring controls')
        if self.smoothing_days<1 or self.smoothing_days>365:raise ValueError('Invalid smoothing_days')
        if not 0<=self.max_missing_climatology_days<365:raise ValueError('Invalid climatology gap limit')
        if self.max_candidates_per_target<1 or self.max_series_bytes<1 or self.series_cache_size<0 or self.retain_regime_curves<0:raise ValueError('Invalid memory limits')
        if self.outlet_reference not in {'donors','none'}:raise ValueError('Invalid outlet_reference')
        if self.network_conflict_policy not in {'flag','reject'}:raise ValueError('Invalid conflict policy')
        if self.min_catchment_iou is not None and not 0<=self.min_catchment_iou<=1:raise ValueError('Invalid minimum IoU')


def _frame_hash(frame):
    """Binding for exact input metadata/topology; no private Q observations saved."""
    if frame is None:return 'none'
    f=pd.DataFrame(frame).copy()
    if 'geometry' in f:
        f['geometry']=f.geometry.map(lambda g:g.wkb_hex if g is not None else None)
    cols=sorted(f.columns,key=str)
    f=pd.DataFrame(f[cols]).astype(str)
    # Order-independent binding; duplicate counts still participate.
    rows=sorted(json.dumps(r,ensure_ascii=False) for r in f.to_numpy().tolist())
    return hashlib.sha256(json.dumps([list(map(str,cols)),rows],sort_keys=True).encode()).hexdigest()


def _pixel_name(x,y):
    return f'{float(x).hex()}:{float(y).hex()}'


def _shape_statistics(reference, candidate, dates, cfg):
    """Shape diagnostics on exactly the SAME admitted daily support.

    TV distance and a daily-Parde-above-1 overlap are bounded. The latter is a
    *diagnostic*, NOT the piecewise HW detector or a replacement for its windows.
    Annual correlations and significant peaks are diagnostics, not ranking terms.
    """
    idx=pd.DatetimeIndex(dates)
    good=np.isfinite(reference)&np.isfinite(candidate)&(reference>=0)&(candidate>=0)
    idx=idx[good];a=np.asarray(reference)[good];b=np.asarray(candidate)[good]
    if len(a)<cfg.min_common_days:return dict(regime_status='INSUFFICIENT_COMMON_DAYS')
    counts=pd.Series(1,index=idx).groupby(idx.year).sum()
    years=[int(y) for y,n in counts.items() if n/len(pd.date_range(f'{y}-01-01',f'{y}-12-31'))>=cfg.min_year_coverage]
    good=np.isin(idx.year,years);idx=idx[good];a=a[good];b=b[good]
    if len(years)<cfg.min_training_years:return dict(regime_status='INSUFFICIENT_COMMON_YEARS')
    noleap=~((idx.month==2)&(idx.day==29));idx=idx[noleap];a=a[noleap];b=b[noleap]
    day=canonical_doy(idx)
    curves=[]
    for values in (a,b):
        curve=pd.Series(values,index=day).groupby(level=0).mean().reindex(range(1,366)).to_numpy()
        missing=~np.isfinite(curve)
        if missing.sum()>cfg.max_missing_climatology_days:
            return dict(regime_status='INCOMPLETE_CLIMATOLOGY')
        if missing.any():
            x=np.arange(365);known=x[~missing]
            curve[missing]=np.interp(x[missing],np.r_[known-365,known,known+365],np.tile(curve[~missing],3))
        curve=uniform_filter1d(curve,size=cfg.smoothing_days,mode='wrap')
        if curve.sum()<=0:return dict(regime_status='ZERO_FLOW_CLIMATOLOGY')
        curves.append(curve/curve.sum())
    pa,pb=curves
    if np.std(pa)/np.mean(pa)<cfg.min_seasonality_cv:
        return dict(regime_status='REFERENCE_NOT_SEASONAL',n_common_days=len(a),training_years=','.join(map(str,years)))
    ma,mb=pa>1/365,pb>1/365;union=np.count_nonzero(ma|mb)
    d=abs(int(np.argmax(pa))-int(np.argmax(pb)));d=min(d,365-d)
    def peaks(p):
        ii,_=find_peaks(np.tile(p,3),prominence=.10*(p.max()-p.min()),distance=30)
        return int(np.count_nonzero((ii>=365)&(ii<730)))
    mean_a=pd.Series(a,index=idx).groupby(idx.year).mean()
    mean_b=pd.Series(b,index=idx).groupby(idx.year).mean()
    corr=float(mean_a.corr(mean_b)) if mean_a.std()>0 and mean_b.std()>0 else np.nan
    return dict(regime_status='OK',regime_tv=float(.5*np.abs(pa-pb).sum()),
        parde_overlap=float(np.count_nonzero(ma&mb)/union) if union else np.nan,
        peak_date_error_days=int(d),reference_peak_doy=int(pa.argmax()+1),candidate_peak_doy=int(pb.argmax()+1),
        reference_n_prominent_peaks=peaks(pa),candidate_n_prominent_peaks=peaks(pb),
        annual_common_support_correlation=corr,n_common_days=int(len(a)),n_training_years=len(years),
        training_years=','.join(map(str,years)),reference_curve=pa,candidate_curve=pb)


class _Matcher:
    def __init__(self,q,network,config,physical,context,anchors=None,reference_masks=None):
        self.q=q;self.network=network;self.config=config;self.physical=physical;self.context=context
        self.cache=OrderedDict();self.reference_masks=reference_masks or {};self.curves={};self.diagnostics=[]
        self.references={};self.reference_origins={};self.anchors={}
        if anchors is not None and len(anchors):
            required={'entity','anchor_x','anchor_y','relation','evidence'}
            if required-set(anchors):raise ValueError(f'Channel anchors need {required}')
            for _,a in anchors.iterrows():
                if not str(a.evidence).strip():raise ValueError('A channel anchor needs independent evidence')
                if a.relation not in {'same_path','upstream','downstream'}:raise ValueError('Unknown anchor relation')
                self.anchors.setdefault(str(a.entity),[]).append((network.pixel(a.anchor_x,a.anchor_y),a.relation,str(a.evidence)))
        self.times=pd.DatetimeIndex(q['T'].values)
        if self.times.has_duplicates or not self.times.equals(self.times.normalize()):
            raise ValueError('Matching requires unique daily valid times, not overlapping forecast initializations')
        self.allowed=context.mask(self.times)
        self.train_times=self.times[self.allowed]
        self.up=exact_subset(network.uparea,q,network.config.coordinate_tolerance)
        self.up_values=np.asarray(self.up,dtype=float)
        self.x=np.asarray(q.X,dtype=float);self.y=np.asarray(q.Y,dtype=float)
        if config.require_channel_mask and network.channel_mask is None:
            raise ValueError('A native CHAN mask is required by this matching configuration')
        if physical.search_radius_km<=0 or physical.max_area_error_pct is None or physical.max_area_error_pct<=0:
            raise ValueError('v5.2 matching needs a positive hard radius and area tolerance')
        if physical.area_weight<0 or physical.distance_weight<0 or physical.area_weight+physical.distance_weight<=0:
            raise ValueError('Invalid physical matching weights')

    def set_reference(self,entity,series,origin):
        # The context is applied BEFORE normalization or any data-derived score.
        s=clean_daily_series(series)
        self.references[str(entity)]=s.where(self.context.mask(s.index))
        self.reference_origins[str(entity)]=origin

    def _series(self,ix,iy):
        key=(int(ix),int(iy))
        if key in self.cache:
            self.cache.move_to_end(key);return self.cache[key]
        a=self.q.isel(T=np.flatnonzero(self.allowed),Y=iy,X=ix).load()
        values=np.asarray(a.values,dtype=float).copy()
        values[(~np.isfinite(values))|(values<0)]=np.nan
        if self.config.series_cache_size:
            self.cache[key]=values
            while len(self.cache)>self.config.series_cache_size:self.cache.popitem(last=False)
        return values

    def physical_candidates(self,row,entity):
        cfg=self.physical;n=self.network
        mask_key=entity
        if mask_key not in self.reference_masks and entity.startswith('outlet:'):
            mask_key='hybas:'+str(int(row.HYBAS_ID))
        if self.config.min_catchment_iou is not None and mask_key not in self.reference_masks:
            return [], 'REFERENCE_CATCHMENT_REQUIRED'
        x=float(row.target_lon);y=float(row.target_lat)
        area=float(row.get('MATCH_UP_AREA_KM2',row.get('HYBAS_UP_AREA_KM2',np.nan)))
        if not np.isfinite(x+y+area) or area<=0:return [],'INVALID_TARGET'
        upstream_status=str(row.get('station_hybas_status','OK'))
        if upstream_status not in {'OK','AREA_PROXY_WITHOUT_OUTLET'}:return [],upstream_status
        radius=cfg.search_radius_km
        yy=np.flatnonzero(np.abs(self.y-y)<=radius/110.574)
        # Modulo distance handles 0..360 versus -180..180 search, without modifying LDD topology.
        delta=(self.x-x+180)%360-180
        xx=np.flatnonzero(np.abs(delta)<=radius/(111.32*max(np.cos(np.deg2rad(y)),.05)))
        if not len(xx) or not len(yy):return [],'OUTSIDE_GLOFAS_GRID'
        X,Y=np.meshgrid(self.x[xx],self.y[yy]);areas=self.up_values[np.ix_(yy,xx)]
        distance=_haversine_km(x,y,X,Y)
        ok=np.isfinite(areas)&(areas>max(0,cfg.min_glofas_uparea_km2))&(distance<=radius)
        locs=np.argwhere(ok)
        if len(locs)>self.config.max_candidates_per_target:
            raise MemoryError(f'{entity}: {len(locs)} potential pixels; reduce radius/domain or explicitly raise max_candidates_per_target')
        candidates=[]
        for r,c in locs:
            ix,iy=int(xx[c]),int(yy[r]);gx,gy=float(self.x[ix]),float(self.y[iy])
            pixel=n.pixel(gx,gy); reasons=[]
            area_pct=100*abs(float(areas[r,c])-area)/area
            if area_pct>cfg.max_area_error_pct:reasons.append('AREA_MISMATCH')
            if not n.valid[pixel]:reasons.append('INVALID_LDD_CELL')
            if n.channel_mask is not None and not n.channel_mask.ravel()[pixel]:reasons.append('NOT_CHANNEL')
            anchor_checks=[]
            for anchor,expected,evidence in self.anchors.get(entity,[]):
                rel=n.relationship(pixel,anchor)
                required={'UPSTREAM','DOWNSTREAM','SAME_PIXEL'} if expected=='same_path' else (
                    {'UPSTREAM','SAME_PIXEL'} if expected=='upstream' else {'DOWNSTREAM','SAME_PIXEL'})
                anchor_checks.append(rel)
                if rel not in required:reasons.append('CHANNEL_ANCHOR_'+rel)
            overlap={}
            if mask_key in self.reference_masks and not reasons:
                overlap=n.catchment_overlap(pixel,self.reference_masks[mask_key])
                if self.config.min_catchment_iou is not None:
                    if not np.isfinite(overlap.get('catchment_iou',np.nan)):
                        reasons.append('CATCHMENT_SUPPORT_UNKNOWN')
                    elif overlap['catchment_iou']<self.config.min_catchment_iou:reasons.append('CATCHMENT_MISMATCH')
            denom=cfg.area_weight+cfg.distance_weight
            # Keep each component dimensionless; divide log-ratio by max allowed
            # symmetric log mismatch, rather than giving area arbitrary units.
            ratio=max(areas[r,c]/area,area/areas[r,c]);area_norm=np.log(ratio)/np.log1p(cfg.max_area_error_pct/100)
            physical=(cfg.area_weight*area_norm+cfg.distance_weight*float(distance[r,c])/radius)/denom
            _,trace_status=n.trace(pixel) if n.valid[pixel] else ([], 'INVALID_START')
            candidates.append(dict(entity=entity,candidate_id=_pixel_name(gx,gy),pixel_id=pixel,
                candidate_lon=gx,candidate_lat=gy,glofas_x_index=ix,glofas_y_index=iy,
                candidate_uparea_km2=float(areas[r,c]),distance_km=float(distance[r,c]),
                area_error_pct=area_pct,area_log_mismatch=float(abs(np.log(areas[r,c]/area))),
                physical_score=float(physical),eligible=not reasons,reject_reason=';'.join(reasons),
                channel_status='CHECKED' if n.channel_mask is not None else 'UNVERIFIED_MASK',
                channel_anchor_relations=';'.join(anchor_checks),trace_status=trace_status,**overlap))
        return candidates, 'OK' if candidates else 'NO_RIVER_PIXEL_IN_RADIUS'

    def match(self,targets,uparea=None,*,config=None,kind='outlet'):
        rows=[];all_candidates=[]
        for ridx,row in targets.reset_index(drop=True).iterrows():
            identifier=str(row.get('station') if kind=='station' else row.outlet_id)
            entity=kind+':'+identifier
            cc,status=self.physical_candidates(row,entity)
            c=pd.DataFrame(cc)
            base=row.to_dict();base['target_row']=ridx
            base.update(entity=entity,match_status=status,review_status='NOT_REPRESENTABLE',
                glofas_lon=np.nan,glofas_lat=np.nan,glofas_uparea_km2=np.nan,
                glofas_x_index=np.nan,glofas_y_index=np.nan,match_score=np.nan,
                score_margin=np.nan,second_best_score=np.nan,candidate_count=len(c),
                reference_source=self.reference_origins.get(entity,'none'))
            if len(c):
                c['target_row']=ridx;c['outlet_id']=str(row.get('outlet_id',identifier))
                c['HYBAS_ID']=row.get('HYBAS_ID',np.nan)
                c['regime_status']='NOT_EVALUATED';c['regime_tv']=np.nan
                c['score']=c.physical_score
                eligible=c.index[c.eligible].tolist()
                reference=self.references.get(entity)
                used=False
                if eligible and self.config.mode=='physical_regime' and reference is not None:
                    memory=len(self.train_times)*len(eligible)*8
                    if memory>self.config.max_series_bytes:
                        raise MemoryError(f'{entity}: candidate-series workspace {memory/1024**2:.1f} MiB exceeds limit')
                    obs=reference.reindex(self.train_times).to_numpy(dtype=float)
                    obs_good=np.isfinite(obs)&(obs>=0)
                    data={i:self._series(int(c.loc[i,'glofas_x_index']),int(c.loc[i,'glofas_y_index'])) for i in eligible}
                    valid=[i for i,v in data.items() if obs_good.sum()>0 and
                        np.count_nonzero(obs_good&np.isfinite(v))/obs_good.sum()>=self.config.min_candidate_coverage]
                    common=obs_good.copy()
                    for i in valid:common &= np.isfinite(data[i])
                    for i in eligible:
                        if i not in valid:
                            c.loc[i,'regime_status']='LOW_MODEL_COVERAGE';continue
                        stats=_shape_statistics(obs[common],data[i][common],self.train_times[common],self.config)
                        curves={k:stats.pop(k) for k in ('reference_curve','candidate_curve') if k in stats}
                        for k,v in stats.items():c.loc[i,k]=v
                        if curves:self.curves[(entity,c.loc[i,'candidate_id'])]=curves
                    good=c.index[c.eligible&c.regime_status.eq('OK')].tolist()
                    if good:
                        # Never favour a data-poor pixel by omitting its regime penalty.
                        insufficient=c.eligible&~c.regime_status.eq('OK')
                        c.loc[insufficient,'eligible']=False
                        c.loc[insufficient,'reject_reason']='INSUFFICIENT_REGIME_SUPPORT'
                        w=self.config.regime_weight
                        c.loc[good,'score']=(1-w)*c.loc[good,'physical_score']+w*c.loc[good,'regime_tv']
                        used=True
                if self.config.require_reference and not used and self.config.mode=='physical_regime':
                    c.loc[c.eligible,'reject_reason']='REFERENCE_UNAVAILABLE';c.loc[:,'eligible']=False
                ranked=c[c.eligible].sort_values(['score','distance_km','candidate_lon','candidate_lat'],kind='stable')
                c['candidate_rank']=0
                c.loc[ranked.index,'candidate_rank']=np.arange(1,len(ranked)+1)
                if len(ranked):
                    best=ranked.iloc[0];margin=float(ranked.iloc[1].score-best.score) if len(ranked)>1 else np.nan
                    ambiguous=bool(np.isfinite(margin) and margin<self.config.ambiguity_margin)
                    status='LOW_CANDIDATE_SUPPORT' if len(ranked)<self.physical.min_candidates else 'OK'
                    base.update(match_status=status,review_status='REVIEW_REQUIRED' if ambiguous else 'AUTO_CANDIDATE',
                        glofas_lon=best.candidate_lon,glofas_lat=best.candidate_lat,
                        glofas_uparea_km2=best.candidate_uparea_km2,glofas_x_index=int(best.glofas_x_index),glofas_y_index=int(best.glofas_y_index),
                        distance_km=best.distance_km,area_error_pct=best.area_error_pct,area_log_mismatch=best.area_log_mismatch,
                        match_score=float(best.score),score_margin=margin,second_best_score=float(ranked.iloc[1].score) if len(ranked)>1 else np.nan,
                        candidate_id=best.candidate_id,pixel_id=int(best.pixel_id),selection_ambiguous=ambiguous,
                        regime_used=used,regime_status=best.regime_status,regime_tv=best.regime_tv,
                        n_admissible_candidates=len(ranked),channel_status=best.channel_status,
                        topology_status=best.trace_status,training_context=self.context.fingerprint,
                        glofas_x_coord_name='X',glofas_y_coord_name='Y')
                else:base['match_status']='NO_ADMISSIBLE_PIXEL'
                keep_curves=set(ranked.head(self.config.retain_regime_curves).candidate_id)
                for key in list(self.curves):
                    if key[0]==entity and key[1] not in keep_curves:self.curves.pop(key)
                all_candidates.append(c)
            rows.append(base)
        return pd.DataFrame(rows),pd.concat(all_candidates,ignore_index=True) if all_candidates else pd.DataFrame()


@dataclass
class GloFASMatchingResult:
    """Frozen experiment mapping. Review functions return copies, not mutations."""
    prepared_outlets: object
    station_metadata: pd.DataFrame
    station_matches: pd.DataFrame
    station_candidates: pd.DataFrame
    network: GloFASStaticNetwork
    context: MatchingTrainingContext
    config: RegimeMatchingConfig
    physical_config: object
    predictand_config: object
    source_grid_fingerprint: str
    basin_fingerprint: str
    outlet_fingerprint: str
    station_attribute_fingerprint: str
    metadata_checks: pd.DataFrame
    network_checks: pd.DataFrame
    station_neighbor_checks: pd.DataFrame
    reference_donors: pd.DataFrame
    curves: dict = field(default_factory=dict, repr=False)
    review_decisions: pd.DataFrame = field(default_factory=pd.DataFrame)
    expected_links: pd.DataFrame = field(default_factory=pd.DataFrame)
    outlet_network_checks: pd.DataFrame = field(default_factory=pd.DataFrame)

    @property
    def selected_pixels(self):
        a=self.prepared_outlets.matches.copy();a['entity_type']='outlet'
        b=self.station_matches.copy();b['entity_type']='station'
        return pd.concat([a,b],ignore_index=True)

    @property
    def candidate_pixels(self):
        a=self.prepared_outlets.alternatives.copy();a['entity_type']='outlet'
        b=self.station_candidates.copy();b['entity_type']='station'
        return pd.concat([a,b],ignore_index=True)

    def validate_for(self,q,basins,outlets,config,match_config=None,station_attributes=None):
        if self.source_grid_fingerprint != self.network.validate_discharge(q):
            raise ValueError('Matching belongs to a different discharge grid; rebuild it')
        if self.basin_fingerprint!=_frame_hash(_canonical_columns(basins)) or self.outlet_fingerprint!=_frame_hash(_canonical_columns(outlets)):
            raise ValueError('Basin/outlet catalogue changed after matching; rebuild rather than remap silently')
        for k in ('coastal_policy','duplicate_outlet_policy','missing_coast_policy','multi_no_match_policy'):
            if getattr(config,k)!=getattr(self.predictand_config,k):
                raise ValueError(f'Matching cleaning configuration differs: {k}')
        if match_config is not None and asdict(match_config)!=asdict(self.physical_config):
            raise ValueError('match_config differs from the frozen mapping')
        if station_attributes is not None and _frame_hash(station_attributes)!=self.station_attribute_fingerprint:
            raise ValueError('station_attributes changed after matching')
        return True


def _normalize_observations(hydro,observations,variable):
    if observations is None:return {}
    if isinstance(observations,dict):
        if any(not isinstance(v,dict) or 'series' not in v for v in observations.values()):
            raise ValueError('Observation dict must be station -> {series,X,Y,...}')
        return {str(k):dict(v,series=clean_daily_series(v['series'])) for k,v in observations.items()}
    return hydro._prepare_highflow_observations(observations,variable=variable,
                missing_values=(-99.,-999.),negative_flow_as_nan=True)


def _inferred_links(metadata,basins):
    """Distinct HYBAS nodes only; no order inferred between gauges in one polygon."""
    import networkx as nx
    G=nx.DiGraph();ids=set(basins.HYBAS_ID.astype(int))
    if 'NEXT_DOWN' not in basins:return pd.DataFrame(columns=['upstream','downstream','evidence'])
    for _,r in basins.iterrows():
        if pd.notna(r.NEXT_DOWN) and int(r.NEXT_DOWN) in ids:G.add_edge(int(r.HYBAS_ID),int(r.NEXT_DOWN))
    rows=[]
    for _,a in metadata.iterrows():
        for _,b in metadata.iterrows():
            if a.station==b.station or pd.isna(a.HYBAS_ID) or pd.isna(b.HYBAS_ID) or a.HYBAS_ID==b.HYBAS_ID:continue
            if int(a.HYBAS_ID) in G and int(b.HYBAS_ID) in G and nx.has_path(G,int(a.HYBAS_ID),int(b.HYBAS_ID)):
                rows.append(dict(upstream=str(a.station),downstream=str(b.station),evidence='HydroBASINS_NEXT_DOWN'))
    return pd.DataFrame(rows,columns=['upstream','downstream','evidence'])


def check_station_network(network,matches,links,observations,context):
    checks=[];neighbors=[]
    mm=matches.set_index('station') if len(matches) else pd.DataFrame()
    if len(links) and not {'upstream','downstream','evidence'}.issubset(links):raise ValueError('Links require upstream,downstream,evidence')
    for _,r in links.iterrows():
        a,b=str(r.upstream),str(r.downstream)
        if a==b or not str(r.evidence).strip():raise ValueError('Invalid expected station relationship')
        rec=dict(upstream=a,downstream=b,evidence=r.evidence)
        if a not in mm.index or b not in mm.index:
            rel='MISSING_STATION'
        elif pd.isna(mm.loc[a].get('pixel_id')) or pd.isna(mm.loc[b].get('pixel_id')):
            rel='NO_MATCH'
        else:rel=network.relationship(int(mm.loc[a].pixel_id),int(mm.loc[b].pixel_id))
        status='OK' if rel=='UPSTREAM' else ('UNRESOLVED_AT_GRID' if rel=='SAME_PIXEL' else ('UNKNOWN' if rel.startswith('UNKNOWN') or rel in {'NO_MATCH','MISSING_STATION'} else 'CONFLICT'))
        checks.append(dict(rec,model_relation=rel,status=status))
        if a in observations and b in observations and a not in context.excluded_stations and b not in context.excluded_stations:
            s=context.filter_series(observations[a]['series'],a);t=context.filter_series(observations[b]['series'],b)
            pair=pd.concat([s.rename('a'),t.rename('b')],axis=1).dropna()
            corr=float(pair.a.corr(pair.b)) if len(pair)>2 and pair.a.std()>0 and pair.b.std()>0 else np.nan
            ratio=float(pair.b.mean()/pair.a.mean()) if len(pair) and pair.a.mean()>0 else np.nan
            neighbors.append(dict(rec,n_pairs=len(pair),pearson_r=corr,mean_downstream_upstream_ratio=ratio,
                status='DIAGNOSTIC_NOT_A_MONOTONIC_FLOW_RULE',context=context.fingerprint))
    return pd.DataFrame(checks),pd.DataFrame(neighbors)


def prepare_glofas_matching(hydro,*,glofas_discharge,static_network=None,glofas_ldd=None,
        glofas_uparea=None,glofas_channel_mask=None,glofas_cell_area=None,network_config=None,
        glofas_version=None,basins=None,outlets=None,hydrosheds=None,hybas_level=None,
        observations=None,observation_variable='Observation',station_attributes=None,
        variable=None,config=None,match_config=None,predictand_config=None,
        training_context=None,channel_anchors=None,reference_catchments=None,expected_station_links=None):
    """Prepare one auditable mapping BEFORE any daily/seasonal bias correction.

    `reference_catchments` is an optional dict entity -> complete native-grid
    boolean mask. Entity keys are 'station:NAME' or 'outlet:SOURCE_OUTLET_ID'.
    `channel_anchors` are independent known river pixels, not learned from Q.
    Reference donor choice uses physical/network metadata only. Auto-review is
    deliberately impossible. Real station names/rivers are required for manual
    verification but missing fields are reported rather than invented.
    """
    if __package__:
        from .highflow_predictand import PredictandConfig,_candidates,_weights
        from .glofas_highflow_bias_correction import build_hybas_digraph
    else:
        from highflow_predictand import PredictandConfig,_candidates,_weights
        from glofas_highflow_bias_correction import build_hybas_digraph
    cfg=config or RegimeMatchingConfig();cfg.validate()
    ctx=training_context or MatchingTrainingContext()
    if not isinstance(ctx,MatchingTrainingContext):raise TypeError('training_context must be MatchingTrainingContext')
    pc=predictand_config or PredictandConfig();pc.validate()
    mc=match_config or GloFASPixelMatchConfig()
    if hydrosheds is not None:
        if basins is not None or outlets is not None:raise ValueError('Pass hydrosheds OR basin/outlet layers')
        basins=hydrosheds['basins'];outlets=hydrosheds['pour_points']
        declared=hydrosheds.get('metadata',{}).get('level')
        if hybas_level is not None and declared is not None and int(hybas_level)!=int(declared):raise ValueError('Conflicting HYBAS level')
        hybas_level=declared if hybas_level is None else hybas_level
    if basins is None or outlets is None:raise ValueError('Basin polygons and outlet points are required')
    basins=_canonical_columns(basins)
    q=as_field(glofas_discharge,variable)
    if set(q.dims)!={'T','Y','X'}:raise ValueError('Matching expects historical Q(T,Y,X), not ensemble or already aggregated data')
    if static_network is None:
        if glofas_ldd is None or glofas_uparea is None:raise ValueError('LDD and UPAREA are required; no silent area-only fallback')
        static_network=GloFASStaticNetwork(glofas_ldd,glofas_uparea,glofas_channel_mask,glofas_cell_area,
            version=glofas_version,config=network_config)
    elif any(x is not None for x in (glofas_ldd,glofas_uparea,glofas_channel_mask,glofas_cell_area)):
        raise ValueError('Pass static_network OR individual maps')
    fingerprint=static_network.validate_discharge(q)
    engine=_Matcher(q,static_network,cfg,mc,ctx,channel_anchors,reference_catchments)
    st=_normalize_observations(hydro,observations,observation_variable)
    if set(ctx.excluded_stations)-set(st):raise ValueError('Unknown excluded station(s) in matching context')
    meta=[];metadata_checks=[]
    for name,info in st.items():
        x=info.get('X_ORIG',info.get('X',np.nan));y=info.get('Y_ORIG',info.get('Y',np.nan))
        meta.append(dict(station=name,X=x,Y=y,ELEV=info.get('ELEV',np.nan)))
        if name not in ctx.excluded_stations:engine.set_reference('station:'+name,info['series'],'observations')
    meta=pd.DataFrame(meta,columns=['station','X','Y','ELEV'])
    if station_attributes is not None:
        attrs=station_attributes.copy();attrs['station']=attrs.station.astype(str)
        if attrs.station.duplicated().any():raise ValueError('Duplicate station attribute identifiers')
        meta=meta.merge(attrs.drop(columns=[c for c in ('X','Y','ELEV') if c in attrs]),on='station',how='left',validate='one_to_one')
    # This pre-catalogue normalizes identities only; it never selects a representative.
    pretab=outlet_table(_canonical_columns(outlets),basins,level=hybas_level,duplicate_policy='separate',warn_multiple=False)
    import geopandas as gpd
    gg=gpd.GeoDataFrame(pretab.drop(columns='geometry',errors='ignore'),geometry=gpd.points_from_xy(pretab.X,pretab.Y),crs=4326)
    station_meta=pd.DataFrame();sm=pd.DataFrame();sa=pd.DataFrame()
    if len(meta):
        station_meta=associate_stations_to_hybas(meta,basins,hybas_outlets=gg,config=mc)
        station_meta['outlet_id']=station_meta.station.astype(str)
        station_meta['multiple_point_candidates']=station_meta.multiple_outlet_basin
        # A measured gauge area is unambiguous even when a non-coastal basin
        # layer contains alternative outlet positions. Keep the original flag.
        measured_noncoastal=station_meta.area_source.eq('station_area') & station_meta.get('COAST',pd.Series(np.nan,index=station_meta.index)).eq(0)
        station_meta.loc[measured_noncoastal,'multiple_outlet_basin']=False
        if not pc.allow_station_area_proxy:
            station_meta.loc[station_meta.area_source!='station_area','station_hybas_status']='STATION_AREA_REQUIRED'
        sm,sa=engine.match(station_meta,kind='station')
        sm['outlet_id']=station_meta.outlet_id.values
        station_meta['matched_source_x']=sm.glofas_lon.to_numpy();station_meta['matched_source_y']=sm.glofas_lat.to_numpy()
        for _,r in station_meta.iterrows():
            metadata_checks.append(dict(station=r.station,coordinates_valid=bool(np.isfinite(r.X+r.Y)),
                station_name_provided=bool(str(r.station).strip()),river_name_provided=bool(pd.notna(r.get('river_name'))),
                area_source=r.area_source,station_hybas_status=r.station_hybas_status,
                cartographic_review='NOT_PERFORMED',excluded_from_training=str(r.station) in ctx.excluded_stations))
    links=expected_station_links.copy() if expected_station_links is not None else _inferred_links(station_meta,basins)
    checks,neighbors=check_station_network(static_network,sm,links,st,ctx)
    flagged=set()
    if len(checks):
        flagged=set(checks.loc[checks.status!='OK','upstream'])|set(checks.loc[checks.status!='OK','downstream'])
        sm.loc[sm.station.isin(flagged),'review_status']='REVIEW_REQUIRED'
        if cfg.network_conflict_policy=='reject':
            conflicts=set(checks.loc[checks.status=='CONFLICT','upstream'])|set(checks.loc[checks.status=='CONFLICT','downstream'])
            sm.loc[sm.station.isin(conflicts),'match_status']='NETWORK_CONFLICT'
    graph=build_hybas_digraph(basins) if len(station_meta) and pc.propagation_method in {'network','hybrid'} else None
    donor_rows=[];relation_cache={}
    allowed_donors=station_meta.copy()
    if len(allowed_donors):
        allowed=set(sm.loc[sm.match_status.isin(mc.accepted_match_statuses),'station'])-set(ctx.excluded_stations)-flagged
        allowed_donors=allowed_donors[allowed_donors.station.isin(allowed)]
    def outlet_matcher(targets,up,*,config):
        if cfg.outlet_reference=='donors' and cfg.mode=='physical_regime' and len(allowed_donors):
            for _,target in targets.iterrows():
                donor_target=target.copy()
                if bool(target.get('representative_area_assumption',False)):
                    donor_target['multiple_outlet_basin']=False
                choices=_candidates(donor_target,allowed_donors,pc,graph,relation_cache)
                if not choices:continue
                choices=choices[:pc.k_donors]
                donor_weights=_weights(choices,pc)
                # _candidates returns geometric/network weights; never choose a
                # donor on target observations or candidate GloFAS fit.
                pieces=[];weights=[]
                for d,w in zip(choices,donor_weights):
                    name=str(d['station']);series=ctx.filter_series(clean_daily_series(st[name]['series']),name)
                    mu=series.mean()
                    if not np.isfinite(mu) or mu<=0:continue
                    pieces.append((series/mu).rename(name));weights.append(float(w))
                if pieces:
                    frame=pd.concat(pieces,axis=1);weights=np.asarray(weights)/sum(weights)
                    for name,w in zip(frame.columns,weights):
                        donor_rows.append(dict(outlet_id=str(target.outlet_id),station=name,weight=float(w),reference='normalized_observed_donor'))
                    ref=frame.mul(weights,axis=1).sum(axis=1,min_count=len(pieces))
                    engine.set_reference('outlet:'+str(target.outlet_id),ref,'regionalized_observations')
        return engine.match(targets,kind='outlet')
    prepared=prepare_prediction_outlets(outlets,basins,engine.up,config=pc,
        match_config=mc,level=hybas_level,matcher=outlet_matcher)
    # final outlet IDs can change after best_glofas; maintain a review key for
    # final targets while retaining pre-selection source IDs in the audit.
    if len(prepared.alternatives):
        prepared.alternatives['source_entity']=prepared.alternatives.entity
        prepared.alternatives['entity']='outlet:'+prepared.alternatives.outlet_id.astype(str)
    for _,r in prepared.matches.iterrows():
        source=str(r.get('entity',''));dest='outlet:'+str(r.outlet_id)
        for (entity,candidate),value in list(engine.curves.items()):
            if entity==source:engine.curves[(dest,candidate)]=value
    prepared.matches['source_entity']=prepared.matches.get('entity','')
    prepared.matches['entity']='outlet:'+prepared.matches.outlet_id.astype(str)
    outlet_checks=check_outlet_network(static_network,prepared.matches,basins)
    if len(outlet_checks):
        flagged_out=set(outlet_checks.loc[outlet_checks.status!='OK','upstream']) | set(outlet_checks.loc[outlet_checks.status!='OK','downstream'])
        prepared.matches.loc[prepared.matches.outlet_id.isin(flagged_out),'review_status']='REVIEW_REQUIRED'
        if cfg.network_conflict_policy=='reject':
            conflicts=set(outlet_checks.loc[outlet_checks.status=='CONFLICT','upstream']) | set(outlet_checks.loc[outlet_checks.status=='CONFLICT','downstream'])
            prepared.matches.loc[prepared.matches.outlet_id.isin(conflicts),'match_status']='NETWORK_CONFLICT'
        prepared.pixel_mapping['status']=np.where(prepared.matches.match_status.isin(mc.accepted_match_statuses),'OK',prepared.matches.match_status)
    return GloFASMatchingResult(prepared,station_meta,sm,sa,static_network,ctx,cfg,mc,pc,
        fingerprint,_frame_hash(basins),_frame_hash(_canonical_columns(outlets)),_frame_hash(station_attributes),
        pd.DataFrame(metadata_checks),checks,neighbors,pd.DataFrame(donor_rows),engine.curves,
        expected_links=links,outlet_network_checks=outlet_checks)


def check_outlet_network(network,matches,basins):
    """Nearest downstream retained HYBAS target; missing graph nodes stay unknown.

    A polygon has no unique network node when multiple outlets were deliberately
    kept: those groups are excluded from inferred ordering, not assigned a
    fabricated channel. Explicit station links cover surveyed gauge relations.
    """
    if 'NEXT_DOWN' not in basins or matches.empty:return pd.DataFrame()
    nxt={int(r.HYBAS_ID):int(r.NEXT_DOWN) for _,r in basins.iterrows() if pd.notna(r.NEXT_DOWN)}
    unique=matches[~matches.HYBAS_ID.duplicated(keep=False)].copy()
    ids=dict(zip(unique.HYBAS_ID.astype(int),unique.outlet_id.astype(str)))
    links=[]
    for hid,oid in ids.items():
        p=nxt.get(hid,0);seen={hid}
        while p and p not in seen:
            if p in ids:
                links.append(dict(upstream=oid,downstream=ids[p],evidence='HYBAS_NEXT_DOWN_nearest_retained'));break
            seen.add(p);p=nxt.get(p,0)
    fake=unique.assign(station=unique.outlet_id.astype(str))
    checks,_=check_station_network(network,fake,pd.DataFrame(links),{},MatchingTrainingContext())
    return checks
