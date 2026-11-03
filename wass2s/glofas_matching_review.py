"""Human-in-the-loop review, export and diagnostic figures (no CLI).

No internet basemap is downloaded. Provide independent river/basin layers in
WGS84 if available. A generated map is NOT evidence of a completed review.
"""
from __future__ import annotations
from dataclasses import asdict
from pathlib import Path
import copy
import json
import numpy as np
import pandas as pd


def review_template(matching):
    cols=['entity','candidate_id','glofas_lon','glofas_lat','review_status']
    out=matching.selected_pixels.reindex(columns=cols).copy()
    out['decision']='PENDING';out['reviewer']='';out['reason']='';out['review_time']=''
    out['context_fingerprint']=matching.context.fingerprint
    return out


def _refresh_maps(matching):
    p=matching.prepared_outlets;sm=p.matches
    p.pixel_mapping=pd.DataFrame(dict(outlet_id=sm.outlet_id,source_x=sm.glofas_lon,source_y=sm.glofas_lat,
        status=np.where(sm.match_status.isin(matching.physical_config.accepted_match_statuses),'OK',sm.match_status)))
    if len(matching.station_metadata):
        m=matching.station_matches.set_index('station')
        matching.station_metadata['matched_source_x']=matching.station_metadata.station.map(m.glofas_lon)
        matching.station_metadata['matched_source_y']=matching.station_metadata.station.map(m.glofas_lat)


def apply_matching_decisions(matching,decisions):
    """Return a reviewed copy. Cannot override hard physical filters.

    Only final retained point targets can be reviewed here. To change which
    distinct HYBAS outlet location was retained, rerun preparation with a
    corrected independent outlet layer/anchor; do not relabel another point.
    Rejection preserves coordinates for audit but makes extraction invalid.
    """
    if __package__:
        from .glofas_matching_pixel import check_station_network
    else:
        from glofas_matching_pixel import check_station_network
    required={'entity','decision','reviewer','reason','review_time','context_fingerprint'}
    if required-set(decisions):raise ValueError(f'Review decisions require {sorted(required)}')
    if decisions.entity.duplicated().any():raise ValueError('Duplicate review entity')
    out=copy.copy(matching);out.prepared_outlets=copy.deepcopy(matching.prepared_outlets)
    out.station_matches=matching.station_matches.copy();out.station_metadata=matching.station_metadata.copy()
    history=[]
    changed_reference_stations=set()
    candidates=out.candidate_pixels
    known=set(out.selected_pixels.entity)
    for _,d in decisions.iterrows():
        entity=str(d.entity);action=str(d.decision).upper()
        if entity not in known:raise ValueError(f'Unknown review entity {entity}')
        if action=='PENDING':continue
        if action not in {'ACCEPT','REJECT','UNRESOLVED'}:raise ValueError('decision: ACCEPT, REJECT or UNRESOLVED')
        if d.context_fingerprint!=matching.context.fingerprint:raise ValueError('Review belongs to a different training context')
        if any(pd.isna(d[k]) or not str(d[k]).strip() for k in ('reviewer','reason','review_time')):
            raise ValueError('Reviewer, time and a nonempty justification are required')
        pd.Timestamp(d.review_time)
        table=out.station_matches if entity.startswith('station:') else out.prepared_outlets.matches
        idx=table.index[table.entity==entity][0]
        if entity.startswith('station:') and action!='ACCEPT':
            changed_reference_stations.add(entity.split(':',1)[1])
        if action=='ACCEPT':
            key=d.get('candidate_id',table.loc[idx].get('candidate_id'))
            hit=candidates[(candidates.entity==entity)&(candidates.candidate_id==key)]
            if len(hit)!=1 or not bool(hit.iloc[0].eligible):
                raise ValueError('Cannot accept an absent or physically/statistically inadmissible candidate')
            c=hit.iloc[0]
            if entity.startswith('station:') and str(table.loc[idx].get('candidate_id'))!=str(c.candidate_id):
                changed_reference_stations.add(entity.split(':',1)[1])
            for source,dest in [('candidate_lon','glofas_lon'),('candidate_lat','glofas_lat'),
                ('candidate_uparea_km2','glofas_uparea_km2'),('score','match_score'),
                ('glofas_x_index','glofas_x_index'),('glofas_y_index','glofas_y_index'),
                ('candidate_id','candidate_id'),('pixel_id','pixel_id'),('distance_km','distance_km'),
                ('area_error_pct','area_error_pct'),('regime_tv','regime_tv'),('regime_status','regime_status')]:
                table.loc[idx,dest]=c[source]
            table.loc[idx,'review_status']='ACCEPTED_REVIEWED';table.loc[idx,'match_status']='OK'
        else:
            table.loc[idx,'review_status']='REJECTED_REVIEWED' if action=='REJECT' else 'UNRESOLVED'
            table.loc[idx,'match_status']='REVIEW_REJECTED' if action=='REJECT' else 'REVIEW_UNRESOLVED'
        history.append(d.to_dict())
    out.review_decisions=pd.concat([matching.review_decisions,pd.DataFrame(history)],ignore_index=True)
    # No raw observations are needed for this topological recheck.
    checks,_=check_station_network(out.network,out.station_matches,out.expected_links,{},out.context)
    out.network_checks=checks
    if len(checks):
        bad=checks[checks.status=='CONFLICT']
        names=set(bad.upstream)|set(bad.downstream)
        out.station_matches.loc[out.station_matches.station.isin(names),'review_status']='REVIEW_REQUIRED'
        if out.config.network_conflict_policy=='reject':
            out.station_matches.loc[out.station_matches.station.isin(names),'match_status']='NETWORK_CONFLICT'
    if len(matching.outlet_network_checks):
        fake=out.prepared_outlets.matches.assign(station=out.prepared_outlets.matches.outlet_id.astype(str))
        oc,_=check_station_network(out.network,fake,matching.outlet_network_checks[['upstream','downstream','evidence']],{},out.context)
        out.outlet_network_checks=oc
        bad=oc[oc.status=='CONFLICT'];names=set(bad.upstream)|set(bad.downstream)
        out.prepared_outlets.matches.loc[out.prepared_outlets.matches.outlet_id.isin(names),'review_status']='REVIEW_REQUIRED'
        if out.config.network_conflict_policy=='reject':
            out.prepared_outlets.matches.loc[out.prepared_outlets.matches.outlet_id.isin(names),'match_status']='NETWORK_CONFLICT'
    # Rejecting/repositioning a station can invalidate the donor reference that
    # was already used to rank outlets. A review must not bless stale derived
    # rankings: rebuild those targets after updating the reference catalogue.
    if len(out.station_matches):
        changed_reference_stations |= set(out.station_matches.loc[
            ~out.station_matches.match_status.isin(out.physical_config.accepted_match_statuses), 'station'].astype(str))
    if changed_reference_stations and len(out.reference_donors):
        affected=set(out.reference_donors.loc[
            out.reference_donors.station.astype(str).isin(changed_reference_stations), 'outlet_id'].astype(str))
        for i,r in out.prepared_outlets.matches.iterrows():
            aliases={str(r.outlet_id), str(r.get('source_entity','')).removeprefix('outlet:')}
            if aliases & affected:
                out.prepared_outlets.matches.loc[i,'review_status']='REVIEW_REQUIRED'
                out.prepared_outlets.matches.loc[i,'match_status']='REFERENCE_REBUILD_REQUIRED'
    _refresh_maps(out)
    return out


def export_matching_review(matching,directory,*,include_curves=True):
    """Write CSV + JSON + candidate/selected GeoJSON suitable for GIS review.

    Daily observations are NOT exported. Normalized curves can be omitted.
    Training dates and station identities are provenance; handle their privacy
    according to the data provider's agreement.
    """
    root=Path(directory);root.mkdir(parents=True,exist_ok=True)
    tables={'selected_pixels':matching.selected_pixels,'candidate_pixels':matching.candidate_pixels,
        'metadata_checks':matching.metadata_checks,'network_checks':matching.network_checks,'outlet_network_checks':matching.outlet_network_checks,
        'station_neighbor_checks':matching.station_neighbor_checks,'reference_donors':matching.reference_donors,
        'review_decisions':matching.review_decisions,'review_template':review_template(matching)}
    for name,table in tables.items():table.to_csv(root/(name+'.csv'),index=False,encoding='utf-8-sig')
    meta=dict(version='5.2.0',training=matching.context.as_dict(),config=asdict(matching.config),
        physical_config=asdict(matching.physical_config),network=matching.network.audit,
        source_grid_fingerprint=matching.source_grid_fingerprint,
        notice='CEMS-guided assistance, not automatic CEMS validation. Maps must be reviewed.')
    (root/'provenance.json').write_text(json.dumps(meta,indent=2,ensure_ascii=False,default=str),encoding='utf-8')
    # GeoJSON manually written: no dependency on a particular GDAL/Fiona backend.
    for name,x,y in [('selected_pixels','glofas_lon','glofas_lat'),('candidate_pixels','candidate_lon','candidate_lat')]:
        table=tables[name];features=[]
        if x in table and y in table:
            for _,r in table.iterrows():
                if not np.isfinite(float(r[x])) or not np.isfinite(float(r[y])):continue
                props=json.loads(pd.Series(r).to_json())
                features.append(dict(type='Feature',geometry=dict(type='Point',coordinates=[float(r[x]),float(r[y])]),properties=props))
        (root/(name+'.geojson')).write_text(json.dumps(dict(type='FeatureCollection',features=features),ensure_ascii=False),encoding='utf-8')
    if include_curves:
        curves=[]
        for (entity,candidate),data in matching.curves.items():
            for d,(o,g) in enumerate(zip(data['reference_curve'],data['candidate_curve']),1):
                curves.append(dict(entity=entity,candidate_id=candidate,doy=d,reference_fraction=o,glofas_fraction=g))
        pd.DataFrame(curves).to_csv(root/'normalized_regimes.csv',index=False)
    return root


def plot_matching_map(matching,entity,*,rivers=None,basins=None,ax=None,padding_degrees=.15):
    """Local lon/lat diagnostic, not an equal-distance projection or satellite validation."""
    import matplotlib.pyplot as plt
    if ax is None:_,ax=plt.subplots(figsize=(8,7))
    c=matching.candidate_pixels;c=c[c.entity==entity]
    selected=matching.selected_pixels;selected=selected[selected.entity==entity]
    if selected.empty:raise KeyError(entity)
    r=selected.iloc[0]
    if basins is not None:
        if basins.crs is None:raise ValueError('Declare basin CRS')
        basins.to_crs(4326).boundary.plot(ax=ax,linewidth=.5)
    if rivers is not None:
        if rivers.crs is None:raise ValueError('Declare river CRS')
        rivers.to_crs(4326).plot(ax=ax,linewidth=.7)
    if len(c):
        valid=c[c.eligible];bad=c[~c.eligible]
        ax.scatter(bad.candidate_lon,bad.candidate_lat,marker='x',label='Rejected candidates')
        points=ax.scatter(valid.candidate_lon,valid.candidate_lat,c=valid.score,label='Admissible candidates')
        ax.figure.colorbar(points,ax=ax,label='Matching score (lower is better)')
    ax.scatter([r.target_lon],[r.target_lat],marker='*',s=160,label='Provided target')
    if np.isfinite(r.glofas_lon):
        ax.scatter([r.glofas_lon],[r.glofas_lat],marker='s',s=100,facecolors='none',label='Selected pixel')
        path,_=matching.network.trace(int(r.pixel_id))
        xy=np.array([matching.network.xy(p) for p in path])
        if len(xy):ax.plot(xy[:,0],xy[:,1],linewidth=.8,label='Model LDD downstream path')
    xs=np.r_[c.candidate_lon if len(c) else [],r.target_lon];ys=np.r_[c.candidate_lat if len(c) else [],r.target_lat]
    ax.set_xlim(np.nanmin(xs)-padding_degrees,np.nanmax(xs)+padding_degrees)
    ax.set_ylim(np.nanmin(ys)-padding_degrees,np.nanmax(ys)+padding_degrees)
    ax.set_xlabel('Longitude');ax.set_ylabel('Latitude')
    ax.set_title(f'{entity} — {r.review_status}\nGeographical verification remains required')
    ax.legend(loc='best');return ax


def plot_matching_regimes(matching,entity,*,top_n=5,ax=None):
    import matplotlib.pyplot as plt
    if ax is None:_,ax=plt.subplots(figsize=(10,4))
    c=matching.candidate_pixels;c=c[(c.entity==entity)&c.eligible].sort_values('score').head(top_n)
    reference_done=False
    for _,r in c.iterrows():
        curves=matching.curves.get((entity,r.candidate_id))
        if curves is None:continue
        if not reference_done:
            ax.plot(np.arange(1,366),365*curves['reference_curve'],linewidth=2,label='Reference (training only)');reference_done=True
        ax.plot(np.arange(1,366),365*curves['candidate_curve'],label=f'({r.candidate_lon:.3f}, {r.candidate_lat:.3f})')
    if not reference_done:ax.text(.5,.5,'No admissible observational regime reference',transform=ax.transAxes,ha='center')
    else:ax.legend()
    ax.set_xlabel('Climatological day (365-day calendar)');ax.set_ylabel('Daily Pardé coefficient')
    ax.set_title(f'{entity} — raw-model regime, before bias correction');return ax
