"""Persist a frozen mapping safely as JSON/CSV, never as executable pickle.

The static network must be supplied when loading and its full fingerprint must
match. Daily Q/OBS values are NOT saved. Exported human-review files remain
separate from this checksum-protected machine-readable bundle.
"""
from __future__ import annotations
from dataclasses import asdict
from pathlib import Path
import hashlib
import json
import pandas as pd


def _hash(path):
    h=hashlib.sha256()
    with open(path,'rb') as f:
        for chunk in iter(lambda:f.read(1024**2),b''):h.update(chunk)
    return h.hexdigest()


def save_glofas_matching(matching,directory):
    p=Path(directory);p.mkdir(parents=True,exist_ok=True)
    tables={'outlets':matching.prepared_outlets.outlets,'outlet_matches':matching.prepared_outlets.matches,
        'outlet_alternatives':matching.prepared_outlets.alternatives,'pixel_mapping':matching.prepared_outlets.pixel_mapping,
        'station_metadata':matching.station_metadata,'station_matches':matching.station_matches,
        'station_candidates':matching.station_candidates,'metadata_checks':matching.metadata_checks,
        'network_checks':matching.network_checks,'station_neighbor_checks':matching.station_neighbor_checks,
        'reference_donors':matching.reference_donors,'review_decisions':matching.review_decisions,
        'expected_links':matching.expected_links,'outlet_network_checks':matching.outlet_network_checks}
    audits={}
    for key,value in matching.prepared_outlets.audit.items():
        if isinstance(value,pd.DataFrame):tables['audit_'+key]=value;audits[key]='audit_'+key
    manifest={}
    for name,table in tables.items():
        path=p/(name+'.csv');tmp=p/(name+'.csv.part');table.to_csv(tmp,index=False);tmp.replace(path)
        manifest[path.name]=_hash(path)
    metadata=dict(format_version=1,package_version='5.2.0',tables=manifest,audit_tables=audits,
        network_fingerprint=matching.network.fingerprint,source_grid_fingerprint=matching.source_grid_fingerprint,
        basin_fingerprint=matching.basin_fingerprint,outlet_fingerprint=matching.outlet_fingerprint,
        station_attribute_fingerprint=matching.station_attribute_fingerprint,
        context=asdict(matching.context),config=asdict(matching.config),
        physical_config=asdict(matching.physical_config),predictand_config=asdict(matching.predictand_config),
        note='Daily observations and regime curves are not included; export review curves separately when permitted.')
    # Write the validating manifest last; interrupted writes cannot be loaded as complete.
    tmp=p/'bundle.json.part';tmp.write_text(json.dumps(metadata,indent=2,ensure_ascii=False),encoding='utf-8');tmp.replace(p/'bundle.json')
    return p


def load_glofas_matching(directory,static_network):
    if __package__:
        from .glofas_matching_pixel import GloFASMatchingResult,RegimeMatchingConfig
        from .glofas_training_context import MatchingTrainingContext
        from .glofas_highflow_bias_correction import GloFASPixelMatchConfig
        from .highflow_predictand import PredictandConfig
        from .highflow_outlet_selection import PreparedOutlets
        from .highflow_monthly_bias import MonthlyBiasConfig
        from .highflow_topkriging import TopKrigingConfig
    else:
        from glofas_matching_pixel import GloFASMatchingResult,RegimeMatchingConfig
        from glofas_training_context import MatchingTrainingContext
        from glofas_highflow_bias_correction import GloFASPixelMatchConfig
        from highflow_predictand import PredictandConfig
        from highflow_outlet_selection import PreparedOutlets
        from highflow_monthly_bias import MonthlyBiasConfig
        from highflow_topkriging import TopKrigingConfig
    p=Path(directory);meta=json.loads((p/'bundle.json').read_text(encoding='utf-8'))
    if meta.get('format_version')!=1:raise ValueError('Unsupported matching storage format')
    if meta['network_fingerprint']!=static_network.fingerprint:raise ValueError('Static network differs from saved matching')
    tables={}
    names=['outlet_id','station','entity','source_outlet_id','upstream','downstream','selected_outlet_id','candidate_id']
    for name,checksum in meta['tables'].items():
        if Path(name).name!=name or not name.endswith('.csv'):raise ValueError('Invalid stored table path')
        file=p/name
        if _hash(file)!=checksum:raise ValueError(f'Saved table checksum mismatch: {name}')
        try:tables[name[:-4]]=pd.read_csv(file,dtype={k:str for k in names},float_precision='round_trip')
        except pd.errors.EmptyDataError:tables[name[:-4]]=pd.DataFrame()
    a={k:tables[v] for k,v in meta['audit_tables'].items()}
    prepared=PreparedOutlets(tables['outlets'],tables['outlet_matches'],tables['outlet_alternatives'],tables['pixel_mapping'],a)
    context=meta['context'];context['excluded_stations']=tuple(context['excluded_stations']);context['excluded_periods']=tuple(tuple(v) for v in context['excluded_periods'])
    physical=meta['physical_config'];physical['accepted_match_statuses']=tuple(physical['accepted_match_statuses'])
    pc=meta['predictand_config']
    if pc.get('monthly_bias') is not None:pc['monthly_bias']=MonthlyBiasConfig(**pc['monthly_bias'])
    if pc.get('topkriging_config') is not None:pc['topkriging_config']=TopKrigingConfig(**pc['topkriging_config'])
    return GloFASMatchingResult(prepared,tables['station_metadata'],tables['station_matches'],tables['station_candidates'],
        static_network,MatchingTrainingContext(**context),RegimeMatchingConfig(**meta['config']),GloFASPixelMatchConfig(**physical),PredictandConfig(**pc),
        meta['source_grid_fingerprint'],meta['basin_fingerprint'],meta['outlet_fingerprint'],meta['station_attribute_fingerprint'],
        tables['metadata_checks'],tables['network_checks'],tables['station_neighbor_checks'],tables['reference_donors'],
        review_decisions=tables['review_decisions'],expected_links=tables['expected_links'],outlet_network_checks=tables['outlet_network_checks'])
