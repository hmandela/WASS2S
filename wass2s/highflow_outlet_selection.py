"""Auditable target cleaning before high-water detection and xarray formatting.

Order: exact (HYBAS_ID, X, Y) duplicates -> COAST=1 -> GloFAS area/distance
matching -> one representative point per non-coastal HYBAS_ID (optional).

The original polygons are NEVER filtered in place. They remain available for
upstream closure, NEXT_DOWN navigation and top-kriging supports. This operation
selects prediction targets; it is not a modification of river topology.

No I/O, downloads or command-line execution occurs on import.
"""
from __future__ import annotations
from dataclasses import dataclass
import logging
import warnings
import numpy as np
import pandas as pd

if __package__:
    from .highflow_outlet_aggregation import outlet_table
    from .glofas_highflow_bias_correction import match_hybas_outlets_to_glofas
else:
    from highflow_outlet_aggregation import outlet_table
    from glofas_highflow_bias_correction import match_hybas_outlets_to_glofas

LOG = logging.getLogger(__name__)


class NoUsableOutletsError(ValueError):
    """No prediction target remains; ``audit`` exposes the cleaning decisions."""
    def __init__(self, message, audit):
        super().__init__(message)
        self.audit = audit


@dataclass
class PreparedOutlets:
    outlets: pd.DataFrame
    matches: pd.DataFrame
    alternatives: pd.DataFrame
    pixel_mapping: pd.DataFrame
    audit: dict


def _canonical_columns(frame):
    """Accept official mixed-case attributes without renaming user columns."""
    if not isinstance(frame, pd.DataFrame):
        raise TypeError('outlets/basins must be pandas or GeoPandas tables')
    known = {'HYBAS_ID','COAST','UP_AREA','SUB_AREA','NEXT_DOWN','MAIN_BAS','DIST_MAIN'}
    ren = {c: str(c).upper() for c in frame.columns if str(c).upper() in known}
    names = [ren.get(c, c) for c in frame.columns]
    if len(names) != len(set(names)):
        raise ValueError('Ambiguous duplicate HydroBASINS attribute names')
    return frame.rename(columns=ren).copy()


def prepare_prediction_outlets(outlets, basins, uparea, *, config, match_config,
                                level=None) -> PreparedOutlets:
    """Clean and select physical outlets once, before any temporal calculation.

    ``duplicate_outlet_policy='best_glofas'`` interprets distinct points of a
    known COAST=0 basin as ALTERNATIVE locations of its outlet. Each location is
    matched against the SAME polygon UP_AREA. The accepted location with the
    lowest normalized log-area + distance score is selected. This is a plausible
    representative, not proof of the true river reach.

    COAST=1 aggregates must not be treated this way: they can drain through
    separate small catchments. Use coastal_policy='exclude' (default), or
    duplicate_outlet_policy='separate' with per-point areas for that application.
    Unknown COAST status is an error by default; an explicit keep policy retains
    a warning, never an invented value of zero.

    All discarded locations and all best-pixel scores are retained in ``audit``.
    If no acceptable point of a multiple-point basin is found, the whole target
    group is excluded (or an explicit error is raised). No arbitrary first-row
    selection and no nearest-pixel fallback are used.
    """
    config.validate()
    raw = _canonical_columns(outlets)
    b = _canonical_columns(basins)
    # A conflict in an existing COAST field must not be silently overwritten by
    # the polygon join. Other topological fields remain authoritative in basins.
    if 'COAST' in raw and 'COAST' in b and not b.HYBAS_ID.duplicated().any():
        reference = pd.to_numeric(raw.HYBAS_ID, errors='coerce').map(
            pd.Series(pd.to_numeric(b.COAST, errors='coerce').values,
                      index=pd.to_numeric(b.HYBAS_ID, errors='coerce')))
        own = pd.to_numeric(raw.COAST, errors='coerce')
        conflict = reference.notna() & own.notna() & reference.ne(own)
        if conflict.any():
            raise ValueError('COAST disagrees between outlet and basin layers')
    # Always normalize as separate FIRST: an error-on-multiple policy must not
    # interrupt before coastal groups have had a chance to be excluded.
    tab = outlet_table(raw, b, level=level, duplicate_policy='separate',
                       warn_multiple=config.duplicate_outlet_policy=='separate')
    exact = pd.DataFrame(tab.attrs.get('duplicate_outlet_audit', []))
    audit = dict(exact_duplicates=exact)
    stages = []
    def stage(name, frame):
        stages.append(dict(stage=name, n_outlets=int(len(frame)),
                           n_basins=int(frame.HYBAS_ID.nunique())))
    stage('input', raw)
    stage('after_exact_duplicates', tab)
    if tab.empty:
        audit['summary'] = pd.DataFrame(stages)
        raise NoUsableOutletsError('No outlet provided', audit)

    coast = pd.to_numeric(tab.get('COAST', pd.Series(np.nan,index=tab.index)), errors='coerce')
    if (~coast.isna() & ~coast.isin([0, 1])).any():
        raise ValueError('COAST must be 0, 1 or missing; do not infer from distance to the sea')
    unknown = coast.isna()
    if unknown.any() and config.coastal_policy == 'exclude':
        if config.missing_coast_policy == 'error':
            raise ValueError('COAST is missing for some outlets. Supply the HydroBASINS COAST '
                             'attribute, or explicitly set missing_coast_policy="keep"; '
                             'unknown coastal status is not COAST=0.')
        warnings.warn('Unknown COAST status retained explicitly; those points were not '
                      'verified non-coastal.', RuntimeWarning, stacklevel=2)
    tab['COAST'] = coast
    tab['coast_check_status'] = np.where(unknown, 'UNKNOWN', 'KNOWN')
    remove = coast.eq(1) if config.coastal_policy == 'exclude' else pd.Series(False,index=tab.index)
    audit['coastal_removed'] = tab.loc[remove].assign(cleaning_action='EXCLUDED_COAST_1').copy()
    tab = tab.loc[~remove].reset_index(drop=True)
    stage('after_coast_filter', tab)
    if tab.empty:
        audit['summary'] = pd.DataFrame(stages)
        raise NoUsableOutletsError('No target remains after excluding COAST=1. '
                                  'Inspect exception.audit["coastal_removed"].', audit)

    counts = tab.groupby('HYBAS_ID').HYBAS_ID.transform('size')
    multiple = counts.gt(1)
    if config.duplicate_outlet_policy == 'error' and multiple.any():
        raise ValueError('Distinct outlets remain for a HYBAS_ID after coastal filtering. '
                         'Use duplicate_outlet_policy="best_glofas" or "separate".')
    original_multiple = tab.multiple_outlet_basin.copy()
    tab['candidate_outlet_count'] = counts.astype(int)
    tab['source_outlet_id'] = tab.outlet_id.astype(str)
    tab['selection_area_source'] = tab.match_area_source
    tab['representative_area_assumption'] = False

    # For BEST, compare alternative locations on the common basin-area scale.
    # Do not select a tiny independent catchment merely because its own small
    # point-specific area gives a perfect match to a small river pixel.
    if config.duplicate_outlet_policy == 'best_glofas':
        if multiple.any() and float(match_config.area_weight) <= 0:
            raise ValueError('best_glofas requires a positive area_weight')
        admissible_basin = multiple & tab.COAST.eq(0)
        parent = pd.to_numeric(tab.get('UP_AREA', pd.Series(np.nan,index=tab.index)), errors='coerce')
        good_parent = admissible_basin & np.isfinite(parent) & parent.gt(0)
        tab.loc[good_parent,'MATCH_UP_AREA_KM2'] = parent.loc[good_parent]
        tab.loc[good_parent,'match_area_source'] = 'HYBAS_UP_AREA_candidate_reference'
        tab.loc[good_parent,'selection_area_source'] = 'common_HYBAS_UP_AREA'
        tab.loc[good_parent,'outlet_match_status'] = 'OK'
        tab.loc[good_parent,'representative_area_assumption'] = True
        # Kept coastal or unknown multiple groups cannot share the aggregate area.
        bad = multiple & ~good_parent
        tab.loc[bad,'MATCH_UP_AREA_KM2'] = np.nan
        tab.loc[bad,'outlet_match_status'] = np.where(
            tab.loc[bad,'COAST'].eq(1),'COAST_MULTI_OUTLET_NOT_REDUCIBLE',
            np.where(tab.loc[bad,'COAST'].isna(),'UNKNOWN_COAST_MULTI_OUTLET',
                     'INVALID_TARGET_AREA'))

    targets = tab.assign(target_lon=tab.X, target_lat=tab.Y,
        HYBAS_UP_AREA_KM2=tab.get('UP_AREA',np.nan),
        target_coordinate_source='hybas_outlet', station_hybas_status=tab.outlet_match_status)
    matches, alternatives = match_hybas_outlets_to_glofas(targets, uparea, config=match_config)
    matches['outlet_id'] = tab.outlet_id.values
    if len(alternatives):
        alternatives['outlet_id'] = alternatives.target_row.map(dict(enumerate(tab.outlet_id)))
    audit['candidate_pixels'] = alternatives.copy()
    # Independent copy: its original target_row identifies pre-selection rows.
    candidate_audit = matches.copy()
    candidate_audit['selected'] = False
    candidate_audit['selection_action'] = 'NOT_SELECTED'
    candidate_audit['selected_outlet_id'] = ''
    rows, selected_indices, groups = [], [], []
    failures = []
    for hid, g in tab.groupby('HYBAS_ID',sort=True):
        idx = g.index.tolist()
        local = matches.loc[idx]
        winner = None
        if config.duplicate_outlet_policy != 'best_glofas' or len(idx) == 1:
            chosen = idx
            action = 'KEPT_SINGLE' if len(idx)==1 else 'KEPT_SEPARATE'
            margin, ambiguous = np.nan, False
        else:
            valid = local.match_status.isin(match_config.accepted_match_statuses)
            valid &= np.isfinite(pd.to_numeric(local.match_score,errors='coerce'))
            eligible = local.loc[valid].sort_values(
                ['match_score','distance_km','target_lon','target_lat','outlet_id'],
                kind='mergesort')
            if eligible.empty:
                chosen = []
                action = 'NO_ACCEPTABLE_POINT'
                margin, ambiguous = np.nan, False
                failures.append(int(hid))
                candidate_audit.loc[idx,'selection_action'] = action
            else:
                winner = eligible.index[0]
                chosen = [winner]
                scores = eligible.match_score.to_numpy(float)
                margin = float(scores[1]-scores[0]) if len(scores)>1 else np.nan
                ambiguous = bool(len(scores)>1 and margin <= 1e-10)
                action = 'SELECTED_BEST_GLOFAS'
                candidate_audit.loc[idx,'selection_action'] = 'REJECTED_ALTERNATIVE'
        for i in chosen:
            row = tab.loc[i].copy()
            if config.duplicate_outlet_policy == 'best_glofas':
                row['outlet_id'] = str(int(hid))
                row['outlet_point_count'] = 1
                # Non-coastal alternatives are represented by ONE chosen point.
                # Preserve coastal flags for single kept coastal points.
                if len(idx)>1:
                    row['multiple_outlet_basin'] = False
                    row['outlet_identity_status'] = 'REPRESENTATIVE_OUTLET_SELECTED'
                    row['outlet_identity_method'] = 'basin_after_best_glofas_selection'
            row['selection_action'] = action
            row['selection_score_margin'] = margin
            row['selection_ambiguous'] = ambiguous
            row['original_multiple_outlet_flag'] = bool(original_multiple.loc[i])
            rows.append(row)
            selected_indices.append(i)
            candidate_audit.loc[i,['selected','selection_action','selected_outlet_id']] = (
                True,action,str(row.outlet_id))
        groups.append(dict(HYBAS_ID=int(hid), n_candidates=len(idx), n_selected=len(chosen),
            selection_action=action, selected_source_outlet_id=';'.join(tab.loc[chosen,'source_outlet_id']),
            score_margin=margin, selection_ambiguous=ambiguous))
    audit['candidates'] = candidate_audit
    audit['selection'] = pd.DataFrame(groups)
    selected = pd.DataFrame(rows).reset_index(drop=True) if rows else tab.iloc[:0].copy()
    stage('after_representative_selection', selected)
    audit['summary'] = pd.DataFrame(stages)
    if failures:
        message = f'No admissible representative for {len(failures)} multiple-point HYBAS group(s): {failures[:12]}'
        if config.multi_no_match_policy == 'error':
            raise NoUsableOutletsError(message, audit)
        warnings.warn(message + '. Groups excluded; see outlet_cleaning["candidates"].',
                      RuntimeWarning,stacklevel=2)
    if selected.empty:
        raise NoUsableOutletsError('No target remains after outlet selection', audit)
    if not selected.outlet_id.is_unique:
        raise ValueError('Selected outlet_id keys are not unique')
    if config.duplicate_outlet_policy == 'best_glofas' and selected.HYBAS_ID.duplicated().any():
        raise AssertionError('Representative selection did not produce one point per HYBAS_ID')

    # Reindex the selected matches AND their pixel alternatives to the final
    # catalogue. Reuse these results; there is no second area-based search.
    sm = matches.loc[selected_indices].copy().reset_index(drop=True)
    sm['source_outlet_id'] = sm.outlet_id.astype(str)
    sm['outlet_id'] = selected.outlet_id.to_numpy()
    for col in ('selection_action','outlet_identity_status','outlet_identity_method',
                'outlet_point_count','multiple_outlet_basin','candidate_outlet_count',
                'original_multiple_outlet_flag','selection_score_margin','selection_ambiguous'):
        sm[col] = selected[col].to_numpy()
    sm['target_row'] = np.arange(len(sm),dtype=int)
    alt = alternatives.copy()
    if len(alt):
        new_index = dict(zip(selected_indices,range(len(selected))))
        new_keys = dict(zip(selected_indices,selected.outlet_id))
        alt = alt[alt.target_row.isin(selected_indices)].copy()
        alt['source_outlet_id'] = alt.outlet_id.astype(str)
        alt['outlet_id'] = alt.target_row.map(new_keys)
        alt['target_row'] = alt.target_row.map(new_index).astype(int)
        alt = alt.sort_values(['target_row','candidate_rank']).reset_index(drop=True)
    mp = pd.DataFrame(dict(outlet_id=sm.outlet_id,
        source_x=sm.glofas_lon, source_y=sm.glofas_lat,
        status=np.where(sm.match_status.isin(match_config.accepted_match_statuses),
                        'OK',sm.match_status)))
    # Formatted audits stay outside .attrs (NetCDF serialization remains safe).
    selected.attrs = {}
    LOG.info('HYBAS targets: %d input -> %d exact-unique -> %d non-excluded -> %d selected',
             len(raw),stages[1]['n_outlets'],stages[2]['n_outlets'],len(selected))
    return PreparedOutlets(selected,sm,alt,mp,audit)
