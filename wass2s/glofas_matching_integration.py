"""Validation at the boundary between a frozen mapping and predictand creation."""
from __future__ import annotations
import copy
import warnings
import numpy as np
import pandas as pd
if __package__:
    from .glofas_static_network import exact_subset
else:
    from glofas_static_network import exact_subset


def matching_for_application(matching,q,up,basins,outlets,config,match_config,station_attributes,monthly_config):
    matching.validate_for(q,basins,outlets,config,match_config,station_attributes)
    # Native coordinate equality alone cannot detect a different auxiliary map.
    if __package__:
        from .highflow_utils import area_to_km2
    else:
        from highflow_utils import area_to_km2
    actual=area_to_km2(up)
    expected=exact_subset(matching.network.uparea,q,matching.network.config.coordinate_tolerance)
    if not np.allclose(actual.values,expected.values,rtol=1e-10,atol=1e-10,equal_nan=True):
        raise ValueError('UPAREA values differ from the frozen matching network')
    if monthly_config is not None:
        if __package__:
            from .highflow_monthly_bias import reference_training_mask
        else:
            from highflow_monthly_bias import reference_training_mask
        if set(monthly_config.excluded_stations)-set(matching.context.excluded_stations):
            raise ValueError('Monthly exclusions were not excluded during matching; prepare a new shared context')
        # Compare GLOBAL exclusions only. LOMO cannot retrospectively turn a
        # fixed all-years regime selection into an out-of-sample selection.
        allowed=matching.context.mask(pd.DatetimeIndex(q['T'].values))
        monthly_allowed=reference_training_mask(pd.DatetimeIndex(q['T'].values),monthly_config)
        if np.any(allowed & ~monthly_allowed):
            raise ValueError('The frozen matching used dates forbidden by monthly outer exclusions/cutoff')
        used=matching.selected_pixels.get('regime_used',pd.Series(False)).fillna(False).any()
        if used and monthly_config.temporal_mode=='expanding' and matching.context.information_cutoff is None:
            raise ValueError('Regime matching + expanding correction requires a fixed information_cutoff per issue/fold')
        if used and monthly_config.temporal_mode in {'leave_month_out','leave_year_out'}:
            warnings.warn('Monthly exclusions protect factors only; pixel selection stays conditional on its fixed training context. '
                          'Use explicit external blocks to validate the complete chain.',RuntimeWarning,stacklevel=2)
    out=copy.copy(matching)
    out.prepared_outlets=copy.deepcopy(matching.prepared_outlets)
    out.station_matches=matching.station_matches.copy()
    out.station_metadata=matching.station_metadata.copy()
    if matching.config.require_reviewed:
        for table in (out.prepared_outlets.matches,out.station_matches):
            if len(table):
                unreviewed=~table.review_status.eq('ACCEPTED_REVIEWED')
                table.loc[unreviewed,'match_status']='REVIEW_REQUIRED'
        sm=out.prepared_outlets.matches
        out.prepared_outlets.pixel_mapping['status']=np.where(
            sm.match_status.isin(match_config.accepted_match_statuses),'OK',sm.match_status)
    if matching.config.require_reviewed and not out.prepared_outlets.pixel_mapping.status.eq('OK').any():
        raise ValueError('No reviewed admissible outlet remains. Inspect mapping and apply explicit review decisions first.')
    if not out.prepared_outlets.pixel_mapping.status.eq('OK').any():
        raise ValueError('No admissible outlet mapping remains. Inspect selected_pixels, candidates and network_checks before constructing targets.')
    return out
