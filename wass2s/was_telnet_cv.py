"""Bridge the annual WASS2S validator to TelNet's monthly sequence inputs."""
from __future__ import annotations
import numpy as np
import xarray as xr
from .was_probability_calibration import normalize_probability_calibration


def cross_validate_telnet(cv, model, target, predictor, clim_start, clim_end, options):
    """Preserve annual test partitions while purging monthly sequence overlap."""
    if not isinstance(predictor, dict) or not {'observations','covariates'} <= set(predictor):
        raise ValueError("TelNet Predictor must be {'observations': monthly_T_Y_X, "
                         "'covariates': monthly_T_feature}; a seasonal predictor cannot recover monthly sequences")
    if 'M' in target.dims:
        if target.sizes['M'] != 1:
            raise ValueError('TelNet verification observations must have singleton M')
        target=target.isel(M=0,drop=True)
    if set(target.dims) != {'T','Y','X'}:
        raise ValueError('TelNet verification target requires T,Y,X')
    opts=dict(options)
    if not opts.pop('strict_oos_probability',True):
        raise ValueError('TelNet provides a native predictive distribution; diagnostic residual dressing is not applicable')
    reference=opts.pop('tercile_reference',getattr(cv,'tercile_reference','climatology'))
    if reference!='climatology':
        raise ValueError('TelNet WASS2S uses fixed declared climatological terciles')
    calibration=normalize_probability_calibration(opts.pop('probability_calibration','outer'))
    # Native empirical members require no additional residual calibration.
    for key in ('probability_inner_folds','probability_inner_retune'):
        opts.pop(key,None)
    if opts.pop('nested_tuning',False):
        raise ValueError('TelNet architecture is prespecified; choose its settings independently or validate a research search inside each training partition')
    splits=list(cv.custom_cv.split(target['T'],cv.nb_omit))
    opts.setdefault('gap_years',int(np.ceil(cv.nb_omit/2)))
    det,prob=model.cross_validate(predictor['observations'],predictor['covariates'],
        target_times=target['T'].values,clim_year_start=clim_start,
        clim_year_end=clim_end,splits=splits,probability_calibration=calibration,**opts)
    dataset=model.hindcast_dataset_
    observed=dataset['observation'] if 'observation' in dataset else None
    if observed is not None and set(observed.dims)=={'T','Y','X'}:
        expected,actual=xr.align(target.transpose('T','Y','X'),observed,join='exact')
        comparable=np.isfinite(expected)&np.isfinite(actual)
        if bool(comparable.any()) and not bool((abs(expected-actual)<=1e-5+1e-5*abs(actual)).where(comparable,True).all()):
            raise ValueError('TelNet monthly observations aggregate to different verification values than Predictant; use one observation source and unit convention')
        cv.verification_target_=actual
    else:
        cv.verification_target_=target.transpose('T','Y','X')
    cv.cross_validation_diagnostics_=getattr(model,'cross_validation_diagnostics_',None)
    cv.telnet_model_=model
    return det,prob
