"""WASS2S validation with operational outer and research nested calibration.

Outer residual reuse is the operational default; ``inner`` aliases ``nested``.
Model selection remains a separate training-only control. Explicit
strict_oos_probability=False requests diagnostic calibration.

Provides a leave-one-out splitter with a configurable exclusion window and
a high-level orchestrator that dispatches each fold to the correct
preprocessing path for every supported model family.

Classes
-------
CustomTimeSeriesSplit
    Symmetric-window leave-one-out splitter; prevents near-neighbour
    temporal leakage.
WAS_Cross_Validator
    Orchestrates cross-validation for CCA, Kernel CCA, PCR, linear models,
    machine-learning models, MME methods, and analog ensembles.

Helper functions
----------------
_pcr_isel_T
    ``isel(T=idx)`` for a DataArray or list/tuple of DataArrays.
_pcr_T_index
    Return the ``T`` coordinate used to drive the splitter.
"""
from wass2s.was_crossfit import WAS_CrossFit, SplitPlan
from wass2s.was_scientific_utils import align_predictor_target, apply_physical_support
from wass2s.was_probability_reference import climatological_terciles
from wass2s.was_probability_calibration import normalize_probability_calibration

import hashlib
import inspect
from copy import copy, deepcopy
import os
import warnings

import numpy as np
import xarray as xr
from tqdm.auto import tqdm
from wass2s.was_linear_models import *
from wass2s.was_eof import *
from wass2s.was_pcr import *
from wass2s.was_cca import *
from wass2s.was_cca import _field_fingerprint, _resolve_pca_modes, _nonnegative_support
from wass2s.was_cca_2 import *
from wass2s.was_machine_learning import *
from wass2s.was_analog import *
from wass2s.utils import *
from wass2s.was_transformdata import *
from wass2s.was_bias_correction import *
from wass2s.was_mme import *
from wass2s.was_mme import _model_covariates, _regional_params_from


def _diagnostic_probabilities(model, *args, **kwargs):
    """Label historical calibration truthfully instead of promising nested CV."""
    warnings.warn('This model family uses diagnostic full-hindcast probability calibration. '
                  'Use WAS_CrossFit/WASModelSpec for training-only nested calibration.',
                  RuntimeWarning, stacklevel=3)
    parameters=inspect.signature(model.compute_prob).parameters
    if 'diagnostic' in parameters or any(p.kind==inspect.Parameter.VAR_KEYWORD for p in parameters.values()):
        kwargs['diagnostic']=True
    result = model.compute_prob(*args, **kwargs)
    if isinstance(result, xr.DataArray):
        result.attrs['calibration'] = 'diagnostic full-hindcast pool; not strict outer-fold calibration'
    return result



def _calibration_mode(options, model):
    """Normalize the two validation modes without hiding diagnostic overrides."""
    supplied = options.get('probability_calibration')
    if not options.get('strict_oos_probability', True):
        if supplied is not None and str(supplied).strip().lower() != 'diagnostic':
            raise ValueError("strict_oos_probability=False requests diagnostic calibration; "
                             "do not combine it with probability_calibration='outer'/'nested'/'inner'")
        return 'diagnostic'
    return normalize_probability_calibration(supplied,
        default=getattr(model, 'probability_calibration', 'outer'))


def _hyperparameter_kwargs(model, result):
    """Map native search outputs to the exact estimator API, including logistic."""
    values = result if isinstance(result, tuple) else (result,)
    signature = inspect.signature(model.compute_model).parameters
    if isinstance(model, WAS_LogisticRegression_Model):
        if len(values) < 4:
            raise ValueError('Logistic tuning must return C, class-weight, max-iteration and solver maps')
        return dict(zip(('C_da', 'cw_code_da', 'maxiter_da', 'solver_da'), values[:4]))
    params = {}
    if 'hyperparams' in signature:
        params['hyperparams'] = values[0]
    if 'alpha' in signature:
        params['alpha'] = values[0]
    if 'l1_ratio' in signature:
        if len(values) < 2:
            raise ValueError('ElasticNet tuning must return alpha and l1_ratio maps')
        params['l1_ratio'] = values[1]
    return params


def _validate_fixed_parameters(model, params, tuning):
    """Fail before fitting when a fixed-configuration procedure is incomplete."""
    if tuning:
        return
    signature = inspect.signature(model.compute_model).parameters
    # The public API's first three positional arguments are training X,
    # training y and query X; wrappers may legitimately rename them. Bound
    # signatures omit self. Do not classify those data as hyperparameters.
    positional=[name for name, parameter in signature.items()
                if name != 'self' and parameter.kind in
                (inspect.Parameter.POSITIONAL_ONLY, inspect.Parameter.POSITIONAL_OR_KEYWORD)]
    data_names = set(positional[:3]) | {'self', 'y_test'}
    missing = [name for name, parameter in signature.items()
        if name not in data_names and parameter.kind not in
        (inspect.Parameter.VAR_POSITIONAL, inspect.Parameter.VAR_KEYWORD)
        and parameter.default is inspect.Parameter.empty and params.get(name) is None]
    if missing:
        raise ValueError('nested_tuning=False requires prespecified fixed parameters for '
                         f'{type(model).__name__}: {", ".join(missing)}. '
                         'Use parameters chosen independently of the evaluated observations, '
                         'or enable nested_tuning.')


def _calibration_metadata(probability, mode, options, *, native=False):
    """Record what actually ran, separately from hyperparameter tuning."""
    probability.attrs['probability_calibration'] = mode
    probability.attrs['nested_tuning'] = int(bool(options.get('nested_tuning', True)))
    probability.attrs['probability_inner_retune'] = int(options.get('probability_inner_retune', True) is not False)
    if native:
        probability.attrs['probability_calibration_requested'] = mode
        probability.attrs['probability_calibration'] = 'native'
        probability.attrs['probability_calibration_effective'] = 'native_training_fit'
        probability.attrs['residual_calibration'] = 'not_applicable_native_distribution'
        probability.attrs['residual_calibration_applicable'] = 0
        probability.attrs['calibration_strictly_nested'] = 1
    elif mode == 'nested':
        reused = bool(options.get('_calibration_hyperparameters_reused', False))
        probability.attrs['calibration_strictly_nested'] = int(not reused)
        if reused:
            probability.attrs['calibration'] += '; hyperparameter configuration reused from enclosing outer training'
    else:
        probability.attrs['calibration_strictly_nested'] = 0


def _pcr_isel_T(P, idx):
    """isel(T=idx) for a DataArray or a list/tuple of DataArrays."""
    if isinstance(P, dict):
        return {k:v.isel(T=idx) for k,v in P.items()}
    if isinstance(P, (list, tuple)):
        return [p.isel(T=idx) for p in P]
    return P.isel(T=idx)


def _pcr_T_index(P):
    """The T coordinate used to drive the splitter (first field if a list)."""
    return (P[0] if isinstance(P, (list, tuple)) else P)["T"]



def _cv_target(da):
    """Return a strict (T,Y,X) observation field without fragile squeeze/drop_vars."""
    if "M" in da.dims:
        if da.sizes["M"] != 1:
            raise ValueError("Observation member dimension M must be singleton")
        da=da.isel(M=0,drop=True)
    return da.transpose("T","Y","X")


def _signature_kwargs(func, params):
    """Route only real callable parameters; **kwargs accepts the remainder."""
    sig=inspect.signature(func)
    names=set(sig.parameters)-{"self"}
    has_var=any(p.kind==inspect.Parameter.VAR_KEYWORD for p in sig.parameters.values())
    return dict(params) if has_var else {k:v for k,v in params.items() if k in names}


def _fold_standardize(train_da,test_da,clim_start,clim_end):
    """Standardize with statistics estimated from TRAIN only."""
    clim=train_da
    if clim_start is not None and clim_end is not None:
        q=train_da.sel(T=slice(str(clim_start),str(clim_end)))
        if q.sizes.get("T",0)>=2:
            clim=q
    mean=clim.mean("T",skipna=True)
    std=clim.std("T",skipna=True)
    std=xr.where(np.isfinite(std) & (std == 0), 1.0, std)
    return (train_da-mean)/std,(test_da-mean)/std,mean,std


def _inner_splits_from_outer(train_T,nb_omit=1,max_folds=8):
    """Thinned symmetric-window LOO splits for nested model selection."""
    n=int(train_T.sizes.get("T",len(train_T)))
    if n<6: return []
    cv=CustomTimeSeriesSplit(n_splits=n)
    all_s=list(cv.split(train_T,nb_omit))
    if len(all_s)<=max_folds: return all_s
    step=max(1,int(np.ceil(len(all_s)/max_folds)))
    return all_s[::step][:max_folds]


def _inner_block_splits(train_T, n_folds=5, nb_omit=1):
    """Complete blocked inner coverage with the same calendar-year buffer."""
    if len(train_T) < 6:
        return []
    try:
        plan = SplitPlan.build(train_T, mode='blocked',
            gap_years=int(np.ceil(nb_omit/2)), n_blocks=n_folds, min_train=5)
    except ValueError as exc:
        if 'Insufficient' not in str(exc):
            raise
        return []
    return [(np.asarray(f.train), np.asarray(f.test)) for f in plan.folds]


def _class_from_training_climatology(y_train,clim_start,clim_end,thresholds=None):
    """Class labels under the declared reference; fitting still uses TRAIN only."""
    terc=thresholds
    if terc is None:
        terc,_=climatological_terciles(y_train,clim_start,clim_end)
    t1=terc.isel(quantile=0,drop=True);t2=terc.isel(quantile=1,drop=True)
    valid=np.isfinite(y_train)&np.isfinite(t1)&np.isfinite(t2)
    return xr.where(y_train<=t1,0,xr.where(y_train<=t2,1,2)).where(valid),(t1,t2)


def _climatology_options(cv,y,clim_start,clim_end,options):
    """Compute the event reference once, independently of prediction fitting."""
    options=dict(options)
    mode=options.get('tercile_reference',getattr(cv,'tercile_reference','climatology'))
    if mode not in ('climatology','training'):
        raise ValueError("tercile_reference must be 'climatology' or 'training'")
    options['tercile_reference']=mode
    external=options.pop('obs_for_terciles',None)
    if mode=='training':
        if external is not None:
            raise ValueError('obs_for_terciles requires tercile_reference="climatology"')
        return options
    reference=_cv_target(y if external is None else external)
    reference=xr.align(reference,_cv_target(y).isel(T=0,drop=True),join='exact')[0]
    q,ref=climatological_terciles(reference,clim_start,clim_end)
    if clim_start is not None or clim_end is not None:
        reference=reference.sel(T=slice(None if clim_start is None else str(clim_start),
                                       None if clim_end is None else str(clim_end)))
    options.update(_probability_thresholds=q,_probability_reference=ref,
                   _probability_reference_sample=reference)
    return options


def _event_thresholds(train,cfg,options):
    q=options.get('_probability_thresholds')
    if q is not None:
        return q,options['_probability_reference']
    return climatological_terciles(train,cfg.climatology_start,cfg.climatology_end,
                                  min_count=cfg.min_oof)


def _fixed_event_diagnostics(cv,target,options):
    """Expose fixed category boundaries to verification even for classifiers."""
    if options.get('_probability_thresholds') is not None:
        cv.cross_validation_diagnostics_=xr.Dataset({
            'thresholds':options['_probability_thresholds'].expand_dims(T=target['T']),
            'reference_probabilities':options['_probability_reference'].expand_dims(T=target['T'])})
        cv.cross_validation_diagnostics_.attrs['tercile_reference']='climatology'


def _maybe_align_predictor_time(Predictor, Predictant):
    return align_predictor_target(Predictor, Predictant)


def _finalize_det(hindcast,target,mask):
    h=xr.concat(hindcast,dim="T").sortby("T").reindex(T=target["T"])
    h=h.transpose("T","Y","X")*mask
    return apply_physical_support(h,target)


# Classes routed to the pooled-regression MME branch of cross_validate(), resolved
# by NAME in this module's globals (``from wass2s.was_mme import *``). Any WAS_mme_*
# regressor added to was_mme.py and to the CLI factory must be listed here, or carry
# ``_was_cv_family = "mme_regression"``; otherwise cross_validate() has no branch.
MME_REGRESSION_CV_NAMES = (
    "WAS_mme_MLP", "WAS_mme_hpELM", "WAS_mme_XGBoosting", "WAS_mme_RF",
    "WAS_mme_PCR", "WAS_mme_Ridge", "WAS_mme_ElasticNet", "WAS_mme_PLS",
    "WAS_mme_SVR", "WAS_mme_MARS", "WAS_mme_ExtraTrees",
    "WAS_mme_HistGradientBoosting",
)


def _validate_ensemble_predictors(model, predictors):
    """Keep per-GCM distribution summaries on the predictor axis during CV.

    These summaries are computed within each issued ensemble, independently of
    observations and other years. Their temporal standardization still belongs
    to each training fold. Physical ensemble combinations cannot average a
    rainfall mean, spread and quantile as if they were interchangeable members.
    """
    fields = predictors.values() if isinstance(predictors, dict) else [predictors]
    statistics = []
    for field in fields:
        if not isinstance(field, xr.DataArray):
            continue
        marker = field.attrs.get('wass2s_ensemble_mode') == 'statistics'
        statistic = field.coords.get('ensemble_statistic')
        if statistic is not None:
            marker = marker or bool(np.any(np.asarray(statistic).astype(str) != 'mean'))
        if not marker:
            continue
        if set(field.dims) != {'T', 'M', 'Y', 'X'}:
            raise ValueError('Ensemble statistics require predictor dimensions T,M,Y,X; '
                             'M contains features and must not be stacked into T')
        for coord in ('M', 'source_model', 'ensemble_statistic'):
            if coord not in field.coords or field[coord].dims != ('M',):
                raise ValueError(f'Ensemble statistics require coordinate {coord!r} on M')
        if field.sizes['M'] == 0 or field.get_index('M').has_duplicates:
            raise ValueError('Ensemble feature channel labels M must be nonempty and unique')
        statistics.append(field)
    if not statistics:
        return
    supported = (any(isinstance(model, globals().get(name, ()))
                    for name in MME_REGRESSION_CV_NAMES if name != 'WAS_mme_PCR')
        or getattr(model, '_was_cv_family', None) == 'mme_regression'
        or getattr(model, '_was_spatial_cnn', False)
        or getattr(model, '_was_unet', False)
        or getattr(model, '_was_rescnn', False)
        or getattr(model, '_was_unetpp', False)
        or getattr(model, '_was_rise_unet', False))
    if not supported:
        raise ValueError(f'{type(model).__name__} does not support per-GCM ensemble '
            'statistics as predictor channels in this cross-validation path. '
            'Use ensemble_mean="mean" for methods that combine physical forecasts.')


# ---------------------------------------------------------------------------
# CCA cross-validation helpers
#
# Every transformation is refitted on the training fold. Fitting detrended_data
# or standardize_timeseries on the full record and reusing its coefficients to
# rebuild the hindcast lets the fit see the validation year: negligible on
# rainfall, material on temperature, where the trend carries most of the
# variance.
# ---------------------------------------------------------------------------
def _cca_field(da):
    """Drop the member dimension and return a (T, Y, X) field."""
    if "M" in da.dims:
        da = da.isel(M=0, drop=True)
    return da.transpose("T", "Y", "X")


def _split_cca_params(model, model_params):
    """Route kwargs by real signatures, not by ``__code__.co_varnames``.

    ``co_varnames`` also lists a function's local variables, so the previous
    split routed anything named like a local (``y_pred``, ``X_test_prepared``)
    to the wrong side.
    """
    all_params = dict(model_params)
    truncation = all_params.pop("truncation", None) or {}

    model_args = set(inspect.signature(model.compute_model).parameters) - {"self"}
    prob_args = set(inspect.signature(model.compute_prob).parameters) - {"self"}

    params_models = {k: v for k, v in all_params.items() if k in model_args}
    params_prob = {k: v for k, v in all_params.items() if k in prob_args}

    unknown = set(all_params) - model_args - prob_args
    if unknown:
        warnings.warn(f"Ignoring unknown model_params: {sorted(unknown)}.",
                      RuntimeWarning, stacklevel=3)
    return params_models, params_prob, truncation


def _cca_finalize(hindcast, Predictant_obs, mask):
    """Align on the observed T axis and apply the sign-aware clip."""
    hindcast = hindcast.sortby("T").reindex(T=Predictant_obs["T"])
    hindcast = hindcast.transpose("T", "Y", "X") * mask
    return apply_physical_support(hindcast, Predictant_obs)


def _cca_inner_splits(n_train, nb_omit=1, max_folds=10):
    """Thinned leave-one-out inner folds, for nested truncation selection."""
    idx = np.arange(n_train)
    step = max(1, int(np.ceil(n_train / max_folds)))
    out = []
    for i in range(0, n_train, step):
        te = idx[i: i + max(1, int(nb_omit))]
        tr = np.setdiff1d(idx, te)
        if tr.size >= 5 and te.size >= 1:
            out.append((tr, te))
    return out


def _cca_tune(model, X_fit, Y_fit, splits, truncation):
    """Select the truncations once, before the outer loop (CPT-like)."""
    if not truncation:
        return
    kwargs = dict(truncation)
    kwargs.pop("nested", None)
    kwargs.pop("inner_max_folds", None)
    res = model.tune(X_fit, Y_fit, splits, **kwargs)
    # One short line, deliberately. The selection is part of the model
    # configuration and belongs in the run log even when nobody reads it;
    # WAS_CCA_QUIET_TUNE=1 silences it.
    if os.environ.get("WAS_CCA_QUIET_TUNE", "0") != "1":
        print(f"  truncation: n_modes={res['n_modes']} n_pca_modes={res['n_pca_modes']} "
              f"{res['metric']}={res['score']:.4f} ({res['rule']})", flush=True)


def _cca_mdl_options(truncation):
    """Return training-only MDL options, or None for the historical CV path."""
    if not truncation or str(truncation.get('method', 'cv')).lower() != 'mdl':
        return None
    return {k: v for k, v in truncation.items()
            if k not in ('method', 'nested', 'inner_max_folds')}


def _cca_unfitted_copy(model):
    """Copy MDL configuration without copying a previous GCM's fitted arrays.

    CLI instances are reused across domains and models. Their EOF engine and
    plotting adapter can hold full spatial fields, none of which is needed
    when a new training fold is about to be fitted.
    """
    unfitted = copy(model)
    for name in ('cca', 'cca_model', '_engine', '_engine_modes', '_prep',
                 '_x_train_ref', '_y_train_ref', '_x_valid_mask', '_y_valid_mask', '_mdl_baseline',
                 'baseline_hindcast_', 'truncation_', 'truncation_folds_',
                 'truncation_diagnostics_folds_'):
        if hasattr(unfitted, name):
            setattr(unfitted, name, None)
    unfitted._truncation_cache = {}
    return deepcopy(unfitted)


def _cca_cv_eeof(model, Predictant, Predictor, splits, clim_start, clim_end,
                 params_models, truncation):
    """Fit the EEOF decomposition only on each training fold.

    At unseen dates the training reference field is used, as in the explicit
    WASModelSpec adapter. A temporal EEOF component at a held-out date cannot
    be recovered from the held-out observation.
    """
    parts, fold_truncations, fold_diagnostics = [], [], []
    mdl_options = _cca_mdl_options(truncation)
    for tr, te in splits:
        fitted = _cca_unfitted_copy(model) if mdl_options is not None else deepcopy(model)
        ya, yb = Predictant.isel(T=tr), Predictant.isel(T=te)
        yy, _, mean, scale = _fold_standardize(ya, yb, clim_start, clim_end)
        xx, _, refx = fitted._eeof_split(Predictor.isel(T=tr))
        yy, _, refy = fitted._eeof_split(yy)
        if mdl_options is not None:
            pred = fitted.compute_model_mdl(xx, yy, Predictor.isel(T=te)-refx,
                                            **mdl_options)
            fold_truncations.append((fitted.n_modes, tuple(_resolve_pca_modes(fitted.n_pca_modes))))
            fold_diagnostics.append(deepcopy(fitted.truncation_))
        elif truncation:
            inner = _inner_splits_from_outer(ya['T'])
            if not inner:
                raise ValueError('Insufficient inner CCA folds for truncation tuning')
            _cca_tune(fitted, xx, yy, inner, truncation)
            fold_truncations.append((fitted.n_modes, tuple(_resolve_pca_modes(fitted.n_pca_modes))))
        if mdl_options is None:
            pred = fitted.compute_model(xx, yy, Predictor.isel(T=te)-refx, **params_models)
        pred = pred.assign_coords(T=yb['T']).transpose('T','Y','X')
        parts.append((pred+refy)*scale+mean)
    if truncation:
        model.truncation_folds_ = fold_truncations
    if mdl_options is not None:
        model.truncation_diagnostics_folds_ = fold_diagnostics
    return xr.concat(parts, dim='T').sortby('T').reindex(T=Predictant['T'])


def _cca_cv_base(model, Predictant, Predictor, splits, clim_start, clim_end,
                 params_models, truncation):
    """No detrending; target scaling and truncation use each TRAIN only."""
    parts, fold_truncations, fold_diagnostics = [], [], []
    mdl_options = _cca_mdl_options(truncation)
    for tr, te in splits:
        fitted = _cca_unfitted_copy(model) if mdl_options is not None else deepcopy(model)
        ya, yb = Predictant.isel(T=tr), Predictant.isel(T=te)
        yy, _, mean, scale = _fold_standardize(ya, yb, clim_start, clim_end)
        xx = Predictor.isel(T=tr)
        if mdl_options is not None:
            pred = fitted.compute_model_mdl(xx, yy, Predictor.isel(T=te),
                                            **mdl_options)
            fold_truncations.append((fitted.n_modes, tuple(_resolve_pca_modes(fitted.n_pca_modes))))
            fold_diagnostics.append(deepcopy(fitted.truncation_))
        elif truncation:
            inner = _inner_splits_from_outer(ya['T'])
            if not inner:
                raise ValueError('Insufficient inner CCA folds for truncation tuning')
            _cca_tune(fitted, xx, yy, inner, truncation)
            fold_truncations.append((fitted.n_modes, tuple(_resolve_pca_modes(fitted.n_pca_modes))))
        if mdl_options is None:
            pred = fitted.compute_model(xx, yy, Predictor.isel(T=te), **params_models)
        pred = pred.assign_coords(T=yb['T']).transpose('T','Y','X')
        parts.append(pred*scale+mean)
    if truncation:
        model.truncation_folds_ = fold_truncations
    if mdl_options is not None:
        model.truncation_diagnostics_folds_ = fold_diagnostics
    return xr.concat(parts, dim='T').sortby('T').reindex(T=Predictant['T'])


def _cca_cv_detrended(model, Predictant, Predictor, splits, clim_start, clim_end,
                      params_models, truncation, strict=False, nb_omit=1):
    """Op/strict CCA with fold-local scaling and optional detrending.

    Disabling the temporal trend preserves each class's spatial preparation;
    it must not route a strict model through the separate base-CCA path.
    """
    # No grid (autotuning off, or the truncation of the enclosing outer fold
    # imposed on its calibration blocks) means the instance's own modes.
    truncation = truncation or {}
    mdl_options = _cca_mdl_options(truncation)
    nested = mdl_options is None and bool(truncation) and bool(truncation.get("nested", True))
    inner_max_folds = int(truncation.get("inner_max_folds", 10))
    detrend = bool(getattr(model, "detrend", True))

    if nested:
        print(f"  nested truncation selection: about {len(splits)} x |grid| x "
              f"{inner_max_folds} extra fits", flush=True)
    elif truncation and mdl_options is None:
        X_fit = model._fill_spatial_gaps(Predictor) if strict else Predictor
        Y_st = standardize_timeseries(Predictant, clim_start, clim_end)
        Y_fit = Y_st.where(np.isfinite(Y_st))
        if detrend:
            X_fit, _, _ = detrended_data(X_fit, dim="T")
            Y_fit, _, _ = detrended_data(Y_fit, dim="T")
        _cca_tune(model, X_fit.fillna(0.0), Y_fit.fillna(0.0), splits, truncation)

    base_modes, base_pca = model.n_modes, model.n_pca_modes
    fold_truncations, fold_diagnostics = [], []
    hindcast_list = []
    # Référence "climatologie + tendance optionnelle" : la prévision obtenue avec une
    # anomalie CCA nulle, construite avec les mêmes plis, les mêmes coeffs_Y
    # et la même dé-standardisation que le hindcast lui-même. Sert à séparer
    # l'apport du prédicteur de la tendance commune à tous les domaines.
    baseline_list = []

    for train_index, test_index in tqdm(splits, total=len(splits)):
        X_train_raw = Predictor.isel(T=train_index)
        X_test_raw = Predictor.isel(T=test_index)
        y_train_raw = Predictant.isel(T=train_index)

        if strict:
            X_train_raw = model._fill_spatial_gaps(X_train_raw)
            X_test_raw = model._fill_spatial_gaps(X_test_raw, ref=X_train_raw)

        X_train_ready, y_train_ready, X_test_ready, coeffs_Y, meta_Y = model.prepare_fold(
            X_train_raw, y_train_raw, X_test_raw, clim_start, clim_end)
        X_train_ready = X_train_ready.fillna(0.0)
        X_test_ready = X_test_ready.fillna(0.0)
        y_train_ready = y_train_ready.fillna(0.0)

        if mdl_options is not None:
            # Constructor values are upper bounds for every independent fit;
            # a small selected rank in one fold must not shrink later searches.
            model.n_modes, model.n_pca_modes = base_modes, base_pca
            pred = model.compute_model_mdl(X_train_ready, y_train_ready, X_test_ready,
                                           **mdl_options)
            fold_truncations.append(
                (model.n_modes, tuple(_resolve_pca_modes(model.n_pca_modes))))
            fold_diagnostics.append(deepcopy(model.truncation_))
        elif nested:
            model.n_modes, model.n_pca_modes = base_modes, base_pca
            kwargs = dict(truncation)
            kwargs.pop("nested", None)
            kwargs.pop("inner_max_folds", None)
            kwargs.setdefault("progress", False)
            model.tune(X_train_ready, y_train_ready,
                       _inner_splits_from_outer(y_train_raw['T'], nb_omit=nb_omit,
                                                max_folds=inner_max_folds),
                       **kwargs)
            fold_truncations.append(
                (model.n_modes, tuple(_resolve_pca_modes(model.n_pca_modes)))
            )

        if mdl_options is None:
            pred = model.compute_model(X_train_ready, y_train_ready, X_test_ready,
                                       **params_models)
        pred = pred.assign_coords(T=Predictant["T"].isel(T=test_index))
        # Only an enabled trend is restored. A zero standardized anomaly is
        # the training-fold climatology when detrending is disabled.
        trend_at_test = (apply_detrend_data(pred, coeffs_Y, meta_Y)
                         if detrend else xr.zeros_like(pred))
        pred = pred + trend_at_test
        pred = reverse_standardize(pred, y_train_raw, clim_start, clim_end)
        hindcast_list.append(pred.transpose("T", "Y", "X"))

        # Une anomalie CCA nulle donne la climatologie du pli, augmentée de
        # la tendance uniquement si elle est activée. Le même champ de
        # référence est réutilisé pour garder exactement la même échelle.
        baseline = reverse_standardize(trend_at_test, y_train_raw,
                                       clim_start, clim_end)
        baseline_list.append(baseline.transpose("T", "Y", "X"))

    if nested or mdl_options is not None:
        model.n_modes, model.n_pca_modes = base_modes, base_pca
        # Retain the training-only choices for stability diagnostics. MDL
        # makes this choice from each training spectrum without any selection
        # sub-folds; the CV method uses its nested validation grid.
        model.truncation_folds_ = fold_truncations
        if mdl_options is not None:
            model.truncation_diagnostics_folds_ = fold_diagnostics
        # _cca_cross_validate calls this one outer split at a time and prints
        # the stability over all outer folds itself; a 1/1 line says nothing.
        if len(fold_truncations) > 1:
            counts = {}
            for item in fold_truncations:
                counts[item] = counts.get(item, 0) + 1
            top, n = max(counts.items(), key=lambda kv: kv[1])
            label = 'MDL' if mdl_options is not None else 'nested'
            print(f"  {label} selection stability: {n}/{len(fold_truncations)} folds "
                  f"chose n_modes={top[0]} n_pca_modes={list(top[1])} "
                  f"({len(counts)} distinct)", flush=True)

    hindcast = xr.concat(hindcast_list, dim="T").sortby("T")
    # Exposé en attribut plutôt qu'en valeur de retour : la signature de
    # cross_validate est utilisée partout et ne doit pas changer. Même idiome
    # que model.truncation_folds_ ci-dessus.
    if baseline_list:
        model.baseline_hindcast_ = (
            xr.concat(baseline_list, dim="T").sortby("T")
            .reindex(T=Predictant["T"])
        )
    return hindcast.reindex(T=Predictant["T"])


def _check_cca_selection(value):
    value = str(value).strip().lower()
    if value not in ('cv', 'aic'):
        raise ValueError("truncation_selection must be 'cv' or 'aic'.")
    return value


def _cca_cross_validate(cv, model, Predictant, Predictor, clim_start, clim_end,
                        model_params):
    """CCA driver with outer-training-only residual calibration.

    Every fold works on a copy of ``model``. What the calling instance gets
    back when the truncation is autotuned: ``truncation_folds_`` (the choice
    of each outer fold, for the stability diagnostic), ``truncation_`` and the
    fingerprint cache (the final selection on the complete record, made by
    ``model.finalize_truncation`` with the same outer splits and the same
    fold transformations), which ``forecast`` applies for these fields. The
    constructor ``n_modes`` / ``n_pca_modes`` are left untouched.

    With ``mode_selection='mdl'``, each training fit selects its ranks from
    the same decompositions used for prediction. No mode-selection sub-folds
    are built; independent probability-calibration folds remain unchanged.
    """
    from .was_crossfit import CrossFitConfig, _thresholds, _probabilities
    options = dict(model_params)
    strict_prob = bool(options.pop('strict_oos_probability', True))
    blocks = int(options.pop('probability_inner_folds', 5))
    # probability_inner_retune=False: the calibration blocks of an outer fold
    # reuse the truncation selected on that outer training fold instead of
    # re-running the whole selection (x probability_inner_folds cost). The
    # default (True) repeats it, the strictly nested estimate.
    inner_retune = options.pop('probability_inner_retune', True) is not False
    # 'aic': Bartlett-corrected AIC on each training fold instead of the inner
    # cross-validated grid search (see WAS_CCA*.tune); default: the model's own.
    selection = options.pop('truncation_selection', None)
    # 'outer': approximate reuse of other folds; not strictly nested calibration.
    calibration = _calibration_mode({**options, 'strict_oos_probability': strict_prob}, model)
    probability_options={k:v for k,v in options.items() if k in _PROBABILITY_CONTROLS or k in _DISTRIBUTION_MAP_KEYS}
    params_models, params_prob, truncation = _split_cca_params(model, {k:v for k,v in options.items() if k not in _PROBABILITY_CONTROLS})
    y, x = _cca_field(Predictant), _cca_field(Predictor)
    if not truncation and getattr(model, 'autotune_enabled', lambda: False)():
        truncation = model.default_truncation_grid()
    mdl_options = _cca_mdl_options(truncation)
    if mdl_options is not None and isinstance(model, (WAS_CCA_Regularized, WAS_CCA_Regularized_strict)):
        raise ValueError('MDL mode selection requires the unregularized exact CCA engine; '
                         'use mode_selection="cv" with a regularized CCA')
    if mdl_options is not None and getattr(model, 'mode_selection', 'cv') != 'mdl':
        raise ValueError('MDL truncation requires mode_selection="mdl" on the CCA model '
                         'so hindcast and forecast use the same selection method')
    if truncation and mdl_options is None and getattr(model, 'mode_selection', 'cv') == 'mdl':
        raise ValueError('A CV truncation grid requires mode_selection="cv"; '
                         'an MDL model accepts truncation={"method": "mdl", ...} only')
    if truncation and mdl_options is None and truncation.get('nested', True) is False:
        raise ValueError('Full-record CCA truncation selection is not a validation procedure; '
                         'pass fixed constructor modes with autotuning disabled, or nested=True')
    cfg = CrossFitConfig(climatology_start=clim_start, climatology_end=clim_end)
    splits = list(cv.custom_cv.split(x['T'], cv.nb_omit))
    # Same non-negativity rule as forecast (declared support, not observed
    # minimum) for the calibrator's exponential family.
    model.positive_support = _nonnegative_support(y)
    if selection is not None:
        model.truncation_selection = _check_cca_selection(selection)
        if mdl_options is not None and model.truncation_selection != 'cv':
            raise ValueError('MDL and an AIC truncation override cannot be combined')
    base = _cca_unfitted_copy(model) if mdl_options is not None else deepcopy(model)
    fold_truncations, fold_diagnostics = [], []
    outer_folds = []

    def compute(xx, yy, ss, record=False, fixed=None):
        fitted = deepcopy(base)
        trunc = truncation
        if fixed is not None:
            # truncation of the enclosing outer fold, no new selection
            fitted.n_modes, fitted.n_pca_modes = fixed[0], list(fixed[1])
            trunc = None
        if isinstance(fitted, (WAS_CCA_strict, WAS_CCA_op)):
            out = _cca_cv_detrended(fitted, yy, xx, ss, clim_start, clim_end,
                params_models, trunc, strict=isinstance(fitted,WAS_CCA_strict), nb_omit=cv.nb_omit)
        else:
            fun = _cca_cv_eeof if isinstance(fitted, WAS_CCA) else _cca_cv_base
            out = fun(fitted, yy, xx, ss, clim_start, clim_end, params_models, trunc)
        if record:
            fold_truncations.extend(getattr(fitted, 'truncation_folds_', None) or [])
            fold_diagnostics.extend(getattr(fitted, 'truncation_diagnostics_folds_', None) or [])
        return out

    dets, probs, qs, refs = [], [], [], []
    for tr, te in tqdm(splits, total=len(splits)):
        n_recorded = len(fold_truncations)
        pred = compute(x, y, [(tr,te)], record=True).isel(T=te)
        pred = apply_physical_support(pred, y.isel(T=tr))
        dets.append(pred)
        if strict_prob and calibration == 'outer':
            outer_folds.append((tr, pred))
            continue
        if strict_prob:
            ya, xa = y.isel(T=tr), x.isel(T=tr)
            inner = _inner_block_splits(ya['T'], blocks, cv.nb_omit)
            if not inner:
                raise ValueError('Insufficient history for nested CCA probability calibration')
            outer_choice = (fold_truncations[n_recorded]
                            if truncation and not inner_retune and len(fold_truncations) > n_recorded
                            else None)
            ih = compute(xa, ya, inner, fixed=outer_choice)
            prob,q,ref,_ = _training_probabilities(model,pred,ya-ih,ya,cfg,probability_options)
            probs.append(prob)
            qs.append(q.expand_dims(T=pred['T']))
            refs.append(ref.expand_dims(T=pred['T']))
    hd = xr.concat(dets,dim='T').sortby('T').reindex(T=y['T'])
    if outer_folds:
        errors = (y - hd.transpose(*y.dims))
        for tr, pred in outer_folds:
            ya = y.isel(T=tr)
            residuals = errors.sel(T=ya['T'].values)
            prob,q,ref,_ = _training_probabilities(model,pred,residuals,ya,cfg,probability_options)
            probs.append(prob)
            qs.append(q.expand_dims(T=pred['T']))
            refs.append(ref.expand_dims(T=pred['T']))
    if truncation:
        # Per-fold choices on the caller's instance (stability diagnostic),
        # then the final selection on the complete record so that forecast
        # emits the truncation family the hindcast was validated with.
        model.truncation_folds_ = fold_truncations
        if mdl_options is not None:
            model.truncation_diagnostics_folds_ = fold_diagnostics
        if fold_truncations:
            counts = {}
            for item in fold_truncations:
                counts[item] = counts.get(item, 0) + 1
            top, n = max(counts.items(), key=lambda kv: kv[1])
            print(f"  outer-fold selection stability: {n}/{len(fold_truncations)} folds "
                  f"chose n_modes={top[0]} n_pca_modes={list(top[1])} "
                  f"({len(counts)} distinct)", flush=True)
        model.finalize_truncation(x, y, clim_start, clim_end, splits=splits,
                                  **{k:v for k,v in truncation.items()
                                     if k not in ('nested','inner_max_folds')})
    if strict_prob:
        hp = xr.concat(probs,dim='T').sortby('T').reindex(T=y['T']).transpose('probability','T','Y','X')
        cv.cross_validation_diagnostics_ = xr.Dataset({
            'thresholds':xr.concat(qs,dim='T').reindex(T=y['T']),
            'reference_probabilities':xr.concat(refs,dim='T').reindex(T=y['T'])})
        hp.attrs['calibration'] = ('approximate reuse of other outer-fold residuals; '
            'calibration models may include the verification year' if outer_folds else
            'training-only nested residuals and marginal fit; declared empirical terciles')
        hp.attrs['calibration_strictly_nested'] = int(not outer_folds)
        hp.attrs['tercile_reference'] = options.get('tercile_reference','training')
    else:
        hp = _diagnostic_probabilities(model,y, clim_start, clim_end, hd, **params_prob)
        hp.attrs['calibration'] = 'diagnostic full-hindcast residual pool'
    _calibration_metadata(hp, calibration, {**options, 'probability_inner_retune': inner_retune,
        '_calibration_hyperparameters_reused': bool(strict_prob and calibration=='nested'
            and not inner_retune and truncation and fold_truncations)})
    return hd, hp


_PROBABILITY_CONTROLS = {"strict_oos_probability", "probability_inner_folds",
    "probability_fit", "probability_distribution_map", "probability_fit_mode",
    "probability_calibration", "tercile_reference", "_probability_thresholds",
    "_probability_reference", "_probability_reference_sample", "obs_for_terciles"}
_DISTRIBUTION_MAP_KEYS = ("best_code_da", "best_shape_da", "best_loc_da", "best_scale_da")


_FOLD_LOCAL_WARNED = False


def _fold_local_distribution_options(options):
    """Discard every full-period fit, including its learned family selection.

    The candidate catalog must be declared independently through
    ``probability_distribution_map`` or ``probability_fit``. Otherwise the
    model's count/continuous type and declared physical support determine a
    fixed catalog. A full-period best-code map cannot define this catalog:
    its family selection has itself inspected verification observations.
    """
    global _FOLD_LOCAL_WARNED
    if not any(options.get(k) is not None for k in _DISTRIBUTION_MAP_KEYS):
        return options
    options = dict(options)
    for key in _DISTRIBUTION_MAP_KEYS:
        options.pop(key, None)
    if not _FOLD_LOCAL_WARNED:
        warnings.warn('Full-period best-fit maps, including their selected family codes, '
                      'are ignored in cross-validation. Families and parameters are fitted '
                      'inside each training fold from an independently declared or fixed default catalog.',
                      RuntimeWarning, stacklevel=3)
        _FOLD_LOCAL_WARNED = True
    return options


def _default_distribution_catalog(model, reference):
    """A-priori families based only on model type and declared variable support."""
    if isinstance(model, (WAS_PoissonRegression, WAS_NegativeBinomial_Model,
                          WAS_ZINB_Model, WAS_HurdleNB_Model)):
        return {'poisson': 7, 'nbinom': 8}
    nonnegative = bool((apply_physical_support(xr.DataArray(-1.), reference) == 0).item())
    if nonnegative:
        return {'norm': 1, 'lognorm': 2, 'expon': 3, 'gamma': 4,
                'weibull_min': 5, 't_dist': 6}
    return {'norm': 1, 't_dist': 6}


_BESTFIT_MEMORY = {}


def _bestfit_key(reference, distribution_map, mode):
    digest = hashlib.blake2b(digest_size=20)
    digest.update(np.ascontiguousarray(np.asarray(reference.values, dtype=np.float64)).tobytes())
    for dim in reference.dims:
        digest.update(np.asarray(reference[dim].values).astype(str).tobytes())
    digest.update(repr(sorted((distribution_map or {}).items())).encode())
    digest.update(str(mode).encode())
    return digest.hexdigest()


def _cached_best_distribution(reference, distribution_map, mode):
    """Fit best distributions once per identical training reference.

    The fit depends only on the training observations, candidate families and
    mode, never on the regression method: all methods of a run (and, with
    WAS_BESTFIT_CACHE_DIR, of a campaign) share it without any leakage.
    """
    from .was_transformdata import WAS_TransformData
    key = _bestfit_key(reference, distribution_map, mode)
    if key in _BESTFIT_MEMORY:
        return _BESTFIT_MEMORY[key]
    directory = os.environ.get('WAS_BESTFIT_CACHE_DIR')
    path = None
    if directory:
        from pathlib import Path
        import pickle
        path = Path(directory) / f'bestfit_{key}.pkl'
        if path.is_file():
            try:
                with path.open('rb') as source:
                    maps = pickle.load(source)
                _BESTFIT_MEMORY[key] = maps
                return maps
            except Exception:
                pass
    maps = tuple(m.load() if hasattr(m, 'load') else m for m in WAS_TransformData(
        reference, distribution_map=distribution_map).fit_best_distribution(mode=mode)[:4])
    if len(_BESTFIT_MEMORY) >= 64:
        _BESTFIT_MEMORY.pop(next(iter(_BESTFIT_MEMORY)))
    _BESTFIT_MEMORY[key] = maps
    if path is not None:
        import pickle
        path.parent.mkdir(parents=True, exist_ok=True)
        temporary = path.with_suffix(f'.{os.getpid()}.tmp')
        with temporary.open('wb') as output:
            pickle.dump(maps, output, protocol=pickle.HIGHEST_PROTOCOL)
        os.replace(temporary, path)
    return maps


def _training_probabilities(model, pred, residuals, train, cfg, options):
    """Calibrate a new forecast exclusively from outer-training observations.

    Empirical dressing preserves its historic meaning. ``bestfit`` fits the
    marginal family on the outer-training reference, and uses the requested
    model's predictive-family CDF with inner-OOF residual variance. Global
    fitted maps are prohibited; a custom fitter receives only that reference.
    """
    from .was_crossfit import _thresholds, _probabilities
    from .was_cca import _DISTRIBUTION_CODES
    method = str(getattr(model, 'dist_method', 'nonparam')).lower()
    residuals=residuals.where(np.isfinite(residuals))
    q, ref = _event_thresholds(train, cfg, options)
    if method == 'nonparam':
        return _probabilities(pred, residuals, q, train, cfg.min_oof), q, ref, None
    reference = train
    if cfg.climatology_start is not None:
        reference = train.sel(T=slice(str(cfg.climatology_start),str(cfg.climatology_end)))
    if method == 'bestfit':
        options = _fold_local_distribution_options(options)
        fitter = options.get('probability_fit')
        if fitter is None:
            catalog=options.get('probability_distribution_map')
            if catalog is None:
                catalog=_default_distribution_catalog(model,train)
            if not isinstance(catalog,dict) or not catalog:
                raise ValueError('probability_distribution_map must be a nonempty a-priori family dictionary')
            maps = _cached_best_distribution(reference,catalog,
                options.get('probability_fit_mode', 'grid'))
        else:
            if not callable(fitter):
                raise TypeError('probability_fit must be a training-only callable')
            maps = fitter(reference.copy(deep=True))
            if isinstance(maps, dict):
                maps = tuple(maps[k] for k in _DISTRIBUTION_MAP_KEYS)
        if len(maps) != 4:
            raise ValueError('probability_fit must return code, shape, loc, scale')
        code, shape, loc, scale = xr.align(*maps, reference.isel(T=0,drop=True),join='exact')[:4]
        # The marginal family is fitted on TRAIN, but its fitted quantiles
        # never replace the empirical boundaries of the declared climatology.
        q1,q2=q.isel(quantile=0,drop=True),q.isel(quantile=1,drop=True)
    elif method in _DISTRIBUTION_CODES:
        code=xr.full_like(train.isel(T=0,drop=True),_DISTRIBUTION_CODES[method],dtype=float)
        q1,q2=q.isel(quantile=0,drop=True),q.isel(quantile=1,drop=True)
    else:
        raise ValueError(f'Unknown probability distribution {method!r}')
    variance=residuals.var('T',ddof=1,skipna=True)
    residual_count=residuals.notnull().sum('T')
    degrees=xr.where(residual_count>4,residual_count-1,3)
    kwargs={}
    kernel=getattr(model,'calculate_tercile_probabilities_bestfit',None)
    if kernel is None:
        # classes whose probabilities are those of the equal-weight MME
        kernel=WAS_mme_Weighted.calculate_tercile_probabilities_bestfit
    if 'nonneg' in inspect.signature(kernel).parameters:
        kwargs['nonneg']=bool((apply_physical_support(xr.DataArray(-1.),train)==0).item())
    calibrated_mean=apply_physical_support(pred+residuals.mean('T'),train)
    def conditional_kernel(values,var,low,high,family,dof):
        return kernel(values,var,low,high,family,dof=float(dof),**kwargs)
    prob=xr.apply_ufunc(conditional_kernel,calibrated_mean,variance,q1,q2,code,degrees,
        input_core_dims=[['T'],[],[],[],[],[]],output_core_dims=[['probability','T']],
        vectorize=True,dask='parallelized',output_dtypes=[float],
        dask_gufunc_kwargs={'output_sizes':{'probability':3},'allow_rechunk':True})
    prob=prob.assign_coords(probability=['PB','PN','PA'])
    point=xr.concat([(calibrated_mean<=q1).astype(float),((calibrated_mean>q1)&(calibrated_mean<=q2)).astype(float),
        (calibrated_mean>q2).astype(float)],dim=xr.IndexVariable('probability',['PB','PN','PA']))
    prob=xr.where(variance==0,point,prob).where(
        np.isfinite(pred)&np.isfinite(calibrated_mean)&np.isfinite(q1)&np.isfinite(q2))
    total=prob.sum('probability',skipna=False)
    prob=(prob/total).where((total>0)&(residual_count>=cfg.min_oof))
    return prob.transpose('probability','T','Y','X'),q,ref,code


def _strict_fold_driver(cv, model, y, x, fit_predict, clim_start, clim_end, options):
    """A single calibration contract for all native continuous regressors."""
    from .was_crossfit import CrossFitConfig
    strict=bool(options.get('strict_oos_probability',True))
    blocks=int(options.get('probability_inner_folds',5))
    cfg=CrossFitConfig(climatology_start=clim_start,climatology_end=clim_end)
    fitting_options=options  # callbacks record actual configuration reuse here
    if strict and getattr(model,'dist_method','nonparam')=='bestfit':
        options=_fold_local_distribution_options(options)
    dets,probs,qs,refs,counts,codes=[],[],[],[],[],[]
    # Callbacks that accept ``inner=True`` learn whether a call is one of the
    # nested calibration fits (so they can honour probability_inner_retune).
    accepts_inner='inner' in inspect.signature(fit_predict).parameters
    inner_kwargs={'inner':True} if accepts_inner else {}
    calibration=_calibration_mode(options,model)
    outer=strict and calibration=='outer'
    folds=[]
    for tr,te in cv.custom_cv.split(y['T'],cv.nb_omit):
        xa,xb,ya=_pcr_isel_T(x,tr),_pcr_isel_T(x,te),y.isel(T=tr)
        pred=fit_predict(xa,ya,xb)
        dets.append(pred)
        if outer:
            folds.append((tr,te,pred))
            continue
        if strict:
            inner=_inner_block_splits(ya['T'],blocks,cv.nb_omit)
            if not inner:
                raise ValueError('Insufficient history for nested probability calibration')
            parts=[fit_predict(_pcr_isel_T(xa,a),ya.isel(T=a),_pcr_isel_T(xa,b),**inner_kwargs)
                   for a,b in inner]
            residuals=ya-xr.concat(parts,dim='T').sortby('T').reindex(T=ya['T'])
            prob,q,ref,code=_training_probabilities(model,pred,residuals,ya,cfg,options)
            probs.append(prob);qs.append(q.expand_dims(T=pred['T']))
            refs.append(ref.expand_dims(T=pred['T']))
            counts.append(residuals.notnull().sum('T').expand_dims(T=pred['T']))
            if code is not None: codes.append(code.expand_dims(T=pred['T']))
    hd=xr.concat(dets,dim='T').sortby('T').reindex(T=y['T']).transpose('T','Y','X')
    if outer:
        # Approximate calibration from other outer folds. Their fits may have
        # seen the current verification years; resulting probability scores
        # are not a strictly nested estimate. No size of the bias is assumed.
        errors=(y-hd).transpose('T','Y','X')
        for tr,te,pred in folds:
            ya=y.isel(T=tr)
            # the fold's training years (its test years and buffer excluded)
            residuals=errors.sel(T=ya['T'].values)
            prob,q,ref,code=_training_probabilities(model,pred,residuals,ya,cfg,options)
            probs.append(prob);qs.append(q.expand_dims(T=pred['T']))
            refs.append(ref.expand_dims(T=pred['T']))
            counts.append(residuals.notnull().sum('T').expand_dims(T=pred['T']))
            if code is not None: codes.append(code.expand_dims(T=pred['T']))
    if strict:
        hp=xr.concat(probs,dim='T').sortby('T').reindex(T=y['T']).transpose('probability','T','Y','X')
        diagnostics={'thresholds':xr.concat(qs,dim='T').reindex(T=y['T']),
            'reference_probabilities':xr.concat(refs,dim='T').reindex(T=y['T']),
            'calibration_count':xr.concat(counts,dim='T').reindex(T=y['T'])}
        if codes: diagnostics['distribution_code']=xr.concat(codes,dim='T').reindex(T=y['T'])
        cv.cross_validation_diagnostics_=xr.Dataset(diagnostics)
        hp.attrs['calibration'] = ('approximate reuse of other outer-fold residuals; '
            'calibration models may include the verification year' if outer else
            'training-only nested residuals and marginal fit; declared empirical terciles')
        hp.attrs['calibration_strictly_nested'] = int(not outer)
        hp.attrs['tercile_reference'] = options.get('tercile_reference','training')
    else:
        diagnostic_keys=set(_DISTRIBUTION_MAP_KEYS)|{'obs_for_terciles','tercile_reference',
            'probability_fit','probability_distribution_map','probability_fit_mode'}
        params={k:v for k,v in options.items() if k in diagnostic_keys}
        hp=_diagnostic_probabilities(model,y,clim_start,clim_end,hd,
                                      **_signature_kwargs(model.compute_prob,params))
    _calibration_metadata(hp,calibration,{**options,
        '_calibration_hyperparameters_reused':fitting_options.get('_calibration_hyperparameters_reused',False)})
    return hd,hp



def _pcr_parameter_maps(inner,candidate,template):
    """Encode one declared candidate for the original grid-model API."""
    values={k:v for k,v in candidate.items() if k!='n_modes'}
    if isinstance(inner,WAS_LogisticRegression_Model):
        return {'C_da':xr.full_like(template,values['C'],dtype=float),
            'cw_code_da':xr.full_like(template,inner._cw_map[values.get('class_weight')],dtype=float),
            'maxiter_da':xr.full_like(template,values['max_iter'],dtype=float),
            'solver_da':xr.full_like(template,values['solver'],dtype=object)}
    if hasattr(inner,'_param_names'):
        return {'hyperparams':xr.Dataset({k:xr.full_like(template,v,dtype=float) for k,v in values.items()})}
    return {k:xr.full_like(template,v,dtype=float) for k,v in values.items()}


def _pcr_prepare_inner_fold(model,x,y,train,test,clim_start,clim_end,classification=False,thresholds=None):
    """Fit EOF and target preprocessing before inspecting validation targets."""
    fitted=deepcopy(model)
    raw=y.isel(T=train)
    xa,xb=fitted._prepare_pcs(_pcr_isel_T(x,train),_pcr_isel_T(x,test))
    if classification:
        yy,(q1,q2)=_class_from_training_climatology(raw,clim_start,clim_end,thresholds)
        mean,scale=0.,1.
    else:
        yy,_,mean,scale=_fold_standardize(raw,raw,clim_start,clim_end)
        q1=q2=None
    return dict(model=fitted.reg_model,xa=xa,xb=xb,yy=yy,mean=mean,scale=scale,
        train_reference=raw,validation=y.isel(T=test),q1=q1,q2=q2)


def _pcr_candidate_predict(prepared,candidate):
    fitted=deepcopy(prepared['model'])
    params=_pcr_parameter_maps(fitted,candidate,prepared['train_reference'].isel(T=0,drop=True))
    signature=inspect.signature(fitted.compute_model).parameters
    if 'y_test' in signature:
        params['y_test']=xr.full_like(prepared['validation'],np.nan)
    if prepared['q1'] is not None:
        # Historical classifier kernels accept one query at a time. Keep the
        # complete validation block excluded while predicting its dates.
        pred=xr.concat([deepcopy(fitted).compute_model(prepared['xa'],prepared['yy'],
            prepared['xb'].isel(T=[i]),**params) for i in range(prepared['xb'].sizes['T'])],dim='T')
    else:
        pred=fitted.compute_model(prepared['xa'],prepared['yy'],prepared['xb'],**params)
    if 'T' not in pred.dims:pred=pred.expand_dims(T=prepared['xb']['T'])
    if 'probability' in pred.dims:return pred
    return apply_physical_support(pred*prepared['scale']+prepared['mean'],prepared['train_reference'])


def _tune_pcr_raw(model,x,y,clim_start,clim_end,nb_omit,options,classification=False):
    """Tune the original PCR estimator with inductive, cached inner EOF fits.

    Parameter candidates are pooled across the declared spatial domain. This
    avoids learning a target-derived clustering across validation observations.
    Grid/random/Optuna use the model's declared spaces and optimizer controls.
    """
    inner=model.reg_model
    folds=_inner_block_splits(y['T'],int(options.get('pcr_inner_folds',
        getattr(inner,'inner_cv',getattr(inner,'cv',3)))),nb_omit)
    if not folds:raise ValueError('Insufficient history for PCR full-pipeline hyperparameter selection')
    mode_grid=list(options.get('pcr_n_modes_grid',[model.n_modes]))
    explicit=options.get('pcr_candidates')
    if explicit is not None:
        mode_grid=list(dict.fromkeys(c.get('n_modes',model.n_modes) for c in explicit))
        if not explicit:raise ValueError('pcr_candidates cannot be empty')
    cache={}
    def prepared(mode):
        if mode not in cache:
            template=deepcopy(model);template.n_modes=mode
            cache[mode]=[_pcr_prepare_inner_fold(template,x,y,a,b,clim_start,clim_end,classification,
                options.get('_probability_thresholds'))
                         for a,b in folds]
        return cache[mode]
    records=[]
    def objective(candidate):
        mode=candidate.get('n_modes',model.n_modes)
        errors=[]
        for data in prepared(mode):
            pred=_pcr_candidate_predict(data,candidate)
            if classification:
                truth=data['validation'];q1,q2=data['q1'],data['q2']
                valid=np.isfinite(truth)&np.isfinite(q1)&np.isfinite(q2)
                cls=xr.where(truth<=q1,0,xr.where(truth<=q2,1,2)).where(valid)
                labels=list(pred.probability.values)
                if labels!=['PB','PN','PA']:pred=pred.assign_coords(probability=['PB','PN','PA'])
                target=xr.concat([(cls==i).astype(float) for i in range(3)],
                    dim=xr.IndexVariable('probability',['PB','PN','PA'])).where(valid)
                # cumsum() de xarray remplace les NaN par 0 : sans le masque
                # final, une cellule non predite ressort avec une erreur NULLE,
                # c'est-a-dire un score PARFAIT. Tant que tous les candidats
                # partagent le meme masque la dilution est identique et le
                # classement tient, mais un candidat qui echoue a predire
                # certaines cellules recolte des zeros gratuits et parait
                # meilleur -- exactement le candidat a ecarter. Meme garde que
                # was_crossfit._probabilistic_scores.
                scored=valid&np.isfinite(pred).all('probability')
                error=(((pred.cumsum('probability')-target.cumsum('probability'))**2)
                       .sum('probability',skipna=False).where(scored))
            else:error=(pred-data['validation'])**2
            errors.append(np.asarray(error).ravel())
        loss=np.concatenate(errors)
        score=float(np.nanmean(loss)) if np.isfinite(loss).any() else np.inf
        records.append((score,dict(candidate)))
        return score
    from sklearn.model_selection import ParameterGrid,ParameterSampler
    optimizer=getattr(inner,'hyperparam_optimizer',getattr(inner,'optimization_method','grid'))
    if explicit is not None:
        for candidate in explicit:objective(candidate)
    elif optimizer in {'bayesian','optuna'}:
        import optuna
        def optuna_objective(trial):
            if hasattr(inner,'_suggest_params'):
                candidate=inner._suggest_params(trial)
            elif hasattr(inner,'alpha_range'):
                alphas=np.asarray(inner.alpha_range,float)
                candidate={'alpha':trial.suggest_float('alpha',float(alphas.min()),float(alphas.max()),log=True)}
                if hasattr(inner,'l1_ratio_range'):
                    bounds=np.asarray(inner.l1_ratio_range,float)
                    candidate['l1_ratio']=trial.suggest_float('l1_ratio',float(bounds.min()),float(bounds.max()))
            elif hasattr(inner,'param_grid'):
                candidate={k:trial.suggest_categorical(k,list(v)) for k,v in inner.param_grid.items()}
            else:raise TypeError('No declared PCR parameter space; pass pcr_candidates')
            candidate['n_modes']=trial.suggest_categorical('n_modes',mode_grid)
            return objective(candidate)
        study=optuna.create_study(direction='minimize',sampler=optuna.samplers.TPESampler(
            seed=getattr(inner,'random_state',42)))
        study.optimize(optuna_objective,n_trials=int(getattr(inner,'n_trials',20)),show_progress_bar=False)
    else:
        if hasattr(inner,'alpha_range'):
            grid={'alpha':list(inner.alpha_range)}
            if hasattr(inner,'l1_ratio_range'):grid['l1_ratio']=list(inner.l1_ratio_range)
        elif hasattr(inner,'param_grid'):grid=dict(inner.param_grid)
        elif hasattr(inner,'param_space'):
            grid={name:sorted(set(np.geomspace(lo,hi,5) if log else np.linspace(lo,hi,5)))
                  for name,(lo,hi,integer,log) in inner.param_space.items()}
            grid={name:sorted(set(int(round(v)) for v in values)) if name in inner._int_params else values
                  for name,values in grid.items()}
        else:raise TypeError('No declared PCR parameter space; pass pcr_candidates')
        grid['n_modes']=mode_grid
        if optimizer in {'random','randomized'}:
            if hasattr(inner,'_param_distributions'):
                grid={**inner._param_distributions(),'n_modes':mode_grid}
            candidates=ParameterSampler(grid,n_iter=int(getattr(inner,'n_iter',getattr(inner,'n_trials',20))),
                random_state=getattr(inner,'random_state',42))
        else:candidates=ParameterGrid(grid)
        for candidate in candidates:
            if hasattr(inner,'_cast_params'):
                mode=candidate['n_modes']
                candidate={**inner._cast_params({k:v for k,v in candidate.items() if k!='n_modes'}),'n_modes':mode}
            objective(candidate)
    finite=[r for r in records if np.isfinite(r[0])]
    if not finite:raise ValueError('No finite PCR hyperparameter candidate')
    _,best=min(finite,key=lambda r:r[0])
    model.n_modes=best.get('n_modes',model.n_modes)
    model.pcr_tuning_={'scope':'fixed spatial domain','inner_folds':len(folds),
        'best':best,'scores':records,'preprocessing':'EOF and target scale fitted separately in each inner training'}
    return _pcr_parameter_maps(inner,best,y.isel(T=0,drop=True))


def _native_regression_cv(cv, model, y, x, clim_start, clim_end, options, family):
    """Rebuild state, tuning, EOF and scaling at EACH prediction/calibration fit."""
    y=_cv_target(y)
    options=dict(options)
    base=deepcopy(model)
    nested=bool(options.get('nested_tuning',True))
    controls=_PROBABILITY_CONTROLS|set(_DISTRIBUTION_MAP_KEYS)|{
        'nested_tuning','inner_max_folds','probability_inner_retune','truncation',
        'truncation_selection',
        'pcr_candidates','pcr_n_modes_grid','pcr_inner_folds'}
    fixed={k:v for k,v in options.items() if k not in controls}
    wrapped=isinstance(model,WAS_PCR)
    probability_model=model.reg_model if wrapped else model
    if wrapped:cv.pcr_tuning_history_=[]
    _validate_fixed_parameters(probability_model,fixed,nested)
    # probability_inner_retune=False: the hyperparameters selected on the
    # outer training fold are reused inside its calibration folds (zones and
    # covariates are still refitted fold-locally). Default True repeats the
    # whole HPO in every calibration fold (x probability_inner_folds cost).
    inner_retune=options.get('probability_inner_retune',True) is not False
    outer_state={}

    def fit_predict(xa,ya,xb,inner=False):
        if inner and not inner_retune and outer_state:
            options['_calibration_hyperparameters_reused']=True
        fitted=deepcopy(base)
        inner_model=fitted.reg_model if wrapped else fitted
        sig=inspect.signature(inner_model.compute_model).parameters
        params=_signature_kwargs(inner_model.compute_model,fixed)
        raw_target=family in {'count','kcca'}
        if raw_target:
            yy,mean,scale=ya,0.,1.
        else:
            yy,_,mean,scale=_fold_standardize(ya,ya,clim_start,clim_end)
        xx,xt=xa,xb
        if family in {'mme','mme_cca'}:
            xx,xt,_,_=_fold_standardize(xa,xb,clim_start,clim_end)
        pcr_tuned=False
        if wrapped:
            if nested and (hasattr(inner_model,'compute_hyperparameters') or options.get('pcr_candidates') is not None):
                reuse=outer_state.get('pcr') if (inner and not inner_retune) else None
                if reuse is not None:
                    fitted.n_modes=deepcopy(reuse['n_modes'])
                    params.update(deepcopy(reuse['parameters']))
                else:
                    selected=_tune_pcr_raw(fitted,xa,ya,clim_start,clim_end,cv.nb_omit,options)
                    params.update(selected)
                    cv.pcr_tuning_history_.append({'training_dates':ya['T'].values.copy(),
                        'calibration_fit':bool(inner), **deepcopy(fitted.pcr_tuning_)})
                    if not inner:
                        outer_state['pcr']={'n_modes':deepcopy(fitted.n_modes),
                                            'parameters':deepcopy(selected)}
                pcr_tuned=True
            xx,xt=fitted._prepare_pcs(xa,xb)
        if family=='kcca':
            if nested and not fitted._is_linear_kernel():
                reuse=outer_state.get('kcca') if (inner and not inner_retune) else None
                if reuse is not None:
                    params['best_params']=deepcopy(reuse)
                else:
                    params['best_params']=fitted.compute_hyperparameters(xa,ya,clim_start,clim_end)
                    if not inner:outer_state['kcca']=deepcopy(params['best_params'])
        elif family=='mme_cca':
            grid=dict(options.get('truncation') or {})
            if grid.get('nested',True) is False:
                raise ValueError('MME CCA truncation must be nested or fixed via nested_tuning=False')
            if nested:
                if not grid and fitted.autotune_enabled():grid=fitted.default_truncation_grid()
                reuse=outer_state.get('mme_cca') if (inner and not inner_retune) else None
                if grid and reuse is not None:
                    # truncation of the enclosing outer training fold
                    fitted.n_modes,fitted.n_pca_modes=reuse[0],list(reuse[1])
                elif grid:
                    selection=str(options.get('truncation_selection',
                        getattr(fitted,'truncation_selection','cv'))).lower()
                    if selection=='aic':
                        # information criterion on this training fold, no inner folds
                        res=_select_mme_cca_aic(fitted,xa,ya,clim_start,clim_end,grid)
                    else:
                        # Tune complete fold preprocessing, not outer-standardized
                        # matrices. The raw candidate procedure is repeated below.
                        res=_tune_mme_cca_raw(fitted,xa,ya,clim_start,clim_end,cv.nb_omit,grid)
                    if not inner:outer_state['mme_cca']=(res['n_modes'],res['n_pca_modes'])
        elif family=='mme' and isinstance(fitted,globals().get('WAS_mme_PCR',())):
            if nested and hasattr(fitted,'tune_modes'):
                reuse=outer_state.get('mme_pcr') if (inner and not inner_retune) else None
                if reuse is not None:
                    fitted.n_modes=reuse
                else:
                    res=_tune_mme_cca_raw(fitted,xa,ya,clim_start,clim_end,cv.nb_omit,
                        {'n_modes_grid':options.get('n_modes_grid',range(1,int(getattr(fitted,'_mode_search_upper',fitted.n_modes))+1)),
                         'inner_max_folds':int(options.get('inner_max_folds',8))},pcr=True)
                    if not inner:outer_state['mme_pcr']=res['n_modes']
        elif nested and not pcr_tuned and hasattr(inner_model,'compute_hyperparameters'):
            if family=='mme':
                reuse=outer_state.get('mme') if (inner and not inner_retune) else None
                if reuse is not None:
                    # Hyperparameters of the enclosing outer training fold;
                    # the zone map is refitted on THIS calibration fold.
                    params.update(_regional_params_from(inner_model,reuse,ya))
                else:
                    hp=inner_model.compute_hyperparameters(xa,ya,clim_start,clim_end)
                    if isinstance(hp,tuple):
                        params.update(best_params=hp[0],cluster_da=hp[1])
                        if not inner:outer_state['mme']=hp[0]
            else:
                reuse=outer_state.get('grid') if (inner and not inner_retune) else None
                if reuse is None:
                    hp=inner_model.compute_hyperparameters(ya,xx,clim_start,clim_end)
                    selected=_hyperparameter_kwargs(inner_model,hp)
                    if not inner:outer_state['grid']=deepcopy(selected)
                else:
                    # Reuse only the selected configuration, never a fitted
                    # estimator/scaler/EOF. Its spatial parameter map is a
                    # choice learned on the enclosing outer training sample.
                    selected=deepcopy(reuse)
                params.update(selected)
        if family=='mme':
            if (not nested and hasattr(inner_model,'compute_hyperparameters')
                    and (('best_params' in sig and 'cluster_da' in sig)
                         or any(p.kind==inspect.Parameter.VAR_KEYWORD for p in sig.values()))):
                # An empty declared candidate means estimator constructor
                # defaults, not permission for compute_model to start HPO.
                params.update(_regional_params_from(inner_model,params.get('best_params') or {},ya))
                params['cluster_da'].attrs['hyperparameter_scope']='prespecified_fixed_configuration'
            if 'clim_year_start' in sig:params['clim_year_start']=None
            if 'clim_year_end' in sig:params['clim_year_end']=None
            accepts_cov='covariates' in sig or any(p.kind==inspect.Parameter.VAR_KEYWORD for p in sig.values())
            if accepts_cov and getattr(inner_model,'spatial_mode','zones')!='zones':
                # Fold-local spatial covariates from the RAW training predictand
                # (compute_model receives the standardized target).
                params['covariates']=_model_covariates(inner_model,ya,clim_start,clim_end)
        time=_pcr_T_index(xb)
        blank=xr.full_like(ya.isel(T=[0]),np.nan).reindex(T=time)
        pred=inner_model.compute_model(xx,yy,xt,blank,**params)
        if 'T' not in pred.dims:pred=pred.expand_dims(T=time)
        pred=pred.assign_coords(T=time).transpose('T','Y','X')
        return apply_physical_support(pred*scale+mean,ya)
    return _strict_fold_driver(cv,probability_model,y,x,fit_predict,clim_start,clim_end,options)


def _convolutional_cross_validate(cv, model, y, x, clim_start, clim_end, options):
    """Validate U-Net/ResCNN with fresh networks and nested residual calibration.

    Only constructor inputs are cloned: a previously fitted network, learned
    standardizers and selected hyperparameters are never reused. Optional
    architecture selection is repeated from raw training fields in every outer
    and calibration fold. ``best_params`` must describe a prespecified candidate.
    The historical neural ``compute_prob`` API is intentionally not called.
    """
    from dataclasses import replace
    from .was_crossfit_adapters import WASModelSpec

    options = dict(options)
    if not options.get('strict_oos_probability', True):
        raise ValueError('Convolutional cross-validation requires training-only '
                         'probabilities; strict_oos_probability=False is unsupported')
    if x is None:
        raise ValueError('Convolutional cross-validation requires raw Predictor fields')
    y = _cv_target(y)
    fields = dict(x) if isinstance(x, dict) else {'predictor': x}
    nested = bool(options.get('nested_tuning', True))
    tune = nested and bool(getattr(model, 'tune_hyperparameters', False))
    fixed = options.get('best_params')
    if tune and fixed is not None:
        raise ValueError('Choose nested tuning or prespecified best_params, not both')
    if options.get('cluster_da') is not None:
        raise ValueError('Convolutional models do not use pre-fitted cluster maps')
    controls = _PROBABILITY_CONTROLS | set(_DISTRIBUTION_MAP_KEYS) | {
        'nested_tuning', 'probability_inner_retune', 'best_params', 'cluster_da'}
    unknown = set(options) - controls
    if unknown:
        raise TypeError(f'Unknown convolutional cross-validation controls: {sorted(unknown)}')
    spec = WASModelSpec.from_model(model, fields=tuple(fields),
        min_samples=max(5, int(model.min_valid_years)), seed=int(model.random_state))
    # Reject misspelled architecture controls rather than silently ignoring them.
    if fixed is not None:
        if not isinstance(fixed, dict):
            raise TypeError('best_params must be a prespecified parameter dictionary')
        if set(fixed) == {0} and isinstance(fixed[0], dict):
            fixed = fixed[0]
        unknown = set(fixed) - set(model._effective_params(None))
        if unknown:
            raise TypeError(f'Unknown convolutional parameters: {sorted(unknown)}')
    cv.convolutional_tuning_history_ = []
    outer_parameters = {}
    inner_retune = options.get('probability_inner_retune', True) is not False

    def fit_predict(xa, ya, xb, inner=False):
        params = deepcopy(fixed) if fixed is not None else {}
        reuse = inner and not inner_retune and 'selected' in outer_parameters
        if reuse:
            options['_calibration_hyperparameters_reused']=True
            params = deepcopy(outer_parameters['selected'])
        elif tune:
            # Adapter construction disables native tuning by design. Select
            # explicitly before fitting, using only the current training fold.
            selector = type(model)(*deepcopy(spec.args), **deepcopy(spec.init))
            selector.temporal_buffer = max(selector.temporal_buffer,
                                           int(np.ceil(cv.nb_omit / 2)))
            selector_adapter = spec.create()
            selector_adapter.schemas = {}
            raw = selector_adapter._input(xa, fit=True)
            params, _ = selector.compute_hyperparameters(raw, ya, clim_start, clim_end)
            if not inner:outer_parameters['selected'] = deepcopy(params)
        fit_spec = replace(spec, parameters=deepcopy(params))
        fitted = fit_spec.create().fit(xa, ya)
        pred = fitted.predict(xb)
        cv.convolutional_tuning_history_.append({
            'model': spec.model, 'training_dates': ya['T'].values.copy(),
            'prediction_dates': pred['T'].values.copy(), 'nested_tuning': tune,
            'best_params': deepcopy(fitted.model.best_params_),
        })
        return pred

    # Use the same residual-dressing/distribution kernels as the CLI and other
    # native MME regressors, retaining fold-specific thresholds and diagnostics.
    probability_model = WAS_mme_Weighted(equal_weighted=True, dist_method=model.dist_method)
    return _strict_fold_driver(cv, probability_model, y, fields, fit_predict,
                               clim_start, clim_end, options)


def _select_mme_cca_aic(model,x,y,clim_start,clim_end,grid):
    """Truncation of a WAS_mme_CCA* by the Bartlett-corrected AIC, without inner folds.

    Same candidates as :func:`_tune_mme_cca_raw` and the same fold
    preprocessing as the fit that follows (standardization on these training
    years, then the class's own ``_cca_problem``: multi-model field and
    detrending or EEOF residual). One exact decomposition scores every
    candidate (``was_cca_engine.bartlett_aic``). Applies the selection to
    ``model`` and returns an ``MMETruncationResult``.
    """
    from itertools import product
    from .was_cca_engine import ExactCCA, aic_truncation_table
    from .was_mme import _mme_safe_mode_counts
    pca=_resolve_pca_modes(model.n_pca_modes)
    candidates=[(int(m),int(px),int(py)) for m,px,py in product(
        grid.get('n_modes_grid',[model.n_modes]),grid.get('x_pca_grid',[pca[0]]),
        grid.get('y_pca_grid',[pca[1]]))]
    xa,_,_,_=_fold_standardize(x,x,clim_start,clim_end)
    ya,_,_,_=_fold_standardize(y,y,clim_start,clim_end)
    ftr,yfit,fte,_=model._cca_problem(xa,ya,xa)
    Xtr,Ytr,_=model._cca_inputs(ftr,yfit,fte)
    engine=ExactCCA(Xtr,Ytr,model.standardize,model.use_coslat)
    n=int(Xtr.sizes['T'])
    # Linear detrending consumes a degree of freedom beyond centering.
    # For the legacy EEOF residual, reserve at least one additional degree
    # as a conservative guard; this does not establish an exact IC likelihood.
    detrend_dof = int(bool(getattr(model, 'detrend', False) or
                           getattr(model, 'eeof_detrend', False)))
    rows=aic_truncation_table(engine,[(m,[px,py]) for m,px,py in candidates],n,
                              cap=_mme_safe_mode_counts, detrend_dof=detrend_dof)
    valid=[i for i,row in enumerate(rows) if np.isfinite(row[2])]
    if not valid:
        raise ValueError('No admissible finite MME-CCA AIC candidate: reduce the PCA '
                         'bounds or use more training years; saturated canonical '
                         'spaces are not valid information-criterion candidates.')
    order=sorted(valid,key=lambda i:(rows[i][2],candidates[i][0],
                                     candidates[i][1]+candidates[i][2]))
    chosen=order[0]
    modes,px,py=candidates[chosen]
    model.n_modes=int(modes); model.n_pca_modes=[px,py]
    table=[{'n_modes':c[0],'n_pca_x':c[1],'n_pca_y':c[2],'resolved':list(r[1]),'score':r[2]}
           for c,r in sorted(zip(candidates,rows),key=lambda cr:cr[1][2])]
    return MMETruncationResult(n_modes=int(modes),n_pca_modes=[px,py],score=float(rows[chosen][2]),
        se=np.nan,metric='bartlett_aic',rule='min',table=table,inner_folds=0)


def _tune_mme_cca_raw(model,x,y,clim_start,clim_end,nb_omit,grid,pcr=False,folds=None):
    """Rank CCA truncations after refitting the complete preprocessing chain.

    Applies the selection to ``model`` and returns it as an
    ``MMETruncationResult``. ``folds`` (explicit ``(train, test)`` index
    pairs, e.g. the outer splits for the final full-record selection of
    ``WAS_mme_CCA.finalize_truncation``) replaces the thinned inner LOO folds.
    """
    from itertools import product
    if folds is None:
        folds=_inner_splits_from_outer(y['T'],nb_omit,int(grid.get('inner_max_folds',8)))
    folds=[(np.asarray(a),np.asarray(b)) for a,b in folds]
    if not folds:raise ValueError('Insufficient history for MME CCA mode selection')
    pca=(None,None) if pcr else _resolve_pca_modes(model.n_pca_modes)
    candidates=list(product(grid.get('n_modes_grid',[model.n_modes]),
        grid.get('x_pca_grid',[pca[0]]),grid.get('y_pca_grid',[pca[1]])))
    scores=[]
    metric=grid.get('score','mse')
    if metric not in {'mse','pearson','correlation'}:
        raise ValueError('Raw nested MME CCA tuning supports mse or pearson')
    # This preparation is independent of the candidate. Share it only inside
    # this tuning call and fold: nothing is cached across validation folds.
    prepared=[]
    for a,b in folds:
        xa,xb,_,_=_fold_standardize(x.isel(T=a),x.isel(T=b),clim_start,clim_end)
        ya,_,mean,scale=_fold_standardize(y.isel(T=a),y.isel(T=a),clim_start,clim_end)
        prepared.append((xa.load(),xb.load(),ya.load(),mean.load(),scale.load(),
                         y.isel(T=b).load()))

    def loss_of(pred,fold):
        truth=fold[5]
        if pcr:
            # Define eligibility independently of the candidate's successful
            # predictions. High-rank models must not improve their score by
            # silently dropping cells with missing training targets. Partial
            # predictor fields remain usable under train-only EOF imputation;
            # an entirely unavailable test field is unavailable for every rank.
            test_field=model._field(fold[1])
            available=np.isfinite(test_field).any([d for d in test_field.dims if d!='T'])
            eligible=(np.isfinite(truth) & (np.isfinite(fold[2]).sum('T')>=3)
                      & available)
            if not bool(eligible.any()):return np.nan
            if bool((eligible & ~np.isfinite(pred)).any()):return np.inf
            pred,truth=pred.where(eligible),truth.where(eligible)
        if metric=='mse':return float(((pred-truth)**2).mean())
        va,vb=np.asarray(pred).ravel(),np.asarray(truth).ravel()
        valid=np.isfinite(va)&np.isfinite(vb)
        return -float(np.corrcoef(va[valid],vb[valid])[0,1]) if valid.sum()>2 else np.nan

    def candidate_loss(candidate, fold):
        modes,px,py=candidate
        xa,xb,ya,mean,scale,truth=fold
        fitted=deepcopy(model);fitted.n_modes=int(modes)
        if not pcr:fitted.n_pca_modes=[px,py]
        pred=fitted.compute_model(xa,ya,xb,xr.full_like(truth,np.nan))*scale+mean
        return loss_of(pred,fold)

    def fold_losses(fold):
        # One fold preprocessing (multi-model field, detrending) and one exact
        # PCA of each field serve every candidate; values equal those of
        # candidate_loss (the per-candidate refit it replaces).
        xa,xb,ya,mean,scale,truth=fold
        preds=deepcopy(model).cca_candidates(xa,ya,xb,[(int(m),[px,py]) for m,px,py in candidates])
        return [np.nan if isinstance(p,Exception) else loss_of(p*scale+mean,fold) for p in preds]

    from .was_hpc import ordered_tuning_map
    if not pcr and hasattr(model,'cca_candidates'):
        per_fold=ordered_tuning_map(fold_losses,[(fold,) for fold in prepared])
        scores=list(np.asarray(per_fold,dtype=float).T)
    else:
        losses=ordered_tuning_map(candidate_loss,
            [(candidate,fold) for candidate in candidates for fold in prepared])
        scores=list(np.asarray(losses).reshape(len(candidates),len(folds)))
    means=np.array([np.mean(v) for v in scores])
    if not np.isfinite(means).any():raise ValueError('No finite MME CCA mode candidate')
    best=int(np.nanargmin(means));chosen=best
    if grid.get('one_se',True):
        eligible=[]
        for i,loss in enumerate(scores):
            delta=loss-scores[best]
            se=float(delta.std(ddof=1)/np.sqrt(len(delta))) if len(delta)>1 else 0.
            if means[i]<=means[best]+se:eligible.append(i)
        chosen=min(eligible,key=lambda i:(candidates[i][0],0 if pcr else sum(candidates[i][1:])))
    modes,px,py=candidates[chosen]
    model.n_modes=int(modes)
    if not pcr:model.n_pca_modes=[px,py]
    table=sorted(({'n_modes':int(c[0]),'n_pca_x':c[1],'n_pca_y':c[2],'score':float(m)}
                  for c,m in zip(candidates,means) if np.isfinite(m)),key=lambda r:r['score'])
    return MMETruncationResult(n_modes=int(modes),n_pca_modes=None if pcr else [px,py],
        score=float(means[chosen]),se=np.nan,metric=metric,
        rule='best' if chosen==best else 'one-se-paired',table=table,inner_folds=len(folds))


def _analog_cross_validate(cv,model,y,x,clim_start,clim_end,options):
    """Analogue selection on TRAIN; native or explicit outer/inner dressing.

    ``outer`` reuses outer-fold errors after excluding the current test/buffer.
    Their models may have seen this verification year: the approximation is
    labelled. Native analogue probabilities need no residual calibration.
    """
    from .was_crossfit import CrossFitConfig
    from .was_crossfit_adapters import WASModelSpec
    if not hasattr(model,'_analog_member_cv_cache') and x is None:
        engine=getattr(model,'_scientific_engine',None)
        if callable(engine):
            model=engine()
    y=_cv_target(y)
    cfg=CrossFitConfig(climatology_start=clim_start,climatology_end=clim_end)
    if hasattr(model,'_analog_member_cv_cache'):
        options=dict(options)
        model.tercile_reference=options.get('tercile_reference',getattr(cv,'tercile_reference','climatology'))
        mode=_calibration_mode(options,model)
        native=getattr(model,'probability_method','') in ('analog','analog_ensemble','direct')
        inner_retune=options.get('probability_inner_retune',True) is not False
        cm={k:v for k,v in options.items() if k not in _PROBABILITY_CONTROLS
            and k not in _DISTRIBUTION_MAP_KEYS and k!='probability_inner_retune'}
        shared_raw=getattr(model,'_analog_raw_cache',None)
        base=deepcopy(model,{id(shared_raw):shared_raw} if shared_raw is not None else {})
        cache_names=('_analog_cv_cache','_analog_member_cv_cache','_regional_cv_cache')
        for key in cache_names:
            if hasattr(model,key):getattr(model,key).clear()
        # The declared reference defines events only. All prediction/selection
        # fits below receive observations masked outside their training dates.
        base._analog_event_reference_sample=options.get('_probability_reference_sample')
        base._analog_event_thresholds=options.get('_probability_thresholds')
        for key in ('_analog_reuse_selections','_analog_fixed_selection'):
            if hasattr(base,key):delattr(base,key)
        if hasattr(base,'_analog_feature_cache'):base._analog_feature_cache.clear()

        def one(train_positions,test_position,*,enclosing=None):
            # Downloaded fields and the event reference are immutable inputs.
            # Share them; only fitted transformations and selections are copied.
            shared=(getattr(base,'_analog_raw_cache',None),
                    base._analog_event_reference_sample,base._analog_event_thresholds)
            fitted=deepcopy(base,{id(v):v for v in shared if v is not None})
            for key in cache_names:
                if hasattr(fitted,key):getattr(fitted,key).clear()
            if hasattr(fitted,'_analog_feature_cache'):fitted._analog_feature_cache.clear()
            fit_options=dict(cm)
            if enclosing is not None and not inner_retune:
                fitted._analog_reuse_selections=deepcopy(enclosing)
                fit_options['nested_tuning']=False
            available=y.isel(T=np.asarray(train_positions,dtype=int))
            masked=y.where(y['T'].isin(available['T']))
            pred=fitted.compute_model(masked,np.asarray(train_positions,dtype=int),
                                      np.asarray([test_position],dtype=int),**fit_options)
            if getattr(base,'_analog_raw_cache',None) is None:
                base._analog_raw_cache=getattr(fitted,'_analog_raw_cache',None)
            expected=y['T'].isel(T=[test_position])
            if pred.sizes.get('T')!=1:
                raise ValueError('An analogue prediction must contain its one requested target date')
            pred=pred.assign_coords(T=expected).transpose('T','Y','X')
            return pred,fitted,masked

        records=[];parts=[];probs=[];qs=[];refs=[];counts=[];codes=[]
        for tr,te in cv.custom_cv.split(y['T'],cv.nb_omit):
            tr=np.asarray(tr,dtype=int)
            # Scientific selectors take one target year. For a blocked outer
            # split the whole test block remains excluded from every call.
            for pos in np.asarray(te,dtype=int).ravel():
                pred,fitted,masked=one(tr,int(pos))
                year=int(y['T'].dt.year.isel(T=pos).item())
                for key in cache_names:
                    source=getattr(fitted,key,{})
                    if hasattr(model,key) and year in source:
                        getattr(model,key)[year]=deepcopy(source[year])
                parts.append(pred)
                if not native:
                    selections={
                        'members':getattr(fitted,'_analog_member_cv_cache',{}).get(year),
                        'regions':getattr(fitted,'_regional_cv_cache',{}).get(year),
                        'single':getattr(fitted,'_analog_cv_cache',{}).get(year)}
                    records.append((tr,int(pos),pred,deepcopy(selections)))
                if native:
                    # The model handles both member mixtures and regional
                    # mosaics through its common operational probability path.
                    prob=fitted.compute_prob(masked,clim_start,clim_end,pred,
                        obs_for_terciles=options.get('_probability_reference_sample'))
                    prob=prob.sel(T=pred['T'])
                    q,ref=_event_thresholds(y.isel(T=tr),cfg,options)
                    probs.append(prob);qs.append(q.expand_dims(T=pred['T']))
                    refs.append(ref.expand_dims(T=pred['T']))
        if not parts:raise ValueError('No analogue validation fold was generated')
        hd=xr.concat(parts,dim='T').sortby('T').reindex(T=y['T']).transpose('T','Y','X')
        if not native:
            errors=y-hd
            for tr,pos,pred,selections in records:
                train=y.isel(T=tr)
                if mode=='nested':
                    inner=_inner_block_splits(train['T'],int(options.get('probability_inner_folds',5)),cv.nb_omit)
                    if not inner:raise ValueError('Insufficient history for analogue residual calibration')
                    ih=[]
                    for a,b in inner:
                        for inner_pos in b:
                            ip,_,_=one(tr[a],int(tr[inner_pos]),enclosing=selections)
                            ih.append(ip)
                    residuals=train-xr.concat(ih,dim='T').sortby('T').reindex(T=train['T'])
                elif mode=='outer':
                    residuals=errors.sel(T=train['T'])
                else:
                    residuals=errors  # explicit diagnostic includes own error
                prob,q,ref,code=_training_probabilities(base,pred,residuals,train,cfg,options)
                probs.append(prob);qs.append(q.expand_dims(T=pred['T']))
                refs.append(ref.expand_dims(T=pred['T']))
                counts.append(residuals.notnull().sum('T').expand_dims(T=pred['T']))
                if code is not None:codes.append(code.expand_dims(T=pred['T']))
        hp=xr.concat(probs,dim='T').sortby('T').reindex(T=y['T']).transpose('probability','T','Y','X')
        diagnostics={'thresholds':xr.concat(qs,dim='T').reindex(T=y['T']),
            'reference_probabilities':xr.concat(refs,dim='T').reindex(T=y['T'])}
        if counts:diagnostics['calibration_count']=xr.concat(counts,dim='T').reindex(T=y['T'])
        if codes:diagnostics['distribution_code']=xr.concat(codes,dim='T').reindex(T=y['T'])
        cv.cross_validation_diagnostics_=xr.Dataset(diagnostics)
        hp.attrs['calibration']=('native analogue ensembles selected on outer training only' if native else
            'approximate reuse of other outer-fold residuals; calibration models may include the verification year'
            if mode=='outer' else 'training-only nested analogue residuals' if mode=='nested' else
            'diagnostic full-hindcast residual pool')
        hp.attrs['tercile_reference']=options.get('tercile_reference','training')
        tuned=bool(cm.get('nested_tuning',getattr(model,'nested_tuning',False)))
        metadata={**options,'nested_tuning':tuned,
            '_calibration_hyperparameters_reused':mode=='nested' and not inner_retune and tuned}
        _calibration_metadata(hp,mode,metadata,native=native)
        hp.attrs['analog_inner_preprocessing']=str(getattr(model,'inner_preprocessing','not_applicable'))
        hp.attrs['residual_calibration_applicable']=int(not native)
        model._analog_event_reference_sample=base._analog_event_reference_sample
        model._analog_event_thresholds=base._analog_event_thresholds
        model._analog_raw_cache=getattr(base,'_analog_raw_cache',None)
        return hd,hp
    # Explicit raw predictors retain the inductive historical adapter.
    if x is None:
        raise ValueError('Strict legacy analogue CV needs raw Predictor fields (T,L,Y,X); '
                         'download them first or use a scientific analogue with raw source data')
    fields=dict(x) if isinstance(x,dict) else {'predictor':x}
    spec=WASModelSpec.from_model(model,fields=tuple(fields))
    def predict(xa,ya,xb):
        fitted=spec.create().fit(xa,ya)
        return fitted.predict(xb)
    return _strict_fold_driver(cv,model,y,fields,predict,clim_start,clim_end,options)


def _native_distribution_cv(cv,model,y,x,clim_start,clim_end,options):
    """Fit native NGR/BMA predictive distributions and their events per fold."""
    y=_cv_target(y);dets=[];probs=[];qs=[];refs=[]
    options=dict(options)
    clim_flag=options.pop('clim_terciles',None)
    if clim_flag is not None and not isinstance(clim_flag,(bool,np.bool_)):
        raise ValueError('Pass obs_for_terciles as the observed climatology reference, not precomputed thresholds')
    fit_options={k:v for k,v in options.items() if k not in _PROBABILITY_CONTROLS
                 and k not in {'nested_tuning','probability_inner_retune'}}
    from .was_crossfit import CrossFitConfig
    cfg=CrossFitConfig(climatology_start=clim_start,climatology_end=clim_end)
    for tr,te in cv.custom_cv.split(y['T'],cv.nb_omit):
        fitted=deepcopy(model);xa,xb,ya=x.isel(T=tr),x.isel(T=te),y.isel(T=tr)
        training_reference=ya if clim_start is None else ya.sel(T=slice(str(clim_start),str(clim_end)))
        reference=options.get('_probability_reference_sample',training_reference)
        q,ref=_event_thresholds(ya,cfg,options)
        if isinstance(model,(WAS_mme_NGR_Gaussian,WAS_mme_NGR_NonGaussian)):
            params=_signature_kwargs(fitted.compute_model,fit_options)
            params.update(obs_for_terciles=reference,clim_terciles=True)
            ds=fitted.compute_model(xa,ya,xb,**params)
            det=ds['calibrated_mean']
        else:
            params_fit=_signature_kwargs(fitted.fit,fit_options)
            params_pred=_signature_kwargs(fitted.predict_probabilistic,fit_options)
            if isinstance(fitted,WAS_mme_FullBMA):
                # Fixed family is a declared scientific hypothesis. A learned
                # family map must instead be produced inside this outer fit.
                dist_fit=options.get('distribution_fit')
                if dist_fit is not None:
                    params_fit['dist_map']=dist_fit(training_reference.copy(deep=True))
                elif 'dist_map' in options:
                    raise ValueError('FullBMA strict CV rejects pre-fitted dist_map; '
                        'pass distribution="normal"/"gamma"/"lognormal", or distribution_fit(train)')
                else:
                    name=options.get('distribution','normal')
                    if name not in fitted.DIST_NAME_TO_CODE:
                        raise ValueError('Unknown FullBMA fixed distribution')
                    params_fit['dist_map']=xr.full_like(ya.isel(T=0,drop=True),
                        fitted.DIST_NAME_TO_CODE[name],dtype=float)
            fitted.fit(xa,ya,**params_fit)
            params_pred['obs_for_terciles']=reference
            params_pred['tercile_q']=(1/3,2/3)
            params_pred.setdefault('quantiles',())
            ds=fitted.predict_probabilistic(xb,**params_pred)
            det=ds['predictive_mean']
        prob=ds['tercile_probability'].where(np.isfinite(q).all('quantile'))
        dets.append(det);probs.append(prob);qs.append(q.expand_dims(T=det['T']));refs.append(ref.expand_dims(T=det['T']))
    cv.cross_validation_diagnostics_=xr.Dataset({'thresholds':xr.concat(qs,dim='T').reindex(T=y['T']),
        'reference_probabilities':xr.concat(refs,dim='T').reindex(T=y['T'])})
    hd=xr.concat(dets,dim='T').sortby('T').reindex(T=y['T']).transpose('T','Y','X')
    hp=xr.concat(probs,dim='T').sortby('T').reindex(T=y['T']).transpose('probability','T','Y','X')
    hp.attrs['calibration']='native predictive distribution fitted on outer training only'
    _calibration_metadata(hp,_calibration_mode(options,model),options,native=True)
    hp.attrs['tercile_reference']=options.get('tercile_reference','training')
    if callable(getattr(model,'observation_event_values',None)):
        event_definition=f'max(observation, {float(model.dry_threshold):g})'
        hd.attrs['verified_variable']=event_definition
        hp.attrs['verified_variable']=event_definition
        cv.cross_validation_diagnostics_.attrs['verified_variable']=event_definition
    return hd,hp


class CustomTimeSeriesSplit:
    """Retrospective leave-one-out with a symmetric exclusion buffer.

    For dated seasonal inputs the buffer is measured in calendar years. The
    historical ``nb_omit`` maps to ``ceil(nb_omit/2)`` years on EACH side;
    ``nb_omit=2`` excludes the preceding and following year. Odd values round
    outward rather than introducing an asymmetric window. Edge windows are
    truncated, never shifted. Set ``gap_years`` for an unambiguous contract.
    This is not a forecast-origin/expanding-window backtest.
    """
    def __init__(self, n_splits=None, *, gap_years=None):
        if n_splits is not None and (int(n_splits) != n_splits or n_splits < 1):
            raise ValueError('n_splits must be a positive integer or None')
        if gap_years is not None and (int(gap_years) != gap_years or gap_years < 0):
            raise ValueError('gap_years must be a nonnegative integer')
        self.n_splits = None if n_splits is None else int(n_splits)
        self.gap_years = gap_years

    def split(self, X, nb_omit=0, y=None, groups=None):
        # sklearn calls split(X, y, groups); retain historical split(X, nb_omit).
        if nb_omit is None or not np.isscalar(nb_omit):
            if groups is None and y is not None:
                groups = y
            y, nb_omit = nb_omit, 0
        if int(nb_omit) != nb_omit or nb_omit < 0:
            raise ValueError('nb_omit must be a nonnegative integer')
        gap = self.gap_years if self.gap_years is not None else int(np.ceil(nb_omit / 2))
        coord = X['T'] if isinstance(X, (xr.DataArray, xr.Dataset)) and 'T' in X.coords else X
        n = len(coord)
        idx = np.arange(n)
        if isinstance(coord, xr.DataArray) and hasattr(coord, 'dt'):
            index = coord.to_index()
            if index.has_duplicates or not index.is_monotonic_increasing:
                raise ValueError('Cross-validation dates must be unique and sorted')
            years = coord.dt.year.values
            if len(np.unique(years)) != n:
                raise ValueError('Use one seasonal target per year; repeated-year samples need a grouped SplitPlan')
        elif groups is not None:
            years = np.asarray(groups)
            if years.shape != (n,):
                raise ValueError('groups must supply one ordered year per sample')
        else:
            # Numeric arrays have no calendar; only here is the gap positional.
            years = idx
        for i in range(self.get_n_splits(coord)):
            tr = idx[np.abs(years - years[i]) > gap]
            if tr.size < 2:
                raise ValueError('Exclusion buffer leaves fewer than two training samples')
            yield tr, np.array([i], dtype=int)

    def get_n_splits(self, X=None, y=None, groups=None):
        if X is None:
            if self.n_splits is None:
                raise ValueError('X is required when n_splits is None')
            return self.n_splits
        return len(X) if self.n_splits is None else min(self.n_splits, len(X))


class WAS_Cross_Validator:
    """
    Performs cross-validation for time series forecasting models using a custom time series split.

    This class wraps a custom time series cross-validator to evaluate forecasting models,
    handling both deterministic hindcasts and probabilistic (tercile) predictions.

    Parameters
    ----------
    n_splits : int
        Number of first dates evaluated by leave-one-out; None evaluates all
        dates. This is not a number of blocks covering the complete history.
    nb_omit : int
        Symmetric exclusion window, mapped to ceil(nb_omit/2) calendar years
        on each side of the verification year.
    tercile_reference : {"climatology", "training"}, default "climatology"
        Fixed empirical terciles from the declared observation reference.
        "training" retains fold-local event definitions for research.
    """

    def __init__(self, n_splits, nb_omit, *, tercile_reference='climatology'):
        self.custom_cv = CustomTimeSeriesSplit(n_splits=n_splits)
        self.nb_omit = nb_omit
        if tercile_reference not in ('climatology','training'):
            raise ValueError("tercile_reference must be 'climatology' or 'training'")
        self.tercile_reference=tercile_reference

    def cross_validate(self, model, Predictant, Predictor=None, clim_year_start=None, clim_year_end=None, **model_params):
        """
        Perform cross-validation to compute deterministic hindcasts and tercile probabilities.

        Iterates over time series splits, trains the model on training data, and generates
        predictions for test data. Supports special handling for specific model types
        (e.g., CCA, Analog, ELM, ELR, and various machine learning models).

        Parameters
        ----------
        model : object
            The forecasting model instance to evaluate.
        Predictant : xarray.DataArray
            Target dataset with dimensions ('T', 'Y', 'X') or ('T', 'M', 'Y', 'X').
        Predictor : xarray.DataArray, optional
            Predictor dataset with dimensions ('T', 'M', 'Y', 'X') or ('T', 'features').
            Required for most models except specific cases like WAS_Analog.
        clim_year_start : int or str, optional
            Start year of the climatology period for standardization and probability calculations.
        clim_year_end : int or str, optional
            End year of the climatology period for standardization and probability calculations.
        **model_params : dict
            Additional keyword arguments to pass to the model's compute_model method.

        Returns
        -------
        tuple or xarray.DataArray
            If the model supports probability calculations (has compute_prob method):
                - hindcast_det : xarray.DataArray
                    Deterministic hindcast results with dimensions ('T', 'Y', 'X') or
                    ('probability', 'T', 'Y', 'X') for specific models.
                - hindcast_prob : xarray.DataArray
                    Tercile probabilities with dimensions ('probability', 'T', 'Y', 'X'),
                    where 'probability' includes ['PB', 'PN', 'PA'] (below-normal, normal, above-normal).
            Native probability-only models (ELR and grid classifiers) return
            ``(None, hindcast_prob)``; category probabilities are never passed
            off as deterministic values in physical units.
        """
        if getattr(model, '_was_telnet', False):
            from .was_telnet_cv import cross_validate_telnet
            return cross_validate_telnet(self, model, Predictant, Predictor,
                clim_year_start, clim_year_end, model_params)

        # A reused validator must not publish a previous model's event reference.
        self.cross_validation_diagnostics_=None
        self.verification_target_=None
        model_params=dict(model_params)
        if isinstance(model,WAS_CrossFit) and 'probability_calibration' not in model_params:
            model_params['probability_calibration']=getattr(model.config,'probability_calibration','outer')
        calibration=_calibration_mode(model_params,model)
        if calibration != 'diagnostic':
            model_params['probability_calibration']=calibration
        limit=getattr(self.custom_cv,'n_splits',None)
        if limit is not None and limit < Predictant.sizes['T']:
            warnings.warn(f'n_splits={limit} evaluates only the first {limit} dates, not '
                'blocks covering all years. Other hindcasts remain missing; outer '
                'residual calibration may have too few usable errors. Use n_splits=None '
                'for full coverage.', RuntimeWarning, stacklevel=2)
        if getattr(model, '_was_kharin2017', False):
            # Native multi-season CV groups every season from an excluded year;
            # the annual single-target CrossFit contract cannot represent it.
            options = dict(model_params)
            gap = options.pop('cv_buffer_years', int(np.ceil(self.nb_omit/2)))
            reference=options.pop('tercile_reference',self.tercile_reference)
            event_observations=options.pop('obs_for_terciles',None)
            mode=options.pop('probability_calibration',calibration)
            for key in ('strict_oos_probability','probability_inner_folds','probability_inner_retune','nested_tuning'):
                options.pop(key,None)
            if options:
                raise TypeError(f'Unknown Kharin cross-validation controls: {sorted(options)}')
            result = model.cross_validate(Predictant, Predictor, clim_year_start,
                                          clim_year_end, cv_buffer_years=gap,
                                          tercile_reference=reference, probability_calibration=mode,
                                          obs_for_terciles=event_observations)
            self.kharin_model_ = model
            return result

        if isinstance(model, WAS_CrossFit):
            requested_mode=model_params.pop('probability_calibration',calibration)
            configured=normalize_probability_calibration(getattr(model.config,'probability_calibration','outer'))
            if requested_mode != configured:
                raise ValueError('Set matching probability_calibration in CrossFitConfig')
            plan = model_params.pop("split_plan", None)
            if model_params:
                raise TypeError(f"Unknown strict cross-fit controls: {sorted(model_params)}")
            cfg = model.config
            if clim_year_start is not None or clim_year_end is not None:
                expected = (cfg.climatology_start, cfg.climatology_end)
                actual = (int(clim_year_start), int(clim_year_end)) if clim_year_start is not None and clim_year_end is not None else (None, None)
                if actual != expected:
                    raise ValueError("Set matching climatology bounds in CrossFitConfig; no implicit full-record override")
            if plan is None:
                # Legacy nb_omit is positional/asymmetric. Strict buffers are
                # explicitly configured in calendar years, in CrossFitConfig.
                if cfg.gap_years != int(np.ceil(self.nb_omit / 2)):
                    raise ValueError("Supply an explicit SplitPlan or match config.gap_years to ceil(nb_omit/2)")
                plan = model.default_split_plan(Predictant['T'])
                limit = self.custom_cv.n_splits
                if limit is not None and limit < len(plan.folds):
                    plan = SplitPlan(plan.times,plan.years,plan.folds[:limit],plan.mode,
                                     plan.gap_years,plan.min_train)
            self.crossfit_result_ = model.evaluate(Predictor, Predictant, plan)
            return self.crossfit_result_.deterministic, self.crossfit_result_.probabilities

        # A censored predictive law verifies its censored observed variable.
        # Apply the same declared mapping before deriving fixed event limits,
        # including when a separate observed climatology is supplied.  The
        # native distribution model validates the physical support here.
        event_transform=getattr(model,'observation_event_values',None)
        if callable(event_transform):
            Predictant=event_transform(Predictant)
            model_params=dict(model_params)
            if model_params.get('obs_for_terciles') is not None:
                model_params['obs_for_terciles']=event_transform(model_params['obs_for_terciles'])
        self.verification_target_=_cv_target(Predictant)
        model_params=_climatology_options(self,Predictant,clim_year_start,clim_year_end,model_params)
        model.tercile_reference=model_params['tercile_reference']
        if hasattr(model,'reg_model'):model.reg_model.tercile_reference=model_params['tercile_reference']
        self.tercile_reference_=model_params['tercile_reference']
        # Clear prior diagnostics; direct classifier branches get the same reference.
        self.cross_validation_diagnostics_=None
        _fixed_event_diagnostics(self,_cv_target(Predictant),model_params)
        time_alignment = model_params.pop("time_alignment", "exact")
        Predictor = align_predictor_target(Predictor, Predictant, mode=time_alignment)
        _validate_ensemble_predictors(model, Predictor)
        if (getattr(model, '_was_spatial_cnn', False)
                or getattr(model, '_was_unet', False)
                or getattr(model, '_was_rescnn', False)
                or getattr(model, '_was_unetpp', False)
                or getattr(model, '_was_rise_unet', False)):
            return _convolutional_cross_validate(self, model, Predictant, Predictor,
                clim_year_start, clim_year_end, model_params)
        # Do not disclose the held-out target through input coordinates/values.
        hindcast_det = []
        hindcast_prob = []
        n_splits = len(Predictant.get_index("T"))

        same_kind_model1 = [WAS_LinearRegression_Model, WAS_MARS_Model,
                          WAS_PolynomialRegression,
                           WAS_Ridge_Model, WAS_Lasso_Model,
                           WAS_LassoLars_Model, WAS_ElasticNet_Model,
                          WAS_RandomForest_Model, WAS_ExtraTrees_Model,
                         WAS_GradientBoosting_Model, WAS_AdaBoost_Model,
                         WAS_XGBoost_Model, WAS_LightGBM_Model,
                         WAS_SVR_Model, WAS_KNN_Model]
                         
        same_kind_count = [WAS_PoissonRegression, WAS_NegativeBinomial_Model, WAS_ZINB_Model, WAS_HurdleNB_Model]

        same_kind_model2 = [WAS_KNN_Classifier_Model, WAS_SVC_Classifier_Model, WAS_RandomForest_Classifier_Model, WAS_ExtraTrees_Classifier_Model, WAS_GradientBoosting_Classifier_Model, WAS_AdaBoost_Classifier_Model, WAS_LightGBM_Classifier_Model, WAS_XGBoost_Classifier_Model, WAS_NaiveBayes_Classifier_Model, WAS_LinearDiscriminant_Classifier_Model, WAS_QuadraticDiscriminant_Classifier_Model, WAS_LogisticRegression_Model]


        same_kind_model3 = [globals()[n] for n in MME_REGRESSION_CV_NAMES if n in globals()]

        if type(model).__name__ in {'WAS_ProbWeightedRPS','WAS_ExpertKnowledge','WAS_mme_ExpertKnowledge'}:
            target = _cv_target(Predictant)
            warnings.warn('Weight validation is conditional on supplied forecast probabilities. '
                          'For end-to-end nested validation, generate base probabilities within '
                          'each outer training fold with the same category definition; ordinary '
                          'precomputed OOF forecasts do not establish that guarantee.',
                          RuntimeWarning, stacklevel=2)
            parts=[]
            for tr,te in self.custom_cv.split(target['T'],self.nb_omit):
                train = ({k:v.isel(T=tr) for k,v in Predictor.items()}
                         if isinstance(Predictor,dict) else Predictor.isel(T=tr))
                test = ({k:v.isel(T=te) for k,v in Predictor.items()}
                        if isinstance(Predictor,dict) else Predictor.isel(T=te))
                fitted=deepcopy(model)
                ytrain=target.isel(T=tr)
                if type(model).__name__ in {'WAS_ExpertKnowledge','WAS_mme_ExpertKnowledge'}:
                    labels,_=_class_from_training_climatology(ytrain,clim_year_start,clim_year_end,model_params.get('_probability_thresholds'))
                    pred=fitted.compute_model(train,labels,test,**{k:v for k,v in model_params.items() if k not in _PROBABILITY_CONTROLS})
                else:
                    labels,_=_class_from_training_climatology(ytrain,clim_year_start,clim_year_end,model_params.get('_probability_thresholds'))
                    fitted.fit(ytrain,train,clim_year_start,clim_year_end,obs_class=labels)
                    pred=fitted.transform(test)
                if 'T' not in pred.dims:
                    pred=pred.expand_dims(T=target['T'].isel(T=te))
                parts.append(pred.assign_coords(T=target['T'].isel(T=te)))
            prob=xr.concat(parts,dim='T').sortby('T').reindex(T=target['T']).transpose('probability','T','Y','X')
            prob.attrs['validation']='fold-local weights conditional on supplied base probabilities'
            return prob,prob

        # ------------------------------------------------------------------
        # Fold-safe bias correction / quantile mapping adapters
        # ------------------------------------------------------------------
        if getattr(model, "_was_bias_correction_adapter", False):
            if Predictor is None:
                raise ValueError("Predictor/model data are required for bias-correction cross-validation.")
            obs=_cv_target(Predictant)
            pred=Predictor.copy(deep=False) if isinstance(Predictor,xr.DataArray) else Predictor
            if not isinstance(pred,xr.DataArray) or "T" not in pred.dims:
                raise TypeError("Bias-correction Predictor must be an xarray.DataArray with a T dimension.")
            if pred.sizes["T"]==obs.sizes["T"]:
                pred=pred.assign_coords(T=obs["T"].values)
            folds=[]
            print(f"Cross-validation ongoing (bias correction: {getattr(model,'family','?')}/{getattr(model,'method','?')})")
            for train_index,test_index in tqdm(
                    self.custom_cv.split(obs["T"],self.nb_omit), total=n_splits):
                X_train=pred.isel(T=train_index); X_test=pred.isel(T=test_index)
                y_train=obs.isel(T=train_index); y_test=obs.isel(T=test_index)
                corrected=deepcopy(model).compute_model(X_train,y_train,X_test,xr.full_like(y_test,np.nan))
                if isinstance(corrected,xr.DataArray) and "T" not in corrected.dims:
                    corrected=corrected.expand_dims(T=y_test["T"].values)
                folds.append(corrected)
            hindcast_det=xr.concat(folds,dim="T").sortby("T").reindex(T=obs["T"])
            # Bias correction is deterministic.  Return None explicitly rather
            # than disguising a deterministic field as probabilities.
            return hindcast_det, None

        if isinstance(model, (WAS_CCA_op, WAS_CCA_base, WAS_CCA, WAS_CCA_strict)):
            return _cca_cross_validate(
                self, model, Predictant, Predictor,
                clim_year_start, clim_year_end, model_params,
            )

        elif isinstance(model, WAS_KernelCCA_base):
            return _native_regression_cv(self,model,Predictant,Predictor,
                clim_year_start,clim_year_end,model_params,'kcca')

        elif isinstance(model, WAS_Analog):
            return _analog_cross_validate(self,model,Predictant,Predictor,
                clim_year_start,clim_year_end,model_params)

        elif isinstance(model, WAS_mme_logistic):
            target=_cv_target(Predictant)
            P=_maybe_align_predictor_time(Predictor,target)
            mask=spatial_mask(target)
            nested=bool(model_params.pop("nested_tuning",True))
            user_bp=model_params.pop("best_params",None); user_cl=model_params.pop("cluster_da",None)
            splits=[(np.asarray(a),np.asarray(b)) for a,b in self.custom_cv.split(P["T"],self.nb_omit)]
            print("Cross-validation ongoing (MME logistic, fold-local tuning)")
            dets=[]; probs=[]
            base_model=deepcopy(model)
            for tr,te in tqdm(splits,total=len(splits)):
                model=deepcopy(base_model)
                Xtr_raw,Xte_raw=P.isel(T=tr),P.isel(T=te)
                ytr_raw=target.isel(T=tr)
                Xtr,Xte,_,_=_fold_standardize(Xtr_raw,Xte_raw,clim_year_start,clim_year_end)
                ycls,_=_class_from_training_climatology(ytr_raw,clim_year_start,clim_year_end,model_params.get('_probability_thresholds'))
                bp,cl=user_bp,user_cl
                if nested or bp is None or cl is None:
                    bp,cl=model.compute_hyperparameters(Xtr_raw,ycls,clim_year_start,clim_year_end,target_is_class=True)
                pd,pp=model.compute_model(Xtr,ycls,Xte,xr.full_like(target.isel(T=te),np.nan),
                                          clim_year_start,clim_year_end,
                                          best_params=bp,cluster_da=cl)
                pd=pd.assign_coords(T=target["T"].isel(T=te)); pp=pp.assign_coords(T=target["T"].isel(T=te))
                dets.append(pd); probs.append(pp)
            hd=xr.concat(dets,dim="T").sortby("T").reindex(T=target["T"])*mask
            hp=xr.concat(probs,dim="T").sortby("T").reindex(T=target["T"])
            hp=hp.transpose("probability","T","Y","X")*mask
            ps=hp.sum("probability",skipna=False); hp=xr.where(ps>0,hp/ps,np.nan).transpose("probability","T","Y","X")  # xr.where reorders dims
            return hd,hp

        elif isinstance(model, WAS_mme_ELR):
            target=_cv_target(Predictant)
            P=_maybe_align_predictor_time(Predictor,target)
            mask=spatial_mask(target)
            nested=bool(model_params.pop("nested_tuning",True))
            score=model_params.pop("tuning_score","rps")
            inner_folds=int(model_params.pop('inner_max_folds',5))
            if nested and inner_folds<2:
                raise ValueError('ELR inner_max_folds must be at least 2 for tuning')
            splits=[(np.asarray(a),np.asarray(b)) for a,b in self.custom_cv.split(P["T"],self.nb_omit)]
            base_model=deepcopy(model)
            probs=[]
            print("Cross-validation ongoing (ELR, fold-local preprocessing/tuning)")
            for tr,te in tqdm(splits,total=len(splits)):
                # Hyperparameter fitting mutates l2, transformation and the
                # quantile grid.  Every fold must start with exactly the same
                # original candidates, independent of earlier held-out years.
                fitted=deepcopy(base_model)
                Xtr_raw,Xte_raw=P.isel(T=tr),P.isel(T=te); ytr=target.isel(T=tr)
                _,event_pair=_class_from_training_climatology(ytr,clim_year_start,clim_year_end,model_params.get('_probability_thresholds'))
                if nested:
                    fitted.compute_hyperparameters(Xtr_raw,ytr,clim_year_start,clim_year_end,
                                                   clim_terciles=event_pair,score=score,
                                                   nb_omit=self.nb_omit,
                                                   cv_folds=min(inner_folds,max(2,len(np.unique(ytr["T"].dt.year)))))
                Xtr,Xte=fitted._standardize_predictors(
                    Xtr_raw,Xte_raw,clim_year_start,clim_year_end)
                pp=fitted.compute_model(Xtr,ytr,Xte,xr.full_like(target.isel(T=te),np.nan),
                                        clim_year_start,clim_year_end,clim_terciles=event_pair)
                pp=pp.assign_coords(T=target["T"].isel(T=te)); probs.append(pp)
            hp=xr.concat(probs,dim="T").sortby("T").reindex(T=target["T"])
            hp=hp.transpose("probability","T","Y","X")*mask
            ps=hp.sum("probability",skipna=False); hp=xr.where(ps>0,hp/ps,np.nan).transpose("probability","T","Y","X")  # xr.where reorders dims
            hp.attrs['calibration']='native ELR fitted independently on outer training only'
            _calibration_metadata(hp,calibration,model_params,native=True)
            hp.attrs['tercile_reference']=model_params['tercile_reference']
            return None,hp

        elif isinstance(model,(WAS_mme_NGR_Gaussian,WAS_mme_NGR_NonGaussian,
                               WAS_mme_FullBMA,WAS_mme_FastBMA)):
            return _native_distribution_cv(self,model,Predictant,Predictor,
                clim_year_start,clim_year_end,model_params)

        elif isinstance(model, WAS_mme_gaussian_process):
            target=_cv_target(Predictant); P=_maybe_align_predictor_time(Predictor,target)
            mask=spatial_mask(target); nested=bool(model_params.pop("nested_tuning",True))
            user_bp=model_params.pop("best_params",None); user_cl=model_params.pop("cluster_da",None)
            splits=[(np.asarray(a),np.asarray(b)) for a,b in self.custom_cv.split(P["T"],self.nb_omit)]
            dets=[]; probs=[]; print("Cross-validation ongoing (Gaussian Process MME)")
            base_model=deepcopy(model)
            for tr,te in tqdm(splits,total=len(splits)):
                model=deepcopy(base_model)
                Xtr_raw,Xte_raw=P.isel(T=tr),P.isel(T=te); ytr_raw=target.isel(T=tr)
                Xtr,Xte,_,_=_fold_standardize(Xtr_raw,Xte_raw,clim_year_start,clim_year_end)
                ycls,_=_class_from_training_climatology(ytr_raw,clim_year_start,clim_year_end,model_params.get('_probability_thresholds'))
                bp,cl=user_bp,user_cl
                if nested or bp is None or cl is None:
                    bp,cl=model.compute_hyperparameters(Xtr_raw,ycls,clim_year_start,clim_year_end,target_is_class=True)
                pd,pp=model.compute_model(Xtr,ycls,Xte,xr.full_like(target.isel(T=te),np.nan),clim_year_start,clim_year_end,
                                          best_params=bp,cluster_da=cl)
                pd=pd.assign_coords(T=target["T"].isel(T=te)); pp=pp.assign_coords(T=target["T"].isel(T=te))
                dets.append(pd); probs.append(pp)
            hd=xr.concat(dets,dim="T").sortby("T").reindex(T=target["T"])*mask
            hp=xr.concat(probs,dim="T").sortby("T").reindex(T=target["T"])
            hp=hp.transpose("probability","T","Y","X")*mask
            ps=hp.sum("probability",skipna=False); hp=xr.where(ps>0,hp/ps,np.nan).transpose("probability","T","Y","X")  # xr.where reorders dims
            return hd,hp

        elif isinstance(model, WAS_Min2009_ProbWeighted):

            if "M" in Predictant.coords:
                Predictant = Predictant.isel(M=0, drop=True)
            else:
                Predictant = Predictant
            all_params = {**model_params}

            params_prob = {
                key: value for key, value in all_params.items()
                if key in inspect.signature(model.compute_pmme_probabilities).parameters
            }

            params_models = {
                key: value for key, value in all_params.items()
                if key not in params_prob
            }

            masks = params_models.get('masks')
            ensemble_sizes = params_prob.get('ensemble_sizes')
            climatology = params_prob.get('climatology')

            # mask = spatial_mask(_cv_target(Predictant)).to_numpy()
            Predictor['T'] = Predictant['T']

            from .was_crossfit import CrossFitConfig
            event_cfg=CrossFitConfig(climatology_start=clim_year_start,climatology_end=clim_year_end)
            event_qs,event_refs=[],[]
            print("Cross-validation ongoing")
            for i, (train_index, test_index) in enumerate(tqdm(self.custom_cv.split(Predictor['T'], self.nb_omit), total=n_splits), start=1):
                X_train, X_test = Predictor.isel(T=train_index), Predictor.isel(T=test_index)
                forecasts = {k: X_test.sel(M=v) for k, v in masks.items()}
                hindcasts = {k: X_train.sel(M=v) for k, v in masks.items()}
                q,ref=_event_thresholds(Predictant.isel(T=train_index),event_cfg,model_params)
                pair=tuple(q.isel(quantile=i,drop=True) for i in range(2))
                pred_prob = model.compute_pmme_probabilities(forecasts, hindcasts, climatology, ensemble_sizes,
                                                             clim_terciles=pair)
                pred_prob  = xr.concat([pred_prob["PB"], pred_prob["PN"], pred_prob["PA"]], dim=xr.DataArray(["PB", "PN", "PA"], dims="probability", name="probability"),)
                hindcast_prob.append(pred_prob)
                event_qs.append(q.expand_dims(T=X_test['T']))
                event_refs.append(ref.expand_dims(T=X_test['T']))

            hindcast_prob = xr.concat(hindcast_prob, dim="T")
            hindcast_prob['T'] = Predictant['T']

            self.cross_validation_diagnostics_=xr.Dataset({
                'thresholds':xr.concat(event_qs,dim='T'),
                'reference_probabilities':xr.concat(event_refs,dim='T')})
            return hindcast_prob, hindcast_prob

        # ── native classifiers: labels, tuning, state rebuilt on TRAIN ─────
        elif (any(isinstance(model, cls) for cls in same_kind_model2)
              or (isinstance(model, WAS_PCR) and
                  any(isinstance(model.reg_model, cls) for cls in same_kind_model2))):
            from .was_crossfit import CrossFitConfig, _thresholds
            target = _cv_target(Predictant)
            cfg = CrossFitConfig(climatology_start=clim_year_start,climatology_end=clim_year_end)
            params=dict(model_params)
            nested=bool(params.pop('nested_tuning',True))
            parts, thresholds, refs = [], [], []
            if isinstance(model,WAS_PCR):self.pcr_tuning_history_=[]
            _validate_fixed_parameters(model.reg_model if isinstance(model,WAS_PCR) else model,params,nested)
            for tr,te in self.custom_cv.split(_pcr_T_index(Predictor),self.nb_omit):
                fitted=deepcopy(model)
                inner=fitted.reg_model if isinstance(fitted,WAS_PCR) else fitted
                xa,xb=_pcr_isel_T(Predictor,tr),_pcr_isel_T(Predictor,te)
                ya=target.isel(T=tr)
                labels,_=_class_from_training_climatology(ya,clim_year_start,clim_year_end,params.get('_probability_thresholds'))
                kwargs=_signature_kwargs(inner.compute_model,{k:v for k,v in params.items() if k not in _PROBABILITY_CONTROLS})
                sig=inspect.signature(inner.compute_model).parameters
                if nested and isinstance(fitted,WAS_PCR):
                    kwargs.update(_tune_pcr_raw(fitted,xa,ya,clim_year_start,clim_year_end,
                        self.nb_omit,params,classification=True))
                    self.pcr_tuning_history_.append({'training_dates':ya['T'].values.copy(),
                        **deepcopy(fitted.pcr_tuning_)})
                elif nested and hasattr(inner,'compute_hyperparameters'):
                    fixed_q=params.get('_probability_thresholds')
                    if fixed_q is not None and hasattr(inner,'compute_class'):
                        def classify(obs,start=None,end=None,q=fixed_q):
                            labels,(low,high)=_class_from_training_climatology(obs,start,end,q)
                            return labels,low,high
                        inner.compute_class=classify
                    hp=inner.compute_hyperparameters(ya,xa,clim_year_start,clim_year_end)
                    kwargs.update(_hyperparameter_kwargs(inner,hp))
                if 'hyperparams' in sig and kwargs.get('hyperparams') is None:
                    raise ValueError('This classifier requires hyperparams; enable nested_tuning or supply fixed hyperparameter maps')
                pred=fitted.compute_model(xa,labels,xb,**kwargs)
                if 'T' not in pred.dims:
                    pred=pred.expand_dims(T=target['T'].isel(T=te))
                pred=pred.assign_coords(T=target['T'].isel(T=te))
                parts.append(pred)
                q,ref=_event_thresholds(ya,cfg,params)
                thresholds.append(q.expand_dims(T=pred['T']))
                refs.append(ref.expand_dims(T=pred['T']))
            prob=xr.concat(parts,dim='T').sortby('T').reindex(T=target['T']).transpose('probability','T','Y','X')
            self.cross_validation_diagnostics_=xr.Dataset({
                'thresholds':xr.concat(thresholds,dim='T').reindex(T=target['T']),
                'reference_probabilities':xr.concat(refs,dim='T').reindex(T=target['T'])})
            prob.attrs['calibration']='classifier fitted on training observations with declared tercile boundaries'
            prob.attrs['tercile_reference']=params.get('tercile_reference','training')
            _calibration_metadata(prob,calibration,model_params,native=True)
            return None,prob

        elif any(isinstance(model,cls) for cls in same_kind_count):
            return _native_regression_cv(self,model,Predictant,Predictor,
                clim_year_start,clim_year_end,model_params,'count')

        elif isinstance(model,WAS_PCR) and any(isinstance(model.reg_model,cls)
                for cls in same_kind_count+same_kind_model1):
            family='count' if any(isinstance(model.reg_model,cls) for cls in same_kind_count) else 'continuous'
            return _native_regression_cv(self,model,Predictant,Predictor,
                clim_year_start,clim_year_end,model_params,family)

        elif (any(isinstance(model,cls) for cls in same_kind_model3)
              or getattr(model,'_was_cv_family',None)=='mme_regression'):
            return _native_regression_cv(self,model,Predictant,Predictor,
                clim_year_start,clim_year_end,model_params,'mme')

        elif any(isinstance(model,cls) for cls in same_kind_model1):
            return _native_regression_cv(self,model,Predictant,Predictor,
                clim_year_start,clim_year_end,model_params,'continuous')

        elif isinstance(model,(WAS_mme_CCA,WAS_mme_CCA_eeof)):
            return _native_regression_cv(self,model,Predictant,Predictor,
                clim_year_start,clim_year_end,model_params,'mme_cca')

        else:
            raise TypeError(
                "cross_validate: no branch for "
                f"{type(model).__module__}.{type(model).__qualname__}; "
                f"registered MME regressors: {[c.__name__ for c in same_kind_model3]}"
            )


# =============================================================================
# Bias-correction convenience API (2026-08-26)
# =============================================================================
def _cross_validate_bias_correction_public(
    self, observations, modeled, family="qmap", method=None,
    fit_kwargs=None, apply_kwargs=None, member_dim="M", reduce_members="mean",
    random_state=None,
):
    """Run a fold-safe bias correction using this validator's temporal splits.

    Examples
    --------
    ``cv.cross_validate_bias_correction(obs, hcst, family='qmap', method='RQUANT')``
    ``cv.cross_validate_bias_correction(t2m_obs, t2m_hcst, family='continuous', method='VARSCALE')``
    """
    fam=str(family).lower()
    if fam in {"qmap","precip","precipitation","rainfall"}:
        mdl=WAS_Qmap_CV(method=method or "RQUANT",fit_kwargs=fit_kwargs,apply_kwargs=apply_kwargs,
                        member_dim=member_dim,reduce_members=reduce_members,random_state=random_state)
    elif fam in {"continuous","bc","bias_correction","bias-correction"}:
        mdl=WAS_bias_correction_CV(method=method or "MEAN",fit_kwargs=fit_kwargs,apply_kwargs=apply_kwargs,
                                   member_dim=member_dim,reduce_members=reduce_members,random_state=random_state)
    else:
        raise ValueError("family must be 'qmap' or 'continuous'.")
    det,_=self.cross_validate(mdl,observations,modeled)
    return det


def _compare_bias_corrections_public(
    self, observations, modeled, family="qmap", methods=None,
    fit_kwargs=None, apply_kwargs=None, member_dim="M", reduce_members="mean",
    random_state=None, return_hindcasts=False, include_raw=True,
):
    """Compare correction methods on identical outer folds using RMSE/MAE/bias/correlation."""
    fam=str(family).lower()
    if methods is None:
        methods=("QUANT","RQUANT","SSPLIN","PTF","DIST") if fam in {"qmap","precip","precipitation","rainfall"} else ("MEAN","VARSCALE","QUANT","NORM","DIST")
    obs=_cv_target(observations)
    rows=[]; hcasts={}

    def _metrics(name,h):
        h_eval=h.mean("M",skipna=True) if "M" in h.dims else h
        h_eval=h_eval.transpose("T","Y","X")
        o,he=xr.align(obs,h_eval,join="inner")
        diff=he-o
        rmse=float(np.sqrt((diff**2).mean(skipna=True)).values)
        mae=float(np.abs(diff).mean(skipna=True).values)
        bias=float(diff.mean(skipna=True).values)
        ov=np.asarray(o.values).ravel(); pv=np.asarray(he.values).ravel(); keep=np.isfinite(ov)&np.isfinite(pv)
        corr=float(np.corrcoef(ov[keep],pv[keep])[0,1]) if keep.sum()>=3 and np.std(ov[keep])>0 and np.std(pv[keep])>0 else np.nan
        return {"Method":str(name),"RMSE":rmse,"MAE":mae,"Bias":bias,"Pearson":corr,"N":int(keep.sum())}

    if include_raw:
        raw=modeled
        if isinstance(raw,xr.DataArray) and raw.sizes.get("T") == obs.sizes.get("T"):
            raw=raw.assign_coords(T=obs["T"].values)
        rows.append(_metrics("RAW",raw)); hcasts["RAW"]=raw

    for method in methods:
        try:
            h=_cross_validate_bias_correction_public(\
                self,obs,modeled,family=fam,method=method,fit_kwargs=fit_kwargs,apply_kwargs=apply_kwargs,\
                member_dim=member_dim,reduce_members=reduce_members,random_state=random_state)
            rows.append(_metrics(str(method),h))
            hcasts[str(method)]=h
        except Exception as exc:
            rows.append({"Method":str(method),"RMSE":np.nan,"MAE":np.nan,"Bias":np.nan,"Pearson":np.nan,"N":0,"Error":repr(exc)})
    import pandas as pd
    table=pd.DataFrame(rows)
    if include_raw and (table["Method"]=="RAW").any():
        rawrow=table.loc[table["Method"]=="RAW"].iloc[0]
        table["Delta_RMSE_vs_RAW"]=table["RMSE"]-float(rawrow["RMSE"])
        table["Delta_MAE_vs_RAW"]=table["MAE"]-float(rawrow["MAE"])
        table["RMSE_skill_vs_RAW"]=1.0-(table["RMSE"]**2)/(float(rawrow["RMSE"])**2) if float(rawrow["RMSE"])>0 else np.nan
    table=table.sort_values("RMSE",na_position="last").reset_index(drop=True)
    if return_hindcasts:
        return table,hcasts
    return table


WAS_Cross_Validator.cross_validate_bias_correction=_cross_validate_bias_correction_public
WAS_Cross_Validator.compare_bias_corrections=_compare_bias_corrections_public
