"""Fresh, fold-local adapters for WASS2S' specialised scientific models.

Inputs are raw annual fields, indexed by forecast/verification year T.
Analogue seasonal lead sequences may additionally carry L. No downloads or
pre-fitted objects are accepted by this interface.
"""
from __future__ import annotations
from copy import deepcopy, copy
from dataclasses import dataclass, field
from functools import wraps
from importlib import import_module
import inspect
import hashlib
import numpy as np
import xarray as xr
from .was_scientific_utils import apply_physical_support, climate_cluster_map

CCA_NAMES = ('WAS_CCA', 'WAS_CCA_base', 'WAS_CCA_op', 'WAS_CCA_strict',
             'WAS_CCA_Regularized', 'WAS_CCA_Regularized_strict')
KCCA_NAMES = ('WAS_KCCA', 'WAS_KGCCA', 'WAS_EOF_KCCA')
ANALOG_NAMES = ('WAS_Analog_Scientific', 'WAS_Analog_KNN', 'WAS_Analog_Weighted',
               'WAS_Analog_EOF', 'WAS_Analog_Hybrid', 'WAS_Analog_Constructed',
               'WAS_Analog_LearnedMetric', 'WAS_Analog_LearnedSpatialMetric',
               'WAS_Analog_Regional')
REGISTRY = {'WAS_Analog': ('was_analog', 'legacy_analog'),
            'WAS_Analog__': ('was_analog', 'legacy_analog'),**{n: ('was_cca', 'cca') for n in CCA_NAMES},
            **{n: ('was_cca_2', 'kcca') for n in KCCA_NAMES},
            **{n: ('was_analog', 'analog') for n in ANALOG_NAMES},
            'WAS_mme_hpELM': ('was_mme', 'hpelm'),
            'WAS_mme_FastBMA': ('was_mme', 'bma'),
            'WAS_mme_FullBMA': ('was_mme', 'bma'),
            'WAS_mme_UNet': ('was_mme_unet', 'unet'),
            'WAS_mme_ResCNN': ('was_mme_rescnn', 'rescnn'),
            'WAS_mme_UNetPP': ('was_mme_unet_variants', 'unetpp'),
            'WAS_mme_RISEUNet': ('was_mme_unet_variants', 'rise-unet')}
__all__ = ['WASModelSpec', 'migrated_models']


def _restore_xeofs_container(items, attributes):
    from xeofs.data_container import DataContainer
    result = DataContainer(items)
    result.__dict__.update(attributes)
    return result


def _reduce_xeofs_container(value):
    return _restore_xeofs_container, (dict(value), dict(value.__dict__))


def _enable_xeofs_pickle():
    # xeofs 3.x dict subclass restores items before its _allow_compute state.
    # Initialize the container before restoring attributes; no model refit.
    import copyreg
    from xeofs.data_container import DataContainer
    copyreg.pickle(DataContainer, _reduce_xeofs_container)


def migrated_models():
    return dict(REGISTRY)


def capture_constructors(namespace):
    """Keep constructor inputs, never learned state, for bridge conversion."""
    for name in REGISTRY:
        cls = namespace.get(name)
        if not inspect.isclass(cls) or cls.__module__ != namespace.get('__name__'):
            continue
        original = cls.__init__
        if getattr(original, '_crossfit_capture', False) and '__init__' in cls.__dict__:
            continue
        def wrap(init, owner):
            @wraps(init)
            def constructor(self, *args, **kwargs):
                init(self, *args, **kwargs)
                if type(self) is owner:
                    self._crossfit_constructor = (deepcopy(args), deepcopy(kwargs))
            constructor._crossfit_capture = True
            return constructor
        cls.__init__ = wrap(original, cls)


@dataclass(frozen=True)
class WASModelSpec:
    name: str
    model: str
    init: dict = field(default_factory=dict)
    fields: tuple = ('predictor',)
    parameters: dict = field(default_factory=dict)
    min_samples: int = 5
    seed: int = 42
    # Maps each model field to a separate reanalysis-library field, if available.
    history_fields: dict = field(default_factory=dict)
    args: tuple = ()

    def __post_init__(self):
        if self.model not in REGISTRY:
            raise ValueError(f'No scientific adapter registered for {self.model!r}')
        if not self.name or not self.fields or len(set(self.fields)) != len(self.fields):
            raise ValueError('Supply a name and unique predictor fields')
        if self.min_samples < 3:
            raise ValueError('min_samples must be at least three')
        if set(self.history_fields) - set(self.fields):
            raise ValueError('history_fields keys must name predictor fields')
        if any(isinstance(v, (xr.DataArray, xr.Dataset)) for v in self.parameters.values()):
            raise ValueError('No pre-fitted maps: supply fixed scalar parameters instead')

    @property
    def native_probabilities(self):
        return REGISTRY[self.model][1] in {'bma', 'analog'}

    @classmethod
    def from_model(cls, model, *, name=None, **kwargs):
        model_name = type(model).__name__
        if model_name not in REGISTRY:
            raise ValueError(f'{model_name} is not a migrated specialised model')
        if not hasattr(model, '_crossfit_constructor'):
            raise ValueError('Recreate this model with the migrated package, or use WASModelSpec with explicit init')
        args, init = model._crossfit_constructor
        args, init = list(deepcopy(args)), deepcopy(init)

        def current_option(option, value):
            # A user may change a preprocessing switch after construction.
            # Preserve its effective value when creating fresh fold models,
            # including constructors originally called with positional args.
            for owner in type(model).__mro__:
                positional = [p.name for p in inspect.signature(owner.__init__).parameters.values()
                              if p.name != 'self' and p.kind in (p.POSITIONAL_ONLY, p.POSITIONAL_OR_KEYWORD)]
                if option in positional:
                    position = positional.index(option)
                    if position < len(args):
                        args[position] = value
                        init.pop(option, None)
                        return
                    break
            init[option] = value

        if model_name in {'WAS_CCA_op', 'WAS_CCA_strict',
                          'WAS_CCA_Regularized', 'WAS_CCA_Regularized_strict'}:
            current_option('detrend', bool(model.detrend))
        elif model_name == 'WAS_EOF_KCCA':
            # Both public spellings refer to one EOF preprocessing switch.
            init.pop('detrend', None)
            current_option('eof_detrend', bool(model.eof_detrend))
        return cls(name or model_name, model_name, init=init, args=tuple(args), **kwargs)

    def create(self):
        if REGISTRY[self.model][1] == 'legacy_analog':
            return _LegacyAnalogAdapter(self)
        return _NativeAdapter(self)


class _NativeAdapter:
    def __init__(self, spec):
        self.spec = spec
        self.family = REGISTRY[spec.model][1]
        self.model = None

    def __getstate__(self):
        # The ELM models are plain numpy objects (wass2s.was_elm) and pickle
        # as they are; nothing family-specific to strip any more.
        return dict(self.__dict__)

    def __setstate__(self, state):
        state = dict(state)
        if state.pop('_hpelm_state', None) is not None:
            raise ValueError('This adapter was saved with the hpelm package, which WASS2S no '
                             'longer uses; refit it.')
        self.__dict__.update(state)

    def _input(self, X, fit=False):
        fields = []
        for name in self.spec.fields:
            if name not in X:
                raise ValueError(f'Missing predictor {name!r}')
            a = X[name]
            schema = {d: np.asarray(a[d]).copy() for d in a.dims if d != 'T'}
            if fit:
                self.schemas[name] = schema
            elif set(schema) != set(self.schemas[name]) or any(
                    not np.array_equal(v, self.schemas[name][d]) for d, v in schema.items()):
                raise ValueError(f'Predictor grid/channel identity changed: {name}')
            fields.append(a)
        if self.family == 'analog':
            return dict(zip(self.spec.fields, fields))
        if self.family == 'kcca' and self.spec.model == 'WAS_EOF_KCCA':
            return [self._single(a) for a in fields]
        if self.family in {'cca', 'kcca'}:
            if len(fields) != 1:
                raise ValueError('CCA takes one field; use WAS_EOF_KCCA for multiple grids')
            return self._single(fields[0])
        out = []
        for name, a in zip(self.spec.fields, fields):
            if set(a.dims) - {'T', 'M', 'Y', 'X'}:
                raise ValueError('MME predictors require T,M,Y,X')
            if 'M' not in a.dims:
                a = a.expand_dims(M=[name])
            elif len(fields) > 1:
                a = a.assign_coords(M=[f'{name}:{v}' for v in a.M.values])
            out.append(a)
        result = xr.concat(out, dim='M', join='exact') if len(out) > 1 else out[0]
        if len(np.unique(result.M.values)) != result.sizes['M']:
            raise ValueError('Duplicate M channel labels')
        return result.transpose('T', 'M', 'Y', 'X')

    @staticmethod
    def _single(a):
        if 'M' in a.dims and a.sizes['M'] == 1:
            a = a.isel(M=0, drop=True)
        if set(a.dims) != {'T', 'Y', 'X'}:
            raise ValueError('This CCA requires a T,Y,X field (singleton M accepted)')
        return a.transpose('T', 'Y', 'X')

    @staticmethod
    def _stats(a):
        mean = a.mean('T', skipna=True)
        sd = a.std('T', skipna=True)
        return mean, xr.where(sd > 1e-12, sd, 1.0)

    def fit(self, X, y):
        self.schemas = {}
        x = self._input(X, fit=True)
        if y.sizes['T'] < self.spec.min_samples:
            raise ValueError('Insufficient training years for this candidate')
        self.y = y.copy(deep=True)
        self.mask = y.notnull().sum('T') >= self.spec.min_samples
        self.params = deepcopy(self.spec.parameters)
        kw = deepcopy(self.spec.init)
        args = deepcopy(self.spec.args)
        module = import_module('.' + REGISTRY[self.spec.model][0], __package__)
        cls = getattr(module, self.spec.model)
        if args:
            for owner in cls.__mro__:
                positional = [p.name for p in inspect.signature(owner.__init__).parameters.values()
                              if p.name != 'self' and p.kind in (p.POSITIONAL_ONLY, p.POSITIONAL_OR_KEYWORD)]
                if len(positional) >= len(args):
                    for key, value in zip(positional, args):
                        if key in kw:
                            raise ValueError(f'Duplicate constructor argument: {key}')
                        kw[key] = value
                    args = ()
                    break
            if args:
                raise ValueError('Use named constructor arguments for this model')
        # Caller parameters describe fixed candidates; selection belongs to CrossFit.
        if self.family == 'analog':
            kw.update(nested_tuning=False, probability_method='analog_ensemble')
            if kw.get('probability_shrinkage', 'auto') == 'auto':
                kw['probability_shrinkage'] = float(self.params.pop('probability_shrinkage', 0.0))
            kw.setdefault('verbose_analog', False)
            if not args:
                kw.setdefault('dir_to_save', '.')
                kw.setdefault('year_start', int(y["T"].dt.year.min()))
                kw.setdefault('year_forecast', int(y["T"].dt.year.max()) + 1)
                kw['predictor_vars'] = [{'variable': n} for n in self.spec.fields]
                kw.setdefault('month_of_initialization', int(y["T"].dt.month[0]))
                first = next(iter(x.values()))
                kw.setdefault('lead_time', list(first.L.values) if 'L' in first.dims else [0])
            kw.setdefault('random_state', self.spec.seed)
        elif self.family in {'unet', 'rescnn', 'unetpp', 'rise-unet'}:
            kw.update(tune_hyperparameters=False, random_state=self.spec.seed)
        elif self.family == 'hpelm':
            kw.update(random_state=self.spec.seed, n_jobs=1, cv_n_jobs=1, cluster_n_jobs=1)
        elif self.family == 'kcca':
            kw.setdefault('random_state', self.spec.seed)
        elif self.spec.model == 'WAS_mme_FullBMA':
            kw.setdefault('random_seed', self.spec.seed)
        self.model = cls(*args, **kw)
        if self.family in {'cca', 'kcca'}:
            _enable_xeofs_pickle()
        if self.family in {'unet', 'rescnn', 'unetpp', 'rise-unet', 'hpelm', 'bma'}:
            for d in ('Y', 'X'):
                if not np.array_equal(x[d], y[d]):
                    raise ValueError('MME predictor and target grids must match exactly')
        if self.family == 'cca':
            self.ym, self.ys = self._stats(y)
            yy = (y - self.ym) / self.ys
            self._linear_detrend = (
                self.spec.model not in {'WAS_CCA', 'WAS_CCA_base'}
                and bool(getattr(self.model, 'detrend', True)))
            if self.spec.model == 'WAS_CCA':
                xx, _, self.refx = self.model._eeof_split(x)
                yy, _, self.refy = self.model._eeof_split(yy)
            elif self._linear_detrend:
                from .utils import detrended_data
                xx, self.cx, self.mx = detrended_data(x, min_valid=self.spec.min_samples)
                yy, self.cy, self.my = detrended_data(yy, min_valid=self.spec.min_samples)
            else:
                xx = x
            self.model.fit_cca(xx, yy)
        elif self.family == 'kcca':
            xx = self.model._reduce_fit(x) if self.spec.model == 'WAS_EOF_KCCA' else x
            self.ym, self.ys = self._stats(y) if self.model.standardize else (0., 1.)
            self.ktrain = xx.mean('T', skipna=True).expand_dims(T=xx['T'].isel(T=[0]))
            self.model.fit_cca(xx, (y-self.ym)/self.ys, best_params=self.params)
        elif self.family == 'bma':
            params = dict(self.params)
            if self.spec.model == 'WAS_mme_FullBMA':
                family = params.pop('distribution', 'normal')
                code = self.model.DIST_NAME_TO_CODE.get(family)
                if code is None:
                    raise ValueError('Choose a fixed normal, gamma or lognormal distribution candidate')
                params['dist_map'] = xr.full_like(y.isel(T=0, drop=True), code, dtype=float).where(self.mask)
            self.model.fit(x, y, min_samples=self.spec.min_samples, **params)
        elif self.family in {'unet', 'rescnn', 'unetpp', 'rise-unet', 'hpelm'}:
            self.xm, self.xs = self._stats(x)
            self.ym, self.ys = self._stats(y)
            xx, yy = (x-self.xm)/self.xs, (y-self.ym)/self.ys
            if self.family in {'unet', 'rescnn', 'unetpp', 'rise-unet'}:
                # Static climatology channels need the PHYSICAL training
                # climatology; the network itself sees standardized fields.
                self.model.compute_model(xx, yy, xx.isel(T=[0]), best_params=self.params,
                                         static_climatology=(self.ym, self.ys))
            else:
                self._fit_hpelm(xx, yy)
        else:
            self._query_cache = {}
            self.xtrain = {k: v.copy(deep=True) for k, v in x.items()}
            self.history = {k: X[self.spec.history_fields.get(k, k)].copy(deep=True) for k in x}
            for k, a in self.history.items():
                if not np.array_equal(a["T"].values, y["T"].values):
                    raise ValueError(f'Analogue history times differ: {k}')
            if self.params:
                raise ValueError(f'Unknown analogue parameters: {list(self.params)}; use init')
        return self

    def _fit_hpelm(self, x, y):
        bp = self.params or dict(neurons=int(self.model.neurons_range[0]),
                                activation=self.model.activation_options[0], norm=float(self.model.norm_range[0]))
        if set(bp) != {'neurons', 'activation', 'norm'}:
            raise ValueError('hpELM needs explicit neurons, activation, norm')
        self.clusters = climate_cluster_map(self.y, n_clusters=self.model.n_clusters,
                                           random_state=self.spec.seed, min_valid=self.spec.min_samples)
        xx = x.transpose('T', 'Y', 'X', 'M').values
        yy = y.values
        self.models = {}
        for c in np.unique(self.clusters.values[np.isfinite(self.clusters.values)]):
            mask = np.broadcast_to(self.clusters.values == c, yy.shape)
            valid = mask & np.isfinite(yy) & np.isfinite(xx).all(-1)
            indices = np.flatnonzero(valid.ravel())[::self.model.spatial_subsample]
            if len(indices) < self.spec.min_samples:
                continue
            a, b = xx.reshape(-1, xx.shape[-1])[indices], yy.ravel()[indices, None]
            hp, _ = self.model._fit_predict_cluster(bp, a, b, a[:1])
            self.models[c] = hp
        if not self.models:
            raise ValueError('No hpELM cluster has enough finite training samples')
        self.model.hpelm = self.models

    def predict(self, X):
        if self.model is None:
            raise ValueError('Call fit before predict')
        x = self._input(X)
        if self.family == 'cca':
            if self.spec.model == 'WAS_CCA':
                xx = x - self.refx
            elif self._linear_detrend:
                from .utils import apply_detrend_data
                xx = x - apply_detrend_data(x, self.cx, self.mx)
            else:
                xx = x
            pred = self.model._predict_prepared(self.model.preprocess_test_data(xx)[0])
            if self.spec.model == 'WAS_CCA':
                pred = pred + self.refy
            elif self._linear_detrend:
                pred = pred + apply_detrend_data(pred, self.cy, self.my)
            pred = pred*self.ys + self.ym
        elif self.family == 'kcca':
            xx = self.model._reduce_apply(x) if self.spec.model == 'WAS_EOF_KCCA' else x
            prepared = self.model.preprocess_forecast_data(xx, self.ktrain)
            pred = self.model._predict_field(prepared)*self.ys + self.ym
        elif self.family == 'bma':
            pred = self.model.predict_probabilistic(x, obs_for_terciles=self.y,
                                                    quantiles=())['predictive_mean']
        elif self.family in {'unet', 'rescnn', 'unetpp', 'rise-unet'}:
            pred = self.model.predict_fitted((x-self.xm)/self.xs)*self.ys+self.ym
        elif self.family == 'hpelm':
            xx = ((x-self.xm)/self.xs).transpose('T', 'Y', 'X', 'M').values
            shape = xx.shape[:-1]
            out = np.full(shape, np.nan)
            for c, hp in self.models.items():
                valid = np.broadcast_to(self.clusters.values == c, shape) & np.isfinite(xx).all(-1)
                if valid.any():
                    out[valid] = hp.predict(xx[valid]).ravel()
            pred = xr.DataArray(out, dims=('T', 'Y', 'X'), coords={'T': x["T"], 'Y': self.y.Y, 'X': self.y.X})*self.ys+self.ym
        else:
            pred, _ = self._analog_predict(x)
        return apply_physical_support(pred.transpose('T', 'Y', 'X').where(self.mask), self.y)

    def _analog_predict(self, x, thresholds=None):
        dets, probs = [], []
        times = next(iter(x.values()))["T"]
        for t in times.values:
            test = {k: a.sel(T=[t]) for k, a in x.items()}
            hc = {k: xr.concat([self.xtrain[k], a], dim='T').sortby('T') for k, a in test.items()}
            hc = {k: a.rename({'M': 'number'}) if 'M' in a.dims else a for k, a in hc.items()}
            hc = {k: a if 'L' in a.dims else a.expand_dims(L=[0]) for k, a in hc.items()}
            self.model._analog_raw_cache = (self.history, hc, {})
            blank = xr.full_like(self.y.isel(T=[0]), np.nan).assign_coords(T=[t])
            p = xr.concat([self.y, blank], dim='T').sortby('T')
            ti = int(np.flatnonzero(p["T"].values == t)[0])
            tr = np.flatnonzero(p["T"].values != t)
            if len(tr) != self.y.sizes['T']:
                raise ValueError('Analogue prediction dates must be disjoint from training')
            digest = hashlib.sha256()
            for k, a in test.items():
                digest.update(k.encode())
                digest.update(np.asarray(a.values).tobytes())
            key = (str(t), digest.hexdigest())
            year = int(p['T'].dt.year[ti])
            cached = self._query_cache.get(key)
            if cached is None:
                d = self.model.compute_model(p, tr, [ti], nested_tuning=False)
                caches = {attr: deepcopy(getattr(self.model, attr).get(year))
                          for attr in ('_analog_member_cv_cache', '_regional_cv_cache')
                          if hasattr(self.model, attr)}
                self._query_cache[key] = (d.copy(deep=True), caches)
            else:
                d, caches = cached
                for attr, value in caches.items():
                    getattr(self.model, attr)[year] = deepcopy(value)
            dets.append(d)
            if thresholds is not None:
                # Native selection/weights/shrinkage, shared exact class boundaries.
                def probability(sel):
                    a = self.y.sel(T=self.y["T"].dt.year.isin(sel['analog_years']))
                    year_to_weight = dict(zip(sel['analog_years'], sel['weights']))
                    w = xr.DataArray([year_to_weight[int(v)] for v in a["T"].dt.year.values], dims='T', coords={'T': a["T"]})
                    lo, hi = [thresholds.isel(quantile=i, drop=True) for i in range(2)]
                    valid = a.notnull()
                    den = w.where(valid, 0).sum('T')
                    pp = xr.concat([(w.where(valid & ev, 0).sum('T')/den).where(den > 0)
                                    for ev in (a <= lo, (a > lo) & (a <= hi), a > hi)],
                                   dim=xr.IndexVariable('probability', ['PB','PN','PA']))
                    lam = sel['probability_shrinkage']
                    return (1-lam)*pp + lam/3
                year = int(p["T"].dt.year[ti])
                if hasattr(self.model, '_regional_cv_cache'):
                    pp = xr.concat([xr.full_like(self.mask, 1/3, dtype=float)]*3,
                                   dim=xr.IndexVariable('probability',['PB','PN','PA']))
                    for key, mask in self.model._normalise_region_masks(self.y).items():
                        pp = xr.where(mask, probability(self.model._regional_cv_cache[year][key]), pp)
                else:
                    selections = self.model._analog_member_cv_cache[year]
                    pp = self.model._mean_probabilities([probability(s) for s in selections])
                probs.append(pp.expand_dims(T=[t]))
        return xr.concat(dets, dim='T'), xr.concat(probs, dim='T') if probs else None

    def predict_proba(self, X, thresholds):
        x = self._input(X)
        if self.family == 'bma':
            p = self.model.predict_probabilistic(x, clim_terciles=thresholds, quantiles=())['tercile_probability']
            p = p.assign_coords(probability=['PB', 'PN', 'PA'])
        elif self.family == 'analog':
            _, p = self._analog_predict(x, thresholds)
        else:
            raise ValueError('This family uses nested residual calibration')
        return p.transpose('probability', 'T', 'Y', 'X').where(self.mask)


class _LegacyAnalogAdapter:
    """Inductive migration of the two historical analogue entry points.

    SOM/KMeans/agglomeration are fitted on training states only. The query is
    assigned afterwards. Bias/correlation retain the historical 0.5/0.3
    thresholds, over lead vectors supplied explicitly through L.
    """
    def __init__(self, spec):
        self.spec = spec
        self.model = None
        self.family = 'legacy_analog'

    def _matrix(self, X, fit=False):
        cols = []
        for name in self.spec.fields:
            a = X[name]
            if set(a.dims) - {'T', 'L', 'Y', 'X'}:
                raise ValueError('Historical analogues accept T,L,Y,X or T,Y,X')
            if 'L' not in a.dims:
                a = a.expand_dims(L=[0])
            a = a.transpose('T','L','Y','X')
            schema = {d: a[d].values.copy() for d in ('L','Y','X')}
            if fit:
                self.schemas[name] = schema
            elif any(not np.array_equal(v,self.schemas[name][d]) for d,v in schema.items()):
                raise ValueError('Analogue lead/grid identity changed')
            cols.append(a.values.reshape(a.sizes['T'],-1))
        return np.concatenate(cols,axis=1)

    def fit(self, X, y):
        from .was_crossfit import _FeatureTransform
        from sklearn.decomposition import PCA
        self.y = y.copy(deep=True)
        self.mask = y.notnull().sum('T') >= self.spec.min_samples
        self.schemas = {}
        x = self._matrix(X,True)
        kw = deepcopy(self.spec.init)
        module = import_module('.was_analog',__package__)
        cls = getattr(module,self.spec.model)
        if self.spec.args:
            params=list(inspect.signature(cls.__init__).parameters)[1:]
            for k,v in zip(params,self.spec.args):
                if k in kw: raise ValueError('Duplicate analogue constructor argument')
                kw[k]=v
        kw.setdefault('dir_to_save','.')
        kw.setdefault('year_start',int(y['T'].dt.year.min()))
        kw.setdefault('year_forecast',int(y['T'].dt.year.max())+1)
        self.model = cls(**kw)
        self.method = self.model.method_analog
        if self.method not in {'som','bias_based','cor_based','pca_based','kmeans_based','agglomerative_based'}:
            raise ValueError('Unknown historical analogue method')
        if self.spec.history_fields:
            raise ValueError('Historical analogue adapter uses a common predictor space; use scientific analogues for separate reanalysis/GCM states')
        self.raw = next(iter(X.values()))
        if 'L' not in self.raw.dims: self.raw=self.raw.expand_dims(L=[0])
        self.raw=self.raw.transpose('T','L','Y','X').copy(deep=True)
        self.pre = _FeatureTransform(detrend=self.model.detrend)
        self.z = self.pre.fit_transform(x,y['T'].dt.year.values.astype(float))
        self.pca = None
        if self.method in {'som','pca_based','kmeans_based','agglomerative_based'}:
            ev=self.model.eof_explained_var
            rank=min(self.z.shape)-1
            self.pca=PCA(n_components=min(rank, int(ev)) if ev>=1 else float(ev),svd_solver='full')
            self.z=self.pca.fit_transform(self.z)
        if self.method=='som':
            from minisom import MiniSom
            grid=(4,4) if set(self.model.some_grid_size)=={None} else self.model.some_grid_size
            self.clusterer=MiniSom(*grid,input_len=self.z.shape[1],sigma=self.model.some_sigma,
                                  learning_rate=self.model.some_learning_rate,
                                  neighborhood_function=self.model.some_neighborhood_function,
                                  random_seed=self.spec.seed)
            self.clusterer.random_weights_init(self.z)
            self.clusterer.train_random(self.z,self.model.some_num_iteration)
            self.labels=np.array([self.clusterer.winner(v) for v in self.z])
        elif self.method=='kmeans_based':
            from sklearn.cluster import KMeans
            self.clusterer=KMeans(n_clusters=min(self.model.n_clusters,len(self.z)),random_state=self.spec.seed,n_init=10).fit(self.z)
            self.labels=self.clusterer.labels_
        elif self.method=='agglomerative_based':
            from sklearn.cluster import AgglomerativeClustering
            self.clusterer=AgglomerativeClustering(n_clusters=min(self.model.n_clusters,len(self.z)),metric=self.model.affinity,linkage=self.model.linkage)
            self.labels=self.clusterer.fit_predict(self.z)
        return self

    def predict(self,X):
        from scipy.spatial.distance import cdist
        x=self._matrix(X)
        times=next(iter(X.values()))['T']
        z=self.pre.transform(x,times.dt.year.values.astype(float))
        if self.pca is not None:z=self.pca.transform(z)
        outs=[]
        first=X[self.spec.fields[0]]
        if 'L' not in first.dims:first=first.expand_dims(L=[0])
        first=first.transpose('T','L','Y','X')
        for i,v in enumerate(z):
            if self.method=='som':
                bmu=np.array(self.clusterer.winner(v))
                dist=np.linalg.norm(self.labels-bmu,axis=1)
                take=np.flatnonzero(dist<=self.model.radius)
                if not len(take):take=np.flatnonzero(dist==dist.min())
            elif self.method=='kmeans_based':
                take=np.flatnonzero(self.labels==self.clusterer.predict(v[None])[0])
            elif self.method=='agglomerative_based':
                scores=[]
                groups=np.unique(self.labels)
                for c in groups:
                    a=self.z[self.labels==c]
                    if self.model.linkage=='ward':
                        score=len(a)/(len(a)+1)*np.sum((a.mean(0)-v)**2)
                    else:
                        d=cdist(a,v[None],metric=self.model.affinity).ravel()
                        score={'single':np.min,'complete':np.max,'average':np.mean}[self.model.linkage](d)
                    scores.append(score)
                take=np.flatnonzero(self.labels==groups[np.argmin(scores)])
            elif self.method in {'bias_based','cor_based'}:
                ref=first.isel(T=i,drop=True)
                if self.method=='bias_based':
                    scores=(abs(self.raw-ref).mean('L')<.5).sum(('Y','X')).values
                else:
                    if self.raw.sizes['L']<2:
                        raise ValueError('cor_based needs >=2 explicit lead samples in L')
                    scores=(xr.corr(self.raw,ref,dim='L')>.3).sum(('Y','X')).values
                take=np.argsort(-scores,kind='stable')[:2]
            else:
                a=self.z-self.z.mean(1,keepdims=True); b=v-v.mean()
                den=np.linalg.norm(a,axis=1)*np.linalg.norm(b)
                scores=np.divide(a@b,den,out=np.full(len(a),-np.inf),where=den>0)
                if not np.isfinite(scores).any():
                    scores=-np.linalg.norm(self.z-v,axis=1)
                take=np.argsort(-scores,kind='stable')[:2]
            outs.append(self.y.isel(T=take).mean('T').expand_dims(T=times.isel(T=[i])))
        return apply_physical_support(xr.concat(outs,dim='T').where(self.mask),self.y)
