from wass2s.was_scientific_utils import apply_physical_support, align_predictor_target
"""Machine-learning regression models for seasonal forecasting.

All classes expose the standard
``compute_model`` / ``compute_prob`` / ``forecast`` interface and are
compatible with :class:`~wass2s.was_cross_validate.WAS_Cross_Validator`.

Classes
-------
BaseOptimizer
    Unified hyperparameter optimizer supporting grid search, random search,
    and Optuna Bayesian optimization.
WAS_PolynomialRegression
    Polynomial feature expansion followed by ordinary least squares.
WAS_MARS_Model
    Multivariate adaptive hinge regression implemented in NumPy (see MARS notes).
WAS_LogisticRegression_Model
    Pixel-wise multinomial logistic regression for direct tercile
    classification.
WAS_PoissonRegression
    Pixel-wise Poisson GLM for count / precipitation data.
WAS_NonNeural_Base
    Encompassing others non_neural regression : WAS_RandomForest_Model ...
"""
# Machine Learning and Statistical Modeling
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor
from sklearn.svm import SVR, SVC
from sklearn.model_selection import GridSearchCV, RandomizedSearchCV, cross_val_score
from sklearn.preprocessing import PolynomialFeatures, StandardScaler, RobustScaler, MinMaxScaler
from sklearn.base import clone
from sklearn.cluster import KMeans
from sklearn.pipeline import Pipeline
from sklearn.compose import TransformedTargetRegressor
# xgboost / lightgbm are optional: imported lazily inside the estimator
# factories (WAS_XGBoost_Model, WAS_XGBoost_Classifier_Model, LightGBM ones)
# so that this module, and everything star-importing it, loads without them.
from wass2s.utils import *
import optuna
from scipy.stats import uniform, loguniform, randint
from sklearn.linear_model import LogisticRegression

# Data Manipulation and Analysis
import xarray as xr
from .was_probability_reference import climatological_terciles
from .was_tuning_support import geographical_cluster_map, fold_local_regression_score
import numpy as np
import pandas as pd

# Signal Processing and Interpolation
import scipy.signal as sig
from scipy.interpolate import CubicSpline
from scipy.optimize import fsolve
from scipy.special import gamma as gamma_function

# Statistical Distributions
from scipy import stats
from scipy.stats import (
    norm, lognorm, expon, gamma, weibull_min,
    t as t_dist, poisson, nbinom
)

# Parallel Computing
from multiprocessing import cpu_count
from dask.distributed import Client
import dask.array as da


# ---------------------------------------------------------------------------
# Shared Dask client: created once per process and reused across ALL folds and
# models, instead of spinning up (and tearing down) a fresh LocalCluster on
# every compute_model — which caused nanny-join hangs at close() and worker
# oversubscription (stalled folds). Closed once, at interpreter exit.
# ---------------------------------------------------------------------------
_SHARED_CLIENT = None


def _bounded_close(client):
    """Close a client (and its LocalCluster) with a bounded timeout, swallowing
    teardown errors so a slow nanny shutdown can never abort the run."""
    if client is None:
        return
    import contextlib
    cluster = getattr(client, "cluster", None)
    with contextlib.suppress(Exception):
        client.close(timeout=10)
    if cluster is not None:
        with contextlib.suppress(Exception):
            cluster.close(timeout=10)


def _safe_close_client(client):
    """Intentional no-op. Every client here comes from _get_compute_client and
    is process-wide and reused; it must NOT be torn down per fold/call. The
    shared client is closed exactly once, at interpreter exit, by the atexit
    hook registered in _get_compute_client (or reaped by the OS/Slurm cgroup
    when the CLI ends via os._exit). Call sites keep calling this so their
    'finally:' blocks stay valid without ever closing the shared client."""
    return


def _get_compute_client(nb_cores):
    """Return a process-wide Dask client, created lazily on first use and reused
    afterwards. Returns None if a distributed client cannot be created, in which
    case .compute() falls back to dask's default (threaded) scheduler."""
    global _SHARED_CLIENT
    try:
        from dask.distributed import Client, LocalCluster, get_client
        try:
            return get_client()
        except Exception:
            pass
        if _SHARED_CLIENT is not None:
            try:
                if _SHARED_CLIENT.status == "running":
                    return _SHARED_CLIENT
            except Exception:
                _SHARED_CLIENT = None
        cluster = LocalCluster(
            n_workers=max(1, int(nb_cores)),
            threads_per_worker=1,
            processes=True,
            dashboard_address=None,
            memory_limit=0,
        )
        _SHARED_CLIENT = Client(cluster)
        import atexit
        atexit.register(_bounded_close, _SHARED_CLIENT)
        return _SHARED_CLIENT
    except Exception:
        return None

# Typing and Utilities
from typing import List, Tuple, Optional
from collections import defaultdict

# Warnings
import warnings
# Do not suppress convergence and numerical warnings process-wide.

import numpy as np
import pandas as pd
import xarray as xr

from dask.distributed import Client
from sklearn.cluster import KMeans
from sklearn.model_selection import cross_val_score

from scipy.stats import (norm, lognorm, expon, gamma, weibull_min, t,
                         poisson, nbinom, randint, uniform, loguniform)
from scipy.special import gamma as gamma_function
from scipy.optimize import fsolve

import optuna


def _climatology_positions(time_coord, clim_year_start, clim_year_end, min_years=3):
    """Positional ``[start, stop)`` of the climatology window along a sorted T axis.

    Label slicing (``.sel(T=slice(...))``) tolerates a window that exceeds the
    record, e.g. a cross-validation training fold missing its first or last
    years, where ``get_index('T').get_loc(str(year))`` raises ``KeyError``.
    The declared reference is never silently replaced by another period.
    """
    index = time_coord.get_index("T")
    if index.has_duplicates or not index.is_monotonic_increasing:
        raise ValueError("Climatology T coordinates must be unique and sorted")
    window = time_coord.sel(T=slice(str(clim_year_start), str(clim_year_end)))
    if int(window.sizes["T"]) < min_years:
        raise ValueError(f"At least {min_years} observations are required in the requested climatology interval")
    positions = np.flatnonzero(np.isin(time_coord.values, window.values))
    return int(positions[0]), int(positions[-1]) + 1


def _aligned_training(predictor, predictand):
    """Require the caller to provide matching valid-time coordinates; never relabel."""
    for data in (predictor, predictand):
        index = data.get_index("T")
        if index.has_duplicates or not index.is_monotonic_increasing:
            raise ValueError("Training T coordinates must be unique and sorted")
    predictor = align_predictor_target(predictor, predictand)
    return xr.align(predictor, predictand, join="exact", copy=False)


def _forecast_times(predictor):
    """Retain supplied valid times, including an explicitly supplied scalar T."""
    if "T" not in predictor.dims:
        if "T" not in predictor.coords or predictor.coords["T"].size != 1:
            raise ValueError("Forecast predictors must include their actual valid-time T coordinate")
        value = np.asarray(predictor.coords["T"].values).reshape(-1)[0]
        predictor = predictor.drop_vars("T").expand_dims(T=[value])
    index = predictor.get_index("T")
    if len(index) == 0 or index.has_duplicates or not index.is_monotonic_increasing:
        raise ValueError("Forecast T coordinates must be nonempty, unique and sorted")
    return predictor.transpose(..., "T", "features")


def _spatial_classifier(model, X_train, labels, X_test, parameters=()):
    """One fit per cell and test date without train/test T alignment collisions."""
    X_train, labels = _aligned_training(X_train, labels)
    X_test = _forecast_times(X_test)
    cy = max(1, int(np.ceil(labels.sizes["Y"] / max(1, model.nb_cores))))
    cx = max(1, int(np.ceil(labels.sizes["X"] / max(1, model.nb_cores))))
    labels = labels.transpose("T", "Y", "X").chunk({"T": -1, "Y": cy, "X": cx})
    results = []
    for i in range(X_test.sizes["T"]):
        result = xr.apply_ufunc(model.fit_predict, X_train, labels,
            X_test.isel(T=i, drop=True), *parameters,
            input_core_dims=[("T", "features"), ("T",), ("features",)] + [() for _ in parameters],
            output_core_dims=[("probability",)], vectorize=True, dask="parallelized",
            output_dtypes=[float], dask_gufunc_kwargs={"output_sizes": {"probability": 3}, "allow_rechunk": True})
        results.append(result.expand_dims(T=[X_test.coords["T"].values[i]]))
    result = xr.concat(results, dim="T").assign_coords(probability=["PB", "PN", "PA"])
    result.attrs.update(probability_calibration="native_classifier",
                        residual_calibration="not_applicable")
    if type(model).__name__ == "WAS_SVC_Classifier_Model":
        result.attrs["native_probability_calibration"] = "libsvm internal 5-fold Platt calibration; independent of outer/nested residual controls"
    from threadpoolctl import threadpool_limits
    with threadpool_limits(limits=1):
        return result.transpose("probability", "T", "Y", "X").compute()


def _spatial_regression(model, X_train, y_train, X_test, y_test, parameters=()):
    """Preserve singleton spatial/features axes and support one or several test dates."""
    X_train, y_train = _aligned_training(X_train, y_train)
    X_train = X_train.transpose(..., "T", "features")
    y_train = y_train.transpose("T", "Y", "X")
    cy = max(1, int(np.ceil(y_train.sizes["Y"] / max(1, model.nb_cores))))
    cx = max(1, int(np.ceil(y_train.sizes["X"] / max(1, model.nb_cores))))
    y_train = y_train.chunk({"T": -1, "Y": cy, "X": cx})
    has_time = "T" in X_test.dims
    tests = range(X_test.sizes["T"]) if has_time else range(1)
    outputs = []
    for i in tests:
        xt = X_test.isel(T=i, drop=True) if has_time else X_test
        if y_test is None:
            yt = xr.full_like(y_train.isel(T=0, drop=True), np.nan)
        elif "T" in y_test.dims:
            yt = y_test.isel(T=i, drop=True)
        else:
            yt = y_test.drop_vars("T", errors="ignore")
        value = xr.apply_ufunc(
            model.fit_predict, X_train, y_train, xt, yt, *parameters,
            input_core_dims=[("T", "features"), ("T",), ("features",), ()]
                            + [() for _ in parameters],
            output_core_dims=[("output",)], vectorize=True, dask="parallelized",
            output_dtypes=[float], dask_gufunc_kwargs={"output_sizes": {"output": 2},
                                                     "allow_rechunk": True},
        ).isel(output=1, drop=True)
        if has_time:
            value = value.expand_dims(T=[X_test.coords["T"].values[i]])
        outputs.append(value)
    if not outputs:
        raise ValueError("X_test must contain at least one forecast time")
    result = xr.concat(outputs, dim="T") if has_time else outputs[0]
    try:
        from dask.distributed import get_client
        get_client()
    except ValueError:
        from threadpoolctl import threadpool_limits
        with threadpool_limits(limits=1):
            return result.compute(scheduler="threads", num_workers=max(1, int(model.nb_cores)))
    return result.compute()


def _regression_pair(estimator, x, y, x_test, y_test):
    """Fit only finite training pairs; missing targets remain missing."""
    x, y = np.asarray(x, float), np.asarray(y, float).reshape(-1)
    xt = np.asarray(x_test, float).reshape(1, -1)
    if x.ndim == 1:
        x = x[:, None]
    if x.shape[0] != y.size or x.shape[1] != xt.shape[1]:
        raise ValueError("Incompatible training/test feature shapes")
    good = np.isfinite(y) & np.isfinite(x).all(axis=1)
    if good.sum() < 2 or not np.isfinite(xt).all():
        return np.array([np.nan, np.nan])
    neighbors = {name: min(int(value), int(good.sum()))
        for name, value in estimator.get_params(deep=True).items()
        if name.split("__")[-1] == "n_neighbors"} if hasattr(estimator, "get_params") else {}
    if neighbors:
        estimator.set_params(**neighbors)
    estimator.fit(x[good], y[good])
    prediction = float(np.asarray(estimator.predict(xt)).reshape(-1)[0])
    error = np.nan if y_test is None else float(np.asarray(y_test).reshape(-1)[0]) - prediction
    return np.array([error, prediction])

def _residual_probabilities(forecast, residuals, t1, t2, nonneg=False):
    """Shared empirical CDF: at least three errors, fixed bounds and support."""
    from .was_linear_models import _residual_probabilities as empirical_cdf
    return empirical_cdf(forecast, residuals, t1, t2, nonneg=nonneg)


def _operational_forecast(model, Predictant, clim_year_start, clim_year_end,
                          Predictor, hindcast_det, Predictor_for_year, parameters=(),
                          best_code_da=None, best_shape_da=None, best_loc_da=None,
                          best_scale_da=None, raw_target=False):
    """Fit available observations and calibrate future predictions from OOF errors.

    ``hindcast_det`` must be an out-of-fold hindcast in the target's physical
    units, independently obtained. This operational calibration must not be used
    to score those same hindcast years; use the nested Cross_Validator for that.
    Forecast dates are retained exactly as supplied in Predictor_for_year.
    """
    Predictor, Predictant = _aligned_training(Predictor, Predictant)
    Predictor_for_year = _forecast_times(Predictor_for_year)
    years = Predictant.coords["T"].dt.year
    reference = Predictant.where((years >= int(clim_year_start)) & (years <= int(clim_year_end)), drop=True)
    if reference.sizes.get("T", 0) == 0:
        raise ValueError("No training observations in the requested climatology interval")
    center = reference.mean("T", skipna=True)
    scale = reference.std("T", skipna=True)
    scale = xr.where(scale > 0, scale, 1.0)
    fit_y = Predictant if raw_target else (Predictant-center)/scale
    prediction = _spatial_regression(model, Predictor, fit_y, Predictor_for_year,
                                     None, parameters)
    if not raw_target:
        prediction = prediction*scale + center
    prediction = apply_physical_support(prediction, Predictant).transpose("T", "Y", "X")
    if "output" in hindcast_det.dims:
        hindcast_det = hindcast_det.isel(output=1, drop=True)
    obs, hindcast = xr.align(Predictant, hindcast_det, join="inner")
    if obs.sizes.get("T", 0) < 3:
        raise ValueError("At least three matching out-of-fold hindcast dates are required")
    residuals = (obs-hindcast).rename(T="sample")
    event_reference = Predictant if getattr(model, "tercile_reference", "climatology") == "training" else reference
    q = climatological_terciles(event_reference)[0]
    t1, t2 = q.isel(quantile=0, drop=True), q.isel(quantile=1, drop=True)
    nonnegative = bool(apply_physical_support(xr.DataArray(-1.), Predictant) == 0.)
    if model.dist_method == "nonparam":
        probability = xr.apply_ufunc(_residual_probabilities, prediction, residuals, t1, t2,
            input_core_dims=[("T",), ("sample",), (), ()],
            output_core_dims=[("probability", "T")], vectorize=True, dask="parallelized",
            kwargs={"nonneg":nonnegative}, output_dtypes=[float],
            dask_gufunc_kwargs={"output_sizes":{"probability":3}, "allow_rechunk":True})
    elif model.dist_method == "bestfit":
        if any(v is None for v in (best_code_da, best_shape_da, best_loc_da, best_scale_da)):
            raise ValueError("bestfit requires distribution parameters fitted on available training data")
        # Event bounds are empirical for either declared reference. The fitted
        # probability family must not redefine the meaning of PB/PN/PA.
        variance = residuals.var("sample", skipna=True, ddof=1)
        residual_count = residuals.count("sample")
        dof = xr.where(residual_count > 4, residual_count-1, 3)
        calibrated_mean = apply_physical_support(prediction+residuals.mean("sample", skipna=True), Predictant)
        # External/legacy model kernels may implement the six-argument API.
        # New native kernels accept support explicitly; only forward it when
        # advertised rather than catching a TypeError from inside the kernel.
        import inspect
        kernel = model.calculate_tercile_probabilities_bestfit
        try:
            parameters = inspect.signature(kernel).parameters
            supports_nonneg = "nonneg" in parameters or any(
                p.kind == inspect.Parameter.VAR_KEYWORD for p in parameters.values())
        except (TypeError, ValueError):
            supports_nonneg = False
        probability = xr.apply_ufunc(kernel,
            calibrated_mean, variance, t1, t2, best_code_da, dof,
            input_core_dims=[("T",),(),(),(),(),()], output_core_dims=[("probability","T")],
            kwargs={"nonneg":nonnegative} if supports_nonneg else {},
            vectorize=True, dask="parallelized", output_dtypes=[float],
            dask_gufunc_kwargs={"output_sizes":{"probability":3}, "allow_rechunk":True})
        # A zero residual variance is a degenerate predictive distribution.
        deterministic = xr.concat([xr.where(calibrated_mean<=t1,1.,0.),
            xr.where((calibrated_mean>t1)&(calibrated_mean<=t2),1.,0.),
            xr.where(calibrated_mean>t2,1.,0.)],dim="probability").where(np.isfinite(calibrated_mean))
        probability = xr.where(variance == 0, deterministic, probability)
    else:
        raise ValueError("dist_method must be 'nonparam' or 'bestfit'")
    probability = probability.where(residuals.count("sample") >= 3)
    probability = probability.clip(min=0, max=1)
    probability = probability / probability.sum("probability", skipna=False)
    probability = probability.assign_coords(probability=["PB","PN","PA"])
    probability.attrs.update(probability_calibration="historical_oof_operational",
        probability_bias_correction="mean of historical out-of-fold residuals",
        tercile_reference=getattr(model, "tercile_reference", "climatology"))
    return prediction, probability.transpose("probability","T","Y","X").compute()


def _report_tuned(tag, verbose=True, cluster_da=None, **fields):
    """Log the tuned hyperparameters of a spatially varying model.

    Unlike the MME learners, these classes return hyperparameter FIELDS
    (an alpha map, a Dataset of per-cell parameters) rather than a dict per
    cluster, so there is nothing to print verbatim. What matters operationally
    is the same thing: whether the search landed on strong regularisation
    (a near-linear, heavily shrunk model) or on a flexible one, and whether it
    is spatially coherent. This prints the median and range of each field, plus
    the per-cluster median when a cluster map is supplied.

    Silent tuning was the norm across was_linear_models, was_machine_learning
    and was_cca_2: a run gave no way to tell a converged search from one that
    returned all-NaN.
    """
    if not verbose:
        return
    import numpy as _np
    for name, field in fields.items():
        if field is None:
            continue
        try:
            values = _np.asarray(getattr(field, "values", field), dtype=float)
            finite = values[_np.isfinite(values)]
            if finite.size == 0:
                print(f"[{tag}] {name}: aucune valeur finie (recherche échouée ?)", flush=True)
                continue
            print(f"[{tag}] {name}: médiane={_np.median(finite):.4g} "
                  f"[{finite.min():.4g}, {finite.max():.4g}] "
                  f"sur {finite.size}/{values.size} cellules", flush=True)
            if cluster_da is not None:
                labels = _np.asarray(getattr(cluster_da, "values", cluster_da), dtype=float)
                if labels.shape == values.shape:
                    for c in sorted({v for v in _np.unique(labels) if _np.isfinite(v)}):
                        sel = values[(labels == c) & _np.isfinite(values)]
                        if sel.size:
                            print(f"[{tag}]   cluster {int(c)}: {name}"
                                  f"={_np.median(sel):.4g}", flush=True)
        except Exception as exc:
            print(f"[{tag}] {name}: rapport impossible ({exc!r})", flush=True)


optuna.logging.set_verbosity(optuna.logging.WARNING)
def _classifier_probabilities(estimator, x, y, x_test):
    """Encode each training partition and map probabilities back to PB/PN/PA."""
    x, y = np.asarray(x, float), np.asarray(y, float).reshape(-1)
    xt = np.asarray(x_test, float)
    if xt.ndim == 1:
        xt = xt.reshape(1, -1)
    out = np.full((len(xt), 3), np.nan)
    valid_test = np.isfinite(xt).all(axis=1)
    valid = np.isfinite(y) & np.isfinite(x).all(axis=1)
    if not valid.any() or not valid_test.any():
        return out
    x, y = x[valid], y[valid]
    if not np.isin(y, [0, 1, 2]).all():
        raise ValueError("Tercile classifiers require labels 0, 1, 2 or NaN")
    classes, encoded = np.unique(y.astype(int), return_inverse=True)
    if len(classes) == 1:
        out[valid_test] = 0.0
        out[valid_test, classes[0]] = 1.0
        return out
    neighbors = {name: min(int(value), len(y)) for name, value in estimator.get_params(deep=True).items()
                 if name.split("__")[-1] == "n_neighbors"}
    if neighbors:
        estimator.set_params(**neighbors)
    estimator.fit(x, encoded)
    probabilities = estimator.predict_proba(xt[valid_test])
    fitted_classes = np.asarray(estimator.classes_, int)
    if probabilities.shape[1] != len(fitted_classes):
        raise ValueError("Estimator probability columns disagree with its fitted classes")
    mapped = np.zeros((valid_test.sum(), 3))
    mapped[:, classes[fitted_classes]] = probabilities
    out[valid_test] = mapped
    return out


def _regression_cv_score(owner, X, y, params):
    """Fit each cell's scaling and regional proxy afresh in every inner fold."""
    return fold_local_regression_score(owner._make_estimator(**params), X, y, owner.inner_cv)


def _classification_folds(cv, X, y):
    """Respect custom splitters; cap integer folds for short/rare-class samples."""
    from sklearn.model_selection import check_cv, KFold, StratifiedKFold
    if isinstance(cv, (int, np.integer)):
        count = min(int(cv), len(y))
        if count < 2:
            raise ValueError("At least two classification folds are required")
        _, sizes = np.unique(y, return_counts=True)
        splitter = (StratifiedKFold(min(count, int(sizes.min()))) if len(sizes)>1 and sizes.min()>=2
                    else KFold(count))
    else:
        splitter = check_cv(cv, y=y, classifier=True)
    return list(splitter.split(X, y))


def _classification_score(estimator, X, y, folds, scoring):
    from sklearn.metrics import log_loss, accuracy_score
    scores = []
    for train, valid in folds:
        probability = _classifier_probabilities(clone(estimator), X[train], y[train], X[valid])
        if not np.isfinite(probability).all():
            raise ValueError("No finite inner validation probabilities")
        if scoring == "neg_log_loss":
            scores.append(-log_loss(y[valid], probability, labels=[0,1,2]))
        elif scoring == "accuracy":
            scores.append(accuracy_score(y[valid], probability.argmax(axis=1)))
        else:
            raise ValueError("Classifier tuning supports scoring='neg_log_loss' or 'accuracy'")
    return float(np.mean(scores))


def _classifier_cv_score(owner, X, y, params):
    """Score fixed tercile probabilities even when an inner fold lacks a class."""
    return _classification_score(owner._make_estimator(**params), X, y,
                                 _classification_folds(owner.inner_cv, X, y), owner.scoring)


class BaseOptimizer:
    """One CV protocol across search backends, fresh estimator per fold, bounded parallelism.

    ``cv`` accepts sklearn splitters or an explicit list of folds. Supply a temporal
    splitter for autocorrelated series. This is the inner search only: the outer
    evaluation must rerun the complete preprocessing and search on its training set.
    """
    def __init__(self, optimization_method="grid", n_trials=20, cv=5, random_state=42, n_jobs=1):
        self.optimization_method = optimization_method
        self.n_trials = int(n_trials)
        self.cv = cv
        self.random_state = random_state
        self.n_jobs = n_jobs

    def _prepare_space(self, param_space, model):
        known = model.get_params(deep=True)
        def resolve(name):
            if name in known:
                return name
            matches = [k for k in known if k.endswith("__" + name)]
            if len(matches) != 1:
                raise ValueError(f"Parameter {name!r} is unknown or ambiguous; use its full pipeline name")
            return matches[0]
        self._param_aliases = {}
        def mapped(space):
            result = {}
            for key, value in space.items():
                resolved = resolve(key)
                self._param_aliases[resolved] = key
                result[resolved] = value
            return result
        return [mapped(v) for v in param_space] if isinstance(param_space, list) else mapped(param_space)

    def _clean_best_params(self, best_params):
        return {self._param_aliases.get(k, k): v for k, v in best_params.items() if k != "_branch"}

    def optimize(self, model, param_space, X, y, scoring="neg_mean_squared_error"):
        from sklearn.model_selection import check_cv
        from sklearn.base import is_classifier
        X, y = np.asarray(X, float), np.asarray(y)
        mask = np.isfinite(y) & np.isfinite(X).all(axis=1)
        if not np.all(mask):
            raise ValueError("Optimizer input must contain finite training pairs; mask before constructing CV folds")
        if is_classifier(model):
            return self._classifier_search(model, param_space, X, y, scoring)
        splitter = check_cv(self.cv, y=y, classifier=False)
        folds = list(splitter.split(X, y))
        space = self._prepare_space(param_space, model)
        if self.optimization_method == "grid":
            search = GridSearchCV(model, space, cv=folds, scoring=scoring,
                                  n_jobs=self.n_jobs, error_score="raise")
        elif self.optimization_method == "random":
            # Preserve conditional branches (e.g. distinct SVR kernels).
            search = RandomizedSearchCV(model, space, n_iter=self.n_trials, cv=folds,
                scoring=scoring, random_state=self.random_state, n_jobs=self.n_jobs,
                error_score="raise")
        elif self.optimization_method in {"bayesian", "optuna"}:
            return self._optuna_search(model, space, X, y, scoring, folds)
        else:
            raise ValueError(f"Unknown method: {self.optimization_method}")
        search.fit(X, y)
        return self._clean_best_params(search.best_params_)

    def _classifier_search(self, model, param_space, X, y, scoring):
        from sklearn.model_selection import ParameterGrid, ParameterSampler
        space = self._prepare_space(param_space, model)
        folds = _classification_folds(self.cv, X, y)
        def score(params):
            return _classification_score(clone(model).set_params(**params), X, y, folds, scoring)
        if self.optimization_method == "grid":
            candidates = ParameterGrid(space)
        elif self.optimization_method == "random":
            candidates = ParameterSampler(space, n_iter=self.n_trials, random_state=self.random_state)
        elif self.optimization_method in {"bayesian", "optuna"}:
            branches = space if isinstance(space, list) else [space]
            def objective(trial):
                branch = trial.suggest_int("_branch", 0, len(branches)-1) if len(branches)>1 else 0
                params = {}
                for name, values in branches[branch].items():
                    key = f"branch{branch}:{name}"
                    if isinstance(values, tuple) and len(values) in (2,3) and all(isinstance(v,(int,float)) for v in values[:2]):
                        lo, hi = values[:2]
                        log = len(values)==3 and values[2]=="log"
                        params[name] = (trial.suggest_int(key,lo,hi,log=log) if isinstance(lo,int) and isinstance(hi,int)
                                        else trial.suggest_float(key,lo,hi,log=log))
                    else:
                        params[name] = trial.suggest_categorical(key, list(values))
                trial.set_user_attr("model_params", params)
                return score(params)
            study = optuna.create_study(direction="maximize",
                sampler=optuna.samplers.TPESampler(seed=self.random_state))
            study.optimize(objective, n_trials=self.n_trials, show_progress_bar=False)
            return self._clean_best_params(study.best_trial.user_attrs["model_params"])
        else:
            raise ValueError(f"Unknown method: {self.optimization_method}")
        best, best_score = None, -np.inf
        for params in candidates:
            value = score(params)
            if value > best_score:
                best, best_score = params, value
        if best is None:
            raise ValueError("No finite classification hyperparameter score")
        return self._clean_best_params(best)

    def _optuna_search(self, model, space, X, y, scoring, folds):
        from sklearn.metrics import get_scorer
        scorer = get_scorer(scoring) if isinstance(scoring, str) else scoring
        branches = space if isinstance(space, list) else [space]
        def objective(trial):
            branch = trial.suggest_int("_branch", 0, len(branches)-1) if len(branches)>1 else 0
            params = {}
            for name, values in branches[branch].items():
                key = f"branch{branch}:{name}"
                if isinstance(values, tuple) and len(values) in (2, 3) and all(isinstance(v, (int, float)) for v in values[:2]):
                    lo, hi = values[:2]
                    logarithmic = len(values) == 3 and values[2] == "log"
                    if isinstance(lo, int) and isinstance(hi, int):
                        params[name] = trial.suggest_int(key, lo, hi, log=logarithmic)
                    else:
                        params[name] = trial.suggest_float(key, lo, hi, log=logarithmic)
                else:
                    params[name] = trial.suggest_categorical(key, list(values))
            trial.set_user_attr("model_params", params)
            scores = []
            for fold, (train, valid) in enumerate(folds):
                estimator = clone(model).set_params(**params)
                estimator.fit(X[train], y[train])
                score = float(scorer(estimator, X[valid], y[valid]))
                if not np.isfinite(score):
                    raise optuna.TrialPruned("Non-finite validation score")
                scores.append(score)
                trial.report(float(np.mean(scores)), fold)
                if trial.should_prune():
                    raise optuna.TrialPruned()
            return float(np.mean(scores))
        study = optuna.create_study(direction="maximize",
            sampler=optuna.samplers.TPESampler(seed=self.random_state),
            pruner=optuna.pruners.MedianPruner(n_startup_trials=5, n_warmup_steps=1))
        study.optimize(objective, n_trials=self.n_trials, show_progress_bar=False)
        return self._clean_best_params(study.best_trial.user_attrs["model_params"])


class WAS_LogisticRegression_Model:
    """
    Logistic Regression model for **tercile classification** (Below/Normal/Above = 0/1/2)
    on spatiotemporal seasonal climate data (typically rainfall or temperature anomalies).

    Workflow overview:
    1. **Tercile classification**: Compute spatial map of classes (0=Below, 1=Normal, 2=Above)
       based on climatological terciles (33rd and 67th percentiles).
    2. **Spatial grouping** (KMeans) on geographical coordinates, independent
       of validation target values.
    3. **Per-cluster hyperparameter optimization** of LogisticRegression using
       grid/random/Bayesian search.
    4. **Broadcast best hyperparameters** to every grid cell (Y, X).
    5. **Parallel per-grid-cell classification** using local hyperparameters.
    6. **Direct probabilistic output**: model.predict_proba() gives
       [P(Below), P(Normal), P(Above)] per grid cell and time step.

    Parameters
    ----------
    nb_cores : int, default=1
        Number of CPU cores for parallel processing (dask + joblib).
    dist_method : {'bestfit', 'nonparam'}, default='nonparam'
        Kept for API consistency with other models; unused in this class.
    n_clusters : int, default=5
        Number of target-independent geographical groups for shared tuning.
    param_grid : dict or None, default=None
        Hyperparameter search space for LogisticRegression.
        If None, a safe default grid is used (compatible with multinomial + lbfgs):
            C: [0.1, 0.5, 1.0, 2.0, 5.0]
            class_weight: [None, 'balanced']
            max_iter: [300, 600, 1000]
            solver: ['lbfgs']
    optimization_method : {'grid', 'random', 'bayesian'}, default='grid'
        Strategy for hyperparameter search.
    n_trials : int, default=20
        Number of trials for 'random' or 'bayesian' optimization.
    cv : int, default=5
        Number of cross-validation folds during tuning.
    random_state : int, default=42
        Random seed for reproducibility (KMeans, CV splits, optimizer).
    x_scaler : {None, 'standard', 'robust'}, default=None
        Feature scaling applied before the logistic estimator:
        - None       : no scaling
        - 'standard' : StandardScaler (zero mean, unit variance)
        - 'robust'   : RobustScaler (median-centred, IQR-scaled)

    Methods
    -------
    compute_class(Predictant, clim_year_start, clim_year_end)
        Compute tercile class map (T, Y, X) + climatological tercile thresholds.
    compute_hyperparameters(predictand, predictor, clim_year_start, clim_year_end)
        Cluster predictand → optimize hyperparameters per cluster →
        broadcast arrays of best C, class_weight (coded), max_iter, solver.
    fit_predict(x, y, x_test, C, cw_code, max_iter, solver)
        Fit logistic model on one grid cell, return [P(B), P(N), P(A)].
    compute_model(X_train, y_train, X_test, C_da, cw_code_da, maxiter_da, solver_da)
        Parallel classification across the full spatial domain.
        Returns xr.DataArray with dims (probability=['PB','PN','PA'], T, Y, X).
    forecast(Predictant, clim_year_start, clim_year_end, Predictor,
             Predictor_for_year, C_da, cw_code_da, maxiter_da, solver_da)
        End-to-end forecast for one target year.
    """

    def __init__(
        self,
        nb_cores=1,
        dist_method="nonparam",
        n_clusters=5,
        param_grid=None,
        optimization_method="grid",
        n_trials=20,
        cv=5,
        random_state=42,
        x_scaler=None,
    ):
        self.nb_cores        = int(nb_cores)
        self.dist_method     = dist_method
        self.n_clusters      = int(n_clusters)
        self.optimization_method = optimization_method
        self.n_trials        = int(n_trials)
        self.cv              = cv
        self.random_state    = int(random_state)
        self.x_scaler        = x_scaler

        if param_grid is None:
            self.param_grid = {
                "C":            [0.1, 0.5, 1.0, 2.0, 5.0],
                "class_weight": [None, "balanced"],
                "max_iter":     [300, 600, 1000],
                "solver":       ["lbfgs"],
            }
        else:
            self.param_grid = param_grid

        self.optimizer = BaseOptimizer(
            optimization_method=optimization_method,
            n_trials=n_trials,
            cv=cv,
            random_state=random_state,
        )

        # Numeric encoding for 'class_weight' so it can be broadcast as a
        # float DataArray (None → 0, 'balanced' → 1).
        self._cw_map = {None: 0, "balanced": 1}
        self._cw_inv = {0: None, 1: "balanced"}

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------
    def _safe_chunk_size(self, n: int) -> int:
        return max(int(np.ceil(n / max(self.nb_cores, 1))), 1)

    def _make_estimator(self, C=1.0, class_weight=None, max_iter=500, solver="lbfgs"):
        clf = LogisticRegression(
            C=float(C),
            class_weight=class_weight,
            max_iter=int(max_iter),
            solver=solver,
            # multi_class removed in sklearn 1.6; lbfgs is multinomial by default
            random_state=self.random_state,
        )
        if self.x_scaler is None:
            return clf
        if self.x_scaler == "standard":
            scaler = StandardScaler()
        elif self.x_scaler == "robust":
            scaler = RobustScaler()
        else:
            raise ValueError("x_scaler must be one of {None, 'standard', 'robust'}")
        return Pipeline([("scaler", scaler), ("logit", clf)])

    @staticmethod
    def _mode3_ignore_nan(v):
        """Mode over {0,1,2} for a 1-D array, ignoring NaN."""
        v = v[np.isfinite(v)]
        if v.size == 0:
            return np.nan
        counts = np.bincount(v.astype(int), minlength=3)
        return int(np.argmax(counts))

    # ------------------------------------------------------------------
    # 1) Tercile classification
    # ------------------------------------------------------------------
    @staticmethod
    def classify(y, index_start, index_end):
        y = np.asarray(y, dtype=float)
        reference = y[index_start:index_end]
        if np.isfinite(reference).sum() < 3:
            return np.full(y.shape, np.nan), np.nan, np.nan
        terciles = np.quantile(reference[np.isfinite(reference)], [1.0/3.0, 2.0/3.0])
        labels = np.digitize(y, terciles, right=True).astype(float)
        labels[~np.isfinite(y)] = np.nan
        return labels, float(terciles[0]), float(terciles[1])

    def compute_class(
        self,
        Predictant: xr.DataArray,
        clim_year_start: int,
        clim_year_end: int,
    ):
        """
        Compute tercile class map (T, Y, X) and climatological tercile thresholds.

        Returns
        -------
        y_class : xr.DataArray (T, Y, X)  — integer codes {0, 1, 2}
        terc33  : xr.DataArray (Y, X)     — 33rd-percentile threshold
        terc67  : xr.DataArray (Y, X)     — 67th-percentile threshold
        """
        index_start, index_end = _climatology_positions(Predictant["T"], clim_year_start, clim_year_end)

        y_class, terc33, terc67 = xr.apply_ufunc(
            self.classify,
            Predictant,
            input_core_dims=[("T",)],
            kwargs={"index_start": index_start, "index_end": index_end},
            vectorize=True,
            dask="parallelized",
            output_core_dims=[("T",), (), ()],
            output_dtypes=["float", "float", "float"],
        )
        return y_class.transpose("T", "Y", "X"), terc33, terc67

    # ------------------------------------------------------------------
    # 2) Spatial clustering and per-cluster HPO
    # ------------------------------------------------------------------
    def _build_cluster_map(self, predictand):
        return geographical_cluster_map(predictand, self.n_clusters,
                                   random_state=self.random_state)

    def compute_hyperparameters(
        self,
        predictand: xr.DataArray,
        predictor: xr.DataArray,
        clim_year_start: int,
        clim_year_end: int,
        scoring: str = "neg_log_loss",
    ):
        """
        Cluster predictand spatially → optimise LogisticRegression hyperparameters
        per cluster → broadcast best params to every (Y, X) cell.

        Returns
        -------
        C_da       : xr.DataArray (Y, X) — best C per cell
        cw_code_da : xr.DataArray (Y, X) — class_weight code (0=None, 1='balanced')
        maxiter_da : xr.DataArray (Y, X) — best max_iter per cell
        solver_da  : xr.DataArray (Y, X) — best solver per cell
        cluster_da : xr.DataArray (Y, X) — KMeans cluster labels
        """
        predictor = predictor.copy()
        predictor, predictand = _aligned_training(predictor, predictand)
        if any(dim in predictor.dims for dim in ("Y", "X")):
            raise ValueError("Classifier cluster tuning requires T,features predictors; aggregate spatial predictors explicitly")

        y_class, _, _ = self.compute_class(predictand, clim_year_start, clim_year_end)

        cluster_da = self._build_cluster_map(predictand)
        _, cluster_da = xr.align(predictand.isel(T=0), cluster_da, join="outer")

        clusters = np.unique(cluster_da.values)
        clusters = clusters[np.isfinite(clusters)]

        best_params_for_cluster = {}
        for c in clusters:
            mask_c = cluster_da == c

            y_stack = y_class.where(mask_c).stack(Z=("Y", "X"))
            y_mode  = xr.apply_ufunc(
                self._mode3_ignore_nan,
                y_stack,
                input_core_dims=[("Z",)],
                output_core_dims=[()],
                vectorize=True,
                dask="parallelized",
                output_dtypes=["float"],
            ).dropna("T")

            if y_mode.sizes.get("T", 0) == 0:
                continue

            X_c   = predictor.sel(T=y_mode["T"])
            X_mat = X_c.values
            y_vec = y_mode.values.astype(int)

            pairs = np.isfinite(X_mat).all(axis=1) & np.isfinite(y_vec)
            if pairs.sum() < 2:
                continue
            X_mat, y_vec = X_mat[pairs], y_vec[pairs]
            base_est = self._make_estimator()
            bp = {} if np.unique(y_vec).size == 1 else self.optimizer.optimize(
                base_est, self.param_grid, X_mat, y_vec, scoring=scoring
            )
            best_params_for_cluster[int(c)] = bp

        C_da       = xr.full_like(cluster_da, np.nan, dtype=float)
        cw_code_da = xr.full_like(cluster_da, np.nan, dtype=float)
        maxiter_da = xr.full_like(cluster_da, np.nan, dtype=float)
        solver_da  = xr.full_like(cluster_da, np.nan, dtype=object)

        for c, bp in best_params_for_cluster.items():
            c_mask = cluster_da == c
            C_da       = C_da.where(~c_mask,       other=float(bp.get("C", 1.0)))
            cw_code_da = cw_code_da.where(~c_mask, other=float(self._cw_map.get(bp.get("class_weight"), 0)))
            maxiter_da = maxiter_da.where(~c_mask,  other=float(bp.get("max_iter", 500)))
            solver_da  = solver_da.where(~c_mask,   other=str(bp.get("solver", "lbfgs")))

        _report_tuned("WAS_LogisticRegression_Model", getattr(self, "verbose", True), cluster_da=cluster_da, C=C_da, class_weight=cw_code_da, max_iter=maxiter_da, solver=solver_da)

        return C_da, cw_code_da, maxiter_da, solver_da, cluster_da

    # ------------------------------------------------------------------
    # 3) Per-pixel fit + predict_proba
    # ------------------------------------------------------------------
    def fit_predict(self, x, y, x_test, C, cw_code, max_iter, solver):
        if not np.isfinite(C) or not np.isfinite(cw_code) or not np.isfinite(max_iter):
            return np.full(3, np.nan)
        estimator = self._make_estimator(C=C, class_weight=self._cw_inv.get(int(cw_code)),
                                         max_iter=max_iter, solver=str(solver))
        return _classifier_probabilities(estimator, x, y, x_test)[0]

    # ------------------------------------------------------------------
    # 4) Parallel model over grid with local params
    # ------------------------------------------------------------------
    def compute_model(self, X_train, y_train, X_test, C_da, cw_code_da, maxiter_da, solver_da):
        """Return PB/PN/PA at every supplied test date, retaining singleton axes."""
        return _spatial_classifier(self, X_train, y_train, X_test,
            (C_da, cw_code_da, maxiter_da, solver_da.astype(str)))

    # ------------------------------------------------------------------
    # 5) End-to-end forecast for one target year
    # ------------------------------------------------------------------
    def forecast(self, Predictant, clim_year_start, clim_year_end, Predictor,
                 Predictor_for_year, C_da, cw_code_da, maxiter_da, solver_da):
        """Fit historical classes and predict each explicitly dated future row."""
        y_class, _, _ = self.compute_class(Predictant, clim_year_start, clim_year_end)
        return self.compute_model(Predictor, y_class, _forecast_times(Predictor_for_year),
                                  C_da, cw_code_da, maxiter_da, solver_da)

def _compute_forecast_prob_nonparam(fc_sq_np, error_samples_np, T1_np, T2_np):
    """
    Compute tercile probabilities for a single forecast step (non-parametric).
    Pure numpy — avoids xarray T-alignment clash when forecast has T=1
    and error_samples has T=n_years.
    """
    nY, nX = fc_sq_np.shape
    out = np.full((3, 1, nY, nX), np.nan, dtype=float)
    for iy in range(nY):
        for ix in range(nX):
            bg = fc_sq_np[iy, ix]
            if not np.isfinite(bg):
                continue
            es = error_samples_np[:, iy, ix]
            d  = bg + es; d = d[np.isfinite(d)]
            if len(d) == 0:
                continue
            t1 = T1_np[iy, ix]; t2 = T2_np[iy, ix]
            pb = float(np.mean(d <= t1))
            pn = float(np.mean((d > t1) & (d <= t2)))
            out[0, 0, iy, ix] = pb
            out[1, 0, iy, ix] = pn
            out[2, 0, iy, ix] = 1.0 - (pb + pn)
    return out


class WAS_PolynomialRegression:
    """
    Polynomial Regression model for spatiotemporal climate prediction.

    Implements polynomial feature expansion (PolynomialFeatures + LinearRegression)
    to capture non-linear relationships between predictors and a continuous predictand.

    Parameters
    ----------
    nb_cores : int, default=1
        Number of CPU cores for parallel Dask computation.
    degree : int, default=2
        Degree of the polynomial expansion (2 = quadratic, 3 = cubic).
    dist_method : {'bestfit', 'nonparam'}, default='nonparam'
        Tercile probability method.

    Warnings
    --------
    - degree ≥ 4 with many features causes numerical instability; prefer degree 2–3.
    - Predictor scaling is fitted on the training observations inside the estimator.
    - Signed anomalies are preserved; physical support is applied after inverse scaling.
    """

    def __init__(self, nb_cores=1, degree=2, dist_method="nonparam"):
        self.nb_cores    = nb_cores
        self.degree      = degree
        self.dist_method = dist_method

    # ------------------------------------------------------------------
    # 1) Per-pixel fit + predict
    # ------------------------------------------------------------------
    def fit_predict(self, x, y, x_test, y_test):
        estimator = Pipeline([("scale", StandardScaler()),
                              ("polynomial", PolynomialFeatures(degree=self.degree, include_bias=False)),
                              ("regression", LinearRegression())])
        return _regression_pair(estimator, x, y, x_test, y_test)

    # ------------------------------------------------------------------
    # 2) Parallel spatial hindcast
    # ------------------------------------------------------------------
    def compute_model(self, X_train, y_train, X_test, y_test=None):
        return _spatial_regression(self, X_train, y_train, X_test, y_test)

    # ------------------------------------------------------------------
    # 3) Probability calculation methods
    # ------------------------------------------------------------------
    @staticmethod
    def _ppf_terciles_from_code(dist_code, shape, loc, scale):
        """Tercile thresholds (T1, T2) from best-fit distribution parameters."""
        if np.isnan(dist_code):
            return np.nan, np.nan
        code = int(dist_code)
        try:
            if code == 1:
                return norm.ppf(1.0 / 3.0, loc=loc, scale=scale),      norm.ppf(2.0 / 3.0, loc=loc, scale=scale)
            elif code == 2:
                return lognorm.ppf(1.0 / 3.0, s=shape, loc=loc, scale=scale), lognorm.ppf(2.0 / 3.0, s=shape, loc=loc, scale=scale)
            elif code == 3:
                return expon.ppf(1.0 / 3.0, loc=loc, scale=scale),     expon.ppf(2.0 / 3.0, loc=loc, scale=scale)
            elif code == 4:
                return gamma.ppf(1.0 / 3.0, a=shape, loc=loc, scale=scale), gamma.ppf(2.0 / 3.0, a=shape, loc=loc, scale=scale)
            elif code == 5:
                return weibull_min.ppf(1.0 / 3.0, c=shape, loc=loc, scale=scale), weibull_min.ppf(2.0 / 3.0, c=shape, loc=loc, scale=scale)
            elif code == 6:
                return t.ppf(1.0 / 3.0, df=shape, loc=loc, scale=scale), t.ppf(2.0 / 3.0, df=shape, loc=loc, scale=scale)
            elif code == 7:
                return poisson.ppf(1.0 / 3.0, mu=shape, loc=loc),      poisson.ppf(2.0 / 3.0, mu=shape, loc=loc)
            elif code == 8:
                return nbinom.ppf(1.0 / 3.0, n=shape, p=scale, loc=loc), nbinom.ppf(2.0 / 3.0, n=shape, p=scale, loc=loc)
        except Exception:
            return np.nan, np.nan
        return np.nan, np.nan

    @staticmethod
    def weibull_shape_solver(k, M, V):
        """Root function for the Weibull shape parameter k given mean M and variance V."""
        if k <= 0:
            return -np.inf
        try:
            g1 = gamma_function(1 + 1 / k)
            g2 = gamma_function(1 + 2 / k)
            return V / (M ** 2) - ((g2 / (g1 ** 2)) - 1)
        except ValueError:
            return -np.inf

    @staticmethod
    def calculate_tercile_probabilities_bestfit(best_guess, error_variance, T1, T2, dist_code, dof, nonneg=False):
        """Shared predictive CDF; moment matching and support follow one contract."""
        from .was_linear_models import _regression_tercile_probabilities
        return _regression_tercile_probabilities(best_guess, error_variance, T1, T2,
                                               dist_code, dof, nonneg=nonneg)

    @staticmethod
    def calculate_tercile_probabilities_nonparametric(best_guess, error_samples, first_tercile, second_tercile, nonneg=False):
        """Empirical CDF with finite errors, declared support and fixed events."""
        return _residual_probabilities(best_guess, error_samples, first_tercile, second_tercile, nonneg=nonneg)

    # ------------------------------------------------------------------
    # 4) compute_prob
    # ------------------------------------------------------------------
    def compute_prob(self, Predictant, clim_year_start, clim_year_end, hindcast_det,
                     best_code_da=None, best_shape_da=None, best_loc_da=None,
                     best_scale_da=None, **probability_options):
        """Calibrate hindcasts with outer reuse by default; inner needs explicit residuals.

        The held-out year's error is excluded. Use ``diagnostic=True`` explicitly
        for a full-hindcast residual pool, which is unsuitable for skill scores.
        """
        from .was_probability_calibration import direct_hindcast_probabilities
        probability_options.setdefault("tercile_reference", getattr(self, "tercile_reference", "climatology"))
        return direct_hindcast_probabilities(self, Predictant, clim_year_start,
            clim_year_end, hindcast_det, best_code_da, best_shape_da,
            best_loc_da, best_scale_da, **probability_options)

    # ------------------------------------------------------------------
    # 5) forecast
    # ------------------------------------------------------------------
    def forecast(self, Predictant, clim_year_start, clim_year_end, Predictor,
                 hindcast_det, Predictor_for_year, best_code_da=None,
                 best_shape_da=None, best_loc_da=None, best_scale_da=None):
        return _operational_forecast(self, Predictant, clim_year_start, clim_year_end,
            Predictor, hindcast_det, Predictor_for_year, parameters=(),
            best_code_da=best_code_da, best_shape_da=best_shape_da,
            best_loc_da=best_loc_da, best_scale_da=best_scale_da, raw_target=False)

        
###########################################

class WAS_PoissonRegression:
    """
    Poisson Regression model for spatiotemporal count-data prediction
    (e.g. number of rainy days, dry spells, extreme event counts).

    Parameters
    ----------
    nb_cores : int, default=1
        Number of CPU cores for parallel Dask computation.
    dist_method : {'bestfit', 'nonparam'}, default='nonparam'
        Tercile probability method:
        - 'bestfit'  → parametric best-fit family per grid cell
        - 'nonparam' → empirical error-sample resampling

    Methods
    -------
    fit_predict(x, y, x_test, y_test)
        Fit Poisson on one grid cell, return [error, prediction].
    compute_model(X_train, y_train, X_test, y_test)
        Parallel Poisson across the full spatial domain.
        Returns xr.DataArray (Y, X) — deterministic prediction.
    compute_prob(Predictant, clim_year_start, clim_year_end, hindcast_det, ...)
        Tercile probabilities for a deterministic hindcast.
    forecast(Predictant, clim_year_start, clim_year_end,
             Predictor, hindcast_det, Predictor_for_year, ...)
        End-to-end single-year forecast (deterministic + tercile proba).
    """

    def __init__(self, nb_cores=1, dist_method="nonparam"):
        self.nb_cores    = nb_cores
        self.dist_method = dist_method

    # ------------------------------------------------------------------
    # 1) Per-pixel fit + predict
    # ------------------------------------------------------------------
    def fit_predict(self, x, y, x_test, y_test=None):
        """Poisson log-link mean; nonnegative real y permits quasi-likelihood use."""
        from sklearn.linear_model import PoissonRegressor
        y = np.asarray(y, float)
        if np.any(y[np.isfinite(y)] < 0):
            raise ValueError("Poisson requires nonnegative targets in physical units")
        valid = np.isfinite(y)
        if valid.any() and np.all(y[valid] == 0):
            prediction = 0.0 if np.isfinite(x_test).all() else np.nan
            return np.array([np.nan if y_test is None else y_test-prediction, prediction])
        estimator = Pipeline([("scale", StandardScaler()),
                              ("poisson", PoissonRegressor(alpha=1.0, max_iter=5000))])
        return _regression_pair(estimator, x, y, x_test, y_test)

    # ------------------------------------------------------------------
    # 2) Parallel spatial hindcast
    # ------------------------------------------------------------------
    def compute_model(self, X_train, y_train, X_test, y_test=None):
        return _spatial_regression(self, X_train, y_train, X_test, y_test)

    # ------------------------------------------------------------------
    # 3) Probability methods
    # ------------------------------------------------------------------
    @staticmethod
    def _ppf_terciles_from_code(dist_code, shape, loc, scale):
        """Tercile thresholds (T1, T2) from best-fit distribution parameters."""
        if np.isnan(dist_code):
            return np.nan, np.nan
        code = int(dist_code)
        try:
            if code == 1:
                return norm.ppf(1.0 / 3.0, loc=loc, scale=scale), norm.ppf(2.0 / 3.0, loc=loc, scale=scale)
            elif code == 2:
                return lognorm.ppf(1.0 / 3.0, s=shape, loc=loc, scale=scale), lognorm.ppf(2.0 / 3.0, s=shape, loc=loc, scale=scale)
            elif code == 3:
                return expon.ppf(1.0 / 3.0, loc=loc, scale=scale), expon.ppf(2.0 / 3.0, loc=loc, scale=scale)
            elif code == 4:
                return gamma.ppf(1.0 / 3.0, a=shape, loc=loc, scale=scale), gamma.ppf(2.0 / 3.0, a=shape, loc=loc, scale=scale)
            elif code == 5:
                return weibull_min.ppf(1.0 / 3.0, c=shape, loc=loc, scale=scale), weibull_min.ppf(2.0 / 3.0, c=shape, loc=loc, scale=scale)
            elif code == 6:
                return t.ppf(1.0 / 3.0, df=shape, loc=loc, scale=scale), t.ppf(2.0 / 3.0, df=shape, loc=loc, scale=scale)
            elif code == 7:
                return poisson.ppf(1.0 / 3.0, mu=shape, loc=loc), poisson.ppf(2.0 / 3.0, mu=shape, loc=loc)
            elif code == 8:
                return nbinom.ppf(1.0 / 3.0, n=shape, p=scale, loc=loc), nbinom.ppf(2.0 / 3.0, n=shape, p=scale, loc=loc)
        except Exception:
            return np.nan, np.nan
        return np.nan, np.nan

    @staticmethod
    def weibull_shape_solver(k, M, V):
        """Root function for Weibull shape parameter k given mean M and variance V."""
        if k <= 0:
            return -np.inf
        try:
            g1 = gamma_function(1 + 1 / k)
            g2 = gamma_function(1 + 2 / k)
            return V / (M ** 2) - ((g2 / (g1 ** 2)) - 1)
        except ValueError:
            return -np.inf

    @staticmethod
    def calculate_tercile_probabilities_bestfit(best_guess, error_variance, T1, T2, dist_code, dof, nonneg=False):
        """Shared predictive CDF; moment matching and support follow one contract."""
        from .was_linear_models import _regression_tercile_probabilities
        return _regression_tercile_probabilities(best_guess, error_variance, T1, T2,
                                               dist_code, dof, nonneg=nonneg)

    @staticmethod
    def calculate_tercile_probabilities_nonparametric(best_guess, error_samples, first_tercile, second_tercile, nonneg=False):
        """Empirical CDF with finite errors, declared support and fixed events."""
        return _residual_probabilities(best_guess, error_samples, first_tercile, second_tercile, nonneg=nonneg)

    # ------------------------------------------------------------------
    # 4) compute_prob
    # ------------------------------------------------------------------
    def compute_prob(self, Predictant, clim_year_start, clim_year_end, hindcast_det,
                     best_code_da=None, best_shape_da=None, best_loc_da=None,
                     best_scale_da=None, **probability_options):
        """Calibrate hindcasts with outer reuse by default; inner needs explicit residuals.

        The held-out year's error is excluded. Use ``diagnostic=True`` explicitly
        for a full-hindcast residual pool, which is unsuitable for skill scores.
        """
        from .was_probability_calibration import direct_hindcast_probabilities
        probability_options.setdefault("tercile_reference", getattr(self, "tercile_reference", "climatology"))
        return direct_hindcast_probabilities(self, Predictant, clim_year_start,
            clim_year_end, hindcast_det, best_code_da, best_shape_da,
            best_loc_da, best_scale_da, **probability_options)

    # ------------------------------------------------------------------
    # 5) forecast
    # ------------------------------------------------------------------
    def forecast(self, Predictant, clim_year_start, clim_year_end, Predictor,
                 hindcast_det, Predictor_for_year, best_code_da=None,
                 best_shape_da=None, best_loc_da=None, best_scale_da=None):
        return _operational_forecast(self, Predictant, clim_year_start, clim_year_end,
            Predictor, hindcast_det, Predictor_for_year, parameters=(),
            best_code_da=best_code_da, best_shape_da=best_shape_da,
            best_loc_da=best_loc_da, best_scale_da=best_scale_da, raw_target=True)


class MARS:
    """Multivariate Adaptive Regression Splines with corrected forward/backward passes."""
    
    def __init__(self, max_terms: int = 21, max_degree: int = 2, 
                 penalty: float = 3.0, min_span: int = 5):
        """
        Parameters:
        -----------
        max_terms : int
            Maximum number of basis functions (including intercept)
        max_degree : int
            Maximum interaction degree (1 = additive, 2 = pairwise interactions)
        penalty : float
            GCV penalty per knot (typically 2-4)
        min_span : int
            Minimum observations between knots
        """
        self.max_terms = max_terms
        self.max_degree = max_degree
        self.penalty = penalty
        self.min_span = min_span
        
        self.basis_functions = []  # List of basis functions
        self.coef_ = None          # Coefficients
        self.knots_ = []           # Knot positions for each variable
        self.dof_ = 0              # Degrees of freedom
        
    def _hinge(self, x: np.ndarray, knot: float, side: int) -> np.ndarray:
        """Hinge function: max(0, x - t) or max(0, t - x)."""
        if side == 1:  # Right hinge
            return np.maximum(0, x - knot)
        else:  # Left hinge (-1)
            return np.maximum(0, knot - x)
    
    def _evaluate_basis(self, X: np.ndarray, basis_idx: int) -> np.ndarray:
        """Evaluate a specific basis function."""
        if basis_idx == 0:  # Intercept
            return np.ones(X.shape[0])
        
        basis = self.basis_functions[basis_idx]
        result = np.ones(X.shape[0])
        for var_idx, knot, side in basis:
            result *= self._hinge(X[:, var_idx], knot, side)
        return result
    
    def _create_design_matrix(self, X: np.ndarray) -> np.ndarray:
        """Create design matrix from current basis functions."""
        n_samples = X.shape[0]
        n_basis = len(self.basis_functions)
        B = np.ones((n_samples, n_basis))
        
        for j in range(1, n_basis):  # Skip intercept (j=0)
            B[:, j] = self._evaluate_basis(X, j)
        
        return B
    
    def _find_knot_candidates(self, X: np.ndarray, y: np.ndarray) -> List[np.ndarray]:
        """Find candidate knot positions for each variable."""
        n_samples = X.shape[0]
        candidates = []
        
        for v in range(X.shape[1]):
            # Sort unique values of variable v
            x_sorted = np.sort(X[:, v])
            
            # Create candidate knots at percentiles (more robust than unique values)
            percentiles = np.linspace(10, 90, 20)  # 20 equally spaced percentiles
            knots = np.percentile(x_sorted, percentiles)
            
            # Remove knots too close to edges
            valid_knots = []
            for t in np.unique(knots):
                left_count = np.sum(X[:, v] <= t)
                right_count = n_samples - left_count
                if left_count >= self.min_span and right_count >= self.min_span:
                    valid_knots.append(t)
            
            candidates.append(np.array(valid_knots))
        
        return candidates
    
    def _gcv_score(self, X, y, B, beta, bases=None):
        """GCV = mean squared residual / (1 - effective_dof/n)**2.

        Evaluate the candidate bases, never the previous model's complexity.
        This implementation uses rank(B) + penalty * number of unique knots.
        """
        bases = self.basis_functions if bases is None else bases
        knots = {(v, knot) for basis in bases[1:] for v, knot, _ in basis}
        effective_dof = np.linalg.matrix_rank(B) + self.penalty * len(knots)
        n = len(y)
        if effective_dof >= n:
            return np.inf
        return float(np.mean((y - B @ beta)**2) / (1.0 - effective_dof / n)**2)
    
    def forward_pass(self, X, y):
        """Add reflected hinge pairs using training RSS (Friedman forward stage)."""
        candidates = self._find_knot_candidates(X, y)
        self.basis_functions = [[]]
        B = np.ones((len(y), 1))
        beta = np.array([np.mean(y)])
        rss = float(np.sum((y - B @ beta)**2))
        while len(self.basis_functions) + 2 <= self.max_terms:
            best = None
            best_rss = rss
            for basis in self.basis_functions:
                if len(basis) >= self.max_degree:
                    continue
                used = {v for v, _, _ in basis}
                parent = np.ones(len(y))
                for v, knot, side in basis:
                    parent *= self._hinge(X[:, v], knot, side)
                for v in range(X.shape[1]):
                    if v in used:
                        continue
                    for knot in candidates[v]:
                        pair = [basis + [(v, knot, 1)], basis + [(v, knot, -1)]]
                        candidate = np.column_stack([B, parent*self._hinge(X[:, v], knot, 1),
                                                        parent*self._hinge(X[:, v], knot, -1)])
                        coef, _, rank, _ = np.linalg.lstsq(candidate, y, rcond=None)
                        if rank <= np.linalg.matrix_rank(B):
                            continue
                        candidate_rss = float(np.sum((y-candidate @ coef)**2))
                        if candidate_rss < best_rss - 1e-12*max(1.0, rss):
                            best_rss = candidate_rss
                            best = pair, candidate, coef
            if best is None:
                break
            pair, B, beta = best
            self.basis_functions.extend(pair)
            rss = best_rss
        return B, beta
    
    def backward_pass(self, X, y, B, beta):
        """Explore the complete deletion path, retaining the minimum-GCV model."""
        bases = list(self.basis_functions)
        best_bases, best_coef = list(bases), beta.copy()
        best_gcv = self._gcv_score(X, y, B, beta, bases)
        while len(bases) > 1:
            candidate_best = None
            rss_best = np.inf
            for remove in range(1, len(bases)):
                reduced = np.delete(B, remove, axis=1)
                coef, _, _, _ = np.linalg.lstsq(reduced, y, rcond=None)
                rss = float(np.sum((y - reduced @ coef)**2))
                if rss < rss_best:
                    rss_best = rss
                    candidate_best = remove, reduced, coef
            remove, B, beta = candidate_best
            bases = bases[:remove] + bases[remove+1:]
            score = self._gcv_score(X, y, B, beta, bases)
            if score <= best_gcv:
                best_gcv, best_bases, best_coef = score, list(bases), beta.copy()
        self.basis_functions = best_bases
        self.coef_ = best_coef
        self.gcv_ = best_gcv
        self.dof_ = len(best_bases) + self.penalty * self._count_knots()
    
    def _count_knots(self) -> int:
        """Count unique knots in the model."""
        unique_knots = set()
        for basis in self.basis_functions[1:]:
            for var_idx, knot, _ in basis:
                unique_knots.add((var_idx, knot))
        return len(unique_knots)
    
    def fit(self, X, y):
        X, y = np.asarray(X, float), np.asarray(y, float).reshape(-1)
        if X.ndim != 2 or X.shape[0] != y.size:
            raise ValueError("X must be 2D with the same number of rows as y")
        if y.size < 2 or not np.isfinite(X).all() or not np.isfinite(y).all():
            raise ValueError("MARS requires at least two finite training pairs")
        if self.max_terms < 1 or self.max_degree < 1 or self.penalty < 0:
            raise ValueError("Invalid MARS complexity controls")
        B, beta = self.forward_pass(X, y)
        self.backward_pass(X, y, B, beta)
        # backward_pass owns the retained coefficients; never restore forward beta.
        return self
    
    def predict(self, X: np.ndarray) -> np.ndarray:
        """Predict using fitted MARS model."""
        if self.coef_ is None:
            raise ValueError("Model must be fitted before prediction")
        
        X = np.asarray(X)
        n_samples = X.shape[0]
        n_basis = len(self.basis_functions)
        
        # Create design matrix
        B = np.ones((n_samples, n_basis))
        for j in range(1, n_basis):
            col = np.ones(n_samples)
            for var_idx, knot, side in self.basis_functions[j]:
                col *= self._hinge(X[:, var_idx], knot, side)
            B[:, j] = col
        
        return B @ self.coef_
    
    def get_formula(self, feature_names: Optional[List[str]] = None) -> str:
        """Get human-readable formula for the model."""
        if feature_names is None:
            feature_names = [f"X{i}" for i in range(100)]  # Placeholder
        
        terms = []
        for j, (coef, basis) in enumerate(zip(self.coef_, self.basis_functions)):
            if j == 0:  # Intercept
                terms.append(f"{coef:.4f}")
            elif basis:
                basis_str = " * ".join(
                    f"max(0, {feature_names[v]} {'-' if s==1 else '+'} {t:.3f})"
                    for v, t, s in basis
                )
                terms.append(f"{coef:.4f} * {basis_str}")
        
        return " + ".join(terms)

class WAS_MARS_Model:
    """
    Multivariate Adaptive Regression Splines (MARS) model for spatiotemporal climate prediction.

    MARS is a non-parametric regression technique that builds flexible models by fitting piecewise
    linear or cubic basis functions with knots automatically determined via a forward-backward
    selection process, regularized by Generalized Cross-Validation (GCV).

    This class implements:
    - MARS regression using the pyearth library (or compatible MARS implementation)
    - Spatially parallel fitting and prediction across large grids using dask + xarray
    - Deterministic point predictions with optional error computation
    - Probabilistic tercile forecasting (Below/Normal/Above = PB/PN/PA) using either:
      - Parametric best-fit distributions per grid cell ('bestfit')
      - Non-parametric sampling of historical forecast errors ('nonparam')

    Ideal for modeling non-linear relationships in seasonal climate variables (rainfall, temperature,
    agro-climatic indices) with good interpretability through selected basis functions.

    Parameters
    ----------
    nb_cores : int, default=1
        Number of CPU cores for parallel processing (dask workers).

    dist_method : {'bestfit', 'nonparam'}, default='nonparam'
        Method for computing tercile probabilities:
        - 'bestfit'  → uses best-fit distribution per grid cell (requires distribution fit inputs)
        - 'nonparam' → empirical sampling of historical forecast errors

    max_terms : int, default=21
        Maximum number of basis functions (terms) allowed in the MARS model.

    max_degree : int, default=2
        Maximum degree of interaction (number of variables in a single hinge function).

    c : float, default=3.0
        Penalty cost parameter for effective number of parameters in GCV score
        (higher values → stronger regularization, fewer terms).

    Attributes
    ----------
    nb_cores, dist_method, max_terms, max_degree, c
        Stored initialization parameters.

    Methods
    -------
    fit_predict(x, y, x_test, y_test=None)
        Fits MARS on one grid cell and returns [error, prediction] or just prediction.

    compute_model(X_train, y_train, X_test, y_test)
        Parallel MARS regression across the entire spatial domain.
        Returns xarray.DataArray with dims ('output'=['error','prediction'], Y, X).

    compute_prob(Predictant, clim_year_start, clim_year_end, hindcast_det,
                 best_code_da=None, best_shape_da=None, best_loc_da=None, best_scale_da=None)
        Computes tercile probabilities for deterministic hindcasts.

    forecast(Predictant, clim_year_start, clim_year_end, Predictor, hindcast_det,
             Predictor_for_year, best_code_da=None, best_shape_da=None,
             best_loc_da=None, best_scale_da=None)
        Full end-to-end forecast pipeline for one target year:
        - deterministic MARS prediction
        - tercile probabilities (PB, PN, PA)

    Notes
    -----
    - **Input data requirements**:
      - Target (y) should be continuous (rainfall, temperature, etc.).
      - Predictors (X) can be continuous or categorical (MARS handles both).
    - Negative predictions are clipped to zero (useful for rainfall or non-negative variables).
    - MARS models are interpretable: final model consists of selected hinge functions.
    - Large spatial domains benefit significantly from higher `nb_cores`.
    - For `dist_method='bestfit'`, distribution fit results from `WAS_TransformData` must be provided.
    - No explicit feature or target scaling is applied (MARS is invariant to monotonic transformations).

    Warnings
    --------
    - Very small training sets per grid cell may lead to overfitting or degenerate models.
    - Extremely skewed targets (e.g., heavy-tailed rainfall) may benefit from transformation
      before modeling (not done automatically).
    - MARS can produce piecewise-linear behavior; for very smooth functions, consider higher
      max_degree or compare with other models (e.g., XGBoost, MLP).
    """

    def __init__(self, nb_cores=1, dist_method="nonparam", max_terms=21, max_degree=2, c=3):
        """
        Initializes the WAS_MARS_Model with specified parameters.
        
        Parameters
        ----------
        nb_cores : int, optional
            Number of CPU cores to use for parallel computation, by default 1.
        dist_method : str, optional
            Distribution method to compute tercile probabilities, by default "gamma".
        max_terms : int, optional
            Maximum number of basis functions for MARS, by default 21.
        max_degree : int, optional
            Maximum degree of interaction for MARS, by default 1.
        c : float, optional
            Cost parameter for GCV in MARS, by default 3.
        """
        self.nb_cores = nb_cores
        self.dist_method = dist_method
        self.max_terms = max_terms
        self.max_degree = max_degree
        self.c = c
    
    def fit_predict(self, x, y, x_test, y_test=None):
        estimator = MARS(max_terms=self.max_terms, max_degree=self.max_degree, penalty=self.c)
        return _regression_pair(estimator, x, y, x_test, y_test)

    def compute_model(self, X_train, y_train, X_test, y_test=None):
        return _spatial_regression(self, X_train, y_train, X_test, y_test)
    
    # ------------------ Probability Calculation Methods ------------------

    @staticmethod
    def _ppf_terciles_from_code(dist_code, shape, loc, scale):
        """
        Return tercile thresholds (T1, T2) from best-fit distribution parameters.
    
        dist_code:
            1: norm
            2: lognorm
            3: expon
            4: gamma
            5: weibull_min
            6: t
            7: poisson
            8: nbinom
        """
        if np.isnan(dist_code):
            return np.nan, np.nan
    
        code = int(dist_code)
        try:
            if code == 1:
                return (
                    norm.ppf(1.0 / 3.0, loc=loc, scale=scale),
                    norm.ppf(2.0 / 3.0, loc=loc, scale=scale),
                )
            elif code == 2:
                return (
                    lognorm.ppf(1.0 / 3.0, s=shape, loc=loc, scale=scale),
                    lognorm.ppf(2.0 / 3.0, s=shape, loc=loc, scale=scale),
                )
            elif code == 3:
                return (
                    expon.ppf(1.0 / 3.0, loc=loc, scale=scale),
                    expon.ppf(2.0 / 3.0, loc=loc, scale=scale),
                )
            elif code == 4:
                return (
                    gamma.ppf(1.0 / 3.0, a=shape, loc=loc, scale=scale),
                    gamma.ppf(2.0 / 3.0, a=shape, loc=loc, scale=scale),
                )
            elif code == 5:
                return (
                    weibull_min.ppf(1.0 / 3.0, c=shape, loc=loc, scale=scale),
                    weibull_min.ppf(2.0 / 3.0, c=shape, loc=loc, scale=scale),
                )
            elif code == 6:
                # Note: Renamed 't_dist' to 't' for standard scipy.stats
                return (
                    t.ppf(1.0 / 3.0, df=shape, loc=loc, scale=scale),
                    t.ppf(2.0 / 3.0, df=shape, loc=loc, scale=scale),
                )
            elif code == 7:
                # Poisson: poisson.ppf(q, mu, loc=0)
                # ASSUMPTION: 'mu' (mean) is passed as 'shape'
                #             'loc' is passed as 'loc'
                #             'scale' is unused
                return (
                    poisson.ppf(1.0 / 3.0, mu=shape, loc=loc),
                    poisson.ppf(2.0 / 3.0, mu=shape, loc=loc),
                )
            elif code == 8:
                # Negative Binomial: nbinom.ppf(q, n, p, loc=0)
                # ASSUMPTION: 'n' (successes) is passed as 'shape'
                #             'p' (probability) is passed as 'scale'
                #             'loc' is passed as 'loc'
                return (
                    nbinom.ppf(1.0 / 3.0, n=shape, p=scale, loc=loc),
                    nbinom.ppf(2.0 / 3.0, n=shape, p=scale, loc=loc),
                )
        except Exception:
            return np.nan, np.nan
    
        # Fallback if code is not 1-8
        return np.nan, np.nan
        
    @staticmethod
    def weibull_shape_solver(k, M, V):
        """
        Function to find the root of the Weibull shape parameter 'k'.
        We find 'k' such that the theoretical variance/mean^2 ratio
        matches the observed V/M^2 ratio.
        """
        # Guard against invalid 'k' values during solving
        if k <= 0:
            return -np.inf
        try:
            g1 = gamma_function(1 + 1/k)
            g2 = gamma_function(1 + 2/k)
            
            # This is the V/M^2 ratio *implied by k*
            implied_v_over_m_sq = (g2 / (g1**2)) - 1
            
            # This is the *observed* ratio
            observed_v_over_m_sq = V / (M**2)
            
            # Return the difference (we want this to be 0)
            return observed_v_over_m_sq - implied_v_over_m_sq
        except ValueError:
            return -np.inf # Handle math errors

    @staticmethod
    def calculate_tercile_probabilities_bestfit(best_guess, error_variance, T1, T2, dist_code, dof, nonneg=False):
        """Shared predictive CDF; moment matching and support follow one contract."""
        from .was_linear_models import _regression_tercile_probabilities
        return _regression_tercile_probabilities(best_guess, error_variance, T1, T2,
                                               dist_code, dof, nonneg=nonneg)

    @staticmethod
    def calculate_tercile_probabilities_nonparametric(best_guess, error_samples, first_tercile, second_tercile, nonneg=False):
        """Empirical CDF with finite errors, declared support and fixed events."""
        return _residual_probabilities(best_guess, error_samples, first_tercile, second_tercile, nonneg=nonneg)


    def compute_prob(self, Predictant, clim_year_start, clim_year_end, hindcast_det,
                     best_code_da=None, best_shape_da=None, best_loc_da=None,
                     best_scale_da=None, **probability_options):
        """Calibrate hindcasts with outer reuse by default; inner needs explicit residuals.

        The held-out year's error is excluded. Use ``diagnostic=True`` explicitly
        for a full-hindcast residual pool, which is unsuitable for skill scores.
        """
        from .was_probability_calibration import direct_hindcast_probabilities
        probability_options.setdefault("tercile_reference", getattr(self, "tercile_reference", "climatology"))
        return direct_hindcast_probabilities(self, Predictant, clim_year_start,
            clim_year_end, hindcast_det, best_code_da, best_shape_da,
            best_loc_da, best_scale_da, **probability_options)


    # --------------------------------------------------------------------------
    #  FORECAST METHOD
    # --------------------------------------------------------------------------
    def forecast(self, Predictant, clim_year_start, clim_year_end, Predictor,
                 hindcast_det, Predictor_for_year, best_code_da=None,
                 best_shape_da=None, best_loc_da=None, best_scale_da=None):
        return _operational_forecast(self, Predictant, clim_year_start, clim_year_end,
            Predictor, hindcast_det, Predictor_for_year, parameters=(),
            best_code_da=best_code_da, best_shape_da=best_shape_da,
            best_loc_da=best_loc_da, best_scale_da=best_scale_da, raw_target=False)
        
        
def _add_intercept(X: np.ndarray) -> np.ndarray:
    return np.column_stack([np.ones(X.shape[0], dtype=float), X])


def _wls_solve(X: np.ndarray, z: np.ndarray, w: np.ndarray) -> np.ndarray:
    """Weighted least squares via sqrt(w) trick."""
    sw = np.sqrt(np.clip(w, 1e-12, None))
    Xw = X * sw[:, None]
    zw = z * sw
    beta, *_ = np.linalg.lstsq(Xw, zw, rcond=None)
    return beta


def _poisson_irls_beta(y: np.ndarray, X: np.ndarray, max_iter: int = 60, tol: float = 1e-8) -> np.ndarray:
    """Poisson GLM (log link) IRLS."""
    n, p = X.shape
    beta = np.zeros(p, dtype=float)
    for _ in range(max_iter):
        eta = np.clip(X @ beta, -700.0, 700.0)
        mu = np.exp(eta)
        mu = np.clip(mu, 1e-12, None)

        z = eta + (y - mu) / mu
        w = mu  # Poisson: Var=mu, dmu/deta=mu => w = mu^2/Var = mu

        beta_new = _wls_solve(X, z, w)
        if np.max(np.abs(beta_new - beta)) < tol:
            beta = beta_new
            break
        beta = beta_new
    return beta


def _nb2_irls_beta_alpha(
    y: np.ndarray,
    X: np.ndarray,
    max_iter: int = 80,
    tol: float = 1e-8,
    alpha_init: float = 0.2,
) -> tuple[np.ndarray, float]:
    """
    Negative Binomial (NB2) GLM with log link: Var = mu + alpha*mu^2.

    Practical alternating scheme:
      - update beta with IRLS given alpha
      - update alpha with a method-of-moments estimator from residuals

    This is Dask-safe and robust, but not a full MLE for alpha.
    """
    n, p = X.shape
    df_resid = max(n - p, 1)

    beta = _poisson_irls_beta(y, X, max_iter=40, tol=tol)  # good starting point
    alpha = max(float(alpha_init), 1e-10)

    for _ in range(max_iter):
        eta = np.clip(X @ beta, -700.0, 700.0)
        mu = np.exp(eta)
        mu = np.clip(mu, 1e-12, None)

        # IRLS for NB2 GLM: w = (dmu/deta)^2 / Var = mu^2 / (mu + alpha mu^2) = mu/(1+alpha mu)
        w = mu / (1.0 + alpha * mu)
        z = eta + (y - mu) / mu

        beta_new = _wls_solve(X, z, w)

        # Update alpha (MoM) using: Var - mu ≈ alpha mu^2
        # Use mean of ((y-mu)^2 - y)/mu^2, stabilized and truncated at 0
        resid2 = (y - mu) ** 2
        alpha_raw = np.nanmean((resid2 - y) / (mu ** 2))
        alpha_new = float(np.clip(alpha_raw, 0.0, 1e6))

        # Convergence check
        if (np.max(np.abs(beta_new - beta)) < tol) and (abs(alpha_new - alpha) < 1e-6):
            beta, alpha = beta_new, max(alpha_new, 1e-10)
            break

        beta, alpha = beta_new, max(alpha_new, 1e-10)

    return beta, float(alpha)


def _logit_irls_coef(
    y01: np.ndarray,
    X: np.ndarray,
    max_iter: int = 60,
    tol: float = 1e-8,
) -> np.ndarray:
    """Binomial logistic regression via IRLS (NumPy-only)."""
    n, p = X.shape
    beta = np.zeros(p, dtype=float)

    for _ in range(max_iter):
        eta = np.clip(X @ beta, -35.0, 35.0)
        p_hat = 1.0 / (1.0 + np.exp(-eta))

        # IRLS for logit
        w = p_hat * (1.0 - p_hat)
        w = np.clip(w, 1e-12, None)

        z = eta + (y01 - p_hat) / w  # since dmu/deta = w for logit; this is standard IRLS form

        beta_new = _wls_solve(X, z, w)
        if np.max(np.abs(beta_new - beta)) < tol:
            beta = beta_new
            break
        beta = beta_new

    return beta


def _safe_mask_xy(x: np.ndarray, y: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    m = np.all(np.isfinite(x), axis=1) & np.isfinite(y)
    return x[m], y[m]


def _count_mean_prediction(kind, x, y, x_test, add_intercept=True, alpha_init=0.2):
    """NB2/ZINB/hurdle conditional means fitted by the corresponding likelihood.

    ZINB jointly estimates structural-zero probability and NB mean. Hurdle fits
    occurrence separately and includes the zero-truncation normalizer when fitting
    positive counts. Bounded optimization controls separation and weak dispersion
    identification; these models still need out-of-sample evaluation.
    """
    from scipy.optimize import minimize
    from scipy.special import expit, gammaln
    x, y = np.asarray(x, float), np.asarray(y, float).reshape(-1)
    if x.ndim == 1:
        x = x[:, None]
    x, y = _safe_mask_xy(x, y)
    xt = np.asarray(x_test, float).reshape(1, -1)
    if len(y) < 5 or not np.isfinite(xt).all():
        return np.nan
    if np.any(y < 0) or not np.allclose(y, np.round(y), rtol=0, atol=1e-8):
        raise ValueError("NB, ZINB and hurdle models require nonnegative integer counts")
    if np.max(y) == 0:
        return 0.0
    scaler = StandardScaler(with_mean=bool(add_intercept))
    x, xt = scaler.fit_transform(x), scaler.transform(xt)
    X = _add_intercept(x) if add_intercept else x
    Xt = _add_intercept(xt) if add_intercept else xt
    p = X.shape[1]
    initial_beta = np.zeros(p)
    if add_intercept:
        initial_beta[0] = np.log(max(np.mean(y), 1e-6))
    initial_alpha = np.log(max(float(alpha_init), 1e-5))

    def nb_terms(counts, design, beta, log_alpha):
        alpha = np.exp(log_alpha)
        size = 1.0 / alpha
        eta = np.clip(design @ beta, -30., 30.)
        log1p = np.logaddexp(0., log_alpha + eta)
        logpmf = (gammaln(counts+size)-gammaln(size)-gammaln(counts+1.)
                  + counts*(log_alpha+eta) - (counts+size)*log1p)
        return logpmf, -size*log1p

    if kind == "hurdle":
        positive = y > 0
        # Binary occurrence includes all training observations.
        if positive.all():
            pplus = 1.0
        else:
            from sklearn.linear_model import LogisticRegression
            occurrence = LogisticRegression(C=1e6, fit_intercept=bool(add_intercept),
                                            max_iter=2000, random_state=42)
            occurrence.fit(x, positive.astype(int))
            pplus = float(occurrence.predict_proba(xt)[0, 1])
        if positive.sum() < 3:
            return float(pplus * np.mean(y[positive]))
        def loss(theta):
            logpmf, logzero = nb_terms(y[positive], X[positive], theta[:p], theta[p])
            log_nonzero = np.log(-np.expm1(logzero))
            return -float(np.sum(logpmf - log_nonzero))
        initial = np.r_[initial_beta, initial_alpha]
    elif kind == "zinb":
        pi0 = np.clip(np.mean(y == 0) / 2.0, 1e-4, 1-1e-4)
        gamma0 = np.zeros(p)
        if add_intercept:
            gamma0[0] = np.log(pi0/(1-pi0))
        def loss(theta):
            logpmf, logzero = nb_terms(y, X, theta[:p], theta[p])
            eta_zero = np.clip(X @ theta[p+1:], -30., 30.)
            logpi = -np.logaddexp(0., -eta_zero)
            lognotpi = -np.logaddexp(0., eta_zero)
            values = np.where(y == 0, np.logaddexp(logpi, lognotpi+logzero), lognotpi+logpmf)
            return -float(np.sum(values))
        initial = np.r_[initial_beta, initial_alpha, gamma0]
    else:
        def loss(theta):
            logpmf, _ = nb_terms(y, X, theta[:p], theta[p])
            return -float(np.sum(logpmf))
        initial = np.r_[initial_beta, initial_alpha]
    bounds = [(-25., 25.)]*p + [(-12., 10.)]
    if kind == "zinb":
        bounds += [(-25., 25.)]*p
    result = minimize(loss, initial, method="L-BFGS-B", bounds=bounds,
                      options={"maxiter": 500, "ftol": 1e-10})
    if not result.success or not np.isfinite(result.fun):
        warnings.warn(f"{kind} likelihood did not converge: {result.message}", RuntimeWarning)
        return np.nan
    beta, log_alpha = result.x[:p], result.x[p]
    mu = float(np.exp(np.clip(Xt @ beta, -30., 30.))[0])
    if kind == "zinb":
        return float((1.0-expit(Xt @ result.x[p+1:])[0])*mu)
    if kind == "hurdle":
        alpha = np.exp(log_alpha)
        logzero = -np.log1p(alpha*mu)/alpha
        return float(pplus*mu/(-np.expm1(logzero)))
    return mu


class _WAS_CountProbMixin:
    @staticmethod
    def _ppf_terciles_from_code(dist_code, shape, loc, scale):
        if np.isnan(dist_code):
            return np.nan, np.nan
        code = int(dist_code)
        try:
            if code == 1:
                return (norm.ppf(1.0 / 3.0, loc=loc, scale=scale),
                        norm.ppf(2.0 / 3.0, loc=loc, scale=scale))
            elif code == 2:
                return (lognorm.ppf(1.0 / 3.0, s=shape, loc=loc, scale=scale),
                        lognorm.ppf(2.0 / 3.0, s=shape, loc=loc, scale=scale))
            elif code == 3:
                return (expon.ppf(1.0 / 3.0, loc=loc, scale=scale),
                        expon.ppf(2.0 / 3.0, loc=loc, scale=scale))
            elif code == 4:
                return (gamma.ppf(1.0 / 3.0, a=shape, loc=loc, scale=scale),
                        gamma.ppf(2.0 / 3.0, a=shape, loc=loc, scale=scale))
            elif code == 5:
                return (weibull_min.ppf(1.0 / 3.0, c=shape, loc=loc, scale=scale),
                        weibull_min.ppf(2.0 / 3.0, c=shape, loc=loc, scale=scale))
            elif code == 6:
                return (t.ppf(1.0 / 3.0, df=shape, loc=loc, scale=scale),
                        t.ppf(2.0 / 3.0, df=shape, loc=loc, scale=scale))
            elif code == 7:
                return (poisson.ppf(1.0 / 3.0, mu=shape, loc=loc),
                        poisson.ppf(2.0 / 3.0, mu=shape, loc=loc))
            elif code == 8:
                return (nbinom.ppf(1.0 / 3.0, n=shape, p=scale, loc=loc),
                        nbinom.ppf(2.0 / 3.0, n=shape, p=scale, loc=loc))
        except Exception:
            return np.nan, np.nan
        return np.nan, np.nan

    @staticmethod
    def weibull_shape_solver(k, M, V):
        if k <= 0:
            return -np.inf
        try:
            g1 = gamma_function(1 + 1 / k)
            g2 = gamma_function(1 + 2 / k)
            implied = (g2 / (g1 ** 2)) - 1
            observed = V / (M ** 2)
            return observed - implied
        except ValueError:
            return -np.inf

    @staticmethod
    def calculate_tercile_probabilities_bestfit(best_guess, error_variance, T1, T2, dist_code, dof, nonneg=False):
        """Shared predictive CDF; moment matching and support follow one contract."""
        from .was_linear_models import _regression_tercile_probabilities
        return _regression_tercile_probabilities(best_guess, error_variance, T1, T2,
                                               dist_code, dof, nonneg=nonneg)

    @staticmethod
    def calculate_tercile_probabilities_nonparametric(best_guess, error_samples, first_tercile, second_tercile, nonneg=False):
        """Empirical CDF with finite errors, declared support and fixed events."""
        return _residual_probabilities(best_guess, error_samples, first_tercile, second_tercile, nonneg=nonneg)


class WAS_NegativeBinomial_Model(_WAS_CountProbMixin):
    """
    Negative Binomial (NB2) regression for spatiotemporal count prediction.

    Deterministic prediction: mu_hat = E[Y|X] from NB2 GLM with log link.
        Var(Y|X) = mu + alpha * mu^2
    """

    def __init__(self, nb_cores=1, dist_method="nonparam",
                 alpha_init=0.2, add_intercept=True):
        self.nb_cores      = nb_cores
        self.dist_method   = dist_method
        self.alpha_init    = alpha_init
        self.add_intercept = add_intercept

    # ------------------------------------------------------------------
    def fit_predict(self, x, y, x_test, y_test=None):
        prediction = _count_mean_prediction("nb", x, y, x_test,
            add_intercept=self.add_intercept, alpha_init=self.alpha_init)
        error = np.nan if y_test is None else float(np.asarray(y_test).reshape(-1)[0]) - prediction
        return np.array([error, prediction])

    # ------------------------------------------------------------------
    def compute_model(self, X_train, y_train, X_test, y_test=None):
        return _spatial_regression(self, X_train, y_train, X_test, y_test)

    # ------------------------------------------------------------------
    def compute_prob(self, Predictant, clim_year_start, clim_year_end, hindcast_det,
                     best_code_da=None, best_shape_da=None, best_loc_da=None,
                     best_scale_da=None, **probability_options):
        """Calibrate hindcasts with outer reuse by default; inner needs explicit residuals.

        The held-out year's error is excluded. Use ``diagnostic=True`` explicitly
        for a full-hindcast residual pool, which is unsuitable for skill scores.
        """
        from .was_probability_calibration import direct_hindcast_probabilities
        probability_options.setdefault("tercile_reference", getattr(self, "tercile_reference", "climatology"))
        return direct_hindcast_probabilities(self, Predictant, clim_year_start,
            clim_year_end, hindcast_det, best_code_da, best_shape_da,
            best_loc_da, best_scale_da, **probability_options)

    # ------------------------------------------------------------------
    def forecast(self, Predictant, clim_year_start, clim_year_end, Predictor,
                 hindcast_det, Predictor_for_year, best_code_da=None,
                 best_shape_da=None, best_loc_da=None, best_scale_da=None):
        return _operational_forecast(self, Predictant, clim_year_start, clim_year_end,
            Predictor, hindcast_det, Predictor_for_year, parameters=(),
            best_code_da=best_code_da, best_shape_da=best_shape_da,
            best_loc_da=best_loc_da, best_scale_da=best_scale_da, raw_target=True)


# ============================================================
# WAS_ZINB_Model and WAS_HurdleNB_Model — only fit_predict
# differs; compute_model / compute_prob / forecast are
# inherited from the fixed WAS_NegativeBinomial_Model above.
# ============================================================

class WAS_ZINB_Model(WAS_NegativeBinomial_Model):
    """
    Zero-Inflated Negative Binomial fitted by joint mixture likelihood.

    P(Y=0|X) = pi(X) + (1-pi(X)) * NB(y=0 | mu(X), alpha)
    E[Y|X]   = (1 - pi(X)) * mu(X)
    """

    def fit_predict(self, x, y, x_test, y_test=None):
        prediction = _count_mean_prediction("zinb", x, y, x_test,
            add_intercept=self.add_intercept, alpha_init=self.alpha_init)
        error = np.nan if y_test is None else float(np.asarray(y_test).reshape(-1)[0]) - prediction
        return np.array([error, prediction])


class WAS_HurdleNB_Model(WAS_NegativeBinomial_Model):
    """
    Hurdle Negative Binomial (two-part, Dask-safe).

    P(Y>0|X) = p+(X)  via logistic regression.
    Y|Y>0,X  ~ zero-truncated NB fitted with its normalized likelihood.
    E[Y|X]   = p+(X) * mu(X) / (1 - NB_P(Y=0|mu, alpha))
    """

    def fit_predict(self, x, y, x_test, y_test=None):
        prediction = _count_mean_prediction("hurdle", x, y, x_test,
            add_intercept=self.add_intercept, alpha_init=self.alpha_init)
        error = np.nan if y_test is None else float(np.asarray(y_test).reshape(-1)[0]) - prediction
        return np.array([error, prediction])

# # Choose the model
# model_nb   = WAS_NegativeBinomial_Model(nb_cores=8, dist_method="nonparam")
# model_zinb = WAS_ZINB_Model(nb_cores=8, dist_method="nonparam")
# model_hurd = WAS_HurdleNB_Model(nb_cores=8, dist_method="nonparam")

# # Deterministic hindcast on grid:
# hind_det = model_nb.compute_model(Predictor, Predictant, X_test, y_test)

# # Probabilities for hindcast:
# hind_prob = model_nb.compute_prob(Predictant, clim_year_start, clim_year_end, hind_det, ...)

# # One-year forecast:
# fcst_det, fcst_prob = model_nb.forecast(Predictant, clim_year_start, clim_year_end,
#                                        Predictor, hind_det, Predictor_for_year, ...)


"""
WAS non-neural machine-learning models for spatial seasonal forecasting.


Available estimators
--------------------
    WAS_RandomForest_Model      WAS_ExtraTrees_Model
    WAS_GradientBoosting_Model  WAS_AdaBoost_Model
    WAS_XGBoost_Model           WAS_LightGBM_Model      (xgboost / lightgbm optional)
    WAS_SVR_Model               WAS_KNN_Model           (auto-scaled)
"""


# =====================================================================
#  Base class: everything shared by every non-neural estimator
# =====================================================================
class _WAS_NonNeural_Base:
    """
    Base class implementing the full spatial forecasting pipeline for any
    scikit-learn-compatible regressor. Subclasses only need to define
    ``_default_param_space`` and ``_make_estimator``.

    Parameters
    ----------
    param_space : dict, optional
        Search space, mapping ``name -> (low, high, is_int, is_log)``.
        Defaults to the subclass' ``_default_param_space()``.
    n_clusters : int, default=5
        Number of spatial clusters (mode='cluster').
    nb_cores : int, default=1
        Dask workers for spatial parallelism. Estimators run single-threaded
        (n_jobs=1) so the parallelism stays at the spatial level only.
    dist_method : {'nonparam', 'bestfit'}, default='nonparam'
        Tercile-probability strategy (identical to WAS_Ridge_Model).
    hyperparam_optimizer : {'bayesian', 'random'}, default='bayesian'
        'bayesian' -> Optuna; 'random' -> manual randomised search (both route
        every candidate through ``_make_estimator`` + 3-fold CV neg-MSE, so they
        work uniformly even for pipeline-wrapped estimators like SVR/KNN).
    n_trials : int, default=50
        Optuna trials.
    n_iter : int, default=50
        Randomised-search samples.
    mode : {'cluster', 'grid'}, default='cluster'
        'cluster' -> one hyperparameter set per spatial cluster (recommended).
        'grid'    -> independent optimisation per grid cell (very slow).
    """

    def __init__(self, param_space=None, n_clusters=5, nb_cores=1,
                 dist_method="nonparam", hyperparam_optimizer="bayesian",
                 n_trials=50, n_iter=50, mode="cluster", inner_cv=3, random_state=42):
        self.param_space = param_space or self._default_param_space()
        self._param_names = list(self.param_space.keys())
        self._int_params = {
            name for name, (lo, hi, is_int, is_log) in self.param_space.items() if is_int
        }
        self.n_clusters = n_clusters
        self.nb_cores = nb_cores
        self.dist_method = dist_method
        self.hyperparam_optimizer = hyperparam_optimizer
        self.n_trials = n_trials
        self.n_iter = n_iter
        self.mode = mode
        self.inner_cv = inner_cv
        self.random_state = random_state

    # ----- to be provided by subclasses ------------------------------
    def _default_param_space(self):
        raise NotImplementedError

    def _make_estimator(self, **params):
        raise NotImplementedError

    # ----- hyperparameter helpers ------------------------------------
    def _cast_params(self, params):
        """Cast integer hyperparameters to int, the rest to float."""
        out = {}
        for name, val in params.items():
            out[name] = int(round(val)) if name in self._int_params else float(val)
        return out

    def _suggest_params(self, trial):
        params = {}
        for name, (lo, hi, is_int, is_log) in self.param_space.items():
            if is_int:
                params[name] = trial.suggest_int(name, int(lo), int(hi))
            else:
                params[name] = trial.suggest_float(name, lo, hi, log=is_log)
        return params

    def _param_distributions(self):
        dists = {}
        for name, (lo, hi, is_int, is_log) in self.param_space.items():
            if is_int:
                dists[name] = randint(int(lo), int(hi) + 1)
            elif is_log:
                dists[name] = loguniform(lo, hi)
            else:
                dists[name] = uniform(lo, hi - lo)
        return dists

    def _optimize_optuna(self, X, y):
        def objective(trial):
            params = self._cast_params(self._suggest_params(trial))
            return _regression_cv_score(self, X, y, params)
        study = optuna.create_study(direction="maximize",
            sampler=optuna.samplers.TPESampler(seed=self.random_state))
        study.optimize(objective, n_trials=self.n_trials, show_progress_bar=False)
        return study.best_params

    def _optimize_random(self, X, y):
        rng = np.random.RandomState(self.random_state)
        dists = self._param_distributions()
        best_score, best = -np.inf, None
        for _ in range(self.n_iter):
            params = self._cast_params({name: dist.rvs(random_state=rng) for name, dist in dists.items()})
            score = _regression_cv_score(self, X, y, params)
            if score > best_score:
                best_score, best = score, params
        return best

    def _optimize_single_cell(self, y_vec, X_mat):
        """
        Optimise hyperparameters for one pixel / cluster mean.

        Returns a 1D array of best hyperparameter values, ordered by
        ``self._param_names`` (or an all-NaN array if it cannot be optimised).
        """
        n = len(self._param_names)
        nan_out = np.full(n, np.nan)
        y_vec = np.asarray(y_vec, float)
        valid_y = np.isfinite(y_vec) if y_vec.ndim == 1 else np.isfinite(y_vec).any(axis=1)
        mask = valid_y & np.all(np.isfinite(X_mat), axis=-1)
        if np.sum(mask) < 2:
            return nan_out

        X_clean, y_clean = X_mat[mask], y_vec[mask]
        # Constant fields have no tunable signal: choose legal parameters once.
        columns = y_clean[:, None] if y_clean.ndim == 1 else y_clean
        if all(np.unique(c[np.isfinite(c)]).size <= 1 for c in columns.T):
            return np.array([float(self.param_space[name][0]) for name in self._param_names])
        if np.sum(mask) < 10:  # Avoid unconstrained model selection on too few years.
            return nan_out
        try:
            if self.hyperparam_optimizer == "random":
                best = self._optimize_random(X_clean, y_clean)
            else:
                best = self._optimize_optuna(X_clean, y_clean)
            if best is None:
                return nan_out
            return np.array([float(best[name]) for name in self._param_names])
        except Exception as exc:
            raise RuntimeError(f"{type(self).__name__} hyperparameter tuning failed") from exc

    # ----- core logic ------------------------------------------------
    def compute_hyperparameters(self, predictand, predictor,
                                clim_year_start, clim_year_end):
        """
        Compute spatially varying hyperparameters.

        Returns
        -------
        hyper_ds : xarray.Dataset
            One (Y, X) field per hyperparameter (named after the search space
            keys). Constant per cluster in 'cluster' mode. Feed this back into
            ``compute_model`` / ``forecast`` (replaces Ridge's ``alpha``).
        cluster_da : xarray.DataArray
            Cluster labels (cluster mode) or a 1/NaN validity mask (grid mode).
        """
        predictor, predictand = _aligned_training(predictor, predictand)
        if self.mode not in {"grid", "cluster"}:
            raise ValueError("mode must be 'grid' or 'cluster'")
        if self.mode == "cluster" and any(dim in predictor.dims for dim in ("Y", "X")):
            raise ValueError("Spatial predictors require mode='grid' or explicit regional aggregation")
        predictand_st = predictand  # scaling is fitted only inside each search fold
        names = self._param_names

        if self.mode == "grid":
            chunk_y = max(1, int(np.ceil(len(predictand_st.Y) / max(1, self.nb_cores))))
            chunk_x = max(1, int(np.ceil(len(predictand_st.X) / max(1, self.nb_cores))))
            p_st_chunked = predictand_st.chunk({"T": -1, "Y": chunk_y, "X": chunk_x})

            client = _get_compute_client(self.nb_cores) if self.nb_cores > 1 else None
            hp_array = xr.apply_ufunc(
                self._optimize_single_cell,
                p_st_chunked,
                predictor,
                input_core_dims=[("T",), ("T", "features")],
                output_core_dims=[("param",)],
                vectorize=True,
                dask="parallelized",
                output_dtypes=[float],
                dask_gufunc_kwargs={"output_sizes": {"param": len(names)}, "allow_rechunk": True},
            )
            hp_array = hp_array.compute()
            _safe_close_client(client)

            hyper_ds = xr.Dataset(
                {n: hp_array.isel(param=i, drop=True) for i, n in enumerate(names)}
            )
            cluster_da = xr.where(~np.isnan(hyper_ds[names[0]]), 1, np.nan)
            _report_tuned("_WAS_NonNeural_Base", getattr(self, "verbose", True), cluster_da=cluster_da, **{v: hyper_ds[v] for v in hyper_ds.data_vars})
            return hyper_ds, cluster_da

        # ---- cluster mode ----
        print(f"{self.__class__.__name__}: cluster-wise optimization "
              f"({self.n_clusters} clusters)...")
        cluster_da = geographical_cluster_map(
            predictand, self.n_clusters, random_state=getattr(self, 'random_state', 42))

        x1, x2 = xr.align(predictand_st, cluster_da, join="outer")
        clusters = np.unique(x2)
        clusters = clusters[~np.isnan(clusters)]
        cluster_da = x2

        hyper_maps = {n: xr.full_like(cluster_da, np.nan, dtype=float) for n in names}
        for clus in clusters:
            # Keep raw cell columns: train-only scaling and averaging happen
            # within _regression_cv_score, never before its inner split.
            y_cluster = x1.stack(cell=("Y", "X")).isel(
                cell=np.flatnonzero(np.asarray(x2 == clus).ravel())).transpose("T", "cell")
            if len(y_cluster["T"]) > 0:
                best = self._optimize_single_cell(
                    y_cluster.values, predictor.sel(T=y_cluster["T"]).values
                )
                for i, n in enumerate(names):
                    hyper_maps[n] = hyper_maps[n].where(cluster_da != clus, best[i])

        hyper_ds = xr.Dataset(hyper_maps)
        hyper_ds.attrs.update(tuning_grouping=cluster_da.attrs.get("grouping", "per-cell"),
                              tuning_preprocessing="per-cell scaling fitted inside each inner search fold")
        _report_tuned("_WAS_NonNeural_Base", getattr(self, "verbose", True), cluster_da=cluster_da, **{v: hyper_ds[v] for v in hyper_ds.data_vars})
        return hyper_ds, cluster_da

    def fit_predict(self, x, y, x_test, y_test, *param_values):
        if len(param_values) != len(self._param_names):
            raise ValueError("One parameter value is required for every hyperparameter")
        params = dict(zip(self._param_names, param_values))
        if any(not np.isfinite(v) for v in params.values()):
            return np.array([np.nan, np.nan])
        estimator = self._make_estimator(**self._cast_params(params))
        return _regression_pair(estimator, x, y, x_test, y_test)

    def compute_model(self, X_train, y_train, X_test, y_test, hyperparams):
        parameters = tuple(hyperparams[name] for name in self._param_names)
        return _spatial_regression(self, X_train, y_train, X_test, y_test, parameters)

    # ----- probability calculation (identical to WAS_Ridge_Model) ----
    @staticmethod
    def _ppf_terciles_from_code(dist_code, shape, loc, scale):
        """Tercile thresholds (T1, T2) from best-fit distribution parameters."""
        if np.isnan(dist_code):
            return np.nan, np.nan
        code = int(dist_code)
        try:
            if code == 1:
                return (norm.ppf(1.0 / 3.0, loc=loc, scale=scale),
                        norm.ppf(2.0 / 3.0, loc=loc, scale=scale))
            elif code == 2:
                return (lognorm.ppf(1.0 / 3.0, s=shape, loc=loc, scale=scale),
                        lognorm.ppf(2.0 / 3.0, s=shape, loc=loc, scale=scale))
            elif code == 3:
                return (expon.ppf(1.0 / 3.0, loc=loc, scale=scale),
                        expon.ppf(2.0 / 3.0, loc=loc, scale=scale))
            elif code == 4:
                return (gamma.ppf(1.0 / 3.0, a=shape, loc=loc, scale=scale),
                        gamma.ppf(2.0 / 3.0, a=shape, loc=loc, scale=scale))
            elif code == 5:
                return (weibull_min.ppf(1.0 / 3.0, c=shape, loc=loc, scale=scale),
                        weibull_min.ppf(2.0 / 3.0, c=shape, loc=loc, scale=scale))
            elif code == 6:
                return (t.ppf(1.0 / 3.0, df=shape, loc=loc, scale=scale),
                        t.ppf(2.0 / 3.0, df=shape, loc=loc, scale=scale))
            elif code == 7:
                return (poisson.ppf(1.0 / 3.0, mu=shape, loc=loc),
                        poisson.ppf(2.0 / 3.0, mu=shape, loc=loc))
            elif code == 8:
                return (nbinom.ppf(1.0 / 3.0, n=shape, p=scale, loc=loc),
                        nbinom.ppf(2.0 / 3.0, n=shape, p=scale, loc=loc))
        except Exception:
            return np.nan, np.nan
        return np.nan, np.nan

    @staticmethod
    def weibull_shape_solver(k, M, V):
        """Solver for Weibull shape parameter."""
        if k <= 0:
            return -np.inf
        try:
            g1 = gamma_function(1 + 1 / k)
            g2 = gamma_function(1 + 2 / k)
            implied_v_over_m_sq = (g2 / (g1 ** 2)) - 1
            observed_v_over_m_sq = V / (M ** 2)
            return observed_v_over_m_sq - implied_v_over_m_sq
        except ValueError:
            return -np.inf

    @staticmethod
    def calculate_tercile_probabilities_bestfit(best_guess, error_variance, T1, T2, dist_code, dof, nonneg=False):
        """Shared predictive CDF; moment matching and support follow one contract."""
        from .was_linear_models import _regression_tercile_probabilities
        return _regression_tercile_probabilities(best_guess, error_variance, T1, T2,
                                               dist_code, dof, nonneg=nonneg)

    @staticmethod
    def calculate_tercile_probabilities_nonparametric(best_guess, error_samples, first_tercile, second_tercile, nonneg=False):
        """Empirical CDF with finite errors, declared support and fixed events."""
        return _residual_probabilities(best_guess, error_samples, first_tercile, second_tercile, nonneg=nonneg)

    def compute_prob(self, Predictant, clim_year_start, clim_year_end, hindcast_det,
                     best_code_da=None, best_shape_da=None, best_loc_da=None,
                     best_scale_da=None, **probability_options):
        """Calibrate hindcasts with outer reuse by default; inner needs explicit residuals.

        The held-out year's error is excluded. Use ``diagnostic=True`` explicitly
        for a full-hindcast residual pool, which is unsuitable for skill scores.
        """
        from .was_probability_calibration import direct_hindcast_probabilities
        probability_options.setdefault("tercile_reference", getattr(self, "tercile_reference", "climatology"))
        return direct_hindcast_probabilities(self, Predictant, clim_year_start,
            clim_year_end, hindcast_det, best_code_da, best_shape_da,
            best_loc_da, best_scale_da, **probability_options)

    def forecast(self, Predictant, clim_year_start, clim_year_end, Predictor,
                 hindcast_det, Predictor_for_year, hyperparams, best_code_da=None,
                 best_shape_da=None, best_loc_da=None, best_scale_da=None):
        return _operational_forecast(self, Predictant, clim_year_start, clim_year_end,
            Predictor, hindcast_det, Predictor_for_year, parameters=tuple(hyperparams[name] for name in self._param_names),
            best_code_da=best_code_da, best_shape_da=best_shape_da,
            best_loc_da=best_loc_da, best_scale_da=best_scale_da, raw_target=False)


# =====================================================================
#  Concrete estimators
# =====================================================================
class WAS_RandomForest_Model(_WAS_NonNeural_Base):
    """Random Forest regressor (sklearn). ``max_features`` is a fraction in (0, 1]."""

    def _default_param_space(self):
        return {
            "n_estimators":      (50, 500, True, False),
            "max_depth":         (2, 30, True, False),
            "min_samples_split": (2, 20, True, False),
            "min_samples_leaf":  (1, 20, True, False),
            "max_features":      (0.1, 1.0, False, False),
        }

    def _make_estimator(self, **params):
        from sklearn.ensemble import RandomForestRegressor
        return RandomForestRegressor(random_state=self.random_state, n_jobs=1, **params)


class WAS_ExtraTrees_Model(_WAS_NonNeural_Base):
    """Extremely Randomized Trees regressor (sklearn)."""

    def _default_param_space(self):
        return {
            "n_estimators":      (50, 500, True, False),
            "max_depth":         (2, 30, True, False),
            "min_samples_split": (2, 20, True, False),
            "min_samples_leaf":  (1, 20, True, False),
            "max_features":      (0.1, 1.0, False, False),
        }

    def _make_estimator(self, **params):
        from sklearn.ensemble import ExtraTreesRegressor
        return ExtraTreesRegressor(random_state=self.random_state, n_jobs=1, **params)


class WAS_GradientBoosting_Model(_WAS_NonNeural_Base):
    """Gradient Boosting regressor (sklearn)."""

    def _default_param_space(self):
        return {
            "n_estimators":     (50, 500, True, False),
            "learning_rate":    (1e-3, 0.3, False, True),
            "max_depth":        (2, 8, True, False),
            "subsample":        (0.5, 1.0, False, False),
            "min_samples_leaf": (1, 20, True, False),
        }

    def _make_estimator(self, **params):
        from sklearn.ensemble import GradientBoostingRegressor
        return GradientBoostingRegressor(random_state=self.random_state, **params)


class WAS_AdaBoost_Model(_WAS_NonNeural_Base):
    """AdaBoost regressor (sklearn)."""

    def _default_param_space(self):
        return {
            "n_estimators":  (50, 500, True, False),
            "learning_rate": (1e-3, 1.0, False, True),
        }

    def _make_estimator(self, **params):
        from sklearn.ensemble import AdaBoostRegressor
        return AdaBoostRegressor(random_state=self.random_state, **params)


class WAS_XGBoost_Model(_WAS_NonNeural_Base):
    """XGBoost regressor. Requires the ``xgboost`` package."""

    def _default_param_space(self):
        return {
            "n_estimators":     (50, 600, True, False),
            "max_depth":        (2, 12, True, False),
            "learning_rate":    (1e-3, 0.3, False, True),
            "subsample":        (0.5, 1.0, False, False),
            "colsample_bytree": (0.5, 1.0, False, False),
            "min_child_weight": (1, 10, True, False),
            "reg_alpha":        (1e-8, 10.0, False, True),
            "reg_lambda":       (1e-8, 10.0, False, True),
            "gamma":            (1e-8, 5.0, False, True),
        }

    def _make_estimator(self, **params):
        from xgboost import XGBRegressor
        return XGBRegressor(
            random_state=self.random_state, n_jobs=1, verbosity=0,
            objective="reg:squarederror", tree_method="hist", **params
        )


class WAS_LightGBM_Model(_WAS_NonNeural_Base):
    """LightGBM regressor. Requires the ``lightgbm`` package."""

    def _default_param_space(self):
        return {
            "n_estimators":      (50, 600, True, False),
            "num_leaves":        (15, 255, True, False),
            "max_depth":         (3, 12, True, False),
            "learning_rate":     (1e-3, 0.3, False, True),
            "subsample":         (0.5, 1.0, False, False),
            "colsample_bytree":  (0.5, 1.0, False, False),
            "reg_alpha":         (1e-8, 10.0, False, True),
            "reg_lambda":        (1e-8, 10.0, False, True),
            "min_child_samples": (5, 50, True, False),
        }

    def _make_estimator(self, **params):
        from lightgbm import LGBMRegressor
        return LGBMRegressor(random_state=self.random_state, n_jobs=1, verbosity=-1, subsample_freq=1, **params)


class WAS_SVR_Model(_WAS_NonNeural_Base):
    """Support Vector Regression (RBF kernel), auto-scaled with StandardScaler."""

    def _default_param_space(self):
        return {
            "C":       (1e-2, 1e3, False, True),
            "epsilon": (1e-3, 1.0, False, True),
            "gamma":   (1e-4, 1e1, False, True),
        }

    def _make_estimator(self, **params):
        from sklearn.svm import SVR
        from sklearn.preprocessing import StandardScaler
        from sklearn.pipeline import make_pipeline
        return make_pipeline(StandardScaler(), SVR(kernel="rbf", **params))


class WAS_KNN_Model(_WAS_NonNeural_Base):
    """K-Nearest-Neighbours regressor, auto-scaled with StandardScaler."""

    def _default_param_space(self):
        return {
            "n_neighbors": (2, 30, True, False),
            "p":           (1, 2, True, False),
        }

    def _make_estimator(self, **params):
        from sklearn.neighbors import KNeighborsRegressor
        from sklearn.preprocessing import StandardScaler
        from sklearn.pipeline import make_pipeline
        return make_pipeline(
            StandardScaler(),
            KNeighborsRegressor(weights="distance", n_jobs=1, **params),
        )


# # Convenience registry: name -> class
# WAS_NONNEURAL_MODELS = {
#     "rf":        WAS_RandomForest_Model,
#     "extratrees": WAS_ExtraTrees_Model,
#     "gbr":       WAS_GradientBoosting_Model,
#     "adaboost":  WAS_AdaBoost_Model,
#     "xgboost":   WAS_XGBoost_Model,
#     "lightgbm":  WAS_LightGBM_Model,
#     "svr":       WAS_SVR_Model,
#     "knn":       WAS_KNN_Model,
# }


# ---------------------------------------------------------------------
# Example 
#
#   model = WAS_XGBoost_Model(mode="cluster", n_clusters=8,
#                             hyperparam_optimizer="bayesian",
#                             n_trials=80, nb_cores=12)
#   hp_ds, cluster_map = model.compute_hyperparameters(
#       seasonal_rainfall, predictors, 1991, 2020)        # -> xr.Dataset
#   hindcast = model.compute_model(predictors, seasonal_rainfall,
#                                  predictors_hc, seasonal_rainfall, hp_ds)
#   hc_prob  = model.compute_prob(seasonal_rainfall, 1991, 2020, hindcast)
#   fc_det, fc_prob = model.forecast(seasonal_rainfall, 1991, 2020,
#                                    predictors, hindcast, predictor_2025, hp_ds)
# ---------------------------------------------------------------------


# ======================================================================
#  Base class
# ======================================================================
class _WAS_NonNeural_Classifier_Base:
    """
    Shared infrastructure for non-neural tercile classifiers.

    Subclasses must implement:
        _default_param_space() -> dict
            Keys: hyperparameter names.
            Values: (lo, hi, is_int, is_log) tuples.
        _make_estimator(**params) -> sklearn-compatible classifier
            Must expose ``predict_proba`` and ``classes_``.

    Parameters
    ----------
    n_clusters : int, default=5
        Number of KMeans spatial clusters for hyperparameter optimisation.
    nb_cores : int, default=1
        Dask workers for spatial parallelism.
    hyperparam_optimizer : {'bayesian', 'random'}, default='bayesian'
        Optimiser backend.  'bayesian' uses Optuna TPE; 'random' uses a
        manual random draw evaluated with 3-fold cross-validation.
    n_trials : int, default=50
        Number of Optuna trials (bayesian mode).
    n_iter : int, default=50
        Number of random candidates (random mode).
    scoring : str, default='neg_log_loss'
        sklearn scorer name used during hyperparameter search.
        'neg_log_loss' is recommended for probabilistic classifiers.
    random_state : int, default=42
        Seed for reproducibility (KMeans, CV splits, Optuna sampler).
    """

    def __init__(
        self,
        n_clusters: int = 5,
        nb_cores: int = 1,
        hyperparam_optimizer: str = "bayesian",
        n_trials: int = 50,
        n_iter: int = 50,
        scoring: str = "neg_log_loss",
        random_state: int = 42,
        inner_cv=3,
    ):
        self.n_clusters = int(n_clusters)
        self.nb_cores = int(nb_cores)
        self.hyperparam_optimizer = hyperparam_optimizer
        self.n_trials = int(n_trials)
        self.n_iter = int(n_iter)
        self.scoring = scoring
        self.random_state = int(random_state)
        self.inner_cv = inner_cv

        self.param_space = self._default_param_space()
        self._param_names = list(self.param_space.keys())
        self._int_params = {
            name
            for name, (lo, hi, is_int, is_log) in self.param_space.items()
            if is_int
        }

    # ------------------------------------------------------------------
    # Subclass interface
    # ------------------------------------------------------------------
    def _default_param_space(self) -> dict:
        raise NotImplementedError

    def _make_estimator(self, **params):
        raise NotImplementedError

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------
    def _safe_chunk(self, n: int) -> int:
        return max(int(np.ceil(n / max(self.nb_cores, 1))), 1)

    def _cast_params(self, params: dict) -> dict:
        out = {}
        for name, val in params.items():
            out[name] = int(round(val)) if name in self._int_params else float(val)
        return out

    def _suggest_params(self, trial) -> dict:
        params = {}
        for name, (lo, hi, is_int, is_log) in self.param_space.items():
            if is_int:
                params[name] = trial.suggest_int(name, int(lo), int(hi))
            else:
                params[name] = trial.suggest_float(name, lo, hi, log=is_log)
        return params

    def _param_distributions(self) -> dict:
        dists = {}
        for name, (lo, hi, is_int, is_log) in self.param_space.items():
            if is_int:
                dists[name] = randint(int(lo), int(hi) + 1)
            elif is_log:
                dists[name] = loguniform(lo, hi)
            else:
                dists[name] = uniform(lo, hi - lo)
        return dists

    # ------------------------------------------------------------------
    # Optimisers
    # ------------------------------------------------------------------
    def _optimize_optuna(self, X, y):
        def objective(trial):
            params = self._cast_params(self._suggest_params(trial))
            return _classifier_cv_score(self, X, y, params)
        study = optuna.create_study(direction="maximize",
            sampler=optuna.samplers.TPESampler(seed=self.random_state))
        study.optimize(objective, n_trials=self.n_trials, show_progress_bar=False)
        return study.best_params

    def _optimize_random(self, X, y):
        rng = np.random.RandomState(self.random_state)
        dists = self._param_distributions()
        best_score, best = -np.inf, None
        for _ in range(self.n_iter):
            params = self._cast_params({name: dist.rvs(random_state=rng) for name, dist in dists.items()})
            score = _classifier_cv_score(self, X, y, params)
            if score > best_score:
                best_score, best = score, params
        return best

    def _optimize_single_cluster(
        self, y_vec: np.ndarray, X_mat: np.ndarray
    ) -> np.ndarray:
        """
        Optimise hyperparameters for one cluster's representative time series.

        Returns a 1-D array of best values ordered by ``self._param_names``,
        or all-NaN if optimisation fails (too few samples or single class).
        """
        n_params = len(self._param_names)
        nan_out = np.full(n_params, np.nan)

        mask = np.isfinite(y_vec) & np.all(np.isfinite(X_mat), axis=-1)
        if np.sum(mask) < 2:
            return nan_out
        X_c = X_mat[mask]
        y_c = y_vec[mask].astype(int)
        if len(np.unique(y_c)) < 2 or all(name.startswith("_placeholder") for name in self._param_names):
            # A one-class target or a parameter-free estimator needs no search.
            return np.array([float(self.param_space[n][0]) for n in self._param_names])
        if np.sum(mask) < 10:
            return nan_out

        try:
            if self.hyperparam_optimizer == "random":
                best = self._optimize_random(X_c, y_c)
            else:
                best = self._optimize_optuna(X_c, y_c)
            if best is None:
                return nan_out
            return np.array([float(best[n]) for n in self._param_names])
        except Exception as exc:
            raise RuntimeError(f"{type(self).__name__} hyperparameter tuning failed") from exc

    # ------------------------------------------------------------------
    # 1) Tercile classification
    # ------------------------------------------------------------------
    @staticmethod
    def _classify_pixel(y, index_start, index_end):
        y = np.asarray(y, dtype=float)
        reference = y[index_start:index_end]
        if np.isfinite(reference).sum() < 3:
            return np.full(y.shape, np.nan), np.nan, np.nan
        terciles = np.quantile(reference[np.isfinite(reference)], [1.0/3.0, 2.0/3.0])
        labels = np.digitize(y, terciles, right=True).astype(float)
        labels[~np.isfinite(y)] = np.nan
        return labels, float(terciles[0]), float(terciles[1])

    def compute_class(
        self,
        Predictant: xr.DataArray,
        clim_year_start: int,
        clim_year_end: int,
    ):
        """
        Compute tercile class array and climatological tercile maps.

        Parameters
        ----------
        Predictant      : xr.DataArray (T, Y, X) — continuous predictand
        clim_year_start : int — first year of climatology window
        clim_year_end   : int — last  year of climatology window

        Returns
        -------
        y_class : xr.DataArray (T, Y, X) — integer codes {0=Below, 1=Normal, 2=Above}
        terc33  : xr.DataArray (Y, X)    — 33rd-percentile threshold
        terc67  : xr.DataArray (Y, X)    — 67th-percentile threshold
        """
        index_start, index_end = _climatology_positions(Predictant["T"], clim_year_start, clim_year_end)

        y_class, terc33, terc67 = xr.apply_ufunc(
            self._classify_pixel,
            Predictant,
            input_core_dims=[("T",)],
            kwargs={"index_start": index_start, "index_end": index_end},
            vectorize=True,
            dask="parallelized",
            output_core_dims=[("T",), (), ()],
            output_dtypes=["float", "float", "float"],
        )
        return y_class.transpose("T", "Y", "X"), terc33, terc67

    # ------------------------------------------------------------------
    # 2) Spatial clustering + per-cluster HPO
    # ------------------------------------------------------------------
    @staticmethod
    def _spatial_mode_ignore_nan(v: np.ndarray) -> float:
        """Mode of {0, 1, 2} over a 1-D array, ignoring NaN."""
        v = v[np.isfinite(v)].astype(int)
        if v.size == 0:
            return np.nan
        return float(np.argmax(np.bincount(v, minlength=3)))

    def _build_cluster_map(self, predictand):
        return geographical_cluster_map(predictand, self.n_clusters,
                                   random_state=self.random_state)

    def compute_hyperparameters(
        self,
        predictand: xr.DataArray,
        predictor: xr.DataArray,
        clim_year_start: int,
        clim_year_end: int,
    ):
        """
        Cluster the spatial domain and optimise classifier hyperparameters
        per cluster using the cluster's spatially-aggregated class time series.

        Parameters
        ----------
        predictand      : xr.DataArray (T, Y, X) — continuous predictand
        predictor       : xr.DataArray (T, features) — predictors
        clim_year_start : int
        clim_year_end   : int

        Returns
        -------
        hyper_ds   : xr.Dataset        — one (Y, X) field per hyperparameter,
                                         constant within each cluster.
        cluster_da : xr.DataArray(Y,X) — KMeans cluster labels.
        """
        predictor = predictor.copy()
        predictor, predictand = _aligned_training(predictor, predictand)
        if any(dim in predictor.dims for dim in ("Y", "X")):
            raise ValueError("Classifier cluster tuning requires T,features predictors; aggregate spatial predictors explicitly")

        # Classify predictand into tercile classes
        y_class, _, _ = self.compute_class(predictand, clim_year_start, clim_year_end)

        # Build cluster map
        cluster_da = self._build_cluster_map(predictand)
        _, cluster_da = xr.align(predictand.isel(T=0), cluster_da, join="outer")

        clusters = np.unique(cluster_da.values)
        clusters = clusters[np.isfinite(clusters)]

        names = self._param_names
        hyper_maps = {n: xr.full_like(cluster_da, np.nan, dtype=float) for n in names}

        for c in clusters:
            mask_c = cluster_da == c

            # Representative class label per time step: spatial mode over cluster
            y_stack = y_class.where(mask_c).stack(Z=("Y", "X"))
            y_mode = xr.apply_ufunc(
                self._spatial_mode_ignore_nan,
                y_stack,
                input_core_dims=[("Z",)],
                vectorize=True,
                dask="parallelized",
                output_dtypes=["float"],
            ).dropna("T")

            if y_mode.sizes.get("T", 0) == 0:
                continue

            X_c  = predictor.sel(T=y_mode["T"])
            best = self._optimize_single_cluster(y_mode.values, X_c.values)

            for i, n in enumerate(names):
                hyper_maps[n] = hyper_maps[n].where(~mask_c, other=float(best[i]))

        hyper_ds = xr.Dataset(hyper_maps)
        _report_tuned("_WAS_NonNeural_Classifier_Base", getattr(self, "verbose", True), cluster_da=cluster_da, **{v: hyper_ds[v] for v in hyper_ds.data_vars})
        return hyper_ds, cluster_da

    # ------------------------------------------------------------------
    # 3) Per-pixel fit + predict_proba
    # ------------------------------------------------------------------
    def fit_predict(self, x, y, x_test, *param_values):
        if len(param_values) != len(self._param_names):
            raise ValueError("Missing classifier hyperparameter values")
        params = dict(zip(self._param_names, param_values))
        if any(not np.isfinite(value) for value in params.values()):
            return np.full(3, np.nan)
        estimator = self._make_estimator(**self._cast_params(params))
        return _classifier_probabilities(estimator, x, y, x_test)[0]

    # ------------------------------------------------------------------
    # 4) Parallel spatial classification (hindcast)
    # ------------------------------------------------------------------
    def compute_model(self, X_train, y_class, X_test, hyperparams):
        """Return native probabilities for every supplied test date."""
        return _spatial_classifier(self, X_train, y_class, X_test,
                                  tuple(hyperparams[n] for n in self._param_names))

    # ------------------------------------------------------------------
    # 5) End-to-end single-year forecast
    # ------------------------------------------------------------------
    def forecast(self, Predictant, clim_year_start, clim_year_end, Predictor,
                 Predictor_for_year, hyperparams):
        """Fit historical classes and preserve actual dates of all future rows."""
        y_class, _, _ = self.compute_class(Predictant, clim_year_start, clim_year_end)
        return self.compute_model(Predictor, y_class, _forecast_times(Predictor_for_year), hyperparams)


# ======================================================================
#  Concrete classifiers
# ======================================================================

class WAS_KNN_Classifier_Model(_WAS_NonNeural_Classifier_Base):
    """
    K-Nearest-Neighbours tercile classifier (auto-scaled with StandardScaler).

    Hyperparameters tuned
    ---------------------
    n_neighbors : int   in [2, 30]
    p           : int   in [1, 2]   (Minkowski order; 1=Manhattan, 2=Euclidean)

    Notes
    -----
    - Uses ``weights='distance'`` so closer neighbours dominate.
    - Wrapped in StandardScaler: KNN is distance-sensitive and scaling is mandatory.
    - ``predict_proba`` returns soft votes (weighted class proportion among k
      neighbours). Their reliability must be evaluated on held-out years.
    """

    def _default_param_space(self) -> dict:
        return {
            "n_neighbors": (2, 30, True,  False),
            "p":           (1,  2, True,  False),
        }

    def _make_estimator(self, **params):
        from sklearn.neighbors import KNeighborsClassifier
        from sklearn.preprocessing import StandardScaler
        from sklearn.pipeline import make_pipeline

        return make_pipeline(
            StandardScaler(),
            KNeighborsClassifier(
                weights="distance",
                n_jobs=1,
                **params,
            ),
        )


class WAS_SVC_Classifier_Model(_WAS_NonNeural_Classifier_Base):
    """
    Support Vector Classifier (RBF kernel) for tercile classification.

    ``probability=True`` enables ``predict_proba`` via Platt scaling (logistic
    regression fit on SVM decision values).  Wrapped in StandardScaler.

    Hyperparameters tuned
    ---------------------
    C     : float  in [1e-2, 1e3]  (log scale) — misclassification penalty
    gamma : float  in [1e-4, 1e1]  (log scale) — RBF kernel bandwidth

    Notes
    -----
    - ``class_weight='balanced'`` handles the roughly uniform, but occasionally
      skewed, class distribution in tercile data.
    - Libsvm adds internal five-fold Platt calibration. This cost remains even
      with outer residual calibration; these are separate procedures.
    """

    def _default_param_space(self) -> dict:
        return {
            "C":     (1e-2, 1e3, False, True),
            "gamma": (1e-4, 1e1, False, True),
        }

    def _make_estimator(self, **params):
        from sklearn.svm import SVC
        from sklearn.preprocessing import StandardScaler
        from sklearn.pipeline import make_pipeline

        return make_pipeline(
            StandardScaler(),
            SVC(
                kernel="rbf",
                probability=True,           # required for predict_proba
                class_weight="balanced",
                random_state=self.random_state,
                **params,
            ),
        )


class WAS_RandomForest_Classifier_Model(_WAS_NonNeural_Classifier_Base):
    """
    Random Forest tercile classifier.

    Hyperparameters tuned
    ---------------------
    n_estimators      : int    in [50, 500]
    max_depth         : int    in [2, 30]
    min_samples_split : int    in [2, 20]
    min_samples_leaf  : int    in [1, 20]
    max_features      : float  in [0.1, 1.0]  (fraction of features per split)

    Notes
    -----
    - ``class_weight='balanced_subsample'`` reweights classes per bootstrap
      sample, handling mild imbalance common in tercile label distributions.
    - ``predict_proba`` averages class fractions across trees — naturally
      not guaranteed calibrated; evaluate reliability on held-out years.
    - No feature scaling needed (tree-based, invariant to monotone transforms).
    """

    def _default_param_space(self) -> dict:
        return {
            "n_estimators":      (50,  500, True,  False),
            "max_depth":         (2,   30,  True,  False),
            "min_samples_split": (2,   20,  True,  False),
            "min_samples_leaf":  (1,   20,  True,  False),
            "max_features":      (0.1, 1.0, False, False),
        }

    def _make_estimator(self, **params):
        from sklearn.ensemble import RandomForestClassifier

        return RandomForestClassifier(
            class_weight="balanced_subsample",
            random_state=self.random_state,
            n_jobs=1,
            **params,
        )


class WAS_XGBoost_Classifier_Model(_WAS_NonNeural_Classifier_Base):
    """
    XGBoost multi-class tercile classifier.  Requires ``xgboost`` package.

    Hyperparameters tuned
    ---------------------
    n_estimators     : int    in [50, 600]
    max_depth        : int    in [2, 12]
    learning_rate    : float  in [1e-3, 0.3]    (log scale)
    subsample        : float  in [0.5, 1.0]
    colsample_bytree : float  in [0.5, 1.0]
    min_child_weight : int    in [1, 10]
    reg_alpha        : float  in [1e-8, 10.0]   (log scale)
    reg_lambda       : float  in [1e-8, 10.0]   (log scale)
    gamma            : float  in [1e-8, 5.0]    (log scale)

    Notes
    -----
    - ``objective='multi:softprob'`` makes XGBoost return normalised class
      probabilities, identical in shape to sklearn's ``predict_proba``.
    - ``eval_metric='mlogloss'`` matches the default optimisation scorer.
    - The training classes select binary or multiclass loss; outputs map back to three terciles.
    - ``use_label_encoder=False`` suppresses deprecation warnings in older
      XGBoost versions.
    """

    def _default_param_space(self) -> dict:
        return {
            "n_estimators":      (50,   600,  True,  False),
            "max_depth":         (2,    12,   True,  False),
            "learning_rate":     (1e-3, 0.3,  False, True),
            "subsample":         (0.5,  1.0,  False, False),
            "colsample_bytree":  (0.5,  1.0,  False, False),
            "min_child_weight":  (1,    10,   True,  False),
            "reg_alpha":         (1e-8, 10.0, False, True),
            "reg_lambda":        (1e-8, 10.0, False, True),
            "gamma":             (1e-8, 5.0,  False, True),
        }

    def _make_estimator(self, **params):
        from xgboost import XGBClassifier
        # XGBoost chooses binary/multiclass loss from encoded training classes.
        return XGBClassifier(tree_method="hist", random_state=self.random_state,
                             n_jobs=1, verbosity=0, **params)


class WAS_ExtraTrees_Classifier_Model(_WAS_NonNeural_Classifier_Base):
    """
    Extremely Randomized Trees (Extra-Trees) tercile classifier.

    Extra-Trees differ from Random Forest in two ways:
    - Split thresholds are drawn **randomly** (not optimised) at each node,
      which drastically reduces variance at the cost of a small bias increase.
    - The full training set is used for every tree (no bootstrap sampling),
      so the out-of-bag estimate is not available.

    These properties make Extra-Trees faster to train than Random Forest and
    often comparably accurate on noisy climate data where the optimal split
    threshold is hard to determine from short records (~30 years).

    Hyperparameters tuned
    ---------------------
    n_estimators      : int    in [50, 500]
    max_depth         : int    in [2, 30]
    min_samples_split : int    in [2, 20]
    min_samples_leaf  : int    in [1, 20]
    max_features      : float  in [0.1, 1.0]  (fraction of features per split)

    Notes
    -----
    - ``class_weight='balanced_subsample'`` reweights classes at each tree,
      handling mild imbalance common in tercile label distributions.
    - No feature scaling needed (tree-based, invariant to monotone transforms).
    - ``predict_proba`` averages class fractions across trees — naturally
      not guaranteed calibrated; evaluate reliability on held-out years.
    """

    def _default_param_space(self) -> dict:
        return {
            "n_estimators":      (50,  500, True,  False),
            "max_depth":         (2,   30,  True,  False),
            "min_samples_split": (2,   20,  True,  False),
            "min_samples_leaf":  (1,   20,  True,  False),
            "max_features":      (0.1, 1.0, False, False),
        }

    def _make_estimator(self, **params):
        from sklearn.ensemble import ExtraTreesClassifier

        return ExtraTreesClassifier(
            class_weight="balanced_subsample",
            random_state=self.random_state,
            n_jobs=1,
            **params,
        )


class WAS_GradientBoosting_Classifier_Model(_WAS_NonNeural_Classifier_Base):
    """
    Gradient Boosting tercile classifier (sklearn sequential trees).

    Builds an additive ensemble of shallow decision trees by minimising
    cross-entropy (``loss='log_loss'``) stage by stage.  Each tree corrects
    the residual pseudo-errors of the previous ensemble, which aggressively
    reduces bias — a key advantage over bagging methods when the training
    record is short (~30 years).

    Hyperparameters tuned
    ---------------------
    n_estimators     : int    in [50, 500]
    learning_rate    : float  in [1e-3, 0.3]  (log scale) — shrinkage per stage
    max_depth        : int    in [2, 8]        — shallow trees control variance
    subsample        : float  in [0.5, 1.0]   — stochastic gradient boosting
    min_samples_leaf : int    in [1, 20]

    Notes
    -----
    - ``loss='log_loss'`` is the multi-class cross-entropy, directly targeting
      the calibration of tercile probabilities.
    - Stochastic subsampling (``subsample < 1``) acts as additional
      regularisation and is especially helpful for ~30-sample records.
    - No feature scaling needed (tree-based).
    """

    def _default_param_space(self) -> dict:
        return {
            "n_estimators":      (50,  500, True,  False),
            "learning_rate":     (1e-3, 0.3, False, True),
            "max_depth":         (2,   8,   True,  False),
            "subsample":         (0.5, 1.0, False, False),
            "min_samples_leaf":  (1,   20,  True,  False),
        }

    def _make_estimator(self, **params):
        from sklearn.ensemble import GradientBoostingClassifier

        return GradientBoostingClassifier(
            loss="log_loss",
            random_state=self.random_state,
            **params,
        )


class WAS_AdaBoost_Classifier_Model(_WAS_NonNeural_Classifier_Base):
    """
    Adaptive Boosting (AdaBoost) tercile classifier.

    Combines a sequence of weighted decision-tree stumps (``max_depth=1``),
    up-weighting misclassified samples at each stage.  AdaBoost uses a
    different sample-reweighting scheme than Gradient Boosting, making it
    complementary in an ensemble-of-models context.

    Hyperparameters tuned
    ---------------------
    n_estimators  : int    in [50, 500]
    learning_rate : float  in [1e-3, 1.0]  (log scale) — contribution per stage

    Notes
    -----
    - Base estimator is a ``DecisionTreeClassifier(max_depth=1)`` (stump),
      which is the standard and most regularised AdaBoost configuration.
    - AdaBoost is sensitive to noisy labels; with tercile data derived from
      ~30 years, mild label noise is expected — keep ``n_estimators`` moderate.
    - Multi-class support is handled natively by sklearn's AdaBoostClassifier
      via the internal one-vs-one reduction.
    """

    def _default_param_space(self) -> dict:
        return {
            "n_estimators":  (50,  500, True,  False),
            "learning_rate": (1e-3, 1.0, False, True),
        }

    def _make_estimator(self, **params):
        from sklearn.ensemble import AdaBoostClassifier
        from sklearn.tree import DecisionTreeClassifier

        return AdaBoostClassifier(
            estimator=DecisionTreeClassifier(max_depth=1),
            random_state=self.random_state,
            **params,
        )


class WAS_LightGBM_Classifier_Model(_WAS_NonNeural_Classifier_Base):
    """
    LightGBM leaf-wise boosting tercile classifier.

    Requires the ``lightgbm`` package (``pip install lightgbm``).

    LightGBM grows trees **leaf-wise** (best-leaf-first) rather than
    depth-wise like sklearn GBC.  This can find deeper structures in a single
    tree with fewer total leaves, often outperforming XGBoost on datasets
    with a moderate number of features and short records.

    Hyperparameters tuned
    ---------------------
    n_estimators       : int    in [50, 600]
    num_leaves         : int    in [15, 127]   — controls model complexity
    max_depth          : int    in [3, 12]
    learning_rate      : float  in [1e-3, 0.3]  (log scale)
    subsample          : float  in [0.5, 1.0]   — row subsampling per tree
    colsample_bytree   : float  in [0.5, 1.0]   — feature subsampling per tree
    reg_alpha          : float  in [1e-8, 10.0] (log scale) — L1 regularisation
    reg_lambda         : float  in [1e-8, 10.0] (log scale) — L2 regularisation
    min_child_samples  : int    in [5, 50]      — minimum samples per leaf

    Notes
    -----
    - The objective follows the training classes; soft-max probabilities require independent reliability assessment, not automatic calibration.
      The output gives
      probabilities equivalent to sklearn's ``predict_proba``.
    - ``class_weight='balanced'`` handles tercile imbalance.
    - ``verbosity=-1`` suppresses LightGBM's verbose stdout output.
    """

    def _default_param_space(self) -> dict:
        return {
            "n_estimators":      (50,   600,  True,  False),
            "num_leaves":        (15,   127,  True,  False),
            "max_depth":         (3,    12,   True,  False),
            "learning_rate":     (1e-3, 0.3,  False, True),
            "subsample":         (0.5,  1.0,  False, False),
            "colsample_bytree":  (0.5,  1.0,  False, False),
            "reg_alpha":         (1e-8, 10.0, False, True),
            "reg_lambda":        (1e-8, 10.0, False, True),
            "min_child_samples": (5,    50,   True,  False),
        }

    def _make_estimator(self, **params):
        from lightgbm import LGBMClassifier

        return LGBMClassifier(
            class_weight="balanced",
            random_state=self.random_state,
            n_jobs=1,
            verbosity=-1,
            subsample_freq=1,
            **params,
        )


class WAS_NaiveBayes_Classifier_Model(_WAS_NonNeural_Classifier_Base):
    """
    Gaussian Naive Bayes tercile classifier.

    Fits a per-class multivariate Gaussian with diagonal covariance (feature
    independence assumption) using maximum likelihood.  Despite the strong
    independence assumption, GNB is competitive on short records because:
    - It has **zero hyperparameters** to overfit during search.
    - The per-class Gaussian naturally yields calibrated posterior
      probabilities via Bayes' theorem.
    - It is robust to correlated features when the correlation structure
      is similar across tercile classes.

    The single tunable parameter ``var_smoothing`` adds a fraction of the
    largest per-feature variance to all variances, preventing division-by-zero
    for nearly constant predictors.

    Hyperparameters tuned
    ---------------------
    var_smoothing : float  in [1e-12, 1e-1]  (log scale)

    Notes
    -----
    - Automatically scaled via StandardScaler — while GNB is scale-invariant
      in theory, scaling ensures ``var_smoothing`` operates in a consistent
      numerical range across predictors with different physical units.
    - Best suited as a fast baseline or ensemble member; also useful when the
      number of valid training years is very small (< 20).
    """

    def _default_param_space(self) -> dict:
        return {
            "var_smoothing": (1e-12, 1e-1, False, True),
        }

    def _make_estimator(self, **params):
        from sklearn.naive_bayes import GaussianNB
        from sklearn.preprocessing import StandardScaler
        from sklearn.pipeline import make_pipeline

        return make_pipeline(
            StandardScaler(),
            GaussianNB(**params),
        )


class WAS_LinearDiscriminant_Classifier_Model(_WAS_NonNeural_Classifier_Base):
    """
    Linear Discriminant Analysis (LDA) tercile classifier with automatic
    Ledoit-Wolf shrinkage.

    LDA finds the linear combination of features that maximises the ratio of
    between-class to within-class scatter.  With ``solver='lsqr'`` and
    ``shrinkage='auto'``, sklearn applies the Ledoit-Wolf analytical estimator
    for the regularised pooled covariance matrix — specifically designed for
    small-N / moderate-p settings (exactly the ~30-year seasonal forecast
    regime).

    This model has **no free hyperparameters** to tune: Ledoit-Wolf determines
    the shrinkage coefficient analytically.  The HPO step is therefore skipped
    (a fixed placeholder value is broadcast), and fitting is instant.

    Notes
    -----
    - ``solver='lsqr'`` supports ``shrinkage`` and is numerically stable for
      rank-deficient covariance matrices arising from correlated predictors.
    - LDA assumes equal class covariances; with tercile data this is generally
      a reasonable approximation unless the climate signal is highly non-linear.
    - Gaussian discriminant posterior probabilities need an independent
      reliability assessment; the model assumptions do not ensure calibration.
    - Auto-scaled via StandardScaler for numerical stability.
    """

    def _default_param_space(self) -> dict:
        # LDA with auto-shrinkage has no free hyperparameters.
        # A single fixed placeholder is used so the base-class HPO infrastructure
        # stays intact and the broadcast hyperparams Dataset is well-formed.
        return {
            "_placeholder": (0.0, 1.0, False, False),
        }

    def _make_estimator(self, **params):
        from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
        from sklearn.preprocessing import StandardScaler
        from sklearn.pipeline import make_pipeline

        # Drop placeholder key — LDA takes no tunable params here
        return make_pipeline(
            StandardScaler(),
            LinearDiscriminantAnalysis(solver="lsqr", shrinkage="auto"),
        )


class WAS_QuadraticDiscriminant_Classifier_Model(_WAS_NonNeural_Classifier_Base):
    """
    Quadratic Discriminant Analysis (QDA) tercile classifier.

    QDA relaxes the LDA assumption of a shared covariance matrix by fitting a
    separate covariance per class.  This allows curved (quadratic) decision
    boundaries that can capture class-specific variability in the seasonal
    climate signal.

    With only ~30 training years the per-class covariance is estimated from
    ~10 samples — ill-conditioned without regularisation.  The ``reg_param``
    hyperparameter adds a fraction of the identity matrix to each class
    covariance, interpolating between full QDA (``reg_param=0``) and
    identity covariance for every class (``reg_param=1``). This is not the
    pooled sample covariance estimated by ordinary LDA.

    Hyperparameters tuned
    ---------------------
    reg_param : float  in [0.0, 1.0]  (linear scale) — covariance regularisation

    Notes
    -----
    - Auto-scaled via StandardScaler for numerical stability in covariance
      estimation.
    - Near ``reg_param=1`` the class covariances approach the same identity
      matrix. Tuning minimizes validation log-loss on the cluster class series.
    - QDA is more expressive than LDA but needs more data per class to be
      stable; ``reg_param`` is the primary safeguard.
    """

    def _default_param_space(self) -> dict:
        return {
            "reg_param": (0.0, 1.0, False, False),
        }

    def _make_estimator(self, **params):
        from sklearn.discriminant_analysis import QuadraticDiscriminantAnalysis
        from sklearn.preprocessing import StandardScaler
        from sklearn.pipeline import make_pipeline

        return make_pipeline(
            StandardScaler(),
            QuadraticDiscriminantAnalysis(**params),
        )

