from wass2s.was_scientific_utils import apply_physical_support, climate_cluster_map, align_predictor_target
"""Linear regression models for spatial seasonal forecasting.

Each class implements a ``compute_model`` / ``compute_prob`` / ``forecast``
interface compatible with :class:`~wass2s.was_cross_validate.WAS_Cross_Validator`
and :class:`~wass2s.was_pcr.WAS_PCR`.

Classes
-------
WAS_LinearRegression_Model
    Pixel-wise ordinary least-squares regression with parametric or
    non-parametric tercile probabilities.
WAS_Ridge_Model
    Ridge (L2-regularised) regression with Optuna / randomized
    hyperparameter search.
WAS_Lasso_Model
    Lasso (L1-regularised) regression with Optuna / randomized
    hyperparameter search.
WAS_LassoLars_Model
    LassoLars regression.
WAS_ElasticNet_Model
    Elastic-Net (L1+L2) regression.
"""
################################### Modules ###################################
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.compose import TransformedTargetRegressor
from sklearn.base import clone
from sklearn import linear_model
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import GridSearchCV
from sklearn.model_selection import RandomizedSearchCV, cross_val_score
import statsmodels.api as sm
from statsmodels.stats.outliers_influence import variance_inflation_factor as VIF
from statsmodels.stats.anova import anova_lm
import xarray as xr
from .was_probability_reference import climatological_terciles
import numpy as np
import pandas as pd
from scipy import stats
from scipy.stats import norm, lognorm, gamma, expon, weibull_min, t, poisson, nbinom, loguniform
import scipy.signal as sig
from scipy.interpolate import CubicSpline
from scipy.optimize import fsolve
from scipy.special import gamma as gamma_function
from sklearn.cluster import KMeans
import xeofs as xe
from multiprocessing import cpu_count
from dask.distributed import Client
import dask.array as da


def _safe_close_client(client):
    """Close a Dask client (and its LocalCluster) with a bounded timeout,
    swallowing teardown errors. The compute result is always materialised
    (`.compute()`) before this is called, so a slow/failed nanny shutdown must
    never abort the run: without this, `client.close()` can hang on the nanny
    join and raise TimeoutError (exit code 1 under Slurm)."""
    if client is None:
        return
    import contextlib
    cluster = getattr(client, "cluster", None)
    with contextlib.suppress(Exception):
        client.close(timeout=10)
    if cluster is not None:
        with contextlib.suppress(Exception):
            cluster.close(timeout=10)


# ---------------------------------------------------------------------------
# Shared Dask client: created once per process and reused across ALL folds and
# linear-model calls, instead of spinning up (and tearing down) a fresh
# LocalCluster on every compute_model — which caused nanny-join hangs and, when
# several seasonal jobs ran at once, worker oversubscription (stalled folds).
# ---------------------------------------------------------------------------
_SHARED_CLIENT = None


def _get_compute_client(nb_cores):
    """Return a process-wide Dask client, creating it lazily on first use and
    reusing it afterwards. Returns None if a distributed client cannot be
    created, in which case ``.compute()`` transparently falls back to dask's
    default (threaded) scheduler — still correct, just without process workers.
    The client is closed once, at interpreter exit, via a bounded atexit hook."""
    global _SHARED_CLIENT
    try:
        from dask.distributed import Client, LocalCluster, get_client
        # Reuse a client already active in this process, if any.
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
        n = max(1, int(nb_cores))
        cluster = LocalCluster(
            n_workers=n,
            threads_per_worker=1,
            processes=True,
            dashboard_address=None,
            memory_limit=0,
        )
        _SHARED_CLIENT = Client(cluster)
        import atexit
        atexit.register(_safe_close_client, _SHARED_CLIENT)
        return _SHARED_CLIENT
    except Exception:
        return None
import optuna
import warnings
from wass2s.utils import *


def _aligned_training(predictor, predictand):
    """Require the caller to provide matching valid-time coordinates; never relabel."""
    for data in (predictor, predictand):
        index = data.get_index("T")
        if index.has_duplicates or not index.is_monotonic_increasing:
            raise ValueError("Training T coordinates must be unique and sorted")
    predictor = align_predictor_target(predictor, predictand)
    return xr.align(predictor, predictand, join="exact", copy=False)


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
    estimator.fit(x[good], y[good])
    prediction = float(np.asarray(estimator.predict(xt)).reshape(-1)[0])
    error = np.nan if y_test is None else float(np.asarray(y_test).reshape(-1)[0]) - prediction
    return np.array([error, prediction])

def _scaled_linear(estimator):
    """Learn predictor and target scales on each actual fitting partition."""
    return TransformedTargetRegressor(
        regressor=Pipeline([("scale", StandardScaler()), ("model", estimator)]),
        transformer=StandardScaler())


def _linear_search(owner, X, y, kind, alpha_range, l1_ratio_range=None):
    """One train-only preprocessing contract for grid, random and Bayesian tuning."""
    from scipy.stats import loguniform
    factory = getattr(linear_model, kind)
    base = factory() if kind == "Ridge" else factory(max_iter=10000)
    from .was_tuning_support import FoldLocalTargetRegressor, fold_local_neg_mse
    regional = np.asarray(y).ndim == 2
    def make_estimator(parameters=None):
        fitted = _scaled_linear(clone(base).set_params(**(parameters or {})))
        return FoldLocalTargetRegressor(fitted) if regional else fitted
    prefix = "regressor__model__"
    if regional:
        prefix = "regressor__" + prefix
    scoring = fold_local_neg_mse if regional else "neg_mean_squared_error"
    alphas = np.asarray(alpha_range, float)
    if alphas.size == 0 or np.any(~np.isfinite(alphas)) or np.any(alphas <= 0):
        raise ValueError("alpha_range must contain strictly positive values")
    if l1_ratio_range is not None:
        ratios = np.asarray(l1_ratio_range, float)
        if ratios.size == 0 or np.any(~np.isfinite(ratios)) or np.any((ratios < 0) | (ratios > 1)):
            raise ValueError("l1_ratio_range must contain finite values in [0, 1]")
    # Every alpha produces the same intercept for a constant target. Avoid
    # both needless searches and numerical solver warnings in this case.
    if np.all(np.nanmax(y, axis=0) == np.nanmin(y, axis=0)):
        return (float(alphas[0]), float(ratios[0])) if l1_ratio_range is not None else float(alphas[0])
    cv = owner.inner_cv
    if owner.hyperparam_optimizer == "bayesian":
        def objective(trial):
            params = {"alpha": trial.suggest_float("alpha", float(alphas.min()), float(alphas.max()), log=True)}
            if l1_ratio_range is not None:
                params["l1_ratio"] = trial.suggest_float("l1_ratio", float(np.min(l1_ratio_range)), float(np.max(l1_ratio_range)))
            model = make_estimator(params)
            scores = cross_val_score(model, X, y, cv=cv, scoring=scoring, n_jobs=1, error_score="raise")
            return float(scores.mean())
        study = optuna.create_study(direction="maximize", sampler=optuna.samplers.TPESampler(seed=owner.random_state))
        study.optimize(objective, n_trials=owner.n_trials, show_progress_bar=False)
        best = study.best_params
    else:
        grid = {prefix + "alpha": alphas}
        if l1_ratio_range is not None:
            grid[prefix + "l1_ratio"] = list(l1_ratio_range)
        if owner.hyperparam_optimizer == "random":
            grid[prefix + "alpha"] = loguniform(alphas.min(), alphas.max()) if alphas.max() > alphas.min() else list(alphas)
            search = RandomizedSearchCV(make_estimator(), grid, n_iter=owner.n_iter,
                cv=cv, scoring=scoring, n_jobs=1,
                random_state=owner.random_state, error_score="raise")
        else:
            search = GridSearchCV(make_estimator(), grid, cv=cv,
                scoring=scoring, n_jobs=1, error_score="raise")
        search.fit(X, y)
        best = {key.removeprefix(prefix): value for key, value in search.best_params_.items()}
    return (best["alpha"], best["l1_ratio"]) if l1_ratio_range is not None else best["alpha"]


def _linear_hyperparameters(model, predictand, predictor, kind):
    """Select cell parameters using raw targets and fit-local transformations.

    Regional tuning groups depend only on geographical coordinates. For each
    inner split the cell means/scales and regional proxy are learned again on
    its training rows, preventing validation outcomes from setting the weights.
    The actual prediction model remains fitted separately at every grid cell.
    """
    from .was_tuning_support import geographical_cluster_map
    if "M" in predictand.dims:
        if predictand.sizes["M"] != 1:
            raise ValueError("Predictand member dimension M must be singleton")
        predictand = predictand.isel(M=0, drop=True)
    predictor, predictand = _aligned_training(predictor, predictand)
    predictand = predictand.transpose("T", "Y", "X")
    if model.mode not in {"grid", "cluster"}:
        raise ValueError("mode must be 'grid' or 'cluster'")
    elastic = kind == "ElasticNet"
    if model.mode == "grid":
        cores = max(1, int(model.nb_cores))
        target = predictand.chunk({"T": -1,
            "Y": max(1, int(np.ceil(predictand.sizes["Y"]/cores))),
            "X": max(1, int(np.ceil(predictand.sizes["X"]/cores)))})
        parameters = xr.apply_ufunc(model._optimize_single_cell, target, predictor,
            input_core_dims=[("T",), ("T", "features")],
            output_core_dims=[("params",)] if elastic else [()],
            vectorize=True, dask="parallelized", output_dtypes=[float],
            dask_gufunc_kwargs={"allow_rechunk": True,
                **({"output_sizes": {"params": 2}} if elastic else {})})
        # Reuse an existing distributed client; do not start a process cluster
        # for each tiny hyperparameter search.
        try:
            from dask.distributed import get_client
            get_client()
        except ValueError:
            from threadpoolctl import threadpool_limits
            with threadpool_limits(limits=1):
                parameters = parameters.compute(scheduler="threads", num_workers=cores)
        else:
            parameters = parameters.compute()
        alpha = parameters.isel(params=0, drop=True) if elastic else parameters
        ratio = parameters.isel(params=1, drop=True) if elastic else None
        clusters = xr.where(np.isfinite(alpha), 1., np.nan)
        clusters.attrs["grouping"] = "independent grid cells"
    else:
        if any(dim in predictor.dims for dim in ("Y", "X")):
            raise ValueError("Spatial predictors require mode='grid' or explicit regional aggregation")
        clusters = geographical_cluster_map(predictand, model.n_clusters, model.random_state)
        alpha = xr.full_like(clusters, np.nan)
        ratio = xr.full_like(clusters, np.nan) if elastic else None
        x = np.asarray(predictor.transpose("T", "features"), float)
        cells = np.asarray(predictand, float).reshape(predictand.sizes["T"], -1)
        labels = np.asarray(clusters).reshape(-1)
        for group in np.unique(labels[np.isfinite(labels)]):
            y = cells[:, labels == group]
            usable = np.isfinite(y).any(axis=1) & np.isfinite(x).all(axis=1)
            if usable.sum() < 10:
                continue
            y = y[usable]
            y = y[:, np.isfinite(y).any(axis=0)]
            selected = _linear_search(model, x[usable], y, kind, model.alpha_range,
                model.l1_ratio_range if elastic else None)
            alpha = alpha.where(clusters != group, selected[0] if elastic else selected)
            if elastic:
                ratio = ratio.where(clusters != group, selected[1])
    alpha.attrs.update(tuning_preprocessing="fitted separately inside each inner search split",
                       tuning_grouping=clusters.attrs["grouping"])
    _report_tuned(type(model).__name__, getattr(model, "verbose", True),
                  cluster_da=clusters, alpha=alpha, **({"l1_ratio":ratio} if elastic else {}))
    return (alpha, ratio, clusters) if elastic else (alpha, clusters)


def _regression_tercile_probabilities(best_guess, error_variance, T1, T2,
                                    dist_code, dof, nonneg=False):
    """Moment-matched predictive CDFs for continuous linear/ML regressors.

    ``best_guess`` is the bias-corrected predictive mean and ``error_variance``
    is the centred residual variance. With explicit nonnegative support an
    exponential has location zero and scale equal to its mean; unlike a
    shifted exponential it cannot additionally match an arbitrary variance.
    Poisson similarly imposes variance = mean; a negative binomial requires
    overdispersion. Invalid family moments are missing, never silently replaced
    by another family. Zero residual variance is the exact point distribution.
    """
    from scipy.optimize import brentq
    from scipy.special import gammaln

    means = np.asarray(best_guess, float).reshape(-1)
    variance = float(np.asarray(error_variance))
    out = np.full((3, means.size), np.nan)
    if (not np.isfinite(variance) or variance < 0 or not np.isfinite(T1)
            or not np.isfinite(T2) or T1 > T2 or not np.isfinite(dist_code)):
        return out
    code = int(dist_code)
    if code != dist_code or code not in range(1, 9):
        raise ValueError("dist_code must be an integer in 1..8")
    good = np.isfinite(means)
    if nonneg:
        good &= means >= 0
    if variance == 0:
        out[:, good] = np.vstack((means[good] <= T1,
            (means[good] > T1) & (means[good] <= T2), means[good] > T2))
        return out
    for i in np.flatnonzero(good):
        mean = means[i]
        try:
            if code == 1:
                cdf = lambda q: norm.cdf(q, loc=mean, scale=np.sqrt(variance))
            elif code == 2:
                if mean <= 0:
                    continue
                sigma2 = np.log1p(variance / (mean * mean))
                cdf = lambda q: lognorm.cdf(q, s=np.sqrt(sigma2), scale=np.exp(np.log(mean)-sigma2/2))
            elif code == 3:
                scale = mean if nonneg else np.sqrt(variance)
                if scale <= 0:
                    continue
                loc = 0.0 if nonneg else mean-scale
                cdf = lambda q: expon.cdf(q, loc=loc, scale=scale)
            elif code == 4:
                if mean <= 0:
                    continue
                cdf = lambda q: gamma.cdf(q, a=mean*mean/variance, scale=variance/mean)
            elif code == 5:
                if mean <= 0:
                    continue
                # Solve in log space: stable even for highly skewed rainfall.
                target = np.log1p(variance / (mean*mean))
                equation = lambda k: gammaln(1+2/k)-2*gammaln(1+1/k)-target
                shape = brentq(equation, 0.01, 1.e7, maxiter=250)
                scale = np.exp(np.log(mean)-gammaln(1+1/shape))
                cdf = lambda q: weibull_min.cdf(q, c=shape, scale=scale)
            elif code == 6:
                if not np.isfinite(dof) or dof <= 2:
                    continue
                scale = np.sqrt(variance*(float(dof)-2)/float(dof))
                cdf = lambda q: t.cdf(q, df=dof, loc=mean, scale=scale)
            elif code == 7:
                if mean < 0:
                    continue
                cdf = lambda q: poisson.cdf(q, mu=mean)
            else:
                if mean <= 0 or variance <= mean:
                    continue
                cdf = lambda q: nbinom.cdf(q, n=mean*mean/(variance-mean), p=mean/variance)
            c1, c2 = float(cdf(T1)), float(cdf(T2))
            p = np.clip([c1, c2-c1, 1-c2], 0., 1.)
            if np.isfinite(p).all() and p.sum() > 0:
                out[:, i] = p/p.sum()
        except (ValueError, FloatingPointError, OverflowError, RuntimeError):
            continue
    return out


def _residual_probabilities(forecast, residuals, t1, t2, nonneg=False):
    """Empirical predictive CDF with a distinct residual-sample axis."""
    forecast, residuals = np.asarray(forecast, float), np.asarray(residuals, float)
    output = np.full((3, forecast.size), np.nan)
    residuals = residuals[np.isfinite(residuals)]
    if residuals.size < 3 or not np.isfinite(t1) or not np.isfinite(t2) or t1 > t2:
        return output
    good = np.isfinite(forecast)
    samples = forecast[good, None] + residuals[None, :]
    if nonneg:
        samples = np.maximum(samples, 0.)
    below = np.mean(samples <= t1, axis=1)
    below_normal = np.mean(samples <= t2, axis=1)
    output[:, good] = np.vstack((below, below_normal-below, 1-below_normal))
    return output


def _operational_forecast(model, Predictant, clim_year_start, clim_year_end,
                          Predictor, hindcast_det, Predictor_for_year, parameters=(),
                          best_code_da=None, best_shape_da=None, best_loc_da=None,
                          best_scale_da=None, raw_target=False):
    """Fit available observations and calibrate future predictions from OOF errors.

    ``hindcast_det`` must be an out-of-fold hindcast in the target's physical
    units, independently obtained. This operational calibration must not be used
    to score those same hindcast years; use Cross_Validator (outer or inner) for that.
    Forecast dates are retained exactly as supplied in Predictor_for_year.
    """
    if "M" in Predictant.dims:
        if Predictant.sizes["M"] != 1:
            raise ValueError("Predictant member dimension M must be singleton")
        Predictant = Predictant.isel(M=0, drop=True)
    if "M" in hindcast_det.dims:
        if hindcast_det.sizes["M"] != 1:
            raise ValueError("hindcast_det member dimension M must be singleton")
        hindcast_det = hindcast_det.isel(M=0, drop=True)
    Predictor, Predictant = _aligned_training(Predictor, Predictant)
    if "T" not in Predictor_for_year.dims:
        raise ValueError("Predictor_for_year must include its actual valid-time T coordinate")
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
    residuals = (obs-hindcast).where(np.isfinite(obs) & np.isfinite(hindcast)).rename(T="sample")
    q, ref = climatological_terciles(reference)
    nonnegative = bool(apply_physical_support(xr.DataArray(-1.), Predictant) == 0.)
    if getattr(model, "tercile_reference", "climatology") == "training":
        q, ref = climatological_terciles(Predictant)
        q.attrs["tercile_reference"] = "training"
    t1, t2 = q.isel(quantile=0, drop=True), q.isel(quantile=1, drop=True)
    if model.dist_method == "nonparam":
        probability = xr.apply_ufunc(_residual_probabilities, prediction, residuals, t1, t2,
            input_core_dims=[("T",), ("sample",), (), ()],
            output_core_dims=[("probability", "T")], vectorize=True, dask="parallelized",
            kwargs={"nonneg": nonnegative}, output_dtypes=[float],
            dask_gufunc_kwargs={"output_sizes":{"probability":3}, "allow_rechunk":True})
    elif model.dist_method == "bestfit":
        if any(v is None for v in (best_code_da, best_shape_da, best_loc_da, best_scale_da)):
            raise ValueError("bestfit requires distribution parameters fitted on available training data")
        # The fitted family never redefines the empirical event boundaries.
        variance = residuals.var("sample", skipna=True, ddof=1)
        dof = xr.where(residuals.count("sample") > 3, residuals.count("sample")-1, 3)
        calibrated_mean = apply_physical_support(prediction + residuals.mean("sample"), Predictant)
        # Third-party/legacy kernels retain the six-argument contract. Pass
        # the optional support flag only when their signature declares it.
        import inspect
        kernel = model.calculate_tercile_probabilities_bestfit
        kernel_options = ({"nonneg": nonnegative}
                          if "nonneg" in inspect.signature(kernel).parameters else {})
        probability = xr.apply_ufunc(kernel,
            calibrated_mean, variance, t1, t2, best_code_da, dof,
            input_core_dims=[("T",),(),(),(),(),()], output_core_dims=[("probability","T")],
            kwargs=kernel_options, vectorize=True, dask="parallelized", output_dtypes=[float],
            dask_gufunc_kwargs={"output_sizes":{"probability":3}, "allow_rechunk":True})
    else:
        raise ValueError("dist_method must be 'nonparam' or 'bestfit'")
    probability = probability.assign_coords(probability=["PB","PN","PA"])
    total = probability.sum("probability", skipna=False)
    probability = (probability/total).where((total > 0) & (residuals.count("sample") >= 3)
        & np.isfinite(t1) & np.isfinite(t2) & np.isfinite(prediction))
    probability.attrs.update(q.attrs, probability_calibration="operational_oof",
        residual_source="supplied out-of-fold hindcast", residual_variance_ddof=1,
        minimum_residual_count=3)
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


#### Add Nonexcedance function for all models ##############################################

class WAS_LinearRegression_Model:
    """Ordinary least squares for spatial continuous targets.

    ``compute_model`` returns deterministic predictions on the supplied valid
    times, preserving singleton Y/X/features dimensions. ``compute_prob`` uses
    other outer-fold errors by default; inner/nested calibration requires
    explicitly supplied inner-OOF residuals or the Cross_Validator driver.
    ``forecast`` fits all available training dates and calibrates independent
    future dates from supplied OOF hindcasts. Probabilistic events use empirical
    climatological terciles for both nonparametric and best-fit families.

    ``nb_cores`` controls grid parallelism. ``dist_method`` is ``'nonparam'`` or
    ``'bestfit'`` (the latter requires training-fitted distribution maps).
    Physical support is applied in physical units using target metadata/name;
    temperature predictions retain their sign. Input dates must match exactly.
    """

    def __init__(self, nb_cores=1, dist_method="nonparam"):
        """
        Initializes the WAS_LinearRegression_Model with a specified number of CPU cores.
        
        Parameters
        -----------
        nb_cores : int, optional
            Number of CPU cores to use for parallel computation, by default 1.
        dist_method : str, optional
            Distribution method to compute tercile probabilities, by default "gamma".
        """
        self.nb_cores = nb_cores
        self.dist_method = dist_method
    
    def fit_predict(self, x, y, x_test, y_test=None):
        return _regression_pair(linear_model.LinearRegression(), x, y, x_test, y_test)

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
    def calculate_tercile_probabilities_bestfit(best_guess, error_variance, T1, T2,
                                              dist_code, dof, nonneg=False):
        """Shared predictive CDF; mean and centred residual variance are explicit."""
        return _regression_tercile_probabilities(best_guess, error_variance, T1, T2,
                                                dist_code, dof, nonneg=nonneg)

    @staticmethod
    def calculate_tercile_probabilities_nonparametric(best_guess, error_samples,
                                                    first_tercile, second_tercile,
                                                    nonneg=False):
        """Empirical OOF residual dressing with missing-reference protection."""
        return _residual_probabilities(best_guess, error_samples,
                                       first_tercile, second_tercile, nonneg=nonneg)


    def compute_prob(self, Predictant, clim_year_start, clim_year_end, hindcast_det,
                     best_code_da=None, best_shape_da=None, best_loc_da=None,
                     best_scale_da=None, **probability_options):
        """Calibrate hindcasts with outer residual pools by default.

        The evaluated date is excluded from its own pool; explicit
        ``calibration_splits`` also exclude the corresponding test block and
        buffer. ``probability_calibration='inner'`` (alias ``'nested'``)
        requires per-target inner-OOF ``calibration_residuals``; use
        ``WAS_Cross_Validator`` to construct them. Fixed empirical climatology
        defines all events. Fitted marginal quantiles never replace it.
        """
        from .was_probability_calibration import direct_hindcast_probabilities
        return direct_hindcast_probabilities(
            self, Predictant, clim_year_start, clim_year_end, hindcast_det,
            best_code_da=best_code_da, best_shape_da=best_shape_da,
            best_loc_da=best_loc_da, best_scale_da=best_scale_da,
            **probability_options)

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


class WAS_Ridge_Model:
    """Spatial Ridge regression with independent tuning/calibration controls.

    ``mode='grid'`` selects hyperparameters per cell. ``mode='cluster'`` shares
    them among ``n_clusters`` geographical groups independent of the targets;
    each inner search split learns its own cell scales and regional proxy.
    The final regression is always fitted separately for each grid cell.
    ``alpha_range`` defines candidate regularization strengths; ElasticNet also
    accepts ``l1_ratio_range``. At least ten usable dates are required for HPO.
    Constant targets retain a valid intercept and the first candidate setting.

    Search options are ``'bayesian'`` (``n_trials``), ``'random'`` (``n_iter``),
    or grid search (including historical ``*cv`` aliases). ``inner_cv`` controls
    hyperparameter tuning and is independent of probabilistic inner/outer
    calibration. Predictor and target scales are fitted inside every search
    split. ``nb_cores`` controls parallel grid searches; ``random_state`` seeds
    randomized searches and geographical grouping.

    ``compute_prob`` defaults to other outer-fold residuals; ``'inner'`` and
    ``'nested'`` require explicit inner-OOF pools or the Cross_Validator driver.
    ``forecast`` uses supplied OOF hindcasts to calibrate independent future
    dates and accepts fixed alpha=1.0 by default (ElasticNet l1_ratio=0.5).
    ``dist_method`` is ``'nonparam'`` or ``'bestfit'``; climatological terciles
    stay empirical under either method. Physical support follows the target
    metadata/name, so signed temperatures remain signed.
    """

    def __init__(self, alpha_range=None, n_clusters=5, nb_cores=1, dist_method="nonparam",
                 hyperparam_optimizer="bayesian", n_trials=50, n_iter=50, mode="cluster", inner_cv=3, random_state=42):
        if alpha_range is None:
            alpha_range = np.logspace(-10, 10, 100)
        
        self.alpha_range = alpha_range
        self.n_clusters = n_clusters
        self.nb_cores = nb_cores
        self.dist_method = dist_method
        self.hyperparam_optimizer = hyperparam_optimizer
        self.n_trials = n_trials
        self.n_iter = n_iter
        self.mode = mode  # 'cluster' or 'grid'
        self.inner_cv = inner_cv
        self.random_state = random_state

    # ------------------ Hyperparameter Helpers ------------------

    def _optimize_hyperparameters_optuna(self, X, y, alpha_range):
        return _linear_search(self, X, y, "Ridge", alpha_range)

    def _optimize_hyperparameters_randomized(self, X, y, alpha_range):
        return _linear_search(self, X, y, "Ridge", alpha_range)

    def _optimize_single_cell(self, y_vec, X_mat):
        mask = np.isfinite(y_vec) & np.isfinite(X_mat).all(axis=-1)
        if mask.sum() < 10:
            return np.nan
        result = _linear_search(self, X_mat[mask], y_vec[mask], "Ridge", self.alpha_range)
        return result

    # ------------------ Core Logic Methods ------------------

    def compute_hyperparameters(self, predictand, predictor, clim_year_start, clim_year_end):
        """Tune parameters with independent preprocessing in every inner split.

        Climatology bounds are retained for API compatibility; fitted scales
        depend only on the actual training rows, not on the event climatology.
        ``mode='cluster'`` uses geographical groups and a fold-local normalized
        regional proxy; ``mode='grid'`` selects parameters independently per cell.
        """
        return _linear_hyperparameters(self, predictand, predictor, 'Ridge')

    def fit_predict(self, x, y, x_test, y_test, alpha):
        if not np.isfinite(alpha):
            return np.array([np.nan, np.nan])
        estimator = _scaled_linear(linear_model.Ridge(alpha=float(alpha)))
        return _regression_pair(estimator, x, y, x_test, y_test)

    def compute_model(self, X_train, y_train, X_test, y_test, alpha, **kwargs):
        return _spatial_regression(self, X_train, y_train, X_test, y_test, (alpha,))

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
                return (
                    t.ppf(1.0 / 3.0, df=shape, loc=loc, scale=scale),
                    t.ppf(2.0 / 3.0, df=shape, loc=loc, scale=scale),
                )
            elif code == 7:
                return (
                    poisson.ppf(1.0 / 3.0, mu=shape, loc=loc),
                    poisson.ppf(2.0 / 3.0, mu=shape, loc=loc),
                )
            elif code == 8:
                return (
                    nbinom.ppf(1.0 / 3.0, n=shape, p=scale, loc=loc),
                    nbinom.ppf(2.0 / 3.0, n=shape, p=scale, loc=loc),
                )
        except Exception:
            return np.nan, np.nan
    
        return np.nan, np.nan
        
    @staticmethod
    def weibull_shape_solver(k, M, V):
        """Solver for Weibull shape parameter."""
        if k <= 0:
            return -np.inf
        try:
            g1 = gamma_function(1 + 1/k)
            g2 = gamma_function(1 + 2/k)
            implied_v_over_m_sq = (g2 / (g1**2)) - 1
            observed_v_over_m_sq = V / (M**2)
            return observed_v_over_m_sq - implied_v_over_m_sq
        except ValueError:
            return -np.inf

    @staticmethod
    def calculate_tercile_probabilities_bestfit(best_guess, error_variance, T1, T2,
                                              dist_code, dof, nonneg=False):
        """Shared predictive CDF; mean and centred residual variance are explicit."""
        return _regression_tercile_probabilities(best_guess, error_variance, T1, T2,
                                                dist_code, dof, nonneg=nonneg)

    @staticmethod
    def calculate_tercile_probabilities_nonparametric(best_guess, error_samples,
                                                    first_tercile, second_tercile,
                                                    nonneg=False):
        """Empirical OOF residual dressing with missing-reference protection."""
        return _residual_probabilities(best_guess, error_samples,
                                       first_tercile, second_tercile, nonneg=nonneg)

    def compute_prob(self, Predictant, clim_year_start, clim_year_end, hindcast_det,
                     best_code_da=None, best_shape_da=None, best_loc_da=None,
                     best_scale_da=None, **probability_options):
        """Calibrate hindcasts with outer residual pools by default.

        The evaluated date is excluded from its own pool; explicit
        ``calibration_splits`` also exclude the corresponding test block and
        buffer. ``probability_calibration='inner'`` (alias ``'nested'``)
        requires per-target inner-OOF ``calibration_residuals``; use
        ``WAS_Cross_Validator`` to construct them. Fixed empirical climatology
        defines all events. Fitted marginal quantiles never replace it.
        """
        from .was_probability_calibration import direct_hindcast_probabilities
        return direct_hindcast_probabilities(
            self, Predictant, clim_year_start, clim_year_end, hindcast_det,
            best_code_da=best_code_da, best_shape_da=best_shape_da,
            best_loc_da=best_loc_da, best_scale_da=best_scale_da,
            **probability_options)

    def forecast(self, Predictant, clim_year_start, clim_year_end, Predictor,
                 hindcast_det, Predictor_for_year, alpha=1.0, best_code_da=None,
                 best_shape_da=None, best_loc_da=None, best_scale_da=None):
        return _operational_forecast(self, Predictant, clim_year_start, clim_year_end,
            Predictor, hindcast_det, Predictor_for_year, parameters=(1.0 if alpha is None else alpha,),
            best_code_da=best_code_da, best_shape_da=best_shape_da,
            best_loc_da=best_loc_da, best_scale_da=best_scale_da, raw_target=False)


class WAS_Lasso_Model:
    """Spatial Lasso regression with independent tuning/calibration controls.

    ``mode='grid'`` selects hyperparameters per cell. ``mode='cluster'`` shares
    them among ``n_clusters`` geographical groups independent of the targets;
    each inner search split learns its own cell scales and regional proxy.
    The final regression is always fitted separately for each grid cell.
    ``alpha_range`` defines candidate regularization strengths; ElasticNet also
    accepts ``l1_ratio_range``. At least ten usable dates are required for HPO.
    Constant targets retain a valid intercept and the first candidate setting.

    Search options are ``'bayesian'`` (``n_trials``), ``'random'`` (``n_iter``),
    or grid search (including historical ``*cv`` aliases). ``inner_cv`` controls
    hyperparameter tuning and is independent of probabilistic inner/outer
    calibration. Predictor and target scales are fitted inside every search
    split. ``nb_cores`` controls parallel grid searches; ``random_state`` seeds
    randomized searches and geographical grouping.

    ``compute_prob`` defaults to other outer-fold residuals; ``'inner'`` and
    ``'nested'`` require explicit inner-OOF pools or the Cross_Validator driver.
    ``forecast`` uses supplied OOF hindcasts to calibrate independent future
    dates and accepts fixed alpha=1.0 by default (ElasticNet l1_ratio=0.5).
    ``dist_method`` is ``'nonparam'`` or ``'bestfit'``; climatological terciles
    stay empirical under either method. Physical support follows the target
    metadata/name, so signed temperatures remain signed.
    """

    def __init__(self, alpha_range=None, n_clusters=5, nb_cores=1, dist_method="nonparam",
                 hyperparam_optimizer="bayesian", n_trials=50, n_iter=50, mode="cluster", inner_cv=3, random_state=42):
        if alpha_range is None:
            alpha_range = np.logspace(-6, 2, 100)
        
        self.alpha_range = alpha_range
        self.n_clusters = n_clusters
        self.nb_cores = nb_cores
        self.dist_method = dist_method
        self.hyperparam_optimizer = hyperparam_optimizer
        self.n_trials = n_trials
        self.n_iter = n_iter
        self.mode = mode  # 'cluster' or 'grid'
        self.inner_cv = inner_cv
        self.random_state = random_state

    # ------------------ Hyperparameter Helpers ------------------

    def _optimize_hyperparameters_optuna(self, X, y, alpha_range):
        return _linear_search(self, X, y, "Lasso", alpha_range)

    def _optimize_hyperparameters_randomized(self, X, y, alpha_range):
        return _linear_search(self, X, y, "Lasso", alpha_range)

    def _optimize_single_cell(self, y_vec, X_mat):
        mask = np.isfinite(y_vec) & np.isfinite(X_mat).all(axis=-1)
        if mask.sum() < 10:
            return np.nan
        result = _linear_search(self, X_mat[mask], y_vec[mask], "Lasso", self.alpha_range)
        return result

    # ------------------ Core Logic Methods ------------------

    def compute_hyperparameters(self, predictand, predictor, clim_year_start, clim_year_end):
        """Tune parameters with independent preprocessing in every inner split.

        Climatology bounds are retained for API compatibility; fitted scales
        depend only on the actual training rows, not on the event climatology.
        ``mode='cluster'`` uses geographical groups and a fold-local normalized
        regional proxy; ``mode='grid'`` selects parameters independently per cell.
        """
        return _linear_hyperparameters(self, predictand, predictor, 'Lasso')

    def fit_predict(self, x, y, x_test, y_test, alpha):
        if not np.isfinite(alpha):
            return np.array([np.nan, np.nan])
        estimator = _scaled_linear(linear_model.Lasso(alpha=float(alpha), max_iter=10000))
        return _regression_pair(estimator, x, y, x_test, y_test)
        
    def compute_model(self, X_train, y_train, X_test, y_test, alpha, **kwargs):
        return _spatial_regression(self, X_train, y_train, X_test, y_test, (alpha,))
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
                    norm.ppf(0.32, loc=loc, scale=scale),
                    norm.ppf(2.0 / 3.0, loc=loc, scale=scale),
                )
            elif code == 2:
                return (
                    lognorm.ppf(0.32, s=shape, loc=loc, scale=scale),
                    lognorm.ppf(2.0 / 3.0, s=shape, loc=loc, scale=scale),
                )
            elif code == 3:
                return (
                    expon.ppf(0.32, loc=loc, scale=scale),
                    expon.ppf(2.0 / 3.0, loc=loc, scale=scale),
                )
            elif code == 4:
                return (
                    gamma.ppf(0.32, a=shape, loc=loc, scale=scale),
                    gamma.ppf(2.0 / 3.0, a=shape, loc=loc, scale=scale),
                )
            elif code == 5:
                return (
                    weibull_min.ppf(0.32, c=shape, loc=loc, scale=scale),
                    weibull_min.ppf(2.0 / 3.0, c=shape, loc=loc, scale=scale),
                )
            elif code == 6:
                # Note: Renamed 't_dist' to 't' for standard scipy.stats
                return (
                    t.ppf(0.32, df=shape, loc=loc, scale=scale),
                    t.ppf(2.0 / 3.0, df=shape, loc=loc, scale=scale),
                )
            elif code == 7:
                # Poisson: poisson.ppf(q, mu, loc=0)
                # ASSUMPTION: 'mu' (mean) is passed as 'shape'
                #             'loc' is passed as 'loc'
                #             'scale' is unused
                return (
                    poisson.ppf(0.32, mu=shape, loc=loc),
                    poisson.ppf(2.0 / 3.0, mu=shape, loc=loc),
                )
            elif code == 8:
                # Negative Binomial: nbinom.ppf(q, n, p, loc=0)
                # ASSUMPTION: 'n' (successes) is passed as 'shape'
                #             'p' (probability) is passed as 'scale'
                #             'loc' is passed as 'loc'
                return (
                    nbinom.ppf(0.32, n=shape, p=scale, loc=loc),
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
    def calculate_tercile_probabilities_bestfit(best_guess, error_variance, T1, T2,
                                              dist_code, dof, nonneg=False):
        """Shared predictive CDF; mean and centred residual variance are explicit."""
        return _regression_tercile_probabilities(best_guess, error_variance, T1, T2,
                                                dist_code, dof, nonneg=nonneg)

    @staticmethod
    def calculate_tercile_probabilities_nonparametric(best_guess, error_samples,
                                                    first_tercile, second_tercile,
                                                    nonneg=False):
        """Empirical OOF residual dressing with missing-reference protection."""
        return _residual_probabilities(best_guess, error_samples,
                                       first_tercile, second_tercile, nonneg=nonneg)


    def compute_prob(self, Predictant, clim_year_start, clim_year_end, hindcast_det,
                     best_code_da=None, best_shape_da=None, best_loc_da=None,
                     best_scale_da=None, **probability_options):
        """Calibrate hindcasts with outer residual pools by default.

        The evaluated date is excluded from its own pool; explicit
        ``calibration_splits`` also exclude the corresponding test block and
        buffer. ``probability_calibration='inner'`` (alias ``'nested'``)
        requires per-target inner-OOF ``calibration_residuals``; use
        ``WAS_Cross_Validator`` to construct them. Fixed empirical climatology
        defines all events. Fitted marginal quantiles never replace it.
        """
        from .was_probability_calibration import direct_hindcast_probabilities
        return direct_hindcast_probabilities(
            self, Predictant, clim_year_start, clim_year_end, hindcast_det,
            best_code_da=best_code_da, best_shape_da=best_shape_da,
            best_loc_da=best_loc_da, best_scale_da=best_scale_da,
            **probability_options)

    def forecast(self, Predictant, clim_year_start, clim_year_end, Predictor,
                 hindcast_det, Predictor_for_year, alpha=1.0, best_code_da=None,
                 best_shape_da=None, best_loc_da=None, best_scale_da=None):
        return _operational_forecast(self, Predictant, clim_year_start, clim_year_end,
            Predictor, hindcast_det, Predictor_for_year, parameters=(1.0 if alpha is None else alpha,),
            best_code_da=best_code_da, best_shape_da=best_shape_da,
            best_loc_da=best_loc_da, best_scale_da=best_scale_da, raw_target=False)


class WAS_ElasticNet_Model:
    """Spatial ElasticNet regression with independent tuning/calibration controls.

    ``mode='grid'`` selects hyperparameters per cell. ``mode='cluster'`` shares
    them among ``n_clusters`` geographical groups independent of the targets;
    each inner search split learns its own cell scales and regional proxy.
    The final regression is always fitted separately for each grid cell.
    ``alpha_range`` defines candidate regularization strengths; ElasticNet also
    accepts ``l1_ratio_range``. At least ten usable dates are required for HPO.
    Constant targets retain a valid intercept and the first candidate setting.

    Search options are ``'bayesian'`` (``n_trials``), ``'random'`` (``n_iter``),
    or grid search (including historical ``*cv`` aliases). ``inner_cv`` controls
    hyperparameter tuning and is independent of probabilistic inner/outer
    calibration. Predictor and target scales are fitted inside every search
    split. ``nb_cores`` controls parallel grid searches; ``random_state`` seeds
    randomized searches and geographical grouping.

    ``compute_prob`` defaults to other outer-fold residuals; ``'inner'`` and
    ``'nested'`` require explicit inner-OOF pools or the Cross_Validator driver.
    ``forecast`` uses supplied OOF hindcasts to calibrate independent future
    dates and accepts fixed alpha=1.0 by default (ElasticNet l1_ratio=0.5).
    ``dist_method`` is ``'nonparam'`` or ``'bestfit'``; climatological terciles
    stay empirical under either method. Physical support follows the target
    metadata/name, so signed temperatures remain signed.
    """


    def __init__(self, alpha_range=None, l1_ratio_range=None, n_clusters=5, nb_cores=1, 
                 dist_method="nonparam", hyperparam_optimizer="bayesian", n_trials=50, n_iter=50, mode="cluster", inner_cv=3, random_state=42):
        if alpha_range is None:
            alpha_range = np.logspace(-6, 2, 100)
        if l1_ratio_range is None:
            l1_ratio_range = [0.1, 0.5, 0.7, 0.9, 0.95, 0.99, 1.0]
            
        self.alpha_range = alpha_range
        self.l1_ratio_range = l1_ratio_range
        self.n_clusters = n_clusters
        self.nb_cores = nb_cores
        self.dist_method = dist_method
        self.hyperparam_optimizer = hyperparam_optimizer
        self.n_trials = n_trials
        self.n_iter = n_iter
        self.mode = mode # 'cluster' or 'grid'
        self.inner_cv = inner_cv
        self.random_state = random_state

    # ------------------ Hyperparameter Helpers ------------------

    def _optimize_hyperparameters_optuna(self, X, y, alpha_range, l1_ratio_range):
        return _linear_search(self, X, y, "ElasticNet", alpha_range, l1_ratio_range)

    def _optimize_hyperparameters_randomized(self, X, y, alpha_range, l1_ratio_range):
        return _linear_search(self, X, y, "ElasticNet", alpha_range, l1_ratio_range)

    def _optimize_single_cell(self, y_vec, X_mat):
        mask = np.isfinite(y_vec) & np.isfinite(X_mat).all(axis=-1)
        if mask.sum() < 10:
            return np.array([np.nan, np.nan])
        result = _linear_search(self, X_mat[mask], y_vec[mask], "ElasticNet", self.alpha_range, self.l1_ratio_range)
        return np.array(result)

    # ------------------ Core Logic Methods ------------------

    def compute_hyperparameters(self, predictand, predictor, clim_year_start, clim_year_end):
        """Tune parameters with independent preprocessing in every inner split.

        Climatology bounds are retained for API compatibility; fitted scales
        depend only on the actual training rows, not on the event climatology.
        ``mode='cluster'`` uses geographical groups and a fold-local normalized
        regional proxy; ``mode='grid'`` selects parameters independently per cell.
        """
        return _linear_hyperparameters(self, predictand, predictor, 'ElasticNet')

    def fit_predict(self, x, y, x_test, y_test, alpha, l1_ratio):
        if not np.isfinite(alpha) or not np.isfinite(l1_ratio):
            return np.array([np.nan, np.nan])
        estimator = _scaled_linear(linear_model.ElasticNet(alpha=float(alpha), l1_ratio=float(l1_ratio), max_iter=10000))
        return _regression_pair(estimator, x, y, x_test, y_test)

    def compute_model(self, X_train, y_train, X_test, y_test, alpha, l1_ratio, **kwargs):
        return _spatial_regression(self, X_train, y_train, X_test, y_test, (alpha, l1_ratio,))

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
                    norm.ppf(0.32, loc=loc, scale=scale),
                    norm.ppf(2.0 / 3.0, loc=loc, scale=scale),
                )
            elif code == 2:
                return (
                    lognorm.ppf(0.32, s=shape, loc=loc, scale=scale),
                    lognorm.ppf(2.0 / 3.0, s=shape, loc=loc, scale=scale),
                )
            elif code == 3:
                return (
                    expon.ppf(0.32, loc=loc, scale=scale),
                    expon.ppf(2.0 / 3.0, loc=loc, scale=scale),
                )
            elif code == 4:
                return (
                    gamma.ppf(0.32, a=shape, loc=loc, scale=scale),
                    gamma.ppf(2.0 / 3.0, a=shape, loc=loc, scale=scale),
                )
            elif code == 5:
                return (
                    weibull_min.ppf(0.32, c=shape, loc=loc, scale=scale),
                    weibull_min.ppf(2.0 / 3.0, c=shape, loc=loc, scale=scale),
                )
            elif code == 6:
                # Note: Renamed 't_dist' to 't' for standard scipy.stats
                return (
                    t.ppf(0.32, df=shape, loc=loc, scale=scale),
                    t.ppf(2.0 / 3.0, df=shape, loc=loc, scale=scale),
                )
            elif code == 7:
                # Poisson: poisson.ppf(q, mu, loc=0)
                # ASSUMPTION: 'mu' (mean) is passed as 'shape'
                #             'loc' is passed as 'loc'
                #             'scale' is unused
                return (
                    poisson.ppf(0.32, mu=shape, loc=loc),
                    poisson.ppf(2.0 / 3.0, mu=shape, loc=loc),
                )
            elif code == 8:
                # Negative Binomial: nbinom.ppf(q, n, p, loc=0)
                # ASSUMPTION: 'n' (successes) is passed as 'shape'
                #             'p' (probability) is passed as 'scale'
                #             'loc' is passed as 'loc'
                return (
                    nbinom.ppf(0.32, n=shape, p=scale, loc=loc),
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
    def calculate_tercile_probabilities_bestfit(best_guess, error_variance, T1, T2,
                                              dist_code, dof, nonneg=False):
        """Shared predictive CDF; mean and centred residual variance are explicit."""
        return _regression_tercile_probabilities(best_guess, error_variance, T1, T2,
                                                dist_code, dof, nonneg=nonneg)

    @staticmethod
    def calculate_tercile_probabilities_nonparametric(best_guess, error_samples,
                                                    first_tercile, second_tercile,
                                                    nonneg=False):
        """Empirical OOF residual dressing with missing-reference protection."""
        return _residual_probabilities(best_guess, error_samples,
                                       first_tercile, second_tercile, nonneg=nonneg)


    def compute_prob(self, Predictant, clim_year_start, clim_year_end, hindcast_det,
                     best_code_da=None, best_shape_da=None, best_loc_da=None,
                     best_scale_da=None, **probability_options):
        """Calibrate hindcasts with outer residual pools by default.

        The evaluated date is excluded from its own pool; explicit
        ``calibration_splits`` also exclude the corresponding test block and
        buffer. ``probability_calibration='inner'`` (alias ``'nested'``)
        requires per-target inner-OOF ``calibration_residuals``; use
        ``WAS_Cross_Validator`` to construct them. Fixed empirical climatology
        defines all events. Fitted marginal quantiles never replace it.
        """
        from .was_probability_calibration import direct_hindcast_probabilities
        return direct_hindcast_probabilities(
            self, Predictant, clim_year_start, clim_year_end, hindcast_det,
            best_code_da=best_code_da, best_shape_da=best_shape_da,
            best_loc_da=best_loc_da, best_scale_da=best_scale_da,
            **probability_options)

    # ------------------- FORECAST METHOD -------------------
    def forecast(self, Predictant, clim_year_start, clim_year_end, Predictor,
                 hindcast_det, Predictor_for_year, alpha=1.0, l1_ratio=0.5, best_code_da=None,
                 best_shape_da=None, best_loc_da=None, best_scale_da=None):
        return _operational_forecast(self, Predictant, clim_year_start, clim_year_end,
            Predictor, hindcast_det, Predictor_for_year, parameters=(1.0 if alpha is None else alpha, 0.5 if l1_ratio is None else l1_ratio),
            best_code_da=best_code_da, best_shape_da=best_shape_da,
            best_loc_da=best_loc_da, best_scale_da=best_scale_da, raw_target=False)


class WAS_LassoLars_Model:
    """Spatial LassoLars regression with independent tuning/calibration controls.

    ``mode='grid'`` selects hyperparameters per cell. ``mode='cluster'`` shares
    them among ``n_clusters`` geographical groups independent of the targets;
    each inner search split learns its own cell scales and regional proxy.
    The final regression is always fitted separately for each grid cell.
    ``alpha_range`` defines candidate regularization strengths; ElasticNet also
    accepts ``l1_ratio_range``. At least ten usable dates are required for HPO.
    Constant targets retain a valid intercept and the first candidate setting.

    Search options are ``'bayesian'`` (``n_trials``), ``'random'`` (``n_iter``),
    or grid search (including historical ``*cv`` aliases). ``inner_cv`` controls
    hyperparameter tuning and is independent of probabilistic inner/outer
    calibration. Predictor and target scales are fitted inside every search
    split. ``nb_cores`` controls parallel grid searches; ``random_state`` seeds
    randomized searches and geographical grouping.

    ``compute_prob`` defaults to other outer-fold residuals; ``'inner'`` and
    ``'nested'`` require explicit inner-OOF pools or the Cross_Validator driver.
    ``forecast`` uses supplied OOF hindcasts to calibrate independent future
    dates and accepts fixed alpha=1.0 by default (ElasticNet l1_ratio=0.5).
    ``dist_method`` is ``'nonparam'`` or ``'bestfit'``; climatological terciles
    stay empirical under either method. Physical support follows the target
    metadata/name, so signed temperatures remain signed.
    """


    def __init__(self, alpha_range=None, n_clusters=5, nb_cores=1, dist_method="nonparam",
                 hyperparam_optimizer="bayesian", n_trials=50, n_iter=50, mode="grid", inner_cv=3, random_state=42):
        if alpha_range is None:
            alpha_range = np.logspace(-6, 2, 100)
        
        self.alpha_range = alpha_range
        self.n_clusters = n_clusters
        self.nb_cores = nb_cores
        self.dist_method = dist_method
        self.hyperparam_optimizer = hyperparam_optimizer
        self.n_trials = n_trials
        self.n_iter = n_iter
        self.mode = mode  # 'cluster' or 'grid'
        self.inner_cv = inner_cv
        self.random_state = random_state

    # ------------------ Hyperparameter Helpers ------------------

    def _optimize_hyperparameters_optuna(self, X, y, alpha_range):
        return _linear_search(self, X, y, "LassoLars", alpha_range)

    def _optimize_hyperparameters_randomized(self, X, y, alpha_range):
        return _linear_search(self, X, y, "LassoLars", alpha_range)

    # ------------------ Optimization Helpers ------------------

    def _optimize_single_cell(self, y_vec, X_mat):
        mask = np.isfinite(y_vec) & np.isfinite(X_mat).all(axis=-1)
        if mask.sum() < 10:
            return np.nan
        result = _linear_search(self, X_mat[mask], y_vec[mask], "LassoLars", self.alpha_range)
        return result

    def compute_hyperparameters(self, predictand, predictor, clim_year_start, clim_year_end):
        """Tune parameters with independent preprocessing in every inner split.

        Climatology bounds are retained for API compatibility; fitted scales
        depend only on the actual training rows, not on the event climatology.
        ``mode='cluster'`` uses geographical groups and a fold-local normalized
        regional proxy; ``mode='grid'`` selects parameters independently per cell.
        """
        return _linear_hyperparameters(self, predictand, predictor, 'LassoLars')

    def fit_predict(self, x, y, x_test, y_test, alpha):
        if not np.isfinite(alpha):
            return np.array([np.nan, np.nan])
        estimator = _scaled_linear(linear_model.LassoLars(alpha=float(alpha), max_iter=10000))
        return _regression_pair(estimator, x, y, x_test, y_test)
    
    def compute_model(self, X_train, y_train, X_test, y_test, alpha, **kwargs):
        return _spatial_regression(self, X_train, y_train, X_test, y_test, (alpha,))
    
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
                    norm.ppf(0.32, loc=loc, scale=scale),
                    norm.ppf(2.0 / 3.0, loc=loc, scale=scale),
                )
            elif code == 2:
                return (
                    lognorm.ppf(0.32, s=shape, loc=loc, scale=scale),
                    lognorm.ppf(2.0 / 3.0, s=shape, loc=loc, scale=scale),
                )
            elif code == 3:
                return (
                    expon.ppf(0.32, loc=loc, scale=scale),
                    expon.ppf(2.0 / 3.0, loc=loc, scale=scale),
                )
            elif code == 4:
                return (
                    gamma.ppf(0.32, a=shape, loc=loc, scale=scale),
                    gamma.ppf(2.0 / 3.0, a=shape, loc=loc, scale=scale),
                )
            elif code == 5:
                return (
                    weibull_min.ppf(0.32, c=shape, loc=loc, scale=scale),
                    weibull_min.ppf(2.0 / 3.0, c=shape, loc=loc, scale=scale),
                )
            elif code == 6:
                # Note: Renamed 't_dist' to 't' for standard scipy.stats
                return (
                    t.ppf(0.32, df=shape, loc=loc, scale=scale),
                    t.ppf(2.0 / 3.0, df=shape, loc=loc, scale=scale),
                )
            elif code == 7:
                # Poisson: poisson.ppf(q, mu, loc=0)
                # ASSUMPTION: 'mu' (mean) is passed as 'shape'
                #             'loc' is passed as 'loc'
                #             'scale' is unused
                return (
                    poisson.ppf(0.32, mu=shape, loc=loc),
                    poisson.ppf(2.0 / 3.0, mu=shape, loc=loc),
                )
            elif code == 8:
                # Negative Binomial: nbinom.ppf(q, n, p, loc=0)
                # ASSUMPTION: 'n' (successes) is passed as 'shape'
                #             'p' (probability) is passed as 'scale'
                #             'loc' is passed as 'loc'
                return (
                    nbinom.ppf(0.32, n=shape, p=scale, loc=loc),
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
    def calculate_tercile_probabilities_bestfit(best_guess, error_variance, T1, T2,
                                              dist_code, dof, nonneg=False):
        """Shared predictive CDF; mean and centred residual variance are explicit."""
        return _regression_tercile_probabilities(best_guess, error_variance, T1, T2,
                                                dist_code, dof, nonneg=nonneg)

    @staticmethod
    def calculate_tercile_probabilities_nonparametric(best_guess, error_samples,
                                                    first_tercile, second_tercile,
                                                    nonneg=False):
        """Empirical OOF residual dressing with missing-reference protection."""
        return _residual_probabilities(best_guess, error_samples,
                                       first_tercile, second_tercile, nonneg=nonneg)


    def compute_prob(self, Predictant, clim_year_start, clim_year_end, hindcast_det,
                     best_code_da=None, best_shape_da=None, best_loc_da=None,
                     best_scale_da=None, **probability_options):
        """Calibrate hindcasts with outer residual pools by default.

        The evaluated date is excluded from its own pool; explicit
        ``calibration_splits`` also exclude the corresponding test block and
        buffer. ``probability_calibration='inner'`` (alias ``'nested'``)
        requires per-target inner-OOF ``calibration_residuals``; use
        ``WAS_Cross_Validator`` to construct them. Fixed empirical climatology
        defines all events. Fitted marginal quantiles never replace it.
        """
        from .was_probability_calibration import direct_hindcast_probabilities
        return direct_hindcast_probabilities(
            self, Predictant, clim_year_start, clim_year_end, hindcast_det,
            best_code_da=best_code_da, best_shape_da=best_shape_da,
            best_loc_da=best_loc_da, best_scale_da=best_scale_da,
            **probability_options)

    # ------------------- FORECAST METHOD -------------------

    def forecast(self, Predictant, clim_year_start, clim_year_end, Predictor,
                 hindcast_det, Predictor_for_year, alpha=1.0, best_code_da=None,
                 best_shape_da=None, best_loc_da=None, best_scale_da=None):
        return _operational_forecast(self, Predictant, clim_year_start, clim_year_end,
            Predictor, hindcast_det, Predictor_for_year, parameters=(1.0 if alpha is None else alpha,),
            best_code_da=best_code_da, best_shape_da=best_shape_da,
            best_loc_da=best_loc_da, best_scale_da=best_scale_da, raw_target=False)
