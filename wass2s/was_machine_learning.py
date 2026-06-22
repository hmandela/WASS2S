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
    Polynomial feature expansion followed by ridge regression.
WAS_MARS_Model
    Multivariate Adaptive Regression Splines (py-earth).
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
from sklearn.ensemble import RandomForestRegressor, StackingRegressor
from sklearn.svm import SVR, SVC
from sklearn.model_selection import GridSearchCV, RandomizedSearchCV, cross_val_score
from sklearn.preprocessing import PolynomialFeatures, StandardScaler, RobustScaler, MinMaxScaler
from sklearn.base import clone
from sklearn.neural_network import MLPRegressor
from sklearn.cluster import KMeans
from sklearn.pipeline import Pipeline
from sklearn.compose import TransformedTargetRegressor
import xgboost as xgb
import statsmodels.api as sm
from statsmodels.stats.outliers_influence import variance_inflation_factor as VIF
from statsmodels.stats.anova import anova_lm
from wass2s.utils import *
import optuna
from scipy.stats import uniform, loguniform, randint
from sklearn.linear_model import LogisticRegression

# Data Manipulation and Analysis
import xarray as xr
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

# EOF Analysis
import xeofs as xe

# Parallel Computing
from multiprocessing import cpu_count
from dask.distributed import Client
import dask.array as da

# Typing and Utilities
from typing import List, Tuple, Optional
from collections import defaultdict

# Warnings
import warnings
warnings.filterwarnings('ignore')

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

optuna.logging.set_verbosity(optuna.logging.WARNING)
class BaseOptimizer:
    """
    Unified Optimizer. Supports SVR (multi-grid), MLP, and Stacking architectures.
    """
    
    def __init__(self, optimization_method="grid", n_trials=20, cv=5, random_state=42):
        self.optimization_method = optimization_method
        self.n_trials = n_trials
        self.cv = cv
        self.random_state = random_state

    def optimize(self, model, param_space, X, y, scoring='neg_mean_squared_error'):
        # Ensure data is in numpy format for sklearn stability
        X_data = X.values if hasattr(X, 'values') else X
        y_data = y.values if hasattr(y, 'values') else y
        
        if self.optimization_method == "grid":
            return self._grid_search(model, param_space, X_data, y_data, scoring)
        elif self.optimization_method == "random":
            return self._random_search(model, param_space, X_data, y_data, scoring)
        elif self.optimization_method == "bayesian":
            return self._optuna_search(model, param_space, X_data, y_data, scoring)
        else:
            raise ValueError(f"Unknown method: {self.optimization_method}")

    def _prepare_space(self, param_space, is_wrapped):
        """Adds 'regressor__' prefix if model is a TransformedTargetRegressor."""
        prefix = "regressor__" if is_wrapped else ""
        
        if isinstance(param_space, list):
            return [{f"{prefix}{k}": v for k, v in d.items()} for d in param_space]
        return {f"{prefix}{k}": v for k, v in param_space.items()}

    def _clean_best_params(self, best_params):
        """Removes 'regressor__' prefix from results for class compatibility."""
        return {k.replace("regressor__", ""): v for k, v in best_params.items()}

    def _grid_search(self, model, param_space, X, y, scoring):
        space = self._prepare_space(param_space, hasattr(model, 'regressor'))
        gs = GridSearchCV(model, space, cv=self.cv, scoring=scoring, n_jobs=-1)
        gs.fit(X, y)
        return self._clean_best_params(gs.best_params_)

    def _random_search(self, model, param_space, X, y, scoring):
        # Handle SVR list of dicts by consolidating for RandomSearch
        if isinstance(param_space, list):
            consolidated = {}
            for d in param_space: consolidated.update(d)
            param_space = consolidated
            
        space = self._prepare_space(param_space, hasattr(model, 'regressor'))
        # Convert lists to random distributions
        dist_space = {k: (v if not isinstance(v, list) else v) for k, v in space.items()}
        
        rs = RandomizedSearchCV(model, dist_space, n_iter=self.n_trials, 
                                cv=self.cv, scoring=scoring, random_state=self.random_state, n_jobs=-1)
        rs.fit(X, y)
        return self._clean_best_params(rs.best_params_)

    def _optuna_search(self, model, param_space, X, y, scoring):
            # 1. Prepare the search space
            if isinstance(param_space, list):
                flat_space = {}
                for d in param_space: flat_space.update(d)
            else:
                flat_space = param_space
    
            def objective(trial):
                suggestions = {}
                for name, values in flat_space.items():
                    if isinstance(values, list):
                        suggestions[name] = trial.suggest_categorical(name, values)
                    elif isinstance(values, tuple):
                        low, high = values[0], values[1]
                        log = len(values) == 3 and values[2] == 'log'
                        suggestions[name] = trial.suggest_float(name, low, high, log=log)
    
                model_instance = clone(model)
                prefix = "regressor__" if hasattr(model, 'regressor') else ""
                prefixed_suggestions = {f"{prefix}{k}": v for k, v in suggestions.items()}
                model_instance.set_params(**prefixed_suggestions)
    
                # 2. Implementation of Early Stopping (Pruning)
                # Instead of a simple cross_val_score, we iterate through the CV folds manually
                from sklearn.model_selection import KFold
                from sklearn.metrics import get_scorer
                
                cv_splitter = KFold(n_splits=self.cv, shuffle=True, random_state=self.random_state)
                scorer = get_scorer(scoring)
                cv_scores = []
    
                for i, (train_idx, val_idx) in enumerate(cv_splitter.split(X, y)):
                    X_train_cv, X_val_cv = X[train_idx], X[val_idx]
                    y_train_cv, y_val_cv = y[train_idx], y[val_idx]
                    
                    try:
                        model_instance.fit(X_train_cv, y_train_cv)
                        score = scorer(model_instance, X_val_cv, y_val_cv)
                        cv_scores.append(score)
                        
                        # Report intermediate result to Optuna
                        trial.report(np.mean(cv_scores), i)
                        
                        # 3. Check if we should stop this trial early
                        if trial.should_prune():
                            raise optuna.TrialPruned()
                    except (ValueError, RuntimeError, optuna.TrialPruned) as e:
                        if isinstance(e, optuna.TrialPruned):
                            raise # Re-raise pruning to let Optuna handle it
                        return -np.inf # Penalize math errors
    
                return np.mean(cv_scores)
    
            # 4. Study creation with a MedianPruner
            # It prunes trials whose best intermediate value is worse than the median 
            # of previous trials at the same step.
            study = optuna.create_study(
                direction="maximize", 
                sampler=optuna.samplers.TPESampler(seed=self.random_state),
                pruner=optuna.pruners.MedianPruner(n_startup_trials=5, n_warmup_steps=1)
            )
            
            study.optimize(objective, n_trials=self.n_trials, show_progress_bar=False)
            return study.best_params


class WAS_LogisticRegression_Model:
    """
    Logistic Regression model for **tercile classification** (Below/Normal/Above = 0/1/2)
    on spatiotemporal seasonal climate data (typically rainfall or temperature anomalies).

    Workflow overview:
    1. **Tercile classification**: Compute spatial map of classes (0=Below, 1=Normal, 2=Above)
       based on climatological terciles (33rd and 67th percentiles).
    2. **Spatial clustering** (KMeans) on a summary statistic of the predictand
       (default: mean over time).
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
        Number of spatial clusters for KMeans grouping of the predictand field.
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
        self.cv              = int(cv)
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
        mask = np.isfinite(y)
        if np.any(mask):
            terciles = np.nanpercentile(y[index_start:index_end], [33, 67])
            y_class  = np.digitize(y, bins=terciles, right=True)  # 0/1/2
            return y_class, terciles[0], terciles[1]
        return np.full(y.shape[0], np.nan), np.nan, np.nan

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
        index_start = Predictant.get_index("T").get_loc(str(clim_year_start)).start
        index_end   = Predictant.get_index("T").get_loc(str(clim_year_end)).stop

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
    def _build_cluster_map(self, predictand: xr.DataArray) -> xr.DataArray:
        """KMeans on climatological mean; NaN cells stay NaN."""
        field = predictand.mean("T", skipna=True)
        flat  = field.values.reshape(-1)
        valid = np.isfinite(flat)

        labels = np.full(flat.shape, np.nan)
        if np.any(valid):
            km = KMeans(n_clusters=self.n_clusters, random_state=self.random_state)
            labels[valid] = km.fit_predict(flat[valid].reshape(-1, 1)).astype(float)

        cluster_2d = labels.reshape(field.values.shape)
        cluster_da = xr.DataArray(
            cluster_2d, coords=field.coords, dims=field.dims, name="cluster"
        )
        return cluster_da.where(np.isfinite(field))

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
        predictor["T"] = predictand["T"]

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

            base_est = self._make_estimator()
            bp = self.optimizer.optimize(
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

        return C_da, cw_code_da, maxiter_da, solver_da, cluster_da

    # ------------------------------------------------------------------
    # 3) Per-pixel fit + predict_proba
    # ------------------------------------------------------------------
    def fit_predict(self, x, y, x_test, C, cw_code, max_iter, solver):
        """
        Fit logistic model on one grid cell using broadcast hyperparameters
        and return probability vector [P(Below), P(Normal), P(Above)].

        FIX A – single-class degenerate: zero-initialise out so missing classes
                 get P=0 (not NaN), keeping sum-to-1 invariant.
        FIX B – classes_ extraction: robust Pipeline-unwrapping loop instead of
                 hard-coded named_steps['logit'] lookup.
        """
        if not np.isfinite(C) or not np.isfinite(cw_code) or not np.isfinite(max_iter):
            return np.full(3, np.nan)

        class_weight = self._cw_inv.get(int(cw_code), None)
        est = self._make_estimator(
            C=C, class_weight=class_weight, max_iter=max_iter, solver=str(solver)
        )

        mask = np.isfinite(y) & np.all(np.isfinite(x), axis=-1)
        if not np.any(mask):
            return np.full(3, np.nan)

        x_c  = x[mask, :]
        y_c  = y[mask].astype(int)
        uniq = np.unique(y_c)

        # FIX A: zero-init so absent classes get P=0, not NaN
        if uniq.size < 2:
            out = np.zeros(3)
            out[int(uniq[0])] = 1.0
            return out

        est.fit(x_c, y_c)

        if x_test.ndim == 1:
            x_test = x_test.reshape(1, -1)

        proba = est.predict_proba(x_test).reshape(-1)

        # FIX B: robust classes_ extraction through arbitrary Pipeline depth
        clf = est
        while isinstance(clf, Pipeline):
            clf = clf.steps[-1][1]
        classes = getattr(clf, "classes_", np.arange(len(proba)))

        out = np.zeros(3)   # FIX A: zero-init here too
        for cls, p in zip(classes, proba):
            cls = int(cls)
            if 0 <= cls <= 2:
                out[cls] = p
        return out

    # ------------------------------------------------------------------
    # 4) Parallel model over grid with local params
    # ------------------------------------------------------------------
    def compute_model(
        self,
        X_train,
        y_train,
        X_test,
        C_da,
        cw_code_da,
        maxiter_da,
        solver_da,
    ):
        """
        Parallel tercile classification across the full spatial domain.

        Parameters
        ----------
        X_train    : xr.DataArray (T, features)
        y_train    : xr.DataArray (T, Y, X)  — tercile class labels {0,1,2}
        X_test     : xr.DataArray (T, features) or (features,)
        C_da, cw_code_da, maxiter_da, solver_da : xr.DataArray (Y, X)
            Broadcast hyperparameter arrays from compute_hyperparameters.

        Returns
        -------
        xr.DataArray  dims (probability=['PB','PN','PA'], T, Y, X)

        FIX C – squeeze() replaced by explicit T=1 handling so that single-step
                 CV folds do not silently drop the T dimension.
        FIX D – transpose to ('probability','T','Y','X') added before return,
                 consistent with all other classifier compute_model methods.
        """
        chunksize_x = self._safe_chunk_size(len(y_train.get_index("X")))
        chunksize_y = self._safe_chunk_size(len(y_train.get_index("Y")))

        X_train = X_train.copy()
        X_train["T"] = y_train["T"]
        y_train = y_train.transpose("T", "Y", "X")

        # FIX C: do not use bare .squeeze() — when T=1 it drops the T dim,
        # causing apply_ufunc to omit T from the output.
        X_test        = X_test.transpose("T", "features")
        n_test_steps  = X_test.sizes["T"]
        T_test_coords = X_test["T"].values

        # Cast solver_da from object dtype to fixed unicode so Dask can chunk it
        solver_da = solver_da.astype(str)

        # For each test time step run apply_ufunc independently, then stack.
        # This avoids xarray's T-alignment clash when X_test has a different
        # T index than X_train (which always occurs in LOO-style CV folds).
        client = (
            Client(n_workers=self.nb_cores, threads_per_worker=1)
            if self.nb_cores > 1
            else None
        )
        step_results = []
        try:
            for t_idx in range(n_test_steps):
                X_test_step = X_test.isel(T=t_idx)   # (features,) — no T dim

                step = xr.apply_ufunc(
                    self.fit_predict,
                    X_train,
                    y_train.chunk({"Y": chunksize_y, "X": chunksize_x}),
                    X_test_step,
                    C_da,
                    cw_code_da,
                    maxiter_da,
                    solver_da,
                    input_core_dims=[
                        ("T", "features"), ("T",), ("features",), (), (), (), ()
                    ],
                    output_core_dims=[("probability",)],
                    vectorize=True,
                    dask="parallelized",
                    output_dtypes=["float"],
                    dask_gufunc_kwargs={"output_sizes": {"probability": 3}},
                )
                step = step.compute() if hasattr(step.data, "compute") else step
                # Restore the T coordinate for this step
                step = step.expand_dims(T=[T_test_coords[t_idx]])
                step_results.append(step)
        finally:
            if client is not None:
                client.close()

        result_ = xr.concat(step_results, dim="T")
        result_ = result_.assign_coords(
            probability=("probability", ["PB", "PN", "PA"])
        )

        # FIX D: return consistently transposed
        return result_.transpose("probability", "T", "Y", "X")

    # ------------------------------------------------------------------
    # 5) End-to-end forecast for one target year
    # ------------------------------------------------------------------
    def forecast(
        self,
        Predictant: xr.DataArray,
        clim_year_start: int,
        clim_year_end: int,
        Predictor: xr.DataArray,
        Predictor_for_year: xr.DataArray,
        C_da,
        cw_code_da,
        maxiter_da,
        solver_da,
    ):
        """
        Classify historical data → fit on full history → predict_proba for one year.

        Returns
        -------
        xr.DataArray  dims (probability=['PB','PN','PA'], T=1, Y, X)

        FIX E – chunksize_y / chunksize_x now defined locally (were undefined,
                 causing NameError at runtime).
        FIX F – T-dim check now uses proba_.dims (computed array) instead of
                 proba.dims (lazy dask graph).
        """
        # 1) Classify predictand into tercile classes
        y_class, _, _ = self.compute_class(Predictant, clim_year_start, clim_year_end)

        # FIX E: define chunk sizes here (missing in original)
        chunksize_y = self._safe_chunk_size(len(y_class.get_index("Y")))
        chunksize_x = self._safe_chunk_size(len(y_class.get_index("X")))

        # 2) Align predictor T axis
        Predictor = Predictor.copy()
        Predictor["T"] = y_class["T"]

        # 3) Ensure forecast predictor has a T dimension
        X_test = Predictor_for_year
        if "T" not in X_test.dims:
            if (
                "T" in Predictor_for_year.coords
                and Predictor_for_year.coords["T"].size > 0
            ):
                t0 = pd.Timestamp(
                    Predictor_for_year.coords["T"].values[0]
                ).to_datetime64()
            else:
                t0 = pd.Timestamp(Predictor["T"].values[-1]).to_datetime64()
            X_test = X_test.expand_dims(T=[t0])

        # forecast always has T=1; pass (features,) to fit_predict and restore T after
        X_test_fp     = X_test.transpose("T", "features").isel(T=0)
        T_test_coords = X_test["T"].values

        # Cast solver_da from object dtype to fixed unicode so Dask can chunk it
        solver_da = solver_da.astype(str)

        client = (
            Client(n_workers=self.nb_cores, threads_per_worker=1)
            if self.nb_cores > 1
            else None
        )
        try:
            proba = xr.apply_ufunc(
                self.fit_predict,
                Predictor,
                y_class.chunk({"Y": chunksize_y, "X": chunksize_x}),
                X_test_fp,
                C_da,
                cw_code_da,
                maxiter_da,
                solver_da,
                input_core_dims=[
                    ("T", "features"), ("T",), ("features",), (), (), (), ()
                ],
                output_core_dims=[("probability",)],
                vectorize=True,
                dask="parallelized",
                output_dtypes=["float"],
                dask_gufunc_kwargs={"output_sizes": {"probability": 3}},
            )
            proba_ = proba.compute() if hasattr(proba.data, "compute") else proba
        finally:
            if client is not None:
                client.close()

        proba_ = proba_.assign_coords(
            probability=("probability", ["PB", "PN", "PA"])
        )

        # FIX F: check the computed array (proba_), not the lazy graph (proba)
        if "T" not in proba_.dims:
            proba_ = proba_.expand_dims(T=T_test_coords)

        return proba_.transpose("probability", "T", "Y", "X")

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
            pb = float(np.mean(d < t1))
            pn = float(np.mean((d >= t1) & (d < t2)))
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
    - Predictors should be normalised externally before calling this class.
    - Negative predictions are clipped to zero (suitable for rainfall / non-negative
      variables; remove the clip for temperature or anomalies).
    """

    def __init__(self, nb_cores=1, degree=2, dist_method="nonparam"):
        self.nb_cores    = nb_cores
        self.degree      = degree
        self.dist_method = dist_method

    # ------------------------------------------------------------------
    # 1) Per-pixel fit + predict
    # ------------------------------------------------------------------
    def fit_predict(self, x, y, x_test, y_test):
        """
        Fit polynomial regression on one grid cell, return [error, prediction].

        FIX 1 – scalar clip: use np.maximum instead of item-assignment on a
                 potentially 0-d array (IndexError after squeeze() for T=1 folds).
        """
        poly  = PolynomialFeatures(degree=self.degree)
        model = LinearRegression()

        mask = np.isfinite(y) & np.all(np.isfinite(x), axis=-1)
        if not np.any(mask):
            return np.array([np.nan, np.nan], dtype=float).squeeze()

        y_clean = y[mask]
        x_clean = x[mask, :]

        x_clean_poly = poly.fit_transform(x_clean)
        model.fit(x_clean_poly, y_clean)

        if x_test.ndim == 1:
            x_test = x_test.reshape(1, -1)
        x_test_poly = poly.transform(x_test)

        preds = model.predict(x_test_poly)

        # FIX 1: safe clip for both scalar and array outputs
        preds = np.maximum(preds, 0.0)

        error_ = np.asarray(y_test, float) - preds
        return np.array([error_, preds], dtype=float).squeeze()

    # ------------------------------------------------------------------
    # 2) Parallel spatial hindcast
    # ------------------------------------------------------------------
    def compute_model(self, X_train, y_train, X_test, y_test):
        """
        Parallel polynomial regression across the full (Y, X) domain.


        Returns
        -------
        xr.DataArray (Y, X) or (T, Y, X) — deterministic prediction field.
        """
        # FIX 12: clamp to at least 1
        chunksize_x = max(int(np.round(len(y_train.get_index("X")) / self.nb_cores)), 1)
        chunksize_y = max(int(np.round(len(y_train.get_index("Y")) / self.nb_cores)), 1)

        X_train = X_train.copy()
        X_train["T"] = y_train["T"]
        y_train = y_train.transpose("T", "Y", "X")

        # FIX 2: do NOT use bare .squeeze() — per-step loop preserves T
        X_test        = X_test.transpose("T", "features")
        n_test_steps  = X_test.sizes["T"]
        T_test_coords = X_test["T"].values

        if "T" in y_test.dims:
            y_test = y_test.drop_vars("T")
        y_test = y_test.squeeze().transpose("Y", "X")

        # FIX 3: try/finally for Dask client
        client = (
            Client(n_workers=self.nb_cores, threads_per_worker=1)
            if self.nb_cores > 1 else None
        )

        step_results = []
        try:
            for t_idx in range(n_test_steps):
                X_step = X_test.isel(T=t_idx)      # (features,) — no T dim

                step = xr.apply_ufunc(
                    self.fit_predict,
                    X_train,
                    y_train.chunk({"Y": chunksize_y, "X": chunksize_x}),
                    X_step,
                    y_test.chunk({"Y": chunksize_y, "X": chunksize_x}),
                    input_core_dims=[("T", "features"), ("T",), ("features",), ()],
                    vectorize=True,
                    output_core_dims=[("output",)],
                    dask="parallelized",
                    output_dtypes=["float"],
                    dask_gufunc_kwargs={"output_sizes": {"output": 2}},
                )
                step = step.compute() if hasattr(step.data, "compute") else step
                # Extract the prediction (output index 1) and restore T
                step = step.isel(output=1).expand_dims(T=[T_test_coords[t_idx]])
                step_results.append(step)
        finally:
            if client is not None:
                client.close()

        result_ = xr.concat(step_results, dim="T")
        # For single-step CV folds return (Y, X); for multi-step return (T, Y, X)
        if n_test_steps == 1:
            return result_.squeeze("T", drop=True)
        return result_

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
                return norm.ppf(0.33, loc=loc, scale=scale),      norm.ppf(0.67, loc=loc, scale=scale)
            elif code == 2:
                return lognorm.ppf(0.33, s=shape, loc=loc, scale=scale), lognorm.ppf(0.67, s=shape, loc=loc, scale=scale)
            elif code == 3:
                return expon.ppf(0.33, loc=loc, scale=scale),     expon.ppf(0.67, loc=loc, scale=scale)
            elif code == 4:
                return gamma.ppf(0.33, a=shape, loc=loc, scale=scale), gamma.ppf(0.67, a=shape, loc=loc, scale=scale)
            elif code == 5:
                return weibull_min.ppf(0.33, c=shape, loc=loc, scale=scale), weibull_min.ppf(0.67, c=shape, loc=loc, scale=scale)
            elif code == 6:
                return t.ppf(0.33, df=shape, loc=loc, scale=scale), t.ppf(0.67, df=shape, loc=loc, scale=scale)
            elif code == 7:
                return poisson.ppf(0.33, mu=shape, loc=loc),      poisson.ppf(0.67, mu=shape, loc=loc)
            elif code == 8:
                return nbinom.ppf(0.33, n=shape, p=scale, loc=loc), nbinom.ppf(0.67, n=shape, p=scale, loc=loc)
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
    def calculate_tercile_probabilities_bestfit(best_guess, error_variance, T1, T2, dist_code, dof):
        """
        Tercile probabilities using best-fit climatological distribution family.

        """
        best_guess     = np.asarray(best_guess, float)
        error_variance = np.asarray(error_variance, float)
        n_time         = best_guess.size
        out            = np.full((3, n_time), np.nan, float)

        if (np.all(np.isnan(best_guess)) or np.isnan(dist_code)
                or np.isnan(T1) or np.isnan(T2) or np.isnan(error_variance)):
            return out

        code = int(dist_code)

        if code == 1:
            std = np.sqrt(error_variance)
            out[0, :] = norm.cdf(T1, loc=best_guess, scale=std)
            out[1, :] = norm.cdf(T2, loc=best_guess, scale=std) - norm.cdf(T1, loc=best_guess, scale=std)
            out[2, :] = 1.0 - norm.cdf(T2, loc=best_guess, scale=std)

        elif code == 2:
            sigma = np.sqrt(np.log(1 + error_variance / (best_guess ** 2)))
            mu    = np.log(best_guess) - sigma ** 2 / 2
            out[0, :] = lognorm.cdf(T1, s=sigma, scale=np.exp(mu))
            out[1, :] = lognorm.cdf(T2, s=sigma, scale=np.exp(mu)) - lognorm.cdf(T1, s=sigma, scale=np.exp(mu))
            out[2, :] = 1.0 - lognorm.cdf(T2, s=sigma, scale=np.exp(mu))

        elif code == 3:
            # FIX 4: loc=best_guess (not undefined loc_t)
            scale_ = np.sqrt(error_variance)
            c1 = expon.cdf(T1, loc=best_guess, scale=scale_)
            c2 = expon.cdf(T2, loc=best_guess, scale=scale_)
            out[0, :] = c1
            out[1, :] = c2 - c1
            out[2, :] = 1.0 - c2

        elif code == 4:
            alpha = (best_guess ** 2) / error_variance
            theta = error_variance / best_guess
            c1 = gamma.cdf(T1, a=alpha, scale=theta)
            c2 = gamma.cdf(T2, a=alpha, scale=theta)
            out[0, :] = c1
            out[1, :] = c2 - c1
            out[2, :] = 1.0 - c2

        elif code == 5:
            for i in range(n_time):
                M = best_guess[i]
                V = float(error_variance)
                # FIX 6: no print statements
                if V <= 0 or M <= 0:
                    continue
                # FIX 5: qualified static method reference
                k = fsolve(WAS_PolynomialRegression.weibull_shape_solver, 2.0, args=(M, V))[0]
                if k <= 0:
                    continue
                lambda_scale = M / gamma_function(1 + 1 / k)
                c1 = weibull_min.cdf(T1, c=k, loc=0, scale=lambda_scale)
                c2 = weibull_min.cdf(T2, c=k, loc=0, scale=lambda_scale)
                out[0, i] = c1
                out[1, i] = c2 - c1
                out[2, i] = 1.0 - c2

        elif code == 6:
            if dof <= 2:
                return out
            scale_ = np.sqrt(error_variance * (dof - 2) / dof)
            c1 = t.cdf(T1, df=dof, loc=best_guess, scale=scale_)
            c2 = t.cdf(T2, df=dof, loc=best_guess, scale=scale_)
            out[0, :] = c1
            out[1, :] = c2 - c1
            out[2, :] = 1.0 - c2

        elif code == 7:
            # FIX 7: no print() warning — Poisson appropriateness is a caller concern
            mu_ = best_guess
            c1  = poisson.cdf(T1, mu=mu_)
            c2  = poisson.cdf(T2, mu=mu_)
            out[0, :] = c1
            out[1, :] = c2 - c1
            out[2, :] = 1.0 - c2

        elif code == 8:
            p = np.where(error_variance > best_guess, best_guess / error_variance, np.nan)
            n = np.where(error_variance > best_guess,
                         (best_guess ** 2) / (error_variance - best_guess), np.nan)
            c1 = nbinom.cdf(T1, n=n, p=p)
            c2 = nbinom.cdf(T2, n=n, p=p)
            out[0, :] = c1
            out[1, :] = c2 - c1
            out[2, :] = 1.0 - c2

        else:
            raise ValueError(f"Invalid distribution code: {dist_code}")

        return out

    @staticmethod
    def calculate_tercile_probabilities_nonparametric(best_guess, error_samples, first_tercile, second_tercile):
        """Non-parametric tercile probabilities from historical error samples."""
        n_time    = len(best_guess)
        pred_prob = np.full((3, n_time), np.nan, dtype=float)
        for t_ in range(n_time):
            if np.isnan(best_guess[t_]):
                continue
            dist = best_guess[t_] + error_samples
            dist = dist[np.isfinite(dist)]
            if len(dist) == 0:
                continue
            p_below   = np.mean(dist < first_tercile)
            p_between = np.mean((dist >= first_tercile) & (dist < second_tercile))
            pred_prob[0, t_] = p_below
            pred_prob[1, t_] = p_between
            pred_prob[2, t_] = 1.0 - (p_below + p_between)
        return pred_prob

    # ------------------------------------------------------------------
    # 4) compute_prob
    # ------------------------------------------------------------------
    def compute_prob(
        self,
        Predictant: xr.DataArray,
        clim_year_start,
        clim_year_end,
        hindcast_det: xr.DataArray,
        best_code_da:  xr.DataArray = None,
        best_shape_da: xr.DataArray = None,
        best_loc_da:   xr.DataArray = None,
        best_scale_da: xr.DataArray = None,
    ) -> xr.DataArray:
        """Tercile probabilities for a deterministic hindcast."""
        if "M" in Predictant.dims:
            Predictant = Predictant.isel(M=0).drop_vars("M").squeeze()
        Predictant = Predictant.transpose("T", "Y", "X")

        mask = xr.where(~np.isnan(Predictant.isel(T=0)), 1.0, np.nan)
        clim = Predictant.sel(T=slice(str(clim_year_start), str(clim_year_end)))
        if clim.sizes.get("T", 0) < 3:
            raise ValueError("Not enough years in climatology period for terciles.")

        error_variance = (Predictant - hindcast_det).var(dim="T")
        dof            = max(int(clim.sizes["T"]) - 1, 2)
        terciles_emp   = clim.quantile([0.33, 0.67], dim="T")
        T1_emp = terciles_emp.isel(quantile=0).drop_vars("quantile")
        T2_emp = terciles_emp.isel(quantile=1).drop_vars("quantile")

        dm = self.dist_method

        if dm == "bestfit":
            if any(v is None for v in (best_code_da, best_shape_da, best_loc_da, best_scale_da)):
                raise ValueError("dist_method='bestfit' requires best_code_da, best_shape_da, best_loc_da, best_scale_da.")
            T1, T2 = xr.apply_ufunc(
                self._ppf_terciles_from_code,
                best_code_da, best_shape_da, best_loc_da, best_scale_da,
                input_core_dims=[(), (), (), ()],
                output_core_dims=[(), ()],
                vectorize=True, dask="parallelized", output_dtypes=[float, float],
            )
            hindcast_prob = xr.apply_ufunc(
                self.calculate_tercile_probabilities_bestfit,
                hindcast_det, error_variance, T1, T2, best_code_da,
                input_core_dims=[("T",), (), (), (), ()],
                output_core_dims=[("probability", "T")],
                vectorize=True, kwargs={"dof": dof}, dask="parallelized",
                output_dtypes=[float],
                dask_gufunc_kwargs={"output_sizes": {"probability": 3}, "allow_rechunk": True},
            )

        elif dm == "nonparam":
            error_samples = Predictant - hindcast_det
            hindcast_prob = xr.apply_ufunc(
                self.calculate_tercile_probabilities_nonparametric,
                hindcast_det, error_samples, T1_emp, T2_emp,
                input_core_dims=[("T",), ("T",), (), ()],
                output_core_dims=[("probability", "T")],
                vectorize=True, dask="parallelized", output_dtypes=[float],
                dask_gufunc_kwargs={"output_sizes": {"probability": 3}, "allow_rechunk": True},
            )
        else:
            raise ValueError(f"Invalid dist_method: {self.dist_method}")

        hindcast_prob = hindcast_prob.assign_coords(probability=("probability", ["PB", "PN", "PA"]))
        return (hindcast_prob * mask).transpose("probability", "T", "Y", "X")

    # ------------------------------------------------------------------
    # 5) forecast
    # ------------------------------------------------------------------
    def forecast(
        self,
        Predictant,
        clim_year_start,
        clim_year_end,
        Predictor,
        hindcast_det,
        Predictor_for_year,
        best_code_da=None, best_shape_da=None,
        best_loc_da=None,  best_scale_da=None,
    ):
        """
        End-to-end single-year forecast: deterministic prediction + tercile probabilities.


        Returns
        -------
        forecast_expanded : xr.DataArray (T=1, Y, X) — deterministic forecast
        forecast_prob     : xr.DataArray (probability=3, T=1, Y, X)
        """
        chunksize_x = max(int(np.round(len(Predictant.get_index("X")) / self.nb_cores)), 1)
        chunksize_y = max(int(np.round(len(Predictant.get_index("Y")) / self.nb_cores)), 1)

        # FIX 9: guard T-coord assignment against size mismatch
        Predictor = Predictor.copy()
        if Predictor.sizes["T"] == Predictant.sizes["T"]:
            Predictor["T"] = Predictant["T"]

        Predictant_st = standardize_timeseries(Predictant, clim_year_start, clim_year_end)
        Predictant_st = Predictant_st.transpose("T", "Y", "X")

        # FIX 10: safe single-step extraction — no over-squeezing
        Predictor_for_year_fp = Predictor_for_year.transpose("T", "features").isel(T=0)
        T_forecast_coord      = Predictor_for_year["T"].values

        y_test = xr.full_like(Predictant.isel(T=0), np.nan)

        # FIX 8: try/finally for Dask client
        client = (
            Client(n_workers=self.nb_cores, threads_per_worker=1)
            if self.nb_cores > 1 else None
        )
        try:
            result = xr.apply_ufunc(
                self.fit_predict,
                Predictor,
                Predictant_st.chunk({"Y": chunksize_y, "X": chunksize_x}),
                Predictor_for_year_fp,
                y_test.chunk({"Y": chunksize_y, "X": chunksize_x}),
                input_core_dims=[("T", "features"), ("T",), ("features",), ()],
                vectorize=True,
                dask="parallelized",
                output_core_dims=[("output",)],
                output_dtypes=["float"],
                dask_gufunc_kwargs={"output_sizes": {"output": 2}},
            )
            result_ = result.compute() if hasattr(result.data, "compute") else result
        finally:
            if client is not None:
                client.close()

        result_ = result_.isel(output=1)
        result_ = reverse_standardize(result_, Predictant, clim_year_start, clim_year_end)

        # Build T=1 coordinate
        year    = T_forecast_coord[0].astype("datetime64[Y]").astype(int) + 1970
        month_1 = Predictant.isel(T=0).coords["T"].values.astype("datetime64[M]").astype(int) % 12 + 1
        new_T   = np.datetime64(f"{year}-{month_1:02d}-01")

        forecast_expanded = result_.expand_dims(T=[new_T])
        forecast_expanded["T"] = forecast_expanded["T"].astype("datetime64[ns]")

        # Climatological terciles and error variance
        index_start        = Predictant.get_index("T").get_loc(str(clim_year_start)).start
        index_end          = Predictant.get_index("T").get_loc(str(clim_year_end)).stop
        rainfall_for_tercile = Predictant.isel(T=slice(index_start, index_end))
        terciles           = rainfall_for_tercile.quantile([0.33, 0.67], dim="T")
        T1_emp             = terciles.isel(quantile=0).drop_vars("quantile")
        T2_emp             = terciles.isel(quantile=1).drop_vars("quantile")
        error_variance     = (Predictant - hindcast_det).var(dim="T")
        dof                = max(int(rainfall_for_tercile.sizes["T"]) - 1, 2)

        mask = xr.where(~np.isnan(Predictant.isel(T=0)), 1.0, np.nan).drop_vars("T").squeeze()

        dm = self.dist_method

        # Pass fc_sq (no T dim) to avoid xarray T-alignment clash between
        # the forecast step (T=2025) and training arrays (T=1991..2020).
        # The probability functions receive it as a scalar-per-cell argument
        # and the T=1 dimension is re-attached via expand_dims after ufunc.
        fc_sq = forecast_expanded.squeeze("T", drop=True)   # (Y, X)

        if dm == "bestfit":
            if any(v is None for v in (best_code_da, best_shape_da, best_loc_da, best_scale_da)):
                raise ValueError("dist_method='bestfit' requires best_code_da, best_shape_da, best_loc_da, best_scale_da.")
            T1, T2 = xr.apply_ufunc(
                self._ppf_terciles_from_code,
                best_code_da, best_shape_da, best_loc_da, best_scale_da,
                input_core_dims=[(), (), (), ()], output_core_dims=[(), ()],
                vectorize=True, dask="parallelized", output_dtypes=[float, float],
            )
            # Wrap fc_sq in a length-1 "T" core dim so the static fn receives
            # a (1,) best_guess array — consistent with the hindcast call.
            forecast_prob = xr.apply_ufunc(
                self.calculate_tercile_probabilities_bestfit,
                fc_sq.expand_dims("T"),          # (T=1, Y, X) — new T has no label, no clash
                error_variance, T1, T2, best_code_da,
                input_core_dims=[("T",), (), (), (), ()],
                output_core_dims=[("probability", "T")],
                vectorize=True, dask="parallelized", kwargs={"dof": dof},
                output_dtypes=[float],
                dask_gufunc_kwargs={"output_sizes": {"probability": 3}, "allow_rechunk": True},
            )

        elif dm == "nonparam":
            # Direct numpy computation avoids xarray T-alignment clash between
            # forecast_expanded (T=1) and error_samples (T=n_years).
            error_samples = (Predictant - hindcast_det).transpose("T", "Y", "X")
            T1_np = T1_emp.values
            T2_np = T2_emp.values
            fc_np = fc_sq.values   # (Y, X)
            es_np = error_samples.values   # (T, Y, X)
            prob_np = _compute_forecast_prob_nonparam(fc_np, es_np, T1_np, T2_np)
            # prob_np: (3, 1, Y, X)
            forecast_prob = xr.DataArray(
                prob_np,
                dims=("probability", "T", "Y", "X"),
                coords={"probability": ["PB", "PN", "PA"],
                        "T": [new_T],
                        "Y": Predictant.coords["Y"],
                        "X": Predictant.coords["X"]},
            )
            forecast_prob["T"] = forecast_prob["T"].astype("datetime64[ns]")
            # FIX 11: use result_ (defined), not the non-existent result_da
            return result_ * mask, (mask * forecast_prob).transpose("probability", "T", "Y", "X")
        else:
            raise ValueError(f"Invalid dist_method: {self.dist_method}")

        forecast_prob = forecast_prob.assign_coords(probability=("probability", ["PB", "PN", "PA"]))
        forecast_prob["T"] = forecast_prob["T"].astype("datetime64[ns]")
        # FIX 11: use result_ (defined), not the non-existent result_da
        return result_ * mask, (mask * forecast_prob).transpose("probability", "T", "Y", "X")

        
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
    def fit_predict(self, x, y, x_test, y_test):
        """
        Fit PoissonRegressor on one grid cell and return [error, prediction].

        """
        from sklearn.linear_model import PoissonRegressor

        # FIX 1: mask invalid / NaN samples
        mask = np.isfinite(y) & np.all(np.isfinite(x), axis=-1) & (y >= 0)
        if not np.any(mask):
            return np.array([np.nan, np.nan], dtype=float)

        x_c = x[mask, :]
        y_c = y[mask]

        try:
            model = PoissonRegressor(max_iter=5s00)
            model.fit(x_c, y_c)
        except Exception:
            return np.array([np.nan, np.nan], dtype=float)

        if x_test.ndim == 1:
            x_test = x_test.reshape(1, -1)

        preds = model.predict(x_test).squeeze()

        # FIX 2: safe clip for both scalar and array outputs
        preds = np.maximum(preds, 0.0)

        error_ = np.asarray(y_test, float) - preds
        return np.array([error_, preds], dtype=float).squeeze()

    # ------------------------------------------------------------------
    # 2) Parallel spatial hindcast
    # ------------------------------------------------------------------
    def compute_model(self, X_train, y_train, X_test, y_test):
        """
        Parallel Poisson regression across the full (Y, X) domain.

        FIX 3 – Dask client wrapped in try/finally so it closes even on error.
        FIX 4 – X_test T=1 handling: isel(T=0) + expand_dims to avoid the
                 bare .squeeze() stripping T dimension in CV folds.

        Returns
        -------
        xr.DataArray (Y, X) — deterministic prediction field.
        """
        chunksize_x = max(int(np.round(len(y_train.get_index("X")) / self.nb_cores)), 1)
        chunksize_y = max(int(np.round(len(y_train.get_index("Y")) / self.nb_cores)), 1)

        X_train = X_train.copy()
        X_train["T"] = y_train["T"]
        y_train = y_train.transpose("T", "Y", "X")

        # FIX 4: preserve T dimension for CV folds (T=1 after squeeze → crash)
        X_test        = X_test.transpose("T", "features")
        n_test_steps  = X_test.sizes["T"]
        T_test_coords = X_test["T"].values

        if "T" in y_test.dims:
            y_test = y_test.drop_vars("T")
        y_test = y_test.squeeze().transpose("Y", "X")

        client = (
            Client(n_workers=self.nb_cores, threads_per_worker=1)
            if self.nb_cores > 1 else None
        )

        step_results = []
        try:
            for t_idx in range(n_test_steps):
                X_step = X_test.isel(T=t_idx)          # (features,)
                y_step = y_test                          # (Y, X) — same for all steps

                step = xr.apply_ufunc(
                    self.fit_predict,
                    X_train,
                    y_train.chunk({"Y": chunksize_y, "X": chunksize_x}),
                    X_step,
                    y_step.chunk({"Y": chunksize_y, "X": chunksize_x}),
                    input_core_dims=[("T", "features"), ("T",), ("features",), ()],
                    vectorize=True,
                    output_core_dims=[("output",)],
                    dask="parallelized",
                    output_dtypes=["float"],
                    dask_gufunc_kwargs={"output_sizes": {"output": 2}},
                )
                step = step.compute() if hasattr(step.data, "compute") else step
                step = step.isel(output=1).expand_dims(T=[T_test_coords[t_idx]])
                step_results.append(step)
        finally:
            if client is not None:
                client.close()

        result_ = xr.concat(step_results, dim="T")
        # For hindcast (T=n_years), drop the T dimension to stay consistent
        # with other regression compute_model outputs that return (Y, X).
        if n_test_steps == 1:
            return result_.squeeze("T", drop=True)
        return result_.squeeze("T", drop=True) if n_test_steps == 1 else result_.isel(T=0, drop=True) if n_test_steps == 1 else result_

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
                return norm.ppf(0.33, loc=loc, scale=scale), norm.ppf(0.67, loc=loc, scale=scale)
            elif code == 2:
                return lognorm.ppf(0.33, s=shape, loc=loc, scale=scale), lognorm.ppf(0.67, s=shape, loc=loc, scale=scale)
            elif code == 3:
                return expon.ppf(0.33, loc=loc, scale=scale), expon.ppf(0.67, loc=loc, scale=scale)
            elif code == 4:
                return gamma.ppf(0.33, a=shape, loc=loc, scale=scale), gamma.ppf(0.67, a=shape, loc=loc, scale=scale)
            elif code == 5:
                return weibull_min.ppf(0.33, c=shape, loc=loc, scale=scale), weibull_min.ppf(0.67, c=shape, loc=loc, scale=scale)
            elif code == 6:
                return t.ppf(0.33, df=shape, loc=loc, scale=scale), t.ppf(0.67, df=shape, loc=loc, scale=scale)
            elif code == 7:
                return poisson.ppf(0.33, mu=shape, loc=loc), poisson.ppf(0.67, mu=shape, loc=loc)
            elif code == 8:
                return nbinom.ppf(0.33, n=shape, p=scale, loc=loc), nbinom.ppf(0.67, n=shape, p=scale, loc=loc)
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
    def calculate_tercile_probabilities_bestfit(best_guess, error_variance, T1, T2, dist_code, dof):
        """
        Tercile probabilities using best-fit climatological distribution family.

        FIX 3 – Exponential (code 3): replaced undefined `loc_t` with `best_guess`
                 (the forecast mean shifts the exponential location parameter).
        FIX 4 – Weibull (code 5): replaced bare `weibull_shape_solver` (NameError)
                 with `WAS_PoissonRegression.weibull_shape_solver` (static method ref).
        FIX 5 – Weibull (code 5): removed debug print() statements.
        """
        best_guess     = np.asarray(best_guess, float)
        error_variance = np.asarray(error_variance, float)
        n_time         = best_guess.size
        out            = np.full((3, n_time), np.nan, float)

        if (np.all(np.isnan(best_guess)) or np.isnan(dist_code)
                or np.isnan(T1) or np.isnan(T2) or np.isnan(error_variance)):
            return out

        code = int(dist_code)

        if code == 1:
            std = np.sqrt(error_variance)
            out[0, :] = norm.cdf(T1, loc=best_guess, scale=std)
            out[1, :] = norm.cdf(T2, loc=best_guess, scale=std) - norm.cdf(T1, loc=best_guess, scale=std)
            out[2, :] = 1.0 - norm.cdf(T2, loc=best_guess, scale=std)

        elif code == 2:
            sigma = np.sqrt(np.log(1 + error_variance / (best_guess ** 2)))
            mu    = np.log(best_guess) - sigma ** 2 / 2
            out[0, :] = lognorm.cdf(T1, s=sigma, scale=np.exp(mu))
            out[1, :] = lognorm.cdf(T2, s=sigma, scale=np.exp(mu)) - lognorm.cdf(T1, s=sigma, scale=np.exp(mu))
            out[2, :] = 1.0 - lognorm.cdf(T2, s=sigma, scale=np.exp(mu))

        elif code == 3:
            # FIX 3: loc=best_guess (not undefined loc_t); scale=sqrt(variance)
            scale_ = np.sqrt(error_variance)
            c1 = expon.cdf(T1, loc=best_guess, scale=scale_)
            c2 = expon.cdf(T2, loc=best_guess, scale=scale_)
            out[0, :] = c1
            out[1, :] = c2 - c1
            out[2, :] = 1.0 - c2

        elif code == 4:
            alpha = (best_guess ** 2) / error_variance
            theta = error_variance / best_guess
            c1 = gamma.cdf(T1, a=alpha, scale=theta)
            c2 = gamma.cdf(T2, a=alpha, scale=theta)
            out[0, :] = c1
            out[1, :] = c2 - c1
            out[2, :] = 1.0 - c2

        elif code == 5:
            for i in range(n_time):
                M = best_guess[i]
                V = float(error_variance)
                if V <= 0 or M <= 0:
                    continue
                # FIX 4: qualified static method reference (not bare name)
                # FIX 5: removed print() calls
                k = fsolve(WAS_PoissonRegression.weibull_shape_solver, 2.0, args=(M, V))[0]
                if k <= 0:
                    continue
                lambda_scale = M / gamma_function(1 + 1 / k)
                c1 = weibull_min.cdf(T1, c=k, loc=0, scale=lambda_scale)
                c2 = weibull_min.cdf(T2, c=k, loc=0, scale=lambda_scale)
                out[0, i] = c1
                out[1, i] = c2 - c1
                out[2, i] = 1.0 - c2

        elif code == 6:
            if dof <= 2:
                return out
            scale_ = np.sqrt(error_variance * (dof - 2) / dof)
            c1 = t.cdf(T1, df=dof, loc=best_guess, scale=scale_)
            c2 = t.cdf(T2, df=dof, loc=best_guess, scale=scale_)
            out[0, :] = c1
            out[1, :] = c2 - c1
            out[2, :] = 1.0 - c2

        elif code == 7:
            mu_ = best_guess
            c1 = poisson.cdf(T1, mu=mu_)
            c2 = poisson.cdf(T2, mu=mu_)
            out[0, :] = c1
            out[1, :] = c2 - c1
            out[2, :] = 1.0 - c2

        elif code == 8:
            p = np.where(error_variance > best_guess, best_guess / error_variance, np.nan)
            n = np.where(error_variance > best_guess,
                         (best_guess ** 2) / (error_variance - best_guess), np.nan)
            c1 = nbinom.cdf(T1, n=n, p=p)
            c2 = nbinom.cdf(T2, n=n, p=p)
            out[0, :] = c1
            out[1, :] = c2 - c1
            out[2, :] = 1.0 - c2

        else:
            raise ValueError(f"Invalid distribution code: {dist_code}")

        return out

    @staticmethod
    def calculate_tercile_probabilities_nonparametric(best_guess, error_samples, first_tercile, second_tercile):
        """Non-parametric tercile probabilities from historical error samples."""
        n_time    = len(best_guess)
        pred_prob = np.full((3, n_time), np.nan, dtype=float)
        for t_ in range(n_time):
            if np.isnan(best_guess[t_]):
                continue
            dist = best_guess[t_] + error_samples
            dist = dist[np.isfinite(dist)]
            if len(dist) == 0:
                continue
            p_below   = np.mean(dist < first_tercile)
            p_between = np.mean((dist >= first_tercile) & (dist < second_tercile))
            pred_prob[0, t_] = p_below
            pred_prob[1, t_] = p_between
            pred_prob[2, t_] = 1.0 - (p_below + p_between)
        return pred_prob

    # ------------------------------------------------------------------
    # 4) compute_prob
    # ------------------------------------------------------------------
    def compute_prob(
        self,
        Predictant: xr.DataArray,
        clim_year_start,
        clim_year_end,
        hindcast_det: xr.DataArray,
        best_code_da:  xr.DataArray = None,
        best_shape_da: xr.DataArray = None,
        best_loc_da:   xr.DataArray = None,
        best_scale_da: xr.DataArray = None,
    ) -> xr.DataArray:
        """Tercile probabilities for a deterministic hindcast."""
        if "M" in Predictant.dims:
            Predictant = Predictant.isel(M=0).drop_vars("M").squeeze()
        Predictant = Predictant.transpose("T", "Y", "X")

        mask = xr.where(~np.isnan(Predictant.isel(T=0)), 1.0, np.nan)
        clim = Predictant.sel(T=slice(str(clim_year_start), str(clim_year_end)))
        if clim.sizes.get("T", 0) < 3:
            raise ValueError("Not enough years in climatology period for terciles.")

        error_variance = (Predictant - hindcast_det).var(dim="T")
        dof            = max(int(clim.sizes["T"]) - 1, 2)
        terciles_emp   = clim.quantile([0.33, 0.67], dim="T")
        T1_emp = terciles_emp.isel(quantile=0).drop_vars("quantile")
        T2_emp = terciles_emp.isel(quantile=1).drop_vars("quantile")

        dm = self.dist_method

        if dm == "bestfit":
            if any(v is None for v in (best_code_da, best_shape_da, best_loc_da, best_scale_da)):
                raise ValueError("dist_method='bestfit' requires best_code_da, best_shape_da, best_loc_da, best_scale_da.")
            T1, T2 = xr.apply_ufunc(
                self._ppf_terciles_from_code,
                best_code_da, best_shape_da, best_loc_da, best_scale_da,
                input_core_dims=[(), (), (), ()],
                output_core_dims=[(), ()],
                vectorize=True, dask="parallelized", output_dtypes=[float, float],
            )
            hindcast_prob = xr.apply_ufunc(
                self.calculate_tercile_probabilities_bestfit,
                hindcast_det, error_variance, T1, T2, best_code_da,
                input_core_dims=[("T",), (), (), (), ()],
                output_core_dims=[("probability", "T")],
                vectorize=True, kwargs={"dof": dof}, dask="parallelized",
                output_dtypes=[float],
                dask_gufunc_kwargs={"output_sizes": {"probability": 3}, "allow_rechunk": True},
            )

        elif dm == "nonparam":
            error_samples = Predictant - hindcast_det
            hindcast_prob = xr.apply_ufunc(
                self.calculate_tercile_probabilities_nonparametric,
                hindcast_det, error_samples, T1_emp, T2_emp,
                input_core_dims=[("T",), ("T",), (), ()],
                output_core_dims=[("probability", "T")],
                vectorize=True, dask="parallelized", output_dtypes=[float],
                dask_gufunc_kwargs={"output_sizes": {"probability": 3}, "allow_rechunk": True},
            )
        else:
            raise ValueError(f"Invalid dist_method: {self.dist_method}")

        hindcast_prob = hindcast_prob.assign_coords(probability=("probability", ["PB", "PN", "PA"]))
        return (hindcast_prob * mask).transpose("probability", "T", "Y", "X")

    # ------------------------------------------------------------------
    # 5) forecast
    # ------------------------------------------------------------------
    def forecast(
        self,
        Predictant,
        clim_year_start,
        clim_year_end,
        Predictor,
        hindcast_det,
        Predictor_for_year,
        best_code_da=None, best_shape_da=None, best_loc_da=None, best_scale_da=None,
    ):
        """
        End-to-end deterministic + tercile forecast for one target year.


        Returns
        -------
        forecast_expanded : xr.DataArray (T=1, Y, X) — deterministic forecast
        forecast_prob     : xr.DataArray (probability=3, T=1, Y, X)
        """
        chunksize_x = max(int(np.round(len(Predictant.get_index("X")) / self.nb_cores)), 1)
        chunksize_y = max(int(np.round(len(Predictant.get_index("Y")) / self.nb_cores)), 1)

        # FIX 7: guard T-coord assignment against size mismatch
        Predictor = Predictor.copy()
        if Predictor.sizes["T"] == Predictant.sizes["T"]:
            Predictor["T"] = Predictant["T"]

        Predictant = Predictant.transpose("T", "Y", "X")

        # FIX 8: use isel(T=0) for the single forecast step — no over-squeezing
        Predictor_for_year_fp = Predictor_for_year.transpose("T", "features").isel(T=0)
        T_forecast_coord      = Predictor_for_year["T"].values

        y_test = xr.full_like(Predictant.isel(T=0), np.nan)

        # FIX 6: try/finally for Dask client
        client = (
            Client(n_workers=self.nb_cores, threads_per_worker=1)
            if self.nb_cores > 1 else None
        )
        try:
            result = xr.apply_ufunc(
                self.fit_predict,
                Predictor,
                Predictant.chunk({"Y": chunksize_y, "X": chunksize_x}),
                Predictor_for_year_fp,
                y_test.chunk({"Y": chunksize_y, "X": chunksize_x}),
                input_core_dims=[("T", "features"), ("T",), ("features",), ()],
                vectorize=True,
                dask="parallelized",
                output_core_dims=[("output",)],
                output_dtypes=["float"],
                dask_gufunc_kwargs={"output_sizes": {"output": 2}},
            )
            result_ = result.compute() if hasattr(result.data, "compute") else result
        finally:
            if client is not None:
                client.close()

        forecast_det = result_.isel(output=1)

        # Build T=1 coordinate for probability methods
        year    = T_forecast_coord[0].astype("datetime64[Y]").astype(int) + 1970
        month_1 = Predictant.isel(T=0).coords["T"].values.astype("datetime64[M]").astype(int) % 12 + 1
        new_T   = np.datetime64(f"{year}-{month_1:02d}-01")

        forecast_expanded = forecast_det.expand_dims(T=[new_T])
        forecast_expanded["T"] = forecast_expanded["T"].astype("datetime64[ns]")

        # Climatological terciles and error variance
        rainfall_clim = Predictant.sel(T=slice(str(clim_year_start), str(clim_year_end)))
        terciles      = rainfall_clim.quantile([0.33, 0.67], dim="T")
        T1_emp        = terciles.isel(quantile=0).drop_vars("quantile")
        T2_emp        = terciles.isel(quantile=1).drop_vars("quantile")
        error_variance = (Predictant - hindcast_det).var(dim="T")
        dof = max(int(rainfall_clim.sizes["T"]) - 1, 2)

        dm = self.dist_method

        if dm == "bestfit":
            if any(v is None for v in (best_code_da, best_shape_da, best_loc_da, best_scale_da)):
                raise ValueError("dist_method='bestfit' requires best_code_da, best_shape_da, best_loc_da, best_scale_da.")
            T1, T2 = xr.apply_ufunc(
                self._ppf_terciles_from_code,
                best_code_da, best_shape_da, best_loc_da, best_scale_da,
                input_core_dims=[(), (), (), ()], output_core_dims=[(), ()],
                vectorize=True, dask="parallelized", output_dtypes=[float, float],
            )
            forecast_prob = xr.apply_ufunc(
                self.calculate_tercile_probabilities_bestfit,
                forecast_expanded, error_variance, T1, T2, best_code_da,
                input_core_dims=[("T",), (), (), (), ()],
                output_core_dims=[("probability", "T")],
                vectorize=True, dask="parallelized", kwargs={"dof": dof},
                output_dtypes=[float],
                dask_gufunc_kwargs={"output_sizes": {"probability": 3}, "allow_rechunk": True},
            )

        elif dm == "nonparam":
            error_samples = Predictant - hindcast_det
            forecast_prob = xr.apply_ufunc(
                self.calculate_tercile_probabilities_nonparametric,
                forecast_expanded, error_samples, T1_emp, T2_emp,
                input_core_dims=[("T",), ("T",), (), ()],
                output_core_dims=[("probability", "T")],
                vectorize=True, dask="parallelized", output_dtypes=[float],
                dask_gufunc_kwargs={"output_sizes": {"probability": 3}, "allow_rechunk": True},
            )
        else:
            raise ValueError(f"Invalid dist_method: {self.dist_method}")

        forecast_prob = forecast_prob.assign_coords(probability=("probability", ["PB", "PN", "PA"]))
        return forecast_expanded, forecast_prob.transpose("probability", "T", "Y", "X")


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
    
    def _gcv_score(self, X: np.ndarray, y: np.ndarray, B: np.ndarray, 
                   beta: np.ndarray) -> float:
        """Calculate Generalized Cross-Validation score."""
        n_samples = X.shape[0]
        y_pred = B @ beta
        rss = np.sum((y - y_pred) ** 2)
        
        # Count unique knots
        unique_knots = set()
        for basis in self.basis_functions[1:]:  # Skip intercept
            for var_idx, knot, _ in basis:
                unique_knots.add((var_idx, knot))
        
        n_knots = len(unique_knots)
        n_basis = len(self.basis_functions)
        
        # Effective degrees of freedom: n_basis + penalty * n_knots
        effective_dof = n_basis + self.penalty * n_knots
        
        # GCV formula
        if effective_dof >= n_samples:
            return np.inf
        
        gcv = rss / ((n_samples - effective_dof) ** 2)
        return gcv
    
    def forward_pass(self, X: np.ndarray, y: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """Forward pass to add basis functions."""
        n_samples, n_features = X.shape
        knot_candidates = self._find_knot_candidates(X, y)
        
        # Start with intercept
        self.basis_functions = [[]]  # Intercept
        B = np.ones((n_samples, 1))
        beta = np.array([np.mean(y)])
        best_gcv = self._gcv_score(X, y, B, beta)
        
        iteration = 0
        while len(self.basis_functions) < self.max_terms and iteration < 100:
            iteration += 1
            best_improvement = 0
            best_new_basis = None
            best_new_B = None
            
            # Try adding to each existing basis function
            for parent_idx, parent_basis in enumerate(self.basis_functions):
                # Check degree constraint
                current_degree = len(parent_basis)
                if current_degree >= self.max_degree:
                    continue
                
                # Find variables not yet used in this basis
                used_vars = {v for v, _, _ in parent_basis}
                available_vars = [v for v in range(n_features) if v not in used_vars]
                
                for v in available_vars:
                    for knot in knot_candidates[v]:
                        # Try both hinge directions
                        for side in [1, -1]:
                            # Create new basis function
                            new_basis = parent_basis + [(v, knot, side)]
                            
                            # Evaluate new basis function
                            new_col = np.ones(n_samples)
                            for var_idx, t, s in new_basis:
                                new_col *= self._hinge(X[:, var_idx], t, s)
                            
                            # Add to design matrix
                            new_B = np.column_stack([B, new_col])
                            
                            # Solve least squares
                            try:
                                new_beta, residuals, rank, _ = np.linalg.lstsq(
                                    new_B, y, rcond=None
                                )
                            except np.linalg.LinAlgError:
                                continue
                            
                            # Check if new column is linearly independent
                            if rank <= B.shape[1]:
                                continue
                            
                            # Calculate GCV improvement
                            new_gcv = self._gcv_score(X, y, new_B, new_beta)
                            improvement = best_gcv - new_gcv
                            
                            if improvement > best_improvement:
                                best_improvement = improvement
                                best_new_basis = new_basis
                                best_new_B = new_B
                                best_new_beta = new_beta
            
            # Add the best new basis if it improves GCV
            if best_improvement > 0:
                self.basis_functions.append(best_new_basis)
                B = best_new_B
                beta = best_new_beta
                best_gcv = best_gcv - best_improvement
            else:
                break
        
        return B, beta
    
    def backward_pass(self, X: np.ndarray, y: np.ndarray, B: np.ndarray, beta: np.ndarray):
        """Backward pass to prune basis functions."""
        n_basis = len(self.basis_functions)
        if n_basis <= 2:  # Need at least intercept + 1 basis
            return
        
        current_gcv = self._gcv_score(X, y, B, beta)
        
        improved = True
        while improved and n_basis > 2:
            improved = False
            best_gcv = current_gcv
            best_idx = -1
            best_B = None
            best_beta = None
            
            # Try removing each basis function (except intercept)
            for idx in range(1, n_basis):
                # Create pruned model
                pruned_basis = self.basis_functions[:idx] + self.basis_functions[idx+1:]
                
                # Rebuild design matrix
                pruned_B = np.ones((X.shape[0], len(pruned_basis)))
                for j, basis in enumerate(pruned_basis[1:], 1):
                    col = np.ones(X.shape[0])
                    for var_idx, knot, side in basis:
                        col *= self._hinge(X[:, var_idx], knot, side)
                    pruned_B[:, j] = col
                
                # Fit pruned model
                try:
                    pruned_beta, _, rank, _ = np.linalg.lstsq(pruned_B, y, rcond=None)
                    if rank < pruned_B.shape[1]:
                        continue
                except np.linalg.LinAlgError:
                    continue
                
                # Calculate GCV
                gcv = self._gcv_score(X, y, pruned_B, pruned_beta)
                
                if gcv < best_gcv:
                    best_gcv = gcv
                    best_idx = idx
                    best_B = pruned_B
                    best_beta = pruned_beta
            
            # Apply best pruning if found
            if best_idx > 0:
                self.basis_functions = (
                    self.basis_functions[:best_idx] + 
                    self.basis_functions[best_idx+1:]
                )
                B = best_B
                beta = best_beta
                current_gcv = best_gcv
                n_basis = len(self.basis_functions)
                improved = True
            else:
                break
        
        self.coef_ = beta
        self.dof_ = len(self.basis_functions) + self.penalty * self._count_knots()
    
    def _count_knots(self) -> int:
        """Count unique knots in the model."""
        unique_knots = set()
        for basis in self.basis_functions[1:]:
            for var_idx, knot, _ in basis:
                unique_knots.add((var_idx, knot))
        return len(unique_knots)
    
    def fit(self, X: np.ndarray, y: np.ndarray):
        """Fit MARS model."""
        # Input validation
        X = np.asarray(X)
        y = np.asarray(y).ravel()
        
        if X.shape[0] != y.shape[0]:
            raise ValueError("X and y must have same number of samples")
        
        # Forward pass
        B, beta = self.forward_pass(X, y)
        
        # Backward pass
        self.backward_pass(X, y, B, beta)
        
        # Store final coefficients
        self.coef_ = beta
        self.dof_ = len(self.basis_functions) + self.penalty * self._count_knots()
        
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
        """
        Fits a MARS model to the provided training data, makes predictions 
        on the test data, and calculates the prediction error (if y_test is provided).
        
        Parameters
        ----------
        x : array-like, shape (n_samples, n_features)
            Training data (predictors).
        y : array-like, shape (n_samples,)
            Training targets.
        x_test : array-like, shape (n_features,) or (1, n_features)
            Test data (predictors).
        y_test : float or None
            Test target value. If None, no error is computed.

        Returns
        -------
        np.ndarray
            If y_test is not None, returns [error, prediction].
            If y_test is None, returns [prediction].
        """
        model = MARS(max_terms=self.max_terms, max_degree=self.max_degree, penalty=self.c)
        mask = np.isfinite(y) & np.all(np.isfinite(x), axis=-1)

        if np.any(mask):
            y_clean = y[mask]
            x_clean = x[mask, :]
            model.fit(x_clean, y_clean)

            if x_test.ndim == 1:
                x_test = x_test.reshape(1, -1)

            preds = model.predict(x_test)
            preds[preds < 0] = 0  

            if y_test is not None:
                error_ = y_test - preds
                return np.array([error_, preds]).squeeze()
            else:
                # Only return prediction if y_test is None
                return np.array([preds]).squeeze()
        else:
            # If no valid data, return NaNs
            if y_test is not None:
                return np.array([np.nan, np.nan]).squeeze()
            else:
                return np.array([np.nan]).squeeze()

    def compute_model(self, X_train, y_train, X_test, y_test):
        """
        Applies MARS regression across a spatiotemporal dataset in parallel.

        Parameters
        ----------
        X_train : xarray.DataArray
            Training predictors with dims ('T','features').
        y_train : xarray.DataArray
            Training targets with dims ('T','Y','X').
        X_test : xarray.DataArray
            Test predictors, shape ('features',) or (T, features).
        y_test : xarray.DataArray
            Test targets with dims ('Y','X'), or broadcastable.

        Returns
        -------
        xarray.DataArray
            dims ('output','Y','X'), where 'output'=[error, prediction].
        """
        chunksize_x = int(np.round(len(y_train.get_index("X")) / self.nb_cores))
        chunksize_y = int(np.round(len(y_train.get_index("Y")) / self.nb_cores))
        
        # Align times
        X_train['T'] = y_train['T']
        y_train = y_train.transpose('T', 'Y', 'X')
        X_test = X_test.squeeze()
        y_test = y_test.drop_vars('T').squeeze().transpose('Y', 'X')

        client = Client(n_workers=self.nb_cores, threads_per_worker=1)
        result_da = xr.apply_ufunc(
            self.fit_predict,
            X_train,
            y_train.chunk({'Y': chunksize_y, 'X': chunksize_x}),
            X_test,
            y_test.chunk({'Y': chunksize_y, 'X': chunksize_x}),
            input_core_dims=[('T','features'), ('T',), ('features',), ()],
            vectorize=True,
            dask='parallelized',
            output_core_dims=[('output',)],
            output_dtypes=['float'],
            dask_gufunc_kwargs={'output_sizes': {'output': 2}},
        )
        result_ = result_da.compute()
        client.close()
        return result_.isel(output=1)
    
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
                    norm.ppf(0.67, loc=loc, scale=scale),
                )
            elif code == 2:
                return (
                    lognorm.ppf(0.32, s=shape, loc=loc, scale=scale),
                    lognorm.ppf(0.67, s=shape, loc=loc, scale=scale),
                )
            elif code == 3:
                return (
                    expon.ppf(0.32, loc=loc, scale=scale),
                    expon.ppf(0.67, loc=loc, scale=scale),
                )
            elif code == 4:
                return (
                    gamma.ppf(0.32, a=shape, loc=loc, scale=scale),
                    gamma.ppf(0.67, a=shape, loc=loc, scale=scale),
                )
            elif code == 5:
                return (
                    weibull_min.ppf(0.32, c=shape, loc=loc, scale=scale),
                    weibull_min.ppf(0.67, c=shape, loc=loc, scale=scale),
                )
            elif code == 6:
                # Note: Renamed 't_dist' to 't' for standard scipy.stats
                return (
                    t.ppf(0.32, df=shape, loc=loc, scale=scale),
                    t.ppf(0.67, df=shape, loc=loc, scale=scale),
                )
            elif code == 7:
                # Poisson: poisson.ppf(q, mu, loc=0)
                # ASSUMPTION: 'mu' (mean) is passed as 'shape'
                #             'loc' is passed as 'loc'
                #             'scale' is unused
                return (
                    poisson.ppf(0.32, mu=shape, loc=loc),
                    poisson.ppf(0.67, mu=shape, loc=loc),
                )
            elif code == 8:
                # Negative Binomial: nbinom.ppf(q, n, p, loc=0)
                # ASSUMPTION: 'n' (successes) is passed as 'shape'
                #             'p' (probability) is passed as 'scale'
                #             'loc' is passed as 'loc'
                return (
                    nbinom.ppf(0.32, n=shape, p=scale, loc=loc),
                    nbinom.ppf(0.67, n=shape, p=scale, loc=loc),
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
    def calculate_tercile_probabilities_bestfit(best_guess, error_variance, T1, T2, dist_code, dof 
    ):
        """
        Generic tercile probabilities using best-fit family per grid cell.
    
        Inputs (per grid cell):
        - ``best_guess`` : 1D array over T (hindcast_det or forecast_det)
        - ``T1``, ``T2`` : scalar terciles from climatological best-fit distribution
        - ``dist_code`` : int, as in ``_ppf_terciles_from_code``
        - ``shape``, ``loc``, ``scale`` : scalars from climatology fit
    
        Strategy:
        - For each time step, build a predictive distribution of the same family:
            * Use ``best_guess[t]`` to adjust mean / location;
            * Keep shape parameters from climatology.
        - Then compute probabilities:
            * P(B) = F(T1)
            * P(N) = F(T2) - F(T1)
            * P(A) = 1 - F(T2)
    
        Parameters
        ----------
        best_guess : array-like
            Forecast/hindcast best estimates.
        error_variance : float
            Variance of prediction errors.
        T1 : float
            Lower tercile threshold.
        T2 : float
            Upper tercile threshold.
        dist_code : int
            Distribution code (1-8).
        shape : float
            Shape parameter.
        loc : float
            Location parameter.
        scale : float
            Scale parameter.
    
        Returns
        -------
        array-like
            Probabilities [P(B), P(N), P(A)].
    
        Notes
        -----
        - Uses :math:`F` as the CDF of the predictive distribution.
        """
        
        best_guess = np.asarray(best_guess, float)
        error_variance = np.asarray(error_variance, dtype=float)
        # T1 = np.asarray(T1, dtype=float)
        # T2 = np.asarray(T2, dtype=float)
        n_time = best_guess.size
        out = np.full((3, n_time), np.nan, float)

        if np.all(np.isnan(best_guess)) or np.isnan(dist_code) or np.isnan(T1) or np.isnan(T2) or np.isnan(error_variance):
            return out

        code = int(dist_code)

        # Normal: loc = forecast; scale from clim
        if code == 1:
            error_std = np.sqrt(error_variance)
            out[0, :] = norm.cdf(T1, loc=best_guess, scale=error_std)
            out[1, :] = norm.cdf(T2, loc=best_guess, scale=error_std) - norm.cdf(T1, loc=best_guess, scale=error_std)
            out[2, :] = 1 - norm.cdf(T2, loc=best_guess, scale=error_std)

        # Lognormal: shape = sigma from clim; enforce mean = best_guess
        elif code == 2:
            sigma = np.sqrt(np.log(1 + error_variance / (best_guess**2)))
            mu = np.log(best_guess) - sigma**2 / 2
            out[0, :] = lognorm.cdf(T1, s=sigma, scale=np.exp(mu))
            out[1, :] = lognorm.cdf(T2, s=sigma, scale=np.exp(mu)) - lognorm.cdf(T1, s=sigma, scale=np.exp(mu))
            out[2, :] = 1 - lognorm.cdf(T2, s=sigma, scale=np.exp(mu))      


        # Exponential: keep scale from clim; shift loc so mean = best_guess
        elif code == 3:
            c1 = expon.cdf(T1, loc=best_guess, scale=np.sqrt(error_variance))
            c2 = expon.cdf(T2, loc=loc_t, scale=np.sqrt(error_variance))
            out[0, :] = c1
            out[1, :] = c2 - c1
            out[2, :] = 1.0 - c2

        # Gamma: use shape from clim; set scale so mean = best_guess
        elif code == 4:
            alpha = (best_guess ** 2) / error_variance
            theta = error_variance / best_guess
            c1 = gamma.cdf(T1, a=alpha, scale=theta)
            c2 = gamma.cdf(T2, a=alpha, scale=theta)
            out[0, :] = c1
            out[1, :] = c2 - c1
            out[2, :] = 1.0 - c2

        elif code == 5: # Assuming 5 is for Weibull   
        
            for i in range(n_time):
                # Get the scalar values for this specific element (e.g., grid cell)
                M = best_guess[i]
                print(M)
                V = error_variance
                print(V)
                
                # Handle cases with no variance to avoid division by zero
                if V <= 0 or M <= 0:
                    out[0, i] = np.nan
                    out[1, i] = np.nan
                    out[2, i] = np.nan
                    continue # Skip to the next element
        
                # --- 1. Numerically solve for shape 'k' ---
                # We need a reasonable starting guess. 2.0 is common (Rayleigh dist.)
                initial_guess = 2.0
                
                # fsolve finds the root of our helper function
                k = fsolve(weibull_shape_solver, initial_guess, args=(M, V))[0]
        
                # --- 2. Check for bad solution and calculate scale 'lambda' ---
                if k <= 0:
                    # Solver failed
                    out[0, i] = np.nan
                    out[1, i] = np.nan
                    out[2, i] = np.nan
                    continue
                
                # With 'k' found, we can now algebraically find scale 'lambda'
                # In scipy.stats, scale is 'scale'
                lambda_scale = M / gamma_function(1 + 1/k)
        
                # --- 3. Calculate Probabilities ---
                # In scipy.stats, shape 'k' is 'c'
                # Use the T1 and T2 values for this specific element
                
                c1 = weibull_min.cdf(T1, c=k, loc=0, scale=lambda_scale)
                c2 = weibull_min.cdf(T2, c=k, loc=0, scale=lambda_scale)
        
                out[0, i] = c1
                out[1, i] = c2 - c1
                out[2, i] = 1.0 - c2

        # Student-t: df from clim; scale from clim; loc = best_guess
        elif code == 6:       
            # Check if df is valid for variance calculation
            if dof <= 2:
                # Cannot calculate scale, fill with NaNs
                out[0, :] = np.nan
                out[1, :] = np.nan
                out[2, :] = np.nan
            else:
                # 1. Calculate t-distribution parameters
                # 'loc' (mean) is just the best_guess
                loc = best_guess
                # 'scale' is calculated from the variance and df
                # Variance = scale**2 * (df / (df - 2))
                scale = np.sqrt(error_variance * (dof - 2) / dof)
                
                # 2. Calculate probabilities
                c1 = t.cdf(T1, df=dof, loc=loc, scale=scale)
                c2 = t.cdf(T2, df=dof, loc=loc, scale=scale)

                out[0, :] = c1
                out[1, :] = c2 - c1
                out[2, :] = 1.0 - c2

        elif code == 7: # Assuming 7 is for Poisson
            
            # --- 1. Set the Poisson parameter 'mu' ---
            # The 'mu' parameter is the mean.
            
            # A warning is strongly recommended if error_variance is different from best_guess
            if not np.allclose(best_guess, error_variance, atol=0.5):
                print("Warning: 'error_variance' is not equal to 'best_guess'.")
                print("Poisson model assumes mean=variance and is likely inappropriate.")
                print("Consider using Negative Binomial.")
            
            mu = best_guess
        
            # --- 2. Calculate Probabilities ---
            # poisson.cdf(k, mu) calculates P(X <= k)
            
            c1 = poisson.cdf(T1, mu=mu)
            c2 = poisson.cdf(T2, mu=mu)
            
            out[0, :] = c1
            out[1, :] = c2 - c1
            out[2, :] = 1.0 - c2

        elif code == 8: # Assuming 8 is for Negative Binomial
            
            # --- 1. Calculate Negative Binomial Parameters ---
            # This model is ONLY valid for overdispersion (Variance > Mean).
            # We will use np.where to set parameters to NaN if V <= M.
            
            # p = Mean / Variance
            p = np.where(error_variance > best_guess, 
                         best_guess / error_variance, 
                         np.nan)
            
            # n = Mean^2 / (Variance - Mean)
            n = np.where(error_variance > best_guess, 
                         (best_guess**2) / (error_variance - best_guess), 
                         np.nan)
            
            # --- 2. Calculate Probabilities ---
            # The nbinom.cdf function will propagate NaNs, correctly
            # handling the cases where the model was invalid.
            
            c1 = nbinom.cdf(T1, n=n, p=p)
            c2 = nbinom.cdf(T2, n=n, p=p)
            
            out[0, :] = c1
            out[1, :] = c2 - c1
            out[2, :] = 1.0 - c2
            
        else:
            raise ValueError(f"Invalid distribution")

        return out

    @staticmethod
    def calculate_tercile_probabilities_nonparametric(best_guess, error_samples, first_tercile, second_tercile):
        """Non-parametric method using historical error samples."""
        n_time = len(best_guess)
        pred_prob = np.full((3, n_time), np.nan, dtype=float)
        for t in range(n_time):
            if np.isnan(best_guess[t]):
                continue
            dist = best_guess[t] + error_samples
            dist = dist[np.isfinite(dist)]
            if len(dist) == 0:
                continue
            p_below = np.mean(dist < first_tercile)
            p_between = np.mean((dist >= first_tercile) & (dist < second_tercile))
            p_above = 1.0 - (p_below + p_between)
            pred_prob[0, t] = p_below
            pred_prob[1, t] = p_between
            pred_prob[2, t] = p_above
        return pred_prob


    def compute_prob(
        self,
        Predictant: xr.DataArray,
        clim_year_start,
        clim_year_end,
        hindcast_det: xr.DataArray,
        best_code_da: xr.DataArray = None,
        best_shape_da: xr.DataArray = None,
        best_loc_da: xr.DataArray = None,
        best_scale_da: xr.DataArray = None
    ) -> xr.DataArray:
        """
        Compute tercile probabilities for deterministic hindcasts.

        If dist_method == 'bestfit':
            - Use cluster-based best-fit distributions to:
                * derive terciles analytically from (best_code_da, best_shape_da, best_loc_da, best_scale_da),
                * compute predictive probabilities using the same family.

        Otherwise:
            - Use empirical terciles from Predictant climatology and the selected
              parametric / nonparametric method.

        Parameters
        ----------
        Predictant : xarray.DataArray
            Observed data (T, Y, X) or (T, Y, X, M).
        clim_year_start, clim_year_end : int or str
            Climatology period (inclusive) for thresholds.
        hindcast_det : xarray.DataArray
            Deterministic hindcast (T, Y, X).
        best_code_da, best_shape_da, best_loc_da, best_scale_da : xarray.DataArray, optional
            Output from WAS_TransformData.fit_best_distribution_grid, required for 'bestfit'.

        Returns
        -------
        hindcast_prob : xarray.DataArray
            Probabilities with dims (probability=['PB','PN','PA'], T, Y, X).
        """
        # Handle member dimension if present
        if "M" in Predictant.dims:
            Predictant = Predictant.isel(M=0).drop_vars("M").squeeze()

        # Ensure dimension order
        Predictant = Predictant.transpose("T", "Y", "X")

        # Spatial mask
        mask = xr.where(~np.isnan(Predictant.isel(T=0)), 1.0, np.nan)

        # Climatology subset
        clim = Predictant.sel(T=slice(str(clim_year_start), str(clim_year_end)))
        if clim.sizes.get("T", 0) < 3:
            raise ValueError("Not enough years in climatology period for terciles.")

        # Error variance for predictive distributions
        error_variance = (Predictant - hindcast_det).var(dim="T")
        dof = max(int(clim.sizes["T"]) - 1, 2)

        # Empirical terciles (used by non-bestfit methods)
        terciles_emp = clim.quantile([0.32, 0.67], dim="T")
        T1_emp = terciles_emp.isel(quantile=0).drop_vars("quantile")
        T2_emp = terciles_emp.isel(quantile=1).drop_vars("quantile")
        

        dm = self.dist_method

        # ---------- BESTFIT: zone-wise optimal distributions ----------
        if dm == "bestfit":
            if any(v is None for v in (best_code_da, best_shape_da, best_loc_da, best_scale_da)):
                raise ValueError(
                    "dist_method='bestfit' requires best_code_da, best_shape_da_da, best_loc_da, best_scale_da."
                )

            # T1, T2 from best-fit distributions (per grid)
            T1, T2 = xr.apply_ufunc(
                self._ppf_terciles_from_code,
                best_code_da,
                best_shape_da,
                best_loc_da,
                best_scale_da,
                input_core_dims=[(), (), (), ()],
                output_core_dims=[(), ()],
                vectorize=True,
                dask="parallelized",
                output_dtypes=[float, float],
            )

            # Predictive probabilities using same family
            hindcast_prob = xr.apply_ufunc(
                self.calculate_tercile_probabilities_bestfit,
                hindcast_det,
                error_variance,
                T1,
                T2,
                best_code_da,
                input_core_dims=[("T",), (), (), (), ()],
                output_core_dims=[("probability", "T")],
                vectorize=True,
                kwargs={'dof': dof},
                dask="parallelized",
                output_dtypes=[float],
                dask_gufunc_kwargs={
                    "output_sizes": {"probability": 3},
                    "allow_rechunk": True,
                },
            )

        # ---------- Nonparametric ----------
        elif dm == "nonparam":
            error_samples = Predictant - hindcast_det
            hindcast_prob = xr.apply_ufunc(
                self.calculate_tercile_probabilities_nonparametric,
                hindcast_det,
                error_samples,
                T1_emp,
                T2_emp,
                input_core_dims=[("T",), ("T",), (), ()],
                output_core_dims=[("probability", "T")],
                vectorize=True,
                dask="parallelized",
                output_dtypes=[float],
                dask_gufunc_kwargs={
                    "output_sizes": {"probability": 3},
                    "allow_rechunk": True,
                },
            )

        else:
            raise ValueError(f"Invalid dist_method: {self.dist_method}")

        hindcast_prob = hindcast_prob.assign_coords(
            probability=("probability", ["PB", "PN", "PA"])
        )
        return (hindcast_prob * mask).transpose("probability", "T", "Y", "X")


    # --------------------------------------------------------------------------
    #  FORECAST METHOD
    # --------------------------------------------------------------------------
    def forecast(self, Predictant, clim_year_start, clim_year_end, Predictor, hindcast_det, Predictor_for_year, best_code_da=None, best_shape_da=None, best_loc_da=None, best_scale_da=None):
        """
        Generate a single-year deterministic forecast using the MARS model,
        then compute tercile probabilities (PB, PN, PA) based on the selected
        distribution method (``self.dist_method``).

        The forecast is computed by:
        - Standardizing the predictand over the climatology period
        - Fitting the MARS model on historical data and predicting for the target year
        - Reversing standardization to get raw-scale forecast values
        - Computing probabilities using either best-fit distributions or nonparametric methods

        Parameters
        ----------
        Predictant : xr.DataArray
            Observed target variable (e.g., rainfall) with dimensions (T, Y, X).
        clim_year_start : int
            Start year of climatology period for standardization and terciles.
        clim_year_end : int
            End year of climatology period.
        Predictor : xr.DataArray
            Historical predictors with dimensions (T, features).
        hindcast_det : xr.DataArray
            Historical deterministic hindcasts with dimensions (output=[error, prediction], T, Y, X).
            Used for error variance estimation in nonparametric method.
        Predictor_for_year : xr.DataArray
            Predictors for the target forecast year, shape (features,) or (1, features).
        best_code_da : xr.DataArray, optional
            Distribution codes per grid cell (required for "bestfit" method).
        best_shape_da : xr.DataArray, optional
            Shape parameters per grid cell (required for "bestfit").
        best_loc_da : xr.DataArray, optional
            Location parameters per grid cell (required for "bestfit").
        best_scale_da : xr.DataArray, optional
            Scale parameters per grid cell (required for "bestfit").

        Returns
        -------
        forecast_expanded : xr.DataArray
            Deterministic forecast values with dimensions (T=1, Y, X).
        forecast_prob : xr.DataArray
            Probabilities with dimensions (probability=3, T=1, Y, X):
            - probability: ['PB', 'PN', 'PA'] (Below, Normal, Above).

        Raises
        ------
        ValueError
            If "bestfit" method is selected but required distribution parameters are missing.
            If invalid ``dist_method`` is set.

        Notes
        -----
        - For real forecasts, the "error" in the deterministic output is NaN (no verification available).
        - The forecast date is set to the first day of the month matching the predictand's time structure.
        - Uses Dask for parallel computation across spatial chunks.
        """
        # Provide a dummy y_test with the same shape as the spatial domain => [NaNs]
        y_test_dummy = xr.full_like(Predictant.isel(T=0), np.nan)

        # Chunk sizes
        chunksize_x = int(np.round(len(Predictant.get_index("X")) / self.nb_cores))
        chunksize_y = int(np.round(len(Predictant.get_index("Y")) / self.nb_cores))

        # Align time dimension
        Predictor['T'] = Predictant['T']
        Predictant_st = standardize_timeseries(Predictant, clim_year_start, clim_year_end)
        Predictant_st = Predictant_st.transpose('T','Y','X')
        Predictor_for_year_ = Predictor_for_year.squeeze()

        # 1) Fit+predict in parallel => shape (output=2, Y, X)
        client = Client(n_workers=self.nb_cores, threads_per_worker=1)
        result_da = xr.apply_ufunc(
            self.fit_predict,
            Predictor,
            Predictant_st.chunk({'Y': chunksize_y, 'X': chunksize_x}),
            Predictor_for_year_,
            y_test_dummy.chunk({'Y': chunksize_y, 'X': chunksize_x}),  # dummy y_test
            input_core_dims=[
                ('T','features'),  # x
                ('T',),           # y
                ('features',),    # x_test
                ()
            ],
            vectorize=True,
            dask='parallelized',
            output_core_dims=[('output',)],  # output=2 => [error, prediction]
            output_dtypes=['float'],
            dask_gufunc_kwargs={'output_sizes': {'output':2}},
        )
        result_ = result_da.compute()
        client.close()
        result_ = result_.isel(output=1)
        result_ = reverse_standardize(result_, Predictant, clim_year_start, clim_year_end)

        # 2) Compute thresholds T1, T2 from climatology
        index_start = Predictant.get_index("T").get_loc(str(clim_year_start)).start
        index_end   = Predictant.get_index("T").get_loc(str(clim_year_end)).stop
        rainfall_for_tercile = Predictant.isel(T=slice(index_start, index_end))
        terciles = rainfall_for_tercile.quantile([0.32, 0.67], dim='T')
        T1_emp = terciles.isel(quantile=0).drop_vars('quantile')
        T2_emp = terciles.isel(quantile=1).drop_vars('quantile')
        error_variance = (Predictant - hindcast_det).var(dim='T')
        
        # Expand single prediction to T=1 so probability methods can handle it
        forecast_expanded = result_.expand_dims(
            T=[pd.Timestamp(Predictor_for_year.coords['T'].values[0]).to_pydatetime()]
        )
        year = Predictor_for_year.coords['T'].values[0].astype('datetime64[Y]').astype(int) + 1970
        T_value_1 = Predictant.isel(T=0).coords['T'].values  # Get the datetime64 value from da1
        month_1 = T_value_1.astype('datetime64[M]').astype(int) % 12 + 1  # Extract month
        new_T_value = np.datetime64(f"{year}-{month_1:02d}-01")
        
        forecast_expanded = forecast_expanded.assign_coords(T=xr.DataArray([new_T_value], dims=["T"]))
        forecast_expanded['T'] = forecast_expanded['T'].astype('datetime64[ns]')

        dof = max(int(rainfall_for_tercile.sizes["T"]) - 1, 2)

        dm = self.dist_method

        # ---------- BESTFIT ----------
        if dm == "bestfit":
            if any(v is None for v in (best_code_da, best_shape_da, best_loc_da, best_scale_da)):
                raise ValueError(
                    "dist_method='bestfit' requires best_code_da, best_shape_da, best_loc_da, best_scale_da."
                )
            
            T1, T2 = xr.apply_ufunc(
                self._ppf_terciles_from_code,
                best_code_da,
                best_shape_da,
                best_loc_da,
                best_scale_da,
                input_core_dims=[(), (), (), ()],
                output_core_dims=[(), ()],
                vectorize=True,
                dask="parallelized",
                output_dtypes=[float, float],
            )

            forecast_prob = xr.apply_ufunc(
                self.calculate_tercile_probabilities_bestfit,
                forecast_expanded,
                error_variance,
                T1,
                T2,
                best_code_da,
                input_core_dims=[("T",), (), (), (), ()],
                output_core_dims=[("probability", "T")],
                vectorize=True,
                dask="parallelized",
                kwargs={"dof": dof},
                output_dtypes=[float],
                dask_gufunc_kwargs={
                    "output_sizes": {"probability": 3},
                    "allow_rechunk": True,
                },
            )

        # ---------- Nonparametric ----------
        elif dm == "nonparam":
            error_samples = Predictant - hindcast_det
            forecast_prob = xr.apply_ufunc(
                self.calculate_tercile_probabilities_nonparametric,
                forecast_expanded,
                error_samples,
                T1_emp,
                T2_emp,
                input_core_dims=[("T",), ("T",), (), ()],
                output_core_dims=[("probability", "T")],
                vectorize=True,
                dask="parallelized",
                output_dtypes=[float],
                dask_gufunc_kwargs={
                    "output_sizes": {"probability": 3},
                    "allow_rechunk": True,
                },
            )

        else:
            raise ValueError(f"Invalid dist_method: {self.dist_method}")
        forecast_prob = forecast_prob.assign_coords(probability=('probability', ['PB', 'PN', 'PA']))
        return forecast_expanded, forecast_prob.transpose('probability', 'T', 'Y', 'X')
        
        
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


class _WAS_CountProbMixin:
    @staticmethod
    def _ppf_terciles_from_code(dist_code, shape, loc, scale):
        if np.isnan(dist_code):
            return np.nan, np.nan
        code = int(dist_code)
        try:
            if code == 1:
                return (norm.ppf(0.32, loc=loc, scale=scale),
                        norm.ppf(0.67, loc=loc, scale=scale))
            elif code == 2:
                return (lognorm.ppf(0.32, s=shape, loc=loc, scale=scale),
                        lognorm.ppf(0.67, s=shape, loc=loc, scale=scale))
            elif code == 3:
                return (expon.ppf(0.32, loc=loc, scale=scale),
                        expon.ppf(0.67, loc=loc, scale=scale))
            elif code == 4:
                return (gamma.ppf(0.32, a=shape, loc=loc, scale=scale),
                        gamma.ppf(0.67, a=shape, loc=loc, scale=scale))
            elif code == 5:
                return (weibull_min.ppf(0.32, c=shape, loc=loc, scale=scale),
                        weibull_min.ppf(0.67, c=shape, loc=loc, scale=scale))
            elif code == 6:
                return (t.ppf(0.32, df=shape, loc=loc, scale=scale),
                        t.ppf(0.67, df=shape, loc=loc, scale=scale))
            elif code == 7:
                return (poisson.ppf(0.32, mu=shape, loc=loc),
                        poisson.ppf(0.67, mu=shape, loc=loc))
            elif code == 8:
                return (nbinom.ppf(0.32, n=shape, p=scale, loc=loc),
                        nbinom.ppf(0.67, n=shape, p=scale, loc=loc))
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
    def calculate_tercile_probabilities_bestfit(best_guess, error_variance, T1, T2, dist_code, dof):
        best_guess = np.asarray(best_guess, float)
        error_variance = np.asarray(error_variance, dtype=float)
        n_time = best_guess.size
        out = np.full((3, n_time), np.nan, float)

        if np.all(np.isnan(best_guess)) or np.isnan(dist_code) or np.isnan(T1) or np.isnan(T2) or np.isnan(error_variance):
            return out

        code = int(dist_code)

        if code == 1:
            error_std = np.sqrt(error_variance)
            out[0, :] = norm.cdf(T1, loc=best_guess, scale=error_std)
            out[1, :] = norm.cdf(T2, loc=best_guess, scale=error_std) - norm.cdf(T1, loc=best_guess, scale=error_std)
            out[2, :] = 1 - norm.cdf(T2, loc=best_guess, scale=error_std)

        elif code == 2:
            sigma = np.sqrt(np.log(1 + error_variance / (best_guess ** 2)))
            mu = np.log(best_guess) - sigma ** 2 / 2
            out[0, :] = lognorm.cdf(T1, s=sigma, scale=np.exp(mu))
            out[1, :] = lognorm.cdf(T2, s=sigma, scale=np.exp(mu)) - lognorm.cdf(T1, s=sigma, scale=np.exp(mu))
            out[2, :] = 1 - lognorm.cdf(T2, s=sigma, scale=np.exp(mu))

        elif code == 3:
####### Revoir cette partie dans les autres classes
            scale_ = np.sqrt(error_variance)
            c1 = expon.cdf(T1, loc=best_guess, scale=scale_)
            c2 = expon.cdf(T2, loc=best_guess, scale=scale_)
            out[0, :] = c1
            out[1, :] = c2 - c1
            out[2, :] = 1.0 - c2

        elif code == 4:
            alpha = (best_guess ** 2) / error_variance
            theta = error_variance / best_guess
            c1 = gamma.cdf(T1, a=alpha, scale=theta)
            c2 = gamma.cdf(T2, a=alpha, scale=theta)
            out[0, :] = c1
            out[1, :] = c2 - c1
            out[2, :] = 1.0 - c2

        elif code == 5:
            for i in range(n_time):
                M = best_guess[i]
                V = float(error_variance)
                if (V <= 0) or (M <= 0):
                    continue
                k = fsolve(_WAS_CountProbMixin.weibull_shape_solver, 2.0, args=(M, V))[0]
                if k <= 0:
                    continue
                lambda_scale = M / gamma_function(1 + 1 / k)
                c1 = weibull_min.cdf(T1, c=k, loc=0, scale=lambda_scale)
                c2 = weibull_min.cdf(T2, c=k, loc=0, scale=lambda_scale)
                out[0, i] = c1
                out[1, i] = c2 - c1
                out[2, i] = 1.0 - c2

        elif code == 6:
            if dof <= 2:
                return out
            loc_ = best_guess
            scale_ = np.sqrt(error_variance * (dof - 2) / dof)
            c1 = t.cdf(T1, df=dof, loc=loc_, scale=scale_)
            c2 = t.cdf(T2, df=dof, loc=loc_, scale=scale_)
            out[0, :] = c1
            out[1, :] = c2 - c1
            out[2, :] = 1.0 - c2

        elif code == 7:
            mu_ = best_guess
            c1 = poisson.cdf(T1, mu=mu_)
            c2 = poisson.cdf(T2, mu=mu_)
            out[0, :] = c1
            out[1, :] = c2 - c1
            out[2, :] = 1.0 - c2

        elif code == 8:
            p = np.where(error_variance > best_guess, best_guess / error_variance, np.nan)
            n = np.where(error_variance > best_guess, (best_guess ** 2) / (error_variance - best_guess), np.nan)
            c1 = nbinom.cdf(T1, n=n, p=p)
            c2 = nbinom.cdf(T2, n=n, p=p)
            out[0, :] = c1
            out[1, :] = c2 - c1
            out[2, :] = 1.0 - c2

        else:
            raise ValueError("Invalid distribution code")

        return out

    @staticmethod
    def calculate_tercile_probabilities_nonparametric(best_guess, error_samples, first_tercile, second_tercile):
        n_time = len(best_guess)
        pred_prob = np.full((3, n_time), np.nan, dtype=float)
        for t_ in range(n_time):
            if np.isnan(best_guess[t_]):
                continue
            dist = best_guess[t_] + error_samples
            dist = dist[np.isfinite(dist)]
            if len(dist) == 0:
                continue
            p_below = np.mean(dist < first_tercile)
            p_between = np.mean((dist >= first_tercile) & (dist < second_tercile))
            p_above = 1.0 - (p_below + p_between)
            pred_prob[0, t_] = p_below
            pred_prob[1, t_] = p_between
            pred_prob[2, t_] = p_above
        return pred_prob


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
    def fit_predict(self, x, y, x_test, y_test):
        x      = np.asarray(x, float)
        y      = np.asarray(y, float)
        x_test = np.asarray(x_test, float)

        if x.ndim == 1:
            x = x[:, None]
        x, y = _safe_mask_xy(x, y)

        if y.size < 5 or np.any(y < 0):
            return np.array([np.nan, np.nan], dtype=float)

        X = _add_intercept(x) if self.add_intercept else x
        beta, alpha = _nb2_irls_beta_alpha(y, X, alpha_init=self.alpha_init)

        if x_test.ndim == 1:
            x_test = x_test.reshape(1, -1)
        Xtest   = _add_intercept(x_test) if self.add_intercept else x_test
        eta_test = np.clip(Xtest @ beta, -700.0, 700.0)
        mu_test  = np.maximum(np.exp(eta_test).squeeze(), 0.0)

        err = (np.asarray(y_test, float) - mu_test).squeeze()
        return np.array([err, mu_test], dtype=float).squeeze()

    # ------------------------------------------------------------------
    def compute_model(self, X_train, y_train, X_test, y_test):
        """
        Parallel NB2 regression across the full (Y, X) domain.

        FIX 10 – per-step loop instead of bare .squeeze() on X_test,
                  preserving the T dimension for T=1 CV folds.
        """
        chunksize_x = max(int(np.round(len(y_train.get_index("X")) / self.nb_cores)), 1)
        chunksize_y = max(int(np.round(len(y_train.get_index("Y")) / self.nb_cores)), 1)

        X_train = X_train.copy()
        X_train["T"] = y_train["T"]
        y_train = y_train.transpose("T", "Y", "X")

        # FIX 10: do not use bare .squeeze() — splits into per-step calls
        X_test        = X_test.transpose("T", "features")
        n_test_steps  = X_test.sizes["T"]
        T_test_coords = X_test["T"].values

        if "T" in y_test.dims:
            y_test = y_test.drop_vars("T")
        y_test = y_test.squeeze().transpose("Y", "X")

        client = (
            Client(n_workers=self.nb_cores, threads_per_worker=1)
            if self.nb_cores and self.nb_cores > 1 else None
        )

        step_results = []
        try:
            for t_idx in range(n_test_steps):
                X_step = X_test.isel(T=t_idx)      # (features,)

                step = xr.apply_ufunc(
                    self.fit_predict,
                    X_train,
                    y_train.chunk({"Y": chunksize_y, "X": chunksize_x}),
                    X_step,
                    y_test.chunk({"Y": chunksize_y, "X": chunksize_x}),
                    input_core_dims=[("T", "features"), ("T",), ("features",), ()],
                    vectorize=True,
                    output_core_dims=[("output",)],
                    dask="parallelized",
                    output_dtypes=["float"],
                    dask_gufunc_kwargs={"output_sizes": {"output": 2}},
                )
                step = step.compute() if hasattr(step.data, "compute") else step
                step = step.isel(output=1).expand_dims(T=[T_test_coords[t_idx]])
                step_results.append(step)
        finally:
            if client is not None:
                client.close()

        result_ = xr.concat(step_results, dim="T")
        # Return (Y, X) for hindcast multi-step or (Y, X) for single step —
        # squeeze T so shape matches the rest of the framework's convention.
        if n_test_steps == 1:
            return result_.squeeze("T", drop=True)
        return result_

    # ------------------------------------------------------------------
    def compute_prob(
        self,
        Predictant: xr.DataArray,
        clim_year_start, clim_year_end,
        hindcast_det: xr.DataArray,
        best_code_da=None, best_shape_da=None,
        best_loc_da=None, best_scale_da=None,
    ) -> xr.DataArray:

        if "M" in Predictant.dims:
            Predictant = Predictant.isel(M=0).drop_vars("M").squeeze()

        Predictant = Predictant.transpose("T", "Y", "X")
        mask       = xr.where(~np.isnan(Predictant.isel(T=0)), 1.0, np.nan)

        clim = Predictant.sel(T=slice(str(clim_year_start), str(clim_year_end)))
        if clim.sizes.get("T", 0) < 3:
            raise ValueError("Not enough years in climatology period for terciles.")

        error_variance = (Predictant - hindcast_det).var(dim="T")
        dof            = max(int(clim.sizes["T"]) - 1, 2)
        terciles_emp   = clim.quantile([0.33, 0.67], dim="T")
        T1_emp = terciles_emp.isel(quantile=0).drop_vars("quantile")
        T2_emp = terciles_emp.isel(quantile=1).drop_vars("quantile")

        dm = self.dist_method

        if dm == "bestfit":
            if any(v is None for v in (best_code_da, best_shape_da, best_loc_da, best_scale_da)):
                raise ValueError("dist_method='bestfit' requires all four distribution arrays.")
            T1, T2 = xr.apply_ufunc(
                self._ppf_terciles_from_code,
                best_code_da, best_shape_da, best_loc_da, best_scale_da,
                input_core_dims=[(), (), (), ()], output_core_dims=[(), ()],
                vectorize=True, dask="parallelized", output_dtypes=[float, float],
            )
            hindcast_prob = xr.apply_ufunc(
                self.calculate_tercile_probabilities_bestfit,
                hindcast_det, error_variance, T1, T2, best_code_da,
                input_core_dims=[("T",), (), (), (), ()],
                output_core_dims=[("probability", "T")],
                vectorize=True, kwargs={"dof": dof}, dask="parallelized",
                output_dtypes=[float],
                dask_gufunc_kwargs={"output_sizes": {"probability": 3}, "allow_rechunk": True},
            )

        elif dm == "nonparam":
            error_samples = Predictant - hindcast_det
            hindcast_prob = xr.apply_ufunc(
                self.calculate_tercile_probabilities_nonparametric,
                hindcast_det, error_samples, T1_emp, T2_emp,
                input_core_dims=[("T",), ("T",), (), ()],
                output_core_dims=[("probability", "T")],
                vectorize=True, dask="parallelized", output_dtypes=[float],
                dask_gufunc_kwargs={"output_sizes": {"probability": 3}, "allow_rechunk": True},
            )
        else:
            raise ValueError(f"Invalid dist_method: {self.dist_method}")

        hindcast_prob = hindcast_prob.assign_coords(probability=("probability", ["PB", "PN", "PA"]))
        return (hindcast_prob * mask).transpose("probability", "T", "Y", "X")

    # ------------------------------------------------------------------
    def forecast(
        self,
        Predictant, clim_year_start, clim_year_end,
        Predictor, hindcast_det, Predictor_for_year,
        best_code_da=None, best_shape_da=None,
        best_loc_da=None, best_scale_da=None,
    ):
        """
        End-to-end single-year forecast.

        FIX 11 – T-coord assignment guarded by size check.
        FIX 12 – Predictor_for_year projected via isel(T=0), not bare squeeze.
        """
        chunksize_x = max(int(np.round(len(Predictant.get_index("X")) / self.nb_cores)), 1)
        chunksize_y = max(int(np.round(len(Predictant.get_index("Y")) / self.nb_cores)), 1)

        # FIX 11: guard against size mismatch on T-coord assignment
        Predictor = Predictor.copy()
        if Predictor.sizes["T"] == Predictant.sizes["T"]:
            Predictor["T"] = Predictant["T"]

        Predictant = Predictant.transpose("T", "Y", "X")

        # FIX 12: safe single-step extraction
        Predictor_for_year_fp = Predictor_for_year.transpose("T", "features").isel(T=0)
        T_forecast_coord      = Predictor_for_year["T"].values

        y_test = xr.full_like(Predictant.isel(T=0), np.nan)

        client = (
            Client(n_workers=self.nb_cores, threads_per_worker=1)
            if self.nb_cores and self.nb_cores > 1 else None
        )
        try:
            result = xr.apply_ufunc(
                self.fit_predict,
                Predictor,
                Predictant.chunk({"Y": chunksize_y, "X": chunksize_x}),
                Predictor_for_year_fp,
                y_test.chunk({"Y": chunksize_y, "X": chunksize_x}),
                input_core_dims=[("T", "features"), ("T",), ("features",), ()],
                vectorize=True,
                dask="parallelized",
                output_core_dims=[("output",)],
                output_dtypes=["float"],
                dask_gufunc_kwargs={"output_sizes": {"output": 2}},
            )
            result_ = result.compute() if hasattr(result.data, "compute") else result
        finally:
            if client is not None:
                client.close()

        forecast_det = result_.isel(output=1)

        year    = T_forecast_coord[0].astype("datetime64[Y]").astype(int) + 1970
        month_1 = Predictant.isel(T=0).coords["T"].values.astype("datetime64[M]").astype(int) % 12 + 1
        new_T   = np.datetime64(f"{year}-{month_1:02d}-01")

        forecast_expanded = forecast_det.expand_dims(T=[new_T])
        forecast_expanded["T"] = forecast_expanded["T"].astype("datetime64[ns]")

        rainfall_clim  = Predictant.sel(T=slice(str(clim_year_start), str(clim_year_end)))
        terciles       = rainfall_clim.quantile([0.33, 0.67], dim="T")
        T1_emp         = terciles.isel(quantile=0).drop_vars("quantile")
        T2_emp         = terciles.isel(quantile=1).drop_vars("quantile")
        error_variance = (Predictant - hindcast_det).var(dim="T")
        dof            = max(int(rainfall_clim.sizes["T"]) - 1, 2)

        dm = self.dist_method

        if dm == "bestfit":
            if any(v is None for v in (best_code_da, best_shape_da, best_loc_da, best_scale_da)):
                raise ValueError("dist_method='bestfit' requires all four distribution arrays.")
            T1, T2 = xr.apply_ufunc(
                self._ppf_terciles_from_code,
                best_code_da, best_shape_da, best_loc_da, best_scale_da,
                input_core_dims=[(), (), (), ()], output_core_dims=[(), ()],
                vectorize=True, dask="parallelized", output_dtypes=[float, float],
            )
            forecast_prob = xr.apply_ufunc(
                self.calculate_tercile_probabilities_bestfit,
                forecast_expanded, error_variance, T1, T2, best_code_da,
                input_core_dims=[("T",), (), (), (), ()],
                output_core_dims=[("probability", "T")],
                vectorize=True, dask="parallelized", kwargs={"dof": dof},
                output_dtypes=[float],
                dask_gufunc_kwargs={"output_sizes": {"probability": 3}, "allow_rechunk": True},
            )

        elif dm == "nonparam":
            # Direct numpy computation — avoids xarray T-alignment clash between
            # forecast_expanded (T=1) and error_samples (T=n_years).
            error_samples = (Predictant - hindcast_det).transpose("T", "Y", "X")
            T1_np = T1_emp.values; T2_np = T2_emp.values
            fc_np = forecast_det.values   # (Y, X) — no T dim

            nY2, nX2 = fc_np.shape
            prob_np = np.full((3, 1, nY2, nX2), np.nan, dtype=float)
            es_np = error_samples.values
            for iy_ in range(nY2):
                for ix_ in range(nX2):
                    bg_ = fc_np[iy_, ix_]
                    if not np.isfinite(bg_): continue
                    es_ = es_np[:, iy_, ix_]
                    d_  = bg_ + es_; d_ = d_[np.isfinite(d_)]
                    if not len(d_): continue
                    pb_ = float(np.mean(d_ < T1_np[iy_, ix_]))
                    pn_ = float(np.mean((d_ >= T1_np[iy_, ix_]) & (d_ < T2_np[iy_, ix_])))
                    prob_np[0, 0, iy_, ix_] = pb_
                    prob_np[1, 0, iy_, ix_] = pn_
                    prob_np[2, 0, iy_, ix_] = 1.0 - (pb_ + pn_)

            forecast_prob = xr.DataArray(
                prob_np,
                dims=("probability", "T", "Y", "X"),
                coords={"probability": ["PB", "PN", "PA"],
                        "T": forecast_expanded["T"].values,
                        "Y": Predictant.coords["Y"],
                        "X": Predictant.coords["X"]},
            )
            forecast_prob["T"] = forecast_prob["T"].astype("datetime64[ns]")
            return forecast_expanded, forecast_prob.transpose("probability", "T", "Y", "X")
        else:
            raise ValueError(f"Invalid dist_method: {self.dist_method}")

        forecast_prob = forecast_prob.assign_coords(probability=("probability", ["PB", "PN", "PA"]))
        return forecast_expanded, forecast_prob.transpose("probability", "T", "Y", "X")


# ============================================================
# WAS_ZINB_Model and WAS_HurdleNB_Model — only fit_predict
# differs; compute_model / compute_prob / forecast are
# inherited from the fixed WAS_NegativeBinomial_Model above.
# ============================================================

class WAS_ZINB_Model(WAS_NegativeBinomial_Model):
    """
    Zero-Inflated Negative Binomial (two-part, Dask-safe).

    P(Y=0|X) = pi(X) + (1-pi(X)) * NB(y=0 | mu(X), alpha)
    E[Y|X]   = (1 - pi(X)) * mu(X)
    """

    def fit_predict(self, x, y, x_test, y_test):
        x      = np.asarray(x, float)
        y      = np.asarray(y, float)
        x_test = np.asarray(x_test, float)

        if x.ndim == 1:
            x = x[:, None]
        x, y = _safe_mask_xy(x, y)

        if y.size < 8 or np.any(y < 0):
            return np.array([np.nan, np.nan], dtype=float)

        X       = _add_intercept(x) if self.add_intercept else x
        y0      = (y == 0.0).astype(float)
        gamma_  = _logit_irls_coef(y0, X)
        beta, alpha = _nb2_irls_beta_alpha(y, X, alpha_init=self.alpha_init)

        if x_test.ndim == 1:
            x_test = x_test.reshape(1, -1)
        Xtest  = _add_intercept(x_test) if self.add_intercept else x_test

        mu   = np.maximum(np.exp(np.clip(Xtest @ beta, -700.0, 700.0)).squeeze(), 0.0)
        pi   = np.clip((1.0 / (1.0 + np.exp(-np.clip(Xtest @ gamma_, -35.0, 35.0)))).squeeze(), 0.0, 1.0)
        yhat = (1.0 - pi) * mu

        err = (np.asarray(y_test, float) - yhat).squeeze()
        return np.array([err, yhat], dtype=float).squeeze()


class WAS_HurdleNB_Model(WAS_NegativeBinomial_Model):
    """
    Hurdle Negative Binomial (two-part, Dask-safe).

    P(Y>0|X) = p+(X)  via logistic regression.
    Y|Y>0,X  ~ zero-truncated NB approximated by NB on positives.
    E[Y|X]   = p+(X) * mu(X) / (1 - NB_P(Y=0|mu, alpha))
    """

    def fit_predict(self, x, y, x_test, y_test):
        x      = np.asarray(x, float)
        y      = np.asarray(y, float)
        x_test = np.asarray(x_test, float)

        if x.ndim == 1:
            x = x[:, None]
        x, y = _safe_mask_xy(x, y)

        if y.size < 8 or np.any(y < 0):
            return np.array([np.nan, np.nan], dtype=float)

        X      = _add_intercept(x) if self.add_intercept else x
        y_pos  = (y > 0.0).astype(float)
        gamma_ = _logit_irls_coef(y_pos, X)

        mpos = y > 0.0
        if np.sum(mpos) < 5:
            return np.array([np.nan, np.nan], dtype=float)

        beta, alpha = _nb2_irls_beta_alpha(y[mpos], X[mpos, :], alpha_init=self.alpha_init)

        if x_test.ndim == 1:
            x_test = x_test.reshape(1, -1)
        Xtest = _add_intercept(x_test) if self.add_intercept else x_test

        pplus = np.clip((1.0 / (1.0 + np.exp(-np.clip(Xtest @ gamma_, -35.0, 35.0)))).squeeze(), 0.0, 1.0)
        mu    = np.maximum(np.exp(np.clip(Xtest @ beta, -700.0, 700.0)).squeeze(), 0.0)

        alpha  = max(float(alpha), 1e-10)
        r      = 1.0 / alpha
        p_nb   = r / (r + mu + 1e-12)
        P0     = np.power(p_nb, r)
        trunc_mean = mu / np.clip(1.0 - P0, 1e-12, None)

        yhat = pplus * trunc_mean
        err  = (np.asarray(y_test, float) - yhat).squeeze()
        return np.array([err, yhat], dtype=float).squeeze()

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
                 n_trials=50, n_iter=50, mode="cluster"):
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
            model = self._make_estimator(**params)
            scores = cross_val_score(model, X, y, cv=3,
                                     scoring="neg_mean_squared_error")
            return scores.mean()

        study = optuna.create_study(direction="maximize")
        study.optimize(objective, n_trials=self.n_trials, show_progress_bar=False)
        return study.best_params

    def _optimize_random(self, X, y):
        rng = np.random.RandomState(42)
        dists = self._param_distributions()
        best_score, best = -np.inf, None
        for _ in range(self.n_iter):
            params = {n: d.rvs(random_state=rng) for n, d in dists.items()}
            try:
                model = self._make_estimator(**self._cast_params(params))
                score = cross_val_score(model, X, y, cv=3,
                                        scoring="neg_mean_squared_error").mean()
            except Exception:
                continue
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
        mask = np.isfinite(y_vec) & np.all(np.isfinite(X_mat), axis=-1)
        if np.sum(mask) < 10:  # need >= 10 valid time steps
            return nan_out

        X_clean, y_clean = X_mat[mask], y_vec[mask]
        try:
            if self.hyperparam_optimizer == "random":
                best = self._optimize_random(X_clean, y_clean)
            else:
                best = self._optimize_optuna(X_clean, y_clean)
            if best is None:
                return nan_out
            return np.array([float(best[name]) for name in self._param_names])
        except Exception:
            return nan_out

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
        predictor["T"] = predictand["T"]
        predictand_st = standardize_timeseries(predictand, clim_year_start, clim_year_end)
        names = self._param_names

        if self.mode == "grid":
            chunk_y = int(np.ceil(len(predictand_st.Y) / self.nb_cores))
            chunk_x = int(np.ceil(len(predictand_st.X) / self.nb_cores))
            p_st_chunked = predictand_st.chunk({"Y": chunk_y, "X": chunk_x})

            client = Client(n_workers=self.nb_cores, threads_per_worker=1)
            hp_array = xr.apply_ufunc(
                self._optimize_single_cell,
                p_st_chunked,
                predictor,
                input_core_dims=[("T",), ("T", "features")],
                output_core_dims=[("param",)],
                vectorize=True,
                dask="parallelized",
                output_dtypes=[float],
                dask_gufunc_kwargs={"output_sizes": {"param": len(names)}},
            )
            hp_array = hp_array.compute()
            client.close()

            hyper_ds = xr.Dataset(
                {n: hp_array.isel(param=i, drop=True) for i, n in enumerate(names)}
            )
            cluster_da = xr.where(~np.isnan(hyper_ds[names[0]]), 1, np.nan)
            return hyper_ds, cluster_da

        # ---- cluster mode ----
        print(f"{self.__class__.__name__}: cluster-wise optimization "
              f"({self.n_clusters} clusters)...")
        kmeans = KMeans(n_clusters=self.n_clusters)
        predictand_dropna = (
            predictand.to_dataframe().reset_index().dropna().drop(columns=["T"])
        )
        variable_column = predictand_dropna.columns[2]
        predictand_dropna["cluster"] = kmeans.fit_predict(
            predictand_dropna[[variable_column]]
        )

        df_unique = predictand_dropna.drop_duplicates(subset=["Y", "X"])
        dataset = df_unique.set_index(["Y", "X"]).to_xarray()
        mask = xr.where(~np.isnan(predictand.isel(T=0)), 1, np.nan)
        cluster_da = dataset["cluster"] * mask

        x1, x2 = xr.align(predictand_st, cluster_da, join="outer")
        clusters = np.unique(x2)
        clusters = clusters[~np.isnan(clusters)]
        cluster_da = x2

        hyper_maps = {n: xr.full_like(cluster_da, np.nan, dtype=float) for n in names}
        for clus in clusters:
            y_cluster = (
                x1.where(x2 == clus).mean(dim=["Y", "X"], skipna=True).dropna(dim="T")
            )
            if len(y_cluster["T"]) > 0:
                best = self._optimize_single_cell(
                    y_cluster.values, predictor.sel(T=y_cluster["T"]).values
                )
                for i, n in enumerate(names):
                    hyper_maps[n] = hyper_maps[n].where(cluster_da != clus, best[i])

        hyper_ds = xr.Dataset(hyper_maps)
        return hyper_ds, cluster_da

    def fit_predict(self, x, y, x_test, y_test, *param_values):
        """
        Fit the estimator on one grid cell and predict.

        ``param_values`` are the per-cell hyperparameters, in the order of
        ``self._param_names``. Returns ``[error, prediction]``.
        """
        params = dict(zip(self._param_names, param_values))
        if any(not np.isfinite(v) for v in params.values()):
            return np.array([np.nan, np.nan]).squeeze()

        model = self._make_estimator(**self._cast_params(params))
        mask = np.isfinite(y) & np.all(np.isfinite(x), axis=-1)

        if np.any(mask):
            model.fit(x[mask, :], y[mask])
            if x_test.ndim == 1:
                x_test = x_test.reshape(1, -1)
            preds = model.predict(x_test)
            preds[preds < 0] = 0
            error_ = y_test - preds
            return np.array([error_, preds]).squeeze()
        return np.array([np.nan, np.nan]).squeeze()

    def compute_model(self, X_train, y_train, X_test, y_test, hyperparams):
        """
        Fit & predict over the whole spatial domain, parallelised with Dask.

        ``hyperparams`` is the Dataset returned by ``compute_hyperparameters``
        (replaces Ridge's ``alpha`` map). Returns the prediction field
        (dims Y, X).
        """
        chunksize_x = int(np.round(len(y_train.get_index("X")) / self.nb_cores))
        chunksize_y = int(np.round(len(y_train.get_index("Y")) / self.nb_cores))

        X_train["T"] = y_train["T"]
        y_train = y_train.transpose("T", "Y", "X")
        X_test = X_test.squeeze()
        y_test = y_test.drop_vars("T").squeeze().transpose("Y", "X")

        y_train, hyperparams = xr.align(y_train, hyperparams, join="outer")
        y_test, hyperparams = xr.align(y_test, hyperparams, join="outer")

        names = self._param_names
        hp_chunked = [
            hyperparams[n].chunk({"Y": chunksize_y, "X": chunksize_x}) for n in names
        ]
        core_dims = [("T", "features"), ("T",), ("features",), ()] + [() for _ in names]

        client = Client(n_workers=self.nb_cores, threads_per_worker=1)
        result_da = xr.apply_ufunc(
            self.fit_predict,
            X_train,
            y_train.chunk({"Y": chunksize_y, "X": chunksize_x}),
            X_test,
            y_test.chunk({"Y": chunksize_y, "X": chunksize_x}),
            *hp_chunked,
            input_core_dims=core_dims,
            vectorize=True,
            dask="parallelized",
            output_core_dims=[("output",)],
            output_dtypes=["float"],
            dask_gufunc_kwargs={"output_sizes": {"output": 2}},
        )
        result_ = result_da.compute()
        client.close()
        return result_.isel(output=1)

    # ----- probability calculation (identical to WAS_Ridge_Model) ----
    @staticmethod
    def _ppf_terciles_from_code(dist_code, shape, loc, scale):
        """Tercile thresholds (T1, T2) from best-fit distribution parameters."""
        if np.isnan(dist_code):
            return np.nan, np.nan
        code = int(dist_code)
        try:
            if code == 1:
                return (norm.ppf(0.33, loc=loc, scale=scale),
                        norm.ppf(0.67, loc=loc, scale=scale))
            elif code == 2:
                return (lognorm.ppf(0.33, s=shape, loc=loc, scale=scale),
                        lognorm.ppf(0.67, s=shape, loc=loc, scale=scale))
            elif code == 3:
                return (expon.ppf(0.33, loc=loc, scale=scale),
                        expon.ppf(0.67, loc=loc, scale=scale))
            elif code == 4:
                return (gamma.ppf(0.33, a=shape, loc=loc, scale=scale),
                        gamma.ppf(0.67, a=shape, loc=loc, scale=scale))
            elif code == 5:
                return (weibull_min.ppf(0.33, c=shape, loc=loc, scale=scale),
                        weibull_min.ppf(0.67, c=shape, loc=loc, scale=scale))
            elif code == 6:
                return (t.ppf(0.33, df=shape, loc=loc, scale=scale),
                        t.ppf(0.67, df=shape, loc=loc, scale=scale))
            elif code == 7:
                return (poisson.ppf(0.33, mu=shape, loc=loc),
                        poisson.ppf(0.67, mu=shape, loc=loc))
            elif code == 8:
                return (nbinom.ppf(0.33, n=shape, p=scale, loc=loc),
                        nbinom.ppf(0.67, n=shape, p=scale, loc=loc))
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
    def calculate_tercile_probabilities_bestfit(best_guess, error_variance,
                                                T1, T2, dist_code, dof):
        """Tercile probabilities using the climatological best-fit family."""
        best_guess = np.asarray(best_guess, float)
        error_variance = np.asarray(error_variance, dtype=float)
        n_time = best_guess.size
        out = np.full((3, n_time), np.nan, float)

        if (np.all(np.isnan(best_guess)) or np.isnan(dist_code)
                or np.isnan(T1) or np.isnan(T2) or np.isnan(error_variance)):
            return out

        code = int(dist_code)

        if code == 1:
            error_std = np.sqrt(error_variance)
            out[0, :] = norm.cdf(T1, loc=best_guess, scale=error_std)
            out[1, :] = (norm.cdf(T2, loc=best_guess, scale=error_std)
                         - norm.cdf(T1, loc=best_guess, scale=error_std))
            out[2, :] = 1 - norm.cdf(T2, loc=best_guess, scale=error_std)

        elif code == 2:
            sigma = np.sqrt(np.log(1 + error_variance / (best_guess ** 2)))
            mu = np.log(best_guess) - sigma ** 2 / 2
            out[0, :] = lognorm.cdf(T1, s=sigma, scale=np.exp(mu))
            out[1, :] = (lognorm.cdf(T2, s=sigma, scale=np.exp(mu))
                         - lognorm.cdf(T1, s=sigma, scale=np.exp(mu)))
            out[2, :] = 1 - lognorm.cdf(T2, s=sigma, scale=np.exp(mu))

        elif code == 3:
            scale = np.sqrt(error_variance)
            c1 = expon.cdf(T1, loc=best_guess, scale=scale)
            c2 = expon.cdf(T2, loc=best_guess, scale=scale)
            out[0, :] = c1
            out[1, :] = c2 - c1
            out[2, :] = 1.0 - c2

        elif code == 4:
            alpha = (best_guess ** 2) / error_variance
            theta = error_variance / best_guess
            c1 = gamma.cdf(T1, a=alpha, scale=theta)
            c2 = gamma.cdf(T2, a=alpha, scale=theta)
            out[0, :] = c1
            out[1, :] = c2 - c1
            out[2, :] = 1.0 - c2

        elif code == 5:
            for i in range(n_time):
                M = best_guess[i]
                V = error_variance
                if V <= 0 or M <= 0:
                    out[:, i] = np.nan
                    continue
                k = fsolve(_WAS_NonNeural_Base.weibull_shape_solver, 2.0, args=(M, V))[0]
                if k <= 0:
                    out[:, i] = np.nan
                    continue
                lambda_scale = M / gamma_function(1 + 1 / k)
                c1 = weibull_min.cdf(T1, c=k, loc=0, scale=lambda_scale)
                c2 = weibull_min.cdf(T2, c=k, loc=0, scale=lambda_scale)
                out[0, i] = c1
                out[1, i] = c2 - c1
                out[2, i] = 1.0 - c2

        elif code == 6:
            if dof <= 2:
                out[:, :] = np.nan
            else:
                loc = best_guess
                scale = np.sqrt(error_variance * (dof - 2) / dof)
                c1 = t.cdf(T1, df=dof, loc=loc, scale=scale)
                c2 = t.cdf(T2, df=dof, loc=loc, scale=scale)
                out[0, :] = c1
                out[1, :] = c2 - c1
                out[2, :] = 1.0 - c2

        elif code == 7:
            mu = best_guess
            c1 = poisson.cdf(T1, mu=mu)
            c2 = poisson.cdf(T2, mu=mu)
            out[0, :] = c1
            out[1, :] = c2 - c1
            out[2, :] = 1.0 - c2

        elif code == 8:
            p = np.where(error_variance > best_guess,
                         best_guess / error_variance, np.nan)
            n = np.where(error_variance > best_guess,
                         (best_guess ** 2) / (error_variance - best_guess), np.nan)
            c1 = nbinom.cdf(T1, n=n, p=p)
            c2 = nbinom.cdf(T2, n=n, p=p)
            out[0, :] = c1
            out[1, :] = c2 - c1
            out[2, :] = 1.0 - c2

        else:
            raise ValueError(f"Invalid distribution code: {dist_code}")

        return out

    @staticmethod
    def calculate_tercile_probabilities_nonparametric(best_guess, error_samples,
                                                      first_tercile, second_tercile):
        """Non-parametric method using historical error samples."""
        n_time = len(best_guess)
        pred_prob = np.full((3, n_time), np.nan, dtype=float)
        for tt in range(n_time):
            if np.isnan(best_guess[tt]):
                continue
            dist = best_guess[tt] + error_samples
            dist = dist[np.isfinite(dist)]
            if len(dist) == 0:
                continue
            p_below = np.mean(dist < first_tercile)
            p_between = np.mean((dist >= first_tercile) & (dist < second_tercile))
            p_above = 1.0 - (p_below + p_between)
            pred_prob[0, tt] = p_below
            pred_prob[1, tt] = p_between
            pred_prob[2, tt] = p_above
        return pred_prob

    def compute_prob(self, Predictant, clim_year_start, clim_year_end, hindcast_det,
                     best_code_da=None, best_shape_da=None,
                     best_loc_da=None, best_scale_da=None):
        """Tercile probabilities for deterministic hindcasts (identical to Ridge)."""
        if "M" in Predictant.dims:
            Predictant = Predictant.isel(M=0).drop_vars("M").squeeze()

        Predictant = Predictant.transpose("T", "Y", "X")
        mask = xr.where(~np.isnan(Predictant.isel(T=0)), 1.0, np.nan)

        clim = Predictant.sel(T=slice(str(clim_year_start), str(clim_year_end)))
        if clim.sizes.get("T", 0) < 3:
            raise ValueError("Not enough years in climatology period for terciles.")

        error_variance = (Predictant - hindcast_det).var(dim="T")
        dof = max(int(clim.sizes["T"]) - 1, 2)

        terciles_emp = clim.quantile([0.33, 0.67], dim="T")
        T1_emp = terciles_emp.isel(quantile=0).drop_vars("quantile")
        T2_emp = terciles_emp.isel(quantile=1).drop_vars("quantile")

        dm = self.dist_method
        if dm == "bestfit":
            if any(v is None for v in (best_code_da, best_shape_da,
                                       best_loc_da, best_scale_da)):
                raise ValueError("dist_method='bestfit' requires best_code_da, "
                                 "best_shape_da, best_loc_da, best_scale_da.")
            T1, T2 = xr.apply_ufunc(
                self._ppf_terciles_from_code,
                best_code_da, best_shape_da, best_loc_da, best_scale_da,
                input_core_dims=[(), (), (), ()],
                output_core_dims=[(), ()],
                vectorize=True, dask="parallelized",
                output_dtypes=[float, float],
            )
            hindcast_prob = xr.apply_ufunc(
                self.calculate_tercile_probabilities_bestfit,
                hindcast_det, error_variance, T1, T2, best_code_da,
                input_core_dims=[("T",), (), (), (), ()],
                output_core_dims=[("probability", "T")],
                vectorize=True, kwargs={"dof": dof}, dask="parallelized",
                output_dtypes=[float],
                dask_gufunc_kwargs={"output_sizes": {"probability": 3},
                                    "allow_rechunk": True},
            )
        elif dm == "nonparam":
            error_samples = Predictant - hindcast_det
            hindcast_prob = xr.apply_ufunc(
                self.calculate_tercile_probabilities_nonparametric,
                hindcast_det, error_samples, T1_emp, T2_emp,
                input_core_dims=[("T",), ("T",), (), ()],
                output_core_dims=[("probability", "T")],
                exclude_dims=set(("T",)),
                vectorize=True, dask="parallelized",
                output_dtypes=[float],
                dask_gufunc_kwargs={"output_sizes": {"probability": 3},
                                    "allow_rechunk": True},
            )
        else:
            raise ValueError(f"Invalid dist_method: {self.dist_method}")

        hindcast_prob = hindcast_prob.assign_coords(
            probability=("probability", ["PB", "PN", "PA"])
        )
        return (hindcast_prob * mask).transpose("probability", "T", "Y", "X")

    def forecast(self, Predictant, clim_year_start, clim_year_end, Predictor,
                 hindcast_det, Predictor_for_year, hyperparams,
                 best_code_da=None, best_shape_da=None,
                 best_loc_da=None, best_scale_da=None):
        """End-to-end forecast for a single target year (deterministic + terciles)."""
        y_test_dummy = xr.full_like(Predictant.isel(T=0), np.nan)

        chunksize_x = int(np.round(len(Predictant.get_index("X")) / self.nb_cores))
        chunksize_y = int(np.round(len(Predictant.get_index("Y")) / self.nb_cores))

        Predictor["T"] = Predictant["T"]
        Predictant = Predictant.transpose("T", "Y", "X")
        Predictor_for_year_ = Predictor_for_year.squeeze()
        Predictant_st = standardize_timeseries(Predictant, clim_year_start, clim_year_end)
        Predictant_st, hyperparams = xr.align(Predictant_st, hyperparams, join="outer")

        names = self._param_names
        hp_chunked = [
            hyperparams[n].chunk({"Y": chunksize_y, "X": chunksize_x}) for n in names
        ]
        core_dims = [("T", "features"), ("T",), ("features",), ()] + [() for _ in names]

        client = Client(n_workers=self.nb_cores, threads_per_worker=1)
        result_da = xr.apply_ufunc(
            self.fit_predict,
            Predictor,
            Predictant_st.chunk({"Y": chunksize_y, "X": chunksize_x}),
            Predictor_for_year_,
            y_test_dummy.chunk({"Y": chunksize_y, "X": chunksize_x}),
            *hp_chunked,
            input_core_dims=core_dims,
            vectorize=True,
            dask="parallelized",
            output_core_dims=[("output",)],
            output_dtypes=["float"],
            dask_gufunc_kwargs={"output_sizes": {"output": 2}},
        )
        result_ = result_da.compute()
        client.close()
        result_ = result_.isel(output=1)
        result_ = reverse_standardize(result_, Predictant,
                                      clim_year_start, clim_year_end)

        index_start = Predictant.get_index("T").get_loc(str(clim_year_start)).start
        index_end = Predictant.get_index("T").get_loc(str(clim_year_end)).stop
        rainfall_for_tercile = Predictant.isel(T=slice(index_start, index_end))
        terciles = rainfall_for_tercile.quantile([0.33, 0.67], dim="T")
        T1_emp = terciles.isel(quantile=0).drop_vars("quantile")
        T2_emp = terciles.isel(quantile=1).drop_vars("quantile")
        error_variance = (Predictant - hindcast_det).var(dim="T")

        forecast_expanded = result_.expand_dims(
            T=[pd.Timestamp(Predictor_for_year.coords["T"].values[0]).to_pydatetime()]
        )
        year = Predictor_for_year.coords["T"].values[0].astype("datetime64[Y]").astype(int) + 1970
        T_value_1 = Predictant.isel(T=0).coords["T"].values
        month_1 = T_value_1.astype("datetime64[M]").astype(int) % 12 + 1
        new_T_value = np.datetime64(f"{year}-{month_1:02d}-{1:02d}")

        forecast_expanded = forecast_expanded.assign_coords(
            T=xr.DataArray([new_T_value], dims=["T"])
        )
        forecast_expanded["T"] = forecast_expanded["T"].astype("datetime64[ns]")

        dof = max(int(rainfall_for_tercile.sizes["T"]) - 1, 2)
        dm = self.dist_method

        if dm == "bestfit":
            if any(v is None for v in (best_code_da, best_shape_da,
                                       best_loc_da, best_scale_da)):
                raise ValueError("dist_method='bestfit' requires best_code_da, "
                                 "best_shape_da, best_loc_da, best_scale_da.")
            T1, T2 = xr.apply_ufunc(
                self._ppf_terciles_from_code,
                best_code_da, best_shape_da, best_loc_da, best_scale_da,
                input_core_dims=[(), (), (), ()],
                output_core_dims=[(), ()],
                vectorize=True, dask="parallelized",
                output_dtypes=[float, float],
            )
            forecast_prob = xr.apply_ufunc(
                self.calculate_tercile_probabilities_bestfit,
                forecast_expanded, error_variance, T1, T2, best_code_da,
                input_core_dims=[("T",), (), (), (), ()],
                output_core_dims=[("probability", "T")],
                vectorize=True, dask="parallelized", kwargs={"dof": dof},
                output_dtypes=[float],
                dask_gufunc_kwargs={"output_sizes": {"probability": 3},
                                    "allow_rechunk": True},
            )
        elif dm == "nonparam":
            error_samples = Predictant - hindcast_det
            forecast_prob = xr.apply_ufunc(
                self.calculate_tercile_probabilities_nonparametric,
                forecast_expanded, error_samples, T1_emp, T2_emp,
                input_core_dims=[("T",), ("T",), (), ()],
                output_core_dims=[("probability", "T")],
                exclude_dims=set(("T",)),
                vectorize=True, dask="parallelized",
                output_dtypes=[float],
                dask_gufunc_kwargs={"output_sizes": {"probability": 3},
                                    "allow_rechunk": True},
            )
        else:
            raise ValueError(f"Invalid dist_method: {self.dist_method}")

        forecast_prob = forecast_prob.assign_coords(
            probability=("probability", ["PB", "PN", "PA"])
        )
        return forecast_expanded, forecast_prob.transpose("probability", "T", "Y", "X")


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
        return RandomForestRegressor(random_state=42, n_jobs=1, **params)


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
        return ExtraTreesRegressor(random_state=42, n_jobs=1, **params)


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
        return GradientBoostingRegressor(random_state=42, **params)


class WAS_AdaBoost_Model(_WAS_NonNeural_Base):
    """AdaBoost regressor (sklearn)."""

    def _default_param_space(self):
        return {
            "n_estimators":  (50, 500, True, False),
            "learning_rate": (1e-3, 1.0, False, True),
        }

    def _make_estimator(self, **params):
        from sklearn.ensemble import AdaBoostRegressor
        return AdaBoostRegressor(random_state=42, **params)


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
            random_state=42, n_jobs=1, verbosity=0,
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
        return LGBMRegressor(random_state=42, n_jobs=1, verbosity=-1, **params)


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
    ):
        self.n_clusters = int(n_clusters)
        self.nb_cores = int(nb_cores)
        self.hyperparam_optimizer = hyperparam_optimizer
        self.n_trials = int(n_trials)
        self.n_iter = int(n_iter)
        self.scoring = scoring
        self.random_state = int(random_state)

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
    def _optimize_optuna(self, X: np.ndarray, y: np.ndarray) -> Optional[dict]:
        def objective(trial):
            params = self._cast_params(self._suggest_params(trial))
            model = self._make_estimator(**params)
            scores = cross_val_score(
                model, X, y, cv=3, scoring=self.scoring
            )
            return float(scores.mean())

        study = optuna.create_study(
            direction="maximize",
            sampler=optuna.samplers.TPESampler(seed=self.random_state),
        )
        study.optimize(objective, n_trials=self.n_trials, show_progress_bar=False)
        return study.best_params

    def _optimize_random(self, X: np.ndarray, y: np.ndarray) -> Optional[dict]:
        rng = np.random.RandomState(self.random_state)
        dists = self._param_distributions()
        best_score, best = -np.inf, None
        for _ in range(self.n_iter):
            params = {n: d.rvs(random_state=rng) for n, d in dists.items()}
            try:
                model = self._make_estimator(**self._cast_params(params))
                score = float(
                    cross_val_score(model, X, y, cv=3, scoring=self.scoring).mean()
                )
            except Exception:
                continue
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
        if np.sum(mask) < 10:
            return nan_out
        X_c = X_mat[mask]
        y_c = y_vec[mask].astype(int)
        if len(np.unique(y_c)) < 2:
            return nan_out

        try:
            if self.hyperparam_optimizer == "random":
                best = self._optimize_random(X_c, y_c)
            else:
                best = self._optimize_optuna(X_c, y_c)
            if best is None:
                return nan_out
            return np.array([float(best[n]) for n in self._param_names])
        except Exception:
            return nan_out

    # ------------------------------------------------------------------
    # 1) Tercile classification
    # ------------------------------------------------------------------
    @staticmethod
    def _classify_pixel(y: np.ndarray, index_start: int, index_end: int):
        """Convert a 1-D continuous series to tercile classes {0, 1, 2}."""
        mask = np.isfinite(y)
        if np.any(mask):
            terciles = np.nanpercentile(y[index_start:index_end], [33, 67])
            y_class = np.digitize(y, bins=terciles, right=True)
            return y_class.astype(float), terciles[0], terciles[1]
        return np.full(y.shape[0], np.nan), np.nan, np.nan

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
        index_start = Predictant.get_index("T").get_loc(str(clim_year_start)).start
        index_end   = Predictant.get_index("T").get_loc(str(clim_year_end)).stop

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

    def _build_cluster_map(self, predictand: xr.DataArray) -> xr.DataArray:
        """KMeans on climatological mean; NaN cells stay NaN."""
        field = predictand.mean("T", skipna=True)
        flat  = field.values.reshape(-1)
        valid = np.isfinite(flat)

        labels = np.full(flat.shape, np.nan)
        if np.any(valid):
            km = KMeans(
                n_clusters=self.n_clusters,
                random_state=self.random_state,
                n_init=10,
            )
            labels[valid] = km.fit_predict(flat[valid].reshape(-1, 1)).astype(float)

        cluster_2d = labels.reshape(field.values.shape)
        cluster_da = xr.DataArray(
            cluster_2d, coords=field.coords, dims=field.dims, name="cluster"
        )
        return cluster_da.where(np.isfinite(field))

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
        predictor["T"] = predictand["T"]

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
        return hyper_ds, cluster_da

    # ------------------------------------------------------------------
    # 3) Per-pixel fit + predict_proba
    # ------------------------------------------------------------------
    def fit_predict(self, x, y, x_test, *param_values):
        """
        Fit the classifier on one grid cell and return class probabilities.

        Parameters
        ----------
        x            : (T, features) — training predictors
        y            : (T,)          — training labels (0/1/2, possibly with NaN)
        x_test       : (features,)   — predictor for the target time step
        *param_values : scalars      — per-cell hyperparameters in _param_names order

        Returns
        -------
        np.ndarray shape (3,) : [P(Below), P(Normal), P(Above)]
        """
        params = dict(zip(self._param_names, param_values))
        if any(not np.isfinite(v) for v in params.values()):
            return np.full(3, np.nan)

        est  = self._make_estimator(**self._cast_params(params))
        mask = np.isfinite(y) & np.all(np.isfinite(x), axis=-1)
        if not np.any(mask):
            return np.full(3, np.nan)

        x_c = x[mask, :]
        y_c = y[mask].astype(int)
        uniq = np.unique(y_c)

        # Degenerate: only one class present in training window
        if uniq.size < 2:
            out = np.zeros(3)
            out[int(uniq[0])] = 1.0
            return out

        try:
            est.fit(x_c, y_c)
        except Exception:
            return np.full(3, np.nan)

        if x_test.ndim == 1:
            x_test = x_test.reshape(1, -1)

        try:
            proba = est.predict_proba(x_test).reshape(-1)
        except Exception:
            return np.full(3, np.nan)

        # Safe classes_ extraction through optional Pipeline wrapper
        clf = est
        while isinstance(clf, Pipeline):
            clf = clf.steps[-1][1]
        classes = getattr(clf, "classes_", np.arange(len(proba)))

        # Map variable classes_ layout → fixed positions {0, 1, 2}.
        # Absent classes (not seen in training) receive probability 0, not NaN,
        # so the output always sums to 1 over the classes that were trained on.
        out = np.zeros(3)
        for cls, p in zip(classes, proba):
            if 0 <= int(cls) <= 2:
                out[int(cls)] = p
        return out

    # ------------------------------------------------------------------
    # 4) Parallel spatial classification (hindcast)
    # ------------------------------------------------------------------
    def compute_model(
        self,
        X_train: xr.DataArray,
        y_class: xr.DataArray,
        X_test: xr.DataArray,
        hyperparams: xr.Dataset,
    ) -> xr.DataArray:
        """
        Classify every grid cell in parallel over the full hindcast period.

        Parameters
        ----------
        X_train    : xr.DataArray (T, features) — historical predictors
        y_class    : xr.DataArray (T, Y, X)     — tercile class labels {0,1,2}
                     as returned by ``compute_class``
        X_test     : xr.DataArray (T, features) or (features,)
                     Predictors for the target time step(s)
        hyperparams: xr.Dataset from ``compute_hyperparameters``

        Returns
        -------
        xr.DataArray  dims (probability=['PB','PN','PA'], T, Y, X)
        """
        chunk_y = self._safe_chunk(len(y_class.get_index("Y")))
        chunk_x = self._safe_chunk(len(y_class.get_index("X")))

        X_train = X_train.copy()
        X_train["T"] = y_class["T"]
        y_class = y_class.transpose("T", "Y", "X")

        # Normalise X_test to (T, features) and remember the T coordinate so
        # it can be restored after apply_ufunc.  Do NOT use a bare .squeeze()
        # here: when T=1 that would silently drop the T dimension, causing
        # apply_ufunc to omit T from the output and the final transpose to fail.
        X_test = X_test.transpose("T", "features")
        n_test_steps  = X_test.sizes["T"]
        T_test_coords = X_test["T"].values

        if n_test_steps == 1:
            # Pass a single (features,) vector to fit_predict; T will be
            # re-attached via expand_dims after computation.
            X_test_fp = X_test.isel(T=0)
        else:
            X_test_fp = X_test

        y_class, hyperparams = xr.align(y_class, hyperparams, join="outer")

        names     = self._param_names
        hp_chunks = [
            hyperparams[n].chunk({"Y": chunk_y, "X": chunk_x}) for n in names
        ]
        core_dims = (
            [("T", "features"), ("T",), ("features",)]
            + [() for _ in names]
        )

        client = (
            Client(n_workers=self.nb_cores, threads_per_worker=1)
            if self.nb_cores > 1
            else None
        )
        try:
            result = xr.apply_ufunc(
                self.fit_predict,
                X_train,
                y_class.chunk({"Y": chunk_y, "X": chunk_x}),
                X_test_fp,
                *hp_chunks,
                input_core_dims=core_dims,
                output_core_dims=[("probability",)],
                vectorize=True,
                dask="parallelized",
                output_dtypes=["float"],
                dask_gufunc_kwargs={"output_sizes": {"probability": 3}},
            )
            result_ = result.compute() if hasattr(result.data, "compute") else result
        finally:
            if client is not None:
                client.close()

        result_ = result_.assign_coords(
            probability=("probability", ["PB", "PN", "PA"])
        )

        # Restore the T dimension when it was collapsed by isel above
        if "T" not in result_.dims:
            result_ = result_.expand_dims(T=T_test_coords)

        return result_.transpose("probability", "T", "Y", "X")

    # ------------------------------------------------------------------
    # 5) End-to-end single-year forecast
    # ------------------------------------------------------------------
    def forecast(
        self,
        Predictant: xr.DataArray,
        clim_year_start: int,
        clim_year_end: int,
        Predictor: xr.DataArray,
        Predictor_for_year: xr.DataArray,
        hyperparams: xr.Dataset,
    ) -> xr.DataArray:
        """
        Classify → fit on full history → predict_proba for one target year.

        Parameters
        ----------
        Predictant         : xr.DataArray (T, Y, X) — observed continuous predictand
        clim_year_start    : int — first year of climatology
        clim_year_end      : int — last  year of climatology
        Predictor          : xr.DataArray (T, features) — historical predictors
        Predictor_for_year : xr.DataArray (1, features) or (features,)
        hyperparams        : xr.Dataset from ``compute_hyperparameters``

        Returns
        -------
        xr.DataArray  dims (probability=['PB','PN','PA'], T=1, Y, X)
        """
        # 1) Classify continuous predictand → integer tercile class (T, Y, X)
        y_class, _, _ = self.compute_class(Predictant, clim_year_start, clim_year_end)

        # 2) Align predictor T axis to match class array
        Predictor = Predictor.copy()
        Predictor["T"] = y_class["T"]

        # 3) Ensure forecast predictor has a T dimension
        X_test = Predictor_for_year
        if "T" not in X_test.dims:
            if (
                "T" in Predictor_for_year.coords
                and Predictor_for_year.coords["T"].size > 0
            ):
                t0 = pd.Timestamp(
                    Predictor_for_year.coords["T"].values[0]
                ).to_datetime64()
            else:
                t0 = pd.Timestamp(Predictor["T"].values[-1]).to_datetime64()
            X_test = X_test.expand_dims(T=[t0])

        chunk_y = self._safe_chunk(len(y_class.get_index("Y")))
        chunk_x = self._safe_chunk(len(y_class.get_index("X")))

        y_class, hyperparams = xr.align(y_class, hyperparams, join="outer")

        names     = self._param_names
        hp_chunks = [
            hyperparams[n].chunk({"Y": chunk_y, "X": chunk_x}) for n in names
        ]
        core_dims = (
            [("T", "features"), ("T",), ("features",)]
            + [() for _ in names]
        )

        X_test_sq = X_test.transpose("T", "features").squeeze()

        client = (
            Client(n_workers=self.nb_cores, threads_per_worker=1)
            if self.nb_cores > 1
            else None
        )
        try:
            proba = xr.apply_ufunc(
                self.fit_predict,
                Predictor,
                y_class.chunk({"Y": chunk_y, "X": chunk_x}),
                X_test_sq,
                *hp_chunks,
                input_core_dims=core_dims,
                output_core_dims=[("probability",)],
                vectorize=True,
                dask="parallelized",
                output_dtypes=["float"],
                dask_gufunc_kwargs={"output_sizes": {"probability": 3}},
            )
            proba_ = proba.compute() if hasattr(proba.data, "compute") else proba
        finally:
            if client is not None:
                client.close()

        proba_ = proba_.assign_coords(
            probability=("probability", ["PB", "PN", "PA"])
        )
        if "T" not in proba_.dims:
            proba_ = proba_.expand_dims(T=X_test["T"].values)

        return proba_.transpose("probability", "T", "Y", "X")


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
      neighbours), providing well-calibrated tercile probabilities.
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
    - Platt scaling adds a negligible overhead for seasonal grid sizes.
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
      well-calibrated without additional post-processing.
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
    - ``num_class`` is fixed to 3 (Below / Normal / Above).
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

        return XGBClassifier(
            objective="multi:softprob",
            num_class=3,
            eval_metric="mlogloss",
            tree_method="hist",
            random_state=self.random_state,
            n_jobs=1,
            verbosity=0,
            use_label_encoder=False,
            **params,
        )


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
      well-calibrated without additional post-processing.
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
    - ``objective='multiclass'`` + ``num_class=3`` gives calibrated soft-max
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
            objective="multiclass",
            num_class=3,
            class_weight="balanced",
            random_state=self.random_state,
            n_jobs=1,
            verbosity=-1,
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
    - Naturally calibrated: posterior probabilities come directly from the
      Gaussian discriminant model.
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
    covariance, interpolating between full QDA (``reg_param=0``) and LDA
    (``reg_param=1``).

    Hyperparameters tuned
    ---------------------
    reg_param : float  in [0.0, 1.0]  (linear scale) — covariance regularisation

    Notes
    -----
    - Auto-scaled via StandardScaler for numerical stability in covariance
      estimation.
    - When ``reg_param`` is tuned close to 1.0 the model effectively
      degenerates to LDA; the HPO will choose the value that best calibrates
      the log-loss on the cluster's class series.
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

