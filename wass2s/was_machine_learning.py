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
    2. **Spatial clustering** (KMeans) on a summary statistic of the predictand (default: mean over time).
    3. **Per-cluster hyperparameter optimization** of LogisticRegression using grid/random/Bayesian search.
    4. **Broadcast best hyperparameters** to every grid cell (Y, X).
    5. **Parallel per-grid-cell classification** using local hyperparameters.
    6. **Direct probabilistic output**: model.predict_proba() gives [P(Below), P(Normal), P(Above)]
       per grid cell and time step.

    This class is designed for **probabilistic seasonal forecasting** in tercile format,
    where the target is a categorical variable derived from climatological thresholds.

    Key features:
    - Multiclass logistic regression (multinomial + L2 penalty via 'lbfgs' solver)
    - Optional feature scaling (x_scaler: None | 'standard' | 'robust')
    - No target scaling (y is categorical 0/1/2)
    - Safe handling of class imbalance via 'class_weight' tuning
    - Full parallelization across spatial grid using dask

    Parameters
    ----------
    nb_cores : int, default=1
        Number of CPU cores for parallel processing (dask + joblib).

    dist_method : {'bestfit', 'nonparam'}, default='nonparam'
        Method for computing tercile probabilities (currently unused in this class,
        kept for API consistency with other models).

    n_clusters : int, default=5
        Number of spatial clusters for KMeans grouping of the predictand field.

    param_grid : dict or None, default=None
        Hyperparameter search space for LogisticRegression.
        If None, a safe default grid is used (compatible with multinomial + lbfgs):
            - C: [0.1, 0.5, 1.0, 2.0, 5.0]                # inverse regularization strength
            - class_weight: [None, 'balanced']
            - max_iter: [300, 600, 1000]

    optimization_method : {'grid', 'random', 'bayesian'}, default='grid'
        Strategy for hyperparameter search.

    n_trials : int, default=20
        Number of trials for 'random' or 'bayesian' optimization.

    cv : int, default=5
        Number of cross-validation folds during tuning.

    random_state : int, default=42
        Random seed for reproducibility (KMeans, CV splits, optimizer).

    x_scaler : {None, 'standard', 'robust'}, default=None
        Whether and how to scale input features (X):
        - None: no scaling (default, logistic regression is scale-invariant)
        - 'standard': StandardScaler (zero mean, unit variance)
        - 'robust': RobustScaler (median-centered, IQR-scaled, robust to outliers)

    Attributes
    ----------
    nb_cores, dist_method, n_clusters, param_grid, optimization_method, n_trials, cv, random_state, x_scaler
        Stored initialization parameters.

    optimizer : BaseOptimizer
        Internal hyperparameter optimization object.

    _cw_map / _cw_inv : dict
        Internal mapping for broadcasting 'class_weight' (None ↔ 0, 'balanced' ↔ 1).

    Methods
    -------
    classify(y, index_start, index_end)
        Static method: convert continuous values to tercile classes (0/1/2).

    compute_class(Predictant, clim_year_start, clim_year_end)
        Compute tercile class map (T, Y, X) + climatological terciles.

    compute_hyperparameters(predictand, predictor, clim_year_start, clim_year_end, scoring='neg_log_loss')
        Cluster predictand → optimize logistic hyperparameters per cluster →
        return broadcast arrays of best C, class_weight (coded), max_iter, solver.

    fit_predict(x, y, x_test, C, cw_code, max_iter, solver)
        Fit logistic model on one grid cell using local hyperparameters and return
        probability vector [P(Below), P(Normal), P(Above)].

    compute_model(X_train, y_train, X_test, C_da, cw_code_da, maxiter_da, solver_da)
        Parallel classification across entire spatial domain using local best params.
        Returns xarray.DataArray with dims (probability=['PB','PN','PA'], T, Y, X).

    forecast(Predictant, clim_year_start, clim_year_end, Predictor, Predictor_for_year,
             C_da, cw_code_da, maxiter_da, solver_da)
        Full end-to-end forecast pipeline for one target year:
        - classify historical data
        - optimize hyperparameters
        - predict probabilities for the forecast year

    Notes
    -----
    - Input predictand should be continuous seasonal values (e.g., total rainfall, anomaly).
    - Target y is internally converted to integer classes {0, 1, 2} = {Below, Normal, Above}.
    - Negative predictions are **not** clipped (logistic outputs probabilities 0–1).
    - Large domains benefit greatly from higher `nb_cores`.
    - For very small clusters or poor separability, model may fall back to uniform probabilities.
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
        x_scaler=None,   # None | "standard" | "robust"
    ):
        self.nb_cores = int(nb_cores)
        self.dist_method = dist_method
        self.n_clusters = int(n_clusters)
        self.optimization_method = optimization_method
        self.n_trials = int(n_trials)
        self.cv = int(cv)
        self.random_state = int(random_state)
        self.x_scaler = x_scaler

        # Default search space (SAFE with multinomial + lbfgs: penalty must be 'l2')
        
        if param_grid is None:
            self.param_grid = {
                "C": [0.1, 0.5, 1.0, 2.0, 5.0],
                "class_weight": [None, "balanced"],
                "max_iter": [300, 600, 1000],
                # keep solver fixed to avoid invalid combos
                "solver": ["lbfgs"],
            }
        else:
            self.param_grid = param_grid

        self.optimizer = BaseOptimizer(
            optimization_method=optimization_method,
            n_trials=n_trials,
            cv=cv,
            random_state=random_state
        )

        # Encodings for broadcast arrays (avoid storing dict strings per cell)
        self._cw_map = {None: 0, "balanced": 1}
        self._cw_inv = {0: None, 1: "balanced"}

    # ------------------------------------------------------------------
    # 0) Helpers
    # ------------------------------------------------------------------
    def _safe_chunk_size(self, n: int) -> int:
        return max(int(np.ceil(n / max(self.nb_cores, 1))), 1)

    def _make_estimator(self, C=1.0, class_weight=None, max_iter=500, solver="lbfgs"):
        clf = LogisticRegression(
            C=float(C),
            class_weight=class_weight,
            max_iter=int(max_iter),
            solver=solver,
            multi_class="multinomial",
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
        """Return mode over {0,1,2} for a 1D array, ignoring NaN."""
        v = v[np.isfinite(v)]
        if v.size == 0:
            return np.nan
        v = v.astype(int)
        counts = np.bincount(v, minlength=3)
        return int(np.argmax(counts))

    # ------------------------------------------------------------------
    # 1) Tercile classification
    # ------------------------------------------------------------------
    @staticmethod
    def classify(y, index_start, index_end):
        mask = np.isfinite(y)
        if np.any(mask):
            terciles = np.nanpercentile(y[index_start:index_end], [33, 67])
            y_class = np.digitize(y, bins=terciles, right=True)  # 0/1/2
            return y_class, terciles[0], terciles[1]
        return np.full(y.shape[0], np.nan), np.nan, np.nan

    def compute_class(self, Predictant: xr.DataArray, clim_year_start: int, clim_year_end: int):

        index_start = Predictant.get_index("T").get_loc(str(clim_year_start)).start
        index_end = Predictant.get_index("T").get_loc(str(clim_year_end)).stop

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
    # 2) Clustering (spatial) and HPO per cluster
    # ------------------------------------------------------------------
    def _build_cluster_map(self, predictand: xr.DataArray) -> xr.DataArray:
        """
        KMeans on a spatial statistic (default: climatological mean over T).
        Produces cluster_da with dims (Y,X) and NaN where predictand is NaN.
        """
        field = predictand.mean("T", skipna=True)
        vals = field.values
        flat = vals.reshape(-1)
        valid = np.isfinite(flat)

        labels = np.full(flat.shape, np.nan)
        if np.any(valid):
            km = KMeans(n_clusters=self.n_clusters, random_state=self.random_state)
            labels_valid = km.fit_predict(flat[valid].reshape(-1, 1)).astype(float)
            labels[valid] = labels_valid

        cluster_2d = labels.reshape(vals.shape)
        cluster_da = xr.DataArray(cluster_2d, coords=field.coords, dims=field.dims, name="cluster")

        # Ensure NaNs align with predictand mask
        cluster_da = cluster_da.where(np.isfinite(field))
        return cluster_da

    def compute_hyperparameters(
        self,
        predictand: xr.DataArray,
        predictor: xr.DataArray,
        clim_year_start: int,
        clim_year_end: int,
        scoring: str = "neg_log_loss",
    ):
        """
        Returns broadcast hyperparameter arrays + cluster map.
        
        """
        predictor['T'] = predictand['T']
        # (a) classify predictand into tercile classes
        y_class, _, _ = self.compute_class(predictand, clim_year_start, clim_year_end)

        # (b) build clusters in space
        cluster_da = self._build_cluster_map(predictand)
        _, cluster_da = xr.align(predictand.isel(T=0), cluster_da, join="outer")

        clusters = np.unique(cluster_da.values)
        clusters = clusters[np.isfinite(clusters)]

        best_params_for_cluster = {}

        for c in clusters:
            mask_c = (cluster_da == c)

            # Build a cluster-level time series label: spatial mode over (Y,X) at each T
            y_stack = y_class.where(mask_c).stack(Z=("Y", "X"))
            y_mode = xr.apply_ufunc(
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

            # Align predictors on the same times
            X_c = predictor.sel(T=y_mode["T"])
            X_mat = X_c.values
            y_vec = y_mode.values.astype(int)

            # Optimize hyperparameters for this cluster
            base_est = self._make_estimator()
            bp = self.optimizer.optimize(
                base_est,
                self.param_grid,
                X_mat,
                y_vec,
                scoring=scoring
            )
            best_params_for_cluster[int(c)] = bp

        # (c) broadcast best params to each (Y,X)
        C_da = xr.full_like(cluster_da, np.nan, dtype=float)
        cw_code_da = xr.full_like(cluster_da, np.nan, dtype=float)
        maxiter_da = xr.full_like(cluster_da, np.nan, dtype=float)
        solver_da = xr.full_like(cluster_da, np.nan, dtype=object)

        for c, bp in best_params_for_cluster.items():
            c_mask = (cluster_da == c)

            C_val = float(bp.get("C", 1.0))
            cw_val = bp.get("class_weight", None)
            mi_val = float(bp.get("max_iter", 500))
            sv_val = bp.get("solver", "lbfgs")

            C_da = C_da.where(~c_mask, other=C_val)
            cw_code_da = cw_code_da.where(~c_mask, other=float(self._cw_map.get(cw_val, 0)))
            maxiter_da = maxiter_da.where(~c_mask, other=mi_val)
            solver_da = solver_da.where(~c_mask, other=str(sv_val))

        return C_da, cw_code_da, maxiter_da, solver_da, cluster_da

    # ------------------------------------------------------------------
    # 3) Fit + predict at one grid cell using local/broadcast params
    # ------------------------------------------------------------------
    def fit_predict(self, x, y, x_test, C, cw_code, max_iter, solver):
        """
        Trains with local hyperparameters and returns proba for classes 0/1/2,
        correctly mapped via classes_ (fix vs positional padding). 
        """
        if not np.isfinite(C) or not np.isfinite(cw_code) or not np.isfinite(max_iter):
            return np.full((3,), np.nan)

        class_weight = self._cw_inv.get(int(cw_code), None)
        est = self._make_estimator(C=C, class_weight=class_weight, max_iter=max_iter, solver=str(solver))

        mask = np.isfinite(y) & np.all(np.isfinite(x), axis=-1)
        if not np.any(mask):
            return np.full((3,), np.nan)

        x_c = x[mask, :]
        y_c = y[mask].astype(int)

        uniq = np.unique(y_c)
        if uniq.size < 2:
            out = np.full(3, np.nan)
            out[int(uniq[0])] = 1.0
            return out

        est.fit(x_c, y_c)

        if x_test.ndim == 1:
            x_test = x_test.reshape(1, -1)

        proba = est.predict_proba(x_test).reshape(-1)

        # classes_ location differs if pipeline is used
        classes = getattr(est, "classes_", None)
        if classes is None and isinstance(est, Pipeline):
            classes = est.named_steps["logit"].classes_

        out = np.full(3, np.nan)
        for cls, p in zip(classes, proba):
            cls = int(cls)
            if 0 <= cls <= 2:
                out[cls] = p
        return out

    # ------------------------------------------------------------------
    # 4) Parallel model over grid with local params
    # ------------------------------------------------------------------
    def compute_model(self, X_train, y_train, X_test, C_da, cw_code_da, maxiter_da, solver_da):
        chunksize_x = self._safe_chunk_size(len(y_train.get_index("X")))
        chunksize_y = self._safe_chunk_size(len(y_train.get_index("Y")))

        X_train = X_train.copy()
        X_train["T"] = y_train["T"]
        y_train = y_train.transpose("T", "Y", "X")

        X_test = X_test.transpose("T", "features").squeeze()

        client = Client(n_workers=self.nb_cores, threads_per_worker=1) if self.nb_cores > 1 else None
        try:
            result = xr.apply_ufunc(
                self.fit_predict,
                X_train,
                y_train.chunk({"Y": chunksize_y, "X": chunksize_x}),
                X_test,
                C_da,
                cw_code_da,
                maxiter_da,
                solver_da,
                input_core_dims=[("T", "features"), ("T",), ("features",), (), (), (), ()],
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

        result_ = result_.assign_coords(probability=("probability", ["PB", "PN", "PA"]))
        return result_

    # ------------------------------------------------------------------
    # 5) Forecast (end-to-end): compute classes -> HPO -> local fit/predict
    # ------------------------------------------------------------------
    def forecast(
        self,
        Predictant: xr.DataArray,
        clim_year_start: int,
        clim_year_end: int,
        Predictor: xr.DataArray,
        Predictor_for_year: xr.DataArray,
        C_da, cw_code_da, maxiter_da, solver_da
    ):
        # 1) classify predictand
        y_class, _, _ = self.compute_class(Predictant, clim_year_start, clim_year_end)

        # 3) align T for training predictors
        Predictor = Predictor.copy()
        Predictor["T"] = y_class["T"]

        # 4) ensure forecast predictor has T
        X_test = Predictor_for_year
        if "T" not in X_test.dims:
            if "T" in Predictor_for_year.coords and Predictor_for_year.coords["T"].size > 0:
                t0 = pd.Timestamp(Predictor_for_year.coords["T"].values[0]).to_datetime64()
            else:
                t0 = pd.Timestamp(Predictor["T"].values[-1]).to_datetime64()
            X_test = X_test.expand_dims(T=[t0])
        
        client = Client(n_workers=self.nb_cores, threads_per_worker=1) if self.nb_cores > 1 else None
        try:
            proba = xr.apply_ufunc(
                self.fit_predict,
                Predictor,
                y_class.chunk({"Y": chunksize_y, "X": chunksize_x}),
                X_test,
                C_da,
                cw_code_da,
                maxiter_da,
                solver_da,
                input_core_dims=[("T", "features"), ("T",), ("features",), (), (), (), ()],
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

        # keep time dim for single forecast (T=1)
        if "T" not in proba.dims:
            proba_ = proba_.expand_dims(T=X_test["T"].values)

        return proba_.transpose("probability", "T", "Y", "X")


class WAS_PolynomialRegression:
    """
    Polynomial Regression model for spatiotemporal climate prediction.

    This class implements polynomial regression (via scikit-learn's PolynomialFeatures + LinearRegression)
    to capture non-linear relationships between predictors and a continuous predictand (e.g., seasonal rainfall,
    temperature anomalies, agro-climatic indices).

    Key features:
    - Polynomial feature expansion up to a specified degree
    - Spatially parallel fitting and prediction across large grids using dask + xarray
    - Deterministic point predictions with optional error computation
    - Probabilistic tercile forecasting (Below/Normal/Above = PB/PN/PA) using either:
      - Parametric best-fit distributions per grid cell ('bestfit')
      - Non-parametric sampling of historical forecast errors ('nonparam')

    Suitable for modeling moderate non-linearities where interpretability of polynomial terms is useful.

    Parameters
    -----------
    nb_cores : int, default=1
        Number of CPU cores for parallel processing (dask workers).

    degree : int, default=2
        Degree of the polynomial features (e.g., 2 = quadratic, 3 = cubic).

    dist_method : {'bestfit', 'nonparam'}, default='nonparam'
        Method for computing tercile probabilities:
        - 'bestfit'  → uses best-fit distribution per grid cell (requires distribution fit inputs)
        - 'nonparam' → empirical sampling of historical forecast errors

    Attributes
    -----------
    nb_cores, degree, dist_method
        Stored initialization parameters.

    Methods
    --------
    fit_predict(x, y, x_test, y_test)
        Fits polynomial regression on one grid cell and returns [error, prediction].

    compute_model(X_train, y_train, X_test, y_test)
        Parallel polynomial regression across the entire spatial domain.
        Returns xarray.DataArray with dims ('output'=['error','prediction'], Y, X).

    compute_prob(Predictant, clim_year_start, clim_year_end, hindcast_det,
                 best_code_da=None, best_shape_da=None, best_loc_da=None, best_scale_da=None)
        Computes tercile probabilities for deterministic hindcasts.

    forecast(Predictant, clim_year_start, clim_year_end, Predictor, hindcast_det,
             Predictor_for_year, best_code_da=None, best_shape_da=None,
             best_loc_da=None, best_scale_da=None)
        Full end-to-end forecast pipeline for one target year:
        - deterministic polynomial prediction
        - tercile probabilities (PB, PN, PA)

    Notes
    ------
    - **Input data requirements**:
      - Target (y) should be continuous (rainfall, temperature, etc.).
      - Predictors (X) should be continuous or properly encoded.
    - Negative predictions are automatically clipped to zero (useful for rainfall or non-negative variables).
    - Higher polynomial degrees can lead to overfitting, especially with small training sets per grid cell.
    - Large spatial domains benefit significantly from higher `nb_cores`.
    - For `dist_method='bestfit'`, distribution fit results from `WAS_TransformData` must be provided.
    - No explicit feature or target scaling is applied (linear regression after polynomial expansion is scale-sensitive;
      consider normalizing predictors externally if needed).

    Warnings
    ---------
    - High polynomial degrees (degree ≥ 4) with many features can cause numerical instability
      or extreme overfitting → use cautiously and validate.
    - Very skewed targets (e.g., heavy-tailed rainfall) may benefit from transformation
      (log, square-root) before modeling (not done automatically).
    - Small per-grid-cell training sets can lead to poor fits or unstable coefficients.
    """

    def __init__(self, nb_cores=1, degree=2, dist_method="nonparam"):
        """
        Initializes the WAS_PolynomialRegression with a specified number of CPU cores and polynomial degree.
        
        Parameters
        ----------
        nb_cores : int, optional
            Number of CPU cores to use for parallel computation, by default 1.
        degree : int, optional
            The degree of the polynomial, by default 2.
        dist_method : str, optional
            The method to compute tercile probabilities ("bestfit", "nonparam"), 
            by default "nonparam".
        """
        self.nb_cores = nb_cores
        self.degree = degree
        self.dist_method = dist_method

    def fit_predict(self, x, y, x_test, y_test):
        """
        Fit polynomial regression model and generate predictions.
        
        Parameters
        ----------
        x : array-like, shape (n_samples,) or (n_samples, n_features)
            Training feature data. For polynomial regression, this is typically 
            a 1D array of independent variable values.
        y : array-like, shape (n_samples,)
            Training target values (dependent variable).
        x_test : array-like, shape (n_test_samples,) or (n_test_samples, n_features)
            Test feature data for which to generate predictions.
        y_test : array-like, shape (n_test_samples,), optional
            Test target values used to calculate prediction error. Required for 
            error computation.
        degree : int
            Degree of the polynomial to fit. Must be a non-negative integer.
        
        Returns
        -------
        np.ndarray
            Array containing two elements:
            - error : float
                Prediction error metric (e.g., MSE, RMSE) computed between 
                predicted and actual y_test values.
            - prediction : np.ndarray, shape (n_test_samples,)
                Predicted values for x_test.
        
        Raises
        ------
        ValueError
            If x, y, x_test, or y_test have incompatible shapes.
            If degree is negative.
            If insufficient samples for the specified polynomial degree.
        
        Notes
        -----
        1. Polynomial features are created using np.polyfit for 1D data or 
           sklearn.preprocessing.PolynomialFeatures for multi-dimensional data.
        2. The error metric computed depends on implementation (typically 
           mean squared error or root mean squared error).
        3. For high-degree polynomials, consider regularization to prevent overfitting.
        """
        
        # Create a PolynomialFeatures transformer for the specified degree
        poly = PolynomialFeatures(degree=self.degree)
        model = LinearRegression()

        # Identify valid (finite) samples
        mask = np.isfinite(y) & np.all(np.isfinite(x), axis=-1)
        
        # If we have at least one valid sample, we can train a model
        if np.any(mask):
            y_clean = y[mask]
            x_clean = x[mask, :]

            # Transform x_clean into polynomial feature space
            x_clean_poly = poly.fit_transform(x_clean)
            model.fit(x_clean_poly, y_clean)

            # Reshape x_test if needed and transform it
            if x_test.ndim == 1:
                x_test = x_test.reshape(1, -1)
            x_test_poly = poly.transform(x_test)

            # Make predictions
            preds = model.predict(x_test_poly)

            preds[preds < 0] = 0

            # Compute prediction error
            error_ = y_test - preds
            return np.array([error_, preds]).squeeze()
        else:
            # If no valid data, return NaNs
            return np.array([np.nan, np.nan]).squeeze()

    def compute_model(self, X_train, y_train, X_test, y_test):
        """
        Computes predictions for spatiotemporal data using Polynomial Regression with parallel processing.

        Parameters
        ----------
        X_train : xarray.DataArray
            Training data (predictors) with dimensions (T, features).
            (It must be chunked properly in Dask, or at least be amenable to chunking.)
        y_train : xarray.DataArray
            Training target values with dimensions (T, Y, X).
        X_test : xarray.DataArray
            Test data (predictors) with dimensions (features,) or (T, features).
            Typically, you'd match time steps or have a single test.
        y_test : xarray.DataArray
            Test target values with dimensions (Y, X) or broadcastable to (T, Y, X).

        Returns
        -------
        xarray.DataArray
            An array with shape (2, Y, X) after computing, where the first index 
            is error and the second is the prediction.
        """
        # Determine chunk sizes so each worker handles a portion of the spatial domain
        chunksize_x = int(np.round(len(y_train.get_index("X")) / self.nb_cores))
        chunksize_y = int(np.round(len(y_train.get_index("Y")) / self.nb_cores))

        # Align time dimension: we want X_train and y_train to have the same 'T'
        # (We assume X_train has dimension (T, features) and y_train has dimension (T, Y, X))
        X_train['T'] = y_train['T']
        y_train = y_train.transpose('T', 'Y', 'X')

        # Squeeze X_test (if it has extra dims)
        # Usually, X_test would be (features,) or (T, features)
        X_test = X_test.squeeze()

        # y_test might have shape (Y, X) or (T, Y, X). 
        # If it's purely spatial, no 'T' dimension. We remove it if present.
        if 'T' in y_test.dims:
            y_test = y_test.drop_vars('T')
        y_test = y_test.squeeze().transpose('Y', 'X')

        # Create a Dask client for parallel processing
        client = Client(n_workers=self.nb_cores, threads_per_worker=1)

        # Apply `fit_predict` across each (Y,X) grid cell in parallel.
        
        result = xr.apply_ufunc(
            self.fit_predict,
            X_train,                                   # shape (T, features)
            y_train.chunk({'Y': chunksize_y, 'X': chunksize_x}),
            X_test,
            y_test.chunk({'Y': chunksize_y, 'X': chunksize_x}),
            input_core_dims=[('T', 'features'), ('T',), ('features',), ()],
            vectorize=True,
            output_core_dims=[('output',)],           # We'll have a new dim 'output' of size 2
            dask='parallelized',
            output_dtypes=['float'],
            dask_gufunc_kwargs={'output_sizes': {'output': 2}},
        )

        # Trigger computation
        result_ = result.compute()
        client.close()

        # Return an xarray.DataArray with dimension 'output' of size 2: [error, prediction]
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


    def forecast(self, Predictant, clim_year_start, clim_year_end, Predictor, hindcast_det, Predictor_for_year, best_code_da=None, best_shape_da=None, best_loc_da=None, best_scale_da=None):
        """
        Generate forecasts for a single time (e.g., future year) and compute 
        tercile probabilities based on the chosen distribution method.

        Parameters
        ----------
        Predictant : xarray.DataArray
            Target variable with dimensions (T, Y, X).
        clim_year_start : int
            Start year of climatology period.
        clim_year_end : int
            End year of climatology period.
        Predictor : xarray.DataArray
            Historical predictor data with dimensions (T, features).
        hindcast_det : xarray.DataArray
            Deterministic hindcast array that includes 'error' and 'prediction' over the historical period.
        Predictor_for_year : xarray.DataArray
            Predictor data for the forecast year, shape (features,) or (1, features).

        Returns
        -------
        tuple (result_, hindcast_prob)
            result_  : xarray.DataArray or numpy array with the forecast's [error, prediction].
            hindcast_prob : xarray.DataArray of shape (probability=3, Y, X) with PB, PN, and PA.
        """
        # Chunk sizes for parallel processing
        chunksize_x = int(np.round(len(Predictant.get_index("X")) / self.nb_cores))
        chunksize_y = int(np.round(len(Predictant.get_index("Y")) / self.nb_cores))

        # Align the time dimension
        Predictor['T'] = Predictant['T']
        Predictant_st = standardize_timeseries(Predictant, clim_year_start, clim_year_end) 
        Predictant_st = Predictant_st.transpose('T', 'Y', 'X')

        # Squeeze the forecast predictor data if needed
        Predictor_for_year_ = Predictor_for_year.squeeze()

        # We'll apply our polynomial regression in parallel across Y,X. 
        # Because we are forecasting a single point in time, y_test is unknown, so we omit it or set it to NaN.
        y_test = xr.full_like(Predictant.isel(T=0), np.nan)  # shape (Y,X)

        # Create a Dask client
        client = Client(n_workers=self.nb_cores, threads_per_worker=1)

        # Apply fit_predict to get the forecast for each grid cell 
        # We'll produce shape (2,) for each cell: [error, prediction]
        result = xr.apply_ufunc(
            self.fit_predict,
            Predictor,                         # shape (T, features)
            Predictant_st.chunk({'Y': chunksize_y, 'X': chunksize_x}),
            Predictor_for_year_,
            y_test.chunk({'Y': chunksize_y, 'X': chunksize_x}),
            input_core_dims=[('T', 'features'), ('T',), ('features',), ()],
            vectorize=True,
            dask='parallelized',
            output_core_dims=[('output',)],
            output_dtypes=['float'],
            dask_gufunc_kwargs={'output_sizes': {'output': 2}}
        )

        # Compute and close the client
        result_ = result.compute()
        result_ = result_.isel(output=1)
        result_ = reverse_standardize(result_, Predictant, clim_year_start, clim_year_end)
        # result_ => dims (output=2, Y, X). 
        # For a real future forecast, "error" is NaN, "prediction" is the forecast.

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
        # year = Predictor_for_year.coords['T'].values.astype('datetime64[Y]').astype(int)[0] + 1970  
        T_value_1 = Predictant.isel(T=0).coords['T'].values  # Get the datetime64 value from da1
        month_1 = T_value_1.astype('datetime64[M]').astype(int) % 12 + 1  # Extract month
        new_T_value = np.datetime64(f"{year}-{month_1:02d}-{1:02d}")
        
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
        return result_da * mask, mask * forecast_prob.transpose('probability', 'T', 'Y', 'X')

        
###########################################

class WAS_PoissonRegression:
    """
    Poisson Regression model for spatiotemporal **count data** prediction (e.g., number of rainy days,
    number of dry spells, extreme event counts, or discretized rainfall amounts) in climate applications.

    This class implements:
    - Standard Poisson regression via scikit-learn's `PoissonRegressor`
    - Spatially parallel fitting and prediction across large grids using dask + xarray
    - Deterministic point predictions (expected counts)
    - Optional probabilistic tercile forecasting (Below/Normal/Above) using either:
      - Parametric distributions fitted per grid cell ('bestfit')
      - Non-parametric historical error sampling ('nonparam')

    Designed for seasonal forecasting tasks where the predictand is non-negative integer counts.

    Parameters
    ----------
    nb_cores : int, default=1
        Number of CPU cores to use for parallel computation (dask workers).

    dist_method : {'bestfit', 'nonparam'}, default='nonparam'
        Method for computing tercile probabilities:
        - 'bestfit'  → uses best-fit distribution per grid cell (requires distribution fit inputs)
        - 'nonparam' → empirical sampling of historical forecast errors

    Attributes
    ----------
    nb_cores, dist_method
        Stored initialization parameters.

    Methods
    -------
    fit_predict(x, y, x_test, y_test)
        Fits Poisson regression on one grid cell and returns [error, prediction].

    compute_model(X_train, y_train, X_test, y_test)
        Parallel Poisson regression across the entire spatial domain.
        Returns xarray.DataArray with dims ('output'=['error','prediction'], Y, X).

    compute_prob(Predictant, clim_year_start, clim_year_end, hindcast_det,
                 best_code_da=None, best_shape_da=None, best_loc_da=None, best_scale_da=None)
        Computes tercile probabilities for deterministic hindcasts.

    forecast(Predictant, clim_year_start, clim_year_end, Predictor, hindcast_det,
             Predictor_for_year, best_code_da=None, best_shape_da=None,
             best_loc_da=None, best_scale_da=None)
        Full end-to-end forecast pipeline for one target year:
        - deterministic prediction (Poisson expected count)
        - tercile probabilities (PB, PN, PA)

    Notes
    -----
    - **Input data requirements**:
      - y (target) must be non-negative integers (counts). Non-integer or negative values may cause
        fitting errors or poor performance.
      - Predictors (X) should be continuous or properly encoded.
    - Predictions are clipped to ≥ 0 (Poisson rates cannot be negative).
    - Large spatial domains benefit significantly from higher `nb_cores`.
    - For `dist_method='bestfit'`, distribution fit results from `WAS_TransformData` must be provided.
    - No target scaling is applied (Poisson regression is scale-invariant in the response).
    - Error is computed as `y_test - prediction` (can be negative).

    Warnings
    --------
    - Ensure y (target) contains only non-negative integers.
    - Very low counts or zero-inflated data may require zero-inflated Poisson (not implemented here).
    - Small clusters or sparse data can lead to unstable fits.
    """

    def __init__(self, nb_cores=1, dist_method="nonparam"):
        """
        Initializes the WAS_PoissonRegression with a specified number of CPU cores and 
        a default distribution method for tercile probability calculations.

        Parameters
        ----------
        nb_cores : int, optional
            Number of CPU cores to use for parallel computation, by default 1.
        dist_method : str, optional
            The distribution method to compute tercile probabilities, by default "gamma".
        """
        self.nb_cores = nb_cores
        self.dist_method = dist_method

    def fit_predict(self, x, y, x_test, y_test):
        """
        Fits a Poisson regression model to the provided training data, makes predictions 
        on the test data, and calculates the prediction error.
        
        Parameters
        ----------
        x : array-like, shape (n_samples, n_features)
            Training data (predictors).
        y : array-like, shape (n_samples,)
            Training targets (non-negative count data).
        x_test : array-like, shape (n_features,) or (1, n_features)
            Test data (predictors).
        y_test : float
            Test target value (actual counts).

        Returns
        -------
        np.ndarray of shape (2,)
            [prediction_error, predicted_value]
        """
        # PoissonRegressor requires non-negative y. We assume the user has handled invalid data.
        model = linear_model.PoissonRegressor()

        # Fit on all provided samples. (If any NaNs exist, user must filter them out externally 
        # or we might add a mask for valid data.)
        model.fit(x, y)

        # Predict on the test data
        if x_test.ndim == 1:
            x_test = x_test.reshape(1, -1)
        preds = model.predict(x_test).squeeze()

        # Poisson rates should not be negative, but numeric or solver issues could occur
        preds[preds < 0] = 0

        # Compute difference from actual
        error_ = y_test - preds
        return np.array([error_, preds]).squeeze()

    def compute_model(self, X_train, y_train, X_test, y_test):
        """
        Computes predictions for spatiotemporal data using Poisson Regression with parallel processing.

        Parameters
        ----------
        X_train : xarray.DataArray
            Predictor data with dimensions (T, features).
        y_train : xarray.DataArray
            Training target values (count data) with dimensions (T, Y, X).
        X_test : xarray.DataArray
            Test data (predictors) with shape (features,) or (T, features), typically squeezed.
        y_test : xarray.DataArray
            Test target values (count data) with dimensions (Y, X) or broadcastable to (T, Y, X).

        Returns
        -------
        xarray.DataArray
            An array with a new dimension ('output', size=2) capturing [error, prediction].
        """
        # Determine chunk sizes so each worker handles a portion of the spatial domain
        chunksize_x = int(np.round(len(y_train.get_index("X")) / self.nb_cores))
        chunksize_y = int(np.round(len(y_train.get_index("Y")) / self.nb_cores))

        # Align the 'T' dimension
        X_train['T'] = y_train['T']
        y_train = y_train.transpose('T', 'Y', 'X')

        # Squeeze test arrays in case of extra dimensions
        X_test = X_test.squeeze()
        # If y_test has a 'T' dimension, remove/ignore it since we only need (Y,X)
        if 'T' in y_test.dims:
            y_test = y_test.drop_vars('T')
        y_test = y_test.squeeze().transpose('Y', 'X')

        # Create a Dask client for parallel computing
        client = Client(n_workers=self.nb_cores, threads_per_worker=1)

        # Apply our fit_predict method across each spatial cell in parallel
        result = xr.apply_ufunc(
            self.fit_predict,
            X_train,                                 # shape (T, features)
            y_train.chunk({'Y': chunksize_y, 'X': chunksize_x}),  # shape (T,)
            X_test,
            y_test.chunk({'Y': chunksize_y, 'X': chunksize_x}),
            input_core_dims=[('T', 'features'), ('T',), ('features',), ()],
            vectorize=True,
            output_core_dims=[('output',)],         # We'll have an 'output' dimension of size 2
            dask='parallelized',
            output_dtypes=['float'],
            dask_gufunc_kwargs={'output_sizes': {'output': 2}}
        )

        result_ = result.compute()
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

        
    def forecast(self, Predictant, clim_year_start, clim_year_end, Predictor, hindcast_det, Predictor_for_year, best_code_da=None, best_shape_da=None, best_loc_da=None, best_scale_da=None):
        """
        Generate forecasts for a single time (e.g., future year) and compute 
        tercile probabilities based on the chosen distribution method.

        Parameters
        ----------
        Predictant : xarray.DataArray
            Target variable with dimensions (T, Y, X).
        clim_year_start : int
            Start year of climatology period.
        clim_year_end : int
            End year of climatology period.
        Predictor : xarray.DataArray
            Historical predictor data with dimensions (T, features).
        hindcast_det : xarray.DataArray
            Deterministic hindcast array that includes 'error' and 'prediction' over the historical period.
        Predictor_for_year : xarray.DataArray
            Predictor data for the forecast year, shape (features,) or (1, features).

        Returns
        -------
        tuple (result_, hindcast_prob)
            result_  : xarray.DataArray or numpy array with the forecast's [error, prediction].
            hindcast_prob : xarray.DataArray of shape (probability=3, Y, X) with PB, PN, and PA.
        """
        # Chunk sizes for parallel processing
        chunksize_x = int(np.round(len(Predictant.get_index("X")) / self.nb_cores))
        chunksize_y = int(np.round(len(Predictant.get_index("Y")) / self.nb_cores))

        # Align the time dimension
        Predictor['T'] = Predictant['T']
        Predictant = Predictant.transpose('T', 'Y', 'X')

        # Squeeze the forecast predictor data if needed
        Predictor_for_year_ = Predictor_for_year.squeeze()

        # We'll apply our polynomial regression in parallel across Y,X. 
        # Because we are forecasting a single point in time, y_test is unknown, so we omit it or set it to NaN.
        y_test = xr.full_like(Predictant.isel(T=0), np.nan)  # shape (Y,X)

        # Create a Dask client
        client = Client(n_workers=self.nb_cores, threads_per_worker=1)

        # Apply fit_predict to get the forecast for each grid cell 
        # We'll produce shape (2,) for each cell: [error, prediction]
        result = xr.apply_ufunc(
            self.fit_predict,
            Predictor,                         # shape (T, features)
            Predictant.chunk({'Y': chunksize_y, 'X': chunksize_x}),
            Predictor_for_year_,
            y_test.chunk({'Y': chunksize_y, 'X': chunksize_x}),
            input_core_dims=[('T', 'features'), ('T',), ('features',), ()],
            vectorize=True,
            dask='parallelized',
            output_core_dims=[('output',)],
            output_dtypes=['float'],
            dask_gufunc_kwargs={'output_sizes': {'output': 2}}
        )

        # Compute and close the client
        result_ = result.compute()
        result_ = result_.isel(output=1)

        # result_ => dims (output=2, Y, X). 
        # For a real future forecast, "error" is NaN, "prediction" is the forecast.

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
        # year = Predictor_for_year.coords['T'].values.astype('datetime64[Y]').astype(int)[0] + 1970  
        T_value_1 = Predictant.isel(T=0).coords['T'].values  # Get the datetime64 value from da1
        month_1 = T_value_1.astype('datetime64[M]').astype(int) % 12 + 1  # Extract month
        new_T_value = np.datetime64(f"{year}-{month_1:02d}-{1:02d}")
        
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
        return forecast_expanded,forecast_prob.transpose('probability', 'T', 'Y', 'X')


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
    Negative Binomial (NB2) regression for spatiotemporal count prediction (Dask/xarray safe).

    Deterministic prediction: mu_hat = E[Y|X] from NB2 GLM with log link:
        Var(Y|X) = mu + alpha * mu^2
    """

    def __init__(self, nb_cores=1, dist_method="nonparam", alpha_init=0.2, add_intercept=True):
        self.nb_cores = nb_cores
        self.dist_method = dist_method
        self.alpha_init = alpha_init
        self.add_intercept = add_intercept

    def fit_predict(self, x, y, x_test, y_test):
        x = np.asarray(x, float)
        y = np.asarray(y, float)
        x_test = np.asarray(x_test, float)

        if x.ndim == 1:
            x = x[:, None]
        x, y = _safe_mask_xy(x, y)

        if y.size < 5 or np.any(y < 0):
            return np.array([np.nan, np.nan], dtype=float)

        if self.add_intercept:
            X = _add_intercept(x)
        else:
            X = x

        beta, alpha = _nb2_irls_beta_alpha(y, X, alpha_init=self.alpha_init)

        if x_test.ndim == 1:
            x_test = x_test.reshape(1, -1)

        if self.add_intercept:
            Xtest = _add_intercept(x_test)
        else:
            Xtest = x_test

        eta_test = np.clip(Xtest @ beta, -700.0, 700.0)
        mu_test = np.exp(eta_test).squeeze()
        mu_test = np.maximum(mu_test, 0.0)

        # error (may be NaN for future forecasts)
        err = (np.asarray(y_test, float) - mu_test).squeeze()
        return np.array([err, mu_test], dtype=float).squeeze()

    def compute_model(self, X_train, y_train, X_test, y_test):
        chunksize_x = int(np.round(len(y_train.get_index("X")) / self.nb_cores))
        chunksize_y = int(np.round(len(y_train.get_index("Y")) / self.nb_cores))

        X_train["T"] = y_train["T"]
        y_train = y_train.transpose("T", "Y", "X")

        X_test = X_test.squeeze()
        if "T" in y_test.dims:
            y_test = y_test.drop_vars("T")
        y_test = y_test.squeeze().transpose("Y", "X")

        client = None
        if Client is not None and self.nb_cores and self.nb_cores > 1:
            client = Client(n_workers=self.nb_cores, threads_per_worker=1)

        result = xr.apply_ufunc(
            self.fit_predict,
            X_train,
            y_train.chunk({"Y": chunksize_y, "X": chunksize_x}),
            X_test,
            y_test.chunk({"Y": chunksize_y, "X": chunksize_x}),
            input_core_dims=[("T", "features"), ("T",), ("features",), ()],
            vectorize=True,
            output_core_dims=[("output",)],
            dask="parallelized",
            output_dtypes=["float"],
            dask_gufunc_kwargs={"output_sizes": {"output": 2}},
        )

        result_ = result.compute()
        if client is not None:
            client.close()

        return result_.isel(output=1)

    # ---- probability and forecast: same pattern as your Poisson class ----

    def compute_prob(
        self,
        Predictant: xr.DataArray,
        clim_year_start,
        clim_year_end,
        hindcast_det: xr.DataArray,
        best_code_da: xr.DataArray = None,
        best_shape_da: xr.DataArray = None,
        best_loc_da: xr.DataArray = None,
        best_scale_da: xr.DataArray = None,
    ) -> xr.DataArray:

        if "M" in Predictant.dims:
            Predictant = Predictant.isel(M=0).drop_vars("M").squeeze()

        Predictant = Predictant.transpose("T", "Y", "X")
        mask = xr.where(~np.isnan(Predictant.isel(T=0)), 1.0, np.nan)

        clim = Predictant.sel(T=slice(str(clim_year_start), str(clim_year_end)))
        if clim.sizes.get("T", 0) < 3:
            raise ValueError("Not enough years in climatology period for terciles.")

        error_variance = (Predictant - hindcast_det).var(dim="T")
        dof = max(int(clim.sizes["T"]) - 1, 2)

        terciles_emp = clim.quantile([0.32, 0.67], dim="T")
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
                vectorize=True,
                dask="parallelized",
                output_dtypes=[float, float],
            )

            hindcast_prob = xr.apply_ufunc(
                self.calculate_tercile_probabilities_bestfit,
                hindcast_det,
                error_variance,
                T1, T2,
                best_code_da,
                input_core_dims=[("T",), (), (), (), ()],
                output_core_dims=[("probability", "T")],
                vectorize=True,
                kwargs={"dof": dof},
                dask="parallelized",
                output_dtypes=[float],
                dask_gufunc_kwargs={"output_sizes": {"probability": 3}, "allow_rechunk": True},
            )

        elif dm == "nonparam":
            error_samples = Predictant - hindcast_det
            hindcast_prob = xr.apply_ufunc(
                self.calculate_tercile_probabilities_nonparametric,
                hindcast_det,
                error_samples,
                T1_emp, T2_emp,
                input_core_dims=[("T",), ("T",), (), ()],
                output_core_dims=[("probability", "T")],
                vectorize=True,
                dask="parallelized",
                output_dtypes=[float],
                dask_gufunc_kwargs={"output_sizes": {"probability": 3}, "allow_rechunk": True},
            )
        else:
            raise ValueError(f"Invalid dist_method: {self.dist_method}")

        hindcast_prob = hindcast_prob.assign_coords(probability=("probability", ["PB", "PN", "PA"]))
        return (hindcast_prob * mask).transpose("probability", "T", "Y", "X")

    def forecast(
        self,
        Predictant,
        clim_year_start,
        clim_year_end,
        Predictor,
        hindcast_det,
        Predictor_for_year,
        best_code_da=None,
        best_shape_da=None,
        best_loc_da=None,
        best_scale_da=None,
    ):
        chunksize_x = int(np.round(len(Predictant.get_index("X")) / self.nb_cores))
        chunksize_y = int(np.round(len(Predictant.get_index("Y")) / self.nb_cores))

        Predictor["T"] = Predictant["T"]
        Predictant = Predictant.transpose("T", "Y", "X")

        Predictor_for_year_ = Predictor_for_year.squeeze()
        y_test = xr.full_like(Predictant.isel(T=0), np.nan)

        client = None
        if Client is not None and self.nb_cores and self.nb_cores > 1:
            client = Client(n_workers=self.nb_cores, threads_per_worker=1)

        result = xr.apply_ufunc(
            self.fit_predict,
            Predictor,
            Predictant.chunk({"Y": chunksize_y, "X": chunksize_x}),
            Predictor_for_year_,
            y_test.chunk({"Y": chunksize_y, "X": chunksize_x}),
            input_core_dims=[("T", "features"), ("T",), ("features",), ()],
            vectorize=True,
            dask="parallelized",
            output_core_dims=[("output",)],
            output_dtypes=["float"],
            dask_gufunc_kwargs={"output_sizes": {"output": 2}},
        )

        result_ = result.compute()
        if client is not None:
            client.close()

        forecast_det = result_.isel(output=1)

        # Build a 1-step time coordinate for probability routines
        year = Predictor_for_year.coords["T"].values[0].astype("datetime64[Y]").astype(int) + 1970
        T_value_1 = Predictant.isel(T=0).coords["T"].values
        month_1 = T_value_1.astype("datetime64[M]").astype(int) % 12 + 1
        new_T_value = np.datetime64(f"{year}-{month_1:02d}-{1:02d}")

        forecast_expanded = forecast_det.expand_dims(T=[new_T_value]).astype("float")
        forecast_expanded["T"] = forecast_expanded["T"].astype("datetime64[ns]")

        # climatological terciles (empirical)
        rainfall_for_tercile = Predictant.sel(T=slice(str(clim_year_start), str(clim_year_end)))
        terciles = rainfall_for_tercile.quantile([0.32, 0.67], dim="T")
        T1_emp = terciles.isel(quantile=0).drop_vars("quantile")
        T2_emp = terciles.isel(quantile=1).drop_vars("quantile")

        error_variance = (Predictant - hindcast_det).var(dim="T")
        dof = max(int(rainfall_for_tercile.sizes["T"]) - 1, 2)

        dm = self.dist_method

        if dm == "bestfit":
            if any(v is None for v in (best_code_da, best_shape_da, best_loc_da, best_scale_da)):
                raise ValueError("dist_method='bestfit' requires best_code_da, best_shape_da, best_loc_da, best_scale_da.")

            T1, T2 = xr.apply_ufunc(
                self._ppf_terciles_from_code,
                best_code_da, best_shape_da, best_loc_da, best_scale_da,
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
                T1, T2,
                best_code_da,
                input_core_dims=[("T",), (), (), (), ()],
                output_core_dims=[("probability", "T")],
                vectorize=True,
                dask="parallelized",
                kwargs={"dof": dof},
                output_dtypes=[float],
                dask_gufunc_kwargs={"output_sizes": {"probability": 3}, "allow_rechunk": True},
            )

        elif dm == "nonparam":
            error_samples = Predictant - hindcast_det
            forecast_prob = xr.apply_ufunc(
                self.calculate_tercile_probabilities_nonparametric,
                forecast_expanded,
                error_samples,
                T1_emp, T2_emp,
                input_core_dims=[("T",), ("T",), (), ()],
                output_core_dims=[("probability", "T")],
                vectorize=True,
                dask="parallelized",
                output_dtypes=[float],
                dask_gufunc_kwargs={"output_sizes": {"probability": 3}, "allow_rechunk": True},
            )
        else:
            raise ValueError(f"Invalid dist_method: {self.dist_method}")

        forecast_prob = forecast_prob.assign_coords(probability=("probability", ["PB", "PN", "PA"]))
        return forecast_expanded, forecast_prob.transpose("probability", "T", "Y", "X")


class WAS_ZINB_Model(WAS_NegativeBinomial_Model):
    """
    Zero-Inflated Negative Binomial (pragmatic two-part fit, Dask-safe).

    Model:
      P(Y=0|X) = pi(X) + (1-pi(X)) * NB(y=0 | mu(X), alpha)
      E[Y|X]   = (1 - pi(X)) * mu(X)

    Fitting strategy (per grid cell):
      1) logistic regression on I(y==0) to estimate pi(X)
      2) NB2 regression on y to estimate mu(X), alpha

    """

    def fit_predict(self, x, y, x_test, y_test):
        x = np.asarray(x, float)
        y = np.asarray(y, float)
        x_test = np.asarray(x_test, float)

        if x.ndim == 1:
            x = x[:, None]
        x, y = _safe_mask_xy(x, y)

        if y.size < 8 or np.any(y < 0):
            return np.array([np.nan, np.nan], dtype=float)

        if self.add_intercept:
            X = _add_intercept(x)
        else:
            X = x

        # 1) zero-inflation part
        y0 = (y == 0.0).astype(float)
        gamma_ = _logit_irls_coef(y0, X)

        # 2) NB count part (includes zeros)
        beta, alpha = _nb2_irls_beta_alpha(y, X, alpha_init=self.alpha_init)

        if x_test.ndim == 1:
            x_test = x_test.reshape(1, -1)
        Xtest = _add_intercept(x_test) if self.add_intercept else x_test

        eta_nb = np.clip(Xtest @ beta, -700.0, 700.0)
        mu = np.exp(eta_nb).squeeze()
        mu = np.clip(mu, 0.0, None)

        eta_zi = np.clip(Xtest @ gamma_, -35.0, 35.0)
        pi = (1.0 / (1.0 + np.exp(-eta_zi))).squeeze()
        pi = np.clip(pi, 0.0, 1.0)

        yhat = (1.0 - pi) * mu

        err = (np.asarray(y_test, float) - yhat).squeeze()
        return np.array([err, yhat], dtype=float).squeeze()


class WAS_HurdleNB_Model(WAS_NegativeBinomial_Model):
    """
    Hurdle Negative Binomial (two-part, Dask-safe).

    Model:
      P(Y>0|X) = p+(X)  via logistic regression
      Y|Y>0,X  ~ zero-truncated NB(mu(X), alpha) fit using NB on positives as approximation

    Deterministic mean:
      E[Y|X] = p+(X) * E[Y|Y>0,X]
    where for NB2 (with r=1/alpha, p=r/(r+mu)):
      P0_NB = (r/(r+mu))^r
      E[Y|Y>0] = mu / (1 - P0_NB)

    Notes:
      - We fit NB on positive counts (not fully truncated likelihood) for operational stability.
    """

    def fit_predict(self, x, y, x_test, y_test):
        x = np.asarray(x, float)
        y = np.asarray(y, float)
        x_test = np.asarray(x_test, float)

        if x.ndim == 1:
            x = x[:, None]
        x, y = _safe_mask_xy(x, y)

        if y.size < 8 or np.any(y < 0):
            return np.array([np.nan, np.nan], dtype=float)

        if self.add_intercept:
            X = _add_intercept(x)
        else:
            X = x

        # 1) hurdle part: probability of positive
        y_pos = (y > 0.0).astype(float)
        gamma_ = _logit_irls_coef(y_pos, X)

        # 2) count part: fit NB on positive counts (approximation to truncated NB)
        mpos = y > 0.0
        if np.sum(mpos) < 5:
            return np.array([np.nan, np.nan], dtype=float)

        Xp = X[mpos, :]
        yp = y[mpos]
        beta, alpha = _nb2_irls_beta_alpha(yp, Xp, alpha_init=self.alpha_init)

        if x_test.ndim == 1:
            x_test = x_test.reshape(1, -1)
        Xtest = _add_intercept(x_test) if self.add_intercept else x_test

        # p+(X)
        eta_h = np.clip(Xtest @ gamma_, -35.0, 35.0)
        pplus = (1.0 / (1.0 + np.exp(-eta_h))).squeeze()
        pplus = np.clip(pplus, 0.0, 1.0)

        # NB mean mu(X)
        eta_nb = np.clip(Xtest @ beta, -700.0, 700.0)
        mu = np.exp(eta_nb).squeeze()
        mu = np.clip(mu, 0.0, None)

        # zero-truncation adjustment using NB pmf at 0 from (mu, alpha)
        alpha = max(float(alpha), 1e-10)
        r = 1.0 / alpha
        p = r / (r + mu + 1e-12)
        P0 = np.power(p, r)  # NB P(Y=0)
        trunc_mean = mu / np.clip(1.0 - P0, 1e-12, None)

        yhat = pplus * trunc_mean

        err = (np.asarray(y_test, float) - yhat).squeeze()
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