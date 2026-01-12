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

# import numpy as np
# import optuna
# from sklearn.base import clone
# from sklearn.model_selection import cross_val_score, GridSearchCV, RandomizedSearchCV
# from scipy.stats import randint, uniform, loguniform

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


# class BaseOptimizer__:
#     """
#     Base class for hyperparameter optimization with support for:
#     - GridSearchCV (exhaustive)a
#     - RandomizedSearchCV (random)
#     - Optuna (Bayesian/Bayesian)
#     """
    
#     def __init__(self, optimization_method="grid", n_trials=20, cv=3, random_state=42):
#         self.optimization_method = optimization_method
#         self.n_trials = n_trials
#         self.cv = cv
#         self.random_state = random_state
        
#     def optimize(self, model, param_space, X, y, scoring='neg_mean_squared_error'):
#         if self.optimization_method == "grid":
#             return self._grid_search(model, param_space, X, y, scoring)
#         elif self.optimization_method == "random":
#             return self._random_search(model, param_space, X, y, scoring)
#         elif self.optimization_method == "bayesian":
#             return self._optuna_search(model, param_space, X, y, scoring)
#         else:
#             raise ValueError(f"Unknown optimization method: {self.optimization_method}")

#     def _apply_params(self, model_instance, params):
#         """
#         Helper to apply parameters to a model, handling TransformedTargetRegressor 
#         or Pipelines by adding the appropriate prefix.
#         """
#         if hasattr(model_instance, 'regressor'):
#             # It's a TransformedTargetRegressor
#             keyed_params = {f"regressor__{k}": v for k, v in params.items()}
#             model_instance.set_params(**keyed_params)
#         elif hasattr(model_instance, 'steps'):
#             # It's a Pipeline (assuming 'mlp' is the final step name)
#             # You might need to adjust 'mlp' if your pipeline step name differs
#             keyed_params = {f"mlp__{k}": v for k, v in params.items()}
#             model_instance.set_params(**keyed_params)
#         else:
#             # Standard model
#             model_instance.set_params(**params)
#         return model_instance

#     def _optuna_search(self, model, param_space, X, y, scoring):
#         def objective(trial):
#             params = {}
#             for param_name, param_values in param_space.items():
#                 if isinstance(param_values, list):
#                     params[param_name] = trial.suggest_categorical(param_name, param_values)
#                 elif isinstance(param_values, tuple) and len(param_values) >= 2:
#                     low, high = param_values[0], param_values[1]
#                     is_log = len(param_values) == 3 and param_values[2] == 'log'
#                     params[param_name] = trial.suggest_float(param_name, low, high, log=is_log)
            
#             # FIX: Clone the original model (wrapper and all)
#             model_instance = clone(model)
#             # FIX: Route the suggested params to the internal regressor
#             model_instance = self._apply_params(model_instance, params)
            
#             scores = cross_val_score(model_instance, X, y, cv=self.cv, scoring=scoring, n_jobs=1)
#             return scores.mean()

#         study = optuna.create_study(direction="maximize", 
#                                     sampler=optuna.samplers.TPESampler(seed=self.random_state))
#         study.optimize(objective, n_trials=self.n_trials, show_progress_bar=False)
#         return study.best_params

#     def _grid_search(self, model, param_space, X, y, scoring):
#         from sklearn.model_selection import GridSearchCV
#         # Prefix keys for the grid search if model is wrapped
#         if hasattr(model, 'regressor'):
#             param_space = {f"regressor__{k}": v for k, v in param_space.items()}
        
#         gs = GridSearchCV(model, param_space, cv=self.cv, scoring=scoring, n_jobs=-1)
#         gs.fit(X, y)
        
#         # Clean the double underscores from keys before returning
#         best_params = {k.split('__')[-1]: v for k, v in gs.best_params_.items()}
#         return best_params

#     def _random_search(self, model, param_space, X, y, scoring):
#         from sklearn.model_selection import RandomizedSearchCV
#         param_dist = self._convert_to_distributions(param_space)
        
#         if hasattr(model, 'regressor'):
#             param_dist = {f"regressor__{k}": v for k, v in param_dist.items()}
            
#         rs = RandomizedSearchCV(model, param_dist, n_iter=self.n_trials, 
#                                 cv=self.cv, scoring=scoring, random_state=self.random_state, n_jobs=-1)
#         rs.fit(X, y)
        
#         best_params = {k.split('__')[-1]: v for k, v in rs.best_params_.items()}
#         return best_params

#     def _convert_to_distributions(self, param_space):
#         """Convert parameter space to distributions for random search."""
#         param_distributions = {}
#         for param_name, param_values in param_space.items():
#             if isinstance(param_values, list):
#                 if all(isinstance(v, int) for v in param_values):
#                     param_distributions[param_name] = randint(min(param_values), max(param_values) + 1)
#                 elif all(isinstance(v, float) for v in param_values):
#                     param_distributions[param_name] = uniform(min(param_values), max(param_values) - min(param_values))
#                 else:
#                     param_distributions[param_name] = param_values
#             elif isinstance(param_values, tuple):
#                 if len(param_values) == 2:
#                     if all(isinstance(v, int) for v in param_values):
#                         param_distributions[param_name] = randint(param_values[0], param_values[1] + 1)
#                     else:
#                         param_distributions[param_name] = uniform(param_values[0], param_values[1] - param_values[0])
#                 elif len(param_values) == 3 and param_values[2] == 'log':
#                     param_distributions[param_name] = loguniform(param_values[0], param_values[1])
#         return param_distributions




# class BaseOptimizer_:
#     """
#     Base class for hyperparameter optimization with support for:
#     - GridSearchCV (exhaustive)
#     - RandomizedSearchCV (random)
#     - Optuna (Bayesian)
#     """
    
#     def __init__(self, optimization_method="grid", n_trials=20, cv=3, random_state=42):
#         """
#         Parameters
#         ----------
#         optimization_method : str
#             One of: "grid", "random", "optuna"
#         n_trials : int
#             Number of trials for random/optuna optimization
#         cv : int
#             Cross-validation folds
#         random_state : int
#             Random seed for reproducibility
#         """
#         self.optimization_method = optimization_method
#         self.n_trials = n_trials
#         self.cv = cv
#         self.random_state = random_state
        
#     def optimize(self, model, param_space, X, y, scoring='neg_mean_squared_error'):
#         """
#         Optimize hyperparameters using specified method.
        
#         Parameters
#         ----------
#         model : sklearn estimator
#             Model to optimize
#         param_space : dict
#             Parameter space for optimization
#         X : array-like
#             Features
#         y : array-like
#             Target
#         scoring : str
#             Scoring metric
            
#         Returns
#         -------
#         dict
#             Best parameters
#         """
#         if self.optimization_method == "grid":
#             return self._grid_search(model, param_space, X, y, scoring)
#         elif self.optimization_method == "random":
#             return self._random_search(model, param_space, X, y, scoring)
#         elif self.optimization_method == "bayesian":
#             return self._optuna_search(model, param_space, X, y, scoring)
#         else:
#             raise ValueError(f"Unknown optimization method: {self.optimization_method}")
    
#     def _grid_search(self, model, param_space, X, y, scoring):
#         """Grid search optimization."""
#         from sklearn.model_selection import GridSearchCV
#         grid_search = GridSearchCV(
#             estimator=model,
#             param_grid=param_space,
#             cv=self.cv,
#             scoring=scoring,
#             n_jobs=-1
#         )
#         grid_search.fit(X, y)
#         return grid_search.best_params_
    
#     def _random_search(self, model, param_space, X, y, scoring):
#         """Random search optimization."""
#         from sklearn.model_selection import RandomizedSearchCV
#         # Convert lists to distributions for random search
#         param_distributions = self._convert_to_distributions(param_space)
        
#         random_search = RandomizedSearchCV(
#             estimator=model,
#             param_distributions=param_distributions,
#             n_iter=self.n_trials,
#             cv=self.cv,
#             scoring=scoring,
#             random_state=self.random_state,
#             n_jobs=-1
#         )
#         random_search.fit(X, y)
#         return random_search.best_params_
    
#     def _optuna_search(self, model, param_space, X, y, scoring):
#         """Optuna Bayesian optimization."""
#         # Define objective function for Optuna
#         def objective(trial):
#             # Suggest parameters based on param_space
#             params = {}
#             for param_name, param_values in param_space.items():
#                 if isinstance(param_values, list):
#                     if all(isinstance(v, int) for v in param_values):
#                         params[param_name] = trial.suggest_categorical(param_name, param_values)
#                     elif all(isinstance(v, float) for v in param_values):
#                         params[param_name] = trial.suggest_float(param_name, min(param_values), max(param_values))
#                     else:
#                         params[param_name] = trial.suggest_categorical(param_name, param_values)
#                 elif isinstance(param_values, tuple) and len(param_values) == 2:
#                     # Assume (low, high) for continuous or (low, high, 'log') for log scale
#                     if len(param_values) == 3 and param_values[2] == 'log':
#                         params[param_name] = trial.suggest_float(param_name, param_values[0], param_values[1], log=True)
#                     else:
#                         params[param_name] = trial.suggest_float(param_name, param_values[0], param_values[1])
            
#             # Create and cross-validate model
#             model_instance = model.__class__(**params)
#             from sklearn.model_selection import cross_val_score
#             scores = cross_val_score(model_instance, X, y, cv=self.cv, scoring=scoring)
#             return scores.mean()
        
#         # Create study and optimize
#         study = optuna.create_study(direction="maximize", sampler=optuna.samplers.TPESampler(seed=self.random_state))
#         study.optimize(objective, n_trials=self.n_trials, show_progress_bar=False)
        
#         return study.best_params
    
#     def _convert_to_distributions(self, param_space):
#         """Convert parameter space to distributions for random search."""
#         param_distributions = {}
#         for param_name, param_values in param_space.items():
#             if isinstance(param_values, list):
#                 if all(isinstance(v, int) for v in param_values):
#                     param_distributions[param_name] = randint(min(param_values), max(param_values) + 1)
#                 elif all(isinstance(v, float) for v in param_values):
#                     param_distributions[param_name] = uniform(min(param_values), max(param_values) - min(param_values))
#                 else:
#                     param_distributions[param_name] = param_values
#             elif isinstance(param_values, tuple):
#                 if len(param_values) == 2:
#                     if all(isinstance(v, int) for v in param_values):
#                         param_distributions[param_name] = randint(param_values[0], param_values[1] + 1)
#                     else:
#                         param_distributions[param_name] = uniform(param_values[0], param_values[1] - param_values[0])
#                 elif len(param_values) == 3 and param_values[2] == 'log':
#                     param_distributions[param_name] = loguniform(param_values[0], param_values[1])
#         return param_distributions


class WAS_SVR:
    def __init__(
        self, 
        nb_cores=1, 
        n_clusters=5, 
        kernel='linear',
        gamma=None,
        C_range=[0.1, 1, 10, 100], 
        epsilon_range=[0.01, 0.1, 0.5, 1], 
        degree_range=[2, 3, 4],
        dist_method="nonparam",
        optimization_method="grid",  # optimization method "grid", "random", "bayesian" 
        n_trials=20,   # number of trials for random/optuna
        cv=5,  
        random_state=42  # random seed
    ):
        self.nb_cores = nb_cores
        self.n_clusters = n_clusters
        self.kernel = kernel
        self.gamma = gamma if gamma is not None else ["auto", "scale"]
        self.C_range = C_range
        self.epsilon_range = epsilon_range
        self.degree_range = degree_range
        self.dist_method = dist_method
        self.optimization_method = optimization_method
        self.n_trials = n_trials
        self.cv = cv
        self.random_state = random_state
        
        # Initialize optimizer
        self.optimizer = BaseOptimizer(
            optimization_method=optimization_method,
            n_trials=n_trials,
            cv=cv,
            random_state=random_state
        )

    def compute_hyperparameters(self, predictand, predictor, clim_year_start, clim_year_end):
        predictor['T'] = predictand['T']
        """Optimized version with Bayesian optimization."""
        # Step 1: Perform KMeans clustering
        kmeans = KMeans(n_clusters=self.n_clusters, random_state=42)
        
        predictand_dropna = predictand.to_dataframe().reset_index().dropna().drop(columns=['T'])
        variable_column = predictand_dropna.columns[2]
        predictand_dropna['cluster'] = kmeans.fit_predict(
            predictand_dropna[[variable_column]]
        )
        
        df_unique = predictand_dropna.drop_duplicates(subset=['Y', 'X'])
        dataset = df_unique.set_index(['Y', 'X']).to_xarray()
        mask = xr.where(~np.isnan(predictand.isel(T=0)), 1, np.nan)
        Cluster = (dataset['cluster'] * mask)
        
        xarray1, xarray2 = xr.align(predictand, Cluster, join="outer")
        clusters = np.unique(xarray2)
        clusters = clusters[~np.isnan(clusters)]
        
        cluster_means = {
            int(cluster): xarray1.where(xarray2 == cluster).mean(dim=['Y', 'X'], skipna=True)
            for cluster in clusters
        }

        # Step 2: Prepare parameter space based on kernel
        param_space = []
        
        if self.kernel in ['linear', 'all']:
            param_space.append({
                'kernel': ['linear'], 
                'C': self.C_range, 
                'epsilon': self.epsilon_range
            })
        if self.kernel in ['poly', 'all']:
            param_space.append({
                'kernel': ['poly'], 
                'degree': self.degree_range, 
                'C': self.C_range, 
                'epsilon': self.epsilon_range
            })
        if self.kernel in ['rbf', 'all']:
            param_space.append({
                'kernel': ['rbf'], 
                'C': self.C_range, 
                'epsilon': self.epsilon_range, 
                'gamma': self.gamma
            })

        hyperparams_cluster = {}
        
        # Step 3: Optimize for each cluster
        for cluster_label in clusters:
            cluster_mean = cluster_means[int(cluster_label)].dropna('T')
            predictor['T'] = cluster_mean['T']
            common_times = np.intersect1d(cluster_mean['T'].values, predictor['T'].values)
            
            if len(common_times) == 0:
                continue

            cluster_mean_common = cluster_mean.sel(T=common_times)
            predictor_common = predictor.sel(T=common_times)

            y_cluster = cluster_mean_common.values
            if y_cluster.size > 0:
                # Use optimizer to find best parameters
                svr = SVR()
                model = TransformedTargetRegressor(regressor=svr,
                                                          transformer=StandardScaler()
                                                         )
                best_params = self.optimizer.optimize(
                    model, 
                    param_space, 
                    predictor_common, 
                    y_cluster
                )
                
                hyperparams_cluster[int(cluster_label)] = {
                    'C': best_params['C'],
                    'epsilon': best_params['epsilon'],
                    'kernel': best_params['kernel'],
                    'degree': best_params.get('degree', None),
                    'gamma': best_params.get('gamma', None)
                }
    
        # Step 4: Create DataArrays for best parameters
        C_array = xr.full_like(Cluster, np.nan, dtype=float)
        epsilon_array = xr.full_like(Cluster, np.nan, dtype=float)
        degree_array = xr.full_like(Cluster, np.nan, dtype=int)
        kernel_array = xr.full_like(Cluster, "", dtype=object)
        gamma_array = xr.full_like(Cluster, "", dtype=object)

        for cluster_label, params in hyperparams_cluster.items():
            mask = Cluster == cluster_label
            C_array = C_array.where(~mask, other=params['C'])
            epsilon_array = epsilon_array.where(~mask, other=params['epsilon'])
            degree_array = degree_array.where(~mask, other=params.get('degree', np.nan))
            kernel_array = kernel_array.where(~mask, other=params['kernel'])
            gamma_array = gamma_array.where(~mask, other=params.get('gamma', ""))
    
        C_array, epsilon_array, degree_array, Cluster, _ = xr.align(
            C_array, epsilon_array, degree_array, Cluster, 
            predictand.isel(T=0).drop_vars('T').squeeze(), 
            join="outer"
        )
        
        # Align kernel and gamma arrays
        kernel_array, _ = xr.align(kernel_array, Cluster, join="outer")
        gamma_array, _ = xr.align(gamma_array, Cluster, join="outer")
        
        return C_array, epsilon_array, degree_array, Cluster, kernel_array, gamma_array

    def fit_predict(self, x, y, x_test, y_test, epsilon, C, degree=None):
        """
        Fits an SVR model to the provided training data, makes predictions on the test data, 
        and calculates the prediction error.

        We handle data-type issues (e.g., bytes input), set up the SVR with the requested
        parameters, fit it, and return both the error and the prediction.

        Parameters
        ----------
        x : array-like, shape (n_samples, n_features)
            Training predictors.
        y : array-like, shape (n_samples,)
            Training targets.
        x_test : array-like, shape (n_features,)
            Test predictors.
        y_test : float or None
            Test target value. Used to calculate error if available.
        epsilon : float
            Epsilon parameter for SVR (defines epsilon-tube).
        C : float
            Regularization parameter for SVR.
        degree : int, optional
            Degree for 'poly' kernel. Ignored if kernel != 'poly'.

        Returns
        -------
        np.ndarray
            A 2-element array containing [error, prediction].
        """
        # Convert any byte-string parameters to standard Python strings/integers
        if isinstance(self.kernel, bytes):
            kernel = self.kernel.decode('utf-8')
        if isinstance(degree, bytes) and degree is not None and not np.isnan(degree):
            degree = int(degree)
        if isinstance(self.gamma, bytes) and self.gamma is not None:
            gamma = self.gamma.decode('utf-8')
        
        # Ensure 'degree' has a valid numeric default if not properly set
        if degree is None or degree == 'nan' or (isinstance(degree, float) and np.isnan(degree)):
            degree = 1
        else:
            degree = int(float(degree))

        # Prepare model parameters based on kernel type
        model_params = {'kernel': self.kernel, 'C': C, 'epsilon': epsilon}
        if self.kernel == 'poly' and degree is not None:
            model_params['degree'] = int(degree)
        if self.kernel == 'rbf' and self.gamma[0] is not None:
            model_params['gamma'] = self.gamma[0]

        # Instantiate the SVR model with chosen parameters
        svr = SVR(**model_params)
        
        model = TransformedTargetRegressor(
            regressor=svr,
            transformer=StandardScaler()
)

        # Check for valid (finite) training data
        mask = np.isfinite(y) & np.all(np.isfinite(x), axis=-1)

        # Train only if there's valid data
        if np.any(mask):
            y_clean = y[mask]
            x_clean = x[mask, :]

            model.fit(x_clean, y_clean)

            # If x_test is 1-D, reshape into 2-D for prediction
            if x_test.ndim == 1:
                x_test = x_test.reshape(1, -1)

            # Make predictions
            preds = model.predict(x_test)

            # Ensuring no negative predictions (if that applies to your data domain, e.g., rainfall)
            preds[preds < 0] = 0

            # Calculate error, if y_test is valid
            if y_test is not None and not np.isnan(y_test):
                error_ = y_test - preds
            else:
                error_ = np.nan

            # Return [error, prediction] as a flattened array
            return np.array([error_, preds]).squeeze()
        else:
            # If there's no valid training data, return NaNs
            return np.array([np.nan, np.nan]).squeeze()

    
    def compute_model(self, X_train, y_train, X_test, y_test, epsilon, C, degree_array=None):
        """
        Computes predictions for spatiotemporal data using SVR with parallel processing via Dask.

        We break the data into chunks, apply the `fit_predict` function in parallel,
        and combine the results into an output DataArray.

        Parameters
        ----------
        X_train : xarray.DataArray
            Training predictors with dimensions ('T', 'features').
        y_train : xarray.DataArray
            Training targets with dimensions ('T', 'Y', 'X').
        X_test : xarray.DataArray
            Test predictors with dimensions ('features',).
        y_test : xarray.DataArray
            Test target values with dimensions ('Y', 'X').
        epsilon : xarray.DataArray
            Epsilon hyperparameters per grid point.
        C : xarray.DataArray
            C hyperparameters per grid point.
        degree_array : xarray.DataArray, optional
            Polynomial degrees per grid point (only used if kernel='poly').

        Returns
        -------
        xarray.DataArray
            Predictions & errors, stacked along a new 'output' dimension (size=2).
        """
        # Determine chunk sizes so each worker handles a portion of the spatial domain
        chunksize_x = int(np.round(len(y_train.get_index("X")) / self.nb_cores))
        chunksize_y = int(np.round(len(y_train.get_index("Y")) / self.nb_cores))

        # Align time dimension in X_train with y_train
        X_train['T'] = y_train['T']
        y_train = y_train.transpose('T', 'Y', 'X')
        
        # Squeeze out any singleton dimension in X_test / y_test
        X_test = X_test.squeeze()
        y_test = y_test.squeeze().transpose('Y', 'X')

        # Create a Dask client for parallel processing
        client = Client(n_workers=self.nb_cores, threads_per_worker=1)

        # Apply `fit_predict` across each (Y,X) grid cell in parallel
        result = xr.apply_ufunc(
            self.fit_predict,
            X_train,
            y_train.chunk({'Y': chunksize_y, 'X': chunksize_x}),
            X_test,
            y_test.chunk({'Y': chunksize_y, 'X': chunksize_x}),
            epsilon.chunk({'Y': chunksize_y, 'X': chunksize_x}),
            C.chunk({'Y': chunksize_y, 'X': chunksize_x}),
            degree_array.chunk({'Y': chunksize_y, 'X': chunksize_x}) if degree_array is not None else xr.full_like(epsilon, None),
            input_core_dims=[
                ('T', 'features'),  # x
                ('T',),             # y
                ('features',),      # x_test
                (),                 # y_test
                (),                 # epsilon
                (),                 # C
                ()                  # degree
            ],
            vectorize=True,
            output_core_dims=[('output',)],
            dask='parallelized',
            output_dtypes=['float'],
            dask_gufunc_kwargs={'output_sizes': {'output': 2}},
        )

        # Trigger actual computation
        result_ = result.compute()

        # Close the Dask client
        client.close()

        # Return the results, containing both errors and predictions
        return result_.isel(output=1)

    @staticmethod
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
        - best_guess : 1D array over T (hindcast_det or forecast_det)
        - T1, T2     : scalar terciles from climatological best-fit distribution
        - dist_code  : int, as in _ppf_terciles_from_code
        - shape, loc, scale : scalars from climatology fit

        Strategy:
        - For each time step, build a predictive distribution of the same family:
            * Use best_guess[t] to adjust mean / location;
            * Keep shape parameters from climatology.
        - Then compute probabilities:
            P(B) = F(T1), P(N) = F(T2) - F(T1), P(A) = 1 - F(T2).
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

        
    def forecast(
        self, 
        Predictant, 
        clim_year_start, 
        clim_year_end, 
        Predictor, 
        hindcast_det, 
        Predictor_for_year, 
        epsilon, 
        C, 
        kernel_array, 
        degree_array, 
        gamma_array, best_code_da=None, best_shape_da=None, best_loc_da=None, best_scale_da=None
    ):
        """
        Generates forecasts and computes probabilities for a specific year.

        Parameters
        ----------
        Predictant : xarray.DataArray
            Target variable (T, Y, X).
        clim_year_start : int
            Start year for climatology.
        clim_year_end : int
            End year for climatology.
        Predictor : xarray.DataArray
            Historical predictor data (T, features).
        hindcast_det : xarray.DataArray
            Deterministic hindcasts (includes 'prediction' and 'error' outputs).
        Predictor_for_year : xarray.DataArray
            Predictor data for the target forecast year (features).
        epsilon, C, kernel_array, degree_array, gamma_array : xarray.DataArray
            Hyperparameter grids for the model.

        Returns
        -------
        tuple
            1) The forecast results (error, prediction) for that year.
            2) The corresponding tercile probabilities (PB, PN, PA).
        """
        # Divide the spatial domain into chunks for parallel computation
        chunksize_x = int(np.round(len(Predictant.get_index("X")) / self.nb_cores))
        chunksize_y = int(np.round(len(Predictant.get_index("Y")) / self.nb_cores))

        # Ensure time dimension alignment
        Predictor['T'] = Predictant['T']
        Predictant = Predictant.transpose('T', 'Y', 'X')
        Predictor_for_year_ = Predictor_for_year.squeeze()

        # We don't have an actual observed y_test for the forecast year, so fill with NaNs
        y_test = xr.full_like(epsilon, np.nan)

        # Create a Dask client for parallelization
        client = Client(n_workers=self.nb_cores, threads_per_worker=1)

        # Apply `fit_predict` in parallel across the grid, using the forecast year's predictors
        result = xr.apply_ufunc(
            self.fit_predict,
            Predictor,
            Predictant.chunk({'Y': chunksize_y, 'X': chunksize_x}),
            Predictor_for_year_,
            y_test.chunk({'Y': chunksize_y, 'X': chunksize_x}),
            epsilon.chunk({'Y': chunksize_y, 'X': chunksize_x}),
            C.chunk({'Y': chunksize_y, 'X': chunksize_x}),
            kernel_array.chunk({'Y': chunksize_y, 'X': chunksize_x}),
            degree_array.chunk({'Y': chunksize_y, 'X': chunksize_x}),
            gamma_array.chunk({'Y': chunksize_y, 'X': chunksize_x}),
            input_core_dims=[
                ('T', 'features'),  # x (training)
                ('T',),             # y (training target)
                ('features',),      # x_test (forecast-year predictors)
                (),                 # y_test (unknown, hence NaN)
                (),                 # epsilon
                (),                 # C
                (),                 # kernel
                (),                 # degree
                ()                  # gamma
            ],
            vectorize=True,
            output_core_dims=[('output',)],
            dask='parallelized',
            output_dtypes=['float'],
            dask_gufunc_kwargs={'output_sizes': {'output': 2}},
        )
        result_ = result.compute()
        client.close()
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
        return result_da, forecast_prob.transpose('probability', 'T', 'Y', 'X')






class WAS_MLP:
    """
    A class to perform MLP (Multi-Layer Perceptron) regression on spatiotemporal
    datasets for climate prediction, with hyperparameter tuning via clustering + grid search.

    Parameters
    ----------
    nb_cores : int
        Number of CPU cores to use for parallel computation.
    dist_method : str
        Distribution method for tercile probability calculations. 
        One of {'gamma', 't', 'normal', 'lognormal', 'nonparam'}.
    n_clusters : int
        Number of clusters to use for KMeans.
    param_grid : dict or None
        The hyperparameter search grid for MLPRegressor. 
        If None, a default grid is used.

    Attributes
    ----------
    nb_cores, dist_method, n_clusters, param_grid
    """

    def __init__(
        self,
        nb_cores=1,
        dist_method="nonparam",
        n_clusters=5,
        param_grid=None,
        optimization_method="grid",  # New parameter
        n_trials=20,  # New parameter
        cv=5,  # New parameter
        random_state=42  # New parameter
    ):
        """
        Initializes the WAS_MLP with specified hyperparameter ranges.

        Parameters
        ----------
        nb_cores : int, optional
            Number of CPU cores to use for parallel computation.
        n_clusters : int, optional
            Number of clusters for KMeans.
        kernel : str, optional
            Kernel type to be used in SVR ('linear', 'poly', 'rbf', or 'all').
        gamma : str, optional
            Kernel coefficient for 'rbf' kernel. Ignored otherwise.
        C_range : list, optional
            List of C values for hyperparameter tuning.
        epsilon_range : list, optional
            List of epsilon values for hyperparameter tuning.
        degree_range : list, optional
            List of polynomial degrees for 'poly' kernel.
        dist_method : str, optional
            Distribution method for tercile probability calculations.
        """
        self.nb_cores = nb_cores
        self.dist_method = dist_method
        self.n_clusters = n_clusters
        
        # Define default parameter grid if none provided
        if param_grid is None:
            self.param_grid = {
                'hidden_layer_sizes': [(10,5), (10,), (20,10), (50,)],
                'activation': ['relu', 'tanh', 'logistic'],
                'solver': ['adam', 'lbfgs'],
                'alpha': [0.0001, 0.001, 0.01],
                'learning_rate_init': [0.001, 0.01, 0.1],
                'max_iter': [200, 500, 1000]
            }
        else:
            self.param_grid = param_grid
            
        self.optimization_method = optimization_method
        self.n_trials = n_trials
        self.cv = cv
        self.random_state = random_state
        
        # Initialize optimizer
        self.optimizer = BaseOptimizer(
            optimization_method=optimization_method,
            n_trials=n_trials,
            cv=cv,
            random_state=random_state
        )

    def compute_hyperparameters(self, predictand, predictor, clim_year_start, clim_year_end):
        """Optimized version with Bayesian optimization."""
        predictor['T'] = predictand['T']
        predictand_ = standardize_timeseries(predictand)

        # (a) KMeans clustering
        kmeans = KMeans(n_clusters=self.n_clusters, random_state=42)
        predictand_dropna = (
            predictand.to_dataframe()
                      .reset_index()
                      .dropna()
                      .drop(columns=['T'])
        )
        
        col_name = predictand_dropna.columns[2]
        predictand_dropna['cluster'] = kmeans.fit_predict(
            predictand_dropna[[col_name]]
        )
        
        df_unique = predictand_dropna.drop_duplicates(subset=['Y', 'X'])
        dataset = df_unique.set_index(['Y', 'X']).to_xarray()
        
        cluster_da = (dataset['cluster'] *
                      xr.where(~np.isnan(predictand.isel(T=0)), 1, np.nan)
                     ).drop_vars("T", errors='ignore')
        
        _, cluster_da = xr.align(predictand, cluster_da, join="outer")
        clusters = np.unique(cluster_da)
        clusters = clusters[~np.isnan(clusters)]
    
        hyperparams_cluster = {}
    
        # (b) Optimize for each cluster
        for c in clusters:
            mask_c = (cluster_da == c)
            y_cluster = (
                predictand_.where(mask_c)
                          .mean(dim=["Y", "X"], skipna=True)
                          .dropna(dim="T")
            )
            if len(y_cluster["T"]) == 0:
                continue
    
            predictor_cluster = predictor.sel(T=y_cluster["T"])
            X_mat = predictor_cluster.values
            y_vec = y_cluster.values
            
            # # Wrap MLP in a pipeline with scaling
            # mlp_pipeline = Pipeline([
            #     ('scaler', StandardScaler()),  # Scales features to mean=0, std=1
            #     ('mlp', MLPRegressor(random_state=42))
            # ])

            # MLP with NO X scaling
            mlp = MLPRegressor(random_state=42, max_iter=1000)
            
            # Scale ONLY y for the MLP
            mlp_y_scaled = TransformedTargetRegressor(
                regressor=mlp,
                transformer=StandardScaler()
            )
            
            # # Use optimizer to find best parameters
            # model = MLPRegressor(random_state=42)
            best_params = self.optimizer.optimize(
                mlp_y_scaled,
                self.param_grid,
                X_mat,
                y_vec,
                scoring='neg_mean_squared_error'
            )
            
            hyperparams_cluster[int(c)] = best_params
    
        # (c) Broadcast best hyperparameters to each grid cell
        hl_array = xr.full_like(cluster_da, np.nan, dtype=object)
        act_array = xr.full_like(cluster_da, np.nan, dtype=object)
        solver_array = xr.full_like(cluster_da, np.nan, dtype=object)
        alpha_array = xr.full_like(cluster_da, np.nan, dtype=float)
        lr_array = xr.full_like(cluster_da, np.nan, dtype=float)
        maxiter_array = xr.full_like(cluster_da, np.nan, dtype=float)
        
        for c, bp in hyperparams_cluster.items():
            c_mask = (cluster_da == c)
            hl_str = str(bp.get('hidden_layer_sizes', (10,5)))
            act_str = bp.get('activation', 'relu')
            solver_str = bp.get('solver', 'adam')
            alpha_val = bp.get('alpha', 0.0001)
            lr_val = bp.get('learning_rate_init', 0.001)
            maxiter_val = bp.get('max_iter', 200)
            
            hl_array = hl_array.where(~c_mask, other=hl_str)
            act_array = act_array.where(~c_mask, other=act_str)
            solver_array = solver_array.where(~c_mask, other=solver_str)
            alpha_array = alpha_array.where(~c_mask, other=alpha_val)
            lr_array = lr_array.where(~c_mask, other=lr_val)
            maxiter_array = maxiter_array.where(~c_mask, other=maxiter_val)

        return hl_array, act_array, solver_array, alpha_array, lr_array, maxiter_array, cluster_da

    # ------------------------------------------------------------------
    # 2) FIT + PREDICT ON A SINGLE GRID CELL
    # ------------------------------------------------------------------
    def fit_predict(self, X_train, y_train, X_test, y_test,
                    hl_sizes, activation, lr_init, maxiter):
        """
        Trains an MLP (with local hyperparams) on the provided training data, then predicts on X_test.
        Returns [error, prediction].

        Parameters
        ----------
        X_train : np.ndarray, shape (n_samples, n_features)
        y_train : np.ndarray, shape (n_samples,)
        X_test  : np.ndarray, shape (n_features,) or (1, n_features)
        y_test  : float or np.nan
        hl_sizes : str (stored as string in xarray) or None
        activation : str
        lr_init : float

        Returns
        -------
        np.ndarray of shape (2,)
            [error, predicted_value]
        """
        # Convert hidden_layer_sizes from string if needed
        if hl_sizes is not None and isinstance(hl_sizes, str):
            hl_sizes = eval(hl_sizes)  # parse string into tuple

        mask = np.isfinite(y_train) & np.all(np.isfinite(X_train), axis=-1)

        # mlp_model = Pipeline([
        #     ('scaler', StandardScaler()),
        #     ('mlp', MLPRegressor(
        #     hidden_layer_sizes=hl_sizes if hl_sizes else (10,5),
        #     activation=activation if activation else 'relu',
        #     solver='adam',
        #     max_iter=int(maxiter) if not np.isnan(maxiter) else 1000,
        #     learning_rate_init=lr_init if not np.isnan(lr_init) else 0.001
        #     # learning_rate_init=lr_init if lr_init else 0.001
        # ))
        # ])
        
        # MLP with NO X scaling
        mlp = MLPRegressor(
        hidden_layer_sizes=hl_sizes if hl_sizes else (10,5),
        activation=activation if activation else 'relu',
        solver='adam',
        max_iter=int(maxiter) if not np.isnan(maxiter) else 1000,
        learning_rate_init=lr_init if not np.isnan(lr_init) else 0.001
        # learning_rate_init=lr_init if lr_init else 0.001
        )
            
        # Scale ONLY y for the MLP
        mlp_model = TransformedTargetRegressor(
            regressor=mlp,
            transformer=StandardScaler()
        )
        
        if np.any(mask):
            X_c = X_train[mask, :]
            y_c = y_train[mask]
            mlp_model.fit(X_c, y_c)

            if X_test.ndim == 1:
                X_test = X_test.reshape(1, -1)
            mlp_preds = mlp_model.predict(X_test)
            mlp_preds[mlp_preds < 0] = 0  # clip negative if it's precipitation

            err = np.nan if (y_test is None or np.isnan(y_test)) else (y_test - mlp_preds)
            return np.array([err, mlp_preds]).squeeze()
        else:
            return np.array([np.nan, np.nan]).squeeze()

    # ------------------------------------------------------------------
    # 3) PARALLELIZED MODEL PREDICTION OVER SPACE
    # ------------------------------------------------------------------
    def compute_model(
        self, 
        X_train, y_train, 
        X_test, y_test,
        hl_array, act_array, lr_array, maxiter_array
    ):
        """
        Runs MLP fit/predict for each (Y,X) cell in parallel, using cluster-based hyperparams.
        
        Parameters
        ----------
        X_train : xarray.DataArray
            Training predictors with dims ('T','features').
        y_train : xarray.DataArray
            Training target with dims ('T','Y','X').
        X_test : xarray.DataArray
            Test predictors, shape ('features',) or broadcastable.
        y_test : xarray.DataArray
            Test target with dims ('Y','X').
        hl_array, act_array, lr_array : xarray.DataArray
            Local best hyperparameters from compute_hyperparameters.

        Returns
        -------
        xarray.DataArray
            dims ('output', 'Y', 'X'), where 'output' = [error, prediction].
        """
        chunksize_x = int(np.round(len(y_train.get_index("X")) / self.nb_cores))
        chunksize_y = int(np.round(len(y_train.get_index("Y")) / self.nb_cores))

        # Align time
        X_train['T'] = y_train['T']
        y_train = y_train.transpose('T', 'Y', 'X')

        X_test = X_test.squeeze()
        y_test = y_test.squeeze().transpose('Y', 'X')

        client = Client(n_workers=self.nb_cores, threads_per_worker=1)
        result_da = xr.apply_ufunc(
            self.fit_predict,
            X_train,                           
            y_train.chunk({'Y': chunksize_y, 'X': chunksize_x}),
            X_test,
            y_test.chunk({'Y': chunksize_y, 'X': chunksize_x}),
            hl_array.chunk({'Y': chunksize_y, 'X': chunksize_x}),
            act_array.chunk({'Y': chunksize_y, 'X': chunksize_x}),
            lr_array.chunk({'Y': chunksize_y,  'X': chunksize_x}),
            maxiter_array.chunk({'Y': chunksize_y,  'X': chunksize_x}),
            
            input_core_dims=[
                ('T','features'),  # X_train
                ('T',),           # y_train
                ('features',),    # X_test
                (),               # y_test
                (),               # hidden_layer_sizes
                (),               # activation
                (),                # learning_rate_init
                ()                # max_iter                
            ],
            vectorize=True,
            dask='parallelized',
            output_core_dims=[('output',)],
            output_dtypes=[float],
            dask_gufunc_kwargs={'output_sizes': {'output': 2}},
        )
        result_ = result_da.compute()
        client.close()

        # Return DataArray with dims ('output','Y','X') => [error, prediction]
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
        - best_guess : 1D array over T (hindcast_det or forecast_det)
        - T1, T2     : scalar terciles from climatological best-fit distribution
        - dist_code  : int, as in _ppf_terciles_from_code
        - shape, loc, scale : scalars from climatology fit

        Strategy:
        - For each time step, build a predictive distribution of the same family:
            * Use best_guess[t] to adjust mean / location;
            * Keep shape parameters from climatology.
        - Then compute probabilities:
            P(B) = F(T1), P(N) = F(T2) - F(T1), P(A) = 1 - F(T2).
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

        
    # ------------------------------------------------------------------
    # 6) FORECAST METHOD
    # ------------------------------------------------------------------
    def forecast(
        self, 
        Predictant, 
        clim_year_start, 
        clim_year_end, 
        Predictor, 
        hindcast_det, 
        Predictor_for_year, 
        hl_array, act_array, lr_array, best_code_da=None, best_shape_da=None, best_loc_da=None, best_scale_da=None
    ):
        """
        Generate a forecast for a single future time (e.g., future year), 
        then compute tercile probabilities using the chosen distribution method.

        Parameters
        ----------
        Predictant : xarray.DataArray
            Observed data with dims (T, Y, X), used for computing climatological terciles.
        clim_year_start : int
            Start year of the climatology period.
        clim_year_end : int
            End year of the climatology period.
        Predictor : xarray.DataArray
            Historical predictor data with dims (T, features).
        hindcast_det : xarray.DataArray
            Historical deterministic forecast with dims (output=[error,prediction], T, Y, X).
            Used to compute error variance or error samples.
        Predictor_for_year : xarray.DataArray
            Predictor data for the forecast year, shape (features,) or (1, features).
        hl_array, act_array, lr_array : xarray.DataArray
            Hyperparameters from `compute_hyperparameters`, 
            each with dims (Y, X) specifying local MLP settings.

        Returns
        -------
        result_ : xarray.DataArray
            dims ('output','Y','X'), containing [error, prediction]. 
            For a forecast, the "error" is generally NaN.
        hindcast_prob : xarray.DataArray
            dims (probability=3, Y, X) => PB, PN, PA tercile probabilities.
        """
        # Provide a dummy y_test of NaNs (since we don't have future obs)
        y_test_dummy = xr.full_like(Predictant.isel(T=0), np.nan)  # shape (Y, X)

        # Prepare chunk sizes
        chunksize_x = int(np.round(len(Predictant.get_index("X")) / self.nb_cores))
        chunksize_y = int(np.round(len(Predictant.get_index("Y")) / self.nb_cores))

        # Align times
        Predictor['T'] = Predictant['T']
        Predictant = Predictant.transpose('T', 'Y', 'X')
        Predictor_for_year_ = Predictor_for_year.squeeze()
        Predictant_st = standardize_timeseries(Predictant, clim_year_start, clim_year_end)
        
        # 1) Fit+predict in parallel for each grid cell
        client = Client(n_workers=self.nb_cores, threads_per_worker=1)
        result_da = xr.apply_ufunc(
            self.fit_predict,
            Predictor,                              # X_train
            Predictant.chunk({'Y': chunksize_y, 'X': chunksize_x}),  # y_train
            Predictor_for_year_,                   # X_test
            y_test_dummy.chunk({'Y': chunksize_y, 'X': chunksize_x}),
            hl_array.chunk({'Y': chunksize_y, 'X': chunksize_x}),
            act_array.chunk({'Y': chunksize_y, 'X': chunksize_x}),
            lr_array.chunk({'Y': chunksize_y,  'X': chunksize_x}),

            input_core_dims=[
                ('T','features'),  # X_train
                ('T',),           # y_train
                ('features',),    # X_test
                (),               # y_test
                (),               # hidden_layer_sizes
                (),               # activation
                ()                # learning_rate_init
            ],
            vectorize=True,
            dask='parallelized',
            output_core_dims=[('output',)],
            output_dtypes=[float],
            dask_gufunc_kwargs={'output_sizes': {'output': 2}},
        )
        result_ = result_da.compute()
        client.close()
        result_ = result_.isel(output=1)
        
        result_ = reverse_standardize(result_, Predictant,
                                        clim_year_start, clim_year_end)
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
        return forecast_expanded, forecast_prob.transpose('probability', 'T', 'Y', 'X')

class WAS_RandomForest_XGBoost_ML_Stacking:
    def __init__(
        self,
        nb_cores=1,
        dist_method="nonparam",
        n_clusters=5,
        param_grid=None,
        optimization_method="grid",  # New parameter
        n_trials=20,  # New parameter
        cv=5,  # New parameter
        random_state=42  # New parameter
    ):
        self.nb_cores = nb_cores
        self.dist_method = dist_method
        self.n_clusters = n_clusters
        self.optimization_method = optimization_method
        self.n_trials = n_trials
        self.cv = cv
        self.random_state = random_state

        # Define minimal default param_grid if none is provided
        if param_grid is None:
            self.param_grid = {
                "rf__n_estimators": [50, 100, 200],
                "rf__max_depth": [None, 10, 20],
                "xgb__n_estimators": [50, 100],
                "xgb__max_depth": [3, 6, 9],
                "xgb__learning_rate": [0.01, 0.1, 0.3],
                "xgb__subsample": [0.8, 1.0],
                "final_estimator__fit_intercept": [True, False]
            }
        else:
            self.param_grid = param_grid
            
        # Initialize optimizer
        self.optimizer = BaseOptimizer(
            optimization_method=optimization_method,
            n_trials=n_trials,
            cv=cv,
            random_state=random_state
        )

    def compute_hyperparameters(self, predictand, predictor, clim_year_start, clim_year_end):
        """Optimized version with Bayesian optimization."""
        predictor['T'] = predictand['T']
        df = (
            predictand.to_dataframe()
                      .reset_index()
                      .dropna()
                      .drop(columns=['T'])
        )
        col_name = df.columns[2]
        kmeans = KMeans(n_clusters=self.n_clusters, random_state=42)
        df["cluster"] = kmeans.fit_predict(df[[col_name]])
    
        df_unique = df.drop_duplicates(subset=["Y", "X"])
        dataset = df_unique.set_index(["Y", "X"]).to_xarray()
    
        cluster_da = (dataset["cluster"] *
                      xr.where(~np.isnan(predictand.isel(T=0)), 1, np.nan)
                     ).drop_vars("T", errors="ignore")
    
        _, cluster_da = xr.align(predictand, cluster_da, join="outer")
    
        # Build the stacking model
        base_rf = RandomForestRegressor(n_jobs=-1, random_state=42)
        base_xgb = xgb.XGBRegressor(n_jobs=-1, random_state=42)
        
        meta_pipeline = Pipeline([
            ('scaler', StandardScaler()),
            ('meta_lin', LinearRegression())
        ])
        
        
        stacking_core = StackingRegressor(
            estimators=[
                ("rf", base_rf), 
                ("xgb", base_xgb)
            ],
            final_estimator=meta_pipeline,
            cv=3,
            n_jobs=-1
        )
        
        stacking_model = TransformedTargetRegressor(
            regressor=stacking_core,
            transformer=StandardScaler()
        )
    
        unique_clusters = np.unique(cluster_da)
        unique_clusters = unique_clusters[np.isfinite(unique_clusters)]
        best_params_for_cluster = {}
    
        for c in unique_clusters:
            mask_c = (cluster_da == c)
            y_cluster = (
                predictand.where(mask_c)
                          .mean(dim=["Y", "X"], skipna=True)
                          .dropna(dim="T")
            )
            if len(y_cluster["T"]) == 0:
                continue
    
            predictor_cluster = predictor.sel(T=y_cluster["T"])
            X_mat = predictor_cluster.values
            y_vec = y_cluster.values
    
            # Use optimizer to find best parameters
            best_params = self.optimizer.optimize(
                stacking_model,
                self.param_grid,
                X_mat,
                y_vec,
                scoring='neg_mean_squared_error'
            )
            best_params_for_cluster[int(c)] = best_params
    
        # Broadcast best hyperparameter sets back to each grid cell
        best_param_da = xr.full_like(cluster_da, np.nan, dtype=object)
        for c, bp in best_params_for_cluster.items():
            c_mask = (cluster_da == c)
            best_param_da = best_param_da.where(~c_mask, other=str(bp))
    
        best_param_da, _ = xr.align(best_param_da, predictand, join="outer")

        return best_param_da, cluster_da

    # ----------------------------------------------------------------------
    # 2) FIT + PREDICT FOR A SINGLE GRID CELL
    # ----------------------------------------------------------------------
    def fit_predict(self, X_train, y_train, X_test, y_test, best_params_str):
        """
        Fit a local StackingRegressor with the best hyperparams (parsed from best_params_str),
        then predict on X_test, returning [error, prediction].

        Parameters
        ----------
        X_train : np.ndarray, shape (n_samples, n_features)
        y_train : np.ndarray, shape (n_samples,)
        X_test :  np.ndarray, shape (n_features,) or (1, n_features)
        y_test :  float or np.nan
        best_params_str : str
            String of best_params (e.g. "{'estimators__rf__n_estimators':100, ...}")

        Returns
        -------
        np.ndarray of shape (2,)
            [error, predicted_value]
        """
        mask = np.isfinite(y_train) & np.all(np.isfinite(X_train), axis=-1)
        if not isinstance(best_params_str, str) or len(best_params_str.strip()) == 0:
            return np.array([np.nan, np.nan])

        # Parse param dictionary
        best_params = eval(best_params_str)  # or safer parse, e.g. json.loads

        # Build fresh model
        base_rf = RandomForestRegressor(n_jobs=-1, random_state=42)
        base_xgb = xgb.XGBRegressor(n_jobs=-1, random_state=42)
        
        meta_pipeline = Pipeline([
            ('scaler', StandardScaler()),
            ('meta_lin', LinearRegression())
        ])
        
        
        stacking_core = StackingRegressor(
            estimators=[
                ("rf", base_rf), 
                ("xgb", base_xgb)
            ],
            final_estimator=meta_pipeline,
            cv=3,
            n_jobs=-1
        )
        
        stacking_model = TransformedTargetRegressor(
            regressor=stacking_core,
            transformer=StandardScaler()
        )

        # Set best_params
        # Call set_params on the internal stacking core
        stacking_model.regressor.set_params(**best_params)

        if np.any(mask):
            X_c = X_train[mask, :]
            y_c = y_train[mask]
            stacking_model.fit(X_c, y_c)

            if X_test.ndim == 1:
                X_test = X_test.reshape(1, -1)

            preds = stacking_model.predict(X_test)
            # e.g., clamp negative if precipitation
            preds[preds < 0] = 0

            err = np.nan if np.isnan(y_test) else (y_test - preds)
            return np.array([err, preds]).squeeze()
        else:
            return np.array([np.nan, np.nan]).squeeze()

    # ----------------------------------------------------------------------
    # 3) PARALLELIZED MODEL TRAINING & PREDICTION OVER SPACE
    # ----------------------------------------------------------------------
    def compute_model(self, X_train, y_train, X_test, y_test, best_param_da):
        """
        Parallel fit/predict across the entire spatial domain, using cluster-based hyperparams.

        Parameters
        ----------
        X_train : xarray.DataArray
            Training data (predictors) with dims ('T','features').
        y_train : xarray.DataArray
            Training target with dims ('T','Y','X').
        X_test : xarray.DataArray
            Test data (predictors), shape (features,) or broadcastable across (Y, X).
        y_test : xarray.DataArray
            Test target with dims ('Y','X').
        best_param_da : xarray.DataArray
            The per-grid best_params from compute_hyperparameters (as strings).

        Returns
        -------
        xarray.DataArray
            dims ('output','Y','X'), where 'output' = [error, prediction].
        """
        chunksize_x = int(np.round(len(y_train.get_index("X")) / self.nb_cores))
        chunksize_y = int(np.round(len(y_train.get_index("Y")) / self.nb_cores))

        # Align time
        X_train['T'] = y_train['T']
        y_train = y_train.transpose('T', 'Y', 'X')

        # Squeeze test data
        X_test = X_test.squeeze()
        y_test = y_test.squeeze().transpose('Y','X')

        client = Client(n_workers=self.nb_cores, threads_per_worker=1)
        result_da = xr.apply_ufunc(
            self.fit_predict,
            X_train,
            y_train.chunk({'Y': chunksize_y, 'X': chunksize_x}),
            X_test,
            y_test.chunk({'Y': chunksize_y, 'X': chunksize_x}),
            best_param_da.chunk({'Y': chunksize_y, 'X': chunksize_x}),
            input_core_dims=[
                ('T','features'),  # X_train
                ('T',),           # y_train
                ('features',),    # X_test
                (),               # y_test
                ()                # best_params_str
            ],
            vectorize=True,
            dask='parallelized',
            output_core_dims=[('output',)],
            output_dtypes=[float],
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
        - best_guess : 1D array over T (hindcast_det or forecast_det)
        - T1, T2     : scalar terciles from climatological best-fit distribution
        - dist_code  : int, as in _ppf_terciles_from_code
        - shape, loc, scale : scalars from climatology fit

        Strategy:
        - For each time step, build a predictive distribution of the same family:
            * Use best_guess[t] to adjust mean / location;
            * Keep shape parameters from climatology.
        - Then compute probabilities:
            P(B) = F(T1), P(N) = F(T2) - F(T1), P(A) = 1 - F(T2).
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

    # ----------------------------------------------------------------------
    # 6) FORECAST METHOD
    # ----------------------------------------------------------------------
    def forecast(
        self, 
        Predictant, 
        clim_year_start, 
        clim_year_end, 
        Predictor, 
        hindcast_det, 
        Predictor_for_year, 
        best_param_da, best_code_da=None, best_shape_da=None, best_loc_da=None, best_scale_da=None
    ):
        """
        Generate a forecast for a single time (e.g., future year), then compute 
        tercile probabilities from the chosen distribution method.

        Parameters
        ----------
        Predictant : xarray.DataArray
            Observed data with dims (T, Y, X), used for climatological terciles.
        clim_year_start : int
            Start year of the climatology.
        clim_year_end : int
            End year of the climatology.
        Predictor : xarray.DataArray
            Historical predictor data, dims (T, features).
        hindcast_det : xarray.DataArray
            Historical deterministic forecast, dims (output=[error,prediction], T, Y, X).
            Used to compute error variance or error samples.
        Predictor_for_year : xarray.DataArray
            Predictor data for the forecast year, shape (features,) or (1, features).
        best_param_da : xarray.DataArray
            Grid-based hyperparameters from compute_hyperparameters.

        Returns
        -------
        result_ : xarray.DataArray
            dims ('output','Y','X') => [error, prediction].
            For a forecast, the 'error' will generally be NaN.
        hindcast_prob : xarray.DataArray
            dims (probability=3, Y, X) => tercile probabilities PB, PN, PA.
        """
        # We need a dummy y_test array, because fit_predict expects y_test
        # but we don't have actual future obs.
        y_test_dummy = xr.full_like(Predictant.isel(T=0), np.nan)  # shape (Y, X)

        # Prepare chunk sizes for parallel
        chunksize_x = int(np.round(len(Predictant.get_index("X")) / self.nb_cores))
        chunksize_y = int(np.round(len(Predictant.get_index("Y")) / self.nb_cores))

        # Align times, typically we set Predictor['T'] = Predictant['T']
        Predictor['T'] = Predictant['T']
        Predictant = Predictant.transpose('T', 'Y', 'X')
        Predictant_st = standardize_timeseries(Predictant, clim_year_start, clim_year_end)
        
        # Squeeze the forecast predictor
        Predictor_for_year_ = Predictor_for_year.squeeze()

        # 1) Fit+predict with the stacked model in parallel, returning [error, pred]
        client = Client(n_workers=self.nb_cores, threads_per_worker=1)
        result_da = xr.apply_ufunc(
            self.fit_predict,
            Predictor,                          # X_train
            Predictant_st.chunk({'Y': chunksize_y, 'X': chunksize_x}),  # y_train
            Predictor_for_year_,               # X_test
            y_test_dummy.chunk({'Y': chunksize_y, 'X': chunksize_x}), # y_test (dummy)
            best_param_da.chunk({'Y': chunksize_y, 'X': chunksize_x}),
            input_core_dims=[
                ('T','features'),  # X_train
                ('T',),           # y_train
                ('features',),    # X_test
                (),               # y_test
                ()                # best_params_str
            ],
            vectorize=True,
            dask='parallelized',
            output_core_dims=[('output',)],  # We'll get shape (2,) => [err, pred]
            output_dtypes=[float],
            dask_gufunc_kwargs={'output_sizes': {'output': 2}},
        )
        result_ = result_da.compute()
        client.close()
        result_ = result_.isel(output=1)
        
        result_ = reverse_standardize(result_, Predictant,
                                        clim_year_start, clim_year_end)
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
        return forecast_expanded, forecast_prob.transpose('probability', 'T', 'Y', 'X')


class WAS_RandomForest_XGBoost_Stacking_MLP:
    def __init__(
        self,
        nb_cores=1,
        dist_method="nonparam",
        n_clusters=5,
        param_grid=None,
        optimization_method="grid",  # New parameter
        n_trials=20,  # New parameter
        cv=5,  # New parameter
        random_state=42  # New parameter
    ):
        self.nb_cores = nb_cores
        self.dist_method = dist_method
        self.n_clusters = n_clusters
        self.optimization_method = optimization_method
        self.n_trials = n_trials
        self.cv = cv
        self.random_state = random_state

        # Define minimal param_grid if none is provided
        if param_grid is None:
            self.param_grid = {
                "rf__n_estimators": [50, 100],
                "xgb__max_depth": [3, 6],
                "xgb__learning_rate": [0.01, 0.1],
                "final_estimator__hidden_layer_sizes": [(50,), (30,10)],
                "final_estimator__activation": ["relu", "tanh"],
                "final_estimator__alpha": [0.0001, 0.001],
                "final_estimator__learning_rate_init": [0.001, 0.01]
            }
        else:
            self.param_grid = param_grid
            
        # Initialize optimizer
        self.optimizer = BaseOptimizer(
            optimization_method=optimization_method,
            n_trials=n_trials,
            cv=cv,
            random_state=random_state
        )

    def compute_hyperparameters(self, predictand, predictor, clim_year_start, clim_year_end):
        predictor['T'] = predictand['T']
        """Optimized version with Bayesian optimization."""
        df = (
            predictand.to_dataframe()
                      .reset_index()
                      .dropna()
                      .drop(columns=['T'])
        )
        col_name = df.columns[2]
        kmeans = KMeans(n_clusters=self.n_clusters, random_state=42)
        df["cluster"] = kmeans.fit_predict(df[[col_name]])
    
        df_unique = df.drop_duplicates(subset=["Y", "X"])
        dataset = df_unique.set_index(["Y", "X"]).to_xarray()
        
        cluster_da = (dataset["cluster"] *
                      xr.where(~np.isnan(predictand.isel(T=0)), 1, np.nan)
                     ).drop_vars("T", errors="ignore")
        _, cluster_da = xr.align(predictand, cluster_da, join="outer")
    
        # Build stacking model
        base_rf = RandomForestRegressor(n_jobs=-1, random_state=42)
        base_xgb = xgb.XGBRegressor(n_jobs=-1, random_state=42)
        
        # Wrap MLP (meta) in a pipeline with scaling
        meta_pipeline = Pipeline([
            ('scaler', StandardScaler()),  
            ('meta_mlp', MLPRegressor(random_state=42, max_iter=1000))
        ])
        
        stacking = StackingRegressor(
            estimators=[('rf', base_rf), ('xgb', base_xgb)],
            final_estimator=meta_pipeline,  
            n_jobs=-1
        )
        # Scale ONLY y, and automatically inverse-transform predictions back to original y units
        stacking_model = TransformedTargetRegressor(
            regressor=stacking,
            transformer=StandardScaler()   
        )    
        unique_clusters = np.unique(cluster_da)
        unique_clusters = unique_clusters[np.isfinite(unique_clusters)]
        best_params_for_cluster = {}
    
        for c in unique_clusters:
            mask_c = (cluster_da == c)
            y_cluster = (
                predictand.where(mask_c)
                          .mean(dim=["Y", "X"], skipna=True)
                          .dropna(dim="T")
            )
            if len(y_cluster["T"]) == 0:
                continue
    
            predictor_cluster = predictor.sel(T=y_cluster["T"])
            X_mat = predictor_cluster.values
            y_vec = y_cluster.values
    
            # Use optimizer to find best parameters
            best_params = self.optimizer.optimize(
                stacking_model,
                self.param_grid,
                X_mat,
                y_vec,
                scoring='neg_mean_squared_error'
            )
            best_params_for_cluster[int(c)] = best_params
    
        # Broadcast best hyperparameters
        best_param_da = xr.full_like(cluster_da, np.nan, dtype=object)
        for c, bp in best_params_for_cluster.items():
            c_mask = (cluster_da == c)
            best_param_da = best_param_da.where(~c_mask, other=str(bp))

        return best_param_da, cluster_da

    # -----------------------------------------------------------------
    # 2) FIT + PREDICT FOR A SINGLE GRID CELL
    # -----------------------------------------------------------------
    def fit_predict(self, X_train, y_train, X_test, y_test, best_params_str):
        """
        For a single grid cell, parse the local best_params dict, set them on the 
        StackingRegressor (with RF + XGB base, MLP meta), train and predict.
        
        Returns [error, prediction].
        
        Parameters
        ----------
        X_train : np.ndarray, shape (n_samples, n_features)
        y_train : np.ndarray, shape (n_samples,)
        X_test :  np.ndarray, shape (n_features,) or (1, n_features)
        y_test :  float or np.nan
        best_params_str : str
            Local best hyperparams as a stringified dict.

        Returns
        -------
        np.ndarray of shape (2,)
            [error, prediction]
        """
        mask = np.isfinite(y_train) & np.all(np.isfinite(X_train), axis=-1)

        # If there's no valid best_params or no data, return NaNs
        if not isinstance(best_params_str, str) or len(best_params_str.strip()) == 0:
            return np.array([np.nan, np.nan])

        # Parse the params
        best_params = eval(best_params_str)  
        
        # Build stacking model
        base_rf = RandomForestRegressor(n_jobs=-1, random_state=42)
        base_xgb = xgb.XGBRegressor(n_jobs=-1, random_state=42)
        
        # Wrap MLP (meta) in a pipeline with scaling
        meta_pipeline = Pipeline([
            ('scaler', StandardScaler()),  
            ('meta_mlp', MLPRegressor(random_state=42, max_iter=1000))
        ])
        
        stacking = StackingRegressor(
            estimators=[('rf', base_rf), ('xgb', base_xgb)],
            final_estimator=meta_pipeline,  
            n_jobs=-1
        )
        # Scale ONLY y, and automatically inverse-transform predictions back to original y units
        stacking_model = TransformedTargetRegressor(
            regressor=stacking,
            transformer=StandardScaler()   
        ) 
        # Apply local best params
        stacking_model.regressor.set_params(**best_params)

        if np.any(mask):
            X_c = X_train[mask, :]
            y_c = y_train[mask]

            stacking_model.fit(X_c, y_c)

            if X_test.ndim == 1:
                X_test = X_test.reshape(1, -1)

            preds = stacking_model.predict(X_test)
            preds[preds < 0] = 0  # clip negatives if it's precipitation
            err = np.nan if (np.isnan(y_test)) else (y_test - preds)
            return np.array([err, preds]).squeeze()
        else:
            return np.array([np.nan, np.nan]).squeeze()

    # -----------------------------------------------------------------
    # 3) PARALLELIZED COMPUTE_MODEL
    # -----------------------------------------------------------------
    def compute_model(self, X_train, y_train, X_test, y_test, best_param_da):
        """
        Parallel training + prediction across the entire spatial domain,
        referencing local best_params for each grid cell.

        Returns an xarray.DataArray with dim 'output' = [error, prediction].
        """
        # chunk sizes for parallel
        chunksize_x = int(np.round(len(y_train.get_index("X")) / self.nb_cores))
        chunksize_y = int(np.round(len(y_train.get_index("Y")) / self.nb_cores))

        # Align time
        X_train['T'] = y_train['T']
        y_train = y_train.transpose('T','Y','X')

        X_test = X_test.squeeze()
        y_test = y_test.squeeze().transpose('Y','X')

        # Parallel execution with Dask
        client = Client(n_workers=self.nb_cores, threads_per_worker=1)
        result_da = xr.apply_ufunc(
            self.fit_predict,
            X_train,
            y_train.chunk({'Y': chunksize_y, 'X': chunksize_x}),
            X_test,
            y_test.chunk({'Y': chunksize_y, 'X': chunksize_x}),
            best_param_da.chunk({'Y': chunksize_y, 'X': chunksize_x}),
            input_core_dims=[
                ('T','features'),
                ('T',),
                ('features',),
                (),
                ()
            ],
            vectorize=True,
            dask='parallelized',
            output_core_dims=[('output',)],
            output_dtypes=[float],
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
        - best_guess : 1D array over T (hindcast_det or forecast_det)
        - T1, T2     : scalar terciles from climatological best-fit distribution
        - dist_code  : int, as in _ppf_terciles_from_code
        - shape, loc, scale : scalars from climatology fit

        Strategy:
        - For each time step, build a predictive distribution of the same family:
            * Use best_guess[t] to adjust mean / location;
            * Keep shape parameters from climatology.
        - Then compute probabilities:
            P(B) = F(T1), P(N) = F(T2) - F(T1), P(A) = 1 - F(T2).
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


    # -----------------------------------------------------------------
    # 6) FORECAST METHOD
    # -----------------------------------------------------------------
    def forecast(
        self, 
        Predictant, 
        clim_year_start, 
        clim_year_end, 
        Predictor, 
        hindcast_det, 
        Predictor_for_year, 
        best_param_da, best_code_da=None, best_shape_da=None, best_loc_da=None, best_scale_da=None
    ):
        """
        Generate a forecast for a single future time (e.g., future year),
        then compute tercile probabilities from the chosen distribution method.

        Parameters
        ----------
        Predictant : xarray.DataArray
            Observed data with dims (T, Y, X) used for computing climatological terciles.
        clim_year_start : int
            Start year of the climatology period.
        clim_year_end : int
            End year of the climatology period.
        Predictor : xarray.DataArray
            Historical predictor data, shape (T, features).
        hindcast_det : xarray.DataArray
            Historical deterministic forecast with dims (output=[error,prediction], T, Y, X).
            Used to estimate error variance or error samples.
        Predictor_for_year : xarray.DataArray
            Predictor data for the forecast year, shape (features,) or (1, features).
        best_param_da : xarray.DataArray
            Grid-based best hyperparams from `compute_hyperparameters`.

        Returns
        -------
        result_ : xarray.DataArray
            dims ('output','Y','X') => [error, prediction].
            For a true forecast, the 'error' is typically NaN.
        hindcast_prob : xarray.DataArray
            dims (probability=3, Y, X) => PB, PN, PA tercile probabilities.
        """
        # 1) Provide a dummy y_test => shape (Y, X), all NaN
        y_test_dummy = xr.full_like(Predictant.isel(T=0), np.nan)

        # Prepare chunk sizes
        chunksize_x = int(np.round(len(Predictant.get_index("X")) / self.nb_cores))
        chunksize_y = int(np.round(len(Predictant.get_index("Y")) / self.nb_cores))

        # Align time
        Predictor['T'] = Predictant['T']
        Predictant = Predictant.transpose('T', 'Y', 'X')
        Predictor_for_year_ = Predictor_for_year.squeeze()
        # Predictant_st = standardize_timeseries(Predictant, clim_year_start, clim_year_end)
        
        # 2) Fit+predict in parallel => produce shape (2, Y, X) => [error, prediction]
        client = Client(n_workers=self.nb_cores, threads_per_worker=1)
        result_da = xr.apply_ufunc(
            self.fit_predict,
            Predictor,
            Predictant.chunk({'Y': chunksize_y, 'X': chunksize_x}),
            Predictor_for_year_,
            y_test_dummy.chunk({'Y': chunksize_y, 'X': chunksize_x}),
            best_param_da.chunk({'Y': chunksize_y, 'X': chunksize_x}),
            input_core_dims=[
                ('T','features'),  # X_train
                ('T',),           # y_train
                ('features',),    # X_test
                (),               # y_test (dummy)
                ()                # best_params_str
            ],
            vectorize=True,
            dask='parallelized',
            output_core_dims=[('output',)],
            output_dtypes=[float],
            dask_gufunc_kwargs={'output_sizes': {'output': 2}},
        )
        result_ = result_da.compute()
        client.close()
        result_ = result_.isel(output=1)
        # result_ = reverse_standardize(result_, Predictant, clim_year_start, clim_year_end)
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
        return forecast_expanded, forecast_prob.transpose('probability', 'T', 'Y', 'X')



class WAS_Stacking_Ridge:
    def __init__(
        self,
        nb_cores=1,
        dist_method="nonparam",
        n_clusters=5,
        param_grid=None,
        optimization_method="grid",  # New parameter
        n_trials=20,  # New parameter
        cv=5,  # New parameter
        random_state=42  # New parameter
    ):
        self.nb_cores = nb_cores
        self.dist_method = dist_method
        self.n_clusters = n_clusters
        self.optimization_method = optimization_method
        self.n_trials = n_trials
        self.cv = cv
        self.random_state = random_state

        # Minimal default grid if none is provided
        if param_grid is None:
            self.param_grid = {
                "rf__n_estimators": [50, 100],
                "xgb__max_depth": [3, 6],
                "mlp_base__hidden_layer_sizes": [(20,), (50, 10)],
                "mlp_base__activation": ["relu", "tanh"],
                "mlp_base__alpha": [0.0001, 0.001],
                "final_estimator__alpha": [0.1, 0.9, 5.0]
            }
        else:
            self.param_grid = param_grid
            
        # Initialize optimizer
        self.optimizer = BaseOptimizer(
            optimization_method=optimization_method,
            n_trials=n_trials,
            cv=cv,
            random_state=random_state
        )

    def compute_hyperparameters(self, predictand, predictor, clim_year_start, clim_year_end):
        """Optimized version with Bayesian optimization."""
        predictor['T'] = predictand['T']
        df = (
            predictand.to_dataframe()
                      .reset_index()
                      .dropna()
                      .drop(columns=['T'])
        )
        col_name = df.columns[2]
        kmeans = KMeans(n_clusters=self.n_clusters, random_state=42)
        df["cluster"] = kmeans.fit_predict(df[[col_name]])
    
        df_unique = df.drop_duplicates(subset=["Y", "X"])
        dataset = df_unique.set_index(["Y", "X"]).to_xarray()
    
        cluster_da = (dataset["cluster"] *
                      xr.where(~np.isnan(predictand.isel(T=0)), 1, np.nan)
                     ).drop_vars("T", errors="ignore")
        _, cluster_da = xr.align(predictand, cluster_da, join="outer")
    
        # Build stacking model
        rf_model = RandomForestRegressor(n_jobs=-1, random_state=42)
        xgb_model = xgb.XGBRegressor(n_jobs=-1, random_state=42)
        
        # # # Wrap MLP (base) in a pipeline with scaling
        # # mlp_pipeline = Pipeline([
        # #     ('scaler', StandardScaler()),  
        # #     ('mlp_base', MLPRegressor(random_state=42, max_iter=1000))
        # # ])
        # mlp_pipeline = TransformedTargetRegressor(
        #     regressor=MLPRegressor(random_state=42, max_iter=1000),
        #     transformer=StandardScaler()
        # )  
        # meta_pipeline = Pipeline([
        #     ('scaler', StandardScaler()),  
        #     ('ridge_meta', Ridge(alpha=0.8))
        # ])

        # stacking_ridge = StackingRegressor(
        #     estimators=[("rf", rf_model), ("xgb", xgb_model), ("mlp", mlp_pipeline)],
        #     final_estimator=meta_pipeline,
        #     n_jobs=-1
        # )
        
        mlp_base = Pipeline([
            ('scaler', StandardScaler()), 
            ('mlp', MLPRegressor(random_state=42, max_iter=1000))
        ])
        
        meta_pipeline = Pipeline([
            ('scaler', StandardScaler()),  
            ('ridge_meta', Ridge(alpha=0.8))
        ])
        
        stacking_core = StackingRegressor(
            estimators=[
                ("rf", rf_model),   # RF/XGB are scale-invariant but will target anomalies
                ("xgb", xgb_model), 
                ("mlp", mlp_base)
            ],
            final_estimator=meta_pipeline,
            cv=3,
            n_jobs=-1
        )
        
        stacking_ridge = TransformedTargetRegressor(
            regressor=stacking_core,
            transformer=StandardScaler()
        )
        unique_clusters = np.unique(cluster_da)
        unique_clusters = unique_clusters[np.isfinite(unique_clusters)]
        best_params_for_cluster = {}
    
        for c in unique_clusters:
            mask_c = (cluster_da == c)
            y_cluster = (
                predictand.where(mask_c)
                          .mean(dim=["Y", "X"], skipna=True)
                          .dropna(dim="T")
            )
            if len(y_cluster["T"]) == 0:
                continue
    
            predictor_cluster = predictor.sel(T=y_cluster["T"])
            X_mat = predictor_cluster.values
            y_vec = y_cluster.values
    
            # Use optimizer to find best parameters
            best_params = self.optimizer.optimize(
                stacking_ridge,
                self.param_grid,
                X_mat,
                y_vec,
                scoring='neg_mean_squared_error'
            )
            best_params_for_cluster[int(c)] = best_params
    
        # Broadcast best hyperparameters
        best_param_da = xr.full_like(cluster_da, np.nan, dtype=object)
        for c, bp in best_params_for_cluster.items():
            c_mask = (cluster_da == c)
            best_param_da = best_param_da.where(~c_mask, other=str(bp))

        return best_param_da, cluster_da

    # ------------------------------------------------------------------
    # 2) FIT + PREDICT FOR A SINGLE GRID CELL
    # ------------------------------------------------------------------
    def fit_predict(self, X_train, y_train, X_test, y_test, best_params_str):
        """
        For a single grid cell, parse the best params, instantiate the stacking regressor,
        fit to local data, and predict.

        Returns [error, prediction].
        """
        mask = np.isfinite(y_train) & np.all(np.isfinite(X_train), axis=-1)

        if not isinstance(best_params_str, str) or len(best_params_str.strip()) == 0:
            # No valid hyperparams => return NaN
            return np.array([np.nan, np.nan])

        # Parse param dict from string
        best_params = eval(best_params_str)  # or use a safer parser if you prefer

        mlp_base = Pipeline([
            ('scaler', StandardScaler()), 
            ('mlp', MLPRegressor(random_state=42, max_iter=1000))
        ])
        
        meta_pipeline = Pipeline([
            ('scaler', StandardScaler()),  
            ('ridge_meta', Ridge(alpha=0.8))
        ])
        
        stacking_core = StackingRegressor(
            estimators=[
                ("rf", rf_model),   # RF/XGB are scale-invariant but will target anomalies
                ("xgb", xgb_model), 
                ("mlp", mlp_base)
            ],
            final_estimator=meta_pipeline,
            cv=3,
            n_jobs=-1
        )
        
        stacking_ridge = TransformedTargetRegressor(
            regressor=stacking_core,
            transformer=StandardScaler()
        )

        # Apply local best params
        stacking_ridge.regressor.set_params(**best_params)

        if np.any(mask):
            X_c = X_train[mask, :]
            y_c = y_train[mask]
            stacking_ridge.fit(X_c, y_c)

            if X_test.ndim == 1:
                X_test = X_test.reshape(1, -1)

            preds = stacking_ridge.predict(X_test)
            preds[preds < 0] = 0  # clip negative if modeling precip
            err = np.nan if np.isnan(y_test) else (y_test - preds)
            return np.array([err, preds]).squeeze()
        else:
            return np.array([np.nan, np.nan])

    # ------------------------------------------------------------------
    # 3) PARALLEL MODELING ACROSS SPACE
    # ------------------------------------------------------------------
    def compute_model(self, X_train, y_train, X_test, y_test, best_param_da):
        """
        Parallel training + prediction across all spatial grid points.
        Uses local best hyperparams from best_param_da for each pixel.

        Returns an xarray.DataArray with dim ('output','Y','X') => [error, prediction].
        """
        chunksize_x = int(np.round(len(y_train.get_index("X")) / self.nb_cores))
        chunksize_y = int(np.round(len(y_train.get_index("Y")) / self.nb_cores))

        X_train['T'] = y_train['T']
        y_train = y_train.transpose('T','Y','X')
        X_test = X_test.squeeze()
        y_test = y_test.squeeze().transpose('Y','X')

        client = Client(n_workers=self.nb_cores, threads_per_worker=1)
        result_da = xr.apply_ufunc(
            self.fit_predict,
            X_train,
            y_train.chunk({'Y': chunksize_y, 'X': chunksize_x}),
            X_test,
            y_test.chunk({'Y': chunksize_y, 'X': chunksize_x}),
            best_param_da.chunk({'Y': chunksize_y, 'X': chunksize_x}),
            input_core_dims=[
                ('T','features'),  # X_train
                ('T',),           # y_train
                ('features',),    # X_test
                (),
                ()
            ],
            vectorize=True,
            dask='parallelized',
            output_core_dims=[('output',)],
            output_dtypes=[float],
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
        - best_guess : 1D array over T (hindcast_det or forecast_det)
        - T1, T2     : scalar terciles from climatological best-fit distribution
        - dist_code  : int, as in _ppf_terciles_from_code
        - shape, loc, scale : scalars from climatology fit

        Strategy:
        - For each time step, build a predictive distribution of the same family:
            * Use best_guess[t] to adjust mean / location;
            * Keep shape parameters from climatology.
        - Then compute probabilities:
            P(B) = F(T1), P(N) = F(T2) - F(T1), P(A) = 1 - F(T2).
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


    # ------------------------------------------------------------------
    # 6) FORECAST METHOD
    # ------------------------------------------------------------------
    def forecast(self, Predictant, clim_year_start, clim_year_end, Predictor, hindcast_det, Predictor_for_year, best_param_da, best_code_da=None, best_shape_da=None, best_loc_da=None, best_scale_da=None):
        """
        Forecast for a single future year (or time) and compute tercile probabilities.

        Parameters
        ----------
        Predictant : xarray.DataArray
            Observed data with dims (T, Y, X), used for computing climatology thresholds.
        clim_year_start : int
            Start of climatology period.
        clim_year_end : int
            End of climatology period.
        Predictor : xarray.DataArray
            Historical predictor data, shape (T, features).
        hindcast_det : xarray.DataArray
            Historical deterministic forecast with dims (output=[error,prediction], T, Y, X) 
            for computing error variance or samples.
        Predictor_for_year : xarray.DataArray
            Predictor data for the forecast year, shape (features,) or (1, features).
        best_param_da : xarray.DataArray
            Local best hyperparams from compute_hyperparameters, shape (Y, X).

        Returns
        -------
        result_ : xarray.DataArray
            dims (output=2, Y, X) => [error, prediction].
            In a real forecast, "error" is typically NaN since we have no future observation.
        hindcast_prob : xarray.DataArray
            dims (probability=3, Y, X) => [PB, PN, PA].
        """
        # Create a dummy y_test (NaN) for the forecast
        y_test_dummy = xr.full_like(Predictant.isel(T=0), np.nan)  # shape (Y, X)

        # Chunk sizes for parallel
        chunksize_x = int(np.round(len(Predictant.get_index("X")) / self.nb_cores))
        chunksize_y = int(np.round(len(Predictant.get_index("Y")) / self.nb_cores))

        # Align times
        Predictor['T'] = Predictant['T']
        Predictant = Predictant.transpose('T', 'Y', 'X')
        Predictor_for_year_ = Predictor_for_year.squeeze()
        # Predictant_st = standardize_timeseries(Predictant, clim_year_start, clim_year_end)
        
        # 1) Fit+predict in parallel => shape (2, Y, X)
        client = Client(n_workers=self.nb_cores, threads_per_worker=1)
        result_da = xr.apply_ufunc(
            self.fit_predict,
            Predictor,
            Predictant.chunk({'Y': chunksize_y, 'X': chunksize_x}),
            Predictor_for_year_,
            y_test_dummy.chunk({'Y': chunksize_y, 'X': chunksize_x}),     # dummy y_test
            best_param_da.chunk({'Y': chunksize_y, 'X': chunksize_x}),
            input_core_dims=[
                ('T','features'),  # X_train
                ('T',),           # y_train
                ('features',),    # X_test
                (),               # y_test
                ()                # best_params_str
            ],
            vectorize=True,
            dask='parallelized',
            output_core_dims=[('output',)],
            output_dtypes=[float],
            dask_gufunc_kwargs={'output_sizes': {'output': 2}},
        )
        result_ = result_da.compute()
        client.close()
        result_ = result_.isel(output=1)
        # result_ = reverse_standardize(result_, Predictant, clim_year_start, clim_year_end)
        
        # result_ => dims (output=2, Y, X). 
        # For a real future forecast, "error" is NaN, "prediction" is the forecast.

        # 2) Compute thresholds T1, T2 from climatology
        index_start = Predictant.get_index("T").get_loc(str(clim_year_start)).start
        index_end   = Predictant.get_index("T").get_loc(str(clim_year_end)).stop
        rainfall_for_tercile = Predictant.isel(T=slice(index_start, index_end))
        terciles = rainfall_for_tercile.quantile([0.32, 0.67], dim='T')
        T1 = terciles.isel(quantile=0).drop_vars('quantile')
        T2 = terciles.isel(quantile=1).drop_vars('quantile')
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
        return forecast_expanded, forecast_prob.transpose('probability', 'T', 'Y', 'X')



class WAS_LogisticRegression_Model:
    """
    Logistic regression for tercile classification (0/1/2) with:
      1) compute_class(): build tercile classes
      2) clustering on a spatial statistic of predictand (default: climatological mean)
      3) hyperparameter optimization per cluster (via BaseOptimizer)
      4) broadcast best params to (Y,X)
      5) fit/predict per grid cell using the locally broadcast params

    Notes:
    - "Scale y only" does not apply here because y is categorical (0/1/2).
    - X scaling is OFF by default, but you can enable it (x_scaler='standard' or 'robust').
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
    A class to perform Polynomial Regression on spatiotemporal datasets for climate prediction.

    This class is designed to work with Dask and Xarray for parallelized, high-performance 
    regression computations across large datasets with spatial and temporal dimensions. The primary 
    methods are for fitting the polynomial regression model, making predictions, and calculating 
    probabilistic predictions for climate terciles.

    Attributes
    ----------
    nb_cores : int, optional
        The number of CPU cores to use for parallel computation (default is 1).
    degree : int, optional
        The degree of the polynomial (default is 2).
    dist_method : str, optional
        The distribution method to compute tercile probabilities. One of 
        {"t", "gamma", "normal", "lognormal", "nonparam"} (default is "gamma").

    Methods
    -------
    fit_predict(x, y, x_test, y_test)
        Fits a Polynomial Regression model to the training data, predicts on test data, 
        and computes error.
    compute_model(X_train, y_train, X_test, y_test)
        Applies the Polynomial Regression model across a dataset using parallel computation 
        with Dask, returning predictions and error metrics.
    compute_prob(Predictant, clim_year_start, clim_year_end, Predictor, hindcast_det)
        Computes tercile probabilities for hindcast rainfall predictions 
        over specified climatological years.
    forecast(Predictant, clim_year_start, clim_year_end, Predictor, hindcast_det, Predictor_for_year)
        Generates a forecast for a single year (or time step) and calculates tercile probabilities 
        using the chosen distribution method.
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
            The method to compute tercile probabilities ("t", "gamma", "normal", "lognormal", "nonparam"), 
            by default "gamma".
        """
        self.nb_cores = nb_cores
        self.degree = degree
        self.dist_method = dist_method

    def fit_predict(self, x, y, x_test, y_test):
        """
        Fits a Polynomial Regression model to the provided training data, makes predictions 
        on the test data, and calculates the prediction error.
.
        Parameters
        ----------
        x : array-like, shape (n_samples, n_features)
            Training data (predictors).
        y : array-like, shape (n_samples,)
            Training targets.
        x_test : array-like, shape (n_features,) or (1, n_features)
            Test data (predictors) for which we want predictions.
        y_test : float
            Test target value (for computing error).

        Returns
        -------
        np.ndarray of shape (2,)
            Array containing [prediction_error, predicted_value].
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
        - best_guess : 1D array over T (hindcast_det or forecast_det)
        - T1, T2     : scalar terciles from climatological best-fit distribution
        - dist_code  : int, as in _ppf_terciles_from_code
        - shape, loc, scale : scalars from climatology fit

        Strategy:
        - For each time step, build a predictive distribution of the same family:
            * Use best_guess[t] to adjust mean / location;
            * Keep shape parameters from climatology.
        - Then compute probabilities:
            P(B) = F(T1), P(N) = F(T2) - F(T1), P(A) = 1 - F(T2).
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
    A class to perform Poisson Regression on spatiotemporal datasets for count data prediction.

    This class is designed to work with Dask and Xarray for parallelized, high-performance 
    regression computations across large datasets with spatial and temporal dimensions. The primary 
    methods are for fitting the Poisson regression model, making predictions, and calculating 
    probabilistic predictions for climate terciles.

    Attributes
    ----------
    nb_cores : int
        The number of CPU cores to use for parallel computation (default is 1).
    dist_method : str
        The method to use for tercile probability calculations, e.g. {"t", "gamma", "normal", 
        "lognormal", "nonparam"} (default is "gamma").

    Methods
    -------
    fit_predict(x, y, x_test, y_test)
        Fits a Poisson regression model to the training data, predicts on test data, and computes error.
    compute_model(X_train, y_train, X_test, y_test)
        Applies the Poisson regression model across a dataset using parallel computation 
        with Dask, returning predictions and error metrics.
    compute_prob(Predictant, clim_year_start, clim_year_end, Predictor, hindcast_det)
        Computes tercile probabilities for hindcast rainfall (or count data) predictions 
        over specified climatological years, using the chosen `dist_method`.
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
        - best_guess : 1D array over T (hindcast_det or forecast_det)
        - T1, T2     : scalar terciles from climatological best-fit distribution
        - dist_code  : int, as in _ppf_terciles_from_code
        - shape, loc, scale : scalars from climatology fit

        Strategy:
        - For each time step, build a predictive distribution of the same family:
            * Use best_guess[t] to adjust mean / location;
            * Keep shape parameters from climatology.
        - Then compute probabilities:
            P(B) = F(T1), P(N) = F(T2) - F(T1), P(A) = 1 - F(T2).
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
    A class to perform MARS-based modeling on spatiotemporal datasets for climate prediction.
    MARS stands for Multivariate Adaptive Regression Splines with Generalized Cross-Validation.

    This class is designed to work with Dask and Xarray for parallelized, high-performance 
    regression computations across large datasets with spatial and temporal dimensions. The primary 
    methods are for fitting the model, making predictions, and calculating probabilistic predictions 
    for climate terciles. 

    Attributes
    ----------
    nb_cores : int, optional
        The number of CPU cores to use for parallel computation (default is 1).
    dist_method : str, optional
        Distribution method for tercile probability calculations. One of
        {"t","gamma","normal","lognormal","nonparam"}. Default = "gamma".
    max_terms : int, optional
        Maximum number of basis functions for MARS (default: 21).
    max_degree : int, optional
        Maximum degree of interaction for MARS (default: 1).
    c : float, optional
        Cost parameter for effective parameters in GCV for MARS (default: 3).

    Methods
    -------
    fit_predict(x, y, x_test, y_test=None)
        Fits a MARS model, makes predictions, and calculates error if y_test is provided.

    compute_model(X_train, y_train, X_test, y_test)
        Applies the MARS model across a dataset using parallel computation with Dask.

    compute_prob(Predictant, clim_year_start, clim_year_end, Predictor, hindcast_det)
        Computes tercile probabilities for hindcast predictions over specified years.

    forecast(Predictant, clim_year_start, clim_year_end, Predictor, hindcast_det, Predictor_for_year)
        Generates a single-year forecast and computes tercile probabilities.
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
        - best_guess : 1D array over T (hindcast_det or forecast_det)
        - T1, T2     : scalar terciles from climatological best-fit distribution
        - dist_code  : int, as in _ppf_terciles_from_code
        - shape, loc, scale : scalars from climatology fit

        Strategy:
        - For each time step, build a predictive distribution of the same family:
            * Use best_guess[t] to adjust mean / location;
            * Keep shape parameters from climatology.
        - Then compute probabilities:
            P(B) = F(T1), P(N) = F(T2) - F(T1), P(A) = 1 - F(T2).
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
        Generates a single-year forecast using MARS, then computes 
        tercile probabilities using self.dist_method.

        Parameters
        ----------
        Predictant : xarray.DataArray
            Observed data with dims (T, Y, X).
        clim_year_start : int
            Start year for climatology
        clim_year_end : int
            End year for climatology
        Predictor : xarray.DataArray
            Historical predictor data with dims (T, features).
        hindcast_det : xarray.DataArray
            Historical deterministic forecast with dims (output=[error,prediction], T, Y, X).
        Predictor_for_year : xarray.DataArray
            Single-year predictor with shape (features,) or (1, features).

        Returns
        -------
        result_ : xarray.DataArray
            dims (output=2, Y, X) => [error, prediction]. 
            For a true forecast, error is typically NaN.
        hindcast_prob : xarray.DataArray
            dims (probability=3, Y, X) => [PB, PN, PA].
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