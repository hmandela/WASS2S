########  This code was developed by Mandela Houngnibo et al. within the framework of AGRHYMET WAS-RCC S2S. #################### Version 1.0.0 #########################

######################################################## Modules ########################################################

# Machine Learning and Statistical Modeling
from sklearn import linear_model
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor, StackingRegressor
from sklearn.svm import SVR
from sklearn.model_selection import GridSearchCV
from sklearn.preprocessing import PolynomialFeatures
import xgboost as xgb
import statsmodels.api as sm
from statsmodels.stats.outliers_influence import variance_inflation_factor as VIF
from statsmodels.stats.anova import anova_lm
from sklearn.neural_network import MLPRegressor
from sklearn.cluster import KMeans

# Data Manipulation and Analysis
import xarray as xr
import numpy as np
import pandas as pd

# Signal Processing and Interpolation
import scipy.signal as sig
from scipy.interpolate import CubicSpline
from scipy import stats
from scipy.stats import norm
from scipy.stats import lognorm
from scipy.stats import gamma

# EOF Analysis
import xeofs as xe

# Parallel Computing
from multiprocessing import cpu_count
from dask.distributed import Client
import dask.array as da

import numpy as np
import pandas as pd
import xarray as xr
from sklearn.svm import SVR
from sklearn.model_selection import GridSearchCV
from sklearn.cluster import KMeans
from scipy import stats
from dask.distributed import Client
from sklearn.svm import SVC
from wass2s.utils import *

class WAS_SVR:
    """
    A class to perform Support Vector Regression (SVR) on spatiotemporal datasets for climate prediction.

    This class is designed to work with Dask and Xarray for parallelized, high-performance 
    regression computations across large datasets with spatial and temporal dimensions. The primary 
    methods are for fitting the SVR model, making predictions, and calculating probabilistic predictions 
    for climate terciles.

    Attributes
    ----------
    nb_cores : int, optional
        The number of CPU cores to use for parallel computation (default is 1).
    n_clusters : int, optional
        The number of clusters to use in KMeans clustering (default is 5).
    kernel : str, optional
        Kernel type to be used in SVR ('linear', 'poly', 'rbf', or 'all') (default is 'linear').
    gamma : str, optional
        gamma of 'rbf' kernel function. Ignored by all other kernels, ["auto", "scale", None] by default None.
    C_range : list, optional
        List of C values to consider during hyperparameter tuning.
    epsilon_range : list, optional
        List of epsilon values to consider during hyperparameter tuning.
    degree_range : list, optional
        List of degrees to consider for the 'poly' kernel during hyperparameter tuning.
    dist_method : str, optional
        Distribution method ("gamma", "t", "normal", "lognormal", "nonparam") for probability calculations.
    """

    def __init__(
        self, 
        nb_cores=1, 
        n_clusters=5, 
        kernel='linear',
        gamma=None,
        C_range=[0.1, 1, 10, 100], 
        epsilon_range=[0.01, 0.1, 0.5, 1], 
        degree_range=[2, 3, 4],
        dist_method="nonparam"
    ):
        """
        Initializes the WAS_SVR with specified hyperparameter ranges.

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
        # Store all parameters so they are accessible throughout the class
        self.nb_cores = nb_cores
        self.n_clusters = n_clusters
        self.kernel = kernel
        self.gamma = gamma
        self.C_range = C_range
        self.epsilon_range = epsilon_range
        self.degree_range = degree_range
        self.dist_method = dist_method

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
        model = SVR(**model_params)

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

    def compute_hyperparameters(self, predictand, predictor):
        """
        Computes optimal SVR hyperparameters (C and epsilon) for each spatial cluster.

        We cluster the spatial grid based on the mean values in `predictand`, 
        then do a grid search for SVR hyperparameters on the average time series of each cluster.

        Parameters
        ----------
        predictand : xarray.DataArray
            Target variable with dimensions ('T', 'Y', 'X').
        predictor : xarray.DataArray
            Predictor variables with dimensions ('T', 'features').

        Returns
        -------
        C_array, epsilon_array, degree_array, Cluster
            DataArrays containing the best-fitting hyperparameters and cluster labels for each grid cell.
        """
        # Step 1: Perform KMeans clustering based on predictand's spatial distribution
        kmeans = KMeans(n_clusters=self.n_clusters, random_state=42)
        
        # Flatten spatial and drop time dimension to get a 2D array for KMeans
        predictand_dropna = predictand.to_dataframe().reset_index().dropna().drop(columns=['T'])
        variable_column = predictand_dropna.columns[2]
        predictand_dropna['cluster'] = kmeans.fit_predict(
            predictand_dropna[[variable_column]]
        )
        
        # Convert cluster assignments back into an xarray structure
        df_unique = predictand_dropna.drop_duplicates(subset=['Y', 'X'])
        dataset = df_unique.set_index(['Y', 'X']).to_xarray()
        mask = xr.where(~np.isnan(predictand.isel(T=0)), 1, np.nan)
        Cluster = (dataset['cluster'] * mask)
        
        # Align cluster array with the predictand array
        xarray1, xarray2 = xr.align(predictand, Cluster, join="outer")
        
        # Identify unique cluster labels
        clusters = np.unique(xarray2)
        clusters = clusters[~np.isnan(clusters)]
        
        # Compute mean time series for each cluster
        cluster_means = {
            int(cluster): xarray1.where(xarray2 == cluster).mean(dim=['Y', 'X'], skipna=True)
            for cluster in clusters
        }

        # Step 2: Prepare parameter grids depending on selected kernel(s)
        param_grid = []
        if self.kernel in ['linear', 'all']:
            param_grid.append({
                'kernel': ['linear'], 
                'C': self.C_range, 
                'epsilon': self.epsilon_range
            })
        if self.kernel in ['poly', 'all']:
            param_grid.append({
                'kernel': ['poly'], 
                'degree': self.degree_range, 
                'C': self.C_range, 
                'epsilon': self.epsilon_range
            })
        if self.kernel in ['rbf', 'all']:
            param_grid.append({
                'kernel': ['rbf'], 
                'C': self.C_range, 
                'epsilon': self.epsilon_range, 
                'gamma': self.gamma
            })

        # We'll use sklearn's GridSearchCV to test parameter combinations
        model = SVR()
        grid_search = GridSearchCV(model, param_grid, cv=5, scoring='neg_mean_squared_error')
    
        hyperparams_cluster = {}
        
        # Perform grid search for each cluster's mean time series
        for cluster_label in clusters:
            # Obtain the mean time series for this cluster
            cluster_mean = cluster_means[int(cluster_label)].dropna('T')

            # Ensure predictor time dimension aligns with the same time steps
            predictor['T'] = cluster_mean['T']
            common_times = np.intersect1d(cluster_mean['T'].values, predictor['T'].values)
            
            if len(common_times) == 0:
                # If there are no overlapping times, skip
                continue

            # Select the overlapping times
            cluster_mean_common = cluster_mean.sel(T=common_times)
            predictor_common = predictor.sel(T=common_times)

            y_cluster = cluster_mean_common.values
            if y_cluster.size > 0:
                # Perform grid search for this cluster
                grid_search.fit(predictor_common, y_cluster)
                best_params = grid_search.best_params_
                
                # Record best parameters for the cluster
                hyperparams_cluster[int(cluster_label)] = {
                    'C': best_params['C'],
                    'epsilon': best_params['epsilon'],
                    'kernel': best_params['kernel'],
                    'degree': best_params.get('degree', None),  # Only present if kernel='poly'
                    'gamma': best_params.get('gamma', None)     # Only present if kernel='rbf'
                }
    
        # Step 3: Create DataArrays for the best C, epsilon, etc. in each cluster
        C_array = xr.full_like(Cluster, np.nan, dtype=float)
        epsilon_array = xr.full_like(Cluster, np.nan, dtype=float)
        degree_array = xr.full_like(Cluster, np.nan, dtype=int)

        # Fill each DataArray with the cluster-specific values
        for cluster_label, params in hyperparams_cluster.items():
            mask = Cluster == cluster_label
            C_array = C_array.where(~mask, other=params['C'])
            epsilon_array = epsilon_array.where(~mask, other=params['epsilon'])
            degree_array = degree_array.where(~mask, other=params['degree'])
    
        # Align arrays in case of dimension differences
        C_array, epsilon_array, degree_array, Cluster, _ = xr.align(
            C_array, epsilon_array, degree_array, Cluster, predictand.isel(T=0).drop_vars('T').squeeze(), join="outer"
        )
        return C_array, epsilon_array, degree_array, Cluster

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
                    norm.ppf(0.3, loc=loc, scale=scale),
                    norm.ppf(0.7, loc=loc, scale=scale),
                )
            elif code == 2:
                return (
                    lognorm.ppf(0.3, s=shape, loc=loc, scale=scale),
                    lognorm.ppf(0.7, s=shape, loc=loc, scale=scale),
                )
            elif code == 3:
                return (
                    expon.ppf(0.3, loc=loc, scale=scale),
                    expon.ppf(0.7, loc=loc, scale=scale),
                )
            elif code == 4:
                return (
                    gamma.ppf(0.3, a=shape, loc=loc, scale=scale),
                    gamma.ppf(0.7, a=shape, loc=loc, scale=scale),
                )
            elif code == 5:
                return (
                    weibull_min.ppf(0.3, c=shape, loc=loc, scale=scale),
                    weibull_min.ppf(0.7, c=shape, loc=loc, scale=scale),
                )
            elif code == 6:
                # Note: Renamed 't_dist' to 't' for standard scipy.stats
                return (
                    t.ppf(0.3, df=shape, loc=loc, scale=scale),
                    t.ppf(0.7, df=shape, loc=loc, scale=scale),
                )
            elif code == 7:
                # Poisson: poisson.ppf(q, mu, loc=0)
                # ASSUMPTION: 'mu' (mean) is passed as 'shape'
                #             'loc' is passed as 'loc'
                #             'scale' is unused
                return (
                    poisson.ppf(0.3, mu=shape, loc=loc),
                    poisson.ppf(0.7, mu=shape, loc=loc),
                )
            elif code == 8:
                # Negative Binomial: nbinom.ppf(q, n, p, loc=0)
                # ASSUMPTION: 'n' (successes) is passed as 'shape'
                #             'p' (probability) is passed as 'scale'
                #             'loc' is passed as 'loc'
                return (
                    nbinom.ppf(0.3, n=shape, p=scale, loc=loc),
                    nbinom.ppf(0.7, n=shape, p=scale, loc=loc),
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
        terciles_emp = clim.quantile([0.3, 0.7], dim="T")
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
        terciles = rainfall_for_tercile.quantile([0.3, 0.7], dim='T')
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
                    norm.ppf(0.3, loc=loc, scale=scale),
                    norm.ppf(0.7, loc=loc, scale=scale),
                )
            elif code == 2:
                return (
                    lognorm.ppf(0.3, s=shape, loc=loc, scale=scale),
                    lognorm.ppf(0.7, s=shape, loc=loc, scale=scale),
                )
            elif code == 3:
                return (
                    expon.ppf(0.3, loc=loc, scale=scale),
                    expon.ppf(0.7, loc=loc, scale=scale),
                )
            elif code == 4:
                return (
                    gamma.ppf(0.3, a=shape, loc=loc, scale=scale),
                    gamma.ppf(0.7, a=shape, loc=loc, scale=scale),
                )
            elif code == 5:
                return (
                    weibull_min.ppf(0.3, c=shape, loc=loc, scale=scale),
                    weibull_min.ppf(0.7, c=shape, loc=loc, scale=scale),
                )
            elif code == 6:
                # Note: Renamed 't_dist' to 't' for standard scipy.stats
                return (
                    t.ppf(0.3, df=shape, loc=loc, scale=scale),
                    t.ppf(0.7, df=shape, loc=loc, scale=scale),
                )
            elif code == 7:
                # Poisson: poisson.ppf(q, mu, loc=0)
                # ASSUMPTION: 'mu' (mean) is passed as 'shape'
                #             'loc' is passed as 'loc'
                #             'scale' is unused
                return (
                    poisson.ppf(0.3, mu=shape, loc=loc),
                    poisson.ppf(0.7, mu=shape, loc=loc),
                )
            elif code == 8:
                # Negative Binomial: nbinom.ppf(q, n, p, loc=0)
                # ASSUMPTION: 'n' (successes) is passed as 'shape'
                #             'p' (probability) is passed as 'scale'
                #             'loc' is passed as 'loc'
                return (
                    nbinom.ppf(0.3, n=shape, p=scale, loc=loc),
                    nbinom.ppf(0.7, n=shape, p=scale, loc=loc),
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
        terciles_emp = clim.quantile([0.3, 0.7], dim="T")
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
        terciles = rainfall_for_tercile.quantile([0.3, 0.7], dim='T')
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
                    norm.ppf(0.3, loc=loc, scale=scale),
                    norm.ppf(0.7, loc=loc, scale=scale),
                )
            elif code == 2:
                return (
                    lognorm.ppf(0.3, s=shape, loc=loc, scale=scale),
                    lognorm.ppf(0.7, s=shape, loc=loc, scale=scale),
                )
            elif code == 3:
                return (
                    expon.ppf(0.3, loc=loc, scale=scale),
                    expon.ppf(0.7, loc=loc, scale=scale),
                )
            elif code == 4:
                return (
                    gamma.ppf(0.3, a=shape, loc=loc, scale=scale),
                    gamma.ppf(0.7, a=shape, loc=loc, scale=scale),
                )
            elif code == 5:
                return (
                    weibull_min.ppf(0.3, c=shape, loc=loc, scale=scale),
                    weibull_min.ppf(0.7, c=shape, loc=loc, scale=scale),
                )
            elif code == 6:
                # Note: Renamed 't_dist' to 't' for standard scipy.stats
                return (
                    t.ppf(0.3, df=shape, loc=loc, scale=scale),
                    t.ppf(0.7, df=shape, loc=loc, scale=scale),
                )
            elif code == 7:
                # Poisson: poisson.ppf(q, mu, loc=0)
                # ASSUMPTION: 'mu' (mean) is passed as 'shape'
                #             'loc' is passed as 'loc'
                #             'scale' is unused
                return (
                    poisson.ppf(0.3, mu=shape, loc=loc),
                    poisson.ppf(0.7, mu=shape, loc=loc),
                )
            elif code == 8:
                # Negative Binomial: nbinom.ppf(q, n, p, loc=0)
                # ASSUMPTION: 'n' (successes) is passed as 'shape'
                #             'p' (probability) is passed as 'scale'
                #             'loc' is passed as 'loc'
                return (
                    nbinom.ppf(0.3, n=shape, p=scale, loc=loc),
                    nbinom.ppf(0.7, n=shape, p=scale, loc=loc),
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
        terciles_emp = clim.quantile([0.3, 0.7], dim="T")
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
        terciles = rainfall_for_tercile.quantile([0.3, 0.7], dim='T')
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

class WAS_RandomForest_XGBoost_ML_Stacking:
    """
    A class to perform Stacking Ensemble with RandomForest + XGBoost as base learners
    and a LinearRegression as the meta-model. Also supports:
      - Hyperparameter tuning via KMeans + GridSearchCV
      - Parallel spatiotemporal training/prediction using xarray + Dask
      - Probability computation (terciles) under different distributions.

    Parameters
    ----------
    nb_cores : int, optional
        Number of CPU cores to use for parallel computation (default=1).
    dist_method : str, optional
        Distribution method for tercile probability calculations. 
        One of {'gamma', 't', 'normal', 'lognormal', 'nonparam'} (default='gamma').
    n_clusters : int, optional
        Number of clusters for KMeans (default=5).
    param_grid : dict or None, optional
        The hyperparameter grid for GridSearchCV over the StackingRegressor. 
        If None, uses a default small example grid.

    Notes
    -----
    In scikit-learn, you can reference parameters inside stacking base estimators
    with naming like "estimators__rf__n_estimators", "estimators__xgb__learning_rate", etc. 
    The exact syntax can vary by sklearn version.
    """

    def __init__(
        self,
        nb_cores=1,
        dist_method="nonparam",
        n_clusters=5,
        param_grid=None
    ):
        self.nb_cores = nb_cores
        self.dist_method = dist_method
        self.n_clusters = n_clusters

        # Define a minimal default param_grid if none is provided.
        if param_grid is None:
            self.param_grid = {
                "rf__n_estimators": [5, 10],
                "xgb__learning_rate": [0.05, 0.1],
                "xgb__max_depth": [2, 4],
                "final_estimator__fit_intercept": [True, False]
            }
        else:
            self.param_grid = param_grid

    # ----------------------------------------------------------------------
    # 1) HYPERPARAMETER TUNING WITH KMEANS + GRID SEARCH
    # ----------------------------------------------------------------------
    def compute_hyperparameters(self, predictand, predictor):
        """
        Cluster grid cells (Y,X) via KMeans on the mean of `predictand` (over T).
        Then for each cluster, run a cross-validation GridSearch over a StackingRegressor
        to find best hyperparameters. Store results in DataArrays.

        Parameters
        ----------
        predictand : xarray.DataArray
            Target variable with dims ('T','Y','X').
        predictor : xarray.DataArray
            Predictor variables with dims ('T','features').

        Returns
        -------
        best_param_da : xarray.DataArray (dtype=object or str)
            A DataArray holding best hyperparameter sets (as strings) for each grid cell.
        cluster_da : xarray.DataArray
            The integer cluster assignment for each (Y, X).
        """
        df = (
            predictand.to_dataframe()
                      .reset_index()
                      .dropna()
                      .drop(columns=['T'])
        )
        # Use the first data column as the representative value
        col_name = df.columns[2]
        kmeans = KMeans(n_clusters=self.n_clusters, random_state=42)
        df["cluster"] = kmeans.fit_predict(df[[col_name]])
    
        # Drop duplicate (Y,X) rows
        df_unique = df.drop_duplicates(subset=["Y", "X"])
        dataset = df_unique.set_index(["Y", "X"]).to_xarray()
    
        # Mask out invalid cells (using the first time slice of predictand)
        cluster_da = (dataset["cluster"] *
                      xr.where(~np.isnan(predictand.isel(T=0)), 1, np.nan)
                     ).drop_vars("T", errors="ignore")
    
        # Align with original predictand
        _, cluster_da = xr.align(predictand, cluster_da, join="outer")
    
        # --- (b) Set up the stacking model and grid search ---
        base_rf = RandomForestRegressor(n_jobs=-1, random_state=42)
        base_xgb = xgb.XGBRegressor(n_jobs=-1, random_state=42)
        meta_lin = LinearRegression()
        stacking_model = StackingRegressor(
            estimators=[("rf", base_rf), ("xgb", base_xgb)],
            final_estimator=meta_lin,
            n_jobs=-1
        )
    
        grid_search = GridSearchCV(
            estimator=stacking_model,
            param_grid=self.param_grid,
            cv=5,
            scoring='neg_mean_squared_error',
            n_jobs=-1
        )
    
        # --- (c) For each cluster, compute the cluster-mean time series and run grid search ---
        unique_clusters = np.unique(cluster_da)
        unique_clusters = unique_clusters[np.isfinite(unique_clusters)]
        best_params_for_cluster = {}
    
        for c in unique_clusters:
            mask_c = (cluster_da == c)
            # Aggregate the predictand over Y and X for this cluster to get a time series
            y_cluster = (
                predictand.where(mask_c)
                          .mean(dim=["Y", "X"], skipna=True)
                          .dropna(dim="T")
            )
            if len(y_cluster["T"]) == 0:
                continue
    
            # Select predictor data corresponding to the times in y_cluster
            predictor_cluster = predictor.sel(T=y_cluster["T"])
            X_mat = predictor_cluster.values  # shape: (time, features)
            y_vec = y_cluster.values          # shape: (time,)
    
            grid_search.fit(X_mat, y_vec)
            best_params_for_cluster[int(c)] = grid_search.best_params_
    
        # --- (d) Broadcast best hyperparameter sets (as strings) back to each grid cell ---
        best_param_da = xr.full_like(cluster_da, np.nan, dtype=object)
        for c, bp in best_params_for_cluster.items():
            c_mask = (cluster_da == c)
            best_param_da = best_param_da.where(~c_mask, other=str(bp))
    
        # Align best_param_da with predictand dimensions if necessary
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
        base_rf = RandomForestRegressor(n_jobs=1, random_state=42)
        base_xgb = xgb.XGBRegressor(n_jobs=1, random_state=42)
        meta_lin = LinearRegression()
        stacking_model = StackingRegressor(
            estimators=[("rf", base_rf), ("xgb", base_xgb)],
            final_estimator=meta_lin,
            n_jobs=1
        )

        # Set best_params
        stacking_model.set_params(**best_params)

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
                    norm.ppf(0.3, loc=loc, scale=scale),
                    norm.ppf(0.7, loc=loc, scale=scale),
                )
            elif code == 2:
                return (
                    lognorm.ppf(0.3, s=shape, loc=loc, scale=scale),
                    lognorm.ppf(0.7, s=shape, loc=loc, scale=scale),
                )
            elif code == 3:
                return (
                    expon.ppf(0.3, loc=loc, scale=scale),
                    expon.ppf(0.7, loc=loc, scale=scale),
                )
            elif code == 4:
                return (
                    gamma.ppf(0.3, a=shape, loc=loc, scale=scale),
                    gamma.ppf(0.7, a=shape, loc=loc, scale=scale),
                )
            elif code == 5:
                return (
                    weibull_min.ppf(0.3, c=shape, loc=loc, scale=scale),
                    weibull_min.ppf(0.7, c=shape, loc=loc, scale=scale),
                )
            elif code == 6:
                # Note: Renamed 't_dist' to 't' for standard scipy.stats
                return (
                    t.ppf(0.3, df=shape, loc=loc, scale=scale),
                    t.ppf(0.7, df=shape, loc=loc, scale=scale),
                )
            elif code == 7:
                # Poisson: poisson.ppf(q, mu, loc=0)
                # ASSUMPTION: 'mu' (mean) is passed as 'shape'
                #             'loc' is passed as 'loc'
                #             'scale' is unused
                return (
                    poisson.ppf(0.3, mu=shape, loc=loc),
                    poisson.ppf(0.7, mu=shape, loc=loc),
                )
            elif code == 8:
                # Negative Binomial: nbinom.ppf(q, n, p, loc=0)
                # ASSUMPTION: 'n' (successes) is passed as 'shape'
                #             'p' (probability) is passed as 'scale'
                #             'loc' is passed as 'loc'
                return (
                    nbinom.ppf(0.3, n=shape, p=scale, loc=loc),
                    nbinom.ppf(0.7, n=shape, p=scale, loc=loc),
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
        terciles_emp = clim.quantile([0.3, 0.7], dim="T")
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
        terciles = rainfall_for_tercile.quantile([0.3, 0.7], dim='T')
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
    ):
        self.nb_cores = nb_cores
        self.dist_method = dist_method
        self.n_clusters = n_clusters
        
        # If no param_grid is provided, create a minimal default grid.
        if param_grid is None:
            self.param_grid = {
                'hidden_layer_sizes': [(10,5), (10,)],
                'activation': ['relu', 'tanh', 'sigm'],
                'solver': ['adam'],
                'learning_rate_init': [0.01, 0.1],
                'max_iter': [2000, 6000, 10000]
            }
        else:
            self.param_grid = param_grid

    # ------------------------------------------------------------------
    # 1) HYPERPARAMETER TUNING VIA CLUSTERING + GRID SEARCH
    # ------------------------------------------------------------------
    def compute_hyperparameters(self, predictand, predictor):
        """
        Performs KMeans clustering on the spatial mean of `predictand`, then for each cluster
        runs a cross-validation grid search on MLP hyperparameters using the cluster-mean time series.

        Parameters
        ----------
        predictand : xarray.DataArray
            Target variable with dimensions ('T', 'Y', 'X').
        predictor : xarray.DataArray
            Predictor variables with dimensions ('T', 'features').

        Returns
        -------
        hl_array, act_array, lr_array, cluster_da : xarray.DataArray
            DataArrays storing the best local hyperparameters for each grid cell
            (derived from cluster membership) and the cluster assignments.
            Note: We show example outputs for hidden_layer_sizes, activation, learning_rate_init. 
                  You can extend this to all parameters in your `param_grid`.
        """
        predictand_ = standardize_timeseries(predictand)

        # (a) KMeans clustering on predictand (dropping the 'T' dimension)
        kmeans = KMeans(n_clusters=self.n_clusters, random_state=42)
        predictand_dropna = (
            predictand.to_dataframe()
                      .reset_index()
                      .dropna()
                      .drop(columns=['T'])
        )
        
        # Use one representative column (e.g., the first data column) for clustering.
        col_name = predictand_dropna.columns[2]
        predictand_dropna['cluster'] = kmeans.fit_predict(
            predictand_dropna[[col_name]]
        )
        # Convert clusters back to xarray
        df_unique = predictand_dropna.drop_duplicates(subset=['Y', 'X'])
        dataset = df_unique.set_index(['Y', 'X']).to_xarray()
        
        # Mask out invalid cells (using the first time slice of predictand)
        cluster_da = (dataset['cluster'] *
                      xr.where(~np.isnan(predictand.isel(T=0)), 1, np.nan)
                     ).drop_vars("T", errors='ignore')
        
        # Align with original predictand
        _, cluster_da = xr.align(predictand, cluster_da, join="outer")
        clusters = np.unique(cluster_da)
        clusters = clusters[~np.isnan(clusters)]
    
        # (b) Prepare GridSearchCV for MLP
        grid_search = GridSearchCV(
            estimator=MLPRegressor(),
            param_grid=self.param_grid,
            cv=3,  # or use a time-series split if needed
            scoring='neg_mean_squared_error',
            n_jobs=-1
        )
        
        best_params_for_cluster = {}
    
        # For each cluster, run grid search on the cluster-averaged time series
        for c in clusters:
            mask_c = (cluster_da == c)
            # Compute the mean over the spatial dimensions for the cluster
            y_cluster = (
                predictand_.where(mask_c)
                          .mean(dim=['Y', 'X'], skipna=True)
                          .dropna(dim='T')
            )
            if len(y_cluster['T']) == 0:
                continue
            # Select the corresponding times in predictor
            predictor_cluster = predictor.sel(T=y_cluster['T'])
            X_mat = predictor_cluster.values  # (time, features)
            y_vec = y_cluster.values          # (time,)
            
            grid_search.fit(X_mat, y_vec)
            best_params_for_cluster[int(c)] = grid_search.best_params_
    
        # (c) Broadcast best hyperparameters to each grid cell
        hl_array  = xr.full_like(cluster_da, np.nan, dtype=object)
        act_array = xr.full_like(cluster_da, np.nan, dtype=object)
        lr_array  = xr.full_like(cluster_da, np.nan, dtype=float)
        maxiter_array  = xr.full_like(cluster_da, np.nan, dtype=float)
        
        for c, bp in best_params_for_cluster.items():
            c_mask = (cluster_da == c)
            hl_str  = str(bp.get('hidden_layer_sizes', None))
            act_str = bp.get('activation', None)
            lr_val  = bp.get('learning_rate_init', np.nan)
            maxiter_val = bp.get('max_iter', np.nan)
            hl_array  = hl_array.where(~c_mask, other=hl_str)
            act_array = act_array.where(~c_mask, other=act_str)
            lr_array  = lr_array.where(~c_mask,  other=lr_val)
            maxiter_array  = maxiter_array.where(~c_mask,  other=maxiter_val)

        return hl_array, act_array, lr_array, maxiter_array, cluster_da

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
        mlp_model = MLPRegressor(
            hidden_layer_sizes=hl_sizes if hl_sizes else (10,5),
            activation=activation if activation else 'relu',
            solver='adam',
            max_iter=int(maxiter) if not np.isnan(maxiter) else 6000,
            learning_rate_init=lr_init if not np.isnan(lr_init) else 0.001
            # learning_rate_init=lr_init if lr_init else 0.001
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
                    norm.ppf(0.3, loc=loc, scale=scale),
                    norm.ppf(0.7, loc=loc, scale=scale),
                )
            elif code == 2:
                return (
                    lognorm.ppf(0.3, s=shape, loc=loc, scale=scale),
                    lognorm.ppf(0.7, s=shape, loc=loc, scale=scale),
                )
            elif code == 3:
                return (
                    expon.ppf(0.3, loc=loc, scale=scale),
                    expon.ppf(0.7, loc=loc, scale=scale),
                )
            elif code == 4:
                return (
                    gamma.ppf(0.3, a=shape, loc=loc, scale=scale),
                    gamma.ppf(0.7, a=shape, loc=loc, scale=scale),
                )
            elif code == 5:
                return (
                    weibull_min.ppf(0.3, c=shape, loc=loc, scale=scale),
                    weibull_min.ppf(0.7, c=shape, loc=loc, scale=scale),
                )
            elif code == 6:
                # Note: Renamed 't_dist' to 't' for standard scipy.stats
                return (
                    t.ppf(0.3, df=shape, loc=loc, scale=scale),
                    t.ppf(0.7, df=shape, loc=loc, scale=scale),
                )
            elif code == 7:
                # Poisson: poisson.ppf(q, mu, loc=0)
                # ASSUMPTION: 'mu' (mean) is passed as 'shape'
                #             'loc' is passed as 'loc'
                #             'scale' is unused
                return (
                    poisson.ppf(0.3, mu=shape, loc=loc),
                    poisson.ppf(0.7, mu=shape, loc=loc),
                )
            elif code == 8:
                # Negative Binomial: nbinom.ppf(q, n, p, loc=0)
                # ASSUMPTION: 'n' (successes) is passed as 'shape'
                #             'p' (probability) is passed as 'scale'
                #             'loc' is passed as 'loc'
                return (
                    nbinom.ppf(0.3, n=shape, p=scale, loc=loc),
                    nbinom.ppf(0.7, n=shape, p=scale, loc=loc),
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
        terciles_emp = clim.quantile([0.3, 0.7], dim="T")
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
        terciles = rainfall_for_tercile.quantile([0.3, 0.7], dim='T')
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
    """
    A class that performs stacking of RandomForest + XGBoost (base learners)
    and an MLPRegressor (meta-learner). Features:

      - Hyperparameter tuning via cluster-based GridSearchCV
      - Parallel spatiotemporal training/prediction
      - Tercile probability calculations with various distributions

    Parameters
    ----------
    nb_cores : int
        Number of CPU cores to use for parallel computation.
    dist_method : str
        Distribution method for tercile probability calculations.
        One of {'gamma', 't', 'normal', 'lognormal', 'nonparam'}.
    n_clusters : int
        Number of clusters for KMeans.
    param_grid : dict or None
        Hyperparameter search grid for GridSearchCV. If None, a minimal default is used.

    Notes
    -----
    - When referencing parameters inside stacking estimators, scikit-learn uses
      "estimators__<est_name>__<param_name>" for base models, or
      "final_estimator__<param>" for the meta-model. For example:
        - "estimators__rf__n_estimators" => sets n_estimators for 'rf'
        - "estimators__xgb__max_depth"   => sets max_depth for 'xgb'
        - "final_estimator__hidden_layer_sizes" => sets hidden_layer_sizes in MLPRegressor
    """

    def __init__(
        self,
        nb_cores=1,
        dist_method="nonparam",
        n_clusters=5,
        param_grid=None
    ):
        self.nb_cores = nb_cores
        self.dist_method = dist_method
        self.n_clusters = n_clusters

        # Define a minimal param_grid if none is provided
        if param_grid is None:
            self.param_grid = {
                # Example hyperparams for RandomForest
                "rf__n_estimators": [50, 100],
                # Example hyperparams for XGBoost
                "xgb__max_depth": [3, 5],
                "xgb__learning_rate": [0.01, 0.1],
                # Example hyperparams for MLP meta-learner
                "final_estimator__hidden_layer_sizes": [(50,), (30,10)],
                "final_estimator__activation": ["relu", "tanh"],
            }
        else:
            self.param_grid = param_grid

    # -----------------------------------------------------------------
    # 1) HYPERPARAMETER TUNING VIA KMEANS + GRID SEARCH
    # -----------------------------------------------------------------
    def compute_hyperparameters(self, predictand, predictor):
        """
        Cluster grid cells (Y,X) via KMeans on the mean of `predictand`.
        For each cluster, run GridSearchCV on a StackingRegressor that has:
         - RandomForest (rf) + XGBoost (xgb) as base learners
         - MLPRegressor as meta-learner

        Parameters
        ----------
        predictand : xarray.DataArray
            Target variable with dims ('T', 'Y', 'X').
        predictor : xarray.DataArray
            Predictor variables with dims ('T','features').

        Returns
        -------
        best_param_da : xarray.DataArray
            DataArray storing the best hyperparams (as strings) for each grid cell.
        cluster_da : xarray.DataArray
            The integer cluster assignment for each (Y,X).
        """
        # (a) KMeans clustering on a representative predictand (dropping 'T')
        # Convert to DataFrame, drop the time column, remove rows with missing values,
        # and drop duplicate (Y,X) so that each grid cell appears only once.
        df = (
            predictand.to_dataframe()
                      .reset_index()
                      .dropna()
                      .drop(columns=['T'])
        )
        # Use one representative column (e.g., the first data column) for clustering.
        col_name = df.columns[2]
        kmeans = KMeans(n_clusters=self.n_clusters, random_state=42)
        df["cluster"] = kmeans.fit_predict(df[[col_name]])
    
        # Convert clusters back to xarray: drop duplicates so that each (Y,X) appears once.
        df_unique = df.drop_duplicates(subset=["Y", "X"])
        dataset = df_unique.set_index(["Y", "X"]).to_xarray()
        
        # Mask out invalid cells (using the first time slice of predictand)
        cluster_da = (dataset["cluster"] *
                      xr.where(~np.isnan(predictand.isel(T=0)), 1, np.nan)
                     ).drop_vars("T", errors="ignore")
        # Align cluster_da with the original predictand
        _, cluster_da = xr.align(predictand, cluster_da, join="outer")
    
        # (b) Prepare the stacking model template
        base_rf = RandomForestRegressor(n_jobs=-1, random_state=42)
        base_xgb = xgb.XGBRegressor(n_jobs=-1, random_state=42)
        meta_mlp = MLPRegressor(random_state=42)
        stacking_model = StackingRegressor(
            estimators=[('rf', base_rf), ('xgb', base_xgb)],
            final_estimator=meta_mlp,
            n_jobs=-1
        )
    
        grid_search = GridSearchCV(
            estimator=stacking_model,
            param_grid=self.param_grid,
            cv=5,
            scoring='neg_mean_squared_error',
            n_jobs=-1
        )
    
        unique_clusters = np.unique(cluster_da)
        unique_clusters = unique_clusters[np.isfinite(unique_clusters)]
        best_params_for_cluster = {}
    
        # (c) For each cluster, run cross-validation on the cluster-aggregated time series.
        for c in unique_clusters:
            mask_c = (cluster_da == c)
            # Aggregate predictand for grid cells in cluster c by averaging over Y, X.
            y_cluster = (
                predictand.where(mask_c)
                          .mean(dim=["Y", "X"], skipna=True)
                          .dropna(dim="T")
            )
            if len(y_cluster["T"]) == 0:
                continue
    
            # Select predictor values for the same time stamps.
            predictor_cluster = predictor.sel(T=y_cluster["T"])
            X_mat = predictor_cluster.values  # shape: (time, features)
            y_vec = y_cluster.values          # shape: (time,)
    
            grid_search.fit(X_mat, y_vec)
            best_params_for_cluster[int(c)] = grid_search.best_params_
    
        # (d) Broadcast the best hyperparameters to each grid cell (stored as strings)
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
        best_params = eval(best_params_str)  # could use json.loads(...) if you prefer

        # Create fresh base models & meta-model
        base_rf = RandomForestRegressor(n_jobs=-1, random_state=42)
        base_xgb = xgb.XGBRegressor(n_jobs=-1, random_state=42)
        meta_mlp = MLPRegressor(random_state=42)
        stacking_model = StackingRegressor(
            estimators=[('rf', base_rf), ('xgb', base_xgb)],
            final_estimator=meta_mlp,
            n_jobs=-1
        )

        # Apply local best params
        stacking_model.set_params(**best_params)

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
                    norm.ppf(0.3, loc=loc, scale=scale),
                    norm.ppf(0.7, loc=loc, scale=scale),
                )
            elif code == 2:
                return (
                    lognorm.ppf(0.3, s=shape, loc=loc, scale=scale),
                    lognorm.ppf(0.7, s=shape, loc=loc, scale=scale),
                )
            elif code == 3:
                return (
                    expon.ppf(0.3, loc=loc, scale=scale),
                    expon.ppf(0.7, loc=loc, scale=scale),
                )
            elif code == 4:
                return (
                    gamma.ppf(0.3, a=shape, loc=loc, scale=scale),
                    gamma.ppf(0.7, a=shape, loc=loc, scale=scale),
                )
            elif code == 5:
                return (
                    weibull_min.ppf(0.3, c=shape, loc=loc, scale=scale),
                    weibull_min.ppf(0.7, c=shape, loc=loc, scale=scale),
                )
            elif code == 6:
                # Note: Renamed 't_dist' to 't' for standard scipy.stats
                return (
                    t.ppf(0.3, df=shape, loc=loc, scale=scale),
                    t.ppf(0.7, df=shape, loc=loc, scale=scale),
                )
            elif code == 7:
                # Poisson: poisson.ppf(q, mu, loc=0)
                # ASSUMPTION: 'mu' (mean) is passed as 'shape'
                #             'loc' is passed as 'loc'
                #             'scale' is unused
                return (
                    poisson.ppf(0.3, mu=shape, loc=loc),
                    poisson.ppf(0.7, mu=shape, loc=loc),
                )
            elif code == 8:
                # Negative Binomial: nbinom.ppf(q, n, p, loc=0)
                # ASSUMPTION: 'n' (successes) is passed as 'shape'
                #             'p' (probability) is passed as 'scale'
                #             'loc' is passed as 'loc'
                return (
                    nbinom.ppf(0.3, n=shape, p=scale, loc=loc),
                    nbinom.ppf(0.7, n=shape, p=scale, loc=loc),
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
        terciles_emp = clim.quantile([0.3, 0.7], dim="T")
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
        Predictant_st = standardize_timeseries(Predictant, clim_year_start, clim_year_end)
        
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
        result_ = reverse_standardize(result_, Predictant,
                                        clim_year_start, clim_year_end)
        # result_ => dims (output=2, Y, X). 
        # For a real future forecast, "error" is NaN, "prediction" is the forecast.

        # 2) Compute thresholds T1, T2 from climatology
        index_start = Predictant.get_index("T").get_loc(str(clim_year_start)).start
        index_end   = Predictant.get_index("T").get_loc(str(clim_year_end)).stop
        rainfall_for_tercile = Predictant.isel(T=slice(index_start, index_end))
        terciles = rainfall_for_tercile.quantile([0.3, 0.7], dim='T')
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
    """
    A class that performs stacking of the following base learners:
      - RandomForestRegressor (rf)
      - XGBRegressor (xgb)
      - MLPRegressor (mlp_base)
    and uses Ridge as the meta-model.

    Like the previous classes, this supports:
      - Cluster-based hyperparameter tuning via KMeans + GridSearchCV
      - Parallel spatiotemporal training/prediction with xarray + dask
      - Various distribution methods for tercile probability calculations.

    Parameters
    ----------
    nb_cores : int
        Number of CPU cores to use for parallel computation.
    dist_method : str
        Distribution method for tercile probability calculations:
        One of {'gamma', 't', 'normal', 'lognormal', 'nonparam'}.
    n_clusters : int
        Number of clusters for KMeans (used in hyperparameter tuning).
    param_grid : dict or None
        Hyperparameter grid for GridSearchCV. If None, a minimal default is used.

    Example for param_grid:
      {
        "estimators__rf__n_estimators": [50, 100],
        "estimators__xgb__max_depth": [3, 6],
        "estimators__mlp_base__hidden_layer_sizes": [(20,), (50, 10)],
        "final_estimator__alpha": [0.1, 0.9, 5.0],
      }

    Methods
    -------
    compute_hyperparameters(predictand, predictor)
        Performs cluster-based hyperparam tuning, returns best-param DataArray.
    fit_predict(X_train, y_train, X_test, y_test, best_params_str)
        Trains a local stacking model with the best hyperparams for that grid cell, then predicts.
    compute_model(X_train, y_train, X_test, y_test, best_param_da)
        Calls fit_predict(...) in parallel across all grid cells.
    compute_prob(Predictant, clim_year_start, clim_year_end, hindcast_det)
        Computes tercile probabilities using self.dist_method.
    forecast(Predictant, clim_year_start, clim_year_end, Predictor, hindcast_det, Predictor_for_year, best_param_da)
        Fits a forecast for a single future year (or time) and calculates tercile probabilities.
    """

    def __init__(
        self,
        nb_cores=1,
        dist_method="nonparam",
        n_clusters=5,
        param_grid=None
    ):
        self.nb_cores = nb_cores
        self.dist_method = dist_method
        self.n_clusters = n_clusters

        # Minimal default grid if none is provided:
        if param_grid is None:
            self.param_grid = {
                "rf__n_estimators": [5, 10],
                "xgb__max_depth": [2, 4],
                "mlp_base__hidden_layer_sizes": [(10,), (10, 5), (20, 10)],
                "final_estimator__alpha": [0.1, 0.9, 0.99]
            }
        else:
            self.param_grid = param_grid

    # ------------------------------------------------------------------
    # 1) HYPERPARAMETER TUNING VIA CLUSTERING + GRID SEARCH
    # ------------------------------------------------------------------
    def compute_hyperparameters(self, predictand, predictor):
        """
        Runs KMeans clustering on the mean of `predictand` (over time).
        Then, for each cluster, runs a cross-validation GridSearch over a stacking model with:
          - RF, XGB, MLP (as base estimators)
          - Ridge (as the meta-estimator).

        Parameters
        ----------
        predictand : xarray.DataArray
            Target variable with dims ('T','Y','X').
        predictor : xarray.DataArray
            Predictor variables with dims ('T','features').

        Returns
        -------
        best_param_da : xarray.DataArray
            DataArray storing best hyperparams (as string) per grid cell.
        cluster_da : xarray.DataArray
            Cluster assignment for each (Y,X).
        """
        # --- (a) Clustering: mimic WAS_Ridge ---
        # Convert predictand to DataFrame, drop the time column, and remove duplicates over (Y, X)
        df = (
            predictand.to_dataframe()
                      .reset_index()
                      .dropna()
                      .drop(columns=['T'])
        )
        # Use the first data column (e.g., "mean_val") as representative for clustering
        col_name = df.columns[2]
        kmeans = KMeans(n_clusters=self.n_clusters, random_state=42)
        df["cluster"] = kmeans.fit_predict(df[[col_name]])
        # Drop duplicates so that each grid cell appears only once
        df_unique = df.drop_duplicates(subset=["Y", "X"])
        dataset = df_unique.set_index(["Y", "X"]).to_xarray()
    
        # Create a cluster DataArray and mask out invalid cells using the first time slice
        cluster_da = (dataset["cluster"] *
                      xr.where(~np.isnan(predictand.isel(T=0)), 1, np.nan)
                     ).drop_vars("T", errors="ignore")
        # Align cluster_da with the original predictand
        _, cluster_da = xr.align(predictand, cluster_da, join="outer")
    
        # --- (b) Build the stacking model ---
        rf_model   = RandomForestRegressor(n_jobs=-1, random_state=42)
        xgb_model  = xgb.XGBRegressor(n_jobs=-1, random_state=42)
        mlp_base   = MLPRegressor(random_state=42,max_iter=5000)
        ridge_meta = Ridge(alpha=0.9)
        stacking_ridge = StackingRegressor(
            estimators=[("rf", rf_model), ("xgb", xgb_model), ("mlp_base", mlp_base)],
            final_estimator=ridge_meta,
            n_jobs=-1
        )
    
        # --- (c) Set up GridSearchCV ---
        grid_search = GridSearchCV(
            estimator=stacking_ridge,
            param_grid=self.param_grid,
            cv=5,  # or TimeSeriesSplit if appropriate
            scoring='neg_mean_squared_error',
            n_jobs=-1
        )
    
        unique_clusters = np.unique(cluster_da)
        unique_clusters = unique_clusters[np.isfinite(unique_clusters)]
        best_params_for_cluster = {}
    
        # --- (d) For each cluster, compute the cluster-mean time series and run grid search ---
        for c in unique_clusters:
            mask_c = (cluster_da == c)
            # Aggregate predictand over Y and X (for cells in cluster c) to get a time series
            y_cluster = (
                predictand.where(mask_c)
                          .mean(dim=["Y", "X"], skipna=True)
                          .dropna(dim="T")
            )
            if len(y_cluster["T"]) == 0:
                continue
            # Get predictor data for the matching time stamps
            predictor_cluster = predictor.sel(T=y_cluster["T"])
            X_mat = predictor_cluster.values  # shape: (time, features)
            y_vec = y_cluster.values          # shape: (time,)
    
            grid_search.fit(X_mat, y_vec)
            best_params_for_cluster[int(c)] = grid_search.best_params_
    
        # --- (e) Broadcast best hyperparameters to every grid cell ---
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

        # Base learners
        rf_model   = RandomForestRegressor(n_jobs=-1, random_state=42)
        xgb_model  = xgb.XGBRegressor(n_jobs=-1, random_state=42)
        mlp_base   = MLPRegressor(random_state=42,max_iter=5000)
        ridge_meta = Ridge(alpha=0.9)

        stacking_ridge = StackingRegressor(
            estimators=[("rf", rf_model), ("xgb", xgb_model), ("mlp_base", mlp_base)],
            final_estimator=ridge_meta,
            n_jobs=-1
        )

        # Apply local best params
        stacking_ridge.set_params(**best_params)

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
                    norm.ppf(0.3, loc=loc, scale=scale),
                    norm.ppf(0.7, loc=loc, scale=scale),
                )
            elif code == 2:
                return (
                    lognorm.ppf(0.3, s=shape, loc=loc, scale=scale),
                    lognorm.ppf(0.7, s=shape, loc=loc, scale=scale),
                )
            elif code == 3:
                return (
                    expon.ppf(0.3, loc=loc, scale=scale),
                    expon.ppf(0.7, loc=loc, scale=scale),
                )
            elif code == 4:
                return (
                    gamma.ppf(0.3, a=shape, loc=loc, scale=scale),
                    gamma.ppf(0.7, a=shape, loc=loc, scale=scale),
                )
            elif code == 5:
                return (
                    weibull_min.ppf(0.3, c=shape, loc=loc, scale=scale),
                    weibull_min.ppf(0.7, c=shape, loc=loc, scale=scale),
                )
            elif code == 6:
                # Note: Renamed 't_dist' to 't' for standard scipy.stats
                return (
                    t.ppf(0.3, df=shape, loc=loc, scale=scale),
                    t.ppf(0.7, df=shape, loc=loc, scale=scale),
                )
            elif code == 7:
                # Poisson: poisson.ppf(q, mu, loc=0)
                # ASSUMPTION: 'mu' (mean) is passed as 'shape'
                #             'loc' is passed as 'loc'
                #             'scale' is unused
                return (
                    poisson.ppf(0.3, mu=shape, loc=loc),
                    poisson.ppf(0.7, mu=shape, loc=loc),
                )
            elif code == 8:
                # Negative Binomial: nbinom.ppf(q, n, p, loc=0)
                # ASSUMPTION: 'n' (successes) is passed as 'shape'
                #             'p' (probability) is passed as 'scale'
                #             'loc' is passed as 'loc'
                return (
                    nbinom.ppf(0.3, n=shape, p=scale, loc=loc),
                    nbinom.ppf(0.7, n=shape, p=scale, loc=loc),
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
        terciles_emp = clim.quantile([0.3, 0.7], dim="T")
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
        Predictant_st = standardize_timeseries(Predictant, clim_year_start, clim_year_end)
        
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
        result_ = reverse_standardize(result_, Predictant, clim_year_start, clim_year_end)
        
        # result_ => dims (output=2, Y, X). 
        # For a real future forecast, "error" is NaN, "prediction" is the forecast.

        # 2) Compute thresholds T1, T2 from climatology
        index_start = Predictant.get_index("T").get_loc(str(clim_year_start)).start
        index_end   = Predictant.get_index("T").get_loc(str(clim_year_end)).stop
        rainfall_for_tercile = Predictant.isel(T=slice(index_start, index_end))
        terciles = rainfall_for_tercile.quantile([0.3, 0.7], dim='T')
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
    A logistic regression-based approach to classifying climate data into terciles and 
    then predicting the class probabilities for new data. 
    """

    def __init__(self, nb_cores=1):
        """
        Parameters
        ----------
        nb_cores : int, optional
            Number of CPU cores to use for Dask parallelization (default = 1).

        """
        # Store the number of cores and a distribution method attribute (the latter might be used in future expansions)
        self.nb_cores = nb_cores


    @staticmethod
    def classify(y, index_start, index_end):
        """
        Classifies the values of a 1D array `y` into terciles. 
        We only use a slice of y for the training/climatology period to define the 33rd and 67th percentiles.

        Parameters
        ----------
        y : array-like, shape (n_samples,)
            The time series of values we want to classify (e.g., rainfall).
        index_start, index_end : int
            The start and end indices defining the climatology/training window.

        Returns
        -------
        y_class : array, shape (n_samples,)
            The tercile class of each value in `y`, coded as 0 (below), 1 (middle), or 2 (above).
        tercile_33 : float
            The 33rd percentile threshold used to split the data.
        tercile_67 : float
            The 67th percentile threshold used to split the data.
        """
        # Create a mask of non-NaN entries
        mask = np.isfinite(y)
        # Check if there's any valid data
        if np.any(mask):
            # Compute the 33% and 67% thresholds from the specified slice
            terciles = np.nanpercentile(y[index_start:index_end], [33, 67])
            # Digitize assigns each y-value to a bin: 
            # bin 0: below tercile_33, bin 1: [tercile_33, tercile_67), bin 2: >= tercile_67
            y_class = np.digitize(y, bins=terciles, right=True)
            return y_class, terciles[0], terciles[1]
        else:
            # If data is invalid, return arrays filled with NaN
            return np.full(y.shape[0], np.nan), np.nan, np.nan

    def fit_predict(self, x, y, x_test):
        """
        Trains a logistic regression model on (x, y) and predicts class probabilities for x_test.

        Parameters
        ----------
        x : array-like, shape (n_samples, n_features)
            Predictor data for training.
        y : array-like, shape (n_samples,)
            Class labels (0, 1, 2) for training.
        x_test : array-like, shape (n_features,)
            Predictor data for the forecast/unknown scenario.

        Returns
        -------
        preds_proba : np.ndarray, shape (3,)
            Probability of each of the 3 tercile classes. 
            If fewer than 3 classes were present in training, the array is padded with NaNs.
        """
        # Initialize a logistic regression model. 'lbfgs' is a popular solver.
        model = linear_model.LogisticRegression(solver='lbfgs')

        # Identify rows with valid data
        mask = np.isfinite(y) & np.all(np.isfinite(x), axis=-1)
        if np.any(mask):
            # Subset to valid entries
            y_clean = y[mask]
            x_clean = x[mask, :]

            # Fit logistic regression
            model.fit(x_clean, y_clean)
            
            # Reshape x_test if it is 1D
            if x_test.ndim == 1:
                x_test = x_test.reshape(1, -1)

            # Predict probabilities for each class
            preds_proba = model.predict_proba(x_test).squeeze()  # shape (n_classes,)

            # If the model trained on fewer than 3 classes, we pad probabilities
            if preds_proba.shape[0] < 3:
                preds_proba_padded = np.full(3, np.nan)
                preds_proba_padded[:preds_proba.shape[0]] = preds_proba
                preds_proba = preds_proba_padded
            
            return preds_proba
        else:
            # If no valid data to fit, return NaNs
            return np.full((3,), np.nan)

    def compute_class(self, Predictant, clim_year_start, clim_year_end):
        """
        Assigns tercile classes for each point in the `Predictant` array.

        Parameters
        ----------
        Predictant : xarray.DataArray
            The observed variable (e.g., rainfall) with dimensions (T, Y, X).
        clim_year_start : int
            First year of the climatology period.
        clim_year_end : int
            Last year of the climatology period.

        Returns
        -------
        Predictant_class : xarray.DataArray
            The tercile class for each grid cell and time, labeled 0, 1, or 2.
        """
        # Identify the index range for the climatology period
        index_start = Predictant.get_index("T").get_loc(str(clim_year_start)).start
        index_end = Predictant.get_index("T").get_loc(str(clim_year_end)).stop
        
        # Use xr.apply_ufunc to apply `classify` along the time dimension ('T')
        Predictant_class, tercile_33, tercile_67 = xr.apply_ufunc(
            self.classify,
            Predictant,
            input_core_dims=[('T',)],
            kwargs={'index_start': index_start, 'index_end': index_end},
            vectorize=True,
            dask='parallelized',
            output_core_dims=[('T',), (), ()],
            output_dtypes=['float', 'float', 'float']
        )

        # Return the classified data, ensuring dimensions are consistent
        return Predictant_class.transpose('T', 'Y', 'X')
    
    def compute_model(self, X_train, y_train, X_test):
        """
        Computes logistic-regression-based class probabilities for each grid cell in `y_train`.

        Parameters
        ----------
        X_train : xarray.DataArray
            Predictors with dimensions (T, features).
        y_train : xarray.DataArray
            Tercile class labels with dimensions (T, Y, X).
        X_test : xarray.DataArray
            Test predictors with dimensions (T, features).

        Returns
        -------
        xarray.DataArray
            Class probabilities (3) for each grid cell.
        """
        # Determine chunk sizes based on user-defined number of cores
        chunksize_x = np.round(len(y_train.get_index("X")) / self.nb_cores)
        chunksize_y = np.round(len(y_train.get_index("Y")) / self.nb_cores)
        
        # Align time dimension
        X_train['T'] = y_train['T']
        y_train = y_train.transpose('T', 'Y', 'X')
        
        # Squeeze unnecessary dimensions from X_test for proper shape
        X_test = X_test.transpose('T', 'features').squeeze()

        # Create a Dask client for parallel processing
        client = Client(n_workers=self.nb_cores, threads_per_worker=1)

        # Apply the logistic model in parallel across spatial dimensions
        result = xr.apply_ufunc(
            self.fit_predict,
            X_train,
            y_train.chunk({'Y': chunksize_y, 'X': chunksize_x}),
            X_test,
            input_core_dims=[('T', 'features'), ('T',), ('features',)],
            output_core_dims=[('probability',)],  
            vectorize=True,
            dask='parallelized',
            output_dtypes=['float'],
            dask_gufunc_kwargs={'output_sizes': {'probability': 3}},  
        )
        
        # Compute the Dask result
        result_ = result.compute()
        client.close()
        return result_

    def forecast(self, Predictant, Predictor, Predictor_for_year):
        """
        Runs the trained logistic model on a single forecast year.

        Parameters
        ----------
        Predictant : xarray.DataArray
            The observed variable (T, Y, X), used for classification (training).
        Predictor : xarray.DataArray
            The training predictors (T, features).
        Predictor_for_year : xarray.DataArray
            Predictors for the forecast period or year, shape (features,).

        Returns
        -------
        xarray.DataArray
            Probability of each tercile class (PB, PN, PA) for every grid cell, 
            after removing the time dimension (because it's just one forecast).
        """
        # Define chunk sizes for parallelization
        chunksize_x = np.round(len(Predictant.get_index("X")) / self.nb_cores)
        chunksize_y = np.round(len(Predictant.get_index("Y")) / self.nb_cores)
        
        # Align 'T' dimension so it matches
        Predictor['T'] = Predictant['T']
        Predictant = Predictant.transpose('T', 'Y', 'X')
        Predictor_for_year_ = Predictor_for_year.squeeze()
        
        # Parallel approach with Dask
        client = Client(n_workers=self.nb_cores, threads_per_worker=1)
        result = xr.apply_ufunc(
            self.fit_predict,
            Predictor,
            Predictant.chunk({'Y': chunksize_y, 'X': chunksize_x}),
            Predictor_for_year_,
            input_core_dims=[('T', 'features'), ('T',), ('features',)],
            output_core_dims=[('probability',)],
            vectorize=True,
            dask='parallelized',
            output_dtypes=['float'],
            dask_gufunc_kwargs={'output_sizes': {'probability': 3}},
        )

        # Compute final result, close client
        result_ = result.compute()
        client.close()
        
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
        
        # Label the probability dimension with PB, PN, PA
        result_ = result_.assign_coords(probability=('probability', ['PB', 'PN', 'PA']))

        # Drop the time dimension (we're forecasting a single instance) and reorder dimensions
        return result_.drop_vars('T').squeeze().transpose('probability', 'T', 'Y', 'X')

############################################ Implement MARS ############################################
###########################################################################################################################################################################################################################################################################################################################################

def evaluate_basis(X, basis):
    """Evaluate a basis function on the input data X."""
    if not basis:  # Constant term
        return np.ones(X.shape[0])
    result = np.ones(X.shape[0])
    for v, t, s in basis:
        if s == 1:
            h = np.maximum(0, X[:, v] - t)
        else:
            h = np.maximum(0, t - X[:, v])
        result *= h
    return result

class MARS:
    """Multivariate Adaptive Regression Splines with Generalized Cross-Validation."""
    def __init__(self, max_terms=21, max_degree=1, c=3):
        """
        Initialize MARS model.

        Parameters:
        - max_terms: Maximum number of basis functions (default: 21)
        - max_degree: Maximum degree of interaction (default: 1)
        - c: Cost parameter for effective parameters in GCV (default: 3)
        """
        self.max_terms = max_terms
        self.max_degree = max_degree
        self.c = c
        self.basis_functions = None
        self.beta = None
        self.t_candidates = None

    def calculate_gcv(self, X, y, basis_functions, beta):
        """Calculate the Generalized Cross-Validation score."""
        n = X.shape[0]
        M = len(basis_functions)
        design = np.column_stack([evaluate_basis(X, b) for b in basis_functions])
        y_pred = design @ beta
        rss = np.sum((y - y_pred)**2)
        # Effective parameters: M + c * (M - 1)/2, but 1 if only constant term
        effective_params = M + self.c * (M - 1) / 2 if M > 1 else 1
        gcv = rss / (n * (1 - effective_params / n)**2)
        return gcv

    def fit(self, X, y):
        """Fit the MARS model to the data."""
        n_samples, n_features = X.shape
        self.t_candidates = [np.sort(np.unique(X[:, v]))[1:-1] for v in range(n_features)] if n_samples > 2 else []

        # Forward Pass: Build the initial model
        self.basis_functions = [[]]  # Start with constant term
        current_design = np.ones((n_samples, 1))
        self.beta = np.array([np.mean(y)])
        current_sse = np.sum((y - self.beta[0])**2)

        while len(self.basis_functions) < self.max_terms:
            best_sse = current_sse
            best_parent_idx = None
            best_v = None
            best_t = None
            best_beta_new = None
            best_col_left = None
            best_col_right = None

            current_cols = current_design.shape[1]

            for i, existing_basis in enumerate(self.basis_functions):
                current_degree = len(existing_basis)
                if current_degree >= self.max_degree:
                    continue
                used_vars = {vv for vv, _, _ in existing_basis}
                parent_col = current_design[:, i]

                for v in range(n_features):
                    if v in used_vars:
                        continue
                    for t in self.t_candidates[v]:
                        h_left = np.maximum(0, X[:, v] - t)
                        h_right = np.maximum(0, t - X[:, v])
                        col_left = parent_col * h_left
                        col_right = parent_col * h_right
                        new_design = np.hstack((current_design, col_left[:, np.newaxis], col_right[:, np.newaxis]))
                        beta_new, _, rank, _ = np.linalg.lstsq(new_design, y, rcond=None)
                        if rank <= current_cols:
                            continue
                        sse_new = np.sum((y - new_design @ beta_new)**2)
                        if sse_new < best_sse:
                            best_sse = sse_new
                            best_parent_idx = i
                            best_v = v
                            best_t = t
                            best_beta_new = beta_new
                            best_col_left = col_left
                            best_col_right = col_right

            if best_parent_idx is not None and len(self.basis_functions) + 2 <= self.max_terms:
                existing_basis = self.basis_functions[best_parent_idx]
                new_left = existing_basis + [(best_v, best_t, 1)]
                new_right = existing_basis + [(best_v, best_t, -1)]
                self.basis_functions.append(new_left)
                self.basis_functions.append(new_right)
                current_design = np.hstack((current_design, best_col_left[:, np.newaxis], best_col_right[:, np.newaxis]))
                self.beta = best_beta_new
                current_sse = best_sse
            else:
                break

        # Backward Pass with GCV: Prune the model
        best_gcv = self.calculate_gcv(X, y, self.basis_functions, self.beta)
        best_model = (self.basis_functions.copy(), self.beta.copy(), best_gcv)

        while len(self.basis_functions) > 1:  # Keep at least the constant term
            gcv_scores = []
            for m in range(1, len(self.basis_functions)):  # Skip constant term
                pruned_basis = self.basis_functions[:m] + self.basis_functions[m+1:]
                design = np.column_stack([evaluate_basis(X, b) for b in pruned_basis])
                beta_pruned = np.linalg.lstsq(design, y, rcond=None)[0]
                gcv = self.calculate_gcv(X, y, pruned_basis, beta_pruned)
                gcv_scores.append((gcv, m, beta_pruned))

            if not gcv_scores:
                break

            min_gcv, m_to_remove, beta_pruned = min(gcv_scores, key=lambda x: x[0])

            if min_gcv < best_gcv:
                best_gcv = min_gcv
                self.basis_functions = self.basis_functions[:m_to_remove] + self.basis_functions[m_to_remove+1:]
                self.beta = beta_pruned
                best_model = (self.basis_functions.copy(), self.beta.copy(), best_gcv)
            else:
                break

        self.basis_functions, self.beta, _ = best_model

    def predict(self, X):
        """Predict using the fitted MARS model."""
        n_samples = X.shape[0]
        y_pred = np.zeros(n_samples)
        for m, basis in enumerate(self.basis_functions):
            y_pred += self.beta[m] * evaluate_basis(X, basis)
        return y_pred

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

    def __init__(self, nb_cores=1, dist_method="nonparam", max_terms=21, max_degree=1, c=3):
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
        model = MARS(max_terms=self.max_terms, max_degree=self.max_degree, c=self.c)
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
                    norm.ppf(0.3, loc=loc, scale=scale),
                    norm.ppf(0.7, loc=loc, scale=scale),
                )
            elif code == 2:
                return (
                    lognorm.ppf(0.3, s=shape, loc=loc, scale=scale),
                    lognorm.ppf(0.7, s=shape, loc=loc, scale=scale),
                )
            elif code == 3:
                return (
                    expon.ppf(0.3, loc=loc, scale=scale),
                    expon.ppf(0.7, loc=loc, scale=scale),
                )
            elif code == 4:
                return (
                    gamma.ppf(0.3, a=shape, loc=loc, scale=scale),
                    gamma.ppf(0.7, a=shape, loc=loc, scale=scale),
                )
            elif code == 5:
                return (
                    weibull_min.ppf(0.3, c=shape, loc=loc, scale=scale),
                    weibull_min.ppf(0.7, c=shape, loc=loc, scale=scale),
                )
            elif code == 6:
                # Note: Renamed 't_dist' to 't' for standard scipy.stats
                return (
                    t.ppf(0.3, df=shape, loc=loc, scale=scale),
                    t.ppf(0.7, df=shape, loc=loc, scale=scale),
                )
            elif code == 7:
                # Poisson: poisson.ppf(q, mu, loc=0)
                # ASSUMPTION: 'mu' (mean) is passed as 'shape'
                #             'loc' is passed as 'loc'
                #             'scale' is unused
                return (
                    poisson.ppf(0.3, mu=shape, loc=loc),
                    poisson.ppf(0.7, mu=shape, loc=loc),
                )
            elif code == 8:
                # Negative Binomial: nbinom.ppf(q, n, p, loc=0)
                # ASSUMPTION: 'n' (successes) is passed as 'shape'
                #             'p' (probability) is passed as 'scale'
                #             'loc' is passed as 'loc'
                return (
                    nbinom.ppf(0.3, n=shape, p=scale, loc=loc),
                    nbinom.ppf(0.7, n=shape, p=scale, loc=loc),
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
        terciles_emp = clim.quantile([0.3, 0.7], dim="T")
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
        Predictant = Predictant.transpose('T','Y','X')
        Predictor_for_year_ = Predictor_for_year.squeeze()

        # 1) Fit+predict in parallel => shape (output=2, Y, X)
        client = Client(n_workers=self.nb_cores, threads_per_worker=1)
        result_da = xr.apply_ufunc(
            self.fit_predict,
            Predictor,
            Predictant.chunk({'Y': chunksize_y, 'X': chunksize_x}),
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

        # 2) Compute thresholds T1, T2 from climatology
        index_start = Predictant.get_index("T").get_loc(str(clim_year_start)).start
        index_end   = Predictant.get_index("T").get_loc(str(clim_year_end)).stop
        rainfall_for_tercile = Predictant.isel(T=slice(index_start, index_end))
        terciles = rainfall_for_tercile.quantile([0.3, 0.7], dim='T')
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