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

    Methods
    -------
    fit_predict(...)
        Fits an SVR model to the provided training data, makes predictions on the test data, 
        and calculates the prediction error.

    compute_hyperparameters(...)
        Computes optimal SVR hyperparameters (C and epsilon) for each spatial cluster.

    compute_model(...)
        Computes predictions for spatiotemporal data using SVR with parallel processing.

    calculate_tercile_probabilities(...)
        Calculates the probabilities for three tercile categories (below-normal, normal, above-normal) 
        based on predictions and associated error variance.

    compute_prob(...)
        Computes tercile probabilities for hindcast rainfall predictions over specified climatological years.

    forecast(...)
        Generates forecasts and computes probabilities for a specific year.
    """

    def __init__(
        self, 
        nb_cores=1, 
        n_clusters=5, 
        kernel='linear',
        gamma=None,
        C_range=[0.1, 1, 10, 100], 
        epsilon_range=[0.01, 0.1, 0.5, 1], 
        degree_range=[2, 3, 4]
    ):
        """
        Initializes the WAS_SVR with specified hyperparameter ranges.

        Parameters
        ----------
        nb_cores : int, optional
            Number of CPU cores to use for parallel computation, by default 1.
        n_clusters : int, optional
            Number of clusters for KMeans, by default 5.
        kernel : str, optional
            Kernel type to be used in SVR, by default 'linear'.
        degree : int, optional
            Degree of the polynomial kernel function ('poly'). Ignored by all other kernels, by default 3.
        C_range : list, optional
            List of C values to consider during hyperparameter tuning.
        epsilon_range : list, optional
            List of epsilon values to consider during hyperparameter tuning.
        degree_range : list, optional
            List of degrees to consider for the 'poly' kernel during hyperparameter tuning.
        """
        self.nb_cores = nb_cores
        self.n_clusters = n_clusters
        self.kernel = kernel
        self.gamma = gamma
        self.C_range = C_range
        self.epsilon_range = epsilon_range
        self.degree_range = degree_range

    def fit_predict(self, x, y, x_test, y_test, epsilon, C, degree=None):
        """
        Fits an SVR model to the provided training data, makes predictions on the test data, 
        and calculates the prediction error.

        Parameters
        ----------
        x : array-like, shape (n_samples, n_features)
            Training data (predictors).
        y : array-like, shape (n_samples,)
            Training targets.
        x_test : array-like, shape (n_features,)
            Test data (predictors).
        y_test : float or None
            Test target value. If None, error is set to np.nan.
        epsilon : float
            Epsilon parameter for SVR.
        C : float
            Regularization parameter for SVR.
        kernel : str
            Kernel type for SVR.
        degree : int or None
            Degree for polynomial kernel. Ignored if kernel is not 'poly'.
        gamma : str or None
            Kernel coefficient for 'rbf'. Ignored for other kernels.

        Returns
        -------
        np.ndarray
            Array containing the prediction error and the predicted value.
        """

        # Handle possible data type conversions
        if isinstance(self.kernel, bytes):
            kernel = self.kernel.decode('utf-8')
        if isinstance(degree, bytes) and degree is not None and not np.isnan(degree):
            degree = int(degree)
        if isinstance(self.gamma, bytes) and self.gamma is not None:
            gamma = self.gamma.decode('utf-8')
            
        if degree is None or degree == 'nan' or (isinstance(degree, float) and np.isnan(degree)):
            degree = 1
        else:
            degree = int(float(degree))

        # Prepare model parameters
        model_params = {'kernel': self.kernel, 'C': C, 'epsilon': epsilon}
        if self.kernel == 'poly' and degree is not None:
            model_params['degree'] = int(degree)
        if self.kernel == 'rbf' and self.gamma[0] is not None:
            model_params['gamma'] = self.gamma[0]

        model = SVR(**model_params)
        mask = np.isfinite(y) & np.all(np.isfinite(x), axis=-1)

        if np.any(mask):
            y_clean = y[mask]
            x_clean = x[mask, :]

            model.fit(x_clean, y_clean)

            if x_test.ndim == 1:
                x_test = x_test.reshape(1, -1)

            preds = model.predict(x_test)

            preds[preds < 0] = 0
            if y_test is not None and not np.isnan(y_test):
                error_ = y_test - preds
            else:
                error_ = np.nan
            return np.array([error_, preds]).squeeze()
        else:
            return np.array([np.nan, np.nan]).squeeze()  # Return NaNs if no valid data

    def compute_hyperparameters(self, predictand, predictor):
        """
        Computes optimal SVR hyperparameters (C and epsilon) for each spatial cluster.
    
        Parameters
        ----------
        predictand : xarray.DataArray
            Target variable with dimensions ('T', 'Y', 'X').
        predictor : xarray.DataArray
            Predictor variables with dimensions ('T', 'features').
    
        Returns
        -------
        C_array : xarray.DataArray
            Array of optimal C values for each spatial grid point, matching predictand dimensions.
        epsilon_array : xarray.DataArray
            Array of optimal epsilon values for each spatial grid point, matching predictand dimensions.
        kernel_array : xarray.DataArray
            Kernel types for each spatial grid point.
        degree_array : xarray.DataArray
            Degree values for 'poly' kernel at each spatial grid point.
        gamma_array : xarray.DataArray
            Gamma values for 'rbf' kernel at each spatial grid point.
        Cluster : xarray.DataArray
            Cluster assignments for each spatial grid point, matching predictand dimensions.
        """
        
        # Step 1: Perform KMeans clustering (same as before)
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
        xarray1, xarray2 = xr.align(predictand, Cluster)
        clusters = np.unique(xarray2)
        clusters = clusters[~np.isnan(clusters)]
        cluster_means = {
            int(cluster): xarray1.where(xarray2 == cluster).mean(dim=['Y', 'X'], skipna=True)
            for cluster in clusters
        }
    
        # Step 2: Grid search for each cluster's mean predictand
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
                'gamma': self.gamma #['scale', 'auto']
            })
    
        model = SVR()
        grid_search = GridSearchCV(model, param_grid, cv=5, scoring='neg_mean_squared_error')
    
        hyperparams_cluster = {}
        for cluster_label in clusters:
            # Get the mean time series for this cluster
            cluster_mean = cluster_means[int(cluster_label)].dropna('T')
            predictor['T'] = cluster_mean['T']
            # Get common times between cluster_mean and predictor
            common_times = np.intersect1d(cluster_mean['T'].values, predictor['T'].values)
            
            if len(common_times) == 0:
                # No common times, skip this cluster
                continue
            # Select data for common times
            cluster_mean_common = cluster_mean.sel(T=common_times)
            predictor_common = predictor.sel(T=common_times)
            y_cluster = cluster_mean_common.values
            if y_cluster.size > 0:
                grid_search.fit(predictor_common, y_cluster)
                best_params = grid_search.best_params_
                hyperparams_cluster[int(cluster_label)] = {
                    'C': best_params['C'],
                    'epsilon': best_params['epsilon'],
                    'kernel': best_params['kernel'],
                    'degree': best_params.get('degree', None),  # None if not 'poly'
                    'gamma': best_params.get('gamma', None)    # None if not 'rbf'
                }
    
        # Step 3: Assign hyperparameters to the spatial grid as DataArrays
        # Create DataArrays for C, epsilon, kernel, degree, gamma
        C_array = xr.full_like(Cluster, np.nan, dtype=float)
        epsilon_array = xr.full_like(Cluster, np.nan, dtype=float)
        degree_array = xr.full_like(Cluster, np.nan, dtype=int)

    
        for cluster_label, params in hyperparams_cluster.items():
            mask = Cluster == cluster_label
            C_array = C_array.where(~mask, other=params['C'])
            epsilon_array = epsilon_array.where(~mask, other=params['epsilon'])
            degree_array = degree_array.where(~mask, other=params['degree'])
    
        # Align arrays
        C_array, epsilon_array, degree_array, Cluster, predictand = xr.align(
            C_array, epsilon_array, degree_array, Cluster, predictand, join="outer"
        )
        return C_array, epsilon_array, degree_array, Cluster

    

    def compute_model(self, X_train, y_train, X_test, y_test, epsilon, C, degree_array=None):
        """
        Computes predictions for spatiotemporal data using SVR with parallel processing.

        Parameters
        ----------
        X_train : xarray.DataArray
            Training data (predictors) with dimensions ('T', 'features').
        y_train : xarray.DataArray
            Training target values with dimensions ('T', 'Y', 'X').
        X_test : xarray.DataArray
            Test data (predictors), squeezed to remove singleton dimensions.
        y_test : xarray.DataArray
            Test target values with dimensions ('Y', 'X').
        epsilon : xarray.DataArray
            Epsilon values for each grid point.
        C : xarray.DataArray
            C values for each grid point.
        kernel_array : xarray.DataArray
            Kernel types for each grid point.
        degree_array : xarray.DataArray
            Degrees for the polynomial kernel at each grid point.
        gamma_array : xarray.DataArray
            Gamma values for the 'rbf' kernel at each grid point.

        Returns
        -------
        xarray.DataArray
            The computed model predictions and errors, with an output dimension ('output',).
        """
        chunksize_x = int(np.round(len(y_train.get_index("X")) / self.nb_cores))
        chunksize_y = int(np.round(len(y_train.get_index("Y")) / self.nb_cores))

        X_train['T'] = y_train['T']
        y_train = y_train.transpose('T', 'Y', 'X')
        X_test = X_test.squeeze()
        y_test = y_test.squeeze().transpose('Y', 'X')

        client = Client(n_workers=self.nb_cores, threads_per_worker=1)
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
            # kwargs={'kernel': self.kernel, "gamma":self.gamma},
            output_core_dims=[('output',)],
            dask='parallelized',
            output_dtypes=['float'],
            dask_gufunc_kwargs={'output_sizes': {'output': 2}},
        )
        result_ = result.compute()
        client.close()
        return result_

    @staticmethod
    def calculate_tercile_probabilities(best_guess, error_variance, first_tercile, second_tercile, dof):
        """
        Calculates the probability of each tercile category (below-normal, normal, above-normal) 
        based on the forecasted value, error variance, and specified terciles.

        Parameters
        ----------
        best_guess : array-like
            Forecasted value.
        error_variance : float
            Error variance associated with the forecasted value.
        first_tercile : float
            Value corresponding to the lower tercile threshold.
        second_tercile : float
            Value corresponding to the upper tercile threshold.
        dof : int
            Degrees of freedom for the t-distribution.

        Returns
        -------
        np.ndarray
            An array of shape (3, n_time) representing the probabilities for the three tercile categories.
        """
        n_time = len(best_guess)
        pred_prob = np.empty((3, n_time))

        if np.all(np.isnan(best_guess)):
            pred_prob[:] = np.nan
        else:
            error_std = np.sqrt(error_variance)
            first_t = (first_tercile - best_guess) / error_std
            second_t = (second_tercile - best_guess) / error_std

            pred_prob[0, :] = stats.t.cdf(first_t, df=dof)
            pred_prob[1, :] = stats.t.cdf(second_t, df=dof) - stats.t.cdf(first_t, df=dof)
            pred_prob[2, :] = 1 - stats.t.cdf(second_t, df=dof)

        return pred_prob

    def compute_prob(self, Predictant, clim_year_start, clim_year_end, Predictor, hindcast_det):
        """
        Computes tercile category probabilities for hindcasts over a climatological period.

        Parameters
        ----------
        Predictant : xarray.DataArray
            The target dataset, with dimensions ('T', 'Y', 'X').
        clim_year_start : int
            The starting year of the climatology period.
        clim_year_end : int
            The ending year of the climatology period.
        Predictor : xarray.DataArray
            The predictor dataset with dimensions ('T', 'features').
        hindcast_det : xarray.DataArray
            Hindcast deterministic results from the model.

        Returns
        -------
        xarray.DataArray
            Tercile probabilities for the predicted values, with probability, time, Y, and X dimensions.
        """
        index_start = int(Predictant.get_index("T").get_loc(str(clim_year_start)).start)
        index_end = int(Predictant.get_index("T").get_loc(str(clim_year_end)).stop)

        rainfall_for_tercile = Predictant.isel(T=slice(index_start, index_end))
        terciles = rainfall_for_tercile.quantile([0.333, 0.667], dim='T')
        error_variance = hindcast_det.sel(output=0).drop_vars("output").squeeze().var(dim='T')

        dof = len(Predictant.get_index("T")) - 1 - (len(Predictor.get_index("features")) + 1)

        hindcast_prob = xr.apply_ufunc(
            self.calculate_tercile_probabilities,
            hindcast_det.sel(output=1).drop_vars("output").squeeze(),
            error_variance,
            terciles.isel(quantile=0).drop_vars('quantile'),
            terciles.isel(quantile=1).drop_vars('quantile'),
            input_core_dims=[('T',), (), (), ()],
            vectorize=True,
            kwargs={'dof': dof},
            dask='parallelized',
            output_core_dims=[('probability', 'T')],
            output_dtypes=['float'],
            dask_gufunc_kwargs={'output_sizes': {'probability': 3}},
        )

        hindcast_prob = hindcast_prob.assign_coords(probability=('probability', ['PB', 'PN', 'PA']))
        return hindcast_prob.transpose('probability', 'T', 'Y', 'X')

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
        gamma_array
    ):
        """
        Generates forecasts and computes probabilities for a specific year.

        Parameters
        ----------
        Predictant : xarray.DataArray
            Target variable.
        clim_year_start : int
            Start year for climatology.
        clim_year_end : int
            End year for climatology.
        Predictor : xarray.DataArray
            Predictor variables.
        hindcast_det : xarray.DataArray
            Deterministic hindcasts.
        Predictor_for_year : xarray.DataArray
            Predictor variables for the forecast year.
        epsilon : xarray.DataArray
            Epsilon values for each grid point.
        C : xarray.DataArray
            C values for each grid point.
        kernel_array : xarray.DataArray
            Kernel types for each grid point.
        degree_array : xarray.DataArray
            Degrees for the polynomial kernel at each grid point.
        gamma_array : xarray.DataArray
            Gamma values for the 'rbf' kernel at each grid point.

        Returns
        -------
        tuple
            Tuple containing forecast results and probabilities.
        """
        chunksize_x = int(np.round(len(Predictant.get_index("X")) / self.nb_cores))
        chunksize_y = int(np.round(len(Predictant.get_index("Y")) / self.nb_cores))

        Predictor['T'] = Predictant['T']
        Predictant = Predictant.transpose('T', 'Y', 'X')
        Predictor_for_year_ = Predictor_for_year.squeeze()

        y_test = xr.full_like(epsilon, np.nan)  # Create y_test with np.nan matching the spatial dimensions

        client = Client(n_workers=self.nb_cores, threads_per_worker=1)
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
                ('T', 'features'),  # x
                ('T',),             # y
                ('features',),      # x_test
                (),                 # y_test
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

        index_start = int(Predictant.get_index("T").get_loc(str(clim_year_start)).start)
        index_end = int(Predictant.get_index("T").get_loc(str(clim_year_end)).stop)
        rainfall_for_tercile = Predictant.isel(T=slice(index_start, index_end))
        terciles = rainfall_for_tercile.quantile([0.333, 0.667], dim='T')
        error_variance = hindcast_det.sel(output=0).drop_vars("output").squeeze().var(dim='T')
        dof = len(Predictant.get_index("T")) - 1 - (len(Predictor.get_index("features")) + 1)
        terciles, result_ = xr.align(terciles, result_)
        error_variance, terciles = xr.align(error_variance, terciles)

        hindcast_prob = xr.apply_ufunc(
            self.calculate_tercile_probabilities,
            result_.sel(output=1).drop_vars('output').expand_dims({'T':[pd.Timestamp(Predictor_for_year.coords['T'].values).to_pydatetime()]}),
            error_variance,
            terciles.isel(quantile=0).drop_vars('quantile'),
            terciles.isel(quantile=1).drop_vars('quantile'),
            input_core_dims=[('T',), (), (), ()],
            vectorize=True,
            kwargs={'dof': dof},
            dask='parallelized',
            output_core_dims=[('probability', 'T')],
            output_dtypes=['float'],
            dask_gufunc_kwargs={'output_sizes': {'probability': 3}},
        )
        hindcast_prob = hindcast_prob.assign_coords(probability=('probability', ['PB', 'PN', 'PA']))        
        return result_, hindcast_prob.drop_vars('T').squeeze().transpose('probability', 'Y', 'X')  



# class WAS_SVR:
#     """
#     A class to perform Support Vector Regression (SVR) on spatiotemporal datasets for climate prediction.

#     This class is designed to work with Dask and Xarray for parallelized, high-performance 
#     regression computations across large datasets with spatial and temporal dimensions. The primary 
#     methods are for fitting the SVR model, making predictions, and calculating probabilistic predictions 
#     for climate terciles.

#     Attributes
#     ----------
#     nb_cores : int, optional
#         The number of CPU cores to use for parallel computation (default is 1).
    
#     Methods
#     -------
    
#     fit_predict(x, y, x_test, y_test)
#         Fits a Support Vector Regression (SVR) model to the training data, predicts on test data, 
#         and computes error.
    
#     compute_model(X_train, y_train, X_test, y_test)
#         Applies the SVR model across a dataset using parallel computation with Dask, returning predictions and error metrics.
    
#     calculate_tercile_probabilities(best_guess, error_variance, first_tercile, second_tercile, dof)
#         Calculates the probabilities for three tercile categories (below-normal, normal, above-normal) 
#         based on predictions and associated error variance.
    
#     compute_prob(Predictant, clim_year_start, clim_year_end, Predictor, hindcast_det)
#         Computes tercile probabilities for hindcast rainfall predictions over specified climatological years.
#     """
    
#     def __init__(self, nb_cores=1, n_clusters=5):
#         """
#         Initializes the WAS_SVR with a specified number of CPU cores.
        
#         Parameters
#         ----------
#         nb_cores : int, optional
#             Number of CPU cores to use for parallel computation, by default 1.
#         """
#         self.nb_cores = nb_cores
#         self.n_clusters = n_clusters
    
#     def fit_predict(self, x, y, x_test, y_test, epsilon, C):
#         """
#         Fits an SVR model to the provided training data, makes predictions on the test data, 
#         and calculates the prediction error.
        
#         Parameters
#         ----------
#         x : array-like, shape (n_samples, n_features)
#             Training data (predictors).
#         y : array-like, shape (n_samples,)
#             Training targets.
#         x_test : array-like, shape (n_features,)
#             Test data (predictors).
#         y_test : float
#             Test target value.
        
#         Returns
#         -------
#         np.ndarray
#             Array containing the prediction error and the predicted value.
#         """
#         model = SVR(kernel='linear', C=C, epsilon=epsilon) #'rbf"
#         mask = np.isfinite(y) & np.all(np.isfinite(x), axis=-1)
        
#         if np.any(mask):
#             y_clean = y[mask]
#             x_clean = x[mask, :]
            
#             model.fit(x_clean, y_clean)

#             if x_test.ndim == 1:
#                 x_test = x_test.reshape(1, -1)
            
#             preds = model.predict(x_test)
            
#             preds[preds < 0] = 0
#             error_ = y_test - preds
#             return np.array([error_, preds]).squeeze()
#         else:
#             return np.array([np.nan, np.nan]).squeeze()  # Return NaNs if no valid data

#     def compute_hyperparameters(self, predictand, predictor):
#         """
#         Computes optimal SVR hyperparameters (C and epsilon) for each spatial cluster.
    
#         Parameters
#         ----------
#         predictand : xarray.DataArray
#             Target variable with dimensions ('T', 'Y', 'X').
#         predictor : array-like
#             Predictor variables with dimensions ('T', 'features').
    
#         Returns
#         -------
#         C_array : xarray.DataArray
#             Array of optimal C values for each spatial grid point, matching predictand dimensions.
#         epsilon_array : xarray.DataArray
#             Array of optimal epsilon values for each spatial grid point, matching predictand dimensions.
#         Cluster : xarray.DataArray
#             Cluster assignments for each spatial grid point, matching predictand dimensions.
#         """
    
#         # Step 1: Perform KMeans clustering
#         kmeans = KMeans(n_clusters=self.n_clusters, random_state=42)
#         predictand_dropna = predictand.to_dataframe().reset_index().dropna().drop(columns=['T'])
#         predictand_dropna['cluster'] = kmeans.fit_predict(
#             predictand_dropna[predictand_dropna.columns[2]].to_frame()
#         )
        
#         df_unique = predictand_dropna.drop_duplicates(subset=['Y', 'X'])
#         dataset = df_unique.set_index(['Y', 'X']).to_xarray()
        
#         Cluster = (dataset['cluster'] * xr.where(~np.isnan(predictand.isel(T=0)), 1, np.nan)).drop_vars("T")
#         xarray1, xarray2 = xr.align(predictand, Cluster)
#         clusters = np.unique(xarray2)
#         clusters = clusters[~np.isnan(clusters)]
#         cluster_means = {
#             int(cluster): xarray1.where(xarray2 == cluster).mean(dim=['Y', 'X'], skipna=True)
#             for cluster in clusters
#         }
        
#         # Step 2: Grid search for each cluster's mean predictand
#         param_grid = {
#             'C': [0.1, 1, 10, 100],
#             'epsilon': [0.01, 0.1, 0.5, 1]
#         }
        
#         model = SVR(kernel='rbf')
#         grid_search = GridSearchCV(model, param_grid, cv=5, scoring='neg_mean_squared_error')
    
#         hyperparams_cluster = {}
#         for cluster in clusters:
#             cluster_mean = cluster_means[cluster].to_series().dropna().values
#             if cluster_mean.size > 0:
#                 grid_search.fit(predictor, cluster_mean)
#                 hyperparams_cluster[int(cluster)] = {
#                     'C': grid_search.best_params_['C'],
#                     'epsilon': grid_search.best_params_['epsilon']
#                 }
        
#         # Step 3: Assign hyperparameters to the spatial grid as DataArrays
#         C_array = Cluster.copy()
#         epsilon_array = Cluster.copy()
        
#         for key, value in hyperparams_cluster.items():
#             C_array = C_array.where(C_array != key, other=value['C'])
#             epsilon_array = epsilon_array.where(epsilon_array != key, other=value['epsilon'])
#         C_array, epsilon_array, Cluster, predictand = xr.align(C_array, epsilon_array, Cluster, predictand, join = "outer")
#         return C_array, epsilon_array, Cluster
    
#     def compute_model(self, X_train, y_train, X_test, y_test, epsilon, C):
#         """
#         Computes predictions for spatiotemporal data using SVR with parallel processing.

#         Parameters
#         ----------
#         X_train : xarray.DataArray
#             Training data (predictors) with dimensions ('T', 'Y', 'X').
#         y_train : xarray.DataArray
#             Training target values with dimensions ('T', 'Y', 'X').
#         X_test : xarray.DataArray
#             Test data (predictors), squeezed to remove singleton dimensions.
#         y_test : xarray.DataArray
#             Test target values with dimensions ('Y', 'X').
        
#         Returns
#         -------
#         xarray.DataArray
#             The computed model predictions and errors, with an output dimension ('output',).
#         """
#         chunksize_x = np.round(len(y_train.get_index("X")) / self.nb_cores)
#         chunksize_y = np.round(len(y_train.get_index("Y")) / self.nb_cores)
        
#         X_train['T'] = y_train['T']
#         y_train = y_train.transpose('T', 'Y', 'X')
#         X_test = X_test.squeeze()
#         y_test = y_test.drop_vars('T').squeeze().transpose('Y', 'X')
        
#         client = Client(n_workers=self.nb_cores, threads_per_worker=1)
#         result = xr.apply_ufunc(
#             self.fit_predict,
#             X_train,
#             y_train.chunk({'Y': chunksize_y, 'X': chunksize_x}),
#             X_test,
#             y_test.chunk({'Y': chunksize_y, 'X': chunksize_x}),
#             epsilon.chunk({'Y': chunksize_y, 'X': chunksize_x}),
#             C.chunk({'Y': chunksize_y, 'X': chunksize_x}),
#             input_core_dims=[('T', 'features'), ('T',), ('features',), (),(),()],
#             vectorize=True,
#             output_core_dims=[('output',)],
#             dask='parallelized',
#             output_dtypes=['float'],
#             dask_gufunc_kwargs={'output_sizes': {'output': 2}},
#         )
#         result_ = result.compute()
#         client.close()
#         return result_
    
#     @staticmethod
#     def calculate_tercile_probabilities(best_guess, error_variance, first_tercile, second_tercile, dof):
#         """
#         Calculates the probability of each tercile category (below-normal, normal, above-normal) 
#         based on the forecasted value, error variance, and specified terciles.
        
#         Parameters
#         ----------
#         best_guess : array-like
#             Forecasted value.
#         error_variance : float
#             Error variance associated with the forecasted value.
#         first_tercile : float
#             Value corresponding to the lower tercile threshold.
#         second_tercile : float
#             Value corresponding to the upper tercile threshold.
#         dof : int
#             Degrees of freedom for the t-distribution.
        
#         Returns
#         -------
#         np.ndarray
#             An array of shape (3, n_time) representing the probabilities for the three tercile categories.
#         """
#         n_time = len(best_guess)
#         pred_prob = np.empty((3, n_time))
        
#         if np.all(np.isnan(best_guess)):
#             pred_prob[:] = np.nan
#         else:
#             error_std = np.sqrt(error_variance)
#             first_t = (first_tercile - best_guess) / error_std
#             second_t = (second_tercile - best_guess) / error_std
            
#             pred_prob[0, :] = stats.t.cdf(first_t, df=dof)
#             pred_prob[1, :] = stats.t.cdf(second_t, df=dof) - stats.t.cdf(first_t, df=dof)
#             pred_prob[2, :] = 1 - stats.t.cdf(second_t, df=dof)
        
#         return pred_prob
    
#     def compute_prob(self, Predictant, clim_year_start, clim_year_end, Predictor, hindcast_det):
#         """
#         Computes tercile category probabilities for hindcasts over a climatological period.

#         Parameters
#         ----------
#         Predictant : xarray.DataArray
#             The target dataset, with dimensions ('T', 'Y', 'X').
#         clim_year_start : int
#             The starting year of the climatology period.
#         clim_year_end : int
#             The ending year of the climatology period.
#         Predictor : xarray.DataArray
#             The predictor dataset with dimensions ('T', 'features').
#         hindcast_det : xarray.DataArray
#             Hindcast deterministic results from the model.

#         Returns
#         -------
#         xarray.DataArray
#             Tercile probabilities for the predicted values, with probability, time, Y, and X dimensions.
#         """
#         index_start = Predictant.get_index("T").get_loc(str(clim_year_start)).start
#         index_end = Predictant.get_index("T").get_loc(str(clim_year_end)).stop
        
#         rainfall_for_tercile = Predictant.isel(T=slice(index_start, index_end))
#         terciles = rainfall_for_tercile.quantile([0.333, 0.667], dim='T')
#         error_variance = hindcast_det.sel(output="error").drop_vars("output").squeeze().var(dim='T')
        
#         dof = len(Predictant.get_index("T")) - 1 - (len(Predictor.get_index("features")) + 1)
        
#         hindcast_prob = xr.apply_ufunc(
#             self.calculate_tercile_probabilities,
#             hindcast_det.sel(output="prediction").drop_vars("output").squeeze(),
#             error_variance,
#             terciles.isel(quantile=0).drop_vars('quantile'),
#             terciles.isel(quantile=1).drop_vars('quantile'),
#             input_core_dims=[('T',), (), (), ()],
#             vectorize=True,
#             kwargs={'dof': dof},
#             dask='parallelized',
#             output_core_dims=[('probability', 'T')],
#             output_dtypes=['float'],
#             dask_gufunc_kwargs={'output_sizes': {'probability': 3}},
#         )
        
#         hindcast_prob = hindcast_prob.assign_coords(probability=('probability', ['PB', 'PN', 'PA']))
#         return hindcast_prob.transpose('probability', 'T', 'Y', 'X')


#     def forecast(self, Predictant, clim_year_start, clim_year_end, Predictor, hindcast_det, Predictor_for_year, epsilon, C):

#         chunksize_x = np.round(len(Predictant.get_index("X")) / self.nb_cores)
#         chunksize_y = np.round(len(Predictant.get_index("Y")) / self.nb_cores)
        
#         Predictor['T'] = Predictant['T']
#         Predictant = Predictant.transpose('T', 'Y', 'X')
#         Predictor_for_year_ = Predictor_for_year.squeeze()

        
#         client = Client(n_workers=self.nb_cores, threads_per_worker=1)
#         result = xr.apply_ufunc(
#             self.fit_predict,
#             Predictor,
#             Predictant.chunk({'Y': chunksize_y, 'X': chunksize_x}),
#             Predictor_for_year_,
#             epsilon.chunk({'Y': chunksize_y, 'X': chunksize_x}),
#             C.chunk({'Y': chunksize_y, 'X': chunksize_x}),
#             input_core_dims=[('T', 'features'), ('T',), ('features',),(),()],
#             vectorize=True,
#             output_core_dims=[()],
#             dask='parallelized',
#             output_dtypes=['float'],
#         )
#         result_ = result.compute()
#         client.close()

#         index_start = Predictant.get_index("T").get_loc(str(clim_year_start)).start
#         index_end = Predictant.get_index("T").get_loc(str(clim_year_end)).stop
#         rainfall_for_tercile = Predictant.isel(T=slice(index_start, index_end))
#         terciles = rainfall_for_tercile.quantile([0.333, 0.667], dim='T')
#         error_variance = hindcast_det.sel(output="error").drop_vars("output").squeeze().var(dim='T')
#         dof = len(Predictant.get_index("T")) - 1 - (len(Predictor.get_index("features")) + 1)
#         terciles, result_ =  xr.align(terciles, result_)
#         error_variance, terciles =  xr.align(error_variance, terciles)
        
#         hindcast_prob = xr.apply_ufunc(
#             self.calculate_tercile_probabilities,
#             result_.expand_dims({'T':[pd.Timestamp(Predictor_for_year.coords['T'].values).to_pydatetime()]}),
#             error_variance,
#             terciles.isel(quantile=0).drop_vars('quantile'),
#             terciles.isel(quantile=1).drop_vars('quantile'),
#             input_core_dims=[('T',), (), (), ()],
#             vectorize=True,
#             kwargs={'dof': dof},
#             dask='parallelized',
#             output_core_dims=[('probability','T',)],
#             output_dtypes=['float'],
#             dask_gufunc_kwargs={'output_sizes': {'probability': 3}},
#         )
#         hindcast_prob = hindcast_prob.assign_coords(probability=('probability', ['PB', 'PN', 'PA']))        
#         return result_, hindcast_prob.drop_vars('T').squeeze().transpose('probability', 'Y', 'X') 

##############################################

class WAS_LogisticRegression_Model:
    def __init__(self, nb_cores=1):
        self.nb_cores = nb_cores

    @staticmethod
    def classify(y, index_start, index_end):
        mask = np.isfinite(y)
        if np.any(mask):
            terciles = np.nanpercentile(y[index_start:index_end], [33, 67])
            y_class = np.digitize(y, bins=terciles, right=True)
            return y_class, terciles[0], terciles[1]
        else:
            return np.full(y.shape[0], np.nan), np.nan, np.nan

    def fit_predict(self, x, y, x_test):
        model = linear_model.LogisticRegression(solver='lbfgs')
        mask = np.isfinite(y) & np.all(np.isfinite(x), axis=-1)
        if np.any(mask):
            y_clean = y[mask]
            x_clean = x[mask, :]
            model.fit(x_clean, y_clean)
            
            if x_test.ndim == 1:
                x_test = x_test.reshape(1, -1)
            preds_proba = model.predict_proba(x_test).squeeze()  # Shape (n_classes,)

            # Ensure the output is always 3 classes by padding if necessary
            if preds_proba.shape[0] < 3:
                preds_proba_padded = np.full(3, np.nan)
                preds_proba_padded[:preds_proba.shape[0]] = preds_proba
                preds_proba = preds_proba_padded
            
            return preds_proba
        else:
            return np.full((3,), np.nan)  # Return NaNs for predicted probabilities

    def compute_class(self, Predictant, clim_year_start, clim_year_end):
        index_start = Predictant.get_index("T").get_loc(str(clim_year_start)).start
        index_end = Predictant.get_index("T").get_loc(str(clim_year_end)).stop
        
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

        return Predictant_class.transpose('T', 'Y', 'X')

    def compute_model(self, X_train, y_train, X_test):
        chunksize_x = np.round(len(y_train.get_index("X")) / self.nb_cores)
        chunksize_y = np.round(len(y_train.get_index("Y")) / self.nb_cores)
        
        X_train['T'] = y_train['T']
        y_train = y_train.transpose('T', 'Y', 'X')
        X_test = X_test.transpose('T', 'features').squeeze()
        client = Client(n_workers=self.nb_cores, threads_per_worker=1)        
        result = xr.apply_ufunc(
            self.fit_predict,
            X_train,
            y_train.chunk({'Y': chunksize_y, 'X': chunksize_x}),
            X_test,
            input_core_dims=[('T', 'features'), ('T',), ('features',)],
            output_core_dims=[('probability',)],  # Ensure proper alignment
            vectorize=True,
            dask='parallelized',
            output_dtypes=['float'],
            dask_gufunc_kwargs={'output_sizes': {'probability': 3}},  # Match dimensions
        ) 
        result_ = result.compute()
        client.close()
        return result_

    def forecast(self, Predictant, Predictor, Predictor_for_year):

        chunksize_x = np.round(len(Predictant.get_index("X")) / self.nb_cores)
        chunksize_y = np.round(len(Predictant.get_index("Y")) / self.nb_cores)
        
        Predictor['T'] = Predictant['T']
        Predictant = Predictant.transpose('T', 'Y', 'X')
        Predictor_for_year_ = Predictor_for_year.squeeze()
        
        client = Client(n_workers=self.nb_cores, threads_per_worker=1)    
        result = xr.apply_ufunc(
            self.fit_predict,
            Predictor,
            Predictant.chunk({'Y': chunksize_y, 'X': chunksize_x}),
            Predictor_for_year_,
            input_core_dims=[('T', 'features'), ('T',), ('features',)],
            output_core_dims=[('probability',)],  # Ensure proper alignment
            vectorize=True,
            dask='parallelized',
            output_dtypes=['float'],
            dask_gufunc_kwargs={'output_sizes': {'probability': 3}},  # Match dimensions
        )
        result_ = result.compute()
        client.close()
        result_ = result_.assign_coords(probability=('probability', ['PB', 'PN', 'PA']))        
        return result_.drop_vars('T').squeeze().transpose('probability', 'Y', 'X') 

######################################################################

class WAS_QuantileRegression_Model:
    def __init__(self, nb_cores=1, quantiles=[0.33, 0.67], n_clusters=5, alpha_range=None):
        """
        Initialize the quantile regression model with clustering for hyperparameter tuning.
        
        :param nb_cores: Number of cores to use for parallel processing.
        :param quantiles: List of quantiles to predict, e.g., [0.1, 0.5, 0.9] for the 10th, 50th, and 90th percentiles.
        :param n_clusters: Number of clusters for KMeans clustering.
        :param alpha_range: Range of alpha values for hyperparameter tuning.
        """
        self.nb_cores = nb_cores
        self.quantiles = quantiles
        self.n_clusters = n_clusters
        self.alpha_range = alpha_range if alpha_range is not None else np.logspace(-4, 1, 10)

    @staticmethod
    def polynom_interp(y_pred, T1, T2, pred_quantile):
        n_time = len(y_pred)
        pred_prob = np.empty((3, n_time))
        if np.all(np.isnan(y_pred)):
            pred_prob[:] = np.nan
        else:
            print(y_pred)
            cs = CubicSpline(np.sort(y_pred), pred_quantile)
            prob1 = cs(T1)
            prob2 = cs(T2)
            pred_prob[0, :] = prob1
            pred_prob[1, :] = prob2 - prob1
            pred_prob[2, :] = 1 - prob2
        return pred_prob

    def fit_predict(self, x, y, x_test, alpha, pred_quantile):
        
        """
        Fit and predict using quantile regression for a specific quantile.
        
        :param x: Input features for training.
        :param y: Target variable for training.
        :param x_test: Input features for prediction.
        :param alpha: Regularization parameter for the model.
        :param quantile: Quantile to predict.
        :return: Predicted values for the specified quantile.
        """
        n_quantile = len(pred_quantile)
        predict_error = np.empty((2, n_quantile))
        
        for q, alph in zip(pred_quantile,alpha):                    #self.quantiles:
            model = linear_model.QuantileRegressor(quantile=q, alpha=alph, solver='highs')
            mask = np.isfinite(y) & np.all(np.isfinite(x), axis=-1)
            
            if np.any(mask):
                y_clean = y[mask]
                x_clean = x[mask, :]
                model.fit(x_clean, y_clean)
                
                if x_test.ndim == 1:
                    x_test = x_test.reshape(1, -1)
                preds = model.predict(x_test).squeeze()
                error = np.nanquantile(y_clean, q) - preds  
                predict_error[0, :] = error
                predict_error[1, :] = preds
            else:
                predict_error[:] = np.nan
        return predict_error

    def compute_hyperparameters(self, predictand, predictor):
        """
        Compute the optimal alpha values for each cluster and quantile using KMeans clustering.
        
        :param predictand: Target variable for clustering.
        :param predictor: Predictor variable for model fitting.
        :return: Dictionary of alpha arrays, one for each quantile, and the cluster array.
        """
        # model = linear_model.QuantileRegressor(quantile=q, alpha=alph, solver='highs')
        
        # Clustering on the predictand data
        kmeans = KMeans(n_clusters=self.n_clusters)
        predictand_dropna = predictand.to_dataframe().reset_index().dropna().drop(columns=['T'])
        predictand_dropna['cluster'] = kmeans.fit_predict(predictand_dropna[predictand_dropna.columns[2]].to_frame())
        
        # Convert the clustered data back to xarray format
        df_unique = predictand_dropna.drop_duplicates(subset=['Y', 'X'])
        dataset = df_unique.set_index(['Y', 'X']).to_xarray()
        
        Cluster = (dataset['cluster'] * xr.where(~np.isnan(predictand.isel(T=0)), 1, np.nan)).drop_vars("T")
        xarray1, xarray2 = xr.align(predictand, Cluster)
        
        # Loop through each cluster and each quantile to compute the best alpha value
        clusters = np.unique(xarray2)
        clusters = clusters[~np.isnan(clusters)]
        
        alpha_clusters = {quantile: Cluster.copy() for quantile in self.quantiles}
        alpha_all_qantiles = {}
        for quantile in self.quantiles:
            alpha_cluster = {}
            for cluster in clusters:
                cluster_mean = xarray1.where(xarray2 == cluster).mean(dim=['Y', 'X'], skipna=True)
                
                model = linear_model.QuantileRegressor(quantile=quantile, solver='highs')
                grid_search = GridSearchCV(model, param_grid={'alpha': self.alpha_range}, cv=5)
                grid_search.fit(predictor, cluster_mean)
                
                alpha_cluster[int(cluster)] = grid_search.best_params_['alpha']
            
            # Assign the best alpha values to the respective quantile 
            alpha_array = alpha_clusters[quantile]
            for key, value in alpha_cluster.items():
                alpha_array = alpha_array.where(alpha_array != key, other=value)
            alpha_all_qantiles[quantile] = alpha_array
        
        alpha_values = xr.concat(list(alpha_all_qantiles.values()), dim='quantile')
        alpha_values = alpha_values.assign_coords(quantile=('quantile', list(alpha_all_qantiles.keys())))
        alpha_values, Cluster, predictand = xr.align(alpha_values, Cluster, predictand, join = "outer")    
        return alpha_values, Cluster

    def compute_model(self, X_train, y_train, X_test, alpha):

        chunksize_x = np.round(len(y_train.get_index("X")) / self.nb_cores)
        chunksize_y = np.round(len(y_train.get_index("Y")) / self.nb_cores)
        
        X_train['T'] = y_train['T']
        y_train = y_train.transpose('T', 'Y', 'X')
        X_test = X_test.transpose('T', 'features').squeeze()
        client = Client(n_workers=self.nb_cores, threads_per_worker=1)
        
        result = xr.apply_ufunc(
            self.fit_predict,
            X_train,
            y_train.chunk({'Y': chunksize_y, 'X': chunksize_x}),
            X_test,
            alpha.chunk({'Y': chunksize_y, 'X': chunksize_x}),
            input_core_dims=[('T', 'features'), ('T',), ('features',), ('quantile',)],
            vectorize=True,
            kwargs={'pred_quantile': self.quantiles},
            output_core_dims=[('output','quantiles')],
            dask='parallelized',
            output_dtypes=['float'],
            dask_gufunc_kwargs={'output_sizes': {'output':2, 'quantiles': len(self.quantiles)}}
        )
        
        result_ = result.compute()
        client.close()
        result_ = result_.assign_coords(quantiles=('quantiles', self.quantiles), output=('output', ['error', 'prediction'])) 
        return result_.transpose('quantiles', 'output', 'Y', 'X')

        
    def compute_prob(self, Predictant, clim_year_start, clim_year_end, Predictor, hindcast_det):
        
        Predictor = None
        index_start = Predictant.get_index("T").get_loc(str(clim_year_start)).start
        index_end = Predictant.get_index("T").get_loc(str(clim_year_end)).stop
        Predictant, hindcast_det =  xr.align(Predictant, hindcast_det)
        rainfall_for_tercile = Predictant.isel(T=slice(index_start, index_end))
        terciles = rainfall_for_tercile.quantile([0.333, 0.667], dim='T')
        T1 = len(Predictant.get_index("T"))
        
        hindcast_prob = xr.apply_ufunc(
            self.polynom_interp,
            hindcast_det,
            terciles.isel(quantile=0).drop_vars('quantile'),
            terciles.isel(quantile=1).drop_vars('quantile'),
            input_core_dims=[('quantiles',), (), ()],
            vectorize=True,
            kwargs={'pred_quantile': self.quantiles},
            dask='parallelized',
            output_core_dims=[('probability', 'T1')],
            output_dtypes=['float'],
            dask_gufunc_kwargs={'output_sizes': {'probability': 3, 'T1': T1}},)
        hindcast_prob = hindcast_prob.rename({'T1': 'T'}).assign_coords(probability=('probability', ['PB', 'PN', 'PA']))
        return hindcast_prob.transpose('probability', 'T', 'Y', 'X')


#########################

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
    
    Methods
    -------
    
    fit_predict(x, y, x_test, y_test)
        Fits a Polynomial Regression model to the training data, predicts on test data, 
        and computes error.
    
    compute_model(X_train, y_train, X_test, y_test)
        Applies the Polynomial Regression model across a dataset using parallel computation 
        with Dask, returning predictions and error metrics.
    
    calculate_tercile_probabilities(best_guess, error_variance, first_tercile, second_tercile, dof)
        Calculates the probabilities for three tercile categories (below-normal, normal, above-normal) 
        based on predictions and associated error variance.
    
    compute_prob(Predictant, clim_year_start, clim_year_end, Predictor, hindcast_det)
        Computes tercile probabilities for hindcast rainfall predictions over specified climatological years.
    """
    
    def __init__(self, nb_cores=1, degree=2):
        """
        Initializes the WAS_PolynomialRegression with a specified number of CPU cores and polynomial degree.
        
        Parameters
        ----------
        nb_cores : int, optional
            Number of CPU cores to use for parallel computation, by default 1.
        degree : int, optional
            The degree of the polynomial, by default 2.
        """
        self.nb_cores = nb_cores
        self.degree = degree
    
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
        x_test : array-like, shape (n_features,)
            Test data (predictors).
        y_test : float
            Test target value.
        
        Returns
        -------
        np.ndarray
            Array containing the prediction error and the predicted value.
        """
        poly = PolynomialFeatures(degree=self.degree)      
        model = LinearRegression()
        
        mask = np.isfinite(y) & np.all(np.isfinite(x), axis=-1)
        if np.any(mask):
            y_clean = y[mask]
            x_clean = x[mask, :] 
            x_clean = poly.fit_transform(x_clean)
            model.fit(x_clean, y_clean)
            if x_test.ndim == 1:
                x_test = x_test.reshape(1, -1)
                x_test_poly = poly.transform(x_test)
            preds = model.predict(x_test_poly)
            preds[preds < 0] = 0
            error_ = y_test - preds
            return np.array([error_, preds]).squeeze()
        else:
            return np.array([np.nan, np.nan]).squeeze()        
    
    def compute_model(self, X_train, y_train, X_test, y_test):
        """
        Computes predictions for spatiotemporal data using Polynomial Regression with parallel processing.

        Parameters
        ----------
        X_train : xarray.DataArray
            Training data (predictors) with dimensions ('T', 'Y', 'X').
        y_train : xarray.DataArray
            Training target values with dimensions ('T', 'Y', 'X').
        X_test : xarray.DataArray
            Test data (predictors), squeezed to remove singleton dimensions.
        y_test : xarray.DataArray
            Test target values with dimensions ('Y', 'X').
        
        Returns
        -------
        xarray.DataArray
            The computed model predictions and errors, with an output dimension ('output',).
        """
        chunksize_x = np.round(len(y_train.get_index("X")) / self.nb_cores)
        chunksize_y = np.round(len(y_train.get_index("Y")) / self.nb_cores)
        
        X_train['T'] = y_train['T']
        y_train = y_train.transpose('T', 'Y', 'X')
        X_test = X_test.squeeze()
        y_test = y_test.drop_vars('T').squeeze().transpose('Y', 'X')
        client = Client(n_workers=self.nb_cores, threads_per_worker=1)
        
        result = xr.apply_ufunc(
            self.fit_predict,
            X_train,
            y_train.chunk({'Y': chunksize_y, 'X': chunksize_x}),
            X_test,
            y_test.chunk({'Y': chunksize_y, 'X': chunksize_x}),
            input_core_dims=[('T', 'features'), ('T',), ('features',), ()],
            vectorize=True,
            output_core_dims=[('output',)],
            dask='parallelized',
            output_dtypes=['float'],
            dask_gufunc_kwargs={'output_sizes': {'output': 2}},
        )
        
        result_ = result.compute()
        client.close()
        return result_
    
    @staticmethod
    def calculate_tercile_probabilities(best_guess, error_variance, first_tercile, second_tercile, dof):
        """
        Calculates the probability of each tercile category (below-normal, normal, above-normal) 
        based on the forecasted value, error variance, and specified terciles.
        
        Parameters
        ----------
        best_guess : array-like
            Forecasted value.
        error_variance : float
            Error variance associated with the forecasted value.
        first_tercile : float
            Value corresponding to the lower tercile threshold.
        second_tercile : float
            Value corresponding to the upper tercile threshold.
        dof : int
            Degrees of freedom for the t-distribution.
        
        Returns
        -------
        np.ndarray
            An array of shape (3, n_time) representing the probabilities for the three tercile categories.
        """
        n_time = len(best_guess)
        pred_prob = np.empty((3, n_time))
        
        if np.all(np.isnan(best_guess)):
            pred_prob[:] = np.nan
        else:
            error_std = np.sqrt(error_variance)
            first_t = (first_tercile - best_guess) / error_std
            second_t = (second_tercile - best_guess) / error_std
            
            pred_prob[0, :] = stats.t.cdf(first_t, df=dof)
            pred_prob[1, :] = stats.t.cdf(second_t, df=dof) - stats.t.cdf(first_t, df=dof)
            pred_prob[2, :] = 1 - stats.t.cdf(second_t, df=dof)
        
        return pred_prob
    
    def compute_prob(self, Predictant, clim_year_start, clim_year_end, Predictor, hindcast_det):
        """
        Computes tercile category probabilities for hindcasts over a climatological period.

        Parameters
        ----------
        Predictant : xarray.DataArray
            The target dataset, with dimensions ('T', 'Y', 'X').
        clim_year_start : int
            The starting year of the climatology period.
        clim_year_end : int
            The ending year of the climatology period.
        Predictor : xarray.DataArray
            The predictor dataset with dimensions ('T', 'features').
        hindcast_det : xarray.DataArray
            Hindcast deterministic results from the model.

        Returns
        -------
        xarray.DataArray
            Tercile probabilities for the predicted values, with probability, time, Y, and X dimensions.
        """
        index_start = Predictant.get_index("T").get_loc(str(clim_year_start)).start
        index_end = Predictant.get_index("T").get_loc(str(clim_year_end)).stop
        
        rainfall_for_tercile = Predictant.isel(T=slice(index_start, index_end))
        terciles = rainfall_for_tercile.quantile([0.333, 0.667], dim='T')
        error_variance = hindcast_det.sel(output="error").drop_vars("output").squeeze().var(dim='T')
        
        dof = len(Predictant.get_index("T")) - 1 - (len(Predictor.get_index("features")) + 1)
        
        hindcast_prob = xr.apply_ufunc(
            self.calculate_tercile_probabilities,
            hindcast_det.sel(output="prediction").drop_vars("output").squeeze(),
            error_variance,
            terciles.isel(quantile=0).drop_vars('quantile'),
            terciles.isel(quantile=1).drop_vars('quantile'),
            input_core_dims=[('T',), (), (), ()],
            vectorize=True,
            kwargs={'dof': dof},
            dask='parallelized',
            output_core_dims=[('probability', 'T')],
            output_dtypes=['float'],
            dask_gufunc_kwargs={'output_sizes': {'probability': 3}},
        )
        
        hindcast_prob = hindcast_prob.assign_coords(probability=('probability', ['PB', 'PN', 'PA']))
        return hindcast_prob.transpose('probability', 'T', 'Y', 'X')

    
    def forecast(self, Predictant, clim_year_start, clim_year_end, Predictor, hindcast_det, Predictor_for_year):

        chunksize_x = np.round(len(Predictant.get_index("X")) / self.nb_cores)
        chunksize_y = np.round(len(Predictant.get_index("Y")) / self.nb_cores)
        
        Predictor['T'] = Predictant['T']
        Predictant = Predictant.transpose('T', 'Y', 'X')
        Predictor_for_year_ = Predictor_for_year.squeeze()

        client = Client(n_workers=self.nb_cores, threads_per_worker=1)
        result = xr.apply_ufunc(
            self.fit_predict,
            Predictor,
            Predictant.chunk({'Y': chunksize_y, 'X': chunksize_x}),
            Predictor_for_year_,
            input_core_dims=[('T', 'features'), ('T',), ('features',)],
            vectorize=True,
            output_core_dims=[()],
            # output_core_dims=[('output',)],
            dask='parallelized',
            output_dtypes=['float'],
            # dask_gufunc_kwargs={'output_sizes': {'output': 1}},
        )
        result_ = result.compute()
        client.close() 

        index_start = Predictant.get_index("T").get_loc(str(clim_year_start)).start
        index_end = Predictant.get_index("T").get_loc(str(clim_year_end)).stop
        rainfall_for_tercile = Predictant.isel(T=slice(index_start, index_end))
        terciles = rainfall_for_tercile.quantile([0.333, 0.667], dim='T')
        error_variance = hindcast_det.sel(output="error").drop_vars("output").squeeze().var(dim='T')
        dof = len(Predictant.get_index("T")) - 1 - (len(Predictor.get_index("features")) + 1)
        
        hindcast_prob = xr.apply_ufunc(
            self.calculate_tercile_probabilities,
            result_.expand_dims({'T':[pd.Timestamp(Predictor_for_year.coords['T'].values).to_pydatetime()]}),
            error_variance,
            terciles.isel(quantile=0).drop_vars('quantile'),
            terciles.isel(quantile=1).drop_vars('quantile'),
            input_core_dims=[('T',), (), (), ()],
            vectorize=True,
            kwargs={'dof': dof},
            dask='parallelized',
            output_core_dims=[('probability','T',)],
            output_dtypes=['float'],
            dask_gufunc_kwargs={'output_sizes': {'probability': 3}},
        )
        hindcast_prob = hindcast_prob.assign_coords(probability=('probability', ['PB', 'PN', 'PA']))        
        return result_, hindcast_prob.drop_vars('T').squeeze().transpose('probability', 'Y', 'X')  
        
###########################################

# class WAS_PoissonRegression:
#     """
#     A class to perform Poisson Regression on spatiotemporal datasets for count data prediction.

#     This class is designed to work with Dask and Xarray for parallelized, high-performance 
#     regression computations across large datasets with spatial and temporal dimensions. The primary 
#     methods are for fitting the Poisson regression model, making predictions, and calculating probabilistic predictions 
#     for climate terciles.

#     Attributes
#     ----------
#     nb_cores : int, optional
#         The number of CPU cores to use for parallel computation (default is 1).
    
#     Methods
#     -------
    
#     fit_predict(x, y, x_test, y_test)
#         Fits a Poisson regression model to the training data, predicts on test data, and computes error.
    
#     compute_model(X_train, y_train, X_test, y_test)
#         Applies the Poisson regression model across a dataset using parallel computation 
#         with Dask, returning predictions and error metrics.
    
#     calculate_tercile_probabilities(best_guess, error_variance, first_tercile, second_tercile, dof)
#         Calculates the probabilities for three tercile categories (below-normal, normal, above-normal) 
#         based on predictions and associated error variance.
    
#     compute_prob(Predictant, clim_year_start, clim_year_end, Predictor, hindcast_det)
#         Computes tercile probabilities for hindcast rainfall predictions over specified climatological years.
#     """
    
#     def __init__(self, nb_cores=1):
#         """
#         Initializes the WAS_PoissonRegression with a specified number of CPU cores.
        
#         Parameters
#         ----------
#         nb_cores : int, optional
#             Number of CPU cores to use for parallel computation, by default 1.
#         """
#         self.nb_cores = nb_cores
    
#     def fit_predict(self, x, y, x_test, y_test):
#         """
#         Fits a Poisson regression model to the provided training data, makes predictions 
#         on the test data, and calculates the prediction error.
        
#         Parameters
#         ----------
#         x : array-like, shape (n_samples, n_features)
#             Training data (predictors).
#         y : array-like, shape (n_samples,)
#             Training targets (count data).
#         x_test : array-like, shape (n_features,)
#             Test data (predictors).
#         y_test : float
#             Test target value (actual counts).
        
#         Returns
#         -------
#         np.ndarray
#             Array containing the prediction error and the predicted value.
#         """
#         model = linear_model.PoissonRegressor()  # Initialize the Poisson regression model
        
#         # Fit the Poisson regression model on the training data
#         model.fit(x, y)
        
#         # Predict on the test data
#         preds = model.predict(x_test)
#         preds[preds < 0] = 0
#         error_ = y_test - preds
#         return np.array([error_, preds]).squeeze()
    
#     def compute_model(self, X_train, y_train, X_test, y_test):
#         """
#         Computes predictions for spatiotemporal data using Poisson Regression with parallel processing.

#         Parameters
#         ----------
#         X_train : xarray.DataArray
#             Training data (predictors) with dimensions ('T', 'Y', 'X').
#         y_train : xarray.DataArray
#             Training target values (count data) with dimensions ('T', 'Y', 'X').
#         X_test : xarray.DataArray
#             Test data (predictors), squeezed to remove singleton dimensions.
#         y_test : xarray.DataArray
#             Test target values (count data) with dimensions ('Y', 'X').
        
#         Returns
#         -------
#         xarray.DataArray
#             The computed model predictions and errors, with an output dimension ('output',).
#         """
#         chunksize_x = np.round(len(y_train.get_index("X")) / self.nb_cores)
#         chunksize_y = np.round(len(y_train.get_index("Y")) / self.nb_cores)
        
#         X_train['T'] = y_train['T']
#         y_train = y_train.transpose('T', 'Y', 'X')
#         X_test = X_test.squeeze()
#         y_test = y_test.drop_vars('T').squeeze().transpose('Y', 'X')
#         client = Client(n_workers=self.nb_cores, threads_per_worker=1)
        
#         result = xr.apply_ufunc(
#             self.fit_predict,
#             X_train,
#             y_train.chunk({'Y': chunksize_y, 'X': chunksize_x}),
#             X_test,
#             y_test.chunk({'Y': chunksize_y, 'X': chunksize_x}),
#             input_core_dims=[('T', 'features'), ('T',), ('features',), ()],
#             vectorize=True,
#             output_core_dims=[('output',)],
#             dask='parallelized',
#             output_dtypes=['float'],
#             dask_gufunc_kwargs={'output_sizes': {'output': 2}},
#         )
        
#         result_ = result.compute()
#         client.close()
#         return result_
    
#     @staticmethod
#     def calculate_tercile_probabilities(best_guess, error_variance, first_tercile, second_tercile, dof):
#         """
#         Calculates the probability of each tercile category (below-normal, normal, above-normal) 
#         based on the forecasted value, error variance, and specified terciles.
        
#         Parameters
#         ----------
#         best_guess : array-like
#             Forecasted value.
#         error_variance : float
#             Error variance associated with the forecasted value.
#         first_tercile : float
#             Value corresponding to the lower tercile threshold.
#         second_tercile : float
#             Value corresponding to the upper tercile threshold.
#         dof : int
#             Degrees of freedom for the t-distribution.
        
#         Returns
#         -------
#         np.ndarray
#             An array of shape (3, n_time) representing the probabilities for the three tercile categories.
#         """
#         n_time = len(best_guess)
#         pred_prob = np.empty((3, n_time))
        
#         if np.all(np.isnan(best_guess)):
#             pred_prob[:] = np.nan
#         else:
#             error_std = np.sqrt(error_variance)
#             first_t = (first_tercile - best_guess) / error_std
#             second_t = (second_tercile - best_guess) / error_std
            
#             pred_prob[0, :] = stats.t.cdf(first_t, df=dof)
#             pred_prob[1, :] = stats.t.cdf(second_t, df=dof) - stats.t.cdf(first_t, df=dof)
#             pred_prob[2, :] = 1 - stats.t.cdf(second_t, df=dof)
        
#         return pred_prob
    
#     def compute_prob(self, Predictant, clim_year_start, clim_year_end, Predictor, hindcast_det):
#         """
#         Computes tercile category probabilities for hindcasts over a climatological period.

#         Parameters
#         ----------
#         Predictant : xarray.DataArray
#             The target dataset, with dimensions ('T', 'Y', 'X').
#         clim_year_start : int
#             The starting year of the climatology period.
#         clim_year_end : int
#             The ending year of the climatology period.
#         Predictor : xarray.DataArray
#             The predictor dataset with dimensions ('T', 'features').
#         hindcast_det : xarray.DataArray
#             Hindcast deterministic results from the model.

#         Returns
#         -------
#         xarray.DataArray
#             Tercile probabilities for the predicted values, with probability, time, Y, and X dimensions.
#         """
#         index_start = Predictant.get_index("T").get_loc(str(clim_year_start)).start
#         index_end = Predictant.get_index("T").get_loc(str(clim_year_end)).stop
        
#         rainfall_for_tercile = Predictant.isel(T=slice(index_start, index_end))
#         terciles = rainfall_for_tercile.quantile([0.333, 0.667], dim='T')
#         error_variance = hindcast_det.sel(output="error").drop_vars("output").squeeze().var(dim='T')
        
#         dof = len(Predictant.get_index("T")) - 1 - (len(Predictor.get_index("features")) + 1)
        
#         hindcast_prob = xr.apply_ufunc(
#             self.calculate_tercile_probabilities,
#             hindcast_det.sel(output="prediction").drop_vars("output").squeeze(),
#             error_variance,
#             terciles.isel(quantile=0).drop_vars('quantile'),
#             terciles.isel(quantile=1).drop_vars('quantile'),
#             input_core_dims=[('T',), (), (), ()],
#             vectorize=True,
#             kwargs={'dof': dof},
#             dask='parallelized',
#             output_core_dims=[('probability', 'T')],
#             output_dtypes=['float'],
#             dask_gufunc_kwargs={'output_sizes': {'probability': 3}},
#         )
        
#         hindcast_prob = hindcast_prob.assign_coords(probability=('probability', ['PB', 'PN', 'PA']))
#         return hindcast_prob.transpose('probability', 'T', 'Y', 'X')


class WAS_RandomForest_XGBoost_Stacking:
    """
    A class to perform Stacking Ensemble with Random Forest and XGBoost models.
    The predictions of both models are used as features for a final meta-model 
    (Linear Regression in this case) to predict the target value.
    
    Attributes
    ----------
    nb_cores : int, optional
        The number of CPU cores to use for parallel computation (default is 1).
    rf_model : RandomForestRegressor
        Random Forest Regressor model.
    xgb_model : xgboost.XGBRegressor
        XGBoost Regressor model.
    meta_model : LinearRegression
        Meta-model (Linear Regression) for the stacking ensemble.
    stacking_model : StackingRegressor
        The Stacking Regressor that combines the base models (RF, XGBoost) with the meta-model.
    
    Methods
    -------
    fit(X_train, y_train)
        Fits both base models (RandomForest, XGBoost) and the meta-model on the training data.
        
    predict(X_test)
        Makes predictions by using the Stacking Regressor, combining base models' predictions.
    
    fit_predict(X_train, y_train, X_test, y_test)
        Fits both models and predicts on test data, returning error and final prediction.
    """
    
    def __init__(self, nb_cores=1, rf_params=None, xgb_params=None, stacking_params=None):
        self.nb_cores = nb_cores
    
        # Base models with custom parameters
        self.rf_model = RandomForestRegressor(n_jobs=self.nb_cores, **(rf_params or {}))
        self.xgb_model = xgb.XGBRegressor(n_jobs=self.nb_cores, **(xgb_params or {}))
    
        # Meta-model for stacking
        self.meta_model = LinearRegression()
    
        # Stacking model with custom parameters
        self.stacking_model = StackingRegressor(
            estimators=[('rf', self.rf_model), ('xgb', self.xgb_model)],
            final_estimator=self.meta_model,
            n_jobs=self.nb_cores,
            **(stacking_params or {})
        )

    
    
    def fit_predict(self, X_train, y_train, X_test, y_test):
        mask = np.isfinite(y_train) & np.all(np.isfinite(X_train), axis=-1)

        if np.any(mask):
            y_clean = y_train[mask]
            x_clean = X_train[mask, :]
            
            self.stacking_model.fit(x_clean, y_clean)

            if X_test.ndim == 1:
                X_test = X_test.reshape(1, -1)

            stacked_preds = self.stacking_model.predict(X_test)
            stacked_preds[stacked_preds < 0] = 0
            error_ = y_test - stacked_preds
            return np.array([error_, stacked_preds]).squeeze()
        else:
            return np.array([np.nan, np.nan]).squeeze()

    def compute_model(self, X_train, y_train, X_test, y_test):
        """
        Computes predictions for spatiotemporal data using linear regression with parallel processing.

        Parameters
        ----------
        X_train : xarray.DataArray
            Training data (predictors) with dimensions ('T', 'Y', 'X').
        y_train : xarray.DataArray
            Training target values with dimensions ('T', 'Y', 'X').
        X_test : xarray.DataArray
            Test data (predictors), squeezed to remove singleton dimensions.
        y_test : xarray.DataArray
            Test target values with dimensions ('Y', 'X').
        
        Returns
        -------
        xarray.DataArray
            The computed model predictions and errors, with an output dimension ('output',).
        """
        # chunksize_x = np.round(len(y_train.get_index("X")) / self.nb_cores)
        # chunksize_y = np.round(len(y_train.get_index("Y")) / self.nb_cores)
        
        X_train['T'] = y_train['T']
        y_train = y_train.transpose('T', 'Y', 'X')
        X_test = X_test.squeeze()
        
        y_test = y_test.drop_vars('T').squeeze().transpose('Y', 'X')
        # client = Client(n_workers=self.nb_cores, threads_per_worker=1)
        result = xr.apply_ufunc(
            self.fit_predict,
            X_train,
            y_train,#.chunk({'Y': chunksize_y, 'X': chunksize_x}),
            X_test,
            y_test,#.chunk({'Y': chunksize_y, 'X': chunksize_x}),
            input_core_dims=[('T', 'features'), ('T',), ('features',), ()],
            vectorize=True,
            output_core_dims=[('output',)],
            dask='parallelized',
            output_dtypes=['float'],
            dask_gufunc_kwargs={'output_sizes': {'output': 2}},
        )
        # result_ = result.compute()
        # client.close()           
        return result
    
    @staticmethod
    def calculate_tercile_probabilities(best_guess, error_variance, first_tercile, second_tercile, dof):
        """
        Calculates the probability of each tercile category (below-normal, normal, above-normal) 
        based on the forecasted value, error variance, and specified terciles.
        
        Parameters
        ----------
        best_guess : array-like
            Forecasted value.
        error_variance : float
            Error variance associated with the forecasted value.
        first_tercile : float
            Value corresponding to the lower tercile threshold.
        second_tercile : float
            Value corresponding to the upper tercile threshold.
        dof : int
            Degrees of freedom for the t-distribution.
        
        Returns
        -------
        np.ndarray
            An array of shape (3, n_time) representing the probabilities for the three tercile categories.
        """
        n_time = len(best_guess)
        pred_prob = np.empty((3, n_time))
        
        if np.all(np.isnan(best_guess)):
            pred_prob[:] = np.nan
        else:
            error_std = np.sqrt(error_variance)
            first_t = (first_tercile - best_guess) / error_std
            second_t = (second_tercile - best_guess) / error_std
            
            pred_prob[0, :] = stats.t.cdf(first_t, df=dof)
            pred_prob[1, :] = stats.t.cdf(second_t, df=dof) - stats.t.cdf(first_t, df=dof)
            pred_prob[2, :] = 1 - stats.t.cdf(second_t, df=dof)
        
        return pred_prob

    
    def compute_prob(self, Predictant, clim_year_start, clim_year_end, Predictor, hindcast_det):
        """
        Computes tercile category probabilities for hindcasts over a climatological period.

        Parameters
        ----------
        Predictant : xarray.DataArray
            The target dataset, with dimensions ('T', 'Y', 'X').
        clim_year_start : int
            The starting year of the climatology period.
        clim_year_end : int
            The ending year of the climatology period.
        Predictor : xarray.DataArray
            The predictor dataset with dimensions ('T', 'features').
        hindcast_det : xarray.DataArray
            Hindcast deterministic results from the model.

        Returns
        -------
        xarray.DataArray
            Tercile probabilities for the predicted values, with probability, time, Y, and X dimensions.
        """
        index_start = Predictant.get_index("T").get_loc(str(clim_year_start)).start
        index_end = Predictant.get_index("T").get_loc(str(clim_year_end)).stop
        
        rainfall_for_tercile = Predictant.isel(T=slice(index_start, index_end))
        terciles = rainfall_for_tercile.quantile([0.333, 0.667], dim='T')
        error_variance = hindcast_det.sel(output="error").drop_vars("output").squeeze().var(dim='T')
        
        dof = len(Predictant.get_index("T")) - 1 - (len(Predictor.get_index("features")) + 1)
        
        hindcast_prob = xr.apply_ufunc(
            self.calculate_tercile_probabilities,
            hindcast_det.sel(output="prediction").drop_vars("output").squeeze(),
            error_variance,
            terciles.isel(quantile=0).drop_vars('quantile'),
            terciles.isel(quantile=1).drop_vars('quantile'),
            input_core_dims=[('T',), (), (), ()],
            vectorize=True,
            kwargs={'dof': dof},
            dask='parallelized',
            output_core_dims=[('probability', 'T')],
            output_dtypes=['float'],
            dask_gufunc_kwargs={'output_sizes': {'probability': 3}},
        )
        
        hindcast_prob = hindcast_prob.assign_coords(probability=('probability', ['PB', 'PN', 'PA']))
        return hindcast_prob.transpose('probability', 'T', 'Y', 'X')

    def forecast(self, Predictant, clim_year_start, clim_year_end, Predictor, hindcast_det, Predictor_for_year):

        chunksize_x = np.round(len(Predictant.get_index("X")) / self.nb_cores)
        chunksize_y = np.round(len(Predictant.get_index("Y")) / self.nb_cores)
        
        Predictor['T'] = Predictant['T']
        Predictant = Predictant.transpose('T', 'Y', 'X')
        Predictor_for_year_ = Predictor_for_year.squeeze()

        client = Client(n_workers=self.nb_cores, threads_per_worker=1)
        result = xr.apply_ufunc(
            self.fit_predict,
            Predictor,
            Predictant.chunk({'Y': chunksize_y, 'X': chunksize_x}),
            Predictor_for_year_,
            input_core_dims=[('T', 'features'), ('T',), ('features',)],
            vectorize=True,
            output_core_dims=[()],
            # output_core_dims=[('output',)],
            dask='parallelized',
            output_dtypes=['float'],
            # dask_gufunc_kwargs={'output_sizes': {'output': 1}},
        )
        result_ = result.compute()
        client.close() 

        index_start = Predictant.get_index("T").get_loc(str(clim_year_start)).start
        index_end = Predictant.get_index("T").get_loc(str(clim_year_end)).stop
        rainfall_for_tercile = Predictant.isel(T=slice(index_start, index_end))
        terciles = rainfall_for_tercile.quantile([0.333, 0.667], dim='T')
        error_variance = hindcast_det.sel(output="error").drop_vars("output").squeeze().var(dim='T')
        dof = len(Predictant.get_index("T")) - 1 - (len(Predictor.get_index("features")) + 1)
        
        hindcast_prob = xr.apply_ufunc(
            self.calculate_tercile_probabilities,
            result_.expand_dims({'T':[pd.Timestamp(Predictor_for_year.coords['T'].values).to_pydatetime()]}),
            error_variance,
            terciles.isel(quantile=0).drop_vars('quantile'),
            terciles.isel(quantile=1).drop_vars('quantile'),
            input_core_dims=[('T',), (), (), ()],
            vectorize=True,
            kwargs={'dof': dof},
            dask='parallelized',
            output_core_dims=[('probability','T',)],
            output_dtypes=['float'],
            dask_gufunc_kwargs={'output_sizes': {'probability': 3}},
        )
        hindcast_prob = hindcast_prob.assign_coords(probability=('probability', ['PB', 'PN', 'PA']))        
        return result_, hindcast_prob.drop_vars('T').squeeze().transpose('probability', 'Y', 'X') 

class WAS_MLP:
    
    def __init__(self, nb_cores=1, hidden_layer_sizes=(10,5), activation='relu', max_iter=500, solver='adam', learning_rate_init=0.001):

        self.hidden_layer_sizes=hidden_layer_sizes
        self.activation=activation
        self.solver=solver
        self.max_iter=max_iter
        self.learning_rate_init=learning_rate_init
        self.nb_cores = nb_cores
        

        
    def fit_predict(self, X_train, y_train, X_test, y_test):
        mask = np.isfinite(y_train) & np.all(np.isfinite(X_train), axis=-1)

        self.mlp_model = MLPRegressor(
            hidden_layer_sizes=self.hidden_layer_sizes,
            activation=self.activation,
            solver=self.solver,
            max_iter=self.max_iter,
            learning_rate_init=self.learning_rate_init
        )
        
        if np.any(mask):
            y_clean = y_train[mask]
            x_clean = X_train[mask, :]
                
            self.mlp_model.fit(x_clean, y_clean)
            
            if X_test.ndim == 1:
                X_test = X_test.reshape(1, -1)
    
            mlp_preds = self.mlp_model.predict(X_test)
            mlp_preds[mlp_preds < 0] = 0
            error_ = y_test - mlp_preds
            return np.array([error_, mlp_preds]).squeeze()
        else:
            return np.array([np.nan, np.nan]).squeeze() 

    def compute_model(self, X_train, y_train, X_test, y_test):

        chunksize_x = np.round(len(y_train.get_index("X")) / self.nb_cores)
        chunksize_y = np.round(len(y_train.get_index("Y")) / self.nb_cores)
        
        X_train['T'] = y_train['T']
        y_train = y_train.transpose('T', 'Y', 'X')
        X_test = X_test.squeeze()
        
        y_test = y_test.drop_vars('T').squeeze().transpose('Y', 'X')
        client = Client(n_workers=self.nb_cores, threads_per_worker=1)
        result = xr.apply_ufunc(
            self.fit_predict,
            X_train,
            y_train.chunk({'Y': chunksize_y, 'X': chunksize_x}),
            X_test,
            y_test.chunk({'Y': chunksize_y, 'X': chunksize_x}),
            input_core_dims=[('T', 'features'), ('T',), ('features',), ()],
            vectorize=True,
            output_core_dims=[('output',)],
            dask='parallelized',
            output_dtypes=['float'],
            dask_gufunc_kwargs={'output_sizes': {'output': 2}},
        )
        result_ = result.compute()
        client.close()           
        return result
    
    @staticmethod
    def calculate_tercile_probabilities(best_guess, error_variance, first_tercile, second_tercile, dof):
        """
        Calculates the probability of each tercile category (below-normal, normal, above-normal) 
        based on the forecasted value, error variance, and specified terciles.
        
        Parameters
        ----------
        best_guess : array-like
            Forecasted value.
        error_variance : float
            Error variance associated with the forecasted value.
        first_tercile : float
            Value corresponding to the lower tercile threshold.
        second_tercile : float
            Value corresponding to the upper tercile threshold.
        dof : int
            Degrees of freedom for the t-distribution.
        
        Returns
        -------
        np.ndarray
            An array of shape (3, n_time) representing the probabilities for the three tercile categories.
        """
        n_time = len(best_guess)
        pred_prob = np.empty((3, n_time))
        
        if np.all(np.isnan(best_guess)):
            pred_prob[:] = np.nan
        else:
            error_std = np.sqrt(error_variance)
            first_t = (first_tercile - best_guess) / error_std
            second_t = (second_tercile - best_guess) / error_std
            
            pred_prob[0, :] = stats.t.cdf(first_t, df=dof)
            pred_prob[1, :] = stats.t.cdf(second_t, df=dof) - stats.t.cdf(first_t, df=dof)
            pred_prob[2, :] = 1 - stats.t.cdf(second_t, df=dof)
        
        return pred_prob

    
    def compute_prob(self, Predictant, clim_year_start, clim_year_end, Predictor, hindcast_det):
        """
        Computes tercile category probabilities for hindcasts over a climatological period.

        Parameters
        ----------
        Predictant : xarray.DataArray
            The target dataset, with dimensions ('T', 'Y', 'X').
        clim_year_start : int
            The starting year of the climatology period.
        clim_year_end : int
            The ending year of the climatology period.
        Predictor : xarray.DataArray
            The predictor dataset with dimensions ('T', 'features').
        hindcast_det : xarray.DataArray
            Hindcast deterministic results from the model.

        Returns
        -------
        xarray.DataArray
            Tercile probabilities for the predicted values, with probability, time, Y, and X dimensions.
        """
        index_start = Predictant.get_index("T").get_loc(str(clim_year_start)).start
        index_end = Predictant.get_index("T").get_loc(str(clim_year_end)).stop
        
        rainfall_for_tercile = Predictant.isel(T=slice(index_start, index_end))
        terciles = rainfall_for_tercile.quantile([0.333, 0.667], dim='T')
        error_variance = hindcast_det.sel(output="error").drop_vars("output").squeeze().var(dim='T')
        
        dof = len(Predictant.get_index("T")) - 1 - (len(Predictor.get_index("features")) + 1)
        
        hindcast_prob = xr.apply_ufunc(
            self.calculate_tercile_probabilities,
            hindcast_det.sel(output="prediction").drop_vars("output").squeeze(),
            error_variance,
            terciles.isel(quantile=0).drop_vars('quantile'),
            terciles.isel(quantile=1).drop_vars('quantile'),
            input_core_dims=[('T',), (), (), ()],
            vectorize=True,
            kwargs={'dof': dof},
            dask='parallelized',
            output_core_dims=[('probability', 'T')],
            output_dtypes=['float'],
            dask_gufunc_kwargs={'output_sizes': {'probability': 3}},
        )
        
        hindcast_prob = hindcast_prob.assign_coords(probability=('probability', ['PB', 'PN', 'PA']))
        return hindcast_prob.transpose('probability', 'T', 'Y', 'X')

    def forecast(self, Predictant, clim_year_start, clim_year_end, Predictor, hindcast_det, Predictor_for_year):

        chunksize_x = np.round(len(Predictant.get_index("X")) / self.nb_cores)
        chunksize_y = np.round(len(Predictant.get_index("Y")) / self.nb_cores)
        
        Predictor['T'] = Predictant['T']
        Predictant = Predictant.transpose('T', 'Y', 'X')
        Predictor_for_year_ = Predictor_for_year.squeeze()

        client = Client(n_workers=self.nb_cores, threads_per_worker=1)
        result = xr.apply_ufunc(
            self.fit_predict,
            Predictor,
            Predictant.chunk({'Y': chunksize_y, 'X': chunksize_x}),
            Predictor_for_year_,
            input_core_dims=[('T', 'features'), ('T',), ('features',)],
            vectorize=True,
            output_core_dims=[()],
            # output_core_dims=[('output',)],
            dask='parallelized',
            output_dtypes=['float'],
            # dask_gufunc_kwargs={'output_sizes': {'output': 1}},
        )
        result_ = result.compute()
        client.close() 

        index_start = Predictant.get_index("T").get_loc(str(clim_year_start)).start
        index_end = Predictant.get_index("T").get_loc(str(clim_year_end)).stop
        rainfall_for_tercile = Predictant.isel(T=slice(index_start, index_end))
        terciles = rainfall_for_tercile.quantile([0.333, 0.667], dim='T')
        error_variance = hindcast_det.sel(output="error").drop_vars("output").squeeze().var(dim='T')
        dof = len(Predictant.get_index("T")) - 1 - (len(Predictor.get_index("features")) + 1)
        
        hindcast_prob = xr.apply_ufunc(
            self.calculate_tercile_probabilities,
            result_.expand_dims({'T':[pd.Timestamp(Predictor_for_year.coords['T'].values).to_pydatetime()]}),
            error_variance,
            terciles.isel(quantile=0).drop_vars('quantile'),
            terciles.isel(quantile=1).drop_vars('quantile'),
            input_core_dims=[('T',), (), (), ()],
            vectorize=True,
            kwargs={'dof': dof},
            dask='parallelized',
            output_core_dims=[('probability','T',)],
            output_dtypes=['float'],
            dask_gufunc_kwargs={'output_sizes': {'probability': 3}},
        )
        hindcast_prob = hindcast_prob.assign_coords(probability=('probability', ['PB', 'PN', 'PA']))        
        return result_, hindcast_prob.drop_vars('T').squeeze().transpose('probability', 'Y', 'X') 



class WAS_RandomForest_XGBoost_Stacking_MLP:
    
    def __init__(self, nb_cores=1, rf_params=None, xgb_params=None, stacking_params=None, meta_model_params=None):
        self.nb_cores = nb_cores
    
        # Base models with custom parameters
        self.rf_model = RandomForestRegressor(n_jobs=self.nb_cores, **(rf_params or {}))
        self.xgb_model = xgb.XGBRegressor(n_jobs=self.nb_cores, **(xgb_params or {}))
    
        # Meta-model for stacking - simple neural network
        self.meta_model = MLPRegressor(**(meta_model_params or {}))
    
        # Stacking model with custom parameters
        self.stacking_model = StackingRegressor(
            estimators=[('rf', self.rf_model), ('xgb', self.xgb_model)],
            final_estimator=self.meta_model,
            n_jobs=self.nb_cores,
            **(stacking_params or {})
        )
    
    def fit_predict(self, X_train, y_train, X_test, y_test):
        mask = np.isfinite(y_train) & np.all(np.isfinite(X_train), axis=-1)
    
        if np.any(mask):
            y_clean = y_train[mask]
            x_clean = X_train[mask, :]
                
            self.stacking_model.fit(x_clean, y_clean)
    
            if X_test.ndim == 1:
                X_test = X_test.reshape(1, -1)
    
            stacked_preds = self.stacking_model.predict(X_test)
            stacked_preds[stacked_preds < 0] = 0
            error_ = y_test - stacked_preds
            return np.array([error_, stacked_preds]).squeeze()
        else:
            return np.array([np.nan, np.nan]).squeeze()


    def compute_model(self, X_train, y_train, X_test, y_test):
        """
        Computes predictions for spatiotemporal data using linear regression with parallel processing.

        Parameters
        ----------
        X_train : xarray.DataArray
            Training data (predictors) with dimensions ('T', 'Y', 'X').
        y_train : xarray.DataArray
            Training target values with dimensions ('T', 'Y', 'X').
        X_test : xarray.DataArray
            Test data (predictors), squeezed to remove singleton dimensions.
        y_test : xarray.DataArray
            Test target values with dimensions ('Y', 'X').
        
        Returns
        -------
        xarray.DataArray
            The computed model predictions and errors, with an output dimension ('output',).
        """
        # chunksize_x = np.round(len(y_train.get_index("X")) / self.nb_cores)
        # chunksize_y = np.round(len(y_train.get_index("Y")) / self.nb_cores)
        
        X_train['T'] = y_train['T']
        y_train = y_train.transpose('T', 'Y', 'X')
        X_test = X_test.squeeze()
        
        y_test = y_test.drop_vars('T').squeeze().transpose('Y', 'X')
        # client = Client(n_workers=self.nb_cores, threads_per_worker=1)
        result = xr.apply_ufunc(
            self.fit_predict,
            X_train,
            y_train,#.chunk({'Y': chunksize_y, 'X': chunksize_x}),
            X_test,
            y_test,#.chunk({'Y': chunksize_y, 'X': chunksize_x}),
            input_core_dims=[('T', 'features'), ('T',), ('features',), ()],
            vectorize=True,
            output_core_dims=[('output',)],
            dask='parallelized',
            output_dtypes=['float'],
            dask_gufunc_kwargs={'output_sizes': {'output': 2}},
        )
        # result_ = result.compute()
        # client.close()           
        return result
    
    @staticmethod
    def calculate_tercile_probabilities(best_guess, error_variance, first_tercile, second_tercile, dof):
        """
        Calculates the probability of each tercile category (below-normal, normal, above-normal) 
        based on the forecasted value, error variance, and specified terciles.
        
        Parameters
        ----------
        best_guess : array-like
            Forecasted value.
        error_variance : float
            Error variance associated with the forecasted value.
        first_tercile : float
            Value corresponding to the lower tercile threshold.
        second_tercile : float
            Value corresponding to the upper tercile threshold.
        dof : int
            Degrees of freedom for the t-distribution.
        
        Returns
        -------
        np.ndarray
            An array of shape (3, n_time) representing the probabilities for the three tercile categories.
        """
        n_time = len(best_guess)
        pred_prob = np.empty((3, n_time))
        
        if np.all(np.isnan(best_guess)):
            pred_prob[:] = np.nan
        else:
            error_std = np.sqrt(error_variance)
            first_t = (first_tercile - best_guess) / error_std
            second_t = (second_tercile - best_guess) / error_std
            
            pred_prob[0, :] = stats.t.cdf(first_t, df=dof)
            pred_prob[1, :] = stats.t.cdf(second_t, df=dof) - stats.t.cdf(first_t, df=dof)
            pred_prob[2, :] = 1 - stats.t.cdf(second_t, df=dof)
        
        return pred_prob

    
    def compute_prob(self, Predictant, clim_year_start, clim_year_end, Predictor, hindcast_det):
        """
        Computes tercile category probabilities for hindcasts over a climatological period.

        Parameters
        ----------
        Predictant : xarray.DataArray
            The target dataset, with dimensions ('T', 'Y', 'X').
        clim_year_start : int
            The starting year of the climatology period.
        clim_year_end : int
            The ending year of the climatology period.
        Predictor : xarray.DataArray
            The predictor dataset with dimensions ('T', 'features').
        hindcast_det : xarray.DataArray
            Hindcast deterministic results from the model.

        Returns
        -------
        xarray.DataArray
            Tercile probabilities for the predicted values, with probability, time, Y, and X dimensions.
        """
        index_start = Predictant.get_index("T").get_loc(str(clim_year_start)).start
        index_end = Predictant.get_index("T").get_loc(str(clim_year_end)).stop
        
        rainfall_for_tercile = Predictant.isel(T=slice(index_start, index_end))
        terciles = rainfall_for_tercile.quantile([0.333, 0.667], dim='T')
        error_variance = hindcast_det.sel(output="error").drop_vars("output").squeeze().var(dim='T')
        
        dof = len(Predictant.get_index("T")) - 1 - (len(Predictor.get_index("features")) + 1)
        
        hindcast_prob = xr.apply_ufunc(
            self.calculate_tercile_probabilities,
            hindcast_det.sel(output="prediction").drop_vars("output").squeeze(),
            error_variance,
            terciles.isel(quantile=0).drop_vars('quantile'),
            terciles.isel(quantile=1).drop_vars('quantile'),
            input_core_dims=[('T',), (), (), ()],
            vectorize=True,
            kwargs={'dof': dof},
            dask='parallelized',
            output_core_dims=[('probability', 'T')],
            output_dtypes=['float'],
            dask_gufunc_kwargs={'output_sizes': {'probability': 3}},
        )
        
        hindcast_prob = hindcast_prob.assign_coords(probability=('probability', ['PB', 'PN', 'PA']))
        return hindcast_prob.transpose('probability', 'T', 'Y', 'X')

    def forecast(self, Predictant, clim_year_start, clim_year_end, Predictor, hindcast_det, Predictor_for_year):

        chunksize_x = np.round(len(Predictant.get_index("X")) / self.nb_cores)
        chunksize_y = np.round(len(Predictant.get_index("Y")) / self.nb_cores)
        
        Predictor['T'] = Predictant['T']
        Predictant = Predictant.transpose('T', 'Y', 'X')
        Predictor_for_year_ = Predictor_for_year.squeeze()

        client = Client(n_workers=self.nb_cores, threads_per_worker=1)
        result = xr.apply_ufunc(
            self.fit_predict,
            Predictor,
            Predictant.chunk({'Y': chunksize_y, 'X': chunksize_x}),
            Predictor_for_year_,
            input_core_dims=[('T', 'features'), ('T',), ('features',)],
            vectorize=True,
            output_core_dims=[()],
            # output_core_dims=[('output',)],
            dask='parallelized',
            output_dtypes=['float'],
            # dask_gufunc_kwargs={'output_sizes': {'output': 1}},
        )
        result_ = result.compute()
        client.close() 

        index_start = Predictant.get_index("T").get_loc(str(clim_year_start)).start
        index_end = Predictant.get_index("T").get_loc(str(clim_year_end)).stop
        rainfall_for_tercile = Predictant.isel(T=slice(index_start, index_end))
        terciles = rainfall_for_tercile.quantile([0.333, 0.667], dim='T')
        error_variance = hindcast_det.sel(output="error").drop_vars("output").squeeze().var(dim='T')
        dof = len(Predictant.get_index("T")) - 1 - (len(Predictor.get_index("features")) + 1)
        
        hindcast_prob = xr.apply_ufunc(
            self.calculate_tercile_probabilities,
            result_.expand_dims({'T':[pd.Timestamp(Predictor_for_year.coords['T'].values).to_pydatetime()]}),
            error_variance,
            terciles.isel(quantile=0).drop_vars('quantile'),
            terciles.isel(quantile=1).drop_vars('quantile'),
            input_core_dims=[('T',), (), (), ()],
            vectorize=True,
            kwargs={'dof': dof},
            dask='parallelized',
            output_core_dims=[('probability','T',)],
            output_dtypes=['float'],
            dask_gufunc_kwargs={'output_sizes': {'probability': 3}},
        )
        hindcast_prob = hindcast_prob.assign_coords(probability=('probability', ['PB', 'PN', 'PA']))        
        return result_, hindcast_prob.drop_vars('T').squeeze().transpose('probability', 'Y', 'X') 



# import numpy as np
# import xarray as xr
# import pandas as pd
# import dask
# from dask.distributed import Client
# from sklearn.svm import SVC
# from sklearn.model_selection import GridSearchCV

# class WAS_SVC_Classifier:
#     """
#     A class to perform Support Vector Classification (SVC) on spatiotemporal datasets
#     for climate predictions using tercile-based classes. This adapts the structure
#     of the WAS_SVR (regression) code to classification logic, similar to the
#     WAS_LogisticRegression_Model workflow.

#     Attributes
#     ----------
#     nb_cores : int, optional
#         The number of CPU cores to use for parallel computation (default is 1).
#     kernel : str, optional
#         Kernel type to be used in SVC ('linear', 'poly', 'rbf', 'sigmoid'), by default 'rbf'.
#     gamma : str or float, optional
#         Kernel coefficient for 'rbf', 'poly', and 'sigmoid'. By default 'scale'.
#     C_range : list, optional
#         List of C values to consider during hyperparameter tuning (default [0.1, 1, 10, 100]).
#     degree_range : list, optional
#         List of degrees to consider if using the 'poly' kernel (default [2, 3, 4]).

#     Methods
#     -------
#     classify(y, index_start, index_end)
#         Tercile-based classification of the predictand.
#     fit_predict(x, y, x_test)
#         Fits SVC for the valid (finite) training data and returns predicted class probabilities.
#     compute_class(Predictant, clim_year_start, clim_year_end)
#         Assigns each data point (through time) to one of three terciles: below, near, above normal.
#     compute_model(X_train, y_train, X_test)
#         Applies `fit_predict` across each grid cell in parallel to get class probabilities.
#     forecast(Predictant, Predictor, Predictor_for_year)
#         Generates out-of-sample forecasts (class probabilities) for a given year.
#     """

#     def __init__(
#         self, 
#         nb_cores=1,
#         kernel='rbf',
#         gamma='scale',
#         C_range=[0.1, 1, 10, 100],
#         degree_range=[2, 3, 4]
#     ):
#         """
#         Initializes WAS_SVC_Classifier with specified hyperparameter ranges.

#         Parameters
#         ----------
#         nb_cores : int, optional
#             Number of CPU cores to use for parallel computation.
#         kernel : str, optional
#             Kernel type for SVC.
#         gamma : str or float, optional
#             Kernel coefficient for 'rbf', 'poly', 'sigmoid' kernels.
#         C_range : list, optional
#             Range of C values to consider.
#         degree_range : list, optional
#             Range of polynomial degrees to consider if kernel='poly'.
#         """
#         self.nb_cores = nb_cores
#         self.kernel = kernel
#         self.gamma = gamma
#         self.C_range = C_range
#         self.degree_range = degree_range

#     @staticmethod
#     def classify(y, index_start, index_end):
#         """
#         Assigns tercile-based classes (0: below, 1: normal, 2: above) to 1D array `y`
#         using the data between index_start and index_end to compute the two terciles.
        
#         Parameters
#         ----------
#         y : np.ndarray
#             The predictand array (1D over time).
#         index_start : int
#             The starting index along time dimension used to compute climatology.
#         index_end : int
#             The ending index (slice) along time dimension used to compute climatology.

#         Returns
#         -------
#         (np.ndarray, float, float)
#             Tuple of (tercile_class_array, tercile_33, tercile_67).
#         """
#         mask = np.isfinite(y)
#         if np.any(mask):
#             # Compute climatological terciles from the slice
#             terciles = np.nanpercentile(y[index_start:index_end], [33, 67])
#             # Digitize y into 3 bins based on these terciles
#             y_class = np.digitize(y, bins=terciles, right=True)
#             # y_class will contain values {0, 1, 2}
#             return y_class, terciles[0], terciles[1]
#         else:
#             return np.full(y.shape[0], np.nan), np.nan, np.nan

#     def fit_predict(self, x, y, x_test):
#         """
#         Fits an SVC model for classification on valid data, returns predicted probabilities.

#         Parameters
#         ----------
#         x : np.ndarray, shape (n_samples, n_features)
#             Training data (predictors).
#         y : np.ndarray, shape (n_samples,)
#             Training class labels (0, 1, or 2).
#         x_test : np.ndarray, shape (n_features,) or (n_samples_test, n_features)
#             Test data to predict.

#         Returns
#         -------
#         preds_proba : np.ndarray of shape (3,) or (n_samples_test, 3)
#             Predicted probabilities for each class. If x_test is 1D, output is shape (3,).
#         """
#         # We'll use SVC with probability=True to get class probabilities
#         model = SVC(kernel=self.kernel, gamma=self.gamma, probability=True)
        
#         # Filter out invalid (NaN) samples
#         mask = np.isfinite(y) & np.all(np.isfinite(x), axis=-1)
#         if np.any(mask):
#             x_clean = x[mask, :]
#             y_clean = y[mask]
#             model.fit(x_clean, y_clean)

#             # Reshape x_test if it's 1D
#             if x_test.ndim == 1:
#                 x_test = x_test.reshape(1, -1)
            
#             # Probability predictions
#             preds_proba = model.predict_proba(x_test)

#             # If x_test was originally 1D, squeeze down to shape (n_classes,)
#             preds_proba = preds_proba.squeeze(axis=0) if x_test.shape[0] == 1 else preds_proba
#             # Ensure exactly 3 classes by padding if for some reason fewer appear
#             if preds_proba.shape[-1] < 3:
#                 # this scenario can happen if SVC sees fewer than 3 unique classes in training
#                 if preds_proba.ndim == 1:
#                     # single sample
#                     proba_padded = np.full(3, np.nan)
#                     proba_padded[:preds_proba.shape[0]] = preds_proba
#                     return proba_padded
#                 else:
#                     # multiple samples
#                     proba_padded = np.full((preds_proba.shape[0], 3), np.nan)
#                     proba_padded[:, :preds_proba.shape[1]] = preds_proba
#                     return proba_padded

#             return preds_proba
#         else:
#             # Return NaNs if no valid training data
#             # If x_test is 1D, we return (3,)-sized array. Otherwise, shape (n_samples_test, 3).
#             if x_test.ndim == 1 or x_test.shape[0] == 1:
#                 return np.full((3,), np.nan)
#             else:
#                 return np.full((x_test.shape[0], 3), np.nan)

#     def compute_class(self, Predictant, clim_year_start, clim_year_end):
#         """
#         Applies tercile-based classification to the Predictant data along 'T' dimension.

#         Parameters
#         ----------
#         Predictant : xarray.DataArray
#             Data with dimensions ('T', 'Y', 'X'), e.g. precipitation or temperature fields.
#         clim_year_start : int or str
#             The start year used to slice the climatology period.
#         clim_year_end : int or str
#             The end year used to slice the climatology period.

#         Returns
#         -------
#         Predictant_class : xarray.DataArray
#             DataArray of the same shape as Predictant with classes (0,1,2) along 'T'.
#         """
#         # Convert year to string if needed, get the slice indices
#         index_start = Predictant.get_index("T").get_loc(str(clim_year_start)).start
#         index_end = Predictant.get_index("T").get_loc(str(clim_year_end)).stop

#         # vectorize the `classify` function across space
#         Predictant_class, tercile_33, tercile_67 = xr.apply_ufunc(
#             self.classify,
#             Predictant,
#             input_core_dims=[('T',)],
#             kwargs={'index_start': index_start, 'index_end': index_end},
#             vectorize=True,
#             dask='parallelized',
#             output_core_dims=[('T',), (), ()],
#             output_dtypes=['float', 'float', 'float']
#         )

#         # Return only the classes. If you need tercile values, you can also keep them.
#         return Predictant_class.transpose('T', 'Y', 'X')

#     def compute_model(self, X_train, y_train, X_test):
#         """
#         Applies fit_predict in a parallelized manner over the spatial dimensions
#         to get SVC-based class probabilities at each grid cell.

#         Parameters
#         ----------
#         X_train : xarray.DataArray
#             Training predictors with dimensions ('T', 'features').
#         y_train : xarray.DataArray
#             Class labels with dimensions ('T', 'Y', 'X').
#         X_test : xarray.DataArray
#             Test predictors with dimensions ('T', 'features') or possibly
#             a single time with shape (1, 'features').

#         Returns
#         -------
#         xarray.DataArray
#             Class probabilities with dimension ('probability', 'Y', 'X') if x_test is single-time,
#             or possibly ('T', 'probability', 'Y', 'X') if multiple test times.
#         """
#         # Set chunk sizes for parallelization
#         chunksize_x = np.round(len(y_train.get_index("X")) / self.nb_cores)
#         chunksize_y = np.round(len(y_train.get_index("Y")) / self.nb_cores)
        
#         # Ensure that the time dimension matches
#         X_train['T'] = y_train['T']
#         y_train = y_train.transpose('T', 'Y', 'X')
#         # Squeeze X_test so shape is either (T,features) or (features,)
#         X_test = X_test.transpose('T', 'features').squeeze()

#         client = Client(n_workers=self.nb_cores, threads_per_worker=1)
#         result = xr.apply_ufunc(
#             self.fit_predict,
#             X_train,
#             y_train.chunk({'Y': chunksize_y, 'X': chunksize_x}),
#             X_test,
#             input_core_dims=[('T', 'features'), ('T',), ('features',)],
#             output_core_dims=[('probability',)],  # We'll treat final dimension as "probability"
#             vectorize=True,
#             dask='parallelized',
#             output_dtypes=['float'],
#             # We'll assume exactly 3 classes => output_sizes
#             dask_gufunc_kwargs={'output_sizes': {'probability': 3}},
#         ) 
#         result_ = result.compute()
#         client.close()
#         return result_

#     def forecast(self, Predictant, Predictor, Predictor_for_year):
#         """
#         Generates forecasts (class probabilities) by applying fit_predict at each grid point.

#         Parameters
#         ----------
#         Predictant : xarray.DataArray
#             Data variable with shape ('T', 'Y', 'X') for the training period.
#         Predictor : xarray.DataArray
#             Predictors with shape ('T', 'features') for the training period.
#         Predictor_for_year : xarray.DataArray
#             Predictor for the forecast year, shape (1, 'features') or just ('features',).

#         Returns
#         -------
#         xarray.DataArray
#             The forecast probabilities, shape ('probability', 'Y', 'X').
#         """
#         chunksize_x = np.round(len(Predictant.get_index("X")) / self.nb_cores)
#         chunksize_y = np.round(len(Predictant.get_index("Y")) / self.nb_cores)
        
#         # Align time dimension
#         Predictor['T'] = Predictant['T']
#         Predictant = Predictant.transpose('T', 'Y', 'X')
#         Predictor_for_year_ = Predictor_for_year.squeeze()
        
#         client = Client(n_workers=self.nb_cores, threads_per_worker=1)
#         result = xr.apply_ufunc(
#             self.fit_predict,
#             Predictor,
#             Predictant.chunk({'Y': chunksize_y, 'X': chunksize_x}),
#             Predictor_for_year_,
#             input_core_dims=[('T', 'features'), ('T',), ('features',)],
#             output_core_dims=[('probability',)],
#             vectorize=True,
#             dask='parallelized',
#             output_dtypes=['float'],
#             dask_gufunc_kwargs={'output_sizes': {'probability': 3}},
#         )
#         result_ = result.compute()
#         client.close()

#         # Optionally rename the probability dimension labels to something meaningful
#         # if you have 3 classes: e.g. PB, PN, PA
#         result_ = result_.assign_coords(probability=('probability', ['PB', 'PN', 'PA']))
        
#         # Typically, you'd want to drop 'T' because it's singular or doesn't apply in the forecast
#         return result_.drop_vars('T').squeeze().transpose('probability', 'Y', 'X')




# import numpy as np
# import xarray as xr
# import pandas as pd

# from dask.distributed import Client
# from sklearn.cluster import KMeans
# from sklearn.model_selection import GridSearchCV
# from sklearn.svm import SVC

# class WAS_SVC_Classifier:
#     """
#     A class to perform Support Vector Classification (SVC) on spatiotemporal datasets
#     for climate prediction using tercile-based classes (0: below, 1: normal, 2: above).
#     This version includes a fully parallel hyperparameter search using GridSearchCV,
#     inspired by your WAS_SVR approach.
#     """

#     def __init__(
#         self, 
#         nb_cores=1,
#         n_clusters=5,
#         kernel='rbf',
#         gamma='scale',  # could also be e.g., ['scale', 'auto']
#         C_range=[0.1, 1, 10, 100],
#         degree_range=[2, 3, 4]
#     ):
#         """
#         Initializes WAS_SVC_Classifier with specified hyperparameter ranges.

#         Parameters
#         ----------
#         nb_cores : int
#             Number of CPU cores for parallel processing.
#         n_clusters : int
#             Number of clusters for KMeans.
#         kernel : str
#             Kernel type for SVC.
#         gamma : str or float
#             Kernel coefficient for 'rbf', 'poly', 'sigmoid'.
#         C_range : list
#             Range of C values to consider during hyperparameter tuning.
#         degree_range : list
#             Range of polynomial degrees to consider if kernel='poly'.
#         """
#         self.nb_cores = nb_cores
#         self.n_clusters = n_clusters
#         self.kernel = kernel
#         self.gamma = gamma
#         self.C_range = C_range
#         self.degree_range = degree_range

#     def classify(self, y, index_start, index_end):
#         """
#         Example tercile-based classification function.
#         See your existing code for an actual implementation.
#         Returns 0,1,2 class labels, plus the two terciles.
#         """
#         mask = np.isfinite(y)
#         if np.any(mask):
#             terciles = np.nanpercentile(y[index_start:index_end], [33, 67])
#             y_class = np.digitize(y, bins=terciles, right=True)
#             return y_class, terciles[0], terciles[1]
#         else:
#             return np.full(y.shape[0], np.nan), np.nan, np.nan

#     def fit_predict(self, x, y, x_test):
#         """
#         Fit SVC on valid data, return predicted probabilities for x_test.
#         """
#         model = SVC(probability=True, kernel=self.kernel, gamma=self.gamma)
#         mask = np.isfinite(y) & np.all(np.isfinite(x), axis=-1)
#         if np.any(mask):
#             x_clean = x[mask, :]
#             y_clean = y[mask]
#             model.fit(x_clean, y_clean)

#             if x_test.ndim == 1:
#                 x_test = x_test.reshape(1, -1)

#             preds_proba = model.predict_proba(x_test).squeeze()
#             if preds_proba.ndim == 1 and preds_proba.size < 3:
#                 # If fewer than 3 classes discovered in training
#                 padded = np.full(3, np.nan)
#                 padded[:preds_proba.size] = preds_proba
#                 return padded
#             elif preds_proba.ndim == 2 and preds_proba.shape[1] < 3:
#                 padded = np.full((preds_proba.shape[0], 3), np.nan)
#                 padded[:, :preds_proba.shape[1]] = preds_proba
#                 return padded

#             return preds_proba
#         else:
#             # Return NaNs if no valid data
#             if x_test.ndim == 1 or x_test.shape[0] == 1:
#                 return np.full(3, np.nan)
#             else:
#                 return np.full((x_test.shape[0], 3), np.nan)

#     def compute_hyperparameters(self, predictand, predictor):
#         """
#         Computes optimal SVC hyperparameters (C, gamma, and possibly degree)
#         for each spatial cluster using a KMeans approach.

#         Parameters
#         ----------
#         predictand : xarray.DataArray
#             Classification labels or something from which to derive them.
#             Shape: ('T', 'Y', 'X'). Should be integer labels (0,1,2) or
#             a continuous variable from which we can cluster.
#         predictor : xarray.DataArray
#             Predictor data. Typically has shape ('T', 'features').

#         Returns
#         -------
#         C_array : xarray.DataArray
#             Best C values for each (Y,X) grid cell.
#         gamma_array : xarray.DataArray
#             Best gamma values for each (Y,X) grid cell (if relevant).
#         degree_array : xarray.DataArray
#             Best polynomial degree for each (Y,X) if kernel='poly'.
#         cluster_da : xarray.DataArray
#             The cluster assignments for each (Y,X).
#         """

#         # 1) Flatten spatial dims, drop NaNs, and run KMeans
#         #    We'll cluster based on the mean (or any summary) across time of the predictand.
#         df = predictand.mean(dim='T', skipna=True).to_dataframe(name='mean_label')
#         df = df.dropna()
#         coords_df = df.reset_index()[['Y', 'X', 'mean_label']]
        
#         # KMeans on the single column 'mean_label'
#         kmeans = KMeans(n_clusters=self.n_clusters, random_state=42)
#         coords_df['cluster'] = kmeans.fit_predict(coords_df[['mean_label']])

#         # Convert back to xarray
#         df_unique = coords_df.drop_duplicates(subset=['Y', 'X'])
#         dataset_clusters = df_unique.set_index(['Y', 'X']).to_xarray()
#         cluster_da = dataset_clusters['cluster']  # shape (Y, X)

#         # 2) For each cluster, compute the 'average' or 'representative' time-series
#         #    of the classification label and run a grid search.
#         #    (You need your actual classification labels for each time. This example
#         #     assumes `predictand` is already 0,1,2 or we do something analogous.)
        
#         # param_grid can include multiple kernels if you want: e.g. if self.kernel == 'all'
#         param_grid = []
#         if self.kernel in ['linear', 'all']:
#             param_grid.append({
#                 'kernel': ['linear'],
#                 'C': self.C_range,
#             })
#         if self.kernel in ['poly', 'all', 'rbf']:  # we might unify if statements
#             param_grid.append({
#                 'kernel': ['poly'],
#                 'C': self.C_range,
#                 'degree': self.degree_range,
#                 'gamma': [self.gamma] if isinstance(self.gamma, (str, float)) else self.gamma
#             })
#         if self.kernel in ['rbf', 'all']:
#             param_grid.append({
#                 'kernel': ['rbf'],
#                 'C': self.C_range,
#                 'gamma': [self.gamma] if isinstance(self.gamma, (str, float)) else self.gamma
#             })
#         if self.kernel in ['sigmoid', 'all']:
#             param_grid.append({
#                 'kernel': ['sigmoid'],
#                 'C': self.C_range,
#                 'gamma': [self.gamma] if isinstance(self.gamma, (str, float)) else self.gamma
#             })

#         # Prepare a model & GridSearch
#         # NOTE: In classification, you might use 'accuracy', 'f1_macro', or another metric.
#         model = SVC(probability=True)
#         grid_search = GridSearchCV(
#             estimator=model,
#             param_grid=param_grid,
#             cv=5,
#             scoring='accuracy',  # or 'f1_micro', 'f1_weighted', etc.
#             n_jobs=-1  # parallelize across all local cores
#         )

#         # We'll store best params for each cluster
#         hyperparams_cluster = {}

#         # find all unique cluster labels
#         unique_clusters = np.unique(cluster_da.values)
#         unique_clusters = unique_clusters[~np.isnan(unique_clusters)].astype(int)

#         for cl in unique_clusters:
#             # subselect all grid cells in this cluster
#             # average the classification labels across that cluster
#             # We want a time series of classification labels => shape (T,)
#             # e.g. cluster_mean_class = predictand.where(cluster_da == cl).mean(...)
#             # But for classification, averaging labels doesn't always make sense, so:
#             #   a) you might pick the "most frequent" label per time
#             #   b) or you might do the raw data, flatten them in space, etc.
#             #
#             # For demonstration, let's do a simple approach: flatten all (Y,X) in cluster => (T*Ngrid).
#             cluster_mask = (cluster_da == cl)
#             # Broadcast the mask to match (T, Y, X)
#             mask_3d = xr.where(cluster_mask, 1, np.nan).broadcast_like(predictand)
#             cluster_vals = predictand.where(mask_3d.notnull())

#             # Flatten over Y,X (ignore NaNs)
#             stacked_cluster_vals = cluster_vals.stack(z=('Y', 'X')).dropna(dim='z')
#             # stacked_cluster_vals is shape (T, z), each cell is a label. We can flatten further:
#             # We'll create (T*z,) array of labels:
#             y_cluster_flat = stacked_cluster_vals.values.ravel()

#             # We need corresponding predictor data for these times. The simplest approach:
#             # - Use the same T dimension
#             # - We do not replicate for 'z', since predictor has shape (T, features).
#             # So we just do a time-based model for cluster average or something like that.
#             # This is tricky. For demonstration, let's do a single cluster-mean predictor across space:
#             # (This is quite simplistic. In practice, you might do a more refined approach.)
#             pred_cluster_mean = predictor.mean(dim='features', skipna=True)
#             # shape => (T,)
#             # Then we expand dims for scikit: shape => (T, 1)
#             x_cluster = pred_cluster_mean.values.reshape(-1, 1)

#             # We must be consistent with shape. Because we have y_cluster_flat is (T*z,).
#             # We only have x of shape (T,) for each time. This mismatch is a conceptual challenge.
#             # In a more thorough approach, you would want to replicate x across z or vice versa.
#             # But let's do a 1:1 approach for each time, ignoring spatial flatten:
#             # => We only use cluster-mean label across T => shape (T,) => pick e.g. majority label.
#             # This snippet is demonstration only.
            
#             # For demonstration, let's do:
#             #   y_cluster_time = the "majority" label among the cluster for each time
#             # We'll do that quickly:
#             y_cluster_time = stacked_cluster_vals.to_dataframe(name='label')
#             # group by T, get majority label
#             grouped = y_cluster_time.groupby(level='T')['label'].agg(lambda s: s.value_counts().index[0])
#             # group is now shape (T,)
#             y_cluster_maj = grouped.values  # => shape (T,)

#             # Now x_cluster and y_cluster_maj align by T dimension
#             # shape: x_cluster (T,) => we can do x_cluster.reshape(-1,1) for scikit
#             x_cluster_2D = x_cluster.reshape(-1, 1)  # shape (T,1)
            
#             # Filter out times with NaNs
#             valid_mask = np.isfinite(y_cluster_maj) & np.isfinite(x_cluster_2D).all(axis=1)
#             X_valid = x_cluster_2D[valid_mask]
#             y_valid = y_cluster_maj[valid_mask]

#             if len(X_valid) < 2:
#                 # Not enough data => skip
#                 continue

#             # Fit GridSearch
#             grid_search.fit(X_valid, y_valid)
#             best_params = grid_search.best_params_
#             hyperparams_cluster[cl] = {
#                 'C': best_params['C'],
#                 'kernel': best_params['kernel'],
#                 'gamma': best_params.get('gamma', np.nan),
#                 'degree': best_params.get('degree', np.nan),
#             }

#         # 3) Map best hyperparameters to each grid cell
#         # We'll create xarray DataArrays for C, gamma, degree
#         shape_yx = (predictand.sizes['Y'], predictand.sizes['X'])
#         C_array = xr.full_like(cluster_da, np.nan, dtype=float)
#         gamma_array = xr.full_like(cluster_da, np.nan, dtype=float)
#         degree_array = xr.full_like(cluster_da, np.nan, dtype=float)

#         for cl, params in hyperparams_cluster.items():
#             mask_ = (cluster_da == cl)
#             C_array = C_array.where(~mask_, other=params['C'])
#             gamma_array = gamma_array.where(~mask_, other=params['gamma'])
#             degree_array = degree_array.where(~mask_, other=params['degree'])

#         # Align them
#         C_array, gamma_array, degree_array, cluster_da = xr.align(
#             C_array, gamma_array, degree_array, cluster_da, join='outer'
#         )

#         return C_array, gamma_array, degree_array, cluster_da


#     def compute_model(self, X_train, y_train, X_test, C_array, gamma_array, degree_array):
#         """
#         Parallel classification predictions at each grid cell,
#         using the pre-computed hyperparameters (C, gamma, degree).
        
#         For simplicity, we only pass a single set of kernel/gamma/C,
#         but you could store these as arrays if your domain has different values per cluster.
        
#         Parameters
#         ----------
#         X_train : xarray.DataArray
#             Training predictors with shape ('T', 'features').
#         y_train : xarray.DataArray
#             Classification labels with shape ('T', 'Y', 'X') => 0,1,2.
#         X_test : xarray.DataArray
#             Test predictor(s), e.g. shape ('T', 'features') or (features,).
#         C_array, gamma_array, degree_array : xarray.DataArray
#             Hyperparameter arrays from compute_hyperparameters().

#         Returns
#         -------
#         xarray.DataArray
#             Class probabilities with dimension ('probability', 'Y', 'X') or
#             ('probability', 'T', 'Y', 'X') depending on X_test shape.
#         """
#         # Implementation logic: 
#         #    1) For each (Y,X), pick the best hyperparams from C_array, gamma_array, etc.
#         #    2) Fit an SVC with those hyperparams
#         #    3) Return predicted probabilities
#         #
#         # This can be done with xr.apply_ufunc similarly to your SVR approach.

#         # Example param retrieval:
#         # For each grid cell (y_idx, x_idx):
#         #   pass kernel=self.kernel, 
#         #        C=C_array[y_idx, x_idx], 
#         #        gamma=gamma_array[y_idx, x_idx], 
#         #        degree=degree_array[y_idx, x_idx]
#         #
#         # Then do something akin to `fit_predict(...)`.

#         chunksize_x = int(np.round(len(y_train.get_index("X")) / self.nb_cores))
#         chunksize_y = int(np.round(len(y_train.get_index("Y")) / self.nb_cores))

#         # Align time dimension
#         X_train['T'] = y_train['T']
#         y_train = y_train.transpose('T', 'Y', 'X')
#         X_test = X_test.squeeze()

#         def fit_predict_with_params(x_, y_, x_test_, c_, gamma_, deg_):
#             """
#             A wrapper that sets up an SVC with the given hyperparams, 
#             then fits/predicts probabilities for classification.
#             """
#             # Ensure shape
#             if np.isnan(c_):
#                 # If no valid hyperparams => return all NaNs
#                 if x_test_.ndim == 1:
#                     return np.full(3, np.nan)
#                 else:
#                     return np.full((x_test_.shape[0], 3), np.nan)
            
#             # Build SVC
#             kernel_ = self.kernel
#             if np.isnan(gamma_):  # handle nan
#                 gamma_ = 'scale'
#             model_params = {
#                 'kernel': kernel_,
#                 'C': c_,
#                 'gamma': gamma_,
#                 'probability': True
#             }
#             if kernel_ == 'poly' and not np.isnan(deg_):
#                 model_params['degree'] = int(deg_)

#             clf = SVC(**model_params)
#             mask = np.isfinite(y_) & np.all(np.isfinite(x_), axis=-1)
#             if np.any(mask):
#                 x_clean = x_[mask, :]
#                 y_clean = y_[mask]
#                 if len(np.unique(y_clean)) < 2:
#                     # If only one class in training => fill with NaNs or handle gracefully
#                     if x_test_.ndim == 1:
#                         return np.full(3, np.nan)
#                     else:
#                         return np.full((x_test_.shape[0], 3), np.nan)

#                 clf.fit(x_clean, y_clean)
#                 # shape check for x_test_
#                 if x_test_.ndim == 1:
#                     x_test_ = x_test_.reshape(1, -1)
#                 preds_proba = clf.predict_proba(x_test_)
#                 preds_proba = np.squeeze(preds_proba)
#                 # Pad if fewer than 3 classes discovered
#                 if preds_proba.ndim == 1 and preds_proba.size < 3:
#                     p_ = np.full(3, np.nan)
#                     p_[:preds_proba.size] = preds_proba
#                     return p_
#                 elif preds_proba.ndim == 2 and preds_proba.shape[1] < 3:
#                     p_ = np.full((preds_proba.shape[0], 3), np.nan)
#                     p_[:, :preds_proba.shape[1]] = preds_proba
#                     return p_
#                 return preds_proba
#             else:
#                 # No valid data
#                 if x_test_.ndim == 1:
#                     return np.full(3, np.nan)
#                 else:
#                     return np.full((x_test_.shape[0], 3), np.nan)

#         client = Client(n_workers=self.nb_cores, threads_per_worker=1)
#         result = xr.apply_ufunc(
#             fit_predict_with_params,
#             X_train,
#             y_train.chunk({'Y': chunksize_y, 'X': chunksize_x}),
#             X_test,
#             C_array.chunk({'Y': chunksize_y, 'X': chunksize_x}),
#             gamma_array.chunk({'Y': chunksize_y, 'X': chunksize_x}),
#             degree_array.chunk({'Y': chunksize_y, 'X': chunksize_x}),
#             input_core_dims=[
#                 ('T', 'features'),  # x_
#                 ('T',),             # y_
#                 ('features',),      # x_test_
#                 (),                 # c_
#                 (),                 # gamma_
#                 ()                  # deg_
#             ],
#             vectorize=True,
#             output_core_dims=[('probability',)],
#             dask='parallelized',
#             output_dtypes=['float'],
#             dask_gufunc_kwargs={'output_sizes': {'probability': 3}},
#         )
#         result_ = result.compute()
#         client.close()

#         # Possibly rename probability dimension, e.g. => PB, PN, PA
#         result_ = result_.assign_coords(probability=('probability', ['PB', 'PN', 'PA']))
#         return result_

#     def forecast(self, Predictant, Predictor, Predictor_for_year):
#         """
#         Generates out-of-sample forecast with the currently stored (self) kernel, gamma, etc.
#         If you have a separate hyperparameter array, pass them in similarly to compute_model.
#         This is a simpler version that reuses fit_predict at each grid cell.
#         """
#         # Similar logic to your logistic or WAS_SVR code:
#         #  1) align T dimension
#         #  2) run xr.apply_ufunc to do classification
#         #  3) return the predicted probabilities
#         pass  # Implementation omitted here for brevity
