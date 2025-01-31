########  This code was developed by Mandela Houngnibo et al. within the framework of AGRHYMET WAS-RCC S2S. #################### Version 1.0.0 #########################

######################################################## Modules ########################################################
from sklearn import linear_model
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import GridSearchCV
import statsmodels.api as sm
from statsmodels.stats.outliers_influence import variance_inflation_factor as VIF
from statsmodels.stats.anova import anova_lm
import xarray as xr 
import numpy as np
import pandas as pd
from scipy import stats
from scipy.stats import lognorm
from scipy.stats import gamma
from sklearn.cluster import KMeans
import xeofs as xe
import xarray as xr
import numpy as np
import scipy.signal as sig
from scipy.interpolate import CubicSpline
from multiprocessing import cpu_count
from dask.distributed import Client
import dask.array as da


#### Add Nonexcedance function for all models ##############################################

class WAS_LinearRegression_Model:
    """
    A class to perform linear regression modeling on spatiotemporal datasets for climate prediction.

    This class is designed to work with Dask and Xarray for parallelized, high-performance 
    regression computations across large datasets with spatial and temporal dimensions. The primary 
    methods are for fitting the model, making predictions, and calculating probabilistic predictions 
    for climate terciles. 

    Attributes
    ----------
    nb_cores : int, optional
        The number of CPU cores to use for parallel computation (default is 1).
    
    Methods
    -------
    
    fit_predict(x, y, x_test, y_test)
        Fits a linear regression model to the training data, predicts on test data, and computes error.
    
    compute_model(X_train, y_train, X_test, y_test)
        Applies the linear regression model across a dataset using parallel computation with Dask, 
        returning predictions and error metrics.
    
    calculate_tercile_probabilities(best_guess, error_variance, first_tercile, second_tercile, dof)
        Calculates the probabilities for three tercile categories (below-normal, normal, above-normal) 
        based on predictions and associated error variance.
    
    compute_prob(Predictant, clim_year_start, clim_year_end, Predictor, hindcast_det)
        Computes tercile probabilities for hindcast rainfall predictions over specified climatological 
        years.
    """
    def __init__(self, nb_cores=1):
        """
        Initializes the WAS_LinearRegression_Model with a specified number of CPU cores.
        
        Parameters
        ----------
        nb_cores : int, optional
            Number of CPU cores to use for parallel computation, by default 1.
        """
        self.nb_cores = nb_cores
    
    def fit_predict(self, x, y, x_test, y_test = None):
        """
        Fits a linear regression model to the provided training data, makes predictions on the test data, 
        and calculates the prediction error.
        
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
        model = linear_model.LinearRegression()
        mask = np.isfinite(y) & np.all(np.isfinite(x), axis=-1)

        if y_test is not None:
        
            if np.any(mask):
                y_clean = y[mask]
                x_clean = x[mask, :]
                model.fit(x_clean, y_clean)
                
                if x_test.ndim == 1:
                    x_test = x_test.reshape(1, -1)
                
                preds = model.predict(x_test)
                preds[preds < 0] = 0
                error_ = y_test - preds
                return np.array([error_, preds]).squeeze()
            else:
                return np.array([np.nan, np.nan]).squeeze()
        else:
            if np.any(mask):
                y_clean = y[mask]
                x_clean = x[mask, :]
                model.fit(x_clean, y_clean)
                
                if x_test.ndim == 1:
                    x_test = x_test.reshape(1, -1)
                
                preds = model.predict(x_test)
                preds[preds < 0] = 0
                return np.array([preds]).squeeze()
            else:
                return np.array([np.nan]).squeeze()
    
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

    @staticmethod
    def calculate_tercile_probabilities_gamma(best_guess, error_variance, T1, T2, dof):

        n_time = len(best_guess)
        pred_prob = np.empty((3, n_time), dtype=float)
    
        # If all best_guess are NaN, just fill everything with NaN
        if np.any(np.isnan(best_guess)) or np.any(np.isnan(error_variance)):
            pred_prob[:] = np.nan
            return pred_prob
    
        # Convert inputs to arrays (in case they're lists)
        best_guess = np.asarray(best_guess, dtype=float)
        error_variance = np.asarray(error_variance, dtype=float)
        T1 = np.asarray(T1, dtype=float)
        T2 = np.asarray(T2, dtype=float)
    
        # Calculate shape (alpha) and scale (theta) for the Gamma distribution
        # alpha = (mean^2) / variance
        # theta = variance / mean
        alpha = (best_guess**2) / error_variance
        theta = error_variance / best_guess
    
        # Compute CDF at T1, T2 (no loop over n_time)
        cdf_t1 = gamma.cdf(T1, a=alpha, scale=theta)  # P(X < T1)
        cdf_t2 = gamma.cdf(T2, a=alpha, scale=theta)  # P(X < T2)
    
        # Fill out the probabilities
        pred_prob[0, :] = cdf_t1
        pred_prob[1, :] = cdf_t2 - cdf_t1
        pred_prob[2, :] = 1.0 - cdf_t2

        dof=dof
    
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
        terciles = rainfall_for_tercile.quantile([0.3, 0.666667], dim='T')
        error_variance = hindcast_det.sel(output="error").drop_vars("output").squeeze().var(dim='T')
        
        dof = len(Predictant.get_index("T")) - 1 - (len(Predictor.get_index("features")) + 1)
        
        hindcast_prob = xr.apply_ufunc(
            self.calculate_tercile_probabilities_gamma,
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
        terciles = rainfall_for_tercile.quantile([0.3, 0.66], dim='T')
        error_variance = hindcast_det.sel(output="error").drop_vars("output").squeeze().var(dim='T')
        dof = len(Predictant.get_index("T")) - 1 - (len(Predictor.get_index("features")) + 1)
        
        hindcast_prob = xr.apply_ufunc(
            self.calculate_tercile_probabilities_gamma,
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
    
    # @staticmethod
    # def calculate_tercile_probabilities_lognormal(best_guess, error_variance, first_tercile, second_tercile):
    #     """
    #     Calculate tercile probabilities for predictions using the Log-Normal distribution.
    
    #     Parameters:
    #         best_guess (ndarray): Predicted values (must be positive).
    #         error_variance (ndarray): Error variance of predictions.
    #         first_tercile (float): First tercile value (must be positive).
    #         second_tercile (float): Second tercile value (must be positive).
    
    #     Returns:
    #         ndarray: Probabilities for below normal, normal, and above normal categories.
    #     """
    #     n_time = len(best_guess)
    #     pred_prob = np.empty((3, n_time))
    #     if np.all(np.isnan(best_guess)):
    #         pred_prob[:] = np.nan
    #     else:

    #         mu = np.log(best_guess ** 2 / np.sqrt(error_variance + best_guess ** 2))
    #         sigma = np.sqrt(np.log(error_variance / best_guess ** 2 + 1))
    
    #         # Calculate cumulative probabilities
    #         first_cdf = stats.lognorm.cdf(first_tercile, s=sigma, scale=np.exp(mu))
    #         second_cdf = stats.lognorm.cdf(second_tercile, s=sigma, scale=np.exp(mu))
    
    #         # Assign probabilities
    #         pred_prob[0, :] = first_cdf        # Below Normal
    #         pred_prob[1, :] = second_cdf - first_cdf  # Near Normal
    #         pred_prob[2, :] = 1 - second_cdf   # Above Normal
    
    #     return pred_prob
    
    
    # def compute_prob(self, Predictant, clim_year_start, clim_year_end, Predictor, hindcast_det):
    #     """
    #     Computes tercile category probabilities for hindcasts over a climatological period.

    #     Parameters
    #     ----------
    #     Predictant : xarray.DataArray
    #         The target dataset, with dimensions ('T', 'Y', 'X').
    #     clim_year_start : int
    #         The starting year of the climatology period.
    #     clim_year_end : int
    #         The ending year of the climatology period.
    #     Predictor : xarray.DataArray
    #         The predictor dataset with dimensions ('T', 'features').
    #     hindcast_det : xarray.DataArray
    #         Hindcast deterministic results from the model.

    #     Returns
    #     -------
    #     xarray.DataArray
    #         Tercile probabilities for the predicted values, with probability, time, Y, and X dimensions.
    #     """
    #     index_start = Predictant.get_index("T").get_loc(str(clim_year_start)).start
    #     index_end = Predictant.get_index("T").get_loc(str(clim_year_end)).stop
        
    #     rainfall_for_tercile = Predictant.isel(T=slice(index_start, index_end))
    #     terciles = rainfall_for_tercile.quantile([0.333, 0.667], dim='T')
    #     error_variance = hindcast_det.sel(output="error").drop_vars("output").squeeze().var(dim='T')
        
    #     # dof = len(Predictant.get_index("T")) - 1 - (len(Predictor.get_index("features")) + 1)
        
    #     hindcast_prob = xr.apply_ufunc(
    #         self.calculate_tercile_probabilities_lognormal,
    #         hindcast_det.sel(output="prediction").drop_vars("output").squeeze(),
    #         error_variance,
    #         terciles.isel(quantile=0).drop_vars('quantile'),
    #         terciles.isel(quantile=1).drop_vars('quantile'),
    #         input_core_dims=[('T',), (), (), ()],
    #         vectorize=True,
    #         # kwargs={'dof': dof},
    #         dask='parallelized',
    #         output_core_dims=[('probability', 'T')],
    #         output_dtypes=['float'],
    #         dask_gufunc_kwargs={'output_sizes': {'probability': 3}},
    #     )
        
    #     hindcast_prob = hindcast_prob.assign_coords(probability=('probability', ['PB', 'PN', 'PA']))
    #     return hindcast_prob.transpose('probability', 'T', 'Y', 'X')

    # def forecast(self, Predictant, clim_year_start, clim_year_end, Predictor, hindcast_det, Predictor_for_year):

    #     chunksize_x = np.round(len(Predictant.get_index("X")) / self.nb_cores)
    #     chunksize_y = np.round(len(Predictant.get_index("Y")) / self.nb_cores)
        
    #     Predictor['T'] = Predictant['T']
    #     Predictant = Predictant.transpose('T', 'Y', 'X')
    #     Predictor_for_year_ = Predictor_for_year.squeeze()

    #     client = Client(n_workers=self.nb_cores, threads_per_worker=1)
    #     result = xr.apply_ufunc(
    #         self.fit_predict,
    #         Predictor,
    #         Predictant.chunk({'Y': chunksize_y, 'X': chunksize_x}),
    #         Predictor_for_year_,
    #         input_core_dims=[('T', 'features'), ('T',), ('features',)],
    #         vectorize=True,
    #         output_core_dims=[()],
    #         # output_core_dims=[('output',)],
    #         dask='parallelized',
    #         output_dtypes=['float'],
    #         # dask_gufunc_kwargs={'output_sizes': {'output': 1}},
    #     )
    #     result_ = result.compute()
    #     client.close() 

    #     index_start = Predictant.get_index("T").get_loc(str(clim_year_start)).start
    #     index_end = Predictant.get_index("T").get_loc(str(clim_year_end)).stop
    #     rainfall_for_tercile = Predictant.isel(T=slice(index_start, index_end))
    #     terciles = rainfall_for_tercile.quantile([0.333, 0.667], dim='T')
    #     error_variance = hindcast_det.sel(output="error").drop_vars("output").squeeze().var(dim='T')
    #     dof = len(Predictant.get_index("T")) - 1 - (len(Predictor.get_index("features")) + 1)
        
    #     hindcast_prob = xr.apply_ufunc(
    #         self.calculate_tercile_probabilities_lognormal,
    #         result_.expand_dims({'T':[pd.Timestamp(Predictor_for_year.coords['T'].values).to_pydatetime()]}),
    #         error_variance,
    #         terciles.isel(quantile=0).drop_vars('quantile'),
    #         terciles.isel(quantile=1).drop_vars('quantile'),
    #         input_core_dims=[('T',), (), (), ()],
    #         vectorize=True,
    #         # kwargs={'dof': dof},
    #         dask='parallelized',
    #         output_core_dims=[('probability','T',)],
    #         output_dtypes=['float'],
    #         dask_gufunc_kwargs={'output_sizes': {'probability': 3}},
    #     )
    #     hindcast_prob = hindcast_prob.assign_coords(probability=('probability', ['PB', 'PN', 'PA']))        
    #     return result_, hindcast_prob.drop_vars('T').squeeze().transpose('probability', 'Y', 'X')  


class WAS_Ridge_Model:
    """
    A class to perform ridge regression modeling for rainfall prediction with spatial clustering and hyperparameter optimization. By Mandela HOUNGNIBO

    Attributes:
        alpha_range (array-like): Range of alpha values to explore for ridge regression.
        n_clusters (int): Number of clusters for KMeans clustering.
        nb_cores (int): Number of cores to use for parallel computation.

    Methods:
        fit_predict(x, y, x_test, y_test, alpha): Fits a ridge regression model using the provided data and makes predictions.
        compute_hyperparameters(predictand, predictor): Computes optimal ridge hyperparameters (alpha values) for different clusters.
        compute_model(X_train, y_train, X_test, y_test, alpha): Fits and predicts ridge model for spatiotemporal data using Dask for parallel computation.
        calculate_tercile_probabilities(best_guess, error_variance, first_tercile, second_tercile, dof): Calculates probabilities of tercile events for predictions.
        compute_prob(Predictant, clim_year_start, clim_year_end, Predictor, hindcast_det): Computes the probabilities of tercile categories for rainfall prediction.
    """

    def __init__(self, alpha_range=None, n_clusters=5, nb_cores=1):
        if alpha_range is None:
            alpha_range = np.logspace(-10, 10, 100)
        self.alpha_range = alpha_range
        self.n_clusters = n_clusters
        self.nb_cores = nb_cores

    def fit_predict(self, x, y, x_test, y_test, alpha):
        """
        Fit a ridge regression model and make predictions.

        Parameters:
            x (ndarray): Training data.
            y (ndarray): Target values for training data.
            x_test (ndarray): Test data.
            y_test (ndarray): Target values for test data.
            alpha (float): Regularization strength.

        Returns:
            ndarray: Prediction error and predicted values.
        """
        model = linear_model.Ridge(alpha)
        mask = np.isfinite(y) & np.all(np.isfinite(x), axis=-1)

        if np.any(mask):
            y_clean = y[mask]
            x_clean = x[mask, :]
            model.fit(x_clean, y_clean)

            if x_test.ndim == 1:
                x_test = x_test.reshape(1, -1)

            preds = model.predict(x_test)
            preds[preds < 0] = 0
            error_ = y_test - preds
            return np.array([error_, preds]).squeeze()
        else:
            return np.array([np.nan, np.nan]).squeeze()

    def compute_hyperparameters(self, predictand, predictor):
        """
        Compute optimal hyperparameters (alpha) for ridge regression for different clusters.

        Parameters:
            predictand (xarray.DataArray): Predictand data for clustering.
            predictor (ndarray): Predictor data for model fitting.

        Returns:
            tuple: Alpha values for each cluster and the cluster assignments.
        """
        kmeans = KMeans(n_clusters=self.n_clusters)
        predictand_dropna = predictand.to_dataframe().reset_index().dropna().drop(columns=['T'])
        predictand_dropna['cluster'] = kmeans.fit_predict(predictand_dropna[predictand_dropna.columns[2]].to_frame())

        df_unique = predictand_dropna.drop_duplicates(subset=['Y', 'X'])
        dataset = df_unique.set_index(['Y', 'X']).to_xarray()

        Cluster = (dataset['cluster'] * xr.where(~np.isnan(predictand.isel(T=0)), 1, np.nan)).drop_vars("T")
        xarray1, xarray2 = xr.align(predictand, Cluster)

        clusters = np.unique(xarray2)
        clusters = clusters[~np.isnan(clusters)]

        cluster_means = {
            int(cluster): xarray1.where(xarray2 == cluster).mean(dim=['Y', 'X'], skipna=True)
            for cluster in clusters
        }

        model = linear_model.RidgeCV(alphas=self.alpha_range, cv=5)
        alpha_cluster = {
            int(cluster): model.fit(predictor, cluster_means[cluster]).alpha_
            for cluster in clusters
        }

        alpha_array = Cluster.copy()
        for key, value in alpha_cluster.items():
            alpha_array = alpha_array.where(alpha_array != key, other=value)

        alpha_array, Cluster, predictand = xr.align(alpha_array, Cluster, predictand, join = "outer")

        return alpha_array, Cluster

    def compute_model(self, X_train, y_train, X_test, y_test, alpha):
        """
        Fit and predict ridge regression model for spatiotemporal data using Dask for parallel computation.

        Parameters:
            X_train (xarray.DataArray): Training predictor data.
            y_train (xarray.DataArray): Training predictand data.
            X_test (xarray.DataArray): Test predictor data.
            y_test (xarray.DataArray): Test predictand data.
            alpha (xarray.DataArray): Alpha values for regularization.

        Returns:
            xarray.DataArray: Prediction errors and predicted values.
        """
        chunksize_x = np.round(len(y_train.get_index("X")) / self.nb_cores)
        chunksize_y = np.round(len(y_train.get_index("Y")) / self.nb_cores)

        X_train['T'] = y_train['T']
        y_train = y_train.transpose('T', 'Y', 'X')
        X_test = X_test.squeeze()
        y_test = y_test.drop_vars('T').squeeze().transpose('Y', 'X')
        y_train, alpha = xr.align(y_train, alpha)
        y_test, alpha = xr.align(y_test, alpha)

        client = Client(n_workers=self.nb_cores, threads_per_worker=1)
        result = xr.apply_ufunc(
            self.fit_predict,
            X_train,
            y_train.chunk({'Y': chunksize_y, 'X': chunksize_x}),
            X_test,
            y_test.chunk({'Y': chunksize_y, 'X': chunksize_x}),
            alpha.chunk({'Y': chunksize_y, 'X': chunksize_x}),
            input_core_dims=[('T', 'features'), ('T',), ('features',), (), ()],
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
        Calculate tercile probabilities for predictions using Student's t-distribution.

        Parameters:
            best_guess (ndarray): Predicted values.
            error_variance (ndarray): Error variance of predictions.
            first_tercile (float): First tercile value.
            second_tercile (float): Second tercile value.
            dof (int): Degrees of freedom.

        Returns:
            ndarray: Probabilities for below normal, normal, and above normal categories.
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
        Compute probabilities for tercile categories (below normal, normal, above normal) based on the predictions.

        Parameters:
            Predictant (xarray.DataArray): Observed rainfall data.
            clim_year_start (int): Start year for climatology.
            clim_year_end (int): End year for climatology.
            Predictor (xarray.DataArray): Predictor data.
            hindcast_det (xarray.DataArray): Deterministic hindcast predictions.

        Returns:
            xarray.DataArray: Probabilities for each tercile category.
        Examples: ----
        """
        index_start = Predictant.get_index("T").get_loc(str(clim_year_start)).start
        index_end = Predictant.get_index("T").get_loc(str(clim_year_end)).stop
        Predictant, hindcast_det = xr.align(Predictant, hindcast_det)
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

    def forecast(self, Predictant, clim_year_start, clim_year_end, Predictor, hindcast_det, Predictor_for_year, alpha):

        chunksize_x = np.round(len(Predictant.get_index("X")) / self.nb_cores)
        chunksize_y = np.round(len(Predictant.get_index("Y")) / self.nb_cores)
        
        Predictor['T'] = Predictant['T']
        Predictant = Predictant.transpose('T', 'Y', 'X')
        Predictor_for_year_ = Predictor_for_year.squeeze()
        Predictant, alpha = xr.align(Predictant, alpha)

        client = Client(n_workers=self.nb_cores, threads_per_worker=1)
        result = xr.apply_ufunc(
            self.fit_predict,
            Predictor,
            Predictant.chunk({'Y': chunksize_y, 'X': chunksize_x}),
            Predictor_for_year_,
            alpha.chunk({'Y': chunksize_y, 'X': chunksize_x}),
            input_core_dims=[('T', 'features'), ('T',), ('features',),()],
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
        terciles = rainfall_for_tercile.quantile([0.3, 0.66], dim='T')
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


class WAS_Lasso_Model:
    """
    WAS_Lasso_Model is a class that implements Lasso regression with hyperparameter tuning,
    clustering-based regional optimization, and calculation of tercile probabilities for
    climate prediction.

    Attributes:
    -----------
    alpha_range : numpy.array
        Range of alpha values for Lasso regularization parameter.
    n_clusters : int
        Number of clusters to use for regional optimization.
    nb_cores : int
        Number of cores to use for parallel computation.
    
    Methods:
    --------
    fit_predict(x, y, x_test, y_test, alpha):
        Fits the Lasso model to the provided training data and predicts values for the test set.
        Returns the prediction error and predictions.

    compute_hyperparameters(predictand, predictor):
        Performs clustering of the spatial grid and computes optimal alpha values for each cluster.
        Returns the alpha values as an xarray and the cluster assignments.

    compute_model(X_train, y_train, X_test, y_test, alpha):
        Computes the Lasso model prediction for the training and test datasets using the given alpha values.
        Utilizes Dask for parallelized processing.

    calculate_tercile_probabilities(best_guess, error_variance, first_tercile, second_tercile, dof):
        Calculates the tercile probabilities for a given prediction using the t-distribution.
        Static method to compute tercile probabilities for probabilistic forecasting.

    compute_prob(Predictant, clim_year_start, clim_year_end, Predictor, hindcast_det):
        Computes probabilistic forecasts for rainfall terciles based on climatological terciles,
        utilizing a hindcast and Lasso regression output.

    Examples: -----
    """

    def __init__(self, alpha_range=None, n_clusters=5, nb_cores=1):
        if alpha_range is None:
            alpha_range = np.array([10**i for i in range(-6, 6)])
        self.alpha_range = alpha_range
        self.n_clusters = n_clusters
        self.nb_cores = nb_cores
    
    def fit_predict(self, x, y, x_test, y_test, alpha):
        model = linear_model.Lasso(alpha)
        mask = np.isfinite(y) & np.all(np.isfinite(x), axis=-1)
        
        if np.any(mask):
            y_clean = y[mask]
            x_clean = x[mask, :]
            model.fit(x_clean, y_clean)
            
            if x_test.ndim == 1:
                x_test = x_test.reshape(1, -1)
                
            preds = model.predict(x_test)
            preds[preds < 0] = 0
            error_ = y_test - preds
            return np.array([error_, preds]).squeeze()
        else:
            return np.array([np.nan, np.nan]).squeeze()
    
    def compute_hyperparameters(self, predictand, predictor):
        kmeans = KMeans(n_clusters=self.n_clusters)
        predictand_dropna = predictand.to_dataframe().reset_index().dropna().drop(columns=['T'])
        predictand_dropna['cluster'] = kmeans.fit_predict(predictand_dropna[predictand_dropna.columns[2]].to_frame())
        
        df_unique = predictand_dropna.drop_duplicates(subset=['Y', 'X'])
        dataset = df_unique.set_index(['Y', 'X']).to_xarray()
        
        Cluster = (dataset['cluster'] * xr.where(~np.isnan(predictand.isel(T=0)), 1, np.nan)).drop_vars("T")
        xarray1, xarray2 = xr.align(predictand, Cluster)
        
        clusters = np.unique(xarray2)
        clusters = clusters[~np.isnan(clusters)]
        
        cluster_means = {
            int(cluster): xarray1.where(xarray2 == cluster).mean(dim=['Y', 'X'], skipna=True)
            for cluster in clusters
        }
        
        model = linear_model.LassoCV(alphas=self.alpha_range, cv=5)
        alpha_cluster = {
            int(cluster): model.fit(predictor, cluster_means[cluster]).alpha_
            for cluster in clusters
        }
        
        alpha_array = Cluster.copy()
        for key, value in alpha_cluster.items():
            alpha_array = alpha_array.where(alpha_array != key, other=value)
        alpha_array, Cluster, predictand = xr.align(alpha_array, Cluster, predictand, join = "outer")
        return alpha_array, Cluster

    def compute_model(self, X_train, y_train, X_test, y_test, alpha):
        chunksize_x = np.round(len(y_train.get_index("X")) / self.nb_cores)
        chunksize_y = np.round(len(y_train.get_index("Y")) / self.nb_cores)
        
        X_train['T'] = y_train['T']
        y_train = y_train.transpose('T', 'Y', 'X')
        X_test = X_test.squeeze()
        y_test = y_test.drop_vars('T').squeeze().transpose('Y', 'X')
        y_train, alpha =  xr.align(y_train, alpha)
        y_test, alpha =  xr.align(y_test, alpha)
        
        client = Client(n_workers=self.nb_cores, threads_per_worker=1)        
        result = xr.apply_ufunc(
            self.fit_predict,
            X_train,
            y_train.chunk({'Y': chunksize_y, 'X': chunksize_x}),
            X_test,
            y_test.chunk({'Y': chunksize_y, 'X': chunksize_x}),
            alpha.chunk({'Y': chunksize_y, 'X': chunksize_x}),
            input_core_dims=[('T', 'features'), ('T',), ('features',), (), ()],
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
        index_start = Predictant.get_index("T").get_loc(str(clim_year_start)).start
        index_end = Predictant.get_index("T").get_loc(str(clim_year_end)).stop
        Predictant, hindcast_det =  xr.align(Predictant, hindcast_det)
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

    def forecast(self, Predictant, clim_year_start, clim_year_end, Predictor, hindcast_det, Predictor_for_year, alpha):

        chunksize_x = np.round(len(Predictant.get_index("X")) / self.nb_cores)
        chunksize_y = np.round(len(Predictant.get_index("Y")) / self.nb_cores)
        
        Predictor['T'] = Predictant['T']
        Predictant = Predictant.transpose('T', 'Y', 'X')
        Predictor_for_year_ = Predictor_for_year.squeeze()
        Predictant, alpha = xr.align(Predictant, alpha)

        client = Client(n_workers=self.nb_cores, threads_per_worker=1)
        result = xr.apply_ufunc(
            self.fit_predict,
            Predictor,
            Predictant.chunk({'Y': chunksize_y, 'X': chunksize_x}),
            Predictor_for_year_,
            alpha.chunk({'Y': chunksize_y, 'X': chunksize_x}),
            input_core_dims=[('T', 'features'), ('T',), ('features',),()],
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


class WAS_LassoLars_Model:
    """
    A class to implement the Lasso Least Angle Regression (LassoLars) model for spatiotemporal climate prediction.

    This model is designed to work with climate data by clustering spatial regions, computing hyperparameters for each cluster, 
    and fitting a LassoLars model for predictions. The model is optimized for parallel execution using Dask.

    Parameters
    ----------
    alpha_range : array-like, optional
        Range of alpha values for the LassoLars model. Default is `np.array([10**i for i in range(-6, 6)])`.
    n_clusters : int, default=5
        Number of clusters to partition the spatial data.
    nb_cores : int, default=1
        Number of cores for parallel processing.
    
    Methods
    -------
    fit_predict(x, y, x_test, y_test, alpha)
        Fits the LassoLars model to the training data and predicts values for the test data.
    
    compute_hyperparameters(predictand, predictor)
        Computes cluster-wise optimal alpha values for LassoLars using cross-validation.
    
    compute_model(X_train, y_train, X_test, y_test, alpha)
        Fits and predicts the LassoLars model using Dask for parallel execution across spatial data.
    
    calculate_tercile_probabilities(best_guess, error_variance, first_tercile, second_tercile, dof)
        Calculates the tercile probabilities of a prediction based on Student's t-distribution.
    
    compute_prob(Predictant, clim_year_start, clim_year_end, Predictor, hindcast_det)
        Computes the tercile probabilities for hindcast predictions using a climatological period.
    """

    def __init__(self, alpha_range=None, n_clusters=5, nb_cores=1):
        if alpha_range is None:
            alpha_range = np.array([10**i for i in range(-6, 6)])
        self.alpha_range = alpha_range
        self.n_clusters = n_clusters
        self.nb_cores = nb_cores
    
    def fit_predict(self, x, y, x_test, y_test, alpha):
        """
        Fits the LassoLars model to the training data and predicts values for the test data.

        Parameters
        ----------
        x : array-like
            Training predictors.
        y : array-like
            Training response variable.
        x_test : array-like
            Test predictors.
        y_test : array-like
            Test response variable.
        alpha : float
            Regularization strength parameter for LassoLars.

        Returns
        -------
        array
            Array containing the prediction error and predicted values.
        """
        model = linear_model.LassoLars(alpha)
        mask = np.isfinite(y) & np.all(np.isfinite(x), axis=-1)
        
        if np.any(mask):
            y_clean = y[mask]
            x_clean = x[mask, :]
            model.fit(x_clean, y_clean)
            
            if x_test.ndim == 1:
                x_test = x_test.reshape(1, -1)
                
            preds = model.predict(x_test)
            preds[preds < 0] = 0
            error_ = y_test - preds
            return np.array([error_, preds]).squeeze()
        else:
            return np.array([np.nan, np.nan]).squeeze()
    
    def compute_hyperparameters(self, predictand, predictor):
        """
        Computes cluster-wise optimal alpha values for LassoLars using cross-validation.

        Parameters
        ----------
        predictand : xarray.DataArray
            The response variable for clustering and training.
        predictor : array-like
            Predictor variables used for fitting the model.

        Returns
        -------
        alpha_array : xarray.DataArray
            Cluster-wise optimal alpha values.
        Cluster : xarray.DataArray
            Cluster assignment for each spatial point.
        """
        kmeans = KMeans(n_clusters=self.n_clusters)
        predictand_dropna = predictand.to_dataframe().reset_index().dropna().drop(columns=['T'])
        predictand_dropna['cluster'] = kmeans.fit_predict(predictand_dropna[predictand_dropna.columns[2]].to_frame())
        
        df_unique = predictand_dropna.drop_duplicates(subset=['Y', 'X'])
        dataset = df_unique.set_index(['Y', 'X']).to_xarray()
        
        Cluster = (dataset['cluster'] * xr.where(~np.isnan(predictand.isel(T=0)), 1, np.nan)).drop_vars("T")
        xarray1, xarray2 = xr.align(predictand, Cluster)
        clusters = np.unique(xarray2)
        clusters = clusters[~np.isnan(clusters)]
        cluster_means = {
            int(cluster): xarray1.where(xarray2 == cluster).mean(dim=['Y', 'X'], skipna=True)
            for cluster in clusters
        }
        
        model = linear_model.LassoLarsCV()
        alpha_cluster = {
            int(cluster): model.fit(predictor, cluster_means[cluster]).alpha_
            for cluster in clusters
        }
        alpha_array = Cluster.copy()
        for key, value in alpha_cluster.items():
            alpha_array = alpha_array.where(alpha_array != key, other=value)
        alpha_array, Cluster, predictand = xr.align(alpha_array, Cluster, predictand, join = "outer")
        return alpha_array, Cluster

    def compute_model(self, X_train, y_train, X_test, y_test, alpha):
        """
        Fits and predicts the LassoLars model using Dask for parallel execution across spatial data.

        Parameters
        ----------
        X_train : xarray.DataArray
            Training predictor data.
        y_train : xarray.DataArray
            Training response variable.
        X_test : xarray.DataArray
            Test predictor data.
        y_test : xarray.DataArray
            Test response variable.
        alpha : xarray.DataArray
            Cluster-wise optimal alpha values for LassoLars.

        Returns
        -------
        xarray.DataArray
            Model prediction and error across the spatial domain.
        """
        chunksize_x = np.round(len(y_train.get_index("X")) / self.nb_cores)
        chunksize_y = np.round(len(y_train.get_index("Y")) / self.nb_cores)
        
        X_train['T'] = y_train['T']
        y_train = y_train.transpose('T', 'Y', 'X')
        X_test = X_test.squeeze()
        y_test = y_test.drop_vars('T').squeeze().transpose('Y', 'X')
        y_train, alpha =  xr.align(y_train, alpha)
        y_test, alpha =  xr.align(y_test, alpha)
        client = Client(n_workers=self.nb_cores, threads_per_worker=1)        
        result = xr.apply_ufunc(
            self.fit_predict,
            X_train,
            y_train.chunk({'Y': chunksize_y, 'X': chunksize_x}),
            X_test,
            y_test.chunk({'Y': chunksize_y, 'X': chunksize_x}),
            alpha.chunk({'Y': chunksize_y, 'X': chunksize_x}),
            input_core_dims=[('T', 'features'), ('T',), ('features',), (), ()],
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
        Calculates the tercile probabilities of a prediction based on Student's t-distribution.

        Parameters
        ----------
        best_guess : array-like
            Predicted values.
        error_variance : array-like
            Variance of prediction error.
        first_tercile : float
            First tercile value.
        second_tercile : float
            Second tercile value.
        dof : int
            Degrees of freedom for Student's t-distribution.

        Returns
        -------
        array
            Probabilities for each tercile category.
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
        Computes the tercile probabilities for hindcast predictions using a climatological period.

        Parameters
        ----------
        Predictant : xarray.DataArray
            The observed variable to be predicted.
        clim_year_start : int
            Start year of the climatological period.
        clim_year_end : int
            End year of the climatological period.
        Predictor : xarray.DataArray
            Predictor data used for training.
        hindcast_det : xarray.DataArray
            Deterministic hindcast predictions.

        Returns
        -------
        xarray.DataArray
            Probabilities for each tercile category across the spatial domain.
        """
        index_start = Predictant.get_index("T").get_loc(str(clim_year_start)).start
        index_end = Predictant.get_index("T").get_loc(str(clim_year_end)).stop
        Predictant, hindcast_det =  xr.align(Predictant, hindcast_det)
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

    def forecast(self, Predictant, clim_year_start, clim_year_end, Predictor, hindcast_det, Predictor_for_year, alpha):

        chunksize_x = np.round(len(Predictant.get_index("X")) / self.nb_cores)
        chunksize_y = np.round(len(Predictant.get_index("Y")) / self.nb_cores)
        
        Predictor['T'] = Predictant['T']
        Predictant = Predictant.transpose('T', 'Y', 'X')
        Predictor_for_year_ = Predictor_for_year.squeeze()
        Predictant, alpha = xr.align(Predictant, alpha)

        client = Client(n_workers=self.nb_cores, threads_per_worker=1)
        result = xr.apply_ufunc(
            self.fit_predict,
            Predictor,
            Predictant.chunk({'Y': chunksize_y, 'X': chunksize_x}),
            Predictor_for_year_,
            alpha.chunk({'Y': chunksize_y, 'X': chunksize_x}),
            input_core_dims=[('T', 'features'), ('T',), ('features',),()],
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

class WAS_ElasticNet_Model:
    """
    A class to implement the ElasticNet model for spatiotemporal regression with clustering, cross-validation, and
    probabilistic predictions.

    Attributes:
    -----------
    alpha_range : numpy.ndarray
        Range of alpha values (regularization strength) to search for optimal hyperparameters. Default is from 10^-6 to 10^2.
    l1_ratio_range : list
        Range of l1 ratio values (mixing between L1 and L2 penalties) to search for optimal hyperparameters. Default values are [0.1, 0.5, 0.7, 0.9, 0.95, 0.99, 1].
    n_clusters : int
        Number of clusters to be used for clustering spatial regions. Default is 5.
    nb_cores : int
        Number of cores for parallel computations. Default is 1.

    Methods:
    --------
    fit_predict(x, y, x_test, y_test, alpha, l1_ratio):
        Fits an ElasticNet model to the training data and predicts values for the test data.
    
    compute_hyperparameters(predictand, predictor):
        Computes the optimal alpha and l1 ratio hyperparameters for each cluster using cross-validation.
    
    compute_model(X_train, y_train, X_test, y_test, alpha, l1_ratio):
        Performs parallelized ElasticNet modeling for gridded spatiotemporal data.
    
    calculate_tercile_probabilities(best_guess, error_variance, first_tercile, second_tercile, dof):
        Calculates the tercile probabilities based on predictions, error variance, and given terciles.
    
    compute_prob(Predictant, clim_year_start, clim_year_end, Predictor, hindcast_det):
        Computes probabilistic hindcasts for tercile categories based on climatological terciles.
    """

    def __init__(self, alpha_range=None, l1_ratio_range=None, n_clusters=5, nb_cores=1):
        if alpha_range is None:
            alpha_range = np.array([10**i for i in range(-6, 3)])
        if l1_ratio_range is None:
            l1_ratio_range = [.1, .5, .7, .9, .95, .99, 1]
        self.alpha_range = alpha_range
        self.l1_ratio_range = l1_ratio_range
        self.n_clusters = n_clusters
        self.nb_cores = nb_cores
    
    def fit_predict(self, x, y, x_test, y_test=None, alpha=None, l1_ratio=None):
        model = linear_model.ElasticNet(alpha=alpha, l1_ratio=l1_ratio)
        mask = np.isfinite(y) & np.all(np.isfinite(x), axis=-1)
        
        if y_test is not None:
        
            if np.any(mask):
                y_clean = y[mask]
                x_clean = x[mask, :]
                model.fit(x_clean, y_clean)
                
                if x_test.ndim == 1:
                    x_test = x_test.reshape(1, -1)
                
                preds = model.predict(x_test)
                preds[preds < 0] = 0
                error_ = y_test - preds
                return np.array([error_, preds]).squeeze()
            else:
                return np.array([np.nan, np.nan]).squeeze()
        else:
            if np.any(mask):
                y_clean = y[mask]
                x_clean = x[mask, :]
                model.fit(x_clean, y_clean)
                
                if x_test.ndim == 1:
                    x_test = x_test.reshape(1, -1)
                
                preds = model.predict(x_test)
                preds[preds < 0] = 0
                return np.array([preds]).squeeze()
            else:
                return np.array([np.nan]).squeeze()
    
    def compute_hyperparameters(self, predictand, predictor):
        kmeans = KMeans(n_clusters=self.n_clusters)
        predictand_dropna = predictand.to_dataframe().reset_index().dropna().drop(columns=['T'])
        predictand_dropna['cluster'] = kmeans.fit_predict(predictand_dropna[predictand_dropna.columns[2]].to_frame())
        
        df_unique = predictand_dropna.drop_duplicates(subset=['Y', 'X'])
        dataset = df_unique.set_index(['Y', 'X']).to_xarray()
        
        Cluster = (dataset['cluster'] * xr.where(~np.isnan(predictand.isel(T=0)), 1, np.nan)).drop_vars("T")
        xarray1, xarray2 = xr.align(predictand, Cluster)
        clusters = np.unique(xarray2)
        clusters = clusters[~np.isnan(clusters)]
        cluster_means = {
            int(cluster): xarray1.where(xarray2 == cluster).mean(dim=['Y', 'X'], skipna=True)
            for cluster in clusters
        }
        
        model = linear_model.ElasticNetCV(alphas=self.alpha_range, l1_ratio=self.l1_ratio_range, cv=5)

        alpha_cluster = {
            int(cluster): [model.fit(predictor, cluster_means[cluster]).alpha_, model.fit(predictor, cluster_means[cluster]).l1_ratio_]
            for cluster in clusters
            }
        alpha_array = Cluster.copy()
        l1_ratio_array = Cluster.copy()
    
        for key, value in alpha_cluster.items():
            alpha_array = alpha_array.where(alpha_array != key, other=value[0]) 
            l1_ratio_array = l1_ratio_array.where(l1_ratio_array != key, other=value[1]) 

        alpha_array, l1_ratio_array, Cluster, predictand = xr.align(alpha_array, l1_ratio_array, Cluster, predictand, join="outer")       
        return alpha_array, l1_ratio_array, Cluster

    def compute_model(self, X_train, y_train, X_test, y_test, alpha, l1_ratio):
        chunksize_x = np.round(len(y_train.get_index("X")) / self.nb_cores)
        chunksize_y = np.round(len(y_train.get_index("Y")) / self.nb_cores)
        
        X_train['T'] = y_train['T']
        y_train = y_train.transpose('T', 'Y', 'X')
        X_test = X_test.squeeze()
        y_test = y_test.drop_vars('T').squeeze().transpose('Y', 'X')
        y_train, alpha =  xr.align(y_train, alpha)
        y_test, alpha =  xr.align(y_test, alpha)
        l1_ratio, alpha =  xr.align(l1_ratio, alpha)
        # alpha = alpha.transpose('Y', 'X')
        client = Client(n_workers=self.nb_cores, threads_per_worker=1)        
        result = xr.apply_ufunc(
            self.fit_predict,
            X_train,
            y_train.chunk({'Y': chunksize_y, 'X': chunksize_x}),
            X_test,
            y_test.chunk({'Y': chunksize_y, 'X': chunksize_x}),
            alpha.chunk({'Y': chunksize_y, 'X': chunksize_x}),
            l1_ratio.chunk({'Y': chunksize_y, 'X': chunksize_x}),
            input_core_dims=[('T', 'features'), ('T',), ('features',), (), (), ()],
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
    def calculate_tercile_probabilities_gamma(best_guess, error_variance, T1, T2, dof):

        n_time = len(best_guess)
        pred_prob = np.empty((3, n_time), dtype=float)
    
        # If all best_guess are NaN, just fill everything with NaN
        if np.any(np.isnan(best_guess)) or np.any(np.isnan(error_variance)):
            pred_prob[:] = np.nan
            return pred_prob
    
        # Convert inputs to arrays (in case they're lists)
        best_guess = np.asarray(best_guess, dtype=float)
        error_variance = np.asarray(error_variance, dtype=float)
        T1 = np.asarray(T1, dtype=float)
        T2 = np.asarray(T2, dtype=float)
    
        # Calculate shape (alpha) and scale (theta) for the Gamma distribution
        # alpha = (mean^2) / variance
        # theta = variance / mean
        alpha = (best_guess**2) / error_variance
        theta = error_variance / best_guess
    
        # Compute CDF at T1, T2 (no loop over n_time)
        cdf_t1 = gamma.cdf(T1, a=alpha, scale=theta)  # P(X < T1)
        cdf_t2 = gamma.cdf(T2, a=alpha, scale=theta)  # P(X < T2)
    
        # Fill out the probabilities
        pred_prob[0, :] = cdf_t1
        pred_prob[1, :] = cdf_t2 - cdf_t1
        pred_prob[2, :] = 1.0 - cdf_t2

        dof=dof
    
        return pred_prob    
    
    @staticmethod
    def calculate_tercile_probabilities(best_guess, error_variance, first_tercile, second_tercile, dof):
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
        index_start = Predictant.get_index("T").get_loc(str(clim_year_start)).start
        index_end = Predictant.get_index("T").get_loc(str(clim_year_end)).stop
        Predictant, hindcast_det =  xr.align(Predictant, hindcast_det)
        rainfall_for_tercile = Predictant.isel(T=slice(index_start, index_end))
        terciles = rainfall_for_tercile.quantile([0.3, 0.6666667], dim='T')
        error_variance = hindcast_det.sel(output="error").drop_vars("output").squeeze().var(dim='T')
        
        dof = len(Predictant.get_index("T")) - 1 - (len(Predictor.get_index("features")) + 1)
        
        hindcast_prob = xr.apply_ufunc(
            self.calculate_tercile_probabilities_gamma,
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

    def forecast(self, Predictant, clim_year_start, clim_year_end, Predictor, hindcast_det, Predictor_for_year, alpha, l1_ratio):

        chunksize_x = np.round(len(Predictant.get_index("X")) / self.nb_cores)
        chunksize_y = np.round(len(Predictant.get_index("Y")) / self.nb_cores)
        
        Predictor['T'] = Predictant['T']
        Predictant = Predictant.transpose('T', 'Y', 'X')
        Predictor_for_year_ = Predictor_for_year.squeeze()
        Predictant, alpha, l1_ratio = xr.align(Predictant, alpha, l1_ratio, join="outer")
        y_test = Predictant.isel(T=0)
        
        client = Client(n_workers=self.nb_cores, threads_per_worker=1)
        result = xr.apply_ufunc(
            self.fit_predict,
            Predictor,
            Predictant.chunk({'Y': chunksize_y, 'X': chunksize_x}),
            Predictor_for_year_,
            y_test.chunk({'Y': chunksize_y, 'X': chunksize_x}),
            alpha.chunk({'Y': chunksize_y, 'X': chunksize_x}),
            l1_ratio.chunk({'Y': chunksize_y, 'X': chunksize_x}),
            input_core_dims=[('T', 'features'), ('T',), ('features',),(),(),()],
            vectorize=True,
            output_core_dims=[('output',)],
            dask='parallelized',
            output_dtypes=['float'],
            dask_gufunc_kwargs={'output_sizes': {'output': 2}},
        )
        result_ = result.compute()
        client.close()
        result_ = result_.isel(output=1)
        
        index_start = Predictant.get_index("T").get_loc(str(clim_year_start)).start
        index_end = Predictant.get_index("T").get_loc(str(clim_year_end)).stop
        rainfall_for_tercile = Predictant.isel(T=slice(index_start, index_end))
        terciles = rainfall_for_tercile.quantile([0.3, 0.66], dim='T')
        error_variance = hindcast_det.sel(output="error").drop_vars("output").squeeze().var(dim='T')
        dof = len(Predictant.get_index("T")) - 1 - (len(Predictor.get_index("features")) + 1)
        terciles, result_ =  xr.align(terciles, result_)
        error_variance, terciles =  xr.align(error_variance, terciles)
        
        
        hindcast_prob = xr.apply_ufunc(
            self.calculate_tercile_probabilities,
            result_.expand_dims({'T':[0]},axis=0),#expand_dims({'T': [pd.Timestamp(Predictor_for_year.isel(T=[-1]).coords['T'].values.item()).to_pydatetime()]}),
            error_variance,
            terciles.isel(quantile=0).drop_vars('quantile'),
            terciles.isel(quantile=1).drop_vars('quantile'),
            input_core_dims=[('T'), (), (), ()],
            vectorize=True,
            kwargs={'dof': dof},
            dask='parallelized',
            output_core_dims=[('probability','T',)],
            output_dtypes=['float'],
            dask_gufunc_kwargs={'output_sizes': {'probability': 3}},
        )
        hindcast_prob = hindcast_prob.assign_coords(probability=('probability', ['PB', 'PN', 'PA']))        
        return result_, hindcast_prob.squeeze().transpose('probability', 'Y', 'X') 