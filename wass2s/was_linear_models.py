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
from scipy.stats import norm
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
    dist_method : str, optional
        Distribution method for tercile probability calculations. One of
        {"t","gamma","normal","lognormal","nonparam"}. Default = "gamma".

    Methods
    -------
    fit_predict(x, y, x_test, y_test=None)
        Fits a linear regression model, makes predictions, and calculates error if y_test is provided.

    compute_model(X_train, y_train, X_test, y_test)
        Applies the linear regression model across a dataset using parallel computation with Dask.

    compute_prob(Predictant, clim_year_start, clim_year_end, Predictor, hindcast_det)
        Computes tercile probabilities for hindcast predictions over specified years.

    forecast(Predictant, clim_year_start, clim_year_end, Predictor, hindcast_det, Predictor_for_year)
        Generates a single-year forecast and computes tercile probabilities.
    """

    def __init__(self, nb_cores=1, dist_method="gamma"):
        """
        Initializes the WAS_LinearRegression_Model with a specified number of CPU cores.
        
        Parameters
        ----------
        nb_cores : int, optional
            Number of CPU cores to use for parallel computation, by default 1.
        dist_method : str, optional
            Distribution method to compute tercile probabilities, by default "gamma".
        """
        self.nb_cores = nb_cores
        self.dist_method = dist_method
    
    def fit_predict(self, x, y, x_test, y_test=None):
        """
        Fits a linear regression model to the provided training data, makes predictions 
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
        model = linear_model.LinearRegression()
        mask = np.isfinite(y) & np.all(np.isfinite(x), axis=-1)

        if np.any(mask):
            y_clean = y[mask]
            x_clean = x[mask, :]
            model.fit(x_clean, y_clean)

            if x_test.ndim == 1:
                x_test = x_test.reshape(1, -1)

            preds = model.predict(x_test)
            preds[preds < 0] = 0  # clip negative if modeling precipitation

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
        Applies linear regression across a spatiotemporal dataset in parallel.

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
    
    # --------------------------------------------------------------------------
    #  Probability Calculation Methods
    # --------------------------------------------------------------------------
    @staticmethod
    def calculate_tercile_probabilities(best_guess, error_variance, first_tercile, second_tercile, dof):
        """
        Student's t-based method.
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
    def calculate_tercile_probabilities_gamma(best_guess, error_variance, T1, T2):
        """
        Gamma-based method.
        """
        n_time = len(best_guess)
        pred_prob = np.empty((3, n_time), dtype=float)

        if np.any(np.isnan(best_guess)) or np.any(np.isnan(error_variance)):
            pred_prob[:] = np.nan
            return pred_prob

        best_guess = np.asarray(best_guess, dtype=float)
        error_variance = np.asarray(error_variance, dtype=float)
        T1 = np.asarray(T1, dtype=float)
        T2 = np.asarray(T2, dtype=float)

        alpha = (best_guess**2) / error_variance
        theta = error_variance / best_guess

        cdf_t1 = gamma.cdf(T1, a=alpha, scale=theta)
        cdf_t2 = gamma.cdf(T2, a=alpha, scale=theta)
        pred_prob[0, :] = cdf_t1
        pred_prob[1, :] = cdf_t2 - cdf_t1
        pred_prob[2, :] = 1.0 - cdf_t2
        return pred_prob

    @staticmethod
    def calculate_tercile_probabilities_nonparametric(best_guess, error_samples, first_tercile, second_tercile):
        """
        Non-parametric method (requires historical errors).
        """
        n_time = len(best_guess)
        pred_prob = np.full((3, n_time), np.nan, dtype=float)

        for t in range(n_time):
            if np.isnan(best_guess[t]):
                continue

            dist = best_guess[t] + error_samples  
            dist = dist[np.isfinite(dist)]  
            if len(dist) == 0:
                continue

            p_below   = np.mean(dist < first_tercile)
            p_between = np.mean((dist >= first_tercile) & (dist < second_tercile))
            p_above   = 1.0 - (p_below + p_between)

            pred_prob[0, t] = p_below
            pred_prob[1, t] = p_between
            pred_prob[2, t] = p_above
        return pred_prob

    @staticmethod
    def calculate_tercile_probabilities_normal(best_guess, error_variance, first_tercile, second_tercile):
        """
        Normal-based method using the Gaussian CDF.
        """
        n_time = len(best_guess)
        pred_prob = np.empty((3, n_time))
        
        if np.all(np.isnan(best_guess)):
            pred_prob[:] = np.nan
        else:
            error_std = np.sqrt(error_variance)
            pred_prob[0, :] = stats.norm.cdf(first_tercile, loc=best_guess, scale=error_std)
            pred_prob[1, :] = stats.norm.cdf(second_tercile, loc=best_guess, scale=error_std) - \
                              stats.norm.cdf(first_tercile, loc=best_guess, scale=error_std)
            pred_prob[2, :] = 1 - stats.norm.cdf(second_tercile, loc=best_guess, scale=error_std)
        return pred_prob

    @staticmethod
    def calculate_tercile_probabilities_lognormal(best_guess, error_variance, first_tercile, second_tercile):
        """
        Lognormal-based method.
        """
        n_time = len(best_guess)
        pred_prob = np.empty((3, n_time))
        
        if np.any(np.isnan(best_guess)) or np.any(np.isnan(error_variance)):
            pred_prob[:] = np.nan
            return pred_prob

        sigma = np.sqrt(np.log(1 + error_variance / (best_guess**2)))
        mu = np.log(best_guess) - sigma**2 / 2
        pred_prob[0, :] = lognorm.cdf(first_tercile, s=sigma, scale=np.exp(mu))
        pred_prob[1, :] = lognorm.cdf(second_tercile, s=sigma, scale=np.exp(mu)) - \
                          lognorm.cdf(first_tercile, s=sigma, scale=np.exp(mu))
        pred_prob[2, :] = 1 - lognorm.cdf(second_tercile, s=sigma, scale=np.exp(mu))
        return pred_prob

    def compute_prob(self, Predictant, clim_year_start, clim_year_end, hindcast_det):
        """
        Compute tercile probabilities using self.dist_method.

        Parameters
        ----------
        Predictant : xarray.DataArray (T, Y, X)
            Observed data.
        clim_year_start : int
        clim_year_end : int
            The start and end years for the climatology.
        hindcast_det : xarray.DataArray
            Deterministic forecast with dims (output=2, T, Y, X).

        Returns
        -------
        hindcast_prob : xarray.DataArray
            dims (probability=3, T, Y, X) => [PB, PN, PA].
        """
        # 1) Identify climatology slice
        index_start = Predictant.get_index("T").get_loc(str(clim_year_start)).start
        index_end   = Predictant.get_index("T").get_loc(str(clim_year_end)).stop
        rainfall_for_tercile = Predictant.isel(T=slice(index_start, index_end))
        terciles = rainfall_for_tercile.quantile([0.33, 0.67], dim='T')
        error_variance = (Predictant - hindcast_det).var(dim='T')

        T1 = terciles.isel(quantile=0).drop_vars('quantile')
        T2 = terciles.isel(quantile=1).drop_vars('quantile')

        # 2) Distinguish distribution method
        if self.dist_method == "t":
            dof = len(Predictant.get_index("T")) - 2
            calc_func = self.calculate_tercile_probabilities
            hindcast_prob = xr.apply_ufunc(
                calc_func,
                hindcast_det,
                error_variance,
                T1,
                T2,
                input_core_dims=[('T',), (), (), ()],
                vectorize=True,
                kwargs={'dof': dof},
                dask='parallelized',
                output_core_dims=[('probability','T')],
                output_dtypes=['float'],
                dask_gufunc_kwargs={'output_sizes': {'probability': 3}, "allow_rechunk": True},
            )

        elif self.dist_method == "gamma":
            calc_func = self.calculate_tercile_probabilities_gamma
            hindcast_prob = xr.apply_ufunc(
                calc_func,
                hindcast_det,
                error_variance,
                T1,
                T2,
                input_core_dims=[('T',), (), (), ()],
                vectorize=True,
                dask='parallelized',
                output_core_dims=[('probability','T')],
                output_dtypes=['float'],
                dask_gufunc_kwargs={'output_sizes': {'probability': 3}, "allow_rechunk":True},
            )

        elif self.dist_method == "normal":
            calc_func = self.calculate_tercile_probabilities_normal
            hindcast_prob = xr.apply_ufunc(
                calc_func,
                hindcast_det,
                error_variance,
                T1,
                T2,
                input_core_dims=[('T',), (), (), ()],
                vectorize=True,
                dask='parallelized',
                output_core_dims=[('probability','T')],
                output_dtypes=['float'],
                dask_gufunc_kwargs={'output_sizes': {'probability': 3}, "allow_rechunk":True},
            )

        elif self.dist_method == "lognormal":
            calc_func = self.calculate_tercile_probabilities_lognormal
            hindcast_prob = xr.apply_ufunc(
                calc_func,
                hindcast_det,
                error_variance,
                T1,
                T2,
                input_core_dims=[('T',), (), (), ()],
                vectorize=True,
                dask='parallelized',
                output_core_dims=[('probability','T')],
                output_dtypes=['float'],
                dask_gufunc_kwargs={'output_sizes': {'probability': 3}, "allow_rechunk":True},
            )

        elif self.dist_method == "nonparam":
            calc_func = self.calculate_tercile_probabilities_nonparametric
            error_samples = (Predictant - hindcast_det)
            hindcast_prob = xr.apply_ufunc(
                calc_func,
                hindcast_det,
                error_samples,
                T1,
                T2,
                input_core_dims=[('T',), ('T',), (), ()],
                output_core_dims=[('probability','T')],
                vectorize=True,
                dask='parallelized',
                output_dtypes=[float],
                dask_gufunc_kwargs={'output_sizes': {'probability': 3}},
            )

        else:
            raise ValueError(f"Invalid dist_method: {self.dist_method}. "
                             "Must be one of ['t','gamma','normal','lognormal','nonparam'].")

        hindcast_prob = hindcast_prob.assign_coords(probability=('probability', ['PB','PN','PA']))
        return hindcast_prob.transpose('probability','T','Y','X')

    # --------------------------------------------------------------------------
    #  FORECAST METHOD
    # --------------------------------------------------------------------------
    def forecast(self, Predictant, clim_year_start, clim_year_end, Predictor, hindcast_det, Predictor_for_year):
        """
        Generates a single-year forecast using linear regression, then computes 
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
        terciles = rainfall_for_tercile.quantile([0.33, 0.67], dim='T')
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

        # 3) Tercile probabilities
        if self.dist_method == "t":
            calc_func = self.calculate_tercile_probabilities
            dof = len(Predictant.get_index("T")) - 2


            hindcast_prob = xr.apply_ufunc(
                calc_func,
                forecast_expanded,
                error_variance,
                T1,
                T2,
                input_core_dims=[('T',), (), (), ()],
                vectorize=True,
                kwargs={'dof': dof},
                dask='parallelized',
                output_core_dims=[('probability','T')],
                output_dtypes=['float'],
                dask_gufunc_kwargs={'output_sizes': {'probability': 3}, "allow_rechunk": True},
            )

        elif self.dist_method == "gamma":
            calc_func = self.calculate_tercile_probabilities_gamma

            hindcast_prob = xr.apply_ufunc(
                calc_func,
                forecast_expanded,
                error_variance,
                T1,
                T2,
                input_core_dims=[('T',), (), (), ()],
                vectorize=True,
                dask='parallelized',
                output_core_dims=[('probability','T')],
                output_dtypes=['float'],
                dask_gufunc_kwargs={'output_sizes': {'probability': 3}}
            )

        elif self.dist_method == "normal":
            calc_func = self.calculate_tercile_probabilities_normal

            hindcast_prob = xr.apply_ufunc(
                calc_func,
                forecast_expanded,
                error_variance,
                T1,
                T2,
                input_core_dims=[('T',), (), (), ()],
                vectorize=True,
                dask='parallelized',
                output_core_dims=[('probability','T')],
                output_dtypes=['float'],
                dask_gufunc_kwargs={'output_sizes': {'probability': 3}}
            )

        elif self.dist_method == "lognormal":
            calc_func = self.calculate_tercile_probabilities_lognormal

            hindcast_prob = xr.apply_ufunc(
                calc_func,
                forecast_expanded,
                error_variance,
                T1,
                T2,
                input_core_dims=[('T',), (), (), ()],
                vectorize=True,
                dask='parallelized',
                output_core_dims=[('probability','T')],
                output_dtypes=['float'],
                dask_gufunc_kwargs={'output_sizes': {'probability': 3}}
            )

        elif self.dist_method == "nonparam":
            calc_func = self.calculate_tercile_probabilities_nonparametric
            error_samples = Predictant - hindcast_det
            error_samples = error_samples.rename({'T':'S'})
            hindcast_prob = xr.apply_ufunc(
                calc_func,
                forecast_expanded,
                error_samples,
                T1,
                T2,
                input_core_dims=[('T',), ('S',), (), ()],
                output_core_dims=[('probability','T')],
                vectorize=True, 
                dask='parallelized',
                output_dtypes=[float],
                dask_gufunc_kwargs={'output_sizes': {'probability': 3}}
            )

        else:
            raise ValueError(
                f"Invalid dist_method: {self.dist_method}. "
                "Choose 't','gamma','normal','lognormal','nonparam'."
            )

        hindcast_prob = hindcast_prob.assign_coords(probability=('probability', ['PB','PN','PA']))
        hindcast_prob_out = hindcast_prob.transpose('probability','T','Y','X') #.drop_vars('T').squeeze()

        # Return [error, prediction] plus tercile probabilities
        return forecast_expanded, hindcast_prob_out

      
    

class WAS_Ridge_Model:
    """
    A class to perform ridge regression modeling for rainfall prediction with spatial clustering 
    and hyperparameter optimization. By Mandela HOUNGNIBO.

    Attributes
    ----------
    alpha_range : array-like
        Range of alpha values to explore for ridge regression.
    n_clusters : int
        Number of clusters for KMeans clustering.
    nb_cores : int
        Number of cores to use for parallel computation.
    dist_method : str
        Distribution method for tercile probability calculations:
        One of {'gamma', 't', 'normal', 'lognormal', 'nonparam'} (default = 'gamma').

    Methods
    -------
    fit_predict(x, y, x_test, y_test, alpha)
        Fits a Ridge regression model using the provided data and makes predictions.

    compute_hyperparameters(predictand, predictor)
        Computes optimal ridge hyperparameters (alpha values) for different clusters.

    compute_model(X_train, y_train, X_test, y_test, alpha)
        Fits and predicts a Ridge model for spatiotemporal data using Dask for parallel computation.

    compute_prob(Predictant, clim_year_start, clim_year_end, Predictor, hindcast_det)
        Computes the probabilities of tercile categories for rainfall prediction using the chosen
        distribution method.

    forecast(Predictant, clim_year_start, clim_year_end, Predictor, hindcast_det, Predictor_for_year, alpha)
        Generates a forecast for a single future time (e.g., year) and computes tercile probabilities.
    """

    def __init__(self, alpha_range=None, n_clusters=5, nb_cores=1, dist_method="gamma"):
        """
        Parameters
        ----------
        alpha_range : array-like, optional
            Range of alpha values to explore for ridge regression.
            Defaults to np.logspace(-10, 10, 100).
        n_clusters : int, optional
            Number of clusters for KMeans (default = 5).
        nb_cores : int, optional
            Number of cores to use for parallel computation (default = 1).
        dist_method : str, optional
            Distribution method for tercile probability calculations
            (default = 'gamma').
        """
        if alpha_range is None:
            alpha_range = np.logspace(-10, 10, 100)
        self.alpha_range = alpha_range
        self.n_clusters = n_clusters
        self.nb_cores = nb_cores
        self.dist_method = dist_method

    def fit_predict(self, x, y, x_test, y_test, alpha):
        """
        Fit a ridge regression model and make predictions.

        Parameters
        ----------
        x : ndarray
            Training data (shape = [n_samples, n_features]).
        y : ndarray
            Target values for training data (shape = [n_samples,]).
        x_test : ndarray
            Test data (shape = [n_features,] or [1, n_features]).
        y_test : float
            Target value for test data.
        alpha : float
            Regularization strength for Ridge regression.

        Returns
        -------
        ndarray
            [error, prediction].
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
            # If no valid training data, return NaNs
            return np.array([np.nan, np.nan]).squeeze()

    def compute_hyperparameters(self, predictand, predictor):
        """
        Compute optimal hyperparameters (alpha) for ridge regression for different clusters.

        Parameters
        ----------
        predictand : xarray.DataArray
            Predictand data for clustering (dims: T, Y, X).
        predictor : xarray.DataArray or ndarray
            Predictor data for model fitting (dims: T, features).

        Returns
        -------
        alpha_array : xarray.DataArray
            Spatial map of alpha values for each grid cell.
        cluster_da : xarray.DataArray
            Cluster assignment for each (Y,X).
        """
        predictor['T'] = predictand['T']
        # (a) KMeans clustering on mean predictand
        kmeans = KMeans(n_clusters=self.n_clusters)
        predictand_dropna = (
            predictand.to_dataframe()
            .reset_index()
            .dropna()
            .drop(columns=['T'])
        )
        # Assign cluster labels
        col_name = predictand_dropna.columns[2]
        predictand_dropna['cluster'] = kmeans.fit_predict(
            predictand_dropna[col_name].to_frame()
        )

        # Convert clusters back to xarray
        df_unique = predictand_dropna.drop_duplicates(subset=['Y', 'X'])
        dataset = df_unique.set_index(['Y', 'X']).to_xarray()

        # Mask out invalid cells
        cluster_da = (dataset['cluster'] *
                      xr.where(~np.isnan(predictand.isel(T=0)), 1, np.nan)
                     ).drop_vars("T", errors='ignore')

        # Align with original predictand
        xarray1, xarray2 = xr.align(predictand, cluster_da)
        clusters = np.unique(xarray2)
        clusters = clusters[~np.isnan(clusters)]

        # (b) For each cluster, get mean time series, do RidgeCV
        cluster_means = {
            int(cluster): xarray1.where(xarray2 == cluster).mean(dim=['Y','X'], skipna=True)
            for cluster in clusters
        }
        model_cv = linear_model.RidgeCV(alphas=self.alpha_range, cv=5)
        alpha_cluster = {}

        # Assume 'predictor' is shape (T, features) or similar
        for clus in clusters:
            c_val = int(clus)
            y_cluster = cluster_means[c_val].dropna(dim='T')
            # If no data, skip
            if len(y_cluster['T']) == 0:
                continue
            predictor_cluster = predictor.sel(T=y_cluster['T'])
            X_mat = predictor_cluster.values
            y_vec = y_cluster.values
            model_cv.fit(X_mat, y_vec)
            alpha_cluster[c_val] = model_cv.alpha_

        # (c) Create alpha_array
        alpha_array = cluster_da.copy()
        for key, val in alpha_cluster.items():
            alpha_array = alpha_array.where(alpha_array != key, other=val)

        # Align
        alpha_array, cluster_da, predictand = xr.align(
            alpha_array, cluster_da, predictand, join="outer"
        )
        return alpha_array, cluster_da

    def compute_model(self, X_train, y_train, X_test, y_test, alpha):
        """
        Fit and predict ridge regression model for spatiotemporal data using Dask for parallel computation.

        Parameters
        ----------
        X_train : xarray.DataArray
            Training predictor data (dims: T, features).
        y_train : xarray.DataArray
            Training predictand data (dims: T, Y, X).
        X_test : xarray.DataArray
            Test predictor data (dims: features) or broadcastable.
        y_test : xarray.DataArray
            Test predictand data (dims: Y, X).
        alpha : xarray.DataArray
            Spatial map of alpha values for each grid cell.

        Returns
        -------
        xarray.DataArray
            dims (output=2, Y, X) => [error, prediction].
        """
        chunksize_x = int(np.round(len(y_train.get_index("X")) / self.nb_cores))
        chunksize_y = int(np.round(len(y_train.get_index("Y")) / self.nb_cores))

        X_train['T'] = y_train['T']
        y_train = y_train.transpose('T','Y','X')
        X_test = X_test.squeeze()
        y_test = y_test.drop_vars('T').squeeze().transpose('Y','X')

        # Align alpha with y_train, y_test
        y_train, alpha = xr.align(y_train, alpha, join='outer')
        y_test, alpha = xr.align(y_test, alpha, join='outer')

        client = Client(n_workers=self.nb_cores, threads_per_worker=1)
        result_da = xr.apply_ufunc(
            self.fit_predict,
            X_train,
            y_train.chunk({'Y': chunksize_y, 'X': chunksize_x}),
            X_test,
            y_test.chunk({'Y': chunksize_y, 'X': chunksize_x}),
            alpha.chunk({'Y': chunksize_y, 'X': chunksize_x}),
            input_core_dims=[
                ('T','features'),   # x
                ('T',),            # y
                ('features',),     # x_test
                (),                # y_test
                ()                 # alpha
            ],
            vectorize=True,
            dask='parallelized',
            output_core_dims=[('output',)],
            output_dtypes=['float'],
            dask_gufunc_kwargs={'output_sizes': {'output': 2}},
        )
        result_ = result_da.compute()
        client.close()
        return result_.isel(output=1)

    # ------------------------------------------------------------------
    # Probability Calculation Methods
    # ------------------------------------------------------------------
    @staticmethod
    def calculate_tercile_probabilities(best_guess, error_variance, first_tercile, second_tercile, dof):
        """
        Student's t-based method.
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
    def calculate_tercile_probabilities_gamma(best_guess, error_variance, T1, T2):
        """
        Gamma-based method.
        """
        n_time = len(best_guess)
        pred_prob = np.empty((3, n_time), dtype=float)

        if np.any(np.isnan(best_guess)) or np.any(np.isnan(error_variance)):
            pred_prob[:] = np.nan
            return pred_prob

        best_guess = np.asarray(best_guess, dtype=float)
        error_variance = np.asarray(error_variance, dtype=float)
        T1 = np.asarray(T1, dtype=float)
        T2 = np.asarray(T2, dtype=float)

        alpha = (best_guess**2) / error_variance
        theta = error_variance / best_guess

        cdf_t1 = gamma.cdf(T1, a=alpha, scale=theta)
        cdf_t2 = gamma.cdf(T2, a=alpha, scale=theta)

        pred_prob[0, :] = cdf_t1
        pred_prob[1, :] = cdf_t2 - cdf_t1
        pred_prob[2, :] = 1.0 - cdf_t2
        return pred_prob

    @staticmethod
    def calculate_tercile_probabilities_nonparametric(best_guess, error_samples, first_tercile, second_tercile):
        """
        Non-parametric method (requires historical errors).
        """
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

    @staticmethod
    def calculate_tercile_probabilities_normal(best_guess, error_variance, first_tercile, second_tercile):
        """
        Normal-based method using the Gaussian CDF.
        """
        n_time = len(best_guess)
        pred_prob = np.empty((3, n_time))

        if np.all(np.isnan(best_guess)):
            pred_prob[:] = np.nan
        else:
            error_std = np.sqrt(error_variance)
            pred_prob[0, :] = stats.norm.cdf(first_tercile, loc=best_guess, scale=error_std)
            pred_prob[1, :] = stats.norm.cdf(second_tercile, loc=best_guess, scale=error_std) - \
                              stats.norm.cdf(first_tercile, loc=best_guess, scale=error_std)
            pred_prob[2, :] = 1 - stats.norm.cdf(second_tercile, loc=best_guess, scale=error_std)
        return pred_prob

    @staticmethod
    def calculate_tercile_probabilities_lognormal(best_guess, error_variance, first_tercile, second_tercile):
        """
        Lognormal-based method.
        """
        n_time = len(best_guess)
        pred_prob = np.empty((3, n_time))

        if np.any(np.isnan(best_guess)) or np.any(np.isnan(error_variance)):
            pred_prob[:] = np.nan
            return pred_prob

        sigma = np.sqrt(np.log(1 + error_variance / (best_guess**2)))
        mu = np.log(best_guess) - sigma**2 / 2

        pred_prob[0, :] = lognorm.cdf(first_tercile, s=sigma, scale=np.exp(mu))
        pred_prob[1, :] = lognorm.cdf(second_tercile, s=sigma, scale=np.exp(mu)) - \
                          lognorm.cdf(first_tercile, s=sigma, scale=np.exp(mu))
        pred_prob[2, :] = 1 - lognorm.cdf(second_tercile, s=sigma, scale=np.exp(mu))
        return pred_prob

    def compute_prob(self, Predictant, clim_year_start, clim_year_end, hindcast_det):
        """
        Compute tercile probabilities using either 't', 'gamma', 'normal', 'lognormal', or 'nonparam'.

        Parameters
        ----------
        Predictant : xarray.DataArray
            Observed data array (T, Y, X).
        clim_year_start : int
            Start year for climatology.
        clim_year_end : int
            End year for climatology.
        hindcast_det : xarray.DataArray
            Deterministic forecast with shape (output=2, T, Y, X),
            where output=[error, prediction].

        Returns
        -------
        hindcast_prob : xarray.DataArray
            dims (probability=3, T, Y, X) => [PB, PN, PA].
        """
        index_start = Predictant.get_index("T").get_loc(str(clim_year_start)).start
        index_end   = Predictant.get_index("T").get_loc(str(clim_year_end)).stop
        rainfall_for_tercile = Predictant.isel(T=slice(index_start, index_end))
        terciles = rainfall_for_tercile.quantile([0.33, 0.67], dim='T')
        error_variance = (Predictant - hindcast_det).var(dim='T')
        T1 = terciles.isel(quantile=0).drop_vars('quantile')
        T2 = terciles.isel(quantile=1).drop_vars('quantile')

        if self.dist_method == "t":
            dof = len(Predictant.get_index("T")) - 2
            calc_func = self.calculate_tercile_probabilities
 
            hindcast_prob = xr.apply_ufunc(
                calc_func,
                hindcast_det,
                error_variance,
                T1,
                T2,
                input_core_dims=[('T',), (), (), ()],
                vectorize=True,
                kwargs={'dof': dof},
                dask='parallelized',
                output_core_dims=[('probability','T')],
                output_dtypes=['float'],
                dask_gufunc_kwargs={'output_sizes': {'probability': 3}, "allow_rechunk": True},
            )

        elif self.dist_method == "gamma":
            calc_func = self.calculate_tercile_probabilities_gamma

            hindcast_prob = xr.apply_ufunc(
                calc_func,
                hindcast_det,
                error_variance,
                T1,
                T2,
                input_core_dims=[('T',), (), (), ()],
                vectorize=True,
                dask='parallelized',
                output_core_dims=[('probability','T')],
                output_dtypes=['float'],
                dask_gufunc_kwargs={'output_sizes': {'probability': 3}, "allow_rechunk":True},
            )

        elif self.dist_method == "normal":
            calc_func = self.calculate_tercile_probabilities_normal

            hindcast_prob = xr.apply_ufunc(
                calc_func,
                hindcast_det,
                error_variance,
                T1,
                T2,
                input_core_dims=[('T',), (), (), ()],
                vectorize=True,
                dask='parallelized',
                output_core_dims=[('probability','T')],
                output_dtypes=['float'],
                dask_gufunc_kwargs={'output_sizes': {'probability': 3}, "allow_rechunk":True},
            )

        elif self.dist_method == "lognormal":
            calc_func = self.calculate_tercile_probabilities_lognormal

            hindcast_prob = xr.apply_ufunc(
                calc_func,
                hindcast_det,
                error_variance,
                T1,
                T2,
                input_core_dims=[('T',), (), (), ()],
                vectorize=True,
                dask='parallelized',
                output_core_dims=[('probability','T')],
                output_dtypes=['float'],
                dask_gufunc_kwargs={'output_sizes': {'probability': 3}, "allow_rechunk":True},
            )

        elif self.dist_method == "nonparam":
            calc_func = self.calculate_tercile_probabilities_nonparametric
            error_samples = Predictant - hindcast_det

            hindcast_prob = xr.apply_ufunc(
                calc_func,
                hindcast_det,
                error_samples,
                T1,
                T2,
                input_core_dims=[('T',), ('T',), (), ()],
                output_core_dims=[('probability','T')],
                vectorize=True, 
                dask='parallelized',
                output_dtypes=[float],
                dask_gufunc_kwargs={'output_sizes': {'probability': 3}},
            )

        else:
            raise ValueError(
                f"Invalid dist_method: {self.dist_method}. "
                "Choose from ['t','gamma','normal','lognormal','nonparam']."
            )

        hindcast_prob = hindcast_prob.assign_coords(probability=('probability', ['PB','PN','PA']))
        return hindcast_prob.transpose('probability','T','Y','X')

    # ------------------------------------------------------------------
    # FORECAST METHOD
    # ------------------------------------------------------------------
    def forecast(self, Predictant, clim_year_start, clim_year_end, Predictor, hindcast_det, Predictor_for_year, alpha):
        """
        Generates a ridge-based forecast for a single future time (year) 
        using alpha values and returns both forecast + tercile probabilities.

        Parameters
        ----------
        Predictant : xarray.DataArray
            Observed data (T, Y, X).
        clim_year_start : int
            Start year of climatology.
        clim_year_end : int
            End year of climatology.
        Predictor : xarray.DataArray
            Historical predictor data (T, features).
        hindcast_det : xarray.DataArray
            Historical deterministic forecast with dims (output=2, T, Y, X).
        Predictor_for_year : xarray.DataArray
            Single-year predictor data (features,).
        alpha : xarray.DataArray
            Spatial map of alpha values (Y, X).

        Returns
        -------
        result_ : xarray.DataArray
            dims (output=2, Y, X) => [error, prediction].
            The "error" is NaN in a real forecast scenario.
        hindcast_prob : xarray.DataArray
            dims (probability=3, Y, X) => [PB, PN, PA].
        """
        # Provide dummy y_test so fit_predict can return [error, prediction]
        y_test_dummy = xr.full_like(Predictant.isel(T=0), np.nan)

        # Chunk sizes
        chunksize_x = int(np.round(len(Predictant.get_index("X")) / self.nb_cores))
        chunksize_y = int(np.round(len(Predictant.get_index("Y")) / self.nb_cores))

        # Align
        Predictor['T'] = Predictant['T']
        Predictant = Predictant.transpose('T','Y','X')
        Predictor_for_year_ = Predictor_for_year.squeeze()

        # Align alpha with Predictant
        Predictant, alpha = xr.align(Predictant, alpha, join='outer')

        # 1) Fit+predict in parallel => shape (2, Y, X)
        client = Client(n_workers=self.nb_cores, threads_per_worker=1)
        result_da = xr.apply_ufunc(
            self.fit_predict,
            Predictor,
            Predictant.chunk({'Y': chunksize_y, 'X': chunksize_x}),
            Predictor_for_year_,
            y_test_dummy.chunk({'Y': chunksize_y, 'X': chunksize_x}),  # dummy test
            alpha.chunk({'Y': chunksize_y, 'X': chunksize_x}),
            input_core_dims=[
                ('T','features'),  # x
                ('T',),           # y
                ('features',),    # x_test
                (),               # y_test
                ()                # alpha
            ],
            vectorize=True,
            dask='parallelized',
            output_core_dims=[('output',)],  # => [error, prediction]
            output_dtypes=['float'],
            dask_gufunc_kwargs={'output_sizes': {'output': 2}},
        )
        result_ = result_da.compute()
        client.close()
        result_ = result_.isel(output=1)

        # result_ => dims (output=2, Y, X). 
        # For a real future forecast, "error" is NaN, "prediction" is the forecast.

        # 2) Compute thresholds T1, T2 from climatology
        index_start = Predictant.get_index("T").get_loc(str(clim_year_start)).start
        index_end   = Predictant.get_index("T").get_loc(str(clim_year_end)).stop
        rainfall_for_tercile = Predictant.isel(T=slice(index_start, index_end))
        terciles = rainfall_for_tercile.quantile([0.33, 0.67], dim='T')
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

        # 3) Tercile probabilities
        if self.dist_method == "t":
            calc_func = self.calculate_tercile_probabilities
            dof = len(Predictant.get_index("T")) - 2


            hindcast_prob = xr.apply_ufunc(
                calc_func,
                forecast_expanded,
                error_variance,
                T1,
                T2,
                input_core_dims=[('T',), (), (), ()],
                vectorize=True,
                kwargs={'dof': dof},
                dask='parallelized',
                output_core_dims=[('probability','T')],
                output_dtypes=['float'],
                dask_gufunc_kwargs={'output_sizes': {'probability': 3}, "allow_rechunk": True},
            )

        elif self.dist_method == "gamma":
            calc_func = self.calculate_tercile_probabilities_gamma

            hindcast_prob = xr.apply_ufunc(
                calc_func,
                forecast_expanded,
                error_variance,
                T1,
                T2,
                input_core_dims=[('T',), (), (), ()],
                vectorize=True,
                dask='parallelized',
                output_core_dims=[('probability','T')],
                output_dtypes=['float'],
                dask_gufunc_kwargs={'output_sizes': {'probability': 3}}
            )

        elif self.dist_method == "normal":
            calc_func = self.calculate_tercile_probabilities_normal

            hindcast_prob = xr.apply_ufunc(
                calc_func,
                forecast_expanded,
                error_variance,
                T1,
                T2,
                input_core_dims=[('T',), (), (), ()],
                vectorize=True,
                dask='parallelized',
                output_core_dims=[('probability','T')],
                output_dtypes=['float'],
                dask_gufunc_kwargs={'output_sizes': {'probability': 3}}
            )

        elif self.dist_method == "lognormal":
            calc_func = self.calculate_tercile_probabilities_lognormal

            hindcast_prob = xr.apply_ufunc(
                calc_func,
                forecast_expanded,
                error_variance,
                T1,
                T2,
                input_core_dims=[('T',), (), (), ()],
                vectorize=True,
                dask='parallelized',
                output_core_dims=[('probability','T')],
                output_dtypes=['float'],
                dask_gufunc_kwargs={'output_sizes': {'probability': 3}}
            )

        elif self.dist_method == "nonparam":
            calc_func = self.calculate_tercile_probabilities_nonparametric
            error_samples = Predictant - hindcast_det
            error_samples = error_samples.rename({'T':'S'})
            hindcast_prob = xr.apply_ufunc(
                calc_func,
                forecast_expanded,
                error_samples,
                T1,
                T2,
                input_core_dims=[('T',), ('S',), (), ()],
                output_core_dims=[('probability','T')],
                vectorize=True, 
                dask='parallelized',
                output_dtypes=[float],
                dask_gufunc_kwargs={'output_sizes': {'probability': 3}}
            )

        else:
            raise ValueError(
                f"Invalid dist_method: {self.dist_method}. "
                "Choose 't','gamma','normal','lognormal','nonparam'."
            )

        hindcast_prob = hindcast_prob.assign_coords(probability=('probability', ['PB','PN','PA']))
        hindcast_prob_out = hindcast_prob.transpose('probability','T','Y','X') #.drop_vars('T').squeeze()

        # Return [error, prediction] plus tercile probabilities
        return forecast_expanded, hindcast_prob_out


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
    dist_method : str
        Distribution method for tercile probability calculations:
        one of {'gamma','t','normal','lognormal','nonparam'}.

    Methods:
    --------
    fit_predict(x, y, x_test, y_test, alpha):
        Fits the Lasso model to the provided training data and predicts values 
        for the test set. Returns [error, prediction].

    compute_hyperparameters(predictand, predictor):
        Performs clustering of the spatial grid and computes optimal alpha values 
        for each cluster. Returns the alpha values as an xarray and the cluster assignments.

    compute_model(X_train, y_train, X_test, y_test, alpha):
        Computes the Lasso model prediction for the training/test datasets 
        using the given alpha values. Utilizes Dask for parallelized processing.

    calculate_tercile_probabilities(...):
        Calculates tercile probabilities for a given forecast using the chosen 
        distribution method.

    compute_prob(Predictant, clim_year_start, clim_year_end, Predictor, hindcast_det):
        Computes probabilistic forecasts for rainfall terciles based on 
        climatological terciles, utilizing a hindcast and Lasso regression output.

    forecast(Predictant, clim_year_start, clim_year_end, Predictor, hindcast_det, 
             Predictor_for_year, alpha):
        Generates a forecast for a single time step using Lasso with alpha, 
        then computes tercile probabilities based on the chosen distribution method.
    """

    def __init__(self, alpha_range=None, n_clusters=5, nb_cores=1, dist_method="gamma"):
        if alpha_range is None:
            alpha_range = np.array([10**i for i in range(-6, 6)])  # default
        self.alpha_range = alpha_range
        self.n_clusters = n_clusters
        self.nb_cores = nb_cores
        self.dist_method = dist_method
    
    def fit_predict(self, x, y, x_test, y_test, alpha):
        """
        Fit a Lasso model and make predictions.

        Parameters
        ----------
        x : ndarray
            Training data (shape: [n_samples, n_features]).
        y : ndarray
            Training targets (shape: [n_samples,]).
        x_test : ndarray
            Test data (shape: [n_features,] or [1, n_features]).
        y_test : float
            Target value(s) for test data.
        alpha : float
            Regularization parameter for Lasso.

        Returns
        -------
        ndarray
            [error, prediction], each of shape (1,) or scalar, 
            or [np.nan, np.nan] if no valid data.
        """
        model = linear_model.Lasso(alpha)
        mask = np.isfinite(y) & np.all(np.isfinite(x), axis=-1)

        if np.any(mask):
            y_clean = y[mask]
            x_clean = x[mask, :]
            model.fit(x_clean, y_clean)

            if x_test.ndim == 1:
                x_test = x_test.reshape(1, -1)

            preds = model.predict(x_test)
            preds[preds < 0] = 0  # clip negative if it's precip
            error_ = y_test - preds
            return np.array([error_, preds]).squeeze()
        else:
            return np.array([np.nan, np.nan]).squeeze()

    def compute_hyperparameters(self, predictand, predictor):
        """
        Clusters the spatial domain and finds the best alpha for each cluster by mean time series.

        Returns
        -------
        alpha_array : xarray.DataArray
            Spatial map of best alpha values.
        Cluster : xarray.DataArray
            Cluster assignments for each grid cell.
        """
        predictor['T'] = predictand['T']
        # (a) KMeans on the mean of predictand
        kmeans = KMeans(n_clusters=self.n_clusters)
        predictand_dropna = (
            predictand.to_dataframe()
            .reset_index()
            .dropna()
            .drop(columns=['T'])
        )
        # Use the 3rd column for clustering
        col_name = predictand_dropna.columns[2]
        predictand_dropna['cluster'] = kmeans.fit_predict(
            predictand_dropna[col_name].to_frame()
        )
        # Convert back to xarray
        df_unique = predictand_dropna.drop_duplicates(subset=['Y', 'X'])
        dataset = df_unique.set_index(['Y', 'X']).to_xarray()

        Cluster = (
            dataset['cluster'] *
            xr.where(~np.isnan(predictand.isel(T=0)), 1, np.nan)
        ).drop_vars("T", errors='ignore')
        
        xarray1, xarray2 = xr.align(predictand, Cluster)
        clusters = np.unique(xarray2)
        clusters = clusters[~np.isnan(clusters)]

        # (b) For each cluster, get mean time series, run LassoCV
        cluster_means = {
            int(cval): xarray1.where(xarray2 == cval).mean(dim=['Y','X'], skipna=True)
            for cval in clusters
        }
        model_cv = linear_model.LassoCV(alphas=self.alpha_range, cv=5)
        alpha_cluster = {}
        for cval in clusters:
            cval_int = int(cval)
            y_cluster = cluster_means[cval_int].dropna(dim='T')
            if len(y_cluster['T']) == 0:
                continue
            predictor_cluster = predictor.sel(T=y_cluster['T'])
            X_mat = predictor_cluster.values
            y_vec = y_cluster.values
            model_cv.fit(X_mat, y_vec)
            alpha_cluster[cval_int] = model_cv.alpha_

        # (c) Create alpha_array
        alpha_array = Cluster.copy()
        for key, val in alpha_cluster.items():
            alpha_array = alpha_array.where(alpha_array != key, other=val)
        alpha_array, Cluster, predictand = xr.align(alpha_array, Cluster, predictand, join="outer")
        return alpha_array, Cluster

    def compute_model(self, X_train, y_train, X_test, y_test, alpha):
        """
        Computes Lasso predictions for spatiotemporal data using provided alpha values.

        Returns
        -------
        xarray.DataArray
            dims (output=2, Y, X) => [error, prediction].
        """
        chunksize_x = int(np.round(len(y_train.get_index("X")) / self.nb_cores))
        chunksize_y = int(np.round(len(y_train.get_index("Y")) / self.nb_cores))

        X_train['T'] = y_train['T']
        y_train = y_train.transpose('T','Y','X')
        X_test = X_test.squeeze()
        y_test = y_test.drop_vars('T').squeeze().transpose('Y','X')

        # Align alpha with y_train, y_test
        y_train, alpha = xr.align(y_train, alpha)
        y_test, alpha = xr.align(y_test, alpha)

        client = Client(n_workers=self.nb_cores, threads_per_worker=1)
        result_da = xr.apply_ufunc(
            self.fit_predict,
            X_train,
            y_train.chunk({'Y': chunksize_y, 'X': chunksize_x}),
            X_test,
            y_test.chunk({'Y': chunksize_y, 'X': chunksize_x}),
            alpha.chunk({'Y': chunksize_y, 'X': chunksize_x}),
            input_core_dims=[
                ('T','features'),  # x
                ('T',),           # y
                ('features',),    # x_test
                (),               # y_test
                ()                # alpha
            ],
            vectorize=True,
            dask='parallelized',
            output_core_dims=[('output',)],
            output_dtypes=['float'],
            dask_gufunc_kwargs={'output_sizes': {'output': 2}}
        )
        result_ = result_da.compute()
        client.close()
        return result_.isel(output=1)
    # ------------------------------------------------------------------
    # Probability Calculation Methods
    # ------------------------------------------------------------------

    @staticmethod
    def calculate_tercile_probabilities(best_guess, error_variance, first_tercile, second_tercile, dof):
        """
        Student's t-based method
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
    def calculate_tercile_probabilities_gamma(best_guess, error_variance, T1, T2):
        """
        Gamma-based method
        """
        n_time = len(best_guess)
        pred_prob = np.empty((3, n_time), dtype=float)
        if np.any(np.isnan(best_guess)) or np.any(np.isnan(error_variance)):
            pred_prob[:] = np.nan
            return pred_prob

        best_guess = np.asarray(best_guess, dtype=float)
        error_variance = np.asarray(error_variance, dtype=float)
        T1 = np.asarray(T1, dtype=float)
        T2 = np.asarray(T2, dtype=float)

        alpha = (best_guess**2) / error_variance
        theta = error_variance / best_guess
        cdf_t1 = gamma.cdf(T1, a=alpha, scale=theta)
        cdf_t2 = gamma.cdf(T2, a=alpha, scale=theta)
        pred_prob[0, :] = cdf_t1
        pred_prob[1, :] = cdf_t2 - cdf_t1
        pred_prob[2, :] = 1.0 - cdf_t2
        return pred_prob

    @staticmethod
    def calculate_tercile_probabilities_nonparametric(best_guess, error_samples, first_tercile, second_tercile):
        """
        Non-parametric method (requires historical errors).
        """
        n_time = len(best_guess)
        pred_prob = np.full((3, n_time), np.nan, dtype=float)

        for t in range(n_time):
            if np.isnan(best_guess[t]):
                continue

            dist = best_guess[t] + error_samples  
            dist = dist[np.isfinite(dist)]  
            if len(dist) == 0:
                continue

            p_below   = np.mean(dist < first_tercile)
            p_between = np.mean((dist >= first_tercile) & (dist < second_tercile))
            p_above   = 1.0 - (p_below + p_between)

            pred_prob[0, t] = p_below
            pred_prob[1, t] = p_between
            pred_prob[2, t] = p_above
        return pred_prob

    @staticmethod
    def calculate_tercile_probabilities_normal(best_guess, error_variance, first_tercile, second_tercile):
        """
        Normal-based method using the Gaussian CDF.
        """
        n_time = len(best_guess)
        pred_prob = np.empty((3, n_time))
        
        if np.all(np.isnan(best_guess)):
            pred_prob[:] = np.nan
        else:
            error_std = np.sqrt(error_variance)
            pred_prob[0, :] = stats.norm.cdf(first_tercile, loc=best_guess, scale=error_std)
            pred_prob[1, :] = stats.norm.cdf(second_tercile, loc=best_guess, scale=error_std) - \
                              stats.norm.cdf(first_tercile, loc=best_guess, scale=error_std)
            pred_prob[2, :] = 1 - stats.norm.cdf(second_tercile, loc=best_guess, scale=error_std)
        return pred_prob

    @staticmethod
    def calculate_tercile_probabilities_lognormal(best_guess, error_variance, first_tercile, second_tercile):
        """
        Lognormal-based method.
        """
        n_time = len(best_guess)
        pred_prob = np.empty((3, n_time))
        
        if np.any(np.isnan(best_guess)) or np.any(np.isnan(error_variance)):
            pred_prob[:] = np.nan
            return pred_prob

        sigma = np.sqrt(np.log(1 + error_variance / (best_guess**2)))
        mu = np.log(best_guess) - sigma**2 / 2
        pred_prob[0, :] = lognorm.cdf(first_tercile, s=sigma, scale=np.exp(mu))
        pred_prob[1, :] = lognorm.cdf(second_tercile, s=sigma, scale=np.exp(mu)) - \
                          lognorm.cdf(first_tercile, s=sigma, scale=np.exp(mu))
        pred_prob[2, :] = 1 - lognorm.cdf(second_tercile, s=sigma, scale=np.exp(mu))
        return pred_prob

    def compute_prob(self, Predictant, clim_year_start, clim_year_end, hindcast_det):
        """
        Compute tercile probabilities using self.dist_method.

        Parameters
        ----------
        Predictant : xarray.DataArray (T, Y, X)
            Observed data.
        clim_year_start : int
        clim_year_end : int
            The start and end years for the climatology.
        hindcast_det : xarray.DataArray
            Deterministic forecast with dims (output=2, T, Y, X).

        Returns
        -------
        hindcast_prob : xarray.DataArray
            dims (probability=3, T, Y, X) => [PB, PN, PA].
        """
        # 1) Identify climatology slice
        index_start = Predictant.get_index("T").get_loc(str(clim_year_start)).start
        index_end   = Predictant.get_index("T").get_loc(str(clim_year_end)).stop
        rainfall_for_tercile = Predictant.isel(T=slice(index_start, index_end))
        terciles = rainfall_for_tercile.quantile([0.33, 0.67], dim='T')
        error_variance = (Predictant - hindcast_det).var(dim='T')

        T1 = terciles.isel(quantile=0).drop_vars('quantile')
        T2 = terciles.isel(quantile=1).drop_vars('quantile')

        # 2) Distinguish distribution method
        if self.dist_method == "t":
            dof = len(Predictant.get_index("T")) - 2
            calc_func = self.calculate_tercile_probabilities
            hindcast_prob = xr.apply_ufunc(
                calc_func,
                hindcast_det,
                error_variance,
                T1,
                T2,
                input_core_dims=[('T',), (), (), ()],
                vectorize=True,
                kwargs={'dof': dof},
                dask='parallelized',
                output_core_dims=[('probability','T')],
                output_dtypes=['float'],
                dask_gufunc_kwargs={'output_sizes': {'probability': 3}, "allow_rechunk": True},
            )

        elif self.dist_method == "gamma":
            calc_func = self.calculate_tercile_probabilities_gamma
            hindcast_prob = xr.apply_ufunc(
                calc_func,
                hindcast_det,
                error_variance,
                T1,
                T2,
                input_core_dims=[('T',), (), (), ()],
                vectorize=True,
                dask='parallelized',
                output_core_dims=[('probability','T')],
                output_dtypes=['float'],
                dask_gufunc_kwargs={'output_sizes': {'probability': 3}, "allow_rechunk":True},
            )

        elif self.dist_method == "normal":
            calc_func = self.calculate_tercile_probabilities_normal
            hindcast_prob = xr.apply_ufunc(
                calc_func,
                hindcast_det,
                error_variance,
                T1,
                T2,
                input_core_dims=[('T',), (), (), ()],
                vectorize=True,
                dask='parallelized',
                output_core_dims=[('probability','T')],
                output_dtypes=['float'],
                dask_gufunc_kwargs={'output_sizes': {'probability': 3}, "allow_rechunk":True},
            )

        elif self.dist_method == "lognormal":
            calc_func = self.calculate_tercile_probabilities_lognormal
            hindcast_prob = xr.apply_ufunc(
                calc_func,
                hindcast_det,
                error_variance,
                T1,
                T2,
                input_core_dims=[('T',), (), (), ()],
                vectorize=True,
                dask='parallelized',
                output_core_dims=[('probability','T')],
                output_dtypes=['float'],
                dask_gufunc_kwargs={'output_sizes': {'probability': 3}, "allow_rechunk":True},
            )

        elif self.dist_method == "nonparam":
            calc_func = self.calculate_tercile_probabilities_nonparametric
            error_samples = (Predictant - hindcast_det)
            hindcast_prob = xr.apply_ufunc(
                calc_func,
                hindcast_det,
                error_samples,
                T1,
                T2,
                input_core_dims=[('T',), ('T',), (), ()],
                output_core_dims=[('probability','T')],
                vectorize=True,
                dask='parallelized',
                output_dtypes=[float],
                dask_gufunc_kwargs={'output_sizes': {'probability': 3}},
            )

        else:
            raise ValueError(f"Invalid dist_method: {self.dist_method}. "
                             "Must be one of ['t','gamma','normal','lognormal','nonparam'].")

        hindcast_prob = hindcast_prob.assign_coords(probability=('probability', ['PB','PN','PA']))
        return hindcast_prob.transpose('probability','T','Y','X')

    def forecast(self, Predictant, clim_year_start, clim_year_end, Predictor, hindcast_det, Predictor_for_year, alpha):
        """
        Generate a forecast for a single time (year) using Lasso with alpha, 
        then compute tercile probabilities from the chosen distribution method.

        Parameters
        ----------
        Predictant : xarray.DataArray
            Observed data (T, Y, X).
        clim_year_start : int
            Start year of the climatology period.
        clim_year_end : int
            End year of the climatology period.
        Predictor : xarray.DataArray
            Historical predictor data (T, features).
        hindcast_det : xarray.DataArray
            Historical deterministic forecast with dims (output=2, T, Y, X).
        Predictor_for_year : xarray.DataArray
            Single-year predictor data (features,).
        alpha : xarray.DataArray
            Spatial map of alpha values (Y, X).

        Returns
        -------
        result_ : xarray.DataArray
            dims (output=2, Y, X) => [error, prediction].
            For a real forecast scenario, the error is typically NaN.
        hindcast_prob : xarray.DataArray
            dims (probability=3, Y, X) => [PB, PN, PA].
        """
        # 1) Create dummy y_test => shape (Y, X) with NaN
        y_test_dummy = xr.full_like(Predictant.isel(T=0), np.nan)

        # 2) Chunk sizes
        chunksize_x = int(np.round(len(Predictant.get_index("X")) / self.nb_cores))
        chunksize_y = int(np.round(len(Predictant.get_index("Y")) / self.nb_cores))

        # Align
        Predictor['T'] = Predictant['T']
        Predictant = Predictant.transpose('T','Y','X')
        Predictor_for_year_ = Predictor_for_year.squeeze()

        # Align alpha with the domain
        Predictant, alpha = xr.align(Predictant, alpha, join='outer')

        # 3) Fit+predict in parallel => produce (2, Y, X)
        client = Client(n_workers=self.nb_cores, threads_per_worker=1)
        result_da = xr.apply_ufunc(
            self.fit_predict,
            Predictor,
            Predictant.chunk({'Y': chunksize_y, 'X': chunksize_x}),
            Predictor_for_year_,
            y_test_dummy.chunk({'Y': chunksize_y, 'X': chunksize_x}),  # dummy
            alpha.chunk({'Y': chunksize_y, 'X': chunksize_x}),
            input_core_dims=[
                ('T','features'),  # x
                ('T',),           # y
                ('features',),    # x_test
                (),               # y_test
                ()                # alpha
            ],
            vectorize=True,
            dask='parallelized',
            output_core_dims=[('output',)],  # => [error, prediction]
            output_dtypes=['float'],
            dask_gufunc_kwargs={'output_sizes': {'output': 2}}
        )
        result_ = result_da.compute()
        client.close()
        result_ = result_.isel(output=1)

        # result_ => dims (output=2, Y, X). 
        # For a real future forecast, "error" is NaN, "prediction" is the forecast.

        # 2) Compute thresholds T1, T2 from climatology
        index_start = Predictant.get_index("T").get_loc(str(clim_year_start)).start
        index_end   = Predictant.get_index("T").get_loc(str(clim_year_end)).stop
        rainfall_for_tercile = Predictant.isel(T=slice(index_start, index_end))
        terciles = rainfall_for_tercile.quantile([0.33, 0.67], dim='T')
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

        # 3) Tercile probabilities
        if self.dist_method == "t":
            calc_func = self.calculate_tercile_probabilities
            dof = len(Predictant.get_index("T")) - 2


            hindcast_prob = xr.apply_ufunc(
                calc_func,
                forecast_expanded,
                error_variance,
                T1,
                T2,
                input_core_dims=[('T',), (), (), ()],
                vectorize=True,
                kwargs={'dof': dof},
                dask='parallelized',
                output_core_dims=[('probability','T')],
                output_dtypes=['float'],
                dask_gufunc_kwargs={'output_sizes': {'probability': 3}, "allow_rechunk": True},
            )

        elif self.dist_method == "gamma":
            calc_func = self.calculate_tercile_probabilities_gamma

            hindcast_prob = xr.apply_ufunc(
                calc_func,
                forecast_expanded,
                error_variance,
                T1,
                T2,
                input_core_dims=[('T',), (), (), ()],
                vectorize=True,
                dask='parallelized',
                output_core_dims=[('probability','T')],
                output_dtypes=['float'],
                dask_gufunc_kwargs={'output_sizes': {'probability': 3}}
            )

        elif self.dist_method == "normal":
            calc_func = self.calculate_tercile_probabilities_normal

            hindcast_prob = xr.apply_ufunc(
                calc_func,
                forecast_expanded,
                error_variance,
                T1,
                T2,
                input_core_dims=[('T',), (), (), ()],
                vectorize=True,
                dask='parallelized',
                output_core_dims=[('probability','T')],
                output_dtypes=['float'],
                dask_gufunc_kwargs={'output_sizes': {'probability': 3}}
            )

        elif self.dist_method == "lognormal":
            calc_func = self.calculate_tercile_probabilities_lognormal

            hindcast_prob = xr.apply_ufunc(
                calc_func,
                forecast_expanded,
                error_variance,
                T1,
                T2,
                input_core_dims=[('T',), (), (), ()],
                vectorize=True,
                dask='parallelized',
                output_core_dims=[('probability','T')],
                output_dtypes=['float'],
                dask_gufunc_kwargs={'output_sizes': {'probability': 3}}
            )

        elif self.dist_method == "nonparam":
            calc_func = self.calculate_tercile_probabilities_nonparametric
            error_samples = Predictant - hindcast_det
            error_samples = error_samples.rename({'T':'S'})
            hindcast_prob = xr.apply_ufunc(
                calc_func,
                forecast_expanded,
                error_samples,
                T1,
                T2,
                input_core_dims=[('T',), ('S',), (), ()],
                output_core_dims=[('probability','T')],
                vectorize=True, 
                dask='parallelized',
                output_dtypes=[float],
                dask_gufunc_kwargs={'output_sizes': {'probability': 3}}
            )
        else:
            raise ValueError(
                f"Invalid dist_method: {self.dist_method}. "
                "Choose 't','gamma','normal','lognormal','nonparam'."
            )

        hindcast_prob = hindcast_prob.assign_coords(probability=('probability', ['PB','PN','PA']))
        hindcast_prob_out = hindcast_prob.transpose('probability','T','Y','X') #.drop_vars('T').squeeze()

        # Return [error, prediction] plus tercile probabilities
        return forecast_expanded, hindcast_prob_out

class WAS_LassoLars_Model:
    """
    A class to implement the Lasso Least Angle Regression (LassoLars) model for spatiotemporal 
    climate prediction.

    This model is designed to work with climate data by clustering spatial regions, computing 
    hyperparameters for each cluster, and fitting a LassoLars model for predictions. The model 
    is optimized for parallel execution using Dask.

    Parameters
    ----------
    alpha_range : array-like, optional
        Range of alpha values for the LassoLars model.
        Default is np.array([10**i for i in range(-6, 6)]).
    n_clusters : int, default=5
        Number of clusters to partition the spatial data.
    nb_cores : int, default=1
        Number of cores for parallel processing.
    dist_method : str, default='gamma'
        Distribution method for calculating tercile probabilities. 
        One of {'gamma','t','normal','lognormal','nonparam'}.

    Methods
    -------
    fit_predict(x, y, x_test, y_test, alpha)
        Fits the LassoLars model to the training data and predicts values for the test data.

    compute_hyperparameters(predictand, predictor)
        Computes cluster-wise optimal alpha values for LassoLars using cross-validation.

    compute_model(X_train, y_train, X_test, y_test, alpha)
        Fits and predicts the LassoLars model using Dask for parallel execution.

    calculate_tercile_probabilities(...)
        Calculates tercile probabilities using the chosen distribution method (Student's t, gamma, etc.).

    compute_prob(Predictant, clim_year_start, clim_year_end, Predictor, hindcast_det)
        Computes the tercile probabilities for hindcast predictions using a climatological period.

    forecast(Predictant, clim_year_start, clim_year_end, Predictor, hindcast_det, Predictor_for_year, alpha)
        Generates a forecast for a single time step and computes tercile probabilities.
    """

    def __init__(self, alpha_range=None, n_clusters=5, nb_cores=1, dist_method="gamma"):
        if alpha_range is None:
            alpha_range = np.array([10**i for i in range(-6, 6)])
        self.alpha_range = alpha_range
        self.n_clusters = n_clusters
        self.nb_cores = nb_cores
        self.dist_method = dist_method
    
    def fit_predict(self, x, y, x_test, y_test, alpha):
        """
        Fits the LassoLars model to the training data and predicts values for the test data.

        Parameters
        ----------
        x : array-like
            Training predictors (shape: [n_samples, n_features]).
        y : array-like
            Training response variable (shape: [n_samples,]).
        x_test : array-like
            Test predictors (shape: [n_features,] or [1, n_features]).
        y_test : float
            Test response (scalar or shape [1,]) for test data.
        alpha : float
            Regularization strength parameter for LassoLars.

        Returns
        -------
        np.ndarray of shape (2,)
            [error, prediction], or [np.nan, np.nan] if no valid data.
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
            preds[preds < 0] = 0  # clip negative if it's precipitation
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
            The response variable for clustering and training (dims: T, Y, X).
        predictor : array-like
            Predictor variables used for fitting the model (dims: T, features).

        Returns
        -------
        alpha_array : xarray.DataArray
            Cluster-wise optimal alpha values.
        Cluster : xarray.DataArray
            Cluster assignment for each spatial point.
        """
        predictor['T'] = predictand['T']
        kmeans = KMeans(n_clusters=self.n_clusters)
        df = (
            predictand.mean(dim='T', skipna=True)
                      .to_dataframe(name="mean_val")
                      .reset_index()
        )
        df_nona = df.dropna(subset=["mean_val"])
        df_nona["cluster"] = kmeans.fit_predict(df_nona[["mean_val"]])

        # Convert cluster assignments to xarray
        df_clusters = df_nona[["Y", "X", "cluster"]].set_index(["Y", "X"])
        cluster_da = df_clusters.to_xarray().cluster

        # Align with predictand
        cluster_da, predictand = xr.align(cluster_da, predictand)

        # For each cluster, get mean time series, run LassoLarsCV
        unique_clusters = np.unique(cluster_da)
        unique_clusters = unique_clusters[~np.isnan(unique_clusters)]
        cluster_means = {
            int(cval): predictand.where(cluster_da == cval).mean(dim=['Y','X'], skipna=True)
            for cval in unique_clusters
        }

        model_cv = linear_model.LassoLarsCV()
        alpha_cluster = {}
        for cval in unique_clusters:
            c_int = int(cval)
            y_cluster = cluster_means[c_int].dropna(dim='T')
            if len(y_cluster['T']) == 0:
                continue
            # Align predictor in time
            predictor_cluster = predictor.sel(T=y_cluster['T'])
            X_mat = predictor_cluster.values
            y_vec = y_cluster.values
            model_cv.fit(X_mat, y_vec)
            alpha_cluster[c_int] = model_cv.alpha_

        # Create alpha_array
        alpha_array = cluster_da.copy()
        for key, val in alpha_cluster.items():
            alpha_array = alpha_array.where(alpha_array != key, other=val)

        alpha_array, cluster_da, predictand = xr.align(alpha_array, cluster_da, predictand, join="outer")
        return alpha_array, cluster_da

    def compute_model(self, X_train, y_train, X_test, y_test, alpha):
        """
        Fits and predicts the LassoLars model using Dask for parallel execution.

        Parameters
        ----------
        X_train : xarray.DataArray
            Training predictor data (dims: T, features).
        y_train : xarray.DataArray
            Training response variable (dims: T, Y, X).
        X_test : xarray.DataArray
            Test predictor data (dims: features,) or broadcastable to (Y, X).
        y_test : xarray.DataArray
            Test response variable (dims: Y, X).
        alpha : xarray.DataArray
            Cluster-wise optimal alpha values (dims: Y, X).

        Returns
        -------
        xarray.DataArray
            dims (output=2, Y, X) => [error, prediction].
        """
        chunksize_x = int(np.round(len(y_train.get_index("X")) / self.nb_cores))
        chunksize_y = int(np.round(len(y_train.get_index("Y")) / self.nb_cores))
        
        X_train['T'] = y_train['T']
        y_train = y_train.transpose('T','Y','X')
        X_test = X_test.squeeze()
        y_test = y_test.drop_vars('T').squeeze().transpose('Y','X')
        y_train, alpha = xr.align(y_train, alpha)
        y_test, alpha = xr.align(y_test, alpha)

        client = Client(n_workers=self.nb_cores, threads_per_worker=1)        
        result_da = xr.apply_ufunc(
            self.fit_predict,
            X_train,
            y_train.chunk({'Y': chunksize_y, 'X': chunksize_x}),
            X_test,
            y_test.chunk({'Y': chunksize_y, 'X': chunksize_x}),
            alpha.chunk({'Y': chunksize_y, 'X': chunksize_x}),
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
            output_dtypes=['float'],
            dask_gufunc_kwargs={'output_sizes': {'output': 2}},
        )
        result_ = result_da.compute()
        client.close()
        return result_.isel(output=1)
    
    # ---------------- Probability Calculation Methods ----------------

    @staticmethod
    def calculate_tercile_probabilities(best_guess, error_variance, first_tercile, second_tercile, dof):
        """
        Student's t-based method
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
    def calculate_tercile_probabilities_gamma(best_guess, error_variance, T1, T2):
        """
        Gamma-based method
        """
        n_time = len(best_guess)
        pred_prob = np.empty((3, n_time), dtype=float)
        if np.any(np.isnan(best_guess)) or np.any(np.isnan(error_variance)):
            pred_prob[:] = np.nan
            return pred_prob

        best_guess = np.asarray(best_guess, dtype=float)
        error_variance = np.asarray(error_variance, dtype=float)
        T1 = np.asarray(T1, dtype=float)
        T2 = np.asarray(T2, dtype=float)

        alpha = (best_guess**2) / error_variance
        theta = error_variance / best_guess

        cdf_t1 = gamma.cdf(T1, a=alpha, scale=theta)
        cdf_t2 = gamma.cdf(T2, a=alpha, scale=theta)
        pred_prob[0, :] = cdf_t1
        pred_prob[1, :] = cdf_t2 - cdf_t1
        pred_prob[2, :] = 1.0 - cdf_t2
        return pred_prob

    @staticmethod
    def calculate_tercile_probabilities_nonparametric(best_guess, error_samples, first_tercile, second_tercile):
        """
        Non-parametric method (requires historical errors).
        """
        n_time = len(best_guess)
        pred_prob = np.full((3, n_time), np.nan, dtype=float)

        for t in range(n_time):
            if np.isnan(best_guess[t]):
                continue

            dist = best_guess[t] + error_samples  
            dist = dist[np.isfinite(dist)]  
            if len(dist) == 0:
                continue

            p_below   = np.mean(dist < first_tercile)
            p_between = np.mean((dist >= first_tercile) & (dist < second_tercile))
            p_above   = 1.0 - (p_below + p_between)

            pred_prob[0, t] = p_below
            pred_prob[1, t] = p_between
            pred_prob[2, t] = p_above
        return pred_prob

    @staticmethod
    def calculate_tercile_probabilities_normal(best_guess, error_variance, first_tercile, second_tercile):
        """
        Normal-based method using the Gaussian CDF.
        """
        n_time = len(best_guess)
        pred_prob = np.empty((3, n_time))
        
        if np.all(np.isnan(best_guess)):
            pred_prob[:] = np.nan
        else:
            error_std = np.sqrt(error_variance)
            pred_prob[0, :] = stats.norm.cdf(first_tercile, loc=best_guess, scale=error_std)
            pred_prob[1, :] = stats.norm.cdf(second_tercile, loc=best_guess, scale=error_std) - \
                              stats.norm.cdf(first_tercile, loc=best_guess, scale=error_std)
            pred_prob[2, :] = 1 - stats.norm.cdf(second_tercile, loc=best_guess, scale=error_std)
        return pred_prob

    @staticmethod
    def calculate_tercile_probabilities_lognormal(best_guess, error_variance, first_tercile, second_tercile):
        """
        Lognormal-based method.
        """
        n_time = len(best_guess)
        pred_prob = np.empty((3, n_time))
        
        if np.any(np.isnan(best_guess)) or np.any(np.isnan(error_variance)):
            pred_prob[:] = np.nan
            return pred_prob

        sigma = np.sqrt(np.log(1 + error_variance / (best_guess**2)))
        mu = np.log(best_guess) - sigma**2 / 2
        pred_prob[0, :] = lognorm.cdf(first_tercile, s=sigma, scale=np.exp(mu))
        pred_prob[1, :] = lognorm.cdf(second_tercile, s=sigma, scale=np.exp(mu)) - \
                          lognorm.cdf(first_tercile, s=sigma, scale=np.exp(mu))
        pred_prob[2, :] = 1 - lognorm.cdf(second_tercile, s=sigma, scale=np.exp(mu))
        return pred_prob

    def compute_prob(self, Predictant, clim_year_start, clim_year_end, hindcast_det):
        """
        Compute tercile probabilities using self.dist_method.

        Parameters
        ----------
        Predictant : xarray.DataArray (T, Y, X)
            Observed data.
        clim_year_start : int
        clim_year_end : int
            The start and end years for the climatology.
        hindcast_det : xarray.DataArray
            Deterministic forecast with dims (output=2, T, Y, X).

        Returns
        -------
        hindcast_prob : xarray.DataArray
            dims (probability=3, T, Y, X) => [PB, PN, PA].
        """
        # 1) Identify climatology slice
        index_start = Predictant.get_index("T").get_loc(str(clim_year_start)).start
        index_end   = Predictant.get_index("T").get_loc(str(clim_year_end)).stop
        rainfall_for_tercile = Predictant.isel(T=slice(index_start, index_end))
        terciles = rainfall_for_tercile.quantile([0.33, 0.67], dim='T')
        error_variance = (Predictant - hindcast_det).var(dim='T')

        T1 = terciles.isel(quantile=0).drop_vars('quantile')
        T2 = terciles.isel(quantile=1).drop_vars('quantile')

        # 2) Distinguish distribution method
        if self.dist_method == "t":
            dof = len(Predictant.get_index("T")) - 2
            calc_func = self.calculate_tercile_probabilities
            hindcast_prob = xr.apply_ufunc(
                calc_func,
                hindcast_det,
                error_variance,
                T1,
                T2,
                input_core_dims=[('T',), (), (), ()],
                vectorize=True,
                kwargs={'dof': dof},
                dask='parallelized',
                output_core_dims=[('probability','T')],
                output_dtypes=['float'],
                dask_gufunc_kwargs={'output_sizes': {'probability': 3}, "allow_rechunk": True},
            )

        elif self.dist_method == "gamma":
            calc_func = self.calculate_tercile_probabilities_gamma
            hindcast_prob = xr.apply_ufunc(
                calc_func,
                hindcast_det,
                error_variance,
                T1,
                T2,
                input_core_dims=[('T',), (), (), ()],
                vectorize=True,
                dask='parallelized',
                output_core_dims=[('probability','T')],
                output_dtypes=['float'],
                dask_gufunc_kwargs={'output_sizes': {'probability': 3}, "allow_rechunk":True},
            )

        elif self.dist_method == "normal":
            calc_func = self.calculate_tercile_probabilities_normal
            hindcast_prob = xr.apply_ufunc(
                calc_func,
                hindcast_det,
                error_variance,
                T1,
                T2,
                input_core_dims=[('T',), (), (), ()],
                vectorize=True,
                dask='parallelized',
                output_core_dims=[('probability','T')],
                output_dtypes=['float'],
                dask_gufunc_kwargs={'output_sizes': {'probability': 3}, "allow_rechunk":True},
            )

        elif self.dist_method == "lognormal":
            calc_func = self.calculate_tercile_probabilities_lognormal
            hindcast_prob = xr.apply_ufunc(
                calc_func,
                hindcast_det,
                error_variance,
                T1,
                T2,
                input_core_dims=[('T',), (), (), ()],
                vectorize=True,
                dask='parallelized',
                output_core_dims=[('probability','T')],
                output_dtypes=['float'],
                dask_gufunc_kwargs={'output_sizes': {'probability': 3}, "allow_rechunk":True},
            )

        elif self.dist_method == "nonparam":
            calc_func = self.calculate_tercile_probabilities_nonparametric
            error_samples = (Predictant - hindcast_det)
            hindcast_prob = xr.apply_ufunc(
                calc_func,
                hindcast_det,
                error_samples,
                T1,
                T2,
                input_core_dims=[('T',), ('T',), (), ()],
                output_core_dims=[('probability','T')],
                vectorize=True,
                dask='parallelized',
                output_dtypes=[float],
                dask_gufunc_kwargs={'output_sizes': {'probability': 3}},
            )

        else:
            raise ValueError(f"Invalid dist_method: {self.dist_method}. "
                             "Must be one of ['t','gamma','normal','lognormal','nonparam'].")

        hindcast_prob = hindcast_prob.assign_coords(probability=('probability', ['PB','PN','PA']))
        return hindcast_prob.transpose('probability','T','Y','X')

    # ------------------- FORECAST METHOD -------------------

    def forecast(self, Predictant, clim_year_start, clim_year_end, Predictor, hindcast_det, Predictor_for_year, alpha):
        """
        Generates a forecast for a single time step using LassoLars with alpha,
        then computes tercile probabilities based on the chosen distribution method.

        Parameters
        ----------
        Predictant : xarray.DataArray
            Observed data (T, Y, X).
        clim_year_start : int
            Start year for the climatological reference period.
        clim_year_end : int
            End year for the climatological reference period.
        Predictor : xarray.DataArray
            Historical predictor data (T, features).
        hindcast_det : xarray.DataArray
            Historical deterministic forecast with dims (output=2, T, Y, X).
        Predictor_for_year : xarray.DataArray
            Single-year predictor data (features,) or shape (1, features).
        alpha : xarray.DataArray
            Spatial map of alpha values (Y, X) for LassoLars.

        Returns
        -------
        result_ : xarray.DataArray
            dims (output=2, Y, X) => [error, prediction].
            For a true forecast scenario, the error is typically NaN.
        hindcast_prob : xarray.DataArray
            dims (probability=3, Y, X) => [PB, PN, PA] for tercile categories.
        """
        # 1) Provide a dummy y_test with NaN so fit_predict returns [NaN, forecast]
        y_test_dummy = xr.full_like(Predictant.isel(T=0), np.nan)

        # 2) Chunk sizes
        chunksize_x = int(np.round(len(Predictant.get_index("X")) / self.nb_cores))
        chunksize_y = int(np.round(len(Predictant.get_index("Y")) / self.nb_cores))
        
        Predictor['T'] = Predictant['T']
        Predictant = Predictant.transpose('T','Y','X')
        Predictor_for_year_ = Predictor_for_year.squeeze()

        # Align alpha with the domain
        Predictant, alpha = xr.align(Predictant, alpha, join='outer')

        # 3) Parallel fit+predict => (2, Y, X)
        client = Client(n_workers=self.nb_cores, threads_per_worker=1)
        result_da = xr.apply_ufunc(
            self.fit_predict,
            Predictor,
            Predictant.chunk({'Y': chunksize_y, 'X': chunksize_x}),
            Predictor_for_year_,
            y_test_dummy.chunk({'Y': chunksize_y, 'X': chunksize_x}),   # dummy test
            alpha.chunk({'Y': chunksize_y, 'X': chunksize_x}),
            input_core_dims=[
                ('T','features'),  # x
                ('T',),           # y
                ('features',),    # x_test
                (),               # y_test
                ()                # alpha
            ],
            vectorize=True,
            dask='parallelized',
            output_core_dims=[('output',)],  # => [error, prediction]
            output_dtypes=['float'],
            dask_gufunc_kwargs={'output_sizes': {'output': 2}}
        )
        result_ = result_da.compute()
        client.close()
        result_ = result_.isel(output=1)

        # result_ => dims (output=2, Y, X). 
        # For a real future forecast, "error" is NaN, "prediction" is the forecast.

        # 2) Compute thresholds T1, T2 from climatology
        index_start = Predictant.get_index("T").get_loc(str(clim_year_start)).start
        index_end   = Predictant.get_index("T").get_loc(str(clim_year_end)).stop
        rainfall_for_tercile = Predictant.isel(T=slice(index_start, index_end))
        terciles = rainfall_for_tercile.quantile([0.33, 0.67], dim='T')
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

        # 3) Tercile probabilities
        if self.dist_method == "t":
            calc_func = self.calculate_tercile_probabilities
            dof = len(Predictant.get_index("T")) - 2


            hindcast_prob = xr.apply_ufunc(
                calc_func,
                forecast_expanded,
                error_variance,
                T1,
                T2,
                input_core_dims=[('T',), (), (), ()],
                vectorize=True,
                kwargs={'dof': dof},
                dask='parallelized',
                output_core_dims=[('probability','T')],
                output_dtypes=['float'],
                dask_gufunc_kwargs={'output_sizes': {'probability': 3}, "allow_rechunk": True},
            )

        elif self.dist_method == "gamma":
            calc_func = self.calculate_tercile_probabilities_gamma

            hindcast_prob = xr.apply_ufunc(
                calc_func,
                forecast_expanded,
                error_variance,
                T1,
                T2,
                input_core_dims=[('T',), (), (), ()],
                vectorize=True,
                dask='parallelized',
                output_core_dims=[('probability','T')],
                output_dtypes=['float'],
                dask_gufunc_kwargs={'output_sizes': {'probability': 3}}
            )

        elif self.dist_method == "normal":
            calc_func = self.calculate_tercile_probabilities_normal

            hindcast_prob = xr.apply_ufunc(
                calc_func,
                forecast_expanded,
                error_variance,
                T1,
                T2,
                input_core_dims=[('T',), (), (), ()],
                vectorize=True,
                dask='parallelized',
                output_core_dims=[('probability','T')],
                output_dtypes=['float'],
                dask_gufunc_kwargs={'output_sizes': {'probability': 3}}
            )

        elif self.dist_method == "lognormal":
            calc_func = self.calculate_tercile_probabilities_lognormal

            hindcast_prob = xr.apply_ufunc(
                calc_func,
                forecast_expanded,
                error_variance,
                T1,
                T2,
                input_core_dims=[('T',), (), (), ()],
                vectorize=True,
                dask='parallelized',
                output_core_dims=[('probability','T')],
                output_dtypes=['float'],
                dask_gufunc_kwargs={'output_sizes': {'probability': 3}}
            )

        elif self.dist_method == "nonparam":
            calc_func = self.calculate_tercile_probabilities_nonparametric
            error_samples = Predictant - hindcast_det
            error_samples = error_samples.rename({'T':'S'})
            hindcast_prob = xr.apply_ufunc(
                calc_func,
                forecast_expanded,
                error_samples,
                T1,
                T2,
                input_core_dims=[('T',), ('S',), (), ()],
                output_core_dims=[('probability','T')],
                vectorize=True, 
                dask='parallelized',
                output_dtypes=[float],
                dask_gufunc_kwargs={'output_sizes': {'probability': 3}}
            )

        else:
            raise ValueError(
                f"Invalid dist_method: {self.dist_method}. "
                "Choose 't','gamma','normal','lognormal','nonparam'."
            )

        hindcast_prob = hindcast_prob.assign_coords(probability=('probability', ['PB','PN','PA']))
        hindcast_prob_out = hindcast_prob.transpose('probability','T','Y','X') #.drop_vars('T').squeeze()

        # Return [error, prediction] plus tercile probabilities
        return forecast_expanded, hindcast_prob_out


class WAS_ElasticNet_Model:
    """
    A class to implement the ElasticNet model for spatiotemporal regression with clustering, 
    cross-validation, and probabilistic predictions.

    Attributes:
    -----------
    alpha_range : numpy.ndarray
        Range of alpha values (regularization strength) to explore.
        Default is from 10^-6 to 10^2.
    l1_ratio_range : list
        Range of l1 ratio values (mixing between L1 and L2 penalties). 
        Default is [0.1, 0.5, 0.7, 0.9, 0.95, 0.99, 1].
    n_clusters : int
        Number of clusters for KMeans-based spatial grouping. Default is 5.
    nb_cores : int
        Number of CPU cores for parallel computations. Default is 1.
    dist_method : str
        Distribution method for tercile probability calculations 
        (e.g. 'gamma','t','normal','lognormal','nonparam'). Default = "gamma".

    Methods:
    --------
    fit_predict(x, y, x_test, y_test=None, alpha=None, l1_ratio=None)
        Fits an ElasticNet model to training data and predicts values for test data.

    compute_hyperparameters(predictand, predictor)
        Computes the optimal alpha and l1 ratio for each cluster using cross-validation.

    compute_model(X_train, y_train, X_test, y_test, alpha, l1_ratio)
        Performs parallelized ElasticNet modeling for spatiotemporal data.

    calculate_tercile_probabilities(...)
        Various static methods for computing tercile probabilities (t, gamma, normal, etc.).

    compute_prob(Predictant, clim_year_start, clim_year_end, Predictor, hindcast_det)
        Computes probabilistic hindcasts for tercile categories based on climatological terciles.

    forecast(Predictant, clim_year_start, clim_year_end, Predictor, hindcast_det, Predictor_for_year, alpha, l1_ratio)
        Generates a single-year forecast using ElasticNet with alpha + l1_ratio, 
        and computes tercile probabilities based on the chosen method.
    """

    def __init__(
        self, 
        alpha_range=None, 
        l1_ratio_range=None, 
        n_clusters=5, 
        nb_cores=1, 
        dist_method="gamma"
    ):
        if alpha_range is None:
            alpha_range = np.array([10**i for i in range(-6, 3)])
        if l1_ratio_range is None:
            l1_ratio_range = [0.1, 0.5, 0.7, 0.9, 0.95, 0.99, 1.0]
        self.alpha_range = alpha_range
        self.l1_ratio_range = l1_ratio_range
        self.n_clusters = n_clusters
        self.nb_cores = nb_cores
        self.dist_method = dist_method
    
    def fit_predict(self, x, y, x_test, y_test=None, alpha=None, l1_ratio=None):
        """
        Fits an ElasticNet model to training data and predicts on test data.

        Parameters
        ----------
        x : ndarray, shape (n_samples, n_features)
        y : ndarray, shape (n_samples,)
        x_test : ndarray, shape (n_features,) or (1, n_features)
        y_test : float or None
        alpha : float
        l1_ratio : float

        Returns
        -------
        np.ndarray
            [error, prediction] if y_test is provided;
            [prediction] if y_test is None.
        """

        if alpha is None or np.isnan(alpha):
            alpha = 0.1  # or any small float you prefer
        if l1_ratio is None or np.isnan(l1_ratio):
            l1_ratio = 0.5
    
        model = linear_model.ElasticNet(alpha=alpha, l1_ratio=l1_ratio)
        mask = np.isfinite(y) & np.all(np.isfinite(x), axis=-1)
        
        # If y_test is given, we return [error, prediction]
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
            # If y_test is None, we only return [prediction]
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
        """
        Computes the optimal alpha and l1 ratio for each cluster using cross-validation.
        """
        predictor['T'] = predictand['T']
        
        kmeans = KMeans(n_clusters=self.n_clusters)
        predictand_dropna = (
            predictand.to_dataframe()
            .reset_index()
            .dropna()
            .drop(columns=['T'])
        )
        # We'll cluster on the 3rd column if it exists
        col_name = predictand_dropna.columns[2]
        predictand_dropna['cluster'] = kmeans.fit_predict(
            predictand_dropna[col_name].to_frame()
        )
        
        df_unique = predictand_dropna.drop_duplicates(subset=['Y', 'X'])
        dataset = df_unique.set_index(['Y', 'X']).to_xarray()
        
        Cluster = (
            dataset['cluster'] *
            xr.where(~np.isnan(predictand.isel(T=0)), 1, np.nan)
        ).drop_vars("T", errors='ignore')

        xarray1, xarray2 = xr.align(predictand, Cluster)
        clusters = np.unique(xarray2)
        clusters = clusters[~np.isnan(clusters)]

        # For each cluster, get mean time series -> do cross-validation
        cluster_means = {
            int(cval): xarray1.where(xarray2 == cval).mean(dim=['Y','X'], skipna=True)
            for cval in clusters
        }

        model_cv = linear_model.ElasticNetCV(
            alphas=self.alpha_range, 
            l1_ratio=self.l1_ratio_range, 
            cv=5
        )

        alpha_l1_cluster = {}
        for cval in clusters:
            c_int = int(cval)
            y_cluster = cluster_means[c_int].dropna(dim='T')
            if len(y_cluster['T']) == 0:
                continue
            # Align predictor
            predictor_cluster = predictor.sel(T=y_cluster['T'])
            X_mat = predictor_cluster.values
            y_vec = y_cluster.values
            model_cv.fit(X_mat, y_vec)
            alpha_l1_cluster[c_int] = (model_cv.alpha_, model_cv.l1_ratio_)

        # Create alpha and l1_ratio arrays
        alpha_array = Cluster.copy()
        l1_ratio_array = Cluster.copy()
        for key, (best_a, best_l1) in alpha_l1_cluster.items():
            alpha_array = alpha_array.where(alpha_array != key, other=best_a)
            l1_ratio_array = l1_ratio_array.where(l1_ratio_array != key, other=best_l1)

        alpha_array, l1_ratio_array, Cluster, predictand = xr.align(
            alpha_array, l1_ratio_array, Cluster, predictand, join="outer"
        )       
        return alpha_array, l1_ratio_array, Cluster

    def compute_model(self, X_train, y_train, X_test, y_test, alpha, l1_ratio):
        """
        Performs parallelized ElasticNet modeling for spatiotemporal data.
        Returns [error, prediction] per grid cell.
        """
        chunksize_x = int(np.round(len(y_train.get_index("X")) / self.nb_cores))
        chunksize_y = int(np.round(len(y_train.get_index("Y")) / self.nb_cores))
        
        X_train['T'] = y_train['T']
        y_train = y_train.transpose('T','Y','X')
        X_test = X_test.squeeze()
        y_test = y_test.drop_vars('T').squeeze().transpose('Y','X')
        
        # Align alpha, l1_ratio, y_train, y_test
        y_train, alpha = xr.align(y_train, alpha)
        y_test, alpha = xr.align(y_test, alpha)
        l1_ratio, alpha = xr.align(l1_ratio, alpha)

        client = Client(n_workers=self.nb_cores, threads_per_worker=1)
        result_da = xr.apply_ufunc(
            self.fit_predict,
            X_train,
            y_train.chunk({'Y': chunksize_y, 'X': chunksize_x}),
            X_test,
            y_test.chunk({'Y': chunksize_y, 'X': chunksize_x}),
            alpha.chunk({'Y': chunksize_y, 'X': chunksize_x}),
            l1_ratio.chunk({'Y': chunksize_y, 'X': chunksize_x}),
            input_core_dims=[
                ('T','features'),  # x
                ('T',),           # y
                ('features',),    # x_test
                (),               # y_test
                (),               # alpha
                ()                # l1_ratio
            ],
            vectorize=True,
            dask='parallelized',
            output_core_dims=[('output',)],
            output_dtypes=['float'],
            dask_gufunc_kwargs={'output_sizes': {'output': 2}},
        )
        result_ = result_da.compute()
        client.close()
        return result_.isel(output=1)

    # ---------------- Probability Calculation Methods ----------------

    @staticmethod
    def calculate_tercile_probabilities(best_guess, error_variance, first_tercile, second_tercile, dof):
        """
        Student's t-based method for tercile probabilities.
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
    def calculate_tercile_probabilities_gamma(best_guess, error_variance, T1, T2):
        """
        Gamma-based method.
        """
        n_time = len(best_guess)
        pred_prob = np.empty((3, n_time), dtype=float)
        if np.any(np.isnan(best_guess)) or np.any(np.isnan(error_variance)):
            pred_prob[:] = np.nan
            return pred_prob

        best_guess = np.asarray(best_guess, dtype=float)
        error_variance = np.asarray(error_variance, dtype=float)
        T1 = np.asarray(T1, dtype=float)
        T2 = np.asarray(T2, dtype=float)

        alpha = (best_guess**2) / error_variance
        theta = error_variance / best_guess
        cdf_t1 = gamma.cdf(T1, a=alpha, scale=theta)
        cdf_t2 = gamma.cdf(T2, a=alpha, scale=theta)
        pred_prob[0, :] = cdf_t1
        pred_prob[1, :] = cdf_t2 - cdf_t1
        pred_prob[2, :] = 1.0 - cdf_t2
        return pred_prob

    @staticmethod
    def calculate_tercile_probabilities_nonparametric(best_guess, error_samples, first_tercile, second_tercile):
        """
        Non-parametric method (requires historical errors).
        """
        n_time = len(best_guess)
        pred_prob = np.full((3, n_time), np.nan, dtype=float)

        for t in range(n_time):
            if np.isnan(best_guess[t]):
                continue

            dist = best_guess[t] + error_samples  
            dist = dist[np.isfinite(dist)]  
            if len(dist) == 0:
                continue

            p_below   = np.mean(dist < first_tercile)
            p_between = np.mean((dist >= first_tercile) & (dist < second_tercile))
            p_above   = 1.0 - (p_below + p_between)

            pred_prob[0, t] = p_below
            pred_prob[1, t] = p_between
            pred_prob[2, t] = p_above
        return pred_prob

    @staticmethod
    def calculate_tercile_probabilities_normal(best_guess, error_variance, first_tercile, second_tercile):
        """
        Normal-based method using the Gaussian CDF.
        """
        n_time = len(best_guess)
        pred_prob = np.empty((3, n_time))
        
        if np.all(np.isnan(best_guess)):
            pred_prob[:] = np.nan
        else:
            error_std = np.sqrt(error_variance)
            pred_prob[0, :] = stats.norm.cdf(first_tercile, loc=best_guess, scale=error_std)
            pred_prob[1, :] = stats.norm.cdf(second_tercile, loc=best_guess, scale=error_std) - \
                              stats.norm.cdf(first_tercile, loc=best_guess, scale=error_std)
            pred_prob[2, :] = 1 - stats.norm.cdf(second_tercile, loc=best_guess, scale=error_std)
        return pred_prob

    @staticmethod
    def calculate_tercile_probabilities_lognormal(best_guess, error_variance, first_tercile, second_tercile):
        """
        Lognormal-based method.
        """
        n_time = len(best_guess)
        pred_prob = np.empty((3, n_time))
        
        if np.any(np.isnan(best_guess)) or np.any(np.isnan(error_variance)):
            pred_prob[:] = np.nan
            return pred_prob

        sigma = np.sqrt(np.log(1 + error_variance / (best_guess**2)))
        mu = np.log(best_guess) - sigma**2 / 2
        pred_prob[0, :] = lognorm.cdf(first_tercile, s=sigma, scale=np.exp(mu))
        pred_prob[1, :] = lognorm.cdf(second_tercile, s=sigma, scale=np.exp(mu)) - \
                          lognorm.cdf(first_tercile, s=sigma, scale=np.exp(mu))
        pred_prob[2, :] = 1 - lognorm.cdf(second_tercile, s=sigma, scale=np.exp(mu))
        return pred_prob

    def compute_prob(self, Predictant, clim_year_start, clim_year_end, hindcast_det):
        """
        Compute tercile probabilities using self.dist_method.

        Parameters
        ----------
        Predictant : xarray.DataArray (T, Y, X)
            Observed data.
        clim_year_start : int
        clim_year_end : int
            The start and end years for the climatology.
        hindcast_det : xarray.DataArray
            Deterministic forecast with dims (output=2, T, Y, X).

        Returns
        -------
        hindcast_prob : xarray.DataArray
            dims (probability=3, T, Y, X) => [PB, PN, PA].
        """
        # 1) Identify climatology slice
        index_start = Predictant.get_index("T").get_loc(str(clim_year_start)).start
        index_end   = Predictant.get_index("T").get_loc(str(clim_year_end)).stop
        rainfall_for_tercile = Predictant.isel(T=slice(index_start, index_end))
        terciles = rainfall_for_tercile.quantile([0.33, 0.67], dim='T')
        error_variance = (Predictant - hindcast_det).var(dim='T')

        T1 = terciles.isel(quantile=0).drop_vars('quantile')
        T2 = terciles.isel(quantile=1).drop_vars('quantile')

        # 2) Distinguish distribution method
        if self.dist_method == "t":
            dof = len(Predictant.get_index("T")) - 2
            calc_func = self.calculate_tercile_probabilities
            hindcast_prob = xr.apply_ufunc(
                calc_func,
                hindcast_det,
                error_variance,
                T1,
                T2,
                input_core_dims=[('T',), (), (), ()],
                vectorize=True,
                kwargs={'dof': dof},
                dask='parallelized',
                output_core_dims=[('probability','T')],
                output_dtypes=['float'],
                dask_gufunc_kwargs={'output_sizes': {'probability': 3}, "allow_rechunk": True},
            )

        elif self.dist_method == "gamma":
            calc_func = self.calculate_tercile_probabilities_gamma
            hindcast_prob = xr.apply_ufunc(
                calc_func,
                hindcast_det,
                error_variance,
                T1,
                T2,
                input_core_dims=[('T',), (), (), ()],
                vectorize=True,
                dask='parallelized',
                output_core_dims=[('probability','T')],
                output_dtypes=['float'],
                dask_gufunc_kwargs={'output_sizes': {'probability': 3}, "allow_rechunk":True},
            )

        elif self.dist_method == "normal":
            calc_func = self.calculate_tercile_probabilities_normal
            hindcast_prob = xr.apply_ufunc(
                calc_func,
                hindcast_det,
                error_variance,
                T1,
                T2,
                input_core_dims=[('T',), (), (), ()],
                vectorize=True,
                dask='parallelized',
                output_core_dims=[('probability','T')],
                output_dtypes=['float'],
                dask_gufunc_kwargs={'output_sizes': {'probability': 3}, "allow_rechunk":True},
            )

        elif self.dist_method == "lognormal":
            calc_func = self.calculate_tercile_probabilities_lognormal
            hindcast_prob = xr.apply_ufunc(
                calc_func,
                hindcast_det,
                error_variance,
                T1,
                T2,
                input_core_dims=[('T',), (), (), ()],
                vectorize=True,
                dask='parallelized',
                output_core_dims=[('probability','T')],
                output_dtypes=['float'],
                dask_gufunc_kwargs={'output_sizes': {'probability': 3}, "allow_rechunk":True},
            )

        elif self.dist_method == "nonparam":
            calc_func = self.calculate_tercile_probabilities_nonparametric
            error_samples = (Predictant - hindcast_det)
            hindcast_prob = xr.apply_ufunc(
                calc_func,
                hindcast_det,
                error_samples,
                T1,
                T2,
                input_core_dims=[('T',), ('T',), (), ()],
                output_core_dims=[('probability','T')],
                vectorize=True,
                dask='parallelized',
                output_dtypes=[float],
                dask_gufunc_kwargs={'output_sizes': {'probability': 3}},
            )

        else:
            raise ValueError(f"Invalid dist_method: {self.dist_method}. "
                             "Must be one of ['t','gamma','normal','lognormal','nonparam'].")

        hindcast_prob = hindcast_prob.assign_coords(probability=('probability', ['PB','PN','PA']))
        return hindcast_prob.transpose('probability','T','Y','X')

    # ------------------- FORECAST METHOD -------------------
    def forecast(
        self,
        Predictant,
        clim_year_start,
        clim_year_end,
        Predictor,
        hindcast_det,
        Predictor_for_year,
        alpha,
        l1_ratio
    ):
        """
        Generates a single-year forecast using ElasticNet with (alpha, l1_ratio), 
        and computes tercile probabilities.

        Parameters
        ----------
        Predictant : xarray.DataArray
            Observed data (T, Y, X).
        clim_year_start, clim_year_end : int
            Climatology reference period.
        Predictor : xarray.DataArray
            Historical predictor data (T, features).
        hindcast_det : xarray.DataArray
            Historical deterministic forecast with dims (output=2, T, Y, X).
        Predictor_for_year : xarray.DataArray
            Single-year predictor data (features,).
        alpha : xarray.DataArray
            Spatial map of alpha values (Y, X).
        l1_ratio : xarray.DataArray
            Spatial map of l1_ratio values (Y, X).

        Returns
        -------
        result_ : xarray.DataArray
            dims (output=2, Y, X) => [error, forecast].
            For a real forecast scenario, 'error' is NaN.
        hindcast_prob : xarray.DataArray
            dims (probability=3, Y, X) => [PB, PN, PA].
        """
        # 1) Create a dummy y_test => shape (Y, X) with NaN
        y_test_dummy = xr.full_like(Predictant.isel(T=0), np.nan)

        # 2) Align shapes
        Predictor['T'] = Predictant['T']
        Predictant = Predictant.transpose('T','Y','X')
        Predictor_for_year_ = Predictor_for_year.squeeze()

        # Align alpha, l1_ratio with domain
        Predictant, alpha, l1_ratio = xr.align(Predictant, alpha, l1_ratio, join="outer")

        # 3) Parallel fit+predict => shape (2, Y, X)
        client = Client(n_workers=self.nb_cores, threads_per_worker=1)
        result_da = xr.apply_ufunc(
            self.fit_predict,
            Predictor,
            Predictant.chunk({'Y': int(np.round(len(Predictant.get_index("Y")) / self.nb_cores)),
                              'X': int(np.round(len(Predictant.get_index("X")) / self.nb_cores))}),
            Predictor_for_year_,
            y_test_dummy,   # pass dummy test => yields [NaN, forecast]
            alpha,
            l1_ratio,
            input_core_dims=[
                ('T','features'),  # x
                ('T',),           # y
                ('features',),    # x_test
                (),               # y_test
                (),               # alpha
                ()                # l1_ratio
            ],
            vectorize=True,
            dask='parallelized',
            output_core_dims=[('output',)],  # => [error, prediction]
            output_dtypes=['float'],
            dask_gufunc_kwargs={'output_sizes': {'output': 2}}
        )
        result_ = result_da.compute()
        client.close()

        result_ = result_.isel(output=1)

        # result_ => dims (output=2, Y, X). 
        # For a real future forecast, "error" is NaN, "prediction" is the forecast.

        # 2) Compute thresholds T1, T2 from climatology
        index_start = Predictant.get_index("T").get_loc(str(clim_year_start)).start
        index_end   = Predictant.get_index("T").get_loc(str(clim_year_end)).stop
        rainfall_for_tercile = Predictant.isel(T=slice(index_start, index_end))
        terciles = rainfall_for_tercile.quantile([0.33, 0.67], dim='T')
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

        # 3) Tercile probabilities
        if self.dist_method == "t":
            calc_func = self.calculate_tercile_probabilities
            dof = len(Predictant.get_index("T")) - 2


            hindcast_prob = xr.apply_ufunc(
                calc_func,
                forecast_expanded,
                error_variance,
                T1,
                T2,
                input_core_dims=[('T',), (), (), ()],
                vectorize=True,
                kwargs={'dof': dof},
                dask='parallelized',
                output_core_dims=[('probability','T')],
                output_dtypes=['float'],
                dask_gufunc_kwargs={'output_sizes': {'probability': 3}, "allow_rechunk": True},
            )

        elif self.dist_method == "gamma":
            calc_func = self.calculate_tercile_probabilities_gamma

            hindcast_prob = xr.apply_ufunc(
                calc_func,
                forecast_expanded,
                error_variance,
                T1,
                T2,
                input_core_dims=[('T',), (), (), ()],
                vectorize=True,
                dask='parallelized',
                output_core_dims=[('probability','T')],
                output_dtypes=['float'],
                dask_gufunc_kwargs={'output_sizes': {'probability': 3}}
            )

        elif self.dist_method == "normal":
            calc_func = self.calculate_tercile_probabilities_normal

            hindcast_prob = xr.apply_ufunc(
                calc_func,
                forecast_expanded,
                error_variance,
                T1,
                T2,
                input_core_dims=[('T',), (), (), ()],
                vectorize=True,
                dask='parallelized',
                output_core_dims=[('probability','T')],
                output_dtypes=['float'],
                dask_gufunc_kwargs={'output_sizes': {'probability': 3}}
            )

        elif self.dist_method == "lognormal":
            calc_func = self.calculate_tercile_probabilities_lognormal

            hindcast_prob = xr.apply_ufunc(
                calc_func,
                forecast_expanded,
                error_variance,
                T1,
                T2,
                input_core_dims=[('T',), (), (), ()],
                vectorize=True,
                dask='parallelized',
                output_core_dims=[('probability','T')],
                output_dtypes=['float'],
                dask_gufunc_kwargs={'output_sizes': {'probability': 3}}
            )

        elif self.dist_method == "nonparam":
            calc_func = self.calculate_tercile_probabilities_nonparametric
            error_samples = Predictant - hindcast_det
            error_samples = error_samples.rename({'T':'S'})
            hindcast_prob = xr.apply_ufunc(
                calc_func,
                forecast_expanded,
                error_samples,
                T1,
                T2,
                input_core_dims=[('T',), ('S',), (), ()],
                output_core_dims=[('probability','T')],
                vectorize=True, 
                dask='parallelized',
                output_dtypes=[float],
                dask_gufunc_kwargs={'output_sizes': {'probability': 3}}
            )

        else:
            raise ValueError(
                f"Invalid dist_method: {self.dist_method}. "
                "Choose 't','gamma','normal','lognormal','nonparam'."
            )

        hindcast_prob = hindcast_prob.assign_coords(probability=('probability', ['PB','PN','PA']))
        hindcast_prob_out = hindcast_prob.transpose('probability','T','Y','X') #.drop_vars('T').squeeze()

        # Return [error, prediction] plus tercile probabilities
        return forecast_expanded, hindcast_prob_out