########  This code was developed by Mandela Houngnibo et al. within the framework of AGRHYMET WAS-RCC S2S. #################### Version 1.0.0 ###########################################
###################################################################################



######################################################## Modules ########################################################
from sklearn import linear_model
from sklearn.linear_model import LinearRegression
from sklearn.linear_model import QuantileRegressor
from sklearn.model_selection import GridSearchCV
import statsmodels.api as sm
from statsmodels.stats.outliers_influence import variance_inflation_factor as VIF
from statsmodels.stats.anova import anova_lm
import xarray as xr 
import numpy as np
import pandas as pd
from scipy import stats
from sklearn.cluster import KMeans

import xeofs as xe
import xarray as xr
import numpy as np
import scipy.signal as sig
from scipy.interpolate import CubicSpline

from multiprocessing import cpu_count
from dask.distributed import Client
import dask.array as da


# ######################################################################################################################################################################################   Linear regression model ######################################

# class WAS_LinearRegression_Model:
#     """
#     A class to perform linear regression modeling on spatiotemporal datasets for climate prediction.

#     This class is designed to work with Dask and Xarray for parallelized, high-performance 
#     regression computations across large datasets with spatial and temporal dimensions. The primary 
#     methods are for fitting the model, making predictions, and calculating probabilistic predictions 
#     for climate terciles. By Mandela HOUNGNIBO

#     Attributes
#     ----------
#     nb_cores : int, optional
#         The number of CPU cores to use for parallel computation (default is 1).
    
#     Methods
#     -------
    
#     fit_predict(x, y, x_test, y_test)
#         Fits a linear regression model to the training data, predicts on test data, and computes error.
    
#     compute_model(X_train, y_train, X_test, y_test)
#         Applies the linear regression model across a dataset using parallel computation with Dask, 
#         returning predictions and error metrics.
    
#     calculate_tercile_probabilities(best_guess, error_variance, first_tercile, second_tercile, dof)
#         Calculates the probabilities for three tercile categories (below-normal, normal, above-normal) 
#         based on predictions and associated error variance.
    
#     compute_prob(Predictant, clim_year_start, clim_year_end, Predictor, hindcast_det)
#         Computes tercile probabilities for hindcast rainfall predictions over specified climatological 
#         years.
#     """
#     def __init__(self, nb_cores=1):
#         """
#         Initializes the WAS_LinearRegression_Model with a specified number of CPU cores.
        
#         Parameters
#         ----------
#         nb_cores : int, optional
#             Number of CPU cores to use for parallel computation, by default 1.
#         """
#         self.nb_cores = nb_cores
    
#     def fit_predict(self, x, y, x_test, y_test):
#         """
#         Fits a linear regression model to the provided training data, makes predictions on the test data, 
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
#         model = linear_model.LinearRegression()
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
#             return np.array([np.nan, np.nan]).squeeze()
    
#     def compute_model(self, X_train, y_train, X_test, y_test):
#         """
#         Computes predictions for spatiotemporal data using linear regression with parallel processing.

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


####################################################################################################################################################################################################################################################
                                     # PCR and CCA TO RESOLVE MULTICOLINEARITY #
####################################################################################################################################################################################################################################################

# class WAS_EOF:
#     """
#     A class for performing Empirical Orthogonal Function (EOF) analysis using the xeofs package, 
#     with additional options for detrending and cosine latitude weighting.

#     Parameters
#     ----------
#     n_modes : int, optional
#         The number of EOF modes to retain. If None, the number of modes is determined by 
#         explained variance.
#     use_coslat : bool, optional
#         If True, applies cosine latitude weighting to account for the Earth's spherical geometry.
#     standardize : bool, optional
#         If True, standardizes the input data by removing the mean and dividing by the standard deviation.
#     detrend : bool, optional
#         If True, detrends the input data along the time dimension before performing EOF analysis.
#     opti_explained_variance : float, optional
#         The target cumulative explained variance (in percent) to determine the optimal number of EOF modes.
#     L2norm : bool, optional
#         If True, normalizes the components and scores to have L2 norm.

#     Attributes
#     ----------
#     model : xeofs.models.EOF
#         The EOF model fitted to the predictor data.
#     """

#     def __init__(self, n_modes=None, use_coslat=True, standardize=False, detrend=True, opti_explained_variance=None, L2norm=True):
#         self.n_modes = n_modes
#         self.use_coslat = use_coslat
#         self.standardize = standardize
#         self.detrend = detrend
#         self.opti_explained_variance = opti_explained_variance
#         self.L2norm = L2norm
#         self.model = None

#     def _detrend(self, predictor):
#         """
#         Detrend the input predictor data along the time axis.

#         Parameters
#         ----------
#         predictor : xarray.DataArray
#             The predictor data to detrend, with time as one of the dimensions.

#         Returns
#         -------
#         predictor_detrend : xarray.DataArray
#             Detrended predictor data, with the same dimensions and coordinates as the input.
#         """
#         predictor_ = predictor.fillna(0)
#         predictor_detrend = sig.detrend(predictor_, axis=0)
#         predictor_detrend = xr.DataArray(predictor_detrend, dims=predictor_.dims, coords=predictor_.coords)
#         return predictor_detrend

#     def fit(self, predictor, dim="T"):
#         """
#         Fit the EOF model to the predictor data.

#         Parameters
#         ----------
#         predictor : xarray.DataArray
#             The predictor data to fit the EOF model, with time as one of the dimensions.
#         dim : str, optional
#             The dimension along which to apply the EOF analysis, typically the time dimension (default is 'T').

#         Returns
#         -------
#         s_eofs : xarray.DataArray
#             The spatial patterns (EOFs).
#         s_pcs : xarray.DataArray
#             The temporal patterns (principal components).
#         s_expvar : numpy.ndarray
#             The explained variance for each EOF mode.
#         s_sing_values : numpy.ndarray
#             The singular values from the EOF decomposition.
#         """
#         predictor = predictor.rename({"X": "lon", "Y": "lat"})

#         # Apply detrending if specified
#         if self.detrend:
#             predictor = self._detrend(predictor)

#         # Initialize the EOF model based on the number of modes or explained variance
#         if self.n_modes is not None:
#             self.model = xe.single.EOF(n_modes=self.n_modes, use_coslat=self.use_coslat, standardize=self.standardize)
#         else:
#             self.model = xe.single.EOF(use_coslat=self.use_coslat, standardize=self.standardize)
#             self.model.fit(predictor, dim=dim)

#             # Determine the number of modes based on the cumulative explained variance
#             if self.opti_explained_variance is not None:
#                 npcs = 0
#                 sum_explain_var = 0
#                 while sum_explain_var * 100 < self.opti_explained_variance:
#                     npcs += 1
#                     sum_explain_var = sum(self.model.explained_variance_ratio()[:npcs])
#                 self.model = xe.single.EOF(n_modes=npcs, use_coslat=self.use_coslat, standardize=self.standardize)

#         # Fit the model on the predictor data
#         self.model.fit(predictor, dim=dim)

#         # Return components, scores, explained variance, and singular values
#         s_eofs = self.model.components(normalized=self.L2norm)
#         s_pcs = self.model.scores(normalized=self.L2norm)
#         s_expvar = self.model.explained_variance_ratio()
#         s_sing_values = self.model.singular_values()

#         return s_eofs, s_pcs, s_expvar, s_sing_values

#     def transform(self, predictor):
#         """
#         Transform new predictor data into the EOF space (principal components).

#         Parameters
#         ----------
#         predictor : xarray.DataArray
#             The new predictor data to transform into the EOF space.

#         Returns
#         -------
#         pcs : xarray.DataArray
#             The principal components of the new predictor data.
#         """
#         predictor = predictor.rename({"X": "lon", "Y": "lat"})

#         # # Apply detrending if specified
#         # if self.detrend:
#         #     predictor = self._detrend(predictor)

#         # Ensure that the model is already fitted
#         if self.model is None:
#             raise ValueError("The model has not been fitted yet.")

#         return self.model.transform(predictor, normalized=self.L2norm)

#     def inverse_transform(self, pcs):
#         """
#         Reconstruct the original predictor data from the principal components.

#         Parameters
#         ----------
#         pcs : xarray.DataArray
#             The principal components from which to reconstruct the original predictor data.

#         Returns
#         -------
#         predictor_reconstructed : xarray.DataArray
#             The reconstructed predictor data based on the principal components.
#         """
#         if self.model is None:
#             raise ValueError("The model has not been fitted yet.")

#         return self.model.inverse_transform(pcs, normalized=self.L2norm)

# ##################################### PCR regression ###################################################################

# class WAS_PCR:

#     """
#     A class for performing Principal Component Regression (PCR) using EOF analysis and variable regression models.

#     This class integrates the WAS_EOF for dimensionality reduction through Empirical Orthogonal Function (EOF)
#     analysis and allows the use of different regression models for predicting a target variable based on the
#     principal components.

#     Attributes
#     ----------
#     eof_model : WAS_EOF
#         The EOF analysis model used for dimensionality reduction.
#     reg_model : object
#         A regression model (e.g., WAS_LinearRegression_Model, WAS_Ridge_Model, etc.) used for regression on the PCs.
#     """

#     def __init__(self, regression_model, n_modes=None, use_coslat=True, standardize=False,
#                  detrend=True, opti_explained_variance=None, L2norm=True):
#         """
#         Initializes the WAS_PCR class with EOF and a flexible regression model.

#         Parameters
#         ----------
#         regression_model : object
#             An instance of any regression model class (e.g., WAS_Ridge_Model, WAS_Lasso_Model).
#         n_modes : int, optional
#             Number of EOF modes to retain, passed to WAS_EOF.
#         use_coslat : bool, optional
#             Whether to apply cosine latitude weighting in EOF analysis, passed to WAS_EOF.
#         standardize : bool, optional
#             Whether to standardize the input data, passed to WAS_EOF.
#         detrend : bool, optional
#             Whether to detrend the input data, passed to WAS_EOF.
#         opti_explained_variance : float, optional
#             Target cumulative explained variance to determine the number of EOF modes.
#         L2norm : bool, optional
#             Whether to normalize EOF components and scores to have L2 norm, passed to WAS_EOF.
#         multivariate : bool, optional
#             Whether to perform multivariate EOF analysis.
#         compute_individual : bool, optional
#             Whether to compute separate EOFs for each variable in a multivariate list.
#         """
        
#         self.eof_model = WAS_EOF(n_modes=n_modes, use_coslat=use_coslat, standardize=standardize, detrend=detrend,
#                                  opti_explained_variance=opti_explained_variance, L2norm=L2norm)
        
#         self.reg_model = regression_model  # Set the regression model passed as an argument

#     def compute_model(self, X_train, y_train, X_test, y_test=None, alpha=None, l1_ratio=None, **kwargs):
#         s_eofs, s_pcs, _, _ = self.eof_model.fit(X_train, dim="T")
#         X_train_pcs = s_pcs.rename({"mode": "features"}).transpose('T', 'features')
#         X_test_pcs = self.eof_model.transform(X_test).rename({"mode": "features"}).transpose('T', 'features')
#         if isinstance(self.reg_model, WAS_LinearRegression_Model): 
#             result = self.reg_model.compute_model(X_train_pcs, y_train, X_test_pcs, y_test)
#         if isinstance(self.reg_model, (WAS_Ridge_Model, WAS_Lasso_Model, WAS_LassoLars_Model)):
#             result = self.reg_model.compute_model(X_train_pcs, y_train, X_test_pcs, y_test, alpha)
#         if isinstance(self.reg_model, WAS_ElasticNet_Model): 
#             result = self.reg_model.compute_model(X_train_pcs, y_train, X_test_pcs, y_test, alpha, l1_ratio)
#         if isinstance(self.reg_model, WAS_LogisticRegression_Model): 
#             result = self.reg_model.compute_model(X_train_pcs, y_train, X_test_pcs, alpha)
#         if isinstance(self.reg_model, WAS_QuantileRegression_Model): 
#             result = self.reg_model.compute_model(X_train_pcs, y_train, X_test_pcs, alpha)
#         #if isinstance():
#         return result

#     def compute_prob(self, Predictant, clim_year_start, clim_year_end, Predictor, hindcast_det):
#         _, s_pcs, _, _ = self.eof_model.fit(Predictor, dim="T")
#         Predictor = s_pcs.rename({"mode": "features"}).transpose('T', 'features')
#         if isinstance(self.reg_model, (WAS_LinearRegression_Model, WAS_Ridge_Model, WAS_Lasso_Model, WAS_LassoLars_Model, WAS_ElasticNet_Model)):
#             result = self.reg_model.compute_prob(Predictant, clim_year_start, clim_year_end, Predictor, hindcast_det)
#         if isinstance(self.reg_model, WAS_LogisticRegression_Model): 
#             result = None
#         if isinstance(self.reg_model, WAS_QuantileRegression_Model): 
#             result = self.reg_model.compute_prob(Predictant, clim_year_start, clim_year_end, Predictor, hindcast_det)
#         return result



##################################### CCA regression ##########################################################################

# class WAS_CCA:

#         def __init__(self, regression_model, n_modes=None, use_coslat=False, standardize=False,
#                      use_pca=False, n_pca_modes=0.95, detrend=True, L2norm=True):
#             "blbla"
            

            
            
####################################################################################################################################################################################################################################################
                                     # REGULARIZATION TO RESOLVE MULTICOLINEARITY #
####################################################################################################################################################################################################################################################

# class WAS_Ridge_Model:
#     def __init__(self, alpha_range=None, n_clusters=5, nb_cores=1):
#         if alpha_range is None:
#             alpha_range = np.logspace(-10, 10, 100)
#         self.alpha_range = alpha_range
#         self.n_clusters = n_clusters
#         self.nb_cores = nb_cores
    
#     def fit_predict(self, x, y, x_test, y_test, alpha):
#         model = linear_model.Ridge(alpha)
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
#             return np.array([np.nan, np.nan]).squeeze()
    
#     def compute_hyperparameters(self, predictand, predictor):
#         kmeans = KMeans(n_clusters=self.n_clusters)
#         predictand_dropna = predictand.to_dataframe().reset_index().dropna().drop(columns=['T'])
#         predictand_dropna['cluster'] = kmeans.fit_predict(predictand_dropna[predictand_dropna.columns[2]].to_frame())
        
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
        
#         model = linear_model.RidgeCV(alphas=self.alpha_range, cv=5)
#         alpha_cluster = {
#             int(cluster): model.fit(predictor, cluster_means[cluster]).alpha_
#             for cluster in clusters
#         }
        
#         alpha_array = Cluster.copy()
#         for key, value in alpha_cluster.items():
#             alpha_array = alpha_array.where(alpha_array != key, other=value)
        
#         return alpha_array, Cluster

#     def compute_model(self, X_train, y_train, X_test, y_test, alpha):
#         chunksize_x = np.round(len(y_train.get_index("X")) / self.nb_cores)
#         chunksize_y = np.round(len(y_train.get_index("Y")) / self.nb_cores)
        
#         X_train['T'] = y_train['T']
#         y_train = y_train.transpose('T', 'Y', 'X')
#         X_test = X_test.squeeze()
#         y_test = y_test.drop_vars('T').squeeze().transpose('Y', 'X')
#         y_train, alpha =  xr.align(y_train, alpha)
#         y_test, alpha =  xr.align(y_test, alpha)
#         # alpha = alpha.transpose('Y', 'X')
#         client = Client(n_workers=self.nb_cores, threads_per_worker=1)
#         result = xr.apply_ufunc(
#             self.fit_predict,
#             X_train,
#             y_train.chunk({'Y': chunksize_y, 'X': chunksize_x}),
#             X_test,
#             y_test.chunk({'Y': chunksize_y, 'X': chunksize_x}),
#             alpha.chunk({'Y': chunksize_y, 'X': chunksize_x}),
#             input_core_dims=[('T', 'features'), ('T',), ('features',), (), ()],
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
#         index_start = Predictant.get_index("T").get_loc(str(clim_year_start)).start
#         index_end = Predictant.get_index("T").get_loc(str(clim_year_end)).stop
#         Predictant, hindcast_det =  xr.align(Predictant, hindcast_det)
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
        
# ##################################### Lasso regression ###################################################################

# class WAS_Lasso_Model:
#     def __init__(self, alpha_range=None, n_clusters=5, nb_cores=1):
#         if alpha_range is None:
#             alpha_range = np.array([10**i for i in range(-6, 6)])
#         self.alpha_range = alpha_range
#         self.n_clusters = n_clusters
#         self.nb_cores = nb_cores
    
#     def fit_predict(self, x, y, x_test, y_test, alpha):
#         model = linear_model.Lasso(alpha)
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
#             return np.array([np.nan, np.nan]).squeeze()
    
#     def compute_hyperparameters(self, predictand, predictor):
#         kmeans = KMeans(n_clusters=self.n_clusters)
#         predictand_dropna = predictand.to_dataframe().reset_index().dropna().drop(columns=['T'])
#         predictand_dropna['cluster'] = kmeans.fit_predict(predictand_dropna[predictand_dropna.columns[2]].to_frame())
        
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
        
#         model = linear_model.LassoCV(alphas=self.alpha_range, cv=5)
#         alpha_cluster = {
#             int(cluster): model.fit(predictor, cluster_means[cluster]).alpha_
#             for cluster in clusters
#         }
        
#         alpha_array = Cluster.copy()
#         for key, value in alpha_cluster.items():
#             alpha_array = alpha_array.where(alpha_array != key, other=value)
        
#         return alpha_array, Cluster

#     def compute_model(self, X_train, y_train, X_test, y_test, alpha):
#         chunksize_x = np.round(len(y_train.get_index("X")) / self.nb_cores)
#         chunksize_y = np.round(len(y_train.get_index("Y")) / self.nb_cores)
        
#         X_train['T'] = y_train['T']
#         y_train = y_train.transpose('T', 'Y', 'X')
#         X_test = X_test.squeeze()
#         y_test = y_test.drop_vars('T').squeeze().transpose('Y', 'X')
#         y_train, alpha =  xr.align(y_train, alpha)
#         y_test, alpha =  xr.align(y_test, alpha)
#         # alpha = alpha.transpose('Y', 'X')
#         client = Client(n_workers=self.nb_cores, threads_per_worker=1)        
#         result = xr.apply_ufunc(
#             self.fit_predict,
#             X_train,
#             y_train.chunk({'Y': chunksize_y, 'X': chunksize_x}),
#             X_test,
#             y_test.chunk({'Y': chunksize_y, 'X': chunksize_x}),
#             alpha.chunk({'Y': chunksize_y, 'X': chunksize_x}),
#             input_core_dims=[('T', 'features'), ('T',), ('features',), (), ()],
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
#         index_start = Predictant.get_index("T").get_loc(str(clim_year_start)).start
#         index_end = Predictant.get_index("T").get_loc(str(clim_year_end)).stop
#         Predictant, hindcast_det =  xr.align(Predictant, hindcast_det)
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


# ##################################################   LassoLars ###########################################################

# class WAS_LassoLars_Model:
#     def __init__(self, alpha_range=None, n_clusters=5, nb_cores=1):
#         if alpha_range is None:
#             alpha_range = np.array([10**i for i in range(-6, 6)])
#         self.alpha_range = alpha_range
#         self.n_clusters = n_clusters
#         self.nb_cores = nb_cores
    
#     def fit_predict(self, x, y, x_test, y_test, alpha):
#         model = linear_model.LassoLars(alpha)
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
#             return np.array([np.nan, np.nan]).squeeze()
    
#     def compute_hyperparameters(self, predictand, predictor):
#         kmeans = KMeans(n_clusters=self.n_clusters)
#         predictand_dropna = predictand.to_dataframe().reset_index().dropna().drop(columns=['T'])
#         predictand_dropna['cluster'] = kmeans.fit_predict(predictand_dropna[predictand_dropna.columns[2]].to_frame())
        
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
        
#         model = linear_model.LassoLarsCV()
#         alpha_cluster = {
#             int(cluster): model.fit(predictor, cluster_means[cluster]).alpha_
#             for cluster in clusters
#         }
#         alpha_array = Cluster.copy()
#         for key, value in alpha_cluster.items():
#             alpha_array = alpha_array.where(alpha_array != key, other=value)
        
#         return alpha_array, Cluster

#     def compute_model(self, X_train, y_train, X_test, y_test, alpha):
#         chunksize_x = np.round(len(y_train.get_index("X")) / self.nb_cores)
#         chunksize_y = np.round(len(y_train.get_index("Y")) / self.nb_cores)
        
#         X_train['T'] = y_train['T']
#         y_train = y_train.transpose('T', 'Y', 'X')
#         X_test = X_test.squeeze()
#         y_test = y_test.drop_vars('T').squeeze().transpose('Y', 'X')
#         y_train, alpha =  xr.align(y_train, alpha)
#         y_test, alpha =  xr.align(y_test, alpha)
#         # alpha = alpha.transpose('Y', 'X')
#         client = Client(n_workers=self.nb_cores, threads_per_worker=1)        
#         result = xr.apply_ufunc(
#             self.fit_predict,
#             X_train,
#             y_train.chunk({'Y': chunksize_y, 'X': chunksize_x}),
#             X_test,
#             y_test.chunk({'Y': chunksize_y, 'X': chunksize_x}),
#             alpha.chunk({'Y': chunksize_y, 'X': chunksize_x}),
#             input_core_dims=[('T', 'features'), ('T',), ('features',), (), ()],
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
#         index_start = Predictant.get_index("T").get_loc(str(clim_year_start)).start
#         index_end = Predictant.get_index("T").get_loc(str(clim_year_end)).stop
#         Predictant, hindcast_det =  xr.align(Predictant, hindcast_det)
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


# #################################################    ElasticNet   #######################################################

# class WAS_ElasticNet_Model:
#     def __init__(self, alpha_range=None, l1_ratio_range=None, n_clusters=5, nb_cores=1):
#         if alpha_range is None:
#             alpha_range = np.array([10**i for i in range(-6, 3)])
#         if l1_ratio_range is None:
#             l1_ratio_range = [.1, .5, .7, .9, .95, .99, 1]
#         self.alpha_range = alpha_range
#         self.l1_ratio_range = l1_ratio_range
#         self.n_clusters = n_clusters
#         self.nb_cores = nb_cores
    
#     def fit_predict(self, x, y, x_test, y_test, alpha, l1_ratio):
#         model = linear_model.ElasticNet(alpha=alpha, l1_ratio=l1_ratio)
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
#             return np.array([np.nan, np.nan]).squeeze()
    
#     def compute_hyperparameters(self, predictand, predictor):
#         kmeans = KMeans(n_clusters=self.n_clusters)
#         predictand_dropna = predictand.to_dataframe().reset_index().dropna().drop(columns=['T'])
#         predictand_dropna['cluster'] = kmeans.fit_predict(predictand_dropna[predictand_dropna.columns[2]].to_frame())
        
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
        
#         model = linear_model.ElasticNetCV(alphas=self.alpha_range, l1_ratio=self.l1_ratio_range, cv=5)

#         alpha_cluster = {
#             int(cluster): [model.fit(predictor, cluster_means[cluster]).alpha_, model.fit(predictor, cluster_means[cluster]).l1_ratio_]
#             for cluster in clusters
#             }
#         alpha_array = Cluster.copy()
#         l1_ratio_array = Cluster.copy()
    
#         for key, value in alpha_cluster.items():
#             alpha_array = alpha_array.where(alpha_array != key, other=value[0]) 
#             l1_ratio_array = l1_ratio_array.where(l1_ratio_array != key, other=value[1]) 
        
#         return alpha_array, l1_ratio_array, Cluster

#     def compute_model(self, X_train, y_train, X_test, y_test, alpha, l1_ratio):
#         chunksize_x = np.round(len(y_train.get_index("X")) / self.nb_cores)
#         chunksize_y = np.round(len(y_train.get_index("Y")) / self.nb_cores)
        
#         X_train['T'] = y_train['T']
#         y_train = y_train.transpose('T', 'Y', 'X')
#         X_test = X_test.squeeze()
#         y_test = y_test.drop_vars('T').squeeze().transpose('Y', 'X')
#         y_train, alpha =  xr.align(y_train, alpha)
#         y_test, alpha =  xr.align(y_test, alpha)
#         l1_ratio, alpha =  xr.align(l1_ratio, alpha)
#         # alpha = alpha.transpose('Y', 'X')
#         client = Client(n_workers=self.nb_cores, threads_per_worker=1)        
#         result = xr.apply_ufunc(
#             self.fit_predict,
#             X_train,
#             y_train.chunk({'Y': chunksize_y, 'X': chunksize_x}),
#             X_test,
#             y_test.chunk({'Y': chunksize_y, 'X': chunksize_x}),
#             alpha.chunk({'Y': chunksize_y, 'X': chunksize_x}),
#             l1_ratio.chunk({'Y': chunksize_y, 'X': chunksize_x}),
#             input_core_dims=[('T', 'features'), ('T',), ('features',), (), (), ()],
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
#         index_start = Predictant.get_index("T").get_loc(str(clim_year_start)).start
#         index_end = Predictant.get_index("T").get_loc(str(clim_year_end)).stop
#         Predictant, hindcast_det =  xr.align(Predictant, hindcast_det)
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


#####################################################################################################################################################################   Non-linear Regression ######################################################
##########################################################################################################################


##########################################  Logistic_Regression  #####################################################

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

##########################################     Poisson_Regression  #####################################################




###################################### Multivariate Adaptive Regression Splines (MARS) #########################




######################################################################################################################################################################  NONPARAMETRIC REGRESSION #################################################
##########################################################################################################################


###########################################  Quantile regression ####################################################

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
        
        predict_val = []
        
        for q, alph in zip(pred_quantile,alpha):                    #self.quantiles:
            model = QuantileRegressor(quantile=q, alpha=alph, solver='highs')
            mask = np.isfinite(y) & np.all(np.isfinite(x), axis=-1)
            
            if np.any(mask):
                y_clean = y[mask]
                x_clean = x[mask, :]
                model.fit(x_clean, y_clean)
                
                if x_test.ndim == 1:
                    x_test = x_test.reshape(1, -1)
                preds = model.predict(x_test).squeeze()
                predict_val.append(preds)
                # return np.array([preds]).squeeze()
            else:
                predict_val.append(np.nan) # Return NaNs if no valid data
                # return np.array([np.nan]).squeeze()  # Return NaNs if no valid data
        predict = np.array(predict_val)
        return predict

    def compute_hyperparameters(self, predictand, predictor):
        """
        Compute the optimal alpha values for each cluster and quantile using KMeans clustering.
        
        :param predictand: Target variable for clustering.
        :param predictor: Predictor variable for model fitting.
        :return: Dictionary of alpha arrays, one for each quantile, and the cluster array.
        """
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
                
                model = QuantileRegressor(quantile=quantile, solver='highs')
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
            
        return alpha_values, Cluster

    def compute_model(self, X_train, y_train, X_test, alpha):
        """
        Compute predictions for each quantile on gridded data.
        
        :param X_train: Training data for predictors.
        :param y_train: Training data for the target variable.
        :param X_test: Test data for predictors.
        :param alpha_clusters: Dictionary of alpha arrays for each quantile.
        :return: Predicted quantiles as an xarray DataArray.
        """
        chunksize_x = np.round(len(y_train.get_index("X")) / self.nb_cores)
        chunksize_y = np.round(len(y_train.get_index("Y")) / self.nb_cores)
        
        X_train['T'] = y_train['T']
        y_train = y_train.transpose('T', 'Y', 'X')
        X_test = X_test.transpose('T', 'features').squeeze()
        y_train, alpha__ =  xr.align(y_train, alpha.isel(quantile=0).squeeze())
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
            output_core_dims=[('quantiles',)],
            dask='parallelized',
            output_dtypes=['float'],
            dask_gufunc_kwargs={'output_sizes': {'quantiles': len(self.quantiles)}}
        )
        result_ = result.compute()
        client.close()
        return result_
        
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
###########################################  Polynomial regression ####################################################



######################################################################################################################################################################  Machine learning techniques ###############################################
##########################################################################################################################



######################################################################################################################################################################  Analogues methods 
   ###############################################
##########################################################################################################################
