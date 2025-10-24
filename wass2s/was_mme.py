from __future__ import annotations
from typing import Optional, Tuple
from scipy.optimize import minimize
from scipy.stats import norm
from dask.distributed import Client
from wass2s.utils import *
import numpy as np
import xarray as xr
from dask.distributed import Client
import pandas as pd
import xcast as xc  # 
from scipy import stats
from scipy.stats import norm
from scipy.stats import lognorm
from scipy.stats import norm, logistic, genextreme, gamma as scigamma, weibull_min, laplace, pareto
from scipy.stats import gamma
from scipy.optimize import minimize
from scipy.stats import norm, gamma, lognorm, weibull_min, t
from scipy.optimize import minimize_scalar
from scipy.special import gamma as sp_gamma
from sklearn.neural_network import MLPRegressor
from sklearn.ensemble import StackingRegressor
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor
from sklearn.neural_network import MLPRegressor
from sklearn.model_selection import GridSearchCV, KFold
from sklearn.model_selection import RandomizedSearchCV
from xgboost import XGBRegressor
from sklearn.cluster import KMeans
from hpelm import HPELM
from sklearn.base import BaseEstimator, RegressorMixin
from sklearn.metrics import mean_squared_error
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import RBF, WhiteKernel
# from sklearn.neighbors import KNeighborsRegressor
# from sklearn.tree import DecisionTreeRegressor
from sklearn.svm import SVR
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.ensemble import AdaBoostRegressor
import pymc as pm
import arviz as az
import gc
import operator
import datetime



def process_datasets_for_mme_(rainfall, hdcsted=None, fcsted=None, gcm=True, agroparam=False, ELM_ELR=False, dir_to_save_model=None, best_models=None, scores=None, year_start=None, year_end=None, model=True, month_of_initialization=None, lead_time=None, year_forecast=None):
    
    all_model_hdcst = {}
    all_model_fcst = {}
    if gcm:
        target_prefixes = [model.lower().replace('.prcp', '') for model in best_models]
        scores_organized = {
            model: da for key, da in scores['GROC'].items() 
            for model in best_models if any(key.startswith(prefix) for prefix in target_prefixes)
                        }
        for i in best_models:
            hdcst = load_gridded_predictor(dir_to_save_model, i, year_start, year_end, model=True, month_of_initialization=month_of_initialization, lead_time=lead_time, year_forecast=None)
            all_model_hdcst[i] = hdcst.interp(
                            Y=rainfall.Y,
                            X=rainfall.X,
                            method="linear",
                            kwargs={"fill_value": "extrapolate"}
                        )
            fcst = load_gridded_predictor(dir_to_save_model, i, year_start, year_end, model=True, month_of_initialization=month_of_initialization, lead_time=lead_time, year_forecast=year_forecast)
            all_model_fcst[i] = fcst.interp(
                            Y=rainfall.Y,
                            X=rainfall.X,
                            method="linear",
                            kwargs={"fill_value": "extrapolate"}
                        )
    elif agroparam:
        target_prefixes = [model.split('.')[0].replace('_','').lower() for model in best_models]
        scores_organized = {
            model.split('.')[0].replace('_','').lower(): da for key, da in scores['GROC'].items() 
            for model in best_models if any(key.startswith(prefix) for prefix in target_prefixes)
                        }
        for i in target_prefixes:
            fic = [f for f in list(hdcsted.values()) if i[0:5] in f][0]        
            hdcst = xr.open_dataset(fic).to_array().drop_vars("variable").squeeze("variable")
            hdcst = hdcst.interp(
                            Y=rainfall.Y,
                            X=rainfall.X,
                            method="linear",
                            kwargs={"fill_value": "extrapolate"}
                        )
            all_model_hdcst[i] = myfill(hdcst, rainfall)
            fic = [f for f in list(fcsted.values()) if i[0:5]  in f][0]
            fcst = xr.open_dataset(fic).to_array().drop_vars("variable").squeeze("variable")
            fcst = fcst.interp(
                            Y=rainfall.Y,
                            X=rainfall.X,
                            method="linear",
                            kwargs={"fill_value": "extrapolate"}
                        )
            all_model_fcst[i] = myfill(fcst, rainfall)
    else:

        target_prefixes = [model.replace(model.split('.')[1], '') for model in best_models]

        scores_organized = {
            model: da for key, da in scores['GROC'].items() 
            for model in list(hdcsted.keys()) if any(model.startswith(prefix) for prefix in target_prefixes)
                        }  

        for i in scores_organized.keys():
            all_model_hdcst[i] = hdcsted[i].interp(
                            Y=rainfall.Y,
                            X=rainfall.X,
                            method="linear",
                            kwargs={"fill_value": "extrapolate"}
                        )
            all_model_fcst[i] = fcsted[i].interp(
                            Y=rainfall.Y,
                            X=rainfall.X,
                            method="linear",
                            kwargs={"fill_value": "extrapolate"}
                        )    
    
    # Extract the datasets and keys
    hindcast_det_list = list(all_model_hdcst.values()) 
    forecast_det_list = list(all_model_fcst.values())
    predictor_names = list(all_model_hdcst.keys())    

    mask = xr.where(~np.isnan(rainfall.isel(T=0)), 1, np.nan).drop_vars('T').squeeze()
    mask.name = None
    
    if ELM_ELR:
        # Concatenate along a new dimension ('M') and assign coordinates
        all_model_hdcst = (
            xr.concat(hindcast_det_list, dim='M')
            .assign_coords({'M': predictor_names})  
            .rename({'T': 'S'})                    
            .transpose('S', 'M', 'Y', 'X')         
        )*mask
        
        all_model_fcst = (
            xr.concat(forecast_det_list, dim='M')
            .assign_coords({'M': predictor_names})  
            .rename({'T': 'S'})                    
            .transpose('S', 'M', 'Y', 'X')         
        )*mask
        obs = rainfall.expand_dims({'M':[0]},axis=1)*mask
        # obs = obs.fillna(obs.mean(dim="T"))
    else:
        # Concatenate along a new dimension ('M') and assign coordinates
        all_model_hdcst = (
            xr.concat(hindcast_det_list, dim='M')
            .assign_coords({'M': predictor_names})             
            .transpose('T', 'M', 'Y', 'X')         
        )*mask
        
        all_model_fcst = (
            xr.concat(forecast_det_list, dim='M')
            .assign_coords({'M': predictor_names})                     
            .transpose('T', 'M', 'Y', 'X')         
        )*mask
        obs = rainfall.expand_dims({'M':[0]},axis=1)*mask

    # all_model_hdcst, obs = xr.align(all_model_hdcst, obs) 
    return all_model_hdcst, all_model_fcst, obs, scores_organized

def process_datasets_for_mme(rainfall, hdcsted=None, fcsted=None, 
                             gcm=False, agroparam=False, Prob=False, hydro=False,
                             ELM_ELR=False, dir_to_save_model=None,
                             best_models=None, scores=None,
                             year_start=None, year_end=None, 
                             model=False, month_of_initialization=None, 
                             lead_time=None, year_forecast=None, 
                             score_metric='GROC', var="PRCP"):
    """
    Process hindcast and forecast datasets for a multi-model ensemble.

    This function loads, interpolates, and concatenates hindcast and forecast datasets from various sources 
    (GCMs, agroparameters, or others) to prepare them for a multi-model ensemble. It supports different score 
    metrics and configurations for probabilistic or deterministic outputs.

    Parameters
    ----------
    rainfall : xarray.DataArray
        Observed rainfall data used for interpolation and masking.
    hdcsted : dict, optional
        Dictionary of hindcast datasets for different models.
    fcsted : dict, optional
        Dictionary of forecast datasets for different models.
    gcm : bool, optional
        If True, process data as GCM data. Default is True.
    agroparam : bool, optional
        If True, process data as agroparameter data. Default is False.
    Prob : bool, optional
        If True, process data as probabilistic forecasts. Default is False.
    ELM_ELR : bool, optional
        If True, use ELM_ELR configuration for dimension renaming. Default is False.
    dir_to_save_model : str, optional
        Directory path to load model data.
    best_models : list, optional
        List of model names to include in the ensemble.
    scores : dict, optional
        Dictionary containing model scores, with the key specified by `score_metric`.
    year_start : int, optional
        Starting year for the data range.
    year_end : int, optional
        Ending year for the data range.
    model : bool, optional
        If True, treat data as model-based. Default is True.
    month_of_initialization : int, optional
        Month when the forecast is initialized.
    lead_time : int, optional
        Forecast lead time in months.
    year_forecast : int, optional
        Year for which the forecast is generated.
    score_metric : str, optional
        Metric used to organize scores (e.g., 'Pearson', 'MAE', 'GROC'). Default is 'GROC'.
    var: str, optional
        variables used ( e.g., 'PRCP')
    Returns
    -------
    all_model_hdcst : xarray.DataArray
        Concatenated hindcast data across models.
    all_model_fcst : xarray.DataArray
        Concatenated forecast data across models.
    obs : xarray.DataArray
        Observed rainfall data expanded with a model dimension and masked.
    scores_organized : dict
        Dictionary of organized scores for selected models.
    """

    all_model_hdcst = {}
    all_model_fcst = {}
    
    if gcm:
        # Standardize model keys for matching.
        target_prefixes = [m.lower().replace(f".{var.lower()}", '') for m in best_models]
        # Use the provided score_metric to extract the appropriate scores.
        scores_organized = {
            model: da for key, da in scores[score_metric].items() 
            for model in best_models if any(key.startswith(prefix) for prefix in target_prefixes)
        }
        for m in best_models:
            hdcst = load_gridded_predictor(
                dir_to_save_model, m, year_start, year_end, model=True, 
                month_of_initialization=month_of_initialization, lead_time=lead_time, 
                year_forecast=None
            )
            all_model_hdcst[m] = hdcst.interp(
                Y=rainfall.Y, X=rainfall.X, method="linear", 
                kwargs={"fill_value": "extrapolate"}
            )
            fcst = load_gridded_predictor(
                dir_to_save_model, m, year_start, year_end, model=True, 
                month_of_initialization=month_of_initialization, lead_time=lead_time, 
                year_forecast=year_forecast
            )
            all_model_fcst[m] = fcst.interp(
                Y=rainfall.Y, X=rainfall.X, method="linear", 
                kwargs={"fill_value": "extrapolate"}
            )
    
    elif agroparam:
        target_prefixes = [model.split('.')[0].replace('_','').lower() for model in best_models]
        scores_organized = {
            model.split('.')[0].replace('_','').lower(): da for key, da in scores[score_metric].items() 
            for model in best_models if any(key.startswith(prefix) for prefix in target_prefixes)
                        }
        for i in target_prefixes:
            fic = [f for f in list(hdcsted.values()) if i[0:5] in f][0]        
            hdcst = xr.open_dataset(fic).to_array().drop_vars("variable").squeeze("variable")
            hdcst = hdcst.interp(
                            Y=rainfall.Y,
                            X=rainfall.X,
                            method="linear",
                            kwargs={"fill_value": "extrapolate"}
                        )
            all_model_hdcst[i] = myfill(hdcst, rainfall)
            fic = [f for f in list(fcsted.values()) if i[0:5]  in f][0]
            fcst = xr.open_dataset(fic).to_array().drop_vars("variable").squeeze("variable")
            fcst = fcst.interp(
                            Y=rainfall.Y,
                            X=rainfall.X,
                            method="linear",
                            kwargs={"fill_value": "extrapolate"}
                        )
            all_model_fcst[i] = myfill(fcst, rainfall)

    elif hydro:
        
        if isinstance(hdcsted[list(hdcsted.keys())[0]], xr.DataArray):
            target_prefixes = best_models
            scores_organized = {
                model: da for key, da in scores[score_metric].items() 
                for model in list(hdcsted.keys()) if any(model.startswith(prefix) for prefix in target_prefixes)
            }
            
            for m in scores_organized.keys():
                all_model_hdcst[m] = hdcsted[m]
                all_model_fcst[m] = fcsted[m]
        else:

            target_prefixes = [m.replace('_','').lower() for m in best_models]
            scores_organized = {
                model: da for key, da in scores[score_metric].items() 
                for model in list(hdcsted.keys()) if any(model.startswith(prefix) for prefix in target_prefixes)
            }
            for m in scores_organized.keys():
                hdcst = xr.open_dataset(hdcsted[m])
                hdcst = hdcst['Observation'].astype(float)
                all_model_hdcst[m] = hdcst
    
                fcst = xr.open_dataset(fcsted[m])
                fcst = fcst['Observation'].astype(float)
                all_model_fcst[m] = fcst
    else:
        # target_prefixes = [m.replace(m.split('.')[1], '') for m in best_models]
        target_prefixes = [m.split('.')[0] for m in best_models]
        scores_organized = {
            model: da for key, da in scores[score_metric].items() 
            for model in list(hdcsted.keys()) if any(model.startswith(prefix) for prefix in target_prefixes)
        }
        for m in scores_organized.keys():
            all_model_hdcst[m] = hdcsted[m].interp(
                Y=rainfall.Y, X=rainfall.X, method="linear", 
                kwargs={"fill_value": "extrapolate"}
            )
            all_model_fcst[m] = fcsted[m].interp(
                Y=rainfall.Y, X=rainfall.X, method="linear", 
                kwargs={"fill_value": "extrapolate"}
            )
    
    # Concatenate datasets along the 'M' dimension.
    hindcast_det_list = list(all_model_hdcst.values()) 
    forecast_det_list = list(all_model_fcst.values())
    predictor_names = list(all_model_hdcst.keys())    
    
    # Create a mask based on the rainfall data.
    mask = xr.where(~np.isnan(rainfall.isel(T=0)), 1, np.nan).drop_vars('T').squeeze()
    mask.name = None
    
    if ELM_ELR:
        all_model_hdcst = (
            xr.concat(hindcast_det_list, dim='M')
              .assign_coords({'M': predictor_names})
              .rename({'T': 'S'})
              .transpose('S', 'M', 'Y', 'X')
        ) * mask
        all_model_hdcst = all_model_hdcst.fillna(all_model_hdcst.mean(dim="S", skipna=True))
        all_model_fcst = (
            xr.concat(forecast_det_list, dim='M')
              .assign_coords({'M': predictor_names})
              .rename({'T': 'S'})
              .transpose('S', 'M', 'Y', 'X')
        ) * mask
        all_model_fcst = all_model_fcst.fillna(all_model_hdcst.mean(dim="S", skipna=True))
        obs = rainfall.expand_dims({'M': [0]}, axis=1) * mask
        obs = obs.fillna(obs.mean(dim="T", skipna=True))

    elif Prob:
        all_model_hdcst = (
            xr.concat(hindcast_det_list, dim='M')
              .assign_coords({'M': predictor_names})
              .transpose('probability', 'T', 'M', 'Y', 'X')
        ) * mask
        all_model_hdcst = all_model_hdcst.fillna(all_model_hdcst.mean(dim="T", skipna=True))
        all_model_fcst = (
            xr.concat(forecast_det_list, dim='M')
              .assign_coords({'M': predictor_names})
              .transpose('probability', 'T', 'M', 'Y', 'X')
        ) * mask
        all_model_fcst = all_model_fcst.fillna(all_model_hdcst.mean(dim="T", skipna=True))
        obs = rainfall.expand_dims({'M': [0]}, axis=1) * mask
        obs = obs.fillna(obs.mean(dim="T", skipna=True))

    else:
        all_model_hdcst = (
            xr.concat(hindcast_det_list, dim='M')
              .assign_coords({'M': predictor_names})
              .transpose('T', 'M', 'Y', 'X')
        ) * mask
        all_model_hdcst = all_model_hdcst.fillna(all_model_hdcst.mean(dim="T", skipna=True))
        all_model_fcst = (
            xr.concat(forecast_det_list, dim='M')
              .assign_coords({'M': predictor_names})
              .transpose('T', 'M', 'Y', 'X')
        ) * mask
        all_model_fcst = all_model_fcst.fillna(all_model_hdcst.mean(dim="T", skipna=True))
        obs = rainfall.expand_dims({'M': [0]}, axis=1) * mask
        obs = obs.fillna(obs.mean(dim="T", skipna=True))
    
    return all_model_hdcst, all_model_fcst, obs, scores_organized


def myfill(all_model_fcst, obs):

    """
    Fill missing values in forecast data using random samples from observations.

    This function fills NaN values in the forecast data by randomly sampling values from the observed 
    rainfall data along the time dimension.

    Parameters
    ----------
    all_model_fcst : xarray.DataArray
        Forecast data with dimensions (T, M, Y, X) containing possible NaN values.
    obs : xarray.DataArray
        Observed rainfall data with dimensions (T, Y, X) used for filling NaNs.

    Returns
    -------
    da_filled_random : xarray.DataArray
        Forecast data with NaN values filled using random samples from observations.
    """

    # Suppose all_model_hdcst has dimensions: T, M, Y, X
    da = all_model_fcst
    
    T = da.sizes["T"]
    Y = da.sizes["Y"]
    X = da.sizes["X"]
    
    # Create a DataArray of random T indices with shape (T, M, Y, X)
    # so that each element gets its own random index along T
    random_t_indices_full = xr.DataArray(
        np.random.randint(0, T, size=(T, Y, X)),
        dims=["T", "Y", "X"]
    )
    
    # Use vectorized indexing: for each (T, M, Y, X) location,
    # this picks the value at a random T index for that M, Y, X location.
    random_slices_full = obs.isel(T=random_t_indices_full)
    
    # Fill missing values with these randomly selected values
    da_filled_random = da.fillna(random_slices_full)
    return da_filled_random   




class WAS_mme_Weighted:
    """
    Weighted Multi-Model Ensemble (MME) for climate forecasting.

    This class implements a weighted ensemble approach for combining multiple climate models, 
    supporting both equal weighting and score-based weighting. It also provides methods for 
    computing tercile probabilities using various statistical distributions.

    Parameters
    ----------
    equal_weighted : bool, optional
        If True, use equal weights for all models; otherwise, use score-based weights. Default is False.
    dist_method : str, optional
        Statistical distribution for probability calculations ('t', 'gamma', 'normal', 'lognormal', 
        'weibull_min', 'nonparam'). Default is 'gamma'.
    metric : str, optional
        Performance metric for weighting ('MAE', 'Pearson', 'GROC'). Default is 'GROC'.
    threshold : float, optional
        Threshold for score transformation. Default is 0.5.
    """
    def __init__(self, equal_weighted=False, dist_method="gamma", metric="GROC", threshold=0.5):
        """
        Parameters:
            equal_weighted (bool): If True, use a simple unweighted mean.
            dist_method (str): Distribution method (kept for compatibility).
            metric (str): Score metric name (e.g., 'MAE', 'Pearson', 'GROC').
            threshold (numeric): Threshold value for masking the score.
        """
        self.equal_weighted = equal_weighted
        self.dist_method = dist_method
        self.metric = metric
        self.threshold = threshold

    def transform_score(self, score_array):
        """
        Transform score array based on the chosen metric and threshold.

        For 'MAE', scores below the threshold are set to 1, others to 0. For 'Pearson' or 'GROC', 
        scores above the threshold are set to 1, others to 0.

        Parameters
        ----------
        score_array : xarray.DataArray
            Score array to transform.

        Returns
        -------
        transformed_score : xarray.DataArray
            Transformed score array with binary weights.
        """
        if self.metric.lower() == 'mae':
            return xr.where(
                score_array <= self.threshold,
                1,
                0
            )
        elif self.metric.lower() in ['pearson', 'groc']:
            return xr.where(
                score_array <= self.threshold,
                0, 1
               # xr.where(
               #     score_array <= 0.6,
               #     0.6,
               #     xr.where(score_array <= 0.8, 0.8, 1)
               # )
            )

        else:
            # Default: no masking applied.
            return score_array

    def compute(self, rainfall, hdcst, fcst, scores, complete=False):

        """
        Compute weighted hindcast and forecast using model scores.

        This method calculates weighted averages of hindcast and forecast data based on model scores. 
        If `complete` is True, missing values are filled with unweighted averages.

        Parameters
        ----------
        rainfall : xarray.DataArray
            Observed rainfall data with dimensions (T, Y, X, M).
        hdcst : xarray.DataArray
            Hindcast data with dimensions (T, M, Y, X).
        fcst : xarray.DataArray
            Forecast data with dimensions (T, M, Y, X).
        scores : dict
            Dictionary mapping model names to score arrays.
        complete : bool, optional
            If True, fill missing values with unweighted averages. Default is False.

        Returns
        -------
        hindcast_det : xarray.DataArray
            Weighted hindcast data with dimensions (T, Y, X).
        forecast_det : xarray.DataArray
            Weighted forecast data with dimensions (T, Y, X).
        """

        # Adjust time coordinates as needed.
        year = fcst.coords['T'].values.astype('datetime64[Y]').astype(int)[0] + 1970
        T_value_1 = rainfall.isel(T=0).coords['T'].values
        month_1 = T_value_1.astype('datetime64[M]').astype(int) % 12 + 1
        new_T_value = np.datetime64(f"{year}-{month_1:02d}-01")
        
        fcst = fcst.assign_coords(T=xr.DataArray([new_T_value], dims=["T"]))
        fcst['T'] = fcst['T'].astype('datetime64[ns]')
        hdcst['T'] = rainfall['T'].astype('datetime64[ns]')
        
        # Create a mask based on non-NaN values in the rainfall data.
        mask = xr.where(~np.isnan(rainfall.isel(T=0, M=0)), 1, np.nan)\
                 .drop_vars(['T']).squeeze().to_numpy()

        if self.equal_weighted:
            hindcast_det = hdcst.mean(dim='M')
            forecast_det = fcst.mean(dim='M')
        else:
            model_names = list(hdcst.coords["M"].values)
            selected_models = model_names
            
            hindcast_det = None
            forecast_det = None
            score_sum = None
            hindcast_det_unweighted = None
            forecast_det_unweighted = None

            for model_name in selected_models:
                # Interpolate and mask the score array for the current model.
                score_array = scores[model_name].interp(
                    Y=rainfall.Y,
                    X=rainfall.X,
                    method="nearest",
                    kwargs={"fill_value": "extrapolate"}
                )
                weight_array = self.transform_score(score_array)
    
                # Interpolate hindcast and forecast data to the rainfall grid.
                hindcast_data = hdcst.sel(M=model_name).interp(
                    Y=rainfall.Y,
                    X=rainfall.X,
                    method="nearest",
                    kwargs={"fill_value": "extrapolate"}
                )
    
                forecast_data = fcst.sel(M=model_name).interp(
                    Y=rainfall.Y,
                    X=rainfall.X,
                    method="nearest",
                    kwargs={"fill_value": "extrapolate"}
                )
    
                # Multiply by the weight.
                hindcast_weighted = hindcast_data * weight_array
                forecast_weighted = forecast_data * weight_array
    
                # Also keep an unweighted version for optional complete blending.
                if hindcast_det is None:
                    hindcast_det = hindcast_weighted
                    forecast_det = forecast_weighted
                    score_sum = weight_array
                    hindcast_det_unweighted = hindcast_data
                    forecast_det_unweighted = forecast_data
                else:
                    hindcast_det += hindcast_weighted
                    forecast_det += forecast_weighted
                    score_sum += weight_array
                    hindcast_det_unweighted += hindcast_data
                    forecast_det_unweighted += forecast_data
                    
            # Compute the weighted averages.
            hindcast_det = hindcast_det / score_sum
            forecast_det = forecast_det / score_sum

            # If complete==True, use unweighted averages to fill in missing grid cells.
            if complete:
                num_models = len(selected_models)
                hindcast_det_unweighted = hindcast_det_unweighted / num_models
                forecast_det_unweighted = forecast_det_unweighted / num_models
                mask_hd = xr.where(np.isnan(hindcast_det), 1, 0)
                mask_fc = xr.where(np.isnan(forecast_det), 1, 0)
                hindcast_det = hindcast_det.fillna(0) + hindcast_det_unweighted * mask_hd
                forecast_det = forecast_det.fillna(0) + forecast_det_unweighted * mask_fc
                
        if "M" in hindcast_det.coords:
            hindcast_det = hindcast_det.drop_vars('M')
        if "M" in forecast_det.coords:
            forecast_det = forecast_det.drop_vars('M')
                         
        return hindcast_det * mask, forecast_det * mask


    # ------------------ Probability Calculation Methods ------------------

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
            # Transform thresholds
            first_t = (first_tercile - best_guess) / error_std
            second_t = (second_tercile - best_guess) / error_std

            pred_prob[0, :] = stats.t.cdf(first_t, df=dof)
            pred_prob[1, :] = stats.t.cdf(second_t, df=dof) - stats.t.cdf(first_t, df=dof)
            pred_prob[2, :] = 1 - stats.t.cdf(second_t, df=dof)

        return pred_prob

    @staticmethod
    def calculate_tercile_probabilities_weibull_min(best_guess, error_variance, first_tercile, second_tercile, dof):
        """
        Weibull minimum-based method.
        
        Here, we assume:
          - best_guess is used as the location,
          - error_std (sqrt(error_variance)) as the scale, and 
          - dof (degrees of freedom) as the shape parameter.
        
        Parameters
        ----------
        best_guess : array-like
            Forecast or best guess values.
        error_variance : array-like
            Variance associated with forecast errors.
        first_tercile : array-like
            First tercile threshold values.
        second_tercile : array-like
            Second tercile threshold values.
        dof : float or array-like
            Shape parameter for the Weibull minimum distribution.
            
        Returns
        -------
        pred_prob : np.ndarray
            A 3 x n_time array with probabilities for being below the first tercile,
            between the first and second tercile, and above the second tercile.
        """
        n_time = len(best_guess)
        pred_prob = np.empty((3, n_time))
    
        if np.all(np.isnan(best_guess)):
            pred_prob[:] = np.nan
        else:
            error_std = np.sqrt(error_variance)
    
            # Using the weibull_min CDF with best_guess as loc and error_std as scale.
            pred_prob[0, :] = stats.weibull_min.cdf(first_tercile, c=dof, loc=best_guess, scale=error_std)
            pred_prob[1, :] = stats.weibull_min.cdf(second_tercile, c=dof, loc=best_guess, scale=error_std) - \
                               stats.weibull_min.cdf(first_tercile, c=dof, loc=best_guess, scale=error_std)
            pred_prob[2, :] = 1 - stats.weibull_min.cdf(second_tercile, c=dof, loc=best_guess, scale=error_std)
    
        return pred_prob

    @staticmethod
    def calculate_tercile_probabilities_gamma(best_guess, error_variance, T1, T2, dof=None):
        """Gamma-distribution based method."""
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

    @staticmethod
    def calculate_tercile_probabilities_normal(best_guess, error_variance, first_tercile, second_tercile):
        """Normal-distribution based method."""
        n_time = len(best_guess)
        pred_prob = np.empty((3, n_time))
        if np.all(np.isnan(best_guess)):
            pred_prob[:] = np.nan
        else:
            error_std = np.sqrt(error_variance)
            pred_prob[0, :] = norm.cdf(first_tercile, loc=best_guess, scale=error_std)
            pred_prob[1, :] = norm.cdf(second_tercile, loc=best_guess, scale=error_std) - norm.cdf(first_tercile, loc=best_guess, scale=error_std)
            pred_prob[2, :] = 1 - norm.cdf(second_tercile, loc=best_guess, scale=error_std)
        return pred_prob

    @staticmethod
    def calculate_tercile_probabilities_lognormal(best_guess, error_variance, first_tercile, second_tercile):
        """Lognormal-distribution based method."""
        n_time = len(best_guess)
        pred_prob = np.empty((3, n_time))
        if np.any(np.isnan(best_guess)) or np.any(np.isnan(error_variance)):
            pred_prob[:] = np.nan
            return pred_prob
        sigma = np.sqrt(np.log(1 + error_variance / (best_guess**2)))
        mu = np.log(best_guess) - sigma**2 / 2
        pred_prob[0, :] = lognorm.cdf(first_tercile, s=sigma, scale=np.exp(mu))
        pred_prob[1, :] = lognorm.cdf(second_tercile, s=sigma, scale=np.exp(mu)) - lognorm.cdf(first_tercile, s=sigma, scale=np.exp(mu))
        pred_prob[2, :] = 1 - lognorm.cdf(second_tercile, s=sigma, scale=np.exp(mu))
        return pred_prob

    def compute_prob(self, Predictant, clim_year_start, clim_year_end, hindcast_det):
        """
        Compute tercile probabilities for hindcast data.

        Calculates tercile probabilities based on the specified distribution method, using 
        climatological terciles derived from the predictand data.

        Parameters
        ----------
        Predictant : xarray.DataArray
            Observed predictand data with dimensions (T, Y, X).
        clim_year_start : int or str
            Start year of the climatology period.
        clim_year_end : int or str
            End year of the climatology period.
        hindcast_det : xarray.DataArray
            Deterministic hindcast data with dimensions (T, Y, X).

        Returns
        -------
        hindcast_prob : xarray.DataArray
            Tercile probabilities with dimensions (probability, T, Y, X).
        """

        if "M" in Predictant.coords:
            Predictant = Predictant.isel(M=0).drop_vars('M').squeeze()
        mask = xr.where(~np.isnan(Predictant.isel(T=0)), 1, np.nan).drop_vars(['T']).squeeze().to_numpy()
      
        # Ensure Predictant is (T, Y, X)
        Predictant = Predictant.transpose('T', 'Y', 'X')
        index_start = Predictant.get_index("T").get_loc(str(clim_year_start)).start
        index_end = Predictant.get_index("T").get_loc(str(clim_year_end)).stop
        rainfall_for_tercile = Predictant.isel(T=slice(index_start, index_end))
        terciles = rainfall_for_tercile.quantile([0.33, 0.67], dim='T')
        error_variance = (Predictant - hindcast_det).var(dim='T')
        dof = len(Predictant.get_index("T")) - 2
        
        if self.dist_method == "t":
            calc_func = self.calculate_tercile_probabilities
            hindcast_prob = xr.apply_ufunc(
                calc_func,
                hindcast_det,
                error_variance,
                terciles.isel(quantile=0).drop_vars('quantile'),
                terciles.isel(quantile=1).drop_vars('quantile'),
                input_core_dims=[('T',), (), (), ()],
                vectorize=True,
                kwargs={'dof': dof},
                dask='parallelized',
                output_core_dims=[('probability', 'T')],
                output_dtypes=['float'],
                dask_gufunc_kwargs={'output_sizes': {'probability': 3}, "allow_rechunk": True}
            )

        elif self.dist_method == "weibull_min":
            calc_func = self.calculate_tercile_probabilities_weibull_min
            hindcast_prob = xr.apply_ufunc(
                calc_func,
                hindcast_det,
                error_variance,
                terciles.isel(quantile=0).drop_vars('quantile'),
                terciles.isel(quantile=1).drop_vars('quantile'),
                input_core_dims=[('T',), (), (), ()],
                vectorize=True,
                kwargs={'dof': dof},
                dask='parallelized',
                output_core_dims=[('probability', 'T')],
                output_dtypes=['float'],
                dask_gufunc_kwargs={'output_sizes': {'probability': 3}, "allow_rechunk": True}
            )
        
        elif self.dist_method == "gamma":
            calc_func = self.calculate_tercile_probabilities_gamma
            hindcast_prob = xr.apply_ufunc(
                calc_func,
                hindcast_det,
                error_variance,
                terciles.isel(quantile=0).drop_vars('quantile'),
                terciles.isel(quantile=1).drop_vars('quantile'),
                input_core_dims=[('T',), (), (), ()],
                vectorize=True,
                dask='parallelized',
                output_core_dims=[('probability', 'T')],
                output_dtypes=['float'],
                dask_gufunc_kwargs={'output_sizes': {'probability': 3}, "allow_rechunk": True}
            )
        elif self.dist_method == "nonparam":
            calc_func = self.calculate_tercile_probabilities_nonparametric
            error_samples = (Predictant - hindcast_det)
            hindcast_prob = xr.apply_ufunc(
                calc_func,
                hindcast_det,
                error_samples,
                terciles.isel(quantile=0).drop_vars('quantile'),
                terciles.isel(quantile=1).drop_vars('quantile'),
                input_core_dims=[('T',), ('T',), (), ()],
                output_core_dims=[('probability', 'T')],
                vectorize=True,
                dask='parallelized',
                output_dtypes=[float],
                dask_gufunc_kwargs={'output_sizes': {'probability': 3}}
            )
        elif self.dist_method == "normal":
            calc_func = self.calculate_tercile_probabilities_normal
            hindcast_prob = xr.apply_ufunc(
                calc_func,
                hindcast_det,
                error_variance,
                terciles.isel(quantile=0).drop_vars('quantile'),
                terciles.isel(quantile=1).drop_vars('quantile'),
                input_core_dims=[('T',), (), (), ()],
                vectorize=True,
                dask='parallelized',
                output_core_dims=[('probability', 'T')],
                output_dtypes=['float'],
                dask_gufunc_kwargs={'output_sizes': {'probability': 3}, "allow_rechunk": True}
            )
        elif self.dist_method == "lognormal":
            calc_func = self.calculate_tercile_probabilities_lognormal
            hindcast_prob = xr.apply_ufunc(
                calc_func,
                hindcast_det,
                error_variance,
                terciles.isel(quantile=0).drop_vars('quantile'),
                terciles.isel(quantile=1).drop_vars('quantile'),
                input_core_dims=[('T',), (), (), ()],
                vectorize=True,
                dask='parallelized',
                output_core_dims=[('probability', 'T')],
                output_dtypes=['float'],
                dask_gufunc_kwargs={'output_sizes': {'probability': 3}, "allow_rechunk": True}
            )
        else:
            raise ValueError(f"Invalid dist_method: {self.dist_method}.")
        
        hindcast_prob = hindcast_prob.assign_coords(probability=('probability', ['PB', 'PN', 'PA']))
        return mask*hindcast_prob.transpose('probability', 'T', 'Y', 'X')

    def forecast(self, Predictant, clim_year_start, clim_year_end, hindcast_det, forecast_det):
        if "M" in Predictant.coords:
            Predictant = Predictant.isel(M=0).drop_vars('M').squeeze()
        mask = xr.where(~np.isnan(Predictant.isel(T=0)), 1, np.nan).drop_vars(['T']).squeeze().to_numpy()

        year = forecast_det.coords['T'].values.astype('datetime64[Y]').astype(int)[0] + 1970  # Convert from epoch
        T_value_1 = Predictant.isel(T=0).coords['T'].values  # Get the datetime64 value from da1
        month_1 = T_value_1.astype('datetime64[M]').astype(int) % 12 + 1  # Extract month
        new_T_value = np.datetime64(f"{year}-{month_1:02d}-{1:02d}")
        forecast_det = forecast_det.assign_coords(T=xr.DataArray([new_T_value], dims=["T"]))
        forecast_det['T'] = forecast_det['T'].astype('datetime64[ns]')

        
        # Compute tercile probabilities on the predictions
        index_start = Predictant.get_index("T").get_loc(str(clim_year_start)).start
        index_end = Predictant.get_index("T").get_loc(str(clim_year_end)).stop
        rainfall_for_tercile = Predictant.isel(T=slice(index_start, index_end))
        terciles = rainfall_for_tercile.quantile([0.33, 0.67], dim='T')
        error_variance = (Predictant - hindcast_det).var(dim='T')
        dof = len(Predictant.get_index("T")) - 2
        
        if self.dist_method == "t":
            calc_func = self.calculate_tercile_probabilities
            hindcast_prob = xr.apply_ufunc(
                calc_func,
                forecast_det,
                error_variance,
                terciles.isel(quantile=0).drop_vars('quantile'),
                terciles.isel(quantile=1).drop_vars('quantile'),
                input_core_dims=[('T',), (), (), ()],
                vectorize=True,
                kwargs={'dof': dof},
                dask='parallelized',
                output_core_dims=[('probability','T')],
                output_dtypes=['float'],
                dask_gufunc_kwargs={'output_sizes': {'probability': 3}, "allow_rechunk": True}
            )
        elif self.dist_method == "gamma":
            calc_func = self.calculate_tercile_probabilities_gamma
            hindcast_prob = xr.apply_ufunc(
                calc_func,
                forecast_det,
                error_variance,
                terciles.isel(quantile=0).drop_vars('quantile'),
                terciles.isel(quantile=1).drop_vars('quantile'),
                input_core_dims=[('T',), (), (), ()],
                vectorize=True,
                dask='parallelized',
                output_core_dims=[('probability','T')],
                output_dtypes=['float'],
                dask_gufunc_kwargs={'output_sizes': {'probability': 3}, "allow_rechunk": True}
            )
            
        elif self.dist_method == "weibull_min":
            calc_func = self.calculate_tercile_probabilities_weibull_min
            hindcast_prob = xr.apply_ufunc(
                calc_func,
                hindcast_det,
                error_variance,
                terciles.isel(quantile=0).drop_vars('quantile'),
                terciles.isel(quantile=1).drop_vars('quantile'),
                input_core_dims=[('T',), (), (), ()],
                vectorize=True,
                kwargs={'dof': dof},
                dask='parallelized',
                output_core_dims=[('probability', 'T')],
                output_dtypes=['float'],
                dask_gufunc_kwargs={'output_sizes': {'probability': 3}, "allow_rechunk": True}
            )
            
        elif self.dist_method == "normal":
            calc_func = self.calculate_tercile_probabilities_normal
            hindcast_prob = xr.apply_ufunc(
                calc_func,
                forecast_det,
                error_variance,
                terciles.isel(quantile=0).drop_vars('quantile'),
                terciles.isel(quantile=1).drop_vars('quantile'),
                input_core_dims=[('T',), (), (), ()],
                vectorize=True,
                dask='parallelized',
                output_core_dims=[('probability','T')],
                output_dtypes=['float'],
                dask_gufunc_kwargs={'output_sizes': {'probability': 3}, "allow_rechunk": True}
            )
        elif self.dist_method == "lognormal":
            calc_func = self.calculate_tercile_probabilities_lognormal
            hindcast_prob = xr.apply_ufunc(
                calc_func,
                forecast_det,
                error_variance,
                terciles.isel(quantile=0).drop_vars('quantile'),
                terciles.isel(quantile=1).drop_vars('quantile'),
                input_core_dims=[('T',), (), (), ()],
                vectorize=True,
                dask='parallelized',
                output_core_dims=[('probability','T')],
                output_dtypes=['float'],
                dask_gufunc_kwargs={'output_sizes': {'probability': 3}, "allow_rechunk": True}
            )
        elif self.dist_method == "nonparam":
            calc_func = self.calculate_tercile_probabilities_nonparametric
            error_samples = Predictant - hindcast_det
            error_samples = error_samples.rename({'T':'S'})
            hindcast_prob = xr.apply_ufunc(
                calc_func,
                forecast_det,
                error_samples,
                terciles.isel(quantile=0).drop_vars('quantile'),
                terciles.isel(quantile=1).drop_vars('quantile'),
                input_core_dims=[('T',), ('S',), (), ()],
                output_core_dims=[('probability','T')],
                vectorize=True,
                dask='parallelized',
                output_dtypes=[float],
                dask_gufunc_kwargs={'output_sizes': {'probability': 3}, "allow_rechunk": True}
            )
        else:
            raise ValueError(f"Invalid dist_method: {self.dist_method}.")
        
        hindcast_prob = hindcast_prob.assign_coords(probability=('probability', ['PB', 'PN', 'PA']))
        return mask * hindcast_prob.transpose('probability', 'T', 'Y', 'X')


class WAS_Min2009_ProbWeighted:
    """
    Probability-Weighted Multi-Model Ensemble based on Min et al. (2009).

    Implements a specific weighting scheme for combining multiple climate models, 
    where weights are derived from model scores with a threshold-based transformation.

    Parameters
    ----------
    None
    """
    def __init__(self):
        # Initialize any required attributes here
        pass

    def compute(self, rainfall, hdcst, fcst, scores, threshold=0.5, complete=False):
        """
        Compute probability-weighted ensemble estimates for hindcast and forecast datasets.

        Applies a weighting scheme where scores below the threshold are set to 0, and others to 1. 
        Optionally fills missing values with unweighted averages.

        Parameters
        ----------
        rainfall : xarray.DataArray
            Observed rainfall data with dimensions (T, Y, X, M).
        hdcst : xarray.DataArray
            Hindcast data with dimensions (T, M, Y, X).
        fcst : xarray.DataArray
            Forecast data with dimensions (T, M, Y, X).
        scores : dict
            Dictionary mapping model names to score arrays.
        threshold : float, optional
            Threshold below which scores are set to 0. Default is 0.5.
        complete : bool, optional
            If True, fill missing values with unweighted averages. Default is False.

        Returns
        -------
        hindcast_weighted : xarray.DataArray
            Weighted hindcast ensemble with dimensions (T, Y, X).
        forecast_weighted : xarray.DataArray
            Weighted forecast ensemble with dimensions (T, Y, X).
        """
        
        
        # --- Adjust time coordinates ---
        # Extract the year from the forecast's T coordinate (assuming epoch conversion)
        year = fcst.coords['T'].values.astype('datetime64[Y]').astype(int)[0] + 1970
        T_value_1 = rainfall.isel(T=0).coords['T'].values  # Get the initial time value from rainfall
        month_1 = T_value_1.astype('datetime64[M]').astype(int) % 12 + 1  # Extract month (1-12)
        new_T_value = np.datetime64(f"{year}-{month_1:02d}-01")
        
        # Update forecast and hindcast time coordinates
        fcst = fcst.assign_coords(T=xr.DataArray([new_T_value], dims=["T"]))
        fcst['T'] = fcst['T'].astype('datetime64[ns]')
        hdcst['T'] = rainfall['T'].astype('datetime64[ns]')
        
        # Create a spatial mask from rainfall (using first time and model)
        mask = xr.where(~np.isnan(rainfall.isel(T=0, M=0)), 1, np.nan).drop_vars('T').squeeze().to_numpy()

        # --- Initialize accumulators for weighted and unweighted sums ---
        weighted_hindcast_sum = None
        weighted_forecast_sum = None
        score_sum = None

        hindcast_sum = None
        forecast_sum = None

        model_names = list(hdcst.coords["M"].values)
        
        # --- Loop over each model ---
        for model_name in model_names:
            # Interpolate the score array to the rainfall grid
            score_array = scores[model_name].interp(
                Y=rainfall.Y,
                X=rainfall.X,
                method="nearest",
                kwargs={"fill_value": "extrapolate"}
            )
            # Apply weighting rules: below threshold set to 0; between threshold and 0.6 -> 0.6; 
            # between 0.6 and 0.8 -> 0.8; above 0.8 -> 1.

            # score_array = xr.where(
            #    score_array <= threshold,
            #     0,
            #     xr.where(
            #         score_array <= 0.6,
            #         0.6,
            #        xr.where(score_array <= 0.8, 0.8, 1)
            #     )
            # )

            score_array = xr.where(
                score_array <= threshold,
                0,1
            )
            # Interpolate hindcast and forecast data for the model to the rainfall grid
            hindcast_data = hdcst.sel(M=model_name).interp(
                Y=rainfall.Y,
                X=rainfall.X,
                method="nearest",
                kwargs={"fill_value": "extrapolate"}
            )
            forecast_data = fcst.sel(M=model_name).interp(
                Y=rainfall.Y,
                X=rainfall.X,
                method="nearest",
                kwargs={"fill_value": "extrapolate"}
            )

            # Weight the datasets by the score_array
            weighted_hindcast = hindcast_data * score_array
            weighted_forecast = forecast_data * score_array

            # Accumulate weighted and unweighted sums
            if weighted_hindcast_sum is None:
                weighted_hindcast_sum = weighted_hindcast
                weighted_forecast_sum = weighted_forecast
                score_sum = score_array
                hindcast_sum = hindcast_data
                forecast_sum = forecast_data
            else:
                weighted_hindcast_sum += weighted_hindcast
                weighted_forecast_sum += weighted_forecast
                score_sum += score_array
                hindcast_sum += hindcast_data
                forecast_sum += forecast_data

        # --- Compute weighted ensemble (weighted average) ---
        hindcast_weighted = weighted_hindcast_sum / score_sum
        forecast_weighted = weighted_forecast_sum / score_sum
        
        # --- Optionally complete missing values with unweighted average ---
        if complete:
            n_models = len(model_names)
            hindcast_unweighted = hindcast_sum / n_models
            forecast_unweighted = forecast_sum / n_models
            
            # Identify missing areas in the weighted estimates
            mask_hd = xr.where(np.isnan(hindcast_weighted), 1, 0)
            mask_fc = xr.where(np.isnan(forecast_weighted), 1, 0)
            
            hindcast_weighted = hindcast_weighted.fillna(0) + hindcast_unweighted * mask_hd
            forecast_weighted = forecast_weighted.fillna(0) + forecast_unweighted * mask_fc

        # --- Drop the 'M' dimension if present ---
        if "M" in hindcast_weighted.coords:
            hindcast_weighted = hindcast_weighted.drop_vars('M')
        if "M" in forecast_weighted.coords:
            forecast_weighted = forecast_weighted.drop_vars('M')
        
        return hindcast_weighted * mask, forecast_weighted * mask

# ---------------------------------------------------
# WAS_mme_GA Class
#   - Genetic Algorithm for multi-model ensemble weighting
# ---------------------------------------------------
class WAS_mme_GA:
    """
    Genetic Algorithm-based Multi-Model Ensemble Weighting.

    Uses a Genetic Algorithm to optimize weights for combining multiple climate models, 
    minimizing the mean squared error (MSE) against observations. Supports tercile probability 
    calculations.

    Parameters
    ----------
    population_size : int, optional
        Number of individuals in the GA population. Default is 20.
    max_iter : int, optional
        Maximum number of generations for the GA. Default is 50.
    crossover_rate : float, optional
        Probability of performing crossover. Default is 0.7.
    mutation_rate : float, optional
        Probability of mutating a gene. Default is 0.01.
    random_state : int, optional
        Seed for random number generation. Default is 42.
    dist_method : str, optional
        Distribution method for probability calculations ('t', 'gamma', 'nonparam', 
        'normal', 'lognormal', 'weibull_min'). Default is 'gamma'.
    """

    def __init__(self,
                 population_size=20,
                 max_iter=50,
                 crossover_rate=0.7,
                 mutation_rate=0.01,
                 random_state=42,
                 dist_method="gamma"):
        """
        Initialize the GA population with random weight vectors.

        Each weight vector is normalized to sum to 1.

        Parameters
        ----------
        n_models : int
            Number of models in the ensemble.

        Returns
        -------
        population : list of np.ndarray
            List of normalized weight vectors.
        """
        
        
        self.population_size = population_size
        self.max_iter = max_iter
        self.crossover_rate = crossover_rate
        self.mutation_rate = mutation_rate
        self.random_state = random_state
        self.dist_method = dist_method

        # Set seeds
        random.seed(self.random_state)
        np.random.seed(self.random_state)

        # Best solution found by GA
        self.best_chromosome = None
        self.best_fitness = None

    # ---------------------------------------------------
    # GA Routines for Ensemble Weights
    # ---------------------------------------------------
    def _initialize_population(self, n_models):
        """
        Initialize the GA population with random weight vectors.

        Each weight vector is normalized to sum to 1.

        Parameters
        ----------
        n_models : int
            Number of models in the ensemble.

        Returns
        -------
        population : list of np.ndarray
            List of normalized weight vectors.
        """
        population = []
        for _ in range(self.population_size):
            w = np.random.rand(n_models)
            w /= w.sum()  # normalize so sum=1
            population.append(w)
        return population

    def _fitness_function(self, weights, X, y):
        """
        Compute the negative MSE of the ensemble.

        Parameters
        ----------
        weights : np.ndarray
            Weight vector for the models.
        X : np.ndarray
            Predictor data with shape (n_samples, n_models).
        y : np.ndarray
            Observed data with shape (n_samples,) or (n_samples, 1).

        Returns
        -------
        fitness : float
            Negative mean squared error (GA maximizes fitness).
        """

        if y.ndim == 2 and y.shape[1] == 1:
            y = y.ravel()

        y_pred = np.sum(weights * X, axis=1)  # Weighted sum across models
        mse = np.mean((y - y_pred)**2)
        return -mse  # negative MSE (GA maximizes fitness)

    def _selection(self, population, fitnesses):
        """
        Perform roulette wheel selection based on fitness.

        Parameters
        ----------
        population : list of np.ndarray
            List of weight vectors.
        fitnesses : list of float
            Fitness values for each individual.

        Returns
        -------
        selected : np.ndarray
            Selected weight vector.
        """
        total_fit = sum(fitnesses)
        pick = random.uniform(0, total_fit)
        current = 0
        for chrom, fit in zip(population, fitnesses):
            current += fit
            if current >= pick:
                return chrom
        return population[-1]

    def _crossover(self, parent1, parent2):
        """
        Perform single-point crossover on two parent weight vectors.

        Parameters
        ----------
        parent1 : np.ndarray
            First parent weight vector.
        parent2 : np.ndarray
            Second parent weight vector.

        Returns
        -------
        child1, child2 : tuple of np.ndarray
            Two child weight vectors.
        """
        if random.random() < self.crossover_rate:
            point = random.randint(1, len(parent1) - 1)
            child1 = np.concatenate((parent1[:point], parent2[point:]))
            child2 = np.concatenate((parent2[:point], parent1[point:]))
            return child1, child2
        else:
            return parent1.copy(), parent2.copy()

    def _mutate(self, chromosome):
        """
        Mutation: each weight can be perturbed slightly.
        Then clip negatives to 0 and renormalize to sum=1.
        Mutate a weight vector with Gaussian noise and normalize.

        Parameters
        ----------
        chromosome : np.ndarray
            Weight vector to mutate.

        Returns
        -------
        mutated : np.ndarray
            Mutated and normalized weight vector.
        """
        for i in range(len(chromosome)):
            if random.random() < self.mutation_rate:
                chromosome[i] += np.random.normal(0, 0.1)

        # Clip any negatives
        chromosome = np.clip(chromosome, 0, None)
        # Renormalize
        s = np.sum(chromosome)
        if s == 0:
            # re-init if it all collapsed to zero
            chromosome = np.random.rand(len(chromosome))
        else:
            chromosome /= s
        return chromosome

    def _run_ga(self, X, y):
        """
        Run the GA to find best ensemble weights for M models.
        X shape: (n_samples, n_models). y shape: (n_samples,).
        Run the Genetic Algorithm to find optimal ensemble weights.

        Parameters
        ----------
        X : np.ndarray
            Predictor data with shape (n_samples, n_models).
        y : np.ndarray
            Observed data with shape (n_samples,).

        Returns
        -------
        best_chrom : np.ndarray
            Best weight vector found.
        best_fit : float
            Best fitness value (negative MSE).
        """
        n_models = X.shape[1]
        population = self._initialize_population(n_models)

        best_chrom = None
        best_fit = float('-inf')

        for _ in range(self.max_iter):
            # Evaluate fitness
            fitnesses = [self._fitness_function(ch, X, y) for ch in population]

            # Track best
            gen_best_fit = max(fitnesses)
            gen_best_idx = np.argmax(fitnesses)
            if gen_best_fit > best_fit:
                best_fit = gen_best_fit
                best_chrom = population[gen_best_idx].copy()

            # Create new population
            new_population = []
            while len(new_population) < self.population_size:
                p1 = self._selection(population, fitnesses)
                p2 = self._selection(population, fitnesses)
                c1, c2 = self._crossover(p1, p2)
                c1 = self._mutate(c1)
                c2 = self._mutate(c2)
                new_population.extend([c1, c2])

            population = new_population[:self.population_size]

        return best_chrom, best_fit

    def _predict_ensemble(self, weights, X):
        """
        Weighted sum across models:
           y_pred[i] = sum_j( weights[j] * X[i,j] )
        Compute weighted sum of model predictions.

        Parameters
        ----------
        weights : np.ndarray
            Weight vector for the models.
        X : np.ndarray
            Predictor data with shape (n_samples, n_models) or (n_models,).

        Returns
        -------
        y_pred : np.ndarray
            Weighted predictions.
        """
        if X.ndim == 1:
            # Single sample => dot product
            return np.sum(weights * X)
        else:
            return np.sum(weights * X, axis=1)

    # ---------------------------------------------------
    # compute_model
    # ---------------------------------------------------
    def compute_model(self, X_train, y_train, X_test, y_test):
        """
        Train the GA and predict on test data.

        Parameters
        ----------
        X_train : xarray.DataArray
            Training predictor data with dimensions (T, Y, X, M).
        y_train : xarray.DataArray
            Training predictand data with dimensions (T, Y, X).
        X_test : xarray.DataArray
            Testing predictor data with dimensions (T, Y, X, M).
        y_test : xarray.DataArray
            Testing predictand data with dimensions (T, Y, X).

        Returns
        -------
        predicted_da : xarray.DataArray
            Predictions with dimensions (T, Y, X).
        """
        # 1) Extract coordinates from X_test
        time = X_test['T']
        lat  = X_test['Y']
        lon  = X_test['X']
        n_time = len(time)
        n_lat  = len(lat)
        n_lon  = len(lon)

        # 2) Stack/reshape training data & remove NaNs
        #    X_train => (samples, M), y_train => (samples,)
        X_train_stacked = X_train.stack(sample=('T','Y','X')).transpose('sample','M').values
        y_train_stacked = y_train.stack(sample=('T','Y','X')).transpose('sample', 'M').values
        
        nan_mask_train = (np.any(~np.isfinite(X_train_stacked), axis=1) |
                          np.any(~np.isfinite(y_train_stacked), axis=1))
        
        X_train_clean = X_train_stacked[~nan_mask_train]
        y_train_clean = y_train_stacked[~nan_mask_train]

        # 3) Stack/reshape test data & remove NaNs similarly
        X_test_stacked = X_test.stack(sample=('T','Y','X')).transpose('sample','M').values
        y_test_stacked = y_test.stack(sample=('T','Y','X')).transpose('sample', 'M').values
        nan_mask_test = (np.any(~np.isfinite(X_test_stacked), axis=1) |
                         np.any(~np.isfinite(y_test_stacked), axis=1))

        # 4) Run GA on training data
        if len(X_train_clean) == 0:
            # If no valid training, fill with NaNs
            self.best_chromosome = None
            self.best_fitness = None
            result = np.full_like(y_test_stacked, np.nan)
        else:
            self.best_chromosome, self.best_fitness = self._run_ga(X_train_clean, y_train_clean)

            # 5) Predict on X_test
            X_test_clean = X_test_stacked[~nan_mask_test]
            y_pred_clean = self._predict_ensemble(self.best_chromosome, X_test_clean)

            result = np.empty_like(np.squeeze(y_test_stacked))
            result[np.squeeze(nan_mask_test)] = np.squeeze(y_test_stacked[nan_mask_test])
            result[~np.squeeze(nan_mask_test)] = y_pred_clean
        
        # 6) Reshape predictions back to (T, Y, X)
        predictions_reshaped = result.reshape(n_time, n_lat, n_lon)
        predicted_da = xr.DataArray(
            data=predictions_reshaped,
            coords={'T': time, 'Y': lat, 'X': lon},
            dims=['T','Y','X']
        )
        return predicted_da

    # ---------------------------------------------------
    # Probability Calculation Methods
    # ---------------------------------------------------
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
    def calculate_tercile_probabilities_weibull_min(best_guess, error_variance, first_tercile, second_tercile, dof):
        """
        Weibull minimum-based method.
        """
        n_time = len(best_guess)
        pred_prob = np.empty((3, n_time))
        if np.all(np.isnan(best_guess)):
            pred_prob[:] = np.nan
        else:
            error_std = np.sqrt(error_variance)
            pred_prob[0, :] = stats.weibull_min.cdf(first_tercile, c=dof, loc=best_guess, scale=error_std)
            pred_prob[1, :] = stats.weibull_min.cdf(second_tercile, c=dof, loc=best_guess, scale=error_std) \
                              - stats.weibull_min.cdf(first_tercile, c=dof, loc=best_guess, scale=error_std)
            pred_prob[2, :] = 1 - stats.weibull_min.cdf(second_tercile, c=dof, loc=best_guess, scale=error_std)
        return pred_prob

    @staticmethod
    def calculate_tercile_probabilities_gamma(best_guess, error_variance, T1, T2, dof=None):
        """
        Gamma-distribution based method.
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
        Non-parametric method using historical error samples.
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
        Normal-distribution based method.
        """
        n_time = len(best_guess)
        pred_prob = np.empty((3, n_time))
        if np.all(np.isnan(best_guess)):
            pred_prob[:] = np.nan
        else:
            error_std = np.sqrt(error_variance)
            pred_prob[0, :] = norm.cdf(first_tercile, loc=best_guess, scale=error_std)
            pred_prob[1, :] = norm.cdf(second_tercile, loc=best_guess, scale=error_std) \
                              - norm.cdf(first_tercile, loc=best_guess, scale=error_std)
            pred_prob[2, :] = 1 - norm.cdf(second_tercile, loc=best_guess, scale=error_std)
        return pred_prob

    @staticmethod
    def calculate_tercile_probabilities_lognormal(best_guess, error_variance, first_tercile, second_tercile):
        """
        Lognormal-distribution based method.
        """
        n_time = len(best_guess)
        pred_prob = np.empty((3, n_time))
        if np.any(np.isnan(best_guess)) or np.any(np.isnan(error_variance)):
            pred_prob[:] = np.nan
            return pred_prob
        sigma = np.sqrt(np.log(1 + error_variance / (best_guess**2)))
        mu = np.log(best_guess) - sigma**2 / 2
        pred_prob[0, :] = lognorm.cdf(first_tercile, s=sigma, scale=np.exp(mu))
        pred_prob[1, :] = lognorm.cdf(second_tercile, s=sigma, scale=np.exp(mu)) \
                          - lognorm.cdf(first_tercile, s=sigma, scale=np.exp(mu))
        pred_prob[2, :] = 1 - lognorm.cdf(second_tercile, s=sigma, scale=np.exp(mu))
        return pred_prob

    # ---------------------------------------------------
    # compute_prob
    # ---------------------------------------------------
    def compute_prob(self, Predictant, clim_year_start, clim_year_end, hindcast_det):
        """
        Compute tercile probabilities for hindcast data using the GA-optimized weights.

        Parameters
        ----------
        Predictant : xarray.DataArray
            Observed predictand data with dimensions (T, Y, X).
        clim_year_start : int or str
            Start year of the climatology period.
        clim_year_end : int or str
            End year of the climatology period.
        hindcast_det : xarray.DataArray
            Deterministic hindcast data with dimensions (T, Y, X).

        Returns
        -------
        hindcast_prob : xarray.DataArray
            Tercile probabilities with dimensions (probability, T, Y, X).
        """
        if "M" in Predictant.coords:
            Predictant = Predictant.isel(M=0).drop_vars('M').squeeze()

        mask = xr.where(~np.isnan(Predictant.isel(T=0)), 1, np.nan) \
                 .drop_vars(['T']).squeeze().to_numpy()

        # Ensure (T, Y, X)
        Predictant = Predictant.transpose('T', 'Y', 'X')
        index_start = Predictant.get_index("T").get_loc(str(clim_year_start)).start
        index_end   = Predictant.get_index("T").get_loc(str(clim_year_end)).stop
        rainfall_for_tercile = Predictant.isel(T=slice(index_start, index_end))
        terciles = rainfall_for_tercile.quantile([0.33, 0.67], dim='T')
        error_variance = (Predictant - hindcast_det).var(dim='T')
        dof = len(Predictant.get_index("T")) - 2

        # Choose distribution method
        if self.dist_method == "t":
            calc_func = self.calculate_tercile_probabilities
            hindcast_prob = xr.apply_ufunc(
                calc_func,
                hindcast_det,
                error_variance,
                terciles.isel(quantile=0).drop_vars('quantile'),
                terciles.isel(quantile=1).drop_vars('quantile'),
                input_core_dims=[('T',), (), (), ()],
                vectorize=True,
                kwargs={'dof': dof},
                dask='parallelized',
                output_core_dims=[('probability', 'T')],
                output_dtypes=['float'],
                dask_gufunc_kwargs={'output_sizes': {'probability': 3},
                                    "allow_rechunk": True}
            )
        elif self.dist_method == "weibull_min":
            calc_func = self.calculate_tercile_probabilities_weibull_min
            hindcast_prob = xr.apply_ufunc(
                calc_func,
                hindcast_det,
                error_variance,
                terciles.isel(quantile=0).drop_vars('quantile'),
                terciles.isel(quantile=1).drop_vars('quantile'),
                input_core_dims=[('T',), (), (), ()],
                vectorize=True,
                kwargs={'dof': dof},
                dask='parallelized',
                output_core_dims=[('probability', 'T')],
                output_dtypes=['float'],
                dask_gufunc_kwargs={'output_sizes': {'probability': 3},
                                    "allow_rechunk": True}
            )
        elif self.dist_method == "gamma":
            calc_func = self.calculate_tercile_probabilities_gamma
            hindcast_prob = xr.apply_ufunc(
                calc_func,
                hindcast_det,
                error_variance,
                terciles.isel(quantile=0).drop_vars('quantile'),
                terciles.isel(quantile=1).drop_vars('quantile'),
                input_core_dims=[('T',), (), (), ()],
                vectorize=True,
                dask='parallelized',
                output_core_dims=[('probability', 'T')],
                output_dtypes=['float'],
                dask_gufunc_kwargs={'output_sizes': {'probability': 3},
                                    "allow_rechunk": True}
            )
        elif self.dist_method == "nonparam":
            calc_func = self.calculate_tercile_probabilities_nonparametric
            error_samples = (Predictant - hindcast_det)
            hindcast_prob = xr.apply_ufunc(
                calc_func,
                hindcast_det,
                error_samples,
                terciles.isel(quantile=0).drop_vars('quantile'),
                terciles.isel(quantile=1).drop_vars('quantile'),
                input_core_dims=[('T',), ('T',), (), ()],
                output_core_dims=[('probability', 'T')],
                vectorize=True,
                dask='parallelized',
                output_dtypes=[float],
                dask_gufunc_kwargs={'output_sizes': {'probability': 3},
                                    "allow_rechunk": True}
            )
        elif self.dist_method == "normal":
            calc_func = self.calculate_tercile_probabilities_normal
            hindcast_prob = xr.apply_ufunc(
                calc_func,
                hindcast_det,
                error_variance,
                terciles.isel(quantile=0).drop_vars('quantile'),
                terciles.isel(quantile=1).drop_vars('quantile'),
                input_core_dims=[('T',), (), (), ()],
                vectorize=True,
                dask='parallelized',
                output_core_dims=[('probability', 'T')],
                output_dtypes=['float'],
                dask_gufunc_kwargs={'output_sizes': {'probability': 3},
                                    "allow_rechunk": True}
            )
        elif self.dist_method == "lognormal":
            calc_func = self.calculate_tercile_probabilities_lognormal
            hindcast_prob = xr.apply_ufunc(
                calc_func,
                hindcast_det,
                error_variance,
                terciles.isel(quantile=0).drop_vars('quantile'),
                terciles.isel(quantile=1).drop_vars('quantile'),
                input_core_dims=[('T',), (), (), ()],
                vectorize=True,
                dask='parallelized',
                output_core_dims=[('probability', 'T')],
                output_dtypes=['float'],
                dask_gufunc_kwargs={'output_sizes': {'probability': 3},
                                    "allow_rechunk": True}
            )
        else:
            raise ValueError(f"Invalid dist_method: {self.dist_method}.")

        hindcast_prob = hindcast_prob.assign_coords(probability=('probability', ['PB', 'PN', 'PA']))
        return hindcast_prob.transpose('probability', 'T', 'Y', 'X')

    # ---------------------------------------------------
    # forecast
    # ---------------------------------------------------
    def forecast(self, Predictant, clim_year_start, clim_year_end,
                 hindcast_det, hindcast_det_cross, Predictor_for_year):
        """
        Forecast for a target year and compute tercile probabilities.

        Standardizes data, fits the GA, predicts for the target year, and computes probabilities.

        Parameters
        ----------
        Predictant : xarray.DataArray
            Historical predictand data with dimensions (T, Y, X).
        clim_year_start : int or str
            Start year of the climatology period.
        clim_year_end : int or str
            End year of the climatology period.
        hindcast_det : xarray.DataArray
            Deterministic hindcast data for training with dimensions (T, Y, X, M).
        hindcast_det_cross : xarray.DataArray
            Deterministic hindcast data for error estimation with dimensions (T, Y, X).
        Predictor_for_year : xarray.DataArray
            Predictor data for the target year with dimensions (T, Y, X, M).

        Returns
        -------
        result_da : xarray.DataArray
            Deterministic forecast with dimensions (T, Y, X).
        hindcast_prob : xarray.DataArray
            Tercile probability forecast with dimensions (probability, T, Y, X).
        """
        mask = xr.where(~np.isnan(Predictant.isel(T=0, M=0)), 1, np.nan) \
                 .drop_vars(['T','M']).squeeze().to_numpy()

        # Standardize
        mean_val = hindcast_det.sel(T=slice(str(clim_year_start), str(clim_year_end))).mean(dim='T')
        std_val = hindcast_det.sel(T=slice(str(clim_year_start), str(clim_year_end))).std(dim='T')
        Predictor_for_year_st = (Predictor_for_year - mean_val) / std_val

        hindcast_det_st = standardize_timeseries(hindcast_det, clim_year_start, clim_year_end)
        Predictant_st    = standardize_timeseries(Predictant, clim_year_start, clim_year_end)
        y_test = Predictant_st.isel(T=[-1])

        # If GA not fitted yet, we can fit it on the entire hindcast
        if self.best_chromosome is None:
            # Stack
            X_train_stacked = hindcast_det_st.stack(sample=('T','Y','X')).transpose('sample','M').values
            y_train_stacked = Predictant_st.stack(sample=('T','Y','X')).transpose('sample','M').values

            # # Flatten y if needed
            # if y_train_stacked.shape[1] == 1:
            #     y_train_stacked = y_train_stacked.ravel()

            nan_mask_train = (np.any(~np.isfinite(X_train_stacked), axis=1) |
                              np.any(~np.isfinite(y_train_stacked), axis=1))

            X_train_clean = X_train_stacked[~nan_mask_train]
            y_train_clean = y_train_stacked[~nan_mask_train]

            if len(X_train_clean) > 0:
                self.best_chromosome, self.best_fitness = self._run_ga(X_train_clean, y_train_clean)

        # Now predict for the new year
        time = Predictor_for_year_st['T']
        lat  = Predictor_for_year_st['Y']
        lon  = Predictor_for_year_st['X']
        n_time = len(time)
        n_lat  = len(lat)
        n_lon  = len(lon)

        X_test_stacked = Predictor_for_year_st.stack(sample=('T','Y','X')).transpose('sample','M').values
        y_test_stacked = y_test.stack(sample=('T','Y','X')).transpose('sample','M').values
        # if y_test_stacked.shape[1] == 1:
        #     y_test_stacked = y_test_stacked.ravel()

        nan_mask_test = (np.any(~np.isfinite(X_test_stacked), axis=1) |
                         np.any(~np.isfinite(y_test_stacked), axis=1))

        if self.best_chromosome is not None:
            y_pred_clean = self._predict_ensemble(self.best_chromosome, X_test_stacked[~nan_mask_test])
            result = np.empty_like(np.squeeze(y_test_stacked))
            result[np.squeeze(nan_mask_test)] = np.squeeze(y_test_stacked[nan_mask_test])
            result[~np.squeeze(nan_mask_test)] = y_pred_clean
        else:
            result = np.full_like(np.squeeze(y_test_stacked), np.nan)

        predictions_reshaped = result.reshape(n_time, n_lat, n_lon)
        result_da = xr.DataArray(data=predictions_reshaped,
                                 coords={'T': time, 'Y': lat, 'X': lon},
                                 dims=['T','Y','X']) * mask

        # Reverse-standardize
        result_da = reverse_standardize(
            result_da,
            Predictant.isel(M=0).drop_vars("M").squeeze(),
            clim_year_start, clim_year_end
        )
        
        if "M" in Predictant.coords:
            Predictant = Predictant.isel(M=0).drop_vars('M').squeeze()
            
        # Fix T coordinate for the predicted year (simple approach)
        year = Predictor_for_year.coords['T'].values.astype('datetime64[Y]').astype(int)[0] + 1970
        T_value_1 = Predictant.isel(T=0).coords['T'].values
        month_1 = T_value_1.astype('datetime64[M]').astype(int) % 12 + 1
        new_T_value = np.datetime64(f"{year}-{month_1:02d}-01")
        result_da = result_da.assign_coords(T=xr.DataArray([new_T_value], dims=["T"]))
        result_da['T'] = result_da['T'].astype('datetime64[ns]')

        # Compute tercile probabilities using cross-validated hindcast_det_cross
        index_start = Predictant.get_index("T").get_loc(str(clim_year_start)).start
        index_end   = Predictant.get_index("T").get_loc(str(clim_year_end)).stop
        rainfall_for_tercile = Predictant.isel(T=slice(index_start, index_end))
        terciles = rainfall_for_tercile.quantile([0.33, 0.67], dim='T')
        error_variance = (Predictant - hindcast_det_cross).var(dim='T')
        dof = len(Predictant.get_index("T")) - 2

        # Distribution method
        if self.dist_method == "t":
            calc_func = self.calculate_tercile_probabilities
            hindcast_prob = xr.apply_ufunc(
                calc_func,
                result_da,
                error_variance,
                terciles.isel(quantile=0).drop_vars('quantile'),
                terciles.isel(quantile=1).drop_vars('quantile'),
                input_core_dims=[('T',), (), (), ()],
                vectorize=True,
                kwargs={'dof': dof},
                dask='parallelized',
                output_core_dims=[('probability','T')],
                output_dtypes=['float'],
                dask_gufunc_kwargs={'output_sizes': {'probability': 3},
                                    "allow_rechunk": True}
            )
        elif self.dist_method == "weibull_min":
            calc_func = self.calculate_tercile_probabilities_weibull_min
            hindcast_prob = xr.apply_ufunc(
                calc_func,
                result_da,
                error_variance,
                terciles.isel(quantile=0).drop_vars('quantile'),
                terciles.isel(quantile=1).drop_vars('quantile'),
                input_core_dims=[('T',), (), (), ()],
                vectorize=True,
                kwargs={'dof': dof},
                dask='parallelized',
                output_core_dims=[('probability','T')],
                output_dtypes=['float'],
                dask_gufunc_kwargs={'output_sizes': {'probability': 3},
                                    "allow_rechunk": True}
            )
        elif self.dist_method == "gamma":
            calc_func = self.calculate_tercile_probabilities_gamma
            hindcast_prob = xr.apply_ufunc(
                calc_func,
                result_da,
                error_variance,
                terciles.isel(quantile=0).drop_vars('quantile'),
                terciles.isel(quantile=1).drop_vars('quantile'),
                input_core_dims=[('T',), (), (), ()],
                vectorize=True,
                dask='parallelized',
                output_core_dims=[('probability','T')],
                output_dtypes=['float'],
                dask_gufunc_kwargs={'output_sizes': {'probability': 3},
                                    "allow_rechunk": True}
            )
        elif self.dist_method == "normal":
            calc_func = self.calculate_tercile_probabilities_normal
            hindcast_prob = xr.apply_ufunc(
                calc_func,
                result_da,
                error_variance,
                terciles.isel(quantile=0).drop_vars('quantile'),
                terciles.isel(quantile=1).drop_vars('quantile'),
                input_core_dims=[('T',), (), (), ()],
                vectorize=True,
                dask='parallelized',
                output_core_dims=[('probability','T')],
                output_dtypes=['float'],
                dask_gufunc_kwargs={'output_sizes': {'probability': 3},
                                    "allow_rechunk": True}
            )
        elif self.dist_method == "lognormal":
            calc_func = self.calculate_tercile_probabilities_lognormal
            hindcast_prob = xr.apply_ufunc(
                calc_func,
                result_da,
                error_variance,
                terciles.isel(quantile=0).drop_vars('quantile'),
                terciles.isel(quantile=1).drop_vars('quantile'),
                input_core_dims=[('T',), (), (), ()],
                vectorize=True,
                dask='parallelized',
                output_core_dims=[('probability','T')],
                output_dtypes=['float'],
                dask_gufunc_kwargs={'output_sizes': {'probability': 3},
                                    "allow_rechunk": True}
            )
        elif self.dist_method == "nonparam":
            calc_func = self.calculate_tercile_probabilities_nonparametric
            error_samples = (Predictant - hindcast_det_cross).rename({'T': 'S'})
            hindcast_prob = xr.apply_ufunc(
                calc_func,
                result_da,
                error_samples,
                terciles.isel(quantile=0).drop_vars('quantile'),
                terciles.isel(quantile=1).drop_vars('quantile'),
                input_core_dims=[('T',), ('S',), (), ()],
                output_core_dims=[('probability','T')],
                vectorize=True,
                dask='parallelized',
                output_dtypes=[float],
                dask_gufunc_kwargs={'output_sizes': {'probability': 3},
                                    "allow_rechunk": True}
            )
        else:
            raise ValueError(f"Invalid dist_method: {self.dist_method}.")

        hindcast_prob = hindcast_prob.assign_coords(probability=('probability', ['PB', 'PN', 'PA']))

        # Return the final forecast and its tercile probabilities
        return result_da * mask, mask * hindcast_prob.transpose('probability', 'T', 'Y', 'X')

class BMA:
    def __init__(
        self,
        observations,
        model_predictions,
        model_names=None,
        alpha=1.0,
        error_metric="rmse"
    ):
        """
        Bayesian Model Averaging (BMA) class, supporting either RMSE- or MAE-based priors.

        Parameters
        ----------
        observations : 1D array of length T
            Observed values over time.
        model_predictions : list of 1D arrays
            Each element is a model's predictions over the same time period (length T).
        model_names : list of str, optional
            If not provided or mismatched, names are generated.
        alpha : float
            Hyperparameter controlling how strongly the chosen error metric influences prior weights.
        error_metric : {"rmse", "mae"}, optional
            Which error metric to use for prior weighting. Default is "rmse".
        """
        self.observations = np.asarray(observations)
        self.model_predictions = [np.asarray(mp) for mp in model_predictions]
        self.M = len(self.model_predictions)
        self.T = len(self.observations)
        self.alpha = alpha
        self.error_metric = error_metric.lower()

        # Handle model names
        if model_names is not None and len(model_names) == self.M:
            self.model_names = model_names
        else:
            self.model_names = [f"Model{i+1}" for i in range(self.M)]

        # Attributes to be computed
        self.rmse_or_mae_vals = None   # Will store either RMSEs or MAEs
        self.model_priors = None
        self.waics = None
        self.traces = None
        self.posterior_probs = None

        # Store posterior means
        self.model_offsets = np.zeros(self.M)
        self.model_scales = np.ones(self.M)
        self.model_sigmas = np.zeros(self.M)

    def compute_error_based_prior(self):
        """
        Compute either RMSE or MAE for each model vs. observations, 
        then use exp(-alpha * error) for priors.
        """
        error_vals = []
        for preds in self.model_predictions:
            if self.error_metric == "rmse":
                val = np.sqrt(np.mean((preds - self.observations) ** 2))
            elif self.error_metric == "mae":
                val = np.mean(np.abs(preds - self.observations))
            else:
                raise ValueError(f"Invalid error_metric: {self.error_metric}")
            error_vals.append(val)

        self.rmse_or_mae_vals = error_vals
        unnorm_prior = np.exp(-self.alpha * np.array(error_vals))
        self.model_priors = unnorm_prior / unnorm_prior.sum()

    def fit_models_pymc(
        self,
        draws=2000,
        tune=1000,
        chains=4,
        target_accept=0.9,
        init="adapt_diag",
        verbose=True
    ):
        """
        Fit a PyMC model for each set of predictions: y ~ offset + scale * preds + noise.
        Then compute WAIC and store offset, scale from posterior means.

        Parameters
        ----------
        draws : int
            The number of samples (in each chain) to draw from the posterior.
        tune : int
            The number of tuning (burn-in) steps.
        chains : int
            The number of chains to run.
        target_accept : float
            The target acceptance probability for the sampler.
        init : str
            The initialization method for PyMC's sampler. E.g., "adapt_diag", "jitter+adapt_diag", "advi+adapt_diag", "adapt_full", "jitter+adapt_full", "auto".
        """
        if self.model_priors is None:
            # Default to equal priors if not computed yet
            self.model_priors = np.ones(self.M) / self.M

        self.waics = []
        self.traces = []

        for i, preds in enumerate(self.model_predictions):
            with pm.Model():
                offset = pm.Normal("offset", mu=0.0, sigma=10.0)
                scale = pm.Normal("scale", mu=1.0, sigma=1.0)
                sigma = pm.HalfNormal("sigma", sigma=2.0)

                mu = offset + scale * preds
                y_obs = pm.Normal("y_obs", mu=mu, sigma=sigma, observed=self.observations)

                idata = pm.sample(
                    draws=draws,
                    tune=tune,
                    chains=chains,
                    target_accept=target_accept,
                    progressbar=verbose,
                    return_inferencedata=True,
                    idata_kwargs={"log_likelihood": True},
                    init=init  # <<--- Now user-selectable
                )

                # Compute WAIC
                waic_res = az.waic(idata)
                waic_val = -2 * waic_res.elpd_waic  # from ELPD to 'traditional' WAIC
                self.waics.append(waic_val)
                self.traces.append(idata)

            # Posterior means for offset, scale, sigma
            offset_mean = idata.posterior["offset"].mean().item()
            scale_mean = idata.posterior["scale"].mean().item()
            sigma_mean = idata.posterior["sigma"].mean().item()

            self.model_offsets[i] = offset_mean
            self.model_scales[i] = scale_mean
            self.model_sigmas[i] = sigma_mean

    def compute_model_posterior_probs(self):
        """
        Combine model priors and WAIC-based likelihood approximation to get posterior probabilities.
        """
        if self.waics is None:
            raise RuntimeError("Run fit_models_pymc() first.")

        min_waic = np.min(self.waics)
        delta_waic = np.array(self.waics) - min_waic
        likelihood_approx = np.exp(-0.5 * delta_waic)
        unnorm_posterior = self.model_priors * likelihood_approx
        self.posterior_probs = unnorm_posterior / unnorm_posterior.sum()

    def predict_in_sample(self):
        """
        Compute in-sample predictions using offset + scale for each model, weighted by posterior_probs.
        """
        if self.posterior_probs is None:
            raise RuntimeError("Compute posterior probabilities first.")

        bma_preds = np.zeros(self.T)
        for i, preds in enumerate(self.model_predictions):
            corrected = self.model_offsets[i] + self.model_scales[i] * preds
            bma_preds += self.posterior_probs[i] * corrected
        return bma_preds

    def predict(self, future_model_preds_list):
        """
        Compute out-of-sample (future) predictions for each model, applying offset+scale,
        and then weighting by posterior probabilities.
        """
        if self.posterior_probs is None:
            raise RuntimeError("Compute posterior probabilities first.")

        future_len = len(future_model_preds_list[0])
        bma_pred_future = np.zeros(future_len)
        for i, preds in enumerate(future_model_preds_list):
            corrected = self.model_offsets[i] + self.model_scales[i] * preds
            bma_pred_future += self.posterior_probs[i] * corrected
        return bma_pred_future

    def summary(self):
        """
        Print a summary for whichever error metric is used (RMSE or MAE).
        """
        print("=== BMA Summary ===")

        if self.rmse_or_mae_vals is not None:
            err_str = "RMSE" if self.error_metric == "rmse" else "MAE"
            print(f"\n{err_str}:")
            for name, ev in zip(self.model_names, self.rmse_or_mae_vals):
                print(f"{name}: {ev:.3f}")

        if self.model_priors is not None:
            print("\nPrior probabilities:")
            for name, p in zip(self.model_names, self.model_priors):
                print(f"{name}: {p:.3f}")

        if self.waics is not None:
            print("\nWAIC:")
            for name, w in zip(self.model_names, self.waics):
                print(f"{name}: {w:.3f}")

        if self.posterior_probs is not None:
            print("\nPosterior probabilities:")
            for name, pp in zip(self.model_names, self.posterior_probs):
                print(f"{name}: {pp:.3f}")

        print("\nPosterior Means for offset & scale:")
        for i, name in enumerate(self.model_names):
            print(f"{name}: offset={self.model_offsets[i]:.3f}, scale={self.model_scales[i]:.3f}")
        print("=======================")




class WAS_mme_BMA:
    def __init__(self, obs, all_hdcst_models, all_fcst_models, dist_method="gamma", alpha_=0.5, error_metric_="rmse"):
        """
        Wrapper for Bayesian Model Averaging (BMA) applied to Multi-Model Ensemble data in xarray.

        Parameters
        ----------
        obs : xarray.DataArray
            Observed rainfall, shape (T, M, Y, X) or (T, Y, X) if M=1 is squeezed.
        all_hdcst_models : xarray.DataArray
            Hindcast model outputs, shape (T, M, Y, X).
        all_fcst_models : xarray.DataArray
            Forecast model outputs, shape (T, M, Y, X).
        dist_method : str
            Distribution method for post-processing (e.g., 'gamma').
        """
    
        self.obs = obs
        self.all_hdcst_models = all_hdcst_models
        self.all_fcst_models = all_fcst_models
        self.dist_method = dist_method  # For post-processing methods
        self.alpha_ = alpha_
        self.error_metric_ = error_metric_
    
        # Extract model names from 'M' dimension
        self.model_names = list(all_hdcst_models.coords["M"].values)

        # Reshape/clean data for BMA
        self._reshape_data()

        # Initialize BMA
        self.bma = BMA(
            observations=self.obs_flattened,
            model_predictions=self.hdcst_flattened,
            model_names=self.model_names,
            alpha=self.alpha_,
            error_metric = self.error_metric_,
        )

    def _reshape_data(self):
        """
        Flatten the xarray data from (T, M, Y, X) -> 1D arrays, removing positions that have NaNs
        in obs or *any* of the M models. The same approach is used for the forecast data.
        """
    
        # Extract dimensions
        T, M, Y, X = self.all_hdcst_models.shape
    
        # Observations might have an M dim if they are shaped (T, 1, Y, X)
        # If that's the case, we drop that dimension (we only need the actual obs array)
        if "M" in self.obs.dims:
            obs_2d = self.obs.isel(M=0, drop=True)  # shape (T, Y, X)
        else:
            obs_2d = self.obs  # shape (T, Y, X)
    
        # Flatten observations
        self._obs_flattened_raw = obs_2d.values.reshape(-1)
    
        # -------------------------------------------------------------------------
        # 1) Build training mask across *all* hindcast models + obs
        # -------------------------------------------------------------------------
        # Initialize training mask from obs
        
        self.train_nan_mask = np.isnan(self._obs_flattened_raw)
    
        # Flatten each hindcast model and update the training mask
        self.hdcst_flattened = []
        for model_idx in range(M):
            gc.collect()
            # Select and flatten this model
            da_model = self.all_hdcst_models.isel(M=model_idx)  # shape (T, Y, X)
            da_model_flat = da_model.values.reshape(-1)
    
            # Update the mask: positions with NaNs in this model become True
            self.train_nan_mask |= np.isnan(da_model_flat)
    
            # Append to list for now (we'll mask them next)
            self.hdcst_flattened.append(da_model_flat)
    
        # Now mask out the NaNs from obs and from each model
        self.obs_flattened = self._obs_flattened_raw[~self.train_nan_mask]
        self.hdcst_flattened = [m[~self.train_nan_mask] for m in self.hdcst_flattened]
    
        # -------------------------------------------------------------------------
        # 2) Build forecast mask across *all* forecast models
        # -------------------------------------------------------------------------
        # Flatten each forecast model and update the forecast mask
        self._fcst_flattened_raw = obs_2d.isel(T=[0]).values.reshape(-1)
        self.fcst_nan_mask = np.isnan(self._fcst_flattened_raw)
        
        self.fcst_flattened = []
        for model_idx in range(M):
            gc.collect()
            da_fcst = self.all_fcst_models.isel(M=model_idx)
            da_fcst_flat = da_fcst.values.reshape(-1)
            self.fcst_nan_mask |= np.isnan(da_fcst_flat)
            self.fcst_flattened.append(da_fcst_flat)
        # Now store the forecast data, omitting positions of NaNs across any forecast model
        self.fcst_flattened = [fcst_vals[~self.fcst_nan_mask] for fcst_vals in self.fcst_flattened]
            
        # Store shape for rebuilding
        self.T, self.Y, self.X = T, Y, X


    def compute(self, draws, tune, chains, verbose=False, target_accept=0.9, init="jitter+adapt_diag"):
        """
        Parameters
        ----------
        draws : int
            The number of samples (in each chain) to draw from the posterior.
        tune : int
            The number of tuning (burn-in) steps.
        chains : int
            The number of chains to run.
        verbose: bool
            Show progress.
        target_accept : float
            The target acceptance probability for the sampler.
        init : str
            The initialization method for PyMC's sampler. E.g., "adapt_diag", "jitter+adapt_diag", "advi+adapt_diag", "adapt_full", "jitter+adapt_full", "auto".
        Runs the BMA workflow on hindcasts: 
          1) compute_rmse_based_prior
          2) fit_models_pymc
          3) compute_model_posterior_probs
        Returns in-sample predictions as an xarray.DataArray (T, Y, X).
        
        """
        self.bma.compute_error_based_prior()
        self.bma.fit_models_pymc(draws, tune, chains, verbose=verbose, target_accept=target_accept, init=init)
        self.bma.compute_model_posterior_probs()
        

        # In-sample predictions (1D)
        bma_in_sample_flat = self.bma.predict_in_sample()

        # Put predictions back into the original shape with NaNs
        result = np.full_like(self._obs_flattened_raw, np.nan)
        result[~self.train_nan_mask] = bma_in_sample_flat
        result_3d = result.reshape(self.T, self.Y, self.X)

        # Rebuild as DataArray
        if "M" in self.obs.dims:
            obs_2d = self.obs.isel(M=0, drop=True)  # coords for T, Y, X
        else:
            obs_2d = self.obs

        bma_in_sample_da = xr.DataArray(
            data=result_3d,
            dims=("T", "Y", "X"),
            coords=obs_2d.coords
        )
        return bma_in_sample_da

    # ------------------ Probability Calculation Methods ------------------
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
            # Transform thresholds
            first_t = (first_tercile - best_guess) / error_std
            second_t = (second_tercile - best_guess) / error_std

            pred_prob[0, :] = stats.t.cdf(first_t, df=dof)
            pred_prob[1, :] = stats.t.cdf(second_t, df=dof) - stats.t.cdf(first_t, df=dof)
            pred_prob[2, :] = 1 - stats.t.cdf(second_t, df=dof)

        return pred_prob
    @staticmethod
    def calculate_tercile_probabilities_weibull_min(best_guess, error_variance, first_tercile, second_tercile, dof):
        """
        Weibull minimum-based method.
        
        Here, we assume:
          - best_guess is used as the location,
          - error_std (sqrt(error_variance)) as the scale, and 
          - dof (degrees of freedom) as the shape parameter.
        
        Parameters
        ----------
        best_guess : array-like
            Forecast or best guess values.
        error_variance : array-like
            Variance associated with forecast errors.
        first_tercile : array-like
            First tercile threshold values.
        second_tercile : array-like
            Second tercile threshold values.
        dof : float or array-like
            Shape parameter for the Weibull minimum distribution.
            
        Returns
        -------
        pred_prob : np.ndarray
            A 3 x n_time array with probabilities for being below the first tercile,
            between the first and second tercile, and above the second tercile.
        """
        n_time = len(best_guess)
        pred_prob = np.empty((3, n_time))
    
        if np.all(np.isnan(best_guess)):
            pred_prob[:] = np.nan
        else:
            error_std = np.sqrt(error_variance)
    
            # Using the weibull_min CDF with best_guess as loc and error_std as scale.
            # Note: Adjust these assumptions if your application requires a different parameterization.
            pred_prob[0, :] = stats.weibull_min.cdf(first_tercile, c=dof, loc=best_guess, scale=error_std)
            pred_prob[1, :] = stats.weibull_min.cdf(second_tercile, c=dof, loc=best_guess, scale=error_std) - \
                               stats.weibull_min.cdf(first_tercile, c=dof, loc=best_guess, scale=error_std)
            pred_prob[2, :] = 1 - stats.weibull_min.cdf(second_tercile, c=dof, loc=best_guess, scale=error_std)
    
        return pred_prob
    @staticmethod
    def calculate_tercile_probabilities_gamma(best_guess, error_variance, T1, T2, dof=None):
        """Gamma-distribution based method."""
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
        

    @staticmethod
    def calculate_tercile_probabilities_normal(best_guess, error_variance, first_tercile, second_tercile):
        """Normal-distribution based method."""
        n_time = len(best_guess)
        pred_prob = np.empty((3, n_time))
        if np.all(np.isnan(best_guess)):
            pred_prob[:] = np.nan
        else:
            error_std = np.sqrt(error_variance)
            pred_prob[0, :] = norm.cdf(first_tercile, loc=best_guess, scale=error_std)
            pred_prob[1, :] = norm.cdf(second_tercile, loc=best_guess, scale=error_std) - norm.cdf(first_tercile, loc=best_guess, scale=error_std)
            pred_prob[2, :] = 1 - norm.cdf(second_tercile, loc=best_guess, scale=error_std)
        return pred_prob
        

    @staticmethod
    def calculate_tercile_probabilities_lognormal(best_guess, error_variance, first_tercile, second_tercile):
        """Lognormal-distribution based method."""
        n_time = len(best_guess)
        pred_prob = np.empty((3, n_time))
        if np.any(np.isnan(best_guess)) or np.any(np.isnan(error_variance)):
            pred_prob[:] = np.nan
            return pred_prob
        sigma = np.sqrt(np.log(1 + error_variance / (best_guess**2)))
        mu = np.log(best_guess) - sigma**2 / 2
        pred_prob[0, :] = lognorm.cdf(first_tercile, s=sigma, scale=np.exp(mu))
        pred_prob[1, :] = lognorm.cdf(second_tercile, s=sigma, scale=np.exp(mu)) - lognorm.cdf(first_tercile, s=sigma, scale=np.exp(mu))
        pred_prob[2, :] = 1 - lognorm.cdf(second_tercile, s=sigma, scale=np.exp(mu))
        return pred_prob
        

    def compute_prob(self, Predictant, clim_year_start, clim_year_end, hindcast_det):
        """
        Compute tercile probabilities for the hindcast using the chosen distribution method.
        Predictant is an xarray DataArray with dims (T, Y, X).
        """
        if "M" in Predictant.coords:
            Predictant = Predictant.isel(M=0).drop_vars('M').squeeze()
        mask = xr.where(~np.isnan(Predictant.isel(T=0)), 1, np.nan).drop_vars(['T']).squeeze().to_numpy()
        
        # Ensure Predictant is (T, Y, X)
        Predictant = Predictant.transpose('T', 'Y', 'X')
        index_start = Predictant.get_index("T").get_loc(str(clim_year_start)).start
        index_end = Predictant.get_index("T").get_loc(str(clim_year_end)).stop
        rainfall_for_tercile = Predictant.isel(T=slice(index_start, index_end))
        terciles = rainfall_for_tercile.quantile([0.33, 0.67], dim='T')
        error_variance = (Predictant - hindcast_det).var(dim='T')
        dof = len(Predictant.get_index("T")) - 2
        
        if self.dist_method == "t":
            calc_func = self.calculate_tercile_probabilities
            hindcast_prob = xr.apply_ufunc(
                calc_func,
                hindcast_det,
                error_variance,
                terciles.isel(quantile=0).drop_vars('quantile'),
                terciles.isel(quantile=1).drop_vars('quantile'),
                input_core_dims=[('T',), (), (), ()],
                vectorize=True,
                kwargs={'dof': dof},
                dask='parallelized',
                output_core_dims=[('probability', 'T')],
                output_dtypes=['float'],
                dask_gufunc_kwargs={'output_sizes': {'probability': 3}, "allow_rechunk": True}
            )

        elif self.dist_method == "weibull_min":
            calc_func = self.calculate_tercile_probabilities_weibull_min
            hindcast_prob = xr.apply_ufunc(
                calc_func,
                hindcast_det,
                error_variance,
                terciles.isel(quantile=0).drop_vars('quantile'),
                terciles.isel(quantile=1).drop_vars('quantile'),
                input_core_dims=[('T',), (), (), ()],
                vectorize=True,
                kwargs={'dof': dof},
                dask='parallelized',
                output_core_dims=[('probability', 'T')],
                output_dtypes=['float'],
                dask_gufunc_kwargs={'output_sizes': {'probability': 3}, "allow_rechunk": True}
            )
            
        elif self.dist_method == "gamma":
            calc_func = self.calculate_tercile_probabilities_gamma
            hindcast_prob = xr.apply_ufunc(
                calc_func,
                hindcast_det,
                error_variance,
                terciles.isel(quantile=0).drop_vars('quantile'),
                terciles.isel(quantile=1).drop_vars('quantile'),
                input_core_dims=[('T',), (), (), ()],
                vectorize=True,
                dask='parallelized',
                output_core_dims=[('probability', 'T')],
                output_dtypes=['float'],
                dask_gufunc_kwargs={'output_sizes': {'probability': 3}, "allow_rechunk": True}
            )
        elif self.dist_method == "nonparam":
            calc_func = self.calculate_tercile_probabilities_nonparametric
            error_samples = (Predictant - hindcast_det)
            hindcast_prob = xr.apply_ufunc(
                calc_func,
                hindcast_det,
                error_samples,
                terciles.isel(quantile=0).drop_vars('quantile'),
                terciles.isel(quantile=1).drop_vars('quantile'),
                input_core_dims=[('T',), ('T',), (), ()],
                output_core_dims=[('probability', 'T')],
                vectorize=True,
                dask='parallelized',
                output_dtypes=[float],
                dask_gufunc_kwargs={'output_sizes': {'probability': 3}}
            )
        elif self.dist_method == "normal":
            calc_func = self.calculate_tercile_probabilities_normal
            hindcast_prob = xr.apply_ufunc(
                calc_func,
                hindcast_det,
                error_variance,
                terciles.isel(quantile=0).drop_vars('quantile'),
                terciles.isel(quantile=1).drop_vars('quantile'),
                input_core_dims=[('T',), (), (), ()],
                vectorize=True,
                dask='parallelized',
                output_core_dims=[('probability', 'T')],
                output_dtypes=['float'],
                dask_gufunc_kwargs={'output_sizes': {'probability': 3}, "allow_rechunk": True}
            )
        elif self.dist_method == "lognormal":
            calc_func = self.calculate_tercile_probabilities_lognormal
            hindcast_prob = xr.apply_ufunc(
                calc_func,
                hindcast_det,
                error_variance,
                terciles.isel(quantile=0).drop_vars('quantile'),
                terciles.isel(quantile=1).drop_vars('quantile'),
                input_core_dims=[('T',), (), (), ()],
                vectorize=True,
                dask='parallelized',
                output_core_dims=[('probability', 'T')],
                output_dtypes=['float'],
                dask_gufunc_kwargs={'output_sizes': {'probability': 3}, "allow_rechunk": True}
            )
        else:
            raise ValueError(f"Invalid dist_method: {self.dist_method}.")
        
        hindcast_prob = hindcast_prob.assign_coords(probability=('probability', ['PB', 'PN', 'PA']))
        return hindcast_prob.transpose('probability', 'T', 'Y', 'X')

    
    def forecast(self, Predictant, clim_year_start, clim_year_end, hindcast_det):
        """
        Apply BMA offset+scale weights to future forecasts, returning an xarray.DataArray (T, Y, X).
        """
        if self.bma.posterior_probs is None:
            raise RuntimeError("Run train_bma() before predicting.")

        bma_forecast_flat = self.bma.predict(self.fcst_flattened)

        # Re-insert NaNs
        result_fcst = np.full_like(self._fcst_flattened_raw, np.nan)
        result_fcst[~self.fcst_nan_mask] = bma_forecast_flat
        result_fcst_3d = result_fcst.reshape(1, self.Y, self.X)

        fcst_2d = self.all_fcst_models.isel(M=0, drop=True)  # shape (T, Y, X)

        bma_forecast_da = xr.DataArray(
            data=result_fcst_3d,
            dims=("T", "Y", "X"),
            coords=fcst_2d.coords
        )

        year = self.all_fcst_models.coords['T'].values.astype('datetime64[Y]').astype(int)[0] + 1970  # Convert from epoch
        T_value_1 = Predictant.isel(T=0).coords['T'].values  # Get the datetime64 value from da1
        month_1 = T_value_1.astype('datetime64[M]').astype(int) % 12 + 1  # Extract month
        new_T_value = np.datetime64(f"{year}-{month_1:02d}-{1:02d}")
        
        bma_forecast_da = bma_forecast_da.assign_coords(T=xr.DataArray([new_T_value], dims=["T"]))
        bma_forecast_da['T'] = bma_forecast_da['T'].astype('datetime64[ns]')
        
        if "M" in Predictant.coords:
            Predictant = Predictant.isel(M=0).drop_vars('M').squeeze()
        mask = xr.where(~np.isnan(Predictant.isel(T=0)), 1, np.nan).drop_vars(['T']).squeeze().to_numpy()

        
        # Compute tercile probabilities on the predictions
        index_start = Predictant.get_index("T").get_loc(str(clim_year_start)).start
        index_end = Predictant.get_index("T").get_loc(str(clim_year_end)).stop
        rainfall_for_tercile = Predictant.isel(T=slice(index_start, index_end))
        terciles = rainfall_for_tercile.quantile([0.33, 0.67], dim='T')
        error_variance = (Predictant - hindcast_det).var(dim='T')
        dof = len(Predictant.get_index("T")) - 2
        
        if self.dist_method == "t":
            calc_func = self.calculate_tercile_probabilities
            hindcast_prob = xr.apply_ufunc(
                calc_func,
                bma_forecast_da,
                error_variance,
                terciles.isel(quantile=0).drop_vars('quantile'),
                terciles.isel(quantile=1).drop_vars('quantile'),
                input_core_dims=[('T',), (), (), ()],
                vectorize=True,
                kwargs={'dof': dof},
                dask='parallelized',
                output_core_dims=[('probability','T')],
                output_dtypes=['float'],
                dask_gufunc_kwargs={'output_sizes': {'probability': 3}, "allow_rechunk": True}
            )
            
        elif self.dist_method == "weibull_min":
            calc_func = self.calculate_tercile_probabilities_weibull_min
            hindcast_prob = xr.apply_ufunc(
                calc_func,
                hindcast_det,
                error_variance,
                terciles.isel(quantile=0).drop_vars('quantile'),
                terciles.isel(quantile=1).drop_vars('quantile'),
                input_core_dims=[('T',), (), (), ()],
                vectorize=True,
                kwargs={'dof': dof},
                dask='parallelized',
                output_core_dims=[('probability', 'T')],
                output_dtypes=['float'],
                dask_gufunc_kwargs={'output_sizes': {'probability': 3}, "allow_rechunk": True}
            )
            
        elif self.dist_method == "gamma":
            calc_func = self.calculate_tercile_probabilities_gamma
            hindcast_prob = xr.apply_ufunc(
                calc_func,
                bma_forecast_da,
                error_variance,
                terciles.isel(quantile=0).drop_vars('quantile'),
                terciles.isel(quantile=1).drop_vars('quantile'),
                input_core_dims=[('T',), (), (), ()],
                vectorize=True,
                dask='parallelized',
                output_core_dims=[('probability','T')],
                output_dtypes=['float'],
                dask_gufunc_kwargs={'output_sizes': {'probability': 3}, "allow_rechunk": True}
            )
        elif self.dist_method == "normal":
            calc_func = self.calculate_tercile_probabilities_normal
            hindcast_prob = xr.apply_ufunc(
                calc_func,
                bma_forecast_da,
                error_variance,
                terciles.isel(quantile=0).drop_vars('quantile'),
                terciles.isel(quantile=1).drop_vars('quantile'),
                input_core_dims=[('T',), (), (), ()],
                vectorize=True,
                dask='parallelized',
                output_core_dims=[('probability','T')],
                output_dtypes=['float'],
                dask_gufunc_kwargs={'output_sizes': {'probability': 3}, "allow_rechunk": True}
            )
        elif self.dist_method == "lognormal":
            calc_func = self.calculate_tercile_probabilities_lognormal
            hindcast_prob = xr.apply_ufunc(
                calc_func,
                bma_forecast_da,
                error_variance,
                terciles.isel(quantile=0).drop_vars('quantile'),
                terciles.isel(quantile=1).drop_vars('quantile'),
                input_core_dims=[('T',), (), (), ()],
                vectorize=True,
                dask='parallelized',
                output_core_dims=[('probability','T')],
                output_dtypes=['float'],
                dask_gufunc_kwargs={'output_sizes': {'probability': 3}, "allow_rechunk": True}
            )
        elif self.dist_method == "nonparam":
            calc_func = self.calculate_tercile_probabilities_nonparametric
            error_samples = Predictant - hindcast_det
            error_samples = error_samples.rename({'T':'S'})
            hindcast_prob = xr.apply_ufunc(
                calc_func,
                bma_forecast_da,
                error_samples,
                terciles.isel(quantile=0).drop_vars('quantile'),
                terciles.isel(quantile=1).drop_vars('quantile'),
                input_core_dims=[('T',), ('S',), (), ()],
                output_core_dims=[('probability','T')],
                vectorize=True,
                dask='parallelized',
                output_dtypes=[float],
                dask_gufunc_kwargs={'output_sizes': {'probability': 3}, "allow_rechunk": True}
            )
        else:
            raise ValueError(f"Invalid dist_method: {self.dist_method}.")
            
        hindcast_prob = hindcast_prob.assign_coords(probability=('probability', ['PB', 'PN', 'PA']))
        return bma_forecast_da * mask, mask * hindcast_prob.transpose('probability', 'T', 'Y', 'X')    

    def summary(self):
        """ Print BMA summary information. """
        self.bma.summary()


class WAS_mme_xcELR:
    """
    Extended Logistic Regression (ELR) for Multi-Model Ensemble (MME) forecasting derived from xcast package.

    This class implements an Extended Logistic Regression for probabilistic forecasting,
    directly computing tercile probabilities without requiring separate probability calculations.

    Parameters
    ----------
    elm_kwargs : dict, optional
        Keyword arguments to pass to the xcast ELR model. If None, an empty dictionary is used.
        Default is None.
    """
    def __init__(self, elm_kwargs=None):
        if elm_kwargs is None:
            self.elm_kwargs = {}
        else:
            self.elm_kwargs = elm_kwargs     

    def compute_model(self, X_train, y_train, X_test):
        """
        Compute probabilistic hindcast using the ELR model.

        Fits the ELR model on training data and predicts tercile probabilities for the test data.
        Applies regridding and drymasking to ensure data consistency.

        Parameters
        ----------
        X_train : xarray.DataArray
            Training predictor data with dimensions (T, M, Y, X).
        y_train : xarray.DataArray
            Training predictand data with dimensions (T, Y, X).
        X_test : xarray.DataArray
            Testing predictor data with dimensions (T, M, Y, X).

        Returns
        -------
        result_ : xarray.DataArray
            Tercile probabilities with dimensions (probability, T, Y, X), where probability
            includes ['PB', 'PN', 'PA'] (below-normal, normal, above-normal).
        """
        
        X_train = xc.regrid(X_train,y_train.X,y_train.Y)
        X_test = xc.regrid(X_test,y_train.X,y_train.Y)

        drymask = xc.drymask(
            y_train, dry_threshold=10, quantile_threshold=0.2
                        )
        X_train = X_train*drymask
        X_test = X_test*drymask
        
        model = xc.ELR() # **self.elm_kwargs
        model.fit(X_train, y_train)
        result_ = model.predict_proba(X_test)
        result_ = result_.rename({'S':'T','M':'probability'})
        result_ = result_.assign_coords(probability=('probability', ['PB','PN','PA']))
        return result_.transpose('probability', 'T', 'Y', 'X')

    def forecast(self, Predictant, clim_year_start, clim_year_end, hindcast_det, Predictor_for_year):
        """
        Generate probabilistic forecast for a target year using the ELR model.

        Fits the ELR model on hindcast data and predicts tercile probabilities for the target year.
        Applies regridding and drymasking to ensure data consistency.

        Parameters
        ----------
        Predictant : xarray.DataArray
            Observed predictand data with dimensions (T, Y, X) or (T, M, Y, X).
        clim_year_start : int or str
            Start year of the climatology period (not used in this method).
        clim_year_end : int or str
            End year of the climatology period (not used in this method).
        hindcast_det : xarray.DataArray
            Deterministic hindcast data with dimensions (T, M, Y, X).
        Predictor_for_year : xarray.DataArray
            Predictor data for the target year with dimensions (T, M, Y, X).

        Returns
        -------
        hindcast_prob : xarray.DataArray
            Tercile probabilities with dimensions (probability, T, Y, X), where probability
            includes ['PB', 'PN', 'PA'] (below-normal, normal, above-normal).
        """

        clim_year_end = clim_year_end
        clim_year_start = clim_year_start
        hindcast_det = xc.regrid(hindcast_det,Predictant.X,Predictant.Y)
        Predictor_for_year = xc.regrid(Predictor_for_year,Predictant.X,Predictant.Y)

        drymask = xc.drymask(
            Predictant, dry_threshold=10, quantile_threshold=0.2
                        )
        hindcast_det_ = hindcast_det*drymask
        Predictor_for_year = Predictor_for_year*drymask
        
        model = xc.ELR() 
        model.fit(hindcast_det, Predictant)
        result_ = model.predict_proba(Predictor_for_year)
        result_ = result_.rename({'S':'T','M':'probability'}).transpose('probability','T', 'Y', 'X')
        hindcast_prob = result_.assign_coords(probability=('probability', ['PB', 'PN', 'PA']))
        
        return hindcast_prob.transpose('probability', 'T', 'Y', 'X').load()

    
class WAS_mme_xcELM:
    """
    Extreme Learning Machine (ELM) for Multi-Model Ensemble (MME) forecasting derived from xcast.

    This class implements an Extreme Learning Machine model for deterministic forecasting,
    with optional tercile probability calculations using various statistical distributions.

    Parameters
    ----------
    elm_kwargs : dict, optional
        Keyword arguments to pass to the xcast ELM model. If None, default parameters are used:
        {'regularization': 10, 'hidden_layer_size': 5, 'activation': 'lin', 'preprocessing': 'none', 'n_estimators': 5}.
        Default is None.
    dist_method : str, optional
        Distribution method for tercile probability calculations ('t', 'gamma', 'nonparam', 'normal', 'lognormal', 'weibull_min').
        Default is 'gamma'.
    """
    def __init__(self, elm_kwargs=None, dist_method="gamma"):
        if elm_kwargs is None:
            self.elm_kwargs = {
                'regularization': 10,
                'hidden_layer_size': 5,
                'activation': 'lin',  # 'sigm', 'tanh', 'lin', 'leaky', 'relu', 'softplus'],
                'preprocessing': 'none',  # 'minmax', 'std', 'none' ],
                'n_estimators': 5,
                            }
        else:
            self.elm_kwargs = elm_kwargs
            
        self.dist_method = dist_method         

    def compute_model(self, X_train, y_train, X_test):
        """
        Compute deterministic hindcast using the ELM model.

        Fits the ELM model on training data and predicts deterministic values for the test data.
        Applies regridding and drymasking to ensure data consistency.

        Parameters
        ----------
        X_train : xarray.DataArray
            Training predictor data with dimensions (T, M, Y, X).
        y_train : xarray.DataArray
            Training predictand data with dimensions (T, Y, X).
        X_test : xarray.DataArray
            Testing predictor data with dimensions (T, M, Y, X).

        Returns
        -------
        result_ : xarray.DataArray
            Deterministic hindcast with dimensions (T, Y, X).
        """

        X_train = xc.regrid(X_train,y_train.X,y_train.Y)
        X_test = xc.regrid(X_test,y_train.X,y_train.Y)
        
        # X_train = X_train.fillna(0)
        # y_train = y_train.fillna(0)
        drymask = xc.drymask(
            y_train, dry_threshold=10, quantile_threshold=0.2
                        )
        X_train = X_train*drymask
        X_test = X_test*drymask
        
        model = xc.ELM(**self.elm_kwargs) 
        model.fit(X_train, y_train)
        result_ = model.predict(X_test)
        return result_.rename({'S':'T'}).transpose('T', 'M', 'Y', 'X').drop_vars('M').squeeze()

    # ------------------ Probability Calculation Methods ------------------
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
            # Transform thresholds
            first_t = (first_tercile - best_guess) / error_std
            second_t = (second_tercile - best_guess) / error_std

            pred_prob[0, :] = stats.t.cdf(first_t, df=dof)
            pred_prob[1, :] = stats.t.cdf(second_t, df=dof) - stats.t.cdf(first_t, df=dof)
            pred_prob[2, :] = 1 - stats.t.cdf(second_t, df=dof)

        return pred_prob
    @staticmethod
    def calculate_tercile_probabilities_weibull_min(best_guess, error_variance, first_tercile, second_tercile, dof):
        """
        Weibull minimum-based method.
        
        Here, we assume:
          - best_guess is used as the location,
          - error_std (sqrt(error_variance)) as the scale, and 
          - dof (degrees of freedom) as the shape parameter.
        
        Parameters
        ----------
        best_guess : array-like
            Forecast or best guess values.
        error_variance : array-like
            Variance associated with forecast errors.
        first_tercile : array-like
            First tercile threshold values.
        second_tercile : array-like
            Second tercile threshold values.
        dof : float or array-like
            Shape parameter for the Weibull minimum distribution.
            
        Returns
        -------
        pred_prob : np.ndarray
            A 3 x n_time array with probabilities for being below the first tercile,
            between the first and second tercile, and above the second tercile.
        """
        n_time = len(best_guess)
        pred_prob = np.empty((3, n_time))
    
        if np.all(np.isnan(best_guess)):
            pred_prob[:] = np.nan
        else:
            error_std = np.sqrt(error_variance)
    
            # Using the weibull_min CDF with best_guess as loc and error_std as scale.
            # Note: Adjust these assumptions if your application requires a different parameterization.
            pred_prob[0, :] = stats.weibull_min.cdf(first_tercile, c=dof, loc=best_guess, scale=error_std)
            pred_prob[1, :] = stats.weibull_min.cdf(second_tercile, c=dof, loc=best_guess, scale=error_std) - \
                               stats.weibull_min.cdf(first_tercile, c=dof, loc=best_guess, scale=error_std)
            pred_prob[2, :] = 1 - stats.weibull_min.cdf(second_tercile, c=dof, loc=best_guess, scale=error_std)
    
        return pred_prob
    @staticmethod
    def calculate_tercile_probabilities_gamma(best_guess, error_variance, T1, T2, dof=None):
        """Gamma-distribution based method."""
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

    @staticmethod
    def calculate_tercile_probabilities_normal(best_guess, error_variance, first_tercile, second_tercile):
        """Normal-distribution based method."""
        n_time = len(best_guess)
        pred_prob = np.empty((3, n_time))
        if np.all(np.isnan(best_guess)):
            pred_prob[:] = np.nan
        else:
            error_std = np.sqrt(error_variance)
            pred_prob[0, :] = norm.cdf(first_tercile, loc=best_guess, scale=error_std)
            pred_prob[1, :] = norm.cdf(second_tercile, loc=best_guess, scale=error_std) - norm.cdf(first_tercile, loc=best_guess, scale=error_std)
            pred_prob[2, :] = 1 - norm.cdf(second_tercile, loc=best_guess, scale=error_std)
        return pred_prob

    @staticmethod
    def calculate_tercile_probabilities_lognormal(best_guess, error_variance, first_tercile, second_tercile):
        """Lognormal-distribution based method."""
        n_time = len(best_guess)
        pred_prob = np.empty((3, n_time))
        if np.any(np.isnan(best_guess)) or np.any(np.isnan(error_variance)):
            pred_prob[:] = np.nan
            return pred_prob
        sigma = np.sqrt(np.log(1 + error_variance / (best_guess**2)))
        mu = np.log(best_guess) - sigma**2 / 2
        pred_prob[0, :] = lognorm.cdf(first_tercile, s=sigma, scale=np.exp(mu))
        pred_prob[1, :] = lognorm.cdf(second_tercile, s=sigma, scale=np.exp(mu)) - lognorm.cdf(first_tercile, s=sigma, scale=np.exp(mu))
        pred_prob[2, :] = 1 - lognorm.cdf(second_tercile, s=sigma, scale=np.exp(mu))
        return pred_prob

    def compute_prob(self, Predictant, clim_year_start, clim_year_end, hindcast_det):
        """
        Compute tercile probabilities for hindcast data.

        Calculates probabilities for below-normal, normal, and above-normal categories using
        the specified distribution method, based on climatological terciles.

        Parameters
        ----------
        Predictant : xarray.DataArray
            Observed predictand data with dimensions (T, Y, X).
        clim_year_start : int or str
            Start year of the climatology period.
        clim_year_end : int or str
            End year of the climatology period.
        hindcast_det : xarray.DataArray
            Deterministic hindcast data with dimensions (T, Y, X).

        Returns
        -------
        hindcast_prob : xarray.DataArray
            Tercile probabilities with dimensions (probability, T, Y, X), where probability
            includes ['PB', 'PN', 'PA'] (below-normal, normal, above-normal).
        """
        if "M" in Predictant.coords:
            Predictant = Predictant.isel(M=0).drop_vars('M').squeeze()
        mask = xr.where(~np.isnan(Predictant.isel(T=0)), 1, np.nan).drop_vars(['T']).squeeze().to_numpy()

        # Ensure Predictant is (T, Y, X)
        Predictant = Predictant.transpose('T', 'Y', 'X')
        index_start = Predictant.get_index("T").get_loc(str(clim_year_start)).start
        index_end = Predictant.get_index("T").get_loc(str(clim_year_end)).stop
        rainfall_for_tercile = Predictant.isel(T=slice(index_start, index_end))
        terciles = rainfall_for_tercile.quantile([0.33, 0.67], dim='T')
        error_variance = (Predictant - hindcast_det).var(dim='T')
        dof = len(Predictant.get_index("T")) - 2
        
        if self.dist_method == "t":
            calc_func = self.calculate_tercile_probabilities
            hindcast_prob = xr.apply_ufunc(
                calc_func,
                hindcast_det,
                error_variance,
                terciles.isel(quantile=0).drop_vars('quantile'),
                terciles.isel(quantile=1).drop_vars('quantile'),
                input_core_dims=[('T',), (), (), ()],
                vectorize=True,
                kwargs={'dof': dof},
                dask='parallelized',
                output_core_dims=[('probability', 'T')],
                output_dtypes=['float'],
                dask_gufunc_kwargs={'output_sizes': {'probability': 3}, "allow_rechunk": True}
            )
        elif self.dist_method == "weibull_min":
            calc_func = self.calculate_tercile_probabilities_weibull_min
            hindcast_prob = xr.apply_ufunc(
                calc_func,
                hindcast_det,
                error_variance,
                terciles.isel(quantile=0).drop_vars('quantile'),
                terciles.isel(quantile=1).drop_vars('quantile'),
                input_core_dims=[('T',), (), (), ()],
                vectorize=True,
                kwargs={'dof': dof},
                dask='parallelized',
                output_core_dims=[('probability', 'T')],
                output_dtypes=['float'],
                dask_gufunc_kwargs={'output_sizes': {'probability': 3}, "allow_rechunk": True}
            )
            
        elif self.dist_method == "gamma":
            calc_func = self.calculate_tercile_probabilities_gamma
            hindcast_prob = xr.apply_ufunc(
                calc_func,
                hindcast_det,
                error_variance,
                terciles.isel(quantile=0).drop_vars('quantile'),
                terciles.isel(quantile=1).drop_vars('quantile'),
                input_core_dims=[('T',), (), (), ()],
                vectorize=True,
                dask='parallelized',
                output_core_dims=[('probability', 'T')],
                output_dtypes=['float'],
                dask_gufunc_kwargs={'output_sizes': {'probability': 3}, "allow_rechunk": True}
            )
        elif self.dist_method == "nonparam":
            calc_func = self.calculate_tercile_probabilities_nonparametric
            error_samples = (Predictant - hindcast_det)  
            hindcast_prob = xr.apply_ufunc(
                calc_func,
                hindcast_det,
                error_samples,
                terciles.isel(quantile=0).drop_vars('quantile'),
                terciles.isel(quantile=1).drop_vars('quantile'),
                input_core_dims=[('T',), ('T',), (), ()],
                output_core_dims=[('probability', 'T')],
                vectorize=True,
                dask='parallelized',
                output_dtypes=[float],
                dask_gufunc_kwargs={'output_sizes': {'probability': 3}, "allow_rechunk": True}
            )
        elif self.dist_method == "normal":
            calc_func = self.calculate_tercile_probabilities_normal
            hindcast_prob = xr.apply_ufunc(
                calc_func,
                hindcast_det,
                error_variance,
                terciles.isel(quantile=0).drop_vars('quantile'),
                terciles.isel(quantile=1).drop_vars('quantile'),
                input_core_dims=[('T',), (), (), ()],
                vectorize=True,
                dask='parallelized',
                output_core_dims=[('probability', 'T')],
                output_dtypes=['float'],
                dask_gufunc_kwargs={'output_sizes': {'probability': 3}, "allow_rechunk": True}
            )
        elif self.dist_method == "lognormal":
            calc_func = self.calculate_tercile_probabilities_lognormal
            hindcast_prob = xr.apply_ufunc(
                calc_func,
                hindcast_det,
                error_variance,
                terciles.isel(quantile=0).drop_vars('quantile'),
                terciles.isel(quantile=1).drop_vars('quantile'),
                input_core_dims=[('T',), (), (), ()],
                vectorize=True,
                dask='parallelized',
                output_core_dims=[('probability', 'T')],
                output_dtypes=['float'],
                dask_gufunc_kwargs={'output_sizes': {'probability': 3}, "allow_rechunk": True}
            )
        else:
            raise ValueError(f"Invalid dist_method: {self.dist_method}.")
        
        hindcast_prob = hindcast_prob.assign_coords(probability=('probability', ['PB', 'PN', 'PA']))
        return hindcast_prob.transpose('probability', 'T', 'Y', 'X')

    def forecast(self, Predictant, clim_year_start, clim_year_end, hindcast_det, hindcast_det_cross_val, Predictor_for_year):
        """
        Generate deterministic and probabilistic forecast for a target year using the ELM model.

        Fits the ELM model on hindcast data, predicts deterministic values for the target year,
        and computes tercile probabilities. Applies regridding, drymasking, and standardization.

        Parameters
        ----------
        Predictant : xarray.DataArray
            Observed predictand data with dimensions (T, Y, X) or (T, M, Y, X).
        clim_year_start : int or str
            Start year of the climatology period.
        clim_year_end : int or str
            End year of the climatology period.
        hindcast_det : xarray.DataArray
            Deterministic hindcast data for training with dimensions (T, M, Y, X).
        hindcast_det_cross_val : xarray.DataArray
            Deterministic hindcast data for error estimation with dimensions (T, Y, X).
        Predictor_for_year : xarray.DataArray
            Predictor data for the target year with dimensions (T, M, Y, X).

        Returns
        -------
        result_ : xarray.DataArray
            Deterministic forecast with dimensions (T, Y, X).
        hindcast_prob : xarray.DataArray
            Tercile probabilities with dimensions (probability, T, Y, X), where probability
            includes ['PB', 'PN', 'PA'] (below-normal, normal, above-normal).
        """

        hindcast_det = xc.regrid(hindcast_det,Predictant.X,Predictant.Y)
        Predictor_for_year = xc.regrid(Predictor_for_year,Predictant.X,Predictant.Y)

        drymask = xc.drymask(
            Predictant, dry_threshold=10, quantile_threshold=0.2
                        )
        hindcast_det = hindcast_det*drymask
        Predictor_for_year = Predictor_for_year*drymask
        
        # hindcast_det_ = hindcast_det.fillna(0)
        # Predictant_ = Predictant.fillna(0)
        # Predictor_for_year_ = Predictor_for_year.fillna(0)

        model = xc.ELM(**self.elm_kwargs) 
        model.fit(hindcast_det, Predictant)
        result_ = model.predict(Predictor_for_year)
        result_ = result_.rename({'S':'T'}).transpose('T', 'M', 'Y', 'X').drop_vars('M').squeeze('M').load()

        year = Predictor_for_year.coords['S'].values.astype('datetime64[Y]').astype(int)[0] + 1970  # Convert from epoch
        T_value_1 = Predictant.isel(T=0).coords['T'].values  # Get the datetime64 value from da1
        month_1 = T_value_1.astype('datetime64[M]').astype(int) % 12 + 1  # Extract month
        new_T_value = np.datetime64(f"{year}-{month_1:02d}-{1:02d}")
        result_ = result_.assign_coords(T=xr.DataArray([new_T_value], dims=["T"]))
        result_['T'] = result_['T'].astype('datetime64[ns]')

        if "M" in Predictant.coords:
            Predictant = Predictant.isel(M=0).drop_vars('M').squeeze()
        mask = xr.where(~np.isnan(Predictant.isel(T=0)), 1, np.nan).drop_vars(['T']).squeeze().to_numpy()


        # Compute tercile probabilities on the predictions
        index_start = Predictant.get_index("T").get_loc(str(clim_year_start)).start
        index_end = Predictant.get_index("T").get_loc(str(clim_year_end)).stop
        rainfall_for_tercile = Predictant.isel(T=slice(index_start, index_end))
        terciles = rainfall_for_tercile.quantile([0.33, 0.67], dim='T')
        error_variance = (Predictant - hindcast_det_cross_val).var(dim='T')
        dof = len(Predictant.get_index("T")) - 2
        
        if self.dist_method == "t":
            calc_func = self.calculate_tercile_probabilities
            hindcast_prob = xr.apply_ufunc(
                calc_func,
                result_,
                error_variance,
                terciles.isel(quantile=0).drop_vars('quantile'),
                terciles.isel(quantile=1).drop_vars('quantile'),
                input_core_dims=[('T',), (), (), ()],
                vectorize=True,
                kwargs={'dof': dof},
                dask='parallelized',
                output_core_dims=[('probability','T')],
                output_dtypes=['float'],
                dask_gufunc_kwargs={'output_sizes': {'probability': 3}, "allow_rechunk": True}
            )
        elif self.dist_method == "weibull_min":
            calc_func = self.calculate_tercile_probabilities_weibull_min
            hindcast_prob = xr.apply_ufunc(
                calc_func,
                hindcast_det,
                error_variance,
                terciles.isel(quantile=0).drop_vars('quantile'),
                terciles.isel(quantile=1).drop_vars('quantile'),
                input_core_dims=[('T',), (), (), ()],
                vectorize=True,
                kwargs={'dof': dof},
                dask='parallelized',
                output_core_dims=[('probability', 'T')],
                output_dtypes=['float'],
                dask_gufunc_kwargs={'output_sizes': {'probability': 3}, "allow_rechunk": True}
            )
            
        elif self.dist_method == "gamma":
            calc_func = self.calculate_tercile_probabilities_gamma
            hindcast_prob = xr.apply_ufunc(
                calc_func,
                result_,
                error_variance,
                terciles.isel(quantile=0).drop_vars('quantile'),
                terciles.isel(quantile=1).drop_vars('quantile'),
                input_core_dims=[('T',), (), (), ()],
                vectorize=True,
                dask='parallelized',
                output_core_dims=[('probability','T')],
                output_dtypes=['float'],
                dask_gufunc_kwargs={'output_sizes': {'probability': 3}, "allow_rechunk": True}
            )
        elif self.dist_method == "normal":
            calc_func = self.calculate_tercile_probabilities_normal
            hindcast_prob = xr.apply_ufunc(
                calc_func,
                result_,
                error_variance,
                terciles.isel(quantile=0).drop_vars('quantile'),
                terciles.isel(quantile=1).drop_vars('quantile'),
                input_core_dims=[('T',), (), (), ()],
                vectorize=True,
                dask='parallelized',
                output_core_dims=[('probability','T')],
                output_dtypes=['float'],
                dask_gufunc_kwargs={'output_sizes': {'probability': 3}, "allow_rechunk": True}
            )
        elif self.dist_method == "lognormal":
            calc_func = self.calculate_tercile_probabilities_lognormal
            hindcast_prob = xr.apply_ufunc(
                calc_func,
                result_,
                error_variance,
                terciles.isel(quantile=0).drop_vars('quantile'),
                terciles.isel(quantile=1).drop_vars('quantile'),
                input_core_dims=[('T',), (), (), ()],
                vectorize=True,
                dask='parallelized',
                output_core_dims=[('probability','T')],
                output_dtypes=['float'],
                dask_gufunc_kwargs={'output_sizes': {'probability': 3}, "allow_rechunk": True}
            )
        elif self.dist_method == "nonparam":
            calc_func = self.calculate_tercile_probabilities_nonparametric
            error_samples = Predictant - hindcast_det_cross_val
            error_samples = error_samples.rename({'T':'S'})
            hindcast_prob = xr.apply_ufunc(
                calc_func,
                result_,
                error_samples,
                terciles.isel(quantile=0).drop_vars('quantile'),
                terciles.isel(quantile=1).drop_vars('quantile'),
                input_core_dims=[('T',), ('S',), (), ()],
                output_core_dims=[('probability','T')],
                vectorize=True,
                dask='parallelized',
                output_dtypes=[float],
                dask_gufunc_kwargs={'output_sizes': {'probability': 3}, "allow_rechunk": True}
            )
        else:
            raise ValueError(f"Invalid dist_method: {self.dist_method}.")
        
        hindcast_prob = hindcast_prob.assign_coords(probability=('probability', ['PB', 'PN', 'PA']))
        return result_ * mask, mask * hindcast_prob.transpose('probability', 'T', 'Y', 'X')



class HPELMWrapper(BaseEstimator, RegressorMixin):
    """
    Wrapper for HPELM to make it compatible with scikit-learn's RandomizedSearchCV.
    """
    def __init__(self, neurons=10, activation='sigm', norm=1.0, random_state=42):
        self.neurons = neurons
        self.activation = activation
        self.norm = norm
        self.random_state = random_state
        self.model = None

    def fit(self, X, y):
        self.model = HPELM(inputs=X.shape[1], outputs=1, classification='r', norm=self.norm)
        self.model.add_neurons(self.neurons, self.activation)
        self.model.train(X, y, 'r')
        return self

    def predict(self, X):
        return self.model.predict(X).ravel()

    def get_params(self, deep=True):
        return {'neurons': self.neurons, 'activation': self.activation, 'norm': self.norm, 'random_state': self.random_state}

    def set_params(self, **params):
        for param, value in params.items():
            setattr(self, param, value)
        return self


class WAS_mme_hpELM:
    """
    Extreme Learning Machine (ELM) based Multi-Model Ensemble (MME) forecasting using hpelm library.
    This class implements a single-model forecasting approach using HPELM for deterministic predictions,
    with optional tercile probability calculations using various statistical distributions.
    Implements hyperparameter optimization via randomized search.
    Parameters
    ----------
    neurons_range : list of int, optional
        List of neuron counts to tune for HPELM (default is [10, 20, 50, 100]).
    activation_options : list of str, optional
        Activation functions to tune for HPELM (default is ['sigm', 'tanh', 'relu', 'rbf_linf', 'rbf_gauss']).
    norm_range : list of float, optional
        Regularization parameters to tune for HPELM (default is [0.1, 1.0, 10.0, 100.0]).
    random_state : int, optional
        Seed for reproducibility (default is 42).
    dist_method : str, optional
        Distribution method for tercile probabilities ('t', 'gamma', etc.) (default is 'gamma').
    n_iter_search : int, optional
        Number of iterations for randomized search (default is 10).
    cv_folds : int, optional
        Number of cross-validation folds (default is 3).
    n_clusters : int, optional
        Number of clusters for homogenized zones (default is 4).
    """
    def __init__(self,
                 neurons_range=[10, 20, 50, 100],
                 activation_options=['sigm', 'tanh', 'relu', 'rbf_linf', 'rbf_gauss'],
                 norm_range=[0.1, 1.0, 10.0, 100.0],
                 random_state=42,
                 dist_method="gamma",
                 n_iter_search=10,
                 cv_folds=3,
                 n_clusters=4):
        self.neurons_range = neurons_range
        self.activation_options = activation_options
        self.norm_range = norm_range
        self.random_state = random_state
        self.dist_method = dist_method
        self.n_iter_search = n_iter_search
        self.cv_folds = cv_folds
        self.n_clusters = n_clusters
        self.hpelm = None

    def compute_hyperparameters(self, Predictors, Predictand, clim_year_start, clim_year_end):
        """
        Independently computes the best hyperparameters using randomized search on stacked training data for each homogenized zone.
        Parameters
        ----------
        Predictors : xarray.DataArray
            Training predictor data with dimensions (T, M, Y, X).
        Predictand : xarray.DataArray
            Training predictand data with dimensions (T, Y, X).
        clim_year_start : int
            Start year of the climatology period.
        clim_year_end : int
            End year of the climatology period.
        Returns
        -------
        best_params_dict : dict
            Best hyperparameters for each cluster.
        cluster_da : xarray.DataArray
            Cluster labels with dimensions (Y, X).
        """
        if "M" in Predictand.coords:
            Predictand = Predictand.isel(M=0).drop_vars('M').squeeze()

            
        X_train_std = standardize_timeseries(Predictors, clim_year_start, clim_year_end)
        Predictand.name = "varname"

        # # Cluster on standardized predictand time series
        # y_for_cluster = y_train_std.stack(space=('Y', 'X')).transpose('space', 'T').values
        # print(y_for_cluster)
        # print(y_for_cluster.shape)
        # finite_mask = np.all(np.isfinite(y_for_cluster), axis=1)
        # y_cluster = y_for_cluster[finite_mask]

        # kmeans = KMeans(n_clusters=self.n_clusters, random_state=self.random_state)
        # labels = kmeans.fit_predict(y_cluster)

        # full_labels = np.full(y_for_cluster.shape[0], np.nan)   # -1
        # full_labels[finite_mask] = labels

        # cluster_da = xr.DataArray(
        #     full_labels.reshape(len(y_train_std['Y']), len(y_train_std['X'])),
        #     coords={'Y': y_train_std['Y'], 'X': y_train_std['X']},
        #     dims=['Y', 'X']
        # )
        # cluster_da.plot()


        # Step 1: Perform KMeans clustering based on predictand's spatial distribution
        kmeans = KMeans(n_clusters=self.n_clusters, random_state=self.random_state)
        Predictand_dropna = Predictand.to_dataframe().reset_index().dropna().drop(columns=['T'])
        variable_column = Predictand_dropna.columns[2]
        Predictand_dropna['cluster'] = kmeans.fit_predict(
           Predictand_dropna[[variable_column]]
        )
        
        # Convert cluster assignments back into an xarray structure
        df_unique = Predictand_dropna.drop_duplicates(subset=['Y', 'X'])
        dataset = df_unique.set_index(['Y', 'X']).to_xarray()
        mask = xr.where(~np.isnan(Predictand.isel(T=0)), 1, np.nan)
        Cluster = (dataset['cluster'] * mask)
               
        # Align cluster array with the predictand array
        xarray1, xarray2 = xr.align(Predictand, Cluster, join="outer")
        
        # Identify unique cluster labels
        clusters = np.unique(xarray2)
        clusters = clusters[~np.isnan(clusters)]
        cluster_da = xarray2

        y_train_std = standardize_timeseries(Predictand, clim_year_start, clim_year_end)
        X_train_std['T'] = y_train_std['T']
        # Prepare parameter distributions
        param_dist = {
            'neurons': self.neurons_range,
            'activation': self.activation_options,
            'norm': self.norm_range
        }

        best_params_dict = {}
        for c in clusters: #range(self.n_clusters):
            mask_3d = (cluster_da == c).expand_dims({'T': y_train_std['T']})
            X_stacked_c = X_train_std.where(mask_3d).stack(sample=('T', 'Y', 'X')).transpose('sample', 'M').values
            y_stacked_c = y_train_std.where(mask_3d).stack(sample=('T', 'Y', 'X')).values.ravel()

            nan_mask_c = np.any(~np.isfinite(X_stacked_c), axis=1) | ~np.isfinite(y_stacked_c)
            X_clean_c = X_stacked_c[~nan_mask_c]
            y_clean_c = y_stacked_c[~nan_mask_c]

            if len(X_clean_c) == 0:
                continue

            # Initialize HPELM wrapper for scikit-learn compatibility
            model = HPELMWrapper(random_state=self.random_state)

            # Randomized search
            random_search = RandomizedSearchCV(
                model, param_distributions=param_dist, n_iter=self.n_iter_search,
                cv=self.cv_folds, scoring='neg_mean_squared_error',
                random_state=self.random_state, error_score=np.nan
            )
            random_search.fit(X_clean_c, y_clean_c)
            best_params_dict[c] = random_search.best_params_

        return best_params_dict, cluster_da

    def compute_model(self, X_train, y_train, X_test, y_test, best_params=None, cluster_da=None):
        """
        Compute deterministic hindcast using the HPELM model with injected hyperparameters for each zone.
        Parameters
        ----------
        X_train : xarray.DataArray
            Training predictor data with dimensions (T, M, Y, X).
        y_train : xarray.DataArray
            Training predictand data with dimensions (T, Y, X).
        X_test : xarray.DataArray
            Testing predictor data with dimensions (T, M, Y, X).
        y_test : xarray.DataArray
            Testing predictand data with dimensions (T, Y, X).
        best_params : dict, optional
            Pre-computed best hyperparameters per cluster. If None, computes internally.
        cluster_da : xarray.DataArray, optional
            Pre-computed cluster labels. If None, computes internally.
        Returns
        -------
        predicted_da : xarray.DataArray
            Deterministic hindcast with dimensions (T, Y, X).
        """
        # Standardize inputs
        X_train_std = X_train
        y_train_std = y_train
        X_test_std = X_test
        y_test_std = y_test

        # Extract coordinate variables from X_test
        time = X_test_std['T']
        lat = X_test_std['Y']
        lon = X_test_std['X']
        n_time = len(time)
        n_lat = len(lat)
        n_lon = len(lon)

        # Use provided best_params and cluster_da or compute if None
        if best_params is None:
            best_params, cluster_da = self.compute_hyperparameters(X_train_std, y_train_std, clim_year_start, clim_year_end)

        # Initialize predictions array
        predictions = np.full((n_time, n_lat, n_lon), np.nan)

        self.hpelm = {}  # Dictionary to store models per cluster

        for c in range(self.n_clusters):
            if c not in best_params:
                continue

            bp = best_params[c]

            # Mask for this cluster
            mask_3d_train = (cluster_da == c).expand_dims({'T': X_train_std['T']})
            mask_3d_test = (cluster_da == c).expand_dims({'T': X_test_std['T']})

            # Stack training data for cluster
            X_train_stacked_c = X_train_std.where(mask_3d_train).stack(sample=('T', 'Y', 'X')).transpose('sample', 'M').values
            y_train_stacked_c = y_train_std.where(mask_3d_train).stack(sample=('T', 'Y', 'X')).values.ravel()

            train_nan_mask_c = np.any(~np.isfinite(X_train_stacked_c), axis=1) | ~np.isfinite(y_train_stacked_c)
            X_train_clean_c = X_train_stacked_c[~train_nan_mask_c]
            y_train_clean_c = y_train_stacked_c[~train_nan_mask_c]

            # Stack testing data for cluster
            X_test_stacked_c = X_test_std.where(mask_3d_test).stack(sample=('T', 'Y', 'X')).transpose('sample', 'M').values
            y_test_stacked_c = y_test_std.where(mask_3d_test).stack(sample=('T', 'Y', 'X')).values.ravel()

            test_nan_mask_c = np.any(~np.isfinite(X_test_stacked_c), axis=1) | ~np.isfinite(y_test_stacked_c)
            X_test_clean_c = X_test_stacked_c[~test_nan_mask_c]

            # Initialize and train the HPELM model for this cluster
            hpelm_c = HPELM(
                inputs=X_train_clean_c.shape[1],
                outputs=1,
                classification='r',
                norm=bp['norm']
            )

            # Initialize weights and biases for the neurons
            # Use a random number generator for reproducibility
            # rng = np.random.default_rng(1234)    # isolated RNG
            # n = bp['neurons']
            # W = rng.standard_normal((X_train_clean_c.shape[1], n))
            # B = rng.standard_normal(n)
            # hpelm_c.add_neurons(bp['neurons'], bp['activation'],W=W, B=B)
            
            hpelm_c.add_neurons(bp['neurons'], bp['activation'])
            hpelm_c.train(X_train_clean_c, y_train_clean_c, 'r')
            self.hpelm[c] = hpelm_c

            # Predict
            y_pred_c = hpelm_c.predict(X_test_clean_c).ravel()

            # Reconstruct predictions for this cluster
            full_stacked_c = np.full(len(y_test_stacked_c), np.nan)
            full_stacked_c[~test_nan_mask_c] = y_pred_c
            pred_c_reshaped = full_stacked_c.reshape(n_time, n_lat, n_lon)

            # Fill in the predictions array
            predictions = np.where(np.isnan(predictions), pred_c_reshaped, predictions)

        predicted_da = xr.DataArray(
            data=predictions,
            coords={'T': time, 'Y': lat, 'X': lon},
            dims=['T', 'Y', 'X']
        )
        return predicted_da

    # ------------------ Probability Calculation Methods ------------------
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
            pred_prob[0, :] = t.cdf(first_t, df=dof)
            pred_prob[1, :] = t.cdf(second_t, df=dof) - t.cdf(first_t, df=dof)
            pred_prob[2, :] = 1 - t.cdf(second_t, df=dof)
        return pred_prob

    @staticmethod
    def calculate_tercile_probabilities_weibull_min(best_guess, error_variance, first_tercile, second_tercile, dof):
        """
        Weibull minimum-based method.
        """
        n_time = len(best_guess)
        pred_prob = np.empty((3, n_time))
        if np.all(np.isnan(best_guess)):
            pred_prob[:] = np.nan
        else:
            error_std = np.sqrt(error_variance)
            pred_prob[0, :] = weibull_min.cdf(first_tercile, c=dof, loc=best_guess, scale=error_std)
            pred_prob[1, :] = weibull_min.cdf(second_tercile, c=dof, loc=best_guess, scale=error_std) - \
                              weibull_min.cdf(first_tercile, c=dof, loc=best_guess, scale=error_std)
            pred_prob[2, :] = 1 - weibull_min.cdf(second_tercile, c=dof, loc=best_guess, scale=error_std)
        return pred_prob

    @staticmethod
    def calculate_tercile_probabilities_gamma(best_guess, error_variance, T1, T2, dof=None):
        """Gamma-distribution based method."""
        n_time = len(best_guess)
        pred_prob = np.empty((3, n_time), dtype=float)
        if np.any(np.isnan(best_guess)) or np.any(np.isnan(error_variance)):
            pred_prob[:] = np.nan
            return pred_prob
        best_guess = np.asarray(best_guess, dtype=float)
        error_variance = np.asarray(error_variance, dtype=float)
        T1 = np.asarray(T1, dtype=float)
        T2 = np.asarray(T2, dtype=float)
        alpha = (best_guess ** 2) / error_variance
        theta = error_variance / best_guess
        cdf_t1 = gamma.cdf(T1, a=alpha, scale=theta)
        cdf_t2 = gamma.cdf(T2, a=alpha, scale=theta)
        pred_prob[0, :] = cdf_t1
        pred_prob[1, :] = cdf_t2 - cdf_t1
        pred_prob[2, :] = 1.0 - cdf_t2
        return pred_prob

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

    @staticmethod
    def calculate_tercile_probabilities_normal(best_guess, error_variance, first_tercile, second_tercile):
        """Normal-distribution based method."""
        n_time = len(best_guess)
        pred_prob = np.empty((3, n_time))
        if np.all(np.isnan(best_guess)):
            pred_prob[:] = np.nan
        else:
            error_std = np.sqrt(error_variance)
            pred_prob[0, :] = norm.cdf(first_tercile, loc=best_guess, scale=error_std)
            pred_prob[1, :] = norm.cdf(second_tercile, loc=best_guess, scale=error_std) - norm.cdf(first_tercile, loc=best_guess, scale=error_std)
            pred_prob[2, :] = 1 - norm.cdf(second_tercile, loc=best_guess, scale=error_std)
        return pred_prob

    @staticmethod
    def calculate_tercile_probabilities_lognormal(best_guess, error_variance, first_tercile, second_tercile):
        """Lognormal-distribution based method."""
        n_time = len(best_guess)
        pred_prob = np.empty((3, n_time))
        if np.any(np.isnan(best_guess)) or np.any(np.isnan(error_variance)):
            pred_prob[:] = np.nan
            return pred_prob
        sigma = np.sqrt(np.log(1 + error_variance / (best_guess ** 2)))
        mu = np.log(best_guess) - sigma ** 2 / 2
        pred_prob[0, :] = lognorm.cdf(first_tercile, s=sigma, scale=np.exp(mu))
        pred_prob[1, :] = lognorm.cdf(second_tercile, s=sigma, scale=np.exp(mu)) - lognorm.cdf(first_tercile, s=sigma, scale=np.exp(mu))
        pred_prob[2, :] = 1 - lognorm.cdf(second_tercile, s=sigma, scale=np.exp(mu))
        return pred_prob

    def compute_prob(self, Predictant, clim_year_start, clim_year_end, hindcast_det):
        """
        Compute tercile probabilities using the chosen distribution method.
        Predictant is expected to be an xarray DataArray with dims (T, Y, X).
        """
        if "M" in Predictant.coords:
            Predictant = Predictant.isel(M=0).drop_vars('M').squeeze()
        mask = xr.where(~np.isnan(Predictant.isel(T=0)), 1, np.nan).drop_vars(['T']).squeeze().to_numpy()
        Predictant = Predictant.transpose('T', 'Y', 'X')
        index_start = Predictant.get_index("T").get_loc(str(clim_year_start)).start
        index_end = Predictant.get_index("T").get_loc(str(clim_year_end)).stop
        rainfall_for_tercile = Predictant.isel(T=slice(index_start, index_end))
        terciles = rainfall_for_tercile.quantile([0.33, 0.67], dim='T')
        error_variance = (Predictant - hindcast_det).var(dim='T')
        dof = len(Predictant.get_index("T")) - 2
        if self.dist_method == "t":
            calc_func = self.calculate_tercile_probabilities
            hindcast_prob = xr.apply_ufunc(
                calc_func,
                hindcast_det,
                error_variance,
                terciles.isel(quantile=0).drop_vars('quantile'),
                terciles.isel(quantile=1).drop_vars('quantile'),
                input_core_dims=[('T',), (), (), ()],
                vectorize=True,
                kwargs={'dof': dof},
                dask='parallelized',
                output_core_dims=[('probability', 'T')],
                output_dtypes=['float'],
                dask_gufunc_kwargs={'output_sizes': {'probability': 3}, 'allow_rechunk': True}
            )
        elif self.dist_method == "weibull_min":
            calc_func = self.calculate_tercile_probabilities_weibull_min
            hindcast_prob = xr.apply_ufunc(
                calc_func,
                hindcast_det,
                error_variance,
                terciles.isel(quantile=0).drop_vars('quantile'),
                terciles.isel(quantile=1).drop_vars('quantile'),
                input_core_dims=[('T',), (), (), ()],
                vectorize=True,
                kwargs={'dof': dof},
                dask='parallelized',
                output_core_dims=[('probability', 'T')],
                output_dtypes=['float'],
                dask_gufunc_kwargs={'output_sizes': {'probability': 3}, 'allow_rechunk': True}
            )
        elif self.dist_method == "gamma":
            calc_func = self.calculate_tercile_probabilities_gamma
            hindcast_prob = xr.apply_ufunc(
                calc_func,
                hindcast_det,
                error_variance,
                terciles.isel(quantile=0).drop_vars('quantile'),
                terciles.isel(quantile=1).drop_vars('quantile'),
                input_core_dims=[('T',), (), (), ()],
                vectorize=True,
                dask='parallelized',
                output_core_dims=[('probability', 'T')],
                output_dtypes=['float'],
                dask_gufunc_kwargs={'output_sizes': {'probability': 3}, 'allow_rechunk': True}
            )
        elif self.dist_method == "nonparam":
            calc_func = self.calculate_tercile_probabilities_nonparametric
            error_samples = Predictant - hindcast_det
            hindcast_prob = xr.apply_ufunc(
                calc_func,
                hindcast_det,
                error_samples,
                terciles.isel(quantile=0).drop_vars('quantile'),
                terciles.isel(quantile=1).drop_vars('quantile'),
                input_core_dims=[('T',), ('T',), (), ()],
                output_core_dims=[('probability', 'T')],
                vectorize=True,
                dask='parallelized',
                output_dtypes=[float],
                dask_gufunc_kwargs={'output_sizes': {'probability': 3}, 'allow_rechunk': True}
            )
        elif self.dist_method == "normal":
            calc_func = self.calculate_tercile_probabilities_normal
            hindcast_prob = xr.apply_ufunc(
                calc_func,
                hindcast_det,
                error_variance,
                terciles.isel(quantile=0).drop_vars('quantile'),
                terciles.isel(quantile=1).drop_vars('quantile'),
                input_core_dims=[('T',), (), (), ()],
                vectorize=True,
                dask='parallelized',
                output_core_dims=[('probability', 'T')],
                output_dtypes=['float'],
                dask_gufunc_kwargs={'output_sizes': {'probability': 3}, 'allow_rechunk': True}
            )
        elif self.dist_method == "lognormal":
            calc_func = self.calculate_tercile_probabilities_lognormal
            hindcast_prob = xr.apply_ufunc(
                calc_func,
                hindcast_det,
                error_variance,
                terciles.isel(quantile=0).drop_vars('quantile'),
                terciles.isel(quantile=1).drop_vars('quantile'),
                input_core_dims=[('T',), (), (), ()],
                vectorize=True,
                dask='parallelized',
                output_core_dims=[('probability', 'T')],
                output_dtypes=['float'],
                dask_gufunc_kwargs={'output_sizes': {'probability': 3}, 'allow_rechunk': True}
            )
        else:
            raise ValueError(f"Invalid dist_method: {self.dist_method}.")
        hindcast_prob = hindcast_prob.assign_coords(probability=('probability', ['PB', 'PN', 'PA']))
        return hindcast_prob.transpose('probability', 'T', 'Y', 'X')

    @staticmethod
    def _reshape_and_filter_data(da):
        """
        Helper: stack the DataArray from (T, Y, X[, M]) to (n_samples, n_features)
        and remove rows containing NaNs.
        """
        da_stacked = da.stack(sample=('T', 'Y', 'X'))
        if 'M' in da.dims:
            da_stacked = da_stacked.transpose('sample', 'M')
        else:
            da_stacked = da_stacked.transpose('sample')
        da_values = da_stacked.values
        nan_mask = np.any(np.isnan(da_values), axis=1)
        return da_values[~nan_mask], nan_mask, da_values

    def forecast(self, Predictant, clim_year_start, clim_year_end, hindcast_det, hindcast_det_cross, Predictor_for_year, best_params=None, cluster_da=None):
        """
        Forecast method using a single HPELM model with optimized hyperparameters.
        Parameters
        ----------
        Predictant : xarray.DataArray
            Observed predictand data with dimensions (T, Y, X) or (T, M, Y, X).
        clim_year_start : int or str
            Start year of the climatology period.
        clim_year_end : int or str
            End year of the climatology period.
        hindcast_det : xarray.DataArray
            Deterministic hindcast data for training with dimensions (T, M, Y, X).
        hindcast_det_cross : xarray.DataArray
            Deterministic hindcast data for error estimation with dimensions (T, Y, X).
        Predictor_for_year : xarray.DataArray
            Predictor data for the target year with dimensions (T, M, Y, X).
        best_params : dict, optional
            Pre-computed best hyperparameters. If None, computes internally.
        cluster_da : xarray.DataArray, optional
            Pre-computed cluster labels. If None, computes internally.
        Returns
        -------
        forecast_det : xarray.DataArray
            Deterministic forecast with dimensions (T, Y, X).
        forecast_prob : xarray.DataArray
            Tercile probabilities with dimensions (probability, T, Y, X).
        """
        if "M" in Predictant.coords:
            Predictant_no_m = Predictant.isel(M=0).drop_vars('M').squeeze()
        else:
            Predictant_no_m = Predictant
        mask = xr.where(~np.isnan(Predictant_no_m.isel(T=0)), 1, np.nan).drop_vars(['T']).squeeze().to_numpy()
        # Standardize Predictor_for_year using hindcast climatology
        mean_val = hindcast_det.sel(T=slice(str(clim_year_start), str(clim_year_end))).mean(dim='T')
        std_val = hindcast_det.sel(T=slice(str(clim_year_start), str(clim_year_end))).std(dim='T')
        Predictor_for_year_st = (Predictor_for_year - mean_val) / std_val
        hindcast_det_st = standardize_timeseries(hindcast_det, clim_year_start, clim_year_end)
        Predictant_st = standardize_timeseries(Predictant_no_m, clim_year_start, clim_year_end)
        hindcast_det_st['T'] = Predictant_st['T']
        
        # Extract coordinates from X_test
        time = Predictor_for_year_st['T']
        lat = Predictor_for_year_st['Y']
        lon = Predictor_for_year_st['X']
        n_time = len(time)
        n_lat = len(lat)
        n_lon = len(lon)
        # Use provided best_params and cluster_da or compute if None
        if best_params is None:
            best_params, cluster_da = self.compute_hyperparameters(hindcast_det, Predictant_no_m, clim_year_start, clim_year_end)
        # Initialize predictions array
        predictions = np.full((n_time, n_lat, n_lon), np.nan)
        self.hpelm = {}  # Dictionary to store models per cluster
        for c in range(self.n_clusters):
            if c not in best_params:
                continue
            bp = best_params[c]
            # Mask for this cluster
            mask_3d_train = (cluster_da == c).expand_dims({'T': hindcast_det_st['T']})
            mask_3d_test = (cluster_da == c).expand_dims({'T': Predictor_for_year_st['T']})
            # Stack training data for cluster
            X_train_stacked_c = hindcast_det_st.where(mask_3d_train).stack(sample=('T', 'Y', 'X')).transpose('sample', 'M').values
            y_train_stacked_c = Predictant_st.where(mask_3d_train).stack(sample=('T', 'Y', 'X')).values.ravel()
            train_nan_mask_c = np.any(~np.isfinite(X_train_stacked_c), axis=1) | ~np.isfinite(y_train_stacked_c)
            X_train_clean_c = X_train_stacked_c[~train_nan_mask_c]
            y_train_clean_c = y_train_stacked_c[~train_nan_mask_c]
            # Stack testing data for cluster
            X_test_stacked_c = Predictor_for_year_st.where(mask_3d_test).stack(sample=('T', 'Y', 'X')).transpose('sample', 'M').values
            test_nan_mask_c = np.any(~np.isfinite(X_test_stacked_c), axis=1)
            X_test_clean_c = X_test_stacked_c[~test_nan_mask_c]
            # Initialize and train the HPELM model for this cluster
            hpelm_c = HPELM(
                inputs=X_train_clean_c.shape[1],
                outputs=1,
                classification='r',
                norm=bp['norm']
            )

            # Initialize weights and biases for the neurons
            # Use a random number generator for reproducibility
            # rng = np.random.default_rng(1234)    # isolated RNG
            # n = bp['neurons']
            # W = rng.standard_normal((X_train_clean_c.shape[1], n))
            # B = rng.standard_normal(n)
            # hpelm_c.add_neurons(bp['neurons'], bp['activation'],W=W, B=B)
    
            hpelm_c.add_neurons(bp['neurons'], bp['activation'])
            hpelm_c.train(X_train_clean_c, y_train_clean_c, 'r')
            self.hpelm[c] = hpelm_c
            # Predict
            y_pred_c = hpelm_c.predict(X_test_clean_c).ravel()
            # Reconstruct predictions for this cluster
            full_stacked_c = np.full(X_test_stacked_c.shape[0], np.nan)
            full_stacked_c[~test_nan_mask_c] = y_pred_c
            pred_c_reshaped = full_stacked_c.reshape(n_time, n_lat, n_lon)
            # Fill in the predictions array
            predictions = np.where(np.isnan(predictions), pred_c_reshaped, predictions)
        result_da = xr.DataArray(
            data=predictions,
            coords={'T': time, 'Y': lat, 'X': lon},
            dims=['T', 'Y', 'X']
        ) * mask
        result_da = reverse_standardize(result_da, Predictant_no_m, clim_year_start, clim_year_end)
        year = Predictor_for_year.coords['T'].values.astype('datetime64[Y]').astype(int)[0] + 1970
        T_value_1 = Predictant_no_m.isel(T=0).coords['T'].values
        month_1 = T_value_1.astype('datetime64[M]').astype(int) % 12 + 1
        new_T_value = np.datetime64(f"{year}-{month_1:02d}-01")
        result_da = result_da.assign_coords(T=xr.DataArray([new_T_value], dims=["T"]))
        result_da['T'] = result_da['T'].astype('datetime64[ns]')
        # Compute tercile probabilities
        index_start = Predictant_no_m.get_index("T").get_loc(str(clim_year_start)).start
        index_end = Predictant_no_m.get_index("T").get_loc(str(clim_year_end)).stop
        rainfall_for_tercile = Predictant_no_m.isel(T=slice(index_start, index_end))
        terciles = rainfall_for_tercile.quantile([0.33, 0.67], dim='T')
        error_variance = (Predictant_no_m - hindcast_det_cross).var(dim='T')
        dof = len(Predictant_no_m.get_index("T")) - 2
        if self.dist_method == "t":
            calc_func = self.calculate_tercile_probabilities
            hindcast_prob = xr.apply_ufunc(
                calc_func,
                result_da,
                error_variance,
                terciles.isel(quantile=0).drop_vars('quantile'),
                terciles.isel(quantile=1).drop_vars('quantile'),
                input_core_dims=[('T',), (), (), ()],
                vectorize=True,
                kwargs={'dof': dof},
                dask='parallelized',
                output_core_dims=[('probability', 'T')],
                output_dtypes=['float'],
                dask_gufunc_kwargs={'output_sizes': {'probability': 3}, 'allow_rechunk': True}
            )
        elif self.dist_method == "weibull_min":
            calc_func = self.calculate_tercile_probabilities_weibull_min
            hindcast_prob = xr.apply_ufunc(
                calc_func,
                result_da,
                error_variance,
                terciles.isel(quantile=0).drop_vars('quantile'),
                terciles.isel(quantile=1).drop_vars('quantile'),
                input_core_dims=[('T',), (), (), ()],
                vectorize=True,
                kwargs={'dof': dof},
                dask='parallelized',
                output_core_dims=[('probability', 'T')],
                output_dtypes=['float'],
                dask_gufunc_kwargs={'output_sizes': {'probability': 3}, 'allow_rechunk': True}
            )
        elif self.dist_method == "gamma":
            calc_func = self.calculate_tercile_probabilities_gamma
            hindcast_prob = xr.apply_ufunc(
                calc_func,
                result_da,
                error_variance,
                terciles.isel(quantile=0).drop_vars('quantile'),
                terciles.isel(quantile=1).drop_vars('quantile'),
                input_core_dims=[('T',), (), (), ()],
                vectorize=True,
                dask='parallelized',
                output_core_dims=[('probability', 'T')],
                output_dtypes=['float'],
                dask_gufunc_kwargs={'output_sizes': {'probability': 3}, 'allow_rechunk': True}
            )
        elif self.dist_method == "normal":
            calc_func = self.calculate_tercile_probabilities_normal
            hindcast_prob = xr.apply_ufunc(
                calc_func,
                result_da,
                error_variance,
                terciles.isel(quantile=0).drop_vars('quantile'),
                terciles.isel(quantile=1).drop_vars('quantile'),
                input_core_dims=[('T',), (), (), ()],
                vectorize=True,
                dask='parallelized',
                output_core_dims=[('probability', 'T')],
                output_dtypes=['float'],
                dask_gufunc_kwargs={'output_sizes': {'probability': 3}, 'allow_rechunk': True}
            )
        elif self.dist_method == "lognormal":
            calc_func = self.calculate_tercile_probabilities_lognormal
            hindcast_prob = xr.apply_ufunc(
                calc_func,
                result_da,
                error_variance,
                terciles.isel(quantile=0).drop_vars('quantile'),
                terciles.isel(quantile=1).drop_vars('quantile'),
                input_core_dims=[('T',), (), (), ()],
                vectorize=True,
                dask='parallelized',
                output_core_dims=[('probability', 'T')],
                output_dtypes=['float'],
                dask_gufunc_kwargs={'output_sizes': {'probability': 3}, 'allow_rechunk': True}
            )
        elif self.dist_method == "nonparam":
            calc_func = self.calculate_tercile_probabilities_nonparametric
            error_samples = Predictant_no_m - hindcast_det_cross
            error_samples = error_samples.rename({'T': 'S'})
            hindcast_prob = xr.apply_ufunc(
                calc_func,
                result_da,
                error_samples,
                terciles.isel(quantile=0).drop_vars('quantile'),
                terciles.isel(quantile=1).drop_vars('quantile'),
                input_core_dims=[('T',), ('S',), (), ()],
                output_core_dims=[('probability', 'T')],
                vectorize=True,
                dask='parallelized',
                output_dtypes=[float],
                dask_gufunc_kwargs={'output_sizes': {'probability': 3}, 'allow_rechunk': True}
            )
        else:
            raise ValueError(f"Invalid dist_method: {self.dist_method}.")
        hindcast_prob = hindcast_prob.assign_coords(probability=('probability', ['PB', 'PN', 'PA']))
        return result_da * mask, mask * hindcast_prob.transpose('probability', 'T', 'Y', 'X')

class WAS_mme_hpELM_:
    """
    Extreme Learning Machine (ELM) based Multi-Model Ensemble (MME) forecasting using hpelm library.
    This class implements a single-model forecasting approach using HPELM for deterministic predictions,
    with optional tercile probability calculations using various statistical distributions.
    Implements hyperparameter optimization via randomized search.

    Parameters
    ----------
    neurons_range : list of int, optional
        List of neuron counts to tune for HPELM (default is [10, 20, 50, 100]).
    activation_options : list of str, optional
        Activation functions to tune for HPELM (default is ['sigm', 'tanh', 'relu', 'rbf_linf', 'rbf_gauss']).
    norm_range : list of float, optional
        Regularization parameters to tune for HPELM (default is [0.1, 1.0, 10.0, 100.0]).
    random_state : int, optional
        Seed for reproducibility (default is 42).
    dist_method : str, optional
        Distribution method for tercile probabilities ('t', 'gamma', etc.) (default is 'gamma').
    n_iter_search : int, optional
        Number of iterations for randomized search (default is 10).
    cv_folds : int, optional
        Number of cross-validation folds (default is 3).
    """
    def __init__(self,
                 neurons_range=[10, 20, 50, 100],
                 activation_options=['sigm', 'tanh', 'relu', 'rbf_linf', 'rbf_gauss'],
                 norm_range=[0.1, 1.0, 10.0, 100.0],
                 random_state=42,
                 dist_method="gamma",
                 n_iter_search=10,
                 cv_folds=3):
        self.neurons_range = neurons_range
        self.activation_options = activation_options
        self.norm_range = norm_range
        self.random_state = random_state
        self.dist_method = dist_method
        self.n_iter_search = n_iter_search
        self.cv_folds = cv_folds
        self.hpelm = None

    def compute_hyperparameters(self, Predictors, Predictand, clim_year_start, clim_year_end):
        """
        Independently computes the best hyperparameters using randomized search on stacked training data.

        Parameters
        ----------
        Predictors : xarray.DataArray
            Training predictor data with dimensions (T, M, Y, X).
        Predictand : xarray.DataArray
            Training predictand data with dimensions (T, Y, X).
        clim_year_start : int
            Start year of the climatology period.
        clim_year_end : int
            End year of the climatology period.

        Returns
        -------
        dict
            Best hyperparameters found.
        """
        X_train = standardize_timeseries(Predictors, clim_year_start, clim_year_end)
        y_train = standardize_timeseries(Predictand, clim_year_start, clim_year_end)

        # Stack training data
        X_train_stacked = X_train.stack(sample=('T', 'Y', 'X')).transpose('sample', 'M').values
        y_train_stacked = y_train.stack(sample=('T', 'Y', 'X')).values.ravel()
        train_nan_mask = np.any(~np.isfinite(X_train_stacked), axis=1) | ~np.isfinite(y_train_stacked)
        X_train_clean = X_train_stacked[~train_nan_mask]
        y_train_clean = y_train_stacked[~train_nan_mask]

        # Prepare parameter distributions
        param_dist = {
            'neurons': self.neurons_range,
            'activation': self.activation_options,
            'norm': self.norm_range
        }

        # Initialize HPELM wrapper for scikit-learn compatibility
        model = HPELMWrapper(random_state=self.random_state)

        # Randomized search
        random_search = RandomizedSearchCV(
            model, param_distributions=param_dist, n_iter=self.n_iter_search,
            cv=self.cv_folds, scoring='neg_mean_squared_error',
            random_state=self.random_state, error_score=np.nan
        )

        random_search.fit(X_train_clean, y_train_clean)
        best_params = random_search.best_params_
        return best_params

    def compute_model(self, X_train, y_train, X_test, y_test, best_params=None):
        """
        Compute deterministic hindcast using the HPELM model with injected hyperparameters.

        Parameters
        ----------
        X_train : xarray.DataArray
            Training predictor data with dimensions (T, M, Y, X).
        y_train : xarray.DataArray
            Training predictand data with dimensions (T, Y, X).
        X_test : xarray.DataArray
            Testing predictor data with dimensions (T, M, Y, X).
        y_test : xarray.DataArray
            Testing predictand data with dimensions (T, Y, X).
        best_params : dict, optional
            Pre-computed best hyperparameters. If None, computes internally.

        Returns
        -------
        predicted_da : xarray.DataArray
            Deterministic hindcast with dimensions (T, Y, X).
        """
        # Extract coordinate variables from X_test
        time = X_test['T']
        lat = X_test['Y']
        lon = X_test['X']
        n_time = len(X_test.coords['T'])
        n_lat = len(X_test.coords['Y'])
        n_lon = len(X_test.coords['X'])

        # Stack training data
        X_train_stacked = X_train.stack(sample=('T', 'Y', 'X')).transpose('sample', 'M').values
        y_train_stacked = y_train.stack(sample=('T', 'Y', 'X')).transpose('sample', 'M').values
        train_nan_mask = np.any(~np.isfinite(X_train_stacked), axis=1) | np.any(~np.isfinite(y_train_stacked), axis=1)
        X_train_clean = X_train_stacked[~train_nan_mask]
        y_train_clean = y_train_stacked[~train_nan_mask]

        # Stack testing data
        X_test_stacked = X_test.stack(sample=('T', 'Y', 'X')).transpose('sample', 'M').values
        y_test_stacked = y_test.stack(sample=('T', 'Y', 'X')).transpose('sample', 'M').values
        test_nan_mask = np.any(~np.isfinite(X_test_stacked), axis=1) | np.any(~np.isfinite(y_test_stacked), axis=1)

        # Use provided best_params or compute if None
        if best_params is None:
            best_params = self.compute_hyperparameters(X_train, y_train, clim_year_start, clim_year_end)

        # Initialize the HPELM model with best parameters
        self.hpelm = HPELM(
            inputs=X_train_clean.shape[1],
            outputs=1,
            classification='r',
            norm=best_params['norm']
        )
        self.hpelm.add_neurons(best_params['neurons'], best_params['activation'])
        self.hpelm.train(X_train_clean, y_train_clean, 'r')
        y_pred = self.hpelm.predict(X_test_stacked[~test_nan_mask]).ravel()

        # Reconstruct predictions
        result = np.empty_like(np.squeeze(y_test_stacked))
        result[np.squeeze(test_nan_mask)] = np.squeeze(y_test_stacked[test_nan_mask])
        result[~np.squeeze(test_nan_mask)] = y_pred

        predictions_reshaped = result.reshape(n_time, n_lat, n_lon)
        predicted_da = xr.DataArray(
            data=predictions_reshaped,
            coords={'T': time, 'Y': lat, 'X': lon},
            dims=['T', 'Y', 'X']
        )
        return predicted_da

    # ------------------ Probability Calculation Methods ------------------
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
    def calculate_tercile_probabilities_weibull_min(best_guess, error_variance, first_tercile, second_tercile, dof):
        """
        Weibull minimum-based method.
        """
        n_time = len(best_guess)
        pred_prob = np.empty((3, n_time))
        if np.all(np.isnan(best_guess)):
            pred_prob[:] = np.nan
        else:
            error_std = np.sqrt(error_variance)
            pred_prob[0, :] = stats.weibull_min.cdf(first_tercile, c=dof, loc=best_guess, scale=error_std)
            pred_prob[1, :] = stats.weibull_min.cdf(second_tercile, c=dof, loc=best_guess, scale=error_std) - \
                              stats.weibull_min.cdf(first_tercile, c=dof, loc=best_guess, scale=error_std)
            pred_prob[2, :] = 1 - stats.weibull_min.cdf(second_tercile, c=dof, loc=best_guess, scale=error_std)
        return pred_prob

    @staticmethod
    def calculate_tercile_probabilities_gamma(best_guess, error_variance, T1, T2, dof=None):
        """Gamma-distribution based method."""
        n_time = len(best_guess)
        pred_prob = np.empty((3, n_time), dtype=float)
        if np.any(np.isnan(best_guess)) or np.any(np.isnan(error_variance)):
            pred_prob[:] = np.nan
            return pred_prob
        best_guess = np.asarray(best_guess, dtype=float)
        error_variance = np.asarray(error_variance, dtype=float)
        T1 = np.asarray(T1, dtype=float)
        T2 = np.asarray(T2, dtype=float)
        alpha = (best_guess ** 2) / error_variance
        theta = error_variance / best_guess
        cdf_t1 = gamma.cdf(T1, a=alpha, scale=theta)
        cdf_t2 = gamma.cdf(T2, a=alpha, scale=theta)
        pred_prob[0, :] = cdf_t1
        pred_prob[1, :] = cdf_t2 - cdf_t1
        pred_prob[2, :] = 1.0 - cdf_t2
        return pred_prob

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

    @staticmethod
    def calculate_tercile_probabilities_normal(best_guess, error_variance, first_tercile, second_tercile):
        """Normal-distribution based method."""
        n_time = len(best_guess)
        pred_prob = np.empty((3, n_time))
        if np.all(np.isnan(best_guess)):
            pred_prob[:] = np.nan
        else:
            error_std = np.sqrt(error_variance)
            pred_prob[0, :] = norm.cdf(first_tercile, loc=best_guess, scale=error_std)
            pred_prob[1, :] = norm.cdf(second_tercile, loc=best_guess, scale=error_std) - norm.cdf(first_tercile, loc=best_guess, scale=error_std)
            pred_prob[2, :] = 1 - norm.cdf(second_tercile, loc=best_guess, scale=error_std)
        return pred_prob

    @staticmethod
    def calculate_tercile_probabilities_lognormal(best_guess, error_variance, first_tercile, second_tercile):
        """Lognormal-distribution based method."""
        n_time = len(best_guess)
        pred_prob = np.empty((3, n_time))
        if np.any(np.isnan(best_guess)) or np.any(np.isnan(error_variance)):
            pred_prob[:] = np.nan
            return pred_prob
        sigma = np.sqrt(np.log(1 + error_variance / (best_guess ** 2)))
        mu = np.log(best_guess) - sigma ** 2 / 2
        pred_prob[0, :] = lognorm.cdf(first_tercile, s=sigma, scale=np.exp(mu))
        pred_prob[1, :] = lognorm.cdf(second_tercile, s=sigma, scale=np.exp(mu)) - lognorm.cdf(first_tercile, s=sigma, scale=np.exp(mu))
        pred_prob[2, :] = 1 - lognorm.cdf(second_tercile, s=sigma, scale=np.exp(mu))
        return pred_prob


    def compute_prob(self, Predictant, clim_year_start, clim_year_end, hindcast_det):
        """
        Compute tercile probabilities using the chosen distribution method.
        Predictant is expected to be an xarray DataArray with dims (T, Y, X).
        """
        if "M" in Predictant.coords:
            Predictant = Predictant.isel(M=0).drop_vars('M').squeeze()
        mask = xr.where(~np.isnan(Predictant.isel(T=0)), 1, np.nan).drop_vars(['T']).squeeze().to_numpy()

        Predictant = Predictant.transpose('T', 'Y', 'X')
        index_start = Predictant.get_index("T").get_loc(str(clim_year_start)).start
        index_end = Predictant.get_index("T").get_loc(str(clim_year_end)).stop
        rainfall_for_tercile = Predictant.isel(T=slice(index_start, index_end))
        terciles = rainfall_for_tercile.quantile([0.33, 0.67], dim='T')
        error_variance = (Predictant - hindcast_det).var(dim='T')
        dof = len(Predictant.get_index("T")) - 2

        if self.dist_method == "t":
            calc_func = self.calculate_tercile_probabilities
            hindcast_prob = xr.apply_ufunc(
                calc_func,
                hindcast_det,
                error_variance,
                terciles.isel(quantile=0).drop_vars('quantile'),
                terciles.isel(quantile=1).drop_vars('quantile'),
                input_core_dims=[('T',), (), (), ()],
                vectorize=True,
                kwargs={'dof': dof},
                dask='parallelized',
                output_core_dims=[('probability', 'T')],
                output_dtypes=['float'],
                dask_gufunc_kwargs={'output_sizes': {'probability': 3}, 'allow_rechunk': True}
            )
        elif self.dist_method == "weibull_min":
            calc_func = self.calculate_tercile_probabilities_weibull_min
            hindcast_prob = xr.apply_ufunc(
                calc_func,
                hindcast_det,
                error_variance,
                terciles.isel(quantile=0).drop_vars('quantile'),
                terciles.isel(quantile=1).drop_vars('quantile'),
                input_core_dims=[('T',), (), (), ()],
                vectorize=True,
                kwargs={'dof': dof},
                dask='parallelized',
                output_core_dims=[('probability', 'T')],
                output_dtypes=['float'],
                dask_gufunc_kwargs={'output_sizes': {'probability': 3}, 'allow_rechunk': True}
            )
        elif self.dist_method == "gamma":
            calc_func = self.calculate_tercile_probabilities_gamma
            hindcast_prob = xr.apply_ufunc(
                calc_func,
                hindcast_det,
                error_variance,
                terciles.isel(quantile=0).drop_vars('quantile'),
                terciles.isel(quantile=1).drop_vars('quantile'),
                input_core_dims=[('T',), (), (), ()],
                vectorize=True,
                dask='parallelized',
                output_core_dims=[('probability', 'T')],
                output_dtypes=['float'],
                dask_gufunc_kwargs={'output_sizes': {'probability': 3}, 'allow_rechunk': True}
            )
        elif self.dist_method == "nonparam":
            calc_func = self.calculate_tercile_probabilities_nonparametric
            error_samples = Predictant - hindcast_det
            hindcast_prob = xr.apply_ufunc(
                calc_func,
                hindcast_det,
                error_samples,
                terciles.isel(quantile=0).drop_vars('quantile'),
                terciles.isel(quantile=1).drop_vars('quantile'),
                input_core_dims=[('T',), ('T',), (), ()],
                output_core_dims=[('probability', 'T')],
                vectorize=True,
                dask='parallelized',
                output_dtypes=[float],
                dask_gufunc_kwargs={'output_sizes': {'probability': 3}, 'allow_rechunk': True}
            )
        elif self.dist_method == "normal":
            calc_func = self.calculate_tercile_probabilities_normal
            hindcast_prob = xr.apply_ufunc(
                calc_func,
                hindcast_det,
                error_variance,
                terciles.isel(quantile=0).drop_vars('quantile'),
                terciles.isel(quantile=1).drop_vars('quantile'),
                input_core_dims=[('T',), (), (), ()],
                vectorize=True,
                dask='parallelized',
                output_core_dims=[('probability', 'T')],
                output_dtypes=['float'],
                dask_gufunc_kwargs={'output_sizes': {'probability': 3}, 'allow_rechunk': True}
            )
        elif self.dist_method == "lognormal":
            calc_func = self.calculate_tercile_probabilities_lognormal
            hindcast_prob = xr.apply_ufunc(
                calc_func,
                hindcast_det,
                error_variance,
                terciles.isel(quantile=0).drop_vars('quantile'),
                terciles.isel(quantile=1).drop_vars('quantile'),
                input_core_dims=[('T',), (), (), ()],
                vectorize=True,
                dask='parallelized',
                output_core_dims=[('probability', 'T')],
                output_dtypes=['float'],
                dask_gufunc_kwargs={'output_sizes': {'probability': 3}, 'allow_rechunk': True}
            )
        else:
            raise ValueError(f"Invalid dist_method: {self.dist_method}.")

        hindcast_prob = hindcast_prob.assign_coords(probability=('probability', ['PB', 'PN', 'PA']))
        return hindcast_prob.transpose('probability', 'T', 'Y', 'X')

    
    @staticmethod
    def _reshape_and_filter_data(da):
        """
        Helper: stack the DataArray from (T, Y, X[, M]) to (n_samples, n_features)
        and remove rows containing NaNs.
        """
        da_stacked = da.stack(sample=('T', 'Y', 'X'))
        if 'M' in da.dims:
            da_stacked = da_stacked.transpose('sample', 'M')
        else:
            da_stacked = da_stacked.transpose('sample')
        da_values = da_stacked.values
        nan_mask = np.any(np.isnan(da_values), axis=1)
        return da_values[~nan_mask], nan_mask, da_values




    def forecast(self, Predictant, clim_year_start, clim_year_end, hindcast_det, hindcast_det_cross, Predictor_for_year, best_params=None):
        """
        Forecast method using a single HPELM model with optimized hyperparameters.

        Parameters
        ----------
        Predictant : xarray.DataArray
            Observed predictand data with dimensions (T, Y, X) or (T, M, Y, X).
        clim_year_start : int or str
            Start year of the climatology period.
        clim_year_end : int or str
            End year of the climatology period.
        hindcast_det : xarray.DataArray
            Deterministic hindcast data for training with dimensions (T, M, Y, X).
        hindcast_det_cross : xarray.DataArray
            Deterministic hindcast data for error estimation with dimensions (T, Y, X).
        Predictor_for_year : xarray.DataArray
            Predictor data for the target year with dimensions (T, M, Y, X).
        best_params : dict, optional
            Pre-computed best hyperparameters. If None, computes internally.

        Returns
        -------
        forecast_det : xarray.DataArray
            Deterministic forecast with dimensions (T, Y, X).
        forecast_prob : xarray.DataArray
            Tercile probabilities with dimensions (probability, T, Y, X).
        """
        if "M" in Predictant.coords:
            Predictant_no_m = Predictant.isel(M=0).drop_vars('M').squeeze()
        else:
            Predictant_no_m = Predictant

        mask = xr.where(~np.isnan(Predictant_no_m.isel(T=0)), 1, np.nan).drop_vars(['T']).squeeze().to_numpy()

        # Standardize Predictor_for_year using hindcast climatology
        mean_val = hindcast_det.sel(T=slice(str(clim_year_start), str(clim_year_end))).mean(dim='T')
        std_val = hindcast_det.sel(T=slice(str(clim_year_start), str(clim_year_end))).std(dim='T')
        Predictor_for_year_st = (Predictor_for_year - mean_val) / std_val

        hindcast_det_st = standardize_timeseries(hindcast_det, clim_year_start, clim_year_end)
        Predictant_st = standardize_timeseries(Predictant_no_m, clim_year_start, clim_year_end)
        y_test = Predictant_st.isel(T=[-1])

        # Extract coordinates from X_test
        time = Predictor_for_year_st['T']
        lat = Predictor_for_year_st['Y']
        lon = Predictor_for_year_st['X']
        n_time = len(Predictor_for_year_st.coords['T'])
        n_lat = len(Predictor_for_year_st.coords['Y'])
        n_lon = len(Predictor_for_year_st.coords['X'])

        # Stack training data and remove rows with NaNs
        X_train_stacked = hindcast_det_st.stack(sample=('T', 'Y', 'X')).transpose('sample', 'M').values
        y_train_stacked = Predictant_st.stack(sample=('T', 'Y', 'X')).values.ravel()
        train_nan_mask = np.any(~np.isfinite(X_train_stacked), axis=1) | ~np.isfinite(y_train_stacked)
        X_train_clean = X_train_stacked[~train_nan_mask]
        y_train_clean = y_train_stacked[~train_nan_mask]

        # Stack testing data
        X_test_stacked = Predictor_for_year_st.stack(sample=('T', 'Y', 'X')).transpose('sample', 'M').values
        y_test_stacked = y_test.stack(sample=('T', 'Y', 'X')).values.ravel()
        test_nan_mask = np.any(~np.isfinite(X_test_stacked), axis=1) | ~np.isfinite(y_test_stacked)

        # Use provided best_params or compute if None
        if best_params is None:
            best_params = self.compute_hyperparameters(hindcast_det, Predictant_no_m, clim_year_start, clim_year_end)

        # Initialize and fit the HPELM model with best parameters
        self.hpelm = HPELM(
            inputs=X_train_clean.shape[1],
            outputs=1,
            classification='r',
            norm=best_params['norm']
        )
        self.hpelm.add_neurons(best_params['neurons'], best_params['activation'])
        self.hpelm.train(X_train_clean, y_train_clean, 'r')
        y_pred = self.hpelm.predict(X_test_stacked[~test_nan_mask]).ravel()

        # Reconstruct the prediction array
        result = np.empty_like(np.squeeze(y_test_stacked))
        result[np.squeeze(test_nan_mask)] = np.squeeze(y_test_stacked[test_nan_mask])
        result[~np.squeeze(test_nan_mask)] = y_pred

        predictions_reshaped = result.reshape(n_time, n_lat, n_lon)
        result_da = xr.DataArray(
            data=predictions_reshaped,
            coords={'T': time, 'Y': lat, 'X': lon},
            dims=['T', 'Y', 'X']
        ) * mask

        result_da = reverse_standardize(result_da, Predictant_no_m, clim_year_start, clim_year_end)
        year = Predictor_for_year.coords['T'].values.astype('datetime64[Y]').astype(int)[0] + 1970
        T_value_1 = Predictant_no_m.isel(T=0).coords['T'].values
        month_1 = T_value_1.astype('datetime64[M]').astype(int) % 12 + 1
        new_T_value = np.datetime64(f"{year}-{month_1:02d}-01")
        result_da = result_da.assign_coords(T=xr.DataArray([new_T_value], dims=["T"]))
        result_da['T'] = result_da['T'].astype('datetime64[ns]')

        # Compute tercile probabilities
        index_start = Predictant_no_m.get_index("T").get_loc(str(clim_year_start)).start
        index_end = Predictant_no_m.get_index("T").get_loc(str(clim_year_end)).stop
        rainfall_for_tercile = Predictant_no_m.isel(T=slice(index_start, index_end))
        terciles = rainfall_for_tercile.quantile([0.33, 0.67], dim='T')
        error_variance = (Predictant_no_m - hindcast_det_cross).var(dim='T')
        dof = len(Predictant_no_m.get_index("T")) - 2

        if self.dist_method == "t":
            calc_func = self.calculate_tercile_probabilities
            hindcast_prob = xr.apply_ufunc(
                calc_func,
                result_da,
                error_variance,
                terciles.isel(quantile=0).drop_vars('quantile'),
                terciles.isel(quantile=1).drop_vars('quantile'),
                input_core_dims=[('T',), (), (), ()],
                vectorize=True,
                kwargs={'dof': dof},
                dask='parallelized',
                output_core_dims=[('probability', 'T')],
                output_dtypes=['float'],
                dask_gufunc_kwargs={'output_sizes': {'probability': 3}, 'allow_rechunk': True}
            )
        elif self.dist_method == "weibull_min":
            calc_func = self.calculate_tercile_probabilities_weibull_min
            hindcast_prob = xr.apply_ufunc(
                calc_func,
                result_da,
                error_variance,
                terciles.isel(quantile=0).drop_vars('quantile'),
                terciles.isel(quantile=1).drop_vars('quantile'),
                input_core_dims=[('T',), (), (), ()],
                vectorize=True,
                kwargs={'dof': dof},
                dask='parallelized',
                output_core_dims=[('probability', 'T')],
                output_dtypes=['float'],
                dask_gufunc_kwargs={'output_sizes': {'probability': 3}, 'allow_rechunk': True}
            )
        elif self.dist_method == "gamma":
            calc_func = self.calculate_tercile_probabilities_gamma
            hindcast_prob = xr.apply_ufunc(
                calc_func,
                result_da,
                error_variance,
                terciles.isel(quantile=0).drop_vars('quantile'),
                terciles.isel(quantile=1).drop_vars('quantile'),
                input_core_dims=[('T',), (), (), ()],
                vectorize=True,
                dask='parallelized',
                output_core_dims=[('probability', 'T')],
                output_dtypes=['float'],
                dask_gufunc_kwargs={'output_sizes': {'probability': 3}, 'allow_rechunk': True}
            )
        elif self.dist_method == "normal":
            calc_func = self.calculate_tercile_probabilities_normal
            hindcast_prob = xr.apply_ufunc(
                calc_func,
                result_da,
                error_variance,
                terciles.isel(quantile=0).drop_vars('quantile'),
                terciles.isel(quantile=1).drop_vars('quantile'),
                input_core_dims=[('T',), (), (), ()],
                vectorize=True,
                dask='parallelized',
                output_core_dims=[('probability', 'T')],
                output_dtypes=['float'],
                dask_gufunc_kwargs={'output_sizes': {'probability': 3}, 'allow_rechunk': True}
            )
        elif self.dist_method == "lognormal":
            calc_func = self.calculate_tercile_probabilities_lognormal
            hindcast_prob = xr.apply_ufunc(
                calc_func,
                result_da,
                error_variance,
                terciles.isel(quantile=0).drop_vars('quantile'),
                terciles.isel(quantile=1).drop_vars('quantile'),
                input_core_dims=[('T',), (), (), ()],
                vectorize=True,
                dask='parallelized',
                output_core_dims=[('probability', 'T')],
                output_dtypes=['float'],
                dask_gufunc_kwargs={'output_sizes': {'probability': 3}, 'allow_rechunk': True}
            )
        elif self.dist_method == "nonparam":
            calc_func = self.calculate_tercile_probabilities_nonparametric
            error_samples = Predictant_no_m - hindcast_det_cross
            error_samples = error_samples.rename({'T': 'S'})
            hindcast_prob = xr.apply_ufunc(
                calc_func,
                result_da,
                error_samples,
                terciles.isel(quantile=0).drop_vars('quantile'),
                terciles.isel(quantile=1).drop_vars('quantile'),
                input_core_dims=[('T',), ('S',), (), ()],
                output_core_dims=[('probability', 'T')],
                vectorize=True,
                dask='parallelized',
                output_dtypes=[float],
                dask_gufunc_kwargs={'output_sizes': {'probability': 3}, 'allow_rechunk': True}
            )
        else:
            raise ValueError(f"Invalid dist_method: {self.dist_method}.")

        hindcast_prob = hindcast_prob.assign_coords(probability=('probability', ['PB', 'PN', 'PA']))
        return result_da * mask, mask * hindcast_prob.transpose('probability', 'T', 'Y', 'X')



class WAS_mme_MLP_:
    """
    Multi-Layer Perceptron (MLP) for Multi-Model Ensemble (MME) forecasting.
    This class implements a Multi-Layer Perceptron model using scikit-learn's MLPRegressor
    for deterministic forecasting, with optional tercile probability calculations using
    various statistical distributions. Implements hyperparameter optimization via randomized search.

    Parameters
    ----------
    hidden_layer_sizes_range : list of tuples, optional
        List of hidden layer sizes to tune, e.g., [(10,), (10, 5), (20, 10)] (default).
    activation_options : list of str, optional
        Activation functions to tune ('identity', 'logistic', 'tanh', 'relu') (default is ['relu', 'tanh', 'logistic']).
    solver_options : list of str, optional
        Solvers to tune ('lbfgs', 'sgd', 'adam') (default is ['adam', 'sgd', 'lbfgs']).
    alpha_range : list of float, optional
        L2 regularization parameters to tune (default is [0.0001, 0.001, 0.01, 0.1]).
    max_iter : int, optional
        Maximum iterations (default is 200).
    random_state : int, optional
        Seed for reproducibility (default is 42).
    dist_method : str, optional
        Distribution method for tercile probabilities ('t', 'gamma', etc.) (default is 'gamma').
    n_iter_search : int, optional
        Number of iterations for randomized search (default is 10).
    cv_folds : int, optional
        Number of cross-validation folds (default is 3).
    """
    def __init__(self,
                 hidden_layer_sizes_range=[(10,), (10, 5), (20, 10)],
                 activation_options=['relu', 'tanh', 'logistic'],
                 solver_options=['adam', 'sgd', 'lbfgs'],
                 alpha_range=[0.0001, 0.001, 0.01, 0.1],
                 max_iter=200, random_state=42, dist_method="gamma",
                 n_iter_search=10, cv_folds=3):

        self.hidden_layer_sizes_range = hidden_layer_sizes_range
        self.activation_options = activation_options
        self.solver_options = solver_options
        self.alpha_range = alpha_range
        self.max_iter = max_iter
        self.random_state = random_state
        self.dist_method = dist_method
        self.n_iter_search = n_iter_search
        self.cv_folds = cv_folds
        self.mlp = None

    def compute_hyperparameters(self, Predictors, Predictand, clim_year_start, clim_year_end):
        """
        Independently computes the best hyperparameters using randomized search on stacked training data.

        Parameters
        ----------
        X_train : xarray.DataArray
            Training predictor data with dimensions (T, M, Y, X).
        y_train : xarray.DataArray
            Training predictand data with dimensions (T, Y, X).
        clim_year_start : int
            Start year of the climatology period.
        clim_year_end : int
            End year of the climatology period. 
        Returns
        -------
        dict
            Best hyperparameters found.
        """

        X_train = standardize_timeseries(Predictors, clim_year_start, clim_year_end)
        y_train = standardize_timeseries(Predictand, clim_year_start, clim_year_end)

        # Stack training data
        X_train_stacked = X_train.stack(sample=('T', 'Y', 'X')).transpose('sample', 'M').values
        y_train_stacked = y_train.stack(sample=('T', 'Y', 'X')).values.ravel()  # Flatten to 1D
        train_nan_mask = np.any(~np.isfinite(X_train_stacked), axis=1) | ~np.isfinite(y_train_stacked)
        X_train_clean = X_train_stacked[~train_nan_mask]
        y_train_clean = y_train_stacked[~train_nan_mask]

        # Prepare parameter distributions
        param_dist = {
            'hidden_layer_sizes': self.hidden_layer_sizes_range,
            'activation': self.activation_options,
            'solver': self.solver_options,
            'alpha': self.alpha_range
        }

        # Initialize MLPRegressor base model
        model = MLPRegressor(max_iter=self.max_iter, random_state=self.random_state)

        # Randomized search
        random_search = RandomizedSearchCV(
            model, param_distributions=param_dist, n_iter=self.n_iter_search,
            cv=self.cv_folds, scoring='neg_mean_squared_error',
            random_state=self.random_state, error_score=np.nan
        )

        random_search.fit(X_train_clean, y_train_clean)
        best_params = random_search.best_params_

        return best_params

    def compute_model(self, X_train, y_train, X_test, y_test, best_params=None):
        """
        Compute deterministic hindcast using the MLP model with injected hyperparameters.

        Stacks and cleans the data, uses provided best_params (or computes if None),
        fits the final MLP with best params, and predicts on test data.

        Parameters
        ----------
        X_train : xarray.DataArray
            Training predictor data with dimensions (T, M, Y, X).
        y_train : xarray.DataArray
            Training predictand data with dimensions (T, Y, X).
        X_test : xarray.DataArray
            Testing predictor data with dimensions (T, M, Y, X).
        y_test : xarray.DataArray
            Testing predictand data with dimensions (T, Y, X).
        best_params : dict, optional
            Pre-computed best hyperparameters. If None, computes internally.

        Returns
        -------
        predicted_da : xarray.DataArray
            Deterministic hindcast with dimensions (T, Y, X).
        """
        # Extract coordinates
        time = X_test['T']
        lat = X_test['Y']
        lon = X_test['X']
        n_time = len(X_test.coords['T'])
        n_lat = len(X_test.coords['Y'])
        n_lon = len(X_test.coords['X'])

        # Stack training data
        X_train_stacked = X_train.stack(sample=('T', 'Y', 'X')).transpose('sample', 'M').values
        y_train_stacked = y_train.stack(sample=('T', 'Y', 'X')).values.ravel()  # Flatten to 1D
        train_nan_mask = np.any(~np.isfinite(X_train_stacked), axis=1) | ~np.isfinite(y_train_stacked)
        X_train_clean = X_train_stacked[~train_nan_mask]
        y_train_clean = y_train_stacked[~train_nan_mask]

        # Use provided best_params or compute if None
        if best_params is None:
            best_params = self.compute_hyperparameters(X_train_clean, y_train_clean)


        # Initialize and fit MLP with best params
        self.mlp = MLPRegressor(
            hidden_layer_sizes=best_params['hidden_layer_sizes'],
            activation=best_params['activation'],
            solver=best_params['solver'],
            alpha=best_params['alpha'],
            max_iter=self.max_iter,
            random_state=self.random_state
        )
        self.mlp.fit(X_train_clean, y_train_clean)

        # Stack testing data
        X_test_stacked = X_test.stack(sample=('T', 'Y', 'X')).transpose('sample', 'M').values
        y_test_stacked = y_test.stack(sample=('T', 'Y', 'X')).values.ravel()  # Flatten to 1D
        test_nan_mask = np.any(~np.isfinite(X_test_stacked), axis=1) | ~np.isfinite(y_test_stacked)

        # Predict
        y_pred = self.mlp.predict(X_test_stacked[~test_nan_mask])

        # Reconstruct the prediction array
        result = np.full_like(y_test_stacked, np.nan)
        result[test_nan_mask] = y_test_stacked[test_nan_mask]
        result[~test_nan_mask] = y_pred

        predictions_reshaped = result.reshape(n_time, n_lat, n_lon)
        predicted_da = xr.DataArray(data=predictions_reshaped,
                                    coords={'T': time, 'Y': lat, 'X': lon},
                                    dims=['T', 'Y', 'X'])
        return predicted_da


    # ------------------ Probability Calculation Methods ------------------
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
            # Transform thresholds
            first_t = (first_tercile - best_guess) / error_std
            second_t = (second_tercile - best_guess) / error_std

            pred_prob[0, :] = stats.t.cdf(first_t, df=dof)
            pred_prob[1, :] = stats.t.cdf(second_t, df=dof) - stats.t.cdf(first_t, df=dof)
            pred_prob[2, :] = 1 - stats.t.cdf(second_t, df=dof)

        return pred_prob

    @staticmethod
    def calculate_tercile_probabilities_weibull_min(best_guess, error_variance, first_tercile, second_tercile, dof):
        """
        Weibull minimum-based method.
        
        Here, we assume:
          - best_guess is used as the location,
          - error_std (sqrt(error_variance)) as the scale, and 
          - dof (degrees of freedom) as the shape parameter.
        
        Parameters
        ----------
        best_guess : array-like
            Forecast or best guess values.
        error_variance : array-like
            Variance associated with forecast errors.
        first_tercile : array-like
            First tercile threshold values.
        second_tercile : array-like
            Second tercile threshold values.
        dof : float or array-like
            Shape parameter for the Weibull minimum distribution.
            
        Returns
        -------
        pred_prob : np.ndarray
            A 3 x n_time array with probabilities for being below the first tercile,
            between the first and second tercile, and above the second tercile.
        """
        n_time = len(best_guess)
        pred_prob = np.empty((3, n_time))

        if np.all(np.isnan(best_guess)):
            pred_prob[:] = np.nan
        else:
            error_std = np.sqrt(error_variance)

            # Using the weibull_min CDF with best_guess as loc and error_std as scale.
            pred_prob[0, :] = stats.weibull_min.cdf(first_tercile, c=dof, loc=best_guess, scale=error_std)
            pred_prob[1, :] = stats.weibull_min.cdf(second_tercile, c=dof, loc=best_guess, scale=error_std) - \
                              stats.weibull_min.cdf(first_tercile, c=dof, loc=best_guess, scale=error_std)
            pred_prob[2, :] = 1 - stats.weibull_min.cdf(second_tercile, c=dof, loc=best_guess, scale=error_std)

        return pred_prob

    @staticmethod
    def calculate_tercile_probabilities_gamma(best_guess, error_variance, T1, T2, dof=None):
        """Gamma-distribution based method."""
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

    @staticmethod
    def calculate_tercile_probabilities_normal(best_guess, error_variance, first_tercile, second_tercile):
        """Normal-distribution based method."""
        n_time = len(best_guess)
        pred_prob = np.empty((3, n_time))
        if np.all(np.isnan(best_guess)):
            pred_prob[:] = np.nan
        else:
            error_std = np.sqrt(error_variance)
            pred_prob[0, :] = norm.cdf(first_tercile, loc=best_guess, scale=error_std)
            pred_prob[1, :] = norm.cdf(second_tercile, loc=best_guess, scale=error_std) - norm.cdf(first_tercile, loc=best_guess, scale=error_std)
            pred_prob[2, :] = 1 - norm.cdf(second_tercile, loc=best_guess, scale=error_std)
        return pred_prob

    @staticmethod
    def calculate_tercile_probabilities_lognormal(best_guess, error_variance, first_tercile, second_tercile):
        """Lognormal-distribution based method."""
        n_time = len(best_guess)
        pred_prob = np.empty((3, n_time))
        if np.any(np.isnan(best_guess)) or np.any(np.isnan(error_variance)):
            pred_prob[:] = np.nan
            return pred_prob
        sigma = np.sqrt(np.log(1 + error_variance / (best_guess**2)))
        mu = np.log(best_guess) - sigma**2 / 2
        pred_prob[0, :] = lognorm.cdf(first_tercile, s=sigma, scale=np.exp(mu))
        pred_prob[1, :] = lognorm.cdf(second_tercile, s=sigma, scale=np.exp(mu)) - lognorm.cdf(first_tercile, s=sigma, scale=np.exp(mu))
        pred_prob[2, :] = 1 - lognorm.cdf(second_tercile, s=sigma, scale=np.exp(mu))
        return pred_prob

    def compute_prob(self, Predictant, clim_year_start, clim_year_end, hindcast_det):
        """
        Compute tercile probabilities for the hindcast using the chosen distribution method.
        Predictant is an xarray DataArray with dims (T, Y, X).

        Parameters
        ----------
        Predictant : xarray.DataArray
            Observed data with dimensions (T, Y, X).
        clim_year_start : int or str
            Start year for climatology.
        clim_year_end : int or str
            End year for climatology.
        hindcast_det : xarray.DataArray
            Deterministic hindcast with dimensions (T, Y, X).

        Returns
        -------
        hindcast_prob : xarray.DataArray
            Probabilities for below-normal (PB), normal (PN), and above-normal (PA) with dimensions (probability, T, Y, X).
        """
        if "M" in Predictant.coords:
            Predictant = Predictant.isel(M=0).drop_vars('M').squeeze()
        mask = xr.where(~np.isnan(Predictant.isel(T=0)), 1, np.nan).drop_vars(['T']).squeeze().to_numpy()

        # Ensure Predictant is (T, Y, X)
        Predictant = Predictant.transpose('T', 'Y', 'X')
        index_start = Predictant.get_index("T").get_loc(str(clim_year_start)).start
        index_end = Predictant.get_index("T").get_loc(str(clim_year_end)).stop
        rainfall_for_tercile = Predictant.isel(T=slice(index_start, index_end))
        terciles = rainfall_for_tercile.quantile([0.33, 0.67], dim='T')
        error_variance = (Predictant - hindcast_det).var(dim='T')
        dof = len(Predictant.get_index("T")) - 2

        if self.dist_method == "t":
            calc_func = self.calculate_tercile_probabilities
            hindcast_prob = xr.apply_ufunc(
                calc_func,
                hindcast_det,
                error_variance,
                terciles.isel(quantile=0).drop_vars('quantile'),
                terciles.isel(quantile=1).drop_vars('quantile'),
                input_core_dims=[('T',), (), (), ()],
                vectorize=True,
                kwargs={'dof': dof},
                dask='parallelized',
                output_core_dims=[('probability', 'T')],
                output_dtypes=['float'],
                dask_gufunc_kwargs={'output_sizes': {'probability': 3}, "allow_rechunk": True}
            )
        elif self.dist_method == "weibull_min":
            calc_func = self.calculate_tercile_probabilities_weibull_min
            hindcast_prob = xr.apply_ufunc(
                calc_func,
                hindcast_det,
                error_variance,
                terciles.isel(quantile=0).drop_vars('quantile'),
                terciles.isel(quantile=1).drop_vars('quantile'),
                input_core_dims=[('T',), (), (), ()],
                vectorize=True,
                kwargs={'dof': dof},
                dask='parallelized',
                output_core_dims=[('probability', 'T')],
                output_dtypes=['float'],
                dask_gufunc_kwargs={'output_sizes': {'probability': 3}, "allow_rechunk": True}
            )
        elif self.dist_method == "gamma":
            calc_func = self.calculate_tercile_probabilities_gamma
            hindcast_prob = xr.apply_ufunc(
                calc_func,
                hindcast_det,
                error_variance,
                terciles.isel(quantile=0).drop_vars('quantile'),
                terciles.isel(quantile=1).drop_vars('quantile'),
                input_core_dims=[('T',), (), (), ()],
                vectorize=True,
                dask='parallelized',
                output_core_dims=[('probability', 'T')],
                output_dtypes=['float'],
                dask_gufunc_kwargs={'output_sizes': {'probability': 3}, "allow_rechunk": True}
            )
        elif self.dist_method == "nonparam":
            calc_func = self.calculate_tercile_probabilities_nonparametric
            error_samples = (Predictant - hindcast_det)
            hindcast_prob = xr.apply_ufunc(
                calc_func,
                hindcast_det,
                error_samples,
                terciles.isel(quantile=0).drop_vars('quantile'),
                terciles.isel(quantile=1).drop_vars('quantile'),
                input_core_dims=[('T',), ('T',), (), ()],
                output_core_dims=[('probability', 'T')],
                vectorize=True,
                dask='parallelized',
                output_dtypes=[float],
                dask_gufunc_kwargs={'output_sizes': {'probability': 3}, "allow_rechunk": True}
            )
        elif self.dist_method == "normal":
            calc_func = self.calculate_tercile_probabilities_normal
            hindcast_prob = xr.apply_ufunc(
                calc_func,
                hindcast_det,
                error_variance,
                terciles.isel(quantile=0).drop_vars('quantile'),
                terciles.isel(quantile=1).drop_vars('quantile'),
                input_core_dims=[('T',), (), (), ()],
                vectorize=True,
                dask='parallelized',
                output_core_dims=[('probability', 'T')],
                output_dtypes=['float'],
                dask_gufunc_kwargs={'output_sizes': {'probability': 3}, "allow_rechunk": True}
            )
        elif self.dist_method == "lognormal":
            calc_func = self.calculate_tercile_probabilities_lognormal
            hindcast_prob = xr.apply_ufunc(
                calc_func,
                hindcast_det,
                error_variance,
                terciles.isel(quantile=0).drop_vars('quantile'),
                terciles.isel(quantile=1).drop_vars('quantile'),
                input_core_dims=[('T',), (), (), ()],
                vectorize=True,
                dask='parallelized',
                output_core_dims=[('probability', 'T')],
                output_dtypes=['float'],
                dask_gufunc_kwargs={'output_sizes': {'probability': 3}, "allow_rechunk": True}
            )
        else:
            raise ValueError(f"Invalid dist_method: {self.dist_method}.")

        hindcast_prob = hindcast_prob.assign_coords(probability=('probability', ['PB', 'PN', 'PA']))
        return hindcast_prob.transpose('probability', 'T', 'Y', 'X')

    @staticmethod
    def _reshape_and_filter_data(da):
        """
        Helper: stack the DataArray from (T, Y, X[, M]) to (n_samples, n_features)
        and remove rows with NaNs.
        """
        da_stacked = da.stack(sample=('T', 'Y', 'X'))
        if 'M' in da.dims:
            da_stacked = da_stacked.transpose('sample', 'M')
        else:
            da_stacked = da_stacked.transpose('sample')
        da_values = da_stacked.values
        nan_mask = np.any(np.isnan(da_values), axis=1)
        return da_values[~nan_mask], nan_mask, da_values


    def forecast(self, Predictant, clim_year_start, clim_year_end, hindcast_det, hindcast_det_cross, Predictor_for_year, best_params=None):
        """
        Generate deterministic and probabilistic forecast for a target year using the MLP model with injected hyperparameters.

        Stacks and cleans the data, uses provided best_params (or computes if None),
        fits the final MLP with best params, predicts for the target year, reverses standardization,
        and computes tercile probabilities.

        Parameters
        ----------
        Predictant : xarray.DataArray
            Observed predictand data with dimensions (T, Y, X) or (T, M, Y, X).
        clim_year_start : int or str
            Start year of the climatology period.
        clim_year_end : int or str
            End year of the climatology period.
        hindcast_det : xarray.DataArray
            Deterministic hindcast data for training with dimensions (T, M, Y, X).
        hindcast_det_cross : xarray.DataArray
            Deterministic hindcast data for error estimation with dimensions (T, Y, X).
        Predictor_for_year : xarray.DataArray
            Predictor data for the target year with dimensions (T, M, Y, X).
        best_params : dict, optional
            Pre-computed best hyperparameters. If None, computes internally.

        Returns
        -------
        forecast_det : xarray.DataArray
            Deterministic forecast with dimensions (T, Y, X).
        forecast_prob : xarray.DataArray
            Tercile probabilities with dimensions (probability, T, Y, X).
        """
        if "M" in Predictant.coords:
            Predictant_no_m = Predictant.isel(M=0).drop_vars('M').squeeze()
        else:
            Predictant_no_m = Predictant

        mask = xr.where(~np.isnan(Predictant_no_m.isel(T=0)), 1, np.nan).drop_vars('T').squeeze().to_numpy()

        # Standardize Predictor_for_year using hindcast climatology
        mean_val = hindcast_det.sel(T=slice(str(clim_year_start), str(clim_year_end))).mean(dim='T')
        std_val = hindcast_det.sel(T=slice(str(clim_year_start), str(clim_year_end))).std(dim='T')
        Predictor_for_year_st = (Predictor_for_year - mean_val) / std_val

        hindcast_det_st = standardize_timeseries(hindcast_det, clim_year_start, clim_year_end)
        Predictant_st = standardize_timeseries(Predictant_no_m, clim_year_start, clim_year_end)
        y_test = Predictant_st.isel(T=[-1])

        # Extract coordinates
        time = Predictor_for_year_st['T']
        lat = Predictor_for_year_st['Y']
        lon = Predictor_for_year_st['X']
        n_time = len(Predictor_for_year_st.coords['T'])
        n_lat = len(Predictor_for_year_st.coords['Y'])
        n_lon = len(Predictor_for_year_st.coords['X'])

        # Stack training data
        X_train_stacked = hindcast_det_st.stack(sample=('T', 'Y', 'X')).transpose('sample', 'M').values
        y_train_stacked = Predictant_st.stack(sample=('T', 'Y', 'X')).values.ravel()
        train_nan_mask = np.any(~np.isfinite(X_train_stacked), axis=1) | ~np.isfinite(y_train_stacked)
        X_train_clean = X_train_stacked[~train_nan_mask]
        y_train_clean = y_train_stacked[~train_nan_mask]

        # Use provided best_params or compute if None
        if best_params is None:
            best_params = self.compute_hyperparameters(X_train_clean, y_train_clean)

        # Initialize and fit MLP with best params
        self.mlp = MLPRegressor(
            hidden_layer_sizes=best_params['hidden_layer_sizes'],
            activation=best_params['activation'],
            solver=best_params['solver'],
            alpha=best_params['alpha'],
            max_iter=self.max_iter,
            random_state=self.random_state
        )
        self.mlp.fit(X_train_clean, y_train_clean)

        # Stack testing data
        X_test_stacked = Predictor_for_year_st.stack(sample=('T', 'Y', 'X')).transpose('sample', 'M').values
        y_test_stacked = y_test.stack(sample=('T', 'Y', 'X')).values.ravel()
        test_nan_mask = np.any(~np.isfinite(X_test_stacked), axis=1) | ~np.isfinite(y_test_stacked)

        # Predict
        y_pred = self.mlp.predict(X_test_stacked[~test_nan_mask])

        # Reconstruct the prediction array
        result = np.full_like(y_test_stacked, np.nan)
        result[test_nan_mask] = y_test_stacked[test_nan_mask]
        result[~test_nan_mask] = y_pred

        predictions_reshaped = result.reshape(n_time, n_lat, n_lon)
        result_da = xr.DataArray(data=predictions_reshaped,
                                 coords={'T': time, 'Y': lat, 'X': lon},
                                 dims=['T', 'Y', 'X']) * mask
        result_da = reverse_standardize(result_da, Predictant_no_m,
                                             clim_year_start, clim_year_end)
        year = Predictor_for_year.coords['T'].values.astype('datetime64[Y]').astype(int)[0] + 1970
        T_value_1 = Predictant_no_m.isel(T=0).coords['T'].values
        month_1 = T_value_1.astype('datetime64[M]').astype(int) % 12 + 1
        new_T_value = np.datetime64(f"{year}-{month_1:02d}-01")
        result_da = result_da.assign_coords(T=xr.DataArray([new_T_value], dims=["T"]))
        result_da['T'] = result_da['T'].astype('datetime64[ns]')

        # Compute tercile probabilities
        index_start = Predictant_no_m.get_index("T").get_loc(str(clim_year_start)).start
        index_end = Predictant_no_m.get_index("T").get_loc(str(clim_year_end)).stop
        rainfall_for_tercile = Predictant_no_m.isel(T=slice(index_start, index_end))
        terciles = rainfall_for_tercile.quantile([0.33, 0.67], dim='T')
        error_variance = (Predictant_no_m - hindcast_det_cross).var(dim='T')
        dof = len(Predictant_no_m.get_index("T")) - 2

        if self.dist_method == "t":
            calc_func = self.calculate_tercile_probabilities
            hindcast_prob = xr.apply_ufunc(
                calc_func,
                result_da,
                error_variance,
                terciles.isel(quantile=0).drop_vars('quantile'),
                terciles.isel(quantile=1).drop_vars('quantile'),
                input_core_dims=[('T',), (), (), ()],
                vectorize=True,
                kwargs={'dof': dof},
                dask='parallelized',
                output_core_dims=[('probability','T')],
                output_dtypes=['float'],
                dask_gufunc_kwargs={'output_sizes': {'probability': 3}, "allow_rechunk": True}
            )

        elif self.dist_method == "weibull_min":
            calc_func = self.calculate_tercile_probabilities_weibull_min
            hindcast_prob = xr.apply_ufunc(
                calc_func,
                result_da,
                error_variance,
                terciles.isel(quantile=0).drop_vars('quantile'),
                terciles.isel(quantile=1).drop_vars('quantile'),
                input_core_dims=[('T',), (), (), ()],
                vectorize=True,
                kwargs={'dof': dof},
                dask='parallelized',
                output_core_dims=[('probability', 'T')],
                output_dtypes=['float'],
                dask_gufunc_kwargs={'output_sizes': {'probability': 3}, "allow_rechunk": True}
            )
        elif self.dist_method == "gamma":
            calc_func = self.calculate_tercile_probabilities_gamma
            hindcast_prob = xr.apply_ufunc(
                calc_func,
                result_da,
                error_variance,
                terciles.isel(quantile=0).drop_vars('quantile'),
                terciles.isel(quantile=1).drop_vars('quantile'),
                input_core_dims=[('T',), (), (), ()],
                vectorize=True,
                dask='parallelized',
                output_core_dims=[('probability', 'T')],
                output_dtypes=['float'],
                dask_gufunc_kwargs={'output_sizes': {'probability': 3}, "allow_rechunk": True}
            )
        elif self.dist_method == "normal":
            calc_func = self.calculate_tercile_probabilities_normal
            hindcast_prob = xr.apply_ufunc(
                calc_func,
                result_da,
                error_variance,
                terciles.isel(quantile=0).drop_vars('quantile'),
                terciles.isel(quantile=1).drop_vars('quantile'),
                input_core_dims=[('T',), (), (), ()],
                vectorize=True,
                dask='parallelized',
                output_core_dims=[('probability', 'T')],
                output_dtypes=['float'],
                dask_gufunc_kwargs={'output_sizes': {'probability': 3}, "allow_rechunk": True}
            )
        elif self.dist_method == "lognormal":
            calc_func = self.calculate_tercile_probabilities_lognormal
            hindcast_prob = xr.apply_ufunc(
                calc_func,
                result_da,
                error_variance,
                terciles.isel(quantile=0).drop_vars('quantile'),
                terciles.isel(quantile=1).drop_vars('quantile'),
                input_core_dims=[('T',), (), (), ()],
                vectorize=True,
                dask='parallelized',
                output_core_dims=[('probability', 'T')],
                output_dtypes=['float'],
                dask_gufunc_kwargs={'output_sizes': {'probability': 3}, "allow_rechunk": True}
            )
        elif self.dist_method == "nonparam":
            calc_func = self.calculate_tercile_probabilities_nonparametric
            error_samples = Predictant_no_m - hindcast_det_cross
            error_samples = error_samples.rename({'T': 'S'})
            hindcast_prob = xr.apply_ufunc(
                calc_func,
                result_da,
                error_samples,
                terciles.isel(quantile=0).drop_vars('quantile'),
                terciles.isel(quantile=1).drop_vars('quantile'),
                input_core_dims=[('T',), ('S',), (), ()],
                output_core_dims=[('probability', 'T')],
                vectorize=True,
                dask='parallelized',
                output_dtypes=[float],
                dask_gufunc_kwargs={'output_sizes': {'probability': 3}, "allow_rechunk": True}
            )
        else:
            raise ValueError(f"Invalid dist_method: {self.dist_method}.")

        hindcast_prob = hindcast_prob.assign_coords(probability=('probability', ['PB', 'PN', 'PA']))
        return result_da * mask, mask * hindcast_prob.transpose('probability', 'T', 'Y', 'X')


class WAS_mme_MLP:
    """
    Multi-Layer Perceptron (MLP) for Multi-Model Ensemble (MME) forecasting.
    This class implements a Multi-Layer Perceptron model using scikit-learn's MLPRegressor
    for deterministic forecasting, with optional tercile probability calculations using
    various statistical distributions. Implements hyperparameter optimization via randomized search.
    Parameters
    ----------
    hidden_layer_sizes_range : list of tuples, optional
        List of hidden layer sizes to tune, e.g., [(10,), (10, 5), (20, 10)] (default).
    activation_options : list of str, optional
        Activation functions to tune ('identity', 'logistic', 'tanh', 'relu') (default is ['relu', 'tanh', 'logistic']).
    solver_options : list of str, optional
        Solvers to tune ('lbfgs', 'sgd', 'adam') (default is ['adam', 'sgd', 'lbfgs']).
    alpha_range : list of float, optional
        L2 regularization parameters to tune (default is [0.0001, 0.001, 0.01, 0.1]).
    max_iter : int, optional
        Maximum iterations (default is 200).
    random_state : int, optional
        Seed for reproducibility (default is 42).
    dist_method : str, optional
        Distribution method for tercile probabilities ('t', 'gamma', etc.) (default is 'gamma').
    n_iter_search : int, optional
        Number of iterations for randomized search (default is 10).
    cv_folds : int, optional
        Number of cross-validation folds (default is 3).
    n_clusters : int, optional
        Number of clusters for homogenized zones (default is 4).
    """
    def __init__(self,
                 hidden_layer_sizes_range=[(10,), (10, 5), (20, 10)],
                 activation_options=['relu', 'tanh', 'logistic'],
                 solver_options=['adam', 'sgd', 'lbfgs'],
                 alpha_range=[0.0001, 0.001, 0.01, 0.1],
                 max_iter=200, random_state=42, dist_method="gamma",
                 n_iter_search=10, cv_folds=3, n_clusters=4):
        self.hidden_layer_sizes_range = hidden_layer_sizes_range
        self.activation_options = activation_options
        self.solver_options = solver_options
        self.alpha_range = alpha_range
        self.max_iter = max_iter
        self.random_state = random_state
        self.dist_method = dist_method
        self.n_iter_search = n_iter_search
        self.cv_folds = cv_folds
        self.n_clusters = n_clusters
        self.mlp = None

    def compute_hyperparameters(self, Predictors, Predictand, clim_year_start, clim_year_end):
        """
        Independently computes the best hyperparameters using randomized search on stacked training data for each homogenized zone.
        Parameters
        ----------
        Predictors : xarray.DataArray
            Training predictor data with dimensions (T, M, Y, X).
        Predictand : xarray.DataArray
            Training predictand data with dimensions (T, Y, X).
        clim_year_start : int
            Start year of the climatology period.
        clim_year_end : int
            End year of the climatology period.
        Returns
        -------
        best_params_dict : dict
            Best hyperparameters for each cluster.
        cluster_da : xarray.DataArray
            Cluster labels with dimensions (Y, X).
        """
        if "M" in Predictand.coords:
            Predictand = Predictand.isel(M=0).drop_vars('M').squeeze()
           
        X_train_std = standardize_timeseries(Predictors, clim_year_start, clim_year_end)
        Predictand.name = "varname"
        # Step 1: Perform KMeans clustering based on predictand's spatial distribution
        kmeans = KMeans(n_clusters=self.n_clusters, random_state=self.random_state)
        Predictand_dropna = Predictand.to_dataframe().reset_index().dropna().drop(columns=['T'])
        variable_column = Predictand_dropna.columns[2]
        Predictand_dropna['cluster'] = kmeans.fit_predict(
           Predictand_dropna[[variable_column]]
        )
       
        # Convert cluster assignments back into an xarray structure
        df_unique = Predictand_dropna.drop_duplicates(subset=['Y', 'X'])
        dataset = df_unique.set_index(['Y', 'X']).to_xarray()
        mask = xr.where(~np.isnan(Predictand.isel(T=0)), 1, np.nan)
        Cluster = (dataset['cluster'] * mask)
              
        # Align cluster array with the predictand array
        xarray1, xarray2 = xr.align(Predictand, Cluster, join="outer")
       
        # Identify unique cluster labels
        clusters = np.unique(xarray2)
        clusters = clusters[~np.isnan(clusters)]
        cluster_da = xarray2
        y_train_std = standardize_timeseries(Predictand, clim_year_start, clim_year_end)
        X_train_std['T'] = y_train_std['T']
        
        # Prepare parameter distributions
        param_dist = {
            'hidden_layer_sizes': self.hidden_layer_sizes_range,
            'activation': self.activation_options,
            'solver': self.solver_options,
            'alpha': self.alpha_range
        }
        best_params_dict = {}
        for c in clusters:
            mask_3d = (cluster_da == c).expand_dims({'T': y_train_std['T']})
            X_stacked_c = X_train_std.where(mask_3d).stack(sample=('T', 'Y', 'X')).transpose('sample', 'M').values
            y_stacked_c = y_train_std.where(mask_3d).stack(sample=('T', 'Y', 'X')).values.ravel()
            nan_mask_c = np.any(~np.isfinite(X_stacked_c), axis=1) | ~np.isfinite(y_stacked_c)
            X_clean_c = X_stacked_c[~nan_mask_c]
            y_clean_c = y_stacked_c[~nan_mask_c]
            if len(X_clean_c) == 0:
                continue
            # Initialize MLPRegressor base model
            model = MLPRegressor(max_iter=self.max_iter, random_state=self.random_state)
            # Randomized search
            random_search = RandomizedSearchCV(
                model, param_distributions=param_dist, n_iter=self.n_iter_search,
                cv=self.cv_folds, scoring='neg_mean_squared_error',
                random_state=self.random_state, error_score=np.nan
            )
            random_search.fit(X_clean_c, y_clean_c)
            best_params_dict[c] = random_search.best_params_
        return best_params_dict, cluster_da

    def compute_model(self, X_train, y_train, X_test, y_test, best_params=None, cluster_da=None):
        """
        Compute deterministic hindcast using the MLP model with injected hyperparameters for each zone.
        Parameters
        ----------
        X_train : xarray.DataArray
            Training predictor data with dimensions (T, M, Y, X).
        y_train : xarray.DataArray
            Training predictand data with dimensions (T, Y, X).
        X_test : xarray.DataArray
            Testing predictor data with dimensions (T, M, Y, X).
        y_test : xarray.DataArray
            Testing predictand data with dimensions (T, Y, X).
        clim_year_start : int
            Start year of the climatology period.
        clim_year_end : int
            End year of the climatology period.
        best_params : dict, optional
            Pre-computed best hyperparameters per cluster. If None, computes internally.
        cluster_da : xarray.DataArray, optional
            Pre-computed cluster labels. If None, computes internally.
        Returns
        -------
        predicted_da : xarray.DataArray
            Deterministic hindcast with dimensions (T, Y, X).
        """
        # Standardize inputs
        X_train_std = X_train
        y_train_std = y_train
        X_test_std = X_test
        y_test_std = y_test
        
        # Extract coordinate variables from X_test
        time = X_test_std['T']
        lat = X_test_std['Y']
        lon = X_test_std['X']
        n_time = len(time)
        n_lat = len(lat)
        n_lon = len(lon)
        
        # Use provided best_params and cluster_da or compute if None
        if best_params is None:
            best_params, cluster_da = self.compute_hyperparameters(X_train, y_train, clim_year_start, clim_year_end)
        # Initialize predictions array
        predictions = np.full((n_time, n_lat, n_lon), np.nan)
        self.mlp = {}  # Dictionary to store models per cluster
        for c in range(self.n_clusters):
            if c not in best_params:
                continue
            bp = best_params[c]
            # Mask for this cluster
            mask_3d_train = (cluster_da == c).expand_dims({'T': X_train_std['T']})
            mask_3d_test = (cluster_da == c).expand_dims({'T': X_test_std['T']})
            # Stack training data for cluster
            X_train_stacked_c = X_train_std.where(mask_3d_train).stack(sample=('T', 'Y', 'X')).transpose('sample', 'M').values
            y_train_stacked_c = y_train_std.where(mask_3d_train).stack(sample=('T', 'Y', 'X')).values.ravel()
            train_nan_mask = np.any(~np.isfinite(X_train_stacked_c), axis=1) | ~np.isfinite(y_train_stacked_c)
            X_train_clean_c = X_train_stacked_c[~train_nan_mask]
            y_train_clean_c = y_train_stacked_c[~train_nan_mask]
            
            # Stack testing data for cluster
            X_test_stacked_c = X_test_std.where(mask_3d_test).stack(sample=('T', 'Y', 'X')).transpose('sample', 'M').values
            y_test_stacked_c = y_test_std.where(mask_3d_test).stack(sample=('T', 'Y', 'X')).values.ravel()
            test_nan_mask = np.any(~np.isfinite(X_test_stacked_c), axis=1) | ~np.isfinite(y_test_stacked_c)
            X_test_clean_c = X_test_stacked_c[~test_nan_mask]
            
            # Initialize and fit MLP with best params for this cluster
            mlp_c = MLPRegressor(
                hidden_layer_sizes=bp['hidden_layer_sizes'],
                activation=bp['activation'],
                solver=bp['solver'],
                alpha=bp['alpha'],
                max_iter=self.max_iter,
                random_state=self.random_state
            )
            mlp_c.fit(X_train_clean_c, y_train_clean_c)
            self.mlp[c] = mlp_c
            # Predict
            y_pred_c = mlp_c.predict(X_test_clean_c)
            # Reconstruct predictions for this cluster
            result_c = np.full(len(y_test_stacked_c), np.nan)
            result_c[~test_nan_mask] = y_pred_c
            pred_c_reshaped = result_c.reshape(n_time, n_lat, n_lon)
            # Fill in the predictions array
            predictions = np.where(np.isnan(predictions), pred_c_reshaped, predictions)
        predicted_da = xr.DataArray(
            data=predictions,
            coords={'T': time, 'Y': lat, 'X': lon},
            dims=['T', 'Y', 'X']
        )

        return predicted_da

    # ------------------ Probability Calculation Methods ------------------
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
            # Transform thresholds
            first_t = (first_tercile - best_guess) / error_std
            second_t = (second_tercile - best_guess) / error_std
            pred_prob[0, :] = t.cdf(first_t, df=dof)
            pred_prob[1, :] = t.cdf(second_t, df=dof) - t.cdf(first_t, df=dof)
            pred_prob[2, :] = 1 - t.cdf(second_t, df=dof)
        return pred_prob

    @staticmethod
    def calculate_tercile_probabilities_weibull_min(best_guess, error_variance, first_tercile, second_tercile, dof):
        """
        Weibull minimum-based method.
       
        Here, we assume:
          - best_guess is used as the location,
          - error_std (sqrt(error_variance)) as the scale, and
          - dof (degrees of freedom) as the shape parameter.
       
        Parameters
        ----------
        best_guess : array-like
            Forecast or best guess values.
        error_variance : array-like
            Variance associated with forecast errors.
        first_tercile : array-like
            First tercile threshold values.
        second_tercile : array-like
            Second tercile threshold values.
        dof : float or array-like
            Shape parameter for the Weibull minimum distribution.
           
        Returns
        -------
        pred_prob : np.ndarray
            A 3 x n_time array with probabilities for being below the first tercile,
            between the first and second tercile, and above the second tercile.
        """
        n_time = len(best_guess)
        pred_prob = np.empty((3, n_time))
        if np.all(np.isnan(best_guess)):
            pred_prob[:] = np.nan
        else:
            error_std = np.sqrt(error_variance)
            # Using the weibull_min CDF with best_guess as loc and error_std as scale.
            pred_prob[0, :] = weibull_min.cdf(first_tercile, c=dof, loc=best_guess, scale=error_std)
            pred_prob[1, :] = weibull_min.cdf(second_tercile, c=dof, loc=best_guess, scale=error_std) - \
                              weibull_min.cdf(first_tercile, c=dof, loc=best_guess, scale=error_std)
            pred_prob[2, :] = 1 - weibull_min.cdf(second_tercile, c=dof, loc=best_guess, scale=error_std)
        return pred_prob

    @staticmethod
    def calculate_tercile_probabilities_gamma(best_guess, error_variance, T1, T2, dof=None):
        """Gamma-distribution based method."""
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

    @staticmethod
    def calculate_tercile_probabilities_normal(best_guess, error_variance, first_tercile, second_tercile):
        """Normal-distribution based method."""
        n_time = len(best_guess)
        pred_prob = np.empty((3, n_time))
        if np.all(np.isnan(best_guess)):
            pred_prob[:] = np.nan
        else:
            error_std = np.sqrt(error_variance)
            pred_prob[0, :] = norm.cdf(first_tercile, loc=best_guess, scale=error_std)
            pred_prob[1, :] = norm.cdf(second_tercile, loc=best_guess, scale=error_std) - norm.cdf(first_tercile, loc=best_guess, scale=error_std)
            pred_prob[2, :] = 1 - norm.cdf(second_tercile, loc=best_guess, scale=error_std)
        return pred_prob

    @staticmethod
    def calculate_tercile_probabilities_lognormal(best_guess, error_variance, first_tercile, second_tercile):
        """Lognormal-distribution based method."""
        n_time = len(best_guess)
        pred_prob = np.empty((3, n_time))
        if np.any(np.isnan(best_guess)) or np.any(np.isnan(error_variance)):
            pred_prob[:] = np.nan
            return pred_prob
        sigma = np.sqrt(np.log(1 + error_variance / (best_guess**2)))
        mu = np.log(best_guess) - sigma**2 / 2
        pred_prob[0, :] = lognorm.cdf(first_tercile, s=sigma, scale=np.exp(mu))
        pred_prob[1, :] = lognorm.cdf(second_tercile, s=sigma, scale=np.exp(mu)) - lognorm.cdf(first_tercile, s=sigma, scale=np.exp(mu))
        pred_prob[2, :] = 1 - lognorm.cdf(second_tercile, s=sigma, scale=np.exp(mu))
        return pred_prob

    def compute_prob(self, Predictant, clim_year_start, clim_year_end, hindcast_det):
        """
        Compute tercile probabilities for the hindcast using the chosen distribution method.
        Predictant is an xarray DataArray with dims (T, Y, X).
        Parameters
        ----------
        Predictant : xarray.DataArray
            Observed data with dimensions (T, Y, X).
        clim_year_start : int or str
            Start year for climatology.
        clim_year_end : int or str
            End year for climatology.
        hindcast_det : xarray.DataArray
            Deterministic hindcast with dimensions (T, Y, X).
        Returns
        -------
        hindcast_prob : xarray.DataArray
            Probabilities for below-normal (PB), normal (PN), and above-normal (PA) with dimensions (probability, T, Y, X).
        """
        if "M" in Predictant.coords:
            Predictant = Predictant.isel(M=0).drop_vars('M').squeeze()
        mask = xr.where(~np.isnan(Predictant.isel(T=0)), 1, np.nan).drop_vars(['T']).squeeze().to_numpy()
        # Ensure Predictant is (T, Y, X)
        Predictant = Predictant.transpose('T', 'Y', 'X')
        index_start = Predictant.get_index("T").get_loc(str(clim_year_start)).start
        index_end = Predictant.get_index("T").get_loc(str(clim_year_end)).stop
        rainfall_for_tercile = Predictant.isel(T=slice(index_start, index_end))
        terciles = rainfall_for_tercile.quantile([0.33, 0.67], dim='T')
        error_variance = (Predictant - hindcast_det).var(dim='T')
        dof = len(Predictant.get_index("T")) - 2
        if self.dist_method == "t":
            calc_func = self.calculate_tercile_probabilities
            hindcast_prob = xr.apply_ufunc(
                calc_func,
                hindcast_det,
                error_variance,
                terciles.isel(quantile=0).drop_vars('quantile'),
                terciles.isel(quantile=1).drop_vars('quantile'),
                input_core_dims=[('T',), (), (), ()],
                vectorize=True,
                kwargs={'dof': dof},
                dask='parallelized',
                output_core_dims=[('probability', 'T')],
                output_dtypes=['float'],
                dask_gufunc_kwargs={'output_sizes': {'probability': 3}, "allow_rechunk": True}
            )
        elif self.dist_method == "weibull_min":
            calc_func = self.calculate_tercile_probabilities_weibull_min
            hindcast_prob = xr.apply_ufunc(
                calc_func,
                hindcast_det,
                error_variance,
                terciles.isel(quantile=0).drop_vars('quantile'),
                terciles.isel(quantile=1).drop_vars('quantile'),
                input_core_dims=[('T',), (), (), ()],
                vectorize=True,
                kwargs={'dof': dof},
                dask='parallelized',
                output_core_dims=[('probability', 'T')],
                output_dtypes=['float'],
                dask_gufunc_kwargs={'output_sizes': {'probability': 3}, "allow_rechunk": True}
            )
        elif self.dist_method == "gamma":
            calc_func = self.calculate_tercile_probabilities_gamma
            hindcast_prob = xr.apply_ufunc(
                calc_func,
                hindcast_det,
                error_variance,
                terciles.isel(quantile=0).drop_vars('quantile'),
                terciles.isel(quantile=1).drop_vars('quantile'),
                input_core_dims=[('T',), (), (), ()],
                vectorize=True,
                dask='parallelized',
                output_core_dims=[('probability', 'T')],
                output_dtypes=['float'],
                dask_gufunc_kwargs={'output_sizes': {'probability': 3}, "allow_rechunk": True}
            )
        elif self.dist_method == "nonparam":
            calc_func = self.calculate_tercile_probabilities_nonparametric
            error_samples = (Predictant - hindcast_det)
            hindcast_prob = xr.apply_ufunc(
                calc_func,
                hindcast_det,
                error_samples,
                terciles.isel(quantile=0).drop_vars('quantile'),
                terciles.isel(quantile=1).drop_vars('quantile'),
                input_core_dims=[('T',), ('T',), (), ()],
                output_core_dims=[('probability', 'T')],
                vectorize=True,
                dask='parallelized',
                output_dtypes=[float],
                dask_gufunc_kwargs={'output_sizes': {'probability': 3}, "allow_rechunk": True}
            )
        elif self.dist_method == "normal":
            calc_func = self.calculate_tercile_probabilities_normal
            hindcast_prob = xr.apply_ufunc(
                calc_func,
                hindcast_det,
                error_variance,
                terciles.isel(quantile=0).drop_vars('quantile'),
                terciles.isel(quantile=1).drop_vars('quantile'),
                input_core_dims=[('T',), (), (), ()],
                vectorize=True,
                dask='parallelized',
                output_core_dims=[('probability', 'T')],
                output_dtypes=['float'],
                dask_gufunc_kwargs={'output_sizes': {'probability': 3}, "allow_rechunk": True}
            )
        elif self.dist_method == "lognormal":
            calc_func = self.calculate_tercile_probabilities_lognormal
            hindcast_prob = xr.apply_ufunc(
                calc_func,
                hindcast_det,
                error_variance,
                terciles.isel(quantile=0).drop_vars('quantile'),
                terciles.isel(quantile=1).drop_vars('quantile'),
                input_core_dims=[('T',), (), (), ()],
                vectorize=True,
                dask='parallelized',
                output_core_dims=[('probability', 'T')],
                output_dtypes=['float'],
                dask_gufunc_kwargs={'output_sizes': {'probability': 3}, "allow_rechunk": True}
            )
        else:
            raise ValueError(f"Invalid dist_method: {self.dist_method}.")
        hindcast_prob = hindcast_prob.assign_coords(probability=('probability', ['PB', 'PN', 'PA']))
        return hindcast_prob.transpose('probability', 'T', 'Y', 'X')

    @staticmethod
    def _reshape_and_filter_data(da):
        """
        Helper: stack the DataArray from (T, Y, X[, M]) to (n_samples, n_features)
        and remove rows with NaNs.
        """
        da_stacked = da.stack(sample=('T', 'Y', 'X'))
        if 'M' in da.dims:
            da_stacked = da_stacked.transpose('sample', 'M')
        else:
            da_stacked = da_stacked.transpose('sample')
        da_values = da_stacked.values
        nan_mask = np.any(np.isnan(da_values), axis=1)
        return da_values[~nan_mask], nan_mask, da_values

    def forecast(self, Predictant, clim_year_start, clim_year_end, hindcast_det, hindcast_det_cross, Predictor_for_year, best_params=None, cluster_da=None):
        """
        Generate deterministic and probabilistic forecast for a target year using the MLP model with injected hyperparameters.
        Stacks and cleans the data, uses provided best_params (or computes if None),
        fits the final MLP with best params, predicts for the target year, reverses standardization,
        and computes tercile probabilities.
        Parameters
        ----------
        Predictant : xarray.DataArray
            Observed predictand data with dimensions (T, Y, X) or (T, M, Y, X).
        clim_year_start : int or str
            Start year of the climatology period.
        clim_year_end : int or str
            End year of the climatology period.
        hindcast_det : xarray.DataArray
            Deterministic hindcast data for training with dimensions (T, M, Y, X).
        hindcast_det_cross : xarray.DataArray
            Deterministic hindcast data for error estimation with dimensions (T, Y, X).
        Predictor_for_year : xarray.DataArray
            Predictor data for the target year with dimensions (T, M, Y, X).
        best_params : dict, optional
            Pre-computed best hyperparameters. If None, computes internally.
        cluster_da : xarray.DataArray, optional
            Pre-computed cluster labels. If None, computes internally.
        Returns
        -------
        forecast_det : xarray.DataArray
            Deterministic forecast with dimensions (T, Y, X).
        forecast_prob : xarray.DataArray
            Tercile probabilities with dimensions (probability, T, Y, X).
        """
        if "M" in Predictant.coords:
            Predictant_no_m = Predictant.isel(M=0).drop_vars('M').squeeze()
        else:
            Predictant_no_m = Predictant
        mask = xr.where(~np.isnan(Predictant_no_m.isel(T=0)), 1, np.nan).drop_vars(['T']).squeeze().to_numpy()
        # Standardize Predictor_for_year using hindcast climatology
        mean_val = hindcast_det.sel(T=slice(str(clim_year_start), str(clim_year_end))).mean(dim='T')
        std_val = hindcast_det.sel(T=slice(str(clim_year_start), str(clim_year_end))).std(dim='T')
        Predictor_for_year_st = (Predictor_for_year - mean_val) / std_val
        hindcast_det_st = standardize_timeseries(hindcast_det, clim_year_start, clim_year_end)
        Predictant_st = standardize_timeseries(Predictant_no_m, clim_year_start, clim_year_end)
        hindcast_det_st['T'] = Predictant_st['T']
        
        # Extract coordinates
        time = Predictor_for_year_st['T']
        lat = Predictor_for_year_st['Y']
        lon = Predictor_for_year_st['X']
        n_time = len(time)
        n_lat = len(lat)
        n_lon = len(lon)
        
        # Use provided best_params and cluster_da or compute if None
        if best_params is None:
            best_params, cluster_da = self.compute_hyperparameters(hindcast_det, Predictant_no_m, clim_year_start, clim_year_end)
        # Initialize predictions array
        predictions = np.full((n_time, n_lat, n_lon), np.nan)
        self.mlp = {}  # Dictionary to store models per cluster
        for c in range(self.n_clusters):
            if c not in best_params:
                continue
            bp = best_params[c]
            # Mask for this cluster
            mask_3d_train = (cluster_da == c).expand_dims({'T': hindcast_det_st['T']})
            mask_3d_test = (cluster_da == c).expand_dims({'T': Predictor_for_year_st['T']})
            # Stack training data for cluster
            X_train_stacked_c = hindcast_det_st.where(mask_3d_train).stack(sample=('T', 'Y', 'X')).transpose('sample', 'M').values
            y_train_stacked_c = Predictant_st.where(mask_3d_train).stack(sample=('T', 'Y', 'X')).values.ravel()
            train_nan_mask = np.any(~np.isfinite(X_train_stacked_c), axis=1) | ~np.isfinite(y_train_stacked_c)
            X_train_clean_c = X_train_stacked_c[~train_nan_mask]
            y_train_clean_c = y_train_stacked_c[~train_nan_mask]
            # Stack testing data for cluster
            X_test_stacked_c = Predictor_for_year_st.where(mask_3d_test).stack(sample=('T', 'Y', 'X')).transpose('sample', 'M').values
            test_nan_mask = np.any(~np.isfinite(X_test_stacked_c), axis=1)
            X_test_clean_c = X_test_stacked_c[~test_nan_mask]
            # Initialize and fit MLP with best params for this cluster
            mlp_c = MLPRegressor(
                hidden_layer_sizes=bp['hidden_layer_sizes'],
                activation=bp['activation'],
                solver=bp['solver'],
                alpha=bp['alpha'],
                max_iter=self.max_iter,
                random_state=self.random_state
            )
            mlp_c.fit(X_train_clean_c, y_train_clean_c)
            self.mlp[c] = mlp_c
            # Predict
            y_pred_c = mlp_c.predict(X_test_clean_c)
            # Reconstruct predictions for this cluster
            result_c = np.full(len(X_test_stacked_c), np.nan)
            result_c[~test_nan_mask] = y_pred_c
            pred_c_reshaped = result_c.reshape(n_time, n_lat, n_lon)
            # Fill in the predictions array
            predictions = np.where(np.isnan(predictions), pred_c_reshaped, predictions)
        result_da = xr.DataArray(
            data=predictions,
            coords={'T': time, 'Y': lat, 'X': lon},
            dims=['T', 'Y', 'X']
        ) * mask
        result_da = reverse_standardize(result_da, Predictant_no_m, clim_year_start, clim_year_end)
        year = Predictor_for_year.coords['T'].values.astype('datetime64[Y]').astype(int)[0] + 1970
        T_value_1 = Predictant_no_m.isel(T=0).coords['T'].values
        month_1 = T_value_1.astype('datetime64[M]').astype(int) % 12 + 1
        new_T_value = np.datetime64(f"{year}-{month_1:02d}-01")
        result_da = result_da.assign_coords(T=xr.DataArray([new_T_value], dims=["T"]))
        result_da['T'] = result_da['T'].astype('datetime64[ns]')
        # Compute tercile probabilities
        index_start = Predictant_no_m.get_index("T").get_loc(str(clim_year_start)).start
        index_end = Predictant_no_m.get_index("T").get_loc(str(clim_year_end)).stop
        rainfall_for_tercile = Predictant_no_m.isel(T=slice(index_start, index_end))
        terciles = rainfall_for_tercile.quantile([0.33, 0.67], dim='T')
        error_variance = (Predictant_no_m - hindcast_det_cross).var(dim='T')
        dof = len(Predictant_no_m.get_index("T")) - 2
        if self.dist_method == "t":
            calc_func = self.calculate_tercile_probabilities
            hindcast_prob = xr.apply_ufunc(
                calc_func,
                result_da,
                error_variance,
                terciles.isel(quantile=0).drop_vars('quantile'),
                terciles.isel(quantile=1).drop_vars('quantile'),
                input_core_dims=[('T',), (), (), ()],
                vectorize=True,
                kwargs={'dof': dof},
                dask='parallelized',
                output_core_dims=[('probability', 'T')],
                output_dtypes=['float'],
                dask_gufunc_kwargs={'output_sizes': {'probability': 3}, "allow_rechunk": True}
            )
        elif self.dist_method == "weibull_min":
            calc_func = self.calculate_tercile_probabilities_weibull_min
            hindcast_prob = xr.apply_ufunc(
                calc_func,
                result_da,
                error_variance,
                terciles.isel(quantile=0).drop_vars('quantile'),
                terciles.isel(quantile=1).drop_vars('quantile'),
                input_core_dims=[('T',), (), (), ()],
                vectorize=True,
                kwargs={'dof': dof},
                dask='parallelized',
                output_core_dims=[('probability', 'T')],
                output_dtypes=['float'],
                dask_gufunc_kwargs={'output_sizes': {'probability': 3}, "allow_rechunk": True}
            )
        elif self.dist_method == "gamma":
            calc_func = self.calculate_tercile_probabilities_gamma
            hindcast_prob = xr.apply_ufunc(
                calc_func,
                result_da,
                error_variance,
                terciles.isel(quantile=0).drop_vars('quantile'),
                terciles.isel(quantile=1).drop_vars('quantile'),
                input_core_dims=[('T',), (), (), ()],
                vectorize=True,
                dask='parallelized',
                output_core_dims=[('probability', 'T')],
                output_dtypes=['float'],
                dask_gufunc_kwargs={'output_sizes': {'probability': 3}, "allow_rechunk": True}
            )
        elif self.dist_method == "normal":
            calc_func = self.calculate_tercile_probabilities_normal
            hindcast_prob = xr.apply_ufunc(
                calc_func,
                result_da,
                error_variance,
                terciles.isel(quantile=0).drop_vars('quantile'),
                terciles.isel(quantile=1).drop_vars('quantile'),
                input_core_dims=[('T',), (), (), ()],
                vectorize=True,
                dask='parallelized',
                output_core_dims=[('probability', 'T')],
                output_dtypes=['float'],
                dask_gufunc_kwargs={'output_sizes': {'probability': 3}, "allow_rechunk": True}
            )
        elif self.dist_method == "lognormal":
            calc_func = self.calculate_tercile_probabilities_lognormal
            hindcast_prob = xr.apply_ufunc(
                calc_func,
                result_da,
                error_variance,
                terciles.isel(quantile=0).drop_vars('quantile'),
                terciles.isel(quantile=1).drop_vars('quantile'),
                input_core_dims=[('T',), (), (), ()],
                vectorize=True,
                dask='parallelized',
                output_core_dims=[('probability', 'T')],
                output_dtypes=['float'],
                dask_gufunc_kwargs={'output_sizes': {'probability': 3}, "allow_rechunk": True}
            )
        elif self.dist_method == "nonparam":
            calc_func = self.calculate_tercile_probabilities_nonparametric
            error_samples = Predictant_no_m - hindcast_det_cross
            error_samples = error_samples.rename({'T': 'S'})
            hindcast_prob = xr.apply_ufunc(
                calc_func,
                result_da,
                error_samples,
                terciles.isel(quantile=0).drop_vars('quantile'),
                terciles.isel(quantile=1).drop_vars('quantile'),
                input_core_dims=[('T',), ('S',), (), ()],
                output_core_dims=[('probability', 'T')],
                vectorize=True,
                dask='parallelized',
                output_dtypes=[float],
                dask_gufunc_kwargs={'output_sizes': {'probability': 3}, "allow_rechunk": True}
            )
        else:
            raise ValueError(f"Invalid dist_method: {self.dist_method}.")
        hindcast_prob = hindcast_prob.assign_coords(probability=('probability', ['PB', 'PN', 'PA']))
        return result_da * mask, mask * hindcast_prob.transpose('probability', 'T', 'Y', 'X')          



class WAS_mme_XGBoosting_:
    """
    XGBoost-based Multi-Model Ensemble (MME) forecasting.
    This class implements a single-model forecasting approach using XGBoost's XGBRegressor
    for deterministic predictions, with optional tercile probability calculations using
    various statistical distributions. Implements hyperparameter optimization via randomized search.

    Parameters
    ----------
    n_estimators_range : list of int, optional
        List of n_estimators values to tune (default is [50, 100, 200, 300]).
    learning_rate_range : list of float, optional
        List of learning rates to tune (default is [0.01, 0.05, 0.1, 0.2]).
    max_depth_range : list of int, optional
        List of max depths to tune (default is [3, 5, 7, 9]).
    min_child_weight_range : list of float, optional
        List of minimum child weights to tune (default is [1, 3, 5]).
    subsample_range : list of float, optional
        List of subsample ratios to tune (default is [0.6, 0.8, 1.0]).
    colsample_bytree_range : list of float, optional
        List of column sampling ratios to tune (default is [0.6, 0.8, 1.0]).
    random_state : int, optional
        Seed for reproducibility (default is 42).
    dist_method : str, optional
        Distribution method for tercile probabilities ('t', 'gamma', etc.) (default is 'gamma').
    n_iter_search : int, optional
        Number of iterations for randomized search (default is 10).
    cv_folds : int, optional
        Number of cross-validation folds (default is 3).
    """
    def __init__(self,
                 n_estimators_range=[50, 100, 200, 300],
                 learning_rate_range=[0.01, 0.05, 0.1, 0.2],
                 max_depth_range=[3, 5, 7, 9],
                 min_child_weight_range=[1, 3, 5],
                 subsample_range=[0.6, 0.8, 1.0],
                 colsample_bytree_range=[0.6, 0.8, 1.0],
                 random_state=42,
                 dist_method="gamma",
                 n_iter_search=10,
                 cv_folds=3):
        self.n_estimators_range = n_estimators_range
        self.learning_rate_range = learning_rate_range
        self.max_depth_range = max_depth_range
        self.min_child_weight_range = min_child_weight_range
        self.subsample_range = subsample_range
        self.colsample_bytree_range = colsample_bytree_range
        self.random_state = random_state
        self.dist_method = dist_method
        self.n_iter_search = n_iter_search
        self.cv_folds = cv_folds
        self.xgb = None

    def compute_hyperparameters(self, Predictors, Predictand, clim_year_start, clim_year_end):
        """
        Independently computes the best hyperparameters using randomized search on stacked training data.

        Parameters
        ----------
        Predictors : xarray.DataArray
            Training predictor data with dimensions (T, M, Y, X).
        Predictand : xarray.DataArray
            Training predictand data with dimensions (T, Y, X).
        clim_year_start : int
            Start year of the climatology period.
        clim_year_end : int
            End year of the climatology period.

        Returns
        -------
        dict
            Best hyperparameters found.
        """
        X_train = standardize_timeseries(Predictors, clim_year_start, clim_year_end)
        y_train = standardize_timeseries(Predictand, clim_year_start, clim_year_end)

        # Stack training data
        X_train_stacked = X_train.stack(sample=('T', 'Y', 'X')).transpose('sample', 'M').values
        y_train_stacked = y_train.stack(sample=('T', 'Y', 'X')).values.ravel()
        train_nan_mask = np.any(~np.isfinite(X_train_stacked), axis=1) | ~np.isfinite(y_train_stacked)
        X_train_clean = X_train_stacked[~train_nan_mask]
        y_train_clean = y_train_stacked[~train_nan_mask]

        # Prepare parameter distributions
        param_dist = {
            'n_estimators': self.n_estimators_range,
            'learning_rate': self.learning_rate_range,
            'max_depth': self.max_depth_range,
            'min_child_weight': self.min_child_weight_range,
            'subsample': self.subsample_range,
            'colsample_bytree': self.colsample_bytree_range
        }

        # Initialize XGBRegressor base model
        model = XGBRegressor(random_state=self.random_state, verbosity=0)

        # Randomized search
        random_search = RandomizedSearchCV(
            model, param_distributions=param_dist, n_iter=self.n_iter_search,
            cv=self.cv_folds, scoring='neg_mean_squared_error',
            random_state=self.random_state, error_score=np.nan
        )

        random_search.fit(X_train_clean, y_train_clean)
        best_params = random_search.best_params_

        return best_params

    def compute_model(self, X_train, y_train, X_test, y_test, best_params=None):
        """
        Compute deterministic hindcast using the XGBRegressor model with injected hyperparameters.

        Parameters
        ----------
        X_train : xarray.DataArray
            Training predictor data with dimensions (T, M, Y, X).
        y_train : xarray.DataArray
            Training predictand data with dimensions (T, Y, X).
        X_test : xarray.DataArray
            Testing predictor data with dimensions (T, M, Y, X).
        y_test : xarray.DataArray
            Testing predictand data with dimensions (T, Y, X).
        best_params : dict, optional
            Pre-computed best hyperparameters. If None, computes internally.

        Returns
        -------
        predicted_da : xarray.DataArray
            Deterministic hindcast with dimensions (T, Y, X).
        """
        # Extract coordinate variables from X_test
        time = X_test['T']
        lat = X_test['Y']
        lon = X_test['X']
        n_time = len(X_test.coords['T'])
        n_lat = len(X_test.coords['Y'])
        n_lon = len(X_test.coords['X'])

        # Stack training data
        X_train_stacked = X_train.stack(sample=('T', 'Y', 'X')).transpose('sample', 'M').values
        y_train_stacked = y_train.stack(sample=('T', 'Y', 'X')).transpose('sample', 'M').values
        train_nan_mask = (np.any(~np.isfinite(X_train_stacked), axis=1) | np.any(~np.isfinite(y_train_stacked), axis=1))
        X_train_clean = X_train_stacked[~train_nan_mask]
        y_train_clean = y_train_stacked[~train_nan_mask]

        # Stack testing data
        X_test_stacked = X_test.stack(sample=('T', 'Y', 'X')).transpose('sample', 'M').values
        y_test_stacked = y_test.stack(sample=('T', 'Y', 'X')).transpose('sample', 'M').values
        test_nan_mask = (np.any(~np.isfinite(X_test_stacked), axis=1) | np.any(~np.isfinite(y_test_stacked), axis=1))

        # Use provided best_params or compute if None
        if best_params is None:
            best_params = self.compute_hyperparameters(X_train, y_train, clim_year_start, clim_year_end)

        # Initialize the model with best parameters
        self.xgb = XGBRegressor(
            n_estimators=best_params['n_estimators'],
            learning_rate=best_params['learning_rate'],
            max_depth=best_params['max_depth'],
            min_child_weight=best_params['min_child_weight'],
            subsample=best_params['subsample'],
            colsample_bytree=best_params['colsample_bytree'],
            random_state=self.random_state,
            verbosity=0
        )

        # Fit the model and predict on non-NaN testing data
        self.xgb.fit(X_train_clean, y_train_clean)
        y_pred = self.xgb.predict(X_test_stacked[~test_nan_mask])

        # Reconstruct predictions
        result = np.empty_like(np.squeeze(y_test_stacked))
        result[np.squeeze(test_nan_mask)] = np.squeeze(y_test_stacked[test_nan_mask])
        result[~np.squeeze(test_nan_mask)] = y_pred

        predictions_reshaped = result.reshape(n_time, n_lat, n_lon)
        predicted_da = xr.DataArray(data=predictions_reshaped,
                                    coords={'T': time, 'Y': lat, 'X': lon},
                                    dims=['T', 'Y', 'X'])
        return predicted_da

    # ------------------ Probability Calculation Methods ------------------
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
    def calculate_tercile_probabilities_weibull_min(best_guess, error_variance, first_tercile, second_tercile, dof):
        """
        Weibull minimum-based method.
        """
        n_time = len(best_guess)
        pred_prob = np.empty((3, n_time))

        if np.all(np.isnan(best_guess)):
            pred_prob[:] = np.nan
        else:
            error_std = np.sqrt(error_variance)
            pred_prob[0, :] = stats.weibull_min.cdf(first_tercile, c=dof, loc=best_guess, scale=error_std)
            pred_prob[1, :] = stats.weibull_min.cdf(second_tercile, c=dof, loc=best_guess, scale=error_std) - \
                              stats.weibull_min.cdf(first_tercile, c=dof, loc=best_guess, scale=error_std)
            pred_prob[2, :] = 1 - stats.weibull_min.cdf(second_tercile, c=dof, loc=best_guess, scale=error_std)

        return pred_prob

    @staticmethod
    def calculate_tercile_probabilities_gamma(best_guess, error_variance, T1, T2, dof=None):
        """Gamma-distribution based method."""
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

    @staticmethod
    def calculate_tercile_probabilities_normal(best_guess, error_variance, first_tercile, second_tercile):
        """Normal-distribution based method."""
        n_time = len(best_guess)
        pred_prob = np.empty((3, n_time))
        if np.all(np.isnan(best_guess)):
            pred_prob[:] = np.nan
        else:
            error_std = np.sqrt(error_variance)
            pred_prob[0, :] = norm.cdf(first_tercile, loc=best_guess, scale=error_std)
            pred_prob[1, :] = norm.cdf(second_tercile, loc=best_guess, scale=error_std) - norm.cdf(first_tercile, loc=best_guess, scale=error_std)
            pred_prob[2, :] = 1 - norm.cdf(second_tercile, loc=best_guess, scale=error_std)
        return pred_prob

    @staticmethod
    def calculate_tercile_probabilities_lognormal(best_guess, error_variance, first_tercile, second_tercile):
        """Lognormal-distribution based method."""
        n_time = len(best_guess)
        pred_prob = np.empty((3, n_time))
        if np.any(np.isnan(best_guess)) or np.any(np.isnan(error_variance)):
            pred_prob[:] = np.nan
            return pred_prob
        sigma = np.sqrt(np.log(1 + error_variance / (best_guess**2)))
        mu = np.log(best_guess) - sigma**2 / 2
        pred_prob[0, :] = lognorm.cdf(first_tercile, s=sigma, scale=np.exp(mu))
        pred_prob[1, :] = lognorm.cdf(second_tercile, s=sigma, scale=np.exp(mu)) - lognorm.cdf(first_tercile, s=sigma, scale=np.exp(mu))
        pred_prob[2, :] = 1 - lognorm.cdf(second_tercile, s=sigma, scale=np.exp(mu))
        return pred_prob

    def compute_prob(self, Predictant, clim_year_start, clim_year_end, hindcast_det):
        """
        Compute tercile probabilities using the chosen distribution method.
        Predictant is expected to be an xarray DataArray with dims (T, Y, X).
        """
        if "M" in Predictant.coords:
            Predictant = Predictant.isel(M=0).drop_vars('M').squeeze()
        mask = xr.where(~np.isnan(Predictant.isel(T=0)), 1, np.nan).drop_vars(['T']).squeeze().to_numpy()

        Predictant = Predictant.transpose('T', 'Y', 'X')
        index_start = Predictant.get_index("T").get_loc(str(clim_year_start)).start
        index_end = Predictant.get_index("T").get_loc(str(clim_year_end)).stop
        rainfall_for_tercile = Predictant.isel(T=slice(index_start, index_end))
        terciles = rainfall_for_tercile.quantile([0.33, 0.67], dim='T')
        error_variance = (Predictant - hindcast_det).var(dim='T')
        dof = len(Predictant.get_index("T")) - 2

        if self.dist_method == "t":
            calc_func = self.calculate_tercile_probabilities
            hindcast_prob = xr.apply_ufunc(
                calc_func,
                hindcast_det,
                error_variance,
                terciles.isel(quantile=0).drop_vars('quantile'),
                terciles.isel(quantile=1).drop_vars('quantile'),
                input_core_dims=[('T',), (), (), ()],
                vectorize=True,
                kwargs={'dof': dof},
                dask='parallelized',
                output_core_dims=[('probability','T')],
                output_dtypes=['float'],
                dask_gufunc_kwargs={'output_sizes': {'probability': 3}, "allow_rechunk": True}
            )
        elif self.dist_method == "weibull_min":
            calc_func = self.calculate_tercile_probabilities_weibull_min
            hindcast_prob = xr.apply_ufunc(
                calc_func,
                hindcast_det,
                error_variance,
                terciles.isel(quantile=0).drop_vars('quantile'),
                terciles.isel(quantile=1).drop_vars('quantile'),
                input_core_dims=[('T',), (), (), ()],
                vectorize=True,
                kwargs={'dof': dof},
                dask='parallelized',
                output_core_dims=[('probability', 'T')],
                output_dtypes=['float'],
                dask_gufunc_kwargs={'output_sizes': {'probability': 3}, "allow_rechunk": True}
            )
        elif self.dist_method == "gamma":
            calc_func = self.calculate_tercile_probabilities_gamma
            hindcast_prob = xr.apply_ufunc(
                calc_func,
                hindcast_det,
                error_variance,
                terciles.isel(quantile=0).drop_vars('quantile'),
                terciles.isel(quantile=1).drop_vars('quantile'),
                input_core_dims=[('T',), (), (), ()],
                vectorize=True,
                dask='parallelized',
                output_core_dims=[('probability','T')],
                output_dtypes=['float'],
                dask_gufunc_kwargs={'output_sizes': {'probability': 3}, "allow_rechunk": True}
            )
        elif self.dist_method == "nonparam":
            calc_func = self.calculate_tercile_probabilities_nonparametric
            error_samples = (Predictant - hindcast_det)
            hindcast_prob = xr.apply_ufunc(
                calc_func,
                hindcast_det,
                error_samples,
                terciles.isel(quantile=0).drop_vars('quantile'),
                terciles.isel(quantile=1).drop_vars('quantile'),
                input_core_dims=[('T',), ('T',), (), ()],
                output_core_dims=[('probability','T')],
                vectorize=True,
                dask='parallelized',
                output_dtypes=[float],
                dask_gufunc_kwargs={'output_sizes': {'probability': 3}, "allow_rechunk": True}
            )
        elif self.dist_method == "normal":
            calc_func = self.calculate_tercile_probabilities_normal
            hindcast_prob = xr.apply_ufunc(
                calc_func,
                hindcast_det,
                error_variance,
                terciles.isel(quantile=0).drop_vars('quantile'),
                terciles.isel(quantile=1).drop_vars('quantile'),
                input_core_dims=[('T',), (), (), ()],
                vectorize=True,
                dask='parallelized',
                output_core_dims=[('probability','T')],
                output_dtypes=['float'],
                dask_gufunc_kwargs={'output_sizes': {'probability': 3}, "allow_rechunk": True}
            )
        elif self.dist_method == "lognormal":
            calc_func = self.calculate_tercile_probabilities_lognormal
            hindcast_prob = xr.apply_ufunc(
                calc_func,
                hindcast_det,
                error_variance,
                terciles.isel(quantile=0).drop_vars('quantile'),
                terciles.isel(quantile=1).drop_vars('quantile'),
                input_core_dims=[('T',), (), (), ()],
                vectorize=True,
                dask='parallelized',
                output_core_dims=[('probability','T')],
                output_dtypes=['float'],
                dask_gufunc_kwargs={'output_sizes': {'probability': 3}, "allow_rechunk": True}
            )
        else:
            raise ValueError(f"Invalid dist_method: {self.dist_method}.")

        hindcast_prob = hindcast_prob.assign_coords(probability=('probability', ['PB', 'PN', 'PA']))
        return hindcast_prob.transpose('probability', 'T', 'Y', 'X')

    @staticmethod
    def _reshape_and_filter_data(da):
        """
        Helper: stack the DataArray from (T, Y, X[, M]) to (n_samples, n_features)
        and remove rows containing NaNs.
        """
        da_stacked = da.stack(sample=('T', 'Y', 'X'))
        if 'M' in da.dims:
            da_stacked = da_stacked.transpose('sample', 'M')
        else:
            da_stacked = da_stacked.transpose('sample')
        da_values = da_stacked.values
        nan_mask = np.any(np.isnan(da_values), axis=1)
        return da_values[~nan_mask], nan_mask, da_values

    def forecast(self, Predictant, clim_year_start, clim_year_end, hindcast_det,
                 hindcast_det_cross, Predictor_for_year, best_params=None):
        """
        Forecast method using a single XGBoost model with optimized hyperparameters.

        Parameters
        ----------
        Predictant : xarray.DataArray
            Observed predictand data with dimensions (T, Y, X) or (T, M, Y, X).
        clim_year_start : int or str
            Start year of the climatology period.
        clim_year_end : int or str
            End year of the climatology period.
        hindcast_det : xarray.DataArray
            Deterministic hindcast data for training with dimensions (T, M, Y, X).
        hindcast_det_cross : xarray.DataArray
            Deterministic hindcast data for error estimation with dimensions (T, Y, X).
        Predictor_for_year : xarray.DataArray
            Predictor data for the target year with dimensions (T, M, Y, X).
        best_params : dict, optional
            Pre-computed best hyperparameters. If None, computes internally.

        Returns
        -------
        forecast_det : xarray.DataArray
            Deterministic forecast with dimensions (T, Y, X).
        forecast_prob : xarray.DataArray
            Tercile probabilities with dimensions (probability, T, Y, X).
        """
        if "M" in Predictant.coords:
            Predictant_no_m = Predictant.isel(M=0).drop_vars('M').squeeze()
        else:
            Predictant_no_m = Predictant

        mask = xr.where(~np.isnan(Predictant_no_m.isel(T=0)), 1, np.nan).drop_vars(['T']).squeeze().to_numpy()

        # Standardize Predictor_for_year using hindcast climatology
        mean_val = hindcast_det.sel(T=slice(str(clim_year_start), str(clim_year_end))).mean(dim='T')
        std_val = hindcast_det.sel(T=slice(str(clim_year_start), str(clim_year_end))).std(dim='T')
        Predictor_for_year_st = (Predictor_for_year - mean_val) / std_val

        hindcast_det_st = standardize_timeseries(hindcast_det, clim_year_start, clim_year_end)
        Predictant_st = standardize_timeseries(Predictant_no_m, clim_year_start, clim_year_end)
        y_test = Predictant_st.isel(T=[-1])

        # Extract coordinates from X_test
        time = Predictor_for_year_st['T']
        lat = Predictor_for_year_st['Y']
        lon = Predictor_for_year_st['X']
        n_time = len(Predictor_for_year_st.coords['T'])
        n_lat = len(Predictor_for_year_st.coords['Y'])
        n_lon = len(Predictor_for_year_st.coords['X'])

        # Stack training data and remove rows with NaNs
        X_train_stacked = hindcast_det_st.stack(sample=('T', 'Y', 'X')).transpose('sample', 'M').values
        y_train_stacked = Predictant_st.stack(sample=('T', 'Y', 'X')).values.ravel()
        train_nan_mask = np.any(~np.isfinite(X_train_stacked), axis=1) | ~np.isfinite(y_train_stacked)
        X_train_clean = X_train_stacked[~train_nan_mask]
        y_train_clean = y_train_stacked[~train_nan_mask]

        # Stack testing data
        X_test_stacked = Predictor_for_year_st.stack(sample=('T', 'Y', 'X')).transpose('sample', 'M').values
        y_test_stacked = y_test.stack(sample=('T', 'Y', 'X')).values.ravel()
        test_nan_mask = np.any(~np.isfinite(X_test_stacked), axis=1) | ~np.isfinite(y_test_stacked)

        # Use provided best_params or compute if None
        if best_params is None:
            best_params = self.compute_hyperparameters(hindcast_det, Predictant_no_m, clim_year_start, clim_year_end)

        # Initialize and fit the model with best parameters
        self.xgb = XGBRegressor(
            n_estimators=best_params['n_estimators'],
            learning_rate=best_params['learning_rate'],
            max_depth=best_params['max_depth'],
            min_child_weight=best_params['min_child_weight'],
            subsample=best_params['subsample'],
            colsample_bytree=best_params['colsample_bytree'],
            random_state=self.random_state,
            verbosity=0
        )
        self.xgb.fit(X_train_clean, y_train_clean)
        y_pred = self.xgb.predict(X_test_stacked[~test_nan_mask])

        # Reconstruct the prediction array
        result = np.empty_like(np.squeeze(y_test_stacked))
        result[np.squeeze(test_nan_mask)] = np.squeeze(y_test_stacked[test_nan_mask])
        result[~np.squeeze(test_nan_mask)] = y_pred

        predictions_reshaped = result.reshape(n_time, n_lat, n_lon)
        result_da = xr.DataArray(data=predictions_reshaped,
                                 coords={'T': time, 'Y': lat, 'X': lon},
                                 dims=['T', 'Y', 'X']) * mask

        result_da = reverse_standardize(result_da, Predictant_no_m, clim_year_start, clim_year_end)
        year = Predictor_for_year.coords['T'].values.astype('datetime64[Y]').astype(int)[0] + 1970
        T_value_1 = Predictant_no_m.isel(T=0).coords['T'].values
        month_1 = T_value_1.astype('datetime64[M]').astype(int) % 12 + 1
        new_T_value = np.datetime64(f"{year}-{month_1:02d}-{1:02d}")
        result_da = result_da.assign_coords(T=xr.DataArray([new_T_value], dims=["T"]))
        result_da['T'] = result_da['T'].astype('datetime64[ns]')

        # Compute tercile probabilities
        index_start = Predictant_no_m.get_index("T").get_loc(str(clim_year_start)).start
        index_end = Predictant_no_m.get_index("T").get_loc(str(clim_year_end)).stop
        rainfall_for_tercile = Predictant_no_m.isel(T=slice(index_start, index_end))
        terciles = rainfall_for_tercile.quantile([0.33, 0.67], dim='T')
        error_variance = (Predictant_no_m - hindcast_det_cross).var(dim='T')
        dof = len(Predictant_no_m.get_index("T")) - 2

        if self.dist_method == "t":
            calc_func = self.calculate_tercile_probabilities
            hindcast_prob = xr.apply_ufunc(
                calc_func,
                result_da,
                error_variance,
                terciles.isel(quantile=0).drop_vars('quantile'),
                terciles.isel(quantile=1).drop_vars('quantile'),
                input_core_dims=[('T',), (), (), ()],
                vectorize=True,
                kwargs={'dof': dof},
                dask='parallelized',
                output_core_dims=[('probability','T')],
                output_dtypes=['float'],
                dask_gufunc_kwargs={'output_sizes': {'probability': 3}, "allow_rechunk": True}
            )
        elif self.dist_method == "weibull_min":
            calc_func = self.calculate_tercile_probabilities_weibull_min
            hindcast_prob = xr.apply_ufunc(
                calc_func,
                result_da,
                error_variance,
                terciles.isel(quantile=0).drop_vars('quantile'),
                terciles.isel(quantile=1).drop_vars('quantile'),
                input_core_dims=[('T',), (), (), ()],
                vectorize=True,
                kwargs={'dof': dof},
                dask='parallelized',
                output_core_dims=[('probability', 'T')],
                output_dtypes=['float'],
                dask_gufunc_kwargs={'output_sizes': {'probability': 3}, "allow_rechunk": True}
            )
        elif self.dist_method == "gamma":
            calc_func = self.calculate_tercile_probabilities_gamma
            hindcast_prob = xr.apply_ufunc(
                calc_func,
                result_da,
                error_variance,
                terciles.isel(quantile=0).drop_vars('quantile'),
                terciles.isel(quantile=1).drop_vars('quantile'),
                input_core_dims=[('T',), (), (), ()],
                vectorize=True,
                dask='parallelized',
                output_core_dims=[('probability','T')],
                output_dtypes=['float'],
                dask_gufunc_kwargs={'output_sizes': {'probability': 3}, "allow_rechunk": True}
            )
        elif self.dist_method == "normal":
            calc_func = self.calculate_tercile_probabilities_normal
            hindcast_prob = xr.apply_ufunc(
                calc_func,
                result_da,
                error_variance,
                terciles.isel(quantile=0).drop_vars('quantile'),
                terciles.isel(quantile=1).drop_vars('quantile'),
                input_core_dims=[('T',), (), (), ()],
                vectorize=True,
                dask='parallelized',
                output_core_dims=[('probability','T')],
                output_dtypes=['float'],
                dask_gufunc_kwargs={'output_sizes': {'probability': 3}, "allow_rechunk": True}
            )
        elif self.dist_method == "lognormal":
            calc_func = self.calculate_tercile_probabilities_lognormal
            hindcast_prob = xr.apply_ufunc(
                calc_func,
                result_da,
                error_variance,
                terciles.isel(quantile=0).drop_vars('quantile'),
                terciles.isel(quantile=1).drop_vars('quantile'),
                input_core_dims=[('T',), (), (), ()],
                vectorize=True,
                dask='parallelized',
                output_core_dims=[('probability','T')],
                output_dtypes=['float'],
                dask_gufunc_kwargs={'output_sizes': {'probability': 3}, "allow_rechunk": True}
            )
        elif self.dist_method == "nonparam":
            calc_func = self.calculate_tercile_probabilities_nonparametric
            error_samples = Predictant_no_m - hindcast_det_cross
            error_samples = error_samples.rename({'T': 'S'})
            hindcast_prob = xr.apply_ufunc(
                calc_func,
                result_da,
                error_samples,
                terciles.isel(quantile=0).drop_vars('quantile'),
                terciles.isel(quantile=1).drop_vars('quantile'),
                input_core_dims=[('T',), ('S',), (), ()],
                output_core_dims=[('probability','T')],
                vectorize=True,
                dask='parallelized',
                output_dtypes=[float],
                dask_gufunc_kwargs={'output_sizes': {'probability': 3}, "allow_rechunk": True}
            )
        else:
            raise ValueError(f"Invalid dist_method: {self.dist_method}.")

        hindcast_prob = hindcast_prob.assign_coords(probability=('probability', ['PB', 'PN', 'PA']))
        return result_da * mask, mask * hindcast_prob.transpose('probability', 'T', 'Y', 'X')


class WAS_mme_XGBoosting:
    """
    XGBoost-based Multi-Model Ensemble (MME) forecasting.
    This class implements a single-model forecasting approach using XGBoost's XGBRegressor
    for deterministic predictions, with optional tercile probability calculations using
    various statistical distributions. Implements hyperparameter optimization via randomized search.
    Parameters
    ----------
    n_estimators_range : list of int, optional
        List of n_estimators values to tune (default is [50, 100, 200, 300]).
    learning_rate_range : list of float, optional
        List of learning rates to tune (default is [0.01, 0.05, 0.1, 0.2]).
    max_depth_range : list of int, optional
        List of max depths to tune (default is [3, 5, 7, 9]).
    min_child_weight_range : list of float, optional
        List of minimum child weights to tune (default is [1, 3, 5]).
    subsample_range : list of float, optional
        List of subsample ratios to tune (default is [0.6, 0.8, 1.0]).
    colsample_bytree_range : list of float, optional
        List of column sampling ratios to tune (default is [0.6, 0.8, 1.0]).
    random_state : int, optional
        Seed for reproducibility (default is 42).
    dist_method : str, optional
        Distribution method for tercile probabilities ('t', 'gamma', etc.) (default is 'gamma').
    n_iter_search : int, optional
        Number of iterations for randomized search (default is 10).
    cv_folds : int, optional
        Number of cross-validation folds (default is 3).
    n_clusters : int, optional
        Number of clusters for homogenized zones (default is 4).
    """
    def __init__(self,
                 n_estimators_range=[50, 100, 200, 300],
                 learning_rate_range=[0.01, 0.05, 0.1, 0.2],
                 max_depth_range=[3, 5, 7, 9],
                 min_child_weight_range=[1, 3, 5],
                 subsample_range=[0.6, 0.8, 1.0],
                 colsample_bytree_range=[0.6, 0.8, 1.0],
                 random_state=42,
                 dist_method="gamma",
                 n_iter_search=10,
                 cv_folds=3,
                 n_clusters=4):
        self.n_estimators_range = n_estimators_range
        self.learning_rate_range = learning_rate_range
        self.max_depth_range = max_depth_range
        self.min_child_weight_range = min_child_weight_range
        self.subsample_range = subsample_range
        self.colsample_bytree_range = colsample_bytree_range
        self.random_state = random_state
        self.dist_method = dist_method
        self.n_iter_search = n_iter_search
        self.cv_folds = cv_folds
        self.n_clusters = n_clusters
        self.xgb = None

    def compute_hyperparameters(self, Predictors, Predictand, clim_year_start, clim_year_end):
        """
        Independently computes the best hyperparameters using randomized search on stacked training data for each homogenized zone.
        Parameters
        ----------
        Predictors : xarray.DataArray
            Training predictor data with dimensions (T, M, Y, X).
        Predictand : xarray.DataArray
            Training predictand data with dimensions (T, Y, X).
        clim_year_start : int
            Start year of the climatology period.
        clim_year_end : int
            End year of the climatology period.
        Returns
        -------
        best_params_dict : dict
            Best hyperparameters for each cluster.
        cluster_da : xarray.DataArray
            Cluster labels with dimensions (Y, X).
        """
        if "M" in Predictand.coords:
            Predictand = Predictand.isel(M=0).drop_vars('M').squeeze()
           
        X_train_std = standardize_timeseries(Predictors, clim_year_start, clim_year_end)
        Predictand.name = "varname"
        # Step 1: Perform KMeans clustering based on predictand's spatial distribution
        kmeans = KMeans(n_clusters=self.n_clusters, random_state=self.random_state)
        Predictand_dropna = Predictand.to_dataframe().reset_index().dropna().drop(columns=['T'])
        variable_column = Predictand_dropna.columns[2]
        Predictand_dropna['cluster'] = kmeans.fit_predict(
           Predictand_dropna[[variable_column]]
        )
       
        # Convert cluster assignments back into an xarray structure
        df_unique = Predictand_dropna.drop_duplicates(subset=['Y', 'X'])
        dataset = df_unique.set_index(['Y', 'X']).to_xarray()
        mask = xr.where(~np.isnan(Predictand.isel(T=0)), 1, np.nan)
        Cluster = (dataset['cluster'] * mask)
              
        # Align cluster array with the predictand array
        xarray1, xarray2 = xr.align(Predictand, Cluster, join="outer")
       
        # Identify unique cluster labels
        clusters = np.unique(xarray2)
        clusters = clusters[~np.isnan(clusters)]
        cluster_da = xarray2
        y_train_std = standardize_timeseries(Predictand, clim_year_start, clim_year_end)
        X_train_std['T'] = y_train_std['T']
        # Prepare parameter distributions
        param_dist = {
            'n_estimators': self.n_estimators_range,
            'learning_rate': self.learning_rate_range,
            'max_depth': self.max_depth_range,
            'min_child_weight': self.min_child_weight_range,
            'subsample': self.subsample_range,
            'colsample_bytree': self.colsample_bytree_range
        }
        best_params_dict = {}
        for c in clusters:
            mask_3d = (cluster_da == c).expand_dims({'T': y_train_std['T']})
            X_stacked_c = X_train_std.where(mask_3d).stack(sample=('T', 'Y', 'X')).transpose('sample', 'M').values
            y_stacked_c = y_train_std.where(mask_3d).stack(sample=('T', 'Y', 'X')).values.ravel()
            nan_mask_c = np.any(~np.isfinite(X_stacked_c), axis=1) | ~np.isfinite(y_stacked_c)
            X_clean_c = X_stacked_c[~nan_mask_c]
            y_clean_c = y_stacked_c[~nan_mask_c]
            if len(X_clean_c) == 0:
                continue
            # Initialize XGBRegressor base model
            model = XGBRegressor(random_state=self.random_state, verbosity=0, n_jobs=-1)
            # Randomized search
            random_search = RandomizedSearchCV(
                model, param_distributions=param_dist, n_iter=self.n_iter_search,
                cv=self.cv_folds, scoring='neg_mean_squared_error',
                random_state=self.random_state, error_score=np.nan
            )
            random_search.fit(X_clean_c, y_clean_c)
            best_params_dict[c] = random_search.best_params_
        return best_params_dict, cluster_da

    def compute_model(self, X_train, y_train, X_test, y_test, best_params=None, cluster_da=None):
        """
        Compute deterministic hindcast using the XGBRegressor model with injected hyperparameters for each zone.
        Parameters
        ----------
        X_train : xarray.DataArray
            Training predictor data with dimensions (T, M, Y, X).
        y_train : xarray.DataArray
            Training predictand data with dimensions (T, Y, X).
        X_test : xarray.DataArray
            Testing predictor data with dimensions (T, M, Y, X).
        y_test : xarray.DataArray
            Testing predictand data with dimensions (T, Y, X).
        clim_year_start : int
            Start year of the climatology period.
        clim_year_end : int
            End year of the climatology period.
        best_params : dict, optional
            Pre-computed best hyperparameters per cluster. If None, computes internally.
        cluster_da : xarray.DataArray, optional
            Pre-computed cluster labels. If None, computes internally.
        Returns
        -------
        predicted_da : xarray.DataArray
            Deterministic hindcast with dimensions (T, Y, X).
        """
        # Standardize inputs
        X_train_std = X_train
        y_train_std = y_train
        X_test_std = X_test
        y_test_std = y_test
        # Extract coordinate variables from X_test
        time = X_test_std['T']
        lat = X_test_std['Y']
        lon = X_test_std['X']
        n_time = len(time)
        n_lat = len(lat)
        n_lon = len(lon)
        # Use provided best_params and cluster_da or compute if None
        if best_params is None:
            best_params, cluster_da = self.compute_hyperparameters(X_train, y_train, clim_year_start, clim_year_end)
        # Initialize predictions array
        predictions = np.full((n_time, n_lat, n_lon), np.nan)
        self.xgb = {}  # Dictionary to store models per cluster
        for c in range(self.n_clusters):
            if c not in best_params:
                continue
            bp = best_params[c]
            # Mask for this cluster
            mask_3d_train = (cluster_da == c).expand_dims({'T': X_train_std['T']})
            mask_3d_test = (cluster_da == c).expand_dims({'T': X_test_std['T']})
            # Stack training data for cluster
            X_train_stacked_c = X_train_std.where(mask_3d_train).stack(sample=('T', 'Y', 'X')).transpose('sample', 'M').values
            y_train_stacked_c = y_train_std.where(mask_3d_train).stack(sample=('T', 'Y', 'X')).values.ravel()
            train_nan_mask = np.any(~np.isfinite(X_train_stacked_c), axis=1) | ~np.isfinite(y_train_stacked_c)
            X_train_clean_c = X_train_stacked_c[~train_nan_mask]
            y_train_clean_c = y_train_stacked_c[~train_nan_mask]
            # Stack testing data for cluster
            X_test_stacked_c = X_test_std.where(mask_3d_test).stack(sample=('T', 'Y', 'X')).transpose('sample', 'M').values
            y_test_stacked_c = y_test_std.where(mask_3d_test).stack(sample=('T', 'Y', 'X')).values.ravel()
            test_nan_mask = np.any(~np.isfinite(X_test_stacked_c), axis=1) | ~np.isfinite(y_test_stacked_c)
            X_test_clean_c = X_test_stacked_c[~test_nan_mask]
            # Initialize the model with best parameters for this cluster
            xgb_c = XGBRegressor(
                n_estimators=bp['n_estimators'],
                learning_rate=bp['learning_rate'],
                max_depth=bp['max_depth'],
                min_child_weight=bp['min_child_weight'],
                subsample=bp['subsample'],
                colsample_bytree=bp['colsample_bytree'],
                random_state=self.random_state,
                verbosity=0
            )
            # Fit and predict
            xgb_c.fit(X_train_clean_c, y_train_clean_c)
            self.xgb[c] = xgb_c
            y_pred_c = xgb_c.predict(X_test_clean_c)
            # Reconstruct predictions for this cluster
            result_c = np.full(len(y_test_stacked_c), np.nan)
            result_c[~test_nan_mask] = y_pred_c
            pred_c_reshaped = result_c.reshape(n_time, n_lat, n_lon)
            # Fill in the predictions array
            predictions = np.where(np.isnan(predictions), pred_c_reshaped, predictions)
        predicted_da = xr.DataArray(
            data=predictions,
            coords={'T': time, 'Y': lat, 'X': lon},
            dims=['T', 'Y', 'X']
        )

        return predicted_da

    # ------------------ Probability Calculation Methods ------------------
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
            pred_prob[0, :] = t.cdf(first_t, df=dof)
            pred_prob[1, :] = t.cdf(second_t, df=dof) - t.cdf(first_t, df=dof)
            pred_prob[2, :] = 1 - t.cdf(second_t, df=dof)
        return pred_prob

    @staticmethod
    def calculate_tercile_probabilities_weibull_min(best_guess, error_variance, first_tercile, second_tercile, dof):
        """
        Weibull minimum-based method.
        """
        n_time = len(best_guess)
        pred_prob = np.empty((3, n_time))
        if np.all(np.isnan(best_guess)):
            pred_prob[:] = np.nan
        else:
            error_std = np.sqrt(error_variance)
            pred_prob[0, :] = weibull_min.cdf(first_tercile, c=dof, loc=best_guess, scale=error_std)
            pred_prob[1, :] = weibull_min.cdf(second_tercile, c=dof, loc=best_guess, scale=error_std) - \
                              weibull_min.cdf(first_tercile, c=dof, loc=best_guess, scale=error_std)
            pred_prob[2, :] = 1 - weibull_min.cdf(second_tercile, c=dof, loc=best_guess, scale=error_std)
        return pred_prob

    @staticmethod
    def calculate_tercile_probabilities_gamma(best_guess, error_variance, T1, T2, dof=None):
        """Gamma-distribution based method."""
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

    @staticmethod
    def calculate_tercile_probabilities_normal(best_guess, error_variance, first_tercile, second_tercile):
        """Normal-distribution based method."""
        n_time = len(best_guess)
        pred_prob = np.empty((3, n_time))
        if np.all(np.isnan(best_guess)):
            pred_prob[:] = np.nan
        else:
            error_std = np.sqrt(error_variance)
            pred_prob[0, :] = norm.cdf(first_tercile, loc=best_guess, scale=error_std)
            pred_prob[1, :] = norm.cdf(second_tercile, loc=best_guess, scale=error_std) - norm.cdf(first_tercile, loc=best_guess, scale=error_std)
            pred_prob[2, :] = 1 - norm.cdf(second_tercile, loc=best_guess, scale=error_std)
        return pred_prob

    @staticmethod
    def calculate_tercile_probabilities_lognormal(best_guess, error_variance, first_tercile, second_tercile):
        """Lognormal-distribution based method."""
        n_time = len(best_guess)
        pred_prob = np.empty((3, n_time))
        if np.any(np.isnan(best_guess)) or np.any(np.isnan(error_variance)):
            pred_prob[:] = np.nan
            return pred_prob
        sigma = np.sqrt(np.log(1 + error_variance / (best_guess**2)))
        mu = np.log(best_guess) - sigma**2 / 2
        pred_prob[0, :] = lognorm.cdf(first_tercile, s=sigma, scale=np.exp(mu))
        pred_prob[1, :] = lognorm.cdf(second_tercile, s=sigma, scale=np.exp(mu)) - lognorm.cdf(first_tercile, s=sigma, scale=np.exp(mu))
        pred_prob[2, :] = 1 - lognorm.cdf(second_tercile, s=sigma, scale=np.exp(mu))
        return pred_prob

    def compute_prob(self, Predictant, clim_year_start, clim_year_end, hindcast_det):
        """
        Compute tercile probabilities using the chosen distribution method.
        Predictant is expected to be an xarray DataArray with dims (T, Y, X).
        """
        if "M" in Predictant.coords:
            Predictant = Predictant.isel(M=0).drop_vars('M').squeeze()
        mask = xr.where(~np.isnan(Predictant.isel(T=0)), 1, np.nan).drop_vars(['T']).squeeze().to_numpy()
        Predictant = Predictant.transpose('T', 'Y', 'X')
        index_start = Predictant.get_index("T").get_loc(str(clim_year_start)).start
        index_end = Predictant.get_index("T").get_loc(str(clim_year_end)).stop
        rainfall_for_tercile = Predictant.isel(T=slice(index_start, index_end))
        terciles = rainfall_for_tercile.quantile([0.33, 0.67], dim='T')
        error_variance = (Predictant - hindcast_det).var(dim='T')
        dof = len(Predictant.get_index("T")) - 2
        if self.dist_method == "t":
            calc_func = self.calculate_tercile_probabilities
            hindcast_prob = xr.apply_ufunc(
                calc_func,
                hindcast_det,
                error_variance,
                terciles.isel(quantile=0).drop_vars('quantile'),
                terciles.isel(quantile=1).drop_vars('quantile'),
                input_core_dims=[('T',), (), (), ()],
                vectorize=True,
                kwargs={'dof': dof},
                dask='parallelized',
                output_core_dims=[('probability', 'T')],
                output_dtypes=['float'],
                dask_gufunc_kwargs={'output_sizes': {'probability': 3}, "allow_rechunk": True}
            )
        elif self.dist_method == "weibull_min":
            calc_func = self.calculate_tercile_probabilities_weibull_min
            hindcast_prob = xr.apply_ufunc(
                calc_func,
                hindcast_det,
                error_variance,
                terciles.isel(quantile=0).drop_vars('quantile'),
                terciles.isel(quantile=1).drop_vars('quantile'),
                input_core_dims=[('T',), (), (), ()],
                vectorize=True,
                kwargs={'dof': dof},
                dask='parallelized',
                output_core_dims=[('probability', 'T')],
                output_dtypes=['float'],
                dask_gufunc_kwargs={'output_sizes': {'probability': 3}, "allow_rechunk": True}
            )
        elif self.dist_method == "gamma":
            calc_func = self.calculate_tercile_probabilities_gamma
            hindcast_prob = xr.apply_ufunc(
                calc_func,
                hindcast_det,
                error_variance,
                terciles.isel(quantile=0).drop_vars('quantile'),
                terciles.isel(quantile=1).drop_vars('quantile'),
                input_core_dims=[('T',), (), (), ()],
                vectorize=True,
                dask='parallelized',
                output_core_dims=[('probability', 'T')],
                output_dtypes=['float'],
                dask_gufunc_kwargs={'output_sizes': {'probability': 3}, "allow_rechunk": True}
            )
        elif self.dist_method == "nonparam":
            calc_func = self.calculate_tercile_probabilities_nonparametric
            error_samples = Predictant - hindcast_det
            hindcast_prob = xr.apply_ufunc(
                calc_func,
                hindcast_det,
                error_samples,
                terciles.isel(quantile=0).drop_vars('quantile'),
                terciles.isel(quantile=1).drop_vars('quantile'),
                input_core_dims=[('T',), ('T',), (), ()],
                output_core_dims=[('probability', 'T')],
                vectorize=True,
                dask='parallelized',
                output_dtypes=[float],
                dask_gufunc_kwargs={'output_sizes': {'probability': 3}, "allow_rechunk": True}
            )
        elif self.dist_method == "normal":
            calc_func = self.calculate_tercile_probabilities_normal
            hindcast_prob = xr.apply_ufunc(
                calc_func,
                hindcast_det,
                error_variance,
                terciles.isel(quantile=0).drop_vars('quantile'),
                terciles.isel(quantile=1).drop_vars('quantile'),
                input_core_dims=[('T',), (), (), ()],
                vectorize=True,
                dask='parallelized',
                output_core_dims=[('probability', 'T')],
                output_dtypes=['float'],
                dask_gufunc_kwargs={'output_sizes': {'probability': 3}, "allow_rechunk": True}
            )
        elif self.dist_method == "lognormal":
            calc_func = self.calculate_tercile_probabilities_lognormal
            hindcast_prob = xr.apply_ufunc(
                calc_func,
                hindcast_det,
                error_variance,
                terciles.isel(quantile=0).drop_vars('quantile'),
                terciles.isel(quantile=1).drop_vars('quantile'),
                input_core_dims=[('T',), (), (), ()],
                vectorize=True,
                dask='parallelized',
                output_core_dims=[('probability', 'T')],
                output_dtypes=['float'],
                dask_gufunc_kwargs={'output_sizes': {'probability': 3}, "allow_rechunk": True}
            )
        else:
            raise ValueError(f"Invalid dist_method: {self.dist_method}.")
        hindcast_prob = hindcast_prob.assign_coords(probability=('probability', ['PB', 'PN', 'PA']))
        return hindcast_prob.transpose('probability', 'T', 'Y', 'X')

    @staticmethod
    def _reshape_and_filter_data(da):
        """
        Helper: stack the DataArray from (T, Y, X[, M]) to (n_samples, n_features)
        and remove rows containing NaNs.
        """
        da_stacked = da.stack(sample=('T', 'Y', 'X'))
        if 'M' in da.dims:
            da_stacked = da_stacked.transpose('sample', 'M')
        else:
            da_stacked = da_stacked.transpose('sample')
        da_values = da_stacked.values
        nan_mask = np.any(np.isnan(da_values), axis=1)
        return da_values[~nan_mask], nan_mask, da_values

    def forecast(self, Predictant, clim_year_start, clim_year_end, hindcast_det, hindcast_det_cross, Predictor_for_year, best_params=None, cluster_da=None):
        """
        Forecast method using a single XGBoost model with optimized hyperparameters.
        Parameters
        ----------
        Predictant : xarray.DataArray
            Observed predictand data with dimensions (T, Y, X) or (T, M, Y, X).
        clim_year_start : int or str
            Start year of the climatology period.
        clim_year_end : int or str
            End year of the climatology period.
        hindcast_det : xarray.DataArray
            Deterministic hindcast data for training with dimensions (T, M, Y, X).
        hindcast_det_cross : xarray.DataArray
            Deterministic hindcast data for error estimation with dimensions (T, Y, X).
        Predictor_for_year : xarray.DataArray
            Predictor data for the target year with dimensions (T, M, Y, X).
        best_params : dict, optional
            Pre-computed best hyperparameters. If None, computes internally.
        cluster_da : xarray.DataArray, optional
            Pre-computed cluster labels. If None, computes internally.
        Returns
        -------
        forecast_det : xarray.DataArray
            Deterministic forecast with dimensions (T, Y, X).
        forecast_prob : xarray.DataArray
            Tercile probabilities with dimensions (probability, T, Y, X).
        """
        if "M" in Predictant.coords:
            Predictant_no_m = Predictant.isel(M=0).drop_vars('M').squeeze()
        else:
            Predictant_no_m = Predictant
        mask = xr.where(~np.isnan(Predictant_no_m.isel(T=0)), 1, np.nan).drop_vars(['T']).squeeze().to_numpy()
        # Standardize Predictor_for_year using hindcast climatology
        mean_val = hindcast_det.sel(T=slice(str(clim_year_start), str(clim_year_end))).mean(dim='T')
        std_val = hindcast_det.sel(T=slice(str(clim_year_start), str(clim_year_end))).std(dim='T')
        Predictor_for_year_st = (Predictor_for_year - mean_val) / std_val
        hindcast_det_st = standardize_timeseries(hindcast_det, clim_year_start, clim_year_end)
        Predictant_st = standardize_timeseries(Predictant_no_m, clim_year_start, clim_year_end)
        hindcast_det_st['T'] = Predictant_st['T']
        
        # Extract coordinates from X_test
        time = Predictor_for_year_st['T']
        lat = Predictor_for_year_st['Y']
        lon = Predictor_for_year_st['X']
        n_time = len(time)
        n_lat = len(lat)
        n_lon = len(lon)
        # Use provided best_params and cluster_da or compute if None
        if best_params is None:
            best_params, cluster_da = self.compute_hyperparameters(hindcast_det, Predictant_no_m, clim_year_start, clim_year_end)
        # Initialize predictions array
        predictions = np.full((n_time, n_lat, n_lon), np.nan)
        self.xgb = {}  # Dictionary to store models per cluster
        for c in range(self.n_clusters):
            if c not in best_params:
                continue
            bp = best_params[c]
            # Mask for this cluster
            mask_3d_train = (cluster_da == c).expand_dims({'T': hindcast_det_st['T']})
            mask_3d_test = (cluster_da == c).expand_dims({'T': Predictor_for_year_st['T']})
            # Stack training data for cluster
            X_train_stacked_c = hindcast_det_st.where(mask_3d_train).stack(sample=('T', 'Y', 'X')).transpose('sample', 'M').values
            y_train_stacked_c = Predictant_st.where(mask_3d_train).stack(sample=('T', 'Y', 'X')).values.ravel()
            train_nan_mask = np.any(~np.isfinite(X_train_stacked_c), axis=1) | ~np.isfinite(y_train_stacked_c)
            X_train_clean_c = X_train_stacked_c[~train_nan_mask]
            y_train_clean_c = y_train_stacked_c[~train_nan_mask]
            # Stack testing data for cluster
            X_test_stacked_c = Predictor_for_year_st.where(mask_3d_test).stack(sample=('T', 'Y', 'X')).transpose('sample', 'M').values
            test_nan_mask = np.any(~np.isfinite(X_test_stacked_c), axis=1)
            X_test_clean_c = X_test_stacked_c[~test_nan_mask]
            # Initialize the model with best parameters for this cluster
            xgb_c = XGBRegressor(
                n_estimators=bp['n_estimators'],
                learning_rate=bp['learning_rate'],
                max_depth=bp['max_depth'],
                min_child_weight=bp['min_child_weight'],
                subsample=bp['subsample'],
                colsample_bytree=bp['colsample_bytree'],
                random_state=self.random_state,
                verbosity=0,
                n_jobs=-1
            )
            # Fit and predict
            xgb_c.fit(X_train_clean_c, y_train_clean_c)
            self.xgb[c] = xgb_c
            y_pred_c = xgb_c.predict(X_test_clean_c)
            # Reconstruct predictions for this cluster
            result_c = np.full(len(X_test_stacked_c), np.nan)
            result_c[~test_nan_mask] = y_pred_c
            pred_c_reshaped = result_c.reshape(n_time, n_lat, n_lon)
            # Fill in the predictions array
            predictions = np.where(np.isnan(predictions), pred_c_reshaped, predictions)
        result_da = xr.DataArray(
            data=predictions,
            coords={'T': time, 'Y': lat, 'X': lon},
            dims=['T', 'Y', 'X']
        ) * mask
        result_da = reverse_standardize(result_da, Predictant_no_m, clim_year_start, clim_year_end)
        year = Predictor_for_year.coords['T'].values.astype('datetime64[Y]').astype(int)[0] + 1970
        T_value_1 = Predictant_no_m.isel(T=0).coords['T'].values
        month_1 = T_value_1.astype('datetime64[M]').astype(int) % 12 + 1
        new_T_value = np.datetime64(f"{year}-{month_1:02d}-01")
        result_da = result_da.assign_coords(T=xr.DataArray([new_T_value], dims=["T"]))
        result_da['T'] = result_da['T'].astype('datetime64[ns]')
        # Compute tercile probabilities
        index_start = Predictant_no_m.get_index("T").get_loc(str(clim_year_start)).start
        index_end = Predictant_no_m.get_index("T").get_loc(str(clim_year_end)).stop
        rainfall_for_tercile = Predictant_no_m.isel(T=slice(index_start, index_end))
        terciles = rainfall_for_tercile.quantile([0.33, 0.67], dim='T')
        error_variance = (Predictant_no_m - hindcast_det_cross).var(dim='T')
        dof = len(Predictant_no_m.get_index("T")) - 2
        if self.dist_method == "t":
            calc_func = self.calculate_tercile_probabilities
            hindcast_prob = xr.apply_ufunc(
                calc_func,
                result_da,
                error_variance,
                terciles.isel(quantile=0).drop_vars('quantile'),
                terciles.isel(quantile=1).drop_vars('quantile'),
                input_core_dims=[('T',), (), (), ()],
                vectorize=True,
                kwargs={'dof': dof},
                dask='parallelized',
                output_core_dims=[('probability', 'T')],
                output_dtypes=['float'],
                dask_gufunc_kwargs={'output_sizes': {'probability': 3}, "allow_rechunk": True}
            )
        elif self.dist_method == "weibull_min":
            calc_func = self.calculate_tercile_probabilities_weibull_min
            hindcast_prob = xr.apply_ufunc(
                calc_func,
                result_da,
                error_variance,
                terciles.isel(quantile=0).drop_vars('quantile'),
                terciles.isel(quantile=1).drop_vars('quantile'),
                input_core_dims=[('T',), (), (), ()],
                vectorize=True,
                kwargs={'dof': dof},
                dask='parallelized',
                output_core_dims=[('probability', 'T')],
                output_dtypes=['float'],
                dask_gufunc_kwargs={'output_sizes': {'probability': 3}, "allow_rechunk": True}
            )
        elif self.dist_method == "gamma":
            calc_func = self.calculate_tercile_probabilities_gamma
            hindcast_prob = xr.apply_ufunc(
                calc_func,
                result_da,
                error_variance,
                terciles.isel(quantile=0).drop_vars('quantile'),
                terciles.isel(quantile=1).drop_vars('quantile'),
                input_core_dims=[('T',), (), (), ()],
                vectorize=True,
                dask='parallelized',
                output_core_dims=[('probability', 'T')],
                output_dtypes=['float'],
                dask_gufunc_kwargs={'output_sizes': {'probability': 3}, "allow_rechunk": True}
            )
        elif self.dist_method == "normal":
            calc_func = self.calculate_tercile_probabilities_normal
            hindcast_prob = xr.apply_ufunc(
                calc_func,
                result_da,
                error_variance,
                terciles.isel(quantile=0).drop_vars('quantile'),
                terciles.isel(quantile=1).drop_vars('quantile'),
                input_core_dims=[('T',), (), (), ()],
                vectorize=True,
                dask='parallelized',
                output_core_dims=[('probability', 'T')],
                output_dtypes=['float'],
                dask_gufunc_kwargs={'output_sizes': {'probability': 3}, "allow_rechunk": True}
            )
        elif self.dist_method == "lognormal":
            calc_func = self.calculate_tercile_probabilities_lognormal
            hindcast_prob = xr.apply_ufunc(
                calc_func,
                result_da,
                error_variance,
                terciles.isel(quantile=0).drop_vars('quantile'),
                terciles.isel(quantile=1).drop_vars('quantile'),
                input_core_dims=[('T',), (), (), ()],
                vectorize=True,
                dask='parallelized',
                output_core_dims=[('probability', 'T')],
                output_dtypes=['float'],
                dask_gufunc_kwargs={'output_sizes': {'probability': 3}, "allow_rechunk": True}
            )
        elif self.dist_method == "nonparam":
            calc_func = self.calculate_tercile_probabilities_nonparametric
            error_samples = Predictant_no_m - hindcast_det_cross
            error_samples = error_samples.rename({'T': 'S'})
            hindcast_prob = xr.apply_ufunc(
                calc_func,
                result_da,
                error_samples,
                terciles.isel(quantile=0).drop_vars('quantile'),
                terciles.isel(quantile=1).drop_vars('quantile'),
                input_core_dims=[('T',), ('S',), (), ()],
                output_core_dims=[('probability', 'T')],
                vectorize=True,
                dask='parallelized',
                output_dtypes=[float],
                dask_gufunc_kwargs={'output_sizes': {'probability': 3}, "allow_rechunk": True}
            )
        else:
            raise ValueError(f"Invalid dist_method: {self.dist_method}.")
        hindcast_prob = hindcast_prob.assign_coords(probability=('probability', ['PB', 'PN', 'PA']))
        return result_da * mask, mask * hindcast_prob.transpose('probability', 'T', 'Y', 'X')


class WAS_mme_RF_:
    """
    Random Forest-based Multi-Model Ensemble (MME) forecasting.
    This class implements a single-model forecasting approach using scikit-learn's RandomForestRegressor
    for deterministic predictions, with optional tercile probability calculations using
    various statistical distributions. Implements hyperparameter optimization via randomized search.

    Parameters
    ----------
    n_estimators_range : list of int, optional
        List of n_estimators values to tune (default is [50, 100, 200, 300]).
    max_depth_range : list of int, optional
        List of max depths to tune (default is [None, 10, 20, 30]).
    min_samples_split_range : list of int, optional
        List of minimum samples required to split to tune (default is [2, 5, 10]).
    min_samples_leaf_range : list of int, optional
        List of minimum samples required at leaf node to tune (default is [1, 2, 4]).
    max_features_range : list of str or float, optional
        List of max features to tune (default is ['auto', 'sqrt', 0.33, 0.5]).
    random_state : int, optional
        Seed for reproducibility (default is 42).
    dist_method : str, optional
        Distribution method for tercile probabilities ('t', 'gamma', etc.) (default is 'gamma').
    n_iter_search : int, optional
        Number of iterations for randomized search (default is 10).
    cv_folds : int, optional
        Number of cross-validation folds (default is 3).
    """
    def __init__(self,
                 n_estimators_range=[50, 100, 200, 300],
                 max_depth_range=[None, 10, 20, 30],
                 min_samples_split_range=[2, 5, 10],
                 min_samples_leaf_range=[1, 2, 4],
                 max_features_range=['auto', 'sqrt', 0.33, 0.5],
                 random_state=42,
                 dist_method="gamma",
                 n_iter_search=10,
                 cv_folds=3):
        self.n_estimators_range = n_estimators_range
        self.max_depth_range = max_depth_range
        self.min_samples_split_range = min_samples_split_range
        self.min_samples_leaf_range = min_samples_leaf_range
        self.max_features_range = max_features_range
        self.random_state = random_state
        self.dist_method = dist_method
        self.n_iter_search = n_iter_search
        self.cv_folds = cv_folds
        self.rf = None

    def compute_hyperparameters(self, Predictors, Predictand, clim_year_start, clim_year_end):
        """
        Independently computes the best hyperparameters using randomized search on stacked training data.

        Parameters
        ----------
        Predictors : xarray.DataArray
            Training predictor data with dimensions (T, M, Y, X).
        Predictand : xarray.DataArray
            Training predictand data with dimensions (T, Y, X).
        clim_year_start : int
            Start year of the climatology period.
        clim_year_end : int
            End year of the climatology period.

        Returns
        -------
        dict
            Best hyperparameters found.
        """
        X_train = standardize_timeseries(Predictors, clim_year_start, clim_year_end)
        y_train = standardize_timeseries(Predictand, clim_year_start, clim_year_end)

        # Stack training data
        X_train_stacked = X_train.stack(sample=('T', 'Y', 'X')).transpose('sample', 'M').values
        y_train_stacked = y_train.stack(sample=('T', 'Y', 'X')).values.ravel()
        train_nan_mask = np.any(~np.isfinite(X_train_stacked), axis=1) | ~np.isfinite(y_train_stacked)
        X_train_clean = X_train_stacked[~train_nan_mask]
        y_train_clean = y_train_stacked[~train_nan_mask]

        # Prepare parameter distributions
        param_dist = {
            'n_estimators': self.n_estimators_range,
            'max_depth': self.max_depth_range,
            'min_samples_split': self.min_samples_split_range,
            'min_samples_leaf': self.min_samples_leaf_range,
            'max_features': self.max_features_range
        }

        # Initialize RandomForestRegressor base model
        model = RandomForestRegressor(random_state=self.random_state)

        # Randomized search
        random_search = RandomizedSearchCV(
            model, param_distributions=param_dist, n_iter=self.n_iter_search,
            cv=self.cv_folds, scoring='neg_mean_squared_error',
            random_state=self.random_state, error_score=np.nan
        )

        random_search.fit(X_train_clean, y_train_clean)
        best_params = random_search.best_params_

        return best_params

    def compute_model(self, X_train, y_train, X_test, y_test, best_params=None):
        """
        Compute deterministic hindcast using the RandomForestRegressor model with injected hyperparameters.

        Parameters
        ----------
        X_train : xarray.DataArray
            Training predictor data with dimensions (T, M, Y, X).
        y_train : xarray.DataArray
            Training predictand data with dimensions (T, Y, X).
        X_test : xarray.DataArray
            Testing predictor data with dimensions (T, M, Y, X).
        y_test : xarray.DataArray
            Testing predictand data with dimensions (T, Y, X).
        best_params : dict, optional
            Pre-computed best hyperparameters. If None, computes internally.

        Returns
        -------
        predicted_da : xarray.DataArray
            Deterministic hindcast with dimensions (T, Y, X).
        """
        # Extract coordinate variables from X_test
        time = X_test['T']
        lat = X_test['Y']
        lon = X_test['X']
        n_time = len(X_test.coords['T'])
        n_lat = len(X_test.coords['Y'])
        n_lon = len(X_test.coords['X'])

        # Stack training data
        X_train_stacked = X_train.stack(sample=('T', 'Y', 'X')).transpose('sample', 'M').values
        y_train_stacked = y_train.stack(sample=('T', 'Y', 'X')).transpose('sample', 'M').values
        train_nan_mask = (np.any(~np.isfinite(X_train_stacked), axis=1) | np.any(~np.isfinite(y_train_stacked), axis=1))
        X_train_clean = X_train_stacked[~train_nan_mask]
        y_train_clean = y_train_stacked[~train_nan_mask]

        # Stack testing data
        X_test_stacked = X_test.stack(sample=('T', 'Y', 'X')).transpose('sample', 'M').values
        y_test_stacked = y_test.stack(sample=('T', 'Y', 'X')).transpose('sample', 'M').values
        test_nan_mask = (np.any(~np.isfinite(X_test_stacked), axis=1) | np.any(~np.isfinite(y_test_stacked), axis=1))

        # Use provided best_params or compute if None
        if best_params is None:
            best_params = self.compute_hyperparameters(X_train, y_train, clim_year_start, clim_year_end)

        # Initialize the model with best parameters
        self.rf = RandomForestRegressor(
            n_estimators=best_params['n_estimators'],
            max_depth=best_params['max_depth'],
            min_samples_split=best_params['min_samples_split'],
            min_samples_leaf=best_params['min_samples_leaf'],
            max_features=best_params['max_features'],
            random_state=self.random_state
        )

        # Fit the model and predict on non-NaN testing data
        self.rf.fit(X_train_clean, y_train_clean)
        y_pred = self.rf.predict(X_test_stacked[~test_nan_mask])

        # Reconstruct predictions
        result = np.empty_like(np.squeeze(y_test_stacked))
        result[np.squeeze(test_nan_mask)] = np.squeeze(y_test_stacked[test_nan_mask])
        result[~np.squeeze(test_nan_mask)] = y_pred

        predictions_reshaped = result.reshape(n_time, n_lat, n_lon)
        predicted_da = xr.DataArray(data=predictions_reshaped,
                                    coords={'T': time, 'Y': lat, 'X': lon},
                                    dims=['T', 'Y', 'X'])
        return predicted_da

    # ------------------ Probability Calculation Methods ------------------
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
    def calculate_tercile_probabilities_weibull_min(best_guess, error_variance, first_tercile, second_tercile, dof):
        """
        Weibull minimum-based method.
        """
        n_time = len(best_guess)
        pred_prob = np.empty((3, n_time))

        if np.all(np.isnan(best_guess)):
            pred_prob[:] = np.nan
        else:
            error_std = np.sqrt(error_variance)
            pred_prob[0, :] = stats.weibull_min.cdf(first_tercile, c=dof, loc=best_guess, scale=error_std)
            pred_prob[1, :] = stats.weibull_min.cdf(second_tercile, c=dof, loc=best_guess, scale=error_std) - \
                              stats.weibull_min.cdf(first_tercile, c=dof, loc=best_guess, scale=error_std)
            pred_prob[2, :] = 1 - stats.weibull_min.cdf(second_tercile, c=dof, loc=best_guess, scale=error_std)

        return pred_prob

    @staticmethod
    def calculate_tercile_probabilities_gamma(best_guess, error_variance, T1, T2, dof=None):
        """Gamma-distribution based method."""
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

    @staticmethod
    def calculate_tercile_probabilities_normal(best_guess, error_variance, first_tercile, second_tercile):
        """Normal-distribution based method."""
        n_time = len(best_guess)
        pred_prob = np.empty((3, n_time))
        if np.all(np.isnan(best_guess)):
            pred_prob[:] = np.nan
        else:
            error_std = np.sqrt(error_variance)
            pred_prob[0, :] = norm.cdf(first_tercile, loc=best_guess, scale=error_std)
            pred_prob[1, :] = norm.cdf(second_tercile, loc=best_guess, scale=error_std) - norm.cdf(first_tercile, loc=best_guess, scale=error_std)
            pred_prob[2, :] = 1 - norm.cdf(second_tercile, loc=best_guess, scale=error_std)
        return pred_prob

    @staticmethod
    def calculate_tercile_probabilities_lognormal(best_guess, error_variance, first_tercile, second_tercile):
        """Lognormal-distribution based method."""
        n_time = len(best_guess)
        pred_prob = np.empty((3, n_time))
        if np.any(np.isnan(best_guess)) or np.any(np.isnan(error_variance)):
            pred_prob[:] = np.nan
            return pred_prob
        sigma = np.sqrt(np.log(1 + error_variance / (best_guess**2)))
        mu = np.log(best_guess) - sigma**2 / 2
        pred_prob[0, :] = lognorm.cdf(first_tercile, s=sigma, scale=np.exp(mu))
        pred_prob[1, :] = lognorm.cdf(second_tercile, s=sigma, scale=np.exp(mu)) - lognorm.cdf(first_tercile, s=sigma, scale=np.exp(mu))
        pred_prob[2, :] = 1 - lognorm.cdf(second_tercile, s=sigma, scale=np.exp(mu))
        return pred_prob

    def compute_prob(self, Predictant, clim_year_start, clim_year_end, hindcast_det):
        """
        Compute tercile probabilities using the chosen distribution method.
        Predictant is expected to be an xarray DataArray with dims (T, Y, X).
        """
        if "M" in Predictant.coords:
            Predictant = Predictant.isel(M=0).drop_vars('M').squeeze()
        mask = xr.where(~np.isnan(Predictant.isel(T=0)), 1, np.nan).drop_vars(['T']).squeeze().to_numpy()

        Predictant = Predictant.transpose('T', 'Y', 'X')
        index_start = Predictant.get_index("T").get_loc(str(clim_year_start)).start
        index_end = Predictant.get_index("T").get_loc(str(clim_year_end)).stop
        rainfall_for_tercile = Predictant.isel(T=slice(index_start, index_end))
        terciles = rainfall_for_tercile.quantile([0.33, 0.67], dim='T')
        error_variance = (Predictant - hindcast_det).var(dim='T')
        dof = len(Predictant.get_index("T")) - 2

        if self.dist_method == "t":
            calc_func = self.calculate_tercile_probabilities
            hindcast_prob = xr.apply_ufunc(
                calc_func,
                hindcast_det,
                error_variance,
                terciles.isel(quantile=0).drop_vars('quantile'),
                terciles.isel(quantile=1).drop_vars('quantile'),
                input_core_dims=[('T',), (), (), ()],
                vectorize=True,
                kwargs={'dof': dof},
                dask='parallelized',
                output_core_dims=[('probability','T')],
                output_dtypes=['float'],
                dask_gufunc_kwargs={'output_sizes': {'probability': 3}, "allow_rechunk": True}
            )
        elif self.dist_method == "weibull_min":
            calc_func = self.calculate_tercile_probabilities_weibull_min
            hindcast_prob = xr.apply_ufunc(
                calc_func,
                hindcast_det,
                error_variance,
                terciles.isel(quantile=0).drop_vars('quantile'),
                terciles.isel(quantile=1).drop_vars('quantile'),
                input_core_dims=[('T',), (), (), ()],
                vectorize=True,
                kwargs={'dof': dof},
                dask='parallelized',
                output_core_dims=[('probability', 'T')],
                output_dtypes=['float'],
                dask_gufunc_kwargs={'output_sizes': {'probability': 3}, "allow_rechunk": True}
            )
        elif self.dist_method == "gamma":
            calc_func = self.calculate_tercile_probabilities_gamma
            hindcast_prob = xr.apply_ufunc(
                calc_func,
                hindcast_det,
                error_variance,
                terciles.isel(quantile=0).drop_vars('quantile'),
                terciles.isel(quantile=1).drop_vars('quantile'),
                input_core_dims=[('T',), (), (), ()],
                vectorize=True,
                dask='parallelized',
                output_core_dims=[('probability','T')],
                output_dtypes=['float'],
                dask_gufunc_kwargs={'output_sizes': {'probability': 3}, "allow_rechunk": True}
            )
        elif self.dist_method == "nonparam":
            calc_func = self.calculate_tercile_probabilities_nonparametric
            error_samples = (Predictant - hindcast_det)
            hindcast_prob = xr.apply_ufunc(
                calc_func,
                hindcast_det,
                error_samples,
                terciles.isel(quantile=0).drop_vars('quantile'),
                terciles.isel(quantile=1).drop_vars('quantile'),
                input_core_dims=[('T',), ('T',), (), ()],
                output_core_dims=[('probability','T')],
                vectorize=True,
                dask='parallelized',
                output_dtypes=[float],
                dask_gufunc_kwargs={'output_sizes': {'probability': 3}, "allow_rechunk": True}
            )
        elif self.dist_method == "normal":
            calc_func = self.calculate_tercile_probabilities_normal
            hindcast_prob = xr.apply_ufunc(
                calc_func,
                hindcast_det,
                error_variance,
                terciles.isel(quantile=0).drop_vars('quantile'),
                terciles.isel(quantile=1).drop_vars('quantile'),
                input_core_dims=[('T',), (), (), ()],
                vectorize=True,
                dask='parallelized',
                output_core_dims=[('probability','T')],
                output_dtypes=['float'],
                dask_gufunc_kwargs={'output_sizes': {'probability': 3}, "allow_rechunk": True}
            )
        elif self.dist_method == "lognormal":
            calc_func = self.calculate_tercile_probabilities_lognormal
            hindcast_prob = xr.apply_ufunc(
                calc_func,
                hindcast_det,
                error_variance,
                terciles.isel(quantile=0).drop_vars('quantile'),
                terciles.isel(quantile=1).drop_vars('quantile'),
                input_core_dims=[('T',), (), (), ()],
                vectorize=True,
                dask='parallelized',
                output_core_dims=[('probability','T')],
                output_dtypes=['float'],
                dask_gufunc_kwargs={'output_sizes': {'probability': 3}, "allow_rechunk": True}
            )
        else:
            raise ValueError(f"Invalid dist_method: {self.dist_method}.")

        hindcast_prob = hindcast_prob.assign_coords(probability=('probability', ['PB', 'PN', 'PA']))
        return hindcast_prob.transpose('probability', 'T', 'Y', 'X')

    @staticmethod
    def _reshape_and_filter_data(da):
        """
        Helper: stack the DataArray from (T, Y, X[, M]) to (n_samples, n_features)
        and remove rows containing NaNs.
        """
        da_stacked = da.stack(sample=('T', 'Y', 'X'))
        if 'M' in da.dims:
            da_stacked = da_stacked.transpose('sample', 'M')
        else:
            da_stacked = da_stacked.transpose('sample')
        da_values = da_stacked.values
        nan_mask = np.any(np.isnan(da_values), axis=1)
        return da_values[~nan_mask], nan_mask, da_values

    def forecast(self, Predictant, clim_year_start, clim_year_end, hindcast_det,
                 hindcast_det_cross, Predictor_for_year, best_params=None):
        """
        Forecast method using a single Random Forest model with optimized hyperparameters.

        Parameters
        ----------
        Predictant : xarray.DataArray
            Observed predictand data with dimensions (T, Y, X) or (T, M, Y, X).
        clim_year_start : int or str
            Start year of the climatology period.
        clim_year_end : int or str
            End year of the climatology period.
        hindcast_det : xarray.DataArray
            Deterministic hindcast data for training with dimensions (T, M, Y, X).
        hindcast_det_cross : xarray.DataArray
            Deterministic hindcast data for error estimation with dimensions (T, Y, X).
        Predictor_for_year : xarray.DataArray
            Predictor data for the target year with dimensions (T, M, Y, X).
        best_params : dict, optional
            Pre-computed best hyperparameters. If None, computes internally.

        Returns
        -------
        forecast_det : xarray.DataArray
            Deterministic forecast with dimensions (T, Y, X).
        forecast_prob : xarray.DataArray
            Tercile probabilities with dimensions (probability, T, Y, X).
        """
        if "M" in Predictant.coords:
            Predictant_no_m = Predictant.isel(M=0).drop_vars('M').squeeze()
        else:
            Predictant_no_m = Predictant

        mask = xr.where(~np.isnan(Predictant_no_m.isel(T=0)), 1, np.nan).drop_vars(['T']).squeeze().to_numpy()

        # Standardize Predictor_for_year using hindcast climatology
        mean_val = hindcast_det.sel(T=slice(str(clim_year_start), str(clim_year_end))).mean(dim='T')
        std_val = hindcast_det.sel(T=slice(str(clim_year_start), str(clim_year_end))).std(dim='T')
        Predictor_for_year_st = (Predictor_for_year - mean_val) / std_val

        hindcast_det_st = standardize_timeseries(hindcast_det, clim_year_start, clim_year_end)
        Predictant_st = standardize_timeseries(Predictant_no_m, clim_year_start, clim_year_end)
        y_test = Predictant_st.isel(T=[-1])

        # Extract coordinates from X_test
        time = Predictor_for_year_st['T']
        lat = Predictor_for_year_st['Y']
        lon = Predictor_for_year_st['X']
        n_time = len(Predictor_for_year_st.coords['T'])
        n_lat = len(Predictor_for_year_st.coords['Y'])
        n_lon = len(Predictor_for_year_st.coords['X'])

        # Stack training data and remove rows with NaNs
        X_train_stacked = hindcast_det_st.stack(sample=('T', 'Y', 'X')).transpose('sample', 'M').values
        y_train_stacked = Predictant_st.stack(sample=('T', 'Y', 'X')).values.ravel()
        train_nan_mask = np.any(~np.isfinite(X_train_stacked), axis=1) | ~np.isfinite(y_train_stacked)
        X_train_clean = X_train_stacked[~train_nan_mask]
        y_train_clean = y_train_stacked[~train_nan_mask]

        # Stack testing data
        X_test_stacked = Predictor_for_year_st.stack(sample=('T', 'Y', 'X')).transpose('sample', 'M').values
        y_test_stacked = y_test.stack(sample=('T', 'Y', 'X')).values.ravel()
        test_nan_mask = np.any(~np.isfinite(X_test_stacked), axis=1) | ~np.isfinite(y_test_stacked)

        # Use provided best_params or compute if None
        if best_params is None:
            best_params = self.compute_hyperparameters(hindcast_det, Predictant_no_m, clim_year_start, clim_year_end)

        # Initialize and fit the model with best parameters
        self.rf = RandomForestRegressor(
            n_estimators=best_params['n_estimators'],
            max_depth=best_params['max_depth'],
            min_samples_split=best_params['min_samples_split'],
            min_samples_leaf=best_params['min_samples_leaf'],
            max_features=best_params['max_features'],
            random_state=self.random_state
        )
        self.rf.fit(X_train_clean, y_train_clean)
        y_pred = self.rf.predict(X_test_stacked[~test_nan_mask])

        # Reconstruct the prediction array
        result = np.empty_like(np.squeeze(y_test_stacked))
        result[np.squeeze(test_nan_mask)] = np.squeeze(y_test_stacked[test_nan_mask])
        result[~np.squeeze(test_nan_mask)] = y_pred

        predictions_reshaped = result.reshape(n_time, n_lat, n_lon)
        result_da = xr.DataArray(data=predictions_reshaped,
                                 coords={'T': time, 'Y': lat, 'X': lon},
                                 dims=['T', 'Y', 'X']) * mask

        result_da = reverse_standardize(result_da, Predictant_no_m, clim_year_start, clim_year_end)
        year = Predictor_for_year.coords['T'].values.astype('datetime64[Y]').astype(int)[0] + 1970
        T_value_1 = Predictant_no_m.isel(T=0).coords['T'].values
        month_1 = T_value_1.astype('datetime64[M]').astype(int) % 12 + 1
        new_T_value = np.datetime64(f"{year}-{month_1:02d}-{1:02d}")
        result_da = result_da.assign_coords(T=xr.DataArray([new_T_value], dims=["T"]))
        result_da['T'] = result_da['T'].astype('datetime64[ns]')

        # Compute tercile probabilities
        index_start = Predictant_no_m.get_index("T").get_loc(str(clim_year_start)).start
        index_end = Predictant_no_m.get_index("T").get_loc(str(clim_year_end)).stop
        rainfall_for_tercile = Predictant_no_m.isel(T=slice(index_start, index_end))
        terciles = rainfall_for_tercile.quantile([0.33, 0.67], dim='T')
        error_variance = (Predictant_no_m - hindcast_det_cross).var(dim='T')
        dof = len(Predictant_no_m.get_index("T")) - 2

        if self.dist_method == "t":
            calc_func = self.calculate_tercile_probabilities
            hindcast_prob = xr.apply_ufunc(
                calc_func,
                result_da,
                error_variance,
                terciles.isel(quantile=0).drop_vars('quantile'),
                terciles.isel(quantile=1).drop_vars('quantile'),
                input_core_dims=[('T',), (), (), ()],
                vectorize=True,
                kwargs={'dof': dof},
                dask='parallelized',
                output_core_dims=[('probability','T')],
                output_dtypes=['float'],
                dask_gufunc_kwargs={'output_sizes': {'probability': 3}, "allow_rechunk": True}
            )
        elif self.dist_method == "weibull_min":
            calc_func = self.calculate_tercile_probabilities_weibull_min
            hindcast_prob = xr.apply_ufunc(
                calc_func,
                result_da,
                error_variance,
                terciles.isel(quantile=0).drop_vars('quantile'),
                terciles.isel(quantile=1).drop_vars('quantile'),
                input_core_dims=[('T',), (), (), ()],
                vectorize=True,
                kwargs={'dof': dof},
                dask='parallelized',
                output_core_dims=[('probability', 'T')],
                output_dtypes=['float'],
                dask_gufunc_kwargs={'output_sizes': {'probability': 3}, "allow_rechunk": True}
            )
        elif self.dist_method == "gamma":
            calc_func = self.calculate_tercile_probabilities_gamma
            hindcast_prob = xr.apply_ufunc(
                calc_func,
                result_da,
                error_variance,
                terciles.isel(quantile=0).drop_vars('quantile'),
                terciles.isel(quantile=1).drop_vars('quantile'),
                input_core_dims=[('T',), (), (), ()],
                vectorize=True,
                dask='parallelized',
                output_core_dims=[('probability','T')],
                output_dtypes=['float'],
                dask_gufunc_kwargs={'output_sizes': {'probability': 3}, "allow_rechunk": True}
            )
        elif self.dist_method == "normal":
            calc_func = self.calculate_tercile_probabilities_normal
            hindcast_prob = xr.apply_ufunc(
                calc_func,
                result_da,
                error_variance,
                terciles.isel(quantile=0).drop_vars('quantile'),
                terciles.isel(quantile=1).drop_vars('quantile'),
                input_core_dims=[('T',), (), (), ()],
                vectorize=True,
                dask='parallelized',
                output_core_dims=[('probability','T')],
                output_dtypes=['float'],
                dask_gufunc_kwargs={'output_sizes': {'probability': 3}, "allow_rechunk": True}
            )
        elif self.dist_method == "lognormal":
            calc_func = self.calculate_tercile_probabilities_lognormal
            hindcast_prob = xr.apply_ufunc(
                calc_func,
                result_da,
                error_variance,
                terciles.isel(quantile=0).drop_vars('quantile'),
                terciles.isel(quantile=1).drop_vars('quantile'),
                input_core_dims=[('T',), (), (), ()],
                vectorize=True,
                dask='parallelized',
                output_core_dims=[('probability','T')],
                output_dtypes=['float'],
                dask_gufunc_kwargs={'output_sizes': {'probability': 3}, "allow_rechunk": True}
            )
        elif self.dist_method == "nonparam":
            calc_func = self.calculate_tercile_probabilities_nonparametric
            error_samples = Predictant_no_m - hindcast_det_cross
            error_samples = error_samples.rename({'T': 'S'})
            hindcast_prob = xr.apply_ufunc(
                calc_func,
                result_da,
                error_samples,
                terciles.isel(quantile=0).drop_vars('quantile'),
                terciles.isel(quantile=1).drop_vars('quantile'),
                input_core_dims=[('T',), ('S',), (), ()],
                output_core_dims=[('probability','T')],
                vectorize=True,
                dask='parallelized',
                output_dtypes=[float],
                dask_gufunc_kwargs={'output_sizes': {'probability': 3}, "allow_rechunk": True}
            )
        else:
            raise ValueError(f"Invalid dist_method: {self.dist_method}.")

        hindcast_prob = hindcast_prob.assign_coords(probability=('probability', ['PB', 'PN', 'PA']))
        return result_da * mask, mask * hindcast_prob.transpose('probability', 'T', 'Y', 'X')



class WAS_mme_RF:
    """
    Random Forest-based Multi-Model Ensemble (MME) forecasting.
    This class implements a single-model forecasting approach using scikit-learn's RandomForestRegressor
    for deterministic predictions, with optional tercile probability calculations using
    various statistical distributions. Implements hyperparameter optimization via randomized search.
    Parameters
    ----------
    n_estimators_range : list of int, optional
        List of n_estimators values to tune (default is [50, 100, 200, 300]).
    max_depth_range : list of int, optional
        List of max depths to tune (default is [None, 10, 20, 30]).
    min_samples_split_range : list of int, optional
        List of minimum samples required to split to tune (default is [2, 5, 10]).
    min_samples_leaf_range : list of int, optional
        List of minimum samples required at leaf node to tune (default is [1, 2, 4]).
    max_features_range : list of str or float, optional
        List of max features to tune (default is ['auto', 'sqrt', 0.33, 0.5]).
    random_state : int, optional
        Seed for reproducibility (default is 42).
    dist_method : str, optional
        Distribution method for tercile probabilities ('t', 'gamma', etc.) (default is 'gamma').
    n_iter_search : int, optional
        Number of iterations for randomized search (default is 10).
    cv_folds : int, optional
        Number of cross-validation folds (default is 3).
    n_clusters : int, optional
        Number of clusters for homogenized zones (default is 4).
    """
    def __init__(self,
                 n_estimators_range=[50, 100, 200, 300],
                 max_depth_range=[None, 10, 20, 30],
                 min_samples_split_range=[2, 5, 10],
                 min_samples_leaf_range=[1, 2, 4],
                 max_features_range=['auto', 'sqrt', 0.33, 0.5],
                 random_state=42,
                 dist_method="gamma",
                 n_iter_search=10,
                 cv_folds=3,
                 n_clusters=4):
        self.n_estimators_range = n_estimators_range
        self.max_depth_range = max_depth_range
        self.min_samples_split_range = min_samples_split_range
        self.min_samples_leaf_range = min_samples_leaf_range
        self.max_features_range = max_features_range
        self.random_state = random_state
        self.dist_method = dist_method
        self.n_iter_search = n_iter_search
        self.cv_folds = cv_folds
        self.n_clusters = n_clusters
        self.rf = None

    def compute_hyperparameters(self, Predictors, Predictand, clim_year_start, clim_year_end):
        """
        Independently computes the best hyperparameters using randomized search on stacked training data for each homogenized zone.
        Parameters
        ----------
        Predictors : xarray.DataArray
            Training predictor data with dimensions (T, M, Y, X).
        Predictand : xarray.DataArray
            Training predictand data with dimensions (T, Y, X).
        clim_year_start : int
            Start year of the climatology period.
        clim_year_end : int
            End year of the climatology period.
        Returns
        -------
        best_params_dict : dict
            Best hyperparameters for each cluster.
        cluster_da : xarray.DataArray
            Cluster labels with dimensions (Y, X).
        """
        if "M" in Predictand.coords:
            Predictand = Predictand.isel(M=0).drop_vars('M').squeeze()
           
        X_train_std = standardize_timeseries(Predictors, clim_year_start, clim_year_end)
        Predictand.name = "varname"
        # Step 1: Perform KMeans clustering based on predictand's spatial distribution
        kmeans = KMeans(n_clusters=self.n_clusters, random_state=self.random_state)
        Predictand_dropna = Predictand.to_dataframe().reset_index().dropna().drop(columns=['T'])
        variable_column = Predictand_dropna.columns[2]
        Predictand_dropna['cluster'] = kmeans.fit_predict(
           Predictand_dropna[[variable_column]]
        )
       
        # Convert cluster assignments back into an xarray structure
        df_unique = Predictand_dropna.drop_duplicates(subset=['Y', 'X'])
        dataset = df_unique.set_index(['Y', 'X']).to_xarray()
        mask = xr.where(~np.isnan(Predictand.isel(T=0)), 1, np.nan)
        Cluster = (dataset['cluster'] * mask)
              
        # Align cluster array with the predictand array
        xarray1, xarray2 = xr.align(Predictand, Cluster, join="outer")
       
        # Identify unique cluster labels
        clusters = np.unique(xarray2)
        clusters = clusters[~np.isnan(clusters)]
        cluster_da = xarray2
        y_train_std = standardize_timeseries(Predictand, clim_year_start, clim_year_end)
        X_train_std['T'] = y_train_std['T']
        # Prepare parameter distributions
        param_dist = {
            'n_estimators': self.n_estimators_range,
            'max_depth': self.max_depth_range,
            'min_samples_split': self.min_samples_split_range,
            'min_samples_leaf': self.min_samples_leaf_range,
            'max_features': self.max_features_range
        }
        best_params_dict = {}
        for c in clusters:
            mask_3d = (cluster_da == c).expand_dims({'T': y_train_std['T']})
            X_stacked_c = X_train_std.where(mask_3d).stack(sample=('T', 'Y', 'X')).transpose('sample', 'M').values
            y_stacked_c = y_train_std.where(mask_3d).stack(sample=('T', 'Y', 'X')).values.ravel()
            nan_mask_c = np.any(~np.isfinite(X_stacked_c), axis=1) | ~np.isfinite(y_stacked_c)
            X_clean_c = X_stacked_c[~nan_mask_c]
            y_clean_c = y_stacked_c[~nan_mask_c]
            if len(X_clean_c) == 0:
                continue
            # Initialize RandomForestRegressor base model
            model = RandomForestRegressor(random_state=self.random_state, n_jobs=-1)
            # Randomized search
            random_search = RandomizedSearchCV(
                model, param_distributions=param_dist, n_iter=self.n_iter_search,
                cv=self.cv_folds, scoring='neg_mean_squared_error',
                random_state=self.random_state, error_score=np.nan
            )
            random_search.fit(X_clean_c, y_clean_c)
            best_params_dict[c] = random_search.best_params_
        return best_params_dict, cluster_da

    def compute_model(self, X_train, y_train, X_test, y_test, best_params=None, cluster_da=None):
        """
        Compute deterministic hindcast using the RandomForestRegressor model with injected hyperparameters for each zone.
        Parameters
        ----------
        X_train : xarray.DataArray
            Training predictor data with dimensions (T, M, Y, X).
        y_train : xarray.DataArray
            Training predictand data with dimensions (T, Y, X).
        X_test : xarray.DataArray
            Testing predictor data with dimensions (T, M, Y, X).
        y_test : xarray.DataArray
            Testing predictand data with dimensions (T, Y, X).
        clim_year_start : int
            Start year of the climatology period.
        clim_year_end : int
            End year of the climatology period.
        best_params : dict, optional
            Pre-computed best hyperparameters per cluster. If None, computes internally.
        cluster_da : xarray.DataArray, optional
            Pre-computed cluster labels. If None, computes internally.
        Returns
        -------
        predicted_da : xarray.DataArray
            Deterministic hindcast with dimensions (T, Y, X).
        """
        # Standardize inputs
        X_train_std = X_train
        y_train_std = y_train
        X_test_std = X_test
        y_test_std = y_test
        # Extract coordinate variables from X_test
        time = X_test_std['T']
        lat = X_test_std['Y']
        lon = X_test_std['X']
        n_time = len(time)
        n_lat = len(lat)
        n_lon = len(lon)
        # Use provided best_params and cluster_da or compute if None
        if best_params is None:
            best_params, cluster_da = self.compute_hyperparameters(X_train, y_train, clim_year_start, clim_year_end)
        # Initialize predictions array
        predictions = np.full((n_time, n_lat, n_lon), np.nan)
        self.rf = {}  # Dictionary to store models per cluster
        for c in range(self.n_clusters):
            if c not in best_params:
                continue
            bp = best_params[c]
            # Mask for this cluster
            mask_3d_train = (cluster_da == c).expand_dims({'T': X_train_std['T']})
            mask_3d_test = (cluster_da == c).expand_dims({'T': X_test_std['T']})
            # Stack training data for cluster
            X_train_stacked_c = X_train_std.where(mask_3d_train).stack(sample=('T', 'Y', 'X')).transpose('sample', 'M').values
            y_train_stacked_c = y_train_std.where(mask_3d_train).stack(sample=('T', 'Y', 'X')).values.ravel()
            train_nan_mask = np.any(~np.isfinite(X_train_stacked_c), axis=1) | ~np.isfinite(y_train_stacked_c)
            X_train_clean_c = X_train_stacked_c[~train_nan_mask]
            y_train_clean_c = y_train_stacked_c[~train_nan_mask]
            # Stack testing data for cluster
            X_test_stacked_c = X_test_std.where(mask_3d_test).stack(sample=('T', 'Y', 'X')).transpose('sample', 'M').values
            y_test_stacked_c = y_test_std.where(mask_3d_test).stack(sample=('T', 'Y', 'X')).values.ravel()
            test_nan_mask = np.any(~np.isfinite(X_test_stacked_c), axis=1) | ~np.isfinite(y_test_stacked_c)
            X_test_clean_c = X_test_stacked_c[~test_nan_mask]
            # Initialize the model with best parameters for this cluster
            rf_c = RandomForestRegressor(
                n_estimators=bp['n_estimators'],
                max_depth=bp['max_depth'],
                min_samples_split=bp['min_samples_split'],
                min_samples_leaf=bp['min_samples_leaf'],
                max_features=bp['max_features'],
                random_state=self.random_state,
                n_jobs=-1
            )
            # Fit and predict
            rf_c.fit(X_train_clean_c, y_train_clean_c)
            self.rf[c] = rf_c
            y_pred_c = rf_c.predict(X_test_clean_c)
            # Reconstruct predictions for this cluster
            result_c = np.full(len(y_test_stacked_c), np.nan)
            result_c[~test_nan_mask] = y_pred_c
            pred_c_reshaped = result_c.reshape(n_time, n_lat, n_lon)
            # Fill in the predictions array
            predictions = np.where(np.isnan(predictions), pred_c_reshaped, predictions)
        predicted_da = xr.DataArray(
            data=predictions,
            coords={'T': time, 'Y': lat, 'X': lon},
            dims=['T', 'Y', 'X']
        )
        return predicted_da

    # ------------------ Probability Calculation Methods ------------------
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
            pred_prob[0, :] = t.cdf(first_t, df=dof)
            pred_prob[1, :] = t.cdf(second_t, df=dof) - t.cdf(first_t, df=dof)
            pred_prob[2, :] = 1 - t.cdf(second_t, df=dof)
        return pred_prob

    @staticmethod
    def calculate_tercile_probabilities_weibull_min(best_guess, error_variance, first_tercile, second_tercile, dof):
        """
        Weibull minimum-based method.
        """
        n_time = len(best_guess)
        pred_prob = np.empty((3, n_time))
        if np.all(np.isnan(best_guess)):
            pred_prob[:] = np.nan
        else:
            error_std = np.sqrt(error_variance)
            pred_prob[0, :] = weibull_min.cdf(first_tercile, c=dof, loc=best_guess, scale=error_std)
            pred_prob[1, :] = weibull_min.cdf(second_tercile, c=dof, loc=best_guess, scale=error_std) - \
                              weibull_min.cdf(first_tercile, c=dof, loc=best_guess, scale=error_std)
            pred_prob[2, :] = 1 - weibull_min.cdf(second_tercile, c=dof, loc=best_guess, scale=error_std)
        return pred_prob

    @staticmethod
    def calculate_tercile_probabilities_gamma(best_guess, error_variance, T1, T2, dof=None):
        """Gamma-distribution based method."""
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

    @staticmethod
    def calculate_tercile_probabilities_normal(best_guess, error_variance, first_tercile, second_tercile):
        """Normal-distribution based method."""
        n_time = len(best_guess)
        pred_prob = np.empty((3, n_time))
        if np.all(np.isnan(best_guess)):
            pred_prob[:] = np.nan
        else:
            error_std = np.sqrt(error_variance)
            pred_prob[0, :] = norm.cdf(first_tercile, loc=best_guess, scale=error_std)
            pred_prob[1, :] = norm.cdf(second_tercile, loc=best_guess, scale=error_std) - norm.cdf(first_tercile, loc=best_guess, scale=error_std)
            pred_prob[2, :] = 1 - norm.cdf(second_tercile, loc=best_guess, scale=error_std)
        return pred_prob

    @staticmethod
    def calculate_tercile_probabilities_lognormal(best_guess, error_variance, first_tercile, second_tercile):
        """Lognormal-distribution based method."""
        n_time = len(best_guess)
        pred_prob = np.empty((3, n_time))
        if np.any(np.isnan(best_guess)) or np.any(np.isnan(error_variance)):
            pred_prob[:] = np.nan
            return pred_prob
        sigma = np.sqrt(np.log(1 + error_variance / (best_guess**2)))
        mu = np.log(best_guess) - sigma**2 / 2
        pred_prob[0, :] = lognorm.cdf(first_tercile, s=sigma, scale=np.exp(mu))
        pred_prob[1, :] = lognorm.cdf(second_tercile, s=sigma, scale=np.exp(mu)) - lognorm.cdf(first_tercile, s=sigma, scale=np.exp(mu))
        pred_prob[2, :] = 1 - lognorm.cdf(second_tercile, s=sigma, scale=np.exp(mu))
        return pred_prob

    def compute_prob(self, Predictant, clim_year_start, clim_year_end, hindcast_det):
        """
        Compute tercile probabilities using the chosen distribution method.
        Predictant is expected to be an xarray DataArray with dims (T, Y, X).
        """
        if "M" in Predictant.coords:
            Predictant = Predictant.isel(M=0).drop_vars('M').squeeze()
        mask = xr.where(~np.isnan(Predictant.isel(T=0)), 1, np.nan).drop_vars(['T']).squeeze().to_numpy()
        Predictant = Predictant.transpose('T', 'Y', 'X')
        index_start = Predictant.get_index("T").get_loc(str(clim_year_start)).start
        index_end = Predictant.get_index("T").get_loc(str(clim_year_end)).stop
        rainfall_for_tercile = Predictant.isel(T=slice(index_start, index_end))
        terciles = rainfall_for_tercile.quantile([0.33, 0.67], dim='T')
        error_variance = (Predictant - hindcast_det).var(dim='T')
        dof = len(Predictant.get_index("T")) - 2
        if self.dist_method == "t":
            calc_func = self.calculate_tercile_probabilities
            hindcast_prob = xr.apply_ufunc(
                calc_func,
                hindcast_det,
                error_variance,
                terciles.isel(quantile=0).drop_vars('quantile'),
                terciles.isel(quantile=1).drop_vars('quantile'),
                input_core_dims=[('T',), (), (), ()],
                vectorize=True,
                kwargs={'dof': dof},
                dask='parallelized',
                output_core_dims=[('probability', 'T')],
                output_dtypes=['float'],
                dask_gufunc_kwargs={'output_sizes': {'probability': 3}, "allow_rechunk": True}
            )
        elif self.dist_method == "weibull_min":
            calc_func = self.calculate_tercile_probabilities_weibull_min
            hindcast_prob = xr.apply_ufunc(
                calc_func,
                hindcast_det,
                error_variance,
                terciles.isel(quantile=0).drop_vars('quantile'),
                terciles.isel(quantile=1).drop_vars('quantile'),
                input_core_dims=[('T',), (), (), ()],
                vectorize=True,
                kwargs={'dof': dof},
                dask='parallelized',
                output_core_dims=[('probability', 'T')],
                output_dtypes=['float'],
                dask_gufunc_kwargs={'output_sizes': {'probability': 3}, "allow_rechunk": True}
            )
        elif self.dist_method == "gamma":
            calc_func = self.calculate_tercile_probabilities_gamma
            hindcast_prob = xr.apply_ufunc(
                calc_func,
                hindcast_det,
                error_variance,
                terciles.isel(quantile=0).drop_vars('quantile'),
                terciles.isel(quantile=1).drop_vars('quantile'),
                input_core_dims=[('T',), (), (), ()],
                vectorize=True,
                dask='parallelized',
                output_core_dims=[('probability', 'T')],
                output_dtypes=['float'],
                dask_gufunc_kwargs={'output_sizes': {'probability': 3}, "allow_rechunk": True}
            )
        elif self.dist_method == "nonparam":
            calc_func = self.calculate_tercile_probabilities_nonparametric
            error_samples = Predictant - hindcast_det
            hindcast_prob = xr.apply_ufunc(
                calc_func,
                hindcast_det,
                error_samples,
                terciles.isel(quantile=0).drop_vars('quantile'),
                terciles.isel(quantile=1).drop_vars('quantile'),
                input_core_dims=[('T',), ('T',), (), ()],
                output_core_dims=[('probability', 'T')],
                vectorize=True,
                dask='parallelized',
                output_dtypes=[float],
                dask_gufunc_kwargs={'output_sizes': {'probability': 3}, "allow_rechunk": True}
            )
        elif self.dist_method == "normal":
            calc_func = self.calculate_tercile_probabilities_normal
            hindcast_prob = xr.apply_ufunc(
                calc_func,
                hindcast_det,
                error_variance,
                terciles.isel(quantile=0).drop_vars('quantile'),
                terciles.isel(quantile=1).drop_vars('quantile'),
                input_core_dims=[('T',), (), (), ()],
                vectorize=True,
                dask='parallelized',
                output_core_dims=[('probability', 'T')],
                output_dtypes=['float'],
                dask_gufunc_kwargs={'output_sizes': {'probability': 3}, "allow_rechunk": True}
            )
        elif self.dist_method == "lognormal":
            calc_func = self.calculate_tercile_probabilities_lognormal
            hindcast_prob = xr.apply_ufunc(
                calc_func,
                hindcast_det,
                error_variance,
                terciles.isel(quantile=0).drop_vars('quantile'),
                terciles.isel(quantile=1).drop_vars('quantile'),
                input_core_dims=[('T',), (), (), ()],
                vectorize=True,
                dask='parallelized',
                output_core_dims=[('probability', 'T')],
                output_dtypes=['float'],
                dask_gufunc_kwargs={'output_sizes': {'probability': 3}, "allow_rechunk": True}
            )
        else:
            raise ValueError(f"Invalid dist_method: {self.dist_method}.")
        hindcast_prob = hindcast_prob.assign_coords(probability=('probability', ['PB', 'PN', 'PA']))
        return hindcast_prob.transpose('probability', 'T', 'Y', 'X')

    @staticmethod
    def _reshape_and_filter_data(da):
        """
        Helper: stack the DataArray from (T, Y, X[, M]) to (n_samples, n_features)
        and remove rows containing NaNs.
        """
        da_stacked = da.stack(sample=('T', 'Y', 'X'))
        if 'M' in da.dims:
            da_stacked = da_stacked.transpose('sample', 'M')
        else:
            da_stacked = da_stacked.transpose('sample')
        da_values = da_stacked.values
        nan_mask = np.any(np.isnan(da_values), axis=1)
        return da_values[~nan_mask], nan_mask, da_values

    def forecast(self, Predictant, clim_year_start, clim_year_end, hindcast_det, hindcast_det_cross, Predictor_for_year, best_params=None, cluster_da=None):
        """
        Forecast method using a single Random Forest model with optimized hyperparameters.
        Parameters
        ----------
        Predictant : xarray.DataArray
            Observed predictand data with dimensions (T, Y, X) or (T, M, Y, X).
        clim_year_start : int or str
            Start year of the climatology period.
        clim_year_end : int or str
            End year of the climatology period.
        hindcast_det : xarray.DataArray
            Deterministic hindcast data for training with dimensions (T, M, Y, X).
        hindcast_det_cross : xarray.DataArray
            Deterministic hindcast data for error estimation with dimensions (T, Y, X).
        Predictor_for_year : xarray.DataArray
            Predictor data for the target year with dimensions (T, M, Y, X).
        best_params : dict, optional
            Pre-computed best hyperparameters. If None, computes internally.
        cluster_da : xarray.DataArray, optional
            Pre-computed cluster labels. If None, computes internally.
        Returns
        -------
        forecast_det : xarray.DataArray
            Deterministic forecast with dimensions (T, Y, X).
        forecast_prob : xarray.DataArray
            Tercile probabilities with dimensions (probability, T, Y, X).
        """
        if "M" in Predictant.coords:
            Predictant_no_m = Predictant.isel(M=0).drop_vars('M').squeeze()
        else:
            Predictant_no_m = Predictant
        mask = xr.where(~np.isnan(Predictant_no_m.isel(T=0)), 1, np.nan).drop_vars(['T']).squeeze().to_numpy()
        # Standardize Predictor_for_year using hindcast climatology
        mean_val = hindcast_det.sel(T=slice(str(clim_year_start), str(clim_year_end))).mean(dim='T')
        std_val = hindcast_det.sel(T=slice(str(clim_year_start), str(clim_year_end))).std(dim='T')
        Predictor_for_year_st = (Predictor_for_year - mean_val) / std_val
        hindcast_det_st = standardize_timeseries(hindcast_det, clim_year_start, clim_year_end)
        Predictant_st = standardize_timeseries(Predictant_no_m, clim_year_start, clim_year_end)
        hindcast_det_st['T'] = Predictant_st['T']
        # Extract coordinates from X_test
        time = Predictor_for_year_st['T']
        lat = Predictor_for_year_st['Y']
        lon = Predictor_for_year_st['X']
        n_time = len(time)
        n_lat = len(lat)
        n_lon = len(lon)
        # Use provided best_params and cluster_da or compute if None
        if best_params is None:
            best_params, cluster_da = self.compute_hyperparameters(hindcast_det, Predictant_no_m, clim_year_start, clim_year_end)
        # Initialize predictions array
        predictions = np.full((n_time, n_lat, n_lon), np.nan)
        self.rf = {}  # Dictionary to store models per cluster
        for c in range(self.n_clusters):
            if c not in best_params:
                continue
            bp = best_params[c]
            # Mask for this cluster
            mask_3d_train = (cluster_da == c).expand_dims({'T': hindcast_det_st['T']})
            mask_3d_test = (cluster_da == c).expand_dims({'T': Predictor_for_year_st['T']})
            # Stack training data for cluster
            X_train_stacked_c = hindcast_det_st.where(mask_3d_train).stack(sample=('T', 'Y', 'X')).transpose('sample', 'M').values
            y_train_stacked_c = Predictant_st.where(mask_3d_train).stack(sample=('T', 'Y', 'X')).values.ravel()
            train_nan_mask = np.any(~np.isfinite(X_train_stacked_c), axis=1) | ~np.isfinite(y_train_stacked_c)
            X_train_clean_c = X_train_stacked_c[~train_nan_mask]
            y_train_clean_c = y_train_stacked_c[~train_nan_mask]
            # Stack testing data for cluster
            X_test_stacked_c = Predictor_for_year_st.where(mask_3d_test).stack(sample=('T', 'Y', 'X')).transpose('sample', 'M').values
            test_nan_mask = np.any(~np.isfinite(X_test_stacked_c), axis=1)
            X_test_clean_c = X_test_stacked_c[~test_nan_mask]
            # Initialize the model with best parameters for this cluster
            rf_c = RandomForestRegressor(
                n_estimators=bp['n_estimators'],
                max_depth=bp['max_depth'],
                min_samples_split=bp['min_samples_split'],
                min_samples_leaf=bp['min_samples_leaf'],
                max_features=bp['max_features'],
                random_state=self.random_state,
                n_jobs=-1
            )
            # Fit and predict
            rf_c.fit(X_train_clean_c, y_train_clean_c)
            self.rf[c] = rf_c
            y_pred_c = rf_c.predict(X_test_clean_c)
            # Reconstruct predictions for this cluster
            result_c = np.full(len(X_test_stacked_c), np.nan)
            result_c[~test_nan_mask] = y_pred_c
            pred_c_reshaped = result_c.reshape(n_time, n_lat, n_lon)
            # Fill in the predictions array
            predictions = np.where(np.isnan(predictions), pred_c_reshaped, predictions)
        result_da = xr.DataArray(
            data=predictions,
            coords={'T': time, 'Y': lat, 'X': lon},
            dims=['T', 'Y', 'X']
        ) * mask
        result_da = reverse_standardize(result_da, Predictant_no_m, clim_year_start, clim_year_end)
        year = Predictor_for_year.coords['T'].values.astype('datetime64[Y]').astype(int)[0] + 1970
        T_value_1 = Predictant_no_m.isel(T=0).coords['T'].values
        month_1 = T_value_1.astype('datetime64[M]').astype(int) % 12 + 1
        new_T_value = np.datetime64(f"{year}-{month_1:02d}-01")
        result_da = result_da.assign_coords(T=xr.DataArray([new_T_value], dims=["T"]))
        result_da['T'] = result_da['T'].astype('datetime64[ns]')
        # Compute tercile probabilities
        index_start = Predictant_no_m.get_index("T").get_loc(str(clim_year_start)).start
        index_end = Predictant_no_m.get_index("T").get_loc(str(clim_year_end)).stop
        rainfall_for_tercile = Predictant_no_m.isel(T=slice(index_start, index_end))
        terciles = rainfall_for_tercile.quantile([0.33, 0.67], dim='T')
        error_variance = (Predictant_no_m - hindcast_det_cross).var(dim='T')
        dof = len(Predictant_no_m.get_index("T")) - 2
        if self.dist_method == "t":
            calc_func = self.calculate_tercile_probabilities
            hindcast_prob = xr.apply_ufunc(
                calc_func,
                result_da,
                error_variance,
                terciles.isel(quantile=0).drop_vars('quantile'),
                terciles.isel(quantile=1).drop_vars('quantile'),
                input_core_dims=[('T',), (), (), ()],
                vectorize=True,
                kwargs={'dof': dof},
                dask='parallelized',
                output_core_dims=[('probability', 'T')],
                output_dtypes=['float'],
                dask_gufunc_kwargs={'output_sizes': {'probability': 3}, "allow_rechunk": True}
            )
        elif self.dist_method == "weibull_min":
            calc_func = self.calculate_tercile_probabilities_weibull_min
            hindcast_prob = xr.apply_ufunc(
                calc_func,
                result_da,
                error_variance,
                terciles.isel(quantile=0).drop_vars('quantile'),
                terciles.isel(quantile=1).drop_vars('quantile'),
                input_core_dims=[('T',), (), (), ()],
                vectorize=True,
                kwargs={'dof': dof},
                dask='parallelized',
                output_core_dims=[('probability', 'T')],
                output_dtypes=['float'],
                dask_gufunc_kwargs={'output_sizes': {'probability': 3}, "allow_rechunk": True}
            )
        elif self.dist_method == "gamma":
            calc_func = self.calculate_tercile_probabilities_gamma
            hindcast_prob = xr.apply_ufunc(
                calc_func,
                result_da,
                error_variance,
                terciles.isel(quantile=0).drop_vars('quantile'),
                terciles.isel(quantile=1).drop_vars('quantile'),
                input_core_dims=[('T',), (), (), ()],
                vectorize=True,
                dask='parallelized',
                output_core_dims=[('probability', 'T')],
                output_dtypes=['float'],
                dask_gufunc_kwargs={'output_sizes': {'probability': 3}, "allow_rechunk": True}
            )
        elif self.dist_method == "normal":
            calc_func = self.calculate_tercile_probabilities_normal
            hindcast_prob = xr.apply_ufunc(
                calc_func,
                result_da,
                error_variance,
                terciles.isel(quantile=0).drop_vars('quantile'),
                terciles.isel(quantile=1).drop_vars('quantile'),
                input_core_dims=[('T',), (), (), ()],
                vectorize=True,
                dask='parallelized',
                output_core_dims=[('probability', 'T')],
                output_dtypes=['float'],
                dask_gufunc_kwargs={'output_sizes': {'probability': 3}, "allow_rechunk": True}
            )
        elif self.dist_method == "lognormal":
            calc_func = self.calculate_tercile_probabilities_lognormal
            hindcast_prob = xr.apply_ufunc(
                calc_func,
                result_da,
                error_variance,
                terciles.isel(quantile=0).drop_vars('quantile'),
                terciles.isel(quantile=1).drop_vars('quantile'),
                input_core_dims=[('T',), (), (), ()],
                vectorize=True,
                dask='parallelized',
                output_core_dims=[('probability', 'T')],
                output_dtypes=['float'],
                dask_gufunc_kwargs={'output_sizes': {'probability': 3}, "allow_rechunk": True}
            )
        elif self.dist_method == "nonparam":
            calc_func = self.calculate_tercile_probabilities_nonparametric
            error_samples = Predictant_no_m - hindcast_det_cross
            error_samples = error_samples.rename({'T': 'S'})
            hindcast_prob = xr.apply_ufunc(
                calc_func,
                result_da,
                error_samples,
                terciles.isel(quantile=0).drop_vars('quantile'),
                terciles.isel(quantile=1).drop_vars('quantile'),
                input_core_dims=[('T',), ('S',), (), ()],
                output_core_dims=[('probability', 'T')],
                vectorize=True,
                dask='parallelized',
                output_dtypes=[float],
                dask_gufunc_kwargs={'output_sizes': {'probability': 3}, "allow_rechunk": True}
            )
        else:
            raise ValueError(f"Invalid dist_method: {self.dist_method}.")
        hindcast_prob = hindcast_prob.assign_coords(probability=('probability', ['PB', 'PN', 'PA']))
        return result_da * mask, mask * hindcast_prob.transpose('probability', 'T', 'Y', 'X')










################################## ALL BELOW is in experimentation ####################################################

class WAS_mme_Stacking:
    """
    Stacking ensemble for Multi-Model Ensemble (MME) forecasting using provided base models.
    This class stacks predictions from base models (hpELM, MLP, XGBoost, RF) and uses a meta-learner 
    (ridge, lasso, elasticnet, or linear regression) for final predictions.
    Supports deterministic and probabilistic (tercile) forecasting.
    Includes hyperparameter tuning for the meta-learner.
    Parameters
    ----------
    meta_learner_type : str, optional
        Type of meta-learner: 'ridge', 'lasso', 'elasticnet', or 'linear' (default is 'ridge').
    alpha_range : list of float, optional
        Range of alpha values for tuning ridge, lasso, elasticnet (default is [0.1, 1.0, 10.0, 100.0]).
    l1_ratio_range : list of float, optional
        Range of l1_ratio values for tuning elasticnet (default is [0.1, 0.5, 0.9]).
    random_state : int, optional
        Seed for reproducibility (default is 42).
    dist_method : str, optional
        Distribution method for tercile probabilities ('t', 'gamma', etc.) (default is 'gamma').
    stacking_cv : int, optional
        Number of folds for cross-validation to generate out-of-fold predictions (default is 3).
    meta_cv_folds : int, optional
        Number of cross-validation folds for meta-learner tuning (default is 3).
    meta_n_iter_search : int, optional
        Number of iterations for randomized search in meta-learner tuning (default is 10).
    n_clusters : int, optional
        Number of clusters for homogenized zones (default is 4).
    """
    def __init__(self,
                 meta_learner_type='ridge',
                 alpha_range=[0.1, 1.0, 10.0, 100.0],
                 l1_ratio_range=[0.1, 0.5, 0.9],
                 random_state=42,
                 dist_method="gamma",
                 stacking_cv=3,
                 meta_cv_folds=3,
                 meta_n_iter_search=10,
                 n_clusters=4):
        self.meta_learner_type = meta_learner_type
        self.alpha_range = alpha_range
        self.l1_ratio_range = l1_ratio_range
        self.random_state = random_state
        self.dist_method = dist_method
        self.stacking_cv = stacking_cv
        self.meta_cv_folds = meta_cv_folds
        self.meta_n_iter_search = meta_n_iter_search
        self.n_clusters = n_clusters
        self.base_models = [
            WAS_mme_hpELM(random_state=random_state, dist_method=dist_method, n_clusters=n_clusters),
            WAS_mme_MLP(random_state=random_state, dist_method=dist_method, n_clusters=n_clusters),
            WAS_mme_XGBoosting(random_state=random_state, dist_method=dist_method, n_clusters=n_clusters),
            WAS_mme_RF(random_state=random_state, dist_method=dist_method, n_clusters=n_clusters)
        ]
        self.meta_learners = None

    def compute_hyperparameters(self, Predictors, Predictand, clim_year_start, clim_year_end):
        best_params_list = []
        cluster_da = None
        for base in self.base_models:
            bp, cd = base.compute_hyperparameters(Predictors, Predictand, clim_year_start, clim_year_end)
            best_params_list.append(bp)
            if cluster_da is None:
                cluster_da = cd
        return best_params_list, cluster_da

    def _get_oof_predictions(self, X, y, best_params_list, cluster_da):
        kf = KFold(n_splits=self.stacking_cv, shuffle=False, random_state=self.random_state)
        n_t = len(X['T'])
        oof_preds = [xr.full_like(y, np.nan) for _ in self.base_models]
        for i, base in enumerate(self.base_models):
            bp = best_params_list[i]
            for train_idx, val_idx in kf.split(range(n_t)):
                X_train_fold = X.isel(T=train_idx)
                y_train_fold = y.isel(T=train_idx)
                X_val_fold = X.isel(T=val_idx)
                y_val_fold = y.isel(T=val_idx)
                pred_val = base.compute_model(X_train_fold, y_train_fold, X_val_fold, y_val_fold, best_params=bp, cluster_da=cluster_da)
                oof_preds[i].loc[dict(T=X_val_fold['T'])] = pred_val
        return oof_preds

    def compute_model(self, X_train, y_train, X_test, y_test, best_params=None, cluster_da=None):
        if best_params is None:
            best_params, cluster_da = self.compute_hyperparameters(X_train, y_train, clim_year_start, clim_year_end)  # Note: clim_year_start/end need to be passed or assumed
        best_params_list = best_params  # Assuming best_params is list
        oof_preds = self._get_oof_predictions(X_train, y_train, best_params_list, cluster_da)
        self.meta_learners = {}
        for c in range(self.n_clusters):
            mask_3d = (cluster_da == c).expand_dims({'T': y_train['T']})
            X_meta_stacked = []
            for oof_base in oof_preds:
                stacked = oof_base.where(mask_3d).stack(sample=('T', 'Y', 'X')).values
                X_meta_stacked.append(stacked)
            X_meta = np.column_stack(X_meta_stacked)
            y_meta = y_train.where(mask_3d).stack(sample=('T', 'Y', 'X')).values
            nan_mask = np.any(~np.isfinite(X_meta), axis=1) | ~np.isfinite(y_meta)
            if np.all(nan_mask):
                continue
            X_meta_clean = X_meta[~nan_mask]
            y_meta_clean = y_meta[~nan_mask]
            if self.meta_learner_type == 'linear':
                meta = LinearRegression()
                meta.fit(X_meta_clean, y_meta_clean)
            else:
                if self.meta_learner_type == 'ridge':
                    meta_base = Ridge()
                    param_dist = {'alpha': self.alpha_range}
                elif self.meta_learner_type == 'lasso':
                    meta_base = Lasso()
                    param_dist = {'alpha': self.alpha_range}
                elif self.meta_learner_type == 'elasticnet':
                    meta_base = ElasticNet()
                    param_dist = {'alpha': self.alpha_range, 'l1_ratio': self.l1_ratio_range}
                else:
                    raise ValueError(f"Invalid meta_learner_type: {self.meta_learner_type}")
                random_search = RandomizedSearchCV(
                    meta_base, param_distributions=param_dist, n_iter=self.meta_n_iter_search,
                    cv=self.meta_cv_folds, scoring='neg_mean_squared_error',
                    random_state=self.random_state, error_score=np.nan
                )
                random_search.fit(X_meta_clean, y_meta_clean)
                meta = random_search.best_estimator_
            self.meta_learners[c] = meta
        base_test_preds = []
        for i, base in enumerate(self.base_models):
            bp = best_params_list[i]
            pred_test = base.compute_model(X_train, y_train, X_test, y_test, best_params=bp, cluster_da=cluster_da)
            base_test_preds.append(pred_test)
        time = X_test['T']
        lat = X_test['Y']
        lon = X_test['X']
        n_time = len(time)
        n_lat = len(lat)
        n_lon = len(lon)
        predictions = np.full((n_time, n_lat, n_lon), np.nan)
        for c in range(self.n_clusters):
            if c not in self.meta_learners:
                continue
            meta = self.meta_learners[c]
            mask_3d_test = (cluster_da == c).expand_dims({'T': X_test['T']})
            X_meta_test_stacked = []
            for pred_base in base_test_preds:
                stacked = pred_base.where(mask_3d_test).stack(sample=('T', 'Y', 'X')).values
                X_meta_test_stacked.append(stacked)
            X_meta_test = np.column_stack(X_meta_test_stacked)
            nan_mask_test = np.any(~np.isfinite(X_meta_test), axis=1)
            if np.all(nan_mask_test):
                continue
            X_meta_test_clean = X_meta_test[~nan_mask_test]
            y_pred_clean = meta.predict(X_meta_test_clean)
            full_y_pred = np.full(X_meta_test.shape[0], np.nan)
            full_y_pred[~nan_mask_test] = y_pred_clean
            pred_c_reshaped = full_y_pred.reshape(n_time, n_lat, n_lon)
            predictions = np.where(np.isnan(predictions), pred_c_reshaped, predictions)
        predicted_da = xr.DataArray(
            data=predictions,
            coords={'T': time, 'Y': lat, 'X': lon},
            dims=['T', 'Y', 'X']
        )
        return predicted_da

    # Probability calculation methods (copied from WAS_mme_hpELM)
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
            pred_prob[0, :] = t.cdf(first_t, df=dof)
            pred_prob[1, :] = t.cdf(second_t, df=dof) - t.cdf(first_t, df=dof)
            pred_prob[2, :] = 1 - t.cdf(second_t, df=dof)
        return pred_prob

    @staticmethod
    def calculate_tercile_probabilities_weibull_min(best_guess, error_variance, first_tercile, second_tercile, dof):
        n_time = len(best_guess)
        pred_prob = np.empty((3, n_time))
        if np.all(np.isnan(best_guess)):
            pred_prob[:] = np.nan
        else:
            error_std = np.sqrt(error_variance)
            pred_prob[0, :] = weibull_min.cdf(first_tercile, c=dof, loc=best_guess, scale=error_std)
            pred_prob[1, :] = weibull_min.cdf(second_tercile, c=dof, loc=best_guess, scale=error_std) - \
                              weibull_min.cdf(first_tercile, c=dof, loc=best_guess, scale=error_std)
            pred_prob[2, :] = 1 - weibull_min.cdf(second_tercile, c=dof, loc=best_guess, scale=error_std)
        return pred_prob

    @staticmethod
    def calculate_tercile_probabilities_gamma(best_guess, error_variance, T1, T2, dof=None):
        n_time = len(best_guess)
        pred_prob = np.empty((3, n_time), dtype=float)
        if np.any(np.isnan(best_guess)) or np.any(np.isnan(error_variance)):
            pred_prob[:] = np.nan
            return pred_prob
        best_guess = np.asarray(best_guess, dtype=float)
        error_variance = np.asarray(error_variance, dtype=float)
        T1 = np.asarray(T1, dtype=float)
        T2 = np.asarray(T2, dtype=float)
        alpha = (best_guess ** 2) / error_variance
        theta = error_variance / best_guess
        cdf_t1 = gamma.cdf(T1, a=alpha, scale=theta)
        cdf_t2 = gamma.cdf(T2, a=alpha, scale=theta)
        pred_prob[0, :] = cdf_t1
        pred_prob[1, :] = cdf_t2 - cdf_t1
        pred_prob[2, :] = 1.0 - cdf_t2
        return pred_prob

    @staticmethod
    def calculate_tercile_probabilities_nonparametric(best_guess, error_samples, first_tercile, second_tercile):
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
        n_time = len(best_guess)
        pred_prob = np.empty((3, n_time))
        if np.all(np.isnan(best_guess)):
            pred_prob[:] = np.nan
        else:
            error_std = np.sqrt(error_variance)
            pred_prob[0, :] = norm.cdf(first_tercile, loc=best_guess, scale=error_std)
            pred_prob[1, :] = norm.cdf(second_tercile, loc=best_guess, scale=error_std) - norm.cdf(first_tercile, loc=best_guess, scale=error_std)
            pred_prob[2, :] = 1 - norm.cdf(second_tercile, loc=best_guess, scale=error_std)
        return pred_prob

    @staticmethod
    def calculate_tercile_probabilities_lognormal(best_guess, error_variance, first_tercile, second_tercile):
        n_time = len(best_guess)
        pred_prob = np.empty((3, n_time))
        if np.any(np.isnan(best_guess)) or np.any(np.isnan(error_variance)):
            pred_prob[:] = np.nan
            return pred_prob
        sigma = np.sqrt(np.log(1 + error_variance / (best_guess ** 2)))
        mu = np.log(best_guess) - sigma ** 2 / 2
        pred_prob[0, :] = lognorm.cdf(first_tercile, s=sigma, scale=np.exp(mu))
        pred_prob[1, :] = lognorm.cdf(second_tercile, s=sigma, scale=np.exp(mu)) - lognorm.cdf(first_tercile, s=sigma, scale=np.exp(mu))
        pred_prob[2, :] = 1 - lognorm.cdf(second_tercile, s=sigma, scale=np.exp(mu))
        return pred_prob

    def compute_prob(self, Predictant, clim_year_start, clim_year_end, hindcast_det):
        if "M" in Predictant.coords:
            Predictant = Predictant.isel(M=0).drop_vars('M').squeeze()
        mask = xr.where(~np.isnan(Predictant.isel(T=0)), 1, np.nan).drop_vars(['T']).squeeze().to_numpy()
        Predictant = Predictant.transpose('T', 'Y', 'X')
        index_start = Predictant.get_index("T").get_loc(str(clim_year_start)).start
        index_end = Predictant.get_index("T").get_loc(str(clim_year_end)).stop
        rainfall_for_tercile = Predictant.isel(T=slice(index_start, index_end))
        terciles = rainfall_for_tercile.quantile([0.33, 0.67], dim='T')
        error_variance = (Predictant - hindcast_det).var(dim='T')
        dof = len(Predictant.get_index("T")) - 2
        if self.dist_method == "t":
            calc_func = self.calculate_tercile_probabilities
            kwargs = {'dof': dof}
            input_core_dims = [('T',), (), (), ()]
        elif self.dist_method == "weibull_min":
            calc_func = self.calculate_tercile_probabilities_weibull_min
            kwargs = {'dof': dof}
            input_core_dims = [('T',), (), (), ()]
        elif self.dist_method == "gamma":
            calc_func = self.calculate_tercile_probabilities_gamma
            kwargs = {}
            input_core_dims = [('T',), (), (), ()]
        elif self.dist_method == "nonparam":
            calc_func = self.calculate_tercile_probabilities_nonparametric
            error_samples = Predictant - hindcast_det
            kwargs = {}
            input_core_dims = [('T',), ('T',), (), ()]
            hindcast_det, error_variance, first, second = hindcast_det, error_samples, terciles.isel(quantile=0).drop_vars('quantile'), terciles.isel(quantile=1).drop_vars('quantile')
        elif self.dist_method == "normal":
            calc_func = self.calculate_tercile_probabilities_normal
            kwargs = {}
            input_core_dims = [('T',), (), (), ()]
        elif self.dist_method == "lognormal":
            calc_func = self.calculate_tercile_probabilities_lognormal
            kwargs = {}
            input_core_dims = [('T',), (), (), ()]
        else:
            raise ValueError(f"Invalid dist_method: {self.dist_method}.")
        hindcast_prob = xr.apply_ufunc(
            calc_func,
            hindcast_det,
            error_variance,
            terciles.isel(quantile=0).drop_vars('quantile'),
            terciles.isel(quantile=1).drop_vars('quantile'),
            input_core_dims=input_core_dims,
            vectorize=True,
            dask='parallelized',
            output_core_dims=[('probability', 'T')],
            output_dtypes=['float'],
            dask_gufunc_kwargs={'output_sizes': {'probability': 3}, 'allow_rechunk': True},
            **kwargs
        )
        hindcast_prob = hindcast_prob.assign_coords(probability=('probability', ['PB', 'PN', 'PA']))
        return hindcast_prob.transpose('probability', 'T', 'Y', 'X')

    def forecast(self, Predictant, clim_year_start, clim_year_end, hindcast_det, hindcast_det_cross, Predictor_for_year, best_params=None, cluster_da=None):
        if "M" in Predictant.coords:
            Predictant_no_m = Predictant.isel(M=0).drop_vars('M').squeeze()
        else:
            Predictant_no_m = Predictant
        mask = xr.where(~np.isnan(Predictant_no_m.isel(T=0)), 1, np.nan).drop_vars(['T']).squeeze().to_numpy()
        mean_val = hindcast_det.sel(T=slice(str(clim_year_start), str(clim_year_end))).mean(dim='T')
        std_val = hindcast_det.sel(T=slice(str(clim_year_start), str(clim_year_end))).std(dim='T')
        Predictor_for_year_st = (Predictor_for_year - mean_val) / std_val
        hindcast_det_st = standardize_timeseries(hindcast_det, clim_year_start, clim_year_end)
        Predictant_st = standardize_timeseries(Predictant_no_m, clim_year_start, clim_year_end)
        hindcast_det_st['T'] = Predictant_st['T']
        if best_params is None:
            best_params, cluster_da = self.compute_hyperparameters(hindcast_det, Predictant_no_m, clim_year_start, clim_year_end)
        best_params_list = best_params
        oof_preds = self._get_oof_predictions(hindcast_det_st, Predictant_st, best_params_list, cluster_da)
        self.meta_learners = {}
        for c in range(self.n_clusters):
            mask_3d = (cluster_da == c).expand_dims({'T': Predictant_st['T']})
            X_meta_stacked = []
            for oof_base in oof_preds:
                stacked = oof_base.where(mask_3d).stack(sample=('T', 'Y', 'X')).values
                X_meta_stacked.append(stacked)
            X_meta = np.column_stack(X_meta_stacked)
            y_meta = Predictant_st.where(mask_3d).stack(sample=('T', 'Y', 'X')).values
            nan_mask = np.any(~np.isfinite(X_meta), axis=1) | ~np.isfinite(y_meta)
            if np.all(nan_mask):
                continue
            X_meta_clean = X_meta[~nan_mask]
            y_meta_clean = y_meta[~nan_mask]
            if self.meta_learner_type == 'linear':
                meta = LinearRegression()
                meta.fit(X_meta_clean, y_meta_clean)
            else:
                if self.meta_learner_type == 'ridge':
                    meta_base = Ridge()
                    param_dist = {'alpha': self.alpha_range}
                elif self.meta_learner_type == 'lasso':
                    meta_base = Lasso()
                    param_dist = {'alpha': self.alpha_range}
                elif self.meta_learner_type == 'elasticnet':
                    meta_base = ElasticNet()
                    param_dist = {'alpha': self.alpha_range, 'l1_ratio': self.l1_ratio_range}
                else:
                    raise ValueError(f"Invalid meta_learner_type: {self.meta_learner_type}")
                random_search = RandomizedSearchCV(
                    meta_base, param_distributions=param_dist, n_iter=self.meta_n_iter_search,
                    cv=self.meta_cv_folds, scoring='neg_mean_squared_error',
                    random_state=self.random_state, error_score=np.nan
                )
                random_search.fit(X_meta_clean, y_meta_clean)
                meta = random_search.best_estimator_
            self.meta_learners[c] = meta
        base_forecast_std = []
        time = Predictor_for_year_st['T']
        n_time = len(time)
        y_test_dummy = xr.full_like(Predictant_st.isel(T=0), np.nan).expand_dims(T=time)
        for i, base in enumerate(self.base_models):
            bp = best_params_list[i]
            pred_base = base.compute_model(hindcast_det_st, Predictant_st, Predictor_for_year_st, y_test_dummy, best_params=bp, cluster_da=cluster_da)
            base_forecast_std.append(pred_base)
        lat = Predictor_for_year_st['Y']
        lon = Predictor_for_year_st['X']
        n_lat = len(lat)
        n_lon = len(lon)
        predictions = np.full((n_time, n_lat, n_lon), np.nan)
        for c in range(self.n_clusters):
            if c not in self.meta_learners:
                continue
            meta = self.meta_learners[c]
            mask_3d_test = (cluster_da == c).expand_dims({'T': time})
            X_meta_test_stacked = []
            for pred_base in base_forecast_std:
                stacked = pred_base.where(mask_3d_test).stack(sample=('T', 'Y', 'X')).values
                X_meta_test_stacked.append(stacked)
            X_meta_test = np.column_stack(X_meta_test_stacked)
            nan_mask_test = np.any(~np.isfinite(X_meta_test), axis=1)
            if np.all(nan_mask_test):
                continue
            X_meta_test_clean = X_meta_test[~nan_mask_test]
            y_pred_clean = meta.predict(X_meta_test_clean)
            full_y_pred = np.full(X_meta_test.shape[0], np.nan)
            full_y_pred[~nan_mask_test] = y_pred_clean
            pred_c_reshaped = full_y_pred.reshape(n_time, n_lat, n_lon)
            predictions = np.where(np.isnan(predictions), pred_c_reshaped, predictions)
        result_da = xr.DataArray(
            data=predictions,
            coords={'T': time, 'Y': lat, 'X': lon},
            dims=['T', 'Y', 'X']
        ) * mask
        result_da = reverse_standardize(result_da, Predictant_no_m, clim_year_start, clim_year_end)
        year = Predictor_for_year.coords['T'].values.astype('datetime64[Y]').astype(int)[0] + 1970
        T_value_1 = Predictant_no_m.isel(T=0).coords['T'].values
        month_1 = T_value_1.astype('datetime64[M]').astype(int) % 12 + 1
        new_T_value = np.datetime64(f"{year}-{month_1:02d}-01")
        result_da = result_da.assign_coords(T=xr.DataArray([new_T_value], dims=["T"]))
        result_da['T'] = result_da['T'].astype('datetime64[ns]')
        # Compute probabilities
        index_start = Predictant_no_m.get_index("T").get_loc(str(clim_year_start)).start
        index_end = Predictant_no_m.get_index("T").get_loc(str(clim_year_end)).stop
        rainfall_for_tercile = Predictant_no_m.isel(T=slice(index_start, index_end))
        terciles = rainfall_for_tercile.quantile([0.33, 0.67], dim='T')
        error_variance = (Predictant_no_m - hindcast_det_cross).var(dim='T')
        dof = len(Predictant_no_m.get_index("T")) - 2
        if self.dist_method == "t":
            calc_func = self.calculate_tercile_probabilities
            hindcast_prob = xr.apply_ufunc(
                calc_func,
                result_da,
                error_variance,
                terciles.isel(quantile=0).drop_vars('quantile'),
                terciles.isel(quantile=1).drop_vars('quantile'),
                input_core_dims=[('T',), (), (), ()],
                kwargs={'dof': dof},
                vectorize=True,
                dask='parallelized',
                output_core_dims=[('probability', 'T')],
                output_dtypes=['float'],
                dask_gufunc_kwargs={'output_sizes': {'probability': 3}, 'allow_rechunk': True}
            )
        elif self.dist_method == "weibull_min":
            calc_func = self.calculate_tercile_probabilities_weibull_min
            hindcast_prob = xr.apply_ufunc(
                calc_func,
                result_da,
                error_variance,
                terciles.isel(quantile=0).drop_vars('quantile'),
                terciles.isel(quantile=1).drop_vars('quantile'),
                input_core_dims=[('T',), (), (), ()],
                kwargs={'dof': dof},
                vectorize=True,
                dask='parallelized',
                output_core_dims=[('probability', 'T')],
                output_dtypes=['float'],
                dask_gufunc_kwargs={'output_sizes': {'probability': 3}, 'allow_rechunk': True}
            )
        elif self.dist_method == "gamma":
            calc_func = self.calculate_tercile_probabilities_gamma
            hindcast_prob = xr.apply_ufunc(
                calc_func,
                result_da,
                error_variance,
                terciles.isel(quantile=0).drop_vars('quantile'),
                terciles.isel(quantile=1).drop_vars('quantile'),
                input_core_dims=[('T',), (), (), ()],
                vectorize=True,
                dask='parallelized',
                output_core_dims=[('probability', 'T')],
                output_dtypes=['float'],
                dask_gufunc_kwargs={'output_sizes': {'probability': 3}, 'allow_rechunk': True}
            )
        elif self.dist_method == "normal":
            calc_func = self.calculate_tercile_probabilities_normal
            hindcast_prob = xr.apply_ufunc(
                calc_func,
                result_da,
                error_variance,
                terciles.isel(quantile=0).drop_vars('quantile'),
                terciles.isel(quantile=1).drop_vars('quantile'),
                input_core_dims=[('T',), (), (), ()],
                vectorize=True,
                dask='parallelized',
                output_core_dims=[('probability', 'T')],
                output_dtypes=['float'],
                dask_gufunc_kwargs={'output_sizes': {'probability': 3}, 'allow_rechunk': True}
            )
        elif self.dist_method == "lognormal":
            calc_func = self.calculate_tercile_probabilities_lognormal
            hindcast_prob = xr.apply_ufunc(
                calc_func,
                result_da,
                error_variance,
                terciles.isel(quantile=0).drop_vars('quantile'),
                terciles.isel(quantile=1).drop_vars('quantile'),
                input_core_dims=[('T',), (), (), ()],
                vectorize=True,
                dask='parallelized',
                output_core_dims=[('probability', 'T')],
                output_dtypes=['float'],
                dask_gufunc_kwargs={'output_sizes': {'probability': 3}, 'allow_rechunk': True}
            )
        elif self.dist_method == "nonparam":
            calc_func = self.calculate_tercile_probabilities_nonparametric
            error_samples = Predictant_no_m - hindcast_det_cross
            error_samples = error_samples.rename({'T': 'S'})
            hindcast_prob = xr.apply_ufunc(
                calc_func,
                result_da,
                error_samples,
                terciles.isel(quantile=0).drop_vars('quantile'),
                terciles.isel(quantile=1).drop_vars('quantile'),
                input_core_dims=[('T',), ('S',), (), ()],
                output_core_dims=[('probability', 'T')],
                vectorize=True,
                dask='parallelized',
                output_dtypes=[float],
                dask_gufunc_kwargs={'output_sizes': {'probability': 3}, 'allow_rechunk': True}
            )
        else:
            raise ValueError(f"Invalid dist_method: {self.dist_method}.")
        hindcast_prob = hindcast_prob.assign_coords(probability=('probability', ['PB', 'PN', 'PA']))
        return result_da * mask, mask * hindcast_prob.transpose('probability', 'T', 'Y', 'X')


class WAS_mme_Stacking_:
    """
    Stacking ensemble for Multi-Model Ensemble (MME) forecasting using provided base models.
    This class stacks predictions from base models (hpELM_, MLP_, XGBoost_, RF_) and uses a meta-learner 
    (ridge, lasso, elasticnet, or linear regression) for final predictions.
    Supports deterministic and probabilistic (tercile) forecasting.
    Includes hyperparameter tuning for the meta-learner.
    Parameters
    ----------
    meta_learner_type : str, optional
        Type of meta-learner: 'ridge', 'lasso', 'elasticnet', or 'linear' (default is 'ridge').
    alpha_range : list of float, optional
        Range of alpha values for tuning ridge, lasso, elasticnet (default is [0.1, 1.0, 10.0, 100.0]).
    l1_ratio_range : list of float, optional
        Range of l1_ratio values for tuning elasticnet (default is [0.1, 0.5, 0.9]).
    random_state : int, optional
        Seed for reproducibility (default is 42).
    dist_method : str, optional
        Distribution method for tercile probabilities ('t', 'gamma', etc.) (default is 'gamma').
    stacking_cv : int, optional
        Number of folds for cross-validation to generate out-of-fold predictions (default is 3).
    meta_cv_folds : int, optional
        Number of cross-validation folds for meta-learner tuning (default is 3).
    meta_n_iter_search : int, optional
        Number of iterations for randomized search in meta-learner tuning (default is 10).
    """
    def __init__(self,
                 meta_learner_type='ridge',
                 alpha_range=[0.1, 1.0, 10.0, 100.0],
                 l1_ratio_range=[0.1, 0.5, 0.9],
                 random_state=42,
                 dist_method="gamma",
                 stacking_cv=3,
                 meta_cv_folds=3,
                 meta_n_iter_search=10):
        self.meta_learner_type = meta_learner_type
        self.alpha_range = alpha_range
        self.l1_ratio_range = l1_ratio_range
        self.random_state = random_state
        self.dist_method = dist_method
        self.stacking_cv = stacking_cv
        self.meta_cv_folds = meta_cv_folds
        self.meta_n_iter_search = meta_n_iter_search
        self.base_models = [
            WAS_mme_hpELM_(random_state=random_state, dist_method=dist_method),
            WAS_mme_MLP_(random_state=random_state, dist_method=dist_method),
            WAS_mme_XGBoosting_(random_state=random_state, dist_method=dist_method),
            WAS_mme_RF_(random_state=random_state, dist_method=dist_method)
        ]
        self.meta_learner = None

    def compute_hyperparameters(self, Predictors, Predictand, clim_year_start, clim_year_end):
        best_params_list = []
        for base in self.base_models:
            bp = base.compute_hyperparameters(Predictors, Predictand, clim_year_start, clim_year_end)
            best_params_list.append(bp)
        return best_params_list

    def _get_oof_predictions(self, X, y, best_params_list):
        kf = KFold(n_splits=self.stacking_cv, shuffle=False, random_state=self.random_state)
        n_t = len(X['T'])
        oof_preds = [xr.full_like(y, np.nan) for _ in self.base_models]
        for i, base in enumerate(self.base_models):
            bp = best_params_list[i]
            for train_idx, val_idx in kf.split(range(n_t)):
                X_train_fold = X.isel(T=train_idx)
                y_train_fold = y.isel(T=train_idx)
                X_val_fold = X.isel(T=val_idx)
                y_val_fold = y.isel(T=val_idx)
                pred_val = base.compute_model(X_train_fold, y_train_fold, X_val_fold, y_val_fold, best_params=bp)
                oof_preds[i].loc[dict(T=X_val_fold['T'])] = pred_val
        return oof_preds

    def compute_model(self, X_train, y_train, X_test, y_test, best_params=None, clim_year_start=None, clim_year_end=None):
        if best_params is None:
            best_params = self.compute_hyperparameters(X_train, y_train, clim_year_start, clim_year_end)
        best_params_list = best_params  # Assuming best_params is list
        oof_preds = self._get_oof_predictions(X_train, y_train, best_params_list)
        X_meta_stacked = []
        for oof_base in oof_preds:
            stacked = oof_base.stack(sample=('T', 'Y', 'X')).values
            X_meta_stacked.append(stacked)
        X_meta = np.column_stack(X_meta_stacked)
        y_meta = y_train.stack(sample=('T', 'Y', 'X')).values
        nan_mask = np.any(~np.isfinite(X_meta), axis=1) | ~np.isfinite(y_meta)
        X_meta_clean = X_meta[~nan_mask]
        y_meta_clean = y_meta[~nan_mask]
        if self.meta_learner_type == 'linear':
            meta = LinearRegression()
            meta.fit(X_meta_clean, y_meta_clean)
        else:
            if self.meta_learner_type == 'ridge':
                meta_base = Ridge()
                param_dist = {'alpha': self.alpha_range}
            elif self.meta_learner_type == 'lasso':
                meta_base = Lasso()
                param_dist = {'alpha': self.alpha_range}
            elif self.meta_learner_type == 'elasticnet':
                meta_base = ElasticNet()
                param_dist = {'alpha': self.alpha_range, 'l1_ratio': self.l1_ratio_range}
            else:
                raise ValueError(f"Invalid meta_learner_type: {self.meta_learner_type}")
            random_search = RandomizedSearchCV(
                meta_base, param_distributions=param_dist, n_iter=self.meta_n_iter_search,
                cv=self.meta_cv_folds, scoring='neg_mean_squared_error',
                random_state=self.random_state, error_score=np.nan
            )
            random_search.fit(X_meta_clean, y_meta_clean)
            meta = random_search.best_estimator_
        self.meta_learner = meta
        base_test_preds = []
        for i, base in enumerate(self.base_models):
            bp = best_params_list[i]
            pred_test = base.compute_model(X_train, y_train, X_test, y_test, best_params=bp)
            base_test_preds.append(pred_test)
        time = X_test['T']
        lat = X_test['Y']
        lon = X_test['X']
        n_time = len(time)
        n_lat = len(lat)
        n_lon = len(lon)
        predictions = np.full((n_time, n_lat, n_lon), np.nan)
        X_meta_test_stacked = []
        for pred_base in base_test_preds:
            stacked = pred_base.stack(sample=('T', 'Y', 'X')).values
            X_meta_test_stacked.append(stacked)
        X_meta_test = np.column_stack(X_meta_test_stacked)
        nan_mask_test = np.any(~np.isfinite(X_meta_test), axis=1)
        if not np.all(nan_mask_test):
            X_meta_test_clean = X_meta_test[~nan_mask_test]
            y_pred_clean = meta.predict(X_meta_test_clean)
            full_y_pred = np.full(X_meta_test.shape[0], np.nan)
            full_y_pred[~nan_mask_test] = y_pred_clean
            pred_reshaped = full_y_pred.reshape(n_time, n_lat, n_lon)
            predictions = np.where(np.isnan(predictions), pred_reshaped, predictions)
        predicted_da = xr.DataArray(
            data=predictions,
            coords={'T': time, 'Y': lat, 'X': lon},
            dims=['T', 'Y', 'X']
        )
        return predicted_da

    # Probability calculation methods (copied from WAS_mme_hpELM_)
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
            pred_prob[0, :] = t.cdf(first_t, df=dof)
            pred_prob[1, :] = t.cdf(second_t, df=dof) - t.cdf(first_t, df=dof)
            pred_prob[2, :] = 1 - t.cdf(second_t, df=dof)
        return pred_prob

    @staticmethod
    def calculate_tercile_probabilities_weibull_min(best_guess, error_variance, first_tercile, second_tercile, dof):
        n_time = len(best_guess)
        pred_prob = np.empty((3, n_time))
        if np.all(np.isnan(best_guess)):
            pred_prob[:] = np.nan
        else:
            error_std = np.sqrt(error_variance)
            pred_prob[0, :] = weibull_min.cdf(first_tercile, c=dof, loc=best_guess, scale=error_std)
            pred_prob[1, :] = weibull_min.cdf(second_tercile, c=dof, loc=best_guess, scale=error_std) - \
                              weibull_min.cdf(first_tercile, c=dof, loc=best_guess, scale=error_std)
            pred_prob[2, :] = 1 - weibull_min.cdf(second_tercile, c=dof, loc=best_guess, scale=error_std)
        return pred_prob

    @staticmethod
    def calculate_tercile_probabilities_gamma(best_guess, error_variance, T1, T2, dof=None):
        n_time = len(best_guess)
        pred_prob = np.empty((3, n_time), dtype=float)
        if np.any(np.isnan(best_guess)) or np.any(np.isnan(error_variance)):
            pred_prob[:] = np.nan
            return pred_prob
        best_guess = np.asarray(best_guess, dtype=float)
        error_variance = np.asarray(error_variance, dtype=float)
        T1 = np.asarray(T1, dtype=float)
        T2 = np.asarray(T2, dtype=float)
        alpha = (best_guess ** 2) / error_variance
        theta = error_variance / best_guess
        cdf_t1 = gamma.cdf(T1, a=alpha, scale=theta)
        cdf_t2 = gamma.cdf(T2, a=alpha, scale=theta)
        pred_prob[0, :] = cdf_t1
        pred_prob[1, :] = cdf_t2 - cdf_t1
        pred_prob[2, :] = 1.0 - cdf_t2
        return pred_prob

    @staticmethod
    def calculate_tercile_probabilities_nonparametric(best_guess, error_samples, first_tercile, second_tercile):
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
        n_time = len(best_guess)
        pred_prob = np.empty((3, n_time))
        if np.all(np.isnan(best_guess)):
            pred_prob[:] = np.nan
        else:
            error_std = np.sqrt(error_variance)
            pred_prob[0, :] = norm.cdf(first_tercile, loc=best_guess, scale=error_std)
            pred_prob[1, :] = norm.cdf(second_tercile, loc=best_guess, scale=error_std) - norm.cdf(first_tercile, loc=best_guess, scale=error_std)
            pred_prob[2, :] = 1 - norm.cdf(second_tercile, loc=best_guess, scale=error_std)
        return pred_prob

    @staticmethod
    def calculate_tercile_probabilities_lognormal(best_guess, error_variance, first_tercile, second_tercile):
        n_time = len(best_guess)
        pred_prob = np.empty((3, n_time))
        if np.any(np.isnan(best_guess)) or np.any(np.isnan(error_variance)):
            pred_prob[:] = np.nan
            return pred_prob
        sigma = np.sqrt(np.log(1 + error_variance / (best_guess ** 2)))
        mu = np.log(best_guess) - sigma ** 2 / 2
        pred_prob[0, :] = lognorm.cdf(first_tercile, s=sigma, scale=np.exp(mu))
        pred_prob[1, :] = lognorm.cdf(second_tercile, s=sigma, scale=np.exp(mu)) - lognorm.cdf(first_tercile, s=sigma, scale=np.exp(mu))
        pred_prob[2, :] = 1 - lognorm.cdf(second_tercile, s=sigma, scale=np.exp(mu))
        return pred_prob

    def compute_prob(self, Predictant, clim_year_start, clim_year_end, hindcast_det):
        if "M" in Predictant.coords:
            Predictant = Predictant.isel(M=0).drop_vars('M').squeeze()
        Predictant = Predictant.transpose('T', 'Y', 'X')
        index_start = Predictant.get_index("T").get_loc(str(clim_year_start)).start
        index_end = Predictant.get_index("T").get_loc(str(clim_year_end)).stop
        rainfall_for_tercile = Predictant.isel(T=slice(index_start, index_end))
        terciles = rainfall_for_tercile.quantile([0.33, 0.67], dim='T')
        error_variance = (Predictant - hindcast_det).var(dim='T')
        dof = len(Predictant.get_index("T")) - 2
        if self.dist_method == "t":
            calc_func = self.calculate_tercile_probabilities
            hindcast_prob = xr.apply_ufunc(
                calc_func,
                hindcast_det,
                error_variance,
                terciles.isel(quantile=0).drop_vars('quantile'),
                terciles.isel(quantile=1).drop_vars('quantile'),
                input_core_dims=[('T',), (), (), ()],
                vectorize=True,
                kwargs={'dof': dof},
                dask='parallelized',
                output_core_dims=[('probability', 'T')],
                output_dtypes=['float'],
                dask_gufunc_kwargs={'output_sizes': {'probability': 3}, 'allow_rechunk': True}
            )
        elif self.dist_method == "weibull_min":
            calc_func = self.calculate_tercile_probabilities_weibull_min
            hindcast_prob = xr.apply_ufunc(
                calc_func,
                hindcast_det,
                error_variance,
                terciles.isel(quantile=0).drop_vars('quantile'),
                terciles.isel(quantile=1).drop_vars('quantile'),
                input_core_dims=[('T',), (), (), ()],
                vectorize=True,
                kwargs={'dof': dof},
                dask='parallelized',
                output_core_dims=[('probability', 'T')],
                output_dtypes=['float'],
                dask_gufunc_kwargs={'output_sizes': {'probability': 3}, 'allow_rechunk': True}
            )
        elif self.dist_method == "gamma":
            calc_func = self.calculate_tercile_probabilities_gamma
            hindcast_prob = xr.apply_ufunc(
                calc_func,
                hindcast_det,
                error_variance,
                terciles.isel(quantile=0).drop_vars('quantile'),
                terciles.isel(quantile=1).drop_vars('quantile'),
                input_core_dims=[('T',), (), (), ()],
                vectorize=True,
                dask='parallelized',
                output_core_dims=[('probability', 'T')],
                output_dtypes=['float'],
                dask_gufunc_kwargs={'output_sizes': {'probability': 3}, 'allow_rechunk': True}
            )
        elif self.dist_method == "nonparam":
            calc_func = self.calculate_tercile_probabilities_nonparametric
            error_samples = Predictant - hindcast_det
            hindcast_prob = xr.apply_ufunc(
                calc_func,
                hindcast_det,
                error_samples,
                terciles.isel(quantile=0).drop_vars('quantile'),
                terciles.isel(quantile=1).drop_vars('quantile'),
                input_core_dims=[('T',), ('T',), (), ()],
                output_core_dims=[('probability', 'T')],
                vectorize=True,
                dask='parallelized',
                output_dtypes=[float],
                dask_gufunc_kwargs={'output_sizes': {'probability': 3}, 'allow_rechunk': True}
            )
        elif self.dist_method == "normal":
            calc_func = self.calculate_tercile_probabilities_normal
            hindcast_prob = xr.apply_ufunc(
                calc_func,
                hindcast_det,
                error_variance,
                terciles.isel(quantile=0).drop_vars('quantile'),
                terciles.isel(quantile=1).drop_vars('quantile'),
                input_core_dims=[('T',), (), (), ()],
                vectorize=True,
                dask='parallelized',
                output_core_dims=[('probability', 'T')],
                output_dtypes=['float'],
                dask_gufunc_kwargs={'output_sizes': {'probability': 3}, 'allow_rechunk': True}
            )
        elif self.dist_method == "lognormal":
            calc_func = self.calculate_tercile_probabilities_lognormal
            hindcast_prob = xr.apply_ufunc(
                calc_func,
                hindcast_det,
                error_variance,
                terciles.isel(quantile=0).drop_vars('quantile'),
                terciles.isel(quantile=1).drop_vars('quantile'),
                input_core_dims=[('T',), (), (), ()],
                vectorize=True,
                dask='parallelized',
                output_core_dims=[('probability', 'T')],
                output_dtypes=['float'],
                dask_gufunc_kwargs={'output_sizes': {'probability': 3}, 'allow_rechunk': True}
            )
        else:
            raise ValueError(f"Invalid dist_method: {self.dist_method}.")
        hindcast_prob = hindcast_prob.assign_coords(probability=('probability', ['PB', 'PN', 'PA']))
        return hindcast_prob.transpose('probability', 'T', 'Y', 'X')

    def forecast(self, Predictant, clim_year_start, clim_year_end, hindcast_det, hindcast_det_cross, Predictor_for_year, best_params=None):
        if "M" in Predictant.coords:
            Predictant_no_m = Predictant.isel(M=0).drop_vars('M').squeeze()
        else:
            Predictant_no_m = Predictant
        mask = xr.where(~np.isnan(Predictant_no_m.isel(T=0)), 1, np.nan).drop_vars(['T']).squeeze().to_numpy()
        mean_val = hindcast_det.sel(T=slice(str(clim_year_start), str(clim_year_end))).mean(dim='T')
        std_val = hindcast_det.sel(T=slice(str(clim_year_start), str(clim_year_end))).std(dim='T')
        Predictor_for_year_st = (Predictor_for_year - mean_val) / std_val
        hindcast_det_st = standardize_timeseries(hindcast_det, clim_year_start, clim_year_end)
        Predictant_st = standardize_timeseries(Predictant_no_m, clim_year_start, clim_year_end)
        time = Predictor_for_year_st['T']
        n_time = len(time)
        y_test_dummy = xr.full_like(Predictant_st.isel(T=0), np.nan).expand_dims(T=time)
        if best_params is None:
            best_params = self.compute_hyperparameters(hindcast_det, Predictant_no_m, clim_year_start, clim_year_end)
        best_params_list = best_params
        oof_preds = self._get_oof_predictions(hindcast_det_st, Predictant_st, best_params_list)
        X_meta_stacked = []
        for oof_base in oof_preds:
            stacked = oof_base.stack(sample=('T', 'Y', 'X')).values
            X_meta_stacked.append(stacked)
        X_meta = np.column_stack(X_meta_stacked)
        y_meta = Predictant_st.stack(sample=('T', 'Y', 'X')).values
        nan_mask = np.any(~np.isfinite(X_meta), axis=1) | ~np.isfinite(y_meta)
        X_meta_clean = X_meta[~nan_mask]
        y_meta_clean = y_meta[~nan_mask]
        if self.meta_learner_type == 'linear':
            meta = LinearRegression()
            meta.fit(X_meta_clean, y_meta_clean)
        else:
            if self.meta_learner_type == 'ridge':
                meta_base = Ridge()
                param_dist = {'alpha': self.alpha_range}
            elif self.meta_learner_type == 'lasso':
                meta_base = Lasso()
                param_dist = {'alpha': self.alpha_range}
            elif self.meta_learner_type == 'elasticnet':
                meta_base = ElasticNet()
                param_dist = {'alpha': self.alpha_range, 'l1_ratio': self.l1_ratio_range}
            else:
                raise ValueError(f"Invalid meta_learner_type: {self.meta_learner_type}")
            random_search = RandomizedSearchCV(
                meta_base, param_distributions=param_dist, n_iter=self.meta_n_iter_search,
                cv=self.meta_cv_folds, scoring='neg_mean_squared_error',
                random_state=self.random_state, error_score=np.nan
            )
            random_search.fit(X_meta_clean, y_meta_clean)
            meta = random_search.best_estimator_
        self.meta_learner = meta
        base_forecast_std = []
        for i, base in enumerate(self.base_models):
            bp = best_params_list[i]
            pred_base = base.compute_model(hindcast_det_st, Predictant_st, Predictor_for_year_st, y_test_dummy, best_params=bp)
            base_forecast_std.append(pred_base)
        lat = Predictor_for_year_st['Y']
        lon = Predictor_for_year_st['X']
        n_lat = len(lat)
        n_lon = len(lon)
        predictions = np.full((n_time, n_lat, n_lon), np.nan)
        X_meta_test_stacked = []
        for pred_base in base_forecast_std:
            stacked = pred_base.stack(sample=('T', 'Y', 'X')).values
            X_meta_test_stacked.append(stacked)
        X_meta_test = np.column_stack(X_meta_test_stacked)
        nan_mask_test = np.any(~np.isfinite(X_meta_test), axis=1)
        if not np.all(nan_mask_test):
            X_meta_test_clean = X_meta_test[~nan_mask_test]
            y_pred_clean = meta.predict(X_meta_test_clean)
            full_y_pred = np.full(X_meta_test.shape[0], np.nan)
            full_y_pred[~nan_mask_test] = y_pred_clean
            pred_reshaped = full_y_pred.reshape(n_time, n_lat, n_lon)
            predictions = np.where(np.isnan(predictions), pred_reshaped, predictions)
        result_da = xr.DataArray(
            data=predictions,
            coords={'T': time, 'Y': lat, 'X': lon},
            dims=['T', 'Y', 'X']
        ) * mask
        result_da = reverse_standardize(result_da, Predictant_no_m, clim_year_start, clim_year_end)
        year = Predictor_for_year.coords['T'].values.astype('datetime64[Y]').astype(int)[0] + 1970
        T_value_1 = Predictant_no_m.isel(T=0).coords['T'].values
        month_1 = T_value_1.astype('datetime64[M]').astype(int) % 12 + 1
        new_T_value = np.datetime64(f"{year}-{month_1:02d}-01")
        result_da = result_da.assign_coords(T=xr.DataArray([new_T_value], dims=["T"]))
        result_da['T'] = result_da['T'].astype('datetime64[ns]')
        # Compute probabilities
        index_start = Predictant_no_m.get_index("T").get_loc(str(clim_year_start)).start
        index_end = Predictant_no_m.get_index("T").get_loc(str(clim_year_end)).stop
        rainfall_for_tercile = Predictant_no_m.isel(T=slice(index_start, index_end))
        terciles = rainfall_for_tercile.quantile([0.33, 0.67], dim='T')
        error_variance = (Predictant_no_m - hindcast_det_cross).var(dim='T')
        dof = len(Predictant_no_m.get_index("T")) - 2
        if self.dist_method == "t":
            calc_func = self.calculate_tercile_probabilities
            hindcast_prob = xr.apply_ufunc(
                calc_func,
                result_da,
                error_variance,
                terciles.isel(quantile=0).drop_vars('quantile'),
                terciles.isel(quantile=1).drop_vars('quantile'),
                input_core_dims=[('T',), (), (), ()],
                kwargs={'dof': dof},
                vectorize=True,
                dask='parallelized',
                output_core_dims=[('probability', 'T')],
                output_dtypes=['float'],
                dask_gufunc_kwargs={'output_sizes': {'probability': 3}, 'allow_rechunk': True}
            )
        elif self.dist_method == "weibull_min":
            calc_func = self.calculate_tercile_probabilities_weibull_min
            hindcast_prob = xr.apply_ufunc(
                calc_func,
                result_da,
                error_variance,
                terciles.isel(quantile=0).drop_vars('quantile'),
                terciles.isel(quantile=1).drop_vars('quantile'),
                input_core_dims=[('T',), (), (), ()],
                kwargs={'dof': dof},
                vectorize=True,
                dask='parallelized',
                output_core_dims=[('probability', 'T')],
                output_dtypes=['float'],
                dask_gufunc_kwargs={'output_sizes': {'probability': 3}, 'allow_rechunk': True}
            )
        elif self.dist_method == "gamma":
            calc_func = self.calculate_tercile_probabilities_gamma
            hindcast_prob = xr.apply_ufunc(
                calc_func,
                result_da,
                error_variance,
                terciles.isel(quantile=0).drop_vars('quantile'),
                terciles.isel(quantile=1).drop_vars('quantile'),
                input_core_dims=[('T',), (), (), ()],
                vectorize=True,
                dask='parallelized',
                output_core_dims=[('probability', 'T')],
                output_dtypes=['float'],
                dask_gufunc_kwargs={'output_sizes': {'probability': 3}, 'allow_rechunk': True}
            )
        elif self.dist_method == "normal":
            calc_func = self.calculate_tercile_probabilities_normal
            hindcast_prob = xr.apply_ufunc(
                calc_func,
                result_da,
                error_variance,
                terciles.isel(quantile=0).drop_vars('quantile'),
                terciles.isel(quantile=1).drop_vars('quantile'),
                input_core_dims=[('T',), (), (), ()],
                vectorize=True,
                dask='parallelized',
                output_core_dims=[('probability', 'T')],
                output_dtypes=['float'],
                dask_gufunc_kwargs={'output_sizes': {'probability': 3}, 'allow_rechunk': True}
            )
        elif self.dist_method == "lognormal":
            calc_func = self.calculate_tercile_probabilities_lognormal
            hindcast_prob = xr.apply_ufunc(
                calc_func,
                result_da,
                error_variance,
                terciles.isel(quantile=0).drop_vars('quantile'),
                terciles.isel(quantile=1).drop_vars('quantile'),
                input_core_dims=[('T',), (), (), ()],
                vectorize=True,
                dask='parallelized',
                output_core_dims=[('probability', 'T')],
                output_dtypes=['float'],
                dask_gufunc_kwargs={'output_sizes': {'probability': 3}, 'allow_rechunk': True}
            )
        elif self.dist_method == "nonparam":
            calc_func = self.calculate_tercile_probabilities_nonparametric
            error_samples = Predictant_no_m - hindcast_det_cross
            error_samples = error_samples.rename({'T': 'S'})
            hindcast_prob = xr.apply_ufunc(
                calc_func,
                result_da,
                error_samples,
                terciles.isel(quantile=0).drop_vars('quantile'),
                terciles.isel(quantile=1).drop_vars('quantile'),
                input_core_dims=[('T',), ('S',), (), ()],
                output_core_dims=[('probability', 'T')],
                vectorize=True,
                dask='parallelized',
                output_dtypes=[float],
                dask_gufunc_kwargs={'output_sizes': {'probability': 3}, 'allow_rechunk': True}
            )
        else:
            raise ValueError(f"Invalid dist_method: {self.dist_method}.")
        hindcast_prob = hindcast_prob.assign_coords(probability=('probability', ['PB', 'PN', 'PA']))
        return result_da * mask, mask * hindcast_prob.transpose('probability', 'T', 'Y', 'X')


class WAS_mme_Stacking2:
    """
    Stacking ensemble for Multi-Model Ensemble (MME) forecasting using provided base models
    with Linear Regression as the meta-learner.
    This class implements stacking where base models' out-of-sample predictions (via inner CV)
    are used to train the meta-learner per cluster. For hindcast and forecast, it follows
    a similar structure to the base classes, with tercile probability calculations.
    Parameters
    ----------
    random_state : int, optional
        Seed for reproducibility (default is 42).
    dist_method : str, optional
        Distribution method for tercile probabilities ('t', 'gamma', etc.) (default is 'gamma').
    n_iter_search : int, optional
        Number of iterations for randomized search in base models (default is 10).
    cv_folds : int, optional
        Number of cross-validation folds for inner CV and base models (default is 3).
    n_clusters : int, optional
        Number of clusters for homogenized zones (default is 4).
    """
    def __init__(self,
                 random_state=42,
                 dist_method="gamma",
                 n_iter_search=10,
                 cv_folds=3,
                 n_clusters=4):
        self.random_state = random_state
        self.dist_method = dist_method
        self.n_iter_search = n_iter_search
        self.cv_folds = cv_folds
        self.n_clusters = n_clusters
        self.base_models = [
            WAS_mme_hpELM(neurons_range=[10, 20, 50, 100],
                          activation_options=['sigm', 'tanh', 'relu', 'rbf_linf', 'rbf_gauss'],
                          norm_range=[0.1, 1.0, 10.0, 100.0],
                          random_state=random_state,
                          dist_method=dist_method,
                          n_iter_search=n_iter_search,
                          cv_folds=cv_folds,
                          n_clusters=n_clusters),
            WAS_mme_MLP(hidden_layer_sizes_range=[(10,), (10, 5), (20, 10)],
                        activation_options=['relu', 'tanh', 'logistic'],
                        solver_options=['adam', 'sgd', 'lbfgs'],
                        alpha_range=[0.0001, 0.001, 0.01, 0.1],
                        max_iter=200,
                        random_state=random_state,
                        dist_method=dist_method,
                        n_iter_search=n_iter_search,
                        cv_folds=cv_folds,
                        n_clusters=n_clusters),
            WAS_mme_XGBoosting(n_estimators_range=[50, 100, 200, 300],
                               learning_rate_range=[0.01, 0.05, 0.1, 0.2],
                               max_depth_range=[3, 5, 7, 9],
                               min_child_weight_range=[1, 3, 5],
                               subsample_range=[0.6, 0.8, 1.0],
                               colsample_bytree_range=[0.6, 0.8, 1.0],
                               random_state=random_state,
                               dist_method=dist_method,
                               n_iter_search=n_iter_search,
                               cv_folds=cv_folds,
                               n_clusters=n_clusters),
            WAS_mme_RF(n_estimators_range=[50, 100, 200, 300],
                       max_depth_range=[None, 10, 20, 30],
                       min_samples_split_range=[2, 5, 10],
                       min_samples_leaf_range=[1, 2, 4],
                       max_features_range=['auto', 'sqrt', 0.33, 0.5],
                       random_state=random_state,
                       dist_method=dist_method,
                       n_iter_search=n_iter_search,
                       cv_folds=cv_folds,
                       n_clusters=n_clusters)
        ]
        self.meta_learner = None

    def compute_hyperparameters(self, Predictors, Predictand, clim_year_start, clim_year_end):
        """
        Computes hyperparameters for each base model and returns the cluster labels.
        """
        best_params_dict = {}
        # Compute for the first model to get cluster_da
        best_params_0, cluster_da = self.base_models[0].compute_hyperparameters(Predictors, Predictand, clim_year_start, clim_year_end)
        best_params_dict[0] = best_params_0
        # Compute for others, ignore their cluster_da since same
        for i in range(1, len(self.base_models)):
            best_params_i, _ = self.base_models[i].compute_hyperparameters(Predictors, Predictand, clim_year_start, clim_year_end)
            best_params_dict[i] = best_params_i
        return best_params_dict, cluster_da

    def compute_model(self, X_train, y_train, X_test, y_test, best_params=None, cluster_da=None):
        """
        Computes deterministic hindcast using stacking.
        Uses inner CV to get OOS predictions from base models for meta-learner training.
        """
        # Standardize inputs (assuming inputs are already standardized as in base classes)
        X_train_std = X_train
        y_train_std = y_train
        X_test_std = X_test
        y_test_std = y_test

        # Extract coordinates
        time_train = X_train_std['T']
        time_test = X_test_std['T']
        lat = X_test_std['Y']
        lon = X_test_std['X']
        n_time_test = len(time_test)
        n_lat = len(lat)
        n_lon = len(lon)

        # Use provided best_params and cluster_da or compute if None
        if best_params is None or cluster_da is None:
            best_params, cluster_da = self.compute_hyperparameters(X_train_std, y_train_std, clim_year_start, clim_year_end)

        # Inner CV to get OOS base predictions for train
        kf = KFold(n_splits=self.cv_folds, shuffle=False, random_state=self.random_state)
        n_time_train = len(time_train)
        base_train_preds = [xr.full_like(y_train_std, np.nan) for _ in self.base_models]

        for train_idx, val_idx in kf.split(np.arange(n_time_train)):
            X_sub_train = X_train_std.isel(T=train_idx)
            y_sub_train = y_train_std.isel(T=train_idx)
            X_sub_val = X_train_std.isel(T=val_idx)
            y_sub_val = y_train_std.isel(T=val_idx)
            for i, model in enumerate(self.base_models):
                bp_i = best_params[i]
                pred_val = model.compute_model(X_sub_train, y_sub_train, X_sub_val, y_sub_val, best_params=bp_i, cluster_da=cluster_da)
                base_train_preds[i] = base_train_preds[i].where(~np.isfinite(base_train_preds[i]), pred_val).assign_coords(T=time_train)  # Assign to val positions

        # Now fit meta-learner per cluster using OOS base_train_preds
        self.meta_learner = {}
        clusters = np.unique(cluster_da.values[~np.isnan(cluster_da.values)])
        for c in clusters:
            mask_3d_train = (cluster_da == c).expand_dims({'T': time_train})
            X_meta_train_list = []
            for pred in base_train_preds:
                pred_stacked_c = pred.where(mask_3d_train).stack(sample=('T', 'Y', 'X')).values
                X_meta_train_list.append(pred_stacked_c)
            X_meta_train_c = np.column_stack(X_meta_train_list)
            y_stacked_c = y_train_std.where(mask_3d_train).stack(sample=('T', 'Y', 'X')).values
            nan_mask_c = np.any(~np.isfinite(X_meta_train_c), axis=1) | ~np.isfinite(y_stacked_c)
            if np.all(nan_mask_c):
                continue
            X_meta_clean_c = X_meta_train_c[~nan_mask_c]
            y_clean_c = y_stacked_c[~nan_mask_c]
            meta_c = LinearRegression().fit(X_meta_clean_c, y_clean_c)
            self.meta_learner[c] = meta_c

        # Get base predictions on test (train bases on full train)
        base_test_preds = []
        for i, model in enumerate(self.base_models):
            bp_i = best_params[i]
            pred_test = model.compute_model(X_train_std, y_train_std, X_test_std, y_test_std, best_params=bp_i, cluster_da=cluster_da)
            base_test_preds.append(pred_test)

        # Predict with meta on test per cluster
        predictions = np.full((n_time_test, n_lat, n_lon), np.nan)
        for c in clusters:
            if c not in self.meta_learner:
                continue
            mask_3d_test = (cluster_da == c).expand_dims({'T': time_test})
            X_meta_test_list = []
            full_length = None
            for pred in base_test_preds:
                pred_stacked_c = pred.where(mask_3d_test).stack(sample=('T', 'Y', 'X')).values
                X_meta_test_list.append(pred_stacked_c)
                if full_length is None:
                    full_length = len(pred_stacked_c)
            X_meta_test_c = np.column_stack(X_meta_test_list)
            nan_mask_test_c = np.any(~np.isfinite(X_meta_test_c), axis=1)
            X_meta_test_clean_c = X_meta_test_c[~nan_mask_test_c]
            if len(X_meta_test_clean_c) == 0:
                continue
            y_pred_clean_c = self.meta_learner[c].predict(X_meta_test_clean_c)
            full_stacked_c = np.full(full_length, np.nan)
            full_stacked_c[~nan_mask_test_c] = y_pred_clean_c
            pred_c_reshaped = full_stacked_c.reshape(n_time_test, n_lat, n_lon)
            predictions = np.where(np.isnan(predictions), pred_c_reshaped, predictions)

        predicted_da = xr.DataArray(
            data=predictions,
            coords={'T': time_test, 'Y': lat, 'X': lon},
            dims=['T', 'Y', 'X']
        )
        return predicted_da

    # Probability Calculation Methods (same as base classes)
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
            pred_prob[0, :] = t.cdf(first_t, df=dof)
            pred_prob[1, :] = t.cdf(second_t, df=dof) - t.cdf(first_t, df=dof)
            pred_prob[2, :] = 1 - t.cdf(second_t, df=dof)
        return pred_prob

    @staticmethod
    def calculate_tercile_probabilities_weibull_min(best_guess, error_variance, first_tercile, second_tercile, dof):
        n_time = len(best_guess)
        pred_prob = np.empty((3, n_time))
        if np.all(np.isnan(best_guess)):
            pred_prob[:] = np.nan
        else:
            error_std = np.sqrt(error_variance)
            pred_prob[0, :] = weibull_min.cdf(first_tercile, c=dof, loc=best_guess, scale=error_std)
            pred_prob[1, :] = weibull_min.cdf(second_tercile, c=dof, loc=best_guess, scale=error_std) - \
                              weibull_min.cdf(first_tercile, c=dof, loc=best_guess, scale=error_std)
            pred_prob[2, :] = 1 - weibull_min.cdf(second_tercile, c=dof, loc=best_guess, scale=error_std)
        return pred_prob

    @staticmethod
    def calculate_tercile_probabilities_gamma(best_guess, error_variance, T1, T2, dof=None):
        n_time = len(best_guess)
        pred_prob = np.empty((3, n_time), dtype=float)
        if np.any(np.isnan(best_guess)) or np.any(np.isnan(error_variance)):
            pred_prob[:] = np.nan
            return pred_prob
        best_guess = np.asarray(best_guess, dtype=float)
        error_variance = np.asarray(error_variance, dtype=float)
        T1 = np.asarray(T1, dtype=float)
        T2 = np.asarray(T2, dtype=float)
        alpha = (best_guess ** 2) / error_variance
        theta = error_variance / best_guess
        cdf_t1 = gamma.cdf(T1, a=alpha, scale=theta)
        cdf_t2 = gamma.cdf(T2, a=alpha, scale=theta)
        pred_prob[0, :] = cdf_t1
        pred_prob[1, :] = cdf_t2 - cdf_t1
        pred_prob[2, :] = 1.0 - cdf_t2
        return pred_prob

    @staticmethod
    def calculate_tercile_probabilities_nonparametric(best_guess, error_samples, first_tercile, second_tercile):
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
        n_time = len(best_guess)
        pred_prob = np.empty((3, n_time))
        if np.all(np.isnan(best_guess)):
            pred_prob[:] = np.nan
        else:
            error_std = np.sqrt(error_variance)
            pred_prob[0, :] = norm.cdf(first_tercile, loc=best_guess, scale=error_std)
            pred_prob[1, :] = norm.cdf(second_tercile, loc=best_guess, scale=error_std) - norm.cdf(first_tercile, loc=best_guess, scale=error_std)
            pred_prob[2, :] = 1 - norm.cdf(second_tercile, loc=best_guess, scale=error_std)
        return pred_prob

    @staticmethod
    def calculate_tercile_probabilities_lognormal(best_guess, error_variance, first_tercile, second_tercile):
        n_time = len(best_guess)
        pred_prob = np.empty((3, n_time))
        if np.any(np.isnan(best_guess)) or np.any(np.isnan(error_variance)):
            pred_prob[:] = np.nan
            return pred_prob
        sigma = np.sqrt(np.log(1 + error_variance / (best_guess ** 2)))
        mu = np.log(best_guess) - sigma ** 2 / 2
        pred_prob[0, :] = lognorm.cdf(first_tercile, s=sigma, scale=np.exp(mu))
        pred_prob[1, :] = lognorm.cdf(second_tercile, s=sigma, scale=np.exp(mu)) - lognorm.cdf(first_tercile, s=sigma, scale=np.exp(mu))
        pred_prob[2, :] = 1 - lognorm.cdf(second_tercile, s=sigma, scale=np.exp(mu))
        return pred_prob

    def compute_prob(self, Predictant, clim_year_start, clim_year_end, hindcast_det):
        """
        Compute tercile probabilities using the chosen distribution method.
        """
        if "M" in Predictant.coords:
            Predictant = Predictant.isel(M=0).drop_vars('M').squeeze()
        mask = xr.where(~np.isnan(Predictant.isel(T=0)), 1, np.nan).drop_vars(['T']).squeeze().to_numpy()
        Predictant = Predictant.transpose('T', 'Y', 'X')
        index_start = Predictant.get_index("T").get_loc(str(clim_year_start)).start
        index_end = Predictant.get_index("T").get_loc(str(clim_year_end)).stop
        rainfall_for_tercile = Predictant.isel(T=slice(index_start, index_end))
        terciles = rainfall_for_tercile.quantile([0.33, 0.67], dim='T')
        error_variance = (Predictant - hindcast_det).var(dim='T')
        dof = len(Predictant.get_index("T")) - 2
        if self.dist_method == "t":
            calc_func = self.calculate_tercile_probabilities
            kwargs = {'dof': dof}
            input_core_dims = [('T',), (), (), ()]
        elif self.dist_method == "weibull_min":
            calc_func = self.calculate_tercile_probabilities_weibull_min
            kwargs = {'dof': dof}
            input_core_dims = [('T',), (), (), ()]
        elif self.dist_method == "gamma":
            calc_func = self.calculate_tercile_probabilities_gamma
            kwargs = {}
            input_core_dims = [('T',), (), (), ()]
        elif self.dist_method == "nonparam":
            calc_func = self.calculate_tercile_probabilities_nonparametric
            error_samples = Predictant - hindcast_det
            kwargs = {}
            input_core_dims = [('T',), ('T',), (), ()]
            hindcast_det, error_variance, terciles_0, terciles_1 = hindcast_det, error_samples, terciles.isel(quantile=0).drop_vars('quantile'), terciles.isel(quantile=1).drop_vars('quantile')
        elif self.dist_method == "normal":
            calc_func = self.calculate_tercile_probabilities_normal
            kwargs = {}
            input_core_dims = [('T',), (), (), ()]
        elif self.dist_method == "lognormal":
            calc_func = self.calculate_tercile_probabilities_lognormal
            kwargs = {}
            input_core_dims = [('T',), (), (), ()]
        else:
            raise ValueError(f"Invalid dist_method: {self.dist_method}.")
        hindcast_prob = xr.apply_ufunc(
            calc_func,
            hindcast_det,
            error_variance,
            terciles.isel(quantile=0).drop_vars('quantile'),
            terciles.isel(quantile=1).drop_vars('quantile'),
            input_core_dims=input_core_dims,
            vectorize=True,
            dask='parallelized',
            output_core_dims=[('probability', 'T')],
            output_dtypes=['float'],
            dask_gufunc_kwargs={'output_sizes': {'probability': 3}, 'allow_rechunk': True},
            **kwargs
        )
        hindcast_prob = hindcast_prob.assign_coords(probability=('probability', ['PB', 'PN', 'PA']))
        return hindcast_prob.transpose('probability', 'T', 'Y', 'X')

    def forecast(self, Predictant, clim_year_start, clim_year_end, hindcast_det, hindcast_det_cross, Predictor_for_year, best_params=None, cluster_da=None):
        """
        Generate deterministic and probabilistic forecast for a target year using stacking.
        """
        if "M" in Predictant.coords:
            Predictant_no_m = Predictant.isel(M=0).drop_vars('M').squeeze()
        else:
            Predictant_no_m = Predictant
        mask = xr.where(~np.isnan(Predictant_no_m.isel(T=0)), 1, np.nan).drop_vars(['T']).squeeze().to_numpy()
        # Standardize
        mean_val = hindcast_det.sel(T=slice(str(clim_year_start), str(clim_year_end))).mean(dim='T')
        std_val = hindcast_det.sel(T=slice(str(clim_year_start), str(clim_year_end))).std(dim='T')
        Predictor_for_year_st = (Predictor_for_year - mean_val) / std_val
        hindcast_det_st = standardize_timeseries(hindcast_det, clim_year_start, clim_year_end)
        Predictant_st = standardize_timeseries(Predictant_no_m, clim_year_start, clim_year_end)
        hindcast_det_st['T'] = Predictant_st['T']
        # Compute hyperparameters and cluster if None
        if best_params is None or cluster_da is None:
            best_params, cluster_da = self.compute_hyperparameters(hindcast_det, Predictant_no_m, clim_year_start, clim_year_end)
        # Use compute_model for forecast (here X_test is Predictor_for_year_st, y_test is dummy since not used)
        y_test_dummy = xr.full_like(Predictant_st.isel(T=0).expand_dims(T=Predictor_for_year_st['T']), np.nan)
        forecast_det_st = self.compute_model(hindcast_det_st, Predictant_st, Predictor_for_year_st, y_test_dummy, best_params=best_params, cluster_da=cluster_da)
        forecast_det = reverse_standardize(forecast_det_st, Predictant_no_m, clim_year_start, clim_year_end)
        # Update time coordinate
        year = int(Predictor_for_year.coords['T'].values[0].strftime('%Y'))
        T_value_1 = Predictant_no_m.isel(T=0).coords['T'].values
        month_1 = int(T_value_1.strftime('%m'))
        new_T_value = np.datetime64(f"{year}-{month_1:02d}-01")
        forecast_det = forecast_det.assign_coords(T=[new_T_value])
        forecast_det['T'] = forecast_det['T'].astype('datetime64[ns]')
        # Compute probabilities
        index_start = Predictant_no_m.get_index("T").get_loc(str(clim_year_start)).start
        index_end = Predictant_no_m.get_index("T").get_loc(str(clim_year_end)).stop
        rainfall_for_tercile = Predictant_no_m.isel(T=slice(index_start, index_end))
        terciles = rainfall_for_tercile.quantile([0.33, 0.67], dim='T')
        error_variance = (Predictant_no_m - hindcast_det_cross).var(dim='T')
        dof = len(Predictant_no_m.get_index("T")) - 2
        forecast_det_for_prob = forecast_det.copy()  # Use the det for best_guess
        if self.dist_method == "nonparam":
            error_samples = Predictant_no_m - hindcast_det_cross
            error_samples = error_samples.rename({'T': 'S'})
            calc_func = self.calculate_tercile_probabilities_nonparametric
            forecast_prob = xr.apply_ufunc(
                calc_func,
                forecast_det_for_prob,
                error_samples,
                terciles.isel(quantile=0).drop_vars('quantile'),
                terciles.isel(quantile=1).drop_vars('quantile'),
                input_core_dims=[('T',), ('S',), (), ()],
                output_core_dims=[('probability', 'T')],
                vectorize=True,
                dask='parallelized',
                output_dtypes=[float],
                dask_gufunc_kwargs={'output_sizes': {'probability': 3}, 'allow_rechunk': True}
            )
        else:
            if self.dist_method == "t":
                calc_func = self.calculate_tercile_probabilities
                kwargs = {'dof': dof}
            elif self.dist_method == "weibull_min":
                calc_func = self.calculate_tercile_probabilities_weibull_min
                kwargs = {'dof': dof}
            elif self.dist_method == "gamma":
                calc_func = self.calculate_tercile_probabilities_gamma
                kwargs = {}
            elif self.dist_method == "normal":
                calc_func = self.calculate_tercile_probabilities_normal
                kwargs = {}
            elif self.dist_method == "lognormal":
                calc_func = self.calculate_tercile_probabilities_lognormal
                kwargs = {}
            forecast_prob = xr.apply_ufunc(
                calc_func,
                forecast_det_for_prob,
                error_variance,
                terciles.isel(quantile=0).drop_vars('quantile'),
                terciles.isel(quantile=1).drop_vars('quantile'),
                input_core_dims=[('T',), (), (), ()],
                vectorize=True,
                dask='parallelized',
                output_core_dims=[('probability', 'T')],
                output_dtypes=['float'],
                dask_gufunc_kwargs={'output_sizes': {'probability': 3}, 'allow_rechunk': True},
                **kwargs
            )
        forecast_prob = forecast_prob.assign_coords(probability=('probability', ['PB', 'PN', 'PA']))
        return forecast_det * mask, mask * forecast_prob.transpose('probability', 'T', 'Y', 'X')



class WAS_mme_GP_: # mine
    """
    Gaussian Process-based Multi-Model Ensemble (MME) forecasting.
    This class implements a single-model forecasting approach using scikit-learn's GaussianProcessRegressor
    for probabilistic predictions, computing tercile probabilities directly from the Gaussian predictive distribution.
    Implements hyperparameter optimization via randomized search for kernel parameters.
    Parameters
    ----------
    length_scale_range : list of float, optional
        List of length scales to tune for RBF kernel (default is np.logspace(-1, 1, 5).tolist()).
    noise_level_range : list of float, optional
        List of noise levels to tune for WhiteKernel (default is np.logspace(-5, -1, 5).tolist()).
    random_state : int, optional
        Seed for reproducibility (default is 42).
    dist_method : str, optional
        Distribution method for tercile probabilities ('t', 'gamma', etc.) (default is 'normal' for GP predictive distribution).
    n_iter_search : int, optional
        Number of iterations for randomized search (default is 10).
    cv_folds : int, optional
        Number of cross-validation folds (default is 3).
    n_clusters : int, optional
        Number of clusters for homogenized zones (default is 4).
    """
    def __init__(self,
                 length_scale_range=np.logspace(-1, 1, 5).tolist(),
                 noise_level_range=np.logspace(-5, -1, 5).tolist(),
                 random_state=42,
                 dist_method="normal",
                 n_iter_search=10,
                 cv_folds=3,
                 n_clusters=4):
        self.length_scale_range = length_scale_range
        self.noise_level_range = noise_level_range
        self.random_state = random_state
        self.dist_method = dist_method
        self.n_iter_search = n_iter_search
        self.cv_folds = cv_folds
        self.n_clusters = n_clusters
        self.gp = None

    def compute_hyperparameters(self, Predictors, Predictand, clim_year_start, clim_year_end):
        """
        Independently computes the best hyperparameters using randomized search on stacked training data for each homogenized zone.
        Parameters
        ----------
        Predictors : xarray.DataArray
            Training predictor data with dimensions (T, M, Y, X).
        Predictand : xarray.DataArray
            Training predictand data with dimensions (T, Y, X).
        clim_year_start : int
            Start year of the climatology period.
        clim_year_end : int
            End year of the climatology period.
        Returns
        -------
        best_params_dict : dict
            Best hyperparameters for each cluster.
        cluster_da : xarray.DataArray
            Cluster labels with dimensions (Y, X).
        """
        if "M" in Predictand.coords:
            Predictand = Predictand.isel(M=0).drop_vars('M').squeeze()
           
        X_train_std = standardize_timeseries(Predictors, clim_year_start, clim_year_end)
        Predictand.name = "varname"
        # Step 1: Perform KMeans clustering based on predictand's spatial distribution
        kmeans = KMeans(n_clusters=self.n_clusters, random_state=self.random_state)
        Predictand_dropna = Predictand.to_dataframe().reset_index().dropna().drop(columns=['T'])
        variable_column = Predictand_dropna.columns[2]
        Predictand_dropna['cluster'] = kmeans.fit_predict(
           Predictand_dropna[[variable_column]]
        )
       
        # Convert cluster assignments back into an xarray structure
        df_unique = Predictand_dropna.drop_duplicates(subset=['Y', 'X'])
        dataset = df_unique.set_index(['Y', 'X']).to_xarray()
        mask = xr.where(~np.isnan(Predictand.isel(T=0)), 1, np.nan)
        Cluster = (dataset['cluster'] * mask)
              
        # Align cluster array with the predictand array
        xarray1, xarray2 = xr.align(Predictand, Cluster, join="outer")
       
        # Identify unique cluster labels
        clusters = np.unique(xarray2)
        clusters = clusters[~np.isnan(clusters)]
        cluster_da = xarray2
        y_train_std = standardize_timeseries(Predictand, clim_year_start, clim_year_end)
        X_train_std['T'] = y_train_std['T']
        # Prepare parameter distributions
        param_dist = {
            'kernel__k1__length_scale': self.length_scale_range,
            'kernel__k2__noise_level': self.noise_level_range
        }
        best_params_dict = {}
        for c in clusters:
            mask_3d = (cluster_da == c).expand_dims({'T': y_train_std['T']})
            X_stacked_c = X_train_std.where(mask_3d).stack(sample=('T', 'Y', 'X')).transpose('sample', 'M').values
            y_stacked_c = y_train_std.where(mask_3d).stack(sample=('T', 'Y', 'X')).values.ravel()
            nan_mask_c = np.any(~np.isfinite(X_stacked_c), axis=1) | ~np.isfinite(y_stacked_c)
            X_clean_c = X_stacked_c[~nan_mask_c]
            y_clean_c = y_stacked_c[~nan_mask_c]
            if len(X_clean_c) == 0:
                continue
            # Initialize GaussianProcessRegressor with initial kernel
            kernel = RBF(1.0) + WhiteKernel(1e-5)
            model = GaussianProcessRegressor(kernel=kernel, n_restarts_optimizer=10, random_state=self.random_state)
            # Randomized search
            random_search = RandomizedSearchCV(
                model, param_distributions=param_dist, n_iter=self.n_iter_search,
                cv=self.cv_folds, scoring='neg_mean_squared_error',
                random_state=self.random_state, error_score=np.nan
            )
            random_search.fit(X_clean_c, y_clean_c)
            best_params_dict[c] = random_search.best_params_
        return best_params_dict, cluster_da

    def compute_model(self, X_train, y_train, X_test, y_test, clim_year_start, clim_year_end, best_params=None, cluster_da=None):
        """
        Compute probabilistic hindcast using the GaussianProcessRegressor model with injected hyperparameters for each zone.
        Returns the tercile probabilities directly.
        Parameters
        ----------
        X_train : xarray.DataArray
            Training predictor data with dimensions (T, M, Y, X).
        y_train : xarray.DataArray
            Training predictand data with dimensions (T, Y, X).
        X_test : xarray.DataArray
            Testing predictor data with dimensions (T, M, Y, X).
        y_test : xarray.DataArray
            Testing predictand data with dimensions (T, Y, X).
        clim_year_start : int
            Start year of the climatology period.
        clim_year_end : int
            End year of the climatology period.
        best_params : dict, optional
            Pre-computed best hyperparameters per cluster. If None, computes internally.
        cluster_da : xarray.DataArray, optional
            Pre-computed cluster labels. If None, computes internally.
        Returns
        -------
        hindcast_prob : xarray.DataArray
            Tercile probabilities with dimensions (probability, T, Y, X).
        """
        # Standardize inputs
        X_train_std = X_train
        y_train_std = y_train
        X_test_std = X_test
        y_test_std = y_test
        # y_test_std not needed for probs
        # Extract coordinate variables from X_test
        time = X_test_std['T']
        lat = X_test_std['Y']
        lon = X_test_std['X']
        n_time = len(time)
        n_lat = len(lat)
        n_lon = len(lon)
        # Compute terciles from training data
        terciles = y_train.quantile([0.33, 0.67], dim='T')
        T1 = terciles.isel(quantile=0).drop_vars('quantile')
        T2 = terciles.isel(quantile=1).drop_vars('quantile')
        # Use provided best_params and cluster_da or compute if None
        if best_params is None:
            best_params, cluster_da = self.compute_hyperparameters(X_train, y_train, clim_year_start, clim_year_end)
        # Initialize probabilities array
        probabilities = np.full((3, n_time, n_lat, n_lon), np.nan)
        self.gp = {}  # Dictionary to store models per cluster
        for c in range(self.n_clusters):
            if c not in best_params:
                continue
            bp = best_params[c]
            # Mask for this cluster
            mask_3d_train = (cluster_da == c).expand_dims({'T': X_train_std['T']})
            mask_3d_test = (cluster_da == c).expand_dims({'T': X_test_std['T']})
            # Stack training data for cluster
            X_train_stacked_c = X_train_std.where(mask_3d_train).stack(sample=('T', 'Y', 'X')).transpose('sample', 'M').values
            y_train_stacked_c = y_train_std.where(mask_3d_train).stack(sample=('T', 'Y', 'X')).values.ravel()
            train_nan_mask = np.any(~np.isfinite(X_train_stacked_c), axis=1) | ~np.isfinite(y_train_stacked_c)
            X_train_clean_c = X_train_stacked_c[~train_nan_mask]
            y_train_clean_c = y_train_stacked_c[~train_nan_mask]
            # Stack testing data for cluster
            X_test_stacked_c = X_test_std.where(mask_3d_test).stack(sample=('T', 'Y', 'X')).transpose('sample', 'M').values
            test_nan_mask = np.any(~np.isfinite(X_test_stacked_c), axis=1)
            X_test_clean_c = X_test_stacked_c[~test_nan_mask]
            # Initialize kernel with best params
            kernel = RBF(bp['kernel__k1__length_scale']) + WhiteKernel(bp['kernel__k2__noise_level'])
            # Initialize GaussianProcessRegressor
            gp_c = GaussianProcessRegressor(kernel=kernel, n_restarts_optimizer=10, random_state=self.random_state)
            # Fit
            gp_c.fit(X_train_clean_c, y_train_clean_c)
            self.gp[c] = gp_c
            # Predict mean and std
            y_pred_c, std_c = gp_c.predict(X_test_clean_c, return_std=True)
            # Compute probs assuming normal distribution
            p_less_t1 = norm.cdf(T1.values.flatten()[0], loc=y_pred_c, scale=std_c)
            p_less_t2 = norm.cdf(T2.values.flatten()[0], loc=y_pred_c, scale=std_c)
            p_b = p_less_t1
            p_n = p_less_t2 - p_less_t1
            p_a = 1 - p_less_t2
            probs_c = np.stack([p_b, p_n, p_a], axis=0)
            # Reconstruct probabilities for this cluster
            result_c = np.full((3, len(test_nan_mask)), np.nan)
            result_c[:, ~test_nan_mask] = probs_c
            probs_reshaped = result_c.reshape(3, n_time, n_lat, n_lon)
            # Fill in the probabilities array
            probabilities = np.where(np.isnan(probabilities), probs_reshaped, probabilities)
        hindcast_prob = xr.DataArray(
            data=probabilities,
            coords={'probability': ['PB', 'PN', 'PA'], 'T': time, 'Y': lat, 'X': lon},
            dims=['probability', 'T', 'Y', 'X']
        )
        return hindcast_prob

    def forecast(self, Predictant, clim_year_start, clim_year_end, hindcast_det, hindcast_det_cross, Predictor_for_year, best_params=None, cluster_da=None):
        """
        Generate probabilistic forecast for a target year using the Gaussian Process model with optimized hyperparameters.
        Returns the tercile probabilities directly.
        Parameters
        ----------
        Predictant : xarray.DataArray
            Observed predictand data with dimensions (T, Y, X) or (T, M, Y, X).
        clim_year_start : int or str
            Start year of the climatology period.
        clim_year_end : int or str
            End year of the climatology period.
        hindcast_det : xarray.DataArray
            Deterministic hindcast data for training with dimensions (T, M, Y, X).
        hindcast_det_cross : xarray.DataArray
            Deterministic hindcast data for error estimation with dimensions (T, Y, X).
        Predictor_for_year : xarray.DataArray
            Predictor data for the target year with dimensions (T, M, Y, X).
        best_params : dict, optional
            Pre-computed best hyperparameters. If None, computes internally.
        cluster_da : xarray.DataArray, optional
            Pre-computed cluster labels. If None, computes internally.
        Returns
        -------
        forecast_prob : xarray.DataArray
            Tercile probabilities with dimensions (probability, T, Y, X).
        """
        if "M" in Predictant.coords:
            Predictant_no_m = Predictant.isel(M=0).drop_vars('M').squeeze()
        else:
            Predictant_no_m = Predictant
        mask = xr.where(~np.isnan(Predictant_no_m.isel(T=0)), 1, np.nan).drop_vars(['T']).squeeze().to_numpy()
        # Standardize Predictor_for_year using hindcast climatology
        mean_val = hindcast_det.sel(T=slice(str(clim_year_start), str(clim_year_end))).mean(dim='T')
        std_val = hindcast_det.sel(T=slice(str(clim_year_start), str(clim_year_end))).std(dim='T')
        Predictor_for_year_st = (Predictor_for_year - mean_val) / std_val
        hindcast_det_st = standardize_timeseries(hindcast_det, clim_year_start, clim_year_end)
        Predictant_st = standardize_timeseries(Predictant_no_m, clim_year_start, clim_year_end)
        # Extract coordinates from X_test
        time = Predictor_for_year_st['T']
        lat = Predictor_for_year_st['Y']
        lon = Predictor_for_year_st['X']
        n_time = len(time)
        n_lat = len(lat)
        n_lon = len(lon)
        # Compute terciles
        index_start = Predictant_no_m.get_index("T").get_loc(str(clim_year_start)).start
        index_end = Predictant_no_m.get_index("T").get_loc(str(clim_year_end)).stop
        rainfall_for_tercile = Predictant_no_m.isel(T=slice(index_start, index_end))
        terciles = rainfall_for_tercile.quantile([0.33, 0.67], dim='T')
        T1 = terciles.isel(quantile=0).drop_vars('quantile')
        T2 = terciles.isel(quantile=1).drop_vars('quantile')
        # Use provided best_params and cluster_da or compute if None
        if best_params is None:
            best_params, cluster_da = self.compute_hyperparameters(hindcast_det, Predictant_no_m, clim_year_start, clim_year_end)
        # Initialize probabilities array
        probabilities = np.full((3, n_time, n_lat, n_lon), np.nan)
        self.gp = {}  # Dictionary to store models per cluster
        for c in range(self.n_clusters):
            if c not in best_params:
                continue
            bp = best_params[c]
            # Mask for this cluster
            mask_3d_train = (cluster_da == c).expand_dims({'T': hindcast_det_st['T']})
            mask_3d_test = (cluster_da == c).expand_dims({'T': Predictor_for_year_st['T']})
            # Stack training data for cluster
            X_train_stacked_c = hindcast_det_st.where(mask_3d_train).stack(sample=('T', 'Y', 'X')).transpose('sample', 'M').values
            y_train_stacked_c = Predictant_st.where(mask_3d_train).stack(sample=('T', 'Y', 'X')).values.ravel()
            train_nan_mask = np.any(~np.isfinite(X_train_stacked_c), axis=1) | ~np.isfinite(y_train_stacked_c)
            X_train_clean_c = X_train_stacked_c[~train_nan_mask]
            y_train_clean_c = y_train_stacked_c[~train_nan_mask]
            # Stack testing data for cluster
            X_test_stacked_c = Predictor_for_year_st.where(mask_3d_test).stack(sample=('T', 'Y', 'X')).transpose('sample', 'M').values
            test_nan_mask = np.any(~np.isfinite(X_test_stacked_c), axis=1)
            X_test_clean_c = X_test_stacked_c[~test_nan_mask]
            # Initialize kernel with best params
            kernel = RBF(bp['kernel__k1__length_scale']) + WhiteKernel(bp['kernel__k2__noise_level'])
            # Initialize GaussianProcessRegressor
            gp_c = GaussianProcessRegressor(kernel=kernel, n_restarts_optimizer=10, random_state=self.random_state)
            # Fit
            gp_c.fit(X_train_clean_c, y_train_clean_c)
            self.gp[c] = gp_c
            # Predict mean and std
            y_pred_c, std_c = gp_c.predict(X_test_clean_c, return_std=True)
            # Compute probs assuming normal distribution
            p_less_t1 = norm.cdf(T1.values.flatten()[0], loc=y_pred_c, scale=std_c)
            p_less_t2 = norm.cdf(T2.values.flatten()[0], loc=y_pred_c, scale=std_c)
            p_b = p_less_t1
            p_n = p_less_t2 - p_less_t1
            p_a = 1 - p_less_t2
            probs_c = np.stack([p_b, p_n, p_a], axis=0)
            # Reconstruct probabilities for this cluster
            result_c = np.full((3, len(test_nan_mask)), np.nan)
            result_c[:, ~test_nan_mask] = probs_c
            probs_reshaped = result_c.reshape(3, n_time, n_lat, n_lon)
            # Fill in the probabilities array
            probabilities = np.where(np.isnan(probabilities), probs_reshaped, probabilities)
        hindcast_prob = xr.DataArray(
            data=probabilities,
            coords={'probability': ['PB', 'PN', 'PA'], 'T': time, 'Y': lat, 'X': lon},
            dims=['probability', 'T', 'Y', 'X']
        )
        year = Predictor_for_year.coords['T'].values.astype('datetime64[Y]').astype(int)[0] + 1970
        T_value_1 = Predictant_no_m.isel(T=0).coords['T'].values
        month_1 = T_value_1.astype('datetime64[M]').astype(int) % 12 + 1
        new_T_value = np.datetime64(f"{year}-{month_1:02d}-01")
        hindcast_prob = hindcast_prob.assign_coords(T=xr.DataArray([new_T_value], dims=["T"]))
        hindcast_prob['T'] = hindcast_prob['T'].astype('datetime64[ns]')
        return hindcast_prob    


class WAS_mme_GP: # Correction proposed
    """
    Gaussian Process-based MME with cluster-wise hyperparameter tuning
    and probabilistic outputs (tercile probabilities).
    """

    def __init__(self,
                 length_scale_range=None,
                 noise_level_range=None,
                 random_state=42,
                 dist_method="normal",
                 n_iter_search=10,
                 cv_folds=3,
                 n_clusters=4):

        self.length_scale_range = length_scale_range or np.logspace(-1, 1, 5).tolist()
        self.noise_level_range  = noise_level_range  or np.logspace(-5, -1, 5).tolist()
        self.random_state = random_state
        self.dist_method  = dist_method  # currently 'normal' only
        self.n_iter_search = n_iter_search
        self.cv_folds = cv_folds
        self.n_clusters = n_clusters

        self.gp = {}              # fitted GP per cluster id
        self.best_params_ = None  # best params per cluster
        self.cluster_da_ = None   # (Y,X) cluster map

    # --------- helpers ---------
    @staticmethod
    def _ensure_dims(P, want):
        # rename anonymous dims to wanted names in order
        if list(P.dims) != want:
            mapping = {old:new for old,new in zip(P.dims, want)}
            P = P.rename(mapping)
        return P

    @staticmethod
    def _stack_predictors(X):
        # X dims: (T,M,Y,X) -> (sample, M) with sample = (T,Y,X)
        X = X.transpose("T","M","Y","X")
        return X.stack(sample=("T","Y","X")).transpose("sample","M")

    @staticmethod
    def _stack_target(y):
        # y dims: (T,Y,X) -> (sample,)
        y = y.transpose("T","Y","X")
        return y.stack(sample=("T","Y","X"))

    @staticmethod
    def _make_mask3(cluster_da, T_coord):
        # cluster_da: (Y,X), T_coord: (T,)
        return cluster_da.expand_dims(T=T_coord)

    @staticmethod
    def _bool_index_from_mask(mask3):
        # mask3 dims (T,Y,X) -> boolean index over stacked (T,Y,X)
        return mask3.stack(sample=("T","Y","X")).values.astype(bool)

    @staticmethod
    def _repeat_terciles_over_time(Tgrid, n_time):
        # Tgrid: (Y,X) -> (T,Y,X) repeated along T
        return xr.concat([Tgrid]*n_time, dim="T")

    # --------- clustering & hyperparameter search ---------
    def compute_hyperparameters(self, Predictors, Predictand, clim_year_start, clim_year_end):
        """
        Tune kernel hyperparameters per cluster using randomized search.
        """
        # drop M if present on predictand (should be T,Y,X)
        if "M" in Predictand.dims:
            Predictand = Predictand.isel(M=0, drop=True)

        # Standardize along T using your helper
        X_train_std = standardize_timeseries(Predictors, clim_year_start, clim_year_end)
        y_train_std = standardize_timeseries(Predictand, clim_year_start, clim_year_end)
        # build a spatial field to cluster on (e.g., mean over T)
        y_clim = y_train_std.mean("T", skipna=True)  # (Y,X)

        # KMeans on valid pixels of y_clim
        df = y_clim.to_dataframe(name="val").reset_index().dropna(subset=["val"])
        kmeans = KMeans(n_clusters=self.n_clusters, random_state=self.random_state, n_init="auto")
        df["cluster"] = kmeans.fit_predict(df[["val"]])

        # back to xarray
        cluster_da = (
            df[["Y","X","cluster"]]
            .drop_duplicates(subset=["Y","X"])
            .set_index(["Y","X"])
            .to_xarray()["cluster"]
            .astype(int)
        )

        # mask cluster map by data availability
        maskYX = xr.where(~np.isnan(y_clim), 1, np.nan)
        cluster_da = (cluster_da * maskYX).astype(float)  # NaN where invalid
        cluster_da = cluster_da.transpose("Y","X")

        # align to predictor/predictand grid
        _, cluster_da = xr.align(y_clim, cluster_da, join="outer")

        # stack training data
        Xs = self._stack_predictors(self._ensure_dims(X_train_std, ["T","M","Y","X"]))
        ys = self._stack_target(self._ensure_dims(y_train_std, ["T","Y","X"]))

        # search space
        param_dist = {
            "kernel__k1__length_scale": self.length_scale_range,
            "kernel__k2__noise_level":  self.noise_level_range
        }

        best_params = {}
        clusters = np.unique(cluster_da.values[~np.isnan(cluster_da.values)]).astype(int)

        for c in clusters:
            mask3 = self._make_mask3(cluster_da==c, X_train_std["T"])
            idx = self._bool_index_from_mask(mask3)
            X_c = Xs.values[idx, :]
            y_c = ys.values[idx]

            # drop NaNs
            ok = np.isfinite(y_c) & np.all(np.isfinite(X_c), axis=1)
            X_c = X_c[ok]; y_c = y_c[ok]
            if X_c.shape[0] < max(5, self.cv_folds):  # not enough samples
                continue

            # base kernel
            kernel = RBF(1.0) + WhiteKernel(1e-5)
            gpr = GaussianProcessRegressor(kernel=kernel,
                                           n_restarts_optimizer=5,
                                           random_state=self.random_state,
                                           normalize_y=False)

            search = RandomizedSearchCV(
                gpr,
                param_distributions=param_dist,
                n_iter=self.n_iter_search,
                cv=self.cv_folds,
                scoring="neg_mean_squared_error",
                random_state=self.random_state,
                error_score=np.nan,
            )
            search.fit(X_c, y_c)
            best_params[c] = search.best_params_

        self.best_params_ = best_params
        self.cluster_da_  = cluster_da
        return best_params, cluster_da

    # --------- core fit/predict over a test span ---------
    def compute_model(self, X_train, y_train, X_test, y_test,
                      clim_year_start, clim_year_end,
                      best_params=None, cluster_da=None):
        """
        Returns tercile probabilities (PB, PN, PA) over X_test time span.
        """
        # standardize with training climatology (user function)
        X_train_std = standardize_timeseries(X_train, clim_year_start, clim_year_end)
        y_train_std = standardize_timeseries(y_train.isel(M=0, drop=True) if "M" in y_train.dims else y_train,
                                             clim_year_start, clim_year_end)
        X_test_std  = (X_test - X_train.sel(T=slice(str(clim_year_start), str(clim_year_end))).mean("T")) \
                      / (X_train.sel(T=slice(str(clim_year_start), str(clim_year_end))).std("T"))

        # coords & sizes
        X_test_std = self._ensure_dims(X_test_std, ["T","M","Y","X"])
        y_train    = self._ensure_dims(y_train,    ["T","Y","X"])
        time, lat, lon = X_test_std["T"], X_test_std["Y"], X_test_std["X"]
        n_time, n_lat, n_lon = len(time), len(lat), len(lon)

        # terciles from **raw y_train** over climatology years
        y_clim_slice = y_train.sel(T=slice(str(clim_year_start), str(clim_year_end)))
        Tq = y_clim_slice.quantile([0.33, 0.67], dim="T", skipna=True)  # (quantile,Y,X)
        T1 = Tq.sel(quantile=0.33)  # (Y,X)
        T2 = Tq.sel(quantile=0.67)

        # ensure params/clusters
        if best_params is None or cluster_da is None:
            best_params, cluster_da = self.compute_hyperparameters(X_train, y_train, clim_year_start, clim_year_end)

        # stack arrays
        Xs_tr = self._stack_predictors(self._ensure_dims(X_train_std, ["T","M","Y","X"]))
        ys_tr = self._stack_target(self._ensure_dims(y_train_std,   ["T","Y","X"]))
        Xs_te = self._stack_predictors(X_test_std)

        # precompute global boolean index per cluster for train/test
        probs = np.full((3, n_time*n_lat*n_lon), np.nan, dtype=float)

        # thresholds repeated along time and flattened to match sample order (T,Y,X)
        T1_rep = self._repeat_terciles_over_time(T1, n_time).transpose("T","Y","X").values.reshape(-1)
        T2_rep = self._repeat_terciles_over_time(T2, n_time).transpose("T","Y","X").values.reshape(-1)

        clusters = np.unique(cluster_da.values[~np.isnan(cluster_da.values)]).astype(int)

        self.gp = {}
        for c in clusters:
            if c not in best_params:
                continue

            bp = best_params[c]
            kernel = RBF(bp["kernel__k1__length_scale"]) + WhiteKernel(bp["kernel__k2__noise_level"])

            # build masks
            m_tr = self._bool_index_from_mask(self._make_mask3(cluster_da==c, X_train_std["T"]))
            m_te = self._bool_index_from_mask(self._make_mask3(cluster_da==c, X_test_std["T"]))

            X_tr_c = Xs_tr.values[m_tr, :]
            y_tr_c = ys_tr.values[m_tr]
            ok_tr  = np.isfinite(y_tr_c) & np.all(np.isfinite(X_tr_c), axis=1)
            X_tr_c = X_tr_c[ok_tr]; y_tr_c = y_tr_c[ok_tr]
            if X_tr_c.shape[0] < max(5, self.cv_folds):
                continue

            gp = GaussianProcessRegressor(kernel=kernel,
                                          n_restarts_optimizer=5,
                                          random_state=self.random_state,
                                          normalize_y=False)
            gp.fit(X_tr_c, y_tr_c)
            self.gp[c] = gp

            # test for this cluster
            X_te_c = Xs_te.values[m_te, :]
            ok_te  = np.all(np.isfinite(X_te_c), axis=1)
            idx_c  = np.where(m_te)[0][ok_te]           # positions in flattened sample space
            if idx_c.size == 0:
                continue

            y_mu, y_std = gp.predict(X_te_c[ok_te], return_std=True)
            y_std = np.maximum(y_std, 1e-6)            # numerical floor

            # thresholds aligned to these positions
            t1_c = T1_rep[idx_c]
            t2_c = T2_rep[idx_c]

            # map back to **raw** space: GP was trained on standardized y,
            # but we used y_train_std => y_mu, y_std are in standardized units.
            # To produce probabilities against raw terciles, we can keep everything in standardized space:
            # standardize terciles with the same transform used on y_train.
            y_mean = y_train.sel(T=slice(str(clim_year_start), str(clim_year_end))).mean("T").values
            y_std0 = y_train.sel(T=slice(str(clim_year_start), str(clim_year_end))).std("T").values
            # repeat along time and index to cluster positions
            y_mean_rep = np.repeat(y_mean.reshape(1, n_lat, n_lon), n_time, axis=0).reshape(-1)[idx_c]
            y_std0_rep = np.repeat(y_std0.reshape(1, n_lat, n_lon), n_time, axis=0).reshape(-1)[idx_c]
            y_std0_rep = np.where(y_std0_rep<=0, 1.0, y_std0_rep)

            t1_c_std = (t1_c - y_mean_rep) / y_std0_rep
            t2_c_std = (t2_c - y_mean_rep) / y_std0_rep

            # probabilities in standardized space
            p_b = norm.cdf(t1_c_std, loc=y_mu, scale=y_std)
            p_a = 1.0 - norm.cdf(t2_c_std, loc=y_mu, scale=y_std)
            p_n = 1.0 - p_b - p_a

            # write back
            probs[0, idx_c] = p_b
            probs[1, idx_c] = p_n
            probs[2, idx_c] = p_a

        # reshape to (probability, T, Y, X)
        probs = probs.reshape(3, n_time, n_lat, n_lon)
        hindcast_prob = xr.DataArray(
            probs,
            coords={"probability": ["PB","PN","PA"], "T": time, "Y": lat, "X": lon},
            dims=("probability","T","Y","X"),
            name="tercile_probabilities"
        )
        return hindcast_prob

    # --------- one-year forecast wrapper ---------
    def forecast(self, Predictant, clim_year_start, clim_year_end,
                 hindcast_det, hindcast_det_cross, Predictor_for_year,
                 best_params=None, cluster_da=None):
        """
        Probabilistic forecast for a target year (PB, PN, PA).
        """
        # ensure target has no M
        if "M" in Predictant.dims:
            Predictant = Predictant.isel(M=0, drop=True)

        # build a fake X_test span of T=1 for that target year
        hindcast_prob = self.compute_model(
            X_train=hindcast_det,
            y_train=Predictant,
            X_test=Predictor_for_year,
            y_test=None,
            clim_year_start=clim_year_start,
            clim_year_end=clim_year_end,
            best_params=best_params,
            cluster_da=cluster_da
        )

        # keep T coord from Predictor_for_year (already one-step)
        if "T" in Predictor_for_year.coords:
            hindcast_prob = hindcast_prob.assign_coords(T=Predictor_for_year["T"])
        return hindcast_prob


    
############################################ WAS Genetic Algorithm From ROBBER 2013 and 2015 ############################################

NUM_GENES = 10 # As specified by Roebber (2013, 2015)
DEFAULT_TOURNAMENT_SIZE = 3 # For tournament selection

class Gene:
    """Represents a single gene in an individual's prediction equation."""
    # Operators as described in the paper
    OPERATORS = {
        'ADD': operator.add,
        'MULTIPLY': operator.mul
    }
    RELATIONAL_OPERATORS = {
        '<=': operator.le,
        '>': operator.gt
    }

    def __init__(self, predictor_names):
        """
        Initialize a gene.

        Parameters
        ----------
        predictor_names : list of str
            Names of available predictors (e.g., ensemble members, ensemble stats, covariates).
        """
        self.predictor_names = predictor_names

        # Initialize gene elements randomly
        # We store names; actual values will be looked up during evaluation
        self.v1_name = random.choice(self.predictor_names)
        self.v2_name = random.choice(self.predictor_names)
        self.v3_name = random.choice(self.predictor_names)
        self.v4_name = random.choice(self.predictor_names)
        self.v5_name = random.choice(self.predictor_names)

        # Multiplicative constants in the range -1 to +1
        self.c1 = random.uniform(-1, 1)
        self.c2 = random.uniform(-1, 1)
        self.c3 = random.uniform(-1, 1)

        # Operators: addition or multiplication
        self.O1 = random.choice(list(self.OPERATORS.keys()))
        self.O2 = random.choice(list(self.OPERATORS.keys()))
        # Relational operator: '<=' or '>'
        self.OR = random.choice(list(self.RELATIONAL_OPERATORS.keys()))

    def evaluate(self, data_row_dict):
        """
        Evaluates the gene's contribution for a given data row (t).
        Corresponds to Eq. 3.83.

        Parameters
        ----------
        data_row_dict : dict
            A dictionary where keys are predictor names and values are their corresponding
            normalized numerical values for a single sample. Values are expected to be in [-1, 1].

        Returns
        -------
        float
            The calculated value of the gene for the given data row, or 0 if the condition is false.
        """
        # Get the actual values for the chosen predictors from the data row dict
        # Assume data_row_dict already contains normalized values within [-1, 1]
        try:
            val_v1 = data_row_dict[self.v1_name]
            val_v2 = data_row_dict[self.v2_name]
            val_v3 = data_row_dict[self.v3_name]
            val_v4 = data_row_dict[self.v4_name]
            val_v5 = data_row_dict[self.v5_name]
        except KeyError:
            # If a predictor chosen by the gene is not found in the current data row,
            # this indicates an issue with data consistency or `predictor_names` list.
            # Returning 0 effectively disables this gene for this sample.
            return 0

        # Apply operators
        op1_func = self.OPERATORS[self.O1]
        op2_func = self.OPERATORS[self.O2]
        or_func = self.RELATIONAL_OPERATORS[self.OR]

        # Condition for gene activation: if v4 OR v5 is true
        if or_func(val_v4, val_v5):
            term1 = self.c1 * val_v1
            term2 = self.c2 * val_v2
            term3 = self.c3 * val_v3
            
            result_op1 = op1_func(term1, term2)
            result_op2 = op2_func(result_op1, term3)
            return result_op2
        else:
            return 0

    def mutate(self):
        """Randomly replaces one of the 11 elements of the gene."""
        # List of all mutable elements in a gene
        elements_to_mutate = [
            'v1_name', 'v2_name', 'v3_name', 'v4_name', 'v5_name',
            'c1', 'c2', 'c3', 'O1', 'O2', 'OR'
        ]
        element_to_mutate = random.choice(elements_to_mutate)

        if element_to_mutate in ['v1_name', 'v2_name', 'v3_name', 'v4_name', 'v5_name']:
            # Choose a new predictor name from the available ones
            self.__setattr__(element_to_mutate, random.choice(self.predictor_names))
        elif element_to_mutate in ['c1', 'c2', 'c3']:
            # Choose a new random constant in [-1, 1]
            self.__setattr__(element_to_mutate, random.uniform(-1, 1))
        elif element_to_mutate in ['O1', 'O2']:
            # Choose a new operator (ADD or MULTIPLY)
            self.__setattr__(element_to_mutate, random.choice(list(self.OPERATORS.keys())))
        elif element_to_mutate == 'OR':
            # Choose a new relational operator (<= or >)
            self.__setattr__(element_to_mutate, random.choice(list(self.RELATIONAL_OPERATORS.keys())))

    def copy(self):
        """Returns a deep copy of the gene."""
        new_gene = Gene(self.predictor_names) # Initialize with the same available predictors
        # Copy all attributes explicitly
        new_gene.v1_name = self.v1_name
        new_gene.v2_name = self.v2_name
        new_gene.v3_name = self.v3_name
        new_gene.v4_name = self.v4_name
        new_gene.v5_name = self.v5_name
        new_gene.c1 = self.c1
        new_gene.c2 = self.c2
        new_gene.c3 = self.c3
        new_gene.O1 = self.O1
        new_gene.O2 = self.O2
        new_gene.OR = self.OR
        return new_gene


class Individual:
    """Represents a single prediction equation (an individual in the population)."""
    def __init__(self, predictor_names):
        """
        Initializes an individual with NUM_GENES genes.

        Parameters
        ----------
        predictor_names : list of str
            Names of available predictors to be used by genes.
        """
        self.genes = [Gene(predictor_names) for _ in range(NUM_GENES)]
        self.mse = float('inf')  # Initialize MSE (fitness) to a high value

    def calculate_mse(self, training_data_rows, true_predictands_normalized):
        """
        Calculates the MSE for the individual's prediction.
        Corresponds to Eq. 3.84.

        Parameters
        ----------
        training_data_rows : list of dict
            Each dict represents a data row, with keys being predictor names
            and values being their normalized numerical values.
        true_predictands_normalized : np.ndarray
            Normalized true 'yt' values corresponding to training_data_rows.
        """
        n = len(training_data_rows)
        if n == 0: # Handle empty training data
            self.mse = float('inf')
            return

        predictions = []
        for t in range(n):
            row_prediction_sum_of_genes = 0
            for gene in self.genes:
                row_prediction_sum_of_genes += gene.evaluate(training_data_rows[t])
            predictions.append(row_prediction_sum_of_genes)

        # Compute Mean Squared Error
        squared_errors = (true_predictands_normalized - np.array(predictions))**2
        self.mse = np.mean(squared_errors)

    def reproduce(self, mutation_rate, crossover_rate):
        """
        Reproduces the individual, potentially with mutation or crossover.
        Returns a new individual (offspring).
        """
        # Create a new offspring individual by copying the parent's structure
        offspring = Individual(self.genes[0].predictor_names) # Pass predictor names from parent's gene
        offspring.genes = [gene.copy() for gene in self.genes] # Deep copy all genes

        # Apply mutation
        if random.random() < mutation_rate:
            gene_to_mutate = random.choice(offspring.genes)
            gene_to_mutate.mutate()

        # Apply genetic crossover (intra-individual as per the paper's description)
        if random.random() < crossover_rate and NUM_GENES >= 2: # Crossover needs at least two genes
            gene1_idx = random.randrange(NUM_GENES)
            gene2_idx = random.randrange(NUM_GENES)
            while gene1_idx == gene2_idx: # Ensure different genes are selected for crossover
                gene2_idx = random.randrange(NUM_GENES)

            gene1 = offspring.genes[gene1_idx]
            gene2 = offspring.genes[gene2_idx]

            # The four underlined elements/groups from the paper:
            # 1. (c1, v1_name, O1)
            # 2. (c2, v2_name, O2)
            # 3. (c3, v3_name)
            # 4. (v4_name, OR, v5_name)
            crossover_group = random.choice(['group1', 'group2', 'group3', 'group4'])

            if crossover_group == 'group1':
                gene1.c1, gene2.c1 = gene2.c1, gene1.c1
                gene1.v1_name, gene2.v1_name = gene2.v1_name, gene1.v1_name
                gene1.O1, gene2.O1 = gene2.O1, gene1.O1
            elif crossover_group == 'group2':
                gene1.c2, gene2.c2 = gene2.c2, gene1.c2
                gene1.v2_name, gene2.v2_name = gene2.v2_name, gene1.v2_name
                gene1.O2, gene2.O2 = gene2.O2, gene1.O2
            elif crossover_group == 'group3':
                gene1.c3, gene2.c3 = gene2.c3, gene1.c3
                gene1.v3_name, gene2.v3_name = gene2.v3_name, gene1.v3_name
            elif crossover_group == 'group4':
                gene1.v4_name, gene2.v4_name = gene2.v4_name, gene1.v4_name
                gene1.OR, gene2.OR = gene2.OR, gene1.OR
                gene1.v5_name, gene2.v5_name = gene2.v5_name, gene1.v5_name
        return offspring

    def copy(self):
        """Returns a deep copy of the individual."""
        new_individual = Individual(self.genes[0].predictor_names)
        new_individual.genes = [gene.copy() for gene in self.genes]
        new_individual.mse = self.mse
        return new_individual


class WAS_mme_RoebberGA:
    """
    Genetic Algorithm-based Statistical Learning Method adapted from Roebber (2013, 2015).

    This class evolves complex prediction equations composed of "genes" to minimize MSE.
    It is designed to work with xarray DataArray inputs for training and prediction.

    Parameters
    ----------
    population_size : int, optional
        Number of individuals in the GA population. Default is 50 (increased for better convergence).
    max_iter : int, optional
        Maximum number of generations for the GA. Default is 100 (increased for better search).
    crossover_rate : float, optional
        Probability of performing crossover. Default is 0.7.
    mutation_rate : float, optional
        Probability of mutating a gene. Default is 0.05 (slightly increased for more exploration).
    random_state : int, optional
        Seed for random number generation. Default is 42.
    dist_method : str, optional
        Default is "gamma".
    elite_size : int, optional
        Number of top individuals to use for ensemble prediction to reduce overfitting. Default is 5.
    """

    def __init__(self,
                 population_size=50,
                 max_iter=100,
                 crossover_rate=0.7,
                 mutation_rate=0.05,
                 random_state=42,
                 dist_method="gamma",
                 elite_size=5):
        
        self.population_size = population_size
        self.max_iter = max_iter
        self.crossover_rate = crossover_rate
        self.mutation_rate = mutation_rate
        self.random_state = random_state
        self.dist_method = dist_method 
        self.elite_size = elite_size

        # Set seeds for reproducibility
        random.seed(self.random_state)
        np.random.seed(self.random_state)

        # Best ensemble found by GA during training
        self.best_ensemble = None
        self.best_individual = None  # Still track the single best for reference
        self.best_fitness = float('-inf') # Store negative MSE (maximized)

        # Store normalization ranges for consistency between train/test
        self._y_normalization_min_max = None
        self._predictor_normalization_ranges = {} # Dict: {predictor_name: (min, max)}

    def _normalize_array(self, arr):
        """Normalize array values to the interval [-1, 1]. Handles NaNs and constant arrays."""
        min_val = np.nanmin(arr)
        max_val = np.nanmax(arr)
        if max_val == min_val: # Avoid division by zero for constant arrays
            # If all values are the same, they should map to 0 in [-1, 1] range.
            return np.zeros_like(arr, dtype=float)
        return 2 * ((arr - min_val) / (max_val - min_val)) - 1

    def _denormalize_array(self, normalized_arr, original_min, original_max):
        """Denormalize array values from [-1, 1] back to original range. Handles constant ranges."""
        if original_max == original_min:
            # If original range was constant, all denormalized values should be that constant.
            return np.full_like(normalized_arr, original_min, dtype=float)
        return ((normalized_arr + 1) / 2) * (original_max - original_min) + original_min

    def _prepare_data_for_ga_core(self, X_da, y_da):
        """
        Prepares xarray DataArrays for the core GA logic:
        - Stacks dimensions (T,Y,X) into a 'sample' dimension.
        - Identifies predictor names from the 'M' dimension.
        - Handles NaNs.
        - Normalizes X and y values to [-1, 1] and stores normalization parameters.

        Parameters
        ----------
        X_da : xarray.DataArray
            Input predictor data (e.g., (T,Y,X,M)).
        y_da : xarray.DataArray
            Input predictand data (e.g., (T,Y,X)).

        Returns
        -------
        tuple
            (X_rows_normalized: list of dicts,
             y_normalized: np.ndarray,
             predictor_names: list of str,
             nan_mask_initial: np.ndarray)
            Returns empty lists/arrays if no valid data.
        """
        # 1. Get predictor names from the 'M' dimension
        if 'M' not in X_da.dims:
            raise ValueError("X_da must have an 'M' dimension representing predictors (e.g., models, covariates).")
        predictor_names = X_da['M'].values.tolist()

        # 2. Stack/reshape X and y (T,Y,X) -> 'sample' dimension
        X_stacked_raw = X_da.stack(sample=('T','Y','X')).transpose('sample','M').values
        y_stacked_raw = y_da.stack(sample=('T','Y','X')).values

        # Ensure y_stacked_raw is 1D
        if y_stacked_raw.ndim == 2 and y_stacked_raw.shape[1] == 1:
            y_stacked_raw = y_stacked_raw.ravel()

        # 3. Identify NaNs across both X and y
        # A sample is invalid if any of its predictors or the target is NaN
        nan_mask_initial = (np.any(~np.isfinite(X_stacked_raw), axis=1) |
                            ~np.isfinite(y_stacked_raw))

        X_clean_raw = X_stacked_raw[~nan_mask_initial]
        y_clean_raw = y_stacked_raw[~nan_mask_initial]

        if X_clean_raw.size == 0 or y_clean_raw.size == 0:
            return [], np.array([]), predictor_names, nan_mask_initial # Return empty if no valid data

        # 4. Normalize X and y
        # Store normalization parameters for X (per predictor) and y
        self._predictor_normalization_ranges = {}
        X_normalized_columns = []
        for i, pred_name in enumerate(predictor_names):
            col_data = X_clean_raw[:, i]
            col_min = np.nanmin(col_data)
            col_max = np.nanmax(col_data)
            self._predictor_normalization_ranges[pred_name] = (col_min, col_max)
            X_normalized_columns.append(self._normalize_array(col_data))
        
        # Reconstruct X_clean_normalized as a list of dicts for Gene.evaluate
        X_clean_normalized_rows = []
        if X_normalized_columns: # Only proceed if there are valid predictors
            X_normalized_array = np.array(X_normalized_columns).T # Transpose back to (sample, predictor)
            for row_idx in range(X_normalized_array.shape[0]):
                row_dict = {predictor_names[i]: X_normalized_array[row_idx, i]
                            for i in range(len(predictor_names))}
                X_clean_normalized_rows.append(row_dict)
        else:
            X_clean_normalized_rows = [] # No valid predictors means no valid rows

        # Normalize y and store its original range
        y_min_orig = np.nanmin(y_clean_raw)
        y_max_orig = np.nanmax(y_clean_raw)
        self._y_normalization_min_max = (y_min_orig, y_max_orig)
        y_clean_normalized = self._normalize_array(y_clean_raw)

        return X_clean_normalized_rows, y_clean_normalized, predictor_names, nan_mask_initial

    def _tournament_selection(self, population, tournament_size=DEFAULT_TOURNAMENT_SIZE):
        """
        Selects an individual using tournament selection.

        Parameters
        ----------
        population : list of Individual
            The current population.
        tournament_size : int, optional
            Number of individuals to compete in the tournament. Default is 3.

        Returns
        -------
        Individual
            The winner of the tournament (lowest MSE).
        """
        candidates = random.sample(population, tournament_size)
        return min(candidates, key=lambda x: x.mse)

    def _run_ga_core(self, X_train_rows, y_train_normalized, predictor_names):
        """
        Core Genetic Algorithm logic.

        Parameters
        ----------
        X_train_rows : list of dict
            Normalized training predictor data (list of sample dicts).
        y_train_normalized : np.ndarray
            Normalized training predictand data.
        predictor_names : list of str
            Names of predictors available to genes.

        Returns
        -------
        list of Individual
            The best ensemble of prediction equations found by the GA.
        """
        population = []
        # Initialize population
        for _ in range(self.population_size):
            individual = Individual(predictor_names)
            individual.calculate_mse(X_train_rows, y_train_normalized)
            population.append(individual)

        # Initialize best individual and fitness
        population.sort(key=lambda x: x.mse)
        self.best_individual = population[0].copy()
        self.best_fitness = -self.best_individual.mse

        print(f"--- Initial Population Min MSE: {population[0].mse:.4f} ---")

        for generation in range(self.max_iter):
            # No fixed threshold; use the entire population for selection

            # Sort population by MSE (lower is better)
            population.sort(key=lambda x: x.mse)

            # Update overall best individual (elitism)
            if population[0].mse < self.best_individual.mse:
                self.best_individual = population[0].copy()
                self.best_fitness = -population[0].mse

            # 2. Reproduction: Create the next generation
            new_population = []
            # Elitism: Carry over the top elite individual
            new_population.append(self.best_individual.copy())

            # Fill the rest of the new population using tournament selection for parents
            while len(new_population) < self.population_size:
                # Select parent using tournament selection
                parent = self._tournament_selection(population)
                
                # Create offspring
                offspring = parent.reproduce(self.mutation_rate, self.crossover_rate)
                new_population.append(offspring)

            # Update population
            population = new_population

            # Calculate MSE for the new population
            for individual in population:
                individual.calculate_mse(X_train_rows, y_train_normalized)
            
            # Track current generation's best for logging
            current_gen_best_mse = min([ind.mse for ind in population])
            print(f"Generation {generation + 1} Best MSE: {current_gen_best_mse:.4f} (Overall Best: {self.best_individual.mse:.4f})")

            # Stopping criterion (e.g., very low MSE reached)
            if self.best_individual.mse < 0.001:
                print(f"Stopping criterion met: Overall Best MSE {self.best_individual.mse:.4f}")
                break
        
        # At end, sort final population and select top elite_size for ensemble
        population.sort(key=lambda x: x.mse)
        self.best_ensemble = [ind.copy() for ind in population[:self.elite_size]]
        print(f"--- GA training finished. Overall Best MSE: {self.best_individual.mse:.4f} ---")
        return self.best_ensemble

    def compute_model(self, X_train, y_train, X_test, y_test=None):
        """
        Trains the Roebber GA model and makes predictions on test data using xarray DataArrays.
        Uses ensemble mean from top individuals for prediction to improve robustness.

        Parameters
        ----------
        X_train : xarray.DataArray
            Training predictor data with dimensions (T, Y, X, M).
        y_train : xarray.DataArray
            Training predictand data with dimensions (T, Y, X).
        X_test : xarray.DataArray
            Testing predictor data with dimensions (T, Y, X, M).
        y_test : xarray.DataArray, optional
            Test predictand data.  Default is None.

        Returns
        -------
        predicted_da : xarray.DataArray
            Predictions with dimensions (T, Y, X), denormalized to original y_train range.
        """
        # Ensure y_train is consistently 1D if it has a singleton last dimension
        if y_train.ndim == 3: # Wait, for (T,Y,X), ndim=3
            pass
        if y_train.ndim == 4 and y_train.shape[-1] == 1:
            y_train = y_train.squeeze(dim=y_train.dims[-1])


        # 1. Prepare training data for GA core
        X_train_rows, y_train_normalized, predictor_names, train_nan_mask = \
            self._prepare_data_for_ga_core(X_train, y_train)

        if not X_train_rows:
            print("No valid training data after NaN removal. Cannot train model. Returning NaNs for predictions.")
            # Return NaN-filled DataArray
            return xr.DataArray(
                data=np.full(X_test.shape[:-1], np.nan), # Shape (T,Y,X)
                coords={'T': X_test['T'], 'Y': X_test['Y'], 'X': X_test['X']},
                dims=['T','Y','X']
            )

        # 2. Run GA on training data
        trained_best_ensemble = self._run_ga_core(X_train_rows, y_train_normalized, predictor_names)
        
        if not trained_best_ensemble:
            print("GA training failed to find suitable individuals. Returning NaNs for predictions.")
            return xr.DataArray(
                data=np.full(X_test.shape[:-1], np.nan),
                coords={'T': X_test['T'], 'Y': X_test['Y'], 'X': X_test['X']},
                dims=['T','Y','X']
            )

        # 3. Prepare test data for prediction
        # Stack X_test (T, Y, X, M) -> (sample, M)
        X_test_stacked_raw = X_test.stack(sample=('T','Y','X')).transpose('sample','M').values
        
        # Identify NaNs in test predictors
        test_nan_mask = np.any(~np.isfinite(X_test_stacked_raw), axis=1)
        X_test_clean_raw = X_test_stacked_raw[~test_nan_mask]

        # Normalize test data using training ranges, with clipping to [-1,1] to prevent extrapolation
        X_test_normalized_columns = []
        if X_test_clean_raw.size > 0:
            for i, pred_name in enumerate(predictor_names):
                original_min, original_max = self._predictor_normalization_ranges.get(pred_name, (0, 0))
                
                col_data = X_test_clean_raw[:, i]
                if original_max == original_min:
                    normalized_col = np.zeros_like(col_data, dtype=float)
                else:
                    normalized_col = 2 * ((col_data - original_min) / (original_max - original_min)) - 1
                    normalized_col = np.clip(normalized_col, -1, 1)  # Clip to prevent extrapolation
                X_test_normalized_columns.append(normalized_col)
        
        X_test_rows = []
        if X_test_normalized_columns:
            X_test_normalized_array = np.array(X_test_normalized_columns).T
            for row_idx in range(X_test_normalized_array.shape[0]):
                row_dict = {predictor_names[i]: X_test_normalized_array[row_idx, i]
                            for i in range(len(predictor_names))}
                X_test_rows.append(row_dict)

        # 4. Make predictions using the best ensemble (mean of top individuals)
        predicted_values_normalized = []
        if X_test_rows:
            for t_data_row in X_test_rows:
                ensemble_predictions = []
                for individual in trained_best_ensemble:
                    row_prediction = 0
                    for gene in individual.genes:
                        row_prediction += gene.evaluate(t_data_row)
                    ensemble_predictions.append(row_prediction)
                mean_prediction = np.mean(ensemble_predictions)
                predicted_values_normalized.append(mean_prediction)
        else:
            predicted_values_normalized = np.array([])

        predicted_values_normalized = np.array(predicted_values_normalized)

        # 5. Denormalize predictions
        if self._y_normalization_min_max:
            y_min_orig, y_max_orig = self._y_normalization_min_max
            predicted_values_denormalized_clean = self._denormalize_array(
                predicted_values_normalized,
                y_min_orig,
                y_max_orig
            )
        else:
            predicted_values_denormalized_clean = np.full_like(predicted_values_normalized, np.nan)


        # 6. Reshape back to (T, Y, X)
        n_time = X_test['T'].size
        n_lat = X_test['Y'].size
        n_lon = X_test['X'].size
        
        full_predictions_stacked = np.full(X_test_stacked_raw.shape[0], np.nan)
        if len(predicted_values_denormalized_clean) > 0:
            full_predictions_stacked[~test_nan_mask] = predicted_values_denormalized_clean

        predictions_reshaped = full_predictions_stacked.reshape(n_time, n_lat, n_lon)

        predicted_da = xr.DataArray(
            data=predictions_reshaped,
            coords={'T': X_test['T'], 'Y': X_test['Y'], 'X': X_test['X']},
            dims=['T','Y','X']
        )
        return predicted_da

    # ------------------ Probability Calculation Methods ------------------
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
            # Transform thresholds
            first_t = (first_tercile - best_guess) / error_std
            second_t = (second_tercile - best_guess) / error_std

            pred_prob[0, :] = stats.t.cdf(first_t, df=dof)
            pred_prob[1, :] = stats.t.cdf(second_t, df=dof) - stats.t.cdf(first_t, df=dof)
            pred_prob[2, :] = 1 - stats.t.cdf(second_t, df=dof)

        return pred_prob
    @staticmethod
    def calculate_tercile_probabilities_weibull_min(best_guess, error_variance, first_tercile, second_tercile, dof):

        n_time = len(best_guess)
        pred_prob = np.empty((3, n_time))
    
        if np.all(np.isnan(best_guess)):
            pred_prob[:] = np.nan
        else:
            error_std = np.sqrt(error_variance)
    
            # Using the weibull_min CDF with best_guess as loc and error_std as scale.
            pred_prob[0, :] = stats.weibull_min.cdf(first_tercile, c=dof, loc=best_guess, scale=error_std)
            pred_prob[1, :] = stats.weibull_min.cdf(second_tercile, c=dof, loc=best_guess, scale=error_std) - \
                               stats.weibull_min.cdf(first_tercile, c=dof, loc=best_guess, scale=error_std)
            pred_prob[2, :] = 1 - stats.weibull_min.cdf(second_tercile, c=dof, loc=best_guess, scale=error_std)
    
        return pred_prob
        
    @staticmethod
    def calculate_tercile_probabilities_gamma(best_guess, error_variance, T1, T2, dof=None):
        """Gamma-distribution based method."""
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

    @staticmethod
    def calculate_tercile_probabilities_normal(best_guess, error_variance, first_tercile, second_tercile):
        """Normal-distribution based method."""
        n_time = len(best_guess)
        pred_prob = np.empty((3, n_time))
        if np.all(np.isnan(best_guess)):
            pred_prob[:] = np.nan
        else:
            error_std = np.sqrt(error_variance)
            pred_prob[0, :] = norm.cdf(first_tercile, loc=best_guess, scale=error_std)
            pred_prob[1, :] = norm.cdf(second_tercile, loc=best_guess, scale=error_std) - norm.cdf(first_tercile, loc=best_guess, scale=error_std)
            pred_prob[2, :] = 1 - norm.cdf(second_tercile, loc=best_guess, scale=error_std)
        return pred_prob

    @staticmethod
    def calculate_tercile_probabilities_lognormal(best_guess, error_variance, first_tercile, second_tercile):
        """Lognormal-distribution based method."""
        n_time = len(best_guess)
        pred_prob = np.empty((3, n_time))
        if np.any(np.isnan(best_guess)) or np.any(np.isnan(error_variance)):
            pred_prob[:] = np.nan
            return pred_prob
        sigma = np.sqrt(np.log(1 + error_variance / (best_guess**2)))
        mu = np.log(best_guess) - sigma**2 / 2
        pred_prob[0, :] = lognorm.cdf(first_tercile, s=sigma, scale=np.exp(mu))
        pred_prob[1, :] = lognorm.cdf(second_tercile, s=sigma, scale=np.exp(mu)) - lognorm.cdf(first_tercile, s=sigma, scale=np.exp(mu))
        pred_prob[2, :] = 1 - lognorm.cdf(second_tercile, s=sigma, scale=np.exp(mu))
        return pred_prob

    def compute_prob(self, Predictant, clim_year_start, clim_year_end, hindcast_det):
        """
        Compute tercile probabilities for hindcast data.

        Calculates probabilities for below-normal, normal, and above-normal categories using
        the specified distribution method, based on climatological terciles.

        Parameters
        ----------
        Predictant : xarray.DataArray
            Observed predictand data with dimensions (T, Y, X).
        clim_year_start : int or str
            Start year of the climatology period.
        clim_year_end : int or str
            End year of the climatology period.
        hindcast_det : xarray.DataArray
            Deterministic hindcast data with dimensions (T, Y, X).

        Returns
        -------
        hindcast_prob : xarray.DataArray
            Tercile probabilities with dimensions (probability, T, Y, X), where probability
            includes ['PB', 'PN', 'PA'] (below-normal, normal, above-normal).
        """        """
        Compute tercile probabilities using the chosen distribution method.
        Predictant is expected to be an xarray DataArray with dims (T, Y, X).
        """
        
        if "M" in Predictant.coords:
            Predictant = Predictant.isel(M=0).drop_vars('M').squeeze()
        mask = xr.where(~np.isnan(Predictant.isel(T=0)), 1, np.nan).drop_vars(['T']).squeeze().to_numpy()
        
        # Ensure Predictant is (T, Y, X)
        Predictant = Predictant.transpose('T', 'Y', 'X')
        index_start = Predictant.get_index("T").get_loc(str(clim_year_start)).start
        index_end = Predictant.get_index("T").get_loc(str(clim_year_end)).stop
        rainfall_for_tercile = Predictant.isel(T=slice(index_start, index_end))
        terciles = rainfall_for_tercile.quantile([0.33, 0.67], dim='T')
        error_variance = (Predictant - hindcast_det).var(dim='T')
        dof = len(Predictant.get_index("T")) - 2
        
        if self.dist_method == "t":
            calc_func = self.calculate_tercile_probabilities
            hindcast_prob = xr.apply_ufunc(
                calc_func,
                hindcast_det,
                error_variance,
                terciles.isel(quantile=0).drop_vars('quantile'),
                terciles.isel(quantile=1).drop_vars('quantile'),
                input_core_dims=[('T',), (), (), ()],
                vectorize=True,
                kwargs={'dof': dof},
                dask='parallelized',
                output_core_dims=[('probability','T')],
                output_dtypes=['float'],
                dask_gufunc_kwargs={'output_sizes': {'probability': 3}, "allow_rechunk": True}
            )

        elif self.dist_method == "weibull_min":
            calc_func = self.calculate_tercile_probabilities_weibull_min
            hindcast_prob = xr.apply_ufunc(
                calc_func,
                hindcast_det,
                error_variance,
                terciles.isel(quantile=0).drop_vars('quantile'),
                terciles.isel(quantile=1).drop_vars('quantile'),
                input_core_dims=[('T',), (), (), ()],
                vectorize=True,
                kwargs={'dof': dof},
                dask='parallelized',
                output_core_dims=[('probability', 'T')],
                output_dtypes=['float'],
                dask_gufunc_kwargs={'output_sizes': {'probability': 3}, "allow_rechunk": True}
            )
            
            
        elif self.dist_method == "gamma":
            calc_func = self.calculate_tercile_probabilities_gamma
            hindcast_prob = xr.apply_ufunc(
                calc_func,
                hindcast_det,
                error_variance,
                terciles.isel(quantile=0).drop_vars('quantile'),
                terciles.isel(quantile=1).drop_vars('quantile'),
                input_core_dims=[('T',), (), (), ()],
                vectorize=True,
                dask='parallelized',
                output_core_dims=[('probability','T')],
                output_dtypes=['float'],
                dask_gufunc_kwargs={'output_sizes': {'probability': 3}, "allow_rechunk": True}
            )
        elif self.dist_method == "nonparam":
            calc_func = self.calculate_tercile_probabilities_nonparametric
            error_samples = (Predictant - hindcast_det)  
            hindcast_prob = xr.apply_ufunc(
                calc_func,
                hindcast_det,
                error_samples,
                terciles.isel(quantile=0).drop_vars('quantile'),
                terciles.isel(quantile=1).drop_vars('quantile'),
                input_core_dims=[('T',), ('T',), (), ()],
                output_core_dims=[('probability','T')],
                vectorize=True,
                dask='parallelized',
                output_dtypes=[float],
                dask_gufunc_kwargs={'output_sizes': {'probability': 3}, "allow_rechunk": True}
            )
        elif self.dist_method == "normal":
            calc_func = self.calculate_tercile_probabilities_normal
            hindcast_prob = xr.apply_ufunc(
                calc_func,
                hindcast_det,
                error_variance,
                terciles.isel(quantile=0).drop_vars('quantile'),
                terciles.isel(quantile=1).drop_vars('quantile'),
                input_core_dims=[('T',), (), (), ()],
                vectorize=True,
                dask='parallelized',
                output_core_dims=[('probability','T')],
                output_dtypes=['float'],
                dask_gufunc_kwargs={'output_sizes': {'probability': 3}, "allow_rechunk": True}
            )
        elif self.dist_method == "lognormal":
            calc_func = self.calculate_tercile_probabilities_lognormal
            hindcast_prob = xr.apply_ufunc(
                calc_func,
                hindcast_det,
                error_variance,
                terciles.isel(quantile=0).drop_vars('quantile'),
                terciles.isel(quantile=1).drop_vars('quantile'),
                input_core_dims=[('T',), (), (), ()],
                vectorize=True,
                dask='parallelized',
                output_core_dims=[('probability','T')],
                output_dtypes=['float'],
                dask_gufunc_kwargs={'output_sizes': {'probability': 3}, "allow_rechunk": True}
            )
        else:
            raise ValueError(f"Invalid dist_method: {self.dist_method}.")
        
        hindcast_prob = hindcast_prob.assign_coords(probability=('probability', ['PB', 'PN', 'PA']))
        return hindcast_prob.transpose('probability', 'T', 'Y', 'X')


    def forecast(self, Predictant, clim_year_start, clim_year_end, hindcast_det,
                 hindcast_det_cross, Predictor_for_year):
        
        result_da = self.compute_model(hindcast_det, Predictant, Predictor_for_year)

        if "M" in Predictant.coords:
            Predictant = Predictant.isel(M=0).drop_vars('M').squeeze()
        
        year = Predictor_for_year.coords['T'].values.astype('datetime64[Y]').astype(int)[0] + 1970  # Convert from epoch
        T_value_1 = Predictant.isel(T=0).coords['T'].values  # Get the datetime64 value from da1
        month_1 = T_value_1.astype('datetime64[M]').astype(int) % 12 + 1  # Extract month
        new_T_value = np.datetime64(f"{year}-{month_1:02d}-{1:02d}")
        result_da = result_da.assign_coords(T=xr.DataArray([new_T_value], dims=["T"]))
        result_da['T'] = result_da['T'].astype('datetime64[ns]')
        
        # Compute tercile probabilities on predictions
        index_start = Predictant.get_index("T").get_loc(str(clim_year_start)).start
        index_end = Predictant.get_index("T").get_loc(str(clim_year_end)).stop
        rainfall_for_tercile = Predictant.isel(T=slice(index_start, index_end))
        terciles = rainfall_for_tercile.quantile([0.33, 0.67], dim='T')
        error_variance = (Predictant - hindcast_det_cross).var(dim='T')
        dof = len(Predictant.get_index("T")) - 2
        
        if self.dist_method == "t":
            calc_func = self.calculate_tercile_probabilities
            hindcast_prob = xr.apply_ufunc(
                calc_func,
                result_da,
                error_variance,
                terciles.isel(quantile=0).drop_vars('quantile'),
                terciles.isel(quantile=1).drop_vars('quantile'),
                input_core_dims=[('T',), (), (), ()],
                vectorize=True,
                kwargs={'dof': dof},
                dask='parallelized',
                output_core_dims=[('probability','T')],
                output_dtypes=['float'],
                dask_gufunc_kwargs={'output_sizes': {'probability': 3}, "allow_rechunk": True}
            )
        elif self.dist_method == "weibull_min":
            calc_func = self.calculate_tercile_probabilities_weibull_min
            hindcast_prob = xr.apply_ufunc(
                calc_func,
                hindcast_det,
                error_variance,
                terciles.isel(quantile=0).drop_vars('quantile'),
                terciles.isel(quantile=1).drop_vars('quantile'),
                input_core_dims=[('T',), (), (), ()],
                vectorize=True,
                kwargs={'dof': dof},
                dask='parallelized',
                output_core_dims=[('probability', 'T')],
                output_dtypes=['float'],
                dask_gufunc_kwargs={'output_sizes': {'probability': 3}, "allow_rechunk": True}
            )
            
            
        elif self.dist_method == "gamma":
            calc_func = self.calculate_tercile_probabilities_gamma
            hindcast_prob = xr.apply_ufunc(
                calc_func,
                result_da,
                error_variance,
                terciles.isel(quantile=0).drop_vars('quantile'),
                terciles.isel(quantile=1).drop_vars('quantile'),
                input_core_dims=[('T',), (), (), ()],
                vectorize=True,
                dask='parallelized',
                output_core_dims=[('probability','T')],
                output_dtypes=['float'],
                dask_gufunc_kwargs={'output_sizes': {'probability': 3}, "allow_rechunk": True}
            )
        elif self.dist_method == "normal":
            calc_func = self.calculate_tercile_probabilities_normal
            hindcast_prob = xr.apply_ufunc(
                calc_func,
                result_da,
                error_variance,
                terciles.isel(quantile=0).drop_vars('quantile'),
                terciles.isel(quantile=1).drop_vars('quantile'),
                input_core_dims=[('T',), (), (), ()],
                vectorize=True,
                dask='parallelized',
                output_core_dims=[('probability','T')],
                output_dtypes=['float'],
                dask_gufunc_kwargs={'output_sizes': {'probability': 3}, "allow_rechunk": True}
            )
        elif self.dist_method == "lognormal":
            calc_func = self.calculate_tercile_probabilities_lognormal
            hindcast_prob = xr.apply_ufunc(
                calc_func,
                result_da,
                error_variance,
                terciles.isel(quantile=0).drop_vars('quantile'),
                terciles.isel(quantile=1).drop_vars('quantile'),
                input_core_dims=[('T',), (), (), ()],
                vectorize=True,
                dask='parallelized',
                output_core_dims=[('probability','T')],
                output_dtypes=['float'],
                dask_gufunc_kwargs={'output_sizes': {'probability': 3}, "allow_rechunk": True}
            )
        elif self.dist_method == "nonparam":
            calc_func = self.calculate_tercile_probabilities_nonparametric
            error_samples = Predictant - hindcast_det_cross
            error_samples = error_samples.rename({'T':'S'})
            hindcast_prob = xr.apply_ufunc(
                calc_func,
                result_da,
                error_samples,
                terciles.isel(quantile=0).drop_vars('quantile'),
                terciles.isel(quantile=1).drop_vars('quantile'),
                input_core_dims=[('T',), ('S',), (), ()],
                output_core_dims=[('probability','T')],
                vectorize=True,
                dask='parallelized',
                output_dtypes=[float],
                dask_gufunc_kwargs={'output_sizes': {'probability': 3}, "allow_rechunk": True}
            )
        else:
            raise ValueError(f"Invalid dist_method: {self.dist_method}.")
        
        hindcast_prob = hindcast_prob.assign_coords(probability=('probability', ['PB', 'PN', 'PA']))
        return result_da, hindcast_prob.transpose('probability', 'T','Y', 'X')

###################################### NonHomogenous Regression Models ################################################


class NGR_:
    """
    Nonhomogeneous Gaussian Regression (NGR / EMOS).

    y_t | X_t ~ N(mu_t, sigma_t^2)
    mu_t = a + b * xbar_t  (exchangeable)   or   a + sum_j b_j x_{t,j} (non-exch.)
    sigma_t^2 = c + d * s_t^2,  c>0, d>0 via softplus links.

    s2_mode:
      - 'population' : Eq. (3.4)   s_t^2 = (1/m) * sum_k (x_{t,k} - xbar_t)^2
      - 'sample'     : ddof=1      s_t^2 = sum_k (x_{t,k} - xbar_t)^2 / (m-1)
    """

    def __init__(
        self,
        exchangeable: bool = True,
        estimation_method: str = "log_likelihood",  # or "crps"
        s2_mode: str = "population",
        l2_penalty: float = 0.0,                    # ridge on mean coeffs b (not on a, c, d)
        min_sigma: float = 1e-3,                    # raised floor to stabilize optimizer
        c_cap: float = 1e4,                         # tighter caps to avoid runaway variance
        d_cap: float = 1e4,
        raw_bound: float = 8.0,                     # narrower bounds for gamma_raw, delta_raw
        random_state: Optional[int] = None,
    ):
        self.exchangeable = bool(exchangeable)
        self.estimation_method = str(estimation_method).lower()
        if self.estimation_method not in {"log_likelihood", "crps"}:
            raise ValueError("estimation_method must be 'log_likelihood' or 'crps'")

        s2_mode = str(s2_mode).lower().strip()
        if s2_mode not in {"population", "sample"}:
            raise ValueError("s2_mode must be 'population' or 'sample'")
        self.s2_mode = s2_mode

        self.l2_penalty = float(l2_penalty)
        self.min_sigma = float(min_sigma)
        self.c_cap = float(c_cap)
        self.d_cap = float(d_cap)
        self.raw_bound = float(raw_bound)
        self.random_state = random_state

        self.params: Optional[np.ndarray] = None   # [a, b, gamma_raw, delta_raw] or [a, b1..bm, gamma_raw, delta_raw]
        self.m: Optional[int] = None               # number of ensemble members

    # ------------------------ stable link functions ------------------------
    @staticmethod
    def _softplus(x):
        """c = softplus(gamma_raw) = log(1+exp(x)) computed stably."""
        return np.logaddexp(0.0, x)

    @staticmethod
    def _inv_softplus(y):
        """Inverse of softplus for y>0: x = log(exp(y) - 1) with a stable form."""
        y = np.asarray(y, dtype=float)
        return y + np.log(-np.expm1(-y))

    # ------------------------ spread (s_t^2) helper ------------------------
    def _ensemble_variance(self, X: np.ndarray) -> np.ndarray:
        """
        Return s_t^2 for each row of X, with tiny jitter to avoid zero spread.
        """
        if self.m is None or self.m <= 1:
            return np.zeros(X.shape[0], dtype=float)
        ddof = 0 if self.s2_mode == "population" else 1
        s2 = np.var(X, axis=1, ddof=ddof)
        return s2 + 1e-12  # jitter to prevent singularities in the optimizer

    # ------------------------ mean & std builders ------------------------
    def _compute_mu_t(self, X: np.ndarray, params: np.ndarray) -> np.ndarray:
        if self.exchangeable:
            a, b = params[0], params[1]
            xbar = np.mean(X, axis=1)
            return a + b * xbar
        a = params[0]
        b = params[1:1 + self.m]
        return a + X @ b

    def _compute_sigma_t(self, X: np.ndarray, params: np.ndarray) -> np.ndarray:
        st2 = self._ensemble_variance(X)
        gamma_raw, delta_raw = params[-2], params[-1]
        c = float(np.clip(self._softplus(gamma_raw), 1e-12, self.c_cap))
        d = float(np.clip(self._softplus(delta_raw), 1e-12, self.d_cap))
        sigma2 = np.maximum(c + d * st2, self.min_sigma**2)
        return np.sqrt(sigma2)

    # ------------------------ objectives ------------------------
    def _neg_loglik(self, params: np.ndarray, X: np.ndarray, y: np.ndarray) -> float:
        mu = self._compute_mu_t(X, params)
        sigma = self._compute_sigma_t(X, params)
        if not (np.all(np.isfinite(mu)) and np.all(np.isfinite(sigma))):
            return 1e30
        z2 = ((y - mu) / sigma) ** 2
        nll = 0.5 * np.sum(z2 + 2.0 * np.log(sigma) + np.log(2.0 * np.pi))  # Eq. (3.10)
        if self.l2_penalty > 0:
            if self.exchangeable:
                b = params[1]
                nll += self.l2_penalty * (b ** 2)
            else:
                b = params[1:1 + self.m]
                nll += self.l2_penalty * float(np.sum(b ** 2))
        return float(nll) if np.isfinite(nll) else 1e30

    def _crps(self, params: np.ndarray, X: np.ndarray, y: np.ndarray) -> float:
        mu = self._compute_mu_t(X, params)
        sigma = self._compute_sigma_t(X, params)
        if not (np.all(np.isfinite(mu)) and np.all(np.isfinite(sigma))):
            return 1e30
        z = (y - mu) / sigma
        crps = sigma * (z * (2.0 * norm.cdf(z) - 1.0) + 2.0 * norm.pdf(z) - 1.0 / np.sqrt(np.pi))  # Eq. (3.9)
        loss = float(np.mean(crps))
        if self.l2_penalty > 0:
            if self.exchangeable:
                b = params[1]
                loss += self.l2_penalty * (b ** 2)
            else:
                b = params[1:1 + self.m]
                loss += self.l2_penalty * float(np.sum(b ** 2))
        return loss if np.isfinite(loss) else 1e30

    # ------------------------ fit ------------------------
    def fit(self, X_train: np.ndarray, y_train: np.ndarray, fast: bool = False) -> None:
        """
        Fit the NGR model.

        fast=True → skip iterative optimization and use OLS for mean + OLS(resid^2 ~ s_t^2)
        """
        X = np.asarray(X_train, dtype=float)
        y = np.asarray(y_train, dtype=float).reshape(-1)
        if X.ndim != 2 or y.ndim != 1 or X.shape[0] != y.shape[0]:
            raise ValueError("Shapes must be X:(n,m), y:(n,) with matching n")
        n, m = X.shape
        self.m = m

        # --- OLS init for mean ---
        if self.exchangeable:
            xbar = np.mean(X, axis=1)
            A = np.column_stack([np.ones(n), xbar])  # [1, xbar]
        else:
            A = np.column_stack([np.ones(n), X])     # [1, X]
        try:
            beta, *_ = np.linalg.lstsq(A, y, rcond=None)
        except np.linalg.LinAlgError:
            beta = np.zeros(A.shape[1])
        mu_hat = A @ beta

        # --- OLS init for variance (resid^2 ~ c + d * s_t^2) ---
        st2 = self._ensemble_variance(X)
        resid2 = (y - mu_hat) ** 2
        B = np.column_stack([np.ones(n), st2])
        try:
            theta, *_ = np.linalg.lstsq(B, resid2, rcond=None)
            c0 = max(float(theta[0]), 1e-6)
            d0 = max(float(theta[1]), 1e-6)
        except np.linalg.LinAlgError:
            c0, d0 = 1.0, 0.5

        gamma_raw0 = float(self._inv_softplus(c0))
        delta_raw0 = float(self._inv_softplus(d0))

        if self.exchangeable:
            initial = np.array([beta[0], beta[1], gamma_raw0, delta_raw0], dtype=float)
        else:
            initial = np.concatenate([beta, [gamma_raw0, delta_raw0]]).astype(float)

        # Fast path: store OLS-based params (already good in many cases)
        if fast:
            self.params = initial
            return

        # Objective and bounds
        objective = self._neg_loglik if self.estimation_method == "log_likelihood" else self._crps
        k = initial.size
        lb = [-np.inf] * k
        ub = [ np.inf] * k
        lb[-2] = -self.raw_bound  # gamma_raw
        ub[-2] =  self.raw_bound
        lb[-1] = -self.raw_bound  # delta_raw
        ub[-1] =  self.raw_bound
        bounds = list(zip(lb, ub))

        # --- Multi-start seeds for (c,d) to avoid bad basins ---
        yvar = float(np.var(y)) if np.isfinite(np.var(y)) else 1.0
        c_cands = [0.25 * yvar + 1e-6, 1.0 * yvar + 1e-6, 4.0 * yvar + 1e-6]
        d_cands = [1e-3, 0.1, 1.0]

        cand_thetas = [initial]
        for c_init in c_cands:
            for d_init in d_cands:
                g0 = float(self._inv_softplus(c_init))
                d0r = float(self._inv_softplus(d_init))
                if self.exchangeable:
                    th = np.array([beta[0], beta[1], g0, d0r], float)
                else:
                    th = np.concatenate([beta, [g0, d0r]]).astype(float)
                cand_thetas.append(th)

        best_val = np.inf
        best_theta = initial

        # Short LBFGS passes to pick a good start
        for th0 in cand_thetas:
            trial = minimize(
                objective, th0, args=(X, y),
                method="L-BFGS-B", bounds=bounds,
                options={"maxiter": 150, "ftol": 1e-6, "maxls": 30}
            )
            if np.isfinite(trial.fun) and trial.fun < best_val:
                best_val, best_theta = trial.fun, trial.x

        # Main LBFGS with stricter settings
        res = minimize(
            objective, best_theta, args=(X, y),
            method="L-BFGS-B", bounds=bounds,
            options={"maxiter": 300, "ftol": 1e-8, "maxls": 40}
        )

        # Powell warmup then LBFGS if still not converged
        if not res.success:
            warm = minimize(
                objective, best_theta, args=(X, y),
                method="Powell",
                options={"maxiter": 80, "ftol": 1e-6}
            )
            res = minimize(
                objective, warm.x, args=(X, y),
                method="L-BFGS-B", bounds=bounds,
                options={"maxiter": 300, "ftol": 1e-8, "maxls": 40}
            )
            if not res.success:
                print("Warning: NGR optimization did not converge:", res.message)

        self.params = res.x

    # ------------------------ inference ------------------------
    def predict(self, X_test: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        if self.params is None:
            raise ValueError("Model must be fitted before prediction")
        X = np.asarray(X_test, dtype=float)
        if X.ndim != 2 or X.shape[1] != self.m:
            raise ValueError(f"X_test must be (n_test, {self.m})")
        mu = self._compute_mu_t(X, self.params)
        sigma = self._compute_sigma_t(X, self.params)
        return mu, sigma

    def prob_less_than(self, X_test: np.ndarray, q: float | np.ndarray) -> np.ndarray:
        mu, sigma = self.predict(X_test)
        return norm.cdf(q, loc=mu, scale=sigma)



# -----------------------------------------------------------------------------
# Nonhomogeneous Gaussian Regression (NGR)
# -----------------------------------------------------------------------------

# ------------------------------ Core NGR (vector) ------------------------------ #
class NGR:
    """
    Nonhomogeneous Gaussian Regression (NGR / EMOS).

    y_t | X_t ~ N(mu_t, sigma_t^2)
    mu_t = a + b * xbar_t  (exchangeable)   or   a + sum_j b_j x_{t,j} (non-exch.)
    sigma_t^2 = c + d * s_t^2,  with c,d>0 via softplus links.

    s2_mode: 'population' -> Eq. (3.4) 1/m * sum (x - xbar)^2  ;  'sample' -> ddof=1
    """

    def __init__(
        self,
        exchangeable: bool = True,
        estimation_method: str = "log_likelihood",  # or "crps"
        s2_mode: str = "population",                # literature default (Eq. 3.4)
        l2_penalty: float = 0.0,                    # ridge on mean coeffs b
        min_sigma: float = 1e-4,
        c_cap: float = 1e6,
        d_cap: float = 1e6,
        raw_bound: float = 20.0,                    # bounds for gamma_raw, delta_raw
    ):
        self.exchangeable = bool(exchangeable)
        self.estimation_method = estimation_method.lower()
        if self.estimation_method not in {"log_likelihood", "crps"}:
            raise ValueError("estimation_method must be 'log_likelihood' or 'crps'")
        s2_mode = s2_mode.lower().strip()
        if s2_mode not in {"population", "sample"}:
            raise ValueError("s2_mode must be 'population' or 'sample'")
        self.s2_mode = s2_mode
        self.l2_penalty = float(l2_penalty)
        self.min_sigma = float(min_sigma)
        self.c_cap = float(c_cap)
        self.d_cap = float(d_cap)
        self.raw_bound = float(raw_bound)

        self.params: Optional[np.ndarray] = None   # [a, b, gamma_raw, delta_raw] or [a, b1..bm, gamma_raw, delta_raw]
        self.m: Optional[int] = None               # number of members

    # ---- stable links ----
    @staticmethod
    def _softplus(x):
        return np.logaddexp(0.0, x)

    @staticmethod
    def _inv_softplus(y):
        y = np.asarray(y, dtype=float)
        return y + np.log(-np.expm1(-y))  # stable log(exp(y)-1)

    # ---- s_t^2 ----
    def _ensemble_variance(self, X: np.ndarray) -> np.ndarray:
        if self.m is None or self.m <= 1:
            return np.zeros(X.shape[0], dtype=float)
        ddof = 0 if self.s2_mode == "population" else 1
        return np.var(X, axis=1, ddof=ddof)

    # ---- mu_t, sigma_t ----
    def _compute_mu_t(self, X: np.ndarray, params: np.ndarray) -> np.ndarray:
        if self.exchangeable:
            a, b = params[0], params[1]
            xbar = np.mean(X, axis=1)
            return a + b * xbar
        a = params[0]
        b = params[1:1 + self.m]
        return a + X @ b

    def _compute_sigma_t(self, X: np.ndarray, params: np.ndarray) -> np.ndarray:
        st2 = self._ensemble_variance(X)
        gamma_raw, delta_raw = params[-2], params[-1]
        c = float(np.clip(self._softplus(gamma_raw), 1e-12, self.c_cap))
        d = float(np.clip(self._softplus(delta_raw), 1e-12, self.d_cap))
        sigma2 = np.maximum(c + d * st2, self.min_sigma**2)
        return np.sqrt(sigma2)

    # ---- objectives ----
    def _neg_loglik(self, params: np.ndarray, X: np.ndarray, y: np.ndarray) -> float:
        mu = self._compute_mu_t(X, params)
        sigma = self._compute_sigma_t(X, params)
        if not (np.all(np.isfinite(mu)) and np.all(np.isfinite(sigma))):
            return 1e30
        z2 = ((y - mu) / sigma) ** 2
        nll = 0.5 * np.sum(z2 + 2.0 * np.log(sigma) + np.log(2.0 * np.pi))  # Eq. (3.10)
        if self.l2_penalty > 0:
            if self.exchangeable:
                b = params[1]
                nll += self.l2_penalty * (b ** 2)
            else:
                b = params[1:1 + self.m]
                nll += self.l2_penalty * float(np.sum(b ** 2))
        return float(nll) if np.isfinite(nll) else 1e30

    def _crps(self, params: np.ndarray, X: np.ndarray, y: np.ndarray) -> float:
        mu = self._compute_mu_t(X, params)
        sigma = self._compute_sigma_t(X, params)
        if not (np.all(np.isfinite(mu)) and np.all(np.isfinite(sigma))):
            return 1e30
        z = (y - mu) / sigma
        crps = sigma * (z * (2.0 * norm.cdf(z) - 1.0) + 2.0 * norm.pdf(z) - 1.0 / np.sqrt(np.pi))  # Eq. (3.9)
        loss = float(np.mean(crps))
        if self.l2_penalty > 0:
            if self.exchangeable:
                b = params[1]
                loss += self.l2_penalty * (b ** 2)
            else:
                b = params[1:1 + self.m]
                loss += self.l2_penalty * float(np.sum(b ** 2))
        return loss if np.isfinite(loss) else 1e30

    # ---- fit ----
    def fit(self, X_train: np.ndarray, y_train: np.ndarray) -> None:
        X = np.asarray(X_train, dtype=float)
        y = np.asarray(y_train, dtype=float).reshape(-1)
        if X.ndim != 2 or y.ndim != 1 or X.shape[0] != y.shape[0]:
            raise ValueError("Shapes must be X:(n,m), y:(n,) with matching n")
        n, m = X.shape
        self.m = m

        # OLS for mean: [1, xbar] or [1, X]
        if self.exchangeable:
            xbar = np.mean(X, axis=1)
            A = np.column_stack([np.ones(n), xbar])
        else:
            A = np.column_stack([np.ones(n), X])
        try:
            beta, *_ = np.linalg.lstsq(A, y, rcond=None)
        except np.linalg.LinAlgError:
            beta = np.zeros(A.shape[1])
        mu_hat = A @ beta

        # Regress residual^2 on s_t^2 for c,d init
        st2 = self._ensemble_variance(X)
        resid2 = (y - mu_hat) ** 2
        B = np.column_stack([np.ones(n), st2])
        try:
            theta, *_ = np.linalg.lstsq(B, resid2, rcond=None)
            c0 = max(float(theta[0]), 1e-6)
            d0 = max(float(theta[1]), 1e-6)
        except np.linalg.LinAlgError:
            c0, d0 = 1.0, 0.5

        gamma_raw0 = float(self._inv_softplus(c0))
        delta_raw0 = float(self._inv_softplus(d0))

        if self.exchangeable:
            initial = np.array([beta[0], beta[1], gamma_raw0, delta_raw0], dtype=float)
        else:
            initial = np.concatenate([beta, [gamma_raw0, delta_raw0]]).astype(float)

        # optimize
        objective = self._neg_loglik if self.estimation_method == "log_likelihood" else self._crps
        k = initial.size
        lb = [-np.inf] * k
        ub = [ np.inf] * k
        lb[-2] = -self.raw_bound
        ub[-2] =  self.raw_bound
        lb[-1] = -self.raw_bound
        ub[-1] =  self.raw_bound
        bounds = list(zip(lb, ub))

        res = minimize(
            objective, initial, args=(X, y),
            method="L-BFGS-B", bounds=bounds,
            options={"maxiter": 1000, "ftol": 1e-9}
        )
        if not res.success:
            print("Warning: NGR optimization did not converge:", res.message)
        self.params = res.x

    # ---- inference ----
    def predict(self, X_test: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        if self.params is None:
            raise ValueError("Model must be fitted before prediction")
        X = np.asarray(X_test, dtype=float)
        if X.ndim != 2 or X.shape[1] != self.m:
            raise ValueError(f"X_test must be (n_test,{self.m})")
        mu = self._compute_mu_t(X, self.params)
        sigma = self._compute_sigma_t(X, self.params)
        return mu, sigma

    def prob_less_than(self, X_test: np.ndarray, q: float | np.ndarray) -> np.ndarray:
        mu, sigma = self.predict(X_test)
        return norm.cdf(q, loc=mu, scale=sigma)


# --------- Gridded wrapper (xarray + Dask), terciles + forecasting helpers --------- #
class WAS_mme_NGR_Model:
    """
    Grid-wise NGR over (T, M, Y, X) using xarray.apply_ufunc with Dask.

    - Fits NGR per grid cell on (X_train, y_train)
    - Predicts tercile probs for X_test using thresholds (T1,T2)
    """

    def __init__(self, exchangeable: bool = True, estimation_method: str = 'log_likelihood',
                 nb_cores: int = 1, s2_mode: str = "population", l2_penalty: float = 0.0,
                 min_sigma: float = 1e-4, c_cap: float = 1e6, d_cap: float = 1e6, raw_bound: float = 20.0):
        self.exchangeable = exchangeable
        self.estimation_method = estimation_method
        self.nb_cores = int(nb_cores)
        self.s2_mode = s2_mode
        self.l2_penalty = float(l2_penalty)
        self.min_sigma = float(min_sigma)
        self.c_cap = float(c_cap)
        self.d_cap = float(d_cap)
        self.raw_bound = float(raw_bound)

    # vectorized kernel called by apply_ufunc (single grid cell)
    def fit_predict(self, X: np.ndarray, y: np.ndarray, X_test: np.ndarray, t33: float, t67: float) -> np.ndarray:
        if X_test.ndim == 1:
            X_test = X_test.reshape(1, -1)
        n_test = X_test.shape[0]

        mask = np.isfinite(y) & np.all(np.isfinite(X), axis=-1)
        if not np.any(mask) or not np.isfinite(t33) or not np.isfinite(t67):
            return np.full((3, n_test), np.nan, dtype=float)

        y_c = y[mask]
        X_c = X[mask, :]
        if X_c.shape[0] < max(5, 2 + (0 if self.exchangeable else min(3, X_c.shape[1]))):
            return np.full((3, n_test), np.nan, dtype=float)

        model = NGR(
            exchangeable=self.exchangeable,
            estimation_method=self.estimation_method,
            s2_mode=self.s2_mode,
            l2_penalty=self.l2_penalty,
            min_sigma=self.min_sigma,
            c_cap=self.c_cap,
            d_cap=self.d_cap,
            raw_bound=self.raw_bound,
        )
        model.fit(X_c, y_c)

        p_less_33 = model.prob_less_than(X_test, t33)
        p_less_67 = model.prob_less_than(X_test, t67)
        pB = p_less_33
        pA = 1.0 - p_less_67
        pN = 1.0 - pB - pA
        out = np.stack([np.clip(pB, 0, 1), np.clip(pN, 0, 1), np.clip(pA, 0, 1)], axis=0)
        return out

    def compute_model(self, X_train: xr.DataArray, y_train: xr.DataArray, X_test: xr.DataArray,
                      Predictant: xr.DataArray | None = None,
                      clim_year_start: str | None = None, clim_year_end: str | None = None) -> xr.DataArray:
        """
        X_train: (T, M, Y, X)
        y_train: (T, Y, X)
        X_test : (forecast, M, Y, X)  (rename if needed)
        Returns: probs (probability=3, forecast, Y, X) with labels PB, PN, PA
        """
        # terciles (q33,q67) per grid
        if Predictant is not None and (clim_year_start is not None) and (clim_year_end is not None):
            idx = Predictant.get_index("T")
            sl = idx.slice_indexer(str(clim_year_start), str(clim_year_end))
            hist = Predictant.isel(T=sl)
            qs = hist.quantile([0.33, 0.67], dim='T', skipna=True)
        else:
            qs = y_train.quantile([0.33, 0.67], dim='T', skipna=True)
        T1 = qs.isel(quantile=0).drop_vars('quantile')
        T2 = qs.isel(quantile=1).drop_vars('quantile')

        # chunking
        ny = int(y_train.sizes['Y'])
        nx = int(y_train.sizes['X'])
        chunksize_y = max(1, int(np.ceil(ny / max(1, self.nb_cores))))
        chunksize_x = max(1, int(np.ceil(nx / max(1, self.nb_cores))))

        # align
        X_train = X_train.transpose('T', 'M', 'Y', 'X')
        y_train = y_train.transpose('T', 'Y', 'X')

        if 'T' in X_test.dims:
            X_test = X_test.rename({'T': 'forecast'})
        else:
            X_test = X_test.transpose('forecast', 'M', 'Y', 'X')

        client = Client(n_workers=self.nb_cores, threads_per_worker=1)

        result = xr.apply_ufunc(
            self.fit_predict,
            X_train.chunk({'Y': chunksize_y, 'X': chunksize_x}),
            y_train.chunk({'Y': chunksize_y, 'X': chunksize_x}),
            X_test.chunk({'Y': chunksize_y, 'X': chunksize_x}),
            T1.chunk({'Y': chunksize_y, 'X': chunksize_x}),
            T2.chunk({'Y': chunksize_y, 'X': chunksize_x}),
            input_core_dims=[('T', 'M'), ('T',), ('forecast', 'M'), (), ()],
            output_core_dims=[('probability', 'forecast')],
            vectorize=True,
            dask='parallelized',
            output_dtypes=[float],
            dask_gufunc_kwargs={'output_sizes': {'probability': 3}},
        ).compute()

        client.close()

        result = result.rename({'forecast': 'T'})
        result = result.assign_coords(probability=('probability', ['PB', 'PN', 'PA']))
        return result.transpose('probability', 'T', 'Y', 'X')

    def forecast(self, Predictant: xr.DataArray, clim_year_start: str, clim_year_end: str,
                 Predictor: xr.DataArray, Predictor_for_year: xr.DataArray) -> xr.DataArray:
        """
        One-year forecast:
        Predictant: (T, Y, X) observed
        Predictor: (T, M, Y, X) training ensemble
        Predictor_for_year: (forecast=1, M, Y, X) or (T=1, M, Y, X)
        """
        if "M" in Predictant.coords:  # safety if predictant carries M
            Predictant = Predictant.isel(M=0).drop_vars('M').squeeze()

        idx = Predictant.get_index("T")
        sl = idx.slice_indexer(str(clim_year_start), str(clim_year_end))
        hist = Predictant.isel(T=sl)
        qs = hist.quantile([0.33, 0.67], dim='T', skipna=True)
        T1 = qs.isel(quantile=0).drop_vars('quantile')
        T2 = qs.isel(quantile=1).drop_vars('quantile')

        ny = int(Predictant.sizes['Y'])
        nx = int(Predictant.sizes['X'])
        chunksize_y = max(1, int(np.ceil(ny / max(1, self.nb_cores))))
        chunksize_x = max(1, int(np.ceil(nx / max(1, self.nb_cores))))

        Predictor = Predictor.transpose('T', 'M', 'Y', 'X')
        Predictant = Predictant.transpose('T', 'Y', 'X')

        if 'T' in Predictor_for_year.dims:
            Xfy = Predictor_for_year.rename({'T': 'forecast'})
        else:
            Xfy = Predictor_for_year.transpose('forecast', 'M', 'Y', 'X')

        client = Client(n_workers=self.nb_cores, threads_per_worker=1)

        result = xr.apply_ufunc(
            self.fit_predict,
            Predictor.chunk({'Y': chunksize_y, 'X': chunksize_x}),
            Predictant.chunk({'Y': chunksize_y, 'X': chunksize_x}),
            Xfy.chunk({'Y': chunksize_y, 'X': chunksize_x}),
            T1.chunk({'Y': chunksize_y, 'X': chunksize_x}),
            T2.chunk({'Y': chunksize_y, 'X': chunksize_x}),
            input_core_dims=[('T', 'M'), ('T',), ('forecast', 'M'), (), ()],
            output_core_dims=[('probability', 'forecast')],
            vectorize=True,
            dask='parallelized',
            output_dtypes=[float],
            dask_gufunc_kwargs={'output_sizes': {'probability': 3}},
        ).compute()

        client.close()

        # Put back a forecast 'T' label based on first month of Predictant's calendar
        result = result.rename({'forecast': 'T'})
        result = result.assign_coords(probability=('probability', ['PB', 'PN', 'PA']))

        # If Predictor_for_year had a real date, keep it; else align to first month in Predictant
        if 'T' in Predictor_for_year.coords:
            new_T = Predictor_for_year['T'].values
        else:
            t0 = Predictant['T'].values[0]
            # keep the month, replace year with the forecast's year if available in attrs
            new_T = np.array([t0], dtype='datetime64[ns]')

        result = result.assign_coords(T=xr.DataArray(new_T, dims=["T"]))
        return result.transpose('probability', 'T', 'Y', 'X')



############### Flexible  Nonhomogeneous Regression Model for WAS-MME ################

class FlexibleNGR:
    """
    Flexible Nonhomogeneous Regression (NGR) class supporting multiple predictive distributions.

    This class extends standard NGR to handle non-Gaussian predictands by allowing different
    predictive distributions such as Gaussian, lognormal, logistic, Generalized Extreme Value (GEV),
    gamma, weibull, laplace (biexponential), and pareto.
    It uses ensemble statistics (mean and variance) to parameterize the predictive distribution,
    with parameters estimated via maximum likelihood.

    Attributes:
        distribution (str): The type of predictive distribution ('gaussian', 'lognormal', 'logistic', 'gev',
                            'gamma', 'weibull', 'laplace', 'pareto').
        params (np.ndarray): Fitted parameters specific to the chosen distribution.
    """

    def __init__(self, distribution='gaussian'):
        """
        Initialize the FlexibleNGR model.

        Args:
            distribution (str): The predictive distribution to use.
                                Options: 'gaussian', 'lognormal', 'logistic', 'gev', 'gamma', 'weibull',
                                         'laplace', 'pareto'.
                                Default is 'gaussian' (standard NGR).

        Raises:
            ValueError: If an unsupported distribution is specified.
        """
        self.distribution = distribution.lower()
        supported = ['gaussian', 'lognormal', 'logistic', 'gev', 'gamma', 'weibull', 'laplace', 'pareto']
        if self.distribution not in supported:
            raise ValueError(f"Supported distributions are: {', '.join(supported)}")
        self.params = None

    def _compute_ensemble_stats(self, X):
        """
        Compute ensemble mean and variance for each sample.

        Args:
            X (np.ndarray): Ensemble data of shape (n_samples, n_members).

        Returns:
            tuple: (ensemble_mean, ensemble_var), each of shape (n_samples,).
        """
        ensemble_mean = np.mean(X, axis=1)
        ensemble_var = np.var(X, axis=1, ddof=1)
        return ensemble_mean, ensemble_var

    def _log_likelihood_gaussian(self, params, ensemble_mean, ensemble_var, y):
        a, b, gamma, delta = params
        mu_t = a + b * ensemble_mean
        sigma_t2 = gamma**2 + delta**2 * ensemble_var  # Ensure positivity
        sigma_t = np.sqrt(np.maximum(sigma_t2, 1e-8))  # Avoid division by zero
        ll = -0.5 * (y - mu_t)**2 / sigma_t2 - np.log(sigma_t)
        return -np.sum(ll)

    def _log_likelihood_lognormal(self, params, ensemble_mean, ensemble_var, y):
        if np.any(y <= 0):
            return np.inf  # Invalid for lognormal
        ln_y = np.log(y)
        return self._log_likelihood_gaussian(params, ensemble_mean, ensemble_var, ln_y)

    def _log_likelihood_logistic(self, params, ensemble_mean, ensemble_std, y):
        a, b, c, d = params
        mu_t = a + b * ensemble_mean
        sigma_t = np.exp(c + d * ensemble_std)  # Scale parameter, ensure positivity
        z = (y - mu_t) / sigma_t
        ll = -z - np.log(sigma_t) - 2 * np.log(1 + np.exp(-z))
        return -np.sum(ll)

    def _log_likelihood_gev(self, params, ensemble_mean, y):
        a, b, c, d, xi = params
        mu_t = a + b * ensemble_mean
        sigma_t = np.exp(c + d * ensemble_mean)  # Scale parameter
        z = 1 + xi * (y - mu_t) / sigma_t
        if np.any(z <= 0):
            return np.inf  # Invalid for GEV support
        if xi != 0:
            ll = -np.log(sigma_t) + (-1/xi - 1) * np.log(z) - z**(-1/xi)
        else:  # Gumbel case
            u = (y - mu_t) / sigma_t
            ll = -np.log(sigma_t) - u - np.exp(-u)
        return -np.sum(ll)

    def _log_likelihood_gamma(self, params, ensemble_mean, y):
        a, b, gamma, delta = params
        mu_t = np.maximum(a + b * ensemble_mean, 1e-8)
        sigma2_t = np.maximum(gamma**2 + delta**2 * ensemble_mean, 1e-8)
        alpha_t = mu_t**2 / sigma2_t
        beta_t = sigma2_t / mu_t
        ll = scigamma.logpdf(y, a=alpha_t, scale=beta_t)
        return -np.sum(ll)

    def _log_likelihood_weibull(self, params, ensemble_mean, ensemble_std, y):
        a, b, c, d = params
        scale_t = np.exp(a + b * ensemble_mean)
        shape_t = np.exp(c + d * np.log(ensemble_std + 1e-8))
        ll = weibull_min.logpdf(y, c=shape_t, scale=scale_t)
        return -np.sum(ll)

    def _log_likelihood_laplace(self, params, ensemble_mean, ensemble_std, y):
        a, b, c, d = params
        loc_t = a + b * ensemble_mean
        scale_t = np.exp(c + d * ensemble_std)
        ll = laplace.logpdf(y, loc=loc_t, scale=scale_t)
        return -np.sum(ll)

    def _log_likelihood_pareto(self, params, ensemble_mean, ensemble_std, y):
        a, b, c, d = params
        scale_t = np.exp(a + b * ensemble_mean)
        shape_t = np.exp(c + d * np.log(ensemble_std + 1e-8)) + 1  # Ensure shape > 1 for finite mean
        ll = pareto.logpdf(y, b=shape_t, scale=scale_t)
        return -np.sum(ll)

    def fit(self, X_train, y_train):
        if X_train.shape[0] != y_train.shape[0]:
            raise ValueError("Number of samples in X_train and y_train must match")

        ensemble_mean, ensemble_var = self._compute_ensemble_stats(X_train)
        ensemble_std = np.sqrt(ensemble_var)

        if self.distribution == 'gaussian':
            initial_params = [0.0, 1.0, 1.0, 1.0]
            objective = lambda p: self._log_likelihood_gaussian(p, ensemble_mean, ensemble_var, y_train)
        elif self.distribution == 'lognormal':
            initial_params = [0.0, 1.0, 1.0, 1.0]
            objective = lambda p: self._log_likelihood_lognormal(p, ensemble_mean, ensemble_var, y_train)
        elif self.distribution == 'logistic':
            initial_params = [0.0, 1.0, 0.0, 0.0]
            objective = lambda p: self._log_likelihood_logistic(p, ensemble_mean, ensemble_std, y_train)
        elif self.distribution == 'gev':
            initial_params = [0.0, 1.0, 0.0, 0.0, 0.1]
            bounds = [(-np.inf, np.inf)] * 4 + [(-0.5, 0.5)]
            result = minimize(self._log_likelihood_gev, initial_params, args=(ensemble_mean, y_train),
                              method='L-BFGS-B', bounds=bounds)
            if not result.success:
                print(f"Warning: GEV optimization did not converge: {result.message}")
            self.params = result.x
            return
        elif self.distribution == 'gamma':
            initial_params = [0.0, 1.0, 1.0, 1.0]
            objective = lambda p: self._log_likelihood_gamma(p, ensemble_mean, y_train)
        elif self.distribution == 'weibull':
            initial_params = [0.0, 0.0, 0.0, 0.0]
            objective = lambda p: self._log_likelihood_weibull(p, ensemble_mean, ensemble_std, y_train)
        elif self.distribution == 'laplace':
            initial_params = [0.0, 1.0, 0.0, 0.0]
            objective = lambda p: self._log_likelihood_laplace(p, ensemble_mean, ensemble_std, y_train)
        elif self.distribution == 'pareto':
            initial_params = [0.0, 0.0, 0.0, 0.0]
            objective = lambda p: self._log_likelihood_pareto(p, ensemble_mean, ensemble_std, y_train)
        else:
            raise ValueError("Unsupported distribution")

        result = minimize(objective, initial_params, method='L-BFGS-B')
        if not result.success:
            print(f"Warning: {self.distribution} optimization did not converge: {result.message}")
        self.params = result.x

    def predict(self, X_test):
        if self.params is None:
            raise ValueError("Model must be fitted before prediction")

        ensemble_mean, ensemble_var = self._compute_ensemble_stats(X_test)
        ensemble_std = np.sqrt(ensemble_var)

        if self.distribution in ['gaussian', 'lognormal']:
            a, b, gamma, delta = self.params
            mu_t = a + b * ensemble_mean
            sigma_t = np.sqrt(np.maximum(gamma**2 + delta**2 * ensemble_var, 1e-8))
            return mu_t, sigma_t
        elif self.distribution == 'logistic':
            a, b, c, d = self.params
            mu_t = a + b * ensemble_mean
            sigma_t = np.exp(c + d * ensemble_std)
            return mu_t, sigma_t
        elif self.distribution == 'gev':
            a, b, c, d, xi = self.params
            mu_t = a + b * ensemble_mean
            sigma_t = np.exp(c + d * ensemble_mean)
            return mu_t, sigma_t, xi
        elif self.distribution == 'gamma':
            a, b, gamma, delta = self.params
            mu_t = np.maximum(a + b * ensemble_mean, 1e-8)
            sigma2_t = np.maximum(gamma**2 + delta**2 * ensemble_mean, 1e-8)
            alpha_t = mu_t**2 / sigma2_t
            beta_t = sigma2_t / mu_t
            return alpha_t, beta_t
        elif self.distribution == 'weibull':
            a, b, c, d = self.params
            scale_t = np.exp(a + b * ensemble_mean)
            shape_t = np.exp(c + d * np.log(ensemble_std + 1e-8))
            return scale_t, shape_t
        elif self.distribution == 'laplace':
            a, b, c, d = self.params
            loc_t = a + b * ensemble_mean
            scale_t = np.exp(c + d * ensemble_std)
            return loc_t, scale_t
        elif self.distribution == 'pareto':
            a, b, c, d = self.params
            scale_t = np.exp(a + b * ensemble_mean)
            shape_t = np.exp(c + d * np.log(ensemble_std + 1e-8)) + 1
            return scale_t, shape_t

    def prob_less_than(self, X_test, q):
        if self.distribution == 'gaussian':
            mu_t, sigma_t = self.predict(X_test)
            return norm.cdf(q, loc=mu_t, scale=sigma_t)
        elif self.distribution == 'lognormal':
            mu_t, sigma_t = self.predict(X_test)
            if q <= 0:
                return np.zeros_like(mu_t)
            return norm.cdf(np.log(q), loc=mu_t, scale=sigma_t)
        elif self.distribution == 'logistic':
            mu_t, sigma_t = self.predict(X_test)
            return logistic.cdf(q, loc=mu_t, scale=sigma_t)
        elif self.distribution == 'gev':
            mu_t, sigma_t, xi = self.predict(X_test)
            return genextreme.cdf(q, c=-xi, loc=mu_t, scale=sigma_t)
        elif self.distribution == 'gamma':
            alpha_t, beta_t = self.predict(X_test)
            return scigamma.cdf(q, a=alpha_t, scale=beta_t)
        elif self.distribution == 'weibull':
            scale_t, shape_t = self.predict(X_test)
            return weibull_min.cdf(q, c=shape_t, scale=scale_t)
        elif self.distribution == 'laplace':
            loc_t, scale_t = self.predict(X_test)
            return laplace.cdf(q, loc=loc_t, scale=scale_t)
        elif self.distribution == 'pareto':
            scale_t, shape_t = self.predict(X_test)
            return pareto.cdf(q, b=shape_t, scale=scale_t)

class WAS_mme_FlexibleNGR_Model:
    """
    A Flexible Nonhomogeneous Regression approach to probabilistic forecasting on gridded data. 
    Fits FlexibleNGR per grid point with the specified distribution, predicts the distribution, 
    then computes tercile probabilities for new data.

    Flexible Nonhomogeneous Regression (NGR) class supporting multiple predictive distributions.

    This class extends standard NGR to handle non-Gaussian predictands by allowing different
    predictive distributions such as Gaussian, lognormal, logistic, Generalized Extreme Value (GEV),
    gamma, weibull, laplace (biexponential), and pareto.
    It uses ensemble statistics (mean and variance) to parameterize the predictive distribution,
    with parameters estimated via maximum likelihood.
    """

    def __init__(self, distribution='gaussian', nb_cores=1):
        """
        Parameters
        ----------
        distribution : str, optional
            The predictive distribution to use (default='gaussian').
        nb_cores : int, optional
            Number of CPU cores for Dask (default=1).

        """
        self.distribution = distribution
        self.nb_cores = nb_cores

    def fit_predict(self, X, y, X_test, t33, t67):
        """
        Trains the FlexibleNGR on (X, y) and predicts tercile class probabilities for X_test.

        Parameters
        ----------
        X : np.ndarray, shape (n_samples, n_members)
            Ensemble predictor data for training.
        y : np.ndarray, shape (n_samples,)
            Observations for training.
        X_test : np.ndarray, shape (n_test, n_members)
            Ensemble predictor data for forecast/test.

        Returns
        -------
        np.ndarray, shape (3, n_test)
            Probability of each tercile class (PB, PN, PA). 
            If invalid, filled with NaNs.
        """
        # Reshape X_test if 1D
        if X_test.ndim == 1:
            X_test = X_test.reshape(1, -1)
        n_test = X_test.shape[0]

        # Identify valid rows
        mask = np.isfinite(y) & np.all(np.isfinite(X), axis=1)
        if np.any(mask):
            # Subset to valid
            y_clean = y[mask]
            X_clean = X[mask, :]

            # # Compute empirical terciles from training observations
            # terciles = np.nanpercentile(y_clean, [33.333333333333336, 66.66666666666667])
            # t33, t67 = terciles

            if np.isnan(t33) or np.isnan(t67):
                return np.full((3, n_test), np.nan)

            # Fit FlexibleNGR
            model = FlexibleNGR(distribution=self.distribution)
            try:
                model.fit(X_clean, y_clean)
            except:
                return np.full((3, n_test), np.nan)

            # Compute cumulative probabilities
            p_less_t33 = model.prob_less_than(X_test, t33)
            p_less_t67 = model.prob_less_than(X_test, t67)

            # Compute tercile probabilities
            p_b = p_less_t33
            p_n = p_less_t67 - p_less_t33
            p_a = 1 - p_less_t67

            # Stack into (3, n_test)
            preds_proba = np.stack([p_b, p_n, p_a], axis=0)

            # Check for invalid values
            if np.any(np.isnan(preds_proba)) or np.any(preds_proba < 0):
                return np.full((3, n_test), np.nan)

            return preds_proba
        else:
            return np.full((3, n_test), np.nan)

    def compute_model(self, X_train, y_train, X_test, Predictant=None, clim_year_start=None, clim_year_end=None):
        """
        Computes Flexible NGR-based class probabilities for each grid cell.

        Parameters
        ----------
        X_train : xarray.DataArray
            Predictors with dims (T, M, Y, X).
        y_train : xarray.DataArray
            Observations with dims (T, Y, X).
        X_test : xarray.DataArray
            Test predictors with dims (forecast, M, Y, X), where 'forecast' may be renamed if conflicting.

        Returns
        -------
        xarray.DataArray
            Class probabilities (probability=3, forecast, Y, X).
        """

        if Predictant is not None and clim_year_start is not None and clim_year_end is not None:
            
            index_start = Predictant.get_index("T").get_loc(str(clim_year_start)).start
            index_end = Predictant.get_index("T").get_loc(str(clim_year_end)).stop
            rainfall_for_tercile = Predictant.isel(T=slice(index_start, index_end))
            terciles = rainfall_for_tercile.quantile([0.33, 0.67], dim='T')
            T1 = terciles.isel(quantile=0).drop_vars('quantile')
            T2 = terciles.isel(quantile=1).drop_vars('quantile')

        else:

            terciles = y_train.quantile([0.33, 0.67], dim='T')
            T1 = terciles.isel(quantile=0).drop_vars('quantile')
            T2 = terciles.isel(quantile=1).drop_vars('quantile')
            
        # Chunk sizes
        chunksize_x = np.round(len(y_train.get_index("X")) / self.nb_cores)
        chunksize_y = np.round(len(y_train.get_index("Y")) / self.nb_cores)
        
        # Align and transpose
        X_train['T'] = y_train['T']
        X_train = X_train.transpose('T', 'M', 'Y', 'X')
        y_train = y_train.transpose('T', 'Y', 'X')
        
        # Handle test dim to avoid conflict
        if 'T' in X_test.dims:
            X_test = X_test.rename({'T': 'forecast'})
        X_test = X_test.transpose('forecast', 'M', 'Y', 'X')

        # Dask client
        client = Client(n_workers=self.nb_cores, threads_per_worker=1)

        result = xr.apply_ufunc(
            self.fit_predict,
            X_train.chunk({'Y': chunksize_y, 'X': chunksize_x}),
            y_train.chunk({'Y': chunksize_y, 'X': chunksize_x}),
            X_test.chunk({'Y': chunksize_y, 'X': chunksize_x}),
            T1.chunk({'Y': chunksize_y, 'X': chunksize_x}),
            T2.chunk({'Y': chunksize_y, 'X': chunksize_x}),
            input_core_dims=[('T', 'M'), ('T',), ('forecast', 'M'), (), ()],
            output_core_dims=[('probability', 'forecast')],
            vectorize=True,
            dask='parallelized',
            output_dtypes=['float'],
            dask_gufunc_kwargs={'output_sizes': {'probability': 3}},
        )
        
        result_ = result.compute()
        client.close()

        # Rename back to 'T'
        result_ = result_.rename({'forecast': 'T'})

        return result_.assign_coords(probability=['PB', 'PN', 'PA'])

    def forecast(self, Predictant, clim_year_start, clim_year_end, Predictor, Predictor_for_year):

        """
        Runs the Flexible NGR model for a single forecast period.

        Parameters
        ----------
        Predictant : xarray.DataArray
            Observed variable (T, Y, X).
        Predictor : xarray.DataArray
            Training ensemble predictors (T, M, Y, X).
        Predictor_for_year : xarray.DataArray
            Ensemble predictors for forecast (T, M, Y, X).

        Returns
        -------
        xarray.DataArray
            Probabilities (PB, PN, PA, T=1, Y, X) with forecast time.
        """
        if "M" in Predictant.coords:
            Predictant = Predictant.isel(M=0).drop_vars('M').squeeze()
        else:
            Predictant = Predictant
            
        index_start = Predictant.get_index("T").get_loc(str(clim_year_start)).start
        index_end = Predictant.get_index("T").get_loc(str(clim_year_end)).stop
        rainfall_for_tercile = Predictant.isel(T=slice(index_start, index_end))
        terciles = rainfall_for_tercile.quantile([0.33, 0.67], dim='T')
        T1 = terciles.isel(quantile=0).drop_vars('quantile')
        T2 = terciles.isel(quantile=1).drop_vars('quantile')
        
        # Chunk sizes
        chunksize_x = np.round(len(Predictant.get_index("X")) / self.nb_cores)
        chunksize_y = np.round(len(Predictant.get_index("Y")) / self.nb_cores)
        
        # Align T for training
        Predictor['T'] = Predictant['T']
        Predictor = Predictor.transpose('T', 'M', 'Y', 'X')
        Predictant = Predictant.transpose('T', 'Y', 'X')
        
        # Handle forecast dim to avoid conflict
        if 'T' in Predictor_for_year.dims:
            Predictor_for_year_ = Predictor_for_year.rename({'T': 'forecast'})
        else:
            Predictor_for_year_ = Predictor_for_year.transpose('forecast', 'M', 'Y', 'X')
        
        # Parallel with Dask
        client = Client(n_workers=self.nb_cores, threads_per_worker=1)
        result = xr.apply_ufunc(
            self.fit_predict,
            Predictor.chunk({'Y': chunksize_y, 'X': chunksize_x}),
            Predictant.chunk({'Y': chunksize_y, 'X': chunksize_x}),
            Predictor_for_year_.chunk({'Y': chunksize_y, 'X': chunksize_x}),
            T1.chunk({'Y': chunksize_y, 'X': chunksize_x}),
            T2.chunk({'Y': chunksize_y, 'X': chunksize_x}),
            input_core_dims=[('T', 'M'), ('T',), ('forecast', 'M'), (), ()],
            output_core_dims=[('probability', 'forecast')],
            vectorize=True,
            dask='parallelized',
            output_dtypes=['float'],
            dask_gufunc_kwargs={'output_sizes': {'probability': 3}},
        )

        # Compute
        result_ = result.compute()
        client.close()
        
        # Rename forecast dim to T
        result_ = result_.rename({'forecast': 'T'})
        
        # Adjust the T coordinate value
        year = Predictor_for_year.coords['T'].values[0].astype('datetime64[Y]').astype(int) + 1970
        T_value_1 = Predictant.isel(T=0).coords['T'].values
        month_1 = T_value_1.astype('datetime64[M]').astype(int) % 12 + 1
        new_T_value = np.datetime64(f"{year}-{month_1:02d}-01")
        result_ = result_.assign_coords(T=xr.DataArray([new_T_value], dims=["T"]))
        result_['T'] = result_['T'].astype('datetime64[ns]') 
        
        # Label probabilities
        result_ = result_.assign_coords(probability=['PB', 'PN', 'PA'])
        
        # Transpose and return
        return result_.transpose('probability', 'T', 'Y', 'X')


class BMA_Sloughter:
    def __init__(self, distribution='normal'):
        """
        Initialize the BMA model with the specified distribution.

        Parameters:
        - distribution: str, one of 'gaussian', 'gamma', 'lognormal', 'weibull', 't' (default: 'gaussian')
        """
        if distribution not in ['normal', 'gamma', 'lognormal', 'weibull', 't']:
            raise ValueError("Unsupported distribution. Choose from 'normal', 'gamma', 'lognormal', 'weibull', 't'.")
        self.distribution = distribution
        self.debiasing_models = []
        self.weights = None
        self.disp = None  # Dispersion parameters (e.g., sigma for gaussian, alpha for gamma, etc.)
        self.dist_configs = self._get_dist_configs()
        self.config = self.dist_configs[self.distribution]
        if self.distribution == 't':
            self.df = 22  # Fixed degrees of freedom for Student's t
        else:
            self.df = None

    def _get_dist_configs(self):
        return {
            'normal': {
                'pdf': lambda y, mu, sigma: norm.pdf(y, loc=mu, scale=sigma),
                'cdf': lambda q, mu, sigma: norm.cdf(q, loc=mu, scale=sigma),
                'initial': lambda res: np.std(res) if len(res) > 0 else 1.0,
                'bounds': (1e-5, np.inf),
                'closed_update': lambda r, res: np.sqrt(np.sum(r * res**2) / np.sum(r)) if np.sum(r) > 0 else 1.0
            },
            'gamma': {
                'pdf': lambda y, mu, alpha: gamma.pdf(y, a=alpha, loc=0, scale=mu / alpha),
                'cdf': lambda q, mu, alpha: gamma.cdf(q, a=alpha, loc=0, scale=mu / alpha),
                'initial': lambda y, mu: (np.mean(mu)**2 / np.var(y)) if np.var(y) > 0 and np.mean(mu) > 0 else 1.0,
                'bounds': (0.01, 1000),
                'closed_update': None
            },
            'lognormal': {
                'pdf': lambda y, mu, s: lognorm.pdf(y, s=s, loc=0, scale=np.exp(np.log(mu) - s**2 / 2)),
                'cdf': lambda q, mu, s: lognorm.cdf(q, s=s, loc=0, scale=np.exp(np.log(mu) - s**2 / 2)),
                'initial': lambda y, mu: np.sqrt(np.log(1 + np.var(y) / np.mean(mu)**2)) if np.mean(mu) > 0 and np.var(y) >= 0 else 0.5,
                'bounds': (0.01, 10),
                'closed_update': None
            },
            'weibull': {
                'pdf': lambda y, mu, c: weibull_min.pdf(y, c=c, loc=0, scale=mu / sp_gamma(1 + 1/c)),
                'cdf': lambda q, mu, c: weibull_min.cdf(q, c=c, loc=0, scale=mu / sp_gamma(1 + 1/c)),
                'initial': lambda y, mu: 2.0,
                'bounds': (0.1, 10),
                'closed_update': None
            },
            't': {
                'pdf': lambda y, mu, scale: t.pdf(y, df=self.df, loc=mu, scale=scale),
                'cdf': lambda q, mu, scale: t.cdf(q, df=self.df, loc=mu, scale=scale),
                'initial': lambda res: np.std(res) * np.sqrt((self.df - 2) / self.df) if self.df > 2 else 1.0,
                'bounds': (1e-5, np.inf),
                'closed_update': None
            }
        }

    def fit(self, ensemble_forecasts, observations):
        """
        Fit the BMA model using historical ensemble forecasts and observations.
        
        Parameters:
        - ensemble_forecasts: 2D array of shape (n_samples, m_members)
        - observations: 1D array of shape (n_samples)
        """
        n_samples, m_members = ensemble_forecasts.shape
        
        # Step 1: Debiasing using linear regression for each ensemble member
        self.debiasing_models = []
        μ_train = np.zeros((n_samples, m_members))
        for k in range(m_members):
            X_k = ensemble_forecasts[:, k].reshape(-1, 1)
            model = LinearRegression()
            model.fit(X_k, observations)
            self.debiasing_models.append(model)
            μ_train[:, k] = model.predict(X_k)
        
        # Handle non-positive μ for positive distributions
        if self.distribution in ['gamma', 'lognormal', 'weibull']:
            μ_train = np.maximum(μ_train, 1e-5)
        
        # Step 2: Estimate BMA weights and dispersion parameters using EM algorithm
        wk = np.ones(m_members) / m_members
        disp = np.zeros(m_members)
        for k in range(m_members):
            res_k = observations - μ_train[:, k]
            if self.distribution in ['normal', 't']:
                disp[k] = self.config['initial'](res_k)
            else:
                disp[k] = self.config['initial'](observations, μ_train[:, k])
        
        prev_LL = -np.inf
        tol = 1e-6
        max_iter = 100
        
        for iter in range(max_iter):
            # E-step: Compute responsibilities (r_tk)
            r_tk = np.zeros((n_samples, m_members))
            for t in range(n_samples):
                log_p_tk = np.log(wk)
                for k in range(m_members):
                    pdf_val = self.config['pdf'](observations[t], μ_train[t, k], disp[k])
                    log_p_tk[k] += np.log(pdf_val + 1e-10)
                log_p_t = np.logaddexp.reduce(log_p_tk)
                r_tk[t] = np.exp(log_p_tk - log_p_t)
            
            # M-step: Update weights
            sum_r_tk = np.sum(r_tk, axis=0)
            wk_new = sum_r_tk / n_samples
            
            # Update dispersion parameters
            disp_new = np.zeros(m_members)
            for k in range(m_members):
                if self.config['closed_update'] is not None:
                    res_k = observations - μ_train[:, k]
                    disp_new[k] = self.config['closed_update'](r_tk[:, k], res_k)
                else:
                    def neg_ll(dsp):
                        pdf_vals = self.config['pdf'](observations, μ_train[:, k], dsp)
                        pdf_vals = np.maximum(pdf_vals, 1e-10)
                        return -np.sum(r_tk[:, k] * np.log(pdf_vals))
                    res = minimize_scalar(neg_ll, bounds=self.config['bounds'], method='bounded', options={'xatol': 1e-5})
                    if res.success:
                        disp_new[k] = res.x
                    else:
                        disp_new[k] = disp[k]
            
            # Update parameters
            wk = wk_new
            disp = disp_new
            
            # Compute log-likelihood for convergence check
            LL = 0
            for t in range(n_samples):
                log_p_tk = np.log(wk)
                for k in range(m_members):
                    pdf_val = self.config['pdf'](observations[t], μ_train[t, k], disp[k])
                    log_p_tk[k] += np.log(pdf_val + 1e-10)
                log_p_t = np.logaddexp.reduce(log_p_tk)
                LL += log_p_t
            
            if iter > 0 and abs(LL - prev_LL) < tol:
                break
            prev_LL = LL
        
        self.weights = wk
        self.disp = disp

    def predict_cdf(self, new_forecasts, q):
        """
        Compute the BMA predictive CDF at a given value q for new forecasts.
        
        Parameters:
        - new_forecasts: 2D array of shape (n_new, m_members)
        - q: Scalar value at which to compute the CDF.
        
        Returns:
        - cdf_values: Array of shape (n_new,)
        """
        n_new, m_members = new_forecasts.shape
        if len(self.debiasing_models) != m_members:
            raise ValueError("Number of ensemble members in new_forecasts does not match the trained model.")
        
        # Compute debiased forecasts (μ_new) for new data
        μ_new = np.zeros((n_new, m_members))
        for k in range(m_members):
            X_k = new_forecasts[:, k].reshape(-1, 1)
            μ_new[:, k] = self.debiasing_models[k].predict(X_k)
        
        if self.distribution in ['gamma', 'lognormal', 'weibull']:
            μ_new = np.maximum(μ_new, 1e-5)
        
        # Compute BMA predictive CDF
        cdf_k = np.zeros((n_new, m_members))
        for k in range(m_members):
            cdf_k[:, k] = self.config['cdf'](q, μ_new[:, k], self.disp[k])
        cdf_values = np.sum(self.weights * cdf_k, axis=1)
        return cdf_values

    def predict_mean(self, new_forecasts):
        """
        Compute the BMA predictive mean for new forecasts.
        
        Parameters:
        - new_forecasts: 2D array of shape (n_new, m_members).
        
        Returns:
        - mean_values: Array of shape (n_new,)
        """
        n_new, m_members = new_forecasts.shape
        if len(self.debiasing_models) != m_members:
            raise ValueError("Number of ensemble members in new_forecasts does not match the trained model.")
        
        # Compute debiased forecasts (μ_new) for new data
        μ_new = np.zeros((n_new, m_members))
        for k in range(m_members):
            X_k = new_forecasts[:, k].reshape(-1, 1)
            μ_new[:, k] = self.debiasing_models[k].predict(X_k)
        
        # Compute BMA predictive mean
        mean_values = np.sum(self.weights * μ_new, axis=1)
        return mean_values



class WAS_mme_BMA_Sloughter:
    
    def __init__(self, dist_method='normal', nb_cores=1):
        
        self.dist_method=dist_method
        self.nb_cores=nb_cores
    
    def fit_predict(self, X, y, X_test):
        
        mask = np.isfinite(y) & np.all(np.isfinite(X), axis=-1)

        if np.any(mask):
            # Subset to valid
            y_clean = y[mask]
            X_clean = X[mask, :]
            model = BMA_Sloughter( self.dist_method)
            model.fit(X_clean, y_clean)
            preds = model.predict_mean(X_test)
            return preds
        else:
            return np.full((1,), np.nan)

    def predict_proba(self, X, y, X_test, t33, t67):
        mask = np.isfinite(y) & np.all(np.isfinite(X), axis=-1)
        
        if np.any(mask):
            # Subset to valid
            y_clean = y[mask]
            X_clean = X[mask, :]

            # # Compute terciles from training observations
            # terciles = np.nanpercentile(y_clean, [33, 67])
            # t33, t67 = terciles

            if np.isnan(t33):
                return np.full(3, np.nan)

            model = BMA_Sloughter( self.dist_method)
            model.fit(X_clean, y_clean)

            p_b = model.predict_cdf(X_test, t33)
            p_n = model.predict_cdf(X_test, t67) - p_b
            p_a = 1 - model.predict_cdf(X_test, t67)
            return np.array([p_b, p_n, p_a])
        else:
            return np.full((3,), np.nan)        
        

    def compute_model(self, X_train, y_train, X_test):
        """
        Compute the model for the given training data.
        Parameters
        ----------
        X_train : xarray.DataArray
            Predictors with dimensions (T, member).
        y_train : xarray.DataArray
            Observations with dimensions (T, Y, X).
        X_test : xarray.DataArray
            Test predictors with dimensions (T, member).

        Returns
        -------
        xarray.DataArray
            Class probabilities (3) for each grid cell.
        """
        # Chunk sizes
        chunksize_x = np.round(len(y_train.get_index("X")) / self.nb_cores)
        chunksize_y = np.round(len(y_train.get_index("Y")) / self.nb_cores)
        
        # Align time
        X_train['T'] = y_train['T']
        X_train = X_train.transpose('T', 'M', 'Y', 'X')
        y_train = y_train.transpose('T', 'Y', 'X')
        
        # Squeeze X_test
        #X_test = X_test.transpose('T', 'M', 'Y', 'X')

        # Handle test dim to avoid conflict
        if 'T' in X_test.dims:
            X_test = X_test.rename({'T': 'forecast'})
        X_test = X_test.transpose('forecast', 'M', 'Y', 'X')

        # Dask client
        client = Client(n_workers=self.nb_cores, threads_per_worker=1)

        result = xr.apply_ufunc(
            self.fit_predict,
            X_train.chunk({'Y': chunksize_y, 'X': chunksize_x}),
            y_train.chunk({'Y': chunksize_y, 'X': chunksize_x}),
            X_test.chunk({'Y': chunksize_y, 'X': chunksize_x}),
            input_core_dims=[('T','M'), ('T',), ('forecast','M')],
            output_core_dims=[('forecast',)],  
            vectorize=True,
            dask='parallelized',
            output_dtypes=['float']
        )
        
        result_ = result.compute()
        client.close()
        return result_.rename({'forecast': 'T'}).transpose('T', 'Y', 'X')


    def compute_prob(self, X_train, y_train, X_test, Predictant=None, clim_year_start=None, clim_year_end=None):
        """
        Computes class probabilities for each grid cell in `y_train`.

        Parameters
        ----------
        X_train : xarray.DataArray
            Predictors with dimensions (T, member).
        y_train : xarray.DataArray
            Observations with dimensions (T, Y, X).
        X_test : xarray.DataArray
            Test predictors with dimensions (T, member).

        Returns
        -------
        xarray.DataArray
            Class probabilities (3) for each grid cell.

        """

        if Predictant is not None and clim_year_start is not None and clim_year_end is not None:
            
            index_start = Predictant.get_index("T").get_loc(str(clim_year_start)).start
            index_end = Predictant.get_index("T").get_loc(str(clim_year_end)).stop
            rainfall_for_tercile = Predictant.isel(T=slice(index_start, index_end))
            terciles = rainfall_for_tercile.quantile([0.33, 0.67], dim='T')
            T1 = terciles.isel(quantile=0).drop_vars('quantile')
            T2 = terciles.isel(quantile=1).drop_vars('quantile')

        else:
            
            index_start = y_train.get_index("T").get_loc(str(clim_year_start)).start
            index_end = y_train.get_index("T").get_loc(str(clim_year_end)).stop
            rainfall_for_tercile = y_train.isel(T=slice(index_start, index_end))
            terciles = rainfall_for_tercile.quantile([0.33, 0.67], dim='T')
            T1 = terciles.isel(quantile=0).drop_vars('quantile')
            T2 = terciles.isel(quantile=1).drop_vars('quantile')            

        
        # Chunk sizes
        chunksize_x = np.round(len(y_train.get_index("X")) / self.nb_cores)
        chunksize_y = np.round(len(y_train.get_index("Y")) / self.nb_cores)
        
        # Align time
        X_train['T'] = y_train['T']
        X_train = X_train.transpose('T', 'M', 'Y', 'X')
        y_train = y_train.transpose('T', 'Y', 'X')
        
        # Handle test dim to avoid conflict
        if 'T' in X_test.dims:
            X_test = X_test.rename({'T': 'forecast'})
        X_test = X_test.transpose('forecast', 'M', 'Y', 'X')

        # Dask client
        client = Client(n_workers=self.nb_cores, threads_per_worker=1)

        result = xr.apply_ufunc(
            self.predict_proba,
            X_train.chunk({'Y': chunksize_y, 'X': chunksize_x}),
            y_train.chunk({'Y': chunksize_y, 'X': chunksize_x}),
            X_test.chunk({'Y': chunksize_y, 'X': chunksize_x}),
            T1.chunk({'Y': chunksize_y, 'X': chunksize_x}),
            T2.chunk({'Y': chunksize_y, 'X': chunksize_x}),
            input_core_dims=[('T','M'), ('T',), ('forecast','M'), (), ()],
            output_core_dims=[('probability', 'forecast')],  
            vectorize=True,
            dask='parallelized',
            output_dtypes=['float'],
            dask_gufunc_kwargs={'output_sizes': {'probability': 3}},  
        )
        
        result_ = result.compute()
        client.close()
        return result_.rename({'forecast': 'T'}).assign_coords(probability=('probability', ['PB', 'PN', 'PA'])).transpose('probability', 'T', 'Y', 'X')

    def forecast(self, predictant, clim_year_start, clim_year_end, Predictor, Predictor_for_year):
        
        if "M" in predictant.coords:
            predictant = predictant.isel(M=0).drop_vars('M').squeeze()
        else:
            Predictant = Predictant
            
        predict_mean = self.compute_model(Predictor, predictant,  Predictor_for_year)
        predict_proba = self.compute_prob(Predictor, predictant, Predictor_for_year, Predictant=None, clim_year_start=clim_year_start, clim_year_end=clim_year_end)
        
        return predict_mean, predict_proba


class BMA_Sloughter_:
    def __init__(self, distribution='gaussian'):
        """
        Initialize the BMA model with the specified distribution.

        Parameters:
        - distribution: str, one of 'gaussian', 'gamma', 'lognormal', 'weibull', 't' (default: 'gaussian')
        """
        if distribution not in ['gaussian', 'gamma', 'lognormal', 'weibull', 't']:
            raise ValueError("Unsupported distribution. Choose from 'gaussian', 'gamma', 'lognormal', 'weibull', 't'.")
        self.distribution = distribution
        self.debiasing_models = []
        self.weights = None
        self.disp = None  # Dispersion parameters (e.g., sigma for gaussian, alpha for gamma, etc.)
        self.dist_configs = self._get_dist_configs()
        self.config = self.dist_configs[self.distribution]
        if self.distribution == 't':
            self.df = 22  # Fixed degrees of freedom for Student's t
        else:
            self.df = None

    def _get_dist_configs(self):
        return {
            'gaussian': {
                'pdf': lambda y, mu, sigma: norm.pdf(y, loc=mu, scale=sigma),
                'cdf': lambda q, mu, sigma: norm.cdf(q, loc=mu, scale=sigma),
                'initial': lambda res: np.std(res) if len(res) > 0 else 1.0,
                'bounds': (1e-5, np.inf),
                'closed_update': lambda r, res: np.sqrt(np.sum(r * res**2) / np.sum(r)) if np.sum(r) > 0 else 1.0
            },
            'gamma': {
                'pdf': lambda y, mu, alpha: gamma.pdf(y, a=alpha, loc=0, scale=mu / alpha),
                'cdf': lambda q, mu, alpha: gamma.cdf(q, a=alpha, loc=0, scale=mu / alpha),
                'initial': lambda y, mu: (np.mean(mu)**2 / np.var(y)) if np.var(y) > 0 and np.mean(mu) > 0 else 1.0,
                'bounds': (0.01, 1000),
                'closed_update': None
            },
            'lognormal': {
                'pdf': lambda y, mu, s: lognorm.pdf(y, s=s, loc=0, scale=np.exp(np.log(mu) - s**2 / 2)),
                'cdf': lambda q, mu, s: lognorm.cdf(q, s=s, loc=0, scale=np.exp(np.log(mu) - s**2 / 2)),
                'initial': lambda y, mu: np.sqrt(np.log(1 + np.var(y) / np.mean(mu)**2)) if np.mean(mu) > 0 and np.var(y) >= 0 else 0.5,
                'bounds': (0.01, 10),
                'closed_update': None
            },
            'weibull': {
                'pdf': lambda y, mu, c: weibull_min.pdf(y, c=c, loc=0, scale=mu / sp_gamma(1 + 1/c)),
                'cdf': lambda q, mu, c: weibull_min.cdf(q, c=c, loc=0, scale=mu / sp_gamma(1 + 1/c)),
                'initial': lambda y, mu: 2.0,
                'bounds': (0.1, 10),
                'closed_update': None
            },
            't': {
                'pdf': lambda y, mu, scale: t.pdf(y, df=self.df, loc=mu, scale=scale),
                'cdf': lambda q, mu, scale: t.cdf(q, df=self.df, loc=mu, scale=scale),
                'initial': lambda res: np.std(res) * np.sqrt((self.df - 2) / self.df) if self.df > 2 else 1.0,
                'bounds': (1e-5, np.inf),
                'closed_update': None
            }
        }

    def fit(self, ensemble_forecasts, observations):
        """
        Fit the BMA model using historical ensemble forecasts and observations.
        
        Parameters:
        - ensemble_forecasts: 2D array of shape (n_samples, m_members)
        - observations: 1D array of shape (n_samples)
        """
        n_samples, m_members = ensemble_forecasts.shape
        
        # Step 1: Debiasing using linear regression for each ensemble member
        self.debiasing_models = []
        μ_train = np.zeros((n_samples, m_members))
        for k in range(m_members):
            X_k = ensemble_forecasts[:, k].reshape(-1, 1)
            model = LinearRegression()
            model.fit(X_k, observations)
            self.debiasing_models.append(model)
            μ_train[:, k] = model.predict(X_k)
        
        # Handle non-positive μ for positive distributions
        if self.distribution in ['gamma', 'lognormal', 'weibull']:
            μ_train = np.maximum(μ_train, 1e-5)
        
        # Step 2: Estimate BMA weights and dispersion parameters using EM algorithm
        wk = np.ones(m_members) / m_members
        disp = np.zeros(m_members)
        for k in range(m_members):
            res_k = observations - μ_train[:, k]
            if self.distribution in ['gaussian', 't']:
                disp[k] = self.config['initial'](res_k)
            else:
                disp[k] = self.config['initial'](observations, μ_train[:, k])
        
        prev_LL = -np.inf
        tol = 1e-6
        max_iter = 100
        
        for iter in range(max_iter):
            # E-step: Compute responsibilities (r_tk)
            r_tk = np.zeros((n_samples, m_members))
            for t in range(n_samples):
                log_p_tk = np.log(wk)
                for k in range(m_members):
                    pdf_val = self.config['pdf'](observations[t], μ_train[t, k], disp[k])
                    log_p_tk[k] += np.log(pdf_val + 1e-10)
                log_p_t = np.logaddexp.reduce(log_p_tk)
                r_tk[t] = np.exp(log_p_tk - log_p_t)
            
            # M-step: Update weights
            sum_r_tk = np.sum(r_tk, axis=0)
            wk_new = sum_r_tk / n_samples
            
            # Update dispersion parameters
            disp_new = np.zeros(m_members)
            for k in range(m_members):
                if self.config['closed_update'] is not None:
                    res_k = observations - μ_train[:, k]
                    disp_new[k] = self.config['closed_update'](r_tk[:, k], res_k)
                else:
                    def neg_ll(dsp):
                        pdf_vals = self.config['pdf'](observations, μ_train[:, k], dsp)
                        pdf_vals = np.maximum(pdf_vals, 1e-10)
                        return -np.sum(r_tk[:, k] * np.log(pdf_vals))
                    res = minimize_scalar(neg_ll, bounds=self.config['bounds'], method='bounded', options={'xatol': 1e-5})
                    if res.success:
                        disp_new[k] = res.x
                    else:
                        disp_new[k] = disp[k]
            
            # Update parameters
            wk = wk_new
            disp = disp_new
            
            # Compute log-likelihood for convergence check
            LL = 0
            for t in range(n_samples):
                log_p_tk = np.log(wk)
                for k in range(m_members):
                    pdf_val = self.config['pdf'](observations[t], μ_train[t, k], disp[k])
                    log_p_tk[k] += np.log(pdf_val + 1e-10)
                log_p_t = np.logaddexp.reduce(log_p_tk)
                LL += log_p_t
            
            if iter > 0 and abs(LL - prev_LL) < tol:
                break
            prev_LL = LL
        
        self.weights = wk
        self.disp = disp

    def predict_cdf(self, new_forecasts, q):
        """
        Compute the BMA predictive CDF at a given value q for new forecasts.
        
        Parameters:
        - new_forecasts: 2D array of shape (n_new, m_members)
        - q: Scalar value at which to compute the CDF.
        
        Returns:
        - cdf_values: Array of shape (n_new,)
        """
        n_new, m_members = new_forecasts.shape
        if len(self.debiasing_models) != m_members:
            raise ValueError("Number of ensemble members in new_forecasts does not match the trained model.")
        
        # Compute debiased forecasts (μ_new) for new data
        μ_new = np.zeros((n_new, m_members))
        for k in range(m_members):
            X_k = new_forecasts[:, k].reshape(-1, 1)
            μ_new[:, k] = self.debiasing_models[k].predict(X_k)
        
        if self.distribution in ['gamma', 'lognormal', 'weibull']:
            μ_new = np.maximum(μ_new, 1e-5)
        
        # Compute BMA predictive CDF
        cdf_k = np.zeros((n_new, m_members))
        for k in range(m_members):
            cdf_k[:, k] = self.config['cdf'](q, μ_new[:, k], self.disp[k])
        cdf_values = np.sum(self.weights * cdf_k, axis=1)
        return cdf_values

    def predict_mean(self, new_forecasts):
        """
        Compute the BMA predictive mean for new forecasts.
        
        Parameters:
        - new_forecasts: 2D array of shape (n_new, m_members).
        
        Returns:
        - mean_values: Array of shape (n_new,)
        """
        n_new, m_members = new_forecasts.shape
        if len(self.debiasing_models) != m_members:
            raise ValueError("Number of ensemble members in new_forecasts does not match the trained model.")
        
        # Compute debiased forecasts (μ_new) for new data
        μ_new = np.zeros((n_new, m_members))
        for k in range(m_members):
            X_k = new_forecasts[:, k].reshape(-1, 1)
            μ_new[:, k] = self.debiasing_models[k].predict(X_k)
        
        # Compute BMA predictive mean
        mean_values = np.sum(self.weights * μ_new, axis=1)
        return mean_values



class WAS_BMA_Sloughter_:
    
    def __init__(self, dist_method='gaussian', nb_cores=1):
        
        self.dist_method=dist_method
        self.nb_cores=nb_cores
    
    def fit_predict(self, X, y, X_test):
        
        mask = np.isfinite(y) & np.all(np.isfinite(X), axis=-1)

        if np.any(mask):
            # Subset to valid
            y_clean = y[mask]
            X_clean = X[mask, :]
            model = BMA_Sloughter_( self.dist_method)
            model.fit(X_clean, y_clean)
            preds = model.predict_mean(X_test)
            return preds
        else:
            return np.full(X_test.shape[0], np.nan)

    def predict_proba(self, X, y, X_test):
        mask = np.isfinite(y) & np.all(np.isfinite(X), axis=-1)
        
        if np.any(mask):
            # Subset to valid
            y_clean = y[mask]
            X_clean = X[mask, :]

            # Compute terciles from training observations
            terciles = np.nanpercentile(y_clean, [33, 67])
            t33, t67 = terciles

            if np.isnan(t33):
                return np.full((3, X_test.shape[0]), np.nan)

            model = BMA_Sloughter_(self.dist_method)
            model.fit(X_clean, y_clean)

            p_b = model.predict_cdf(X_test, t33)
            p_n = model.predict_cdf(X_test, t67) - p_b
            p_a = 1 - model.predict_cdf(X_test, t67)
            return np.array([p_b, p_n, p_a])
        else:
            return np.full((3, X_test.shape[0]), np.nan)        
        

    def compute_model(self, X_train, y_train, X_test):
        """
        Computes NGR-based class probabilities for each grid cell in `y_train`.

        Parameters
        ----------
        X_train : xarray.DataArray
            Predictors with dimensions (T, member).
        y_train : xarray.DataArray
            Observations with dimensions (T, Y, X).
        X_test : xarray.DataArray
            Test predictors with dimensions (T, member).

        Returns
        -------
        xarray.DataArray
            Class probabilities (3) for each grid cell.
        """
        # Chunk sizes
        chunksize_x = np.round(len(y_train.get_index("X")) / self.nb_cores)
        chunksize_y = np.round(len(y_train.get_index("Y")) / self.nb_cores)
        
        # Align time
        X_train['T'] = y_train['T']
        
        # Add 'M' dim if missing
        if 'M' not in X_train.dims:
            X_train = X_train.expand_dims('M')
        if 'M' not in X_test.dims:
            X_test = X_test.expand_dims('M')
        
        X_train = X_train.transpose('T', 'M', 'Y', 'X')
        y_train = y_train.transpose('T', 'Y', 'X')
        
        # Squeeze X_test
        X_test = X_test.transpose('T', 'M', 'Y', 'X')

        # Dask client
        client = Client(n_workers=self.nb_cores, threads_per_worker=1)

        result = xr.apply_ufunc(
            self.fit_predict,
            X_train.chunk({'Y': chunksize_y, 'X': chunksize_x}),
            y_train.chunk({'Y': chunksize_y, 'X': chunksize_x}),
            X_test.chunk({'Y': chunksize_y, 'X': chunksize_x}),
            input_core_dims=[('T','M'), ('T',), ('T','M')],
            output_core_dims=[('T',)],  
            vectorize=True,
            dask='parallelized',
            output_dtypes=['float']
        )
        
        result_ = result.compute()
        client.close()
        return result_


    def compute_prob(self, X_train, y_train, X_test, Predictant=None, clim_year_start=None, clim_year_end=None):
        """
        Computes NGR-based class probabilities for each grid cell in `y_train`.

        Parameters
        ----------
        X_train : xarray.DataArray
            Predictors with dimensions (T, member).
        y_train : xarray.DataArray
            Observations with dimensions (T, Y, X).
        X_test : xarray.DataArray
            Test predictors with dimensions (T, member).

        Returns
        -------
        xarray.DataArray
            Class probabilities (3) for each grid cell.
        """
        # Chunk sizes
        chunksize_x = np.round(len(y_train.get_index("X")) / self.nb_cores)
        chunksize_y = np.round(len(y_train.get_index("Y")) / self.nb_cores)
        
        # Align time
        X_train['T'] = y_train['T']
        
        # Add 'M' dim if missing
        if 'M' not in X_train.dims:
            X_train = X_train.expand_dims('M')
        if 'M' not in X_test.dims:
            X_test = X_test.expand_dims('M')
        
        X_train = X_train.transpose('T', 'M', 'Y', 'X')
        y_train = y_train.transpose('T', 'Y', 'X')
        
        # Squeeze X_test
        X_test = X_test.transpose('T', 'M', 'Y', 'X')

        # Dask client
        client = Client(n_workers=self.nb_cores, threads_per_worker=1)

        result = xr.apply_ufunc(
            self.predict_proba,
            X_train.chunk({'Y': chunksize_y, 'X': chunksize_x}),
            y_train.chunk({'Y': chunksize_y, 'X': chunksize_x}),
            X_test.chunk({'Y': chunksize_y, 'X': chunksize_x}),
            input_core_dims=[('T','M'), ('T',), ('T','M')],
            output_core_dims=[('probability', 'T')],  
            vectorize=True,
            dask='parallelized',
            output_dtypes=['float'],
            dask_gufunc_kwargs={'output_sizes': {'probability': 3}},  
        )
        
        result_ = result.compute()
        client.close()
        return result_

    def forecast(self, Predictant, Predictor, Predictor_for_year):
        predict_mean = self.compute_model(Predictant, Predictor, Predictor_for_year)
        predict_proba = self.compute_prob(Predictant, Predictor, Predictor_for_year)
        return predict_mean, predict_proba


##################### Stacking bizarre ############################

class WAS_mme_RF_hpELM_:
    """
    Stacking ensemble with Random Forest as base model and HPELM as meta-model for Multi-Model Ensemble (MME) forecasting.
    This class implements a stacking ensemble using RandomForestRegressor as the base model,
    with an HPELM meta-model, for deterministic forecasting and optional tercile probability calculations.
    Implements hyperparameter optimization via randomized search for both base and meta models.

    Parameters
    ----------
    n_estimators_range : list of int, optional
        List of n_estimators values to tune for Random Forest (default is [50, 100, 200, 300]).
    max_depth_range : list of int or None, optional
        List of max depths to tune for Random Forest (default is [None, 10, 20, 30]).
    min_samples_split_range : list of int, optional
        List of minimum samples required to split to tune for Random Forest (default is [2, 5, 10]).
    min_samples_leaf_range : list of int, optional
        List of minimum samples required at leaf node to tune for Random Forest (default is [1, 2, 4]).
    max_features_range : list of str or float, optional
        List of max features to tune for Random Forest (default is ['auto', 'sqrt', 0.33, 0.5]).
    neurons_range : list of int, optional
        List of neuron counts to tune for HPELM (default is [10, 20, 50, 100]).
    activation_options : list of str, optional
        Activation functions to tune for HPELM (default is ['sigm', 'tanh', 'relu', 'rbf_linf', 'rbf_gauss']).
    norm_range : list of float, optional
        Regularization parameters to tune for HPELM (default is [0.1, 1.0, 10.0, 100.0]).
    random_state : int, optional
        Seed for reproducibility (default is 42).
    dist_method : str, optional
        Distribution method for tercile probabilities ('t', 'gamma', etc.) (default is 'gamma').
    n_iter_search : int, optional
        Number of iterations for randomized search (default is 10).
    cv_folds : int, optional
        Number of cross-validation folds (default is 3).
    """
    def __init__(self,
                 n_estimators_range=[50, 100, 200, 300],
                 max_depth_range=[None, 10, 20, 30],
                 min_samples_split_range=[2, 5, 10],
                 min_samples_leaf_range=[1, 2, 4],
                 max_features_range=['auto', 'sqrt', 0.33, 0.5],
                 neurons_range=[10, 20, 50, 100],
                 activation_options=['sigm', 'tanh', 'relu', 'rbf_linf', 'rbf_gauss'],
                 norm_range=[0.1, 1.0, 10.0, 100.0],
                 random_state=42,
                 dist_method="gamma",
                 n_iter_search=10,
                 cv_folds=3):
        self.n_estimators_range = n_estimators_range
        self.max_depth_range = max_depth_range
        self.min_samples_split_range = min_samples_split_range
        self.min_samples_leaf_range = min_samples_leaf_range
        self.max_features_range = max_features_range
        self.neurons_range = neurons_range
        self.activation_options = activation_options
        self.norm_range = norm_range
        self.random_state = random_state
        self.dist_method = dist_method
        self.n_iter_search = n_iter_search
        self.cv_folds = cv_folds
        self.stacking_model = None

    def compute_hyperparameters(self, Predictors, Predictand, clim_year_start, clim_year_end):
        """
        Independently computes the best hyperparameters using randomized search on stacked training data.

        Parameters
        ----------
        Predictors : xarray.DataArray
            Training predictor data with dimensions (T, M, Y, X).
        Predictand : xarray.DataArray
            Training predictand data with dimensions (T, Y, X).
        clim_year_start : int
            Start year of the climatology period.
        clim_year_end : int
            End year of the climatology period.

        Returns
        -------
        dict
            Best hyperparameters found.
        """
        X_train = standardize_timeseries(Predictors, clim_year_start, clim_year_end)
        y_train = standardize_timeseries(Predictand, clim_year_start, clim_year_end)

        # Stack training data
        X_train_stacked = X_train.stack(sample=('T', 'Y', 'X')).transpose('sample', 'M').values
        y_train_stacked = y_train.stack(sample=('T', 'Y', 'X')).values.ravel()
        train_nan_mask = np.any(~np.isfinite(X_train_stacked), axis=1) | ~np.isfinite(y_train_stacked)
        X_train_clean = X_train_stacked[~train_nan_mask]
        y_train_clean = y_train_stacked[~train_nan_mask]


        # For Random Forest (base model)
        param_dist_rf = {
            'n_estimators': self.n_estimators_range,
            'max_depth': self.max_depth_range,
            'min_samples_split': self.min_samples_split_range,
            'min_samples_leaf': self.min_samples_leaf_range,
            'max_features': self.max_features_range
        }
        model_rf = RandomForestRegressor(random_state=self.random_state)
        random_search_rf = RandomizedSearchCV(
            model_rf, param_distributions=param_dist_rf, n_iter=self.n_iter_search,
            cv=self.cv_folds, scoring='neg_mean_squared_error',
            random_state=self.random_state, error_score=np.nan
        )
        random_search_rf.fit(X_train_clean, y_train_clean)
        best_rf = random_search_rf.best_params_

        # For HPELM (meta model)
        param_dist_hpelm = {
            'neurons': self.neurons_range,
            'activation': self.activation_options,
            'norm': self.norm_range
        }
        model_hpelm = HPELMWrapper(random_state=self.random_state)
        random_search_hpelm = RandomizedSearchCV(
            model_hpelm, param_distributions=param_dist_hpelm, n_iter=self.n_iter_search,
            cv=self.cv_folds, scoring='neg_mean_squared_error',
            random_state=self.random_state, error_score=np.nan
        )
        random_search_hpelm.fit(X_train_clean, y_train_clean)
        best_hpelm = random_search_hpelm.best_params_

        return {'rf': best_rf, 'hpelm': best_hpelm}

    def compute_model(self, X_train, y_train, X_test, y_test, best_params=None):
        """
        Compute deterministic hindcast using the stacking ensemble of Random Forest (base) and HPELM (meta).

        Parameters
        ----------
        X_train : xarray.DataArray
            Training predictor data with dimensions (T, M, Y, X).
        y_train : xarray.DataArray
            Training predictand data with dimensions (T, Y, X).
        X_test : xarray.DataArray
            Testing predictor data with dimensions (T, M, Y, X).
        y_test : xarray.DataArray
            Testing predictand data with dimensions (T, Y, X).
        best_params : dict, optional
            Pre-computed best hyperparameters for 'rf' and 'hpelm'. If None, computes internally.

        Returns
        -------
        predicted_da : xarray.DataArray
            Deterministic hindcast with dimensions (T, Y, X).
        """
        # Extract coordinate variables from X_test
        time = X_test['T']
        lat = X_test['Y']
        lon = X_test['X']
        n_time = len(X_test.coords['T'])
        n_lat = len(X_test.coords['Y'])
        n_lon = len(X_test.coords['X'])

        # Stack training data
        X_train_stacked = X_train.stack(sample=('T', 'Y', 'X')).transpose('sample', 'M').values
        y_train_stacked = y_train.stack(sample=('T', 'Y', 'X')).transpose('sample', 'M').values
        train_nan_mask = np.any(~np.isfinite(X_train_stacked), axis=1) | np.any(~np.isfinite(y_train_stacked), axis=1)
        X_train_clean = X_train_stacked[~train_nan_mask]
        y_train_clean = y_train_stacked[~train_nan_mask]

        # Stack testing data
        X_test_stacked = X_test.stack(sample=('T', 'Y', 'X')).transpose('sample', 'M').values
        y_test_stacked = y_test.stack(sample=('T', 'Y', 'X')).transpose('sample', 'M').values
        test_nan_mask = np.any(~np.isfinite(X_test_stacked), axis=1) | np.any(~np.isfinite(y_test_stacked), axis=1)

        # Use provided best_params or compute if None
        if best_params is None:
            best_params = self.compute_hyperparameters(X_train_clean, y_train_clean)

        # Initialize base model (Random Forest)
        rf = RandomForestRegressor(**best_params['rf'], random_state=self.random_state)
        base_models = [('rf', rf)]

        # Initialize meta-model (HPELM)
        meta_model = HPELMWrapper(**best_params['hpelm'], random_state=self.random_state)

        # Initialize stacking ensemble
        self.stacking_model = StackingRegressor(estimators=base_models, final_estimator=meta_model)

        # Fit the stacking model on clean training data
        self.stacking_model.fit(X_train_clean, y_train_clean)

        # Predict on clean test data
        y_pred_test = self.stacking_model.predict(X_test_stacked[~test_nan_mask])

        # Reconstruct predictions
        result = np.empty_like(np.squeeze(y_test_stacked))
        result[np.squeeze(test_nan_mask)] = np.squeeze(y_test_stacked[test_nan_mask])
        result[~np.squeeze(test_nan_mask)] = y_pred_test

        predictions_reshaped = result.reshape(n_time, n_lat, n_lon)
        predicted_da = xr.DataArray(
            data=predictions_reshaped,
            coords={'T': time, 'Y': lat, 'X': lon},
            dims=['T', 'Y', 'X']
        )
        return predicted_da

    # ------------------ Probability Calculation Methods ------------------
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
    def calculate_tercile_probabilities_weibull_min(best_guess, error_variance, first_tercile, second_tercile, dof):
        """
        Weibull minimum-based method.
        """
        n_time = len(best_guess)
        pred_prob = np.empty((3, n_time))
        if np.all(np.isnan(best_guess)):
            pred_prob[:] = np.nan
        else:
            error_std = np.sqrt(error_variance)
            pred_prob[0, :] = stats.weibull_min.cdf(first_tercile, c=dof, loc=best_guess, scale=error_std)
            pred_prob[1, :] = stats.weibull_min.cdf(second_tercile, c=dof, loc=best_guess, scale=error_std) - \
                              stats.weibull_min.cdf(first_tercile, c=dof, loc=best_guess, scale=error_std)
            pred_prob[2, :] = 1 - stats.weibull_min.cdf(second_tercile, c=dof, loc=best_guess, scale=error_std)
        return pred_prob

    @staticmethod
    def calculate_tercile_probabilities_gamma(best_guess, error_variance, T1, T2, dof=None):
        """Gamma-distribution based method."""
        n_time = len(best_guess)
        pred_prob = np.empty((3, n_time), dtype=float)
        if np.any(np.isnan(best_guess)) or np.any(np.isnan(error_variance)):
            pred_prob[:] = np.nan
            return pred_prob
        best_guess = np.asarray(best_guess, dtype=float)
        error_variance = np.asarray(error_variance, dtype=float)
        T1 = np.asarray(T1, dtype=float)
        T2 = np.asarray(T2, dtype=float)
        alpha = (best_guess ** 2) / error_variance
        theta = error_variance / best_guess
        cdf_t1 = gamma.cdf(T1, a=alpha, scale=theta)
        cdf_t2 = gamma.cdf(T2, a=alpha, scale=theta)
        pred_prob[0, :] = cdf_t1
        pred_prob[1, :] = cdf_t2 - cdf_t1
        pred_prob[2, :] = 1.0 - cdf_t2
        return pred_prob

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

    @staticmethod
    def calculate_tercile_probabilities_normal(best_guess, error_variance, first_tercile, second_tercile):
        """Normal-distribution based method."""
        n_time = len(best_guess)
        pred_prob = np.empty((3, n_time))
        if np.all(np.isnan(best_guess)):
            pred_prob[:] = np.nan
        else:
            error_std = np.sqrt(error_variance)
            pred_prob[0, :] = norm.cdf(first_tercile, loc=best_guess, scale=error_std)
            pred_prob[1, :] = norm.cdf(second_tercile, loc=best_guess, scale=error_std) - norm.cdf(first_tercile, loc=best_guess, scale=error_std)
            pred_prob[2, :] = 1 - norm.cdf(second_tercile, loc=best_guess, scale=error_std)
        return pred_prob

    @staticmethod
    def calculate_tercile_probabilities_lognormal(best_guess, error_variance, first_tercile, second_tercile):
        """Lognormal-distribution based method."""
        n_time = len(best_guess)
        pred_prob = np.empty((3, n_time))
        if np.any(np.isnan(best_guess)) or np.any(np.isnan(error_variance)):
            pred_prob[:] = np.nan
            return pred_prob
        sigma = np.sqrt(np.log(1 + error_variance / (best_guess ** 2)))
        mu = np.log(best_guess) - sigma ** 2 / 2
        pred_prob[0, :] = lognorm.cdf(first_tercile, s=sigma, scale=np.exp(mu))
        pred_prob[1, :] = lognorm.cdf(second_tercile, s=sigma, scale=np.exp(mu)) - lognorm.cdf(first_tercile, s=sigma, scale=np.exp(mu))
        pred_prob[2, :] = 1 - lognorm.cdf(second_tercile, s=sigma, scale=np.exp(mu))
        return pred_prob

    def compute_prob(self, Predictant, clim_year_start, clim_year_end, hindcast_det):
        """
        Compute tercile probabilities using the chosen distribution method.
        Predictant is expected to be an xarray DataArray with dims (T, Y, X).
        """
        if "M" in Predictant.coords:
            Predictant = Predictant.isel(M=0).drop_vars('M').squeeze()
        mask = xr.where(~np.isnan(Predictant.isel(T=0)), 1, np.nan).drop_vars(['T']).squeeze().to_numpy()

        Predictant = Predictant.transpose('T', 'Y', 'X')
        index_start = Predictant.get_index("T").get_loc(str(clim_year_start)).start
        index_end = Predictant.get_index("T").get_loc(str(clim_year_end)).stop
        rainfall_for_tercile = Predictant.isel(T=slice(index_start, index_end))
        terciles = rainfall_for_tercile.quantile([0.33, 0.67], dim='T')
        error_variance = (Predictant - hindcast_det).var(dim='T')
        dof = len(Predictant.get_index("T")) - 2

        if self.dist_method == "t":
            calc_func = self.calculate_tercile_probabilities
            hindcast_prob = xr.apply_ufunc(
                calc_func,
                hindcast_det,
                error_variance,
                terciles.isel(quantile=0).drop_vars('quantile'),
                terciles.isel(quantile=1).drop_vars('quantile'),
                input_core_dims=[('T',), (), (), ()],
                vectorize=True,
                kwargs={'dof': dof},
                dask='parallelized',
                output_core_dims=[('probability', 'T')],
                output_dtypes=['float'],
                dask_gufunc_kwargs={'output_sizes': {'probability': 3}, 'allow_rechunk': True}
            )
        elif self.dist_method == "weibull_min":
            calc_func = self.calculate_tercile_probabilities_weibull_min
            hindcast_prob = xr.apply_ufunc(
                calc_func,
                hindcast_det,
                error_variance,
                terciles.isel(quantile=0).drop_vars('quantile'),
                terciles.isel(quantile=1).drop_vars('quantile'),
                input_core_dims=[('T',), (), (), ()],
                vectorize=True,
                kwargs={'dof': dof},
                dask='parallelized',
                output_core_dims=[('probability', 'T')],
                output_dtypes=['float'],
                dask_gufunc_kwargs={'output_sizes': {'probability': 3}, 'allow_rechunk': True}
            )
        elif self.dist_method == "gamma":
            calc_func = self.calculate_tercile_probabilities_gamma
            hindcast_prob = xr.apply_ufunc(
                calc_func,
                hindcast_det,
                error_variance,
                terciles.isel(quantile=0).drop_vars('quantile'),
                terciles.isel(quantile=1).drop_vars('quantile'),
                input_core_dims=[('T',), (), (), ()],
                vectorize=True,
                dask='parallelized',
                output_core_dims=[('probability', 'T')],
                output_dtypes=['float'],
                dask_gufunc_kwargs={'output_sizes': {'probability': 3}, 'allow_rechunk': True}
            )
        elif self.dist_method == "nonparam":
            calc_func = self.calculate_tercile_probabilities_nonparametric
            error_samples = Predictant - hindcast_det
            hindcast_prob = xr.apply_ufunc(
                calc_func,
                hindcast_det,
                error_samples,
                terciles.isel(quantile=0).drop_vars('quantile'),
                terciles.isel(quantile=1).drop_vars('quantile'),
                input_core_dims=[('T',), ('T',), (), ()],
                output_core_dims=[('probability', 'T')],
                vectorize=True,
                dask='parallelized',
                output_dtypes=[float],
                dask_gufunc_kwargs={'output_sizes': {'probability': 3}, 'allow_rechunk': True}
            )
        elif self.dist_method == "normal":
            calc_func = self.calculate_tercile_probabilities_normal
            hindcast_prob = xr.apply_ufunc(
                calc_func,
                hindcast_det,
                error_variance,
                terciles.isel(quantile=0).drop_vars('quantile'),
                terciles.isel(quantile=1).drop_vars('quantile'),
                input_core_dims=[('T',), (), (), ()],
                vectorize=True,
                dask='parallelized',
                output_core_dims=[('probability', 'T')],
                output_dtypes=['float'],
                dask_gufunc_kwargs={'output_sizes': {'probability': 3}, 'allow_rechunk': True}
            )
        elif self.dist_method == "lognormal":
            calc_func = self.calculate_tercile_probabilities_lognormal
            hindcast_prob = xr.apply_ufunc(
                calc_func,
                hindcast_det,
                error_variance,
                terciles.isel(quantile=0).drop_vars('quantile'),
                terciles.isel(quantile=1).drop_vars('quantile'),
                input_core_dims=[('T',), (), (), ()],
                vectorize=True,
                dask='parallelized',
                output_core_dims=[('probability', 'T')],
                output_dtypes=['float'],
                dask_gufunc_kwargs={'output_sizes': {'probability': 3}, 'allow_rechunk': True}
            )
        else:
            raise ValueError(f"Invalid dist_method: {self.dist_method}.")

        hindcast_prob = hindcast_prob.assign_coords(probability=('probability', ['PB', 'PN', 'PA']))
        return hindcast_prob.transpose('probability', 'T', 'Y', 'X')

    @staticmethod
    def _reshape_and_filter_data(da):
        """
        Helper: stack the DataArray from (T, Y, X[, M]) to (n_samples, n_features)
        and remove rows containing NaNs.
        """
        da_stacked = da.stack(sample=('T', 'Y', 'X'))
        if 'M' in da.dims:
            da_stacked = da_stacked.transpose('sample', 'M')
        else:
            da_stacked = da_stacked.transpose('sample')
        da_values = da_stacked.values
        nan_mask = np.any(np.isnan(da_values), axis=1)
        return da_values[~nan_mask], nan_mask, da_values

    def forecast(self, Predictant, clim_year_start, clim_year_end, hindcast_det, hindcast_det_cross, Predictor_for_year, best_params=None):
        """
        Generate deterministic and probabilistic forecast for a target year using the stacking ensemble.

        Parameters
        ----------
        Predictant : xarray.DataArray
            Observed predictand data with dimensions (T, Y, X) or (T, M, Y, X).
        clim_year_start : int or str
            Start year of the climatology period.
        clim_year_end : int or str
            End year of the climatology period.
        hindcast_det : xarray.DataArray
            Deterministic hindcast data for training with dimensions (T, M, Y, X).
        hindcast_det_cross : xarray.DataArray
            Deterministic hindcast data for error estimation with dimensions (T, Y, X).
        Predictor_for_year : xarray.DataArray
            Predictor data for the target year with dimensions (T, M, Y, X).
        best_params : dict, optional
            Pre-computed best hyperparameters for 'rf' and 'hpelm'. If None, computes internally.

        Returns
        -------
        forecast_det : xarray.DataArray
            Deterministic forecast with dimensions (T, Y, X).
        forecast_prob : xarray.DataArray
            Tercile probabilities with dimensions (probability, T, Y, X).
        """
        if "M" in Predictant.coords:
            Predictant_no_m = Predictant.isel(M=0).drop_vars('M').squeeze()
        else:
            Predictant_no_m = Predictant

        mask = xr.where(~np.isnan(Predictant_no_m.isel(T=0)), 1, np.nan).drop_vars(['T']).squeeze().to_numpy()

        # Standardize Predictor_for_year using hindcast climatology
        mean_val = hindcast_det.sel(T=slice(str(clim_year_start), str(clim_year_end))).mean(dim='T')
        std_val = hindcast_det.sel(T=slice(str(clim_year_start), str(clim_year_end))).std(dim='T')
        Predictor_for_year_st = (Predictor_for_year - mean_val) / std_val

        hindcast_det_st = standardize_timeseries(hindcast_det, clim_year_start, clim_year_end)
        Predictant_st = standardize_timeseries(Predictant_no_m, clim_year_start, clim_year_end)
        y_test = Predictant_st.isel(T=[-1])

        # Extract coordinates from X_test
        time = Predictor_for_year_st['T']
        lat = Predictor_for_year_st['Y']
        lon = Predictor_for_year_st['X']
        n_time = len(Predictor_for_year_st.coords['T'])
        n_lat = len(Predictor_for_year_st.coords['Y'])
        n_lon = len(Predictor_for_year_st.coords['X'])

        # Stack training data and remove rows with NaNs
        X_train_stacked = hindcast_det_st.stack(sample=('T', 'Y', 'X')).transpose('sample', 'M').values
        y_train_stacked = Predictant_st.stack(sample=('T', 'Y', 'X')).transpose('sample', 'M').values
        train_nan_mask = np.any(~np.isfinite(X_train_stacked), axis=1) | ~np.isfinite(y_train_stacked)
        X_train_clean = X_train_stacked[~train_nan_mask]
        y_train_clean = y_train_stacked[~train_nan_mask]

        # Stack testing data
        X_test_stacked = Predictor_for_year_st.stack(sample=('T', 'Y', 'X')).transpose('sample', 'M').values
        y_test_stacked = y_test.stack(sample=('T', 'Y', 'X')).values.ravel()
        test_nan_mask = np.any(~np.isfinite(X_test_stacked), axis=1) | ~np.isfinite(y_test_stacked)

        # Use provided best_params or compute if None
        if best_params is None:
            best_params = self.compute_hyperparameters(X_train_clean, y_train_clean)

        # Initialize base model (Random Forest)
        rf = RandomForestRegressor(**best_params['rf'], random_state=self.random_state)
        base_models = [('rf', rf)]

        # Initialize meta-model (HPELM)
        meta_model = HPELMWrapper(**best_params['hpelm'], random_state=self.random_state)

        # Initialize stacking ensemble
        self.stacking_model = StackingRegressor(estimators=base_models, final_estimator=meta_model)

        # Fit the stacking model
        self.stacking_model.fit(X_train_clean, y_train_clean)
        y_pred = self.stacking_model.predict(X_test_stacked[~test_nan_mask])

        # Reconstruct the prediction array
        result = np.empty_like(np.squeeze(y_test_stacked))
        result[np.squeeze(test_nan_mask)] = np.squeeze(y_test_stacked[test_nan_mask])
        result[~np.squeeze(test_nan_mask)] = y_pred

        predictions_reshaped = result.reshape(n_time, n_lat, n_lon)
        result_da = xr.DataArray(
            data=predictions_reshaped,
            coords={'T': time, 'Y': lat, 'X': lon},
            dims=['T', 'Y', 'X']
        ) * mask

        result_da = reverse_standardize(result_da, Predictant_no_m, clim_year_start, clim_year_end)
        year = Predictor_for_year.coords['T'].values.astype('datetime64[Y]').astype(int)[0] + 1970
        T_value_1 = Predictant_no_m.isel(T=0).coords['T'].values
        month_1 = T_value_1.astype('datetime64[M]').astype(int) % 12 + 1
        new_T_value = np.datetime64(f"{year}-{month_1:02d}-01")
        result_da = result_da.assign_coords(T=xr.DataArray([new_T_value], dims=["T"]))
        result_da['T'] = result_da['T'].astype('datetime64[ns]')

        # Compute tercile probabilities
        index_start = Predictant_no_m.get_index("T").get_loc(str(clim_year_start)).start
        index_end = Predictant_no_m.get_index("T").get_loc(str(clim_year_end)).stop
        rainfall_for_tercile = Predictant_no_m.isel(T=slice(index_start, index_end))
        terciles = rainfall_for_tercile.quantile([0.33, 0.67], dim='T')
        error_variance = (Predictant_no_m - hindcast_det_cross).var(dim='T')
        dof = len(Predictant_no_m.get_index("T")) - 2

        if self.dist_method == "t":
            calc_func = self.calculate_tercile_probabilities
            hindcast_prob = xr.apply_ufunc(
                calc_func,
                result_da,
                error_variance,
                terciles.isel(quantile=0).drop_vars('quantile'),
                terciles.isel(quantile=1).drop_vars('quantile'),
                input_core_dims=[('T',), (), (), ()],
                vectorize=True,
                kwargs={'dof': dof},
                dask='parallelized',
                output_core_dims=[('probability', 'T')],
                output_dtypes=['float'],
                dask_gufunc_kwargs={'output_sizes': {'probability': 3}, 'allow_rechunk': True}
            )
        elif self.dist_method == "weibull_min":
            calc_func = self.calculate_tercile_probabilities_weibull_min
            hindcast_prob = xr.apply_ufunc(
                calc_func,
                result_da,
                error_variance,
                terciles.isel(quantile=0).drop_vars('quantile'),
                terciles.isel(quantile=1).drop_vars('quantile'),
                input_core_dims=[('T',), (), (), ()],
                vectorize=True,
                kwargs={'dof': dof},
                dask='parallelized',
                output_core_dims=[('probability', 'T')],
                output_dtypes=['float'],
                dask_gufunc_kwargs={'output_sizes': {'probability': 3}, 'allow_rechunk': True}
            )
        elif self.dist_method == "gamma":
            calc_func = self.calculate_tercile_probabilities_gamma
            hindcast_prob = xr.apply_ufunc(
                calc_func,
                result_da,
                error_variance,
                terciles.isel(quantile=0).drop_vars('quantile'),
                terciles.isel(quantile=1).drop_vars('quantile'),
                input_core_dims=[('T',), (), (), ()],
                vectorize=True,
                dask='parallelized',
                output_core_dims=[('probability', 'T')],
                output_dtypes=['float'],
                dask_gufunc_kwargs={'output_sizes': {'probability': 3}, 'allow_rechunk': True}
            )
        elif self.dist_method == "normal":
            calc_func = self.calculate_tercile_probabilities_normal
            hindcast_prob = xr.apply_ufunc(
                calc_func,
                result_da,
                error_variance,
                terciles.isel(quantile=0).drop_vars('quantile'),
                terciles.isel(quantile=1).drop_vars('quantile'),
                input_core_dims=[('T',), (), (), ()],
                vectorize=True,
                dask='parallelized',
                output_core_dims=[('probability', 'T')],
                output_dtypes=['float'],
                dask_gufunc_kwargs={'output_sizes': {'probability': 3}, 'allow_rechunk': True}
            )
        elif self.dist_method == "lognormal":
            calc_func = self.calculate_tercile_probabilities_lognormal
            hindcast_prob = xr.apply_ufunc(
                calc_func,
                result_da,
                error_variance,
                terciles.isel(quantile=0).drop_vars('quantile'),
                terciles.isel(quantile=1).drop_vars('quantile'),
                input_core_dims=[('T',), (), (), ()],
                vectorize=True,
                dask='parallelized',
                output_core_dims=[('probability', 'T')],
                output_dtypes=['float'],
                dask_gufunc_kwargs={'output_sizes': {'probability': 3}, 'allow_rechunk': True}
            )
        elif self.dist_method == "nonparam":
            calc_func = self.calculate_tercile_probabilities_nonparametric
            error_samples = Predictant_no_m - hindcast_det_cross
            error_samples = error_samples.rename({'T': 'S'})
            hindcast_prob = xr.apply_ufunc(
                calc_func,
                result_da,
                error_samples,
                terciles.isel(quantile=0).drop_vars('quantile'),
                terciles.isel(quantile=1).drop_vars('quantile'),
                input_core_dims=[('T',), ('S',), (), ()],
                output_core_dims=[('probability', 'T')],
                vectorize=True,
                dask='parallelized',
                output_dtypes=[float],
                dask_gufunc_kwargs={'output_sizes': {'probability': 3}, 'allow_rechunk': True}
            )
        else:
            raise ValueError(f"Invalid dist_method: {self.dist_method}.")

        hindcast_prob = hindcast_prob.assign_coords(probability=('probability', ['PB', 'PN', 'PA']))
        return result_da * mask, mask * hindcast_prob.transpose('probability', 'T', 'Y', 'X')


class WAS_mme_RF_hpELM:
    """
    Stacking ensemble with Random Forest as base model and HPELM as meta-model for Multi-Model Ensemble (MME) forecasting.
    This class implements a stacking ensemble using RandomForestRegressor as the base model,
    with an HPELM meta-model, for deterministic forecasting and optional tercile probability calculations.
    Implements hyperparameter optimization via randomized search for both base and meta models.
    Parameters
    ----------
    n_estimators_range : list of int, optional
        List of n_estimators values to tune for Random Forest (default is [50, 100, 200, 300]).
    max_depth_range : list of int or None, optional
        List of max depths to tune for Random Forest (default is [None, 10, 20, 30]).
    min_samples_split_range : list of int, optional
        List of minimum samples required to split to tune for Random Forest (default is [2, 5, 10]).
    min_samples_leaf_range : list of int, optional
        List of minimum samples required at leaf node to tune for Random Forest (default is [1, 2, 4]).
    max_features_range : list of str or float, optional
        List of max features to tune for Random Forest (default is ['auto', 'sqrt', 0.33, 0.5]).
    neurons_range : list of int, optional
        List of neuron counts to tune for HPELM (default is [10, 20, 50, 100]).
    activation_options : list of str, optional
        Activation functions to tune for HPELM (default is ['sigm', 'tanh', 'relu', 'rbf_linf', 'rbf_gauss']).
    norm_range : list of float, optional
        Regularization parameters to tune for HPELM (default is [0.1, 1.0, 10.0, 100.0]).
    random_state : int, optional
        Seed for reproducibility (default is 42).
    dist_method : str, optional
        Distribution method for tercile probabilities ('t', 'gamma', etc.) (default is 'gamma').
    n_iter_search : int, optional
        Number of iterations for randomized search (default is 10).
    cv_folds : int, optional
        Number of cross-validation folds (default is 3).
    n_clusters : int, optional
        Number of clusters for homogenized zones (default is 4).
    """
    def __init__(self,
                 n_estimators_range=[50, 100, 200, 300],
                 max_depth_range=[None, 10, 20, 30],
                 min_samples_split_range=[2, 5, 10],
                 min_samples_leaf_range=[1, 2, 4],
                 max_features_range=['auto', 'sqrt', 0.33, 0.5],
                 neurons_range=[10, 20, 50, 100],
                 activation_options=['sigm', 'tanh', 'relu', 'rbf_linf', 'rbf_gauss'],
                 norm_range=[0.1, 1.0, 10.0, 100.0],
                 random_state=42,
                 dist_method="gamma",
                 n_iter_search=10,
                 cv_folds=3,
                 n_clusters=4):
        self.n_estimators_range = n_estimators_range
        self.max_depth_range = max_depth_range
        self.min_samples_split_range = min_samples_split_range
        self.min_samples_leaf_range = min_samples_leaf_range
        self.max_features_range = max_features_range
        self.neurons_range = neurons_range
        self.activation_options = activation_options
        self.norm_range = norm_range
        self.random_state = random_state
        self.dist_method = dist_method
        self.n_iter_search = n_iter_search
        self.cv_folds = cv_folds
        self.n_clusters = n_clusters
        self.stacking_model = None

    def compute_hyperparameters(self, Predictors, Predictand, clim_year_start, clim_year_end):
        """
        Independently computes the best hyperparameters using randomized search on stacked training data for each homogenized zone.
        Parameters
        ----------
        Predictors : xarray.DataArray
            Training predictor data with dimensions (T, M, Y, X).
        Predictand : xarray.DataArray
            Training predictand data with dimensions (T, Y, X).
        clim_year_start : int
            Start year of the climatology period.
        clim_year_end : int
            End year of the climatology period.
        Returns
        -------
        best_params_dict : dict
            Best hyperparameters for each cluster.
        cluster_da : xarray.DataArray
            Cluster labels with dimensions (Y, X).
        """
        if "M" in Predictand.coords:
            Predictand = Predictand.isel(M=0).drop_vars('M').squeeze()
           
        X_train_std = standardize_timeseries(Predictors, clim_year_start, clim_year_end)
        Predictand.name = "varname"

        # Cluster on predictand time series
        y_for_cluster = Predictand.stack(space=('Y', 'X')).transpose('space', 'T').values
        finite_mask = np.all(np.isfinite(y_for_cluster), axis=1)
        y_cluster = y_for_cluster[finite_mask]
        kmeans = KMeans(n_clusters=self.n_clusters, random_state=self.random_state)
        labels = kmeans.fit_predict(y_cluster)
        full_labels = np.full(y_for_cluster.shape[0], np.nan)  # Use nan instead of -1 for invalid
        full_labels[finite_mask] = labels
        cluster_da = xr.DataArray(
            full_labels.reshape(len(y_train_std['Y']), len(y_train_std['X'])),
            coords={'Y': y_train_std['Y'], 'X': y_train_std['X']},
            dims=['Y', 'X']
        )
        y_train_std = standardize_timeseries(Predictand, clim_year_start, clim_year_end)
        # Align X_train_std with y_train_std
        X_train_std['T'] = y_train_std['T']
        # Prepare parameter distributions for RF and HPELM
        param_dist_rf = {
            'n_estimators': self.n_estimators_range,
            'max_depth': self.max_depth_range,
            'min_samples_split': self.min_samples_split_range,
            'min_samples_leaf': self.min_samples_leaf_range,
            'max_features': self.max_features_range
        }
        param_dist_hpelm = {
            'neurons': self.neurons_range,
            'activation': self.activation_options,
            'norm': self.norm_range
        }
        best_params_dict = {}
        for c in range(self.n_clusters):
            mask_3d = (cluster_da == c).expand_dims({'T': y_train_std['T']})
            X_stacked_c = X_train_std.where(mask_3d).stack(sample=('T', 'Y', 'X')).transpose('sample', 'M').values
            y_stacked_c = y_train_std.where(mask_3d).stack(sample=('T', 'Y', 'X')).values.ravel()
            nan_mask_c = np.any(~np.isfinite(X_stacked_c), axis=1) | ~np.isfinite(y_stacked_c)
            X_clean_c = X_stacked_c[~nan_mask_c]
            y_clean_c = y_stacked_c[~nan_mask_c]
            if len(X_clean_c) == 0:
                continue
            # Tune Random Forest
            model_rf = RandomForestRegressor(random_state=self.random_state, n_jobs=-1)
            random_search_rf = RandomizedSearchCV(
                model_rf, param_distributions=param_dist_rf, n_iter=self.n_iter_search,
                cv=self.cv_folds, scoring='neg_mean_squared_error',
                random_state=self.random_state, error_score=np.nan
            )
            random_search_rf.fit(X_clean_c, y_clean_c)
            best_rf = random_search_rf.best_params_
            # Tune HPELM
            model_hpelm = HPELMWrapper(random_state=self.random_state)
            random_search_hpelm = RandomizedSearchCV(
                model_hpelm, param_distributions=param_dist_hpelm, n_iter=self.n_iter_search,
                cv=self.cv_folds, scoring='neg_mean_squared_error',
                random_state=self.random_state, error_score=np.nan
            )
            random_search_hpelm.fit(X_clean_c, y_clean_c)
            best_hpelm = random_search_hpelm.best_params_
            best_params_dict[c] = {'rf': best_rf, 'hpelm': best_hpelm}
        return best_params_dict, cluster_da

    def compute_model(self, X_train, y_train, X_test, y_test, best_params=None, cluster_da=None):
        """
        Compute deterministic hindcast using the stacking ensemble of Random Forest (base) and HPELM (meta) for each zone.
        Parameters
        ----------
        X_train : xarray.DataArray
            Training predictor data with dimensions (T, M, Y, X).
        y_train : xarray.DataArray
            Training predictand data with dimensions (T, Y, X).
        X_test : xarray.DataArray
            Testing predictor data with dimensions (T, M, Y, X).
        y_test : xarray.DataArray
            Testing predictand data with dimensions (T, Y, X).
        clim_year_start : int
            Start year of the climatology period.
        clim_year_end : int
            End year of the climatology period.
        best_params : dict, optional
            Pre-computed best hyperparameters per cluster. If None, computes internally.
        cluster_da : xarray.DataArray, optional
            Pre-computed cluster labels. If None, computes internally.
        Returns
        -------
        predicted_da : xarray.DataArray
            Deterministic hindcast with dimensions (T, Y, X).
        """
        # Standardize inputs
        X_train_std = X_train
        y_train_std = y_train
        X_test_std = X_test
        y_test_std = y_test
        
        # Extract coordinate variables from X_test
        time = X_test_std['T']
        lat = X_test_std['Y']
        lon = X_test_std['X']
        
        n_time = len(time)
        n_lat = len(lat)
        n_lon = len(lon)
        # Use provided best_params and cluster_da or compute if None
        if best_params is None:
            best_params, cluster_da = self.compute_hyperparameters(X_train, y_train, clim_year_start, clim_year_end)
            
        # Initialize predictions array
        predictions = np.full((n_time, n_lat, n_lon), np.nan)
        
        self.stacking_model = {}  # Dictionary to store models per cluster
        for c in range(self.n_clusters):
            if c not in best_params:
                continue
                
            bp = best_params[c]
            # Mask for this cluster
            mask_3d_train = (cluster_da == c).expand_dims({'T': X_train_std['T']})
            mask_3d_test = (cluster_da == c).expand_dims({'T': X_test_std['T']})
            # Stack training data for cluster
            X_train_stacked_c = X_train_std.where(mask_3d_train).stack(sample=('T', 'Y', 'X')).transpose('sample', 'M').values
            y_train_stacked_c = y_train_std.where(mask_3d_train).stack(sample=('T', 'Y', 'X')).values.ravel()
            train_nan_mask = np.any(~np.isfinite(X_train_stacked_c), axis=1) | ~np.isfinite(y_train_stacked_c)
            X_train_clean_c = X_train_stacked_c[~train_nan_mask]
            y_train_clean_c = y_train_stacked_c[~train_nan_mask]
            
            # Stack testing data for cluster
            X_test_stacked_c = X_test_std.where(mask_3d_test).stack(sample=('T', 'Y', 'X')).transpose('sample', 'M').values
            y_test_stacked_c = y_test_std.where(mask_3d_test).stack(sample=('T', 'Y', 'X')).values.ravel()
            test_nan_mask = np.any(~np.isfinite(X_test_stacked_c), axis=1) | ~np.isfinite(y_test_stacked_c)
            X_test_clean_c = X_test_stacked_c[~test_nan_mask]
            # Initialize base model (Random Forest)
            rf = RandomForestRegressor(**bp['rf'], random_state=self.random_state, n_jobs=-1)
            # Initialize meta-model (HPELMWrapper)
            meta_model = HPELMWrapper(**bp['hpelm'], random_state=self.random_state)
            # Initialize stacking ensemble for this cluster
            stacking_model_c = StackingRegressor(estimators=[('rf', rf)], final_estimator=meta_model)
            # Fit the stacking model on cluster data
            stacking_model_c.fit(X_train_clean_c, y_train_clean_c)
            self.stacking_model[c] = stacking_model_c
            # Predict
            y_pred_c = stacking_model_c.predict(X_test_clean_c)
            # Reconstruct predictions for this cluster
            result_c = np.empty(len(y_test_stacked_c))
            result_c[test_nan_mask] = y_test_stacked_c[test_nan_mask]
            result_c[~test_nan_mask] = y_pred_c
            pred_c_reshaped = result_c.reshape(n_time, n_lat, n_lon)
            # Fill in the predictions array
            predictions = np.where(np.isnan(predictions), pred_c_reshaped, predictions)
        predicted_da = xr.DataArray(
            data=predictions,
            coords={'T': time, 'Y': lat, 'X': lon},
            dims=['T', 'Y', 'X']
        )
        return predicted_da

    # ------------------ Probability Calculation Methods ------------------
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
            pred_prob[0, :] = t.cdf(first_t, df=dof)
            pred_prob[1, :] = t.cdf(second_t, df=dof) - t.cdf(first_t, df=dof)
            pred_prob[2, :] = 1 - t.cdf(second_t, df=dof)
        return pred_prob

    @staticmethod
    def calculate_tercile_probabilities_weibull_min(best_guess, error_variance, first_tercile, second_tercile, dof):
        """
        Weibull minimum-based method.
        """
        n_time = len(best_guess)
        pred_prob = np.empty((3, n_time))
        if np.all(np.isnan(best_guess)):
            pred_prob[:] = np.nan
        else:
            error_std = np.sqrt(error_variance)
            pred_prob[0, :] = weibull_min.cdf(first_tercile, c=dof, loc=best_guess, scale=error_std)
            pred_prob[1, :] = weibull_min.cdf(second_tercile, c=dof, loc=best_guess, scale=error_std) - \
                              weibull_min.cdf(first_tercile, c=dof, loc=best_guess, scale=error_std)
            pred_prob[2, :] = 1 - weibull_min.cdf(second_tercile, c=dof, loc=best_guess, scale=error_std)
        return pred_prob

    @staticmethod
    def calculate_tercile_probabilities_gamma(best_guess, error_variance, T1, T2, dof=None):
        """Gamma-distribution based method."""
        n_time = len(best_guess)
        pred_prob = np.empty((3, n_time), dtype=float)
        if np.any(np.isnan(best_guess)) or np.any(np.isnan(error_variance)):
            pred_prob[:] = np.nan
            return pred_prob
        best_guess = np.asarray(best_guess, dtype=float)
        error_variance = np.asarray(error_variance, dtype=float)
        T1 = np.asarray(T1, dtype=float)
        T2 = np.asarray(T2, dtype=float)
        alpha = (best_guess ** 2) / error_variance
        theta = error_variance / best_guess
        cdf_t1 = gamma.cdf(T1, a=alpha, scale=theta)
        cdf_t2 = gamma.cdf(T2, a=alpha, scale=theta)
        pred_prob[0, :] = cdf_t1
        pred_prob[1, :] = cdf_t2 - cdf_t1
        pred_prob[2, :] = 1.0 - cdf_t2
        return pred_prob

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

    @staticmethod
    def calculate_tercile_probabilities_normal(best_guess, error_variance, first_tercile, second_tercile):
        """Normal-distribution based method."""
        n_time = len(best_guess)
        pred_prob = np.empty((3, n_time))
        if np.all(np.isnan(best_guess)):
            pred_prob[:] = np.nan
        else:
            error_std = np.sqrt(error_variance)
            pred_prob[0, :] = norm.cdf(first_tercile, loc=best_guess, scale=error_std)
            pred_prob[1, :] = norm.cdf(second_tercile, loc=best_guess, scale=error_std) - norm.cdf(first_tercile, loc=best_guess, scale=error_std)
            pred_prob[2, :] = 1 - norm.cdf(second_tercile, loc=best_guess, scale=error_std)
        return pred_prob

    @staticmethod
    def calculate_tercile_probabilities_lognormal(best_guess, error_variance, first_tercile, second_tercile):
        """Lognormal-distribution based method."""
        n_time = len(best_guess)
        pred_prob = np.empty((3, n_time))
        if np.any(np.isnan(best_guess)) or np.any(np.isnan(error_variance)):
            pred_prob[:] = np.nan
            return pred_prob
        sigma = np.sqrt(np.log(1 + error_variance / (best_guess ** 2)))
        mu = np.log(best_guess) - sigma ** 2 / 2
        pred_prob[0, :] = lognorm.cdf(first_tercile, s=sigma, scale=np.exp(mu))
        pred_prob[1, :] = lognorm.cdf(second_tercile, s=sigma, scale=np.exp(mu)) - lognorm.cdf(first_tercile, s=sigma, scale=np.exp(mu))
        pred_prob[2, :] = 1 - lognorm.cdf(second_tercile, s=sigma, scale=np.exp(mu))
        return pred_prob

    def compute_prob(self, Predictant, clim_year_start, clim_year_end, hindcast_det):
        """
        Compute tercile probabilities using the chosen distribution method.
        Predictant is expected to be an xarray DataArray with dims (T, Y, X).
        """
        if "M" in Predictant.coords:
            Predictant = Predictant.isel(M=0).drop_vars('M').squeeze()
        mask = xr.where(~np.isnan(Predictant.isel(T=0)), 1, np.nan).drop_vars(['T']).squeeze().to_numpy()
        Predictant = Predictant.transpose('T', 'Y', 'X')
        index_start = Predictant.get_index("T").get_loc(str(clim_year_start)).start
        index_end = Predictant.get_index("T").get_loc(str(clim_year_end)).stop
        rainfall_for_tercile = Predictant.isel(T=slice(index_start, index_end))
        terciles = rainfall_for_tercile.quantile([0.33, 0.67], dim='T')
        error_variance = (Predictant - hindcast_det).var(dim='T')
        dof = len(Predictant.get_index("T")) - 2
        if self.dist_method == "t":
            calc_func = self.calculate_tercile_probabilities
            hindcast_prob = xr.apply_ufunc(
                calc_func,
                hindcast_det,
                error_variance,
                terciles.isel(quantile=0).drop_vars('quantile'),
                terciles.isel(quantile=1).drop_vars('quantile'),
                input_core_dims=[('T',), (), (), ()],
                vectorize=True,
                kwargs={'dof': dof},
                dask='parallelized',
                output_core_dims=[('probability', 'T')],
                output_dtypes=['float'],
                dask_gufunc_kwargs={'output_sizes': {'probability': 3}, 'allow_rechunk': True}
            )
        elif self.dist_method == "weibull_min":
            calc_func = self.calculate_tercile_probabilities_weibull_min
            hindcast_prob = xr.apply_ufunc(
                calc_func,
                hindcast_det,
                error_variance,
                terciles.isel(quantile=0).drop_vars('quantile'),
                terciles.isel(quantile=1).drop_vars('quantile'),
                input_core_dims=[('T',), (), (), ()],
                vectorize=True,
                kwargs={'dof': dof},
                dask='parallelized',
                output_core_dims=[('probability', 'T')],
                output_dtypes=['float'],
                dask_gufunc_kwargs={'output_sizes': {'probability': 3}, 'allow_rechunk': True}
            )
        elif self.dist_method == "gamma":
            calc_func = self.calculate_tercile_probabilities_gamma
            hindcast_prob = xr.apply_ufunc(
                calc_func,
                hindcast_det,
                error_variance,
                terciles.isel(quantile=0).drop_vars('quantile'),
                terciles.isel(quantile=1).drop_vars('quantile'),
                input_core_dims=[('T',), (), (), ()],
                vectorize=True,
                dask='parallelized',
                output_core_dims=[('probability', 'T')],
                output_dtypes=['float'],
                dask_gufunc_kwargs={'output_sizes': {'probability': 3}, 'allow_rechunk': True}
            )
        elif self.dist_method == "nonparam":
            calc_func = self.calculate_tercile_probabilities_nonparametric
            error_samples = Predictant - hindcast_det
            hindcast_prob = xr.apply_ufunc(
                calc_func,
                hindcast_det,
                error_samples,
                terciles.isel(quantile=0).drop_vars('quantile'),
                terciles.isel(quantile=1).drop_vars('quantile'),
                input_core_dims=[('T',), ('T',), (), ()],
                output_core_dims=[('probability', 'T')],
                vectorize=True,
                dask='parallelized',
                output_dtypes=[float],
                dask_gufunc_kwargs={'output_sizes': {'probability': 3}, 'allow_rechunk': True}
            )
        elif self.dist_method == "normal":
            calc_func = self.calculate_tercile_probabilities_normal
            hindcast_prob = xr.apply_ufunc(
                calc_func,
                hindcast_det,
                error_variance,
                terciles.isel(quantile=0).drop_vars('quantile'),
                terciles.isel(quantile=1).drop_vars('quantile'),
                input_core_dims=[('T',), (), (), ()],
                vectorize=True,
                dask='parallelized',
                output_core_dims=[('probability', 'T')],
                output_dtypes=['float'],
                dask_gufunc_kwargs={'output_sizes': {'probability': 3}, 'allow_rechunk': True}
            )
        elif self.dist_method == "lognormal":
            calc_func = self.calculate_tercile_probabilities_lognormal
            hindcast_prob = xr.apply_ufunc(
                calc_func,
                hindcast_det,
                error_variance,
                terciles.isel(quantile=0).drop_vars('quantile'),
                terciles.isel(quantile=1).drop_vars('quantile'),
                input_core_dims=[('T',), (), (), ()],
                vectorize=True,
                dask='parallelized',
                output_core_dims=[('probability', 'T')],
                output_dtypes=['float'],
                dask_gufunc_kwargs={'output_sizes': {'probability': 3}, 'allow_rechunk': True}
            )
        else:
            raise ValueError(f"Invalid dist_method: {self.dist_method}.")
        hindcast_prob = hindcast_prob.assign_coords(probability=('probability', ['PB', 'PN', 'PA']))
        return hindcast_prob.transpose('probability', 'T', 'Y', 'X')

    @staticmethod
    def _reshape_and_filter_data(da):
        """
        Helper: stack the DataArray from (T, Y, X[, M]) to (n_samples, n_features)
        and remove rows containing NaNs.
        """
        da_stacked = da.stack(sample=('T', 'Y', 'X'))
        if 'M' in da.dims:
            da_stacked = da_stacked.transpose('sample', 'M')
        else:
            da_stacked = da_stacked.transpose('sample')
        da_values = da_stacked.values
        nan_mask = np.any(np.isnan(da_values), axis=1)
        return da_values[~nan_mask], nan_mask, da_values

    def forecast(self, Predictant, clim_year_start, clim_year_end, hindcast_det, hindcast_det_cross, Predictor_for_year, best_params=None, cluster_da=None):
        """
        Generate deterministic and probabilistic forecast for a target year using the stacking ensemble.
        Parameters
        ----------
        Predictant : xarray.DataArray
            Observed predictand data with dimensions (T, Y, X) or (T, M, Y, X).
        clim_year_start : int or str
            Start year of the climatology period.
        clim_year_end : int or str
            End year of the climatology period.
        hindcast_det : xarray.DataArray
            Deterministic hindcast data for training with dimensions (T, M, Y, X).
        hindcast_det_cross : xarray.DataArray
            Deterministic hindcast data for error estimation with dimensions (T, Y, X).
        Predictor_for_year : xarray.DataArray
            Predictor data for the target year with dimensions (T, M, Y, X).
        best_params : dict, optional
            Pre-computed best hyperparameters per cluster. If None, computes internally.
        cluster_da : xarray.DataArray, optional
            Pre-computed cluster labels. If None, computes internally.
        Returns
        -------
        forecast_det : xarray.DataArray
            Deterministic forecast with dimensions (T, Y, X).
        forecast_prob : xarray.DataArray
            Tercile probabilities with dimensions (probability, T, Y, X).
        """
        if "M" in Predictant.coords:
            Predictant_no_m = Predictant.isel(M=0).drop_vars('M').squeeze()
        else:
            Predictant_no_m = Predictant
        mask = xr.where(~np.isnan(Predictant_no_m.isel(T=0)), 1, np.nan).drop_vars(['T']).squeeze().to_numpy()
        # Standardize Predictor_for_year using hindcast climatology
        mean_val = hindcast_det.sel(T=slice(str(clim_year_start), str(clim_year_end))).mean(dim='T')
        std_val = hindcast_det.sel(T=slice(str(clim_year_start), str(clim_year_end))).std(dim='T')
        Predictor_for_year_st = (Predictor_for_year - mean_val) / std_val
        hindcast_det_st = standardize_timeseries(hindcast_det, clim_year_start, clim_year_end)
        Predictant_st = standardize_timeseries(Predictant_no_m, clim_year_start, clim_year_end)
        hindcast_det_st['T'] = Predictant_st['T']
        
        # Extract coord.inates from X_test
        time = Predictor_for_year_st['T']
        lat = Predictor_for_year_st['Y']
        lon = Predictor_for_year_st['X']
        n_time = len(time)
        n_lat = len(lat)
        n_lon = len(lon)
        
        # Use provided best_params and cluster_da or compute if None
        if best_params is None:
            best_params, cluster_da = self.compute_hyperparameters(hindcast_det, Predictant_no_m, clim_year_start, clim_year_end)
        # Initialize predictions array
        predictions = np.full((n_time, n_lat, n_lon), np.nan)
        self.stacking_model = {}  # Dictionary to store models per cluster
        for c in range(self.n_clusters):
            if c not in best_params:
                continue
            bp = best_params[c]
            # Mask for this cluster
            mask_3d_train = (cluster_da == c).expand_dims({'T': hindcast_det_st['T']})
            mask_3d_test = (cluster_da == c).expand_dims({'T': Predictor_for_year_st['T']})
            # Stack training data for cluster
            X_train_stacked_c = hindcast_det_st.where(mask_3d_train).stack(sample=('T', 'Y', 'X')).transpose('sample', 'M').values
            y_train_stacked_c = Predictant_st.where(mask_3d_train).stack(sample=('T', 'Y', 'X')).values.ravel()
            train_nan_mask = np.any(~np.isfinite(X_train_stacked_c), axis=1) | ~np.isfinite(y_train_stacked_c)
            X_train_clean_c = X_train_stacked_c[~train_nan_mask]
            y_train_clean_c = y_train_stacked_c[~train_nan_mask]
            # Stack testing data for cluster
            X_test_stacked_c = Predictor_for_year_st.where(mask_3d_test).stack(sample=('T', 'Y', 'X')).transpose('sample', 'M').values
            test_nan_mask = np.any(~np.isfinite(X_test_stacked_c), axis=1)
            X_test_clean_c = X_test_stacked_c[~test_nan_mask]
            # Initialize base model (Random Forest)
            rf = RandomForestRegressor(**bp['rf'], random_state=self.random_state, n_jobs=-1)
            # Initialize meta-model (HPELMWrapper)
            meta_model = HPELMWrapper(**bp['hpelm'], random_state=self.random_state)
            # Initialize stacking ensemble for this cluster
            stacking_model_c = StackingRegressor(estimators=[('rf', rf)], final_estimator=meta_model)
            # Fit the stacking model on cluster data
            stacking_model_c.fit(X_train_clean_c, y_train_clean_c)
            self.stacking_model[c] = stacking_model_c
            # Predict
            y_pred_c = stacking_model_c.predict(X_test_clean_c)
            # Reconstruct predictions for this cluster
            result_c = np.full(len(X_test_stacked_c), np.nan)
            result_c[~test_nan_mask] = y_pred_c
            pred_c_reshaped = result_c.reshape(n_time, n_lat, n_lon)
            # Fill in the predictions array
            predictions = np.where(np.isnan(predictions), pred_c_reshaped, predictions)
        result_da = xr.DataArray(
            data=predictions,
            coords={'T': time, 'Y': lat, 'X': lon},
            dims=['T', 'Y', 'X']
        ) * mask
        result_da = reverse_standardize(result_da, Predictant_no_m, clim_year_start, clim_year_end)
        year = Predictor_for_year.coords['T'].values.astype('datetime64[Y]').astype(int)[0] + 1970
        T_value_1 = Predictant_no_m.isel(T=0).coords['T'].values
        month_1 = T_value_1.astype('datetime64[M]').astype(int) % 12 + 1
        new_T_value = np.datetime64(f"{year}-{month_1:02d}-01")
        result_da = result_da.assign_coords(T=xr.DataArray([new_T_value], dims=["T"]))
        result_da['T'] = result_da['T'].astype('datetime64[ns]')
        
        # Compute tercile probabilities
        index_start = Predictant_no_m.get_index("T").get_loc(str(clim_year_start)).start
        index_end = Predictant_no_m.get_index("T").get_loc(str(clim_year_end)).stop
        rainfall_for_tercile = Predictant_no_m.isel(T=slice(index_start, index_end))
        terciles = rainfall_for_tercile.quantile([0.33, 0.67], dim='T')
        error_variance = (Predictant_no_m - hindcast_det_cross).var(dim='T')
        dof = len(Predictant_no_m.get_index("T")) - 2
        if self.dist_method == "t":
            calc_func = self.calculate_tercile_probabilities
            hindcast_prob = xr.apply_ufunc(
                calc_func,
                result_da,
                error_variance,
                terciles.isel(quantile=0).drop_vars('quantile'),
                terciles.isel(quantile=1).drop_vars('quantile'),
                input_core_dims=[('T',), (), (), ()],
                vectorize=True,
                kwargs={'dof': dof},
                dask='parallelized',
                output_core_dims=[('probability', 'T')],
                output_dtypes=['float'],
                dask_gufunc_kwargs={'output_sizes': {'probability': 3}, 'allow_rechunk': True}
            )
        elif self.dist_method == "weibull_min":
            calc_func = self.calculate_tercile_probabilities_weibull_min
            hindcast_prob = xr.apply_ufunc(
                calc_func,
                result_da,
                error_variance,
                terciles.isel(quantile=0).drop_vars('quantile'),
                terciles.isel(quantile=1).drop_vars('quantile'),
                input_core_dims=[('T',), (), (), ()],
                vectorize=True,
                kwargs={'dof': dof},
                dask='parallelized',
                output_core_dims=[('probability', 'T')],
                output_dtypes=['float'],
                dask_gufunc_kwargs={'output_sizes': {'probability': 3}, 'allow_rechunk': True}
            )
        elif self.dist_method == "gamma":
            calc_func = self.calculate_tercile_probabilities_gamma
            hindcast_prob = xr.apply_ufunc(
                calc_func,
                result_da,
                error_variance,
                terciles.isel(quantile=0).drop_vars('quantile'),
                terciles.isel(quantile=1).drop_vars('quantile'),
                input_core_dims=[('T',), (), (), ()],
                vectorize=True,
                dask='parallelized',
                output_core_dims=[('probability', 'T')],
                output_dtypes=['float'],
                dask_gufunc_kwargs={'output_sizes': {'probability': 3}, 'allow_rechunk': True}
            )
        elif self.dist_method == "normal":
            calc_func = self.calculate_tercile_probabilities_normal
            hindcast_prob = xr.apply_ufunc(
                calc_func,
                result_da,
                error_variance,
                terciles.isel(quantile=0).drop_vars('quantile'),
                terciles.isel(quantile=1).drop_vars('quantile'),
                input_core_dims=[('T',), (), (), ()],
                vectorize=True,
                dask='parallelized',
                output_core_dims=[('probability', 'T')],
                output_dtypes=['float'],
                dask_gufunc_kwargs={'output_sizes': {'probability': 3}, 'allow_rechunk': True}
            )
        elif self.dist_method == "lognormal":
            calc_func = self.calculate_tercile_probabilities_lognormal
            hindcast_prob = xr.apply_ufunc(
                calc_func,
                result_da,
                error_variance,
                terciles.isel(quantile=0).drop_vars('quantile'),
                terciles.isel(quantile=1).drop_vars('quantile'),
                input_core_dims=[('T',), (), (), ()],
                vectorize=True,
                dask='parallelized',
                output_core_dims=[('probability', 'T')],
                output_dtypes=['float'],
                dask_gufunc_kwargs={'output_sizes': {'probability': 3}, 'allow_rechunk': True}
            )
        elif self.dist_method == "nonparam":
            calc_func = self.calculate_tercile_probabilities_nonparametric
            error_samples = Predictant_no_m - hindcast_det_cross
            error_samples = error_samples.rename({'T': 'S'})
            hindcast_prob = xr.apply_ufunc(
                calc_func,
                result_da,
                error_samples,
                terciles.isel(quantile=0).drop_vars('quantile'),
                terciles.isel(quantile=1).drop_vars('quantile'),
                input_core_dims=[('T',), ('S',), (), ()],
                output_core_dims=[('probability', 'T')],
                vectorize=True,
                dask='parallelized',
                output_dtypes=[float],
                dask_gufunc_kwargs={'output_sizes': {'probability': 3}, 'allow_rechunk': True}
            )
        else:
            raise ValueError(f"Invalid dist_method: {self.dist_method}.")
        hindcast_prob = hindcast_prob.assign_coords(probability=('probability', ['PB', 'PN', 'PA']))
        return result_da * mask, mask * hindcast_prob.transpose('probability', 'T', 'Y', 'X')


class WAS_mme_StackXGboost_hpELM_:
    """
    Stacking ensemble with XGBoost as base model and HPELM as meta-model for Multi-Model Ensemble (MME) forecasting.
    This class implements a stacking ensemble using XGBRegressor as the base model,
    with an HPELM meta-model, for deterministic forecasting and optional tercile probability calculations.
    Implements hyperparameter optimization via randomized search for both base and meta models.

    Parameters
    ----------
    n_estimators_range : list of int, optional
        List of n_estimators values to tune for XGBoost (default is [50, 100, 200, 300]).
    learning_rate_range : list of float, optional
        List of learning rates to tune for XGBoost (default is [0.01, 0.05, 0.1, 0.2]).
    max_depth_range : list of int, optional
        List of max depths to tune for XGBoost (default is [3, 5, 7, 9]).
    min_child_weight_range : list of float, optional
        List of minimum child weights to tune for XGBoost (default is [1, 3, 5]).
    subsample_range : list of float, optional
        List of subsample ratios to tune for XGBoost (default is [0.6, 0.8, 1.0]).
    colsample_bytree_range : list of float, optional
        List of column sampling ratios to tune for XGBoost (default is [0.6, 0.8, 1.0]).
    neurons_range : list of int, optional
        List of neuron counts to tune for HPELM (default is [10, 20, 50, 100]).
    activation_options : list of str, optional
        Activation functions to tune for HPELM (default is ['sigm', 'tanh', 'relu', 'rbf_linf', 'rbf_gauss']).
    norm_range : list of float, optional
        Regularization parameters to tune for HPELM (default is [0.1, 1.0, 10.0, 100.0]).
    random_state : int, optional
        Seed for reproducibility (default is 42).
    dist_method : str, optional
        Distribution method for tercile probabilities ('t', 'gamma', etc.) (default is 'gamma').
    n_iter_search : int, optional
        Number of iterations for randomized search (default is 10).
    cv_folds : int, optional
        Number of cross-validation folds (default is 3).
    """
    def __init__(self,
                 n_estimators_range=[50, 100, 200, 300],
                 learning_rate_range=[0.01, 0.05, 0.1, 0.2],
                 max_depth_range=[3, 5, 7, 9],
                 min_child_weight_range=[1, 3, 5],
                 subsample_range=[0.6, 0.8, 1.0],
                 colsample_bytree_range=[0.6, 0.8, 1.0],
                 neurons_range=[10, 20, 50, 100],
                 activation_options=['sigm', 'tanh', 'relu', 'rbf_linf', 'rbf_gauss'],
                 norm_range=[0.1, 1.0, 10.0, 100.0],
                 random_state=42,
                 dist_method="gamma",
                 n_iter_search=10,
                 cv_folds=3):
        self.n_estimators_range = n_estimators_range
        self.learning_rate_range = learning_rate_range
        self.max_depth_range = max_depth_range
        self.min_child_weight_range = min_child_weight_range
        self.subsample_range = subsample_range
        self.colsample_bytree_range = colsample_bytree_range
        self.neurons_range = neurons_range
        self.activation_options = activation_options
        self.norm_range = norm_range
        self.random_state = random_state
        self.dist_method = dist_method
        self.n_iter_search = n_iter_search
        self.cv_folds = cv_folds
        self.stacking_model = None


    
    def compute_hyperparameters(self, Predictors, Predictand, clim_year_start, clim_year_end):
        """
        Independently computes the best hyperparameters using randomized search on stacked training data.

        Parameters
        ----------
        Predictors : xarray.DataArray
            Training predictor data with dimensions (T, M, Y, X).
        Predictand : xarray.DataArray
            Training predictand data with dimensions (T, Y, X).
        clim_year_start : int
            Start year of the climatology period.
        clim_year_end : int
            End year of the climatology period.

        Returns
        -------
        dict
            Best hyperparameters found.
        """
        X_train = standardize_timeseries(Predictors, clim_year_start, clim_year_end)
        y_train = standardize_timeseries(Predictand, clim_year_start, clim_year_end)

        # Stack training data
        X_train_stacked = X_train.stack(sample=('T', 'Y', 'X')).transpose('sample', 'M').values
        y_train_stacked = y_train.stack(sample=('T', 'Y', 'X')).values.ravel()
        train_nan_mask = np.any(~np.isfinite(X_train_stacked), axis=1) | ~np.isfinite(y_train_stacked)
        X_train_clean = X_train_stacked[~train_nan_mask]
        y_train_clean = y_train_stacked[~train_nan_mask]

        # For XGBoost (base model)
        param_dist_xgb = {
            'n_estimators': self.n_estimators_range,
            'learning_rate': self.learning_rate_range,
            'max_depth': self.max_depth_range,
            'min_child_weight': self.min_child_weight_range,
            'subsample': self.subsample_range,
            'colsample_bytree': self.colsample_bytree_range
        }
        model_xgb = XGBRegressor(random_state=self.random_state, verbosity=0)
        random_search_xgb = RandomizedSearchCV(
            model_xgb, param_distributions=param_dist_xgb, n_iter=self.n_iter_search,
            cv=self.cv_folds, scoring='neg_mean_squared_error',
            random_state=self.random_state, error_score=np.nan
        )
        random_search_xgb.fit(X_train_clean, y_train_clean)
        best_xgb = random_search_xgb.best_params_

        # For HPELM (meta model)
        param_dist_hpelm = {
            'neurons': self.neurons_range,
            'activation': self.activation_options,
            'norm': self.norm_range
        }
        model_hpelm = HPELMWrapper(random_state=self.random_state)
        random_search_hpelm = RandomizedSearchCV(
            model_hpelm, param_distributions=param_dist_hpelm, n_iter=self.n_iter_search,
            cv=self.cv_folds, scoring='neg_mean_squared_error',
            random_state=self.random_state, error_score=np.nan
        )
        random_search_hpelm.fit(X_train_clean, y_train_clean)
        best_hpelm = random_search_hpelm.best_params_

        return {'xgb': best_xgb, 'hpelm': best_hpelm}

    def compute_model(self, X_train, y_train, X_test, y_test, best_params=None):
        """
        Compute deterministic hindcast using the stacking ensemble of XGBoost (base) and HPELM (meta).

        Parameters
        ----------
        X_train : xarray.DataArray
            Training predictor data with dimensions (T, M, Y, X).
        y_train : xarray.DataArray
            Training predictand data with dimensions (T, Y, X).
        X_test : xarray.DataArray
            Testing predictor data with dimensions (T, M, Y, X).
        y_test : xarray.DataArray
            Testing predictand data with dimensions (T, Y, X).
        best_params : dict, optional
            Pre-computed best hyperparameters for 'xgb' and 'hpelm'. If None, computes internally.

        Returns
        -------
        predicted_da : xarray.DataArray
            Deterministic hindcast with dimensions (T, Y, X).
        """
        # Extract coordinate variables from X_test
        time = X_test['T']
        lat = X_test['Y']
        lon = X_test['X']
        n_time = len(X_test.coords['T'])
        n_lat = len(X_test.coords['Y'])
        n_lon = len(X_test.coords['X'])

        # Stack training data
        X_train_stacked = X_train.stack(sample=('T', 'Y', 'X')).transpose('sample', 'M').values
        y_train_stacked = y_train.stack(sample=('T', 'Y', 'X')).transpose('sample', 'M').values
        train_nan_mask = np.any(~np.isfinite(X_train_stacked), axis=1) | np.any(~np.isfinite(y_train_stacked), axis=1)
        X_train_clean = X_train_stacked[~train_nan_mask]
        y_train_clean = y_train_stacked[~train_nan_mask]

        # Stack testing data
        X_test_stacked = X_test.stack(sample=('T', 'Y', 'X')).transpose('sample', 'M').values
        y_test_stacked = y_test.stack(sample=('T', 'Y', 'X')).transpose('sample', 'M').values
        test_nan_mask = np.any(~np.isfinite(X_test_stacked), axis=1) | np.any(~np.isfinite(y_test_stacked), axis=1)

        # Use provided best_params or compute if None
        if best_params is None:
            best_params = self.compute_hyperparameters(X_train_clean, y_train_clean)

        # Initialize base model (XGBoost)
        xgb = XGBRegressor(**best_params['xgb'], random_state=self.random_state, verbosity=0)
        base_models = [('xgb', xgb)]

        # Initialize meta-model (HPELM)
        meta_model = HPELMWrapper(**best_params['hpelm'], random_state=self.random_state)

        # Initialize stacking ensemble
        self.stacking_model = StackingRegressor(estimators=base_models, final_estimator=meta_model)

        # Fit the stacking model on clean training data
        self.stacking_model.fit(X_train_clean, y_train_clean)

        # Predict on clean test data
        y_pred_test = self.stacking_model.predict(X_test_stacked[~test_nan_mask])

        # Reconstruct predictions
        result = np.empty_like(np.squeeze(y_test_stacked))
        result[np.squeeze(test_nan_mask)] = np.squeeze(y_test_stacked[test_nan_mask])
        result[~np.squeeze(test_nan_mask)] = y_pred_test

        predictions_reshaped = result.reshape(n_time, n_lat, n_lon)
        predicted_da = xr.DataArray(
            data=predictions_reshaped,
            coords={'T': time, 'Y': lat, 'X': lon},
            dims=['T', 'Y', 'X']
        )
        return predicted_da

    # ------------------ Probability Calculation Methods ------------------
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
    def calculate_tercile_probabilities_weibull_min(best_guess, error_variance, first_tercile, second_tercile, dof):
        """
        Weibull minimum-based method.
        """
        n_time = len(best_guess)
        pred_prob = np.empty((3, n_time))
        if np.all(np.isnan(best_guess)):
            pred_prob[:] = np.nan
        else:
            error_std = np.sqrt(error_variance)
            pred_prob[0, :] = stats.weibull_min.cdf(first_tercile, c=dof, loc=best_guess, scale=error_std)
            pred_prob[1, :] = stats.weibull_min.cdf(second_tercile, c=dof, loc=best_guess, scale=error_std) - \
                              stats.weibull_min.cdf(first_tercile, c=dof, loc=best_guess, scale=error_std)
            pred_prob[2, :] = 1 - stats.weibull_min.cdf(second_tercile, c=dof, loc=best_guess, scale=error_std)
        return pred_prob

    @staticmethod
    def calculate_tercile_probabilities_gamma(best_guess, error_variance, T1, T2, dof=None):
        """Gamma-distribution based method."""
        n_time = len(best_guess)
        pred_prob = np.empty((3, n_time), dtype=float)
        if np.any(np.isnan(best_guess)) or np.any(np.isnan(error_variance)):
            pred_prob[:] = np.nan
            return pred_prob
        best_guess = np.asarray(best_guess, dtype=float)
        error_variance = np.asarray(error_variance, dtype=float)
        T1 = np.asarray(T1, dtype=float)
        T2 = np.asarray(T2, dtype=float)
        alpha = (best_guess ** 2) / error_variance
        theta = error_variance / best_guess
        cdf_t1 = gamma.cdf(T1, a=alpha, scale=theta)
        cdf_t2 = gamma.cdf(T2, a=alpha, scale=theta)
        pred_prob[0, :] = cdf_t1
        pred_prob[1, :] = cdf_t2 - cdf_t1
        pred_prob[2, :] = 1.0 - cdf_t2
        return pred_prob

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

    @staticmethod
    def calculate_tercile_probabilities_normal(best_guess, error_variance, first_tercile, second_tercile):
        """Normal-distribution based method."""
        n_time = len(best_guess)
        pred_prob = np.empty((3, n_time))
        if np.all(np.isnan(best_guess)):
            pred_prob[:] = np.nan
        else:
            error_std = np.sqrt(error_variance)
            pred_prob[0, :] = norm.cdf(first_tercile, loc=best_guess, scale=error_std)
            pred_prob[1, :] = norm.cdf(second_tercile, loc=best_guess, scale=error_std) - norm.cdf(first_tercile, loc=best_guess, scale=error_std)
            pred_prob[2, :] = 1 - norm.cdf(second_tercile, loc=best_guess, scale=error_std)
        return pred_prob

    @staticmethod
    def calculate_tercile_probabilities_lognormal(best_guess, error_variance, first_tercile, second_tercile):
        """Lognormal-distribution based method."""
        n_time = len(best_guess)
        pred_prob = np.empty((3, n_time))
        if np.any(np.isnan(best_guess)) or np.any(np.isnan(error_variance)):
            pred_prob[:] = np.nan
            return pred_prob
        sigma = np.sqrt(np.log(1 + error_variance / (best_guess ** 2)))
        mu = np.log(best_guess) - sigma ** 2 / 2
        pred_prob[0, :] = lognorm.cdf(first_tercile, s=sigma, scale=np.exp(mu))
        pred_prob[1, :] = lognorm.cdf(second_tercile, s=sigma, scale=np.exp(mu)) - lognorm.cdf(first_tercile, s=sigma, scale=np.exp(mu))
        pred_prob[2, :] = 1 - lognorm.cdf(second_tercile, s=sigma, scale=np.exp(mu))
        return pred_prob

    def compute_prob(self, Predictant, clim_year_start, clim_year_end, hindcast_det):
        """
        Compute tercile probabilities using the chosen distribution method.
        Predictant is expected to be an xarray DataArray with dims (T, Y, X).
        """
        if "M" in Predictant.coords:
            Predictant = Predictant.isel(M=0).drop_vars('M').squeeze()
        mask = xr.where(~np.isnan(Predictant.isel(T=0)), 1, np.nan).drop_vars(['T']).squeeze().to_numpy()

        Predictant = Predictant.transpose('T', 'Y', 'X')
        index_start = Predictant.get_index("T").get_loc(str(clim_year_start)).start
        index_end = Predictant.get_index("T").get_loc(str(clim_year_end)).stop
        rainfall_for_tercile = Predictant.isel(T=slice(index_start, index_end))
        terciles = rainfall_for_tercile.quantile([0.33, 0.67], dim='T')
        error_variance = (Predictant - hindcast_det).var(dim='T')
        dof = len(Predictant.get_index("T")) - 2

        if self.dist_method == "t":
            calc_func = self.calculate_tercile_probabilities
            hindcast_prob = xr.apply_ufunc(
                calc_func,
                hindcast_det,
                error_variance,
                terciles.isel(quantile=0).drop_vars('quantile'),
                terciles.isel(quantile=1).drop_vars('quantile'),
                input_core_dims=[('T',), (), (), ()],
                vectorize=True,
                kwargs={'dof': dof},
                dask='parallelized',
                output_core_dims=[('probability', 'T')],
                output_dtypes=['float'],
                dask_gufunc_kwargs={'output_sizes': {'probability': 3}, 'allow_rechunk': True}
            )
        elif self.dist_method == "weibull_min":
            calc_func = self.calculate_tercile_probabilities_weibull_min
            hindcast_prob = xr.apply_ufunc(
                calc_func,
                hindcast_det,
                error_variance,
                terciles.isel(quantile=0).drop_vars('quantile'),
                terciles.isel(quantile=1).drop_vars('quantile'),
                input_core_dims=[('T',), (), (), ()],
                vectorize=True,
                kwargs={'dof': dof},
                dask='parallelized',
                output_core_dims=[('probability', 'T')],
                output_dtypes=['float'],
                dask_gufunc_kwargs={'output_sizes': {'probability': 3}, 'allow_rechunk': True}
            )
        elif self.dist_method == "gamma":
            calc_func = self.calculate_tercile_probabilities_gamma
            hindcast_prob = xr.apply_ufunc(
                calc_func,
                hindcast_det,
                error_variance,
                terciles.isel(quantile=0).drop_vars('quantile'),
                terciles.isel(quantile=1).drop_vars('quantile'),
                input_core_dims=[('T',), (), (), ()],
                vectorize=True,
                dask='parallelized',
                output_core_dims=[('probability', 'T')],
                output_dtypes=['float'],
                dask_gufunc_kwargs={'output_sizes': {'probability': 3}, 'allow_rechunk': True}
            )
        elif self.dist_method == "nonparam":
            calc_func = self.calculate_tercile_probabilities_nonparametric
            error_samples = (Predictant - hindcast_det)
            hindcast_prob = xr.apply_ufunc(
                calc_func,
                hindcast_det,
                error_samples,
                terciles.isel(quantile=0).drop_vars('quantile'),
                terciles.isel(quantile=1).drop_vars('quantile'),
                input_core_dims=[('T',), ('T',), (), ()],
                output_core_dims=[('probability', 'T')],
                vectorize=True,
                dask='parallelized',
                output_dtypes=[float],
                dask_gufunc_kwargs={'output_sizes': {'probability': 3}, 'allow_rechunk': True}
            )
        elif self.dist_method == "normal":
            calc_func = self.calculate_tercile_probabilities_normal
            hindcast_prob = xr.apply_ufunc(
                calc_func,
                hindcast_det,
                error_variance,
                terciles.isel(quantile=0).drop_vars('quantile'),
                terciles.isel(quantile=1).drop_vars('quantile'),
                input_core_dims=[('T',), (), (), ()],
                vectorize=True,
                dask='parallelized',
                output_core_dims=[('probability', 'T')],
                output_dtypes=['float'],
                dask_gufunc_kwargs={'output_sizes': {'probability': 3}, 'allow_rechunk': True}
            )
        elif self.dist_method == "lognormal":
            calc_func = self.calculate_tercile_probabilities_lognormal
            hindcast_prob = xr.apply_ufunc(
                calc_func,
                hindcast_det,
                error_variance,
                terciles.isel(quantile=0).drop_vars('quantile'),
                terciles.isel(quantile=1).drop_vars('quantile'),
                input_core_dims=[('T',), (), (), ()],
                vectorize=True,
                dask='parallelized',
                output_core_dims=[('probability', 'T')],
                output_dtypes=['float'],
                dask_gufunc_kwargs={'output_sizes': {'probability': 3}, 'allow_rechunk': True}
            )
        else:
            raise ValueError(f"Invalid dist_method: {self.dist_method}.")

        hindcast_prob = hindcast_prob.assign_coords(probability=('probability', ['PB', 'PN', 'PA']))
        return hindcast_prob.transpose('probability', 'T', 'Y', 'X')

    @staticmethod
    def _reshape_and_filter_data(da):
        """
        Helper: stack the DataArray from (T, Y, X[, M]) to (n_samples, n_features)
        and remove rows containing NaNs.
        """
        da_stacked = da.stack(sample=('T', 'Y', 'X'))
        if 'M' in da.dims:
            da_stacked = da_stacked.transpose('sample', 'M')
        else:
            da_stacked = da_stacked.transpose('sample')
        da_values = da_stacked.values
        nan_mask = np.any(np.isnan(da_values), axis=1)
        return da_values[~nan_mask], nan_mask, da_values

    def forecast(self, Predictant, clim_year_start, clim_year_end, hindcast_det, hindcast_det_cross, Predictor_for_year, best_params=None):
        """
        Generate deterministic and probabilistic forecast for a target year using the stacking ensemble.

        Parameters
        ----------
        Predictant : xarray.DataArray
            Observed predictand data with dimensions (T, Y, X) or (T, M, Y, X).
        clim_year_start : int or str
            Start year of the climatology period.
        clim_year_end : int or str
            End year of the climatology period.
        hindcast_det : xarray.DataArray
            Deterministic hindcast data for training with dimensions (T, M, Y, X).
        hindcast_det_cross : xarray.DataArray
            Deterministic hindcast data for error estimation with dimensions (T, Y, X).
        Predictor_for_year : xarray.DataArray
            Predictor data for the target year with dimensions (T, M, Y, X).
        best_params : dict, optional
            Pre-computed best hyperparameters for 'xgb' and 'hpelm'. If None, computes internally.

        Returns
        -------
        forecast_det : xarray.DataArray
            Deterministic forecast with dimensions (T, Y, X).
        forecast_prob : xarray.DataArray
            Tercile probabilities with dimensions (probability, T, Y, X).
        """
        if "M" in Predictant.coords:
            Predictant_no_m = Predictant.isel(M=0).drop_vars('M').squeeze()
        else:
            Predictant_no_m = Predictant

        mask = xr.where(~np.isnan(Predictant_no_m.isel(T=0)), 1, np.nan).drop_vars(['T']).squeeze().to_numpy()

        # Standardize Predictor_for_year using hindcast climatology
        mean_val = hindcast_det.sel(T=slice(str(clim_year_start), str(clim_year_end))).mean(dim='T')
        std_val = hindcast_det.sel(T=slice(str(clim_year_start), str(clim_year_end))).std(dim='T')
        Predictor_for_year_st = (Predictor_for_year - mean_val) / std_val

        hindcast_det_st = standardize_timeseries(hindcast_det, clim_year_start, clim_year_end)
        Predictant_st = standardize_timeseries(Predictant_no_m, clim_year_start, clim_year_end)
        y_test = Predictant_st.isel(T=[-1])

        # Extract coordinates from X_test
        time = Predictor_for_year_st['T']
        lat = Predictor_for_year_st['Y']
        lon = Predictor_for_year_st['X']
        n_time = len(Predictor_for_year_st.coords['T'])
        n_lat = len(Predictor_for_year_st.coords['Y'])
        n_lon = len(Predictor_for_year_st.coords['X'])

        # Stack training data and remove rows with NaNs
        X_train_stacked = hindcast_det_st.stack(sample=('T', 'Y', 'X')).transpose('sample', 'M').values
        y_train_stacked = Predictant_st.stack(sample=('T', 'Y', 'X')).values.ravel()
        train_nan_mask = np.any(~np.isfinite(X_train_stacked), axis=1) | ~np.isfinite(y_train_stacked)
        X_train_clean = X_train_stacked[~train_nan_mask]
        y_train_clean = y_train_stacked[~train_nan_mask]

        # Stack testing data
        X_test_stacked = Predictor_for_year_st.stack(sample=('T', 'Y', 'X')).transpose('sample', 'M').values
        y_test_stacked = y_test.stack(sample=('T', 'Y', 'X')).values.ravel()
        test_nan_mask = np.any(~np.isfinite(X_test_stacked), axis=1) | ~np.isfinite(y_test_stacked)

        # Use provided best_params or compute if None
        if best_params is None:
            best_params = self.compute_hyperparameters(X_train_clean, y_train_clean)

        # Initialize base model (XGBoost)
        xgb = XGBRegressor(**best_params['xgb'], random_state=self.random_state, verbosity=0)
        base_models = [('xgb', xgb)]

        # Initialize meta-model (HPELM)
        meta_model = HPELMWrapper(**best_params['hpelm'], random_state=self.random_state)

        # Initialize stacking ensemble
        self.stacking_model = StackingRegressor(estimators=base_models, final_estimator=meta_model)

        # Fit the stacking model on clean training data
        self.stacking_model.fit(X_train_clean, y_train_clean)

        # Predict on clean test data
        y_pred = self.stacking_model.predict(X_test_stacked[~test_nan_mask])

        # Reconstruct the prediction array
        result = np.empty_like(np.squeeze(y_test_stacked))
        result[np.squeeze(test_nan_mask)] = np.squeeze(y_test_stacked[test_nan_mask])
        result[~np.squeeze(test_nan_mask)] = y_pred

        predictions_reshaped = result.reshape(n_time, n_lat, n_lon)
        result_da = xr.DataArray(data=predictions_reshaped,
                                 coords={'T': time, 'Y': lat, 'X': lon},
                                 dims=['T', 'Y', 'X']) * mask

        result_da = reverse_standardize(result_da, Predictant_no_m, clim_year_start, clim_year_end)
        year = Predictor_for_year.coords['T'].values.astype('datetime64[Y]').astype(int)[0] + 1970
        T_value_1 = Predictant_no_m.isel(T=0).coords['T'].values
        month_1 = T_value_1.astype('datetime64[M]').astype(int) % 12 + 1
        new_T_value = np.datetime64(f"{year}-{month_1:02d}-01")
        result_da = result_da.assign_coords(T=xr.DataArray([new_T_value], dims=["T"]))
        result_da['T'] = result_da['T'].astype('datetime64[ns]')

        # Compute tercile probabilities
        index_start = Predictant_no_m.get_index("T").get_loc(str(clim_year_start)).start
        index_end = Predictant_no_m.get_index("T").get_loc(str(clim_year_end)).stop
        rainfall_for_tercile = Predictant_no_m.isel(T=slice(index_start, index_end))
        terciles = rainfall_for_tercile.quantile([0.33, 0.67], dim='T')
        error_variance = (Predictant_no_m - hindcast_det_cross).var(dim='T')
        dof = len(Predictant_no_m.get_index("T")) - 2

        if self.dist_method == "t":
            calc_func = self.calculate_tercile_probabilities
            hindcast_prob = xr.apply_ufunc(
                calc_func,
                result_da,
                error_variance,
                terciles.isel(quantile=0).drop_vars('quantile'),
                terciles.isel(quantile=1).drop_vars('quantile'),
                input_core_dims=[('T',), (), (), ()],
                vectorize=True,
                kwargs={'dof': dof},
                dask='parallelized',
                output_core_dims=[('probability','T')],
                output_dtypes=['float'],
                dask_gufunc_kwargs={'output_sizes': {'probability': 3}, 'allow_rechunk': True}
            )
        elif self.dist_method == "weibull_min":
            calc_func = self.calculate_tercile_probabilities_weibull_min
            hindcast_prob = xr.apply_ufunc(
                calc_func,
                result_da,
                error_variance,
                terciles.isel(quantile=0).drop_vars('quantile'),
                terciles.isel(quantile=1).drop_vars('quantile'),
                input_core_dims=[('T',), (), (), ()],
                vectorize=True,
                kwargs={'dof': dof},
                dask='parallelized',
                output_core_dims=[('probability', 'T')],
                output_dtypes=['float'],
                dask_gufunc_kwargs={'output_sizes': {'probability': 3}, 'allow_rechunk': True}
            )
        elif self.dist_method == "gamma":
            calc_func = self.calculate_tercile_probabilities_gamma
            hindcast_prob = xr.apply_ufunc(
                calc_func,
                result_da,
                error_variance,
                terciles.isel(quantile=0).drop_vars('quantile'),
                terciles.isel(quantile=1).drop_vars('quantile'),
                input_core_dims=[('T',), (), (), ()],
                vectorize=True,
                dask='parallelized',
                output_core_dims=[('probability','T')],
                output_dtypes=['float'],
                dask_gufunc_kwargs={'output_sizes': {'probability': 3}, 'allow_rechunk': True}
            )
        elif self.dist_method == "normal":
            calc_func = self.calculate_tercile_probabilities_normal
            hindcast_prob = xr.apply_ufunc(
                calc_func,
                result_da,
                error_variance,
                terciles.isel(quantile=0).drop_vars('quantile'),
                terciles.isel(quantile=1).drop_vars('quantile'),
                input_core_dims=[('T',), (), (), ()],
                vectorize=True,
                dask='parallelized',
                output_core_dims=[('probability','T')],
                output_dtypes=['float'],
                dask_gufunc_kwargs={'output_sizes': {'probability': 3}, 'allow_rechunk': True}
            )
        elif self.dist_method == "lognormal":
            calc_func = self.calculate_tercile_probabilities_lognormal
            hindcast_prob = xr.apply_ufunc(
                calc_func,
                result_da,
                error_variance,
                terciles.isel(quantile=0).drop_vars('quantile'),
                terciles.isel(quantile=1).drop_vars('quantile'),
                input_core_dims=[('T',), (), (), ()],
                vectorize=True,
                dask='parallelized',
                output_core_dims=[('probability','T')],
                output_dtypes=['float'],
                dask_gufunc_kwargs={'output_sizes': {'probability': 3}, 'allow_rechunk': True}
            )
        elif self.dist_method == "nonparam":
            calc_func = self.calculate_tercile_probabilities_nonparametric
            error_samples = Predictant_no_m - hindcast_det_cross
            error_samples = error_samples.rename({'T': 'S'})
            hindcast_prob = xr.apply_ufunc(
                calc_func,
                result_da,
                error_samples,
                terciles.isel(quantile=0).drop_vars('quantile'),
                terciles.isel(quantile=1).drop_vars('quantile'),
                input_core_dims=[('T',), ('S',), (), ()],
                output_core_dims=[('probability','T')],
                vectorize=True,
                dask='parallelized',
                output_dtypes=[float],
                dask_gufunc_kwargs={'output_sizes': {'probability': 3}, 'allow_rechunk': True}
            )
        else:
            raise ValueError(f"Invalid dist_method: {self.dist_method}.")

        hindcast_prob = hindcast_prob.assign_coords(probability=('probability', ['PB', 'PN', 'PA']))
        return result_da * mask, mask * hindcast_prob.transpose('probability', 'T', 'Y', 'X')



class WAS_mme_StackXGboost_hpELM:
    """
    Stacking ensemble with XGBoost as base model and HPELM as meta-model for Multi-Model Ensemble (MME) forecasting.
    This class implements a stacking ensemble using XGBRegressor as the base model,
    with an HPELM meta-model, for deterministic forecasting and optional tercile probability calculations.
    Implements hyperparameter optimization via randomized search for both base and meta models.
    Parameters
    ----------
    n_estimators_range : list of int, optional
        List of n_estimators values to tune for XGBoost (default is [50, 100, 200, 300]).
    learning_rate_range : list of float, optional
        List of learning rates to tune for XGBoost (default is [0.01, 0.05, 0.1, 0.2]).
    max_depth_range : list of int, optional
        List of max depths to tune for XGBoost (default is [3, 5, 7, 9]).
    min_child_weight_range : list of float, optional
        List of minimum child weights to tune for XGBoost (default is [1, 3, 5]).
    subsample_range : list of float, optional
        List of subsample ratios to tune for XGBoost (default is [0.6, 0.8, 1.0]).
    colsample_bytree_range : list of float, optional
        List of column sampling ratios to tune for XGBoost (default is [0.6, 0.8, 1.0]).
    neurons_range : list of int, optional
        List of neuron counts to tune for HPELM (default is [10, 20, 50, 100]).
    activation_options : list of str, optional
        Activation functions to tune for HPELM (default is ['sigm', 'tanh', 'relu', 'rbf_linf', 'rbf_gauss']).
    norm_range : list of float, optional
        Regularization parameters to tune for HPELM (default is [0.1, 1.0, 10.0, 100.0]).
    random_state : int, optional
        Seed for reproducibility (default is 42).
    dist_method : str, optional
        Distribution method for tercile probabilities ('t', 'gamma', etc.) (default is 'gamma').
    n_iter_search : int, optional
        Number of iterations for randomized search (default is 10).
    cv_folds : int, optional
        Number of cross-validation folds (default is 3).
    n_clusters : int, optional
        Number of clusters for homogenized zones (default is 4).
    """
    def __init__(self,
                 n_estimators_range=[50, 100, 200, 300],
                 learning_rate_range=[0.01, 0.05, 0.1, 0.2],
                 max_depth_range=[3, 5, 7, 9],
                 min_child_weight_range=[1, 3, 5],
                 subsample_range=[0.6, 0.8, 1.0],
                 colsample_bytree_range=[0.6, 0.8, 1.0],
                 neurons_range=[10, 20, 50, 100],
                 activation_options=['sigm', 'tanh', 'relu', 'rbf_linf', 'rbf_gauss'],
                 norm_range=[0.1, 1.0, 10.0, 100.0],
                 random_state=42,
                 dist_method="gamma",
                 n_iter_search=10,
                 cv_folds=3,
                 n_clusters=4):
        self.n_estimators_range = n_estimators_range
        self.learning_rate_range = learning_rate_range
        self.max_depth_range = max_depth_range
        self.min_child_weight_range = min_child_weight_range
        self.subsample_range = subsample_range
        self.colsample_bytree_range = colsample_bytree_range
        self.neurons_range = neurons_range
        self.activation_options = activation_options
        self.norm_range = norm_range
        self.random_state = random_state
        self.dist_method = dist_method
        self.n_iter_search = n_iter_search
        self.cv_folds = cv_folds
        self.n_clusters = n_clusters
        self.stacking_model = None

    def compute_hyperparameters(self, Predictors, Predictand, clim_year_start, clim_year_end):
        """
        Independently computes the best hyperparameters using randomized search on stacked training data for each homogenized zone.
        Parameters
        ----------
        Predictors : xarray.DataArray
            Training predictor data with dimensions (T, M, Y, X).
        Predictand : xarray.DataArray
            Training predictand data with dimensions (T, Y, X).
        clim_year_start : int
            Start year of the climatology period.
        clim_year_end : int
            End year of the climatology period.
        Returns
        -------
        best_params_dict : dict
            Best hyperparameters for each cluster.
        cluster_da : xarray.DataArray
            Cluster labels with dimensions (Y, X).
        """
        if "M" in Predictand.coords:
            Predictand = Predictand.isel(M=0).drop_vars('M').squeeze()
           
        X_train_std = standardize_timeseries(Predictors, clim_year_start, clim_year_end)
        Predictand.name = "varname"
        # Step 1: Perform KMeans clustering based on predictand's spatial distribution
        kmeans = KMeans(n_clusters=self.n_clusters, random_state=self.random_state)
        Predictand_dropna = Predictand.to_dataframe().reset_index().dropna().drop(columns=['T'])
        variable_column = Predictand_dropna.columns[2]
        Predictand_dropna['cluster'] = kmeans.fit_predict(
           Predictand_dropna[[variable_column]]
        )
       
        # Convert cluster assignments back into an xarray structure
        df_unique = Predictand_dropna.drop_duplicates(subset=['Y', 'X'])
        dataset = df_unique.set_index(['Y', 'X']).to_xarray()
        mask = xr.where(~np.isnan(Predictand.isel(T=0)), 1, np.nan)
        Cluster = (dataset['cluster'] * mask)
              
        # Align cluster array with the predictand array
        xarray1, xarray2 = xr.align(Predictand, Cluster, join="outer")
       
        # Identify unique cluster labels
        clusters = np.unique(xarray2)
        clusters = clusters[~np.isnan(clusters)]
        cluster_da = xarray2
        y_train_std = standardize_timeseries(Predictand, clim_year_start, clim_year_end)
        X_train_std['T'] = y_train_std['T']
        # Prepare parameter distributions
        param_dist_xgb = {
            'n_estimators': self.n_estimators_range,
            'learning_rate': self.learning_rate_range,
            'max_depth': self.max_depth_range,
            'min_child_weight': self.min_child_weight_range,
            'subsample': self.subsample_range,
            'colsample_bytree': self.colsample_bytree_range
        }
        param_dist_hpelm = {
            'neurons': self.neurons_range,
            'activation': self.activation_options,
            'norm': self.norm_range
        }
        best_params_dict = {}
        for c in clusters:
            mask_3d = (cluster_da == c).expand_dims({'T': y_train_std['T']})
            X_stacked_c = X_train_std.where(mask_3d).stack(sample=('T', 'Y', 'X')).transpose('sample', 'M').values
            y_stacked_c = y_train_std.where(mask_3d).stack(sample=('T', 'Y', 'X')).values.ravel()
            nan_mask_c = np.any(~np.isfinite(X_stacked_c), axis=1) | ~np.isfinite(y_stacked_c)
            X_clean_c = X_stacked_c[~nan_mask_c]
            y_clean_c = y_stacked_c[~nan_mask_c]
            if len(X_clean_c) == 0:
                continue
            # Tune XGBoost
            model_xgb = XGBRegressor(random_state=self.random_state, verbosity=0, n_jobs=-1)
            random_search_xgb = RandomizedSearchCV(
                model_xgb, param_distributions=param_dist_xgb, n_iter=self.n_iter_search,
                cv=self.cv_folds, scoring='neg_mean_squared_error',
                random_state=self.random_state, error_score=np.nan
            )
            random_search_xgb.fit(X_clean_c, y_clean_c)
            best_xgb = random_search_xgb.best_params_
            # Tune HPELM
            model_hpelm = HPELMWrapper(random_state=self.random_state)
            random_search_hpelm = RandomizedSearchCV(
                model_hpelm, param_distributions=param_dist_hpelm, n_iter=self.n_iter_search,
                cv=self.cv_folds, scoring='neg_mean_squared_error',
                random_state=self.random_state, error_score=np.nan
            )
            random_search_hpelm.fit(X_clean_c, y_clean_c)
            best_hpelm = random_search_hpelm.best_params_
            best_params_dict[c] = {'xgb': best_xgb, 'hpelm': best_hpelm}
        return best_params_dict, cluster_da

    def compute_model(self, X_train, y_train, X_test, y_test, best_params=None, cluster_da=None):
        """
        Compute deterministic hindcast using the stacking ensemble of XGBoost (base) and HPELM (meta) for each zone.
        Parameters
        ----------
        X_train : xarray.DataArray
            Training predictor data with dimensions (T, M, Y, X).
        y_train : xarray.DataArray
            Training predictand data with dimensions (T, Y, X).
        X_test : xarray.DataArray
            Testing predictor data with dimensions (T, M, Y, X).
        y_test : xarray.DataArray
            Testing predictand data with dimensions (T, Y, X).
        clim_year_start : int
            Start year of the climatology period.
        clim_year_end : int
            End year of the climatology period.
        best_params : dict, optional
            Pre-computed best hyperparameters per cluster. If None, computes internally.
        cluster_da : xarray.DataArray, optional
            Pre-computed cluster labels. If None, computes internally.
        Returns
        -------
        predicted_da : xarray.DataArray
            Deterministic hindcast with dimensions (T, Y, X).
        """
        # Standardize inputs
        X_train_std = X_train
        y_train_std = y_train
        X_test_std = X_test
        y_test_std = y_test
        
        # Extract coordinate variables from X_test
        time = X_test_std['T']
        lat = X_test_std['Y']
        lon = X_test_std['X']
        
        n_time = len(time)
        n_lat = len(lat)
        n_lon = len(lon)
        # Use provided best_params and cluster_da or compute if None
        if best_params is None:
            best_params, cluster_da = self.compute_hyperparameters(X_train, y_train, clim_year_start, clim_year_end)
        # Initialize predictions array
        predictions = np.full((n_time, n_lat, n_lon), np.nan)
        self.stacking_model = {}  # Dictionary to store models per cluster
        for c in range(self.n_clusters):
            if c not in best_params:
                continue
            bp = best_params[c]
            # Mask for this cluster
            mask_3d_train = (cluster_da == c).expand_dims({'T': X_train_std['T']})
            mask_3d_test = (cluster_da == c).expand_dims({'T': X_test_std['T']})
            
            # Stack training data for cluster
            X_train_stacked_c = X_train_std.where(mask_3d_train).stack(sample=('T', 'Y', 'X')).transpose('sample', 'M').values
            y_train_stacked_c = y_train_std.where(mask_3d_train).stack(sample=('T', 'Y', 'X')).values.ravel()
            train_nan_mask = np.any(~np.isfinite(X_train_stacked_c), axis=1) | ~np.isfinite(y_train_stacked_c)
            X_train_clean_c = X_train_stacked_c[~train_nan_mask]
            y_train_clean_c = y_train_stacked_c[~train_nan_mask]
            # Stack testing data for cluster
            X_test_stacked_c = X_test_std.where(mask_3d_test).stack(sample=('T', 'Y', 'X')).transpose('sample', 'M').values
            y_test_stacked_c = y_test_std.where(mask_3d_test).stack(sample=('T', 'Y', 'X')).values.ravel()
            test_nan_mask = np.any(~np.isfinite(X_test_stacked_c), axis=1) | ~np.isfinite(y_test_stacked_c)
            X_test_clean_c = X_test_stacked_c[~test_nan_mask]
            
            # Initialize base model (XGBoost)
            xgb = XGBRegressor(**bp['xgb'], random_state=self.random_state, verbosity=0, n_jobs=-1)
            # Initialize meta-model (HPELMWrapper)
            meta_model = HPELMWrapper(**bp['hpelm'], random_state=self.random_state)
            # Initialize stacking ensemble for this cluster
            stacking_model_c = StackingRegressor(estimators=[('xgb', xgb)], final_estimator=meta_model)
            # Fit the stacking model on cluster data
            stacking_model_c.fit(X_train_clean_c, y_train_clean_c)
            self.stacking_model[c] = stacking_model_c
            # Predict
            y_pred_c = stacking_model_c.predict(X_test_clean_c)
            # Reconstruct predictions for this cluster
            result_c = np.full(len(y_test_stacked_c), np.nan)
            result_c[~test_nan_mask] = y_pred_c
            pred_c_reshaped = result_c.reshape(n_time, n_lat, n_lon)
            # Fill in the predictions array
            predictions = np.where(np.isnan(predictions), pred_c_reshaped, predictions)
        predicted_da = xr.DataArray(
            data=predictions,
            coords={'T': time, 'Y': lat, 'X': lon},
            dims=['T', 'Y', 'X']
        )

        return predicted_da

    # ------------------ Probability Calculation Methods ------------------
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
            pred_prob[0, :] = t.cdf(first_t, df=dof)
            pred_prob[1, :] = t.cdf(second_t, df=dof) - t.cdf(first_t, df=dof)
            pred_prob[2, :] = 1 - t.cdf(second_t, df=dof)
        return pred_prob

    @staticmethod
    def calculate_tercile_probabilities_weibull_min(best_guess, error_variance, first_tercile, second_tercile, dof):
        """
        Weibull minimum-based method.
        """
        n_time = len(best_guess)
        pred_prob = np.empty((3, n_time))
        if np.all(np.isnan(best_guess)):
            pred_prob[:] = np.nan
        else:
            error_std = np.sqrt(error_variance)
            pred_prob[0, :] = weibull_min.cdf(first_tercile, c=dof, loc=best_guess, scale=error_std)
            pred_prob[1, :] = weibull_min.cdf(second_tercile, c=dof, loc=best_guess, scale=error_std) - \
                              weibull_min.cdf(first_tercile, c=dof, loc=best_guess, scale=error_std)
            pred_prob[2, :] = 1 - weibull_min.cdf(second_tercile, c=dof, loc=best_guess, scale=error_std)
        return pred_prob

    @staticmethod
    def calculate_tercile_probabilities_gamma(best_guess, error_variance, T1, T2, dof=None):
        """Gamma-distribution based method."""
        n_time = len(best_guess)
        pred_prob = np.empty((3, n_time), dtype=float)
        if np.any(np.isnan(best_guess)) or np.any(np.isnan(error_variance)):
            pred_prob[:] = np.nan
            return pred_prob
        best_guess = np.asarray(best_guess, dtype=float)
        error_variance = np.asarray(error_variance, dtype=float)
        T1 = np.asarray(T1, dtype=float)
        T2 = np.asarray(T2, dtype=float)
        alpha = (best_guess ** 2) / error_variance
        theta = error_variance / best_guess
        cdf_t1 = gamma.cdf(T1, a=alpha, scale=theta)
        cdf_t2 = gamma.cdf(T2, a=alpha, scale=theta)
        pred_prob[0, :] = cdf_t1
        pred_prob[1, :] = cdf_t2 - cdf_t1
        pred_prob[2, :] = 1.0 - cdf_t2
        return pred_prob

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

    @staticmethod
    def calculate_tercile_probabilities_normal(best_guess, error_variance, first_tercile, second_tercile):
        """Normal-distribution based method."""
        n_time = len(best_guess)
        pred_prob = np.empty((3, n_time))
        if np.all(np.isnan(best_guess)):
            pred_prob[:] = np.nan
        else:
            error_std = np.sqrt(error_variance)
            pred_prob[0, :] = norm.cdf(first_tercile, loc=best_guess, scale=error_std)
            pred_prob[1, :] = norm.cdf(second_tercile, loc=best_guess, scale=error_std) - norm.cdf(first_tercile, loc=best_guess, scale=error_std)
            pred_prob[2, :] = 1 - norm.cdf(second_tercile, loc=best_guess, scale=error_std)
        return pred_prob

    @staticmethod
    def calculate_tercile_probabilities_lognormal(best_guess, error_variance, first_tercile, second_tercile):
        """Lognormal-distribution based method."""
        n_time = len(best_guess)
        pred_prob = np.empty((3, n_time))
        if np.any(np.isnan(best_guess)) or np.any(np.isnan(error_variance)):
            pred_prob[:] = np.nan
            return pred_prob
        sigma = np.sqrt(np.log(1 + error_variance / (best_guess ** 2)))
        mu = np.log(best_guess) - sigma ** 2 / 2
        pred_prob[0, :] = lognorm.cdf(first_tercile, s=sigma, scale=np.exp(mu))
        pred_prob[1, :] = lognorm.cdf(second_tercile, s=sigma, scale=np.exp(mu)) - lognorm.cdf(first_tercile, s=sigma, scale=np.exp(mu))
        pred_prob[2, :] = 1 - lognorm.cdf(second_tercile, s=sigma, scale=np.exp(mu))
        return pred_prob

    def compute_prob(self, Predictant, clim_year_start, clim_year_end, hindcast_det):
        """
        Compute tercile probabilities using the chosen distribution method.
        Predictant is expected to be an xarray DataArray with dims (T, Y, X).
        """
        if "M" in Predictant.coords:
            Predictant = Predictant.isel(M=0).drop_vars('M').squeeze()
        mask = xr.where(~np.isnan(Predictant.isel(T=0)), 1, np.nan).drop_vars(['T']).squeeze().to_numpy()
        Predictant = Predictant.transpose('T', 'Y', 'X')
        index_start = Predictant.get_index("T").get_loc(str(clim_year_start)).start
        index_end = Predictant.get_index("T").get_loc(str(clim_year_end)).stop
        rainfall_for_tercile = Predictant.isel(T=slice(index_start, index_end))
        terciles = rainfall_for_tercile.quantile([0.33, 0.67], dim='T')
        error_variance = (Predictant - hindcast_det).var(dim='T')
        dof = len(Predictant.get_index("T")) - 2
        if self.dist_method == "t":
            calc_func = self.calculate_tercile_probabilities
            hindcast_prob = xr.apply_ufunc(
                calc_func,
                hindcast_det,
                error_variance,
                terciles.isel(quantile=0).drop_vars('quantile'),
                terciles.isel(quantile=1).drop_vars('quantile'),
                input_core_dims=[('T',), (), (), ()],
                vectorize=True,
                kwargs={'dof': dof},
                dask='parallelized',
                output_core_dims=[('probability', 'T')],
                output_dtypes=['float'],
                dask_gufunc_kwargs={'output_sizes': {'probability': 3}, 'allow_rechunk': True}
            )
        elif self.dist_method == "weibull_min":
            calc_func = self.calculate_tercile_probabilities_weibull_min
            hindcast_prob = xr.apply_ufunc(
                calc_func,
                hindcast_det,
                error_variance,
                terciles.isel(quantile=0).drop_vars('quantile'),
                terciles.isel(quantile=1).drop_vars('quantile'),
                input_core_dims=[('T',), (), (), ()],
                vectorize=True,
                kwargs={'dof': dof},
                dask='parallelized',
                output_core_dims=[('probability', 'T')],
                output_dtypes=['float'],
                dask_gufunc_kwargs={'output_sizes': {'probability': 3}, 'allow_rechunk': True}
            )
        elif self.dist_method == "gamma":
            calc_func = self.calculate_tercile_probabilities_gamma
            hindcast_prob = xr.apply_ufunc(
                calc_func,
                hindcast_det,
                error_variance,
                terciles.isel(quantile=0).drop_vars('quantile'),
                terciles.isel(quantile=1).drop_vars('quantile'),
                input_core_dims=[('T',), (), (), ()],
                vectorize=True,
                dask='parallelized',
                output_core_dims=[('probability', 'T')],
                output_dtypes=['float'],
                dask_gufunc_kwargs={'output_sizes': {'probability': 3}, 'allow_rechunk': True}
            )
        elif self.dist_method == "nonparam":
            calc_func = self.calculate_tercile_probabilities_nonparametric
            error_samples = Predictant - hindcast_det
            hindcast_prob = xr.apply_ufunc(
                calc_func,
                hindcast_det,
                error_samples,
                terciles.isel(quantile=0).drop_vars('quantile'),
                terciles.isel(quantile=1).drop_vars('quantile'),
                input_core_dims=[('T',), ('T',), (), ()],
                output_core_dims=[('probability', 'T')],
                vectorize=True,
                dask='parallelized',
                output_dtypes=[float],
                dask_gufunc_kwargs={'output_sizes': {'probability': 3}, 'allow_rechunk': True}
            )
        elif self.dist_method == "normal":
            calc_func = self.calculate_tercile_probabilities_normal
            hindcast_prob = xr.apply_ufunc(
                calc_func,
                hindcast_det,
                error_variance,
                terciles.isel(quantile=0).drop_vars('quantile'),
                terciles.isel(quantile=1).drop_vars('quantile'),
                input_core_dims=[('T',), (), (), ()],
                vectorize=True,
                dask='parallelized',
                output_core_dims=[('probability', 'T')],
                output_dtypes=['float'],
                dask_gufunc_kwargs={'output_sizes': {'probability': 3}, 'allow_rechunk': True}
            )
        elif self.dist_method == "lognormal":
            calc_func = self.calculate_tercile_probabilities_lognormal
            hindcast_prob = xr.apply_ufunc(
                calc_func,
                hindcast_det,
                error_variance,
                terciles.isel(quantile=0).drop_vars('quantile'),
                terciles.isel(quantile=1).drop_vars('quantile'),
                input_core_dims=[('T',), (), (), ()],
                vectorize=True,
                dask='parallelized',
                output_core_dims=[('probability', 'T')],
                output_dtypes=['float'],
                dask_gufunc_kwargs={'output_sizes': {'probability': 3}, 'allow_rechunk': True}
            )
        else:
            raise ValueError(f"Invalid dist_method: {self.dist_method}.")
        hindcast_prob = hindcast_prob.assign_coords(probability=('probability', ['PB', 'PN', 'PA']))
        return hindcast_prob.transpose('probability', 'T', 'Y', 'X')

    @staticmethod
    def _reshape_and_filter_data(da):
        """
        Helper: stack the DataArray from (T, Y, X[, M]) to (n_samples, n_features)
        and remove rows containing NaNs.
        """
        da_stacked = da.stack(sample=('T', 'Y', 'X'))
        if 'M' in da.dims:
            da_stacked = da_stacked.transpose('sample', 'M')
        else:
            da_stacked = da_stacked.transpose('sample')
        da_values = da_stacked.values
        nan_mask = np.any(np.isnan(da_values), axis=1)
        return da_values[~nan_mask], nan_mask, da_values

    def forecast(self, Predictant, clim_year_start, clim_year_end, hindcast_det, hindcast_det_cross, Predictor_for_year, best_params=None, cluster_da=None):
        """
        Generate deterministic and probabilistic forecast for a target year using the stacking ensemble.
        Parameters
        ----------
        Predictant : xarray.DataArray
            Observed predictand data with dimensions (T, Y, X) or (T, M, Y, X).
        clim_year_start : int or str
            Start year of the climatology period.
        clim_year_end : int or str
            End year of the climatology period.
        hindcast_det : xarray.DataArray
            Deterministic hindcast data for training with dimensions (T, M, Y, X).
        hindcast_det_cross : xarray.DataArray
            Deterministic hindcast data for error estimation with dimensions (T, Y, X).
        Predictor_for_year : xarray.DataArray
            Predictor data for the target year with dimensions (T, M, Y, X).
        best_params : dict, optional
            Pre-computed best hyperparameters per cluster. If None, computes internally.
        cluster_da : xarray.DataArray, optional
            Pre-computed cluster labels. If None, computes internally.
        Returns
        -------
        forecast_det : xarray.DataArray
            Deterministic forecast with dimensions (T, Y, X).
        forecast_prob : xarray.DataArray
            Tercile probabilities with dimensions (probability, T, Y, X).
        """
        if "M" in Predictant.coords:
            Predictant_no_m = Predictant.isel(M=0).drop_vars('M').squeeze()
        else:
            Predictant_no_m = Predictant
        mask = xr.where(~np.isnan(Predictant_no_m.isel(T=0)), 1, np.nan).drop_vars(['T']).squeeze().to_numpy()
        # Standardize Predictor_for_year using hindcast climatology
        mean_val = hindcast_det.sel(T=slice(str(clim_year_start), str(clim_year_end))).mean(dim='T')
        std_val = hindcast_det.sel(T=slice(str(clim_year_start), str(clim_year_end))).std(dim='T')
        Predictor_for_year_st = (Predictor_for_year - mean_val) / std_val
        hindcast_det_st = standardize_timeseries(hindcast_det, clim_year_start, clim_year_end)
        Predictant_st = standardize_timeseries(Predictant_no_m, clim_year_start, clim_year_end)
        hindcast_det_st['T'] = Predictant_st['T']
        
        # Extract coordinates from X_test
        time = Predictor_for_year_st['T']
        lat = Predictor_for_year_st['Y']
        lon = Predictor_for_year_st['X']
        n_time = len(time)
        n_lat = len(lat)
        n_lon = len(lon)
        # Use provided best_params and cluster_da or compute if None
        if best_params is None:
            best_params, cluster_da = self.compute_hyperparameters(hindcast_det, Predictant_no_m, clim_year_start, clim_year_end)
        # Initialize predictions array
        predictions = np.full((n_time, n_lat, n_lon), np.nan)
        self.stacking_model = {}  # Dictionary to store models per cluster
        for c in range(self.n_clusters):
            if c not in best_params:
                continue
            bp = best_params[c]
            # Mask for this cluster
            mask_3d_train = (cluster_da == c).expand_dims({'T': hindcast_det_st['T']})
            mask_3d_test = (cluster_da == c).expand_dims({'T': Predictor_for_year_st['T']})
            # Stack training data for cluster
            X_train_stacked_c = hindcast_det_st.where(mask_3d_train).stack(sample=('T', 'Y', 'X')).transpose('sample', 'M').values
            y_train_stacked_c = Predictant_st.where(mask_3d_train).stack(sample=('T', 'Y', 'X')).values.ravel()
            train_nan_mask = np.any(~np.isfinite(X_train_stacked_c), axis=1) | ~np.isfinite(y_train_stacked_c)
            X_train_clean_c = X_train_stacked_c[~train_nan_mask]
            y_train_clean_c = y_train_stacked_c[~train_nan_mask]
            # Stack testing data for cluster
            X_test_stacked_c = Predictor_for_year_st.where(mask_3d_test).stack(sample=('T', 'Y', 'X')).transpose('sample', 'M').values
            test_nan_mask = np.any(~np.isfinite(X_test_stacked_c), axis=1)
            X_test_clean_c = X_test_stacked_c[~test_nan_mask]
            # Initialize base model (XGBoost)
            xgb = XGBRegressor(**bp['xgb'], random_state=self.random_state, verbosity=0)
            # Initialize meta-model (HPELMWrapper)
            meta_model = HPELMWrapper(**bp['hpelm'], random_state=self.random_state)
            # Initialize stacking ensemble for this cluster
            stacking_model_c = StackingRegressor(estimators=[('xgb', xgb)], final_estimator=meta_model)
            # Fit the stacking model on cluster data
            stacking_model_c.fit(X_train_clean_c, y_train_clean_c)
            self.stacking_model[c] = stacking_model_c
            # Predict
            y_pred_c = stacking_model_c.predict(X_test_clean_c)
            # Reconstruct predictions for this cluster
            result_c = np.full(len(X_test_stacked_c), np.nan)
            result_c[~test_nan_mask] = y_pred_c
            pred_c_reshaped = result_c.reshape(n_time, n_lat, n_lon)
            # Fill in the predictions array
            predictions = np.where(np.isnan(predictions), pred_c_reshaped, predictions)
        result_da = xr.DataArray(
            data=predictions,
            coords={'T': time, 'Y': lat, 'X': lon},
            dims=['T', 'Y', 'X']
        ) * mask
        result_da = reverse_standardize(result_da, Predictant_no_m, clim_year_start, clim_year_end)
        year = Predictor_for_year.coords['T'].values.astype('datetime64[Y]').astype(int)[0] + 1970
        T_value_1 = Predictant_no_m.isel(T=0).coords['T'].values
        month_1 = T_value_1.astype('datetime64[M]').astype(int) % 12 + 1
        new_T_value = np.datetime64(f"{year}-{month_1:02d}-01")
        result_da = result_da.assign_coords(T=xr.DataArray([new_T_value], dims=["T"]))
        result_da['T'] = result_da['T'].astype('datetime64[ns]')
        # Compute tercile probabilities
        index_start = Predictant_no_m.get_index("T").get_loc(str(clim_year_start)).start
        index_end = Predictant_no_m.get_index("T").get_loc(str(clim_year_end)).stop
        rainfall_for_tercile = Predictant_no_m.isel(T=slice(index_start, index_end))
        terciles = rainfall_for_tercile.quantile([0.33, 0.67], dim='T')
        error_variance = (Predictant_no_m - hindcast_det_cross).var(dim='T')
        dof = len(Predictant_no_m.get_index("T")) - 2
        if self.dist_method == "t":
            calc_func = self.calculate_tercile_probabilities
            hindcast_prob = xr.apply_ufunc(
                calc_func,
                result_da,
                error_variance,
                terciles.isel(quantile=0).drop_vars('quantile'),
                terciles.isel(quantile=1).drop_vars('quantile'),
                input_core_dims=[('T',), (), (), ()],
                vectorize=True,
                kwargs={'dof': dof},
                dask='parallelized',
                output_core_dims=[('probability', 'T')],
                output_dtypes=['float'],
                dask_gufunc_kwargs={'output_sizes': {'probability': 3}, 'allow_rechunk': True}
            )
        elif self.dist_method == "weibull_min":
            calc_func = self.calculate_tercile_probabilities_weibull_min
            hindcast_prob = xr.apply_ufunc(
                calc_func,
                result_da,
                error_variance,
                terciles.isel(quantile=0).drop_vars('quantile'),
                terciles.isel(quantile=1).drop_vars('quantile'),
                input_core_dims=[('T',), (), (), ()],
                vectorize=True,
                kwargs={'dof': dof},
                dask='parallelized',
                output_core_dims=[('probability', 'T')],
                output_dtypes=['float'],
                dask_gufunc_kwargs={'output_sizes': {'probability': 3}, 'allow_rechunk': True}
            )
        elif self.dist_method == "gamma":
            calc_func = self.calculate_tercile_probabilities_gamma
            hindcast_prob = xr.apply_ufunc(
                calc_func,
                result_da,
                error_variance,
                terciles.isel(quantile=0).drop_vars('quantile'),
                terciles.isel(quantile=1).drop_vars('quantile'),
                input_core_dims=[('T',), (), (), ()],
                vectorize=True,
                dask='parallelized',
                output_core_dims=[('probability', 'T')],
                output_dtypes=['float'],
                dask_gufunc_kwargs={'output_sizes': {'probability': 3}, 'allow_rechunk': True}
            )
        elif self.dist_method == "normal":
            calc_func = self.calculate_tercile_probabilities_normal
            hindcast_prob = xr.apply_ufunc(
                calc_func,
                result_da,
                error_variance,
                terciles.isel(quantile=0).drop_vars('quantile'),
                terciles.isel(quantile=1).drop_vars('quantile'),
                input_core_dims=[('T',), (), (), ()],
                vectorize=True,
                dask='parallelized',
                output_core_dims=[('probability', 'T')],
                output_dtypes=['float'],
                dask_gufunc_kwargs={'output_sizes': {'probability': 3}, 'allow_rechunk': True}
            )
        elif self.dist_method == "lognormal":
            calc_func = self.calculate_tercile_probabilities_lognormal
            hindcast_prob = xr.apply_ufunc(
                calc_func,
                result_da,
                error_variance,
                terciles.isel(quantile=0).drop_vars('quantile'),
                terciles.isel(quantile=1).drop_vars('quantile'),
                input_core_dims=[('T',), (), (), ()],
                vectorize=True,
                dask='parallelized',
                output_core_dims=[('probability', 'T')],
                output_dtypes=['float'],
                dask_gufunc_kwargs={'output_sizes': {'probability': 3}, 'allow_rechunk': True}
            )
        elif self.dist_method == "nonparam":
            calc_func = self.calculate_tercile_probabilities_nonparametric
            error_samples = Predictant_no_m - hindcast_det_cross
            error_samples = error_samples.rename({'T': 'S'})
            hindcast_prob = xr.apply_ufunc(
                calc_func,
                result_da,
                error_samples,
                terciles.isel(quantile=0).drop_vars('quantile'),
                terciles.isel(quantile=1).drop_vars('quantile'),
                input_core_dims=[('T',), ('S',), (), ()],
                output_core_dims=[('probability', 'T')],
                vectorize=True,
                dask='parallelized',
                output_dtypes=[float],
                dask_gufunc_kwargs={'output_sizes': {'probability': 3}, 'allow_rechunk': True}
            )
        else:
            raise ValueError(f"Invalid dist_method: {self.dist_method}.")
        hindcast_prob = hindcast_prob.assign_coords(probability=('probability', ['PB', 'PN', 'PA']))
        return result_da * mask, mask * hindcast_prob.transpose('probability', 'T', 'Y', 'X')



class WAS_mme_StackXGboost_MLP_:
    """
    Stacking ensemble with XGBoost base model and MLP meta-model for Multi-Model Ensemble (MME) forecasting.
    This class implements a stacking ensemble using XGBRegressor as the base model,
    with an MLPRegressor meta-model, for deterministic forecasting and optional tercile probability calculations.
    Implements hyperparameter optimization via randomized search for both base and meta models.

    Parameters
    ----------
    n_estimators_range : list of int, optional
        List of n_estimators values to tune for XGBoost (default is [50, 100, 200, 300]).
    learning_rate_range : list of float, optional
        List of learning rates to tune for XGBoost (default is [0.01, 0.05, 0.1, 0.2]).
    max_depth_range : list of int, optional
        List of max depths to tune for XGBoost (default is [3, 5, 7, 9]).
    min_child_weight_range : list of float, optional
        List of minimum child weights to tune for XGBoost (default is [1, 3, 5]).
    subsample_range : list of float, optional
        List of subsample ratios to tune for XGBoost (default is [0.6, 0.8, 1.0]).
    colsample_bytree_range : list of float, optional
        List of column sampling ratios to tune for XGBoost (default is [0.6, 0.8, 1.0]).
    hidden_layer_sizes_range : list of tuples, optional
        List of hidden layer sizes to tune for MLP (default is [(10,), (10, 5), (20, 10)]).
    activation_options : list of str, optional
        Activation functions to tune for MLP (default is ['relu', 'tanh', 'logistic']).
    solver_options : list of str, optional
        Solvers to tune for MLP (default is ['adam', 'sgd', 'lbfgs']).
    alpha_range : list of float, optional
        L2 regularization parameters to tune for MLP (default is [0.0001, 0.001, 0.01, 0.1]).
    max_iter : int, optional
        Maximum iterations for MLP (default is 200).
    random_state : int, optional
        Seed for reproducibility (default is 42).
    dist_method : str, optional
        Distribution method for tercile probabilities ('t', 'gamma', etc.) (default is 'gamma').
    n_iter_search : int, optional
        Number of iterations for randomized search (default is 10).
    cv_folds : int, optional
        Number of cross-validation folds (default is 3).
    """
    def __init__(self,
                 n_estimators_range=[50, 100, 200, 300],
                 learning_rate_range=[0.01, 0.05, 0.1, 0.2],
                 max_depth_range=[3, 5, 7, 9],
                 min_child_weight_range=[1, 3, 5],
                 subsample_range=[0.6, 0.8, 1.0],
                 colsample_bytree_range=[0.6, 0.8, 1.0],
                 hidden_layer_sizes_range=[(10,), (10, 5), (20, 10)],
                 activation_options=['relu', 'tanh', 'logistic'],
                 solver_options=['adam', 'sgd', 'lbfgs'],
                 alpha_range=[0.0001, 0.001, 0.01, 0.1],
                 max_iter=200,
                 random_state=42,
                 dist_method="gamma",
                 n_iter_search=10,
                 cv_folds=3):
        self.n_estimators_range = n_estimators_range
        self.learning_rate_range = learning_rate_range
        self.max_depth_range = max_depth_range
        self.min_child_weight_range = min_child_weight_range
        self.subsample_range = subsample_range
        self.colsample_bytree_range = colsample_bytree_range
        self.hidden_layer_sizes_range = hidden_layer_sizes_range
        self.activation_options = activation_options
        self.solver_options = solver_options
        self.alpha_range = alpha_range
        self.max_iter = max_iter
        self.random_state = random_state
        self.dist_method = dist_method
        self.n_iter_search = n_iter_search
        self.cv_folds = cv_folds
        self.stacking_model = None

    def compute_hyperparameters(self, Predictors, Predictand, clim_year_start, clim_year_end):
        """
        Independently computes the best hyperparameters using randomized search on stacked training data.

        Parameters
        ----------
        Predictors : xarray.DataArray
            Training predictor data with dimensions (T, M, Y, X).
        Predictand : xarray.DataArray
            Training predictand data with dimensions (T, Y, X).
        clim_year_start : int
            Start year of the climatology period.
        clim_year_end : int
            End year of the climatology period.

        Returns
        -------
        dict
            Best hyperparameters found.
        """
        X_train = standardize_timeseries(Predictors, clim_year_start, clim_year_end)
        y_train = standardize_timeseries(Predictand, clim_year_start, clim_year_end)

        # Stack training data
        X_train_stacked = X_train.stack(sample=('T', 'Y', 'X')).transpose('sample', 'M').values
        y_train_stacked = y_train.stack(sample=('T', 'Y', 'X')).values.ravel()
        train_nan_mask = np.any(~np.isfinite(X_train_stacked), axis=1) | ~np.isfinite(y_train_stacked)
        X_train_clean = X_train_stacked[~train_nan_mask]
        y_train_clean = y_train_stacked[~train_nan_mask]

        # For XGBoost (base model)
        param_dist_xgb = {
            'n_estimators': self.n_estimators_range,
            'learning_rate': self.learning_rate_range,
            'max_depth': self.max_depth_range,
            'min_child_weight': self.min_child_weight_range,
            'subsample': self.subsample_range,
            'colsample_bytree': self.colsample_bytree_range
        }
        model_xgb = XGBRegressor(random_state=self.random_state, verbosity=0)
        random_search_xgb = RandomizedSearchCV(
            model_xgb, param_distributions=param_dist_xgb, n_iter=self.n_iter_search,
            cv=self.cv_folds, scoring='neg_mean_squared_error',
            random_state=self.random_state, error_score=np.nan
        )
        random_search_xgb.fit(X_train_clean, y_train_clean)
        best_xgb = random_search_xgb.best_params_

        # For MLP (meta model)
        param_dist_mlp = {
            'hidden_layer_sizes': self.hidden_layer_sizes_range,
            'activation': self.activation_options,
            'solver': self.solver_options,
            'alpha': self.alpha_range
        }
        model_mlp = MLPRegressor(max_iter=self.max_iter, random_state=self.random_state)
        random_search_mlp = RandomizedSearchCV(
            model_mlp, param_distributions=param_dist_mlp, n_iter=self.n_iter_search,
            cv=self.cv_folds, scoring='neg_mean_squared_error',
            random_state=self.random_state, error_score=np.nan
        )
        random_search_mlp.fit(X_train_clean, y_train_clean)
        best_mlp = random_search_mlp.best_params_

        return {'xgb': best_xgb, 'mlp': best_mlp}

    def compute_model(self, X_train, y_train, X_test, y_test, best_params=None):
        """
        Compute deterministic hindcast using the stacking ensemble of XGBoost (base) and MLP (meta).

        Parameters
        ----------
        X_train : xarray.DataArray
            Training predictor data with dimensions (T, M, Y, X).
        y_train : xarray.DataArray
            Training predictand data with dimensions (T, Y, X).
        X_test : xarray.DataArray
            Testing predictor data with dimensions (T, M, Y, X).
        y_test : xarray.DataArray
            Testing predictand data with dimensions (T, Y, X).
        best_params : dict, optional
            Pre-computed best hyperparameters for 'xgb' and 'mlp'. If None, computes internally.

        Returns
        -------
        predicted_da : xarray.DataArray
            Deterministic hindcast with dimensions (T, Y, X).
        """
        # Extract coordinate variables from X_test
        time = X_test['T']
        lat = X_test['Y']
        lon = X_test['X']
        n_time = len(X_test.coords['T'])
        n_lat = len(X_test.coords['Y'])
        n_lon = len(X_test.coords['X'])

        # Stack training data
        X_train_stacked = X_train.stack(sample=('T', 'Y', 'X')).transpose('sample', 'M').values
        y_train_stacked = y_train.stack(sample=('T', 'Y', 'X')).transpose('sample', 'M').values
        train_nan_mask = (np.any(~np.isfinite(X_train_stacked), axis=1) | np.any(~np.isfinite(y_train_stacked), axis=1))
        X_train_clean = X_train_stacked[~train_nan_mask]
        y_train_clean = y_train_stacked[~train_nan_mask]

        # Stack testing data
        X_test_stacked = X_test.stack(sample=('T', 'Y', 'X')).transpose('sample', 'M').values
        y_test_stacked = y_test.stack(sample=('T', 'Y', 'X')).transpose('sample', 'M').values
        test_nan_mask = (np.any(~np.isfinite(X_test_stacked), axis=1) | np.any(~np.isfinite(y_test_stacked), axis=1))

        # Use provided best_params or compute if None
        if best_params is None:
            best_params = self.compute_hyperparameters(X_train_clean, y_train_clean)

        # Initialize base model (XGBoost)
        xgb = XGBRegressor(**best_params['xgb'], random_state=self.random_state, verbosity=0)
        base_models = [('xgb', xgb)]

        # Initialize meta-model (MLP)
        meta_model = MLPRegressor(**best_params['mlp'], max_iter=self.max_iter, random_state=self.random_state)

        # Initialize stacking ensemble
        self.stacking_model = StackingRegressor(estimators=base_models, final_estimator=meta_model)

        # Fit the stacking model on clean training data
        self.stacking_model.fit(X_train_clean, y_train_clean)

        # Predict on clean test data
        y_pred_test = self.stacking_model.predict(X_test_stacked[~test_nan_mask])

        # Reconstruct predictions
        result = np.empty_like(np.squeeze(y_test_stacked))
        result[np.squeeze(test_nan_mask)] = np.squeeze(y_test_stacked[test_nan_mask])
        result[~np.squeeze(test_nan_mask)] = y_pred_test

        predictions_reshaped = result.reshape(n_time, n_lat, n_lon)
        predicted_da = xr.DataArray(data=predictions_reshaped,
                                    coords={'T': time, 'Y': lat, 'X': lon},
                                    dims=['T', 'Y', 'X'])
        return predicted_da

    # ------------------ Probability Calculation Methods ------------------
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
    def calculate_tercile_probabilities_weibull_min(best_guess, error_variance, first_tercile, second_tercile, dof):
        """
        Weibull minimum-based method.
        """
        n_time = len(best_guess)
        pred_prob = np.empty((3, n_time))
        if np.all(np.isnan(best_guess)):
            pred_prob[:] = np.nan
        else:
            error_std = np.sqrt(error_variance)
            pred_prob[0, :] = stats.weibull_min.cdf(first_tercile, c=dof, loc=best_guess, scale=error_std)
            pred_prob[1, :] = stats.weibull_min.cdf(second_tercile, c=dof, loc=best_guess, scale=error_std) - \
                              stats.weibull_min.cdf(first_tercile, c=dof, loc=best_guess, scale=error_std)
            pred_prob[2, :] = 1 - stats.weibull_min.cdf(second_tercile, c=dof, loc=best_guess, scale=error_std)
        return pred_prob

    @staticmethod
    def calculate_tercile_probabilities_gamma(best_guess, error_variance, T1, T2, dof=None):
        """Gamma-distribution based method."""
        n_time = len(best_guess)
        pred_prob = np.empty((3, n_time), dtype=float)
        if np.any(np.isnan(best_guess)) or np.any(np.isnan(error_variance)):
            pred_prob[:] = np.nan
            return pred_prob
        best_guess = np.asarray(best_guess, dtype=float)
        error_variance = np.asarray(error_variance, dtype=float)
        T1 = np.asarray(T1, dtype=float)
        T2 = np.asarray(T2, dtype=float)
        alpha = (best_guess ** 2) / error_variance
        theta = error_variance / best_guess
        cdf_t1 = gamma.cdf(T1, a=alpha, scale=theta)
        cdf_t2 = gamma.cdf(T2, a=alpha, scale=theta)
        pred_prob[0, :] = cdf_t1
        pred_prob[1, :] = cdf_t2 - cdf_t1
        pred_prob[2, :] = 1.0 - cdf_t2
        return pred_prob

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

    @staticmethod
    def calculate_tercile_probabilities_normal(best_guess, error_variance, first_tercile, second_tercile):
        """Normal-distribution based method."""
        n_time = len(best_guess)
        pred_prob = np.empty((3, n_time))
        if np.all(np.isnan(best_guess)):
            pred_prob[:] = np.nan
        else:
            error_std = np.sqrt(error_variance)
            pred_prob[0, :] = norm.cdf(first_tercile, loc=best_guess, scale=error_std)
            pred_prob[1, :] = norm.cdf(second_tercile, loc=best_guess, scale=error_std) - norm.cdf(first_tercile, loc=best_guess, scale=error_std)
            pred_prob[2, :] = 1 - norm.cdf(second_tercile, loc=best_guess, scale=error_std)
        return pred_prob

    @staticmethod
    def calculate_tercile_probabilities_lognormal(best_guess, error_variance, first_tercile, second_tercile):
        """Lognormal-distribution based method."""
        n_time = len(best_guess)
        pred_prob = np.empty((3, n_time))
        if np.any(np.isnan(best_guess)) or np.any(np.isnan(error_variance)):
            pred_prob[:] = np.nan
            return pred_prob
        sigma = np.sqrt(np.log(1 + error_variance / (best_guess ** 2)))
        mu = np.log(best_guess) - sigma ** 2 / 2
        pred_prob[0, :] = lognorm.cdf(first_tercile, s=sigma, scale=np.exp(mu))
        pred_prob[1, :] = lognorm.cdf(second_tercile, s=sigma, scale=np.exp(mu)) - lognorm.cdf(first_tercile, s=sigma, scale=np.exp(mu))
        pred_prob[2, :] = 1 - lognorm.cdf(second_tercile, s=sigma, scale=np.exp(mu))
        return pred_prob

    def compute_prob(self, Predictant, clim_year_start, clim_year_end, hindcast_det):
        """
        Compute tercile probabilities using the chosen distribution method.
        Predictant is expected to be an xarray DataArray with dims (T, Y, X).
        """
        if "M" in Predictant.coords:
            Predictant = Predictant.isel(M=0).drop_vars('M').squeeze()
        mask = xr.where(~np.isnan(Predictant.isel(T=0)), 1, np.nan).drop_vars(['T']).squeeze().to_numpy()

        Predictant = Predictant.transpose('T', 'Y', 'X')
        index_start = Predictant.get_index("T").get_loc(str(clim_year_start)).start
        index_end = Predictant.get_index("T").get_loc(str(clim_year_end)).stop
        rainfall_for_tercile = Predictant.isel(T=slice(index_start, index_end))
        terciles = rainfall_for_tercile.quantile([0.33, 0.67], dim='T')
        error_variance = (Predictant - hindcast_det).var(dim='T')
        dof = len(Predictant.get_index("T")) - 2

        if self.dist_method == "t":
            calc_func = self.calculate_tercile_probabilities
            hindcast_prob = xr.apply_ufunc(
                calc_func,
                hindcast_det,
                error_variance,
                terciles.isel(quantile=0).drop_vars('quantile'),
                terciles.isel(quantile=1).drop_vars('quantile'),
                input_core_dims=[('T',), (), (), ()],
                vectorize=True,
                kwargs={'dof': dof},
                dask='parallelized',
                output_core_dims=[('probability', 'T')],
                output_dtypes=['float'],
                dask_gufunc_kwargs={'output_sizes': {'probability': 3}, "allow_rechunk": True}
            )
        elif self.dist_method == "weibull_min":
            calc_func = self.calculate_tercile_probabilities_weibull_min
            hindcast_prob = xr.apply_ufunc(
                calc_func,
                hindcast_det,
                error_variance,
                terciles.isel(quantile=0).drop_vars('quantile'),
                terciles.isel(quantile=1).drop_vars('quantile'),
                input_core_dims=[('T',), (), (), ()],
                vectorize=True,
                kwargs={'dof': dof},
                dask='parallelized',
                output_core_dims=[('probability', 'T')],
                output_dtypes=['float'],
                dask_gufunc_kwargs={'output_sizes': {'probability': 3}, "allow_rechunk": True}
            )
        elif self.dist_method == "gamma":
            calc_func = self.calculate_tercile_probabilities_gamma
            hindcast_prob = xr.apply_ufunc(
                calc_func,
                hindcast_det,
                error_variance,
                terciles.isel(quantile=0).drop_vars('quantile'),
                terciles.isel(quantile=1).drop_vars('quantile'),
                input_core_dims=[('T',), (), (), ()],
                vectorize=True,
                dask='parallelized',
                output_core_dims=[('probability', 'T')],
                output_dtypes=['float'],
                dask_gufunc_kwargs={'output_sizes': {'probability': 3}, "allow_rechunk": True}
            )
        elif self.dist_method == "nonparam":
            calc_func = self.calculate_tercile_probabilities_nonparametric
            error_samples = (Predictant - hindcast_det)
            hindcast_prob = xr.apply_ufunc(
                calc_func,
                hindcast_det,
                error_samples,
                terciles.isel(quantile=0).drop_vars('quantile'),
                terciles.isel(quantile=1).drop_vars('quantile'),
                input_core_dims=[('T',), ('T',), (), ()],
                output_core_dims=[('probability', 'T')],
                vectorize=True,
                dask='parallelized',
                output_dtypes=[float],
                dask_gufunc_kwargs={'output_sizes': {'probability': 3}, "allow_rechunk": True}
            )
        elif self.dist_method == "normal":
            calc_func = self.calculate_tercile_probabilities_normal
            hindcast_prob = xr.apply_ufunc(
                calc_func,
                hindcast_det,
                error_variance,
                terciles.isel(quantile=0).drop_vars('quantile'),
                terciles.isel(quantile=1).drop_vars('quantile'),
                input_core_dims=[('T',), (), (), ()],
                vectorize=True,
                dask='parallelized',
                output_core_dims=[('probability', 'T')],
                output_dtypes=['float'],
                dask_gufunc_kwargs={'output_sizes': {'probability': 3}, "allow_rechunk": True}
            )
        elif self.dist_method == "lognormal":
            calc_func = self.calculate_tercile_probabilities_lognormal
            hindcast_prob = xr.apply_ufunc(
                calc_func,
                hindcast_det,
                error_variance,
                terciles.isel(quantile=0).drop_vars('quantile'),
                terciles.isel(quantile=1).drop_vars('quantile'),
                input_core_dims=[('T',), (), (), ()],
                vectorize=True,
                dask='parallelized',
                output_core_dims=[('probability', 'T')],
                output_dtypes=['float'],
                dask_gufunc_kwargs={'output_sizes': {'probability': 3}, "allow_rechunk": True}
            )
        else:
            raise ValueError(f"Invalid dist_method: {self.dist_method}.")

        hindcast_prob = hindcast_prob.assign_coords(probability=('probability', ['PB', 'PN', 'PA']))
        return hindcast_prob.transpose('probability', 'T', 'Y', 'X')

    @staticmethod
    def _reshape_and_filter_data(da):
        """
        Helper: stack the DataArray from (T, Y, X[, M]) to (n_samples, n_features)
        and remove rows containing NaNs.
        """
        da_stacked = da.stack(sample=('T', 'Y', 'X'))
        if 'M' in da.dims:
            da_stacked = da_stacked.transpose('sample', 'M')
        else:
            da_stacked = da_stacked.transpose('sample')
        da_values = da_stacked.values
        nan_mask = np.any(np.isnan(da_values), axis=1)
        return da_values[~nan_mask], nan_mask, da_values

    def forecast(self, Predictant, clim_year_start, clim_year_end, hindcast_det, hindcast_det_cross, Predictor_for_year, best_params=None):
        """
        Generate deterministic and probabilistic forecast for a target year using the stacking ensemble.

        Parameters
        ----------
        Predictant : xarray.DataArray
            Observed predictand data with dimensions (T, Y, X) or (T, M, Y, X).
        clim_year_start : int or str
            Start year of the climatology period.
        clim_year_end : int or str
            End year of the climatology period.
        hindcast_det : xarray.DataArray
            Deterministic hindcast data for training with dimensions (T, M, Y, X).
        hindcast_det_cross : xarray.DataArray
            Deterministic hindcast data for error estimation with dimensions (T, Y, X).
        Predictor_for_year : xarray.DataArray
            Predictor data for the target year with dimensions (T, M, Y, X).
        best_params : dict, optional
            Pre-computed best hyperparameters for 'xgb' and 'mlp'. If None, computes internally.

        Returns
        -------
        result_da : xarray.DataArray
            Deterministic forecast with dimensions (T, Y, X).
        hindcast_prob : xarray.DataArray
            Tercile probabilities with dimensions (probability, T, Y, X).
        """
        if "M" in Predictant.coords:
            Predictant_no_m = Predictant.isel(M=0).drop_vars('M').squeeze()
        else:
            Predictant_no_m = Predictant

        mask = xr.where(~np.isnan(Predictant_no_m.isel(T=0)), 1, np.nan).drop_vars(['T']).squeeze().to_numpy()

        # Standardize Predictor_for_year using hindcast climatology
        mean_val = hindcast_det.sel(T=slice(str(clim_year_start), str(clim_year_end))).mean(dim='T')
        std_val = hindcast_det.sel(T=slice(str(clim_year_start), str(clim_year_end))).std(dim='T')
        Predictor_for_year_st = (Predictor_for_year - mean_val) / std_val

        hindcast_det_st = standardize_timeseries(hindcast_det, clim_year_start, clim_year_end)
        Predictant_st = standardize_timeseries(Predictant_no_m, clim_year_start, clim_year_end)
        y_test = Predictant_st.isel(T=[-1])

        # Extract coordinates from X_test
        time = Predictor_for_year_st['T']
        lat = Predictor_for_year_st['Y']
        lon = Predictor_for_year_st['X']
        n_time = len(Predictor_for_year_st.coords['T'])
        n_lat = len(Predictor_for_year_st.coords['Y'])
        n_lon = len(Predictor_for_year_st.coords['X'])

        # Stack training data and remove rows with NaNs
        X_train_stacked = hindcast_det_st.stack(sample=('T', 'Y', 'X')).transpose('sample', 'M').values
        y_train_stacked = Predictant_st.stack(sample=('T', 'Y', 'X')).values.ravel()
        train_nan_mask = np.any(~np.isfinite(X_train_stacked), axis=1) | ~np.isfinite(y_train_stacked)
        X_train_clean = X_train_stacked[~train_nan_mask]
        y_train_clean = y_train_stacked[~train_nan_mask]

        # Stack testing data
        X_test_stacked = Predictor_for_year_st.stack(sample=('T', 'Y', 'X')).transpose('sample', 'M').values
        y_test_stacked = y_test.stack(sample=('T', 'Y', 'X')).values.ravel()
        test_nan_mask = np.any(~np.isfinite(X_test_stacked), axis=1) | ~np.isfinite(y_test_stacked)

        # Use provided best_params or compute if None
        if best_params is None:
            best_params = self.compute_hyperparameters(X_train_clean, y_train_clean)

        # Initialize base model (XGBoost)
        xgb = XGBRegressor(**best_params['xgb'], random_state=self.random_state, verbosity=0)
        base_models = [('xgb', xgb)]

        # Initialize meta-model (MLP)
        meta_model = MLPRegressor(**best_params['mlp'], max_iter=self.max_iter, random_state=self.random_state)

        # Initialize stacking ensemble
        self.stacking_model = StackingRegressor(estimators=base_models, final_estimator=meta_model)

        # Fit the stacking model
        self.stacking_model.fit(X_train_clean, y_train_clean)
        y_pred = self.stacking_model.predict(X_test_stacked[~test_nan_mask])

        # Reconstruct the prediction array
        result = np.empty_like(np.squeeze(y_test_stacked))
        result[np.squeeze(test_nan_mask)] = np.squeeze(y_test_stacked[test_nan_mask])
        result[~np.squeeze(test_nan_mask)] = y_pred

        predictions_reshaped = result.reshape(n_time, n_lat, n_lon)
        result_da = xr.DataArray(data=predictions_reshaped,
                                 coords={'T': time, 'Y': lat, 'X': lon},
                                 dims=['T', 'Y', 'X']) * mask

        result_da = reverse_standardize(result_da, Predictant_no_m, clim_year_start, clim_year_end)
        year = Predictor_for_year.coords['T'].values.astype('datetime64[Y]').astype(int)[0] + 1970
        T_value_1 = Predictant_no_m.isel(T=0).coords['T'].values
        month_1 = T_value_1.astype('datetime64[M]').astype(int) % 12 + 1
        new_T_value = np.datetime64(f"{year}-{month_1:02d}-{1:02d}")
        result_da = result_da.assign_coords(T=xr.DataArray([new_T_value], dims=["T"]))
        result_da['T'] = result_da['T'].astype('datetime64[ns]')

        # Compute tercile probabilities
        index_start = Predictant_no_m.get_index("T").get_loc(str(clim_year_start)).start
        index_end = Predictant_no_m.get_index("T").get_loc(str(clim_year_end)).stop
        rainfall_for_tercile = Predictant_no_m.isel(T=slice(index_start, index_end))
        terciles = rainfall_for_tercile.quantile([0.33, 0.67], dim='T')
        error_variance = (Predictant_no_m - hindcast_det_cross).var(dim='T')
        dof = len(Predictant_no_m.get_index("T")) - 2

        if self.dist_method == "t":
            calc_func = self.calculate_tercile_probabilities
            hindcast_prob = xr.apply_ufunc(
                calc_func,
                result_da,
                error_variance,
                terciles.isel(quantile=0).drop_vars('quantile'),
                terciles.isel(quantile=1).drop_vars('quantile'),
                input_core_dims=[('T',), (), (), ()],
                vectorize=True,
                kwargs={'dof': dof},
                dask='parallelized',
                output_core_dims=[('probability','T')],
                output_dtypes=['float'],
                dask_gufunc_kwargs={'output_sizes': {'probability': 3}, "allow_rechunk": True}
            )
        elif self.dist_method == "weibull_min":
            calc_func = self.calculate_tercile_probabilities_weibull_min
            hindcast_prob = xr.apply_ufunc(
                calc_func,
                result_da,
                error_variance,
                terciles.isel(quantile=0).drop_vars('quantile'),
                terciles.isel(quantile=1).drop_vars('quantile'),
                input_core_dims=[('T',), (), (), ()],
                vectorize=True,
                kwargs={'dof': dof},
                dask='parallelized',
                output_core_dims=[('probability', 'T')],
                output_dtypes=['float'],
                dask_gufunc_kwargs={'output_sizes': {'probability': 3}, "allow_rechunk": True}
            )
        elif self.dist_method == "gamma":
            calc_func = self.calculate_tercile_probabilities_gamma
            hindcast_prob = xr.apply_ufunc(
                calc_func,
                result_da,
                error_variance,
                terciles.isel(quantile=0).drop_vars('quantile'),
                terciles.isel(quantile=1).drop_vars('quantile'),
                input_core_dims=[('T',), (), (), ()],
                vectorize=True,
                dask='parallelized',
                output_core_dims=[('probability','T')],
                output_dtypes=['float'],
                dask_gufunc_kwargs={'output_sizes': {'probability': 3}, "allow_rechunk": True}
            )
        elif self.dist_method == "normal":
            calc_func = self.calculate_tercile_probabilities_normal
            hindcast_prob = xr.apply_ufunc(
                calc_func,
                result_da,
                error_variance,
                terciles.isel(quantile=0).drop_vars('quantile'),
                terciles.isel(quantile=1).drop_vars('quantile'),
                input_core_dims=[('T',), (), (), ()],
                vectorize=True,
                dask='parallelized',
                output_core_dims=[('probability','T')],
                output_dtypes=['float'],
                dask_gufunc_kwargs={'output_sizes': {'probability': 3}, "allow_rechunk": True}
            )
        elif self.dist_method == "lognormal":
            calc_func = self.calculate_tercile_probabilities_lognormal
            hindcast_prob = xr.apply_ufunc(
                calc_func,
                result_da,
                error_variance,
                terciles.isel(quantile=0).drop_vars('quantile'),
                terciles.isel(quantile=1).drop_vars('quantile'),
                input_core_dims=[('T',), (), (), ()],
                vectorize=True,
                dask='parallelized',
                output_core_dims=[('probability','T')],
                output_dtypes=['float'],
                dask_gufunc_kwargs={'output_sizes': {'probability': 3}, "allow_rechunk": True}
            )
        elif self.dist_method == "nonparam":
            calc_func = self.calculate_tercile_probabilities_nonparametric
            error_samples = Predictant_no_m - hindcast_det_cross
            error_samples = error_samples.rename({'T': 'S'})
            hindcast_prob = xr.apply_ufunc(
                calc_func,
                result_da,
                error_samples,
                terciles.isel(quantile=0).drop_vars('quantile'),
                terciles.isel(quantile=1).drop_vars('quantile'),
                input_core_dims=[('T',), ('S',), (), ()],
                output_core_dims=[('probability','T')],
                vectorize=True,
                dask='parallelized',
                output_dtypes=[float],
                dask_gufunc_kwargs={'output_sizes': {'probability': 3}, "allow_rechunk": True}
            )
        else:
            raise ValueError(f"Invalid dist_method: {self.dist_method}.")

        hindcast_prob = hindcast_prob.assign_coords(probability=('probability', ['PB', 'PN', 'PA']))
        return result_da * mask, mask * hindcast_prob.transpose('probability', 'T', 'Y', 'X')



class WAS_mme_StackXGboost_MLP:
    """
    Stacking ensemble with XGBoost base model and MLP meta-model for Multi-Model Ensemble (MME) forecasting.
    This class implements a stacking ensemble using XGBRegressor as the base model,
    with an MLPRegressor meta-model, for deterministic forecasting and optional tercile probability calculations.
    Implements hyperparameter optimization via randomized search for both base and meta models.
    Parameters
    ----------
    n_estimators_range : list of int, optional
        List of n_estimators values to tune for XGBoost (default is [50, 100, 200, 300]).
    learning_rate_range : list of float, optional
        List of learning rates to tune for XGBoost (default is [0.01, 0.05, 0.1, 0.2]).
    max_depth_range : list of int, optional
        List of max depths to tune for XGBoost (default is [3, 5, 7, 9]).
    min_child_weight_range : list of float, optional
        List of minimum child weights to tune for XGBoost (default is [1, 3, 5]).
    subsample_range : list of float, optional
        List of subsample ratios to tune for XGBoost (default is [0.6, 0.8, 1.0]).
    colsample_bytree_range : list of float, optional
        List of column sampling ratios to tune for XGBoost (default is [0.6, 0.8, 1.0]).
    hidden_layer_sizes_range : list of tuples, optional
        List of hidden layer sizes to tune for MLP (default is [(10,), (10, 5), (20, 10)]).
    activation_options : list of str, optional
        Activation functions to tune for MLP (default is ['relu', 'tanh', 'logistic']).
    solver_options : list of str, optional
        Solvers to tune for MLP (default is ['adam', 'sgd', 'lbfgs']).
    alpha_range : list of float, optional
        L2 regularization parameters to tune for MLP (default is [0.0001, 0.001, 0.01, 0.1]).
    max_iter : int, optional
        Maximum iterations for MLP (default is 200).
    random_state : int, optional
        Seed for reproducibility (default is 42).
    dist_method : str, optional
        Distribution method for tercile probabilities ('t', 'gamma', etc.) (default is 'gamma').
    n_iter_search : int, optional
        Number of iterations for randomized search (default is 10).
    cv_folds : int, optional
        Number of cross-validation folds (default is 3).
    n_clusters : int, optional
        Number of clusters for homogenized zones (default is 4).
    """
    def __init__(self,
                 n_estimators_range=[50, 100, 200, 300],
                 learning_rate_range=[0.01, 0.05, 0.1, 0.2],
                 max_depth_range=[3, 5, 7, 9],
                 min_child_weight_range=[1, 3, 5],
                 subsample_range=[0.6, 0.8, 1.0],
                 colsample_bytree_range=[0.6, 0.8, 1.0],
                 hidden_layer_sizes_range=[(10,), (10, 5), (20, 10)],
                 activation_options=['relu', 'tanh', 'logistic'],
                 solver_options=['adam', 'sgd', 'lbfgs'],
                 alpha_range=[0.0001, 0.001, 0.01, 0.1],
                 max_iter=200,
                 random_state=42,
                 dist_method="gamma",
                 n_iter_search=10,
                 cv_folds=3,
                 n_clusters=4):
        self.n_estimators_range = n_estimators_range
        self.learning_rate_range = learning_rate_range
        self.max_depth_range = max_depth_range
        self.min_child_weight_range = min_child_weight_range
        self.subsample_range = subsample_range
        self.colsample_bytree_range = colsample_bytree_range
        self.hidden_layer_sizes_range = hidden_layer_sizes_range
        self.activation_options = activation_options
        self.solver_options = solver_options
        self.alpha_range = alpha_range
        self.max_iter = max_iter
        self.random_state = random_state
        self.dist_method = dist_method
        self.n_iter_search = n_iter_search
        self.cv_folds = cv_folds
        self.n_clusters = n_clusters
        self.stacking_model = None

    def compute_hyperparameters(self, Predictors, Predictand, clim_year_start, clim_year_end):
        """
        Independently computes the best hyperparameters using randomized search on stacked training data for each homogenized zone.
        Parameters
        ----------
        Predictors : xarray.DataArray
            Training predictor data with dimensions (T, M, Y, X).
        Predictand : xarray.DataArray
            Training predictand data with dimensions (T, Y, X).
        clim_year_start : int
            Start year of the climatology period.
        clim_year_end : int
            End year of the climatology period.
        Returns
        -------
        best_params_dict : dict
            Best hyperparameters for each cluster.
        cluster_da : xarray.DataArray
            Cluster labels with dimensions (Y, X).
        """
        if "M" in Predictand.coords:
            Predictand = Predictand.isel(M=0).drop_vars('M').squeeze()
           
        X_train_std = standardize_timeseries(Predictors, clim_year_start, clim_year_end)
        Predictand.name = "varname"
        # Step 1: Perform KMeans clustering based on predictand's spatial distribution
        kmeans = KMeans(n_clusters=self.n_clusters, random_state=self.random_state)
        Predictand_dropna = Predictand.to_dataframe().reset_index().dropna().drop(columns=['T'])
        variable_column = Predictand_dropna.columns[2]
        Predictand_dropna['cluster'] = kmeans.fit_predict(
           Predictand_dropna[[variable_column]]
        )
       
        # Convert cluster assignments back into an xarray structure
        df_unique = Predictand_dropna.drop_duplicates(subset=['Y', 'X'])
        dataset = df_unique.set_index(['Y', 'X']).to_xarray()
        mask = xr.where(~np.isnan(Predictand.isel(T=0)), 1, np.nan)
        Cluster = (dataset['cluster'] * mask)
              
        # Align cluster array with the predictand array
        xarray1, xarray2 = xr.align(Predictand, Cluster, join="outer")
       
        # Identify unique cluster labels
        clusters = np.unique(xarray2)
        clusters = clusters[~np.isnan(clusters)]
        cluster_da = xarray2
        y_train_std = standardize_timeseries(Predictand, clim_year_start, clim_year_end)
        X_train_std['T'] = y_train_std['T']
        # Prepare parameter distributions for XGBoost and MLP
        param_dist_xgb = {
            'n_estimators': self.n_estimators_range,
            'learning_rate': self.learning_rate_range,
            'max_depth': self.max_depth_range,
            'min_child_weight': self.min_child_weight_range,
            'subsample': self.subsample_range,
            'colsample_bytree': self.colsample_bytree_range
        }
        param_dist_mlp = {
            'hidden_layer_sizes': self.hidden_layer_sizes_range,
            'activation': self.activation_options,
            'solver': self.solver_options,
            'alpha': self.alpha_range
        }
        best_params_dict = {}
        for c in clusters:
            mask_3d = (cluster_da == c).expand_dims({'T': y_train_std['T']})
            X_stacked_c = X_train_std.where(mask_3d).stack(sample=('T', 'Y', 'X')).transpose('sample', 'M').values
            y_stacked_c = y_train_std.where(mask_3d).stack(sample=('T', 'Y', 'X')).values.ravel()
            nan_mask_c = np.any(~np.isfinite(X_stacked_c), axis=1) | ~np.isfinite(y_stacked_c)
            X_clean_c = X_stacked_c[~nan_mask_c]
            y_clean_c = y_stacked_c[~nan_mask_c]
            if len(X_clean_c) == 0:
                continue
            # Tune XGBoost
            model_xgb = XGBRegressor(random_state=self.random_state, verbosity=0, n_jobs=-1)
            random_search_xgb = RandomizedSearchCV(
                model_xgb, param_distributions=param_dist_xgb, n_iter=self.n_iter_search,
                cv=self.cv_folds, scoring='neg_mean_squared_error',
                random_state=self.random_state, error_score=np.nan
            )
            random_search_xgb.fit(X_clean_c, y_clean_c)
            best_xgb = random_search_xgb.best_params_
            # Tune MLP
            model_mlp = MLPRegressor(max_iter=self.max_iter, random_state=self.random_state)
            random_search_mlp = RandomizedSearchCV(
                model_mlp, param_distributions=param_dist_mlp, n_iter=self.n_iter_search,
                cv=self.cv_folds, scoring='neg_mean_squared_error',
                random_state=self.random_state, error_score=np.nan
            )
            random_search_mlp.fit(X_clean_c, y_clean_c)
            best_mlp = random_search_mlp.best_params_
            best_params_dict[c] = {'xgb': best_xgb, 'mlp': best_mlp}
        return best_params_dict, cluster_da

    def compute_model(self, X_train, y_train, X_test, y_test, best_params=None, cluster_da=None):
        """
        Compute deterministic hindcast using the stacking ensemble of XGBoost (base) and MLP (meta) for each zone.
        Parameters
        ----------
        X_train : xarray.DataArray
            Training predictor data with dimensions (T, M, Y, X).
        y_train : xarray.DataArray
            Training predictand data with dimensions (T, Y, X).
        X_test : xarray.DataArray
            Testing predictor data with dimensions (T, M, Y, X).
        y_test : xarray.DataArray
            Testing predictand data with dimensions (T, Y, X).
        clim_year_start : int
            Start year of the climatology period.
        clim_year_end : int
            End year of the climatology period.
        best_params : dict, optional
            Pre-computed best hyperparameters per cluster. If None, computes internally.
        cluster_da : xarray.DataArray, optional
            Pre-computed cluster labels. If None, computes internally.
        Returns
        -------
        predicted_da : xarray.DataArray
            Deterministic hindcast with dimensions (T, Y, X).
        """
        # Standardize inputs
        X_train_std = X_train
        y_train_std = y_train
        X_test_std = X_test
        y_test_std = y_test
        
        # Extract coordinate variables from X_test
        time = X_test_std['T']
        lat = X_test_std['Y']
        lon = X_test_std['X']
        n_time = len(time)
        n_lat = len(lat)
        n_lon = len(lon)
        # Use provided best_params and cluster_da or compute if None
        if best_params is None:
            best_params, cluster_da = self.compute_hyperparameters(X_train, y_train, clim_year_start, clim_year_end)
        # Initialize predictions array
        predictions = np.full((n_time, n_lat, n_lon), np.nan)
        self.stacking_model = {}  # Dictionary to store models per cluster
        for c in range(self.n_clusters):
            if c not in best_params:
                continue
            bp = best_params[c]
            # Mask for this cluster
            mask_3d_train = (cluster_da == c).expand_dims({'T': X_train_std['T']})
            mask_3d_test = (cluster_da == c).expand_dims({'T': X_test_std['T']})
            # Stack training data for cluster
            X_train_stacked_c = X_train_std.where(mask_3d_train).stack(sample=('T', 'Y', 'X')).transpose('sample', 'M').values
            y_train_stacked_c = y_train_std.where(mask_3d_train).stack(sample=('T', 'Y', 'X')).values.ravel()
            train_nan_mask = np.any(~np.isfinite(X_train_stacked_c), axis=1) | ~np.isfinite(y_train_stacked_c)
            X_train_clean_c = X_train_stacked_c[~train_nan_mask]
            y_train_clean_c = y_train_stacked_c[~train_nan_mask]
            # Stack testing data for cluster
            X_test_stacked_c = X_test_std.where(mask_3d_test).stack(sample=('T', 'Y', 'X')).transpose('sample', 'M').values
            y_test_stacked_c = y_test_std.where(mask_3d_test).stack(sample=('T', 'Y', 'X')).values.ravel()
            test_nan_mask = np.any(~np.isfinite(X_test_stacked_c), axis=1) | ~np.isfinite(y_test_stacked_c)
            X_test_clean_c = X_test_stacked_c[~test_nan_mask]
            # Initialize base model (XGBoost)
            xgb = XGBRegressor(**bp['xgb'], random_state=self.random_state, verbosity=0)
            # Initialize meta-model (MLPRegressor)
            meta_model = MLPRegressor(**bp['mlp'], max_iter=self.max_iter, random_state=self.random_state)
            # Initialize stacking ensemble for this cluster
            stacking_model_c = StackingRegressor(estimators=[('xgb', xgb)], final_estimator=meta_model)
            # Fit the stacking model on cluster data
            stacking_model_c.fit(X_train_clean_c, y_train_clean_c)
            self.stacking_model[c] = stacking_model_c
            # Predict
            y_pred_c = stacking_model_c.predict(X_test_clean_c)
            # Reconstruct predictions for this cluster
            result_c = np.full(len(y_test_stacked_c), np.nan)
            result_c[~test_nan_mask] = y_pred_c
            pred_c_reshaped = result_c.reshape(n_time, n_lat, n_lon)
            # Fill in the predictions array
            predictions = np.where(np.isnan(predictions), pred_c_reshaped, predictions)
        predicted_da = xr.DataArray(
            data=predictions,
            coords={'T': time, 'Y': lat, 'X': lon},
            dims=['T', 'Y', 'X']
        )
        return predicted_da

    # ------------------ Probability Calculation Methods ------------------
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
            pred_prob[0, :] = t.cdf(first_t, df=dof)
            pred_prob[1, :] = t.cdf(second_t, df=dof) - t.cdf(first_t, df=dof)
            pred_prob[2, :] = 1 - t.cdf(second_t, df=dof)
        return pred_prob

    @staticmethod
    def calculate_tercile_probabilities_weibull_min(best_guess, error_variance, first_tercile, second_tercile, dof):
        """
        Weibull minimum-based method.
        """
        n_time = len(best_guess)
        pred_prob = np.empty((3, n_time))
        if np.all(np.isnan(best_guess)):
            pred_prob[:] = np.nan
        else:
            error_std = np.sqrt(error_variance)
            pred_prob[0, :] = weibull_min.cdf(first_tercile, c=dof, loc=best_guess, scale=error_std)
            pred_prob[1, :] = weibull_min.cdf(second_tercile, c=dof, loc=best_guess, scale=error_std) - \
                              weibull_min.cdf(first_tercile, c=dof, loc=best_guess, scale=error_std)
            pred_prob[2, :] = 1 - weibull_min.cdf(second_tercile, c=dof, loc=best_guess, scale=error_std)
        return pred_prob

    @staticmethod
    def calculate_tercile_probabilities_gamma(best_guess, error_variance, T1, T2, dof=None):
        """Gamma-distribution based method."""
        n_time = len(best_guess)
        pred_prob = np.empty((3, n_time), dtype=float)
        if np.any(np.isnan(best_guess)) or np.any(np.isnan(error_variance)):
            pred_prob[:] = np.nan
            return pred_prob
        best_guess = np.asarray(best_guess, dtype=float)
        error_variance = np.asarray(error_variance, dtype=float)
        T1 = np.asarray(T1, dtype=float)
        T2 = np.asarray(T2, dtype=float)
        alpha = (best_guess ** 2) / error_variance
        theta = error_variance / best_guess
        cdf_t1 = gamma.cdf(T1, a=alpha, scale=theta)
        cdf_t2 = gamma.cdf(T2, a=alpha, scale=theta)
        pred_prob[0, :] = cdf_t1
        pred_prob[1, :] = cdf_t2 - cdf_t1
        pred_prob[2, :] = 1.0 - cdf_t2
        return pred_prob

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

    @staticmethod
    def calculate_tercile_probabilities_normal(best_guess, error_variance, first_tercile, second_tercile):
        """Normal-distribution based method."""
        n_time = len(best_guess)
        pred_prob = np.empty((3, n_time))
        if np.all(np.isnan(best_guess)):
            pred_prob[:] = np.nan
        else:
            error_std = np.sqrt(error_variance)
            pred_prob[0, :] = norm.cdf(first_tercile, loc=best_guess, scale=error_std)
            pred_prob[1, :] = norm.cdf(second_tercile, loc=best_guess, scale=error_std) - norm.cdf(first_tercile, loc=best_guess, scale=error_std)
            pred_prob[2, :] = 1 - norm.cdf(second_tercile, loc=best_guess, scale=error_std)
        return pred_prob

    @staticmethod
    def calculate_tercile_probabilities_lognormal(best_guess, error_variance, first_tercile, second_tercile):
        """Lognormal-distribution based method."""
        n_time = len(best_guess)
        pred_prob = np.empty((3, n_time))
        if np.any(np.isnan(best_guess)) or np.any(np.isnan(error_variance)):
            pred_prob[:] = np.nan
            return pred_prob
        sigma = np.sqrt(np.log(1 + error_variance / (best_guess ** 2)))
        mu = np.log(best_guess) - sigma ** 2 / 2
        pred_prob[0, :] = lognorm.cdf(first_tercile, s=sigma, scale=np.exp(mu))
        pred_prob[1, :] = lognorm.cdf(second_tercile, s=sigma, scale=np.exp(mu)) - lognorm.cdf(first_tercile, s=sigma, scale=np.exp(mu))
        pred_prob[2, :] = 1 - lognorm.cdf(second_tercile, s=sigma, scale=np.exp(mu))
        return pred_prob

    def compute_prob(self, Predictant, clim_year_start, clim_year_end, hindcast_det):
        """
        Compute tercile probabilities using the chosen distribution method.
        Predictant is expected to be an xarray DataArray with dims (T, Y, X).
        """
        if "M" in Predictant.coords:
            Predictant = Predictant.isel(M=0).drop_vars('M').squeeze()
        mask = xr.where(~np.isnan(Predictant.isel(T=0)), 1, np.nan).drop_vars(['T']).squeeze().to_numpy()
        Predictant = Predictant.transpose('T', 'Y', 'X')
        index_start = Predictant.get_index("T").get_loc(str(clim_year_start)).start
        index_end = Predictant.get_index("T").get_loc(str(clim_year_end)).stop
        rainfall_for_tercile = Predictant.isel(T=slice(index_start, index_end))
        terciles = rainfall_for_tercile.quantile([0.33, 0.67], dim='T')
        error_variance = (Predictant - hindcast_det).var(dim='T')
        dof = len(Predictant.get_index("T")) - 2
        if self.dist_method == "t":
            calc_func = self.calculate_tercile_probabilities
            hindcast_prob = xr.apply_ufunc(
                calc_func,
                hindcast_det,
                error_variance,
                terciles.isel(quantile=0).drop_vars('quantile'),
                terciles.isel(quantile=1).drop_vars('quantile'),
                input_core_dims=[('T',), (), (), ()],
                vectorize=True,
                kwargs={'dof': dof},
                dask='parallelized',
                output_core_dims=[('probability', 'T')],
                output_dtypes=['float'],
                dask_gufunc_kwargs={'output_sizes': {'probability': 3}, "allow_rechunk": True}
            )
        elif self.dist_method == "weibull_min":
            calc_func = self.calculate_tercile_probabilities_weibull_min
            hindcast_prob = xr.apply_ufunc(
                calc_func,
                hindcast_det,
                error_variance,
                terciles.isel(quantile=0).drop_vars('quantile'),
                terciles.isel(quantile=1).drop_vars('quantile'),
                input_core_dims=[('T',), (), (), ()],
                vectorize=True,
                kwargs={'dof': dof},
                dask='parallelized',
                output_core_dims=[('probability', 'T')],
                output_dtypes=['float'],
                dask_gufunc_kwargs={'output_sizes': {'probability': 3}, "allow_rechunk": True}
            )
        elif self.dist_method == "gamma":
            calc_func = self.calculate_tercile_probabilities_gamma
            hindcast_prob = xr.apply_ufunc(
                calc_func,
                hindcast_det,
                error_variance,
                terciles.isel(quantile=0).drop_vars('quantile'),
                terciles.isel(quantile=1).drop_vars('quantile'),
                input_core_dims=[('T',), (), (), ()],
                vectorize=True,
                dask='parallelized',
                output_core_dims=[('probability', 'T')],
                output_dtypes=['float'],
                dask_gufunc_kwargs={'output_sizes': {'probability': 3}, "allow_rechunk": True}
            )
        elif self.dist_method == "nonparam":
            calc_func = self.calculate_tercile_probabilities_nonparametric
            error_samples = Predictant - hindcast_det
            hindcast_prob = xr.apply_ufunc(
                calc_func,
                hindcast_det,
                error_samples,
                terciles.isel(quantile=0).drop_vars('quantile'),
                terciles.isel(quantile=1).drop_vars('quantile'),
                input_core_dims=[('T',), ('T',), (), ()],
                output_core_dims=[('probability', 'T')],
                vectorize=True,
                dask='parallelized',
                output_dtypes=[float],
                dask_gufunc_kwargs={'output_sizes': {'probability': 3}, "allow_rechunk": True}
            )
        elif self.dist_method == "normal":
            calc_func = self.calculate_tercile_probabilities_normal
            hindcast_prob = xr.apply_ufunc(
                calc_func,
                hindcast_det,
                error_variance,
                terciles.isel(quantile=0).drop_vars('quantile'),
                terciles.isel(quantile=1).drop_vars('quantile'),
                input_core_dims=[('T',), (), (), ()],
                vectorize=True,
                dask='parallelized',
                output_core_dims=[('probability', 'T')],
                output_dtypes=['float'],
                dask_gufunc_kwargs={'output_sizes': {'probability': 3}, "allow_rechunk": True}
            )
        elif self.dist_method == "lognormal":
            calc_func = self.calculate_tercile_probabilities_lognormal
            hindcast_prob = xr.apply_ufunc(
                calc_func,
                hindcast_det,
                error_variance,
                terciles.isel(quantile=0).drop_vars('quantile'),
                terciles.isel(quantile=1).drop_vars('quantile'),
                input_core_dims=[('T',), (), (), ()],
                vectorize=True,
                dask='parallelized',
                output_core_dims=[('probability', 'T')],
                output_dtypes=['float'],
                dask_gufunc_kwargs={'output_sizes': {'probability': 3}, "allow_rechunk": True}
            )
        else:
            raise ValueError(f"Invalid dist_method: {self.dist_method}.")
        hindcast_prob = hindcast_prob.assign_coords(probability=('probability', ['PB', 'PN', 'PA']))
        return hindcast_prob.transpose('probability', 'T', 'Y', 'X')

    @staticmethod
    def _reshape_and_filter_data(da):
        """
        Helper: stack the DataArray from (T, Y, X[, M]) to (n_samples, n_features)
        and remove rows containing NaNs.
        """
        da_stacked = da.stack(sample=('T', 'Y', 'X'))
        if 'M' in da.dims:
            da_stacked = da_stacked.transpose('sample', 'M')
        else:
            da_stacked = da_stacked.transpose('sample')
        da_values = da_stacked.values
        nan_mask = np.any(np.isnan(da_values), axis=1)
        return da_values[~nan_mask], nan_mask, da_values

    def forecast(self, Predictant, clim_year_start, clim_year_end, hindcast_det, hindcast_det_cross, Predictor_for_year, best_params=None, cluster_da=None):
        """
        Generate deterministic and probabilistic forecast for a target year using the stacking ensemble.
        Parameters
        ----------
        Predictant : xarray.DataArray
            Observed predictand data with dimensions (T, Y, X) or (T, M, Y, X).
        clim_year_start : int or str
            Start year of the climatology period.
        clim_year_end : int or str
            End year of the climatology period.
        hindcast_det : xarray.DataArray
            Deterministic hindcast data for training with dimensions (T, M, Y, X).
        hindcast_det_cross : xarray.DataArray
            Deterministic hindcast data for error estimation with dimensions (T, Y, X).
        Predictor_for_year : xarray.DataArray
            Predictor data for the target year with dimensions (T, M, Y, X).
        best_params : dict, optional
            Pre-computed best hyperparameters per cluster. If None, computes internally.
        cluster_da : xarray.DataArray, optional
            Pre-computed cluster labels. If None, computes internally.
        Returns
        -------
        forecast_det : xarray.DataArray
            Deterministic forecast with dimensions (T, Y, X).
        forecast_prob : xarray.DataArray
            Tercile probabilities with dimensions (probability, T, Y, X).
        """
        if "M" in Predictant.coords:
            Predictant_no_m = Predictant.isel(M=0).drop_vars('M').squeeze()
        else:
            Predictant_no_m = Predictant
        mask = xr.where(~np.isnan(Predictant_no_m.isel(T=0)), 1, np.nan).drop_vars(['T']).squeeze().to_numpy()
        # Standardize Predictor_for_year using hindcast climatology
        mean_val = hindcast_det.sel(T=slice(str(clim_year_start), str(clim_year_end))).mean(dim='T')
        std_val = hindcast_det.sel(T=slice(str(clim_year_start), str(clim_year_end))).std(dim='T')
        Predictor_for_year_st = (Predictor_for_year - mean_val) / std_val
        hindcast_det_st = standardize_timeseries(hindcast_det, clim_year_start, clim_year_end)
        Predictant_st = standardize_timeseries(Predictant_no_m, clim_year_start, clim_year_end)
        hindcast_det_st['T'] = Predictant_st['T']
        
        # Extract coordinates from X_test
        time = Predictor_for_year_st['T']
        lat = Predictor_for_year_st['Y']
        lon = Predictor_for_year_st['X']
        n_time = len(time)
        n_lat = len(lat)
        n_lon = len(lon)
        # Use provided best_params and cluster_da or compute if None
        if best_params is None:
            best_params, cluster_da = self.compute_hyperparameters(hindcast_det, Predictant_no_m, clim_year_start, clim_year_end)
        # Initialize predictions array
        predictions = np.full((n_time, n_lat, n_lon), np.nan)
        self.stacking_model = {}  # Dictionary to store models per cluster
        for c in range(self.n_clusters):
            if c not in best_params:
                continue
            bp = best_params[c]
            # Mask for this cluster
            mask_3d_train = (cluster_da == c).expand_dims({'T': hindcast_det_st['T']})
            mask_3d_test = (cluster_da == c).expand_dims({'T': Predictor_for_year_st['T']})
            # Stack training data for cluster
            X_train_stacked_c = hindcast_det_st.where(mask_3d_train).stack(sample=('T', 'Y', 'X')).transpose('sample', 'M').values
            y_train_stacked_c = Predictant_st.where(mask_3d_train).stack(sample=('T', 'Y', 'X')).values.ravel()
            train_nan_mask = np.any(~np.isfinite(X_train_stacked_c), axis=1) | ~np.isfinite(y_train_stacked_c)
            X_train_clean_c = X_train_stacked_c[~train_nan_mask]
            y_train_clean_c = y_train_stacked_c[~train_nan_mask]
            # Stack testing data for cluster
            X_test_stacked_c = Predictor_for_year_st.where(mask_3d_test).stack(sample=('T', 'Y', 'X')).transpose('sample', 'M').values
            test_nan_mask = np.any(~np.isfinite(X_test_stacked_c), axis=1)
            X_test_clean_c = X_test_stacked_c[~test_nan_mask]
            # Initialize base model (XGBoost)
            xgb = XGBRegressor(**bp['xgb'], random_state=self.random_state, verbosity=0)
            # Initialize meta-model (MLPRegressor)
            meta_model = MLPRegressor(**bp['mlp'], max_iter=self.max_iter, random_state=self.random_state)
            # Initialize stacking ensemble for this cluster
            stacking_model_c = StackingRegressor(estimators=[('xgb', xgb)], final_estimator=meta_model)
            # Fit the stacking model on cluster data
            stacking_model_c.fit(X_train_clean_c, y_train_clean_c)
            self.stacking_model[c] = stacking_model_c
            # Predict
            y_pred_c = stacking_model_c.predict(X_test_clean_c)
            # Reconstruct predictions for this cluster
            result_c = np.full(len(X_test_stacked_c), np.nan)
            result_c[~test_nan_mask] = y_pred_c
            pred_c_reshaped = result_c.reshape(n_time, n_lat, n_lon)
            # Fill in the predictions array
            predictions = np.where(np.isnan(predictions), pred_c_reshaped, predictions)
        result_da = xr.DataArray(
            data=predictions,
            coords={'T': time, 'Y': lat, 'X': lon},
            dims=['T', 'Y', 'X']
        ) * mask
        result_da = reverse_standardize(result_da, Predictant_no_m, clim_year_start, clim_year_end)
        year = Predictor_for_year.coords['T'].values.astype('datetime64[Y]').astype(int)[0] + 1970
        T_value_1 = Predictant_no_m.isel(T=0).coords['T'].values
        month_1 = T_value_1.astype('datetime64[M]').astype(int) % 12 + 1
        new_T_value = np.datetime64(f"{year}-{month_1:02d}-01")
        result_da = result_da.assign_coords(T=xr.DataArray([new_T_value], dims=["T"]))
        result_da['T'] = result_da['T'].astype('datetime64[ns]')
        # Compute tercile probabilities
        index_start = Predictant_no_m.get_index("T").get_loc(str(clim_year_start)).start
        index_end = Predictant_no_m.get_index("T").get_loc(str(clim_year_end)).stop
        rainfall_for_tercile = Predictant_no_m.isel(T=slice(index_start, index_end))
        terciles = rainfall_for_tercile.quantile([0.33, 0.67], dim='T')
        error_variance = (Predictant_no_m - hindcast_det_cross).var(dim='T')
        dof = len(Predictant_no_m.get_index("T")) - 2
        if self.dist_method == "t":
            calc_func = self.calculate_tercile_probabilities
            hindcast_prob = xr.apply_ufunc(
                calc_func,
                result_da,
                error_variance,
                terciles.isel(quantile=0).drop_vars('quantile'),
                terciles.isel(quantile=1).drop_vars('quantile'),
                input_core_dims=[('T',), (), (), ()],
                vectorize=True,
                kwargs={'dof': dof},
                dask='parallelized',
                output_core_dims=[('probability', 'T')],
                output_dtypes=['float'],
                dask_gufunc_kwargs={'output_sizes': {'probability': 3}, "allow_rechunk": True}
            )
        elif self.dist_method == "weibull_min":
            calc_func = self.calculate_tercile_probabilities_weibull_min
            hindcast_prob = xr.apply_ufunc(
                calc_func,
                result_da,
                error_variance,
                terciles.isel(quantile=0).drop_vars('quantile'),
                terciles.isel(quantile=1).drop_vars('quantile'),
                input_core_dims=[('T',), (), (), ()],
                vectorize=True,
                kwargs={'dof': dof},
                dask='parallelized',
                output_core_dims=[('probability', 'T')],
                output_dtypes=['float'],
                dask_gufunc_kwargs={'output_sizes': {'probability': 3}, "allow_rechunk": True}
            )
        elif self.dist_method == "gamma":
            calc_func = self.calculate_tercile_probabilities_gamma
            hindcast_prob = xr.apply_ufunc(
                calc_func,
                result_da,
                error_variance,
                terciles.isel(quantile=0).drop_vars('quantile'),
                terciles.isel(quantile=1).drop_vars('quantile'),
                input_core_dims=[('T',), (), (), ()],
                vectorize=True,
                dask='parallelized',
                output_core_dims=[('probability', 'T')],
                output_dtypes=['float'],
                dask_gufunc_kwargs={'output_sizes': {'probability': 3}, "allow_rechunk": True}
            )
        elif self.dist_method == "normal":
            calc_func = self.calculate_tercile_probabilities_normal
            hindcast_prob = xr.apply_ufunc(
                calc_func,
                result_da,
                error_variance,
                terciles.isel(quantile=0).drop_vars('quantile'),
                terciles.isel(quantile=1).drop_vars('quantile'),
                input_core_dims=[('T',), (), (), ()],
                vectorize=True,
                dask='parallelized',
                output_core_dims=[('probability', 'T')],
                output_dtypes=['float'],
                dask_gufunc_kwargs={'output_sizes': {'probability': 3}, "allow_rechunk": True}
            )
        elif self.dist_method == "lognormal":
            calc_func = self.calculate_tercile_probabilities_lognormal
            hindcast_prob = xr.apply_ufunc(
                calc_func,
                result_da,
                error_variance,
                terciles.isel(quantile=0).drop_vars('quantile'),
                terciles.isel(quantile=1).drop_vars('quantile'),
                input_core_dims=[('T',), (), (), ()],
                vectorize=True,
                dask='parallelized',
                output_core_dims=[('probability', 'T')],
                output_dtypes=['float'],
                dask_gufunc_kwargs={'output_sizes': {'probability': 3}, "allow_rechunk": True}
            )
        elif self.dist_method == "nonparam":
            calc_func = self.calculate_tercile_probabilities_nonparametric
            error_samples = Predictant_no_m - hindcast_det_cross
            error_samples = error_samples.rename({'T': 'S'})
            hindcast_prob = xr.apply_ufunc(
                calc_func,
                result_da,
                error_samples,
                terciles.isel(quantile=0).drop_vars('quantile'),
                terciles.isel(quantile=1).drop_vars('quantile'),
                input_core_dims=[('T',), ('S',), (), ()],
                output_core_dims=[('probability', 'T')],
                vectorize=True,
                dask='parallelized',
                output_dtypes=[float],
                dask_gufunc_kwargs={'output_sizes': {'probability': 3}, "allow_rechunk": True}
            )
        else:
            raise ValueError(f"Invalid dist_method: {self.dist_method}.")
        hindcast_prob = hindcast_prob.assign_coords(probability=('probability', ['PB', 'PN', 'PA']))
        return result_da * mask, mask * hindcast_prob.transpose('probability', 'T', 'Y', 'X')


class WAS_mme_LightGBM:
    """
    LightGBM-based Multi-Model Ensemble (MME) forecasting.
    This class implements a single-model forecasting approach using LightGBM's LGBMRegressor
    for deterministic predictions, with optional tercile probability calculations using
    various statistical distributions. Implements hyperparameter optimization via randomized search.
    Parameters
    ----------
    n_estimators_range : list of int, optional
        List of n_estimators values to tune (default is [50, 100, 200, 300]).
    learning_rate_range : list of float, optional
        List of learning rates to tune (default is [0.01, 0.05, 0.1, 0.2]).
    max_depth_range : list of int, optional
        List of max depths to tune (default is [3, 5, 7, 9]).
    num_leaves_range : list of int, optional
        List of num_leaves to tune (default is [31, 50, 100]).
    subsample_range : list of float, optional
        List of subsample ratios to tune (default is [0.6, 0.8, 1.0]).
    colsample_bytree_range : list of float, optional
        List of column sampling ratios to tune (default is [0.6, 0.8, 1.0]).
    random_state : int, optional
        Seed for reproducibility (default is 42).
    dist_method : str, optional
        Distribution method for tercile probabilities ('t', 'gamma', etc.) (default is 'gamma').
    n_iter_search : int, optional
        Number of iterations for randomized search (default is 10).
    cv_folds : int, optional
        Number of cross-validation folds (default is 3).
    n_clusters : int, optional
        Number of clusters for homogenized zones (default is 4).
    """
    def __init__(self,
                 n_estimators_range=[50, 100, 200, 300],
                 learning_rate_range=[0.01, 0.05, 0.1, 0.2],
                 max_depth_range=[3, 5, 7, 9],
                 num_leaves_range=[31, 50, 100],
                 subsample_range=[0.6, 0.8, 1.0],
                 colsample_bytree_range=[0.6, 0.8, 1.0],
                 random_state=42,
                 dist_method="gamma",
                 n_iter_search=10,
                 cv_folds=3,
                 n_clusters=4):
        self.n_estimators_range = n_estimators_range
        self.learning_rate_range = learning_rate_range
        self.max_depth_range = max_depth_range
        self.num_leaves_range = num_leaves_range
        self.subsample_range = subsample_range
        self.colsample_bytree_range = colsample_bytree_range
        self.random_state = random_state
        self.dist_method = dist_method
        self.n_iter_search = n_iter_search
        self.cv_folds = cv_folds
        self.n_clusters = n_clusters
        self.lgbm = None

    def compute_hyperparameters(self, Predictors, Predictand, clim_year_start, clim_year_end):
        """
        Independently computes the best hyperparameters using randomized search on stacked training data for each homogenized zone.
        Parameters
        ----------
        Predictors : xarray.DataArray
            Training predictor data with dimensions (T, M, Y, X).
        Predictand : xarray.DataArray
            Training predictand data with dimensions (T, Y, X).
        clim_year_start : int
            Start year of the climatology period.
        clim_year_end : int
            End year of the climatology period.
        Returns
        -------
        best_params_dict : dict
            Best hyperparameters for each cluster.
        cluster_da : xarray.DataArray
            Cluster labels with dimensions (Y, X).
        """
        if "M" in Predictand.coords:
            Predictand = Predictand.isel(M=0).drop_vars('M').squeeze()
           
        X_train_std = standardize_timeseries(Predictors, clim_year_start, clim_year_end)
        Predictand.name = "varname"
        # Step 1: Perform KMeans clustering based on predictand's spatial distribution
        kmeans = KMeans(n_clusters=self.n_clusters, random_state=self.random_state)
        Predictand_dropna = Predictand.to_dataframe().reset_index().dropna().drop(columns=['T'])
        variable_column = Predictand_dropna.columns[2]
        Predictand_dropna['cluster'] = kmeans.fit_predict(
           Predictand_dropna[[variable_column]]
        )
       
        # Convert cluster assignments back into an xarray structure
        df_unique = Predictand_dropna.drop_duplicates(subset=['Y', 'X'])
        dataset = df_unique.set_index(['Y', 'X']).to_xarray()
        mask = xr.where(~np.isnan(Predictand.isel(T=0)), 1, np.nan)
        Cluster = (dataset['cluster'] * mask)
              
        # Align cluster array with the predictand array
        xarray1, xarray2 = xr.align(Predictand, Cluster, join="outer")
       
        # Identify unique cluster labels
        clusters = np.unique(xarray2)
        clusters = clusters[~np.isnan(clusters)]
        cluster_da = xarray2
        y_train_std = standardize_timeseries(Predictand, clim_year_start, clim_year_end)
        X_train_std['T'] = y_train_std['T']
        # Prepare parameter distributions
        param_dist = {
            'n_estimators': self.n_estimators_range,
            'learning_rate': self.learning_rate_range,
            'max_depth': self.max_depth_range,
            'num_leaves': self.num_leaves_range,
            'subsample': self.subsample_range,
            'colsample_bytree': self.colsample_bytree_range
        }
        best_params_dict = {}
        for c in clusters:
            mask_3d = (cluster_da == c).expand_dims({'T': y_train_std['T']})
            X_stacked_c = X_train_std.where(mask_3d).stack(sample=('T', 'Y', 'X')).transpose('sample', 'M').values
            y_stacked_c = y_train_std.where(mask_3d).stack(sample=('T', 'Y', 'X')).values.ravel()
            nan_mask_c = np.any(~np.isfinite(X_stacked_c), axis=1) | ~np.isfinite(y_stacked_c)
            X_clean_c = X_stacked_c[~nan_mask_c]
            y_clean_c = y_stacked_c[~nan_mask_c]
            if len(X_clean_c) == 0:
                continue
            # Initialize LGBMRegressor base model
            model = LGBMRegressor(random_state=self.random_state, verbosity=-1)
            # Randomized search
            random_search = RandomizedSearchCV(
                model, param_distributions=param_dist, n_iter=self.n_iter_search,
                cv=self.cv_folds, scoring='neg_mean_squared_error',
                random_state=self.random_state, error_score=np.nan
            )
            random_search.fit(X_clean_c, y_clean_c)
            best_params_dict[c] = random_search.best_params_
        return best_params_dict, cluster_da

    def compute_model(self, X_train, y_train, X_test, y_test, best_params=None, cluster_da=None):
        """
        Compute deterministic hindcast using the LGBMRegressor model with injected hyperparameters for each zone.
        Parameters
        ----------
        X_train : xarray.DataArray
            Training predictor data with dimensions (T, M, Y, X).
        y_train : xarray.DataArray
            Training predictand data with dimensions (T, Y, X).
        X_test : xarray.DataArray
            Testing predictor data with dimensions (T, M, Y, X).
        y_test : xarray.DataArray
            Testing predictand data with dimensions (T, Y, X).
        clim_year_start : int
            Start year of the climatology period.
        clim_year_end : int
            End year of the climatology period.
        best_params : dict, optional
            Pre-computed best hyperparameters per cluster. If None, computes internally.
        cluster_da : xarray.DataArray, optional
            Pre-computed cluster labels. If None, computes internally.
        Returns
        -------
        predicted_da : xarray.DataArray
            Deterministic hindcast with dimensions (T, Y, X).
        """
        # Standardize inputs
        X_train_std = X_train
        y_train_std = y_train
        X_test_std = X_test
        y_test_std = y_test
        # Extract coordinate variables from X_test
        time = X_test_std['T']
        lat = X_test_std['Y']
        lon = X_test_std['X']
        n_time = len(time)
        n_lat = len(lat)
        n_lon = len(lon)
        # Use provided best_params and cluster_da or compute if None
        if best_params is None:
            best_params, cluster_da = self.compute_hyperparameters(X_train, y_train, clim_year_start, clim_year_end)
        # Initialize predictions array
        predictions = np.full((n_time, n_lat, n_lon), np.nan)
        self.lgbm = {}  # Dictionary to store models per cluster
        for c in range(self.n_clusters):
            if c not in best_params:
                continue
            bp = best_params[c]
            # Mask for this cluster
            mask_3d_train = (cluster_da == c).expand_dims({'T': X_train_std['T']})
            mask_3d_test = (cluster_da == c).expand_dims({'T': X_test_std['T']})
            # Stack training data for cluster
            X_train_stacked_c = X_train_std.where(mask_3d_train).stack(sample=('T', 'Y', 'X')).transpose('sample', 'M').values
            y_train_stacked_c = y_train_std.where(mask_3d_train).stack(sample=('T', 'Y', 'X')).values.ravel()
            train_nan_mask = np.any(~np.isfinite(X_train_stacked_c), axis=1) | ~np.isfinite(y_train_stacked_c)
            X_train_clean_c = X_train_stacked_c[~train_nan_mask]
            y_train_clean_c = y_train_stacked_c[~train_nan_mask]
            # Stack testing data for cluster
            X_test_stacked_c = X_test_std.where(mask_3d_test).stack(sample=('T', 'Y', 'X')).transpose('sample', 'M').values
            y_test_stacked_c = y_test_std.where(mask_3d_test).stack(sample=('T', 'Y', 'X')).values.ravel()
            test_nan_mask = np.any(~np.isfinite(X_test_stacked_c), axis=1) | ~np.isfinite(y_test_stacked_c)
            X_test_clean_c = X_test_stacked_c[~test_nan_mask]
            # Initialize the model with best parameters for this cluster
            lgbm_c = LGBMRegressor(
                n_estimators=bp['n_estimators'],
                learning_rate=bp['learning_rate'],
                max_depth=bp['max_depth'],
                num_leaves=bp['num_leaves'],
                subsample=bp['subsample'],
                colsample_bytree=bp['colsample_bytree'],
                random_state=self.random_state,
                verbosity=-1
            )
            # Fit and predict
            lgbm_c.fit(X_train_clean_c, y_train_clean_c)
            self.lgbm[c] = lgbm_c
            y_pred_c = lgbm_c.predict(X_test_clean_c)
            # Reconstruct predictions for this cluster
            result_c = np.full(len(y_test_stacked_c), np.nan)
            result_c[~test_nan_mask] = y_pred_c
            pred_c_reshaped = result_c.reshape(n_time, n_lat, n_lon)
            # Fill in the predictions array
            predictions = np.where(np.isnan(predictions), pred_c_reshaped, predictions)
        predicted_da = xr.DataArray(
            data=predictions,
            coords={'T': time, 'Y': lat, 'X': lon},
            dims=['T', 'Y', 'X']
        )
        return predicted_da

    # ------------------ Probability Calculation Methods ------------------
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
            pred_prob[0, :] = t.cdf(first_t, df=dof)
            pred_prob[1, :] = t.cdf(second_t, df=dof) - t.cdf(first_t, df=dof)
            pred_prob[2, :] = 1 - t.cdf(second_t, df=dof)
        return pred_prob

    @staticmethod
    def calculate_tercile_probabilities_weibull_min(best_guess, error_variance, first_tercile, second_tercile, dof):
        """
        Weibull minimum-based method.
        """
        n_time = len(best_guess)
        pred_prob = np.empty((3, n_time))
        if np.all(np.isnan(best_guess)):
            pred_prob[:] = np.nan
        else:
            error_std = np.sqrt(error_variance)
            pred_prob[0, :] = weibull_min.cdf(first_tercile, c=dof, loc=best_guess, scale=error_std)
            pred_prob[1, :] = weibull_min.cdf(second_tercile, c=dof, loc=best_guess, scale=error_std) - \
                              weibull_min.cdf(first_tercile, c=dof, loc=best_guess, scale=error_std)
            pred_prob[2, :] = 1 - weibull_min.cdf(second_tercile, c=dof, loc=best_guess, scale=error_std)
        return pred_prob

    @staticmethod
    def calculate_tercile_probabilities_gamma(best_guess, error_variance, T1, T2, dof=None):
        """Gamma-distribution based method."""
        n_time = len(best_guess)
        pred_prob = np.empty((3, n_time), dtype=float)
        if np.any(np.isnan(best_guess)) or np.any(np.isnan(error_variance)):
            pred_prob[:] = np.nan
            return pred_prob
        best_guess = np.asarray(best_guess, dtype=float)
        error_variance = np.asarray(error_variance, dtype=float)
        T1 = np.asarray(T1, dtype=float)
        T2 = np.asarray(T2, dtype=float)
        alpha = (best_guess ** 2) / error_variance
        theta = error_variance / best_guess
        cdf_t1 = gamma.cdf(T1, a=alpha, scale=theta)
        cdf_t2 = gamma.cdf(T2, a=alpha, scale=theta)
        pred_prob[0, :] = cdf_t1
        pred_prob[1, :] = cdf_t2 - cdf_t1
        pred_prob[2, :] = 1.0 - cdf_t2
        return pred_prob

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

    @staticmethod
    def calculate_tercile_probabilities_normal(best_guess, error_variance, first_tercile, second_tercile):
        """Normal-distribution based method."""
        n_time = len(best_guess)
        pred_prob = np.empty((3, n_time))
        if np.all(np.isnan(best_guess)):
            pred_prob[:] = np.nan
        else:
            error_std = np.sqrt(error_variance)
            pred_prob[0, :] = norm.cdf(first_tercile, loc=best_guess, scale=error_std)
            pred_prob[1, :] = norm.cdf(second_tercile, loc=best_guess, scale=error_std) - norm.cdf(first_tercile, loc=best_guess, scale=error_std)
            pred_prob[2, :] = 1 - norm.cdf(second_tercile, loc=best_guess, scale=error_std)
        return pred_prob

    @staticmethod
    def calculate_tercile_probabilities_lognormal(best_guess, error_variance, first_tercile, second_tercile):
        """Lognormal-distribution based method."""
        n_time = len(best_guess)
        pred_prob = np.empty((3, n_time))
        if np.any(np.isnan(best_guess)) or np.any(np.isnan(error_variance)):
            pred_prob[:] = np.nan
            return pred_prob
        sigma = np.sqrt(np.log(1 + error_variance / (best_guess ** 2)))
        mu = np.log(best_guess) - sigma ** 2 / 2
        pred_prob[0, :] = lognorm.cdf(first_tercile, s=sigma, scale=np.exp(mu))
        pred_prob[1, :] = lognorm.cdf(second_tercile, s=sigma, scale=np.exp(mu)) - lognorm.cdf(first_tercile, s=sigma, scale=np.exp(mu))
        pred_prob[2, :] = 1 - lognorm.cdf(second_tercile, s=sigma, scale=np.exp(mu))
        return pred_prob

    def compute_prob(self, Predictant, clim_year_start, clim_year_end, hindcast_det):
        """
        Compute tercile probabilities using the chosen distribution method.
        Predictant is expected to be an xarray DataArray with dims (T, Y, X).
        """
        if "M" in Predictant.coords:
            Predictant = Predictant.isel(M=0).drop_vars('M').squeeze()
        mask = xr.where(~np.isnan(Predictant.isel(T=0)), 1, np.nan).drop_vars(['T']).squeeze().to_numpy()
        Predictant = Predictant.transpose('T', 'Y', 'X')
        index_start = Predictant.get_index("T").get_loc(str(clim_year_start)).start
        index_end = Predictant.get_index("T").get_loc(str(clim_year_end)).stop
        rainfall_for_tercile = Predictant.isel(T=slice(index_start, index_end))
        terciles = rainfall_for_tercile.quantile([0.33, 0.67], dim='T')
        error_variance = (Predictant - hindcast_det).var(dim='T')
        dof = len(Predictant.get_index("T")) - 2
        if self.dist_method == "t":
            calc_func = self.calculate_tercile_probabilities
            hindcast_prob = xr.apply_ufunc(
                calc_func,
                hindcast_det,
                error_variance,
                terciles.isel(quantile=0).drop_vars('quantile'),
                terciles.isel(quantile=1).drop_vars('quantile'),
                input_core_dims=[('T',), (), (), ()],
                vectorize=True,
                kwargs={'dof': dof},
                dask='parallelized',
                output_core_dims=[('probability', 'T')],
                output_dtypes=['float'],
                dask_gufunc_kwargs={'output_sizes': {'probability': 3}, "allow_rechunk": True}
            )
        elif self.dist_method == "weibull_min":
            calc_func = self.calculate_tercile_probabilities_weibull_min
            hindcast_prob = xr.apply_ufunc(
                calc_func,
                hindcast_det,
                error_variance,
                terciles.isel(quantile=0).drop_vars('quantile'),
                terciles.isel(quantile=1).drop_vars('quantile'),
                input_core_dims=[('T',), (), (), ()],
                vectorize=True,
                kwargs={'dof': dof},
                dask='parallelized',
                output_core_dims=[('probability', 'T')],
                output_dtypes=['float'],
                dask_gufunc_kwargs={'output_sizes': {'probability': 3}, "allow_rechunk": True}
            )
        elif self.dist_method == "gamma":
            calc_func = self.calculate_tercile_probabilities_gamma
            hindcast_prob = xr.apply_ufunc(
                calc_func,
                hindcast_det,
                error_variance,
                terciles.isel(quantile=0).drop_vars('quantile'),
                terciles.isel(quantile=1).drop_vars('quantile'),
                input_core_dims=[('T',), (), (), ()],
                vectorize=True,
                dask='parallelized',
                output_core_dims=[('probability', 'T')],
                output_dtypes=['float'],
                dask_gufunc_kwargs={'output_sizes': {'probability': 3}, "allow_rechunk": True}
            )
        elif self.dist_method == "nonparam":
            calc_func = self.calculate_tercile_probabilities_nonparametric
            error_samples = Predictant - hindcast_det
            hindcast_prob = xr.apply_ufunc(
                calc_func,
                hindcast_det,
                error_samples,
                terciles.isel(quantile=0).drop_vars('quantile'),
                terciles.isel(quantile=1).drop_vars('quantile'),
                input_core_dims=[('T',), ('T',), (), ()],
                output_core_dims=[('probability', 'T')],
                vectorize=True,
                dask='parallelized',
                output_dtypes=[float],
                dask_gufunc_kwargs={'output_sizes': {'probability': 3}, "allow_rechunk": True}
            )
        elif self.dist_method == "normal":
            calc_func = self.calculate_tercile_probabilities_normal
            hindcast_prob = xr.apply_ufunc(
                calc_func,
                hindcast_det,
                error_variance,
                terciles.isel(quantile=0).drop_vars('quantile'),
                terciles.isel(quantile=1).drop_vars('quantile'),
                input_core_dims=[('T',), (), (), ()],
                vectorize=True,
                dask='parallelized',
                output_core_dims=[('probability', 'T')],
                output_dtypes=['float'],
                dask_gufunc_kwargs={'output_sizes': {'probability': 3}, "allow_rechunk": True}
            )
        elif self.dist_method == "lognormal":
            calc_func = self.calculate_tercile_probabilities_lognormal
            hindcast_prob = xr.apply_ufunc(
                calc_func,
                hindcast_det,
                error_variance,
                terciles.isel(quantile=0).drop_vars('quantile'),
                terciles.isel(quantile=1).drop_vars('quantile'),
                input_core_dims=[('T',), (), (), ()],
                vectorize=True,
                dask='parallelized',
                output_core_dims=[('probability', 'T')],
                output_dtypes=['float'],
                dask_gufunc_kwargs={'output_sizes': {'probability': 3}, "allow_rechunk": True}
            )
        else:
            raise ValueError(f"Invalid dist_method: {self.dist_method}.")
        hindcast_prob = hindcast_prob.assign_coords(probability=('probability', ['PB', 'PN', 'PA']))
        return hindcast_prob.transpose('probability', 'T', 'Y', 'X')

    @staticmethod
    def _reshape_and_filter_data(da):
        """
        Helper: stack the DataArray from (T, Y, X[, M]) to (n_samples, n_features)
        and remove rows containing NaNs.
        """
        da_stacked = da.stack(sample=('T', 'Y', 'X'))
        if 'M' in da.dims:
            da_stacked = da_stacked.transpose('sample', 'M')
        else:
            da_stacked = da_stacked.transpose('sample')
        da_values = da_stacked.values
        nan_mask = np.any(np.isnan(da_values), axis=1)
        return da_values[~nan_mask], nan_mask, da_values

    def forecast(self, Predictant, clim_year_start, clim_year_end, hindcast_det, hindcast_det_cross, Predictor_for_year, best_params=None, cluster_da=None):
        """
        Generate deterministic and probabilistic forecast for a target year using the stacking ensemble.
        Parameters
        ----------
        Predictant : xarray.DataArray
            Observed predictand data with dimensions (T, Y, X) or (T, M, Y, X).
        clim_year_start : int or str
            Start year of the climatology period.
        clim_year_end : int or str
            End year of the climatology period.
        hindcast_det : xarray.DataArray
            Deterministic hindcast data for training with dimensions (T, M, Y, X).
        hindcast_det_cross : xarray.DataArray
            Deterministic hindcast data for error estimation with dimensions (T, Y, X).
        Predictor_for_year : xarray.DataArray
            Predictor data for the target year with dimensions (T, M, Y, X).
        best_params : dict, optional
            Pre-computed best hyperparameters per cluster. If None, computes internally.
        cluster_da : xarray.DataArray, optional
            Pre-computed cluster labels. If None, computes internally.
        Returns
        -------
        forecast_det : xarray.DataArray
            Deterministic forecast with dimensions (T, Y, X).
        forecast_prob : xarray.DataArray
            Tercile probabilities with dimensions (probability, T, Y, X).
        """
        if "M" in Predictant.coords:
            Predictant_no_m = Predictant.isel(M=0).drop_vars('M').squeeze()
        else:
            Predictant_no_m = Predictant
        mask = xr.where(~np.isnan(Predictant_no_m.isel(T=0)), 1, np.nan).drop_vars(['T']).squeeze().to_numpy()
        # Standardize Predictor_for_year using hindcast climatology
        mean_val = hindcast_det.sel(T=slice(str(clim_year_start), str(clim_year_end))).mean(dim='T')
        std_val = hindcast_det.sel(T=slice(str(clim_year_start), str(clim_year_end))).std(dim='T')
        Predictor_for_year_st = (Predictor_for_year - mean_val) / std_val
        hindcast_det_st = standardize_timeseries(hindcast_det, clim_year_start, clim_year_end)
        Predictant_st = standardize_timeseries(Predictant_no_m, clim_year_start, clim_year_end)
        hindcast_det_st['T'] = Predictant_st['T']
        
        # Extract coordinates from X_test
        time = Predictor_for_year_st['T']
        lat = Predictor_for_year_st['Y']
        lon = Predictor_for_year_st['X']
        n_time = len(time)
        n_lat = len(lat)
        n_lon = len(lon)
        # Use provided best_params and cluster_da or compute if None
        if best_params is None:
            best_params, cluster_da = self.compute_hyperparameters(hindcast_det, Predictant_no_m, clim_year_start, clim_year_end)
        # Initialize predictions array
        predictions = np.full((n_time, n_lat, n_lon), np.nan)
        self.stacking_model = {}  # Dictionary to store models per cluster
        for c in range(self.n_clusters):
            if c not in best_params:
                continue
            bp = best_params[c]
            # Mask for this cluster
            mask_3d_train = (cluster_da == c).expand_dims({'T': hindcast_det_st['T']})
            mask_3d_test = (cluster_da == c).expand_dims({'T': Predictor_for_year_st['T']})
            # Stack training data for cluster
            X_train_stacked_c = hindcast_det_st.where(mask_3d_train).stack(sample=('T', 'Y', 'X')).transpose('sample', 'M').values
            y_train_stacked_c = Predictant_st.where(mask_3d_train).stack(sample=('T', 'Y', 'X')).values.ravel()
            train_nan_mask = np.any(~np.isfinite(X_train_stacked_c), axis=1) | ~np.isfinite(y_train_stacked_c)
            X_train_clean_c = X_train_stacked_c[~train_nan_mask]
            y_train_clean_c = y_train_stacked_c[~train_nan_mask]
            # Stack testing data for cluster
            X_test_stacked_c = Predictor_for_year_st.where(mask_3d_test).stack(sample=('T', 'Y', 'X')).transpose('sample', 'M').values
            test_nan_mask = np.any(~np.isfinite(X_test_stacked_c), axis=1)
            X_test_clean_c = X_test_stacked_c[~test_nan_mask]
            # Initialize base model (XGBoost)
            xgb = XGBRegressor(**bp['xgb'], random_state=self.random_state, verbosity=0)
            # Initialize meta-model (MLPRegressor)
            meta_model = MLPRegressor(**bp['mlp'], max_iter=self.max_iter, random_state=self.random_state)
            # Initialize stacking ensemble for this cluster
            stacking_model_c = StackingRegressor(estimators=[('xgb', xgb)], final_estimator=meta_model)
            # Fit the stacking model on cluster data
            stacking_model_c.fit(X_train_clean_c, y_train_clean_c)
            self.stacking_model[c] = stacking_model_c
            # Predict
            y_pred_c = stacking_model_c.predict(X_test_clean_c)
            # Reconstruct predictions for this cluster
            result_c = np.full(len(X_test_stacked_c), np.nan)
            result_c[~test_nan_mask] = y_pred_c
            pred_c_reshaped = result_c.reshape(n_time, n_lat, n_lon)
            # Fill in the predictions array
            predictions = np.where(np.isnan(predictions), pred_c_reshaped, predictions)
        result_da = xr.DataArray(
            data=predictions,
            coords={'T': time, 'Y': lat, 'X': lon},
            dims=['T', 'Y', 'X']
        ) * mask
        result_da = reverse_standardize(result_da, Predictant_no_m, clim_year_start, clim_year_end)
        year = Predictor_for_year.coords['T'].values.astype('datetime64[Y]').astype(int)[0] + 1970
        T_value_1 = Predictant_no_m.isel(T=0).coords['T'].values
        month_1 = T_value_1.astype('datetime64[M]').astype(int) % 12 + 1
        new_T_value = np.datetime64(f"{year}-{month_1:02d}-01")
        result_da = result_da.assign_coords(T=xr.DataArray([new_T_value], dims=["T"]))
        result_da['T'] = result_da['T'].astype('datetime64[ns]')
        # Compute tercile probabilities
        index_start = Predictant_no_m.get_index("T").get_loc(str(clim_year_start)).start
        index_end = Predictant_no_m.get_index("T").get_loc(str(clim_year_end)).stop
        rainfall_for_tercile = Predictant_no_m.isel(T=slice(index_start, index_end))
        terciles = rainfall_for_tercile.quantile([0.33, 0.67], dim='T')
        error_variance = (Predictant_no_m - hindcast_det_cross).var(dim='T')
        dof = len(Predictant_no_m.get_index("T")) - 2
        if self.dist_method == "t":
            calc_func = self.calculate_tercile_probabilities
            hindcast_prob = xr.apply_ufunc(
                calc_func,
                result_da,
                error_variance,
                terciles.isel(quantile=0).drop_vars('quantile'),
                terciles.isel(quantile=1).drop_vars('quantile'),
                input_core_dims=[('T',), (), (), ()],
                vectorize=True,
                kwargs={'dof': dof},
                dask='parallelized',
                output_core_dims=[('probability', 'T')],
                output_dtypes=['float'],
                dask_gufunc_kwargs={'output_sizes': {'probability': 3}, "allow_rechunk": True}
            )
        elif self.dist_method == "weibull_min":
            calc_func = self.calculate_tercile_probabilities_weibull_min
            hindcast_prob = xr.apply_ufunc(
                calc_func,
                result_da,
                error_variance,
                terciles.isel(quantile=0).drop_vars('quantile'),
                terciles.isel(quantile=1).drop_vars('quantile'),
                input_core_dims=[('T',), (), (), ()],
                vectorize=True,
                kwargs={'dof': dof},
                dask='parallelized',
                output_core_dims=[('probability', 'T')],
                output_dtypes=['float'],
                dask_gufunc_kwargs={'output_sizes': {'probability': 3}, "allow_rechunk": True}
            )
        elif self.dist_method == "gamma":
            calc_func = self.calculate_tercile_probabilities_gamma
            hindcast_prob = xr.apply_ufunc(
                calc_func,
                result_da,
                error_variance,
                terciles.isel(quantile=0).drop_vars('quantile'),
                terciles.isel(quantile=1).drop_vars('quantile'),
                input_core_dims=[('T',), (), (), ()],
                vectorize=True,
                dask='parallelized',
                output_core_dims=[('probability', 'T')],
                output_dtypes=['float'],
                dask_gufunc_kwargs={'output_sizes': {'probability': 3}, "allow_rechunk": True}
            )
        elif self.dist_method == "normal":
            calc_func = self.calculate_tercile_probabilities_normal
            hindcast_prob = xr.apply_ufunc(
                calc_func,
                result_da,
                error_variance,
                terciles.isel(quantile=0).drop_vars('quantile'),
                terciles.isel(quantile=1).drop_vars('quantile'),
                input_core_dims=[('T',), (), (), ()],
                vectorize=True,
                dask='parallelized',
                output_core_dims=[('probability', 'T')],
                output_dtypes=['float'],
                dask_gufunc_kwargs={'output_sizes': {'probability': 3}, "allow_rechunk": True}
            )
        elif self.dist_method == "lognormal":
            calc_func = self.calculate_tercile_probabilities_lognormal
            hindcast_prob = xr.apply_ufunc(
                calc_func,
                result_da,
                error_variance,
                terciles.isel(quantile=0).drop_vars('quantile'),
                terciles.isel(quantile=1).drop_vars('quantile'),
                input_core_dims=[('T',), (), (), ()],
                vectorize=True,
                dask='parallelized',
                output_core_dims=[('probability', 'T')],
                output_dtypes=['float'],
                dask_gufunc_kwargs={'output_sizes': {'probability': 3}, "allow_rechunk": True}
            )
        elif self.dist_method == "nonparam":
            calc_func = self.calculate_tercile_probabilities_nonparametric
            error_samples = Predictant_no_m - hindcast_det_cross
            error_samples = error_samples.rename({'T': 'S'})
            hindcast_prob = xr.apply_ufunc(
                calc_func,
                result_da,
                error_samples,
                terciles.isel(quantile=0).drop_vars('quantile'),
                terciles.isel(quantile=1).drop_vars('quantile'),
                input_core_dims=[('T',), ('S',), (), ()],
                output_core_dims=[('probability', 'T')],
                vectorize=True,
                dask='parallelized',
                output_dtypes=[float],
                dask_gufunc_kwargs={'output_sizes': {'probability': 3}, "allow_rechunk": True}
            )
        else:
            raise ValueError(f"Invalid dist_method: {self.dist_method}.")
        hindcast_prob = hindcast_prob.assign_coords(probability=('probability', ['PB', 'PN', 'PA']))
        return result_da * mask, mask * hindcast_prob.transpose('probability', 'T', 'Y', 'X')


class WAS_mme_Stack_RF_MLP_:
    """
    Stacking ensemble with Random Forest as base model and MLP as meta-model for Multi-Model Ensemble (MME) forecasting.
    This class implements a stacking ensemble using RandomForestRegressor as the base model,
    with an MLPRegressor meta-model, for deterministic forecasting and optional tercile probability calculations.
    Implements hyperparameter optimization via randomized search for both base and meta models.

    Parameters
    ----------
    n_estimators_range : list of int, optional
        List of n_estimators values to tune for Random Forest (default is [50, 100, 200, 300]).
    max_depth_range : list of int or None, optional
        List of max depths to tune for Random Forest (default is [None, 10, 20, 30]).
    min_samples_split_range : list of int, optional
        List of minimum samples required to split to tune for Random Forest (default is [2, 5, 10]).
    min_samples_leaf_range : list of int, optional
        List of minimum samples required at leaf node to tune for Random Forest (default is [1, 2, 4]).
    max_features_range : list of str or float, optional
        List of max features to tune for Random Forest (default is ['auto', 'sqrt', 0.33, 0.5]).
    hidden_layer_sizes_range : list of tuples, optional
        List of hidden layer sizes to tune for MLP (default is [(10,), (10, 5), (20, 10)]).
    activation_options : list of str, optional
        Activation functions to tune for MLP (default is ['relu', 'tanh', 'logistic']).
    solver_options : list of str, optional
        Solvers to tune for MLP (default is ['adam', 'sgd', 'lbfgs']).
    alpha_range : list of float, optional
        L2 regularization parameters to tune for MLP (default is [0.0001, 0.001, 0.01, 0.1]).
    max_iter : int, optional
        Maximum iterations for MLP (default is 200).
    random_state : int, optional
        Seed for reproducibility (default is 42).
    dist_method : str, optional
        Distribution method for tercile probabilities ('t', 'gamma', etc.) (default is 'gamma').
    n_iter_search : int, optional
        Number of iterations for randomized search (default is 10).
    cv_folds : int, optional
        Number of cross-validation folds (default is 3).
    """
    def __init__(self,
                 n_estimators_range=[50, 100, 200, 300],
                 max_depth_range=[None, 10, 20, 30],
                 min_samples_split_range=[2, 5, 10],
                 min_samples_leaf_range=[1, 2, 4],
                 max_features_range=['auto', 'sqrt', 0.33, 0.5],
                 hidden_layer_sizes_range=[(10,), (10, 5), (20, 10)],
                 activation_options=['relu', 'tanh', 'logistic'],
                 solver_options=['adam', 'sgd', 'lbfgs'],
                 alpha_range=[0.0001, 0.001, 0.01, 0.1],
                 max_iter=200,
                 random_state=42,
                 dist_method="gamma",
                 n_iter_search=10,
                 cv_folds=3):
        self.n_estimators_range = n_estimators_range
        self.max_depth_range = max_depth_range
        self.min_samples_split_range = min_samples_split_range
        self.min_samples_leaf_range = min_samples_leaf_range
        self.max_features_range = max_features_range
        self.hidden_layer_sizes_range = hidden_layer_sizes_range
        self.activation_options = activation_options
        self.solver_options = solver_options
        self.alpha_range = alpha_range
        self.max_iter = max_iter
        self.random_state = random_state
        self.dist_method = dist_method
        self.n_iter_search = n_iter_search
        self.cv_folds = cv_folds
        self.stacking_model = None

    def compute_hyperparameters(self, Predictors, Predictand, clim_year_start, clim_year_end):
        """
        Independently computes the best hyperparameters using randomized search on stacked training data.

        Parameters
        ----------
        Predictors : xarray.DataArray
            Training predictor data with dimensions (T, M, Y, X).
        Predictand : xarray.DataArray
            Training predictand data with dimensions (T, Y, X).
        clim_year_start : int
            Start year of the climatology period.
        clim_year_end : int
            End year of the climatology period.

        Returns
        -------
        dict
            Best hyperparameters found.
        """
        X_train = standardize_timeseries(Predictors, clim_year_start, clim_year_end)
        y_train = standardize_timeseries(Predictand, clim_year_start, clim_year_end)

        # Stack training data
        X_train_stacked = X_train.stack(sample=('T', 'Y', 'X')).transpose('sample', 'M').values
        y_train_stacked = y_train.stack(sample=('T', 'Y', 'X')).values.ravel()
        train_nan_mask = np.any(~np.isfinite(X_train_stacked), axis=1) | ~np.isfinite(y_train_stacked)
        X_train_clean = X_train_stacked[~train_nan_mask]
        y_train_clean = y_train_stacked[~train_nan_mask]

        # For Random Forest (base model)
        param_dist_rf = {
            'n_estimators': self.n_estimators_range,
            'max_depth': self.max_depth_range,
            'min_samples_split': self.min_samples_split_range,
            'min_samples_leaf': self.min_samples_leaf_range,
            'max_features': self.max_features_range
        }
        model_rf = RandomForestRegressor(random_state=self.random_state)
        random_search_rf = RandomizedSearchCV(
            model_rf, param_distributions=param_dist_rf, n_iter=self.n_iter_search,
            cv=self.cv_folds, scoring='neg_mean_squared_error',
            random_state=self.random_state, error_score=np.nan
        )
        random_search_rf.fit(X_train_clean, y_train_clean)
        best_rf = random_search_rf.best_params_

        # For MLP (meta model)
        param_dist_mlp = {
            'hidden_layer_sizes': self.hidden_layer_sizes_range,
            'activation': self.activation_options,
            'solver': self.solver_options,
            'alpha': self.alpha_range
        }
        model_mlp = MLPRegressor(max_iter=self.max_iter, random_state=self.random_state)
        random_search_mlp = RandomizedSearchCV(
            model_mlp, param_distributions=param_dist_mlp, n_iter=self.n_iter_search,
            cv=self.cv_folds, scoring='neg_mean_squared_error',
            random_state=self.random_state, error_score=np.nan
        )
        random_search_mlp.fit(X_train_clean, y_train_clean)
        best_mlp = random_search_mlp.best_params_

        return {'rf': best_rf, 'mlp': best_mlp}

    def compute_model(self, X_train, y_train, X_test, y_test, best_params=None):
        """
        Compute deterministic hindcast using the stacking ensemble of Random Forest (base) and MLP (meta).

        Parameters
        ----------
        X_train : xarray.DataArray
            Training predictor data with dimensions (T, M, Y, X).
        y_train : xarray.DataArray
            Training predictand data with dimensions (T, Y, X).
        X_test : xarray.DataArray
            Testing predictor data with dimensions (T, M, Y, X).
        y_test : xarray.DataArray
            Testing predictand data with dimensions (T, Y, X).
        best_params : dict, optional
            Pre-computed best hyperparameters for 'rf' and 'mlp'. If None, computes internally.

        Returns
        -------
        predicted_da : xarray.DataArray
            Deterministic hindcast with dimensions (T, Y, X).
        """
        # Extract coordinate variables from X_test
        time = X_test['T']
        lat = X_test['Y']
        lon = X_test['X']
        n_time = len(X_test.coords['T'])
        n_lat = len(X_test.coords['Y'])
        n_lon = len(X_test.coords['X'])

        # Stack training data
        X_train_stacked = X_train.stack(sample=('T', 'Y', 'X')).transpose('sample', 'M').values
        y_train_stacked = y_train.stack(sample=('T', 'Y', 'X')).transpose('sample', 'M').values
        train_nan_mask = (np.any(~np.isfinite(X_train_stacked), axis=1) | np.any(~np.isfinite(y_train_stacked), axis=1))
        X_train_clean = X_train_stacked[~train_nan_mask]
        y_train_clean = y_train_stacked[~train_nan_mask]

        # Stack testing data
        X_test_stacked = X_test.stack(sample=('T', 'Y', 'X')).transpose('sample', 'M').values
        y_test_stacked = y_test.stack(sample=('T', 'Y', 'X')).transpose('sample', 'M').values
        test_nan_mask = (np.any(~np.isfinite(X_test_stacked), axis=1) | np.any(~np.isfinite(y_test_stacked), axis=1))

        # Use provided best_params or compute if None
        if best_params is None:
            best_params = self.compute_hyperparameters(X_train_clean, y_train_clean)

        # Initialize base model (Random Forest)
        rf = RandomForestRegressor(**best_params['rf'], random_state=self.random_state)
        base_models = [('rf', rf)]

        # Initialize meta-model (MLP)
        meta_model = MLPRegressor(**best_params['mlp'], max_iter=self.max_iter, random_state=self.random_state)

        # Initialize stacking ensemble
        self.stacking_model = StackingRegressor(estimators=base_models, final_estimator=meta_model)

        # Fit the stacking model on clean training data
        self.stacking_model.fit(X_train_clean, y_train_clean)

        # Predict on clean test data
        y_pred_test = self.stacking_model.predict(X_test_stacked[~test_nan_mask])

        # Reconstruct predictions
        result = np.empty_like(np.squeeze(y_test_stacked))
        result[np.squeeze(test_nan_mask)] = np.squeeze(y_test_stacked[test_nan_mask])
        result[~np.squeeze(test_nan_mask)] = y_pred_test

        predictions_reshaped = result.reshape(n_time, n_lat, n_lon)
        predicted_da = xr.DataArray(data=predictions_reshaped,
                                    coords={'T': time, 'Y': lat, 'X': lon},
                                    dims=['T', 'Y', 'X'])
        return predicted_da

    # ------------------ Probability Calculation Methods ------------------
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
    def calculate_tercile_probabilities_weibull_min(best_guess, error_variance, first_tercile, second_tercile, dof):
        """
        Weibull minimum-based method.
        """
        n_time = len(best_guess)
        pred_prob = np.empty((3, n_time))

        if np.all(np.isnan(best_guess)):
            pred_prob[:] = np.nan
        else:
            error_std = np.sqrt(error_variance)
            pred_prob[0, :] = stats.weibull_min.cdf(first_tercile, c=dof, loc=best_guess, scale=error_std)
            pred_prob[1, :] = stats.weibull_min.cdf(second_tercile, c=dof, loc=best_guess, scale=error_std) - \
                              stats.weibull_min.cdf(first_tercile, c=dof, loc=best_guess, scale=error_std)
            pred_prob[2, :] = 1 - stats.weibull_min.cdf(second_tercile, c=dof, loc=best_guess, scale=error_std)
        return pred_prob

    @staticmethod
    def calculate_tercile_probabilities_gamma(best_guess, error_variance, T1, T2, dof=None):
        """Gamma-distribution based method."""
        n_time = len(best_guess)
        pred_prob = np.empty((3, n_time), dtype=float)
        if np.any(np.isnan(best_guess)) or np.any(np.isnan(error_variance)):
            pred_prob[:] = np.nan
            return pred_prob
        best_guess = np.asarray(best_guess, dtype=float)
        error_variance = np.asarray(error_variance, dtype=float)
        T1 = np.asarray(T1, dtype=float)
        T2 = np.asarray(T2, dtype=float)
        alpha = (best_guess ** 2) / error_variance
        theta = error_variance / best_guess
        cdf_t1 = gamma.cdf(T1, a=alpha, scale=theta)
        cdf_t2 = gamma.cdf(T2, a=alpha, scale=theta)
        pred_prob[0, :] = cdf_t1
        pred_prob[1, :] = cdf_t2 - cdf_t1
        pred_prob[2, :] = 1.0 - cdf_t2
        return pred_prob

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

    @staticmethod
    def calculate_tercile_probabilities_normal(best_guess, error_variance, first_tercile, second_tercile):
        """Normal-distribution based method."""
        n_time = len(best_guess)
        pred_prob = np.empty((3, n_time))
        if np.all(np.isnan(best_guess)):
            pred_prob[:] = np.nan
        else:
            error_std = np.sqrt(error_variance)
            pred_prob[0, :] = norm.cdf(first_tercile, loc=best_guess, scale=error_std)
            pred_prob[1, :] = norm.cdf(second_tercile, loc=best_guess, scale=error_std) - norm.cdf(first_tercile, loc=best_guess, scale=error_std)
            pred_prob[2, :] = 1 - norm.cdf(second_tercile, loc=best_guess, scale=error_std)
        return pred_prob

    @staticmethod
    def calculate_tercile_probabilities_lognormal(best_guess, error_variance, first_tercile, second_tercile):
        """Lognormal-distribution based method."""
        n_time = len(best_guess)
        pred_prob = np.empty((3, n_time))
        if np.any(np.isnan(best_guess)) or np.any(np.isnan(error_variance)):
            pred_prob[:] = np.nan
            return pred_prob
        sigma = np.sqrt(np.log(1 + error_variance / (best_guess ** 2)))
        mu = np.log(best_guess) - sigma ** 2 / 2
        pred_prob[0, :] = lognorm.cdf(first_tercile, s=sigma, scale=np.exp(mu))
        pred_prob[1, :] = lognorm.cdf(second_tercile, s=sigma, scale=np.exp(mu)) - lognorm.cdf(first_tercile, s=sigma, scale=np.exp(mu))
        pred_prob[2, :] = 1 - lognorm.cdf(second_tercile, s=sigma, scale=np.exp(mu))
        return pred_prob

    def compute_prob(self, Predictant, clim_year_start, clim_year_end, hindcast_det):
        """
        Compute tercile probabilities using the chosen distribution method.
        Predictant is expected to be an xarray DataArray with dims (T, Y, X).
        """
        if "M" in Predictant.coords:
            Predictant = Predictant.isel(M=0).drop_vars('M').squeeze()
        mask = xr.where(~np.isnan(Predictant.isel(T=0)), 1, np.nan).drop_vars(['T']).squeeze().to_numpy()

        Predictant = Predictant.transpose('T', 'Y', 'X')
        index_start = Predictant.get_index("T").get_loc(str(clim_year_start)).start
        index_end = Predictant.get_index("T").get_loc(str(clim_year_end)).stop
        rainfall_for_tercile = Predictant.isel(T=slice(index_start, index_end))
        terciles = rainfall_for_tercile.quantile([0.33, 0.67], dim='T')
        error_variance = (Predictant - hindcast_det).var(dim='T')
        dof = len(Predictant.get_index("T")) - 2

        if self.dist_method == "t":
            calc_func = self.calculate_tercile_probabilities
            hindcast_prob = xr.apply_ufunc(
                calc_func,
                hindcast_det,
                error_variance,
                terciles.isel(quantile=0).drop_vars('quantile'),
                terciles.isel(quantile=1).drop_vars('quantile'),
                input_core_dims=[('T',), (), (), ()],
                vectorize=True,
                kwargs={'dof': dof},
                dask='parallelized',
                output_core_dims=[('probability', 'T')],
                output_dtypes=['float'],
                dask_gufunc_kwargs={'output_sizes': {'probability': 3}, "allow_rechunk": True}
            )
        elif self.dist_method == "weibull_min":
            calc_func = self.calculate_tercile_probabilities_weibull_min
            hindcast_prob = xr.apply_ufunc(
                calc_func,
                hindcast_det,
                error_variance,
                terciles.isel(quantile=0).drop_vars('quantile'),
                terciles.isel(quantile=1).drop_vars('quantile'),
                input_core_dims=[('T',), (), (), ()],
                vectorize=True,
                kwargs={'dof': dof},
                dask='parallelized',
                output_core_dims=[('probability', 'T')],
                output_dtypes=['float'],
                dask_gufunc_kwargs={'output_sizes': {'probability': 3}, "allow_rechunk": True}
            )
        elif self.dist_method == "gamma":
            calc_func = self.calculate_tercile_probabilities_gamma
            hindcast_prob = xr.apply_ufunc(
                calc_func,
                hindcast_det,
                error_variance,
                terciles.isel(quantile=0).drop_vars('quantile'),
                terciles.isel(quantile=1).drop_vars('quantile'),
                input_core_dims=[('T',), (), (), ()],
                vectorize=True,
                dask='parallelized',
                output_core_dims=[('probability', 'T')],
                output_dtypes=['float'],
                dask_gufunc_kwargs={'output_sizes': {'probability': 3}, "allow_rechunk": True}
            )
        elif self.dist_method == "nonparam":
            calc_func = self.calculate_tercile_probabilities_nonparametric
            error_samples = (Predictant - hindcast_det)
            hindcast_prob = xr.apply_ufunc(
                calc_func,
                hindcast_det,
                error_samples,
                terciles.isel(quantile=0).drop_vars('quantile'),
                terciles.isel(quantile=1).drop_vars('quantile'),
                input_core_dims=[('T',), ('T',), (), ()],
                output_core_dims=[('probability', 'T')],
                vectorize=True,
                dask='parallelized',
                output_dtypes=[float],
                dask_gufunc_kwargs={'output_sizes': {'probability': 3}, "allow_rechunk": True}
            )
        elif self.dist_method == "normal":
            calc_func = self.calculate_tercile_probabilities_normal
            hindcast_prob = xr.apply_ufunc(
                calc_func,
                hindcast_det,
                error_variance,
                terciles.isel(quantile=0).drop_vars('quantile'),
                terciles.isel(quantile=1).drop_vars('quantile'),
                input_core_dims=[('T',), (), (), ()],
                vectorize=True,
                dask='parallelized',
                output_core_dims=[('probability', 'T')],
                output_dtypes=['float'],
                dask_gufunc_kwargs={'output_sizes': {'probability': 3}, "allow_rechunk": True}
            )
        elif self.dist_method == "lognormal":
            calc_func = self.calculate_tercile_probabilities_lognormal
            hindcast_prob = xr.apply_ufunc(
                calc_func,
                hindcast_det,
                error_variance,
                terciles.isel(quantile=0).drop_vars('quantile'),
                terciles.isel(quantile=1).drop_vars('quantile'),
                input_core_dims=[('T',), (), (), ()],
                vectorize=True,
                dask='parallelized',
                output_core_dims=[('probability', 'T')],
                output_dtypes=['float'],
                dask_gufunc_kwargs={'output_sizes': {'probability': 3}, "allow_rechunk": True}
            )
        else:
            raise ValueError(f"Invalid dist_method: {self.dist_method}.")

        hindcast_prob = hindcast_prob.assign_coords(probability=('probability', ['PB', 'PN', 'PA']))
        return hindcast_prob.transpose('probability', 'T', 'Y', 'X')

    @staticmethod
    def _reshape_and_filter_data(da):
        """
        Helper: stack the DataArray from (T, Y, X[, M]) to (n_samples, n_features)
        and remove rows containing NaNs.
        """
        da_stacked = da.stack(sample=('T', 'Y', 'X'))
        if 'M' in da.dims:
            da_stacked = da_stacked.transpose('sample', 'M')
        else:
            da_stacked = da_stacked.transpose('sample')
        da_values = da_stacked.values
        nan_mask = np.any(np.isnan(da_values), axis=1)
        return da_values[~nan_mask], nan_mask, da_values

    def forecast(self, Predictant, clim_year_start, clim_year_end, hindcast_det, hindcast_det_cross, Predictor_for_year, best_params=None):
        """
        Generate deterministic and probabilistic forecast for a target year using the stacking ensemble.

        Parameters
        ----------
        Predictant : xarray.DataArray
            Observed predictand data with dimensions (T, Y, X) or (T, M, Y, X).
        clim_year_start : int or str
            Start year of the climatology period.
        clim_year_end : int or str
            End year of the climatology period.
        hindcast_det : xarray.DataArray
            Deterministic hindcast data for training with dimensions (T, M, Y, X).
        hindcast_det_cross : xarray.DataArray
            Deterministic hindcast data for error estimation with dimensions (T, Y, X).
        Predictor_for_year : xarray.DataArray
            Predictor data for the target year with dimensions (T, M, Y, X).
        best_params : dict, optional
            Pre-computed best hyperparameters for 'rf' and 'mlp'. If None, computes internally.

        Returns
        -------
        result_da : xarray.DataArray
            Deterministic forecast with dimensions (T, Y, X).
        hindcast_prob : xarray.DataArray
            Tercile probabilities with dimensions (probability, T, Y, X).
        """
        if "M" in Predictant.coords:
            Predictant_no_m = Predictant.isel(M=0).drop_vars('M').squeeze()
        else:
            Predictant_no_m = Predictant

        mask = xr.where(~np.isnan(Predictant_no_m.isel(T=0)), 1, np.nan).drop_vars(['T']).squeeze().to_numpy()

        # Standardize Predictor_for_year using hindcast climatology
        mean_val = hindcast_det.sel(T=slice(str(clim_year_start), str(clim_year_end))).mean(dim='T')
        std_val = hindcast_det.sel(T=slice(str(clim_year_start), str(clim_year_end))).std(dim='T')
        Predictor_for_year_st = (Predictor_for_year - mean_val) / std_val

        hindcast_det_st = standardize_timeseries(hindcast_det, clim_year_start, clim_year_end)
        Predictant_st = standardize_timeseries(Predictant_no_m, clim_year_start, clim_year_end)
        y_test = Predictant_st.isel(T=[-1])

        # Extract coordinates from X_test
        time = Predictor_for_year_st['T']
        lat = Predictor_for_year_st['Y']
        lon = Predictor_for_year_st['X']
        n_time = len(Predictor_for_year_st.coords['T'])
        n_lat = len(Predictor_for_year_st.coords['Y'])
        n_lon = len(Predictor_for_year_st.coords['X'])

        # Stack training data and remove rows with NaNs
        X_train_stacked = hindcast_det_st.stack(sample=('T', 'Y', 'X')).transpose('sample', 'M').values
        y_train_stacked = Predictant_st.stack(sample=('T', 'Y', 'X')).values.ravel()
        train_nan_mask = np.any(~np.isfinite(X_train_stacked), axis=1) | ~np.isfinite(y_train_stacked)
        X_train_clean = X_train_stacked[~train_nan_mask]
        y_train_clean = y_train_stacked[~train_nan_mask]

        # Stack testing data
        X_test_stacked = Predictor_for_year_st.stack(sample=('T', 'Y', 'X')).transpose('sample', 'M').values
        y_test_stacked = y_test.stack(sample=('T', 'Y', 'X')).values.ravel()
        test_nan_mask = np.any(~np.isfinite(X_test_stacked), axis=1) | ~np.isfinite(y_test_stacked)

        # Use provided best_params or compute if None
        if best_params is None:
            best_params = self.compute_hyperparameters(X_train_clean, y_train_clean)

        # Initialize base model (Random Forest)
        rf = RandomForestRegressor(**best_params['rf'], random_state=self.random_state)
        base_models = [('rf', rf)]

        # Initialize meta-model (MLP)
        meta_model = MLPRegressor(**best_params['mlp'], max_iter=self.max_iter, random_state=self.random_state)

        # Initialize stacking ensemble
        self.stacking_model = StackingRegressor(estimators=base_models, final_estimator=meta_model)

        # Fit the stacking model on clean training data
        self.stacking_model.fit(X_train_clean, y_train_clean)

        # Predict on clean test data
        y_pred = self.stacking_model.predict(X_test_stacked[~test_nan_mask])

        # Reconstruct the prediction array
        result = np.empty_like(np.squeeze(y_test_stacked))
        result[np.squeeze(test_nan_mask)] = np.squeeze(y_test_stacked[test_nan_mask])
        result[~np.squeeze(test_nan_mask)] = y_pred

        predictions_reshaped = result.reshape(n_time, n_lat, n_lon)
        result_da = xr.DataArray(data=predictions_reshaped,
                                 coords={'T': time, 'Y': lat, 'X': lon},
                                 dims=['T', 'Y', 'X']) * mask

        result_da = reverse_standardize(result_da, Predictant_no_m, clim_year_start, clim_year_end)
        year = Predictor_for_year.coords['T'].values.astype('datetime64[Y]').astype(int)[0] + 1970
        T_value_1 = Predictant_no_m.isel(T=0).coords['T'].values
        month_1 = T_value_1.astype('datetime64[M]').astype(int) % 12 + 1
        new_T_value = np.datetime64(f"{year}-{month_1:02d}-{1:02d}")
        result_da = result_da.assign_coords(T=xr.DataArray([new_T_value], dims=["T"]))
        result_da['T'] = result_da['T'].astype('datetime64[ns]')

        # Compute tercile probabilities
        index_start = Predictant_no_m.get_index("T").get_loc(str(clim_year_start)).start
        index_end = Predictant_no_m.get_index("T").get_loc(str(clim_year_end)).stop
        rainfall_for_tercile = Predictant_no_m.isel(T=slice(index_start, index_end))
        terciles = rainfall_for_tercile.quantile([0.33, 0.67], dim='T')
        error_variance = (Predictant_no_m - hindcast_det_cross).var(dim='T')
        dof = len(Predictant_no_m.get_index("T")) - 2

        if self.dist_method == "t":
            calc_func = self.calculate_tercile_probabilities
            hindcast_prob = xr.apply_ufunc(
                calc_func,
                result_da,
                error_variance,
                terciles.isel(quantile=0).drop_vars('quantile'),
                terciles.isel(quantile=1).drop_vars('quantile'),
                input_core_dims=[('T',), (), (), ()],
                vectorize=True,
                kwargs={'dof': dof},
                dask='parallelized',
                output_core_dims=[('probability','T')],
                output_dtypes=['float'],
                dask_gufunc_kwargs={'output_sizes': {'probability': 3}, "allow_rechunk": True}
            )
        elif self.dist_method == "weibull_min":
            calc_func = self.calculate_tercile_probabilities_weibull_min
            hindcast_prob = xr.apply_ufunc(
                calc_func,
                result_da,
                error_variance,
                terciles.isel(quantile=0).drop_vars('quantile'),
                terciles.isel(quantile=1).drop_vars('quantile'),
                input_core_dims=[('T',), (), (), ()],
                vectorize=True,
                kwargs={'dof': dof},
                dask='parallelized',
                output_core_dims=[('probability', 'T')],
                output_dtypes=['float'],
                dask_gufunc_kwargs={'output_sizes': {'probability': 3}, "allow_rechunk": True}
            )
        elif self.dist_method == "gamma":
            calc_func = self.calculate_tercile_probabilities_gamma
            hindcast_prob = xr.apply_ufunc(
                calc_func,
                result_da,
                error_variance,
                terciles.isel(quantile=0).drop_vars('quantile'),
                terciles.isel(quantile=1).drop_vars('quantile'),
                input_core_dims=[('T',), (), (), ()],
                vectorize=True,
                dask='parallelized',
                output_core_dims=[('probability','T')],
                output_dtypes=['float'],
                dask_gufunc_kwargs={'output_sizes': {'probability': 3}, "allow_rechunk": True}
            )
        elif self.dist_method == "normal":
            calc_func = self.calculate_tercile_probabilities_normal
            hindcast_prob = xr.apply_ufunc(
                calc_func,
                result_da,
                error_variance,
                terciles.isel(quantile=0).drop_vars('quantile'),
                terciles.isel(quantile=1).drop_vars('quantile'),
                input_core_dims=[('T',), (), (), ()],
                vectorize=True,
                dask='parallelized',
                output_core_dims=[('probability','T')],
                output_dtypes=['float'],
                dask_gufunc_kwargs={'output_sizes': {'probability': 3}, "allow_rechunk": True}
            )
        elif self.dist_method == "lognormal":
            calc_func = self.calculate_tercile_probabilities_lognormal
            hindcast_prob = xr.apply_ufunc(
                calc_func,
                result_da,
                error_variance,
                terciles.isel(quantile=0).drop_vars('quantile'),
                terciles.isel(quantile=1).drop_vars('quantile'),
                input_core_dims=[('T',), (), (), ()],
                vectorize=True,
                dask='parallelized',
                output_core_dims=[('probability','T')],
                output_dtypes=['float'],
                dask_gufunc_kwargs={'output_sizes': {'probability': 3}, "allow_rechunk": True}
            )
        elif self.dist_method == "nonparam":
            calc_func = self.calculate_tercile_probabilities_nonparametric
            error_samples = Predictant_no_m - hindcast_det_cross
            error_samples = error_samples.rename({'T': 'S'})
            hindcast_prob = xr.apply_ufunc(
                calc_func,
                result_da,
                error_samples,
                terciles.isel(quantile=0).drop_vars('quantile'),
                terciles.isel(quantile=1).drop_vars('quantile'),
                input_core_dims=[('T',), ('S',), (), ()],
                output_core_dims=[('probability','T')],
                vectorize=True,
                dask='parallelized',
                output_dtypes=[float],
                dask_gufunc_kwargs={'output_sizes': {'probability': 3}, "allow_rechunk": True}
            )
        else:
            raise ValueError(f"Invalid dist_method: {self.dist_method}.")

        hindcast_prob = hindcast_prob.assign_coords(probability=('probability', ['PB', 'PN', 'PA']))
        return result_da * mask, mask * hindcast_prob.transpose('probability', 'T', 'Y', 'X')


class WAS_mme_Stack_RF_MLP:
    """
    Stacking ensemble with Random Forest as base model and MLP as meta-model for Multi-Model Ensemble (MME) forecasting.
    This class implements a stacking ensemble using RandomForestRegressor as the base model,
    with an MLPRegressor meta-model, for deterministic forecasting and optional tercile probability calculations.
    Implements hyperparameter optimization via randomized search for both base and meta models.
    Parameters
    ----------
    n_estimators_range : list of int, optional
        List of n_estimators values to tune for Random Forest (default is [50, 100, 200, 300]).
    max_depth_range : list of int or None, optional
        List of max depths to tune for Random Forest (default is [None, 10, 20, 30]).
    min_samples_split_range : list of int, optional
        List of minimum samples required to split to tune for Random Forest (default is [2, 5, 10]).
    min_samples_leaf_range : list of int, optional
        List of minimum samples required at leaf node to tune for Random Forest (default is [1, 2, 4]).
    max_features_range : list of str or float, optional
        List of max features to tune for Random Forest (default is ['auto', 'sqrt', 0.33, 0.5]).
    hidden_layer_sizes_range : list of tuples, optional
        List of hidden layer sizes to tune for MLP (default is [(10,), (10, 5), (20, 10)]).
    activation_options : list of str, optional
        Activation functions to tune for MLP (default is ['relu', 'tanh', 'logistic']).
    solver_options : list of str, optional
        Solvers to tune for MLP (default is ['adam', 'sgd', 'lbfgs']).
    alpha_range : list of float, optional
        L2 regularization parameters to tune for MLP (default is [0.0001, 0.001, 0.01, 0.1]).
    max_iter : int, optional
        Maximum iterations for MLP (default is 200).
    random_state : int, optional
        Seed for reproducibility (default is 42).
    dist_method : str, optional
        Distribution method for tercile probabilities ('t', 'gamma', etc.) (default is 'gamma').
    n_iter_search : int, optional
        Number of iterations for randomized search (default is 10).
    cv_folds : int, optional
        Number of cross-validation folds (default is 3).
    n_clusters : int, optional
        Number of clusters for homogenized zones (default is 4).
    """
    def __init__(self,
                 n_estimators_range=[50, 100, 200, 300],
                 max_depth_range=[None, 10, 20, 30],
                 min_samples_split_range=[2, 5, 10],
                 min_samples_leaf_range=[1, 2, 4],
                 max_features_range=['auto', 'sqrt', 0.33, 0.5],
                 hidden_layer_sizes_range=[(10,), (10, 5), (20, 10)],
                 activation_options=['relu', 'tanh', 'logistic'],
                 solver_options=['adam', 'sgd', 'lbfgs'],
                 alpha_range=[0.0001, 0.001, 0.01, 0.1],
                 max_iter=200,
                 random_state=42,
                 dist_method="gamma",
                 n_iter_search=10,
                 cv_folds=3,
                 n_clusters=4):
        self.n_estimators_range = n_estimators_range
        self.max_depth_range = max_depth_range
        self.min_samples_split_range = min_samples_split_range
        self.min_samples_leaf_range = min_samples_leaf_range
        self.max_features_range = max_features_range
        self.hidden_layer_sizes_range = hidden_layer_sizes_range
        self.activation_options = activation_options
        self.solver_options = solver_options
        self.alpha_range = alpha_range
        self.max_iter = max_iter
        self.random_state = random_state
        self.dist_method = dist_method
        self.n_iter_search = n_iter_search
        self.cv_folds = cv_folds
        self.n_clusters = n_clusters
        self.stacking_model = None

    def compute_hyperparameters(self, Predictors, Predictand, clim_year_start, clim_year_end):
        """
        Independently computes the best hyperparameters using randomized search on stacked training data for each homogenized zone.
        Parameters
        ----------
        Predictors : xarray.DataArray
            Training predictor data with dimensions (T, M, Y, X).
        Predictand : xarray.DataArray
            Training predictand data with dimensions (T, Y, X).
        clim_year_start : int
            Start year of the climatology period.
        clim_year_end : int
            End year of the climatology period.
        Returns
        -------
        best_params_dict : dict
            Best hyperparameters for each cluster.
        cluster_da : xarray.DataArray
            Cluster labels with dimensions (Y, X).
        """
        if "M" in Predictand.coords:
            Predictand = Predictand.isel(M=0).drop_vars('M').squeeze()
           
        X_train_std = standardize_timeseries(Predictors, clim_year_start, clim_year_end)
        Predictand.name = "varname"
        # Step 1: Perform KMeans clustering based on predictand's spatial distribution
        kmeans = KMeans(n_clusters=self.n_clusters, random_state=self.random_state)
        Predictand_dropna = Predictand.to_dataframe().reset_index().dropna().drop(columns=['T'])
        variable_column = Predictand_dropna.columns[2]
        Predictand_dropna['cluster'] = kmeans.fit_predict(
           Predictand_dropna[[variable_column]]
        )
       
        # Convert cluster assignments back into an xarray structure
        df_unique = Predictand_dropna.drop_duplicates(subset=['Y', 'X'])
        dataset = df_unique.set_index(['Y', 'X']).to_xarray()
        mask = xr.where(~np.isnan(Predictand.isel(T=0)), 1, np.nan)
        Cluster = (dataset['cluster'] * mask)
              
        # Align cluster array with the predictand array
        xarray1, xarray2 = xr.align(Predictand, Cluster, join="outer")
       
        # Identify unique cluster labels
        clusters = np.unique(xarray2)
        clusters = clusters[~np.isnan(clusters)]
        cluster_da = xarray2
        y_train_std = standardize_timeseries(Predictand, clim_year_start, clim_year_end)
        X_train_std['T'] = y_train_std['T']
        # Prepare parameter distributions for RF and MLP
        param_dist_rf = {
            'n_estimators': self.n_estimators_range,
            'max_depth': self.max_depth_range,
            'min_samples_split': self.min_samples_split_range,
            'min_samples_leaf': self.min_samples_leaf_range,
            'max_features': self.max_features_range
        }
        param_dist_mlp = {
            'hidden_layer_sizes': self.hidden_layer_sizes_range,
            'activation': self.activation_options,
            'solver': self.solver_options,
            'alpha': self.alpha_range
        }
        best_params_dict = {}
        for c in clusters:
            mask_3d = (cluster_da == c).expand_dims({'T': y_train_std['T']})
            X_stacked_c = X_train_std.where(mask_3d).stack(sample=('T', 'Y', 'X')).transpose('sample', 'M').values
            y_stacked_c = y_train_std.where(mask_3d).stack(sample=('T', 'Y', 'X')).values.ravel()
            nan_mask_c = np.any(~np.isfinite(X_stacked_c), axis=1) | ~np.isfinite(y_stacked_c)
            X_clean_c = X_stacked_c[~nan_mask_c]
            y_clean_c = y_stacked_c[~nan_mask_c]
            if len(X_clean_c) == 0:
                continue
            # Tune Random Forest
            model_rf = RandomForestRegressor(random_state=self.random_state, n_jobs=-1)
            random_search_rf = RandomizedSearchCV(
                model_rf, param_distributions=param_dist_rf, n_iter=self.n_iter_search,
                cv=self.cv_folds, scoring='neg_mean_squared_error',
                random_state=self.random_state, error_score=np.nan
            )
            random_search_rf.fit(X_clean_c, y_clean_c)
            best_rf = random_search_rf.best_params_
            # Tune MLP
            model_mlp = MLPRegressor(max_iter=self.max_iter, random_state=self.random_state)
            random_search_mlp = RandomizedSearchCV(
                model_mlp, param_distributions=param_dist_mlp, n_iter=self.n_iter_search,
                cv=self.cv_folds, scoring='neg_mean_squared_error',
                random_state=self.random_state, error_score=np.nan
            )
            random_search_mlp.fit(X_clean_c, y_clean_c)
            best_mlp = random_search_mlp.best_params_
            best_params_dict[c] = {'rf': best_rf, 'mlp': best_mlp}
        return best_params_dict, cluster_da

    def compute_model(self, X_train, y_train, X_test, y_test, best_params=None, cluster_da=None):
        """
        Compute deterministic hindcast using the stacking ensemble of Random Forest (base) and MLP (meta) for each zone.
        Parameters
        ----------
        X_train : xarray.DataArray
            Training predictor data with dimensions (T, M, Y, X).
        y_train : xarray.DataArray
            Training predictand data with dimensions (T, Y, X).
        X_test : xarray.DataArray
            Testing predictor data with dimensions (T, M, Y, X).
        y_test : xarray.DataArray
            Testing predictand data with dimensions (T, Y, X).
        clim_year_start : int
            Start year of the climatology period.
        clim_year_end : int
            End year of the climatology period.
        best_params : dict, optional
            Pre-computed best hyperparameters per cluster. If None, computes internally.
        cluster_da : xarray.DataArray, optional
            Pre-computed cluster labels. If None, computes internally.
        Returns
        -------
        predicted_da : xarray.DataArray
            Deterministic hindcast with dimensions (T, Y, X).
        """
        # Standardize inputs
        X_train_std = X_train
        y_train_std = y_train
        X_test_std = X_test
        y_test_std = y_test
        
        # Extract coordinate variables from X_test
        time = X_test_std['T']
        lat = X_test_std['Y']
        lon = X_test_std['X']
        n_time = len(time)
        n_lat = len(lat)
        n_lon = len(lon)
        # Use provided best_params and cluster_da or compute if None
        if best_params is None:
            best_params, cluster_da = self.compute_hyperparameters(X_train, y_train, clim_year_start, clim_year_end)
        # Initialize predictions array
        predictions = np.full((n_time, n_lat, n_lon), np.nan)
        self.stacking_model = {}  # Dictionary to store models per cluster
        for c in range(self.n_clusters):
            if c not in best_params:
                continue
            bp = best_params[c]
            # Mask for this cluster
            mask_3d_train = (cluster_da == c).expand_dims({'T': X_train_std['T']})
            mask_3d_test = (cluster_da == c).expand_dims({'T': X_test_std['T']})
            # Stack training data for cluster
            X_train_stacked_c = X_train_std.where(mask_3d_train).stack(sample=('T', 'Y', 'X')).transpose('sample', 'M').values
            y_train_stacked_c = y_train_std.where(mask_3d_train).stack(sample=('T', 'Y', 'X')).values.ravel()
            train_nan_mask = np.any(~np.isfinite(X_train_stacked_c), axis=1) | ~np.isfinite(y_train_stacked_c)
            X_train_clean_c = X_train_stacked_c[~train_nan_mask]
            y_train_clean_c = y_train_stacked_c[~train_nan_mask]
            # Stack testing data for cluster
            X_test_stacked_c = X_test_std.where(mask_3d_test).stack(sample=('T', 'Y', 'X')).transpose('sample', 'M').values
            y_test_stacked_c = y_test_std.where(mask_3d_test).stack(sample=('T', 'Y', 'X')).values.ravel()
            test_nan_mask = np.any(~np.isfinite(X_test_stacked_c), axis=1) | ~np.isfinite(y_test_stacked_c)
            X_test_clean_c = X_test_stacked_c[~test_nan_mask]
            # Initialize base model (Random Forest)
            rf = RandomForestRegressor(**bp['rf'], random_state=self.random_state, n_jobs=-1)
            # Initialize meta-model (MLPRegressor)
            meta_model = MLPRegressor(**bp['mlp'], max_iter=self.max_iter, random_state=self.random_state)
            # Initialize stacking ensemble for this cluster
            stacking_model_c = StackingRegressor(estimators=[('rf', rf)], final_estimator=meta_model)
            # Fit the stacking model on cluster data
            stacking_model_c.fit(X_train_clean_c, y_train_clean_c)
            self.stacking_model[c] = stacking_model_c
            # Predict
            y_pred_c = stacking_model_c.predict(X_test_clean_c)
            # Reconstruct predictions for this cluster
            result_c = np.full(len(y_test_stacked_c), np.nan)
            result_c[~test_nan_mask] = y_pred_c
            pred_c_reshaped = result_c.reshape(n_time, n_lat, n_lon)
            # Fill in the predictions array
            predictions = np.where(np.isnan(predictions), pred_c_reshaped, predictions)
        predicted_da = xr.DataArray(
            data=predictions,
            coords={'T': time, 'Y': lat, 'X': lon},
            dims=['T', 'Y', 'X']
        )
        return predicted_da

    # ------------------ Probability Calculation Methods ------------------
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
            pred_prob[0, :] = t.cdf(first_t, df=dof)
            pred_prob[1, :] = t.cdf(second_t, df=dof) - t.cdf(first_t, df=dof)
            pred_prob[2, :] = 1 - t.cdf(second_t, df=dof)
        return pred_prob

    @staticmethod
    def calculate_tercile_probabilities_weibull_min(best_guess, error_variance, first_tercile, second_tercile, dof):
        """
        Weibull minimum-based method.
        """
        n_time = len(best_guess)
        pred_prob = np.empty((3, n_time))
        if np.all(np.isnan(best_guess)):
            pred_prob[:] = np.nan
        else:
            error_std = np.sqrt(error_variance)
            pred_prob[0, :] = weibull_min.cdf(first_tercile, c=dof, loc=best_guess, scale=error_std)
            pred_prob[1, :] = weibull_min.cdf(second_tercile, c=dof, loc=best_guess, scale=error_std) - \
                              weibull_min.cdf(first_tercile, c=dof, loc=best_guess, scale=error_std)
            pred_prob[2, :] = 1 - weibull_min.cdf(second_tercile, c=dof, loc=best_guess, scale=error_std)
        return pred_prob

    @staticmethod
    def calculate_tercile_probabilities_gamma(best_guess, error_variance, T1, T2, dof=None):
        """Gamma-distribution based method."""
        n_time = len(best_guess)
        pred_prob = np.empty((3, n_time), dtype=float)
        if np.any(np.isnan(best_guess)) or np.any(np.isnan(error_variance)):
            pred_prob[:] = np.nan
            return pred_prob
        best_guess = np.asarray(best_guess, dtype=float)
        error_variance = np.asarray(error_variance, dtype=float)
        T1 = np.asarray(T1, dtype=float)
        T2 = np.asarray(T2, dtype=float)
        alpha = (best_guess ** 2) / error_variance
        theta = error_variance / best_guess
        cdf_t1 = gamma.cdf(T1, a=alpha, scale=theta)
        cdf_t2 = gamma.cdf(T2, a=alpha, scale=theta)
        pred_prob[0, :] = cdf_t1
        pred_prob[1, :] = cdf_t2 - cdf_t1
        pred_prob[2, :] = 1.0 - cdf_t2
        return pred_prob

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

    @staticmethod
    def calculate_tercile_probabilities_normal(best_guess, error_variance, first_tercile, second_tercile):
        """Normal-distribution based method."""
        n_time = len(best_guess)
        pred_prob = np.empty((3, n_time))
        if np.all(np.isnan(best_guess)):
            pred_prob[:] = np.nan
        else:
            error_std = np.sqrt(error_variance)
            pred_prob[0, :] = norm.cdf(first_tercile, loc=best_guess, scale=error_std)
            pred_prob[1, :] = norm.cdf(second_tercile, loc=best_guess, scale=error_std) - norm.cdf(first_tercile, loc=best_guess, scale=error_std)
            pred_prob[2, :] = 1 - norm.cdf(second_tercile, loc=best_guess, scale=error_std)
        return pred_prob

    @staticmethod
    def calculate_tercile_probabilities_lognormal(best_guess, error_variance, first_tercile, second_tercile):
        """Lognormal-distribution based method."""
        n_time = len(best_guess)
        pred_prob = np.empty((3, n_time))
        if np.any(np.isnan(best_guess)) or np.any(np.isnan(error_variance)):
            pred_prob[:] = np.nan
            return pred_prob
        sigma = np.sqrt(np.log(1 + error_variance / (best_guess ** 2)))
        mu = np.log(best_guess) - sigma ** 2 / 2
        pred_prob[0, :] = lognorm.cdf(first_tercile, s=sigma, scale=np.exp(mu))
        pred_prob[1, :] = lognorm.cdf(second_tercile, s=sigma, scale=np.exp(mu)) - lognorm.cdf(first_tercile, s=sigma, scale=np.exp(mu))
        pred_prob[2, :] = 1 - lognorm.cdf(second_tercile, s=sigma, scale=np.exp(mu))
        return pred_prob

    def compute_prob(self, Predictant, clim_year_start, clim_year_end, hindcast_det):
        """
        Compute tercile probabilities using the chosen distribution method.
        Predictant is expected to be an xarray DataArray with dims (T, Y, X).
        """
        if "M" in Predictant.coords:
            Predictant = Predictant.isel(M=0).drop_vars('M').squeeze()
        mask = xr.where(~np.isnan(Predictant.isel(T=0)), 1, np.nan).drop_vars(['T']).squeeze().to_numpy()
        Predictant = Predictant.transpose('T', 'Y', 'X')
        index_start = Predictant.get_index("T").get_loc(str(clim_year_start)).start
        index_end = Predictant.get_index("T").get_loc(str(clim_year_end)).stop
        rainfall_for_tercile = Predictant.isel(T=slice(index_start, index_end))
        terciles = rainfall_for_tercile.quantile([0.33, 0.67], dim='T')
        error_variance = (Predictant - hindcast_det).var(dim='T')
        dof = len(Predictant.get_index("T")) - 2
        if self.dist_method == "t":
            calc_func = self.calculate_tercile_probabilities
            hindcast_prob = xr.apply_ufunc(
                calc_func,
                hindcast_det,
                error_variance,
                terciles.isel(quantile=0).drop_vars('quantile'),
                terciles.isel(quantile=1).drop_vars('quantile'),
                input_core_dims=[('T',), (), (), ()],
                vectorize=True,
                kwargs={'dof': dof},
                dask='parallelized',
                output_core_dims=[('probability', 'T')],
                output_dtypes=['float'],
                dask_gufunc_kwargs={'output_sizes': {'probability': 3}, "allow_rechunk": True}
            )
        elif self.dist_method == "weibull_min":
            calc_func = self.calculate_tercile_probabilities_weibull_min
            hindcast_prob = xr.apply_ufunc(
                calc_func,
                hindcast_det,
                error_variance,
                terciles.isel(quantile=0).drop_vars('quantile'),
                terciles.isel(quantile=1).drop_vars('quantile'),
                input_core_dims=[('T',), (), (), ()],
                vectorize=True,
                kwargs={'dof': dof},
                dask='parallelized',
                output_core_dims=[('probability', 'T')],
                output_dtypes=['float'],
                dask_gufunc_kwargs={'output_sizes': {'probability': 3}, "allow_rechunk": True}
            )
        elif self.dist_method == "gamma":
            calc_func = self.calculate_tercile_probabilities_gamma
            hindcast_prob = xr.apply_ufunc(
                calc_func,
                hindcast_det,
                error_variance,
                terciles.isel(quantile=0).drop_vars('quantile'),
                terciles.isel(quantile=1).drop_vars('quantile'),
                input_core_dims=[('T',), (), (), ()],
                vectorize=True,
                dask='parallelized',
                output_core_dims=[('probability', 'T')],
                output_dtypes=['float'],
                dask_gufunc_kwargs={'output_sizes': {'probability': 3}, "allow_rechunk": True}
            )
        elif self.dist_method == "nonparam":
            calc_func = self.calculate_tercile_probabilities_nonparametric
            error_samples = Predictant - hindcast_det
            hindcast_prob = xr.apply_ufunc(
                calc_func,
                hindcast_det,
                error_samples,
                terciles.isel(quantile=0).drop_vars('quantile'),
                terciles.isel(quantile=1).drop_vars('quantile'),
                input_core_dims=[('T',), ('T',), (), ()],
                output_core_dims=[('probability', 'T')],
                vectorize=True,
                dask='parallelized',
                output_dtypes=[float],
                dask_gufunc_kwargs={'output_sizes': {'probability': 3}, "allow_rechunk": True}
            )
        elif self.dist_method == "normal":
            calc_func = self.calculate_tercile_probabilities_normal
            hindcast_prob = xr.apply_ufunc(
                calc_func,
                hindcast_det,
                error_variance,
                terciles.isel(quantile=0).drop_vars('quantile'),
                terciles.isel(quantile=1).drop_vars('quantile'),
                input_core_dims=[('T',), (), (), ()],
                vectorize=True,
                dask='parallelized',
                output_core_dims=[('probability', 'T')],
                output_dtypes=['float'],
                dask_gufunc_kwargs={'output_sizes': {'probability': 3}, "allow_rechunk": True}
            )
        elif self.dist_method == "lognormal":
            calc_func = self.calculate_tercile_probabilities_lognormal
            hindcast_prob = xr.apply_ufunc(
                calc_func,
                hindcast_det,
                error_variance,
                terciles.isel(quantile=0).drop_vars('quantile'),
                terciles.isel(quantile=1).drop_vars('quantile'),
                input_core_dims=[('T',), (), (), ()],
                vectorize=True,
                dask='parallelized',
                output_core_dims=[('probability', 'T')],
                output_dtypes=['float'],
                dask_gufunc_kwargs={'output_sizes': {'probability': 3}, "allow_rechunk": True}
            )
        else:
            raise ValueError(f"Invalid dist_method: {self.dist_method}.")
        hindcast_prob = hindcast_prob.assign_coords(probability=('probability', ['PB', 'PN', 'PA']))
        return hindcast_prob.transpose('probability', 'T', 'Y', 'X')

    @staticmethod
    def _reshape_and_filter_data(da):
        """
        Helper: stack the DataArray from (T, Y, X[, M]) to (n_samples, n_features)
        and remove rows containing NaNs.
        """
        da_stacked = da.stack(sample=('T', 'Y', 'X'))
        if 'M' in da.dims:
            da_stacked = da_stacked.transpose('sample', 'M')
        else:
            da_stacked = da_stacked.transpose('sample')
        da_values = da_stacked.values
        nan_mask = np.any(np.isnan(da_values), axis=1)
        return da_values[~nan_mask], nan_mask, da_values

    def forecast(self, Predictant, clim_year_start, clim_year_end, hindcast_det, hindcast_det_cross, Predictor_for_year, best_params=None, cluster_da=None):
        """
        Generate deterministic and probabilistic forecast for a target year using the stacking ensemble.
        Parameters
        ----------
        Predictant : xarray.DataArray
            Observed predictand data with dimensions (T, Y, X) or (T, M, Y, X).
        clim_year_start : int or str
            Start year of the climatology period.
        clim_year_end : int or str
            End year of the climatology period.
        hindcast_det : xarray.DataArray
            Deterministic hindcast data for training with dimensions (T, M, Y, X).
        hindcast_det_cross : xarray.DataArray
            Deterministic hindcast data for error estimation with dimensions (T, Y, X).
        Predictor_for_year : xarray.DataArray
            Predictor data for the target year with dimensions (T, M, Y, X).
        best_params : dict, optional
            Pre-computed best hyperparameters per cluster. If None, computes internally.
        cluster_da : xarray.DataArray, optional
            Pre-computed cluster labels. If None, computes internally.
        Returns
        -------
        forecast_det : xarray.DataArray
            Deterministic forecast with dimensions (T, Y, X).
        forecast_prob : xarray.DataArray
            Tercile probabilities with dimensions (probability, T, Y, X).
        """
        if "M" in Predictant.coords:
            Predictant_no_m = Predictant.isel(M=0).drop_vars('M').squeeze()
        else:
            Predictant_no_m = Predictant
        mask = xr.where(~np.isnan(Predictant_no_m.isel(T=0)), 1, np.nan).drop_vars(['T']).squeeze().to_numpy()
        # Standardize Predictor_for_year using hindcast climatology
        mean_val = hindcast_det.sel(T=slice(str(clim_year_start), str(clim_year_end))).mean(dim='T')
        std_val = hindcast_det.sel(T=slice(str(clim_year_start), str(clim_year_end))).std(dim='T')
        Predictor_for_year_st = (Predictor_for_year - mean_val) / std_val
        hindcast_det_st = standardize_timeseries(hindcast_det, clim_year_start, clim_year_end)
        Predictant_st = standardize_timeseries(Predictant_no_m, clim_year_start, clim_year_end)
        # Extract coordinates from X_test
        time = Predictor_for_year_st['T']
        lat = Predictor_for_year_st['Y']
        lon = Predictor_for_year_st['X']
        n_time = len(time)
        n_lat = len(lat)
        n_lon = len(lon)
        # Use provided best_params and cluster_da or compute if None
        if best_params is None:
            best_params, cluster_da = self.compute_hyperparameters(hindcast_det, Predictant_no_m, clim_year_start, clim_year_end)
        # Initialize predictions array
        predictions = np.full((n_time, n_lat, n_lon), np.nan)
        self.stacking_model = {}  # Dictionary to store models per cluster
        for c in range(self.n_clusters):
            if c not in best_params:
                continue
            bp = best_params[c]
            # Mask for this cluster
            mask_3d_train = (cluster_da == c).expand_dims({'T': hindcast_det_st['T']})
            mask_3d_test = (cluster_da == c).expand_dims({'T': Predictor_for_year_st['T']})
            # Stack training data for cluster
            X_train_stacked_c = hindcast_det_st.where(mask_3d_train).stack(sample=('T', 'Y', 'X')).transpose('sample', 'M').values
            y_train_stacked_c = Predictant_st.where(mask_3d_train).stack(sample=('T', 'Y', 'X')).values.ravel()
            train_nan_mask = np.any(~np.isfinite(X_train_stacked_c), axis=1) | ~np.isfinite(y_train_stacked_c)
            X_train_clean_c = X_train_stacked_c[~train_nan_mask]
            y_train_clean_c = y_train_stacked_c[~train_nan_mask]
            # Stack testing data for cluster
            X_test_stacked_c = Predictor_for_year_st.where(mask_3d_test).stack(sample=('T', 'Y', 'X')).transpose('sample', 'M').values
            test_nan_mask = np.any(~np.isfinite(X_test_stacked_c), axis=1)
            X_test_clean_c = X_test_stacked_c[~test_nan_mask]
            # Initialize base model (Random Forest)
            rf = RandomForestRegressor(**bp['rf'], random_state=self.random_state)
            # Initialize meta-model (MLPRegressor)
            meta_model = MLPRegressor(**bp['mlp'], max_iter=self.max_iter, random_state=self.random_state)
            # Initialize stacking ensemble for this cluster
            stacking_model_c = StackingRegressor(estimators=[('rf', rf)], final_estimator=meta_model)
            # Fit the stacking model on cluster data
            stacking_model_c.fit(X_train_clean_c, y_train_clean_c)
            self.stacking_model[c] = stacking_model_c
            # Predict
            y_pred_c = stacking_model_c.predict(X_test_clean_c)
            # Reconstruct predictions for this cluster
            result_c = np.full(len(X_test_stacked_c), np.nan)
            result_c[~test_nan_mask] = y_pred_c
            pred_c_reshaped = result_c.reshape(n_time, n_lat, n_lon)
            # Fill in the predictions array
            predictions = np.where(np.isnan(predictions), pred_c_reshaped, predictions)
        result_da = xr.DataArray(
            data=predictions,
            coords={'T': time, 'Y': lat, 'X': lon},
            dims=['T', 'Y', 'X']
        ) * mask
        result_da = reverse_standardize(result_da, Predictant_no_m, clim_year_start, clim_year_end)
        year = Predictor_for_year.coords['T'].values.astype('datetime64[Y]').astype(int)[0] + 1970
        T_value_1 = Predictant_no_m.isel(T=0).coords['T'].values
        month_1 = T_value_1.astype('datetime64[M]').astype(int) % 12 + 1
        new_T_value = np.datetime64(f"{year}-{month_1:02d}-01")
        result_da = result_da.assign_coords(T=xr.DataArray([new_T_value], dims=["T"]))
        result_da['T'] = result_da['T'].astype('datetime64[ns]')
        # Compute tercile probabilities
        index_start = Predictant_no_m.get_index("T").get_loc(str(clim_year_start)).start
        index_end = Predictant_no_m.get_index("T").get_loc(str(clim_year_end)).stop
        rainfall_for_tercile = Predictant_no_m.isel(T=slice(index_start, index_end))
        terciles = rainfall_for_tercile.quantile([0.33, 0.67], dim='T')
        error_variance = (Predictant_no_m - hindcast_det_cross).var(dim='T')
        dof = len(Predictant_no_m.get_index("T")) - 2
        if self.dist_method == "t":
            calc_func = self.calculate_tercile_probabilities
            hindcast_prob = xr.apply_ufunc(
                calc_func,
                result_da,
                error_variance,
                terciles.isel(quantile=0).drop_vars('quantile'),
                terciles.isel(quantile=1).drop_vars('quantile'),
                input_core_dims=[('T',), (), (), ()],
                vectorize=True,
                kwargs={'dof': dof},
                dask='parallelized',
                output_core_dims=[('probability', 'T')],
                output_dtypes=['float'],
                dask_gufunc_kwargs={'output_sizes': {'probability': 3}, "allow_rechunk": True}
            )
        elif self.dist_method == "weibull_min":
            calc_func = self.calculate_tercile_probabilities_weibull_min
            hindcast_prob = xr.apply_ufunc(
                calc_func,
                result_da,
                error_variance,
                terciles.isel(quantile=0).drop_vars('quantile'),
                terciles.isel(quantile=1).drop_vars('quantile'),
                input_core_dims=[('T',), (), (), ()],
                vectorize=True,
                kwargs={'dof': dof},
                dask='parallelized',
                output_core_dims=[('probability', 'T')],
                output_dtypes=['float'],
                dask_gufunc_kwargs={'output_sizes': {'probability': 3}, "allow_rechunk": True}
            )
        elif self.dist_method == "gamma":
            calc_func = self.calculate_tercile_probabilities_gamma
            hindcast_prob = xr.apply_ufunc(
                calc_func,
                result_da,
                error_variance,
                terciles.isel(quantile=0).drop_vars('quantile'),
                terciles.isel(quantile=1).drop_vars('quantile'),
                input_core_dims=[('T',), (), (), ()],
                vectorize=True,
                dask='parallelized',
                output_core_dims=[('probability', 'T')],
                output_dtypes=['float'],
                dask_gufunc_kwargs={'output_sizes': {'probability': 3}, "allow_rechunk": True}
            )
        elif self.dist_method == "normal":
            calc_func = self.calculate_tercile_probabilities_normal
            hindcast_prob = xr.apply_ufunc(
                calc_func,
                result_da,
                error_variance,
                terciles.isel(quantile=0).drop_vars('quantile'),
                terciles.isel(quantile=1).drop_vars('quantile'),
                input_core_dims=[('T',), (), (), ()],
                vectorize=True,
                dask='parallelized',
                output_core_dims=[('probability', 'T')],
                output_dtypes=['float'],
                dask_gufunc_kwargs={'output_sizes': {'probability': 3}, "allow_rechunk": True}
            )
        elif self.dist_method == "lognormal":
            calc_func = self.calculate_tercile_probabilities_lognormal
            hindcast_prob = xr.apply_ufunc(
                calc_func,
                result_da,
                error_variance,
                terciles.isel(quantile=0).drop_vars('quantile'),
                terciles.isel(quantile=1).drop_vars('quantile'),
                input_core_dims=[('T',), (), (), ()],
                vectorize=True,
                dask='parallelized',
                output_core_dims=[('probability', 'T')],
                output_dtypes=['float'],
                dask_gufunc_kwargs={'output_sizes': {'probability': 3}, "allow_rechunk": True}
            )
        elif self.dist_method == "nonparam":
            calc_func = self.calculate_tercile_probabilities_nonparametric
            error_samples = Predictant_no_m - hindcast_det_cross
            error_samples = error_samples.rename({'T': 'S'})
            hindcast_prob = xr.apply_ufunc(
                calc_func,
                result_da,
                error_samples,
                terciles.isel(quantile=0).drop_vars('quantile'),
                terciles.isel(quantile=1).drop_vars('quantile'),
                input_core_dims=[('T',), ('S',), (), ()],
                output_core_dims=[('probability', 'T')],
                vectorize=True,
                dask='parallelized',
                output_dtypes=[float],
                dask_gufunc_kwargs={'output_sizes': {'probability': 3}, "allow_rechunk": True}
            )
        else:
            raise ValueError(f"Invalid dist_method: {self.dist_method}.")
        hindcast_prob = hindcast_prob.assign_coords(probability=('probability', ['PB', 'PN', 'PA']))
        return result_da * mask, mask * hindcast_prob.transpose('probability', 'T', 'Y', 'X')
