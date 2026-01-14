from __future__ import annotations

import operator
import random
import gc
import datetime
import warnings
from dataclasses import dataclass
from typing import Literal, Tuple, Optional, Dict, List

import numpy as np
import pandas as pd
import xarray as xr
import pymc as pm
import arviz as az

from scipy import stats
from scipy.optimize import minimize, minimize_scalar, fsolve
from scipy.special import gamma as gamma_function
from scipy.special import expit
from scipy.stats import (
    norm, lognorm, expon, gamma, weibull_min, t, poisson, nbinom,
    logistic, genextreme, laplace, pareto, loguniform, randint, uniform,
    linregress, t as tdist
)

from tqdm.auto import tqdm

from sklearn.base import BaseEstimator, RegressorMixin
from sklearn.cluster import KMeans
from sklearn.ensemble import (
    RandomForestRegressor, StackingRegressor,
    GradientBoostingRegressor, AdaBoostRegressor
)
from sklearn.linear_model import LinearRegression, LogisticRegression
from sklearn.neural_network import MLPRegressor
from sklearn.svm import SVR
from sklearn.gaussian_process import GaussianProcessRegressor, GaussianProcessClassifier
from sklearn.gaussian_process.kernels import RBF, Matern, ConstantKernel as C, WhiteKernel
from sklearn.model_selection import GridSearchCV, RandomizedSearchCV, KFold
from sklearn.metrics import mean_squared_error, log_loss
from sklearn.exceptions import ConvergenceWarning

from xgboost import XGBRegressor

# Optional / third-party imports
try:
    from tqdm import tqdm  # fallback if tqdm.auto not available
except ImportError:
    def tqdm(iterable, **kwargs):
        return iterable

try:
    from joblib import Parallel, delayed
    JOBLIB_AVAILABLE = True
except ImportError:
    JOBLIB_AVAILABLE = False

try:
    import optuna
    from optuna.samplers import TPESampler
    OPTUNA_AVAILABLE = True
except ImportError:
    OPTUNA_AVAILABLE = False
    warnings.warn("Optuna not installed. Bayesian optimization will not be available.")

try:
    from hpelm import HPELM
except ImportError:
    HPELM = None  # or raise/warn as needed

from sklearn.model_selection import GridSearchCV, RandomizedSearchCV
from sklearn.base import BaseEstimator, RegressorMixin
import optuna
from optuna.samplers import TPESampler, RandomSampler
from functools import partial
import numpy as np
import xarray as xr
from hpelm import HPELM


from dask.distributed import Client

# Project-specific imports
from wass2s.utils import *
from wass2s.was_verification import *
import xcast as xc

# Suppress specific warnings for cleaner output
warnings.filterwarnings('ignore', category=RuntimeWarning)
warnings.filterwarnings('ignore', category=ConvergenceWarning)

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
    def __init__(self, equal_weighted=False, dist_method="nonparam", metric="GROC", threshold=0.5):
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
                    norm.ppf(0.33, loc=loc, scale=scale),
                    norm.ppf(0.67, loc=loc, scale=scale),
                )
            elif code == 2:
                return (
                    lognorm.ppf(0.33, s=shape, loc=loc, scale=scale),
                    lognorm.ppf(0.67, s=shape, loc=loc, scale=scale),
                )
            elif code == 3:
                return (
                    expon.ppf(0.33, loc=loc, scale=scale),
                    expon.ppf(0.67, loc=loc, scale=scale),
                )
            elif code == 4:
                return (
                    gamma.ppf(0.33, a=shape, loc=loc, scale=scale),
                    gamma.ppf(0.67, a=shape, loc=loc, scale=scale),
                )
            elif code == 5:
                return (
                    weibull_min.ppf(0.33, c=shape, loc=loc, scale=scale),
                    weibull_min.ppf(0.67, c=shape, loc=loc, scale=scale),
                )
            elif code == 6:
                # Note: Renamed 't_dist' to 't' for standard scipy.stats
                return (
                    t.ppf(0.33, df=shape, loc=loc, scale=scale),
                    t.ppf(0.67, df=shape, loc=loc, scale=scale),
                )
            elif code == 7:
                # Poisson: poisson.ppf(q, mu, loc=0)
                # ASSUMPTION: 'mu' (mean) is passed as 'shape'
                #             'loc' is passed as 'loc'
                #             'scale' is unused
                return (
                    poisson.ppf(0.33, mu=shape, loc=loc),
                    poisson.ppf(0.67, mu=shape, loc=loc),
                )
            elif code == 8:
                # Negative Binomial: nbinom.ppf(q, n, p, loc=0)
                # ASSUMPTION: 'n' (successes) is passed as 'shape'
                #             'p' (probability) is passed as 'scale'
                #             'loc' is passed as 'loc'
                return (
                    nbinom.ppf(0.33, n=shape, p=scale, loc=loc),
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
        terciles_emp = clim.quantile([0.33, 0.67], dim="T")
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


    def forecast(self, Predictant, clim_year_start, clim_year_end, hindcast_det, forecast_det, best_code_da=None, best_shape_da=None, best_loc_da=None, best_scale_da=None):
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
        T1_emp = terciles.isel(quantile=0).drop_vars('quantile')
        T2_emp = terciles.isel(quantile=1).drop_vars('quantile')
        error_variance = (Predictant - hindcast_det).var(dim='T')
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
                forecast_det,
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
                forecast_det,
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
        return forecast_det * mask, mask * forecast_prob.transpose('probability', 'T', 'Y', 'X')


class WAS_ProbWeighted:
    """
    Probability-Weighted Multi-Model Ensemble for seasonal rainfall forecasting.

    Implements a threshold-based weighting scheme for combining multiple climate
    model outputs (hindcasts and forecasts). Model weights are derived from performance.
    It's currently for only GROC scores. With a stepwise transformation:
    - ≤ threshold → weight = 0 (excluded)
    - threshold < score ≤ 0.6 → weight = 0.6
    - 0.6 < score ≤ 0.8 → weight = 0.8
    - > 0.8 → weight = 1.0

    This approach aims to emphasize better-performing models while maintaining some
    contribution from moderately skilled ones, and completely excluding very poor models.

    Parameters
    ----------
    None

    Notes
    -----
    - All input data (hindcast, forecast, scores) are interpolated to the rainfall grid
      using nearest-neighbor interpolation with extrapolation.
    - A spatial mask is derived from the first time step of rainfall observations.
    - Supports an optional 'complete' mode that fills missing values with simple
      unweighted ensemble mean.
    - Designed primarily for seasonal climate forecasting applications.
    """

    def __init__(self):
        # Initialize any required attributes here
        pass

    def compute(self, rainfall, hdcst, fcst, scores, threshold=0.5, complete=False):
        """
        Compute probability-weighted multi-model ensemble mean for hindcast and forecast.

        Parameters
        ----------
        rainfall : xarray.DataArray
            Observed rainfall used as reference grid and for masking.
            Expected dimensions: (T, Y, X, M) or (T, Y, X).
            The M dimension (if present) is ignored.
        hdcst : xarray.DataArray
            Multi-model hindcast dataset.
            Expected dimensions: (T, M, Y, X)
        fcst : xarray.DataArray
            Multi-model forecast dataset (single lead time).
            Expected dimensions: (T, M, Y, X)
        scores : dict
            Dictionary mapping model names (str) to performance score arrays.
            Each score array should be an xarray.DataArray with spatial coordinates (Y, X).
        threshold : float, default=0.5 for GROC
            Minimum score value below which a model is completely excluded (weight = 0).
        complete : bool, default=False
            If True, areas where weighted mean is NaN are filled with the simple
            unweighted ensemble mean.

        Returns
        -------
        hindcast_weighted : xarray.DataArray
            Weighted hindcast ensemble mean.
            Dimensions: (T, Y, X)
        forecast_weighted : xarray.DataArray
            Weighted forecast ensemble mean.
            Dimensions: (T, Y, X)

        Notes
        -----
        - Time coordinate of forecast is adjusted to match the expected seasonal month
          based on the first timestep of rainfall observations.
        - All interpolations use 'nearest' method with extrapolation for boundary points.
        - Final output is masked using the spatial coverage of observed rainfall.
        - Models with NaN scores or failed interpolation will contribute zero weight.
        - The stepwise weighting is currently hardcoded (0.6 / 0.8 / 1.0 breakpoints). 
        - It's only for GROC currently.
 

        Warnings
        --------
        - Make sure that model names in `hdcst.M`, `fcst.M` and `scores.keys()` match exactly.
        - Performance may degrade if score grids have very different resolution/spatial extent
          from the rainfall target grid.
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

            score_array = xr.where(
               score_array <= threshold,
                0,
                xr.where(
                    score_array <= 0.6,
                    0.6,
                   xr.where(score_array <= 0.8, 0.8, 1)
                )
            )

            # score_array = xr.where(
            #     score_array <= threshold,
            #     0,1
            # )
            
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






class WAS_Min2009_ProbWeighted:
    """
    Implementation of Min et al. (2009) Probability-Weighted Multi-Model Ensemble.
    
    Based on: "Probabilistic Multimodel Ensemble (PMME) forecasting at APCC"
    Journal: Weather and Forecasting, 2009
    
    Key methodology:
    1. Individual model probabilistic forecasts are computed using Gaussian approximation
    2. Model weights are proportional to sqrt(ensemble_size) [Eq. 6 in paper]
    3. Tercile probabilities (BN, NN, AN) are combined using total probability formula
    
    Notes:
    - For temperature: Gaussian assumption is reasonable
    - For precipitation: Consider using log-normal, Gamma distribution, or empirical methods
    - The χ² test uses configurable n (ensemble size) - avoid overly conservative n=1
    - Cross-validation is recommended for hindcast statistics
    """
    
    def __init__(self, distribution='gaussian', cv_method=None, n_samples_for_chisq='total_ensemble'):
        """
        Initialize PMME processor.
        
        Parameters
        ----------
        distribution : str
            Distribution assumption: 'gaussian', 'gamma', 'lognormal', or 'empirical'
        cv_method : str or None
            Cross-validation method: None, 'leave_one_out', or 'rolling_window'
        n_samples_for_chisq : str or int
            How to compute n for χ² test: 
            'total_ensemble' (default), 'effective_sample_size', or integer value
        """
        self.distribution = distribution
        self.cv_method = cv_method
        self.n_samples_for_chisq = n_samples_for_chisq
        
    def _compute_cross_validated_stats(self, hindcasts, climatology):
        """
        Compute cross-validated mean and standard deviation from hindcasts.
        
        Parameters
        ----------
        hindcasts : xarray.DataArray
            Hindcast ensemble data with dimensions (T, M, Y, X)
        climatology : xarray.DataArray
            Climatological data with dimensions (Y, X)
            
        Returns
        -------
        mu_cv : xarray.DataArray
            Cross-validated mean with dimensions (T, Y, X)
        sigma_cv : xarray.DataArray
            Cross-validated standard deviation with dimensions (T, Y, X)
        """
        n_times = hindcasts.sizes['time']
        
        if self.cv_method is None:
            # Use full hindcast period (not recommended for operational use)
            mu_cv = hindcasts.mean(dim=['time', 'ensemble'])
            sigma_cv = hindcasts.std(dim=['time', 'ensemble'])
            # Expand to match time dimension
            mu_cv = mu_cv.expand_dims(time=hindcasts.time).transpose('time', 'lat', 'lon')
            sigma_cv = sigma_cv.expand_dims(time=hindcasts.time).transpose('time', 'lat', 'lon')
            
        elif self.cv_method == 'leave_one_out':
            # Leave-one-out cross-validation
            mu_list = []
            sigma_list = []
            
            for i in range(n_times):
                # Leave out year i
                hindcast_train = hindcasts.isel(time=[j for j in range(n_times) if j != i])
                mu_i = hindcast_train.mean(dim=['time', 'ensemble'])
                sigma_i = hindcast_train.std(dim=['time', 'ensemble'])
                mu_list.append(mu_i)
                sigma_list.append(sigma_i)
            
            mu_cv = xr.concat(mu_list, dim=hindcasts.time)
            sigma_cv = xr.concat(sigma_list, dim=hindcasts.time)
            
        elif self.cv_method == 'rolling_window':
            # Rolling window validation (e.g., 15-year window)
            window_size = 15
            mu_list = []
            sigma_list = []
            
            for i in range(n_times):
                start = max(0, i - window_size // 2)
                end = min(n_times, i + window_size // 2 + 1)
                hindcast_train = hindcasts.isel(time=slice(start, end))
                # Exclude the current year if possible
                hindcast_train = hindcast_train.isel(time=[j for j in range(hindcast_train.sizes['time']) 
                                                          if start + j != i])
                
                mu_i = hindcast_train.mean(dim=['time', 'ensemble'])
                sigma_i = hindcast_train.std(dim=['time', 'ensemble'])
                mu_list.append(mu_i)
                sigma_list.append(sigma_i)
            
            mu_cv = xr.concat(mu_list, dim=hindcasts.time)
            sigma_cv = xr.concat(sigma_list, dim=hindcasts.time)
        
        return mu_cv, sigma_cv
    
    def _compute_tercile_probabilities(self, forecasts, hindcasts, climatology):
        """
        Compute tercile probabilities for individual models.
        
        Parameters
        ----------
        forecasts : xarray.DataArray
            Forecast ensemble data with dimensions (T, M, Y, X)
        hindcasts : xarray.DataArray
            Hindcast ensemble data with dimensions (T, M, Y, X)
        climatology : xarray.DataArray
            Climatological data with dimensions (Y, X)
            
        Returns
        -------
        probs_bn : xarray.DataArray
            Probability of below-normal category
        probs_nn : xarray.DataArray
            Probability of near-normal category
        probs_an : xarray.DataArray
            Probability of above-normal category
        """
        # Compute cross-validated statistics
        mu_cv, sigma_cv = self._compute_cross_validated_stats(hindcasts, climatology)
        
        # Compute forecast ensemble mean
        forecast_mean = forecasts.mean(dim='ensemble')
        
        if self.distribution == 'gaussian':
            # Gaussian approximation (suitable for temperature)
            # Tercile boundaries at approximately ±0.43σ (for Gaussian)
            # Actually, for terciles, boundaries are at Φ⁻¹(1/3) ≈ -0.43 and Φ⁻¹(2/3) ≈ 0.43
            # The paper uses ±1.43σ which seems incorrect - this would be for much wider intervals
            # Let's use the correct tercile boundaries:
            lower_boundary = -0.4307  # Φ⁻¹(1/3)
            upper_boundary = 0.4307   # Φ⁻¹(2/3)
            
            # Standardized anomalies
            z_lower = (mu_cv + lower_boundary * sigma_cv - forecast_mean) / sigma_cv
            z_upper = (mu_cv + upper_boundary * sigma_cv - forecast_mean) / sigma_cv
            
            # Gaussian CDF probabilities
            probs_bn = stats.norm.cdf(z_lower)
            probs_an = 1 - stats.norm.cdf(z_upper)
            probs_nn = 1 - probs_bn - probs_an
            
        elif self.distribution == 'lognormal':
            # Log-normal distribution (suitable for precipitation)
            # Transform to log space
            log_hindcasts = xr.where(hindcasts > 0, np.log(hindcasts), np.log(0.01))
            log_mu_cv, log_sigma_cv = self._compute_cross_validated_stats(log_hindcasts, climatology)
            log_forecast = xr.where(forecasts > 0, np.log(forecasts), np.log(0.01))
            log_forecast_mean = log_forecast.mean(dim='ensemble')
            
            # Compute tercile boundaries in log space
            lower_boundary = -0.4307
            upper_boundary = 0.4307
            
            z_lower = (log_mu_cv + lower_boundary * log_sigma_cv - log_forecast_mean) / log_sigma_cv
            z_upper = (log_mu_cv + upper_boundary * log_sigma_cv - log_forecast_mean) / log_sigma_cv
            
            probs_bn = stats.norm.cdf(z_lower)
            probs_an = 1 - stats.norm.cdf(z_upper)
            probs_nn = 1 - probs_bn - probs_an
            
        elif self.distribution == 'empirical':
            # Empirical quantile mapping
            # This is a simplified version - consider more sophisticated methods
            forecast_flat = forecasts.stack(sample=('time', 'ensemble')).transpose('sample', 'lat', 'lon')
            hindcast_flat = hindcasts.stack(sample=('time', 'ensemble')).transpose('sample', 'lat', 'lon')
            
            # Compute empirical CDF for each grid point
            probs_bn = xr.full_like(forecast_mean, fill_value=np.nan)
            probs_nn = xr.full_like(forecast_mean, fill_value=np.nan)
            probs_an = xr.full_like(forecast_mean, fill_value=np.nan)
            
            # This is computationally intensive - consider optimization
            for lat in forecast_mean.lat.values:
                for lon in forecast_mean.lon.values:
                    hindcast_vals = hindcast_flat.sel(lat=lat, lon=lon).values
                    forecast_val = forecast_mean.sel(lat=lat, lon=lon).values
                    
                    # Compute empirical terciles from hindcast
                    lower_tercile = np.percentile(hindcast_vals, 100/3)
                    upper_tercile = np.percentile(hindcast_vals, 100*2/3)
                    
                    # Empirical probabilities
                    probs_bn.loc[dict(lat=lat, lon=lon)] = np.mean(forecast_val < lower_tercile)
                    probs_an.loc[dict(lat=lat, lon=lon)] = np.mean(forecast_val > upper_tercile)
                    probs_nn.loc[dict(lat=lat, lon=lon)] = 1 - probs_bn.loc[dict(lat=lat, lon=lon)] - probs_an.loc[dict(lat=lat, lon=lon)]
        
        # Ensure probabilities are between 0 and 1
        probs_bn = xr.where(probs_bn < 0, 0, xr.where(probs_bn > 1, 1, probs_bn))
        probs_an = xr.where(probs_an < 0, 0, xr.where(probs_an > 1, 1, probs_an))
        probs_nn = xr.where(probs_nn < 0, 0, xr.where(probs_nn > 1, 1, probs_nn))
        
        # Renormalize to ensure sum to 1 (accounting for numerical errors)
        total = probs_bn + probs_nn + probs_an
        probs_bn = probs_bn / total
        probs_nn = probs_nn / total
        probs_an = probs_an / total
        
        return probs_bn, probs_nn, probs_an
    
    def _compute_model_weights(self, ensemble_sizes):
        """
        Compute model weights according to Min et al. (2009) Eq. 6.
        
        Parameters
        ----------
        ensemble_sizes : dict
            Dictionary mapping model names to ensemble sizes
            
        Returns
        -------
        weights : dict
            Dictionary mapping model names to normalized weights
        """
        # Weight proportional to sqrt(ensemble_size)
        sqrt_sizes = {model: np.sqrt(size) for model, size in ensemble_sizes.items()}
        total = sum(sqrt_sizes.values())
        
        # Normalize so weights sum to 1
        weights = {model: sqrt_sizes[model] / total for model in ensemble_sizes.keys()}
        
        return weights
    
    def _compute_n_for_chisq(self, ensemble_sizes, model_names):
        """
        Compute n for χ² test based on configuration.
        
        Parameters
        ----------
        ensemble_sizes : dict
            Dictionary mapping model names to ensemble sizes
        model_names : list
            List of model names
            
        Returns
        -------
        n : float
            Value to use for n in χ² test
        """
        if isinstance(self.n_samples_for_chisq, (int, float)):
            return float(self.n_samples_for_chisq)
        elif self.n_samples_for_chisq == 'total_ensemble':
            # Sum of all ensemble members across models
            return sum(ensemble_sizes[model] for model in model_names)
        elif self.n_samples_for_chisq == 'effective_sample_size':
            # Approximate effective sample size
            total_ensemble = sum(ensemble_sizes[model] for model in model_names)
            n_models = len(model_names)
            # Simple approximation: account for correlation between models
            return total_ensemble / np.sqrt(n_models)
        else:
            # Default to total ensemble size
            return sum(ensemble_sizes[model] for model in model_names)
    
    def compute_pmme_probabilities(self, forecasts, hindcasts, climatology, ensemble_sizes):
        """
        Compute PMME probabilities according to Min et al. (2009).
        
        Parameters
        ----------
        forecasts : dict of xarray.DataArray
            Dictionary of forecast ensembles for each model
        hindcasts : dict of xarray.DataArray
            Dictionary of hindcast ensembles for each model
        climatology : xarray.DataArray
            Climatological data
        ensemble_sizes : dict
            Dictionary mapping model names to ensemble sizes
            
        Returns
        -------
        pmme_probs : dict
            Dictionary with keys 'BN', 'NN', 'AN' containing PMME probabilities
        """
        # Get model names
        model_names = list(forecasts.keys())
        
        # Compute individual model probabilities
        model_probs = {}
        for model in model_names:
            probs_bn, probs_nn, probs_an = self._compute_tercile_probabilities(
                forecasts[model], hindcasts[model], climatology
            )
            model_probs[model] = {
                'BN': probs_bn,
                'NN': probs_nn,
                'AN': probs_an
            }
        
        # Compute model weights (Eq. 6)
        weights = self._compute_model_weights(ensemble_sizes)
        
        # Combine probabilities using total probability formula (Eq. 1)
        pmme_probs = {}
        for category in ['BN', 'NN', 'AN']:
            weighted_sum = xr.zeros_like(next(iter(model_probs.values()))[category])
            for model in model_names:
                weighted_prob = model_probs[model][category] * weights[model]
                weighted_sum = weighted_sum + weighted_prob
            
            pmme_probs[category] = weighted_sum
        
        return pmme_probs
    
    def compute_combined_map(self, pmme_probs, ensemble_sizes, model_names, significance_level=0.05):
        """
        Compute combined map with significance testing (χ² test).
        
        Parameters
        ----------
        pmme_probs : dict
            PMME probabilities for BN, NN, AN categories
        ensemble_sizes : dict
            Dictionary mapping model names to ensemble sizes
        model_names : list
            List of model names
        significance_level : float
            Significance level for χ² test (default 0.05)
            
        Returns
        -------
        combined_map : xarray.DataArray
            Combined map showing dominant category where significant
        chi_square : xarray.DataArray
            χ² statistic values
        """
        # Find dominant category
        probs_array = xr.concat([pmme_probs['BN'], pmme_probs['NN'], pmme_probs['AN']], 
                               dim='category')
        dominant_category = probs_array.argmax(dim='category')
        
        # Compute n for χ² test
        n = self._compute_n_for_chisq(ensemble_sizes, model_names)
        
        # Compute χ² statistic (Eq. in section 5)
        # χ² = n * Σ (P(Ej) - 1/3)² / (1/3)
        expected_prob = 1/3
        
        chi_square = n * (
            (pmme_probs['BN'] - expected_prob)**2 / expected_prob +
            (pmme_probs['NN'] - expected_prob)**2 / expected_prob +
            (pmme_probs['AN'] - expected_prob)**2 / expected_prob
        )
        
        # Critical value for χ² with 2 degrees of freedom
        critical_value = stats.chi2.ppf(1 - significance_level, df=2)
        
        # Create combined map: show dominant category where significant
        combined_map = xr.where(
            chi_square > critical_value,
            dominant_category + 1,  # 1 for BN, 2 for NN, 3 for AN
            0  # No significant deviation from climatology
        )
        
        # Add attributes for interpretation
        combined_map.attrs = {
            'description': 'PMME combined forecast map',
            'values': '0=no significant deviation, 1=BN, 2=NN, 3=AN',
            'significance_level': significance_level,
            'chi2_critical_value': critical_value
        }
        
        return combined_map, chi_square


# # Example usage function
# def example_usage():
#     """
#     Example of how to use the PMME class with proper cross-validation.
#     """
#     # Initialize with cross-validation and appropriate settings
#     pmme = WAS_Min2009_ProbWeighted(
#         distribution='gaussian',  # Use 'gamma' or 'lognormal' for precipitation
#         cv_method='leave_one_out',  # Use cross-validation
#         n_samples_for_chisq='total_ensemble'  # Use total ensemble size for χ² test
#     )
    
#     # Example data structure (adapt to your actual data)
#     forecasts = {
#         'model1': xr.DataArray(np.random.randn(10, 5, 20, 40), 
#                               dims=['time', 'ensemble', 'lat', 'lon']),
#         'model2': xr.DataArray(np.random.randn(10, 8, 20, 40),
#                               dims=['time', 'ensemble', 'lat', 'lon'])
#     }
    
#     hindcasts = {
#         'model1': xr.DataArray(np.random.randn(20, 5, 20, 40),
#                               dims=['time', 'ensemble', 'lat', 'lon']),
#         'model2': xr.DataArray(np.random.randn(20, 8, 20, 40),
#                               dims=['time', 'ensemble', 'lat', 'lon'])
#     }
    
#     climatology = xr.DataArray(np.random.randn(20, 40),
#                               dims=['lat', 'lon'])
    
#     ensemble_sizes = {'model1': 5, 'model2': 8}
#     model_names = ['model1', 'model2']
    
#     # Compute PMME probabilities
#     pmme_probs = pmme.compute_pmme_probabilities(
#         forecasts, hindcasts, climatology, ensemble_sizes
#     )
    
#     # Compute combined map with significance
#     combined_map, chi_square = pmme.compute_combined_map(
#         pmme_probs, ensemble_sizes, model_names, significance_level=0.05
#     )
    
#     return pmme_probs, combined_map, chi_square

# class WAS_Min2009_ProbWeighted_:
#     """
#     Implementation of Min et al. (2009) Probability-Weighted Multi-Model Ensemble.
    
#     Based on: "Probabilistic Multimodel Ensemble (PMME) forecasting at APCC"
#     Journal: Weather and Forecasting, 2009
    
#     Key methodology:
#     1. Individual model probabilistic forecasts are computed using Gaussian approximation
#     2. Model weights are proportional to sqrt(ensemble_size) [Eq. 6 in paper]
#     3. Tercile probabilities (BN, NN, AN) are combined using total probability formula
#     """
    
#     def __init__(self):
#         pass
    
#     def _compute_tercile_probabilities(self, forecasts, hindcasts, climatology):
#         """
#         Compute tercile probabilities for individual models using Gaussian approximation.
        
#         Parameters
#         ----------
#         forecasts : xarray.DataArray
#             Forecast ensemble data with dimensions (T, M, Y, X)
#         hindcasts : xarray.DataArray
#             Hindcast ensemble data with dimensions (T, M, Y, X)
#         climatology : xarray.DataArray
#             Climatological data with dimensions (Y, X)
            
#         Returns
#         -------
#         probs_bn : xarray.DataArray
#             Probability of below-normal category
#         probs_nn : xarray.DataArray
#             Probability of near-normal category
#         probs_an : xarray.DataArray
#             Probability of above-normal category
#         """
#         # Compute mean and std from hindcasts (cross-validated)
#         mu = hindcasts.mean(dim='ensemble')  # Model ensemble mean
#         sigma = hindcasts.std(dim='ensemble')  # Model ensemble spread
        
#         # Compute tercile boundaries (Eq. in section 5)
#         # Lower tercile: μ - 1.43σ, Upper tercile: μ + 1.43σ
#         lower_tercile = mu - 1.43 * sigma
#         upper_tercile = mu + 1.43 * sigma
        
#         # Compute forecast ensemble mean
#         forecast_mean = forecasts.mean(dim='ensemble')
        
#         # For Gaussian approximation, compute probabilities using CDF
#         # Probability of below-normal: Φ((lower_tercile - forecast_mean)/sigma)
#         # Probability of above-normal: 1 - Φ((upper_tercile - forecast_mean)/sigma)
#         # Probability of near-normal: 1 - P(BN) - P(AN)
        
#         # Standardized anomalies
#         z_lower = (lower_tercile - forecast_mean) / sigma
#         z_upper = (upper_tercile - forecast_mean) / sigma
        
#         # Gaussian CDF probabilities
#         probs_bn = stats.norm.cdf(z_lower)
#         probs_an = 1 - stats.norm.cdf(z_upper)
#         probs_nn = 1 - probs_bn - probs_an
        
#         # Ensure probabilities are between 0 and 1
#         probs_bn = xr.where(probs_bn < 0, 0, xr.where(probs_bn > 1, 1, probs_bn))
#         probs_an = xr.where(probs_an < 0, 0, xr.where(probs_an > 1, 1, probs_an))
#         probs_nn = xr.where(probs_nn < 0, 0, xr.where(probs_nn > 1, 1, probs_nn))
        
#         return probs_bn, probs_nn, probs_an
    
#     def _compute_model_weights(self, ensemble_sizes):
#         """
#         Compute model weights according to Min et al. (2009) Eq. 6.
        
#         Parameters
#         ----------
#         ensemble_sizes : dict
#             Dictionary mapping model names to ensemble sizes
            
#         Returns
#         -------
#         weights : dict
#             Dictionary mapping model names to normalized weights
#         """
#         # Weight proportional to sqrt(ensemble_size)
#         sqrt_sizes = {model: np.sqrt(size) for model, size in ensemble_sizes.items()}
#         total = sum(sqrt_sizes.values())
        
#         # Normalize so weights sum to 1
#         weights = {model: sqrt_sizes[model] / total for model in ensemble_sizes.keys()}
        
#         return weights
    
#     def compute_pmme_probabilities(self, forecasts, hindcasts, climatology, ensemble_sizes):
#         """
#         Compute PMME probabilities according to Min et al. (2009).
        
#         Parameters
#         ----------
#         forecasts : dict of xarray.DataArray
#             Dictionary of forecast ensembles for each model
#         hindcasts : dict of xarray.DataArray
#             Dictionary of hindcast ensembles for each model
#         climatology : xarray.DataArray
#             Climatological data
#         ensemble_sizes : dict
#             Dictionary mapping model names to ensemble sizes
            
#         Returns
#         -------
#         pmme_probs : dict
#             Dictionary with keys 'BN', 'NN', 'AN' containing PMME probabilities
#         """
#         # Get model names
#         model_names = list(forecasts.keys())
        
#         # Compute individual model probabilities
#         model_probs = {}
#         for model in model_names:
#             probs_bn, probs_nn, probs_an = self._compute_tercile_probabilities(
#                 forecasts[model], hindcasts[model], climatology
#             )
#             model_probs[model] = {
#                 'BN': probs_bn,
#                 'NN': probs_nn,
#                 'AN': probs_an
#             }
        
#         # Compute model weights (Eq. 6)
#         weights = self._compute_model_weights(ensemble_sizes)
        
#         # Combine probabilities using total probability formula (Eq. 1)
#         pmme_probs = {}
#         for category in ['BN', 'NN', 'AN']:
#             weighted_sum = None
#             for model in model_names:
#                 weighted_prob = model_probs[model][category] * weights[model]
#                 if weighted_sum is None:
#                     weighted_sum = weighted_prob
#                 else:
#                     weighted_sum = weighted_sum + weighted_prob
            
#             pmme_probs[category] = weighted_sum
        
#         return pmme_probs
    
#     def compute_combined_map(self, pmme_probs, significance_level=0.05):
#         """
#         Compute combined map with significance testing (χ² test).
        
#         Parameters
#         ----------
#         pmme_probs : dict
#             PMME probabilities for BN, NN, AN categories
#         significance_level : float
#             Significance level for χ² test (default 0.05)
            
#         Returns
#         -------
#         combined_map : xarray.DataArray
#             Combined map showing dominant category where significant
#         """
#         # Find dominant category
#         probs_array = xr.concat([pmme_probs['BN'], pmme_probs['NN'], pmme_probs['AN']], 
#                                dim='category')
#         dominant_category = probs_array.argmax(dim='category')
        
#         # Compute χ² statistic (Eq. in section 5)
#         # χ² = n * Σ (P(Ej) - 1/3)² / (1/3)
#         n = 1  # This should be total ensemble size across all models
#         expected_prob = 1/3
        
#         chi_square = n * (
#             (pmme_probs['BN'] - expected_prob)**2 / expected_prob +
#             (pmme_probs['NN'] - expected_prob)**2 / expected_prob +
#             (pmme_probs['AN'] - expected_prob)**2 / expected_prob
#         )
        
#         # Critical value for χ² with 2 degrees of freedom
#         critical_value = stats.chi2.ppf(1 - significance_level, df=2)
        
#         # Create combined map: show dominant category where significant
#         combined_map = xr.where(
#             chi_square > critical_value,
#             dominant_category + 1,  # 1 for BN, 2 for NN, 3 for AN
#             0  # No significant deviation from climatology
#         )
        
#         return combined_map
    
# # Example 
# forecasts = {
#     'model1': xr.DataArray(...),  # dimensions: (time, ensemble, lat, lon)
#     'model2': xr.DataArray(...),
#     # ... more models
# }

# hindcasts = {
#     'model1': xr.DataArray(...),  # cross-validated hindcasts
#     'model2': xr.DataArray(...),
#     # ... more models
# }

# ensemble_sizes = {
#     'model1': 20,
#     'model2': 10,
#     # ... ensemble sizes for each model
# }

# # Initialize and compute
# pmme = WAS_Min2009_ProbWeighted()

# # Compute PMME probabilities
# pmme_probs = pmme.compute_pmme_probabilities(
#     forecasts, hindcasts, climatology, ensemble_sizes
# )

# # Get combined map
# combined_map = pmme.compute_combined_map(pmme_probs)



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
        
        model = xc.MELR(preprocessing='minmax') # **self.elm_kwargs
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
        
        model = xc.MELR(preprocessing='minmax') 
        model.fit(hindcast_det, Predictant)
        result_ = model.predict_proba(Predictor_for_year)
        result_ = result_.rename({'S':'T','M':'probability'}).transpose('probability','T', 'Y', 'X')
        hindcast_prob = result_.assign_coords(probability=('probability', ['PB', 'PN', 'PA']))
        
        return hindcast_prob.transpose('probability', 'T', 'Y', 'X').load()


class WAS_mme_logistic:
    """
    Multinomial Logistic Regression MME with multiple hyperparameter optimization methods.
    
    This class implements a unified approach for logistic regression with spatial clustering
    and three hyperparameter optimization strategies:
    1. Grid Search
    2. Random Search
    3. Bayesian Optimization (using Optuna)
    
    Parameters
    ----------
    optimization_method : str, optional
        Method for hyperparameter optimization. Options: 
        'grid', 'random', 'bayesian', 'none' (default: 'grid')
    C_range : list or tuple, optional
        For grid/random search: list of C values
        For Bayesian: tuple of (min_C, max_C) for log-uniform sampling
    solver_options : list, optional
        List of solver algorithms to consider
    random_state : int, optional
        Random seed for reproducibility (default: 42)
    cv_folds : int, optional
        Number of cross-validation folds (default: 5)
    n_clusters : int, optional
        Number of spatial clusters (default: 4)
    n_iter_search : int, optional
        Number of iterations for random/bayesian search (default: 20)
    n_trials : int, optional
        Number of trials for Bayesian optimization (default: 50)
    timeout : int, optional
        Timeout in seconds for Bayesian optimization (default: None)
    """
    
    def __init__(self,
                 optimization_method='grid',
                 C_range=[0.1, 1.0, 10.0, 100.0],
                 solver_options=['newton-cg', 'lbfgs', 'sag', 'saga'],
                 random_state=42,
                 cv_folds=5,
                 n_clusters=4,
                 n_iter_search=20,
                 n_trials=50,
                 timeout=None):
        
        self.optimization_method = optimization_method
        self.C_range = C_range
        self.solver_options = solver_options
        self.random_state = random_state
        self.cv_folds = cv_folds
        self.n_clusters = n_clusters
        self.n_iter_search = n_iter_search
        self.n_trials = n_trials
        self.timeout = timeout
        self.models = None
        self.best_params_dict = None
        self.cluster_da = None
        self.study_dict = None  # For storing Optuna studies
        
    def _create_parameter_grids(self):
        """
        Create parameter grids for different optimization methods.
        """
        # Grid 1: L2 Penalty
        solvers_l2 = [s for s in self.solver_options 
                     if s in ['newton-cg', 'lbfgs', 'sag', 'saga']]
        param_grid_l2 = {
            'penalty': ['l2'],
            'solver': solvers_l2,
            'C': self.C_range
        }
        
        # Grid 2: L1 Penalty (ONLY for 'saga')
        solvers_l1 = [s for s in self.solver_options if s == 'saga']
        param_grid_l1 = {}
        if solvers_l1:
            param_grid_l1 = {
                'penalty': ['l1'],
                'solver': solvers_l1,
                'C': self.C_range
            }
        
        param_grids = [param_grid_l2]
        if param_grid_l1:
            param_grids.append(param_grid_l1)
            
        return param_grids
    
    def _create_parameter_distributions(self):
        """
        Create parameter distributions for random search.
        """
        param_dist = {
            'C': self.C_range,
            'solver': self.solver_options,
            'penalty': ['l2', 'l1']  # Include both penalties
        }
        return param_dist
    
    def _grid_search_optimization(self, X, y):
        """
        Perform grid search optimization.
        """
        param_grids = self._create_parameter_grids()
        
        model = LogisticRegression(
            random_state=self.random_state, 
            max_iter=1000
        )
        
        cv_splitter = KFold(
            n_splits=self.cv_folds, 
            shuffle=True, 
            random_state=self.random_state
        )
        
        grid_search = GridSearchCV(
            model, 
            param_grid=param_grids, 
            cv=cv_splitter, 
            scoring='neg_log_loss',
            error_score=np.nan,
            n_jobs=-1
        )
        
        grid_search.fit(X, y)
        return grid_search.best_params_
    
    def _random_search_optimization(self, X, y):
        """
        Perform random search optimization.
        """
        param_dist = self._create_parameter_distributions()
        
        model = LogisticRegression(
            random_state=self.random_state, 
            max_iter=1000
        )
        
        cv_splitter = KFold(
            n_splits=self.cv_folds, 
            shuffle=True, 
            random_state=self.random_state
        )
        
        random_search = RandomizedSearchCV(
            model, 
            param_distributions=param_dist,
            n_iter=self.n_iter_search,
            cv=cv_splitter, 
            scoring='neg_log_loss',
            random_state=self.random_state,
            error_score=np.nan,
            n_jobs=-1
        )
        
        random_search.fit(X, y)
        return random_search.best_params_
    
    def _bayesian_objective(self, trial, X, y):
        """
        Objective function for Bayesian optimization with Optuna.
        """
        # Define hyperparameter search space
        penalty = trial.suggest_categorical('penalty', ['l2', 'l1'])
        
        # Choose solver based on penalty
        if penalty == 'l1':
            solver = 'saga'
        else:
            solver = trial.suggest_categorical('solver', 
                [s for s in self.solver_options if s != 'saga'])
        
        # Define C with log-uniform distribution
        if isinstance(self.C_range, (list, tuple)) and len(self.C_range) == 2:
            C_min, C_max = min(self.C_range), max(self.C_range)
            C = trial.suggest_float('C', C_min, C_max, log=True)
        else:
            # Use discrete values if provided
            C = trial.suggest_categorical('C', self.C_range)
        
        # Create and evaluate model
        model = LogisticRegression(
            penalty=penalty,
            solver=solver,
            C=C,
            random_state=self.random_state,
            max_iter=1000
        )
        
        # Cross-validation
        cv_scores = []
        cv = KFold(n_splits=self.cv_folds, shuffle=True, 
                   random_state=self.random_state)
        
        for train_idx, val_idx in cv.split(X, y):
            X_train, X_val = X[train_idx], X[val_idx]
            y_train, y_val = y[train_idx], y[val_idx]
            
            model.fit(X_train, y_train)
            
            # Use negative log loss (higher is better for Optuna)
            try:
                score = model.score(X_val, y_val)
                cv_scores.append(score)
            except:
                cv_scores.append(np.nan)
        
        # Return mean score (handle NaN)
        cv_scores = np.array(cv_scores)
        valid_scores = cv_scores[~np.isnan(cv_scores)]
        
        if len(valid_scores) > 0:
            return np.mean(valid_scores)
        else:
            return np.nan
    
    def _bayesian_optimization(self, X, y, cluster_id=None):
        """
        Perform Bayesian optimization using Optuna.
        """
        # Create study
        study_name = f'logistic_cluster_{cluster_id}' if cluster_id else 'logistic'
        study = optuna.create_study(
            study_name=study_name,
            direction='maximize',  # We want to maximize accuracy/score
            sampler=TPESampler(seed=self.random_state)
        )
        
        # Define objective function with data
        objective_with_data = lambda trial: self._bayesian_objective(trial, X, y)
        
        # Optimize
        study.optimize(
            objective_with_data,
            n_trials=self.n_trials,
            timeout=self.timeout,
            show_progress_bar=False
        )
        
        # Store study for later analysis
        if cluster_id is not None:
            if self.study_dict is None:
                self.study_dict = {}
            self.study_dict[cluster_id] = study
        
        # Extract best parameters
        best_params = study.best_params
        
        # Ensure penalty-solver compatibility
        if best_params.get('penalty') == 'l1':
            best_params['solver'] = 'saga'
        
        return best_params
    
    def compute_hyperparameters(self, Predictors, Predictand, clim_year_start, clim_year_end):
        """
        Computes best hyperparameters using the selected optimization method.
        
        Parameters
        ----------
        Predictors : xarray.DataArray
            Predictor data with dimensions (T, M, Y, X)
        Predictand : xarray.DataArray
            Predictand data with dimensions (T, Y, X)
        clim_year_start : int
            Start year of climatology period
        clim_year_end : int
            End year of climatology period
            
        Returns
        -------
        best_params_dict : dict
            Best hyperparameters for each cluster
        cluster_da : xarray.DataArray
            Cluster labels for each spatial point
        """

        if "M" in Predictand.coords:
            Predictand = Predictand.isel(M=0).drop_vars('M').squeeze()
            
        X_train_std = standardize_timeseries(Predictors, clim_year_start, clim_year_end)
        Predictand.name = "varname"

        # Step 1: Perform KMeans clustering based on predictand's spatial distribution
        from sklearn.cluster import KMeans
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

        # Assuming WAS_Verification is available
        verify = WAS_Verification()
        y_train_std = verify.compute_class(Predictand, clim_year_start, clim_year_end)
               
        # Align cluster array with the predictand array
        xarray1, xarray2 = xr.align(y_train_std, Cluster, join="outer")
        
        # Identify unique cluster labels
        clusters = np.unique(xarray2)
        clusters = clusters[~np.isnan(clusters)]
        cluster_da = xarray2

        X_train_std['T'] = y_train_std['T']

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

        
###############################""""""
        # # Remove M dimension if present
        # if "M" in Predictand.coords:
        #     Predictand = Predictand.isel(M=0).drop_vars('M').squeeze()
        
        # # Standardize predictors
        # X_train_std = standardize_timeseries(Predictors, clim_year_start, clim_year_end)
        
        # # Cluster on predictand time series
        # y_for_cluster = Predictand.stack(space=('Y', 'X')).transpose('space', 'T').values
        # finite_mask = np.all(np.isfinite(y_for_cluster), axis=1)
        # y_cluster = y_for_cluster[finite_mask]
        
        # kmeans = KMeans(n_clusters=self.n_clusters, random_state=self.random_state)
        # labels = kmeans.fit_predict(y_cluster)
        
        # full_labels = np.full(y_for_cluster.shape[0], np.nan)
        # full_labels[finite_mask] = labels
        # cluster_da = xr.DataArray(
        #     full_labels.reshape(len(Predictand['Y']), len(Predictand['X'])),
        #     coords={'Y': Predictand['Y'], 'X': Predictand['X']},
        #     dims=['Y', 'X']
        # )
        
        # clusters = np.unique(labels[~np.isnan(labels)])
        
        # # Assuming WAS_Verification is available
        # verify = WAS_Verification()
        # Predictand = verify.compute_class(Predictand, clim_year_start, clim_year_end)
        
        # X_train_std['T'] = Predictand['T']
        
        # best_params_dict = {}
        
        # for c in clusters:
        #     mask_3d = (cluster_da == c).expand_dims({'T': Predictand['T']})
            
        #     # Stack data for current cluster
        #     X_stacked_c = X_train_std.where(mask_3d).stack(sample=('T', 'Y', 'X')).transpose('sample', 'M').values
        #     y_stacked_c = Predictand.where(mask_3d).stack(sample=('T', 'Y', 'X')).values.ravel()
            
        #     # Remove NaN values
        #     nan_mask_c = np.any(~np.isfinite(X_stacked_c), axis=1) | ~np.isfinite(y_stacked_c)
        #     X_clean_c = X_stacked_c[~nan_mask_c]
        #     y_clean_c = y_stacked_c[~nan_mask_c]
            
        #     if len(X_clean_c) == 0 or len(np.unique(y_clean_c)) < 2:
        #         continue
            
            # Apply selected optimization method
            if self.optimization_method == 'grid':
                best_params = self._grid_search_optimization(X_clean_c, y_clean_c)
            elif self.optimization_method == 'random':
                best_params = self._random_search_optimization(X_clean_c, y_clean_c)
            elif self.optimization_method == 'bayesian':
                best_params = self._bayesian_optimization(X_clean_c, y_clean_c, cluster_id=c)
            elif self.optimization_method == 'none':
                # Use default parameters
                best_params = {
                    'C': 1.0,
                    'solver': 'lbfgs',
                    'penalty': 'l2'
                }
            else:
                raise ValueError(f"Unknown optimization method: {self.optimization_method}")
            
            best_params_dict[c] = best_params
        
        self.best_params_dict = best_params_dict
        self.cluster_da = cluster_da
        
        return best_params_dict, cluster_da
    
    def compute_model(self, X_train, y_train, X_test, y_test, clim_year_start, clim_year_end, 
                      best_params=None, cluster_da=None):
        """
        Train models and compute predictions.
        
        Parameters
        ----------
        X_train : xarray.DataArray
            Training predictor data
        y_train : xarray.DataArray
            Training predictand data (classes)
        X_test : xarray.DataArray
            Testing predictor data
        y_test : xarray.DataArray
            Testing predictand data
        clim_year_start : int
            Start year of climatology period
        clim_year_end : int
            End year of climatology period
        best_params : dict, optional
            Pre-computed best parameters
        cluster_da : xarray.DataArray, optional
            Pre-computed cluster labels
            
        Returns
        -------
        predicted_da : xarray.DataArray
            Predicted classes
        predicted_prob_da : xarray.DataArray
            Predicted probabilities
        """
        # Standardize data
        X_train_std = X_train #standardize_timeseries(X_train, clim_year_start, clim_year_end)
        
        # Standardize test data using training statistics
        # clim_slice = slice(str(clim_year_start), str(clim_year_end))
        # mean_val = X_train.sel(T=clim_slice).mean(dim='T')
        # std_val = X_train.sel(T=clim_slice).std(dim='T')
        X_test_std = X_test # (X_test - mean_val) / std_val
        
        # Get or compute hyperparameters
        if best_params is None or cluster_da is None:
            best_params, cluster_da = self.compute_hyperparameters(
                X_train, y_train, clim_year_start, clim_year_end
            )
        
        # Extract coordinates
        time = X_test_std['T']
        lat = X_test_std['Y']
        lon = X_test_std['X']
        n_time = len(time)
        n_lat = len(lat)
        n_lon = len(lon)
        
        # Initialize prediction arrays
        predictions = np.full((n_time, n_lat, n_lon), np.nan)
        predictions_prob = np.full((n_time, n_lat, n_lon, 3), np.nan)
        self.models = {}
        
        # Train model for each cluster
        for c, bp in best_params.items():
            mask_3d_train = (cluster_da == c).expand_dims({'T': X_train_std['T']})
            mask_3d_test = (cluster_da == c).expand_dims({'T': X_test_std['T']})
            
            # Stack training data
            X_train_stacked_c = X_train_std.where(mask_3d_train).stack(
                sample=('T', 'Y', 'X')).transpose('sample', 'M').values
            y_train_stacked_c = y_train.where(mask_3d_train).stack(
                sample=('T', 'Y', 'X')).values.ravel()
            
            train_nan_mask_c = np.any(~np.isfinite(X_train_stacked_c), axis=1) | ~np.isfinite(y_train_stacked_c)
            X_train_clean_c = X_train_stacked_c[~train_nan_mask_c]
            y_train_clean_c = y_train_stacked_c[~train_nan_mask_c]
            
            # Stack testing data
            X_test_stacked_c = X_test_std.where(mask_3d_test).stack(
                sample=('T', 'Y', 'X')).transpose('sample', 'M').values
            test_nan_mask_c = np.any(~np.isfinite(X_test_stacked_c), axis=1)
            X_test_clean_c = X_test_stacked_c[~test_nan_mask_c]
            
            # Create and train model
            model_c = LogisticRegression(
                **bp,
                random_state=self.random_state,
                max_iter=1000
            )
            
            model_c.fit(X_train_clean_c, y_train_clean_c)
            self.models[c] = model_c
            
            # Predict
            y_pred_c = model_c.predict(X_test_clean_c)
            y_prob_c = model_c.predict_proba(X_test_clean_c)
            
            # Reconstruct spatial fields
            full_stacked_c = np.full(X_test_stacked_c.shape[0], np.nan)
            full_stacked_c[~test_nan_mask_c] = y_pred_c
            pred_c_reshaped = full_stacked_c.reshape(n_time, n_lat, n_lon)
            predictions = np.where(np.isnan(predictions), pred_c_reshaped, predictions)
            
            full_stacked_prob_c = np.full((X_test_stacked_c.shape[0], 3), np.nan)
            full_stacked_prob_c[~test_nan_mask_c] = y_prob_c
            pred_prob_c_reshaped = full_stacked_prob_c.reshape(n_time, n_lat, n_lon, 3)
            predictions_prob = np.where(np.isnan(predictions_prob), 
                                        pred_prob_c_reshaped, predictions_prob)
        
        # Create output DataArrays
        predicted_da = xr.DataArray(
            data=predictions,
            coords={'T': time, 'Y': lat, 'X': lon},
            dims=['T', 'Y', 'X']
        )
        
        predicted_prob_da = xr.DataArray(
            data=predictions_prob,
            coords={'T': time, 'Y': lat, 'X': lon, 'probability': [0, 1, 2]},
            dims=['T', 'Y', 'X', 'probability']
        )
        
        predicted_prob_da = predicted_prob_da.transpose('probability', 'T', 'Y', 'X')
        predicted_prob_da = predicted_prob_da.assign_coords(
            probability=['PB', 'PN', 'PA']
        )
        
        return predicted_da, predicted_prob_da
    
    def forecast(self, Predictant, clim_year_start, clim_year_end, 
                 Predictors_train, Predictor_for_year, 
                 best_params=None, cluster_da=None):
        """
        Generate forecasts for a target year.
        
        Parameters
        ----------
        Predictant : xarray.DataArray
            Observed predictand data
        clim_year_start : int
            Start year of climatology period
        clim_year_end : int
            End year of climatology period
        Predictors_train : xarray.DataArray
            Training predictor data
        Predictor_for_year : xarray.DataArray
            Predictor data for target year
        best_params : dict, optional
            Pre-computed best parameters
        cluster_da : xarray.DataArray, optional
            Pre-computed cluster labels
            
        Returns
        -------
        forecast_det : xarray.DataArray
            Deterministic forecast (classes)
        forecast_prob : xarray.DataArray
            Probabilistic forecast
        """
        # Remove M dimension if present
        if "M" in Predictant.coords:
            Predictant_no_m = Predictant.isel(M=0).drop_vars('M').squeeze()
        else:
            Predictant_no_m = Predictant
            
        # Compute classes
        verify = WAS_Verification()
        Predictant_no_m = verify.compute_class(Predictant_no_m, 
                                               clim_year_start, clim_year_end)
        
        # Create mask
        mask = xr.where(~np.isnan(Predictant_no_m.isel(T=0)), 1, np.nan)
        
        # Standardize data
        Predictors_train_st = standardize_timeseries(Predictors_train, 
                                                     clim_year_start, clim_year_end)
        
        clim_slice = slice(str(clim_year_start), str(clim_year_end))
        mean_val = Predictors_train.sel(T=clim_slice).mean(dim='T')
        std_val = Predictors_train.sel(T=clim_slice).std(dim='T')
        Predictor_for_year_st = (Predictor_for_year - mean_val) / std_val
        
        # Align time coordinates
        Predictors_train_st['T'] = Predictant_no_m['T']
        
        # Get or compute hyperparameters
        if best_params is None or cluster_da is None:
            best_params, cluster_da = self.compute_hyperparameters(
                Predictors_train, Predictant_no_m, 
                clim_year_start, clim_year_end
            )
        
        # Extract coordinates
        time = Predictor_for_year_st['T']
        lat = Predictor_for_year_st['Y']
        lon = Predictor_for_year_st['X']
        n_time = len(time)
        n_lat = len(lat)
        n_lon = len(lon)
        
        # Initialize prediction arrays
        predictions = np.full((n_time, n_lat, n_lon), np.nan)
        predictions_prob = np.full((n_time, n_lat, n_lon, 3), np.nan)
        self.models = {}
        
        # Train and predict for each cluster
        for c, bp in best_params.items():
            mask_3d_train = (cluster_da == c).expand_dims({'T': Predictors_train_st['T']})
            mask_3d_test = (cluster_da == c).expand_dims({'T': Predictor_for_year_st['T']})
            
            # Stack training data
            X_train_stacked_c = Predictors_train_st.where(mask_3d_train).stack(
                sample=('T', 'Y', 'X')).transpose('sample', 'M').values
            y_train_stacked_c = Predictant_no_m.where(mask_3d_train).stack(
                sample=('T', 'Y', 'X')).values.ravel()
            
            train_nan_mask_c = np.any(~np.isfinite(X_train_stacked_c), axis=1) | ~np.isfinite(y_train_stacked_c)
            X_train_clean_c = X_train_stacked_c[~train_nan_mask_c]
            y_train_clean_c = y_train_stacked_c[~train_nan_mask_c]
            
            # Stack forecast data
            X_test_stacked_c = Predictor_for_year_st.where(mask_3d_test).stack(
                sample=('T', 'Y', 'X')).transpose('sample', 'M').values
            test_nan_mask_c = np.any(~np.isfinite(X_test_stacked_c), axis=1)
            X_test_clean_c = X_test_stacked_c[~test_nan_mask_c]
            
            # Create and train model
            model_c = LogisticRegression(
                **bp,
                random_state=self.random_state,
                max_iter=1000
            )
            
            model_c.fit(X_train_clean_c, y_train_clean_c)
            self.models[c] = model_c
            
            # Predict
            y_pred_c = model_c.predict(X_test_clean_c)
            y_prob_c = model_c.predict_proba(X_test_clean_c)
            
            # Reconstruct spatial fields
            full_stacked_c = np.full(X_test_stacked_c.shape[0], np.nan)
            full_stacked_c[~test_nan_mask_c] = y_pred_c
            pred_c_reshaped = full_stacked_c.reshape(n_time, n_lat, n_lon)
            predictions = np.where(np.isnan(predictions), pred_c_reshaped, predictions)
            
            full_stacked_prob_c = np.full((X_test_stacked_c.shape[0], 3), np.nan)
            full_stacked_prob_c[~test_nan_mask_c] = y_prob_c
            pred_prob_c_reshaped = full_stacked_prob_c.reshape(n_time, n_lat, n_lon, 3)
            predictions_prob = np.where(np.isnan(predictions_prob), 
                                        pred_prob_c_reshaped, predictions_prob)
        
        # Create output DataArrays
        forecast_det = xr.DataArray(
            data=predictions,
            coords={'T': time, 'Y': lat, 'X': lon},
            dims=['T', 'Y', 'X']
        ) * mask
        
        forecast_prob = xr.DataArray(
            data=predictions_prob,
            coords={'T': time, 'Y': lat, 'X': lon, 'probability': [0, 1, 2]},
            dims=['T', 'Y', 'X', 'probability']
        ) * mask
        
        forecast_prob = forecast_prob.transpose('probability', 'T', 'Y', 'X')
        forecast_prob = forecast_prob.assign_coords(
            probability=['PB', 'PN', 'PA']
        )
        
        # Update time coordinate for forecast year
        year = Predictor_for_year.coords['T'].values.astype('datetime64[Y]').astype(int)[0] + 1970
        T_value_1 = Predictant_no_m.isel(T=0).coords['T'].values
        month_1 = T_value_1.astype('datetime64[M]').astype(int) % 12 + 1
        new_T_value = np.datetime64(f"{year}-{month_1:02d}-01")
        
        forecast_det = forecast_det.assign_coords(T=xr.DataArray([new_T_value], dims=["T"]))
        forecast_det['T'] = forecast_det['T'].astype('datetime64[ns]')
        forecast_prob = forecast_prob.assign_coords(T=xr.DataArray([new_T_value], dims=["T"]))
        forecast_prob['T'] = forecast_prob['T'].astype('datetime64[ns]')
        
        return forecast_det, forecast_prob
    
    def get_optimization_results(self):
        """
        Get detailed results from hyperparameter optimization.
        
        Returns
        -------
        results : dict
            Dictionary containing optimization results
        """
        results = {
            'optimization_method': self.optimization_method,
            'best_parameters': self.best_params_dict,
            'cluster_labels': self.cluster_da,
            'models': self.models
        }
        
        if self.optimization_method == 'bayesian' and self.study_dict:
            results['optuna_studies'] = self.study_dict
            
        return results    
# #Exmaple MINE
# # Grid Search (default)
# model_grid = WAS_mme_logistic(optimization_method='grid')
# best_params, clusters = model_grid.compute_hyperparameters(predictors, predictand, 1981, 2010)

# # Random Search
# model_random = WAS_mme_logistic(
#     optimization_method='random',
#     n_iter_search=30
# )

# # Bayesian Optimization
# model_bayesian = WAS_mme_logistic(
#     optimization_method='bayesian',
#     C_range=(0.01, 1000.0),  # Continuous range for Bayesian
#     n_trials=100,
#     timeout=300  # 5 minutes timeout
# )

# # No optimization (use defaults)
# model_default = WAS_mme_logistic(optimization_method='none')
    




class WAS_mme_gaussian_process:
    """
    Gaussian Process Classifier MME (multiclass via One-vs-Rest in sklearn).

    This class mirrors the structure of WAS_mme_logistic:
      - cluster the predictand time series (per gridpoint) -> cluster_da
      - for each cluster: stack samples (T,Y,X), fit a classifier, predict class + proba
      - reconstruct to (T,Y,X) and return deterministic forecast + tercile probabilities

    IMPORTANT: GP classification scales cubically with N samples. To avoid infeasible runs,
    we subsample per cluster to max_train_samples during training and HPO.

    Supports three hyperparameter optimization methods:
      - 'grid': GridSearchCV (default)
      - 'random': RandomizedSearchCV
      - 'bayesian': Optuna Bayesian optimization
    """

    def __init__(
        self,
        random_state=42,
        cv_folds=5,
        n_clusters=4,
        # GP-related controls
        kernel_options=None,
        n_restarts_optimizer_options=(0, 2),
        max_iter_predict_options=(100, 200),
        max_train_samples=5000,
        # HPO method selection
        hpo_method='grid',  # 'grid', 'random', or 'bayesian'
        # RandomizedSearchCV parameters
        n_random_iter=10,
        # Optuna parameters
        n_trials=20,
        timeout=None,  # seconds
        n_jobs=1,  # for Optuna
        # if you want a little numerical stability:
        warm_start=False,  # kept for compatibility; sklearn GPC ignores this concept
    ):
        self.random_state = random_state
        self.cv_folds = cv_folds
        self.n_clusters = n_clusters

        # Default kernels (reasonable starters for standardized predictors)
        if kernel_options is None:
            kernel_options = [
                C(1.0, (1e-2, 1e2)) * RBF(length_scale=1.0, length_scale_bounds=(1e-2, 1e2)),
                C(1.0, (1e-2, 1e2)) * Matern(length_scale=1.0, nu=1.5, length_scale_bounds=(1e-2, 1e2)),
                C(1.0, (1e-2, 1e2)) * Matern(length_scale=1.0, nu=2.5, length_scale_bounds=(1e-2, 1e2)),
            ]
        self.kernel_options = kernel_options
        self.n_restarts_optimizer_options = tuple(n_restarts_optimizer_options)
        self.max_iter_predict_options = tuple(max_iter_predict_options)
        self.max_train_samples = int(max_train_samples)
        
        # HPO method
        self.hpo_method = hpo_method.lower()
        if self.hpo_method not in ['grid', 'random', 'bayesian']:
            raise ValueError(f"hpo_method must be 'grid', 'random', or 'bayesian', got {hpo_method}")
        
        # RandomizedSearchCV parameters
        self.n_random_iter = n_random_iter
        
        # Optuna parameters
        self.n_trials = n_trials
        self.timeout = timeout
        self.n_jobs = n_jobs
        
        if self.hpo_method == 'bayesian' and not OPTUNA_AVAILABLE:
            warnings.warn("Optuna not available. Falling back to RandomizedSearchCV.")
            self.hpo_method = 'random'

        self.models = None

    def _subsample(self, X, y, nmax):
        """Reproducible subsampling to keep GP feasible."""
        if (nmax is None) or (nmax <= 0) or (X.shape[0] <= nmax):
            return X, y
        rng = np.random.RandomState(self.random_state)
        idx = rng.choice(X.shape[0], size=nmax, replace=False)
        return X[idx], y[idx]

    def _create_param_grid(self):
        """Create parameter grid for grid/random search."""
        param_grid = {
            "kernel": self.kernel_options,
            "n_restarts_optimizer": list(self.n_restarts_optimizer_options),
            "max_iter_predict": list(self.max_iter_predict_options),
        }
        return param_grid

    def _create_param_distributions(self):
        """Create parameter distributions for RandomizedSearchCV."""
        # For randomized search, we can sample from distributions
        param_dist = {
            "kernel": self.kernel_options,
            "n_restarts_optimizer": list(self.n_restarts_optimizer_options),
            "max_iter_predict": list(self.max_iter_predict_options),
        }
        return param_dist

    def _optuna_objective(self, trial, X_train, y_train, cv_splitter):
        """Objective function for Optuna Bayesian optimization."""
        # Suggest kernel type
        kernel_type = trial.suggest_categorical('kernel_type', ['RBF', 'Matern_1.5', 'Matern_2.5'])
        
        # Suggest kernel parameters
        length_scale = trial.suggest_float('length_scale', 0.1, 10.0, log=True)
        constant_value = trial.suggest_float('constant_value', 0.1, 10.0, log=True)
        
        # Build kernel
        if kernel_type == 'RBF':
            kernel = C(constant_value, (1e-2, 1e2)) * RBF(
                length_scale=length_scale, 
                length_scale_bounds=(1e-2, 1e2)
            )
        elif kernel_type == 'Matern_1.5':
            kernel = C(constant_value, (1e-2, 1e2)) * Matern(
                length_scale=length_scale, 
                nu=1.5, 
                length_scale_bounds=(1e-2, 1e2)
            )
        else:  # 'Matern_2.5'
            kernel = C(constant_value, (1e-2, 1e2)) * Matern(
                length_scale=length_scale, 
                nu=2.5, 
                length_scale_bounds=(1e-2, 1e2)
            )
        
        # Add white noise kernel for numerical stability
        kernel = kernel + WhiteKernel(noise_level=1e-5, noise_level_bounds=(1e-10, 1e-2))
        
        # Suggest other hyperparameters
        n_restarts_optimizer = trial.suggest_int('n_restarts_optimizer', 
                                                 min(self.n_restarts_optimizer_options),
                                                 max(self.n_restarts_optimizer_options))
        max_iter_predict = trial.suggest_int('max_iter_predict',
                                            min(self.max_iter_predict_options),
                                            max(self.max_iter_predict_options))
        
        # Create and evaluate model
        model = GaussianProcessClassifier(
            kernel=kernel,
            n_restarts_optimizer=n_restarts_optimizer,
            max_iter_predict=max_iter_predict,
            random_state=self.random_state
        )
        
        # Perform cross-validation
        scores = []
        for train_idx, val_idx in cv_splitter.split(X_train, y_train):
            X_train_fold, X_val_fold = X_train[train_idx], X_train[val_idx]
            y_train_fold, y_val_fold = y_train[train_idx], y_train[val_idx]
            
            model.fit(X_train_fold, y_train_fold)
            
            # Use negative log loss (to maximize)
            try:
                y_proba = model.predict_proba(X_val_fold)
                score = -log_loss(y_val_fold, y_proba, labels=[0, 1, 2])
                scores.append(score)
            except:
                scores.append(-1e10)  # Very bad score for failed trials
        
        return np.mean(scores)

    def compute_hyperparameters(self, Predictors, Predictand, clim_year_start, clim_year_end):
        """
        Computes best hyperparameters using selected HPO method.

        NOTE: As in your logistic version, this uses pre-standardized data.
        That preserves your current behavior but can introduce leakage if the
        standardization includes information beyond each CV fold.
        """

        if "M" in Predictand.coords:
            Predictand = Predictand.isel(M=0).drop_vars('M').squeeze()
            
        X_train_std = standardize_timeseries(Predictors, clim_year_start, clim_year_end)
        Predictand.name = "varname"

        # Step 1: Perform KMeans clustering based on predictand's spatial distribution
        from sklearn.cluster import KMeans
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

        # Assuming WAS_Verification is available
        verify = WAS_Verification()
        y_train_std = verify.compute_class(Predictand, clim_year_start, clim_year_end)
               
        # Align cluster array with the predictand array
        xarray1, xarray2 = xr.align(y_train_std, Cluster, join="outer")
        
        # Identify unique cluster labels
        clusters = np.unique(xarray2)
        clusters = clusters[~np.isnan(clusters)]
        cluster_da = xarray2

        X_train_std['T'] = y_train_std['T']

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


        
###########################################"""""
        # if "M" in Predictand.coords:
        #     Predictand = Predictand.isel(M=0).drop_vars("M").squeeze()

        # # Standardize predictors 
        # X_train_std = standardize_timeseries(Predictors, clim_year_start, clim_year_end)

        # # Cluster on predictand time series 
        # y_for_cluster = Predictand.stack(space=("Y", "X")).transpose("space", "T").values
        # finite_mask = np.all(np.isfinite(y_for_cluster), axis=1)
        # y_cluster = y_for_cluster[finite_mask]

        # kmeans = KMeans(n_clusters=self.n_clusters, random_state=self.random_state)
        # labels = kmeans.fit_predict(y_cluster)

        # full_labels = np.full(y_for_cluster.shape[0], np.nan)
        # full_labels[finite_mask] = labels

        # cluster_da = xr.DataArray(
        #     full_labels.reshape(len(Predictand["Y"]), len(Predictand["X"])),
        #     coords={"Y": Predictand["Y"], "X": Predictand["X"]},
        #     dims=["Y", "X"],
        # )
        # clusters = np.unique(labels)
        
        # # Import WAS_Verification from your module
        # # Assuming WAS_Verification is available in the namespace
        # verify = WAS_Verification()
        # y_train_std = verify.compute_class(Predictand, clim_year_start, clim_year_end)
        # X_train_std['T'] = y_train_std['T']
        # best_params_dict = {}
        # for c in clusters:
        #     mask_3d = (cluster_da == c).expand_dims({"T": y_train_std["T"]})

        #     X_stacked_c = (
        #         X_train_std.where(mask_3d)
        #         .stack(sample=("T", "Y", "X"))
        #         .transpose("sample", "M")
        #         .values
        #     )
        #     y_stacked_c = (
        #         y_train_std.where(mask_3d)
        #         .stack(sample=("T", "Y", "X"))
        #         .values
        #         .ravel()
        #     )
            
        #     nan_mask_c = np.any(~np.isfinite(X_stacked_c), axis=1) | ~np.isfinite(y_stacked_c)
        #     X_clean_c = X_stacked_c[~nan_mask_c]
        #     y_clean_c = y_stacked_c[~nan_mask_c].astype(int)

        #     if len(X_clean_c) == 0 or len(np.unique(y_clean_c)) < 2:
        #         best_params_dict[c] = None
        #         continue

###############
            
            # Subsample to keep GP feasible
            X_clean_c, y_clean_c = self._subsample(X_clean_c, y_clean_c, self.max_train_samples)
            
            # Create CV splitter
            cv_splitter = KFold(n_splits=self.cv_folds, shuffle=True, 
                               random_state=self.random_state)

            if self.hpo_method == 'grid':
                # Grid Search
                param_grid = self._create_param_grid()
                model = GaussianProcessClassifier(random_state=self.random_state)
                
                grid_search = GridSearchCV(
                    estimator=model,
                    param_grid=param_grid,
                    cv=cv_splitter,
                    scoring="neg_log_loss",
                    error_score=np.nan,
                    n_jobs=-1,
                )
                
                grid_search.fit(X_clean_c, y_clean_c)
                best_params_dict[c] = grid_search.best_params_
                
            elif self.hpo_method == 'random':
                # Randomized Search
                param_dist = self._create_param_distributions()
                model = GaussianProcessClassifier(random_state=self.random_state)
                
                random_search = RandomizedSearchCV(
                    estimator=model,
                    param_distributions=param_dist,
                    n_iter=self.n_random_iter,
                    cv=cv_splitter,
                    scoring="neg_log_loss",
                    error_score=np.nan,
                    random_state=self.random_state,
                    n_jobs=-1,
                )
                
                random_search.fit(X_clean_c, y_clean_c)
                best_params_dict[c] = random_search.best_params_
                
            elif self.hpo_method == 'bayesian':
                # Optuna Bayesian Optimization
                # Create study
                study = optuna.create_study(
                    direction='maximize',
                    sampler=optuna.samplers.TPESampler(seed=self.random_state),
                    pruner=optuna.pruners.MedianPruner()
                )
                
                # Define objective with closure
                def objective(trial):
                    return self._optuna_objective(trial, X_clean_c, y_clean_c, cv_splitter)
                
                # Optimize
                study.optimize(
                    objective,
                    n_trials=self.n_trials,
                    timeout=self.timeout,
                    n_jobs=self.n_jobs,
                    show_progress_bar=False
                )
                
                if study.best_trial is None:
                    best_params_dict[c] = None
                    continue
                    
                # Convert best trial to sklearn-compatible parameters
                best_trial = study.best_trial
                kernel_type = best_trial.params['kernel_type']
                length_scale = best_trial.params['length_scale']
                constant_value = best_trial.params['constant_value']
                
                if kernel_type == 'RBF':
                    kernel = C(constant_value, (1e-2, 1e2)) * RBF(
                        length_scale=length_scale, 
                        length_scale_bounds=(1e-2, 1e2)
                    )
                elif kernel_type == 'Matern_1.5':
                    kernel = C(constant_value, (1e-2, 1e2)) * Matern(
                        length_scale=length_scale, 
                        nu=1.5, 
                        length_scale_bounds=(1e-2, 1e2)
                    )
                else:  # 'Matern_2.5'
                    kernel = C(constant_value, (1e-2, 1e2)) * Matern(
                        length_scale=length_scale, 
                        nu=2.5, 
                        length_scale_bounds=(1e-2, 1e2)
                    )
                
                # Add white noise kernel
                kernel = kernel + WhiteKernel(noise_level=1e-5, noise_level_bounds=(1e-10, 1e-2))
                
                best_params_dict[c] = {
                    'kernel': kernel,
                    'n_restarts_optimizer': best_trial.params['n_restarts_optimizer'],
                    'max_iter_predict': best_trial.params['max_iter_predict']
                }

        return best_params_dict, cluster_da

    def compute_model(
        self,
        X_train,
        y_train,
        X_test,
        y_test,
        clim_year_start,
        clim_year_end,
        best_params=None,
        cluster_da=None,
    ):
        """
        Fit per-cluster GP models and predict on X_test.

        Assumes y_train already contains classes (0/1/2) consistent with your pipeline.
        """
        X_train_std = X_train
        X_test_std = X_test
        y_train_std = y_train

        time = X_test_std["T"]
        lat = X_test_std["Y"]
        lon = X_test_std["X"]
        n_time, n_lat, n_lon = len(time), len(lat), len(lon)

        if best_params is None or cluster_da is None:
            best_params, cluster_da = self.compute_hyperparameters(
                X_train, y_train, clim_year_start, clim_year_end
            )

        predictions = np.full((n_time, n_lat, n_lon), np.nan)
        predictions_prob = np.full((n_time, n_lat, n_lon, 3), np.nan)
        self.models = {}

        for c, bp in best_params.items():
            if bp is None:
                continue
                
            mask_3d_train = (cluster_da == c).expand_dims({"T": X_train_std["T"]})
            mask_3d_test = (cluster_da == c).expand_dims({"T": X_test_std["T"]})

            X_train_stacked_c = (
                X_train_std.where(mask_3d_train)
                .stack(sample=("T", "Y", "X"))
                .transpose("sample", "M")
                .values
            )
            y_train_stacked_c = (
                y_train_std.where(mask_3d_train)
                .stack(sample=("T", "Y", "X"))
                .values
                .ravel()
            )

            train_nan_mask_c = np.any(~np.isfinite(X_train_stacked_c), axis=1) | ~np.isfinite(y_train_stacked_c)
            X_train_clean_c = X_train_stacked_c[~train_nan_mask_c]
            y_train_clean_c = y_train_stacked_c[~train_nan_mask_c].astype(int)

            X_test_stacked_c = (
                X_test_std.where(mask_3d_test)
                .stack(sample=("T", "Y", "X"))
                .transpose("sample", "M")
                .values
            )
            test_nan_mask_c = np.any(~np.isfinite(X_test_stacked_c), axis=1)
            X_test_clean_c = X_test_stacked_c[~test_nan_mask_c]

            if len(X_train_clean_c) == 0:
                continue

            # Subsample training for feasibility
            X_train_clean_c, y_train_clean_c = self._subsample(
                X_train_clean_c, y_train_clean_c, self.max_train_samples
            )

            model_c = GaussianProcessClassifier(
                random_state=self.random_state,
                **bp,
            )
            model_c.fit(X_train_clean_c, y_train_clean_c)
            self.models[c] = model_c

            if len(X_test_clean_c) == 0:
                continue
                
            y_pred_c = model_c.predict(X_test_clean_c)
            y_prob_c = model_c.predict_proba(X_test_clean_c)

            # Reconstruct
            full_stacked_c = np.full(X_test_stacked_c.shape[0], np.nan)
            full_stacked_c[~test_nan_mask_c] = y_pred_c
            pred_c_reshaped = full_stacked_c.reshape(n_time, n_lat, n_lon)
            
            # Use np.nan_to_num to handle NaN values in the mask
            mask = np.isnan(predictions)
            predictions = np.where(mask, pred_c_reshaped, predictions)

            # Ensure we always output 3 columns (PB, PN, PA)
            full_stacked_prob_c = np.full((X_test_stacked_c.shape[0], 3), np.nan)

            # Map model class columns -> [0,1,2]
            # In sklearn, classes_ may be subset if a class was absent from training.
            cols = model_c.classes_.astype(int)
            tmp = np.full((y_prob_c.shape[0], 3), 0.0)
            for j, cls in enumerate(cols):
                if 0 <= cls <= 2:
                    tmp[:, cls] = y_prob_c[:, j]

            full_stacked_prob_c[~test_nan_mask_c] = tmp

            pred_prob_c_reshaped = full_stacked_prob_c.reshape(n_time, n_lat, n_lon, 3)
            mask_prob = np.isnan(predictions_prob)
            predictions_prob = np.where(mask_prob, pred_prob_c_reshaped, predictions_prob)

        predicted_da = xr.DataArray(
            data=predictions,
            coords={"T": time, "Y": lat, "X": lon},
            dims=["T", "Y", "X"],
        )

        predicted_prob_da = xr.DataArray(
            data=predictions_prob,
            coords={"T": time, "Y": lat, "X": lon, "probability": [0, 1, 2]},
            dims=["T", "Y", "X", "probability"],
        ).transpose("probability", "T", "Y", "X")

        predicted_prob_da = predicted_prob_da.assign_coords(probability=["PB", "PN", "PA"])
        return predicted_da, predicted_prob_da

    def forecast(
        self,
        Predictant,
        clim_year_start,
        clim_year_end,
        Predictors_train,
        Predictor_for_year,
        best_params=None,
        cluster_da=None,
    ):
        """
        Train on Predictors_train / Predictant, forecast for Predictor_for_year (1 time slice),
        returning deterministic class + tercile probabilities.
        """
        if "M" in Predictant.coords:
            Predictant_no_m = Predictant.isel(M=0).drop_vars("M").squeeze()
        else:
            Predictant_no_m = Predictant

        verify = WAS_Verification()
        Predictant_no_m = verify.compute_class(Predictant_no_m, clim_year_start, clim_year_end)

        mask = xr.where(~np.isnan(Predictant_no_m.isel(T=0)), 1, np.nan)

        Predictors_train_st = standardize_timeseries(Predictors_train, clim_year_start, clim_year_end)
        mean_val = Predictors_train.sel(T=slice(str(clim_year_start), str(clim_year_end))).mean(dim="T")
        std_val = Predictors_train.sel(T=slice(str(clim_year_start), str(clim_year_end))).std(dim="T")
        Predictor_for_year_st = (Predictor_for_year - mean_val) / std_val

        Predictant_st = Predictant_no_m
        Predictors_train_st["T"] = Predictant_st["T"]  # Align time

        time = Predictor_for_year_st["T"]
        lat = Predictor_for_year_st["Y"]
        lon = Predictor_for_year_st["X"]
        n_time, n_lat, n_lon = len(time), len(lat), len(lon)

        if best_params is None or cluster_da is None:
            best_params, cluster_da = self.compute_hyperparameters(
                Predictors_train, Predictant_no_m, clim_year_start, clim_year_end
            )

        predictions = np.full((n_time, n_lat, n_lon), np.nan)
        predictions_prob = np.full((n_time, n_lat, n_lon, 3), np.nan)
        self.models = {}

        for c, bp in best_params.items():
            if bp is None:
                continue
                
            mask_3d_train = (cluster_da == c).expand_dims({"T": Predictors_train_st["T"]})
            mask_3d_test = (cluster_da == c).expand_dims({"T": Predictor_for_year_st["T"]})

            X_train_stacked_c = (
                Predictors_train_st.where(mask_3d_train)
                .stack(sample=("T", "Y", "X"))
                .transpose("sample", "M")
                .values
            )
            y_train_stacked_c = (
                Predictant_st.where(mask_3d_train)
                .stack(sample=("T", "Y", "X"))
                .values
                .ravel()
            )

            train_nan_mask_c = np.any(~np.isfinite(X_train_stacked_c), axis=1) | ~np.isfinite(y_train_stacked_c)
            X_train_clean_c = X_train_stacked_c[~train_nan_mask_c]
            y_train_clean_c = y_train_stacked_c[~train_nan_mask_c].astype(int)

            # Subsample training for feasibility
            X_train_clean_c, y_train_clean_c = self._subsample(
                X_train_clean_c, y_train_clean_c, self.max_train_samples
            )

            X_test_stacked_c = (
                Predictor_for_year_st.where(mask_3d_test)
                .stack(sample=("T", "Y", "X"))
                .transpose("sample", "M")
                .values
            )
            test_nan_mask_c = np.any(~np.isfinite(X_test_stacked_c), axis=1)
            X_test_clean_c = X_test_stacked_c[~test_nan_mask_c]

            if len(X_train_clean_c) == 0:
                continue

            model_c = GaussianProcessClassifier(
                random_state=self.random_state,
                **bp,
            )
            model_c.fit(X_train_clean_c, y_train_clean_c)
            self.models[c] = model_c

            if len(X_test_clean_c) == 0:
                continue
                
            y_pred_c = model_c.predict(X_test_clean_c)
            y_prob_c = model_c.predict_proba(X_test_clean_c)

            # Reconstruct deterministic
            full_stacked_c = np.full(X_test_stacked_c.shape[0], np.nan)
            full_stacked_c[~test_nan_mask_c] = y_pred_c
            pred_c_reshaped = full_stacked_c.reshape(n_time, n_lat, n_lon)
            mask_det = np.isnan(predictions)
            predictions = np.where(mask_det, pred_c_reshaped, predictions)

            # Reconstruct probabilities (always PB/PN/PA)
            full_stacked_prob_c = np.full((X_test_stacked_c.shape[0], 3), np.nan)
            cols = model_c.classes_.astype(int)
            tmp = np.full((y_prob_c.shape[0], 3), 0.0)
            for j, cls in enumerate(cols):
                if 0 <= cls <= 2:
                    tmp[:, cls] = y_prob_c[:, j]
            full_stacked_prob_c[~test_nan_mask_c] = tmp

            pred_prob_c_reshaped = full_stacked_prob_c.reshape(n_time, n_lat, n_lon, 3)
            mask_prob = np.isnan(predictions_prob)
            predictions_prob = np.where(mask_prob, pred_prob_c_reshaped, predictions_prob)

        forecast_det = xr.DataArray(
            data=predictions,
            coords={"T": time, "Y": lat, "X": lon},
            dims=["T", "Y", "X"],
        ) * mask

        forecast_prob = xr.DataArray(
            data=predictions_prob,
            coords={"T": time, "Y": lat, "X": lon, "probability": [0, 1, 2]},
            dims=["T", "Y", "X", "probability"],
        ) * mask

        forecast_prob = forecast_prob.transpose("probability", "T", "Y", "X")
        forecast_prob = forecast_prob.assign_coords(probability=["PB", "PN", "PA"])

        # Update T coordinate (same as your logistic version)
        year = Predictor_for_year.coords["T"].values.astype("datetime64[Y]").astype(int)[0] + 1970
        T_value_1 = Predictant_no_m.isel(T=0).coords["T"].values
        month_1 = T_value_1.astype("datetime64[M]").astype(int) % 12 + 1
        new_T_value = np.datetime64(f"{year}-{month_1:02d}-01")

        forecast_det = forecast_det.assign_coords(T=xr.DataArray([new_T_value], dims=["T"]))
        forecast_det["T"] = forecast_det["T"].astype("datetime64[ns]")

        forecast_prob = forecast_prob.assign_coords(T=xr.DataArray([new_T_value], dims=["T"]))
        forecast_prob["T"] = forecast_prob["T"].astype("datetime64[ns]")

        return forecast_det, forecast_prob

    def get_hpo_summary(self):
        """Get summary of HPO method and parameters."""
        summary = {
            'hpo_method': self.hpo_method,
            'n_clusters': self.n_clusters,
            'max_train_samples': self.max_train_samples,
            'cv_folds': self.cv_folds,
        }
        
        if self.hpo_method == 'random':
            summary['n_random_iter'] = self.n_random_iter
        elif self.hpo_method == 'bayesian':
            summary['n_trials'] = self.n_trials
            summary['timeout'] = self.timeout
        
        return summary



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
                    norm.ppf(0.33, loc=loc, scale=scale),
                    norm.ppf(0.67, loc=loc, scale=scale),
                )
            elif code == 2:
                return (
                    lognorm.ppf(0.33, s=shape, loc=loc, scale=scale),
                    lognorm.ppf(0.67, s=shape, loc=loc, scale=scale),
                )
            elif code == 3:
                return (
                    expon.ppf(0.33, loc=loc, scale=scale),
                    expon.ppf(0.67, loc=loc, scale=scale),
                )
            elif code == 4:
                return (
                    gamma.ppf(0.33, a=shape, loc=loc, scale=scale),
                    gamma.ppf(0.67, a=shape, loc=loc, scale=scale),
                )
            elif code == 5:
                return (
                    weibull_min.ppf(0.33, c=shape, loc=loc, scale=scale),
                    weibull_min.ppf(0.67, c=shape, loc=loc, scale=scale),
                )
            elif code == 6:
                # Note: Renamed 't_dist' to 't' for standard scipy.stats
                return (
                    t.ppf(0.33, df=shape, loc=loc, scale=scale),
                    t.ppf(0.67, df=shape, loc=loc, scale=scale),
                )
            elif code == 7:
                # Poisson: poisson.ppf(q, mu, loc=0)
                # ASSUMPTION: 'mu' (mean) is passed as 'shape'
                #             'loc' is passed as 'loc'
                #             'scale' is unused
                return (
                    poisson.ppf(0.33, mu=shape, loc=loc),
                    poisson.ppf(0.67, mu=shape, loc=loc),
                )
            elif code == 8:
                # Negative Binomial: nbinom.ppf(q, n, p, loc=0)
                # ASSUMPTION: 'n' (successes) is passed as 'shape'
                #             'p' (probability) is passed as 'scale'
                #             'loc' is passed as 'loc'
                return (
                    nbinom.ppf(0.33, n=shape, p=scale, loc=loc),
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
        terciles_emp = clim.quantile([0.33, 0.67], dim="T")
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


    def forecast(self, Predictant, clim_year_start, clim_year_end, hindcast_det, hindcast_det_cross_val, Predictor_for_year, best_code_da=None, best_shape_da=None, best_loc_da=None, best_scale_da=None):
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
        T1_emp = terciles.isel(quantile=0).drop_vars('quantile')
        T2_emp = terciles.isel(quantile=1).drop_vars('quantile')
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
                result_,
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
                result_,
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
        return result_ * mask, mask * forecast_prob.transpose('probability', 'T', 'Y', 'X')



class HPELMWrapper(BaseEstimator, RegressorMixin):
    """
    Wrapper for HPELM to make it compatible with scikit-learn's hyperparameter optimization.
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

    def score(self, X, y):
        from sklearn.metrics import mean_squared_error
        y_pred = self.predict(X)
        return -mean_squared_error(y, y_pred)  # Negative MSE for optimization

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
    Implements hyperparameter optimization via multiple methods.
    
    Parameters
    ----------
    neurons_range : list of int, optional
        List of neuron counts to tune for HPELM (default is [10, 20, 50, 100]).
    activation_options : list of str, optional
        Activation functions to tune for HPELM (default is ['sigm', 'tanh', 'lin', 'rbf_l1', 'rbf_l2', 'rbf_linf']).
    norm_range : list of float, optional
        Regularization parameters to tune for HPELM (default is [0.1, 1.0, 10.0, 100.0]).
    random_state : int, optional
        Seed for reproducibility (default is 42).
    dist_method : str, optional
        Distribution method for tercile probabilities.
    search_method : str, optional
        Hyperparameter optimization method: 'grid', 'random', or 'bayesian' (default: 'random').
    n_iter_search : int, optional
        Number of iterations for randomized search (default is 10).
    cv_folds : int, optional
        Number of cross-validation folds (default is 3).
    n_clusters : int, optional
        Number of clusters for homogenized zones (default is 4).
    n_trials_bayesian : int, optional
        Number of trials for Bayesian optimization (default=50).
    bayesian_sampler : str, optional
        Sampler for Bayesian optimization: 'tpe' or 'random' (default='tpe').
    scoring : str, optional
        Scoring metric for optimization (default='neg_mean_squared_error').
    """
    def __init__(self,
                 neurons_range=[10, 20, 50, 100],
                 activation_options=['sigm', 'tanh', 'relu', 'rbf_linf', 'rbf_gauss'],
                 norm_range=[0.1, 1.0, 10.0, 100.0],
                 random_state=42,
                 dist_method="nonparam",
                 search_method='random',
                 n_iter_search=10,
                 cv_folds=3,
                 n_clusters=4,
                 n_trials_bayesian=50,
                 bayesian_sampler='tpe',
                 scoring='neg_mean_squared_error'):
        
        self.neurons_range = neurons_range
        self.activation_options = activation_options
        self.norm_range = norm_range
        self.random_state = random_state
        self.dist_method = dist_method
        self.search_method = search_method
        self.n_iter_search = n_iter_search
        self.cv_folds = cv_folds
        self.n_clusters = n_clusters
        self.n_trials_bayesian = n_trials_bayesian
        self.bayesian_sampler = bayesian_sampler
        self.scoring = scoring
        self.hpelm = None
        self.best_params_dict = None
        self.bayesian_studies = {}
        
        # Validate optimization method
        valid_methods = ['grid', 'random', 'bayesian']
        if self.search_method not in valid_methods:
            raise ValueError(f"search_method must be one of {valid_methods}, got '{self.search_method}'")

    def _get_bounds(self, param_range):
            """Helper to extract (min, max, is_dist) from a list or scipy distribution."""
            if hasattr(param_range, 'support'): # Scipy distribution
                low, high = param_range.support()
                return float(low), float(high), True
            else: # List or array
                return float(min(param_range)), float(max(param_range)), False
    

    def _create_bayesian_sampler(self):
        """Create sampler for Bayesian optimization."""
        if self.bayesian_sampler == 'tpe':
            return TPESampler(seed=self.random_state)
        elif self.bayesian_sampler == 'random':
            return RandomSampler(seed=self.random_state)
        else:
            return TPESampler(seed=self.random_state)

    # def _bayesian_objective(self, trial, X, y):
    #     """Objective function for Bayesian optimization."""
    #     neurons = trial.suggest_categorical('neurons', self.neurons_range)
    #     activation = trial.suggest_categorical('activation', self.activation_options)
    #     norm = trial.suggest_float('norm', min(self.norm_range), max(self.norm_range), log=True)
        
    #     # Create and evaluate model
    #     model = HPELMWrapper(neurons=neurons, activation=activation, norm=norm, 
    #                        random_state=self.random_state)
        
    #     # Simple cross-validation
    #     from sklearn.model_selection import cross_val_score
    #     scores = cross_val_score(model, X, y, cv=self.cv_folds, 
    #                              scoring=self.scoring, n_jobs=-1)
    #     return np.mean(scores)

    def _bayesian_objective(self, trial, X, y):
            """Objective function for Bayesian optimization."""
            # Handle Neurons
            low_n, high_n, is_dist_n = self._get_bounds(self.neurons_range)
            if is_dist_n:
                neurons = trial.suggest_int('neurons', int(low_n), int(high_n))
            else:
                neurons = trial.suggest_categorical('neurons', [int(n) for n in self.neurons_range])
    
            # Handle Activation
            activation = trial.suggest_categorical('activation', self.activation_options)
    
            # Handle Norm
            low_f, high_f, _ = self._get_bounds(self.norm_range)
            # We use log=True assuming regularization parameters often span orders of magnitude
            norm = trial.suggest_float('norm', low_f, high_f, log=True)
            
            model = HPELMWrapper(neurons=neurons, activation=activation, norm=norm, 
                                 random_state=self.random_state)
            
            # Set n_jobs=1 here to avoid conflicts with Optuna's n_jobs=-1
            from sklearn.model_selection import cross_val_score
            scores = cross_val_score(model, X, y, cv=self.cv_folds, 
                                     scoring=self.scoring, n_jobs=1)
            return np.mean(scores)

    # def _grid_search_optimization(self, X, y):
    #     """Perform grid search optimization."""
    #     param_grid = {
    #         'neurons': self.neurons_range,
    #         'activation': self.activation_options,
    #         'norm': self.norm_range
    #     }
        
    #     model = HPELMWrapper(random_state=self.random_state)
    #     grid_search = GridSearchCV(
    #         model, param_grid=param_grid, cv=self.cv_folds,
    #         scoring=self.scoring, n_jobs=-1, verbose=0
    #     )
    #     grid_search.fit(X, y)
    #     return grid_search.best_params_

    def _grid_search_optimization(self, X, y):
            """Perform grid search optimization with distribution handling."""
            # Grid search CANNOT take a distribution object. 
            # If a distribution is provided, we must sample N points to create a grid.
            param_grid = {
                'activation': self.activation_options
            }
    
            for name, p_range in zip(['neurons', 'norm'], [self.neurons_range, self.norm_range]):
                if hasattr(p_range, 'support'):
                    low, high = p_range.support()
                    # Create 5 pointall_model_hdcsts across the range for the grid
                    if name == 'norm':
                        param_grid[name] = np.logspace(np.log10(low), np.log10(high), 5).tolist()
                    else:
                        param_grid[name] = np.linspace(low, high, 5).astype(int).tolist()
                else:
                    param_grid[name] = p_range
            
            model = HPELMWrapper(random_state=self.random_state)
            grid_search = GridSearchCV(model, param_grid=param_grid, cv=self.cv_folds,
                                       scoring=self.scoring, n_jobs=-1)
            grid_search.fit(X, y)
            return grid_search.best_params_

    def _random_search_optimization(self, X, y):
        """Perform random search optimization."""
        param_dist = {
            'neurons': self.neurons_range,
            'activation': self.activation_options,
            'norm': self.norm_range
        }
        
        model = HPELMWrapper(random_state=self.random_state)
        random_search = RandomizedSearchCV(
            model, param_distributions=param_dist, n_iter=self.n_iter_search,
            cv=self.cv_folds, scoring=self.scoring, random_state=self.random_state,
            n_jobs=-1, verbose=0
        )
        random_search.fit(X, y)
        return random_search.best_params_

    def _bayesian_optimization(self, X, y, cluster_id=None):
        """Perform Bayesian optimization with Optuna."""
        study_name = f"cluster_{cluster_id}" if cluster_id is not None else "global"
        
        study = optuna.create_study(
            direction='maximize',  # maximize negative MSE = minimize MSE
            sampler=self._create_bayesian_sampler(),
            study_name=study_name
        )
        
        # Optimize
        objective_func = partial(self._bayesian_objective, X=X, y=y)
        study.optimize(objective_func, n_trials=self.n_trials_bayesian, n_jobs=-1)
        
        # Store study for analysis
        if cluster_id is not None:
            self.bayesian_studies[cluster_id] = study
        
        return study.best_params

    def compute_hyperparameters(self, Predictors, Predictand, clim_year_start, clim_year_end):
        """
        Independently computes the best hyperparameters using selected optimization method
        on stacked training data for each homogenized zone.
        
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
        from sklearn.cluster import KMeans
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

            # Select optimization method
            if self.search_method == 'grid':
                best_params = self._grid_search_optimization(X_clean_c, y_clean_c)
            elif self.search_method == 'random':
                best_params = self._random_search_optimization(X_clean_c, y_clean_c)
            elif self.search_method == 'bayesian':
                best_params = self._bayesian_optimization(X_clean_c, y_clean_c, cluster_id=c)
            else:
                raise ValueError(f"Unknown optimization method: {self.search_method}")

            best_params_dict[c] = best_params

        self.best_params_dict = best_params_dict
        return best_params_dict, cluster_da

    def get_optimization_results(self):
        """
        Return optimization results for analysis.
        
        Returns
        -------
        dict
            Dictionary containing optimization results.
        """
        results = {
            'best_params': self.best_params_dict,
            'search_method': self.search_method,
            'bayesian_studies': self.bayesian_studies
        }
        
        if self.search_method == 'bayesian' and self.bayesian_studies:
            for cluster_id, study in self.bayesian_studies.items():
                results[f'bayesian_trials_cluster_{cluster_id}'] = study.trials_dataframe()
        
        return results

    def plot_optimization_history(self, cluster_id=None):
        """
        Plot optimization history (for Bayesian optimization only).
        
        Parameters
        ----------
        cluster_id : int, optional
            Specific cluster to plot. If None, plots all clusters.
        """
        if self.search_method != 'bayesian':
            print("Plotting only available for Bayesian optimization")
            return
        
        import matplotlib.pyplot as plt
        
        if cluster_id is not None:
            if cluster_id in self.bayesian_studies:
                fig = optuna.visualization.plot_optimization_history(
                    self.bayesian_studies[cluster_id]
                )
                fig.show()
        else:
            fig, axes = plt.subplots(len(self.bayesian_studies), 1, 
                                     figsize=(10, 4*len(self.bayesian_studies)))
            if len(self.bayesian_studies) == 1:
                axes = [axes]
            
            for idx, (c_id, study) in enumerate(self.bayesian_studies.items()):
                df = study.trials_dataframe()
                axes[idx].plot(df['number'], df['value'], 'o-')
                axes[idx].set_title(f'Cluster {c_id}')
                axes[idx].set_xlabel('Trial')
                axes[idx].set_ylabel('Score (Negative MSE)')
                axes[idx].grid(True)
            
            plt.tight_layout()
            plt.show()
    
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
                    norm.ppf(0.33, loc=loc, scale=scale),
                    norm.ppf(0.67, loc=loc, scale=scale),
                )
            elif code == 2:
                return (
                    lognorm.ppf(0.33, s=shape, loc=loc, scale=scale),
                    lognorm.ppf(0.67, s=shape, loc=loc, scale=scale),
                )
            elif code == 3:
                return (
                    expon.ppf(0.33, loc=loc, scale=scale),
                    expon.ppf(0.67, loc=loc, scale=scale),
                )
            elif code == 4:
                return (
                    gamma.ppf(0.33, a=shape, loc=loc, scale=scale),
                    gamma.ppf(0.67, a=shape, loc=loc, scale=scale),
                )
            elif code == 5:
                return (
                    weibull_min.ppf(0.33, c=shape, loc=loc, scale=scale),
                    weibull_min.ppf(0.67, c=shape, loc=loc, scale=scale),
                )
            elif code == 6:
                # Note: Renamed 't_dist' to 't' for standard scipy.stats
                return (
                    t.ppf(0.33, df=shape, loc=loc, scale=scale),
                    t.ppf(0.67, df=shape, loc=loc, scale=scale),
                )
            elif code == 7:
                # Poisson: poisson.ppf(q, mu, loc=0)
                # ASSUMPTION: 'mu' (mean) is passed as 'shape'
                #             'loc' is passed as 'loc'
                #             'scale' is unused
                return (
                    poisson.ppf(0.33, mu=shape, loc=loc),
                    poisson.ppf(0.67, mu=shape, loc=loc),
                )
            elif code == 8:
                # Negative Binomial: nbinom.ppf(q, n, p, loc=0)
                # ASSUMPTION: 'n' (successes) is passed as 'shape'
                #             'p' (probability) is passed as 'scale'
                #             'loc' is passed as 'loc'
                return (
                    nbinom.ppf(0.33, n=shape, p=scale, loc=loc),
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
        terciles_emp = clim.quantile([0.33, 0.67], dim="T")
        T1_emp = terciles_emp.isel(quantile=0).drop_vars("quantile")
        T2_emp = terciles_emp.isel(quantile=1).drop_vars("quantile")
        

        dm = self.dist_method

        # ---------- BESTFIT: zone-wise optimal distributions ----------
        if dm == "bestfit":
            if any(v is None for v in (best_code_da, best_shape_da, best_loc_da, best_scale_da)):
                raise ValueError(
                    "dist_method='bestfit' requires best_code_da, best_shape_da, best_loc_da, best_scale_da."
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

    def forecast(self, Predictant, clim_year_start, clim_year_end, hindcast_det, hindcast_det_cross, Predictor_for_year, best_params=None, cluster_da=None, best_code_da=None, best_shape_da=None, best_loc_da=None, best_scale_da=None
                ):
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
        T1_emp = terciles.isel(quantile=0).drop_vars('quantile')
        T2_emp = terciles.isel(quantile=1).drop_vars('quantile')
        error_variance = (Predictant_no_m - hindcast_det_cross).var(dim='T')
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
                result_da,
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
                result_da,
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

    def compare_optimization_methods(self, X_train, y_train, methods=['grid', 'random', 'bayesian']):
        """
        Compare different hyperparameter optimization methods.
        
        Parameters
        ----------
        X_train : array-like
            Training features
        y_train : array-like
            Training targets
        methods : list
            List of methods to compare
            
        Returns
        -------
        dict
            Comparison results for each method
        """
        results = {}
        
        for method in methods:
            print(f"\n{'='*60}")
            print(f"Testing {method.upper()} optimization")
            print('='*60)
            
            # Create model with specific optimization method
            model = WAS_mme_hpELM(
                search_method=method,
                n_iter_search=10 if method == 'random' else None,
                n_trials_bayesian=30 if method == 'bayesian' else None
            )
            
            # Time the optimization
            import time
            start_time = time.time()
            
            # Create dummy cluster data for single cluster test
            from sklearn.cluster import KMeans
            kmeans = KMeans(n_clusters=1, random_state=42)
            cluster_labels = np.zeros(len(y_train))
            
            # Optimize for single cluster
            if method == 'grid':
                best_params = model._grid_search_optimization(X_train, y_train)
            elif method == 'random':
                best_params = model._random_search_optimization(X_train, y_train)
            else:  # bayesian
                best_params = model._bayesian_optimization(X_train, y_train)
            
            elapsed_time = time.time() - start_time
            
            # Store results
            results[method] = {
                'best_params': best_params,
                'time': elapsed_time,
                'method': method
            }
            
            print(f"Best parameters: {best_params}")
            print(f"Time elapsed: {elapsed_time:.2f} seconds")
        
        return results


# # 1. Grid Search (exhaustive)
# model_grid = WAS_mme_hpELM(
#     search_method='grid',
#     neurons_range=[10, 50, 100],
#     activation_options=['sigm', 'tanh'],
#     norm_range=[0.1, 1.0, 10.0]
# )

# # 2. Random Search (default)
# model_random = WAS_mme_hpELM(
#     search_method='random',  # Default
#     n_iter_search=20,
#     cv_folds=3
# )

# # 3. Bayesian Optimization
# model_bayesian = WAS_mme_hpELM(
#     search_method='bayesian',
#     n_trials_bayesian=100,
#     bayesian_sampler='tpe',  # Tree-structured Parzen Estimator
#     cv_folds=3
# )

# # Compute hyperparameters with selected method
# best_params, cluster_da = model_bayesian.compute_hyperparameters(
#     predictors, predictand, 1981, 2010
# )

# # Get optimization results
# results = model_bayesian.get_optimization_results()
# print(f"Optimization method: {results['search_method']}")
# print(f"Best parameters: {results['best_params']}")

# # Plot optimization history for Bayesian method
# if model_bayesian.search_method == 'bayesian':
#     model_bayesian.plot_optimization_history(cluster_id=0)
    
class WAS_mme_hpELM_:
    """
    Extreme Learning Machine (ELM) based Multi-Model Ensemble (MME) forecasting using hpelm library.
    This class implements a single-model forecasting approach using HPELM for deterministic predictions,
    with optional tercile probability calculations using various statistical distributions.
    Implements hyperparameter optimization via multiple methods.

    Parameters
    ----------
    neurons_range : list of int, optional
        List of neuron counts to tune for HPELM (default is [10, 20, 50, 100]).
    activation_options : list of str, optional
        Activation functions to tune for HPELM (default is ['sigm', 'tanh', 'lin', 'rbf_l1', 'rbf_l2', 'rbf_linf']).
    norm_range : list of float, optional
        Regularization parameters to tune for HPELM (default is [0.1, 1.0, 10.0, 100.0]).
    random_state : int, optional
        Seed for reproducibility (default is 42).
    dist_method : str, optional
        Distribution method for tercile probabilities ('bestfit' or 'nonparam'.) (default is 'bestfit').
    search_method : str, optional
        Hyperparameter optimization method: 'grid', 'random', or 'bayesian' (default: 'random').
    n_iter_search : int, optional
        Number of iterations for randomized search (default is 10).
    cv_folds : int, optional
        Number of cross-validation folds (default is 3).
    n_trials_bayesian : int, optional
        Number of trials for Bayesian optimization (default=50).
    bayesian_sampler : str, optional
        Sampler for Bayesian optimization: 'tpe' or 'random' (default='tpe').
    scoring : str, optional
        Scoring metric for optimization (default='neg_mean_squared_error').
    """
    def __init__(self,
                 neurons_range=[10, 20, 50, 100],
                 activation_options=['sigm', 'tanh', 'relu', 'rbf_linf', 'rbf_gauss'],
                 norm_range=[0.1, 1.0, 10.0, 100.0],
                 random_state=42,
                 dist_method="gamma",
                 search_method='random',
                 n_iter_search=10,
                 cv_folds=3,
                 n_trials_bayesian=50,
                 bayesian_sampler='tpe',
                 scoring='neg_mean_squared_error'):
        
        self.neurons_range = neurons_range
        self.activation_options = activation_options
        self.norm_range = norm_range
        self.random_state = random_state
        self.dist_method = dist_method
        self.search_method = search_method
        self.n_iter_search = n_iter_search
        self.cv_folds = cv_folds
        self.n_trials_bayesian = n_trials_bayesian
        self.bayesian_sampler = bayesian_sampler
        self.scoring = scoring
        self.hpelm = None
        self.best_params = None
        self.bayesian_study = None
        
        # Validate optimization method
        valid_methods = ['grid', 'random', 'bayesian']
        if self.search_method not in valid_methods:
            raise ValueError(f"search_method must be one of {valid_methods}, got '{self.search_method}'")

    def _get_bounds(self, param_range):
            """Helper to extract (min, max, is_dist) from a list or scipy distribution."""
            if hasattr(param_range, 'support'): # Scipy distribution
                low, high = param_range.support()
                return float(low), float(high), True
            else: # List or array
                return float(min(param_range)), float(max(param_range)), False
    

    def _create_bayesian_sampler(self):
        """Create sampler for Bayesian optimization."""
        if self.bayesian_sampler == 'tpe':
            return TPESampler(seed=self.random_state)
        elif self.bayesian_sampler == 'random':
            return RandomSampler(seed=self.random_state)
        else:
            return TPESampler(seed=self.random_state)

    # def _bayesian_objective(self, trial, X, y):
    #     """Objective function for Bayesian optimization."""
    #     neurons = trial.suggest_categorical('neurons', self.neurons_range)
    #     activation = trial.suggest_categorical('activation', self.activation_options)
    #     norm = trial.suggest_float('norm', min(self.norm_range), max(self.norm_range), log=True)
        
    #     # Create and evaluate model
    #     model = HPELMWrapper(neurons=neurons, activation=activation, norm=norm, 
    #                        random_state=self.random_state)
        
    #     # Simple cross-validation
    #     from sklearn.model_selection import cross_val_score
    #     scores = cross_val_score(model, X, y, cv=self.cv_folds, 
    #                              scoring=self.scoring, n_jobs=-1)
    #     return np.mean(scores)

    def _bayesian_objective(self, trial, X, y):
            """Objective function for Bayesian optimization."""
            # Handle Neurons
            low_n, high_n, is_dist_n = self._get_bounds(self.neurons_range)
            if is_dist_n:
                neurons = trial.suggest_int('neurons', int(low_n), int(high_n))
            else:
                neurons = trial.suggest_categorical('neurons', [int(n) for n in self.neurons_range])
    
            # Handle Activation
            activation = trial.suggest_categorical('activation', self.activation_options)
    
            # Handle Norm
            low_f, high_f, _ = self._get_bounds(self.norm_range)
            # We use log=True assuming regularization parameters often span orders of magnitude
            norm = trial.suggest_float('norm', low_f, high_f, log=True)
            
            model = HPELMWrapper(neurons=neurons, activation=activation, norm=norm, 
                                 random_state=self.random_state)
            
            # Set n_jobs=1 here to avoid conflicts with Optuna's n_jobs=-1
            from sklearn.model_selection import cross_val_score
            scores = cross_val_score(model, X, y, cv=self.cv_folds, 
                                     scoring=self.scoring, n_jobs=1)
            return np.mean(scores)

    # def _grid_search_optimization(self, X, y):
    #     """Perform grid search optimization."""
    #     param_grid = {
    #         'neurons': self.neurons_range,
    #         'activation': self.activation_options,
    #         'norm': self.norm_range
    #     }
        
    #     model = HPELMWrapper(random_state=self.random_state)
    #     grid_search = GridSearchCV(
    #         model, param_grid=param_grid, cv=self.cv_folds,
    #         scoring=self.scoring, n_jobs=-1, verbose=0
    #     )
    #     grid_search.fit(X, y)
    #     return grid_search.best_params_

    def _grid_search_optimization(self, X, y):
            """Perform grid search optimization with distribution handling."""
            # Grid search CANNOT take a distribution object. 
            # If a distribution is provided, we must sample N points to create a grid.
            param_grid = {
                'activation': self.activation_options
            }
    
            for name, p_range in zip(['neurons', 'norm'], [self.neurons_range, self.norm_range]):
                if hasattr(p_range, 'support'):
                    low, high = p_range.support()
                    # Create 5 pointall_model_hdcsts across the range for the grid
                    if name == 'norm':
                        param_grid[name] = np.logspace(np.log10(low), np.log10(high), 5).tolist()
                    else:
                        param_grid[name] = np.linspace(low, high, 5).astype(int).tolist()
                else:
                    param_grid[name] = p_range
            
            model = HPELMWrapper(random_state=self.random_state)
            grid_search = GridSearchCV(model, param_grid=param_grid, cv=self.cv_folds,
                                       scoring=self.scoring, n_jobs=-1)
            grid_search.fit(X, y)
            return grid_search.best_params_

    def _random_search_optimization(self, X, y):
        """Perform random search optimization."""
        param_dist = {
            'neurons': self.neurons_range,
            'activation': self.activation_options,
            'norm': self.norm_range
        }
        
        model = HPELMWrapper(random_state=self.random_state)
        random_search = RandomizedSearchCV(
            model, param_distributions=param_dist, n_iter=self.n_iter_search,
            cv=self.cv_folds, scoring=self.scoring, random_state=self.random_state,
            n_jobs=-1, verbose=0
        )
        random_search.fit(X, y)
        return random_search.best_params_

    def _bayesian_optimization(self, X, y, cluster_id=None):
        """Perform Bayesian optimization with Optuna."""
        study_name = f"cluster_{cluster_id}" if cluster_id is not None else "global"
        
        study = optuna.create_study(
            direction='maximize',  # maximize negative MSE = minimize MSE
            sampler=self._create_bayesian_sampler(),
            study_name=study_name
        )
        
        # Optimize
        objective_func = partial(self._bayesian_objective, X=X, y=y)
        study.optimize(objective_func, n_trials=self.n_trials_bayesian, n_jobs=-1)
        
        # Store study for analysis
        if cluster_id is not None:
            self.bayesian_studies[cluster_id] = study
        
        return study.best_params

    def compute_hyperparameters(self, Predictors, Predictand, clim_year_start, clim_year_end):
        """
        Computes the best hyperparameters using selected optimization method on stacked training data.

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
        if "M" in Predictand.coords:
            Predictand = Predictand.isel(M=0).drop_vars('M').squeeze()

        # Predictand.name = "varname"
        
        X_train = standardize_timeseries(Predictors, clim_year_start, clim_year_end)
        y_train = standardize_timeseries(Predictand, clim_year_start, clim_year_end)

        # Stack training data
        X_train_stacked = X_train.stack(sample=('T', 'Y', 'X')).transpose('sample', 'M').values
        y_train_stacked = y_train.stack(sample=('T', 'Y', 'X')).values.ravel()
        train_nan_mask = np.any(~np.isfinite(X_train_stacked), axis=1) | ~np.isfinite(y_train_stacked)
        X_train_clean = X_train_stacked[~train_nan_mask]
        y_train_clean = y_train_stacked[~train_nan_mask]

        if len(X_train_clean) == 0:
            raise ValueError("No valid training data after removing NaNs")

        # Select optimization method
        if self.search_method == 'grid':
            best_params = self._grid_search_optimization(X_train_clean, y_train_clean)
        elif self.search_method == 'random':
            best_params = self._random_search_optimization(X_train_clean, y_train_clean)
        elif self.search_method == 'bayesian':
            best_params = self._bayesian_optimization(X_train_clean, y_train_clean)
        else:
            raise ValueError(f"Unknown optimization method: {self.search_method}")

        self.best_params = best_params
        return best_params

    def get_optimization_results(self):
        """
        Return optimization results for analysis.
        
        Returns
        -------
        dict
            Dictionary containing optimization results.
        """
        results = {
            'best_params': self.best_params,
            'search_method': self.search_method,
            'bayesian_study': self.bayesian_study
        }
        
        if self.search_method == 'bayesian' and self.bayesian_study is not None:
            results['bayesian_trials'] = self.bayesian_study.trials_dataframe()
            results['best_trial'] = self.bayesian_study.best_trial.params
        
        return results

    def plot_optimization_history(self):
        """
        Plot optimization history (for Bayesian optimization only).
        """
        if self.search_method != 'bayesian' or self.bayesian_study is None:
            print("Plotting only available for Bayesian optimization with completed study")
            return
        
        import matplotlib.pyplot as plt
        
        fig = plt.figure(figsize=(10, 6))
        df = self.bayesian_study.trials_dataframe()
        
        plt.plot(df['number'], df['value'], 'o-', label='Trial Score')
        plt.axhline(y=self.bayesian_study.best_value, color='r', linestyle='--', 
                   label=f'Best Score: {self.bayesian_study.best_value:.4f}')
        
        plt.xlabel('Trial Number')
        plt.ylabel('Score (Negative MSE)')
        plt.title('Bayesian Optimization History')
        plt.legend()
        plt.grid(True, alpha=0.3)
        plt.tight_layout()
        plt.show()


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
                    norm.ppf(0.33, loc=loc, scale=scale),
                    norm.ppf(0.67, loc=loc, scale=scale),
                )
            elif code == 2:
                return (
                    lognorm.ppf(0.33, s=shape, loc=loc, scale=scale),
                    lognorm.ppf(0.67, s=shape, loc=loc, scale=scale),
                )
            elif code == 3:
                return (
                    expon.ppf(0.33, loc=loc, scale=scale),
                    expon.ppf(0.67, loc=loc, scale=scale),
                )
            elif code == 4:
                return (
                    gamma.ppf(0.33, a=shape, loc=loc, scale=scale),
                    gamma.ppf(0.67, a=shape, loc=loc, scale=scale),
                )
            elif code == 5:
                return (
                    weibull_min.ppf(0.33, c=shape, loc=loc, scale=scale),
                    weibull_min.ppf(0.67, c=shape, loc=loc, scale=scale),
                )
            elif code == 6:
                # Note: Renamed 't_dist' to 't' for standard scipy.stats
                return (
                    t.ppf(0.33, df=shape, loc=loc, scale=scale),
                    t.ppf(0.67, df=shape, loc=loc, scale=scale),
                )
            elif code == 7:
                # Poisson: poisson.ppf(q, mu, loc=0)
                # ASSUMPTION: 'mu' (mean) is passed as 'shape'
                #             'loc' is passed as 'loc'
                #             'scale' is unused
                return (
                    poisson.ppf(0.33, mu=shape, loc=loc),
                    poisson.ppf(0.67, mu=shape, loc=loc),
                )
            elif code == 8:
                # Negative Binomial: nbinom.ppf(q, n, p, loc=0)
                # ASSUMPTION: 'n' (successes) is passed as 'shape'
                #             'p' (probability) is passed as 'scale'
                #             'loc' is passed as 'loc'
                return (
                    nbinom.ppf(0.33, n=shape, p=scale, loc=loc),
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
        terciles_emp = clim.quantile([0.33, 0.67], dim="T")
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


    def forecast(self, Predictant, clim_year_start, clim_year_end, hindcast_det, hindcast_det_cross, Predictor_for_year, best_params=None, best_code_da=None, best_shape_da=None, best_loc_da=None, best_scale_da=None):
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
        Predictor_for_year_st = Predictor_for_year_st * mask

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
        T1_emp = terciles.isel(quantile=0).drop_vars('quantile')
        T2_emp = terciles.isel(quantile=1).drop_vars('quantile')
        error_variance = (Predictant_no_m - hindcast_det_cross).var(dim='T')
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
                result_da,
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
                result_da,
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



class WAS_mme_MLP_:
    """
    Multi-Layer Perceptron (MLP) for Multi-Model Ensemble (MME) forecasting.
    This class implements a Multi-Layer Perceptron model using scikit-learn's MLPRegressor
    for deterministic forecasting, with optional tercile probability calculations using
    various statistical distributions. Implements multiple hyperparameter optimization methods.
    
    Parameters
    ----------
    search_method : str, optional
        Hyperparameter optimization method: 'grid', 'random', or 'bayesian' (default: 'random').
    hidden_layer_sizes_range : list of tuples, optional
        List of hidden layer sizes to tune, e.g., [(10,), (10, 5), (20, 10)] (default).
    activation_options : list of str, optional
        Activation functions to tune ('identity', 'logistic', 'tanh', 'relu') (default is ['relu', 'tanh', 'logistic']).
    solver_options : list of str, optional
        Solvers to tune ('lbfgs', 'sgd', 'adam') (default is ['adam', 'sgd', 'lbfgs']).
    alpha_range : list of float or scipy.stats distribution, optional
        L2 regularization parameters to tune (default is [0.0001, 0.001, 0.01, 0.1]).
        Can be a list for grid search or a distribution for random/bayesian search.
    learning_rate_init_range : list or scipy.stats distribution, optional
        Learning rate initialization range (default is loguniform(0.0001, 0.01)).
        For grid search, provide a list of values.
    max_iter : int, optional
        Maximum iterations (default is 200).
    random_state : int, optional
        Seed for reproducibility (default is 42).
    dist_method : str, optional
        Distribution method for tercile probabilities ('bestfit', 'nonparam', etc.) (default is 'nonparam').
    n_iter_search : int, optional
        Number of iterations for randomized/bayesian search or points to sample for grid search (default is 10).
    cv_folds : int, optional
        Number of cross-validation folds (default is 3).
    optuna_n_jobs : int, optional
        Number of parallel jobs for Optuna (default is 1).
    optuna_timeout : int, optional
        Timeout in seconds for Optuna optimization (default is None).
    """
    def __init__(self,
                 search_method='random',
                 hidden_layer_sizes_range=[(10,), (10, 5), (20, 10)],
                 learning_rate_init_range=loguniform(0.0001, 0.01),
                 activation_options=['relu', 'tanh', 'logistic'],
                 solver_options=['adam', 'sgd', 'lbfgs'],
                 alpha_range=[0.0001, 0.001, 0.01, 0.1],
                 max_iter=200, 
                 random_state=42, 
                 dist_method="nonparam",
                 n_iter_search=10, 
                 cv_folds=3,
                 optuna_n_jobs=1,
                 optuna_timeout=None):

        self.search_method = search_method
        self.hidden_layer_sizes_range = hidden_layer_sizes_range
        self.learning_rate_init_range = learning_rate_init_range
        self.activation_options = activation_options
        self.solver_options = solver_options
        self.alpha_range = alpha_range
        self.max_iter = max_iter
        self.random_state = random_state
        self.dist_method = dist_method
        self.n_iter_search = n_iter_search
        self.cv_folds = cv_folds
        self.optuna_n_jobs = optuna_n_jobs
        self.optuna_timeout = optuna_timeout
        self.mlp = None

    def _objective(self, trial, X_train, y_train):
        """
        Objective function for Optuna optimization.
        """
        # Define hyperparameter search space
        hidden_layer_sizes = trial.suggest_categorical(
            'hidden_layer_sizes', self.hidden_layer_sizes_range
        )
        activation = trial.suggest_categorical(
            'activation', self.activation_options
        )
        solver = trial.suggest_categorical(
            'solver', self.solver_options
        )
        
        # Handle learning_rate_init based on search method
        if isinstance(self.learning_rate_init_range, list):
            learning_rate_init = trial.suggest_categorical(
                'learning_rate_init', self.learning_rate_init_range
            )
        else:
            # Assume it's a distribution
            learning_rate_init = trial.suggest_float(
                'learning_rate_init', 
                self.learning_rate_init_range.a,
                self.learning_rate_init_range.b,
                log=True
            )
        
        # Handle alpha based on search method
        if isinstance(self.alpha_range, list):
            alpha = trial.suggest_categorical('alpha', self.alpha_range)
        else:
            # Assume it's a distribution
            alpha = trial.suggest_float(
                'alpha', 
                self.alpha_range.a,
                self.alpha_range.b,
                log=True
            )
        
        # Create and train model
        model = MLPRegressor(
            hidden_layer_sizes=hidden_layer_sizes,
            learning_rate_init=learning_rate_init,
            activation=activation,
            solver=solver,
            alpha=alpha,
            max_iter=self.max_iter,
            random_state=self.random_state
        )
        
        # Use cross-validation
        from sklearn.model_selection import cross_val_score
        scores = cross_val_score(
            model, X_train, y_train, 
            cv=self.cv_folds, 
            scoring='neg_mean_squared_error',
            n_jobs=-1
        )
        
        return np.mean(scores)

    def compute_hyperparameters(self, Predictors, Predictand, clim_year_start, clim_year_end):
        """
        Independently computes the best hyperparameters using selected optimization method
        on stacked training data.

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
        y_train_stacked = y_train.stack(sample=('T', 'Y', 'X')).values.ravel()  # Flatten to 1D
        train_nan_mask = np.any(~np.isfinite(X_train_stacked), axis=1) | ~np.isfinite(y_train_stacked)
        X_train_clean = X_train_stacked[~train_nan_mask]
        y_train_clean = y_train_stacked[~train_nan_mask]

        if self.search_method == 'grid':
            # Prepare parameter grid for GridSearchCV
            param_grid = {}
            
            # Handle hidden_layer_sizes
            param_grid['hidden_layer_sizes'] = self.hidden_layer_sizes_range
            
            # Handle learning_rate_init
            if isinstance(self.learning_rate_init_range, list):
                param_grid['learning_rate_init'] = self.learning_rate_init_range
            else:
                # Sample from distribution for grid search
                n_samples = min(5, self.n_iter_search)
                samples = self.learning_rate_init_range.rvs(size=n_samples, random_state=self.random_state)
                param_grid['learning_rate_init'] = np.unique(samples)
            
            # Handle activation
            param_grid['activation'] = self.activation_options
            
            # Handle solver
            param_grid['solver'] = self.solver_options
            
            # Handle alpha
            if isinstance(self.alpha_range, list):
                param_grid['alpha'] = self.alpha_range
            else:
                # Sample from distribution for grid search
                n_samples = min(5, self.n_iter_search)
                samples = self.alpha_range.rvs(size=n_samples, random_state=self.random_state)
                param_grid['alpha'] = np.unique(samples)
            
            # Initialize MLPRegressor base model
            model = MLPRegressor(max_iter=self.max_iter, random_state=self.random_state)
            
            # Grid search
            grid_search = GridSearchCV(
                model, param_grid=param_grid,
                cv=self.cv_folds, scoring='neg_mean_squared_error',
                error_score=np.nan, n_jobs=-1
            )
            grid_search.fit(X_train_clean, y_train_clean)
            best_params = grid_search.best_params_
            
        elif self.search_method == 'random':
            # Prepare parameter distributions for RandomizedSearchCV
            param_dist = {}
            
            # Handle hidden_layer_sizes
            param_dist['hidden_layer_sizes'] = self.hidden_layer_sizes_range
            
            # Handle learning_rate_init
            param_dist['learning_rate_init'] = self.learning_rate_init_range
            
            # Handle activation
            param_dist['activation'] = self.activation_options
            
            # Handle solver
            param_dist['solver'] = self.solver_options
            
            # Handle alpha
            param_dist['alpha'] = self.alpha_range
            
            # Initialize MLPRegressor base model
            model = MLPRegressor(max_iter=self.max_iter, random_state=self.random_state)
            
            # Randomized search
            random_search = RandomizedSearchCV(
                model, param_distributions=param_dist, n_iter=self.n_iter_search,
                cv=self.cv_folds, scoring='neg_mean_squared_error',
                random_state=self.random_state, error_score=np.nan, n_jobs=-1
            )
            random_search.fit(X_train_clean, y_train_clean)
            best_params = random_search.best_params_
            
        elif self.search_method == 'bayesian':
            # Bayesian optimization with Optuna
            study = optuna.create_study(
                direction='maximize',  # We're maximizing negative MSE
                sampler=optuna.samplers.TPESampler(seed=self.random_state),
                pruner=optuna.pruners.MedianPruner(n_startup_trials=5)
            )
            
            # Create objective function with data
            objective_with_data = lambda trial: self._objective(trial, X_train_clean, y_train_clean)
            
            # Optimize
            study.optimize(
                objective_with_data,
                n_trials=self.n_iter_search,
                timeout=self.optuna_timeout,
                n_jobs=self.optuna_n_jobs
            )
            
            # Extract best parameters
            best_params = study.best_params
            
            # Convert Optuna's best_params to scikit-learn format
            sklearn_params = {
                'hidden_layer_sizes': best_params['hidden_layer_sizes'],
                'learning_rate_init': best_params['learning_rate_init'],
                'activation': best_params['activation'],
                'solver': best_params['solver'],
                'alpha': best_params['alpha']
            }
            best_params = sklearn_params
            
        else:
            raise ValueError(f"Unknown search_method: {self.search_method}. Choose from 'grid', 'random', or 'bayesian'.")

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
            learning_rate_init=best_params['learning_rate_init'],
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
                    norm.ppf(0.33, loc=loc, scale=scale),
                    norm.ppf(0.67, loc=loc, scale=scale),
                )
            elif code == 2:
                return (
                    lognorm.ppf(0.33, s=shape, loc=loc, scale=scale),
                    lognorm.ppf(0.67, s=shape, loc=loc, scale=scale),
                )
            elif code == 3:
                return (
                    expon.ppf(0.33, loc=loc, scale=scale),
                    expon.ppf(0.67, loc=loc, scale=scale),
                )
            elif code == 4:
                return (
                    gamma.ppf(0.33, a=shape, loc=loc, scale=scale),
                    gamma.ppf(0.67, a=shape, loc=loc, scale=scale),
                )
            elif code == 5:
                return (
                    weibull_min.ppf(0.33, c=shape, loc=loc, scale=scale),
                    weibull_min.ppf(0.67, c=shape, loc=loc, scale=scale),
                )
            elif code == 6:
                # Note: Renamed 't_dist' to 't' for standard scipy.stats
                return (
                    t.ppf(0.33, df=shape, loc=loc, scale=scale),
                    t.ppf(0.67, df=shape, loc=loc, scale=scale),
                )
            elif code == 7:
                # Poisson: poisson.ppf(q, mu, loc=0)
                # ASSUMPTION: 'mu' (mean) is passed as 'shape'
                #             'loc' is passed as 'loc'
                #             'scale' is unused
                return (
                    poisson.ppf(0.33, mu=shape, loc=loc),
                    poisson.ppf(0.67, mu=shape, loc=loc),
                )
            elif code == 8:
                # Negative Binomial: nbinom.ppf(q, n, p, loc=0)
                # ASSUMPTION: 'n' (successes) is passed as 'shape'
                #             'p' (probability) is passed as 'scale'
                #             'loc' is passed as 'loc'
                return (
                    nbinom.ppf(0.33, n=shape, p=scale, loc=loc),
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
        terciles_emp = clim.quantile([0.33, 0.67], dim="T")
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


    def forecast(self, Predictant, clim_year_start, clim_year_end, hindcast_det, hindcast_det_cross, Predictor_for_year, best_params=None, best_code_da=None, best_shape_da=None, best_loc_da=None, best_scale_da=None):
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
            learning_rate_init=best_params['learning_rate_init'],
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
        T1_emp = terciles.isel(quantile=0).drop_vars('quantile')
        T2_emp = terciles.isel(quantile=1).drop_vars('quantile')
        error_variance = (Predictant_no_m - hindcast_det_cross).var(dim='T')
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
                result_da,
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
                result_da,
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


class WAS_mme_MLP:
    """
    Multi-Layer Perceptron (MLP) for Multi-Model Ensemble (MME) forecasting.
    This class implements a Multi-Layer Perceptron model using scikit-learn's MLPRegressor
    for deterministic forecasting, with optional tercile probability calculations using
    various statistical distributions. Implements multiple hyperparameter optimization methods.
    
    Parameters
    ----------
    search_method : str, optional
        Hyperparameter optimization method: 'grid', 'random', or 'bayesian' (default: 'random').
    hidden_layer_sizes_range : list of tuples, optional
        List of hidden layer sizes to tune, e.g., [(10,), (10, 5), (20, 10)] (default).
    activation_options : list of str, optional
        Activation functions to tune ('identity', 'logistic', 'tanh', 'relu') (default is ['relu', 'tanh', 'logistic']).
    solver_options : list of str, optional
        Solvers to tune ('lbfgs', 'sgd', 'adam') (default is ['adam', 'sgd', 'lbfgs']).
    alpha_range : list of float, optional
        L2 regularization parameters to tune (default is [0.0001, 0.001, 0.01, 0.1]).
    learning_rate_init_range : list or scipy.stats distribution, optional
        Learning rate initialization range for random/bayesian search (default is loguniform(0.0001, 0.01)).
        For grid search, provide a list of values.
    max_iter : int, optional
        Maximum iterations (default is 200).
    random_state : int, optional
        Seed for reproducibility (default is 42).
    dist_method : str, optional
        Distribution method for tercile probabilities ('bestfit', 'nonparam') (default is 'nonparam').
    n_iter_search : int, optional
        Number of iterations for randomized/bayesian search (default is 10).
    cv_folds : int, optional
        Number of cross-validation folds (default is 3).
    n_clusters : int, optional
        Number of clusters for homogenized zones (default is 4).
    optuna_n_jobs : int, optional
        Number of parallel jobs for Optuna (default is 1).
    optuna_timeout : int, optional
        Timeout in seconds for Optuna optimization (default is None).
    """
    def __init__(self,
                 search_method='random',
                 hidden_layer_sizes_range=[(10,), (10, 5), (20, 10)],
                 activation_options=['relu', 'tanh', 'logistic'],
                 learning_rate_init_range=loguniform(0.0001, 0.01),
                 solver_options=['adam', 'sgd', 'lbfgs'],
                 alpha_range=[0.0001, 0.001, 0.01, 0.1],
                 max_iter=200, 
                 random_state=42, 
                 dist_method="nonparam",
                 n_iter_search=10, 
                 cv_folds=3, 
                 n_clusters=4,
                 optuna_n_jobs=1,
                 optuna_timeout=None):
        
        self.search_method = search_method
        self.hidden_layer_sizes_range = hidden_layer_sizes_range
        self.learning_rate_init_range = learning_rate_init_range
        self.activation_options = activation_options
        self.solver_options = solver_options
        self.alpha_range = alpha_range
        self.max_iter = max_iter
        self.random_state = random_state
        self.dist_method = dist_method
        self.n_iter_search = n_iter_search
        self.cv_folds = cv_folds
        self.n_clusters = n_clusters
        self.optuna_n_jobs = optuna_n_jobs
        self.optuna_timeout = optuna_timeout
        self.mlp = None

    def _objective(self, trial, X_train, y_train):
        """
        Objective function for Optuna optimization.
        """
        # Define hyperparameter search space
        hidden_layer_sizes = trial.suggest_categorical(
            'hidden_layer_sizes', self.hidden_layer_sizes_range
        )
        activation = trial.suggest_categorical(
            'activation', self.activation_options
        )
        solver = trial.suggest_categorical(
            'solver', self.solver_options
        )
        
        # Handle learning_rate_init based on search method
        if isinstance(self.learning_rate_init_range, list):
            learning_rate_init = trial.suggest_categorical(
                'learning_rate_init', self.learning_rate_init_range
            )
        else:
            # Assume it's a distribution
            learning_rate_init = trial.suggest_float(
                'learning_rate_init', 
                self.learning_rate_init_range.a,
                self.learning_rate_init_range.b,
                log=True
            )
        
        # Handle alpha based on search method
        if isinstance(self.alpha_range, list):
            alpha = trial.suggest_categorical('alpha', self.alpha_range)
        else:
            # Assume it's a distribution
            alpha = trial.suggest_float(
                'alpha', 
                self.alpha_range.a,
                self.alpha_range.b,
                log=True
            )
        
        # Create and train model
        model = MLPRegressor(
            hidden_layer_sizes=hidden_layer_sizes,
            learning_rate_init=learning_rate_init,
            activation=activation,
            solver=solver,
            alpha=alpha,
            max_iter=self.max_iter,
            random_state=self.random_state
        )
        
        # Use cross-validation
        from sklearn.model_selection import cross_val_score
        scores = cross_val_score(
            model, X_train, y_train, 
            cv=self.cv_folds, 
            scoring='neg_mean_squared_error',
            n_jobs=-1
        )
        
        return np.mean(scores)

    def compute_hyperparameters(self, Predictors, Predictand, clim_year_start, clim_year_end):
        """
        Independently computes the best hyperparameters using selected optimization method
        on stacked training data for each homogenized zone.
        
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
            
            if self.search_method == 'grid':
                # Prepare parameter grid for GridSearchCV
                param_grid = {}
                
                # Handle hidden_layer_sizes
                param_grid['hidden_layer_sizes'] = self.hidden_layer_sizes_range
                
                # Handle learning_rate_init
                if isinstance(self.learning_rate_init_range, list):
                    param_grid['learning_rate_init'] = self.learning_rate_init_range
                else:
                    # Sample from distribution for grid search
                    n_samples = min(5, self.n_iter_search)
                    samples = self.learning_rate_init_range.rvs(size=n_samples, random_state=self.random_state)
                    param_grid['learning_rate_init'] = np.unique(samples)
                
                # Handle activation
                param_grid['activation'] = self.activation_options
                
                # Handle solver
                param_grid['solver'] = self.solver_options
                
                # Handle alpha
                if isinstance(self.alpha_range, list):
                    param_grid['alpha'] = self.alpha_range
                else:
                    # Sample from distribution for grid search
                    n_samples = min(5, self.n_iter_search)
                    samples = self.alpha_range.rvs(size=n_samples, random_state=self.random_state)
                    param_grid['alpha'] = np.unique(samples)
                
                # Initialize MLPRegressor base model
                model = MLPRegressor(max_iter=self.max_iter, random_state=self.random_state)
                
                # Grid search
                grid_search = GridSearchCV(
                    model, param_grid=param_grid,
                    cv=self.cv_folds, scoring='neg_mean_squared_error',
                    error_score=np.nan, n_jobs=-1
                )
                grid_search.fit(X_clean_c, y_clean_c)
                best_params_dict[c] = grid_search.best_params_
                
            elif self.search_method == 'random':
                # Prepare parameter distributions for RandomizedSearchCV
                param_dist = {}
                
                # Handle hidden_layer_sizes
                param_dist['hidden_layer_sizes'] = self.hidden_layer_sizes_range
                
                # Handle learning_rate_init
                param_dist['learning_rate_init'] = self.learning_rate_init_range
                
                # Handle activation
                param_dist['activation'] = self.activation_options
                
                # Handle solver
                param_dist['solver'] = self.solver_options
                
                # Handle alpha
                param_dist['alpha'] = self.alpha_range
                
                # Initialize MLPRegressor base model
                model = MLPRegressor(max_iter=self.max_iter, random_state=self.random_state)
                
                # Randomized search
                random_search = RandomizedSearchCV(
                    model, param_distributions=param_dist, n_iter=self.n_iter_search,
                    cv=self.cv_folds, scoring='neg_mean_squared_error',
                    random_state=self.random_state, error_score=np.nan, n_jobs=-1
                )
                random_search.fit(X_clean_c, y_clean_c)
                best_params_dict[c] = random_search.best_params_
                
            elif self.search_method == 'bayesian':
                # Bayesian optimization with Optuna
                study = optuna.create_study(
                    direction='maximize',  # We're maximizing negative MSE
                    sampler=optuna.samplers.TPESampler(seed=self.random_state),
                    pruner=optuna.pruners.MedianPruner(n_startup_trials=5)
                )
                
                # Create objective function with data
                objective_with_data = lambda trial: self._objective(trial, X_clean_c, y_clean_c)
                
                # Optimize
                study.optimize(
                    objective_with_data,
                    n_trials=self.n_iter_search,
                    timeout=self.optuna_timeout,
                    n_jobs=self.optuna_n_jobs
                )
                
                # Extract best parameters
                best_params = study.best_params
                
                # Convert Optuna's best_params to scikit-learn format
                sklearn_params = {
                    'hidden_layer_sizes': best_params['hidden_layer_sizes'],
                    'learning_rate_init': best_params['learning_rate_init'],
                    'activation': best_params['activation'],
                    'solver': best_params['solver'],
                    'alpha': best_params['alpha']
                }
                best_params_dict[c] = sklearn_params
                
            else:
                raise ValueError(f"Unknown search_method: {self.search_method}. Choose from 'grid', 'random', or 'bayesian'.")
        
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
                learning_rate_init=bp['learning_rate_init'],
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
                    norm.ppf(0.33, loc=loc, scale=scale),
                    norm.ppf(0.67, loc=loc, scale=scale),
                )
            elif code == 2:
                return (
                    lognorm.ppf(0.33, s=shape, loc=loc, scale=scale),
                    lognorm.ppf(0.67, s=shape, loc=loc, scale=scale),
                )
            elif code == 3:
                return (
                    expon.ppf(0.33, loc=loc, scale=scale),
                    expon.ppf(0.67, loc=loc, scale=scale),
                )
            elif code == 4:
                return (
                    gamma.ppf(0.33, a=shape, loc=loc, scale=scale),
                    gamma.ppf(0.67, a=shape, loc=loc, scale=scale),
                )
            elif code == 5:
                return (
                    weibull_min.ppf(0.33, c=shape, loc=loc, scale=scale),
                    weibull_min.ppf(0.67, c=shape, loc=loc, scale=scale),
                )
            elif code == 6:
                # Note: Renamed 't_dist' to 't' for standard scipy.stats
                return (
                    t.ppf(0.33, df=shape, loc=loc, scale=scale),
                    t.ppf(0.67, df=shape, loc=loc, scale=scale),
                )
            elif code == 7:
                # Poisson: poisson.ppf(q, mu, loc=0)
                # ASSUMPTION: 'mu' (mean) is passed as 'shape'
                #             'loc' is passed as 'loc'
                #             'scale' is unused
                return (
                    poisson.ppf(0.33, mu=shape, loc=loc),
                    poisson.ppf(0.67, mu=shape, loc=loc),
                )
            elif code == 8:
                # Negative Binomial: nbinom.ppf(q, n, p, loc=0)
                # ASSUMPTION: 'n' (successes) is passed as 'shape'
                #             'p' (probability) is passed as 'scale'
                #             'loc' is passed as 'loc'
                return (
                    nbinom.ppf(0.33, n=shape, p=scale, loc=loc),
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
        terciles_emp = clim.quantile([0.33, 0.67], dim="T")
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

    def forecast(self, Predictant, clim_year_start, clim_year_end, hindcast_det, hindcast_det_cross, Predictor_for_year, best_params=None, cluster_da=None, best_code_da=None, best_shape_da=None, best_loc_da=None, best_scale_da=None):
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
                learning_rate_init=bp['learning_rate_init'],
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
        T1_emp = terciles.isel(quantile=0).drop_vars('quantile')
        T2_emp = terciles.isel(quantile=1).drop_vars('quantile')
        error_variance = (Predictant_no_m - hindcast_det_cross).var(dim='T')
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
                result_da,
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
                result_da,
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



class WAS_mme_XGBoosting_:
    """
    XGBoost-based Multi-Model Ensemble (MME) forecasting.
    This class implements a single-model forecasting approach using XGBoost's XGBRegressor
    for deterministic predictions, with optional tercile probability calculations using
    various statistical distributions. Implements multiple hyperparameter optimization methods.
    
    Parameters
    ----------
    search_method : str, optional
        Hyperparameter optimization method: 'grid', 'random', or 'bayesian' (default: 'random').
    n_estimators_range : list of int or scipy.stats distribution, optional
        List of n_estimators values to tune (default is [50, 100, 200, 300]).
        Can be a list for grid search or a distribution for random/bayesian search.
    learning_rate_range : list of float or scipy.stats distribution, optional
        List of learning rates to tune (default is [0.01, 0.05, 0.1, 0.2]).
        Can be a list for grid search or a distribution for random/bayesian search.
    max_depth_range : list of int or scipy.stats distribution, optional
        List of max depths to tune (default is [3, 5, 7, 9]).
        Can be a list for grid search or a distribution for random/bayesian search.
    min_child_weight_range : list of float or scipy.stats distribution, optional
        List of minimum child weights to tune (default is [1, 3, 5]).
        Can be a list for grid search or a distribution for random/bayesian search.
    subsample_range : list of float or scipy.stats distribution, optional
        List of subsample ratios to tune (default is [0.6, 0.8, 1.0]).
        Can be a list for grid search or a distribution for random/bayesian search.
    colsample_bytree_range : list of float or scipy.stats distribution, optional
        List of column sampling ratios to tune (default is [0.6, 0.8, 1.0]).
        Can be a list for grid search or a distribution for random/bayesian search.
    random_state : int, optional
        Seed for reproducibility (default is 42).
    dist_method : str, optional
        Distribution method for tercile probabilities ('bestfit', 'nonparam', etc.) (default is 'nonparam').
    n_iter_search : int, optional
        Number of iterations for randomized/bayesian search or points to sample for grid search (default is 10).
    cv_folds : int, optional
        Number of cross-validation folds (default is 3).
    optuna_n_jobs : int, optional
        Number of parallel jobs for Optuna (default is 1).
    optuna_timeout : int, optional
        Timeout in seconds for Optuna optimization (default is None).
    """
    def __init__(self,
                 search_method='random',
                 n_estimators_range=[50, 100, 200, 300],
                 learning_rate_range=[0.01, 0.05, 0.1, 0.2],
                 max_depth_range=[3, 5, 7, 9],
                 min_child_weight_range=[1, 3, 5],
                 subsample_range=[0.6, 0.8, 1.0],
                 colsample_bytree_range=[0.6, 0.8, 1.0],
                 random_state=42,
                 dist_method="nonparam",
                 n_iter_search=10,
                 cv_folds=3,
                 optuna_n_jobs=1,
                 optuna_timeout=None):
        
        self.search_method = search_method
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
        self.optuna_n_jobs = optuna_n_jobs
        self.optuna_timeout = optuna_timeout
        self.xgb = None

    def _objective(self, trial, X_train, y_train):
        """
        Objective function for Optuna optimization.
        """
        # Define hyperparameter search space
        # Handle n_estimators
        if isinstance(self.n_estimators_range, list):
            n_estimators = trial.suggest_categorical('n_estimators', self.n_estimators_range)
        else:
            # Assume it's a distribution
            n_estimators = trial.suggest_int(
                'n_estimators', 
                int(self.n_estimators_range.a),
                int(self.n_estimators_range.b)
            )
        
        # Handle learning_rate
        if isinstance(self.learning_rate_range, list):
            learning_rate = trial.suggest_categorical('learning_rate', self.learning_rate_range)
        else:
            # Assume it's a distribution
            learning_rate = trial.suggest_float(
                'learning_rate', 
                self.learning_rate_range.a,
                self.learning_rate_range.b,
                log=True
            )
        
        # Handle max_depth
        if isinstance(self.max_depth_range, list):
            max_depth = trial.suggest_categorical('max_depth', self.max_depth_range)
        else:
            # Assume it's a distribution
            max_depth = trial.suggest_int(
                'max_depth', 
                int(self.max_depth_range.a),
                int(self.max_depth_range.b)
            )
        
        # Handle min_child_weight
        if isinstance(self.min_child_weight_range, list):
            min_child_weight = trial.suggest_categorical('min_child_weight', self.min_child_weight_range)
        else:
            # Assume it's a distribution
            min_child_weight = trial.suggest_float(
                'min_child_weight', 
                self.min_child_weight_range.a,
                self.min_child_weight_range.b
            )
        
        # Handle subsample
        if isinstance(self.subsample_range, list):
            subsample = trial.suggest_categorical('subsample', self.subsample_range)
        else:
            # Assume it's a distribution
            subsample = trial.suggest_float(
                'subsample', 
                self.subsample_range.a,
                self.subsample_range.b
            )
        
        # Handle colsample_bytree
        if isinstance(self.colsample_bytree_range, list):
            colsample_bytree = trial.suggest_categorical('colsample_bytree', self.colsample_bytree_range)
        else:
            # Assume it's a distribution
            colsample_bytree = trial.suggest_float(
                'colsample_bytree', 
                self.colsample_bytree_range.a,
                self.colsample_bytree_range.b
            )
        
        # Create and train model
        model = XGBRegressor(
            n_estimators=n_estimators,
            learning_rate=learning_rate,
            max_depth=max_depth,
            min_child_weight=min_child_weight,
            subsample=subsample,
            colsample_bytree=colsample_bytree,
            random_state=self.random_state,
            verbosity=0,
            n_jobs=-1
        )
        
        # Use cross-validation
        from sklearn.model_selection import cross_val_score
        scores = cross_val_score(
            model, X_train, y_train, 
            cv=self.cv_folds, 
            scoring='neg_mean_squared_error',
            n_jobs=-1
        )
        
        return np.mean(scores)

    def compute_hyperparameters(self, Predictors, Predictand, clim_year_start, clim_year_end):
        """
        Independently computes the best hyperparameters using selected optimization method
        on stacked training data.

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
        X_train = Predictors
        y_train = Predictand

        # Stack training data
        X_train_stacked = X_train.stack(sample=('T', 'Y', 'X')).transpose('sample', 'M').values
        y_train_stacked = y_train.stack(sample=('T', 'Y', 'X')).values.ravel()
        train_nan_mask = np.any(~np.isfinite(X_train_stacked), axis=1) | ~np.isfinite(y_train_stacked)
        X_train_clean = X_train_stacked[~train_nan_mask]
        y_train_clean = y_train_stacked[~train_nan_mask]

        if self.search_method == 'grid':
            # Prepare parameter grid for GridSearchCV
            param_grid = {}
            
            # Handle n_estimators
            if isinstance(self.n_estimators_range, list):
                param_grid['n_estimators'] = self.n_estimators_range
            else:
                # Sample from distribution for grid search
                n_samples = min(5, self.n_iter_search)
                samples = self.n_estimators_range.rvs(size=n_samples, random_state=self.random_state)
                param_grid['n_estimators'] = np.unique(samples.astype(int))
            
            # Handle learning_rate
            if isinstance(self.learning_rate_range, list):
                param_grid['learning_rate'] = self.learning_rate_range
            else:
                # Sample from distribution for grid search
                n_samples = min(5, self.n_iter_search)
                samples = self.learning_rate_range.rvs(size=n_samples, random_state=self.random_state)
                param_grid['learning_rate'] = np.unique(samples)
            
            # Handle max_depth
            if isinstance(self.max_depth_range, list):
                param_grid['max_depth'] = self.max_depth_range
            else:
                # Sample from distribution for grid search
                n_samples = min(5, self.n_iter_search)
                samples = self.max_depth_range.rvs(size=n_samples, random_state=self.random_state)
                param_grid['max_depth'] = np.unique(samples.astype(int))
            
            # Handle min_child_weight
            if isinstance(self.min_child_weight_range, list):
                param_grid['min_child_weight'] = self.min_child_weight_range
            else:
                # Sample from distribution for grid search
                n_samples = min(5, self.n_iter_search)
                samples = self.min_child_weight_range.rvs(size=n_samples, random_state=self.random_state)
                param_grid['min_child_weight'] = np.unique(samples)
            
            # Handle subsample
            if isinstance(self.subsample_range, list):
                param_grid['subsample'] = self.subsample_range
            else:
                # Sample from distribution for grid search
                n_samples = min(5, self.n_iter_search)
                samples = self.subsample_range.rvs(size=n_samples, random_state=self.random_state)
                param_grid['subsample'] = np.unique(samples)
            
            # Handle colsample_bytree
            if isinstance(self.colsample_bytree_range, list):
                param_grid['colsample_bytree'] = self.colsample_bytree_range
            else:
                # Sample from distribution for grid search
                n_samples = min(5, self.n_iter_search)
                samples = self.colsample_bytree_range.rvs(size=n_samples, random_state=self.random_state)
                param_grid['colsample_bytree'] = np.unique(samples)
            
            # Initialize XGBRegressor base model
            model = XGBRegressor(random_state=self.random_state, verbosity=0, n_jobs=-1)
            
            # Grid search
            grid_search = GridSearchCV(
                model, param_grid=param_grid,
                cv=self.cv_folds, scoring='neg_mean_squared_error',
                error_score=np.nan, n_jobs=-1
            )
            grid_search.fit(X_train_clean, y_train_clean)
            best_params = grid_search.best_params_
            
        elif self.search_method == 'random':
            # Prepare parameter distributions for RandomizedSearchCV
            param_dist = {}
            
            # Handle n_estimators
            param_dist['n_estimators'] = self.n_estimators_range
            
            # Handle learning_rate
            param_dist['learning_rate'] = self.learning_rate_range
            
            # Handle max_depth
            param_dist['max_depth'] = self.max_depth_range
            
            # Handle min_child_weight
            param_dist['min_child_weight'] = self.min_child_weight_range
            
            # Handle subsample
            param_dist['subsample'] = self.subsample_range
            
            # Handle colsample_bytree
            param_dist['colsample_bytree'] = self.colsample_bytree_range
            
            # Initialize XGBRegressor base model
            model = XGBRegressor(random_state=self.random_state, verbosity=0, n_jobs=-1)
            
            # Randomized search
            random_search = RandomizedSearchCV(
                model, param_distributions=param_dist, n_iter=self.n_iter_search,
                cv=self.cv_folds, scoring='neg_mean_squared_error',
                random_state=self.random_state, error_score=np.nan, n_jobs=-1
            )
            random_search.fit(X_train_clean, y_train_clean)
            best_params = random_search.best_params_
            
        elif self.search_method == 'bayesian':
            # Bayesian optimization with Optuna
            study = optuna.create_study(
                direction='maximize',  # We're maximizing negative MSE
                sampler=optuna.samplers.TPESampler(seed=self.random_state),
                pruner=optuna.pruners.MedianPruner(n_startup_trials=5)
            )
            
            # Create objective function with data
            objective_with_data = lambda trial: self._objective(trial, X_train_clean, y_train_clean)
            
            # Optimize
            study.optimize(
                objective_with_data,
                n_trials=self.n_iter_search,
                timeout=self.optuna_timeout,
                n_jobs=self.optuna_n_jobs
            )
            
            # Extract best parameters
            best_params = study.best_params
            
            # Convert Optuna's best_params to scikit-learn format
            sklearn_params = {
                'n_estimators': best_params['n_estimators'],
                'learning_rate': best_params['learning_rate'],
                'max_depth': best_params['max_depth'],
                'min_child_weight': best_params['min_child_weight'],
                'subsample': best_params['subsample'],
                'colsample_bytree': best_params['colsample_bytree']
            }
            best_params = sklearn_params
            
        else:
            raise ValueError(f"Unknown search_method: {self.search_method}. Choose from 'grid', 'random', or 'bayesian'.")

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
            verbosity=0,
            n_jobs=-1
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
                    norm.ppf(0.33, loc=loc, scale=scale),
                    norm.ppf(0.67, loc=loc, scale=scale),
                )
            elif code == 2:
                return (
                    lognorm.ppf(0.33, s=shape, loc=loc, scale=scale),
                    lognorm.ppf(0.67, s=shape, loc=loc, scale=scale),
                )
            elif code == 3:
                return (
                    expon.ppf(0.33, loc=loc, scale=scale),
                    expon.ppf(0.67, loc=loc, scale=scale),
                )
            elif code == 4:
                return (
                    gamma.ppf(0.33, a=shape, loc=loc, scale=scale),
                    gamma.ppf(0.67, a=shape, loc=loc, scale=scale),
                )
            elif code == 5:
                return (
                    weibull_min.ppf(0.33, c=shape, loc=loc, scale=scale),
                    weibull_min.ppf(0.67, c=shape, loc=loc, scale=scale),
                )
            elif code == 6:
                # Note: Renamed 't_dist' to 't' for standard scipy.stats
                return (
                    t.ppf(0.33, df=shape, loc=loc, scale=scale),
                    t.ppf(0.67, df=shape, loc=loc, scale=scale),
                )
            elif code == 7:
                # Poisson: poisson.ppf(q, mu, loc=0)
                # ASSUMPTION: 'mu' (mean) is passed as 'shape'
                #             'loc' is passed as 'loc'
                #             'scale' is unused
                return (
                    poisson.ppf(0.33, mu=shape, loc=loc),
                    poisson.ppf(0.67, mu=shape, loc=loc),
                )
            elif code == 8:
                # Negative Binomial: nbinom.ppf(q, n, p, loc=0)
                # ASSUMPTION: 'n' (successes) is passed as 'shape'
                #             'p' (probability) is passed as 'scale'
                #             'loc' is passed as 'loc'
                return (
                    nbinom.ppf(0.33, n=shape, p=scale, loc=loc),
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
        terciles_emp = clim.quantile([0.33, 0.67], dim="T")
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
                 hindcast_det_cross, Predictor_for_year, best_params=None, best_code_da=None, best_shape_da=None, best_loc_da=None, best_scale_da=None):
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
        # Predictor_for_year_st = (Predictor_for_year - mean_val) / std_val
        
        Predictor_for_year_st = Predictor_for_year

        # hindcast_det_st = standardize_timeseries(hindcast_det, clim_year_start, clim_year_end)
        
        hindcast_det_st = hindcast_det
        
        # Predictant_st = standardize_timeseries(Predictant_no_m, clim_year_start, clim_year_end)
        Predictant_st = Predictant_no_m
        
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

        # result_da = reverse_standardize(result_da, Predictant_no_m, clim_year_start, clim_year_end)

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
        T1_emp = terciles.isel(quantile=0).drop_vars('quantile')
        T2_emp = terciles.isel(quantile=1).drop_vars('quantile')
        error_variance = (Predictant_no_m - hindcast_det_cross).var(dim='T')
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
                result_da,
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
                result_da,
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


class WAS_mme_XGBoosting:
    """
    XGBoost-based Multi-Model Ensemble (MME) forecasting.
    This class implements a single-model forecasting approach using XGBoost's XGBRegressor
    for deterministic predictions, with optional tercile probability calculations using
    various statistical distributions. Implements multiple hyperparameter optimization methods.
    
    Parameters
    ----------
    search_method : str, optional
        Hyperparameter optimization method: 'grid', 'random', or 'bayesian' (default: 'random').
    n_estimators_range : list of int or scipy.stats distribution, optional
        List of n_estimators values to tune (default is [50, 100, 200, 300]).
        Can be a list for grid search or a distribution for random/bayesian search.
    learning_rate_range : list of float or scipy.stats distribution, optional
        List of learning rates to tune (default is [0.01, 0.05, 0.1, 0.2]).
        Can be a list for grid search or a distribution for random/bayesian search.
    max_depth_range : list of int or scipy.stats distribution, optional
        List of max depths to tune (default is [3, 5, 7, 9]).
        Can be a list for grid search or a distribution for random/bayesian search.
    min_child_weight_range : list of float or scipy.stats distribution, optional
        List of minimum child weights to tune (default is [1, 3, 5]).
        Can be a list for grid search or a distribution for random/bayesian search.
    subsample_range : list of float or scipy.stats distribution, optional
        List of subsample ratios to tune (default is [0.6, 0.8, 1.0]).
        Can be a list for grid search or a distribution for random/bayesian search.
    colsample_bytree_range : list of float or scipy.stats distribution, optional
        List of column sampling ratios to tune (default is [0.6, 0.8, 1.0]).
        Can be a list for grid search or a distribution for random/bayesian search.
    random_state : int, optional
        Seed for reproducibility (default is 42).
    dist_method : str, optional
        Distribution method for tercile probabilities ('bestfit', 'nonparam', etc.) (default is 'nonparam').
    n_iter_search : int, optional
        Number of iterations for randomized/bayesian search or points to sample for grid search (default is 10).
    cv_folds : int, optional
        Number of cross-validation folds (default is 3).
    n_clusters : int, optional
        Number of clusters for homogenized zones (default is 4).
    optuna_n_jobs : int, optional
        Number of parallel jobs for Optuna (default is 1).
    optuna_timeout : int, optional
        Timeout in seconds for Optuna optimization (default is None).
    """
    def __init__(self,
                 search_method='random',
                 n_estimators_range=[50, 100, 200, 300],
                 learning_rate_range=[0.01, 0.05, 0.1, 0.2],
                 max_depth_range=[3, 5, 7, 9],
                 min_child_weight_range=[1, 3, 5],
                 subsample_range=[0.6, 0.8, 1.0],
                 colsample_bytree_range=[0.6, 0.8, 1.0],
                 random_state=42,
                 dist_method="nonparam",
                 n_iter_search=10,
                 cv_folds=3,
                 n_clusters=4,
                 optuna_n_jobs=1,
                 optuna_timeout=None):
        
        self.search_method = search_method
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
        self.optuna_n_jobs = optuna_n_jobs
        self.optuna_timeout = optuna_timeout
        self.xgb = None

    def _objective(self, trial, X_train, y_train):
        """
        Objective function for Optuna optimization.
        """
        # Define hyperparameter search space
        # Handle n_estimators
        if isinstance(self.n_estimators_range, list):
            n_estimators = trial.suggest_categorical('n_estimators', self.n_estimators_range)
        else:
            # Assume it's a distribution
            n_estimators = trial.suggest_int(
                'n_estimators', 
                int(self.n_estimators_range.a),
                int(self.n_estimators_range.b)
            )
        
        # Handle learning_rate
        if isinstance(self.learning_rate_range, list):
            learning_rate = trial.suggest_categorical('learning_rate', self.learning_rate_range)
        else:
            # Assume it's a distribution
            learning_rate = trial.suggest_float(
                'learning_rate', 
                self.learning_rate_range.a,
                self.learning_rate_range.b,
                log=True
            )
        
        # Handle max_depth
        if isinstance(self.max_depth_range, list):
            max_depth = trial.suggest_categorical('max_depth', self.max_depth_range)
        else:
            # Assume it's a distribution
            max_depth = trial.suggest_int(
                'max_depth', 
                int(self.max_depth_range.a),
                int(self.max_depth_range.b)
            )
        
        # Handle min_child_weight
        if isinstance(self.min_child_weight_range, list):
            min_child_weight = trial.suggest_categorical('min_child_weight', self.min_child_weight_range)
        else:
            # Assume it's a distribution
            min_child_weight = trial.suggest_float(
                'min_child_weight', 
                self.min_child_weight_range.a,
                self.min_child_weight_range.b
            )
        
        # Handle subsample
        if isinstance(self.subsample_range, list):
            subsample = trial.suggest_categorical('subsample', self.subsample_range)
        else:
            # Assume it's a distribution
            subsample = trial.suggest_float(
                'subsample', 
                self.subsample_range.a,
                self.subsample_range.b
            )
        
        # Handle colsample_bytree
        if isinstance(self.colsample_bytree_range, list):
            colsample_bytree = trial.suggest_categorical('colsample_bytree', self.colsample_bytree_range)
        else:
            # Assume it's a distribution
            colsample_bytree = trial.suggest_float(
                'colsample_bytree', 
                self.colsample_bytree_range.a,
                self.colsample_bytree_range.b
            )
        
        # Create and train model
        model = XGBRegressor(
            n_estimators=n_estimators,
            learning_rate=learning_rate,
            max_depth=max_depth,
            min_child_weight=min_child_weight,
            subsample=subsample,
            colsample_bytree=colsample_bytree,
            random_state=self.random_state,
            verbosity=0,
            n_jobs=-1
        )
        
        # Use cross-validation
        from sklearn.model_selection import cross_val_score
        scores = cross_val_score(
            model, X_train, y_train, 
            cv=self.cv_folds, 
            scoring='neg_mean_squared_error',
            n_jobs=-1
        )
        
        return np.mean(scores)

    def compute_hyperparameters(self, Predictors, Predictand, clim_year_start, clim_year_end):
        """
        Independently computes the best hyperparameters using selected optimization method
        on stacked training data for each homogenized zone.
        
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
        
        X_train_std = Predictors
        
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
        y_train_std = Predictand
        
        X_train_std['T'] = y_train_std['T']
        
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
            
            if self.search_method == 'grid':
                # Prepare parameter grid for GridSearchCV
                param_grid = {}
                
                # Handle n_estimators
                if isinstance(self.n_estimators_range, list):
                    param_grid['n_estimators'] = self.n_estimators_range
                else:
                    # Sample from distribution for grid search
                    n_samples = min(5, self.n_iter_search)
                    samples = self.n_estimators_range.rvs(size=n_samples, random_state=self.random_state)
                    param_grid['n_estimators'] = np.unique(samples.astype(int))
                
                # Handle learning_rate
                if isinstance(self.learning_rate_range, list):
                    param_grid['learning_rate'] = self.learning_rate_range
                else:
                    # Sample from distribution for grid search
                    n_samples = min(5, self.n_iter_search)
                    samples = self.learning_rate_range.rvs(size=n_samples, random_state=self.random_state)
                    param_grid['learning_rate'] = np.unique(samples)
                
                # Handle max_depth
                if isinstance(self.max_depth_range, list):
                    param_grid['max_depth'] = self.max_depth_range
                else:
                    # Sample from distribution for grid search
                    n_samples = min(5, self.n_iter_search)
                    samples = self.max_depth_range.rvs(size=n_samples, random_state=self.random_state)
                    param_grid['max_depth'] = np.unique(samples.astype(int))
                
                # Handle min_child_weight
                if isinstance(self.min_child_weight_range, list):
                    param_grid['min_child_weight'] = self.min_child_weight_range
                else:
                    # Sample from distribution for grid search
                    n_samples = min(5, self.n_iter_search)
                    samples = self.min_child_weight_range.rvs(size=n_samples, random_state=self.random_state)
                    param_grid['min_child_weight'] = np.unique(samples)
                
                # Handle subsample
                if isinstance(self.subsample_range, list):
                    param_grid['subsample'] = self.subsample_range
                else:
                    # Sample from distribution for grid search
                    n_samples = min(5, self.n_iter_search)
                    samples = self.subsample_range.rvs(size=n_samples, random_state=self.random_state)
                    param_grid['subsample'] = np.unique(samples)
                
                # Handle colsample_bytree
                if isinstance(self.colsample_bytree_range, list):
                    param_grid['colsample_bytree'] = self.colsample_bytree_range
                else:
                    # Sample from distribution for grid search
                    n_samples = min(5, self.n_iter_search)
                    samples = self.colsample_bytree_range.rvs(size=n_samples, random_state=self.random_state)
                    param_grid['colsample_bytree'] = np.unique(samples)
                
                # Initialize XGBRegressor base model
                model = XGBRegressor(random_state=self.random_state, verbosity=0, n_jobs=-1)
                
                # Grid search
                grid_search = GridSearchCV(
                    model, param_grid=param_grid,
                    cv=self.cv_folds, scoring='neg_mean_squared_error',
                    error_score=np.nan, n_jobs=-1
                )
                grid_search.fit(X_clean_c, y_clean_c)
                best_params_dict[c] = grid_search.best_params_
                
            elif self.search_method == 'random':
                # Prepare parameter distributions for RandomizedSearchCV
                param_dist = {}
                
                # Handle n_estimators
                param_dist['n_estimators'] = self.n_estimators_range
                
                # Handle learning_rate
                param_dist['learning_rate'] = self.learning_rate_range
                
                # Handle max_depth
                param_dist['max_depth'] = self.max_depth_range
                
                # Handle min_child_weight
                param_dist['min_child_weight'] = self.min_child_weight_range
                
                # Handle subsample
                param_dist['subsample'] = self.subsample_range
                
                # Handle colsample_bytree
                param_dist['colsample_bytree'] = self.colsample_bytree_range
                
                # Initialize XGBRegressor base model
                model = XGBRegressor(random_state=self.random_state, verbosity=0, n_jobs=-1)
                
                # Randomized search
                random_search = RandomizedSearchCV(
                    model, param_distributions=param_dist, n_iter=self.n_iter_search,
                    cv=self.cv_folds, scoring='neg_mean_squared_error',
                    random_state=self.random_state, error_score=np.nan, n_jobs=-1
                )
                random_search.fit(X_clean_c, y_clean_c)
                best_params_dict[c] = random_search.best_params_
                
            elif self.search_method == 'bayesian':
                # Bayesian optimization with Optuna
                study = optuna.create_study(
                    direction='maximize',  # We're maximizing negative MSE
                    sampler=optuna.samplers.TPESampler(seed=self.random_state),
                    pruner=optuna.pruners.MedianPruner(n_startup_trials=5)
                )
                
                # Create objective function with data
                objective_with_data = lambda trial: self._objective(trial, X_clean_c, y_clean_c)
                
                # Optimize
                study.optimize(
                    objective_with_data,
                    n_trials=self.n_iter_search,
                    timeout=self.optuna_timeout,
                    n_jobs=self.optuna_n_jobs
                )
                
                # Extract best parameters
                best_params = study.best_params
                
                # Convert Optuna's best_params to scikit-learn format
                sklearn_params = {
                    'n_estimators': best_params['n_estimators'],
                    'learning_rate': best_params['learning_rate'],
                    'max_depth': best_params['max_depth'],
                    'min_child_weight': best_params['min_child_weight'],
                    'subsample': best_params['subsample'],
                    'colsample_bytree': best_params['colsample_bytree']
                }
                best_params_dict[c] = sklearn_params
                
            else:
                raise ValueError(f"Unknown search_method: {self.search_method}. Choose from 'grid', 'random', or 'bayesian'.")
        
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
                    norm.ppf(0.33, loc=loc, scale=scale),
                    norm.ppf(0.67, loc=loc, scale=scale),
                )
            elif code == 2:
                return (
                    lognorm.ppf(0.33, s=shape, loc=loc, scale=scale),
                    lognorm.ppf(0.67, s=shape, loc=loc, scale=scale),
                )
            elif code == 3:
                return (
                    expon.ppf(0.33, loc=loc, scale=scale),
                    expon.ppf(0.67, loc=loc, scale=scale),
                )
            elif code == 4:
                return (
                    gamma.ppf(0.33, a=shape, loc=loc, scale=scale),
                    gamma.ppf(0.67, a=shape, loc=loc, scale=scale),
                )
            elif code == 5:
                return (
                    weibull_min.ppf(0.33, c=shape, loc=loc, scale=scale),
                    weibull_min.ppf(0.67, c=shape, loc=loc, scale=scale),
                )
            elif code == 6:
                # Note: Renamed 't_dist' to 't' for standard scipy.stats
                return (
                    t.ppf(0.33, df=shape, loc=loc, scale=scale),
                    t.ppf(0.67, df=shape, loc=loc, scale=scale),
                )
            elif code == 7:
                # Poisson: poisson.ppf(q, mu, loc=0)
                # ASSUMPTION: 'mu' (mean) is passed as 'shape'
                #             'loc' is passed as 'loc'
                #             'scale' is unused
                return (
                    poisson.ppf(0.33, mu=shape, loc=loc),
                    poisson.ppf(0.67, mu=shape, loc=loc),
                )
            elif code == 8:
                # Negative Binomial: nbinom.ppf(q, n, p, loc=0)
                # ASSUMPTION: 'n' (successes) is passed as 'shape'
                #             'p' (probability) is passed as 'scale'
                #             'loc' is passed as 'loc'
                return (
                    nbinom.ppf(0.33, n=shape, p=scale, loc=loc),
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
        terciles_emp = clim.quantile([0.33, 0.67], dim="T")
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

    def forecast(self, Predictant, clim_year_start, clim_year_end, hindcast_det, hindcast_det_cross, Predictor_for_year, best_params=None, cluster_da=None, best_code_da=None, best_shape_da=None, best_loc_da=None, best_scale_da=None):
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
        # Predictor_for_year_st = (Predictor_for_year - mean_val) / std_val
        Predictor_for_year_st = Predictor_for_year
        # hindcast_det_st = standardize_timeseries(hindcast_det, clim_year_start, clim_year_end)
        hindcast_det_st = hindcast_det
        # Predictant_st = standardize_timeseries(Predictant_no_m, clim_year_start, clim_year_end)
        Predictant_st = Predictant_no_m
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
        
        # result_da = reverse_standardize(result_da, Predictant_no_m, clim_year_start, clim_year_end)
        
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
        T1_emp = terciles.isel(quantile=0).drop_vars('quantile')
        T2_emp = terciles.isel(quantile=1).drop_vars('quantile')
        error_variance = (Predictant_no_m - hindcast_det_cross).var(dim='T')
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
                result_da,
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
                result_da,
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


class WAS_mme_RF_:
    """
    Random Forest-based Multi-Model Ensemble (MME) forecasting.
    This class implements a single-model forecasting approach using scikit-learn's RandomForestRegressor
    for deterministic predictions, with optional tercile probability calculations using
    various statistical distributions. Implements multiple hyperparameter optimization methods.

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
        List of max features to tune (default is [None, 'sqrt', 0.33, 0.5]).
    random_state : int, optional
        Seed for reproducibility (default is 42).
    dist_method : str, optional
        Distribution method for tercile probabilities ('bestfit', 'nonparam', etc.) (default is 'nonparam').
    search_method : str, optional
        Hyperparameter optimization method: 'random' (RandomizedSearchCV), 
        'grid' (GridSearchCV), or 'bayesian' (Optuna) (default is 'random').
    n_iter_search : int, optional
        Number of iterations for randomized search (default is 10).
    n_trials : int, optional
        Number of trials for Bayesian optimization with Optuna (default is 100).
    cv_folds : int, optional
        Number of cross-validation folds (default is 3).
    """
    def __init__(self,
                 n_estimators_range=[50, 100, 200, 300],
                 max_depth_range=[None, 10, 20, 30],
                 min_samples_split_range=[2, 5, 10],
                 min_samples_leaf_range=[1, 2, 4],
                 max_features_range=[None, 'sqrt', 'log2' ,0.33, 0.5],
                 random_state=42,
                 dist_method="nonparam",
                 search_method="random",
                 n_iter_search=10,
                 n_trials=100,
                 cv_folds=3):
        self.n_estimators_range = n_estimators_range
        self.max_depth_range = max_depth_range
        self.min_samples_split_range = min_samples_split_range
        self.min_samples_leaf_range = min_samples_leaf_range
        self.max_features_range = max_features_range
        self.random_state = random_state
        self.dist_method = dist_method
        self.search_method = search_method
        self.n_iter_search = n_iter_search
        self.n_trials = n_trials
        self.cv_folds = cv_folds
        self.rf = None
        self.best_params_ = None  # Store best parameters found
        self.study_ = None  # Store Optuna study

    def _prepare_param_distributions(self):
        """Prepare parameter distributions for different search methods."""
        if self.search_method == 'grid':
            # For GridSearch, we need all combinations
            param_grid = {
                'n_estimators': self.n_estimators_range,
                'max_depth': self.max_depth_range,
                'min_samples_split': self.min_samples_split_range,
                'min_samples_leaf': self.min_samples_leaf_range,
                'max_features': self.max_features_range
            }
            return param_grid
        else:
            # For random search and bayesian, we can use distributions
            param_dist = {
                'n_estimators': self.n_estimators_range,
                'max_depth': self.max_depth_range,
                'min_samples_split': self.min_samples_split_range,
                'min_samples_leaf': self.min_samples_leaf_range,
                'max_features': self.max_features_range
            }
            return param_dist

    def _objective(self, trial, X_train, y_train):
        """Objective function for Bayesian optimization with Optuna."""
        n_estimators = trial.suggest_categorical('n_estimators', self.n_estimators_range)
        max_depth = trial.suggest_categorical('max_depth', self.max_depth_range)
        min_samples_split = trial.suggest_categorical('min_samples_split', self.min_samples_split_range)
        min_samples_leaf = trial.suggest_categorical('min_samples_leaf', self.min_samples_leaf_range)
        max_features = trial.suggest_categorical('max_features', self.max_features_range)
        
        # Handle 'None' and 'sqrt' for max_features
        if isinstance(max_features, str):
            if max_features == None:
                max_features = X_train.shape[1]
            elif max_features == 'sqrt':
                max_features = 'sqrt'
        
        model = RandomForestRegressor(
            n_estimators=n_estimators,
            max_depth=max_depth,
            min_samples_split=min_samples_split,
            min_samples_leaf=min_samples_leaf,
            max_features=max_features,
            random_state=self.random_state,
            n_jobs=-1
        )
        
        # Simple cross-validation (could be improved with sklearn's cross_val_score)
        from sklearn.model_selection import KFold
        kf = KFold(n_splits=self.cv_folds, shuffle=True, random_state=self.random_state)
        
        scores = []
        for train_idx, val_idx in kf.split(X_train):
            X_train_fold, X_val_fold = X_train[train_idx], X_train[val_idx]
            y_train_fold, y_val_fold = y_train[train_idx], y_train[val_idx]
            
            model.fit(X_train_fold, y_train_fold)
            y_pred = model.predict(X_val_fold)
            mse = np.mean((y_val_fold - y_pred) ** 2)
            scores.append(-mse)  # Negative MSE for maximization
        
        return np.mean(scores)

    def compute_hyperparameters(self, Predictors, Predictand, clim_year_start, clim_year_end):
        """
        Independently computes the best hyperparameters using selected optimization method.

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
        X_train = Predictors
        y_train = Predictand

        # Stack training data
        X_train_stacked = X_train.stack(sample=('T', 'Y', 'X')).transpose('sample', 'M').values
        y_train_stacked = y_train.stack(sample=('T', 'Y', 'X')).values.ravel()
        train_nan_mask = np.any(~np.isfinite(X_train_stacked), axis=1) | ~np.isfinite(y_train_stacked)
        X_train_clean = X_train_stacked[~train_nan_mask]
        y_train_clean = y_train_stacked[~train_nan_mask]

        # Initialize base model
        model = RandomForestRegressor(random_state=self.random_state, n_jobs=-1)

        if self.search_method == 'grid':
            # Grid Search
            param_grid = self._prepare_param_distributions()
            grid_search = GridSearchCV(
                model, param_grid=param_grid,
                cv=self.cv_folds, scoring='neg_mean_squared_error',
                error_score=np.nan, verbose=0
            )
            grid_search.fit(X_train_clean, y_train_clean)
            best_params = grid_search.best_params_
            self.best_params_ = best_params

        elif self.search_method == 'bayesian':
            # Bayesian Optimization with Optuna
            sampler = TPESampler(seed=self.random_state)
            self.study_ = optuna.create_study(
                direction='maximize',
                sampler=sampler
            )
            
            # Create objective function with data
            objective_with_data = lambda trial: self._objective(trial, X_train_clean, y_train_clean)
            self.study_.optimize(objective_with_data, n_trials=self.n_trials)
            
            # Get best parameters
            best_params = self.study_.best_params
            self.best_params_ = best_params

        else:  # Default to random search
            # Random Search (default)
            param_dist = self._prepare_param_distributions()
            random_search = RandomizedSearchCV(
                model, param_distributions=param_dist, n_iter=self.n_iter_search,
                cv=self.cv_folds, scoring='neg_mean_squared_error',
                random_state=self.random_state, error_score=np.nan
            )
            random_search.fit(X_train_clean, y_train_clean)
            best_params = random_search.best_params_
            self.best_params_ = best_params

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
            random_state=self.random_state,
            n_jobs=-1
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
                    norm.ppf(0.33, loc=loc, scale=scale),
                    norm.ppf(0.67, loc=loc, scale=scale),
                )
            elif code == 2:
                return (
                    lognorm.ppf(0.33, s=shape, loc=loc, scale=scale),
                    lognorm.ppf(0.67, s=shape, loc=loc, scale=scale),
                )
            elif code == 3:
                return (
                    expon.ppf(0.33, loc=loc, scale=scale),
                    expon.ppf(0.67, loc=loc, scale=scale),
                )
            elif code == 4:
                return (
                    gamma.ppf(0.33, a=shape, loc=loc, scale=scale),
                    gamma.ppf(0.67, a=shape, loc=loc, scale=scale),
                )
            elif code == 5:
                return (
                    weibull_min.ppf(0.33, c=shape, loc=loc, scale=scale),
                    weibull_min.ppf(0.67, c=shape, loc=loc, scale=scale),
                )
            elif code == 6:
                # Note: Renamed 't_dist' to 't' for standard scipy.stats
                return (
                    t.ppf(0.33, df=shape, loc=loc, scale=scale),
                    t.ppf(0.67, df=shape, loc=loc, scale=scale),
                )
            elif code == 7:
                # Poisson: poisson.ppf(q, mu, loc=0)
                # ASSUMPTION: 'mu' (mean) is passed as 'shape'
                #             'loc' is passed as 'loc'
                #             'scale' is unused
                return (
                    poisson.ppf(0.33, mu=shape, loc=loc),
                    poisson.ppf(0.67, mu=shape, loc=loc),
                )
            elif code == 8:
                # Negative Binomial: nbinom.ppf(q, n, p, loc=0)
                # ASSUMPTION: 'n' (successes) is passed as 'shape'
                #             'p' (probability) is passed as 'scale'
                #             'loc' is passed as 'loc'
                return (
                    nbinom.ppf(0.33, n=shape, p=scale, loc=loc),
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
        terciles_emp = clim.quantile([0.33, 0.67], dim="T")
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
                 hindcast_det_cross, Predictor_for_year, best_params=None, best_code_da=None, best_shape_da=None, best_loc_da=None, best_scale_da=None):
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
        # Predictor_for_year_st = (Predictor_for_year - mean_val) / std_val

        Predictor_for_year_st = Predictor_for_year

        # hindcast_det_st = standardize_timeseries(hindcast_det, clim_year_start, clim_year_end)
        hindcast_det_st = hindcast_det
        
        # Predictant_st = standardize_timeseries(Predictant_no_m, clim_year_start, clim_year_end)
        Predictant_st = Predictant_no_m
        
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
            random_state=self.random_state,
            n_jobs=-1
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

        # result_da = reverse_standardize(result_da, Predictant_no_m, clim_year_start, clim_year_end)
        
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
        T1_emp = terciles.isel(quantile=0).drop_vars('quantile')
        T2_emp = terciles.isel(quantile=1).drop_vars('quantile')
        error_variance = (Predictant_no_m - hindcast_det_cross).var(dim='T')
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
                result_da,
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
                result_da,
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



class WAS_mme_RF:
    """
    Random Forest-based Multi-Model Ensemble (MME) forecasting.
    This class implements a single-model forecasting approach using scikit-learn's RandomForestRegressor
    for deterministic predictions, with optional tercile probability calculations using
    various statistical distributions. Implements multiple hyperparameter optimization methods.
    
    Parameters
    ----------
    search_method : str, optional
        Hyperparameter optimization method: 'grid', 'random', or 'bayesian' (default: 'random').
    n_estimators_range : list of int or scipy.stats distribution, optional
        List of n_estimators values to tune (default is [50, 100, 200, 300]).
        Can be a list for grid search or a distribution for random/bayesian search.
    max_depth_range : list of int or scipy.stats distribution, optional
        List of max depths to tune (default is [None, 10, 20, 30]).
        Can be a list for grid search or a distribution for random/bayesian search.
    min_samples_split_range : list of int or scipy.stats distribution, optional
        List of minimum samples required to split to tune (default is [2, 5, 10]).
        Can be a list for grid search or a distribution for random/bayesian search.
    min_samples_leaf_range : list of int or scipy.stats distribution, optional
        List of minimum samples required at leaf node to tune (default is [1, 2, 4]).
        Can be a list for grid search or a distribution for random/bayesian search.
    max_features_range : list of str or float, optional
        List of max features to tune (default is [None, 'sqrt', 0.33, 0.5]).
    random_state : int, optional
        Seed for reproducibility (default is 42).
    dist_method : str, optional
        Distribution method for tercile probabilities ('bestfit', 'nonparam', etc.) (default is 'nonparam').
    n_iter_search : int, optional
        Number of iterations for randomized/bayesian search or points to sample for grid search (default is 10).
    cv_folds : int, optional
        Number of cross-validation folds (default is 3).
    n_clusters : int, optional
        Number of clusters for homogenized zones (default is 4).
    optuna_n_jobs : int, optional
        Number of parallel jobs for Optuna (default is 1).
    optuna_timeout : int, optional
        Timeout in seconds for Optuna optimization (default is None).
    """
    def __init__(self,
                 search_method='random',
                 n_estimators_range=[50, 100, 200, 300],
                 max_depth_range=[None, 10, 20, 30],
                 min_samples_split_range=[2, 5, 10],
                 min_samples_leaf_range=[1, 2, 4],
                 max_features_range=[None, 'sqrt', 'log2',  0.33, 0.5],
                 random_state=42,
                 dist_method="nonparam",
                 n_iter_search=10,
                 cv_folds=3,
                 n_clusters=4,
                 optuna_n_jobs=1,
                 optuna_timeout=None):
        
        self.search_method = search_method
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
        self.optuna_n_jobs = optuna_n_jobs
        self.optuna_timeout = optuna_timeout
        self.rf = None

    def _objective(self, trial, X_train, y_train):
        """
        Objective function for Optuna optimization.
        """
        # Define hyperparameter search space
        # Handle n_estimators
        if isinstance(self.n_estimators_range, list):
            n_estimators = trial.suggest_categorical('n_estimators', self.n_estimators_range)
        else:
            # Assume it's a distribution
            n_estimators = trial.suggest_int(
                'n_estimators', 
                int(self.n_estimators_range.a),
                int(self.n_estimators_range.b)
            )
        
        # Handle max_depth
        if isinstance(self.max_depth_range, list):
            max_depth_options = self.max_depth_range
            # Convert None to string for Optuna categorical suggestion
            max_depth_str_options = [str(opt) if opt is None else opt for opt in max_depth_options]
            max_depth_str = trial.suggest_categorical('max_depth', max_depth_str_options)
            # Convert back to None if needed
            max_depth = None if max_depth_str == 'None' else int(max_depth_str)
        else:
            # Assume it's a distribution for depth values (excluding None)
            max_depth = trial.suggest_int('max_depth', 5, 50)
        
        # Handle min_samples_split
        if isinstance(self.min_samples_split_range, list):
            min_samples_split = trial.suggest_categorical('min_samples_split', self.min_samples_split_range)
        else:
            # Assume it's a distribution
            min_samples_split = trial.suggest_int(
                'min_samples_split', 
                int(self.min_samples_split_range.a),
                int(self.min_samples_split_range.b)
            )
        
        # Handle min_samples_leaf
        if isinstance(self.min_samples_leaf_range, list):
            min_samples_leaf = trial.suggest_categorical('min_samples_leaf', self.min_samples_leaf_range)
        else:
            # Assume it's a distribution
            min_samples_leaf = trial.suggest_int(
                'min_samples_leaf', 
                int(self.min_samples_leaf_range.a),
                int(self.min_samples_leaf_range.b)
            )
        
        # Handle max_features
        # For Optuna, we need to handle both string and float options
        max_features_options = []
        for opt in self.max_features_range:
            if isinstance(opt, str):
                max_features_options.append(opt)
            else:
                max_features_options.append(str(opt))
        
        max_features_str = trial.suggest_categorical('max_features', max_features_options)
        
        # Convert back to appropriate type
        if max_features_str in [None, 'sqrt', 'log2']:
            max_features = max_features_str
        else:
            max_features = float(max_features_str)
        
        # Create and train model
        model = RandomForestRegressor(
            n_estimators=n_estimators,
            max_depth=max_depth,
            min_samples_split=min_samples_split,
            min_samples_leaf=min_samples_leaf,
            max_features=max_features,
            random_state=self.random_state,
            n_jobs=-1
        )
        
        # Use cross-validation
        from sklearn.model_selection import cross_val_score
        scores = cross_val_score(
            model, X_train, y_train, 
            cv=self.cv_folds, 
            scoring='neg_mean_squared_error',
            n_jobs=-1
        )
        
        return np.mean(scores)

    def compute_hyperparameters(self, Predictors, Predictand, clim_year_start, clim_year_end):
        """
        Independently computes the best hyperparameters using selected optimization method
        on stacked training data for each homogenized zone.
        
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
        
        X_train_std = Predictors
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
        y_train_std = Predictand
        X_train_std['T'] = y_train_std['T']
        
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
            
            if self.search_method == 'grid':
                # Prepare parameter grid for GridSearchCV
                param_grid = {}
                
                # Handle n_estimators
                if isinstance(self.n_estimators_range, list):
                    param_grid['n_estimators'] = self.n_estimators_range
                else:
                    # Sample from distribution for grid search
                    n_samples = min(5, self.n_iter_search)
                    samples = self.n_estimators_range.rvs(size=n_samples, random_state=self.random_state)
                    param_grid['n_estimators'] = np.unique(samples.astype(int))
                
                # Handle max_depth
                if isinstance(self.max_depth_range, list):
                    param_grid['max_depth'] = self.max_depth_range
                else:
                    # Sample from distribution for grid search
                    n_samples = min(5, self.n_iter_search)
                    samples = self.max_depth_range.rvs(size=n_samples, random_state=self.random_state)
                    # Filter out None values and convert to int
                    samples = samples[~np.isnan(samples)].astype(int)
                    unique_samples = np.unique(samples)
                    # Add None option back if it was in the original range
                    if None in self.max_depth_range:
                        unique_samples = list(unique_samples)
                        unique_samples.append(None)
                    param_grid['max_depth'] = unique_samples
                
                # Handle min_samples_split
                if isinstance(self.min_samples_split_range, list):
                    param_grid['min_samples_split'] = self.min_samples_split_range
                else:
                    # Sample from distribution for grid search
                    n_samples = min(5, self.n_iter_search)
                    samples = self.min_samples_split_range.rvs(size=n_samples, random_state=self.random_state)
                    param_grid['min_samples_split'] = np.unique(samples.astype(int))
                
                # Handle min_samples_leaf
                if isinstance(self.min_samples_leaf_range, list):
                    param_grid['min_samples_leaf'] = self.min_samples_leaf_range
                else:
                    # Sample from distribution for grid search
                    n_samples = min(5, self.n_iter_search)
                    samples = self.min_samples_leaf_range.rvs(size=n_samples, random_state=self.random_state)
                    param_grid['min_samples_leaf'] = np.unique(samples.astype(int))
                
                # Handle max_features
                param_grid['max_features'] = self.max_features_range
                
                # Initialize RandomForestRegressor base model
                model = RandomForestRegressor(random_state=self.random_state, n_jobs=-1)
                
                # Grid search
                grid_search = GridSearchCV(
                    model, param_grid=param_grid,
                    cv=self.cv_folds, scoring='neg_mean_squared_error',
                    error_score=np.nan, n_jobs=-1
                )
                grid_search.fit(X_clean_c, y_clean_c)
                best_params_dict[c] = grid_search.best_params_
                
            elif self.search_method == 'random':
                # Prepare parameter distributions for RandomizedSearchCV
                param_dist = {}
                
                # Handle n_estimators
                param_dist['n_estimators'] = self.n_estimators_range
                
                # Handle max_depth
                param_dist['max_depth'] = self.max_depth_range
                
                # Handle min_samples_split
                param_dist['min_samples_split'] = self.min_samples_split_range
                
                # Handle min_samples_leaf
                param_dist['min_samples_leaf'] = self.min_samples_leaf_range
                
                # Handle max_features
                param_dist['max_features'] = self.max_features_range
                
                # Initialize RandomForestRegressor base model
                model = RandomForestRegressor(random_state=self.random_state, n_jobs=-1)
                
                # Randomized search
                random_search = RandomizedSearchCV(
                    model, param_distributions=param_dist, n_iter=self.n_iter_search,
                    cv=self.cv_folds, scoring='neg_mean_squared_error',
                    random_state=self.random_state, error_score=np.nan, n_jobs=-1
                )
                random_search.fit(X_clean_c, y_clean_c)
                best_params_dict[c] = random_search.best_params_
                
            elif self.search_method == 'bayesian':
                # Bayesian optimization with Optuna
                study = optuna.create_study(
                    direction='maximize',  # We're maximizing negative MSE
                    sampler=optuna.samplers.TPESampler(seed=self.random_state),
                    pruner=optuna.pruners.MedianPruner(n_startup_trials=5)
                )
                
                # Create objective function with data
                objective_with_data = lambda trial: self._objective(trial, X_clean_c, y_clean_c)
                
                # Optimize
                study.optimize(
                    objective_with_data,
                    n_trials=self.n_iter_search,
                    timeout=self.optuna_timeout,
                    n_jobs=self.optuna_n_jobs
                )
                
                # Extract best parameters
                best_params = study.best_params
                
                # Convert Optuna's best_params to scikit-learn format
                # Handle max_depth conversion
                max_depth = best_params['max_depth']
                
                # Convert max_features back to appropriate type
                max_features_str = best_params['max_features']
                if max_features_str in [None, 'sqrt', 'log2']:
                    max_features = max_features_str
                else:
                    max_features = float(max_features_str)
                
                sklearn_params = {
                    'n_estimators': best_params['n_estimators'],
                    'max_depth': max_depth,
                    'min_samples_split': best_params['min_samples_split'],
                    'min_samples_leaf': best_params['min_samples_leaf'],
                    'max_features': max_features
                }
                best_params_dict[c] = sklearn_params
                
            else:
                raise ValueError(f"Unknown search_method: {self.search_method}. Choose from 'grid', 'random', or 'bayesian'.")
        
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
                    norm.ppf(0.33, loc=loc, scale=scale),
                    norm.ppf(0.67, loc=loc, scale=scale),
                )
            elif code == 2:
                return (
                    lognorm.ppf(0.33, s=shape, loc=loc, scale=scale),
                    lognorm.ppf(0.67, s=shape, loc=loc, scale=scale),
                )
            elif code == 3:
                return (
                    expon.ppf(0.33, loc=loc, scale=scale),
                    expon.ppf(0.67, loc=loc, scale=scale),
                )
            elif code == 4:
                return (
                    gamma.ppf(0.33, a=shape, loc=loc, scale=scale),
                    gamma.ppf(0.67, a=shape, loc=loc, scale=scale),
                )
            elif code == 5:
                return (
                    weibull_min.ppf(0.33, c=shape, loc=loc, scale=scale),
                    weibull_min.ppf(0.67, c=shape, loc=loc, scale=scale),
                )
            elif code == 6:
                # Note: Renamed 't_dist' to 't' for standard scipy.stats
                return (
                    t.ppf(0.33, df=shape, loc=loc, scale=scale),
                    t.ppf(0.67, df=shape, loc=loc, scale=scale),
                )
            elif code == 7:
                # Poisson: poisson.ppf(q, mu, loc=0)
                # ASSUMPTION: 'mu' (mean) is passed as 'shape'
                #             'loc' is passed as 'loc'
                #             'scale' is unused
                return (
                    poisson.ppf(0.33, mu=shape, loc=loc),
                    poisson.ppf(0.67, mu=shape, loc=loc),
                )
            elif code == 8:
                # Negative Binomial: nbinom.ppf(q, n, p, loc=0)
                # ASSUMPTION: 'n' (successes) is passed as 'shape'
                #             'p' (probability) is passed as 'scale'
                #             'loc' is passed as 'loc'
                return (
                    nbinom.ppf(0.33, n=shape, p=scale, loc=loc),
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
        terciles_emp = clim.quantile([0.33, 0.67], dim="T")
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

    def forecast(self, Predictant, clim_year_start, clim_year_end, hindcast_det, hindcast_det_cross, Predictor_for_year, best_params=None, cluster_da=None, best_code_da=None, best_shape_da=None, best_loc_da=None, best_scale_da=None):
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
        # Predictor_for_year_st = (Predictor_for_year - mean_val) / std_val
        Predictor_for_year_st = Predictor_for_year
        # hindcast_det_st = standardize_timeseries(hindcast_det, clim_year_start, clim_year_end)
        hindcast_det_st = hindcast_det
        # Predictant_st = standardize_timeseries(Predictant_no_m, clim_year_start, clim_year_end)
        Predictant_st = Predictant_no_m
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
        # result_da = reverse_standardize(result_da, Predictant_no_m, clim_year_start, clim_year_end)
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
        T1_emp = terciles.isel(quantile=0).drop_vars('quantile')
        T2_emp = terciles.isel(quantile=1).drop_vars('quantile')
        error_variance = (Predictant_no_m - hindcast_det_cross).var(dim='T')
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
                result_da,
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
                result_da,
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

class WAS_mme_Stacking:
    """
    Stacking ensemble for Multi-Model Ensemble (MME) forecasting.

    Base models (fixed in this implementation):
        - WAS_mme_hpELM
        - WAS_mme_MLP
        - WAS_mme_XGBoosting
        - WAS_mme_RF

    Each base model is expected to expose:
        * compute_hyperparameters(Predictors, Predictand, clim_year_start, clim_year_end)
            -> (best_params_per_cluster: dict[int, dict], cluster_da: xr.DataArray[Y,X])
        * compute_model(X_train, y_train, X_test, y_test,
                        best_params=best_params_per_cluster,
                        cluster_da=cluster_da)
            -> xr.DataArray[T,Y,X] deterministic predictions

    This class:
        1) Computes/uses hyperparameters for each base & cluster.
        2) Builds out-of-fold (OOF) predictions per base on training data.
        3) Trains a meta-learner separately for each cluster.
        4) For new data (hindcast test or forecast), predicts via:
              meta( base_1_pred, base_2_pred, base_3_pred, base_4_pred )
        5) Provides tercile probabilities via compute_prob / forecast.
    """

    # ------------------------------------------------------------------
    # Constructor
    # ------------------------------------------------------------------
    def __init__(self,
                 meta_learner_type='ridge',      # 'ridge', 'lasso', 'elasticnet', 'linear'
                 alpha_range=None,
                 l1_ratio_range=None,
                 random_state=42,
                 dist_method="nonparam",         # 'nonparam' or 'bestfit'
                 stacking_cv=3,
                 meta_search_method="random",    # 'grid', 'random', 'bayesian'
                 meta_cv_folds=3,
                 meta_n_iter_search=10,
                 meta_n_trials=100,              # For Bayesian optimization
                 n_clusters=4):

        if alpha_range is None:
            alpha_range = [0.1, 1.0, 10.0, 100.0]
        if l1_ratio_range is None:
            l1_ratio_range = [0.1, 0.5, 0.9]

        self.meta_learner_type = meta_learner_type
        self.alpha_range = alpha_range
        self.l1_ratio_range = l1_ratio_range
        self.random_state = random_state
        self.dist_method = dist_method
        self.stacking_cv = stacking_cv
        self.meta_search_method = meta_search_method
        self.meta_cv_folds = meta_cv_folds
        self.meta_n_iter_search = meta_n_iter_search
        self.meta_n_trials = meta_n_trials
        self.n_clusters = n_clusters

        # Instantiate base models with consistent config
        # Note: You'll need to import these classes or adjust as needed
        self.base_models = [
            WAS_mme_hpELM(random_state=random_state,
                          dist_method=dist_method,
                          n_clusters=n_clusters),
            WAS_mme_MLP(random_state=random_state,
                        dist_method=dist_method,
                        n_clusters=n_clusters),
            WAS_mme_XGBoosting(random_state=random_state,
                               dist_method=dist_method,
                               n_clusters=n_clusters),
            WAS_mme_RF(random_state=random_state,
                       dist_method=dist_method,
                       n_clusters=n_clusters),
        ]

        # Learned attributes
        self.best_params_list = None   # list[dict], one dict per base: {cluster_id -> params}
        self.cluster_da = None         # xr.DataArray[Y,X], int cluster labels
        self.meta_learners = {}        # {cluster_id -> fitted meta model}
        self.study_ = None            # For Bayesian optimization

    # ------------------------------------------------------------------
    # 1. Hyperparameter computation for base models
    # ------------------------------------------------------------------
    def compute_hyperparameters(self, Predictors, Predictand,
                                clim_year_start, clim_year_end):
        """
        Run each base model's hyperparameter search.
        Use clustering from the first base model as the common cluster_da.
        """
        if "M" in Predictand.dims:
            Predictand = Predictand.isel(M=0).drop_vars("M").squeeze()

        best_params_list = []
        cluster_da = None

        for i, base in enumerate(self.base_models):
            
            bp, cd = base.compute_hyperparameters(Predictors,
                                                  Predictand,
                                                  clim_year_start,
                                                  clim_year_end)
            best_params_list.append(bp)

            # Take cluster_da from the first model; assume others use same clustering scheme
            if cluster_da is None:
                cluster_da = cd

        # Make sure cluster labels are integers where valid
        if cluster_da is not None:
            cluster_vals = cluster_da.values
            mask = np.isfinite(cluster_vals)
            cluster_vals[mask] = cluster_vals[mask].astype(int)
            cluster_da = xr.DataArray(cluster_vals,
                                      coords=cluster_da.coords,
                                      dims=cluster_da.dims)

        self.best_params_list = best_params_list
        self.cluster_da = cluster_da
        return best_params_list, cluster_da

    # ------------------------------------------------------------------
    # 2. Internal: OOF predictions for stacking
    # ------------------------------------------------------------------
    def _get_oof_predictions(self, X, y,
                             best_params_list,
                             cluster_da):
        """
        Compute out-of-fold predictions for each base model over the training period.

        Returns:
            list of xr.DataArray (same shape as y: T,Y,X),
            OOF predictions for each base model.
        """
        n_time = X.sizes["T"]
        kf = KFold(n_splits=self.stacking_cv, shuffle=False)

        # Initialize OOF arrays
        oof_preds = [xr.full_like(y, np.nan) for _ in self.base_models]

        for m_idx, base in enumerate(self.base_models):
            base_best_params = best_params_list[m_idx]

            for train_idx, val_idx in kf.split(np.arange(n_time)):
                T_train = X["T"].values[train_idx]
                T_val = X["T"].values[val_idx]

                X_train_fold = X.sel(T=T_train)
                y_train_fold = y.sel(T=T_train)
                X_val_fold = X.sel(T=T_val)
                y_val_fold = y.sel(T=T_val)

                # Base hindcast on this fold
                pred_val = base.compute_model(
                    X_train_fold,
                    y_train_fold,
                    X_val_fold,
                    y_val_fold,
                    best_params=base_best_params,
                    cluster_da=cluster_da
                )

                # Insert fold predictions into OOF container
                oof_preds[m_idx].loc[dict(T=T_val)] = pred_val

        return oof_preds

    # ------------------------------------------------------------------
    # 3. Internal: hyperparameter optimization for meta-learners
    # ------------------------------------------------------------------
    def _objective_bayesian(self, trial, X, y, learner_type):
        """
        Objective function for Bayesian optimization with Optuna.
        """
        if learner_type == 'ridge':
            alpha = trial.suggest_categorical('alpha', self.alpha_range)
            model = Ridge(alpha=alpha, random_state=self.random_state)
        elif learner_type == 'lasso':
            alpha = trial.suggest_categorical('alpha', self.alpha_range)
            model = Lasso(alpha=alpha, random_state=self.random_state, max_iter=10000)
        elif learner_type == 'elasticnet':
            alpha = trial.suggest_categorical('alpha', self.alpha_range)
            l1_ratio = trial.suggest_categorical('l1_ratio', self.l1_ratio_range)
            model = ElasticNet(alpha=alpha, l1_ratio=l1_ratio, 
                               random_state=self.random_state, max_iter=10000)
        
        # Cross-validation score
        from sklearn.model_selection import cross_val_score
        scores = cross_val_score(model, X, y, cv=self.meta_cv_folds, 
                                 scoring='neg_mean_squared_error')
        return np.mean(scores)

    def _fit_meta_learners(self, oof_preds, y_train, cluster_da):
        """
        Fit one meta-learner per cluster using selected optimization method.
        """
        self.meta_learners = {}
        T_coord = y_train["T"]
        clusters = np.unique(cluster_da.values[np.isfinite(cluster_da.values)])

        for c_val in clusters:
            c = int(c_val)

            # Mask (T,Y,X) for this cluster
            mask_3d = (cluster_da == c).expand_dims(T=T_coord)

            # Build meta-features from OOF predictions
            X_cols = []
            for oof in oof_preds:
                vals = oof.where(mask_3d).stack(sample=("T", "Y", "X")).values
                X_cols.append(vals)

            if not X_cols:
                continue

            X_meta = np.column_stack(X_cols)
            y_meta = (y_train.where(mask_3d)
                              .stack(sample=("T", "Y", "X"))
                              .values
                              .astype(float))

            nan_mask = np.any(~np.isfinite(X_meta), axis=1) | ~np.isfinite(y_meta)
            if np.all(nan_mask):
                continue

            X_clean = X_meta[~nan_mask]
            y_clean = y_meta[~nan_mask]

            # Select and fit meta model
            if self.meta_learner_type == "linear":
                meta = LinearRegression()
                meta.fit(X_clean, y_clean)

            else:
                if self.meta_learner_type == "ridge":
                    base = Ridge(random_state=self.random_state)
                    param_dist = {"alpha": self.alpha_range}
                elif self.meta_learner_type == "lasso":
                    base = Lasso(random_state=self.random_state, max_iter=10000)
                    param_dist = {"alpha": self.alpha_range}
                elif self.meta_learner_type == "elasticnet":
                    base = ElasticNet(random_state=self.random_state, max_iter=10000)
                    param_dist = {
                        "alpha": self.alpha_range,
                        "l1_ratio": self.l1_ratio_range,
                    }
                else:
                    raise ValueError(f"Invalid meta_learner_type: {self.meta_learner_type}")

                # Choose optimization method
                if self.meta_search_method == "grid":
                    # Grid Search
                    search = GridSearchCV(
                        base,
                        param_grid=param_dist,
                        cv=self.meta_cv_folds,
                        scoring="neg_mean_squared_error",
                        error_score=np.nan,
                        verbose=0
                    )
                    search.fit(X_clean, y_clean)
                    meta = search.best_estimator_
                    
                elif self.meta_search_method == "bayesian":
                    # Bayesian Optimization with Optuna
                    sampler = TPESampler(seed=self.random_state)
                    study = optuna.create_study(
                        direction="maximize",
                        sampler=sampler
                    )
                    
                    # Create objective function with data
                    objective_with_data = lambda trial: self._objective_bayesian(
                        trial, X_clean, y_clean, self.meta_learner_type
                    )
                    
                    study.optimize(objective_with_data, n_trials=self.meta_n_trials)
                    
                    # Get best parameters and create model
                    best_params = study.best_params
                    if self.meta_learner_type == "ridge":
                        meta = Ridge(**best_params, random_state=self.random_state)
                    elif self.meta_learner_type == "lasso":
                        meta = Lasso(**best_params, random_state=self.random_state, max_iter=10000)
                    elif self.meta_learner_type == "elasticnet":
                        meta = ElasticNet(**best_params, random_state=self.random_state, max_iter=10000)
                    
                    meta.fit(X_clean, y_clean)
                    self.study_ = study  # Store the study for reference
                    
                else:  # Default to random search
                    # Random Search
                    search = RandomizedSearchCV(
                        base,
                        param_distributions=param_dist,
                        n_iter=self.meta_n_iter_search,
                        cv=self.meta_cv_folds,
                        scoring="neg_mean_squared_error",
                        random_state=self.random_state,
                        error_score=np.nan,
                    )
                    search.fit(X_clean, y_clean)
                    meta = search.best_estimator_

            self.meta_learners[c] = meta

    # ------------------------------------------------------------------
    # 4. Deterministic hindcast on test period
    # ------------------------------------------------------------------
    def compute_model(self,
                      X_train, y_train,
                      X_test, y_test,
                      best_params=None,
                      cluster_da=None,
                      clim_year_start=None,
                      clim_year_end=None):
        """
        Train stacked model on (X_train, y_train) and predict on X_test.

        Arguments:
            - If best_params & cluster_da are provided, they are used.
            - Otherwise, clim_year_start & clim_year_end must be given in
              order to compute base hyperparameters.

        Returns:
            xr.DataArray[T,Y,X] predictions on X_test.
        """

        
        # Ensure predictand has no member dim & consistent order
        if "M" in y_train.dims:
            y_train = y_train.isel(M=0).drop_vars("M").squeeze()
        if "M" in y_test.dims:
            y_test = y_test.isel(M=0).drop_vars("M").squeeze()

        y_train = y_train.transpose("T", "Y", "X")
        y_test = y_test.transpose("T", "Y", "X")

        # Get / compute hyperparameters + cluster_da
        if best_params is None or cluster_da is None:
            if clim_year_start is None or clim_year_end is None:
                raise ValueError(
                    "clim_year_start and clim_year_end must be provided if "
                    "best_params/cluster_da are not supplied."
                )
            best_params, cluster_da = self.compute_hyperparameters(
                X_train, y_train, clim_year_start, clim_year_end
            )

        self.best_params_list = best_params
        self.cluster_da = cluster_da

        # 1) OOF predictions over training
        oof_preds = self._get_oof_predictions(
            X_train, y_train, self.best_params_list, self.cluster_da
        )

        # 2) Fit meta-learners per cluster
        self._fit_meta_learners(oof_preds, y_train, self.cluster_da)

        # 3) Base predictions on X_test
        base_test_preds = []
        for m_idx, base in enumerate(self.base_models):
            bp = self.best_params_list[m_idx]
            pred = base.compute_model(
                X_train, y_train,
                X_test, y_test,
                best_params=bp,
                cluster_da=self.cluster_da
            )
            base_test_preds.append(pred)

        # 4) Apply meta-learners to stacked base predictions
        time = X_test["T"]
        lat = X_test["Y"]
        lon = X_test["X"]
        n_time, n_lat, n_lon = len(time), len(lat), len(lon)
        final_preds = np.full((n_time, n_lat, n_lon), np.nan)

        clusters = np.unique(self.cluster_da.values[np.isfinite(self.cluster_da.values)])

        for c_val in clusters:
            c = int(c_val)
            if c not in self.meta_learners:
                continue

            meta = self.meta_learners[c]
            mask_3d = (self.cluster_da == c).expand_dims(T=time)

            # Stack base predictions at cluster locations
            X_cols = []
            for pred_base in base_test_preds:
                vals = pred_base.where(mask_3d).stack(sample=("T", "Y", "X")).values
                X_cols.append(vals)

            X_meta = np.column_stack(X_cols)
            nan_mask = np.any(~np.isfinite(X_meta), axis=1)
            if np.all(nan_mask):
                continue

            X_clean = X_meta[~nan_mask]
            y_pred_clean = meta.predict(X_clean)

            full = np.full(X_meta.shape[0], np.nan)
            full[~nan_mask] = y_pred_clean
            pred_c = full.reshape(n_time, n_lat, n_lon)

            fill_mask = np.isnan(final_preds) & np.isfinite(pred_c)
            final_preds[fill_mask] = pred_c[fill_mask]

        predicted_da = xr.DataArray(
            final_preds,
            coords={"T": time, "Y": lat, "X": lon},
            dims=("T", "Y", "X"),
        )
        return predicted_da

    # ------------------------------------------------------------------
    # 5. Probabilities for hindcast (same pattern as other classes)
    # ------------------------------------------------------------------
    # Reuse your existing probability helpers; I keep them here with key fixes.

    @staticmethod
    def _ppf_terciles_from_code(dist_code, shape, loc, scale):
        if np.isnan(dist_code):
            return np.nan, np.nan
        code = int(dist_code)
        try:
            if code == 1:
                return norm.ppf(0.33, loc=loc, scale=scale), norm.ppf(0.67, loc=loc, scale=scale)
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
        if k <= 0:
            return np.inf
        try:
            g1 = gamma_function(1.0 + 1.0 / k)
            g2 = gamma_function(1.0 + 2.0 / k)
            implied = (g2 / (g1 ** 2)) - 1.0
            observed = V / (M ** 2)
            return implied - observed
        except Exception:
            return np.inf

    @staticmethod
    def calculate_tercile_probabilities_bestfit(best_guess, error_variance,
                                                T1, T2, dist_code, dof):

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
        error_variance = np.asarray(error_variance, float)
        n_time = best_guess.size
        out = np.full((3, n_time), np.nan)

        if (np.all(np.isnan(best_guess)) or np.isnan(dist_code)
                or np.isnan(T1) or np.isnan(T2) or np.isnan(error_variance)):
            return out

        code = int(dist_code)

        # Normal
        if code == 1:
            sigma = np.sqrt(error_variance)
            out[0] = norm.cdf(T1, loc=best_guess, scale=sigma)
            out[1] = norm.cdf(T2, loc=best_guess, scale=sigma) - out[0]
            out[2] = 1.0 - norm.cdf(T2, loc=best_guess, scale=sigma)

        # Lognormal
        elif code == 2:
            # assume best_guess > 0
            sigma2 = np.log(1.0 + error_variance / (best_guess ** 2))
            sigma = np.sqrt(np.maximum(sigma2, 1e-12))
            mu = np.log(np.maximum(best_guess, 1e-12)) - 0.5 * sigma ** 2
            c1 = lognorm.cdf(T1, s=sigma, scale=np.exp(mu))
            c2 = lognorm.cdf(T2, s=sigma, scale=np.exp(mu))
            out[0], out[1], out[2] = c1, c2 - c1, 1.0 - c2

        # Exponential (approx; mean = best_guess, scale from variance)
        elif code == 3:
            scale = np.sqrt(error_variance)
            c1 = expon.cdf(T1, loc=best_guess - scale, scale=scale)
            c2 = expon.cdf(T2, loc=best_guess - scale, scale=scale)
            out[0], out[1], out[2] = c1, c2 - c1, 1.0 - c2

        # Gamma
        elif code == 4:
            alpha = (best_guess ** 2) / error_variance
            theta = error_variance / np.maximum(best_guess, 1e-12)
            c1 = gamma.cdf(T1, a=alpha, scale=theta)
            c2 = gamma.cdf(T2, a=alpha, scale=theta)
            out[0], out[1], out[2] = c1, c2 - c1, 1.0 - c2

        # Weibull
        elif code == 5:
            for i in range(n_time):
                M = best_guess[i]
                V = error_variance
                if not np.isfinite(M) or not np.isfinite(V) or M <= 0 or V <= 0:
                    continue
                k = fsolve(WAS_mme_Stacking.weibull_shape_solver, 2.0, args=(M, V))[0]
                if k <= 0:
                    continue
                lam = M / gamma_function(1.0 + 1.0 / k)
                c1 = weibull_min.cdf(T1, c=k, scale=lam)
                c2 = weibull_min.cdf(T2, c=k, scale=lam)
                out[0, i], out[1, i], out[2, i] = c1, c2 - c1, 1.0 - c2

        # Student-t
        elif code == 6:
            if dof <= 2:
                return out
            scale = np.sqrt(error_variance * (dof - 2) / dof)
            c1 = t.cdf(T1, df=dof, loc=best_guess, scale=scale)
            c2 = t.cdf(T2, df=dof, loc=best_guess, scale=scale)
            out[0], out[1], out[2] = c1, c2 - c1, 1.0 - c2

        # Poisson
        elif code == 7:
            mu = np.maximum(best_guess, 0.0)
            c1 = poisson.cdf(T1, mu=mu)
            c2 = poisson.cdf(T2, mu=mu)
            out[0], out[1], out[2] = c1, c2 - c1, 1.0 - c2

        # Negative Binomial
        elif code == 8:
            # Overdispersed only
            p = np.where(error_variance > best_guess,
                         best_guess / error_variance, np.nan)
            n = np.where(error_variance > best_guess,
                         (best_guess ** 2) / (error_variance - best_guess), np.nan)
            c1 = nbinom.cdf(T1, n=n, p=p)
            c2 = nbinom.cdf(T2, n=n, p=p)
            out[0], out[1], out[2] = c1, c2 - c1, 1.0 - c2

        return out

    @staticmethod
    def calculate_tercile_probabilities_nonparametric(best_guess,
                                                      error_samples,
                                                      first_tercile,
                                                      second_tercile):
        best_guess = np.asarray(best_guess, float)
        error_samples = np.asarray(error_samples, float)
        n_time = best_guess.size
        out = np.full((3, n_time), np.nan)

        for t_idx in range(n_time):
            bg = best_guess[t_idx]
            if not np.isfinite(bg):
                continue
            dist = bg + error_samples
            dist = dist[np.isfinite(dist)]
            if dist.size == 0:
                continue
            p_b = np.mean(dist < first_tercile)
            p_n = np.mean((dist >= first_tercile) & (dist < second_tercile))
            p_a = 1.0 - (p_b + p_n)
            out[:, t_idx] = [p_b, p_n, p_a]
        return out

    def compute_prob(self,
                     Predictant,
                     clim_year_start,
                     clim_year_end,
                     hindcast_det,
                     best_code_da=None,
                     best_shape_da=None,
                     best_loc_da=None,
                     best_scale_da=None):
        """
        Compute tercile probabilities for stacked deterministic hindcasts.
        """
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
            if any(v is None for v in
                   (best_code_da, best_shape_da, best_loc_da, best_scale_da)):
                raise ValueError(
                    "dist_method='bestfit' requires best_code_da, best_shape_da, "
                    "best_loc_da, best_scale_da."
                )

            T1, T2 = xr.apply_ufunc(
                WAS_mme_Stacking._ppf_terciles_from_code,
                best_code_da, best_shape_da, best_loc_da, best_scale_da,
                input_core_dims=[(), (), (), ()],
                output_core_dims=[(), ()],
                vectorize=True,
                dask="parallelized",
                output_dtypes=[float, float],
            )

            hindcast_prob = xr.apply_ufunc(
                WAS_mme_Stacking.calculate_tercile_probabilities_bestfit,
                hindcast_det,
                error_variance,
                T1,
                T2,
                best_code_da,
                input_core_dims=[("T",), (), (), (), ()],
                output_core_dims=[("probability", "T")],
                vectorize=True,
                kwargs={"dof": dof},
                dask="parallelized",
                output_dtypes=[float],
                dask_gufunc_kwargs={"output_sizes": {"probability": 3}},
            )

        elif dm == "nonparam":
            error_samples = Predictant - hindcast_det
            hindcast_prob = xr.apply_ufunc(
                WAS_mme_Stacking.calculate_tercile_probabilities_nonparametric,
                hindcast_det,
                error_samples,
                T1_emp,
                T2_emp,
                input_core_dims=[("T",), ("T",), (), ()],
                output_core_dims=[("probability", "T")],
                vectorize=True,
                dask="parallelized",
                output_dtypes=[float],
                dask_gufunc_kwargs={"output_sizes": {"probability": 3}},
            )
        else:
            raise ValueError(f"Invalid dist_method: {dm}")

        hindcast_prob = hindcast_prob.assign_coords(
            probability=("probability", ["PB", "PN", "PA"])
        )
        return (hindcast_prob * mask).transpose("probability", "T", "Y", "X")

    # ------------------------------------------------------------------
    # 6. Forecast: stacked deterministic + tercile probabilities
    # ------------------------------------------------------------------
    def forecast(self,
                 Predictant,
                 clim_year_start,
                 clim_year_end,
                 hindcast_det,
                 hindcast_det_cross,
                 Predictor_for_year,
                 best_params=None,
                 cluster_da=None,
                 best_code_da=None,
                 best_shape_da=None,
                 best_loc_da=None,
                 best_scale_da=None):
        """
        One-step forecast for target year using stacking.

        Mirrors pattern of other WAS_mme_* classes:
        - uses hindcast_det as training predictors
        - uses Predictor_for_year as new predictors
        - reuses per-base hyperparameters & cluster_da
        - meta-learners trained on standardized hindcast period
        """
        # Remove member dimension if present
        if "M" in Predictant.dims:
            Predictant_no_m = Predictant.isel(M=0).drop_vars("M").squeeze()
        else:
            Predictant_no_m = Predictant

        # Mask
        mask = xr.where(~np.isnan(Predictant_no_m.isel(T=0)), 1.0, np.nan)
        mask_np = mask.squeeze().to_numpy()

        # Standardize hindcast_det for training predictors
        mean_val = hindcast_det.sel(T=slice(str(clim_year_start),
                                            str(clim_year_end))).mean(dim="T")
        std_val = hindcast_det.sel(T=slice(str(clim_year_start),
                                           str(clim_year_end))).std(dim="T")

        Predictor_for_year_st = (Predictor_for_year - mean_val) / std_val
        hindcast_det_st = standardize_timeseries(hindcast_det,
                                                 clim_year_start,
                                                 clim_year_end)
        Predictant_st = standardize_timeseries(Predictant_no_m,
                                               clim_year_start,
                                               clim_year_end)
        hindcast_det_st["T"] = Predictant_st["T"]

        # Hyperparameters / clustering
        if best_params is None or cluster_da is None:
            best_params, cluster_da = self.compute_hyperparameters(
                hindcast_det, Predictant_no_m, clim_year_start, clim_year_end
            )

        self.best_params_list = best_params
        self.cluster_da = cluster_da

        # OOF on standardized hindcast
        oof_preds = self._get_oof_predictions(
            hindcast_det_st, Predictant_st,
            self.best_params_list, self.cluster_da
        )
        self._fit_meta_learners(oof_preds, Predictant_st, self.cluster_da)

        # Base forecasts (standardized space)
        time = Predictor_for_year_st["T"]
        lat = Predictor_for_year_st["Y"]
        lon = Predictor_for_year_st["X"]
        n_time, n_lat, n_lon = len(time), len(lat), len(lon)

        y_test_dummy = xr.full_like(Predictant_st.isel(T=0),
                                    np.nan).expand_dims(T=time)

        base_forecast_std = []
        for m_idx, base in enumerate(self.base_models):
            bp = self.best_params_list[m_idx]
            pred_std = base.compute_model(
                hindcast_det_st,
                Predictant_st,
                Predictor_for_year_st,
                y_test_dummy,
                best_params=bp,
                cluster_da=self.cluster_da
            )
            base_forecast_std.append(pred_std)

        # Meta stacked forecast (standardized)
        preds_std = np.full((n_time, n_lat, n_lon), np.nan)

        clusters = np.unique(self.cluster_da.values[np.isfinite(self.cluster_da.values)])
        for c_val in clusters:
            c = int(c_val)
            if c not in self.meta_learners:
                continue
            meta = self.meta_learners[c]
            mask_3d = (self.cluster_da == c).expand_dims(T=time)

            X_cols = []
            for pred_b in base_forecast_std:
                vals = pred_b.where(mask_3d).stack(sample=("T", "Y", "X")).values
                X_cols.append(vals)
            X_meta = np.column_stack(X_cols)
            nan_mask = np.any(~np.isfinite(X_meta), axis=1)
            if np.all(nan_mask):
                continue

            X_clean = X_meta[~nan_mask]
            y_pred_clean = meta.predict(X_clean)
            full = np.full(X_meta.shape[0], np.nan)
            full[~nan_mask] = y_pred_clean
            pred_c = full.reshape(n_time, n_lat, n_lon)

            fill_mask = np.isnan(preds_std) & np.isfinite(pred_c)
            preds_std[fill_mask] = pred_c[fill_mask]

        # De-standardize to physical space
        result_da = xr.DataArray(
            preds_std,
            coords={"T": time, "Y": lat, "X": lon},
            dims=("T", "Y", "X"),
        ) * mask_np

        result_da = reverse_standardize(result_da,
                                        Predictant_no_m,
                                        clim_year_start,
                                        clim_year_end)

        # Set forecast time stamp consistent with other classes
        year = (Predictor_for_year["T"].values
                .astype("datetime64[Y]").astype(int)[0] + 1970)
        first_T = Predictant_no_m.isel(T=0)["T"].values
        first_month = (first_T.astype("datetime64[M]").astype(int) % 12) + 1
        new_T = np.datetime64(f"{year}-{int(first_month):02d}-01")

        result_da = result_da.assign_coords(T=xr.DataArray([new_T], dims=["T"]))
        result_da["T"] = result_da["T"].astype("datetime64[ns]")

        # --- Probabilities for the forecast ---
        # Use same climatology window as for other models
        index_start = Predictant_no_m.get_index("T").get_loc(str(clim_year_start)).start
        index_end = Predictant_no_m.get_index("T").get_loc(str(clim_year_end)).stop
        rainfall_clim = Predictant_no_m.isel(T=slice(index_start, index_end))

        terciles = rainfall_clim.quantile([0.33, 0.67], dim="T")
        T1_emp = terciles.isel(quantile=0).drop_vars("quantile")
        T2_emp = terciles.isel(quantile=1).drop_vars("quantile")

        error_variance = (Predictant_no_m - hindcast_det_cross).var(dim="T")
        dof = max(int(rainfall_clim.sizes["T"]) - 1, 2)

        dm = self.dist_method

        if dm == "bestfit":
            if any(v is None for v in
                   (best_code_da, best_shape_da, best_loc_da, best_scale_da)):
                raise ValueError(
                    "dist_method='bestfit' requires best_code_da, best_shape_da, "
                    "best_loc_da, best_scale_da."
                )
            T1, T2 = xr.apply_ufunc(
                WAS_mme_Stacking._ppf_terciles_from_code,
                best_code_da, best_shape_da, best_loc_da, best_scale_da,
                input_core_dims=[(), (), (), ()],
                output_core_dims=[(), ()],
                vectorize=True,
                dask="parallelized",
                output_dtypes=[float, float],
            )
            forecast_prob = xr.apply_ufunc(
                WAS_mme_Stacking.calculate_tercile_probabilities_bestfit,
                result_da,
                error_variance,
                T1,
                T2,
                best_code_da,
                input_core_dims=[("T",), (), (), (), ()],
                output_core_dims=[("probability", "T")],
                vectorize=True,
                kwargs={"dof": dof},
                dask="parallelized",
                output_dtypes=[float],
                dask_gufunc_kwargs={"output_sizes": {"probability": 3}},
            )
        elif dm == "nonparam":
            error_samples = Predictant_no_m - hindcast_det
            forecast_prob = xr.apply_ufunc(
                WAS_mme_Stacking.calculate_tercile_probabilities_nonparametric,
                result_da,
                error_samples,
                T1_emp,
                T2_emp,
                input_core_dims=[("T",), ("T",), (), ()],
                output_core_dims=[("probability", "T")],
                vectorize=True,
                dask="parallelized",
                output_dtypes=[float],
                dask_gufunc_kwargs={"output_sizes": {"probability": 3}},
            )
        else:
            raise ValueError(f"Invalid dist_method: {dm}")

        forecast_prob = forecast_prob.assign_coords(
            probability=("probability", ["PB", "PN", "PA"])
        )

        return result_da * mask_np, (forecast_prob * mask_np).transpose(
            "probability", "T", "Y", "X"
        )


class WAS_mme_Stacking_:   
    """
    Stacking ensemble for Multi-Model Ensemble (MME) forecasting using provided base models.
    Stacks hpELM_, MLP_, XGBoosting_, RF_ via a meta-learner (ridge/lasso/elasticnet/linear).
    Supports deterministic and probabilistic (tercile) outputs.
    """

    def __init__(self,
                 meta_learner_type='ridge',
                 alpha_range=None,
                 l1_ratio_range=None,
                 random_state=42,
                 dist_method="nonparam",       # 'nonparam' or 'bestfit'
                 stacking_cv=3,
                 meta_search_method="random",  # 'random', 'grid', or 'bayesian'
                 meta_cv_folds=3,
                 meta_n_iter_search=10,
                 meta_n_trials=100):           # For Bayesian optimization
        if alpha_range is None:
            alpha_range = [0.1, 1.0, 10.0, 100.0]
        if l1_ratio_range is None:
            l1_ratio_range = [0.1, 0.5, 0.9]

        self.meta_learner_type = meta_learner_type
        self.alpha_range = alpha_range
        self.l1_ratio_range = l1_ratio_range
        self.random_state = random_state
        self.dist_method = dist_method
        self.stacking_cv = stacking_cv
        self.meta_search_method = meta_search_method
        self.meta_cv_folds = meta_cv_folds
        self.meta_n_iter_search = meta_n_iter_search
        self.meta_n_trials = meta_n_trials
        self.study_ = None  # For Bayesian optimization

        self.base_models = [
            WAS_mme_hpELM_(random_state=random_state, dist_method=dist_method),
            WAS_mme_MLP_(random_state=random_state, dist_method=dist_method),
            WAS_mme_XGBoosting_(random_state=random_state, dist_method=dist_method),
            WAS_mme_RF_(random_state=random_state, dist_method=dist_method),
        ]
        self.meta_learner = None

    # ------------------ Hyperparameter optimization methods ------------------

    def _objective_bayesian(self, trial, X, y, learner_type):
        """
        Objective function for Bayesian optimization with Optuna.
        """
        if learner_type == 'ridge':
            alpha = trial.suggest_categorical('alpha', self.alpha_range)
            model = Ridge(alpha=alpha, random_state=self.random_state)
        elif learner_type == 'lasso':
            alpha = trial.suggest_categorical('alpha', self.alpha_range)
            model = Lasso(alpha=alpha, random_state=self.random_state, max_iter=10000)
        elif learner_type == 'elasticnet':
            alpha = trial.suggest_categorical('alpha', self.alpha_range)
            l1_ratio = trial.suggest_categorical('l1_ratio', self.l1_ratio_range)
            model = ElasticNet(alpha=alpha, l1_ratio=l1_ratio, 
                               random_state=self.random_state, max_iter=10000)
        
        # Cross-validation score
        from sklearn.model_selection import cross_val_score
        scores = cross_val_score(model, X, y, cv=self.meta_cv_folds, 
                                 scoring='neg_mean_squared_error')
        return np.mean(scores)

    def _fit_meta_learner_with_search(self, X_meta_clean, y_meta_clean):
        """
        Fit meta-learner using selected optimization method.
        """
        if self.meta_learner_type == "linear":
            meta = LinearRegression()
            meta.fit(X_meta_clean, y_meta_clean)
            return meta
        
        # Prepare base model and parameter distribution
        if self.meta_learner_type == "ridge":
            base = Ridge(random_state=self.random_state)
            param_dist = {"alpha": self.alpha_range}
        elif self.meta_learner_type == "lasso":
            base = Lasso(random_state=self.random_state, max_iter=10000)
            param_dist = {"alpha": self.alpha_range}
        elif self.meta_learner_type == "elasticnet":
            base = ElasticNet(random_state=self.random_state, max_iter=10000)
            param_dist = {"alpha": self.alpha_range, "l1_ratio": self.l1_ratio_range}
        else:
            raise ValueError(f"Invalid meta_learner_type: {self.meta_learner_type}")

        # Select optimization method
        if self.meta_search_method == "grid":
            # Grid Search
            search = GridSearchCV(
                base,
                param_grid=param_dist,
                cv=self.meta_cv_folds,
                scoring="neg_mean_squared_error",
                error_score=np.nan,
                verbose=0
            )
            search.fit(X_meta_clean, y_meta_clean)
            meta = search.best_estimator_
            
        elif self.meta_search_method == "bayesian":
            # Bayesian Optimization with Optuna
            sampler = TPESampler(seed=self.random_state)
            self.study_ = optuna.create_study(
                direction="maximize",
                sampler=sampler
            )
            
            # Create objective function with data
            objective_with_data = lambda trial: self._objective_bayesian(
                trial, X_meta_clean, y_meta_clean, self.meta_learner_type
            )
            
            self.study_.optimize(objective_with_data, n_trials=self.meta_n_trials)
            
            # Get best parameters and create model
            best_params = self.study_.best_params
            if self.meta_learner_type == "ridge":
                meta = Ridge(**best_params, random_state=self.random_state)
            elif self.meta_learner_type == "lasso":
                meta = Lasso(**best_params, random_state=self.random_state, max_iter=10000)
            elif self.meta_learner_type == "elasticnet":
                meta = ElasticNet(**best_params, random_state=self.random_state, max_iter=10000)
            
            meta.fit(X_meta_clean, y_meta_clean)
            
        else:  # Default to random search
            # Random Search
            search = RandomizedSearchCV(
                base,
                param_distributions=param_dist,
                n_iter=self.meta_n_iter_search,
                cv=self.meta_cv_folds,
                scoring="neg_mean_squared_error",
                random_state=self.random_state,
                error_score=np.nan
            )
            search.fit(X_meta_clean, y_meta_clean)
            meta = search.best_estimator_

        return meta

    # ------------------ Hyperparameters for base models ----------------

    def compute_hyperparameters(self, Predictors, Predictand, clim_year_start, clim_year_end):
        """
        Collect per-base best params (no clustering assumed for underscore bases).
        """
        if "M" in Predictand.dims:
            Predictand = Predictand.isel(M=0).drop_vars("M").squeeze()
        best_params_list = []
        for base in self.base_models:
            bp = base.compute_hyperparameters(Predictors, Predictand, clim_year_start, clim_year_end)
            best_params_list.append(bp)
        return best_params_list

    # ------------------ OOF predictions (stacking features) ------------------

    def _get_oof_predictions(self, X, y, best_params_list):
        """
        Out-of-fold predictions for each base over training period.
        Returns list of DataArrays, each (T,Y,X).
        """
        X = X.transpose("T", "Y", "X")
        y = y.transpose("T", "Y", "X")

        kf = KFold(n_splits=self.stacking_cv, shuffle=False)
        n_t = X.sizes["T"]
        oof_preds = [xr.full_like(y, np.nan) for _ in self.base_models]

        for i, base in enumerate(self.base_models):
            bp = best_params_list[i]
            for train_idx, val_idx in kf.split(range(n_t)):
                X_train_fold = X.isel(T=train_idx)
                y_train_fold = y.isel(T=train_idx)
                X_val_fold = X.isel(T=val_idx)
                y_val_fold = y.isel(T=val_idx)

                pred_val = base.compute_model(
                    X_train_fold, y_train_fold,
                    X_val_fold, y_val_fold,
                    best_params=bp
                )
                # write fold preds into OOF container
                oof_preds[i].loc[dict(T=X_val_fold["T"])] = pred_val
        return oof_preds

    # ------------------ Deterministic hindcast/test ------------------

    def compute_model(self, X_train, y_train, X_test, y_test,
                      best_params=None, clim_year_start=None, clim_year_end=None):
        """
        Fit stacking on training (using OOF), then predict on X_test.
        """
        # handle member dim and order
        if "M" in y_train.dims:
            y_train = y_train.isel(M=0).drop_vars("M").squeeze()
        if "M" in y_test.dims:
            y_test = y_test.isel(M=0).drop_vars("M").squeeze()
        X_train = X_train.transpose("T", "Y", "X")
        y_train = y_train.transpose("T", "Y", "X")
        X_test  = X_test.transpose("T", "Y", "X")
        y_test  = y_test.transpose("T", "Y", "X")

        if best_params is None:
            if clim_year_start is None or clim_year_end is None:
                raise ValueError("Need clim_year_start/clim_year_end to compute base hyperparameters.")
            best_params = self.compute_hyperparameters(X_train, y_train, clim_year_start, clim_year_end)
        best_params_list = best_params

        # 1) OOF
        oof_preds = self._get_oof_predictions(X_train, y_train, best_params_list)

        # 2) Build meta design matrix (NaN-safe)
        X_cols = [oof.stack(sample=("T","Y","X")).values for oof in oof_preds]
        X_meta = np.column_stack(X_cols)
        y_meta = y_train.stack(sample=("T","Y","X")).values
        nan_mask = np.any(~np.isfinite(X_meta), axis=1) | ~np.isfinite(y_meta)

        if np.all(nan_mask):
            # Fallback: no clean rows, use simple mean-ensemble at prediction time
            self.meta_learner = None
        else:
            X_meta_clean = X_meta[~nan_mask]
            y_meta_clean = y_meta[~nan_mask]

            # 3) Fit meta-learner
            if self.meta_learner_type == "linear":
                meta = LinearRegression()
                meta.fit(X_meta_clean, y_meta_clean)
            else:
                if self.meta_learner_type == "ridge":
                    meta_base = Ridge()
                    param_dist = {"alpha": self.alpha_range}
                elif self.meta_learner_type == "lasso":
                    meta_base = Lasso(max_iter=10000)
                    param_dist = {"alpha": self.alpha_range}
                elif self.meta_learner_type == "elasticnet":
                    meta_base = ElasticNet(max_iter=10000)
                    param_dist = {"alpha": self.alpha_range, "l1_ratio": self.l1_ratio_range}
                else:
                    raise ValueError(f"Invalid meta_learner_type: {self.meta_learner_type}")

                search = RandomizedSearchCV(
                    meta_base,
                    param_distributions=param_dist,
                    n_iter=self.meta_n_iter_search,
                    cv=self.meta_cv_folds,
                    scoring="neg_mean_squared_error",
                    random_state=self.random_state,
                    error_score=np.nan
                )
                search.fit(X_meta_clean, y_meta_clean)
                meta = search.best_estimator_
            self.meta_learner = meta

        # 4) Base predictions on test
        base_test_preds = []
        for i, base in enumerate(self.base_models):
            bp = best_params_list[i]
            pred_test = base.compute_model(X_train, y_train, X_test, y_test, best_params=bp)
            base_test_preds.append(pred_test)

        # 5) Stack & predict
        time = X_test["T"]; lat = X_test["Y"]; lon = X_test["X"]
        n_time, n_lat, n_lon = len(time), len(lat), len(lon)
        X_cols_test = [p.stack(sample=("T","Y","X")).values for p in base_test_preds]
        X_meta_test = np.column_stack(X_cols_test)
        nan_mask_test = np.any(~np.isfinite(X_meta_test), axis=1)

        if self.meta_learner is None:
            # mean of bases (ignoring NaNs) as last resort
            pred_mean = xr.concat(base_test_preds, dim="model").mean(dim="model", skipna=True)
            predicted_da = pred_mean
        else:
            predictions = np.full((n_time, n_lat, n_lon), np.nan)
            if not np.all(nan_mask_test):
                X_meta_test_clean = X_meta_test[~nan_mask_test]
                y_pred_clean = self.meta_learner.predict(X_meta_test_clean)
                full = np.full(X_meta_test.shape[0], np.nan)
                full[~nan_mask_test] = y_pred_clean
                pred_reshaped = full.reshape(n_time, n_lat, n_lon)
                predictions = np.where(np.isnan(predictions), pred_reshaped, predictions)

            predicted_da = xr.DataArray(
                predictions, coords={"T": time, "Y": lat, "X": lon}, dims=("T","Y","X")
            )
        return predicted_da

    # ------------------ Probability helpers ------------------

    @staticmethod
    def _ppf_terciles_from_code(dist_code, shape, loc, scale):
        """
        Map best-fit family code -> tercile thresholds (T1, T2).
        Codes:
          1:norm, 2:lognorm, 3:expon, 4:gamma, 5:weibull_min, 6:t, 7:poisson, 8:nbinom
        """
        if np.isnan(dist_code):
            return np.nan, np.nan
        code = int(dist_code)
        try:
            if code == 1:
                return norm.ppf(0.33, loc=loc, scale=scale), norm.ppf(0.67, loc=loc, scale=scale)
            elif code == 2:
                return (lognorm.ppf(0.33, s=shape, loc=loc, scale=scale),
                        lognorm.ppf(0.67, s=shape, loc=loc, scale=scale))
            elif code == 3:
                return expon.ppf(0.33, loc=loc, scale=scale), expon.ppf(0.67, loc=loc, scale=scale)
            elif code == 4:
                return (gamma.ppf(0.33, a=shape, loc=loc, scale=scale),
                        gamma.ppf(0.67, a=shape, loc=loc, scale=scale))
            elif code == 5:
                return (weibull_min.ppf(0.33, c=shape, loc=loc, scale=scale),
                        weibull_min.ppf(0.67, c=shape, loc=loc, scale=scale))
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
        """
        Root for Weibull shape k: (Gamma(1+2/k)/Gamma(1+1/k)^2 - 1) - V/M^2 = 0
        """
        if k <= 0:
            return np.inf
        try:
            g1 = gamma_function(1.0 + 1.0 / k)
            g2 = gamma_function(1.0 + 2.0 / k)
            implied = (g2 / (g1 ** 2)) - 1.0
            observed = V / (M ** 2)
            return implied - observed
        except Exception:
            return np.inf

    @staticmethod
    def calculate_tercile_probabilities_bestfit(best_guess, error_variance, T1, T2, dist_code, dof):
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
        error_variance = np.asarray(error_variance, float)
        n_time = best_guess.size
        out = np.full((3, n_time), np.nan)

        if (np.all(np.isnan(best_guess)) or np.isnan(dist_code)
                or np.isnan(T1) or np.isnan(T2) or np.isnan(error_variance)):
            return out

        code = int(dist_code)

        # Normal
        if code == 1:
            sigma = np.sqrt(error_variance)
            out[0] = norm.cdf(T1, loc=best_guess, scale=sigma)
            out[1] = norm.cdf(T2, loc=best_guess, scale=sigma) - out[0]
            out[2] = 1.0 - norm.cdf(T2, loc=best_guess, scale=sigma)

        # Lognormal (ensure positivity)
        elif code == 2:
            eps = 1e-12
            bg = np.maximum(best_guess, eps)
            sigma2 = np.log(1.0 + error_variance / (bg ** 2))
            sigma = np.sqrt(np.maximum(sigma2, eps))
            mu = np.log(bg) - 0.5 * sigma ** 2
            c1 = lognorm.cdf(T1, s=sigma, scale=np.exp(mu))
            c2 = lognorm.cdf(T2, s=sigma, scale=np.exp(mu))
            out[0], out[1], out[2] = c1, c2 - c1, 1.0 - c2

        # Exponential (variance = scale^2, mean = loc + scale)
        elif code == 3:
            scale = np.sqrt(error_variance)
            loc = best_guess - scale
            c1 = expon.cdf(T1, loc=loc, scale=scale)
            c2 = expon.cdf(T2, loc=loc, scale=scale)
            out[0], out[1], out[2] = c1, c2 - c1, 1.0 - c2

        # Gamma (alpha=k, theta=scale)
        elif code == 4:
            eps = 1e-12
            alpha = (best_guess ** 2) / np.maximum(error_variance, eps)
            theta = np.maximum(error_variance, eps) / np.maximum(best_guess, eps)
            c1 = gamma.cdf(T1, a=alpha, scale=theta)
            c2 = gamma.cdf(T2, a=alpha, scale=theta)
            out[0], out[1], out[2] = c1, c2 - c1, 1.0 - c2

        # Weibull
        elif code == 5:
            for i in range(n_time):
                M = best_guess[i]
                V = error_variance
                if not np.isfinite(M) or not np.isfinite(V) or M <= 0 or V <= 0:
                    continue
                k = fsolve(WAS_mme_Stacking_.weibull_shape_solver, 2.0, args=(M, V))[0]
                if k <= 0:
                    continue
                lam = M / gamma_function(1.0 + 1.0 / k)
                c1 = weibull_min.cdf(T1, c=k, scale=lam)
                c2 = weibull_min.cdf(T2, c=k, scale=lam)
                out[0, i], out[1, i], out[2, i] = c1, c2 - c1, 1.0 - c2

        # Student-t
        elif code == 6:
            if dof <= 2:
                return out
            scale = np.sqrt(error_variance * (dof - 2) / dof)
            c1 = t.cdf(T1, df=dof, loc=best_guess, scale=scale)
            c2 = t.cdf(T2, df=dof, loc=best_guess, scale=scale)
            out[0], out[1], out[2] = c1, c2 - c1, 1.0 - c2

        # Poisson
        elif code == 7:
            mu = np.maximum(best_guess, 0.0)
            c1 = poisson.cdf(T1, mu=mu)
            c2 = poisson.cdf(T2, mu=mu)
            out[0], out[1], out[2] = c1, c2 - c1, 1.0 - c2

        # Negative Binomial (overdispersed)
        elif code == 8:
            valid = error_variance > best_guess
            p = np.where(valid, best_guess / error_variance, np.nan)
            n = np.where(valid, (best_guess ** 2) / (error_variance - best_guess), np.nan)
            c1 = nbinom.cdf(T1, n=n, p=p)
            c2 = nbinom.cdf(T2, n=n, p=p)
            out[0], out[1], out[2] = c1, c2 - c1, 1.0 - c2

        else:
            # Unknown code
            return out

        return out

    @staticmethod
    def calculate_tercile_probabilities_nonparametric(best_guess, error_samples, first_tercile, second_tercile):
        """
        Nonparametric terciles using historical error samples (gridwise, vectorized over T).
        """
        best_guess = np.asarray(best_guess, float)
        error_samples = np.asarray(error_samples, float)
        n_time = best_guess.size
        pred_prob = np.full((3, n_time), np.nan, float)

        for t in range(n_time):
            bg = best_guess[t]
            if not np.isfinite(bg):
                continue
            dist = bg + error_samples
            dist = dist[np.isfinite(dist)]
            if dist.size == 0:
                continue
            p_below = np.mean(dist < first_tercile)
            p_between = np.mean((dist >= first_tercile) & (dist < second_tercile))
            p_above = 1.0 - (p_below + p_between)
            pred_prob[:, t] = [p_below, p_between, p_above]
        return pred_prob

    # ------------------ Hindcast probabilities ------------------

    def compute_prob(self,
                     Predictant: xr.DataArray,
                     clim_year_start,
                     clim_year_end,
                     hindcast_det: xr.DataArray,
                     best_code_da: xr.DataArray = None,
                     best_shape_da: xr.DataArray = None,
                     best_loc_da: xr.DataArray = None,
                     best_scale_da: xr.DataArray = None) -> xr.DataArray:
        """
        Compute tercile probabilities for deterministic hindcasts.
        Returns (probability=['PB','PN','PA'], T, Y, X).
        """
        if "M" in Predictant.dims:
            Predictant = Predictant.isel(M=0).drop_vars("M").squeeze()
        Predictant = Predictant.transpose("T","Y","X")
        hindcast_det = hindcast_det.transpose("T","Y","X")

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
            if any(v is None for v in (best_code_da, best_shape_da, best_loc_da, best_scale_da)):
                raise ValueError("dist_method='bestfit' requires best_code_da, best_shape_da, best_loc_da, best_scale_da.")
            T1, T2 = xr.apply_ufunc(
                self._ppf_terciles_from_code,
                best_code_da, best_shape_da, best_loc_da, best_scale_da,
                input_core_dims=[(),(),(),()],
                output_core_dims=[(),()],
                vectorize=True, dask="parallelized",
                output_dtypes=[float, float],
            )
            hindcast_prob = xr.apply_ufunc(
                self.calculate_tercile_probabilities_bestfit,
                hindcast_det, error_variance, T1, T2, best_code_da,
                input_core_dims=[("T",), (), (), (), ()],
                output_core_dims=[("probability","T")],
                vectorize=True, dask="parallelized",
                kwargs={"dof": dof},
                output_dtypes=[float],
                dask_gufunc_kwargs={"output_sizes":{"probability":3}, "allow_rechunk": True},
            )
        elif dm == "nonparam":
            error_samples = Predictant - hindcast_det
            hindcast_prob = xr.apply_ufunc(
                self.calculate_tercile_probabilities_nonparametric,
                hindcast_det, error_samples, T1_emp, T2_emp,
                input_core_dims=[("T",), ("T",), (), ()],
                output_core_dims=[("probability","T")],
                vectorize=True, dask="parallelized",
                output_dtypes=[float],
                dask_gufunc_kwargs={"output_sizes":{"probability":3}, "allow_rechunk": True},
            )
        else:
            raise ValueError(f"Invalid dist_method: {dm}")

        hindcast_prob = hindcast_prob.assign_coords(probability=("probability", ["PB","PN","PA"]))
        return (hindcast_prob * mask).transpose("probability","T","Y","X")

    # ------------------ One-step forecast (det + probs) ------------------

    def forecast(self,
                 Predictant,
                 clim_year_start,
                 clim_year_end,
                 hindcast_det,
                 hindcast_det_cross,
                 Predictor_for_year,
                 best_params=None,
                 best_code_da=None,
                 best_shape_da=None,
                 best_loc_da=None,
                 best_scale_da=None):
        """
        One-step forecast for target period using stacking; returns (deterministic, probabilities).
        """
        # strip member dim if present
        if "M" in Predictant.dims:
            Predictant_no_m = Predictant.isel(M=0).drop_vars("M").squeeze()
        else:
            Predictant_no_m = Predictant

        Predictant_no_m = Predictant_no_m.transpose("T","Y","X")
        hindcast_det = hindcast_det.transpose("T","Y","X")
        hindcast_det_cross = hindcast_det_cross.transpose("T","Y","X")
        Predictor_for_year = Predictor_for_year.transpose("T","Y","X")

        # mask as DA (keep coords)
        mask = xr.where(~np.isnan(Predictant_no_m.isel(T=0)), 1.0, np.nan)

        # standardize training predictors/targets
        mean_val = hindcast_det.sel(T=slice(str(clim_year_start), str(clim_year_end))).mean(dim="T")
        std_val  = hindcast_det.sel(T=slice(str(clim_year_start), str(clim_year_end))).std(dim="T")

        Predictor_for_year_st = (Predictor_for_year - mean_val) / std_val
        hindcast_det_st = standardize_timeseries(hindcast_det, clim_year_start, clim_year_end)
        Predictant_st   = standardize_timeseries(Predictant_no_m, clim_year_start, clim_year_end)

        time = Predictor_for_year_st["T"]
        y_test_dummy = xr.full_like(Predictant_st.isel(T=0), np.nan).expand_dims(T=time)

        # base hparams
        if best_params is None:
            best_params = self.compute_hyperparameters(hindcast_det, Predictant_no_m, clim_year_start, clim_year_end)
        best_params_list = best_params

        # OOF & meta on standardized hindcast
        oof_preds = self._get_oof_predictions(hindcast_det_st, Predictant_st, best_params_list)
        X_cols = [oof.stack(sample=("T","Y","X")).values for oof in oof_preds]
        X_meta = np.column_stack(X_cols)
        y_meta = Predictant_st.stack(sample=("T","Y","X")).values
        nan_mask = np.any(~np.isfinite(X_meta), axis=1) | ~np.isfinite(y_meta)

        if np.all(nan_mask):
            self.meta_learner = None
        else:
            X_meta_clean = X_meta[~nan_mask]; y_meta_clean = y_meta[~nan_mask]
            if self.meta_learner_type == "linear":
                meta = LinearRegression().fit(X_meta_clean, y_meta_clean)
            else:
                if self.meta_learner_type == "ridge":
                    meta_base = Ridge(); param_dist = {"alpha": self.alpha_range}
                elif self.meta_learner_type == "lasso":
                    meta_base = Lasso(max_iter=10000); param_dist = {"alpha": self.alpha_range}
                elif self.meta_learner_type == "elasticnet":
                    meta_base = ElasticNet(max_iter=10000); param_dist = {"alpha": self.alpha_range, "l1_ratio": self.l1_ratio_range}
                else:
                    raise ValueError(f"Invalid meta_learner_type: {self.meta_learner_type}")
                search = RandomizedSearchCV(
                    meta_base, param_distributions=param_dist, n_iter=self.meta_n_iter_search,
                    cv=self.meta_cv_folds, scoring="neg_mean_squared_error",
                    random_state=self.random_state, error_score=np.nan
                )
                search.fit(X_meta_clean, y_meta_clean)
                meta = search.best_estimator_
            self.meta_learner = meta

        # base forecasts in standardized space
        base_forecast_std = []
        for i, base in enumerate(self.base_models):
            bp = best_params_list[i]
            pred_base = base.compute_model(
                hindcast_det_st, Predictant_st, Predictor_for_year_st, y_test_dummy, best_params=bp
            )
            base_forecast_std.append(pred_base)

        # meta prediction in standardized space
        lat = Predictor_for_year_st["Y"]; lon = Predictor_for_year_st["X"]
        n_time, n_lat, n_lon = len(time), len(lat), len(lon)

        if self.meta_learner is None:
            preds_std = xr.concat(base_forecast_std, dim="model").mean(dim="model", skipna=True)
        else:
            X_cols_test = [p.stack(sample=("T","Y","X")).values for p in base_forecast_std]
            X_meta_test = np.column_stack(X_cols_test)
            nan_mask_test = np.any(~np.isfinite(X_meta_test), axis=1)

            preds_std_arr = np.full((n_time, n_lat, n_lon), np.nan)
            if not np.all(nan_mask_test):
                X_meta_test_clean = X_meta_test[~nan_mask_test]
                y_pred_clean = self.meta_learner.predict(X_meta_test_clean)
                full = np.full(X_meta_test.shape[0], np.nan)
                full[~nan_mask_test] = y_pred_clean
                preds_std_arr = full.reshape(n_time, n_lat, n_lon)

            preds_std = xr.DataArray(
                preds_std_arr, coords={"T": time, "Y": lat, "X": lon}, dims=("T","Y","X")
            )

        # de-standardize to original space & apply spatial mask
        result_da = reverse_standardize(preds_std, Predictant_no_m, clim_year_start, clim_year_end) * mask

        # fix forecast time coordinate to target year/month
        year = Predictor_for_year["T"].values.astype("datetime64[Y]").astype(int)[0] + 1970
        first_T = Predictant_no_m.isel(T=0)["T"].values
        month_1 = (first_T.astype("datetime64[M]").astype(int) % 12) + 1
        new_T_value = np.datetime64(f"{year}-{int(month_1):02d}-01")
        result_da = result_da.assign_coords(T=xr.DataArray([new_T_value], dims=["T"]))
        result_da["T"] = result_da["T"].astype("datetime64[ns]")

        # ---------- probabilities ----------
        index_start = Predictant_no_m.get_index("T").get_loc(str(clim_year_start)).start
        index_end   = Predictant_no_m.get_index("T").get_loc(str(clim_year_end)).stop
        rainfall_for_tercile = Predictant_no_m.isel(T=slice(index_start, index_end))
        terciles = rainfall_for_tercile.quantile([0.33, 0.67], dim="T")
        T1_emp = terciles.isel(quantile=0).drop_vars("quantile")
        T2_emp = terciles.isel(quantile=1).drop_vars("quantile")

        error_variance = (Predictant_no_m - hindcast_det_cross).var(dim="T")
        dof = max(int(rainfall_for_tercile.sizes["T"]) - 1, 2)

        dm = self.dist_method
        if dm == "bestfit":
            if any(v is None for v in (best_code_da, best_shape_da, best_loc_da, best_scale_da)):
                raise ValueError("dist_method='bestfit' requires best_code_da, best_shape_da, best_loc_da, best_scale_da.")
            T1, T2 = xr.apply_ufunc(
                self._ppf_terciles_from_code,
                best_code_da, best_shape_da, best_loc_da, best_scale_da,
                input_core_dims=[(),(),(),()],
                output_core_dims=[(),()],
                vectorize=True, dask="parallelized",
                output_dtypes=[float, float],
            )
            forecast_prob = xr.apply_ufunc(
                self.calculate_tercile_probabilities_bestfit,
                result_da, error_variance, T1, T2, best_code_da,
                input_core_dims=[("T",), (), (), (), ()],
                output_core_dims=[("probability","T")],
                vectorize=True, dask="parallelized",
                kwargs={"dof": dof},
                output_dtypes=[float],
                dask_gufunc_kwargs={"output_sizes":{"probability":3}, "allow_rechunk": True},
            )
        elif dm == "nonparam":
            # use historical hindcast errors (hindcast_det) with observed Predictant_no_m
            error_samples = Predictant_no_m - hindcast_det
            forecast_prob = xr.apply_ufunc(
                self.calculate_tercile_probabilities_nonparametric,
                result_da, error_samples, T1_emp, T2_emp,
                input_core_dims=[("T",), ("T",), (), ()],
                output_core_dims=[("probability","T")],
                vectorize=True, dask="parallelized",
                output_dtypes=[float],
                dask_gufunc_kwargs={"output_sizes":{"probability":3}, "allow_rechunk": True},
            )
        else:
            raise ValueError(f"Invalid dist_method: {dm}")

        forecast_prob = forecast_prob.assign_coords(probability=("probability", ["PB","PN","PA"]))
        return result_da * mask, (forecast_prob * mask).transpose("probability","T","Y","X")




def _ensemble_crps(ens, obs, fair=True):
    """
    Compute the Continuous Ranked Probability Score (CRPS) for an ensemble.
    
    Parameters
    ----------
    ens : array-like
        Ensemble forecast members
    obs : float
        Observation value
    fair : bool, default=True
        Apply fair CRPS correction (m/(m-1)) for finite ensemble sizes
        
    Returns
    -------
    crps : float
        CRPS value
    """
    ens = np.asarray(ens, dtype=float)
    m = ens.size
    
    # Handle edge cases
    if m == 0:
        return np.nan
    elif m == 1:
        return np.abs(ens[0] - obs)
    
    # Traditional CRPS formula for ensemble
    term1 = np.mean(np.abs(ens - obs))
    
    # Efficient computation of pairwise differences
    if m <= 1000:  # Use matrix for small ensembles
        # This computes all pairwise absolute differences
        # Using broadcasting: ens[:, None] - ens[None, :] creates m x m matrix
        term2 = 0.5 * np.mean(np.abs(ens[:, None] - ens[None, :]))
    else:  
        # Use sorted method for O(m log m) complexity (Gneiting et al. 2005)
        sorted_ens = np.sort(ens)
        k = np.arange(1, m + 1)
        # Formula: (1/m^2) * Σ (2k - m - 1) * x_(k)
        term2 = np.sum((2 * k - m - 1) * sorted_ens) / (m ** 2)
    
    crps = term1 - term2
    
    # Fair CRPS correction for finite ensemble size (Ferro et al. 2005)
    if fair and m > 1:
        crps *= m / (m - 1.0)
        
    return crps


def _gauss_crps(obs, mu, sig):
    """
    Compute the CRPS for a Gaussian distribution.
    
    Parameters
    ----------
    obs : float or array-like
        Observation value(s)
    mu : float or array-like
        Mean of the Gaussian distribution
    sig : float or array-like
        Standard deviation of the Gaussian distribution
        
    Returns
    -------
    crps : float or array-like
        CRPS value(s)
    """
    # Ensure inputs are numpy arrays
    obs = np.asarray(obs)
    mu = np.asarray(mu)
    sig = np.asarray(sig)
    
    # Enhanced numerical stability
    eps = np.finfo(float).eps * 100
    sig = np.maximum(sig, eps)
    
    z = (obs - mu) / sig
    pdf = norm.pdf(z)
    cdf = norm.cdf(z)
    
    return sig * (z * (2 * cdf - 1) + 2 * pdf - 1 / np.sqrt(np.pi))


class WAS_mme_NGR:
    """
    Operational WAS_mme_NGR calibration for ensemble post-processing.
    
    Implements Non-homogeneous Gaussian Regression (NGR) with two variants:
    - "NGR": Minimize Gaussian CRPS (parametric)
    - "ensNGR": Minimize ensemble CRPS (non-parametric)
    
    The calibration model: 
    μ = a + b * ensemble_mean
    σ = sqrt(c² + d² * ensemble_variance)
    
    Parameters
    ----------
    type : {'NGR', 'ensNGR'}, default='NGR'
        CRPS minimization method
    apply_to : {'all', 'sig', 'pos', 'neg'}, default='all'
        When to apply calibration:
        - 'all': Always apply calibration
        - 'sig': Only apply if regression is statistically significant
        - 'pos': Only apply if positive correlation is significant
        - 'neg': Only apply if negative correlation is significant
    alpha : float, default=0.1
        Significance level for statistical tests (0 < alpha < 1)
    test_direction : {'two-sided', 'greater', 'less'}, default='two-sided'
        Direction of statistical test for regression significance
    param_bounds : dict or None, default=None
        Custom bounds for parameters [a, b, c, d]
        Example: {'a': (-10, 10), 'b': (0, 5), 'c': (0, 5), 'd': (0, 5)}
        
    Attributes
    ----------
    params : xarray.DataArray or numpy.ndarray
        Calibration parameters [a, b, c, d]
    clim_terciles : xarray.DataArray or numpy.ndarray
        Climate terciles computed from observations
    fitted : bool
        Whether the model has been fitted
    """
    
    # Default parameter bounds ensuring physical plausibility
    DEFAULT_BOUNDS = {
        'a': (-np.inf, np.inf),  # Intercept can be any value
        'b': (0.0, np.inf),      # Regression coefficient should be non-negative
        'c': (0.0, np.inf),      # Spread adjustment should be non-negative
        'd': (0.0, np.inf),      # Spread scaling should be non-negative
    }
    
    def __init__(self, type="NGR", apply_to="all", alpha=0.1, 
                 test_direction="two-sided", param_bounds=None):
        # Validate inputs
        if type not in {"NGR", "ensNGR"}:
            raise ValueError("type must be 'NGR' or 'ensNGR'")
        if apply_to not in {"all", "sig", "pos", "neg"}:
            raise ValueError("apply_to must be 'all', 'sig', 'pos', or 'neg'")
        if not 0 < alpha < 1:
            raise ValueError("alpha must be between 0 and 1")
        if test_direction not in {"two-sided", "greater", "less"}:
            raise ValueError("test_direction must be 'two-sided', 'greater', or 'less'")
            
        self.type = type
        self.apply_to = apply_to
        self.alpha = alpha
        self.test_direction = test_direction
        self.is_gaussian = (type == "NGR")
        self.is_ensemble = (type == "ensNGR")
        
        # Set parameter bounds
        self.param_bounds = self.DEFAULT_BOUNDS.copy()
        if param_bounds:
            for param, bounds in param_bounds.items():
                if param in self.param_bounds:
                    self.param_bounds[param] = bounds
        
        # Initialize state
        self.params = None
        self.clim_terciles = None
        self.fitted = False
        self._xarray = False
        self._optimization_stats = None
        self.attrs = {}
        
        # Dimension names (configurable)
        self._param_dim = 'parameter'
        self._lat_dim = 'Y'
        self._lon_dim = 'X'
        self._time_dim = 'T'
        self._member_dim = 'M'
        self._lat_coords = None
        self._lon_coords = None
        
    def __repr__(self):
        status = "fitted" if self.fitted else "unfitted"
        return (f"WAS_mme_NGR(type='{self.type}', apply_to='{self.apply_to}', "
                f"alpha={self.alpha}, {status})")
    
    def _create_objective_function(self, fmn, fsd, fanom, o_train):
        """Create objective function for parameter optimization."""
        n_times = len(o_train)
        
        if self.is_ensemble:
            def obj(pars):
                a, b, c, d = pars
                mu = a + b * fmn
                sigma = np.sqrt(c**2 + d**2 * fsd**2)
                
                # Vectorized ensemble CRPS calculation
                cal_ens = mu[None, :] + sigma[None, :] * fanom
                
                # Compute CRPS for all time steps at once
                crps_vals = np.zeros(n_times)
                for k in range(n_times):
                    crps_vals[k] = _ensemble_crps(cal_ens[:, k], o_train[k], fair=True)
                
                return np.mean(crps_vals)
        else:
            def obj(pars):
                a, b, c, d = pars
                mu = a + b * fmn
                sigma = np.sqrt(c**2 + d**2 * fsd**2)
                return np.mean(_gauss_crps(o_train, mu, sigma))
        
        return obj
    
    def _test_regression_significance(self, r, n, test_direction):
        """Test if regression correlation is statistically significant."""
        if n <= 2:
            return False
            
        # Handle perfect correlation
        if np.abs(r) >= 1.0 - 1e-12:
            return True
            
        # Compute t-statistic
        t_stat = r * np.sqrt((n - 2) / max(1 - r**2, 1e-12))
        
        # Compute p-value based on test direction
        if test_direction == "two-sided":
            p_value = 2 * tdist.sf(np.abs(t_stat), n - 2)
        elif test_direction == "greater":
            p_value = tdist.sf(t_stat, n - 2)
        else:  # "less"
            p_value = tdist.cdf(t_stat, n - 2)
            
        return p_value < self.alpha
    
    def _apply_calibration_decision(self, lm, n_valid):
        """Decide whether to apply calibration based on statistical test."""
        if self.apply_to == "all":
            return True
        elif self.apply_to == "sig":
            return self._test_regression_significance(lm.rvalue, n_valid, self.test_direction)
        elif self.apply_to == "pos":
            return (lm.rvalue > 0 and 
                    self._test_regression_significance(lm.rvalue, n_valid, "greater"))
        else:  # "neg"
            return (lm.rvalue < 0 and 
                    self._test_regression_significance(lm.rvalue, n_valid, "less"))
    
    def fit(self, hcst_grid, obs_grid, clim_terciles=False,
            member_dim='M', time_dim='T', lat_dim='Y', lon_dim='X',
            show_progress=True):
        """
        Fit calibration parameters using hindcast data.
        
        Parameters
        ----------
        hcst_grid : xarray.DataArray or numpy.ndarray
            Hindcast ensemble data with dimensions (member, time, lat, lon)
        obs_grid : xarray.DataArray or numpy.ndarray
            Observations with dimensions (time, lat, lon)
        clim_terciles : bool, default=False
            Compute climate terciles from observations
        member_dim, time_dim, lat_dim, lon_dim : str
            Dimension names
        show_progress : bool, default=True
            Show progress bar during fitting
            
        Returns
        -------
        self : WAS_mme_NGR
            Fitted model
        """
        # Check if xarray is available and inputs are DataArrays
        use_xarray = (xr is not None and 
                     isinstance(hcst_grid, xr.DataArray) and 
                     isinstance(obs_grid, xr.DataArray))
        
        if use_xarray:
            # Ensure correct dimension order and extract values
            hcst = hcst_grid.transpose(member_dim, time_dim, lat_dim, lon_dim).values
            obs = obs_grid.transpose(time_dim, lat_dim, lon_dim).values
            
            # Store metadata
            self._lat_dim = lat_dim
            self._lon_dim = lon_dim
            self._time_dim = time_dim
            self._member_dim = member_dim
            self._lat_coords = hcst_grid.coords[lat_dim]
            self._lon_coords = hcst_grid.coords[lon_dim]
            self.attrs = hcst_grid.attrs.copy()
            self._xarray = True
        else:
            # Convert to numpy arrays
            hcst = np.asarray(hcst_grid, dtype=float)
            obs = np.asarray(obs_grid, dtype=float)
            
            # Store dimension names
            self._lat_dim = lat_dim
            self._lon_dim = lon_dim
            self._time_dim = time_dim
            self._member_dim = member_dim
        
        # Get dimensions
        nmemb, ntimes, nlat, nlon = hcst.shape
        
        # Validate shapes
        if obs.shape != (ntimes, nlat, nlon):
            raise ValueError(f"obs_grid must have shape (time, lat, lon) = ({ntimes}, {nlat}, {nlon})")
        
        # Initialize parameter array and optimization statistics
        params = np.full((4, nlat, nlon), np.nan)
        opt_stats = {
            'success': np.zeros((nlat, nlon), dtype=bool),
            'iterations': np.zeros((nlat, nlon), dtype=int),
            'crps_reduction': np.zeros((nlat, nlon), dtype=float),
            'final_crps': np.zeros((nlat, nlon), dtype=float)
        }
        
        # Compute climate terciles if requested
        if clim_terciles:
            # Compute terciles along time dimension
            terc_np = np.nanpercentile(obs, [100/3, 200/3], axis=0)
            if use_xarray:
                self.clim_terciles = xr.DataArray(
                    terc_np,
                    dims=('tercile', self._lat_dim, self._lon_dim),
                    coords={
                        'tercile': ['lower', 'upper'],
                        self._lat_dim: self._lat_coords,
                        self._lon_dim: self._lon_coords
                    }
                )
            else:
                self.clim_terciles = terc_np
        
        # Prepare bounds for optimization
        bounds = [
            self.param_bounds['a'],
            self.param_bounds['b'],
            self.param_bounds['c'],
            self.param_bounds['d']
        ]
        
        # Fit parameters for each grid point
        if show_progress:
            print(f"Fitting WAS_mme_NGR parameters ({self.type})...")
            print(f"Grid: {nlat} x {nlon} ({nlat * nlon} points)")
            lat_iter = tqdm(range(nlat), desc="Latitude")
        else:
            lat_iter = range(nlat)
        
        for ilat in lat_iter:
            for ilon in range(nlon):
                try:
                    # Extract data for this grid point
                    o_gp = obs[:, ilat, ilon].copy()
                    h_gp = hcst[:, :, ilat, ilon].copy()
                    
                    # Find valid (non-NaN) time steps
                    valid = ~np.isnan(o_gp)
                    n_valid = valid.sum()
                    
                    # Need at least 3 valid points for regression
                    if n_valid < 3:
                        params[:, ilat, ilon] = [0.0, 1.0, 0.0, 1.0]  # No calibration
                        opt_stats['success'][ilat, ilon] = False
                        continue
                    
                    # Extract valid data
                    o_train = o_gp[valid]
                    h_gp_valid = h_gp[:, valid]
                    
                    # Compute ensemble statistics for valid time steps
                    ens_mean = np.nanmean(h_gp_valid, axis=0)
                    sigma_e = np.nanstd(h_gp_valid, axis=0, ddof=1)
                    
                    # Standardize ensemble anomalies with numerical stability
                    eps = np.finfo(float).eps * 100
                    sigma_e_safe = np.maximum(sigma_e, eps)
                    anom = (h_gp_valid - ens_mean[np.newaxis, :]) / sigma_e_safe[np.newaxis, :]
                    
                    # Extract training data
                    fmn = ens_mean
                    fsd = sigma_e
                    fanom = anom
                    
                    # Initial guess from linear regression
                    mask_lm = ~np.isnan(fmn)
                    if mask_lm.sum() < 3:
                        params[:, ilat, ilon] = [0.0, 1.0, 0.0, 1.0]
                        opt_stats['success'][ilat, ilon] = False
                        continue
                    
                    lm = linregress(fmn[mask_lm], o_train[mask_lm])
                    initial = np.array([lm.intercept, lm.slope, 0.0, 1.0])
                    
                    # Ensure initial b parameter is non-negative
                    initial[1] = max(initial[1], 0.0)
                    
                    # Decide whether to apply calibration
                    if not self._apply_calibration_decision(lm, mask_lm.sum()):
                        params[:, ilat, ilon] = [0.0, 1.0, 0.0, 1.0]
                        opt_stats['success'][ilat, ilon] = True  # Considered successful (no calibration needed)
                        continue
                    
                    # Create objective function
                    obj_func = self._create_objective_function(fmn, fsd, fanom, o_train)
                    
                    # Compute initial CRPS (no calibration)
                    initial_crps = obj_func([0.0, 1.0, 0.0, 1.0])
                    
                    # Optimize parameters
                    res = minimize(
                        obj_func, 
                        initial, 
                        method="L-BFGS-B",  # Supports bounds
                        bounds=bounds,
                        options={
                            "maxiter": 1000, 
                            "ftol": 1e-8,
                            "gtol": 1e-6,
                            "disp": False
                        }
                    )
                    
                    # Store results
                    params[:, ilat, ilon] = res.x
                    opt_stats['success'][ilat, ilon] = res.success
                    opt_stats['iterations'][ilat, ilon] = res.nit
                    opt_stats['final_crps'][ilat, ilon] = res.fun if res.success else initial_crps
                    
                    # Compute CRPS reduction
                    if initial_crps > 0:
                        crps_reduction = (initial_crps - opt_stats['final_crps'][ilat, ilon]) / initial_crps
                    else:
                        crps_reduction = 0.0
                    opt_stats['crps_reduction'][ilat, ilon] = crps_reduction
                    
                except Exception as e:
                    # Fallback to no calibration
                    params[:, ilat, ilon] = [0.0, 1.0, 0.0, 1.0]
                    opt_stats['success'][ilat, ilon] = False
        
        # Store parameters
        if use_xarray:
            self.params = xr.DataArray(
                params,
                dims=(self._param_dim, self._lat_dim, self._lon_dim),
                coords={
                    self._param_dim: ['a', 'b', 'c', 'd'],
                    self._lat_dim: self._lat_coords,
                    self._lon_dim: self._lon_coords
                },
                attrs=self.attrs
            )
        else:
            self.params = params
        
        # Store optimization statistics
        self._optimization_stats = opt_stats
        self.fitted = True
        
        # Print summary statistics
        if show_progress:
            success_rate = np.mean(opt_stats['success']) * 100
            success_mask = opt_stats['success']
            if np.any(success_mask):
                avg_crps_reduction = np.nanmean(opt_stats['crps_reduction'][success_mask]) * 100
                avg_iterations = np.nanmean(opt_stats['iterations'][success_mask])
                print(f"Fit complete: Success rate = {success_rate:.1f}%, "
                      f"Average CRPS reduction = {avg_crps_reduction:.1f}%, "
                      f"Avg iterations = {avg_iterations:.1f}")
            else:
                print(f"Fit complete: Success rate = {success_rate:.1f}%")
        
        return self
    
    def transform(self, fcst_grid, member_dim='M', time_dim='T', lat_dim='Y', lon_dim='X'):
        """
        Apply calibration to forecast data (returns calibrated ensemble only).
        
        Parameters
        ----------
        fcst_grid : xarray.DataArray or numpy.ndarray
            Forecast ensemble data
        member_dim, time_dim, lat_dim, lon_dim : str
            Dimension names
            
        Returns
        -------
        cal_ens : xarray.DataArray or numpy.ndarray
            Calibrated ensemble
        mu : xarray.DataArray or numpy.ndarray
            Calibrated mean
        sigma : xarray.DataArray or numpy.ndarray
            Calibrated standard deviation
        """
        result = self.predict(
            fcst_grid,
            quantiles=None,
            clim_terciles=False,
            parametric=None,
            member_dim=member_dim,
            time_dim=time_dim,
            lat_dim=lat_dim,
            lon_dim=lon_dim
        )
        
        if self._xarray:
            return result['calibrated_ensemble'], result['calibrated_mean'], result['calibrated_std']
        else:
            return result[0], result[1], result[2]
    
    def predict(self, fcst_grid, quantiles=None, clim_terciles=False,
                parametric=None, member_dim='M', time_dim='T', 
                lat_dim='Y', lon_dim='X', show_progress=True):
        """
        Apply calibration and compute requested outputs.
        
        Parameters
        ----------
        fcst_grid : xarray.DataArray or numpy.ndarray
            Forecast ensemble data
        quantiles : list of float or None, default=None
            Quantile levels to compute (e.g., [0.05, 0.5, 0.95])
        clim_terciles : bool, default=False
            Compute tercile probabilities
        parametric : bool or None, default=None
            Use parametric (Gaussian) method for quantiles/terciles
            If None, uses the model type (NGR=parametric, ensNGR=non-parametric)
        member_dim, time_dim, lat_dim, lon_dim : str
            Dimension names
        show_progress : bool, default=True
            Show progress bar
            
        Returns
        -------
        results : xarray.Dataset or tuple
            Calibrated products
        """
        if not self.fitted:
            raise RuntimeError("Model must be fitted before prediction. Call fit() first.")
        
        # Determine whether to use parametric method
        if parametric is None:
            parametric = self.is_gaussian
        
        # Check what outputs are needed
        need_ensemble = False
        if quantiles is not None and not parametric:
            need_ensemble = True
        if clim_terciles and not parametric:
            need_ensemble = True
        if not any([need_ensemble, quantiles is not None, clim_terciles]):
            need_ensemble = True  # At least return the calibrated ensemble
        
        # Handle xarray or numpy input
        use_xarray = (xr is not None and isinstance(fcst_grid, xr.DataArray))
        
        if use_xarray:
            fcst_trans = fcst_grid.transpose(member_dim, time_dim, lat_dim, lon_dim)
            fcst = fcst_trans.values
            time_coords = fcst_trans.coords[time_dim]
            member_coords = fcst_trans.coords[member_dim]
            attrs = fcst_grid.attrs.copy()
            name = fcst_grid.name or "forecast"
        else:
            fcst = np.asarray(fcst_grid, dtype=float)
        
        nmemb, ntimes, nlat, nlon = fcst.shape
        
        # Validate parameter dimensions
        if self.params.shape[1:] != (nlat, nlon):
            raise ValueError(f"Parameter dimensions {self.params.shape[1:]} don't match forecast dimensions {nlat, nlon}")
        
        # Initialize output arrays
        cal_ens = np.full(fcst.shape, np.nan) if need_ensemble else None
        mu = np.full((ntimes, nlat, nlon), np.nan)
        sigma = np.full_like(mu, np.nan)
        
        # Extract parameters for fast access
        if hasattr(self.params, 'values'):
            param_np = self.params.values
        else:
            param_np = self.params
        
        # Apply calibration to each grid point
        if show_progress:
            print("Applying calibration to forecast...")
            lat_iter = tqdm(range(nlat), desc="Latitude")
        else:
            lat_iter = range(nlat)
        
        for ilat in lat_iter:
            for ilon in range(nlon):
                try:
                    f_gp = fcst[:, :, ilat, ilon].copy()
                    
                    # Skip if all data is NaN
                    if np.isnan(f_gp).all():
                        continue
                    
                    # Compute ensemble statistics
                    ens_mean = np.nanmean(f_gp, axis=0)
                    sigma_e = np.nanstd(f_gp, axis=0, ddof=1)
                    
                    # Handle cases where std is zero
                    eps = np.finfo(float).eps * 100
                    sigma_e_safe = np.maximum(sigma_e, eps)
                    
                    # Standardize anomalies
                    anom = (f_gp - ens_mean[np.newaxis, :]) / sigma_e_safe[np.newaxis, :]
                    
                    # Get parameters for this grid point
                    a, b, c, d = param_np[:, ilat, ilon]
                    
                    # Check for NaN parameters (no calibration applied)
                    if np.isnan(a):
                        mu[:, ilat, ilon] = ens_mean
                        sigma[:, ilat, ilon] = sigma_e
                        if need_ensemble:
                            cal_ens[:, :, ilat, ilon] = f_gp
                    else:
                        # Apply calibration
                        mu[:, ilat, ilon] = a + b * ens_mean
                        sigma[:, ilat, ilon] = np.sqrt(c**2 + d**2 * sigma_e**2)
                        if need_ensemble:
                            cal_ens[:, :, ilat, ilon] = (mu[:, ilat, ilon][None, :] + 
                                                         sigma[:, ilat, ilon][None, :] * anom)
                            
                except Exception:
                    # Fallback to raw forecast
                    mu[:, ilat, ilon] = np.nanmean(fcst[:, :, ilat, ilon], axis=0)
                    sigma[:, ilat, ilon] = np.nanstd(fcst[:, :, ilat, ilon], axis=0, ddof=1)
                    if need_ensemble:
                        cal_ens[:, :, ilat, ilon] = fcst[:, :, ilat, ilon]
        
        # Package results
        if use_xarray:
            ds = xr.Dataset(attrs=attrs)
            
            if need_ensemble:
                ens_da = xr.DataArray(
                    cal_ens,
                    dims=(member_dim, time_dim, lat_dim, lon_dim),
                    coords={
                        member_dim: member_coords,
                        time_dim: time_coords,
                        lat_dim: self._lat_coords,
                        lon_dim: self._lon_coords
                    },
                    name='calibrated_ensemble'
                )
                ds['calibrated_ensemble'] = ens_da
            
            # Add mean and std
            mu_da = xr.DataArray(
                mu,
                dims=(time_dim, lat_dim, lon_dim),
                coords={
                    time_dim: time_coords,
                    lat_dim: self._lat_coords,
                    lon_dim: self._lon_coords
                },
                name='calibrated_mean'
            )
            ds['calibrated_mean'] = mu_da
            
            sigma_da = xr.DataArray(
                sigma,
                dims=(time_dim, lat_dim, lon_dim),
                coords=mu_da.coords,
                name='calibrated_std'
            )
            ds['calibrated_std'] = sigma_da
            
            # Compute quantiles if requested
            if quantiles is not None:
                q_levels = np.asarray(quantiles)
                if parametric:
                    # Parametric quantiles from Gaussian distribution
                    quant_np = norm.ppf(q_levels)[:, None, None, None] * sigma[None, ...] + mu[None, ...]
                else:
                    # Empirical quantiles from calibrated ensemble
                    quant_np = np.nanquantile(cal_ens, q_levels, axis=0)
                
                quant_da = xr.DataArray(
                    quant_np,
                    dims=('quantile', time_dim, lat_dim, lon_dim),
                    coords={
                        'quantile': list(q_levels),
                        time_dim: time_coords,
                        lat_dim: self._lat_coords,
                        lon_dim: self._lon_coords
                    },
                    name='calibrated_quantiles'
                )
                ds['calibrated_quantiles'] = quant_da
            
            # Compute tercile probabilities if requested
            if clim_terciles:
                if self.clim_terciles is None:
                    raise RuntimeError("clim_terciles=True in fit() required for tercile probabilities")
                
                # Extract climate terciles
                if hasattr(self.clim_terciles, 'isel'):
                    lower_np = self.clim_terciles.isel(tercile=0).values
                    upper_np = self.clim_terciles.isel(tercile=1).values
                else:
                    lower_np = self.clim_terciles[0]
                    upper_np = self.clim_terciles[1]
                
                if parametric:
                    # Parametric probabilities from Gaussian distribution
                    p_below = norm.cdf(lower_np[np.newaxis, :, :], loc=mu, scale=sigma)
                    p_above = norm.sf(upper_np[np.newaxis, :, :], loc=mu, scale=sigma)
                    p_near = 1.0 - p_below - p_above
                else:
                    # Empirical probabilities from calibrated ensemble
                    lower_b = lower_np[None, None, :, :]
                    upper_b = upper_np[None, None, :, :]
                    p_below = np.nanmean(cal_ens < lower_b, axis=0)
                    p_above = np.nanmean(cal_ens > upper_b, axis=0)
                    p_near = 1.0 - p_below - p_above
                
                # Package probabilities
                cat_np = np.stack([p_below, p_near, p_above])
                cat_da = xr.DataArray(
                    cat_np,
                    dims=('probability', time_dim, lat_dim, lon_dim),
                    coords={
                        'probability': ['PB', 'PN', 'PA'],
                        time_dim: time_coords,
                        lat_dim: self._lat_coords,
                        lon_dim: self._lon_coords
                    },
                    name='tercile_probability'
                )
                ds['tercile_probability'] = cat_da
            
            return ds
        
        else:
            # Return numpy arrays
            results = []
            
            if need_ensemble:
                results.append(cal_ens)
            
            results.append(mu)
            results.append(sigma)
            
            if quantiles is not None:
                q_levels = np.asarray(quantiles)
                if parametric:
                    quant_np = norm.ppf(q_levels)[:, None, None, None] * sigma[None, ...] + mu[None, ...]
                else:
                    quant_np = np.nanquantile(cal_ens, q_levels, axis=0)
                results.append(quant_np)
            
            if clim_terciles:
                if self.clim_terciles is None:
                    raise RuntimeError("clim_terciles=True in fit() required")
                
                if hasattr(self.clim_terciles, 'isel'):
                    lower_np = self.clim_terciles.isel(tercile=0).values
                    upper_np = self.clim_terciles.isel(tercile=1).values
                else:
                    lower_np = self.clim_terciles[0]
                    upper_np = self.clim_terciles[1]
                
                if parametric:
                    p_below = norm.cdf(lower_np[np.newaxis, :, :], loc=mu, scale=sigma)
                    p_above = norm.sf(upper_np[np.newaxis, :, :], loc=mu, scale=sigma)
                    p_near = 1.0 - p_below - p_above
                else:
                    lower_b = lower_np[None, None, :, :]
                    upper_b = upper_np[None, None, :, :]
                    p_below = np.nanmean(cal_ens < lower_b, axis=0)
                    p_above = np.nanmean(cal_ens > upper_b, axis=0)
                    p_near = 1.0 - p_below - p_above
                
                cat_np = np.stack([p_below, p_near, p_above])
                results.append(cat_np)
            
            return tuple(results) if len(results) > 1 else results[0]
    
    def get_optimization_stats(self):
        """
        Get optimization statistics from fitting.
        
        Returns
        -------
        stats : dict or None
            Dictionary containing optimization statistics if model is fitted
        """
        if not self.fitted:
            warnings.warn("Model not fitted yet. Call fit() first.")
            return None
        
        stats = self._optimization_stats.copy()
        
        # Convert to xarray if using xarray
        if self._xarray and self.params is not None:
            for key, value in stats.items():
                if isinstance(value, np.ndarray):
                    stats[key] = xr.DataArray(
                        value,
                        dims=(self._lat_dim, self._lon_dim),
                        coords={
                            self._lat_dim: self._lat_coords,
                            self._lon_dim: self._lon_coords
                        },
                        name=key
                    )
        
        return stats
    
    def summary(self):
        """
        Print summary of the fitted model.
        """
        if not self.fitted:
            print("Model not fitted.")
            return
        
        print(f"WAS_mme_NGR Model Summary")
        print(f"========================")
        print(f"Type: {self.type}")
        print(f"Apply to: {self.apply_to}")
        print(f"Alpha: {self.alpha}")
        print(f"Test direction: {self.test_direction}")
        print()
        
        if self._optimization_stats:
            success_rate = np.mean(self._optimization_stats['success']) * 100
            valid_mask = self._optimization_stats['success']
            if np.any(valid_mask):
                avg_crps_reduction = np.nanmean(self._optimization_stats['crps_reduction'][valid_mask]) * 100
                avg_iterations = np.nanmean(self._optimization_stats['iterations'][valid_mask])
                print(f"Optimization Statistics:")
                print(f"  Success rate: {success_rate:.1f}%")
                print(f"  Avg CRPS reduction: {avg_crps_reduction:.1f}%")
                print(f"  Avg iterations: {avg_iterations:.1f}")
            else:
                print("No successful optimizations.")
        
        if self.params is not None:
            print(f"\nParameter Statistics:")
            param_names = ['a', 'b', 'c', 'd']
            for i, name in enumerate(param_names):
                if hasattr(self.params, 'values'):
                    param_data = self.params.values[i]
                else:
                    param_data = self.params[i]
                valid_params = param_data[~np.isnan(param_data)]
                if len(valid_params) > 0:
                    print(f"  {name}: mean={np.mean(valid_params):.3f}, "
                          f"std={np.std(valid_params):.3f}, "
                          f"range=({np.min(valid_params):.3f}, {np.max(valid_params):.3f})")


# Optional: Add parallel processing support for large grids
try:
    import numba as nb
    
    @nb.njit(parallel=True, fastmath=True)
    def _apply_calibration_numba(fcst, params, mu, sigma, cal_ens):
        """Numba-accelerated calibration application."""
        nmemb, ntimes, nlat, nlon = fcst.shape
        
        for ilat in nb.prange(nlat):
            for ilon in range(nlon):
                # Get parameters for this grid point
                a, b, c, d = params[0, ilat, ilon], params[1, ilat, ilon], params[2, ilat, ilon], params[3, ilat, ilon]
                
                for itime in range(ntimes):
                    # Extract ensemble members for this time and grid point
                    members = fcst[:, itime, ilat, ilon]
                    
                    # Check for NaN values
                    valid = ~np.isnan(members)
                    n_valid = np.sum(valid)
                    
                    if n_valid == 0:
                        mu[itime, ilat, ilon] = np.nan
                        sigma[itime, ilat, ilon] = np.nan
                        if cal_ens is not None:
                            for imemb in range(nmemb):
                                cal_ens[imemb, itime, ilat, ilon] = np.nan
                        continue
                    
                    # Compute mean and std of valid members
                    valid_members = members[valid]
                    ens_mean = np.mean(valid_members)
                    ens_std = np.std(valid_members, ddof=1)
                    
                    # Avoid division by zero
                    if ens_std < 1e-12:
                        ens_std = 1.0
                    
                    # Compute calibrated mean and std
                    mu_val = a + b * ens_mean
                    sigma_val = np.sqrt(c**2 + d**2 * ens_std**2)
                    
                    mu[itime, ilat, ilon] = mu_val
                    sigma[itime, ilat, ilon] = sigma_val
                    
                    # Generate calibrated ensemble if needed
                    if cal_ens is not None:
                        # Standardize anomalies
                        anom = (valid_members - ens_mean) / ens_std
                        cal_members = mu_val + sigma_val * anom
                        
                        # Fill calibrated members back
                        idx = 0
                        for imemb in range(nmemb):
                            if valid[imemb]:
                                cal_ens[imemb, itime, ilat, ilon] = cal_members[idx]
                                idx += 1
                            else:
                                cal_ens[imemb, itime, ilat, ilon] = np.nan
        
        return mu, sigma, cal_ens
    
    def predict_fast(self, fcst_grid, **kwargs):
        """Accelerated prediction using Numba (if available)."""
        if not self.fitted:
            raise RuntimeError("Model must be fitted before prediction.")
        
        # Fall back to regular predict if Numba not available or inputs are xarray
        if xr is not None and isinstance(fcst_grid, xr.DataArray):
            return self.predict(fcst_grid, **kwargs)
        
        # Extract arrays
        fcst = np.asarray(fcst_grid, dtype=float)
        if hasattr(self.params, 'values'):
            params = self.params.values
        else:
            params = self.params
        
        nmemb, ntimes, nlat, nlon = fcst.shape
        
        # Validate parameter dimensions
        if params.shape[1:] != (nlat, nlon):
            raise ValueError(f"Parameter dimensions {params.shape[1:]} don't match forecast dimensions {nlat, nlon}")
        
        # Initialize output arrays
        mu = np.full((ntimes, nlat, nlon), np.nan)
        sigma = np.full_like(mu, np.nan)
        cal_ens = np.full(fcst.shape, np.nan)
        
        # Apply calibration using Numba
        mu, sigma, cal_ens = _apply_calibration_numba(fcst, params, mu, sigma, cal_ens)
        
        return cal_ens, mu, sigma
    
    # Add the method to the class
    WAS_mme_NGR.predict_fast = predict_fast
    
except ImportError:
    # Numba not available, use regular implementation
    pass

######################################################################################################################################################################################################################################################################################

#################################################################################################################################################### MVA ############################################################

from typing import Optional, Literal

# Optional SciPy (required for bestfit path)
try:
    from scipy.stats import norm, lognorm, expon, gamma, weibull_min, t, poisson, nbinom
    from scipy.optimize import fsolve
    from scipy.special import gamma as gamma_function
    _HAS_SCIPY = True
except Exception:
    _HAS_SCIPY = False


# -----------------------------
# Per-cell helpers for ufuncs
# -----------------------------
def _train_cell_mva(fc_train: np.ndarray,  # (M, T)
                    ob_train: np.ndarray   # (T,)
                    ) -> np.ndarray:
    """
    Compute MVA parameters for one (Y,X) cell over training period.
    Returns [clim_obs, clim_fcst, sigma_e, sigma_ref]
    """
    clim_obs  = np.nanmean(ob_train)
    clim_fcst = np.nanmean(fc_train)                  # over all members×times
    sigma_e   = np.nanstd(fc_train, ddof=1)           # over all members×times
    sigma_ref = np.nanstd(ob_train, ddof=1)           # over time
    return np.array([clim_obs, clim_fcst, sigma_e, sigma_ref], dtype=np.float64)


def _apply_cell_mva(fc_new: np.ndarray,      # (M, Tnew) or (M,)
                    clim_obs: float,
                    clim_fcst: float,
                    sigma_e: float,
                    sigma_ref: float) -> np.ndarray:
    """
    Apply MVA to forecast members (Y,X cell). Vectorized over time.
    Fallback (ill-conditioned): climatology-only shift: (F - clim_fcst) + clim_obs
    """
    if fc_new.ndim == 1:
        fc_new = fc_new[:, None]
    if (not np.isfinite(sigma_e)) or (sigma_e <= 0.0) or (not np.isfinite(sigma_ref)) or (sigma_ref <= 0.0):
        return (fc_new - clim_fcst + clim_obs).astype(fc_new.dtype, copy=False)

    scale = sigma_ref / sigma_e
    out = (fc_new - clim_fcst) * scale + clim_obs
    return out.astype(fc_new.dtype, copy=False)


def _loocv_cell_mva(fc_cell: np.ndarray,  # (M, T)
                    ob_cell: np.ndarray   # (T,)
                    ) -> np.ndarray:
    """
    LOOCV over T for one (Y,X) cell.
    For each held-out t, train on T\\{t} and apply to column t.
    """
    M, T = fc_cell.shape
    out = np.full((M, T), np.nan, dtype=fc_cell.dtype)

    for t in range(T):
        mask = np.ones(T, dtype=bool); mask[t] = False
        fc_tr = fc_cell[:, mask]
        ob_tr = ob_cell[mask]

        clim_obs, clim_fcst, sigma_e, sigma_ref = _train_cell_mva(fc_tr, ob_tr)
        col = fc_cell[:, t]  # (M,)
        out[:, t] = _apply_cell_mva(col, clim_obs, clim_fcst, sigma_e, sigma_ref)[:, 0]
    return out


# -----------------------------
# Probability helpers (shared)
# -----------------------------
def _ppf_terciles_from_code(dist_code, shape, loc, scale):
    """
    Return terciles (T1, T2) from best-fit distribution parameters.
    Codes:
        1: norm, 2: lognorm, 3: expon, 4: gamma, 5: weibull_min,
        6: t,    7: poisson, 8: nbinom
    """
    if (not _HAS_SCIPY) or np.isnan(dist_code):
        return np.nan, np.nan

    code = int(dist_code)
    try:
        if code == 1:
            return (float(norm.ppf(0.33, loc=loc, scale=scale)),
                    float(norm.ppf(0.67, loc=loc, scale=scale)))
        elif code == 2:
            return (float(lognorm.ppf(0.33, s=shape, loc=loc, scale=scale)),
                    float(lognorm.ppf(0.67, s=shape, loc=loc, scale=scale)))
        elif code == 3:
            return (float(expon.ppf(0.33, loc=loc, scale=scale)),
                    float(expon.ppf(0.67, loc=loc, scale=scale)))
        elif code == 4:
            return (float(gamma.ppf(0.33, a=shape, loc=loc, scale=scale)),
                    float(gamma.ppf(0.67, a=shape, loc=loc, scale=scale)))
        elif code == 5:
            return (float(weibull_min.ppf(0.33, c=shape, loc=loc, scale=scale)),
                    float(weibull_min.ppf(0.67, c=shape, loc=loc, scale=scale)))
        elif code == 6:
            return (float(t.ppf(0.33, df=shape, loc=loc, scale=scale)),
                    float(t.ppf(0.67, df=shape, loc=loc, scale=scale)))
        elif code == 7:
            return (float(poisson.ppf(0.33, mu=shape, loc=loc)),
                    float(poisson.ppf(0.67, mu=shape, loc=loc)))
        elif code == 8:
            return (float(nbinom.ppf(0.33, n=shape, p=scale, loc=loc)),
                    float(nbinom.ppf(0.67, n=shape, p=scale, loc=loc)))
    except Exception:
        return np.nan, np.nan
    return np.nan, np.nan


def _weibull_shape_solver(k, M, V):
    """Root of Weibull shape 'k' so that Var/Mean^2 matches V/M^2."""
    if k <= 0:
        return -np.inf
    try:
        g1 = gamma_function(1.0 + 1.0/k)
        g2 = gamma_function(1.0 + 2.0/k)
        implied = (g2 / (g1**2)) - 1.0
        observed = V / (M**2)
        return observed - implied
    except Exception:
        return -np.inf


def _calc_tercile_probs_bestfit(best_guess, error_variance, T1, T2, dist_code, dof):
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

    
    if not _HAS_SCIPY:
        n_time = np.asarray(best_guess).size
        return np.full((3, n_time), np.nan, float)

    best_guess = np.asarray(best_guess, float)
    ev = float(error_variance)
    n_time = best_guess.size
    out = np.full((3, n_time), np.nan, float)

    if (np.all(np.isnan(best_guess))
        or np.isnan(dist_code) or np.isnan(T1) or np.isnan(T2)
        or not np.isfinite(ev) or ev < 0.0):
        return out

    code = int(dist_code)

    if code == 1:  # Normal
        sd = np.sqrt(max(0.0, ev))
        out[0, :] = norm.cdf(T1, loc=best_guess, scale=sd)
        out[1, :] = norm.cdf(T2, loc=best_guess, scale=sd) - norm.cdf(T1, loc=best_guess, scale=sd)
        out[2, :] = 1.0 - norm.cdf(T2, loc=best_guess, scale=sd)

    elif code == 2:  # Lognormal
        mu_x = np.maximum(best_guess, 1e-12)
        var_x = np.maximum(ev, 1e-24)
        sigma = np.sqrt(np.log(1.0 + var_x / (mu_x**2)))
        mu = np.log(mu_x) - 0.5 * sigma**2
        out[0, :] = lognorm.cdf(T1, s=sigma, scale=np.exp(mu))
        out[1, :] = lognorm.cdf(T2, s=sigma, scale=np.exp(mu)) - lognorm.cdf(T1, s=sigma, scale=np.exp(mu))
        out[2, :] = 1.0 - lognorm.cdf(T2, s=sigma, scale=np.exp(mu))

    elif code == 3:  # Exponential
        sd = np.sqrt(max(0.0, ev))
        out[0, :] = expon.cdf(T1, loc=best_guess, scale=sd)
        out[1 :] = expon.cdf(T2, loc=best_guess, scale=sd) - expon.cdf(T1, loc=best_guess, scale=sd)
        out[2, :] = 1.0 - expon.cdf(T2, loc=best_guess, scale=sd)

    elif code == 4:  # Gamma
        mu_x = np.maximum(best_guess, 1e-12)
        var_x = np.maximum(ev, 1e-24)
        alpha = mu_x**2 / var_x
        theta = var_x / mu_x
        c1 = gamma.cdf(T1, a=alpha, scale=theta)
        c2 = gamma.cdf(T2, a=alpha, scale=theta)
        out[0, :] = c1
        out[1, :] = np.maximum(0.0, c2 - c1)
        out[2, :] = 1.0 - c2

    elif code == 5:  # Weibull
        var_x = float(np.maximum(ev, 0.0))
        for i in range(n_time):
            M = float(max(best_guess[i], 0.0))
            V = var_x
            if V <= 0.0 or M <= 0.0:
                continue
            try:
                k = float(fsolve(_weibull_shape_solver, 2.0, args=(M, V))[0])
                if k <= 0.0 or not np.isfinite(k):
                    continue
                lam = M / float(gamma_function(1.0 + 1.0/k))
                c1 = weibull_min.cdf(T1, c=k, loc=0.0, scale=lam)
                c2 = weibull_min.cdf(T2, c=k, loc=0.0, scale=lam)
                out[0, i] = c1
                out[1, i] = max(0.0, c2 - c1)
                out[2, i] = 1.0 - c2
            except Exception:
                continue

    elif code == 6:  # Student-t
        if dof is None or dof <= 2:
            return out
        scale = np.sqrt(max(0.0, ev) * (dof - 2.0) / max(dof, 1.0))
        c1 = t.cdf(T1, df=dof, loc=best_guess, scale=scale)
        c2 = t.cdf(T2, df=dof, loc=best_guess, scale=scale)
        out[0, :] = c1
        out[1, :] = np.maximum(0.0, c2 - c1)
        out[2, :] = 1.0 - c2

    elif code == 7:  # Poisson
        mu = np.maximum(best_guess, 0.0)
        c1 = poisson.cdf(T1, mu=mu)
        c2 = poisson.cdf(T2, mu=mu)
        out[0, :] = c1
        out[1, :] = np.maximum(0.0, c2 - c1)
        out[2, :] = 1.0 - c2

    elif code == 8:  # Negative Binomial (overdispersed)
        mu = np.maximum(best_guess, 1e-12)
        V  = np.maximum(ev, 1e-12)
        valid = V > mu
        n = np.where(valid, (mu**2) / (V - mu), np.nan)
        p = np.where(valid, mu / V, np.nan)
        c1 = nbinom.cdf(T1, n=n, p=p)
        c2 = nbinom.cdf(T2, n=n, p=p)
        out[0, :] = c1
        out[1, :] = np.maximum(0.0, c2 - c1)
        out[2, :] = 1.0 - c2

    return out


def _calc_tercile_probs_nonparam(best_guess, error_samples, first_tercile, second_tercile):
    """Non-parametric method using historical error samples."""
    best_guess = np.asarray(best_guess, float)   # (T,)
    dist_err   = np.asarray(error_samples, float)  # (T,)
    n_time = best_guess.size
    pred = np.full((3, n_time), np.nan, float)
    for t in range(n_time):
        x = best_guess[t]
        if not np.isfinite(x):
            continue
        dist = x + dist_err
        dist = dist[np.isfinite(dist)]
        if dist.size == 0:
            continue
        p_below   = float(np.mean(dist <  first_tercile))
        p_between = float(np.mean((dist >= first_tercile) & (dist < second_tercile)))
        p_above   = float(1.0 - (p_below + p_between))
        pred[0, t] = p_below
        pred[1, t] = p_between
        pred[2, t] = p_above
    return pred


# -----------------------------
# Public API (M, T, Y, X)
# -----------------------------
class WAS_mme_MVA:
    """
    Method 1 of Torralba et al. (2017) (MVA): rescales ensemble to match obs mean & std.
    Assumes both reference and predicted distributions are approximately Gaussian.

    Expected dims:
      hindcast: (M, T, Y, X)
      obs     : (T, Y, X)
      forecast: (M, T, Y, X)

    Methods
    -------
    - fit(hindcast, obs)
    - transform(forecast)
    - fit_transform_loocv(hindcast, obs)
    - compute_prob(Predictant, clim_year_start, clim_year_end, hindcast_det, ...)
    - forecast(Predictant, clim_year_start, clim_year_end, hindcast_det, forecast_det, ...)
    """

    def __init__(self, dist_method: Literal["bestfit", "nonparam"] = "nonparam"):
        self.params_: Optional[dict[str, xr.DataArray]] = None
        self.dist_method = dist_method

    # -------- fit --------
    def fit(self, hindcast: xr.DataArray, obs: xr.DataArray) -> "WAS_mme_MVA":
        # Coerce order
        if set(("M","T")).issubset(hindcast.dims):
            hindcast = hindcast.transpose("M", "T", ..., missing_dims="ignore")
        if "T" in obs.dims:
            obs = obs.transpose("T", ..., missing_dims="ignore")

        assert ("M" in hindcast.dims and "T" in hindcast.dims), "hindcast must include (M,T,...)"
        assert ("T" in obs.dims), "obs must include T"

        hc, ob = xr.align(hindcast, obs, join="inner")

        packed = xr.apply_ufunc(
            _train_cell_mva,
            hc, ob,
            input_core_dims=[["M","T"], ["T"]],
            output_core_dims=[["param"]],
            output_sizes={"param": 4},              # [clim_obs, clim_fcst, sigma_e, sigma_ref]
            dask="parallelized",
            vectorize=True,
            output_dtypes=[np.float64],
        )
        packed = packed.assign_coords(param=["clim_obs","clim_fcst","sigma_e","sigma_ref"])

        self.params_ = {
            "clim_obs":  packed.sel(param="clim_obs"),
            "clim_fcst": packed.sel(param="clim_fcst"),
            "sigma_e":   packed.sel(param="sigma_e"),
            "sigma_ref": packed.sel(param="sigma_ref"),
        }
        return self

    # -------- transform --------
    def transform(self, forecast: xr.DataArray) -> xr.DataArray:
        if self.params_ is None:
            raise RuntimeError("Call fit() before transform().")

        # Ensure (M, T, ...) order; promote deterministic to M=1
        if set(("M","T")).issubset(forecast.dims):
            forecast = forecast.transpose("M", "T", ..., missing_dims="ignore")
        else:
            if "M" not in forecast.dims:
                forecast = forecast.expand_dims(M=[0])
            assert "T" in forecast.dims, "forecast must include T"
            forecast = forecast.transpose("M", "T", ..., missing_dims="ignore")

        # broadcast params to spatial template
        tpl_yx = forecast.isel(M=0, T=0, drop=True)
        clim_obs  = self.params_["clim_obs"].broadcast_like(tpl_yx)
        clim_fcst = self.params_["clim_fcst"].broadcast_like(tpl_yx)
        sigma_e   = self.params_["sigma_e"].broadcast_like(tpl_yx)
        sigma_ref = self.params_["sigma_ref"].broadcast_like(tpl_yx)

        def _apply(fc, co, cf, se, sr):
            return _apply_cell_mva(fc, float(co), float(cf), float(se), float(sr))

        out = xr.apply_ufunc(
            _apply,
            forecast, clim_obs, clim_fcst, sigma_e, sigma_ref,
            input_core_dims=[["M","T"], [], [], [], []],
            output_core_dims=[["M","T"]],
            dask="parallelized",
            vectorize=True,
            output_dtypes=[forecast.dtype],
        )
        out.name = forecast.name or "calibrated_mva"
        return out

    # -------- LOOCV on hindcast (crossval=TRUE) --------
    def fit_transform_loocv(self, hindcast: xr.DataArray, obs: xr.DataArray) -> xr.DataArray:
        if set(("M","T")).issubset(hindcast.dims):
            hindcast = hindcast.transpose("M", "T", ..., missing_dims="ignore")
        if "T" in obs.dims:
            obs = obs.transpose("T", ..., missing_dims="ignore")

        assert ("M" in hindcast.dims and "T" in hindcast.dims), "hindcast must include (M,T,...)"
        assert ("T" in obs.dims), "obs must include T"

        hc, ob = xr.align(hindcast, obs, join="inner")

        out = xr.apply_ufunc(
            _loocv_cell_mva,
            hc, ob,
            input_core_dims=[["M","T"], ["T"]],
            output_core_dims=[["M","T"]],
            dask="parallelized",
            vectorize=True,
            output_dtypes=[hindcast.dtype],
        )
        out.name = (hindcast.name or "hindcast") + "_mva_loocv"
        return out

    # -------- Calibrated ensemble mean --------
    def predict_mean(self, forecast: xr.DataArray) -> xr.DataArray:
        """Calibrate members then return ensemble mean over M: (T,Y,X)."""
        cal = self.transform(forecast)
        return cal.mean(dim="M", skipna=True)

    # -------- Hindcast probabilities --------
    def compute_prob(
        self,
        Predictant: xr.DataArray,   # obs (T,Y,X) or (T,M,Y,X) -> squeezed to (T,Y,X)
        clim_year_start,
        clim_year_end,
        hindcast_cross: xr.DataArray, 
        best_code_da: Optional[xr.DataArray] = None,
        best_shape_da: Optional[xr.DataArray] = None,
        best_loc_da: Optional[xr.DataArray] = None,
        best_scale_da: Optional[xr.DataArray] = None,
    ) -> xr.DataArray:
        """
        Compute tercile probabilities for deterministic hindcasts.

        If dist_method == 'bestfit':
          - derive climatological terciles analytically from (best_code/shape/loc/scale)
            and compute predictive probabilities with the same family.
        Else ('nonparam'):
          - use empirical terciles and historical error samples.
        """
        # squeeze potential member dim in obs; enforce order
        if "M" in Predictant.dims:
            Predictant = Predictant.isel(M=0, drop=True)
        Predictant   = Predictant.transpose("T","Y","X")


        mask = xr.where(np.isfinite(Predictant.isel(T=0)), 1.0, np.nan)

        # climatology slice
        clim = Predictant.sel(T=slice(str(clim_year_start), str(clim_year_end)))
        if clim.sizes.get("T", 0) < 3:
            raise ValueError("Not enough years in climatology period for terciles.")

        # error variance (per grid) and dof
        error_variance = (Predictant - hindcast_cross.mean(dim="M")).var(dim="T", skipna=True)
        dof = max(int(clim.sizes["T"]) - 1, 2)

        dm = self.dist_method

        if dm == "bestfit":
            if any(v is None for v in (best_code_da, best_shape_da, best_loc_da, best_scale_da)):
                raise ValueError("dist_method='bestfit' requires best_code_da, best_shape_da, best_loc_da, best_scale_da.")

            # T1,T2 from best-fit family
            T1, T2 = xr.apply_ufunc(
                _ppf_terciles_from_code,
                best_code_da, best_shape_da, best_loc_da, best_scale_da,
                input_core_dims=[(),(),(),()],
                output_core_dims=[(),()],
                vectorize=True,
                dask="parallelized",
                output_dtypes=[float, float],
            )

            hindcast_prob = xr.apply_ufunc(
                _calc_tercile_probs_bestfit,
                hindcast_cross.mean(dim="M"), error_variance, T1, T2, best_code_da,
                input_core_dims=[("T",), (), (), (), ()],
                output_core_dims=[("probability","T")],
                vectorize=True,
                kwargs={"dof": dof},
                dask="parallelized",
                output_dtypes=[float],
                dask_gufunc_kwargs={"output_sizes": {"probability": 3}, "allow_rechunk": True},
            )

        else:  # 'nonparam'
            # empirical terciles on clim
            terc = clim.quantile([1/3, 2/3], dim="T", skipna=True)
            T1_emp = terc.sel(quantile=1/3)
            T2_emp = terc.sel(quantile=2/3)
            error_samples = (Predictant - hindcast_det)

            hindcast_prob = xr.apply_ufunc(
                _calc_tercile_probs_nonparam,
                hindcast_cross.mean(dim="M"), error_samples, T1_emp, T2_emp,
                input_core_dims=[("T",), ("T",), (), ()],
                output_core_dims=[("probability","T")],
                vectorize=True,
                dask="parallelized",
                output_dtypes=[float],
                dask_gufunc_kwargs={"output_sizes": {"probability": 3}, "allow_rechunk": True},
            )

        hindcast_prob = hindcast_prob.assign_coords(probability=("probability", ["PB","PN","PA"]))
        return (hindcast_prob * mask).transpose("probability","T","Y","X")

    # -------- Forecast (deterministic mean + probabilities) --------
    def forecast(
        self,
        Predictant: xr.DataArray,    # obs (T,Y,X) or (T,M,Y,X) -> squeezed to (T,Y,X)
        clim_year_start,
        clim_year_end,
        hindcast_det: xr.DataArray,  
        hindcast_cross: xr.DataArray,  # deterministic hindcast (T,Y,X)
        forecast_det: xr.DataArray,  # (M,T,Y,X) or (T,Y,X)
        best_code_da: Optional[xr.DataArray] = None,
        best_shape_da: Optional[xr.DataArray] = None,
        best_loc_da: Optional[xr.DataArray] = None,
        best_scale_da: Optional[xr.DataArray] = None,
    ):
        """Calibrate forecast and return (calibrated_deterministic_mean, tercile_probs)."""
        # Prepare obs/hindcast
        if "M" in Predictant.dims:
            Predictant = Predictant.isel(M=0, drop=True)
        Predictant   = Predictant.transpose("T","Y","X")
        hindcast_det = hindcast_det.transpose("T", "M", "Y","X")

        mask = xr.where(np.isfinite(Predictant.isel(T=0)), 1.0, np.nan)

        # Fit MVA on hindcast/obs
        self.fit(hindcast_det, Predictant)

        # Calibrated deterministic forecast mean
        if "M" in forecast_det.dims:
            fc_mean = self.predict_mean(forecast_det).transpose("T","Y","X")
        else:
            fc_mean = self.transform(forecast_det.expand_dims(M=[0])).mean("M").transpose("T","Y","X")

        # Stamp one T (use forecast year + obs first-month)
        t_fc0 = np.datetime64(fc_mean["T"].values[0], "Y")
        year  = int(t_fc0.astype(int) + 1970)
        t_obs0 = np.datetime64(Predictant["T"].values[0], "M")
        month  = int(t_obs0.astype(int) % 12 + 1)
        new_T  = np.datetime64(f"{year:04d}-{month:02d}-01")
        fc_mean = fc_mean.assign_coords(T=("T", [new_T])).astype(float)

        # Probabilities
        dm = self.dist_method
        if dm == "bestfit":
            if any(v is None for v in (best_code_da, best_shape_da, best_loc_da, best_scale_da)):
                raise ValueError("dist_method='bestfit' requires best_code_da, best_shape_da, best_loc_da, best_scale_da.")

            T1, T2 = xr.apply_ufunc(
                _ppf_terciles_from_code,
                best_code_da, best_shape_da, best_loc_da, best_scale_da,
                input_core_dims=[(),(),(),()],
                output_core_dims=[(),()],
                vectorize=True,
                dask="parallelized",
                output_dtypes=[float, float],
            )

            error_variance = (Predictant - hindcast_cross.mean(dim="M")).var(dim="T", skipna=True)
            dof = max(int(Predictant.sizes["T"]) - 1, 2)

            forecast_prob = xr.apply_ufunc(
                _calc_tercile_probs_bestfit,
                fc_mean, error_variance, T1, T2, best_code_da,
                input_core_dims=[("T",), (), (), (), ()],
                output_core_dims=[("probability","T")],
                vectorize=True,
                kwargs={"dof": dof},
                dask="parallelized",
                output_dtypes=[float],
                dask_gufunc_kwargs={"output_sizes": {"probability": 3}, "allow_rechunk": True},
            )

        else:  # 'nonparam'
            # empirical terciles on climatology period
            idx_start = Predictant.get_index("T").get_loc(str(clim_year_start)).start
            idx_end   = Predictant.get_index("T").get_loc(str(clim_year_end)).stop
            clim = Predictant.isel(T=slice(idx_start, idx_end))
            terc = clim.quantile([1/3, 2/3], dim="T", skipna=True)
            T1_emp = terc.sel(quantile=1/3)
            T2_emp = terc.sel(quantile=2/3)
            error_samples = (Predictant - hindcast_cross.mean(dim="M"))

            forecast_prob = xr.apply_ufunc(
                _calc_tercile_probs_nonparam,
                fc_mean, error_samples, T1_emp, T2_emp,
                input_core_dims=[("T",), ("T",), (), ()],
                output_core_dims=[("probability","T")],
                vectorize=True,
                dask="parallelized",
                output_dtypes=[float],
                dask_gufunc_kwargs={"output_sizes": {"probability": 3}, "allow_rechunk": True},
            )

        forecast_prob = forecast_prob.assign_coords(probability=("probability", ["PB","PN","PA"]))
        return (fc_mean * mask), (forecast_prob * mask).transpose("probability","T","Y","X")

    
# ==========================================================
# Roebber (2015) procedural EP/GA implementation
#   - Sexual selection
#   - Paternal coefficients retained in crossover
#   - Mutation + transposition (copy error)
#   - Survivors define predictive distribution
# ==========================================================

CONST1 = "__CONST1__"


def _safe_float(x) -> float:
    try:
        return float(x)
    except Exception:
        return float("nan")


def _norm01(a: np.ndarray) -> Tuple[np.ndarray, float, float]:
    """Normalize to [0,1] using min/max; return (norm, min, max)."""
    a = np.asarray(a, dtype=float)
    mn = np.nanmin(a)
    mx = np.nanmax(a)
    if (not np.isfinite(mn)) or (not np.isfinite(mx)) or mx == mn:
        return np.zeros_like(a, dtype=float), float(mn), float(mx)
    out = (a - mn) / (mx - mn)
    out = np.clip(out, 0.0, 1.0)
    return out, float(mn), float(mx)


def _inv_norm01(a01: np.ndarray, mn: float, mx: float) -> np.ndarray:
    a01 = np.asarray(a01, dtype=float)
    if (not np.isfinite(mn)) or (not np.isfinite(mx)) or mx == mn:
        return np.full_like(a01, np.nan, dtype=float)
    return a01 * (mx - mn) + mn


# ==========================================================
# EP Gene (faithful roles):
#   Gate uses (V1, OR, V2)
#   Expression uses (C1*V3 O1 C2*V4) O2 C3*V5
#   Variables may be predictors or CONST1
# ==========================================================
class EP_Gene:
    OPERATORS = {
        "ADD": operator.add,
        "MULTIPLY": operator.mul,
    }
    REL_OPS = {
        "<=": operator.le,
        ">": operator.gt,
    }

    def __init__(self, predictor_names: List[str], rng: random.Random):
        if not predictor_names:
            raise ValueError("predictor_names must be non-empty")
        self.predictor_names = list(predictor_names)
        self.rng = rng

        # gate vars
        self.v1 = rng.choice(self.predictor_names)
        self.v2 = rng.choice(self.predictor_names)
        self.OR = rng.choice(list(self.REL_OPS.keys()))

        # expression vars
        self.v3 = rng.choice(self.predictor_names)
        self.v4 = rng.choice(self.predictor_names)
        self.v5 = rng.choice(self.predictor_names)

        # coefficients in [-1,1]
        self.c1 = rng.uniform(-1.0, 1.0)
        self.c2 = rng.uniform(-1.0, 1.0)
        self.c3 = rng.uniform(-1.0, 1.0)

        self.O1 = rng.choice(list(self.OPERATORS.keys()))
        self.O2 = rng.choice(list(self.OPERATORS.keys()))

    def copy(self) -> "EP_Gene":
        g = EP_Gene(self.predictor_names, self.rng)
        g.v1, g.v2, g.OR = self.v1, self.v2, self.OR
        g.v3, g.v4, g.v5 = self.v3, self.v4, self.v5
        g.c1, g.c2, g.c3 = self.c1, self.c2, self.c3
        g.O1, g.O2 = self.O1, self.O2
        return g

    def _get_val(self, row: Dict[str, float], name: str) -> float:
        if name == CONST1:
            return 1.0
        return _safe_float(row.get(name, np.nan))

    def evaluate(self, row: Dict[str, float]) -> float:
        v1 = self._get_val(row, self.v1)
        v2 = self._get_val(row, self.v2)
        if not (np.isfinite(v1) and np.isfinite(v2)):
            return 0.0

        # gate: v1 OR v2 (relational comparison)
        # Example: if OR is "<=", then check if v1 <= v2
        try:
            gate_result = self.REL_OPS[self.OR](v1, v2)
        except (TypeError, ValueError):
            gate_result = False
        
        if not gate_result:
            return 0.0

        v3 = self._get_val(row, self.v3)
        v4 = self._get_val(row, self.v4)
        v5 = self._get_val(row, self.v5)
        if not (np.isfinite(v3) and np.isfinite(v4) and np.isfinite(v5)):
            return 0.0

        t1 = self.c1 * v3
        t2 = self.c2 * v4
        t3 = self.c3 * v5
        
        try:
            # First operation
            res = self.OPERATORS[self.O1](t1, t2)
            # Second operation
            res = self.OPERATORS[self.O2](res, t3)
            return float(res)
        except (ZeroDivisionError, ValueError, OverflowError):
            return 0.0

    # ---- Roebber (2015) crossover rule helper
    def replace_structure_from_maternal_keep_paternal_coeffs(self, maternal: "EP_Gene") -> None:
        # keep c1,c2,c3; replace variables + operators
        self.v1, self.v2, self.OR = maternal.v1, maternal.v2, maternal.OR
        self.v3, self.v4, self.v5 = maternal.v3, maternal.v4, maternal.v5
        self.O1, self.O2 = maternal.O1, maternal.O2

    # ---- Roebber (2015) mutation (at most one gene; element-wise weights)
    def mutate_element(self, rng: random.Random) -> None:
        """
        Implements the mutation menu and approximate weights from Roebber (2015):
          overall mutation is handled by the caller (50%).
          then select ONE element type to mutate in this gene.
        """
        # weights (relative) — we use exact probabilities as weights
        choices = [
            ("V1", 0.03125),
            ("OR", 0.03125),
            ("O1", 0.03125),
            ("O2", 0.03125),

            ("V2", 0.015625),
            ("V3", 0.015625),
            ("V4", 0.015625),
            ("V5", 0.015625),

            ("V2_TO_CONST1", 0.015625),
            ("V3_TO_CONST1", 0.015625),
            ("V4_TO_CONST1", 0.015625),
            ("V5_TO_CONST1", 0.015625),

            ("C1", 0.0833),
            ("C2", 0.0833),
            ("C3", 0.0833),
        ]
        labels, weights = zip(*choices)
        pick = rng.choices(labels, weights=weights, k=1)[0]

        if pick == "V1":
            self.v1 = rng.choice(self.predictor_names)
        elif pick == "V2":
            self.v2 = rng.choice(self.predictor_names)
        elif pick == "V3":
            self.v3 = rng.choice(self.predictor_names)
        elif pick == "V4":
            self.v4 = rng.choice(self.predictor_names)
        elif pick == "V5":
            self.v5 = rng.choice(self.predictor_names)
        elif pick == "V2_TO_CONST1":
            self.v2 = CONST1
        elif pick == "V3_TO_CONST1":
            self.v3 = CONST1
        elif pick == "V4_TO_CONST1":
            self.v4 = CONST1
        elif pick == "V5_TO_CONST1":
            self.v5 = CONST1
        elif pick == "OR":
            self.OR = rng.choice(list(self.REL_OPS.keys()))
        elif pick == "O1":
            self.O1 = rng.choice(list(self.OPERATORS.keys()))
        elif pick == "O2":
            self.O2 = rng.choice(list(self.OPERATORS.keys()))
        elif pick == "C1":
            self.c1 = rng.uniform(-1.0, 1.0)
        elif pick == "C2":
            self.c2 = rng.uniform(-1.0, 1.0)
        elif pick == "C3":
            self.c3 = rng.uniform(-1.0, 1.0)


# ==========================================================
# Individual (algorithm) = sum of EP-genes
# ==========================================================
class EP_Individual:
    def __init__(self, predictor_names: List[str], num_genes: int, rng: random.Random, sex: str):
        self.predictor_names = list(predictor_names)
        self.rng = rng
        self.sex = sex  # "M" or "F"
        self.genes: List[EP_Gene] = [EP_Gene(self.predictor_names, rng) for _ in range(int(num_genes))]
        self.mse: float = float("inf")

    def copy(self) -> "EP_Individual":
        c = EP_Individual(self.predictor_names, num_genes=len(self.genes), rng=self.rng, sex=self.sex)
        c.genes = [g.copy() for g in self.genes]
        c.mse = self.mse
        return c

    def predict_row(self, row: Dict[str, float]) -> float:
        if not row:
            return 0.0
        total = 0.0
        for g in self.genes:
            val = g.evaluate(row)
            if np.isfinite(val):
                total += val
        return float(total)

    def compute_mse(self, rows: List[Dict[str, float]], y_norm: np.ndarray) -> None:
        n = len(rows)
        if n == 0:
            self.mse = float("inf")
            return
        preds = np.fromiter((self.predict_row(rows[i]) for i in range(n)), dtype=float, count=n)
        preds = np.clip(preds, 0.0, 1.0)  # work in normalized [0,1] space
        err = y_norm - preds
        self.mse = float(np.mean(err * err))

    # ---- Roebber (2015) crossover
    @staticmethod
    def crossover(paternal: "EP_Individual", maternal: "EP_Individual", rng: random.Random) -> "EP_Individual":
        child = paternal.copy()  # seed with paternal EP-genes
        # per gene: 50% replace structure with maternal gene but keep paternal coefficients
        for j in range(len(child.genes)):
            if rng.random() < 0.5:
                child.genes[j].replace_structure_from_maternal_keep_paternal_coeffs(maternal.genes[j])
        return child

    # ---- Roebber (2015) transposition (copy error)
    @staticmethod
    def transposition(child: "EP_Individual", rng: random.Random) -> None:
        """
        Copy one EP-gene segment from one line to another.
        Segment set (as in Roebber 2015 text):
          A: (V1, OR, V2)
          B: (C1, V3, O1)
          C: (C2, V4, O2)
          D: (C3, V5)
        """
        if len(child.genes) < 2:
            return
        src = rng.randrange(len(child.genes))
        dst = rng.randrange(len(child.genes))
        while dst == src:
            dst = rng.randrange(len(child.genes))

        seg = rng.choice(["A", "B", "C", "D"])
        gs = child.genes[src]
        gd = child.genes[dst]

        if seg == "A":
            gd.v1, gd.OR, gd.v2 = gs.v1, gs.OR, gs.v2
        elif seg == "B":
            gd.c1, gd.v3, gd.O1 = gs.c1, gs.v3, gs.O1
        elif seg == "C":
            gd.c2, gd.v4, gd.O2 = gs.c2, gs.v4, gs.O2
        else:  # "D"
            gd.c3, gd.v5 = gs.c3, gs.v5


# ==========================================================
# Main Driver Class (Roebber 2015 faithful evolution)
# ==========================================================
@dataclass
class Roebber2015Config:
    population_size: int = 50
    num_genes: int = 10
    max_iter: int = 100

    max_mates_per_male: int = 10  # top male can mate up to 10 females (and similarly for next males)
    qualify_quantile: float = 0.50  # starting point for an MSE threshold (tightened each generation)
    tighten_factor: float = 0.97    # decreases threshold each generation (stricter)
    relax_factor: float = 1.10      # if qualifying pop < min_qualify, relax threshold upward by 10%
    min_qualify: int = 10           # "critical number" mentioned in Roebber 2015

    p_mutation: float = 0.50        # overall mutation probability
    p_transposition: float = 0.50   # overall transposition probability

    random_state: int = 42


class WAS_mme_Roebber2015:
    """
    Procedurally faithful Roebber (2015) EP / GA trainer for gridded xarray inputs.

      - sexual selection mating system
      - paternal coefficients retained in crossover
      - mutation + transposition rules
      - probabilistic output uses surviving population predictions
    """

    def __init__(self, config: Roebber2015Config = Roebber2015Config()):
        self.cfg = config
        self._rng = random.Random(int(config.random_state))
        np.random.seed(int(config.random_state))

        # Stored normalization for later prediction
        self._y_minmax: Optional[Tuple[float, float]] = None
        self._pred_minmax: Dict[str, Tuple[float, float]] = {}

        # survivors after training at a gridpoint
        self._survivors: Optional[List[EP_Individual]] = None
        self._threshold_final: Optional[float] = None
        self._predictor_names: Optional[List[str]] = None

    # -------------------------
    # data prep per grid-point
    # -------------------------
    def _prep_rows(
        self,
        X_train_1d: np.ndarray,  # (n, M)
        y_train_1d: np.ndarray,  # (n,)
        predictor_names: List[str],
    ) -> Tuple[List[Dict[str, float]], np.ndarray]:
        # normalize y to [0,1]
        y_norm, y_min, y_max = _norm01(y_train_1d)
        self._y_minmax = (y_min, y_max)

        # normalize each predictor to [0,1]
        rows: List[Dict[str, float]] = []
        self._pred_minmax = {}
        Xn = np.empty_like(X_train_1d, dtype=float)

        for j, name in enumerate(predictor_names):
            col = X_train_1d[:, j]
            coln, mn, mx = _norm01(col)
            self._pred_minmax[name] = (mn, mx)
            Xn[:, j] = coln

        # build rows; drop samples with any NaN predictor or y
        valid = np.isfinite(y_norm) & np.all(np.isfinite(Xn), axis=1)
        Xn = Xn[valid]
        y_norm = y_norm[valid]

        for i in range(Xn.shape[0]):
            row = {predictor_names[j]: float(Xn[i, j]) for j in range(Xn.shape[1])}
            rows.append(row)

        return rows, y_norm

    def _norm_test_rows(self, X_test_1d: np.ndarray, predictor_names: List[str]) -> Tuple[List[Dict[str, float]], np.ndarray]:
        Xn = np.empty_like(X_test_1d, dtype=float)
        for j, name in enumerate(predictor_names):
            mn, mx = self._pred_minmax.get(name, (np.nan, np.nan))
            col = X_test_1d[:, j].astype(float)
            if (not np.isfinite(mn)) or (not np.isfinite(mx)) or mx == mn:
                Xn[:, j] = 0.0
            else:
                with np.errstate(invalid='ignore'):
                    normed = (col - mn) / (mx - mn)
                    Xn[:, j] = np.clip(normed, 0.0, 1.0)

        valid = np.all(np.isfinite(Xn), axis=1)
        rows = [{predictor_names[j]: float(Xn[i, j]) for j in range(Xn.shape[1])} 
                for i, is_valid in enumerate(valid) if is_valid]
        return rows, valid

    # -------------------------
    # Evolution: sexual selection + replacement + thresholds
    # -------------------------
    def _init_population(self, predictor_names: List[str]) -> List[EP_Individual]:
        pop = []
        for _ in range(int(self.cfg.population_size)):
            sex = "M" if self._rng.random() < 0.5 else "F"
            pop.append(EP_Individual(predictor_names, num_genes=self.cfg.num_genes, rng=self._rng, sex=sex))
        return pop

    def _compute_threshold(self, mses: np.ndarray, prev_threshold: Optional[float]) -> float:
        mses = np.asarray(mses, dtype=float)
        mses = mses[np.isfinite(mses)]
        if mses.size == 0:
            return float("inf")
        
        base = float(np.quantile(mses, self.cfg.qualify_quantile))
        
        if prev_threshold is None or not np.isfinite(prev_threshold):
            thr = base
        else:
            # tighten by factor (stricter = smaller MSE allowed)
            thr = prev_threshold * float(self.cfg.tighten_factor)
        
        # Never let threshold go below base quantile
        thr = min(thr, base)
        
        return thr

    def _evolve(
        self,
        rows: List[Dict[str, float]],
        y_norm: np.ndarray,
        predictor_names: List[str],
    ) -> List[EP_Individual]:
        pop = self._init_population(predictor_names)
        thr: Optional[float] = None
        
        for gen in range(int(self.cfg.max_iter)):
            # evaluate MSE
            for ind in pop:
                ind.compute_mse(rows, y_norm)
            
            mses = np.array([ind.mse for ind in pop], dtype=float)
            thr = self._compute_threshold(mses, thr)
            
            # Count qualifying individuals
            qualify = np.array([ind.mse <= thr for ind in pop], dtype=bool)
            qcount = int(np.sum(qualify))
            
            # Relax threshold if too few qualify
            relax_loops = 0
            while qcount < int(self.cfg.min_qualify) and relax_loops < 20:
                thr *= float(self.cfg.relax_factor)  # +10% relaxation
                qualify = np.array([ind.mse <= thr for ind in pop], dtype=bool)
                qcount = int(np.sum(qualify))
                relax_loops += 1
            
            # If still not enough qualifiers, use all individuals
            if qcount < int(self.cfg.min_qualify):
                # Keep all individuals for this generation
                pass
            
            # partition qualifying males/females
            males = [pop[i] for i in range(len(pop)) if qualify[i] and pop[i].sex == "M"]
            females = [pop[i] for i in range(len(pop)) if qualify[i] and pop[i].sex == "F"]
            
            # rank males by MSE (lower is better)
            males.sort(key=lambda z: z.mse)
            
            offspring: List[EP_Individual] = []
            
            # cloning of top-ranked male (if available)
            if males:
                offspring.append(males[0].copy())
            
            # mating loop
            remaining_females = females[:]
            for m in males:
                if not remaining_females:
                    break
                n_mates = min(int(self.cfg.max_mates_per_male), len(remaining_females))
                mates = self._rng.sample(remaining_females, k=n_mates)
                # remove females after mating
                remaining_females = [f for f in remaining_females if f not in mates]
                
                for f in mates:
                    child = EP_Individual.crossover(m, f, self._rng)
                    child.sex = "M" if self._rng.random() < 0.5 else "F"
                    
                    # mutation (50% overall; at most one gene)
                    if self._rng.random() < float(self.cfg.p_mutation):
                        gidx = self._rng.randrange(len(child.genes))
                        child.genes[gidx].mutate_element(self._rng)
                    
                    # transposition (50% overall; independent)
                    if self._rng.random() < float(self.cfg.p_transposition):
                        EP_Individual.transposition(child, self._rng)
                    
                    offspring.append(child)
            
            # Create new population with qualifying individuals
            new_pop = []
            qualified_indices = [i for i, ok in enumerate(qualify) if ok]
            non_qualified_indices = [i for i, ok in enumerate(qualify) if not ok]
            
            # Keep all qualified individuals
            for idx in qualified_indices:
                new_pop.append(pop[idx])
            
            # Replace non-qualified with offspring (up to available offspring)
            offspring_to_use = min(len(offspring), len(non_qualified_indices))
            for i in range(offspring_to_use):
                if i < len(non_qualified_indices):
                    new_pop.append(offspring[i])
                else:
                    break
            
            # If more offspring than non-qualified slots, add to population (capped at population_size)
            if len(offspring) > offspring_to_use:
                extra = offspring[offspring_to_use:]
                for ind in extra:
                    if len(new_pop) < self.cfg.population_size:
                        new_pop.append(ind)
            
            # Fill remaining slots with random individuals from old population if needed
            if len(new_pop) < self.cfg.population_size:
                needed = self.cfg.population_size - len(new_pop)
                available = [pop[i] for i in range(len(pop)) if pop[i] not in new_pop]
                if available:
                    selected = self._rng.sample(available, min(needed, len(available)))
                    new_pop.extend(selected)
            
            # Trim to population size if needed
            if len(new_pop) > self.cfg.population_size:
                # Sort by MSE and keep best
                new_pop.sort(key=lambda z: z.mse)
                new_pop = new_pop[:self.cfg.population_size]
            
            pop = new_pop
        
        # Final evaluation
        for ind in pop:
            ind.compute_mse(rows, y_norm)
        mses = np.array([ind.mse for ind in pop], dtype=float)
        
        # Final threshold with relaxation safeguard
        valid_mses = mses[np.isfinite(mses)]
        if len(valid_mses) > 0:
            thr = float(np.quantile(valid_mses, self.cfg.qualify_quantile))
            qualify = np.array([ind.mse <= thr for ind in pop], dtype=bool)
            qcount = int(np.sum(qualify))
            
            # Relax if needed
            relax_loops = 0
            while qcount < int(self.cfg.min_qualify) and relax_loops < 20:
                thr *= float(self.cfg.relax_factor)
                qualify = np.array([ind.mse <= thr for ind in pop], dtype=bool)
                qcount = int(np.sum(qualify))
                relax_loops += 1
            
            survivors = [pop[i] for i in range(len(pop)) if qualify[i]]
        else:
            survivors = []
        
        if not survivors:
            # Fallback: keep best individuals
            pop.sort(key=lambda z: z.mse)
            survivors = pop[:max(1, min(len(pop), int(self.cfg.population_size * 0.2)))]  # Keep top 20%
        
        self._threshold_final = thr if 'thr' in locals() else None
        return survivors

    # -------------------------
    # Core API: compute_model
    # -------------------------
    def compute_model(
        self,
        X_train: xr.DataArray,   # (T,M,Y,X) preferred
        y_train: xr.DataArray,   # (T,Y,X)
        X_test: xr.DataArray,    # (T,M,Y,X) or (T=1,M,Y,X)
        return_ensemble: bool = True,
        max_members: Optional[int] = None,
    ) -> Tuple[xr.DataArray, Optional[xr.DataArray]]:
        """
        Returns:
          y_mean: (T,Y,X) mean of survivors' predictions
          y_ens:  (member,T,Y,X) survivors' predictions (optional)

        Note: This is *probabilistic* by construction (survivors define predictive distribution).
        """
        if "M" not in X_train.dims:
            raise ValueError("X_train must have dimension 'M' (predictor/member)")
        if not set(["T", "Y", "X"]).issubset(set(y_train.dims)):
            raise ValueError("y_train must have dims including (T,Y,X)")

        # enforce ordering
        Xtr = X_train.transpose("T", "M", "Y", "X")
        Ytr = y_train.transpose("T", "Y", "X")
        Xte = X_test.transpose("T", "M", "Y", "X")

        predictor_names = [str(v) for v in Xtr["M"].values.tolist()]
        self._predictor_names = predictor_names

        # stack samples for training and testing
        Xtr2 = Xtr.stack(sample=("T", "Y", "X")).transpose("sample", "M").values
        Ytr1 = Ytr.stack(sample=("T", "Y", "X")).values

        Xte2 = Xte.stack(sample=("T", "Y", "X")).transpose("sample", "M").values

        # drop NaNs in training
        train_valid = np.isfinite(Ytr1) & np.all(np.isfinite(Xtr2), axis=1)
        Xtr2 = Xtr2[train_valid]
        Ytr1 = Ytr1[train_valid]

        if Xtr2.shape[0] < 10:
            # too few samples
            y_mean = xr.full_like(Xte.isel(M=0), np.nan).drop_vars("M", errors="ignore")
            y_ens = None if not return_ensemble else xr.DataArray(
                np.full((1,) + y_mean.shape, np.nan, dtype=float),
                coords={"member": [0], "T": y_mean["T"], "Y": y_mean["Y"], "X": y_mean["X"]},
                dims=("member", "T", "Y", "X"),
            )
            return y_mean, y_ens

        rows, y_norm = self._prep_rows(Xtr2, Ytr1, predictor_names)

        # evolve survivors
        survivors = self._evolve(rows, y_norm, predictor_names)
        self._survivors = survivors

        # predict on test
        test_rows, test_valid = self._norm_test_rows(Xte2, predictor_names)
        # map back into the full sample layout
        n_full = Xte2.shape[0]
        preds_members = []

        survivors_use = survivors
        if max_members is not None and len(survivors_use) > int(max_members):
            # keep best max_members by training MSE
            survivors_use = sorted(survivors_use, key=lambda z: z.mse)[: int(max_members)]

        for ind in survivors_use:
            p = np.full(n_full, np.nan, dtype=float)
            # predict only valid rows
            if test_rows:
                vals = np.fromiter((ind.predict_row(test_rows[i]) for i in range(len(test_rows))), dtype=float, count=len(test_rows))
                vals = np.clip(vals, 0.0, 1.0)
                # place back
                p[test_valid] = vals
            preds_members.append(p)

        preds_members = np.asarray(preds_members, dtype=float)  # (member, sample)
        # inverse normalize y
        y_min, y_max = self._y_minmax if self._y_minmax is not None else (np.nan, np.nan)
        preds_members_phys = _inv_norm01(preds_members, y_min, y_max)

        # reshape to (member,T,Y,X)
        Tn, Yn, Xn = Xte.sizes["T"], Xte.sizes["Y"], Xte.sizes["X"]
        preds_members_phys = preds_members_phys.reshape(preds_members_phys.shape[0], Tn, Yn, Xn)

        y_ens = None
        if return_ensemble:
            y_ens = xr.DataArray(
                preds_members_phys,
                coords={"member": np.arange(preds_members_phys.shape[0]), "T": Xte["T"], "Y": Xte["Y"], "X": Xte["X"]},
                dims=("member", "T", "Y", "X"),
                name="roebber2015_ens",
            )

        y_mean = xr.DataArray(
            np.nanmean(preds_members_phys, axis=0),
            coords={"T": Xte["T"], "Y": Xte["Y"], "X": Xte["X"]},
            dims=("T", "Y", "X"),
            name="roebber2015_mean",
        )
        return y_mean, y_ens

    # -------------------------
    # Probabilities from evolved predictive distribution (terciles)
    # -------------------------
    @staticmethod
    def tercile_thresholds_from_obs(
        Predictant: xr.DataArray,
        clim_year_start: int,
        clim_year_end: int,
    ) -> Tuple[xr.DataArray, xr.DataArray]:
        Y = Predictant.transpose("T", "Y", "X").sel(T=slice(str(clim_year_start), str(clim_year_end)))
        q33 = Y.quantile(1 / 3, dim="T")
        q66 = Y.quantile(2 / 3, dim="T")
        return q33, q66

    @staticmethod
    def tercile_probs_from_ensemble(
        ens: xr.DataArray,   # (member,T,Y,X)
        q33: xr.DataArray,   # (Y,X)
        q66: xr.DataArray,   # (Y,X)
    ) -> xr.DataArray:
        # Broadcast thresholds
        below = (ens < q33).mean(dim="member")
        above = (ens > q66).mean(dim="member")
        normal = 1.0 - below - above

        below = below.drop_vars("quantile", errors="ignore")
        above = above.drop_vars("quantile", errors="ignore")
        normal = normal.drop_vars("quantile", errors="ignore")
        
        prob = xr.concat([below, normal, above], dim="probability")
        prob = prob.assign_coords(probability=["PB", "PN", "PA"])
        return prob.transpose("probability", "T", "Y", "X")

    # -------------------------
    # Forecast wrapper 
    # -------------------------
    def forecast(
        self,
        Predictant: xr.DataArray,
        clim_year_start: int,
        clim_year_end: int,
        hindcast_det: xr.DataArray,        # (T,M,Y,X)
        hindcast_det_cross: xr.DataArray,  # kept for API compatibility 
        Predictor_for_year: xr.DataArray,  # (T,M,Y,X) typically T=1
    ) -> Tuple[xr.DataArray, xr.DataArray]:
        # deterministic + ensemble
        y_mean, y_ens = self.compute_model(
            X_train=hindcast_det,
            y_train=Predictant,
            X_test=Predictor_for_year,
            return_ensemble=True,
        )
        if y_ens is None:
            prob = xr.full_like(y_mean.expand_dims(probability=["PB", "PN", "PA"]), np.nan)
            return y_mean, prob

        # enforce your forecast time logic
        if "T" in Predictor_for_year.coords:
            year = int(Predictor_for_year["T"].values.astype("datetime64[Y]").astype(int)[0] + 1970)
        else:
            year = 1970
        month_1 = int(Predictant["T"].values[0].astype("datetime64[M]").astype(int) % 12 + 1)
        new_T = np.datetime64(f"{year}-{month_1:02d}-01")

        y_mean = y_mean.assign_coords(T=xr.DataArray([new_T], dims=["T"]))
        y_mean["T"] = y_mean["T"].astype("datetime64[ns]")

        y_ens = y_ens.assign_coords(T=y_mean["T"])

        # tercile thresholds and probabilities from evolved predictive distribution
        q33, q66 = self.tercile_thresholds_from_obs(Predictant, clim_year_start, clim_year_end)
        prob = self.tercile_probs_from_ensemble(y_ens, q33, q66)

        return y_mean, prob

# ============================================================
# BMA Shared utilities
# ============================================================

def _require_dims(da: xr.DataArray, dims: Sequence[str], name: str) -> None:
    missing = [d for d in dims if d not in da.dims]
    if missing:
        raise ValueError(f"{name} is missing required dims: {missing}. Found dims={da.dims}")


def _normalize_weights(w: np.ndarray) -> np.ndarray:
    w = np.asarray(w, dtype=float)
    w = np.clip(w, 0.0, np.inf)
    s = np.sum(w)
    if not np.isfinite(s) or s <= 0:
        return np.full_like(w, 1.0 / w.size)
    return w / s


def compute_gridpoint_terciles_from_obs(
    obs: xr.DataArray,
    *,
    time_dim: str = "T",
    lat_dim: str = "Y",
    lon_dim: str = "X",
    q: Tuple[float, float] = (1.0 / 3.0, 2.0 / 3.0),
    labels: Tuple[str, str] = ("lower", "upper"),
) -> xr.DataArray:
    """
    Compute gridpoint tercile thresholds from observations.

    Returns
    -------
    xr.DataArray
        dims ('tercile', Y, X) with tercile labels ['lower','upper'] (by default).
    """
    _require_dims(obs, [time_dim, lat_dim, lon_dim], "obs_for_terciles")
    thr = obs.quantile([q[0], q[1]], dim=time_dim, skipna=True)  # dim name 'quantile'
    if "quantile" in thr.dims and "tercile" not in thr.dims:
        thr = thr.rename({"quantile": "tercile"})
    if thr.sizes.get("tercile", None) == 2:
        thr = thr.assign_coords(tercile=list(labels))
    return thr.transpose("tercile", lat_dim, lon_dim)


def _invert_cdf(cdf_fn, p: float, lo: float, hi: float) -> float:
    return brentq(lambda z: cdf_fn(z) - p, lo, hi)


def _fit_linear_bias_memberwise(x: np.ndarray, y: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    """
    Memberwise OLS: y = a_k + b_k x_{k}
    x: (M,N), y: (N,)
    """
    M, _N = x.shape
    a = np.zeros(M, dtype=float)
    b = np.ones(M, dtype=float)
    for k in range(M):
        Xk = x[k, :]
        A = np.vstack([np.ones_like(Xk), Xk]).T
        beta, *_ = np.linalg.lstsq(A, y, rcond=None)
        a[k], b[k] = float(beta[0]), float(beta[1])
    return a, b


def _fit_linear_bias_exchangeable(x: np.ndarray, y: np.ndarray) -> Tuple[float, float]:
    """
    Exchangeable pooled OLS: treat members as replicates.
    x: (M,N), y: (N,)
    """
    M, N = x.shape
    X = x.reshape(-1)
    Y = np.tile(y, M)
    A = np.vstack([np.ones_like(X), X]).T
    beta, *_ = np.linalg.lstsq(A, Y, rcond=None)
    return float(beta[0]), float(beta[1])


def _apply_bias(x: np.ndarray, a: np.ndarray, b: np.ndarray) -> np.ndarray:
    return a[:, None] + b[:, None] * x


def _resolve_terciles_for_prediction(
    *,
    new_forecasts: xr.DataArray,
    lat_dim: str,
    lon_dim: str,
    time_dim: str,
    clim_terciles: Optional[xr.DataArray],
    obs_for_terciles: Optional[xr.DataArray],
    stored_terciles: Optional[xr.DataArray],
    tercile_q: Tuple[float, float],
) -> Optional[xr.DataArray]:
    """
    Priority:
      1) obs_for_terciles -> compute
      2) clim_terciles -> use
      3) stored_terciles -> use
    """
    terc = None

    if obs_for_terciles is not None:
        obs_sel = obs_for_terciles.sel(
            {lat_dim: new_forecasts[lat_dim], lon_dim: new_forecasts[lon_dim]},
            method="nearest",
        )
        terc = compute_gridpoint_terciles_from_obs(
            obs_sel, time_dim=time_dim, lat_dim=lat_dim, lon_dim=lon_dim, q=tercile_q
        )
    elif clim_terciles is not None:
        # Accept either ('tercile',Y,X) or ('quantile',Y,X)
        if "tercile" not in clim_terciles.dims and "quantile" in clim_terciles.dims:
            clim_terciles = clim_terciles.rename({"quantile": "tercile"})
        _require_dims(clim_terciles, ["tercile", lat_dim, lon_dim], "clim_terciles")
        terc = clim_terciles.sel(
            {lat_dim: new_forecasts[lat_dim], lon_dim: new_forecasts[lon_dim]},
            method="nearest",
        )
    elif stored_terciles is not None:
        terc = stored_terciles.sel(
            {lat_dim: new_forecasts[lat_dim], lon_dim: new_forecasts[lon_dim]},
            method="nearest",
        )

    return terc


def _tercile_probs_from_cdf(cdf_fn, low: float, up: float) -> Tuple[float, float, float]:
    pb = float(cdf_fn(low))
    pu = float(cdf_fn(up))
    pn = float(np.clip(pu - pb, 0.0, 1.0))
    pa = float(np.clip(1.0 - pu, 0.0, 1.0))
    pb = float(np.clip(pb, 0.0, 1.0))
    # Renormalize lightly in case of numerical issues
    s = pb + pn + pa
    if np.isfinite(s) and s > 0:
        pb, pn, pa = pb / s, pn / s, pa / s
    return pb, pn, pa


# ============================================================
# 1) Baseline “mixture BMA-like” class (SLSQP fit), with terciles
# ============================================================

class WAS_EnsembleBMA:
    """
    Baseline mixture postprocessor, with optional terciles.

    model_type:
      - 'normal' : Gaussian mixture around each member forecast (shared sigma per gridpoint)
      - 'gamma'  : Gamma mixture; component mean = member forecast; shape is fitted per gridpoint
      - 'gamma0' : Zero-adjusted gamma (precip): POP via logistic + Gamma mixture for positive part

    fit() can store gridpoint terciles from obs.

    predict_probabilistic() returns:
      - predictive_mean (T,Y,X)
      - predictive_quantiles ('quantile',T,Y,X)
      - tercile_thresholds ('tercile',Y,X)  [if available]
      - tercile_probability ('category',T,Y,X) with ['PB','PN','PA'] [if available]
    """

    def __init__(self, model_type: str = "normal", tol: float = 1e-3, compute_terciles: bool = True):
        if model_type not in {"normal", "gamma", "gamma0"}:
            raise NotImplementedError(f"Model type '{model_type}' not supported.")
        self.model_type = model_type
        self.tol = tol
        self.compute_terciles = compute_terciles
        self.fitted = False

        self.weights: Optional[xr.DataArray] = None  # (M,Y,X)
        self.sigma: Optional[xr.DataArray] = None    # (Y,X) if normal
        self.shape: Optional[xr.DataArray] = None    # (Y,X) if gamma/gamma0
        self.logistic_params: Optional[xr.DataArray] = None  # (param,Y,X) if gamma0

        self.clim_terciles: Optional[xr.DataArray] = None  # ('tercile',Y,X)

    def fit(
        self,
        hcst_grid: xr.DataArray,
        obs_grid: xr.DataArray,
        *,
        member_dim: str = "M",
        time_dim: str = "T",
        lat_dim: str = "Y",
        lon_dim: str = "X",
        min_samples: int = 10,
        tercile_q: Tuple[float, float] = (1 / 3, 2 / 3),
    ):
        _require_dims(hcst_grid, [member_dim, time_dim, lat_dim, lon_dim], "hcst_grid")
        _require_dims(obs_grid, [time_dim, lat_dim, lon_dim], "obs_grid")

        if self.compute_terciles:
            self.clim_terciles = compute_gridpoint_terciles_from_obs(
                obs_grid, time_dim=time_dim, lat_dim=lat_dim, lon_dim=lon_dim, q=tercile_q
            )

        hcst = hcst_grid.transpose(member_dim, time_dim, lat_dim, lon_dim).values
        obs = obs_grid.transpose(time_dim, lat_dim, lon_dim).values

        M, T, Y, X = hcst.shape

        weights_map = np.full((M, Y, X), 1.0 / M, dtype=float)
        p1_map = np.full((Y, X), np.nan, dtype=float)         # sigma or shape
        p2_map = np.full((2, Y, X), np.nan, dtype=float)      # logistic params for gamma0

        print(f"Fitting {self.model_type} baseline mixture...")

        for iy in tqdm(range(Y), desc="Training"):
            for ix in range(X):
                f_raw = hcst[:, :, iy, ix]
                o_raw = obs[:, iy, ix]
                valid = np.isfinite(o_raw) & np.isfinite(f_raw).all(axis=0)
                if int(valid.sum()) < min_samples:
                    continue

                f = f_raw[:, valid]
                o = o_raw[valid]

                if self.model_type == "normal":

                    def nll(p):
                        ws = p[:-1]
                        sigma = p[-1]
                        w_last = 1.0 - np.sum(ws)
                        if (w_last < 0) or (sigma <= 1e-6) or (not np.isfinite(sigma)):
                            return 1e12
                        w_all = np.append(ws, w_last)
                        z = (o[None, :] - f) / sigma
                        pdf = np.dot(w_all, norm.pdf(z) / sigma)
                        return -np.sum(np.log(np.maximum(pdf, 1e-12)))

                    x0 = np.append(np.full(M - 1, 1.0 / M), np.nanstd(o))
                    bounds = [(0.0, 1.0)] * (M - 1) + [(1e-3, None)]
                    res = minimize(nll, x0, method="SLSQP", bounds=bounds, tol=self.tol)
                    if res.success:
                        w = np.append(res.x[:-1], 1.0 - np.sum(res.x[:-1]))
                        weights_map[:, iy, ix] = _normalize_weights(w)
                        p1_map[iy, ix] = float(res.x[-1])

                else:
                    if self.model_type == "gamma0":
                        # POP: logistic on cube-root ensemble mean
                        y_bin = (o > 0).astype(int)
                        ens_mean = np.mean(f, axis=0)
                        x_pred = np.cbrt(np.maximum(ens_mean, 0.0))

                        def nll_logit(beta):
                            p = expit(beta[0] + beta[1] * x_pred)
                            p = np.clip(p, 1e-8, 1 - 1e-8)
                            return -np.sum(y_bin * np.log(p) + (1 - y_bin) * np.log(1 - p))

                        res_log = minimize(nll_logit, [0.0, 1.0], method="BFGS")
                        b0, b1 = (res_log.x if res_log.success else np.array([-5.0, 0.0]))
                        p2_map[:, iy, ix] = [b0, b1]

                        mask_pos = (o > 0) & np.isfinite(o) & np.isfinite(f).all(axis=0)
                        if int(mask_pos.sum()) < max(5, min_samples // 2):
                            continue
                        o_g = o[mask_pos]
                        f_g = f[:, mask_pos]
                    else:
                        mask_pos = (o > 0) & np.isfinite(o) & np.isfinite(f).all(axis=0)
                        if int(mask_pos.sum()) < min_samples:
                            continue
                        o_g = o[mask_pos]
                        f_g = f[:, mask_pos]

                    f_g = np.maximum(f_g, 1e-3)

                    def nll_gamma(p):
                        ws = p[:-1]
                        shape_val = p[-1]
                        w_last = 1.0 - np.sum(ws)
                        if (w_last < 0) or (shape_val < 0.1) or (not np.isfinite(shape_val)):
                            return 1e12
                        w_all = np.append(ws, w_last)
                        scales = f_g / shape_val
                        pdfs = gamma.pdf(o_g[None, :], a=shape_val, scale=scales)
                        mix = np.dot(w_all, pdfs)
                        return -np.sum(np.log(np.maximum(mix, 1e-12)))

                    x0 = np.append(np.full(M - 1, 1.0 / M), 2.0)
                    bounds = [(0.0, 1.0)] * (M - 1) + [(0.1, 50.0)]
                    res = minimize(nll_gamma, x0, method="SLSQP", bounds=bounds, tol=self.tol)
                    if res.success:
                        w = np.append(res.x[:-1], 1.0 - np.sum(res.x[:-1]))
                        weights_map[:, iy, ix] = _normalize_weights(w)
                        p1_map[iy, ix] = float(res.x[-1])

        coords_m = {member_dim: hcst_grid[member_dim], lat_dim: hcst_grid[lat_dim], lon_dim: hcst_grid[lon_dim]}
        self.weights = xr.DataArray(weights_map, dims=(member_dim, lat_dim, lon_dim), coords=coords_m)

        if self.model_type == "normal":
            self.sigma = xr.DataArray(p1_map, dims=(lat_dim, lon_dim), coords={lat_dim: coords_m[lat_dim], lon_dim: coords_m[lon_dim]})
        else:
            self.shape = xr.DataArray(p1_map, dims=(lat_dim, lon_dim), coords={lat_dim: coords_m[lat_dim], lon_dim: coords_m[lon_dim]})
            if self.model_type == "gamma0":
                self.logistic_params = xr.DataArray(
                    p2_map,
                    dims=("param", lat_dim, lon_dim),
                    coords={"param": ["b0", "b1"], lat_dim: coords_m[lat_dim], lon_dim: coords_m[lon_dim]},
                )

        self.fitted = True
        return self

    def predict_probabilistic(
        self,
        new_forecasts: xr.DataArray,
        *,
        quantiles: Sequence[float] = (0.1, 0.5, 0.9),
        member_dim: str = "M",
        time_dim: str = "T",
        lat_dim: str = "Y",
        lon_dim: str = "X",
        # terciles:
        return_terciles: bool = True,
        clim_terciles: Optional[xr.DataArray] = None,
        obs_for_terciles: Optional[xr.DataArray] = None,
        tercile_q: Tuple[float, float] = (1 / 3, 2 / 3),
    ) -> xr.Dataset:
        if not self.fitted:
            raise ValueError("Fit model first.")
        _require_dims(new_forecasts, [member_dim, time_dim, lat_dim, lon_dim], "new_forecasts")

        w_ds = self.weights.sel({lat_dim: new_forecasts[lat_dim], lon_dim: new_forecasts[lon_dim]}, method="nearest")
        if self.model_type == "normal":
            p1_ds = self.sigma.sel({lat_dim: new_forecasts[lat_dim], lon_dim: new_forecasts[lon_dim]}, method="nearest")
            p2_ds = None
        else:
            p1_ds = self.shape.sel({lat_dim: new_forecasts[lat_dim], lon_dim: new_forecasts[lon_dim]}, method="nearest")
            p2_ds = (
                self.logistic_params.sel({lat_dim: new_forecasts[lat_dim], lon_dim: new_forecasts[lon_dim]}, method="nearest")
                if self.model_type == "gamma0"
                else None
            )

        terc_ds = _resolve_terciles_for_prediction(
            new_forecasts=new_forecasts,
            lat_dim=lat_dim,
            lon_dim=lon_dim,
            time_dim=time_dim,
            clim_terciles=clim_terciles,
            obs_for_terciles=obs_for_terciles,
            stored_terciles=self.clim_terciles,
            tercile_q=tercile_q,
        )

        fc = new_forecasts.transpose(member_dim, time_dim, lat_dim, lon_dim).values
        wv = w_ds.transpose(member_dim, lat_dim, lon_dim).values
        p1 = p1_ds.transpose(lat_dim, lon_dim).values
        p2 = p2_ds.transpose("param", lat_dim, lon_dim).values if p2_ds is not None else None

        M, T, Y, X = fc.shape
        nq = len(quantiles)

        out_mean = np.full((T, Y, X), np.nan, float)
        out_q = np.full((nq, T, Y, X), np.nan, float)

        do_terc = return_terciles and (terc_ds is not None)
        terc_np = terc_ds.transpose("tercile", lat_dim, lon_dim).values if do_terc else None
        out_probs = np.full((3, T, Y, X), np.nan, float) if do_terc else None

        for iy in tqdm(range(Y), desc="Predict"):
            for ix in range(X):
                par = p1[iy, ix]
                if not np.isfinite(par):
                    continue
                wk = _normalize_weights(wv[:, iy, ix])

                low_th = up_th = np.nan
                if do_terc:
                    low_th, up_th = terc_np[:, iy, ix]

                if self.model_type == "gamma0":
                    b0, b1 = p2[:, iy, ix]
                    if not (np.isfinite(b0) and np.isfinite(b1)):
                        continue

                for t in range(T):
                    x_m = fc[:, t, iy, ix]
                    if not np.isfinite(x_m).all():
                        continue

                    if self.model_type == "normal":
                        mu = float(np.dot(wk, x_m))
                        out_mean[t, iy, ix] = mu

                        def cdf_mix(z):
                            return float(np.dot(wk, norm.cdf(z, loc=x_m, scale=par)))

                        lo, hi = mu - 8.0 * par, mu + 8.0 * par
                        for iq, qv in enumerate(quantiles):
                            try:
                                out_q[iq, t, iy, ix] = _invert_cdf(cdf_mix, float(qv), lo, hi)
                            except Exception:
                                pass

                        if do_terc and np.isfinite(low_th) and np.isfinite(up_th):
                            out_probs[:, t, iy, ix] = _tercile_probs_from_cdf(cdf_mix, float(low_th), float(up_th))

                    elif self.model_type == "gamma":
                        x_safe = np.maximum(x_m, 1e-3)
                        scales = x_safe / par
                        out_mean[t, iy, ix] = float(np.dot(wk, x_m))

                        def cdf_mix(z):
                            z = max(float(z), 0.0)
                            return float(np.dot(wk, gamma.cdf(z, a=par, scale=scales)))

                        top = float(np.max(x_safe) * 8.0 + 10.0)
                        for iq, qv in enumerate(quantiles):
                            try:
                                out_q[iq, t, iy, ix] = _invert_cdf(cdf_mix, float(qv), 1e-6, top)
                            except Exception:
                                pass

                        if do_terc and np.isfinite(low_th) and np.isfinite(up_th):
                            out_probs[:, t, iy, ix] = _tercile_probs_from_cdf(cdf_mix, float(low_th), float(up_th))

                    else:  # gamma0
                        x_mean_cbrt = np.cbrt(np.maximum(float(np.mean(x_m)), 0.0))
                        pop = float(expit(b0 + b1 * x_mean_cbrt))  # P(Y>0)
                        x_safe = np.maximum(x_m, 1e-3)
                        scales = x_safe / par

                        out_mean[t, iy, ix] = pop * float(np.dot(wk, x_m))

                        def cdf_pos(z):
                            z = max(float(z), 0.0)
                            return float(np.dot(wk, gamma.cdf(z, a=par, scale=scales)))

                        def cdf_mix(z):
                            z = float(z)
                            if z <= 0.0:
                                return float(1.0 - pop)
                            return float((1.0 - pop) + pop * cdf_pos(z))

                        top = float(np.max(x_safe) * 8.0 + 10.0)
                        for iq, qv in enumerate(quantiles):
                            try:
                                qv = float(qv)
                                if qv <= (1.0 - pop):
                                    out_q[iq, t, iy, ix] = 0.0
                                else:
                                    target = (qv - (1.0 - pop)) / max(pop, 1e-12)
                                    out_q[iq, t, iy, ix] = _invert_cdf(cdf_pos, float(target), 1e-6, top)
                            except Exception:
                                pass

                        if do_terc and np.isfinite(low_th) and np.isfinite(up_th):
                            out_probs[:, t, iy, ix] = _tercile_probs_from_cdf(cdf_mix, float(low_th), float(up_th))

        coords = {time_dim: new_forecasts[time_dim], lat_dim: new_forecasts[lat_dim], lon_dim: new_forecasts[lon_dim]}
        ds = xr.Dataset()
        ds["predictive_mean"] = xr.DataArray(out_mean, dims=(time_dim, lat_dim, lon_dim), coords=coords)
        ds["predictive_quantiles"] = xr.DataArray(
            out_q, dims=("quantile", time_dim, lat_dim, lon_dim), coords={**coords, "quantile": list(quantiles)}
        )

        if do_terc:
            ds["tercile_thresholds"] = terc_ds
            ds["tercile_probability"] = xr.DataArray(
                out_probs,
                dims=("category", time_dim, lat_dim, lon_dim),
                coords={**coords, "category": ["PB", "PN", "PA"]},
            )

        return ds


# ============================================================
# 2) Gaussian BMA with bias correction + EM, with terciles
# ============================================================

@dataclass
class WAS_GaussianBMA_EM:
    """
    Gaussian BMA (Raftery/Wilks-style) with optional terciles.

    Bias correction (Eq. 8.34):
        m_{t,k} = a_k + b_k x_{t,k}

    Predictive CDF:
        F(q) = sum_k w_k Phi((q - m_{t,k}) / sigma_k)

    EM estimates weights and sigma_k (optionally constrained equal).
    """
    tol: float = 1e-4
    max_iter: int = 200
    min_sigma: float = 1e-3

    bias_mode: str = "memberwise"     # "memberwise" | "exchangeable"
    equal_sigma: bool = True
    equal_weights: bool = False

    compute_terciles: bool = True
    tercile_q_fit: Tuple[float, float] = (1 / 3, 2 / 3)

    # fitted grids
    weights: Optional[xr.DataArray] = None  # (M,Y,X)
    sigma: Optional[xr.DataArray] = None    # (M,Y,X) expanded even if equal_sigma
    a: Optional[xr.DataArray] = None        # (M,Y,X)
    b: Optional[xr.DataArray] = None        # (M,Y,X)
    clim_terciles: Optional[xr.DataArray] = None  # ('tercile',Y,X)

    fitted: bool = False

    def _fit_point(self, x: np.ndarray, y: np.ndarray) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        M, _N = x.shape

        # bias correction
        if self.bias_mode == "exchangeable":
            a0, b0 = _fit_linear_bias_exchangeable(x, y)
            a = np.full(M, a0, float)
            b = np.full(M, b0, float)
        else:
            a, b = _fit_linear_bias_memberwise(x, y)

        m = _apply_bias(x, a, b)

        # init
        w = np.full(M, 1.0 / M, float)
        if self.equal_weights:
            w[:] = 1.0 / M

        resid = y[None, :] - m
        sig = np.sqrt(np.nanmean(resid ** 2, axis=1))
        sig = np.maximum(sig, self.min_sigma)
        if self.equal_sigma:
            sig[:] = float(np.maximum(np.sqrt(np.mean(sig ** 2)), self.min_sigma))

        for _ in range(self.max_iter):
            ll = np.empty((M, y.size), dtype=float)
            for k in range(M):
                ll[k, :] = np.log(max(w[k], 1e-12)) + norm.logpdf(y, loc=m[k, :], scale=sig[k])

            ll_max = np.max(ll, axis=0, keepdims=True)
            r = np.exp(ll - ll_max)
            r_sum = np.sum(r, axis=0, keepdims=True)
            r = r / np.maximum(r_sum, 1e-12)

            w_new = r.mean(axis=1)
            if self.equal_weights:
                w_new[:] = 1.0 / M
            else:
                w_new = np.clip(w_new, 1e-12, 1.0)
                w_new /= w_new.sum()

            sig_new = np.empty_like(sig)
            for k in range(M):
                num = np.sum(r[k, :] * (y - m[k, :]) ** 2)
                den = np.sum(r[k, :])
                sig_new[k] = np.sqrt(num / np.maximum(den, 1e-12))
            sig_new = np.maximum(sig_new, self.min_sigma)

            if self.equal_sigma:
                common = np.sqrt(np.sum(r * (y[None, :] - m) ** 2) / np.maximum(np.sum(r), 1e-12))
                common = float(np.maximum(common, self.min_sigma))
                sig_new[:] = common

            if max(np.max(np.abs(w_new - w)), np.max(np.abs(sig_new - sig))) < self.tol:
                w, sig = w_new, sig_new
                break

            w, sig = w_new, sig_new

        return w, sig, a, b

    def fit(
        self,
        hcst_grid: xr.DataArray,
        obs_grid: xr.DataArray,
        *,
        member_dim: str = "M",
        time_dim: str = "T",
        lat_dim: str = "Y",
        lon_dim: str = "X",
        min_samples: int = 20,
    ):
        _require_dims(hcst_grid, [member_dim, time_dim, lat_dim, lon_dim], "hcst_grid")
        _require_dims(obs_grid, [time_dim, lat_dim, lon_dim], "obs_grid")

        if self.compute_terciles:
            self.clim_terciles = compute_gridpoint_terciles_from_obs(
                obs_grid, time_dim=time_dim, lat_dim=lat_dim, lon_dim=lon_dim, q=self.tercile_q_fit
            )

        hcst = hcst_grid.transpose(member_dim, time_dim, lat_dim, lon_dim).values
        obs = obs_grid.transpose(time_dim, lat_dim, lon_dim).values
        M, _T, Y, X = hcst.shape

        w_map = np.full((M, Y, X), np.nan, float)
        s_map = np.full((M, Y, X), np.nan, float)
        a_map = np.full((M, Y, X), np.nan, float)
        b_map = np.full((M, Y, X), np.nan, float)

        print("Fitting Gaussian BMA (EM + bias correction)...")

        for iy in tqdm(range(Y), desc="Training"):
            for ix in range(X):
                x_raw = hcst[:, :, iy, ix]
                y_raw = obs[:, iy, ix]
                valid = np.isfinite(y_raw) & np.isfinite(x_raw).all(axis=0)
                if int(valid.sum()) < min_samples:
                    continue

                w, sig, a, b = self._fit_point(x_raw[:, valid], y_raw[valid])
                w_map[:, iy, ix] = w
                s_map[:, iy, ix] = sig
                a_map[:, iy, ix] = a
                b_map[:, iy, ix] = b

        coords_m = {member_dim: hcst_grid[member_dim], lat_dim: hcst_grid[lat_dim], lon_dim: hcst_grid[lon_dim]}
        self.weights = xr.DataArray(w_map, dims=(member_dim, lat_dim, lon_dim), coords=coords_m)
        self.sigma = xr.DataArray(s_map, dims=(member_dim, lat_dim, lon_dim), coords=coords_m)
        self.a = xr.DataArray(a_map, dims=(member_dim, lat_dim, lon_dim), coords=coords_m)
        self.b = xr.DataArray(b_map, dims=(member_dim, lat_dim, lon_dim), coords=coords_m)

        self.fitted = True
        return self

    def predict_probabilistic(
        self,
        new_forecasts: xr.DataArray,
        *,
        quantiles: Sequence[float] = (0.1, 0.5, 0.9),
        member_dim: str = "M",
        time_dim: str = "T",
        lat_dim: str = "Y",
        lon_dim: str = "X",
        # terciles:
        return_terciles: bool = True,
        clim_terciles: Optional[xr.DataArray] = None,
        obs_for_terciles: Optional[xr.DataArray] = None,
        tercile_q: Tuple[float, float] = (1 / 3, 2 / 3),
    ) -> xr.Dataset:
        if not self.fitted:
            raise ValueError("Fit model first.")
        _require_dims(new_forecasts, [member_dim, time_dim, lat_dim, lon_dim], "new_forecasts")

        w = self.weights.sel({lat_dim: new_forecasts[lat_dim], lon_dim: new_forecasts[lon_dim]}, method="nearest")
        s = self.sigma.sel({lat_dim: new_forecasts[lat_dim], lon_dim: new_forecasts[lon_dim]}, method="nearest")
        a = self.a.sel({lat_dim: new_forecasts[lat_dim], lon_dim: new_forecasts[lon_dim]}, method="nearest")
        b = self.b.sel({lat_dim: new_forecasts[lat_dim], lon_dim: new_forecasts[lon_dim]}, method="nearest")

        terc_ds = _resolve_terciles_for_prediction(
            new_forecasts=new_forecasts,
            lat_dim=lat_dim,
            lon_dim=lon_dim,
            time_dim=time_dim,
            clim_terciles=clim_terciles,
            obs_for_terciles=obs_for_terciles,
            stored_terciles=self.clim_terciles,
            tercile_q=tercile_q,
        )

        fc = new_forecasts.transpose(member_dim, time_dim, lat_dim, lon_dim).values
        wv = w.transpose(member_dim, lat_dim, lon_dim).values
        sv = s.transpose(member_dim, lat_dim, lon_dim).values
        av = a.transpose(member_dim, lat_dim, lon_dim).values
        bv = b.transpose(member_dim, lat_dim, lon_dim).values

        M, T, Y, X = fc.shape
        nq = len(quantiles)

        out_mean = np.full((T, Y, X), np.nan, float)
        out_q = np.full((nq, T, Y, X), np.nan, float)

        do_terc = return_terciles and (terc_ds is not None)
        terc_np = terc_ds.transpose("tercile", lat_dim, lon_dim).values if do_terc else None
        out_probs = np.full((3, T, Y, X), np.nan, float) if do_terc else None

        for iy in tqdm(range(Y), desc="Predict"):
            for ix in range(X):
                wk = _normalize_weights(wv[:, iy, ix])
                sigk = sv[:, iy, ix]
                if not np.isfinite(sigk).all():
                    continue

                low_th = up_th = np.nan
                if do_terc:
                    low_th, up_th = terc_np[:, iy, ix]

                for t in range(T):
                    x_m = fc[:, t, iy, ix]
                    if not np.isfinite(x_m).all():
                        continue

                    mtk = av[:, iy, ix] + bv[:, iy, ix] * x_m
                    if not np.isfinite(mtk).all():
                        continue

                    out_mean[t, iy, ix] = float(np.sum(wk * mtk))

                    def cdf_mix(z):
                        return float(np.sum(wk * norm.cdf((float(z) - mtk) / sigk)))

                    lo = float(np.min(mtk - 8.0 * sigk))
                    hi = float(np.max(mtk + 8.0 * sigk))

                    for iq, qv in enumerate(quantiles):
                        try:
                            out_q[iq, t, iy, ix] = _invert_cdf(cdf_mix, float(qv), lo, hi)
                        except Exception:
                            pass

                    if do_terc and np.isfinite(low_th) and np.isfinite(up_th):
                        out_probs[:, t, iy, ix] = _tercile_probs_from_cdf(cdf_mix, float(low_th), float(up_th))

        coords = {time_dim: new_forecasts[time_dim], lat_dim: new_forecasts[lat_dim], lon_dim: new_forecasts[lon_dim]}
        ds = xr.Dataset()
        ds["predictive_mean"] = xr.DataArray(out_mean, dims=(time_dim, lat_dim, lon_dim), coords=coords)
        ds["predictive_quantiles"] = xr.DataArray(
            out_q, dims=("quantile", time_dim, lat_dim, lon_dim), coords={**coords, "quantile": list(quantiles)}
        )

        if do_terc:
            ds["tercile_thresholds"] = terc_ds
            ds["tercile_probability"] = xr.DataArray(
                out_probs,
                dims=("category", time_dim, lat_dim, lon_dim),
                coords={**coords, "category": ["PB", "PN", "PA"]},
            )

        return ds


# ============================================================
# 3) Wang & Bishop (2005) Gaussian dressing, with terciles
# ============================================================

@dataclass
class WAS_GaussianDressing_WangBishop:
    """
    Gaussian dressing (Wang & Bishop 2005), with optional terciles.

    Uses bias-corrected members m_{t,k} and equal weights 1/M and a common dressing sD.
    """
    min_sigma: float = 1e-3
    clip_negative: bool = True
    bias_mode: str = "exchangeable"  # typical in MOS

    compute_terciles: bool = True
    tercile_q_fit: Tuple[float, float] = (1 / 3, 2 / 3)

    # fitted
    a: Optional[xr.DataArray] = None   # (M,Y,X)
    b: Optional[xr.DataArray] = None   # (M,Y,X)
    sD: Optional[xr.DataArray] = None  # (Y,X)
    clim_terciles: Optional[xr.DataArray] = None

    fitted: bool = False

    def fit(
        self,
        hcst_grid: xr.DataArray,
        obs_grid: xr.DataArray,
        *,
        member_dim: str = "M",
        time_dim: str = "T",
        lat_dim: str = "Y",
        lon_dim: str = "X",
        min_samples: int = 20,
    ):
        _require_dims(hcst_grid, [member_dim, time_dim, lat_dim, lon_dim], "hcst_grid")
        _require_dims(obs_grid, [time_dim, lat_dim, lon_dim], "obs_grid")

        if self.compute_terciles:
            self.clim_terciles = compute_gridpoint_terciles_from_obs(
                obs_grid, time_dim=time_dim, lat_dim=lat_dim, lon_dim=lon_dim, q=self.tercile_q_fit
            )

        hcst = hcst_grid.transpose(member_dim, time_dim, lat_dim, lon_dim).values
        obs = obs_grid.transpose(time_dim, lat_dim, lon_dim).values
        M, _T, Y, X = hcst.shape

        a_map = np.full((M, Y, X), np.nan, float)
        b_map = np.full((M, Y, X), np.nan, float)
        sD_map = np.full((Y, X), np.nan, float)

        print("Fitting Wang & Bishop Gaussian dressing...")

        for iy in tqdm(range(Y), desc="Training"):
            for ix in range(X):
                x_raw = hcst[:, :, iy, ix]
                y_raw = obs[:, iy, ix]
                valid = np.isfinite(y_raw) & np.isfinite(x_raw).all(axis=0)
                if int(valid.sum()) < min_samples:
                    continue

                x = x_raw[:, valid]
                y = y_raw[valid]

                if self.bias_mode == "exchangeable":
                    a0, b0 = _fit_linear_bias_exchangeable(x, y)
                    a = np.full(M, a0, float)
                    b = np.full(M, b0, float)
                else:
                    a, b = _fit_linear_bias_memberwise(x, y)

                m = _apply_bias(x, a, b)  # (M,N)
                mbar = m.mean(axis=0)
                err = mbar - y

                var_err = np.var(err, ddof=1)
                var_ens_t = np.var(m, axis=0, ddof=1)
                mean_var_ens = np.mean(var_ens_t)

                sD2 = var_err - (1.0 + 1.0 / M) * mean_var_ens
                if sD2 <= 0 and self.clip_negative:
                    sD2 = self.min_sigma ** 2

                if sD2 > 0:
                    a_map[:, iy, ix] = a
                    b_map[:, iy, ix] = b
                    sD_map[iy, ix] = float(np.sqrt(sD2))

        coords_m = {member_dim: hcst_grid[member_dim], lat_dim: hcst_grid[lat_dim], lon_dim: hcst_grid[lon_dim]}
        self.a = xr.DataArray(a_map, dims=(member_dim, lat_dim, lon_dim), coords=coords_m)
        self.b = xr.DataArray(b_map, dims=(member_dim, lat_dim, lon_dim), coords=coords_m)
        self.sD = xr.DataArray(sD_map, dims=(lat_dim, lon_dim), coords={lat_dim: hcst_grid[lat_dim], lon_dim: hcst_grid[lon_dim]})

        self.fitted = True
        return self

    def predict_probabilistic(
        self,
        new_forecasts: xr.DataArray,
        *,
        quantiles: Sequence[float] = (0.1, 0.5, 0.9),
        member_dim: str = "M",
        time_dim: str = "T",
        lat_dim: str = "Y",
        lon_dim: str = "X",
        # terciles:
        return_terciles: bool = True,
        clim_terciles: Optional[xr.DataArray] = None,
        obs_for_terciles: Optional[xr.DataArray] = None,
        tercile_q: Tuple[float, float] = (1 / 3, 2 / 3),
    ) -> xr.Dataset:
        if not self.fitted:
            raise ValueError("Fit model first.")
        _require_dims(new_forecasts, [member_dim, time_dim, lat_dim, lon_dim], "new_forecasts")

        a = self.a.sel({lat_dim: new_forecasts[lat_dim], lon_dim: new_forecasts[lon_dim]}, method="nearest")
        b = self.b.sel({lat_dim: new_forecasts[lat_dim], lon_dim: new_forecasts[lon_dim]}, method="nearest")
        sD = self.sD.sel({lat_dim: new_forecasts[lat_dim], lon_dim: new_forecasts[lon_dim]}, method="nearest")

        terc_ds = _resolve_terciles_for_prediction(
            new_forecasts=new_forecasts,
            lat_dim=lat_dim,
            lon_dim=lon_dim,
            time_dim=time_dim,
            clim_terciles=clim_terciles,
            obs_for_terciles=obs_for_terciles,
            stored_terciles=self.clim_terciles,
            tercile_q=tercile_q,
        )

        fc = new_forecasts.transpose(member_dim, time_dim, lat_dim, lon_dim).values
        av = a.transpose(member_dim, lat_dim, lon_dim).values
        bv = b.transpose(member_dim, lat_dim, lon_dim).values
        sv = sD.transpose(lat_dim, lon_dim).values

        M, T, Y, X = fc.shape
        nq = len(quantiles)

        out_mean = np.full((T, Y, X), np.nan, float)
        out_q = np.full((nq, T, Y, X), np.nan, float)

        do_terc = return_terciles and (terc_ds is not None)
        terc_np = terc_ds.transpose("tercile", lat_dim, lon_dim).values if do_terc else None
        out_probs = np.full((3, T, Y, X), np.nan, float) if do_terc else None

        for iy in tqdm(range(Y), desc="Predict"):
            for ix in range(X):
                s = float(sv[iy, ix])
                if not np.isfinite(s) or s <= 0:
                    continue

                low_th = up_th = np.nan
                if do_terc:
                    low_th, up_th = terc_np[:, iy, ix]

                for t in range(T):
                    x_m = fc[:, t, iy, ix]
                    if not np.isfinite(x_m).all():
                        continue

                    mtk = av[:, iy, ix] + bv[:, iy, ix] * x_m
                    out_mean[t, iy, ix] = float(np.mean(mtk))

                    def cdf_mix(z):
                        return float(np.mean(norm.cdf((float(z) - mtk) / s)))

                    lo = float(np.min(mtk - 8.0 * s))
                    hi = float(np.max(mtk + 8.0 * s))

                    for iq, qv in enumerate(quantiles):
                        try:
                            out_q[iq, t, iy, ix] = _invert_cdf(cdf_mix, float(qv), lo, hi)
                        except Exception:
                            pass

                    if do_terc and np.isfinite(low_th) and np.isfinite(up_th):
                        out_probs[:, t, iy, ix] = _tercile_probs_from_cdf(cdf_mix, float(low_th), float(up_th))

        coords = {time_dim: new_forecasts[time_dim], lat_dim: new_forecasts[lat_dim], lon_dim: new_forecasts[lon_dim]}
        ds = xr.Dataset()
        ds["predictive_mean"] = xr.DataArray(out_mean, dims=(time_dim, lat_dim, lon_dim), coords=coords)
        ds["predictive_quantiles"] = xr.DataArray(
            out_q, dims=("quantile", time_dim, lat_dim, lon_dim), coords={**coords, "quantile": list(quantiles)}
        )

        if do_terc:
            ds["tercile_thresholds"] = terc_ds
            ds["tercile_probability"] = xr.DataArray(
                out_probs,
                dims=("category", time_dim, lat_dim, lon_dim),
                coords={**coords, "category": ["PB", "PN", "PA"]},
            )

        return ds


# ============================================================
# 4) Precipitation BMA (Sloughter et al. 2007 style), with terciles
# ============================================================

@dataclass
class WAS_PrecipBMA_Sloughter2007:
    """
    Precipitation BMA with mixed discrete-continuous components.

    - Logistic regression for P(Y=0) per member:
        p0_{t,k} = logistic(a0_k + a1_k * x_{t,k}^{1/3} + a2_k * I(x_{t,k}=0))

    - Gamma distribution on cube-root transformed positive amounts z=y^(1/3):
        z ~ Gamma(shape_k, scale_k)

    Predictive CDF for q>=0:
        F(q) = sum_k w_k [ p0_{t,k} + (1 - p0_{t,k}) * F_gamma(z_q) ]
      where z_q = q^(1/3)

    Tercile probabilities computed directly from this predictive CDF.
    """
    tol: float = 1e-4

    compute_terciles: bool = True
    tercile_q_fit: Tuple[float, float] = (1 / 3, 2 / 3)

    weights: Optional[xr.DataArray] = None    # (M,Y,X)
    logit: Optional[xr.DataArray] = None      # (param,M,Y,X) param=['a0','a1','a2']
    g_shape: Optional[xr.DataArray] = None    # (M,Y,X)
    g_scale: Optional[xr.DataArray] = None    # (M,Y,X)

    clim_terciles: Optional[xr.DataArray] = None

    fitted: bool = False

    def fit(
        self,
        hcst_grid: xr.DataArray,
        obs_grid: xr.DataArray,
        *,
        member_dim: str = "M",
        time_dim: str = "T",
        lat_dim: str = "Y",
        lon_dim: str = "X",
        min_samples: int = 30,
    ):
        _require_dims(hcst_grid, [member_dim, time_dim, lat_dim, lon_dim], "hcst_grid")
        _require_dims(obs_grid, [time_dim, lat_dim, lon_dim], "obs_grid")

        if self.compute_terciles:
            self.clim_terciles = compute_gridpoint_terciles_from_obs(
                obs_grid, time_dim=time_dim, lat_dim=lat_dim, lon_dim=lon_dim, q=self.tercile_q_fit
            )

        hcst = np.maximum(hcst_grid.transpose(member_dim, time_dim, lat_dim, lon_dim).values, 0.0)
        obs = np.maximum(obs_grid.transpose(time_dim, lat_dim, lon_dim).values, 0.0)
        M, T, Y, X = hcst.shape

        w_map = np.full((M, Y, X), np.nan, float)
        log_map = np.full((3, M, Y, X), np.nan, float)
        sh_map = np.full((M, Y, X), np.nan, float)
        sc_map = np.full((M, Y, X), np.nan, float)

        print("Fitting precipitation BMA (Sloughter 2007-like)...")

        for iy in tqdm(range(Y), desc="Training"):
            for ix in range(X):
                x_raw = hcst[:, :, iy, ix]
                y_raw = obs[:, iy, ix]
                valid = np.isfinite(y_raw) & np.isfinite(x_raw).all(axis=0)
                if int(valid.sum()) < min_samples:
                    continue

                x = x_raw[:, valid]
                y = y_raw[valid]
                z = np.cbrt(y)

                # weights: keep equal by default (can be extended to MLE if desired)
                w_map[:, iy, ix] = np.full(M, 1.0 / M, float)

                for k in range(M):
                    xk = x[k, :]
                    xk_c = np.cbrt(xk)
                    ind0 = (xk == 0.0).astype(float)
                    ybin = (y == 0.0).astype(int)

                    def nll_logit(beta):
                        p0 = expit(beta[0] + beta[1] * xk_c + beta[2] * ind0)
                        p0 = np.clip(p0, 1e-8, 1 - 1e-8)
                        return -np.sum(ybin * np.log(p0) + (1 - ybin) * np.log(1 - p0))

                    res = minimize(nll_logit, [0.0, 1.0, 0.0], method="BFGS")
                    log_map[:, k, iy, ix] = res.x if res.success else np.array([0.0, 0.0, 0.0])

                    pos = y > 0
                    zk = z[pos]
                    if zk.size >= 10:
                        m1 = float(np.mean(zk))
                        v1 = float(np.var(zk, ddof=1))
                        if (m1 > 0) and (v1 > 0) and np.isfinite(m1) and np.isfinite(v1):
                            sh_map[k, iy, ix] = (m1 ** 2) / v1
                            sc_map[k, iy, ix] = v1 / m1

        coords_m = {member_dim: hcst_grid[member_dim], lat_dim: hcst_grid[lat_dim], lon_dim: hcst_grid[lon_dim]}
        self.weights = xr.DataArray(w_map, dims=(member_dim, lat_dim, lon_dim), coords=coords_m)
        self.logit = xr.DataArray(
            log_map,
            dims=("param", member_dim, lat_dim, lon_dim),
            coords={"param": ["a0", "a1", "a2"], **coords_m},
        )
        self.g_shape = xr.DataArray(sh_map, dims=(member_dim, lat_dim, lon_dim), coords=coords_m)
        self.g_scale = xr.DataArray(sc_map, dims=(member_dim, lat_dim, lon_dim), coords=coords_m)

        self.fitted = True
        return self

    def predict_probabilistic(
        self,
        new_forecasts: xr.DataArray,
        *,
        quantiles: Sequence[float] = (0.1, 0.5, 0.9),
        member_dim: str = "M",
        time_dim: str = "T",
        lat_dim: str = "Y",
        lon_dim: str = "X",
        # terciles:
        return_terciles: bool = True,
        clim_terciles: Optional[xr.DataArray] = None,
        obs_for_terciles: Optional[xr.DataArray] = None,
        tercile_q: Tuple[float, float] = (1 / 3, 2 / 3),
    ) -> xr.Dataset:
        if not self.fitted:
            raise ValueError("Fit model first.")
        _require_dims(new_forecasts, [member_dim, time_dim, lat_dim, lon_dim], "new_forecasts")

        w = self.weights.sel({lat_dim: new_forecasts[lat_dim], lon_dim: new_forecasts[lon_dim]}, method="nearest")
        logit = self.logit.sel({lat_dim: new_forecasts[lat_dim], lon_dim: new_forecasts[lon_dim]}, method="nearest")
        sh = self.g_shape.sel({lat_dim: new_forecasts[lat_dim], lon_dim: new_forecasts[lon_dim]}, method="nearest")
        sc = self.g_scale.sel({lat_dim: new_forecasts[lat_dim], lon_dim: new_forecasts[lon_dim]}, method="nearest")

        terc_ds = _resolve_terciles_for_prediction(
            new_forecasts=new_forecasts,
            lat_dim=lat_dim,
            lon_dim=lon_dim,
            time_dim=time_dim,
            clim_terciles=clim_terciles,
            obs_for_terciles=obs_for_terciles,
            stored_terciles=self.clim_terciles,
            tercile_q=tercile_q,
        )

        fc = np.maximum(new_forecasts.transpose(member_dim, time_dim, lat_dim, lon_dim).values, 0.0)

        wv = w.transpose(member_dim, lat_dim, lon_dim).values
        lv = logit.transpose("param", member_dim, lat_dim, lon_dim).values
        shv = sh.transpose(member_dim, lat_dim, lon_dim).values
        scv = sc.transpose(member_dim, lat_dim, lon_dim).values

        M, T, Y, X = fc.shape
        nq = len(quantiles)

        out_mean = np.full((T, Y, X), np.nan, float)
        out_q = np.full((nq, T, Y, X), np.nan, float)

        do_terc = return_terciles and (terc_ds is not None)
        terc_np = terc_ds.transpose("tercile", lat_dim, lon_dim).values if do_terc else None
        out_probs = np.full((3, T, Y, X), np.nan, float) if do_terc else None

        for iy in tqdm(range(Y), desc="Predict"):
            for ix in range(X):
                wk = _normalize_weights(wv[:, iy, ix])
                a0 = lv[0, :, iy, ix]
                a1 = lv[1, :, iy, ix]
                a2 = lv[2, :, iy, ix]
                shape_k = shv[:, iy, ix]
                scale_k = scv[:, iy, ix]

                if not (np.isfinite(a0).all() and np.isfinite(a1).all() and np.isfinite(a2).all()):
                    continue
                if not (np.isfinite(shape_k).any() and np.isfinite(scale_k).any()):
                    continue

                low_th = up_th = np.nan
                if do_terc:
                    low_th, up_th = terc_np[:, iy, ix]

                for t in range(T):
                    x_m = fc[:, t, iy, ix]
                    if not np.isfinite(x_m).all():
                        continue

                    x_c = np.cbrt(x_m)
                    ind0 = (x_m == 0.0).astype(float)
                    p0 = expit(a0 + a1 * x_c + a2 * ind0)   # P(Y=0)
                    p0 = np.clip(p0, 1e-8, 1 - 1e-8)

                    # pragmatic mean proxy (consistent with many operational pipelines):
                    out_mean[t, iy, ix] = float(np.sum(wk * (1.0 - p0) * x_m))

                    def cdf_mix(q):
                        q = max(float(q), 0.0)
                        zq = q ** (1.0 / 3.0)
                        cdfz = gamma.cdf(zq, a=shape_k, scale=scale_k)
                        cdfz = np.nan_to_num(cdfz, nan=0.0)
                        return float(np.sum(wk * (p0 + (1.0 - p0) * cdfz)))

                    # quantiles by inversion on y-scale
                    hi = float(np.max(x_m) * 12.0 + 50.0)
                    for iq, qv in enumerate(quantiles):
                        qv = float(qv)
                        try:
                            # mass at zero: F(0) = sum w_k p0_k
                            p_at_0 = float(np.sum(wk * p0))
                            if qv <= p_at_0:
                                out_q[iq, t, iy, ix] = 0.0
                            else:
                                out_q[iq, t, iy, ix] = _invert_cdf(cdf_mix, qv, 0.0, hi)
                        except Exception:
                            pass

                    if do_terc and np.isfinite(low_th) and np.isfinite(up_th):
                        out_probs[:, t, iy, ix] = _tercile_probs_from_cdf(cdf_mix, float(low_th), float(up_th))

        coords = {time_dim: new_forecasts[time_dim], lat_dim: new_forecasts[lat_dim], lon_dim: new_forecasts[lon_dim]}
        ds = xr.Dataset()
        ds["predictive_mean"] = xr.DataArray(out_mean, dims=(time_dim, lat_dim, lon_dim), coords=coords)
        ds["predictive_quantiles"] = xr.DataArray(
            out_q, dims=("quantile", time_dim, lat_dim, lon_dim), coords={**coords, "quantile": list(quantiles)}
        )

        if do_terc:
            ds["tercile_thresholds"] = terc_ds
            ds["tercile_probability"] = xr.DataArray(
                out_probs,
                dims=("category", time_dim, lat_dim, lon_dim),
                coords={**coords, "category": ["PB", "PN", "PA"]},
            )

        return ds


# ============================================================
# 5) Schmeits & Kok (2010) precip variant, with terciles
# ============================================================

@dataclass
class WAS_PrecipBMA_SchmeitsKok2010(WAS_PrecipBMA_Sloughter2007):
    """
    Variant where P(Y=0) uses a shared logistic based on ensemble-mean cube-root:
        p0_t = logistic(a0 + a1 * mean_k x_{t,k}^{1/3})
    applied to all members.
    """

    def predict_probabilistic(
        self,
        new_forecasts: xr.DataArray,
        *,
        quantiles: Sequence[float] = (0.1, 0.5, 0.9),
        member_dim: str = "M",
        time_dim: str = "T",
        lat_dim: str = "Y",
        lon_dim: str = "X",
        # terciles:
        return_terciles: bool = True,
        clim_terciles: Optional[xr.DataArray] = None,
        obs_for_terciles: Optional[xr.DataArray] = None,
        tercile_q: Tuple[float, float] = (1 / 3, 2 / 3),
    ) -> xr.Dataset:
        if not self.fitted:
            raise ValueError("Fit model first.")
        _require_dims(new_forecasts, [member_dim, time_dim, lat_dim, lon_dim], "new_forecasts")

        w = self.weights.sel({lat_dim: new_forecasts[lat_dim], lon_dim: new_forecasts[lon_dim]}, method="nearest")
        logit = self.logit.sel({lat_dim: new_forecasts[lat_dim], lon_dim: new_forecasts[lon_dim]}, method="nearest")
        sh = self.g_shape.sel({lat_dim: new_forecasts[lat_dim], lon_dim: new_forecasts[lon_dim]}, method="nearest")
        sc = self.g_scale.sel({lat_dim: new_forecasts[lat_dim], lon_dim: new_forecasts[lon_dim]}, method="nearest")

        terc_ds = _resolve_terciles_for_prediction(
            new_forecasts=new_forecasts,
            lat_dim=lat_dim,
            lon_dim=lon_dim,
            time_dim=time_dim,
            clim_terciles=clim_terciles,
            obs_for_terciles=obs_for_terciles,
            stored_terciles=self.clim_terciles,
            tercile_q=tercile_q,
        )

        fc = np.maximum(new_forecasts.transpose(member_dim, time_dim, lat_dim, lon_dim).values, 0.0)
        wv = w.transpose(member_dim, lat_dim, lon_dim).values
        lv = logit.transpose("param", member_dim, lat_dim, lon_dim).values
        shv = sh.transpose(member_dim, lat_dim, lon_dim).values
        scv = sc.transpose(member_dim, lat_dim, lon_dim).values

        M, T, Y, X = fc.shape
        nq = len(quantiles)

        out_mean = np.full((T, Y, X), np.nan, float)
        out_q = np.full((nq, T, Y, X), np.nan, float)

        do_terc = return_terciles and (terc_ds is not None)
        terc_np = terc_ds.transpose("tercile", lat_dim, lon_dim).values if do_terc else None
        out_probs = np.full((3, T, Y, X), np.nan, float) if do_terc else None

        for iy in tqdm(range(Y), desc="Predict"):
            for ix in range(X):
                wk = _normalize_weights(wv[:, iy, ix])

                # use first member's (a0,a1) as shared (you may also refit shared parameters explicitly)
                a0 = float(lv[0, 0, iy, ix])
                a1 = float(lv[1, 0, iy, ix])

                shape_k = shv[:, iy, ix]
                scale_k = scv[:, iy, ix]
                if not (np.isfinite(shape_k).any() and np.isfinite(scale_k).any()):
                    continue

                low_th = up_th = np.nan
                if do_terc:
                    low_th, up_th = terc_np[:, iy, ix]

                for t in range(T):
                    x_m = fc[:, t, iy, ix]
                    if not np.isfinite(x_m).all():
                        continue

                    x_c_mean = float(np.mean(np.cbrt(x_m)))
                    p0 = float(expit(a0 + a1 * x_c_mean))
                    p0 = float(np.clip(p0, 1e-8, 1 - 1e-8))

                    out_mean[t, iy, ix] = float((1.0 - p0) * np.sum(wk * x_m))

                    def cdf_mix(q):
                        q = max(float(q), 0.0)
                        zq = q ** (1.0 / 3.0)
                        cdfz = gamma.cdf(zq, a=shape_k, scale=scale_k)
                        cdfz = np.nan_to_num(cdfz, nan=0.0)
                        return float(p0 + (1.0 - p0) * np.sum(wk * cdfz))

                    hi = float(np.max(x_m) * 12.0 + 50.0)
                    for iq, qv in enumerate(quantiles):
                        qv = float(qv)
                        try:
                            if qv <= p0:
                                out_q[iq, t, iy, ix] = 0.0
                            else:
                                out_q[iq, t, iy, ix] = _invert_cdf(cdf_mix, qv, 0.0, hi)
                        except Exception:
                            pass

                    if do_terc and np.isfinite(low_th) and np.isfinite(up_th):
                        out_probs[:, t, iy, ix] = _tercile_probs_from_cdf(cdf_mix, float(low_th), float(up_th))

        coords = {time_dim: new_forecasts[time_dim], lat_dim: new_forecasts[lat_dim], lon_dim: new_forecasts[lon_dim]}
        ds = xr.Dataset()
        ds["predictive_mean"] = xr.DataArray(out_mean, dims=(time_dim, lat_dim, lon_dim), coords=coords)
        ds["predictive_quantiles"] = xr.DataArray(
            out_q, dims=("quantile", time_dim, lat_dim, lon_dim), coords={**coords, "quantile": list(quantiles)}
        )

        if do_terc:
            ds["tercile_thresholds"] = terc_ds
            ds["tercile_probability"] = xr.DataArray(
                out_probs,
                dims=("category", time_dim, lat_dim, lon_dim),
                coords={**coords, "category": ["PB", "PN", "PA"]},
            )

        return ds


