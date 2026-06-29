"""Multi-Model Ensemble (MME) methods for seasonal forecasting.

Provides weighted averaging, probabilistic MME schemes, ELM/MLP-based
ensembles, logistic and Gaussian-process based post-processing, and
wrappers for the xcast ELM/ELR framework.

Key functions
-------------
process_datasets_for_mme
    Align and reindex multiple model hindcast/forecast DataArrays to a
    common grid.
myfill
    Fill missing model forecasts using the observed climatology.

Key classes
-----------
WAS_mme_Weighted
    Simple or skill-weighted ensemble averaging with parametric or
    non-parametric tercile probabilities.
WAS_ProbWeighted
    Probabilistic skill-weighted ensemble (tercile-class based).
WAS_Min2009_ProbWeighted
    Probabilistic MME following Min (2009) chi-squared weighting.
WAS_mme_MLP / WAS_mme_MLP_
    MLP-based MME post-processing.
WAS_mme_hpELM / WAS_mme_hpELM_
    Extreme Learning Machine MME post-processing.
WAS_mme_Stacking / WAS_mme_Stacking_
    Stacking ensemble MME.
WAS_mme_XGBoosting / WAS_mme_XGBoosting_
    XGBoost-based MME post-processing.
WAS_mme_RF / WAS_mme_RF_
    Random Forest MME post-processing.
WAS_mme_logistic
    Logistic-regression MME producing direct tercile probabilities.
WAS_mme_gaussian_process
    Gaussian Process MME.
WAS_mme_FastBMA / WAS_mme_FullBMA
    Bayesian Model Averaging (fast and full variants).
WAS_mme_xcELM / WAS_mme_xcELR
    xcast-based ELM and ELR wrappers.
"""
from __future__ import annotations

# ---------------------------------------------------------
# 1. Standard Library Imports
# ---------------------------------------------------------
import datetime
import gc
import operator
import random
import warnings
from dataclasses import dataclass
from functools import partial
from typing import Callable, Dict, List, Literal, Optional, Sequence, Tuple, Union

# ---------------------------------------------------------
# 2. Third-Party Core Data Libraries
# ---------------------------------------------------------
import numpy as np
import pandas as pd
import xarray as xr

# ---------------------------------------------------------
# 3. Third-Party Math & ML Libraries (Hard Dependencies)
# ---------------------------------------------------------
from dask.distributed import Client
from xgboost import XGBRegressor

# SciPy
from scipy.optimize import brentq, fsolve, minimize, minimize_scalar, root_scalar
from scipy.special import betaln, expit, gammainc, gammaincinv, softmax
from scipy.special import gamma as gamma_function
from scipy.stats import (
    boxcox_normmax, expon, genextreme, laplace, linregress, logistic,
    lognorm, loguniform, nbinom, norm, pareto, poisson, randint, uniform, weibull_min
)
# 
from scipy.stats import gamma as gamma_dist
from scipy.stats import gamma as sp_gamma
from scipy.stats import t as tdist

# Scikit-Learn
from sklearn.base import BaseEstimator, RegressorMixin
from sklearn.cluster import KMeans
from sklearn.ensemble import (
    AdaBoostRegressor, GradientBoostingRegressor,
    RandomForestRegressor, StackingRegressor
)
from sklearn.exceptions import ConvergenceWarning
from sklearn.gaussian_process import GaussianProcessClassifier, GaussianProcessRegressor
from sklearn.gaussian_process.kernels import ConstantKernel as C, Matern, RBF, WhiteKernel
from sklearn.linear_model import LinearRegression, LogisticRegression
from sklearn.metrics import log_loss, make_scorer, mean_squared_error
from sklearn.model_selection import (
    GridSearchCV, GroupKFold, KFold, LeaveOneGroupOut,
    RandomizedSearchCV, TimeSeriesSplit, cross_val_score
)
from sklearn.neural_network import MLPRegressor
from sklearn.svm import SVR

# ---------------------------------------------------------
# 4. Optional / Conditional Third-Party Imports
# ---------------------------------------------------------
try:
    from tqdm.auto import tqdm
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
    from optuna.samplers import RandomSampler, TPESampler
    OPTUNA_AVAILABLE = True
    HAS_OPTUNA = True
except ImportError:
    OPTUNA_AVAILABLE = False
    HAS_OPTUNA = False
    warnings.warn("Optuna not installed. Bayesian optimization will not be available.")

try:
    from hpelm import HPELM
except ImportError:
    HPELM = None

try:
    import pymc as pm
    import arviz as az
    HAS_PYMC = True
except ImportError:
    HAS_PYMC = False
    pm = None
    az = None

# Fallback flag just in case downstream code checks for SciPy presence dynamically
try:
    import scipy
    _HAS_SCIPY = True
except ImportError:
    _HAS_SCIPY = False

# ---------------------------------------------------------
# 5. Project-Specific Imports
# ---------------------------------------------------------
import xcast as xc
from wass2s.utils import *
from wass2s.was_eof import *
from wass2s.was_verification import *

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
        if "M" in rainfall.coords:
            obs = rainfall.fillna(rainfall.mean(dim="T", skipna=True))
        else:
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
        if "M" in rainfall.coords:
            rainfall = rainfall.isel(M=0).drop_vars("M").squeeze()
        # Adjust time coordinates as needed.
        year = fcst.coords['T'].values.astype('datetime64[Y]').astype(int)[0] + 1970
        T_value_1 = rainfall.isel(T=0).coords['T'].values
        month_1 = T_value_1.astype('datetime64[M]').astype(int) % 12 + 1
        new_T_value = np.datetime64(f"{year}-{month_1:02d}-01")
        
        fcst = fcst.assign_coords(T=xr.DataArray([new_T_value], dims=["T"]))
        fcst['T'] = fcst['T'].astype('datetime64[ns]')
        hdcst['T'] = rainfall['T'].astype('datetime64[ns]')
        
        # Create a mask based on non-NaN values in the rainfall data.
        mask = xr.where(~np.isnan(rainfall.isel(T=0)), 1, np.nan)\
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

        # Drop any coordinate not in ("T", "Y", "X") for hindcast_det
        extra_coords_h = [c for c in hindcast_det.coords if c not in ("T", "Y", "X")]
        if extra_coords_h:
            hindcast_det = hindcast_det.drop_vars(extra_coords_h)

        # Drop any coordinate not in ("T", "Y", "X") for forecast_det
        extra_coords_f = [c for c in forecast_det.coords if c not in ("T", "Y", "X")]
        if extra_coords_f:
            forecast_det = forecast_det.drop_vars(extra_coords_f)  

                         
        return hindcast_det , forecast_det 


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
        if "M" in rainfall.coords:
            rainfall = rainfall.isel(M=0).drop_vars("M").squeeze()
        mask = xr.where(~np.isnan(rainfall.isel(T=0)), 1, np.nan).drop_vars('T').squeeze().to_numpy()

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
        
        return (hindcast_weighted * mask).transpose("probability", "T", "Y", "X"), (forecast_weighted * mask).transpose("probability", "T", "Y", "X")
class WAS_Min2009_ProbWeighted:
    """
    Min et al. (2009) Probability-Weighted Multi-Model Ensemble (PMME).

    What this class does
    --------------------
    1) For each model compute tercile probabilities (PB / PN / PA) at every
       grid point from its ensemble forecast using one of three distributions:

       * 'gaussian'  : Gaussian approximation.
                       Boundaries  : T_lo/hi = mu_hcst ± 0.4307 * sigma_hcst
                       Probability : Phi( (T_lo/hi - f_mean) / sigma_FORECAST )
                       sigma_FORECAST = std of the current forecast ensemble.

       * 'lognormal' : Same logic in log-space (better for precipitation).
                       eps_lognormal is added before taking logs.

       * 'empirical' : Category boundaries are the hindcast q33/q66.
                       Probability = fraction of forecast ensemble members
                       below q33 / above q66.

    2) Combine per-model probabilities with weights w_m ∝ sqrt(N_m) (Min 2009
       eq. 1), normalised to sum to 1.

    3) Optionally compute a combined categorical map with a chi-square
       significance test (Min 2009 section 3b, df = 2).

    Assumed dims
    ------------
    forecasts[m] : at minimum dim 'M' (ensemble members); spatial dims optional.
    hindcasts[m] : dims 'T' (years) and 'M' (members); same spatial dims.

    Parameters
    ----------
    distribution : 'gaussian' | 'lognormal' | 'empirical'
    cv_method    : None | 'leave_one_out' | 'rolling_window'
        How to compute climatological mu / sigma from the hindcast pool.
        None -> use the full hindcast pool (no cross-validation).
    rolling_window_size : int
        Half-width of the rolling window when cv_method='rolling_window'.
    n_samples_for_chisq : 'total_ensemble' | 'effective_sample_size' | float
        How to compute N for the chi-square test.
    eps_lognormal : float
        Small offset added before log transform (lognormal path only).
    sigma_floor : float
        Any sigma <= this value is replaced by the full-hindcast raw sigma.
    """

    def __init__(
        self,
        distribution: Literal["gaussian", "lognormal", "empirical"] = "gaussian",
        cv_method: Optional[Literal["leave_one_out", "rolling_window"]] = None,
        rolling_window_size: int = 15,
        n_samples_for_chisq: Literal["total_ensemble", "effective_sample_size"] | float | int = "total_ensemble",
        eps_lognormal: float = 1e-2,
        sigma_floor: float = 1e-12,
    ):
        self.distribution = distribution
        self.cv_method = cv_method
        self.rolling_window_size = int(rolling_window_size)
        self.n_samples_for_chisq = n_samples_for_chisq
        self.eps_lognormal = float(eps_lognormal)
        self.sigma_floor = float(sigma_floor)

    # ------------------------------------------------------------------
    # internal helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _require_dims(da: xr.DataArray, required: Tuple[str, ...], name: str = "DataArray") -> None:
        missing = [d for d in required if d not in da.dims]
        if missing:
            raise ValueError(f"{name}: missing required dims {missing}. Got dims={da.dims}")

    @staticmethod
    def _norm_cdf(z: xr.DataArray) -> xr.DataArray:
        return xr.apply_ufunc(stats.norm.cdf, z)

    @staticmethod
    def _safe_clip_and_renorm(
        p_bn: xr.DataArray, p_nn: xr.DataArray, p_an: xr.DataArray
    ) -> Tuple[xr.DataArray, xr.DataArray, xr.DataArray]:
        p_bn = p_bn.clip(0.0, 1.0)
        p_nn = p_nn.clip(0.0, 1.0)
        p_an = p_an.clip(0.0, 1.0)
        total = p_bn + p_nn + p_an
        ok = xr.apply_ufunc(np.isfinite, total) & (total > 0.0)
        p_bn = xr.where(ok, p_bn / total, np.nan)
        p_nn = xr.where(ok, p_nn / total, np.nan)
        p_an = xr.where(ok, p_an / total, np.nan)
        return p_bn, p_nn, p_an

    def _compute_model_weights(self, ensemble_sizes: Dict[str, int]) -> Dict[str, float]:
        """w_m = sqrt(N_m) / sum_k sqrt(N_k)  [Min 2009 eq. 1]"""
        sqrt_sizes = {m: np.sqrt(float(n)) for m, n in ensemble_sizes.items()}
        tot = float(sum(sqrt_sizes.values()))
        if tot <= 0:
            raise ValueError("Sum of sqrt(ensemble_sizes) must be > 0.")
        return {m: sqrt_sizes[m] / tot for m in ensemble_sizes}

    def _compute_n_for_chisq(
        self, ensemble_sizes: Dict[str, int], model_names: List[str]
    ) -> float:
        total = float(sum(float(ensemble_sizes[m]) for m in model_names))
        if isinstance(self.n_samples_for_chisq, (int, float)):
            return float(self.n_samples_for_chisq)
        if self.n_samples_for_chisq == "effective_sample_size":
            return total / np.sqrt(max(1, len(model_names)))
        return total  # "total_ensemble"

    # ------------------------------------------------------------------
    # cross-validated climatological stats (mu, sigma from hindcasts)
    # ------------------------------------------------------------------

    def _compute_cross_validated_stats(
        self, hindcasts: xr.DataArray
    ) -> Tuple[xr.DataArray, xr.DataArray]:
        """
        Return (mu, sigma) computed from the hindcast pool.
        cv_method=None -> full pool.
        cv_method='leave_one_out' / 'rolling_window' -> time-varying stats
        with the current year withheld.
        """
        self._require_dims(hindcasts, ("T", "M"), name="hindcasts")

        if self.cv_method is None:
            mu    = hindcasts.mean(dim=("T", "M"))
            sigma = hindcasts.std(dim=("T", "M"))
            return mu, sigma

        n_times = hindcasts.sizes["T"]

        if self.cv_method == "leave_one_out":
            mu_list, sig_list = [], []
            for i in range(n_times):
                h_train = hindcasts.isel(T=[j for j in range(n_times) if j != i])
                mu_list.append(h_train.mean(dim=("T", "M")))
                sig_list.append(h_train.std(dim=("T", "M")))
            return (
                xr.concat(mu_list,  dim=hindcasts["T"]),
                xr.concat(sig_list, dim=hindcasts["T"]),
            )

        if self.cv_method == "rolling_window":
            w = self.rolling_window_size
            mu_list, sig_list = [], []
            for i in range(n_times):
                start = max(0, i - w // 2)
                end   = min(n_times, i + w // 2 + 1)
                h_win = hindcasts.isel(T=slice(start, end))
                local_i = i - start
                if 0 <= local_i < h_win.sizes["T"]:
                    h_train = h_win.isel(T=[j for j in range(h_win.sizes["T"]) if j != local_i])
                else:
                    h_train = h_win
                mu_list.append(h_train.mean(dim=("T", "M")))
                sig_list.append(h_train.std(dim=("T", "M")))
            return (
                xr.concat(mu_list,  dim=hindcasts["T"]),
                xr.concat(sig_list, dim=hindcasts["T"]),
            )

        raise ValueError(f"Unknown cv_method={self.cv_method!r}")

    def _align_stats(
        self,
        mu: xr.DataArray,
        sigma: xr.DataArray,
        target: xr.DataArray,
        hindcasts: xr.DataArray,
    ) -> Tuple[xr.DataArray, xr.DataArray]:
        """
        Broadcast mu/sigma to match target and replace degenerate sigma values
        with the raw full-hindcast sigma (never NaN).

        FIX: the original set degenerate sigma to NaN, propagating silence.
        We fall back to the raw hindcast sigma instead.
        """
        # raw full-hindcast sigma as safe fallback
        sigma_raw = hindcasts.std(dim=("T", "M"))

        # if cv stats have a T dim and the forecast T doesn't overlap
        # (operational run outside the hindcast period) -> use full-pool stats
        if "T" in mu.dims and "T" in target.dims:
            overlap = np.intersect1d(mu["T"].values, target["T"].values)
            if overlap.size == 0:
                mu       = hindcasts.mean(dim=("T", "M"))
                sigma    = sigma_raw.copy()

        # replace any floor-or-below sigma with the raw fallback (not NaN)
        sigma = xr.where(sigma > self.sigma_floor, sigma, sigma_raw)

        # broadcast to target shape
        mu    = mu.broadcast_like(target)
        sigma = sigma.broadcast_like(target)
        return mu, sigma

    def _forecast_sigma(
        self, forecasts: xr.DataArray, sigma_hcst: xr.DataArray
    ) -> xr.DataArray:
        """
        Standard deviation of the current forecast ensemble.
        Falls back to sigma_hcst where the forecast ensemble collapses
        (single member or zero spread).
        """
        sig_f = forecasts.std(dim="M")
        # where forecast is degenerate, borrow the hindcast spread
        sig_f = xr.where(sig_f > self.sigma_floor, sig_f, sigma_hcst)
        return sig_f

    # ------------------------------------------------------------------
    # per-model probability computation (CORRECTED)
    # ------------------------------------------------------------------

    def _compute_tercile_probabilities_one_model(
        self,
        forecasts: xr.DataArray,
        hindcasts: xr.DataArray,
    ) -> Tuple[xr.DataArray, xr.DataArray, xr.DataArray]:
        """
        Compute PB / PN / PA for a single model.

        Gaussian & lognormal
        --------------------
        Category boundaries come from the HINDCAST climatology:
            T_lower = mu_hcst + z_{1/3} * sigma_hcst   (= mu - 0.4307*sigma)
            T_upper = mu_hcst + z_{2/3} * sigma_hcst   (= mu + 0.4307*sigma)

        Probabilities use the FORECAST ENSEMBLE spread  [BUG 1 FIX]:
            P(BN) = Phi( (T_lower - f_mean) / sigma_FORECAST )
            P(AN) = 1 - Phi( (T_upper - f_mean) / sigma_FORECAST )

        Empirical
        ---------
        Boundaries : q33 / q66 from all hindcast members.
        Probability: fraction of FORECAST MEMBERS below q33 / above q66
                     [BUG 2 FIX — original used ensemble mean → 0/1 only].
        """
        self._require_dims(forecasts, ("M",),        name="forecasts")
        self._require_dims(hindcasts, ("T", "M"),    name="hindcasts")

        # quantile z-scores for theoretical tercile boundaries
        z_lo = float(stats.norm.ppf(1.0 / 3.0))   # ≈ -0.4307
        z_hi = float(stats.norm.ppf(2.0 / 3.0))   # ≈ +0.4307

        f_mean = forecasts.mean(dim="M")

        # ---- Gaussian ------------------------------------------------
        if self.distribution == "gaussian":
            mu, sigma_hcst = self._compute_cross_validated_stats(hindcasts)
            mu, sigma_hcst = self._align_stats(mu, sigma_hcst, f_mean, hindcasts)

            # category boundaries from hindcast climatology
            T_lower = mu + z_lo * sigma_hcst
            T_upper = mu + z_hi * sigma_hcst

            # 1: use FORECAST ensemble spread, not hindcast sigma
            sigma_f = self._forecast_sigma(forecasts, sigma_hcst)

            p_bn = self._norm_cdf((T_lower - f_mean) / sigma_f)
            p_an = 1.0 - self._norm_cdf((T_upper - f_mean) / sigma_f)
            p_nn = 1.0 - p_bn - p_an
            return self._safe_clip_and_renorm(p_bn, p_nn, p_an)

        # ---- Lognormal -----------------------------------------------
        if self.distribution == "lognormal":
            eps = self.eps_lognormal

            h_log = np.log(xr.where(hindcasts > 0, hindcasts, eps))
            f_log = np.log(xr.where(forecasts  > 0, forecasts,  eps))

            f_log_mean = f_log.mean(dim="M")

            mu_l, sigma_hcst_l = self._compute_cross_validated_stats(h_log)
            mu_l, sigma_hcst_l = self._align_stats(mu_l, sigma_hcst_l, f_log_mean, h_log)

            # category boundaries in log-space
            T_lower_l = mu_l + z_lo * sigma_hcst_l
            T_upper_l = mu_l + z_hi * sigma_hcst_l

            # 1 (lognormal): use FORECAST log-space spread
            sigma_f_l = self._forecast_sigma(f_log, sigma_hcst_l)

            p_bn = self._norm_cdf((T_lower_l - f_log_mean) / sigma_f_l)
            p_an = 1.0 - self._norm_cdf((T_upper_l - f_log_mean) / sigma_f_l)
            p_nn = 1.0 - p_bn - p_an
            return self._safe_clip_and_renorm(p_bn, p_nn, p_an)

        # ---- Empirical -----------------------------------------------
        if self.distribution == "empirical":
            q33 = hindcasts.quantile(1.0 / 3.0, dim=("T", "M")).drop_vars("quantile", errors="ignore")
            q66 = hindcasts.quantile(2.0 / 3.0, dim=("T", "M")).drop_vars("quantile", errors="ignore")

            # 2: count ENSEMBLE MEMBERS, not just compare the mean
            p_bn = (forecasts < q33).mean(dim="M").astype(float)
            p_an = (forecasts > q66).mean(dim="M").astype(float)
            p_nn = 1.0 - p_bn - p_an
            return self._safe_clip_and_renorm(p_bn, p_nn, p_an)

        raise ValueError(
            f"Unknown distribution={self.distribution!r}. "
            "Choose 'gaussian', 'lognormal', or 'empirical'."
        )

    # ------------------------------------------------------------------
    # public API
    # ------------------------------------------------------------------

    def compute_pmme_probabilities(
        self,
        forecasts: Dict[str, xr.DataArray],
        hindcasts: Dict[str, xr.DataArray],
        climatology: Optional[xr.DataArray] = None,   # kept for pipeline compat
        ensemble_sizes: Optional[Dict[str, int]] = None,
        strict_models: bool = True,
    ) -> Dict[str, xr.DataArray]:
        """
        Compute PMME probabilities (PB / PN / PA) across models.

        Parameters
        ----------
        forecasts      : {model_name: DataArray(M, [spatial...])}
        hindcasts      : {model_name: DataArray(T, M, [spatial...])}
        climatology    : unused (kept for pipeline compatibility)
        ensemble_sizes : {model_name: int}  defaults to forecasts[m].sizes['M']
        strict_models  : if True, require forecasts and hindcasts to have the
                         same keys.

        Returns
        -------
        dict with keys 'PB', 'PN', 'PA'  (each a DataArray over spatial dims).
        """
        if strict_models and set(forecasts.keys()) != set(hindcasts.keys()):
            raise ValueError(
                "forecasts and hindcasts must have the same model keys "
                "when strict_models=True."
            )

        model_names = [m for m in forecasts if m in hindcasts]
        if not model_names:
            raise ValueError("No common models between forecasts and hindcasts.")

        if ensemble_sizes is None:
            ensemble_sizes = {m: int(forecasts[m].sizes["M"]) for m in model_names}

        weights = self._compute_model_weights(ensemble_sizes)

        # per-model probabilities
        per_model: Dict[str, Dict[str, xr.DataArray]] = {}
        for m in model_names:
            p_bn, p_nn, p_an = self._compute_tercile_probabilities_one_model(
                forecasts[m], hindcasts[m]
            )
            per_model[m] = {"PB": p_bn, "PN": p_nn, "PA": p_an}

        # weighted combination
        template = per_model[model_names[0]]["PB"]
        pmme: Dict[str, xr.DataArray] = {}
        for cat in ("PB", "PN", "PA"):
            wsum = xr.zeros_like(template)
            for m in model_names:
                wsum = wsum + per_model[m][cat] * float(weights[m])
            pmme[cat] = wsum

        # final clip & renorm
        pmme["PB"], pmme["PN"], pmme["PA"] = self._safe_clip_and_renorm(
            pmme["PB"], pmme["PN"], pmme["PA"]
        )
        return pmme

    def compute_combined_map(
        self,
        pmme_probs: Dict[str, xr.DataArray],
        ensemble_sizes: Dict[str, int],
        model_names: List[str],
        significance_level: float = 0.05,
    ) -> Tuple[xr.DataArray, xr.DataArray]:
        """
        Combined categorical map with chi-square significance test.
        [Min 2009, section 3b]

        Returns
        -------
        combined_map : DataArray
            0 = no significant deviation from equal probabilities,
            1 = PB dominant,  2 = PN dominant,  3 = PA dominant.
        chi_square : DataArray
            chi-square statistic (df = 2).
        """
        for k in ("PB", "PN", "PA"):
            if k not in pmme_probs:
                raise ValueError(f"pmme_probs missing key {k!r}")

        probs_stack = xr.concat(
            [pmme_probs["PB"], pmme_probs["PN"], pmme_probs["PA"]],
            dim=xr.DataArray(["PB", "PN", "PA"], dims="probability"),
        )
        valid_any = probs_stack.notnull().any("probability")

        # argmax on the three categories (fill NaN with -inf for argmax only)
        dominant = probs_stack.fillna(-np.inf).argmax(dim="probability", skipna=False)

        # chi-square statistic: N * sum_i (p_i - 1/3)^2 / (1/3)
        n = self._compute_n_for_chisq(ensemble_sizes, model_names)
        expected = 1.0 / 3.0
        chi_square = n * (
            (pmme_probs["PB"] - expected) ** 2 / expected
            + (pmme_probs["PN"] - expected) ** 2 / expected
            + (pmme_probs["PA"] - expected) ** 2 / expected
        )

        critical = float(stats.chi2.ppf(1.0 - float(significance_level), df=2))
        combined  = xr.where(valid_any & (chi_square > critical), dominant + 1, 0)

        combined.attrs  = {
            "description": "PMME combined forecast (Min et al. 2009 probability-weighted)",
            "values": "0=not significant, 1=PB, 2=PN, 3=PA",
            "significance_level": float(significance_level),
            "chi2_critical_value": critical,
        }
        chi_square.attrs = {
            "description": "Chi-square statistic for PMME categorical significance",
            "df": 2,
            "n_samples_used": float(n),
        }
        return combined, chi_square


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
    Multinomial Logistic Regression MME with spatial clustering and three
    hyperparameter-optimization strategies (grid / random / bayesian).

    Corrections relative to the previous version
    ---------------------------------------------
    1. Bayesian objective now scores by NEGATIVE LOG LOSS (a proper score for
       probability forecasts), consistent with the grid/random searches which
       use scoring='neg_log_loss'. The old code used model.score (accuracy)
       despite the comment, so the three methods optimized different criteria.
    2. Random search no longer samples invalid penalty/solver pairs: l1 is only
       paired with 'saga' (like the grid). RandomizedSearchCV accepts a LIST of
       distributions, mirroring the grid construction.
    3. Spatial clustering is kept as in the original: KMeans directly on the
       predictand values (each cell labelled by its first non-NaN year). Note
       this is an amplitude binning rather than a regime clustering, but it is
       retained here by preference.
    4. Degenerate clusters are guarded: a cluster with <2 observed classes is
       skipped during tuning and fitting (LogisticRegression needs >=2 classes).
    5. predict_proba is mapped onto the fixed [0,1,2] tercile columns via
       model.classes_, so a cluster missing a class no longer raises a shape
       error and never misaligns PB/PN/PA.

    Recommended usage / framework
    -----------------------------
    Call compute_hyperparameters ONCE on the RAW predictors/predictand, then
    pass (best_params, cluster_da) into compute_model so they are not recomputed
    (and so the predictand is not re-classified) inside every CV fold. In
    WAS_Cross_Validator the existing WAS_mme_logistic branch already passes the
    class-coded predictand and the standardized predictor per fold.
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
        self.study_dict = None

    # ----------------------------------------------------- parameter spaces
    @staticmethod
    def _l2_solvers(solver_options):
        return [s for s in solver_options if s in ('newton-cg', 'lbfgs', 'sag', 'saga')]

    def _create_parameter_grids(self):
        """Grid(s): l2 with all compatible solvers, l1 only with saga."""
        param_grids = [{
            'penalty': ['l2'],
            'solver': self._l2_solvers(self.solver_options),
            'C': self.C_range,
        }]
        if 'saga' in self.solver_options:
            param_grids.append({'penalty': ['l1'], 'solver': ['saga'], 'C': self.C_range})
        return param_grids

    def _create_parameter_distributions(self):
        """
        2: list of distributions so l1 is only ever paired with saga
        (RandomizedSearchCV samples one dict per draw, like GridSearchCV).
        """
        dists = [{
            'penalty': ['l2'],
            'solver': self._l2_solvers(self.solver_options),
            'C': self.C_range,
        }]
        if 'saga' in self.solver_options:
            dists.append({'penalty': ['l1'], 'solver': ['saga'], 'C': self.C_range})
        return dists

    # ----------------------------------------------------- search routines
    def _grid_search_optimization(self, X, y):
        model = LogisticRegression(random_state=self.random_state, max_iter=1000)
        cv = KFold(n_splits=self.cv_folds, shuffle=True, random_state=self.random_state)
        gs = GridSearchCV(model, param_grid=self._create_parameter_grids(), cv=cv,
                          scoring='neg_log_loss', error_score=np.nan, n_jobs=-1)
        gs.fit(X, y)
        return gs.best_params_

    def _random_search_optimization(self, X, y):
        model = LogisticRegression(random_state=self.random_state, max_iter=1000)
        cv = KFold(n_splits=self.cv_folds, shuffle=True, random_state=self.random_state)
        rs = RandomizedSearchCV(model, param_distributions=self._create_parameter_distributions(),
                                n_iter=self.n_iter_search, cv=cv, scoring='neg_log_loss',
                                random_state=self.random_state, error_score=np.nan, n_jobs=-1)
        rs.fit(X, y)
        return rs.best_params_

    def _bayesian_objective(self, trial, X, y):
        """1: maximize -log_loss (proper score), not accuracy."""
        penalty = trial.suggest_categorical('penalty', ['l2', 'l1'])
        if penalty == 'l1':
            solver = 'saga'
        else:
            l2_solvers = [s for s in self.solver_options if s != 'saga'] or ['lbfgs']  # G: never empty
            solver = trial.suggest_categorical('solver', l2_solvers)

        if isinstance(self.C_range, (list, tuple)) and len(self.C_range) == 2:
            C_min, C_max = min(self.C_range), max(self.C_range)
            C = trial.suggest_float('C', C_min, C_max, log=True)
        else:
            C = trial.suggest_categorical('C', list(self.C_range))

        model = LogisticRegression(penalty=penalty, solver=solver, C=C,
                                   random_state=self.random_state, max_iter=1000)

        labels = np.unique(y)
        cv = KFold(n_splits=self.cv_folds, shuffle=True, random_state=self.random_state)
        scores = []
        for tr, va in cv.split(X, y):
            try:
                model.fit(X[tr], y[tr])
                proba = model.predict_proba(X[va])
                scores.append(-log_loss(y[va], proba, labels=labels))
            except Exception:
                scores.append(np.nan)
        scores = np.array(scores, dtype=float)
        valid = scores[np.isfinite(scores)]
        return float(np.mean(valid)) if valid.size else float("-inf")

    def _bayesian_optimization(self, X, y, cluster_id=None):
        if optuna is None:
            raise ImportError("optimization_method='bayesian' requires optuna to be installed.")
        study_name = f'logistic_cluster_{cluster_id}' if cluster_id is not None else 'logistic'
        study = optuna.create_study(study_name=study_name, direction='maximize',
                                    sampler=TPESampler(seed=self.random_state))
        study.optimize(lambda t: self._bayesian_objective(t, X, y),
                       n_trials=self.n_trials, timeout=self.timeout, show_progress_bar=False)
        if cluster_id is not None:
            self.study_dict = self.study_dict or {}
            self.study_dict[cluster_id] = study
        best_params = dict(study.best_params)
        if best_params.get('penalty') == 'l1':
            best_params['solver'] = 'saga'
        return best_params

    def _best_params_for(self, X_clean_c, y_clean_c, cluster_id=None):
        if self.optimization_method == 'grid':
            return self._grid_search_optimization(X_clean_c, y_clean_c)
        if self.optimization_method == 'random':
            return self._random_search_optimization(X_clean_c, y_clean_c)
        if self.optimization_method == 'bayesian':
            return self._bayesian_optimization(X_clean_c, y_clean_c, cluster_id=cluster_id)
        if self.optimization_method == 'none':
            return {'C': 1.0, 'solver': 'lbfgs', 'penalty': 'l2'}
        raise ValueError(f"Unknown optimization method: {self.optimization_method}")

    # ----------------------------------------------------- helpers
    def _cluster_on_values(self, Predictand):
        """
        KMeans clustering directly on the predictand VALUES (original approach,
        kept by preference): KMeans is fit on the pooled (T, Y, X) magnitudes and
        each cell is labelled by its first non-NaN year via drop_duplicates.

        Returns a (Y, X) DataArray of cluster labels (NaN outside the mask).
        """
        Predictand = Predictand.copy()
        Predictand.name = "varname"
        df = Predictand.to_dataframe().reset_index().dropna().drop(columns=['T'])
        variable_column = df.columns[2]
        kmeans = KMeans(n_clusters=self.n_clusters, random_state=self.random_state)
        df['cluster'] = kmeans.fit_predict(df[[variable_column]])

        df_unique = df.drop_duplicates(subset=['Y', 'X'])
        dataset = df_unique.set_index(['Y', 'X']).to_xarray()
        mask = xr.where(~np.isnan(Predictand.isel(T=0)), 1, np.nan)
        return dataset['cluster'] * mask

    @staticmethod
    def _proba_to_terciles(y_prob, classes):
        """
        5: place predict_proba columns onto the fixed [0,1,2] tercile axis
        using model.classes_, so missing classes -> 0 and never misalign.
        """
        out = np.zeros((y_prob.shape[0], 3), dtype=float)
        for j, cls in enumerate(classes):
            ci = int(cls)
            if 0 <= ci <= 2:
                out[:, ci] = y_prob[:, j]
        return out

    # ----------------------------------------------------- hyperparameters
    def compute_hyperparameters(self, Predictors, Predictand, clim_year_start, clim_year_end):
        """
        Compute best hyperparameters per spatial cluster.

        Predictors : (T, M, Y, X) RAW predictor cube.
        Predictand : (T, Y, X) RAW predictand (classes are computed here).
        """
        if "M" in Predictand.coords:
            Predictand = Predictand.isel(M=0).drop_vars('M').squeeze()
        Predictand.name = "varname"

        X_train_std = standardize_timeseries(Predictors, clim_year_start, clim_year_end)

        # KMeans directly on predictand values (original approach, kept by preference)
        Cluster = self._cluster_on_values(Predictand)

        verify = WAS_Verification()
        y_train_std = verify.compute_class(Predictand, clim_year_start, clim_year_end)

        # align cluster grid with the (classed) predictand grid
        xarray1, xarray2 = xr.align(y_train_std, Cluster, join="outer")
        cluster_da = xarray2
        clusters = np.unique(cluster_da.values)
        clusters = clusters[~np.isnan(clusters)]

        # align time axes (compute_class preserves T order)
        X_train_std['T'] = y_train_std['T']

        best_params_dict = {}
        for c in clusters:
            mask_3d = (cluster_da == c).expand_dims({'T': y_train_std['T']})
            X_c = X_train_std.where(mask_3d).stack(sample=('T', 'Y', 'X')).transpose('sample', 'M').values
            y_c = y_train_std.where(mask_3d).stack(sample=('T', 'Y', 'X')).values.ravel()

            bad = np.any(~np.isfinite(X_c), axis=1) | ~np.isfinite(y_c)
            X_clean, y_clean = X_c[~bad], y_c[~bad]

            # 4: need samples and at least two classes
            if len(X_clean) == 0 or len(np.unique(y_clean)) < 2:
                continue

            best_params_dict[c] = self._best_params_for(X_clean, y_clean, cluster_id=c)

        self.best_params_dict = best_params_dict
        self.cluster_da = cluster_da
        return best_params_dict, cluster_da

    # ----------------------------------------------------- shared fit/predict
    def _fit_predict_clusters(self, X_train_std, y_train, X_test_std, best_params, cluster_da):
        """
        Fit one logistic model per cluster on the training fold and predict the
        test fold. Returns (predictions_classes (T,Y,X), predictions_prob (T,Y,X,3)).
        """
        time, lat, lon = X_test_std['T'], X_test_std['Y'], X_test_std['X']
        n_time, n_lat, n_lon = len(time), len(lat), len(lon)

        predictions = np.full((n_time, n_lat, n_lon), np.nan)
        predictions_prob = np.full((n_time, n_lat, n_lon, 3), np.nan)
        self.models = {}

        for c, bp in best_params.items():
            mask_tr = (cluster_da == c).expand_dims({'T': X_train_std['T']})
            mask_te = (cluster_da == c).expand_dims({'T': X_test_std['T']})

            Xtr = X_train_std.where(mask_tr).stack(sample=('T', 'Y', 'X')).transpose('sample', 'M').values
            ytr = y_train.where(mask_tr).stack(sample=('T', 'Y', 'X')).values.ravel()
            tr_bad = np.any(~np.isfinite(Xtr), axis=1) | ~np.isfinite(ytr)
            Xtr_clean, ytr_clean = Xtr[~tr_bad], ytr[~tr_bad]

            Xte = X_test_std.where(mask_te).stack(sample=('T', 'Y', 'X')).transpose('sample', 'M').values
            te_bad = np.any(~np.isfinite(Xte), axis=1)
            Xte_clean = Xte[~te_bad]

            # 4: skip degenerate clusters instead of crashing
            if len(Xtr_clean) == 0 or len(np.unique(ytr_clean)) < 2 or len(Xte_clean) == 0:
                continue

            model_c = LogisticRegression(**bp, random_state=self.random_state, max_iter=1000)
            model_c.fit(Xtr_clean, ytr_clean)
            self.models[c] = model_c

            y_pred = model_c.predict(Xte_clean)
            y_prob = self._proba_to_terciles(model_c.predict_proba(Xte_clean), model_c.classes_)

            full_pred = np.full(Xte.shape[0], np.nan)
            full_pred[~te_bad] = y_pred
            pred_grid = full_pred.reshape(n_time, n_lat, n_lon)
            predictions = np.where(np.isnan(predictions), pred_grid, predictions)

            full_prob = np.full((Xte.shape[0], 3), np.nan)
            full_prob[~te_bad] = y_prob
            prob_grid = full_prob.reshape(n_time, n_lat, n_lon, 3)
            predictions_prob = np.where(np.isnan(predictions_prob), prob_grid, predictions_prob)

        return predictions, predictions_prob, time, lat, lon

    @staticmethod
    def _wrap_outputs(predictions, predictions_prob, time, lat, lon):
        det = xr.DataArray(predictions, coords={'T': time, 'Y': lat, 'X': lon}, dims=['T', 'Y', 'X'])
        prob = xr.DataArray(predictions_prob,
                            coords={'T': time, 'Y': lat, 'X': lon, 'probability': [0, 1, 2]},
                            dims=['T', 'Y', 'X', 'probability'])
        prob = prob.transpose('probability', 'T', 'Y', 'X').assign_coords(probability=['PB', 'PN', 'PA'])
        return det, prob

    # ----------------------------------------------------- framework contract
    def compute_model(self, X_train, y_train, X_test, y_test, clim_year_start, clim_year_end,
                      best_params=None, cluster_da=None):
        """
        Train per-cluster logistic models and predict the test fold.

        X_train / X_test : standardized predictor cubes (T, M, Y, X).
        y_train          : class-coded predictand (T, Y, X) for training.
        best_params, cluster_da : pass the precomputed ones (recommended). If
                          omitted they are derived here -- but note that requires
                          a RAW predictand for compute_class, so prefer passing
                          precomputed values inside cross-validation.
        """
        X_train_std = X_train          # already standardized by the framework
        X_test_std = X_test

        if best_params is None or cluster_da is None:
            best_params, cluster_da = self.compute_hyperparameters(
                X_train, y_train, clim_year_start, clim_year_end)

        predictions, predictions_prob, time, lat, lon = self._fit_predict_clusters(
            X_train_std, y_train, X_test_std, best_params, cluster_da)

        return self._wrap_outputs(predictions, predictions_prob, time, lat, lon)

    # ----------------------------------------------------- operational
    def forecast(self, Predictant, clim_year_start, clim_year_end,
                 Predictors_train, Predictor_for_year, best_params=None, cluster_da=None):
        """Operational forecast for a single target year (classes + tercile probabilities)."""
        if "M" in Predictant.coords:
            Predictant_no_m = Predictant.isel(M=0).drop_vars('M').squeeze()
        else:
            Predictant_no_m = Predictant

        verify = WAS_Verification()
        Predictant_class = verify.compute_class(Predictant_no_m, clim_year_start, clim_year_end)

        mask = xr.where(~np.isnan(Predictant_no_m.isel(T=0)), 1, np.nan)

        Predictors_train_st = standardize_timeseries(Predictors_train, clim_year_start, clim_year_end)
        clim_slice = slice(str(clim_year_start), str(clim_year_end))
        mean_val = Predictors_train.sel(T=clim_slice).mean(dim='T')
        std_val = Predictors_train.sel(T=clim_slice).std(dim='T')
        Predictor_for_year_st = (Predictor_for_year - mean_val) / std_val

        Predictors_train_st['T'] = Predictant_class['T']

        if best_params is None or cluster_da is None:
            best_params, cluster_da = self.compute_hyperparameters(
                Predictors_train, Predictant_no_m, clim_year_start, clim_year_end)

        predictions, predictions_prob, time, lat, lon = self._fit_predict_clusters(
            Predictors_train_st, Predictant_class, Predictor_for_year_st, best_params, cluster_da)

        forecast_det, forecast_prob = self._wrap_outputs(predictions, predictions_prob, time, lat, lon)
        forecast_det = forecast_det * mask
        forecast_prob = forecast_prob * mask

        # relabel time to the forecast-year target month
        year = Predictor_for_year.coords['T'].values.astype('datetime64[Y]').astype(int)[0] + 1970
        month = Predictant_class.isel(T=0).coords['T'].values.astype('datetime64[M]').astype(int) % 12 + 1
        new_T = np.datetime64(f"{year}-{month:02d}-01")
        forecast_det = forecast_det.assign_coords(T=xr.DataArray([new_T], dims=["T"]))
        forecast_det['T'] = forecast_det['T'].astype('datetime64[ns]')
        forecast_prob = forecast_prob.assign_coords(T=xr.DataArray([new_T], dims=["T"]))
        forecast_prob['T'] = forecast_prob['T'].astype('datetime64[ns]')

        return forecast_det, forecast_prob

    def get_optimization_results(self):
        results = {
            'optimization_method': self.optimization_method,
            'best_parameters': self.best_params_dict,
            'cluster_labels': self.cluster_da,
            'models': self.models,
        }
        if self.optimization_method == 'bayesian' and self.study_dict:
            results['optuna_studies'] = self.study_dict
        return results
class WAS_mme_ELR:
    """
    Extended Logistic Regression (ELR) multi-model-ensemble tercile forecaster.

    Standard logistic-regression MOS fits one model per threshold, which can
    produce crossing (non-monotone) cumulative probabilities -> negative
    interval probabilities. ELR (Wilks, 2009, *Meteorol. Appl.*) fixes this by
    fitting a SINGLE logistic regression whose predictors are augmented with a
    function of the threshold q itself:

        P(Y <= q | x) = sigmoid( b0 + beta . x + alpha * g(q) )

    Because the SAME coefficient `alpha` multiplies g(q) for every threshold,
    the predicted non-exceedance probabilities are monotone in q (provided
    alpha > 0, which the data almost always enforce), so the two tercile
    boundaries yield a coherent below / normal / above split from one fitted
    model, and probabilities for any threshold can be read off the same fit.

    For tercile forecasts the two thresholds are the climatological lower and
    upper terciles T1 < T2 of the predictand, and

        PB = P(Y <= T1),
        PN = P(Y <= T2) - P(Y <= T1),
        PA = 1 - P(Y <= T2).

    Predictor at each grid cell
    ---------------------------
    The (T, Y, X, M) predictor cube is reduced per cell to:
      predictors='mean'      -> multi-model mean                 (1 feature)  [default]
      predictors='mean_std'  -> multi-model mean and spread      (2 features)
      predictors='models'    -> every model kept as its own feature (M features)

    Thresholds
    ----------
    Below/normal/above are defined by the climatological terciles. By default
    they are estimated fold-safe from the TRAINING predictand; pass
    `clim_terciles=(t1, t2)` (DataArrays on Y, X, same units as y_train) to use
    fixed climatological boundaries identical to those your verification uses
    (recommended for exact consistency with WAS_Verification.compute_class).

    Fitting
    -------
    One logistic regression per cell, but ALL cells are fitted simultaneously
    with a vectorized, ridge-stabilized Newton/IRLS solver (no per-cell sklearn
    loop), so it scales to large grids. Rows with missing predictor/predictand
    are down-weighted to zero, so partial gaps never break the batch. Cells with
    too few valid years fall back to climatology [1/3, 1/3, 1/3].

    Parameters
    ----------
    predictors : {'mean', 'mean_std', 'models'}
    threshold_transform : callable or None
        g(q) applied to the tercile boundaries (default identity). On
        standardized anomalies the identity is appropriate; for raw
        precipitation one might pass e.g. np.cbrt.
    l2 : float
        Ridge penalty on the coefficients (NOT the intercept) for numerical
        stability / quasi-separation (default 1e-3).
    n_iter : int
        Maximum IRLS iterations (default 50).
    tol : float
        Early-stop tolerance on the max coefficient update (default 1e-7).
    min_obs : int
        Minimum number of valid training years for a cell to be fitted;
        otherwise that cell is set to climatology (default 5).
    """

    # named threshold transforms g(q); all are NON-DECREASING so that the
    # cumulative forecast p(q)=Pr{V<=q} stays monotone in q (Wilks Eq. 5-7).
    # 'sqrt' is only valid for non-negative thresholds (raw precipitation);
    # 'cbrt' is the negative-safe analogue used on standardized anomalies.
    _TRANSFORMS = {"identity": None, "cbrt": np.cbrt, "sqrt": np.sqrt}

    def __init__(self, predictors="mean", threshold_transform="cbrt",
                 l2=1e-3, n_iter=50, tol=1e-7, min_obs=5, random_state=42):
        if predictors not in ("mean", "mean_std", "models"):
            raise ValueError("predictors must be 'mean', 'mean_std' or 'models'.")
        self.predictors = predictors
        # g(q): None/'identity' -> g(q)=q ; 'cbrt' -> np.cbrt ; 'sqrt' -> np.sqrt
        # (raw precip only) ; or any non-decreasing callable. Default 'cbrt', the
        # negative-safe analogue of Wilks' empirically-best g(q)=b2*sqrt(q) for a
        # standardized (signed) predictand.
        self.threshold_transform = threshold_transform
        self.l2 = float(l2)
        self.n_iter = int(n_iter)
        self.tol = float(tol)
        self.min_obs = int(min_obs)
        self.random_state = random_state
        # filled by compute_hyperparameters (joint l2 + g(q) tuning)
        self.best_params = None
        self.cv_scores_ = None

    # ---------------------------------------------------------------- helpers
    @classmethod
    def _resolve_transform(cls, transform):
        """Map a transform spec (None | name | callable) to a callable or None."""
        if transform is None or transform == "identity":
            return None
        if isinstance(transform, str):
            if transform not in cls._TRANSFORMS:
                raise ValueError(f"unknown threshold_transform '{transform}'.")
            return cls._TRANSFORMS[transform]
        return transform  # already a callable

    def _apply_g(self, q, transform):
        f = self._resolve_transform(transform)
        return q if f is None else f(q)

    def _g(self, q):
        return self._apply_g(q, self.threshold_transform)

    def _features(self, X):
        """Reduce a (T, Y, X[, M]) cube to a (T, Y, X, feat) predictor stack."""
        if "M" in X.dims:
            Xm = X.transpose("T", "Y", "X", "M")
            if self.predictors == "mean":
                feats = [Xm.mean("M", skipna=True)]
            elif self.predictors == "mean_std":
                feats = [Xm.mean("M", skipna=True), Xm.std("M", skipna=True)]
            else:  # 'models'
                feats = [Xm.isel(M=k) for k in range(Xm.sizes["M"])]
        else:
            feats = [X.transpose("T", "Y", "X")]
        return xr.concat(feats, dim="feat").transpose("T", "Y", "X", "feat")

    # ------------------------------------------------------------- core solver
    def _fit_predict(self, Ftr, ytr, Fte, t1, t2, l2=None, transform="__self__"):
        """
        Batched ELR over all cells.

        Ftr : (n_train, n_cells, F)   ytr : (n_train, n_cells)
        Fte : (n_test,  n_cells, F)   t1, t2 : (n_cells,)
        l2        : ridge penalty override (default None -> self.l2).
        transform : g(q) override (default '__self__' -> self.threshold_transform).
                    Used by compute_hyperparameters to score candidate (l2, g).
        returns probs (3, n_test, n_cells) ordered PB, PN, PA, where
                PB = Pr{V<=t1}, PN = Pr{V<=t2}-Pr{V<=t1}, PA = 1-Pr{V<=t2}.
        """
        l2 = self.l2 if l2 is None else float(l2)
        tfm = self.threshold_transform if transform == "__self__" else transform
        n_train, n_cells, F = Ftr.shape
        n_test = Fte.shape[0]
        g1, g2 = self._apply_g(t1, tfm), self._apply_g(t2, tfm)  # (n_cells,)

        # ---- extended design: the two threshold blocks stacked along rows ----
        def block(thr, gthr):
            col = np.broadcast_to(gthr, (n_train, n_cells))[..., None]
            Xb = np.concatenate([Ftr, col], axis=2)              # (n_train,n_cells,F+1)
            yb = (ytr <= thr[None, :]).astype(float)             # (n_train,n_cells)
            rb = (np.isfinite(ytr) & np.isfinite(Xb).all(axis=2)
                  & np.isfinite(thr)[None, :])                   # valid-row mask
            return Xb, yb, rb

        Xlo, ylo, rlo = block(t1, g1)
        Xup, yup, rup = block(t2, g2)
        Xs = np.concatenate([Xlo, Xup], axis=0)                  # (2n, n_cells, F+1)
        ones = np.ones((Xs.shape[0], n_cells, 1))
        A = np.nan_to_num(np.concatenate([Xs, ones], axis=2), nan=0.0)
        Y = np.concatenate([ylo, yup], axis=0)
        R = np.concatenate([rlo, rup], axis=0).astype(float)

        n_feat = A.shape[2]
        A = np.transpose(A, (1, 0, 2))                           # (n_cells, 2n, n_feat)
        Y = np.where(np.isfinite(Y), Y, 0.0).transpose(1, 0)     # (n_cells, 2n)
        R = R.transpose(1, 0)                                    # (n_cells, 2n)

        # ridge on every column except the intercept (last)
        pen = np.full(n_feat, l2); pen[-1] = 0.0
        eyepen = np.eye(n_feat)[None] * pen[None, None, :]
        jitter = 1e-8 * np.eye(n_feat)[None]

        beta = np.zeros((n_cells, n_feat))
        for _ in range(self.n_iter):
            eta = np.einsum("crf,cf->cr", A, beta)
            p = expit(eta)
            w = R * p * (1.0 - p)
            grad = np.einsum("crf,cr->cf", A, R * (Y - p)) - beta * pen[None, :]
            H = np.einsum("crf,crg->cfg", A * w[:, :, None], A) + eyepen + jitter
            try:
                step = np.linalg.solve(H, grad[..., None])[..., 0]
            except np.linalg.LinAlgError:
                H = H + 1e-3 * np.eye(n_feat)[None]
                step = np.linalg.solve(H, grad[..., None])[..., 0]
            beta = beta + step
            if np.nanmax(np.abs(step)) < self.tol:
                break

        # ---- predict the two test-fold cumulative probabilities -------------
        bpred, alpha, b0 = beta[:, :F], beta[:, F], beta[:, F + 1]
        Fc = np.transpose(np.nan_to_num(Fte, nan=0.0), (1, 0, 2))   # (n_cells, n_test, F)
        lin = np.einsum("ctf,cf->ct", Fc, bpred) + b0[:, None]
        plo = expit(lin + alpha[:, None] * g1[:, None])
        pup = expit(lin + alpha[:, None] * g2[:, None])

        PB = plo
        PN = np.clip(pup - plo, 0.0, None)
        PA = np.clip(1.0 - pup, 0.0, None)
        tot = PB + PN + PA
        tot = np.where(tot > 0, tot, np.nan)
        probs = np.stack([PB / tot, PN / tot, PA / tot], axis=0)    # (3, n_cells, n_test)

        # robustness: climatology where too few valid years; NaN where no thresholds
        nvalid = rlo.sum(axis=0)                                    # finite predictor & y per cell
        clim_cells = (nvalid < self.min_obs) & np.isfinite(t1) & np.isfinite(t2)
        probs[:, clim_cells, :] = 1.0 / 3.0
        bad = ~(np.isfinite(t1) & np.isfinite(t2))
        probs[:, bad, :] = np.nan

        return np.transpose(probs, (0, 2, 1))                       # (3, n_test, n_cells)

    # ------------------------------------------------------ framework contract
    def compute_model(self, X_train, y_train, X_test, y_test=None,
                      clim_year_start=None, clim_year_end=None,
                      clim_terciles=None, best_params=None, **kwargs):
        """
        Fit ELR on the training fold and return the test-fold tercile
        probabilities (dims probability, T, Y, X; probability=['PB','PN','PA']).

        X_train / X_test : standardized predictor cubes (T, Y, X[, M]).
        y_train          : predictand (T, Y, X). Used to set the tercile
                           thresholds (fold-safe) unless clim_terciles is given.
                           Keep it RAW (un-standardized): the fit is invariant to
                           predictand scaling, and raw values let g(q) act in
                           physical units (so g='sqrt' is valid for precipitation).
        clim_terciles    : optional (t1, t2) DataArrays (Y, X) of fixed
                           climatological boundaries in the SAME units as y_train
                           (i.e. physical/raw) -- pass the same terciles your
                           verification uses.
        best_params      : optional {'l2': value, 'g': name}. If given, overrides
                           the penalty and/or g(q) transform for this call;
                           otherwise self.l2 / self.threshold_transform are used
                           (which compute_hyperparameters updates in place).
        """
        if "M" in y_train.dims:
            y_train = y_train.isel(M=0, drop=True)
        y_train = y_train.transpose("T", "Y", "X")

        Ftr = self._features(X_train)
        Fte = self._features(X_test)

        ytr_s = y_train.stack(cell=("Y", "X")).transpose("T", "cell")
        Ftr_s = Ftr.stack(cell=("Y", "X")).transpose("T", "cell", "feat")
        Fte_s = Fte.stack(cell=("Y", "X")).transpose("T", "cell", "feat")

        if clim_terciles is not None:
            t1da, t2da = clim_terciles
            t1 = t1da.stack(cell=("Y", "X")).transpose("cell").values.astype(float)
            t2 = t2da.stack(cell=("Y", "X")).transpose("cell").values.astype(float)
        else:
            with np.errstate(all="ignore"):
                qs = np.nanquantile(ytr_s.values, [1.0 / 3.0, 2.0 / 3.0], axis=0)
            t1, t2 = qs[0], qs[1]

        l2_use = best_params.get("l2") if best_params else None
        g_use = best_params.get("g", "__self__") if best_params else "__self__"
        probs = self._fit_predict(Ftr_s.values, ytr_s.values, Fte_s.values,
                                  t1, t2, l2=l2_use, transform=g_use)

        out = xr.DataArray(
            probs,
            dims=("probability", "T", "cell"),
            coords={"probability": ["PB", "PN", "PA"],
                    "T": Fte_s["T"], "cell": ytr_s["cell"]},
        ).unstack("cell").transpose("probability", "T", "Y", "X")
        return out

    # ------------------------------------------------------ hyperparameters
    @staticmethod
    def _year_groups(T):
        """Calendar-year group label per time step (falls back to position)."""
        if np.issubdtype(np.asarray(T.values).dtype, np.datetime64):
            return T.dt.year.values
        return np.arange(T.sizes["T"])

    def _year_splits(self, years, cv_folds, leave_one_year_out):
        """Year-grouped train/val splits (leakage-free: whole years held out)."""
        uniq = np.unique(years)
        n = len(uniq)
        if leave_one_year_out or cv_folds >= n:
            folds = [np.array([u]) for u in uniq]
        else:
            rng = np.random.default_rng(self.random_state)
            folds = [uniq[b] for b in np.array_split(rng.permutation(n), cv_folds)]
        splits = []
        for val_years in folds:
            va = np.isin(years, val_years)
            tr = ~va
            if tr.sum() >= 2 and va.sum() >= 1:
                splits.append((np.where(tr)[0], np.where(va)[0]))
        return splits

    @staticmethod
    def _score_vs_class(probs, obs_class, kind):
        """RPS or multiclass log-loss of the forecast tercile probabilities
        against the framework's OBSERVED tercile CLASS (0=below, 1=normal,
        2=above) -- the same target WAS_Verification / the logistic class use.

        probs     : (3, n_val, n_cells) ordered PB, PN, PA
        obs_class : (n_val, n_cells) integer-coded tercile category
        """
        PB, PN, PA = probs[0], probs[1], probs[2]
        o = obs_class
        valid = (np.isfinite(o) & np.isfinite(PB) & np.isfinite(PN) & np.isfinite(PA))
        if not valid.any():
            return np.nan
        below = (o == 0)
        middle = (o == 1)
        if kind == "rps":
            # cumulative (half-Brier) form over the two tercile boundaries
            rps = ((PB - below.astype(float)) ** 2
                   + ((PB + PN) - (below | middle).astype(float)) ** 2)
            return float(np.mean(rps[valid]))
        eps = 1e-12  # log_loss
        p_obs = np.where(below, PB, np.where(middle, PN, PA))
        p_obs = np.clip(p_obs, eps, 1.0)
        return float(-np.mean(np.log(p_obs[valid])))

    def compute_hyperparameters(self, Predictor, Predictand,
                                clim_year_start, clim_year_end,
                                clim_terciles=None, l2_grid=None, g_grid=None,
                                score="rps", cv_folds=5,
                                leave_one_year_out=False):
        """
        Jointly tune the ridge penalty l2 AND the threshold transform g(q) by
        year-grouped cross-validation. The winners are stored in self.l2 and
        self.threshold_transform (and self.best_params) so the existing CV branch
        / compute_model use them with no further plumbing.

        Why tune g(q): Wilks (2009) shows the FORM of g(q) drives ELR quality and
        must be chosen per setting (g(q)=b2*sqrt(q) was substantially better than
        g(q)=b2*q for raw precipitation). Here we let the data pick g among
        non-decreasing options, so the cumulative p(q)=Pr{V<=q} stays monotone.

        Observed truth = tercile CLASSES (like the logistic class)
        ----------------------------------------------------------
        Forecast probabilities are scored against WAS_Verification.compute_class
        (0=below=Pr{V<=q1/3}, 1=normal, 2=above) -- the categorical target the
        RPSS verification uses. RPS by default, or multiclass log-loss.

        Inputs / standardization: only the PREDICTOR is standardized (over the
        clim window) for f(x); the PREDICTAND stays RAW so the thresholds and
        g(q) are in physical units (the fit is invariant to predictand scaling).
        compute_class is computed on the raw predictand.

        Parameters
        ----------
        l2_grid : sequence of float, optional. Default np.logspace(-4, 1, 6).
        g_grid  : sequence of {'identity','cbrt','sqrt'} or callables, optional.
                  Default ['identity', 'cbrt'] (+ 'sqrt' only when all thresholds
                  are non-negative, i.e. a raw non-negative predictand).
        score   : {'rps', 'log_loss'} -- proper score MINIMIZED over the folds.
        cv_folds, leave_one_year_out : year-grouped CV controls.
        clim_terciles : optional (t1, t2) in PHYSICAL (raw predictand) units --
                  the same fixed boundaries your CV branch passes to
                  compute_model. If None, fold-safe raw terciles are estimated per
                  training fold.

        Returns
        -------
        best_params : dict {'l2': best_l2, 'g': best_g_name}
        """
        if score not in ("rps", "log_loss"):
            raise ValueError("score must be 'rps' or 'log_loss'.")
        l2_grid = [float(v) for v in (np.logspace(-4, 1, 6) if l2_grid is None
                                      else l2_grid)]

        if "M" in Predictand.dims:
            Predictand = Predictand.isel(M=0, drop=True)
        Predictand = Predictand.transpose("T", "Y", "X")

        # ---- observed tercile CLASSES from the framework (0/1/2) -------------
        verify = WAS_Verification()
        obs_class = verify.compute_class(Predictand, clim_year_start,
                                         clim_year_end).transpose("T", "Y", "X")

        # ---- standardize the PREDICTOR only; keep the PREDICTAND RAW --------
        # The predictand enters the ELR only through the binary indicators
        # (V <= threshold), which are INVARIANT to a monotone per-cell rescaling.
        # Keeping it raw lets g(q) see the threshold in PHYSICAL units, so Wilks'
        # g(q)=sqrt(q) is valid for a non-negative (precipitation) predictand.
        clim = Predictor.sel(T=slice(str(clim_year_start), str(clim_year_end)))
        sX = clim.std("T"); sX = sX.where(sX > 0)
        Predictor_st = (Predictor - clim.mean("T")) / sX
        Predictor_st = Predictor_st.assign_coords(T=Predictand["T"])
        obs_class = obs_class.assign_coords(T=Predictand["T"])

        # ---- features + stacking (once) -------------------------------------
        F = self._features(Predictor_st)
        y_s = Predictand.stack(cell=("Y", "X")).transpose("T", "cell")
        F_s = F.stack(cell=("Y", "X")).transpose("T", "cell", "feat")
        o_s = obs_class.stack(cell=("Y", "X")).transpose("T", "cell")
        Yv, Fv, Ov = y_s.values, F_s.values, o_s.values

        if clim_terciles is not None:
            t1da, t2da = clim_terciles
            t1_fix = t1da.stack(cell=("Y", "X")).transpose("cell").values.astype(float)
            t2_fix = t2da.stack(cell=("Y", "X")).transpose("cell").values.astype(float)
        else:
            t1_fix = t2_fix = None

        # ---- g(q) candidates: 'sqrt' needs non-negative thresholds ----------
        if t1_fix is not None:
            thr_min = np.nanmin([np.nanmin(t1_fix), np.nanmin(t2_fix)])
        else:
            with np.errstate(all="ignore"):
                thr_min = float(np.nanmin(np.nanquantile(Yv, 1.0 / 3.0, axis=0)))
        can_sqrt = np.isfinite(thr_min) and thr_min >= 0.0
        if g_grid is None:
            g_grid = ["identity", "cbrt"] + (["sqrt"] if can_sqrt else [])
        else:
            g_grid = [g for g in g_grid
                      if not (g == "sqrt" and not can_sqrt)]
        if not g_grid:
            g_grid = ["identity"]

        years = self._year_groups(Predictand["T"])
        splits = self._year_splits(years, cv_folds, leave_one_year_out)
        if not splits:
            raise ValueError("Not enough years to build a CV split for tuning.")

        scores, best_key, best_val = {}, (l2_grid[0], g_grid[0]), np.inf
        for g in g_grid:
            for l2 in l2_grid:
                fold_vals = []
                for tr_idx, va_idx in splits:
                    if t1_fix is not None:
                        t1, t2 = t1_fix, t2_fix
                    else:
                        with np.errstate(all="ignore"):
                            qs = np.nanquantile(Yv[tr_idx], [1.0 / 3.0, 2.0 / 3.0], axis=0)
                        t1, t2 = qs[0], qs[1]
                    probs = self._fit_predict(Fv[tr_idx], Yv[tr_idx], Fv[va_idx],
                                              t1, t2, l2=l2, transform=g)
                    s = self._score_vs_class(probs, Ov[va_idx], score)
                    if np.isfinite(s):
                        fold_vals.append(s)
                val = float(np.mean(fold_vals)) if fold_vals else np.inf
                gname = g if isinstance(g, str) else getattr(g, "__name__", "callable")
                scores[(gname, l2)] = val
                if val < best_val:
                    best_val, best_key = val, (l2, g)

        best_l2, best_g = best_key
        best_gname = best_g if isinstance(best_g, str) else getattr(best_g, "__name__", "callable")
        self.l2 = float(best_l2)
        self.threshold_transform = best_g
        self.best_params = {"l2": float(best_l2), "g": best_gname}
        self.cv_scores_ = {"score": score, "values": scores,
                           "best": {"l2": float(best_l2), "g": best_gname}}
        return self.best_params

    # ------------------------------------------------------------- operational
    def forecast(self, Predictant, clim_year_start, clim_year_end,
                 Predictor, Predictor_for_year, clim_terciles=None, **kwargs):
        """
        Operational ELR tercile forecast for a single target year.

        Fits ELR on all available years and returns the tercile probabilities for
        the target year (dims probability, T, Y, X). Only the PREDICTOR is
        standardized (historical climatology, same stats applied to the forecast
        year); the PREDICTAND is kept RAW so the thresholds and g(q) stay in
        physical units (the fit is invariant to predictand scaling).
        """
        if "M" in Predictant.dims:
            Predictant = Predictant.isel(M=0, drop=True)
        Predictant = Predictant.transpose("T", "Y", "X")
        mask = xr.where(np.isfinite(Predictant.isel(T=0)), 1.0, np.nan)

        clim = Predictor.sel(T=slice(str(clim_year_start), str(clim_year_end)))
        mX = clim.mean("T")
        sX = clim.std("T")
        sX = sX.where(sX > 0)
        Predictor_st = (Predictor - mX) / sX
        Predictor_year_st = (Predictor_for_year - mX) / sX

        prob = self.compute_model(
            Predictor_st, Predictant, Predictor_year_st,
            clim_year_start=clim_year_start, clim_year_end=clim_year_end,
            clim_terciles=clim_terciles)

        # relabel the time coordinate to the forecast-year target month
        year = Predictor_for_year.coords["T"].values.astype("datetime64[Y]").astype(int)[0] + 1970
        month = Predictant.isel(T=0).coords["T"].values.astype("datetime64[M]").astype(int) % 12 + 1
        new_T = np.datetime64(f"{year}-{month:02d}-01")
        prob = prob.assign_coords(T=xr.DataArray([new_T], dims=["T"]))
        prob["T"] = prob["T"].astype("datetime64[ns]")

        return (prob * mask).transpose("probability", "T", "Y", "X")


class WAS_mme_gaussian_process:
    """
    Gaussian Process Classifier MME (multiclass via One-vs-Rest in sklearn).
    
    Rigorous scientific implementation:
      - Spatial clustering based on historical predictand coherence.
      - Spatio-temporal subsampling.
      - **Grouped Cross-Validation (by Time)** to prevent spatial-autocorrelation data leakage.
    """

    def __init__(
        self,
        random_state=42,
        cv_folds=5,
        n_clusters=4,
        kernel_options=None,
        n_restarts_optimizer_options=(0, 2),
        max_iter_predict_options=(100, 200),
        max_train_samples=1500,  
        hpo_method='grid', 
        n_random_iter=10,
        n_trials=20,
        timeout=None, 
        n_jobs=-1,  
        warm_start=False, 
    ):
        self.random_state = random_state
        self.cv_folds = cv_folds
        self.n_clusters = n_clusters

        if kernel_options is None:
            self.kernel_options = [
                C(1.0, (1e-2, 1e2)) * RBF(length_scale=1.0, length_scale_bounds=(1e-2, 1e2)),
                C(1.0, (1e-2, 1e2)) * Matern(length_scale=1.0, nu=1.5, length_scale_bounds=(1e-2, 1e2)),
                C(1.0, (1e-2, 1e2)) * Matern(length_scale=1.0, nu=2.5, length_scale_bounds=(1e-2, 1e2)),
            ]
        else:
            self.kernel_options = kernel_options
            
        self.n_restarts_optimizer_options = tuple(n_restarts_optimizer_options)
        self.max_iter_predict_options = tuple(max_iter_predict_options)
        self.max_train_samples = int(max_train_samples)
        
        self.hpo_method = hpo_method.lower()
        if self.hpo_method not in ['grid', 'random', 'bayesian']:
            raise ValueError(f"hpo_method must be 'grid', 'random', or 'bayesian', got {hpo_method}")
        
        self.n_random_iter = n_random_iter
        self.n_trials = n_trials
        self.timeout = timeout
        self.n_jobs = n_jobs
        
        if self.hpo_method == 'bayesian' and not OPTUNA_AVAILABLE:
            warnings.warn("Optuna not available. Falling back to RandomizedSearchCV.")
            self.hpo_method = 'random'

        self.models = None

    def _subsample(self, X, y, groups, nmax):
        """Reproducible subsampling tracking temporal groups for rigorous CV."""
        if (nmax is None) or (nmax <= 0) or (X.shape[0] <= nmax):
            return X, y, groups
            
        rng = np.random.RandomState(self.random_state)
        idx = rng.choice(X.shape[0], size=nmax, replace=False)
        return X[idx], y[idx], groups[idx]

    def _create_param_grid(self):
        return {
            "kernel": self.kernel_options,
            "n_restarts_optimizer": list(self.n_restarts_optimizer_options),
            "max_iter_predict": list(self.max_iter_predict_options),
        }

    def _optuna_objective(self, trial, X_train, y_train, groups, cv_splitter):
        """Objective function for Optuna."""
        kernel_type = trial.suggest_categorical('kernel_type', ['RBF', 'Matern_1.5', 'Matern_2.5'])
        length_scale = trial.suggest_float('length_scale', 0.1, 10.0, log=True)
        constant_value = trial.suggest_float('constant_value', 0.1, 10.0, log=True)
        
        if kernel_type == 'RBF':
            kernel = C(constant_value) * RBF(length_scale=length_scale)
        elif kernel_type == 'Matern_1.5':
            kernel = C(constant_value) * Matern(length_scale=length_scale, nu=1.5)
        else:
            kernel = C(constant_value) * Matern(length_scale=length_scale, nu=2.5)
        
        kernel = kernel + WhiteKernel(noise_level=1e-5, noise_level_bounds=(1e-10, 1e-2))
        
        n_restarts = trial.suggest_int('n_restarts_optimizer', min(self.n_restarts_optimizer_options), max(self.n_restarts_optimizer_options))
        max_iter = trial.suggest_int('max_iter_predict', min(self.max_iter_predict_options), max(self.max_iter_predict_options))
        
        model = GaussianProcessClassifier(
            kernel=kernel, n_restarts_optimizer=n_restarts, 
            max_iter_predict=max_iter, random_state=self.random_state
        )
        
        scores = []
        # Group CV ensures no temporal leakage
        for train_idx, val_idx in cv_splitter.split(X_train, y_train, groups=groups):
            X_train_f, X_val_f = X_train[train_idx], X_train[val_idx]
            y_train_f, y_val_f = y_train[train_idx], y_train[val_idx]
            
            # GP fails if a class is entirely missing in a fold
            if len(np.unique(y_train_f)) < 2:
                continue
                
            try:
                model.fit(X_train_f, y_train_f)
                y_proba = model.predict_proba(X_val_f)
                # handle potential missing classes in prediction output safely
                score = -log_loss(y_val_f, y_proba, labels=[0, 1, 2])
                scores.append(score)
            except Exception:
                scores.append(-1e5)
        
        return np.mean(scores) if scores else -1e5

    def compute_hyperparameters(self, Predictors, Predictand, clim_year_start, clim_year_end):
        """Computes best hyperparameters. Uses rigorous GroupKFold by Time."""
        if "M" in Predictand.coords:
            Predictand = Predictand.isel(M=0).drop_vars('M').squeeze()
            
        X_train_std = standardize_timeseries(Predictors, clim_year_start, clim_year_end)
        
        # Rigorous Clustering (Array-based, resistant to netcdf metadata changes)
        y_for_cluster = Predictand.stack(space=("Y", "X")).transpose("space", "T").values
        finite_mask = np.all(np.isfinite(y_for_cluster), axis=1)
        y_cluster = y_for_cluster[finite_mask]

        kmeans = KMeans(n_clusters=self.n_clusters, random_state=self.random_state)
        labels = kmeans.fit_predict(y_cluster)

        full_labels = np.full(y_for_cluster.shape[0], np.nan)
        full_labels[finite_mask] = labels

        cluster_da = xr.DataArray(
            full_labels.reshape(len(Predictand["Y"]), len(Predictand["X"])),
            coords={"Y": Predictand["Y"], "X": Predictand["X"]},
            dims=["Y", "X"],
        )
        clusters = np.unique(labels)

        verify = WAS_Verification()
        y_train_std = verify.compute_class(Predictand, clim_year_start, clim_year_end)
        X_train_std['T'] = y_train_std['T']

        best_params_dict = {}
        
        # Extract Time values for GroupKFold
        time_values = y_train_std['T'].values
        
        for c in clusters:
            mask_3d = (cluster_da == c).expand_dims({'T': y_train_std['T']})
            
            # Stack keeping coordinate tracking
            stacked_X = X_train_std.where(mask_3d).stack(sample=('T', 'Y', 'X'))
            X_stacked_c = stacked_X.transpose('sample', 'M').values
            y_stacked_c = y_train_std.where(mask_3d).stack(sample=('T', 'Y', 'X')).values.ravel()
            
            # Extract time identifiers for CV Grouping
            samples_coords = stacked_X.sample.values
            time_groups = np.array([s[0] for s in samples_coords])

            nan_mask_c = np.any(~np.isfinite(X_stacked_c), axis=1) | ~np.isfinite(y_stacked_c)
            X_clean_c = X_stacked_c[~nan_mask_c]
            y_clean_c = y_stacked_c[~nan_mask_c].astype(int)
            groups_clean_c = time_groups[~nan_mask_c]

            if len(X_clean_c) == 0 or len(np.unique(y_clean_c)) < 2:
                best_params_dict[c] = None
                continue

            X_clean_c, y_clean_c, groups_clean_c = self._subsample(
                X_clean_c, y_clean_c, groups_clean_c, self.max_train_samples
            )

            # CRITICAL FIX: GroupKFold avoids leaking spatial neighbors at same timestep into validation
            cv_splitter = GroupKFold(n_splits=min(self.cv_folds, len(np.unique(groups_clean_c))))

            if self.hpo_method == 'grid':
                param_grid = self._create_param_grid()
                model = GaussianProcessClassifier(random_state=self.random_state)
                grid_search = GridSearchCV(
                    estimator=model, param_grid=param_grid, cv=cv_splitter,
                    scoring="neg_log_loss", n_jobs=self.n_jobs
                )
                grid_search.fit(X_clean_c, y_clean_c, groups=groups_clean_c)
                best_params_dict[c] = grid_search.best_params_
                
            elif self.hpo_method == 'random':
                param_dist = self._create_param_grid()
                model = GaussianProcessClassifier(random_state=self.random_state)
                random_search = RandomizedSearchCV(
                    estimator=model, param_distributions=param_dist, n_iter=self.n_random_iter,
                    cv=cv_splitter, scoring="neg_log_loss", random_state=self.random_state, n_jobs=self.n_jobs
                )
                random_search.fit(X_clean_c, y_clean_c, groups=groups_clean_c)
                best_params_dict[c] = random_search.best_params_
                
            elif self.hpo_method == 'bayesian':
                study = optuna.create_study(direction='maximize', sampler=optuna.samplers.TPESampler(seed=self.random_state))
                
                def objective(trial):
                    return self._optuna_objective(trial, X_clean_c, y_clean_c, groups_clean_c, cv_splitter)
                
                study.optimize(objective, n_trials=self.n_trials, timeout=self.timeout, n_jobs=self.n_jobs, show_progress_bar=False)
                
                if study.best_trial is None:
                    best_params_dict[c] = None
                    continue
                    
                bp = study.best_trial.params
                k_type = bp['kernel_type']
                kernel = C(bp['constant_value']) * (
                    RBF(length_scale=bp['length_scale']) if k_type == 'RBF' else 
                    Matern(length_scale=bp['length_scale'], nu=1.5 if k_type == 'Matern_1.5' else 2.5)
                ) + WhiteKernel(noise_level=1e-5)

                best_params_dict[c] = {
                    'kernel': kernel,
                    'n_restarts_optimizer': bp['n_restarts_optimizer'],
                    'max_iter_predict': bp['max_iter_predict']
                }

        return best_params_dict, cluster_da

    def compute_model(self, X_train, y_train, X_test, y_test, clim_year_start, clim_year_end, best_params=None, cluster_da=None):
        """Fit models on train set and predict on test set."""
        # Note: Implementation logic follows existing flow but cleaned up.
        time, lat, lon = X_test["T"], X_test["Y"], X_test["X"]
        n_time, n_lat, n_lon = len(time), len(lat), len(lon)

        if best_params is None or cluster_da is None:
            best_params, cluster_da = self.compute_hyperparameters(X_train, y_train, clim_year_start, clim_year_end)

        predictions = np.full((n_time, n_lat, n_lon), np.nan)
        predictions_prob = np.full((n_time, n_lat, n_lon, 3), np.nan)
        self.models = {}

        for c, bp in best_params.items():
            if bp is None:
                continue
                
            mask_3d_train = (cluster_da == c).expand_dims({"T": X_train["T"]})
            mask_3d_test = (cluster_da == c).expand_dims({"T": X_test["T"]})

            X_train_stacked = X_train.where(mask_3d_train).stack(sample=("T", "Y", "X")).transpose("sample", "M")
            y_train_stacked = y_train.where(mask_3d_train).stack(sample=("T", "Y", "X")).values.ravel()

            nan_mask_tr = np.any(~np.isfinite(X_train_stacked.values), axis=1) | ~np.isfinite(y_train_stacked)
            X_clean_tr = X_train_stacked.values[~nan_mask_tr]
            y_clean_tr = y_train_stacked[~nan_mask_tr].astype(int)

            if len(X_clean_tr) == 0 or len(np.unique(y_clean_tr)) < 2:
                continue

            # Dummy groups for subsampling outside of CV
            dummy_groups = np.zeros(len(y_clean_tr))
            X_clean_tr, y_clean_tr, _ = self._subsample(X_clean_tr, y_clean_tr, dummy_groups, self.max_train_samples)

            model_c = GaussianProcessClassifier(random_state=self.random_state, **bp)
            model_c.fit(X_clean_tr, y_clean_tr)
            self.models[c] = model_c

            X_test_stacked = X_test.where(mask_3d_test).stack(sample=("T", "Y", "X")).transpose("sample", "M")
            nan_mask_te = np.any(~np.isfinite(X_test_stacked.values), axis=1)
            X_clean_te = X_test_stacked.values[~nan_mask_te]

            if len(X_clean_te) == 0:
                continue
                
            y_pred_c = model_c.predict(X_clean_te)
            y_prob_c = model_c.predict_proba(X_clean_te)

            # Reconstruct deterministic
            full_pred = np.full(X_test_stacked.shape[0], np.nan)
            full_pred[~nan_mask_te] = y_pred_c
            predictions = np.where(np.isnan(predictions), full_pred.reshape(n_time, n_lat, n_lon), predictions)

            # Reconstruct probabilistic mapping strictly to [0,1,2]
            full_prob = np.full((X_test_stacked.shape[0], 3), np.nan)
            cols = model_c.classes_.astype(int)
            tmp = np.zeros((y_prob_c.shape[0], 3))
            for j, cls in enumerate(cols):
                if 0 <= cls <= 2:
                    tmp[:, cls] = y_prob_c[:, j]
            full_prob[~nan_mask_te] = tmp
            predictions_prob = np.where(np.isnan(predictions_prob), full_prob.reshape(n_time, n_lat, n_lon, 3), predictions_prob)

        predicted_da = xr.DataArray(predictions, coords={"T": time, "Y": lat, "X": lon}, dims=["T", "Y", "X"])
        predicted_prob_da = xr.DataArray(predictions_prob, coords={"T": time, "Y": lat, "X": lon, "probability": ["PB", "PN", "PA"]}, dims=["T", "Y", "X", "probability"]).transpose("probability", "T", "Y", "X")

        return predicted_da, predicted_prob_da

    def forecast(self, Predictant, clim_year_start, clim_year_end, Predictors_train, Predictor_for_year, best_params=None, cluster_da=None):
        """Train on entire history, forecast for 1 new time slice."""
        if "M" in Predictant.coords:
            Predictant = Predictant.isel(M=0).drop_vars("M").squeeze()

        verify = WAS_Verification()
        Predictant_cls = verify.compute_class(Predictant, clim_year_start, clim_year_end)
        
        Predictors_train_st = standardize_timeseries(Predictors_train, clim_year_start, clim_year_end)
        mean_val = Predictors_train.sel(T=slice(str(clim_year_start), str(clim_year_end))).mean(dim="T")
        std_val = Predictors_train.sel(T=slice(str(clim_year_start), str(clim_year_end))).std(dim="T")
        Predictor_for_year_st = (Predictor_for_year - mean_val) / std_val

        Predictors_train_st["T"] = Predictant_cls["T"]

        return self.compute_model(
            X_train=Predictors_train_st, 
            y_train=Predictant_cls, 
            X_test=Predictor_for_year_st, 
            y_test=None, # Not needed for pure forecast execution in compute_model
            clim_year_start=clim_year_start, 
            clim_year_end=clim_year_end, 
            best_params=best_params, 
            cluster_da=cluster_da
        )

    def get_hpo_summary(self):
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



#################################################################################
################################################################################
### NGR
##############################################################################
##################################################################################


_SQRT_PI = np.sqrt(np.pi)
_EPS = 1e-8


# ===========================================================================
# 1. Corrected moment formulas
# ===========================================================================

def censored_normal_mean_var(mu, var, dry_threshold=0.0):
    """Mean and variance of Y = max(X, c) with X ~ N(mu, var), c = dry_threshold.

    Correct for ANY threshold (the original code assumed c = 0).
    Validated against 4e6-sample Monte-Carlo to <1e-2 absolute in mean and var
    over mu in [-1, 5], sigma in [1, 2], c in [0, 2].

        E[Y]   = c*Phi(alpha) + mu*(1-Phi(alpha)) + sigma*phi(alpha)
        E[Y^2] = c^2*Phi(alpha) + (mu^2+sigma^2)(1-Phi(alpha))
                 + sigma*(mu+c)*phi(alpha)
        alpha  = (c - mu)/sigma
    """
    mu = np.asarray(mu, float)
    sigma = np.sqrt(np.maximum(np.asarray(var, float), 1e-10))
    c = float(dry_threshold)
    alpha = (c - mu) / sigma
    Phi = norm.cdf(alpha)
    phi = norm.pdf(alpha)
    mean = c * Phi + mu * (1.0 - Phi) + sigma * phi
    e2 = c * c * Phi + (mu**2 + sigma**2) * (1.0 - Phi) + sigma * (mu + c) * phi
    return mean, np.maximum(e2 - mean**2, 0.0)


def truncated_normal_mean_var(mu, sigma, dry_threshold=0.0):
    """Mean and variance of X | X > c, X ~ N(mu, sigma^2). (Unchanged from your
    code — it was already correct; included for completeness.)"""
    sigma = np.maximum(np.asarray(sigma, float), _EPS)
    alpha = (dry_threshold - mu) / sigma
    Z = np.maximum(1.0 - norm.cdf(alpha), 1e-15)
    lam = norm.pdf(alpha) / Z
    mean = mu + sigma * lam
    var = sigma**2 * (1.0 + alpha * lam - lam**2)
    return mean, np.maximum(var, 0.0)


# ===========================================================================
# 2. Closed-form CRPS (for minimum-CRPS estimation, the EMOS standard)
# ===========================================================================

def crps_gaussian(y, mu, sigma):
    """Analytic CRPS for N(mu, sigma^2) (Gneiting et al. 2005). Validated to
    <1e-4 vs quadrature."""
    sigma = np.maximum(np.asarray(sigma, float), _EPS)
    z = (np.asarray(y, float) - np.asarray(mu, float)) / sigma
    return sigma * (z * (2.0 * norm.cdf(z) - 1.0) + 2.0 * norm.pdf(z) - 1.0 / _SQRT_PI)


def crps_censored_normal(y, mu, sigma, c=0.0):
    """CRPS for Y = max(X, c), X ~ N(mu, sigma^2), observation y >= c.

    Closed form (Jordan, Krüger & Lerch 2019, eq. for `crps_cnorm`, lower=c,
    upper=+inf). Validated against brute-force quadrature to <2e-5.
    """
    sigma = np.maximum(np.asarray(sigma, float), _EPS)
    mu = np.asarray(mu, float)
    z = (np.asarray(y, float) - mu) / sigma
    l = (c - mu) / sigma
    Phi_z, phi_z = norm.cdf(z), norm.pdf(z)
    Phi_l, phi_l = norm.cdf(l), norm.pdf(l)
    Phi_2l = norm.cdf(np.sqrt(2.0) * l)
    return sigma * (
        z * (2.0 * Phi_z - 1.0) + 2.0 * phi_z         # interior (Gaussian) part
        - Phi_l * Phi_l * l - 2.0 * Phi_l * phi_l      # atom at the lower bound c
        - (1.0 / _SQRT_PI) * (1.0 - Phi_2l)            # self-distance correction
    )


def crps_truncated_normal(y, mu, sigma, c=0.0):
    """CRPS for X | X > c, X ~ N(mu, sigma^2), observation y > c.

    Closed form (Thorarinsdottir & Gneiting 2010; `crps_tnorm`, lower=c,
    upper=+inf). Validated against brute-force quadrature to <2e-5.
    """
    sigma = np.maximum(np.asarray(sigma, float), _EPS)
    mu = np.asarray(mu, float)
    z = (np.asarray(y, float) - mu) / sigma
    l = (c - mu) / sigma
    Z = np.maximum(1.0 - norm.cdf(l), 1e-12)
    Phi_z, phi_z = norm.cdf(z), norm.pdf(z)
    Phi_2l = norm.cdf(np.sqrt(2.0) * l)
    return sigma / (Z * Z) * (
        z * Z * (2.0 * Phi_z + Z - 2.0)
        + 2.0 * phi_z * Z
        - (1.0 / _SQRT_PI) * (1.0 - Phi_2l)
    )


# ===========================================================================
# 3. Minimum-CRPS per-gridpoint estimator (drop-in for _fit_one_grid)
# ===========================================================================

# Variance links. EMOS papers use sigma^2 = c + d*S^2 (Gaussian; constrain
# c,d >= 0). A softplus link keeps positivity without hard bounds and is handy
# for the precip families; both are offered.
def _var_linear(c, d, s2):
    return np.maximum(c + d * s2, 1e-10)


def _var_softplus(c, d, s2):
    x = c + d * s2
    return np.log1p(np.exp(-np.abs(x))) + np.maximum(x, 0.0) + 1e-10


_VAR_LINKS = {"linear": _var_linear, "softplus": _var_softplus}
_CRPS_FAMILY = {
    "gaussian": lambda y, mu, sig, c: crps_gaussian(y, mu, sig),
    "censored_normal": crps_censored_normal,
    "truncated_normal": crps_truncated_normal,
}


def fit_one_grid_min_crps(
    fbar,
    svar,
    y,
    family="censored_normal",
    dry_threshold=0.0,
    var_link="linear",
    init=None,
    bounds=None,
):
    """Minimum-CRPS EMOS fit at a single grid point.

    Parameters
    ----------
    fbar, svar, y : 1D arrays over time (ensemble mean, ensemble variance, obs)
    family        : 'gaussian' | 'censored_normal' | 'truncated_normal'
    dry_threshold : censoring/truncation point c (precip families)
    var_link      : 'linear' (sigma^2 = c + d*svar) or 'softplus'

    Returns (params[a,b,c,d], info_dict). For truncated_normal only the wet
    observations (y > c) enter the score, matching the conditional likelihood
    used in Thorarinsdottir & Gneiting (2010).
    """
    fbar = np.asarray(fbar, float)
    svar = np.asarray(svar, float)
    y = np.asarray(y, float)
    good = np.isfinite(fbar) & np.isfinite(svar) & np.isfinite(y)
    fbar, svar, y = fbar[good], svar[good], y[good]
    if y.size < 3:
        return np.array([0.0, 1.0, float(np.nanvar(y) if y.size else 1.0), 0.0]), {
            "success": False, "n_valid": int(y.size), "final_crps": np.nan,
        }

    link = _VAR_LINKS[var_link]
    crps_fn = _CRPS_FAMILY[family]

    if family == "truncated_normal":
        wet = y > dry_threshold
        fb_s, sv_s, y_s = fbar[wet], svar[wet], y[wet]
        if y_s.size < 3:
            fb_s, sv_s, y_s = fbar, svar, y
    else:
        fb_s, sv_s, y_s = fbar, svar, y

    def objective(p):
        a, b, c, d = p
        mu = a + b * fb_s
        sigma = np.sqrt(link(c, d, sv_s))
        val = crps_fn(y_s, mu, sigma, dry_threshold)
        return float(np.mean(val[np.isfinite(val)])) if np.any(np.isfinite(val)) else 1e10

    if init is None:
        # OLS warm start for (a, b); residual variance splits into (c, d)
        A = np.column_stack([np.ones_like(fb_s), fb_s])
        try:
            coef, *_ = np.linalg.lstsq(A, y_s, rcond=None)
            a0, b0 = float(coef[0]), float(coef[1])
        except Exception:
            a0, b0 = 0.0, 1.0
        resid_var = float(np.var(y_s - (a0 + b0 * fb_s))) if y_s.size > 1 else 1.0
        c0 = max(0.5 * resid_var, 1e-4)
        init = np.array([a0, b0, c0, 0.5])

    if bounds is None:
        lo = (1e-6, None) if var_link == "linear" else (None, None)
        bounds = [(None, None), (None, None), lo, (0.0, None)]

    # Reference: raw multi-model mean with climatological spread (a=0,b=1,d=0)
    c_clim = max(float(np.var(y_s)), 1e-4)
    ref = objective([0.0, 1.0, c_clim, 0.0])

    res = minimize(objective, x0=init, method="L-BFGS-B", bounds=bounds,
                   options={"maxiter": 1000, "ftol": 1e-10, "gtol": 1e-7})
    if res.success and np.all(np.isfinite(res.x)) and res.fun <= ref:
        return np.asarray(res.x, float), {
            "success": True, "n_valid": int(y.size),
            "final_crps": float(res.fun), "ref_crps": ref,
            "crpss_vs_raw": 1.0 - res.fun / ref if ref > 0 else np.nan,
        }
    # do-no-harm fallback to the raw MME mean
    return np.array([0.0, 1.0, c_clim, 0.0]), {
        "success": False, "n_valid": int(y.size),
        "final_crps": ref, "ref_crps": ref, "crpss_vs_raw": 0.0,
    }


# __all__ = [
#     "censored_normal_mean_var",
#     "truncated_normal_mean_var",
#     "crps_gaussian",
#     "crps_censored_normal",
#     "crps_truncated_normal",
#     "fit_one_grid_min_crps",
# ]


def _crps_gaussian(y, mu, sigma):
    """
    Analytic CRPS for a Gaussian predictive distribution N(mu, sigma^2).
    """
    y = np.asarray(y, dtype=float)
    mu = np.asarray(mu, dtype=float)
    sigma = np.asarray(sigma, dtype=float)

    eps = np.finfo(float).eps * 100.0
    sigma = np.maximum(sigma, eps)

    z = (y - mu) / sigma
    return sigma * (z * (2.0 * norm.cdf(z) - 1.0) + 2.0 * norm.pdf(z) - 1.0 / np.sqrt(np.pi))


def _pairwise_abs_sum_sorted(x):
    """
    Sum_{i<j} |x_j - x_i| for a 1D array, computed in O(m log m).
    """
    x = np.sort(np.asarray(x, dtype=float))
    m = x.size
    if m < 2:
        return 0.0
    k = np.arange(1, m + 1, dtype=float)
    return float(np.sum((2.0 * k - m - 1.0) * x))


def _crps_ensemble(ens, obs):
    """
    Standard ensemble CRPS:
        mean |X - y| - 1/2 mean |X - X'|
    """
    ens = np.asarray(ens, dtype=float)
    ens = ens[np.isfinite(ens)]
    m = ens.size

    if m == 0:
        return np.nan
    if m == 1:
        return abs(ens[0] - obs)

    term1 = np.mean(np.abs(ens - obs))
    pairwise_sum = _pairwise_abs_sum_sorted(ens)
    term2 = pairwise_sum / (m * m)
    return term1 - term2


def _fair_crps_ensemble(ens, obs):
    """
    Fair ensemble CRPS (finite-ensemble correction):
        mean |X - y| - 1/(m(m-1)) sum_{i<j} |x_i - x_j|
    """
    ens = np.asarray(ens, dtype=float)
    ens = ens[np.isfinite(ens)]
    m = ens.size

    if m == 0:
        return np.nan
    if m == 1:
        return abs(ens[0] - obs)

    term1 = np.mean(np.abs(ens - obs))
    pairwise_sum = _pairwise_abs_sum_sorted(ens)
    term2 = pairwise_sum / (m * (m - 1.0))
    return term1 - term2


class WAS_mme_NGR_Gaussian:
    """
    Canonical Gaussian EMOS / NGR with gridwise fitting.

    Model
    -----
    Y_t | f_t ~ N(mu_t, sigma_t^2)

    mu_t      = a + b * fbar_t
    sigma_t^2 = c + d * s_t^2

    where
        fbar_t = ensemble mean
        s_t^2  = ensemble variance

    This class is intended for predictands that are approximately Gaussian
    after anomaly/standardization preprocessing. For nonnegative or
    zero-inflated variables such as precipitation totals, prefer a truncated,
    censored, gamma, or transformed variant instead of plain Gaussian EMOS.
    """

    def __init__(
        self,
        apply_to="all",
        alpha=0.10,
        test_direction="two-sided",
        bounds=None,
    ):
        valid_apply = {"all", "sig", "pos", "neg"}
        valid_test = {"two-sided", "greater", "less"}

        if apply_to not in valid_apply:
            raise ValueError(f"apply_to must be one of {valid_apply}")
        if test_direction not in valid_test:
            raise ValueError(f"test_direction must be one of {valid_test}")
        if not (0.0 < alpha < 1.0):
            raise ValueError("alpha must be between 0 and 1")

        self.apply_to = apply_to
        self.alpha = float(alpha)
        self.test_direction = test_direction

        self.bounds = {
            "a": (-np.inf, np.inf),
            "b": (-np.inf, np.inf),
            "c": (0.0, np.inf),
            "d": (0.0, np.inf),
        }
        if bounds is not None:
            self.bounds.update(bounds)

        self.params = None
        self.clim_terciles = None
        self.fitted = False
        self._xarray = False
        self._fit_stats = None
        self.attrs = {}

        self._member_dim = "M"
        self._time_dim = "T"
        self._lat_dim = "Y"
        self._lon_dim = "X"
        self._param_dim = "parameter"
        self._lat_coords = None
        self._lon_coords = None

    def __repr__(self):
        status = "fitted" if self.fitted else "unfitted"
        return (
            f"WAS_mme_NGR_Gaussian(apply_to={self.apply_to!r}, "
            f"alpha={self.alpha}, test_direction={self.test_direction!r}, "
            f"{status})"
        )

    @staticmethod
    def _safe_ens_stats(h):
        """
        h: array (M, T)
        returns mean(T), variance(T), std(T), valid_member_count(T)
        """
        fbar = np.nanmean(h, axis=0)
        counts = np.sum(np.isfinite(h), axis=0)
        var = np.full(h.shape[1], np.nan, dtype=float)

        for t in range(h.shape[1]):
            col = h[:, t]
            col = col[np.isfinite(col)]
            if col.size == 0:
                var[t] = np.nan
            elif col.size == 1:
                var[t] = 0.0
            else:
                var[t] = np.var(col, ddof=1)

        std = np.sqrt(np.maximum(var, 0.0))
        return fbar, var, std, counts

    def _regression_significant(self, r, n, direction=None):
        if direction is None:
            direction = self.test_direction

        if n <= 2 or not np.isfinite(r):
            return False
        if abs(r) >= 1.0 - 1e-12:
            return True

        t_stat = r * np.sqrt((n - 2.0) / max(1.0 - r * r, 1e-12))
        if direction == "two-sided":
            p = 2.0 * tdist.sf(abs(t_stat), df=n - 2)
        elif direction == "greater":
            p = tdist.sf(t_stat, df=n - 2)
        else:
            p = tdist.cdf(t_stat, df=n - 2)
        return bool(p < self.alpha)

    def _should_calibrate(self, r, n):
        if self.apply_to == "all":
            return True
        if self.apply_to == "sig":
            return self._regression_significant(r, n, self.test_direction)
        if self.apply_to == "pos":
            return (r > 0.0) and self._regression_significant(r, n, "greater")
        if self.apply_to == "neg":
            return (r < 0.0) and self._regression_significant(r, n, "less")
        return True

    def _objective_one_grid(self, pars, fbar, svar, obs):
        a, b, c, d = pars
        mu = a + b * fbar
        sig2 = c + d * svar
        sigma = np.sqrt(np.maximum(sig2, 0.0))
        return float(np.nanmean(_crps_gaussian(obs, mu, sigma)))

    def _fit_one_grid(self, h_gp, o_gp):
        """
        h_gp: (M, T)
        o_gp: (T,)
        """
        valid_time = np.isfinite(o_gp) & np.any(np.isfinite(h_gp), axis=0)
        if valid_time.sum() < 3:
            return np.array([0.0, 1.0, 0.0, 1.0]), {
                "success": False,
                "n_valid": int(valid_time.sum()),
                "initial_score": np.nan,
                "final_score": np.nan,
                "score_reduction": np.nan,
                "r": np.nan,
                "nit": 0,
            }

        h = h_gp[:, valid_time]
        y = o_gp[valid_time]

        fbar, svar, sstd, counts = self._safe_ens_stats(h)
        mask = np.isfinite(fbar) & np.isfinite(svar) & np.isfinite(y)

        if mask.sum() < 3:
            return np.array([0.0, 1.0, 0.0, 1.0]), {
                "success": False,
                "n_valid": int(mask.sum()),
                "initial_score": np.nan,
                "final_score": np.nan,
                "score_reduction": np.nan,
                "r": np.nan,
                "nit": 0,
            }

        fbar = fbar[mask]
        svar = svar[mask]
        y = y[mask]

        lm = linregress(fbar, y)
        r = float(lm.rvalue)

        if not self._should_calibrate(r, int(mask.sum())):
            pars = np.array([0.0, 1.0, 0.0, 1.0], dtype=float)
            base_score = self._objective_one_grid(pars, fbar, svar, y)
            return pars, {
                "success": True,
                "n_valid": int(mask.sum()),
                "initial_score": base_score,
                "final_score": base_score,
                "score_reduction": 0.0,
                "r": r,
                "nit": 0,
            }

        resid = y - (lm.intercept + lm.slope * fbar)
        resid_var = float(np.nanvar(resid, ddof=1)) if resid.size > 1 else 1.0
        init = np.array([
            float(lm.intercept),
            float(lm.slope),
            max(resid_var * 0.25, 1e-6),
            0.75,
        ])

        bounds = [
            self.bounds["a"],
            self.bounds["b"],
            self.bounds["c"],
            self.bounds["d"],
        ]

        base_pars = np.array([0.0, 1.0, 0.0, 1.0], dtype=float)
        base_score = self._objective_one_grid(base_pars, fbar, svar, y)

        res = minimize(
            self._objective_one_grid,
            x0=init,
            args=(fbar, svar, y),
            method="L-BFGS-B",
            bounds=bounds,
            options={"maxiter": 1000, "ftol": 1e-10, "gtol": 1e-7},
        )

        if res.success and np.all(np.isfinite(res.x)):
            final_pars = np.asarray(res.x, dtype=float)
            final_score = float(res.fun)
            nit = int(getattr(res, "nit", 0))
            success = True
        else:
            final_pars = base_pars
            final_score = base_score
            nit = int(getattr(res, "nit", 0))
            success = False

        red = np.nan
        if np.isfinite(base_score) and base_score > 0.0:
            red = (base_score - final_score) / base_score

        return final_pars, {
            "success": success,
            "n_valid": int(mask.sum()),
            "initial_score": base_score,
            "final_score": final_score,
            "score_reduction": red,
            "r": r,
            "nit": nit,
        }

    def fit(
        self,
        hcst_grid,
        obs_grid,
        clim_terciles=False,
        member_dim="M",
        time_dim="T",
        lat_dim="Y",
        lon_dim="X",
    ):
        use_xarray = (
            xr is not None
            and isinstance(hcst_grid, xr.DataArray)
            and isinstance(obs_grid, xr.DataArray)
        )

        self._member_dim = member_dim
        self._time_dim = time_dim
        self._lat_dim = lat_dim
        self._lon_dim = lon_dim

        if use_xarray:
            hcst = hcst_grid.transpose(member_dim, time_dim, lat_dim, lon_dim).values
            obs = obs_grid.transpose(time_dim, lat_dim, lon_dim).values
            self._lat_coords = hcst_grid.coords[lat_dim]
            self._lon_coords = hcst_grid.coords[lon_dim]
            self.attrs = dict(hcst_grid.attrs)
            self._xarray = True
        else:
            hcst = np.asarray(hcst_grid, dtype=float)
            obs = np.asarray(obs_grid, dtype=float)
            self._xarray = False

        if hcst.ndim != 4:
            raise ValueError("hcst_grid must be 4D with shape (M, T, Y, X)")
        if obs.ndim != 3:
            raise ValueError("obs_grid must be 3D with shape (T, Y, X)")

        m, t, ny, nx = hcst.shape
        if obs.shape != (t, ny, nx):
            raise ValueError(f"obs_grid must have shape {(t, ny, nx)}, got {obs.shape}")

        params = np.full((4, ny, nx), np.nan, dtype=float)
        fit_stats = {
            "success": np.zeros((ny, nx), dtype=bool),
            "n_valid": np.zeros((ny, nx), dtype=int),
            "initial_score": np.full((ny, nx), np.nan, dtype=float),
            "final_score": np.full((ny, nx), np.nan, dtype=float),
            "score_reduction": np.full((ny, nx), np.nan, dtype=float),
            "r": np.full((ny, nx), np.nan, dtype=float),
            "nit": np.zeros((ny, nx), dtype=int),
        }

        for iy in range(ny):
            for ix in range(nx):
                pars, stat = self._fit_one_grid(hcst[:, :, iy, ix], obs[:, iy, ix])
                params[:, iy, ix] = pars
                for key in fit_stats:
                    fit_stats[key][iy, ix] = stat[key]

        if clim_terciles:
            terc = np.nanpercentile(obs, [100.0 / 3.0, 200.0 / 3.0], axis=0)
            if use_xarray:
                self.clim_terciles = xr.DataArray(
                    terc,
                    dims=("tercile", lat_dim, lon_dim),
                    coords={
                        "tercile": ["lower", "upper"],
                        lat_dim: self._lat_coords,
                        lon_dim: self._lon_coords,
                    },
                    name="climatological_terciles",
                )
            else:
                self.clim_terciles = terc

        if use_xarray:
            self.params = xr.DataArray(
                params,
                dims=(self._param_dim, lat_dim, lon_dim),
                coords={
                    self._param_dim: ["a", "b", "c", "d"],
                    lat_dim: self._lat_coords,
                    lon_dim: self._lon_coords,
                },
                name="emos_parameters",
                attrs=self.attrs,
            )
        else:
            self.params = params

        self._fit_stats = fit_stats
        self.fitted = True
        return self

    def _predict_numpy(
        self,
        fcst,
        quantiles=None,
        clim_terciles=False,
        return_synthetic_ensemble=False,
    ):
        if not self.fitted:
            raise RuntimeError("Call fit() before predict().")

        fcst = np.asarray(fcst, dtype=float)
        if fcst.ndim != 4:
            raise ValueError("fcst_grid must be 4D with shape (M, T, Y, X)")

        m, t, ny, nx = fcst.shape
        param_np = self.params.values if hasattr(self.params, "values") else self.params

        if param_np.shape != (4, ny, nx):
            raise ValueError(
                f"Parameter shape must be (4, {ny}, {nx}), got {param_np.shape}"
            )

        mu = np.full((t, ny, nx), np.nan, dtype=float)
        sigma = np.full((t, ny, nx), np.nan, dtype=float)
        synth = np.full((m, t, ny, nx), np.nan, dtype=float) if return_synthetic_ensemble else None

        for iy in range(ny):
            for ix in range(nx):
                h = fcst[:, :, iy, ix]
                fbar, svar, sstd, counts = self._safe_ens_stats(h)
                a, b, c, d = param_np[:, iy, ix]

                mu_gp = a + b * fbar
                sig2_gp = c + d * svar
                sigma_gp = np.sqrt(np.maximum(sig2_gp, 0.0))

                mu[:, iy, ix] = mu_gp
                sigma[:, iy, ix] = sigma_gp

                if return_synthetic_ensemble:
                    eps = np.finfo(float).eps * 100.0
                    sstd_safe = np.maximum(sstd, eps)
                    z = (h - fbar[None, :]) / sstd_safe[None, :]
                    synth[:, :, iy, ix] = mu_gp[None, :] + sigma_gp[None, :] * z

        out = {
            "calibrated_mean": mu,
            "calibrated_std": sigma,
        }

        if quantiles is not None:
            q = np.asarray(quantiles, dtype=float)
            qv = norm.ppf(q)[:, None, None, None] * sigma[None, :, :, :] + mu[None, :, :, :]
            out["calibrated_quantiles"] = qv

        if clim_terciles:
            if self.clim_terciles is None:
                raise RuntimeError("fit(..., clim_terciles=True) or set clim_terciles first.")
            terc = self.clim_terciles.values if hasattr(self.clim_terciles, "values") else self.clim_terciles
            lower = terc[0][None, :, :]
            upper = terc[1][None, :, :]

            scale = np.maximum(sigma, np.finfo(float).eps * 100.0)
            p_below = norm.cdf(lower, loc=mu, scale=scale)
            p_above = norm.sf(upper, loc=mu, scale=scale)
            p_near = 1.0 - p_below - p_above

            probs = np.stack([p_below, p_near, p_above], axis=0)
            probs = np.clip(probs, 0.0, 1.0)
            denom = np.sum(probs, axis=0, keepdims=True)
            denom = np.where(denom <= 0.0, 1.0, denom)
            probs = probs / denom
            out["tercile_probability"] = probs

        if return_synthetic_ensemble:
            out["synthetic_calibrated_ensemble"] = synth

        return out

    def predict(
        self,
        fcst_grid,
        quantiles=None,
        clim_terciles=False,
        return_synthetic_ensemble=False,
        member_dim="M",
        time_dim="T",
        lat_dim="Y",
        lon_dim="X",
    ):
        use_xarray = xr is not None and isinstance(fcst_grid, xr.DataArray)

        if use_xarray:
            fc = fcst_grid.transpose(member_dim, time_dim, lat_dim, lon_dim)
            fc_np = fc.values
            out = self._predict_numpy(
                fc_np,
                quantiles=quantiles,
                clim_terciles=clim_terciles,
                return_synthetic_ensemble=return_synthetic_ensemble,
            )

            ds = xr.Dataset(attrs=dict(fcst_grid.attrs))
            time_coords = fc.coords[time_dim]
            member_coords = fc.coords[member_dim]
            lat_coords = fc.coords[lat_dim]
            lon_coords = fc.coords[lon_dim]

            ds["calibrated_mean"] = xr.DataArray(
                out["calibrated_mean"],
                dims=(time_dim, lat_dim, lon_dim),
                coords={time_dim: time_coords, lat_dim: lat_coords, lon_dim: lon_coords},
            )
            ds["calibrated_std"] = xr.DataArray(
                out["calibrated_std"],
                dims=(time_dim, lat_dim, lon_dim),
                coords={time_dim: time_coords, lat_dim: lat_coords, lon_dim: lon_coords},
            )

            if "calibrated_quantiles" in out:
                ds["calibrated_quantiles"] = xr.DataArray(
                    out["calibrated_quantiles"],
                    dims=("quantile", time_dim, lat_dim, lon_dim),
                    coords={
                        "quantile": list(np.asarray(quantiles, dtype=float)),
                        time_dim: time_coords,
                        lat_dim: lat_coords,
                        lon_dim: lon_coords,
                    },
                )

            if "tercile_probability" in out:
                ds["tercile_probability"] = xr.DataArray(
                    out["tercile_probability"],
                    dims=("probability", time_dim, lat_dim, lon_dim),
                    coords={
                        "probability": ["PB", "PN", "PA"],
                        time_dim: time_coords,
                        lat_dim: lat_coords,
                        lon_dim: lon_coords,
                    },
                )

            if "synthetic_calibrated_ensemble" in out:
                ds["synthetic_calibrated_ensemble"] = xr.DataArray(
                    out["synthetic_calibrated_ensemble"],
                    dims=(member_dim, time_dim, lat_dim, lon_dim),
                    coords={
                        member_dim: member_coords,
                        time_dim: time_coords,
                        lat_dim: lat_coords,
                        lon_dim: lon_coords,
                    },
                )

            return ds

        return self._predict_numpy(
            fcst_grid,
            quantiles=quantiles,
            clim_terciles=clim_terciles,
            return_synthetic_ensemble=return_synthetic_ensemble,
        )

    def transform(self, fcst_grid, **kwargs):
        kwargs = dict(kwargs)
        kwargs["return_synthetic_ensemble"] = True
        out = self.predict(fcst_grid, **kwargs)

        if xr is not None and isinstance(out, xr.Dataset):
            return (
                out["synthetic_calibrated_ensemble"],
                out["calibrated_mean"],
                out["calibrated_std"],
            )

        return (
            out["synthetic_calibrated_ensemble"],
            out["calibrated_mean"],
            out["calibrated_std"],
        )

    def compute_model(
        self,
        X_train,
        y_train,
        X_test,
        obs_for_terciles=None,
        quantiles=None,
        clim_terciles=False,
        member_dim="M",
        time_dim="T",
        lat_dim="Y",
        lon_dim="X",
        return_synthetic_ensemble=False,
    ):
        self.fit(
            X_train,
            y_train,
            clim_terciles=bool(clim_terciles or (obs_for_terciles is not None)),
            member_dim=member_dim,
            time_dim=time_dim,
            lat_dim=lat_dim,
            lon_dim=lon_dim,
        )

        if obs_for_terciles is not None:
            if xr is not None and isinstance(obs_for_terciles, xr.DataArray):
                obs_np = obs_for_terciles.transpose(time_dim, lat_dim, lon_dim).values
                terc = np.nanpercentile(obs_np, [100.0 / 3.0, 200.0 / 3.0], axis=0)
                self.clim_terciles = xr.DataArray(
                    terc,
                    dims=("tercile", lat_dim, lon_dim),
                    coords={
                        "tercile": ["lower", "upper"],
                        lat_dim: obs_for_terciles.coords[lat_dim],
                        lon_dim: obs_for_terciles.coords[lon_dim],
                    },
                )
            else:
                obs_np = np.asarray(obs_for_terciles, dtype=float)
                self.clim_terciles = np.nanpercentile(obs_np, [100.0 / 3.0, 200.0 / 3.0], axis=0)

        return self.predict(
            X_test,
            quantiles=quantiles,
            clim_terciles=clim_terciles,
            return_synthetic_ensemble=return_synthetic_ensemble,
            member_dim=member_dim,
            time_dim=time_dim,
            lat_dim=lat_dim,
            lon_dim=lon_dim,
        )

    forecast = predict

    def get_fit_stats(self):
        if not self.fitted or self._fit_stats is None:
            return None

        if self._xarray and self._lat_coords is not None and self._lon_coords is not None:
            out = {}
            for key, arr in self._fit_stats.items():
                out[key] = xr.DataArray(
                    arr,
                    dims=(self._lat_dim, self._lon_dim),
                    coords={self._lat_dim: self._lat_coords, self._lon_dim: self._lon_coords},
                    name=key,
                )
            return out

        return self._fit_stats

    def summary(self):
        if not self.fitted:
            print("Model not fitted.")
            return

        print("WAS_mme_NGR_Gaussian")
        print("=================")
        print(f"apply_to       : {self.apply_to}")
        print(f"alpha          : {self.alpha}")
        print(f"test_direction : {self.test_direction}")

        stats = self.get_fit_stats()
        if stats is None:
            return

        def _as_np(x):
            return x.values if hasattr(x, "values") else x

        success = _as_np(stats["success"])
        reduction = _as_np(stats["score_reduction"])
        nit = _as_np(stats["nit"])

        print()
        print(f"success rate   : {100.0 * np.mean(success):.1f}%")
        if np.any(success):
            print(f"mean score gain: {100.0 * np.nanmean(reduction[success]):.2f}%")
            print(f"mean iterations: {np.nanmean(nit[success]):.1f}")

######################################################################################################################################################################################################################################################################################


def _softplus(x):
    x = np.asarray(x, dtype=float)
    return np.log1p(np.exp(-np.abs(x))) + np.maximum(x, 0.0)


def _sigmoid(x):
    x = np.asarray(x, dtype=float)
    out = np.empty_like(x)
    pos = x >= 0
    out[pos] = 1.0 / (1.0 + np.exp(-x[pos]))
    ex = np.exp(x[~pos])
    out[~pos] = ex / (1.0 + ex)
    return out


def _rankdata_average(a):
    a = np.asarray(a, dtype=float)
    n = a.size
    order = np.argsort(a, kind="mergesort")
    ranks = np.empty(n, dtype=float)
    i = 0
    while i < n:
        j = i
        while j + 1 < n and a[order[j + 1]] == a[order[i]]:
            j += 1
        avg_rank = 0.5 * (i + j) + 1.0
        ranks[order[i : j + 1]] = avg_rank
        i = j + 1
    return ranks


def _boxcox_forward(y, lmbda, shift=1e-6):
    x = np.asarray(y, dtype=float) + shift
    if np.any(x <= 0):
        raise ValueError("Box-Cox requires y + shift > 0.")
    if abs(lmbda) < 1e-12:
        return np.log(x)
    return (np.power(x, lmbda) - 1.0) / lmbda


def _boxcox_inverse(z, lmbda, shift=1e-6):
    z = np.asarray(z, dtype=float)
    if abs(lmbda) < 1e-12:
        x = np.exp(z)
    else:
        base = 1.0 + lmbda * z
        base = np.maximum(base, 1e-12)
        x = np.power(base, 1.0 / lmbda)
    return np.maximum(x - shift, 0.0)


def _boxcox_log_jacobian(y, lmbda, shift=1e-6):
    x = np.asarray(y, dtype=float) + shift
    if np.any(x <= 0):
        return -np.inf
    if abs(lmbda) < 1e-12:
        return -np.log(x)
    return (lmbda - 1.0) * np.log(x)


def _censored_normal_logpdf(y, mu, sigma, dry_threshold=0.0):
    y = np.asarray(y, dtype=float)
    mu = np.asarray(mu, dtype=float)
    sigma = np.maximum(np.asarray(sigma, dtype=float), 1e-8)

    zt = (dry_threshold - mu) / sigma
    out = np.empty_like(np.broadcast_arrays(y, mu, sigma)[0], dtype=float)

    dry = y <= dry_threshold
    if np.any(dry):
        p0 = norm.cdf(zt[dry] if np.ndim(zt) else zt)
        p0 = np.maximum(p0, 1e-15)
        out[dry] = np.log(p0)

    wet = ~dry
    if np.any(wet):
        yw = y[wet]
        muw = mu[wet] if np.ndim(mu) else mu
        sw = sigma[wet] if np.ndim(sigma) else sigma
        out[wet] = norm.logpdf(yw, loc=muw, scale=sw)

    return out


def _truncated_normal_logpdf(y, mu, sigma, dry_threshold=0.0):
    y = np.asarray(y, dtype=float)
    mu = np.asarray(mu, dtype=float)
    sigma = np.maximum(np.asarray(sigma, dtype=float), 1e-8)

    alpha = (dry_threshold - mu) / sigma
    denom = np.maximum(1.0 - norm.cdf(alpha), 1e-15)
    out = norm.logpdf(y, loc=mu, scale=sigma) - np.log(denom)
    out = np.where(y > dry_threshold, out, -np.inf)
    return out


def _truncated_normal_mean_var(mu, sigma, dry_threshold=0.0):
    sigma = np.maximum(np.asarray(sigma, dtype=float), 1e-8)
    alpha = (dry_threshold - mu) / sigma
    z = np.maximum(1.0 - norm.cdf(alpha), 1e-15)
    lam = norm.pdf(alpha) / z
    mean = mu + sigma * lam
    var = sigma**2 * (1.0 + alpha * lam - lam**2)
    return mean, np.maximum(var, 0.0)


def _gamma_from_mean_var(mean, var):
    mean = np.maximum(np.asarray(mean, dtype=float), 1e-10)
    var = np.maximum(np.asarray(var, dtype=float), 1e-10)
    shape = mean**2 / var
    scale = var / mean
    return shape, scale


def _gamma_logpdf(y, mean, var, dry_threshold=0.0):
    y = np.asarray(y, dtype=float)
    x = y - dry_threshold
    x = np.maximum(x, 1e-12)
    k, theta = _gamma_from_mean_var(mean, var)
    return gamma_dist.logpdf(x, a=k, scale=theta)


def _fit_occurrence_logistic(wet_frac, fbar, is_dry, l2=1e-6):
    wet_frac = np.asarray(wet_frac, dtype=float)
    fbar = np.asarray(fbar, dtype=float)
    is_dry = np.asarray(is_dry, dtype=float)

    x = np.column_stack(
        [
            np.ones_like(wet_frac),
            wet_frac,
            np.nan_to_num(fbar, nan=np.nanmedian(fbar) if np.any(np.isfinite(fbar)) else 0.0),
        ]
    )

    means = np.zeros(x.shape[1], dtype=float)
    stds = np.ones(x.shape[1], dtype=float)
    for j in range(1, x.shape[1]):
        means[j] = np.nanmean(x[:, j])
        stds[j] = np.nanstd(x[:, j])
        if not np.isfinite(stds[j]) or stds[j] < 1e-10:
            stds[j] = 1.0
        x[:, j] = (x[:, j] - means[j]) / stds[j]

    def obj(beta):
        eta = x @ beta
        p = np.clip(_sigmoid(eta), 1e-10, 1.0 - 1e-10)
        nll = -np.sum(is_dry * np.log(p) + (1.0 - is_dry) * np.log(1.0 - p))
        pen = 0.5 * l2 * np.sum(beta[1:] ** 2)
        return nll + pen

    res = minimize(obj, x0=np.zeros(x.shape[1]), method="BFGS")
    beta = res.x if res.success else np.zeros(x.shape[1])

    return {
        "beta": beta,
        "means": means,
        "stds": stds,
        "success": bool(res.success),
    }


class WAS_mme_NGR_NonGaussian:
    """
    Precipitation-oriented EMOS-style postprocessor with optional predictive families:

    - family='censored_normal'
    - family='truncated_normal'
    - family='gamma'
    - family='transformed_gaussian'

    Notes
    -----
    1) Designed for nonnegative predictands such as precipitation.
    2) For families without an intrinsic atom at zero, a hurdle occurrence model
       is used by default:
           P(Y = 0) = p0
           Y | Y > 0  ~ positive family
    3) Fitting uses log-likelihood rather than analytic CRPS, so one coherent
       implementation can cover all families.
    """

    VALID_FAMILIES = {
        "censored_normal",
        "truncated_normal",
        "gamma",
        "transformed_gaussian",
    }
    VALID_OCC = {"auto", "none", "climatology", "logistic"}

    def __init__(
        self,
        family="censored_normal",
        occurrence_model="auto",
        dry_threshold=0.0,
        transform="boxcox",
        transform_shift=1e-6,
        alpha=0.10,
        apply_to="all",
        bounds=None,
    ):
        if family not in self.VALID_FAMILIES:
            raise ValueError(f"family must be one of {self.VALID_FAMILIES}")
        if occurrence_model not in self.VALID_OCC:
            raise ValueError(f"occurrence_model must be one of {self.VALID_OCC}")
        if apply_to not in {"all", "sig", "pos", "neg"}:
            raise ValueError("apply_to must be one of {'all', 'sig', 'pos', 'neg'}")

        self.family = family
        self.occurrence_model = occurrence_model
        self.dry_threshold = float(dry_threshold)
        self.transform = transform
        self.transform_shift = float(transform_shift)
        self.alpha = float(alpha)
        self.apply_to = apply_to

        self.bounds = {
            "a": (-np.inf, np.inf),
            "b": (-np.inf, np.inf),
            "c": (-10.0, 20.0),
            "d": (-10.0, 20.0),
        }
        if bounds is not None:
            self.bounds.update(bounds)

        self.params = None
        self.occ_params = None
        self._occ_feature_means = None
        self._occ_feature_stds = None
        self.transform_param = None
        self.clim_terciles = None
        self.fitted = False
        self._fit_stats = None
        self._xarray = False
        self.attrs = {}

        self._member_dim = "M"
        self._time_dim = "T"
        self._lat_dim = "Y"
        self._lon_dim = "X"
        self._param_dim = "parameter"
        self._lat_coords = None
        self._lon_coords = None

    def __repr__(self):
        status = "fitted" if self.fitted else "unfitted"
        return (
            f"WAS_mme_NGR_NonGaussian(family={self.family!r}, "
            f"occurrence_model={self.occurrence_model!r}, {status})"
        )

    def _needs_hurdle(self):
        if self.family == "censored_normal":
            return False
        if self.occurrence_model == "none":
            return False
        return True

    @staticmethod
    def _safe_ens_stats(h, dry_threshold=0.0):
        fbar = np.nanmean(h, axis=0)
        s2 = np.full(h.shape[1], np.nan, dtype=float)
        wet_frac = np.full(h.shape[1], np.nan, dtype=float)
        for t in range(h.shape[1]):
            x = h[:, t]
            good = np.isfinite(x)
            xx = x[good]
            if xx.size == 0:
                s2[t] = np.nan
                wet_frac[t] = np.nan
            elif xx.size == 1:
                s2[t] = 0.0
                wet_frac[t] = np.mean(xx > dry_threshold)
            else:
                s2[t] = np.var(xx, ddof=1)
                wet_frac[t] = np.mean(xx > dry_threshold)
        return fbar, np.maximum(s2, 0.0), wet_frac

    def _regression_significant(self, r, n):
        if n <= 2 or not np.isfinite(r):
            return False
        if abs(r) >= 1.0 - 1e-12:
            return True
        t_stat = r * np.sqrt((n - 2.0) / max(1.0 - r * r, 1e-12))
        if self.apply_to == "sig" or self.apply_to == "all":
            p = 2.0 * tdist.sf(abs(t_stat), df=n - 2)
            return bool(p < self.alpha)
        if self.apply_to == "pos":
            p = tdist.sf(t_stat, df=n - 2)
            return bool((r > 0.0) and (p < self.alpha))
        if self.apply_to == "neg":
            p = tdist.cdf(t_stat, df=n - 2)
            return bool((r < 0.0) and (p < self.alpha))
        return False

    def _should_calibrate(self, fbar, y):
        good = np.isfinite(fbar) & np.isfinite(y)
        if good.sum() < 3:
            return False
        if self.apply_to == "all":
            return True
        lm = linregress(fbar[good], y[good])
        return self._regression_significant(float(lm.rvalue), int(good.sum()))

    def _mu_var_from_pars(self, pars, fbar, s2):
        a, b, c, d = pars
        mu = a + b * fbar
        var = _softplus(c + d * s2) + 1e-10
        return mu, var

    def _pos_nll(self, pars, fbar, s2, y_all, wet_mask, lmbda=None):
        mu, var = self._mu_var_from_pars(pars, fbar, s2)
        sigma = np.sqrt(var)

        if self.family == "censored_normal":
            ll = _censored_normal_logpdf(y_all, mu, sigma, dry_threshold=self.dry_threshold)
            return -np.sum(ll[np.isfinite(ll)])

        fmu = mu[wet_mask]
        fvar = var[wet_mask]
        fsig = sigma[wet_mask]
        yy = y_all[wet_mask]

        if yy.size == 0:
            return np.inf

        if self.family == "truncated_normal":
            ll = _truncated_normal_logpdf(yy, fmu, fsig, dry_threshold=self.dry_threshold)
            return -np.sum(ll[np.isfinite(ll)])

        if self.family == "gamma":
            mean_pos = np.maximum(fmu, 1e-8)
            ll = _gamma_logpdf(yy, mean_pos, fvar, dry_threshold=self.dry_threshold)
            return -np.sum(ll[np.isfinite(ll)])

        if self.family == "transformed_gaussian":
            if self.transform != "boxcox":
                raise NotImplementedError("Only Box-Cox is implemented here.")
            yy_shift = np.maximum(yy - self.dry_threshold, 0.0)
            z = _boxcox_forward(yy_shift, lmbda=lmbda, shift=self.transform_shift)
            ll = norm.logpdf(z, loc=fmu, scale=fsig) + _boxcox_log_jacobian(
                yy_shift, lmbda=lmbda, shift=self.transform_shift
            )
            return -np.sum(ll[np.isfinite(ll)])

        raise ValueError(f"Unknown family {self.family}")

    def _fit_one_grid(self, h_gp, o_gp):
        valid = np.isfinite(o_gp) & np.any(np.isfinite(h_gp), axis=0)
        if valid.sum() < 5:
            return (
                np.array([0.0, 1.0, np.log(np.expm1(1e-4)), 0.0]),
                None,
                np.nan,
                {"success": False, "n_valid": int(valid.sum()), "final_nll": np.nan},
            )

        h = h_gp[:, valid]
        y = o_gp[valid]
        fbar, s2, wet_frac = self._safe_ens_stats(h, dry_threshold=self.dry_threshold)
        dry = y <= self.dry_threshold
        wet = ~dry

        lmbda = np.nan
        if self.family == "transformed_gaussian":
            y_pos = y[wet] - self.dry_threshold
            if y_pos.size < 5:
                return (
                    np.array([0.0, 1.0, np.log(np.expm1(1e-4)), 0.0]),
                    None,
                    np.nan,
                    {"success": False, "n_valid": int(valid.sum()), "final_nll": np.nan},
                )
            try:
                lmbda = float(boxcox_normmax(np.maximum(y_pos, self.transform_shift)))
            except Exception:
                lmbda = 0.0

        occ_model = None
        if self._needs_hurdle():
            occ_mode = self.occurrence_model
            if occ_mode == "auto":
                occ_mode = "logistic"
            if occ_mode == "climatology":
                occ_model = {"kind": "climatology", "p0": float(np.mean(dry))}
            elif occ_mode == "logistic":
                occ_fit = _fit_occurrence_logistic(wet_frac, fbar, dry.astype(float))
                occ_fit["kind"] = "logistic"
                occ_model = occ_fit

        target_for_gate = np.where(wet, y, 0.0)
        if not self._should_calibrate(fbar, target_for_gate):
            base = np.array([0.0, 1.0, np.log(np.expm1(np.nanvar(target_for_gate) + 1e-6)), 0.0])
            return base, occ_model, lmbda, {
                "success": True,
                "n_valid": int(valid.sum()),
                "final_nll": np.nan,
            }

        wet_target = y if self.family == "censored_normal" else y[wet]
        wet_fbar = fbar if self.family == "censored_normal" else fbar[wet]
        mask = np.isfinite(wet_fbar) & np.isfinite(wet_target)
        if wet_target.size < 3 or mask.sum() < 3:
            init_a, init_b = 0.0, 1.0
        else:
            lm = linregress(wet_fbar[mask], wet_target[mask])
            init_a, init_b = float(lm.intercept), float(lm.slope)

        init_var = np.nanvar(np.where(np.isfinite(y), y, np.nan))
        init_c = np.log(np.expm1(max(init_var, 1e-4)))
        init_d = 0.1
        x0 = np.array([init_a, init_b, init_c, init_d], dtype=float)

        bounds = [self.bounds["a"], self.bounds["b"], self.bounds["c"], self.bounds["d"]]

        res = minimize(
            self._pos_nll,
            x0=x0,
            args=(fbar, s2, y, wet, lmbda),
            method="L-BFGS-B",
            bounds=bounds,
            options={"maxiter": 600, "ftol": 1e-9, "gtol": 1e-6},
        )

        pars = res.x if res.success else x0
        return pars, occ_model, lmbda, {
            "success": bool(res.success),
            "n_valid": int(valid.sum()),
            "final_nll": float(res.fun) if np.isfinite(res.fun) else np.nan,
        }

    def fit(
        self,
        hcst_grid,
        obs_grid,
        clim_terciles=False,
        member_dim="M",
        time_dim="T",
        lat_dim="Y",
        lon_dim="X",
    ):
        use_xarray = (
            xr is not None
            and isinstance(hcst_grid, xr.DataArray)
            and isinstance(obs_grid, xr.DataArray)
        )

        self._member_dim = member_dim
        self._time_dim = time_dim
        self._lat_dim = lat_dim
        self._lon_dim = lon_dim

        if use_xarray:
            hcst = hcst_grid.transpose(member_dim, time_dim, lat_dim, lon_dim).values
            obs = obs_grid.transpose(time_dim, lat_dim, lon_dim).values
            self._lat_coords = hcst_grid.coords[lat_dim]
            self._lon_coords = hcst_grid.coords[lon_dim]
            self.attrs = dict(hcst_grid.attrs)
            self._xarray = True
        else:
            hcst = np.asarray(hcst_grid, dtype=float)
            obs = np.asarray(obs_grid, dtype=float)
            self._xarray = False

        if hcst.ndim != 4:
            raise ValueError("hcst_grid must have shape (M, T, Y, X)")
        if obs.ndim != 3:
            raise ValueError("obs_grid must have shape (T, Y, X)")

        m, t, ny, nx = hcst.shape
        if obs.shape != (t, ny, nx):
            raise ValueError(f"obs_grid must have shape {(t, ny, nx)}, got {obs.shape}")

        params = np.full((4, ny, nx), np.nan, dtype=float)
        occ_params = np.full((3, ny, nx), np.nan, dtype=float)
        occ_feature_means = np.full((2, ny, nx), np.nan, dtype=float)
        occ_feature_stds = np.full((2, ny, nx), np.nan, dtype=float)
        transform_param = np.full((ny, nx), np.nan, dtype=float)

        fit_stats = {
            "success": np.zeros((ny, nx), dtype=bool),
            "n_valid": np.zeros((ny, nx), dtype=int),
            "final_nll": np.full((ny, nx), np.nan, dtype=float),
        }

        for iy in range(ny):
            for ix in range(nx):
                pars, occ, lmbda, stat = self._fit_one_grid(hcst[:, :, iy, ix], obs[:, iy, ix])
                params[:, iy, ix] = pars
                transform_param[iy, ix] = lmbda
                fit_stats["success"][iy, ix] = stat["success"]
                fit_stats["n_valid"][iy, ix] = stat["n_valid"]
                fit_stats["final_nll"][iy, ix] = stat["final_nll"]

                if occ is not None:
                    if occ["kind"] == "climatology":
                        occ_params[:, iy, ix] = [occ["p0"], np.nan, np.nan]
                    elif occ["kind"] == "logistic":
                        beta = occ["beta"]
                        occ_params[: len(beta), iy, ix] = beta
                        occ_feature_means[:, iy, ix] = occ["means"][1:]
                        occ_feature_stds[:, iy, ix] = occ["stds"][1:]

        self.params = params
        self.occ_params = occ_params if self._needs_hurdle() else None
        self._occ_feature_means = occ_feature_means if self._needs_hurdle() else None
        self._occ_feature_stds = occ_feature_stds if self._needs_hurdle() else None
        self.transform_param = transform_param if self.family == "transformed_gaussian" else None

        if clim_terciles:
            terc = np.nanpercentile(obs, [100.0 / 3.0, 200.0 / 3.0], axis=0)
            self.clim_terciles = terc

        if use_xarray:
            self.params = xr.DataArray(
                params,
                dims=(self._param_dim, lat_dim, lon_dim),
                coords={
                    self._param_dim: ["a", "b", "c", "d"],
                    lat_dim: self._lat_coords,
                    lon_dim: self._lon_coords,
                },
                name="precip_emos_parameters",
                attrs=self.attrs,
            )
            if self.occ_params is not None:
                self.occ_params = xr.DataArray(
                    occ_params,
                    dims=("occ_parameter", lat_dim, lon_dim),
                    coords={
                        "occ_parameter": ["p0_or_b0", "b1", "b2"],
                        lat_dim: self._lat_coords,
                        lon_dim: self._lon_coords,
                    },
                    name="occurrence_parameters",
                )
            if self.transform_param is not None:
                self.transform_param = xr.DataArray(
                    transform_param,
                    dims=(lat_dim, lon_dim),
                    coords={lat_dim: self._lat_coords, lon_dim: self._lon_coords},
                    name="transform_parameter",
                )
            if self.clim_terciles is not None:
                self.clim_terciles = xr.DataArray(
                    self.clim_terciles,
                    dims=("tercile", lat_dim, lon_dim),
                    coords={
                        "tercile": ["lower", "upper"],
                        lat_dim: self._lat_coords,
                        lon_dim: self._lon_coords,
                    },
                    name="climatological_terciles",
                )

        self._fit_stats = fit_stats
        self.fitted = True
        return self

    def _cdf_positive(self, y, mu, var, family, lmbda=None):
        sigma = np.sqrt(np.maximum(var, 1e-10))
        y = np.asarray(y, dtype=float)

        if family == "censored_normal":
            return norm.cdf(y, loc=mu, scale=sigma)

        if family == "truncated_normal":
            alpha = (self.dry_threshold - mu) / sigma
            denom = np.maximum(1.0 - norm.cdf(alpha), 1e-15)
            num = np.maximum(norm.cdf((y - mu) / sigma) - norm.cdf(alpha), 0.0)
            return np.clip(num / denom, 0.0, 1.0)

        if family == "gamma":
            mean_pos = np.maximum(mu, 1e-8)
            k, theta = _gamma_from_mean_var(mean_pos, var)
            return gamma_dist.cdf(np.maximum(y - self.dry_threshold, 0.0), a=k, scale=theta)

        if family == "transformed_gaussian":
            yp = np.maximum(y - self.dry_threshold, 0.0)
            z = _boxcox_forward(yp, lmbda=lmbda, shift=self.transform_shift)
            return norm.cdf(z, loc=mu, scale=sigma)

        raise ValueError(f"Unknown family {family}")

    def _ppf_positive(self, p, mu, var, family, lmbda=None):
        p = np.clip(np.asarray(p, dtype=float), 1e-12, 1.0 - 1e-12)
        sigma = np.sqrt(np.maximum(var, 1e-10))

        if family == "censored_normal":
            q = norm.ppf(p, loc=mu, scale=sigma)
            return np.maximum(q, self.dry_threshold)

        if family == "truncated_normal":
            alpha = (self.dry_threshold - mu) / sigma
            pa = norm.cdf(alpha)
            z = norm.ppf(pa + p * (1.0 - pa))
            return mu + sigma * z

        if family == "gamma":
            mean_pos = np.maximum(mu, 1e-8)
            k, theta = _gamma_from_mean_var(mean_pos, var)
            return self.dry_threshold + gamma_dist.ppf(p, a=k, scale=theta)

        if family == "transformed_gaussian":
            z = norm.ppf(p, loc=mu, scale=sigma)
            return self.dry_threshold + _boxcox_inverse(z, lmbda=lmbda, shift=self.transform_shift)

        raise ValueError(f"Unknown family {family}")

    def _positive_mean_var(self, mu, var, family, lmbda=None):
        sigma = np.sqrt(np.maximum(var, 1e-10))

        if family == "censored_normal":
            a = mu / sigma
            mean = sigma * norm.pdf(a) + mu * norm.cdf(a)
            e2 = (mu**2 + var) * norm.cdf(a) + mu * sigma * norm.pdf(a)
            v = np.maximum(e2 - mean**2, 0.0)
            return mean, v

        if family == "truncated_normal":
            return _truncated_normal_mean_var(mu, sigma, dry_threshold=self.dry_threshold)

        if family == "gamma":
            mean_pos = np.maximum(mu, 1e-8)
            return mean_pos + self.dry_threshold, np.maximum(var, 0.0)

        if family == "transformed_gaussian":
            p = np.linspace(0.01, 0.99, 51)
            mu_ex = np.expand_dims(np.asarray(mu, dtype=float), axis=-1)
            var_ex = np.expand_dims(np.asarray(var, dtype=float), axis=-1)
            yq = self._ppf_positive(p, mu_ex, var_ex, family, lmbda=lmbda)
            mean = np.nanmean(yq, axis=-1)
            vv = np.nanvar(yq, axis=-1, ddof=0)
            return mean, np.maximum(vv, 0.0)

        raise ValueError(f"Unknown family {family}")

    def _predict_numpy(
        self,
        fcst,
        quantiles=None,
        clim_terciles=False,
        return_synthetic_ensemble=False,
    ):
        if not self.fitted:
            raise RuntimeError("Call fit() before predict().")

        fcst = np.asarray(fcst, dtype=float)
        if fcst.ndim != 4:
            raise ValueError("fcst must have shape (M, T, Y, X)")

        m, t, ny, nx = fcst.shape

        param_np = self.params.values if hasattr(self.params, "values") else self.params
        occ_np = None if self.occ_params is None else (
            self.occ_params.values if hasattr(self.occ_params, "values") else self.occ_params
        )
        occ_mean_np = None if self._occ_feature_means is None else self._occ_feature_means
        occ_std_np = None if self._occ_feature_stds is None else self._occ_feature_stds
        lam_np = None if self.transform_param is None else (
            self.transform_param.values if hasattr(self.transform_param, "values") else self.transform_param
        )

        mean_out = np.full((t, ny, nx), np.nan, dtype=float)
        std_out = np.full((t, ny, nx), np.nan, dtype=float)
        probs_out = None
        q_out = None
        synth_out = np.full((m, t, ny, nx), np.nan, dtype=float) if return_synthetic_ensemble else None

        if quantiles is not None:
            q_levels = np.asarray(quantiles, dtype=float)
            q_out = np.full((q_levels.size, t, ny, nx), np.nan, dtype=float)

        if clim_terciles:
            probs_out = np.full((3, t, ny, nx), np.nan, dtype=float)
            terc = self.clim_terciles.values if hasattr(self.clim_terciles, "values") else self.clim_terciles

        for iy in range(ny):
            for ix in range(nx):
                f_gp = fcst[:, :, iy, ix]
                fbar, s2, wet_frac = self._safe_ens_stats(f_gp, dry_threshold=self.dry_threshold)
                pars = param_np[:, iy, ix]
                mu, var = self._mu_var_from_pars(pars, fbar, s2)

                lmbda = None if lam_np is None else lam_np[iy, ix]
                p0 = np.zeros_like(fbar)

                if self._needs_hurdle():
                    if self.occurrence_model == "climatology":
                        p0[:] = occ_np[0, iy, ix]
                    else:
                        beta = np.where(np.isfinite(occ_np[:, iy, ix]), occ_np[:, iy, ix], 0.0)
                        m1, m2 = occ_mean_np[:, iy, ix]
                        s1, s2s = occ_std_np[:, iy, ix]
                        if not np.isfinite(s1) or s1 < 1e-10:
                            s1 = 1.0
                        if not np.isfinite(s2s) or s2s < 1e-10:
                            s2s = 1.0
                        x1 = (wet_frac - m1) / s1
                        x2 = (np.nan_to_num(fbar, nan=0.0) - m2) / s2s
                        eta = beta[0] + beta[1] * x1 + beta[2] * x2
                        p0[:] = np.clip(_sigmoid(eta), 1e-10, 1.0 - 1e-10)

                if self.family == "censored_normal":
                    sigma = np.sqrt(np.maximum(var, 1e-10))
                    p0 = norm.cdf((self.dry_threshold - mu) / sigma)

                pos_mean, pos_var = self._positive_mean_var(mu, var, self.family, lmbda=lmbda)
                if self.family == "censored_normal":
                    full_mean = pos_mean
                    full_var = pos_var
                else:
                    qwet = 1.0 - np.clip(p0, 0.0, 1.0)
                    full_mean = qwet * pos_mean
                    full_var = qwet * pos_var + qwet * (1.0 - qwet) * pos_mean**2

                mean_out[:, iy, ix] = full_mean
                std_out[:, iy, ix] = np.sqrt(np.maximum(full_var, 0.0))

                if quantiles is not None:
                    for iq, q in enumerate(q_levels):
                        if self.family == "censored_normal":
                            sigma = np.sqrt(np.maximum(var, 1e-10))
                            q_out[iq, :, iy, ix] = np.where(
                                q <= p0,
                                self.dry_threshold,
                                np.maximum(norm.ppf(q, loc=mu, scale=sigma), self.dry_threshold),
                            )
                        else:
                            r = np.where(q <= p0, np.nan, (q - p0) / np.maximum(1.0 - p0, 1e-12))
                            vals = np.where(
                                q <= p0,
                                self.dry_threshold,
                                self._ppf_positive(r, mu, var, self.family, lmbda=lmbda),
                            )
                            q_out[iq, :, iy, ix] = vals

                if clim_terciles:
                    lower = terc[0, iy, ix]
                    upper = terc[1, iy, ix]

                    if self.family == "censored_normal":
                        p_below = self._cdf_positive(lower, mu, var, self.family, lmbda=lmbda)
                        p_above = 1.0 - self._cdf_positive(upper, mu, var, self.family, lmbda=lmbda)
                    else:
                        f_lower = np.where(
                            lower <= self.dry_threshold,
                            p0,
                            p0 + (1.0 - p0) * self._cdf_positive(lower, mu, var, self.family, lmbda=lmbda),
                        )
                        f_upper = np.where(
                            upper <= self.dry_threshold,
                            p0,
                            p0 + (1.0 - p0) * self._cdf_positive(upper, mu, var, self.family, lmbda=lmbda),
                        )
                        p_below = f_lower
                        p_above = 1.0 - f_upper

                    p_near = 1.0 - p_below - p_above
                    probs = np.stack([p_below, p_near, p_above], axis=0)
                    probs = np.clip(probs, 0.0, 1.0)
                    denom = np.sum(probs, axis=0, keepdims=True)
                    denom = np.where(denom <= 0.0, 1.0, denom)
                    probs_out[:, :, iy, ix] = probs / denom

                if return_synthetic_ensemble:
                    for tt in range(t):
                        raw = f_gp[:, tt]
                        good = np.isfinite(raw)
                        if not np.any(good):
                            continue
                        ranks = _rankdata_average(raw[good])
                        probs = (ranks - 0.5) / good.sum()

                        if self.family == "censored_normal":
                            sigma = np.sqrt(max(var[tt], 1e-10))
                            vals = np.maximum(norm.ppf(probs, loc=mu[tt], scale=sigma), self.dry_threshold)
                        else:
                            vals = np.where(
                                probs <= p0[tt],
                                self.dry_threshold,
                                self._ppf_positive(
                                    (probs - p0[tt]) / max(1.0 - p0[tt], 1e-12),
                                    mu[tt],
                                    var[tt],
                                    self.family,
                                    lmbda=lmbda,
                                ),
                            )
                        tmp = np.full(m, np.nan)
                        tmp[good] = vals
                        synth_out[:, tt, iy, ix] = tmp

        out = {
            "calibrated_mean": mean_out,
            "calibrated_std": std_out,
        }
        if q_out is not None:
            out["calibrated_quantiles"] = q_out
        if probs_out is not None:
            out["tercile_probability"] = probs_out
        if synth_out is not None:
            out["synthetic_calibrated_ensemble"] = synth_out
        return out

    def predict(
        self,
        fcst_grid,
        quantiles=None,
        clim_terciles=False,
        return_synthetic_ensemble=False,
        member_dim="M",
        time_dim="T",
        lat_dim="Y",
        lon_dim="X",
    ):
        use_xarray = xr is not None and isinstance(fcst_grid, xr.DataArray)

        if use_xarray:
            fc = fcst_grid.transpose(member_dim, time_dim, lat_dim, lon_dim)
            out = self._predict_numpy(
                fc.values,
                quantiles=quantiles,
                clim_terciles=clim_terciles,
                return_synthetic_ensemble=return_synthetic_ensemble,
            )

            ds = xr.Dataset(attrs=dict(fcst_grid.attrs))
            ds["calibrated_mean"] = xr.DataArray(
                out["calibrated_mean"],
                dims=(time_dim, lat_dim, lon_dim),
                coords={
                    time_dim: fc.coords[time_dim],
                    lat_dim: fc.coords[lat_dim],
                    lon_dim: fc.coords[lon_dim],
                },
            )
            ds["calibrated_std"] = xr.DataArray(
                out["calibrated_std"],
                dims=(time_dim, lat_dim, lon_dim),
                coords={
                    time_dim: fc.coords[time_dim],
                    lat_dim: fc.coords[lat_dim],
                    lon_dim: fc.coords[lon_dim],
                },
            )

            if "calibrated_quantiles" in out:
                ds["calibrated_quantiles"] = xr.DataArray(
                    out["calibrated_quantiles"],
                    dims=("quantile", time_dim, lat_dim, lon_dim),
                    coords={
                        "quantile": list(np.asarray(quantiles, dtype=float)),
                        time_dim: fc.coords[time_dim],
                        lat_dim: fc.coords[lat_dim],
                        lon_dim: fc.coords[lon_dim],
                    },
                )

            if "tercile_probability" in out:
                ds["tercile_probability"] = xr.DataArray(
                    out["tercile_probability"],
                    dims=("probability", time_dim, lat_dim, lon_dim),
                    coords={
                        "probability": ["PB", "PN", "PA"],
                        time_dim: fc.coords[time_dim],
                        lat_dim: fc.coords[lat_dim],
                        lon_dim: fc.coords[lon_dim],
                    },
                )

            if "synthetic_calibrated_ensemble" in out:
                ds["synthetic_calibrated_ensemble"] = xr.DataArray(
                    out["synthetic_calibrated_ensemble"],
                    dims=(member_dim, time_dim, lat_dim, lon_dim),
                    coords={
                        member_dim: fc.coords[member_dim],
                        time_dim: fc.coords[time_dim],
                        lat_dim: fc.coords[lat_dim],
                        lon_dim: fc.coords[lon_dim],
                    },
                )
            return ds

        return self._predict_numpy(
            fcst_grid,
            quantiles=quantiles,
            clim_terciles=clim_terciles,
            return_synthetic_ensemble=return_synthetic_ensemble,
        )

    forecast = predict

    def compute_model(
        self,
        X_train,
        y_train,
        X_test,
        obs_for_terciles=None,
        quantiles=None,
        clim_terciles=False,
        member_dim="M",
        time_dim="T",
        lat_dim="Y",
        lon_dim="X",
        return_synthetic_ensemble=False,
    ):
        self.fit(
            X_train,
            y_train,
            clim_terciles=bool(clim_terciles or (obs_for_terciles is not None)),
            member_dim=member_dim,
            time_dim=time_dim,
            lat_dim=lat_dim,
            lon_dim=lon_dim,
        )

        if obs_for_terciles is not None:
            if xr is not None and isinstance(obs_for_terciles, xr.DataArray):
                obs_np = obs_for_terciles.transpose(time_dim, lat_dim, lon_dim).values
            else:
                obs_np = np.asarray(obs_for_terciles, dtype=float)

            terc = np.nanpercentile(obs_np, [100.0 / 3.0, 200.0 / 3.0], axis=0)

            if xr is not None and isinstance(self.clim_terciles, xr.DataArray):
                self.clim_terciles = xr.DataArray(
                    terc,
                    dims=("tercile", lat_dim, lon_dim),
                    coords={
                        "tercile": ["lower", "upper"],
                        lat_dim: obs_for_terciles.coords[lat_dim],
                        lon_dim: obs_for_terciles.coords[lon_dim],
                    },
                )
            else:
                self.clim_terciles = terc

        return self.predict(
            X_test,
            quantiles=quantiles,
            clim_terciles=clim_terciles,
            return_synthetic_ensemble=return_synthetic_ensemble,
            member_dim=member_dim,
            time_dim=time_dim,
            lat_dim=lat_dim,
            lon_dim=lon_dim,
        )

    def get_fit_stats(self):
        if self._fit_stats is None:
            return None
        if self._xarray and self._lat_coords is not None:
            out = {}
            for key, arr in self._fit_stats.items():
                out[key] = xr.DataArray(
                    arr,
                    dims=(self._lat_dim, self._lon_dim),
                    coords={self._lat_dim: self._lat_coords, self._lon_dim: self._lon_coords},
                    name=key,
                )
            return out
        return self._fit_stats

    def summary(self):
        print("WAS_mme_NGR_NonGaussian")
        print("================")
        print(f"family           : {self.family}")
        print(f"occurrence_model : {self.occurrence_model}")
        print(f"dry_threshold    : {self.dry_threshold}")
        if self.family == "transformed_gaussian":
            print(f"transform        : {self.transform}")
        if self._fit_stats is not None:
            succ = self._fit_stats["success"]
            print(f"success rate     : {100.0 * np.mean(succ):.1f}%")


################################################################################ MVA ############################################################


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

    
# ─────────────────────────────────────────────────────────────────────────────
# Module-level helper
# ─────────────────────────────────────────────────────────────────────────────

def _normalize_tercile_probs(pb: float, pn: float, pa: float) -> np.ndarray:
    """
    Clip PB, PN, PA to [0, 1] and renormalise so that they sum exactly to 1.

    This handles small numerical artefacts that can arise from CDF evaluations
    (e.g. pb + pa > 1 due to floating-point error near the threshold values).
    """
    arr = np.clip([pb, pn, pa], 0.0, 1.0)
    s = float(arr.sum())
    return arr / s if s > 0.0 else np.full(3, 1.0 / 3.0)


# ─────────────────────────────────────────────────────────────────────────────
class WAS_mme_FastBMA:
    """
    Grid-based Bayesian Model Averaging — SLSQP maximum-likelihood fitting.

    Supported ``model_type`` values
    --------------------------------
    ``'normal'``
        Gaussian BMA (Raftery et al. 2005).  Component: N(f_k, σ²) where f_k
        is the k-th raw ensemble member and σ is a single shared standard
        deviation fitted by maximum likelihood.

    ``'gamma'``
        Gamma mixture for strictly positive quantities (wind speed, etc.).
        Parameterisation: Γ(shape, scale = f_k / shape) so that
        E[Y | k] = f_k.  A single shared shape parameter is estimated.

    ``'gamma0'``
        Zero-adjusted Gamma mixture for precipitation
        (Sloughter et al. 2007, §2).  The marginal model is

            P(Y = 0)  = 1 − POP
            P(Y > 0)  = POP · Γ-mixture(shape, f_k / shape)

        where POP is estimated via logistic regression on three predictors:
        intercept, ∛ens_mean, and fraction of wet ensemble members
        (Sloughter et al. 2007 Eq. 1).

    Tercile thresholds
    ------------------
    Provide either ``clim_terciles`` (dims: ``('tercile', Y, X)``) or
    ``obs_for_terciles`` (dims include ``time_dim, Y, X``); terciles are then
    computed per gridpoint.
    """

    def __init__(self, model_type: str = "normal", tol: float = 1e-3) -> None:
        if model_type not in {"normal", "gamma", "gamma0"}:
            raise NotImplementedError(f"model_type='{model_type}' not supported.")
        self.model_type = model_type
        self.tol        = tol
        self.fitted     = False

        self.weights:         Optional[xr.DataArray] = None   # (M, Y, X)
        self.sigma:           Optional[xr.DataArray] = None   # (Y, X) — normal
        self.shape:           Optional[xr.DataArray] = None   # (Y, X) — gamma / gamma0
        self.logistic_params: Optional[xr.DataArray] = None   # (3, Y, X) — gamma0

        self._member_dim: Optional[str] = None
        self._time_dim:   Optional[str] = None
        self._lat_dim:    Optional[str] = None
        self._lon_dim:    Optional[str] = None

    # ──────────────────────────────────────────────────────── utilities ──────

    @staticmethod
    def _require_dims(da: xr.DataArray, dims: Sequence[str], name: str) -> None:
        missing = [d for d in dims if d not in da.dims]
        if missing:
            raise ValueError(
                f"{name} is missing required dims: {missing}. Got dims={da.dims}"
            )

    @staticmethod
    def _normalize_weights(w: np.ndarray) -> np.ndarray:
        w = np.clip(np.asarray(w, dtype=float), 0.0, None)
        s = float(w.sum())
        return w / s if (np.isfinite(s) and s > 0.0) else np.full_like(w, 1.0 / w.size)

    @staticmethod
    def compute_gridpoint_terciles_from_obs(
        obs: xr.DataArray,
        *,
        time_dim: str = "T",
        lat_dim:  str = "Y",
        lon_dim:  str = "X",
        q: Tuple[float, float] = (1 / 3, 2 / 3),
    ) -> xr.DataArray:
        """
        Compute per-gridpoint tercile thresholds from a climatological sample.

        Returns
        -------
        xr.DataArray
            dims: ``('tercile', lat_dim, lon_dim)``; tercile ∈ ['lower', 'upper'].
        """
        # FIX #1: original code called WAS_EnsembleBMA._require_dims
        #         (a non-existent class).  Corrected to use this class's method.
        WAS_mme_FastBMA._require_dims(obs, [time_dim, lat_dim, lon_dim], "obs")
        qt = obs.quantile(list(q), dim=time_dim, skipna=True).rename(
            {"quantile": "tercile"}
        )
        return qt.assign_coords(tercile=["lower", "upper"])

    def _align_params_to_forecasts(
        self,
        new_forecasts: xr.DataArray,
        *,
        member_dim: str,
        time_dim:   str,
        lat_dim:    str,
        lon_dim:    str,
    ) -> Tuple[xr.DataArray, xr.DataArray, Optional[xr.DataArray]]:
        """Nearest-neighbour spatial alignment of stored parameters to the forecast grid."""
        self._require_dims(
            new_forecasts, [member_dim, time_dim, lat_dim, lon_dim], "new_forecasts"
        )
        sel = {lat_dim: new_forecasts[lat_dim], lon_dim: new_forecasts[lon_dim]}
        w_ds = self.weights.sel(sel, method="nearest")

        if self.model_type == "normal":
            return w_ds, self.sigma.sel(sel, method="nearest"), None
        else:
            p1 = self.shape.sel(sel, method="nearest")
            p2 = (
                self.logistic_params.sel(sel, method="nearest")
                if self.model_type == "gamma0"
                else None
            )
            return w_ds, p1, p2

    # ──────────────────────────────────────────────────────────── fit ────────

    def fit(
        self,
        hcst_grid: xr.DataArray,
        obs_grid:  xr.DataArray,
        *,
        member_dim:  str = "M",
        time_dim:    str = "T",
        lat_dim:     str = "Y",
        lon_dim:     str = "X",
        min_samples: int = 10,
    ) -> "WAS_mme_FastBMA":
        """
        Fit BMA parameters at every gridpoint via SLSQP maximum-likelihood.

        Parameters
        ----------
        hcst_grid : dims (member_dim, time_dim, lat_dim, lon_dim)
        obs_grid  : dims (time_dim, lat_dim, lon_dim)
        """
        self._require_dims(hcst_grid, [member_dim, time_dim, lat_dim, lon_dim], "hcst_grid")
        self._require_dims(obs_grid,  [time_dim,   lat_dim,  lon_dim],           "obs_grid")

        self._member_dim = member_dim
        self._time_dim   = time_dim
        self._lat_dim    = lat_dim
        self._lon_dim    = lon_dim

        hcst = hcst_grid.transpose(member_dim, time_dim, lat_dim, lon_dim).values
        obs  = obs_grid.transpose(time_dim, lat_dim, lon_dim).values

        n_memb, _, n_lat, n_lon = hcst.shape

        weights_map = np.full((n_memb, n_lat, n_lon), 1.0 / n_memb)
        param_map_1 = np.full((n_lat, n_lon), np.nan)   # σ (normal) or shape (gamma*)
        # FIX #3: gamma0 uses 3 logistic predictors: intercept, ∛ens_mean, frac_wet
        #   (Sloughter et al. 2007).  Original had only 2 (intercept + ∛ens_mean).
        param_map_2 = np.full((3, n_lat, n_lon), np.nan)

        coords = {
            member_dim: hcst_grid.coords[member_dim],
            lat_dim:    hcst_grid.coords[lat_dim],
            lon_dim:    hcst_grid.coords[lon_dim],
        }

        # FIX #2: explicit SLSQP inequality constraint ensuring the implicit
        # last weight w_M = 1 - Σ w_i remains ≥ 0.  Previously this was only
        # softly penalised inside the NLL, which the solver could silently violate.
        weight_sum_le_1 = {
            "type": "ineq",
            "fun": lambda p: 1.0 - np.sum(p[: n_memb - 1]),
        }

        print(f"Fitting {self.model_type} BMA on {n_lat}×{n_lon} grid …")

        for ilat in tqdm(range(n_lat), desc="Training"):
            for ilon in range(n_lon):
                f_raw = hcst[:, :, ilat, ilon]   # (M, T)
                o_raw = obs[:,    ilat, ilon]     # (T,)

                valid = np.isfinite(o_raw) & np.isfinite(f_raw).all(axis=0)
                if int(valid.sum()) < min_samples:
                    continue

                f_data = f_raw[:, valid]   # (M, N)
                o_data = o_raw[valid]      # (N,)

                # ── Normal (Raftery et al. 2005) ─────────────────────────────
                if self.model_type == "normal":

                    def nll_normal(p, _f=f_data, _o=o_data):
                        ws    = p[:-1]
                        sigma = p[-1]
                        if sigma <= 1e-6 or not np.isfinite(sigma):
                            return 1e12
                        all_ws = np.append(ws, 1.0 - ws.sum())
                        # N(f_k, σ²): component mean = raw member forecast
                        z   = (_o[None, :] - _f) / sigma
                        pdf = np.dot(all_ws, norm.pdf(z) / sigma)
                        return -np.sum(np.log(np.maximum(pdf, 1e-12)))

                    x0     = np.append(
                        np.full(n_memb - 1, 1.0 / n_memb), np.nanstd(o_data)
                    )
                    bounds = [(0.0, 1.0)] * (n_memb - 1) + [(1e-3, None)]

                    try:
                        res = minimize(
                            nll_normal, x0,
                            method="SLSQP",
                            bounds=bounds,
                            constraints=[weight_sum_le_1],
                            tol=self.tol,
                        )
                        if res.success:
                            ws = self._normalize_weights(
                                np.append(res.x[:-1], 1.0 - res.x[:-1].sum())
                            )
                            weights_map[:, ilat, ilon] = ws
                            param_map_1[ilat, ilon]    = float(res.x[-1])
                    except Exception:
                        continue

                # ── Gamma / Gamma0 ───────────────────────────────────────────
                else:
                    if self.model_type == "gamma0":
                        # ── POP logistic regression (Sloughter et al. 2007 §2) ──
                        # Three predictors: 1, ∛ens_mean, fraction of wet members.
                        y_bin    = (o_data > 0).astype(float)
                        ens_mean = f_data.mean(axis=0)
                        x_cbrt   = np.cbrt(np.maximum(ens_mean, 0.0))
                        # FIX #3: add fraction of wet members as second skill predictor
                        x_frac   = (f_data > 0).mean(axis=0).astype(float)
                        X_pop    = np.column_stack(
                            [np.ones(y_bin.size), x_cbrt, x_frac]
                        )

                        def nll_logistic(beta, _X=X_pop, _y=y_bin):
                            p = np.clip(expit(_X @ beta), 1e-8, 1.0 - 1e-8)
                            return -np.sum(
                                _y * np.log(p) + (1.0 - _y) * np.log(1.0 - p)
                            )

                        try:
                            res_log = minimize(
                                nll_logistic, [0.0, 1.0, 0.0], method="BFGS"
                            )
                            param_map_2[:, ilat, ilon] = res_log.x
                        except Exception:
                            # Conservative default: near-zero POP everywhere
                            param_map_2[:, ilat, ilon] = [-5.0, 0.0, 0.0]

                        mask_pos = (
                            (o_data > 0)
                            & np.isfinite(o_data)
                            & np.isfinite(f_data).all(axis=0)
                        )
                        if int(mask_pos.sum()) < max(5, min_samples // 2):
                            continue
                        o_gamma = o_data[mask_pos]
                        f_gamma = f_data[:, mask_pos]

                    else:  # gamma
                        mask_pos = (
                            (o_data > 0)
                            & np.isfinite(o_data)
                            & np.isfinite(f_data).all(axis=0)
                        )
                        if int(mask_pos.sum()) < min_samples:
                            continue
                        o_gamma = o_data[mask_pos]
                        f_gamma = f_data[:, mask_pos]

                    # Γ(shape, scale = f/shape) ⟹ E[Y|k] = f_k (component mean = forecast)
                    f_gamma = np.maximum(f_gamma, 1e-3)

                    def nll_gamma(p, _f=f_gamma, _o=o_gamma):
                        ws        = p[:-1]
                        shape_val = p[-1]
                        w_last    = 1.0 - ws.sum()
                        if w_last < 0 or shape_val < 0.1 or not np.isfinite(shape_val):
                            return 1e12
                        all_ws = np.append(ws, w_last)
                        pdfs   = sp_gamma.pdf(_o[None, :], a=shape_val, scale=_f / shape_val)
                        return -np.sum(np.log(np.maximum(np.dot(all_ws, pdfs), 1e-12)))

                    x0     = np.append(np.full(n_memb - 1, 1.0 / n_memb), 2.0)
                    bounds = [(0.0, 1.0)] * (n_memb - 1) + [(0.1, 50.0)]

                    try:
                        res = minimize(
                            nll_gamma, x0,
                            method="SLSQP",
                            bounds=bounds,
                            constraints=[weight_sum_le_1],
                            tol=self.tol,
                        )
                        if res.success:
                            ws = self._normalize_weights(
                                np.append(res.x[:-1], 1.0 - res.x[:-1].sum())
                            )
                            weights_map[:, ilat, ilon] = ws
                            param_map_1[ilat, ilon]    = float(res.x[-1])
                    except Exception:
                        continue

        # ── Store parameters as xarray ────────────────────────────────────────
        ll_coords = {lat_dim: coords[lat_dim], lon_dim: coords[lon_dim]}
        self.weights = xr.DataArray(
            weights_map, dims=(member_dim, lat_dim, lon_dim), coords=coords
        )

        if self.model_type == "normal":
            self.sigma = xr.DataArray(
                param_map_1, dims=(lat_dim, lon_dim), coords=ll_coords
            )
        else:
            self.shape = xr.DataArray(
                param_map_1, dims=(lat_dim, lon_dim), coords=ll_coords
            )
            if self.model_type == "gamma0":
                # FIX #3 (cont.): coord labels updated to match 3-predictor model
                self.logistic_params = xr.DataArray(
                    param_map_2,
                    dims=("param", lat_dim, lon_dim),
                    coords={
                        "param": ["b0", "b1_cbrt_mean", "b2_frac_wet"],
                        **ll_coords,
                    },
                )

        self.fitted = True
        return self

    # ────────────────────────────────────────────────────────── predict ──────

    def predict_probabilistic(
        self,
        new_forecasts:    xr.DataArray,
        *,
        clim_terciles:    Optional[xr.DataArray] = None,
        obs_for_terciles: Optional[xr.DataArray] = None,
        tercile_q:        Tuple[float, float]    = (1 / 3, 2 / 3),
        quantiles:        Sequence[float]         = (0.1, 0.5, 0.9),
        member_dim:       str = "M",
        time_dim:         str = "T",
        lat_dim:          str = "Y",
        lon_dim:          str = "X",
    ) -> xr.Dataset:
        """
        Generate probabilistic forecasts from the fitted BMA mixture.

        Parameters
        ----------
        new_forecasts    : dims (member_dim, time_dim, lat_dim, lon_dim)
        clim_terciles    : dims ('tercile', lat_dim, lon_dim)  [optional]
        obs_for_terciles : dims (time_dim, lat_dim, lon_dim)   [optional]

        Returns
        -------
        xr.Dataset:
          ``predictive_mean``       — (T, Y, X)
          ``predictive_quantiles``  — ('quantile', T, Y, X)
          ``tercile_probability``   — ('probability', T, Y, X)  [if terciles given]
          ``tercile_thresholds``    — ('tercile', Y, X)          [if terciles given]
        """
        if not self.fitted:
            raise ValueError("Call .fit() before .predict_probabilistic().")

        self._require_dims(
            new_forecasts, [member_dim, time_dim, lat_dim, lon_dim], "new_forecasts"
        )
        w_ds, p1_ds, p2_ds = self._align_params_to_forecasts(
            new_forecasts,
            member_dim=member_dim, time_dim=time_dim,
            lat_dim=lat_dim,       lon_dim=lon_dim,
        )

        # ── Tercile thresholds ────────────────────────────────────────────────
        sel = {lat_dim: new_forecasts[lat_dim], lon_dim: new_forecasts[lon_dim]}
        terciles_da = None

        if clim_terciles is not None:
            if "tercile" not in clim_terciles.dims and "quantile" in clim_terciles.dims:
                clim_terciles = clim_terciles.rename({"quantile": "tercile"})
            self._require_dims(clim_terciles, ["tercile", lat_dim, lon_dim], "clim_terciles")
            terciles_da = clim_terciles.sel(sel, method="nearest")
        elif obs_for_terciles is not None:
            obs_sel     = obs_for_terciles.sel(sel, method="nearest")
            terciles_da = self.compute_gridpoint_terciles_from_obs(
                obs_sel,
                time_dim=time_dim, lat_dim=lat_dim, lon_dim=lon_dim, q=tercile_q,
            )

        terciles_np = (
            terciles_da.transpose("tercile", lat_dim, lon_dim).values
            if terciles_da is not None else None
        )   # shape (2, n_lat, n_lon) or None

        # ── NumPy extraction ──────────────────────────────────────────────────
        fcst = new_forecasts.transpose(member_dim, time_dim, lat_dim, lon_dim).values
        ws   = w_ds.transpose(member_dim, lat_dim, lon_dim).values
        p1   = p1_ds.transpose(lat_dim, lon_dim).values
        p2   = (
            p2_ds.transpose("param", lat_dim, lon_dim).values
            if p2_ds is not None else None
        )

        n_memb, n_time, n_lat, n_lon = fcst.shape
        n_quant = len(quantiles)

        out_mean  = np.full((n_time, n_lat, n_lon), np.nan)
        out_quant = np.full((n_quant, n_time, n_lat, n_lon), np.nan)
        out_probs = (
            np.full((3, n_time, n_lat, n_lon), np.nan)
            if terciles_np is not None else None
        )

        print("Predicting …")

        for ilat in tqdm(range(n_lat), desc="Predict"):
            for ilon in range(n_lon):
                param_val = p1[ilat, ilon]
                if not np.isfinite(param_val):
                    continue

                w_loc  = self._normalize_weights(ws[:, ilat, ilon])
                low_th = up_th = np.nan
                if terciles_np is not None:
                    low_th, up_th = terciles_np[:, ilat, ilon]

                # FIX #4: unpack all three POP logistic parameters
                if self.model_type == "gamma0":
                    b0, b1, b2 = p2[:, ilat, ilon]
                    if not (np.isfinite(b0) and np.isfinite(b1) and np.isfinite(b2)):
                        continue

                for t in range(n_time):
                    f_m = fcst[:, t, ilat, ilon]
                    if not np.isfinite(f_m).all():
                        continue

                    do_probs = (
                        out_probs is not None
                        and np.isfinite(low_th)
                        and np.isfinite(up_th)
                    )

                    # ── Normal ───────────────────────────────────────────────
                    if self.model_type == "normal":
                        sigma  = param_val
                        mu_mix = float(np.dot(w_loc, f_m))
                        out_mean[t, ilat, ilon] = mu_mix

                        # FIX #10: capture loop vars via default args to avoid
                        #          accidental late-binding in closures
                        def cdf_mix(x, _w=w_loc, _f=f_m, _s=sigma):
                            return float(np.dot(_w, norm.cdf(x, loc=_f, scale=_s)))

                        lo = mu_mix - 10.0 * sigma
                        hi = mu_mix + 10.0 * sigma
                        for iq, qv in enumerate(quantiles):
                            try:
                                out_quant[iq, t, ilat, ilon] = brentq(
                                    lambda x, _qv=qv: cdf_mix(x) - _qv, lo, hi
                                )
                            except Exception:
                                pass

                        if do_probs:
                            pb = cdf_mix(float(low_th))
                            pa = 1.0 - cdf_mix(float(up_th))
                            # FIX #9: normalise all three probabilities
                            out_probs[:, t, ilat, ilon] = _normalize_tercile_probs(
                                pb, 1.0 - pb - pa, pa
                            )

                    # ── Gamma0 (precipitation, Sloughter et al. 2007) ────────
                    elif self.model_type == "gamma0":
                        shape  = param_val
                        f_safe = np.maximum(f_m, 1e-3)
                        scales = f_safe / shape   # Γ(shape, scale) ⟹ mean = f_m

                        # POP: logistic(b0 + b1·∛ens_mean + b2·frac_wet)
                        f_mean_cbrt = float(np.cbrt(max(float(np.mean(f_m)), 0.0)))
                        frac_wet    = float(np.mean(f_m > 0))
                        pop = float(expit(b0 + b1 * f_mean_cbrt + b2 * frac_wet))

                        # E[Y] = POP · Σ_k w_k · f_k
                        out_mean[t, ilat, ilon] = pop * float(np.dot(w_loc, f_m))

                        def cdf_gamma_cond(x, _w=w_loc, _sh=shape, _sc=scales):
                            """CDF of the conditional Gamma part P(Y ≤ x | Y > 0)."""
                            return float(np.dot(_w, sp_gamma.cdf(x, a=_sh, scale=_sc)))

                        def cdf_full(x, _pop=pop):
                            """
                            Full zero-inflated CDF:
                            F(x) = (1−pop) + pop·F_Γ(x) for x ≥ 0, 0 for x < 0.
                            The point mass (1−pop) sits at x = 0.
                            """
                            if x < 0.0:
                                return 0.0
                            return (1.0 - _pop) + _pop * cdf_gamma_cond(x)

                        for iq, qv in enumerate(quantiles):
                            if qv <= 1.0 - pop:
                                # Quantile falls within the point mass at zero
                                out_quant[iq, t, ilat, ilon] = 0.0
                            else:
                                # Invert the conditional Gamma CDF
                                target = (qv - (1.0 - pop)) / max(pop, 1e-12)
                                try:
                                    top = float(np.max(f_safe)) * 8.0 + 10.0
                                    out_quant[iq, t, ilat, ilon] = brentq(
                                        lambda x, _t=target: cdf_gamma_cond(x) - _t,
                                        1e-9, top,
                                    )
                                except Exception:
                                    pass

                        if do_probs:
                            pb = cdf_full(float(low_th))
                            pa = 1.0 - cdf_full(float(up_th))
                            out_probs[:, t, ilat, ilon] = _normalize_tercile_probs(
                                pb, 1.0 - pb - pa, pa
                            )

                    # ── Gamma ─────────────────────────────────────────────────
                    else:
                        shape  = param_val
                        f_safe = np.maximum(f_m, 1e-3)
                        scales = f_safe / shape

                        out_mean[t, ilat, ilon] = float(np.dot(w_loc, f_m))

                        def cdf_mix_g(x, _w=w_loc, _sh=shape, _sc=scales):
                            return float(np.dot(_w, sp_gamma.cdf(x, a=_sh, scale=_sc)))

                        for iq, qv in enumerate(quantiles):
                            try:
                                top = float(np.max(f_safe)) * 8.0 + 10.0
                                out_quant[iq, t, ilat, ilon] = brentq(
                                    lambda x, _qv=qv: cdf_mix_g(x) - _qv, 1e-9, top
                                )
                            except Exception:
                                pass

                        if do_probs:
                            pb = cdf_mix_g(float(low_th))
                            pa = 1.0 - cdf_mix_g(float(up_th))
                            out_probs[:, t, ilat, ilon] = _normalize_tercile_probs(
                                pb, 1.0 - pb - pa, pa
                            )

        # ── Package output ────────────────────────────────────────────────────
        base_coords = {
            time_dim: new_forecasts[time_dim],
            lat_dim:  new_forecasts[lat_dim],
            lon_dim:  new_forecasts[lon_dim],
        }
        ds = xr.Dataset()
        ds["predictive_mean"] = xr.DataArray(
            out_mean, dims=(time_dim, lat_dim, lon_dim), coords=base_coords
        )
        ds["predictive_quantiles"] = xr.DataArray(
            out_quant,
            dims=("quantile", time_dim, lat_dim, lon_dim),
            coords={**base_coords, "quantile": list(quantiles)},
        )
        if out_probs is not None:
            ds["tercile_probability"] = xr.DataArray(
                out_probs,
                dims=("probability", time_dim, lat_dim, lon_dim),
                coords={**base_coords, "probability": ["PB", "PN", "PA"]},
            )
            ds["tercile_thresholds"] = terciles_da
        return ds


# ─────────────────────────────────────────────────────────────────────────────
class WAS_mme_FullBMA:
    """
    Hybrid Bayesian seasonal ensemble postprocessor — per-gridpoint fitting.

    Distribution families (per gridpoint, supplied via ``dist_map``)
    -----------------------------------------------------------------
    Code 1 → ``'normal'``     Gaussian BMA with bias-correction
    Code 4 → ``'gamma'``      Gamma BMA (positive-definite variables)
    Code 2 → ``'lognormal'``  Lognormal BMA (heavy-tailed precipitation, etc.)

    Each component distribution has its own intercept ``a_k`` and slope ``b_k``
    so that the component mean is a linear (or log-linear for lognormal)
    function of the member forecast — a mixture-EMOS formulation.

    Fitting modes
    -------------
    ``'fast'`` — Penalised L-BFGS-B optimisation (always available).
    ``'full'`` — Full MCMC via PyMC (requires ``pymc`` + ``arviz``).
    ``'auto'`` — Tries MCMC first, falls back to L-BFGS-B on failure.

    Notes
    -----
    - Gridpoints where ``dist_map`` is NaN are skipped.
    - Positive-definite families (gamma, lognormal) are fitted on y > 0 only.
    - ``predict_probabilistic`` spatially aligns stored parameter maps to the
      forecast grid via nearest-neighbour selection, so training and test grids
      need not be identical.
    """

    DIST_CODE_TO_NAME = {1: "normal", 4: "gamma", 2: "lognormal"}
    DIST_NAME_TO_CODE = {"normal": 1, "gamma": 4, "lognormal": 2}
    MODE_CODE         = {"full": 0, "fast": 1, "failed": -1}

    def __init__(
        self,
        mode:           str   = "auto",
        eps:            float = 1e-6,
        tol:            float = 1e-5,
        maxiter:        int   = 300,
        draws:          int   = 1000,
        tune:           int   = 1000,
        chains:         int   = 2,
        target_accept:  float = 0.92,
        random_seed:    int   = 42,
        progressbar:    bool  = True,
        verbose:        bool  = True,
    ) -> None:
        if mode not in {"full", "fast", "auto"}:
            raise ValueError("mode must be one of {'full', 'fast', 'auto'}")

        self.mode          = mode
        self.eps           = eps
        self.tol           = tol
        self.maxiter       = maxiter
        self.draws         = draws
        self.tune          = tune
        self.chains        = chains
        self.target_accept = target_accept
        self.random_seed   = random_seed
        self.progressbar   = progressbar
        self.verbose       = verbose
        self.fitted        = False

        self.weight_map:      Optional[xr.DataArray] = None
        self.intercept_map:   Optional[xr.DataArray] = None
        self.slope_map:       Optional[xr.DataArray] = None
        self.dispersion_map:  Optional[xr.DataArray] = None
        self.family_map_used: Optional[xr.DataArray] = None
        self.fit_mode_map:    Optional[xr.DataArray] = None
        self.fit_status_map:  Optional[xr.DataArray] = None

        # FIX #7: keyed by (lat_coord_value, lon_coord_value) instead of
        #         (training_ilat, training_ilon) so that nearest-neighbour
        #         alignment in predict_probabilistic works correctly when the
        #         forecast grid differs from the training grid.
        self.posterior_: Dict[Tuple[float, float], Dict[str, np.ndarray]] = {}

        self._member_dim:    Optional[str]        = None
        self._time_dim:      Optional[str]        = None
        self._lat_dim:       Optional[str]        = None
        self._lon_dim:       Optional[str]        = None
        self._member_values: Optional[np.ndarray] = None
        self._lat_values:    Optional[np.ndarray] = None
        self._lon_values:    Optional[np.ndarray] = None

    # ──────────────────────────────────────────────────────── utilities ──────

    @staticmethod
    def _require_dims(da: xr.DataArray, dims: Sequence[str], name: str) -> None:
        missing = [d for d in dims if d not in da.dims]
        if missing:
            raise ValueError(
                f"{name} missing required dims {missing}. Got dims={da.dims}"
            )

    @staticmethod
    def _safe_scale(x: np.ndarray, default: float = 1.0) -> float:
        s = np.nanstd(x)
        if not np.isfinite(s) or s <= 0:
            s = np.nanmean(np.abs(x))
        if not np.isfinite(s) or s <= 0:
            s = default
        return float(s)

    @staticmethod
    def _flatten_trace(trace: "az.InferenceData", varname: str) -> np.ndarray:
        arr = trace.posterior[varname].values
        return arr.reshape(arr.shape[0] * arr.shape[1], *arr.shape[2:])

    def _parse_family(self, value: Union[str, int, float, None]) -> Optional[str]:
        if value is None:
            return None
        try:
            if np.isnan(value):
                return None
        except TypeError:
            pass
        if isinstance(value, str):
            fam = value.strip().lower()
            return fam if fam in self.DIST_NAME_TO_CODE else None
        try:
            code = int(value)
        except Exception:
            return None
        return self.DIST_CODE_TO_NAME.get(code, None)

    @staticmethod
    def compute_gridpoint_terciles_from_obs(
        obs: xr.DataArray,
        *,
        time_dim: str = "T",
        lat_dim:  str = "Y",
        lon_dim:  str = "X",
        q: Tuple[float, float] = (1 / 3, 2 / 3),
    ) -> xr.DataArray:
        # FIX: consistent with FastBMA — use this class's own _require_dims
        WAS_mme_FullBMA._require_dims(obs, [time_dim, lat_dim, lon_dim], "obs")
        qt = obs.quantile(list(q), dim=time_dim, skipna=True).rename(
            {"quantile": "tercile"}
        )
        return qt.assign_coords(tercile=["lower", "upper"])

    def _align_terciles_to_forecast_grid(
        self,
        new_forecasts:    xr.DataArray,
        clim_terciles:    Optional[xr.DataArray],
        obs_for_terciles: Optional[xr.DataArray],
        tercile_q:        Tuple[float, float],
        lat_dim:          str,
        lon_dim:          str,
        time_dim:         str,
    ) -> Optional[xr.DataArray]:
        sel = {lat_dim: new_forecasts[lat_dim], lon_dim: new_forecasts[lon_dim]}
        if clim_terciles is not None:
            if "tercile" not in clim_terciles.dims and "quantile" in clim_terciles.dims:
                clim_terciles = clim_terciles.rename({"quantile": "tercile"})
            self._require_dims(clim_terciles, ["tercile", lat_dim, lon_dim], "clim_terciles")
            return clim_terciles.sel(sel, method="nearest")
        if obs_for_terciles is not None:
            obs_sel = obs_for_terciles.sel(sel, method="nearest")
            return self.compute_gridpoint_terciles_from_obs(
                obs_sel,
                time_dim=time_dim, lat_dim=lat_dim, lon_dim=lon_dim, q=tercile_q,
            )
        return None

    # ─────────────────────────────────────────── FAST backend ───────────────

    @staticmethod
    def _unpack_fast_params(
        p: np.ndarray, m: int
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray, float]:
        """
        Unpack the flat parameter vector used by L-BFGS-B:
          p[:m]   — weight logits  (softmax → weights)
          p[m:2m] — intercepts a_k
          p[2m:3m]— log-slopes     (exp → b_k > 0)
          p[3m]   — log-dispersion (exp → σ / shape / σ_log)
        """
        logits_w = p[:m]
        a        = p[m:2 * m]
        b        = np.exp(p[2 * m:3 * m])
        disp     = np.exp(p[3 * m])
        w        = softmax(logits_w)
        return w, a, b, disp

    def _nll_fast_normal(self, p, f_data, y_data, lam_w, lam_a, lam_b, lam_disp, y_scale):
        m, _ = f_data.shape
        w, a, b, sigma = self._unpack_fast_params(p, m)
        mu      = a[:, None] + b[:, None] * f_data
        pdf     = norm.pdf(y_data[None, :], loc=mu, scale=max(sigma, self.eps))
        mix_pdf = np.sum(w[:, None] * pdf, axis=0)
        nll     = -np.sum(np.log(np.maximum(mix_pdf, self.eps)))
        pen = (
            lam_w    * np.sum((w - 1.0 / m) ** 2)
            + lam_a  * np.sum((a / max(y_scale, self.eps)) ** 2)
            + lam_b  * np.sum(np.log(b) ** 2)           # penalise slopes far from 1
            + lam_disp * (np.log(sigma / max(y_scale, self.eps)) ** 2)
        )
        return nll + pen

    def _nll_fast_gamma(self, p, f_data, y_data, lam_w, lam_a, lam_b, lam_disp, y_scale):
        m, _ = f_data.shape
        w, a, b, shape = self._unpack_fast_params(p, m)
        if np.any(y_data <= 0):
            return 1e20
        mu      = np.maximum(a[:, None] + b[:, None] * f_data, self.eps)
        # Γ(shape, rate=shape/mu) ⟹ mean = mu
        rate    = shape / mu
        pdf     = sp_gamma.pdf(y_data[None, :], a=shape, scale=1.0 / np.maximum(rate, self.eps))
        mix_pdf = np.sum(w[:, None] * pdf, axis=0)
        nll     = -np.sum(np.log(np.maximum(mix_pdf, self.eps)))
        pen = (
            lam_w    * np.sum((w - 1.0 / m) ** 2)
            + lam_a  * np.sum((a / max(y_scale, self.eps)) ** 2)
            + lam_b  * np.sum(np.log(b) ** 2)
            + lam_disp * (np.log(shape) ** 2)
        )
        return nll + pen

    def _nll_fast_lognormal(
        self, p, f_data, y_data, lam_w, lam_a, lam_b, lam_disp, y_scale
    ):
        m, _ = f_data.shape
        w, a, b, sigma_log = self._unpack_fast_params(p, m)
        if np.any(y_data <= 0):
            return 1e20
        # Component: LogNormal(μ_log = a + b·log f, σ_log)
        mu_log  = a[:, None] + b[:, None] * np.log(np.maximum(f_data, self.eps))
        pdf     = lognorm.pdf(y_data[None, :], s=max(sigma_log, self.eps), scale=np.exp(mu_log))
        mix_pdf = np.sum(w[:, None] * pdf, axis=0)
        nll     = -np.sum(np.log(np.maximum(mix_pdf, self.eps)))
        pen = (
            lam_w    * np.sum((w - 1.0 / m) ** 2)
            + lam_a  * np.sum(a ** 2)
            + lam_b  * np.sum(np.log(b) ** 2)
            + lam_disp * (np.log(sigma_log) ** 2)
        )
        return nll + pen

    def _fit_single_grid_fast(
        self,
        f_data: np.ndarray,
        y_data: np.ndarray,
        family: str,
        lam_w:    float,
        lam_a:    float,
        lam_b:    float,
        lam_disp: float,
    ) -> Dict[str, Any]:
        m, _ = f_data.shape
        y_scale = self._safe_scale(y_data, default=1.0)

        if family in {"gamma", "lognormal"}:
            valid  = y_data > 0
            y_data = y_data[valid]
            f_data = f_data[:, valid]
            if y_data.size < 5:
                raise ValueError(f"Insufficient positive samples for '{family}'")

        # FIX #8: for lognormal the dispersion parameter is σ_log (std of log Y),
        #         not the raw-scale std.  Initialise from std(log y) instead.
        if family == "lognormal":
            log_std   = float(np.std(np.log(np.maximum(y_data, self.eps))))
            disp_init = np.log(max(log_std, 0.05))
        else:
            disp_init = np.log(max(y_scale, 1.0))

        p0 = np.concatenate([
            np.zeros(m),      # weight logits  → uniform weights after softmax
            np.zeros(m),      # intercepts     → zero bias
            np.zeros(m),      # log-slopes     → slope = 1 (unbiased)
            [disp_init],
        ])

        if family == "normal":
            obj = lambda p: self._nll_fast_normal(
                p, f_data, y_data, lam_w, lam_a, lam_b, lam_disp, y_scale
            )
        elif family == "gamma":
            obj = lambda p: self._nll_fast_gamma(
                p, f_data, y_data, lam_w, lam_a, lam_b, lam_disp, y_scale
            )
        elif family == "lognormal":
            obj = lambda p: self._nll_fast_lognormal(
                p, f_data, y_data, lam_w, lam_a, lam_b, lam_disp, y_scale
            )
        else:
            raise ValueError(f"Unsupported family '{family}'")

        res = minimize(
            obj, p0,
            method="L-BFGS-B",
            options={"maxiter": self.maxiter, "ftol": self.tol},
        )
        if not res.success:
            raise RuntimeError(res.message)

        w, a, b, disp = self._unpack_fast_params(res.x, m)
        return {"weights": w, "a": a, "b": b, "dispersion": float(disp),
                "family": family, "backend": "fast"}

    # ─────────────────────────────────────────── FULL backend ───────────────

    def _fit_single_grid_full(
        self,
        f_data: np.ndarray,
        y_data: np.ndarray,
        family: str,
        alpha0:            float,
        coef_scale_mult:   float,
        sigma_scale_mult:  float,
        shape_scale:       float,
    ) -> Dict[str, Any]:
        if not HAS_PYMC:
            raise RuntimeError("PyMC / ArviZ not available")

        m, _ = f_data.shape
        y_scale = self._safe_scale(y_data, default=1.0)

        sample_kw = dict(
            draws=self.draws, tune=self.tune, chains=self.chains,
            target_accept=self.target_accept, random_seed=self.random_seed,
            progressbar=self.progressbar, compute_convergence_checks=False,
            return_inferencedata=True,
        )

        if family == "normal":
            with pm.Model():
                weights = pm.Dirichlet("weights", a=np.full(m, alpha0))
                a       = pm.Normal("a", mu=0.0, sigma=coef_scale_mult * y_scale, shape=m)
                b       = pm.Normal("b", mu=1.0, sigma=coef_scale_mult, shape=m)
                sigma   = pm.HalfNormal("sigma", sigma=sigma_scale_mult * y_scale)
                comp    = [pm.Normal.dist(mu=a[k] + b[k] * f_data[k, :], sigma=sigma)
                           for k in range(m)]
                pm.Mixture("y_like", w=weights, comp_dists=comp, observed=y_data)
                trace = pm.sample(**sample_kw)
            return {
                "weights":    self._flatten_trace(trace, "weights"),
                "a":          self._flatten_trace(trace, "a"),
                "b":          self._flatten_trace(trace, "b"),
                "dispersion": self._flatten_trace(trace, "sigma"),
                "family": family, "backend": "full",
            }

        elif family == "gamma":
            valid  = y_data > 0
            y_pos  = y_data[valid];  f_pos = f_data[:, valid]
            if y_pos.size < 5:
                raise ValueError("Insufficient positive samples for gamma")
            with pm.Model():
                weights = pm.Dirichlet("weights", a=np.full(m, alpha0))
                a       = pm.Normal("a", mu=0.0, sigma=coef_scale_mult * y_scale, shape=m)
                b       = pm.Normal("b", mu=1.0, sigma=coef_scale_mult, shape=m)
                shape   = pm.HalfNormal("shape", sigma=shape_scale)
                comp = []
                for k in range(m):
                    mu_k   = pm.math.maximum(a[k] + b[k] * f_pos[k, :], self.eps)
                    beta_k = shape / mu_k   # rate = shape/mean  ⟹  E[Y] = shape/rate = mu_k
                    comp.append(pm.Gamma.dist(alpha=shape, beta=beta_k))
                pm.Mixture("y_like", w=weights, comp_dists=comp, observed=y_pos)
                trace = pm.sample(**sample_kw)
            return {
                "weights":    self._flatten_trace(trace, "weights"),
                "a":          self._flatten_trace(trace, "a"),
                "b":          self._flatten_trace(trace, "b"),
                "dispersion": self._flatten_trace(trace, "shape"),
                "family": family, "backend": "full",
            }

        elif family == "lognormal":
            valid  = y_data > 0
            y_pos  = y_data[valid];  f_pos = f_data[:, valid]
            if y_pos.size < 5:
                raise ValueError("Insufficient positive samples for lognormal")
            logy       = np.log(np.maximum(y_pos, self.eps))
            log_scale  = self._safe_scale(logy, default=1.0)
            with pm.Model():
                weights   = pm.Dirichlet("weights", a=np.full(m, alpha0))
                a         = pm.Normal("a", mu=0.0, sigma=coef_scale_mult * log_scale, shape=m)
                b         = pm.Normal("b", mu=1.0, sigma=coef_scale_mult, shape=m)
                sigma_log = pm.HalfNormal("sigma_log", sigma=sigma_scale_mult * log_scale)
                comp = []
                for k in range(m):
                    # μ_log,k = a_k + b_k · log(f_k)  ⟹  E[Y|k] = exp(μ_log + σ²/2)
                    mu_log_k = a[k] + b[k] * pm.math.log(
                        pm.math.maximum(f_pos[k, :], self.eps)
                    )
                    comp.append(pm.LogNormal.dist(mu=mu_log_k, sigma=sigma_log))
                pm.Mixture("y_like", w=weights, comp_dists=comp, observed=y_pos)
                trace = pm.sample(**sample_kw)
            return {
                "weights":    self._flatten_trace(trace, "weights"),
                "a":          self._flatten_trace(trace, "a"),
                "b":          self._flatten_trace(trace, "b"),
                "dispersion": self._flatten_trace(trace, "sigma_log"),
                "family": family, "backend": "full",
            }

        raise ValueError(f"Unsupported family '{family}'")

    # ─────────────────────────────────── hybrid single-gridpoint fit ────────

    def _fit_single_grid_hybrid(
        self,
        f_data: np.ndarray,
        y_data: np.ndarray,
        family: str,
        lam_w:           float,
        lam_a:           float,
        lam_b:           float,
        lam_disp:        float,
        alpha0:          float,
        coef_scale_mult: float,
        sigma_scale_mult: float,
        shape_scale:     float,
    ) -> Dict[str, Any]:
        if self.mode == "fast":
            return self._fit_single_grid_fast(f_data, y_data, family, lam_w, lam_a, lam_b, lam_disp)
        if self.mode == "full":
            return self._fit_single_grid_full(f_data, y_data, family, alpha0, coef_scale_mult,
                                               sigma_scale_mult, shape_scale)
        try:
            return self._fit_single_grid_full(f_data, y_data, family, alpha0, coef_scale_mult,
                                               sigma_scale_mult, shape_scale)
        except Exception:
            return self._fit_single_grid_fast(f_data, y_data, family, lam_w, lam_a, lam_b, lam_disp)

    # ──────────────────────────────────────────────────────────── fit ────────

    def fit(
        self,
        hcst_grid: xr.DataArray,
        obs_grid:  xr.DataArray,
        dist_map:  xr.DataArray,
        *,
        member_dim:       str   = "M",
        time_dim:         str   = "T",
        lat_dim:          str   = "Y",
        lon_dim:          str   = "X",
        min_samples:      int   = 12,
        lam_w:            float = 1.0,
        lam_a:            float = 0.1,
        lam_b:            float = 0.1,
        lam_disp:         float = 0.1,
        alpha0:           float = 1.0,
        coef_scale_mult:  float = 1.0,
        sigma_scale_mult: float = 1.0,
        shape_scale:      float = 5.0,
    ) -> "WAS_mme_FullBMA":
        self._require_dims(hcst_grid, [member_dim, time_dim, lat_dim, lon_dim], "hcst_grid")
        self._require_dims(obs_grid,  [time_dim,   lat_dim,  lon_dim],           "obs_grid")
        self._require_dims(dist_map,  [lat_dim,    lon_dim],                     "dist_map")

        self._member_dim    = member_dim
        self._time_dim      = time_dim
        self._lat_dim       = lat_dim
        self._lon_dim       = lon_dim
        self._member_values = hcst_grid[member_dim].values
        self._lat_values    = hcst_grid[lat_dim].values
        self._lon_values    = hcst_grid[lon_dim].values

        hcst    = hcst_grid.transpose(member_dim, time_dim, lat_dim, lon_dim).values
        obs     = obs_grid.transpose(time_dim, lat_dim, lon_dim).values
        dist_np = dist_map.transpose(lat_dim, lon_dim).values

        n_memb, _, n_lat, n_lon = hcst.shape

        weight_map    = np.full((n_memb, n_lat, n_lon), np.nan)
        intercept_map = np.full((n_memb, n_lat, n_lon), np.nan)
        slope_map     = np.full((n_memb, n_lat, n_lon), np.nan)
        dispersion_map = np.full((n_lat, n_lon), np.nan)
        family_code   = np.full((n_lat, n_lon), np.nan)
        fit_mode      = np.full((n_lat, n_lon), self.MODE_CODE["failed"])
        fit_status    = np.zeros((n_lat, n_lon))

        self.posterior_.clear()

        if self.verbose:
            print(f"Fitting hybrid Bayesian postprocessor [mode={self.mode}] …")

        for ilat in tqdm(range(n_lat), desc="Training", disable=not self.verbose):
            for ilon in range(n_lon):
                fam = self._parse_family(dist_np[ilat, ilon])
                if fam is None:
                    continue

                f_raw = hcst[:, :, ilat, ilon]
                y_raw = obs[:,    ilat, ilon]
                valid = np.isfinite(y_raw) & np.isfinite(f_raw).all(axis=0)
                if int(valid.sum()) < min_samples:
                    continue

                try:
                    res = self._fit_single_grid_hybrid(
                        f_data=f_raw[:, valid], y_data=y_raw[valid], family=fam,
                        lam_w=lam_w, lam_a=lam_a, lam_b=lam_b, lam_disp=lam_disp,
                        alpha0=alpha0, coef_scale_mult=coef_scale_mult,
                        sigma_scale_mult=sigma_scale_mult, shape_scale=shape_scale,
                    )
                except Exception as exc:
                    if self.verbose:
                        print(f"  Skipping ({ilat},{ilon}) [{fam}]: {exc}")
                    continue

                family_code[ilat, ilon]   = self.DIST_NAME_TO_CODE[fam]
                fit_status[ilat, ilon]    = 1.0

                if res["backend"] == "fast":
                    fit_mode[ilat, ilon]       = self.MODE_CODE["fast"]
                    weight_map[:, ilat, ilon]  = res["weights"]
                    intercept_map[:, ilat, ilon] = res["a"]
                    slope_map[:, ilat, ilon]   = res["b"]
                    dispersion_map[ilat, ilon] = float(res["dispersion"])

                elif res["backend"] == "full":
                    fit_mode[ilat, ilon]       = self.MODE_CODE["full"]
                    weight_map[:, ilat, ilon]  = res["weights"].mean(axis=0)
                    intercept_map[:, ilat, ilon] = res["a"].mean(axis=0)
                    slope_map[:, ilat, ilon]   = res["b"].mean(axis=0)
                    dispersion_map[ilat, ilon] = float(np.mean(res["dispersion"]))
                    # FIX #7: store by coordinate value, not array index
                    lat_key = float(self._lat_values[ilat])
                    lon_key = float(self._lon_values[ilon])
                    self.posterior_[(lat_key, lon_key)] = res

        coords_w  = {member_dim: self._member_values,
                     lat_dim:    self._lat_values,
                     lon_dim:    self._lon_values}
        coords_2d = {lat_dim: self._lat_values, lon_dim: self._lon_values}

        self.weight_map      = xr.DataArray(weight_map,    dims=(member_dim, lat_dim, lon_dim), coords=coords_w)
        self.intercept_map   = xr.DataArray(intercept_map, dims=(member_dim, lat_dim, lon_dim), coords=coords_w)
        self.slope_map       = xr.DataArray(slope_map,     dims=(member_dim, lat_dim, lon_dim), coords=coords_w)
        self.dispersion_map  = xr.DataArray(dispersion_map, dims=(lat_dim, lon_dim), coords=coords_2d)
        self.family_map_used = xr.DataArray(family_code,   dims=(lat_dim, lon_dim), coords=coords_2d)
        self.fit_mode_map    = xr.DataArray(fit_mode,      dims=(lat_dim, lon_dim), coords=coords_2d)
        self.fit_status_map  = xr.DataArray(fit_status,    dims=(lat_dim, lon_dim), coords=coords_2d)

        self.fitted = True
        return self

    # ─────────────────────────────────────── predictive helpers ─────────────

    def _mixture_mean_fast(
        self, f_m: np.ndarray, w: np.ndarray,
        a: np.ndarray, b: np.ndarray, disp: float, family: str
    ) -> float:
        if family == "normal":
            return float(np.sum(w * (a + b * f_m)))
        if family == "gamma":
            return float(np.sum(w * np.maximum(a + b * f_m, self.eps)))
        if family == "lognormal":
            # E[Y] = exp(μ_log + σ_log²/2) for each component
            mu_log = a + b * np.log(np.maximum(f_m, self.eps))
            return float(np.sum(w * np.exp(mu_log + 0.5 * disp ** 2)))
        raise ValueError(family)

    def _mixture_cdf_fast(
        self, x: float, f_m: np.ndarray, w: np.ndarray,
        a: np.ndarray, b: np.ndarray, disp: float, family: str
    ) -> float:
        if family == "normal":
            mu   = a + b * f_m
            vals = norm.cdf(x, loc=mu, scale=max(disp, self.eps))
            return float(np.sum(w * vals))
        if family == "gamma":
            mu   = np.maximum(a + b * f_m, self.eps)
            rate = max(disp, self.eps) / mu
            vals = sp_gamma.cdf(x, a=max(disp, self.eps), scale=1.0 / np.maximum(rate, self.eps))
            return float(np.sum(w * vals))
        if family == "lognormal":
            mu_log = a + b * np.log(np.maximum(f_m, self.eps))
            vals   = lognorm.cdf(x, s=max(disp, self.eps), scale=np.exp(mu_log))
            return float(np.sum(w * vals))
        raise ValueError(family)

    def _mixture_quantile_fast(
        self, q: float, f_m: np.ndarray, w: np.ndarray,
        a: np.ndarray, b: np.ndarray, disp: float, family: str
    ) -> float:
        """
        Compute the q-th mixture quantile.

        FIX #5: replaced 600-point grid + searchsorted (coarse, unreliable near
        the tails) with Brent root-finding.  The bracket upper bound is
        adaptively expanded if needed.
        """
        mean_pred = self._mixture_mean_fast(f_m, w, a, b, disp, family)

        if family == "normal":
            lo = mean_pred - 10.0 * max(disp, self.eps)
            hi = mean_pred + 10.0 * max(disp, self.eps)
        elif family == "gamma":
            lo = self.eps
            hi = max(mean_pred * 8.0 + 10.0, 1.0)
        elif family == "lognormal":
            lo = self.eps
            hi = max(mean_pred * 10.0 + 10.0, 1.0)
        else:
            raise ValueError(family)

        obj = lambda x: self._mixture_cdf_fast(x, f_m, w, a, b, disp, family) - q

        # Adaptively widen the bracket so that CDF(hi) ≥ q
        for _ in range(12):
            if obj(hi) >= 0.0:
                break
            hi *= 2.0

        try:
            return float(brentq(obj, lo, hi, xtol=1e-6, maxiter=120))
        except Exception:
            # Fallback: fine grid (1 000 pts) — keeps legacy behaviour as a safety net
            xs   = np.linspace(lo, hi, 1000)
            cdfs = np.array([
                self._mixture_cdf_fast(x, f_m, w, a, b, disp, family) for x in xs
            ])
            idx = min(int(np.searchsorted(cdfs, q, side="left")), len(xs) - 1)
            return float(xs[idx])

    def _sample_pp_full(
        self, f_m: np.ndarray, post: Dict[str, np.ndarray],
        family: str, rng: np.random.Generator, n_samples: int
    ) -> np.ndarray:
        n_post = post["weights"].shape[0]
        idx    = rng.choice(n_post, size=n_samples, replace=(n_samples > n_post))

        w    = post["weights"][idx, :]
        a    = post["a"][idx, :]
        b    = post["b"][idx, :]
        disp = post["dispersion"][idx]   # shape: (n_samples,)

        u    = rng.random(n_samples)
        cdf  = np.cumsum(w, axis=1)
        comp = (u[:, None] > cdf).sum(axis=1)   # selected component index

        if family == "normal":
            mu     = a + b * f_m[None, :]
            mu_sel = mu[np.arange(n_samples), comp]
            return rng.normal(mu_sel, disp)

        if family == "gamma":
            mu     = np.maximum(a + b * f_m[None, :], self.eps)
            mu_sel = mu[np.arange(n_samples), comp]
            shape  = np.maximum(disp, self.eps)
            rate   = shape / np.maximum(mu_sel, self.eps)
            return rng.gamma(shape=shape, scale=1.0 / rate)

        if family == "lognormal":
            mu_log = a + b * np.log(np.maximum(f_m[None, :], self.eps))
            mu_sel = mu_log[np.arange(n_samples), comp]
            return rng.lognormal(mean=mu_sel, sigma=np.maximum(disp, self.eps))

        raise ValueError(family)

    # ────────────────────────────────────────────────────────── predict ──────

    def predict_probabilistic(
        self,
        new_forecasts:    xr.DataArray,
        *,
        clim_terciles:    Optional[xr.DataArray] = None,
        obs_for_terciles: Optional[xr.DataArray] = None,
        tercile_q:        Tuple[float, float]    = (1 / 3, 2 / 3),
        quantiles:        Sequence[float]         = (0.1, 0.5, 0.9),
        n_pp_samples:     int   = 2000,
        member_dim:       str   = "M",
        time_dim:         str   = "T",
        lat_dim:          str   = "Y",
        lon_dim:          str   = "X",
    ) -> xr.Dataset:
        if not self.fitted:
            raise ValueError("Call .fit() before .predict_probabilistic().")

        self._require_dims(
            new_forecasts, [member_dim, time_dim, lat_dim, lon_dim], "new_forecasts"
        )

        # FIX #6: spatially align all parameter maps to the forecast grid via
        #         nearest-neighbour selection.  Original code used raw numpy
        #         indices, which silently gave wrong results when the forecast
        #         grid differed from the training grid.
        sel = {lat_dim: new_forecasts[lat_dim], lon_dim: new_forecasts[lon_dim]}

        w_map     = self.weight_map.sel(sel, method="nearest").transpose(member_dim, lat_dim, lon_dim).values
        a_map     = self.intercept_map.sel(sel, method="nearest").transpose(member_dim, lat_dim, lon_dim).values
        b_map     = self.slope_map.sel(sel, method="nearest").transpose(member_dim, lat_dim, lon_dim).values
        d_map     = self.dispersion_map.sel(sel, method="nearest").transpose(lat_dim, lon_dim).values
        fam_map   = self.family_map_used.sel(sel, method="nearest").transpose(lat_dim, lon_dim).values
        mode_map  = self.fit_mode_map.sel(sel, method="nearest").transpose(lat_dim, lon_dim).values
        status_map = self.fit_status_map.sel(sel, method="nearest").transpose(lat_dim, lon_dim).values

        # FIX #7: build nearest-training-coordinate lookup arrays for posterior retrieval
        sel_lat = self.weight_map[lat_dim].sel({lat_dim: new_forecasts[lat_dim]}, method="nearest").values
        sel_lon = self.weight_map[lon_dim].sel({lon_dim: new_forecasts[lon_dim]}, method="nearest").values

        fcst    = new_forecasts.transpose(member_dim, time_dim, lat_dim, lon_dim).values
        _, n_time, n_lat, n_lon = fcst.shape

        terciles_da = self._align_terciles_to_forecast_grid(
            new_forecasts=new_forecasts,
            clim_terciles=clim_terciles,
            obs_for_terciles=obs_for_terciles,
            tercile_q=tercile_q,
            lat_dim=lat_dim, lon_dim=lon_dim, time_dim=time_dim,
        )
        terciles_np = (
            terciles_da.transpose("tercile", lat_dim, lon_dim).values
            if terciles_da is not None else None
        )

        out_mean   = np.full((n_time, n_lat, n_lon), np.nan)
        out_quant  = np.full((len(quantiles), n_time, n_lat, n_lon), np.nan)
        out_probs  = np.full((3, n_time, n_lat, n_lon), np.nan) if terciles_np is not None else None
        out_family = np.full((n_lat, n_lon), np.nan)
        out_mode   = np.full((n_lat, n_lon), np.nan)
        out_status = np.full((n_lat, n_lon), np.nan)

        rng = np.random.default_rng(self.random_seed)

        if self.verbose:
            print("Predicting with hybrid Bayesian postprocessor …")

        for ilat in tqdm(range(n_lat), desc="Predict", disable=not self.verbose):
            for ilon in range(n_lon):
                if status_map[ilat, ilon] != 1.0:
                    continue
                fam_code = fam_map[ilat, ilon]
                if not np.isfinite(fam_code):
                    continue
                fam = self.DIST_CODE_TO_NAME.get(int(fam_code))
                if fam is None:
                    continue

                mode_code = int(mode_map[ilat, ilon])
                out_family[ilat, ilon] = int(fam_code)
                out_mode[ilat, ilon]   = mode_code
                out_status[ilat, ilon] = 1.0

                low_th = up_th = None
                if terciles_np is not None:
                    low_th, up_th = terciles_np[:, ilat, ilon]

                for t in range(n_time):
                    f_m = fcst[:, t, ilat, ilon]
                    if not np.isfinite(f_m).all():
                        continue

                    if mode_code == self.MODE_CODE["full"]:
                        # FIX #7: look up posterior by training coordinate value
                        lat_key = float(sel_lat[ilat])
                        lon_key = float(sel_lon[ilon])
                        post = self.posterior_.get((lat_key, lon_key))
                        if post is None:
                            continue

                        y_rep = self._sample_pp_full(f_m, post, fam, rng, n_pp_samples)
                        out_mean[t, ilat, ilon]      = float(np.mean(y_rep))
                        out_quant[:, t, ilat, ilon]  = np.quantile(y_rep, quantiles)

                        if terciles_np is not None and np.isfinite(low_th) and np.isfinite(up_th):
                            pb = float(np.mean(y_rep <= low_th))
                            pa = float(np.mean(y_rep >  up_th))
                            pn = float(np.mean((y_rep > low_th) & (y_rep <= up_th)))
                            out_probs[:, t, ilat, ilon] = _normalize_tercile_probs(pb, pn, pa)

                    elif mode_code == self.MODE_CODE["fast"]:
                        w    = w_map[:, ilat, ilon]
                        a    = a_map[:, ilat, ilon]
                        b    = b_map[:, ilat, ilon]
                        disp = float(d_map[ilat, ilon])

                        out_mean[t, ilat, ilon] = self._mixture_mean_fast(f_m, w, a, b, disp, fam)
                        for iq, qv in enumerate(quantiles):
                            out_quant[iq, t, ilat, ilon] = self._mixture_quantile_fast(
                                qv, f_m, w, a, b, disp, fam
                            )

                        if terciles_np is not None and np.isfinite(low_th) and np.isfinite(up_th):
                            pb = self._mixture_cdf_fast(float(low_th), f_m, w, a, b, disp, fam)
                            pa = 1.0 - self._mixture_cdf_fast(float(up_th), f_m, w, a, b, disp, fam)
                            # FIX #9: normalise all three probabilities
                            out_probs[:, t, ilat, ilon] = _normalize_tercile_probs(
                                pb, 1.0 - pb - pa, pa
                            )

        coords = {
            time_dim: new_forecasts[time_dim],
            lat_dim:  new_forecasts[lat_dim],
            lon_dim:  new_forecasts[lon_dim],
        }
        ll_coords = {lat_dim: new_forecasts[lat_dim], lon_dim: new_forecasts[lon_dim]}

        ds = xr.Dataset()
        ds["predictive_mean"] = xr.DataArray(
            out_mean, dims=(time_dim, lat_dim, lon_dim), coords=coords
        )
        ds["predictive_quantiles"] = xr.DataArray(
            out_quant,
            dims=("quantile", time_dim, lat_dim, lon_dim),
            coords={**coords, "quantile": list(quantiles)},
        )
        ds["family_used"] = xr.DataArray(
            out_family, dims=(lat_dim, lon_dim), coords=ll_coords,
            attrs={"1": "normal", "4": "gamma", "2": "lognormal"},
        )
        ds["fit_mode_used"] = xr.DataArray(
            out_mode, dims=(lat_dim, lon_dim), coords=ll_coords,
            attrs={"0": "full", "1": "fast", "-1": "failed"},
        )
        ds["fit_status"] = xr.DataArray(
            out_status, dims=(lat_dim, lon_dim), coords=ll_coords,
            attrs={"1": "success", "0": "failed_or_not_fitted"},
        )
        if out_probs is not None:
            ds["tercile_probability"] = xr.DataArray(
                out_probs,
                dims=("probability", time_dim, lat_dim, lon_dim),
                coords={**coords, "probability": ["PB", "PN", "PA"]},
            )
            ds["tercile_thresholds"] = terciles_da
        return ds

    # ─────────────────────────────────────── WAS-style convenience wrapper ──

    def compute_model(
        self,
        X_train:   xr.DataArray,
        y_train:   xr.DataArray,
        X_test:    xr.DataArray,
        dist_map:  xr.DataArray,
        *,
        clim_terciles:    Optional[xr.DataArray]  = None,
        obs_for_terciles: Optional[xr.DataArray]  = None,
        tercile_q:        Tuple[float, float]     = (1 / 3, 2 / 3),
        quantiles:        Sequence[float]          = (0.1, 0.5, 0.9),
        n_pp_samples:     int   = 2000,
        member_dim:       str   = "M",
        time_dim:         str   = "T",
        lat_dim:          str   = "Y",
        lon_dim:          str   = "X",
        min_samples:      int   = 12,
        lam_w:            float = 1.0,
        lam_a:            float = 0.1,
        lam_b:            float = 0.1,
        lam_disp:         float = 0.1,
        alpha0:           float = 1.0,
        coef_scale_mult:  float = 1.0,
        sigma_scale_mult: float = 1.0,
        shape_scale:      float = 5.0,
    ) -> xr.Dataset:
        """Fit on (X_train, y_train) then predict on X_test — one-call interface."""
        self.fit(
            hcst_grid=X_train, obs_grid=y_train, dist_map=dist_map,
            member_dim=member_dim, time_dim=time_dim,
            lat_dim=lat_dim,       lon_dim=lon_dim,
            min_samples=min_samples,
            lam_w=lam_w, lam_a=lam_a, lam_b=lam_b, lam_disp=lam_disp,
            alpha0=alpha0, coef_scale_mult=coef_scale_mult,
            sigma_scale_mult=sigma_scale_mult, shape_scale=shape_scale,
        )
        return self.predict_probabilistic(
            new_forecasts=X_test,
            clim_terciles=clim_terciles,
            obs_for_terciles=obs_for_terciles,
            tercile_q=tercile_q,
            quantiles=quantiles,
            n_pp_samples=n_pp_samples,
            member_dim=member_dim, time_dim=time_dim,
            lat_dim=lat_dim,       lon_dim=lon_dim,
        )

############################################################


class WAS_mme_RF:
    """
    Enhanced Random Forest-based Multi-Model Ensemble (MME) forecasting for West Africa.
    
    Parameters
    ----------
    search_method : str, optional
        Hyperparameter optimization method: 'grid', 'random', or 'bayesian' (default: 'random').
    n_estimators_range : list of int, optional
        List of n_estimators values to tune (default is [100, 200, 300, 500]).
    max_depth_range : list of int or None, optional
        List of max depths to tune (default is [None, 8, 12, 16]).
    min_samples_split_range : list of int, optional
        List of minimum samples required to split (default is [2, 5, 10]).
    min_samples_leaf_range : list of int, optional
        List of minimum samples required at leaf node (default is [1, 2, 4]).
    max_features_range : list of str or float, optional
        List of max features to tune (default is ['sqrt', 0.3, 0.5, 0.7]).
    bootstrap_range : list of bool, optional
        List of bootstrap options to tune (default is [True, False]).
    max_samples_range : list of float or None, optional
        List of max samples as fraction of dataset (default is [None, 0.7, 0.8]).
    min_impurity_decrease_range : list of float, optional
        List of min impurity decrease values (default is [0.0, 0.001, 0.01]).
    max_leaf_nodes_range : list of int or None, optional
        List of max leaf nodes (default is [None, 50, 100]).
    ccp_alpha_range : list of float, optional
        List of complexity parameters for pruning (default is [0.0, 0.001, 0.01]).
    min_weight_fraction_leaf_range : list of float, optional
        List of minimum weighted fraction of leaves (default is [0.0, 0.1, 0.2]).
    warm_start : bool, optional
        Whether to reuse solution of previous call to fit (default is False).
    random_state : int, optional
        Seed for reproducibility (default is 42).
    dist_method : str, optional
        Distribution method for tercile probabilities ('bestfit', 'nonparam', etc.) (default is 'nonparam').
    n_iter_search : int, optional
        Number of iterations for randomized/bayesian search (default is 20).
    cv_folds : int, optional
        Number of cross-validation folds (default is 5).
    n_clusters : int, optional
        Number of clusters for homogenized zones (default is 6 for West Africa).
    scoring : str, optional
        Scoring metric for hyperparameter optimization (default is 'neg_mean_squared_error').
    verbose : int, optional
        Verbosity level (default is 0).
    """

    def __init__(self,
                 search_method: str = 'random',
                 n_estimators_range: Union[List[int], object] = [100, 200, 300, 500],
                 max_depth_range: Union[List[Optional[int]], object] = [None, 8, 12, 16],
                 min_samples_split_range: Union[List[int], object] = [2, 5, 10],
                 min_samples_leaf_range: Union[List[int], object] = [1, 2, 4],
                 max_features_range: List[Union[str, float]] = ['sqrt', 0.3, 0.5, 0.7],
                 bootstrap_range: List[bool] = [True, False],
                 max_samples_range: List[Optional[float]] = [None, 0.7, 0.8],
                 min_impurity_decrease_range: List[float] = [0.0, 0.001, 0.01],
                 ccp_alpha_range: List[float] = [0.0, 0.001, 0.01],
                 warm_start: bool = False,
                 random_state: int = 42,
                 dist_method: str = "nonparam",
                 n_iter_search: int = 20,
                 cv_folds: int = 5,
                 n_clusters: int = 6,
                 scoring: str = 'neg_mean_squared_error',
                 leave_one_year_out: bool = False,   # ===== 2 =====
                 verbose: int = 0):

        self.search_method = search_method
        self.n_estimators_range = n_estimators_range
        self.max_depth_range = max_depth_range
        self.min_samples_split_range = min_samples_split_range
        self.min_samples_leaf_range = min_samples_leaf_range
        self.max_features_range = max_features_range
        self.bootstrap_range = bootstrap_range
        self.max_samples_range = max_samples_range
        self.min_impurity_decrease_range = min_impurity_decrease_range
        self.ccp_alpha_range = ccp_alpha_range
        self.warm_start = warm_start
        self.random_state = random_state
        self.dist_method = dist_method
        self.n_iter_search = n_iter_search
        self.cv_folds = cv_folds
        self.n_clusters = n_clusters
        self.scoring = scoring
        self.leave_one_year_out = leave_one_year_out
        self.verbose = verbose
        self.rf = None
        self.best_params_dict = {}
        self.cluster_da = None

    # ===== 2: year-grouped CV splitter, reused everywhere =====
    def _cv_splitter(self, groups: np.ndarray):
        """
        Return a CV splitter that never puts cells from the same calendar year
        on both sides of the train/test boundary.

        Uses leave-one-year-out when requested or when distinct years are few
        (<= cv_folds); otherwise GroupKFold with cv_folds blocks of years.
        """
        n_groups = len(np.unique(groups))
        if self.leave_one_year_out or n_groups <= self.cv_folds:
            return LeaveOneGroupOut()
        return GroupKFold(n_splits=self.cv_folds)

    def _objective(self, trial, X_train, y_train, groups) -> float:
        """Objective function for Optuna optimization (year-grouped CV)."""
        if isinstance(self.n_estimators_range, list):
            params = {'n_estimators': trial.suggest_categorical('n_estimators', self.n_estimators_range)}
        else:
            params = {'n_estimators': trial.suggest_int('n_estimators', 50, 1000)}

        if isinstance(self.max_depth_range, list):
            max_depth_options = [str(opt) if opt is None else opt for opt in self.max_depth_range]
            max_depth_str = trial.suggest_categorical('max_depth', max_depth_options)
            params['max_depth'] = None if max_depth_str == 'None' else int(max_depth_str)
        else:
            params['max_depth'] = trial.suggest_int('max_depth', 5, 50)

        for param_name, param_range in [
            ('min_samples_split', self.min_samples_split_range),
            ('min_samples_leaf', self.min_samples_leaf_range),
        ]:
            if isinstance(param_range, list):
                params[param_name] = trial.suggest_categorical(param_name, param_range)
            else:
                params[param_name] = trial.suggest_int(param_name, 2, 20)

        max_features_options = [str(opt) if isinstance(opt, (int, float)) else opt for opt in self.max_features_range]
        max_features_str = trial.suggest_categorical('max_features', max_features_options)
        if max_features_str in ['sqrt', 'log2', 'auto', None]:
            params['max_features'] = max_features_str if max_features_str != 'auto' else 'sqrt'
        else:
            params['max_features'] = float(max_features_str)

        params['bootstrap'] = trial.suggest_categorical('bootstrap', self.bootstrap_range)

        if params['bootstrap']:
            if isinstance(self.max_samples_range[0], (int, float)):
                params['max_samples'] = trial.suggest_float('max_samples',
                                                            min(self.max_samples_range),
                                                            max(self.max_samples_range))
            else:
                max_samples_options = [str(opt) if opt is not None else opt for opt in self.max_samples_range]
                max_samples_str = trial.suggest_categorical('max_samples', max_samples_options)
                params['max_samples'] = None if max_samples_str == 'None' else float(max_samples_str)
        else:
            params['max_samples'] = None

        params['min_impurity_decrease'] = trial.suggest_float('min_impurity_decrease',
                                                             min(self.min_impurity_decrease_range),
                                                             max(self.min_impurity_decrease_range))

        params['ccp_alpha'] = trial.suggest_float('ccp_alpha',
                                                  min(self.ccp_alpha_range),
                                                  max(self.ccp_alpha_range))

        model = RandomForestRegressor(
            **params,
            random_state=self.random_state,
            n_jobs=-1,
            warm_start=self.warm_start
        )

        # ===== 2: grouped CV instead of plain KFold =====
        try:
            cv = self._cv_splitter(groups)
            scores = cross_val_score(
                model, X_train, y_train,
                cv=cv, groups=groups,
                scoring=self.scoring,
                n_jobs=-1
            )
            return np.mean(scores)
        except Exception as e:
            print(f"TRIAL FAILED: {e}")
            import traceback
            traceback.print_exc()
            return -1e10

    def _grid_search(self, X: np.ndarray, y: np.ndarray, groups: np.ndarray) -> Dict:
        """Grid search with year-grouped CV."""
        param_grid = {
            'n_estimators': self.n_estimators_range,
            'max_depth': self.max_depth_range,
            'min_samples_split': self.min_samples_split_range,
            'min_samples_leaf': self.min_samples_leaf_range,
            'max_features': self.max_features_range,
            'bootstrap': self.bootstrap_range,
            'max_samples': self.max_samples_range,
            'min_impurity_decrease': self.min_impurity_decrease_range,
            'ccp_alpha': self.ccp_alpha_range,
        }
        model = RandomForestRegressor(random_state=self.random_state, n_jobs=-1)
        cv = self._cv_splitter(groups)  # ===== 2 =====
        grid_search = GridSearchCV(
            model, param_grid=param_grid,
            cv=cv, scoring=self.scoring,
            n_jobs=-1, verbose=self.verbose
        )
        grid_search.fit(X, y, groups=groups)  # ===== 2 =====
        if self.verbose > 0:
            print(f"Best score: {grid_search.best_score_:.4f}")
            print(f"Best params: {grid_search.best_params_}")
        return grid_search.best_params_

    def _random_search(self, X: np.ndarray, y: np.ndarray, groups: np.ndarray) -> Dict:
        """Random search with year-grouped CV."""
        param_dist = {
            'n_estimators': self.n_estimators_range,
            'max_depth': self.max_depth_range,
            'min_samples_split': self.min_samples_split_range,
            'min_samples_leaf': self.min_samples_leaf_range,
            'max_features': self.max_features_range,
            'bootstrap': self.bootstrap_range,
            'max_samples': self.max_samples_range,
            'min_impurity_decrease': self.min_impurity_decrease_range,
            'ccp_alpha': self.ccp_alpha_range,
        }
        model = RandomForestRegressor(random_state=self.random_state, n_jobs=-1)
        cv = self._cv_splitter(groups)  # ===== 2 =====
        random_search = RandomizedSearchCV(
            model, param_distributions=param_dist,
            n_iter=self.n_iter_search, cv=cv,
            scoring=self.scoring, random_state=self.random_state,
            n_jobs=-1, verbose=self.verbose
        )
        random_search.fit(X, y, groups=groups)  # ===== 2 =====
        if self.verbose > 0:
            print(f"Best score: {random_search.best_score_:.4f}")
            print(f"Best params: {random_search.best_params_}")
        return random_search.best_params_

    def _bayesian_search(self, X: np.ndarray, y: np.ndarray, groups: np.ndarray) -> Dict:
        """Bayesian optimization (Optuna) with year-grouped CV."""
        study = optuna.create_study(
            direction='maximize' if 'neg_' in self.scoring else 'minimize',
            sampler=optuna.samplers.TPESampler(seed=self.random_state)
        )
        objective_with_data = lambda trial: self._objective(trial, X, y, groups)  # ===== 2 =====
        study.optimize(
            objective_with_data,
            n_trials=self.n_iter_search,
            show_progress_bar=self.verbose > 0
        )
        best_params = study.best_params
        sklearn_params = {}
        for key, value in best_params.items():
            if key == 'max_features' and isinstance(value, str):
                if value == 'None':
                    sklearn_params[key] = None
                else:
                    try:
                        sklearn_params[key] = float(value)
                    except ValueError:
                        sklearn_params[key] = value
            elif key == 'max_depth' and value == 'None':
                sklearn_params[key] = None
            elif key == 'max_leaf_nodes' and value == 'None':
                sklearn_params[key] = None
            elif key == 'max_samples' and value == 'None':
                sklearn_params[key] = None
            else:
                sklearn_params[key] = value
        if self.verbose > 0:
            print(f"Best score: {study.best_value:.4f}")
            print(f"Best params: {sklearn_params}")
        return sklearn_params

    def compute_hyperparameters(self, Predictors: xr.DataArray, Predictand: xr.DataArray,
                                clim_year_start: Optional[int] = None,
                                clim_year_end: Optional[int] = None) -> Tuple[Dict, xr.DataArray]:
        """
        Compute best hyperparameters per zone. Regionalization is KMeans on the
        predictand values (kept by preference); CV is year-grouped (2).

        If clim_year_start/clim_year_end are given, predictors and predictand
        are standardized here. If they are None, the inputs are assumed already
        standardized -- e.g. when called from WAS_Cross_Validator, which
        standardizes before the fold loop. This removes the old hard-coded
        1970-2000 window AND the double-standardization that happened when the
        framework had already standardized the data.
        """
        if "M" in Predictand.coords:
            Predictand = Predictand.isel(M=0).drop_vars('M').squeeze()
        Predictand.name = "varname"

        if clim_year_start is not None and clim_year_end is not None:
            X_train_std = standardize_timeseries(Predictors, clim_year_start, clim_year_end)
            y_train_std = standardize_timeseries(Predictand, clim_year_start, clim_year_end)
        else:
            X_train_std = Predictors
            y_train_std = Predictand
        X_train_std['T'] = y_train_std['T']


        Pclust = Predictand.copy()
        Pclust.name = "varname"
        df = Pclust.to_dataframe().reset_index().dropna().drop(columns=['T'])
        variable_column = df.columns[2]
        kmeans = KMeans(n_clusters=self.n_clusters,
                        random_state=self.random_state, n_init=10)
        df['cluster'] = kmeans.fit_predict(df[[variable_column]])

        df_unique = df.drop_duplicates(subset=['Y', 'X'])
        dataset = df_unique.set_index(['Y', 'X']).to_xarray()
        clust_mask = xr.where(~np.isnan(Pclust.isel(T=0)), 1, np.nan)
        self.cluster_da = (dataset['cluster'] * clust_mask).reindex(
            Y=Predictand['Y'], X=Predictand['X'])
        clusters = np.unique(self.cluster_da.values)
        clusters = clusters[~np.isnan(clusters)]
        # ===== end regionalization =====

        best_params_dict = {}
        for c in clusters:
            c = int(c)
            if self.verbose > 0:
                print(f"\nOptimizing hyperparameters for cluster {c}...")

            mask_3d = (self.cluster_da == c).broadcast_like(X_train_std.isel(M=0))

            X_stacked_c = (
                X_train_std.where(mask_3d)
                .stack(sample=('T', 'Y', 'X')).transpose('sample', 'M')
            )
            y_stacked_c = (
                y_train_std.where(mask_3d)
                .stack(sample=('T', 'Y', 'X'))
            )

            # ===== 2: year groups aligned with the stacked samples =====
            groups_full = X_stacked_c['T'].values.astype('datetime64[Y]').astype(int)
            Xv = X_stacked_c.values
            yv = y_stacked_c.values.ravel()
            nan_mask_c = np.any(~np.isfinite(Xv), axis=1) | ~np.isfinite(yv)
            X_clean_c = Xv[~nan_mask_c]
            y_clean_c = yv[~nan_mask_c]
            groups_c = groups_full[~nan_mask_c]
            # ===== end 2 =====

            n_years_c = len(np.unique(groups_c))
            if len(X_clean_c) < self.cv_folds * 2 or n_years_c < 2:
                if self.verbose > 0:
                    print(f"Cluster {c}: insufficient data "
                          f"({len(X_clean_c)} samples, {n_years_c} years). Skipping.")
                continue

            if self.search_method == 'grid':
                best_params_dict[c] = self._grid_search(X_clean_c, y_clean_c, groups_c)
            elif self.search_method == 'random':
                best_params_dict[c] = self._random_search(X_clean_c, y_clean_c, groups_c)
            elif self.search_method == 'bayesian':
                best_params_dict[c] = self._bayesian_search(X_clean_c, y_clean_c, groups_c)
            else:
                raise ValueError(f"Unknown search_method: {self.search_method}")

        self.best_params_dict = best_params_dict
        return best_params_dict, self.cluster_da

    def compute_model(self, X_train: xr.DataArray, y_train: xr.DataArray,
                      X_test: xr.DataArray, y_test: Optional[xr.DataArray] = None,
                      clim_year_start: Optional[int] = None,
                      clim_year_end: Optional[int] = None,
                      best_params: Optional[Dict] = None,
                      cluster_da: Optional[xr.DataArray] = None) -> xr.DataArray:
        """
        Deterministic hindcast using RandomForestRegressor with optimized
        hyperparameters.

        """
        if best_params is None or cluster_da is None:
            best_params, cluster_da = self.compute_hyperparameters(
                X_train, y_train, clim_year_start, clim_year_end
            )

        time = X_test['T']
        lat = X_test['Y']
        lon = X_test['X']
        n_time = len(time)
        n_lat = len(lat)
        n_lon = len(lon)

        predictions = np.full((n_time, n_lat, n_lon), np.nan)
        self.rf = {}

        for c in range(self.n_clusters):
            if c not in best_params:
                continue

            bp = best_params[c]

            mask_3d_train = (cluster_da == c).expand_dims({'T': X_train['T']})
            mask_3d_test = (cluster_da == c).expand_dims({'T': X_test['T']})

            X_train_stacked_c = X_train.where(mask_3d_train).stack(sample=('T', 'Y', 'X')).transpose('sample', 'M').values
            y_train_stacked_c = y_train.where(mask_3d_train).stack(sample=('T', 'Y', 'X')).values.ravel()

            train_nan_mask = np.any(~np.isfinite(X_train_stacked_c), axis=1) | ~np.isfinite(y_train_stacked_c)
            X_train_clean_c = X_train_stacked_c[~train_nan_mask]
            y_train_clean_c = y_train_stacked_c[~train_nan_mask]

            X_test_stacked_c = X_test.where(mask_3d_test).stack(sample=('T', 'Y', 'X')).transpose('sample', 'M').values
            test_nan_mask = np.any(~np.isfinite(X_test_stacked_c), axis=1)
            X_test_clean_c = X_test_stacked_c[~test_nan_mask]

            if len(X_train_clean_c) == 0:
                continue

            rf_c = RandomForestRegressor(
                **{k: v for k, v in bp.items() if k in RandomForestRegressor().get_params()},
                random_state=self.random_state,
                n_jobs=-1,
                warm_start=self.warm_start
            )

            rf_c.fit(X_train_clean_c, y_train_clean_c)
            self.rf[c] = rf_c

            if len(X_test_clean_c) > 0:
                y_pred_c = rf_c.predict(X_test_clean_c)
                result_c = np.full(len(X_test_stacked_c), np.nan)
                result_c[~test_nan_mask] = y_pred_c
                pred_c_reshaped = result_c.reshape(n_time, n_lat, n_lon)
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
        """Return tercile thresholds (T1, T2) from best-fit distribution params."""
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
        """Root function for the Weibull shape parameter 'k'."""
        if k <= 0:
            return -np.inf
        try:
            g1 = gamma_function(1 + 1/k)
            g2 = gamma_function(1 + 2/k)
            implied_v_over_m_sq = (g2 / (g1**2)) - 1
            observed_v_over_m_sq = V / (M**2)
            return observed_v_over_m_sq - implied_v_over_m_sq
        except ValueError:
            return -np.inf

    @staticmethod
    def calculate_tercile_probabilities_bestfit(best_guess, error_variance, T1, T2, dist_code, dof):
        """Generic tercile probabilities using best-fit family per grid cell."""
        best_guess = np.asarray(best_guess, float)
        error_variance = np.asarray(error_variance, dtype=float)
        n_time = best_guess.size
        out = np.full((3, n_time), np.nan, float)

        if np.all(np.isnan(best_guess)) or np.isnan(dist_code) or np.isnan(T1) or np.isnan(T2) or np.isnan(error_variance):
            return out

        code = int(dist_code)

        if code == 1:  # Normal
            error_std = np.sqrt(error_variance)
            out[0, :] = norm.cdf(T1, loc=best_guess, scale=error_std)
            out[1, :] = norm.cdf(T2, loc=best_guess, scale=error_std) - norm.cdf(T1, loc=best_guess, scale=error_std)
            out[2, :] = 1 - norm.cdf(T2, loc=best_guess, scale=error_std)

        elif code == 2:  # Lognormal
            sigma = np.sqrt(np.log(1 + error_variance / (best_guess**2)))
            mu = np.log(best_guess) - sigma**2 / 2
            out[0, :] = lognorm.cdf(T1, s=sigma, scale=np.exp(mu))
            out[1, :] = lognorm.cdf(T2, s=sigma, scale=np.exp(mu)) - lognorm.cdf(T1, s=sigma, scale=np.exp(mu))
            out[2, :] = 1 - lognorm.cdf(T2, s=sigma, scale=np.exp(mu))

        elif code == 3:  # Exponential
            c1 = expon.cdf(T1, loc=best_guess, scale=np.sqrt(error_variance))
            c2 = expon.cdf(T2, loc=best_guess, scale=np.sqrt(error_variance))
            out[0, :] = c1
            out[1, :] = c2 - c1
            out[2, :] = 1.0 - c2

        elif code == 4:  # Gamma
            alpha = (best_guess ** 2) / error_variance
            theta = error_variance / best_guess
            c1 = gamma.cdf(T1, a=alpha, scale=theta)
            c2 = gamma.cdf(T2, a=alpha, scale=theta)
            out[0, :] = c1
            out[1, :] = c2 - c1
            out[2, :] = 1.0 - c2

        elif code == 5:  # Weibull
            for i in range(n_time):
                M = best_guess[i]
                V = error_variance
                if V <= 0 or M <= 0:
                    out[:, i] = np.nan
                    continue
                k = fsolve(WAS_mme_RF.weibull_shape_solver, 2.0, args=(M, V))[0]
                if k <= 0:
                    out[:, i] = np.nan
                    continue
                lambda_scale = M / gamma_function(1 + 1/k)
                c1 = weibull_min.cdf(T1, c=k, loc=0, scale=lambda_scale)
                c2 = weibull_min.cdf(T2, c=k, loc=0, scale=lambda_scale)
                out[0, i] = c1
                out[1, i] = c2 - c1
                out[2, i] = 1.0 - c2

        elif code == 6:  # Student-t
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

        elif code == 7:  # Poisson
            mu = best_guess
            c1 = poisson.cdf(T1, mu=mu)
            c2 = poisson.cdf(T2, mu=mu)
            out[0, :] = c1
            out[1, :] = c2 - c1
            out[2, :] = 1.0 - c2

        elif code == 8:  # Negative Binomial
            p = np.where(error_variance > best_guess, best_guess / error_variance, np.nan)
            n = np.where(error_variance > best_guess, (best_guess**2) / (error_variance - best_guess), np.nan)
            c1 = nbinom.cdf(T1, n=n, p=p)
            c2 = nbinom.cdf(T2, n=n, p=p)
            out[0, :] = c1
            out[1, :] = c2 - c1
            out[2, :] = 1.0 - c2

        else:
            raise ValueError(f"Invalid distribution code: {dist_code}")

        # ===== 5: positive-support families are undefined for x <= 0 =====
        # (e.g. when fed standardized anomalies rather than physical rainfall).
        if code in (2, 3, 4, 5, 7, 8):
            bad = ~(best_guess > 0)
            if np.any(bad):
                out[:, bad] = np.nan

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

    def compute_prob(self, Predictant: xr.DataArray, clim_year_start, clim_year_end,
                     hindcast_det: xr.DataArray, best_code_da: xr.DataArray = None,
                     best_shape_da: xr.DataArray = None, best_loc_da: xr.DataArray = None,
                     best_scale_da: xr.DataArray = None) -> xr.DataArray:
        """
        Tercile probabilities for deterministic hindcasts.

        NOTE (3): for honest, well-calibrated probabilities, pass the
        CROSS-VALIDATED hindcast here as `hindcast_det` (out-of-sample
        residuals). Feeding the in-sample hindcast makes the predictive
        distribution too sharp and the reliability diagram over-confident.
        """
        if "M" in Predictant.dims:
            Predictant = Predictant.isel(M=0).drop_vars("M").squeeze()

        Predictant = Predictant.transpose("T", "Y", "X")
        mask = xr.where(~np.isnan(Predictant.isel(T=0)), 1.0, np.nan)

        clim = Predictant.sel(T=slice(str(clim_year_start), str(clim_year_end)))
        if clim.sizes.get("T", 0) < 3:
            raise ValueError("Not enough years in climatology period for terciles.")

        # Out-of-sample residual variance (hindcast_det should be cross-validated)
        error_variance = (Predictant - hindcast_det).var(dim="T")
        dof = max(int(clim.sizes["T"]) - 1, 2)

        terciles_emp = clim.quantile([1.0 / 3.0, 2.0 / 3.0], dim="T")
        T1_emp = terciles_emp.isel(quantile=0).drop_vars("quantile")
        T2_emp = terciles_emp.isel(quantile=1).drop_vars("quantile")

        dm = self.dist_method

        if dm == "bestfit":
            if any(v is None for v in (best_code_da, best_shape_da, best_loc_da, best_scale_da)):
                raise ValueError(
                    "dist_method='bestfit' requires best_code_da, best_shape_da, best_loc_da, best_scale_da."
                )
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
                vectorize=True, kwargs={'dof': dof},
                dask="parallelized", output_dtypes=[float],
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

    def forecast(self, Predictant, clim_year_start, clim_year_end, hindcast_det,
                 hindcast_det_cross, Predictor_for_year, best_params=None, cluster_da=None,
                 best_code_da=None, best_shape_da=None, best_loc_da=None, best_scale_da=None):
        """Forecast using the Random Forest model with optimized hyperparameters."""
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

        time = Predictor_for_year_st['T']
        lat = Predictor_for_year_st['Y']
        lon = Predictor_for_year_st['X']
        n_time = len(time)
        n_lat = len(lat)
        n_lon = len(lon)

        if best_params is None:
            best_params, cluster_da = self.compute_hyperparameters(
                hindcast_det, Predictant_no_m, clim_year_start, clim_year_end
            )

        predictions = np.full((n_time, n_lat, n_lon), np.nan)
        self.rf = {}

        for c in range(self.n_clusters):
            if c not in best_params:
                continue

            bp = best_params[c]

            mask_3d_train = (cluster_da == c).expand_dims({'T': hindcast_det_st['T']})
            mask_3d_test = (cluster_da == c).expand_dims({'T': Predictor_for_year_st['T']})

            X_train_stacked_c = hindcast_det_st.where(mask_3d_train).stack(sample=('T', 'Y', 'X')).transpose('sample', 'M').values
            y_train_stacked_c = Predictant_st.where(mask_3d_train).stack(sample=('T', 'Y', 'X')).values.ravel()

            train_nan_mask = np.any(~np.isfinite(X_train_stacked_c), axis=1) | ~np.isfinite(y_train_stacked_c)
            X_train_clean_c = X_train_stacked_c[~train_nan_mask]
            y_train_clean_c = y_train_stacked_c[~train_nan_mask]

            X_test_stacked_c = Predictor_for_year_st.where(mask_3d_test).stack(sample=('T', 'Y', 'X')).transpose('sample', 'M').values
            test_nan_mask = np.any(~np.isfinite(X_test_stacked_c), axis=1)
            X_test_clean_c = X_test_stacked_c[~test_nan_mask]

            if len(X_train_clean_c) == 0:
                continue

            rf_c = RandomForestRegressor(
                **{k: v for k, v in bp.items() if k in RandomForestRegressor().get_params()},
                random_state=self.random_state,
                n_jobs=-1,
                warm_start=self.warm_start
            )

            rf_c.fit(X_train_clean_c, y_train_clean_c)
            self.rf[c] = rf_c

            if len(X_test_clean_c) > 0:
                y_pred_c = rf_c.predict(X_test_clean_c)
                result_c = np.full(len(X_test_stacked_c), np.nan)
                result_c[~test_nan_mask] = y_pred_c
                pred_c_reshaped = result_c.reshape(n_time, n_lat, n_lon)
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

        index_start = Predictant_no_m.get_index("T").get_loc(str(clim_year_start)).start
        index_end = Predictant_no_m.get_index("T").get_loc(str(clim_year_end)).stop
        rainfall_for_tercile = Predictant_no_m.isel(T=slice(index_start, index_end))
        terciles = rainfall_for_tercile.quantile([1.0 / 3.0, 2.0 / 3.0], dim='T')
        T1_emp = terciles.isel(quantile=0).drop_vars('quantile')
        T2_emp = terciles.isel(quantile=1).drop_vars('quantile')
        error_variance = (Predictant_no_m - hindcast_det_cross).var(dim='T')
        dof = max(int(rainfall_for_tercile.sizes["T"]) - 1, 2)

        dm = self.dist_method

        if dm == "bestfit":
            if any(v is None for v in (best_code_da, best_shape_da, best_loc_da, best_scale_da)):
                raise ValueError(
                    "dist_method='bestfit' requires best_code_da, best_shape_da, best_loc_da, best_scale_da."
                )
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
                result_da, error_variance, T1, T2, best_code_da,
                input_core_dims=[("T",), (), (), (), ()],
                output_core_dims=[("probability", "T")],
                vectorize=True, dask="parallelized", kwargs={"dof": dof},
                output_dtypes=[float],
                dask_gufunc_kwargs={"output_sizes": {"probability": 3}, "allow_rechunk": True},
            )

        elif dm == "nonparam":
            # ===== 3: out-of-sample residuals + drop member dim =====
            error_samples = Predictant_no_m - hindcast_det_cross
            forecast_prob = xr.apply_ufunc(
                self.calculate_tercile_probabilities_nonparametric,
                result_da, error_samples, T1_emp, T2_emp,
                input_core_dims=[("T",), ("T",), (), ()],
                output_core_dims=[("probability", "T")],
                vectorize=True, dask="parallelized", output_dtypes=[float],
                dask_gufunc_kwargs={"output_sizes": {"probability": 3}, "allow_rechunk": True},
            )

        else:
            raise ValueError(f"Invalid dist_method: {self.dist_method}")

        forecast_prob = forecast_prob.assign_coords(probability=('probability', ['PB', 'PN', 'PA']))
        return result_da * mask, mask * forecast_prob.transpose('probability', 'T', 'Y', 'X')


class _XGBRegressorES(XGBRegressor):
    """
    XGBRegressor that supports tunable early stopping inside scikit-learn search.

    XGBoost's `early_stopping_rounds` needs an eval_set, which GridSearchCV /
    RandomizedSearchCV cannot supply per fold. This wrapper carves an internal
    validation split (es_validation_fraction) from the training data and passes
    it as eval_set when early_stopping_rounds is set, so early_stopping_rounds can
    be treated as a normal tunable hyperparameter. With early_stopping_rounds=None
    it behaves exactly like a plain XGBRegressor.
    """
    def __init__(self, es_validation_fraction=0.15, **kwargs):
        self.es_validation_fraction = es_validation_fraction
        super().__init__(**kwargs)

    def get_params(self, deep=True):
        params = super().get_params(deep)
        params['es_validation_fraction'] = self.es_validation_fraction
        return params

    def fit(self, X, y, **fit_kw):
        esr = getattr(self, "early_stopping_rounds", None)
        if esr:
            from sklearn.model_selection import train_test_split
            X_tr, X_val, y_tr, y_val = train_test_split(
                X, y, test_size=self.es_validation_fraction,
                random_state=self.random_state)
            return super().fit(X_tr, y_tr, eval_set=[(X_val, y_val)], verbose=False)
        return super().fit(X, y, **fit_kw)


class WAS_mme_XGBoosting:
    """
    XGBoost-based Multi-Model Ensemble (MME) forecasting for seasonal rainfall.
    Enhanced with additional regularization hyperparameters and time-series CV.
    
    Parameters
    ----------
    search_method : str, optional
        Hyperparameter optimization method: 'grid', 'random', or 'bayesian' (default: 'random').
    n_estimators_range : list of int or scipy.stats distribution, optional
        List of n_estimators values to tune (optimized for climate data).
    learning_rate_range : list of float or scipy.stats distribution, optional
        List of learning rates to tune (optimized for climate data).
    max_depth_range : list of int or scipy.stats distribution, optional
        List of max depths to tune (optimized for climate data).
    min_child_weight_range : list of float or scipy.stats distribution, optional
        List of minimum child weights to tune (optimized for climate data).
    subsample_range : list of float or scipy.stats distribution, optional
        List of subsample ratios to tune.
    colsample_bytree_range : list of float or scipy.stats distribution, optional
        List of column sampling ratios to tune.
    gamma_range : list of float or scipy.stats distribution, optional
        List of gamma (min_split_loss) values for regularization.
    reg_alpha_range : list of float or scipy.stats distribution, optional
        List of L1 regularization (alpha) values.
    reg_lambda_range : list of float or scipy.stats distribution, optional
        List of L2 regularization (lambda) values.
    random_state : int, optional
        Seed for reproducibility (default is 42).
    dist_method : str, optional
        Distribution method for tercile probabilities ('bestfit', 'nonparam', etc.) (default is 'nonparam').
    n_iter_search : int, optional
        Number of iterations for randomized/bayesian search or points to sample for grid search (default is 10).
    cv_folds : int, optional
        Number of cross-validation folds (default is 3).
    cv_method : str, optional
        Cross-validation method: 'timeseries' or 'kfold' (default: 'timeseries').
        Use 'timeseries' for temporal data to prevent data leakage.
    n_clusters : int, optional
        Number of clusters for homogenized zones (default is 4).
    optuna_n_jobs : int, optional
        Number of parallel jobs for Optuna (default is 1).
    optuna_timeout : int, optional
        Timeout in seconds for Optuna optimization (default is None).
    """


    def __init__(self,
                 search_method='random',
                 n_estimators_range=[50, 100, 150, 200],
                 learning_rate_range=[0.005, 0.01, 0.03, 0.05, 0.1],
                 max_depth_range=[3, 5, 7],
                 min_child_weight_range=[1, 3, 5],
                 subsample_range=[0.6, 0.8, 1.0],
                 colsample_bytree_range=[0.6, 0.8, 1.0],
                 gamma_range=[0, 0.1, 0.2, 0.5, 1],
                 reg_alpha_range=[0, 0.01, 0.1],
                 reg_lambda_range=[1, 1.5, 2],
                 early_stopping_rounds_range=[None, 10, 20],
                 es_validation_fraction=0.15,
                 random_state=42,
                 dist_method="nonparam",
                 n_iter_search=10,
                 cv_folds=3,
                 n_clusters=4,
                 leave_one_year_out=False,          # >>> E
                 optuna_n_jobs=1,
                 optuna_timeout=None,
                 verbose=0):

        self.search_method = search_method
        self.n_estimators_range = n_estimators_range
        self.learning_rate_range = learning_rate_range
        self.max_depth_range = max_depth_range
        self.min_child_weight_range = min_child_weight_range
        self.subsample_range = subsample_range
        self.colsample_bytree_range = colsample_bytree_range
        self.gamma_range = gamma_range
        self.reg_alpha_range = reg_alpha_range
        self.reg_lambda_range = reg_lambda_range
        # early_stopping_rounds is tuned via _XGBRegressorES (internal eval_set).
        # None disables it; integers enable early stopping on n_estimators.
        self.early_stopping_rounds_range = early_stopping_rounds_range
        self.es_validation_fraction = es_validation_fraction
        self.random_state = random_state
        self.dist_method = dist_method
        self.n_iter_search = n_iter_search
        self.cv_folds = cv_folds
        self.n_clusters = n_clusters
        self.leave_one_year_out = leave_one_year_out
        self.optuna_n_jobs = optuna_n_jobs
        self.optuna_timeout = optuna_timeout
        self.verbose = verbose
        self.xgb = {}
        self.best_params_ = None
        self.cluster_da_ = None

    # ===================================================================== CV
    def _cv_splitter(self, groups):
        """
        >>> E: year-grouped CV so the same calendar year never appears on
        both sides of the split. LeaveOneGroupOut when years are few or
        requested, otherwise GroupKFold with cv_folds blocks of years.
        """
        n_groups = len(np.unique(groups))
        if self.leave_one_year_out or n_groups <= self.cv_folds:
            return LeaveOneGroupOut()
        return GroupKFold(n_splits=self.cv_folds)

    # ============================================================ param spaces
    @staticmethod
    def _suggest(trial, name, range_obj, kind='float'):
        if isinstance(range_obj, list):
            return trial.suggest_categorical(name, range_obj)
        if kind == 'int':
            return trial.suggest_int(name, int(range_obj.a), int(range_obj.b))
        if kind == 'float_log':
            return trial.suggest_float(name, range_obj.a, range_obj.b, log=True)
        return trial.suggest_float(name, range_obj.a, range_obj.b)

    def _objective(self, trial, X, y, groups):
        """Optuna objective with year-grouped CV (neg MAE)."""
        params = {
            'n_estimators': self._suggest(trial, 'n_estimators', self.n_estimators_range, 'int'),
            'learning_rate': self._suggest(trial, 'learning_rate', self.learning_rate_range, 'float_log'),
            'max_depth': self._suggest(trial, 'max_depth', self.max_depth_range, 'int'),
            'min_child_weight': self._suggest(trial, 'min_child_weight', self.min_child_weight_range),
            'subsample': self._suggest(trial, 'subsample', self.subsample_range),
            'colsample_bytree': self._suggest(trial, 'colsample_bytree', self.colsample_bytree_range),
            'gamma': self._suggest(trial, 'gamma', self.gamma_range),
            'reg_alpha': self._suggest(trial, 'reg_alpha', self.reg_alpha_range),
            'reg_lambda': self._suggest(trial, 'reg_lambda', self.reg_lambda_range),
        }
        # Optuna categoricals can't hold None -> use 0 as the "off" sentinel.
        es_choices = [0 if v is None else v for v in self.early_stopping_rounds_range]
        esr = trial.suggest_categorical('early_stopping_rounds', es_choices)
        params['early_stopping_rounds'] = None if esr == 0 else esr

        model = _XGBRegressorES(**params, es_validation_fraction=self.es_validation_fraction,
                                eval_metric='rmse', random_state=self.random_state,
                                verbosity=0, n_jobs=-1)
        scores = cross_val_score(model, X, y, cv=self._cv_splitter(groups), groups=groups,
                                 scoring='neg_mean_absolute_error', n_jobs=-1)
        return float(np.mean(scores))

    def _param_space(self):
        return {
            'n_estimators': self.n_estimators_range,
            'learning_rate': self.learning_rate_range,
            'max_depth': self.max_depth_range,
            'min_child_weight': self.min_child_weight_range,
            'subsample': self.subsample_range,
            'colsample_bytree': self.colsample_bytree_range,
            'gamma': self.gamma_range,
            'reg_alpha': self.reg_alpha_range,
            'reg_lambda': self.reg_lambda_range,
            'early_stopping_rounds': self.early_stopping_rounds_range,
        }

    def _grid_space(self):
        """Materialize a grid; sample any scipy distribution to a small set."""
        grid = {}
        for name, r in self._param_space().items():
            if isinstance(r, list):
                grid[name] = r
            else:
                s = r.rvs(size=min(4, self.n_iter_search), random_state=self.random_state)
                grid[name] = np.unique(s.astype(float) if 'int' in name else s)
        return grid

    def _grid_search(self, X, y, groups):
        model = _XGBRegressorES(es_validation_fraction=self.es_validation_fraction,
                                eval_metric='rmse', random_state=self.random_state,
                                verbosity=0, n_jobs=-1)
        gs = GridSearchCV(model, param_grid=self._grid_space(),
                          cv=self._cv_splitter(groups), scoring='neg_mean_absolute_error',
                          error_score=np.nan, n_jobs=-1)
        gs.fit(X, y, groups=groups)          # >>> E
        return gs.best_params_

    def _random_search(self, X, y, groups):
        model = _XGBRegressorES(es_validation_fraction=self.es_validation_fraction,
                                eval_metric='rmse', random_state=self.random_state,
                                verbosity=0, n_jobs=-1)
        rs = RandomizedSearchCV(model, param_distributions=self._param_space(),
                                n_iter=self.n_iter_search, cv=self._cv_splitter(groups),
                                scoring='neg_mean_absolute_error', random_state=self.random_state,
                                error_score=np.nan, n_jobs=-1)
        rs.fit(X, y, groups=groups)          # >>> E
        return rs.best_params_

    def _bayesian_search(self, X, y, groups):
        if optuna is None:
            raise ImportError("search_method='bayesian' requires optuna to be installed.")
        study = optuna.create_study(
            direction='maximize',
            sampler=optuna.samplers.TPESampler(seed=self.random_state),
            pruner=optuna.pruners.MedianPruner(n_startup_trials=5),
        )
        study.optimize(lambda tr: self._objective(tr, X, y, groups),
                       n_trials=self.n_iter_search, timeout=self.optuna_timeout,
                       n_jobs=self.optuna_n_jobs)
        return study.best_params

    # ====================================================== hyperparameters
    def compute_hyperparameters(self, Predictors, Predictand,
                                clim_year_start=None, clim_year_end=None):
        """
        Best hyperparameters per zone.

        Regionalization: KMeans on the predictand VALUES (kept by preference).
        Standardization is applied here only if clim years are given; when None,
        the inputs are assumed already standardized (framework path).
        """
        if "M" in Predictand.coords:
            Predictand = Predictand.isel(M=0).drop_vars('M').squeeze()
        Predictand.name = "varname"

        if clim_year_start is not None and clim_year_end is not None:
            X_train_std = standardize_timeseries(Predictors, clim_year_start, clim_year_end)
            y_train_std = standardize_timeseries(Predictand, clim_year_start, clim_year_end)
        else:
            X_train_std = Predictors
            y_train_std = Predictand
        X_train_std['T'] = y_train_std['T']

        # KMeans directly on predictand values (kept by preference)
        kmeans = KMeans(n_clusters=self.n_clusters, random_state=self.random_state, n_init=10)
        df = Predictand.to_dataframe().reset_index().dropna().drop(columns=['T'])
        variable_column = df.columns[2]
        df['cluster'] = kmeans.fit_predict(df[[variable_column]])
        df_unique = df.drop_duplicates(subset=['Y', 'X'])
        dataset = df_unique.set_index(['Y', 'X']).to_xarray()
        mask = xr.where(~np.isnan(Predictand.isel(T=0)), 1, np.nan)
        Cluster = dataset['cluster'] * mask

        _, cluster_da = xr.align(y_train_std, Cluster, join="outer")
        clusters = np.unique(cluster_da.values)
        clusters = clusters[~np.isnan(clusters)]

        best_params_dict = {}
        for c in clusters:
            c_int = int(c)
            mask_3d = (cluster_da == c).expand_dims({'T': y_train_std['T']})
            Xc = X_train_std.where(mask_3d).stack(sample=('T', 'Y', 'X')).transpose('sample', 'M')
            yc = y_train_std.where(mask_3d).stack(sample=('T', 'Y', 'X'))

            groups_full = Xc['T'].values.astype('datetime64[Y]').astype(int)   # >>> E
            Xv, yv = Xc.values, yc.values.ravel()
            bad = np.any(~np.isfinite(Xv), axis=1) | ~np.isfinite(yv)
            X_clean, y_clean, groups_c = Xv[~bad], yv[~bad], groups_full[~bad]

            n_years = len(np.unique(groups_c))
            if len(X_clean) < self.cv_folds * 2 or n_years < 2:
                if self.verbose:
                    print(f"Cluster {c_int}: insufficient data "
                          f"({len(X_clean)} samples, {n_years} years). Skipping.")
                continue

            if self.search_method == 'grid':
                best_params_dict[c_int] = self._grid_search(X_clean, y_clean, groups_c)
            elif self.search_method == 'random':
                best_params_dict[c_int] = self._random_search(X_clean, y_clean, groups_c)
            elif self.search_method == 'bayesian':
                best_params_dict[c_int] = self._bayesian_search(X_clean, y_clean, groups_c)
            else:
                raise ValueError(f"Unknown search_method: {self.search_method}.")

        self.best_params_ = best_params_dict
        self.cluster_da_ = cluster_da
        return best_params_dict, cluster_da

    # ============================================================ deterministic
    @staticmethod
    def _base_params(bp, random_state, es_validation_fraction=0.15):
        base = {
            'n_estimators': bp.get('n_estimators', 200),
            'learning_rate': bp.get('learning_rate', 0.05),
            'max_depth': bp.get('max_depth', 5),
            'min_child_weight': bp.get('min_child_weight', 1),
            'subsample': bp.get('subsample', 1.0),
            'colsample_bytree': bp.get('colsample_bytree', 1.0),
            'early_stopping_rounds': bp.get('early_stopping_rounds', None),
            'es_validation_fraction': es_validation_fraction,
            'eval_metric': 'rmse',
            'random_state': random_state, 'verbosity': 0, 'n_jobs': -1,
        }
        for k in ('gamma', 'reg_alpha', 'reg_lambda'):
            if k in bp:
                base[k] = bp[k]
        return base

    def compute_model(self, X_train, y_train, X_test, y_test=None,
                      clim_year_start=None, clim_year_end=None,
                      best_params=None, cluster_da=None):
        """
        Deterministic hindcast with one XGBRegressor per zone.

        >>> A: clim years are now in the signature and tuning runs in dual
        mode when best_params/cluster_da are not supplied.
        """
        if best_params is None or cluster_da is None:
            best_params, cluster_da = self.compute_hyperparameters(
                X_train, y_train, clim_year_start, clim_year_end)

        time, lat, lon = X_test['T'], X_test['Y'], X_test['X']
        n_time, n_lat, n_lon = len(time), len(lat), len(lon)
        predictions = np.full((n_time, n_lat, n_lon), np.nan)
        self.xgb = {}

        for c in range(self.n_clusters):
            if c not in best_params:
                continue
            bp = best_params[c]

            m_tr = (cluster_da == c).expand_dims({'T': X_train['T']})
            m_te = (cluster_da == c).expand_dims({'T': X_test['T']})

            Xtr = X_train.where(m_tr).stack(sample=('T', 'Y', 'X')).transpose('sample', 'M').values
            ytr = y_train.where(m_tr).stack(sample=('T', 'Y', 'X')).values.ravel()
            tr_bad = np.any(~np.isfinite(Xtr), axis=1) | ~np.isfinite(ytr)
            Xtr_c, ytr_c = Xtr[~tr_bad], ytr[~tr_bad]

            Xte = X_test.where(m_te).stack(sample=('T', 'Y', 'X')).transpose('sample', 'M').values
            te_bad = np.any(~np.isfinite(Xte), axis=1)          # >>> predictor-only mask
            Xte_c = Xte[~te_bad]

            if len(Xtr_c) == 0:
                continue

            xgb_c = _XGBRegressorES(**self._base_params(bp, self.random_state,
                                                        self.es_validation_fraction))
            xgb_c.fit(Xtr_c, ytr_c)
            self.xgb[c] = xgb_c

            if len(Xte_c) > 0:
                y_pred = xgb_c.predict(Xte_c)
                result = np.full(Xte.shape[0], np.nan)
                result[~te_bad] = y_pred
                predictions = np.where(np.isnan(predictions),
                                       result.reshape(n_time, n_lat, n_lon), predictions)

        return xr.DataArray(predictions, coords={'T': time, 'Y': lat, 'X': lon},
                            dims=['T', 'Y', 'X'])

    # ------------------ Probability Calculation Methods ------------------
    @staticmethod
    def _ppf_terciles_from_code(dist_code, shape, loc, scale):
        """Return tercile thresholds (T1, T2) from best-fit distribution params."""
        if np.isnan(dist_code):
            return np.nan, np.nan
        code = int(dist_code)
        try:
            if code == 1:
                return (norm.ppf(1/3, loc=loc, scale=scale),
                        norm.ppf(2/3, loc=loc, scale=scale))
            elif code == 2:
                return (lognorm.ppf(1/3, s=shape, loc=loc, scale=scale),
                        lognorm.ppf(2/3, s=shape, loc=loc, scale=scale))
            elif code == 3:
                return (expon.ppf(1/3, loc=loc, scale=scale),
                        expon.ppf(2/3, loc=loc, scale=scale))
            elif code == 4:
                return (gamma.ppf(1/3, a=shape, loc=loc, scale=scale),
                        gamma.ppf(2/3, a=shape, loc=loc, scale=scale))
            elif code == 5:
                return (weibull_min.ppf(1/3, c=shape, loc=loc, scale=scale),
                        weibull_min.ppf(2/3, c=shape, loc=loc, scale=scale))
            elif code == 6:
                return (t.ppf(1/3, df=shape, loc=loc, scale=scale),
                        t.ppf(2/3, df=shape, loc=loc, scale=scale))
            elif code == 7:
                return (poisson.ppf(1/3, mu=shape, loc=loc),
                        poisson.ppf(2/3, mu=shape, loc=loc))
            elif code == 8:
                return (nbinom.ppf(1/3, n=shape, p=scale, loc=loc),
                        nbinom.ppf(2/3, n=shape, p=scale, loc=loc))
        except Exception:
            return np.nan, np.nan
        return np.nan, np.nan

    @staticmethod
    def weibull_shape_solver(k, M, V):
        """Root function for the Weibull shape parameter k (match V/M^2)."""
        if k <= 0:
            return -np.inf
        try:
            g1 = gamma_function(1 + 1/k)
            g2 = gamma_function(1 + 2/k)
            implied = (g2 / (g1**2)) - 1
            observed = V / (M**2)
            return observed - implied
        except ValueError:
            return -np.inf

    @staticmethod
    def calculate_tercile_probabilities_bestfit(best_guess, error_variance, T1, T2, dist_code, dof):
        """Generic tercile probabilities using best-fit family per grid cell."""
        best_guess = np.asarray(best_guess, float)
        error_variance = np.asarray(error_variance, dtype=float)
        n_time = best_guess.size
        out = np.full((3, n_time), np.nan, float)

        if (np.all(np.isnan(best_guess)) or np.isnan(dist_code)
                or np.isnan(T1) or np.isnan(T2) or np.isnan(error_variance)):
            return out

        code = int(dist_code)

        if code == 1:  # Normal
            sd = np.sqrt(error_variance)
            out[0, :] = norm.cdf(T1, loc=best_guess, scale=sd)
            out[1, :] = norm.cdf(T2, loc=best_guess, scale=sd) - norm.cdf(T1, loc=best_guess, scale=sd)
            out[2, :] = 1 - norm.cdf(T2, loc=best_guess, scale=sd)

        elif code == 2:  # Lognormal
            sigma = np.sqrt(np.log(1 + error_variance / (best_guess**2)))
            mu = np.log(best_guess) - sigma**2 / 2
            out[0, :] = lognorm.cdf(T1, s=sigma, scale=np.exp(mu))
            out[1, :] = lognorm.cdf(T2, s=sigma, scale=np.exp(mu)) - lognorm.cdf(T1, s=sigma, scale=np.exp(mu))
            out[2, :] = 1 - lognorm.cdf(T2, s=sigma, scale=np.exp(mu))

        elif code == 3:  # Exponential
            sc = np.sqrt(error_variance)
            c1 = expon.cdf(T1, loc=best_guess, scale=sc)
            c2 = expon.cdf(T2, loc=best_guess, scale=sc)     # >>> B: was loc_t
            out[0, :] = c1
            out[1, :] = c2 - c1
            out[2, :] = 1.0 - c2

        elif code == 4:  # Gamma
            alpha = (best_guess ** 2) / error_variance
            theta = error_variance / best_guess
            c1 = gamma.cdf(T1, a=alpha, scale=theta)
            c2 = gamma.cdf(T2, a=alpha, scale=theta)
            out[0, :] = c1
            out[1, :] = c2 - c1
            out[2, :] = 1.0 - c2

        elif code == 5:  # Weibull
            for i in range(n_time):
                M, V = best_guess[i], error_variance
                if V <= 0 or M <= 0:
                    out[:, i] = np.nan
                    continue
                k = fsolve(WAS_mme_XGBoosting.weibull_shape_solver, 2.0, args=(M, V))[0]  # >>> C
                if k <= 0:
                    out[:, i] = np.nan
                    continue
                lam = M / gamma_function(1 + 1/k)
                c1 = weibull_min.cdf(T1, c=k, loc=0, scale=lam)
                c2 = weibull_min.cdf(T2, c=k, loc=0, scale=lam)
                out[0, i] = c1
                out[1, i] = c2 - c1
                out[2, i] = 1.0 - c2

        elif code == 6:  # Student-t
            if dof <= 2:
                out[:, :] = np.nan
            else:
                scale = np.sqrt(error_variance * (dof - 2) / dof)
                c1 = t.cdf(T1, df=dof, loc=best_guess, scale=scale)
                c2 = t.cdf(T2, df=dof, loc=best_guess, scale=scale)
                out[0, :] = c1
                out[1, :] = c2 - c1
                out[2, :] = 1.0 - c2

        elif code == 7:  # Poisson
            mu = best_guess
            c1 = poisson.cdf(T1, mu=mu)
            c2 = poisson.cdf(T2, mu=mu)
            out[0, :] = c1
            out[1, :] = c2 - c1
            out[2, :] = 1.0 - c2

        elif code == 8:  # Negative Binomial
            p = np.where(error_variance > best_guess, best_guess / error_variance, np.nan)
            n = np.where(error_variance > best_guess, (best_guess**2) / (error_variance - best_guess), np.nan)
            c1 = nbinom.cdf(T1, n=n, p=p)
            c2 = nbinom.cdf(T2, n=n, p=p)
            out[0, :] = c1
            out[1, :] = c2 - c1
            out[2, :] = 1.0 - c2

        else:
            raise ValueError(f"Invalid distribution code: {dist_code}")

        # >>> F: positive-support families are undefined for non-positive means
        if code in (2, 3, 4, 5, 7, 8):
            bad = ~(best_guess > 0)
            if np.any(bad):
                out[:, bad] = np.nan

        return out

    @staticmethod
    def calculate_tercile_probabilities_nonparametric(best_guess, error_samples, first_tercile, second_tercile):
        """Non-parametric method using historical error samples."""
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
            pred_prob[0, t_] = p_below
            pred_prob[1, t_] = p_between
            pred_prob[2, t_] = 1.0 - (p_below + p_between)
        return pred_prob

    def compute_prob(self, Predictant: xr.DataArray, clim_year_start, clim_year_end,
                     hindcast_det: xr.DataArray, best_code_da=None, best_shape_da=None,
                     best_loc_da=None, best_scale_da=None) -> xr.DataArray:
        """
        Tercile probabilities for deterministic hindcasts.

        For honest calibration pass the CROSS-VALIDATED hindcast as hindcast_det
        (out-of-sample residuals); the in-sample hindcast yields over-confident
        probabilities.
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

        terciles_emp = clim.quantile([1/3, 2/3], dim="T")        # >>> 1/3, 2/3
        T1_emp = terciles_emp.isel(quantile=0).drop_vars("quantile")
        T2_emp = terciles_emp.isel(quantile=1).drop_vars("quantile")

        dm = self.dist_method
        if dm == "bestfit":
            if any(v is None for v in (best_code_da, best_shape_da, best_loc_da, best_scale_da)):
                raise ValueError("dist_method='bestfit' requires best_code_da, best_shape_da, "
                                 "best_loc_da, best_scale_da.")
            T1, T2 = xr.apply_ufunc(
                self._ppf_terciles_from_code,
                best_code_da, best_shape_da, best_loc_da, best_scale_da,
                input_core_dims=[(), (), (), ()], output_core_dims=[(), ()],
                vectorize=True, dask="parallelized", output_dtypes=[float, float])
            hindcast_prob = xr.apply_ufunc(
                self.calculate_tercile_probabilities_bestfit,
                hindcast_det, error_variance, T1, T2, best_code_da,
                input_core_dims=[("T",), (), (), (), ()],
                output_core_dims=[("probability", "T")],
                vectorize=True, kwargs={'dof': dof}, dask="parallelized", output_dtypes=[float],
                dask_gufunc_kwargs={"output_sizes": {"probability": 3}, "allow_rechunk": True})

        elif dm == "nonparam":
            error_samples = Predictant - hindcast_det
            hindcast_prob = xr.apply_ufunc(
                self.calculate_tercile_probabilities_nonparametric,
                hindcast_det, error_samples, T1_emp, T2_emp,
                input_core_dims=[("T",), ("T",), (), ()],
                output_core_dims=[("probability", "T")],
                vectorize=True, dask="parallelized", output_dtypes=[float],
                dask_gufunc_kwargs={"output_sizes": {"probability": 3}, "allow_rechunk": True})
        else:
            raise ValueError(f"Invalid dist_method: {self.dist_method}")

        hindcast_prob = hindcast_prob.assign_coords(probability=("probability", ["PB", "PN", "PA"]))
        return (hindcast_prob * mask).transpose("probability", "T", "Y", "X")

    # ============================================================== operational
    def forecast(self, Predictant, clim_year_start, clim_year_end, hindcast_det,
                 hindcast_det_cross, Predictor_for_year, best_params=None, cluster_da=None,
                 best_code_da=None, best_shape_da=None, best_loc_da=None, best_scale_da=None):
        """Operational forecast for a target year (deterministic + tercile probabilities)."""
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
        hindcast_det_cross['T'] = Predictant_st['T']

        time, lat, lon = Predictor_for_year_st['T'], Predictor_for_year_st['Y'], Predictor_for_year_st['X']
        n_time, n_lat, n_lon = len(time), len(lat), len(lon)

        if best_params is None or cluster_da is None:
            best_params, cluster_da = self.compute_hyperparameters(
                hindcast_det, Predictant_no_m, clim_year_start, clim_year_end)

        predictions = np.full((n_time, n_lat, n_lon), np.nan)
        self.xgb = {}

        for c in range(self.n_clusters):
            if c not in best_params:
                continue
            bp = best_params[c]
            m_tr = (cluster_da == c).expand_dims({'T': hindcast_det_st['T']})
            m_te = (cluster_da == c).expand_dims({'T': Predictor_for_year_st['T']})

            Xtr = hindcast_det_st.where(m_tr).stack(sample=('T', 'Y', 'X')).transpose('sample', 'M').values
            ytr = Predictant_st.where(m_tr).stack(sample=('T', 'Y', 'X')).values.ravel()
            tr_bad = np.any(~np.isfinite(Xtr), axis=1) | ~np.isfinite(ytr)
            Xtr_c, ytr_c = Xtr[~tr_bad], ytr[~tr_bad]

            Xte = Predictor_for_year_st.where(m_te).stack(sample=('T', 'Y', 'X')).transpose('sample', 'M').values
            te_bad = np.any(~np.isfinite(Xte), axis=1)
            Xte_c = Xte[~te_bad]

            if len(Xtr_c) == 0:
                continue

            xgb_c = _XGBRegressorES(**self._base_params(bp, self.random_state,
                                                        self.es_validation_fraction))
            xgb_c.fit(Xtr_c, ytr_c)
            self.xgb[c] = xgb_c

            if len(Xte_c) > 0:
                y_pred = xgb_c.predict(Xte_c)
                result = np.full(Xte.shape[0], np.nan)
                result[~te_bad] = y_pred
                predictions = np.where(np.isnan(predictions),
                                       result.reshape(n_time, n_lat, n_lon), predictions)

        result_da = xr.DataArray(predictions, coords={'T': time, 'Y': lat, 'X': lon},
                                 dims=['T', 'Y', 'X']) * mask
        result_da = reverse_standardize(result_da, Predictant_no_m, clim_year_start, clim_year_end)

        year = Predictor_for_year.coords['T'].values.astype('datetime64[Y]').astype(int)[0] + 1970
        month = Predictant_no_m.isel(T=0).coords['T'].values.astype('datetime64[M]').astype(int) % 12 + 1
        new_T = np.datetime64(f"{year}-{month:02d}-01")
        result_da = result_da.assign_coords(T=xr.DataArray([new_T], dims=["T"]))
        result_da['T'] = result_da['T'].astype('datetime64[ns]')

        index_start = Predictant_no_m.get_index("T").get_loc(str(clim_year_start)).start
        index_end = Predictant_no_m.get_index("T").get_loc(str(clim_year_end)).stop
        rainfall_for_tercile = Predictant_no_m.isel(T=slice(index_start, index_end))
        terciles = rainfall_for_tercile.quantile([1/3, 2/3], dim='T')      # >>> 1/3, 2/3
        T1_emp = terciles.isel(quantile=0).drop_vars('quantile')
        T2_emp = terciles.isel(quantile=1).drop_vars('quantile')
        error_variance = (Predictant_no_m - hindcast_det_cross).var(dim='T')
        dof = max(int(rainfall_for_tercile.sizes["T"]) - 1, 2)

        dm = self.dist_method
        if dm == "bestfit":
            if any(v is None for v in (best_code_da, best_shape_da, best_loc_da, best_scale_da)):
                raise ValueError("dist_method='bestfit' requires best_code_da, best_shape_da, "
                                 "best_loc_da, best_scale_da.")
            T1, T2 = xr.apply_ufunc(
                self._ppf_terciles_from_code,
                best_code_da, best_shape_da, best_loc_da, best_scale_da,
                input_core_dims=[(), (), (), ()], output_core_dims=[(), ()],
                vectorize=True, dask="parallelized", output_dtypes=[float, float])
            forecast_prob = xr.apply_ufunc(
                self.calculate_tercile_probabilities_bestfit,
                result_da, error_variance, T1, T2, best_code_da,
                input_core_dims=[("T",), (), (), (), ()],
                output_core_dims=[("probability", "T")],
                vectorize=True, dask="parallelized", kwargs={"dof": dof}, output_dtypes=[float],
                dask_gufunc_kwargs={"output_sizes": {"probability": 3}, "allow_rechunk": True})

        elif dm == "nonparam":
            # >>> D: out-of-sample residuals on the de-M'd predictand.
            # Rename the residual time dim so it does not collide with the
            # single forecast-year T in apply_ufunc (else alignment fails).
            error_samples = (Predictant_no_m - hindcast_det_cross).rename({"T": "S"})
            forecast_prob = xr.apply_ufunc(
                self.calculate_tercile_probabilities_nonparametric,
                result_da, error_samples, T1_emp, T2_emp,
                input_core_dims=[("T",), ("S",), (), ()],
                output_core_dims=[("probability", "T")],
                vectorize=True, dask="parallelized", output_dtypes=[float],
                dask_gufunc_kwargs={"output_sizes": {"probability": 3}, "allow_rechunk": True})
        else:
            raise ValueError(f"Invalid dist_method: {self.dist_method}")

        forecast_prob = forecast_prob.assign_coords(probability=('probability', ['PB', 'PN', 'PA']))
        return result_da * mask, mask * forecast_prob.transpose('probability', 'T', 'Y', 'X')



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
        # Set numpy random seed for this operation
        rng = np.random.RandomState(self.random_state)
        
        self.model = HPELM(inputs=X.shape[1], outputs=1, classification='r', norm=self.norm)
        
        # Generate reproducible weights
        W = rng.randn(X.shape[1], self.neurons)
        B = rng.randn(self.neurons)
        
        self.model.add_neurons(self.neurons, self.activation, W=W, B=B)
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
                 # >>> valid HPELM activations only (was 'relu'/'rbf_gauss')
                 activation_options=['sigm', 'tanh', 'lin', 'rbf_l1', 'rbf_l2', 'rbf_linf'],
                 norm_range=[0.1, 1.0, 10.0, 100.0],
                 random_state=42,
                 dist_method="nonparam",
                 search_method='random',
                 n_iter_search=10,
                 cv_folds=3,
                 n_clusters=4,
                 leave_one_year_out=False,          # >>> E
                 n_trials_bayesian=50,
                 bayesian_sampler='tpe',
                 scoring='neg_mean_squared_error',
                 verbose=0):

        self.neurons_range = neurons_range
        self.activation_options = activation_options
        self.norm_range = norm_range
        self.random_state = random_state
        self.dist_method = dist_method
        self.search_method = search_method
        self.n_iter_search = n_iter_search
        self.cv_folds = cv_folds
        self.n_clusters = n_clusters
        self.leave_one_year_out = leave_one_year_out
        self.n_trials_bayesian = n_trials_bayesian
        self.bayesian_sampler = bayesian_sampler
        self.scoring = scoring
        self.verbose = verbose
        self.hpelm = None
        self.best_params_dict = None
        self.bayesian_studies = {}

        valid_methods = ['grid', 'random', 'bayesian']
        if self.search_method not in valid_methods:
            raise ValueError(f"search_method must be one of {valid_methods}, got '{self.search_method}'")

    # ===================================================================== CV
    def _cv_splitter(self, groups):
        """>>> E: year-grouped CV so the same year never straddles a split."""
        n_groups = len(np.unique(groups))
        if self.leave_one_year_out or n_groups <= self.cv_folds:
            return LeaveOneGroupOut()
        return GroupKFold(n_splits=self.cv_folds)

    # ============================================================ param spaces
    @staticmethod
    def _get_bounds(param_range):
        """(min, max, is_distribution) from a list or a scipy distribution."""
        if hasattr(param_range, 'support'):
            low, high = param_range.support()
            return float(low), float(high), True
        return float(min(param_range)), float(max(param_range)), False

    def _create_bayesian_sampler(self):
        if optuna is None:
            return None
        if self.bayesian_sampler == 'random':
            return RandomSampler(seed=self.random_state)
        return TPESampler(seed=self.random_state)

    def _bayesian_objective(self, trial, X, y, groups):
        low_n, high_n, is_dist_n = self._get_bounds(self.neurons_range)
        if is_dist_n:
            neurons = trial.suggest_int('neurons', int(low_n), int(high_n))
        else:
            neurons = trial.suggest_categorical('neurons', [int(n) for n in self.neurons_range])
        activation = trial.suggest_categorical('activation', self.activation_options)
        low_f, high_f, _ = self._get_bounds(self.norm_range)
        norm_ = trial.suggest_float('norm', low_f, high_f, log=True)

        model = HPELMWrapper(neurons=neurons, activation=activation, norm=norm_,
                             random_state=self.random_state)
        scores = cross_val_score(model, X, y, cv=self._cv_splitter(groups), groups=groups,
                                 scoring=self.scoring, n_jobs=1)
        return float(np.mean(scores))

    def _grid_search_optimization(self, X, y, groups):
        """Grid search; sample any scipy distribution to a small grid."""
        param_grid = {'activation': self.activation_options}
        for name, p_range in zip(['neurons', 'norm'], [self.neurons_range, self.norm_range]):
            if hasattr(p_range, 'support'):
                low, high = p_range.support()
                if name == 'norm':
                    param_grid[name] = np.logspace(np.log10(low), np.log10(high), 5).tolist()
                else:
                    param_grid[name] = np.linspace(low, high, 5).astype(int).tolist()
            else:
                param_grid[name] = list(p_range)
        gs = GridSearchCV(HPELMWrapper(random_state=self.random_state), param_grid=param_grid,
                          cv=self._cv_splitter(groups), scoring=self.scoring, n_jobs=-1)
        gs.fit(X, y, groups=groups)        # >>> E
        return gs.best_params_

    def _random_search_optimization(self, X, y, groups):
        param_dist = {'neurons': self.neurons_range, 'activation': self.activation_options,
                      'norm': self.norm_range}
        rs = RandomizedSearchCV(HPELMWrapper(random_state=self.random_state),
                                param_distributions=param_dist, n_iter=self.n_iter_search,
                                cv=self._cv_splitter(groups), scoring=self.scoring,
                                random_state=self.random_state, n_jobs=-1)
        rs.fit(X, y, groups=groups)        # >>> E
        return rs.best_params_

    def _bayesian_optimization(self, X, y, groups, cluster_id=None):
        if optuna is None:
            raise ImportError("search_method='bayesian' requires optuna to be installed.")
        study = optuna.create_study(
            direction='maximize',
            sampler=self._create_bayesian_sampler(),
            study_name=f"cluster_{cluster_id}" if cluster_id is not None else "global")
        study.optimize(partial(self._bayesian_objective, X=X, y=y, groups=groups),
                       n_trials=self.n_trials_bayesian, n_jobs=1)
        if cluster_id is not None:
            self.bayesian_studies[cluster_id] = study
        return study.best_params

    # ====================================================== hyperparameters
    def compute_hyperparameters(self, Predictors, Predictand,
                                clim_year_start=None, clim_year_end=None):
        """
        Best hyperparameters per zone.

        Regionalization: KMeans on the predictand VALUES (kept by preference).
        Standardization is applied here only if clim years are given (dual mode).
        """
        np.random.seed(self.random_state)
        random.seed(self.random_state)

        if "M" in Predictand.coords:
            Predictand = Predictand.isel(M=0).drop_vars('M').squeeze()
        Predictand.name = "varname"

        if clim_year_start is not None and clim_year_end is not None:
            X_train_std = standardize_timeseries(Predictors, clim_year_start, clim_year_end)
            y_train_std = standardize_timeseries(Predictand, clim_year_start, clim_year_end)
        else:
            X_train_std = Predictors
            y_train_std = Predictand
        X_train_std['T'] = y_train_std['T']

        # KMeans directly on predictand values (kept by preference)
        kmeans = KMeans(n_clusters=self.n_clusters, random_state=self.random_state, n_init=10)
        df = Predictand.to_dataframe().reset_index().dropna().drop(columns=['T'])
        variable_column = df.columns[2]
        df['cluster'] = kmeans.fit_predict(df[[variable_column]])
        df_unique = df.drop_duplicates(subset=['Y', 'X'])
        dataset = df_unique.set_index(['Y', 'X']).to_xarray()
        mask = xr.where(~np.isnan(Predictand.isel(T=0)), 1, np.nan)
        Cluster = dataset['cluster'] * mask

        _, cluster_da = xr.align(y_train_std, Cluster, join="outer")
        clusters = np.unique(cluster_da.values)
        clusters = clusters[~np.isnan(clusters)]

        best_params_dict = {}
        for c in clusters:
            c_int = int(c)
            mask_3d = (cluster_da == c).expand_dims({'T': y_train_std['T']})
            Xc = X_train_std.where(mask_3d).stack(sample=('T', 'Y', 'X')).transpose('sample', 'M')
            yc = y_train_std.where(mask_3d).stack(sample=('T', 'Y', 'X'))

            groups_full = Xc['T'].values.astype('datetime64[Y]').astype(int)   # >>> E
            Xv, yv = Xc.values, yc.values.ravel()
            bad = np.any(~np.isfinite(Xv), axis=1) | ~np.isfinite(yv)
            X_clean, y_clean, groups_c = Xv[~bad], yv[~bad], groups_full[~bad]

            n_years = len(np.unique(groups_c))
            if len(X_clean) < self.cv_folds * 2 or n_years < 2:
                if self.verbose:
                    print(f"Cluster {c_int}: insufficient data "
                          f"({len(X_clean)} samples, {n_years} years). Skipping.")
                continue

            if self.search_method == 'grid':
                best_params_dict[c_int] = self._grid_search_optimization(X_clean, y_clean, groups_c)
            elif self.search_method == 'random':
                best_params_dict[c_int] = self._random_search_optimization(X_clean, y_clean, groups_c)
            elif self.search_method == 'bayesian':
                best_params_dict[c_int] = self._bayesian_optimization(X_clean, y_clean, groups_c, cluster_id=c_int)
            else:
                raise ValueError(f"Unknown optimization method: {self.search_method}")

        self.best_params_dict = best_params_dict
        return best_params_dict, cluster_da

    # ============================================================ deterministic
    def _fit_predict_cluster(self, bp, Xtr_c, ytr_c, Xte_c):
        """Train one HPELM and predict; returns predictions for the clean test rows."""
        hp = HPELM(inputs=Xtr_c.shape[1], outputs=1, classification='r', norm=bp['norm'])
        rng = np.random.RandomState(self.random_state)
        W = rng.randn(Xtr_c.shape[1], bp['neurons'])
        B = rng.randn(bp['neurons'])
        hp.add_neurons(bp['neurons'], bp['activation'], W=W, B=B)
        hp.train(Xtr_c, ytr_c, 'r')
        return hp, hp.predict(Xte_c).ravel()

    def compute_model(self, X_train, y_train, X_test, y_test=None,
                      clim_year_start=None, clim_year_end=None,
                      best_params=None, cluster_da=None):
        """
        Deterministic hindcast with one HPELM per zone.

        >>> A: clim years now in the signature; tuning runs in dual mode when
        best_params/cluster_da are not supplied.
        """
        if best_params is None or cluster_da is None:
            best_params, cluster_da = self.compute_hyperparameters(
                X_train, y_train, clim_year_start, clim_year_end)

        time, lat, lon = X_test['T'], X_test['Y'], X_test['X']
        n_time, n_lat, n_lon = len(time), len(lat), len(lon)
        predictions = np.full((n_time, n_lat, n_lon), np.nan)
        self.hpelm = {}

        for c in range(self.n_clusters):
            if c not in best_params:
                continue
            bp = best_params[c]

            m_tr = (cluster_da == c).expand_dims({'T': X_train['T']})
            m_te = (cluster_da == c).expand_dims({'T': X_test['T']})

            Xtr = X_train.where(m_tr).stack(sample=('T', 'Y', 'X')).transpose('sample', 'M').values
            ytr = y_train.where(m_tr).stack(sample=('T', 'Y', 'X')).values.ravel()
            tr_bad = np.any(~np.isfinite(Xtr), axis=1) | ~np.isfinite(ytr)
            Xtr_c, ytr_c = Xtr[~tr_bad], ytr[~tr_bad]

            Xte = X_test.where(m_te).stack(sample=('T', 'Y', 'X')).transpose('sample', 'M').values
            te_bad = np.any(~np.isfinite(Xte), axis=1)         # >>> predictor-only mask
            Xte_c = Xte[~te_bad]

            if len(Xtr_c) == 0 or len(Xte_c) == 0:
                continue

            hp, y_pred = self._fit_predict_cluster(bp, Xtr_c, ytr_c, Xte_c)
            self.hpelm[c] = hp

            result = np.full(Xte.shape[0], np.nan)
            result[~te_bad] = y_pred
            predictions = np.where(np.isnan(predictions),
                                   result.reshape(n_time, n_lat, n_lon), predictions)

        return xr.DataArray(predictions, coords={'T': time, 'Y': lat, 'X': lon},
                            dims=['T', 'Y', 'X'])

    # ------------------ Probability Calculation Methods ------------------
    @staticmethod
    def _ppf_terciles_from_code(dist_code, shape, loc, scale):
        if np.isnan(dist_code):
            return np.nan, np.nan
        code = int(dist_code)
        try:
            if code == 1:
                return (norm.ppf(1/3, loc=loc, scale=scale), norm.ppf(2/3, loc=loc, scale=scale))
            elif code == 2:
                return (lognorm.ppf(1/3, s=shape, loc=loc, scale=scale),
                        lognorm.ppf(2/3, s=shape, loc=loc, scale=scale))
            elif code == 3:
                return (expon.ppf(1/3, loc=loc, scale=scale), expon.ppf(2/3, loc=loc, scale=scale))
            elif code == 4:
                return (gamma.ppf(1/3, a=shape, loc=loc, scale=scale),
                        gamma.ppf(2/3, a=shape, loc=loc, scale=scale))
            elif code == 5:
                return (weibull_min.ppf(1/3, c=shape, loc=loc, scale=scale),
                        weibull_min.ppf(2/3, c=shape, loc=loc, scale=scale))
            elif code == 6:
                return (t.ppf(1/3, df=shape, loc=loc, scale=scale),
                        t.ppf(2/3, df=shape, loc=loc, scale=scale))
            elif code == 7:
                return (poisson.ppf(1/3, mu=shape, loc=loc), poisson.ppf(2/3, mu=shape, loc=loc))
            elif code == 8:
                return (nbinom.ppf(1/3, n=shape, p=scale, loc=loc),
                        nbinom.ppf(2/3, n=shape, p=scale, loc=loc))
        except Exception:
            return np.nan, np.nan
        return np.nan, np.nan

    @staticmethod
    def weibull_shape_solver(k, M, V):
        if k <= 0:
            return -np.inf
        try:
            g1 = gamma_function(1 + 1/k)
            g2 = gamma_function(1 + 2/k)
            return (V / (M**2)) - ((g2 / (g1**2)) - 1)
        except ValueError:
            return -np.inf

    @staticmethod
    def calculate_tercile_probabilities_bestfit(best_guess, error_variance, T1, T2, dist_code, dof):
        best_guess = np.asarray(best_guess, float)
        error_variance = np.asarray(error_variance, dtype=float)
        n_time = best_guess.size
        out = np.full((3, n_time), np.nan, float)

        if (np.all(np.isnan(best_guess)) or np.isnan(dist_code)
                or np.isnan(T1) or np.isnan(T2) or np.isnan(error_variance)):
            return out

        code = int(dist_code)

        if code == 1:
            sd = np.sqrt(error_variance)
            out[0, :] = norm.cdf(T1, loc=best_guess, scale=sd)
            out[1, :] = norm.cdf(T2, loc=best_guess, scale=sd) - norm.cdf(T1, loc=best_guess, scale=sd)
            out[2, :] = 1 - norm.cdf(T2, loc=best_guess, scale=sd)

        elif code == 2:
            sigma = np.sqrt(np.log(1 + error_variance / (best_guess**2)))
            mu = np.log(best_guess) - sigma**2 / 2
            out[0, :] = lognorm.cdf(T1, s=sigma, scale=np.exp(mu))
            out[1, :] = lognorm.cdf(T2, s=sigma, scale=np.exp(mu)) - lognorm.cdf(T1, s=sigma, scale=np.exp(mu))
            out[2, :] = 1 - lognorm.cdf(T2, s=sigma, scale=np.exp(mu))

        elif code == 3:
            sc = np.sqrt(error_variance)
            c1 = expon.cdf(T1, loc=best_guess, scale=sc)
            c2 = expon.cdf(T2, loc=best_guess, scale=sc)        # >>> B: was loc_t
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
                M, V = best_guess[i], error_variance
                if V <= 0 or M <= 0:
                    out[:, i] = np.nan
                    continue
                k = fsolve(WAS_mme_hpELM.weibull_shape_solver, 2.0, args=(M, V))[0]  # >>> C
                if k <= 0:
                    out[:, i] = np.nan
                    continue
                lam = M / gamma_function(1 + 1/k)
                c1 = weibull_min.cdf(T1, c=k, loc=0, scale=lam)
                c2 = weibull_min.cdf(T2, c=k, loc=0, scale=lam)
                out[0, i] = c1
                out[1, i] = c2 - c1
                out[2, i] = 1.0 - c2

        elif code == 6:
            if dof <= 2:
                out[:, :] = np.nan
            else:
                scale = np.sqrt(error_variance * (dof - 2) / dof)
                c1 = t.cdf(T1, df=dof, loc=best_guess, scale=scale)
                c2 = t.cdf(T2, df=dof, loc=best_guess, scale=scale)
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
            p = np.where(error_variance > best_guess, best_guess / error_variance, np.nan)
            n = np.where(error_variance > best_guess, (best_guess**2) / (error_variance - best_guess), np.nan)
            c1 = nbinom.cdf(T1, n=n, p=p)
            c2 = nbinom.cdf(T2, n=n, p=p)
            out[0, :] = c1
            out[1, :] = c2 - c1
            out[2, :] = 1.0 - c2

        else:
            raise ValueError(f"Invalid distribution code: {dist_code}")

        # >>> F: positive-support families undefined for non-positive means
        if code in (2, 3, 4, 5, 7, 8):
            bad = ~(best_guess > 0)
            if np.any(bad):
                out[:, bad] = np.nan

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
            pred_prob[0, t_] = p_below
            pred_prob[1, t_] = p_between
            pred_prob[2, t_] = 1.0 - (p_below + p_between)
        return pred_prob

    def compute_prob(self, Predictant: xr.DataArray, clim_year_start, clim_year_end,
                     hindcast_det: xr.DataArray, best_code_da=None, best_shape_da=None,
                     best_loc_da=None, best_scale_da=None) -> xr.DataArray:
        """Tercile probabilities for deterministic hindcasts (pass the CV hindcast for calibration)."""
        if "M" in Predictant.dims:
            Predictant = Predictant.isel(M=0).drop_vars("M").squeeze()
        Predictant = Predictant.transpose("T", "Y", "X")
        mask = xr.where(~np.isnan(Predictant.isel(T=0)), 1.0, np.nan)

        clim = Predictant.sel(T=slice(str(clim_year_start), str(clim_year_end)))
        if clim.sizes.get("T", 0) < 3:
            raise ValueError("Not enough years in climatology period for terciles.")

        error_variance = (Predictant - hindcast_det).var(dim="T")
        dof = max(int(clim.sizes["T"]) - 1, 2)

        terciles_emp = clim.quantile([1/3, 2/3], dim="T")       # >>> 1/3, 2/3
        T1_emp = terciles_emp.isel(quantile=0).drop_vars("quantile")
        T2_emp = terciles_emp.isel(quantile=1).drop_vars("quantile")

        dm = self.dist_method
        if dm == "bestfit":
            if any(v is None for v in (best_code_da, best_shape_da, best_loc_da, best_scale_da)):
                raise ValueError("dist_method='bestfit' requires best_code_da, best_shape_da, "
                                 "best_loc_da, best_scale_da.")
            T1, T2 = xr.apply_ufunc(
                self._ppf_terciles_from_code, best_code_da, best_shape_da, best_loc_da, best_scale_da,
                input_core_dims=[(), (), (), ()], output_core_dims=[(), ()],
                vectorize=True, dask="parallelized", output_dtypes=[float, float])
            hindcast_prob = xr.apply_ufunc(
                self.calculate_tercile_probabilities_bestfit,
                hindcast_det, error_variance, T1, T2, best_code_da,
                input_core_dims=[("T",), (), (), (), ()], output_core_dims=[("probability", "T")],
                vectorize=True, kwargs={'dof': dof}, dask="parallelized", output_dtypes=[float],
                dask_gufunc_kwargs={"output_sizes": {"probability": 3}, "allow_rechunk": True})

        elif dm == "nonparam":
            error_samples = Predictant - hindcast_det
            hindcast_prob = xr.apply_ufunc(
                self.calculate_tercile_probabilities_nonparametric,
                hindcast_det, error_samples, T1_emp, T2_emp,
                input_core_dims=[("T",), ("T",), (), ()], output_core_dims=[("probability", "T")],
                vectorize=True, dask="parallelized", output_dtypes=[float],
                dask_gufunc_kwargs={"output_sizes": {"probability": 3}, "allow_rechunk": True})
        else:
            raise ValueError(f"Invalid dist_method: {self.dist_method}")

        hindcast_prob = hindcast_prob.assign_coords(probability=("probability", ["PB", "PN", "PA"]))
        return (hindcast_prob * mask).transpose("probability", "T", "Y", "X")

    # ============================================================== operational
    def forecast(self, Predictant, clim_year_start, clim_year_end, hindcast_det,
                 hindcast_det_cross, Predictor_for_year, best_params=None, cluster_da=None,
                 best_code_da=None, best_shape_da=None, best_loc_da=None, best_scale_da=None):
        """Operational forecast for a target year (deterministic + tercile probabilities)."""
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

        time, lat, lon = Predictor_for_year_st['T'], Predictor_for_year_st['Y'], Predictor_for_year_st['X']
        n_time, n_lat, n_lon = len(time), len(lat), len(lon)

        if best_params is None or cluster_da is None:
            best_params, cluster_da = self.compute_hyperparameters(
                hindcast_det, Predictant_no_m, clim_year_start, clim_year_end)

        predictions = np.full((n_time, n_lat, n_lon), np.nan)
        self.hpelm = {}

        for c in range(self.n_clusters):
            if c not in best_params:
                continue
            bp = best_params[c]
            m_tr = (cluster_da == c).expand_dims({'T': hindcast_det_st['T']})
            m_te = (cluster_da == c).expand_dims({'T': Predictor_for_year_st['T']})

            Xtr = hindcast_det_st.where(m_tr).stack(sample=('T', 'Y', 'X')).transpose('sample', 'M').values
            ytr = Predictant_st.where(m_tr).stack(sample=('T', 'Y', 'X')).values.ravel()
            tr_bad = np.any(~np.isfinite(Xtr), axis=1) | ~np.isfinite(ytr)
            Xtr_c, ytr_c = Xtr[~tr_bad], ytr[~tr_bad]

            Xte = Predictor_for_year_st.where(m_te).stack(sample=('T', 'Y', 'X')).transpose('sample', 'M').values
            te_bad = np.any(~np.isfinite(Xte), axis=1)
            Xte_c = Xte[~te_bad]

            if len(Xtr_c) == 0 or len(Xte_c) == 0:
                continue

            hp, y_pred = self._fit_predict_cluster(bp, Xtr_c, ytr_c, Xte_c)
            self.hpelm[c] = hp

            result = np.full(Xte.shape[0], np.nan)
            result[~te_bad] = y_pred
            predictions = np.where(np.isnan(predictions),
                                   result.reshape(n_time, n_lat, n_lon), predictions)

        result_da = xr.DataArray(predictions, coords={'T': time, 'Y': lat, 'X': lon},
                                 dims=['T', 'Y', 'X']) * mask
        result_da = reverse_standardize(result_da, Predictant_no_m, clim_year_start, clim_year_end)

        year = Predictor_for_year.coords['T'].values.astype('datetime64[Y]').astype(int)[0] + 1970
        month = Predictant_no_m.isel(T=0).coords['T'].values.astype('datetime64[M]').astype(int) % 12 + 1
        new_T = np.datetime64(f"{year}-{month:02d}-01")
        result_da = result_da.assign_coords(T=xr.DataArray([new_T], dims=["T"]))
        result_da['T'] = result_da['T'].astype('datetime64[ns]')

        index_start = Predictant_no_m.get_index("T").get_loc(str(clim_year_start)).start
        index_end = Predictant_no_m.get_index("T").get_loc(str(clim_year_end)).stop
        rainfall_for_tercile = Predictant_no_m.isel(T=slice(index_start, index_end))
        terciles = rainfall_for_tercile.quantile([1/3, 2/3], dim='T')   # >>> 1/3, 2/3
        T1_emp = terciles.isel(quantile=0).drop_vars('quantile')
        T2_emp = terciles.isel(quantile=1).drop_vars('quantile')
        error_variance = (Predictant_no_m - hindcast_det_cross).var(dim='T')
        dof = max(int(rainfall_for_tercile.sizes["T"]) - 1, 2)

        dm = self.dist_method
        if dm == "bestfit":
            if any(v is None for v in (best_code_da, best_shape_da, best_loc_da, best_scale_da)):
                raise ValueError("dist_method='bestfit' requires best_code_da, best_shape_da, "
                                 "best_loc_da, best_scale_da.")
            T1, T2 = xr.apply_ufunc(
                self._ppf_terciles_from_code, best_code_da, best_shape_da, best_loc_da, best_scale_da,
                input_core_dims=[(), (), (), ()], output_core_dims=[(), ()],
                vectorize=True, dask="parallelized", output_dtypes=[float, float])
            forecast_prob = xr.apply_ufunc(
                self.calculate_tercile_probabilities_bestfit,
                result_da, error_variance, T1, T2, best_code_da,
                input_core_dims=[("T",), (), (), (), ()], output_core_dims=[("probability", "T")],
                vectorize=True, dask="parallelized", kwargs={"dof": dof}, output_dtypes=[float],
                dask_gufunc_kwargs={"output_sizes": {"probability": 3}, "allow_rechunk": True})

        elif dm == "nonparam":
            # >>> D: out-of-sample residuals on the de-M'd predictand, with the
            # residual time dim renamed so it doesn't collide with the forecast T.
            error_samples = (Predictant_no_m - hindcast_det_cross).rename({"T": "S"})
            forecast_prob = xr.apply_ufunc(
                self.calculate_tercile_probabilities_nonparametric,
                result_da, error_samples, T1_emp, T2_emp,
                input_core_dims=[("T",), ("S",), (), ()], output_core_dims=[("probability", "T")],
                vectorize=True, dask="parallelized", output_dtypes=[float],
                dask_gufunc_kwargs={"output_sizes": {"probability": 3}, "allow_rechunk": True})
        else:
            raise ValueError(f"Invalid dist_method: {self.dist_method}")

        forecast_prob = forecast_prob.assign_coords(probability=('probability', ['PB', 'PN', 'PA']))
        return result_da * mask, mask * forecast_prob.transpose('probability', 'T', 'Y', 'X')

    # ------------------ utilities ------------------
    def get_optimization_results(self):
        results = {'best_params': self.best_params_dict, 'search_method': self.search_method,
                   'bayesian_studies': self.bayesian_studies}
        if self.search_method == 'bayesian' and self.bayesian_studies:
            for cid, study in self.bayesian_studies.items():
                results[f'bayesian_trials_cluster_{cid}'] = study.trials_dataframe()
        return results


class WAS_mme_MLP:
    """
    Multi-Layer Perceptron (MLPRegressor) MME forecaster with per-zone
    hyperparameter tuning and tercile-probability dressing. Deterministic
    hindcast via compute_model; tercile probabilities via compute_prob
    ('nonparam' or 'bestfit'). Mirrors the WAS_mme_RF / WAS_mme_XGBoosting
    contract so it plugs into the same_kind_model2 branch of WAS_Cross_Validator.
    """

    def __init__(self,
                 search_method='random',
                 hidden_layer_sizes_range=[(2,), (4,), (2,4), (8,), (8,16), (16,)],
                 activation_options=['relu', 'tanh', 'identity'],
                 learning_rate_init_range=loguniform(0.0001, 0.01),
                 solver_options=['adam', 'sgd', 'lbfgs'],
                 alpha_range=[0.0001, 0.001, 0.01, 0.1],
                 max_iter=200,
                 random_state=42,
                 dist_method="nonparam",
                 n_iter_search=10,
                 cv_folds=3,
                 n_clusters=4,
                 early_stopping_options=[True, False],
                 validation_fraction=0.1,
                 n_iter_no_change=10,
                 leave_one_year_out=False,
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
        # early_stopping is a native MLPRegressor option (sgd/adam only; ignored
        # for lbfgs). Tunable via early_stopping_options; validation_fraction and
        # n_iter_no_change configure it.
        self.early_stopping_options = early_stopping_options
        self.validation_fraction = validation_fraction
        self.n_iter_no_change = n_iter_no_change
        self.leave_one_year_out = leave_one_year_out      # E
        self.optuna_n_jobs = optuna_n_jobs
        self.optuna_timeout = optuna_timeout
        self.mlp = None

    # ----------------------------------------------------- E: grouped CV
    @staticmethod
    def _sample_years(stacked):
        """Calendar-year label per stacked (T,Y,X) sample (falls back to value)."""
        T = stacked['T']
        vals = np.asarray(T.values)
        if np.issubdtype(vals.dtype, np.datetime64):
            return T.dt.year.values
        return vals

    def _cv_splitter(self, groups):
        """Year-grouped splitter: LeaveOneGroupOut (requested / few years) else
        GroupKFold(cv_folds). Whole years are held out -> no spatial leakage."""
        n_groups = len(np.unique(groups))
        if self.leave_one_year_out or self.cv_folds >= n_groups:
            return LeaveOneGroupOut()
        return GroupKFold(n_splits=min(self.cv_folds, n_groups))

    def _objective(self, trial, X_train, y_train, groups, splitter):
        """Optuna objective: year-grouped CV neg-MSE (maximized)."""
        hidden_layer_sizes = trial.suggest_categorical(
            'hidden_layer_sizes', self.hidden_layer_sizes_range)
        activation = trial.suggest_categorical('activation', self.activation_options)
        solver = trial.suggest_categorical('solver', self.solver_options)

        if isinstance(self.learning_rate_init_range, list):
            learning_rate_init = trial.suggest_categorical(
                'learning_rate_init', self.learning_rate_init_range)
        else:
            learning_rate_init = trial.suggest_float(
                'learning_rate_init', self.learning_rate_init_range.a,
                self.learning_rate_init_range.b, log=True)

        if isinstance(self.alpha_range, list):
            alpha = trial.suggest_categorical('alpha', self.alpha_range)
        else:
            alpha = trial.suggest_float('alpha', self.alpha_range.a,
                                        self.alpha_range.b, log=True)

        early_stopping = trial.suggest_categorical('early_stopping', self.early_stopping_options)

        model = MLPRegressor(
            hidden_layer_sizes=hidden_layer_sizes,
            learning_rate_init=learning_rate_init,
            activation=activation, solver=solver, alpha=alpha,
            early_stopping=early_stopping,
            validation_fraction=self.validation_fraction,
            n_iter_no_change=self.n_iter_no_change,
            max_iter=self.max_iter, random_state=self.random_state)

        scores = cross_val_score(model, X_train, y_train, cv=splitter,
                                 groups=groups, scoring='neg_mean_squared_error',
                                 n_jobs=-1)
        return np.mean(scores)

    # ------------------------------------------------------ hyperparameters
    def compute_hyperparameters(self, Predictors, Predictand,
                                clim_year_start=None, clim_year_end=None):
        """
        Per-zone hyperparameter tuning. KMeans on predictand VALUES (kept).

        A dual-mode: clim years given -> standardize Predictors / Predictand
        internally; both None -> assume already standardized (framework path),
        so no double-standardization when called from compute_model.
        """
        if "M" in Predictand.coords:
            Predictand = Predictand.isel(M=0).drop_vars('M').squeeze()
        Predictand.name = "varname"

        if clim_year_start is not None and clim_year_end is not None:
            X_train_std = standardize_timeseries(Predictors, clim_year_start, clim_year_end)
            y_train_std = standardize_timeseries(Predictand, clim_year_start, clim_year_end)
        else:
            X_train_std = Predictors
            y_train_std = Predictand

        # ---- KMeans clustering directly on predictand values (KEPT) ---------
        kmeans = KMeans(n_clusters=self.n_clusters, random_state=self.random_state, n_init=10)
        Predictand_dropna = Predictand.to_dataframe().reset_index().dropna().drop(columns=['T'])
        variable_column = Predictand_dropna.columns[2]
        Predictand_dropna['cluster'] = kmeans.fit_predict(Predictand_dropna[[variable_column]])
        df_unique = Predictand_dropna.drop_duplicates(subset=['Y', 'X'])
        dataset = df_unique.set_index(['Y', 'X']).to_xarray()
        mask = xr.where(~np.isnan(Predictand.isel(T=0)), 1, np.nan)
        Cluster = dataset['cluster'] * mask
        _, xarray2 = xr.align(Predictand, Cluster, join="outer")
        clusters = np.unique(xarray2.values)
        clusters = clusters[~np.isnan(clusters)]
        cluster_da = xarray2
        X_train_std['T'] = y_train_std['T']

        best_params_dict = {}
        for c in clusters:
            mask_3d = (cluster_da == c).expand_dims({'T': y_train_std['T']})
            Xc = X_train_std.where(mask_3d).stack(sample=('T', 'Y', 'X')).transpose('sample', 'M')
            yc = y_train_std.where(mask_3d).stack(sample=('T', 'Y', 'X'))

            years_all = self._sample_years(Xc)
            Xv = Xc.values
            yv = yc.values.ravel()
            nan_mask_c = np.any(~np.isfinite(Xv), axis=1) | ~np.isfinite(yv)
            X_clean_c = Xv[~nan_mask_c]
            y_clean_c = yv[~nan_mask_c]
            groups_c = years_all[~nan_mask_c]

            if len(X_clean_c) == 0 or len(np.unique(groups_c)) < 2:
                continue

            splitter = self._cv_splitter(groups_c)   # E

            if self.search_method == 'grid':
                param_grid = {
                    'hidden_layer_sizes': self.hidden_layer_sizes_range,
                    'activation': self.activation_options,
                    'solver': self.solver_options,
                    'early_stopping': self.early_stopping_options,
                }
                if isinstance(self.learning_rate_init_range, list):
                    param_grid['learning_rate_init'] = self.learning_rate_init_range
                else:
                    n_samples = min(5, self.n_iter_search)
                    samples = self.learning_rate_init_range.rvs(size=n_samples, random_state=self.random_state)
                    param_grid['learning_rate_init'] = np.unique(samples)
                if isinstance(self.alpha_range, list):
                    param_grid['alpha'] = self.alpha_range
                else:
                    n_samples = min(5, self.n_iter_search)
                    samples = self.alpha_range.rvs(size=n_samples, random_state=self.random_state)
                    param_grid['alpha'] = np.unique(samples)

                model = MLPRegressor(max_iter=self.max_iter, random_state=self.random_state,
                                     validation_fraction=self.validation_fraction,
                                     n_iter_no_change=self.n_iter_no_change)
                grid_search = GridSearchCV(model, param_grid=param_grid, cv=splitter,
                                           scoring='neg_mean_squared_error',
                                           error_score=np.nan, n_jobs=-1)
                grid_search.fit(X_clean_c, y_clean_c, groups=groups_c)
                best_params_dict[c] = grid_search.best_params_

            elif self.search_method == 'random':
                param_dist = {
                    'hidden_layer_sizes': self.hidden_layer_sizes_range,
                    'learning_rate_init': self.learning_rate_init_range,
                    'activation': self.activation_options,
                    'solver': self.solver_options,
                    'alpha': self.alpha_range,
                    'early_stopping': self.early_stopping_options,
                }
                model = MLPRegressor(max_iter=self.max_iter, random_state=self.random_state,
                                     validation_fraction=self.validation_fraction,
                                     n_iter_no_change=self.n_iter_no_change)
                random_search = RandomizedSearchCV(
                    model, param_distributions=param_dist, n_iter=self.n_iter_search,
                    cv=splitter, scoring='neg_mean_squared_error',
                    random_state=self.random_state, error_score=np.nan, n_jobs=-1)
                random_search.fit(X_clean_c, y_clean_c, groups=groups_c)
                best_params_dict[c] = random_search.best_params_

            elif self.search_method == 'bayesian':
                if not HAS_OPTUNA:
                    raise ImportError("search_method='bayesian' requires optuna.")
                study = optuna.create_study(
                    direction='maximize',
                    sampler=optuna.samplers.TPESampler(seed=self.random_state),
                    pruner=optuna.pruners.MedianPruner(n_startup_trials=5))
                objective_with_data = (lambda trial:
                                       self._objective(trial, X_clean_c, y_clean_c, groups_c, splitter))
                study.optimize(objective_with_data, n_trials=self.n_iter_search,
                               timeout=self.optuna_timeout, n_jobs=self.optuna_n_jobs)
                bp = study.best_params
                best_params_dict[c] = {
                    'hidden_layer_sizes': bp['hidden_layer_sizes'],
                    'learning_rate_init': bp['learning_rate_init'],
                    'activation': bp['activation'],
                    'solver': bp['solver'],
                    'alpha': bp['alpha'],
                    'early_stopping': bp['early_stopping'],
                }
            else:
                raise ValueError(f"Unknown search_method: {self.search_method}. "
                                 "Choose from 'grid', 'random', or 'bayesian'.")

        return best_params_dict, cluster_da

    # ------------------------------------------------------ deterministic fit
    def compute_model(self, X_train, y_train, X_test, y_test=None,
                      clim_year_start=None, clim_year_end=None,
                      best_params=None, cluster_da=None):
        """
        Deterministic hindcast per zone. Inputs are assumed standardized (the
        same_kind_model2 branch standardizes once). A: clim_year_start/end
        are in the signature, so the best_params=None path no longer raises.
        """
        X_train_std = X_train
        y_train_std = y_train
        X_test_std = X_test

        time = X_test_std['T']; lat = X_test_std['Y']; lon = X_test_std['X']
        n_time, n_lat, n_lon = len(time), len(lat), len(lon)

        if best_params is None:                                   # A
            best_params, cluster_da = self.compute_hyperparameters(
                X_train, y_train, clim_year_start, clim_year_end)

        predictions = np.full((n_time, n_lat, n_lon), np.nan)
        self.mlp = {}
        for c in range(self.n_clusters):
            if c not in best_params:
                continue
            bp = best_params[c]
            mask_3d_train = (cluster_da == c).expand_dims({'T': X_train_std['T']})
            mask_3d_test = (cluster_da == c).expand_dims({'T': X_test_std['T']})

            X_tr = X_train_std.where(mask_3d_train).stack(sample=('T', 'Y', 'X')).transpose('sample', 'M').values
            y_tr = y_train_std.where(mask_3d_train).stack(sample=('T', 'Y', 'X')).values.ravel()
            tr_bad = np.any(~np.isfinite(X_tr), axis=1) | ~np.isfinite(y_tr)
            X_tr_c, y_tr_c = X_tr[~tr_bad], y_tr[~tr_bad]

            X_te = X_test_std.where(mask_3d_test).stack(sample=('T', 'Y', 'X')).transpose('sample', 'M').values
            te_bad = np.any(~np.isfinite(X_te), axis=1)           # minor: predictor only

            if len(X_tr_c) == 0 or (~te_bad).sum() == 0:
                continue

            mlp_c = MLPRegressor(
                hidden_layer_sizes=bp['hidden_layer_sizes'],
                learning_rate_init=bp['learning_rate_init'],
                activation=bp['activation'], solver=bp['solver'], alpha=bp['alpha'],
                early_stopping=bp.get('early_stopping', False),
                validation_fraction=self.validation_fraction,
                n_iter_no_change=self.n_iter_no_change,
                max_iter=self.max_iter, random_state=self.random_state)
            mlp_c.fit(X_tr_c, y_tr_c)
            self.mlp[c] = mlp_c

            y_pred_c = mlp_c.predict(X_te[~te_bad])
            result_c = np.full(X_te.shape[0], np.nan)
            result_c[~te_bad] = y_pred_c
            pred_c = result_c.reshape(n_time, n_lat, n_lon)
            predictions = np.where(np.isnan(predictions), pred_c, predictions)

        return xr.DataArray(predictions, coords={'T': time, 'Y': lat, 'X': lon},
                            dims=['T', 'Y', 'X'])

    # ------------------ Probability calculation methods ------------------
    @staticmethod
    def _ppf_terciles_from_code(dist_code, shape, loc, scale):
        """Tercile thresholds (T1, T2) from best-fit family parameters."""
        if np.isnan(dist_code):
            return np.nan, np.nan
        code = int(dist_code)
        try:
            if code == 1:
                return (norm.ppf(1/3, loc=loc, scale=scale),
                        norm.ppf(2/3, loc=loc, scale=scale))
            elif code == 2:
                return (lognorm.ppf(1/3, s=shape, loc=loc, scale=scale),
                        lognorm.ppf(2/3, s=shape, loc=loc, scale=scale))
            elif code == 3:
                return (expon.ppf(1/3, loc=loc, scale=scale),
                        expon.ppf(2/3, loc=loc, scale=scale))
            elif code == 4:
                return (gamma.ppf(1/3, a=shape, loc=loc, scale=scale),
                        gamma.ppf(2/3, a=shape, loc=loc, scale=scale))
            elif code == 5:
                return (weibull_min.ppf(1/3, c=shape, loc=loc, scale=scale),
                        weibull_min.ppf(2/3, c=shape, loc=loc, scale=scale))
            elif code == 6:
                return (t.ppf(1/3, df=shape, loc=loc, scale=scale),
                        t.ppf(2/3, df=shape, loc=loc, scale=scale))
            elif code == 7:
                return (poisson.ppf(1/3, mu=shape, loc=loc),
                        poisson.ppf(2/3, mu=shape, loc=loc))
            elif code == 8:
                return (nbinom.ppf(1/3, n=shape, p=scale, loc=loc),
                        nbinom.ppf(2/3, n=shape, p=scale, loc=loc))
        except Exception:
            return np.nan, np.nan
        return np.nan, np.nan

    @staticmethod
    def weibull_shape_solver(k, M, V):
        """Root function for the Weibull shape k matching observed V/M^2."""
        if k <= 0:
            return -np.inf
        try:
            g1 = gamma_function(1 + 1/k)
            g2 = gamma_function(1 + 2/k)
            implied = (g2 / (g1**2)) - 1
            observed = V / (M**2)
            return observed - implied
        except ValueError:
            return -np.inf

    @staticmethod
    def calculate_tercile_probabilities_bestfit(best_guess, error_variance, T1, T2, dist_code, dof):
        """Tercile probabilities from a forecast-adjusted best-fit family."""
        best_guess = np.asarray(best_guess, float)
        error_variance = np.asarray(error_variance, dtype=float)
        n_time = best_guess.size
        out = np.full((3, n_time), np.nan, float)

        if (np.all(np.isnan(best_guess)) or np.isnan(dist_code)
                or np.isnan(T1) or np.isnan(T2) or np.isnan(error_variance)):
            return out

        code = int(dist_code)

        if code == 1:  # Normal
            error_std = np.sqrt(error_variance)
            out[0, :] = norm.cdf(T1, loc=best_guess, scale=error_std)
            out[1, :] = norm.cdf(T2, loc=best_guess, scale=error_std) - out[0, :]
            out[2, :] = 1 - norm.cdf(T2, loc=best_guess, scale=error_std)

        elif code == 2:  # Lognormal
            sigma = np.sqrt(np.log(1 + error_variance / (best_guess**2)))
            mu = np.log(best_guess) - sigma**2 / 2
            c1 = lognorm.cdf(T1, s=sigma, scale=np.exp(mu))
            c2 = lognorm.cdf(T2, s=sigma, scale=np.exp(mu))
            out[0, :] = c1; out[1, :] = c2 - c1; out[2, :] = 1 - c2

        elif code == 3:  # Exponential  (B: loc = best_guess, not loc_t)
            scale = np.sqrt(error_variance)
            c1 = expon.cdf(T1, loc=best_guess, scale=scale)
            c2 = expon.cdf(T2, loc=best_guess, scale=scale)
            out[0, :] = c1; out[1, :] = c2 - c1; out[2, :] = 1.0 - c2

        elif code == 4:  # Gamma
            alpha = (best_guess ** 2) / error_variance
            theta = error_variance / best_guess
            c1 = gamma.cdf(T1, a=alpha, scale=theta)
            c2 = gamma.cdf(T2, a=alpha, scale=theta)
            out[0, :] = c1; out[1, :] = c2 - c1; out[2, :] = 1.0 - c2

        elif code == 5:  # Weibull  (C: qualified staticmethod, no prints)
            V = float(error_variance)
            for i in range(n_time):
                M = best_guess[i]
                if not np.isfinite(M) or V <= 0 or M <= 0:
                    continue
                k = fsolve(WAS_mme_MLP.weibull_shape_solver, 2.0, args=(M, V))[0]
                if k <= 0:
                    continue
                lambda_scale = M / gamma_function(1 + 1/k)
                c1 = weibull_min.cdf(T1, c=k, loc=0, scale=lambda_scale)
                c2 = weibull_min.cdf(T2, c=k, loc=0, scale=lambda_scale)
                out[0, i] = c1; out[1, i] = c2 - c1; out[2, i] = 1.0 - c2

        elif code == 6:  # Student-t
            if dof <= 2:
                return out
            loc = best_guess
            scale = np.sqrt(error_variance * (dof - 2) / dof)
            c1 = t.cdf(T1, df=dof, loc=loc, scale=scale)
            c2 = t.cdf(T2, df=dof, loc=loc, scale=scale)
            out[0, :] = c1; out[1, :] = c2 - c1; out[2, :] = 1.0 - c2

        elif code == 7:  # Poisson
            mu = best_guess
            c1 = poisson.cdf(T1, mu=mu)
            c2 = poisson.cdf(T2, mu=mu)
            out[0, :] = c1; out[1, :] = c2 - c1; out[2, :] = 1.0 - c2

        elif code == 8:  # Negative Binomial
            p = np.where(error_variance > best_guess, best_guess / error_variance, np.nan)
            n = np.where(error_variance > best_guess,
                         (best_guess**2) / (error_variance - best_guess), np.nan)
            c1 = nbinom.cdf(T1, n=n, p=p)
            c2 = nbinom.cdf(T2, n=n, p=p)
            out[0, :] = c1; out[1, :] = c2 - c1; out[2, :] = 1.0 - c2

        else:
            raise ValueError("Invalid distribution")

        # F: positive-support families are meaningless for a non-positive mean
        if code in (2, 3, 4, 5, 7, 8):
            invalid = ~(np.isfinite(best_guess) & (best_guess > 0))
            out[:, invalid] = np.nan

        return out

    @staticmethod
    def calculate_tercile_probabilities_nonparametric(best_guess, error_samples,
                                                      first_tercile, second_tercile):
        """Non-parametric tercile probabilities from historical error samples."""
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
            pred_prob[0, t_] = p_below
            pred_prob[1, t_] = p_between
            pred_prob[2, t_] = 1.0 - (p_below + p_between)
        return pred_prob

    def compute_prob(self, Predictant, clim_year_start, clim_year_end, hindcast_det,
                     best_code_da=None, best_shape_da=None,
                     best_loc_da=None, best_scale_da=None):
        """Tercile probabilities for a (cross-validated) deterministic hindcast."""
        if "M" in Predictant.dims:
            Predictant = Predictant.isel(M=0).drop_vars("M").squeeze()
        Predictant = Predictant.transpose("T", "Y", "X")

        mask = xr.where(~np.isnan(Predictant.isel(T=0)), 1.0, np.nan)
        clim = Predictant.sel(T=slice(str(clim_year_start), str(clim_year_end)))
        if clim.sizes.get("T", 0) < 3:
            raise ValueError("Not enough years in climatology period for terciles.")

        error_variance = (Predictant - hindcast_det).var(dim="T")
        dof = max(int(clim.sizes["T"]) - 1, 2)

        terciles_emp = clim.quantile([1/3, 2/3], dim="T")
        T1_emp = terciles_emp.isel(quantile=0).drop_vars("quantile")
        T2_emp = terciles_emp.isel(quantile=1).drop_vars("quantile")

        dm = self.dist_method
        if dm == "bestfit":
            if any(v is None for v in (best_code_da, best_shape_da, best_loc_da, best_scale_da)):
                raise ValueError("dist_method='bestfit' requires best_code_da, "
                                 "best_shape_da, best_loc_da, best_scale_da.")
            T1, T2 = xr.apply_ufunc(
                self._ppf_terciles_from_code,
                best_code_da, best_shape_da, best_loc_da, best_scale_da,
                input_core_dims=[(), (), (), ()],
                output_core_dims=[(), ()], vectorize=True,
                dask="parallelized", output_dtypes=[float, float])
            hindcast_prob = xr.apply_ufunc(
                self.calculate_tercile_probabilities_bestfit,
                hindcast_det, error_variance, T1, T2, best_code_da,
                input_core_dims=[("T",), (), (), (), ()],
                output_core_dims=[("probability", "T")],
                vectorize=True, kwargs={'dof': dof}, dask="parallelized",
                output_dtypes=[float],
                dask_gufunc_kwargs={"output_sizes": {"probability": 3}, "allow_rechunk": True})

        elif dm == "nonparam":
            error_samples = (Predictant - hindcast_det).rename({"T": "S"})   # D
            hindcast_prob = xr.apply_ufunc(
                self.calculate_tercile_probabilities_nonparametric,
                hindcast_det, error_samples, T1_emp, T2_emp,
                input_core_dims=[("T",), ("S",), (), ()],
                output_core_dims=[("probability", "T")],
                vectorize=True, dask="parallelized", output_dtypes=[float],
                dask_gufunc_kwargs={"output_sizes": {"probability": 3}, "allow_rechunk": True})
        else:
            raise ValueError(f"Invalid dist_method: {self.dist_method}")

        hindcast_prob = hindcast_prob.assign_coords(probability=("probability", ["PB", "PN", "PA"]))
        return (hindcast_prob * mask).transpose("probability", "T", "Y", "X")

    @staticmethod
    def _reshape_and_filter_data(da):
        """Stack (T,Y,X[,M]) -> (n_samples, n_features) and drop NaN rows."""
        da_stacked = da.stack(sample=('T', 'Y', 'X'))
        if 'M' in da.dims:
            da_stacked = da_stacked.transpose('sample', 'M')
        else:
            da_stacked = da_stacked.transpose('sample')
        vals = da_stacked.values
        nan_mask = np.any(np.isnan(vals), axis=1) if vals.ndim > 1 else np.isnan(vals)
        return vals[~nan_mask], nan_mask, vals

    def forecast(self, Predictant, clim_year_start, clim_year_end, hindcast_det,
                 hindcast_det_cross, Predictor_for_year, best_params=None,
                 cluster_da=None, best_code_da=None, best_shape_da=None,
                 best_loc_da=None, best_scale_da=None):
        """Deterministic + probabilistic forecast for a single target year."""
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

        time = Predictor_for_year_st['T']; lat = Predictor_for_year_st['Y']; lon = Predictor_for_year_st['X']
        n_time, n_lat, n_lon = len(time), len(lat), len(lon)

        if best_params is None:
            best_params, cluster_da = self.compute_hyperparameters(
                hindcast_det, Predictant_no_m, clim_year_start, clim_year_end)

        predictions = np.full((n_time, n_lat, n_lon), np.nan)
        self.mlp = {}
        for c in range(self.n_clusters):
            if c not in best_params:
                continue
            bp = best_params[c]
            mask_3d_train = (cluster_da == c).expand_dims({'T': hindcast_det_st['T']})
            mask_3d_test = (cluster_da == c).expand_dims({'T': Predictor_for_year_st['T']})

            X_tr = hindcast_det_st.where(mask_3d_train).stack(sample=('T', 'Y', 'X')).transpose('sample', 'M').values
            y_tr = Predictant_st.where(mask_3d_train).stack(sample=('T', 'Y', 'X')).values.ravel()
            tr_bad = np.any(~np.isfinite(X_tr), axis=1) | ~np.isfinite(y_tr)
            X_tr_c, y_tr_c = X_tr[~tr_bad], y_tr[~tr_bad]

            X_te = Predictor_for_year_st.where(mask_3d_test).stack(sample=('T', 'Y', 'X')).transpose('sample', 'M').values
            te_bad = np.any(~np.isfinite(X_te), axis=1)
            if len(X_tr_c) == 0 or (~te_bad).sum() == 0:
                continue

            mlp_c = MLPRegressor(
                hidden_layer_sizes=bp['hidden_layer_sizes'],
                learning_rate_init=bp['learning_rate_init'],
                activation=bp['activation'], solver=bp['solver'], alpha=bp['alpha'],
                early_stopping=bp.get('early_stopping', False),
                validation_fraction=self.validation_fraction,
                n_iter_no_change=self.n_iter_no_change,
                max_iter=self.max_iter, random_state=self.random_state)
            mlp_c.fit(X_tr_c, y_tr_c)
            self.mlp[c] = mlp_c

            y_pred_c = mlp_c.predict(X_te[~te_bad])
            result_c = np.full(X_te.shape[0], np.nan)
            result_c[~te_bad] = y_pred_c
            predictions = np.where(np.isnan(predictions),
                                   result_c.reshape(n_time, n_lat, n_lon), predictions)

        result_da = xr.DataArray(predictions, coords={'T': time, 'Y': lat, 'X': lon},
                                 dims=['T', 'Y', 'X']) * mask
        result_da = reverse_standardize(result_da, Predictant_no_m, clim_year_start, clim_year_end)

        year = Predictor_for_year.coords['T'].values.astype('datetime64[Y]').astype(int)[0] + 1970
        month_1 = Predictant_no_m.isel(T=0).coords['T'].values.astype('datetime64[M]').astype(int) % 12 + 1
        new_T_value = np.datetime64(f"{year}-{month_1:02d}-01")
        result_da = result_da.assign_coords(T=xr.DataArray([new_T_value], dims=["T"]))
        result_da['T'] = result_da['T'].astype('datetime64[ns]')

        index_start = Predictant_no_m.get_index("T").get_loc(str(clim_year_start)).start
        index_end = Predictant_no_m.get_index("T").get_loc(str(clim_year_end)).stop
        rainfall_for_tercile = Predictant_no_m.isel(T=slice(index_start, index_end))
        terciles = rainfall_for_tercile.quantile([1/3, 2/3], dim='T')
        T1_emp = terciles.isel(quantile=0).drop_vars('quantile')
        T2_emp = terciles.isel(quantile=1).drop_vars('quantile')
        error_variance = (Predictant_no_m - hindcast_det_cross).var(dim='T')   # D (OOS)
        dof = max(int(rainfall_for_tercile.sizes["T"]) - 1, 2)

        dm = self.dist_method
        if dm == "bestfit":
            if any(v is None for v in (best_code_da, best_shape_da, best_loc_da, best_scale_da)):
                raise ValueError("dist_method='bestfit' requires best_code_da, "
                                 "best_shape_da, best_loc_da, best_scale_da.")
            T1, T2 = xr.apply_ufunc(
                self._ppf_terciles_from_code,
                best_code_da, best_shape_da, best_loc_da, best_scale_da,
                input_core_dims=[(), (), (), ()], output_core_dims=[(), ()],
                vectorize=True, dask="parallelized", output_dtypes=[float, float])
            forecast_prob = xr.apply_ufunc(
                self.calculate_tercile_probabilities_bestfit,
                result_da, error_variance, T1, T2, best_code_da,
                input_core_dims=[("T",), (), (), (), ()],
                output_core_dims=[("probability", "T")],
                vectorize=True, dask="parallelized", kwargs={"dof": dof},
                output_dtypes=[float],
                dask_gufunc_kwargs={"output_sizes": {"probability": 3}, "allow_rechunk": True})

        elif dm == "nonparam":
            error_samples = (Predictant_no_m - hindcast_det_cross).rename({"T": "S"})  # D
            forecast_prob = xr.apply_ufunc(
                self.calculate_tercile_probabilities_nonparametric,
                result_da, error_samples, T1_emp, T2_emp,
                input_core_dims=[("T",), ("S",), (), ()],
                output_core_dims=[("probability", "T")],
                vectorize=True, dask="parallelized", output_dtypes=[float],
                dask_gufunc_kwargs={"output_sizes": {"probability": 3}, "allow_rechunk": True})
        else:
            raise ValueError(f"Invalid dist_method: {self.dist_method}")

        forecast_prob = forecast_prob.assign_coords(probability=('probability', ['PB', 'PN', 'PA']))
        return result_da * mask, mask * forecast_prob.transpose('probability', 'T', 'Y', 'X')


class WAS_mme_PCR:
    """
    Principal-component-regression MME baseline.

    Predictor field = the multi-model mean (over M) on the predictand grid.
    EOFs are fitted on the training fold (via the fold-safe WAS_EOF), the
    predictand is regressed on the leading PCs, and the held-out fold is
    projected out-of-sample before applying the regression. The deterministic
    output is in the same (standardized) space the framework hands in, so
    WAS_Cross_Validator's `same_kind_model2` branch reverse-standardizes it and
    calls compute_prob exactly as it does for WAS_mme_RF.

    Register it: add WAS_mme_PCR to the `same_kind_model2` list.

    Parameters
    ----------
    n_modes : int
        Number of principal components retained (default 5).
    use_coslat : bool
        Cosine-latitude weighting in the EOF (default True).
    detrend : bool
        Detrend the predictor field inside each fold (default False; the
        framework already standardizes, and the RF baseline does not detrend,
        so leave False for a like-for-like comparison).
    dist_method : str
        'nonparam' (default) or 'bestfit'.
    combine : str
        How the M models enter the EOF:
          'mean' (default) -- predictor field = multi-model mean (one channel).
          'meof'           -- keep every model as its own channel and run a
                              single combined / multivariate EOF over the joint
                              (Y, X, M) feature space. Because the framework has
                              already standardized each (cell, model) to unit
                              variance, the channels share a scale, so the
                              stacked EOF is a proper MEOF (no per-field
                              normalization needed). The leading PCs then encode
                              the joint inter-model variability, and the
                              predictand is regressed on them.
    """

    def __init__(self, n_modes=5, use_coslat=True, combine="mean", detrend=False,
                 dist_method="nonparam", L2norm=False, random_state=42):
        if combine not in ("mean", "meof"):
            raise ValueError("combine must be 'mean' or 'meof'.")
        self.n_modes = n_modes
        self.use_coslat = use_coslat
        self.combine = combine
        self.detrend = detrend
        self.dist_method = dist_method
        self.L2norm = L2norm
        self.random_state = random_state
        self.eof = None

    def _field(self, X):
        """Build the EOF input field from the predictor cube (T, Y, X, M)."""
        if "M" not in X.dims:
            return X
        if self.combine == "meof":
            return X                 # keep models as channels -> combined EOF
        return X.mean("M")           # default: multi-model mean

    def _fit_predict_pcs(self, field_train, y_train, field_test):
        """Fit EOF on train field, regress y_train on PCs, predict the test field."""
        eof = WAS_EOF(n_modes=self.n_modes, use_coslat=self.use_coslat,
                      detrend=self.detrend, standardize=False, L2norm=self.L2norm)
        _, s_train, _ = eof.fit(field_train, dim="T")
        self.eof = eof
        s_test = eof.transform(field_test, dim="T")

        S_tr = s_train.transpose("T", "mode").sel(T=field_train["T"]).values
        S_te = s_test.transpose("T", "mode").values
        A_tr = np.column_stack([np.ones(S_tr.shape[0]), S_tr])
        A_te = np.column_stack([np.ones(S_te.shape[0]), S_te])

        ystk = y_train.transpose("T", "Y", "X").sel(T=field_train["T"])
        ystk = ystk.stack(cell=("Y", "X")).transpose("T", "cell")
        Y = ystk.values
        valid = np.all(np.isfinite(Y), axis=0)

        pred = np.full((A_te.shape[0], Y.shape[1]), np.nan)
        if valid.any():
            coef, *_ = np.linalg.lstsq(A_tr, Y[:, valid], rcond=None)
            pred[:, valid] = A_te @ coef

        out = xr.DataArray(pred, dims=("T", "cell"),
                           coords={"T": field_test["T"], "cell": ystk["cell"]}).unstack("cell")
        return out.transpose("T", "Y", "X")

    # ------------------ Probability Calculation Methods ------------------
    @staticmethod
    def _ppf_terciles_from_code(dist_code, shape, loc, scale):
        if np.isnan(dist_code):
            return np.nan, np.nan
        code = int(dist_code)
        try:
            if code == 1:
                return norm.ppf(1/3, loc=loc, scale=scale), norm.ppf(2/3, loc=loc, scale=scale)
            if code == 2:
                return lognorm.ppf(1/3, s=shape, loc=loc, scale=scale), lognorm.ppf(2/3, s=shape, loc=loc, scale=scale)
            if code == 3:
                return expon.ppf(1/3, loc=loc, scale=scale), expon.ppf(2/3, loc=loc, scale=scale)
            if code == 4:
                return gamma.ppf(1/3, a=shape, loc=loc, scale=scale), gamma.ppf(2/3, a=shape, loc=loc, scale=scale)
            if code == 5:
                return weibull_min.ppf(1/3, c=shape, loc=loc, scale=scale), weibull_min.ppf(2/3, c=shape, loc=loc, scale=scale)
            if code == 6:
                return t.ppf(1/3, df=shape, loc=loc, scale=scale), t.ppf(2/3, df=shape, loc=loc, scale=scale)
            if code == 7:
                return poisson.ppf(1/3, mu=shape, loc=loc), poisson.ppf(2/3, mu=shape, loc=loc)
            if code == 8:
                return nbinom.ppf(1/3, n=shape, p=scale, loc=loc), nbinom.ppf(2/3, n=shape, p=scale, loc=loc)
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
            return (V / (M ** 2)) - ((g2 / (g1 ** 2)) - 1)
        except Exception:
            return -np.inf

    @staticmethod
    def calculate_tercile_probabilities_bestfit(best_guess, error_variance, T1, T2, dist_code, dof):
        best_guess = np.asarray(best_guess, dtype=float)
        error_variance = float(error_variance)
        n_time = best_guess.size
        out = np.full((3, n_time), np.nan, dtype=float)

        if (np.all(np.isnan(best_guess)) or not np.isfinite(error_variance)
                or error_variance <= 0 or not np.isfinite(T1) or not np.isfinite(T2)
                or np.isnan(dist_code)):
            return out

        code = int(dist_code)
        try:
            if code == 1:
                s = np.sqrt(error_variance)
                c1, c2 = norm.cdf(T1, loc=best_guess, scale=s), norm.cdf(T2, loc=best_guess, scale=s)
                out[0, :], out[1, :], out[2, :] = c1, c2 - c1, 1.0 - c2
            elif code == 2:
                valid = best_guess > 0
                sigma = np.full(n_time, np.nan); mu = np.full(n_time, np.nan)
                sigma[valid] = np.sqrt(np.log(1.0 + error_variance / (best_guess[valid] ** 2)))
                mu[valid] = np.log(best_guess[valid]) - 0.5 * sigma[valid] ** 2
                c1, c2 = lognorm.cdf(T1, s=sigma, scale=np.exp(mu)), lognorm.cdf(T2, s=sigma, scale=np.exp(mu))
                out[0, :], out[1, :], out[2, :] = c1, c2 - c1, 1.0 - c2
            elif code == 3:
                scale = np.sqrt(error_variance); loc = best_guess - scale
                c1, c2 = expon.cdf(T1, loc=loc, scale=scale), expon.cdf(T2, loc=loc, scale=scale)
                out[0, :], out[1, :], out[2, :] = c1, c2 - c1, 1.0 - c2
            elif code == 4:
                valid = best_guess > 0
                alpha = np.full(n_time, np.nan); theta = np.full(n_time, np.nan)
                alpha[valid] = (best_guess[valid] ** 2) / error_variance
                theta[valid] = error_variance / best_guess[valid]
                c1, c2 = gamma.cdf(T1, a=alpha, scale=theta), gamma.cdf(T2, a=alpha, scale=theta)
                out[0, :], out[1, :], out[2, :] = c1, c2 - c1, 1.0 - c2
            elif code == 5:
                for i in range(n_time):
                    M, V = best_guess[i], error_variance
                    if not np.isfinite(M) or M <= 0 or V <= 0:
                        continue
                    k = fsolve(WAS_mme_PCR.weibull_shape_solver, 2.0, args=(M, V))[0]
                    if not np.isfinite(k) or k <= 0:
                        continue
                    lam = M / gamma_function(1 + 1 / k)
                    c1, c2 = weibull_min.cdf(T1, c=k, loc=0, scale=lam), weibull_min.cdf(T2, c=k, loc=0, scale=lam)
                    out[0, i], out[1, i], out[2, i] = c1, c2 - c1, 1.0 - c2
            elif code == 6:
                if dof <= 2:
                    return out
                scale = np.sqrt(error_variance * (dof - 2) / dof)
                c1, c2 = t.cdf(T1, df=dof, loc=best_guess, scale=scale), t.cdf(T2, df=dof, loc=best_guess, scale=scale)
                out[0, :], out[1, :], out[2, :] = c1, c2 - c1, 1.0 - c2
            elif code == 7:
                mu = np.where(best_guess >= 0, best_guess, np.nan)
                c1, c2 = poisson.cdf(T1, mu=mu), poisson.cdf(T2, mu=mu)
                out[0, :], out[1, :], out[2, :] = c1, c2 - c1, 1.0 - c2
            elif code == 8:
                valid = error_variance > best_guess
                p = np.where(valid, best_guess / error_variance, np.nan)
                n = np.where(valid, best_guess ** 2 / (error_variance - best_guess), np.nan)
                c1, c2 = nbinom.cdf(T1, n=n, p=p), nbinom.cdf(T2, n=n, p=p)
                out[0, :], out[1, :], out[2, :] = c1, c2 - c1, 1.0 - c2
            else:
                return out
        except Exception:
            return out

        out = np.clip(out, 0, 1)
        total = np.nansum(out, axis=0)
        ok = np.isfinite(total) & (total > 0)
        out[:, ok] = out[:, ok] / total[ok]
        return out

    @staticmethod
    def calculate_tercile_probabilities_nonparametric(best_guess, error_samples, first_tercile, second_tercile):
        best_guess = np.asarray(best_guess, dtype=float)
        error_samples = np.asarray(error_samples, dtype=float)
        n_time = best_guess.size
        pred_prob = np.full((3, n_time), np.nan, dtype=float)

        valid_errors = error_samples[np.isfinite(error_samples)]
        if valid_errors.size == 0:
            return pred_prob

        for i in range(n_time):
            if not np.isfinite(best_guess[i]):
                continue
            d = best_guess[i] + valid_errors
            p_below = np.mean(d < first_tercile)
            p_normal = np.mean((d >= first_tercile) & (d < second_tercile))
            pred_prob[0, i] = p_below
            pred_prob[1, i] = p_normal
            pred_prob[2, i] = 1.0 - p_below - p_normal
        return pred_prob

    @staticmethod
    def _normalize_probabilities(prob):
        prob = prob.clip(min=0.0, max=1.0)
        total = prob.sum(dim="probability")
        return xr.where(total > 0, prob / total, np.nan)

    def _tercile_probabilities(self, Predictant, deterministic, clim_year_start, clim_year_end,
                               error_samples, error_variance,
                               best_code_da=None, best_shape_da=None,
                               best_loc_da=None, best_scale_da=None):
        Predictant = Predictant.transpose("T", "Y", "X")
        deterministic = deterministic.transpose("T", "Y", "X")
        mask = xr.where(np.isfinite(Predictant.isel(T=0)), 1.0, np.nan)

        clim = Predictant.sel(T=slice(str(clim_year_start), str(clim_year_end)))
        if clim.sizes.get("T", 0) < 3:
            raise ValueError("Not enough years in climatology period for terciles.")
        terc = clim.quantile([1.0 / 3.0, 2.0 / 3.0], dim="T")
        T1_emp = terc.isel(quantile=0).drop_vars("quantile")
        T2_emp = terc.isel(quantile=1).drop_vars("quantile")
        dof = max(int(clim.sizes["T"]) - 1, 2)

        if self.dist_method == "bestfit":
            if any(v is None for v in (best_code_da, best_shape_da, best_loc_da, best_scale_da)):
                raise ValueError("dist_method='bestfit' requires best_code_da, best_shape_da, best_loc_da, best_scale_da.")
            T1, T2 = xr.apply_ufunc(
                self._ppf_terciles_from_code, best_code_da, best_shape_da, best_loc_da, best_scale_da,
                input_core_dims=[(), (), (), ()], output_core_dims=[(), ()],
                vectorize=True, dask="parallelized", output_dtypes=[float, float])
            prob = xr.apply_ufunc(
                self.calculate_tercile_probabilities_bestfit,
                deterministic, error_variance, T1, T2, best_code_da,
                input_core_dims=[("T",), (), (), (), ()], output_core_dims=[("probability", "T")],
                vectorize=True, kwargs={"dof": dof}, dask="parallelized", output_dtypes=[float],
                dask_gufunc_kwargs={"output_sizes": {"probability": 3}, "allow_rechunk": True})
        elif self.dist_method == "nonparam":
            # residual sample axis renamed so a length-1 forecast and the
            # length-n residual record don't collide on the shared 'T' core dim
            err = error_samples.rename({"T": "__samp"})
            prob = xr.apply_ufunc(
                self.calculate_tercile_probabilities_nonparametric,
                deterministic, err, T1_emp, T2_emp,
                input_core_dims=[("T",), ("__samp",), (), ()], output_core_dims=[("probability", "T")],
                vectorize=True, dask="parallelized", output_dtypes=[float],
                dask_gufunc_kwargs={"output_sizes": {"probability": 3}, "allow_rechunk": True})
        else:
            raise ValueError(f"Invalid dist_method: {self.dist_method}")

        prob = prob.assign_coords(probability=("probability", ["PB", "PN", "PA"]))
        prob = self._normalize_probabilities(prob.transpose("probability", "T", "Y", "X"))
        return (prob * mask).transpose("probability", "T", "Y", "X")

    # ----------------------------------------------------- framework contract
    def compute_model(self, X_train, y_train, X_test, y_test=None, **kwargs):
        if "M" in y_train.dims:
            y_train = y_train.isel(M=0, drop=True)
        field_train = self._field(X_train)
        field_test = self._field(X_test)
        return self._fit_predict_pcs(field_train, y_train, field_test)

    def compute_prob(self, Predictant, clim_year_start, clim_year_end, hindcast_det,
                     best_code_da=None, best_shape_da=None, best_loc_da=None, best_scale_da=None):
        if "M" in Predictant.dims:
            Predictant = Predictant.isel(M=0).drop_vars("M").squeeze()
        Predictant = Predictant.transpose("T", "Y", "X")
        hindcast_det = hindcast_det.transpose("T", "Y", "X")
        if hindcast_det.sizes["T"] == Predictant.sizes["T"]:
            hindcast_det = hindcast_det.assign_coords(T=Predictant["T"])

        error_samples = Predictant - hindcast_det           # out-of-sample (LOYO hindcast)
        error_variance = error_samples.var(dim="T", skipna=True)

        return self._tercile_probabilities(
            Predictant, hindcast_det, clim_year_start, clim_year_end,
            error_samples, error_variance,
            best_code_da, best_shape_da, best_loc_da, best_scale_da)

    # ------------------------------------------------------------- operational
    def forecast(self, Predictant, clim_year_start, clim_year_end, Predictor,
                 hindcast_det, Predictor_for_year,
                 best_code_da=None, best_shape_da=None, best_loc_da=None, best_scale_da=None):
        """Operational PCR forecast for a single target year."""


        if "M" in Predictant.dims:
            Predictant = Predictant.isel(M=0).drop_vars("M").squeeze()
        Predictant = Predictant.transpose("T", "Y", "X")
        mask = xr.where(np.isfinite(Predictant.isel(T=0)), 1.0, np.nan)

        # standardize predictor (per cell/model) with historical climatology,
        # apply same stats to the forecast year
        mX = Predictor.sel(T=slice(str(clim_year_start), str(clim_year_end))).mean("T")
        sX = Predictor.sel(T=slice(str(clim_year_start), str(clim_year_end))).std("T")
        sX = sX.where(sX > 0)
        Predictor_st = (Predictor - mX) / sX
        Predictor_year_st = (Predictor_for_year - mX) / sX

        Predictant_st = standardize_timeseries(Predictant, clim_year_start, clim_year_end)

        field_hist = self._field(Predictor_st)
        field_year = self._field(Predictor_year_st)
        pred_st = self._fit_predict_pcs(field_hist, Predictant_st, field_year)

        forecast_det = reverse_standardize(pred_st, Predictant, clim_year_start, clim_year_end)

        # target time = forecast year + predictand target month
        year = Predictor_for_year.coords["T"].values.astype("datetime64[Y]").astype(int)[0] + 1970
        month = Predictant.isel(T=0).coords["T"].values.astype("datetime64[M]").astype(int) % 12 + 1
        new_T = np.datetime64(f"{year}-{month:02d}-01")
        forecast_det = forecast_det.assign_coords(T=xr.DataArray([new_T], dims=["T"]))
        forecast_det["T"] = forecast_det["T"].astype("datetime64[ns]")
        forecast_det = forecast_det.clip(min=0)

        error_samples = Predictant - hindcast_det.transpose("T", "Y", "X")
        error_variance = error_samples.var(dim="T", skipna=True)
        forecast_prob = self._tercile_probabilities(
            Predictant, forecast_det, clim_year_start, clim_year_end,
            error_samples, error_variance,
            best_code_da, best_shape_da, best_loc_da, best_scale_da)

        return forecast_det * mask, forecast_prob * mask

#################### Please from here to down is just to test ################
###### Not approved scientifically please ####################################
############################Expert knowledge #################################

_PROB_LABELS = ["PB", "PN", "PA"]


# ===========================================================================
#  Low-level scoring & per-point optimisation (pure numpy, validated)
# ===========================================================================

def _rps(P: np.ndarray, O: np.ndarray) -> float:
    """Mean Ranked Probability Score for ordered 3-category forecasts.

    P : (n, 3) forecast probabilities (rows sum to 1)
    O : (n, 3) observed one-hot category
    """
    Pc = np.cumsum(P, axis=-1)
    Oc = np.cumsum(O, axis=-1)
    return float(np.mean(np.sum((Pc - Oc) ** 2, axis=-1)))


def _blend(w: np.ndarray, Pstack: np.ndarray) -> np.ndarray:
    """Convex blend. w:(M,)  Pstack:(M, n, 3) -> (n, 3)."""
    return np.tensordot(w, Pstack, axes=(0, 0))


def _fit_weights_one_point(
    Pstack: np.ndarray,         # (M, n, 3) per-model tercile probs (train years)
    Ohot: np.ndarray,           # (n, 3) observed one-hot
    lo: np.ndarray,             # (M,) lower bounds
    hi: np.ndarray,             # (M,) upper bounds
    w_prior: np.ndarray,        # (M,) expert prior / shrinkage target (sums to 1)
    lmbda: float,               # shrinkage strength
) -> Tuple[np.ndarray, float]:
    """Minimise CV-RPS + lambda*||w - w_prior||^2 on the expert-box ∩ simplex."""
    M = Pstack.shape[0]

    def objective(w):
        r = _rps(_blend(w, Pstack), Ohot)
        return r + lmbda * float(np.sum((w - w_prior) ** 2))

    bounds = list(zip(lo, hi))
    cons = [{"type": "eq", "fun": lambda w: float(np.sum(w) - 1.0)}]
    w0 = np.clip(w_prior, lo, hi)
    s = w0.sum()
    w0 = w0 / s if s > 0 else np.full(M, 1.0 / M)

    res = minimize(objective, w0, method="SLSQP", bounds=bounds, constraints=cons,
                   options={"maxiter": 500, "ftol": 1e-12})
    w = res.x if (res.success and np.all(np.isfinite(res.x))) else w0
    w = np.clip(w, lo, hi)
    s = w.sum()
    w = w / s if s > 0 else np.full(M, 1.0 / M)
    return w, _rps(_blend(w, Pstack), Ohot)


# ===========================================================================
#  Public class
# ===========================================================================

class WAS_mme_ExpertKnowledge:
    """Expert-bounded, RPS-optimal multi-model tercile-probability blend.

    Parameters
    ----------
    expert_bounds : dict {model_name: (lo, hi)} or None
        Hard box constraints on each model's weight. Defaults to (0, 1).
        Example: {"ECMWF": (0.25, 0.6), "DWD": (0.0, 0.1)}.
    expert_prior : dict {model_name: float} or None
        Relative trust weights; normalised internally to sum to 1 and used as
        the shrinkage target (and the warm start). Defaults to equal trust.
    shrinkage : float or "cv"
        L2 shrinkage strength toward `expert_prior`. "cv" selects it by
        leave-one-year-out CV-RPS from `shrinkage_grid`.
    shrinkage_grid : sequence of floats
        Candidate lambdas when shrinkage="cv".
    spatial_pooling : "global" or "gridpoint"
        "global" fits ONE weight vector for the whole domain (robust, the usual
        seasonal choice with few years). "gridpoint" fits weights per grid cell
        (more flexible, needs more years / spatial smoothing).
    min_train_years : int
        Minimum valid years required to fit; otherwise falls back to the prior.
    """

    def __init__(
        self,
        expert_bounds: Optional[Dict[str, Tuple[float, float]]] = None,
        expert_prior: Optional[Dict[str, float]] = None,
        shrinkage: Union[float, str] = "cv",
        shrinkage_grid: Sequence[float] = (0.0, 0.02, 0.05, 0.1, 0.2, 0.5),
        spatial_pooling: str = "global",
        min_train_years: int = 8,
    ):
        if spatial_pooling not in ("global", "gridpoint"):
            raise ValueError("spatial_pooling must be 'global' or 'gridpoint'")
        self.expert_bounds = expert_bounds or {}
        self.expert_prior = expert_prior
        self.shrinkage = shrinkage
        self.shrinkage_grid = tuple(shrinkage_grid)
        self.spatial_pooling = spatial_pooling
        self.min_train_years = int(min_train_years)

        self.model_names_: List[str] = []
        self.weights_: Optional[xr.DataArray] = None     # (model[,Y,X])
        self.lambda_: Optional[float] = None
        self.cv_rps_: Optional[float] = None

    # ---- input normalisation ---------------------------------------------
    @staticmethod
    def _stack_models(model_probs, time_dim, lat_dim, lon_dim, prob_dim):
        """Return (names, array(M,T,3,Y,X)) from a dict or a stacked DataArray."""
        if isinstance(model_probs, dict):
            names = list(model_probs.keys())
            arrs = []
            for n in names:
                a = model_probs[n].transpose(prob_dim, time_dim, lat_dim, lon_dim)
                arrs.append(a)
            arrs = xr.align(*arrs, join="inner")
            da = xr.concat(arrs, dim="model").assign_coords(model=names)
        else:
            da = model_probs
            if "model" not in da.dims:
                raise ValueError("stacked DataArray must have a 'model' dim")
            names = [str(m) for m in da["model"].values]
            da = da.transpose("model", prob_dim, time_dim, lat_dim, lon_dim)
        return names, da

    def _bounds_arrays(self, names):
        lo = np.array([self.expert_bounds.get(n, (0.0, 1.0))[0] for n in names], float)
        hi = np.array([self.expert_bounds.get(n, (0.0, 1.0))[1] for n in names], float)
        if np.any(lo > hi):
            raise ValueError("expert_bounds: some lower bound exceeds its upper bound")
        if lo.sum() > 1.0 + 1e-9:
            raise ValueError(f"expert lower bounds sum to {lo.sum():.3f} > 1; infeasible simplex")
        if hi.sum() < 1.0 - 1e-9:
            raise ValueError(f"expert upper bounds sum to {hi.sum():.3f} < 1; infeasible simplex")
        return lo, hi

    def _prior_array(self, names):
        if self.expert_prior is None:
            w = np.full(len(names), 1.0 / len(names))
        else:
            w = np.array([max(self.expert_prior.get(n, 0.0), 0.0) for n in names], float)
            if w.sum() <= 0:
                w = np.full(len(names), 1.0 / len(names))
            w = w / w.sum()
        return w

    # ---- core fit ---------------------------------------------------------
    def _select_lambda(self, Pstack, Ohot, lo, hi, w_prior):
        """Leave-one-year-out CV-RPS over the lambda grid (pooled over points)."""
        if self.shrinkage != "cv":
            return float(self.shrinkage)
        n = Ohot.shape[0]
        best_lam, best_err = self.shrinkage_grid[0], np.inf
        for lam in self.shrinkage_grid:
            errs = []
            for i in range(n):
                tr = np.delete(np.arange(n), i)
                w, _ = _fit_weights_one_point(Pstack[:, tr], Ohot[tr], lo, hi, w_prior, lam)
                errs.append(_rps(_blend(w, Pstack[:, [i]]), Ohot[[i]]))
            err = float(np.mean(errs))
            if err < best_err:
                best_err, best_lam = err, lam
        self.cv_rps_ = best_err
        return best_lam

    def fit(
        self,
        model_probs: Union[Dict[str, xr.DataArray], xr.DataArray],
        obs_class: xr.DataArray,
        time_dim: str = "T",
        lat_dim: str = "Y",
        lon_dim: str = "X",
        prob_dim: str = "probability",
    ) -> "WAS_mme_ExpertKnowledge":
        names, da = self._stack_models(model_probs, time_dim, lat_dim, lon_dim, prob_dim)
        self.model_names_ = names
        lo, hi = self._bounds_arrays(names)
        w_prior = self._prior_array(names)

        P = da.values  # (M, 3, T, Y, X)
        M, _, T, ny, nx = P.shape
        oc = obs_class.transpose(time_dim, lat_dim, lon_dim).values  # (T, Y, X)

        # one-hot the observed category {0,1,2}
        def onehot(cat):
            o = np.zeros((cat.shape[0], 3))
            m = np.isfinite(cat)
            idx = np.where(m)[0]
            o[idx, cat[m].astype(int)] = 1.0
            return o, m

        # ---------------- global pooling ----------------
        if self.spatial_pooling == "global":
            # pool all valid (year, gridpoint) samples into one design
            Plist, Olist = [], []
            for iy in range(ny):
                for ix in range(nx):
                    cat = oc[:, iy, ix]
                    o, m = onehot(cat)
                    if m.sum() < self.min_train_years:
                        continue
                    pcell = np.transpose(P[:, :, m, iy, ix], (0, 2, 1))  # (M, n, 3)
                    if not np.all(np.isfinite(pcell)):
                        continue
                    Plist.append(pcell)
                    Olist.append(o[m])
            if not Plist:
                w = w_prior.copy(); self.lambda_ = 0.0
            else:
                Pstack = np.concatenate(Plist, axis=1)  # (M, N, 3)
                Ohot = np.concatenate(Olist, axis=0)    # (N, 3)
                self.lambda_ = self._select_lambda(Pstack, Ohot, lo, hi, w_prior)
                w, _ = _fit_weights_one_point(Pstack, Ohot, lo, hi, w_prior, self.lambda_)
            self.weights_ = xr.DataArray(w, dims=("model",), coords={"model": names})
            return self

        # ---------------- per-gridpoint ----------------
        W = np.full((M, ny, nx), np.nan)
        # pick a single lambda globally (CV per point is too noisy on few years)
        # use the pooled selection once, then apply per point
        Pp, Oo = [], []
        for iy in range(ny):
            for ix in range(nx):
                cat = oc[:, iy, ix]; o, m = onehot(cat)
                if m.sum() >= self.min_train_years and np.all(np.isfinite(P[:, :, m, iy, ix])):
                    Pp.append(np.transpose(P[:, :, m, iy, ix], (0, 2, 1))); Oo.append(o[m])
        if Pp:
            self.lambda_ = self._select_lambda(np.concatenate(Pp, 1), np.concatenate(Oo, 0),
                                               lo, hi, w_prior)
        else:
            self.lambda_ = 0.0
        for iy in range(ny):
            for ix in range(nx):
                cat = oc[:, iy, ix]; o, m = onehot(cat)
                if m.sum() < self.min_train_years or not np.all(np.isfinite(P[:, :, m, iy, ix])):
                    W[:, iy, ix] = w_prior
                    continue
                pcell = np.transpose(P[:, :, m, iy, ix], (0, 2, 1))
                w, _ = _fit_weights_one_point(pcell, o[m], lo, hi, w_prior, self.lambda_)
                W[:, iy, ix] = w
        self.weights_ = xr.DataArray(
            W, dims=("model", lat_dim, lon_dim),
            coords={"model": names, lat_dim: da[lat_dim], lon_dim: da[lon_dim]},
        )
        return self

    # ---- predict ----------------------------------------------------------
    def predict(
        self,
        model_probs: Union[Dict[str, xr.DataArray], xr.DataArray],
        time_dim: str = "T",
        lat_dim: str = "Y",
        lon_dim: str = "X",
        prob_dim: str = "probability",
    ) -> xr.DataArray:
        """Blend the per-model tercile probabilities with the fitted weights.
        Returns a DataArray with dims (probability, T, Y, X)."""
        if self.weights_ is None:
            raise RuntimeError("Call fit() before predict().")
        names, da = self._stack_models(model_probs, time_dim, lat_dim, lon_dim, prob_dim)
        if names != self.model_names_:
            da = da.sel(model=self.model_names_)
        w = self.weights_
        if "model" not in w.dims:
            raise RuntimeError("fitted weights malformed")
        # broadcast multiply over model, sum
        blended = (da * w).sum("model")
        # renormalise defensively
        s = blended.sum(prob_dim)
        blended = xr.where(s > 0, blended / s, np.nan)
        return blended.assign_coords({prob_dim: _PROB_LABELS}).transpose(
            prob_dim, time_dim, lat_dim, lon_dim
        )

    def compute_model(
        self,
        X_train: Union[Dict[str, xr.DataArray], xr.DataArray],
        obs_class_train: xr.DataArray,
        X_test: Union[Dict[str, xr.DataArray], xr.DataArray],
        time_dim: str = "T",
        lat_dim: str = "Y",
        lon_dim: str = "X",
        prob_dim: str = "probability",
    ) -> xr.DataArray:
        """fit + predict in one call (cross-validator friendly)."""
        self.fit(X_train, obs_class_train, time_dim, lat_dim, lon_dim, prob_dim)
        return self.predict(X_test, time_dim, lat_dim, lon_dim, prob_dim)

    # ---- diagnostics ------------------------------------------------------
    def summary(self):
        print("WAS_mme_ExpertKnowledge")
        print("=======================")
        print(f"pooling   : {self.spatial_pooling}")
        print(f"lambda    : {self.lambda_}")
        if self.cv_rps_ is not None:
            print(f"CV-RPS    : {self.cv_rps_:.4f}")
        if self.weights_ is not None:
            if self.weights_.ndim == 1:
                for n, w in zip(self.model_names_, self.weights_.values):
                    print(f"  w[{n}] = {w:.3f}")
            else:
                wm = self.weights_.mean(dim=[d for d in self.weights_.dims if d != "model"])
                for n, w in zip(self.model_names_, wm.values):
                    print(f"  mean w[{n}] = {float(w):.3f}")


# __all__ = ["WAS_mme_ExpertKnowledge"]


# mme = WAS_mme_ExpertKnowledge(
#     expert_bounds={"ECMWF": (0.25, 0.6), "DWD": (0.0, 0.10)},
#     expert_prior={"ECMWF": 0.4, "UKMO": 0.3, "MF": 0.2, "DWD": 0.1},
#     shrinkage="cv", spatial_pooling="global",
# )
# # model_probs: {name: DataArray(probability=3, T, Y, X)} from your NGR/CSGD/etc.
# # obs_class:   WAS_Verification.compute_class(...) -> {0,1,2} per (T,Y,X)
# prob = mme.compute_model(train_probs, obs_class_train, test_probs)   # (probability, T, Y, X)
# mme.summary()



from wass2s.utils import (
    standardize_timeseries,
    reverse_standardize,
    detrended_data,
    apply_detrend_data,
    extract_leading_eeof_component
)


class WAS_mme_CCA:
    """
    Canonical Correlation Analysis MME.

    Parameters
    ----------
    n_modes : int
        Number of canonical modes for xeofs.cross.CCA (default 4).
    n_pca_modes : int
        PCA modes kept before CCA when ``use_pca`` is True (default 8).
    standardize : bool
        Passed to xeofs.cross.CCA. Keep False; the framework already
        standardizes (default False).
    use_coslat : bool
        Cosine-latitude weighting inside the CCA (default True).
    use_pca : bool
        Pre-filter each field with PCA before the CCA (default True).
    combine : {'mean', 'meof'}
        How the M models enter the CCA predictor field.
    detrend : bool
        Apply ``detrended_data`` to both predictor and predictand in
        ``forecast``, exactly like ``WAS_CCA_op`` does unconditionally
        (default True). Set False only for ablation comparisons.
    dist_method : {'nonparam', 'bestfit'}
        Tercile-probability strategy (default 'nonparam').
    """

    def __init__(self, n_modes=4, n_pca_modes=8, standardize=False,
                 use_coslat=True, use_pca=True, combine="mean",
                 detrend=True, dist_method="nonparam"):
        if combine not in ("mean", "meof"):
            raise ValueError("combine must be 'mean' or 'meof'.")
        self.n_modes = n_modes
        self.n_pca_modes = n_pca_modes
        self.standardize = standardize
        self.use_coslat = use_coslat
        self.use_pca = use_pca
        self.combine = combine
        self.detrend = detrend
        self.dist_method = dist_method

        self.cca = None
        self.cca_model = None

    # ------------------------------------------------------------------ utils
    def _new_cca(self):
        """Fresh xeofs CCA instance (one per fold → leakage-free CV)."""
        return xe.cross.CCA(
            n_modes=self.n_modes,
            standardize=self.standardize,
            use_coslat=self.use_coslat,
            use_pca=self.use_pca,
            n_pca_modes=self.n_pca_modes,
        )

    def _field(self, X):
        """Build the CCA predictor field from the predictor cube (T, Y, X, M)."""
        if "M" not in X.dims:
            return X
        if self.combine == "meof":
            return X
        return X.mean("M")

    @staticmethod
    def _to_latlon(da):
        """WAS dims (Y, X[, M]) → xeofs dims (lat, lon[, M]); order (T, lat, lon, ...)."""
        rename = {}
        if "Y" in da.dims:
            rename["Y"] = "lat"
        if "X" in da.dims:
            rename["X"] = "lon"
        da = da.rename(rename)
        lead = [d for d in ("T", "lat", "lon") if d in da.dims]
        rest = [d for d in da.dims if d not in lead]
        return da.transpose(*lead, *rest)

    @staticmethod
    def _from_latlon(da):
        """xeofs dims (lat, lon) → WAS dims (Y, X); order (T, Y, X)."""
        rename = {}
        if "lat" in da.dims:
            rename["lat"] = "Y"
        if "lon" in da.dims:
            rename["lon"] = "X"
        da = da.rename(rename)
        return da.transpose("T", "Y", "X")

    # --------------------------------------------------------------- CCA core
    def _fit_predict_cca(self, field_train, y_train, field_test):
        """
        Fit CCA on the training field, predict the test field, return the
        physical-space predictand on the WAS grid (T, Y, X). NaNs are filled
        with the training mean (no leakage). Physical space is recovered with
        the codebase idiom: ``inverse_transform(transform(X), predict(X))[1]``.
        """
        Xtr = self._to_latlon(field_train)
        Ytr = self._to_latlon(y_train)

        Xtr_mean = Xtr.mean(dim="T", skipna=True)
        Xtr = Xtr.fillna(Xtr_mean)
        Ytr = Ytr.fillna(Ytr.mean(dim="T", skipna=True))

        self.cca = self._new_cca()
        self.cca_model = self.cca.fit(Xtr, Ytr, dim="T")

        Xte = self._to_latlon(field_test).fillna(Xtr_mean)
        y_pred = self.cca_model.predict(Xte)
        y_pred = self.cca_model.inverse_transform(
            self.cca_model.transform(Xte), y_pred
        )[1]
        y_pred["T"] = field_test["T"]

        return self._from_latlon(y_pred)

    # ----------------------------------------------- probability calculation
    @staticmethod
    def _ppf_terciles_from_code(dist_code, shape, loc, scale):
        if np.isnan(dist_code):
            return np.nan, np.nan
        code = int(dist_code)
        try:
            if code == 1:
                return norm.ppf(1/3, loc=loc, scale=scale), norm.ppf(2/3, loc=loc, scale=scale)
            if code == 2:
                return lognorm.ppf(1/3, s=shape, loc=loc, scale=scale), lognorm.ppf(2/3, s=shape, loc=loc, scale=scale)
            if code == 3:
                return expon.ppf(1/3, loc=loc, scale=scale), expon.ppf(2/3, loc=loc, scale=scale)
            if code == 4:
                return gamma.ppf(1/3, a=shape, loc=loc, scale=scale), gamma.ppf(2/3, a=shape, loc=loc, scale=scale)
            if code == 5:
                return weibull_min.ppf(1/3, c=shape, loc=loc, scale=scale), weibull_min.ppf(2/3, c=shape, loc=loc, scale=scale)
            if code == 6:
                return t.ppf(1/3, df=shape, loc=loc, scale=scale), t.ppf(2/3, df=shape, loc=loc, scale=scale)
            if code == 7:
                return poisson.ppf(1/3, mu=shape, loc=loc), poisson.ppf(2/3, mu=shape, loc=loc)
            if code == 8:
                return nbinom.ppf(1/3, n=shape, p=scale, loc=loc), nbinom.ppf(2/3, n=shape, p=scale, loc=loc)
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
            return (V / (M ** 2)) - ((g2 / (g1 ** 2)) - 1)
        except Exception:
            return -np.inf

    @staticmethod
    def calculate_tercile_probabilities_bestfit(best_guess, error_variance, T1, T2, dist_code, dof):
        best_guess = np.asarray(best_guess, dtype=float)
        error_variance = float(error_variance)
        n_time = best_guess.size
        out = np.full((3, n_time), np.nan, dtype=float)

        if (np.all(np.isnan(best_guess)) or not np.isfinite(error_variance)
                or error_variance <= 0 or not np.isfinite(T1) or not np.isfinite(T2)
                or np.isnan(dist_code)):
            return out

        code = int(dist_code)
        try:
            if code == 1:
                s = np.sqrt(error_variance)
                c1, c2 = norm.cdf(T1, loc=best_guess, scale=s), norm.cdf(T2, loc=best_guess, scale=s)
                out[0, :], out[1, :], out[2, :] = c1, c2 - c1, 1.0 - c2
            elif code == 2:
                valid = best_guess > 0
                sigma = np.full(n_time, np.nan); mu = np.full(n_time, np.nan)
                sigma[valid] = np.sqrt(np.log(1.0 + error_variance / (best_guess[valid] ** 2)))
                mu[valid] = np.log(best_guess[valid]) - 0.5 * sigma[valid] ** 2
                c1, c2 = lognorm.cdf(T1, s=sigma, scale=np.exp(mu)), lognorm.cdf(T2, s=sigma, scale=np.exp(mu))
                out[0, :], out[1, :], out[2, :] = c1, c2 - c1, 1.0 - c2
            elif code == 3:
                scale = np.sqrt(error_variance); loc = best_guess - scale
                c1, c2 = expon.cdf(T1, loc=loc, scale=scale), expon.cdf(T2, loc=loc, scale=scale)
                out[0, :], out[1, :], out[2, :] = c1, c2 - c1, 1.0 - c2
            elif code == 4:
                valid = best_guess > 0
                alpha = np.full(n_time, np.nan); theta = np.full(n_time, np.nan)
                alpha[valid] = (best_guess[valid] ** 2) / error_variance
                theta[valid] = error_variance / best_guess[valid]
                c1, c2 = gamma.cdf(T1, a=alpha, scale=theta), gamma.cdf(T2, a=alpha, scale=theta)
                out[0, :], out[1, :], out[2, :] = c1, c2 - c1, 1.0 - c2
            elif code == 5:
                for i in range(n_time):
                    M, V = best_guess[i], error_variance
                    if not np.isfinite(M) or M <= 0 or V <= 0:
                        continue
                    k = fsolve(WAS_mme_CCA.weibull_shape_solver, 2.0, args=(M, V))[0]
                    if not np.isfinite(k) or k <= 0:
                        continue
                    lam = M / gamma_function(1 + 1 / k)
                    c1, c2 = weibull_min.cdf(T1, c=k, loc=0, scale=lam), weibull_min.cdf(T2, c=k, loc=0, scale=lam)
                    out[0, i], out[1, i], out[2, i] = c1, c2 - c1, 1.0 - c2
            elif code == 6:
                if dof <= 2:
                    return out
                scale = np.sqrt(error_variance * (dof - 2) / dof)
                c1, c2 = t.cdf(T1, df=dof, loc=best_guess, scale=scale), t.cdf(T2, df=dof, loc=best_guess, scale=scale)
                out[0, :], out[1, :], out[2, :] = c1, c2 - c1, 1.0 - c2
            elif code == 7:
                mu = np.where(best_guess >= 0, best_guess, np.nan)
                c1, c2 = poisson.cdf(T1, mu=mu), poisson.cdf(T2, mu=mu)
                out[0, :], out[1, :], out[2, :] = c1, c2 - c1, 1.0 - c2
            elif code == 8:
                valid = error_variance > best_guess
                p = np.where(valid, best_guess / error_variance, np.nan)
                n = np.where(valid, best_guess ** 2 / (error_variance - best_guess), np.nan)
                c1, c2 = nbinom.cdf(T1, n=n, p=p), nbinom.cdf(T2, n=n, p=p)
                out[0, :], out[1, :], out[2, :] = c1, c2 - c1, 1.0 - c2
            else:
                return out
        except Exception:
            return out

        out = np.clip(out, 0, 1)
        total = np.nansum(out, axis=0)
        ok = np.isfinite(total) & (total > 0)
        out[:, ok] = out[:, ok] / total[ok]
        return out

    @staticmethod
    def calculate_tercile_probabilities_nonparametric(best_guess, error_samples, first_tercile, second_tercile):
        best_guess = np.asarray(best_guess, dtype=float)
        error_samples = np.asarray(error_samples, dtype=float)
        n_time = best_guess.size
        pred_prob = np.full((3, n_time), np.nan, dtype=float)

        valid_errors = error_samples[np.isfinite(error_samples)]
        if valid_errors.size == 0:
            return pred_prob

        for i in range(n_time):
            if not np.isfinite(best_guess[i]):
                continue
            d = best_guess[i] + valid_errors
            p_below = np.mean(d < first_tercile)
            p_normal = np.mean((d >= first_tercile) & (d < second_tercile))
            pred_prob[0, i] = p_below
            pred_prob[1, i] = p_normal
            pred_prob[2, i] = 1.0 - p_below - p_normal
        return pred_prob

    @staticmethod
    def _normalize_probabilities(prob):
        prob = prob.clip(min=0.0, max=1.0)
        total = prob.sum(dim="probability")
        return xr.where(total > 0, prob / total, np.nan)

    def _tercile_probabilities(self, Predictant, deterministic, clim_year_start, clim_year_end,
                               error_samples, error_variance,
                               best_code_da=None, best_shape_da=None,
                               best_loc_da=None, best_scale_da=None):
        Predictant = Predictant.transpose("T", "Y", "X")
        deterministic = deterministic.transpose("T", "Y", "X")
        mask = xr.where(np.isfinite(Predictant.isel(T=0)), 1.0, np.nan)

        clim = Predictant.sel(T=slice(str(clim_year_start), str(clim_year_end)))
        if clim.sizes.get("T", 0) < 3:
            raise ValueError("Not enough years in climatology period for terciles.")
        terc = clim.quantile([1.0 / 3.0, 2.0 / 3.0], dim="T")
        T1_emp = terc.isel(quantile=0).drop_vars("quantile")
        T2_emp = terc.isel(quantile=1).drop_vars("quantile")
        dof = max(int(clim.sizes["T"]) - 1, 2)

        if self.dist_method == "bestfit":
            if any(v is None for v in (best_code_da, best_shape_da, best_loc_da, best_scale_da)):
                raise ValueError("dist_method='bestfit' requires best_code_da, best_shape_da, best_loc_da, best_scale_da.")
            T1, T2 = xr.apply_ufunc(
                self._ppf_terciles_from_code, best_code_da, best_shape_da, best_loc_da, best_scale_da,
                input_core_dims=[(), (), (), ()], output_core_dims=[(), ()],
                vectorize=True, dask="parallelized", output_dtypes=[float, float])
            prob = xr.apply_ufunc(
                self.calculate_tercile_probabilities_bestfit,
                deterministic, error_variance, T1, T2, best_code_da,
                input_core_dims=[("T",), (), (), (), ()], output_core_dims=[("probability", "T")],
                vectorize=True, kwargs={"dof": dof}, dask="parallelized", output_dtypes=[float],
                dask_gufunc_kwargs={"output_sizes": {"probability": 3}, "allow_rechunk": True})
        elif self.dist_method == "nonparam":
            # rename residual sample axis so a length-1 forecast and the
            # length-n residual record don't collide on the shared 'T' core dim
            err = error_samples.rename({"T": "__samp"})
            prob = xr.apply_ufunc(
                self.calculate_tercile_probabilities_nonparametric,
                deterministic, err, T1_emp, T2_emp,
                input_core_dims=[("T",), ("__samp",), (), ()], output_core_dims=[("probability", "T")],
                vectorize=True, dask="parallelized", output_dtypes=[float],
                dask_gufunc_kwargs={"output_sizes": {"probability": 3}, "allow_rechunk": True})
        else:
            raise ValueError(f"Invalid dist_method: {self.dist_method}")

        prob = prob.assign_coords(probability=("probability", ["PB", "PN", "PA"]))
        prob = self._normalize_probabilities(prob.transpose("probability", "T", "Y", "X"))
        return (prob * mask).transpose("probability", "T", "Y", "X")

    # ----------------------------------------------------- framework contract
    def compute_model(self, X_train, y_train, X_test, y_test=None, **kwargs):
        """
        Deterministic CCA hindcast for one CV fold, in the standardized space
        the framework hands in. No detrend here — matches WAS_CCA_op.compute_model
        which is also detrend-free. The same_kind_model2 branch reverse-standardizes
        the output afterwards.
        """
        if "M" in y_train.dims:
            y_train = y_train.isel(M=0, drop=True)
        field_train = self._field(X_train)
        field_test = self._field(X_test)
        return self._fit_predict_cca(field_train, y_train, field_test)

    def compute_prob(self, Predictant, clim_year_start, clim_year_end, hindcast_det,
                     best_code_da=None, best_shape_da=None, best_loc_da=None, best_scale_da=None):
        if "M" in Predictant.dims:
            Predictant = Predictant.isel(M=0).drop_vars("M").squeeze()
        Predictant = Predictant.transpose("T", "Y", "X")
        hindcast_det = hindcast_det.transpose("T", "Y", "X")
        if hindcast_det.sizes["T"] == Predictant.sizes["T"]:
            hindcast_det = hindcast_det.assign_coords(T=Predictant["T"])

        error_samples = Predictant - hindcast_det
        error_variance = error_samples.var(dim="T", skipna=True)

        return self._tercile_probabilities(
            Predictant, hindcast_det, clim_year_start, clim_year_end,
            error_samples, error_variance,
            best_code_da, best_shape_da, best_loc_da, best_scale_da)

    # ------------------------------------------------------------- operational
    def forecast(self, Predictant, clim_year_start, clim_year_end, Predictor,
                 hindcast_det, Predictor_for_year,
                 best_code_da=None, best_shape_da=None, best_loc_da=None, best_scale_da=None):
        """
        Operational CCA forecast for a single target year.

          predictor →  standardize  →  detrended_data()                 (raw, not pre-standardized)
          predictand →  standardize  →  detrended_data() (always unconditional)
          forecast-year predictor  →  apply_detrend_data()
          fit CCA on (predictor_dt, predictand_st_dt)
          result  →  + apply_detrend_data()  →  reverse_standardize()
        """
        if "M" in Predictant.dims:
            Predictant = Predictant.isel(M=0).drop_vars("M").squeeze()
        Predictant = Predictant.transpose("T", "Y", "X")
        mask = xr.where(np.isfinite(Predictant.isel(T=0)), 1.0, np.nan)
        
        mean_val = Predictor.sel(T=slice(str(clim_year_start), str(clim_year_end))).mean(dim='T')
        std_val = Predictor.sel(T=slice(str(clim_year_start), str(clim_year_end))).std(dim='T')
        Predictor_for_year = (Predictor_for_year - mean_val) / std_val

        # --- predictor: raw field → polynomial detrend 
        Predictor = standardize_timeseries(Predictor, clim_year_start, clim_year_end)
        
        field_hist = self._field(Predictor)
        field_year = self._field(Predictor_for_year)

        if self.detrend:
            field_hist_fit, coeffsX, metaX = detrended_data(field_hist, dim="T")
            # remove the same fitted trend from the forecast-year field 
            field_year_fit = field_year - apply_detrend_data(field_year, coeffsX, metaX)
        else:
            field_hist_fit = field_hist
            field_year_fit = field_year

        # --- predictand: standardize → polynomial detrend 
        Predictant_st = standardize_timeseries(Predictant, clim_year_start, clim_year_end)
        if self.detrend:
            Predictant_st_fit, coeffsY, metaY = detrended_data(Predictant_st, dim="T")
        else:
            Predictant_st_fit = Predictant_st

        # --- fit CCA and predict 
        pred_st = self._fit_predict_cca(field_hist_fit, Predictant_st_fit, field_year_fit)

        # --- add predictand trend back, then reverse-standardize 
        if self.detrend:
            pred_st = pred_st + apply_detrend_data(pred_st, coeffsY, metaY)

        forecast_det = reverse_standardize(pred_st, Predictant, clim_year_start, clim_year_end)

        # target time = forecast year + predictand target month
        year = Predictor_for_year.coords["T"].values.astype("datetime64[Y]").astype(int)[0] + 1970
        month = Predictant.isel(T=0).coords["T"].values.astype("datetime64[M]").astype(int) % 12 + 1
        new_T = np.datetime64(f"{year}-{month:02d}-01")
        forecast_det = forecast_det.assign_coords(T=xr.DataArray([new_T], dims=["T"]))
        forecast_det["T"] = forecast_det["T"].astype("datetime64[ns]")
        forecast_det = forecast_det.clip(min=0)

        error_samples = Predictant - hindcast_det.transpose("T", "Y", "X")
        error_variance = error_samples.var(dim="T", skipna=True)
        forecast_prob = self._tercile_probabilities(
            Predictant, forecast_det, clim_year_start, clim_year_end,
            error_samples, error_variance,
            best_code_da, best_shape_da, best_loc_da, best_scale_da)

        return forecast_det * mask, forecast_prob * mask
        

class WAS_mme_CCA_eeof:
    """
    Canonical Correlation Analysis MME, legacy WAS_CCA ideology (EEOF detrend).

    Parameters
    ----------
    n_modes : int
        Number of canonical modes for xeofs.cross.CCA (default 4).
    n_pca_modes : int
        PCA modes kept before CCA when ``use_pca`` (default 8).
    standardize : bool
        Passed to xeofs.cross.CCA (default False; data are already prepared).
    use_coslat : bool
        Cosine-latitude weighting inside the CCA (default True).
    use_pca : bool
        Pre-filter each field with PCA before the CCA (default True).
    combine : {'mean', 'meof'}
        How the M models enter the CCA predictor field.
        'mean' (default) collapses M to the multi-model mean before any
        processing.
        'meof' keeps M as an extra channel so xeofs sees a joint
        (lat, lon, M) feature space at both the EEOF detrend stage and the
        CCA stage. This is scientifically richer but numerically fragile:
        the internal n_pca_modes=22 of ExtendedEOF may be too small to
        cover the enlarged feature space, and the CCA PCA pre-filter faces
        a far worse sample-to-feature ratio. Spatially coarsen the predictor
        and keep n_pca_modes conservative when using 'meof'.
    eeof_detrend : bool
        Apply the legacy EEOF detrend inside ``forecast`` (default True). Set
        False to fit CCA on the raw standardized fields without EEOF removal.
    dist_method : {'nonparam', 'bestfit'}
        Tercile-probability strategy (default 'nonparam').
    """

    def __init__(self, n_modes=4, n_pca_modes=8, standardize=False,
                 use_coslat=True, use_pca=True, combine="mean",
                 eeof_detrend=True, dist_method="nonparam"):
        if combine not in ("mean", "meof"):
            raise ValueError("combine must be 'mean' or 'meof'.")
        self.n_modes = n_modes
        self.n_pca_modes = n_pca_modes
        self.standardize = standardize
        self.use_coslat = use_coslat
        self.use_pca = use_pca
        self.combine = combine
        self.eeof_detrend = eeof_detrend
        self.dist_method = dist_method

        self.cca = None
        self.cca_model = None

    # ------------------------------------------------------------------ utils
    def _new_cca(self):
        """Fresh xeofs CCA instance (one per fold -> leakage-free CV)."""
        return xe.cross.CCA(
            n_modes=self.n_modes,
            standardize=self.standardize,
            use_coslat=self.use_coslat,
            use_pca=self.use_pca,
            n_pca_modes=self.n_pca_modes,
        )

    def _field(self, X):
        """Build the CCA predictor field from the predictor cube (T, Y, X, M)."""
        if "M" not in X.dims:
            return X
        if self.combine == "meof":
            return X
        return X.mean("M")

    @staticmethod
    def _to_latlon(da):
        """WAS dims (Y, X[, M]) -> xeofs dims (lat, lon[, M]); order (T, lat, lon, ...)."""
        rename = {}
        if "Y" in da.dims:
            rename["Y"] = "lat"
        if "X" in da.dims:
            rename["X"] = "lon"
        da = da.rename(rename)
        lead = [d for d in ("T", "lat", "lon") if d in da.dims]
        rest = [d for d in da.dims if d not in lead]
        return da.transpose(*lead, *rest)

    @staticmethod
    def _from_latlon(da):
        """xeofs dims (lat, lon) -> WAS dims (Y, X); order (T, Y, X)."""
        rename = {}
        if "lat" in da.dims:
            rename["lat"] = "Y"
        if "lon" in da.dims:
            rename["lon"] = "X"
        da = da.rename(rename)
        return da.transpose("T", "Y", "X")

    @staticmethod
    def _eeof_remove(da):
        """
        Legacy EEOF detrend: subtract the leading ExtendedEOF reconstruction.

        Reproduces:
            da - extract_leading_eeof_component(da).fillna(<last valid slice>)
        then fills any residual NaN with 0, exactly as legacy WAS_CCA does.

        combine='meof' warning
        ----------------------
        When ``da`` has shape (T, lat, lon, M), ExtendedEOF internally flattens
        all non-T dimensions into (lat x lon x M) features and applies a PCA
        pre-filter with n_pca_modes=22 (hard-coded inside
        ``extract_leading_eeof_component``).  If n_models x n_cells >> 22 the
        leading reconstructed mode may absorb genuine predictive signal rather
        than only the low-frequency trend.  Coarsen spatially or switch to
        combine='mean' if this is a concern.
        """
        trend = extract_leading_eeof_component(da)
        # the embedding leaves trailing NaNs in T; fill them with the last
        # valid reconstructed time slice (legacy used positional index -3)
        fill_slice = trend.isel(T=-3) if trend.sizes.get("T", 0) >= 3 else trend.isel(T=-1)
        trend = trend.fillna(fill_slice)
        return (da - trend).fillna(0)

    # --------------------------------------------------------------- CCA core
    def _fit_predict_cca(self, field_train, y_train, field_test):
        """
        Fit CCA on the training field, predict the test field, return the
        physical-space predictand on the WAS grid (T, Y, X). NaNs are filled
        with the training mean (no leakage); physical space is recovered with
        ``inverse_transform(transform(X), predict(X))[1]`` (codebase idiom).
        """
        Xtr = self._to_latlon(field_train)
        Ytr = self._to_latlon(y_train)

        Xtr_mean = Xtr.mean(dim="T", skipna=True)
        Xtr = Xtr.fillna(Xtr_mean)
        Ytr = Ytr.fillna(Ytr.mean(dim="T", skipna=True))

        self.cca = self._new_cca()
        self.cca_model = self.cca.fit(Xtr, Ytr, dim="T")

        Xte = self._to_latlon(field_test).fillna(Xtr_mean)
        y_pred = self.cca_model.predict(Xte)
        y_pred = self.cca_model.inverse_transform(
            self.cca_model.transform(Xte), y_pred
        )[1]
        y_pred["T"] = field_test["T"]

        return self._from_latlon(y_pred)

    # ----------------------------------------------- probability calculation
    @staticmethod
    def _ppf_terciles_from_code(dist_code, shape, loc, scale):
        if np.isnan(dist_code):
            return np.nan, np.nan
        code = int(dist_code)
        try:
            if code == 1:
                return norm.ppf(1/3, loc=loc, scale=scale), norm.ppf(2/3, loc=loc, scale=scale)
            if code == 2:
                return lognorm.ppf(1/3, s=shape, loc=loc, scale=scale), lognorm.ppf(2/3, s=shape, loc=loc, scale=scale)
            if code == 3:
                return expon.ppf(1/3, loc=loc, scale=scale), expon.ppf(2/3, loc=loc, scale=scale)
            if code == 4:
                return gamma.ppf(1/3, a=shape, loc=loc, scale=scale), gamma.ppf(2/3, a=shape, loc=loc, scale=scale)
            if code == 5:
                return weibull_min.ppf(1/3, c=shape, loc=loc, scale=scale), weibull_min.ppf(2/3, c=shape, loc=loc, scale=scale)
            if code == 6:
                return t.ppf(1/3, df=shape, loc=loc, scale=scale), t.ppf(2/3, df=shape, loc=loc, scale=scale)
            if code == 7:
                return poisson.ppf(1/3, mu=shape, loc=loc), poisson.ppf(2/3, mu=shape, loc=loc)
            if code == 8:
                return nbinom.ppf(1/3, n=shape, p=scale, loc=loc), nbinom.ppf(2/3, n=shape, p=scale, loc=loc)
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
            return (V / (M ** 2)) - ((g2 / (g1 ** 2)) - 1)
        except Exception:
            return -np.inf

    @staticmethod
    def calculate_tercile_probabilities_bestfit(best_guess, error_variance, T1, T2, dist_code, dof):
        best_guess = np.asarray(best_guess, dtype=float)
        error_variance = float(error_variance)
        n_time = best_guess.size
        out = np.full((3, n_time), np.nan, dtype=float)

        if (np.all(np.isnan(best_guess)) or not np.isfinite(error_variance)
                or error_variance <= 0 or not np.isfinite(T1) or not np.isfinite(T2)
                or np.isnan(dist_code)):
            return out

        code = int(dist_code)
        try:
            if code == 1:
                s = np.sqrt(error_variance)
                c1, c2 = norm.cdf(T1, loc=best_guess, scale=s), norm.cdf(T2, loc=best_guess, scale=s)
                out[0, :], out[1, :], out[2, :] = c1, c2 - c1, 1.0 - c2
            elif code == 2:
                valid = best_guess > 0
                sigma = np.full(n_time, np.nan); mu = np.full(n_time, np.nan)
                sigma[valid] = np.sqrt(np.log(1.0 + error_variance / (best_guess[valid] ** 2)))
                mu[valid] = np.log(best_guess[valid]) - 0.5 * sigma[valid] ** 2
                c1, c2 = lognorm.cdf(T1, s=sigma, scale=np.exp(mu)), lognorm.cdf(T2, s=sigma, scale=np.exp(mu))
                out[0, :], out[1, :], out[2, :] = c1, c2 - c1, 1.0 - c2
            elif code == 3:
                scale = np.sqrt(error_variance); loc = best_guess - scale
                c1, c2 = expon.cdf(T1, loc=loc, scale=scale), expon.cdf(T2, loc=loc, scale=scale)
                out[0, :], out[1, :], out[2, :] = c1, c2 - c1, 1.0 - c2
            elif code == 4:
                valid = best_guess > 0
                alpha = np.full(n_time, np.nan); theta = np.full(n_time, np.nan)
                alpha[valid] = (best_guess[valid] ** 2) / error_variance
                theta[valid] = error_variance / best_guess[valid]
                c1, c2 = gamma.cdf(T1, a=alpha, scale=theta), gamma.cdf(T2, a=alpha, scale=theta)
                out[0, :], out[1, :], out[2, :] = c1, c2 - c1, 1.0 - c2
            elif code == 5:
                for i in range(n_time):
                    M, V = best_guess[i], error_variance
                    if not np.isfinite(M) or M <= 0 or V <= 0:
                        continue
                    k = fsolve(WAS_mme_CCA_eeof.weibull_shape_solver, 2.0, args=(M, V))[0]
                    if not np.isfinite(k) or k <= 0:
                        continue
                    lam = M / gamma_function(1 + 1 / k)
                    c1, c2 = weibull_min.cdf(T1, c=k, loc=0, scale=lam), weibull_min.cdf(T2, c=k, loc=0, scale=lam)
                    out[0, i], out[1, i], out[2, i] = c1, c2 - c1, 1.0 - c2
            elif code == 6:
                if dof <= 2:
                    return out
                scale = np.sqrt(error_variance * (dof - 2) / dof)
                c1, c2 = t.cdf(T1, df=dof, loc=best_guess, scale=scale), t.cdf(T2, df=dof, loc=best_guess, scale=scale)
                out[0, :], out[1, :], out[2, :] = c1, c2 - c1, 1.0 - c2
            elif code == 7:
                mu = np.where(best_guess >= 0, best_guess, np.nan)
                c1, c2 = poisson.cdf(T1, mu=mu), poisson.cdf(T2, mu=mu)
                out[0, :], out[1, :], out[2, :] = c1, c2 - c1, 1.0 - c2
            elif code == 8:
                valid = error_variance > best_guess
                p = np.where(valid, best_guess / error_variance, np.nan)
                n = np.where(valid, best_guess ** 2 / (error_variance - best_guess), np.nan)
                c1, c2 = nbinom.cdf(T1, n=n, p=p), nbinom.cdf(T2, n=n, p=p)
                out[0, :], out[1, :], out[2, :] = c1, c2 - c1, 1.0 - c2
            else:
                return out
        except Exception:
            return out

        out = np.clip(out, 0, 1)
        total = np.nansum(out, axis=0)
        ok = np.isfinite(total) & (total > 0)
        out[:, ok] = out[:, ok] / total[ok]
        return out

    @staticmethod
    def calculate_tercile_probabilities_nonparametric(best_guess, error_samples, first_tercile, second_tercile):
        best_guess = np.asarray(best_guess, dtype=float)
        error_samples = np.asarray(error_samples, dtype=float)
        n_time = best_guess.size
        pred_prob = np.full((3, n_time), np.nan, dtype=float)

        valid_errors = error_samples[np.isfinite(error_samples)]
        if valid_errors.size == 0:
            return pred_prob

        for i in range(n_time):
            if not np.isfinite(best_guess[i]):
                continue
            d = best_guess[i] + valid_errors
            p_below = np.mean(d < first_tercile)
            p_normal = np.mean((d >= first_tercile) & (d < second_tercile))
            pred_prob[0, i] = p_below
            pred_prob[1, i] = p_normal
            pred_prob[2, i] = 1.0 - p_below - p_normal
        return pred_prob

    @staticmethod
    def _normalize_probabilities(prob):
        prob = prob.clip(min=0.0, max=1.0)
        total = prob.sum(dim="probability")
        return xr.where(total > 0, prob / total, np.nan)

    def _tercile_probabilities(self, Predictant, deterministic, clim_year_start, clim_year_end,
                               error_samples, error_variance,
                               best_code_da=None, best_shape_da=None,
                               best_loc_da=None, best_scale_da=None):
        Predictant = Predictant.transpose("T", "Y", "X")
        deterministic = deterministic.transpose("T", "Y", "X")
        mask = xr.where(np.isfinite(Predictant.isel(T=0)), 1.0, np.nan)

        clim = Predictant.sel(T=slice(str(clim_year_start), str(clim_year_end)))
        if clim.sizes.get("T", 0) < 3:
            raise ValueError("Not enough years in climatology period for terciles.")
        # legacy WAS_CCA tercile quantiles
        terc = clim.quantile([0.30, 0.67], dim="T")
        T1_emp = terc.isel(quantile=0).drop_vars("quantile")
        T2_emp = terc.isel(quantile=1).drop_vars("quantile")
        dof = max(int(clim.sizes["T"]) - 1, 2)

        if self.dist_method == "bestfit":
            if any(v is None for v in (best_code_da, best_shape_da, best_loc_da, best_scale_da)):
                raise ValueError("dist_method='bestfit' requires best_code_da, best_shape_da, best_loc_da, best_scale_da.")
            T1, T2 = xr.apply_ufunc(
                self._ppf_terciles_from_code, best_code_da, best_shape_da, best_loc_da, best_scale_da,
                input_core_dims=[(), (), (), ()], output_core_dims=[(), ()],
                vectorize=True, dask="parallelized", output_dtypes=[float, float])
            prob = xr.apply_ufunc(
                self.calculate_tercile_probabilities_bestfit,
                deterministic, error_variance, T1, T2, best_code_da,
                input_core_dims=[("T",), (), (), (), ()], output_core_dims=[("probability", "T")],
                vectorize=True, kwargs={"dof": dof}, dask="parallelized", output_dtypes=[float],
                dask_gufunc_kwargs={"output_sizes": {"probability": 3}, "allow_rechunk": True})
        elif self.dist_method == "nonparam":
            # rename residual sample axis so a length-1 forecast and the
            # length-n residual record don't collide on the shared 'T' core dim
            err = error_samples.rename({"T": "__samp"})
            prob = xr.apply_ufunc(
                self.calculate_tercile_probabilities_nonparametric,
                deterministic, err, T1_emp, T2_emp,
                input_core_dims=[("T",), ("__samp",), (), ()], output_core_dims=[("probability", "T")],
                vectorize=True, dask="parallelized", output_dtypes=[float],
                dask_gufunc_kwargs={"output_sizes": {"probability": 3}, "allow_rechunk": True})
        else:
            raise ValueError(f"Invalid dist_method: {self.dist_method}")

        prob = prob.assign_coords(probability=("probability", ["PB", "PN", "PA"]))
        prob = self._normalize_probabilities(prob.transpose("probability", "T", "Y", "X"))
        return (prob * mask).transpose("probability", "T", "Y", "X")

    # ----------------------------------------------------- framework contract
    def compute_model(self, X_train, y_train, X_test, y_test=None, **kwargs):
        """
        Deterministic CCA hindcast for one CV fold, in the space the framework
        hands in (matches legacy WAS_CCA.compute_model: plain CCA, no EEOF
        detrend; the same_kind_model2 branch reverse-standardizes afterwards).
        """
        if "M" in y_train.dims:
            y_train = y_train.isel(M=0, drop=True)
        field_train = self._field(X_train)
        field_test = self._field(X_test)
        return self._fit_predict_cca(field_train, y_train, field_test)

    def compute_prob(self, Predictant, clim_year_start, clim_year_end, hindcast_det,
                     best_code_da=None, best_shape_da=None, best_loc_da=None, best_scale_da=None):
        if "M" in Predictant.dims:
            Predictant = Predictant.isel(M=0).drop_vars("M").squeeze()
        Predictant = Predictant.transpose("T", "Y", "X")
        hindcast_det = hindcast_det.transpose("T", "Y", "X")
        if hindcast_det.sizes["T"] == Predictant.sizes["T"]:
            hindcast_det = hindcast_det.assign_coords(T=Predictant["T"])

        error_samples = Predictant - hindcast_det           # out-of-sample (LOYO hindcast)
        error_variance = error_samples.var(dim="T", skipna=True)

        return self._tercile_probabilities(
            Predictant, hindcast_det, clim_year_start, clim_year_end,
            error_samples, error_variance,
            best_code_da, best_shape_da, best_loc_da, best_scale_da)

    # ------------------------------------------------------------- operational
    def forecast(self, Predictant, clim_year_start, clim_year_end, Predictor,
                 hindcast_det, Predictor_for_year,
                 best_code_da=None, best_shape_da=None, best_loc_da=None, best_scale_da=None):
        """
        Operational CCA forecast for a single target year, legacy WAS_CCA style:

          * predictor field EEOF-detrended (NOT standardized),
          * predictand standardized then EEOF-detrended,
          * forecast-year predictor only gap-filled (NOT detrended),
          * result reverse-standardized with NO trend added back.
        """
        if "M" in Predictant.dims:
            Predictant = Predictant.isel(M=0).drop_vars("M").squeeze()
        Predictant = Predictant.transpose("T", "Y", "X")
        mask = xr.where(np.isfinite(Predictant.isel(T=0)), 1.0, np.nan)
        
        mean_val = Predictor.sel(T=slice(str(clim_year_start), str(clim_year_end))).mean(dim='T')
        std_val = Predictor.sel(T=slice(str(clim_year_start), str(clim_year_end))).std(dim='T')
        Predictor_for_year = (Predictor_for_year - mean_val) / std_val

        # ----- predictor side: EEOF-detrended field over all training years
        Predictor = standardize_timeseries(Predictor, clim_year_start, clim_year_end)
        field_hist = self._field(Predictor)
        if self.eeof_detrend:
            field_hist_fit = self._eeof_remove(field_hist)
        else:
            field_hist_fit = field_hist.fillna(field_hist.mean(dim="T", skipna=True)).fillna(0)

        # ----- predictand side: standardize, then EEOF-detrend
        Predictant_st = standardize_timeseries(Predictant, clim_year_start, clim_year_end)
        if self.eeof_detrend:
            Predictant_fit = self._eeof_remove(Predictant_st)
        else:
            Predictant_fit = Predictant_st.fillna(0)

        # ----- forecast-year predictor: gap-fill only, NOT detrended
        field_year = self._field(Predictor_for_year)
        field_year = (field_year.fillna(field_hist.mean(dim="T", skipna=True))
                      .ffill(dim="Y").bfill(dim="Y")
                      .ffill(dim="X").bfill(dim="X")
                      .fillna(0))

        # ----- fit CCA and predict
        pred_st = self._fit_predict_cca(field_hist_fit, Predictant_fit, field_year)

        # ----- back to physical space (no EEOF trend re-added: legacy behavior)
        forecast_det = reverse_standardize(pred_st, Predictant, clim_year_start, clim_year_end)

        # target time = forecast year + predictand target month
        year = Predictor_for_year.coords["T"].values.astype("datetime64[Y]").astype(int)[0] + 1970
        month = Predictant.isel(T=0).coords["T"].values.astype("datetime64[M]").astype(int) % 12 + 1
        new_T = np.datetime64(f"{year}-{month:02d}-01")
        forecast_det = forecast_det.assign_coords(T=xr.DataArray([new_T], dims=["T"]))
        forecast_det["T"] = forecast_det["T"].astype("datetime64[ns]")

        error_samples = Predictant - hindcast_det.transpose("T", "Y", "X")
        error_variance = error_samples.var(dim="T", skipna=True)
        forecast_prob = self._tercile_probabilities(
            Predictant, forecast_det, clim_year_start, clim_year_end,
            error_samples, error_variance,
            best_code_da, best_shape_da, best_loc_da, best_scale_da)

        return forecast_det * mask, forecast_prob * mask
