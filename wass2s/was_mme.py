from __future__ import annotations

import operator
import random
import gc
import datetime
import warnings
from dataclasses import dataclass
from typing import Literal, Tuple, Optional, Dict, List, Sequence, Callable




import numpy as np
import pandas as pd
import xarray as xr


from scipy import stats
from scipy.optimize import minimize, minimize_scalar, fsolve, root_scalar, brentq
from scipy.special import gamma as gamma_function
from scipy.special import expit, softmax
from scipy.stats import (
    norm, lognorm, expon, gamma, weibull_min, t, poisson, nbinom,
    logistic, genextreme, laplace, pareto, loguniform, randint, uniform,
    linregress, t as tdist, gamma as sp_gamma, gamma as gamma_dist, boxcox_normmax
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
from sklearn.model_selection import GridSearchCV, RandomizedSearchCV, KFold, TimeSeriesSplit, cross_val_score
from sklearn.metrics import mean_squared_error, log_loss, make_scorer
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


# full-Bayesian backend
try:
    import pymc as pm
    import arviz as az
    HAS_PYMC = True
except Exception:
    HAS_PYMC = False
    pm = None
    az = None
    
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
    Min et al. (2009) Probability-Weighted Multi-Model Ensemble (PMME).

    What this class does
    --------------------
    1) For each model:
       - compute tercile probabilities (PB/PN/PA) at each gridpoint
         from its ensemble forecast using:
           * 'gaussian'  : Gaussian approximation (good for temperature anomalies)
           * 'lognormal' : log-normal approximation (better for precip amounts)
           * 'empirical' : hindcast terciles (q33/q66) + deterministic 0/1 probs
    2) Combine model probabilities with weights:
         w_m ∝ sqrt(Nm)  (Nm = ensemble size), then normalized to sum to 1
    3) Optionally compute a combined categorical map with a chi-square significance test.

    Assumed dims
    ------------
    Each model DataArray should have at least:
      - forecasts[m]: dims include ('M',) and optionally 'time'
      - hindcasts[m]: dims include ('T','M')
    Spatial dims can be ('lat','lon') or ('Y','X') or anything; the code is generic.

    Notes
    -----
    - For temperature terciles, Min et al. uses ±0.4307*σ around climatological mean (Gaussian terciles).
    - For precipitation, 'lognormal' or a proper Gamma/empirical approach is usually preferable.
    """

    def __init__(
        self,
        distribution: Literal["gaussian", "lognormal", "empirical"] = "gaussian",
        cv_method: Optional[Literal[None, "leave_one_out", "rolling_window"]] = None,
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

    # ---------------------------------------------------------------------
    # internal helpers
    # ---------------------------------------------------------------------
    @staticmethod
    def _require_dims(da: xr.DataArray, required: Tuple[str, ...], name: str = "DataArray") -> None:
        missing = [d for d in required if d not in da.dims]
        if missing:
            raise ValueError(f"{name}: missing required dims {missing}. Got dims={da.dims}")

    @staticmethod
    def _norm_cdf(z: xr.DataArray) -> xr.DataArray:
        return xr.apply_ufunc(stats.norm.cdf, z)

    @staticmethod
    def _safe_clip_and_renorm(p_bn: xr.DataArray, p_nn: xr.DataArray, p_an: xr.DataArray) -> Tuple[xr.DataArray, xr.DataArray, xr.DataArray]:
        p_bn = p_bn.clip(0.0, 1.0)
        p_nn = p_nn.clip(0.0, 1.0)
        p_an = p_an.clip(0.0, 1.0)

        total = p_bn + p_nn + p_an
        ok = xr.ufuncs.isfinite(total) & (total > 0.0)

        p_bn = xr.where(ok, p_bn / total, np.nan)
        p_nn = xr.where(ok, p_nn / total, np.nan)
        p_an = xr.where(ok, p_an / total, np.nan)
        return p_bn, p_nn, p_an

    def _compute_model_weights(self, ensemble_sizes: Dict[str, int]) -> Dict[str, float]:
        sqrt_sizes = {m: np.sqrt(float(n)) for m, n in ensemble_sizes.items()}
        tot = float(sum(sqrt_sizes.values()))
        if tot <= 0:
            raise ValueError("Sum of sqrt(ensemble_sizes) must be > 0.")
        return {m: sqrt_sizes[m] / tot for m in ensemble_sizes}

    def _compute_n_for_chisq(self, ensemble_sizes: Dict[str, int], model_names: List[str]) -> float:
        total_ensemble = float(sum(float(ensemble_sizes[m]) for m in model_names))
        if isinstance(self.n_samples_for_chisq, (int, float)):
            return float(self.n_samples_for_chisq)
        if self.n_samples_for_chisq == "effective_sample_size":
            k = max(1, len(model_names))
            return total_ensemble / np.sqrt(float(k))
        return total_ensemble

    # ---------------------------------------------------------------------
    # cross-validated stats
    # ---------------------------------------------------------------------
    def _compute_cross_validated_stats(self, hindcasts: xr.DataArray) -> Tuple[xr.DataArray, xr.DataArray]:
        """
        Return (mu, sigma). If cv_method is None -> spatial-only stats.
        If cv_method in {leave_one_out, rolling_window} -> returns stats with a 'time' dim.
        """
        self._require_dims(hindcasts, ("T", "M"), name="hindcasts")

        if self.cv_method is None:
            mu = hindcasts.mean(dim=("T", "M"))
            sigma = hindcasts.std(dim=("T", "M"))
            return mu, sigma

        n_times = hindcasts.sizes["T"]

        if self.cv_method == "leave_one_out":
            mu_list, sig_list = [], []
            for i in range(n_times):
                h_train = hindcasts.isel(T=[j for j in range(n_times) if j != i])
                mu_list.append(h_train.mean(dim=("T", "M")))
                sig_list.append(h_train.std(dim=("T", "M")))
            mu = xr.concat(mu_list, dim=hindcasts["T"])
            sigma = xr.concat(sig_list, dim=hindcasts["T"])
            return mu, sigma

        if self.cv_method == "rolling_window":
            w = self.rolling_window_size
            mu_list, sig_list = [], []
            for i in range(n_times):
                start = max(0, i - w // 2)
                end = min(n_times, i + w // 2 + 1)
                h_win = hindcasts.isel(T=slice(start, end))
                local_i = i - start
                if 0 <= local_i < h_win.sizes["T"]:
                    h_train = h_win.isel(T=[j for j in range(h_win.sizes["T"]) if j != local_i])
                else:
                    h_train = h_win
                mu_list.append(h_train.mean(dim=("T", "M")))
                sig_list.append(h_train.std(dim=("T", "M")))
            mu = xr.concat(mu_list, dim=hindcasts["T"])
            sigma = xr.concat(sig_list, dim=hindcasts["T"])
            return mu, sigma

        raise ValueError(f"Unknown cv_method={self.cv_method!r}")

    def _align_or_fallback_stats(
        self,
        mu: xr.DataArray,
        sigma: xr.DataArray,
        f_mean: xr.DataArray,
        hindcasts: xr.DataArray,
    ) -> Tuple[xr.DataArray, xr.DataArray]:
        """
        Ensure mu/sigma align with f_mean without producing time:0 outputs.
        If mu has time and forecast time doesn't overlap, fallback to spatial-only stats.
        """
        # clamp sigma early
        sigma = xr.where(sigma <= self.sigma_floor, np.nan, sigma)

        if "T" in mu.dims and "T" in f_mean.dims:
            overlap = np.intersect1d(mu["T"].values, f_mean["T"].values)
            if overlap.size == 0:
                # operational forecast date, not in hindcast -> use full hindcast stats
                mu = hindcasts.mean(dim=("T", "M"))
                sigma = hindcasts.std(dim=("T", "M"))
                sigma = xr.where(sigma <= self.sigma_floor, np.nan, sigma)

        mu = mu.broadcast_like(f_mean)
        sigma = sigma.broadcast_like(f_mean)
        return mu, sigma

    # ---------------------------------------------------------------------
    # probabilities for one model
    # ---------------------------------------------------------------------
    def _compute_tercile_probabilities_one_model(
        self,
        forecasts: xr.DataArray,
        hindcasts: xr.DataArray,
    ) -> Tuple[xr.DataArray, xr.DataArray, xr.DataArray]:
        """
        Compute BN/NN/AN for a single model, returning three DataArrays
        aligned to the forecast mean (and its time, if present).
        """
        self._require_dims(forecasts, ("M",), name="forecasts")
        self._require_dims(hindcasts, ("T", "M"), name="hindcasts")

        f_mean = forecasts.mean(dim="M")
        lower_k, upper_k = -0.4307, 0.4307

        if self.distribution == "gaussian":
            mu, sigma = self._compute_cross_validated_stats(hindcasts)
            mu, sigma = self._align_or_fallback_stats(mu, sigma, f_mean, hindcasts)

            z_lower = (mu + lower_k * sigma - f_mean) / sigma
            z_upper = (mu + upper_k * sigma - f_mean) / sigma

            p_bn = self._norm_cdf(z_lower)
            p_an = 1.0 - self._norm_cdf(z_upper)
            p_nn = 1.0 - p_bn - p_an
            return self._safe_clip_and_renorm(p_bn, p_nn, p_an)

        if self.distribution == "lognormal":
            eps = self.eps_lognormal

            h_pos = xr.where(hindcasts > 0, hindcasts, eps)
            f_pos = xr.where(forecasts > 0, forecasts, eps)

            log_h = np.log(h_pos)
            log_f_mean = np.log(f_pos).mean(dim="M")

            mu, sigma = self._compute_cross_validated_stats(log_h)
            mu, sigma = self._align_or_fallback_stats(mu, sigma, log_f_mean, log_h)

            z_lower = (mu + lower_k * sigma - log_f_mean) / sigma
            z_upper = (mu + upper_k * sigma - log_f_mean) / sigma

            p_bn = self._norm_cdf(z_lower)
            p_an = 1.0 - self._norm_cdf(z_upper)
            p_nn = 1.0 - p_bn - p_an
            return self._safe_clip_and_renorm(p_bn, p_nn, p_an)

        if self.distribution == "empirical":
            # empirical terciles from hindcasts
            q33 = hindcasts.quantile(1 / 3, dim=("T", "M"))
            q66 = hindcasts.quantile(2 / 3, dim=("T", "M"))

            q33 = q33.broadcast_like(f_mean)
            q66 = q66.broadcast_like(f_mean)

            p_bn = xr.where(f_mean < q33, 1.0, 0.0)
            p_an = xr.where(f_mean > q66, 1.0, 0.0)
            p_nn = 1.0 - p_bn - p_an
            return self._safe_clip_and_renorm(p_bn, p_nn, p_an)

        raise ValueError(f"Unknown distribution={self.distribution!r}")

    # ---------------------------------------------------------------------
    # public API
    # ---------------------------------------------------------------------
    def compute_pmme_probabilities(
        self,
        forecasts: Dict[str, xr.DataArray],
        hindcasts: Dict[str, xr.DataArray],
        climatology: Optional[xr.DataArray] = None,  # placeholder for your pipeline
        ensemble_sizes: Optional[Dict[str, int]] = None,
        strict_models: bool = True,
    ) -> Dict[str, xr.DataArray]:
        """
        Compute PMME probabilities (PB/PN/PA) across models.

        Returns
        -------
        pmme_probs : dict with keys 'PB','PN','PA' (DataArray maps)
        """
        if strict_models and set(forecasts.keys()) != set(hindcasts.keys()):
            raise ValueError("forecasts and hindcasts must have the same model keys when strict_models=True.")

        model_names = [m for m in forecasts.keys() if m in hindcasts]
        if not model_names:
            raise ValueError("No common models between forecasts and hindcasts.")

        if ensemble_sizes is None:
            ensemble_sizes = {m: int(forecasts[m].sizes["M"]) for m in model_names}

        weights = self._compute_model_weights(ensemble_sizes)

        # compute per-model probs
        per_model = {}
        for m in model_names:
            p_bn, p_nn, p_an = self._compute_tercile_probabilities_one_model(forecasts[m], hindcasts[m])
            per_model[m] = {"PB": p_bn, "PN": p_nn, "PA": p_an}

        # weighted sum
        template = per_model[model_names[0]]["PB"]
        pmme_probs = {}
        for cat in ("PB", "PN", "PA"):
            wsum = xr.zeros_like(template)
            for m in model_names:
                wsum = wsum + per_model[m][cat] * float(weights[m])
            pmme_probs[cat] = wsum

        # (optional) ensure valid probs
        pmme_probs["PB"], pmme_probs["PN"], pmme_probs["PA"] = self._safe_clip_and_renorm(
            pmme_probs["PB"], pmme_probs["PN"], pmme_probs["PA"]
        )
        return pmme_probs

    def compute_combined_map(
        self,
        pmme_probs: Dict[str, xr.DataArray],
        ensemble_sizes: Dict[str, int],
        model_names: List[str],
        significance_level: float = 0.05,
    ) -> Tuple[xr.DataArray, xr.DataArray]:
        """
        Combined categorical map with chi-square significance test.

        Returns
        -------
        combined_map : DataArray
            values: 0=no significant deviation, 1=PB, 2=PN, 3=PA
        chi_square : DataArray
            chi-square statistic
        """
        for k in ("PB", "PN", "PA"):
            if k not in pmme_probs:
                raise ValueError(f"pmme_probs missing key {k!r}")

        # stack probs
        probs_array = xr.concat(
            [pmme_probs["PB"], pmme_probs["PN"], pmme_probs["PA"]],
            dim=xr.DataArray(["PB", "PN", "PA"], dims="probability", name="probability"),
        )

        # valid pixels = at least one category not null
        valid_any = probs_array.notnull().any("probability")

        # SAFE argmax: fill NaNs with -inf just for argmax
        dominant = probs_array.fillna(-np.inf).argmax(dim="probability", skipna=False)

        # chi-square
        n = self._compute_n_for_chisq(ensemble_sizes, model_names)
        expected = 1.0 / 3.0

        chi_square = n * (
            (pmme_probs["PB"] - expected) ** 2 / expected
            + (pmme_probs["PN"] - expected) ** 2 / expected
            + (pmme_probs["PA"] - expected) ** 2 / expected
        )

        critical = float(stats.chi2.ppf(1.0 - float(significance_level), df=2))

        combined = xr.where(valid_any & (chi_square > critical), dominant + 1, 0)

        combined.attrs = {
            "description": "PMME combined forecast map (Min et al. 2009 probability-weighted)",
            "values": "0=no significant deviation, 1=PB, 2=PN, 3=PA",
            "significance_level": float(significance_level),
            "chi2_critical_value": critical,
        }
        chi_square.attrs = {
            "description": "Chi-square statistic for PMME categorical significance",
            "df": 2,
            "n_samples_used": float(n),
        }
        return combined, chi_square



# class WAS_Min2009_ProbWeighted:
#     """
#     Implementation of Min et al. (2009) Probability-Weighted Multi-Model Ensemble.
    
#     Based on: "Probabilistic Multimodel Ensemble (PMME) forecasting at APCC"
#     Journal: Weather and Forecasting, 2009
    
#     Key methodology:
#     1. Individual model probabilistic forecasts are computed using Gaussian approximation
#     2. Model weights are proportional to sqrt(ensemble_size) [Eq. 6 in paper]
#     3. Tercile probabilities (BN, NN, AN) are combined using total probability formula
    
#     Notes:
#     - For temperature: Gaussian assumption is reasonable
#     - For precipitation: Consider using log-normal, Gamma distribution, or empirical methods
#     - The χ² test uses configurable n (ensemble size) - avoid overly conservative n=1
#     - Cross-validation is recommended for hindcast statistics
#     """
    
#     def __init__(self, distribution='gaussian', cv_method=None, n_samples_for_chisq='total_ensemble'):
#         """
#         Initialize PMME processor.
        
#         Parameters
#         ----------
#         distribution : str
#             Distribution assumption: 'gaussian', 'gamma', 'lognormal', or 'empirical'
#         cv_method : str or None
#             Cross-validation method: None, 'leave_one_out', or 'rolling_window'
#         n_samples_for_chisq : str or int
#             How to compute n for χ² test: 
#             'total_ensemble' (default), 'effective_sample_size', or integer value
#         """
#         self.distribution = distribution
#         self.cv_method = cv_method
#         self.n_samples_for_chisq = n_samples_for_chisq
        
#     def _compute_cross_validated_stats(self, hindcasts, climatology):
#         """
#         Compute cross-validated mean and standard deviation from hindcasts.
        
#         Parameters
#         ----------
#         hindcasts : xarray.DataArray
#             Hindcast ensemble data with dimensions (T, M, Y, X)
#         climatology : xarray.DataArray
#             Climatological data with dimensions (Y, X)
            
#         Returns
#         -------
#         mu_cv : xarray.DataArray
#             Cross-validated mean with dimensions (T, Y, X)
#         sigma_cv : xarray.DataArray
#             Cross-validated standard deviation with dimensions (T, Y, X)
#         """
#         n_times = hindcasts.sizes['time']
        
#         if self.cv_method is None:
#             # Use full hindcast period (not recommended for operational use)
#             mu_cv = hindcasts.mean(dim=['time', 'ensemble'])
#             sigma_cv = hindcasts.std(dim=['time', 'ensemble'])
#             # Expand to match time dimension
#             mu_cv = mu_cv.expand_dims(time=hindcasts.time).transpose('time', 'lat', 'lon')
#             sigma_cv = sigma_cv.expand_dims(time=hindcasts.time).transpose('time', 'lat', 'lon')
            
#         elif self.cv_method == 'leave_one_out':
#             # Leave-one-out cross-validation
#             mu_list = []
#             sigma_list = []
            
#             for i in range(n_times):
#                 # Leave out year i
#                 hindcast_train = hindcasts.isel(time=[j for j in range(n_times) if j != i])
#                 mu_i = hindcast_train.mean(dim=['time', 'ensemble'])
#                 sigma_i = hindcast_train.std(dim=['time', 'ensemble'])
#                 mu_list.append(mu_i)
#                 sigma_list.append(sigma_i)
            
#             mu_cv = xr.concat(mu_list, dim=hindcasts.time)
#             sigma_cv = xr.concat(sigma_list, dim=hindcasts.time)
            
#         elif self.cv_method == 'rolling_window':
#             # Rolling window validation (e.g., 15-year window)
#             window_size = 15
#             mu_list = []
#             sigma_list = []
            
#             for i in range(n_times):
#                 start = max(0, i - window_size // 2)
#                 end = min(n_times, i + window_size // 2 + 1)
#                 hindcast_train = hindcasts.isel(time=slice(start, end))
#                 # Exclude the current year if possible
#                 hindcast_train = hindcast_train.isel(time=[j for j in range(hindcast_train.sizes['time']) 
#                                                           if start + j != i])
                
#                 mu_i = hindcast_train.mean(dim=['time', 'ensemble'])
#                 sigma_i = hindcast_train.std(dim=['time', 'ensemble'])
#                 mu_list.append(mu_i)
#                 sigma_list.append(sigma_i)
            
#             mu_cv = xr.concat(mu_list, dim=hindcasts.time)
#             sigma_cv = xr.concat(sigma_list, dim=hindcasts.time)
        
#         return mu_cv, sigma_cv
    
#     def _compute_tercile_probabilities(self, forecasts, hindcasts, climatology):
#         """
#         Compute tercile probabilities for individual models.
        
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
#         # Compute cross-validated statistics
#         mu_cv, sigma_cv = self._compute_cross_validated_stats(hindcasts, climatology)
        
#         # Compute forecast ensemble mean
#         forecast_mean = forecasts.mean(dim='ensemble')
        
#         if self.distribution == 'gaussian':
#             # Gaussian approximation (suitable for temperature)
#             # Tercile boundaries at approximately ±0.43σ (for Gaussian)
#             # Actually, for terciles, boundaries are at Φ⁻¹(1/3) ≈ -0.43 and Φ⁻¹(2/3) ≈ 0.43
#             # The paper uses ±1.43σ which seems incorrect - this would be for much wider intervals
#             # Let's use the correct tercile boundaries:
#             lower_boundary = -0.4307  # Φ⁻¹(1/3)
#             upper_boundary = 0.4307   # Φ⁻¹(2/3)
            
#             # Standardized anomalies
#             z_lower = (mu_cv + lower_boundary * sigma_cv - forecast_mean) / sigma_cv
#             z_upper = (mu_cv + upper_boundary * sigma_cv - forecast_mean) / sigma_cv
            
#             # Gaussian CDF probabilities
#             probs_bn = stats.norm.cdf(z_lower)
#             probs_an = 1 - stats.norm.cdf(z_upper)
#             probs_nn = 1 - probs_bn - probs_an
            
#         elif self.distribution == 'lognormal':
#             # Log-normal distribution (suitable for precipitation)
#             # Transform to log space
#             log_hindcasts = xr.where(hindcasts > 0, np.log(hindcasts), np.log(0.01))
#             log_mu_cv, log_sigma_cv = self._compute_cross_validated_stats(log_hindcasts, climatology)
#             log_forecast = xr.where(forecasts > 0, np.log(forecasts), np.log(0.01))
#             log_forecast_mean = log_forecast.mean(dim='ensemble')
            
#             # Compute tercile boundaries in log space
#             lower_boundary = -0.4307
#             upper_boundary = 0.4307
            
#             z_lower = (log_mu_cv + lower_boundary * log_sigma_cv - log_forecast_mean) / log_sigma_cv
#             z_upper = (log_mu_cv + upper_boundary * log_sigma_cv - log_forecast_mean) / log_sigma_cv
            
#             probs_bn = stats.norm.cdf(z_lower)
#             probs_an = 1 - stats.norm.cdf(z_upper)
#             probs_nn = 1 - probs_bn - probs_an
            
#         elif self.distribution == 'empirical':
#             # Empirical quantile mapping
#             # This is a simplified version - consider more sophisticated methods
#             forecast_flat = forecasts.stack(sample=('time', 'ensemble')).transpose('sample', 'lat', 'lon')
#             hindcast_flat = hindcasts.stack(sample=('time', 'ensemble')).transpose('sample', 'lat', 'lon')
            
#             # Compute empirical CDF for each grid point
#             probs_bn = xr.full_like(forecast_mean, fill_value=np.nan)
#             probs_nn = xr.full_like(forecast_mean, fill_value=np.nan)
#             probs_an = xr.full_like(forecast_mean, fill_value=np.nan)
            
#             # This is computationally intensive - consider optimization
#             for lat in forecast_mean.lat.values:
#                 for lon in forecast_mean.lon.values:
#                     hindcast_vals = hindcast_flat.sel(lat=lat, lon=lon).values
#                     forecast_val = forecast_mean.sel(lat=lat, lon=lon).values
                    
#                     # Compute empirical terciles from hindcast
#                     lower_tercile = np.percentile(hindcast_vals, 100/3)
#                     upper_tercile = np.percentile(hindcast_vals, 100*2/3)
                    
#                     # Empirical probabilities
#                     probs_bn.loc[dict(lat=lat, lon=lon)] = np.mean(forecast_val < lower_tercile)
#                     probs_an.loc[dict(lat=lat, lon=lon)] = np.mean(forecast_val > upper_tercile)
#                     probs_nn.loc[dict(lat=lat, lon=lon)] = 1 - probs_bn.loc[dict(lat=lat, lon=lon)] - probs_an.loc[dict(lat=lat, lon=lon)]
        
#         # Ensure probabilities are between 0 and 1
#         probs_bn = xr.where(probs_bn < 0, 0, xr.where(probs_bn > 1, 1, probs_bn))
#         probs_an = xr.where(probs_an < 0, 0, xr.where(probs_an > 1, 1, probs_an))
#         probs_nn = xr.where(probs_nn < 0, 0, xr.where(probs_nn > 1, 1, probs_nn))
        
#         # Renormalize to ensure sum to 1 (accounting for numerical errors)
#         total = probs_bn + probs_nn + probs_an
#         probs_bn = probs_bn / total
#         probs_nn = probs_nn / total
#         probs_an = probs_an / total
        
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
    
#     def _compute_n_for_chisq(self, ensemble_sizes, model_names):
#         """
#         Compute n for χ² test based on configuration.
        
#         Parameters
#         ----------
#         ensemble_sizes : dict
#             Dictionary mapping model names to ensemble sizes
#         model_names : list
#             List of model names
            
#         Returns
#         -------
#         n : float
#             Value to use for n in χ² test
#         """
#         if isinstance(self.n_samples_for_chisq, (int, float)):
#             return float(self.n_samples_for_chisq)
#         elif self.n_samples_for_chisq == 'total_ensemble':
#             # Sum of all ensemble members across models
#             return sum(ensemble_sizes[model] for model in model_names)
#         elif self.n_samples_for_chisq == 'effective_sample_size':
#             # Approximate effective sample size
#             total_ensemble = sum(ensemble_sizes[model] for model in model_names)
#             n_models = len(model_names)
#             # Simple approximation: account for correlation between models
#             return total_ensemble / np.sqrt(n_models)
#         else:
#             # Default to total ensemble size
#             return sum(ensemble_sizes[model] for model in model_names)
    
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
#             weighted_sum = xr.zeros_like(next(iter(model_probs.values()))[category])
#             for model in model_names:
#                 weighted_prob = model_probs[model][category] * weights[model]
#                 weighted_sum = weighted_sum + weighted_prob
            
#             pmme_probs[category] = weighted_sum
        
#         return pmme_probs
    
#     def compute_combined_map(self, pmme_probs, ensemble_sizes, model_names, significance_level=0.05):
#         """
#         Compute combined map with significance testing (χ² test).
        
#         Parameters
#         ----------
#         pmme_probs : dict
#             PMME probabilities for BN, NN, AN categories
#         ensemble_sizes : dict
#             Dictionary mapping model names to ensemble sizes
#         model_names : list
#             List of model names
#         significance_level : float
#             Significance level for χ² test (default 0.05)
            
#         Returns
#         -------
#         combined_map : xarray.DataArray
#             Combined map showing dominant category where significant
#         chi_square : xarray.DataArray
#             χ² statistic values
#         """
#         # Find dominant category
#         probs_array = xr.concat([pmme_probs['BN'], pmme_probs['NN'], pmme_probs['AN']], 
#                                dim='category')
#         dominant_category = probs_array.argmax(dim='category')
        
#         # Compute n for χ² test
#         n = self._compute_n_for_chisq(ensemble_sizes, model_names)
        
#         # Compute χ² statistic (Eq. in section 5)
#         # χ² = n * Σ (P(Ej) - 1/3)² / (1/3)
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
        
#         # Add attributes for interpretation
#         combined_map.attrs = {
#             'description': 'PMME combined forecast map',
#             'values': '0=no significant deviation, 1=BN, 2=NN, 3=AN',
#             'significance_level': significance_level,
#             'chi2_critical_value': critical_value
#         }
        
#         return combined_map, chi_square


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
        # Set global seeds for reproducibility
        np.random.seed(self.random_state)
        random.seed(self.random_state)
        
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

            # Create reproducible random weights
            rng = np.random.RandomState(self.random_state)
            W = rng.randn(X_train_clean_c.shape[1], bp['neurons'])
            B = rng.randn(bp['neurons'])
            hpelm_c.add_neurons(bp['neurons'], bp['activation'], W=W, B=B)
            # hpelm_c.add_neurons(bp['neurons'], bp['activation'])
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

            # Create reproducible random weights
            rng = np.random.RandomState(self.random_state)  # Add cluster for variety
            W = rng.randn(X_train_clean_c.shape[1], bp['neurons'])
            B = rng.randn(bp['neurons'])
            hpelm_c.add_neurons(bp['neurons'], bp['activation'], W=W, B=B)
            
            # hpelm_c.add_neurons(bp['neurons'], bp['activation'])
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

        # Set global seeds for reproducibility
        np.random.seed(self.random_state)
        random.seed(self.random_state)
        
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
        y_train_stacked = y_train.stack(sample=('T', 'Y', 'X')).values.ravel()
        # y_train_stacked = y_train.stack(sample=('T', 'Y', 'X')).transpose('sample', 'M').values
        train_nan_mask = np.any(~np.isfinite(X_train_stacked), axis=1) | ~np.isfinite(y_train_stacked)
        X_train_clean = X_train_stacked[~train_nan_mask]
        y_train_clean = y_train_stacked[~train_nan_mask]

        # Stack testing data
        X_test_stacked = X_test.stack(sample=('T', 'Y', 'X')).transpose('sample', 'M').values
        y_test_stacked = y_test.stack(sample=('T', 'Y', 'X')).values.ravel()  # Flatten to 1D
        test_nan_mask = np.any(~np.isfinite(X_test_stacked), axis=1) | ~np.isfinite(y_test_stacked)

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
       
        # Create reproducible random weights
        rng = np.random.RandomState(self.random_state)  # Add cluster for variety
        W = rng.randn(X_train_clean.shape[1], best_params['neurons'])
        B = rng.randn(best_params['neurons'])
        self.hpelm.add_neurons(best_params['neurons'], best_params['activation'], W=W, B=B)        

        # self.hpelm.add_neurons(best_params['neurons'], best_params['activation'])
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

        # Create reproducible random weights
        rng = np.random.RandomState(self.random_state)  # Add cluster for variety
        W = rng.randn(X_train_clean.shape[1], best_params['neurons'])
        B = rng.randn(best_params['neurons'])
        self.hpelm.add_neurons(best_params['neurons'], best_params['activation'], W=W, B=B) 
        
        # self.hpelm.add_neurons(best_params['neurons'], best_params['activation'])
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

## Je peux add mean over M et std over M in hindcast. ça aidera mon MLP et ELM

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
    ### add early stopping here: early_stopping_rounds=50
    """
    XGBoost-based Multi-Model Ensemble (MME) forecasting for seasonal rainfall.
    Enhanced with additional regularization hyperparameters for climate forecasting.
    
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
    optuna_n_jobs : int, optional
        Number of parallel jobs for Optuna (default is 1).
    optuna_timeout : int, optional
        Timeout in seconds for Optuna optimization (default is None).
    """
    def __init__(self,
                 search_method='random',
                 n_estimators_range=[50, 100, 150, 200, 250],  # Reduced upper bound
                 learning_rate_range=[0.005, 0.01, 0.03, 0.05, 0.1],  # Finer low-end
                 max_depth_range=[3, 5, 7],  # More conservative for climate data
                 min_child_weight_range=[1, 3, 5],
                 subsample_range=[0.6, 0.8, 1.0],
                 colsample_bytree_range=[0.6, 0.8, 1.0],
                 gamma_range=[0, 0.1, 0.2, 0.5, 1],  #  Gamma regularization
                 reg_alpha_range=[0, 0.01, 0.1],  # L1 regularization
                 reg_lambda_range=[1, 1.5, 2],  # ADDED: L2 regularization
                 random_state=42,
                 dist_method="nonparam",
                 n_iter_search=10,
                 cv_folds=3,
                 cv_method='timeseries',  #  CV method for temporal data or None
                 optuna_n_jobs=1,
                 optuna_timeout=None):
        
        self.search_method = search_method
        self.n_estimators_range = n_estimators_range
        self.learning_rate_range = learning_rate_range
        self.max_depth_range = max_depth_range
        self.min_child_weight_range = min_child_weight_range
        self.subsample_range = subsample_range
        self.colsample_bytree_range = colsample_bytree_range
        self.gamma_range = gamma_range  # ADDED
        self.reg_alpha_range = reg_alpha_range  # ADDED
        self.reg_lambda_range = reg_lambda_range  # ADDED
        self.random_state = random_state
        self.dist_method = dist_method
        self.n_iter_search = n_iter_search
        self.cv_folds = cv_folds
        self.cv_method = cv_method  # ADDED
        self.optuna_n_jobs = optuna_n_jobs
        self.optuna_timeout = optuna_timeout
        self.xgb = None
        self.best_params_ = None

    def _get_cv_splitter(self):
        """Get appropriate cross-validator for time series data."""
        if self.cv_method == 'timeseries':
            return TimeSeriesSplit(n_splits=self.cv_folds)
        else:
            return self.cv_folds  # Regular k-fold

    def _objective(self, trial, X_train, y_train):
        """
        Objective function for Optuna optimization with all hyperparameters.
        """
        # Define hyperparameter search space
        params = {}
        
        # Helper function to handle list vs distribution
        def suggest_param(name, range_obj, param_type='float'):
            if isinstance(range_obj, list):
                return trial.suggest_categorical(name, range_obj)
            else:
                if param_type == 'int':
                    return trial.suggest_int(name, int(range_obj.a), int(range_obj.b))
                elif param_type == 'float_log':
                    return trial.suggest_float(name, range_obj.a, range_obj.b, log=True)
                else:  # float linear
                    return trial.suggest_float(name, range_obj.a, range_obj.b)
        
        # Core parameters
        params['n_estimators'] = suggest_param('n_estimators', self.n_estimators_range, 'int')
        params['learning_rate'] = suggest_param('learning_rate', self.learning_rate_range, 'float_log')
        params['max_depth'] = suggest_param('max_depth', self.max_depth_range, 'int')
        params['min_child_weight'] = suggest_param('min_child_weight', self.min_child_weight_range, 'float')
        params['subsample'] = suggest_param('subsample', self.subsample_range, 'float')
        params['colsample_bytree'] = suggest_param('colsample_bytree', self.colsample_bytree_range, 'float')
        
        # New regularization parameters
        params['gamma'] = suggest_param('gamma', self.gamma_range, 'float')
        params['reg_alpha'] = suggest_param('reg_alpha', self.reg_alpha_range, 'float')
        params['reg_lambda'] = suggest_param('reg_lambda', self.reg_lambda_range, 'float')
        
        # Create and train model
        model = XGBRegressor(
            **params,
            random_state=self.random_state,
            verbosity=0,
            n_jobs=-1
        )
        
        # Use appropriate cross-validation
        from sklearn.model_selection import cross_val_score
        cv_splitter = self._get_cv_splitter()
        
        # Use MAE for rainfall prediction (less sensitive to outliers than MSE)
        scores = cross_val_score(
            model, X_train, y_train, 
            cv=cv_splitter, 
            scoring='neg_mean_absolute_error',
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
        y_train_stacked = y_train.stack(sample=('T', 'Y', 'X')).values.ravel()
        train_nan_mask = np.any(~np.isfinite(X_train_stacked), axis=1) | ~np.isfinite(y_train_stacked)
        X_train_clean = X_train_stacked[~train_nan_mask]
        y_train_clean = y_train_stacked[~train_nan_mask]

        # For climate forecasting, use MAE as it's more robust for rainfall
        scoring = make_scorer(lambda y_true, y_pred: -np.mean(np.abs(y_true - y_pred)))

        if self.search_method == 'grid':
            # Prepare parameter grid for GridSearchCV
            param_grid = {}
            
            # Helper function to handle list vs distribution
            def prepare_grid(name, range_obj, n_samples=4):
                if isinstance(range_obj, list):
                    param_grid[name] = range_obj
                else:
                    # Sample from distribution for grid search
                    samples = range_obj.rvs(size=min(n_samples, self.n_iter_search), 
                                          random_state=self.random_state)
                    param_grid[name] = np.unique(samples.astype(float) if 'int' in name else samples)
            
            prepare_grid('n_estimators', self.n_estimators_range)
            prepare_grid('learning_rate', self.learning_rate_range)
            prepare_grid('max_depth', self.max_depth_range)
            prepare_grid('min_child_weight', self.min_child_weight_range)
            prepare_grid('subsample', self.subsample_range)
            prepare_grid('colsample_bytree', self.colsample_bytree_range)
            prepare_grid('gamma', self.gamma_range)
            prepare_grid('reg_alpha', self.reg_alpha_range)
            prepare_grid('reg_lambda', self.reg_lambda_range)
            
            # Initialize XGBRegressor base model
            model = XGBRegressor(random_state=self.random_state, verbosity=0, n_jobs=-1)
            
            # Get appropriate CV splitter
            cv_splitter = self._get_cv_splitter()
            
            # Grid search
            grid_search = GridSearchCV(
                model, param_grid=param_grid,
                cv=cv_splitter, scoring=scoring,
                error_score=np.nan, n_jobs=-1
            )
            grid_search.fit(X_train_clean, y_train_clean)
            best_params = grid_search.best_params_
            
        elif self.search_method == 'random':
            # Prepare parameter distributions for RandomizedSearchCV
            param_dist = {}
            
            # All parameters can be lists or distributions
            param_dist['n_estimators'] = self.n_estimators_range
            param_dist['learning_rate'] = self.learning_rate_range
            param_dist['max_depth'] = self.max_depth_range
            param_dist['min_child_weight'] = self.min_child_weight_range
            param_dist['subsample'] = self.subsample_range
            param_dist['colsample_bytree'] = self.colsample_bytree_range
            param_dist['gamma'] = self.gamma_range
            param_dist['reg_alpha'] = self.reg_alpha_range
            param_dist['reg_lambda'] = self.reg_lambda_range
            
            # Initialize XGBRegressor base model
            model = XGBRegressor(random_state=self.random_state, verbosity=0, n_jobs=-1)
            
            # Get appropriate CV splitter
            cv_splitter = self._get_cv_splitter()
            
            # Randomized search
            random_search = RandomizedSearchCV(
                model, param_distributions=param_dist, n_iter=self.n_iter_search,
                cv=cv_splitter, scoring=scoring,
                random_state=self.random_state, error_score=np.nan, n_jobs=-1
            )
            random_search.fit(X_train_clean, y_train_clean)
            best_params = random_search.best_params_
            
        elif self.search_method == 'bayesian':
            # Bayesian optimization with Optuna
            study = optuna.create_study(
                direction='maximize',  # We're maximizing negative MAE
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
            
        else:
            raise ValueError(f"Unknown search_method: {self.search_method}. Choose from 'grid', 'random', or 'bayesian'.")

        # Store best parameters
        self.best_params_ = best_params
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
        y_train_stacked = y_train.stack(sample=('T', 'Y', 'X')).values.ravel()
        train_nan_mask = np.any(~np.isfinite(X_train_stacked), axis=1) | ~np.isfinite(y_train_stacked)
        X_train_clean = X_train_stacked[~train_nan_mask]
        y_train_clean = y_train_stacked[~train_nan_mask]

        # Stack testing data
        X_test_stacked = X_test.stack(sample=('T', 'Y', 'X')).transpose('sample', 'M').values
        y_test_stacked = y_test.stack(sample=('T', 'Y', 'X')).values.ravel()
        test_nan_mask = np.any(~np.isfinite(X_test_stacked), axis=1) | ~np.isfinite(y_test_stacked)

        # Use provided best_params or compute if None
        if best_params is None:
            best_params = self.compute_hyperparameters(X_train, y_train, clim_year_start, clim_year_end)

        # Initialize the model with best parameters including new ones
        base_params = {
            'n_estimators': best_params.get('n_estimators', 100),
            'learning_rate': best_params.get('learning_rate', 0.1),
            'max_depth': best_params.get('max_depth', 5),
            'min_child_weight': best_params.get('min_child_weight', 1),
            'subsample': best_params.get('subsample', 1.0),
            'colsample_bytree': best_params.get('colsample_bytree', 1.0),
            'random_state': self.random_state,
            'verbosity': 0,
            'n_jobs': -1
        }
        
        # Add new regularization parameters if present
        if 'gamma' in best_params:
            base_params['gamma'] = best_params['gamma']
        if 'reg_alpha' in best_params:
            base_params['reg_alpha'] = best_params['reg_alpha']
        if 'reg_lambda' in best_params:
            base_params['reg_lambda'] = best_params['reg_lambda']
        
        self.xgb = XGBRegressor(**base_params)

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
        Predictor_for_year_st = (Predictor_for_year - mean_val) / std_val
        
        # Predictor_for_year_st = Predictor_for_year

        hindcast_det_st = standardize_timeseries(hindcast_det, clim_year_start, clim_year_end)
        
        # hindcast_det_st = hindcast_det
        
        Predictant_st = standardize_timeseries(Predictant_no_m, clim_year_start, clim_year_end)
        # Predictant_st = Predictant_no_m
        
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
        
        # Initialize model with all parameters including new ones
        base_params = {
            'n_estimators': best_params.get('n_estimators', 100),
            'learning_rate': best_params.get('learning_rate', 0.1),
            'max_depth': best_params.get('max_depth', 5),
            'min_child_weight': best_params.get('min_child_weight', 1),
            'subsample': best_params.get('subsample', 1.0),
            'colsample_bytree': best_params.get('colsample_bytree', 1.0),
            'random_state': self.random_state,
            'verbosity': 0,
            'n_jobs': -1
        }
        
        # Add new regularization parameters if present
        if 'gamma' in best_params:
            base_params['gamma'] = best_params['gamma']
        if 'reg_alpha' in best_params:
            base_params['reg_alpha'] = best_params['reg_alpha']
        if 'reg_lambda' in best_params:
            base_params['reg_lambda'] = best_params['reg_lambda']
        
        self.xgb = XGBRegressor(**base_params)
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
    ### add early stopping here: early_stopping_rounds=50
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
                 n_estimators_range=[50, 100, 150, 200],  # Reduced for climate data
                 learning_rate_range=[0.005, 0.01, 0.03, 0.05, 0.1],  # Finer low-end
                 max_depth_range=[3, 5, 7],  # More conservative
                 min_child_weight_range=[1, 3, 5],
                 subsample_range=[0.6, 0.8, 1.0],
                 colsample_bytree_range=[0.6, 0.8, 1.0],
                 gamma_range=[0, 0.1, 0.2, 0.5, 1],  # ADDED: Gamma regularization
                 reg_alpha_range=[0, 0.01, 0.1],  # ADDED: L1 regularization
                 reg_lambda_range=[1, 1.5, 2],  # ADDED: L2 regularization
                 random_state=42,
                 dist_method="nonparam",
                 n_iter_search=10,
                 cv_folds=3,
                 cv_method=None,  # ADDED: CV method for temporal data
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
        self.gamma_range = gamma_range  # ADDED
        self.reg_alpha_range = reg_alpha_range  # ADDED
        self.reg_lambda_range = reg_lambda_range  # ADDED
        self.random_state = random_state
        self.dist_method = dist_method
        self.n_iter_search = n_iter_search
        self.cv_folds = cv_folds
        self.cv_method = cv_method  # ADDED
        self.n_clusters = n_clusters
        self.optuna_n_jobs = optuna_n_jobs
        self.optuna_timeout = optuna_timeout
        self.xgb = {}  # Dictionary to store models per cluster
        self.best_params_ = None
        self.cluster_da_ = None

    def _get_cv_splitter(self):
        """Get appropriate cross-validator for time series data."""
        if self.cv_method == 'timeseries':
            return TimeSeriesSplit(n_splits=self.cv_folds)
        else:
            return self.cv_folds  # Regular k-fold

    def _objective(self, trial, X_train, y_train):
        """
        Objective function for Optuna optimization with all hyperparameters.
        """
        # Define hyperparameter search space
        params = {}
        
        # Helper function to handle list vs distribution
        def suggest_param(name, range_obj, param_type='float'):
            if isinstance(range_obj, list):
                return trial.suggest_categorical(name, range_obj)
            else:
                if param_type == 'int':
                    return trial.suggest_int(name, int(range_obj.a), int(range_obj.b))
                elif param_type == 'float_log':
                    return trial.suggest_float(name, range_obj.a, range_obj.b, log=True)
                else:  # float linear
                    return trial.suggest_float(name, range_obj.a, range_obj.b)
        
        # Core parameters
        params['n_estimators'] = suggest_param('n_estimators', self.n_estimators_range, 'int')
        params['learning_rate'] = suggest_param('learning_rate', self.learning_rate_range, 'float_log')
        params['max_depth'] = suggest_param('max_depth', self.max_depth_range, 'int')
        params['min_child_weight'] = suggest_param('min_child_weight', self.min_child_weight_range, 'float')
        params['subsample'] = suggest_param('subsample', self.subsample_range, 'float')
        params['colsample_bytree'] = suggest_param('colsample_bytree', self.colsample_bytree_range, 'float')
        
        # New regularization parameters
        params['gamma'] = suggest_param('gamma', self.gamma_range, 'float')
        params['reg_alpha'] = suggest_param('reg_alpha', self.reg_alpha_range, 'float')
        params['reg_lambda'] = suggest_param('reg_lambda', self.reg_lambda_range, 'float')
        
        # Create and train model
        model = XGBRegressor(
            **params,
            random_state=self.random_state,
            verbosity=0,
            n_jobs=-1
        )
        
        # Use appropriate cross-validation
        from sklearn.model_selection import cross_val_score
        cv_splitter = self._get_cv_splitter()
        
        # Use MAE for rainfall prediction (less sensitive to outliers than MSE)
        scores = cross_val_score(
            model, X_train, y_train, 
            cv=cv_splitter, 
            scoring='neg_mean_absolute_error',
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
        # y_train_std = xarray1
        y_train_std = standardize_timeseries(Predictand, clim_year_start, clim_year_end)
        X_train_std['T'] = y_train_std['T']
        
        
        best_params_dict = {}
        
        # For climate forecasting, use MAE as it's more robust for rainfall
        scoring = make_scorer(lambda y_true, y_pred: -np.mean(np.abs(y_true - y_pred)))
        
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
                
                # Helper function to handle list vs distribution
                def prepare_grid(name, range_obj, n_samples=4):
                    if isinstance(range_obj, list):
                        param_grid[name] = range_obj
                    else:
                        # Sample from distribution for grid search
                        samples = range_obj.rvs(size=min(n_samples, self.n_iter_search), 
                                              random_state=self.random_state)
                        param_grid[name] = np.unique(samples.astype(float) if 'int' in name else samples)
                
                prepare_grid('n_estimators', self.n_estimators_range)
                prepare_grid('learning_rate', self.learning_rate_range)
                prepare_grid('max_depth', self.max_depth_range)
                prepare_grid('min_child_weight', self.min_child_weight_range)
                prepare_grid('subsample', self.subsample_range)
                prepare_grid('colsample_bytree', self.colsample_bytree_range)
                prepare_grid('gamma', self.gamma_range)
                prepare_grid('reg_alpha', self.reg_alpha_range)
                prepare_grid('reg_lambda', self.reg_lambda_range)
                
                # Initialize XGBRegressor base model
                model = XGBRegressor(random_state=self.random_state, verbosity=0, n_jobs=-1)
                
                # Get appropriate CV splitter
                cv_splitter = self._get_cv_splitter()
                
                # Grid search
                grid_search = GridSearchCV(
                    model, param_grid=param_grid,
                    cv=cv_splitter, scoring=scoring,
                    error_score=np.nan, n_jobs=-1
                )
                grid_search.fit(X_clean_c, y_clean_c)
                best_params_dict[c] = grid_search.best_params_
                
            elif self.search_method == 'random':
                # Prepare parameter distributions for RandomizedSearchCV
                param_dist = {}
                
                # All parameters can be lists or distributions
                param_dist['n_estimators'] = self.n_estimators_range
                param_dist['learning_rate'] = self.learning_rate_range
                param_dist['max_depth'] = self.max_depth_range
                param_dist['min_child_weight'] = self.min_child_weight_range
                param_dist['subsample'] = self.subsample_range
                param_dist['colsample_bytree'] = self.colsample_bytree_range
                param_dist['gamma'] = self.gamma_range
                param_dist['reg_alpha'] = self.reg_alpha_range
                param_dist['reg_lambda'] = self.reg_lambda_range
                
                # Initialize XGBRegressor base model
                model = XGBRegressor(random_state=self.random_state, verbosity=0, n_jobs=-1)
                
                # Get appropriate CV splitter
                cv_splitter = self._get_cv_splitter()
                
                # Randomized search
                random_search = RandomizedSearchCV(
                    model, param_distributions=param_dist, n_iter=self.n_iter_search,
                    cv=cv_splitter, scoring=scoring,
                    random_state=self.random_state, error_score=np.nan, n_jobs=-1
                )
                random_search.fit(X_clean_c, y_clean_c)
                best_params_dict[c] = random_search.best_params_
                
            elif self.search_method == 'bayesian':
                # Bayesian optimization with Optuna
                study = optuna.create_study(
                    direction='maximize',  # We're maximizing negative MAE
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
                best_params_dict[c] = best_params
                
            else:
                raise ValueError(f"Unknown search_method: {self.search_method}. Choose from 'grid', 'random', or 'bayesian'.")
        
        # Store best parameters and cluster data
        self.best_params_ = best_params_dict
        self.cluster_da_ = cluster_da
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
        if best_params is None or cluster_da is None:
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
            
            # Initialize the model with best parameters including new ones
            base_params = {
                'n_estimators': bp.get('n_estimators', 500),
                'learning_rate': bp.get('learning_rate', 0.01),
                'max_depth': bp.get('max_depth', 7),
                'min_child_weight': bp.get('min_child_weight', 1),
                'subsample': bp.get('subsample', 1.0),
                'colsample_bytree': bp.get('colsample_bytree', 1.0),
                'random_state': self.random_state,
                'verbosity': 0,
                'n_jobs': -1
            }
            
            # Add new regularization parameters if present
            if 'gamma' in bp:
                base_params['gamma'] = bp['gamma']
            if 'reg_alpha' in bp:
                base_params['reg_alpha'] = bp['reg_alpha']
            if 'reg_lambda' in bp:
                base_params['reg_lambda'] = bp['reg_lambda']

            # print(base_params)
            
            # Fit and predict
            xgb_c = XGBRegressor(**base_params)
            xgb_c.fit(X_train_clean_c, y_train_clean_c)
            self.xgb[c] = xgb_c
            y_pred_c = xgb_c.predict(X_test_clean_c)
            
            # Reconstruct predictions for this cluster
            result_c = np.full(len(y_test_stacked_c), np.nan)
            result_c[~test_nan_mask] = y_pred_c
            pred_c_reshaped = result_c.reshape(n_time, n_lat, n_lon)
            
            # Fill in the predictions array (using np.where to handle overlapping regions)
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

    def forecast(self, Predictant, clim_year_start, clim_year_end, hindcast_det, 
                 hindcast_det_cross, Predictor_for_year, best_params=None, cluster_da=None,
                 best_code_da=None, best_shape_da=None, best_loc_da=None, best_scale_da=None):
        """
        Forecast method using XGBoost models with optimized hyperparameters per cluster.
        
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
        best_code_da, best_shape_da, best_loc_da, best_scale_da : xarray.DataArray, optional
            Distribution parameters for 'bestfit' method.
        
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
        hindcast_det_cross['T'] = Predictant_st['T']      
        
        # Extract coordinates from X_test
        time = Predictor_for_year_st['T']
        lat = Predictor_for_year_st['Y']
        lon = Predictor_for_year_st['X']
        
        n_time = len(time)
        n_lat = len(lat)
        n_lon = len(lon)
        
        # Use provided best_params and cluster_da or compute if None
        if best_params is None or cluster_da is None:
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
            
            # Initialize the model with best parameters including new ones
            base_params = {
                'n_estimators': bp.get('n_estimators', 100),
                'learning_rate': bp.get('learning_rate', 0.1),
                'max_depth': bp.get('max_depth', 5),
                'min_child_weight': bp.get('min_child_weight', 1),
                'subsample': bp.get('subsample', 1.0),
                'colsample_bytree': bp.get('colsample_bytree', 1.0),
                'random_state': self.random_state,
                'verbosity': 0,
                'n_jobs': -1
            }
            
            # Add new regularization parameters if present
            if 'gamma' in bp:
                base_params['gamma'] = bp['gamma']
            if 'reg_alpha' in bp:
                base_params['reg_alpha'] = bp['reg_alpha']
            if 'reg_lambda' in bp:
                base_params['reg_lambda'] = bp['reg_lambda']
            
            # Fit and predict
            xgb_c = XGBRegressor(**base_params)
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
        
        # Adjust time coordinate
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
    Enhanced Random Forest-based Multi-Model Ensemble (MME) forecasting.
    This class implements a single-model forecasting approach using scikit-learn's RandomForestRegressor
    for deterministic predictions, with optional tercile probability calculations using
    various statistical distributions. Implements multiple hyperparameter optimization methods.

    Parameters
    ----------
    n_estimators_range : list of int, optional
        List of n_estimators values to tune (default is [100, 200, 300, 500]).
    max_depth_range : list of int or None, optional
        List of max depths to tune (default is [None, 5, 10, 15, 20]).
    min_samples_split_range : list of int, optional
        List of minimum samples required to split to tune (default is [2, 5, 10, 20]).
    min_samples_leaf_range : list of int, optional
        List of minimum samples required at leaf node to tune (default is [1, 2, 4, 8]).
    max_features_range : list of str, float, or None, optional
        List of max features to tune (default is [None, 'sqrt', 'log2', 0.33, 0.5, 0.7]).
    max_samples_range : list of float, optional
        List of bootstrap sample fractions (default is [None, 0.7, 0.8, 0.9, 1.0]).
    max_leaf_nodes_range : list of int or None, optional
        List of maximum leaf nodes to tune (default is [None, 100, 500, 1000]).
    min_impurity_decrease_range : list of float, optional
        List of minimum impurity decrease values (default is [0.0, 0.01, 0.1]).
    warm_start : bool, optional
        Whether to reuse solution of previous call to fit (default is False).
    random_state : int, optional
        Seed for reproducibility (default is 42).
    dist_method : str, optional
        Distribution method for tercile probabilities ('bestfit', 'nonparam', etc.) (default is 'nonparam').
    search_method : str, optional
        Hyperparameter optimization method: 'random' (RandomizedSearchCV), 
        'grid' (GridSearchCV), or 'bayesian' (Optuna) (default is 'random').
    n_iter_search : int, optional
        Number of iterations for randomized search (default is 20).
    n_trials : int, optional
        Number of trials for Bayesian optimization with Optuna (default is 50).
    cv_folds : int, optional
        Number of cross-validation folds (default is 5).
    scoring : str, optional
        Scoring metric for hyperparameter optimization (default is 'neg_mean_squared_error').
    n_jobs : int, optional
        Number of parallel jobs for RandomForest (default is -1, all cores).
    verbose : int, optional
        Verbosity level (default is 0, silent).
    """
    def __init__(self,
                 n_estimators_range: List[int] = [100, 200, 300, 500],
                 max_depth_range: List[Union[int, None]] = [None, 5, 10, 15, 20],
                 min_samples_split_range: List[int] = [2, 5, 10, 20],
                 min_samples_leaf_range: List[int] = [1, 2, 4, 8],
                 max_features_range: List[Union[str, float, None]] = [None, 'sqrt', 'log2', 0.33, 0.5, 0.7],
                 max_samples_range: List[Union[float, None]] = [None, 0.7, 0.8, 0.9, 1.0],
                 # max_leaf_nodes_range: List[Union[int, None]] = [100, 500, 1000],
                 min_impurity_decrease_range: List[float] = [0.0, 0.01, 0.1],
                 warm_start: bool = False,
                 random_state: int = 42,
                 dist_method: str = "nonparam",
                 search_method: str = "random",
                 n_iter_search: int = 20,
                 n_trials: int = 50,
                 cv_folds: int = 5,
                 scoring: str = 'neg_mean_squared_error',
                 n_jobs: int = -1,
                 verbose: int = 0):
        
        # Core Random Forest hyperparameters
        self.n_estimators_range = n_estimators_range
        self.max_depth_range = max_depth_range
        self.min_samples_split_range = min_samples_split_range
        self.min_samples_leaf_range = min_samples_leaf_range
        self.max_features_range = max_features_range
        
        # Additional Random Forest hyperparameters
        self.max_samples_range = max_samples_range
        # self.max_leaf_nodes_range = max_leaf_nodes_range
        self.min_impurity_decrease_range = min_impurity_decrease_range
        self.warm_start = warm_start
        
        # Training and optimization parameters
        self.random_state = random_state
        self.dist_method = dist_method
        self.search_method = search_method
        self.n_iter_search = n_iter_search
        self.n_trials = n_trials
        self.cv_folds = cv_folds
        self.scoring = scoring
        self.n_jobs = n_jobs
        self.verbose = verbose
        
        # Model storage
        self.rf = None
        self.best_params_ = None
        self.study_ = None
        self.feature_importances_ = None
        self.cv_results_ = None

    def _prepare_param_distributions(self) -> Dict:
        """Prepare parameter distributions for different search methods."""
        param_dict = {
            'n_estimators': self.n_estimators_range,
            'max_depth': self.max_depth_range,
            'min_samples_split': self.min_samples_split_range,
            'min_samples_leaf': self.min_samples_leaf_range,
            'max_features': self.max_features_range,
            'max_samples': self.max_samples_range,
            # 'max_leaf_nodes': self.max_leaf_nodes_range,
            'min_impurity_decrease': self.min_impurity_decrease_range
        }
        
        # Remove None values for grid search (handled differently)
        if self.search_method == 'grid':
            # For grid search, we need to handle None values carefully
            # They will be passed as is
            return param_dict
        else:
            # For random and bayesian search
            return param_dict

    def _objective(self, trial, X_train: np.ndarray, y_train: np.ndarray) -> float:
        """Objective function for Bayesian optimization with Optuna."""
        params = {
            'n_estimators': trial.suggest_categorical('n_estimators', self.n_estimators_range),
            'max_depth': trial.suggest_categorical('max_depth', self.max_depth_range),
            'min_samples_split': trial.suggest_categorical('min_samples_split', 
                                                          self.min_samples_split_range),
            'min_samples_leaf': trial.suggest_categorical('min_samples_leaf', 
                                                         self.min_samples_leaf_range),
            'max_features': trial.suggest_categorical('max_features', self.max_features_range),
            'max_samples': trial.suggest_categorical('max_samples', self.max_samples_range),
            # 'max_leaf_nodes': trial.suggest_categorical('max_leaf_nodes', 
            #                                            self.max_leaf_nodes_range),
            'min_impurity_decrease': trial.suggest_categorical('min_impurity_decrease',
                                                              self.min_impurity_decrease_range)
        }
        
        # Handle string and None values for max_features
        max_features = params['max_features']
        if isinstance(max_features, str):
            if max_features == 'None':
                max_features = None
        
        # Initialize model
        model = RandomForestRegressor(
            n_estimators=params['n_estimators'],
            max_depth=params['max_depth'],
            min_samples_split=params['min_samples_split'],
            min_samples_leaf=params['min_samples_leaf'],
            max_features=max_features,
            max_samples=params['max_samples'],
            max_leaf_nodes=params['max_leaf_nodes'],
            min_impurity_decrease=params['min_impurity_decrease'],
            warm_start=self.warm_start,
            random_state=self.random_state,
            n_jobs=self.n_jobs
        )
        
        # Cross-validation
        from sklearn.model_selection import KFold
        kf = KFold(n_splits=self.cv_folds, shuffle=True, random_state=self.random_state)
        
        scores = []
        for train_idx, val_idx in kf.split(X_train):
            X_train_fold, X_val_fold = X_train[train_idx], X_train[val_idx]
            y_train_fold, y_val_fold = y_train[train_idx], y_train[val_idx]
            
            model.fit(X_train_fold, y_train_fold.ravel())
            y_pred = model.predict(X_val_fold)
            
            # Use negative MSE for maximization
            mse = np.mean((y_val_fold - y_pred) ** 2)
            scores.append(-mse)
        
        return np.mean(scores)

    def compute_hyperparameters(self, Predictors: xr.DataArray, Predictand: xr.DataArray,
                                clim_year_start: int, clim_year_end: int) -> Dict:
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

        X_train = standardize_timeseries(Predictors, clim_year_start, clim_year_end)
        y_train = standardize_timeseries(Predictand, clim_year_start, clim_year_end)

        # Stack and clean data
        X_train = Predictors.stack(sample=('T', 'Y', 'X')).transpose('sample', 'M').values
        y_train = Predictand.stack(sample=('T', 'Y', 'X')).values.ravel()
        # Remove NaN values
        train_nan_mask = np.any(~np.isfinite(X_train), axis=1) | ~np.isfinite(y_train)
        X_train_clean = X_train[~train_nan_mask]
        y_train_clean = y_train[~train_nan_mask]
        
        if len(y_train_clean) == 0:
            raise ValueError("No valid training data after removing NaN values")
        
        # Initialize base model
        model = RandomForestRegressor(
            random_state=self.random_state,
            n_jobs=self.n_jobs,
            warm_start=self.warm_start
        )

        if self.search_method == 'grid':
            # Grid Search
            param_grid = self._prepare_param_distributions()
            grid_search = GridSearchCV(
                model,
                param_grid=param_grid,
                cv=self.cv_folds,
                scoring=self.scoring,
                n_jobs=self.n_jobs,
                verbose=self.verbose,
                error_score='raise'
            )
            grid_search.fit(X_train_clean, y_train_clean)
            self.best_params_ = grid_search.best_params_
            self.cv_results_ = grid_search.cv_results_

        elif self.search_method == 'bayesian':
            # Bayesian Optimization with Optuna
            sampler = TPESampler(seed=self.random_state)
            self.study_ = optuna.create_study(
                direction='maximize',
                sampler=sampler,
                study_name=f"RF_Optuna_{clim_year_start}_{clim_year_end}"
            )
            
            # Create objective function with data
            objective_with_data = lambda trial: self._objective(trial, X_train_clean, y_train_clean)
            self.study_.optimize(objective_with_data, n_trials=self.n_trials, show_progress_bar=self.verbose>0)
            
            # Get best parameters
            self.best_params_ = self.study_.best_params
            
            # Convert best params to proper types
            for key, value in self.best_params_.items():
                if value == 'None':
                    self.best_params_[key] = None

        else:  # Default to random search
            # Random Search
            param_dist = self._prepare_param_distributions()
            random_search = RandomizedSearchCV(
                model,
                param_distributions=param_dist,
                n_iter=self.n_iter_search,
                cv=self.cv_folds,
                scoring=self.scoring,
                random_state=self.random_state,
                n_jobs=self.n_jobs,
                verbose=self.verbose,
                error_score='raise'
            )
            random_search.fit(X_train_clean, y_train_clean)
            self.best_params_ = random_search.best_params_
            self.cv_results_ = random_search.cv_results_

        # Store feature importances if model is fitted
        if self.search_method == 'grid' or self.search_method == 'random':
            self.rf = random_search.best_estimator_ if self.search_method == 'random' else grid_search.best_estimator_
            self.feature_importances_ = self.rf.feature_importances_

        return self.best_params_

    def compute_model(self, X_train: xr.DataArray, y_train: xr.DataArray,
                      X_test: xr.DataArray, y_test: xr.DataArray,
                      best_params: Optional[Dict] = None) -> xr.DataArray:
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
        
        # Stack training data
        X_train_stacked = X_train.stack(sample=('T', 'Y', 'X')).transpose('sample', 'M').values
        y_train_stacked = y_train.stack(sample=('T', 'Y', 'X')).values.ravel()
        train_nan_mask = np.any(~np.isfinite(X_train_stacked), axis=1) | ~np.isfinite(y_train_stacked)
        X_train_clean = X_train_stacked[~train_nan_mask]
        y_train_clean = y_train_stacked[~train_nan_mask]
        
        if len(y_train_clean) == 0:
            raise ValueError("No valid training data after removing NaN values")
        
        # Stack testing data
        X_test_stacked = X_test.stack(sample=('T', 'Y', 'X')).transpose('sample', 'M').values
        y_test_stacked = y_test.stack(sample=('T', 'Y', 'X')).values.ravel()
        test_nan_mask = np.any(~np.isfinite(X_test_stacked), axis=1) | ~np.isfinite(y_test_stacked)
        
        # Use provided best_params or compute if None
        if best_params is None:
            # For consistency, we need clim_year_start and end, but they're not available here so 1991, 2020
            # We'll compute hyperparameters using the training data only
            best_params = self.compute_hyperparameters(X_train, y_train, 1991, 2020)
        
        # Handle string and None values in best_params
        processed_params = {}
        for key, value in best_params.items():
            if key == 'max_features' and isinstance(value, str):
                if value == 'None':
                    processed_params[key] = None
                else:
                    processed_params[key] = value
            elif value == 'None':
                processed_params[key] = None
            else:
                processed_params[key] = value
        
        # Initialize and train the model
        self.rf = RandomForestRegressor(
            **processed_params,
            random_state=self.random_state,
            n_jobs=self.n_jobs,
            warm_start=self.warm_start
        )
        
        self.rf.fit(X_train_clean, y_train_clean)
        
        # Store feature importances
        self.feature_importances_ = self.rf.feature_importances_
        
        # Predict on non-NaN testing data
        if np.any(~test_nan_mask):
            y_pred = self.rf.predict(X_test_stacked[~test_nan_mask])
        else:
            y_pred = np.array([])
        
        # Reconstruct predictions
        result = np.full_like(y_test_stacked, np.nan, dtype=float)
        if len(y_pred) > 0:
            result[~test_nan_mask] = y_pred
        
        # Reshape to original dimensions
        n_time = len(time)
        n_lat = len(lat)
        n_lon = len(lon)
        predictions_reshaped = result.reshape(n_time, n_lat, n_lon)
        
        # Create DataArray
        predicted_da = xr.DataArray(
            data=predictions_reshaped,
            coords={'T': time, 'Y': lat, 'X': lon},
            dims=['T', 'Y', 'X'],
            attrs={'model': 'RandomForest', 'hyperparameters': str(best_params)}
        )
        
        return predicted_da

    def get_feature_importance(self, feature_names: Optional[List[str]] = None) -> xr.DataArray:
        """
        Get feature importance scores.
        
        Parameters
        ----------
        feature_names : list of str, optional
            Names of features. If None, uses generic names.
            
        Returns
        -------
        xarray.DataArray
            Feature importance scores.
        """
        if self.feature_importances_ is None:
            raise ValueError("Model not fitted yet. Call compute_model or compute_hyperparameters first.")
        
        if feature_names is None:
            feature_names = [f'Feature_{i}' for i in range(len(self.feature_importances_))]
        
        return xr.DataArray(
            data=self.feature_importances_,
            coords={'feature': feature_names},
            dims=['feature'],
            name='feature_importance'
        )

    def get_oob_score(self) -> Optional[float]:
        """
        Get out-of-bag score if available.
        
        Returns
        -------
        float or None
            Out-of-bag score if model was trained with oob_score=True.
        """
        if self.rf is not None and hasattr(self.rf, 'oob_score_'):
            return self.rf.oob_score_
        return None

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
        Predictor_for_year_st = (Predictor_for_year - mean_val) / std_val

        # Predictor_for_year_st = Predictor_for_year

        hindcast_det_st = standardize_timeseries(hindcast_det, clim_year_start, clim_year_end)
        # hindcast_det_st = hindcast_det
        
        Predictant_st = standardize_timeseries(Predictant_no_m, clim_year_start, clim_year_end)
        # Predictant_st = Predictant_no_m
        
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

        # Handle string and None values in best_params
        processed_params = {}
        for key, value in best_params.items():
            if key == 'max_features' and isinstance(value, str):
                if value == 'None':
                    processed_params[key] = None
                else:
                    processed_params[key] = value
            elif value == 'None':
                processed_params[key] = None
            else:
                processed_params[key] = value
        
        # Initialize and train the model
        self.rf = RandomForestRegressor(
            **processed_params,
            random_state=self.random_state,
            n_jobs=self.n_jobs,
            warm_start=self.warm_start
        )
        
        self.rf.fit(X_train_clean, y_train_clean)
        
        # Store feature importances
        self.feature_importances_ = self.rf.feature_importances_


        
        # # Initialize and fit the model with best parameters
        # self.rf = RandomForestRegressor(
        #     n_estimators=best_params['n_estimators'],
        #     max_depth=best_params['max_depth'],
        #     min_samples_split=best_params['min_samples_split'],
        #     min_samples_leaf=best_params['min_samples_leaf'],
        #     max_features=best_params['max_features'],
        #     random_state=self.random_state,
        #     n_jobs=-1
        # )
        # self.rf.fit(X_train_clean, y_train_clean)
        
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
        # self.min_weight_fraction_leaf_range = min_weight_fraction_leaf_range
        self.warm_start = warm_start
        self.random_state = random_state
        self.dist_method = dist_method
        self.n_iter_search = n_iter_search
        self.cv_folds = cv_folds
        self.n_clusters = n_clusters
        self.scoring = scoring
        self.verbose = verbose
        self.rf = None
        self.best_params_dict = {}
        self.cluster_da = None
        
    def _objective(self, trial, X_train, y_train) -> float:
        """
        Objective function for Optuna optimization.
        """
        # Core hyperparameters
        if isinstance(self.n_estimators_range, list):
            params = {'n_estimators': trial.suggest_categorical('n_estimators', self.n_estimators_range)}
        else:
            params = {'n_estimators': trial.suggest_int('n_estimators', 50, 1000)}
        
        # Max depth with None handling
        if isinstance(self.max_depth_range, list):
            max_depth_options = [str(opt) if opt is None else opt for opt in self.max_depth_range]
            max_depth_str = trial.suggest_categorical('max_depth', max_depth_options)
            params['max_depth'] = None if max_depth_str == 'None' else int(max_depth_str)
        else:
            params['max_depth'] = trial.suggest_int('max_depth', 5, 50)
        
        # Other core parameters
        for param_name, param_range in [
            ('min_samples_split', self.min_samples_split_range),
            ('min_samples_leaf', self.min_samples_leaf_range),
        ]:
            if isinstance(param_range, list):
                params[param_name] = trial.suggest_categorical(param_name, param_range)
            else:
                params[param_name] = trial.suggest_int(param_name, 2, 20)
        
        # Max features
        max_features_options = [str(opt) if isinstance(opt, (int, float)) else opt for opt in self.max_features_range]
        max_features_str = trial.suggest_categorical('max_features', max_features_options)
        if max_features_str in ['sqrt', 'log2', 'auto', None]:
            params['max_features'] = max_features_str if max_features_str != 'auto' else 'sqrt'
        else:
            params['max_features'] = float(max_features_str)
        
        # Bootstrap and max_samples
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
        
        # Additional regularization parameters
        params['min_impurity_decrease'] = trial.suggest_float('min_impurity_decrease',
                                                             min(self.min_impurity_decrease_range),
                                                             max(self.min_impurity_decrease_range))
        
        # # Max leaf nodes
        # if isinstance(self.max_leaf_nodes_range[0], (int, type(None))):
        #     max_leaf_options = [str(opt) if opt is not None else opt for opt in self.max_leaf_nodes_range]
        #     max_leaf_str = trial.suggest_categorical('max_leaf_nodes', max_leaf_options)
        #     params['max_leaf_nodes'] = None if max_leaf_str == 'None' else int(max_leaf_str)
        
        # Complexity parameter for pruning
        params['ccp_alpha'] = trial.suggest_float('ccp_alpha',
                                                  min(self.ccp_alpha_range),
                                                  max(self.ccp_alpha_range))
        
        # # Min weight fraction leaf
        # params['min_weight_fraction_leaf'] = trial.suggest_float('min_weight_fraction_leaf',
        #                                                         min(self.min_weight_fraction_leaf_range),
        #                                                         max(self.min_weight_fraction_leaf_range))
        
        # Create and train model
        model = RandomForestRegressor(
            **params,
            random_state=self.random_state,
            n_jobs=-1,
            warm_start=self.warm_start
        )
        
        from sklearn.model_selection import cross_val_score
        # Cross-validation
        try:
            scores = cross_val_score(
                model, X_train, y_train, 
                cv=self.cv_folds, 
                scoring=self.scoring,
                n_jobs=-1
            )
            return np.mean(scores)
        # except:
        #     return -1e10
        except Exception as e:
            print(f"TRIAL FAILED: {e}")  
            import traceback
            traceback.print_exc()        # <--- see the full error
            return -1e10

    def _grid_search(self, X: np.ndarray, y: np.ndarray) -> Dict:
        """Perform grid search hyperparameter optimization."""
        param_grid = {}
        
        # Core parameters
        param_grid['n_estimators'] = self.n_estimators_range
        param_grid['max_depth'] = self.max_depth_range
        param_grid['min_samples_split'] = self.min_samples_split_range
        param_grid['min_samples_leaf'] = self.min_samples_leaf_range
        param_grid['max_features'] = self.max_features_range
        param_grid['bootstrap'] = self.bootstrap_range
        param_grid['max_samples'] = self.max_samples_range
        param_grid['min_impurity_decrease'] = self.min_impurity_decrease_range
        # param_grid['max_leaf_nodes'] = self.max_leaf_nodes_range
        param_grid['ccp_alpha'] = self.ccp_alpha_range
        # param_grid['min_weight_fraction_leaf'] = self.min_weight_fraction_leaf_range
        
        # Initialize model
        model = RandomForestRegressor(random_state=self.random_state, n_jobs=-1)
        
        # Grid search
        grid_search = GridSearchCV(
            model, param_grid=param_grid,
            cv=self.cv_folds, scoring=self.scoring,
            n_jobs=-1, verbose=self.verbose
        )
        
        grid_search.fit(X, y)
        
        if self.verbose > 0:
            print(f"Best score: {grid_search.best_score_:.4f}")
            print(f"Best params: {grid_search.best_params_}")
        
        return grid_search.best_params_
    
    def _random_search(self, X: np.ndarray, y: np.ndarray) -> Dict:
        """Perform random search hyperparameter optimization."""
        param_dist = {}
        
        # Core parameters
        param_dist['n_estimators'] = self.n_estimators_range
        param_dist['max_depth'] = self.max_depth_range
        param_dist['min_samples_split'] = self.min_samples_split_range
        param_dist['min_samples_leaf'] = self.min_samples_leaf_range
        param_dist['max_features'] = self.max_features_range
        param_dist['bootstrap'] = self.bootstrap_range
        param_dist['max_samples'] = self.max_samples_range
        param_dist['min_impurity_decrease'] = self.min_impurity_decrease_range
        # param_dist['max_leaf_nodes'] = self.max_leaf_nodes_range
        param_dist['ccp_alpha'] = self.ccp_alpha_range
        # param_dist['min_weight_fraction_leaf'] = self.min_weight_fraction_leaf_range
        
        # Initialize model
        model = RandomForestRegressor(random_state=self.random_state, n_jobs=-1)
        
        # Randomized search
        random_search = RandomizedSearchCV(
            model, param_distributions=param_dist,
            n_iter=self.n_iter_search, cv=self.cv_folds,
            scoring=self.scoring, random_state=self.random_state,
            n_jobs=-1, verbose=self.verbose
        )
        
        random_search.fit(X, y)
        
        if self.verbose > 0:
            print(f"Best score: {random_search.best_score_:.4f}")
            print(f"Best params: {random_search.best_params_}")
        
        return random_search.best_params_
    
    def _bayesian_search(self, X: np.ndarray, y: np.ndarray) -> Dict:
        """Perform Bayesian optimization with Optuna."""
        study = optuna.create_study(
            direction='maximize' if 'neg_' in self.scoring else 'minimize',
            sampler=optuna.samplers.TPESampler(seed=self.random_state)
        )
        
        # Create objective function with data
        objective_with_data = lambda trial: self._objective(trial, X, y)
        
        # Optimize
        study.optimize(
            objective_with_data,
            n_trials=self.n_iter_search,
            show_progress_bar=self.verbose > 0
        )
        
        # Convert Optuna's best_params to scikit-learn format
        best_params = study.best_params
        
        # Post-process parameters
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
                                clim_year_start: int, clim_year_end: int) -> Tuple[Dict, xr.DataArray]:
        """
        Independently computes the best hyperparameters using selected optimization method
        on stacked training data for each homogenized zone.
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
        Predictand, cluster_aligned = xr.align(Predictand, Cluster, join="outer")
        
        # Identify unique cluster labels
        clusters = np.unique(cluster_aligned)
        clusters = clusters[~np.isnan(clusters)]
        self.cluster_da = cluster_aligned

        y_train_std = standardize_timeseries(Predictand, clim_year_start, clim_year_end)
        X_train_std['T'] = y_train_std['T']

        # Predictors['T'] = Predictand['T']
        
        best_params_dict = {}
        
        for c in clusters:
            if self.verbose > 0:
                print(f"\nOptimizing hyperparameters for cluster {int(c)}...")
            
            # Create mask for current cluster
            mask_3d = (self.cluster_da == c).expand_dims({'T': Predictand['T']})
            # Stack data for current cluster
            X_stacked_c = X_train_std.where(mask_3d).stack(sample=('T', 'Y', 'X')).transpose('sample', 'M').values
            y_stacked_c = y_train_std.where(mask_3d).stack(sample=('T', 'Y', 'X')).values.ravel()
            
            # Remove NaN values
            nan_mask_c = np.any(~np.isfinite(X_stacked_c), axis=1) | ~np.isfinite(y_stacked_c)
            X_clean_c = X_stacked_c[~nan_mask_c]
            y_clean_c = y_stacked_c[~nan_mask_c]
            
            if len(X_clean_c) < self.cv_folds * 2:
                if self.verbose > 0:
                    print(f"Cluster {int(c)} has insufficient data ({len(X_clean_c)} samples). Skipping.")
                continue
            
            if self.search_method == 'grid':
                best_params_dict[c] = self._grid_search(X_clean_c, y_clean_c)
                
            elif self.search_method == 'random':
                best_params_dict[c] = self._random_search(X_clean_c, y_clean_c)
                
            elif self.search_method == 'bayesian':
                best_params_dict[c] = self._bayesian_search(X_clean_c, y_clean_c)
                
            else:
                raise ValueError(f"Unknown search_method: {self.search_method}")
        
        self.best_params_dict = best_params_dict
        return best_params_dict, self.cluster_da
    
    def compute_model(self, X_train: xr.DataArray, y_train: xr.DataArray,
                      X_test: xr.DataArray, y_test: xr.DataArray,
                      best_params: Optional[Dict] = None,
                      cluster_da: Optional[xr.DataArray] = None) -> xr.DataArray:
        """
        Compute deterministic hindcast using RandomForestRegressor with optimized hyperparameters.
        """
        # Use provided best_params and cluster_da or compute if None
        if best_params is None or cluster_da is None:
            best_params, cluster_da = self.compute_hyperparameters(X_train, y_train, 1970, 2000)
        
        # Extract coordinate variables from X_test
        time = X_test['T']
        lat = X_test['Y']
        lon = X_test['X']
        n_time = len(time)
        n_lat = len(lat)
        n_lon = len(lon)
        
        # Initialize predictions array
        predictions = np.full((n_time, n_lat, n_lon), np.nan)
        self.rf = {}  # Dictionary to store models per cluster
        
        for c in range(self.n_clusters):
            if c not in best_params:
                continue
            
            bp = best_params[c]
            
            # Mask for this cluster
            mask_3d_train = (cluster_da == c).expand_dims({'T': X_train['T']})
            mask_3d_test = (cluster_da == c).expand_dims({'T': X_test['T']})
            
            # Stack training data for cluster
            X_train_stacked_c = X_train.where(mask_3d_train).stack(sample=('T', 'Y', 'X')).transpose('sample', 'M').values
            y_train_stacked_c = y_train.where(mask_3d_train).stack(sample=('T', 'Y', 'X')).values.ravel()
            
            train_nan_mask = np.any(~np.isfinite(X_train_stacked_c), axis=1) | ~np.isfinite(y_train_stacked_c)
            X_train_clean_c = X_train_stacked_c[~train_nan_mask]
            y_train_clean_c = y_train_stacked_c[~train_nan_mask]
            
            # Stack testing data for cluster
            X_test_stacked_c = X_test.where(mask_3d_test).stack(sample=('T', 'Y', 'X')).transpose('sample', 'M').values
            test_nan_mask = np.any(~np.isfinite(X_test_stacked_c), axis=1)
            X_test_clean_c = X_test_stacked_c[~test_nan_mask]
            
            # Skip if no training data
            if len(X_train_clean_c) == 0:
                continue
            
            # Initialize the model with best parameters
            rf_c = RandomForestRegressor(
                **{k: v for k, v in bp.items() if k in RandomForestRegressor().get_params()},
                random_state=self.random_state,
                n_jobs=-1,
                warm_start=self.warm_start
            )
            
            # Fit and predict
            rf_c.fit(X_train_clean_c, y_train_clean_c)
            self.rf[c] = rf_c
            
            if len(X_test_clean_c) > 0:
                y_pred_c = rf_c.predict(X_test_clean_c)
                
                # Reconstruct predictions for this cluster
                result_c = np.full(len(X_test_stacked_c), np.nan)
                result_c[~test_nan_mask] = y_pred_c
                pred_c_reshaped = result_c.reshape(n_time, n_lat, n_lon)
                
                # Fill in the predictions array
                predictions = np.where(np.isnan(predictions), pred_c_reshaped, predictions)
        
        # Create output DataArray
        predicted_da = xr.DataArray(
            data=predictions,
            coords={'T': time, 'Y': lat, 'X': lon},
            dims=['T', 'Y', 'X']
        )
        
        return predicted_da
    
    # ------------------ Probability Calculation Methods ------------------
    # (Keeping all your original probability calculation methods exactly as they were)

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
                return (
                    t.ppf(0.33, df=shape, loc=loc, scale=scale),
                    t.ppf(0.67, df=shape, loc=loc, scale=scale),
                )
            elif code == 7:
                return (
                    poisson.ppf(0.33, mu=shape, loc=loc),
                    poisson.ppf(0.67, mu=shape, loc=loc),
                )
            elif code == 8:
                return (
                    nbinom.ppf(0.33, n=shape, p=scale, loc=loc),
                    nbinom.ppf(0.67, n=shape, p=scale, loc=loc),
                )
        except Exception:
            return np.nan, np.nan
    
        return np.nan, np.nan
        
    @staticmethod
    def weibull_shape_solver(k, M, V):
        """
        Function to find the root of the Weibull shape parameter 'k'.
        """
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
        """
        Generic tercile probabilities using best-fit family per grid cell.
        """
        best_guess = np.asarray(best_guess, float)
        error_variance = np.asarray(error_variance, dtype=float)
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
            c2 = expon.cdf(T2, loc=best_guess, scale=np.sqrt(error_variance))
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

        elif code == 5: # Weibull
            for i in range(n_time):
                M = best_guess[i]
                V = error_variance
                
                if V <= 0 or M <= 0:
                    out[0, i] = np.nan
                    out[1, i] = np.nan
                    out[2, i] = np.nan
                    continue
        
                initial_guess = 2.0
                k = fsolve(WAS_mme_RF.weibull_shape_solver, initial_guess, args=(M, V))[0]
        
                if k <= 0:
                    out[0, i] = np.nan
                    out[1, i] = np.nan
                    out[2, i] = np.nan
                    continue
                
                lambda_scale = M / gamma_function(1 + 1/k)
                c1 = weibull_min.cdf(T1, c=k, loc=0, scale=lambda_scale)
                c2 = weibull_min.cdf(T2, c=k, loc=0, scale=lambda_scale)
        
                out[0, i] = c1
                out[1, i] = c2 - c1
                out[2, i] = 1.0 - c2

        # Student-t
        elif code == 6:       
            if dof <= 2:
                out[0, :] = np.nan
                out[1, :] = np.nan
                out[2, :] = np.nan
            else:
                loc = best_guess
                scale = np.sqrt(error_variance * (dof - 2) / dof)
                c1 = t.cdf(T1, df=dof, loc=loc, scale=scale)
                c2 = t.cdf(T2, df=dof, loc=loc, scale=scale)

                out[0, :] = c1
                out[1, :] = c2 - c1
                out[2, :] = 1.0 - c2

        elif code == 7: # Poisson
            mu = best_guess
            c1 = poisson.cdf(T1, mu=mu)
            c2 = poisson.cdf(T2, mu=mu)
            
            out[0, :] = c1
            out[1, :] = c2 - c1
            out[2, :] = 1.0 - c2

        elif code == 8: # Negative Binomial
            p = np.where(error_variance > best_guess, 
                         best_guess / error_variance, 
                         np.nan)
            n = np.where(error_variance > best_guess, 
                         (best_guess**2) / (error_variance - best_guess), 
                         np.nan)
            
            c1 = nbinom.cdf(T1, n=n, p=p)
            c2 = nbinom.cdf(T2, n=n, p=p)
            
            out[0, :] = c1
            out[1, :] = c2 - c1
            out[2, :] = 1.0 - c2
            
        else:
            raise ValueError(f"Invalid distribution code: {dist_code}")

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
        Compute tercile probabilities for deterministic hindcasts.
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

    def forecast(self, Predictant, clim_year_start, clim_year_end, hindcast_det, 
                 hindcast_det_cross, Predictor_for_year, best_params=None, cluster_da=None,
                 best_code_da=None, best_shape_da=None, best_loc_da=None, best_scale_da=None):
        """
        Forecast method using Random Forest model with optimized hyperparameters.
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
        # Predictor_for_year_st = Predictor_for_year
        hindcast_det_st = standardize_timeseries(hindcast_det, clim_year_start, clim_year_end)
        # hindcast_det_st = hindcast_det
        Predictant_st = standardize_timeseries(Predictant_no_m, clim_year_start, clim_year_end)
        # Predictant_st = Predictant_no_m
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
            
            # Initialize the model with best parameters
            rf_c = RandomForestRegressor(
                **{k: v for k, v in bp.items() if k in RandomForestRegressor().get_params()},
                random_state=self.random_state,
                n_jobs=-1,
                warm_start=self.warm_start
            )            
            
            # Fit and predict
            rf_c.fit(X_train_clean_c, y_train_clean_c)
            self.rf[c] = rf_c
            
            if len(X_test_clean_c) > 0:
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
        
        # Adjust time coordinate
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


class WAS_mme_RF_old:
    """
    Enhanced Random Forest-based Multi-Model Ensemble (MME) forecasting for West Africa.
    This class implements a single-model forecasting approach using scikit-learn's RandomForestRegressor
    for deterministic predictions, with optional tercile probability calculations using
    various statistical distributions. Implements multiple hyperparameter optimization methods.
    
    Key Features:
    - Spatial clustering for region-specific tuning
    - Multiple hyperparameter optimization methods
    - Advanced tercile probability calculations
    - Comprehensive Random Forest hyperparameter support
    
    Parameters
    ----------
    search_method : str, optional
        Hyperparameter optimization method: 'grid', 'random', or 'bayesian' (default: 'random').
    n_estimators_range : list of int or scipy.stats distribution, optional
        List of n_estimators values to tune (default is [100, 200, 300, 500]).
        Can be a list for grid search or a distribution for random/bayesian search.
    max_depth_range : list of int or scipy.stats distribution, optional
        List of max depths to tune (default is [None, 8, 12, 16]).
        Can be a list for grid search or a distribution for random/bayesian search.
    min_samples_split_range : list of int or scipy.stats distribution, optional
        List of minimum samples required to split to tune (default is [2, 5, 10]).
        Can be a list for grid search or a distribution for random/bayesian search.
    min_samples_leaf_range : list of int or scipy.stats distribution, optional
        List of minimum samples required at leaf node to tune (default is [1, 2, 4]).
        Can be a list for grid search or a distribution for random/bayesian search.
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
    random_state : int, optional
        Seed for reproducibility (default is 42).
    dist_method : str, optional
        Distribution method for tercile probabilities ('bestfit', 'nonparam', etc.) (default is 'nonparam').
    n_iter_search : int, optional
        Number of iterations for randomized/bayesian search or points to sample for grid search (default is 20).
    cv_folds : int, optional
        Number of cross-validation folds (default is 5).
    n_clusters : int, optional
        Number of clusters for homogenized zones (default is 6 for West Africa).
    optuna_n_jobs : int, optional
        Number of parallel jobs for Optuna (default is 1).
    optuna_timeout : int, optional
        Timeout in seconds for Optuna optimization (default is None).
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
                 # max_leaf_nodes_range: List[Optional[int]] = [None, 50, 100],
                 ccp_alpha_range: List[float] = [0.0, 0.001, 0.01],
                 # min_weight_fraction_leaf_range: List[float] = [0.0, 0.1, 0.2],
                 max_samples_discrete_range: List[Optional[int]] = [None, 1000, 2000],
                 warm_start: bool = False,
                 random_state: int = 42,
                 dist_method: str = "nonparam",
                 n_iter_search: int = 20,
                 cv_folds: int = 5,
                 n_clusters: int = 6,
                 optuna_n_jobs: int = 1,
                 optuna_timeout: Optional[int] = None,
                 scoring: str = 'neg_mean_squared_error',
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
        # self.max_leaf_nodes_range = max_leaf_nodes_range
        self.ccp_alpha_range = ccp_alpha_range
        # self.min_weight_fraction_leaf_range = min_weight_fraction_leaf_range
        self.max_samples_discrete_range = max_samples_discrete_range
        self.warm_start = warm_start
        self.random_state = random_state
        self.dist_method = dist_method
        self.n_iter_search = n_iter_search
        self.cv_folds = cv_folds
        self.n_clusters = n_clusters
        self.optuna_n_jobs = optuna_n_jobs
        self.optuna_timeout = optuna_timeout
        self.scoring = scoring
        self.verbose = verbose
        self.rf = None
        self.best_params_dict = {}
        self.cluster_da = None
        
    def _objective(self, trial, X_train, y_train) -> float:
        """
        Objective function for Optuna optimization with enhanced hyperparameter space.
        
        Parameters
        ----------
        trial : optuna.Trial
            Optuna trial object.
        X_train : np.ndarray
            Training features.
        y_train : np.ndarray
            Training target.
            
        Returns
        -------
        float
            Cross-validation score.
        """
        # Define hyperparameter search space
        params = {}
        
        # Core hyperparameters
        if isinstance(self.n_estimators_range, list):
            params['n_estimators'] = trial.suggest_categorical('n_estimators', self.n_estimators_range)
        else:
            params['n_estimators'] = trial.suggest_int(
                'n_estimators', 
                int(self.n_estimators_range.a),
                int(self.n_estimators_range.b)
            )
        
        # Max depth with proper handling of None
        if isinstance(self.max_depth_range, list):
            max_depth_options = [str(opt) if opt is None else opt for opt in self.max_depth_range]
            max_depth_str = trial.suggest_categorical('max_depth', max_depth_options)
            params['max_depth'] = None if max_depth_str == 'None' else int(max_depth_str)
        else:
            params['max_depth'] = trial.suggest_int('max_depth', 5, 50)
        
        # Other core parameters
        for param_name, param_range in [
            ('min_samples_split', self.min_samples_split_range),
            ('min_samples_leaf', self.min_samples_leaf_range),
        ]:
            if isinstance(param_range, list):
                params[param_name] = trial.suggest_categorical(param_name, param_range)
            else:
                params[param_name] = trial.suggest_int(
                    param_name,
                    int(param_range.a),
                    int(param_range.b)
                )
        
        # Max features
        max_features_options = [str(opt) if isinstance(opt, (int, float)) else opt for opt in self.max_features_range]
        max_features_str = trial.suggest_categorical('max_features', max_features_options)
        if max_features_str in ['sqrt', 'log2', 'auto', None]:
            params['max_features'] = max_features_str if max_features_str != 'auto' else 'sqrt'
        else:
            params['max_features'] = float(max_features_str)
        
        # Enhanced hyperparameters
        params['bootstrap'] = trial.suggest_categorical('bootstrap', self.bootstrap_range)
        
        # Conditional max_samples
        if params['bootstrap']:
            if isinstance(self.max_samples_range[0], (int, float)):
                params['max_samples'] = trial.suggest_float('max_samples', 
                                                          min(self.max_samples_range),
                                                          max(self.max_samples_range))
            else:
                # Handle None values in list
                max_samples_options = [str(opt) if opt is not None else opt for opt in self.max_samples_range]
                max_samples_str = trial.suggest_categorical('max_samples', max_samples_options)
                params['max_samples'] = None if max_samples_str == 'None' else float(max_samples_str)
        else:
            params['max_samples'] = None
        
        # Additional regularization parameters
        params['min_impurity_decrease'] = trial.suggest_float('min_impurity_decrease',
                                                             min(self.min_impurity_decrease_range),
                                                             max(self.min_impurity_decrease_range))
        
        # # Max leaf nodes with None handling
        # if isinstance(self.max_leaf_nodes_range[0], (int, type(None))):
        #     max_leaf_options = [str(opt) if opt is not None else opt for opt in self.max_leaf_nodes_range]
        #     max_leaf_str = trial.suggest_categorical('max_leaf_nodes', max_leaf_options)
        #     params['max_leaf_nodes'] = None if max_leaf_str == 'None' else int(max_leaf_str)
        
        # Complexity parameter for pruning
        params['ccp_alpha'] = trial.suggest_float('ccp_alpha',
                                                  min(self.ccp_alpha_range),
                                                  max(self.ccp_alpha_range))
        
        # # Min weight fraction leaf
        # params['min_weight_fraction_leaf'] = trial.suggest_float('min_weight_fraction_leaf',
        #                                                         min(self.min_weight_fraction_leaf_range),
        #                                                         max(self.min_weight_fraction_leaf_range))
        
        # Create and train model
        model = RandomForestRegressor(
            **params,
            random_state=self.random_state,
            n_jobs=-1,
            warm_start=self.warm_start
        )
        
        # Use cross-validation
        try:
            scores = cross_val_score(
                model, X_train, y_train, 
                cv=self.cv_folds, 
                scoring=self.scoring,
                n_jobs=-1
            )
            return np.mean(scores)
        except Exception as e:
            # Return a very poor score if model fails
            if self.verbose > 0:
                print(f"Model failed during CV: {e}")
            return -1e10

    def _grid_search(self, X: np.ndarray, y: np.ndarray) -> Dict:
        """Perform grid search hyperparameter optimization."""
        param_grid = {}
        
        # Core parameters
        param_grid['n_estimators'] = self._process_param_range(self.n_estimators_range, 'int')
        param_grid['max_depth'] = self._process_param_range(self.max_depth_range, 'depth')
        param_grid['min_samples_split'] = self._process_param_range(self.min_samples_split_range, 'int')
        param_grid['min_samples_leaf'] = self._process_param_range(self.min_samples_leaf_range, 'int')
        param_grid['max_features'] = self.max_features_range
        param_grid['bootstrap'] = self.bootstrap_range
        param_grid['max_samples'] = self.max_samples_range
        param_grid['min_impurity_decrease'] = self.min_impurity_decrease_range
        # param_grid['max_leaf_nodes'] = self.max_leaf_nodes_range
        param_grid['ccp_alpha'] = self.ccp_alpha_range
        # param_grid['min_weight_fraction_leaf'] = self.min_weight_fraction_leaf_range
        
        # Initialize model
        model = RandomForestRegressor(random_state=self.random_state, n_jobs=-1)
        
        # Grid search
        grid_search = GridSearchCV(
            model, param_grid=param_grid,
            cv=self.cv_folds, scoring=self.scoring,
            n_jobs=-1, verbose=self.verbose,
            error_score='raise'
        )
        
        grid_search.fit(X, y)
        
        if self.verbose > 0:
            print(f"Best score: {grid_search.best_score_:.4f}")
            print(f"Best params: {grid_search.best_params_}")
        
        return grid_search.best_params_
    
    def _random_search(self, X: np.ndarray, y: np.ndarray) -> Dict:
        """Perform random search hyperparameter optimization."""
        param_dist = {}
        
        # Core parameters
        param_dist['n_estimators'] = self.n_estimators_range
        param_dist['max_depth'] = self.max_depth_range
        param_dist['min_samples_split'] = self.min_samples_split_range
        param_dist['min_samples_leaf'] = self.min_samples_leaf_range
        param_dist['max_features'] = self.max_features_range
        param_dist['bootstrap'] = self.bootstrap_range
        param_dist['max_samples'] = self.max_samples_range
        param_dist['min_impurity_decrease'] = self.min_impurity_decrease_range
        # param_dist['max_leaf_nodes'] = self.max_leaf_nodes_range
        param_dist['ccp_alpha'] = self.ccp_alpha_range
        # param_dist['min_weight_fraction_leaf'] = self.min_weight_fraction_leaf_range
        
        # Initialize model
        model = RandomForestRegressor(random_state=self.random_state, n_jobs=-1)
        
        # Randomized search
        random_search = RandomizedSearchCV(
            model, param_distributions=param_dist,
            n_iter=self.n_iter_search, cv=self.cv_folds,
            scoring=self.scoring, random_state=self.random_state,
            n_jobs=-1, verbose=self.verbose
        )
        
        random_search.fit(X, y)
        
        if self.verbose > 0:
            print(f"Best score: {random_search.best_score_:.4f}")
            print(f"Best params: {random_search.best_params_}")
        
        return random_search.best_params_
    
    def _bayesian_search(self, X: np.ndarray, y: np.ndarray) -> Dict:
        """Perform Bayesian optimization with Optuna."""
        study = optuna.create_study(
            direction='maximize' if 'neg_' in self.scoring else 'minimize',
            sampler=optuna.samplers.TPESampler(seed=self.random_state),
            pruner=optuna.pruners.MedianPruner(n_startup_trials=5)
        )
        
        # Create objective function with data
        objective_with_data = lambda trial: self._objective(trial, X, y)
        
        # Optimize
        study.optimize(
            objective_with_data,
            n_trials=self.n_iter_search,
            timeout=self.optuna_timeout,
            n_jobs=self.optuna_n_jobs,
            show_progress_bar=self.verbose > 0
        )
        
        # Convert Optuna's best_params to scikit-learn format
        best_params = study.best_params
        
        # Post-process parameters
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
    
    def _process_param_range(self, param_range, param_type: str = 'int') -> List:
        """Process parameter range for grid search."""
        if isinstance(param_range, list):
            return param_range
        else:
            # Sample from distribution
            n_samples = min(5, self.n_iter_search)
            samples = param_range.rvs(size=n_samples, random_state=self.random_state)
            
            if param_type == 'depth':
                # Handle None in max_depth
                samples = samples[~np.isnan(samples)].astype(int)
                unique_samples = np.unique(samples).tolist()
                if None in self.max_depth_range:
                    unique_samples.append(None)
                return unique_samples
            elif param_type == 'int':
                return np.unique(samples.astype(int)).tolist()
            else:
                return np.unique(samples).tolist()


    
    def compute_hyperparameters(self, Predictors: xr.DataArray, Predictand: xr.DataArray,
                                clim_year_start: int, clim_year_end: int) -> Tuple[Dict, xr.DataArray]:
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
        _, cluster_aligned = xr.align(Predictand, Cluster, join="outer")
        
        # Identify unique cluster labels
        clusters = np.unique(cluster_aligned)
        clusters = clusters[~np.isnan(clusters)]
        self.cluster_da = cluster_aligned
        
        best_params_dict = {}
        
        for c in clusters:
            if self.verbose > 0:
                print(f"\nOptimizing hyperparameters for cluster {int(c)}...")
            
            # Create mask for current cluster
            mask_3d = (self.cluster_da == c).expand_dims({'T': Predictand['T']})
            
            # Stack data for current cluster
            X_stacked_c = Predictors.where(mask_3d).stack(sample=('T', 'Y', 'X')).transpose('sample', 'M').values
            y_stacked_c = Predictand.where(mask_3d).stack(sample=('T', 'Y', 'X')).values.ravel()
            
            # Remove NaN values
            nan_mask_c = np.any(~np.isfinite(X_stacked_c), axis=1) | ~np.isfinite(y_stacked_c)
            X_clean_c = X_stacked_c[~nan_mask_c]
            y_clean_c = y_stacked_c[~nan_mask_c]
            
            if len(X_clean_c) < self.cv_folds * 2:
                if self.verbose > 0:
                    print(f"Cluster {int(c)} has insufficient data ({len(X_clean_c)} samples). Skipping.")
                continue
            
            if self.search_method == 'grid':
                best_params_dict[c] = self._grid_search(X_clean_c, y_clean_c)
                
            elif self.search_method == 'random':
                best_params_dict[c] = self._random_search(X_clean_c, y_clean_c)
                
            elif self.search_method == 'bayesian':
                best_params_dict[c] = self._bayesian_search(X_clean_c, y_clean_c)
                
            else:
                raise ValueError(f"Unknown search_method: {self.search_method}. "
                               f"Choose from 'grid', 'random', or 'bayesian'.")
        
        self.best_params_dict = best_params_dict
        return best_params_dict, self.cluster_da
    
    
    def compute_model(self, X_train: xr.DataArray, y_train: xr.DataArray,
                      X_test: xr.DataArray, y_test: xr.DataArray,
                      best_params: Optional[Dict] = None,
                      cluster_da: Optional[xr.DataArray] = None) -> xr.DataArray:
        """
        Compute deterministic hindcast using RandomForestRegressor with optimized hyperparameters.
        
        Parameters
        ----------
        X_train : xarray.DataArray
            Training predictor data with dimensions (T, M, Y, X).
        y_train : xarray.DataArray
            Training predictand data with dimensions (T, Y, X).
        X_test : xarray.DataArray
            Testing predictor data with dimensions (T, M, Y, X).
        y_test : xarray.DataArray
            T.esting predictand data with dimensions (T, Y, X).
        best_params : dict, optional
            Pre-computed best hyperparameters per cluster.
        cluster_da : xarray.DataArray, optional
            Pre-computed cluster labels.
        
        Returns
        -------
        predicted_da : xarray.DataArray
            Deterministic hindcast with dimensions (T, Y, X).
        """
        # Use provided best_params and cluster_da or compute if None
        if best_params is None or cluster_da is None:
            best_params, cluster_da = self.compute_hyperparameters(X_train, y_train, 1970, 2000)
        
        # Extract coordinate variables from X_test
        time = X_test['T']
        lat = X_test['Y']
        lon = X_test['X']
        n_time = len(time)
        n_lat = len(lat)
        n_lon = len(lon)
        
        # Initialize predictions array
        predictions = np.full((n_time, n_lat, n_lon), np.nan)
        self.rf = {}  # Dictionary to store models per cluster
        
        for c in range(self.n_clusters):
            if c not in best_params:
                continue
            
            bp = best_params[c]
            
            # Mask for this cluster
            mask_3d_train = (cluster_da == c).expand_dims({'T': X_train['T']})
            mask_3d_test = (cluster_da == c).expand_dims({'T': X_test['T']})
            
            # Stack training data for cluster
            X_train_stacked_c = X_train.where(mask_3d_train).stack(sample=('T', 'Y', 'X')).transpose('sample', 'M').values
            y_train_stacked_c = y_train.where(mask_3d_train).stack(sample=('T', 'Y', 'X')).values.ravel()
            
            train_nan_mask = np.any(~np.isfinite(X_train_stacked_c), axis=1) | ~np.isfinite(y_train_stacked_c)
            X_train_clean_c = X_train_stacked_c[~train_nan_mask]
            y_train_clean_c = y_train_stacked_c[~train_nan_mask]
            
            # Stack testing data for cluster
            X_test_stacked_c = X_test.where(mask_3d_test).stack(sample=('T', 'Y', 'X')).transpose('sample', 'M').values
            test_nan_mask = np.any(~np.isfinite(X_test_stacked_c), axis=1)
            X_test_clean_c = X_test_stacked_c[~test_nan_mask]
            
            # Skip if no training data
            if len(X_train_clean_c) == 0:
                continue
            
            # Initialize the model with best parameters
            rf_c = RandomForestRegressor(
                **{k: v for k, v in bp.items() if k in RandomForestRegressor().get_params()},
                random_state=self.random_state,
                n_jobs=-1,
                warm_start=self.warm_start
            )
            
            # Fit and predict
            rf_c.fit(X_train_clean_c, y_train_clean_c)
            self.rf[c] = rf_c
            
            if len(X_test_clean_c) > 0:
                y_pred_c = rf_c.predict(X_test_clean_c)
                
                # Reconstruct predictions for this cluster
                result_c = np.full(len(X_test_stacked_c), np.nan)
                result_c[~test_nan_mask] = y_pred_c
                pred_c_reshaped = result_c.reshape(n_time, n_lat, n_lon)
                
                # Fill in the predictions array
                predictions = np.where(np.isnan(predictions), pred_c_reshaped, predictions)
        
        # Create output DataArray
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
            # # Initialize the model with best parameters for this cluster
            # rf_c = RandomForestRegressor(
            #     n_estimators=bp['n_estimators'],
            #     max_depth=bp['max_depth'],
            #     min_samples_split=bp['min_samples_split'],
            #     min_samples_leaf=bp['min_samples_leaf'],
            #     max_features=bp['max_features'],
            #     random_state=self.random_state,
            #     n_jobs=-1
            # )

            # Initialize the model with best parameters
            rf_c = RandomForestRegressor(
                **{k: v for k, v in bp.items() if k in RandomForestRegressor().get_params()},
                random_state=self.random_state,
                n_jobs=-1,
                warm_start=self.warm_start
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

    


class WAS_mme_FastBMA:
    """
    Bayesian Model Averaging (BMA) for ensemble forecasts on a grid (xarray-friendly).

    Supported model_type:
      - 'normal' : Gaussian mixture around each member forecast (temperature, pressure)
      - 'gamma'  : Gamma mixture using member forecast as component mean (wind speed, etc.)
      - 'gamma0' : Zero-adjusted Gamma mixture (precipitation): POP via logistic + Gamma for positive part

    Gridpoint terciles only:
      - You can provide clim_terciles (dims: ('tercile', Y, X)) OR
      - provide obs_for_terciles (dims include time_dim, Y, X), and terciles are computed per gridpoint.
    """

    def __init__(self, model_type: str = "normal", tol: float = 1e-3):
        if model_type not in {"normal", "gamma", "gamma0"}:
            raise NotImplementedError(f"Model type '{model_type}' not supported.")
        self.model_type = model_type
        self.tol = tol
        self.fitted = False

        # learned parameters on training grid
        self.weights: Optional[xr.DataArray] = None              # (M, Y, X)
        self.sigma: Optional[xr.DataArray] = None                # (Y, X) for normal
        self.shape: Optional[xr.DataArray] = None                # (Y, X) for gamma/gamma0
        self.logistic_params: Optional[xr.DataArray] = None      # (param, Y, X) for gamma0

        # remember dims used in fit
        self._member_dim: Optional[str] = None
        self._time_dim: Optional[str] = None
        self._lat_dim: Optional[str] = None
        self._lon_dim: Optional[str] = None

    # ----------------------------
    # Utilities
    # ----------------------------
    @staticmethod
    def _require_dims(da: xr.DataArray, dims: Sequence[str], name: str) -> None:
        missing = [d for d in dims if d not in da.dims]
        if missing:
            raise ValueError(f"{name} is missing required dims: {missing}. Found dims={da.dims}")

    @staticmethod
    def _normalize_weights(w: np.ndarray) -> np.ndarray:
        w = np.asarray(w, dtype=float)
        w = np.clip(w, 0.0, np.inf)
        s = np.sum(w)
        if not np.isfinite(s) or s <= 0:
            return np.full_like(w, 1.0 / w.size)
        return w / s

    @staticmethod
    def compute_gridpoint_terciles_from_obs(
        obs: xr.DataArray,
        *,
        time_dim: str = "T",
        lat_dim: str = "Y",
        lon_dim: str = "X",
        q: Tuple[float, float] = (1 / 3, 2 / 3),
    ) -> xr.DataArray:
        """
        Compute gridpoint tercile thresholds from observations.

        Parameters
        ----------
        obs : xr.DataArray
            Must include dims (time_dim, lat_dim, lon_dim).
        Returns
        -------
        xr.DataArray
            dims: ('tercile', lat_dim, lon_dim) with tercile=['lower','upper'].
        """
        WAS_EnsembleBMA._require_dims(obs, [time_dim, lat_dim, lon_dim], "obs")
        qt = obs.quantile(list(q), dim=time_dim, skipna=True).rename({"quantile": "tercile"})
        qt = qt.assign_coords(tercile=["lower", "upper"])
        return qt

    def _align_params_to_forecasts(
        self,
        new_forecasts: xr.DataArray,
        *,
        member_dim: str,
        time_dim: str,
        lat_dim: str,
        lon_dim: str,
    ):
        """
        Align stored parameter grids to the new_forecasts grid using nearest neighbor selection.
        """
        self._require_dims(new_forecasts, [member_dim, time_dim, lat_dim, lon_dim], "new_forecasts")

        w_ds = self.weights.sel({lat_dim: new_forecasts[lat_dim], lon_dim: new_forecasts[lon_dim]}, method="nearest")

        if self.model_type == "normal":
            p1_ds = self.sigma.sel({lat_dim: new_forecasts[lat_dim], lon_dim: new_forecasts[lon_dim]}, method="nearest")
            p2_ds = None
        else:
            p1_ds = self.shape.sel({lat_dim: new_forecasts[lat_dim], lon_dim: new_forecasts[lon_dim]}, method="nearest")
            p2_ds = None
            if self.model_type == "gamma0":
                p2_ds = self.logistic_params.sel(
                    {lat_dim: new_forecasts[lat_dim], lon_dim: new_forecasts[lon_dim]},
                    method="nearest",
                )

        return w_ds, p1_ds, p2_ds

    # ----------------------------
    # Fit
    # ----------------------------
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
    ):
        """
        Fit BMA parameters at each gridpoint.

        hcst_grid: dims (member_dim, time_dim, lat_dim, lon_dim)
        obs_grid : dims (time_dim, lat_dim, lon_dim)
        """
        self._require_dims(hcst_grid, [member_dim, time_dim, lat_dim, lon_dim], "hcst_grid")
        self._require_dims(obs_grid, [time_dim, lat_dim, lon_dim], "obs_grid")

        self._member_dim, self._time_dim, self._lat_dim, self._lon_dim = member_dim, time_dim, lat_dim, lon_dim

        hcst = hcst_grid.transpose(member_dim, time_dim, lat_dim, lon_dim).values
        obs = obs_grid.transpose(time_dim, lat_dim, lon_dim).values

        n_memb, n_time, n_lat, n_lon = hcst.shape

        weights_map = np.full((n_memb, n_lat, n_lon), 1.0 / n_memb, dtype=float)
        param_map_1 = np.full((n_lat, n_lon), np.nan, dtype=float)        # sigma or shape
        param_map_2 = np.full((2, n_lat, n_lon), np.nan, dtype=float)     # logistic params for gamma0

        coords_dict = {
            member_dim: hcst_grid.coords[member_dim],
            lat_dim: hcst_grid.coords[lat_dim],
            lon_dim: hcst_grid.coords[lon_dim],
        }

        print(f"Fitting {self.model_type} BMA...")

        for ilat in tqdm(range(n_lat), desc="Training"):
            for ilon in range(n_lon):
                f_raw = hcst[:, :, ilat, ilon]          # (M, T)
                o_raw = obs[:, ilat, ilon]              # (T,)

                # base validity: obs finite AND all members finite at time t
                valid = np.isfinite(o_raw) & np.isfinite(f_raw).all(axis=0)
                if int(valid.sum()) < min_samples:
                    continue

                f_data = f_raw[:, valid]                # (M, N)
                o_data = o_raw[valid]                   # (N,)

                # -------------------------
                # NORMAL
                # -------------------------
                if self.model_type == "normal":

                    def nll_normal(p):
                        ws = p[:-1]
                        sigma = p[-1]
                        w_last = 1.0 - np.sum(ws)
                        if (w_last < 0) or (sigma <= 1e-6) or (not np.isfinite(sigma)):
                            return 1e12
                        all_ws = np.append(ws, w_last)

                        z = (o_data[None, :] - f_data) / sigma
                        pdf = np.dot(all_ws, norm.pdf(z) / sigma)
                        return -np.sum(np.log(np.maximum(pdf, 1e-12)))

                    x0 = np.append(np.full(n_memb - 1, 1.0 / n_memb), np.nanstd(o_data))
                    bounds = [(0.0, 1.0)] * (n_memb - 1) + [(1e-3, None)]

                    try:
                        res = minimize(nll_normal, x0, method="SLSQP", bounds=bounds, tol=self.tol)
                        if res.success:
                            ws = np.append(res.x[:-1], 1.0 - np.sum(res.x[:-1]))
                            ws = self._normalize_weights(ws)
                            weights_map[:, ilat, ilon] = ws
                            param_map_1[ilat, ilon] = float(res.x[-1])
                    except Exception:
                        continue

                # -------------------------
                # GAMMA / GAMMA0
                # -------------------------
                else:
                    # For gamma-family, ensure positivity for likelihood fit
                    # gamma0: fit POP on all data, then gamma part only on positive obs.
                    # gamma : fit only on positive obs (and positive forecasts).
                    if self.model_type == "gamma0":
                        # logistic POP: y=1 if obs>0 else 0; x=cuberoot(ens_mean)
                        y_bin = (o_data > 0).astype(int)
                        ens_mean = np.mean(f_data, axis=0)
                        x_pred = np.cbrt(np.maximum(ens_mean, 0.0))

                        def nll_logistic(beta):
                            probs = expit(beta[0] + beta[1] * x_pred)
                            probs = np.clip(probs, 1e-8, 1 - 1e-8)
                            return -np.sum(y_bin * np.log(probs) + (1 - y_bin) * np.log(1 - probs))

                        try:
                            res_log = minimize(nll_logistic, [0.0, 1.0], method="BFGS")
                            param_map_2[:, ilat, ilon] = res_log.x
                        except Exception:
                            param_map_2[:, ilat, ilon] = [-5.0, 0.0]

                        mask_pos = (o_data > 0) & np.isfinite(o_data) & np.isfinite(f_data).all(axis=0)
                        if int(mask_pos.sum()) < max(5, min_samples // 2):
                            continue

                        o_gamma = o_data[mask_pos]
                        f_gamma = f_data[:, mask_pos]

                    else:
                        # gamma: only use positive obs and positive member means
                        mask_pos = (o_data > 0) & np.isfinite(o_data) & np.isfinite(f_data).all(axis=0)
                        if int(mask_pos.sum()) < min_samples:
                            continue
                        o_gamma = o_data[mask_pos]
                        f_gamma = f_data[:, mask_pos]

                    # Additional safety for forecasts: component means must be >0 to define scales
                    if np.any(f_gamma <= 0):
                        f_gamma = np.maximum(f_gamma, 1e-3)

                    def nll_gamma(p):
                        ws = p[:-1]
                        shape_val = p[-1]
                        w_last = 1.0 - np.sum(ws)
                        if (w_last < 0) or (shape_val < 0.1) or (not np.isfinite(shape_val)):
                            return 1e12
                        all_ws = np.append(ws, w_last)

                        f_safe = np.maximum(f_gamma, 1e-3)
                        scales = f_safe / shape_val
                        pdfs = gamma.pdf(o_gamma[None, :], a=shape_val, scale=scales)
                        mix_pdf = np.dot(all_ws, pdfs)
                        return -np.sum(np.log(np.maximum(mix_pdf, 1e-12)))

                    x0 = np.append(np.full(n_memb - 1, 1.0 / n_memb), 2.0)
                    bounds = [(0.0, 1.0)] * (n_memb - 1) + [(0.1, 50.0)]

                    try:
                        res = minimize(nll_gamma, x0, method="SLSQP", bounds=bounds, tol=self.tol)
                        if res.success:
                            ws = np.append(res.x[:-1], 1.0 - np.sum(res.x[:-1]))
                            ws = self._normalize_weights(ws)
                            weights_map[:, ilat, ilon] = ws
                            param_map_1[ilat, ilon] = float(res.x[-1])
                    except Exception:
                        continue

        # save parameters to xarray
        self.weights = xr.DataArray(weights_map, dims=(member_dim, lat_dim, lon_dim), coords=coords_dict)

        if self.model_type == "normal":
            self.sigma = xr.DataArray(
                param_map_1,
                dims=(lat_dim, lon_dim),
                coords={lat_dim: coords_dict[lat_dim], lon_dim: coords_dict[lon_dim]},
            )
        else:
            self.shape = xr.DataArray(
                param_map_1,
                dims=(lat_dim, lon_dim),
                coords={lat_dim: coords_dict[lat_dim], lon_dim: coords_dict[lon_dim]},
            )
            if self.model_type == "gamma0":
                self.logistic_params = xr.DataArray(
                    param_map_2,
                    dims=("param", lat_dim, lon_dim),
                    coords={"param": ["b0", "b1"], lat_dim: coords_dict[lat_dim], lon_dim: coords_dict[lon_dim]},
                )

        self.fitted = True
        return self

    # ----------------------------
    # Predict
    # ----------------------------
    def predict_probabilistic(
        self,
        new_forecasts: xr.DataArray,
        *,
        clim_terciles: Optional[xr.DataArray] = None,
        obs_for_terciles: Optional[xr.DataArray] = None,
        tercile_q: Tuple[float, float] = (1 / 3, 2 / 3),
        quantiles: Sequence[float] = (0.1, 0.5, 0.9),
        member_dim: str = "M",
        time_dim: str = "T",
        lat_dim: str = "Y",
        lon_dim: str = "X",
    ) -> xr.Dataset:
        """
        Probabilistic prediction from BMA mixture.

        If you want tercile probabilities, provide either:
          - clim_terciles: dims ('tercile', Y, X) with tercile=['lower','upper'], OR
          - obs_for_terciles: dims (T, Y, X) and terciles are computed per gridpoint.

        Returns xr.Dataset with:
          - predictive_mean: (T, Y, X)
          - predictive_quantiles: ('quantile', T, Y, X)
          - tercile_probability (optional): ('probability', T, Y, X) with ['PB','PN','PA']
          - tercile_thresholds (optional): ('tercile', Y, X)
        """
        if not self.fitted:
            raise ValueError("Fit model first.")

        self._require_dims(new_forecasts, [member_dim, time_dim, lat_dim, lon_dim], "new_forecasts")

        w_ds, p1_ds, p2_ds = self._align_params_to_forecasts(
            new_forecasts,
            member_dim=member_dim,
            time_dim=time_dim,
            lat_dim=lat_dim,
            lon_dim=lon_dim,
        )

        # --- terciles: gridpoint only ---
        terciles_da = None
        if clim_terciles is not None:
            if "tercile" not in clim_terciles.dims and "quantile" in clim_terciles.dims:
                clim_terciles = clim_terciles.rename({"quantile": "tercile"})
            self._require_dims(clim_terciles, ["tercile", lat_dim, lon_dim], "clim_terciles")
            terciles_da = clim_terciles.sel(
                {lat_dim: new_forecasts[lat_dim], lon_dim: new_forecasts[lon_dim]},
                method="nearest",
            )
        elif obs_for_terciles is not None:
            # align obs to forecast grid then compute per-gridpoint terciles
            obs_sel = obs_for_terciles.sel(
                {lat_dim: new_forecasts[lat_dim], lon_dim: new_forecasts[lon_dim]},
                method="nearest",
            )
            terciles_da = self.compute_gridpoint_terciles_from_obs(
                obs_sel,
                time_dim=time_dim,
                lat_dim=lat_dim,
                lon_dim=lon_dim,
                q=tercile_q,
            )

        terciles_np = None
        if terciles_da is not None:
            terciles_np = terciles_da.transpose("tercile", lat_dim, lon_dim).values  # (2, n_lat, n_lon)

        # Extract numpy arrays
        fcst = new_forecasts.transpose(member_dim, time_dim, lat_dim, lon_dim).values
        ws = w_ds.transpose(member_dim, lat_dim, lon_dim).values
        p1 = p1_ds.transpose(lat_dim, lon_dim).values

        p2 = None
        if self.model_type == "gamma0":
            p2 = p2_ds.transpose("param", lat_dim, lon_dim).values

        n_memb, n_time, n_lat, n_lon = fcst.shape
        n_quant = len(quantiles)

        out_mean = np.full((n_time, n_lat, n_lon), np.nan, dtype=float)
        out_quant = np.full((n_quant, n_time, n_lat, n_lon), np.nan, dtype=float)
        out_probs = np.full((3, n_time, n_lat, n_lon), np.nan, dtype=float) if terciles_np is not None else None

        print("Predicting...")

        for ilat in tqdm(range(n_lat), desc="Predict"):
            for ilon in range(n_lon):
                param_val = p1[ilat, ilon]          # sigma (normal) or shape (gamma*)
                if not np.isfinite(param_val):
                    continue

                w_local = self._normalize_weights(ws[:, ilat, ilon])

                low_th, up_th = (np.nan, np.nan)
                if terciles_np is not None:
                    low_th, up_th = terciles_np[:, ilat, ilon]

                if self.model_type == "gamma0":
                    b0, b1 = p2[:, ilat, ilon]
                    if not (np.isfinite(b0) and np.isfinite(b1)):
                        # still allow prediction but POP will be unstable; skip this cell
                        continue

                for t in range(n_time):
                    f_m = fcst[:, t, ilat, ilon]
                    if not np.isfinite(f_m).all():
                        continue

                    # -------------------------
                    # NORMAL mixture
                    # -------------------------
                    if self.model_type == "normal":
                        out_mean[t, ilat, ilon] = float(np.dot(w_local, f_m))

                        def cdf_mix(x):
                            return float(np.dot(w_local, norm.cdf(x, loc=f_m, scale=param_val)))

                        mn = out_mean[t, ilat, ilon]
                        for iq, qv in enumerate(quantiles):
                            try:
                                out_quant[iq, t, ilat, ilon] = brentq(
                                    lambda x: cdf_mix(x) - qv,
                                    mn - 8.0 * param_val,
                                    mn + 8.0 * param_val,
                                )
                            except Exception:
                                pass

                        if out_probs is not None:
                            pb = cdf_mix(low_th)
                            pa = 1.0 - cdf_mix(up_th)
                            pn = float(np.clip(1.0 - pb - pa, 0.0, 1.0))
                            out_probs[:, t, ilat, ilon] = [pb, pn, pa]

                    # -------------------------
                    # GAMMA0 mixture (precip)
                    # -------------------------
                    elif self.model_type == "gamma0":
                        f_mean_cbrt = np.cbrt(np.maximum(float(np.mean(f_m)), 0.0))
                        pop = float(expit(b0 + b1 * f_mean_cbrt))

                        # mean of mixed distribution: POP * E[Gamma-mixture] where component means = f_m
                        out_mean[t, ilat, ilon] = pop * float(np.dot(w_local, f_m))

                        f_safe = np.maximum(f_m, 1e-3)
                        scales = f_safe / param_val  # component scale = mean/shape

                        def cdf_gamma_part(x):
                            return float(np.dot(w_local, gamma.cdf(x, a=param_val, scale=scales)))

                        for iq, qv in enumerate(quantiles):
                            if qv <= (1.0 - pop):
                                out_quant[iq, t, ilat, ilon] = 0.0
                            else:
                                target = (qv - (1.0 - pop)) / max(pop, 1e-12)
                                try:
                                    top = float(np.max(f_safe) * 8.0 + 10.0)
                                    out_quant[iq, t, ilat, ilon] = brentq(
                                        lambda x: cdf_gamma_part(x) - target,
                                        1e-6,
                                        top,
                                    )
                                except Exception:
                                    pass

                        if out_probs is not None:
                            # Mixed CDF: F(x)= (1-POP) + POP*F_gamma(x) for x>0
                            cdf_low = (1.0 - pop) + pop * cdf_gamma_part(low_th)
                            cdf_up = (1.0 - pop) + pop * cdf_gamma_part(up_th)
                            pb = float(cdf_low)
                            pa = float(1.0 - cdf_up)
                            pn = float(np.clip(1.0 - pb - pa, 0.0, 1.0))
                            out_probs[:, t, ilat, ilon] = [pb, pn, pa]

                    # -------------------------
                    # GAMMA mixture (wind, etc.)
                    # -------------------------
                    else:  # self.model_type == "gamma"
                        f_safe = np.maximum(f_m, 1e-3)
                        scales = f_safe / param_val

                        # Mixture mean: sum(w_k * mean_k) with mean_k = f_m[k]
                        out_mean[t, ilat, ilon] = float(np.dot(w_local, f_m))

                        def cdf_mix_gamma(x):
                            return float(np.dot(w_local, gamma.cdf(x, a=param_val, scale=scales)))

                        for iq, qv in enumerate(quantiles):
                            try:
                                top = float(np.max(f_safe) * 8.0 + 10.0)
                                out_quant[iq, t, ilat, ilon] = brentq(
                                    lambda x: cdf_mix_gamma(x) - qv,
                                    1e-6,
                                    top,
                                )
                            except Exception:
                                pass

                        if out_probs is not None:
                            pb = cdf_mix_gamma(low_th)
                            pa = 1.0 - cdf_mix_gamma(up_th)
                            pn = float(np.clip(1.0 - pb - pa, 0.0, 1.0))
                            out_probs[:, t, ilat, ilon] = [pb, pn, pa]

        # Package outputs
        coords = {
            time_dim: new_forecasts[time_dim],
            lat_dim: new_forecasts[lat_dim],
            lon_dim: new_forecasts[lon_dim],
        }

        ds_out = xr.Dataset()
        ds_out["predictive_mean"] = xr.DataArray(out_mean, dims=(time_dim, lat_dim, lon_dim), coords=coords)

        ds_out["predictive_quantiles"] = xr.DataArray(
            out_quant,
            dims=("quantile", time_dim, lat_dim, lon_dim),
            coords={**coords, "quantile": list(quantiles)},
        )

        if out_probs is not None:
            ds_out["tercile_probability"] = xr.DataArray(
                out_probs,
                dims=("probability", time_dim, lat_dim, lon_dim),
                coords={**coords, "probability": ["PB", "PN", "PA"]},
            )
            ds_out["tercile_thresholds"] = terciles_da

        return ds_out

class WAS_mme_FullBMA:
    """
    Hybrid Bayesian seasonal ensemble postprocessor for seasonal CF.

    Supported families per gridpoint
    --------------------------------
    - 1 -> normal
    - 4 -> gamma
    - 2 -> lognormal

    Modes
    -----
    - mode="full" : full Bayesian MCMC if available
    - mode="fast" : fast penalized optimization
    - mode="auto" : try full first, fallback to fast

    Notes
    -----
    - dist_map may contain NaN; those gridpoints are skipped.
    - Positive families (gamma, lognormal) are fitted only on y > 0.
    - The class returns predictive mean, quantiles, tercile probabilities,
      family used, fit mode used, and fit status.
    """

    DIST_CODE_TO_NAME = {1: "normal", 4: "gamma", 2: "lognormal"}
    DIST_NAME_TO_CODE = {"normal": 1, "gamma": 4, "lognormal": 2}
    MODE_CODE = {"full": 0, "fast": 1, "failed": -1}

    def __init__(
        self,
        mode: str = "auto",
        eps: float = 1e-6,
        tol: float = 1e-5,
        maxiter: int = 300,
        draws: int = 1000,
        tune: int = 1000,
        chains: int = 2,
        target_accept: float = 0.92,
        random_seed: int = 42,
        progressbar: bool = True,
        verbose: bool = True,
    ):
        if mode not in {"full", "fast", "auto"}:
            raise ValueError("mode must be one of {'full', 'fast', 'auto'}")

        self.mode = mode
        self.eps = eps
        self.tol = tol
        self.maxiter = maxiter

        self.draws = draws
        self.tune = tune
        self.chains = chains
        self.target_accept = target_accept
        self.random_seed = random_seed
        self.progressbar = progressbar
        self.verbose = verbose

        self.fitted = False

        self.weight_map: Optional[xr.DataArray] = None
        self.intercept_map: Optional[xr.DataArray] = None
        self.slope_map: Optional[xr.DataArray] = None
        self.dispersion_map: Optional[xr.DataArray] = None
        self.family_map_used: Optional[xr.DataArray] = None
        self.fit_mode_map: Optional[xr.DataArray] = None
        self.fit_status_map: Optional[xr.DataArray] = None

        self.posterior_: Dict[Tuple[int, int], Dict[str, np.ndarray]] = {}

        self._member_dim = None
        self._time_dim = None
        self._lat_dim = None
        self._lon_dim = None
        self._member_values = None
        self._lat_values = None
        self._lon_values = None

    # ------------------------------------------------------------------
    # Utilities
    # ------------------------------------------------------------------
    @staticmethod
    def _require_dims(da: xr.DataArray, dims: Sequence[str], name: str) -> None:
        missing = [d for d in dims if d not in da.dims]
        if missing:
            raise ValueError(f"{name} missing required dims {missing}. Found dims={da.dims}")

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
        """
        Robust parser for distribution family.
        Returns None for missing / invalid entries.
        """
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
        lat_dim: str = "Y",
        lon_dim: str = "X",
        q: Tuple[float, float] = (1/3, 2/3),
    ) -> xr.DataArray:
        qt = obs.quantile(list(q), dim=time_dim, skipna=True).rename({"quantile": "tercile"})
        qt = qt.assign_coords(tercile=["lower", "upper"])
        return qt

    def _align_terciles_to_forecast_grid(
        self,
        new_forecasts: xr.DataArray,
        clim_terciles: Optional[xr.DataArray],
        obs_for_terciles: Optional[xr.DataArray],
        tercile_q: Tuple[float, float],
        lat_dim: str,
        lon_dim: str,
        time_dim: str,
    ) -> Optional[xr.DataArray]:
        if clim_terciles is not None:
            if "tercile" not in clim_terciles.dims and "quantile" in clim_terciles.dims:
                clim_terciles = clim_terciles.rename({"quantile": "tercile"})
            self._require_dims(clim_terciles, ["tercile", lat_dim, lon_dim], "clim_terciles")
            return clim_terciles.sel(
                {lat_dim: new_forecasts[lat_dim], lon_dim: new_forecasts[lon_dim]},
                method="nearest",
            )

        if obs_for_terciles is not None:
            obs_sel = obs_for_terciles.sel(
                {lat_dim: new_forecasts[lat_dim], lon_dim: new_forecasts[lon_dim]},
                method="nearest",
            )
            return self.compute_gridpoint_terciles_from_obs(
                obs_sel, time_dim=time_dim, lat_dim=lat_dim, lon_dim=lon_dim, q=tercile_q
            )

        return None

    # ------------------------------------------------------------------
    # FAST backend
    # ------------------------------------------------------------------
    def _unpack_fast_params(self, p: np.ndarray, m: int):
        logits_w = p[:m]
        a = p[m:2*m]
        b_raw = p[2*m:3*m]
        disp_raw = p[3*m]

        w = softmax(logits_w)
        b = np.exp(b_raw)
        disp = np.exp(disp_raw)
        return w, a, b, disp

    def _nll_fast_normal(self, p, f_data, y_data, lam_w, lam_a, lam_b, lam_disp, y_scale):
        m, _ = f_data.shape
        w, a, b, sigma = self._unpack_fast_params(p, m)

        mu = a[:, None] + b[:, None] * f_data
        pdf = norm.pdf(y_data[None, :], loc=mu, scale=max(sigma, self.eps))
        mix_pdf = np.sum(w[:, None] * pdf, axis=0)
        nll = -np.sum(np.log(np.maximum(mix_pdf, self.eps)))

        pen = (
            lam_w * np.sum((w - 1.0 / m) ** 2)
            + lam_a * np.sum((a / max(y_scale, self.eps)) ** 2)
            + lam_b * np.sum((np.log(b)) ** 2)
            + lam_disp * (np.log(sigma / max(y_scale, self.eps)) ** 2)
        )
        return nll + pen

    def _nll_fast_gamma(self, p, f_data, y_data, lam_w, lam_a, lam_b, lam_disp, y_scale):
        m, _ = f_data.shape
        w, a, b, shape = self._unpack_fast_params(p, m)

        if np.any(y_data <= 0):
            return 1e20

        mu = np.maximum(a[:, None] + b[:, None] * f_data, self.eps)
        rate = shape / mu
        pdf = sp_gamma.pdf(y_data[None, :], a=shape, scale=1.0 / np.maximum(rate, self.eps))
        mix_pdf = np.sum(w[:, None] * pdf, axis=0)
        nll = -np.sum(np.log(np.maximum(mix_pdf, self.eps)))

        pen = (
            lam_w * np.sum((w - 1.0 / m) ** 2)
            + lam_a * np.sum((a / max(y_scale, self.eps)) ** 2)
            + lam_b * np.sum((np.log(b)) ** 2)
            + lam_disp * (np.log(shape) ** 2)
        )
        return nll + pen

    def _nll_fast_lognormal(self, p, f_data, y_data, lam_w, lam_a, lam_b, lam_disp, y_scale):
        m, _ = f_data.shape
        w, a, b, sigma_log = self._unpack_fast_params(p, m)

        if np.any(y_data <= 0):
            return 1e20

        mu_log = a[:, None] + b[:, None] * np.log(np.maximum(f_data, self.eps))
        pdf = lognorm.pdf(
            y_data[None, :],
            s=max(sigma_log, self.eps),
            scale=np.exp(mu_log),
        )
        mix_pdf = np.sum(w[:, None] * pdf, axis=0)
        nll = -np.sum(np.log(np.maximum(mix_pdf, self.eps)))

        pen = (
            lam_w * np.sum((w - 1.0 / m) ** 2)
            + lam_a * np.sum(a ** 2)
            + lam_b * np.sum((np.log(b)) ** 2)
            + lam_disp * (np.log(sigma_log) ** 2)
        )
        return nll + pen

    def _fit_single_grid_fast(
        self,
        f_data: np.ndarray,
        y_data: np.ndarray,
        family: str,
        lam_w: float,
        lam_a: float,
        lam_b: float,
        lam_disp: float,
    ) -> Dict[str, Any]:
        m, _ = f_data.shape
        y_scale = self._safe_scale(y_data, default=1.0)

        if family in {"gamma", "lognormal"}:
            valid = y_data > 0
            y_data = y_data[valid]
            f_data = f_data[:, valid]
            if y_data.size < 5:
                raise ValueError(f"Not enough positive samples for {family}")

        p0 = np.concatenate([
            np.zeros(m),
            np.zeros(m),
            np.zeros(m),
            [np.log(max(y_scale, 1.0))],
        ])

        if family == "normal":
            obj = lambda p: self._nll_fast_normal(p, f_data, y_data, lam_w, lam_a, lam_b, lam_disp, y_scale)
        elif family == "gamma":
            obj = lambda p: self._nll_fast_gamma(p, f_data, y_data, lam_w, lam_a, lam_b, lam_disp, y_scale)
        elif family == "lognormal":
            obj = lambda p: self._nll_fast_lognormal(p, f_data, y_data, lam_w, lam_a, lam_b, lam_disp, y_scale)
        else:
            raise ValueError(f"Unsupported family {family}")

        res = minimize(
            obj,
            p0,
            method="L-BFGS-B",
            options={"maxiter": self.maxiter, "ftol": self.tol},
        )

        if not res.success:
            raise RuntimeError(res.message)

        w, a, b, disp = self._unpack_fast_params(res.x, m)
        return {
            "weights": w,
            "a": a,
            "b": b,
            "dispersion": float(disp),
            "family": family,
            "backend": "fast",
        }

    # ------------------------------------------------------------------
    # FULL backend
    # ------------------------------------------------------------------
    def _fit_single_grid_full(
        self,
        f_data: np.ndarray,
        y_data: np.ndarray,
        family: str,
        alpha0: float,
        coef_scale_mult: float,
        sigma_scale_mult: float,
        shape_scale: float,
    ) -> Dict[str, Any]:
        if not HAS_PYMC:
            raise RuntimeError("PyMC/ArviZ unavailable")

        m, _ = f_data.shape
        y_scale = self._safe_scale(y_data, default=1.0)

        if family == "normal":
            with pm.Model() as model:
                weights = pm.Dirichlet("weights", a=np.full(m, alpha0))
                a = pm.Normal("a", mu=0.0, sigma=coef_scale_mult * y_scale, shape=m)
                b = pm.Normal("b", mu=1.0, sigma=coef_scale_mult, shape=m)
                sigma = pm.HalfNormal("sigma", sigma=sigma_scale_mult * y_scale)

                comp_dists = [pm.Normal.dist(mu=a[k] + b[k] * f_data[k, :], sigma=sigma) for k in range(m)]
                pm.Mixture("y_like", w=weights, comp_dists=comp_dists, observed=y_data)

                trace = pm.sample(
                    draws=self.draws,
                    tune=self.tune,
                    chains=self.chains,
                    target_accept=self.target_accept,
                    random_seed=self.random_seed,
                    progressbar=self.progressbar,
                    compute_convergence_checks=False,
                    return_inferencedata=True,
                )

            return {
                "weights": self._flatten_trace(trace, "weights"),
                "a": self._flatten_trace(trace, "a"),
                "b": self._flatten_trace(trace, "b"),
                "dispersion": self._flatten_trace(trace, "sigma"),
                "family": family,
                "backend": "full",
            }

        elif family == "gamma":
            valid = y_data > 0
            y_pos = y_data[valid]
            f_pos = f_data[:, valid]

            if y_pos.size < 5:
                raise ValueError("Not enough positive samples for gamma")

            with pm.Model() as model:
                weights = pm.Dirichlet("weights", a=np.full(m, alpha0))
                a = pm.Normal("a", mu=0.0, sigma=coef_scale_mult * y_scale, shape=m)
                b = pm.Normal("b", mu=1.0, sigma=coef_scale_mult, shape=m)
                shape = pm.HalfNormal("shape", sigma=shape_scale)

                comp_dists = []
                for k in range(m):
                    mu_k = pm.math.maximum(a[k] + b[k] * f_pos[k, :], self.eps)
                    beta_k = shape / mu_k
                    comp_dists.append(pm.Gamma.dist(alpha=shape, beta=beta_k))

                pm.Mixture("y_like", w=weights, comp_dists=comp_dists, observed=y_pos)

                trace = pm.sample(
                    draws=self.draws,
                    tune=self.tune,
                    chains=self.chains,
                    target_accept=self.target_accept,
                    random_seed=self.random_seed,
                    progressbar=self.progressbar,
                    compute_convergence_checks=False,
                    return_inferencedata=True,
                )

            return {
                "weights": self._flatten_trace(trace, "weights"),
                "a": self._flatten_trace(trace, "a"),
                "b": self._flatten_trace(trace, "b"),
                "dispersion": self._flatten_trace(trace, "shape"),
                "family": family,
                "backend": "full",
            }

        elif family == "lognormal":
            valid = y_data > 0
            y_pos = y_data[valid]
            f_pos = f_data[:, valid]

            if y_pos.size < 5:
                raise ValueError("Not enough positive samples for lognormal")

            logy = np.log(np.maximum(y_pos, self.eps))
            log_scale = self._safe_scale(logy, default=1.0)

            with pm.Model() as model:
                weights = pm.Dirichlet("weights", a=np.full(m, alpha0))
                a = pm.Normal("a", mu=0.0, sigma=coef_scale_mult * log_scale, shape=m)
                b = pm.Normal("b", mu=1.0, sigma=coef_scale_mult, shape=m)
                sigma_log = pm.HalfNormal("sigma_log", sigma=sigma_scale_mult * log_scale)

                comp_dists = []
                for k in range(m):
                    mu_log_k = a[k] + b[k] * pm.math.log(pm.math.maximum(f_pos[k, :], self.eps))
                    comp_dists.append(pm.LogNormal.dist(mu=mu_log_k, sigma=sigma_log))

                pm.Mixture("y_like", w=weights, comp_dists=comp_dists, observed=y_pos)

                trace = pm.sample(
                    draws=self.draws,
                    tune=self.tune,
                    chains=self.chains,
                    target_accept=self.target_accept,
                    random_seed=self.random_seed,
                    progressbar=self.progressbar,
                    compute_convergence_checks=False,
                    return_inferencedata=True,
                )

            return {
                "weights": self._flatten_trace(trace, "weights"),
                "a": self._flatten_trace(trace, "a"),
                "b": self._flatten_trace(trace, "b"),
                "dispersion": self._flatten_trace(trace, "sigma_log"),
                "family": family,
                "backend": "full",
            }

        raise ValueError(f"Unsupported family {family}")

    # ------------------------------------------------------------------
    # Hybrid single-grid fit
    # ------------------------------------------------------------------
    def _fit_single_grid_hybrid(
        self,
        f_data: np.ndarray,
        y_data: np.ndarray,
        family: str,
        lam_w: float,
        lam_a: float,
        lam_b: float,
        lam_disp: float,
        alpha0: float,
        coef_scale_mult: float,
        sigma_scale_mult: float,
        shape_scale: float,
    ) -> Dict[str, Any]:
        if self.mode == "fast":
            return self._fit_single_grid_fast(f_data, y_data, family, lam_w, lam_a, lam_b, lam_disp)

        if self.mode == "full":
            return self._fit_single_grid_full(f_data, y_data, family, alpha0, coef_scale_mult, sigma_scale_mult, shape_scale)

        try:
            return self._fit_single_grid_full(f_data, y_data, family, alpha0, coef_scale_mult, sigma_scale_mult, shape_scale)
        except Exception:
            return self._fit_single_grid_fast(f_data, y_data, family, lam_w, lam_a, lam_b, lam_disp)

    # ------------------------------------------------------------------
    # Fit
    # ------------------------------------------------------------------
    def fit(
        self,
        hcst_grid: xr.DataArray,
        obs_grid: xr.DataArray,
        dist_map: xr.DataArray,
        *,
        member_dim: str = "M",
        time_dim: str = "T",
        lat_dim: str = "Y",
        lon_dim: str = "X",
        min_samples: int = 12,
        lam_w: float = 1.0,
        lam_a: float = 0.1,
        lam_b: float = 0.1,
        lam_disp: float = 0.1,
        alpha0: float = 1.0,
        coef_scale_mult: float = 1.0,
        sigma_scale_mult: float = 1.0,
        shape_scale: float = 5.0,
    ):
        self._require_dims(hcst_grid, [member_dim, time_dim, lat_dim, lon_dim], "hcst_grid")
        self._require_dims(obs_grid, [time_dim, lat_dim, lon_dim], "obs_grid")
        self._require_dims(dist_map, [lat_dim, lon_dim], "dist_map")

        self._member_dim = member_dim
        self._time_dim = time_dim
        self._lat_dim = lat_dim
        self._lon_dim = lon_dim

        self._member_values = hcst_grid[member_dim].values
        self._lat_values = hcst_grid[lat_dim].values
        self._lon_values = hcst_grid[lon_dim].values

        hcst = hcst_grid.transpose(member_dim, time_dim, lat_dim, lon_dim).values
        obs = obs_grid.transpose(time_dim, lat_dim, lon_dim).values
        dist_np = dist_map.transpose(lat_dim, lon_dim).values

        n_memb, _, n_lat, n_lon = hcst.shape

        weight_map = np.full((n_memb, n_lat, n_lon), np.nan, dtype=float)
        intercept_map = np.full((n_memb, n_lat, n_lon), np.nan, dtype=float)
        slope_map = np.full((n_memb, n_lat, n_lon), np.nan, dtype=float)
        dispersion_map = np.full((n_lat, n_lon), np.nan, dtype=float)
        family_code = np.full((n_lat, n_lon), np.nan, dtype=float)
        fit_mode = np.full((n_lat, n_lon), self.MODE_CODE["failed"], dtype=float)
        fit_status = np.zeros((n_lat, n_lon), dtype=float)

        self.posterior_.clear()

        if self.verbose:
            print(f"Fitting hybrid Bayesian seasonal postprocessor [mode={self.mode}]...")

        for ilat in tqdm(range(n_lat), desc="Training", disable=not self.verbose):
            for ilon in range(n_lon):
                fam = self._parse_family(dist_np[ilat, ilon])
                if fam is None:
                    continue

                f_raw = hcst[:, :, ilat, ilon]
                y_raw = obs[:, ilat, ilon]

                valid = np.isfinite(y_raw) & np.isfinite(f_raw).all(axis=0)
                if int(valid.sum()) < min_samples:
                    continue

                f_data = f_raw[:, valid]
                y_data = y_raw[valid]

                try:
                    res = self._fit_single_grid_hybrid(
                        f_data=f_data,
                        y_data=y_data,
                        family=fam,
                        lam_w=lam_w,
                        lam_a=lam_a,
                        lam_b=lam_b,
                        lam_disp=lam_disp,
                        alpha0=alpha0,
                        coef_scale_mult=coef_scale_mult,
                        sigma_scale_mult=sigma_scale_mult,
                        shape_scale=shape_scale,
                    )
                except Exception as e:
                    if self.verbose:
                        print(f"Skipping grid ({ilat},{ilon}) [{fam}] due to fit error: {e}")
                    continue

                family_code[ilat, ilon] = self.DIST_NAME_TO_CODE[fam]
                fit_status[ilat, ilon] = 1.0

                if res["backend"] == "fast":
                    fit_mode[ilat, ilon] = self.MODE_CODE["fast"]
                    weight_map[:, ilat, ilon] = res["weights"]
                    intercept_map[:, ilat, ilon] = res["a"]
                    slope_map[:, ilat, ilon] = res["b"]
                    dispersion_map[ilat, ilon] = float(res["dispersion"])

                elif res["backend"] == "full":
                    fit_mode[ilat, ilon] = self.MODE_CODE["full"]
                    self.posterior_[(ilat, ilon)] = res
                    weight_map[:, ilat, ilon] = res["weights"].mean(axis=0)
                    intercept_map[:, ilat, ilon] = res["a"].mean(axis=0)
                    slope_map[:, ilat, ilon] = res["b"].mean(axis=0)
                    dispersion_map[ilat, ilon] = float(np.mean(res["dispersion"]))

        coords_w = {
            member_dim: self._member_values,
            lat_dim: self._lat_values,
            lon_dim: self._lon_values,
        }
        coords_2d = {
            lat_dim: self._lat_values,
            lon_dim: self._lon_values,
        }

        self.weight_map = xr.DataArray(weight_map, dims=(member_dim, lat_dim, lon_dim), coords=coords_w)
        self.intercept_map = xr.DataArray(intercept_map, dims=(member_dim, lat_dim, lon_dim), coords=coords_w)
        self.slope_map = xr.DataArray(slope_map, dims=(member_dim, lat_dim, lon_dim), coords=coords_w)
        self.dispersion_map = xr.DataArray(dispersion_map, dims=(lat_dim, lon_dim), coords=coords_2d)
        self.family_map_used = xr.DataArray(family_code, dims=(lat_dim, lon_dim), coords=coords_2d)
        self.fit_mode_map = xr.DataArray(fit_mode, dims=(lat_dim, lon_dim), coords=coords_2d)
        self.fit_status_map = xr.DataArray(fit_status, dims=(lat_dim, lon_dim), coords=coords_2d)

        self.fitted = True
        return self

    # ------------------------------------------------------------------
    # Predictive helpers
    # ------------------------------------------------------------------
    def _mixture_mean_fast(self, f_m, w, a, b, disp, family):
        if family == "normal":
            mu = a + b * f_m
            return float(np.sum(w * mu))
        if family == "gamma":
            mu = np.maximum(a + b * f_m, self.eps)
            return float(np.sum(w * mu))
        if family == "lognormal":
            mu_log = a + b * np.log(np.maximum(f_m, self.eps))
            means = np.exp(mu_log + 0.5 * disp**2)
            return float(np.sum(w * means))
        raise ValueError(family)

    def _mixture_cdf_fast(self, x, f_m, w, a, b, disp, family):
        if family == "normal":
            mu = a + b * f_m
            vals = norm.cdf(x, loc=mu, scale=max(disp, self.eps))
            return float(np.sum(w * vals))

        if family == "gamma":
            mu = np.maximum(a + b * f_m, self.eps)
            shape = max(disp, self.eps)
            rate = shape / mu
            vals = sp_gamma.cdf(x, a=shape, scale=1.0 / np.maximum(rate, self.eps))
            return float(np.sum(w * vals))

        if family == "lognormal":
            mu_log = a + b * np.log(np.maximum(f_m, self.eps))
            vals = lognorm.cdf(x, s=max(disp, self.eps), scale=np.exp(mu_log))
            return float(np.sum(w * vals))

        raise ValueError(family)

    def _mixture_quantile_fast(self, q, f_m, w, a, b, disp, family):
        mean_pred = self._mixture_mean_fast(f_m, w, a, b, disp, family)

        if family == "normal":
            lo = mean_pred - 8 * max(disp, self.eps)
            hi = mean_pred + 8 * max(disp, self.eps)
        elif family == "gamma":
            lo = 0.0
            hi = max(mean_pred * 6.0 + 10.0, 1.0)
        elif family == "lognormal":
            lo = 0.0
            hi = max(mean_pred * 8.0 + 10.0, 1.0)
        else:
            raise ValueError(family)

        xs = np.linspace(lo, hi, 600)
        cdfs = np.array([self._mixture_cdf_fast(xx, f_m, w, a, b, disp, family) for xx in xs])
        idx = np.searchsorted(cdfs, q, side="left")
        idx = min(max(idx, 0), len(xs) - 1)
        return float(xs[idx])

    def _sample_pp_full(self, f_m, post, family, rng, n_samples):
        n_post = post["weights"].shape[0]
        idx = rng.choice(n_post, size=n_samples, replace=(n_samples > n_post))

        w = post["weights"][idx, :]
        a = post["a"][idx, :]
        b = post["b"][idx, :]
        disp = post["dispersion"][idx]

        u = rng.random(n_samples)
        cdf = np.cumsum(w, axis=1)
        comp = (u[:, None] > cdf).sum(axis=1)

        if family == "normal":
            mu = a + b * f_m[None, :]
            mu_sel = mu[np.arange(n_samples), comp]
            return rng.normal(mu_sel, disp)

        if family == "gamma":
            mu = np.maximum(a + b * f_m[None, :], self.eps)
            mu_sel = mu[np.arange(n_samples), comp]
            shape = np.maximum(disp, self.eps)
            rate = shape / np.maximum(mu_sel, self.eps)
            return rng.gamma(shape=shape, scale=1.0 / rate)

        if family == "lognormal":
            mu_log = a + b * np.log(np.maximum(f_m[None, :], self.eps))
            mu_sel = mu_log[np.arange(n_samples), comp]
            return rng.lognormal(mean=mu_sel, sigma=np.maximum(disp, self.eps))

        raise ValueError(family)

    # ------------------------------------------------------------------
    # Predict
    # ------------------------------------------------------------------
    def predict_probabilistic(
        self,
        new_forecasts: xr.DataArray,
        *,
        clim_terciles: Optional[xr.DataArray] = None,
        obs_for_terciles: Optional[xr.DataArray] = None,
        tercile_q: Tuple[float, float] = (1/3, 2/3),
        quantiles: Sequence[float] = (0.1, 0.5, 0.9),
        n_pp_samples: int = 2000,
        member_dim: str = "M",
        time_dim: str = "T",
        lat_dim: str = "Y",
        lon_dim: str = "X",
    ) -> xr.Dataset:
        if not self.fitted:
            raise ValueError("Model must be fitted before prediction.")

        self._require_dims(new_forecasts, [member_dim, time_dim, lat_dim, lon_dim], "new_forecasts")

        fcst = new_forecasts.transpose(member_dim, time_dim, lat_dim, lon_dim).values
        _, n_time, n_lat, n_lon = fcst.shape

        w_map = self.weight_map.transpose(member_dim, lat_dim, lon_dim).values
        a_map = self.intercept_map.transpose(member_dim, lat_dim, lon_dim).values
        b_map = self.slope_map.transpose(member_dim, lat_dim, lon_dim).values
        d_map = self.dispersion_map.transpose(lat_dim, lon_dim).values
        fam_map = self.family_map_used.transpose(lat_dim, lon_dim).values
        mode_map = self.fit_mode_map.transpose(lat_dim, lon_dim).values
        status_map = self.fit_status_map.transpose(lat_dim, lon_dim).values

        terciles_da = self._align_terciles_to_forecast_grid(
            new_forecasts=new_forecasts,
            clim_terciles=clim_terciles,
            obs_for_terciles=obs_for_terciles,
            tercile_q=tercile_q,
            lat_dim=lat_dim,
            lon_dim=lon_dim,
            time_dim=time_dim,
        )

        terciles_np = None
        if terciles_da is not None:
            terciles_np = terciles_da.transpose("tercile", lat_dim, lon_dim).values

        out_mean = np.full((n_time, n_lat, n_lon), np.nan, dtype=float)
        out_quant = np.full((len(quantiles), n_time, n_lat, n_lon), np.nan, dtype=float)
        out_probs = np.full((3, n_time, n_lat, n_lon), np.nan, dtype=float) if terciles_np is not None else None
        out_family = np.full((n_lat, n_lon), np.nan, dtype=float)
        out_mode = np.full((n_lat, n_lon), np.nan, dtype=float)
        out_status = np.full((n_lat, n_lon), np.nan, dtype=float)

        rng = np.random.default_rng(self.random_seed)

        if self.verbose:
            print("Predicting with hybrid Bayesian seasonal postprocessor...")

        for ilat in tqdm(range(n_lat), desc="Predict", disable=not self.verbose):
            for ilon in range(n_lon):
                if not np.isfinite(status_map[ilat, ilon]) or status_map[ilat, ilon] != 1:
                    continue

                fam_code = fam_map[ilat, ilon]
                if not np.isfinite(fam_code):
                    continue

                fam = self.DIST_CODE_TO_NAME.get(int(fam_code), None)
                if fam is None:
                    continue

                mode_code = int(mode_map[ilat, ilon])

                out_family[ilat, ilon] = int(fam_code)
                out_mode[ilat, ilon] = mode_code
                out_status[ilat, ilon] = 1.0

                low_th = up_th = None
                if terciles_np is not None:
                    low_th, up_th = terciles_np[:, ilat, ilon]

                for t in range(n_time):
                    f_m = fcst[:, t, ilat, ilon]
                    if not np.isfinite(f_m).all():
                        continue

                    if mode_code == self.MODE_CODE["full"]:
                        post = self.posterior_.get((ilat, ilon))
                        if post is None:
                            continue

                        y_rep = self._sample_pp_full(f_m, post, fam, rng, n_pp_samples)
                        out_mean[t, ilat, ilon] = np.mean(y_rep)
                        out_quant[:, t, ilat, ilon] = np.quantile(y_rep, quantiles)

                        if terciles_np is not None:
                            pb = np.mean(y_rep <= low_th)
                            pa = np.mean(y_rep > up_th)
                            pn = np.mean((y_rep > low_th) & (y_rep <= up_th))
                            out_probs[:, t, ilat, ilon] = [pb, pn, pa]

                    elif mode_code == self.MODE_CODE["fast"]:
                        w = w_map[:, ilat, ilon]
                        a = a_map[:, ilat, ilon]
                        b = b_map[:, ilat, ilon]
                        disp = float(d_map[ilat, ilon])

                        out_mean[t, ilat, ilon] = self._mixture_mean_fast(f_m, w, a, b, disp, fam)
                        for iq, qv in enumerate(quantiles):
                            out_quant[iq, t, ilat, ilon] = self._mixture_quantile_fast(qv, f_m, w, a, b, disp, fam)

                        if terciles_np is not None:
                            pb = self._mixture_cdf_fast(low_th, f_m, w, a, b, disp, fam)
                            pa = 1.0 - self._mixture_cdf_fast(up_th, f_m, w, a, b, disp, fam)
                            pn = max(0.0, 1.0 - pb - pa)
                            out_probs[:, t, ilat, ilon] = [pb, pn, pa]

        coords = {
            time_dim: new_forecasts[time_dim],
            lat_dim: new_forecasts[lat_dim],
            lon_dim: new_forecasts[lon_dim],
        }

        ds = xr.Dataset()
        ds["predictive_mean"] = xr.DataArray(out_mean, dims=(time_dim, lat_dim, lon_dim), coords=coords)
        ds["predictive_quantiles"] = xr.DataArray(
            out_quant,
            dims=("quantile", time_dim, lat_dim, lon_dim),
            coords={**coords, "quantile": list(quantiles)},
        )

        ds["family_used"] = xr.DataArray(
            out_family,
            dims=(lat_dim, lon_dim),
            coords={lat_dim: new_forecasts[lat_dim], lon_dim: new_forecasts[lon_dim]},
            attrs={"1": "normal", "4": "gamma", "2": "lognormal"},
        )

        ds["fit_mode_used"] = xr.DataArray(
            out_mode,
            dims=(lat_dim, lon_dim),
            coords={lat_dim: new_forecasts[lat_dim], lon_dim: new_forecasts[lon_dim]},
            attrs={"0": "full", "1": "fast", "-1": "failed"},
        )

        ds["fit_status"] = xr.DataArray(
            out_status,
            dims=(lat_dim, lon_dim),
            coords={lat_dim: new_forecasts[lat_dim], lon_dim: new_forecasts[lon_dim]},
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

    # ------------------------------------------------------------------
    # WAS-style wrapper
    # ------------------------------------------------------------------
    def compute_model(
        self,
        X_train: xr.DataArray,
        y_train: xr.DataArray,
        X_test: xr.DataArray,
        dist_map: xr.DataArray,
        *,
        clim_terciles: Optional[xr.DataArray] = None,
        obs_for_terciles: Optional[xr.DataArray] = None,
        tercile_q: Tuple[float, float] = (1/3, 2/3),
        quantiles: Sequence[float] = (0.1, 0.5, 0.9),
        n_pp_samples: int = 2000,
        member_dim: str = "M",
        time_dim: str = "T",
        lat_dim: str = "Y",
        lon_dim: str = "X",
        min_samples: int = 12,
        lam_w: float = 1.0,
        lam_a: float = 0.1,
        lam_b: float = 0.1,
        lam_disp: float = 0.1,
        alpha0: float = 1.0,
        coef_scale_mult: float = 1.0,
        sigma_scale_mult: float = 1.0,
        shape_scale: float = 5.0,
    ) -> xr.Dataset:
        self.fit(
            hcst_grid=X_train,
            obs_grid=y_train,
            dist_map=dist_map,
            member_dim=member_dim,
            time_dim=time_dim,
            lat_dim=lat_dim,
            lon_dim=lon_dim,
            min_samples=min_samples,
            lam_w=lam_w,
            lam_a=lam_a,
            lam_b=lam_b,
            lam_disp=lam_disp,
            alpha0=alpha0,
            coef_scale_mult=coef_scale_mult,
            sigma_scale_mult=sigma_scale_mult,
            shape_scale=shape_scale,
        )

        return self.predict_probabilistic(
            new_forecasts=X_test,
            clim_terciles=clim_terciles,
            obs_for_terciles=obs_for_terciles,
            tercile_q=tercile_q,
            quantiles=quantiles,
            n_pp_samples=n_pp_samples,
            member_dim=member_dim,
            time_dim=time_dim,
            lat_dim=lat_dim,
            lon_dim=lon_dim,
        )