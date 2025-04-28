from wass2s.utils import *
import numpy as np
import xarray as xr
from dask.distributed import Client
import pandas as pd
import xcast as xc  # 
from scipy import stats
from scipy.stats import norm
from scipy.stats import lognorm
from scipy.stats import gamma
from sklearn.neural_network import MLPRegressor
from sklearn.ensemble import StackingRegressor
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor
from sklearn.neural_network import MLPRegressor
from sklearn.model_selection import GridSearchCV, KFold
from xgboost import XGBRegressor
from lightgbm import LGBMRegressor
from sklearn.linear_model import Ridge
from sklearn.linear_model import Lasso
from sklearn.neighbors import KNeighborsRegressor
from sklearn.tree import DecisionTreeRegressor
from sklearn.svm import SVR
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.ensemble import AdaBoostRegressor
import pymc as pm
import arviz as az
import gc

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
                             gcm=True, agroparam=False, Prob=False,
                             ELM_ELR=False, dir_to_save_model=None,
                             best_models=None, scores=None,
                             year_start=None, year_end=None, 
                             model=True, month_of_initialization=None, 
                             lead_time=None, year_forecast=None, 
                             score_metric='GROC'):
    """
    Process hindcast and forecast datasets for a multi-model ensemble.
    
    The 'score_metric' parameter determines which metric (e.g., 'Pearson', 
    'MAE', or 'GROC') is used to organize the scores.
    """
    all_model_hdcst = {}
    all_model_fcst = {}
    
    if gcm:
        # Standardize model keys for matching.
        target_prefixes = [m.lower().replace('.prcp', '') for m in best_models]
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
    else:
        target_prefixes = [m.replace(m.split('.')[1], '') for m in best_models]
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
        
        all_model_fcst = (
            xr.concat(forecast_det_list, dim='M')
              .assign_coords({'M': predictor_names})
              .rename({'T': 'S'})
              .transpose('S', 'M', 'Y', 'X')
        ) * mask
        
        obs = rainfall.expand_dims({'M': [0]}, axis=1) * mask

    elif Prob:
        all_model_hdcst = (
            xr.concat(hindcast_det_list, dim='M')
              .assign_coords({'M': predictor_names})
              .transpose('probability', 'T', 'M', 'Y', 'X')
        ) * mask
        
        all_model_fcst = (
            xr.concat(forecast_det_list, dim='M')
              .assign_coords({'M': predictor_names})
              .transpose('probability', 'T', 'M', 'Y', 'X')
        ) * mask
        
        obs = rainfall.expand_dims({'M': [0]}, axis=1) * mask

    else:
        all_model_hdcst = (
            xr.concat(hindcast_det_list, dim='M')
              .assign_coords({'M': predictor_names})
              .transpose('T', 'M', 'Y', 'X')
        ) * mask
        
        all_model_fcst = (
            xr.concat(forecast_det_list, dim='M')
              .assign_coords({'M': predictor_names})
              .transpose('T', 'M', 'Y', 'X')
        ) * mask
        
        obs = rainfall.expand_dims({'M': [0]}, axis=1) * mask
    
    return all_model_hdcst, all_model_fcst, obs, scores_organized


def myfill(all_model_fcst, obs):
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
        Apply a fixed masking to the score_array based on the chosen metric and threshold.

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
        Compute weighted forecasts/hindcasts using model scores.
        
        Parameters:
            rainfall: xarray DataArray with rainfall observations.
            hdcst: xarray DataArray with hindcast forecasts; assumed to have a model coordinate 'M'.
            fcst: xarray DataArray with forecasts; assumed to have a model coordinate 'M'.
            scores: Dictionary with model names as keys and corresponding score arrays as values.
            complete (bool): If True, grid cells missing weighted data are filled with unweighted averages.
        
        Returns:
            Tuple of weighted hindcast and forecast DataArrays (with spatial masking applied).
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
    def __init__(self):
        # Initialize any required attributes here
        pass

    def compute(self, rainfall, hdcst, fcst, scores, threshold=0.5, complete=False):
        """
        Compute the weighted ensemble estimates for hindcast and forecast datasets.
        
        Parameters:
          rainfall  : xarray.DataArray containing observed rainfall data.
          hdcst     : xarray.DataArray with hindcast data having a 'M' (model) dimension.
          fcst      : xarray.DataArray with forecast data having a 'M' (model) dimension.
          scores    : Dictionary mapping model names to score arrays.
          threshold : Lower threshold below which the score is set to 0.
          complete  : If True, fill missing values with unweighted ensemble averages.
        
        Returns:
          A tuple of (hindcast_weighted, forecast_weighted), each multiplied by the spatial mask.
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
    Genetic Algorithm for multi-model ensemble weighting.
    
    Each chromosome = [w1, w2, ..., wM] for M models.
    GA searches for the best set of weights that minimize MSE vs. observations.
    Weights are nonnegative and sum to 1.
    """

    def __init__(self,
                 population_size=20,
                 max_iter=50,
                 crossover_rate=0.7,
                 mutation_rate=0.01,
                 random_state=42,
                 dist_method="gamma"):
        """
        Constructor for the GA-based ensemble.

        Parameters
        ----------
        population_size : int
            Number of individuals in the GA population.
        max_iter : int
            Maximum number of generations for the GA.
        crossover_rate : float
            Probability of performing crossover on selected parents.
        mutation_rate : float
            Probability of mutating a gene in a chromosome.
        random_state : int
            Seed for reproducibility.
        dist_method : str
            Distribution method for tercile probability calculations.
            Options: 't', 'gamma', 'nonparam', 'normal', 'lognormal', 'weibull_min'.
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
        Initialize population of weight vectors, each length = n_models.
        We start them as random values in [0,1], then normalize so sum=1.
        """
        population = []
        for _ in range(self.population_size):
            w = np.random.rand(n_models)
            w /= w.sum()  # normalize so sum=1
            population.append(w)
        return population

    def _fitness_function(self, weights, X, y):
        """
        Negative MSE of the ensemble:
            y_pred = sum_j weights[j] * X[:, j]
        We assume y is (n_samples,) or (n_samples,1).
        """
        if y.ndim == 2 and y.shape[1] == 1:
            y = y.ravel()

        y_pred = np.sum(weights * X, axis=1)  # Weighted sum across models
        mse = np.mean((y - y_pred)**2)
        return -mse  # negative MSE (GA maximizes fitness)

    def _selection(self, population, fitnesses):
        """
        Roulette Wheel Selection (fitness proportional).
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
        Single-point crossover for weighting vectors.
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
        Train the GA on (X_train, y_train). Then predict on (X_test).

        We assume:
          - X_train: (T, Y, X, M) => stacked => (samples, M)
          - y_train: (T, Y, X) => stacked => (samples,)
          - X_test:  (T, Y, X, M) => stacked => (samples, M)
          - y_test:  (T, Y, X) => stacked => (samples,)

        Returns
        -------
        predicted_da : xarray.DataArray with shape (T, Y, X).
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
        Compute tercile probabilities for the given 'hindcast_det' using
        the distribution specified by self.dist_method.
        
        This method is basically the same as in your original code.
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
         1) Standardize input data.
         2) (Re)fit the GA if not already fitted.
         3) Predict for the new year using ensemble weights.
         4) Reverse-standardize the forecast.
         5) Compute tercile probabilities.
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
        Basic Bayesian Model Averaging (BMA) class, supporting either RMSE- or MAE-based priors.

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


class WAS_mme_ELR:
    def __init__(self, elm_kwargs=None):
        if elm_kwargs is None:
            self.elm_kwargs = {}
        else:
            self.elm_kwargs = elm_kwargs     

    def compute_model(self, X_train, y_train, X_test):
        
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

class NonHomogeneousGaussianRegression:
    pass
    
class WAS_mme_ELM:
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


class WAS_mme_EPOELM:
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

        X_train = xc.regrid(X_train,y_train.X,y_train.Y)
        X_test = xc.regrid(X_test,y_train.X,y_train.Y)
        
        # X_train = X_train.fillna(0)
        # y_train = y_train.fillna(0)
        drymask = xc.drymask(
            y_train, dry_threshold=10, quantile_threshold=0.2
                        )
        X_train = X_train*drymask
        X_test = X_test*drymask
        
        model = xc.EPOELM(**self.elm_kwargs) 
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

        hindcast_det = xc.regrid(hindcast_det,Predictant.X,Predictant.Y)
        Predictor_for_year = xc.regrid(Predictor_for_year,Predictant.X,Predictant.Y)

        drymask = xc.drymask(
            Predictant, dry_threshold=10, quantile_threshold=0.2
                        )
        hindcast_det_ = hindcast_det*drymask
        Predictor_for_year = Predictor_for_year*drymask
        
        # hindcast_det_ = hindcast_det.fillna(0)
        # Predictant_ = Predictant.fillna(0)
        # Predictor_for_year_ = Predictor_for_year.fillna(0)
        
        model = xc.EPOELM(**self.elm_kwargs) 
        model.fit(hindcast_det_, Predictant)
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
        return result_ * mask, mask * hindcast_prob.transpose('probability', 'T', 'Y', 'X') #.drop_vars('T').squeeze()

        
    
class WAS_mme_MLP:
    def __init__(self, hidden_layer_sizes=(10, 5), activation='relu', solver='adam',
                 max_iter=200, alpha=0.01, random_state=42, dist_method="gamma"):
        """
        Single-model implementation using MLPRegressor.
        
        Parameters:
          - hidden_layer_sizes: tuple, sizes of hidden layers.
          - activation: string, activation function.
          - solver: string, optimizer to use.
          - max_iter: int, maximum iterations.
          - alpha: float, regularization parameter.
          - random_state: int, seed for reproducibility.
          - dist_method: string, method for tercile probability calculations.
                        Options: 't', 'gamma', 'nonparam', 'normal', 'lognormal'.
        """
        self.hidden_layer_sizes = hidden_layer_sizes
        self.activation = activation
        self.solver = solver
        self.max_iter = max_iter
        self.alpha = alpha
        self.random_state = random_state
        self.dist_method = dist_method
        self.mlp = None

    def compute_model(self, X_train, y_train, X_test, y_test):
        """
        Fit the MLPRegressor on the training data and predict on X_test.
        The input data are expected to be xarray DataArrays with dimensions that include 'T', 'Y', 'X'
        (and optionally 'M'). The predictions are reshaped back to (T, Y, X).
        """
        # Initialize MLPRegressor
        self.mlp = MLPRegressor(hidden_layer_sizes=self.hidden_layer_sizes,
                                activation=self.activation,
                                solver=self.solver,
                                max_iter=self.max_iter,
                                alpha=self.alpha,
                                random_state=self.random_state)
        
        # Extract coordinate variables from X_test
        time = X_test['T']
        lat = X_test['Y']
        lon = X_test['X']
        n_time = len(X_test.coords['T'])
        n_lat  = len(X_test.coords['Y'])
        n_lon  = len(X_test.coords['X'])
        
        # Stack training data
        X_train_stacked = X_train.stack(sample=('T', 'Y', 'X')).transpose('sample', 'M').values
        y_train_stacked = y_train.stack(sample=('T', 'Y', 'X')).transpose('sample', 'M').values
        # train_nan_mask = np.any(np.isnan(X_train_stacked), axis=1) | np.any(np.isnan(y_train_stacked), axis=1)
        train_nan_mask = (np.any(~np.isfinite(X_train_stacked), axis=1) | np.any(~np.isfinite(y_train_stacked), axis=1))
        X_train_clean = X_train_stacked[~train_nan_mask]
        y_train_clean = y_train_stacked[~train_nan_mask]
        
        # Stack testing data
        X_test_stacked = X_test.stack(sample=('T', 'Y', 'X')).transpose('sample', 'M').values
        y_test_stacked = y_test.stack(sample=('T', 'Y', 'X')).transpose('sample', 'M').values
        # test_nan_mask = np.any(np.isnan(X_test_stacked), axis=1) | np.any(np.isnan(y_test_stacked), axis=1)
        test_nan_mask = (np.any(~np.isfinite(X_test_stacked), axis=1) | np.any(~np.isfinite(y_test_stacked), axis=1))
        
        # Fit the MLP model
        self.mlp.fit(X_train_clean, y_train_clean)
        y_pred = self.mlp.predict(X_test_stacked[~test_nan_mask])
        
        # Reconstruct the prediction array (keeping NaN rows intact)
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

    def forecast(self, Predictant, clim_year_start, clim_year_end, hindcast_det, hindcast_det_cross, Predictor_for_year):
        """
        Forecast method that uses a single MLP model to predict for a new year.
        Steps:
         - Standardize the predictor for the target year using hindcast climatology.
         - Fit the MLP (if not already fitted) on standardized hindcast data.
         - Predict for the target year.
         - Reconstruct predictions into original (T, Y, X) shape.
         - Reverse the standardization.
         - Compute tercile probabilities using the chosen distribution.
        """
        # if "M" in Predictant.coords:
        #     Predictant = Predictant.isel(M=0).drop_vars('M').squeeze()
        mask = xr.where(~np.isnan(Predictant.isel(T=0,M=0)), 1, np.nan).drop_vars(['T','M']).squeeze().to_numpy()
        
        # Standardize Predictor_for_year using hindcast climatology
        mean_val = hindcast_det.sel(T=slice(str(clim_year_start), str(clim_year_end))).mean(dim='T')
        std_val = hindcast_det.sel(T=slice(str(clim_year_start), str(clim_year_end))).std(dim='T')
        Predictor_for_year_st = (Predictor_for_year - mean_val) / std_val
        
        hindcast_det_st = standardize_timeseries(hindcast_det, clim_year_start, clim_year_end)
        Predictant_st = standardize_timeseries(Predictant, clim_year_start, clim_year_end)
        y_test = Predictant_st.isel(T=[-1])
        
        # For forecast, we use the same single MLP model.
        if self.mlp is None:
            self.mlp = MLPRegressor(hidden_layer_sizes=self.hidden_layer_sizes,
                                    activation=self.activation,
                                    solver=self.solver,
                                    max_iter=self.max_iter,
                                    alpha=self.alpha,
                                    random_state=self.random_state)

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
        # train_nan_mask = np.any(np.isnan(X_train_stacked), axis=1) | np.any(np.isnan(y_train_stacked), axis=1)
        train_nan_mask = (np.any(~np.isfinite(X_train_stacked), axis=1) | np.any(~np.isfinite(y_train_stacked), axis=1))
        
        X_train_clean = X_train_stacked[~train_nan_mask]
        y_train_clean = y_train_stacked[~train_nan_mask]
        
        # Stack testing data
        X_test_stacked = Predictor_for_year_st.stack(sample=('T', 'Y', 'X')).transpose('sample', 'M').values
        y_test_stacked = y_test.stack(sample=('T', 'Y', 'X')).transpose('sample', 'M').values
        # test_nan_mask = np.any(np.isnan(X_test_stacked), axis=1) | np.any(np.isnan(y_test_stacked), axis=1)
        test_nan_mask = (np.any(~np.isfinite(X_test_stacked), axis=1) | np.any(~np.isfinite(y_test_stacked), axis=1))        
        # Fit the MLP model
        self.mlp.fit(X_train_clean, y_train_clean)
        y_pred = self.mlp.predict(X_test_stacked[~test_nan_mask])
        
        # Reconstruct the prediction array (keeping NaN rows intact)
        result = np.empty_like(np.squeeze(y_test_stacked))
        result[np.squeeze(test_nan_mask)] = np.squeeze(y_test_stacked[test_nan_mask])
        result[~np.squeeze(test_nan_mask)] = y_pred
        
        predictions_reshaped = result.reshape(n_time, n_lat, n_lon)
        result_da = xr.DataArray(data=predictions_reshaped,
                                    coords={'T': time, 'Y': lat, 'X': lon},
                                    dims=['T', 'Y', 'X'])* mask

        result_da = reverse_standardize(result_da, Predictant.isel(M=0).drop_vars("M").squeeze(),
                                        clim_year_start, clim_year_end)

        if "M" in Predictant.coords:
            Predictant = Predictant.isel(M=0).drop_vars('M').squeeze()

        year = Predictor_for_year.coords['T'].values.astype('datetime64[Y]').astype(int)[0] + 1970  # Convert from epoch
        T_value_1 = Predictant.isel(T=0).coords['T'].values  # Get the datetime64 value from da1
        month_1 = T_value_1.astype('datetime64[M]').astype(int) % 12 + 1  # Extract month
        new_T_value = np.datetime64(f"{year}-{month_1:02d}-{1:02d}")
        result_da = result_da.assign_coords(T=xr.DataArray([new_T_value], dims=["T"]))
        result_da['T'] = result_da['T'].astype('datetime64[ns]')


        
        # Compute tercile probabilities on the predictions
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
        return result_da * mask, mask * hindcast_prob.transpose('probability', 'T', 'Y', 'X')


class WAS_mme_GradientBoosting:
    def __init__(self, n_estimators=100, learning_rate=0.1, max_depth=3,
                 random_state=42, dist_method="gamma"):
        """
        Single-model implementation using GradientBoostingRegressor.
        
        Parameters:
         - n_estimators: int, number of boosting iterations.
         - learning_rate: float, learning rate.
         - max_depth: int, maximum depth of individual regression estimators.
         - random_state: int, for reproducibility.
         - dist_method: string, method for tercile probability calculations.
                        Options: 't', 'gamma', 'nonparam', 'normal', 'lognormal'.
        """
        self.n_estimators = n_estimators
        self.learning_rate = learning_rate
        self.max_depth = max_depth
        self.random_state = random_state
        self.dist_method = dist_method
        
        self.gb = None

    def compute_model(self, X_train, y_train, X_test, y_test):
        """
        Fit the GradientBoostingRegressor on the training data and predict on X_test.
        The input data (xarray DataArrays) are assumed to have dimensions including 'T', 'Y', 'X'.
        The predictions are then reshaped back to (T, Y, X).
        """
        # Initialize the model
        self.gb = GradientBoostingRegressor(n_estimators=self.n_estimators,
                                            learning_rate=self.learning_rate,
                                            max_depth=self.max_depth,
                                            random_state=self.random_state)
        
        # Extract coordinate variables from X_test
        time = X_test['T']
        lat = X_test['Y']
        lon = X_test['X']
        n_time = len(X_test.coords['T'])
        n_lat  = len(X_test.coords['Y'])
        n_lon  = len(X_test.coords['X'])
        
        # Stack training data
        X_train_stacked = X_train.stack(sample=('T', 'Y', 'X')).transpose('sample', 'M').values
        y_train_stacked = y_train.stack(sample=('T', 'Y', 'X')).transpose('sample', 'M').values
        # train_nan_mask = np.any(np.isnan(X_train_stacked), axis=1) | np.any(np.isnan(y_train_stacked), axis=1)
        train_nan_mask = (np.any(~np.isfinite(X_train_stacked), axis=1) | np.any(~np.isfinite(y_train_stacked), axis=1))
        X_train_clean = X_train_stacked[~train_nan_mask]
        y_train_clean = y_train_stacked[~train_nan_mask]
        
        # Stack testing data
        X_test_stacked = X_test.stack(sample=('T', 'Y', 'X')).transpose('sample', 'M').values
        y_test_stacked = y_test.stack(sample=('T', 'Y', 'X')).transpose('sample', 'M').values
        # test_nan_mask = np.any(np.isnan(X_test_stacked), axis=1) | np.any(np.isnan(y_test_stacked), axis=1)
        test_nan_mask = (np.any(~np.isfinite(X_test_stacked), axis=1) | np.any(~np.isfinite(y_test_stacked), axis=1))
        
        # Fit the model and predict on non-NaN testing data
        self.gb.fit(X_train_clean, y_train_clean)
        y_pred = self.gb.predict(X_test_stacked[~test_nan_mask])
        
        # Reconstruct predictions (leaving NaN rows unchanged)
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
                 hindcast_det_cross, Predictor_for_year):
        """
        Forecast method using a single MLP model.
        Steps:
         - Standardize the predictor for the target year using hindcast climatology.
         - Fit the MLP (if not already fitted) on standardized hindcast data.
         - Predict for the target year.
         - Reconstruct predictions into original (T, Y, X) shape.
         - Reverse the standardization.
         - Compute tercile probabilities using the chosen distribution.
        """
        # if "M" in Predictant.coords:
        #     Predictant = Predictant.isel(M=0).drop_vars('M').squeeze()
        mask = xr.where(~np.isnan(Predictant.isel(T=0)), 1, np.nan).drop_vars(['T']).squeeze().to_numpy()
        
        # Standardize Predictor_for_year using hindcast climatology
        mean_val = hindcast_det.sel(T=slice(str(clim_year_start), str(clim_year_end))).mean(dim='T')
        std_val = hindcast_det.sel(T=slice(str(clim_year_start), str(clim_year_end))).std(dim='T')
        Predictor_for_year_st = (Predictor_for_year - mean_val) / std_val
        
        hindcast_det_st = standardize_timeseries(hindcast_det, clim_year_start, clim_year_end)
        Predictant_st = standardize_timeseries(Predictant, clim_year_start, clim_year_end)
        y_test = Predictant_st.isel(T=[-1])
        

        if self.gb is None:
            self.gb = GradientBoostingRegressor(n_estimators=self.n_estimators,
                                                learning_rate=self.learning_rate,
                                                max_depth=self.max_depth,
                                                random_state=self.random_state)
        
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
        # train_nan_mask = np.any(np.isnan(X_train_stacked), axis=1) | np.any(np.isnan(y_train_stacked), axis=1)
        train_nan_mask = (np.any(~np.isfinite(X_train_stacked), axis=1) | np.any(~np.isfinite(y_train_stacked), axis=1))
        
        X_train_clean = X_train_stacked[~train_nan_mask]
        y_train_clean = y_train_stacked[~train_nan_mask]
        
        # Stack testing data
        X_test_stacked = Predictor_for_year_st.stack(sample=('T', 'Y', 'X')).transpose('sample', 'M').values
        y_test_stacked = y_test.stack(sample=('T', 'Y', 'X')).transpose('sample', 'M').values
        # test_nan_mask = np.any(np.isnan(X_test_stacked), axis=1) | np.any(np.isnan(y_test_stacked), axis=1)
        test_nan_mask = (np.any(~np.isfinite(X_test_stacked), axis=1) | np.any(~np.isfinite(y_test_stacked), axis=1))
        
        # Fit the MLP model
        self.gb.fit(X_train_clean, y_train_clean)
        y_pred = self.gb.predict(X_test_stacked[~test_nan_mask])
        
        # Reconstruct the prediction array (keeping NaN rows intact)
        result = np.empty_like(np.squeeze(y_test_stacked))
        result[np.squeeze(test_nan_mask)] = np.squeeze(y_test_stacked[test_nan_mask])
        result[~np.squeeze(test_nan_mask)] = y_pred
        
        predictions_reshaped = result.reshape(n_time, n_lat, n_lon)
        result_da = xr.DataArray(data=predictions_reshaped,
                                    coords={'T': time, 'Y': lat, 'X': lon},
                                    dims=['T', 'Y', 'X'])* mask

        result_da = reverse_standardize(result_da, Predictant.isel(M=0).drop_vars("M").squeeze(),
                                        clim_year_start, clim_year_end)
        if "M" in Predictant.coords:
            Predictant = Predictant.isel(M=0).drop_vars('M').squeeze()
        year = Predictor_for_year.coords['T'].values.astype('datetime64[Y]').astype(int)[0] + 1970  # Convert from epoch
        T_value_1 = Predictant.isel(T=0).coords['T'].values  # Get the datetime64 value from da1
        month_1 = T_value_1.astype('datetime64[M]').astype(int) % 12 + 1  # Extract month
        new_T_value = np.datetime64(f"{year}-{month_1:02d}-{1:02d}")
        result_da = result_da.assign_coords(T=xr.DataArray([new_T_value], dims=["T"]))
        result_da['T'] = result_da['T'].astype('datetime64[ns]')        
        
        # Compute tercile probabilities on the predictions
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
        return result_da * mask, mask * hindcast_prob.drop_vars('T').squeeze().transpose('probability', 'Y', 'X')



class WAS_mme_XGBoosting:
    def __init__(self, n_estimators=100, learning_rate=0.1, max_depth=3,
                 random_state=42, dist_method="gamma"):
        """
        Single-model implementation using XGBRegressor.
        
        Parameters:
         - n_estimators: int, number of boosting rounds.
         - learning_rate: float, learning rate.
         - max_depth: int, maximum tree depth.
         - random_state: int, seed for reproducibility.
         - dist_method: string, method for tercile probability calculations.
                        Options: 't', 'gamma', 'nonparam', 'normal', 'lognormal'.
        """
        self.n_estimators = n_estimators
        self.learning_rate = learning_rate
        self.max_depth = max_depth
        self.random_state = random_state
        self.dist_method = dist_method
        
        self.xgb = None

    def compute_model(self, X_train, y_train, X_test, y_test):
        """
        Fit the XGBRegressor on the training data and predict on X_test.
        The input data are expected to be xarray DataArrays with dimensions that include 'T', 'Y', 'X'
        (and optionally 'M'). Predictions are reshaped back into (T, Y, X).
        """
        # Initialize the XGBRegressor
        self.xgb = XGBRegressor(n_estimators=self.n_estimators,
                                learning_rate=self.learning_rate,
                                max_depth=self.max_depth,
                                random_state=self.random_state,
                                verbosity=0)
        
        # Extract coordinate variables from X_test
        time = X_test['T']
        lat = X_test['Y']
        lon = X_test['X']
        n_time = len(X_test.coords['T'])
        n_lat  = len(X_test.coords['Y'])
        n_lon  = len(X_test.coords['X'])
        
        # Stack training data
        X_train_stacked = X_train.stack(sample=('T', 'Y', 'X')).transpose('sample', 'M').values
        y_train_stacked = y_train.stack(sample=('T', 'Y', 'X')).transpose('sample', 'M').values
        # train_nan_mask = np.any(np.isnan(X_train_stacked), axis=1) | np.any(np.isnan(y_train_stacked), axis=1)
        train_nan_mask = (np.any(~np.isfinite(X_train_stacked), axis=1) | np.any(~np.isfinite(y_train_stacked), axis=1))
        X_train_clean = X_train_stacked[~train_nan_mask]
        y_train_clean = y_train_stacked[~train_nan_mask]
        
        # Stack testing data
        X_test_stacked = X_test.stack(sample=('T', 'Y', 'X')).transpose('sample', 'M').values
        y_test_stacked = y_test.stack(sample=('T', 'Y', 'X')).transpose('sample', 'M').values
        # test_nan_mask = np.any(np.isnan(X_test_stacked), axis=1) | np.any(np.isnan(y_test_stacked), axis=1)
        test_nan_mask = (np.any(~np.isfinite(X_test_stacked), axis=1) | np.any(~np.isfinite(y_test_stacked), axis=1))

        
        # Fit the XGBRegressor and predict on non-NaN testing rows
        self.xgb.fit(X_train_clean, y_train_clean)
        y_pred = self.xgb.predict(X_test_stacked[~test_nan_mask])
        
        # Reconstruct predictions: keep rows with NaNs unchanged
        result = np.empty_like(np.squeeze(y_test_stacked))
        result[np.squeeze(test_nan_mask)] = np.squeeze(y_test_stacked[test_nan_mask])
        result[~np.squeeze(test_nan_mask)] = y_pred
        
        predictions_reshaped = result.reshape(n_time, n_lat, n_lon)
        predicted_da = xr.DataArray(data=predictions_reshaped,
                                    coords={'T': time, 'Y': lat, 'X': lon},
                                    dims=['T', 'Y', 'X'])
        return predicted_da

    # def compute_model(self, X_train, y_train, X_test, y_test):
    #     """
    #     Fit an XGBRegressor to spatiotemporal training data and predict on the test data.
        
    #     Parameters
    #     ----------
    #     X_train, y_train : xarray.DataArray
    #         Training features and target values with dims ('T', 'Y', 'X', 'M') or ('T', 'Y', 'X') if M=1.
        
    #     X_test, y_test : xarray.DataArray
    #         Test features and target values with the same structure.
        
    #     Returns
    #     -------
    #     predicted_da : xarray.DataArray
    #         Predictions in the same shape as y_test, with coords (T, Y, X).
    #     """
    #     # === Ensure model hyperparameters are valid ===
    #     n_estimators = int(self.n_estimators) if hasattr(self, 'n_estimators') else 100
    #     learning_rate = float(self.learning_rate) if hasattr(self, 'learning_rate') else 0.1
    #     max_depth = int(self.max_depth) if hasattr(self, 'max_depth') else 3
    #     random_state = int(self.random_state) if hasattr(self, 'random_state') else 42
    
    #     # === Initialize model ===
    #     self.xgb = XGBRegressor(
    #         n_estimators=n_estimators,
    #         learning_rate=learning_rate,
    #         max_depth=max_depth,
    #         random_state=random_state,
    #         verbosity=0
    #     )
    
    #     # === Extract coordinates for output reconstruction ===
    #     time = X_test['T']
    #     lat = X_test['Y']
    #     lon = X_test['X']
    #     n_time, n_lat, n_lon = len(time), len(lat), len(lon)
    
    #     # === Stack training data ===
    #     X_train_stacked = X_train.stack(sample=('T', 'Y', 'X')).transpose('sample', ...).values
    #     y_train_stacked = y_train.stack(sample=('T', 'Y', 'X')).transpose('sample', ...).values
    
    #     train_nan_mask = np.any(np.isnan(X_train_stacked), axis=1) | np.any(np.isnan(y_train_stacked), axis=1)
    #     X_train_clean = X_train_stacked[~train_nan_mask]
    #     y_train_clean = y_train_stacked[~train_nan_mask].ravel()
    
    #     # === Stack test data ===
    #     X_test_stacked = X_test.stack(sample=('T', 'Y', 'X')).transpose('sample', ...).values
    #     y_test_stacked = y_test.stack(sample=('T', 'Y', 'X')).transpose('sample', ...).values
    #     test_nan_mask = np.any(np.isnan(X_test_stacked), axis=1) | np.any(np.isnan(y_test_stacked), axis=1)
    
    #     # === Fit model and predict ===
    #     self.xgb.fit(X_train_clean, y_train_clean)
    #     y_pred = self.xgb.predict(X_test_stacked[~test_nan_mask])
    
    #     # === Reconstruct prediction array ===
    #     result = np.full(y_test_stacked.shape, np.nan)
    #     result[~test_nan_mask] = y_pred.reshape(-1, 1)
    
    #     predictions_reshaped = result.reshape(n_time, n_lat, n_lon)
    
    #     predicted_da = xr.DataArray(
    #         data=predictions_reshaped,
    #         coords={'T': time, 'Y': lat, 'X': lon},
    #         dims=['T', 'Y', 'X']
    #     )
    
    #     return predicted_da


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
                 hindcast_det_cross, Predictor_for_year):
        """
        Forecast method using a single XGBoost model.
        Steps:
         - Standardize the predictor for the target year using hindcast climatology.
         - Fit the XGBoost model on standardized hindcast data.
         - Predict for the target year.
         - Reconstruct predictions into original (T, Y, X) shape.
         - Reverse the standardization.
         - Compute tercile probabilities using the chosen distribution.
        """
        # if "M" in Predictant.coords:
        #     Predictant = Predictant.isel(M=0).drop_vars('M').squeeze()
        mask = xr.where(~np.isnan(Predictant.isel(T=0)), 1, np.nan).drop_vars(['T']).squeeze().to_numpy()
        
        # Standardize Predictor_for_year using hindcast climatology
        mean_val = hindcast_det.sel(T=slice(str(clim_year_start), str(clim_year_end))).mean(dim='T')
        std_val = hindcast_det.sel(T=slice(str(clim_year_start), str(clim_year_end))).std(dim='T')
        Predictor_for_year_st = (Predictor_for_year - mean_val) / std_val
        
        hindcast_det_st = standardize_timeseries(hindcast_det, clim_year_start, clim_year_end)
        Predictant_st = standardize_timeseries(Predictant, clim_year_start, clim_year_end)
        y_test = Predictant_st.isel(T=[-1])
        
        # Use the same single XGBoost model for forecasting.
        if self.xgb is None:
            self.xgb = XGBRegressor(n_estimators=self.n_estimators,
                                    learning_rate=self.learning_rate,
                                    max_depth=self.max_depth,
                                    random_state=self.random_state,
                                    verbosity=0)
        
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
        # train_nan_mask = np.any(np.isnan(X_train_stacked), axis=1) | np.any(np.isnan(y_train_stacked), axis=1)
        train_nan_mask = (np.any(~np.isfinite(X_train_stacked), axis=1) | np.any(~np.isfinite(y_train_stacked), axis=1))
        
        X_train_clean = X_train_stacked[~train_nan_mask]
        y_train_clean = y_train_stacked[~train_nan_mask]
        
        # Stack testing data
        X_test_stacked = Predictor_for_year_st.stack(sample=('T', 'Y', 'X')).transpose('sample', 'M').values
        y_test_stacked = y_test.stack(sample=('T', 'Y', 'X')).transpose('sample', 'M').values
        # test_nan_mask = np.any(np.isnan(X_test_stacked), axis=1) | np.any(np.isnan(y_test_stacked), axis=1)
        test_nan_mask = (np.any(~np.isfinite(X_test_stacked), axis=1) | np.any(~np.isfinite(y_test_stacked), axis=1))
        
        # Fit the MLP model
        self.xgb.fit(X_train_clean, y_train_clean)
        y_pred = self.xgb.predict(X_test_stacked[~test_nan_mask])
        
        # Reconstruct the prediction array (keeping NaN rows intact)
        result = np.empty_like(np.squeeze(y_test_stacked))
        result[np.squeeze(test_nan_mask)] = np.squeeze(y_test_stacked[test_nan_mask])
        result[~np.squeeze(test_nan_mask)] = y_pred
        
        predictions_reshaped = result.reshape(n_time, n_lat, n_lon)
        result_da = xr.DataArray(data=predictions_reshaped,
                                    coords={'T': time, 'Y': lat, 'X': lon},
                                    dims=['T', 'Y', 'X'])* mask

        result_da = reverse_standardize(result_da, Predictant.isel(M=0).drop_vars("M").squeeze(),
                                        clim_year_start, clim_year_end)
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
        return result_da * mask, mask * hindcast_prob.transpose('probability', 'T','Y', 'X')


class WAS_mme_AdaBoost:
    def __init__(self, n_estimators=50, learning_rate=0.1, random_state=42, dist_method="gamma"):
        """
        Single-model implementation using AdaBoostRegressor.
        
        Parameters:
          - n_estimators: int, the number of boosting iterations.
          - learning_rate: float, the learning rate.
          - random_state: int, seed for reproducibility.
          - dist_method: string, method for tercile probability calculations.
                        Options: 't', 'gamma', 'nonparam', 'normal', 'lognormal'.
        """
        self.n_estimators = n_estimators
        self.learning_rate = learning_rate
        self.random_state = random_state
        self.dist_method = dist_method
        
        self.adaboost = None

    def compute_model(self, X_train, y_train, X_test, y_test):
        """
        Fit the AdaBoostRegressor on the training data and predict on X_test.
        The input data (xarray DataArrays) must include dimensions 'T', 'Y', 'X'
        (and optionally 'M'). Predictions are reshaped back to (T, Y, X).
        """
        # Initialize the AdaBoostRegressor
        self.adaboost = AdaBoostRegressor(n_estimators=self.n_estimators,
                                          learning_rate=self.learning_rate,
                                          random_state=self.random_state)
        
        # Extract coordinate variables from X_test
        time = X_test['T']
        lat = X_test['Y']
        lon = X_test['X']
        n_time = len(X_test.coords['T'])
        n_lat  = len(X_test.coords['Y'])
        n_lon  = len(X_test.coords['X'])
        
        # Stack training data
        X_train_stacked = X_train.stack(sample=('T', 'Y', 'X')).transpose('sample', 'M').values
        y_train_stacked = y_train.stack(sample=('T', 'Y', 'X')).transpose('sample', 'M').values
        # train_nan_mask = np.any(np.isnan(X_train_stacked), axis=1) | np.any(np.isnan(y_train_stacked), axis=1)
        train_nan_mask = (np.any(~np.isfinite(X_train_stacked), axis=1) | np.any(~np.isfinite(y_train_stacked), axis=1))
        X_train_clean = X_train_stacked[~train_nan_mask]
        y_train_clean = y_train_stacked[~train_nan_mask]
        
        # Stack testing data
        X_test_stacked = X_test.stack(sample=('T', 'Y', 'X')).transpose('sample', 'M').values
        y_test_stacked = y_test.stack(sample=('T', 'Y', 'X')).transpose('sample', 'M').values
        # test_nan_mask = np.any(np.isnan(X_test_stacked), axis=1) | np.any(np.isnan(y_test_stacked), axis=1)
        test_nan_mask = (np.any(~np.isfinite(X_test_stacked), axis=1) | np.any(~np.isfinite(y_test_stacked), axis=1))
        
        # Fit the AdaBoost model and predict on non-NaN testing rows
        self.adaboost.fit(X_train_clean, y_train_clean)
        y_pred = self.adaboost.predict(X_test_stacked[~test_nan_mask])
        
        # Reconstruct prediction array (leaving rows with NaNs unchanged)
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
                 hindcast_det_cross, Predictor_for_year):
        """
        Forecast method using a single XGBoost model.
        
        Steps:
         - Standardize the predictor for the target year using hindcast climatology.
         - Fit the XGBoost model on the standardized hindcast data.
         - Predict for the target year.
         - Reconstruct predictions into the original (T, Y, X) shape.
         - Reverse the standardization.
         - Compute tercile probabilities using the chosen distribution.
        """
        # Create a land mask from the first time step
        # if "M" in Predictant.coords:
        #     Predictant = Predictant.isel(M=0).drop_vars('M').squeeze()
        mask = xr.where(~np.isnan(Predictant.isel(T=0)), 1, np.nan).drop_vars(['T']).squeeze().to_numpy()
        
        # Standardize Predictor_for_year using hindcast climatology
        mean_val = hindcast_det.sel(T=slice(str(clim_year_start), str(clim_year_end))).mean(dim='T')
        std_val = hindcast_det.sel(T=slice(str(clim_year_start), str(clim_year_end))).std(dim='T')
        Predictor_for_year_st = (Predictor_for_year - mean_val) / std_val
        
        # Standardize hindcast and predictant
        hindcast_det_st = standardize_timeseries(hindcast_det, clim_year_start, clim_year_end)
        Predictant_st = standardize_timeseries(Predictant, clim_year_start, clim_year_end)
        y_test = Predictant_st.isel(T=[-1])
        
        # Use the same XGBoost model for forecasting.
        if self.adaboost is None:
            self.adaboost = AdaBoostRegressor(n_estimators=self.n_estimators,
                                              learning_rate=self.learning_rate,
                                              random_state=self.random_state)
        
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
        # train_nan_mask = np.any(np.isnan(X_train_stacked), axis=1) | np.any(np.isnan(y_train_stacked), axis=1)
        train_nan_mask = (np.any(~np.isfinite(X_train_stacked), axis=1) | np.any(~np.isfinite(y_train_stacked), axis=1))
        
        X_train_clean = X_train_stacked[~train_nan_mask]
        y_train_clean = y_train_stacked[~train_nan_mask]
        
        # Stack testing data
        X_test_stacked = Predictor_for_year_st.stack(sample=('T', 'Y', 'X')).transpose('sample', 'M').values
        y_test_stacked = y_test.stack(sample=('T', 'Y', 'X')).transpose('sample', 'M').values
        # test_nan_mask = np.any(np.isnan(X_test_stacked), axis=1) | np.any(np.isnan(y_test_stacked), axis=1)
        test_nan_mask = (np.any(~np.isfinite(X_test_stacked), axis=1) | np.any(~np.isfinite(y_test_stacked), axis=1))
        
        # Fit the MLP model
        self.adaboost.fit(X_train_clean, y_train_clean)
        y_pred = self.adaboost.predict(X_test_stacked[~test_nan_mask])
        
        # Reconstruct the prediction array (keeping NaN rows intact)
        result = np.empty_like(np.squeeze(y_test_stacked))
        result[np.squeeze(test_nan_mask)] = np.squeeze(y_test_stacked[test_nan_mask])
        result[~np.squeeze(test_nan_mask)] = y_pred
        
        predictions_reshaped = result.reshape(n_time, n_lat, n_lon)
        result_da = xr.DataArray(data=predictions_reshaped,
                                    coords={'T': time, 'Y': lat, 'X': lon},
                                    dims=['T', 'Y', 'X'])* mask

        result_da = reverse_standardize(result_da, Predictant.isel(M=0).drop_vars("M").squeeze(),
                                        clim_year_start, clim_year_end)
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
        return result_da * mask, mask * hindcast_prob.transpose('probability', 'T','Y', 'X')



class WAS_mme_LGBM_Boosting:
    def __init__(self, n_estimators=100, learning_rate=0.1, max_depth=-1,
                 random_state=42, dist_method="gamma"):
        """
        Single-model implementation using LGBMRegressor.
        
        Parameters:
          - n_estimators: int, number of boosting iterations.
          - learning_rate: float, learning rate.
          - max_depth: int, maximum tree depth (set -1 for no limit).
          - random_state: int, seed for reproducibility.
          - dist_method: string, method for tercile probability calculations.
                        Options: 't', 'gamma', 'nonparam', 'normal', 'lognormal'.
        """
        self.n_estimators = n_estimators
        self.learning_rate = learning_rate
        self.max_depth = max_depth
        self.random_state = random_state
        self.dist_method = dist_method
        
        self.lgbm = None

    def compute_model(self, X_train, y_train, X_test, y_test):
        """
        Fit the LGBMRegressor on the training data and predict on X_test.
        Input data (xarray DataArrays) should include dimensions 'T', 'Y', 'X'
        (and optionally 'M'). The predictions are reshaped back into (T, Y, X).
        """
        # Initialize LGBMRegressor
        self.lgbm = LGBMRegressor(n_estimators=self.n_estimators,
                                  learning_rate=self.learning_rate,
                                  max_depth=self.max_depth,
                                  random_state=self.random_state)
        
        # Extract coordinate variables from X_test
        time = X_test['T']
        lat = X_test['Y']
        lon = X_test['X']
        n_time = len(X_test.coords['T'])
        n_lat  = len(X_test.coords['Y'])
        n_lon  = len(X_test.coords['X'])
        
        # Stack training data
        X_train_stacked = X_train.stack(sample=('T', 'Y', 'X')).transpose('sample', 'M').values
        y_train_stacked = y_train.stack(sample=('T', 'Y', 'X')).transpose('sample', 'M').values
        # train_nan_mask = np.any(np.isnan(X_train_stacked), axis=1) | np.any(np.isnan(y_train_stacked), axis=1)
        train_nan_mask = (np.any(~np.isfinite(X_train_stacked), axis=1) | np.any(~np.isfinite(y_train_stacked), axis=1))
        X_train_clean = X_train_stacked[~train_nan_mask]
        y_train_clean = y_train_stacked[~train_nan_mask]
        
        # Stack testing data
        X_test_stacked = X_test.stack(sample=('T', 'Y', 'X')).transpose('sample', 'M').values
        y_test_stacked = y_test.stack(sample=('T', 'Y', 'X')).transpose('sample', 'M').values
        # test_nan_mask = np.any(np.isnan(X_test_stacked), axis=1) | np.any(np.isnan(y_test_stacked), axis=1)
        test_nan_mask = (np.any(~np.isfinite(X_test_stacked), axis=1) | np.any(~np.isfinite(y_test_stacked), axis=1))
        
        # Fit the model and predict on non-NaN rows
        self.lgbm.fit(X_train_clean, y_train_clean)
        y_pred = self.lgbm.predict(X_test_stacked[~test_nan_mask])
        
        # Reconstruct predictions, leaving rows with NaNs intact
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
        Compute tercile probabilities using the chosen distribution method.
        Predictant is expected to be an xarray DataArray with dims (T, Y, X).
        """
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
                 hindcast_det_cross, Predictor_for_year):
        """
        Forecast method using a single XGBoost model.
        
        Steps:
         - Standardize the predictor for the target year using hindcast climatology.
         - Fit the XGBoost model on standardized hindcast data.
         - Predict for the target year.
         - Reconstruct predictions into the original (T, Y, X) shape.
         - Reverse the standardization.
         - Compute tercile probabilities using the chosen distribution.
        """
        # # Create a mask from the first time step (excluding NaNs)
        # if "M" in Predictant.coords:
        #     Predictant = Predictant.isel(M=0).drop_vars('M').squeeze()
        mask = xr.where(~np.isnan(Predictant.isel(T=0)), 1, np.nan).drop_vars(['T']).squeeze().to_numpy()
        
        # Standardize Predictor_for_year using hindcast climatology
        mean_val = hindcast_det.sel(T=slice(str(clim_year_start), str(clim_year_end))).mean(dim='T')
        std_val = hindcast_det.sel(T=slice(str(clim_year_start), str(clim_year_end))).std(dim='T')
        Predictor_for_year_st = (Predictor_for_year - mean_val) / std_val
        
        # Standardize hindcast and predictant
        hindcast_det_st = standardize_timeseries(hindcast_det, clim_year_start, clim_year_end)
        Predictant_st = standardize_timeseries(Predictant, clim_year_start, clim_year_end)
        y_test = Predictant_st.isel(T=[-1])
        
        # Use the same XGBoost model for forecasting.
        if self.lgbm is None:
            self.lgbm = LGBMRegressor(n_estimators=self.n_estimators,
                                      learning_rate=self.learning_rate,
                                      max_depth=self.max_depth,
                                      random_state=self.random_state)
        
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
        # train_nan_mask = np.any(np.isnan(X_train_stacked), axis=1) | np.any(np.isnan(y_train_stacked), axis=1)
        train_nan_mask = (np.any(~np.isfinite(X_train_stacked), axis=1) | np.any(~np.isfinite(y_train_stacked), axis=1))
        
        X_train_clean = X_train_stacked[~train_nan_mask]
        y_train_clean = y_train_stacked[~train_nan_mask]
        
        # Stack testing data
        X_test_stacked = Predictor_for_year_st.stack(sample=('T', 'Y', 'X')).transpose('sample', 'M').values
        y_test_stacked = y_test.stack(sample=('T', 'Y', 'X')).transpose('sample', 'M').values
        # test_nan_mask = np.any(np.isnan(X_test_stacked), axis=1) | np.any(np.isnan(y_test_stacked), axis=1)
        test_nan_mask = (np.any(~np.isfinite(X_test_stacked), axis=1) | np.any(~np.isfinite(y_test_stacked), axis=1))        
        # Fit the MLP model
        self.lgbm.fit(X_train_clean, y_train_clean)
        y_pred = self.lgbm.predict(X_test_stacked[~test_nan_mask])
        
        # Reconstruct the prediction array (keeping NaN rows intact)
        result = np.empty_like(np.squeeze(y_test_stacked))
        result[np.squeeze(test_nan_mask)] = np.squeeze(y_test_stacked[test_nan_mask])
        result[~np.squeeze(test_nan_mask)] = y_pred
        
        predictions_reshaped = result.reshape(n_time, n_lat, n_lon)
        result_da = xr.DataArray(data=predictions_reshaped,
                                    coords={'T': time, 'Y': lat, 'X': lon},
                                    dims=['T', 'Y', 'X'])* mask

        result_da = reverse_standardize(result_da, Predictant.isel(M=0).drop_vars("M").squeeze(),
                                        clim_year_start, clim_year_end)
        if "M" in Predictant.coords:
            Predictant = Predictant.isel(M=0).drop_vars('M').squeeze()
            
        year = Predictor_for_year.coords['T'].values.astype('datetime64[Y]').astype(int)[0] + 1970  # Convert from epoch
        T_value_1 = Predictant.isel(T=0).coords['T'].values  # Get the datetime64 value from da1
        month_1 = T_value_1.astype('datetime64[M]').astype(int) % 12 + 1  # Extract month
        new_T_value = np.datetime64(f"{year}-{month_1:02d}-{1:02d}")
        result_da = result_da.assign_coords(T=xr.DataArray([new_T_value], dims=["T"]))
        result_da['T'] = result_da['T'].astype('datetime64[ns]')
        
        # Compute tercile probabilities on predictions using the chosen distribution method
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
        return result_da * mask, mask * hindcast_prob.transpose('probability', 'T', 'Y', 'X')


    
class WAS_mme_Stack_MLP_RF:
    def __init__(self, 
                 hidden_layer_sizes=(10, 5),
                 activation='relu',
                 max_iter=200, 
                 solver='adam',
                 random_state=42,
                 alpha=0.01, 
                 n_estimators=100, 
                 dist_method="gamma"):
        self.hidden_layer_sizes = hidden_layer_sizes
        self.activation = activation
        self.solver = solver
        self.max_iter = max_iter
        self.random_state = random_state
        self.alpha = alpha
        self.n_estimators = n_estimators
        self.dist_method = dist_method

    def compute_model(self, X_train, y_train, X_test, y_test):
        # Initialize the base models (MLP and Random Forest)
        self.base_models = [
            ('mlp', MLPRegressor(
                hidden_layer_sizes=self.hidden_layer_sizes,
                activation=self.activation,
                solver=self.solver,
                max_iter=self.max_iter,
                random_state=self.random_state,
                alpha=self.alpha
            )),
            ('rf', RandomForestRegressor(n_estimators=self.n_estimators))
        ]
        
        # Initialize the meta-model (Linear Regression)
        self.meta_model = LinearRegression()
        
        # Initialize the stacking ensemble
        self.stacking_model = StackingRegressor(
            estimators=self.base_models,
            final_estimator=self.meta_model
        )
        
        # Extract coordinate variables from X_test
        time = X_test['T']
        lat = X_test['Y']
        lon = X_test['X']
        n_time = len(X_test.coords['T'])
        n_lat  = len(X_test.coords['Y'])
        n_lon  = len(X_test.coords['X'])
        
        # Stack training data
        X_train_stacked = X_train.stack(sample=('T', 'Y', 'X')).transpose('sample', 'M').values
        y_train_stacked = y_train.stack(sample=('T', 'Y', 'X')).transpose('sample', 'M').values
        # train_nan_mask = np.any(np.isnan(X_train_stacked), axis=1) | np.any(np.isnan(y_train_stacked), axis=1)
        train_nan_mask = (np.any(~np.isfinite(X_train_stacked), axis=1) | np.any(~np.isfinite(y_train_stacked), axis=1))
        X_train_clean = X_train_stacked[~train_nan_mask]
        y_train_clean = y_train_stacked[~train_nan_mask]
        
        # Stack testing data
        X_test_stacked = X_test.stack(sample=('T', 'Y', 'X')).transpose('sample', 'M').values
        y_test_stacked = y_test.stack(sample=('T', 'Y', 'X')).transpose('sample', 'M').values
        # test_nan_mask = np.any(np.isnan(X_test_stacked), axis=1) | np.any(np.isnan(y_test_stacked), axis=1)
        test_nan_mask = (np.any(~np.isfinite(X_test_stacked), axis=1) | np.any(~np.isfinite(y_test_stacked), axis=1))
        
        # Fit the stacking ensemble only on rows without NaNs
        self.stacking_model.fit(X_train_clean, y_train_clean)
        y_pred_test = self.stacking_model.predict(X_test_stacked[~test_nan_mask])
        
        # Rebuild the predictions into the original shape,
        # leaving NaN rows intact.
        # Reconstruct predictions, leaving rows with NaNs intact
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
        This method extracts the climatology terciles from Predictant over the period
        [clim_year_start, clim_year_end], computes an error variance (or uses error samples),
        and then applies the chosen probability function.
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
        # For error variance, here we use the difference between Predictant and hindcast_det
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
                dask_gufunc_kwargs={'output_sizes': {'probability': 3}, "allow_rechunk": True},
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
                dask_gufunc_kwargs={'output_sizes': {'probability': 3}, "allow_rechunk": True},
            )
        elif self.dist_method == "nonparam":
            # For nonparametric, assume hindcast_det contains error samples
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
                dask_gufunc_kwargs={'output_sizes': {'probability': 3}, "allow_rechunk": True},
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
                dask_gufunc_kwargs={'output_sizes': {'probability': 3}, "allow_rechunk": True},
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
                dask_gufunc_kwargs={'output_sizes': {'probability': 3}, "allow_rechunk": True},
            )
        else:
            raise ValueError(f"Invalid dist_method: {self.dist_method}.")
        
        hindcast_prob = hindcast_prob.assign_coords(probability=('probability', ['PB', 'PN', 'PA']))
        return hindcast_prob.transpose('probability', 'T', 'Y', 'X')

    @staticmethod
    def _reshape_and_filter_data(da):
        """
        Helper: stack the DataArray from (T, Y, X[, M]) to (n_samples, n_features)
        and remove any rows with NaNs.
        """
        da_stacked = da.stack(sample=('T', 'Y', 'X'))
        if 'M' in da.dims:
            da_stacked = da_stacked.transpose('sample', 'M')
        else:
            da_stacked = da_stacked.transpose('sample')
        da_values = da_stacked.values
        nan_mask = np.any(np.isnan(da_values), axis=1)
        return da_values[~nan_mask], nan_mask, da_values

    def forecast(self, Predictant, clim_year_start, clim_year_end, hindcast_det, hindcast_det_cross, Predictor_for_year):
        """
        Forecast method that uses the trained stacking ensemble to predict for a new year,
        then computes tercile probabilities.
        """
        # if "M" in Predictant.coords:
        #     Predictant = Predictant.isel(M=0).drop_vars('M').squeeze()
        mask = xr.where(~np.isnan(Predictant.isel(T=0)), 1, np.nan).drop_vars(['T']).squeeze().to_numpy()
        
        # Standardize predictor for the target year using hindcast stats
        mean_val = hindcast_det.sel(T=slice(str(clim_year_start), str(clim_year_end))).mean(dim='T')
        std_val = hindcast_det.sel(T=slice(str(clim_year_start), str(clim_year_end))).std(dim='T')
        Predictor_for_year_st = (Predictor_for_year - mean_val) / std_val
        
        hindcast_det_st = standardize_timeseries(hindcast_det, clim_year_start, clim_year_end)
        Predictant_st = standardize_timeseries(Predictant, clim_year_start, clim_year_end)
        y_test = Predictant_st.isel(T=[-1])
        
        # Initialize stacking ensemble (if not already done)
        self.base_models = [
            ('mlp', MLPRegressor(
                hidden_layer_sizes=self.hidden_layer_sizes,
                activation=self.activation,
                solver=self.solver,
                max_iter=self.max_iter,
                random_state=self.random_state,
                alpha=self.alpha
            )),
            ('rf', RandomForestRegressor(n_estimators=self.n_estimators))
        ]
        self.meta_model = LinearRegression()
        self.stacking_model = StackingRegressor(estimators=self.base_models, final_estimator=self.meta_model)
        
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
        # train_nan_mask = np.any(np.isnan(X_train_stacked), axis=1) | np.any(np.isnan(y_train_stacked), axis=1)
        train_nan_mask = (np.any(~np.isfinite(X_train_stacked), axis=1) | np.any(~np.isfinite(y_train_stacked), axis=1))
        
        X_train_clean = X_train_stacked[~train_nan_mask]
        y_train_clean = y_train_stacked[~train_nan_mask]
        
        # Stack testing data
        X_test_stacked = Predictor_for_year_st.stack(sample=('T', 'Y', 'X')).transpose('sample', 'M').values
        y_test_stacked = y_test.stack(sample=('T', 'Y', 'X')).transpose('sample', 'M').values
        # test_nan_mask = np.any(np.isnan(X_test_stacked), axis=1) | np.any(np.isnan(y_test_stacked), axis=1)
        test_nan_mask = (np.any(~np.isfinite(X_test_stacked), axis=1) | np.any(~np.isfinite(y_test_stacked), axis=1))        
        # Fit the MLP model
        self.stacking_model.fit(X_train_clean, y_train_clean)
        y_pred = self.stacking_model.predict(X_test_stacked[~test_nan_mask])
        
        # Reconstruct the prediction array (keeping NaN rows intact)
        result = np.empty_like(np.squeeze(y_test_stacked))
        result[np.squeeze(test_nan_mask)] = np.squeeze(y_test_stacked[test_nan_mask])
        result[~np.squeeze(test_nan_mask)] = y_pred
        
        predictions_reshaped = result.reshape(n_time, n_lat, n_lon)
        result_da = xr.DataArray(data=predictions_reshaped,
                                    coords={'T': time, 'Y': lat, 'X': lon},
                                    dims=['T', 'Y', 'X'])* mask
        result_da = reverse_standardize(result_da, Predictant.isel(M=0).drop_vars("M").squeeze(),
                                        clim_year_start, clim_year_end)
        if "M" in Predictant.coords:
            Predictant = Predictant.isel(M=0).drop_vars('M').squeeze()
            
        year = Predictor_for_year.coords['T'].values.astype('datetime64[Y]').astype(int)[0] + 1970  # Convert from epoch
        T_value_1 = Predictant.isel(T=0).coords['T'].values  # Get the datetime64 value from da1
        month_1 = T_value_1.astype('datetime64[M]').astype(int) % 12 + 1  # Extract month
        new_T_value = np.datetime64(f"{year}-{month_1:02d}-{1:02d}")
        result_da = result_da.assign_coords(T=xr.DataArray([new_T_value], dims=["T"]))
        result_da['T'] = result_da['T'].astype('datetime64[ns]')
        
        # Compute tercile probabilities on the predictions using the chosen distribution
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
                output_core_dims=[('probability', 'T')],
                output_dtypes=['float'],
                dask_gufunc_kwargs={'output_sizes': {'probability': 3}, "allow_rechunk": True},
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
                output_core_dims=[('probability', 'T')],
                output_dtypes=['float'],
                dask_gufunc_kwargs={'output_sizes': {'probability': 3}, "allow_rechunk": True},
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
                dask_gufunc_kwargs={'output_sizes': {'probability': 3}, "allow_rechunk": True},
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
                dask_gufunc_kwargs={'output_sizes': {'probability': 3}, "allow_rechunk": True},
            )
        elif self.dist_method == "nonparam":
            calc_func = self.calculate_tercile_probabilities_nonparametric
            error_samples = Predictant - hindcast_det_cross
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
        return result_da * mask, mask * hindcast_prob.transpose('probability', 'T', 'Y', 'X')


class WAS_mme_Stack_Lasso_RF_MLP:
    def __init__(self, 
                 lasso_alpha=0.01, 
                 n_estimators=100, 
                 mlp_hidden_layer_sizes=(10, 5), 
                 mlp_activation='relu', 
                 mlp_solver='adam', 
                 mlp_max_iter=200, 
                 mlp_alpha=0.01, 
                 random_state=42,
                dist_method="gamma"):
        """
        Base models: Lasso and RandomForestRegressor.
        Meta-model: MLPRegressor.
        """
        self.lasso_alpha = lasso_alpha
        self.n_estimators = n_estimators
        self.mlp_hidden_layer_sizes = mlp_hidden_layer_sizes
        self.mlp_activation = mlp_activation
        self.mlp_solver = mlp_solver
        self.mlp_max_iter = mlp_max_iter
        self.mlp_alpha = mlp_alpha
        self.random_state = random_state
        self.dist_method = dist_method

        self.base_models = None
        self.meta_model = None
        self.stacking_model = None

    def compute_model(self, X_train, y_train, X_test, y_test):
        # Define base models: Lasso and RandomForestRegressor
        self.base_models = [
            ('lasso', Lasso(alpha=self.lasso_alpha, random_state=self.random_state)),
            ('rf', RandomForestRegressor(n_estimators=self.n_estimators, random_state=self.random_state))
        ]
        # Define meta-model: MLPRegressor
        self.meta_model = MLPRegressor(
            hidden_layer_sizes=self.mlp_hidden_layer_sizes,
            activation=self.mlp_activation,
            solver=self.mlp_solver,
            max_iter=self.mlp_max_iter,
            random_state=self.random_state,
            alpha=self.mlp_alpha
        )
        # Create stacking ensemble
        self.stacking_model = StackingRegressor(
            estimators=self.base_models,
            final_estimator=self.meta_model
        )
        
        # Extract coordinate variables from X_test
        time = X_test['T']
        lat = X_test['Y']
        lon = X_test['X']
        n_time = len(X_test.coords['T'])
        n_lat  = len(X_test.coords['Y'])
        n_lon  = len(X_test.coords['X'])
        
        # Stack training data
        X_train_stacked = X_train.stack(sample=('T', 'Y', 'X')).transpose('sample', 'M').values
        y_train_stacked = y_train.stack(sample=('T', 'Y', 'X')).transpose('sample', 'M').values
        # train_nan_mask = np.any(np.isnan(X_train_stacked), axis=1) | np.any(np.isnan(y_train_stacked), axis=1)
        train_nan_mask = (np.any(~np.isfinite(X_train_stacked), axis=1) | np.any(~np.isfinite(y_train_stacked), axis=1))
        X_train_clean = X_train_stacked[~train_nan_mask]
        y_train_clean = y_train_stacked[~train_nan_mask]
        
        # Stack testing data
        X_test_stacked = X_test.stack(sample=('T', 'Y', 'X')).transpose('sample', 'M').values
        y_test_stacked = y_test.stack(sample=('T', 'Y', 'X')).transpose('sample', 'M').values
        # test_nan_mask = np.any(np.isnan(X_test_stacked), axis=1) | np.any(np.isnan(y_test_stacked), axis=1)
        test_nan_mask = (np.any(~np.isfinite(X_test_stacked), axis=1) | np.any(~np.isfinite(y_test_stacked), axis=1))
        
        # Fit the stacking ensemble only on rows without NaNs
        self.stacking_model.fit(X_train_clean, y_train_clean)
        y_pred_test = self.stacking_model.predict(X_test_stacked[~test_nan_mask])
        
        # Rebuild the predictions into the original shape,
        # leaving NaN rows intact.
        # Reconstruct predictions, leaving rows with NaNs intact
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
        Compute tercile probabilities using the chosen distribution method.
        Predictant should be an xarray DataArray with dims (T, Y, X).
        """
        if "M" in Predictant.coords:
            Predictant = Predictant.isel(M=0).drop_vars('M').squeeze()
        mask = xr.where(~np.isnan(Predictant.isel(T=0)), 1, np.nan).drop_vars(['T']).squeeze().to_numpy()
        
        # Ensure Predictant is in (T, Y, X)
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
                dask_gufunc_kwargs={'output_sizes': {'probability': 3}, "allow_rechunk": True},
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
                dask_gufunc_kwargs={'output_sizes': {'probability': 3}, "allow_rechunk": True},
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
                dask_gufunc_kwargs={'output_sizes': {'probability': 3}, "allow_rechunk": True},
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
                dask_gufunc_kwargs={'output_sizes': {'probability': 3}, "allow_rechunk": True},
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
                dask_gufunc_kwargs={'output_sizes': {'probability': 3}, "allow_rechunk": True},
            )
        else:
            raise ValueError(f"Invalid dist_method: {self.dist_method}.")
        
        hindcast_prob = hindcast_prob.assign_coords(probability=('probability', ['PB', 'PN', 'PA']))
        return hindcast_prob.transpose('probability', 'T', 'Y', 'X')

    @staticmethod
    def _reshape_and_filter_data(da):
        """
        Helper: stack DataArray from (T, Y, X[, M]) to (n_samples, n_features),
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

    def forecast(self, Predictant, clim_year_start, clim_year_end, hindcast_det, hindcast_det_cross, Predictor_for_year):
        """
        Forecast method: standardize the predictor, fit the stacking model,
        make predictions for a new year, reverse standardization, and compute tercile probabilities.
        """
        if "M" in Predictant.coords:
            Predictant = Predictant.isel(M=0).drop_vars('M').squeeze()
        mask = xr.where(~np.isnan(Predictant.isel(T=0)), 1, np.nan).drop_vars(['T']).squeeze().to_numpy()
      
        
        # Standardize predictor for the target year using hindcast stats
        mean_val = hindcast_det.sel(T=slice(str(clim_year_start), str(clim_year_end))).mean(dim='T')
        std_val = hindcast_det.sel(T=slice(str(clim_year_start), str(clim_year_end))).std(dim='T')
        Predictor_for_year_st = (Predictor_for_year - mean_val) / std_val
        
        hindcast_det_st = standardize_timeseries(hindcast_det, clim_year_start, clim_year_end)
        Predictant_st = standardize_timeseries(Predictant, clim_year_start, clim_year_end)
        y_test = Predictant_st.isel(T=[-1])
        
        # Initialize stacking ensemble if not already done
        self.base_models = [
            ('lasso', Lasso(alpha=self.lasso_alpha, random_state=self.random_state)),
            ('rf', RandomForestRegressor(n_estimators=self.n_estimators, random_state=self.random_state))
        ]
        self.meta_model = MLPRegressor(
            hidden_layer_sizes=self.mlp_hidden_layer_sizes,
            activation=self.mlp_activation,
            solver=self.mlp_solver,
            max_iter=self.mlp_max_iter,
            random_state=self.random_state,
            alpha=self.mlp_alpha
        )
        self.stacking_model = StackingRegressor(estimators=self.base_models, final_estimator=self.meta_model)
        
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
        # train_nan_mask = np.any(np.isnan(X_train_stacked), axis=1) | np.any(np.isnan(y_train_stacked), axis=1)
        train_nan_mask = (np.any(~np.isfinite(X_train_stacked), axis=1) | np.any(~np.isfinite(y_train_stacked), axis=1))
        
        X_train_clean = X_train_stacked[~train_nan_mask]
        y_train_clean = y_train_stacked[~train_nan_mask]
        
        # Stack testing data
        X_test_stacked = Predictor_for_year_st.stack(sample=('T', 'Y', 'X')).transpose('sample', 'M').values
        y_test_stacked = y_test.stack(sample=('T', 'Y', 'X')).transpose('sample', 'M').values
        # test_nan_mask = np.any(np.isnan(X_test_stacked), axis=1) | np.any(np.isnan(y_test_stacked), axis=1)
        test_nan_mask = (np.any(~np.isfinite(X_test_stacked), axis=1) | np.any(~np.isfinite(y_test_stacked), axis=1))        
        # Fit the MLP model
        self.stacking_model.fit(X_train_clean, y_train_clean)
        y_pred = self.stacking_model.predict(X_test_stacked[~test_nan_mask])
        
        # Reconstruct the prediction array (keeping NaN rows intact)
        result = np.empty_like(np.squeeze(y_test_stacked))
        result[np.squeeze(test_nan_mask)] = np.squeeze(y_test_stacked[test_nan_mask])
        result[~np.squeeze(test_nan_mask)] = y_pred
        
        predictions_reshaped = result.reshape(n_time, n_lat, n_lon)
        result_da = xr.DataArray(data=predictions_reshaped,
                                    coords={'T': time, 'Y': lat, 'X': lon},
                                    dims=['T', 'Y', 'X'])* mask
        result_da = reverse_standardize(result_da, Predictant.isel(M=0).drop_vars("M").squeeze(),
                                        clim_year_start, clim_year_end)
        if "M" in Predictant.coords:
            Predictant = Predictant.isel(M=0).drop_vars('M').squeeze()
        
        # Compute tercile probabilities using the chosen distribution method
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
                output_core_dims=[('probability', 'T')],
                output_dtypes=['float'],
                dask_gufunc_kwargs={'output_sizes': {'probability': 3}, "allow_rechunk": True},
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
                output_core_dims=[('probability', 'T')],
                output_dtypes=['float'],
                dask_gufunc_kwargs={'output_sizes': {'probability': 3}, "allow_rechunk": True},
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
                dask_gufunc_kwargs={'output_sizes': {'probability': 3}, "allow_rechunk": True},
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
                dask_gufunc_kwargs={'output_sizes': {'probability': 3}, "allow_rechunk": True},
            )
        elif self.dist_method == "nonparam":
            calc_func = self.calculate_tercile_probabilities_nonparametric
            error_samples = Predictant - hindcast_det_cross
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
        return result_da * mask, mask * hindcast_prob.transpose('probability', 'T', 'Y', 'X')


class WAS_mme_Stack_MLP_Ada_Ridge:
    def __init__(self, 
                 hidden_layer_sizes=(10, 5), 
                 activation='relu', 
                 max_iter=200, 
                 solver='adam', 
                 mlp_alpha=0.01,
                 n_estimators_adaboost=50, 
                 ridge_alpha=1.0, 
                 random_state=42,
                 dist_method="gamma"):
        """
        Base models: MLPRegressor and AdaBoostRegressor.
        Meta-model: Ridge.
        
        Parameters:
         - hidden_layer_sizes, activation, max_iter, solver, mlp_alpha: for MLPRegressor.
         - n_estimators_adaboost: for AdaBoostRegressor.
         - ridge_alpha: for Ridge.
         - random_state: seed for reproducibility.
         - dist_method: distribution method for tercile probability calculations. Options: 't','gamma','nonparam','normal','lognormal'.
        """
        self.hidden_layer_sizes = hidden_layer_sizes
        self.activation = activation
        self.max_iter = max_iter
        self.solver = solver
        self.mlp_alpha = mlp_alpha
        self.n_estimators_adaboost = n_estimators_adaboost
        self.ridge_alpha = ridge_alpha
        self.random_state = random_state
        self.dist_method = dist_method

        self.base_models = None
        self.meta_model = None
        self.stacking_model = None

    def compute_model(self, X_train, y_train, X_test, y_test):
        """
        Fit the stacking ensemble using:
         - Base models: MLPRegressor and AdaBoostRegressor.
         - Meta-model: Ridge.
        Data is flattened and NaN rows are excluded for training.
        Predictions are then reshaped back to (T, Y, X).
        """
        # Define base models
        self.base_models = [
            ('mlp', MLPRegressor(
                hidden_layer_sizes=self.hidden_layer_sizes,
                activation=self.activation,
                solver=self.solver,
                max_iter=self.max_iter,
                random_state=self.random_state,
                alpha=self.mlp_alpha
            )),
            ('ada', AdaBoostRegressor(
                n_estimators=self.n_estimators_adaboost,
                random_state=self.random_state
            ))
        ]
        # Define meta-model (Ridge)
        self.meta_model = Ridge(alpha=self.ridge_alpha, random_state=self.random_state)
        
        # Build stacking ensemble
        self.stacking_model = StackingRegressor(
            estimators=self.base_models,
            final_estimator=self.meta_model
        )
        
        # Extract coordinate variables from X_test
        time = X_test['T']
        lat = X_test['Y']
        lon = X_test['X']
        n_time = len(X_test.coords['T'])
        n_lat  = len(X_test.coords['Y'])
        n_lon  = len(X_test.coords['X'])
        
        # Stack training data
        X_train_stacked = X_train.stack(sample=('T', 'Y', 'X')).transpose('sample', 'M').values
        y_train_stacked = y_train.stack(sample=('T', 'Y', 'X')).transpose('sample', 'M').values
        # train_nan_mask = np.any(np.isnan(X_train_stacked), axis=1) | np.any(np.isnan(y_train_stacked), axis=1)
        train_nan_mask = (np.any(~np.isfinite(X_train_stacked), axis=1) | np.any(~np.isfinite(y_train_stacked), axis=1))
        X_train_clean = X_train_stacked[~train_nan_mask]
        y_train_clean = y_train_stacked[~train_nan_mask]
        
        # Stack testing data
        X_test_stacked = X_test.stack(sample=('T', 'Y', 'X')).transpose('sample', 'M').values
        y_test_stacked = y_test.stack(sample=('T', 'Y', 'X')).transpose('sample', 'M').values
        # test_nan_mask = np.any(np.isnan(X_test_stacked), axis=1) | np.any(np.isnan(y_test_stacked), axis=1)
        test_nan_mask = (np.any(~np.isfinite(X_test_stacked), axis=1) | np.any(~np.isfinite(y_test_stacked), axis=1))
        
        # Fit the stacking ensemble only on rows without NaNs
        self.stacking_model.fit(X_train_clean, y_train_clean)
        y_pred_test = self.stacking_model.predict(X_test_stacked[~test_nan_mask])
        
        # Rebuild the predictions into the original shape,
        # leaving NaN rows intact.
        # Reconstruct predictions, leaving rows with NaNs intact
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
                dask_gufunc_kwargs={'output_sizes': {'probability': 3}, "allow_rechunk": True},
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
                dask_gufunc_kwargs={'output_sizes': {'probability': 3}, "allow_rechunk": True},
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
                dask_gufunc_kwargs={'output_sizes': {'probability': 3}, "allow_rechunk": True},
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
                dask_gufunc_kwargs={'output_sizes': {'probability': 3}, "allow_rechunk": True},
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
                dask_gufunc_kwargs={'output_sizes': {'probability': 3}, "allow_rechunk": True},
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

    def forecast(self, Predictant, clim_year_start, clim_year_end, hindcast_det, hindcast_det_cross, Predictor_for_year):
        """
        Forecast method:
         - Standardizes the predictor for the target year.
         - Fits the stacking ensemble on standardized hindcast and predictant data.
         - Predicts for the target year.
         - Reconstructs the prediction to its original (T, Y, X) shape.
         - Reverses the standardization.
         - Computes tercile probabilities using the chosen distribution.
        """
        mask = xr.where(~np.isnan(Predictant.isel(T=0, M=0)), 1, np.nan).drop_vars(['T','M']).squeeze()
        mask.name = None

        # Standardize predictor for the target year using hindcast statistics
        mean_val = hindcast_det.sel(T=slice(str(clim_year_start), str(clim_year_end))).mean(dim='T')
        std_val = hindcast_det.sel(T=slice(str(clim_year_start), str(clim_year_end))).std(dim='T')
        Predictor_for_year_st = (Predictor_for_year - mean_val) / std_val

        hindcast_det_st = standardize_timeseries(hindcast_det, clim_year_start, clim_year_end)
        Predictant_st = standardize_timeseries(Predictant, clim_year_start, clim_year_end)
        y_test = Predictant_st.isel(T=[-1])
        
        # Initialize stacking ensemble if not already done
        self.base_models = [
            ('mlp', MLPRegressor(
                hidden_layer_sizes=self.hidden_layer_sizes,
                activation=self.activation,
                solver=self.solver,
                max_iter=self.max_iter,
                random_state=self.random_state,
                alpha=self.mlp_alpha
            )),
            ('ada', AdaBoostRegressor(n_estimators=self.n_estimators_adaboost, random_state=self.random_state))
        ]
        self.meta_model = Ridge(alpha=self.ridge_alpha, random_state=self.random_state)
        self.stacking_model = StackingRegressor(estimators=self.base_models, final_estimator=self.meta_model)
        
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
        # train_nan_mask = np.any(np.isnan(X_train_stacked), axis=1) | np.any(np.isnan(y_train_stacked), axis=1)
        train_nan_mask = (np.any(~np.isfinite(X_train_stacked), axis=1) | np.any(~np.isfinite(y_train_stacked), axis=1))
        
        X_train_clean = X_train_stacked[~train_nan_mask]
        y_train_clean = y_train_stacked[~train_nan_mask]
        
        # Stack testing data
        X_test_stacked = Predictor_for_year_st.stack(sample=('T', 'Y', 'X')).transpose('sample', 'M').values
        y_test_stacked = y_test.stack(sample=('T', 'Y', 'X')).transpose('sample', 'M').values
        # test_nan_mask = np.any(np.isnan(X_test_stacked), axis=1) | np.any(np.isnan(y_test_stacked), axis=1)
        test_nan_mask = (np.any(~np.isfinite(X_test_stacked), axis=1) | np.any(~np.isfinite(y_test_stacked), axis=1))        
        # Fit the MLP model
        self.stacking_model.fit(X_train_clean, y_train_clean)
        y_pred = self.stacking_model.predict(X_test_stacked[~test_nan_mask])
        
        # Reconstruct the prediction array (keeping NaN rows intact)
        result = np.empty_like(np.squeeze(y_test_stacked))
        result[np.squeeze(test_nan_mask)] = np.squeeze(y_test_stacked[test_nan_mask])
        result[~np.squeeze(test_nan_mask)] = y_pred
        
        predictions_reshaped = result.reshape(n_time, n_lat, n_lon)
        result_da = xr.DataArray(data=predictions_reshaped,
                                    coords={'T': time, 'Y': lat, 'X': lon},
                                    dims=['T', 'Y', 'X'])* mask
        result_da = reverse_standardize(result_da, Predictant.isel(M=0).drop_vars("M").squeeze(),
                                        clim_year_start, clim_year_end)
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
                dask_gufunc_kwargs={'output_sizes': {'probability': 3}, "allow_rechunk": True},
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
                dask_gufunc_kwargs={'output_sizes': {'probability': 3}, "allow_rechunk": True},
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
                dask_gufunc_kwargs={'output_sizes': {'probability': 3}, "allow_rechunk": True},
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
                dask_gufunc_kwargs={'output_sizes': {'probability': 3}, "allow_rechunk": True},
            )
        elif self.dist_method == "nonparam":
            calc_func = self.calculate_tercile_probabilities_nonparametric
            error_samples = Predictant - hindcast_det_cross
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
        return result_da * mask, mask * hindcast_prob.transpose('probability', 'T', 'Y', 'X')


class WAS_mme_Stack_RF_GB_Ridge:
    def __init__(self, 
                 n_estimators_rf=100, 
                 max_depth_rf=None,
                 n_estimators_gb=100,
                 learning_rate_gb=0.1,
                 ridge_alpha=1.0,
                 random_state=42,
                 dist_method="gamma"):
        """
        Stacking ensemble with:
          - Base models: RandomForestRegressor and GradientBoostingRegressor.
          - Meta-model: Ridge.
          
        Parameters:
          n_estimators_rf: int, number of trees for RandomForestRegressor.
          max_depth_rf: int or None, maximum tree depth for RandomForestRegressor.
          n_estimators_gb: int, number of boosting iterations for GradientBoostingRegressor.
          learning_rate_gb: float, learning rate for GradientBoostingRegressor.
          ridge_alpha: float, regularization strength for Ridge.
          random_state: int, for reproducibility.
          dist_method: str, distribution method for tercile probability calculations.
                      Options: 't', 'gamma', 'nonparam', 'normal', 'lognormal'.
        """
        self.n_estimators_rf = n_estimators_rf
        self.max_depth_rf = max_depth_rf
        self.n_estimators_gb = n_estimators_gb
        self.learning_rate_gb = learning_rate_gb
        self.ridge_alpha = ridge_alpha
        self.random_state = random_state
        self.dist_method = dist_method

        self.base_models = None
        self.meta_model = None
        self.stacking_model = None

    def compute_model(self, X_train, y_train, X_test, y_test):
        """
        Fit the stacking ensemble on the training data and predict on X_test.
        Data (xarray DataArrays) are flattened and rows with NaNs are excluded for training.
        Predictions are reshaped back into the original (T, Y, X) format.
        """
        # Define base models:
        self.base_models = [
            ('rf', RandomForestRegressor(n_estimators=self.n_estimators_rf,
                                         max_depth=self.max_depth_rf,
                                         random_state=self.random_state)),
            ('gb', GradientBoostingRegressor(n_estimators=self.n_estimators_gb,
                                             learning_rate=self.learning_rate_gb,
                                             random_state=self.random_state))
        ]
        # Define meta-model (Ridge)
        self.meta_model = Ridge(alpha=self.ridge_alpha, random_state=self.random_state)
        
        # Create the stacking ensemble
        self.stacking_model = StackingRegressor(
            estimators=self.base_models,
            final_estimator=self.meta_model
        )
        
        # Extract coordinate variables from X_test
        time = X_test['T']
        lat = X_test['Y']
        lon = X_test['X']
        n_time = len(X_test.coords['T'])
        n_lat  = len(X_test.coords['Y'])
        n_lon  = len(X_test.coords['X'])
        
        # Stack training data
        X_train_stacked = X_train.stack(sample=('T', 'Y', 'X')).transpose('sample', 'M').values
        y_train_stacked = y_train.stack(sample=('T', 'Y', 'X')).transpose('sample', 'M').values
        # train_nan_mask = np.any(np.isnan(X_train_stacked), axis=1) | np.any(np.isnan(y_train_stacked), axis=1)
        train_nan_mask = (np.any(~np.isfinite(X_train_stacked), axis=1) | np.any(~np.isfinite(y_train_stacked), axis=1))
        X_train_clean = X_train_stacked[~train_nan_mask]
        y_train_clean = y_train_stacked[~train_nan_mask]
        
        # Stack testing data
        X_test_stacked = X_test.stack(sample=('T', 'Y', 'X')).transpose('sample', 'M').values
        y_test_stacked = y_test.stack(sample=('T', 'Y', 'X')).transpose('sample', 'M').values
        # test_nan_mask = np.any(np.isnan(X_test_stacked), axis=1) | np.any(np.isnan(y_test_stacked), axis=1)
        test_nan_mask = (np.any(~np.isfinite(X_test_stacked), axis=1) | np.any(~np.isfinite(y_test_stacked), axis=1))
        
        # Fit the stacking ensemble only on rows without NaNs
        self.stacking_model.fit(X_train_clean, y_train_clean)
        y_pred_test = self.stacking_model.predict(X_test_stacked[~test_nan_mask])
        
        # Rebuild the predictions into the original shape,
        # leaving NaN rows intact.
        # Reconstruct predictions, leaving rows with NaNs intact
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
        Compute tercile probabilities for hindcasts based on the chosen distribution.
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

    def forecast(self, Predictant, clim_year_start, clim_year_end, hindcast_det, hindcast_det_cross, Predictor_for_year):
        """
        Forecast method that:
          - Standardizes the target-year predictor using hindcast statistics.
          - Fits the stacking ensemble on standardized hindcast and predictant data.
          - Predicts for the target year.
          - Reconstructs the prediction to the original (T, Y, X) shape.
          - Reverses the standardization.
          - Computes tercile probabilities using the chosen distribution.
        """
        mask = xr.where(~np.isnan(Predictant.isel(T=0, M=0)), 1, np.nan)\
                .drop_vars(['T', 'M']).squeeze()
        mask.name = None
        
        # Standardize Predictor_for_year using hindcast climatology
        mean_val = hindcast_det.sel(T=slice(str(clim_year_start), str(clim_year_end))).mean(dim='T')
        std_val = hindcast_det.sel(T=slice(str(clim_year_start), str(clim_year_end))).std(dim='T')
        Predictor_for_year_st = (Predictor_for_year - mean_val) / std_val
        
        hindcast_det_st = standardize_timeseries(hindcast_det, clim_year_start, clim_year_end)
        Predictant_st = standardize_timeseries(Predictant, clim_year_start, clim_year_end)
        y_test = Predictant_st.isel(T=[-1])
        
        # Initialize stacking ensemble if not already set
        self.base_models = [
            ('mlp', MLPRegressor(
                hidden_layer_sizes=(10,5),  # Or set via parameters if needed
                activation='relu',
                solver='adam',
                max_iter=200,
                random_state=self.random_state,
                alpha=0.01
            )),
            ('ada', AdaBoostRegressor(n_estimators=50, random_state=self.random_state))
        ]
        self.meta_model = Ridge(alpha=1.0, random_state=self.random_state)
        self.stacking_model = StackingRegressor(estimators=self.base_models, final_estimator=self.meta_model)
        
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
        # train_nan_mask = np.any(np.isnan(X_train_stacked), axis=1) | np.any(np.isnan(y_train_stacked), axis=1)
        train_nan_mask = (np.any(~np.isfinite(X_train_stacked), axis=1) | np.any(~np.isfinite(y_train_stacked), axis=1))
        
        X_train_clean = X_train_stacked[~train_nan_mask]
        y_train_clean = y_train_stacked[~train_nan_mask]
        
        # Stack testing data
        X_test_stacked = Predictor_for_year_st.stack(sample=('T', 'Y', 'X')).transpose('sample', 'M').values
        y_test_stacked = y_test.stack(sample=('T', 'Y', 'X')).transpose('sample', 'M').values
        # test_nan_mask = np.any(np.isnan(X_test_stacked), axis=1) | np.any(np.isnan(y_test_stacked), axis=1)
        test_nan_mask = (np.any(~np.isfinite(X_test_stacked), axis=1) | np.any(~np.isfinite(y_test_stacked), axis=1))        
        # Fit the MLP model
        self.stacking_model.fit(X_train_clean, y_train_clean)
        y_pred = self.stacking_model.predict(X_test_stacked[~test_nan_mask])
        
        # Reconstruct the prediction array (keeping NaN rows intact)
        result = np.empty_like(np.squeeze(y_test_stacked))
        result[np.squeeze(test_nan_mask)] = np.squeeze(y_test_stacked[test_nan_mask])
        result[~np.squeeze(test_nan_mask)] = y_pred
        
        predictions_reshaped = result.reshape(n_time, n_lat, n_lon)
        result_da = xr.DataArray(data=predictions_reshaped,
                                    coords={'T': time, 'Y': lat, 'X': lon},
                                    dims=['T', 'Y', 'X'])* mask
        result_da = reverse_standardize(result_da, Predictant.isel(M=0).drop_vars("M").squeeze(),
                                        clim_year_start, clim_year_end)

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
        return result_da * mask, mask * hindcast_prob.transpose('probability', 'T', 'Y', 'X')



class WAS_mme_Stack_KNN_Tree_SVR:
    def __init__(self, 
                 n_neighbors=5,
                 tree_max_depth=None,
                 svr_C=1.0,
                 svr_kernel='rbf',
                 random_state=42,
                 dist_method="gamma"):
        """
        Stacking ensemble with:
          Base models: KNeighborsRegressor and DecisionTreeRegressor.
          Meta-model: SVR.
        
        Parameters:
          - n_neighbors: int, number of neighbors for KNN.
          - tree_max_depth: int or None, maximum depth for the Decision Tree.
          - svr_C: float, regularization parameter for SVR.
          - svr_kernel: string, kernel to use for SVR.
          - random_state: int, for reproducibility.
          - dist_method: string, one of ['t', 'gamma', 'nonparam', 'normal', 'lognormal'] to select
                         the probability calculation method.
        """
        self.n_neighbors = n_neighbors
        self.tree_max_depth = tree_max_depth
        self.svr_C = svr_C
        self.svr_kernel = svr_kernel
        self.random_state = random_state
        self.dist_method = dist_method

        self.base_models = None
        self.meta_model = None
        self.stacking_model = None

    def compute_model(self, X_train, y_train, X_test, y_test):
        """
        Fits the stacking ensemble on the training data and predicts on X_test.
        Data is assumed to be provided as xarray DataArrays with dimensions that include
        'T', 'Y', 'X' (and optionally 'M').
        """
        # Define base models:
        self.base_models = [
            ('knn', KNeighborsRegressor(n_neighbors=self.n_neighbors)),
            ('tree', DecisionTreeRegressor(max_depth=self.tree_max_depth, random_state=self.random_state))
        ]
        # Define meta-model: SVR
        self.meta_model = SVR(C=self.svr_C, kernel=self.svr_kernel)
        
        # Build stacking ensemble
        self.stacking_model = StackingRegressor(
            estimators=self.base_models,
            final_estimator=self.meta_model,
            n_jobs=-1
        )
        
        # Extract coordinate variables from X_test
        time = X_test['T']
        lat = X_test['Y']
        lon = X_test['X']
        n_time = len(X_test.coords['T'])
        n_lat  = len(X_test.coords['Y'])
        n_lon  = len(X_test.coords['X'])
        
        # Stack training data
        X_train_stacked = X_train.stack(sample=('T', 'Y', 'X')).transpose('sample', 'M').values
        y_train_stacked = y_train.stack(sample=('T', 'Y', 'X')).transpose('sample', 'M').values
        # train_nan_mask = np.any(np.isnan(X_train_stacked), axis=1) | np.any(np.isnan(y_train_stacked), axis=1)
        train_nan_mask = (np.any(~np.isfinite(X_train_stacked), axis=1) | np.any(~np.isfinite(y_train_stacked), axis=1))
        X_train_clean = X_train_stacked[~train_nan_mask]
        y_train_clean = y_train_stacked[~train_nan_mask]
        
        # Stack testing data
        X_test_stacked = X_test.stack(sample=('T', 'Y', 'X')).transpose('sample', 'M').values
        y_test_stacked = y_test.stack(sample=('T', 'Y', 'X')).transpose('sample', 'M').values
        # test_nan_mask = np.any(np.isnan(X_test_stacked), axis=1) | np.any(np.isnan(y_test_stacked), axis=1)
        test_nan_mask = (np.any(~np.isfinite(X_test_stacked), axis=1) | np.any(~np.isfinite(y_test_stacked), axis=1))
        
        # Fit the stacking ensemble only on rows without NaNs
        self.stacking_model.fit(X_train_clean, y_train_clean)
        y_pred_test = self.stacking_model.predict(X_test_stacked[~test_nan_mask])
        
        # Rebuild the predictions into the original shape,
        # leaving NaN rows intact.
        # Reconstruct predictions, leaving rows with NaNs intact
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
        Predictant is expected to be an xarray DataArray with dimensions (T, Y, X).
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
                dask_gufunc_kwargs={'output_sizes': {'probability': 3}, "allow_rechunk": True},
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
                dask_gufunc_kwargs={'output_sizes': {'probability': 3}, "allow_rechunk": True},
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
                dask_gufunc_kwargs={'output_sizes': {'probability': 3}, "allow_rechunk": True},
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
                dask_gufunc_kwargs={'output_sizes': {'probability': 3}, "allow_rechunk": True},
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

    def forecast(self, Predictant, clim_year_start, clim_year_end, hindcast_det, hindcast_det_cross, Predictor_for_year):
        """
        Forecast method:
          - Standardizes the target-year predictor using hindcast statistics.
          - Fits the stacking ensemble on standardized hindcast and predictant data.
          - Predicts for the target year.
          - Reconstructs the prediction to its original (T, Y, X) shape.
          - Reverses the standardization.
          - Computes tercile probabilities using the chosen distribution.
        """
        mask = xr.where(~np.isnan(Predictant.isel(T=0, M=0)), 1, np.nan)\
                .drop_vars(['T', 'M']).squeeze()
        mask.name = None
        
        # Standardize the predictor for the target year using hindcast climatology
        mean_val = hindcast_det.sel(T=slice(str(clim_year_start), str(clim_year_end))).mean(dim='T')
        std_val = hindcast_det.sel(T=slice(str(clim_year_start), str(clim_year_end))).std(dim='T')
        Predictor_for_year_st = (Predictor_for_year - mean_val) / std_val
        
        hindcast_det_st = standardize_timeseries(hindcast_det, clim_year_start, clim_year_end)
        Predictant_st = standardize_timeseries(Predictant, clim_year_start, clim_year_end)
        y_test = Predictant_st.isel(T=[-1])
        
        # Initialize stacking ensemble if not already done
        self.base_models = [
            ('knn', KNeighborsRegressor(n_neighbors=self.n_neighbors)),
            ('tree', DecisionTreeRegressor(max_depth=self.tree_max_depth, random_state=self.random_state))
        ]
        self.meta_model = SVR(C=self.svr_C, kernel=self.svr_kernel)
        self.stacking_model = StackingRegressor(estimators=self.base_models, final_estimator=self.meta_model)
        
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
        # train_nan_mask = np.any(np.isnan(X_train_stacked), axis=1) | np.any(np.isnan(y_train_stacked), axis=1)
        train_nan_mask = (np.any(~np.isfinite(X_train_stacked), axis=1) | np.any(~np.isfinite(y_train_stacked), axis=1))
        
        X_train_clean = X_train_stacked[~train_nan_mask]
        y_train_clean = y_train_stacked[~train_nan_mask]
        
        # Stack testing data
        X_test_stacked = Predictor_for_year_st.stack(sample=('T', 'Y', 'X')).transpose('sample', 'M').values
        y_test_stacked = y_test.stack(sample=('T', 'Y', 'X')).transpose('sample', 'M').values
        # test_nan_mask = np.any(np.isnan(X_test_stacked), axis=1) | np.any(np.isnan(y_test_stacked), axis=1)
        test_nan_mask = (np.any(~np.isfinite(X_test_stacked), axis=1) | np.any(~np.isfinite(y_test_stacked), axis=1))        
        # Fit the MLP model
        self.stacking_model.fit(X_train_clean, y_train_clean)
        y_pred = self.stacking_model.predict(X_test_stacked[~test_nan_mask])
        
        # Reconstruct the prediction array (keeping NaN rows intact)
        result = np.empty_like(np.squeeze(y_test_stacked))
        result[np.squeeze(test_nan_mask)] = np.squeeze(y_test_stacked[test_nan_mask])
        result[~np.squeeze(test_nan_mask)] = y_pred
        
        predictions_reshaped = result.reshape(n_time, n_lat, n_lon)
        result_da = xr.DataArray(data=predictions_reshaped,
                                    coords={'T': time, 'Y': lat, 'X': lon},
                                    dims=['T', 'Y', 'X'])* mask
        result_da = reverse_standardize(result_da, Predictant.isel(M=0).drop_vars("M").squeeze(),
                                        clim_year_start, clim_year_end)
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
                dask_gufunc_kwargs={'output_sizes': {'probability': 3}, "allow_rechunk": True},
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
                dask_gufunc_kwargs={'output_sizes': {'probability': 3}, "allow_rechunk": True},
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
                dask_gufunc_kwargs={'output_sizes': {'probability': 3}, "allow_rechunk": True},
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
                dask_gufunc_kwargs={'output_sizes': {'probability': 3}, "allow_rechunk": True},
            )
        elif self.dist_method == "nonparam":
            calc_func = self.calculate_tercile_probabilities_nonparametric
            error_samples = Predictant - hindcast_det_cross
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
        return result_da * mask, mask * hindcast_prob.transpose('probability', 'T', 'Y', 'X')

class WAS_mme_StackXGboost_Ml:
    def __init__(
        self,
        n_estimators=100,
        max_depth=3,
        learning_rate=0.1,
        random_state=42,
        dist_method="gamma"
    ):
        """
        Initializes the stacking model with XGBoost as the base model
        and Linear Regression as the meta-model.
        
        Parameters
        ----------
        n_estimators : int
            Number of gradient boosted trees in XGBoost.
        max_depth : int
            Maximum depth of each tree in XGBoost.
        learning_rate : float
            Boosting learning rate (xgb’s “eta”).
        random_state : int
            Random seed for reproducibility.
        dist_method : str
            Distribution method to use for probability calculations.
        """
        self.n_estimators = n_estimators
        self.max_depth = max_depth
        self.learning_rate = learning_rate
        self.random_state = random_state
        self.dist_method = dist_method

    def compute_model(self, X_train, y_train, X_test, y_test):
        """
        Train a stacking regressor with XGBoost as the base model and 
        Linear Regression as the meta-model, then generate deterministic
        predictions on X_test.
        
        Parameters
        ----------
        X_train : xarray.DataArray
            Predictor training data with dimensions (T, Y, X, M) or (T, Y, X).
        y_train : xarray.DataArray
            Predictand training data with dimensions (T, Y, X, M) or (T, Y, X).
        X_test : xarray.DataArray
            Predictor testing data with dimensions (T, Y, X, M) or (T, Y, X).
        y_test : xarray.DataArray
            Predictand testing data with dimensions (T, Y, X, M) or (T, Y, X).

        Returns
        -------
        predicted_da : xarray.DataArray
            The predictions over X_test in the original grid shape.
        """
        # Initialize the base model (XGBoost)
        self.base_models = [
            ('xgb', XGBRegressor(
                n_estimators=self.n_estimators,
                max_depth=self.max_depth,
                learning_rate=self.learning_rate,
                random_state=self.random_state
            ))
        ]
        
        # Initialize the meta-model (Linear Regression)
        self.meta_model = LinearRegression()
        
        # Initialize the stacking ensemble
        self.stacking_model = StackingRegressor(
            estimators=self.base_models,
            final_estimator=self.meta_model
        )
        
        # Extract coordinate variables from X_test
        time = X_test['T']
        lat = X_test['Y']
        lon = X_test['X']
        n_time = len(X_test.coords['T'])
        n_lat  = len(X_test.coords['Y'])
        n_lon  = len(X_test.coords['X'])
        
        # Stack training data
        X_train_stacked = X_train.stack(sample=('T', 'Y', 'X')).transpose('sample', 'M').values
        y_train_stacked = y_train.stack(sample=('T', 'Y', 'X')).transpose('sample', 'M').values
        # train_nan_mask = np.any(np.isnan(X_train_stacked), axis=1) | np.any(np.isnan(y_train_stacked), axis=1)
        train_nan_mask = (np.any(~np.isfinite(X_train_stacked), axis=1) | np.any(~np.isfinite(y_train_stacked), axis=1))
        X_train_clean = X_train_stacked[~train_nan_mask]
        y_train_clean = y_train_stacked[~train_nan_mask]
        
        # Stack testing data
        X_test_stacked = X_test.stack(sample=('T', 'Y', 'X')).transpose('sample', 'M').values
        y_test_stacked = y_test.stack(sample=('T', 'Y', 'X')).transpose('sample', 'M').values
        # test_nan_mask = np.any(np.isnan(X_test_stacked), axis=1) | np.any(np.isnan(y_test_stacked), axis=1)
        test_nan_mask = (np.any(~np.isfinite(X_test_stacked), axis=1) | np.any(~np.isfinite(y_test_stacked), axis=1))
        
        # Fit the stacking ensemble only on rows without NaNs
        self.stacking_model.fit(X_train_clean, y_train_clean)
        y_pred_test = self.stacking_model.predict(X_test_stacked[~test_nan_mask])
        
        # Rebuild the predictions into the original shape,
        # leaving NaN rows intact.
        # Reconstruct predictions, leaving rows with NaNs intact
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
        This method extracts the climatology terciles from Predictant over the period
        [clim_year_start, clim_year_end], computes an error variance (or uses error samples),
        and then applies the chosen probability function.
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
        
        # For error variance, here we use the difference between Predictant and hindcast_det
        error_variance = (Predictant - hindcast_det).var(dim='T')
        dof = len(Predictant.get_index("T")) - 2
        
        # Choose the appropriate probability calculation function
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
                dask_gufunc_kwargs={'output_sizes': {'probability': 3}, "allow_rechunk": True},
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
                dask_gufunc_kwargs={'output_sizes': {'probability': 3}, "allow_rechunk": True},
            )
        elif self.dist_method == "nonparam":
            # For nonparametric, assume hindcast_det contains error samples
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
                dask_gufunc_kwargs={'output_sizes': {'probability': 3}, "allow_rechunk": True},
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
                dask_gufunc_kwargs={'output_sizes': {'probability': 3}, "allow_rechunk": True},
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
                dask_gufunc_kwargs={'output_sizes': {'probability': 3}, "allow_rechunk": True},
            )
        else:
            raise ValueError(f"Invalid dist_method: {self.dist_method}.")

        hindcast_prob = hindcast_prob.assign_coords(probability=('probability', ['PB', 'PN', 'PA']))
        return hindcast_prob.transpose('probability', 'T', 'Y', 'X')

    @staticmethod
    def _reshape_and_filter_data(da):
        """
        Helper: stack the DataArray from (T, Y, X[, M]) to (n_samples, n_features)
        and remove any rows with NaNs.
        """
        da_stacked = da.stack(sample=('T', 'Y', 'X'))
        if 'M' in da.dims:
            da_stacked = da_stacked.transpose('sample', 'M')
        else:
            da_stacked = da_stacked.transpose('sample')
        da_values = da_stacked.values
        nan_mask = np.any(np.isnan(da_values), axis=1)
        return da_values[~nan_mask], nan_mask, da_values

    def forecast(self, Predictant, clim_year_start, clim_year_end, hindcast_det, hindcast_det_cross, Predictor_for_year):
        """
        Forecast method that uses the trained stacking ensemble to predict for a new year,
        then computes tercile probabilities.
        
        Parameters
        ----------
        Predictant : xarray.DataArray
            Historical predictand data.
        clim_year_start : int or str
            Start of the climatology period (e.g., 1981).
        clim_year_end : int or str
            End of the climatology period (e.g., 2010).
        hindcast_det : xarray.DataArray
            Deterministic hindcasts used for training (predictors).
        hindcast_det_cross : xarray.DataArray
            Deterministic hindcasts for cross-validation (for error estimation).
        Predictor_for_year : xarray.DataArray
            Predictor data for the target forecast year.
        
        Returns
        -------
        result_da : xarray.DataArray
            The deterministic forecast for the target year.
        hindcast_prob : xarray.DataArray
            Tercile probability forecast for the target year.
        """
        mask = xr.where(~np.isnan(Predictant.isel(T=0, M=0)), 1, np.nan)\
                .drop_vars(['T', 'M']).squeeze()
        mask.name = None
        
        # Standardize the predictor for the target year using hindcast climatology
        mean_val = hindcast_det.sel(T=slice(str(clim_year_start), str(clim_year_end))).mean(dim='T')
        std_val = hindcast_det.sel(T=slice(str(clim_year_start), str(clim_year_end))).std(dim='T')
        Predictor_for_year_st = (Predictor_for_year - mean_val) / std_val
        
        hindcast_det_st = standardize_timeseries(hindcast_det, clim_year_start, clim_year_end)
        Predictant_st = standardize_timeseries(Predictant, clim_year_start, clim_year_end)
        y_test = Predictant_st.isel(T=[-1])
        
        # Initialize stacking ensemble (if not already done or to retrain)
        self.base_models = [
            ('xgb', XGBRegressor(
                n_estimators=self.n_estimators,
                max_depth=self.max_depth,
                learning_rate=self.learning_rate,
                random_state=self.random_state
            ))
        ]
        self.meta_model = LinearRegression()
        self.stacking_model = StackingRegressor(
            estimators=self.base_models,
            final_estimator=self.meta_model
        )
        
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
        # train_nan_mask = np.any(np.isnan(X_train_stacked), axis=1) | np.any(np.isnan(y_train_stacked), axis=1)
        train_nan_mask = (np.any(~np.isfinite(X_train_stacked), axis=1) | np.any(~np.isfinite(y_train_stacked), axis=1))
        
        X_train_clean = X_train_stacked[~train_nan_mask]
        y_train_clean = y_train_stacked[~train_nan_mask]
        
        # Stack testing data
        X_test_stacked = Predictor_for_year_st.stack(sample=('T', 'Y', 'X')).transpose('sample', 'M').values
        y_test_stacked = y_test.stack(sample=('T', 'Y', 'X')).transpose('sample', 'M').values
        # test_nan_mask = np.any(np.isnan(X_test_stacked), axis=1) | np.any(np.isnan(y_test_stacked), axis=1)
        test_nan_mask = (np.any(~np.isfinite(X_test_stacked), axis=1) | np.any(~np.isfinite(y_test_stacked), axis=1))         
        # Fit the stacking model
        self.stacking_model.fit(X_train_clean, y_train_clean)
        y_pred = self.stacking_model.predict(X_test_stacked[~test_nan_mask])
        
        # Reconstruct the prediction array (keeping NaN rows intact)
        result = np.empty_like(np.squeeze(y_test_stacked))
        result[np.squeeze(test_nan_mask)] = np.squeeze(y_test_stacked[test_nan_mask])
        result[~np.squeeze(test_nan_mask)] = y_pred
        
        predictions_reshaped = result.reshape(n_time, n_lat, n_lon)
        result_da = xr.DataArray(data=predictions_reshaped,
                                    coords={'T': time, 'Y': lat, 'X': lon},
                                    dims=['T', 'Y', 'X'])* mask
        result_da = reverse_standardize(result_da, Predictant.isel(M=0).drop_vars("M").squeeze(),
                                        clim_year_start, clim_year_end)
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
                dask_gufunc_kwargs={'output_sizes': {'probability': 3}, "allow_rechunk": True},
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
                dask_gufunc_kwargs={'output_sizes': {'probability': 3}, "allow_rechunk": True},
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
                dask_gufunc_kwargs={'output_sizes': {'probability': 3}, "allow_rechunk": True},
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
                dask_gufunc_kwargs={'output_sizes': {'probability': 3}, "allow_rechunk": True},
            )
        elif self.dist_method == "nonparam":
            calc_func = self.calculate_tercile_probabilities_nonparametric
            error_samples = Predictant - hindcast_det_cross
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
        return result_da * mask, mask * hindcast_prob.transpose('probability', 'T', 'Y', 'X')
