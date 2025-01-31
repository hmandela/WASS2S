from core.utils import *
import xcast as xc
import numpy as np
import xarray as xr
from dask.distributed import Client
from scipy import stats
import pandas as pd
import xcast as xc  # Assuming xc is a module that provides the ELM implementation
from scipy.stats import gamma
from sklearn.neural_network import MLPRegressor
from sklearn.ensemble import StackingRegressor
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor
from sklearn.neural_network import MLPRegressor

class WAS_mme_Weighted():
    def __init__(self):
        pass

    def compute(self, rainfall, hdcst, fcst, scores, complete=False):
        mask = xr.where(~np.isnan(rainfall.isel(T=0)), 1, np.nan).drop_vars(['T']).squeeze().to_numpy()
        # 2. Initialize accumulators
        score_sum = None         # Will hold the sum of MAE-derived weights (or any chosen metric)
        hindcast_det = None      # Will hold the weighted sum of hindcasts
        forecast_det = None      # Will hold the weighted sum of forecasts
        
        hindcast_det_ = None      # Will hold the not weighted sum of hindcasts
        forecast_det_ = None      # Will hold the not weighted sum of forecasts 
        
        for model_name in hdcst.keys():
            # 3a. Get the model's score array (MAE) and interpolate it to Obs grid
            score_array = scores["GROC"][model_name]
            score_array = score_array.interp(
                Y=rainfall.Y,
                X=rainfall.X,
                method="nearest",
                kwargs={"fill_value": "extrapolate"}
            )
            score_array = xr.where(score_array<0.5,0,1)
    
            hincast_data = hdcst[model_name].interp(
                Y=rainfall.Y,
                X=rainfall.X,
                method="nearest",
                kwargs={"fill_value": "extrapolate"}
            )

            forecast_data = fcst[model_name].interp(
                Y=rainfall.Y,
                X=rainfall.X,
                method="nearest",
                kwargs={"fill_value": "extrapolate"}
            )
    
            # 3e. Multiply each dataset by its score_array for weighting
            hincast_weighted = hincast_data * score_array
            forecast_weighted = forecast_data * score_array


            hincast_not_weighted = hincast_data
            forecast_not_weighted = forecast_data

            # 3f. Accumulate into the running total
            if hindcast_det is None:
                # First iteration
                hindcast_det = hincast_weighted
                forecast_det = forecast_weighted
                score_sum = score_array
                hindcast_det_ = hincast_not_weighted
                forecast_det_ = forecast_not_weighted
            else:
                # Subsequent iterations: add to existing
                hindcast_det = hindcast_det + hincast_weighted
                forecast_det = forecast_det + forecast_weighted
                score_sum = score_sum + score_array
                
                hindcast_det_ = hindcast_det_ + hincast_not_weighted
                forecast_det_ = forecast_det_ + forecast_not_weighted    
        # 4. Convert sums to weighted means
        hindcast_det = hindcast_det / score_sum
        forecast_det = forecast_det / score_sum
        if complete:
            hindcast_det_ = hindcast_det_/len(hdcst.keys())
            forecast_det_ = forecast_det_/len(hdcst.keys())
            mask_hd = xr.where(np.isnan(hindcast_det),1,0)
            mask_fc = xr.where(np.isnan(forecast_det),1,0)
            hindcast_det = hindcast_det.fillna(0) + hindcast_det_*mask_hd
            forecast_det = forecast_det.fillna(0)  +  forecast_det_*mask_fc       
            
    
        return hindcast_det*mask, forecast_det*mask
       
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


    def compute_prob(self, Predictant, clim_year_start, clim_year_end, hindcast_det):
        
        index_start = Predictant.get_index("T").get_loc(str(clim_year_start)).start
        index_end = Predictant.get_index("T").get_loc(str(clim_year_end)).stop

        rainfall_for_tercile = Predictant.isel(T=slice(index_start, index_end))
        terciles = rainfall_for_tercile.quantile([0.3, 0.67], dim='T')
        error_variance = (Predictant - hindcast_det).var(dim='T')

        dof = len(Predictant.get_index("T"))

        hindcast_prob = xr.apply_ufunc(
            self.calculate_tercile_probabilities_gamma,
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
            dask_gufunc_kwargs={'output_sizes': {'probability': 3}},
        )

        hindcast_prob = hindcast_prob.assign_coords(probability=('probability', ['PB', 'PN', 'PA']))
        return hindcast_prob.transpose('probability', 'T', 'Y', 'X')

    def forecast(self, Predictant, clim_year_start, clim_year_end, hindcast_det, forecast_det):
        
        index_start = Predictant.get_index("T").get_loc(str(clim_year_start)).start
        index_end = Predictant.get_index("T").get_loc(str(clim_year_end)).stop

        rainfall_for_tercile = Predictant.isel(T=slice(index_start, index_end))
        terciles = rainfall_for_tercile.quantile([0.3, 0.6666667], dim='T')
        error_variance = (Predictant - hindcast_det).var(dim='T')

        dof = len(Predictant.get_index("T"))

        hindcast_prob = xr.apply_ufunc(
            self.calculate_tercile_probabilities_gamma,
            forecast_det,
            error_variance,
            terciles.isel(quantile=0).drop_vars('quantile'),
            terciles.isel(quantile=1).drop_vars('quantile'),
            input_core_dims=[('T',), (), (), ()],
            vectorize=True,
            kwargs={'dof': dof},
            dask='parallelized',
            output_core_dims=[('probability', 'T',)],
            output_dtypes=['float'],
            dask_gufunc_kwargs={'output_sizes': {'probability': 3}},
        )
        hindcast_prob = hindcast_prob.assign_coords(probability=('probability', ['PB', 'PN', 'PA']))
        return hindcast_prob.drop_vars('T').squeeze().transpose('probability', 'Y', 'X')
        

class WAS_mme_ELM:

    def __init__(self, elm_kwargs=None):
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
     

    def compute_model(self, X_train, y_train, X_test):
        
        X_train = X_train.fillna(0)
        y_train = y_train.fillna(0)
        
        model = xc.ELM(**self.elm_kwargs) 
        model.fit(X_train, y_train)
        result_ = model.predict(X_test)
        return result_.rename({'S':'T'}).transpose('T', 'M', 'Y', 'X').drop_vars('M').squeeze()

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
        
    @staticmethod
    def calculate_tercile_probabilities_gamma(best_guess, error_variance, T1, T2, dof):

        n_time = len(best_guess)
        pred_prob = np.empty((3, n_time), dtype=float)
    
        # If all best_guess are NaN, just fill everything with NaN
        if np.all(np.isnan(best_guess)):
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


    def compute_prob(self, Predictant, clim_year_start, clim_year_end, hindcast_det):
        
        Predictant = Predictant.transpose('T', 'M', 'Y', 'X').drop_vars('M').squeeze('M')
        index_start = Predictant.get_index("T").get_loc(str(clim_year_start)).start
        index_end = Predictant.get_index("T").get_loc(str(clim_year_end)).stop

        rainfall_for_tercile = Predictant.isel(T=slice(index_start, index_end))
        terciles = rainfall_for_tercile.quantile([0.3, 0.67], dim='T')
        error_variance = (Predictant - hindcast_det).var(dim='T')

        dof = len(Predictant.get_index("T"))

        hindcast_prob = xr.apply_ufunc(
            self.calculate_tercile_probabilities_gamma,
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
            dask_gufunc_kwargs={'output_sizes': {'probability': 3}, "allow_rechunk":True},
        )

        hindcast_prob = hindcast_prob.assign_coords(probability=('probability', ['PB', 'PN', 'PA']))
        return hindcast_prob.transpose('probability', 'T', 'Y', 'X')

    def forecast(self, Predictant, clim_year_start, clim_year_end, hindcast_det, hindcast_det_cross, Predictor_for_year):
        
        hindcast_det_ = hindcast_det.fillna(0)
        Predictant_ = Predictant.fillna(0)
        Predictor_for_year_ = Predictor_for_year.fillna(0)
        
        model = xc.ELM(**self.elm_kwargs) 
        model.fit(hindcast_det_, Predictant_)
        result_ = model.predict(Predictor_for_year_)
        result_ = result_.rename({'S':'T'}).transpose('T', 'M', 'Y', 'X').drop_vars('M').squeeze('M')
        Predictant = Predictant.transpose('T', 'M', 'Y', 'X').drop_vars('M').squeeze('M')
        # result_ = xr.where(result_ > (np.nanmax(np.unique(Predictant.max(dim='T')))+300), (np.nanmax(np.unique(Predictant.max(dim='T')))+300), result_)
        # result_ = xr.where(result_<0,0,result_)

        index_start = Predictant.get_index("T").get_loc(str(clim_year_start)).start
        index_end = Predictant.get_index("T").get_loc(str(clim_year_end)).stop

        rainfall_for_tercile = Predictant.isel(T=slice(index_start, index_end))
        terciles = rainfall_for_tercile.quantile([0.3, 0.67], dim='T')
        error_variance = (Predictant - hindcast_det_cross).var(dim='T')

        dof = len(Predictant.get_index("T"))

        hindcast_prob = xr.apply_ufunc(
            self.calculate_tercile_probabilities_gamma,
            result_,#.expand_dims({'T': [pd.Timestamp(Predictor_for_year.coords['T'].values).to_pydatetime()]}),
            error_variance,
            terciles.isel(quantile=0).drop_vars('quantile'),
            terciles.isel(quantile=1).drop_vars('quantile'),
            input_core_dims=[('T',), (), (), ()],
            vectorize=True,
            kwargs={'dof': dof},
            dask='parallelized',
            output_core_dims=[('probability', 'T',)],
            output_dtypes=['float'],
            dask_gufunc_kwargs={'output_sizes': {'probability': 3}, "allow_rechunk":True},
        )
        hindcast_prob = hindcast_prob.assign_coords(probability=('probability', ['PB', 'PN', 'PA']))
        return result_, hindcast_prob.drop_vars('T').squeeze().transpose('probability', 'Y', 'X')



class WAS_mme_MLP:
    def __init__(self, hidden_layer_sizes=(10,5), activation='relu', max_iter=200, solver='adam', random_state=42, alpha=0.01):

        self.hidden_layer_sizes=hidden_layer_sizes
        self.activation=activation
        self.solver=solver
        self.max_iter=max_iter
        self.random_state=random_state
        self.alpha=alpha  
        
    def compute_model(self, X_train, y_train, X_test, y_test):
        
        self.mlp_model = MLPRegressor(
            hidden_layer_sizes=self.hidden_layer_sizes,
            activation=self.activation,
            solver=self.solver,
            max_iter=self.max_iter,
            random_state=self.random_state,
            alpha=self.alpha
        )
        
        time = X_test['T']
        lat = X_test['Y']
        lon = X_test['X']
        
        n_time = len(X_test.coords['T'])
        n_lat  = len(X_test.coords['Y'])
        n_lon  = len(X_test.coords['X'])

        
        X_train_ = X_train.stack(sample=('T', 'Y', 'X')) #.fillna(0)
        X_train_ = X_train_.transpose('sample', 'M')
        X_train_ = X_train_.values
        
        X_train_nan_indices = np.any(np.isnan(X_train_), axis=1)   
        X_train_array_with_nan = X_train_[X_train_nan_indices]
        X_train_array_without_nan = X_train_[~X_train_nan_indices]
               
        y_train_ = y_train.stack(sample=('T', 'Y', 'X'))  #.fillna(0)
        y_train_ = y_train_.transpose('sample', 'M')
        y_train_ = y_train_.values

        y_train_nan_indices = np.any(np.isnan(y_train_), axis=1)
        y_train_array_with_nan = y_train_[y_train_nan_indices]
        y_train_array_without_nan = y_train_[~y_train_nan_indices]

        X_test_ = X_test.stack(sample=('T', 'Y', 'X')) #.fillna(0)
        X_test_ = X_test_.transpose('sample', 'M')
        X_test_ = X_test_.values 
        
        X_test_nan_indices = np.any(np.isnan(X_test_), axis=1)
        X_test_array_with_nan = X_test_[X_test_nan_indices]
        X_test_array_without_nan = X_test_[~X_test_nan_indices] 
    
        y_test_ = y_test.stack(sample=('T', 'Y', 'X')) #.fillna(0)
        y_test_ = y_test_.transpose('sample', 'M')
        y_test_ = y_test_.values 

        
        y_test_nan_indices = np.any(np.isnan(y_test_), axis=1)
        y_test_array_with_nan = y_test_[y_test_nan_indices]
        y_test_array_without_nan = y_test_[~y_test_nan_indices]
        
        self.mlp_model.fit(X_train_array_without_nan, y_train_array_without_nan)
        y_pred_test = self.mlp_model.predict(X_test_array_without_nan)

        
        result = np.empty_like(np.squeeze(y_test_))
        print(result.shape)
        result[np.squeeze(y_test_nan_indices)] = np.squeeze(y_test_array_with_nan)
        result[~np.squeeze(y_test_nan_indices)] = y_pred_test
        print(result.shape)
                
        predictions_reshaped = result.reshape(n_time, n_lat, n_lon)
        predicted_da = xr.DataArray(
            data=predictions_reshaped,
            coords={
                'T': time,
                'Y': lat,
                'X': lon
            },
            dims=['T', 'Y', 'X'],
        )  
        return predicted_da

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


    def compute_prob(self, Predictant, clim_year_start, clim_year_end, hindcast_det):
        
        Predictant = Predictant.transpose('T', 'M', 'Y', 'X').drop_vars('M').squeeze('M')
        index_start = Predictant.get_index("T").get_loc(str(clim_year_start)).start
        index_end = Predictant.get_index("T").get_loc(str(clim_year_end)).stop

        rainfall_for_tercile = Predictant.isel(T=slice(index_start, index_end))
        terciles = rainfall_for_tercile.quantile([0.3, 0.67], dim='T')
        error_variance = (Predictant - hindcast_det).var(dim='T')

        dof = len(Predictant.get_index("T"))

        hindcast_prob = xr.apply_ufunc(
            self.calculate_tercile_probabilities_gamma,
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
            dask_gufunc_kwargs={'output_sizes': {'probability': 3}, "allow_rechunk":True},
        )

        hindcast_prob = hindcast_prob.assign_coords(probability=('probability', ['PB', 'PN', 'PA']))
        return hindcast_prob.transpose('probability', 'T', 'Y', 'X')

    def forecast(self, Predictant, clim_year_start, clim_year_end, hindcast_det, hindcast_det_cross, Predictor_for_year):
        
        mask = xr.where(~np.isnan(Predictant.isel(T=0,M=0)), 1, np.nan).drop_vars(['T','M']).squeeze()
        mask.name = None
        
        Predictor_for_year_st = (Predictor_for_year - hindcast_det.sel(slice(str(clim_year_start),str(clim_year_end))).mean(dim='T'))/hindcast_det.sel(slice(str(clim_year_start),str(clim_year_end))).std(dim='T')  
        hindcast_det_st = standardize_timeseries(hindcast_det, clim_year_start, clim_year_end)
        Predictant_st = standardize_timeseries(Predictant, clim_year_start, clim_year_end) 
        
        self.mlp_model = MLPRegressor(
            hidden_layer_sizes=self.hidden_layer_sizes,
            activation=self.activation,
            solver=self.solver,
            max_iter=self.max_iter,
            random_state=self.random_state,
            alpha=self.alpha
        )
        
        time = Predictor_for_year_['T']
        lat = Predictor_for_year_['Y']
        lon = Predictor_for_year_['X']
        
        n_time = len(Predictor_for_year_.coords['T'])
        n_lat  = len(Predictor_for_year_.coords['Y'])
        n_lon  = len(Predictor_for_year_.coords['X'])
        

        
        hindcast_det_ = hindcast_det_st.fillna(0).stack(sample=('T', 'Y', 'X'))
        hindcast_det_ = hindcast_det_.transpose('sample', 'M')
        hindcast_det_ = hindcast_det_.values

        Predictant_ = Predictant_st.fillna(0).stack(sample=('T', 'Y', 'X'))
        Predictant_ = Predictant_.transpose('sample', 'M')
        Predictant_ = Predictant_.values

        Predictor_for_year_ = Predictor_for_year_st.fillna(0).stack(sample=('T', 'Y', 'X'))
        Predictor_for_year_ = Predictor_for_year_.transpose('sample', 'M')
        Predictor_for_year_ = Predictor_for_year_.values 

        self.mlp_model.fit(hindcast_det_, Predictant_)
        y_pred_test = self.mlp_model.predict(Predictor_for_year_)
        
        predictions_reshaped = y_pred_test.reshape(n_time, n_lat, n_lon)
        result_ = xr.DataArray(
            data=predictions_reshaped,
            coords={
                'T': time,
                'Y': lat,
                'X': lon
            },
            dims=['T', 'Y', 'X'],
        )*mask  

        result_ = reverse_standardize(result_, Predictant.isel(M=0).drop_var("M").squeeze(), clim_year_start, clim_year_end)
        
        
        index_start = Predictant.get_index("T").get_loc(str(clim_year_start)).start
        index_end = Predictant.get_index("T").get_loc(str(clim_year_end)).stop

        rainfall_for_tercile = Predictant.isel(T=slice(index_start, index_end))
        terciles = rainfall_for_tercile.quantile([0.3, 0.67], dim='T')
        error_variance = (Predictant - hindcast_det_cross).var(dim='T')

        dof = len(Predictant.get_index("T"))

        hindcast_prob = xr.apply_ufunc(
            self.calculate_tercile_probabilities_gamma,
            result_,#.expand_dims({'T': [pd.Timestamp(Predictor_for_year.coords['T'].values).to_pydatetime()]}),
            error_variance,
            terciles.isel(quantile=0).drop_vars('quantile'),
            terciles.isel(quantile=1).drop_vars('quantile'),
            input_core_dims=[('T',), (), (), ()],
            vectorize=True,
            kwargs={'dof': dof},
            dask='parallelized',
            output_core_dims=[('probability', 'T',)],
            output_dtypes=['float'],
            dask_gufunc_kwargs={'output_sizes': {'probability': 3}, "allow_rechunk":True},
        )
        hindcast_prob = hindcast_prob.assign_coords(probability=('probability', ['PB', 'PN', 'PA']))
        return result_*mask, mask*hindcast_prob.drop_vars('T').squeeze().transpose('probability', 'Y', 'X')




class WAS_mme_BMA:
    pass

class WAS_mme_ELR:
    pass

class WAS_mme_Stack_MLP_RF:
    def __init__(self, hidden_layer_sizes=(10,5), activation='relu', max_iter=200, solver='adam', random_state=42, alpha=0.01, n_estimators = 100):

        self.hidden_layer_sizes=hidden_layer_sizes
        self.activation=activation
        self.solver=solver
        self.max_iter=max_iter
        self.random_state = random_state
        self.alpha = alpha
        self.n_estimators = n_estimators
    
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
            )
            ),
            (
            'rf', RandomForestRegressor(n_estimators=self.n_estimators)
            
            )]
        
        # Initialize the meta-model (Linear Regression)
        self.meta_model = LinearRegression()
        
        # Initialize the stacking ensemble
        self.stacking_model = StackingRegressor(estimators=self.base_models, final_estimator=self.meta_model)
        
        time = X_test['T']
        lat = X_test['Y']
        lon = X_test['X']
        
        n_time = len(X_test.coords['T'])
        n_lat  = len(X_test.coords['Y'])
        n_lon  = len(X_test.coords['X'])

        
        X_train_ = X_train.stack(sample=('T', 'Y', 'X')) #.fillna(0)
        X_train_ = X_train_.transpose('sample', 'M')
        X_train_ = X_train_.values
        
        X_train_nan_indices = np.any(np.isnan(X_train_), axis=1)   
        X_train_array_with_nan = X_train_[X_train_nan_indices]
        X_train_array_without_nan = X_train_[~X_train_nan_indices]
               
        y_train_ = y_train.stack(sample=('T', 'Y', 'X'))  #.fillna(0)
        y_train_ = y_train_.transpose('sample', 'M')
        y_train_ = y_train_.values

        y_train_nan_indices = np.any(np.isnan(y_train_), axis=1)
        y_train_array_with_nan = y_train_[y_train_nan_indices]
        y_train_array_without_nan = y_train_[~y_train_nan_indices]

        X_test_ = X_test.stack(sample=('T', 'Y', 'X')) #.fillna(0)
        X_test_ = X_test_.transpose('sample', 'M')
        X_test_ = X_test_.values 
        
        X_test_nan_indices = np.any(np.isnan(X_test_), axis=1)
        X_test_array_with_nan = X_test_[X_test_nan_indices]
        X_test_array_without_nan = X_test_[~X_test_nan_indices] 
    
        y_test_ = y_test.stack(sample=('T', 'Y', 'X')) #.fillna(0)
        y_test_ = y_test_.transpose('sample', 'M')
        y_test_ = y_test_.values 

        y_test_nan_indices = np.any(np.isnan(y_test_), axis=1)
        y_test_array_with_nan = y_test_[y_test_nan_indices]
        y_test_array_without_nan = y_test_[~y_test_nan_indices]

        self.stacking_model.fit(X_train_array_without_nan, y_train_array_without_nan)
        y_pred_test = self.stacking_model.predict(X_test_array_without_nan)

        result = np.empty_like(np.squeeze(y_test_))
        result[np.squeeze(y_test_nan_indices)] = np.squeeze(y_test_array_with_nan)
        result[~np.squeeze(y_test_nan_indices)] = y_pred_test  

        
        predictions_reshaped = result.reshape(n_time, n_lat, n_lon)
        predicted_da = xr.DataArray(
            data=predictions_reshaped,
            coords={
                'T': time,
                'Y': lat,
                'X': lon
            },
            dims=['T', 'Y', 'X'],
        )  
        return predicted_da

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


    def compute_prob(self, Predictant, clim_year_start, clim_year_end, hindcast_det):
        
        Predictant = Predictant.transpose('T', 'M', 'Y', 'X').drop_vars('M').squeeze('M')
        index_start = Predictant.get_index("T").get_loc(str(clim_year_start)).start
        index_end = Predictant.get_index("T").get_loc(str(clim_year_end)).stop

        rainfall_for_tercile = Predictant.isel(T=slice(index_start, index_end))
        terciles = rainfall_for_tercile.quantile([0.3, 0.67], dim='T')
        error_variance = (Predictant - hindcast_det).var(dim='T')

        dof = len(Predictant.get_index("T"))

        hindcast_prob = xr.apply_ufunc(
            self.calculate_tercile_probabilities_gamma,
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
            dask_gufunc_kwargs={'output_sizes': {'probability': 3}, "allow_rechunk":True},
        )

        hindcast_prob = hindcast_prob.assign_coords(probability=('probability', ['PB', 'PN', 'PA']))
        return hindcast_prob.transpose('probability', 'T', 'Y', 'X')

    def forecast(self, Predictant, clim_year_start, clim_year_end, hindcast_det, hindcast_det_cross, Predictor_for_year):
        
        mask = xr.where(~np.isnan(Predictant.isel(T=0,M=0)), 1, np.nan).drop_vars(['T','M']).squeeze()
        mask.name = None
        
        Predictor_for_year_st = (Predictor_for_year - hindcast_det.sel(T=slice(str(clim_year_start),str(clim_year_end))).mean(dim='T'))/hindcast_det.sel(T=slice(str(clim_year_start),str(clim_year_end))).std(dim='T')  
        hindcast_det_st = standardize_timeseries(hindcast_det, clim_year_start, clim_year_end)
        Predictant_st = standardize_timeseries(Predictant, clim_year_start, clim_year_end) 
        y_test = Predictant_st.isel(T=[-1])
        
        # Initialize the base models (MLP and Random Forest)
        self.base_models = [
            ('mlp', MLPRegressor(
            hidden_layer_sizes=self.hidden_layer_sizes,
            activation=self.activation,
            solver=self.solver,
            max_iter=self.max_iter,
            random_state=self.random_state,
            alpha=self.alpha
            )
            ),
            ('rf', RandomForestRegressor(n_estimators=self.n_estimators)
            
            )]
        
        # Initialize the meta-model (Linear Regression)
        self.meta_model = LinearRegression()
        
        # Initialize the stacking ensemble
        self.stacking_model = StackingRegressor(estimators=self.base_models, final_estimator=self.meta_model)

        
        
        time = Predictor_for_year_st['T']
        lat = Predictor_for_year_st['Y']
        lon = Predictor_for_year_st['X']
        
        n_time = len(Predictor_for_year_st.coords['T'])
        n_lat  = len(Predictor_for_year_st.coords['Y'])
        n_lon  = len(Predictor_for_year_st.coords['X'])
        

        
        hindcast_det_ = hindcast_det_st.stack(sample=('T', 'Y', 'X'))
        hindcast_det_ = hindcast_det_.transpose('sample', 'M')
        hindcast_det_ = hindcast_det_.values

        hindcast_det_nan_indices = np.any(np.isnan(hindcast_det_), axis=1)   
        hindcast_det_array_with_nan = hindcast_det_[hindcast_det_nan_indices]
        hindcast_det_array_without_nan = hindcast_det_[~hindcast_det_nan_indices]        

        Predictant_ = Predictant_st.stack(sample=('T', 'Y', 'X'))
        Predictant_ = Predictant_.transpose('sample', 'M')
        Predictant_ = Predictant_.values
        
        Predictant_nan_indices = np.any(np.isnan(Predictant_), axis=1)
        Predictant_array_with_nan = Predictant_[Predictant_nan_indices]
        Predictant_array_without_nan = Predictant_[~Predictant_nan_indices]

        Predictor_for_year_ = Predictor_for_year_st.stack(sample=('T', 'Y', 'X'))
        Predictor_for_year_ = Predictor_for_year_.transpose('sample', 'M')
        Predictor_for_year_ = Predictor_for_year_.values 
        
        Predictor_for_year_nan_indices = np.any(np.isnan(Predictor_for_year_), axis=1)
        Predictor_for_year_array_with_nan = Predictor_for_year_[Predictor_for_year_nan_indices]
        Predictor_for_year_array_without_nan = Predictor_for_year_[~Predictor_for_year_nan_indices]

        y_test_ = y_test.stack(sample=('T', 'Y', 'X')) #.fillna(0)
        y_test_ = y_test_.transpose('sample', 'M')
        y_test_ = y_test_.values 

        y_test_nan_indices = np.any(np.isnan(y_test_), axis=1)
        y_test_array_with_nan = y_test_[y_test_nan_indices]
        y_test_array_without_nan = y_test_[~y_test_nan_indices]        
        
        self.stacking_model.fit(hindcast_det_array_without_nan, Predictant_array_without_nan)
        y_pred_test = self.stacking_model.predict(Predictor_for_year_array_without_nan)
        
        result = np.empty_like(np.squeeze(y_test_))
        result[np.squeeze(y_test_nan_indices)] = np.squeeze(y_test_array_with_nan)
        result[~np.squeeze(y_test_nan_indices)] = y_pred_test  
        
        predictions_reshaped = result.reshape(n_time, n_lat, n_lon)
        result_ = xr.DataArray(
            data=predictions_reshaped,
            coords={
                'T': time,
                'Y': lat,
                'X': lon
            },
            dims=['T', 'Y', 'X'],
        )*mask  

        result_ = reverse_standardize(result_, Predictant.isel(M=0).drop_vars("M").squeeze(), clim_year_start, clim_year_end)
        
        
        index_start = Predictant.get_index("T").get_loc(str(clim_year_start)).start
        index_end = Predictant.get_index("T").get_loc(str(clim_year_end)).stop

        rainfall_for_tercile = Predictant.isel(T=slice(index_start, index_end))
        terciles = rainfall_for_tercile.quantile([0.3, 0.67], dim='T')
        error_variance = (Predictant - hindcast_det_cross).var(dim='T')

        dof = len(Predictant.get_index("T"))

        hindcast_prob = xr.apply_ufunc(
            self.calculate_tercile_probabilities_gamma,
            result_,#.expand_dims({'T': [pd.Timestamp(Predictor_for_year.coords['T'].values).to_pydatetime()]}),
            error_variance,
            terciles.isel(quantile=0).drop_vars('quantile'),
            terciles.isel(quantile=1).drop_vars('quantile'),
            input_core_dims=[('T',), (), (), ()],
            vectorize=True,
            kwargs={'dof': dof},
            dask='parallelized',
            output_core_dims=[('probability', 'T',)],
            output_dtypes=['float'],
            dask_gufunc_kwargs={'output_sizes': {'probability': 3}, "allow_rechunk":True},
        )
        hindcast_prob = hindcast_prob.assign_coords(probability=('probability', ['PB', 'PN', 'PA']))
        return result_*mask, mask*hindcast_prob.drop_vars('T').squeeze().transpose('probability', 'Y', 'X')

class WAS_mme_Stack_MLP_AdaBoost:
    def __init__(self, hidden_layer_sizes=(10,5), activation='relu', max_iter=200, solver='adam', random_state=42, alpha=0.01, n_estimators = 100):

        self.hidden_layer_sizes=hidden_layer_sizes
        self.activation=activation
        self.solver=solver
        self.max_iter=max_iter
        self.random_state = 42
        self.alpha = 0.01 
        self.n_estimators = 50 

    def compute_model(self, X_train, y_train, X_test):
        
        self.mlp_model = MLPRegressor(
            hidden_layer_sizes=self.hidden_layer_sizes,
            activation=self.activation,
            solver=self.solver,
            max_iter=self.max_iter,
            random_state=self.random_state,
            alpha=self.alpha
        )
        
        # Initialize the boosting ensemble
        self.boosting_model = AdaBoostRegressor(self.mlp_model, n_estimators=self.n_estimators, random_state=self.random_state)

        
        time = X_test['T']
        lat = X_test['Y']
        lon = X_test['X']
        
        n_time = len(X_test.coords['T'])
        n_lat  = len(X_test.coords['Y'])
        n_lon  = len(X_test.coords['X'])
                
        X_train_ = X_train.fillna(0).stack(sample=('T', 'Y', 'X'))
        X_train_ = X_train_.transpose('sample', 'M')
        X_train_ = X_train_.values

        y_train_ = y_train.fillna(0).stack(sample=('T', 'Y', 'X'))
        y_train_ = y_train_.transpose('sample', 'M')
        y_train_ = y_train_.values

        X_test_ = X_test.fillna(0).stack(sample=('T', 'Y', 'X'))
        X_test_ = X_test_.transpose('sample', 'M')
        X_test_ = X_test_.values 
        
        # valid_mask = np.isfinite(y_train_)
        # y_train_test = y_train.isel(T=[0]).stack(sample=('T', 'Y', 'X'))
        # y_train_test = y_train_test.transpose('sample', 'M')
        # y_train_test = y_train_test.values
        # valid_mask_test = np.isfinite(y_train_test)
  
        # X_train_ = X_train_[valid_mask[:,0],:]
        # y_train_ = y_train_[valid_mask]
        # X_test_ = X_test_[valid_mask_test[:,0],:]

        self.boosting_model.fit(X_train_, y_train_)
        y_pred_test = self.boosting_model.predict(X_test_)
        
        predictions_reshaped = y_pred_test.reshape(n_time, n_lat, n_lon)
        predicted_da = xr.DataArray(
            data=predictions_reshaped,
            coords={
                'T': time,
                'Y': lat,
                'X': lon
            },
            dims=['T', 'Y', 'X'],
        )  
        
        return predicted_da

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


    def compute_prob(self, Predictant, clim_year_start, clim_year_end, hindcast_det):
        
        Predictant = Predictant.transpose('T', 'M', 'Y', 'X').drop_vars('M').squeeze('M')
        index_start = Predictant.get_index("T").get_loc(str(clim_year_start)).start
        index_end = Predictant.get_index("T").get_loc(str(clim_year_end)).stop

        rainfall_for_tercile = Predictant.isel(T=slice(index_start, index_end))
        terciles = rainfall_for_tercile.quantile([0.3, 0.67], dim='T')
        error_variance = (Predictant - hindcast_det).var(dim='T')

        dof = len(Predictant.get_index("T"))

        hindcast_prob = xr.apply_ufunc(
            self.calculate_tercile_probabilities_gamma,
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
            dask_gufunc_kwargs={'output_sizes': {'probability': 3}, "allow_rechunk":True},
        )

        hindcast_prob = hindcast_prob.assign_coords(probability=('probability', ['PB', 'PN', 'PA']))
        return hindcast_prob.transpose('probability', 'T', 'Y', 'X')

    def forecast(self, Predictant, clim_year_start, clim_year_end, hindcast_det, hindcast_det_cross, Predictor_for_year):
        
        mask = xr.where(~np.isnan(Predictant.isel(T=0,M=0)), 1, np.nan).drop_vars(['T','M']).squeeze()
        mask.name = None
        
        Predictor_for_year_st = (Predictor_for_year - hindcast_det.sel(slice(str(clim_year_start),str(clim_year_end))).mean(dim='T'))/hindcast_det.sel(slice(str(clim_year_start),str(clim_year_end))).std(dim='T')  
        hindcast_det_st = standardize_timeseries(hindcast_det, clim_year_start, clim_year_end)
        Predictant_st = standardize_timeseries(Predictant, clim_year_start, clim_year_end) 
        
        self.mlp_model = MLPRegressor(
            hidden_layer_sizes=self.hidden_layer_sizes,
            activation=self.activation,
            solver=self.solver,
            max_iter=self.max_iter,
            random_state=self.random_state,
            alpha=self.alpha
        )
        
        time = Predictor_for_year_['T']
        lat = Predictor_for_year_['Y']
        lon = Predictor_for_year_['X']
        
        n_time = len(Predictor_for_year_.coords['T'])
        n_lat  = len(Predictor_for_year_.coords['Y'])
        n_lon  = len(Predictor_for_year_.coords['X'])
        

        
        hindcast_det_ = hindcast_det_st.fillna(0).stack(sample=('T', 'Y', 'X'))
        hindcast_det_ = hindcast_det_.transpose('sample', 'M')
        hindcast_det_ = hindcast_det_.values

        Predictant_ = Predictant_st.fillna(0).stack(sample=('T', 'Y', 'X'))
        Predictant_ = Predictant_.transpose('sample', 'M')
        Predictant_ = Predictant_.values

        Predictor_for_year_ = Predictor_for_year_st.fillna(0).stack(sample=('T', 'Y', 'X'))
        Predictor_for_year_ = Predictor_for_year_.transpose('sample', 'M')
        Predictor_for_year_ = Predictor_for_year_.values 

        self.mlp_model.fit(hindcast_det_, Predictant_)
        y_pred_test = self.mlp_model.predict(Predictor_for_year_)
        
        predictions_reshaped = y_pred_test.reshape(n_time, n_lat, n_lon)
        result_ = xr.DataArray(
            data=predictions_reshaped,
            coords={
                'T': time,
                'Y': lat,
                'X': lon
            },
            dims=['T', 'Y', 'X'],
        )*mask  

        result_ = reverse_standardize(result_, Predictant.isel(M=0).drop_vars("M").squeeze(), clim_year_start, clim_year_end)
        
        
        index_start = Predictant.get_index("T").get_loc(str(clim_year_start)).start
        index_end = Predictant.get_index("T").get_loc(str(clim_year_end)).stop

        rainfall_for_tercile = Predictant.isel(T=slice(index_start, index_end))
        terciles = rainfall_for_tercile.quantile([0.3, 0.67], dim='T')
        error_variance = (Predictant - hindcast_det_cross).var(dim='T')

        dof = len(Predictant.get_index("T"))

        hindcast_prob = xr.apply_ufunc(
            self.calculate_tercile_probabilities_gamma,
            result_,#.expand_dims({'T': [pd.Timestamp(Predictor_for_year.coords['T'].values).to_pydatetime()]}),
            error_variance,
            terciles.isel(quantile=0).drop_vars('quantile'),
            terciles.isel(quantile=1).drop_vars('quantile'),
            input_core_dims=[('T',), (), (), ()],
            vectorize=True,
            kwargs={'dof': dof},
            dask='parallelized',
            output_core_dims=[('probability', 'T',)],
            output_dtypes=['float'],
            dask_gufunc_kwargs={'output_sizes': {'probability': 3}, "allow_rechunk":True},
        )
        hindcast_prob = hindcast_prob.assign_coords(probability=('probability', ['PB', 'PN', 'PA']))
        return result_*mask, mask*hindcast_prob.drop_vars('T').squeeze().transpose('probability', 'Y', 'X')

class WAS_mme_ELM_:
    pass
# # ---------------------------------------------------------------------
# # 1. DEFINE A SIMPLE ELM CLASS
# # ---------------------------------------------------------------------
# # class ExtremeLearningMachine:
#     """
#     A simple implementation of a single-hidden-layer ELM for regression.
    
#     Attributes:
#         hidden_layer_size (int): Number of neurons in the hidden layer.
#         activation (callable): Activation function for the hidden layer.
#     """
#     def __init__(self, hidden_layer_size=50, activation='sigmoid', random_state=None):
#         self.hidden_layer_size = hidden_layer_size
#         self.random_state = random_state
#         if activation == 'sigmoid':
#             self.activation = self._sigmoid
#         elif activation == 'tanh':
#             self.activation = np.tanh
#         elif activation == 'relu':
#             self.activation = self._relu
#         else:
#             raise ValueError("Unsupported activation. Choose from ['sigmoid','tanh','relu']")
        
#     def _sigmoid(self, x):
#         return 1.0 / (1.0 + np.exp(-x))
    
#     def _relu(self, x):
#         return np.maximum(0, x)
    
#     def fit(self, X, y):
#         """
#         Fit the ELM model on training data.
        
#         Args:
#             X (ndarray): Shape [N, D], input features.
#             y (ndarray): Shape [N, ] or [N, 1], target values.
#         """
#         # Set random seed if provided
#         if self.random_state is not None:
#             np.random.seed(self.random_state)

#         N, D = X.shape

#         # 1. Randomly initialize hidden-layer weights and biases
#         self.W = np.random.randn(D, self.hidden_layer_size)  # input-to-hidden weights
#         self.b = np.random.randn(self.hidden_layer_size)     # hidden bias

#         # 2. Compute hidden-layer output matrix H
#         H = self.activation(X @ self.W + self.b)

#         # 3. Solve for output weights (Beta) using pseudo-inverse: Beta = pinv(H) @ y
#         #    Ensure y is 2D if you want multi-target output. For single target, 1D is fine.
#         #    but for consistent matrix multiplication, we reshape y to [N,1].
#         if y.ndim == 1:
#             y = y.reshape(-1, 1)
#         # Compute pseudo-inverse of H
#         # A stable approach: Beta = np.linalg.pinv(H) @ y
#         self.Beta = np.linalg.pinv(H) @ y

#     def predict(self, X):
#         """
#         Generate predictions for new inputs.
        
#         Args:
#             X (ndarray): Shape [N, D], input features.
        
#         Returns:
#             (ndarray): Predictions of shape [N, ].
#         """
#         # Compute hidden-layer output
#         H = self.activation(X @ self.W + self.b)
#         # Multiply by output weights
#         y_pred = H @ self.Beta
#         # Flatten if it’s single target
#         return y_pred.ravel()

# # ---------------------------------------------------------------------
# # 2. LOAD/READ DATA (EXAMPLE)
# # ---------------------------------------------------------------------

# # NOTE: Replace 'forecastX.nc' and 'observed.nc' with your actual file paths/names
# # Each dataset should have dimensions [time, lat, lon] and a variable 'precip'.
# ds_forecast1 = xr.open_dataset('forecast1.nc')  
# ds_forecast2 = xr.open_dataset('forecast2.nc')
# ds_forecast3 = xr.open_dataset('forecast3.nc')
# ds_observed  = xr.open_dataset('observed.nc')

# f1 = ds_forecast1['precip']  # shape: [time, lat, lon]
# f2 = ds_forecast2['precip']
# f3 = ds_forecast3['precip']
# obs = ds_observed['precip']

# # Example alignment (if needed, adjust accordingly):
# # f1 = f1.sel(time=obs.time, lat=obs.lat, lon=obs.lon)
# # f2 = f2.sel(time=obs.time, lat=obs.lat, lon=obs.lon)
# # f3 = f3.sel(time=obs.time, lat=obs.lat, lon=obs.lon)

# # ---------------------------------------------------------------------
# # 3. FLATTEN DATA
# # ---------------------------------------------------------------------
# f1_flat  = f1.values.ravel()     # shape: [time*lat*lon]
# f2_flat  = f2.values.ravel()
# f3_flat  = f3.values.ravel()
# obs_flat = obs.values.ravel()

# # Stack features into [N, 3]
# X = np.column_stack((f1_flat, f2_flat, f3_flat))  # shape: [N, 3]
# y = obs_flat                                       # shape: [N,]

# # Remove invalid (NaN) samples
# valid_mask = ~np.isnan(X).any(axis=1) & ~np.isnan(y)
# X = X[valid_mask]
# y = y[valid_mask]

# # ---------------------------------------------------------------------
# # 4. TRAIN-TEST SPLIT
# # ---------------------------------------------------------------------
# X_train, X_test, y_train, y_test = train_test_split(
#     X, y, test_size=0.2, random_state=42
# )

# # ---------------------------------------------------------------------
# # 5. BUILD AND TRAIN THE ELM
# # ---------------------------------------------------------------------
# elm = ExtremeLearningMachine(
#     hidden_layer_size=50,
#     activation='sigmoid',  # or 'tanh', 'relu'
#     random_state=42
# )

# print("Fitting ELM model...")
# elm.fit(X_train, y_train)

# # ---------------------------------------------------------------------
# # 6. EVALUATE
# # ---------------------------------------------------------------------
# y_pred_test = elm.predict(X_test)

# rmse_test = np.sqrt(mean_squared_error(y_test, y_pred_test))
# r2_test   = r2_score(y_test, y_pred_test)

# print(f"Test RMSE: {rmse_test:.3f}")
# print(f"Test R^2:  {r2_test:.3f}")

# # ---------------------------------------------------------------------
# # 7. OPTIONAL: RE-GRID PREDICTIONS FOR ENTIRE DATASET
# # ---------------------------------------------------------------------
# # If you want to produce a recalibrated forecast for the entire domain/time:
# all_predictions = elm.predict(X)

# n_time = len(f1.time)
# n_lat  = len(f1.lat)
# n_lon  = len(f1.lon)

# all_predictions_reshaped = np.full((n_time*n_lat*n_lon,), np.nan)
# all_predictions_reshaped[valid_mask] = all_predictions  # fill only valid cells
# all_predictions_reshaped = all_predictions_reshaped.reshape(n_time, n_lat, n_lon)

# predicted_da = xr.DataArray(
#     data=all_predictions_reshaped,
#     coords={
#         'time': f1.time,
#         'lat': f1.lat,
#         'lon': f1.lon
#     },
#     dims=['time', 'lat', 'lon'],
#     name='precip_calibrated'
# )

# # Save to NetCDF if desired
# predicted_da.to_netcdf('precip_calibrated_elm.nc')

