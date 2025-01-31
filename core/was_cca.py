import numpy as np
import xarray as xr
from scipy import stats
from scipy import signal as sig
from scipy.stats import gamma
import matplotlib.pyplot as plt
import xeofs as xe
from core.utils import *


class WAS_CCA_:
    def __init__(self, n_modes=5, n_pca_modes=10, standardize=False, use_coslat=True, use_pca=True):
        """
        Initialize the WAS_CCA class with specified parameters.

        Parameters:
        - n_modes: Number of canonical modes to compute.
        - n_pca_modes: Number of PCA modes to use before CCA.
        - standardize: Whether to standardize the data. Keep it False in our case data already standardize
        - use_coslat: Whether to use cosine latitude weighting.
        - use_pca: Whether to perform PCA before CCA.
        - detrend: Whether to apply detrending to the data.
        """
        self.n_modes = n_modes
        self.n_pca_modes = n_pca_modes
        self.standardize = standardize
        self.use_coslat = use_coslat
        self.use_pca = use_pca
        # self.detrend = detrend
        self.cca = xe.cross.CCA(
            n_modes=self.n_modes,
            standardize=self.standardize,
            use_coslat=self.use_coslat,
            use_pca=self.use_pca,
            n_pca_modes=self.n_pca_modes
        )
        self.cca_model = None
    
    def fit_cca(self, X_train, y_train):
        """
        Fit the CCA model using the training data.

        Parameters:
        - X_train: xarray DataArray for predictor training data.
        - y_train: xarray DataArray for predictand training data.
        """
        # Preprocess the data
        X_train_final, y_train_final = self.preprocess_data(X_train, y_train)
        # Fit the CCA model
        self.cca_model = self.cca.fit(X_train_final, y_train_final, dim="T")

    # def detrend_data(self, data): ### Replace detrend by extendedEOF after
    #     """
    #     Detrend the data along the 'T' dimension if detrending is enabled.

    #     Parameters:
    #     - data: xarray DataArray to be detrended.

    #     Returns:
    #     - data_detrended: Detrended xarray DataArray.
    #     """
    #     if self.detrend:
    #         # Create mask for missing data
    #         mask = xr.where(np.isnan(data), np.nan, 1)
    #         # mask = np.where(np.isnan(data.isel(T=0)), np.nan, 1)
    #         # Fill missing values with zero before detrending
    #         data_filled = data.fillna(0)
    #         # Detrend data along 'T' axis (axis=0)
    #         data_detrended = sig.detrend(data_filled, axis=0)
    #         data_detrended = xr.DataArray(data_detrended, dims=data.dims, coords=data.coords)
    #         # Apply mask after detrending
    #         # data_detrended = data_detrended * mask
    #         return data_detrended
    #     else:
    #         return data

    # def preprocess_data(self, X, Y):
    #     """
    #     Preprocess the data by detrending, masking, and filling missing values.

    #     Parameters:
    #     - X: xarray DataArray for predictors.
    #     - Y: xarray DataArray for predictands.

    #     Returns:
    #     - X_final: Preprocessed X data.
    #     - Y_final: Preprocessed Y data.
    #     """
    #     # Apply detrending and masking (masking is now inside detrend_data)
    #     X_processed = self.detrend_data(X)
    #     # Fill missing values with mean along 'T'
    #     X_final = X_processed.fillna(X_processed.mean(dim="T", skipna=True))

    #     # Process Y data (assuming we do not detrend Y)
    #     Y_final = Y.fillna(Y.mean(dim="T", skipna=True))

    #     # Rename dimensions and transpose
    #     dims_rename = {"X": "lon", "Y": "lat"}
    #     X_final = X_final.rename(dims_rename).transpose('T', 'lat', 'lon')
    #     Y_final = Y_final.rename(dims_rename).transpose('T', 'lat', 'lon')

    #     return X_final, Y_final

    def preprocess_data(self, X, Y):
        """
        Preprocess the data by detrending, masking, and filling missing values.

        Parameters:
        - X: xarray DataArray for predictors.
        - Y: xarray DataArray for predictands.

        Returns:
        - X_final: Preprocessed X data.
        - Y_final: Preprocessed Y data.
        """
        # Apply detrending to both X and Y
        X_processed = X #- self.detrend_data(X)
        Y_processed = Y #- self.detrend_data(Y)

        # Fill missing values with mean along 'T'
        X_final = X_processed.fillna(X_processed.mean(dim="T", skipna=True))
        Y_final = Y_processed.fillna(Y_processed.mean(dim="T", skipna=True))

        # Rename dimensions and transpose
        dims_rename = {"X": "lon", "Y": "lat"}
        X_final = X_final.rename(dims_rename).transpose('T', 'lat', 'lon')
        Y_final = Y_final.rename(dims_rename).transpose('T', 'lat', 'lon')

        return X_final, Y_final

    def preprocess_test_data(self, X_test, y_test, X_train, y_train):
        """
        Preprocess the test data.

        Parameters:
        - X_test: xarray DataArray for predictor testing data.
        - y_test: xarray DataArray for predictand testing data.
        - X_train: xarray DataArray for predictor training data.
        - y_train: xarray DataArray for predictand training data.

        Returns:
        - X_test_prepared: Preprocessed X test data.
        - y_test_prepared: Preprocessed Y test data.
        """
        # Apply detrending and masking
        X_test_processed = X_test #- self.detrend_data(X_train).mean(dim="T", skipna=True)
        y_test_processed = y_test 
        
        # Fill missing values with mean from training data along 'T'
        X_test_prepared = X_test_processed.fillna(X_train.mean(dim="T", skipna=True))        
        y_test_prepared = y_test_processed.fillna(y_train.mean(dim="T", skipna=True))

        # Rename dimensions and transpose
        dims_rename = {"X": "lon", "Y": "lat"}
        X_test_prepared = X_test_prepared.rename(dims_rename).transpose('T', 'lat', 'lon')
        y_test_prepared = y_test_prepared.rename(dims_rename).transpose('T', 'lat', 'lon')

        return X_test_prepared, y_test_prepared

    def compute_model(self, X_train, y_train, X_test, y_test):
        """
        Compute the CCA model and generate hindcasts.

        Parameters:
        - X_train: xarray DataArray for predictor training data.
        - y_train: xarray DataArray for predictand training data.
        - X_test: xarray DataArray for predictor testing data.
        - y_test: xarray DataArray for predictand testing data.

        Returns:
        - hindcast: xarray DataArray containing predictions and errors.
        """
        
        
        # Fit the CCA model
        
        self.fit_cca(X_train, y_train)
            
        # Prepare test data
        X_test_prepared, y_test_prepared = self.preprocess_test_data(X_test, y_test, X_train, y_train)

        # Predict
        y_pred = self.cca_model.predict(X_test_prepared)
        y_pred = self.cca_model.inverse_transform(self.cca_model.transform(X_test_prepared), y_pred)[1] 

        y_pred['T'] = y_test_prepared['T']
        # Calculate error
        # error = y_test_prepared - y_pred

        # Combine prediction and error into a DataArray
        # hindcast = xr.concat([error, y_pred], dim="output")
        hindcast = y_pred.rename({"lon": "X", "lat": "Y"})
        # hindcast = hindcast.assign_coords(output=['error', 'prediction'])

        return hindcast

    @staticmethod
    def calculate_tercile_probabilities(best_guess, error_variance, first_tercile, second_tercile, dof):
        """
        Calculates the probability of each tercile category.
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
        """
        Computes tercile category probabilities for hindcasts over a climatological period.
        """
        index_start = Predictant.get_index("T").get_loc(str(clim_year_start)).start
        index_end = Predictant.get_index("T").get_loc(str(clim_year_end)).stop

        rainfall_for_tercile = Predictant.isel(T=slice(index_start, index_end))
        terciles = rainfall_for_tercile.quantile([0.333, 0.667], dim='T')
        error_variance = (Predictant - hindcast_det).var(dim='T')

        # Calculate degrees of freedom
        dof = len(Predictant.get_index("T")) - 1 - (self.n_modes + 1)

        # Compute probabilities using xr.apply_ufunc
        hindcast_prob = xr.apply_ufunc(
            self.calculate_tercile_probabilities_gamma,
            hindcast_det,#.sel(output="prediction").drop_vars("output").squeeze(),
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
        mask = xr.where(~np.isnan(Predictant.isel(T=0)), 1, np.nan).drop_vars(['T']).squeeze().to_numpy()
        Predictor_ = (Predictor - trend_data(Predictor).fillna(trend_data(Predictor)[-3])).fillna(0)
        Predictant_st = standardize_timeseries(Predictant, clim_year_start, clim_year_end)
        Predictant_ = (Predictant_st - trend_data(Predictant_st).fillna(trend_data(Predictant_st)[-3])).fillna(0)
        
        Predictor_for_year_ = ((((Predictor_for_year.fillna(Predictor.mean(dim="T", skipna=True))).ffill(dim="Y").bfill(dim="Y")).ffill(dim="X").bfill(dim="X")).fillna(0)).transpose('T', 'Y', 'X')

        # last_trend_X = ((trend_data(standardize_timeseries(Predictor, clim_year_start, clim_year_end))).isel(T=[-3]))
        last_trend_X = ((trend_data(Predictor)).isel(T=[-3]))
        last_trend_X['T'] = Predictor_for_year_['T']
        # Predictor_for_year__ = Predictor_for_year_.fillna(0)
        Predictor_for_year__ = (Predictor_for_year_ - last_trend_X).fillna(0)

        # Fit the CCA model
        self.fit_cca(Predictor_, Predictant_)
            
        # Prepare test data
        X_test_prepared = Predictor_for_year__.rename({"X": "lon", "Y": "lat"}).transpose('T', 'lat', 'lon')

        # Predict
        y_pred = self.cca_model.predict(X_test_prepared)
        y_pred = self.cca_model.inverse_transform(self.cca_model.transform(X_test_prepared), y_pred)[1] 
        result_ = y_pred.rename({"lon": "X", "lat": "Y"})
        
        last_trend_Y = ((trend_data(Predictant_st)).isel(T=[-3]))
        last_trend_Y['T'] = result_['T']
        result_ = (result_ + last_trend_Y)
        result_ = reverse_standardize(result_, Predictant, clim_year_start, clim_year_end) 
        
        index_start = Predictant_.get_index("T").get_loc(str(clim_year_start)).start
        index_end = Predictant_.get_index("T").get_loc(str(clim_year_end)).stop
        rainfall_for_tercile = Predictant_.isel(T=slice(index_start, index_end))
        terciles = rainfall_for_tercile.quantile([0.333, 0.667], dim='T')
        error_variance = (Predictant - hindcast_det).var(dim='T')
        dof = len(Predictant_.get_index("T")) - 1 - (self.n_modes + 1)
        
        hindcast_prob = xr.apply_ufunc(
            self.calculate_tercile_probabilities_gamma,
            result_,#.expand_dims({'T':1}),
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
        return result_*mask, hindcast_prob.drop_vars('T').squeeze().transpose('probability', 'Y', 'X')*mask
        

    def plot_cca_results(self, X=None, Y=None, n_modes=None, clim_year_start=None, clim_year_end=None):
        """
        Plots the CCA modes and scores.

        Parameters:
        - X: Optional xarray DataArray for predictors. If provided, the model will be fitted using X and Y.
        - Y: Optional xarray DataArray for predictands.
        - n_modes: Number of modes to plot. If None, plots all modes.
        """
        if X is not None and Y is not None:
            mask = xr.where(~np.isnan(Y.isel(T=0)), 1, np.nan).drop_vars(['T']).squeeze().to_numpy()
            # mask.name = None
            
            X_ = standardize_timeseries(X, clim_year_start, clim_year_end) - trend_data(standardize_timeseries(X, clim_year_start, clim_year_end)).fillna(trend_data(standardize_timeseries(X, clim_year_start, clim_year_end))[-3])
            Y_ = standardize_timeseries(Y, clim_year_start, clim_year_end) - trend_data(standardize_timeseries(Y, clim_year_start, clim_year_end)).fillna(trend_data(standardize_timeseries(Y, clim_year_start, clim_year_end))[-3])
            
            # Fit the model using the provided data
            self.fit_cca(X_.isel(T= slice(0,-2)).fillna(0), Y_.isel(T=slice(0,-2)).fillna(0))
        elif self.cca_model is None:
            raise ValueError("The CCA model has not been fitted yet. Provide X and Y data to fit the model.")

        # Get components (modes) and scores
        X_modes, Y_modes = self.cca_model.components()  # Spatial patterns
        X_scores, Y_scores = self.cca_model.scores()    # Temporal projections (canonical variates)

        # Get explained variances
        var_explained_X = self.cca_model.fraction_variance_X_explained_by_X()
        var_explained_Y = self.cca_model.fraction_variance_Y_explained_by_Y()
        var_explained_Y_by_X = self.cca_model.fraction_variance_Y_explained_by_X()

        # Determine number of modes to plot
        if n_modes is None:
            n_modes = self.n_modes

        # Mode indices start from 1 in xeofs
        mode_indices = range(1, n_modes + 1)

        # Create subplots
        fig, axes = plt.subplots(n_modes, 3, figsize=(15, 3 * n_modes))

        if n_modes == 1:
            axes = np.array([axes])

        for i, mode in enumerate(mode_indices):

            # First Column: Plot X_modes
            ax = axes[i, 0]
            X_mode = X_modes.sel(mode=mode)
            X_mode.plot(ax=ax, vmin=-1, vmax=1, cmap= "RdBu_r")
            var_X = var_explained_X.sel(mode=mode).values * 100
            ax.set_title(f'X Mode {mode} ({var_X:.2f}% variance explained)')

            # Second Column: Plot X_scores and Y_scores
            ax = axes[i, 1]
            X_score = X_scores.sel(mode=mode)
            Y_score = Y_scores.sel(mode=mode)
            var_Y_X = var_explained_Y_by_X.sel(mode=mode).values * 100
            ax.plot(X_score['T'], X_score, label='X Score')
            ax.plot(Y_score['T'], Y_score, label='Y Score')
            ax.axhline(0, linestyle='--', lw=0.8, label="") #### line Canonical Variate = 0
            ax.legend()
            ax.set_title(f'Scores for Mode {mode} ({var_Y_X:.2f}% variance Y explained by X)')
            ax.set_xlabel('Time')
            ax.set_ylabel('Canonical Variate')

            # Third Column: Plot Y_modes
            ax = axes[i, 2]
            Y_mode = (Y_modes.sel(mode=mode))*mask
            Y_mode.plot(ax=ax, vmin=None, vmax=None, cmap= "RdBu_r")
            var_Y = var_explained_Y.sel(mode=mode).values * 100
            ax.set_title(f'Y Mode {mode} ({var_Y:.2f}% variance explained)')

        plt.tight_layout()
        plt.show()



class WAS_CCA:
    def __init__(self, n_modes=5, n_pca_modes=10, standardize=False, use_coslat=True, use_pca=True):
        """
        Initialize the WAS_CCA class with specified parameters.

        Parameters:
        - n_modes: Number of canonical modes to compute.
        - n_pca_modes: Number of PCA modes to use before CCA.
        - standardize: Whether to standardize the data. Keep it False in our case data already standardize
        - use_coslat: Whether to use cosine latitude weighting.
        - use_pca: Whether to perform PCA before CCA.
        - detrend: Whether to apply detrending to the data.
        """
        self.n_modes = n_modes
        self.n_pca_modes = n_pca_modes
        self.standardize = standardize
        self.use_coslat = use_coslat
        self.use_pca = use_pca
        # self.detrend = detrend
        self.cca = xe.cross.CCA(
            n_modes=self.n_modes,
            standardize=self.standardize,
            use_coslat=self.use_coslat,
            use_pca=self.use_pca,
            n_pca_modes=self.n_pca_modes
        )
        self.cca_model = None
    
    def fit_cca(self, X_train, y_train):
        """
        Fit the CCA model using the training data.

        Parameters:
        - X_train: xarray DataArray for predictor training data.
        - y_train: xarray DataArray for predictand training data.
        """
        # Preprocess the data
        X_train_final, y_train_final = self.preprocess_data(X_train, y_train)
        # Fit the CCA model
        self.cca_model = self.cca.fit(X_train_final, y_train_final, dim="T")

    # def detrend_data(self, data): ### Replace detrend by extendedEOF after
    #     """
    #     Detrend the data along the 'T' dimension if detrending is enabled.

    #     Parameters:
    #     - data: xarray DataArray to be detrended.

    #     Returns:
    #     - data_detrended: Detrended xarray DataArray.
    #     """
    #     if self.detrend:
    #         # Create mask for missing data
    #         mask = xr.where(np.isnan(data), np.nan, 1)
    #         # mask = np.where(np.isnan(data.isel(T=0)), np.nan, 1)
    #         # Fill missing values with zero before detrending
    #         data_filled = data.fillna(0)
    #         # Detrend data along 'T' axis (axis=0)
    #         data_detrended = sig.detrend(data_filled, axis=0)
    #         data_detrended = xr.DataArray(data_detrended, dims=data.dims, coords=data.coords)
    #         # Apply mask after detrending
    #         # data_detrended = data_detrended * mask
    #         return data_detrended
    #     else:
    #         return data

    # def preprocess_data(self, X, Y):
    #     """
    #     Preprocess the data by detrending, masking, and filling missing values.

    #     Parameters:
    #     - X: xarray DataArray for predictors.
    #     - Y: xarray DataArray for predictands.

    #     Returns:
    #     - X_final: Preprocessed X data.
    #     - Y_final: Preprocessed Y data.
    #     """
    #     # Apply detrending and masking (masking is now inside detrend_data)
    #     X_processed = self.detrend_data(X)
    #     # Fill missing values with mean along 'T'
    #     X_final = X_processed.fillna(X_processed.mean(dim="T", skipna=True))

    #     # Process Y data (assuming we do not detrend Y)
    #     Y_final = Y.fillna(Y.mean(dim="T", skipna=True))

    #     # Rename dimensions and transpose
    #     dims_rename = {"X": "lon", "Y": "lat"}
    #     X_final = X_final.rename(dims_rename).transpose('T', 'lat', 'lon')
    #     Y_final = Y_final.rename(dims_rename).transpose('T', 'lat', 'lon')

    #     return X_final, Y_final

    def preprocess_data(self, X, Y):
        """
        Preprocess the data by detrending, masking, and filling missing values.

        Parameters:
        - X: xarray DataArray for predictors.
        - Y: xarray DataArray for predictands.

        Returns:
        - X_final: Preprocessed X data.
        - Y_final: Preprocessed Y data.
        """
        # Apply detrending to both X and Y
        X_processed = X #- self.detrend_data(X)
        Y_processed = Y #- self.detrend_data(Y)

        # Fill missing values with mean along 'T'
        X_final = X_processed.fillna(X_processed.mean(dim="T", skipna=True))
        Y_final = Y_processed.fillna(Y_processed.mean(dim="T", skipna=True))

        # Rename dimensions and transpose
        dims_rename = {"X": "lon", "Y": "lat"}
        X_final = X_final.rename(dims_rename).transpose('T', 'lat', 'lon')
        Y_final = Y_final.rename(dims_rename).transpose('T', 'lat', 'lon')

        return X_final, Y_final

    def preprocess_test_data(self, X_test, y_test, X_train, y_train):
        """
        Preprocess the test data.

        Parameters:
        - X_test: xarray DataArray for predictor testing data.
        - y_test: xarray DataArray for predictand testing data.
        - X_train: xarray DataArray for predictor training data.
        - y_train: xarray DataArray for predictand training data.

        Returns:
        - X_test_prepared: Preprocessed X test data.
        - y_test_prepared: Preprocessed Y test data.
        """
        # Apply detrending and masking
        X_test_processed = X_test #- self.detrend_data(X_train).mean(dim="T", skipna=True)
        y_test_processed = y_test 
        
        # Fill missing values with mean from training data along 'T'
        X_test_prepared = X_test_processed.fillna(X_train.mean(dim="T", skipna=True))        
        y_test_prepared = y_test_processed.fillna(y_train.mean(dim="T", skipna=True))

        # Rename dimensions and transpose
        dims_rename = {"X": "lon", "Y": "lat"}
        X_test_prepared = X_test_prepared.rename(dims_rename).transpose('T', 'lat', 'lon')
        y_test_prepared = y_test_prepared.rename(dims_rename).transpose('T', 'lat', 'lon')

        return X_test_prepared, y_test_prepared

    def compute_model(self, X_train, y_train, X_test, y_test):
        """
        Compute the CCA model and generate hindcasts.

        Parameters:
        - X_train: xarray DataArray for predictor training data.
        - y_train: xarray DataArray for predictand training data.
        - X_test: xarray DataArray for predictor testing data.
        - y_test: xarray DataArray for predictand testing data.

        Returns:
        - hindcast: xarray DataArray containing predictions and errors.
        """
        
        
        # Fit the CCA model
        
        self.fit_cca(X_train, y_train)
            
        # Prepare test data
        X_test_prepared, y_test_prepared = self.preprocess_test_data(X_test, y_test, X_train, y_train)

        # Predict
        y_pred = self.cca_model.predict(X_test_prepared)
        y_pred = self.cca_model.inverse_transform(self.cca_model.transform(X_test_prepared), y_pred)[1] 

        y_pred['T'] = y_test_prepared['T']
        # Calculate error
        # error = y_test_prepared - y_pred

        # Combine prediction and error into a DataArray
        # hindcast = xr.concat([error, y_pred], dim="output")
        hindcast = y_pred.rename({"lon": "X", "lat": "Y"})
        # hindcast = hindcast.assign_coords(output=['error', 'prediction'])

        return hindcast

    @staticmethod
    def calculate_tercile_probabilities(best_guess, error_variance, first_tercile, second_tercile, dof):
        """
        Calculates the probability of each tercile category.
        """
        n_time = len(best_guess)
        pred_prob = np.empty((3, n_time))

        if np.any(np.isnan(best_guess)) or np.any(np.isnan(error_variance)):
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
        """
        Computes tercile category probabilities for hindcasts over a climatological period.
        """
        index_start = Predictant.get_index("T").get_loc(str(clim_year_start)).start
        index_end = Predictant.get_index("T").get_loc(str(clim_year_end)).stop

        rainfall_for_tercile = Predictant.isel(T=slice(index_start, index_end))
        terciles = rainfall_for_tercile.quantile([0.333, 0.667], dim='T')
        error_variance = (Predictant - hindcast_det).var(dim='T')

        # Calculate degrees of freedom
        dof = len(Predictant.get_index("T")) - 1 - (self.n_modes + 1)

        # Compute probabilities using xr.apply_ufunc
        hindcast_prob = xr.apply_ufunc(
            self.calculate_tercile_probabilities_gamma,
            hindcast_det,#.sel(output="prediction").drop_vars("output").squeeze(),
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
        mask = xr.where(~np.isnan(Predictant.isel(T=0)), 1, np.nan).drop_vars(['T']).squeeze().to_numpy()
        Predictor_ = (Predictor - trend_data(Predictor).fillna(trend_data(Predictor)[-3])).fillna(0)
        Predictant_st = standardize_timeseries(Predictant, clim_year_start, clim_year_end)
        Predictant_ = (Predictant_st - trend_data(Predictant_st).fillna(trend_data(Predictant_st)[-3])).fillna(0)
        
        Predictor_for_year_ = ((((Predictor_for_year.fillna(Predictor.mean(dim="T", skipna=True))).ffill(dim="Y").bfill(dim="Y")).ffill(dim="X").bfill(dim="X")).fillna(0)).transpose('T', 'Y', 'X')

        # last_trend_X = ((trend_data(standardize_timeseries(Predictor, clim_year_start, clim_year_end))).isel(T=[-3]))
        # last_trend_X = ((trend_data(Predictor)).isel(T=[-3]))
        # last_trend_X['T'] = Predictor_for_year_['T']
        # Predictor_for_year__ = Predictor_for_year_.fillna(0)
        # Predictor_for_year__ = (Predictor_for_year_ - last_trend_X).fillna(0)

        Predictor_for_year__ = Predictor_for_year_

        # Fit the CCA model
        self.fit_cca(Predictor_, Predictant_)
            
        # Prepare test data
        X_test_prepared = Predictor_for_year__.rename({"X": "lon", "Y": "lat"}).transpose('T', 'lat', 'lon')

        # Predict
        y_pred = self.cca_model.predict(X_test_prepared)
        y_pred = self.cca_model.inverse_transform(self.cca_model.transform(X_test_prepared), y_pred)[1] 
        result_ = y_pred.rename({"lon": "X", "lat": "Y"})
        
        # last_trend_Y = ((trend_data(Predictant_st)).isel(T=[-3]))
        # last_trend_Y['T'] = result_['T']
        # result_ = (result_ + last_trend_Y)
       
        result_ = reverse_standardize(result_, Predictant, clim_year_start, clim_year_end) 
        
        index_start = Predictant_.get_index("T").get_loc(str(clim_year_start)).start
        index_end = Predictant_.get_index("T").get_loc(str(clim_year_end)).stop
        rainfall_for_tercile = Predictant.isel(T=slice(index_start, index_end))
        terciles = rainfall_for_tercile.quantile([0.333, 0.667], dim='T')
        error_variance = (Predictant - hindcast_det).var(dim='T')
        dof = len(Predictant_.get_index("T")) - 1 - (self.n_modes + 1)
        
        hindcast_prob = xr.apply_ufunc(
            self.calculate_tercile_probabilities_gamma,
            result_,#.expand_dims({'T':1}),
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
        return result_*mask, hindcast_prob.drop_vars('T').squeeze().transpose('probability', 'Y', 'X')*mask
        

    def plot_cca_results(self, X=None, Y=None, n_modes=None, clim_year_start=None, clim_year_end=None):
        """
        Plots the CCA modes and scores.

        Parameters:
        - X: Optional xarray DataArray for predictors. If provided, the model will be fitted using X and Y.
        - Y: Optional xarray DataArray for predictands.
        - n_modes: Number of modes to plot. If None, plots all modes.
        """
        if X is not None and Y is not None:
            mask = xr.where(~np.isnan(Y.isel(T=0)), 1, np.nan).drop_vars(['T']).squeeze().to_numpy()
            # mask.name = None
            
            X_ = standardize_timeseries(X, clim_year_start, clim_year_end) - trend_data(standardize_timeseries(X, clim_year_start, clim_year_end)).fillna(trend_data(standardize_timeseries(X, clim_year_start, clim_year_end))[-3])
            Y_ = standardize_timeseries(Y, clim_year_start, clim_year_end) - trend_data(standardize_timeseries(Y, clim_year_start, clim_year_end)).fillna(trend_data(standardize_timeseries(Y, clim_year_start, clim_year_end))[-3])
            
            # Fit the model using the provided data
            self.fit_cca(X_.isel(T= slice(0,-2)).fillna(0), Y_.isel(T=slice(0,-2)).fillna(0))
        elif self.cca_model is None:
            raise ValueError("The CCA model has not been fitted yet. Provide X and Y data to fit the model.")

        # Get components (modes) and scores
        X_modes, Y_modes = self.cca_model.components()  # Spatial patterns
        X_scores, Y_scores = self.cca_model.scores()    # Temporal projections (canonical variates)

        # Get explained variances
        var_explained_X = self.cca_model.fraction_variance_X_explained_by_X()
        var_explained_Y = self.cca_model.fraction_variance_Y_explained_by_Y()
        var_explained_Y_by_X = self.cca_model.fraction_variance_Y_explained_by_X()

        # Determine number of modes to plot
        if n_modes is None:
            n_modes = self.n_modes

        # Mode indices start from 1 in xeofs
        mode_indices = range(1, n_modes + 1)

        # Create subplots
        fig, axes = plt.subplots(n_modes, 3, figsize=(15, 3 * n_modes))

        if n_modes == 1:
            axes = np.array([axes])

        for i, mode in enumerate(mode_indices):

            # First Column: Plot X_modes
            ax = axes[i, 0]
            X_mode = X_modes.sel(mode=mode)
            X_mode.plot(ax=ax, vmin=-1, vmax=1, cmap= "RdBu_r")
            var_X = var_explained_X.sel(mode=mode).values * 100
            ax.set_title(f'X Mode {mode} ({var_X:.2f}% variance explained)')

            # Second Column: Plot X_scores and Y_scores
            ax = axes[i, 1]
            X_score = X_scores.sel(mode=mode)
            Y_score = Y_scores.sel(mode=mode)
            var_Y_X = var_explained_Y_by_X.sel(mode=mode).values * 100
            ax.plot(X_score['T'], X_score, label='X Score')
            ax.plot(Y_score['T'], Y_score, label='Y Score')
            ax.axhline(0, linestyle='--', lw=0.8, label="") #### line Canonical Variate = 0
            ax.legend()
            ax.set_title(f'Scores for Mode {mode} ({var_Y_X:.2f}% variance Y explained by X)')
            ax.set_xlabel('Time')
            ax.set_ylabel('Canonical Variate')

            # Third Column: Plot Y_modes
            ax = axes[i, 2]
            Y_mode = (Y_modes.sel(mode=mode))*mask
            Y_mode.plot(ax=ax, vmin=None, vmax=None, cmap= "RdBu_r")
            var_Y = var_explained_Y.sel(mode=mode).values * 100
            ax.set_title(f'Y Mode {mode} ({var_Y:.2f}% variance explained)')

        plt.tight_layout()
        plt.show()
