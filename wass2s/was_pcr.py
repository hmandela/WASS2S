# import xarray as xr
# import numpy as np
# import pandas as pd
# from wass2s.was_linear_models import *
# from wass2s.was_eof import *
# from wass2s.was_machine_learning import *

# class WAS_PCR:
#     """
#     A class for Principal Component Regression (PCR) integrating EOF analysis
#     with flexible regression models.
    
#     Parameters
#     ----------
#     regression_model : object
#         An instance of a WAS regression model class.
#     n_modes : int, optional
#         Number of EOF modes to retain.
#     use_coslat : bool, default=True
#         Apply cosine latitude weighting in EOF analysis.
#     standardize : bool, default=False
#         Standardize the input data.
#     detrend : bool, default=True
#         Detrend the input data.
#     opti_explained_variance : float, optional
#         Target cumulative explained variance to determine number of EOF modes.
#     L2norm : bool, default=False
#         Normalize EOF components and scores to have L2 norm.
#     """

#     def __init__(self, regression_model, n_modes=None, use_coslat=True, standardize=False,
#                  detrend=True, opti_explained_variance=None, L2norm=False):
#         """
#         Initializes the PCR class using a WAS_EOF instance and a flexible regression model.
#         """
#         self.eof_model = WAS_EOF(
#             n_modes=n_modes, 
#             use_coslat=use_coslat, 
#             standardize=standardize, 
#             detrend=detrend,
#             opti_explained_variance=opti_explained_variance, 
#             L2norm=L2norm
#         )
#         self.reg_model = regression_model

#     def _prepare_pcs(self, X_train, X_test):
#         """
#         Helper to fit EOF on train and transform test data.
        
#         Parameters
#         ----------
#         X_train : xarray.DataArray
#             Training predictor data with time dimension 'T'
#         X_test : xarray.DataArray
#             Test predictor data with time dimension 'T'
            
#         Returns
#         -------
#         X_train_pcs : xarray.DataArray
#             Principal components for training data
#         X_test_pcs : xarray.DataArray
#             Principal components for test data
#         """
#         # Fit EOF on training data
#         _, s_pcs, _, = self.eof_model.fit(X_train, dim="T")
#         X_train_pcs = s_pcs.rename({"mode": "features"}).transpose('T', 'features')
        
#         # Transform test data
#         X_test_filled = X_test.fillna(X_train.mean())
        
#         # Ensure T dimension exists for transform
#         if 'T' not in X_test_filled.dims:
#             X_test_filled = X_test_filled.expand_dims('T')
             
#         X_test_pcs = self.eof_model.transform(X_test_filled)
#         X_test_pcs = X_test_pcs.rename({"mode": "features"}).transpose('T', 'features')
        
#         return X_train_pcs, X_test_pcs

#     def compute_model(self, X_train, y_train, X_test, y_test, **kwargs):
#         """
#         Fits EOF, transforms predictors, and runs the deterministic regression.
        
#         Parameters
#         ----------
#         X_train : xarray.DataArray
#             Training predictor data
#         y_train : array-like or xarray.DataArray
#             Training target data
#         X_test : xarray.DataArray
#             Test predictor data
#         **kwargs : dict
#             Additional parameters passed to the regression model's compute_model method
            
#         Returns
#         -------
#         result : object
#             Result from the regression model's compute_model method
#         """
#         # X_train_pcs, X_test_pcs = self._prepare_pcs(X_train, X_test)
#         X_train_pcs, X_test_pcs = X_train, X_test

#         # all_params = {**kwargs}
#         # params_prob = {
#         #     key: value for key, value in all_params.items() 
#         #     if key not in self.reg_model.compute_model.__code__.co_varnames
#         # }

#         # params_models = {
#         #     key: value for key, value in all_params.items() 
#         #     if key not in params_prob
#         # } 
        
#         # Forward all parameters to the underlying compute_model
#         if 'y_test' not in self.reg_model.compute_model.__code__.co_varnames:
#             return self.reg_model.compute_model(X_train_pcs, y_train, X_test_pcs, **kwargs)
#         else:
#             return self.reg_model.compute_model(X_train_pcs, y_train, X_test_pcs, y_test, **kwargs)

#     def compute_prob(self, Predictant, clim_year_start, clim_year_end, hindcast_det, **kwargs):
#         """
#         Computes probabilistic forecasts using the regression model's probability method.
        
#         Parameters
#         ----------
#         Predictant : xarray.DataArray
#             Target variable data
#         clim_year_start : int
#             Start year for climatology
#         clim_year_end : int
#             End year for climatology
#         hindcast_det : xarray.DataArray
#             Hindcast deterministic predictions
#         **kwargs : dict
#             Additional parameters for the regression model
            
#         Returns
#         -------
#         result : object or None
#             Probabilistic forecast result or None for unsupported models
#         """
#         import inspect
        
#         # Check if the regression model has a compute_prob method
#         if not hasattr(self.reg_model, 'compute_prob'):
#             return None
                    
#         all_params = {**kwargs}
#         params_prob = {
#             key: value for key, value in all_params.items() 
#             if key not in self.reg_model.compute_model.__code__.co_varnames
#         }

#         # Handle Logistic Regression (usually returns classes, not tercile probs)
#         if isinstance(self.reg_model, WAS_LogisticRegression_Model):
#             return None
            
#         return self.reg_model.compute_prob(
#             Predictant, clim_year_start, clim_year_end, hindcast_det, **params_prob
#         )

#     def forecast(self, Predictant, clim_year_start, clim_year_end, Predictor, 
#                  hindcast_det, Predictor_for_year, **kwargs):
#         """
#         Generates an operational forecast by projecting the forecast year onto EOF space.
        
#         Parameters
#         ----------
#         Predictant : xarray.DataArray
#             Target variable data
#         clim_year_start : int
#             Start year for climatology
#         clim_year_end : int
#             End year for climatology
#         Predictor : xarray.DataArray
#             Historical predictor data for training
#         hindcast_det : xarray.DataArray
#             Hindcast deterministic predictions
#         Predictor_for_year : xarray.DataArray
#             Predictor data for the forecast year
#         **kwargs : dict
#             Additional parameters for the regression model
            
#         Returns
#         -------
#         result : object
#             Forecast result from the regression model
#         """
#         # # Fit EOF on historical predictors
#         # _, s_pcs, _, _ = self.eof_model.fit(Predictor, dim="T")
#         # Predictor_pcs = s_pcs.rename({"mode": "features"}).transpose('T', 'features')
        
#         # # Transform predictor for forecast year
#         # Predictor_for_year_filled = Predictor_for_year.fillna(Predictor.mean())
#         # Predictor_for_year_pcs = self.eof_model.transform(Predictor_for_year_filled)
#         # Predictor_for_year_pcs = Predictor_for_year_pcs.rename({"mode": "features"}).transpose('T', 'features')

#         Predictor_pcs, Predictor_for_year_pcs = self._prepare_pcs(Predictor, Predictor_for_year)
        
#         # Run the underlying model's forecast method
#         return self.reg_model.forecast(
#             Predictant, clim_year_start, clim_year_end, 
#             Predictor_pcs, hindcast_det, Predictor_for_year_pcs, **kwargs
#         )


import xarray as xr
import numpy as np
from xeofs.single import EOF
from wass2s.was_linear_models import *
from wass2s.utils import *
from wass2s.was_machine_learning import *

class WAS_PCR:
    """
    A class for Principal Component Regression (PCR) integrating EOF analysis
    directly using xeofs, assuming external detrending.
    """

    def __init__(self, regression_model, n_modes=None, use_coslat=True, standardize=False,
                 opti_explained_variance=None, L2norm=False):
        """
        Parameters
        ----------
        regression_model : object
            An instance of a WAS regression model class.
        n_modes : int, optional
            Number of EOF modes to retain.
        use_coslat : bool, default=True
            Apply cosine latitude weighting in EOF analysis.
        standardize : bool, default=False
            Standardize the input data (useful if inputs are not standardized anomalies).
        opti_explained_variance : float, optional
            Target cumulative explained variance (e.g., 90.0) to determine optimal n_modes.
        L2norm : bool, default=False
            Normalize EOF components and scores to have L2 norm.
        """
        self.reg_model = regression_model
        self.n_modes = n_modes
        self.use_coslat = use_coslat
        self.standardize = standardize
        self.opti_explained_variance = opti_explained_variance
        self.L2norm = L2norm
        
        # Internal storage for the fitted EOF model
        self.eof_model = None

    def _prepare_pcs(self, X_train, X_test):
        """
        Internal helper: Fits EOF on X_train and projects X_test.
        """
        # 1. Handle Dimensions (Rename to T if needed, as per your convention)
        if "time" in X_train.dims and "T" not in X_train.dims:
            X_train = X_train.rename({"time": "T"})
        if "time" in X_test.dims and "T" not in X_test.dims:
            X_test = X_test.rename({"time": "T"})

        # 2. Initial EOF Fit
        # Start with a high number of modes or the user requested number
        initial_modes = self.n_modes if self.n_modes else 50
        
        model = EOF(n_modes=initial_modes, use_coslat=self.use_coslat, standardize=self.standardize)
        model.fit(X_train, dim="T")

        # 3. Variance Optimization (if requested)
        if self.opti_explained_variance is not None:
            exp_var_cum = model.explained_variance_ratio().cumsum() * 100
            # Find index where variance threshold is met
            n_needed = int(np.searchsorted(exp_var_cum.values, self.opti_explained_variance) + 1)
            
            # Refit with optimal modes if different
            if n_needed != initial_modes:
                model = EOF(n_modes=n_needed, use_coslat=self.use_coslat, standardize=self.standardize)
                model.fit(X_train, dim="T")
        
        self.eof_model = model

        # 4. Extract Training PCs (Scores)
        # xeofs returns dim 'mode', WAS_PCR expects 'features'
        s_pcs = model.scores(normalized=self.L2norm)
        X_train_pcs = s_pcs.rename({"mode": "features"})

        # 5. Transform Test Data
        # Ensure test data handles NaNs implicitly via xeofs
        # If X_test is a single time step without 'T' dim, expand it for xeofs
        if "T" not in X_test.dims:
             X_test_to_transform = X_test.expand_dims("T")
        else:
             X_test_to_transform = X_test

        X_test_pcs = model.transform(X_test_to_transform, normalized=self.L2norm)
        X_test_pcs = X_test_pcs.rename({"mode": "features"})

        return X_train_pcs, X_test_pcs

    def compute_model(self, X_train, y_train, X_test, y_test=None, **kwargs):
        """
        Main pipeline: EOF Analysis -> PC Extraction -> Regression.
        """
        # Calculate PCs
        X_train_pcs, X_test_pcs = self._prepare_pcs(X_train, X_test)
        
        # Forward to regression model
        if y_test is not None and 'y_test' in self.reg_model.compute_model.__code__.co_varnames:
            return self.reg_model.compute_model(X_train_pcs, y_train, X_test_pcs, y_test, **kwargs)
        else:
            return self.reg_model.compute_model(X_train_pcs, y_train, X_test_pcs, **kwargs)

    def compute_prob(self, Predictant, clim_year_start, clim_year_end, hindcast_det, **kwargs):
        """
        Computes probabilistic forecasts (Terciles).
        """
        if not hasattr(self.reg_model, 'compute_prob'):
            return None
        
        # Filter params specifically for compute_prob if needed
        all_params = {**kwargs}
        params_prob = {
            k: v for k, v in all_params.items() 
            if k not in self.reg_model.compute_model.__code__.co_varnames
        }

        # Exclude specific models if necessary
        if isinstance(self.reg_model, WAS_LogisticRegression_Model):
            return None
            
        return self.reg_model.compute_prob(
            Predictant, clim_year_start, clim_year_end, hindcast_det, **params_prob
        )

    ### Revenir sur ce cas leakage et gestion detrend coté predictand
    ### Cela sera fait aussi bien pour du PCR mais les MLR simples avec indices
    def forecast(self, Predictant, clim_year_start, clim_year_end, Predictor, 
                 hindcast_det, Predictor_for_year, **kwargs):
        """
        Operational Forecast.
        """

        Predictor_detrend, coeffs, meta = detrended_data(Predictor, dim="T") 
        Predictor_for_year_detrend = Predictor_for_year - apply_detrend_data(Predictor_for_year, coeffs, meta)
        
        
        # Prepare PCs for both history (training) and the target year
        Predictor_pcs, Predictor_for_year_pcs = self._prepare_pcs(Predictor_detrend, Predictor_for_year_detrend)
        
        return self.reg_model.forecast(
            Predictant, clim_year_start, clim_year_end, 
            Predictor_pcs, hindcast_det, Predictor_for_year_pcs, **kwargs
        )