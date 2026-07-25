import xarray as xr
import numpy as np
from xeofs.single import EOF
from wass2s.was_linear_models import *
from wass2s.utils import *
from wass2s.was_machine_learning import *
from wass2s.was_eof import *

# class WAS_PCR:
#     """
#     A class for Principal Component Regression (PCR) integrating EOF analysis
#     directly using xeofs, assuming external detrending.
#     """

#     def __init__(self, regression_model, n_modes=None, use_coslat=True, standardize=False,
#                  opti_explained_variance=None, L2norm=False):
#         """
#         Parameters
#         ----------
#         regression_model : object
#             An instance of a WAS regression model class.
#         n_modes : int, optional
#             Number of EOF modes to retain.
#         use_coslat : bool, default=True
#             Apply cosine latitude weighting in EOF analysis.
#         standardize : bool, default=False
#             Standardize the input data (useful if inputs are not standardized anomalies).
#         opti_explained_variance : float, optional
#             Target cumulative explained variance (e.g., 90.0) to determine optimal n_modes.
#         L2norm : bool, default=False
#             Normalize EOF components and scores to have L2 norm.
#         """
#         self.reg_model = regression_model
#         self.n_modes = n_modes
#         self.use_coslat = use_coslat
#         self.standardize = standardize
#         self.opti_explained_variance = opti_explained_variance
#         self.L2norm = L2norm
        
#         # Internal storage for the fitted EOF model
#         self.eof_model = None

#     def _prepare_pcs(self, X_train, X_test):
#         """
#         Internal helper: Fits EOF on X_train and projects X_test.
#         """
#         # 1. Handle Dimensions (Rename to T if needed, as per your convention)
#         if "time" in X_train.dims and "T" not in X_train.dims:
#             X_train = X_train.rename({"time": "T"})
#         if "time" in X_test.dims and "T" not in X_test.dims:
#             X_test = X_test.rename({"time": "T"})

#         # 2. Initial EOF Fit
#         # Start with a high number of modes or the user requested number
#         initial_modes = self.n_modes if self.n_modes else 50
        
#         model = EOF(n_modes=initial_modes, use_coslat=self.use_coslat, standardize=self.standardize)
#         model.fit(X_train, dim="T")

#         # 3. Variance Optimization (if requested)
#         if self.opti_explained_variance is not None:
#             exp_var_cum = model.explained_variance_ratio().cumsum() * 100
#             # Find index where variance threshold is met
#             n_needed = int(np.searchsorted(exp_var_cum.values, self.opti_explained_variance) + 1)
            
#             # Refit with optimal modes if different
#             if n_needed != initial_modes:
#                 model = EOF(n_modes=n_needed, use_coslat=self.use_coslat, standardize=self.standardize)
#                 model.fit(X_train, dim="T")
        
#         self.eof_model = model

#         # 4. Extract Training PCs (Scores)
#         # xeofs returns dim 'mode', WAS_PCR expects 'features'
#         s_pcs = model.scores(normalized=self.L2norm)
#         X_train_pcs = s_pcs.rename({"mode": "features"})

#         # 5. Transform Test Data
#         # Ensure test data handles NaNs implicitly via xeofs
#         # If X_test is a single time step without 'T' dim, expand it for xeofs
#         if "T" not in X_test.dims:
#              X_test_to_transform = X_test.expand_dims("T")
#         else:
#              X_test_to_transform = X_test

#         X_test_pcs = model.transform(X_test_to_transform, normalized=self.L2norm)
#         X_test_pcs = X_test_pcs.rename({"mode": "features"})

#         return X_train_pcs, X_test_pcs

#     def compute_model(self, X_train, y_train, X_test, y_test=None, **kwargs):
#         """
#         Main pipeline: EOF Analysis -> PC Extraction -> Regression.
#         """
#         # Calculate PCs
#         X_train_pcs, X_test_pcs = self._prepare_pcs(X_train, X_test)
        
#         # Forward to regression model
#         if y_test is not None and 'y_test' in self.reg_model.compute_model.__code__.co_varnames:
#             return self.reg_model.compute_model(X_train_pcs, y_train, X_test_pcs, y_test, **kwargs)
#         else:
#             return self.reg_model.compute_model(X_train_pcs, y_train, X_test_pcs, **kwargs)

#     def compute_prob(self, Predictant, clim_year_start, clim_year_end, hindcast_det, **kwargs):
#         """
#         Computes probabilistic forecasts (Terciles).
#         """
#         if not hasattr(self.reg_model, 'compute_prob'):
#             return None
        
#         # Filter params specifically for compute_prob if needed
#         all_params = {**kwargs}
#         params_prob = {
#             k: v for k, v in all_params.items() 
#             if k not in self.reg_model.compute_model.__code__.co_varnames
#         }

#         # Exclude specific models if necessary
#         if isinstance(self.reg_model, WAS_LogisticRegression_Model):
#             return None
            
#         return self.reg_model.compute_prob(
#             Predictant, clim_year_start, clim_year_end, hindcast_det, **params_prob
#         )

#     ### Revenir sur ce cas leakage et gestion detrend coté predictand
#     ### Cela sera fait aussi bien pour du PCR mais les MLR simples avec indices
#     def forecast(self, Predictant, clim_year_start, clim_year_end, Predictor, 
#                  hindcast_det, Predictor_for_year, **kwargs):
#         """
#         Operational Forecast.
#         """

#         Predictor_detrend, coeffs, meta = detrended_data(Predictor, dim="T") 
#         Predictor_for_year_detrend = Predictor_for_year - apply_detrend_data(Predictor_for_year, coeffs, meta)
        
        
#         # Prepare PCs for both history (training) and the target year
#         Predictor_pcs, Predictor_for_year_pcs = self._prepare_pcs(Predictor_detrend, Predictor_for_year_detrend)
        
#         return self.reg_model.forecast(
#             Predictant, clim_year_start, clim_year_end, 
#             Predictor_pcs, hindcast_det, Predictor_for_year_pcs, **kwargs
#         )


class WAS_PCR:
    """
    Spatial principal-component regression wrapping a WAS regression model.

    Single field OR multivariate predictors
    ---------------------------------------
    The predictor may be a single DataArray, or a LIST of DataArrays
    [da1, da2, ...] of DIFFERENT variables and/or grids/scales (e.g. SST + SLP
    + rainfall). A list triggers a combined / multivariate EOF: each field is
    preprocessed and normalized INDEPENDENTLY on the training fold (fill, trend,
    and a per-field scalar rescaling so no field dominates the joint
    covariance), then xeofs builds one shared set of principal components that
    feed the regression. Per-field normalization is essential here and is on by
    default for a list (controllable via `normalize`).

    Corrections (vs the previous version)
    -------------------------------------
    * Detrend / NaN-fill happen INSIDE each fold via the fold-safe WAS_EOF (fit
      on train, applied to test), instead of a full-series detrend before the CV
      loop (which leaked the test-year trend).
    * Per-field fold-safe normalization for the multivariate case.

    -> In WAS_Cross_Validator's WAS_PCR branch, pass the RAW predictor and REMOVE
       the pre-loop `detrended_data(Predictor)` call. If the predictor is a list,
       slice every field per fold, e.g.
           X_train = [p.isel(T=train_index) for p in Predictor]
           X_test  = [p.isel(T=test_index)  for p in Predictor]
    """

    def __init__(self, regression_model, n_modes=None, use_coslat=True, standardize=False,
                 detrend=True, opti_explained_variance=None, L2norm=False, normalize=None):
        self.reg_model = regression_model
        self.n_modes = n_modes
        self.use_coslat = use_coslat
        self.standardize = standardize
        self.detrend = detrend
        self.opti_explained_variance = opti_explained_variance
        self.L2norm = L2norm
        self.normalize = normalize
        self.eof_model = None

    @staticmethod
    def _ensure_T(x):
        """Ensure each field carries a T dim (handles single DataArray or list)."""
        if isinstance(x, (list, tuple)):
            return [d if "T" in d.dims else d.expand_dims("T") for d in x]
        return x if "T" in x.dims else x.expand_dims("T")

    def _prepare_pcs(self, X_train, X_test):
        eof = WAS_EOF(n_modes=self.n_modes, use_coslat=self.use_coslat,
                      standardize=self.standardize, detrend=self.detrend,
                      opti_explained_variance=self.opti_explained_variance,
                      L2norm=self.L2norm, normalize=self.normalize)
        _, s_train, _ = eof.fit(X_train, dim="T")
        self.eof_model = eof

        X_train_pcs = s_train.rename({"mode": "features"}).transpose("T", "features")
        s_test = eof.transform(self._ensure_T(X_test), dim="T")
        X_test_pcs = s_test.rename({"mode": "features"}).transpose("T", "features")
        return X_train_pcs, X_test_pcs

    def compute_model(self, X_train, y_train, X_test, y_test=None, **kwargs):
        X_train_pcs, X_test_pcs = self._prepare_pcs(X_train, X_test)
        if y_test is not None and "y_test" in self.reg_model.compute_model.__code__.co_varnames:
            return self.reg_model.compute_model(X_train_pcs, y_train, X_test_pcs, y_test, **kwargs)
        return self.reg_model.compute_model(X_train_pcs, y_train, X_test_pcs, **kwargs)

    def compute_prob(self, Predictant, clim_year_start, clim_year_end, hindcast_det, **kwargs):
        if not hasattr(self.reg_model, "compute_prob"):
            return None
        params_prob = {k: v for k, v in kwargs.items()
                       if k not in self.reg_model.compute_model.__code__.co_varnames}
        return self.reg_model.compute_prob(Predictant, clim_year_start, clim_year_end, hindcast_det, **params_prob)

    def forecast(self, Predictant, clim_year_start, clim_year_end, Predictor,
                 hindcast_det, Predictor_for_year, **kwargs):
        Predictor_pcs, Predictor_year_pcs = self._prepare_pcs(Predictor, Predictor_for_year)
        return self.reg_model.forecast(
            Predictant, clim_year_start, clim_year_end,
            Predictor_pcs, hindcast_det, Predictor_year_pcs, **kwargs)