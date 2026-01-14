################################### Modules ###################################
from sklearn import linear_model
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import GridSearchCV
from sklearn.model_selection import RandomizedSearchCV, cross_val_score
import statsmodels.api as sm
from statsmodels.stats.outliers_influence import variance_inflation_factor as VIF
from statsmodels.stats.anova import anova_lm
import xarray as xr
import numpy as np
import pandas as pd
from scipy import stats
from scipy.stats import norm, lognorm, gamma, expon, weibull_min, t, poisson, nbinom, loguniform
import scipy.signal as sig
from scipy.interpolate import CubicSpline
from scipy.optimize import fsolve
from scipy.special import gamma as gamma_function
from sklearn.cluster import KMeans
import xeofs as xe
from multiprocessing import cpu_count
from dask.distributed import Client
import dask.array as da
import optuna
import warnings
from wass2s.utils import *

#### Add Nonexcedance function for all models ##############################################

class WAS_LinearRegression_Model:
    """
    Simple Linear Regression model for spatiotemporal climate prediction.

    This class implements ordinary least squares linear regression (via scikit-learn's
    `LinearRegression`) to model linear relationships between predictors and a continuous
    predictand (e.g., seasonal rainfall totals, temperature anomalies, agro-climatic indices).

    Key features:
    - Basic linear modeling (no polynomial features or regularization by default)
    - Spatially parallel fitting and prediction across large grids using dask + xarray
    - Deterministic point predictions with optional error computation
    - Probabilistic tercile forecasting (Below/Normal/Above = PB/PN/PA) using either:
      - Parametric best-fit distributions per grid cell ('bestfit')
      - Non-parametric sampling of historical forecast errors ('nonparam')

    Ideal as a baseline model or when relationships are expected to be approximately linear.

    Parameters
    -----------
    nb_cores : int, default=1
        Number of CPU cores for parallel processing (dask workers).

    dist_method : {'bestfit', 'nonparam'}, default='nonparam'
        Method for computing tercile probabilities:
        - 'bestfit'  → uses best-fit distribution per grid cell (requires distribution fit inputs)
        - 'nonparam' → empirical sampling of historical forecast errors

    Attributes
    -----------
    nb_cores, dist_method
        Stored initialization parameters.

    Methods
    --------
    fit_predict(x, y, x_test, y_test=None)
        Fits linear regression on one grid cell and returns [error, prediction] or just prediction.

    compute_model(X_train, y_train, X_test, y_test)
        Parallel linear regression across the entire spatial domain.
        Returns xarray.DataArray with dims ('output'=['error','prediction'], Y, X).

    compute_prob(Predictant, clim_year_start, clim_year_end, hindcast_det,
                 best_code_da=None, best_shape_da=None, best_loc_da=None, best_scale_da=None)
        Computes tercile probabilities for deterministic hindcasts.

    forecast(Predictant, clim_year_start, clim_year_end, Predictor, hindcast_det,
             Predictor_for_year, best_code_da=None, best_shape_da=None,
             best_loc_da=None, best_scale_da=None)
        Full end-to-end forecast pipeline for one target year:
        - deterministic linear prediction
        - tercile probabilities (PB, PN, PA)

    Notes
    ------
    - **Input data requirements**:
      - Target (y) should be continuous (rainfall, temperature, etc.).
      - Predictors (X) should be continuous or properly encoded.
    - Negative predictions are automatically clipped to zero (useful for rainfall or non-negative variables).
    - Linear regression assumes linear relationships; for non-linear patterns, consider higher-degree
      polynomial, tree-based models (XGBoost, RF), or neural approaches (MLP).
    - Large spatial domains benefit significantly from higher `nb_cores`.
    - For `dist_method='bestfit'`, distribution fit results from `WAS_TransformData` must be provided.
    - No explicit feature or target scaling is applied (linear regression coefficients adjust automatically,
      but centering/scaling predictors can improve numerical stability when features have very different scales).


    Warnings
    ---------
    - Linear regression can produce unrealistic negative predictions for non-negative variables
      (e.g., rainfall); clipping is applied automatically but may bias results.
    - Assumes homoscedasticity and no strong multicollinearity; check residuals and VIF if needed.
    - Small per-grid-cell training sets can lead to unstable coefficient estimates.
    - For very non-linear relationships, performance will be poor compared to tree-based or spline models.
    """

    def __init__(self, nb_cores=1, dist_method="nonparam"):
        """
        Initializes the WAS_LinearRegression_Model with a specified number of CPU cores.
        
        Parameters
        -----------
        nb_cores : int, optional
            Number of CPU cores to use for parallel computation, by default 1.
        dist_method : str, optional
            Distribution method to compute tercile probabilities, by default "gamma".
        """
        self.nb_cores = nb_cores
        self.dist_method = dist_method
    
    def fit_predict(self, x, y, x_test, y_test=None):
        """
        Fits a linear regression model to the provided training data, makes predictions 
        on the test data, and calculates the prediction error (if y_test is provided).
        
        Parameters
        -----------
        x : array-like, shape (n_samples, n_features)
            Training data (predictors).
        y : array-like, shape (n_samples,)
            Training targets.
        x_test : array-like, shape (n_features,) or (1, n_features)
            Test data (predictors).
        y_test : float or None
            Test target value. If None, no error is computed.

        Returns
        --------
        np.ndarray
            If y_test is not None, returns [error, prediction].
            If y_test is None, returns [prediction].
        """
        model = linear_model.LinearRegression()
        mask = np.isfinite(y) & np.all(np.isfinite(x), axis=-1)

        if np.any(mask):
            y_clean = y[mask]
            x_clean = x[mask, :]
            model.fit(x_clean, y_clean)

            if x_test.ndim == 1:
                x_test = x_test.reshape(1, -1)

            preds = model.predict(x_test)
            preds[preds < 0] = 0  # clip negative if modeling precipitation

            if y_test is not None:
                error_ = y_test - preds
                return np.array([error_, preds]).squeeze()
            else:
                # Only return prediction if y_test is None
                return np.array([preds]).squeeze()
        else:
            # If no valid data, return NaNs
            if y_test is not None:
                return np.array([np.nan, np.nan]).squeeze()
            else:
                return np.array([np.nan]).squeeze()

    def compute_model(self, X_train, y_train, X_test, y_test):
        """
        Applies linear regression across a spatiotemporal dataset in parallel.

        Parameters
        ----------
        X_train : xarray.DataArray
            Training predictors with dims ('T','features').
        y_train : xarray.DataArray
            Training targets with dims ('T','Y','X').
        X_test : xarray.DataArray
            Test predictors, shape ('features',) or (T, features).
        y_test : xarray.DataArray
            Test targets with dims ('Y','X'), or broadcastable.

        Returns
        -------
        xarray.DataArray
            dims ('output','Y','X'), where 'output'=[error, prediction].
        """
        chunksize_x = int(np.round(len(y_train.get_index("X")) / self.nb_cores))
        chunksize_y = int(np.round(len(y_train.get_index("Y")) / self.nb_cores))
        
        # Align times
        X_train['T'] = y_train['T']
        y_train = y_train.transpose('T', 'Y', 'X')
        X_test = X_test.squeeze()
        y_test = y_test.drop_vars('T').squeeze().transpose('Y', 'X')

        client = Client(n_workers=self.nb_cores, threads_per_worker=1)
        result_da = xr.apply_ufunc(
            self.fit_predict,
            X_train,
            y_train.chunk({'Y': chunksize_y, 'X': chunksize_x}),
            X_test,
            y_test.chunk({'Y': chunksize_y, 'X': chunksize_x}),
            input_core_dims=[('T','features'), ('T',), ('features',), ()],
            vectorize=True,
            dask='parallelized',
            output_core_dims=[('output',)],
            output_dtypes=['float'],
            dask_gufunc_kwargs={'output_sizes': {'output': 2}},
        )
        result_ = result_da.compute()
        client.close()
        return result_.isel(output=1)
    
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

    # --------------------------------------------------------------------------
    #  FORECAST METHOD
    # --------------------------------------------------------------------------
    def forecast(self, Predictant, clim_year_start, clim_year_end, Predictor, hindcast_det, Predictor_for_year, best_code_da=None, best_shape_da=None, best_loc_da=None, best_scale_da=None):
        """
        Generates a single-year forecast using linear regression, then computes 
        tercile probabilities using self.dist_method.

        Parameters
        ----------
        Predictant : xarray.DataArray
            Observed data with dims (T, Y, X).
        clim_year_start : int
            Start year for climatology
        clim_year_end : int
            End year for climatology
        Predictor : xarray.DataArray
            Historical predictor data with dims (T, features).
        hindcast_det : xarray.DataArray
            Historical deterministic forecast with dims (output=[error,prediction], T, Y, X).
        Predictor_for_year : xarray.DataArray
            Single-year predictor with shape (features,) or (1, features).

        Returns
        -------
        result_ : xarray.DataArray
            dims (output=2, Y, X) => [error, prediction]. 
            For a true forecast, error is typically NaN.
        hindcast_prob : xarray.DataArray
            dims (probability=3, Y, X) => [PB, PN, PA].
        """
        # Provide a dummy y_test with the same shape as the spatial domain => [NaNs]
        y_test_dummy = xr.full_like(Predictant.isel(T=0), np.nan)

        # Chunk sizes
        chunksize_x = int(np.round(len(Predictant.get_index("X")) / self.nb_cores))
        chunksize_y = int(np.round(len(Predictant.get_index("Y")) / self.nb_cores))

        # Align time dimension
        Predictor['T'] = Predictant['T']
        Predictant_st = standardize_timeseries(Predictant, clim_year_start, clim_year_end)
        Predictant_st = Predictant_st.transpose('T','Y','X')
        Predictor_for_year_ = Predictor_for_year.squeeze()

        # 1) Fit+predict in parallel => shape (output=2, Y, X)
        client = Client(n_workers=self.nb_cores, threads_per_worker=1)
        result_da = xr.apply_ufunc(
            self.fit_predict,
            Predictor,
            Predictant_st.chunk({'Y': chunksize_y, 'X': chunksize_x}),
            Predictor_for_year_,
            y_test_dummy.chunk({'Y': chunksize_y, 'X': chunksize_x}),  # dummy y_test
            input_core_dims=[
                ('T','features'),  # x
                ('T',),           # y
                ('features',),    # x_test
                ()
            ],
            vectorize=True,
            dask='parallelized',
            output_core_dims=[('output',)],  # output=2 => [error, prediction]
            output_dtypes=['float'],
            dask_gufunc_kwargs={'output_sizes': {'output':2}},
        )
        result_ = result_da.compute()
        client.close()
        result_ = result_.isel(output=1)
        result_ = reverse_standardize(result_, Predictant, clim_year_start, clim_year_end)
        # 2) Compute thresholds T1, T2 from climatology
        index_start = Predictant.get_index("T").get_loc(str(clim_year_start)).start
        index_end   = Predictant.get_index("T").get_loc(str(clim_year_end)).stop
        rainfall_for_tercile = Predictant.isel(T=slice(index_start, index_end))
        terciles = rainfall_for_tercile.quantile([0.33, 0.67], dim='T')
        T1_emp = terciles.isel(quantile=0).drop_vars('quantile')
        T2_emp = terciles.isel(quantile=1).drop_vars('quantile')
        error_variance = (Predictant - hindcast_det).var(dim='T')
        
        # Expand single prediction to T=1 so probability methods can handle it
        forecast_expanded = result_.expand_dims(
            T=[pd.Timestamp(Predictor_for_year.coords['T'].values[0]).to_pydatetime()]
        )
        year = Predictor_for_year.coords['T'].values[0].astype('datetime64[Y]').astype(int) + 1970
        # year = Predictor_for_year.coords['T'].values.astype('datetime64[Y]').astype(int)[0] + 1970  
        T_value_1 = Predictant.isel(T=0).coords['T'].values  # Get the datetime64 value from da1
        month_1 = T_value_1.astype('datetime64[M]').astype(int) % 12 + 1  # Extract month
        new_T_value = np.datetime64(f"{year}-{month_1:02d}-{1:02d}")
        
        forecast_expanded = forecast_expanded.assign_coords(T=xr.DataArray([new_T_value], dims=["T"]))
        forecast_expanded['T'] = forecast_expanded['T'].astype('datetime64[ns]')

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
                forecast_expanded,
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
                forecast_expanded,
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
        return forecast_expanded, forecast_prob.transpose('probability', 'T', 'Y', 'X')


class WAS_Ridge_Model:
    """
    Ridge regression model for spatial rainfall (or continuous climate variable) prediction.

    Supports two optimization modes:
    - **'grid'**: Independent alpha optimization for each grid cell (pixel-wise)
      → Most local fit, but computationally expensive
    - **'cluster'** (default): Spatially cluster the predictand field → one alpha per cluster
      → Much faster, more robust to noise, better generalization on small/sparse data

    Hyperparameter search methods:
    - 'bayesian' → Optuna Bayesian optimization (recommended)
    - 'random'   → scikit-learn RandomizedSearchCV
    - 'ridgecv'  → classic RidgeCV grid search

    The model uses scikit-learn's Ridge regression with L2 regularization to prevent overfitting,
    especially useful when predictors are correlated or when per-cell training data is limited.

    Parameters
    ----------
    alpha_range : array-like, optional
        Range of alpha (regularization strength) values to search.
        Default: np.logspace(-10, 10, 100) — wide log-scale range.

    n_clusters : int, default=5
        Number of spatial clusters (used only when mode='cluster').

    nb_cores : int, default=1
        Number of CPU cores for parallel computation (mainly used in grid mode).

    dist_method : str, default='nonparam'
        Method for computing tercile probabilities (PB/PN/PA).
        Currently only 'nonparam' fully implemented; 'bestfit' requires additional inputs.

    hyperparam_optimizer : {'bayesian', 'random', 'ridgecv'}, default='bayesian'
        Strategy for finding optimal alpha:
        - 'bayesian' → Optuna Bayesian optimization (most efficient)
        - 'random'   → RandomizedSearchCV
        - 'ridgecv'  → Built-in RidgeCV grid search

    n_trials : int, default=50
        Number of trials for Bayesian optimization (Optuna).

    n_iter : int, default=50
        Number of parameter settings sampled in RandomizedSearchCV.

    mode : {'cluster', 'grid'}, default='cluster'
        Optimization strategy:
        - 'cluster' → one alpha per spatial cluster (recommended for most cases)
        - 'grid'    → independent alpha per grid cell (very slow, but maximally local)

    Notes
    -----
    - Cluster mode is significantly faster and usually more stable, especially for noisy or
      short time series data common in climate modeling.
    - All NaN values in time series are properly masked.
    - Requires at least 10 valid time steps per location/cluster to attempt optimization.
    - Negative predictions are clipped to zero (useful for rainfall or non-negative variables).
    - Large domains in 'grid' mode benefit greatly from higher `nb_cores`.
    - For `dist_method='bestfit'`, additional distribution fit parameters must be provided.

    Methods
    -------
    compute_hyperparameters(predictand, predictor, clim_year_start, clim_year_end)
        Computes spatially varying optimal alpha values (and cluster map if mode='cluster').

    fit_predict(x, y, x_test, y_test, alpha)
        Fits Ridge model on one grid cell using given alpha and makes prediction.

    compute_model(X_train, y_train, X_test, y_test, alpha)
        Parallel ridge regression across entire spatial domain using provided alpha map.

    compute_prob(Predictant, clim_year_start, clim_year_end, hindcast_det,
                 best_code_da=None, best_shape_da=None, best_loc_da=None, best_scale_da=None)
        Computes tercile probabilities for hindcast predictions.

    forecast(Predictant, clim_year_start, clim_year_end, Predictor, hindcast_det,
             Predictor_for_year, alpha, best_code_da=None, best_shape_da=None,
             best_loc_da=None, best_scale_da=None)
        Full end-to-end forecast pipeline for one target year:
        - deterministic ridge prediction
        - tercile probabilities (PB, PN, PA)

    Examples
    --------
    Typical seasonal rainfall forecasting workflow (recommended cluster mode):

    >>> ridge_model = WAS_Ridge_Model(
    ...     mode='cluster',
    ...     n_clusters=8,
    ...     hyperparam_optimizer='bayesian',
    ...     n_trials=80,
    ...     nb_cores=12
    ... )

    # 1. Compute spatially varying alpha (one per cluster)
    >>> alpha_map, cluster_map = ridge_model.compute_hyperparameters(
    ...     seasonal_rainfall, predictors, 1991, 2020)

    # 2. Train & predict on hindcast period
    >>> hindcast_pred = ridge_model.compute_model(
    ...     X_train=predictors,
    ...     y_train=seasonal_rainfall,
    ...     X_test=predictors_hindcast,
    ...     y_test=None,
    ...     alpha=alpha_map
    ... )

    # 3. Compute probabilistic hindcast (terciles)
    >>> hindcast_prob = ridge_model.compute_prob(
    ...     seasonal_rainfall, 1991, 2020, hindcast_pred,
    ...     best_code_da=dist_code_da, best_shape_da=shape_da,
    ...     best_loc_da=loc_da, best_scale_da=scale_da
    ... )

    # 4. Forecast next year (e.g. 2025)
    >>> forecast_det, forecast_prob = ridge_model.forecast(
    ...     seasonal_rainfall, 1991, 2020,
    ...     predictors, hindcast_pred,
    ...     predictor_2025,
    ...     alpha=alpha_map,
    ...     best_code_da=dist_code_da, best_shape_da=shape_da,
    ...     best_loc_da=loc_da, best_scale_da=scale_da
    ... )
    """

    def __init__(self, alpha_range=None, n_clusters=5, nb_cores=1, dist_method="nonparam",
                 hyperparam_optimizer="bayesian", n_trials=50, n_iter=50, mode="cluster"):
        if alpha_range is None:
            alpha_range = np.logspace(-10, 10, 100)
        
        self.alpha_range = alpha_range
        self.n_clusters = n_clusters
        self.nb_cores = nb_cores
        self.dist_method = dist_method
        self.hyperparam_optimizer = hyperparam_optimizer
        self.n_trials = n_trials
        self.n_iter = n_iter
        self.mode = mode  # 'cluster' or 'grid'

    # ------------------ Hyperparameter Helpers ------------------

    def _optimize_hyperparameters_optuna(self, X, y, alpha_range):
        def objective(trial):
            alpha = trial.suggest_float('alpha', np.min(alpha_range), np.max(alpha_range), log=True)
            model = linear_model.Ridge(alpha=alpha)
            scores = cross_val_score(model, X, y, cv=3, scoring='neg_mean_squared_error')
            return scores.mean()
        
        study = optuna.create_study(direction='maximize')
        study.optimize(objective, n_trials=self.n_trials, show_progress_bar=False)
        return study.best_params['alpha']

    def _optimize_hyperparameters_randomized(self, X, y, alpha_range):
        param_distributions = {'alpha': loguniform(np.min(alpha_range), np.max(alpha_range))}
        model = linear_model.Ridge()
        random_search = RandomizedSearchCV(model, param_distributions, n_iter=self.n_iter,
                                           cv=3, scoring='neg_mean_squared_error', random_state=42)
        random_search.fit(X, y)
        return random_search.best_params_['alpha']

    def _optimize_single_cell(self, y_vec, X_mat):
        """Unified helper to optimize a single pixel or cluster mean."""
        mask = np.isfinite(y_vec) & np.all(np.isfinite(X_mat), axis=-1)
        if np.sum(mask) < 10:  # Require at least 10 valid time steps
            return np.nan
        
        X_clean, y_clean = X_mat[mask], y_vec[mask]
        
        try:
            if self.hyperparam_optimizer == 'bayesian':
                return self._optimize_hyperparameters_optuna(X_clean, y_clean, self.alpha_range)
            elif self.hyperparam_optimizer == 'random':
                return self._optimize_hyperparameters_randomized(X_clean, y_clean, self.alpha_range)
            else:
                model_cv = linear_model.RidgeCV(alphas=self.alpha_range, cv=3)
                model_cv.fit(X_clean, y_clean)
                return model_cv.alpha_
        except:
            return np.nan

    # ------------------ Core Logic Methods ------------------

    def compute_hyperparameters(self, predictand, predictor, clim_year_start, clim_year_end):
        """
        Compute spatially varying Ridge alpha parameter(s).

        Parameters
        ----------
        predictand : xarray.DataArray
            Target variable (usually rainfall)
            Must have dimensions: ('T', 'Y', 'X')
        predictor : xarray.DataArray
            Predictor array with dimensions: ('T', 'Y', 'X', 'features') or ('T', 'features')
        clim_year_start : int
            Start year of climatological period for standardization
        clim_year_end : int
            End year of climatological period for standardization

        Returns
        -------
        alpha_map : xarray.DataArray
            Spatial field of optimal alpha values
            - shape (Y, X) when mode='grid'
            - shape (Y, X) with constant values per cluster when mode='cluster'
        cluster_da : xarray.DataArray or None
            Cluster labels (integers) when mode='cluster'
            Contains NaN where clustering was not possible
            Returns dummy array filled with 1s when mode='grid'

        Raises
        ------
        ValueError
            If mode is neither 'cluster' nor 'grid'
        RuntimeError
            If dask client could not be created or computation failed (grid mode)

        Notes
        -----
        - All NaN values in time series are handled (masked)
        - Minimum 10 valid time steps required per location/cluster
        - Standardization is applied to predictand using specified climatology period
        """
        predictor['T'] = predictand['T']
        predictand_st = standardize_timeseries(predictand, clim_year_start, clim_year_end)

        if self.mode == "grid":
            # Grid-wise: Parallelized over every pixel
            chunk_y = int(np.ceil(len(predictand_st.Y) / self.nb_cores))
            chunk_x = int(np.ceil(len(predictand_st.X) / self.nb_cores))
            p_st_chunked = predictand_st.chunk({'Y': chunk_y, 'X': chunk_x})

            client = Client(n_workers=self.nb_cores, threads_per_worker=1)
            alpha_array = xr.apply_ufunc(
                self._optimize_single_cell,
                p_st_chunked,
                predictor,
                input_core_dims=[('T',), ('T', 'features')],
                output_core_dims=[()],
                vectorize=True,
                dask='parallelized',
                output_dtypes=[float]
            )
            alpha_array = alpha_array.compute()
            client.close()
            cluster_da = xr.where(~np.isnan(alpha_array), 1, np.nan)
            return alpha_array, cluster_da

        else:
            print(f"Ridge: Running Cluster-wise optimization ({self.n_clusters} clusters)...")
            kmeans = KMeans(n_clusters=self.n_clusters)
            predictand_dropna = predictand.to_dataframe().reset_index().dropna().drop(columns=['T'])
            variable_column = predictand_dropna.columns[2]
            predictand_dropna['cluster'] = kmeans.fit_predict(
                predictand_dropna[[variable_column]]
            )
            
            # Convert cluster assignments back into an xarray structure
            df_unique = predictand_dropna.drop_duplicates(subset=['Y', 'X'])
            dataset = df_unique.set_index(['Y', 'X']).to_xarray()
            mask = xr.where(~np.isnan(predictand.isel(T=0)), 1, np.nan)
            cluster_da = (dataset['cluster'] * mask)
                   
            # Align cluster array with the predictand array
            x1, x2 = xr.align(predictand_st, cluster_da, join="outer")
            
            # Identify unique cluster labels
            clusters = np.unique(x2)
            clusters = clusters[~np.isnan(clusters)]
            cluster_da = x2
            
            alpha_map = xr.full_like(cluster_da, np.nan, dtype=float)
            
            for clus in clusters:
                y_cluster = x1.where(x2 == clus).mean(dim=['Y','X'], skipna=True).dropna(dim='T')
                if len(y_cluster['T']) > 0:
                    best_alpha = self._optimize_single_cell(y_cluster.values, predictor.sel(T=y_cluster['T']).values)
                    alpha_map = alpha_map.where(cluster_da != clus, best_alpha)
            
            return alpha_map, cluster_da

    def fit_predict(self, x, y, x_test, y_test, alpha):
        """
        Fit a ridge regression model and make predictions.

        Parameters
        ----------
        x : ndarray
            Training data (shape = [n_samples, n_features]).
        y : ndarray
            Target values for training data (shape = [n_samples,]).
        x_test : ndarray
            Test data (shape = [n_features,] or [1, n_features]).
        y_test : float
            Target value for test data.
        alpha : float
            Regularization strength for Ridge regression.

        Returns
        -------
        ndarray
            [error, prediction].
        """
        model = linear_model.Ridge(alpha)
        mask = np.isfinite(y) & np.all(np.isfinite(x), axis=-1)

        if np.any(mask):
            y_clean = y[mask]
            x_clean = x[mask, :]
            model.fit(x_clean, y_clean)

            if x_test.ndim == 1:
                x_test = x_test.reshape(1, -1)

            preds = model.predict(x_test)
            preds[preds < 0] = 0
            error_ = y_test - preds
            return np.array([error_, preds]).squeeze()
        else:
            # If no valid training data, return NaNs
            return np.array([np.nan, np.nan]).squeeze()

    def compute_model(self, X_train, y_train, X_test, y_test, alpha):
                      # =None, clim_year_start=None, clim_year_end=None):
        """
        Fit and predict ridge regression model for spatiotemporal data using Dask for parallel computation.

        Parameters
        ----------
        X_train : xarray.DataArray
            Training predictor data (dims: T, features).
        y_train : xarray.DataArray
            Training predictand data (dims: T, Y, X).
        X_test : xarray.DataArray
            Test predictor data (dims: features) or broadcastable.
        y_test : xarray.DataArray
            Test predictand data (dims: Y, X).
        alpha : xarray.DataArray
            Spatial map of alpha values for each grid cell.

        Returns
        -------
        xarray.DataArray
            dims (output=2, Y, X) => [error, prediction].
        """

        chunksize_x = int(np.round(len(y_train.get_index("X")) / self.nb_cores))
        chunksize_y = int(np.round(len(y_train.get_index("Y")) / self.nb_cores))

        X_train['T'] = y_train['T']
        y_train = y_train.transpose('T','Y','X')
        X_test = X_test.squeeze()
        y_test = y_test.drop_vars('T').squeeze().transpose('Y','X')

        # Align alpha with y_train, y_test
        y_train, alpha = xr.align(y_train, alpha, join='outer')
        y_test, alpha = xr.align(y_test, alpha, join='outer')

        # if alpha is None:
        #     alpha, _ = self.compute_hyperparameters(
        #          y_train, X_train, clim_year_start, clim_year_end
        #     )

        client = Client(n_workers=self.nb_cores, threads_per_worker=1)
        result_da = xr.apply_ufunc(
            self.fit_predict,
            X_train,
            y_train.chunk({'Y': chunksize_y, 'X': chunksize_x}),
            X_test,
            y_test.chunk({'Y': chunksize_y, 'X': chunksize_x}),
            alpha.chunk({'Y': chunksize_y, 'X': chunksize_x}),
            input_core_dims=[
                ('T','features'),   # x
                ('T',),            # y
                ('features',),     # x_test
                (),                # y_test
                ()                 # alpha
            ],
            vectorize=True,
            dask='parallelized',
            output_core_dims=[('output',)],
            output_dtypes=['float'],
            dask_gufunc_kwargs={'output_sizes': {'output': 2}},
        )
        result_ = result_da.compute()
        client.close()
        return result_.isel(output=1)

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
        """Solver for Weibull shape parameter."""
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
        n_time = best_guess.size
        out = np.full((3, n_time), np.nan, float)

        if np.all(np.isnan(best_guess)) or np.isnan(dist_code) or np.isnan(T1) or np.isnan(T2) or np.isnan(error_variance):
            return out

        code = int(dist_code)

        if code == 1:
            error_std = np.sqrt(error_variance)
            out[0, :] = norm.cdf(T1, loc=best_guess, scale=error_std)
            out[1, :] = norm.cdf(T2, loc=best_guess, scale=error_std) - norm.cdf(T1, loc=best_guess, scale=error_std)
            out[2, :] = 1 - norm.cdf(T2, loc=best_guess, scale=error_std)

        elif code == 2:
            sigma = np.sqrt(np.log(1 + error_variance / (best_guess**2)))
            mu = np.log(best_guess) - sigma**2 / 2
            out[0, :] = lognorm.cdf(T1, s=sigma, scale=np.exp(mu))
            out[1, :] = lognorm.cdf(T2, s=sigma, scale=np.exp(mu)) - lognorm.cdf(T1, s=sigma, scale=np.exp(mu))
            out[2, :] = 1 - lognorm.cdf(T2, s=sigma, scale=np.exp(mu))      

        elif code == 3:
            scale = np.sqrt(error_variance)
            c1 = expon.cdf(T1, loc=best_guess, scale=scale)
            c2 = expon.cdf(T2, loc=best_guess, scale=scale)
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
                M = best_guess[i]
                V = error_variance
                
                if V <= 0 or M <= 0:
                    out[0, i] = np.nan
                    out[1, i] = np.nan
                    out[2, i] = np.nan
                    continue
        
                initial_guess = 2.0
                k = fsolve(WAS_Ridge_Model.weibull_shape_solver, initial_guess, args=(M, V))[0]

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

        elif code == 7:
            mu = best_guess
            c1 = poisson.cdf(T1, mu=mu)
            c2 = poisson.cdf(T2, mu=mu)
            
            out[0, :] = c1
            out[1, :] = c2 - c1
            out[2, :] = 1.0 - c2

        elif code == 8:
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
            if any(v is None for v in (best_code_da, best_shape_da, best_loc_da, best_scale_da)):
                raise ValueError(
                    "dist_method='bestfit' requires best_code_da, best_shape_da_da, best_loc_da, best_scale_da."
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

    def forecast(self, Predictant, clim_year_start, clim_year_end, Predictor, hindcast_det, Predictor_for_year, alpha, best_code_da=None, best_shape_da=None, best_loc_da=None, best_scale_da=None):
        """
        Generates a ridge-based forecast for a single future time (year).
        """
        y_test_dummy = xr.full_like(Predictant.isel(T=0), np.nan)

        chunksize_x = int(np.round(len(Predictant.get_index("X")) / self.nb_cores))
        chunksize_y = int(np.round(len(Predictant.get_index("Y")) / self.nb_cores))

        Predictor['T'] = Predictant['T']
        Predictant = Predictant.transpose('T','Y','X')
        Predictor_for_year_ = Predictor_for_year.squeeze()
        Predictant_st = standardize_timeseries(Predictant, clim_year_start, clim_year_end)        
        Predictant_st, alpha = xr.align(Predictant_st, alpha, join='outer')

        client = Client(n_workers=self.nb_cores, threads_per_worker=1)
        result_da = xr.apply_ufunc(
            self.fit_predict,
            Predictor,
            Predictant_st.chunk({'Y': chunksize_y, 'X': chunksize_x}),
            Predictor_for_year_,
            y_test_dummy.chunk({'Y': chunksize_y, 'X': chunksize_x}),
            alpha.chunk({'Y': chunksize_y, 'X': chunksize_x}),
            input_core_dims=[
                ('T','features'),
                ('T',),
                ('features',),
                (),
                ()
            ],
            vectorize=True,
            dask='parallelized',
            output_core_dims=[('output',)],
            output_dtypes=['float'],
            dask_gufunc_kwargs={'output_sizes': {'output': 2}},
        )
        result_ = result_da.compute()
        client.close()
        result_ = result_.isel(output=1)
        result_ = reverse_standardize(result_, Predictant, clim_year_start, clim_year_end) 
        index_start = Predictant.get_index("T").get_loc(str(clim_year_start)).start
        index_end = Predictant.get_index("T").get_loc(str(clim_year_end)).stop
        rainfall_for_tercile = Predictant.isel(T=slice(index_start, index_end))
        terciles = rainfall_for_tercile.quantile([0.33, 0.67], dim='T')
        T1_emp = terciles.isel(quantile=0).drop_vars('quantile')
        T2_emp = terciles.isel(quantile=1).drop_vars('quantile')
        error_variance = (Predictant - hindcast_det).var(dim='T')
        
        forecast_expanded = result_.expand_dims(
            T=[pd.Timestamp(Predictor_for_year.coords['T'].values[0]).to_pydatetime()]
        )
        year = Predictor_for_year.coords['T'].values[0].astype('datetime64[Y]').astype(int) + 1970
        T_value_1 = Predictant.isel(T=0).coords['T'].values
        month_1 = T_value_1.astype('datetime64[M]').astype(int) % 12 + 1
        new_T_value = np.datetime64(f"{year}-{month_1:02d}-{1:02d}")
        
        forecast_expanded = forecast_expanded.assign_coords(T=xr.DataArray([new_T_value], dims=["T"]))
        forecast_expanded['T'] = forecast_expanded['T'].astype('datetime64[ns]')

        dof = max(int(rainfall_for_tercile.sizes["T"]) - 1, 2)

        dm = self.dist_method

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
                forecast_expanded,
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

        elif dm == "nonparam":
            error_samples = Predictant - hindcast_det
            forecast_prob = xr.apply_ufunc(
                self.calculate_tercile_probabilities_nonparametric,
                forecast_expanded,
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
        return forecast_expanded, forecast_prob.transpose('probability', 'T', 'Y', 'X')


class WAS_Lasso_Model:
    """
    Lasso regression model for spatial rainfall (or continuous climate variable) prediction.

    Lasso (L1 regularization) performs automatic feature selection by driving less important
    predictor coefficients exactly to zero — particularly useful when working with many
    atmospheric, oceanic, or teleconnection indices where some may be redundant or noisy.

    Supports two optimization modes:
    - **'cluster'** (default): One optimal alpha per spatial cluster (fast, robust, recommended)
    - **'grid'**: Independent alpha optimization for each grid cell (very local fit, slow)

    Hyperparameter search methods:
    - 'bayesian' → Optuna Bayesian optimization (recommended, efficient)
    - 'random'   → scikit-learn RandomizedSearchCV
    - 'lassocv'  → scikit-learn LassoCV (fastest, grid-based)

    Compared to Ridge, Lasso typically requires:
    - smaller alpha values (less aggressive regularization)
    - higher max_iter (convergence can be slower near zero coefficients)

    Parameters
    ----------
    alpha_range : array-like, optional
        Range of alpha (regularization strength) values to search.
        Default: np.logspace(-6, 2, 100) — suitable for standardized rainfall data.

    n_clusters : int, default=5
        Number of spatial clusters (only used when mode='cluster').

    nb_cores : int, default=1
        Number of CPU cores for parallel computation (mainly used in grid mode).

    dist_method : str, default='nonparam'
        Method for computing tercile probabilities (PB/PN/PA).
        Currently only 'nonparam' fully implemented; 'bestfit' requires additional inputs.

    hyperparam_optimizer : {'bayesian', 'random', 'lassocv'}, default='bayesian'
        Strategy for finding optimal alpha:
        - 'bayesian' → Optuna Bayesian optimization (best quality/speed trade-off)
        - 'random'   → Randomized search
        - 'lassocv'  → Built-in LassoCV grid search (fastest)

    n_trials : int, default=50
        Number of trials for Bayesian optimization (Optuna).

    n_iter : int, default=50
        Number of parameter settings sampled in randomized search.

    mode : {'cluster', 'grid'}, default='cluster'
        Optimization strategy:
        - 'cluster' → one alpha per spatial cluster (fast, stable, recommended)
        - 'grid'    → independent alpha per grid cell (slow, maximally local)

    Notes
    -----
    - **Cluster mode** is strongly recommended for most rainfall/climate applications:
      - Much faster
      - More robust to noise and short time series
      - Better generalization
    - **Grid mode** should only be used when maximum spatial detail is critical and
      sufficient computing resources are available.
    - Lasso is more sensitive to feature scaling than Ridge → predictors and predictand
      should be properly standardized/normalized.
    - Requires at least 10 valid time steps per location/cluster to attempt optimization.
    - Negative predictions are clipped to zero (useful for rainfall).
    - Large domains in 'grid' mode benefit greatly from higher `nb_cores`.

    Methods
    -------
    compute_hyperparameters(predictand, predictor, clim_year_start, clim_year_end)
        Computes spatially varying optimal alpha values (and cluster map if mode='cluster').

    fit_predict(x, y, x_test, y_test, alpha)
        Fits Lasso model on one grid cell using given alpha and makes prediction.

    compute_model(X_train, y_train, X_test, y_test, alpha=None, clim_year_start=None, clim_year_end=None)
        Parallel Lasso regression across entire spatial domain using provided alpha map.

    compute_prob(Predictant, clim_year_start, clim_year_end, hindcast_det,
                 best_code_da=None, best_shape_da=None, best_loc_da=None, best_scale_da=None)
        Computes tercile probabilities for hindcast predictions.

    forecast(Predictant, clim_year_start, clim_year_end, Predictor, hindcast_det,
             Predictor_for_year, alpha, best_code_da=None, best_shape_da=None,
             best_loc_da=None, best_scale_da=None)
        Full end-to-end forecast pipeline for one target year:
        - deterministic Lasso prediction
        - tercile probabilities (PB, PN, PA)

    Examples
    --------
    Recommended workflow (cluster mode with Bayesian optimization):

    >>> lasso_model = WAS_Lasso_Model(
    ...     mode='cluster',
    ...     n_clusters=8,
    ...     hyperparam_optimizer='bayesian',
    ...     n_trials=80,
    ...     nb_cores=12
    ... )

    # 1. Compute spatially varying alpha (one per cluster)
    >>> alpha_map, cluster_map = lasso_model.compute_hyperparameters(
    ...     seasonal_rainfall, predictors, 1991, 2020)

    # 2. Train & predict on hindcast period
    >>> hindcast_pred = lasso_model.compute_model(
    ...     X_train=predictors,
    ...     y_train=seasonal_rainfall,
    ...     X_test=predictors_hindcast,
    ...     y_test=None,
    ...     alpha=alpha_map
    ... )

    # 3. Compute probabilistic hindcast (terciles)
    >>> hindcast_prob = lasso_model.compute_prob(
    ...     seasonal_rainfall, 1991, 2020, hindcast_pred,
    ...     best_code_da=dist_code_da, best_shape_da=shape_da,
    ...     best_loc_da=loc_da, best_scale_da=scale_da
    ... )

    # 4. Forecast next year (e.g. 2025)
    >>> forecast_det, forecast_prob = lasso_model.forecast(
    ...     seasonal_rainfall, 1991, 2020,
    ...     predictors, hindcast_pred,
    ...     predictor_2025,
    ...     alpha=alpha_map,
    ...     best_code_da=dist_code_da, best_shape_da=shape_da,
    ...     best_loc_da=loc_da, best_scale_da=scale_da
    ... )

    Warnings
    --------
    - Lasso can eliminate all predictors if alpha is too large → monitor selected features.
    - Very small training sets per cell/cluster may lead to unstable results.
    - For heavy-tailed rainfall distributions, consider log-transformation before modeling.
    - Use 'bayesian' optimizer for best quality; 'lassocv' is fastest but least flexible.
    """

    def __init__(self, alpha_range=None, n_clusters=5, nb_cores=1, dist_method="nonparam",
                 hyperparam_optimizer="bayesian", n_trials=50, n_iter=50, mode="cluster"):
        if alpha_range is None:
            alpha_range = np.logspace(-6, 2, 100)
        
        self.alpha_range = alpha_range
        self.n_clusters = n_clusters
        self.nb_cores = nb_cores
        self.dist_method = dist_method
        self.hyperparam_optimizer = hyperparam_optimizer
        self.n_trials = n_trials
        self.n_iter = n_iter
        self.mode = mode  # 'cluster' or 'grid'

    # ------------------ Hyperparameter Helpers ------------------

    def _optimize_hyperparameters_optuna(self, X, y, alpha_range):
        def objective(trial):
            alpha = trial.suggest_float('alpha', np.min(alpha_range), np.max(alpha_range), log=True)
            model = linear_model.Lasso(alpha=alpha, max_iter=10000)
            # Lasso is prone to zeroing out everything; negative MSE is a good metric here
            scores = cross_val_score(model, X, y, cv=3, scoring='neg_mean_squared_error')
            return scores.mean()
        
        study = optuna.create_study(direction='maximize')
        study.optimize(objective, n_trials=self.n_trials, show_progress_bar=False)
        return study.best_params['alpha']

    def _optimize_hyperparameters_randomized(self, X, y, alpha_range):
        param_distributions = {'alpha': loguniform(np.min(alpha_range), np.max(alpha_range))}
        model = linear_model.Lasso(max_iter=10000)
        random_search = RandomizedSearchCV(model, param_distributions, n_iter=self.n_iter, cv=3, scoring='neg_mean_squared_error', random_state=42)
        random_search.fit(X, y)
        return random_search.best_params_['alpha']

    def _optimize_single_cell(self, y_vec, X_mat):
        """Unified helper to optimize a single pixel or cluster mean for Lasso."""
        mask = np.isfinite(y_vec) & np.all(np.isfinite(X_mat), axis=-1)
        if np.sum(mask) < 10:  # Minimum valid years
            return np.nan
        
        X_clean, y_clean = X_mat[mask], y_vec[mask]
        
        try:
            if self.hyperparam_optimizer == 'bayesian':
                return self._optimize_hyperparameters_optuna(X_clean, y_clean, self.alpha_range)
            elif self.hyperparam_optimizer == 'random':
                return self._optimize_hyperparameters_randomized(X_clean, y_clean, self.alpha_range)
            else:
                # Fallback to LassoCV for speed in grid-mode if needed
                model_cv = linear_model.LassoCV(alphas=self.alpha_range, cv=3, verbose=1, n_jobs=-1)
                model_cv.fit(X_clean, y_clean)
                return model_cv.alpha_
        except:
            return np.nan

    # ------------------ Core Logic Methods ------------------

    def compute_hyperparameters(self, predictand, predictor, clim_year_start, clim_year_end):
        """
        Compute spatially varying optimal Lasso regularization parameter alpha.

        Parameters
        ----------
        predictand : xarray.DataArray
            Target variable (typically rainfall amount)
            Expected dimensions: ('T', 'Y', 'X')
        predictor : xarray.DataArray
            Predictor fields
            Expected dimensions: ('T', 'Y', 'X', 'features') or ('T', 'features')
        clim_year_start : int
            Start year of the reference climatology period for standardization
        clim_year_end : int
            End year of the reference climatology period for standardization

        Returns
        -------
        alpha_map : xarray.DataArray
            Spatial field (Y, X) containing optimal alpha value(s)
            - In 'grid' mode: different value possible for each cell
            - In 'cluster' mode: same value for all cells in the same cluster
        cluster_da : xarray.DataArray
            Cluster labels (when mode='cluster')
            Dummy array filled with 1s when mode='grid' (for compatibility)

        Raises
        ------
        ValueError
            If unknown mode is provided
        RuntimeError
            If dask parallel computation fails (grid mode)

        Notes
        -----
        - Time series with fewer than 10 valid points are skipped (return NaN)
        - Lasso uses increased max_iter=10000 to help convergence
        - All NaN handling is automatic (masking invalid time steps)
        - Progress messages are printed during long computations
        """
        predictor['T'] = predictand['T']
        predictand_st = standardize_timeseries(predictand, clim_year_start, clim_year_end)

        if self.mode == "grid":
            print(f"Lasso: Running Grid-wise optimization on {self.nb_cores} cores...")
            chunk_y = int(np.ceil(len(predictand_st.Y) / self.nb_cores))
            chunk_x = int(np.ceil(len(predictand_st.X) / self.nb_cores))
            p_st_chunked = predictand_st.chunk({'Y': chunk_y, 'X': chunk_x})

            client = Client(n_workers=self.nb_cores, threads_per_worker=1)
            alpha_array = xr.apply_ufunc(
                self._optimize_single_cell,
                p_st_chunked,
                predictor,
                input_core_dims=[('T',), ('T', 'features')],
                output_core_dims=[()],
                vectorize=True,
                dask='parallelized',
                output_dtypes=[float]
            )
            alpha_array = alpha_array.compute()
            client.close()
            cluster_da = xr.where(~np.isnan(alpha_array), 1, np.nan)
            return alpha_array, cluster_da

        else:
            print(f"Lasso: Running Cluster-wise optimization ({self.n_clusters} clusters)...")
            kmeans = KMeans(n_clusters=self.n_clusters)
            predictand_dropna = predictand.to_dataframe().reset_index().dropna().drop(columns=['T'])
            variable_column = predictand_dropna.columns[2]
            predictand_dropna['cluster'] = kmeans.fit_predict(
                predictand_dropna[[variable_column]]
            )
            
            # Convert cluster assignments back into an xarray structure
            df_unique = predictand_dropna.drop_duplicates(subset=['Y', 'X'])
            dataset = df_unique.set_index(['Y', 'X']).to_xarray()
            mask = xr.where(~np.isnan(predictand.isel(T=0)), 1, np.nan)
            cluster_da = (dataset['cluster'] * mask)
                   
            # Align cluster array with the predictand array
            x1, x2 = xr.align(predictand_st, cluster_da, join="outer")
            
            # Identify unique cluster labels
            clusters = np.unique(x2)
            clusters = clusters[~np.isnan(clusters)]
            cluster_da = x2
            
            alpha_map = xr.full_like(cluster_da, np.nan, dtype=float)
            
            for clus in clusters:
                y_cluster = x1.where(x2 == clus).mean(dim=['Y','X'], skipna=True).dropna(dim='T')
                if len(y_cluster['T']) > 0:
                    best_alpha = self._optimize_single_cell(y_cluster.values, predictor.sel(T=y_cluster['T']).values)
                    alpha_map = alpha_map.where(cluster_da != clus, best_alpha)
            
            return alpha_map, cluster_da

    def fit_predict(self, x, y, x_test, y_test, alpha):
        """
        Fit a Lasso model and make predictions.
        """
        model = linear_model.Lasso(alpha=alpha)
        mask = np.isfinite(y) & np.all(np.isfinite(x), axis=-1)

        if np.any(mask):
            y_clean = y[mask]
            x_clean = x[mask, :]
            model.fit(x_clean, y_clean)

            if x_test.ndim == 1:
                x_test = x_test.reshape(1, -1)

            preds = model.predict(x_test)
            preds[preds < 0] = 0
            error_ = y_test - preds
            return np.array([error_, preds]).squeeze()
        else:
            return np.array([np.nan, np.nan]).squeeze()
        
    def compute_model(self, X_train, y_train, X_test, y_test, alpha=None, clim_year_start=None, clim_year_end=None):
        """
        Computes Lasso predictions for spatiotemporal data using provided alpha values.

        Returns
        -------
        xarray.DataArray
            dims (output=2, Y, X) => [error, prediction].
        """
        chunksize_x = int(np.round(len(y_train.get_index("X")) / self.nb_cores))
        chunksize_y = int(np.round(len(y_train.get_index("Y")) / self.nb_cores))

        X_train['T'] = y_train['T']
        y_train = y_train.transpose('T','Y','X')
        X_test = X_test.squeeze()
        y_test = y_test.drop_vars('T').squeeze().transpose('Y','X')

        # Align alpha with y_train, y_test
        y_train, alpha = xr.align(y_train, alpha)
        y_test, alpha = xr.align(y_test, alpha)

        if alpha is None:
            alpha, _ = self.compute_hyperparameters(
                 y_train, X_train, clim_year_start, clim_year_end
            )
        
        client = Client(n_workers=self.nb_cores, threads_per_worker=1)
        result_da = xr.apply_ufunc(
            self.fit_predict,
            X_train,
            y_train.chunk({'Y': chunksize_y, 'X': chunksize_x}),
            X_test,
            y_test.chunk({'Y': chunksize_y, 'X': chunksize_x}),
            alpha.chunk({'Y': chunksize_y, 'X': chunksize_x}),
            input_core_dims=[
                ('T','features'),  # x
                ('T',),           # y
                ('features',),    # x_test
                (),               # y_test
                ()                # alpha
            ],
            vectorize=True,
            dask='parallelized',
            output_core_dims=[('output',)],
            output_dtypes=['float'],
            dask_gufunc_kwargs={'output_sizes': {'output': 2}}
        )
        result_ = result_da.compute()
        client.close()
        return result_.isel(output=1)
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
                    norm.ppf(0.32, loc=loc, scale=scale),
                    norm.ppf(0.67, loc=loc, scale=scale),
                )
            elif code == 2:
                return (
                    lognorm.ppf(0.32, s=shape, loc=loc, scale=scale),
                    lognorm.ppf(0.67, s=shape, loc=loc, scale=scale),
                )
            elif code == 3:
                return (
                    expon.ppf(0.32, loc=loc, scale=scale),
                    expon.ppf(0.67, loc=loc, scale=scale),
                )
            elif code == 4:
                return (
                    gamma.ppf(0.32, a=shape, loc=loc, scale=scale),
                    gamma.ppf(0.67, a=shape, loc=loc, scale=scale),
                )
            elif code == 5:
                return (
                    weibull_min.ppf(0.32, c=shape, loc=loc, scale=scale),
                    weibull_min.ppf(0.67, c=shape, loc=loc, scale=scale),
                )
            elif code == 6:
                # Note: Renamed 't_dist' to 't' for standard scipy.stats
                return (
                    t.ppf(0.32, df=shape, loc=loc, scale=scale),
                    t.ppf(0.67, df=shape, loc=loc, scale=scale),
                )
            elif code == 7:
                # Poisson: poisson.ppf(q, mu, loc=0)
                # ASSUMPTION: 'mu' (mean) is passed as 'shape'
                #             'loc' is passed as 'loc'
                #             'scale' is unused
                return (
                    poisson.ppf(0.32, mu=shape, loc=loc),
                    poisson.ppf(0.67, mu=shape, loc=loc),
                )
            elif code == 8:
                # Negative Binomial: nbinom.ppf(q, n, p, loc=0)
                # ASSUMPTION: 'n' (successes) is passed as 'shape'
                #             'p' (probability) is passed as 'scale'
                #             'loc' is passed as 'loc'
                return (
                    nbinom.ppf(0.32, n=shape, p=scale, loc=loc),
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
        terciles_emp = clim.quantile([0.32, 0.67], dim="T")
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

    def forecast(self, Predictant, clim_year_start, clim_year_end, Predictor, hindcast_det, Predictor_for_year, alpha, best_code_da=None, best_shape_da=None, best_loc_da=None, best_scale_da=None):
        """
        Generate a forecast for a single time (year) using Lasso with alpha, 
        then compute tercile probabilities from the chosen distribution method.

        Parameters
        ----------
        Predictant : xarray.DataArray
            Observed data (T, Y, X).
        clim_year_start : int
            Start year of the climatology period.
        clim_year_end : int
            End year of the climatology period.
        Predictor : xarray.DataArray
            Historical predictor data (T, features).
        hindcast_det : xarray.DataArray
            Historical deterministic forecast with dims (output=2, T, Y, X).
        Predictor_for_year : xarray.DataArray
            Single-year predictor data (features,).
        alpha : xarray.DataArray
            Spatial map of alpha values (Y, X).

        Returns
        -------
        result_ : xarray.DataArray
            dims (output=2, Y, X) => [error, prediction].
            For a real forecast scenario, the error is typically NaN.
        hindcast_prob : xarray.DataArray
            dims (probability=3, Y, X) => [PB, PN, PA].
        """
        # 1) Create dummy y_test => shape (Y, X) with NaN
        y_test_dummy = xr.full_like(Predictant.isel(T=0), np.nan)

        # 2) Chunk sizes
        chunksize_x = int(np.round(len(Predictant.get_index("X")) / self.nb_cores))
        chunksize_y = int(np.round(len(Predictant.get_index("Y")) / self.nb_cores))
        
        Predictant_st = standardize_timeseries(Predictant, clim_year_start, clim_year_end) 
        
        # Align
        Predictor['T'] = Predictant['T']
        Predictant = Predictant.transpose('T','Y','X')
        Predictor_for_year_ = Predictor_for_year.squeeze()

        # Align alpha with the domain
        Predictant_st, alpha = xr.align(Predictant_st, alpha, join='outer')

        # 3) Fit+predict in parallel => produce (2, Y, X)
        client = Client(n_workers=self.nb_cores, threads_per_worker=1)
        result_da = xr.apply_ufunc(
            self.fit_predict,
            Predictor,
            Predictant_st.chunk({'Y': chunksize_y, 'X': chunksize_x}),
            Predictor_for_year_,
            y_test_dummy.chunk({'Y': chunksize_y, 'X': chunksize_x}),  # dummy
            alpha.chunk({'Y': chunksize_y, 'X': chunksize_x}),
            input_core_dims=[
                ('T','features'),  # x
                ('T',),           # y
                ('features',),    # x_test
                (),               # y_test
                ()                # alpha
            ],
            vectorize=True,
            dask='parallelized',
            output_core_dims=[('output',)],  # => [error, prediction]
            output_dtypes=['float'],
            dask_gufunc_kwargs={'output_sizes': {'output': 2}}
        )
        result_ = result_da.compute()
        client.close()
        result_ = result_.isel(output=1)
        result_ = reverse_standardize(result_, Predictant, clim_year_start, clim_year_end) 
        # result_ => dims (output=2, Y, X). 
        # For a real future forecast, "error" is NaN, "prediction" is the forecast.

        # 2) Compute thresholds T1, T2 from climatology
        index_start = Predictant.get_index("T").get_loc(str(clim_year_start)).start
        index_end   = Predictant.get_index("T").get_loc(str(clim_year_end)).stop
        rainfall_for_tercile = Predictant.isel(T=slice(index_start, index_end))
        terciles = rainfall_for_tercile.quantile([0.33, 0.67], dim='T')
        T1_emp = terciles.isel(quantile=0).drop_vars('quantile')
        T2_emp = terciles.isel(quantile=1).drop_vars('quantile')
        error_variance = (Predictant - hindcast_det).var(dim='T')
        
        # Expand single prediction to T=1 so probability methods can handle it
        forecast_expanded = result_.expand_dims(
            T=[pd.Timestamp(Predictor_for_year.coords['T'].values[0]).to_pydatetime()]
        )
        year = Predictor_for_year.coords['T'].values[0].astype('datetime64[Y]').astype(int) + 1970
        # year = Predictor_for_year.coords['T'].values.astype('datetime64[Y]').astype(int)[0] + 1970  
        T_value_1 = Predictant.isel(T=0).coords['T'].values  # Get the datetime64 value from da1
        month_1 = T_value_1.astype('datetime64[M]').astype(int) % 12 + 1  # Extract month
        new_T_value = np.datetime64(f"{year}-{month_1:02d}-{1:02d}")
        
        forecast_expanded = forecast_expanded.assign_coords(T=xr.DataArray([new_T_value], dims=["T"]))
        forecast_expanded['T'] = forecast_expanded['T'].astype('datetime64[ns]')

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
                forecast_expanded,
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
                forecast_expanded,
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
        return forecast_expanded, forecast_prob.transpose('probability', 'T', 'Y', 'X')



class WAS_ElasticNet_Model:
    """
    ElasticNet regression model for spatial rainfall (or continuous climate variable) prediction.

    ElasticNet combines L1 (Lasso) and L2 (Ridge) penalties in a single model:
    - L1 penalty → feature selection (sparsity)
    - L2 penalty → coefficient shrinkage and stability with correlated predictors
    This makes ElasticNet particularly suitable for climate modeling where:
    - Many predictors (teleconnections, atmospheric fields) are often correlated
    - Some predictors may be irrelevant or redundant

    Two spatial optimization strategies:
    - **'cluster'** (default): One optimal (alpha, l1_ratio) pair per spatial cluster
      → Fast, robust, excellent generalization — **recommended for most applications**
    - **'grid'**: Independent optimization per grid cell
      → Maximum spatial detail, but **very computationally expensive**

    Supported hyperparameter search methods:
    - 'bayesian' → Optuna Bayesian optimization (best quality/speed trade-off)
    - 'random'   → scikit-learn RandomizedSearchCV
    - 'elasticnetcv' → scikit-learn ElasticNetCV (fastest, discrete grid)

    Parameters
    ----------
    alpha_range : array-like, optional
        Range of total regularization strength (alpha).
        Default: np.logspace(-6, 2, 100) — well-suited for standardized rainfall data.

    l1_ratio_range : array-like, optional
        Range of L1 penalty ratio (0 = pure Ridge, 1 = pure Lasso).
        Default: [0.1, 0.5, 0.7, 0.9, 0.95, 0.99, 1.0]

    n_clusters : int, default=5
        Number of spatial clusters (only used when mode='cluster').

    nb_cores : int, default=1
        Number of CPU cores for parallel computation (mainly used in grid mode).

    dist_method : str, default='nonparam'
        Method for computing tercile probabilities (PB/PN/PA).
        Currently only 'nonparam' fully implemented; 'bestfit' requires additional inputs.

    hyperparam_optimizer : {'bayesian', 'random', 'elasticnetcv'}, default='bayesian'
        Strategy for finding optimal (alpha, l1_ratio):
        - 'bayesian' → Optuna Bayesian optimization (recommended)
        - 'random'   → Randomized search
        - 'elasticnetcv' → Built-in ElasticNetCV (fastest)

    n_trials : int, default=50
        Number of trials for Bayesian optimization (Optuna).

    n_iter : int, default=50
        Number of parameter settings sampled in randomized search.

    mode : {'cluster', 'grid'}, default='cluster'
        Optimization approach:
        - 'cluster' → one (alpha, l1_ratio) per spatial cluster (fast & stable)
        - 'grid'    → independent optimization per grid cell (slow, maximally local)

    Notes
    -----
    - **Cluster mode** is the recommended default for most rainfall/climate applications:
      - Much faster
      - More robust to noise, short time series, and data sparsity
      - Excellent generalization
    - **Grid mode** should only be used when maximum spatial detail is critical and
      sufficient computing resources are available.
    - ElasticNet benefits from feature/predictand standardization — ensure proper scaling.
    - Requires at least 10 valid time steps per location/cluster to attempt optimization.
    - Negative predictions are automatically clipped to zero (useful for rainfall).
    - Large domains in 'grid' mode benefit greatly from higher `nb_cores`.

    Methods
    -------
    compute_hyperparameters(predictand, predictor, clim_year_start, clim_year_end)
        Computes spatially varying optimal (alpha, l1_ratio) pairs (and cluster map if mode='cluster').

    fit_predict(x, y, x_test, y_test, alpha, l1_ratio)
        Fits ElasticNet model on one grid cell using given parameters and makes prediction.

    compute_model(X_train, y_train, X_test, y_test, alpha=None, l1_ratio=None, clim_year_start=None, clim_year_end=None)
        Parallel ElasticNet regression across entire spatial domain using provided parameter maps.

    compute_prob(Predictant, clim_year_start, clim_year_end, hindcast_det,
                 best_code_da=None, best_shape_da=None, best_loc_da=None, best_scale_da=None)
        Computes tercile probabilities for hindcast predictions.

    forecast(Predictant, clim_year_start, clim_year_end, Predictor, hindcast_det,
             Predictor_for_year, alpha, l1_ratio, best_code_da=None, best_shape_da=None,
             best_loc_da=None, best_scale_da=None)
        Full end-to-end forecast pipeline for one target year:
        - deterministic ElasticNet prediction
        - tercile probabilities (PB, PN, PA)

    Examples
    --------
    Recommended workflow (cluster mode with Bayesian optimization):

    >>> enet_model = WAS_ElasticNet_Model(
    ...     mode='cluster',
    ...     n_clusters=8,
    ...     hyperparam_optimizer='bayesian',
    ...     n_trials=80,
    ...     nb_cores=12
    ... )

    # 1. Compute spatially varying (alpha, l1_ratio) — one pair per cluster
    >>> alpha_map, l1_map, cluster_map = enet_model.compute_hyperparameters(
    ...     seasonal_rainfall, predictors, 1991, 2020)

    # 2. Train & predict on hindcast period
    >>> hindcast_pred = enet_model.compute_model(
    ...     X_train=predictors,
    ...     y_train=seasonal_rainfall,
    ...     X_test=predictors_hindcast,
    ...     y_test=None,
    ...     alpha=alpha_map,
    ...     l1_ratio=l1_map
    ... )

    # 3. Compute probabilistic hindcast (terciles)
    >>> hindcast_prob = enet_model.compute_prob(
    ...     seasonal_rainfall, 1991, 2020, hindcast_pred,
    ...     best_code_da=dist_code_da, best_shape_da=shape_da,
    ...     best_loc_da=loc_da, best_scale_da=scale_da
    ... )

    # 4. Forecast next year (e.g. 2025)
    >>> forecast_det, forecast_prob = enet_model.forecast(
    ...     seasonal_rainfall, 1991, 2020,
    ...     predictors, hindcast_pred,
    ...     predictor_2025,
    ...     alpha=alpha_map,
    ...     l1_ratio=l1_map,
    ...     best_code_da=dist_code_da, best_shape_da=shape_da,
    ...     best_loc_da=loc_da, best_scale_da=scale_da
    ... )

    Warnings
    --------
    - Very high alpha or l1_ratio=1.0 can eliminate all predictors → monitor selected features.
    - Small training sets per cell/cluster may lead to unstable results.
    - For heavy-tailed rainfall, consider log-transformation before modeling.
    - Use 'bayesian' optimizer for best quality; 'elasticnetcv' is fastest but least flexible.
    """


    def __init__(self, alpha_range=None, l1_ratio_range=None, n_clusters=5, nb_cores=1, 
                 dist_method="nonparam", hyperparam_optimizer="bayesian", n_trials=50, n_iter=50, mode="cluster"):
        if alpha_range is None:
            alpha_range = np.logspace(-6, 2, 100)
        if l1_ratio_range is None:
            l1_ratio_range = [0.1, 0.5, 0.7, 0.9, 0.95, 0.99, 1.0]
            
        self.alpha_range = alpha_range
        self.l1_ratio_range = l1_ratio_range
        self.n_clusters = n_clusters
        self.nb_cores = nb_cores
        self.dist_method = dist_method
        self.hyperparam_optimizer = hyperparam_optimizer
        self.n_trials = n_trials
        self.n_iter = n_iter
        self.mode = mode # 'cluster' or 'grid'

    # ------------------ Hyperparameter Helpers ------------------

    def _optimize_hyperparameters_optuna(self, X, y, alpha_range, l1_ratio_range):
        def objective(trial):
            alpha = trial.suggest_float('alpha', np.min(alpha_range), np.max(alpha_range), log=True)
            l1_ratio = trial.suggest_float('l1_ratio', np.min(l1_ratio_range), np.max(l1_ratio_range))
            model = linear_model.ElasticNet(alpha=alpha, l1_ratio=l1_ratio, max_iter=10000)
            scores = cross_val_score(model, X, y, cv=3, scoring='neg_mean_squared_error')
            return scores.mean()
        
        study = optuna.create_study(direction='maximize')
        study.optimize(objective, n_trials=self.n_trials, show_progress_bar=False)
        return study.best_params['alpha'], study.best_params['l1_ratio']

    def _optimize_hyperparameters_randomized(self, X, y, alpha_range, l1_ratio_range):
        param_distributions = {
            'alpha': loguniform(np.min(alpha_range), np.max(alpha_range)),
            'l1_ratio': l1_ratio_range
        }
        model = linear_model.ElasticNet(max_iter=10000)
        random_search = RandomizedSearchCV(model, param_distributions, n_iter=self.n_iter,
                                           cv=3, scoring='neg_mean_squared_error', random_state=42)
        random_search.fit(X, y)
        return random_search.best_params_['alpha'], random_search.best_params_['l1_ratio']

    def _optimize_single_cell(self, y_vec, X_mat):
        """Unified helper to optimize a single pixel or cluster mean for ElasticNet."""
        mask = np.isfinite(y_vec) & np.all(np.isfinite(X_mat), axis=-1)
        if np.sum(mask) < 10:
            return np.array([np.nan, np.nan]) # Return [alpha, l1_ratio]
        
        X_clean, y_clean = X_mat[mask], y_vec[mask]
        
        try:
            if self.hyperparam_optimizer == 'bayesian':
                a, l1 = self._optimize_hyperparameters_optuna(X_clean, y_clean, self.alpha_range, self.l1_ratio_range)
            elif self.hyperparam_optimizer == 'random':
                a, l1 = self._optimize_hyperparameters_randomized(X_clean, y_clean, self.alpha_range, self.l1_ratio_range)
            else:
                model_cv = linear_model.ElasticNetCV(alphas=self.alpha_range, l1_ratio=self.l1_ratio_range, cv=3, verbose=1, n_jobs=-1)
                model_cv.fit(X_clean, y_clean)
                a, l1 = model_cv.alpha_, model_cv.l1_ratio_
            return np.array([a, l1])
        except:
            return np.array([np.nan, np.nan])

    # ------------------ Core Logic Methods ------------------

    def compute_hyperparameters(self, predictand, predictor, clim_year_start, clim_year_end):
        
        """
        Compute spatially varying optimal ElasticNet parameters (alpha and l1_ratio).

        Parameters
        ----------
        predictand : xarray.DataArray
            Target variable (typically rainfall)
            Expected dimensions: ('T', 'Y', 'X')
        predictor : xarray.DataArray
            Predictor fields
            Expected dimensions: ('T', 'Y', 'X', 'features') or ('T', 'features')
        clim_year_start : int
            Start year of climatological standardization period
        clim_year_end : int
            End year of climatological standardization period

        Returns
        -------
        alpha_map : xarray.DataArray
            Spatial field (Y, X) of optimal alpha values
        l1_ratio_map : xarray.DataArray
            Spatial field (Y, X) of optimal l1_ratio values
        cluster_da : xarray.DataArray
            Cluster labels (integers) when mode='cluster'
            Dummy array filled with 1s when mode='grid'

        Raises
        ------
        ValueError
            If invalid mode is provided
        RuntimeError
            If dask parallel computation fails (in grid mode)

        Notes
        -----
        - Locations/time series with < 10 valid points are skipped (return NaN)
        - NaN handling is automatic (invalid time steps are masked)
        - Progress messages are printed for long-running computations
        """

        predictor['T'] = predictand['T']
        predictand_st = standardize_timeseries(predictand, clim_year_start, clim_year_end)

        if self.mode == "grid":
            print(f"ElasticNet: Running Grid-wise optimization on {self.nb_cores} cores...")
            chunk_y = int(np.ceil(len(predictand_st.Y) / self.nb_cores))
            chunk_x = int(np.ceil(len(predictand_st.X) / self.nb_cores))
            p_st_chunked = predictand_st.chunk({'Y': chunk_y, 'X': chunk_x})

            client = Client(n_workers=self.nb_cores, threads_per_worker=1)
            # Returns a DataArray with a new 'params' dimension [alpha, l1_ratio]
            param_array = xr.apply_ufunc(
                self._optimize_single_cell, p_st_chunked, predictor,
                input_core_dims=[('T',), ('T', 'features')],
                output_core_dims=[('params',)],
                vectorize=True, dask='parallelized', output_dtypes=[float],
                dask_gufunc_kwargs={'output_sizes': {'params': 2}}
            ).compute()
            client.close()
            
            alpha_array = param_array.isel(params=0)
            l1_ratio_array = param_array.isel(params=1)
            cluster_da = xr.where(~np.isnan(alpha_array), 1, np.nan)
            return alpha_array, l1_ratio_array, cluster_da

        else:
            print(f"ElasticNet: Running Cluster-wise optimization...")
            kmeans = KMeans(n_clusters=self.n_clusters)
            predictand_dropna = predictand.to_dataframe().reset_index().dropna().drop(columns=['T'])
            variable_column = predictand_dropna.columns[2]
            predictand_dropna['cluster'] = kmeans.fit_predict(
                predictand_dropna[[variable_column]]
            )
            
            # Convert cluster assignments back into an xarray structure
            df_unique = predictand_dropna.drop_duplicates(subset=['Y', 'X'])
            dataset = df_unique.set_index(['Y', 'X']).to_xarray()
            mask = xr.where(~np.isnan(predictand.isel(T=0)), 1, np.nan)
            cluster_da = (dataset['cluster'] * mask)
                   
            # Align cluster array with the predictand array
            x1, x2 = xr.align(predictand_st, cluster_da, join="outer")
            
            # Identify unique cluster labels
            clusters = np.unique(x2)
            clusters = clusters[~np.isnan(clusters)]
            cluster_da = x2
            
            alpha_map = xr.full_like(cluster_da, np.nan, dtype=float)
            l1_map = xr.full_like(cluster_da, np.nan, dtype=float)
            
            for clus in clusters:
                y_cluster = x1.where(x2 == clus).mean(dim=['Y','X'], skipna=True).dropna(dim='T')
                if len(y_cluster['T']) > 0:
                    params = self._optimize_single_cell(y_cluster.values, predictor.sel(T=y_cluster['T']).values)
                    alpha_map = alpha_map.where(cluster_da != clus, params[0])
                    l1_map = l1_map.where(cluster_da != clus, params[1])
            
            return alpha_map, l1_map, cluster_da

    def fit_predict(self, x, y, x_test, y_test, alpha, l1_ratio):
        if np.isnan(alpha) or np.isnan(l1_ratio): return np.array([np.nan, np.nan])
        model = linear_model.ElasticNet(alpha=alpha, l1_ratio=l1_ratio)
        mask = np.isfinite(y) & np.all(np.isfinite(x), axis=-1)
        if np.any(mask):
            model.fit(x[mask], y[mask])
            pred = model.predict(x_test.reshape(1, -1) if x_test.ndim == 1 else x_test)
            pred = np.maximum(pred, 0)
            return np.array([(y_test - pred).squeeze(), pred.squeeze()]).squeeze()
        return np.array([np.nan, np.nan])

    def compute_model(self, X_train, y_train, X_test, y_test, alpha=None, l1_ratio=None, clim_year_start=None, clim_year_end=None):
        """
        Performs parallelized ElasticNet modeling for spatiotemporal data.
        Returns [error, prediction] per grid cell.
        """
            
        chunksize_x = int(np.round(len(y_train.get_index("X")) / self.nb_cores))
        chunksize_y = int(np.round(len(y_train.get_index("Y")) / self.nb_cores))
        
        X_train['T'] = y_train['T']
        y_train = y_train.transpose('T','Y','X')
        X_test = X_test.squeeze()
        y_test = y_test.drop_vars('T').squeeze().transpose('Y','X')
        
        # Align alpha, l1_ratio, y_train, y_test
        y_train, alpha = xr.align(y_train, alpha)
        y_test, alpha = xr.align(y_test, alpha)
        l1_ratio, alpha = xr.align(l1_ratio, alpha)

        if alpha is None or l1_ratio is None:
            alpha, l1_ratio, _ = self.compute_hyperparameters(
                 y_train, X_train, clim_year_start, clim_year_end
            )

        client = Client(n_workers=self.nb_cores, threads_per_worker=1)
        result_da = xr.apply_ufunc(
            self.fit_predict,
            X_train,
            y_train.chunk({'Y': chunksize_y, 'X': chunksize_x}),
            X_test,
            y_test.chunk({'Y': chunksize_y, 'X': chunksize_x}),
            alpha.chunk({'Y': chunksize_y, 'X': chunksize_x}),
            l1_ratio.chunk({'Y': chunksize_y, 'X': chunksize_x}),
            input_core_dims=[
                ('T','features'),  # x
                ('T',),           # y
                ('features',),    # x_test
                (),               # y_test
                (),               # alpha
                ()                # l1_ratio
            ],
            vectorize=True,
            dask='parallelized',
            output_core_dims=[('output',)],
            output_dtypes=['float'],
            dask_gufunc_kwargs={'output_sizes': {'output': 2}},
        )
        result_ = result_da.compute()
        client.close()
        return result_.isel(output=1)

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
                    norm.ppf(0.32, loc=loc, scale=scale),
                    norm.ppf(0.67, loc=loc, scale=scale),
                )
            elif code == 2:
                return (
                    lognorm.ppf(0.32, s=shape, loc=loc, scale=scale),
                    lognorm.ppf(0.67, s=shape, loc=loc, scale=scale),
                )
            elif code == 3:
                return (
                    expon.ppf(0.32, loc=loc, scale=scale),
                    expon.ppf(0.67, loc=loc, scale=scale),
                )
            elif code == 4:
                return (
                    gamma.ppf(0.32, a=shape, loc=loc, scale=scale),
                    gamma.ppf(0.67, a=shape, loc=loc, scale=scale),
                )
            elif code == 5:
                return (
                    weibull_min.ppf(0.32, c=shape, loc=loc, scale=scale),
                    weibull_min.ppf(0.67, c=shape, loc=loc, scale=scale),
                )
            elif code == 6:
                # Note: Renamed 't_dist' to 't' for standard scipy.stats
                return (
                    t.ppf(0.32, df=shape, loc=loc, scale=scale),
                    t.ppf(0.67, df=shape, loc=loc, scale=scale),
                )
            elif code == 7:
                # Poisson: poisson.ppf(q, mu, loc=0)
                # ASSUMPTION: 'mu' (mean) is passed as 'shape'
                #             'loc' is passed as 'loc'
                #             'scale' is unused
                return (
                    poisson.ppf(0.32, mu=shape, loc=loc),
                    poisson.ppf(0.67, mu=shape, loc=loc),
                )
            elif code == 8:
                # Negative Binomial: nbinom.ppf(q, n, p, loc=0)
                # ASSUMPTION: 'n' (successes) is passed as 'shape'
                #             'p' (probability) is passed as 'scale'
                #             'loc' is passed as 'loc'
                return (
                    nbinom.ppf(0.32, n=shape, p=scale, loc=loc),
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
        terciles_emp = clim.quantile([0.32, 0.67], dim="T")
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

    # ------------------- FORECAST METHOD -------------------
    def forecast(
        self,
        Predictant,
        clim_year_start,
        clim_year_end,
        Predictor,
        hindcast_det,
        Predictor_for_year,
        alpha,
        l1_ratio, best_code_da=None, best_shape_da=None, best_loc_da=None, best_scale_da=None
    ):
        """
        Generates a single-year forecast using ElasticNet with (alpha, l1_ratio), 
        and computes tercile probabilities.

        Parameters
        ----------
        Predictant : xarray.DataArray
            Observed data (T, Y, X).
        clim_year_start, clim_year_end : int
            Climatology reference period.
        Predictor : xarray.DataArray
            Historical predictor data (T, features).
        hindcast_det : xarray.DataArray
            Historical deterministic forecast with dims (output=2, T, Y, X).
        Predictor_for_year : xarray.DataArray
            Single-year predictor data (features,).
        alpha : xarray.DataArray
            Spatial map of alpha values (Y, X).
        l1_ratio : xarray.DataArray
            Spatial map of l1_ratio values (Y, X).

        Returns
        -------
        result_ : xarray.DataArray
            dims (output=2, Y, X) => [error, forecast].
            For a real forecast scenario, 'error' is NaN.
        hindcast_prob : xarray.DataArray
            dims (probability=3, Y, X) => [PB, PN, PA].
        """
        # 1) Create a dummy y_test => shape (Y, X) with NaN
        y_test_dummy = xr.full_like(Predictant.isel(T=0), np.nan)

        # 2) Align shapes
        Predictor['T'] = Predictant['T']
        Predictant = Predictant.transpose('T','Y','X')
        Predictor_for_year_ = Predictor_for_year.squeeze()
        Predictant_st = standardize_timeseries(Predictant, clim_year_start, clim_year_end)
        # Align alpha, l1_ratio with domain
        Predictant_st, alpha, l1_ratio = xr.align(Predictant_st, alpha, l1_ratio, join="outer")

        # 3) Parallel fit+predict => shape (2, Y, X)
        client = Client(n_workers=self.nb_cores, threads_per_worker=1)
        result_da = xr.apply_ufunc(
            self.fit_predict,
            Predictor,
            Predictant_st.chunk({'Y': int(np.round(len(Predictant.get_index("Y")) / self.nb_cores)),
                              'X': int(np.round(len(Predictant.get_index("X")) / self.nb_cores))}),
            Predictor_for_year_,
            y_test_dummy,   # pass dummy test => yields [NaN, forecast]
            alpha,
            l1_ratio,
            input_core_dims=[
                ('T','features'),  # x
                ('T',),           # y
                ('features',),    # x_test
                (),               # y_test
                (),               # alpha
                ()                # l1_ratio
            ],
            vectorize=True,
            dask='parallelized',
            output_core_dims=[('output',)],  # => [error, prediction]
            output_dtypes=['float'],
            dask_gufunc_kwargs={'output_sizes': {'output': 2}}
        )
        result_ = result_da.compute()
        client.close()

        result_ = result_.isel(output=1)
        result_ = reverse_standardize(result_, Predictant, clim_year_start, clim_year_end)
        # result_ => dims (output=2, Y, X). 
        # For a real future forecast, "error" is NaN, "prediction" is the forecast.

        # 2) Compute thresholds T1, T2 from climatology
        index_start = Predictant.get_index("T").get_loc(str(clim_year_start)).start
        index_end   = Predictant.get_index("T").get_loc(str(clim_year_end)).stop
        rainfall_for_tercile = Predictant.isel(T=slice(index_start, index_end))
        terciles = rainfall_for_tercile.quantile([0.32, 0.67], dim='T')
        T1_emp = terciles.isel(quantile=0).drop_vars('quantile')
        T2_emp = terciles.isel(quantile=1).drop_vars('quantile')
        error_variance = (Predictant - hindcast_det).var(dim='T')
        
        # Expand single prediction to T=1 so probability methods can handle it
        forecast_expanded = result_.expand_dims(
            T=[pd.Timestamp(Predictor_for_year.coords['T'].values[0]).to_pydatetime()]
        )
        year = Predictor_for_year.coords['T'].values[0].astype('datetime64[Y]').astype(int) + 1970
        # year = Predictor_for_year.coords['T'].values.astype('datetime64[Y]').astype(int)[0] + 1970  
        T_value_1 = Predictant.isel(T=0).coords['T'].values  # Get the datetime64 value from da1
        month_1 = T_value_1.astype('datetime64[M]').astype(int) % 12 + 1  # Extract month
        new_T_value = np.datetime64(f"{year}-{month_1:02d}-{1:02d}")
        
        forecast_expanded = forecast_expanded.assign_coords(T=xr.DataArray([new_T_value], dims=["T"]))
        forecast_expanded['T'] = forecast_expanded['T'].astype('datetime64[ns]')

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
                forecast_expanded,
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
                forecast_expanded,
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
        return forecast_expanded, forecast_prob.transpose('probability', 'T', 'Y', 'X')



class WAS_LassoLars_Model:
    """
    LassoLars regression model for spatial rainfall (or continuous climate variable) prediction.

    LassoLars implements **Lasso** (L1 regularization) using the **Least-Angle Regression (LARS)** algorithm,
    which computes the full regularization path efficiently. It is particularly well-suited when:

    - Number of samples (time steps) is small per grid cell/cluster
    - High numerical stability is required (LARS is less prone to convergence issues than coordinate descent)
    - Predictors are highly correlated (common in climate/teleconnection data)
    - You want precise control over the regularization path

    Compared to standard Lasso (coordinate descent), LassoLars is:
    - **More stable** for small/noisy datasets
    - **Slightly slower** (especially for many features)
    - **Better behaved** near the sparse solution

    Two spatial optimization strategies are supported:
    - **'cluster'** (recommended): One optimal alpha per spatial cluster → fast, robust, excellent generalization
    - **'grid'**: Independent alpha optimization per grid cell → maximum spatial detail, very computationally expensive

    Hyperparameter search methods:
    - 'bayesian' → Optuna Bayesian optimization (recommended — best quality/speed trade-off)
    - 'random'   → scikit-learn RandomizedSearchCV
    - 'lassolars_cv' → scikit-learn LassoLarsCV (fastest, uses full LARS path)

    Parameters
    ----------
    alpha_range : array-like, optional
        Range of regularization strength (alpha) values to search.
        Default: np.logspace(-6, 2, 100) — well-suited for standardized rainfall data.

    n_clusters : int, default=5
        Number of spatial clusters (only used when mode='cluster').

    nb_cores : int, default=1
        Number of CPU cores for parallel computation (mainly used in grid mode).

    dist_method : str, default='nonparam'
        Method for computing tercile probabilities (PB/PN/PA).
        Currently only 'nonparam' fully implemented; 'bestfit' requires additional inputs.

    hyperparam_optimizer : {'bayesian', 'random', 'lassolars_cv'}, default='bayesian'
        Strategy for finding optimal alpha:
        - 'bayesian' → Optuna Bayesian optimization (recommended)
        - 'random'   → Randomized search
        - 'lassolars_cv' → Built-in LassoLarsCV (fastest, uses full LARS path)

    n_trials : int, default=50
        Number of trials for Bayesian optimization (Optuna).

    n_iter : int, default=50
        Number of parameter settings sampled in randomized search.

    mode : {'cluster', 'grid'}, default='cluster'
        Optimization approach:
        - 'cluster' → one alpha per spatial cluster (fast & robust — recommended)
        - 'grid'    → independent alpha per grid cell (slow, maximally local)

    Notes
    -----
    - **Cluster mode** is the recommended default for most rainfall/climate applications:
      - Much faster
      - More robust to noise, short time series, and data sparsity
      - Excellent generalization
    - **Grid mode** should only be used when maximum spatial detail is critical and
      sufficient computing resources are available.
    - LassoLars is more stable than coordinate-descent Lasso for small datasets,
      but may be slower for many features.
    - Requires at least 10 valid time steps per location/cluster to attempt optimization.
    - Negative predictions are automatically clipped to zero (useful for rainfall).
    - Large domains in 'grid' mode benefit greatly from higher `nb_cores`.

    Methods
    -------
    compute_hyperparameters(predictand, predictor, clim_year_start, clim_year_end)
        Computes spatially varying optimal alpha values (and cluster map if mode='cluster').

    fit_predict(x, y, x_test, y_test, alpha)
        Fits LassoLars model on one grid cell using given alpha and makes prediction.

    compute_model(X_train, y_train, X_test, y_test, alpha=None, clim_year_start=None, clim_year_end=None)
        Parallel LassoLars regression across entire spatial domain using provided alpha map.

    compute_prob(Predictant, clim_year_start, clim_year_end, hindcast_det,
                 best_code_da=None, best_shape_da=None, best_loc_da=None, best_scale_da=None)
        Computes tercile probabilities for hindcast predictions.

    forecast(Predictant, clim_year_start, clim_year_end, Predictor, hindcast_det,
             Predictor_for_year, alpha, best_code_da=None, best_shape_da=None,
             best_loc_da=None, best_scale_da=None)
        Full end-to-end forecast pipeline for one target year:
        - deterministic LassoLars prediction
        - tercile probabilities (PB, PN, PA)

    Examples
    --------
    Recommended workflow (cluster mode with Bayesian optimization):

    >>> lars_model = WAS_LassoLars_Model(
    ...     mode='cluster',
    ...     n_clusters=8,
    ...     hyperparam_optimizer='bayesian',
    ...     n_trials=80,
    ...     nb_cores=12
    ... )

    # 1. Compute spatially varying alpha (one per cluster)
    >>> alpha_map, cluster_map = lars_model.compute_hyperparameters(
    ...     seasonal_rainfall, predictors, 1991, 2020)

    # 2. Train & predict on hindcast period
    >>> hindcast_pred = lars_model.compute_model(
    ...     X_train=predictors,
    ...     y_train=seasonal_rainfall,
    ...     X_test=predictors_hindcast,
    ...     y_test=None,
    ...     alpha=alpha_map
    ... )

    # 3. Compute probabilistic hindcast (terciles)
    >>> hindcast_prob = lars_model.compute_prob(
    ...     seasonal_rainfall, 1991, 2020, hindcast_pred,
    ...     best_code_da=dist_code_da, best_shape_da=shape_da,
    ...     best_loc_da=loc_da, best_scale_da=scale_da
    ... )

    # 4. Forecast next year (e.g. 2025)
    >>> forecast_det, forecast_prob = lars_model.forecast(
    ...     seasonal_rainfall, 1991, 2020,
    ...     predictors, hindcast_pred,
    ...     predictor_2025,
    ...     alpha=alpha_map,
    ...     best_code_da=dist_code_da, best_shape_da=shape_da,
    ...     best_loc_da=loc_da, best_scale_da=scale_da
    ... )

    Warnings
    --------
    - Very high alpha can eliminate all predictors → monitor selected features.
    - Small training sets per cell/cluster may still lead to unstable results.
    - For heavy-tailed rainfall, consider log-transformation before modeling.
    - Use 'bayesian' optimizer for best quality; 'lassolars_cv' is fastest but least flexible.
    """


    def __init__(self, alpha_range=None, n_clusters=5, nb_cores=1, dist_method="nonparam",
                 hyperparam_optimizer="bayesian", n_trials=50, n_iter=50, mode="grid"):
        if alpha_range is None:
            alpha_range = np.logspace(-6, 2, 100)
        
        self.alpha_range = alpha_range
        self.n_clusters = n_clusters
        self.nb_cores = nb_cores
        self.dist_method = dist_method
        self.hyperparam_optimizer = hyperparam_optimizer
        self.n_trials = n_trials
        self.n_iter = n_iter
        self.mode = mode  # 'cluster' or 'grid'

    # ------------------ Hyperparameter Helpers ------------------

    def _optimize_hyperparameters_optuna(self, X, y, alpha_range):
        def objective(trial):
            alpha = trial.suggest_float('alpha', np.min(alpha_range), np.max(alpha_range), log=True)
            model = linear_model.LassoLars(alpha=alpha, max_iter=10000)
            # Lasso is prone to zeroing out everything; negative MSE is a good metric here
            scores = cross_val_score(model, X, y, cv=3, scoring='neg_mean_squared_error')
            return scores.mean()
        
        study = optuna.create_study(direction='maximize')
        study.optimize(objective, n_trials=self.n_trials, show_progress_bar=False)
        return study.best_params['alpha']

    def _optimize_hyperparameters_randomized(self, X, y, alpha_range):
        param_distributions = {'alpha': loguniform(np.min(alpha_range), np.max(alpha_range))}
        model = linear_model.LassoLars(max_iter=10000)
        random_search = RandomizedSearchCV(model, param_distributions, n_iter=self.n_iter, cv=3, scoring='neg_mean_squared_error', random_state=42)
        random_search.fit(X, y)
        return random_search.best_params_['alpha']

    # ------------------ Optimization Helpers ------------------

    def _optimize_single_cell(self, y_vec, X_mat):
        """Finds best alpha for a single pixel or cluster mean using chosen optimizer."""
        mask = np.isfinite(y_vec) & np.all(np.isfinite(X_mat), axis=-1)
        if np.sum(mask) < 10: 
            return np.nan
        
        X_c, y_c = X_mat[mask], y_vec[mask]
        
        try:
            if self.hyperparam_optimizer == 'bayesian':
                study = optuna.create_study(direction='maximize')
                study.optimize(lambda t: cross_val_score(
                    linear_model.LassoLars(
                        alpha=t.suggest_float('a', np.min(self.alpha_range), np.max(self.alpha_range), log=True),
                        max_iter=10000), X_c, y_c, cv=3, scoring='neg_mean_squared_error').mean(), 
                    n_trials=self.n_trials)
                return study.best_params['a']
            
            elif self.hyperparam_optimizer == 'random':
                param_dist = {'alpha': loguniform(np.min(self.alpha_range), np.max(self.alpha_range))}
                rs = RandomizedSearchCV(linear_model.LassoLars(max_iter=1000), param_dist, 
                                        n_iter=self.n_iter, cv=3, scoring='neg_mean_squared_error')
                rs.fit(X_c, y_c)
                return rs.best_params_['alpha']
            
            else:
                model_cv = linear_model.LassoLarsCV(cv=3, max_iter=1000, verbose=1, n_jobs=-1)
                model_cv.fit(X_c, y_c)
                return model_cv.alpha_
        except:
            return np.nan

    def compute_hyperparameters(self, predictand, predictor, clim_year_start, clim_year_end):
        """
        Compute spatially varying optimal LassoLars regularization parameter (alpha).

        Parameters
        ----------
        predictand : xarray.DataArray
            Target variable (typically rainfall)
            Expected dimensions: ('T', 'Y', 'X')
        predictor : xarray.DataArray
            Predictor fields
            Expected dimensions: ('T', 'Y', 'X', 'features') or ('T', 'features')
        clim_year_start : int
            Start year of the climatological standardization period
        clim_year_end : int
            End year of the climatological standardization period

        Returns
        -------
        alpha_map : xarray.DataArray
            Spatial field (Y, X) containing optimal alpha value(s)
            - Different value per cell when mode='grid'
            - Constant per cluster when mode='cluster'
        cluster_da : xarray.DataArray
            Cluster labels (integers) when mode='cluster'
            Dummy array filled with 1s when mode='grid' (for API consistency)

        Raises
        ------
        ValueError
            If mode is neither 'cluster' nor 'grid'
        RuntimeError
            If dask parallel computation fails (grid mode)

        Notes
        -----
        - Time series with fewer than 10 valid points are automatically skipped (return NaN)
        - All NaN values in input are properly masked
        - Progress messages are printed during long computations
        - LassoLarsCV does **not** require pre-defined alpha grid (uses full LARS path)
        """
        predictor['T'] = predictand['T']
        predictand_st = standardize_timeseries(predictand, clim_year_start, clim_year_end)

        if self.mode == "grid":
            print(f"LassoLars: Running Pixel-wise optimization on {self.nb_cores} cores...")
            client = Client(n_workers=self.nb_cores, threads_per_worker=1)
            
            chunk_y = int(np.ceil(len(predictand_st.Y) / self.nb_cores))
            chunk_x = int(np.ceil(len(predictand_st.X) / self.nb_cores))
            p_st_chunked = predictand_st.chunk({'Y': chunk_y, 'X': chunk_x})

            alpha_array = xr.apply_ufunc(
                self._optimize_single_cell, p_st_chunked, predictor,
                input_core_dims=[('T',), ('T', 'features')],
                output_core_dims=[()],
                vectorize=True, dask='parallelized', output_dtypes=[float]
            ).compute()
            client.close()
            
            cluster_da = xr.where(~np.isnan(alpha_array), 1, np.nan)
            return alpha_array, cluster_da

        else:
            print(f"LassoLars: Running Cluster-wise optimization ({self.n_clusters} clusters)...")
            kmeans = KMeans(n_clusters=self.n_clusters)
            predictand_dropna = predictand.to_dataframe().reset_index().dropna().drop(columns=['T'])
            variable_column = predictand_dropna.columns[2]
            predictand_dropna['cluster'] = kmeans.fit_predict(
                predictand_dropna[[variable_column]]
            )
            
            # Convert cluster assignments back into an xarray structure
            df_unique = predictand_dropna.drop_duplicates(subset=['Y', 'X'])
            dataset = df_unique.set_index(['Y', 'X']).to_xarray()
            mask = xr.where(~np.isnan(predictand.isel(T=0)), 1, np.nan)
            cluster_da = (dataset['cluster'] * mask)
                   
            # Align cluster array with the predictand array
            x1, x2 = xr.align(predictand_st, cluster_da, join="outer")
            
            # Identify unique cluster labels
            clusters = np.unique(x2)
            clusters = clusters[~np.isnan(clusters)]
            cluster_da = x2
            
            alpha_map = xr.full_like(cluster_da, np.nan, dtype=float)

            for clus in clusters:
                y_cluster = x1.where(x2 == clus).mean(dim=['Y','X'], skipna=True).dropna(dim='T')
                if len(y_cluster['T']) > 0:
                    best_alpha = self._optimize_single_cell(y_cluster.values, predictor.sel(T=y_cluster['T']).values)
                    alpha_map = alpha_map.where(cluster_da != clus, best_alpha)
            
            return alpha_map, cluster_da

    def fit_predict(self, x, y, x_test, y_test, alpha):
            if np.isnan(alpha): return np.array([np.nan, np.nan])
            model = linear_model.LassoLars(alpha=alpha, max_iter=100)
            mask = np.isfinite(y) & np.all(np.isfinite(x), axis=-1)
            if np.any(mask):
                model.fit(x[mask], y[mask])
                pred = model.predict(x_test.reshape(1, -1) if x_test.ndim == 1 else x_test)
                pred = np.maximum(pred, 0)
                return np.array([(y_test - pred).squeeze(), pred.squeeze()]).squeeze()
            return np.array([np.nan, np.nan])
    
    def compute_model(self, X_train, y_train, X_test, y_test, alpha=None, clim_year_start=None, clim_year_end=None):
        """
        Fits and predicts the LassoLars model using Dask for parallel execution.

        Parameters
        ----------
        X_train : xarray.DataArray
            Training predictor data (dims: T, features).
        y_train : xarray.DataArray
            Training response variable (dims: T, Y, X).
        X_test : xarray.DataArray
            Test predictor data (dims: features,) or broadcastable to (Y, X).
        y_test : xarray.DataArray
            Test response variable (dims: Y, X).
        alpha : xarray.DataArray
            Cluster-wise optimal alpha values (dims: Y, X).

        Returns
        -------
        xarray.DataArray
            dims (output=2, Y, X) => [error, prediction].
        """
        chunksize_x = int(np.round(len(y_train.get_index("X")) / self.nb_cores))
        chunksize_y = int(np.round(len(y_train.get_index("Y")) / self.nb_cores))
        
        X_train['T'] = y_train['T']
        y_train = y_train.transpose('T','Y','X')
        X_test = X_test.squeeze()
        y_test = y_test.drop_vars('T').squeeze().transpose('Y','X')
        y_train, alpha = xr.align(y_train, alpha)
        y_test, alpha = xr.align(y_test, alpha)
        
        if alpha is None:
            alpha, _ = self.compute_hyperparameters(
                 y_train, X_train, clim_year_start, clim_year_end
            )
            
        client = Client(n_workers=self.nb_cores, threads_per_worker=1)        
        result_da = xr.apply_ufunc(
            self.fit_predict,
            X_train,
            y_train.chunk({'Y': chunksize_y, 'X': chunksize_x}),
            X_test,
            y_test.chunk({'Y': chunksize_y, 'X': chunksize_x}),
            alpha.chunk({'Y': chunksize_y, 'X': chunksize_x}),
            input_core_dims=[
                ('T','features'), 
                ('T',),
                ('features',),
                (),
                ()
            ],
            vectorize=True,
            dask='parallelized',
            output_core_dims=[('output',)],
            output_dtypes=['float'],
            dask_gufunc_kwargs={'output_sizes': {'output': 2}},
        )
        result_ = result_da.compute()
        client.close()
        return result_.isel(output=1)
    
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
                    norm.ppf(0.32, loc=loc, scale=scale),
                    norm.ppf(0.67, loc=loc, scale=scale),
                )
            elif code == 2:
                return (
                    lognorm.ppf(0.32, s=shape, loc=loc, scale=scale),
                    lognorm.ppf(0.67, s=shape, loc=loc, scale=scale),
                )
            elif code == 3:
                return (
                    expon.ppf(0.32, loc=loc, scale=scale),
                    expon.ppf(0.67, loc=loc, scale=scale),
                )
            elif code == 4:
                return (
                    gamma.ppf(0.32, a=shape, loc=loc, scale=scale),
                    gamma.ppf(0.67, a=shape, loc=loc, scale=scale),
                )
            elif code == 5:
                return (
                    weibull_min.ppf(0.32, c=shape, loc=loc, scale=scale),
                    weibull_min.ppf(0.67, c=shape, loc=loc, scale=scale),
                )
            elif code == 6:
                # Note: Renamed 't_dist' to 't' for standard scipy.stats
                return (
                    t.ppf(0.32, df=shape, loc=loc, scale=scale),
                    t.ppf(0.67, df=shape, loc=loc, scale=scale),
                )
            elif code == 7:
                # Poisson: poisson.ppf(q, mu, loc=0)
                # ASSUMPTION: 'mu' (mean) is passed as 'shape'
                #             'loc' is passed as 'loc'
                #             'scale' is unused
                return (
                    poisson.ppf(0.32, mu=shape, loc=loc),
                    poisson.ppf(0.67, mu=shape, loc=loc),
                )
            elif code == 8:
                # Negative Binomial: nbinom.ppf(q, n, p, loc=0)
                # ASSUMPTION: 'n' (successes) is passed as 'shape'
                #             'p' (probability) is passed as 'scale'
                #             'loc' is passed as 'loc'
                return (
                    nbinom.ppf(0.32, n=shape, p=scale, loc=loc),
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
        terciles_emp = clim.quantile([0.32, 0.67], dim="T")
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

    # ------------------- FORECAST METHOD -------------------

    def forecast(self, Predictant, clim_year_start, clim_year_end, Predictor, hindcast_det, Predictor_for_year, alpha, best_code_da=None, best_shape_da=None, best_loc_da=None, best_scale_da=None):
        """
        Generates a forecast for a single time step using LassoLars with alpha,
        then computes tercile probabilities based on the chosen distribution method.

        Parameters
        ----------
        Predictant : xarray.DataArray
            Observed data (T, Y, X).
        clim_year_start : int
            Start year for the climatological reference period.
        clim_year_end : int
            End year for the climatological reference period.
        Predictor : xarray.DataArray
            Historical predictor data (T, features).
        hindcast_det : xarray.DataArray
            Historical deterministic forecast with dims (output=2, T, Y, X).
        Predictor_for_year : xarray.DataArray
            Single-year predictor data (features,) or shape (1, features).
        alpha : xarray.DataArray
            Spatial map of alpha values (Y, X) for LassoLars.

        Returns
        -------
        result_ : xarray.DataArray
            dims (output=2, Y, X) => [error, prediction].
            For a true forecast scenario, the error is typically NaN.
        hindcast_prob : xarray.DataArray
            dims (probability=3, Y, X) => [PB, PN, PA] for tercile categories.
        """
        # 1) Provide a dummy y_test with NaN so fit_predict returns [NaN, forecast]
        y_test_dummy = xr.full_like(Predictant.isel(T=0), np.nan)

        # 2) Chunk sizes
        chunksize_x = int(np.round(len(Predictant.get_index("X")) / self.nb_cores))
        chunksize_y = int(np.round(len(Predictant.get_index("Y")) / self.nb_cores))
        
        Predictor['T'] = Predictant['T']
        Predictant = Predictant.transpose('T','Y','X')
        Predictor_for_year_ = Predictor_for_year.squeeze()
        Predictant_st = standardize_timeseries(Predictant, clim_year_start, clim_year_end)
        # Align alpha with the domain
        Predictant_st, alpha = xr.align(Predictant_st, alpha, join='outer')

        # 3) Parallel fit+predict => (2, Y, X)
        client = Client(n_workers=self.nb_cores, threads_per_worker=1)
        result_da = xr.apply_ufunc(
            self.fit_predict,
            Predictor,
            Predictant_st.chunk({'Y': chunksize_y, 'X': chunksize_x}),
            Predictor_for_year_,
            y_test_dummy.chunk({'Y': chunksize_y, 'X': chunksize_x}),   # dummy test
            alpha.chunk({'Y': chunksize_y, 'X': chunksize_x}),
            input_core_dims=[
                ('T','features'),  # x
                ('T',),           # y
                ('features',),    # x_test
                (),               # y_test
                ()                # alpha
            ],
            vectorize=True,
            dask='parallelized',
            output_core_dims=[('output',)],  # => [error, prediction]
            output_dtypes=['float'],
            dask_gufunc_kwargs={'output_sizes': {'output': 2}}
        )
        result_ = result_da.compute()
        client.close()
        result_ = result_.isel(output=1)
        result_ = reverse_standardize(result_, Predictant, clim_year_start, clim_year_end) 
        # result_ => dims (output=2, Y, X). 
        # For a real future forecast, "error" is NaN, "prediction" is the forecast.

        # 2) Compute thresholds T1, T2 from climatology
        index_start = Predictant.get_index("T").get_loc(str(clim_year_start)).start
        index_end   = Predictant.get_index("T").get_loc(str(clim_year_end)).stop
        rainfall_for_tercile = Predictant.isel(T=slice(index_start, index_end))
        terciles = rainfall_for_tercile.quantile([0.32, 0.67], dim='T')
        T1_emp = terciles.isel(quantile=0).drop_vars('quantile')
        T2_emp = terciles.isel(quantile=1).drop_vars('quantile')
        error_variance = (Predictant - hindcast_det).var(dim='T')
        
        # Expand single prediction to T=1 so probability methods can handle it
        forecast_expanded = result_.expand_dims(
            T=[pd.Timestamp(Predictor_for_year.coords['T'].values[0]).to_pydatetime()]
        )
        year = Predictor_for_year.coords['T'].values[0].astype('datetime64[Y]').astype(int) + 1970
        # year = Predictor_for_year.coords['T'].values.astype('datetime64[Y]').astype(int)[0] + 1970  
        T_value_1 = Predictant.isel(T=0).coords['T'].values  # Get the datetime64 value from da1
        month_1 = T_value_1.astype('datetime64[M]').astype(int) % 12 + 1  # Extract month
        new_T_value = np.datetime64(f"{year}-{month_1:02d}-{1:02d}")
        
        forecast_expanded = forecast_expanded.assign_coords(T=xr.DataArray([new_T_value], dims=["T"]))
        forecast_expanded['T'] = forecast_expanded['T'].astype('datetime64[ns]')

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
                forecast_expanded,
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
                forecast_expanded,
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
        return forecast_expanded, forecast_prob.transpose('probability', 'T', 'Y', 'X')