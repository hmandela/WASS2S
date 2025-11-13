"""
WAS_TransformData: Skewness Analysis and Transformation for Geospatial Time-Series

This module provides the `WAS_TransformData` class to analyze skewness, apply
transformations, fit distributions, and visualize geospatial time-series data with
dimensions (T, Y, X) representing time, latitude, and longitude, respectively.
"""

import xarray as xr
import numpy as np
from scipy.stats import skew, boxcox
from sklearn.cluster import KMeans
from sklearn.preprocessing import PowerTransformer
from fitter import Fitter
import pandas as pd
import matplotlib.pyplot as plt
import cartopy.crs as ccrs
import cartopy.feature as cfeature
from matplotlib.colors import ListedColormap, BoundaryNorm
from scipy.stats import (norm, lognorm, expon, gamma as gamma_dist, weibull_min, t as t_dist, poisson, nbinom,)

def inv_boxcox(y, lmbda):
    """
    Inverse Box-Cox transformation for SciPy 1.11.3 compatibility.

    Parameters
    ----------
    y : array_like
        Transformed data.
    lmbda : float
        Box-Cox lambda parameter.

    Returns
    -------
    x : ndarray
        Original data before Box-Cox transformation.

    Notes
    -----
    Implements the inverse of the Box-Cox transformation manually
    """
    if abs(lmbda) < 1e-6:
        return np.exp(y)
    return (y * lmbda + 1) ** (1 / lmbda)

class WAS_TransformData:
    """
    Manage skewness analysis, data transformation, distribution fitting, and visualization
    for geospatial time-series data.

    Parameters
    ----------
    data : xarray.DataArray
        Input data with dimensions (T, Y, X) for time, latitude, and longitude.
    distribution_map : dict, optional
        Mapping of distribution names to numeric codes. Default is:
        {'norm': 1, 'lognorm': 2, 'expon': 3, 'gamma': 4, 'weibull_min': 5}.
    n_clusters : int, optional
        Number of clusters for KMeans in distribution fitting. Default is 5.

    Attributes
    ----------
    data : xarray.DataArray
        Input geospatial time-series data.
    distribution_map : dict
        Mapping of distribution names to codes.
    n_clusters : int
        Number of clusters for KMeans.
    transformed_data : xarray.DataArray or None
        Transformed data after applying transformations.
    transform_methods : xarray.DataArray or None
        Transformation methods applied per grid cell.
    transform_params : xarray.DataArray or None
        Parameters for parametric transformations (e.g., Box-Cox lambda).
    skewness_ds : xarray.Dataset or None
        Skewness analysis results.
    handle_ds : xarray.Dataset or None
        Skewness handling recommendations.

    Methods
    -------
    detect_skewness()
        Compute and classify skewness per grid cell.
    handle_skewness()
        Recommend transformations based on skewness.
    apply_transformation(method=None)
        Apply transformations to data.
    inverse_transform()
        Reverse transformations to recover original data.
    find_best_distribution_grid(use_transformed=False)
        Fit distributions to data using KMeans clustering.
    plot_best_fit_map(data_array, map_dict, output_file='map.png', ...)
        Plot categorical map of distributions or skewness classes.

    Raises
    ------
    ValueError
        If `data` is not an xarray.DataArray or lacks required dimensions.
    """

    def __init__(self, data, distribution_map=None, n_clusters=1000):
        if not isinstance(data, xr.DataArray):
            raise ValueError("`data` must be an xarray.DataArray")
        if not all(dim in data.dims for dim in ('T', 'Y', 'X')):
            raise ValueError("`data` must have dimensions ('T', 'Y', 'X')")

        self.data = data
        self.distribution_map = distribution_map or {
            'norm': 1,
            'lognorm': 2,
            'expon': 3,
            'gamma': 4,
            'weibull_min': 5,
            "t_dist": 6,
            "poisson":7,
            "nbinom":8
        
            
        }
        self.n_clusters = n_clusters
        self.transformed_data = None
        self.transform_methods = None
        self.transform_params = None
        self.skewness_ds = None
        self.handle_ds = None

    @staticmethod
    def _safe_boxcox(arr1d):
        """
        Apply Box-Cox transformation while handling NaNs.

        Parameters
        ----------
        arr1d : array_like
            1D array of data to transform.

        Returns
        -------
        transformed : ndarray
            Transformed array, same shape as input, with NaNs preserved.
        lmbda : float
            Box-Cox lambda parameter.

        Raises
        ------
        ValueError
            If fewer than 2 non-NaN values or if data is not strictly positive.
        """
        out = arr1d.copy()
        valid = ~np.isnan(arr1d)
        if valid.sum() < 2:
            raise ValueError("Need at least two non-NaN values for Box-Cox")
        if not np.all(arr1d[valid] > 0):
            raise ValueError("Box-Cox requires strictly positive data")
        out[valid], lmbda = boxcox(arr1d[valid])
        return out, lmbda

    def detect_skewness(self):
        """
        Compute and classify skewness for each grid cell.

        Returns
        -------
        skewness_ds : xarray.Dataset
            Dataset with variables 'skewness' (float) and 'skewness_class' (str).
            Skewness classes: 'symmetric', 'moderate_positive', 'moderate_negative',
            'high_positive', 'high_negative', 'invalid'.
        summary : dict
            Dictionary with 'class_counts' mapping skewness classes to grid cell counts.

        Notes
        -----
        Skewness is computed using `scipy.stats.skew` with `nan_policy='omit'`.
        Classification thresholds:
        - Symmetric: -0.5 ≤ skewness ≤ 0.5
        - Moderate positive: 0.5 < skewness ≤ 1
        - Moderate negative: -1 ≤ skewness < -0.5
        - High positive: skewness > 1
        - High negative: skewness < -1
        - Invalid: insufficient data (< 3 non-NaN values).
        """
        def _compute(precip):
            precip = np.asarray(precip)
            valid = ~np.isnan(precip)
            if valid.sum() < 3:
                return np.nan, 'invalid'
            sk = skew(precip[valid], axis=0, nan_policy='omit')
            if np.isnan(sk):
                cls = 'invalid'
            elif -0.5 <= sk <= 0.5:
                cls = 'symmetric'
            elif 0.5 < sk <= 1:
                cls = 'moderate_positive'
            elif -1 <= sk < -0.5:
                cls = 'moderate_negative'
            elif sk > 1:
                cls = 'high_positive'
            else:
                cls = 'high_negative'
            return sk, cls

        res = xr.apply_ufunc(
            _compute,
            self.data,
            input_core_dims=[['T']],
            output_core_dims=[[], []],
            vectorize=True,
            dask='parallelized',
            output_dtypes=[float, str]
        )

        self.skewness_ds = xr.Dataset(
            {
                'skewness': (('Y', 'X'), res[0].data),
                'skewness_class': (('Y', 'X'), res[1].data)
            },
            coords={'Y': self.data.Y, 'X': self.data.X}
        )

        counts = pd.Series(self.skewness_ds['skewness_class'].values.ravel()).value_counts().to_dict()
        return self.skewness_ds, {'class_counts': counts}

    def handle_skewness(self):
        """
        Recommend transformations based on skewness and data properties.

        Returns
        -------
        handle_ds : xarray.Dataset
            Dataset with variables 'skewness', 'skewness_class', and 'recommended_methods'
            (semicolon-separated string of transformation methods).
        summary : dict
            Dictionary with 'general_recommendations' mapping skewness classes to advice.

        Raises
        ------
        ValueError
            If `detect_skewness` has not been called.

        Notes
        -----
        Recommendations consider data properties (e.g., zeros, negatives) and skewness class.
        Example methods: 'log', 'square_root', 'box_cox', 'yeo_johnson', 'clipping', 'binning'.
        """
        if self.skewness_ds is None:
            raise ValueError("Run detect_skewness() first")

        def _suggest(precip, sk_class):
            if sk_class == 'invalid':
                return 'none'
            precip = np.asarray(precip)
            valid = precip[~np.isnan(precip)]
            all_pos = np.all(valid > 0)
            has_zeros = np.any(valid == 0)
            methods = []
            if sk_class in ('moderate_positive', 'high_positive'):
                if all_pos and not has_zeros:
                    methods += ['log', 'square_root', 'box_cox']
                elif all_pos:
                    methods += ['square_root', 'box_cox']
                methods += ['yeo_johnson', 'clipping', 'binning']
            elif sk_class in ('moderate_negative', 'high_negative'):
                if all_pos and not has_zeros:
                    methods += ['reflect_log']
                elif all_pos:
                    methods += ['reflect_square_root']
                methods += ['reflect_yeo_johnson', 'clipping', 'binning']
            else:
                methods.append('none')
            return ';'.join(methods)

        recommended = xr.apply_ufunc(
            _suggest,
            self.data,
            self.skewness_ds['skewness_class'],
            input_core_dims=[['T'], []],
            output_core_dims=[[]],
            vectorize=True,
            dask='parallelized',
            output_dtypes=[str]
        )

        self.handle_ds = xr.Dataset(
            {
                'skewness': self.skewness_ds['skewness'],
                'skewness_class': self.skewness_ds['skewness_class'],
                'recommended_methods': (('Y', 'X'), recommended.data)
            },
            coords={'Y': self.data.Y, 'X': self.data.X}
        )

        general = {
            'symmetric': 'No transformation needed.',
            'moderate_positive': (
                'Consider square root or Yeo-Johnson; log or Box-Cox if no zeros; '
                'clip or bin outliers.'
            ),
            'high_positive': (
                'Strongly consider log (no zeros), Box-Cox (positive), or Yeo-Johnson; '
                'clip or bin extremes.'
            ),
            'moderate_negative': (
                'Reflect and apply square root or Yeo-Johnson; clip or bin outliers.'
            ),
            'high_negative': (
                'Reflect and apply log (no zeros), Box-Cox, or Yeo-Johnson; '
                'clip or bin extremes.'
            ),
            'invalid': 'Insufficient valid data for skewness calculation.'
        }

        return self.handle_ds, {'general_recommendations': general}

    def apply_transformation(self, method=None):
        """
        Apply transformations to reduce skewness in the data.

        Parameters
        ----------
        method : str or xarray.DataArray, optional
            Transformation method to apply. Options:
            - None: Use first recommended method per grid cell from `handle_skewness`.
            - str: Apply the same method to all grid cells (e.g., 'log', 'box_cox').
            - xarray.DataArray: Specify method per grid cell with dimensions (Y, X).
            Default is None.

        Returns
        -------
        transformed_data : xarray.DataArray
            Transformed data with same shape as input.

        Raises
        ------
        ValueError
            If `method` is None and `handle_skewness` has not been called.

        Notes
        -----
        Supported methods: 'log', 'square_root', 'box_cox', 'yeo_johnson',
        'reflect_log', 'reflect_square_root', 'reflect_yeo_johnson', 'clipping', 'binning'.
        Transformations are skipped for invalid data or methods, with warnings printed.
        """
        if method is None and self.handle_ds is None:
            raise ValueError("Run handle_skewness() first or specify `method`")

        if method is None:
            def extract_first_method(x):
                if isinstance(x, str) and x and x != 'none':
                    return x.split(';')[0]
                return 'none'
            method = xr.apply_ufunc(
                extract_first_method,
                self.handle_ds['recommended_methods'],
                vectorize=True,
                output_dtypes=[str]
            )

        self.transformed_data = self.data.copy()
        self.transform_methods = method if isinstance(method, xr.DataArray) else xr.DataArray(
            np.full((self.data.sizes['Y'], self.data.sizes['X']), method),
            coords={'Y': self.data.Y, 'X': self.data.X},
            dims=('Y', 'X')
        )
        self.transform_params = xr.DataArray(
            np.empty((self.data.sizes['Y'], self.data.sizes['X']), dtype=object),
            coords={'Y': self.data.Y, 'X': self.data.X},
            dims=('Y', 'X')
        )

        for iy in range(self.data.sizes['Y']):
            for ix in range(self.data.sizes['X']):
                m = self.transform_methods[iy, ix].item()
                if m == 'none' or np.all(np.isnan(self.data[:, iy, ix])):
                    continue
                cell = self.data[:, iy, ix].values
                valid = cell[~np.isnan(cell)]
                if len(valid) < 2:
                    continue

                if m == 'log':
                    if np.any(valid <= 0):
                        print(f"Skip log at Y={iy}, X={ix}: non-positive values")
                        continue
                    self.transformed_data[:, iy, ix] = np.log(cell)
                elif m == 'square_root':
                    if np.any(valid < 0):
                        print(f"Skip square_root at Y={iy}, X={ix}: negative values")
                        continue
                    self.transformed_data[:, iy, ix] = np.sqrt(cell)
                elif m == 'box_cox':
                    try:
                        transformed, lam = self._safe_boxcox(cell)
                        self.transformed_data[:, iy, ix] = transformed
                        self.transform_params[iy, ix] = {'lambda': lam}
                    except ValueError as err:
                        print(f"Skip Box-Cox at Y={iy}, X={ix}: {err}")
                        continue
                elif m == 'yeo_johnson':
                    pt = PowerTransformer(method='yeo-johnson')
                    transformed = pt.fit_transform(cell.reshape(-1, 1)).ravel()
                    self.transformed_data[:, iy, ix] = transformed
                    self.transform_params[iy, ix] = {'transformer': pt}
                elif m == 'reflect_log':
                    cell_ref = -cell
                    if np.any(cell_ref <= 0):
                        print(f"Skip reflect_log at Y={iy}, X={ix}: non-positive values")
                        continue
                    self.transformed_data[:, iy, ix] = np.log(cell_ref)
                elif m == 'reflect_square_root':
                    cell_ref = -cell
                    if np.any(cell_ref < 0):
                        print(f"Skip reflect_square_root at Y={iy}, X={ix}: negative values")
                        continue
                    self.transformed_data[:, iy, ix] = np.sqrt(cell_ref)
                elif m == 'reflect_yeo_johnson':
                    pt = PowerTransformer(method='yeo-johnson')
                    transformed = pt.fit_transform((-cell).reshape(-1, 1)).ravel()
                    self.transformed_data[:, iy, ix] = transformed
                    self.transform_params[iy, ix] = {'transformer': pt}
                elif m in ('clipping', 'binning'):
                    self.transformed_data[:, iy, ix] = cell
                else:
                    pass
                    #print(f"Warning: unknown method '{m}' at Y={iy}, X={ix}")

        return self.transformed_data

    def inverse_transform(self):
        """
        Reverse transformations to recover original data scale.

        Returns
        -------
        inverse_data : xarray.DataArray
            Data in original scale with same shape as input.

        Raises
        ------
        ValueError
            If no transformation has been applied or required parameters are missing.

        Notes
        -----
        Non-invertible methods ('clipping', 'binning') return unchanged data with a warning.
        """
        if self.transformed_data is None or self.transform_methods is None:
            raise ValueError("No transformation applied. Run apply_transformation() first")

        def _inv(vec, method, params):
            if method in ('none', None) or (isinstance(method, float) and np.isnan(method)):
                return vec
            if method in ('clipping', 'binning'):
                print(f"Warning: '{method}' is not invertible")
                return vec
            if method == 'log':
                return np.exp(vec)
            if method == 'square_root':
                return vec ** 2
            if method == 'box_cox':
                lam = params.get('lambda') if params else None
                if lam is None:
                    raise ValueError("Missing lambda for Box-Cox inversion")
                return inv_boxcox(vec, lam)
            if method == 'yeo_johnson':
                tr = params.get('transformer') if params else None
                if tr is None:
                    raise ValueError("Missing transformer for Yeo-Johnson inversion")
                return tr.inverse_transform(vec.reshape(-1, 1)).ravel()
            if method.startswith('reflect_'):
                if method == 'reflect_log':
                    temp = np.exp(vec)
                elif method == 'reflect_square_root':
                    temp = vec ** 2
                else:  # reflect_yeo_johnson
                    tr = params.get('transformer') if params else None
                    if tr is None:
                        raise ValueError("Missing transformer for reflect_yeo_johnson")
                    temp = tr.inverse_transform(vec.reshape(-1, 1)).ravel()
                return -temp
            raise ValueError(f"Unknown method '{method}'")

        return xr.apply_ufunc(
            _inv,
            self.transformed_data,
            self.transform_methods,
            self.transform_params,
            input_core_dims=[['T'], [], []],
            output_core_dims=[['T']],
            vectorize=True,
            dask='parallelized',
            output_dtypes=[float]
        )




    def fit_best_distribution_grid_onlycluster(self, use_transformed=False):
        """
        Fit best distributions by homogeneous zones (clusters) and map to each grid cell.
    
        Clustering:
            - Done on simple distributional features per grid cell (mean, std over T).
            - Produces n_clusters homogeneous zones.
    
        For each cluster:
            - Pool all time series values from its grid cells.
            - For each candidate distribution in self.distribution_map:
                * Fit parameters by MLE (SciPy).
                * Compute AIC = 2k - 2 logL on that cluster sample.
            - Select distribution with minimum AIC.
            - Assign its (code, shape, loc, scale) to all grid cells in that cluster.
    
        Supports (if present in distribution_map):
            - 'norm'        -> code 1
            - 'lognorm'     -> code 2
            - 'expon'       -> code 3
            - 'gamma'       -> code 4
            - 'weibull_min' -> code 5
            - 't'           -> code 6
            - 'poisson'     -> code 7
            - 'nbinom'      -> code 8
    
        Parameters
        ----------
        use_transformed : bool, optional
            If True, use self.transformed_data; otherwise use original self.data.
    
        Returns
        -------
        best_code  : xarray.DataArray, shape (Y, X)
            Code of best distribution per grid cell.
        best_shape : xarray.DataArray, shape (Y, X)
            Primary shape param (e.g., 'a' in gamma, 'mu' in poisson, 'n' in nbinom).
        best_loc   : xarray.DataArray, shape (Y, X)
            Location parameter.
        best_scale : xarray.DataArray, shape (Y, X)
            Scale parameter (e.g., 'p' in nbinom).
    
        Notes
        -----
        - Continuous (precip-like):
            * 'lognorm', 'gamma', 'weibull_min', 'expon' are fitted on non-negative values with loc=0.
        - Continuous (unbounded):
            * 'norm' and 't' are fitted on all finite values (no non-negativity constraint).
        - Discrete (counts):
            * 'poisson', 'nbinom' are fitted on non-negative INTEGER values with loc=0.
        - Requires enough valid data per cluster; if not, affected cells get NaNs.
        """
        import numpy as np
        import xarray as xr
        from sklearn.cluster import KMeans
        from scipy.stats import (
            norm,
            lognorm,
            expon,
            gamma as gamma_dist,
            weibull_min,
            t as t_dist,
            poisson,
            nbinom,
        )
    
        # -------- Select working data --------
        data = self.transformed_data if (use_transformed and self.transformed_data is not None) else self.data
    
        if not isinstance(data, xr.DataArray):
            raise ValueError("Internal error: data must be an xarray.DataArray")
    
        if not all(dim in data.dims for dim in ("T", "Y", "X")):
            raise ValueError("`data` must have dimensions ('T', 'Y', 'X')")
    
        Y = data.sizes["Y"]
        X = data.sizes["X"]
        coords = {"Y": data.Y, "X": data.X}
    
        # Map distribution names to scipy objects
        dist_objs = {
            "norm": norm,
            "lognorm": lognorm,
            "expon": expon,
            "gamma": gamma_dist,
            "weibull_min": weibull_min,
            "t": t_dist,
            "poisson": poisson,
            "nbinom": nbinom,
        }
    
        # -------- 1. Build clustering features (mean, std) per grid cell --------
        mean_da = data.mean("T", skipna=True)
        std_da = data.std("T", skipna=True)
    
        feat_ds = xr.Dataset({"mean": mean_da, "std": std_da})
        feat_df = feat_ds.to_dataframe().dropna()  # index: (Y, X), cols: mean, std
    
        if feat_df.shape[0] < self.n_clusters:
            # Not enough valid cells for the requested clusters
            print("Warning: insufficient valid grid cells for clustering; returning NaNs.")
            nan_arr = np.full((Y, X), np.nan, float)
            return (
                xr.DataArray(nan_arr, coords=coords, dims=("Y", "X")),
                xr.DataArray(nan_arr, coords=coords, dims=("Y", "X")),
                xr.DataArray(nan_arr, coords=coords, dims=("Y", "X")),
                xr.DataArray(nan_arr, coords=coords, dims=("Y", "X")),
            )
    
        # Run KMeans on [mean, std]
        features = feat_df[["mean", "std"]].values
        kmeans = KMeans(n_clusters=self.n_clusters, random_state=42, n_init=10)
        feat_df["cluster"] = kmeans.fit_predict(features)
    
        # -------- 2. Map cluster labels back to full (Y, X) grid --------
        cluster_da = (
            feat_df["cluster"]
            .to_xarray()
            .reindex(Y=data.Y, X=data.X)
        )
        # Note: Removed the .where(valid_mask) as it was redundant with .dropna()
    
        # -------- 3. For each cluster, fit best distribution on pooled values --------
        best_params_by_cluster = {}  # cl -> (code, shape, loc, scale)
    
        unique_clusters = np.unique(cluster_da.values)
        unique_clusters = unique_clusters[np.isfinite(unique_clusters)]
    
        for cl_val in unique_clusters:
            cl = int(cl_val)
    
            # Extract pooled values for this cluster
            mask_cl = (cluster_da == cl)
            cl_vals = data.where(mask_cl).values
            cl_vals = cl_vals[np.isfinite(cl_vals)]
    
            if cl_vals.size < 30:
                best_params_by_cluster[cl] = None
                continue
    
            # Create specific subsets for different distribution families
            cl_vals_pos = cl_vals[cl_vals >= 0]
            cl_vals_int = cl_vals_pos[(cl_vals_pos == np.floor(cl_vals_pos))]
    
            best_aic = np.inf
            best_choice = None # Will store (code, (shape, loc, scale))
    
            for name, code in self.distribution_map.items():
                if name not in dist_objs:
                    continue
    
                dist = dist_objs[name]
                is_discrete = False
                
                # Choose appropriate sample for this distribution
                if name in ("poisson", "nbinom"):
                    sample = cl_vals_int
                    is_discrete = True
                elif name in ("lognorm", "gamma", "weibull_min", "expon"):
                    sample = cl_vals_pos
                else:
                    # 'norm' and 't' work on full sample
                    sample = cl_vals
    
                if sample.size < 30:
                    continue
    
                try:
                    # --- Fit parameters and get k (number of estimated params) ---
                    if name == "poisson":
                        # Manual MLE fit for Poisson (mu=mean), loc is fixed
                        mu = sample.mean()
                        loc = 0
                        params = (mu, loc)
                        k = 1  # Only mu is estimated
                    
                    elif name in ("lognorm", "gamma", "weibull_min", "expon", "nbinom"):
                        # Fit with fixed location at 0
                        params = dist.fit(sample, floc=0)
                        k = len(params) - 1  # loc is fixed, not estimated
                    
                    else:
                        # 'norm' and 't' (and any others with free loc)
                        params = dist.fit(sample)
                        k = len(params) # All params (loc, scale, shapes) are estimated
    
                    # --- Calculate Log-Likelihood ---
                    if is_discrete:
                        if name == "poisson":
                            logL = np.sum(dist.logpmf(sample, mu=params[0], loc=params[1]))
                        else: # nbinom
                            logL = np.sum(dist.logpmf(sample, *params))
                    else:
                        logL = np.sum(dist.logpdf(sample, *params))
    
                    if not np.isfinite(logL):
                        continue
                        
                    aic = 2 * k - 2 * logL
    
                    if aic < best_aic:
                        best_aic = aic
                        # --- Standardize parameters to (shape, loc, scale) ---
                        if name == "poisson":
                            # params = (mu, loc)
                            best_choice = (code, (params[0], params[1], np.nan))
                        elif name == "nbinom":
                            # params = (n, p, loc)
                            # Store as (shape=n, loc=loc, scale=p)
                            best_choice = (code, (params[0], params[2], params[1]))
                        elif name in ("norm", "expon"):
                            # params = (loc, scale)
                            # Store as (shape=nan, loc=loc, scale=scale)
                            best_choice = (code, (np.nan, params[0], params[1]))
                        else: 
                            # 't', 'gamma', 'lognorm', 'weibull_min'
                            # params = (shape, loc, scale)
                            best_choice = (code, (params[0], params[1], params[2]))
    
                except Exception:
                    continue # Fitting failed
    
            if best_choice is None:
                best_params_by_cluster[cl] = None
            else:
                code, params_tuple = best_choice
                # params_tuple is already (shape, loc, scale)
                best_params_by_cluster[cl] = (code, params_tuple[0], params_tuple[1], params_tuple[2])
    
        # -------- 4. Broadcast cluster-level params to each grid cell (Vectorized) --------
        
        # Create empty arrays with the correct coordinates and NaNs
        best_code_da = xr.DataArray(np.full((Y, X), np.nan, dtype=float), coords=coords, dims=("Y", "X"))
        best_shape_da = best_code_da.copy()
        best_loc_da = best_code_da.copy()
        best_scale_da = best_code_da.copy()
    
        # Loop over the few clusters, not the millions of pixels
        for cl, params in best_params_by_cluster.items():
            if params is None:
                continue # Leave these cells as NaN
    
            # Create a boolean mask for all cells belonging to this cluster
            mask = (cluster_da == cl)
            
            # Unpack the parameters
            code, shape, loc, scale = params
            
            # "Paint" the values onto the grid where the mask is True
            best_code_da = best_code_da.where(~mask, code)
            best_shape_da = best_shape_da.where(~mask, shape)
            best_loc_da = best_loc_da.where(~mask, loc)
            best_scale_da = best_scale_da.where(~mask, scale)
    
        return best_code_da, best_shape_da, best_loc_da, best_scale_da, cluster_da
        
   
    
    def fit_best_distribution_grid_onlygrid(self, use_transformed=False):
        """
        Fit candidate distributions per grid cell and select best by AIC.
    
        Parameters
        ----------
        use_transformed : bool, optional
            If True, use self.transformed_data; otherwise use original data.
    
        Returns
        -------
        best_code  : xarray.DataArray  (Y, X)
            Code of best distribution per grid (per self.distribution_map).
        best_shape : xarray.DataArray  (Y, X)
        best_loc   : xarray.DataArray  (Y, X)
        best_scale : xarray.DataArray  (Y, X)
    
        Notes
        -----
        - Fits are done independently per grid cell (no clustering).
        - For precip-like variables:
            * 'lognorm', 'gamma', 'weibull_min', 'expon' are fitted on positive values with loc=0.
            * 'norm' and 't' are fitted on all finite values.
        - AIC = 2k - 2 ln(L) is used for model selection.
        """
        import numpy as np
        import xarray as xr
        from scipy.stats import norm, lognorm, expon, gamma as gamma_dist, weibull_min, t as t_dist
    
        data = self.transformed_data if (use_transformed and self.transformed_data is not None) else self.data
    
        if not isinstance(data, xr.DataArray):
            raise ValueError("`data` must be an xarray.DataArray")
        if not all(dim in data.dims for dim in ("T", "Y", "X")):
            raise ValueError("`data` must have dimensions ('T', 'Y', 'X')")
    
        # Map distribution names to scipy.stats objects
        dist_objs = {
            "norm": norm,
            "lognorm": lognorm,
            "expon": expon,
            "gamma": gamma_dist,
            "weibull_min": weibull_min,
            "t": t_dist,
        }
    
        Y = data.sizes["Y"]
        X = data.sizes["X"]
    
        best_code = np.full((Y, X), np.nan, dtype=float)
        best_shape = np.full((Y, X), np.nan, dtype=float)
        best_loc = np.full((Y, X), np.nan, dtype=float)
        best_scale = np.full((Y, X), np.nan, dtype=float)
    
        for iy in range(Y):
            for ix in range(X):
                vals = data[:, iy, ix].values
                vals = vals[np.isfinite(vals)]
                if vals.size < 10:
                    continue
    
                # Positive subset for positive-support distributions
                vals_pos = vals[vals > 0]
    
                best_aic = np.inf
                best = None
    
                for name, code in self.distribution_map.items():
                    if name not in dist_objs:
                        continue
    
                    dist = dist_objs[name]
    
                    # Choose sample depending on support
                    if name in ("lognorm", "gamma", "weibull_min", "expon"):
                        sample = vals_pos
                        if sample.size < 10:
                            continue
                    else:  # 'norm', 't', or any other real-line distribution
                        sample = vals
                        if sample.size < 10:
                            continue
    
                    try:
                        # Fit parameters
                        if name in ("lognorm", "gamma", "weibull_min", "expon"):
                            params = dist.fit(sample, floc=0)
                        else:
                            # norm, t: free loc/scale; t has (df, loc, scale)
                            params = dist.fit(sample)
    
                        # Compute AIC
                        k = len(params)
                        logL = np.sum(dist.logpdf(sample, *params))
                        aic = 2 * k - 2.0 * logL
    
                        if np.isfinite(aic) and aic < best_aic:
                            best_aic = aic
                            best = (code, params)
                    except Exception:
                        # Skip distributions that fail to fit at this grid
                        continue
    
                if best is not None:
                    code, params = best
                    # Normalise to (shape, loc, scale)
                    if len(params) == 2:
                        # e.g. norm, expon when not forcing shape
                        shape, loc, scale = (np.nan, params[0], params[1])
                    else:
                        # e.g. gamma (k, loc, scale), t (df, loc, scale), etc.
                        shape, loc, scale = params[0], params[1], params[2]
    
                    best_code[iy, ix] = code
                    best_shape[iy, ix] = shape
                    best_loc[iy, ix] = loc
                    best_scale[iy, ix] = scale
    
        coords = {"Y": data.Y, "X": data.X}
        best_code_da = xr.DataArray(best_code, coords=coords, dims=("Y", "X"))
        best_shape_da = xr.DataArray(best_shape, coords=coords, dims=("Y", "X"))
        best_loc_da = xr.DataArray(best_loc, coords=coords, dims=("Y", "X"))
        best_scale_da = xr.DataArray(best_scale, coords=coords, dims=("Y", "X"))
    
        return best_code_da, best_shape_da, best_loc_da, best_scale_da


    def fit_best_distribution_grid_two_options(self, use_transformed=False, mode="cluster"):
        """
        Fit best distributions either by:
          - homogeneous zones (mode='cluster'), or
          - per grid cell (mode='grid').
    
        Uses AIC for model selection and supports:
          'norm', 'lognorm', 'expon', 'gamma', 'weibull_min', 't', 'poisson', 'nbinom'
        (if present in self.distribution_map).
    
        Parameters
        ----------
        use_transformed : bool, optional
            If True, use self.transformed_data, else self.data.
        mode : {'cluster', 'grid'}, optional
            'cluster' : KMeans on (mean,std) → one distribution per cluster.
            'grid'    : independent fit at each (Y,X) via xr.apply_ufunc.
    
        Returns
        -------
        best_code  : xarray.DataArray (Y, X)
        best_shape : xarray.DataArray (Y, X)
        best_loc   : xarray.DataArray (Y, X)
        best_scale : xarray.DataArray (Y, X)
        cluster_da : xarray.DataArray (Y, X)
            Cluster labels for mode='cluster'; all-NaN for mode='grid'.
        """
    
        # ------------------------------------------------------------------
        # Select data
        # ------------------------------------------------------------------
        data = self.transformed_data if (use_transformed and self.transformed_data is not None) else self.data
    
        if not isinstance(data, xr.DataArray):
            raise ValueError("`data` must be an xarray.DataArray")
        if not all(dim in data.dims for dim in ("T", "Y", "X")):
            raise ValueError("`data` must have dimensions ('T', 'Y', 'X')")
    
        if not hasattr(self, "distribution_map") or not isinstance(self.distribution_map, dict):
            raise ValueError("`self.distribution_map` must be a dict of {name: code}")
    
        Y = data.sizes["Y"]
        X = data.sizes["X"]
        coords = {"Y": data.Y, "X": data.X}
    
        # ------------------------------------------------------------------
        # Map distribution names in distribution_map to scipy.stats objects
        # (allow aliases like 't_dist' → t)
        # ------------------------------------------------------------------
        name_to_dist = {
            "norm": norm,
            "lognorm": lognorm,
            "expon": expon,
            "gamma": gamma_dist,
            "weibull_min": weibull_min,
            "t": t_dist,
            "t_dist": t_dist,
            "poisson": poisson,
            "nbinom": nbinom,
        }
    
        # Helper: filter only supported distributions
        dist_candidates = {
            name: (name_to_dist[name], code)
            for name, code in self.distribution_map.items()
            if name in name_to_dist
        }
        if not dist_candidates:
            raise ValueError("No valid distributions found in distribution_map")
    
        # ------------------------------------------------------------------
        # Core 1D fitter using AIC
        # ------------------------------------------------------------------
        def _fit_best(sample_1d, min_n):
            """
            Fit all candidate distributions to a 1D sample; return (code, shape, loc, scale)
            for the best by AIC. If no fit, return NaNs.
            """
            vals = np.asarray(sample_1d, float)
            vals = vals[np.isfinite(vals)]
            if vals.size < min_n:
                return np.nan, np.nan, np.nan, np.nan
    
            vals_pos = vals[vals > 0]
            vals_int = vals_pos[(vals_pos == np.floor(vals_pos))]
    
            best_aic = np.inf
            best = None  # (code, (shape, loc, scale))
    
            for name, (dist, code) in dist_candidates.items():
                is_discrete = name in ("poisson", "nbinom")
    
                # respect support
                if name in ("poisson", "nbinom"):
                    sample = vals_int
                elif name in ("lognorm", "gamma", "weibull_min", "expon"):
                    sample = vals_pos
                else:  # norm, t, etc.
                    sample = vals
    
                if sample.size < min_n:
                    continue
    
                try:
                    # ---- fit parameters ----
                    if name == "poisson":
                        # MLE: mu = mean, loc=0
                        mu = sample.mean()
                        if mu <= 0 or not np.isfinite(mu):
                            continue
                        params = (mu, 0.0)
                        k = 1  # only mu
    
                    elif name == "nbinom":
                        # fit (n, p, loc=0)
                        params = dist.fit(sample, floc=0)
                        # (n, p, loc) with loc fixed
                        k = len(params) - 1
    
                    elif name in ("lognorm", "gamma", "weibull_min", "expon"):
                        # positive-support, loc=0 fixed
                        params = dist.fit(sample, floc=0)
                        k = len(params) - 1
    
                    else:
                        # norm, t: all params free
                        params = dist.fit(sample)
                        k = len(params)
    
                    # ---- log-likelihood ----
                    if is_discrete:
                        if name == "poisson":
                            mu, loc = params
                            logL = np.sum(dist.logpmf(sample, mu, loc=loc))
                        else:  # nbinom
                            logL = np.sum(dist.logpmf(sample, *params))
                    else:
                        logL = np.sum(dist.logpdf(sample, *params))
    
                    if not np.isfinite(logL):
                        continue
    
                    aic = 2.0 * k - 2.0 * logL
                    if aic < best_aic:
                        best_aic = aic
    
                        # normalize to (shape, loc, scale)
                        if name == "poisson":
                            mu, loc = params
                            best = (code, (mu, loc, np.nan))
                        elif name == "nbinom":
                            n_, p_, loc_ = params  # (n, p, loc)
                            best = (code, (n_, loc_, p_))  # shape=n, loc=loc, scale=p
                        elif name in ("norm", "expon"):
                            loc_, scale_ = params
                            best = (code, (np.nan, loc_, scale_))
                        else:
                            # (shape, loc, scale): t, gamma, lognorm, weibull_min
                            shape_, loc_, scale_ = params
                            best = (code, (shape_, loc_, scale_))
    
                except Exception:
                    continue
    
            if best is None:
                return np.nan, np.nan, np.nan, np.nan
    
            code, (shape, loc, scale) = best
            return float(code), float(shape), float(loc), float(scale)
    
        # ------------------------------------------------------------------
        # Mode: per-grid using xr.apply_ufunc
        # ------------------------------------------------------------------
        if mode == "grid":
            min_n_grid = 15
    
            def _fit_best_grid(cell_ts):
                return _fit_best(cell_ts, min_n_grid)
    
            best_code_da, best_shape_da, best_loc_da, best_scale_da = xr.apply_ufunc(
                _fit_best_grid,
                data,
                input_core_dims=[["T"]],
                output_core_dims=[[], [], [], []],
                vectorize=True,
                dask="parallelized",
                output_dtypes=[float, float, float, float],
            )
    
            cluster_da = xr.full_like(best_code_da, np.nan)
            return (
                best_code_da,
                best_shape_da,
                best_loc_da,
                best_scale_da,
                cluster_da,
            )
    
        # ------------------------------------------------------------------
        # Mode: homogeneous zones via KMeans clustering
        # ------------------------------------------------------------------
        if mode == "cluster":
            if not hasattr(self, "n_clusters"):
                raise ValueError("For mode='cluster', self.n_clusters must be defined.")
    
            min_n_cluster = 30

            kmeans = KMeans(n_clusters=self.n_clusters, random_state=42, n_init=10)
            data_dropna = data.to_dataframe().reset_index().dropna().drop(columns=['T'])
            variable_column = data_dropna.columns[2]
            data_dropna['cluster'] = kmeans.fit_predict(
               data_dropna[[variable_column]]
            )
            
            # Convert cluster assignments back into an xarray structure
            df_unique = data_dropna.drop_duplicates(subset=['Y', 'X'])
            dataset = df_unique.set_index(['Y', 'X']).to_xarray()
            mask = xr.where(~np.isnan(data.isel(T=0)), 1, np.nan)
            Cluster = (dataset['cluster'] * mask)
                   
            # Align cluster array with the predictand array
            xarray1, xarray2 = xr.align(data, Cluster, join="outer")
            
            # Identify unique cluster labels
            clusters = np.unique(xarray2)
            unique_clusters = clusters[~np.isnan(clusters)]
            cluster_da = xarray2
    
    
            # 4) Fit best distribution for each cluster (pooled values)
            best_params_by_cluster = {}
    
            for cl_val in unique_clusters:
                cl = int(cl_val)
                mask_cl = (cluster_da == cl).expand_dims({'T': data['T']})
                cl_vals = data.where(mask_cl).stack(sample=('T', 'Y', 'X')).values.ravel()
                cl_vals = cl_vals[np.isfinite(cl_vals)]
                code, shape, loc, scale = _fit_best(cl_vals, min_n_cluster)
                if np.isnan(code):
                    best_params_by_cluster[cl] = None
                else:
                    best_params_by_cluster[cl] = (code, shape, loc, scale)
    
            # 5) Broadcast cluster-level params back to (Y,X) using xarray masks
            cluster_da_ = cluster_da.drop_vars("T")
            best_code_da = xr.full_like(cluster_da_, np.nan, dtype=float)
            best_shape_da = xr.full_like(cluster_da_, np.nan, dtype=float)
            best_loc_da = xr.full_like(cluster_da_, np.nan, dtype=float)
            best_scale_da = xr.full_like(cluster_da_, np.nan, dtype=float)
    
            for cl, params in best_params_by_cluster.items():
                if params is None:
                    continue
                code, shape, loc, scale = params
                mask_cl_ = (cluster_da_ == cl)
    
                best_code_da = best_code_da.where(~mask_cl_, code)
                best_shape_da = best_shape_da.where(~mask_cl_, shape)
                best_loc_da = best_loc_da.where(~mask_cl_, loc)
                best_scale_da = best_scale_da.where(~mask_cl_, scale)
    
            return best_code_da, best_shape_da, best_loc_da, best_scale_da, cluster_da_
    
        # ------------------------------------------------------------------
        # Invalid mode
        # ------------------------------------------------------------------
        raise ValueError("mode must be 'cluster' or 'grid'")

    

    def find_best_distribution_grid___(self, use_transformed=False):
        """
        Fit distributions to data using KMeans clustering.

        Parameters
        ----------
        use_transformed : bool, optional
            If True, use transformed data; otherwise, use original data. Default is False.

        Returns
        -------
        dist_codes : xarray.DataArray
            Numeric codes for best-fitting distributions per grid cell.

        Notes
        -----
        Uses `fitter.Fitter` to fit distributions (e.g., normal, lognormal) to clustered data.
        Clusters are determined by mean values using KMeans.
        """
        data = self.transformed_data if use_transformed and self.transformed_data is not None else self.data
        dist_names = tuple(self.distribution_map.keys())
        df_mean = data.mean('T', skipna=True).to_dataframe(name='value').dropna()
        if len(df_mean) < self.n_clusters:
            print("Warning: Insufficient data for clustering, returning NaN array")
            return xr.DataArray(
                np.full((self.data.sizes['Y'], self.data.sizes['X']), np.nan),
                coords={'Y': self.data.Y, 'X': self.data.X},
                dims=('Y', 'X')
            )
        kmeans = KMeans(n_clusters=self.n_clusters, random_state=42)
        df_mean['cluster'] = kmeans.fit_predict(df_mean[['value']])
        # clusters_da = df_mean.set_index(['Y', 'X'])['cluster'].to_xarray()
        clusters_da = df_mean['cluster'].to_xarray()
        valid_mask = ~np.isnan(data.isel(T=0))
        clusters_da = clusters_da * xr.where(valid_mask, 1, np.nan)
        _, clusters_aligned = xr.align(data, clusters_da, join='inner')
        dist_codes = {}
        for cl in np.unique(clusters_aligned):
            if np.isnan(cl):
                continue
            cl = int(cl)
            cl_data = data.where(clusters_aligned == cl).values
            cl_data = cl_data[~np.isnan(cl_data)]
            if cl_data.size < 2:
                dist_codes[cl] = np.nan
                continue
            try:
                ftr = Fitter(cl_data, distributions=dist_names, timeout=120)
                ftr.fit()
                best_name = next(iter(ftr.get_best(method='sumsquare_error')))
                dist_codes[cl] = self.distribution_map[best_name]
            except (RuntimeError, ValueError):
                dist_codes[cl] = np.nan
        return xr.apply_ufunc(
            lambda x: dist_codes.get(int(x), np.nan) if not np.isnan(x) else np.nan,
            clusters_aligned,
            vectorize=True,
            output_dtypes=[np.float32]
        )

    def plot_best_fit_map(
        self,
        data_array,
        map_dict,
        output_file='map.png',
        title='Categorical Map',
        colors=None,
        figsize=(10, 6),
        extent=None,
        show_plot=False
    ):
        """
        Plot a categorical map of distributions or skewness classes.

        Parameters
        ----------
        data_array : xarray.DataArray
            Data to plot (e.g., distribution codes or skewness classes) with dimensions (Y, X).
        map_dict : dict
            Mapping of category names to numeric codes (e.g., distribution_map).
        output_file : str, optional
            Path to save the plot. Default is 'map.png'.
        title : str, optional
            Plot title. Default is 'Categorical Map'.
        colors : list, optional
            Colors for each code. Default is ['blue', 'green', 'red', 'purple', 'orange'].
        figsize : tuple, optional
            Figure size (width, height) in inches. Default is (10, 6).
        extent : tuple, optional
            Map extent (lon_min, lon_max, lat_min, lat_max). Default is data bounds.
        show_plot : bool, optional
            If True, display the plot interactively. Default is False.

        Raises
        ------
        ValueError
            If insufficient colors are provided for the number of categories.

        Notes
        -----
        Uses `cartopy` for geospatial visualization with PlateCarree projection.
        Saves the plot as a PNG file.
        """
        if colors is None:
            colors = ['blue', 'green', 'red', 'purple', 'orange']
        code2name = {v: k for k, v in map_dict.items()}
        codes = np.unique(data_array.values[~np.isnan(data_array.values)]).astype(int)
        if len(colors) < len(codes):
            raise ValueError(f"Need at least {len(codes)} colors, got {len(colors)}")
        cmap = ListedColormap([colors[i % len(colors)] for i in range(len(codes))])
        bounds = np.concatenate([codes - 0.5, [codes[-1] + 0.5]])
        norm = BoundaryNorm(bounds, cmap.N)
        fig = plt.figure(figsize=figsize)
        ax = plt.axes(projection=ccrs.PlateCarree())
        if extent is None:
            extent = [
                float(data_array.X.min()),
                float(data_array.X.max()),
                float(data_array.Y.min()),
                float(data_array.Y.max())
            ]
        ax.set_extent(extent, crs=ccrs.PlateCarree())
        ax.add_feature(cfeature.COASTLINE, linewidth=0.4)
        ax.add_feature(cfeature.BORDERS, linewidth=0.4)
        ax.gridlines(draw_labels=True, linestyle='--', alpha=0.5)
        mesh = data_array.plot.pcolormesh(
            ax=ax,
            transform=ccrs.PlateCarree(),
            cmap=cmap,
            norm=norm,
            add_colorbar=False
        )
        cbar = plt.colorbar(mesh, ax=ax, ticks=codes, pad=0.05)
        cbar.set_ticklabels([code2name.get(c, 'unknown') for c in codes])
        cbar.set_label('Category')
        ax.set_title(title)
        ax.set_xlabel('Longitude')
        ax.set_ylabel('Latitude')
        plt.savefig(output_file, dpi=300, bbox_inches='tight')
        if show_plot:
            plt.show()
        plt.close()