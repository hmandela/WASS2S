"""Data transformation and distribution fitting for forecast fields.

Handles skewed precipitation distributions via Box-Cox, Yeo-Johnson,
and quantile-based transformations, and fits the best parametric
distribution per grid cell.

Standalone functions
--------------------
inv_boxcox
    Inverse Box-Cox transformation.
inv_yeojohnson
    Inverse Yeo-Johnson transformation.

Class
-----
WAS_TransformData
    End-to-end transformation pipeline:

    ``detect_skewness``
        Classify each grid cell as low / moderate / high skew.
    ``handle_skewness``
        Recommend the appropriate transformation per skewness class.
    ``apply_transformation``
        Apply Box-Cox, Yeo-Johnson, log, or square-root per grid cell.
    ``inverse_transform``
        Reverse the applied transformation.
    ``fit_best_distribution``
        Fit candidate parametric distributions (cluster or grid mode) and
        select the best-fitting one by AIC/BIC.
    ``plot_best_fit_map``
        Cartopy map of the winning distribution per grid cell.
"""
import xarray as xr
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import cartopy.crs as ccrs
import cartopy.feature as cfeature
from matplotlib.colors import ListedColormap, BoundaryNorm
from sklearn.cluster import KMeans
from scipy.stats import (
    skew, boxcox, yeojohnson,
    norm, lognorm, expon, gamma as gamma_dist, 
    weibull_min, t as t_dist, poisson, nbinom
)
import warnings

# Suppress warnings that occur during fitting (convergence warnings)
warnings.filterwarnings("ignore", category=RuntimeWarning)

def inv_boxcox(y, lmbda):
    """Inverse Box-Cox transformation."""
    if abs(lmbda) < 1e-6:
        return np.exp(y)
    return (y * lmbda + 1) ** (1 / lmbda)

def inv_yeojohnson(y, lmbda):
    """
    Inverse Yeo-Johnson transformation.
    """
    out = np.zeros_like(y)
    pos = y >= 0
    neg = ~pos

    # Case 1: y >= 0
    if abs(lmbda) < 1e-6:
        out[pos] = np.exp(y[pos]) - 1
    else:
        out[pos] = (y[pos] * lmbda + 1) ** (1 / lmbda) - 1

    # Case 2: y < 0
    if abs(lmbda - 2) < 1e-6:
        out[neg] = -np.exp(-y[neg]) + 1
    else:
        out[neg] = -((-y[neg] * (2 - lmbda) + 1) ** (1 / (2 - lmbda))) + 1
        
    return out

class WAS_TransformData:
    """
    Optimized class for geospatial Time-Series analysis: 
    Skewness detection, Transformation, and Distribution Fitting.
    """

    def __init__(self, data, distribution_map=None, n_clusters=1000):
        if not isinstance(data, xr.DataArray):
            raise ValueError("`data` must be an xarray.DataArray")
        if not all(dim in data.dims for dim in ('T', 'Y', 'X')):
            raise ValueError("`data` must have dimensions ('T', 'Y', 'X')")

        self.data = data
        self.distribution_map = distribution_map or {
            'norm': 1, 'lognorm': 2, 'expon': 3, 'gamma': 4, 
            'weibull_min': 5, "t_dist": 6, "poisson": 7, "nbinom": 8
        }
        self.n_clusters = n_clusters
        
        # Results containers
        self.transformed_data = None
        self.transform_methods = None
        self.transform_params = None # Now stores floats (lambdas), not objects
        self.skewness_ds = None
        self.handle_ds = None

    # ----------------------------------------------------------------------
    # 1. Skewness Detection (Unchanged - already efficient)
    # ----------------------------------------------------------------------
    def detect_skewness(self):
        """Compute and classify skewness for each grid cell."""
        def _compute(precip):
            precip = np.asarray(precip)
            valid = ~np.isnan(precip)
            if valid.sum() < 3:
                return np.nan, 'invalid'
            
            sk = skew(precip[valid], axis=0, nan_policy='omit')
            
            if np.isnan(sk): cls = 'invalid'
            elif -0.5 <= sk <= 0.5: cls = 'symmetric'
            elif 0.5 < sk <= 1: cls = 'moderate_positive'
            elif -1 <= sk < -0.5: cls = 'moderate_negative'
            elif sk > 1: cls = 'high_positive'
            else: cls = 'high_negative'
            
            return sk, cls

        res = xr.apply_ufunc(
            _compute, self.data,
            input_core_dims=[['T']], output_core_dims=[[], []],
            vectorize=True, dask='parallelized',
            output_dtypes=[float, str]
        )

        self.skewness_ds = xr.Dataset(
            {'skewness': res[0], 'skewness_class': res[1]}
        )
        counts = pd.Series(res[1].values.ravel()).value_counts().to_dict()
        return self.skewness_ds, {'class_counts': counts}

    # ----------------------------------------------------------------------
    # 2. Handle Skewness 
    # ----------------------------------------------------------------------
    def handle_skewness(self):
        """Recommend transformations based on skewness."""
        if self.skewness_ds is None:
            raise ValueError("Run detect_skewness() first")

        def _suggest(precip, sk_class):
            if sk_class == 'invalid': return 'none'
            valid = precip[~np.isnan(precip)]
            if len(valid) == 0: return 'none'
            
            all_pos = np.all(valid > 0)
            has_zeros = np.any(valid == 0)
            
            methods = []
            if sk_class in ('moderate_positive', 'high_positive'):
                if all_pos: methods += ['log', 'box_cox']
                methods += ['yeo_johnson', 'square_root'] # Yeo works on zeros
            elif sk_class in ('moderate_negative', 'high_negative'):
                methods += ['reflect_log'] if all_pos else []
                methods += ['reflect_yeo_johnson']
            else:
                methods.append('none')
            
            return ';'.join(methods) if methods else 'none'

        recommended = xr.apply_ufunc(
            _suggest, self.data, self.skewness_ds['skewness_class'],
            input_core_dims=[['T'], []], output_core_dims=[[]],
            vectorize=True, dask='parallelized', output_dtypes=[str]
        )

        self.handle_ds = xr.Dataset({
            'skewness': self.skewness_ds['skewness'],
            'skewness_class': self.skewness_ds['skewness_class'],
            'recommended_methods': recommended
        })
        return self.handle_ds

    # ----------------------------------------------------------------------
    # 3. Apply Transformation (Vectorized)
    # ----------------------------------------------------------------------
    def apply_transformation(self, method=None):
        """
        Apply transformations using vectorized ufuncs for speed.
        Stores parameters (lambdas) in self.transform_params.
        """
        if method is None and self.handle_ds is None:
            raise ValueError("Run handle_skewness() first or specify `method`")

        # 1. Determine method map
        if method is None:
            def extract_first(x):
                return x.split(';')[0] if (isinstance(x, str) and x) else 'none'
            method_da = xr.apply_ufunc(extract_first, self.handle_ds['recommended_methods'], vectorize=True)
        elif isinstance(method, str):
            method_da = xr.full_like(self.data.isel(T=0), method, dtype=object)
        else:
            method_da = method

        self.transform_methods = method_da

        # 2. Define the Core Transformation Logic (1D)
        def _transform_core(arr, method_str):
            # Returns: (transformed_array, parameter)
            # parameter is lambda for boxcox/yeo, or NaN for others
            
            out = arr.copy()
            param = np.nan
            
            mask = ~np.isnan(arr)
            vals = arr[mask]
            
            if len(vals) < 2 or method_str == 'none':
                return out, param

            try:
                if method_str == 'log':
                    if np.all(vals > 0): out[mask] = np.log(vals)
                
                elif method_str == 'square_root':
                    if np.all(vals >= 0): out[mask] = np.sqrt(vals)
                
                elif method_str == 'box_cox':
                    if np.all(vals > 0):
                        out[mask], param = boxcox(vals)
                
                elif method_str == 'yeo_johnson':
                    out[mask], param = yeojohnson(vals)
                
                elif method_str == 'reflect_log':
                    # Reflect: new = log(max(vals) + 1 - vals) or simple reflection -vals
                    # Assuming simple reflection based on previous code context:
                    ref = -vals
                    if np.all(ref > 0): out[mask] = np.log(ref)

                elif method_str == 'reflect_yeo_johnson':
                    out[mask], param = yeojohnson(-vals)
                    
                elif method_str == 'clipping':
                    # Example: Clip to 1st/99th percentile
                    lower, upper = np.percentile(vals, [1, 99])
                    out[mask] = np.clip(vals, lower, upper)
            
            except Exception:
                pass # Return original if fail
            
            return out, param

        # 3. Apply using xarray wrapper
        transformed, params = xr.apply_ufunc(
            _transform_core,
            self.data,
            self.transform_methods,
            input_core_dims=[['T'], []],
            output_core_dims=[['T'], []],
            vectorize=True,
            dask='parallelized',
            output_dtypes=[float, float] 
        )

        self.transformed_data = transformed
        self.transform_params = params # Now a DataArray of floats (lambdas)
        return self.transformed_data

    # ----------------------------------------------------------------------
    # 4. Inverse Transform 
    # ----------------------------------------------------------------------

    def inverse_transform(self, data=None):
        """
        Reverse transformations to recover original scale.
        
        Parameters
        ----------
        data : xarray.DataArray, optional
            If provided (e.g., model predictions), this data will be inverted 
            using the parameters learned during the `apply_transformation` step.
            If None, the internal `self.transformed_data` is inverted.
            
            IMPORTANT: `data` must have the same Y and X coordinates/dimensions 
            as the original data so the correct lambda matches the correct pixel.
        """
        # 1. Decide what to invert
        if data is not None:
            target_data = data
        elif self.transformed_data is not None:
            target_data = self.transformed_data
        else:
            raise ValueError("No data found to inverse. Pass `data` or run apply_transformation() first.")

        if self.transform_params is None:
             raise ValueError("Transform parameters missing. Run apply_transformation() first.")

        # 2. Define Inverse Logic
        def _inv_core(vec, method, param):
            # Check for missing/invalid methods
            if method == 'none' or pd.isnull(method): 
                return vec
            
            try:
                if method == 'log': return np.exp(vec)
                if method == 'square_root': return vec ** 2
                
                if method == 'box_cox':
                    if np.isnan(param): return vec
                    return inv_boxcox(vec, param)
                
                if method == 'yeo_johnson':
                    if np.isnan(param): return vec
                    return inv_yeojohnson(vec, param)
                
                if method == 'reflect_log':
                    return -np.exp(vec)
                
                if method == 'reflect_yeo_johnson':
                    if np.isnan(param): return -vec
                    return -inv_yeojohnson(vec, param)
                    
            except Exception:
                return vec 
            return vec

        # 3. Apply
        # Note: We use xarray broadcasting. 
        # transform_params has (Y, X). target_data has (T_new, Y, X).
        # xarray automatically aligns them by Y and X.
        return xr.apply_ufunc(
            _inv_core,
            target_data,
            self.transform_methods,
            self.transform_params,
            input_core_dims=[['T'], [], []], # vectors along time
            output_core_dims=[['T']],
            vectorize=True,
            dask='parallelized',
            output_dtypes=[float]
        )

    # ----------------------------------------------------------------------
    # 5. Unified Distribution Fitting (Grid + Cluster modes)
    # ----------------------------------------------------------------------
    def fit_best_distribution(self, use_transformed=False, mode="cluster"):
        """
        Fit best distributions using either clustering or per-grid methods.
        
        Parameters
        ----------
        mode : {'cluster', 'grid'}
        """
        # Data Selection
        data = self.transformed_data if (use_transformed and self.transformed_data is not None) else self.data
        Y, X = data.sizes["Y"], data.sizes["X"]
        coords = {"Y": data.Y, "X": data.X}

        # Map setup
        name_to_dist = {
            "norm": norm, "lognorm": lognorm, "expon": expon, 
            "gamma": gamma_dist, "weibull_min": weibull_min, 
            "t": t_dist, "t_dist": t_dist,
            "poisson": poisson, "nbinom": nbinom
        }
        dist_candidates = {
            name: (name_to_dist[name], code)
            for name, code in self.distribution_map.items() if name in name_to_dist
        }

        # --- Core Fitter Function ---
        def _fit_1d_core(sample, min_n=15):
            vals = sample[np.isfinite(sample)]
            if len(vals) < min_n:
                return np.nan, np.nan, np.nan, np.nan

            vals_pos = vals[vals > 0]
            
            best_aic = np.inf
            best_res = (np.nan, np.nan, np.nan, np.nan) # code, shape, loc, scale

            for name, (dist_obj, code) in dist_candidates.items():
                is_discrete = name in ("poisson", "nbinom")
                
                # 1. Select correct subset based on support
                if is_discrete:
                    # Enforce integer inputs for discrete distributions
                    current_sample = np.round(vals_pos)
                    # Filter out any lingering non-integers or zeros if strict
                    current_sample = current_sample[current_sample >= 0]
                elif name in ("lognorm", "gamma", "weibull_min", "expon"):
                    current_sample = vals_pos
                else:
                    current_sample = vals

                if len(current_sample) < min_n: continue

                try:
                    # 2. Fit Parameters
                    # Note: floc=0 enforces 0 bound for precip-like data
                    if name == "poisson":
                        mu = current_sample.mean()
                        if mu <= 0: continue
                        params = (mu, 0)
                        k = 1
                    elif name == "nbinom":
                        params = dist_obj.fit(current_sample, floc=0)
                        k = 2 # n, p (loc fixed)
                    elif name in ("lognorm", "gamma", "weibull_min", "expon"):
                        params = dist_obj.fit(current_sample, floc=0)
                        k = len(params) - 1
                    else:
                        params = dist_obj.fit(current_sample)
                        k = len(params)

                    # 3. Calculate AIC
                    if is_discrete:
                        if name == "poisson":
                            logL = np.sum(dist_obj.logpmf(current_sample, params[0], loc=params[1]))
                        else:
                            logL = np.sum(dist_obj.logpmf(current_sample, *params))
                    else:
                        logL = np.sum(dist_obj.logpdf(current_sample, *params))

                    aic = 2*k - 2*logL

                    if aic < best_aic:
                        best_aic = aic
                        # Standardize output to (code, shape, loc, scale)
                        # Helper to extract based on param length
                        p_len = len(params)
                        if name == "poisson":
                            # (mu, loc) -> shape=mu, loc=loc
                            best_res = (code, params[0], params[1], np.nan)
                        elif name == "nbinom":
                            # (n, p, loc) -> shape=n, loc=loc, scale=p
                            best_res = (code, params[0], params[2], params[1])
                        elif p_len == 2: # norm, expon (loc, scale)
                            best_res = (code, np.nan, params[0], params[1])
                        elif p_len == 3: # gamma, etc (shape, loc, scale)
                            best_res = (code, params[0], params[1], params[2])

                except Exception:
                    continue
            
            return best_res

        # --- MODE: Grid (Pixel by Pixel) ---
        if mode == "grid":
            res = xr.apply_ufunc(
                _fit_1d_core, data,
                input_core_dims=[['T']],
                output_core_dims=[[], [], [], []],
                vectorize=True, dask='parallelized',
                output_dtypes=[float, float, float, float]
            )
            return res[0], res[1], res[2], res[3], xr.full_like(res[0], np.nan)

        # --- MODE: Cluster (Regional Frequency Analysis) ---
        elif mode == "cluster":
            # 1. KMeans on Mean/Std
            df = data.to_dataframe().reset_index().dropna().drop(columns=['T'])
            if df.empty or len(df) < self.n_clusters:
                print("Not enough data for clustering.")
                nan_da = xr.full_like(data.isel(T=0), np.nan)
                return nan_da, nan_da, nan_da, nan_da, nan_da

            var_col = df.columns[-1] # Assuming data column is last
            kmeans = KMeans(n_clusters=self.n_clusters, n_init=10, random_state=42)
            df['cluster'] = kmeans.fit_predict(df[[var_col]])

            # Map clusters back to grid
            cluster_da = df.set_index(['Y', 'X'])['cluster'].to_xarray().reindex_like(data.isel(T=0))

            # 2. Loop over clusters (fast, N=1000 not N=1Million)
            best_code = xr.full_like(cluster_da, np.nan, dtype=float)
            best_shape = xr.full_like(cluster_da, np.nan, dtype=float)
            best_loc = xr.full_like(cluster_da, np.nan, dtype=float)
            best_scale = xr.full_like(cluster_da, np.nan, dtype=float)

            unique_clusters = np.unique(df['cluster'].values)
            
            for cl in unique_clusters:
                # Mask: All time steps for all pixels in this cluster
                mask = (cluster_da == cl)
                # Extract pooled data
                pooled = data.where(mask).values.flatten()
                
                # Fit
                c, s, l, sc = _fit_1d_core(pooled, min_n=30)
                
                # Broadcast results back to map
                if not np.isnan(c):
                    best_code = best_code.where(~mask, c)
                    best_shape = best_shape.where(~mask, s)
                    best_loc = best_loc.where(~mask, l)
                    best_scale = best_scale.where(~mask, sc)
            
            return best_code, best_shape, best_loc, best_scale, cluster_da

        else:
            raise ValueError("mode must be 'grid' or 'cluster'")
    def plot_best_fit_map(self, data_array, map_dict, output_file='map.png', 
                              title='Map', colors=None, show_plot=True):
            if colors is None:
                colors = ['#a6cee3', '#1f78b4', '#b2df8a', '#33a02c', 
                          '#fb9a99', '#e31a1c', '#fdbf6f', '#ff7f00']
                
            unique_vals = np.unique(data_array.values[~np.isnan(data_array.values)])
            if len(unique_vals) == 0:
                print("No data to plot.")
                return
    
            if len(colors) < len(unique_vals):
                colors = colors * (len(unique_vals) // len(colors) + 1)
                
            cmap = ListedColormap(colors[:len(unique_vals)])
            bounds = np.arange(len(unique_vals) + 1)
            norm = BoundaryNorm(bounds, cmap.N)
            
            val_to_idx = {v: i for i, v in enumerate(unique_vals)}
            code_to_name = {v: k for k, v in map_dict.items()}
            
            plot_data = xr.full_like(data_array, fill_value=np.nan, dtype=float)
            
            for val, idx in val_to_idx.items():
                plot_data = xr.where(data_array == val, idx, plot_data)
    
            fig = plt.figure(figsize=(10, 6))
            ax = plt.axes(projection=ccrs.PlateCarree())
            
            extent = [float(data_array.X.min()), float(data_array.X.max()), 
                      float(data_array.Y.min()), float(data_array.Y.max())]
            ax.set_extent(extent, crs=ccrs.PlateCarree())
            
            ax.add_feature(cfeature.COASTLINE, linewidth=0.5)
            ax.add_feature(cfeature.BORDERS, linewidth=0.5)
            ax.gridlines(draw_labels=True, linestyle='--', alpha=0.5)
    
            mesh = plot_data.plot.pcolormesh(
                ax=ax, transform=ccrs.PlateCarree(),
                cmap=cmap, norm=norm, add_colorbar=False
            )
            
            cbar = plt.colorbar(mesh, ax=ax, ticks=bounds[:-1] + 0.5, pad=0.05)
            cbar.set_ticklabels([code_to_name.get(v, str(v)) for v in unique_vals])
            
            ax.set_title(title)
            plt.savefig(output_file, dpi=300, bbox_inches='tight')
            if show_plot:
                plt.show()
            plt.close()