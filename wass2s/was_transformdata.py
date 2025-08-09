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

    def __init__(self, data, distribution_map=None, n_clusters=5):
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
            'weibull_min': 5
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

    def find_best_distribution_grid(self, use_transformed=False):
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