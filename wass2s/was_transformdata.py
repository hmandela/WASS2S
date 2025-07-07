import xarray as xr
import numpy as np
from scipy.stats import skew, boxcox, inv_boxcox
from sklearn.cluster import KMeans
from sklearn.preprocessing import PowerTransformer
from fitter import Fitter
import pandas as pd
import matplotlib.pyplot as plt
import cartopy.crs as ccrs
import cartopy.feature as cfeature
from matplotlib.colors import ListedColormap, BoundaryNorm

class WAS_TransformData:
    """
    A class to manage skewness analysis, data transformation, distribution fitting, and visualization
    for geospatial time-series data.

    Attributes
    ----------
    data : xarray.DataArray
        Input data with dimensions 'T' (time), 'Y', 'X'.
    distribution_map : dict
        Mapping of distribution names to numeric codes.
    n_clusters : int
        Number of clusters for KMeans in distribution fitting.
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
    """

    def __init__(self, data, distribution_map=None, n_clusters=5):

        """
        Initialize the WAS_TransformData class.

        Parameters
        ----------
        data : xarray.DataArray
            Input data with dimensions 'T', 'Y', 'X'.
        distribution_map : dict, optional
            Mapping of distribution names to codes. Default is:
            {'norm': 1, 'lognorm': 2, 'expon': 3, 'gamma': 4, 'weibull_min': 5}.
        n_clusters : int, optional
            Number of clusters for KMeans. Default is 5.
        """

        if not isinstance(data, xr.DataArray):
            raise ValueError("data must be an xarray.DataArray")
        if not all(dim in data.dims for dim in ['T', 'Y', 'X']):
            raise ValueError("data must have dimensions 'T', 'Y', 'X'")
        
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

    def detect_skewness(self):
        """
        Detect skewness in the data and classify it.

        Returns
        -------
        xarray.Dataset
            Contains 'skewness' and 'skewness_class' ('symmetric', 'moderate_positive', etc.).
        dict
            Summary with counts of grid cells by skewness class.
        """
        def compute_skewness(precip_data):
            precip_data = np.asarray(precip_data)
            if np.all(np.isnan(precip_data)) or np.sum(~np.isnan(precip_data)) < 3:
                return np.nan, 'invalid'
            skewness = skew(precip_data, axis=0, nan_policy='omit')
            if np.isnan(skewness):
                skew_class = 'invalid'
            elif -0.5 <= skewness <= 0.5:
                skew_class = 'symmetric'
            elif 0.5 < skewness <= 1:
                skew_class = 'moderate_positive'
            elif -1 <= skewness < -0.5:
                skew_class = 'moderate_negative'
            elif skewness > 1:
                skew_class = 'high_positive'
            else:
                skew_class = 'high_negative'
            return skewness, skew_class

        result = xr.apply_ufunc(
            compute_skewness,
            self.data,
            input_core_dims=[['T']],
            output_core_dims=[[], []],
            vectorize=True,
            dask='parallelized',
            output_dtypes=[float, str]
        )

        self.skewness_ds = xr.Dataset({
            'skewness': (['Y', 'X'], result[0].data),
            'skewness_class': (['Y', 'X'], result[1].data)
        }, coords={'Y': self.data.Y, 'X': self.data.X})

        class_counts = pd.Series(self.skewness_ds['skewness_class'].values.ravel()).value_counts().to_dict()
        summary = {'class_counts': class_counts}

        return self.skewness_ds, summary

    def handle_skewness(self):
        """
        Propose methods to handle skewness based on skewness analysis.

        Returns
        -------
        xarray.Dataset
            Contains 'skewness', 'skewness_class', 'recommended_methods'.
        dict
            General recommendations for each skewness class.
        """
        if self.skewness_ds is None:
            raise ValueError("Run detect_skewness first")

        def recommend_methods(precip_data, skew_class):
            if skew_class == 'invalid':
                return 'none'
            precip_data = np.asarray(precip_data)
            valid_data = precip_data[~np.isnan(precip_data)]
            has_zeros = np.any(valid_data == 0)
            has_negatives = np.any(valid_data < 0)
            all_positive = np.all(valid_data > 0)
            methods = []
            if skew_class in ['moderate_positive', 'high_positive']:
                if all_positive and not has_zeros:
                    methods.extend(['log', 'square_root', 'box_cox'])
                elif all_positive:
                    methods.extend(['square_root', 'box_cox'])
                methods.append('yeo_johnson')
                methods.append('clipping')
                methods.append('binning')
            elif skew_class in ['moderate_negative', 'high_negative']:
                if all_positive and not has_zeros:
                    methods.append('reflect_log')
                elif all_positive:
                    methods.append('reflect_square_root')
                methods.append('reflect_yeo_johnson')
                methods.append('clipping')
                methods.append('binning')
            else:
                methods.append('none')
            return ';'.join(methods)

        recommended = xr.apply_ufunc(
            recommend_methods,
            self.data,
            self.skewness_ds['skewness_class'],
            input_core_dims=[['T'], []],
            output_core_dims=[[]],
            vectorize=True,
            dask='parallelized',
            output_dtypes=[str]
        )

        self.handle_ds = xr.Dataset({
            'skewness': self.skewness_ds['skewness'],
            'skewness_class': self.skewness_ds['skewness_class'],
            'recommended_methods': (['Y', 'X'], recommended.data)
        }, coords={'Y': self.data.Y, 'X': self.data.X})

        summary = {
            'general_recommendations': {
                'symmetric': 'No transformation needed.',
                'moderate_positive': (
                    'Consider square root or Yeo-Johnson transformations. '
                    'Log or Box-Cox if no zeros. Clipping or binning for outliers.'
                ),
                'high_positive': (
                    'Strongly consider log (if no zeros), Box-Cox (if positive), or '
                    'Yeo-Johnson. Clipping or binning for extreme values.'
                ),
                'moderate_negative': (
                    'Reflect data and apply square root or Yeo-Johnson. '
                    'Clipping or binning for outliers.'
                ),
                'high_negative': (
                    'Reflect data and apply log (if no zeros), Box-Cox, or Yeo-Johnson. '
                    'Clipping or binning for extreme values.'
                ),
                'invalid': 'Insufficient data for skewness calculation.'
            }
        }

        return self.handle_ds, summary

    def apply_transformation(self, method=None):
        """
        Apply transformations to the data based on recommendations or a specified method.

        Parameters
        ----------
        method : str or xarray.DataArray, optional
            Transformation method to apply. If str, applies uniformly. If DataArray,
            specifies method per grid cell. If None, uses first recommended method
            from handle_skewness. Default is None.

        Returns
        -------
        xarray.DataArray
            Transformed data.
        """
        if method is None and self.handle_ds is None:
            raise ValueError("Run handle_skewness or specify a method")
        
        if method is None:
            # Use first recommended method per cell
            method = self.handle_ds['recommended_methods'].apply(
                lambda x: x.split(';')[0] if x and x != 'none' else 'none'
            )

        self.transformed_data = self.data.copy()
        self.transform_methods = method if isinstance(method, xr.DataArray) else xr.DataArray(
            np.full((len(self.data.Y), len(self.data.X)), method),
            coords={'Y': self.data.Y, 'X': self.data.X},
            dims=['Y', 'X']
        )
        self.transform_params = xr.DataArray(
            np.empty((len(self.data.Y), len(self.data.X)), dtype=object),
            coords={'Y': self.data.Y, 'X': self.data.X},
            dims=['Y', 'X']
        )

        for y in range(len(self.data.Y)):
            for x in range(len(self.data.X)):
                m = self.transform_methods.isel(Y=y, X=x).item()
                if m == 'none' or np.all(np.isnan(self.data[:, y, x])):
                    continue
                cell_data = self.data.isel(Y=y, X=x).values
                valid_data = cell_data[~np.isnan(cell_data)]
                
                if m == 'log' and np.all(valid_data > 0):
                    self.transformed_data[:, y, x] = np.log(cell_data)
                elif m == 'square_root':
                    self.transformed_data[:, y, x] = np.sqrt(cell_data)
                elif m == 'box_cox' and np.all(valid_data > 0):
                    self.transformed_data[:, y, x], lambda_param = boxcox(cell_data)
                    self.transform_params[y, x] = {'lambda': lambda_param}
                elif m == 'yeo_johnson':
                    transformer = PowerTransformer(method='yeo-johnson')
                    self.transformed_data[:, y, x] = transformer.fit_transform(
                        cell_data.reshape(-1, 1)
                    ).ravel()
                    self.transform_params[y, x] = {'transformer': transformer}
                elif m == 'reflect_log' and np.all(valid_data > 0):
                    self.transformed_data[:, y, x] = np.log(-cell_data)
                elif m == 'reflect_square_root':
                    self.transformed_data[:, y, x] = np.sqrt(-cell_data)
                elif m == 'reflect_yeo_johnson':
                    transformer = PowerTransformer(method='yeo-johnson')
                    self.transformed_data[:, y, x] = transformer.fit_transform(
                        (-cell_data).reshape(-1, 1)
                    ).ravel()
                    self.transform_params[y, x] = {'transformer': transformer}
                else:
                    print(f"Warning: Skipping invalid method '{m}' at Y={y}, X={x}")

        return self.transformed_data

    def inverse_transform(self):
        """
        Reverse transformations to recover original data scale.

        Returns
        -------
        xarray.DataArray
            Inversely transformed data.
        """
        if self.transformed_data is None or self.transform_methods is None:
            raise ValueError("No transformation applied. Run apply_transformation first.")

        def inverse_transform_cell(data, method, params):
            if method == 'none' or method is None or isinstance(method, float) and np.isnan(method):
                return data
            if method in ['clipping', 'binning']:
                print(f"Warning: '{method}' is not invertible.")
                return data
            if method == 'log':
                return np.exp(data)
            if method == 'square_root':
                return data ** 2
            if method == 'box_cox':
                if params is None or 'lambda' not in params:
                    raise ValueError("Box-Cox inversion requires 'lambda' in params")
                return inv_boxcox(data, params['lambda'])
            if method == 'yeo_johnson':
                if params is None or 'transformer' not in params:
                    raise ValueError("Yeo-Johnson inversion requires 'transformer' in params")
                return params['transformer'].inverse_transform(data.reshape(-1, 1)).ravel()
            if method in ['reflect_log', 'reflect_square_root', 'reflect_yeo_johnson']:
                if method == 'reflect_log':
                    temp = np.exp(data)
                elif method == 'reflect_square_root':
                    temp = data ** 2
                else:
                    if params is None or 'transformer' not in params:
                        raise ValueError("reflect_yeo_johnson requires 'transformer' in params")
                    temp = params['transformer'].inverse_transform(data.reshape(-1, 1)).ravel()
                return -temp
            raise ValueError(f"Unknown method: {method}")

        result = xr.apply_ufunc(
            inverse_transform_cell,
            self.transformed_data,
            self.transform_methods,
            self.transform_params,
            input_core_dims=[['T'], [], []],
            output_core_dims=[['T']],
            vectorize=True,
            dask='parallelized',
            output_dtypes=[float]
        )

        return result

    def find_best_distribution_grid(self, use_transformed=False):
        """
        Fit distributions to data using KMeans clustering.

        Parameters
        ----------
        use_transformed : bool, optional
            If True, use transformed data. Default is False (use original data).

        Returns
        -------
        xarray.DataArray
            Numeric codes for best-fitting distributions per grid cell.
        """
        data = self.transformed_data if use_transformed and self.transformed_data is not None else self.data
        dist_names = tuple(self.distribution_map.keys())
        data_mean = data.mean(dim='T', skipna=True)
        df = data_mean.to_dataframe(name='value').reset_index().dropna()
        kmeans = KMeans(n_clusters=self.n_clusters, random_state=42)
        df['cluster'] = kmeans.fit_predict(df[['value']])
        cluster_da = df.set_index(['Y', 'X'])['cluster'].to_xarray()
        valid_mask = ~np.isnan(data.isel(T=0))
        cluster_da = cluster_da * xr.where(valid_mask, 1, np.nan)
        _, cluster_da_aligned = xr.align(data, cluster_da, join='inner')
        clusters = np.unique(cluster_da_aligned)
        clusters = clusters[~np.isnan(clusters)].astype(int)
        dist_codes = {}
        for cluster in clusters:
            cluster_data = data.where(cluster_da_aligned == cluster).to_numpy()
            cluster_data = cluster_data[~np.isnan(cluster_data)]
            if len(cluster_data) < 2:
                dist_codes[cluster] = np.nan
                continue
            try:
                f = Fitter(cluster_data, distributions=dist_names, timeout=30)
                f.fit()
                best_dist_name = next(iter(f.get_best(method='sumsquare_error')))
                dist_codes[cluster] = self.distribution_map[best_dist_name]
            except (ValueError, RuntimeError):
                dist_codes[cluster] = np.nan
        output = xr.apply_ufunc(
            lambda x: dist_codes.get(int(x), np.nan) if not np.isnan(x) else np.nan,
            cluster_da_aligned,
            vectorize=True,
            output_dtypes=[np.float32]
        )
        return output

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
            Data to plot (e.g., best-fit distributions or skewness classes).
        map_dict : dict
            Mapping of names to numeric codes (e.g., distribution_map or class_map).
        output_file : str, optional
            Path to save the plot. Default is 'map.png'.
        title : str, optional
            Plot title. Default is 'Categorical Map'.
        colors : list, optional
            Colors for each code. Default is ['blue', 'green', 'red', 'purple', 'orange'].
        figsize : tuple, optional
            Figure size (width, height). Default is (10, 6).
        extent : tuple, optional
            Map extent (lon_min, lon_max, lat_min, lat_max). Default is data bounds.
        show_plot : bool, optional
            If True, display the plot. Default is False.

        Returns
        -------
        None
            Saves the plot to output_file and optionally displays it.
        """
        if colors is None:
            colors = ['blue', 'green', 'red', 'purple', 'orange']
        code_to_name = {code: name for name, code in map_dict.items()}
        valid_codes = np.unique(data_array.values[~np.isnan(data_array.values)])
        valid_codes = valid_codes.astype(int)
        if len(colors) < len(valid_codes):
            raise ValueError(f"Need at least {len(valid_codes)} colors, got {len(colors)}")
        cmap = ListedColormap([colors[i % len(colors)] for i in range(len(valid_codes))])
        bounds = np.concatenate([valid_codes - 0.5, [valid_codes[-1] + 0.5]])
        norm = BoundaryNorm(bounds, cmap.N)
        fig = plt.figure(figsize=figsize)
        ax = plt.axes(projection=ccrs.PlateCarree())
        if extent is None:
            lon_min, lon_max = data_array.X.min(), data_array.X.max()
            lat_min, lat_max = data_array.Y.min(), data_array.Y.max()
            extent = [lon_min, lon_max, lat_min, lat_max]
        ax.set_extent(extent, crs=ccrs.PlateCarree())
        ax.add_feature(cfeature.COASTLINES, linewidth=0.5)
        ax.add_feature(cfeature.BORDERS, linewidth=0.5)
        ax.gridlines(draw_labels=True, linestyle='--', alpha=0.5)
        plot = data_array.plot.pcolormesh(
            ax=ax,
            transform=ccrs.PlateCarree(),
            cmap=cmap,
            norm=norm,
            add_colorbar=False
        )
        cbar = plt.colorbar(plot, ax=ax, ticks=valid_codes, orientation='vertical', pad=0.05)
        cbar.set_label('Category')
        cbar.set_ticklabels([code_to_name.get(code, 'Unknown') for code in valid_codes])
        plt.title(title)
        ax.set_xlabel('Longitude')
        ax.set_ylabel('Latitude')
        plt.savefig(output_file, dpi=300, bbox_inches='tight')
        if show_plot:
            plt.show()
        plt.close()