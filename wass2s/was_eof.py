# import matplotlib.pyplot as plt
# import cartopy.crs as ccrs
# import cartopy.feature as cfeature
# from matplotlib import colors
# import xarray as xr 
# import numpy as np
# import pandas as pd
# from scipy import stats
# import xeofs as xe
# import scipy.signal as sig

# ### Complete WAS_EOF  with multiple eof zone!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
# class WAS_EOF_:
#     """
#     A class for performing Empirical Orthogonal Function (EOF) analysis using the xeofs package, 
#     with additional options for detrending and cosine latitude weighting.

#     Parameters
#     ----------
#     n_modes : int, optional
#         The number of EOF modes to retain. If None, the number of modes is determined by 
#         explained variance.
#     use_coslat : bool, optional
#         If True, applies cosine latitude weighting to account for the Earth's spherical geometry.
#     standardize : bool, optional
#         If True, standardizes the input data by removing the mean and dividing by the standard deviation.
#     detrend : bool, optional
#         If True, detrends the input data along the time dimension before performing EOF analysis.
#     opti_explained_variance : float, optional
#         The target cumulative explained variance (in percent) to determine the optimal number of EOF modes.
#     L2norm : bool, optional
#         If True, normalizes the components and scores to have L2 norm.

#     Attributes
#     ----------
#     model : xeofs.models.EOF
#         The EOF model fitted to the predictor data.
#     """

#     def __init__(self, n_modes=None, use_coslat=True, standardize=False,
#                   opti_explained_variance=None, detrend=True, L2norm=True):
#         self.n_modes = n_modes
#         self.use_coslat = use_coslat
#         self.standardize = standardize
#         self.opti_explained_variance = opti_explained_variance
#         self.detrend = detrend
#         self.L2norm = L2norm
#         self.model = None

#     def _detrended_da(self, da):
#         """Detrend a DataArray by removing the linear trend."""
#         if 'T' not in da.dims:
#             raise ValueError("DataArray must have a time dimension 'T' for detrending.")
#         trend = da.polyfit(dim='T', deg=1)
#         da_detrended = da - (trend.polyval(da['T']) if 'polyval' in dir(trend) else trend)
#         return da_detrended.isel(degree=0, drop=True).to_array().drop_vars('variable').squeeze(),\
#              trend.isel(degree=0, drop=True).to_array().drop_vars('variable').squeeze()

#     def fit(self, predictor, dim="T", clim_year_start=None, clim_year_end=None):
#         predictor = predictor.fillna(predictor.mean(dim="T", skipna=True))
#         predictor = predictor.rename({"X": "lon", "Y": "lat"})
#         if self.detrend:
#             predictor, _ = self._detrended_da(predictor)

#         if self.n_modes is not None:
#             self.model = xe.single.EOF(n_modes=self.n_modes, use_coslat=self.use_coslat, standardize=self.standardize)
#         else:
#             self.model = xe.single.EOF(n_modes=100, use_coslat=self.use_coslat, standardize=self.standardize)
#             self.model.fit(predictor, dim=dim)

#             if self.opti_explained_variance is not None:
#                 npcs = 0
#                 sum_explain_var = 0
#                 while sum_explain_var * 100 < self.opti_explained_variance:
#                     npcs += 1
#                     sum_explain_var = sum(self.model.explained_variance_ratio()[:npcs])
#                 self.model = xe.single.EOF(n_modes=npcs, use_coslat=self.use_coslat, standardize=self.standardize)

#         self.model.fit(predictor, dim=dim)

#         s_eofs = self.model.components(normalized=self.L2norm)
#         s_pcs = self.model.scores(normalized=self.L2norm)
#         s_expvar = self.model.explained_variance_ratio()
#         s_sing_values = self.model.singular_values()

#         return s_eofs, s_pcs, s_expvar, s_sing_values

#     def transform(self, predictor):
#         predictor = predictor.rename({"X": "lon", "Y": "lat"})

#         if self.model is None:
#             raise ValueError("The model has not been fitted yet.")

#         return self.model.transform(predictor, normalized=self.L2norm)

#     def inverse_transform(self, pcs):
#         if self.model is None:
#             raise ValueError("The model has not been fitted yet.")

#         return self.model.inverse_transform(pcs, normalized=self.L2norm)

#     def plot_EOF(self, s_eofs, s_expvar):
#         """
#         Plot the EOF spatial patterns and their explained variance.

#         Parameters
#         ----------
#         s_eofs : xarray.DataArray
#             The EOF spatial patterns to plot.
#         s_expvar : numpy.ndarray
#             The explained variance for each EOF mode.
#         """
#         s_expvar = s_expvar.values.tolist() 
#         n_modes = len(s_eofs.coords['mode'].values.tolist())
#         n_cols = 3
#         n_rows = (n_modes + n_cols - 1) // n_cols
        
#         fig, axes = plt.subplots(
#             n_rows, n_cols, 
#             figsize=(n_cols * 6, n_rows * 4),
#             subplot_kw={'projection': ccrs.PlateCarree()}
#         )
        
#         axes = axes.flatten()
#         norm = colors.Normalize(vmin=s_eofs.min(dim=["lon", "lat", "mode"]), 
#                                 vmax=s_eofs.max(dim=["lon", "lat", "mode"]), clip=False)
        
#         for i, mode in enumerate(s_eofs.coords['mode'].values.tolist()):
#             ax = axes[i]
#             data = s_eofs.sel(mode=mode)
            
#             im = ax.pcolormesh(
#                 s_eofs.lon, s_eofs.lat, data, cmap="RdBu_r", norm=norm, 
#                 transform=ccrs.PlateCarree()
#             )

#             ax.coastlines()
#             ax.add_feature(cfeature.LAND, edgecolor="black")
#             ax.add_feature(cfeature.OCEAN, facecolor="lightblue")
#             ax.set_title(f"Mode {mode} -- Explained variance {round(s_expvar[i], 2) * 100}%")
        
#         for j in range(n_modes, len(axes)):
#             fig.delaxes(axes[j])
        
#         bottom_margin = 0.1 + 0.075 * n_rows
#         cbar = fig.colorbar(im, ax=axes, orientation="horizontal", shrink=0.5, aspect=40, pad=0.1)
#         cbar.set_label('EOF Values')
#         fig.suptitle("EOF Modes", fontsize=16)
#         plt.tight_layout()
#         fig.subplots_adjust(top=0.9, bottom=bottom_margin)
#         plt.show()



# import matplotlib.pyplot as plt
# import cartopy.crs as ccrs
# import cartopy.feature as cfeature
# from matplotlib import colors
# import xarray as xr 
# import numpy as np
# from xeofs.single import EOF

# class WAS_EOF:
#     """
#     Empirical Orthogonal Function (EOF) analysis class using xeofs.
#     Features: Automatic detrending, variance optimization, and projection.
#     """

#     def __init__(self, n_modes=None, use_coslat=True, standardize=False, 
#                  opti_explained_variance=None, detrend=True, L2norm=True):
#         self.n_modes = n_modes
#         self.use_coslat = use_coslat
#         self.standardize = standardize
#         self.opti_explained_variance = opti_explained_variance
#         self.detrend = detrend
#         self.L2norm = L2norm
        
#         self.model = None
#         self.trend_coeffs = None 
#         self.time_dim = None

#     def _detrend_data(self, da, dim):
#         """Internal: Removes linear trend and returns anomalies + coefficients."""
#         if dim not in da.dims:
#             raise ValueError(f"Dimension '{dim}' not found in DataArray.")
        
#         coeffs = da.polyfit(dim=dim, deg=1)
#         trend = xr.polyval(da[dim], coeffs.polyfit_coefficients)
#         return da - trend, coeffs

#     def fit(self, predictor, dim="T", clim_year_start=None, clim_year_end=None):
#         self.time_dim = dim
        
#         # Standardize dimensions
#         rename_map = {}
#         if "X" in predictor.dims: rename_map["X"] = "lon"
#         if "Y" in predictor.dims: rename_map["Y"] = "lat"
#         if "T" in predictor.dims: rename_map["T"] = dim
#         if rename_map:
#             predictor = predictor.rename(rename_map)

#         # Handle NaNs
#         if predictor.isnull().any():
#             predictor = predictor.fillna(predictor.mean(dim=dim, skipna=True))

#         # Detrending logic
#         data_to_fit = predictor
#         if self.detrend:
#             data_to_fit, self.trend_coeffs = self._detrend_data(predictor, dim=dim)

#         # Variance optimization
#         if self.opti_explained_variance is not None:
#             temp_model = EOF(n_modes=50, use_coslat=self.use_coslat, standardize=self.standardize)
#             temp_model.fit(data_to_fit, dim=dim)
#             exp_var = temp_model.explained_variance_ratio().cumsum()
#             self.n_modes = int(np.searchsorted(exp_var.values * 100, self.opti_explained_variance) + 1)

#         # Final Fit
#         final_modes = self.n_modes if self.n_modes else 50
#         self.model = EOF(n_modes=final_modes, use_coslat=self.use_coslat, standardize=self.standardize)
#         self.model.fit(data_to_fit, dim=dim)

#         return (self.model.components(normalized=self.L2norm), 
#                 self.model.scores(normalized=self.L2norm), 
#                 self.model.explained_variance_ratio())

    
#     def transform(self, predictor, dim="T"):
#         if self.model is None:
#             raise ValueError("Model not fitted.")

#         rename_map = {}
#         if "X" in predictor.dims: rename_map["X"] = "lon"
#         if "Y" in predictor.dims: rename_map["Y"] = "lat"
#         if "T" in predictor.dims: rename_map["T"] = dim
#         if rename_map:
#             predictor = predictor.rename(rename_map)

#         data_to_transform = predictor
#         # Apply historical trend to new data
#         if self.detrend and self.trend_coeffs is not None:
#             trend = xr.polyval(predictor[dim], self.trend_coeffs.polyfit_coefficients)
#             data_to_transform = predictor - trend


#         return self.model.transform(data_to_transform, normalized=self.L2norm)

#     def inverse_transform(self, pcs, return_anomalies=False):
#         if self.model is None:
#             raise ValueError("Model not fitted.")

#         reconstructed = self.model.inverse_transform(pcs, normalized=self.L2norm)

#         # Add trend back if needed
#         if self.detrend and self.trend_coeffs is not None and not return_anomalies:
#             trend = xr.polyval(pcs[self.time_dim], self.trend_coeffs.polyfit_coefficients)
#             reconstructed = reconstructed + trend

#         return reconstructed

    # def plot_EOF(self, s_eofs, s_expvar):
    #     """
    #     Plot the EOF spatial patterns and their explained variance.

    #     Parameters
    #     ----------
    #     s_eofs : xarray.DataArray
    #         The EOF spatial patterns to plot.
    #     s_expvar : numpy.ndarray
    #         The explained variance for each EOF mode.
    #     """
    #     s_expvar = s_expvar.values.tolist() 
    #     n_modes = len(s_eofs.coords['mode'].values.tolist())
    #     n_cols = 3
    #     n_rows = (n_modes + n_cols - 1) // n_cols
        
    #     fig, axes = plt.subplots(
    #         n_rows, n_cols, 
    #         figsize=(n_cols * 6, n_rows * 4),
    #         subplot_kw={'projection': ccrs.PlateCarree()}
    #     )
        
    #     axes = axes.flatten()
    #     norm = colors.Normalize(vmin=s_eofs.min(dim=["lon", "lat", "mode"]), 
    #                             vmax=s_eofs.max(dim=["lon", "lat", "mode"]), clip=False)
        
    #     for i, mode in enumerate(s_eofs.coords['mode'].values.tolist()):
    #         ax = axes[i]
    #         data = s_eofs.sel(mode=mode)
            
    #         im = ax.pcolormesh(
    #             s_eofs.lon, s_eofs.lat, data, cmap="RdBu_r", norm=norm, 
    #             transform=ccrs.PlateCarree()
    #         )

    #         ax.coastlines()
    #         ax.add_feature(cfeature.LAND, edgecolor="black")
    #         ax.add_feature(cfeature.OCEAN, facecolor="lightblue")
    #         ax.set_title(f"Mode {mode} -- Explained variance {round(s_expvar[i], 2) * 100}%")
        
    #     for j in range(n_modes, len(axes)):
    #         fig.delaxes(axes[j])
        
    #     bottom_margin = 0.1 + 0.075 * n_rows
    #     cbar = fig.colorbar(im, ax=axes, orientation="horizontal", shrink=0.5, aspect=40, pad=0.1)
    #     cbar.set_label('EOF Values')
    #     fig.suptitle("EOF Modes", fontsize=16)
    #     plt.tight_layout()
    #     fig.subplots_adjust(top=0.9, bottom=bottom_margin)
    #     plt.show()
    
    
# import matplotlib.pyplot as plt
# import cartopy.crs as ccrs
# import cartopy.feature as cfeature
# from matplotlib import colors
# import xarray as xr 
# import numpy as np
# from xeofs.single import EOF

# class WAS_EOF:
#     def __init__(self, n_modes=None, use_coslat=True, standardize=False, 
#                  opti_explained_variance=None, detrend=True, L2norm=True):
#         self.n_modes = n_modes
#         self.use_coslat = use_coslat
#         self.standardize = standardize
#         self.opti_explained_variance = opti_explained_variance
#         self.detrend = detrend
#         self.L2norm = L2norm
        
#         self.model = None
#         self.trend_coeffs = None 
#         self.trend_meta = None 
#         self.time_dim = None

#     def _detrended_da(self, da, dim="T", min_valid=2):
#         if dim not in da.dims:
#             raise ValueError(f"Dimension '{dim}' not found in DataArray.")
        
#         x = da[dim]
#         if np.issubdtype(x.dtype, np.datetime64):
#             x0 = x.isel({dim: 0}).values
#             x_days = (x - x.isel({dim: 0})).astype("timedelta64[D]").astype(np.float64)
#         else:
#             x0 = float(x.isel({dim: 0}).values) if x.size else 0.0
#             x_days = x.astype(np.float64)

#         try:
#             coeffs = da.assign_coords({dim: x_days}).polyfit(dim=dim, deg=1, skipna=True)
#         except TypeError:
#             coeffs = da.assign_coords({dim: x_days}).polyfit(dim=dim, deg=1)

#         if not isinstance(x_days, xr.DataArray):
#              x_days = xr.DataArray(x_days, dims=dim, coords={dim: da[dim]})
             
#         trend = xr.polyval(x_days, coeffs.polyfit_coefficients)
        
#         da_detrended = da - trend
        
#         meta = {"dim": dim, "x0": x0, "type": "datetime" if np.issubdtype(x.dtype, np.datetime64) else "numeric"}
        
#         return da_detrended, coeffs, meta

#     def _apply_detrend(self, da, coeffs, meta):
#         dim = meta["dim"]
#         if dim not in da.dims:
#              if 'T' in da.dims: dim = 'T'
#              elif 'time' in da.dims: dim = 'time'
        
#         x = da[dim]
        
#         if meta["type"] == "datetime":
#             x0 = np.datetime64(meta["x0"])
#             x_days = (x - x0).astype("timedelta64[D]").astype(np.float64)
#         else:
#             x_days = x.astype(np.float64)
            
#         if not isinstance(x_days, xr.DataArray):
#              x_days = xr.DataArray(x_days, coords={dim: x}, dims=dim)

#         trend = xr.polyval(x_days, coeffs.polyfit_coefficients)
#         return trend

#     def fit(self, predictor, dim="T", clim_year_start=None, clim_year_end=None):
#         self.time_dim = dim
        
#         rename_map = {}
#         if "X" in predictor.dims: rename_map["X"] = "lon"
#         if "Y" in predictor.dims: rename_map["Y"] = "lat"
#         if "T" in predictor.dims: rename_map["T"] = dim
#         if rename_map:
#             predictor = predictor.rename(rename_map)

#         if predictor.isnull().any():
#             predictor = predictor.fillna(predictor.mean(dim=dim, skipna=True))

#         data_to_fit = predictor
#         if self.detrend:
#             data_to_fit, self.trend_coeffs, self.trend_meta = self._detrended_da(predictor, dim=dim)

#         if self.opti_explained_variance is not None:
#             temp_model = EOF(n_modes=50, use_coslat=self.use_coslat, standardize=self.standardize)
#             temp_model.fit(data_to_fit, dim=dim)
#             exp_var = temp_model.explained_variance_ratio().cumsum()
#             self.n_modes = int(np.searchsorted(exp_var.values * 100, self.opti_explained_variance) + 1)

#         final_modes = self.n_modes if self.n_modes else 50
#         self.model = EOF(n_modes=final_modes, use_coslat=self.use_coslat, standardize=self.standardize)
#         self.model.fit(data_to_fit, dim=dim)

#         return (self.model.components(normalized=self.L2norm), 
#                 self.model.scores(normalized=self.L2norm), 
#                 self.model.explained_variance_ratio())

#     def transform(self, predictor, dim="T"):
#         if self.model is None:
#             raise ValueError("Model not fitted.")

#         rename_map = {}
#         if "X" in predictor.dims: rename_map["X"] = "lon"
#         if "Y" in predictor.dims: rename_map["Y"] = "lat"
#         if "T" in predictor.dims: rename_map["T"] = dim
#         if rename_map:
#             predictor = predictor.rename(rename_map)

#         data_to_transform = predictor
        
#         if self.detrend and self.trend_coeffs is not None:
#             trend = self._apply_detrend(predictor, self.trend_coeffs, self.trend_meta)
#             data_to_transform = predictor - trend

#         return self.model.transform(data_to_transform, normalized=self.L2norm)

#     def inverse_transform(self, pcs, return_anomalies=False):
#         if self.model is None:
#             raise ValueError("Model not fitted.")

#         reconstructed = self.model.inverse_transform(pcs, normalized=self.L2norm)

#         if self.detrend and self.trend_coeffs is not None and not return_anomalies:
#             trend = self._apply_detrend(pcs, self.trend_coeffs, self.trend_meta)
#             reconstructed = reconstructed + trend

#         return reconstructed

#     def plot_EOF(self, s_eofs, s_expvar):
#         """
#         Plot the EOF spatial patterns and their explained variance.

#         Parameters
#         ----------
#         s_eofs : xarray.DataArray
#             The EOF spatial patterns to plot.
#         s_expvar : numpy.ndarray
#             The explained variance for each EOF mode.
#         """
#         s_expvar = s_expvar.values.tolist() 
#         n_modes = len(s_eofs.coords['mode'].values.tolist())
#         n_cols = 3
#         n_rows = (n_modes + n_cols - 1) // n_cols
        
#         fig, axes = plt.subplots(
#             n_rows, n_cols, 
#             figsize=(n_cols * 6, n_rows * 4),
#             subplot_kw={'projection': ccrs.PlateCarree()}
#         )
        
#         axes = axes.flatten()
#         norm = colors.Normalize(vmin=s_eofs.min(dim=["lon", "lat", "mode"]), 
#                                 vmax=s_eofs.max(dim=["lon", "lat", "mode"]), clip=False)
        
#         for i, mode in enumerate(s_eofs.coords['mode'].values.tolist()):
#             ax = axes[i]
#             data = s_eofs.sel(mode=mode)
            
#             im = ax.pcolormesh(
#                 s_eofs.lon, s_eofs.lat, data, cmap="RdBu_r", norm=norm, 
#                 transform=ccrs.PlateCarree()
#             )

#             ax.coastlines()
#             ax.add_feature(cfeature.LAND, edgecolor="black")
#             ax.add_feature(cfeature.OCEAN, facecolor="lightblue")
#             ax.set_title(f"Mode {mode} -- Explained variance {round(s_expvar[i], 2) * 100}%")
        
#         for j in range(n_modes, len(axes)):
#             fig.delaxes(axes[j])
        
#         bottom_margin = 0.1 + 0.075 * n_rows
#         cbar = fig.colorbar(im, ax=axes, orientation="horizontal", shrink=0.5, aspect=40, pad=0.1)
#         cbar.set_label('EOF Values')
#         fig.suptitle("EOF Modes", fontsize=16)
#         plt.tight_layout()
#         fig.subplots_adjust(top=0.9, bottom=bottom_margin)
#         plt.show()
    
    
import numpy as np
import xarray as xr
import matplotlib.pyplot as plt
from matplotlib import colors
import cartopy.crs as ccrs
import cartopy.feature as cfeature
from xeofs.single import EOF

class WAS_EOF:
    def __init__(self, n_modes=None, use_coslat=True, standardize=False, 
                 opti_explained_variance=None, detrend=True, L2norm=True):
        self.n_modes = n_modes
        self.use_coslat = use_coslat
        self.standardize = standardize
        self.opti_explained_variance = opti_explained_variance
        self.detrend = detrend
        self.L2norm = L2norm

        self.model = None
        self.trend_coeffs = None
        self.trend_meta = None
        self.time_dim = None

    def _detrended_da(self, da, dim="T", min_valid=10):
        if dim not in da.dims:
            raise ValueError(f"Dimension '{dim}' not found in DataArray.")

        x = da[dim]

        if np.issubdtype(x.dtype, np.datetime64):
            x0 = x.isel({dim: 0}).values
            x_days = (x - x.isel({dim: 0})).astype("timedelta64[D]").astype(np.float64)
            x_type = "datetime"
        else:
            x0 = float(x.isel({dim: 0}).values) if x.size else 0.0
            x_days = x.astype(np.float64)
            x_type = "numeric"

        ok = da.notnull().sum(dim=dim) >= int(min_valid)
        da_fit = da.where(ok)

        try:
            coeffs = da_fit.assign_coords({dim: x_days}).polyfit(dim=dim, deg=1, skipna=True)
        except TypeError:
            coeffs = da_fit.assign_coords({dim: x_days}).polyfit(dim=dim, deg=1)

        if not isinstance(x_days, xr.DataArray):
             x_days = xr.DataArray(x_days, dims=dim, coords={dim: da[dim]})

        trend = xr.polyval(x_days, coeffs.polyfit_coefficients)
        da_detrended = da - trend
        da_detrended.attrs = da.attrs.copy()

        meta = {
            "dim": dim,
            "x0": x0,
            "x_units": "days",
            "type": x_type,
            "min_valid": int(min_valid),
        }
        return da_detrended, coeffs, meta

    def _apply_detrend(self, da, coeffs, meta):
        dim = meta.get("dim", "T")
        if dim not in da.dims:
            if "T" in da.dims:
                dim = "T"
            elif "time" in da.dims:
                dim = "time"
            else:
                raise ValueError(f"Cannot find time dim. Expected '{meta.get('dim')}'. Found: {da.dims}")

        x = da[dim]

        if meta.get("type") == "datetime":
            x0 = np.datetime64(meta["x0"])
            x_days = (x - x0).astype("timedelta64[D]").astype(np.float64)
        else:
            x_days = x.astype(np.float64)

        if not isinstance(x_days, xr.DataArray):
             x_days = xr.DataArray(x_days, dims=dim, coords={dim: da[dim]})

        trend = xr.polyval(x_days, coeffs.polyfit_coefficients)
        return trend

    def fit(self, predictor, dim="T", clim_year_start=None, clim_year_end=None):
        self.time_dim = dim

        rename_map = {}
        if "X" in predictor.dims and "lon" not in predictor.dims:
            rename_map["X"] = "lon"
        if "Y" in predictor.dims and "lat" not in predictor.dims:
            rename_map["Y"] = "lat"
        if "T" in predictor.dims and dim not in predictor.dims:
            rename_map["T"] = dim
        if "time" in predictor.dims and dim not in predictor.dims and dim != "time":
            rename_map["time"] = dim

        if rename_map:
            predictor = predictor.rename(rename_map)

        data_to_fit = predictor

        if self.detrend:
            data_to_fit, self.trend_coeffs, self.trend_meta = self._detrended_da(data_to_fit, dim=dim)

        if self.opti_explained_variance is not None:
            tmp = EOF(n_modes=50, use_coslat=self.use_coslat, standardize=self.standardize)
            tmp.fit(data_to_fit, dim=dim)
            cum = tmp.explained_variance_ratio().cumsum()
            self.n_modes = int(np.searchsorted(cum.values * 100.0, self.opti_explained_variance) + 1)

        final_modes = int(self.n_modes) if self.n_modes else 50
        self.model = EOF(n_modes=final_modes, use_coslat=self.use_coslat, standardize=self.standardize)
        self.model.fit(data_to_fit, dim=dim)

        return (
            self.model.components(normalized=self.L2norm),
            self.model.scores(normalized=self.L2norm),
            self.model.explained_variance_ratio(),
        )

    def transform(self, predictor, dim="T"):
        if self.model is None:
            raise ValueError("Model not fitted.")

        rename_map = {}
        if "X" in predictor.dims and "lon" not in predictor.dims:
            rename_map["X"] = "lon"
        if "Y" in predictor.dims and "lat" not in predictor.dims:
            rename_map["Y"] = "lat"
        if "T" in predictor.dims and dim not in predictor.dims:
            rename_map["T"] = dim
        if "time" in predictor.dims and dim not in predictor.dims and dim != "time":
            rename_map["time"] = dim
        if rename_map:
            predictor = predictor.rename(rename_map)

        data_to_transform = predictor

        if self.detrend and (self.trend_coeffs is not None) and (self.trend_meta is not None):
            trend = self._apply_detrend(data_to_transform, self.trend_coeffs, self.trend_meta)
            data_to_transform = data_to_transform - trend

        return self.model.transform(data_to_transform, normalized=self.L2norm)

    def inverse_transform(self, pcs, return_anomalies=False):
        if self.model is None:
            raise ValueError("Model not fitted.")

        reconstructed = self.model.inverse_transform(pcs, normalized=self.L2norm)

        if self.detrend and (self.trend_coeffs is not None) and (self.trend_meta is not None) and (not return_anomalies):
            trend = self._apply_detrend(reconstructed, self.trend_coeffs, self.trend_meta)
            reconstructed = reconstructed + trend
        return reconstructed

    def plot_EOF(self, s_eofs, s_expvar):
        """
        Plot the EOF spatial patterns and their explained variance.

        Parameters
        ----------
        s_eofs : xarray.DataArray
            The EOF spatial patterns to plot.
        s_expvar : numpy.ndarray
            The explained variance for each EOF mode.
        """
        s_expvar = s_expvar.values.tolist() 
        n_modes = len(s_eofs.coords['mode'].values.tolist())
        n_cols = 3
        n_rows = (n_modes + n_cols - 1) // n_cols
        
        fig, axes = plt.subplots(
            n_rows, n_cols, 
            figsize=(n_cols * 6, n_rows * 4),
            subplot_kw={'projection': ccrs.PlateCarree()}
        )
        
        axes = axes.flatten()
        norm = colors.Normalize(vmin=s_eofs.min(dim=["lon", "lat", "mode"]), 
                                vmax=s_eofs.max(dim=["lon", "lat", "mode"]), clip=False)
        
        for i, mode in enumerate(s_eofs.coords['mode'].values.tolist()):
            ax = axes[i]
            data = s_eofs.sel(mode=mode)
            
            im = ax.pcolormesh(
                s_eofs.lon, s_eofs.lat, data, cmap="RdBu_r", norm=norm, 
                transform=ccrs.PlateCarree()
            )

            ax.coastlines()
            ax.add_feature(cfeature.LAND, edgecolor="black")
            ax.add_feature(cfeature.OCEAN, facecolor="lightblue")
            ax.set_title(f"Mode {mode} -- Explained variance {round(s_expvar[i], 2) * 100}%")
        
        for j in range(n_modes, len(axes)):
            fig.delaxes(axes[j])
        
        bottom_margin = 0.1 + 0.075 * n_rows
        cbar = fig.colorbar(im, ax=axes, orientation="horizontal", shrink=0.5, aspect=40, pad=0.1)
        cbar.set_label('EOF Values')
        fig.suptitle("EOF Modes", fontsize=16)
        plt.tight_layout()
        fig.subplots_adjust(top=0.9, bottom=bottom_margin)
        plt.show()