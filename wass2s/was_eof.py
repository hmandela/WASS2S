import matplotlib.pyplot as plt
import cartopy.crs as ccrs
import cartopy.feature as cfeature
from matplotlib import colors
import xarray as xr 
import numpy as np
import pandas as pd
from scipy import stats
import xeofs as xe
import scipy.signal as sig

### Complete WAS_EOF  with multiple eof zone!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
class WAS_EOF:
    """
    A class for performing Empirical Orthogonal Function (EOF) analysis using the xeofs package, 
    with additional options for detrending and cosine latitude weighting.

    Parameters
    ----------
    n_modes : int, optional
        The number of EOF modes to retain. If None, the number of modes is determined by 
        explained variance.
    use_coslat : bool, optional
        If True, applies cosine latitude weighting to account for the Earth's spherical geometry.
    standardize : bool, optional
        If True, standardizes the input data by removing the mean and dividing by the standard deviation.
    detrend : bool, optional
        If True, detrends the input data along the time dimension before performing EOF analysis.
    opti_explained_variance : float, optional
        The target cumulative explained variance (in percent) to determine the optimal number of EOF modes.
    L2norm : bool, optional
        If True, normalizes the components and scores to have L2 norm.

    Attributes
    ----------
    model : xeofs.models.EOF
        The EOF model fitted to the predictor data.
    """

    def __init__(self, n_modes=None, use_coslat=True, standardize=False,
                  opti_explained_variance=None, L2norm=True):
        self.n_modes = n_modes
        self.use_coslat = use_coslat
        self.standardize = standardize
        self.opti_explained_variance = opti_explained_variance
        self.L2norm = L2norm
        self.model = None

    def fit(self, predictor, dim="T", clim_year_start=None, clim_year_end=None):
        predictor = predictor.fillna(predictor.mean(dim="T", skipna=True))
        predictor = predictor.rename({"X": "lon", "Y": "lat"})

        if self.n_modes is not None:
            self.model = xe.single.EOF(n_modes=self.n_modes, use_coslat=self.use_coslat, standardize=self.standardize)
        else:
            self.model = xe.single.EOF(n_modes=self.n_modes, use_coslat=self.use_coslat, standardize=self.standardize)
            self.model.fit(predictor, dim=dim)

            if self.opti_explained_variance is not None:
                npcs = 0
                sum_explain_var = 0
                while sum_explain_var * 100 < self.opti_explained_variance:
                    npcs += 1
                    sum_explain_var = sum(self.model.explained_variance_ratio()[:npcs])
                self.model = xe.single.EOF(n_modes=npcs, use_coslat=self.use_coslat, standardize=self.standardize)

        self.model.fit(predictor, dim=dim)

        s_eofs = self.model.components(normalized=self.L2norm)
        s_pcs = self.model.scores(normalized=self.L2norm)
        s_expvar = self.model.explained_variance_ratio()
        s_sing_values = self.model.singular_values()

        return s_eofs, s_pcs, s_expvar, s_sing_values

    def transform(self, predictor):
        predictor = predictor.rename({"X": "lon", "Y": "lat"})

        if self.model is None:
            raise ValueError("The model has not been fitted yet.")

        return self.model.transform(predictor, normalized=self.L2norm)

    def inverse_transform(self, pcs):
        if self.model is None:
            raise ValueError("The model has not been fitted yet.")

        return self.model.inverse_transform(pcs, normalized=self.L2norm)

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
