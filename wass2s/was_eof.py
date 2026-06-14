"""Leakage-free EOF analysis for spatial predictors.

Wraps the xeofs ``EOF`` class with fold-safe preprocessing (NaN fill,
optional linear detrending, optional per-field normalization) so that all
statistics are estimated on the training fold and applied to the test fold
without information leakage.

Class
-----
WAS_EOF
    EOF analysis for single-field or multivariate (list) predictors.
    Key methods: ``fit``, ``transform``, ``inverse_transform``,
    ``plot_EOF``.
"""
import numpy as np
import xarray as xr

from scipy.stats import norm, lognorm, expon, gamma, weibull_min, t, poisson, nbinom
from scipy.optimize import fsolve
from scipy.special import gamma as gamma_function

from xeofs.single import EOF


# ===========================================================================
#  WAS_EOF  (corrected: fold-safe, mirrors the WAS_CCA correction)
# ===========================================================================
class WAS_EOF:
    """
    EOF analysis with leakage-free preprocessing.

    Corrections relative to my previous version
    -------------------------------------------
    1. The training-fold time-mean is STORED at fit and reused to fill NaNs in
       transform(). Previously the test fold was filled implicitly (or with its
       own statistics), which is the same leakage WAS_CCA_base fixed with
       `_fill_test_data`.
    2. The linear trend is fitted on the training fold and the SAME coefficients
       are evaluated at the test/forecast times (already the intent, now made
       robust and always paired with the stored fill).
    3. Variance optimization (opti_explained_variance) is done on the training
       fold only.
    4. Dimension renaming is reversible (components can be mapped back to Y/X).
    """

    def __init__(self, n_modes=None, use_coslat=True, standardize=False,
                 opti_explained_variance=None, detrend=True, L2norm=True,
                 normalize=None):
        self.n_modes = n_modes
        self.use_coslat = use_coslat
        self.standardize = standardize
        self.opti_explained_variance = opti_explained_variance
        self.detrend = detrend
        self.L2norm = L2norm
        # normalize: per-field scalar rescaling so no field dominates the joint
        # covariance. None = auto (on for a list of fields, off for a single
        # field, where it is a no-op for the regression anyway).
        self.normalize = normalize

        self.model = None
        self.is_list = False
        self._stats = None            # per-field fold-safe statistics
        self._do_normalize = False
        self.trend_coeffs = None
        self.trend_meta = None
        self.train_mean = None        # stored training climatology (single-field)
        self.time_dim = None

    # ------------------------------------------------------------------ utils
    @staticmethod
    def _rename_in(da, dim):
        """WAS dims (T, Y, X) -> xeofs dims (dim, lat, lon)."""
        rename = {}
        if "X" in da.dims and "lon" not in da.dims:
            rename["X"] = "lon"
        if "Y" in da.dims and "lat" not in da.dims:
            rename["Y"] = "lat"
        if "T" in da.dims and dim not in da.dims:
            rename["T"] = dim
        if "time" in da.dims and dim not in da.dims and dim != "time":
            rename["time"] = dim
        return da.rename(rename) if rename else da

    @staticmethod
    def _rename_out(da):
        """xeofs dims (lat, lon) -> WAS dims (Y, X)."""
        rename = {}
        if "lat" in da.dims:
            rename["lat"] = "Y"
        if "lon" in da.dims:
            rename["lon"] = "X"
        return da.rename(rename) if rename else da

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

        meta = {"dim": dim, "x0": x0, "type": x_type, "min_valid": int(min_valid)}
        return da_detrended, coeffs, meta

    def _apply_detrend(self, da, coeffs, meta):
        dim = meta.get("dim", "T")
        if dim not in da.dims:
            if "T" in da.dims:
                dim = "T"
            elif "time" in da.dims:
                dim = "time"
            else:
                raise ValueError(f"Cannot find time dim. Found: {da.dims}")

        x = da[dim]
        if meta.get("type") == "datetime":
            x0 = np.datetime64(meta["x0"])
            x_days = (x - x0).astype("timedelta64[D]").astype(np.float64)
        else:
            x_days = x.astype(np.float64)

        if not isinstance(x_days, xr.DataArray):
            x_days = xr.DataArray(x_days, dims=dim, coords={dim: da[dim]})

        return xr.polyval(x_days, coeffs.polyfit_coefficients)

    # --------------------------------------------------- per-field preprocess
    def _prep_one(self, da, dim, stats=None):
        """
        Fold-safe preprocessing of ONE field.

        stats is None  -> FIT mode: learn (mean / trend / norm) on this fold and
                          return them.
        stats given    -> APPLY mode: use the training-fold statistics, never
                          recompute from the test fold.
        """
        fitting = stats is None
        da = self._rename_in(da, dim)

        train_mean = da.mean(dim=dim, skipna=True) if fitting else stats["mean"]
        da = da.fillna(train_mean)

        coeffs = meta = None
        if self.detrend:
            if fitting:
                da, coeffs, meta = self._detrended_da(da, dim=dim)
            else:
                coeffs, meta = stats["coeffs"], stats["meta"]
                da = da - self._apply_detrend(da, coeffs, meta)
        da = da.fillna(0.0)

        if self._do_normalize:
            if fitting:
                norm = float(da.std(skipna=True))
                if (not np.isfinite(norm)) or norm == 0.0:
                    norm = 1.0
            else:
                norm = stats["norm"]
            da = da / norm
        else:
            norm = 1.0

        out = stats if not fitting else {"mean": train_mean, "coeffs": coeffs,
                                         "meta": meta, "norm": norm}
        return da, out

    # ------------------------------------------------------------------- fit
    def fit(self, predictor, dim="T", clim_year_start=None, clim_year_end=None):
        self.time_dim = dim
        self.is_list = isinstance(predictor, (list, tuple))
        # per-field normalization: auto-on for a list, off for a single field
        self._do_normalize = self.normalize if self.normalize is not None else self.is_list

        if self.is_list:
            self._stats, data_to_fit = [], []
            for da in predictor:
                p, st = self._prep_one(da, dim, stats=None)
                data_to_fit.append(p)
                self._stats.append(st)
            nT = predictor[0].sizes[dim]
            self.train_mean = self.trend_coeffs = self.trend_meta = None
        else:
            data_to_fit, self._stats = self._prep_one(predictor, dim, stats=None)
            nT = predictor.sizes[dim]
            self.train_mean = self._stats["mean"]
            self.trend_coeffs = self._stats["coeffs"]
            self.trend_meta = self._stats["meta"]

        if self.opti_explained_variance is not None:
            tmp = EOF(n_modes=min(50, nT - 1), use_coslat=self.use_coslat, standardize=self.standardize)
            tmp.fit(data_to_fit, dim=dim)
            cum = tmp.explained_variance_ratio().cumsum()
            self.n_modes = int(np.searchsorted(cum.values * 100.0, self.opti_explained_variance) + 1)

        final_modes = int(self.n_modes) if self.n_modes else 50
        final_modes = max(1, min(final_modes, nT - 1))

        self.model = EOF(n_modes=final_modes, use_coslat=self.use_coslat, standardize=self.standardize)
        self.model.fit(data_to_fit, dim=dim)

        return (self.model.components(normalized=self.L2norm),
                self.model.scores(normalized=self.L2norm),
                self.model.explained_variance_ratio())

    # -------------------------------------------------------------- transform
    def transform(self, predictor, dim=None):
        if self.model is None:
            raise ValueError("Model not fitted.")
        dim = dim or self.time_dim

        if self.is_list:
            fields = [self._prep_one(da, dim, stats=st)[0]
                      for da, st in zip(predictor, self._stats)]
            return self.model.transform(fields, normalized=self.L2norm)

        p, _ = self._prep_one(predictor, dim, stats=self._stats)
        return self.model.transform(p, normalized=self.L2norm)

    def inverse_transform(self, pcs, return_anomalies=False):
        if self.model is None:
            raise ValueError("Model not fitted.")
        reconstructed = self.model.inverse_transform(pcs, normalized=self.L2norm)
        if self.detrend and (self.trend_coeffs is not None) and (not return_anomalies):
            reconstructed = reconstructed + self._apply_detrend(reconstructed, self.trend_coeffs, self.trend_meta)
        return reconstructed

    # ------------------------------------------------------------ visualization
    def plot_EOF(self, s_eofs=None, s_expvar=None, cmap="RdBu_r", n_cols=3):
        """
        Plot EOF spatial patterns and their explained variance.

        Works for a single field AND for a multivariate (list) fit: in the list
        case `components()` returns one pattern per field, and each is drawn in
        its own figure (the fields may live on different grids).

        Parameters
        ----------
        s_eofs : xarray.DataArray or list of DataArray, optional
            EOF spatial patterns. If None, taken from the fitted model, so you
            can simply call ``eof.plot_EOF()`` after ``fit``.
        s_expvar : array-like, optional
            Explained-variance ratio per mode (shared across fields for a
            combined EOF). If None, taken from the fitted model.
        cmap : str
            Diverging colormap (default 'RdBu_r').
        n_cols : int
            Mode columns per figure row (default 3).
        """
        import matplotlib.pyplot as plt
        from matplotlib import colors
        import cartopy.crs as ccrs
        import cartopy.feature as cfeature

        if s_eofs is None or s_expvar is None:
            if self.model is None:
                raise ValueError("Model not fitted; call fit() or pass s_eofs/s_expvar.")
            if s_eofs is None:
                s_eofs = self.model.components(normalized=self.L2norm)
            if s_expvar is None:
                s_expvar = self.model.explained_variance_ratio()

        # explained variance is a fraction per mode; shared across fields
        expvar = np.asarray(getattr(s_expvar, "values", s_expvar)).ravel().tolist()
        fields = list(s_eofs) if isinstance(s_eofs, (list, tuple)) else [s_eofs]

        for idx, comp in enumerate(fields):
            if "X" in comp.dims:
                comp = comp.rename({"X": "lon"})
            if "Y" in comp.dims:
                comp = comp.rename({"Y": "lat"})

            modes = comp.coords["mode"].values.tolist()
            n_modes = len(modes)
            n_rows = (n_modes + n_cols - 1) // n_cols

            fig, axes = plt.subplots(
                n_rows, n_cols,
                figsize=(n_cols * 6, n_rows * 4),
                subplot_kw={"projection": ccrs.PlateCarree()},
                squeeze=False,
            )
            axes = axes.flatten()

            # symmetric diverging scale centred on 0 (EOF sign structure reads better)
            vmax = float(abs(comp).max())
            if (not np.isfinite(vmax)) or vmax == 0.0:
                vmax = 1.0
            norm = colors.Normalize(vmin=-vmax, vmax=vmax, clip=False)

            im = None
            for i, mode in enumerate(modes):
                ax = axes[i]
                data = comp.sel(mode=mode).transpose("lat", "lon")
                im = ax.pcolormesh(comp.lon, comp.lat, data, cmap=cmap, norm=norm,
                                   transform=ccrs.PlateCarree())
                ax.coastlines()
                ax.add_feature(cfeature.LAND, edgecolor="black")
                ax.add_feature(cfeature.OCEAN, facecolor="lightblue")
                ev = expvar[i] * 100.0 if i < len(expvar) else float("nan")
                ax.set_title(f"Mode {mode} -- explained variance {ev:.1f}%")

            for j in range(n_modes, len(axes)):
                fig.delaxes(axes[j])

            if im is not None:
                cbar = fig.colorbar(im, ax=list(axes[:n_modes]), orientation="horizontal",
                                    shrink=0.5, aspect=40, pad=0.1)
                cbar.set_label("EOF values")

            suffix = f" -- field {idx + 1}" if len(fields) > 1 else ""
            fig.suptitle(f"EOF Modes{suffix}", fontsize=16)
            plt.tight_layout()
            fig.subplots_adjust(top=0.9, bottom=0.1 + 0.075 * n_rows)
            plt.show()
