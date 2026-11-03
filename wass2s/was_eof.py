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


class EOFRankError(ValueError):
    """Training predictors contain no numerically resolved temporal variation."""


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
                 normalize=None, random_state=None):
        self.n_modes = n_modes
        self.use_coslat = use_coslat
        self.standardize = standardize
        self.opti_explained_variance = opti_explained_variance
        self.detrend = detrend
        self.L2norm = L2norm
        # normalize: equal total weighted anomaly variance per field. This
        # balances differing units AND grid sizes; static spatial climatology
        # is excluded. None enables it for a list of fields only. Cellwise
        # standardization, when requested, happens before this field scaling.
        self.normalize = normalize
        self.random_state = random_state

        self.model = None
        self.is_list = False
        self._stats = None            # per-field fold-safe statistics
        self._do_normalize = False
        self.trend_coeffs = None
        self.trend_meta = None
        self.train_mean = None        # stored training climatology (single-field)
        self.time_dim = None
        self.n_modes_ = None          # fitted count after numerical-rank filter
        self.rank_tolerance_ = None

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
            x_days = (x - x.isel({dim: 0})) / np.timedelta64(1, "D")
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
        # An unavailable trend must not erase an otherwise usable short series.
        # Keep these cells undetrended and store the same zero trend for apply.
        coeffs["polyfit_coefficients"] = coeffs.polyfit_coefficients.fillna(0.0)
        trend = trend.fillna(0.0)
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
            x_days = (x - x0) / np.timedelta64(1, "D")
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
        da = self._rename_in(da, dim).astype(np.float64)
        da = da.where(np.isfinite(da))

        train_mean = da.mean(dim=dim, skipna=True) if fitting else stats["mean"]

        coeffs = meta = None
        if self.detrend:
            if fitting:
                # Fit and count observations BEFORE imputation. Filling first
                # artificially met min_valid and biased the fitted slope.
                da, coeffs, meta = self._detrended_da(da, dim=dim)
            else:
                coeffs, meta = stats["coeffs"], stats["meta"]
                da = da - self._apply_detrend(da, coeffs, meta)
        fill_mean = da.mean(dim=dim, skipna=True).fillna(0.0) if fitting else stats["fill_mean"]
        da = da.fillna(fill_mean)

        if fitting:
            cell_std = da.std(dim=dim)
            # Remove exact constants and roundoff left by detrending an exact
            # line BEFORE scaling; otherwise standardization promotes noise.
            scale = abs(da).max(dim=dim)
            if self.detrend:
                scale = xr.apply_ufunc(np.maximum, scale, abs(train_mean).fillna(0.0))
                if coeffs is not None:
                    trend_scale = abs(self._apply_detrend(da, coeffs, meta)).max(dim=dim)
                    scale = xr.apply_ufunc(np.maximum, scale, trend_scale)
            tolerance = np.finfo(np.float64).eps * max(2, da.sizes[dim]) * scale
            active = np.isfinite(cell_std) & (cell_std > tolerance)
            cell_std = xr.where(active & self.standardize, cell_std, 1.0)
        else:
            cell_std, active = stats["cell_std"], stats["active"]
        da = xr.where(active, da, fill_mean) / cell_std

        if self._do_normalize:
            if fitting:
                anomaly_variance = da.var(dim=dim)
                if self.use_coslat and "lat" in da.coords:
                    anomaly_variance = anomaly_variance * np.cos(np.deg2rad(da.lat)).clip(min=0.0)
                norm = float(np.sqrt(anomaly_variance.sum(skipna=True)))
                if (not np.isfinite(norm)) or norm == 0.0:
                    norm = 1.0
            else:
                norm = stats["norm"]
            da = da / norm
        else:
            norm = 1.0

        out = stats if not fitting else {"mean": train_mean, "coeffs": coeffs,
                                         "meta": meta, "norm": norm,
                                         "fill_mean": fill_mean, "cell_std": cell_std,
                                         "active": active}
        return da, out

    # ------------------------------------------------------------------- fit
    def fit(self, predictor, dim="T", clim_year_start=None, clim_year_end=None):
        # A failed refit must not leave a previous fitted EOF usable.
        self.model = None
        self.n_modes_ = None
        fields = list(predictor) if isinstance(predictor, (list, tuple)) else [predictor]
        if not fields:
            raise ValueError("At least one predictor field is required.")
        fields = [self._rename_in(f, dim) for f in fields]
        if any(dim not in f.dims for f in fields):
            raise ValueError(f"Every predictor must have dimension {dim!r}.")
        if fields[0].sizes[dim] < 2:
            raise ValueError("EOF analysis requires at least two training samples.")
        for f in fields[1:]:
            if not f[dim].equals(fields[0][dim]):
                raise ValueError("Multivariate EOF fields must share identical training times.")
        predictor = fields if isinstance(predictor, (list, tuple)) else fields[0]
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

        n_features = sum(int(np.prod([f.sizes[d] for d in f.dims if d != dim])) for f in fields)
        rank_cap = min(nT - 1, n_features)
        if not any(bool(st["active"].any()) for st in (self._stats if self.is_list else [self._stats])):
            raise EOFRankError("EOF training predictors have numerical rank zero after preprocessing.")
        if self.opti_explained_variance is not None:
            if not 0 < float(self.opti_explained_variance) <= 100:
                raise ValueError("opti_explained_variance must be in (0, 100].")
            # The full admissible spectrum is needed for a variance target;
            # a fixed budget of 50 silently failed on higher-rank predictors.
            candidate_modes = rank_cap
        else:
            candidate_modes = max(1, min(int(self.n_modes) if self.n_modes else 50, rank_cap))

        model = EOF(n_modes=candidate_modes, use_coslat=self.use_coslat, standardize=False,
                    random_state=getattr(self, 'random_state', None))
        model.fit(data_to_fit, dim=dim)
        singular_values = np.asarray(model.singular_values(), dtype=np.float64)
        self.rank_tolerance_ = (np.finfo(np.float64).eps * max(nT, n_features)
                                * float(np.max(singular_values)))
        keep = np.flatnonzero(np.isfinite(singular_values) & (singular_values > self.rank_tolerance_))
        if not keep.size:
            raise EOFRankError("EOF training predictors have numerical rank zero after preprocessing.")
        if self.opti_explained_variance is not None:
            fractions = np.asarray(model.explained_variance_ratio())[keep]
            cumulative = np.cumsum(fractions)
            target = float(self.opti_explained_variance) / 100.0
            # A relative rounding allowance avoids selecting a 91st component
            # for an exactly 90% spectrum; all numerical modes suffice at 100%.
            count = min(keep.size, int(np.searchsorted(cumulative + 1e-12, target)) + 1)
            if cumulative[-1] + 1e-10 < target:
                raise ValueError("Requested EOF explained variance is not reachable at numerical rank.")
            keep = keep[:count]
            self.n_modes = int(count)
        # xeofs keeps mode-indexed results in this public data container.
        # Truncate BEFORE its L2 division, including for subsequent transform.
        for key, value in list(model.data.items()):
            if "mode" in value.dims:
                model.data[key] = value.isel(mode=keep)
        self.model = model
        self.n_modes_ = int(keep.size)

        return (self.model.components(normalized=self.L2norm),
                self.model.scores(normalized=self.L2norm),
                self.model.explained_variance_ratio())

    # -------------------------------------------------------------- transform
    def transform(self, predictor, dim=None):
        if self.model is None:
            raise ValueError("Model not fitted.")
        dim = dim or self.time_dim

        if self.is_list:
            if not isinstance(predictor, (list, tuple)) or len(predictor) != len(self._stats):
                raise ValueError("Predictor fields must match the number and order used at fit.")
            fields = [self._prep_one(da, dim, stats=st)[0]
                      for da, st in zip(predictor, self._stats)]
            return self.model.transform(fields, normalized=self.L2norm)

        p, _ = self._prep_one(predictor, dim, stats=self._stats)
        return self.model.transform(p, normalized=self.L2norm)

    def inverse_transform(self, pcs, return_anomalies=False):
        if self.model is None:
            raise ValueError("Model not fitted.")
        reconstructed = self.model.inverse_transform(pcs, normalized=self.L2norm)
        def restore(field, stats):
            field = field * stats["norm"] * stats["cell_std"]
            if self.detrend and stats["coeffs"] is not None and not return_anomalies:
                field = field + self._apply_detrend(field, stats["coeffs"], stats["meta"])
            return field
        if self.is_list:
            return [restore(f, st) for f, st in zip(reconstructed, self._stats)]
        return restore(reconstructed, self._stats)

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
