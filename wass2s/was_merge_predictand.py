"""Station–gridded merging methods for observed rainfall.

Combines sparse station observations with a gridded first-guess field
(e.g. CHIRPS, TAMSAT, or a model hindcast) using a suite of spatial
interpolation and machine-learning techniques.

Classes
-------
WAS_Merging
    First-generation merging class (legacy interface).  Implements simple
    bias adjustment, regression kriging, and neural-network kriging.
WAS_Merging_Extended
    Refactored merging class with a richer method suite, DEM support,
    leave-one-out cross-validation, and a unified comparison plot.

    Available methods:

    ``simple_bias_adjustment``
        Additive or multiplicative mean-bias correction at station points.
    ``regression_kriging``
        OLS regression on the gridded field + kriging of residuals.
    ``conditional_merging``
        Preserve gridded spatial structure while matching station totals.
    ``kriging_with_external_drift``
        Kriging with the gridded field as an external drift variable.
    ``optimal_interpolation``
        Gaussian-covariance optimal interpolation (Gandin 1963).
    ``barnes_interpolation``
        Multi-pass Barnes successive correction.
    ``random_forest_merging`` / ``xgboost_merging``
        Machine-learning regression with k-NN neighbourhood features.
    ``validate_merging_methods``
        Leave-one-out cross-validation across all methods.
"""
import warnings
import pandas as pd
import xarray as xr
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from pykrige.ok import OrdinaryKriging
from sklearn.model_selection import KFold
from sklearn.metrics import mean_squared_error
from sklearn.linear_model import LinearRegression
from sklearn.neural_network import MLPRegressor
from sklearn.model_selection import GridSearchCV
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.compose import TransformedTargetRegressor
from scipy.spatial import cKDTree
try:
    import cartopy.crs as ccrs
except ImportError:
    ccrs = None


class WAS_Merging:
    def __init__(
        self,
        df: pd.DataFrame,
        da: xr.Dataset,
        date_month_day: str = "08-01"
    ):

        self.df = df
        self.da = da
        self.date_month_day = date_month_day

    def adjust_duplicates(self, series: pd.Series, increment: float = 0.00001) -> pd.Series:
        """
        If any values in the Series repeat, nudge them by a tiny increment
        so that all are unique (to avoid indexing collisions).
        """
        counts = series.value_counts()
        for val, count in counts[counts > 1].items():
            duplicates = series[series == val].index
            for i, idx in enumerate(duplicates):
                series.at[idx] += increment * i
        return series

    def transform_cpt(self, df, missing_value=-999.0):
        """Parse CPT LAT/LON or CDT LON/LAT/ELEV metadata without moving gauges."""
        return _station_input_dataset(df, self.date_month_day, missing_value)


    def transform_cdt(self, df, missing_value=-999.0):
        """Parse daily CDT station series, preserving physical coordinates."""
        return _station_input_dataset(df, self.date_month_day, missing_value)


    def auto_select_kriging_parameters(
        self,
        df: pd.DataFrame,
        x_col: str = 'X',
        y_col: str = 'Y',
        z_col: str = 'residuals',
        variogram_models: list = None,
        nlags_range=range(3, 10),
        n_splits: int = 5,
        random_state: int = 42,
        verbose: bool = False,
        enable_plotting: bool = False
    ):
        """
        Automatically selects the best variogram_model and nlags for Ordinary Kriging 
        using cross-validation.

        Returns:
         - best_model (str)
         - best_nlags (int)
         - ok_best (OrdinaryKriging object)
         - results_df (pd.DataFrame)
        """
        if variogram_models is None:
            variogram_models = ['linear', 'power', 'gaussian', 'spherical', 'exponential']
        
        # Suppress warnings for cleaner output
        # Preserve warnings: failed or degenerate scientific fits must remain visible.
        
        kf = KFold(n_splits=n_splits, shuffle=True, random_state=random_state)
        best_score = np.inf
        best_model = None
        best_nlags = None
        results = []

        for model in variogram_models:
            for nlags in nlags_range:
                cv_errors = []
                for train_index, test_index in kf.split(df):
                    train, test = df.iloc[train_index], df.iloc[test_index]
                    try:
                        ok = OrdinaryKriging(
                            x=train[x_col],
                            y=train[y_col],
                            z=train[z_col],
                            variogram_model=model,
                            nlags=nlags,
                            verbose=False,
                            enable_plotting=False
                        )
                        z_pred, ss = ok.execute('points', test[x_col].values, test[y_col].values)
                        mse = mean_squared_error(test[z_col], z_pred)
                        cv_errors.append(mse)
                    except Exception as e:
                        # E.g., convergence issues
                        cv_errors.append(np.inf)
                        if verbose:
                            print(f"Exception for model {model} with nlags {nlags}: {e}")
                        break
                
                avg_error = np.mean(cv_errors)
                results.append({
                    'variogram_model': model,
                    'nlags': nlags,
                    'cv_mse': avg_error
                })

                if avg_error < best_score:
                    best_score = avg_error
                    best_model = model
                    best_nlags = nlags

                if verbose:
                    print(f"Model: {model}, nlags: {nlags}, CV MSE: {avg_error:.4f}")

        results_df = pd.DataFrame(results).sort_values(by='cv_mse').reset_index(drop=True)

        if verbose:
            print(f"\nBest Variogram Model: {best_model}")
            print(f"Best nlags: {best_nlags}")
            print(f"Best CV MSE: {best_score:.4f}")

        # Fit the best model on the entire dataset
        try:
            ok_best = OrdinaryKriging(
                x=df[x_col],
                y=df[y_col],
                z=df[z_col],
                variogram_model=best_model,
                nlags=best_nlags,
                verbose=verbose,
                enable_plotting=enable_plotting
            )
        except Exception as e:
            raise RuntimeError(f"Failed to fit the best Ordinary Kriging model: {e}")

        return best_model, best_nlags, ok_best, results_df

    def simple_bias_adjustment(self, missing_value=-999.0, do_cross_validation=False):
        """Numerically consistent implementation through the shared merging engine."""
        return _legacy_sba(self, missing_value, do_cross_validation)


    def regression_kriging(self, missing_value=-999.0, do_cross_validation=False):
        """Numerically consistent implementation through the shared merging engine."""
        return _legacy_rk(self, missing_value, do_cross_validation)


    def neural_network_kriging_(self, missing_value=-999.0, do_cross_validation=False):
        """Numerically consistent implementation through the shared merging engine."""
        return _legacy_nn(self, missing_value, do_cross_validation)


    def neural_network_kriging(self, missing_value=-999.0, do_cross_validation=False):
        """Numerically consistent implementation through the shared merging engine."""
        return _legacy_nn(self, missing_value, do_cross_validation)


    
    def multiplicative_bias(self, missing_value=-999.0, do_cross_validation=False):
        """Numerically consistent implementation through the shared merging engine."""
        return _legacy_multiplicative(self, missing_value, do_cross_validation)


    def plot_merging_comparaison(self, df_Obs, da_estimated, da_corrected, missing_value = -999.0):
        
        da_Obs = self.transform_cpt(df_Obs, missing_value=missing_value)
        
        da_estimated_ = da_estimated.interp(
            X=da_Obs.coords['X'],
            Y=da_Obs.coords['Y'],
        )
        da_estimated_.name = "Estimation"
        
        da_corrected_ = da_corrected.interp(
            X=da_Obs.coords['X'],
            Y=da_Obs.coords['Y'],
        )
        da_corrected_.name = "Correction"
            
        merged_df = pd.merge(
            pd.merge(da_Obs.to_dataframe(), 
                     da_estimated_.to_dataframe(), 
                     on=["T", "Y", "X"], 
                     how="outer"),
            da_corrected_.to_dataframe(),
            on=["T", "Y", "X"],
            how="outer"
        ).reset_index()
        
        df = merged_df.dropna(subset=["Observation",  "Estimation", "Correction"])
        
        # Create a 1-row, 2-column figure
        fig, axes = plt.subplots(1, 2, figsize=(12, 6), sharex=True, sharey=True)
        
        # Define limits for square axes
        x_min, x_max = df["Observation"].min(), df["Observation"].max()
        y_min, y_max = df["Observation"].min(), df["Observation"].max()
        # y_min = min(df["Estimation"].min(), df["Correction"].min())
        # y_max = max(df["Estimation"].max(), df["Correction"].max())
        
        # Set equal axis limits
        axes[0].set_xlim(x_min, x_max)
        axes[0].set_ylim(y_min, y_max)
        axes[1].set_xlim(x_min, x_max)
        axes[1].set_ylim(y_min, y_max)
        
        # Define y=x reference line range
        y_line = np.linspace(x_min, x_max, 100)
        
        # Panel 1: Scatterplot for Observations vs Estimated
        sns.scatterplot(data=df, x="Observation", y="Estimation", ax=axes[0], color="blue")
        axes[0].plot(y_line, y_line, linestyle="--", color="black", label="y = x")  # y=x line
        axes[0].set_title("Observation vs Estimation")
        axes[0].set_xlabel("Observation")
        axes[0].set_ylabel("Estimation")
        axes[0].set_aspect('equal', adjustable='box')  # Make square
        axes[0].legend()
        
        # Panel 2: Scatterplot for Observations vs Corrected
        sns.scatterplot(data=df, x="Observation", y="Correction", ax=axes[1], color="red")
        axes[1].plot(y_line, y_line, linestyle="--", color="black", label="y = x")  # y=x line
        axes[1].set_title("Observation vs Correction")
        axes[1].set_xlabel("Observation")
        axes[1].set_ylabel("Correction")
        axes[1].set_aspect('equal', adjustable='box')  # Make square
        axes[1].legend()
        
        # Adjust layout
        plt.tight_layout()
        plt.show()

        
import warnings
import numpy as np
import pandas as pd
import xarray as xr
try:
    import xgboost as xgb
except ImportError:
    xgb = None
import matplotlib.pyplot as plt
import seaborn as sns

from scipy.spatial.distance import cdist
from scipy.linalg import solve

from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import GridSearchCV, KFold
from sklearn.metrics import mean_squared_error
from sklearn.linear_model import LinearRegression

from pykrige.ok import OrdinaryKriging
from pykrige.uk import UniversalKriging

# Preserve warnings: failed or degenerate scientific fits must remain visible.


class WAS_Merging_Extended:
    """
    Merging class for station observations with gridded estimates,
    using geostatistical and machine learning methods.
    """

    def __init__(
        self,
        df: pd.DataFrame,
        da: xr.DataArray | xr.Dataset,
        dem: xr.DataArray | None = None,
        date_month_day: str = "08-01",
        varname: str | None = None,
        projection: str = "utm",      # "utm" or "laea"
        utm_zone: int | None = None,
        hemisphere: str | None = None,
        # kriging speed
        kriging_cv: bool = True,
        kriging_fixed: dict | None = None,  # {"variogram_model": "spherical", "nlags": 6}
        kriging_cache: bool = True,
        kriging_cache_key: str = "global",  # "global" or "per_time"
        # memory control
        max_grid_points_warn: int = 200_000,
        # verbose
        verbose: bool = True,
    ):
        self.df = df
        self.dem = dem
        self.date_month_day = date_month_day
        self.verbose = verbose

        self.da = self._coerce_to_dataarray(da, varname=varname)
        self._validate_grids()

        self.projection = projection.lower()
        self.utm_zone = utm_zone
        self.hemisphere = hemisphere

        self.kriging_cv = bool(kriging_cv)
        self.kriging_fixed = kriging_fixed
        self.kriging_cache = bool(kriging_cache)
        self.kriging_cache_key = kriging_cache_key

        self.max_grid_points_warn = int(max_grid_points_warn)

        self._to_proj = None
        self._to_geo = None

        self._grid_cache = None
        self._kriging_param_cache = {}
        self.unique_stations = None

        self._init_projection()
        self._cache_projected_grid()

    # --------------------- base guards ---------------------

    def _coerce_to_dataarray(self, da: xr.DataArray | xr.Dataset, varname: str | None) -> xr.DataArray:
        if isinstance(da, xr.DataArray):
            return da
        if not isinstance(da, xr.Dataset):
            raise TypeError("da must be an xarray.DataArray or xarray.Dataset")
        if varname is None:
            if len(da.data_vars) != 1:
                raise ValueError("da has multiple variables; set varname=...")
            varname = list(da.data_vars)[0]
        if varname not in da:
            raise KeyError(f"'{varname}' not found in dataset.")
        return da[varname]

    def _validate_grids(self):
        if set(self.da.dims) != {"T", "Y", "X"}:
            raise ValueError("da must have exactly the named dimensions T, Y, X.")
        for d in ("T", "Y", "X"):
            if not self.da.get_index(d).is_unique:
                raise ValueError(f"Coordinate {d!r} must be unique.")
        self.da = self.da.transpose("T", "Y", "X")
        if self.dem is not None:
            if not isinstance(self.dem, xr.DataArray) or set(self.dem.dims) != {"Y", "X"}:
                raise ValueError("dem must be a DataArray with exactly dimensions Y, X.")
            self.dem = self.dem.transpose("Y", "X").interp(Y=self.da.Y, X=self.da.X)


    def _mask_like_estim(self, estim):
        # Preserve missingness separately for every date, including transient gaps.
        return xr.where(np.isfinite(estim), 1.0, np.nan)


    def _expand_T(self, da2d: xr.DataArray, t) -> xr.DataArray:
        return da2d.expand_dims(T=[t])

    def _xy_stack(self, y, x):
        return np.column_stack([np.asarray(y, float), np.asarray(x, float)])


    def adjust_duplicates(self, series: pd.Series, increment: float = 0.00001) -> pd.Series:
        """
        If any values in the Series repeat, nudge them by a tiny increment
        so that all are unique (to avoid indexing collisions).
        """
        counts = series.value_counts()
        for val, count in counts[counts > 1].items():
            duplicates = series[series == val].index
            for i, idx in enumerate(duplicates):
                series.at[idx] += increment * i
        return series


    def transform_cpt(self, df, missing_value=-999.0):
        """Parse CPT LAT/LON or CDT LON/LAT/ELEV metadata without moving gauges."""
        return _station_input_dataset(df, self.date_month_day, missing_value)


    def transform_cdt(self, df, missing_value=-999.0):
        """Parse daily CDT station series, preserving physical coordinates."""
        return _station_input_dataset(df, self.date_month_day, missing_value)


    # --------------------- projection ---------------------

    def _init_projection(self):
        try:
            from pyproj import CRS, Transformer
        except Exception as e:
            raise ImportError("pyproj is required. Install: pip/conda install pyproj") from e

        lat0 = float(np.nanmean(self.da["Y"].values))
        lon0 = float(np.nanmean(self.da["X"].values))

        if self.projection == "utm":
            zone = self.utm_zone
            if zone is None:
                zone = int(np.floor((lon0 + 180.0) / 6.0) + 1)
                zone = max(1, min(zone, 60))
            hemi = self.hemisphere
            if hemi is None:
                hemi = "north" if lat0 >= 0 else "south"
            epsg = 32600 + zone if hemi == "north" else 32700 + zone
            crs_proj = CRS.from_epsg(epsg)

        elif self.projection == "laea":
            proj4 = f"+proj=laea +lat_0={lat0} +lon_0={lon0} +datum=WGS84 +units=m +no_defs"
            crs_proj = CRS.from_proj4(proj4)

        else:
            raise ValueError("projection must be 'utm' or 'laea'.")

        crs_geo = CRS.from_epsg(4326)
        self._to_proj = Transformer.from_crs(crs_geo, crs_proj, always_xy=True)
        self._to_geo = Transformer.from_crs(crs_proj, crs_geo, always_xy=True)

    def _cache_projected_grid(self):
        lon = self.da["X"].values
        lat = self.da["Y"].values
        gx, gy = np.meshgrid(lon, lat)  # lon2d, lat2d (deg)
        xp, yp = self._to_proj.transform(gx, gy)  # meters
        self._grid_cache = {
            "gx": gx, "gy": gy,
            "xp": xp, "yp": yp,
            "xp_flat": xp.ravel().astype(float),
            "yp_flat": yp.ravel().astype(float),
        }
        npts = self._grid_cache["xp_flat"].size
        if npts > self.max_grid_points_warn:
            warnings.warn(
                f"Grid has {npts:,} points; full distance matrices (OI/Barnes) may be heavy. "
                f"Use chunking options in OI/Barnes if needed.",
                RuntimeWarning
            )

    def _project_points(self, lon: np.ndarray, lat: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        xp, yp = self._to_proj.transform(np.asarray(lon, float), np.asarray(lat, float))
        return np.asarray(xp, float), np.asarray(yp, float)

    # --------------------- station sampling ---------------------
    # obs is a Dataset with variable "Observation"

    def _station_df_for_time(self, obs: xr.Dataset, estim: xr.DataArray, t, add_dem: bool = False) -> pd.DataFrame:
        df = obs.sel(T=t).to_dataframe().reset_index().dropna()
        if df.empty:
            return df

        xp = xr.DataArray(df["X"].values, dims="points")
        yp = xr.DataArray(df["Y"].values, dims="points")
        est_pts = estim.sel(T=t).interp(X=xp, Y=yp)
        df["Estimation"] = np.asarray(est_pts.values, dtype=float)

        if add_dem and (self.dem is not None):
            dem_pts = self.dem.interp(X=xp, Y=yp)
            df["DEM"] = np.asarray(dem_pts.values, dtype=float)

        Xp, Yp = self._project_points(df["X"].values, df["Y"].values)
        df["Xp"] = Xp
        df["Yp"] = Yp

        return df.replace([np.inf, -np.inf], np.nan).dropna(subset=["Observation", "Estimation", "Xp", "Yp"])

    # --------------------- kriging selection + cache ---------------------

    def _kriging_cache_id(self, method: str, t=None) -> str:
        if not self.kriging_cache:
            return ""
        if self.kriging_cache_key == "per_time" and t is not None:
            return f"{method}:{pd.to_datetime(t).date()}"
        return f"{method}:global"

    def _fit_ok(self, x, y, z, variogram_model, nlags):
        x, y, z = (np.asarray(v, float) for v in (x, y, z))
        if not (np.isfinite(x).all() and np.isfinite(y).all() and np.isfinite(z).all()):
            raise ValueError("Kriging inputs must be finite.")
        # Zero residual variance is a valid limiting case, not an optimizer failure.
        if np.ptp(z) <= np.finfo(float).eps * max(1.0, np.max(np.abs(z))) * 16:
            return _ConstantKriging(float(np.mean(z)))
        return OrdinaryKriging(x, y, z, variogram_model=variogram_model,
                               nlags=int(nlags), enable_plotting=False, verbose=False,
                               pseudo_inv=True)


    def auto_select_kriging_parameters(self, *args, **kwargs):
        """Select and fit the variogram using the shared spatial CV implementation."""
        return _auto_select_kriging_parameters_scientific(self, *args, **kwargs)


    def _krige_grid_points(self, df_planar: pd.DataFrame, z_col: str, estim: xr.DataArray, method_key: str, t=None):
        ok, info = self.auto_select_kriging_parameters(df_planar, method_key=method_key, t=t, z_col=z_col)
        xp = self._grid_cache["xp_flat"]
        yp = self._grid_cache["yp_flat"]
        z_pred_flat, _ = ok.execute("points", xp, yp)
        z_pred = np.asarray(z_pred_flat, float).reshape(estim.sizes["Y"], estim.sizes["X"])
        return z_pred, info

    # --------------------- merging methods ---------------------
    

    def simple_bias_adjustment(self, missing_value=-999.0):
        obs = self.transform_cpt(self.df, missing_value)
        estim = self.da
        mask = self._mask_like_estim(estim)

        t_common = np.intersect1d(obs["T"].values, estim["T"].values)
        out = []
        diag = {"method": "SBA", "n_times": 0}

        for t in t_common:
            m_df = self._station_df_for_time(obs, estim, t, add_dem=False)
            if m_df.empty:
                continue

            if self.verbose:
                print(f"\nYear = {pd.to_datetime(t).year}, number in-situ = {m_df.shape[0]}")

            m_df["residuals"] = m_df["Observation"] - m_df["Estimation"]
            try:
                z_res, info = self._krige_grid_points(m_df, "residuals", estim, method_key="SBA", t=t)
            except Exception:
                continue
            res2d = estim.sel(T=t) + xr.DataArray(z_res, coords=[estim["Y"], estim["X"]], dims=["Y", "X"])
            out.append(self._expand_T(np.maximum(res2d, 0), t))
            diag["n_times"] += 1

        if not out:
            return xr.DataArray(np.nan), diag
        return xr.concat(out, dim="T") * mask, diag

    def regression_kriging(self, missing_value=-999.0, use_dem: bool = False, dem_fill: str = "mask"):
        obs = self.transform_cpt(self.df, missing_value)
        estim = self.da
        mask = self._mask_like_estim(estim)

        t_common = np.intersect1d(obs["T"].values, estim["T"].values)
        out = []
        diag = {"method": "RK", "n_times": 0, "use_dem": bool(use_dem), "dem_fill": dem_fill}

        dem2d = self.dem if (use_dem and self.dem is not None) else None
        if dem2d is not None and dem_fill == "mean":
            dem2d = dem2d.fillna(float(dem2d.mean().values))

        for t in t_common:
            m_df = self._station_df_for_time(obs, estim, t, add_dem=use_dem)
            if len(m_df) < 5:
                continue

            if self.verbose:
                print(f"\nYear = {pd.to_datetime(t).year}, number in-situ = {m_df.shape[0]}")

            if use_dem and ("DEM" in m_df.columns) and (dem2d is not None):
                if dem_fill == "mean":
                    m_df["DEM"] = m_df["DEM"].fillna(float(np.nanmean(m_df["DEM"].values)))
                else:
                    m_df = m_df.dropna(subset=["DEM"])
                if len(m_df) < 5:
                    continue

                reg = LinearRegression().fit(m_df[["Estimation", "DEM"]].values, m_df["Observation"].values)

                
                estim_t = estim.sel(T=t).values
                dem_t = dem2d.values
                X_grid = np.column_stack([estim_t.ravel(), dem_t.ravel()])
                valid_grid = np.isfinite(X_grid).all(axis=1)
                trend = np.full(len(X_grid), np.nan)
                trend[valid_grid] = reg.predict(X_grid[valid_grid])
                trend = trend.reshape(estim.sizes["Y"], estim.sizes["X"])

                m_df["residuals"] = m_df["Observation"] - reg.predict(m_df[["Estimation", "DEM"]].values)
            else:
                reg = LinearRegression().fit(m_df[["Estimation"]].values, m_df["Observation"].values)

                estim_t = estim.sel(T=t).values
                estim_t_no_nan = np.nan_to_num(estim_t, nan=0.0)
                trend = reg.predict(estim_t_no_nan.reshape(-1, 1)).reshape(estim.sizes["Y"], estim.sizes["X"])
                m_df["residuals"] = m_df["Observation"] - reg.predict(m_df[["Estimation"]].values)

            # Inner variogram CV must refit the regression as well as kriging.
            m_df.attrs["residual_model"] = "RK"
            m_df.attrs["residual_features"] = ["Estimation", "DEM"] if reg.n_features_in_ == 2 else ["Estimation"]
            try:
                z_res, info = self._krige_grid_points(m_df, "residuals", estim, method_key="RK", t=t)
            except Exception:
                continue

            res2d = xr.DataArray(trend + z_res, coords=[estim["Y"], estim["X"]], dims=["Y", "X"])
            out.append(self._expand_T(np.maximum(res2d, 0), t))
            diag["n_times"] += 1

        if not out:
            return xr.DataArray(np.nan), diag
        return xr.concat(out, dim="T") * mask, diag

    def conditional_merging(self, missing_value=-999.0):
        obs = self.transform_cpt(self.df, missing_value)
        estim = self.da
        mask = self._mask_like_estim(estim)

        t_common = np.intersect1d(obs["T"].values, estim["T"].values)
        out = []
        diag = {"method": "CM", "n_times": 0}

        for t in t_common:
            m_df = self._station_df_for_time(obs, estim, t, add_dem=False)
            if len(m_df) < 5:
                continue

            if self.verbose:
                print(f"\nYear = {pd.to_datetime(t).year}, number in-situ = {m_df.shape[0]}")

            try:
                z_obs, _ = self._krige_grid_points(m_df, "Observation", estim, method_key="CM_obs", t=t)
                z_est, _ = self._krige_grid_points(m_df, "Estimation", estim, method_key="CM_est", t=t)
            except Exception:
                continue
            cm_field = z_obs + (estim.sel(T=t).values - z_est)
            res2d = xr.DataArray(cm_field, coords=[estim["Y"], estim["X"]], dims=["Y", "X"])
            out.append(self._expand_T(np.maximum(res2d, 0), t))
            diag["n_times"] += 1

        if not out:
            return xr.DataArray(np.nan), diag
        return xr.concat(out, dim="T") * mask, diag

    def kriging_with_external_drift(
        self,
        missing_value=-999.0,
        variogram_model="linear",
        drift_terms: tuple[str, ...] = ("Estimation",),  # subset of ("Estimation","DEM")
    ):
        obs = self.transform_cpt(self.df, missing_value)
        estim = self.da
        mask = self._mask_like_estim(estim)

        if ("DEM" in drift_terms) and (self.dem is None):
            raise ValueError("DEM drift requested but dem is None.")

        t_common = np.intersect1d(obs["T"].values, estim["T"].values)
        out = []
        diag = {"method": "KED", "n_times": 0, "variogram_model": variogram_model, "drift_terms": drift_terms}

        xp_grid = self._grid_cache["xp_flat"]
        yp_grid = self._grid_cache["yp_flat"]

        for t in t_common:
            m_df = self._station_df_for_time(obs, estim, t, add_dem=("DEM" in drift_terms))
            if len(m_df) < 5:
                continue

            if self.verbose:
                print(f"\nYear = {pd.to_datetime(t).year}, number in-situ = {m_df.shape[0]}")

            if "DEM" in drift_terms:
                m_df = m_df.dropna(subset=["DEM"])
            if len(m_df) < 5:
                continue
            specified_drift = []
            if "Estimation" in drift_terms:
                specified_drift.append(m_df["Estimation"].values.astype(float))
            if "DEM" in drift_terms:
                specified_drift.append(m_df["DEM"].values.astype(float))

            try:
                uk = UniversalKriging(
                    m_df["Xp"].values.astype(float),
                    m_df["Yp"].values.astype(float),
                    m_df["Observation"].values.astype(float),
                    variogram_model=variogram_model,
                    drift_terms=["specified"],
                    specified_drift=specified_drift,
                    enable_plotting=False,
                    verbose=False,
                )

                grid_drifts = []
                if "Estimation" in drift_terms:
                    grid_drifts.append(estim.sel(T=t).values.ravel().astype(float))
                if "DEM" in drift_terms:
                    grid_drifts.append(self.dem.values.ravel().astype(float))

                valid = np.isfinite(estim.sel(T=t).values.ravel())
                for drift in grid_drifts:
                    valid &= np.isfinite(drift)
                z_pred_flat = np.full(xp_grid.shape, np.nan)
                if valid.any():
                    zp, _ = uk.execute("points", xp_grid[valid], yp_grid[valid],
                        specified_drift_arrays=[v[valid] for v in grid_drifts])
                    z_pred_flat[valid] = np.asarray(zp, float)
                z_pred = z_pred_flat.reshape(estim.sizes["Y"], estim.sizes["X"])
            except Exception as exc:
                warnings.warn(f"KED failed for {t}: {exc}", RuntimeWarning)
                z_pred = np.full_like(estim.sel(T=t).values, np.nan, dtype=float)

            res2d = xr.DataArray(z_pred, coords=[estim["Y"], estim["X"]], dims=["Y", "X"])
            out.append(self._expand_T(np.maximum(res2d, 0), t))
            diag["n_times"] += 1

        if not out:
            return xr.DataArray(np.nan), diag
        return xr.concat(out, dim="T") * mask, diag

    def optimal_interpolation(self, missing_value=-999.0, L_m: float = 150_000.0, epsilon: float = 0.1, chunk_size: int | None = None):
        if L_m <= 0 or epsilon < 0:
            raise ValueError("Invalid positive spatial scale or regularization parameter.")
        if chunk_size is not None and int(chunk_size) < 1:
            raise ValueError("chunk_size must be positive.")
        obs = self.transform_cpt(self.df, missing_value)
        estim = self.da
        mask = self._mask_like_estim(estim)

        grid_xy = self._xy_stack(self._grid_cache["yp_flat"], self._grid_cache["xp_flat"])
        t_common = np.intersect1d(obs["T"].values, estim["T"].values)

        out = []
        diag = {"method": "OI", "n_times": 0, "L_m": float(L_m), "epsilon": float(epsilon), "chunk_size": chunk_size}

        for t in t_common:
            m_df = self._station_df_for_time(obs, estim, t, add_dem=False)
            if len(m_df) < 5:
                continue

            if self.verbose:
                print(f"\nYear = {pd.to_datetime(t).year}, number in-situ = {m_df.shape[0]}")

            stn_xy = self._xy_stack(m_df["Yp"].values, m_df["Xp"].values)
            residuals = (m_df["Observation"].values - m_df["Estimation"].values).astype(float)

            dist_stn = cdist(stn_xy, stn_xy)
            C = np.exp(-(dist_stn**2) / (2.0 * (float(L_m) ** 2))) + float(epsilon) * np.eye(len(stn_xy))

            try:
                w = solve(C, residuals, assume_a="pos")
            except Exception:
                continue

            z_pred_flat = np.empty(grid_xy.shape[0], dtype=float)
            if chunk_size is None:
                dist_grid = cdist(grid_xy, stn_xy)
                Wg = np.exp(-(dist_grid**2) / (2.0 * (float(L_m) ** 2)))
                z_pred_flat[:] = Wg @ w
            else:
                cs = int(chunk_size)
                for i0 in range(0, grid_xy.shape[0], cs):
                    i1 = min(i0 + cs, grid_xy.shape[0])
                    dist_grid = cdist(grid_xy[i0:i1], stn_xy)
                    Wg = np.exp(-(dist_grid**2) / (2.0 * (float(L_m) ** 2)))
                    z_pred_flat[i0:i1] = Wg @ w

            z_pred = z_pred_flat.reshape(estim.sizes["Y"], estim.sizes["X"])
            res2d = estim.sel(T=t) + xr.DataArray(z_pred, coords=[estim["Y"], estim["X"]], dims=["Y", "X"])
            out.append(self._expand_T(np.maximum(res2d, 0), t))
            diag["n_times"] += 1

        if not out:
            return xr.DataArray(np.nan), diag
        return xr.concat(out, dim="T") * mask, diag

    def barnes_interpolation(self, missing_value=-999.0, kappa_m: float = 200_000.0, gamma: float = 0.3, chunk_size: int | None = None):
        if kappa_m <= 0 or not 0 < gamma <= 1:
            raise ValueError("Invalid positive spatial scale or regularization parameter.")
        if chunk_size is not None and int(chunk_size) < 1:
            raise ValueError("chunk_size must be positive.")
        obs = self.transform_cpt(self.df, missing_value)
        estim = self.da
        mask = self._mask_like_estim(estim)

        grid_xy = self._xy_stack(self._grid_cache["yp_flat"], self._grid_cache["xp_flat"])
        t_common = np.intersect1d(obs["T"].values, estim["T"].values)

        out = []
        diag = {"method": "Barnes", "n_times": 0, "kappa_m": float(kappa_m), "gamma": float(gamma), "chunk_size": chunk_size}

        for t in t_common:
            m_df = self._station_df_for_time(obs, estim, t, add_dem=False)
            if len(m_df) < 5:
                continue

            if self.verbose:
                print(f"\nYear = {pd.to_datetime(t).year}, number in-situ = {m_df.shape[0]}")

            stn_xy = self._xy_stack(m_df["Yp"].values, m_df["Xp"].values)
            residuals = (m_df["Observation"].values - m_df["Estimation"].values).astype(float)

            dist_stn = cdist(stn_xy, stn_xy)
            w1s = np.exp(-(dist_stn**2) / (float(kappa_m) ** 2))
            s1s = np.sum(w1s, axis=1)
            s1s[s1s == 0] = 1e-12
            z1s = (w1s @ residuals) / s1s
            new_res = residuals - z1s

            z1_flat = np.empty(grid_xy.shape[0], dtype=float)
            z2_flat = np.empty(grid_xy.shape[0], dtype=float)

            def _barnes_pass(vals, scale_m2):
                outv = np.empty(grid_xy.shape[0], dtype=float)
                if chunk_size is None:
                    d = cdist(grid_xy, stn_xy)
                    w = np.exp(-(d**2) / scale_m2)
                    s = np.sum(w, axis=1)
                    s[s == 0] = 1e-12
                    outv[:] = (w @ vals) / s
                else:
                    cs = int(chunk_size)
                    for i0 in range(0, grid_xy.shape[0], cs):
                        i1 = min(i0 + cs, grid_xy.shape[0])
                        d = cdist(grid_xy[i0:i1], stn_xy)
                        w = np.exp(-(d**2) / scale_m2)
                        s = np.sum(w, axis=1)
                        s[s == 0] = 1e-12
                        outv[i0:i1] = (w @ vals) / s
                return outv

            z1_flat = _barnes_pass(residuals, float(kappa_m) ** 2)
            z2_flat = _barnes_pass(new_res, (float(gamma) * float(kappa_m)) ** 2)

            z_pred = (z1_flat + z2_flat).reshape(estim.sizes["Y"], estim.sizes["X"])
            res2d = estim.sel(T=t) + xr.DataArray(z_pred, coords=[estim["Y"], estim["X"]], dims=["Y", "X"])
            out.append(self._expand_T(np.maximum(res2d, 0), t))
            diag["n_times"] += 1

        if not out:
            return xr.DataArray(np.nan), diag
        return xr.concat(out, dim="T") * mask, diag

    # --------------------- ML (uses projected distances) -------------------------
    # obs is a Dataset

    def _prepare_ml(self, missing_value, k_nearest=10):
        obs = self.transform_cpt(self.df, missing_value)   # Dataset
        estim = self.da

        t_common = np.intersect1d(obs["T"].values, estim["T"].values)
        rows = []
        for t in t_common:
            df_t = obs.sel(T=t).to_dataframe().reset_index().dropna()
            if df_t.empty:
                continue
            # Add Estimation
            xp = xr.DataArray(df_t["X"].values, dims="points")
            yp = xr.DataArray(df_t["Y"].values, dims="points")
            est_pts = estim.sel(T=t).interp(X=xp, Y=yp)
            df_t["Estimation"] = np.asarray(est_pts.values, dtype=float)

            if self.dem is not None:
                dem_pts = self.dem.interp(X=xp, Y=yp)
                df_t["DEM"] = np.asarray(dem_pts.values, dtype=float)

            Xp, Yp = self._project_points(df_t["X"].values, df_t["Y"].values)
            df_t["Xp"] = Xp
            df_t["Yp"] = Yp

            df_t = df_t.dropna(subset=["Observation", "Estimation", "Xp", "Yp"])
            if not df_t.empty:
                df_t["T"] = t
                rows.append(df_t)

        if not rows:
            raise ValueError("No station samples after alignment.")

        g_df = pd.concat(rows, ignore_index=True)
        stn_unique = g_df[["Yp", "Xp"]].drop_duplicates().reset_index(drop=True)
        self.unique_stations = stn_unique.values.astype(float)

        feats = ["X", "Y", "Estimation"]
        if (self.dem is not None) and ("DEM" in g_df.columns):
            feats.append("DEM")

        # Distance features to the training network are deterministic covariates.
        # No rainfall values enter this feature construction.
        k = max(0, min(int(k_nearest), len(self.unique_stations)))
        if k:
            knn, _ = cKDTree(self.unique_stations).query(g_df[["Yp", "Xp"]].values, k=list(range(1, k + 1)))
            for i in range(k):
                col = f"DIST_{i+1}"
                g_df[col] = knn[:, i]
                feats.append(col)
        g_df = g_df.replace([np.inf, -np.inf], np.nan).dropna(subset=feats + ["Observation"])
        if g_df.empty:
            raise ValueError("No finite station training samples remain.")

        return g_df, feats, estim, obs, k

    def _predict_grid(self, model, estim, obs, feats, k_nearest):
        mask = self._mask_like_estim(estim)

        xp_grid = self._grid_cache["xp_flat"]
        yp_grid = self._grid_cache["yp_flat"]
        grid_xy = np.column_stack([yp_grid, xp_grid]).astype(float)

        k = max(0, min(int(k_nearest), len(self.unique_stations)))
        knn_grid = np.empty((len(grid_xy), 0))
        if k:
            knn_grid, _ = cKDTree(self.unique_stations).query(grid_xy, k=list(range(1, k + 1)))

        t_common = np.intersect1d(obs["T"].values, estim["T"].values)
        out = []

        for t in t_common:
            X_pred = pd.DataFrame(
                {
                    "X": self._grid_cache["gx"].ravel(),
                    "Y": self._grid_cache["gy"].ravel(),
                    "Estimation": estim.sel(T=t).values.ravel(),
                }
            )
            if self.dem is not None:
                X_pred["DEM"] = self.dem.values.ravel()

            for i in range(k):
                X_pred[f"DIST_{i+1}"] = knn_grid[:, i]

            valid = np.isfinite(X_pred[feats].to_numpy(float)).all(axis=1)
            flat = np.full(len(X_pred), np.nan)
            if valid.any():
                flat[valid] = model.predict(X_pred.loc[valid, feats])
            pred = flat.reshape(estim.sizes["Y"], estim.sizes["X"])
            da2d = xr.DataArray(np.maximum(pred, 0), coords=[estim["Y"], estim["X"]], dims=["Y", "X"])
            out.append(self._expand_T(da2d, t))

        if not out:
            return xr.DataArray(np.nan)
        return xr.concat(out, dim="T") * mask

    def random_forest_merging(self, missing_value=-999.0, k_nearest=10,
                              n_estimators=300, n_jobs=1, random_state=42):
        g_df, feats, estim, obs, k = self._prepare_ml(missing_value, k_nearest=k_nearest)
        model = RandomForestRegressor(n_estimators=int(n_estimators), max_features=1.0,
            min_samples_leaf=3, n_jobs=n_jobs, random_state=random_state)
        model.fit(g_df[feats], g_df["Observation"])
        return self._predict_grid(model, estim, obs, feats, k), {
            "method": "RF", "n_samples": len(g_df), "k_nearest": k,
            "evaluation": "fitted reconstruction; use validate_merging_methods for held-out skill"}


    def xgboost_merging(self, missing_value=-999.0, tune=True, k_nearest=10,
                        n_estimators=300, n_jobs=1, random_state=42):
        if xgb is None:
            raise ImportError("xgboost is required for xgboost_merging.")
        g_df, feats, estim, obs, k = self._prepare_ml(missing_value, k_nearest=k_nearest)
        base = xgb.XGBRegressor(n_estimators=int(n_estimators), learning_rate=0.05,
            max_depth=3, subsample=0.9, colsample_bytree=0.9,
            objective="reg:squarederror", random_state=random_state, n_jobs=1)
        params = None
        if tune:
            groups = pd.DatetimeIndex(g_df["T"]).year.to_numpy()
            if len(np.unique(groups)) < 2:
                # A single date supports spatial tuning only; disclose this scope.
                groups = np.array([_cv_station_key(x, y) for x, y in zip(g_df.Xp, g_df.Yp)])
            if len(np.unique(groups)) < 2:
                raise ValueError("XGB tuning requires at least two independent groups.")
            splitter = _GroupKFold(n_splits=min(3, len(np.unique(groups))))
            # No network features during inner tuning: recomputing a network built
            # from all gauges would leak held-out station presence into spatial CV.
            cv_feats = [f for f in feats if not f.startswith("DIST_")]
            search = GridSearchCV(base, {"max_depth": [2, 3, 5], "learning_rate": [0.03, 0.1]},
                cv=splitter, scoring="neg_mean_squared_error", n_jobs=n_jobs)
            search.fit(g_df[cv_feats], g_df["Observation"], groups=groups)
            params = search.best_params_
            base.set_params(**params)
        base.set_params(n_jobs=n_jobs)
        base.fit(g_df[feats], g_df["Observation"])
        return self._predict_grid(base, estim, obs, feats, k), {
            "method": "XGB", "best_params": params, "n_samples": len(g_df), "k_nearest": k,
            "evaluation": "fitted reconstruction; use held-out validation for skill"}


    # --------------------- helper for validation features ---------------------

    def _add_knn_features(self, df, k_nearest, train_stations_only=False):
        df = df.copy()
        if not train_stations_only:
            self.unique_stations = df[["Yp", "Xp"]].drop_duplicates().to_numpy(float)
        if self.unique_stations is None or not len(self.unique_stations):
            raise ValueError("Training station network is empty.")
        k = max(0, min(int(k_nearest), len(self.unique_stations)))
        if k:
            distance, _ = cKDTree(self.unique_stations).query(
                df[["Yp", "Xp"]].to_numpy(float), k=list(range(1, k + 1)))
            for i in range(k):
                df[f"DIST_{i+1}"] = distance[:, i]
        return df, k


    # --------------------- validation (LOYO) ---------------------

    def validate_merging_methods(self, *args, **kwargs):
        """See the shared validator for temporal/spatial scope and coverage checks."""
        return _validate_merging_methods_scientific(self, *args, **kwargs)


    # --------------------- plotting ---------------------

    def plot_merging_comparison(self, da_corrected, da_estimated=None, missing_value=-999.0):
        """
        Create a side-by-side scatter plot comparing observations against
        the original estimation and a corrected field.

        Parameters
        ----------
        da_corrected : xr.DataArray
            Corrected field (output from a merging method) with dimensions (T, Y, X).
        da_estimated : xr.DataArray, optional
            Original estimation grid. If None, uses self.da.
        missing_value : float, optional
            Missing value indicator for observations (passed to transform_cpt).
        """
        # Get observation Dataset
        ds_obs = self.transform_cpt(self.df, missing_value=missing_value)   # Dataset

        if da_estimated is None:
            da_estimated = self.da

        # Interpolate estimation and correction to observation points
        da_estimated_interp = da_estimated.interp(X=ds_obs["X"], Y=ds_obs["Y"])
        da_corrected_interp = da_corrected.interp(X=ds_obs["X"], Y=ds_obs["Y"])

        # Convert to DataFrames
        df_obs = ds_obs.to_dataframe().reset_index()   # columns: "Y", "X", "T", "Observation"
        df_est = da_estimated_interp.to_dataframe(name="Estimation").reset_index()
        df_cor = da_corrected_interp.to_dataframe(name="Correction").reset_index()

        merged = pd.merge(pd.merge(df_obs, df_est, on=["T", "Y", "X"]), df_cor, on=["T", "Y", "X"])
        df = merged.dropna(subset=["Observation", "Estimation", "Correction"])

        # Create figure
        fig, axes = plt.subplots(1, 2, figsize=(12, 6), sharex=True, sharey=True)

        # Common limits for square axes
        x_min, x_max = df["Observation"].min(), df["Observation"].max()
        y_min, y_max = x_min, x_max  # use same for y to keep square

        for ax in axes:
            ax.set_xlim(x_min, x_max)
            ax.set_ylim(y_min, y_max)
            ax.set_aspect('equal', adjustable='box')

        # y=x reference line
        line_x = np.linspace(x_min, x_max, 100)

        # Left: Observation vs Estimation
        sns.scatterplot(data=df, x="Observation", y="Estimation", ax=axes[0], color="blue", alpha=0.6)
        axes[0].plot(line_x, line_x, 'k--', label='1:1 line')
        axes[0].set_title("Observation vs Estimation")
        axes[0].set_xlabel("Observation")
        axes[0].set_ylabel("Estimation")
        axes[0].legend()

        # Right: Observation vs Correction
        sns.scatterplot(data=df, x="Observation", y="Correction", ax=axes[1], color="red", alpha=0.6)
        axes[1].plot(line_x, line_x, 'k--', label='1:1 line')
        axes[1].set_title("Observation vs Correction")
        axes[1].set_xlabel("Observation")
        axes[1].set_ylabel("Correction")
        axes[1].legend()

        plt.tight_layout()
        plt.show()

# =============================================================================
# Scientific spatial cross-validation layer (2026-08-26)
# =============================================================================
# Appended to preserve every historical public method.  The enhanced
# validate_merging_methods() accepts temporal, random, station and spatially
# blocked validation strategies without changing old calls (default remains LOYO).

from sklearn.model_selection import GroupKFold as _GroupKFold, LeaveOneGroupOut as _LeaveOneGroupOut
from sklearn.cluster import KMeans as _KMeans


def _cv_station_key(x, y):
    return f"{float(x):.7f}_{float(y):.7f}"


def _collect_merging_cv_rows(self, missing_value=-999.0):
    obs=self.transform_cpt(self.df, missing_value)
    estim=self.da
    rows=[]
    common=np.intersect1d(obs["T"].values, estim["T"].values)
    for t in common:
        d=self._station_df_for_time(obs,estim,t,add_dem=False)
        if d.empty:
            continue
        d=d.copy()
        d["TIME"]=pd.Timestamp(t)
        d["YEAR"]=int(pd.Timestamp(t).year)
        d["STATION_KEY"]=[_cv_station_key(x,y) for x,y in zip(d["X"],d["Y"])]
        rows.append(d)
    if not rows:
        return pd.DataFrame()
    full=pd.concat(rows,ignore_index=True)
    keys=sorted(full["STATION_KEY"].dropna().unique().tolist())
    key_to_id={k:i for i,k in enumerate(keys)}
    full["STATION_ID"]=full["STATION_KEY"].map(key_to_id).astype(int)
    return full


def _resolve_regional_labels(df, regional_labels=None, n_regions=5, random_state=42):
    st=(df[["STATION_ID","STATION_KEY","Xp","Yp"]]
        .drop_duplicates("STATION_ID").sort_values("STATION_ID").copy())
    if st.empty:
        return pd.Series(dtype=object)
    if regional_labels is None:
        k=max(2,min(int(n_regions),len(st))) if len(st)>1 else 1
        if k==1:
            lab=np.zeros(len(st),dtype=int)
        else:
            lab=_KMeans(n_clusters=k,random_state=int(random_state),n_init=20).fit_predict(st[["Xp","Yp"]].values)
        return pd.Series(lab,index=st["STATION_ID"].values)

    mapping={}
    if callable(regional_labels):
        vals=regional_labels(st.copy())
        if len(vals)!=len(st):
            raise ValueError("regional_labels callable must return one label per unique station.")
        return pd.Series(np.asarray(vals),index=st["STATION_ID"].values)
    if isinstance(regional_labels,pd.DataFrame):
        cols=set(regional_labels.columns)
        if {"STATION_ID","REGION"}.issubset(cols):
            mapping=dict(zip(regional_labels["STATION_ID"],regional_labels["REGION"]))
        elif {"STATION_KEY","REGION"}.issubset(cols):
            bykey=dict(zip(regional_labels["STATION_KEY"],regional_labels["REGION"]))
            mapping={int(r.STATION_ID):bykey.get(r.STATION_KEY,np.nan) for r in st.itertuples()}
        else:
            raise ValueError("regional_labels DataFrame needs STATION_ID/REGION or STATION_KEY/REGION.")
    elif isinstance(regional_labels,pd.Series):
        mapping=regional_labels.to_dict()
    elif isinstance(regional_labels,dict):
        mapping=regional_labels
    else:
        arr=np.asarray(regional_labels)
        if arr.size!=len(st):
            raise ValueError("regional_labels array must match the number of unique stations.")
        return pd.Series(arr,index=st["STATION_ID"].values)

    out={}
    for r in st.itertuples():
        if r.STATION_ID in mapping:
            out[r.STATION_ID]=mapping[r.STATION_ID]
        elif r.STATION_KEY in mapping:
            out[r.STATION_ID]=mapping[r.STATION_KEY]
        else:
            out[r.STATION_ID]=np.nan
    ser=pd.Series(out)
    if ser.isna().any():
        missing=ser[ser.isna()].index.tolist()
        raise ValueError(f"regional_labels missing station ids: {missing[:10]}")
    return ser


def _merging_cv_splits(df, strategy="loyo", n_splits=5, random_state=42,
                        spatial_block_size_km=150.0, regional_labels=None, n_regions=5):
    strategy=str(strategy).lower().replace("-","_")
    aliases={"leave_one_year_out":"loyo","leave_one_station_out":"loso",
             "spatial":"spatial_block","regional":"regional_block","kfold":"random"}
    strategy=aliases.get(strategy,strategy)
    idx=np.arange(len(df))
    if len(df) < 2:
        raise ValueError("At least two samples are needed for validation.")
    if strategy=="random":
        k=max(2,min(int(n_splits),len(df)))
        splitter=KFold(n_splits=k,shuffle=True,random_state=int(random_state))
        return [(tr,te,f"fold_{i+1}") for i,(tr,te) in enumerate(splitter.split(idx))]
    if strategy=="loyo":
        groups=df["YEAR"].to_numpy()
        spl=_LeaveOneGroupOut()
        return [(tr,te,f"year_{int(groups[te[0]])}") for tr,te in spl.split(idx,groups=groups)]
    if strategy=="loso":
        groups=df["STATION_ID"].to_numpy()
        spl=_LeaveOneGroupOut()
        return [(tr,te,f"station_{int(groups[te[0]])}") for tr,te in spl.split(idx,groups=groups)]
    if strategy=="spatial_block":
        if float(spatial_block_size_km) <= 0:
            raise ValueError("spatial_block_size_km must be positive.")
        size=float(spatial_block_size_km)*1000.0
        bx=np.floor((df["Xp"].to_numpy()-df["Xp"].min())/size).astype(np.int64)
        by=np.floor((df["Yp"].to_numpy()-df["Yp"].min())/size).astype(np.int64)
        _,groups=np.unique(np.column_stack([bx,by]),axis=0,return_inverse=True)
        ng=len(np.unique(groups))
        if ng<2:
            raise ValueError("spatial_block produced <2 blocks; reduce spatial_block_size_km.")
        k=max(2,min(int(n_splits),ng))
        spl=_GroupKFold(n_splits=k)
        return [(tr,te,f"spatial_fold_{i+1}") for i,(tr,te) in enumerate(spl.split(idx,groups=groups))]
    if strategy=="regional_block":
        station_region=_resolve_regional_labels(df,regional_labels,n_regions,random_state)
        groups=df["STATION_ID"].map(station_region).to_numpy()
        if pd.isna(groups).any():
            raise ValueError("Some stations do not have a regional label.")
        if len(np.unique(groups))<2:
            raise ValueError("regional_block needs at least two regions.")
        spl=_LeaveOneGroupOut()
        return [(tr,te,f"region_{groups[te[0]]}") for tr,te in spl.split(idx,groups=groups)]
    raise ValueError("cv_strategy must be one of: 'loyo', 'random', 'loso', 'spatial_block', 'regional_block'.")


def _safe_rmse(obs,pred):
    obs=np.asarray(obs,float); pred=np.asarray(pred,float)
    keep=np.isfinite(obs)&np.isfinite(pred)
    if keep.sum()==0: return np.nan
    return float(np.sqrt(np.mean((obs[keep]-pred[keep])**2)))


def _geostat_predict_one(self, method, train_df, test_df, fold_key="cv"):
    if train_df.empty or test_df.empty:
        return np.full(len(test_df),np.nan)
    method=method.upper()
    if method=="SBA":
        tr=train_df.copy(); tr["residuals"]=tr["Observation"]-tr["Estimation"]
        ok,_=self.auto_select_kriging_parameters(tr,method_key=f"{fold_key}_SBA",z_col="residuals")
        z,_=ok.execute("points",test_df["Xp"].values,test_df["Yp"].values)
        return test_df["Estimation"].values+np.asarray(z,float).ravel()
    if method in {"RK", "NN"}:
        features = ["Estimation"] if method == "RK" else ["Xp", "Yp", "Estimation"]
        reg = LinearRegression() if method == "RK" else _merging_neural_regressor()
        reg.fit(train_df[features].values, train_df["Observation"].values)
        tr=train_df.copy(); tr["residuals"]=tr["Observation"]-reg.predict(tr[features].values)
        tr.attrs["residual_model"] = method
        tr.attrs["residual_features"] = features
        ok,_=self.auto_select_kriging_parameters(tr,method_key=f"{fold_key}_{method}",z_col="residuals")
        z,_=ok.execute("points",test_df["Xp"].values,test_df["Yp"].values)
        return reg.predict(test_df[features].values)+np.asarray(z,float).ravel()
    if method=="CM":
        oko,_=self.auto_select_kriging_parameters(train_df,method_key=f"{fold_key}_CMo",z_col="Observation")
        oke,_=self.auto_select_kriging_parameters(train_df,method_key=f"{fold_key}_CMe",z_col="Estimation")
        zo,_=oko.execute("points",test_df["Xp"].values,test_df["Yp"].values)
        ze,_=oke.execute("points",test_df["Xp"].values,test_df["Yp"].values)
        return np.asarray(zo,float).ravel()+(test_df["Estimation"].values-np.asarray(ze,float).ravel())
    if method=="KED":
        uk=UniversalKriging(train_df["Xp"].values.astype(float),train_df["Yp"].values.astype(float),
                            train_df["Observation"].values.astype(float),variogram_model="linear",
                            drift_terms=["specified"],specified_drift=[train_df["Estimation"].values.astype(float)],
                            enable_plotting=False,verbose=False)
        z,_=uk.execute("points",test_df["Xp"].values.astype(float),test_df["Yp"].values.astype(float),
                       specified_drift_arrays=[test_df["Estimation"].values.astype(float)])
        return np.asarray(z,float).ravel()
    st=self._xy_stack(train_df["Yp"].values,train_df["Xp"].values)
    te=self._xy_stack(test_df["Yp"].values,test_df["Xp"].values)
    res=(train_df["Observation"].values-train_df["Estimation"].values).astype(float)
    if method=="OI":
        ds=cdist(st,st); L=150000.0; eps=0.1
        C=np.exp(-(ds**2)/(2.0*L**2))+eps*np.eye(len(st)); w=solve(C,res,assume_a="pos")
        return test_df["Estimation"].values + np.exp(-(cdist(te,st)**2)/(2.0*L**2))@w
    if method=="BARNES":
        dte=cdist(te,st); kappa=200000.0; gamma=0.3
        w1=np.exp(-(dte**2)/kappa**2); s1=w1.sum(1); s1[s1==0]=1e-12; z1=(w1@res)/s1
        dst=cdist(st,st); w1s=np.exp(-(dst**2)/kappa**2); ss=w1s.sum(1); ss[ss==0]=1e-12
        z1s=(w1s@res)/ss; new=res-z1s
        w2=np.exp(-(dte**2)/(gamma*kappa)**2); s2=w2.sum(1); s2[s2==0]=1e-12; z2=(w2@new)/s2
        return test_df["Estimation"].values+z1+z2
    raise ValueError(method)


def _geostat_predict(self, method, train_df, test_df, fold_key, same_time=True, min_train_points=5):
    if not same_time:
        raise ValueError("Spatial merging requires contemporaneous training gauges; pooled years are not equivalent.")
    pred = np.full(len(test_df), np.nan)
    for t, sub in test_df.groupby("TIME", sort=False):
        tr = train_df[train_df["TIME"] == t]
        if len(tr) < int(min_train_points):
            continue
        # Every fit and variogram selection uses only this outer training fold.
        vals = _geostat_predict_one(self, method, tr, sub, f"{fold_key}_{pd.Timestamp(t).date()}")
        pred[test_df.index.get_indexer(sub.index)] = vals
    return np.maximum(pred, 0.0)



def _ml_predict(self, method, train_df, test_df, k_nearest=10, random_state=42):
    train_f, k_used = self._add_knn_features(train_df, k_nearest)
    test_f, _ = self._add_knn_features(test_df, k_nearest, train_stations_only=True)
    feats = ["X", "Y", "Estimation"] + [f"DIST_{i+1}" for i in range(k_used)]
    if self.dem is not None:
        for d in (train_f, test_f):
            xp = xr.DataArray(d["X"].values, dims="points")
            yp = xr.DataArray(d["Y"].values, dims="points")
            d["DEM"] = np.asarray(self.dem.interp(X=xp, Y=yp).values, float)
        feats.append("DEM")
    keep_train = np.isfinite(train_f[feats + ["Observation"]].to_numpy(float)).all(axis=1)
    keep_test = np.isfinite(test_f[feats].to_numpy(float)).all(axis=1)
    out = np.full(len(test_f), np.nan)
    if keep_train.sum() < 5 or not keep_test.any():
        return out
    if method == "RF":
        mdl = RandomForestRegressor(n_estimators=300, max_features=1.0,
            min_samples_leaf=3, n_jobs=1, random_state=int(random_state))
    else:
        if xgb is None:
            raise ImportError("xgboost is required to validate XGB.")
        mdl = xgb.XGBRegressor(n_estimators=300, learning_rate=0.05, max_depth=3,
            subsample=0.9, colsample_bytree=0.9, objective="reg:squarederror",
            random_state=int(random_state), n_jobs=1)
    mdl.fit(train_f.loc[keep_train, feats], train_f.loc[keep_train, "Observation"])
    out[keep_test] = mdl.predict(test_f.loc[keep_test, feats])
    return np.maximum(out, 0.0)



def _validate_merging_methods_scientific(
    self, missing_value=-999.0, min_points_year=5, k_nearest=10,
    cv_strategy="loyo", n_splits=5, random_state=42,
    spatial_block_size_km=150.0, regional_labels=None, n_regions=5,
    methods=("SBA", "RK", "CM", "KED", "OI", "Barnes", "RF", "XGB"),
    per_time_geostatistics=True, kriging_cv_strategy="match", return_details=False,
):
    """Evaluate held-out station predictions and report all failures/coverage.

    LOYO tests temporal transfer of RF/XGB. Same-time spatial merging (SBA, RK,
    NN, CM, KED, OI, Barnes) has no training gauge in a held-out year, hence is
    explicitly non-evaluable under LOYO. Use LOSO/spatial/regional blocks for
    these methods. No pooled-year spatial covariance is substituted silently.
    Scores are sqrt(total held-out squared error / number of valid predictions).
    Methods with incomplete coverage have a NaN headline score; their conditional
    error on successful predictions remains available as Partial_RMSE in attrs.
    """
    if not per_time_geostatistics:
        raise ValueError("Pooled historical spatial residuals are not a synchronous merging method.")
    all_df = _collect_merging_cv_rows(self, missing_value)
    aliases = {"leave_one_year_out": "loyo", "leave_one_station_out": "loso",
               "spatial": "spatial_block", "regional": "regional_block", "kfold": "random"}
    strategy = str(cv_strategy).lower().replace("-", "_")
    strategy = aliases.get(strategy, strategy)
    absent = object()
    old = {key: getattr(self, key, absent) for key in (
        "kriging_cv_strategy", "kriging_spatial_block_size_km", "kriging_n_regions",
        "_kriging_param_cache", "unique_stations")}
    records = []
    selected = [str(m).upper() for m in methods]
    geostat = {"SBA", "RK", "NN", "CM", "KED", "OI", "BARNES"}
    if any(m not in geostat | {"RF", "XGB"} for m in selected):
        raise ValueError("Unknown merging validation method.")
    kcv = str(kriging_cv_strategy).lower().replace("-", "_")
    if kcv == "match":
        kcv = strategy if strategy in {"random", "loso", "spatial_block", "regional_block"} else "random"
    try:
        self.kriging_cv_strategy = kcv
        self.kriging_spatial_block_size_km = float(spatial_block_size_km)
        self.kriging_n_regions = int(n_regions)
        self._kriging_param_cache = {}
        splits = [] if all_df.empty else _merging_cv_splits(all_df, strategy, n_splits,
            random_state, spatial_block_size_km, regional_labels, n_regions)
        for fold_i, (tr_idx, te_idx, label) in enumerate(splits, 1):
            tr = all_df.iloc[tr_idx].copy().reset_index(drop=True)
            te = all_df.iloc[te_idx].copy().reset_index(drop=True)
            for m in selected:
                pred = np.full(len(te), np.nan)
                status, error = "ok", ""
                if strategy == "loyo" and m in geostat:
                    status = "not_applicable_no_contemporaneous_gauges"
                elif len(tr) < 5 or (strategy == "loyo" and len(te) < int(min_points_year)):
                    status = "insufficient_samples"
                else:
                    try:
                        if m in geostat:
                            pred = _geostat_predict(self, m, tr, te, label, min_train_points=5)
                        else:
                            pred = _ml_predict(self, m, tr, te, k_nearest, random_state)
                    except Exception as exc:
                        status, error = "failed", str(exc)
                pred = np.maximum(np.asarray(pred, float), 0.0)
                observed = te.Observation.to_numpy(float)
                keep = np.isfinite(pred) & np.isfinite(observed)
                n = int(keep.sum())
                sse = float(np.sum((observed[keep] - pred[keep]) ** 2))
                if status == "ok" and n != len(te):
                    status = "partial" if n else "no_predictions"
                records.append({"Fold": label, "FoldIndex": fold_i, "Strategy": strategy,
                    "Method": m, "RMSE": np.sqrt(sse / n) if n else np.nan, "SSE": sse,
                    "N_test": n, "N_train": len(tr), "N_test_total": len(te),
                    "Status": status, "Error": error})
    finally:
        for key, value in old.items():
            if value is absent:
                if hasattr(self, key):
                    delattr(self, key)
            else:
                setattr(self, key, value)
    cols = ["Fold", "FoldIndex", "Strategy", "Method", "RMSE", "SSE", "N_test", "N_train",
            "N_test_total", "Status", "Error"]
    details = pd.DataFrame(records, columns=cols)
    scores, coverage, partial = {}, {}, {}
    for m in selected:
        d = details[details.Method == m]
        n, total = int(d.N_test.sum()), int(d.N_test_total.sum())
        coverage[m] = n / total if total else 0.0
        partial[m] = float(np.sqrt(d.SSE.sum() / n)) if n else np.nan
        scores[m] = partial[m] if total and n == total else np.nan
    summary = pd.Series(scores, name="RMSE", dtype=float).sort_values()
    summary.index.name = "Method"
    summary.attrs.update({"cv_strategy": strategy, "n_folds": int(details.Fold.nunique()),
        "kriging_cv_strategy": kcv, "Coverage": coverage, "Partial_RMSE": partial,
        "score": "pooled RMSE; headline score requires complete coverage"})
    return (summary, details) if return_details else summary



# Public helper so notebooks can inspect exactly which rows a strategy withholds.
def _merging_cv_plan_public(self, missing_value=-999.0, cv_strategy="loyo", n_splits=5,
                            random_state=42, spatial_block_size_km=150.0,
                            regional_labels=None, n_regions=5):
    df=_collect_merging_cv_rows(self,missing_value)
    splits=_merging_cv_splits(df,cv_strategy,n_splits,random_state,
                              spatial_block_size_km,regional_labels,n_regions)
    rows=[]
    for i,(tr,te,label) in enumerate(splits,1):
        test=df.iloc[te]
        rows.append({"Fold":label,"FoldIndex":i,"N_train":len(tr),"N_test":len(te),
                     "TestStations":int(test["STATION_ID"].nunique()),
                     "TestYears":int(test["YEAR"].nunique())})
    return pd.DataFrame(rows)


WAS_Merging_Extended.validate_merging_methods=_validate_merging_methods_scientific
WAS_Merging_Extended.cv_plan=_merging_cv_plan_public

# -----------------------------------------------------------------------------
# Spatially-aware variogram tuning (replaces the historical random KFold only)
# -----------------------------------------------------------------------------
def _point_cv_splits(df, x_col, y_col, strategy, n_splits, random_state,
                     spatial_block_size_km=150.0, n_regions=5):
    strategy=str(strategy or "random").lower().replace("-","_")
    aliases={"kfold":"random","spatial":"spatial_block","regional":"regional_block",
             "leave_one_station_out":"loso"}
    strategy=aliases.get(strategy,strategy)
    idx=np.arange(len(df))
    if len(df) < 2:
        raise ValueError("At least two samples are needed for validation.")
    if strategy=="random":
        k=max(2,min(int(n_splits),len(df)))
        return list(KFold(n_splits=k,shuffle=True,random_state=int(random_state)).split(idx))
    # At a single analysis time every coordinate is a station.  For pooled data,
    # identical coordinates are kept in the same group.
    keys=np.array([_cv_station_key(a,b) for a,b in zip(df[x_col],df[y_col])],dtype=object)
    if strategy=="loso":
        return list(_LeaveOneGroupOut().split(idx,groups=keys))
    if strategy=="spatial_block":
        if float(spatial_block_size_km) <= 0:
            raise ValueError("spatial_block_size_km must be positive.")
        size=float(spatial_block_size_km)*1000.0
        x=df[x_col].to_numpy(float); y=df[y_col].to_numpy(float)
        bx=np.floor((x-x.min())/size).astype(np.int64); by=np.floor((y-y.min())/size).astype(np.int64)
        _,g=np.unique(np.column_stack([bx,by]),axis=0,return_inverse=True)
        ng=len(np.unique(g))
        if ng < 2:
            raise ValueError("Inner spatial CV has fewer than two blocks; reduce block size.")
        return list(_GroupKFold(n_splits=max(2,min(int(n_splits),ng))).split(idx,groups=g))
    if strategy=="regional_block":
        k=max(2,min(int(n_regions),len(df)))
        g=_KMeans(n_clusters=k,random_state=int(random_state),n_init=20).fit_predict(df[[x_col,y_col]].to_numpy(float))
        return list(_LeaveOneGroupOut().split(idx,groups=g))
    raise ValueError("kriging cv_strategy must be random, loso, spatial_block, or regional_block.")


def _auto_select_kriging_parameters_scientific(
    self, df, method_key, t=None, x_col="Xp", y_col="Yp", z_col="residuals",
    variogram_models=None, nlags_range=range(3,10), n_splits=5, random_state=42,
    cv_strategy=None, spatial_block_size_km=150.0, n_regions=5,
):
    residual_model = df.attrs.get("residual_model")
    residual_features = df.attrs.get("residual_features", [])
    cols = list(dict.fromkeys([x_col, y_col, z_col] +
        (["Observation"] + list(residual_features) if residual_model else [])))
    df = df[cols].replace([np.inf, -np.inf], np.nan).dropna().copy()
    if len(df)<max(5,min(int(n_splits),len(df))):
        raise ValueError("Not enough points for kriging.")
    x=df[x_col].to_numpy(float); y=df[y_col].to_numpy(float); z=df[z_col].to_numpy(float)
    if not self.kriging_cv:
        fixed = self.kriging_fixed or {}
        vm=fixed.get("variogram_model","spherical"); nl=int(fixed.get("nlags",6))
        return self._fit_ok(x,y,z,vm,nl),{"variogram":vm,"nlags":nl,"cv_mse":None,"cv_strategy":"fixed"}
    strategy=str(cv_strategy or getattr(self,"kriging_cv_strategy","random")).lower().replace("-","_")
    block_km=float(getattr(self,"kriging_spatial_block_size_km",spatial_block_size_km))
    regions=int(getattr(self,"kriging_n_regions",n_regions))
    cid=self._kriging_cache_id(method_key,t=t)
    if cid: cid=f"{cid}|{strategy}|{block_km:g}|{regions}"
    if cid and cid in self._kriging_param_cache:
        vm,nl=self._kriging_param_cache[cid]
        return self._fit_ok(x,y,z,vm,nl),{"variogram":vm,"nlags":nl,"cv_mse":"cached","cv_strategy":strategy}
    if variogram_models is None:
        variogram_models=["linear","power","gaussian","spherical","exponential"]
    splits=_point_cv_splits(df,x_col,y_col,strategy,n_splits,random_state,block_km,regions)
    if not splits: raise ValueError("No valid kriging CV split could be built.")
    # Regressions and their normalisations are refitted once per inner fold,
    # then reused across variogram candidates (no held-out residual leakage).
    residual_folds = []
    for tr, te in splits:
        if len(tr) < 3 or len(te) < 1:
            continue
        if residual_model:
            reg = LinearRegression() if residual_model == "RK" else _merging_neural_regressor()
            reg.fit(df.iloc[tr][residual_features].to_numpy(), df.iloc[tr].Observation.to_numpy())
            train_z = df.iloc[tr].Observation.to_numpy() - reg.predict(df.iloc[tr][residual_features].to_numpy())
            test_z = df.iloc[te].Observation.to_numpy() - reg.predict(df.iloc[te][residual_features].to_numpy())
        else:
            train_z, test_z = z[tr], z[te]
        residual_folds.append((tr, te, train_z, test_z))
    best_score=np.inf; best_model=None; best_nlags=None
    for model_name in variogram_models:
        for nlags in nlags_range:
            squared_errors, n_valid = [], 0
            for tr, te, train_z, test_z in residual_folds:
                try:
                    ok=self._fit_ok(x[tr],y[tr],train_z,model_name,nlags)
                    zp,_=ok.execute("points",x[te],y[te]); zp=np.asarray(zp,float).ravel()
                    if not np.isfinite(zp).all():
                        raise ValueError("Non-finite inner validation predictions.")
                    squared_errors.append(float(np.sum((test_z-zp)**2)))
                except Exception:
                    squared_errors.append(np.inf)
                n_valid += len(te)
            mse=float(np.sum(squared_errors) / n_valid) if n_valid else np.inf
            if mse<best_score:
                best_score,best_model,best_nlags=mse,model_name,int(nlags)
    if best_model is None:
        raise RuntimeError("No variogram configuration could be fitted under the requested CV strategy.")
    if cid: self._kriging_param_cache[cid]=(best_model,best_nlags)
    return self._fit_ok(x,y,z,best_model,best_nlags),{
        "variogram":best_model,"nlags":best_nlags,"cv_mse":best_score,
        "cv_strategy":strategy,"n_cv_folds":len(splits),
    }


def _set_kriging_cv_strategy_public(self, strategy="random", spatial_block_size_km=150.0, n_regions=5):
    strategy=str(strategy).lower().replace("-","_")
    if strategy not in {"random","loso","spatial_block","regional_block"}:
        raise ValueError("strategy must be random, loso, spatial_block, or regional_block.")
    self.kriging_cv_strategy=strategy
    self.kriging_spatial_block_size_km=float(spatial_block_size_km)
    self.kriging_n_regions=int(n_regions)
    # Cached variogram choices are strategy-dependent; clear them on strategy change.
    self._kriging_param_cache={}
    return self

WAS_Merging_Extended.auto_select_kriging_parameters=_auto_select_kriging_parameters_scientific
WAS_Merging_Extended.set_kriging_cv_strategy=_set_kriging_cv_strategy_public

def _cv_station_table_public(self, missing_value=-999.0):
    """Return stable station identifiers/coordinates used by LOSO/regional CV."""
    df=_collect_merging_cv_rows(self,missing_value)
    if df.empty:
        return pd.DataFrame(columns=["STATION_ID","STATION_KEY","X","Y","Xp","Yp"])
    return (df[["STATION_ID","STATION_KEY","X","Y","Xp","Yp"]]
            .drop_duplicates("STATION_ID").sort_values("STATION_ID").reset_index(drop=True))

WAS_Merging_Extended.cv_station_table=_cv_station_table_public

# Backward-compatible bridge: old WAS_Merging notebooks can request the new CV
# without migrating construction code.  Actual prediction methods remain untouched.
def _legacy_merging_extended(self):
    return WAS_Merging_Extended(
        self.df, self.da, date_month_day=getattr(self,"date_month_day","08-01"),
        verbose=False,
    )

def _legacy_validate_merging_methods(self,*args,**kwargs):
    return _legacy_merging_extended(self).validate_merging_methods(*args,**kwargs)

def _legacy_merging_cv_plan(self,*args,**kwargs):
    return _legacy_merging_extended(self).cv_plan(*args,**kwargs)

def _legacy_merging_station_table(self,*args,**kwargs):
    return _legacy_merging_extended(self).cv_station_table(*args,**kwargs)

WAS_Merging.validate_merging_methods=_legacy_validate_merging_methods
WAS_Merging.cv_plan=_legacy_merging_cv_plan
WAS_Merging.cv_station_table=_legacy_merging_station_table


def _station_input_dataset(df, date_month_day="08-01", missing_value=-999.0):
    """Convert wide station data without fabricating or nudging station locations.

    Coincident gauge records are averaged per date at their common physical point;
    separate gauges on the same latitude or longitude retain exact coordinates.
    """
    id_col = "STATION" if "STATION" in df.columns else "ID" if "ID" in df.columns else None
    if id_col is None:
        raise ValueError("Station table requires STATION (CPT) or ID (CDT).")
    labels = df[id_col].astype(str).str.strip().str.upper()
    metadata = labels.isin(["LAT", "LON", "ELEV"])
    meta = df.loc[metadata].copy()
    meta[id_col] = labels[metadata]
    if not {"LAT", "LON"}.issubset(set(meta[id_col])):
        raise ValueError("Explicit LAT and LON metadata rows are required.")
    if meta[id_col].duplicated().any():
        raise ValueError("Duplicate station metadata rows.")
    coords = meta.set_index(id_col).T
    for c in ("LAT", "LON"):
        coords[c] = pd.to_numeric(coords[c], errors="raise")
    if not (np.isfinite(coords[["LAT", "LON"]]).all().all()
            and coords.LAT.between(-90, 90).all() and coords.LON.between(-180, 360).all()):
        raise ValueError("Station latitude/longitude coordinates are invalid.")
    data = df.loc[~metadata].rename(columns={id_col: "DATE"}).melt(
        id_vars="DATE", var_name="STATION", value_name="Observation")
    raw = data.DATE.astype(str).str.strip().str.replace(r"\.0$", "", regex=True)
    if raw.str.fullmatch(r"\d{4}").all():
        dates = pd.to_datetime(raw + "-" + date_month_day, errors="raise")
    elif raw.str.fullmatch(r"\d{8}").all():
        dates = pd.to_datetime(raw, format="%Y%m%d", errors="raise")
    else:
        dates = pd.to_datetime(raw, errors="raise")
    data["T"] = dates
    data["Observation"] = pd.to_numeric(data.Observation, errors="coerce")
    data = data.join(coords[["LAT", "LON"]], on="STATION").rename(columns={"LAT": "Y", "LON": "X"})
    bad = ~np.isfinite(data.Observation)
    if missing_value is not None:
        bad |= data.Observation == missing_value
    data.loc[bad, "Observation"] = np.nan
    if data.duplicated(["T", "Y", "X"]).any():
        warnings.warn("Coincident gauges are averaged per date at their true location.", RuntimeWarning)
    return data.groupby(["T", "Y", "X"], sort=True).Observation.mean().to_xarray().to_dataset()


class _ConstantKriging:
    """Exact constant-field limit of ordinary kriging, with zero residual variance."""
    def __init__(self, value):
        self.value = value

    def execute(self, style, x, y, **kwargs):
        shape = (len(y), len(x)) if style == "grid" else np.asarray(x).shape
        return np.full(shape, self.value, dtype=float), np.zeros(shape, dtype=float)


def _merging_neural_regressor():
    return TransformedTargetRegressor(
        regressor=make_pipeline(StandardScaler(), MLPRegressor(
            hidden_layer_sizes=(10,), activation="tanh", solver="lbfgs",
            alpha=1.0, max_iter=500, random_state=42)),
        transformer=StandardScaler())


def _legacy_spatial_method(self, method, missing_value=-999.0, do_cross_validation=False):
    merger = _legacy_merging_extended(self)
    # Legacy methods return (field, optional time-level CV DataFrame).
    obs = merger.transform_cpt(self.df, missing_value)
    times = np.intersect1d(obs["T"].values, merger.da["T"].values)
    fields, records = [], []
    for t in times:
        train = merger._station_df_for_time(obs, merger.da, t)
        values = np.full((merger.da.sizes["Y"], merger.da.sizes["X"]), np.nan)
        if len(train) >= 5:
            flat_est = merger.da.sel(T=t).values.ravel()
            test = pd.DataFrame({"Xp": merger._grid_cache["xp_flat"],
                "Yp": merger._grid_cache["yp_flat"], "Estimation": flat_est})
            good = np.isfinite(flat_est)
            try:
                prediction = _geostat_predict_one(merger, method, train, test.loc[good],
                    fold_key=f"production_{pd.Timestamp(t).date()}")
                values.ravel()[good] = np.maximum(prediction, 0.0)
            except Exception as exc:
                warnings.warn(f"{method} failed for {t}: {exc}", RuntimeWarning)
            if do_cross_validation:
                predicted = np.full(len(train), np.nan)
                for j in range(len(train)):
                    tr = train.drop(train.index[j])
                    if len(tr) < 5:
                        continue
                    try:
                        predicted[j] = np.maximum(_geostat_predict_one(merger, method,
                            tr, train.iloc[[j]], fold_key=f"loo_{pd.Timestamp(t).date()}_{j}"), 0)[0]
                    except Exception:
                        pass
                n = int(np.isfinite(predicted).sum())
                records.append({"time": t, "LOO_RMSE": _safe_rmse(train.Observation, predicted)
                    if n == len(train) else np.nan, "num_stations": len(train),
                    "N_valid": n, "Coverage": n / len(train)})
        fields.append(xr.DataArray(values, dims=("Y", "X"),
            coords={"Y": merger.da.Y, "X": merger.da.X}).expand_dims(T=[t]))
    output = xr.concat(fields, dim="T").reindex(T=merger.da["T"]) if fields else xr.full_like(merger.da, np.nan)
    output.attrs = dict(merger.da.attrs)
    return output, pd.DataFrame(records) if do_cross_validation else None


def _legacy_sba(self, missing_value=-999.0, do_cross_validation=False):
    return _legacy_spatial_method(self, "SBA", missing_value, do_cross_validation)


def _legacy_rk(self, missing_value=-999.0, do_cross_validation=False):
    return _legacy_spatial_method(self, "RK", missing_value, do_cross_validation)


def _legacy_nn(self, missing_value=-999.0, do_cross_validation=False):
    return _legacy_spatial_method(self, "NN", missing_value, do_cross_validation)


def _legacy_multiplicative(self, missing_value=-999.0, do_cross_validation=False):
    """Leave-one-year-out ratio-of-means correction on paired historical data.

    This is temporal bias correction, not a synchronous station assimilation.
    Held-out rainfall never enters its ratio or spatial interpolation.
    """
    merger = _legacy_merging_extended(self)
    data = _collect_merging_cv_rows(merger, missing_value)
    fields, records = [], []
    for t in merger.da["T"].values:
        field = xr.full_like(merger.da.sel(T=t), np.nan)
        year = pd.Timestamp(t).year
        train = data[data.YEAR != year] if not data.empty else data
        if len(train):
            pooled = train.groupby(["X", "Y", "Xp", "Yp"], as_index=False)[["Observation", "Estimation"]].mean()
            valid = (pooled.Estimation > np.finfo(float).eps) & (pooled.Observation >= 0)
            pooled = pooled.loc[valid].copy()
            pooled["ratio"] = pooled.Observation / pooled.Estimation
            if len(pooled) >= 5:
                try:
                    ok, _ = merger.auto_select_kriging_parameters(pooled,
                        method_key=f"MULT_train_without_{year}", z_col="ratio")
                    ratio, _ = ok.execute("points", merger._grid_cache["xp_flat"], merger._grid_cache["yp_flat"])
                    ratio = np.maximum(np.asarray(ratio, float).reshape(field.shape), 0.0)
                    field = merger.da.sel(T=t) * ratio
                    if do_cross_validation:
                        test = data[data.TIME == pd.Timestamp(t)]
                        pred_ratio, _ = ok.execute("points", test.Xp.to_numpy(), test.Yp.to_numpy())
                        pred = test.Estimation.to_numpy() * np.maximum(np.asarray(pred_ratio, float), 0)
                        records.append({"time": t, "LOO_RMSE": _safe_rmse(test.Observation, pred),
                            "num_stations": len(test), "CV_scope": "leave_one_year_out_actual_rainfall"})
                except Exception as exc:
                    warnings.warn(f"Multiplicative correction failed for {t}: {exc}", RuntimeWarning)
        fields.append(field.expand_dims(T=[t]))
    output = xr.concat(fields, dim="T") if fields else xr.full_like(merger.da, np.nan)
    output.attrs = dict(merger.da.attrs)
    return output, pd.DataFrame(records) if do_cross_validation else None


WAS_Merging.simple_bias_adjustment = _legacy_sba
WAS_Merging.regression_kriging = _legacy_rk
WAS_Merging.neural_network_kriging = _legacy_nn
WAS_Merging.neural_network_kriging_ = _legacy_nn
WAS_Merging.multiplicative_bias = _legacy_multiplicative
