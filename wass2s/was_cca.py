"""Canonical Correlation Analysis (CCA) for seasonal forecasting.

Four CCA implementations are provided and retain their distinct transformation logic:

``WAS_CCA``
    Legacy CCA built on xeofs EOF analysis and numpy CCA.  Suitable for
    quick exploratory use.

``WAS_CCA_base``
    Leakage-free CCA that fits all preprocessing (NaN fill, standardization,
    EOF decomposition) strictly on the training fold and applies the fitted
    transforms to the test fold.  Recommended for cross-validation.

All classes expose ``compute_model``, ``compute_prob``, ``forecast``, and
``plot_cca_results`` methods.
"""
from scipy import stats
from scipy import signal as sig
import numpy as np
import xarray as xr
from scipy.stats import gamma, norm, lognorm, expon, weibull_min, t, poisson, nbinom
from scipy.optimize import brentq
from scipy.special import gamma as gamma_function
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import xeofs as xe
from wass2s.utils import * 


# -----------------------------------------------------------------------------
# Shared safety utilities. They do not alter the scientific transformation used
# by each CCA class; they only enforce dimensions, alignment and output contracts.
# -----------------------------------------------------------------------------

_DISTRIBUTION_CODES = {
    "normal": 1,
    "norm": 1,
    "gaussian": 1,
    "lognormal": 2,
    "lognorm": 2,
    "exponential": 3,
    "expon": 3,
    "gamma": 4,
    "weibull": 5,
    "weibull_min": 5,
    "t": 6,
    "student_t": 6,
    "student-t": 6,
    "poisson": 7,
    "nbinom": 8,
    "negative_binomial": 8,
    "negative-binomial": 8,
}


def _prepare_tyx(da: xr.DataArray, name: str) -> xr.DataArray:
    """Validate a WAS field and return it in ``(T, Y, X)`` order."""
    if not isinstance(da, xr.DataArray):
        raise TypeError(f"{name} must be an xarray.DataArray, got {type(da)!r}.")

    if "M" in da.dims:
        if da.sizes["M"] < 1:
            raise ValueError(f"{name} has an empty M dimension.")
        da = da.isel(M=0, drop=True)

    extra_dims = [d for d in da.dims if d not in ("T", "Y", "X")]
    singleton_dims = [d for d in extra_dims if da.sizes[d] == 1]
    if singleton_dims:
        da = da.squeeze(singleton_dims, drop=True)

    missing = [d for d in ("T", "Y", "X") if d not in da.dims]
    extra = [d for d in da.dims if d not in ("T", "Y", "X")]
    if missing or extra:
        raise ValueError(
            f"{name} must have dimensions T, Y, X after removing M. "
            f"Missing={missing}, extra={extra}, received={da.dims}."
        )

    return da.transpose("T", "Y", "X")


def _safe_mode_counts(n_modes: int, n_pca_modes: int, n_samples: int):
    """Limit CCA/PCA modes to what the training sample can support."""
    if n_samples < 2:
        raise ValueError("CCA requires at least two training samples along T.")
    max_modes = max(1, n_samples - 1)
    safe_pca = max(1, min(int(n_pca_modes), max_modes))
    safe_cca = max(1, min(int(n_modes), safe_pca, max_modes))
    return safe_cca, safe_pca


def _new_safe_cca(owner, X_train: xr.DataArray, y_train: xr.DataArray):
    """Create xeofs CCA with fold-compatible mode counts."""
    n_samples = min(X_train.sizes["T"], y_train.sizes["T"])
    n_modes, n_pca_modes = _safe_mode_counts(
        owner.n_modes, owner.n_pca_modes, n_samples
    )
    return xe.cross.CCA(
        n_modes=n_modes,
        standardize=owner.standardize,
        use_coslat=owner.use_coslat,
        use_pca=owner.use_pca,
        n_pca_modes=n_pca_modes,
    )


def _coordinates_compatible(reference, candidate, dim: str) -> bool:
    if reference.sizes[dim] != candidate.sizes[dim]:
        return False
    ref_values = np.asarray(reference[dim].values)
    cand_values = np.asarray(candidate[dim].values)
    if np.array_equal(ref_values, cand_values):
        return True
    if np.issubdtype(ref_values.dtype, np.number) and np.issubdtype(
        cand_values.dtype, np.number
    ):
        return np.allclose(ref_values, cand_values, equal_nan=True)
    return False


def _align_observation_and_hindcast(Predictant, hindcast_det):
    """Safely align historical observations and deterministic hindcasts."""
    Predictant = _prepare_tyx(Predictant, "Predictant")
    hindcast_det = _prepare_tyx(hindcast_det, "hindcast_det")

    for dim in ("Y", "X"):
        if _coordinates_compatible(Predictant, hindcast_det, dim):
            hindcast_det = hindcast_det.assign_coords({dim: Predictant[dim]})
            continue
        try:
            reordered = hindcast_det.reindex({dim: Predictant[dim]})
        except Exception as exc:
            raise ValueError(
                f"Predictant and hindcast_det have incompatible {dim} coordinates."
            ) from exc
        all_missing = reordered.isnull().all()
        try:
            all_missing = all_missing.compute()
        except Exception:
            pass
        try:
            all_missing = bool(all_missing.item())
        except Exception:
            all_missing = False
        if all_missing:
            raise ValueError(
                f"Predictant and hindcast_det have no compatible {dim} coordinates."
            )
        hindcast_det = reordered

    if np.array_equal(Predictant["T"].values, hindcast_det["T"].values):
        return Predictant, hindcast_det

    # Cross-validation outputs are commonly positionally correct but carry a
    # synthetic or integer T coordinate. Preserve that established WAS logic.
    if Predictant.sizes["T"] == hindcast_det.sizes["T"]:
        hindcast_det = hindcast_det.assign_coords(T=Predictant["T"])
        return Predictant, hindcast_det

    common_time = np.intersect1d(Predictant["T"].values, hindcast_det["T"].values)
    if common_time.size < 2:
        raise ValueError(
            "Predictant and hindcast_det do not have enough common T coordinates "
            "to estimate forecast residuals."
        )
    return Predictant.sel(T=common_time), hindcast_det.sel(T=common_time)


def _spatial_mask(Predictant: xr.DataArray) -> xr.DataArray:
    return xr.where(np.isfinite(Predictant.isel(T=0, drop=True)), 1.0, np.nan)


def _fill_spatial_gaps_safe(da: xr.DataArray) -> xr.DataArray:
    """Forward/backward spatial filling, avoiding an unnecessary bottleneck import."""
    missing = da.isnull().any()
    try:
        missing = missing.compute()
    except Exception:
        pass
    try:
        has_missing = bool(missing.item())
    except Exception:
        has_missing = True
    if not has_missing:
        return da

    out = da
    try:
        for dim in ("Y", "X", "lat", "lon"):
            if dim in out.dims:
                out = out.ffill(dim=dim).bfill(dim=dim)
    except ModuleNotFoundError:
        # ``xarray.ffill`` uses optional bottleneck. Nearest interpolation is a
        # safe fallback; any grid cell still undefined is handled by fillna(0).
        for dim in ("Y", "X", "lat", "lon"):
            if dim in out.dims:
                try:
                    out = out.interpolate_na(
                        dim=dim, method="nearest", fill_value="extrapolate"
                    )
                except Exception:
                    pass
    return out


def _common_validity_mask(field_imputed: xr.DataArray) -> xr.DataArray:
    """Shared (Y, X) validity mask for a spatial field fed to xeofs.

    A cell is kept when it is finite at every training step AND not temporally
    constant (zero variance). Failing cells are excluded (set to NaN) identically
    in the training, cross-validation and forecast inputs, so xeofs drops the
    same features on both sides instead of ingesting constant-zero fills. Falls
    back to the plain finite mask if the strict test would leave nothing.
    """
    finite_all = np.isfinite(field_imputed).all(dim="T")
    non_constant = field_imputed.std(dim="T", skipna=True) > 0
    mask = finite_all & non_constant
    try:
        if not bool(mask.any()):
            mask = finite_all
    except Exception:
        pass
    return mask


def _normalize_probabilities(prob: xr.DataArray) -> xr.DataArray:
    """Enforce finite probabilities in [0, 1] whose category sum is one."""
    prob = prob.clip(min=0.0, max=1.0)
    valid = prob.notnull().any(dim="probability")
    total = prob.sum(dim="probability", skipna=True)
    prob = xr.where(valid & (total > 0), prob / total, np.nan)
    return prob.transpose("probability", "T", "Y", "X")


def _finalize_probabilities(prob: xr.DataArray, mask: xr.DataArray) -> xr.DataArray:
    prob = prob.assign_coords(probability=("probability", ["PB", "PN", "PA"]))
    prob = prob.transpose("probability", "T", "Y", "X") * mask
    return _normalize_probabilities(prob)


def _conditional_nonnegative_clip(
    deterministic: xr.DataArray, Predictant: xr.DataArray
) -> xr.DataArray:
    """Clip at zero only for predictands whose observations are non-negative."""
    observed_min = Predictant.min(skipna=True)
    try:
        observed_min = observed_min.compute()
    except Exception:
        pass
    try:
        value = float(observed_min.values)
    except Exception:
        return deterministic
    if np.isfinite(value) and value >= 0.0:
        return deterministic.clip(min=0.0)
    return deterministic


def _target_time(Predictant: xr.DataArray, Predictor_for_year: xr.DataArray):
    year = int(Predictor_for_year["T"].dt.year.values[0])
    month = int(Predictant["T"].dt.month.values[0])
    return np.array([np.datetime64(f"{year:04d}-{month:02d}-01")], dtype="datetime64[ns]")


def _solve_weibull_shape(M: float, V: float) -> float:
    """Solve the Weibull shape robustly from a positive mean and variance."""
    if not np.isfinite(M) or not np.isfinite(V) or M <= 0 or V <= 0:
        return np.nan

    target = V / (M * M)

    def equation(k):
        g1 = gamma_function(1.0 + 1.0 / k)
        g2 = gamma_function(1.0 + 2.0 / k)
        return (g2 / (g1 * g1) - 1.0) - target

    try:
        return float(brentq(equation, 0.05, 100.0, maxiter=200))
    except (ValueError, RuntimeError, OverflowError, FloatingPointError):
        return np.nan


def _compute_probability_field(
    owner,
    deterministic: xr.DataArray,
    error_samples: xr.DataArray,
    error_variance: xr.DataArray,
    T1_emp: xr.DataArray,
    T2_emp: xr.DataArray,
    dof: int,
    mask: xr.DataArray,
    best_code_da=None,
    best_shape_da=None,
    best_loc_da=None,
    best_scale_da=None,
):
    """Apply the selected probability family without changing CCA transforms."""
    deterministic = _prepare_tyx(deterministic, "deterministic")
    dm = str(owner.dist_method).strip().lower()

    if dm == "bestfit":
        if any(v is None for v in (best_code_da, best_shape_da, best_loc_da, best_scale_da)):
            raise ValueError(
                "dist_method='bestfit' requires best_code_da, best_shape_da, "
                "best_loc_da and best_scale_da."
            )
        T1, T2 = xr.apply_ufunc(
            owner._ppf_terciles_from_code,
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
        code_da = best_code_da
        probability_function = owner.calculate_tercile_probabilities_bestfit

    elif dm == "nonparam":
        error_samples = _prepare_tyx(error_samples, "error_samples").rename(
            {"T": "T_error"}
        )
        prob = xr.apply_ufunc(
            owner.calculate_tercile_probabilities_nonparametric,
            deterministic,
            error_samples,
            T1_emp,
            T2_emp,
            input_core_dims=[("T",), ("T_error",), (), ()],
            output_core_dims=[("probability", "T")],
            vectorize=True,
            dask="parallelized",
            output_dtypes=[float],
            dask_gufunc_kwargs={
                "output_sizes": {"probability": 3},
                "allow_rechunk": True,
            },
        )
        return _finalize_probabilities(prob, mask)

    elif dm in _DISTRIBUTION_CODES:
        T1, T2 = T1_emp, T2_emp
        code_da = xr.full_like(error_variance, _DISTRIBUTION_CODES[dm], dtype=float)
        probability_function = owner.calculate_tercile_probabilities_bestfit

    else:
        valid = ["nonparam", "bestfit"] + sorted(set(_DISTRIBUTION_CODES))
        raise ValueError(
            f"Invalid dist_method={owner.dist_method!r}. Supported methods: {valid}."
        )

    prob = xr.apply_ufunc(
        probability_function,
        deterministic,
        error_variance,
        T1,
        T2,
        code_da,
        input_core_dims=[("T",), (), (), (), ()],
        output_core_dims=[("probability", "T")],
        vectorize=True,
        kwargs={"dof": dof},
        dask="parallelized",
        output_dtypes=[float],
        dask_gufunc_kwargs={
            "output_sizes": {"probability": 3},
            "allow_rechunk": True,
        },
    )
    return _finalize_probabilities(prob, mask)



class WAS_CCA:
    def __init__(self, n_modes=4, n_pca_modes=8, standardize=False, use_coslat=True, use_pca=True, dist_method="nonparam"):
        """
        Initialize the WAS_CCA class with specified parameters.

        Parameters:
        - n_modes: Number of canonical modes to compute.
        - n_pca_modes: Number of PCA modes to use before CCA.
        - standardize: Whether to standardize the data. Keep it False in our case data already standardize
        - use_coslat: Whether to use cosine latitude weighting.
        - use_pca: Whether to perform PCA before CCA.
        - detrend: Whether to apply detrending to the data. Extended EOF to detrend
        """
        
        self.n_modes = n_modes
        self.n_pca_modes = n_pca_modes
        self.standardize = standardize
        self.use_coslat = use_coslat
        self.use_pca = use_pca
        self.dist_method = dist_method

        self.cca = xe.cross.CCA(
            n_modes=self.n_modes,
            standardize=self.standardize,
            use_coslat=self.use_coslat,
            use_pca=self.use_pca,
            n_pca_modes=self.n_pca_modes
        )
        self.cca_model = None
    
    def fit_cca(self, X_train, y_train):
        X_train_final, y_train_final = self.preprocess_data(X_train, y_train)
        self.cca = _new_safe_cca(self, X_train_final, y_train_final)
        self.cca_model = self.cca.fit(X_train_final, y_train_final, dim="T")
        return self.cca_model
    def preprocess_data(self, X, Y):
        X = _prepare_tyx(X, "X_train")
        Y = _prepare_tyx(Y, "y_train")
        # Common upstream validity mask: impute partial gaps at otherwise-valid
        # cells with the training temporal mean, then EXCLUDE (leave NaN) cells
        # that are not fully observed or are temporally constant, so xeofs drops
        # those features instead of ingesting constant-zero fills. Mask/reference
        # are stored so CV and forecast inputs are masked identically.
        self._x_train_ref = X.mean(dim="T", skipna=True)
        self._y_train_ref = Y.mean(dim="T", skipna=True)
        X_imp = X.fillna(self._x_train_ref)
        Y_imp = Y.fillna(self._y_train_ref)
        self._x_valid_mask = _common_validity_mask(X_imp)
        self._y_valid_mask = _common_validity_mask(Y_imp)
        X_final = X_imp.where(self._x_valid_mask)
        Y_final = Y_imp.where(self._y_valid_mask)
        return (
            X_final.rename({"X": "lon", "Y": "lat"}).transpose("T", "lat", "lon"),
            Y_final.rename({"X": "lon", "Y": "lat"}).transpose("T", "lat", "lon"),
        )

    def preprocess_test_data(self, X_test, y_test, X_train, y_train):
        X_test = _prepare_tyx(X_test, "X_test")
        y_test = _prepare_tyx(y_test, "y_test")
        X_train = _prepare_tyx(X_train, "X_train")
        y_train = _prepare_tyx(y_train, "y_train")
        x_ref = getattr(self, "_x_train_ref", X_train.mean(dim="T", skipna=True))
        y_ref = getattr(self, "_y_train_ref", y_train.mean(dim="T", skipna=True))
        X_test_prepared = X_test.fillna(x_ref)
        y_test_prepared = y_test.fillna(y_ref)
        x_mask = getattr(self, "_x_valid_mask", None)
        y_mask = getattr(self, "_y_valid_mask", None)
        if x_mask is not None:
            X_test_prepared = X_test_prepared.where(x_mask)
        if y_mask is not None:
            y_test_prepared = y_test_prepared.where(y_mask)
        return (
            X_test_prepared.rename({"X": "lon", "Y": "lat"}).transpose("T", "lat", "lon"),
            y_test_prepared.rename({"X": "lon", "Y": "lat"}).transpose("T", "lat", "lon"),
        )

    def compute_model(self, X_train, y_train, X_test, y_test):
        self.fit_cca(X_train, y_train)
        X_test_prepared, y_test_prepared = self.preprocess_test_data(
            X_test, y_test, X_train, y_train
        )
        y_pred = self.cca_model.predict(X_test_prepared)
        y_pred = self.cca_model.inverse_transform(
            self.cca_model.transform(X_test_prepared), y_pred
        )[1]
        y_pred = y_pred.assign_coords(T=y_test_prepared["T"])
        return y_pred.rename({"lon": "X", "lat": "Y"}).transpose("T", "Y", "X")

    # --------------------------------------------------------------------------
    #  Probability Calculation Methods
    # --------------------------------------------------------------------------
    # ------------------ Probability Calculation Methods ------------------

    @staticmethod
    @staticmethod
    def _ppf_terciles_from_code(dist_code, shape, loc, scale):
        """Return exact one-third and two-third thresholds from fitted parameters."""
        if not np.isfinite(dist_code):
            return np.nan, np.nan

        q1, q2 = 1.0 / 3.0, 2.0 / 3.0
        code = int(dist_code)
        try:
            if code == 1:
                return norm.ppf(q1, loc=loc, scale=scale), norm.ppf(q2, loc=loc, scale=scale)
            if code == 2:
                return lognorm.ppf(q1, s=shape, loc=loc, scale=scale), lognorm.ppf(q2, s=shape, loc=loc, scale=scale)
            if code == 3:
                return expon.ppf(q1, loc=loc, scale=scale), expon.ppf(q2, loc=loc, scale=scale)
            if code == 4:
                return gamma.ppf(q1, a=shape, loc=loc, scale=scale), gamma.ppf(q2, a=shape, loc=loc, scale=scale)
            if code == 5:
                return weibull_min.ppf(q1, c=shape, loc=loc, scale=scale), weibull_min.ppf(q2, c=shape, loc=loc, scale=scale)
            if code == 6:
                return t.ppf(q1, df=shape, loc=loc, scale=scale), t.ppf(q2, df=shape, loc=loc, scale=scale)
            if code == 7:
                return poisson.ppf(q1, mu=shape, loc=loc), poisson.ppf(q2, mu=shape, loc=loc)
            if code == 8:
                return nbinom.ppf(q1, n=shape, p=scale, loc=loc), nbinom.ppf(q2, n=shape, p=scale, loc=loc)
        except Exception:
            return np.nan, np.nan
        return np.nan, np.nan
        
    @staticmethod
    @staticmethod
    def weibull_shape_solver(k, M, V):
        """Compatibility wrapper for the historical public helper."""
        k = float(np.asarray(k).reshape(-1)[0])
        if k <= 0 or M <= 0 or V <= 0:
            return np.nan
        g1 = gamma_function(1.0 + 1.0 / k)
        g2 = gamma_function(1.0 + 2.0 / k)
        return (g2 / (g1 ** 2) - 1.0) - V / (M ** 2)

    @staticmethod
    @staticmethod
    def calculate_tercile_probabilities_bestfit(
        best_guess, error_variance, T1, T2, dist_code, dof
    ):
        """Compute PB/PN/PA for the requested predictive distribution."""
        best_guess = np.asarray(best_guess, dtype=float)
        error_variance = float(error_variance)
        n_time = best_guess.size
        out = np.full((3, n_time), np.nan, dtype=float)

        if (
            np.all(~np.isfinite(best_guess))
            or not np.isfinite(error_variance)
            or error_variance <= 0
            or not np.isfinite(T1)
            or not np.isfinite(T2)
            or not np.isfinite(dist_code)
        ):
            return out

        code = int(dist_code)
        try:
            if code == 1:
                scale = np.sqrt(error_variance)
                c1 = norm.cdf(T1, loc=best_guess, scale=scale)
                c2 = norm.cdf(T2, loc=best_guess, scale=scale)

            elif code == 2:
                valid = best_guess > 0
                sigma = np.full(n_time, np.nan)
                mu = np.full(n_time, np.nan)
                sigma[valid] = np.sqrt(
                    np.log1p(error_variance / (best_guess[valid] ** 2))
                )
                mu[valid] = np.log(best_guess[valid]) - 0.5 * sigma[valid] ** 2
                c1 = lognorm.cdf(T1, s=sigma, scale=np.exp(mu))
                c2 = lognorm.cdf(T2, s=sigma, scale=np.exp(mu))

            elif code == 3:
                scale = np.sqrt(error_variance)
                loc = best_guess - scale
                c1 = expon.cdf(T1, loc=loc, scale=scale)
                c2 = expon.cdf(T2, loc=loc, scale=scale)

            elif code == 4:
                valid = best_guess > 0
                alpha = np.full(n_time, np.nan)
                theta = np.full(n_time, np.nan)
                alpha[valid] = best_guess[valid] ** 2 / error_variance
                theta[valid] = error_variance / best_guess[valid]
                c1 = gamma.cdf(T1, a=alpha, scale=theta)
                c2 = gamma.cdf(T2, a=alpha, scale=theta)

            elif code == 5:
                c1 = np.full(n_time, np.nan)
                c2 = np.full(n_time, np.nan)
                for i, mean_value in enumerate(best_guess):
                    shape = _solve_weibull_shape(mean_value, error_variance)
                    if not np.isfinite(shape):
                        continue
                    scale = mean_value / gamma_function(1.0 + 1.0 / shape)
                    c1[i] = weibull_min.cdf(T1, c=shape, loc=0.0, scale=scale)
                    c2[i] = weibull_min.cdf(T2, c=shape, loc=0.0, scale=scale)

            elif code == 6:
                if dof <= 2:
                    return out
                scale = np.sqrt(error_variance * (dof - 2.0) / dof)
                c1 = t.cdf(T1, df=dof, loc=best_guess, scale=scale)
                c2 = t.cdf(T2, df=dof, loc=best_guess, scale=scale)

            elif code == 7:
                mu = np.where(best_guess >= 0, best_guess, np.nan)
                c1 = poisson.cdf(T1, mu=mu)
                c2 = poisson.cdf(T2, mu=mu)

            elif code == 8:
                valid = (best_guess > 0) & (error_variance > best_guess)
                p = np.where(valid, best_guess / error_variance, np.nan)
                n = np.where(
                    valid,
                    best_guess ** 2 / (error_variance - best_guess),
                    np.nan,
                )
                c1 = nbinom.cdf(T1, n=n, p=p)
                c2 = nbinom.cdf(T2, n=n, p=p)

            else:
                return out

            out[0, :] = c1
            out[1, :] = c2 - c1
            out[2, :] = 1.0 - c2

        except (ValueError, FloatingPointError, OverflowError, RuntimeError):
            return out

        out = np.clip(out, 0.0, 1.0)
        total = np.nansum(out, axis=0)
        valid_total = np.isfinite(total) & (total > 0)
        out[:, valid_total] /= total[valid_total]
        return out

    @staticmethod
    @staticmethod
    def calculate_tercile_probabilities_nonparametric(
        best_guess, error_samples, first_tercile, second_tercile
    ):
        """Non-parametric probabilities from the historical residual sample."""
        best_guess = np.asarray(best_guess, dtype=float)
        valid_errors = np.asarray(error_samples, dtype=float)
        valid_errors = valid_errors[np.isfinite(valid_errors)]
        out = np.full((3, best_guess.size), np.nan, dtype=float)
        if valid_errors.size == 0:
            return out

        for i, prediction in enumerate(best_guess):
            if not np.isfinite(prediction):
                continue
            sample = prediction + valid_errors
            pb = np.mean(sample < first_tercile)
            pn = np.mean((sample >= first_tercile) & (sample < second_tercile))
            out[0, i] = pb
            out[1, i] = pn
            out[2, i] = 1.0 - pb - pn
        return out


    def compute_prob(
        self,
        Predictant: xr.DataArray,
        clim_year_start,
        clim_year_end,
        hindcast_det: xr.DataArray,
        best_code_da: xr.DataArray = None,
        best_shape_da: xr.DataArray = None,
        best_loc_da: xr.DataArray = None,
        best_scale_da: xr.DataArray = None,
    ) -> xr.DataArray:
        """Compute hindcast tercile probabilities with guaranteed output order."""
        Predictant, hindcast_det = _align_observation_and_hindcast(
            Predictant, hindcast_det
        )
        mask = _spatial_mask(Predictant)
        clim = Predictant.sel(T=slice(str(clim_year_start), str(clim_year_end)))
        if clim.sizes.get("T", 0) < 3:
            raise ValueError("Not enough years in climatology period for terciles.")

        terciles = clim.quantile([1.0 / 3.0, 2.0 / 3.0], dim="T")
        T1_emp = terciles.isel(quantile=0, drop=True)
        T2_emp = terciles.isel(quantile=1, drop=True)
        error_samples = Predictant - hindcast_det
        error_variance = error_samples.var(dim="T", skipna=True)
        dof = max(int(clim.sizes["T"]) - 1, 2)

        return _compute_probability_field(
            self,
            deterministic=hindcast_det,
            error_samples=error_samples,
            error_variance=error_variance,
            T1_emp=T1_emp,
            T2_emp=T2_emp,
            dof=dof,
            mask=mask,
            best_code_da=best_code_da,
            best_shape_da=best_shape_da,
            best_loc_da=best_loc_da,
            best_scale_da=best_scale_da,
        )


    def forecast(
        self, Predictant, clim_year_start, clim_year_end, Predictor,
        hindcast_det, Predictor_for_year, best_code_da=None,
        best_shape_da=None, best_loc_da=None, best_scale_da=None
    ):
        Predictant = _prepare_tyx(Predictant, "Predictant")
        Predictor = _prepare_tyx(Predictor, "Predictor")
        Predictor_for_year = _prepare_tyx(Predictor_for_year, "Predictor_for_year")
        Predictant_resid, hindcast_resid = _align_observation_and_hindcast(
            Predictant, hindcast_det
        )
        mask = _spatial_mask(Predictant)

        # Preserve the original WAS_CCA EEOF-removal logic.
        trend_X = extract_leading_eeof_component(Predictor)
        ref_X = trend_X.isel(T=max(0, trend_X.sizes["T"] - 3))
        Predictor_ = (Predictor - trend_X.fillna(ref_X)).fillna(0.0)

        Predictant_st = standardize_timeseries(
            Predictant, clim_year_start, clim_year_end
        )
        trend_Y = extract_leading_eeof_component(Predictant_st)
        ref_Y = trend_Y.isel(T=max(0, trend_Y.sizes["T"] - 3))
        Predictant_ = (Predictant_st - trend_Y.fillna(ref_Y)).fillna(0.0)

        Predictor_for_year_ = Predictor_for_year.fillna(
            Predictor.mean(dim="T", skipna=True)
        )
        Predictor_for_year_ = _fill_spatial_gaps_safe(
            Predictor_for_year_
        ).fillna(0.0).transpose("T", "Y", "X")

        self.fit_cca(Predictor_, Predictant_)
        # Apply the SAME upstream validity mask used at fit time (exclude masked
        # cells as NaN instead of feeding constant zeros) so predict matches fit.
        X_test_prepared = (
            Predictor_for_year_
            .fillna(self._x_train_ref)
            .where(self._x_valid_mask)
            .rename({"X": "lon", "Y": "lat"})
            .transpose("T", "lat", "lon")
        )
        y_pred = self.cca_model.predict(X_test_prepared)
        y_pred = self.cca_model.inverse_transform(
            self.cca_model.transform(X_test_prepared), y_pred
        )[1]
        result_ = y_pred.rename({"lon": "X", "lat": "Y"}).transpose("T", "Y", "X")
        forecast_det = reverse_standardize(
            result_, Predictant, clim_year_start, clim_year_end
        )
        forecast_det = forecast_det.assign_coords(
            T=xr.DataArray(_target_time(Predictant, Predictor_for_year), dims=["T"])
        ).transpose("T", "Y", "X")
        forecast_det = _conditional_nonnegative_clip(forecast_det, Predictant)

        clim = Predictant.sel(T=slice(str(clim_year_start), str(clim_year_end)))
        if clim.sizes.get("T", 0) < 3:
            raise ValueError("Not enough years in climatology period for terciles.")
        terciles = clim.quantile([1.0 / 3.0, 2.0 / 3.0], dim="T")
        T1_emp = terciles.isel(quantile=0, drop=True)
        T2_emp = terciles.isel(quantile=1, drop=True)
        error_samples = Predictant_resid - hindcast_resid
        error_variance = error_samples.var(dim="T", skipna=True)
        dof = max(int(clim.sizes["T"]) - 1, 2)

        forecast_prob = _compute_probability_field(
            self, forecast_det, error_samples, error_variance,
            T1_emp, T2_emp, dof, mask,
            best_code_da, best_shape_da, best_loc_da, best_scale_da,
        )
        return (forecast_det * mask).transpose("T", "Y", "X"), forecast_prob
        

    def plot_cca_results(self, X=None, Y=None, n_modes=None, clim_year_start=None, clim_year_end=None):
        """
        Plots the CCA modes and scores.

        Parameters:
        - X: Optional xarray DataArray for predictors. If provided, the model will be fitted using X and Y.
        - Y: Optional xarray DataArray for predictands.
        - n_modes: Number of modes to plot. If None, plots all modes.
        """
        if X is not None and Y is not None:
            mask = xr.where(~np.isnan(Y.isel(T=0)), 1, np.nan).drop_vars(['T']).squeeze().to_numpy()
            # mask.name = None
            
            X_ = standardize_timeseries(X, clim_year_start, clim_year_end) - extract_leading_eeof_component(standardize_timeseries(X, clim_year_start, clim_year_end)).fillna(extract_leading_eeof_component(standardize_timeseries(X, clim_year_start, clim_year_end))[-3])
            Y_ = standardize_timeseries(Y, clim_year_start, clim_year_end) - extract_leading_eeof_component(standardize_timeseries(Y, clim_year_start, clim_year_end)).fillna(extract_leading_eeof_component(standardize_timeseries(Y, clim_year_start, clim_year_end))[-3])
            
            # Fit the model using the provided data
            self.fit_cca(X_.isel(T= slice(0,-2)).fillna(0), Y_.isel(T=slice(0,-2)).fillna(0))
        elif self.cca_model is None:
            raise ValueError("The CCA model has not been fitted yet. Provide X and Y data to fit the model.")

        # Get components (modes) and scores
        X_modes, Y_modes = self.cca_model.components()  # Spatial patterns
        X_scores, Y_scores = self.cca_model.scores()    # Temporal projections (canonical variates)

        # Get explained variances
        var_explained_X = self.cca_model.fraction_variance_X_explained_by_X()
        var_explained_Y = self.cca_model.fraction_variance_Y_explained_by_Y()
        var_explained_Y_by_X = self.cca_model.fraction_variance_Y_explained_by_X()

        # Determine number of modes to plot
        if n_modes is None:
            n_modes = self.n_modes

        # Mode indices start from 1 in xeofs
        mode_indices = range(1, n_modes + 1)

        # Create subplots
        fig, axes = plt.subplots(n_modes, 3, figsize=(15, 3 * n_modes))

        if n_modes == 1:
            axes = np.array([axes])

        for i, mode in enumerate(mode_indices):

            # First Column: Plot X_modes
            ax = axes[i, 0]
            X_mode = X_modes.sel(mode=mode)
            X_mode.plot(ax=ax, vmin=-1, vmax=1, cmap= "RdBu_r")
            var_X = var_explained_X.sel(mode=mode).values * 100
            ax.set_title(f'X Mode {mode} ({var_X:.2f}% variance explained)')

            # Second Column: Plot X_scores and Y_scores
            ax = axes[i, 1]
            X_score = X_scores.sel(mode=mode)
            Y_score = Y_scores.sel(mode=mode)
            var_Y_X = var_explained_Y_by_X.sel(mode=mode).values * 100
            ax.plot(X_score['T'].dt.year.values, X_score, label='X Score')
            ax.plot(Y_score['T'].dt.year.values, Y_score, label='Y Score')
            ax.axhline(0, linestyle='--', lw=0.8, label="") #### line Canonical Variate = 0
            ax.legend()
            ax.set_title(f'Scores for Mode {mode} ({var_Y_X:.2f}% variance Y explained by X)')
            ax.set_xlabel('Time')
            ax.set_ylabel('Canonical Variate')

            # Third Column: Plot Y_modes
            ax = axes[i, 2]
            Y_mode = (Y_modes.sel(mode=mode))*mask
            Y_mode.plot(ax=ax, vmin=None, vmax=None, cmap= "RdBu_r")
            var_Y = var_explained_Y.sel(mode=mode).values * 100
            ax.set_title(f'Y Mode {mode} ({var_Y:.2f}% variance explained)')
            
        plt.tight_layout()
        plt.show()


class WAS_CCA_base:
    """
    Canonical Correlation Analysis model based on xeofs.cross.CCA.

    Important:
    ----------
    - No detrend no extract_leading_eeof_component is used here.
    - The CCA basis is entirely handled by xeofs.cross.CCA.
    - The predictand is standardized before CCA in forecast/cross-validation,
      then transformed back to the original scale with reverse_standardize.
    - Missing values are handled using training-period means to avoid leakage.
    """

    def __init__(
        self,
        n_modes=4,
        n_pca_modes=8,
        standardize=False,
        use_coslat=True,
        use_pca=True,
        dist_method="nonparam",
    ):
        self.n_modes = n_modes
        self.n_pca_modes = n_pca_modes
        self.standardize = standardize
        self.use_coslat = use_coslat
        self.use_pca = use_pca
        self.dist_method = dist_method

        self.cca = None
        self.cca_model = None

    # ---------------------------------------------------------------------
    # Internal utilities
    # ---------------------------------------------------------------------

    def _new_cca(self):
        """
        Create a fresh xeofs CCA object.

        This is important during cross-validation because each fold must be
        fitted independently.
        """
        return xe.cross.CCA(
            n_modes=self.n_modes,
            standardize=self.standardize,
            use_coslat=self.use_coslat,
            use_pca=self.use_pca,
            n_pca_modes=self.n_pca_modes,
        )

    @staticmethod
    def _drop_member_dim(da: xr.DataArray) -> xr.DataArray:
        """
        If the DataArray has an ensemble member dimension M, keep the first member.
        """
        if "M" in da.dims:
            da = da.isel(M=0, drop=True)
        return da

    @staticmethod
    def _ensure_tyx(da: xr.DataArray, name="data") -> xr.DataArray:
        """
        Ensure data has dimensions T, Y, X and is ordered as T, Y, X.
        """
        missing = [d for d in ("T", "Y", "X") if d not in da.dims]
        if missing:
            raise ValueError(f"{name} must contain dimensions T, Y, X. Missing: {missing}")

        return da.transpose("T", "Y", "X")

    @staticmethod
    def _to_xeofs_dims(da: xr.DataArray) -> xr.DataArray:
        """
        Rename WAS dimensions to xeofs-compatible dimensions.
        WAS:   T, Y, X
        xeofs: T, lat, lon
        """
        rename = {}
        if "Y" in da.dims:
            rename["Y"] = "lat"
        if "X" in da.dims:
            rename["X"] = "lon"

        da = da.rename(rename)

        return da.transpose("T", "lat", "lon")

    @staticmethod
    def _from_xeofs_dims(da: xr.DataArray) -> xr.DataArray:
        """
        Rename xeofs dimensions back to WAS dimensions.
        xeofs: T, lat, lon
        WAS:   T, Y, X
        """
        rename = {}
        if "lat" in da.dims:
            rename["lat"] = "Y"
        if "lon" in da.dims:
            rename["lon"] = "X"

        da = da.rename(rename)

        return da.transpose("T", "Y", "X")

    @staticmethod
    def _fill_train_data(da: xr.DataArray) -> xr.DataArray:
        """
        Fill missing values in training data using the training mean along T.
        Remaining NaNs are filled with 0.
        """
        mean_da = da.mean(dim="T", skipna=True)
        return da.fillna(mean_da).fillna(0)

    @staticmethod
    def _fill_test_data(test_da: xr.DataArray, train_da: xr.DataArray) -> xr.DataArray:
        """
        Fill missing values in test/forecast data using the training mean along T.
        Remaining NaNs are filled with 0.

        This avoids using information from the test fold.
        """
        train_mean = train_da.mean(dim="T", skipna=True)
        return test_da.fillna(train_mean).fillna(0)

    @staticmethod
    def _spatial_mask(Predictant: xr.DataArray) -> xr.DataArray:
        """
        Spatial mask based on valid predictand grid cells at the first time step.
        """
        return xr.where(np.isfinite(Predictant.isel(T=0)), 1.0, np.nan)

    @staticmethod
    @staticmethod
    def _normalize_probabilities(prob):
        return _normalize_probabilities(prob)

    @staticmethod
    def _make_forecast_time(Predictant: xr.DataArray, Predictor_for_year: xr.DataArray):
        """
        Construct forecast time using:
        - the year of Predictor_for_year
        - the month of the predictand seasonal target
        """
        forecast_year = (
            Predictor_for_year.coords["T"]
            .values.astype("datetime64[Y]")
            .astype(int)[0]
            + 1970
        )

        first_target_time = Predictant.isel(T=0).coords["T"].values
        target_month = first_target_time.astype("datetime64[M]").astype(int) % 12 + 1

        new_time = np.datetime64(f"{forecast_year}-{target_month:02d}-01")
        return np.array([new_time], dtype="datetime64[ns]")

    # ---------------------------------------------------------------------
    # CCA preprocessing and model fitting
    # ---------------------------------------------------------------------

    def preprocess_data(self, X, Y):
        X = _prepare_tyx(X, "X_train")
        Y = _prepare_tyx(Y, "y_train")
        # Common upstream validity mask: impute partial gaps at otherwise-valid
        # cells with the training temporal mean, then EXCLUDE (leave NaN) cells
        # that are not fully observed or are temporally constant, so xeofs drops
        # those features instead of ingesting constant-zero fills. Mask/reference
        # are stored so CV and forecast inputs are masked identically.
        self._x_train_ref = X.mean(dim="T", skipna=True)
        self._y_train_ref = Y.mean(dim="T", skipna=True)
        X_imp = X.fillna(self._x_train_ref)
        Y_imp = Y.fillna(self._y_train_ref)
        self._x_valid_mask = _common_validity_mask(X_imp)
        self._y_valid_mask = _common_validity_mask(Y_imp)
        X_final = X_imp.where(self._x_valid_mask)
        Y_final = Y_imp.where(self._y_valid_mask)
        return (
            X_final.rename({"X": "lon", "Y": "lat"}).transpose("T", "lat", "lon"),
            Y_final.rename({"X": "lon", "Y": "lat"}).transpose("T", "lat", "lon"),
        )

    def preprocess_test_data(self, X_test, y_test, X_train, y_train):
        X_test = _prepare_tyx(X_test, "X_test")
        y_test = _prepare_tyx(y_test, "y_test")
        X_train = _prepare_tyx(X_train, "X_train")
        y_train = _prepare_tyx(y_train, "y_train")
        x_ref = getattr(self, "_x_train_ref", X_train.mean(dim="T", skipna=True))
        y_ref = getattr(self, "_y_train_ref", y_train.mean(dim="T", skipna=True))
        X_test_prepared = X_test.fillna(x_ref)
        y_test_prepared = y_test.fillna(y_ref)
        x_mask = getattr(self, "_x_valid_mask", None)
        y_mask = getattr(self, "_y_valid_mask", None)
        if x_mask is not None:
            X_test_prepared = X_test_prepared.where(x_mask)
        if y_mask is not None:
            y_test_prepared = y_test_prepared.where(y_mask)
        return (
            X_test_prepared.rename({"X": "lon", "Y": "lat"}).transpose("T", "lat", "lon"),
            y_test_prepared.rename({"X": "lon", "Y": "lat"}).transpose("T", "lat", "lon"),
        )

    def preprocess_forecast_data(
        self,
        Predictor_for_year: xr.DataArray,
        Predictor_train: xr.DataArray,
    ):
        """
        Prepare real-time predictor data for forecast.

        Missing values are filled using the historical predictor mean.
        """
        Predictor_for_year = self._drop_member_dim(Predictor_for_year)
        Predictor_train = self._drop_member_dim(Predictor_train)

        Predictor_for_year = self._ensure_tyx(Predictor_for_year, name="Predictor_for_year")
        Predictor_train = self._ensure_tyx(Predictor_train, name="Predictor_train")

        Predictor_for_year = self._fill_test_data(Predictor_for_year, Predictor_train)
        Predictor_for_year = self._to_xeofs_dims(Predictor_for_year)

        return Predictor_for_year

    def fit_cca(self, X_train, y_train):
        X_train_final, y_train_final = self.preprocess_data(X_train, y_train)
        self.cca = _new_safe_cca(self, X_train_final, y_train_final)
        self.cca_model = self.cca.fit(X_train_final, y_train_final, dim="T")
        return self.cca_model

    def compute_model(self, X_train, y_train, X_test, y_test):
        self.fit_cca(X_train, y_train)
        X_test_prepared, y_test_prepared = self.preprocess_test_data(
            X_test, y_test, X_train, y_train
        )
        y_pred = self.cca_model.predict(X_test_prepared)
        y_pred = self.cca_model.inverse_transform(
            self.cca_model.transform(X_test_prepared), y_pred
        )[1]
        y_pred = y_pred.assign_coords(T=y_test_prepared["T"])
        return y_pred.rename({"lon": "X", "lat": "Y"}).transpose("T", "Y", "X")

    # ---------------------------------------------------------------------
    # Probability calculation utilities
    # ---------------------------------------------------------------------

    @staticmethod
    @staticmethod
    def _ppf_terciles_from_code(dist_code, shape, loc, scale):
        """Return exact one-third and two-third thresholds from fitted parameters."""
        if not np.isfinite(dist_code):
            return np.nan, np.nan

        q1, q2 = 1.0 / 3.0, 2.0 / 3.0
        code = int(dist_code)
        try:
            if code == 1:
                return norm.ppf(q1, loc=loc, scale=scale), norm.ppf(q2, loc=loc, scale=scale)
            if code == 2:
                return lognorm.ppf(q1, s=shape, loc=loc, scale=scale), lognorm.ppf(q2, s=shape, loc=loc, scale=scale)
            if code == 3:
                return expon.ppf(q1, loc=loc, scale=scale), expon.ppf(q2, loc=loc, scale=scale)
            if code == 4:
                return gamma.ppf(q1, a=shape, loc=loc, scale=scale), gamma.ppf(q2, a=shape, loc=loc, scale=scale)
            if code == 5:
                return weibull_min.ppf(q1, c=shape, loc=loc, scale=scale), weibull_min.ppf(q2, c=shape, loc=loc, scale=scale)
            if code == 6:
                return t.ppf(q1, df=shape, loc=loc, scale=scale), t.ppf(q2, df=shape, loc=loc, scale=scale)
            if code == 7:
                return poisson.ppf(q1, mu=shape, loc=loc), poisson.ppf(q2, mu=shape, loc=loc)
            if code == 8:
                return nbinom.ppf(q1, n=shape, p=scale, loc=loc), nbinom.ppf(q2, n=shape, p=scale, loc=loc)
        except Exception:
            return np.nan, np.nan
        return np.nan, np.nan

    @staticmethod
    @staticmethod
    def weibull_shape_solver(k, M, V):
        """Compatibility wrapper for the historical public helper."""
        k = float(np.asarray(k).reshape(-1)[0])
        if k <= 0 or M <= 0 or V <= 0:
            return np.nan
        g1 = gamma_function(1.0 + 1.0 / k)
        g2 = gamma_function(1.0 + 2.0 / k)
        return (g2 / (g1 ** 2) - 1.0) - V / (M ** 2)

    @staticmethod
    @staticmethod
    def calculate_tercile_probabilities_bestfit(
        best_guess, error_variance, T1, T2, dist_code, dof
    ):
        """Compute PB/PN/PA for the requested predictive distribution."""
        best_guess = np.asarray(best_guess, dtype=float)
        error_variance = float(error_variance)
        n_time = best_guess.size
        out = np.full((3, n_time), np.nan, dtype=float)

        if (
            np.all(~np.isfinite(best_guess))
            or not np.isfinite(error_variance)
            or error_variance <= 0
            or not np.isfinite(T1)
            or not np.isfinite(T2)
            or not np.isfinite(dist_code)
        ):
            return out

        code = int(dist_code)
        try:
            if code == 1:
                scale = np.sqrt(error_variance)
                c1 = norm.cdf(T1, loc=best_guess, scale=scale)
                c2 = norm.cdf(T2, loc=best_guess, scale=scale)

            elif code == 2:
                valid = best_guess > 0
                sigma = np.full(n_time, np.nan)
                mu = np.full(n_time, np.nan)
                sigma[valid] = np.sqrt(
                    np.log1p(error_variance / (best_guess[valid] ** 2))
                )
                mu[valid] = np.log(best_guess[valid]) - 0.5 * sigma[valid] ** 2
                c1 = lognorm.cdf(T1, s=sigma, scale=np.exp(mu))
                c2 = lognorm.cdf(T2, s=sigma, scale=np.exp(mu))

            elif code == 3:
                scale = np.sqrt(error_variance)
                loc = best_guess - scale
                c1 = expon.cdf(T1, loc=loc, scale=scale)
                c2 = expon.cdf(T2, loc=loc, scale=scale)

            elif code == 4:
                valid = best_guess > 0
                alpha = np.full(n_time, np.nan)
                theta = np.full(n_time, np.nan)
                alpha[valid] = best_guess[valid] ** 2 / error_variance
                theta[valid] = error_variance / best_guess[valid]
                c1 = gamma.cdf(T1, a=alpha, scale=theta)
                c2 = gamma.cdf(T2, a=alpha, scale=theta)

            elif code == 5:
                c1 = np.full(n_time, np.nan)
                c2 = np.full(n_time, np.nan)
                for i, mean_value in enumerate(best_guess):
                    shape = _solve_weibull_shape(mean_value, error_variance)
                    if not np.isfinite(shape):
                        continue
                    scale = mean_value / gamma_function(1.0 + 1.0 / shape)
                    c1[i] = weibull_min.cdf(T1, c=shape, loc=0.0, scale=scale)
                    c2[i] = weibull_min.cdf(T2, c=shape, loc=0.0, scale=scale)

            elif code == 6:
                if dof <= 2:
                    return out
                scale = np.sqrt(error_variance * (dof - 2.0) / dof)
                c1 = t.cdf(T1, df=dof, loc=best_guess, scale=scale)
                c2 = t.cdf(T2, df=dof, loc=best_guess, scale=scale)

            elif code == 7:
                mu = np.where(best_guess >= 0, best_guess, np.nan)
                c1 = poisson.cdf(T1, mu=mu)
                c2 = poisson.cdf(T2, mu=mu)

            elif code == 8:
                valid = (best_guess > 0) & (error_variance > best_guess)
                p = np.where(valid, best_guess / error_variance, np.nan)
                n = np.where(
                    valid,
                    best_guess ** 2 / (error_variance - best_guess),
                    np.nan,
                )
                c1 = nbinom.cdf(T1, n=n, p=p)
                c2 = nbinom.cdf(T2, n=n, p=p)

            else:
                return out

            out[0, :] = c1
            out[1, :] = c2 - c1
            out[2, :] = 1.0 - c2

        except (ValueError, FloatingPointError, OverflowError, RuntimeError):
            return out

        out = np.clip(out, 0.0, 1.0)
        total = np.nansum(out, axis=0)
        valid_total = np.isfinite(total) & (total > 0)
        out[:, valid_total] /= total[valid_total]
        return out

    @staticmethod
    @staticmethod
    def calculate_tercile_probabilities_nonparametric(
        best_guess, error_samples, first_tercile, second_tercile
    ):
        """Non-parametric probabilities from the historical residual sample."""
        best_guess = np.asarray(best_guess, dtype=float)
        valid_errors = np.asarray(error_samples, dtype=float)
        valid_errors = valid_errors[np.isfinite(valid_errors)]
        out = np.full((3, best_guess.size), np.nan, dtype=float)
        if valid_errors.size == 0:
            return out

        for i, prediction in enumerate(best_guess):
            if not np.isfinite(prediction):
                continue
            sample = prediction + valid_errors
            pb = np.mean(sample < first_tercile)
            pn = np.mean((sample >= first_tercile) & (sample < second_tercile))
            out[0, i] = pb
            out[1, i] = pn
            out[2, i] = 1.0 - pb - pn
        return out

    def _compute_tercile_probabilities(
        self,
        Predictant: xr.DataArray,
        deterministic: xr.DataArray,
        clim_year_start,
        clim_year_end,
        error_samples: xr.DataArray,
        error_variance: xr.DataArray,
        best_code_da: xr.DataArray = None,
        best_shape_da: xr.DataArray = None,
        best_loc_da: xr.DataArray = None,
        best_scale_da: xr.DataArray = None,
    ) -> xr.DataArray:
        """Backward-compatible probability entry point used by older workflows."""
        Predictant = _prepare_tyx(Predictant, "Predictant")
        deterministic = _prepare_tyx(deterministic, "deterministic")
        mask = _spatial_mask(Predictant)
        clim = Predictant.sel(T=slice(str(clim_year_start), str(clim_year_end)))
        if clim.sizes.get("T", 0) < 3:
            raise ValueError("Not enough years in climatology period for terciles.")
        terciles = clim.quantile([1.0 / 3.0, 2.0 / 3.0], dim="T")
        T1_emp = terciles.isel(quantile=0, drop=True)
        T2_emp = terciles.isel(quantile=1, drop=True)
        dof = max(int(clim.sizes["T"]) - 1, 2)
        return _compute_probability_field(
            self, deterministic, error_samples, error_variance,
            T1_emp, T2_emp, dof, mask,
            best_code_da, best_shape_da, best_loc_da, best_scale_da,
        )

    def compute_prob(
        self,
        Predictant: xr.DataArray,
        clim_year_start,
        clim_year_end,
        hindcast_det: xr.DataArray,
        best_code_da: xr.DataArray = None,
        best_shape_da: xr.DataArray = None,
        best_loc_da: xr.DataArray = None,
        best_scale_da: xr.DataArray = None,
    ) -> xr.DataArray:
        """Compute hindcast tercile probabilities with guaranteed output order."""
        Predictant, hindcast_det = _align_observation_and_hindcast(
            Predictant, hindcast_det
        )
        mask = _spatial_mask(Predictant)
        clim = Predictant.sel(T=slice(str(clim_year_start), str(clim_year_end)))
        if clim.sizes.get("T", 0) < 3:
            raise ValueError("Not enough years in climatology period for terciles.")

        terciles = clim.quantile([1.0 / 3.0, 2.0 / 3.0], dim="T")
        T1_emp = terciles.isel(quantile=0, drop=True)
        T2_emp = terciles.isel(quantile=1, drop=True)
        error_samples = Predictant - hindcast_det
        error_variance = error_samples.var(dim="T", skipna=True)
        dof = max(int(clim.sizes["T"]) - 1, 2)

        return _compute_probability_field(
            self,
            deterministic=hindcast_det,
            error_samples=error_samples,
            error_variance=error_variance,
            T1_emp=T1_emp,
            T2_emp=T2_emp,
            dof=dof,
            mask=mask,
            best_code_da=best_code_da,
            best_shape_da=best_shape_da,
            best_loc_da=best_loc_da,
            best_scale_da=best_scale_da,
        )

    # ---------------------------------------------------------------------
    # Real-time forecast
    # ---------------------------------------------------------------------

    def forecast(
        self, Predictant, clim_year_start, clim_year_end, Predictor,
        hindcast_det, Predictor_for_year, best_code_da=None,
        best_shape_da=None, best_loc_da=None, best_scale_da=None,
    ):
        Predictant = _prepare_tyx(Predictant, "Predictant")
        Predictor = _prepare_tyx(Predictor, "Predictor")
        Predictor_for_year = _prepare_tyx(Predictor_for_year, "Predictor_for_year")
        Predictant_resid, hindcast_resid = _align_observation_and_hindcast(
            Predictant, hindcast_det
        )
        mask = _spatial_mask(Predictant)

        # Preserve WAS_CCA_base: no EEOF detrending of the predictor.
        Predictant_st = standardize_timeseries(
            Predictant, clim_year_start, clim_year_end
        )
        self.fit_cca(Predictor, Predictant_st)
        # Same upstream validity mask used at fit time (exclude masked cells as NaN).
        X_forecast = (
            Predictor_for_year
            .fillna(self._x_train_ref)
            .where(self._x_valid_mask)
            .rename({"X": "lon", "Y": "lat"})
            .transpose("T", "lat", "lon")
        )
        y_pred = self.cca_model.predict(X_forecast)
        y_pred = self.cca_model.inverse_transform(
            self.cca_model.transform(X_forecast), y_pred
        )[1]
        forecast_st = y_pred.rename({"lon": "X", "lat": "Y"}).transpose("T", "Y", "X")
        forecast_det = reverse_standardize(
            forecast_st, Predictant, clim_year_start, clim_year_end
        )
        forecast_det = forecast_det.assign_coords(
            T=xr.DataArray(_target_time(Predictant, Predictor_for_year), dims=["T"])
        ).transpose("T", "Y", "X")
        forecast_det = _conditional_nonnegative_clip(forecast_det, Predictant)

        clim = Predictant.sel(T=slice(str(clim_year_start), str(clim_year_end)))
        if clim.sizes.get("T", 0) < 3:
            raise ValueError("Not enough years in climatology period for terciles.")
        terciles = clim.quantile([1.0 / 3.0, 2.0 / 3.0], dim="T")
        T1_emp = terciles.isel(quantile=0, drop=True)
        T2_emp = terciles.isel(quantile=1, drop=True)
        error_samples = Predictant_resid - hindcast_resid
        error_variance = error_samples.var(dim="T", skipna=True)
        dof = max(int(clim.sizes["T"]) - 1, 2)
        forecast_prob = _compute_probability_field(
            self, forecast_det, error_samples, error_variance,
            T1_emp, T2_emp, dof, mask,
            best_code_da, best_shape_da, best_loc_da, best_scale_da,
        )
        return (forecast_det * mask).transpose("T", "Y", "X"), forecast_prob

    # ---------------------------------------------------------------------
    # Plot CCA modes
    # ---------------------------------------------------------------------

    def plot_cca_results(
        self,
        X: xr.DataArray = None,
        Y: xr.DataArray = None,
        n_modes=None,
        clim_year_start=None,
        clim_year_end=None,
    ):
        """
        Plot CCA spatial modes and canonical scores.
        """

        def _spatial_to_was(da):
            """Rename xeofs lat/lon -> WAS Y/X for a spatial mode (no T axis)."""
            rename = {}
            if "lat" in da.dims:
                rename["lat"] = "Y"
            if "lon" in da.dims:
                rename["lon"] = "X"
            da = da.rename(rename)
            order = [d for d in ("T", "Y", "X") if d in da.dims]
            return da.transpose(*order)

        if X is not None and Y is not None:
            X = self._drop_member_dim(X)
            Y = self._drop_member_dim(Y)

            X = self._ensure_tyx(X, name="X")
            Y = self._ensure_tyx(Y, name="Y")

            mask = self._spatial_mask(Y)

            if clim_year_start is not None and clim_year_end is not None:
                Y_fit = standardize_timeseries(Y, clim_year_start, clim_year_end)
            else:
                Y_fit = Y

            X_fit = X

            self.fit_cca(X_fit, Y_fit)

        elif self.cca_model is None:
            raise ValueError(
                "The CCA model has not been fitted yet. "
                "Provide X and Y or call fit_cca first."
            )
        else:
            mask = None

        X_modes, Y_modes = self.cca_model.components()
        X_scores, Y_scores = self.cca_model.scores()

        var_explained_X = self.cca_model.fraction_variance_X_explained_by_X()
        var_explained_Y = self.cca_model.fraction_variance_Y_explained_by_Y()
        var_explained_Y_by_X = self.cca_model.fraction_variance_Y_explained_by_X()

        if n_modes is None:
            n_modes = self.n_modes

        fig, axes = plt.subplots(
            n_modes,
            3,
            figsize=(15, 3.5 * n_modes),
            squeeze=False,
        )

        for i, mode in enumerate(range(1, n_modes + 1)):
            ax = axes[i, 0]
            X_mode = X_modes.sel(mode=mode)
            X_mode.plot(ax=ax, cmap="RdBu_r")
            var_X = float(var_explained_X.sel(mode=mode).values) * 100
            ax.set_title(f"X Mode {mode} ({var_X:.2f}% variance)")

            ax = axes[i, 1]
            X_score = X_scores.sel(mode=mode)
            Y_score = Y_scores.sel(mode=mode)
            var_Y_by_X = float(var_explained_Y_by_X.sel(mode=mode).values) * 100

            ax.plot(X_score["T"].dt.year.values, X_score, label="X score")
            ax.plot(Y_score["T"].dt.year.values, Y_score, label="Y score")
            ax.axhline(0, linestyle="--", linewidth=0.8)
            ax.legend()
            ax.set_title(f"Scores Mode {mode} ({var_Y_by_X:.2f}% Y by X)")
            ax.set_xlabel("Time")
            ax.set_ylabel("Canonical variate")

            ax = axes[i, 2]
            Y_mode = _spatial_to_was(Y_modes.sel(mode=mode))
            if mask is not None:
                Y_mode = Y_mode * mask
            Y_mode.plot(ax=ax, cmap="RdBu_r")
            var_Y = float(var_explained_Y.sel(mode=mode).values) * 100
            ax.set_title(f"Y Mode {mode} ({var_Y:.2f}% variance)")

        plt.tight_layout()
        plt.show()


class WAS_CCA_op:
    def __init__(self, n_modes=4, n_pca_modes=8, standardize=False, use_coslat=True, use_pca=True, dist_method="bestfit"):
        """
        Initialize parameters. The CCA model instance is created in fit_cca to ensure safe CV.
        """
        self.n_modes = n_modes
        self.n_pca_modes = n_pca_modes
        self.standardize = standardize
        self.use_coslat = use_coslat
        self.use_pca = use_pca
        self.dist_method = dist_method
        
        self.cca_model = None
        self.cca = None

    def fit_cca(self, X_train, y_train):
        X_train_final, y_train_final = self.preprocess_data(X_train, y_train)
        self.cca = _new_safe_cca(self, X_train_final, y_train_final)
        self.cca_model = self.cca.fit(X_train_final, y_train_final, dim="T")
        return self.cca_model

    def preprocess_data(self, X, Y):
        X = _prepare_tyx(X, "X_train")
        Y = _prepare_tyx(Y, "y_train")
        # Common upstream validity mask: impute partial gaps at otherwise-valid
        # cells with the training temporal mean, then EXCLUDE (leave NaN) cells
        # that are not fully observed or are temporally constant, so xeofs drops
        # those features instead of ingesting constant-zero fills. Mask/reference
        # are stored so CV and forecast inputs are masked identically.
        self._x_train_ref = X.mean(dim="T", skipna=True)
        self._y_train_ref = Y.mean(dim="T", skipna=True)
        X_imp = X.fillna(self._x_train_ref)
        Y_imp = Y.fillna(self._y_train_ref)
        self._x_valid_mask = _common_validity_mask(X_imp)
        self._y_valid_mask = _common_validity_mask(Y_imp)
        X_final = X_imp.where(self._x_valid_mask)
        Y_final = Y_imp.where(self._y_valid_mask)
        return (
            X_final.rename({"X": "lon", "Y": "lat"}).transpose("T", "lat", "lon"),
            Y_final.rename({"X": "lon", "Y": "lat"}).transpose("T", "lat", "lon"),
        )

    def preprocess_test_data(self, X_test, y_test, X_train, y_train):
        X_test = _prepare_tyx(X_test, "X_test")
        y_test = _prepare_tyx(y_test, "y_test")
        X_train = _prepare_tyx(X_train, "X_train")
        y_train = _prepare_tyx(y_train, "y_train")
        x_ref = getattr(self, "_x_train_ref", X_train.mean(dim="T", skipna=True))
        y_ref = getattr(self, "_y_train_ref", y_train.mean(dim="T", skipna=True))
        X_test_prepared = X_test.fillna(x_ref)
        y_test_prepared = y_test.fillna(y_ref)
        x_mask = getattr(self, "_x_valid_mask", None)
        y_mask = getattr(self, "_y_valid_mask", None)
        if x_mask is not None:
            X_test_prepared = X_test_prepared.where(x_mask)
        if y_mask is not None:
            y_test_prepared = y_test_prepared.where(y_mask)
        return (
            X_test_prepared.rename({"X": "lon", "Y": "lat"}).transpose("T", "lat", "lon"),
            y_test_prepared.rename({"X": "lon", "Y": "lat"}).transpose("T", "lat", "lon"),
        )

    def compute_model(self, X_train, y_train, X_test, y_test):
        self.fit_cca(X_train, y_train)
        X_test_prepared, y_test_prepared = self.preprocess_test_data(
            X_test, y_test, X_train, y_train
        )
        y_pred = self.cca_model.predict(X_test_prepared)
        y_pred = self.cca_model.inverse_transform(
            self.cca_model.transform(X_test_prepared), y_pred
        )[1]
        y_pred = y_pred.assign_coords(T=y_test_prepared["T"])
        return y_pred.rename({"lon": "X", "lat": "Y"}).transpose("T", "Y", "X")
    def forecast(
        self, Predictant, clim_year_start, clim_year_end, Predictor,
        hindcast_det, Predictor_for_year, best_code_da=None,
        best_shape_da=None, best_loc_da=None, best_scale_da=None
    ):
        Predictant = _prepare_tyx(Predictant, "Predictant")
        Predictor = _prepare_tyx(Predictor, "Predictor")
        Predictor_for_year = _prepare_tyx(Predictor_for_year, "Predictor_for_year")
        Predictant_resid, hindcast_resid = _align_observation_and_hindcast(
            Predictant, hindcast_det
        )
        mask = _spatial_mask(Predictant)

        # Preserve WAS_CCA_op: linear detrending of X and standardized Y.
        Predictor_detrend, coeffs_X, meta_X = detrended_data(Predictor, dim="T")
        Predictant_st = standardize_timeseries(
            Predictant, clim_year_start, clim_year_end
        )
        Predictant_st_detrend, coeffs_Y, meta_Y = detrended_data(
            Predictant_st, dim="T"
        )

        Predictor_for_year_filled = Predictor_for_year.fillna(
            Predictor.mean(dim="T", skipna=True)
        ).fillna(0.0)
        Predictor_for_year_detrended = Predictor_for_year_filled - apply_detrend_data(
            Predictor_for_year_filled, coeffs_X, meta_X
        )

        self.fit_cca(Predictor_detrend, Predictant_st_detrend)
        # Same upstream validity mask used at fit time: exclude masked cells as
        # NaN (never constant-zero fills) so xeofs.predict matches the fit footprint.
        X_test_prepared = (
            Predictor_for_year_detrended
            .fillna(self._x_train_ref)
            .where(self._x_valid_mask)
            .rename({"X": "lon", "Y": "lat"})
            .transpose("T", "lat", "lon")
        )
        y_pred = self.cca_model.predict(X_test_prepared)
        y_pred = self.cca_model.inverse_transform(
            self.cca_model.transform(X_test_prepared), y_pred
        )[1]
        result_detrended = y_pred.rename({"lon": "X", "lat": "Y"}).transpose(
            "T", "Y", "X"
        )
        # Evaluate the Y trend at the predictand target time, not the init month.
        result_detrended = result_detrended.assign_coords(
            T=xr.DataArray(_target_time(Predictant, Predictor_for_year), dims=["T"])
        )
        result_st = result_detrended + apply_detrend_data(
            result_detrended, coeffs_Y, meta_Y
        )
        forecast_det = reverse_standardize(
            result_st, Predictant, clim_year_start, clim_year_end
        ).transpose("T", "Y", "X")
        forecast_det = _conditional_nonnegative_clip(forecast_det, Predictant)

        clim = Predictant.sel(T=slice(str(clim_year_start), str(clim_year_end)))
        if clim.sizes.get("T", 0) < 3:
            raise ValueError("Not enough years in climatology period for terciles.")
        terciles = clim.quantile([1.0 / 3.0, 2.0 / 3.0], dim="T")
        T1_emp = terciles.isel(quantile=0, drop=True)
        T2_emp = terciles.isel(quantile=1, drop=True)
        error_samples = Predictant_resid - hindcast_resid
        error_variance = error_samples.var(dim="T", skipna=True)
        dof = max(int(clim.sizes["T"]) - 1, 2)
        forecast_prob = _compute_probability_field(
            self, forecast_det, error_samples, error_variance,
            T1_emp, T2_emp, dof, mask,
            best_code_da, best_shape_da, best_loc_da, best_scale_da,
        )
        return (forecast_det * mask).transpose("T", "Y", "X"), forecast_prob
     

    def plot_cca_results(self, X=None, Y=None, n_modes=None, clim_year_start=None, clim_year_end=None):
        """
        Plots the CCA modes and scores.

        Parameters:
        - X: Optional xarray DataArray for predictors. If provided, the model will be fitted using X and Y.
        - Y: Optional xarray DataArray for predictands.
        - n_modes: Number of modes to plot. If None, plots all modes.
        """
        if X is not None and Y is not None:
            mask = xr.where(~np.isnan(Y.isel(T=0)), 1, np.nan).drop_vars(['T']).squeeze().to_numpy()
            # mask.name = None
            

            X, coeffss, metas = detrended_data(X, dim="T")
        
            Y = standardize_timeseries(Y, clim_year_start, clim_year_end)
            Y, coeffs, meta = detrended_data(Y, dim="T")
        
            # Fit the model using the provided data
            self.fit_cca(X.fillna(0), Y.fillna(0))
        elif self.cca_model is None:
            raise ValueError("The CCA model has not been fitted yet. Provide X and Y data to fit the model.")

        # Get components (modes) and scores
        X_modes, Y_modes = self.cca_model.components()  # Spatial patterns
        X_scores, Y_scores = self.cca_model.scores()    # Temporal projections (canonical variates)

        # Get explained variances
        var_explained_X = self.cca_model.fraction_variance_X_explained_by_X()
        var_explained_Y = self.cca_model.fraction_variance_Y_explained_by_Y()
        var_explained_Y_by_X = self.cca_model.fraction_variance_Y_explained_by_X()

        # Determine number of modes to plot
        if n_modes is None:
            n_modes = self.n_modes

        # Mode indices start from 1 in xeofs
        mode_indices = range(1, n_modes + 1)

        # Create subplots
        fig, axes = plt.subplots(n_modes, 3, figsize=(15, 3 * n_modes))

        if n_modes == 1:
            axes = np.array([axes])

        for i, mode in enumerate(mode_indices):

            # First Column: Plot X_modes
            ax = axes[i, 0]
            X_mode = X_modes.sel(mode=mode)
            X_mode.plot(ax=ax, vmin=-1, vmax=1, cmap= "RdBu_r")
            var_X = var_explained_X.sel(mode=mode).values * 100
            ax.set_title(f'X Mode {mode} ({var_X:.2f}% variance explained)')

            # Second Column: Plot X_scores and Y_scores
            ax = axes[i, 1]
            X_score = X_scores.sel(mode=mode)
            Y_score = Y_scores.sel(mode=mode)
            var_Y_X = var_explained_Y_by_X.sel(mode=mode).values * 100
            ax.plot(X_score['T'].dt.year.values, X_score, label='X Score')
            ax.plot(Y_score['T'].dt.year.values, Y_score, label='Y Score')
            ax.axhline(0, linestyle='--', lw=0.8, label="") #### line Canonical Variate = 0
            ax.legend()
            ax.set_title(f'Scores for Mode {mode} ({var_Y_X:.2f}% variance Y explained by X)')
            ax.set_xlabel('Time')
            ax.set_ylabel('Canonical Variate')

            # Third Column: Plot Y_modes
            ax = axes[i, 2]
            Y_mode = (Y_modes.sel(mode=mode))*mask
            Y_mode.plot(ax=ax, vmin=None, vmax=None, cmap= "RdBu_r")
            var_Y = var_explained_Y.sel(mode=mode).values * 100
            ax.set_title(f'Y Mode {mode} ({var_Y:.2f}% variance explained)')
            
        plt.tight_layout()
        plt.show()
        
    @staticmethod
    @staticmethod
    def _ppf_terciles_from_code(dist_code, shape, loc, scale):
        """Return exact one-third and two-third thresholds from fitted parameters."""
        if not np.isfinite(dist_code):
            return np.nan, np.nan

        q1, q2 = 1.0 / 3.0, 2.0 / 3.0
        code = int(dist_code)
        try:
            if code == 1:
                return norm.ppf(q1, loc=loc, scale=scale), norm.ppf(q2, loc=loc, scale=scale)
            if code == 2:
                return lognorm.ppf(q1, s=shape, loc=loc, scale=scale), lognorm.ppf(q2, s=shape, loc=loc, scale=scale)
            if code == 3:
                return expon.ppf(q1, loc=loc, scale=scale), expon.ppf(q2, loc=loc, scale=scale)
            if code == 4:
                return gamma.ppf(q1, a=shape, loc=loc, scale=scale), gamma.ppf(q2, a=shape, loc=loc, scale=scale)
            if code == 5:
                return weibull_min.ppf(q1, c=shape, loc=loc, scale=scale), weibull_min.ppf(q2, c=shape, loc=loc, scale=scale)
            if code == 6:
                return t.ppf(q1, df=shape, loc=loc, scale=scale), t.ppf(q2, df=shape, loc=loc, scale=scale)
            if code == 7:
                return poisson.ppf(q1, mu=shape, loc=loc), poisson.ppf(q2, mu=shape, loc=loc)
            if code == 8:
                return nbinom.ppf(q1, n=shape, p=scale, loc=loc), nbinom.ppf(q2, n=shape, p=scale, loc=loc)
        except Exception:
            return np.nan, np.nan
        return np.nan, np.nan
        
    @staticmethod
    @staticmethod
    def weibull_shape_solver(k, M, V):
        """Compatibility wrapper for the historical public helper."""
        k = float(np.asarray(k).reshape(-1)[0])
        if k <= 0 or M <= 0 or V <= 0:
            return np.nan
        g1 = gamma_function(1.0 + 1.0 / k)
        g2 = gamma_function(1.0 + 2.0 / k)
        return (g2 / (g1 ** 2) - 1.0) - V / (M ** 2)

    @staticmethod
    @staticmethod
    def calculate_tercile_probabilities_bestfit(
        best_guess, error_variance, T1, T2, dist_code, dof
    ):
        """Compute PB/PN/PA for the requested predictive distribution."""
        best_guess = np.asarray(best_guess, dtype=float)
        error_variance = float(error_variance)
        n_time = best_guess.size
        out = np.full((3, n_time), np.nan, dtype=float)

        if (
            np.all(~np.isfinite(best_guess))
            or not np.isfinite(error_variance)
            or error_variance <= 0
            or not np.isfinite(T1)
            or not np.isfinite(T2)
            or not np.isfinite(dist_code)
        ):
            return out

        code = int(dist_code)
        try:
            if code == 1:
                scale = np.sqrt(error_variance)
                c1 = norm.cdf(T1, loc=best_guess, scale=scale)
                c2 = norm.cdf(T2, loc=best_guess, scale=scale)

            elif code == 2:
                valid = best_guess > 0
                sigma = np.full(n_time, np.nan)
                mu = np.full(n_time, np.nan)
                sigma[valid] = np.sqrt(
                    np.log1p(error_variance / (best_guess[valid] ** 2))
                )
                mu[valid] = np.log(best_guess[valid]) - 0.5 * sigma[valid] ** 2
                c1 = lognorm.cdf(T1, s=sigma, scale=np.exp(mu))
                c2 = lognorm.cdf(T2, s=sigma, scale=np.exp(mu))

            elif code == 3:
                scale = np.sqrt(error_variance)
                loc = best_guess - scale
                c1 = expon.cdf(T1, loc=loc, scale=scale)
                c2 = expon.cdf(T2, loc=loc, scale=scale)

            elif code == 4:
                valid = best_guess > 0
                alpha = np.full(n_time, np.nan)
                theta = np.full(n_time, np.nan)
                alpha[valid] = best_guess[valid] ** 2 / error_variance
                theta[valid] = error_variance / best_guess[valid]
                c1 = gamma.cdf(T1, a=alpha, scale=theta)
                c2 = gamma.cdf(T2, a=alpha, scale=theta)

            elif code == 5:
                c1 = np.full(n_time, np.nan)
                c2 = np.full(n_time, np.nan)
                for i, mean_value in enumerate(best_guess):
                    shape = _solve_weibull_shape(mean_value, error_variance)
                    if not np.isfinite(shape):
                        continue
                    scale = mean_value / gamma_function(1.0 + 1.0 / shape)
                    c1[i] = weibull_min.cdf(T1, c=shape, loc=0.0, scale=scale)
                    c2[i] = weibull_min.cdf(T2, c=shape, loc=0.0, scale=scale)

            elif code == 6:
                if dof <= 2:
                    return out
                scale = np.sqrt(error_variance * (dof - 2.0) / dof)
                c1 = t.cdf(T1, df=dof, loc=best_guess, scale=scale)
                c2 = t.cdf(T2, df=dof, loc=best_guess, scale=scale)

            elif code == 7:
                mu = np.where(best_guess >= 0, best_guess, np.nan)
                c1 = poisson.cdf(T1, mu=mu)
                c2 = poisson.cdf(T2, mu=mu)

            elif code == 8:
                valid = (best_guess > 0) & (error_variance > best_guess)
                p = np.where(valid, best_guess / error_variance, np.nan)
                n = np.where(
                    valid,
                    best_guess ** 2 / (error_variance - best_guess),
                    np.nan,
                )
                c1 = nbinom.cdf(T1, n=n, p=p)
                c2 = nbinom.cdf(T2, n=n, p=p)

            else:
                return out

            out[0, :] = c1
            out[1, :] = c2 - c1
            out[2, :] = 1.0 - c2

        except (ValueError, FloatingPointError, OverflowError, RuntimeError):
            return out

        out = np.clip(out, 0.0, 1.0)
        total = np.nansum(out, axis=0)
        valid_total = np.isfinite(total) & (total > 0)
        out[:, valid_total] /= total[valid_total]
        return out

    @staticmethod
    @staticmethod
    def calculate_tercile_probabilities_nonparametric(
        best_guess, error_samples, first_tercile, second_tercile
    ):
        """Non-parametric probabilities from the historical residual sample."""
        best_guess = np.asarray(best_guess, dtype=float)
        valid_errors = np.asarray(error_samples, dtype=float)
        valid_errors = valid_errors[np.isfinite(valid_errors)]
        out = np.full((3, best_guess.size), np.nan, dtype=float)
        if valid_errors.size == 0:
            return out

        for i, prediction in enumerate(best_guess):
            if not np.isfinite(prediction):
                continue
            sample = prediction + valid_errors
            pb = np.mean(sample < first_tercile)
            pn = np.mean((sample >= first_tercile) & (sample < second_tercile))
            out[0, i] = pb
            out[1, i] = pn
            out[2, i] = 1.0 - pb - pn
        return out


    def compute_prob(
        self,
        Predictant: xr.DataArray,
        clim_year_start,
        clim_year_end,
        hindcast_det: xr.DataArray,
        best_code_da: xr.DataArray = None,
        best_shape_da: xr.DataArray = None,
        best_loc_da: xr.DataArray = None,
        best_scale_da: xr.DataArray = None,
    ) -> xr.DataArray:
        """Compute hindcast tercile probabilities with guaranteed output order."""
        Predictant, hindcast_det = _align_observation_and_hindcast(
            Predictant, hindcast_det
        )
        mask = _spatial_mask(Predictant)
        clim = Predictant.sel(T=slice(str(clim_year_start), str(clim_year_end)))
        if clim.sizes.get("T", 0) < 3:
            raise ValueError("Not enough years in climatology period for terciles.")

        terciles = clim.quantile([1.0 / 3.0, 2.0 / 3.0], dim="T")
        T1_emp = terciles.isel(quantile=0, drop=True)
        T2_emp = terciles.isel(quantile=1, drop=True)
        error_samples = Predictant - hindcast_det
        error_variance = error_samples.var(dim="T", skipna=True)
        dof = max(int(clim.sizes["T"]) - 1, 2)

        return _compute_probability_field(
            self,
            deterministic=hindcast_det,
            error_samples=error_samples,
            error_variance=error_variance,
            T1_emp=T1_emp,
            T2_emp=T2_emp,
            dof=dof,
            mask=mask,
            best_code_da=best_code_da,
            best_shape_da=best_shape_da,
            best_loc_da=best_loc_da,
            best_scale_da=best_scale_da,
        )


class WAS_CCA_strict:
    def __init__(self, n_modes=4, n_pca_modes=8, standardize=False, use_coslat=True, use_pca=True, dist_method="bestfit"):
        """
        Initialize parameters. The CCA model instance is created in fit_cca to ensure safe CV.
        """
        self.n_modes = n_modes
        self.n_pca_modes = n_pca_modes
        self.standardize = standardize
        self.use_coslat = use_coslat
        self.use_pca = use_pca
        self.dist_method = dist_method
        
        self.cca_model = None
        self.cca = None

    @staticmethod
    def _safe_drop_vars(da, names):
        """Drop variables if they exist; compatible with older xarray versions."""
        existing = [name for name in names if name in da.coords or name in da.variables]
        if existing:
            return da.drop_vars(existing)
        return da
    
    
    @staticmethod
    def _spatial_mask(da):
        """
        Return a Y/X mask preserving xarray coordinates.
        Do not convert to NumPy, because xarray broadcasting is safer.
        """
        mask = xr.where(~np.isnan(da.isel(T=0)), 1.0, np.nan)
        mask = WAS_CCA_strict._safe_drop_vars(mask, ["T"])
        return mask.squeeze()
    
    
    @staticmethod
    def _rename_to_latlon(da):
        """
        Convert WAS dimensions Y/X to xeofs dimensions lat/lon.
        """
        out = da
        rename = {}
        if "Y" in out.dims:
            rename["Y"] = "lat"
        if "X" in out.dims:
            rename["X"] = "lon"
        if rename:
            out = out.rename(rename)
    
        expected = [d for d in ("T", "lat", "lon") if d in out.dims]
        other = [d for d in out.dims if d not in expected]
        return out.transpose(*(expected + other))
    
    
    @staticmethod
    def _rename_to_YX(da):
        """
        Convert xeofs dimensions lat/lon back to WAS dimensions Y/X.
        """
        rename = {}
        if "lat" in da.dims:
            rename["lat"] = "Y"
        if "lon" in da.dims:
            rename["lon"] = "X"
        if rename:
            da = da.rename(rename)
    
        expected = [d for d in ("T", "Y", "X") if d in da.dims]
        other = [d for d in da.dims if d not in expected]
        return da.transpose(*(expected + other))
    
    
    @staticmethod
    @staticmethod
    def _fill_spatial_gaps(da, ref=None):
        """Fill forecast gaps using the historical mean, then spatial neighbours."""
        out = da
        if ref is not None and "T" in ref.dims:
            out = out.fillna(ref.mean(dim="T", skipna=True))
        return _fill_spatial_gaps_safe(out).fillna(0.0)
    
    
    @staticmethod
    def _target_time_from_predictor(Predictant, Predictor_for_year):
        """
        Build the forecast target time.
    
        Example:
        Predictor_for_year T may be April 2026, but predictand target season may
        be represented by July. This function keeps the forecast year from the
        predictor and the target month from Predictant.
        """
        try:
            year = int(Predictor_for_year["T"].dt.year.values[0])
        except Exception:
            year = (
                Predictor_for_year.coords["T"]
                .values.astype("datetime64[Y]")
                .astype(int)[0] + 1970
            )
    
        target_month = (
            Predictant.isel(T=0)
            .coords["T"]
            .values.astype("datetime64[M]")
            .astype(int) % 12 + 1
        )
    
        return np.datetime64(f"{year}-{int(target_month):02d}-01")
    
    
    @staticmethod
    @staticmethod
    def _normalize_probabilities(prob):
        return _normalize_probabilities(prob)

    def fit_cca(self, X_train, y_train):
        X_train_final, y_train_final = self.preprocess_data(X_train, y_train)
        self.cca = _new_safe_cca(self, X_train_final, y_train_final)
        self.cca_model = self.cca.fit(X_train_final, y_train_final, dim="T")
        return self.cca_model
    
    
    def preprocess_data(self, X, Y):
        X = _prepare_tyx(X, "X_train")
        Y = _prepare_tyx(Y, "y_train")
        # Common upstream validity mask: impute partial gaps at otherwise-valid
        # cells with the training temporal mean, then EXCLUDE (leave NaN) cells
        # that are not fully observed or are temporally constant, so xeofs drops
        # those features instead of ingesting constant-zero fills. Mask/reference
        # are stored so CV and forecast inputs are masked identically.
        self._x_train_ref = X.mean(dim="T", skipna=True)
        self._y_train_ref = Y.mean(dim="T", skipna=True)
        X_imp = X.fillna(self._x_train_ref)
        Y_imp = Y.fillna(self._y_train_ref)
        self._x_valid_mask = _common_validity_mask(X_imp)
        self._y_valid_mask = _common_validity_mask(Y_imp)
        X_final = X_imp.where(self._x_valid_mask)
        Y_final = Y_imp.where(self._y_valid_mask)
        return (
            X_final.rename({"X": "lon", "Y": "lat"}).transpose("T", "lat", "lon"),
            Y_final.rename({"X": "lon", "Y": "lat"}).transpose("T", "lat", "lon"),
        )
    
    
    def preprocess_test_data(self, X_test, y_test=None, X_train=None, y_train=None):
        X_test = _prepare_tyx(X_test, "X_test")
        if X_train is None:
            raise ValueError("X_train is required to fill validation data safely.")
        X_train = _prepare_tyx(X_train, "X_train")
        x_ref = getattr(self, "_x_train_ref", X_train.mean(dim="T", skipna=True))
        X_test_prepared = X_test.fillna(x_ref)
        x_mask = getattr(self, "_x_valid_mask", None)
        if x_mask is not None:
            X_test_prepared = X_test_prepared.where(x_mask)
        X_test_prepared = self._rename_to_latlon(X_test_prepared)

        if y_test is None:
            return X_test_prepared, None
        if y_train is None:
            raise ValueError("y_train is required to fill validation data safely.")
        y_test = _prepare_tyx(y_test, "y_test")
        y_train = _prepare_tyx(y_train, "y_train")
        y_ref = getattr(self, "_y_train_ref", y_train.mean(dim="T", skipna=True))
        y_test_prepared = y_test.fillna(y_ref)
        y_mask = getattr(self, "_y_valid_mask", None)
        if y_mask is not None:
            y_test_prepared = y_test_prepared.where(y_mask)
        return X_test_prepared, self._rename_to_latlon(y_test_prepared)
    
    
    def compute_model(self, X_train, y_train, X_test, y_test=None):
        self.fit_cca(X_train, y_train)
        X_test_prepared, y_test_prepared = self.preprocess_test_data(
            X_test=X_test, y_test=y_test, X_train=X_train, y_train=y_train
        )
        y_pred = self.cca_model.predict(X_test_prepared)
        y_pred = self.cca_model.inverse_transform(
            self.cca_model.transform(X_test_prepared), y_pred
        )[1]
        if y_test_prepared is not None:
            y_pred = y_pred.assign_coords(T=y_test_prepared["T"])
        return self._rename_to_YX(y_pred).transpose("T", "Y", "X")

    def forecast(
        self, Predictant, clim_year_start, clim_year_end, Predictor,
        hindcast_det, Predictor_for_year, best_code_da=None,
        best_shape_da=None, best_loc_da=None, best_scale_da=None,
    ):
        Predictant = _prepare_tyx(Predictant, "Predictant")
        Predictor = _prepare_tyx(Predictor, "Predictor")
        Predictor_for_year = _prepare_tyx(Predictor_for_year, "Predictor_for_year")
        Predictant_resid, hindcast_resid = _align_observation_and_hindcast(
            Predictant, hindcast_det
        )
        mask = _spatial_mask(Predictant)

        # Preserve strict spatial filling and operational detrending.
        Predictor_filled = self._fill_spatial_gaps(Predictor)
        Predictor_detrend, coeffs_X, meta_X = detrended_data(
            Predictor_filled, dim="T"
        )
        Predictor_detrend = Predictor_detrend.fillna(0.0)

        Predictant_st = standardize_timeseries(
            Predictant, clim_year_start, clim_year_end
        )
        Predictant_st_detrend, coeffs_Y, meta_Y = detrended_data(
            Predictant_st, dim="T"
        )
        Predictant_st_detrend = Predictant_st_detrend.fillna(0.0)

        Predictor_for_year_filled = self._fill_spatial_gaps(
            Predictor_for_year, ref=Predictor_filled
        )
        Predictor_for_year_detrended = Predictor_for_year_filled - apply_detrend_data(
            Predictor_for_year_filled, coeffs_X, meta_X
        )
        self.fit_cca(Predictor_detrend, Predictant_st_detrend)
        # Same upstream validity mask used at fit time (exclude masked cells as NaN).
        Predictor_for_year_detrended = (
            Predictor_for_year_detrended
            .fillna(self._x_train_ref)
            .where(self._x_valid_mask)
        )
        X_forecast = self._rename_to_latlon(Predictor_for_year_detrended)
        y_pred = self.cca_model.predict(X_forecast)
        y_pred = self.cca_model.inverse_transform(
            self.cca_model.transform(X_forecast), y_pred
        )[1]
        result_detrended = self._rename_to_YX(y_pred).transpose("T", "Y", "X")
        result_detrended = result_detrended.assign_coords(
            T=xr.DataArray(_target_time(Predictant, Predictor_for_year), dims=["T"])
        )
        result_st = result_detrended + apply_detrend_data(
            result_detrended, coeffs_Y, meta_Y
        )
        forecast_det = reverse_standardize(
            result_st, Predictant, clim_year_start, clim_year_end
        ).transpose("T", "Y", "X")
        forecast_det = _conditional_nonnegative_clip(forecast_det, Predictant)

        clim = Predictant.sel(T=slice(str(clim_year_start), str(clim_year_end)))
        if clim.sizes.get("T", 0) < 3:
            raise ValueError("Not enough years in climatology period for terciles.")
        terciles = clim.quantile([1.0 / 3.0, 2.0 / 3.0], dim="T")
        T1_emp = terciles.isel(quantile=0, drop=True)
        T2_emp = terciles.isel(quantile=1, drop=True)
        error_samples = Predictant_resid - hindcast_resid
        error_variance = error_samples.var(dim="T", skipna=True)
        dof = max(int(clim.sizes["T"]) - 1, 2)
        forecast_prob = _compute_probability_field(
            self, forecast_det, error_samples, error_variance,
            T1_emp, T2_emp, dof, mask,
            best_code_da, best_shape_da, best_loc_da, best_scale_da,
        )
        return (forecast_det * mask).transpose("T", "Y", "X"), forecast_prob

    def plot_cca_results(
        self,
        X=None,
        Y=None,
        n_modes=None,
        clim_year_start=None,
        clim_year_end=None,
    ):
        """
        Plot CCA spatial modes and canonical scores.
    
        If X and Y are provided, the CCA model is refitted using the same
        transformation logic used for the CCA workflow:
          - predictor: linear detrending
          - predictand: standardization + linear detrending
    
        Parameters
        ----------
        X : xarray.DataArray, optional
            Predictor field with dimensions ('T', 'Y', 'X') or ('T', 'lat', 'lon').
        Y : xarray.DataArray, optional
            Predictand field with dimensions ('T', 'Y', 'X') or ('T', 'lat', 'lon').
        n_modes : int, optional
            Number of CCA modes to plot. If None, uses self.n_modes.
        clim_year_start, clim_year_end : int or str, optional
            Climatology period used to standardize Y.
        """
    
        mask = None
    
        # ------------------------------------------------------------------
        # 1. Optionally fit/refit the CCA model
        # ------------------------------------------------------------------
        if X is not None and Y is not None:
            # Keep mask as xarray, not NumPy
            mask = xr.where(~np.isnan(Y.isel(T=0)), 1.0, np.nan).squeeze()
    
            if "T" in mask.coords:
                mask = mask.drop_vars("T")
    
            # Rename mask to match xeofs output dimensions
            rename_mask = {}
            if "Y" in mask.dims:
                rename_mask["Y"] = "lat"
            if "X" in mask.dims:
                rename_mask["X"] = "lon"
            if rename_mask:
                mask = mask.rename(rename_mask)
    
            # Predictor transformation
            X_detrended, _, _ = detrended_data(X, dim="T")
            X_ready = X_detrended.fillna(0.0)
    
            # Predictand transformation
            Y_st = standardize_timeseries(
                Y,
                clim_year_start,
                clim_year_end,
            )
    
            # Avoid inf values where std = 0
            Y_st = Y_st.where(np.isfinite(Y_st))
    
            Y_st_detrended, _, _ = detrended_data(Y_st, dim="T")
            Y_ready = Y_st_detrended.fillna(0.0)
    
            # Fit model using transformed data
            self.fit_cca(X_ready, Y_ready)
    
        elif self.cca_model is None:
            raise ValueError(
                "The CCA model has not been fitted yet. Provide X and Y data to fit the model."
            )
    
        # ------------------------------------------------------------------
        # 2. Extract CCA outputs
        # ------------------------------------------------------------------
        X_modes, Y_modes = self.cca_model.components()
        X_scores, Y_scores = self.cca_model.scores()
    
        var_explained_X = self.cca_model.fraction_variance_X_explained_by_X()
        var_explained_Y = self.cca_model.fraction_variance_Y_explained_by_Y()
        var_explained_Y_by_X = self.cca_model.fraction_variance_Y_explained_by_X()
    
        # ------------------------------------------------------------------
        # 3. Determine number of modes safely
        # ------------------------------------------------------------------
        available_modes = len(X_modes["mode"])
    
        if n_modes is None:
            n_modes = min(self.n_modes, available_modes)
        else:
            n_modes = min(int(n_modes), available_modes)
    
        mode_indices = X_modes["mode"].values[:n_modes]
    
        # ------------------------------------------------------------------
        # 4. Create plot
        # ------------------------------------------------------------------
        fig, axes = plt.subplots(
            n_modes,
            3,
            figsize=(15, 3.5 * n_modes),
            squeeze=False,
        )
    
        for i, mode in enumerate(mode_indices):
            # --------------------------------------------------------------
            # Column 1: Predictor spatial mode
            # --------------------------------------------------------------
            ax = axes[i, 0]
    
            X_mode = X_modes.sel(mode=mode)
            X_mode.plot(ax=ax, cmap="RdBu_r")
    
            var_X = float(var_explained_X.sel(mode=mode).values) * 100.0
            ax.set_title(f"X Mode {mode} ({var_X:.2f}% X variance)")
    
            # --------------------------------------------------------------
            # Column 2: Canonical scores
            # --------------------------------------------------------------
            ax = axes[i, 1]
    
            X_score = X_scores.sel(mode=mode)
            Y_score = Y_scores.sel(mode=mode)
    
            if np.issubdtype(X_score["T"].dtype, np.datetime64):
                time_axis = X_score["T"].dt.year.values
                xlabel = "Year"
            else:
                time_axis = X_score["T"].values
                xlabel = "Time"
    
            var_Y_by_X = float(var_explained_Y_by_X.sel(mode=mode).values) * 100.0
    
            ax.plot(time_axis, X_score.values, label="X score")
            ax.plot(time_axis, Y_score.values, label="Y score")
            ax.axhline(0.0, linestyle="--", lw=0.8)
    
            ax.legend()
            ax.set_title(f"Scores Mode {mode} ({var_Y_by_X:.2f}% Y explained by X)")
            ax.set_xlabel(xlabel)
            ax.set_ylabel("Canonical variate")
    
            # --------------------------------------------------------------
            # Column 3: Predictand spatial mode
            # --------------------------------------------------------------
            ax = axes[i, 2]
    
            Y_mode = Y_modes.sel(mode=mode)
    
            if mask is not None:
                Y_mode = Y_mode * mask
    
            Y_mode.plot(ax=ax, cmap="RdBu_r")
    
            var_Y = float(var_explained_Y.sel(mode=mode).values) * 100.0
            ax.set_title(f"Y Mode {mode} ({var_Y:.2f}% Y variance)")
    
        plt.tight_layout()
        plt.show()
        
    @staticmethod
    @staticmethod
    def _ppf_terciles_from_code(dist_code, shape, loc, scale):
        """Return exact one-third and two-third thresholds from fitted parameters."""
        if not np.isfinite(dist_code):
            return np.nan, np.nan

        q1, q2 = 1.0 / 3.0, 2.0 / 3.0
        code = int(dist_code)
        try:
            if code == 1:
                return norm.ppf(q1, loc=loc, scale=scale), norm.ppf(q2, loc=loc, scale=scale)
            if code == 2:
                return lognorm.ppf(q1, s=shape, loc=loc, scale=scale), lognorm.ppf(q2, s=shape, loc=loc, scale=scale)
            if code == 3:
                return expon.ppf(q1, loc=loc, scale=scale), expon.ppf(q2, loc=loc, scale=scale)
            if code == 4:
                return gamma.ppf(q1, a=shape, loc=loc, scale=scale), gamma.ppf(q2, a=shape, loc=loc, scale=scale)
            if code == 5:
                return weibull_min.ppf(q1, c=shape, loc=loc, scale=scale), weibull_min.ppf(q2, c=shape, loc=loc, scale=scale)
            if code == 6:
                return t.ppf(q1, df=shape, loc=loc, scale=scale), t.ppf(q2, df=shape, loc=loc, scale=scale)
            if code == 7:
                return poisson.ppf(q1, mu=shape, loc=loc), poisson.ppf(q2, mu=shape, loc=loc)
            if code == 8:
                return nbinom.ppf(q1, n=shape, p=scale, loc=loc), nbinom.ppf(q2, n=shape, p=scale, loc=loc)
        except Exception:
            return np.nan, np.nan
        return np.nan, np.nan
        
    @staticmethod
    @staticmethod
    def weibull_shape_solver(k, M, V):
        """Compatibility wrapper for the historical public helper."""
        k = float(np.asarray(k).reshape(-1)[0])
        if k <= 0 or M <= 0 or V <= 0:
            return np.nan
        g1 = gamma_function(1.0 + 1.0 / k)
        g2 = gamma_function(1.0 + 2.0 / k)
        return (g2 / (g1 ** 2) - 1.0) - V / (M ** 2)

    @staticmethod
    @staticmethod
    def calculate_tercile_probabilities_bestfit(
        best_guess, error_variance, T1, T2, dist_code, dof
    ):
        """Compute PB/PN/PA for the requested predictive distribution."""
        best_guess = np.asarray(best_guess, dtype=float)
        error_variance = float(error_variance)
        n_time = best_guess.size
        out = np.full((3, n_time), np.nan, dtype=float)

        if (
            np.all(~np.isfinite(best_guess))
            or not np.isfinite(error_variance)
            or error_variance <= 0
            or not np.isfinite(T1)
            or not np.isfinite(T2)
            or not np.isfinite(dist_code)
        ):
            return out

        code = int(dist_code)
        try:
            if code == 1:
                scale = np.sqrt(error_variance)
                c1 = norm.cdf(T1, loc=best_guess, scale=scale)
                c2 = norm.cdf(T2, loc=best_guess, scale=scale)

            elif code == 2:
                valid = best_guess > 0
                sigma = np.full(n_time, np.nan)
                mu = np.full(n_time, np.nan)
                sigma[valid] = np.sqrt(
                    np.log1p(error_variance / (best_guess[valid] ** 2))
                )
                mu[valid] = np.log(best_guess[valid]) - 0.5 * sigma[valid] ** 2
                c1 = lognorm.cdf(T1, s=sigma, scale=np.exp(mu))
                c2 = lognorm.cdf(T2, s=sigma, scale=np.exp(mu))

            elif code == 3:
                scale = np.sqrt(error_variance)
                loc = best_guess - scale
                c1 = expon.cdf(T1, loc=loc, scale=scale)
                c2 = expon.cdf(T2, loc=loc, scale=scale)

            elif code == 4:
                valid = best_guess > 0
                alpha = np.full(n_time, np.nan)
                theta = np.full(n_time, np.nan)
                alpha[valid] = best_guess[valid] ** 2 / error_variance
                theta[valid] = error_variance / best_guess[valid]
                c1 = gamma.cdf(T1, a=alpha, scale=theta)
                c2 = gamma.cdf(T2, a=alpha, scale=theta)

            elif code == 5:
                c1 = np.full(n_time, np.nan)
                c2 = np.full(n_time, np.nan)
                for i, mean_value in enumerate(best_guess):
                    shape = _solve_weibull_shape(mean_value, error_variance)
                    if not np.isfinite(shape):
                        continue
                    scale = mean_value / gamma_function(1.0 + 1.0 / shape)
                    c1[i] = weibull_min.cdf(T1, c=shape, loc=0.0, scale=scale)
                    c2[i] = weibull_min.cdf(T2, c=shape, loc=0.0, scale=scale)

            elif code == 6:
                if dof <= 2:
                    return out
                scale = np.sqrt(error_variance * (dof - 2.0) / dof)
                c1 = t.cdf(T1, df=dof, loc=best_guess, scale=scale)
                c2 = t.cdf(T2, df=dof, loc=best_guess, scale=scale)

            elif code == 7:
                mu = np.where(best_guess >= 0, best_guess, np.nan)
                c1 = poisson.cdf(T1, mu=mu)
                c2 = poisson.cdf(T2, mu=mu)

            elif code == 8:
                valid = (best_guess > 0) & (error_variance > best_guess)
                p = np.where(valid, best_guess / error_variance, np.nan)
                n = np.where(
                    valid,
                    best_guess ** 2 / (error_variance - best_guess),
                    np.nan,
                )
                c1 = nbinom.cdf(T1, n=n, p=p)
                c2 = nbinom.cdf(T2, n=n, p=p)

            else:
                return out

            out[0, :] = c1
            out[1, :] = c2 - c1
            out[2, :] = 1.0 - c2

        except (ValueError, FloatingPointError, OverflowError, RuntimeError):
            return out

        out = np.clip(out, 0.0, 1.0)
        total = np.nansum(out, axis=0)
        valid_total = np.isfinite(total) & (total > 0)
        out[:, valid_total] /= total[valid_total]
        return out

    @staticmethod
    @staticmethod
    def calculate_tercile_probabilities_nonparametric(
        best_guess, error_samples, first_tercile, second_tercile
    ):
        """Non-parametric probabilities from the historical residual sample."""
        best_guess = np.asarray(best_guess, dtype=float)
        valid_errors = np.asarray(error_samples, dtype=float)
        valid_errors = valid_errors[np.isfinite(valid_errors)]
        out = np.full((3, best_guess.size), np.nan, dtype=float)
        if valid_errors.size == 0:
            return out

        for i, prediction in enumerate(best_guess):
            if not np.isfinite(prediction):
                continue
            sample = prediction + valid_errors
            pb = np.mean(sample < first_tercile)
            pn = np.mean((sample >= first_tercile) & (sample < second_tercile))
            out[0, i] = pb
            out[1, i] = pn
            out[2, i] = 1.0 - pb - pn
        return out


    def compute_prob(
        self,
        Predictant: xr.DataArray,
        clim_year_start,
        clim_year_end,
        hindcast_det: xr.DataArray,
        best_code_da: xr.DataArray = None,
        best_shape_da: xr.DataArray = None,
        best_loc_da: xr.DataArray = None,
        best_scale_da: xr.DataArray = None,
    ) -> xr.DataArray:
        """Compute hindcast tercile probabilities with guaranteed output order."""
        Predictant, hindcast_det = _align_observation_and_hindcast(
            Predictant, hindcast_det
        )
        mask = _spatial_mask(Predictant)
        clim = Predictant.sel(T=slice(str(clim_year_start), str(clim_year_end)))
        if clim.sizes.get("T", 0) < 3:
            raise ValueError("Not enough years in climatology period for terciles.")

        terciles = clim.quantile([1.0 / 3.0, 2.0 / 3.0], dim="T")
        T1_emp = terciles.isel(quantile=0, drop=True)
        T2_emp = terciles.isel(quantile=1, drop=True)
        error_samples = Predictant - hindcast_det
        error_variance = error_samples.var(dim="T", skipna=True)
        dof = max(int(clim.sizes["T"]) - 1, 2)

        return _compute_probability_field(
            self,
            deterministic=hindcast_det,
            error_samples=error_samples,
            error_variance=error_variance,
            T1_emp=T1_emp,
            T2_emp=T2_emp,
            dof=dof,
            mask=mask,
            best_code_da=best_code_da,
            best_shape_da=best_shape_da,
            best_loc_da=best_loc_da,
            best_scale_da=best_scale_da,
        )



# """Canonical Correlation Analysis (CCA) for seasonal forecasting.

# Two implementations are provided:

# ``WAS_CCA``
#     Legacy CCA built on xeofs EOF analysis and numpy CCA.  Suitable for
#     quick exploratory use.

# ``WAS_CCA_base``
#     Leakage-free CCA that fits all preprocessing (NaN fill, standardization,
#     EOF decomposition) strictly on the training fold and applies the fitted
#     transforms to the test fold.  Recommended for cross-validation.

# Both classes expose ``compute_model``, ``compute_prob``, ``forecast``, and
# ``plot_cca_results`` methods.
# """
# from scipy import stats
# from scipy import signal as sig
# import numpy as np
# import xarray as xr
# from scipy.stats import gamma, norm, lognorm, expon, weibull_min, t, poisson, nbinom
# from scipy.optimize import fsolve
# from scipy.special import gamma as gamma_function
# import matplotlib.pyplot as plt
# import matplotlib.dates as mdates
# import xeofs as xe
# from wass2s.utils import * 


# class WAS_CCA:
#     def __init__(self, n_modes=4, n_pca_modes=8, standardize=False, use_coslat=True, use_pca=True, dist_method="nonparam"):
#         """
#         Initialize the WAS_CCA class with specified parameters.

#         Parameters:
#         - n_modes: Number of canonical modes to compute.
#         - n_pca_modes: Number of PCA modes to use before CCA.
#         - standardize: Whether to standardize the data. Keep it False in our case data already standardize
#         - use_coslat: Whether to use cosine latitude weighting.
#         - use_pca: Whether to perform PCA before CCA.
#         - detrend: Whether to apply detrending to the data. Extended EOF to detrend
#         """
        
#         self.n_modes = n_modes
#         self.n_pca_modes = n_pca_modes
#         self.standardize = standardize
#         self.use_coslat = use_coslat
#         self.use_pca = use_pca
#         self.dist_method = dist_method

#         self.cca = xe.cross.CCA(
#             n_modes=self.n_modes,
#             standardize=self.standardize,
#             use_coslat=self.use_coslat,
#             use_pca=self.use_pca,
#             n_pca_modes=self.n_pca_modes
#         )
#         self.cca_model = None
    
#     def fit_cca(self, X_train, y_train):
#         """
#         Fit the CCA model using the training data.

#         Parameters:
#         - X_train: xarray DataArray for predictor training data.
#         - y_train: xarray DataArray for predictand training data.
#         """
#         # Preprocess the data
#         X_train_final, y_train_final = self.preprocess_data(X_train, y_train)
#         # Fit the CCA model
#         self.cca_model = self.cca.fit(X_train_final, y_train_final, dim="T")
#     def preprocess_data(self, X, Y):
#         """
#         Preprocess the data by detrending, masking, and filling missing values.

#         Parameters:
#         - X: xarray DataArray for predictors.
#         - Y: xarray DataArray for predictands.

#         Returns:
#         - X_final: Preprocessed X data.
#         - Y_final: Preprocessed Y data.
#         """
#         # Apply detrending to both X and Y
#         X_processed = X #- self.detrend_data(X)
#         Y_processed = Y #- self.detrend_data(Y)

#         # Fill missing values with mean along 'T'
#         X_final = X_processed.fillna(X_processed.mean(dim="T", skipna=True))
#         Y_final = Y_processed.fillna(Y_processed.mean(dim="T", skipna=True))

#         # Rename dimensions and transpose
#         dims_rename = {"X": "lon", "Y": "lat"}
#         X_final = X_final.rename(dims_rename).transpose('T', 'lat', 'lon')
#         Y_final = Y_final.rename(dims_rename).transpose('T', 'lat', 'lon')

#         return X_final, Y_final

#     def preprocess_test_data(self, X_test, y_test, X_train, y_train):
#         """
#         Preprocess the test data.

#         Parameters:
#         - X_test: xarray DataArray for predictor testing data.
#         - y_test: xarray DataArray for predictand testing data.
#         - X_train: xarray DataArray for predictor training data.
#         - y_train: xarray DataArray for predictand training data.

#         Returns:
#         - X_test_prepared: Preprocessed X test data.
#         - y_test_prepared: Preprocessed Y test data.
#         """
#         # Apply detrending and masking
#         X_test_processed = X_test #- self.detrend_data(X_train).mean(dim="T", skipna=True)
#         y_test_processed = y_test 
        
#         # Fill missing values with mean from training data along 'T'
#         X_test_prepared = X_test_processed.fillna(X_train.mean(dim="T", skipna=True))        
#         y_test_prepared = y_test_processed.fillna(y_train.mean(dim="T", skipna=True))

#         # Rename dimensions and transpose
#         dims_rename = {"X": "lon", "Y": "lat"}
#         X_test_prepared = X_test_prepared.rename(dims_rename).transpose('T', 'lat', 'lon')
#         y_test_prepared = y_test_prepared.rename(dims_rename).transpose('T', 'lat', 'lon')

#         return X_test_prepared, y_test_prepared

#     def compute_model(self, X_train, y_train, X_test, y_test):
#         """
#         Compute the CCA model and generate hindcasts.

#         Parameters:
#         - X_train: xarray DataArray for predictor training data.
#         - y_train: xarray DataArray for predictand training data.
#         - X_test: xarray DataArray for predictor testing data.
#         - y_test: xarray DataArray for predictand testing data.

#         Returns:
#         - hindcast: xarray DataArray containing predictions and errors.
#         """
        
        
#         # Fit the CCA model
        
#         self.fit_cca(X_train, y_train)
            
#         # Prepare test data
#         X_test_prepared, y_test_prepared = self.preprocess_test_data(X_test, y_test, X_train, y_train)

#         # Predict
#         y_pred = self.cca_model.predict(X_test_prepared)
#         y_pred = self.cca_model.inverse_transform(self.cca_model.transform(X_test_prepared), y_pred)[1] 

#         y_pred['T'] = y_test_prepared['T']
#         # Calculate error
#         # error = y_test_prepared - y_pred

#         # Combine prediction and error into a DataArray
#         # hindcast = xr.concat([error, y_pred], dim="output")
#         hindcast = y_pred.rename({"lon": "X", "lat": "Y"})
#         # hindcast = hindcast.assign_coords(output=['error', 'prediction'])

#         return hindcast

#     # --------------------------------------------------------------------------
#     #  Probability Calculation Methods
#     # --------------------------------------------------------------------------
#     # ------------------ Probability Calculation Methods ------------------

#     @staticmethod
#     def _ppf_terciles_from_code(dist_code, shape, loc, scale):
#         """
#         Return tercile thresholds (T1, T2) from best-fit distribution parameters.
    
#         dist_code:
#             1: norm
#             2: lognorm
#             3: expon
#             4: gamma
#             5: weibull_min
#             6: t
#             7: poisson
#             8: nbinom
#         """
#         if np.isnan(dist_code):
#             return np.nan, np.nan
    
#         code = int(dist_code)
#         try:
#             if code == 1:
#                 return (
#                     norm.ppf(0.3, loc=loc, scale=scale),
#                     norm.ppf(0.67, loc=loc, scale=scale),
#                 )
#             elif code == 2:
#                 return (
#                     lognorm.ppf(0.3, s=shape, loc=loc, scale=scale),
#                     lognorm.ppf(0.67, s=shape, loc=loc, scale=scale),
#                 )
#             elif code == 3:
#                 return (
#                     expon.ppf(0.3, loc=loc, scale=scale),
#                     expon.ppf(0.67, loc=loc, scale=scale),
#                 )
#             elif code == 4:
#                 return (
#                     gamma.ppf(0.3, a=shape, loc=loc, scale=scale),
#                     gamma.ppf(0.67, a=shape, loc=loc, scale=scale),
#                 )
#             elif code == 5:
#                 return (
#                     weibull_min.ppf(0.3, c=shape, loc=loc, scale=scale),
#                     weibull_min.ppf(0.67, c=shape, loc=loc, scale=scale),
#                 )
#             elif code == 6:
#                 # Note: Renamed 't_dist' to 't' for standard scipy.stats
#                 return (
#                     t.ppf(0.3, df=shape, loc=loc, scale=scale),
#                     t.ppf(0.67, df=shape, loc=loc, scale=scale),
#                 )
#             elif code == 7:
#                 # Poisson: poisson.ppf(q, mu, loc=0)
#                 # ASSUMPTION: 'mu' (mean) is passed as 'shape'
#                 #             'loc' is passed as 'loc'
#                 #             'scale' is unused
#                 return (
#                     poisson.ppf(0.3, mu=shape, loc=loc),
#                     poisson.ppf(0.67, mu=shape, loc=loc),
#                 )
#             elif code == 8:
#                 # Negative Binomial: nbinom.ppf(q, n, p, loc=0)
#                 # ASSUMPTION: 'n' (successes) is passed as 'shape'
#                 #             'p' (probability) is passed as 'scale'
#                 #             'loc' is passed as 'loc'
#                 return (
#                     nbinom.ppf(0.3, n=shape, p=scale, loc=loc),
#                     nbinom.ppf(0.67, n=shape, p=scale, loc=loc),
#                 )
#         except Exception:
#             return np.nan, np.nan
    
#         # Fallback if code is not 1-8
#         return np.nan, np.nan
        
#     @staticmethod
#     def weibull_shape_solver(k, M, V):
#         """
#         Function to find the root of the Weibull shape parameter 'k'.
#         We find 'k' such that the theoretical variance/mean^2 ratio
#         matches the observed V/M^2 ratio.
#         """
#         # Guard against invalid 'k' values during solving
#         if k <= 0:
#             return -np.inf
#         try:
#             g1 = gamma_function(1 + 1/k)
#             g2 = gamma_function(1 + 2/k)
            
#             # This is the V/M^2 ratio *implied by k*
#             implied_v_over_m_sq = (g2 / (g1**2)) - 1
            
#             # This is the *observed* ratio
#             observed_v_over_m_sq = V / (M**2)
            
#             # Return the difference (we want this to be 0)
#             return observed_v_over_m_sq - implied_v_over_m_sq
#         except ValueError:
#             return -np.inf # Handle math errors

#     @staticmethod
#     def calculate_tercile_probabilities_bestfit(best_guess, error_variance, T1, T2, dist_code, dof 
#     ):
#         """
#         Generic tercile probabilities using best-fit family per grid cell.

#         Inputs (per grid cell):
#         - best_guess : 1D array over T (hindcast_det or forecast_det)
#         - T1, T2     : scalar terciles from climatological best-fit distribution
#         - dist_code  : int, as in _ppf_terciles_from_code
#         - shape, loc, scale : scalars from climatology fit

#         Strategy:
#         - For each time step, build a predictive distribution of the same family:
#             * Use best_guess[t] to adjust mean / location;
#             * Keep shape parameters from climatology.
#         - Then compute probabilities:
#             P(B) = F(T1), P(N) = F(T2) - F(T1), P(A) = 1 - F(T2).
#         """
        
#         best_guess = np.asarray(best_guess, float)
#         error_variance = np.asarray(error_variance, dtype=float)
#         # T1 = np.asarray(T1, dtype=float)
#         # T2 = np.asarray(T2, dtype=float)
#         n_time = best_guess.size
#         out = np.full((3, n_time), np.nan, float)

#         if np.all(np.isnan(best_guess)) or np.isnan(dist_code) or np.isnan(T1) or np.isnan(T2) or np.isnan(error_variance):
#             return out

#         code = int(dist_code)

#         # Normal: loc = forecast; scale from clim
#         if code == 1:
#             error_std = np.sqrt(error_variance)
#             out[0, :] = norm.cdf(T1, loc=best_guess, scale=error_std)
#             out[1, :] = norm.cdf(T2, loc=best_guess, scale=error_std) - norm.cdf(T1, loc=best_guess, scale=error_std)
#             out[2, :] = 1 - norm.cdf(T2, loc=best_guess, scale=error_std)

#         # Lognormal: shape = sigma from clim; enforce mean = best_guess
#         elif code == 2:
#             sigma = np.sqrt(np.log(1 + error_variance / (best_guess**2)))
#             mu = np.log(best_guess) - sigma**2 / 2
#             out[0, :] = lognorm.cdf(T1, s=sigma, scale=np.exp(mu))
#             out[1, :] = lognorm.cdf(T2, s=sigma, scale=np.exp(mu)) - lognorm.cdf(T1, s=sigma, scale=np.exp(mu))
#             out[2, :] = 1 - lognorm.cdf(T2, s=sigma, scale=np.exp(mu))      


#         # Exponential: keep scale from clim; shift loc so mean = best_guess
#         elif code == 3:
#             scale = np.sqrt(error_variance)
#             scale = np.where(scale <= 0, np.nan, scale)
        
#             # Exponential mean = loc + scale.
#             # Use loc = best_guess - scale so the predictive mean is best_guess.
#             loc = best_guess - scale
        
#             c1 = expon.cdf(T1, loc=loc, scale=scale)
#             c2 = expon.cdf(T2, loc=loc, scale=scale)
        
#             out[0, :] = c1
#             out[1, :] = c2 - c1
#             out[2, :] = 1.0 - c2

#         # Gamma: use shape from clim; set scale so mean = best_guess
#         elif code == 4:
#             alpha = (best_guess ** 2) / error_variance
#             theta = error_variance / best_guess
#             c1 = gamma.cdf(T1, a=alpha, scale=theta)
#             c2 = gamma.cdf(T2, a=alpha, scale=theta)
#             out[0, :] = c1
#             out[1, :] = c2 - c1
#             out[2, :] = 1.0 - c2

#         elif code == 5: # Assuming 5 is for Weibull   
        
#             for i in range(n_time):
#                 # Get the scalar values for this specific element (e.g., grid cell)
#                 M = best_guess[i]
#                 print(M)
#                 V = error_variance
#                 print(V)
                
#                 # Handle cases with no variance to avoid division by zero
#                 if V <= 0 or M <= 0:
#                     out[0, i] = np.nan
#                     out[1, i] = np.nan
#                     out[2, i] = np.nan
#                     continue # Skip to the next element
        
#                 # --- 1. Numerically solve for shape 'k' ---
#                 # We need a reasonable starting guess. 2.0 is common (Rayleigh dist.)
#                 initial_guess = 2.0
                
#                 # fsolve finds the root of our helper function
#                 k = fsolve(weibull_shape_solver, initial_guess, args=(M, V))[0]
        
#                 # --- 2. Check for bad solution and calculate scale 'lambda' ---
#                 if k <= 0:
#                     # Solver failed
#                     out[0, i] = np.nan
#                     out[1, i] = np.nan
#                     out[2, i] = np.nan
#                     continue
                
#                 # With 'k' found, we can now algebraically find scale 'lambda'
#                 # In scipy.stats, scale is 'scale'
#                 lambda_scale = M / gamma_function(1 + 1/k)
        
#                 # --- 3. Calculate Probabilities ---
#                 # In scipy.stats, shape 'k' is 'c'
#                 # Use the T1 and T2 values for this specific element
                
#                 c1 = weibull_min.cdf(T1, c=k, loc=0, scale=lambda_scale)
#                 c2 = weibull_min.cdf(T2, c=k, loc=0, scale=lambda_scale)
        
#                 out[0, i] = c1
#                 out[1, i] = c2 - c1
#                 out[2, i] = 1.0 - c2

#         # Student-t: df from clim; scale from clim; loc = best_guess
#         elif code == 6:       
#             # Check if df is valid for variance calculation
#             if dof <= 2:
#                 # Cannot calculate scale, fill with NaNs
#                 out[0, :] = np.nan
#                 out[1, :] = np.nan
#                 out[2, :] = np.nan
#             else:
#                 # 1. Calculate t-distribution parameters
#                 # 'loc' (mean) is just the best_guess
#                 loc = best_guess
#                 # 'scale' is calculated from the variance and df
#                 # Variance = scale**2 * (df / (df - 2))
#                 scale = np.sqrt(error_variance * (dof - 2) / dof)
                
#                 # 2. Calculate probabilities
#                 c1 = t.cdf(T1, df=dof, loc=loc, scale=scale)
#                 c2 = t.cdf(T2, df=dof, loc=loc, scale=scale)

#                 out[0, :] = c1
#                 out[1, :] = c2 - c1
#                 out[2, :] = 1.0 - c2

#         elif code == 7: # Assuming 7 is for Poisson
            
#             # --- 1. Set the Poisson parameter 'mu' ---
#             # The 'mu' parameter is the mean.
            
#             # A warning is strongly recommended if error_variance is different from best_guess
#             if not np.allclose(best_guess, error_variance, atol=0.5):
#                 print("Warning: 'error_variance' is not equal to 'best_guess'.")
#                 print("Poisson model assumes mean=variance and is likely inappropriate.")
#                 print("Consider using Negative Binomial.")
            
#             mu = best_guess
        
#             # --- 2. Calculate Probabilities ---
#             # poisson.cdf(k, mu) calculates P(X <= k)
            
#             c1 = poisson.cdf(T1, mu=mu)
#             c2 = poisson.cdf(T2, mu=mu)
            
#             out[0, :] = c1
#             out[1, :] = c2 - c1
#             out[2, :] = 1.0 - c2

#         elif code == 8: # Assuming 8 is for Negative Binomial
            
#             # --- 1. Calculate Negative Binomial Parameters ---
#             # This model is ONLY valid for overdispersion (Variance > Mean).
#             # We will use np.where to set parameters to NaN if V <= M.
            
#             # p = Mean / Variance
#             p = np.where(error_variance > best_guess, 
#                          best_guess / error_variance, 
#                          np.nan)
            
#             # n = Mean^2 / (Variance - Mean)
#             n = np.where(error_variance > best_guess, 
#                          (best_guess**2) / (error_variance - best_guess), 
#                          np.nan)
            
#             # --- 2. Calculate Probabilities ---
#             # The nbinom.cdf function will propagate NaNs, correctly
#             # handling the cases where the model was invalid.
            
#             c1 = nbinom.cdf(T1, n=n, p=p)
#             c2 = nbinom.cdf(T2, n=n, p=p)
            
#             out[0, :] = c1
#             out[1, :] = c2 - c1
#             out[2, :] = 1.0 - c2
            
#         else:
#             raise ValueError(f"Invalid distribution")

#         return out

#     @staticmethod
#     def calculate_tercile_probabilities_nonparametric(best_guess, error_samples, first_tercile, second_tercile):
#         """Non-parametric method using historical error samples."""
#         n_time = len(best_guess)
#         pred_prob = np.full((3, n_time), np.nan, dtype=float)
#         for t in range(n_time):
#             if np.isnan(best_guess[t]):
#                 continue
#             dist = best_guess[t] + error_samples
#             dist = dist[np.isfinite(dist)]
#             if len(dist) == 0:
#                 continue
#             p_below = np.mean(dist < first_tercile)
#             p_between = np.mean((dist >= first_tercile) & (dist < second_tercile))
#             p_above = 1.0 - (p_below + p_between)
#             pred_prob[0, t] = p_below
#             pred_prob[1, t] = p_between
#             pred_prob[2, t] = p_above
#         return pred_prob


#     def compute_prob(
#         self,
#         Predictant: xr.DataArray,
#         clim_year_start,
#         clim_year_end,
#         hindcast_det: xr.DataArray,
#         best_code_da: xr.DataArray = None,
#         best_shape_da: xr.DataArray = None,
#         best_loc_da: xr.DataArray = None,
#         best_scale_da: xr.DataArray = None
#     ) -> xr.DataArray:
#         """
#         Compute tercile probabilities for deterministic hindcasts.

#         If dist_method == 'bestfit':
#             - Use cluster-based best-fit distributions to:
#                 * derive terciles analytically from (best_code_da, best_shape_da, best_loc_da, best_scale_da),
#                 * compute predictive probabilities using the same family.

#         Otherwise:
#             - Use empirical terciles from Predictant climatology and the selected
#               parametric / nonparametric method.

#         Parameters
#         ----------
#         Predictant : xarray.DataArray
#             Observed data (T, Y, X) or (T, Y, X, M).
#         clim_year_start, clim_year_end : int or str
#             Climatology period (inclusive) for thresholds.
#         hindcast_det : xarray.DataArray
#             Deterministic hindcast (T, Y, X).
#         best_code_da, best_shape_da, best_loc_da, best_scale_da : xarray.DataArray, optional
#             Output from WAS_TransformData.fit_best_distribution_grid, required for 'bestfit'.

#         Returns
#         -------
#         hindcast_prob : xarray.DataArray
#             Probabilities with dims (probability=['PB','PN','PA'], T, Y, X).
#         """
#         # Handle member dimension if present
#         if "M" in Predictant.dims:
#             Predictant = Predictant.isel(M=0).drop_vars("M").squeeze()

#         # Ensure dimension order
#         Predictant = Predictant.transpose("T", "Y", "X")

#         # Spatial mask
#         mask = xr.where(~np.isnan(Predictant.isel(T=0)), 1.0, np.nan)

#         # Climatology subset
#         clim = Predictant.sel(T=slice(str(clim_year_start), str(clim_year_end)))
#         if clim.sizes.get("T", 0) < 3:
#             raise ValueError("Not enough years in climatology period for terciles.")

#         # Error variance for predictive distributions
#         error_variance = (Predictant - hindcast_det).var(dim="T")
#         dof = max(int(clim.sizes["T"]) - 1, 2)

#         # Empirical terciles (used by non-bestfit methods)
#         terciles_emp = clim.quantile([0.3, 0.67], dim="T")
#         T1_emp = terciles_emp.isel(quantile=0).drop_vars("quantile")
#         T2_emp = terciles_emp.isel(quantile=1).drop_vars("quantile")
        

#         dm = self.dist_method

#         # ---------- BESTFIT: zone-wise optimal distributions ----------
#         if dm == "bestfit":
#             if any(v is None for v in (best_code_da, best_shape_da, best_loc_da, best_scale_da)):
#                 raise ValueError(
#                     "dist_method='bestfit' requires best_code_da, best_shape_da_da, best_loc_da, best_scale_da."
#                 )

#             # T1, T2 from best-fit distributions (per grid)
#             T1, T2 = xr.apply_ufunc(
#                 self._ppf_terciles_from_code,
#                 best_code_da,
#                 best_shape_da,
#                 best_loc_da,
#                 best_scale_da,
#                 input_core_dims=[(), (), (), ()],
#                 output_core_dims=[(), ()],
#                 vectorize=True,
#                 dask="parallelized",
#                 output_dtypes=[float, float],
#             )

#             # Predictive probabilities using same family
#             hindcast_prob = xr.apply_ufunc(
#                 self.calculate_tercile_probabilities_bestfit,
#                 hindcast_det,
#                 error_variance,
#                 T1,
#                 T2,
#                 best_code_da,
#                 input_core_dims=[("T",), (), (), (), ()],
#                 output_core_dims=[("probability", "T")],
#                 vectorize=True,
#                 kwargs={'dof': dof},
#                 dask="parallelized",
#                 output_dtypes=[float],
#                 dask_gufunc_kwargs={
#                     "output_sizes": {"probability": 3},
#                     "allow_rechunk": True,
#                 },
#             )

#         # ---------- Nonparametric ----------
#         elif dm == "nonparam":
#             error_samples = Predictant - hindcast_det
#             hindcast_prob = xr.apply_ufunc(
#                 self.calculate_tercile_probabilities_nonparametric,
#                 hindcast_det,
#                 error_samples,
#                 T1_emp,
#                 T2_emp,
#                 input_core_dims=[("T",), ("T",), (), ()],
#                 output_core_dims=[("probability", "T")],
#                 vectorize=True,
#                 dask="parallelized",
#                 output_dtypes=[float],
#                 dask_gufunc_kwargs={
#                     "output_sizes": {"probability": 3},
#                     "allow_rechunk": True,
#                 },
#             )

#         else:
#             raise ValueError(f"Invalid dist_method: {self.dist_method}")

#         hindcast_prob = hindcast_prob.assign_coords(
#             probability=("probability", ["PB", "PN", "PA"])
#         )
#         return (hindcast_prob * mask).transpose("probability", "T", "Y", "X")


#     def forecast(self, Predictant, clim_year_start, clim_year_end, Predictor, hindcast_det, Predictor_for_year, best_code_da=None, best_shape_da=None, best_loc_da=None, best_scale_da=None):
#         mask = xr.where(~np.isnan(Predictant.isel(T=0)), 1, np.nan).drop_vars(['T']).squeeze().to_numpy()
#         Predictor_ = (Predictor - extract_leading_eeof_component(Predictor).fillna(extract_leading_eeof_component(Predictor)[-3])).fillna(0)
#         Predictant_st = standardize_timeseries(Predictant, clim_year_start, clim_year_end)
#         Predictant_ = (Predictant_st - extract_leading_eeof_component(Predictant_st).fillna(extract_leading_eeof_component(Predictant_st)[-3])).fillna(0)
        
#         Predictor_for_year_ = ((((Predictor_for_year.fillna(Predictor.mean(dim="T", skipna=True))).ffill(dim="Y").bfill(dim="Y")).ffill(dim="X").bfill(dim="X")).fillna(0)).transpose('T', 'Y', 'X')

#         # last_trend_X = ((extract_leading_eeof_component(standardize_timeseries(Predictor, clim_year_start, clim_year_end))).isel(T=[-3]))
#         # last_trend_X = ((extract_leading_eeof_component(Predictor)).isel(T=[-3]))
#         # last_trend_X['T'] = Predictor_for_year_['T']
#         # Predictor_for_year__ = Predictor_for_year_.fillna(0)
#         # Predictor_for_year__ = (Predictor_for_year_ - last_trend_X).fillna(0)

#         Predictor_for_year__ = Predictor_for_year_

#         # Fit the CCA model
#         self.fit_cca(Predictor_, Predictant_)
            
#         # Prepare test data
#         X_test_prepared = Predictor_for_year__.rename({"X": "lon", "Y": "lat"}).transpose('T', 'lat', 'lon')

#         # Predict
#         y_pred = self.cca_model.predict(X_test_prepared)
#         y_pred = self.cca_model.inverse_transform(self.cca_model.transform(X_test_prepared), y_pred)[1] 
#         result_ = y_pred.rename({"lon": "X", "lat": "Y"})
        
#         # last_trend_Y = ((extract_leading_eeof_component(Predictant_st)).isel(T=[-3]))
#         # last_trend_Y['T'] = result_['T']
#         # result_ = (result_ + last_trend_Y)
       
#         result_ = reverse_standardize(result_, Predictant, clim_year_start, clim_year_end) 
        
#         # 2) Compute thresholds T1, T2 from climatology
#         index_start = Predictant.get_index("T").get_loc(str(clim_year_start)).start
#         index_end   = Predictant.get_index("T").get_loc(str(clim_year_end)).stop
#         rainfall_for_tercile = Predictant.isel(T=slice(index_start, index_end))
#         terciles = rainfall_for_tercile.quantile([0.3, 0.67], dim='T')
#         T1_emp = terciles.isel(quantile=0).drop_vars('quantile')
#         T2_emp = terciles.isel(quantile=1).drop_vars('quantile')
#         error_variance = (Predictant - hindcast_det).var(dim='T')

#         year = Predictor_for_year.coords['T'].values.astype('datetime64[Y]').astype(int)[0] + 1970  # Convert from epoch
#         T_value_1 = Predictant.isel(T=0).coords['T'].values  # Get the datetime64 value from da1
#         month_1 = T_value_1.astype('datetime64[M]').astype(int) % 12 + 1  # Extract month
#         new_T_value = np.datetime64(f"{year}-{month_1:02d}-{1:02d}")
#         forecast_expanded = result_.assign_coords(T=xr.DataArray([new_T_value], dims=["T"]))
#         forecast_expanded['T'] = forecast_expanded['T'].astype('datetime64[ns]')
        
        
#         dof = max(int(rainfall_for_tercile.sizes["T"]) - 1, 2)

#         dm = self.dist_method

#         # ---------- BESTFIT ----------
#         if dm == "bestfit":
#             if any(v is None for v in (best_code_da, best_shape_da, best_loc_da, best_scale_da)):
#                 raise ValueError(
#                     "dist_method='bestfit' requires best_code_da, best_shape_da, best_loc_da, best_scale_da."
#                 )
            
#             T1, T2 = xr.apply_ufunc(
#                 self._ppf_terciles_from_code,
#                 best_code_da,
#                 best_shape_da,
#                 best_loc_da,
#                 best_scale_da,
#                 input_core_dims=[(), (), (), ()],
#                 output_core_dims=[(), ()],
#                 vectorize=True,
#                 dask="parallelized",
#                 output_dtypes=[float, float],
#             )

#             forecast_prob = xr.apply_ufunc(
#                 self.calculate_tercile_probabilities_bestfit,
#                 forecast_expanded,
#                 error_variance,
#                 T1,
#                 T2,
#                 best_code_da,
#                 input_core_dims=[("T",), (), (), (), ()],
#                 output_core_dims=[("probability", "T")],
#                 vectorize=True,
#                 dask="parallelized",
#                 kwargs={"dof": dof},
#                 output_dtypes=[float],
#                 dask_gufunc_kwargs={
#                     "output_sizes": {"probability": 3},
#                     "allow_rechunk": True,
#                 },
#             )

#         # ---------- Nonparametric ----------
#         elif dm == "nonparam":
#             error_samples = Predictant - hindcast_det
#             forecast_prob = xr.apply_ufunc(
#                 self.calculate_tercile_probabilities_nonparametric,
#                 forecast_expanded,
#                 error_samples,
#                 T1_emp,
#                 T2_emp,
#                 input_core_dims=[("T",), ("T",), (), ()],
#                 output_core_dims=[("probability", "T")],
#                 vectorize=True,
#                 dask="parallelized",
#                 output_dtypes=[float],
#                 dask_gufunc_kwargs={
#                     "output_sizes": {"probability": 3},
#                     "allow_rechunk": True,
#                 },
#             )

#         else:
#             raise ValueError(f"Invalid dist_method: {self.dist_method}")
#         forecast_prob = forecast_prob.assign_coords(probability=('probability', ['PB', 'PN', 'PA']))
#         return forecast_expanded * mask, mask * forecast_prob.transpose('probability', 'T', 'Y', 'X')
        

#     def plot_cca_results(self, X=None, Y=None, n_modes=None, clim_year_start=None, clim_year_end=None):
#         """
#         Plots the CCA modes and scores.

#         Parameters:
#         - X: Optional xarray DataArray for predictors. If provided, the model will be fitted using X and Y.
#         - Y: Optional xarray DataArray for predictands.
#         - n_modes: Number of modes to plot. If None, plots all modes.
#         """
#         if X is not None and Y is not None:
#             mask = xr.where(~np.isnan(Y.isel(T=0)), 1, np.nan).drop_vars(['T']).squeeze().to_numpy()
#             # mask.name = None
            
#             X_ = standardize_timeseries(X, clim_year_start, clim_year_end) - extract_leading_eeof_component(standardize_timeseries(X, clim_year_start, clim_year_end)).fillna(extract_leading_eeof_component(standardize_timeseries(X, clim_year_start, clim_year_end))[-3])
#             Y_ = standardize_timeseries(Y, clim_year_start, clim_year_end) - extract_leading_eeof_component(standardize_timeseries(Y, clim_year_start, clim_year_end)).fillna(extract_leading_eeof_component(standardize_timeseries(Y, clim_year_start, clim_year_end))[-3])
            
#             # Fit the model using the provided data
#             self.fit_cca(X_.isel(T= slice(0,-2)).fillna(0), Y_.isel(T=slice(0,-2)).fillna(0))
#         elif self.cca_model is None:
#             raise ValueError("The CCA model has not been fitted yet. Provide X and Y data to fit the model.")

#         # Get components (modes) and scores
#         X_modes, Y_modes = self.cca_model.components()  # Spatial patterns
#         X_scores, Y_scores = self.cca_model.scores()    # Temporal projections (canonical variates)

#         # Get explained variances
#         var_explained_X = self.cca_model.fraction_variance_X_explained_by_X()
#         var_explained_Y = self.cca_model.fraction_variance_Y_explained_by_Y()
#         var_explained_Y_by_X = self.cca_model.fraction_variance_Y_explained_by_X()

#         # Determine number of modes to plot
#         if n_modes is None:
#             n_modes = self.n_modes

#         # Mode indices start from 1 in xeofs
#         mode_indices = range(1, n_modes + 1)

#         # Create subplots
#         fig, axes = plt.subplots(n_modes, 3, figsize=(15, 3 * n_modes))

#         if n_modes == 1:
#             axes = np.array([axes])

#         for i, mode in enumerate(mode_indices):

#             # First Column: Plot X_modes
#             ax = axes[i, 0]
#             X_mode = X_modes.sel(mode=mode)
#             X_mode.plot(ax=ax, vmin=-1, vmax=1, cmap= "RdBu_r")
#             var_X = var_explained_X.sel(mode=mode).values * 100
#             ax.set_title(f'X Mode {mode} ({var_X:.2f}% variance explained)')

#             # Second Column: Plot X_scores and Y_scores
#             ax = axes[i, 1]
#             X_score = X_scores.sel(mode=mode)
#             Y_score = Y_scores.sel(mode=mode)
#             var_Y_X = var_explained_Y_by_X.sel(mode=mode).values * 100
#             ax.plot(X_score['T'].dt.year.values, X_score, label='X Score')
#             ax.plot(Y_score['T'].dt.year.values, Y_score, label='Y Score')
#             ax.axhline(0, linestyle='--', lw=0.8, label="") #### line Canonical Variate = 0
#             ax.legend()
#             ax.set_title(f'Scores for Mode {mode} ({var_Y_X:.2f}% variance Y explained by X)')
#             ax.set_xlabel('Time')
#             ax.set_ylabel('Canonical Variate')

#             # Third Column: Plot Y_modes
#             ax = axes[i, 2]
#             Y_mode = (Y_modes.sel(mode=mode))*mask
#             Y_mode.plot(ax=ax, vmin=None, vmax=None, cmap= "RdBu_r")
#             var_Y = var_explained_Y.sel(mode=mode).values * 100
#             ax.set_title(f'Y Mode {mode} ({var_Y:.2f}% variance explained)')
            
#         plt.tight_layout()
#         plt.show()


# class WAS_CCA_base:
#     """
#     Canonical Correlation Analysis model based on xeofs.cross.CCA.

#     Important:
#     ----------
#     - No detrend no extract_leading_eeof_component is used here.
#     - The CCA basis is entirely handled by xeofs.cross.CCA.
#     - The predictand is standardized before CCA in forecast/cross-validation,
#       then transformed back to the original scale with reverse_standardize.
#     - Missing values are handled using training-period means to avoid leakage.
#     """

#     def __init__(
#         self,
#         n_modes=4,
#         n_pca_modes=8,
#         standardize=False,
#         use_coslat=True,
#         use_pca=True,
#         dist_method="nonparam",
#     ):
#         self.n_modes = n_modes
#         self.n_pca_modes = n_pca_modes
#         self.standardize = standardize
#         self.use_coslat = use_coslat
#         self.use_pca = use_pca
#         self.dist_method = dist_method

#         self.cca = None
#         self.cca_model = None

#     # ---------------------------------------------------------------------
#     # Internal utilities
#     # ---------------------------------------------------------------------

#     def _new_cca(self):
#         """
#         Create a fresh xeofs CCA object.

#         This is important during cross-validation because each fold must be
#         fitted independently.
#         """
#         return xe.cross.CCA(
#             n_modes=self.n_modes,
#             standardize=self.standardize,
#             use_coslat=self.use_coslat,
#             use_pca=self.use_pca,
#             n_pca_modes=self.n_pca_modes,
#         )

#     @staticmethod
#     def _drop_member_dim(da: xr.DataArray) -> xr.DataArray:
#         """
#         If the DataArray has an ensemble member dimension M, keep the first member.
#         """
#         if "M" in da.dims:
#             da = da.isel(M=0, drop=True)
#         return da

#     @staticmethod
#     def _ensure_tyx(da: xr.DataArray, name="data") -> xr.DataArray:
#         """
#         Ensure data has dimensions T, Y, X and is ordered as T, Y, X.
#         """
#         missing = [d for d in ("T", "Y", "X") if d not in da.dims]
#         if missing:
#             raise ValueError(f"{name} must contain dimensions T, Y, X. Missing: {missing}")

#         return da.transpose("T", "Y", "X")

#     @staticmethod
#     def _to_xeofs_dims(da: xr.DataArray) -> xr.DataArray:
#         """
#         Rename WAS dimensions to xeofs-compatible dimensions.
#         WAS:   T, Y, X
#         xeofs: T, lat, lon
#         """
#         rename = {}
#         if "Y" in da.dims:
#             rename["Y"] = "lat"
#         if "X" in da.dims:
#             rename["X"] = "lon"

#         da = da.rename(rename)

#         return da.transpose("T", "lat", "lon")

#     @staticmethod
#     def _from_xeofs_dims(da: xr.DataArray) -> xr.DataArray:
#         """
#         Rename xeofs dimensions back to WAS dimensions.
#         xeofs: T, lat, lon
#         WAS:   T, Y, X
#         """
#         rename = {}
#         if "lat" in da.dims:
#             rename["lat"] = "Y"
#         if "lon" in da.dims:
#             rename["lon"] = "X"

#         da = da.rename(rename)

#         return da.transpose("T", "Y", "X")

#     @staticmethod
#     def _fill_train_data(da: xr.DataArray) -> xr.DataArray:
#         """
#         Fill missing values in training data using the training mean along T.
#         Remaining NaNs are filled with 0.
#         """
#         mean_da = da.mean(dim="T", skipna=True)
#         return da.fillna(mean_da).fillna(0)

#     @staticmethod
#     def _fill_test_data(test_da: xr.DataArray, train_da: xr.DataArray) -> xr.DataArray:
#         """
#         Fill missing values in test/forecast data using the training mean along T.
#         Remaining NaNs are filled with 0.

#         This avoids using information from the test fold.
#         """
#         train_mean = train_da.mean(dim="T", skipna=True)
#         return test_da.fillna(train_mean).fillna(0)

#     @staticmethod
#     def _spatial_mask(Predictant: xr.DataArray) -> xr.DataArray:
#         """
#         Spatial mask based on valid predictand grid cells at the first time step.
#         """
#         return xr.where(np.isfinite(Predictant.isel(T=0)), 1.0, np.nan)

#     @staticmethod
#     def _normalize_probabilities(prob: xr.DataArray) -> xr.DataArray:
#         """
#         Clip probabilities to [0, 1] and normalize them so PB + PN + PA = 1.
#         """
#         prob = prob.clip(min=0, max=1)
#         total = prob.sum(dim="probability", skipna=True)
#         prob = prob / xr.where(total > 0, total, np.nan)
#         return prob

#     @staticmethod
#     def _make_forecast_time(Predictant: xr.DataArray, Predictor_for_year: xr.DataArray):
#         """
#         Construct forecast time using:
#         - the year of Predictor_for_year
#         - the month of the predictand seasonal target
#         """
#         forecast_year = (
#             Predictor_for_year.coords["T"]
#             .values.astype("datetime64[Y]")
#             .astype(int)[0]
#             + 1970
#         )

#         first_target_time = Predictant.isel(T=0).coords["T"].values
#         target_month = first_target_time.astype("datetime64[M]").astype(int) % 12 + 1

#         new_time = np.datetime64(f"{forecast_year}-{target_month:02d}-01")
#         return np.array([new_time], dtype="datetime64[ns]")

#     # ---------------------------------------------------------------------
#     # CCA preprocessing and model fitting
#     # ---------------------------------------------------------------------

#     def preprocess_data(self, X_train: xr.DataArray, y_train: xr.DataArray):
#         """
#         Prepare training data for xeofs CCA.

#         No detrending is applied.
#         Missing values are filled using training means.
#         """
#         X_train = self._drop_member_dim(X_train)
#         y_train = self._drop_member_dim(y_train)

#         X_train = self._ensure_tyx(X_train, name="X_train")
#         y_train = self._ensure_tyx(y_train, name="y_train")

#         X_train = self._fill_train_data(X_train)
#         y_train = self._fill_train_data(y_train)

#         X_train = self._to_xeofs_dims(X_train)
#         y_train = self._to_xeofs_dims(y_train)

#         return X_train, y_train

#     def preprocess_test_data(
#         self,
#         X_test: xr.DataArray,
#         y_test: xr.DataArray,
#         X_train: xr.DataArray,
#         y_train: xr.DataArray,
#     ):
#         """
#         Prepare test data for xeofs CCA.

#         Missing values in X_test and y_test are filled using training means.
#         y_test is mainly used to preserve the correct T coordinate.
#         """
#         X_test = self._drop_member_dim(X_test)
#         y_test = self._drop_member_dim(y_test)

#         X_train = self._drop_member_dim(X_train)
#         y_train = self._drop_member_dim(y_train)

#         X_test = self._ensure_tyx(X_test, name="X_test")
#         y_test = self._ensure_tyx(y_test, name="y_test")
#         X_train = self._ensure_tyx(X_train, name="X_train")
#         y_train = self._ensure_tyx(y_train, name="y_train")

#         X_test = self._fill_test_data(X_test, X_train)
#         y_test = self._fill_test_data(y_test, y_train)

#         X_test = self._to_xeofs_dims(X_test)
#         y_test = self._to_xeofs_dims(y_test)

#         return X_test, y_test

#     def preprocess_forecast_data(
#         self,
#         Predictor_for_year: xr.DataArray,
#         Predictor_train: xr.DataArray,
#     ):
#         """
#         Prepare real-time predictor data for forecast.

#         Missing values are filled using the historical predictor mean.
#         """
#         Predictor_for_year = self._drop_member_dim(Predictor_for_year)
#         Predictor_train = self._drop_member_dim(Predictor_train)

#         Predictor_for_year = self._ensure_tyx(Predictor_for_year, name="Predictor_for_year")
#         Predictor_train = self._ensure_tyx(Predictor_train, name="Predictor_train")

#         Predictor_for_year = self._fill_test_data(Predictor_for_year, Predictor_train)
#         Predictor_for_year = self._to_xeofs_dims(Predictor_for_year)

#         return Predictor_for_year

#     def fit_cca(self, X_train: xr.DataArray, y_train: xr.DataArray):
#         """
#         Fit xeofs CCA on training data.
#         """
#         X_train_final, y_train_final = self.preprocess_data(X_train, y_train)

#         self.cca = self._new_cca()
#         self.cca_model = self.cca.fit(X_train_final, y_train_final, dim="T")

#         return self

#     def compute_model(
#         self,
#         X_train: xr.DataArray,
#         y_train: xr.DataArray,
#         X_test: xr.DataArray,
#         y_test: xr.DataArray,
#     ) -> xr.DataArray:
#         """
#         Fit CCA on training data and predict the test fold.

#         Returns
#         -------
#         hindcast : xr.DataArray
#             Deterministic hindcast with dimensions T, Y, X.
#         """
#         self.fit_cca(X_train, y_train)

#         X_test_prepared, y_test_prepared = self.preprocess_test_data(
#             X_test=X_test,
#             y_test=y_test,
#             X_train=X_train,
#             y_train=y_train,
#         )

#         y_pred = self.cca_model.predict(X_test_prepared)

#         # Reconstruct prediction in original xeofs spatial space.
#         y_pred = self.cca_model.inverse_transform(
#             self.cca_model.transform(X_test_prepared),
#             y_pred,
#         )[1]

#         y_pred = y_pred.assign_coords(T=y_test_prepared["T"])

#         hindcast = self._from_xeofs_dims(y_pred)

#         return hindcast

#     # ---------------------------------------------------------------------
#     # Probability calculation utilities
#     # ---------------------------------------------------------------------

#     @staticmethod
#     def _ppf_terciles_from_code(dist_code, shape, loc, scale):
#         """
#         Return tercile thresholds T1 and T2 from distribution parameters.

#         dist_code:
#             1: normal
#             2: lognormal
#             3: exponential
#             4: gamma
#             5: Weibull
#             6: Student-t
#             7: Poisson
#             8: Negative Binomial
#         """
#         if np.isnan(dist_code):
#             return np.nan, np.nan

#         code = int(dist_code)

#         try:
#             if code == 1:
#                 return (
#                     norm.ppf(0.30, loc=loc, scale=scale),
#                     norm.ppf(0.67, loc=loc, scale=scale),
#                 )

#             if code == 2:
#                 return (
#                     lognorm.ppf(0.30, s=shape, loc=loc, scale=scale),
#                     lognorm.ppf(0.67, s=shape, loc=loc, scale=scale),
#                 )

#             if code == 3:
#                 return (
#                     expon.ppf(0.30, loc=loc, scale=scale),
#                     expon.ppf(0.67, loc=loc, scale=scale),
#                 )

#             if code == 4:
#                 return (
#                     gamma.ppf(0.30, a=shape, loc=loc, scale=scale),
#                     gamma.ppf(0.67, a=shape, loc=loc, scale=scale),
#                 )

#             if code == 5:
#                 return (
#                     weibull_min.ppf(0.30, c=shape, loc=loc, scale=scale),
#                     weibull_min.ppf(0.67, c=shape, loc=loc, scale=scale),
#                 )

#             if code == 6:
#                 return (
#                     t.ppf(0.30, df=shape, loc=loc, scale=scale),
#                     t.ppf(0.67, df=shape, loc=loc, scale=scale),
#                 )

#             if code == 7:
#                 return (
#                     poisson.ppf(0.30, mu=shape, loc=loc),
#                     poisson.ppf(0.67, mu=shape, loc=loc),
#                 )

#             if code == 8:
#                 return (
#                     nbinom.ppf(0.30, n=shape, p=scale, loc=loc),
#                     nbinom.ppf(0.67, n=shape, p=scale, loc=loc),
#                 )

#         except Exception:
#             return np.nan, np.nan

#         return np.nan, np.nan

#     @staticmethod
#     def weibull_shape_solver(k, M, V):
#         """
#         Solve Weibull shape parameter k from mean M and variance V.
#         """
#         if k <= 0:
#             return -np.inf

#         try:
#             g1 = gamma_function(1 + 1 / k)
#             g2 = gamma_function(1 + 2 / k)

#             implied_v_over_m2 = (g2 / (g1**2)) - 1
#             observed_v_over_m2 = V / (M**2)

#             return observed_v_over_m2 - implied_v_over_m2

#         except Exception:
#             return -np.inf

#     @staticmethod
#     def calculate_tercile_probabilities_bestfit(
#         best_guess,
#         error_variance,
#         T1,
#         T2,
#         dist_code,
#         dof,
#     ):
#         """
#         Compute tercile probabilities using a best-fit distribution family.

#         Returns
#         -------
#         out : np.ndarray
#             Shape: probability, T
#         """
#         best_guess = np.asarray(best_guess, dtype=float)
#         error_variance = float(error_variance)

#         n_time = best_guess.size
#         out = np.full((3, n_time), np.nan, dtype=float)

#         if (
#             np.all(np.isnan(best_guess))
#             or not np.isfinite(error_variance)
#             or error_variance <= 0
#             or not np.isfinite(T1)
#             or not np.isfinite(T2)
#             or np.isnan(dist_code)
#         ):
#             return out

#         code = int(dist_code)

#         try:
#             if code == 1:
#                 error_std = np.sqrt(error_variance)

#                 c1 = norm.cdf(T1, loc=best_guess, scale=error_std)
#                 c2 = norm.cdf(T2, loc=best_guess, scale=error_std)

#                 out[0, :] = c1
#                 out[1, :] = c2 - c1
#                 out[2, :] = 1.0 - c2

#             elif code == 2:
#                 valid = best_guess > 0

#                 sigma = np.full(n_time, np.nan)
#                 mu = np.full(n_time, np.nan)

#                 sigma[valid] = np.sqrt(
#                     np.log(1.0 + error_variance / (best_guess[valid] ** 2))
#                 )
#                 mu[valid] = np.log(best_guess[valid]) - 0.5 * sigma[valid] ** 2

#                 c1 = lognorm.cdf(T1, s=sigma, scale=np.exp(mu))
#                 c2 = lognorm.cdf(T2, s=sigma, scale=np.exp(mu))

#                 out[0, :] = c1
#                 out[1, :] = c2 - c1
#                 out[2, :] = 1.0 - c2

#             elif code == 3:
#                 scale = np.sqrt(error_variance)
#                 loc = best_guess - scale

#                 c1 = expon.cdf(T1, loc=loc, scale=scale)
#                 c2 = expon.cdf(T2, loc=loc, scale=scale)

#                 out[0, :] = c1
#                 out[1, :] = c2 - c1
#                 out[2, :] = 1.0 - c2

#             elif code == 4:
#                 valid = best_guess > 0

#                 alpha = np.full(n_time, np.nan)
#                 theta = np.full(n_time, np.nan)

#                 alpha[valid] = (best_guess[valid] ** 2) / error_variance
#                 theta[valid] = error_variance / best_guess[valid]

#                 c1 = gamma.cdf(T1, a=alpha, scale=theta)
#                 c2 = gamma.cdf(T2, a=alpha, scale=theta)

#                 out[0, :] = c1
#                 out[1, :] = c2 - c1
#                 out[2, :] = 1.0 - c2

#             elif code == 5:
#                 for i in range(n_time):
#                     M = best_guess[i]
#                     V = error_variance

#                     if not np.isfinite(M) or M <= 0 or V <= 0:
#                         continue

#                     k = fsolve(
#                         WAS_CCA.weibull_shape_solver,
#                         2.0,
#                         args=(M, V),
#                     )[0]

#                     if not np.isfinite(k) or k <= 0:
#                         continue

#                     lambda_scale = M / gamma_function(1 + 1 / k)

#                     c1 = weibull_min.cdf(T1, c=k, loc=0, scale=lambda_scale)
#                     c2 = weibull_min.cdf(T2, c=k, loc=0, scale=lambda_scale)

#                     out[0, i] = c1
#                     out[1, i] = c2 - c1
#                     out[2, i] = 1.0 - c2

#             elif code == 6:
#                 if dof <= 2:
#                     return out

#                 loc = best_guess
#                 scale = np.sqrt(error_variance * (dof - 2) / dof)

#                 c1 = t.cdf(T1, df=dof, loc=loc, scale=scale)
#                 c2 = t.cdf(T2, df=dof, loc=loc, scale=scale)

#                 out[0, :] = c1
#                 out[1, :] = c2 - c1
#                 out[2, :] = 1.0 - c2

#             elif code == 7:
#                 valid = best_guess >= 0
#                 mu = np.where(valid, best_guess, np.nan)

#                 c1 = poisson.cdf(T1, mu=mu)
#                 c2 = poisson.cdf(T2, mu=mu)

#                 out[0, :] = c1
#                 out[1, :] = c2 - c1
#                 out[2, :] = 1.0 - c2

#             elif code == 8:
#                 valid = error_variance > best_guess

#                 p = np.where(valid, best_guess / error_variance, np.nan)
#                 n = np.where(
#                     valid,
#                     best_guess**2 / (error_variance - best_guess),
#                     np.nan,
#                 )

#                 c1 = nbinom.cdf(T1, n=n, p=p)
#                 c2 = nbinom.cdf(T2, n=n, p=p)

#                 out[0, :] = c1
#                 out[1, :] = c2 - c1
#                 out[2, :] = 1.0 - c2

#             else:
#                 return out

#         except Exception:
#             return out

#         out = np.clip(out, 0, 1)

#         total = np.nansum(out, axis=0)
#         valid_total = np.isfinite(total) & (total > 0)

#         out[:, valid_total] = out[:, valid_total] / total[valid_total]

#         return out

#     @staticmethod
#     def calculate_tercile_probabilities_nonparametric(
#         best_guess,
#         error_samples,
#         first_tercile,
#         second_tercile,
#     ):
#         """
#         Non-parametric probability method using historical error samples.
#         """
#         best_guess = np.asarray(best_guess, dtype=float)
#         error_samples = np.asarray(error_samples, dtype=float)

#         n_time = best_guess.size
#         pred_prob = np.full((3, n_time), np.nan, dtype=float)

#         valid_errors = error_samples[np.isfinite(error_samples)]

#         if valid_errors.size == 0:
#             return pred_prob

#         for i in range(n_time):
#             if not np.isfinite(best_guess[i]):
#                 continue

#             predictive_distribution = best_guess[i] + valid_errors

#             p_below = np.mean(predictive_distribution < first_tercile)
#             p_normal = np.mean(
#                 (predictive_distribution >= first_tercile)
#                 & (predictive_distribution < second_tercile)
#             )
#             p_above = 1.0 - p_below - p_normal

#             pred_prob[0, i] = p_below
#             pred_prob[1, i] = p_normal
#             pred_prob[2, i] = p_above

#         return pred_prob

#     def _compute_tercile_probabilities(
#         self,
#         Predictant: xr.DataArray,
#         deterministic: xr.DataArray,
#         clim_year_start,
#         clim_year_end,
#         error_samples: xr.DataArray,
#         error_variance: xr.DataArray,
#         best_code_da: xr.DataArray = None,
#         best_shape_da: xr.DataArray = None,
#         best_loc_da: xr.DataArray = None,
#         best_scale_da: xr.DataArray = None,
#     ) -> xr.DataArray:
#         """
#         Shared probability engine for hindcast and forecast.
#         """
#         Predictant = self._drop_member_dim(Predictant)
#         Predictant = self._ensure_tyx(Predictant, name="Predictant")

#         deterministic = self._drop_member_dim(deterministic)
#         deterministic = self._ensure_tyx(deterministic, name="deterministic")

#         mask = self._spatial_mask(Predictant)

#         clim = Predictant.sel(T=slice(str(clim_year_start), str(clim_year_end)))

#         if clim.sizes.get("T", 0) < 3:
#             raise ValueError("Not enough years in climatology period for tercile computation.")

#         terciles_emp = clim.quantile([0.30, 0.67], dim="T")
#         T1_emp = terciles_emp.isel(quantile=0).drop_vars("quantile")
#         T2_emp = terciles_emp.isel(quantile=1).drop_vars("quantile")

#         dof = max(int(clim.sizes["T"]) - 1, 2)

#         if self.dist_method == "bestfit":
#             if any(
#                 v is None
#                 for v in (best_code_da, best_shape_da, best_loc_da, best_scale_da)
#             ):
#                 raise ValueError(
#                     "dist_method='bestfit' requires best_code_da, "
#                     "best_shape_da, best_loc_da and best_scale_da."
#                 )

#             T1, T2 = xr.apply_ufunc(
#                 self._ppf_terciles_from_code,
#                 best_code_da,
#                 best_shape_da,
#                 best_loc_da,
#                 best_scale_da,
#                 input_core_dims=[(), (), (), ()],
#                 output_core_dims=[(), ()],
#                 vectorize=True,
#                 dask="parallelized",
#                 output_dtypes=[float, float],
#             )

#             prob = xr.apply_ufunc(
#                 self.calculate_tercile_probabilities_bestfit,
#                 deterministic,
#                 error_variance,
#                 T1,
#                 T2,
#                 best_code_da,
#                 input_core_dims=[("T",), (), (), (), ()],
#                 output_core_dims=[("probability", "T")],
#                 vectorize=True,
#                 kwargs={"dof": dof},
#                 dask="parallelized",
#                 output_dtypes=[float],
#                 dask_gufunc_kwargs={
#                     "output_sizes": {"probability": 3},
#                     "allow_rechunk": True,
#                 },
#             )

#         elif self.dist_method == "nonparam":
#             prob = xr.apply_ufunc(
#                 self.calculate_tercile_probabilities_nonparametric,
#                 deterministic,
#                 error_samples,
#                 T1_emp,
#                 T2_emp,
#                 input_core_dims=[("T",), ("T",), (), ()],
#                 output_core_dims=[("probability", "T")],
#                 vectorize=True,
#                 dask="parallelized",
#                 output_dtypes=[float],
#                 dask_gufunc_kwargs={
#                     "output_sizes": {"probability": 3},
#                     "allow_rechunk": True,
#                 },
#             )

#         else:
#             raise ValueError(f"Invalid dist_method: {self.dist_method}")

#         prob = prob.assign_coords(
#             probability=("probability", ["PB", "PN", "PA"])
#         )

#         prob = prob.transpose("probability", "T", "Y", "X")
#         prob = self._normalize_probabilities(prob)

#         return prob * mask

#     def compute_prob(
#         self,
#         Predictant: xr.DataArray,
#         clim_year_start,
#         clim_year_end,
#         hindcast_det: xr.DataArray,
#         best_code_da: xr.DataArray = None,
#         best_shape_da: xr.DataArray = None,
#         best_loc_da: xr.DataArray = None,
#         best_scale_da: xr.DataArray = None,
#     ) -> xr.DataArray:
#         """
#         Compute tercile probabilities for deterministic hindcasts.
#         """
#         Predictant = self._drop_member_dim(Predictant)
#         Predictant = self._ensure_tyx(Predictant, name="Predictant")

#         hindcast_det = self._drop_member_dim(hindcast_det)
#         hindcast_det = self._ensure_tyx(hindcast_det, name="hindcast_det")

#         if hindcast_det.sizes["T"] == Predictant.sizes["T"]:
#             hindcast_det = hindcast_det.assign_coords(T=Predictant["T"])

#         error_samples = Predictant - hindcast_det
#         error_variance = error_samples.var(dim="T", skipna=True)

#         return self._compute_tercile_probabilities(
#             Predictant=Predictant,
#             deterministic=hindcast_det,
#             clim_year_start=clim_year_start,
#             clim_year_end=clim_year_end,
#             error_samples=error_samples,
#             error_variance=error_variance,
#             best_code_da=best_code_da,
#             best_shape_da=best_shape_da,
#             best_loc_da=best_loc_da,
#             best_scale_da=best_scale_da,
#         )

#     # ---------------------------------------------------------------------
#     # Real-time forecast
#     # ---------------------------------------------------------------------

#     def forecast(
#         self,
#         Predictant: xr.DataArray,
#         clim_year_start,
#         clim_year_end,
#         Predictor: xr.DataArray,
#         hindcast_det: xr.DataArray,
#         Predictor_for_year: xr.DataArray,
#         best_code_da: xr.DataArray = None,
#         best_shape_da: xr.DataArray = None,
#         best_loc_da: xr.DataArray = None,
#         best_scale_da: xr.DataArray = None,
#     ):
#         """
#         Generate deterministic and probabilistic forecast with xeofs CCA.

#         """
#         Predictant = self._drop_member_dim(Predictant)
#         Predictant = self._ensure_tyx(Predictant, name="Predictant")

#         Predictor = self._drop_member_dim(Predictor)
#         Predictor = self._ensure_tyx(Predictor, name="Predictor")

#         hindcast_det = self._drop_member_dim(hindcast_det)
#         hindcast_det = self._ensure_tyx(hindcast_det, name="hindcast_det")

#         mask = self._spatial_mask(Predictant)

#         # Target is standardized before CCA.
#         Predictant_st = standardize_timeseries(
#             Predictant,
#             clim_year_start,
#             clim_year_end,
#         )

#         # Fit xeofs CCA on historical predictor and standardized predictand.
#         self.fit_cca(Predictor, Predictant_st)

#         # Prepare operational predictor using historical predictor mean.
#         X_forecast_prepared = self.preprocess_forecast_data(
#             Predictor_for_year=Predictor_for_year,
#             Predictor_train=Predictor,
#         )

#         # Predict in standardized predictand space.
#         y_pred = self.cca_model.predict(X_forecast_prepared)

#         y_pred = self.cca_model.inverse_transform(
#             self.cca_model.transform(X_forecast_prepared),
#             y_pred,
#         )[1]

#         forecast_det_st = self._from_xeofs_dims(y_pred)

#         # Return to original predictand scale.
#         forecast_det = reverse_standardize(
#             forecast_det_st,
#             Predictant,
#             clim_year_start,
#             clim_year_end,
#         )

#         # Assign target forecast time.
#         forecast_time = self._make_forecast_time(Predictant, Predictor_for_year)
#         forecast_det = forecast_det.assign_coords(
#             T=xr.DataArray(forecast_time, dims=["T"])
#         )

#         # Residual distribution from hindcast.
#         if hindcast_det.sizes["T"] == Predictant.sizes["T"]:
#             hindcast_det = hindcast_det.assign_coords(T=Predictant["T"])

#         error_samples = Predictant - hindcast_det
#         error_variance = error_samples.var(dim="T", skipna=True)

#         forecast_prob = self._compute_tercile_probabilities(
#             Predictant=Predictant,
#             deterministic=forecast_det,
#             clim_year_start=clim_year_start,
#             clim_year_end=clim_year_end,
#             error_samples=error_samples,
#             error_variance=error_variance,
#             best_code_da=best_code_da,
#             best_shape_da=best_shape_da,
#             best_loc_da=best_loc_da,
#             best_scale_da=best_scale_da,
#         )

#         forecast_det = xr.where(forecast_det < 0, 0, forecast_det)

#         return forecast_det * mask, forecast_prob * mask

#     # ---------------------------------------------------------------------
#     # Plot CCA modes
#     # ---------------------------------------------------------------------

#     def plot_cca_results(
#         self,
#         X: xr.DataArray = None,
#         Y: xr.DataArray = None,
#         n_modes=None,
#         clim_year_start=None,
#         clim_year_end=None,
#     ):
#         """
#         Plot CCA spatial modes and canonical scores.
#         """

#         def _spatial_to_was(da):
#             """Rename xeofs lat/lon -> WAS Y/X for a spatial mode (no T axis)."""
#             rename = {}
#             if "lat" in da.dims:
#                 rename["lat"] = "Y"
#             if "lon" in da.dims:
#                 rename["lon"] = "X"
#             da = da.rename(rename)
#             order = [d for d in ("T", "Y", "X") if d in da.dims]
#             return da.transpose(*order)

#         if X is not None and Y is not None:
#             X = self._drop_member_dim(X)
#             Y = self._drop_member_dim(Y)

#             X = self._ensure_tyx(X, name="X")
#             Y = self._ensure_tyx(Y, name="Y")

#             mask = self._spatial_mask(Y)

#             if clim_year_start is not None and clim_year_end is not None:
#                 Y_fit = standardize_timeseries(Y, clim_year_start, clim_year_end)
#             else:
#                 Y_fit = Y

#             X_fit = X

#             self.fit_cca(X_fit, Y_fit)

#         elif self.cca_model is None:
#             raise ValueError(
#                 "The CCA model has not been fitted yet. "
#                 "Provide X and Y or call fit_cca first."
#             )
#         else:
#             mask = None

#         X_modes, Y_modes = self.cca_model.components()
#         X_scores, Y_scores = self.cca_model.scores()

#         var_explained_X = self.cca_model.fraction_variance_X_explained_by_X()
#         var_explained_Y = self.cca_model.fraction_variance_Y_explained_by_Y()
#         var_explained_Y_by_X = self.cca_model.fraction_variance_Y_explained_by_X()

#         if n_modes is None:
#             n_modes = self.n_modes

#         fig, axes = plt.subplots(
#             n_modes,
#             3,
#             figsize=(15, 3.5 * n_modes),
#             squeeze=False,
#         )

#         for i, mode in enumerate(range(1, n_modes + 1)):
#             ax = axes[i, 0]
#             X_mode = X_modes.sel(mode=mode)
#             X_mode.plot(ax=ax, cmap="RdBu_r")
#             var_X = float(var_explained_X.sel(mode=mode).values) * 100
#             ax.set_title(f"X Mode {mode} ({var_X:.2f}% variance)")

#             ax = axes[i, 1]
#             X_score = X_scores.sel(mode=mode)
#             Y_score = Y_scores.sel(mode=mode)
#             var_Y_by_X = float(var_explained_Y_by_X.sel(mode=mode).values) * 100

#             ax.plot(X_score["T"].dt.year.values, X_score, label="X score")
#             ax.plot(Y_score["T"].dt.year.values, Y_score, label="Y score")
#             ax.axhline(0, linestyle="--", linewidth=0.8)
#             ax.legend()
#             ax.set_title(f"Scores Mode {mode} ({var_Y_by_X:.2f}% Y by X)")
#             ax.set_xlabel("Time")
#             ax.set_ylabel("Canonical variate")

#             ax = axes[i, 2]
#             Y_mode = _spatial_to_was(Y_modes.sel(mode=mode))
#             if mask is not None:
#                 Y_mode = Y_mode * mask
#             Y_mode.plot(ax=ax, cmap="RdBu_r")
#             var_Y = float(var_explained_Y.sel(mode=mode).values) * 100
#             ax.set_title(f"Y Mode {mode} ({var_Y:.2f}% variance)")

#         plt.tight_layout()
#         plt.show()


# class WAS_CCA_op:
#     def __init__(self, n_modes=4, n_pca_modes=8, standardize=False, use_coslat=True, use_pca=True, dist_method="bestfit"):
#         """
#         Initialize parameters. The CCA model instance is created in fit_cca to ensure safe CV.
#         """
#         self.n_modes = n_modes
#         self.n_pca_modes = n_pca_modes
#         self.standardize = standardize
#         self.use_coslat = use_coslat
#         self.use_pca = use_pca
#         self.dist_method = dist_method
        
#         self.cca_model = None
#         self.cca = None

#     def fit_cca(self, X_train, y_train):
#         """
#         Fit the CCA model. Creates a fresh xeofs instance every time.
#         """
#         self.cca = xe.cross.CCA(
#             n_modes=self.n_modes,
#             standardize=self.standardize,
#             use_coslat=self.use_coslat,
#             use_pca=self.use_pca,
#             n_pca_modes=self.n_pca_modes
#         )

#         X_train_final, y_train_final = self.preprocess_data(X_train, y_train)
#         self.cca_model = self.cca.fit(X_train_final, y_train_final, dim="T")

#     def preprocess_data(self, X, Y):
#         X_final = X.fillna(X.mean(dim="T", skipna=True))
#         Y_final = Y.fillna(Y.mean(dim="T", skipna=True))

#         dims_rename = {"X": "lon", "Y": "lat"}
#         if "X" in X_final.dims: X_final = X_final.rename(dims_rename)
#         if "Y" in X_final.dims: X_final = X_final.transpose('T', 'lat', 'lon')
        
#         if "X" in Y_final.dims: Y_final = Y_final.rename(dims_rename)
#         if "Y" in Y_final.dims: Y_final = Y_final.transpose('T', 'lat', 'lon')

#         return X_final, Y_final

#     def preprocess_test_data(self, X_test, y_test, X_train, y_train):
#         X_test_prepared = X_test.fillna(X_train.mean(dim="T", skipna=True))        
#         y_test_prepared = y_test.fillna(y_train.mean(dim="T", skipna=True))

#         dims_rename = {"X": "lon", "Y": "lat"}
#         if "X" in X_test_prepared.dims: X_test_prepared = X_test_prepared.rename(dims_rename)
#         if "Y" in X_test_prepared.dims: X_test_prepared = X_test_prepared.transpose('T', 'lat', 'lon')
        
#         if "X" in y_test_prepared.dims: y_test_prepared = y_test_prepared.rename(dims_rename)
#         if "Y" in y_test_prepared.dims: y_test_prepared = y_test_prepared.transpose('T', 'lat', 'lon')

#         return X_test_prepared, y_test_prepared

#     def compute_model(self, X_train, y_train, X_test, y_test):
#         self.fit_cca(X_train, y_train)
#         X_test_prepared, y_test_prepared = self.preprocess_test_data(X_test, y_test, X_train, y_train)

#         y_pred = self.cca_model.predict(X_test_prepared)
#         y_pred_phys = self.cca_model.inverse_transform(self.cca_model.transform(X_test_prepared), y_pred)[1] 

#         y_pred_phys['T'] = y_test_prepared['T']
#         hindcast = y_pred_phys.rename({"lon": "X", "lat": "Y"})

#         return hindcast
#     def forecast(self, Predictant, clim_year_start, clim_year_end, Predictor, hindcast_det, Predictor_for_year, best_code_da=None, best_shape_da=None, best_loc_da=None, best_scale_da=None):
#         mask = xr.where(~np.isnan(Predictant.isel(T=0)), 1, np.nan).drop_vars(['T']).squeeze().to_numpy()
#         # Predictor_ = (Predictor - extract_leading_eeof_component(Predictor).fillna(extract_leading_eeof_component(Predictor)[-3])).fillna(0)
#         # Predictant_st = standardize_timeseries(Predictant, clim_year_start, clim_year_end)
#         # Predictant_ = (Predictant_st - extract_leading_eeof_component(Predictant_st).fillna(extract_leading_eeof_component(Predictant_st)[-3])).fillna(0)
        
#         # Predictor_for_year_ = ((((Predictor_for_year.fillna(Predictor.mean(dim="T", skipna=True))).ffill(dim="Y").bfill(dim="Y")).ffill(dim="X").bfill(dim="X")).fillna(0)).transpose('T', 'Y', 'X

#         Predictor_detrend, coeffss, metas = detrended_data(Predictor, dim="T")
#         # Predictor_detrend = Predictor_detrend.fillna(Predictor_detrend.mean(dim="T", skipna=True))
        
#         Predictant_st = standardize_timeseries(Predictant, clim_year_start, clim_year_end)
#         Predictant_st_detrend, coeffs, meta = detrended_data(Predictant_st, dim="T")
#         # Predictant_st_detrend = Predictant_st_detrend.fillna(Predictant_st_detrend.mean(dim="T", skipna=True))

#         # last_trend_X = ((extract_leading_eeof_component(standardize_timeseries(Predictor, clim_year_start, clim_year_end))).isel(T=[-3]))
#         # last_trend_X = ((extract_leading_eeof_component(Predictor)).isel(T=[-3]))
#         # last_trend_X['T'] = Predictor_for_year_['T']
#         # Predictor_for_year__ = Predictor_for_year_.fillna(0)
#         # Predictor_for_year__ = (Predictor_for_year_ - last_trend_X).fillna(0)

#         # Predictor_for_year__ = Predictor_for_year_

#         Predictor_for_year__ = Predictor_for_year - apply_detrend_data(Predictor_for_year, coeffss, metas)

#         # Fit the CCA model
#         # self.fit_cca(Predictor_, Predictant_)
#         self.fit_cca(Predictor_detrend, Predictant_st_detrend)
            
#         # Prepare test data
#         X_test_prepared = Predictor_for_year__.rename({"X": "lon", "Y": "lat"}).transpose('T', 'lat', 'lon')

#         # Predict
#         y_pred = self.cca_model.predict(X_test_prepared)
#         y_pred = self.cca_model.inverse_transform(self.cca_model.transform(X_test_prepared), y_pred)[1] 
#         result_ = y_pred.rename({"lon": "X", "lat": "Y"})
        
#         # last_trend_Y = ((extract_leading_eeof_component(Predictant_st)).isel(T=[-3]))
#         # last_trend_Y['T'] = result_['T']
#         # result_ = (result_ + last_trend_Y)
#         result_ = result_ + apply_detrend_data(result_, coeffs, meta)
#         result_ = reverse_standardize(result_, Predictant, clim_year_start, clim_year_end) 
        
#         # 2) Compute thresholds T1, T2 from climatology
#         index_start = Predictant.get_index("T").get_loc(str(clim_year_start)).start
#         index_end   = Predictant.get_index("T").get_loc(str(clim_year_end)).stop
#         rainfall_for_tercile = Predictant.isel(T=slice(index_start, index_end))
#         terciles = rainfall_for_tercile.quantile([0.33, 0.67], dim='T')
#         T1_emp = terciles.isel(quantile=0).drop_vars('quantile')
#         T2_emp = terciles.isel(quantile=1).drop_vars('quantile')
#         error_variance = (Predictant - hindcast_det).var(dim='T')

#         year = Predictor_for_year.coords['T'].values.astype('datetime64[Y]').astype(int)[0] + 1970  # Convert from epoch
#         T_value_1 = Predictant.isel(T=0).coords['T'].values  # Get the datetime64 value from da1
#         month_1 = T_value_1.astype('datetime64[M]').astype(int) % 12 + 1  # Extract month
#         new_T_value = np.datetime64(f"{year}-{month_1:02d}-{1:02d}")
#         forecast_expanded = result_.assign_coords(T=xr.DataArray([new_T_value], dims=["T"]))
#         forecast_expanded['T'] = forecast_expanded['T'].astype('datetime64[ns]')
        
        
#         dof = max(int(rainfall_for_tercile.sizes["T"]) - 1, 2)

#         dm = self.dist_method

#         # ---------- BESTFIT ----------
#         if dm == "bestfit":
#             if any(v is None for v in (best_code_da, best_shape_da, best_loc_da, best_scale_da)):
#                 raise ValueError(
#                     "dist_method='bestfit' requires best_code_da, best_shape_da, best_loc_da, best_scale_da."
#                 )
            
#             T1, T2 = xr.apply_ufunc(
#                 self._ppf_terciles_from_code,
#                 best_code_da,
#                 best_shape_da,
#                 best_loc_da,
#                 best_scale_da,
#                 input_core_dims=[(), (), (), ()],
#                 output_core_dims=[(), ()],
#                 vectorize=True,
#                 dask="parallelized",
#                 output_dtypes=[float, float],
#             )

#             forecast_prob = xr.apply_ufunc(
#                 self.calculate_tercile_probabilities_bestfit,
#                 forecast_expanded,
#                 error_variance,
#                 T1,
#                 T2,
#                 best_code_da,
#                 input_core_dims=[("T",), (), (), (), ()],
#                 output_core_dims=[("probability", "T")],
#                 vectorize=True,
#                 dask="parallelized",
#                 kwargs={"dof": dof},
#                 output_dtypes=[float],
#                 dask_gufunc_kwargs={
#                     "output_sizes": {"probability": 3},
#                     "allow_rechunk": True,
#                 },
#             )

#         # ---------- Nonparametric ----------
#         elif dm == "nonparam":
#             error_samples = Predictant - hindcast_det
#             forecast_prob = xr.apply_ufunc(
#                 self.calculate_tercile_probabilities_nonparametric,
#                 forecast_expanded,
#                 error_samples,
#                 T1_emp,
#                 T2_emp,
#                 input_core_dims=[("T",), ("T",), (), ()],
#                 output_core_dims=[("probability", "T")],
#                 vectorize=True,
#                 dask="parallelized",
#                 output_dtypes=[float],
#                 dask_gufunc_kwargs={
#                     "output_sizes": {"probability": 3},
#                     "allow_rechunk": True,
#                 },
#             )

#         else:
#             raise ValueError(f"Invalid dist_method: {self.dist_method}")
#         forecast_prob = forecast_prob.assign_coords(probability=('probability', ['PB', 'PN', 'PA']))
#         return forecast_expanded * mask, mask * forecast_prob.transpose('probability', 'T', 'Y', 'X')
     

#     def plot_cca_results(self, X=None, Y=None, n_modes=None, clim_year_start=None, clim_year_end=None):
#         """
#         Plots the CCA modes and scores.

#         Parameters:
#         - X: Optional xarray DataArray for predictors. If provided, the model will be fitted using X and Y.
#         - Y: Optional xarray DataArray for predictands.
#         - n_modes: Number of modes to plot. If None, plots all modes.
#         """
#         if X is not None and Y is not None:
#             mask = xr.where(~np.isnan(Y.isel(T=0)), 1, np.nan).drop_vars(['T']).squeeze().to_numpy()
#             # mask.name = None
            

#             X, coeffss, metas = detrended_data(X, dim="T")
        
#             Y = standardize_timeseries(Y, clim_year_start, clim_year_end)
#             Y, coeffs, meta = detrended_data(Y, dim="T")
        
#             # Fit the model using the provided data
#             self.fit_cca(X.fillna(0), Y.fillna(0))
#         elif self.cca_model is None:
#             raise ValueError("The CCA model has not been fitted yet. Provide X and Y data to fit the model.")

#         # Get components (modes) and scores
#         X_modes, Y_modes = self.cca_model.components()  # Spatial patterns
#         X_scores, Y_scores = self.cca_model.scores()    # Temporal projections (canonical variates)

#         # Get explained variances
#         var_explained_X = self.cca_model.fraction_variance_X_explained_by_X()
#         var_explained_Y = self.cca_model.fraction_variance_Y_explained_by_Y()
#         var_explained_Y_by_X = self.cca_model.fraction_variance_Y_explained_by_X()

#         # Determine number of modes to plot
#         if n_modes is None:
#             n_modes = self.n_modes

#         # Mode indices start from 1 in xeofs
#         mode_indices = range(1, n_modes + 1)

#         # Create subplots
#         fig, axes = plt.subplots(n_modes, 3, figsize=(15, 3 * n_modes))

#         if n_modes == 1:
#             axes = np.array([axes])

#         for i, mode in enumerate(mode_indices):

#             # First Column: Plot X_modes
#             ax = axes[i, 0]
#             X_mode = X_modes.sel(mode=mode)
#             X_mode.plot(ax=ax, vmin=-1, vmax=1, cmap= "RdBu_r")
#             var_X = var_explained_X.sel(mode=mode).values * 100
#             ax.set_title(f'X Mode {mode} ({var_X:.2f}% variance explained)')

#             # Second Column: Plot X_scores and Y_scores
#             ax = axes[i, 1]
#             X_score = X_scores.sel(mode=mode)
#             Y_score = Y_scores.sel(mode=mode)
#             var_Y_X = var_explained_Y_by_X.sel(mode=mode).values * 100
#             ax.plot(X_score['T'].dt.year.values, X_score, label='X Score')
#             ax.plot(Y_score['T'].dt.year.values, Y_score, label='Y Score')
#             ax.axhline(0, linestyle='--', lw=0.8, label="") #### line Canonical Variate = 0
#             ax.legend()
#             ax.set_title(f'Scores for Mode {mode} ({var_Y_X:.2f}% variance Y explained by X)')
#             ax.set_xlabel('Time')
#             ax.set_ylabel('Canonical Variate')

#             # Third Column: Plot Y_modes
#             ax = axes[i, 2]
#             Y_mode = (Y_modes.sel(mode=mode))*mask
#             Y_mode.plot(ax=ax, vmin=None, vmax=None, cmap= "RdBu_r")
#             var_Y = var_explained_Y.sel(mode=mode).values * 100
#             ax.set_title(f'Y Mode {mode} ({var_Y:.2f}% variance explained)')
            
#         plt.tight_layout()
#         plt.show()
        
#     @staticmethod
#     def _ppf_terciles_from_code(dist_code, shape, loc, scale):
#         """
#         Return tercile thresholds (T1, T2) from best-fit distribution parameters.
    
#         dist_code:
#             1: norm
#             2: lognorm
#             3: expon
#             4: gamma
#             5: weibull_min
#             6: t
#             7: poisson
#             8: nbinom
#         """
#         if np.isnan(dist_code):
#             return np.nan, np.nan
    
#         code = int(dist_code)
#         try:
#             if code == 1:
#                 return (
#                     norm.ppf(0.33, loc=loc, scale=scale),
#                     norm.ppf(0.67, loc=loc, scale=scale),
#                 )
#             elif code == 2:
#                 return (
#                     lognorm.ppf(0.33, s=shape, loc=loc, scale=scale),
#                     lognorm.ppf(0.67, s=shape, loc=loc, scale=scale),
#                 )
#             elif code == 3:
#                 return (
#                     expon.ppf(0.33, loc=loc, scale=scale),
#                     expon.ppf(0.67, loc=loc, scale=scale),
#                 )
#             elif code == 4:
#                 return (
#                     gamma.ppf(0.33, a=shape, loc=loc, scale=scale),
#                     gamma.ppf(0.67, a=shape, loc=loc, scale=scale),
#                 )
#             elif code == 5:
#                 return (
#                     weibull_min.ppf(0.33, c=shape, loc=loc, scale=scale),
#                     weibull_min.ppf(0.67, c=shape, loc=loc, scale=scale),
#                 )
#             elif code == 6:
#                 # Note: Renamed 't_dist' to 't' for standard scipy.stats
#                 return (
#                     t.ppf(0.33, df=shape, loc=loc, scale=scale),
#                     t.ppf(0.67, df=shape, loc=loc, scale=scale),
#                 )
#             elif code == 7:
#                 # Poisson: poisson.ppf(q, mu, loc=0)
#                 # ASSUMPTION: 'mu' (mean) is passed as 'shape'
#                 #             'loc' is passed as 'loc'
#                 #             'scale' is unused
#                 return (
#                     poisson.ppf(0.33, mu=shape, loc=loc),
#                     poisson.ppf(0.67, mu=shape, loc=loc),
#                 )
#             elif code == 8:
#                 # Negative Binomial: nbinom.ppf(q, n, p, loc=0)
#                 # ASSUMPTION: 'n' (successes) is passed as 'shape'
#                 #             'p' (probability) is passed as 'scale'
#                 #             'loc' is passed as 'loc'
#                 return (
#                     nbinom.ppf(0.33, n=shape, p=scale, loc=loc),
#                     nbinom.ppf(0.67, n=shape, p=scale, loc=loc),
#                 )
#         except Exception:
#             return np.nan, np.nan
    
#         # Fallback if code is not 1-8
#         return np.nan, np.nan
        
#     @staticmethod
#     def weibull_shape_solver(k, M, V):
#         """
#         Function to find the root of the Weibull shape parameter 'k'.
#         We find 'k' such that the theoretical variance/mean^2 ratio
#         matches the observed V/M^2 ratio.
#         """
#         # Guard against invalid 'k' values during solving
#         if k <= 0:
#             return -np.inf
#         try:
#             g1 = gamma_function(1 + 1/k)
#             g2 = gamma_function(1 + 2/k)
            
#             # This is the V/M^2 ratio *implied by k*
#             implied_v_over_m_sq = (g2 / (g1**2)) - 1
            
#             # This is the *observed* ratio
#             observed_v_over_m_sq = V / (M**2)
            
#             # Return the difference (we want this to be 0)
#             return observed_v_over_m_sq - implied_v_over_m_sq
#         except ValueError:
#             return -np.inf # Handle math errors

#     @staticmethod
#     def calculate_tercile_probabilities_bestfit(best_guess, error_variance, T1, T2, dist_code, dof 
#     ):
#         """
#         Generic tercile probabilities using best-fit family per grid cell.

#         Inputs (per grid cell):
#         - best_guess : 1D array over T (hindcast_det or forecast_det)
#         - T1, T2     : scalar terciles from climatological best-fit distribution
#         - dist_code  : int, as in _ppf_terciles_from_code
#         - shape, loc, scale : scalars from climatology fit

#         Strategy:
#         - For each time step, build a predictive distribution of the same family:
#             * Use best_guess[t] to adjust mean / location;
#             * Keep shape parameters from climatology.
#         - Then compute probabilities:
#             P(B) = F(T1), P(N) = F(T2) - F(T1), P(A) = 1 - F(T2).
#         """
        
#         best_guess = np.asarray(best_guess, float)
#         error_variance = np.asarray(error_variance, dtype=float)
#         # T1 = np.asarray(T1, dtype=float)
#         # T2 = np.asarray(T2, dtype=float)
#         n_time = best_guess.size
#         out = np.full((3, n_time), np.nan, float)

#         if np.all(np.isnan(best_guess)) or np.isnan(dist_code) or np.isnan(T1) or np.isnan(T2) or np.isnan(error_variance):
#             return out

#         code = int(dist_code)

#         # Normal: loc = forecast; scale from clim
#         if code == 1:
#             error_std = np.sqrt(error_variance)
#             out[0, :] = norm.cdf(T1, loc=best_guess, scale=error_std)
#             out[1, :] = norm.cdf(T2, loc=best_guess, scale=error_std) - norm.cdf(T1, loc=best_guess, scale=error_std)
#             out[2, :] = 1 - norm.cdf(T2, loc=best_guess, scale=error_std)

#         # Lognormal: shape = sigma from clim; enforce mean = best_guess
#         elif code == 2:
#             sigma = np.sqrt(np.log(1 + error_variance / (best_guess**2)))
#             mu = np.log(best_guess) - sigma**2 / 2
#             out[0, :] = lognorm.cdf(T1, s=sigma, scale=np.exp(mu))
#             out[1, :] = lognorm.cdf(T2, s=sigma, scale=np.exp(mu)) - lognorm.cdf(T1, s=sigma, scale=np.exp(mu))
#             out[2, :] = 1 - lognorm.cdf(T2, s=sigma, scale=np.exp(mu))      


#         # Exponential: keep scale from clim; shift loc so mean = best_guess
#         elif code == 3:
#             c1 = expon.cdf(T1, loc=best_guess, scale=np.sqrt(error_variance))
#             c2 = expon.cdf(T2, loc=loc_t, scale=np.sqrt(error_variance))
#             out[0, :] = c1
#             out[1, :] = c2 - c1
#             out[2, :] = 1.0 - c2

#         # Gamma: use shape from clim; set scale so mean = best_guess
#         elif code == 4:
#             alpha = (best_guess ** 2) / error_variance
#             theta = error_variance / best_guess
#             c1 = gamma.cdf(T1, a=alpha, scale=theta)
#             c2 = gamma.cdf(T2, a=alpha, scale=theta)
#             out[0, :] = c1
#             out[1, :] = c2 - c1
#             out[2, :] = 1.0 - c2

#         elif code == 5: # Assuming 5 is for Weibull   
        
#             for i in range(n_time):
#                 # Get the scalar values for this specific element (e.g., grid cell)
#                 M = best_guess[i]
#                 print(M)
#                 V = error_variance
#                 print(V)
                
#                 # Handle cases with no variance to avoid division by zero
#                 if V <= 0 or M <= 0:
#                     out[0, i] = np.nan
#                     out[1, i] = np.nan
#                     out[2, i] = np.nan
#                     continue # Skip to the next element
        
#                 # --- 1. Numerically solve for shape 'k' ---
#                 # We need a reasonable starting guess. 2.0 is common (Rayleigh dist.)
#                 initial_guess = 2.0
                
#                 # fsolve finds the root of our helper function
#                 k = fsolve(weibull_shape_solver, initial_guess, args=(M, V))[0]
        
#                 # --- 2. Check for bad solution and calculate scale 'lambda' ---
#                 if k <= 0:
#                     # Solver failed
#                     out[0, i] = np.nan
#                     out[1, i] = np.nan
#                     out[2, i] = np.nan
#                     continue
                
#                 # With 'k' found, we can now algebraically find scale 'lambda'
#                 # In scipy.stats, scale is 'scale'
#                 lambda_scale = M / gamma_function(1 + 1/k)
        
#                 # --- 3. Calculate Probabilities ---
#                 # In scipy.stats, shape 'k' is 'c'
#                 # Use the T1 and T2 values for this specific element
                
#                 c1 = weibull_min.cdf(T1, c=k, loc=0, scale=lambda_scale)
#                 c2 = weibull_min.cdf(T2, c=k, loc=0, scale=lambda_scale)
        
#                 out[0, i] = c1
#                 out[1, i] = c2 - c1
#                 out[2, i] = 1.0 - c2

#         # Student-t: df from clim; scale from clim; loc = best_guess
#         elif code == 6:       
#             # Check if df is valid for variance calculation
#             if dof <= 2:
#                 # Cannot calculate scale, fill with NaNs
#                 out[0, :] = np.nan
#                 out[1, :] = np.nan
#                 out[2, :] = np.nan
#             else:
#                 # 1. Calculate t-distribution parameters
#                 # 'loc' (mean) is just the best_guess
#                 loc = best_guess
#                 # 'scale' is calculated from the variance and df
#                 # Variance = scale**2 * (df / (df - 2))
#                 scale = np.sqrt(error_variance * (dof - 2) / dof)
                
#                 # 2. Calculate probabilities
#                 c1 = t.cdf(T1, df=dof, loc=loc, scale=scale)
#                 c2 = t.cdf(T2, df=dof, loc=loc, scale=scale)

#                 out[0, :] = c1
#                 out[1, :] = c2 - c1
#                 out[2, :] = 1.0 - c2

#         elif code == 7: # Assuming 7 is for Poisson
            
#             # --- 1. Set the Poisson parameter 'mu' ---
#             # The 'mu' parameter is the mean.
            
#             # A warning is strongly recommended if error_variance is different from best_guess
#             if not np.allclose(best_guess, error_variance, atol=0.5):
#                 print("Warning: 'error_variance' is not equal to 'best_guess'.")
#                 print("Poisson model assumes mean=variance and is likely inappropriate.")
#                 print("Consider using Negative Binomial.")
            
#             mu = best_guess
        
#             # --- 2. Calculate Probabilities ---
#             # poisson.cdf(k, mu) calculates P(X <= k)
            
#             c1 = poisson.cdf(T1, mu=mu)
#             c2 = poisson.cdf(T2, mu=mu)
            
#             out[0, :] = c1
#             out[1, :] = c2 - c1
#             out[2, :] = 1.0 - c2

#         elif code == 8: # Assuming 8 is for Negative Binomial
            
#             # --- 1. Calculate Negative Binomial Parameters ---
#             # This model is ONLY valid for overdispersion (Variance > Mean).
#             # We will use np.where to set parameters to NaN if V <= M.
            
#             # p = Mean / Variance
#             p = np.where(error_variance > best_guess, 
#                          best_guess / error_variance, 
#                          np.nan)
            
#             # n = Mean^2 / (Variance - Mean)
#             n = np.where(error_variance > best_guess, 
#                          (best_guess**2) / (error_variance - best_guess), 
#                          np.nan)
            
#             # --- 2. Calculate Probabilities ---
#             # The nbinom.cdf function will propagate NaNs, correctly
#             # handling the cases where the model was invalid.
            
#             c1 = nbinom.cdf(T1, n=n, p=p)
#             c2 = nbinom.cdf(T2, n=n, p=p)
            
#             out[0, :] = c1
#             out[1, :] = c2 - c1
#             out[2, :] = 1.0 - c2
            
#         else:
#             raise ValueError(f"Invalid distribution")

#         return out

#     @staticmethod
#     def calculate_tercile_probabilities_nonparametric(best_guess, error_samples, first_tercile, second_tercile):
#         """Non-parametric method using historical error samples."""
#         n_time = len(best_guess)
#         pred_prob = np.full((3, n_time), np.nan, dtype=float)
#         for t in range(n_time):
#             if np.isnan(best_guess[t]):
#                 continue
#             dist = best_guess[t] + error_samples
#             dist = dist[np.isfinite(dist)]
#             if len(dist) == 0:
#                 continue
#             p_below = np.mean(dist < first_tercile)
#             p_between = np.mean((dist >= first_tercile) & (dist < second_tercile))
#             p_above = 1.0 - (p_below + p_between)
#             pred_prob[0, t] = p_below
#             pred_prob[1, t] = p_between
#             pred_prob[2, t] = p_above
#         return pred_prob


#     def compute_prob(
#         self,
#         Predictant: xr.DataArray,
#         clim_year_start,
#         clim_year_end,
#         hindcast_det: xr.DataArray,
#         best_code_da: xr.DataArray = None,
#         best_shape_da: xr.DataArray = None,
#         best_loc_da: xr.DataArray = None,
#         best_scale_da: xr.DataArray = None
#     ) -> xr.DataArray:
#         """
#         Compute tercile probabilities for deterministic hindcasts.

#         If dist_method == 'bestfit':
#             - Use cluster-based best-fit distributions to:
#                 * derive terciles analytically from (best_code_da, best_shape_da, best_loc_da, best_scale_da),
#                 * compute predictive probabilities using the same family.

#         Otherwise:
#             - Use empirical terciles from Predictant climatology and the selected
#               parametric / nonparametric method.

#         Parameters
#         ----------
#         Predictant : xarray.DataArray
#             Observed data (T, Y, X) or (T, Y, X, M).
#         clim_year_start, clim_year_end : int or str
#             Climatology period (inclusive) for thresholds.
#         hindcast_det : xarray.DataArray
#             Deterministic hindcast (T, Y, X).
#         best_code_da, best_shape_da, best_loc_da, best_scale_da : xarray.DataArray, optional
#             Output from WAS_TransformData.fit_best_distribution_grid, required for 'bestfit'.

#         Returns
#         -------
#         hindcast_prob : xarray.DataArray
#             Probabilities with dims (probability=['PB','PN','PA'], T, Y, X).
#         """
#         # Handle member dimension if present
#         if "M" in Predictant.dims:
#             Predictant = Predictant.isel(M=0).drop_vars("M").squeeze()

#         # Ensure dimension order
#         Predictant = Predictant.transpose("T", "Y", "X")

#         # Spatial mask
#         mask = xr.where(~np.isnan(Predictant.isel(T=0)), 1.0, np.nan)

#         # Climatology subset
#         clim = Predictant.sel(T=slice(str(clim_year_start), str(clim_year_end)))
#         if clim.sizes.get("T", 0) < 3:
#             raise ValueError("Not enough years in climatology period for terciles.")

#         # Error variance for predictive distributions
#         error_variance = (Predictant - hindcast_det).var(dim="T")
#         dof = max(int(clim.sizes["T"]) - 1, 2)

#         # Empirical terciles (used by non-bestfit methods)
#         terciles_emp = clim.quantile([0.33, 0.67], dim="T")
#         T1_emp = terciles_emp.isel(quantile=0).drop_vars("quantile")
#         T2_emp = terciles_emp.isel(quantile=1).drop_vars("quantile")
        

#         dm = self.dist_method

#         # ---------- BESTFIT: zone-wise optimal distributions ----------
#         if dm == "bestfit":
#             if any(v is None for v in (best_code_da, best_shape_da, best_loc_da, best_scale_da)):
#                 raise ValueError(
#                     "dist_method='bestfit' requires best_code_da, best_shape_da_da, best_loc_da, best_scale_da."
#                 )

#             # T1, T2 from best-fit distributions (per grid)
#             T1, T2 = xr.apply_ufunc(
#                 self._ppf_terciles_from_code,
#                 best_code_da,
#                 best_shape_da,
#                 best_loc_da,
#                 best_scale_da,
#                 input_core_dims=[(), (), (), ()],
#                 output_core_dims=[(), ()],
#                 vectorize=True,
#                 dask="parallelized",
#                 output_dtypes=[float, float],
#             )

#             # Predictive probabilities using same family
#             hindcast_prob = xr.apply_ufunc(
#                 self.calculate_tercile_probabilities_bestfit,
#                 hindcast_det,
#                 error_variance,
#                 T1,
#                 T2,
#                 best_code_da,
#                 input_core_dims=[("T",), (), (), (), ()],
#                 output_core_dims=[("probability", "T")],
#                 vectorize=True,
#                 kwargs={'dof': dof},
#                 dask="parallelized",
#                 output_dtypes=[float],
#                 dask_gufunc_kwargs={
#                     "output_sizes": {"probability": 3},
#                     "allow_rechunk": True,
#                 },
#             )

#         # ---------- Nonparametric ----------
#         elif dm == "nonparam":
#             error_samples = Predictant - hindcast_det
#             hindcast_prob = xr.apply_ufunc(
#                 self.calculate_tercile_probabilities_nonparametric,
#                 hindcast_det,
#                 error_samples,
#                 T1_emp,
#                 T2_emp,
#                 input_core_dims=[("T",), ("T",), (), ()],
#                 output_core_dims=[("probability", "T")],
#                 vectorize=True,
#                 dask="parallelized",
#                 output_dtypes=[float],
#                 dask_gufunc_kwargs={
#                     "output_sizes": {"probability": 3},
#                     "allow_rechunk": True,
#                 },
#             )

#         else:
#             raise ValueError(f"Invalid dist_method: {self.dist_method}")

#         hindcast_prob = hindcast_prob.assign_coords(
#             probability=("probability", ["PB", "PN", "PA"])
#         )
#         return (hindcast_prob * mask).transpose("probability", "T", "Y", "X")


# class WAS_CCA_strict:
#     def __init__(self, n_modes=4, n_pca_modes=8, standardize=False, use_coslat=True, use_pca=True, dist_method="bestfit"):
#         """
#         Initialize parameters. The CCA model instance is created in fit_cca to ensure safe CV.
#         """
#         self.n_modes = n_modes
#         self.n_pca_modes = n_pca_modes
#         self.standardize = standardize
#         self.use_coslat = use_coslat
#         self.use_pca = use_pca
#         self.dist_method = dist_method
        
#         self.cca_model = None
#         self.cca = None

#     @staticmethod
#     def _safe_drop_vars(da, names):
#         """Drop variables if they exist; compatible with older xarray versions."""
#         existing = [name for name in names if name in da.coords or name in da.variables]
#         if existing:
#             return da.drop_vars(existing)
#         return da
    
    
#     @staticmethod
#     def _spatial_mask(da):
#         """
#         Return a Y/X mask preserving xarray coordinates.
#         Do not convert to NumPy, because xarray broadcasting is safer.
#         """
#         mask = xr.where(~np.isnan(da.isel(T=0)), 1.0, np.nan)
#         mask = WAS_CCA_strict._safe_drop_vars(mask, ["T"])
#         return mask.squeeze()
    
    
#     @staticmethod
#     def _rename_to_latlon(da):
#         """
#         Convert WAS dimensions Y/X to xeofs dimensions lat/lon.
#         """
#         out = da
#         rename = {}
#         if "Y" in out.dims:
#             rename["Y"] = "lat"
#         if "X" in out.dims:
#             rename["X"] = "lon"
#         if rename:
#             out = out.rename(rename)
    
#         expected = [d for d in ("T", "lat", "lon") if d in out.dims]
#         other = [d for d in out.dims if d not in expected]
#         return out.transpose(*(expected + other))
    
    
#     @staticmethod
#     def _rename_to_YX(da):
#         """
#         Convert xeofs dimensions lat/lon back to WAS dimensions Y/X.
#         """
#         rename = {}
#         if "lat" in da.dims:
#             rename["lat"] = "Y"
#         if "lon" in da.dims:
#             rename["lon"] = "X"
#         if rename:
#             da = da.rename(rename)
    
#         expected = [d for d in ("T", "Y", "X") if d in da.dims]
#         other = [d for d in da.dims if d not in expected]
#         return da.transpose(*(expected + other))
    
    
#     @staticmethod
#     def _fill_spatial_gaps(da, ref=None):
#         """
#         Fill missing values before CCA prediction.
    
#         ref is normally the historical predictor used to compute a time-mean field.
#         """
#         out = da
    
#         if ref is not None and "T" in ref.dims:
#             out = out.fillna(ref.mean(dim="T", skipna=True))
    
#         for dim in ("Y", "X", "lat", "lon"):
#             if dim in out.dims:
#                 out = out.ffill(dim=dim).bfill(dim=dim)
    
#         return out.fillna(0.0)
    
    
#     @staticmethod
#     def _target_time_from_predictor(Predictant, Predictor_for_year):
#         """
#         Build the forecast target time.
    
#         Example:
#         Predictor_for_year T may be April 2026, but predictand target season may
#         be represented by July. This function keeps the forecast year from the
#         predictor and the target month from Predictant.
#         """
#         try:
#             year = int(Predictor_for_year["T"].dt.year.values[0])
#         except Exception:
#             year = (
#                 Predictor_for_year.coords["T"]
#                 .values.astype("datetime64[Y]")
#                 .astype(int)[0] + 1970
#             )
    
#         target_month = (
#             Predictant.isel(T=0)
#             .coords["T"]
#             .values.astype("datetime64[M]")
#             .astype(int) % 12 + 1
#         )
    
#         return np.datetime64(f"{year}-{int(target_month):02d}-01")
    
    
#     @staticmethod
#     def _normalize_probabilities(prob):
#         """
#         Clip and renormalize PB/PN/PA so that probabilities sum to one.
#         """
#         prob = prob.clip(min=0.0, max=1.0)
#         total = prob.sum(dim="probability")
#         prob = xr.where(total > 0, prob / total, np.nan)
#         return prob

#     def fit_cca(self, X_train, y_train):
#         """
#         Fit CCA using already prepared fields.
    
#         Important:
#         In the CCA branch of WAS_Cross_Validator, X_train and y_train are already
#         detrended / standardized before this method is called.
#         """
#         self.cca = xe.cross.CCA(
#             n_modes=self.n_modes,
#             standardize=self.standardize,
#             use_coslat=self.use_coslat,
#             use_pca=self.use_pca,
#             n_pca_modes=self.n_pca_modes,
#         )
    
#         X_train_final, y_train_final = self.preprocess_data(X_train, y_train)
#         self.cca_model = self.cca.fit(X_train_final, y_train_final, dim="T")
#         return self.cca_model
    
    
#     def preprocess_data(self, X, Y):
#         """
#         Prepare training data for xeofs CCA.
#         This method does not detrend or standardize; those operations are handled
#         upstream in forecast() or in the CCA cross-validation branch.
#         """
#         X_final = X.fillna(X.mean(dim="T", skipna=True)).fillna(0.0)
#         Y_final = Y.fillna(Y.mean(dim="T", skipna=True)).fillna(0.0)
    
#         X_final = self._rename_to_latlon(X_final)
#         Y_final = self._rename_to_latlon(Y_final)
    
#         return X_final, Y_final
    
    
#     def preprocess_test_data(self, X_test, y_test=None, X_train=None, y_train=None):
#         """
#         Prepare test data for xeofs CCA.
#         """
#         X_test_prepared = X_test
    
#         if X_train is not None and "T" in X_train.dims:
#             X_test_prepared = X_test_prepared.fillna(X_train.mean(dim="T", skipna=True))
    
#         X_test_prepared = X_test_prepared.fillna(0.0)
#         X_test_prepared = self._rename_to_latlon(X_test_prepared)
    
#         if y_test is None:
#             return X_test_prepared, None
    
#         y_test_prepared = y_test
#         if y_train is not None and "T" in y_train.dims:
#             y_test_prepared = y_test_prepared.fillna(y_train.mean(dim="T", skipna=True))
    
#         y_test_prepared = y_test_prepared.fillna(0.0)
#         y_test_prepared = self._rename_to_latlon(y_test_prepared)
    
#         return X_test_prepared, y_test_prepared
    
    
#     def compute_model(self, X_train, y_train, X_test, y_test=None):
#         """
#         Low-level CCA prediction.
    
#         Expected input:
#         - X_train and X_test are already in the same transformed predictor space.
#         - y_train is already in transformed predictand space.
#         - y_test is used only to preserve the correct T coordinate during CV.
#         """
#         self.fit_cca(X_train, y_train)
    
#         X_test_prepared, y_test_prepared = self.preprocess_test_data(
#             X_test=X_test,
#             y_test=y_test,
#             X_train=X_train,
#             y_train=y_train,
#         )
    
#         y_pred = self.cca_model.predict(X_test_prepared)
    
#         y_pred_phys = self.cca_model.inverse_transform(
#             self.cca_model.transform(X_test_prepared),
#             y_pred,
#         )[1]
    
#         if y_test_prepared is not None:
#             y_pred_phys = y_pred_phys.assign_coords(T=y_test_prepared["T"])
    
#         hindcast = self._rename_to_YX(y_pred_phys)
#         return hindcast

#     def forecast(
#         self,
#         Predictant,
#         clim_year_start,
#         clim_year_end,
#         Predictor,
#         hindcast_det,
#         Predictor_for_year,
#         best_code_da=None,
#         best_shape_da=None,
#         best_loc_da=None,
#         best_scale_da=None,
#     ):
#         """
#         Operational CCA forecast.
    
#         Historical transformation:
#         - Predictor is linearly detrended.
#         - Predictant is standardized, then linearly detrended.
    
#         Forecast transformation:
#         - Predictor_for_year is filled.
#         - The historical predictor trend is evaluated at Predictor_for_year T.
#         - Forecast predictor anomaly is passed to CCA.
    
#         Output transformation:
#         - CCA output is assigned the target predictand T.
#         - Predictand trend is added back at target T.
#         - Standardization is reversed to physical units.
#         """
    
#         Predictant_safe = Predictant.copy(deep=True)
#         mask = self._spatial_mask(Predictant_safe)
    
#         # 1. Historical predictor transformation
#         Predictor_filled = self._fill_spatial_gaps(Predictor)
#         Predictor_detrend, coeffs_X, meta_X = detrended_data(Predictor_filled, dim="T")
#         Predictor_detrend = Predictor_detrend.fillna(0.0)
    
#         # 2. Historical predictand transformation
#         Predictant_st = standardize_timeseries(
#             Predictant_safe,
#             clim_year_start,
#             clim_year_end,
#         )
#         Predictant_st_detrend, coeffs_Y, meta_Y = detrended_data(Predictant_st, dim="T")
#         Predictant_st_detrend = Predictant_st_detrend.fillna(0.0)
    
#         # 3. Forecast predictor transformation
#         Predictor_for_year_filled = self._fill_spatial_gaps(
#             Predictor_for_year,
#             ref=Predictor_filled,
#         )
    
#         Predictor_for_year_detrended = (
#             Predictor_for_year_filled
#             - apply_detrend_data(Predictor_for_year_filled, coeffs_X, meta_X)
#         )
#         Predictor_for_year_detrended = Predictor_for_year_detrended.fillna(0.0)
    
#         # 4. Fit CCA and predict in transformed space
#         self.fit_cca(Predictor_detrend, Predictant_st_detrend)
    
#         X_forecast_prepared = self._rename_to_latlon(Predictor_for_year_detrended)
    
#         y_pred = self.cca_model.predict(X_forecast_prepared)
#         y_pred = self.cca_model.inverse_transform(
#             self.cca_model.transform(X_forecast_prepared),
#             y_pred,
#         )[1]
    
#         result_st_detrended = self._rename_to_YX(y_pred)
    
#         # 5. Assign target predictand time BEFORE adding back predictand trend
#         target_T = self._target_time_from_predictor(Predictant_safe, Predictor_for_year)
#         result_st_detrended = result_st_detrended.assign_coords(
#             T=xr.DataArray([target_T], dims=["T"])
#         )
#         result_st_detrended["T"] = result_st_detrended["T"].astype("datetime64[ns]")
    
#         # 6. Add predictand trend back at the target predictand time
#         result_st = result_st_detrended + apply_detrend_data(
#             result_st_detrended,
#             coeffs_Y,
#             meta_Y,
#         )
    
#         # 7. Reverse standardization to physical units
#         forecast_det = reverse_standardize(
#             result_st,
#             Predictant_safe,
#             clim_year_start,
#             clim_year_end,
#         )
    
#         forecast_det = forecast_det.transpose("T", "Y", "X") * mask
    
#         # Optional physical bound for rainfall-like predictands
#         forecast_det = forecast_det.clip(min=0.0)
    
#         # 8. Probabilities
#         clim = Predictant_safe.sel(T=slice(str(clim_year_start), str(clim_year_end)))
#         if clim.sizes.get("T", 0) < 3:
#             raise ValueError("Not enough years in climatology period for terciles.")
    
#         terciles = clim.quantile([1.0 / 3.0, 2.0 / 3.0], dim="T")
#         T1_emp = terciles.isel(quantile=0).drop_vars("quantile")
#         T2_emp = terciles.isel(quantile=1).drop_vars("quantile")
    
#         error_variance = (Predictant_safe - hindcast_det).var(dim="T")
#         dof = max(int(clim.sizes["T"]) - 1, 2)
    
#         dm = self.dist_method
    
#         if dm == "bestfit":
#             if any(v is None for v in (best_code_da, best_shape_da, best_loc_da, best_scale_da)):
#                 raise ValueError(
#                     "dist_method='bestfit' requires best_code_da, best_shape_da, "
#                     "best_loc_da, and best_scale_da."
#                 )
    
#             T1, T2 = xr.apply_ufunc(
#                 self._ppf_terciles_from_code,
#                 best_code_da,
#                 best_shape_da,
#                 best_loc_da,
#                 best_scale_da,
#                 input_core_dims=[(), (), (), ()],
#                 output_core_dims=[(), ()],
#                 vectorize=True,
#                 dask="parallelized",
#                 output_dtypes=[float, float],
#             )
    
#             forecast_prob = xr.apply_ufunc(
#                 self.calculate_tercile_probabilities_bestfit,
#                 forecast_det,
#                 error_variance,
#                 T1,
#                 T2,
#                 best_code_da,
#                 input_core_dims=[("T",), (), (), (), ()],
#                 output_core_dims=[("probability", "T")],
#                 vectorize=True,
#                 dask="parallelized",
#                 kwargs={"dof": dof},
#                 output_dtypes=[float],
#                 dask_gufunc_kwargs={
#                     "output_sizes": {"probability": 3},
#                     "allow_rechunk": True,
#                 },
#             )
    
#         elif dm == "nonparam":
#             error_samples = Predictant_safe - hindcast_det
    
#             forecast_prob = xr.apply_ufunc(
#                 self.calculate_tercile_probabilities_nonparametric,
#                 forecast_det,
#                 error_samples,
#                 T1_emp,
#                 T2_emp,
#                 input_core_dims=[("T",), ("T",), (), ()],
#                 output_core_dims=[("probability", "T")],
#                 vectorize=True,
#                 dask="parallelized",
#                 output_dtypes=[float],
#                 dask_gufunc_kwargs={
#                     "output_sizes": {"probability": 3},
#                     "allow_rechunk": True,
#                 },
#             )
    
#         else:
#             raise ValueError(f"Invalid dist_method: {self.dist_method}")
    
#         forecast_prob = forecast_prob.assign_coords(
#             probability=("probability", ["PB", "PN", "PA"])
#         )
    
#         forecast_prob = forecast_prob.transpose("probability", "T", "Y", "X") * mask
#         forecast_prob = self._normalize_probabilities(forecast_prob)
    
#         return forecast_det, forecast_prob     

#     def plot_cca_results(
#         self,
#         X=None,
#         Y=None,
#         n_modes=None,
#         clim_year_start=None,
#         clim_year_end=None,
#     ):
#         """
#         Plot CCA spatial modes and canonical scores.
    
#         If X and Y are provided, the CCA model is refitted using the same
#         transformation logic used for the CCA workflow:
#           - predictor: linear detrending
#           - predictand: standardization + linear detrending
    
#         Parameters
#         ----------
#         X : xarray.DataArray, optional
#             Predictor field with dimensions ('T', 'Y', 'X') or ('T', 'lat', 'lon').
#         Y : xarray.DataArray, optional
#             Predictand field with dimensions ('T', 'Y', 'X') or ('T', 'lat', 'lon').
#         n_modes : int, optional
#             Number of CCA modes to plot. If None, uses self.n_modes.
#         clim_year_start, clim_year_end : int or str, optional
#             Climatology period used to standardize Y.
#         """
    
#         mask = None
    
#         # ------------------------------------------------------------------
#         # 1. Optionally fit/refit the CCA model
#         # ------------------------------------------------------------------
#         if X is not None and Y is not None:
#             # Keep mask as xarray, not NumPy
#             mask = xr.where(~np.isnan(Y.isel(T=0)), 1.0, np.nan).squeeze()
    
#             if "T" in mask.coords:
#                 mask = mask.drop_vars("T")
    
#             # Rename mask to match xeofs output dimensions
#             rename_mask = {}
#             if "Y" in mask.dims:
#                 rename_mask["Y"] = "lat"
#             if "X" in mask.dims:
#                 rename_mask["X"] = "lon"
#             if rename_mask:
#                 mask = mask.rename(rename_mask)
    
#             # Predictor transformation
#             X_detrended, _, _ = detrended_data(X, dim="T")
#             X_ready = X_detrended.fillna(0.0)
    
#             # Predictand transformation
#             Y_st = standardize_timeseries(
#                 Y,
#                 clim_year_start,
#                 clim_year_end,
#             )
    
#             # Avoid inf values where std = 0
#             Y_st = Y_st.where(np.isfinite(Y_st))
    
#             Y_st_detrended, _, _ = detrended_data(Y_st, dim="T")
#             Y_ready = Y_st_detrended.fillna(0.0)
    
#             # Fit model using transformed data
#             self.fit_cca(X_ready, Y_ready)
    
#         elif self.cca_model is None:
#             raise ValueError(
#                 "The CCA model has not been fitted yet. Provide X and Y data to fit the model."
#             )
    
#         # ------------------------------------------------------------------
#         # 2. Extract CCA outputs
#         # ------------------------------------------------------------------
#         X_modes, Y_modes = self.cca_model.components()
#         X_scores, Y_scores = self.cca_model.scores()
    
#         var_explained_X = self.cca_model.fraction_variance_X_explained_by_X()
#         var_explained_Y = self.cca_model.fraction_variance_Y_explained_by_Y()
#         var_explained_Y_by_X = self.cca_model.fraction_variance_Y_explained_by_X()
    
#         # ------------------------------------------------------------------
#         # 3. Determine number of modes safely
#         # ------------------------------------------------------------------
#         available_modes = len(X_modes["mode"])
    
#         if n_modes is None:
#             n_modes = min(self.n_modes, available_modes)
#         else:
#             n_modes = min(int(n_modes), available_modes)
    
#         mode_indices = X_modes["mode"].values[:n_modes]
    
#         # ------------------------------------------------------------------
#         # 4. Create plot
#         # ------------------------------------------------------------------
#         fig, axes = plt.subplots(
#             n_modes,
#             3,
#             figsize=(15, 3.5 * n_modes),
#             squeeze=False,
#         )
    
#         for i, mode in enumerate(mode_indices):
#             # --------------------------------------------------------------
#             # Column 1: Predictor spatial mode
#             # --------------------------------------------------------------
#             ax = axes[i, 0]
    
#             X_mode = X_modes.sel(mode=mode)
#             X_mode.plot(ax=ax, cmap="RdBu_r")
    
#             var_X = float(var_explained_X.sel(mode=mode).values) * 100.0
#             ax.set_title(f"X Mode {mode} ({var_X:.2f}% X variance)")
    
#             # --------------------------------------------------------------
#             # Column 2: Canonical scores
#             # --------------------------------------------------------------
#             ax = axes[i, 1]
    
#             X_score = X_scores.sel(mode=mode)
#             Y_score = Y_scores.sel(mode=mode)
    
#             if np.issubdtype(X_score["T"].dtype, np.datetime64):
#                 time_axis = X_score["T"].dt.year.values
#                 xlabel = "Year"
#             else:
#                 time_axis = X_score["T"].values
#                 xlabel = "Time"
    
#             var_Y_by_X = float(var_explained_Y_by_X.sel(mode=mode).values) * 100.0
    
#             ax.plot(time_axis, X_score.values, label="X score")
#             ax.plot(time_axis, Y_score.values, label="Y score")
#             ax.axhline(0.0, linestyle="--", lw=0.8)
    
#             ax.legend()
#             ax.set_title(f"Scores Mode {mode} ({var_Y_by_X:.2f}% Y explained by X)")
#             ax.set_xlabel(xlabel)
#             ax.set_ylabel("Canonical variate")
    
#             # --------------------------------------------------------------
#             # Column 3: Predictand spatial mode
#             # --------------------------------------------------------------
#             ax = axes[i, 2]
    
#             Y_mode = Y_modes.sel(mode=mode)
    
#             if mask is not None:
#                 Y_mode = Y_mode * mask
    
#             Y_mode.plot(ax=ax, cmap="RdBu_r")
    
#             var_Y = float(var_explained_Y.sel(mode=mode).values) * 100.0
#             ax.set_title(f"Y Mode {mode} ({var_Y:.2f}% Y variance)")
    
#         plt.tight_layout()
#         plt.show()
        
#     @staticmethod
#     def _ppf_terciles_from_code(dist_code, shape, loc, scale):
#         """
#         Return tercile thresholds (T1, T2) from best-fit distribution parameters.
    
#         dist_code:
#             1: norm
#             2: lognorm
#             3: expon
#             4: gamma
#             5: weibull_min
#             6: t
#             7: poisson
#             8: nbinom
#         """
#         if np.isnan(dist_code):
#             return np.nan, np.nan
    
#         code = int(dist_code)
#         try:
#             if code == 1:
#                 return (
#                     norm.ppf(0.33, loc=loc, scale=scale),
#                     norm.ppf(0.67, loc=loc, scale=scale),
#                 )
#             elif code == 2:
#                 return (
#                     lognorm.ppf(0.33, s=shape, loc=loc, scale=scale),
#                     lognorm.ppf(0.67, s=shape, loc=loc, scale=scale),
#                 )
#             elif code == 3:
#                 return (
#                     expon.ppf(0.33, loc=loc, scale=scale),
#                     expon.ppf(0.67, loc=loc, scale=scale),
#                 )
#             elif code == 4:
#                 return (
#                     gamma.ppf(0.33, a=shape, loc=loc, scale=scale),
#                     gamma.ppf(0.67, a=shape, loc=loc, scale=scale),
#                 )
#             elif code == 5:
#                 return (
#                     weibull_min.ppf(0.33, c=shape, loc=loc, scale=scale),
#                     weibull_min.ppf(0.67, c=shape, loc=loc, scale=scale),
#                 )
#             elif code == 6:
#                 # Note: Renamed 't_dist' to 't' for standard scipy.stats
#                 return (
#                     t.ppf(0.33, df=shape, loc=loc, scale=scale),
#                     t.ppf(0.67, df=shape, loc=loc, scale=scale),
#                 )
#             elif code == 7:
#                 # Poisson: poisson.ppf(q, mu, loc=0)
#                 # ASSUMPTION: 'mu' (mean) is passed as 'shape'
#                 #             'loc' is passed as 'loc'
#                 #             'scale' is unused
#                 return (
#                     poisson.ppf(0.33, mu=shape, loc=loc),
#                     poisson.ppf(0.67, mu=shape, loc=loc),
#                 )
#             elif code == 8:
#                 # Negative Binomial: nbinom.ppf(q, n, p, loc=0)
#                 # ASSUMPTION: 'n' (successes) is passed as 'shape'
#                 #             'p' (probability) is passed as 'scale'
#                 #             'loc' is passed as 'loc'
#                 return (
#                     nbinom.ppf(0.33, n=shape, p=scale, loc=loc),
#                     nbinom.ppf(0.67, n=shape, p=scale, loc=loc),
#                 )
#         except Exception:
#             return np.nan, np.nan
    
#         # Fallback if code is not 1-8
#         return np.nan, np.nan
        
#     @staticmethod
#     def weibull_shape_solver(k, M, V):
#         """
#         Function to find the root of the Weibull shape parameter 'k'.
#         We find 'k' such that the theoretical variance/mean^2 ratio
#         matches the observed V/M^2 ratio.
#         """
#         # Guard against invalid 'k' values during solving
#         if k <= 0:
#             return -np.inf
#         try:
#             g1 = gamma_function(1 + 1/k)
#             g2 = gamma_function(1 + 2/k)
            
#             # This is the V/M^2 ratio *implied by k*
#             implied_v_over_m_sq = (g2 / (g1**2)) - 1
            
#             # This is the *observed* ratio
#             observed_v_over_m_sq = V / (M**2)
            
#             # Return the difference (we want this to be 0)
#             return observed_v_over_m_sq - implied_v_over_m_sq
#         except ValueError:
#             return -np.inf # Handle math errors

#     @staticmethod
#     def calculate_tercile_probabilities_bestfit(best_guess, error_variance, T1, T2, dist_code, dof 
#     ):
#         """
#         Generic tercile probabilities using best-fit family per grid cell.

#         Inputs (per grid cell):
#         - best_guess : 1D array over T (hindcast_det or forecast_det)
#         - T1, T2     : scalar terciles from climatological best-fit distribution
#         - dist_code  : int, as in _ppf_terciles_from_code
#         - shape, loc, scale : scalars from climatology fit

#         Strategy:
#         - For each time step, build a predictive distribution of the same family:
#             * Use best_guess[t] to adjust mean / location;
#             * Keep shape parameters from climatology.
#         - Then compute probabilities:
#             P(B) = F(T1), P(N) = F(T2) - F(T1), P(A) = 1 - F(T2).
#         """
        
#         best_guess = np.asarray(best_guess, float)
#         error_variance = np.asarray(error_variance, dtype=float)
#         # T1 = np.asarray(T1, dtype=float)
#         # T2 = np.asarray(T2, dtype=float)
#         n_time = best_guess.size
#         out = np.full((3, n_time), np.nan, float)

#         if np.all(np.isnan(best_guess)) or np.isnan(dist_code) or np.isnan(T1) or np.isnan(T2) or np.isnan(error_variance):
#             return out

#         code = int(dist_code)

#         # Normal: loc = forecast; scale from clim
#         if code == 1:
#             error_std = np.sqrt(error_variance)
#             out[0, :] = norm.cdf(T1, loc=best_guess, scale=error_std)
#             out[1, :] = norm.cdf(T2, loc=best_guess, scale=error_std) - norm.cdf(T1, loc=best_guess, scale=error_std)
#             out[2, :] = 1 - norm.cdf(T2, loc=best_guess, scale=error_std)

#         # Lognormal: shape = sigma from clim; enforce mean = best_guess
#         elif code == 2:
#             sigma = np.sqrt(np.log(1 + error_variance / (best_guess**2)))
#             mu = np.log(best_guess) - sigma**2 / 2
#             out[0, :] = lognorm.cdf(T1, s=sigma, scale=np.exp(mu))
#             out[1, :] = lognorm.cdf(T2, s=sigma, scale=np.exp(mu)) - lognorm.cdf(T1, s=sigma, scale=np.exp(mu))
#             out[2, :] = 1 - lognorm.cdf(T2, s=sigma, scale=np.exp(mu))      


#         # Exponential: keep scale from clim; shift loc so mean = best_guess
#         elif code == 3:
#             scale = np.sqrt(error_variance)
#             scale = np.where(scale <= 0, np.nan, scale)
        
#             # Exponential mean = loc + scale.
#             # Use loc = best_guess - scale so the predictive mean is best_guess.
#             loc = best_guess - scale
        
#             c1 = expon.cdf(T1, loc=loc, scale=scale)
#             c2 = expon.cdf(T2, loc=loc, scale=scale)
        
#             out[0, :] = c1
#             out[1, :] = c2 - c1
#             out[2, :] = 1.0 - c2

#         # Gamma: use shape from clim; set scale so mean = best_guess
#         elif code == 4:
#             alpha = (best_guess ** 2) / error_variance
#             theta = error_variance / best_guess
#             c1 = gamma.cdf(T1, a=alpha, scale=theta)
#             c2 = gamma.cdf(T2, a=alpha, scale=theta)
#             out[0, :] = c1
#             out[1, :] = c2 - c1
#             out[2, :] = 1.0 - c2

#         elif code == 5: # Assuming 5 is for Weibull   
        
#             for i in range(n_time):
#                 # Get the scalar values for this specific element (e.g., grid cell)
#                 M = best_guess[i]
#                 print(M)
#                 V = error_variance
#                 print(V)
                
#                 # Handle cases with no variance to avoid division by zero
#                 if V <= 0 or M <= 0:
#                     out[0, i] = np.nan
#                     out[1, i] = np.nan
#                     out[2, i] = np.nan
#                     continue # Skip to the next element
        
#                 # --- 1. Numerically solve for shape 'k' ---
#                 # We need a reasonable starting guess. 2.0 is common (Rayleigh dist.)
#                 initial_guess = 2.0
                
#                 # fsolve finds the root of our helper function
#                 k = fsolve(weibull_shape_solver, initial_guess, args=(M, V))[0]
        
#                 # --- 2. Check for bad solution and calculate scale 'lambda' ---
#                 if k <= 0:
#                     # Solver failed
#                     out[0, i] = np.nan
#                     out[1, i] = np.nan
#                     out[2, i] = np.nan
#                     continue
                
#                 # With 'k' found, we can now algebraically find scale 'lambda'
#                 # In scipy.stats, scale is 'scale'
#                 lambda_scale = M / gamma_function(1 + 1/k)
        
#                 # --- 3. Calculate Probabilities ---
#                 # In scipy.stats, shape 'k' is 'c'
#                 # Use the T1 and T2 values for this specific element
                
#                 c1 = weibull_min.cdf(T1, c=k, loc=0, scale=lambda_scale)
#                 c2 = weibull_min.cdf(T2, c=k, loc=0, scale=lambda_scale)
        
#                 out[0, i] = c1
#                 out[1, i] = c2 - c1
#                 out[2, i] = 1.0 - c2

#         # Student-t: df from clim; scale from clim; loc = best_guess
#         elif code == 6:       
#             # Check if df is valid for variance calculation
#             if dof <= 2:
#                 # Cannot calculate scale, fill with NaNs
#                 out[0, :] = np.nan
#                 out[1, :] = np.nan
#                 out[2, :] = np.nan
#             else:
#                 # 1. Calculate t-distribution parameters
#                 # 'loc' (mean) is just the best_guess
#                 loc = best_guess
#                 # 'scale' is calculated from the variance and df
#                 # Variance = scale**2 * (df / (df - 2))
#                 scale = np.sqrt(error_variance * (dof - 2) / dof)
                
#                 # 2. Calculate probabilities
#                 c1 = t.cdf(T1, df=dof, loc=loc, scale=scale)
#                 c2 = t.cdf(T2, df=dof, loc=loc, scale=scale)

#                 out[0, :] = c1
#                 out[1, :] = c2 - c1
#                 out[2, :] = 1.0 - c2

#         elif code == 7: # Assuming 7 is for Poisson
            
#             # --- 1. Set the Poisson parameter 'mu' ---
#             # The 'mu' parameter is the mean.
            
#             # A warning is strongly recommended if error_variance is different from best_guess
#             if not np.allclose(best_guess, error_variance, atol=0.5):
#                 print("Warning: 'error_variance' is not equal to 'best_guess'.")
#                 print("Poisson model assumes mean=variance and is likely inappropriate.")
#                 print("Consider using Negative Binomial.")
            
#             mu = best_guess
        
#             # --- 2. Calculate Probabilities ---
#             # poisson.cdf(k, mu) calculates P(X <= k)
            
#             c1 = poisson.cdf(T1, mu=mu)
#             c2 = poisson.cdf(T2, mu=mu)
            
#             out[0, :] = c1
#             out[1, :] = c2 - c1
#             out[2, :] = 1.0 - c2

#         elif code == 8: # Assuming 8 is for Negative Binomial
            
#             # --- 1. Calculate Negative Binomial Parameters ---
#             # This model is ONLY valid for overdispersion (Variance > Mean).
#             # We will use np.where to set parameters to NaN if V <= M.
            
#             # p = Mean / Variance
#             p = np.where(error_variance > best_guess, 
#                          best_guess / error_variance, 
#                          np.nan)
            
#             # n = Mean^2 / (Variance - Mean)
#             n = np.where(error_variance > best_guess, 
#                          (best_guess**2) / (error_variance - best_guess), 
#                          np.nan)
            
#             # --- 2. Calculate Probabilities ---
#             # The nbinom.cdf function will propagate NaNs, correctly
#             # handling the cases where the model was invalid.
            
#             c1 = nbinom.cdf(T1, n=n, p=p)
#             c2 = nbinom.cdf(T2, n=n, p=p)
            
#             out[0, :] = c1
#             out[1, :] = c2 - c1
#             out[2, :] = 1.0 - c2
            
#         else:
#             raise ValueError(f"Invalid distribution")

#         return out

#     @staticmethod
#     def calculate_tercile_probabilities_nonparametric(best_guess, error_samples, first_tercile, second_tercile):
#         """Non-parametric method using historical error samples."""
#         n_time = len(best_guess)
#         pred_prob = np.full((3, n_time), np.nan, dtype=float)
#         for t in range(n_time):
#             if np.isnan(best_guess[t]):
#                 continue
#             dist = best_guess[t] + error_samples
#             dist = dist[np.isfinite(dist)]
#             if len(dist) == 0:
#                 continue
#             p_below = np.mean(dist < first_tercile)
#             p_between = np.mean((dist >= first_tercile) & (dist < second_tercile))
#             p_above = 1.0 - (p_below + p_between)
#             pred_prob[0, t] = p_below
#             pred_prob[1, t] = p_between
#             pred_prob[2, t] = p_above
#         return pred_prob


#     def compute_prob(
#         self,
#         Predictant: xr.DataArray,
#         clim_year_start,
#         clim_year_end,
#         hindcast_det: xr.DataArray,
#         best_code_da: xr.DataArray = None,
#         best_shape_da: xr.DataArray = None,
#         best_loc_da: xr.DataArray = None,
#         best_scale_da: xr.DataArray = None,
#     ) -> xr.DataArray:
#         """
#         Compute tercile probabilities from deterministic hindcasts.
#         """
    
#         if "M" in Predictant.dims:
#             Predictant = Predictant.isel(M=0).drop_vars("M").squeeze()
    
#         Predictant = Predictant.transpose("T", "Y", "X")
#         hindcast_det = hindcast_det.transpose("T", "Y", "X")
    
#         mask = self._spatial_mask(Predictant)
    
#         clim = Predictant.sel(T=slice(str(clim_year_start), str(clim_year_end)))
#         if clim.sizes.get("T", 0) < 3:
#             raise ValueError("Not enough years in climatology period for terciles.")
    
#         error_variance = (Predictant - hindcast_det).var(dim="T")
#         dof = max(int(clim.sizes["T"]) - 1, 2)
    
#         terciles_emp = clim.quantile([1.0 / 3.0, 2.0 / 3.0], dim="T")
#         T1_emp = terciles_emp.isel(quantile=0).drop_vars("quantile")
#         T2_emp = terciles_emp.isel(quantile=1).drop_vars("quantile")
    
#         dm = self.dist_method
    
#         if dm == "bestfit":
#             if any(v is None for v in (best_code_da, best_shape_da, best_loc_da, best_scale_da)):
#                 raise ValueError(
#                     "dist_method='bestfit' requires best_code_da, best_shape_da, "
#                     "best_loc_da, and best_scale_da."
#                 )
    
#             T1, T2 = xr.apply_ufunc(
#                 self._ppf_terciles_from_code,
#                 best_code_da,
#                 best_shape_da,
#                 best_loc_da,
#                 best_scale_da,
#                 input_core_dims=[(), (), (), ()],
#                 output_core_dims=[(), ()],
#                 vectorize=True,
#                 dask="parallelized",
#                 output_dtypes=[float, float],
#             )
    
#             hindcast_prob = xr.apply_ufunc(
#                 self.calculate_tercile_probabilities_bestfit,
#                 hindcast_det,
#                 error_variance,
#                 T1,
#                 T2,
#                 best_code_da,
#                 input_core_dims=[("T",), (), (), (), ()],
#                 output_core_dims=[("probability", "T")],
#                 vectorize=True,
#                 dask="parallelized",
#                 kwargs={"dof": dof},
#                 output_dtypes=[float],
#                 dask_gufunc_kwargs={
#                     "output_sizes": {"probability": 3},
#                     "allow_rechunk": True,
#                 },
#             )
    
#         elif dm == "nonparam":
#             error_samples = Predictant - hindcast_det
    
#             hindcast_prob = xr.apply_ufunc(
#                 self.calculate_tercile_probabilities_nonparametric,
#                 hindcast_det,
#                 error_samples,
#                 T1_emp,
#                 T2_emp,
#                 input_core_dims=[("T",), ("T",), (), ()],
#                 output_core_dims=[("probability", "T")],
#                 vectorize=True,
#                 dask="parallelized",
#                 output_dtypes=[float],
#                 dask_gufunc_kwargs={
#                     "output_sizes": {"probability": 3},
#                     "allow_rechunk": True,
#                 },
#             )
    
#         else:
#             raise ValueError(f"Invalid dist_method: {self.dist_method}")
    
#         hindcast_prob = hindcast_prob.assign_coords(
#             probability=("probability", ["PB", "PN", "PA"])
#         )
    
#         hindcast_prob = hindcast_prob.transpose("probability", "T", "Y", "X") * mask
#         hindcast_prob = self._normalize_probabilities(hindcast_prob)
    
#         return hindcast_prob