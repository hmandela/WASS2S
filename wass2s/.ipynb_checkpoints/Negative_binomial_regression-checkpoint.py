import numpy as np
import pandas as pd
import xarray as xr

try:
    from dask.distributed import Client
except Exception:
    Client = None

from scipy import stats


# =============================================================================
# Low-level solvers (NumPy-only; Dask-safe)
# =============================================================================

def _add_intercept(X: np.ndarray) -> np.ndarray:
    return np.column_stack([np.ones(X.shape[0], dtype=float), X])


def _wls_solve(X: np.ndarray, z: np.ndarray, w: np.ndarray) -> np.ndarray:
    sw = np.sqrt(np.clip(w, 1e-12, None))
    Xw = X * sw[:, None]
    zw = z * sw
    beta, *_ = np.linalg.lstsq(Xw, zw, rcond=None)
    return beta


def _poisson_irls_beta(y: np.ndarray, X: np.ndarray, max_iter: int = 60, tol: float = 1e-8) -> np.ndarray:
    beta = np.zeros(X.shape[1], dtype=float)
    for _ in range(max_iter):
        eta = np.clip(X @ beta, -700.0, 700.0)
        mu = np.clip(np.exp(eta), 1e-12, None)

        z = eta + (y - mu) / mu
        w = mu

        beta_new = _wls_solve(X, z, w)
        if np.max(np.abs(beta_new - beta)) < tol:
            beta = beta_new
            break
        beta = beta_new
    return beta


def _nb2_irls_beta_alpha(
    y: np.ndarray,
    X: np.ndarray,
    max_iter: int = 80,
    tol: float = 1e-8,
    alpha_init: float = 0.2,
) -> tuple[np.ndarray, float]:
    beta = _poisson_irls_beta(y, X, max_iter=40, tol=tol)
    alpha = max(float(alpha_init), 1e-10)

    for _ in range(max_iter):
        eta = np.clip(X @ beta, -700.0, 700.0)
        mu = np.clip(np.exp(eta), 1e-12, None)

        # NB2 weights
        w = mu / (1.0 + alpha * mu)
        z = eta + (y - mu) / mu

        beta_new = _wls_solve(X, z, w)

        # MoM alpha update from residuals
        resid2 = (y - mu) ** 2
        alpha_raw = np.nanmean((resid2 - y) / (mu ** 2))
        alpha_new = float(np.clip(alpha_raw, 0.0, 1e6))

        if (np.max(np.abs(beta_new - beta)) < tol) and (abs(alpha_new - alpha) < 1e-6):
            beta, alpha = beta_new, max(alpha_new, 1e-10)
            break

        beta, alpha = beta_new, max(alpha_new, 1e-10)

    return beta, float(alpha)


def _logit_irls_coef(y01: np.ndarray, X: np.ndarray, max_iter: int = 60, tol: float = 1e-8) -> np.ndarray:
    beta = np.zeros(X.shape[1], dtype=float)
    for _ in range(max_iter):
        eta = np.clip(X @ beta, -35.0, 35.0)
        p_hat = 1.0 / (1.0 + np.exp(-eta))

        w = np.clip(p_hat * (1.0 - p_hat), 1e-12, None)
        z = eta + (y01 - p_hat) / w

        beta_new = _wls_solve(X, z, w)
        if np.max(np.abs(beta_new - beta)) < tol:
            beta = beta_new
            break
        beta = beta_new
    return beta


def _safe_mask_xy(x: np.ndarray, y: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    m = np.all(np.isfinite(x), axis=1) & np.isfinite(y)
    return x[m], y[m]


def _ct_overdispersion_p_one_sided(y: np.ndarray, X: np.ndarray) -> float:
    """
    Cameron–Trivedi auxiliary regression test (one-sided p for alpha>0) using Poisson IRLS.
    """
    beta_p = _poisson_irls_beta(y, X, max_iter=50)
    mu = np.clip(np.exp(np.clip(X @ beta_p, -700.0, 700.0)), 1e-12, None)

    aux_y = ((y - mu) ** 2 - y) / mu
    aux_X = np.column_stack([np.ones_like(mu), mu])

    b, *_ = np.linalg.lstsq(aux_X, aux_y, rcond=None)
    resid = aux_y - aux_X @ b
    df = max(len(mu) - aux_X.shape[1], 1)
    s2 = float((resid @ resid) / df)

    XtX_inv = np.linalg.pinv(aux_X.T @ aux_X)
    se = np.sqrt(np.diag(s2 * XtX_inv))
    if not np.isfinite(se[1]) or se[1] <= 0:
        return np.nan

    tval = float(b[1] / se[1])
    return float(stats.t.sf(tval, df=df))  # one-sided


# =============================================================================
# Backends (numpy_nb2 vs statsmodels) for validation runs
# =============================================================================

def _fit_predict_poisson_backend(x, y, x_test, y_test, add_intercept=True, backend="numpy_nb2"):
    x = np.asarray(x, float); y = np.asarray(y, float); x_test = np.asarray(x_test, float)
    if x.ndim == 1: x = x[:, None]
    x, y = _safe_mask_xy(x, y)
    if y.size < 5 or np.any(y < 0):
        return np.array([np.nan, np.nan, np.nan], float)

    X = _add_intercept(x) if add_intercept else x

    if backend == "numpy_nb2":
        beta = _poisson_irls_beta(y, X)
    elif backend == "statsmodels":
        import statsmodels.api as sm
        res = sm.GLM(y, X, family=sm.families.Poisson()).fit()
        beta = res.params
    else:
        raise ValueError(f"Unknown backend: {backend}")

    if x_test.ndim == 1: x_test = x_test.reshape(1, -1)
    Xtest = _add_intercept(x_test) if add_intercept else x_test

    mu = np.exp(np.clip(Xtest @ beta, -700.0, 700.0)).squeeze()
    mu = np.maximum(mu, 0.0)
    err = (np.asarray(y_test, float) - mu).squeeze()
    return np.array([err, mu, np.nan], float).squeeze()


def _fit_predict_nb_backend(x, y, x_test, y_test, add_intercept=True, backend="numpy_nb2", alpha_init=0.2):
    x = np.asarray(x, float); y = np.asarray(y, float); x_test = np.asarray(x_test, float)
    if x.ndim == 1: x = x[:, None]
    x, y = _safe_mask_xy(x, y)
    if y.size < 5 or np.any(y < 0):
        return np.array([np.nan, np.nan, np.nan], float)

    X = _add_intercept(x) if add_intercept else x

    if backend == "numpy_nb2":
        beta, alpha = _nb2_irls_beta_alpha(y, X, alpha_init=alpha_init)
    elif backend == "statsmodels":
        import statsmodels.api as sm
        res = sm.GLM(y, X, family=sm.families.NegativeBinomial()).fit()
        beta = res.params
        # MoM alpha (for consistent reporting)
        mu = np.clip(np.exp(np.clip(X @ beta, -700.0, 700.0)), 1e-12, None)
        resid2 = (y - mu) ** 2
        alpha_raw = np.nanmean((resid2 - y) / (mu ** 2))
        alpha = float(np.clip(alpha_raw, 0.0, 1e6))
    else:
        raise ValueError(f"Unknown backend: {backend}")

    if x_test.ndim == 1: x_test = x_test.reshape(1, -1)
    Xtest = _add_intercept(x_test) if add_intercept else x_test
    mu = np.exp(np.clip(Xtest @ beta, -700.0, 700.0)).squeeze()
    mu = np.clip(mu, 0.0, None)

    err = (np.asarray(y_test, float) - mu).squeeze()
    return np.array([err, mu, float(alpha)], float).squeeze()


def _fit_predict_zinb_backend(x, y, x_test, y_test, add_intercept=True, backend="numpy_nb2", alpha_init=0.2):
    x = np.asarray(x, float); y = np.asarray(y, float); x_test = np.asarray(x_test, float)
    if x.ndim == 1: x = x[:, None]
    x, y = _safe_mask_xy(x, y)
    if y.size < 8 or np.any(y < 0):
        return np.array([np.nan, np.nan, np.nan, np.nan], float)

    X = _add_intercept(x) if add_intercept else x

    y0 = (y == 0.0).astype(float)
    gamma_ = _logit_irls_coef(y0, X)

    _, mu_test, alpha = _fit_predict_nb_backend(
        x, y, x_test, y_test, add_intercept=add_intercept, backend=backend, alpha_init=alpha_init
    )

    if x_test.ndim == 1: x_test = x_test.reshape(1, -1)
    Xtest = _add_intercept(x_test) if add_intercept else x_test

    pi = (1.0 / (1.0 + np.exp(-np.clip(Xtest @ gamma_, -35.0, 35.0)))).squeeze()
    pi = np.clip(pi, 0.0, 1.0)

    yhat = (1.0 - pi) * mu_test
    err = (np.asarray(y_test, float) - yhat).squeeze()
    return np.array([err, yhat, float(alpha), float(pi)], float).squeeze()


def _fit_predict_hurdle_nb_backend(x, y, x_test, y_test, add_intercept=True, backend="numpy_nb2", alpha_init=0.2):
    x = np.asarray(x, float); y = np.asarray(y, float); x_test = np.asarray(x_test, float)
    if x.ndim == 1: x = x[:, None]
    x, y = _safe_mask_xy(x, y)
    if y.size < 8 or np.any(y < 0):
        return np.array([np.nan, np.nan, np.nan, np.nan, np.nan], float)

    X = _add_intercept(x) if add_intercept else x

    y_pos = (y > 0.0).astype(float)
    gamma_ = _logit_irls_coef(y_pos, X)

    mpos = y > 0.0
    if np.sum(mpos) < 5:
        return np.array([np.nan, np.nan, np.nan, np.nan, np.nan], float)

    xp = x[mpos, :]
    yp = y[mpos]

    _, mu, alpha = _fit_predict_nb_backend(
        xp, yp, x_test, y_test, add_intercept=add_intercept, backend=backend, alpha_init=alpha_init
    )

    if x_test.ndim == 1: x_test = x_test.reshape(1, -1)
    Xtest = _add_intercept(x_test) if add_intercept else x_test

    pplus = (1.0 / (1.0 + np.exp(-np.clip(Xtest @ gamma_, -35.0, 35.0)))).squeeze()
    pplus = np.clip(pplus, 0.0, 1.0)

    alpha = max(float(alpha), 1e-10)
    r = 1.0 / alpha
    p = r / (r + mu + 1e-12)
    P0 = float(np.power(p, r))

    trunc_mean = float(mu / np.clip(1.0 - P0, 1e-12, None))
    yhat = float(pplus * trunc_mean)

    err = (np.asarray(y_test, float) - yhat).squeeze()
    return np.array([err, yhat, float(alpha), float(pplus), float(P0)], float).squeeze()


# =============================================================================
# Mixin: reuse your existing tercile probability machinery
# =============================================================================

class _WAS_CountProbMixin:
    """
    This mixin intentionally expects you already have the following static methods
    implemented exactly as in your WAS_PoissonRegression:

      - _ppf_terciles_from_code(dist_code, shape, loc, scale)
      - calculate_tercile_probabilities_bestfit(best_guess, error_variance, T1, T2, dist_code, dof)
      - calculate_tercile_probabilities_nonparametric(best_guess, error_samples, first_tercile, second_tercile)

    If you want this class fully standalone, paste those methods here.
    """
    @staticmethod
    def _ppf_terciles_from_code(dist_code, shape, loc, scale):
        raise NotImplementedError("Paste your WAS_PoissonRegression._ppf_terciles_from_code here.")

    @staticmethod
    def calculate_tercile_probabilities_bestfit(best_guess, error_variance, T1, T2, dist_code, dof):
        raise NotImplementedError("Paste your WAS_PoissonRegression.calculate_tercile_probabilities_bestfit here.")

    @staticmethod
    def calculate_tercile_probabilities_nonparametric(best_guess, error_samples, first_tercile, second_tercile):
        raise NotImplementedError("Paste your WAS_PoissonRegression.calculate_tercile_probabilities_nonparametric here.")


# =============================================================================
# Adapted class: same signature/behavior style as your WAS_PoissonRegression
# =============================================================================

class WAS_CountModel_AutoGate(_WAS_CountProbMixin):
    """
    Drop-in count model for your pipeline with:
      - deterministic auto selection per grid cell: Poisson / NB2 / ZINB / Hurdle-NB
      - return_params=True: alpha, pi, pplus, P0, model_code maps
      - backend switch: WAS_count_fit_backend="numpy_nb2" | "statsmodels"
      - compute_prob() and forecast() signatures consistent with WAS_PoissonRegression
    """

    MODEL_CODES = {"poisson": 1, "nb2": 2, "zinb": 3, "hurdle_nb": 4}

    def __init__(
        self,
        nb_cores=1,
        dist_method="nonparam",
        WAS_count_fit_backend="numpy_nb2",
        add_intercept=True,
        alpha_init=0.2,
        gate_alpha=0.05,
        pearson_dispersion_threshold=1.2,
        zero_inflation_threshold=0.35,
        hurdle_threshold=0.45,
    ):
        self.nb_cores = nb_cores
        self.dist_method = dist_method
        self.WAS_count_fit_backend = WAS_count_fit_backend
        self.add_intercept = add_intercept
        self.alpha_init = alpha_init

        self.gate_alpha = gate_alpha
        self.pearson_dispersion_threshold = pearson_dispersion_threshold
        self.zero_inflation_threshold = zero_inflation_threshold
        self.hurdle_threshold = hurdle_threshold

    # ---------------- Gate ----------------

    def _gate_model_choice_1d(self, y: np.ndarray, x: np.ndarray) -> float:
        if x.ndim == 1:
            x = x[:, None]
        x, y = _safe_mask_xy(x, y)
        n = y.size
        if n < 8 or np.any(y < 0):
            return np.nan

        X = _add_intercept(x) if self.add_intercept else x

        frac0 = float(np.mean(y == 0.0))

        beta_p = _poisson_irls_beta(y, X)
        mu = np.clip(np.exp(np.clip(X @ beta_p, -700.0, 700.0)), 1e-12, None)
        df_resid = max(n - X.shape[1], 1)
        pearson_chi2 = float(np.sum((y - mu) ** 2 / mu))
        pearson_disp = pearson_chi2 / df_resid

        ct_p = _ct_overdispersion_p_one_sided(y, X)
        overdispersed = (pearson_disp >= self.pearson_dispersion_threshold) or (np.isfinite(ct_p) and ct_p < self.gate_alpha)

        # Model selection
        if frac0 >= self.zero_inflation_threshold:
            if frac0 >= self.hurdle_threshold:
                return float(self.MODEL_CODES["hurdle_nb"] if overdispersed else self.MODEL_CODES["poisson"])
            return float(self.MODEL_CODES["zinb"] if overdispersed else self.MODEL_CODES["poisson"])

        return float(self.MODEL_CODES["nb2"] if overdispersed else self.MODEL_CODES["poisson"])

    # ---------------- Fit/predict core ----------------

    def fit_predict(self, x, y, x_test, y_test, return_params=False):
        x = np.asarray(x, float); y = np.asarray(y, float); x_test = np.asarray(x_test, float)
        if x.ndim == 1: x = x[:, None]
        x, y = _safe_mask_xy(x, y)

        if y.size < 8 or np.any(y < 0):
            if return_params:
                return np.array([np.nan, np.nan, np.nan, np.nan, np.nan, np.nan, np.nan], float)
            return np.array([np.nan, np.nan], float)

        model_code = self._gate_model_choice_1d(y, x)
        backend = self.WAS_count_fit_backend

        alpha = np.nan; pi = np.nan; pplus = np.nan; P0 = np.nan

        if model_code == self.MODEL_CODES["poisson"]:
            err, pred, _ = _fit_predict_poisson_backend(
                x, y, x_test, y_test, add_intercept=self.add_intercept, backend=backend
            )
        elif model_code == self.MODEL_CODES["nb2"]:
            err, pred, alpha = _fit_predict_nb_backend(
                x, y, x_test, y_test, add_intercept=self.add_intercept, backend=backend, alpha_init=self.alpha_init
            )
        elif model_code == self.MODEL_CODES["zinb"]:
            err, pred, alpha, pi = _fit_predict_zinb_backend(
                x, y, x_test, y_test, add_intercept=self.add_intercept, backend=backend, alpha_init=self.alpha_init
            )
        elif model_code == self.MODEL_CODES["hurdle_nb"]:
            err, pred, alpha, pplus, P0 = _fit_predict_hurdle_nb_backend(
                x, y, x_test, y_test, add_intercept=self.add_intercept, backend=backend, alpha_init=self.alpha_init
            )
        else:
            err, pred = np.nan, np.nan

        if return_params:
            return np.array([err, pred, alpha, pi, pplus, P0, float(model_code)], float).squeeze()

        return np.array([err, pred], float).squeeze()

    # ---------------- Deterministic compute_model (as in your Poisson class) ----------------

    def compute_model(self, X_train, y_train, X_test, y_test, return_params=False):
        chunksize_x = int(np.round(len(y_train.get_index("X")) / self.nb_cores))
        chunksize_y = int(np.round(len(y_train.get_index("Y")) / self.nb_cores))

        X_train["T"] = y_train["T"]
        y_train = y_train.transpose("T", "Y", "X")

        X_test = X_test.squeeze()
        if "T" in y_test.dims:
            y_test = y_test.drop_vars("T")
        y_test = y_test.squeeze().transpose("Y", "X")

        client = None
        if Client is not None and self.nb_cores and self.nb_cores > 1:
            client = Client(n_workers=self.nb_cores, threads_per_worker=1)

        out_size = 7 if return_params else 2

        result = xr.apply_ufunc(
            self.fit_predict,
            X_train,
            y_train.chunk({"Y": chunksize_y, "X": chunksize_x}),
            X_test,
            y_test.chunk({"Y": chunksize_y, "X": chunksize_x}),
            kwargs={"return_params": return_params},
            input_core_dims=[("T", "features"), ("T",), ("features",), ()],
            vectorize=True,
            dask="parallelized",
            output_core_dims=[("output",)],
            output_dtypes=["float"],
            dask_gufunc_kwargs={"output_sizes": {"output": out_size}},
        )

        result_ = result.compute()
        if client is not None:
            client.close()

        if not return_params:
            return result_.isel(output=1)  # prediction (Y,X)

        return xr.Dataset(
            {
                "prediction": result_.isel(output=1),
                "alpha": result_.isel(output=2),
                "pi": result_.isel(output=3),
                "pplus": result_.isel(output=4),
                "P0": result_.isel(output=5),
                "model_code": result_.isel(output=6),
            }
        )

    # ---------------- compute_prob (same signature as your Poisson class) ----------------

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

        if "M" in Predictant.dims:
            Predictant = Predictant.isel(M=0).drop_vars("M").squeeze()

        Predictant = Predictant.transpose("T", "Y", "X")
        mask = xr.where(~np.isnan(Predictant.isel(T=0)), 1.0, np.nan)

        clim = Predictant.sel(T=slice(str(clim_year_start), str(clim_year_end)))
        if clim.sizes.get("T", 0) < 3:
            raise ValueError("Not enough years in climatology period for terciles.")

        error_variance = (Predictant - hindcast_det).var(dim="T")
        dof = max(int(clim.sizes["T"]) - 1, 2)

        terciles_emp = clim.quantile([0.32, 0.67], dim="T")
        T1_emp = terciles_emp.isel(quantile=0).drop_vars("quantile")
        T2_emp = terciles_emp.isel(quantile=1).drop_vars("quantile")

        dm = self.dist_method

        if dm == "bestfit":
            if any(v is None for v in (best_code_da, best_shape_da, best_loc_da, best_scale_da)):
                raise ValueError("dist_method='bestfit' requires best_code_da, best_shape_da, best_loc_da, best_scale_da.")

            T1, T2 = xr.apply_ufunc(
                self._ppf_terciles_from_code,
                best_code_da, best_shape_da, best_loc_da, best_scale_da,
                input_core_dims=[(), (), (), ()],
                output_core_dims=[(), ()],
                vectorize=True,
                dask="parallelized",
                output_dtypes=[float, float],
            )

            hindcast_prob = xr.apply_ufunc(
                self.calculate_tercile_probabilities_bestfit,
                hindcast_det,
                error_variance,
                T1, T2,
                best_code_da,
                input_core_dims=[("T",), (), (), (), ()],
                output_core_dims=[("probability", "T")],
                vectorize=True,
                kwargs={"dof": dof},
                dask="parallelized",
                output_dtypes=[float],
                dask_gufunc_kwargs={"output_sizes": {"probability": 3}, "allow_rechunk": True},
            )

        elif dm == "nonparam":
            error_samples = Predictant - hindcast_det
            hindcast_prob = xr.apply_ufunc(
                self.calculate_tercile_probabilities_nonparametric,
                hindcast_det,
                error_samples,
                T1_emp, T2_emp,
                input_core_dims=[("T",), ("T",), (), ()],
                output_core_dims=[("probability", "T")],
                vectorize=True,
                dask="parallelized",
                output_dtypes=[float],
                dask_gufunc_kwargs={"output_sizes": {"probability": 3}, "allow_rechunk": True},
            )
        else:
            raise ValueError(f"Invalid dist_method: {self.dist_method}")

        hindcast_prob = hindcast_prob.assign_coords(probability=("probability", ["PB", "PN", "PA"]))
        return (hindcast_prob * mask).transpose("probability", "T", "Y", "X")

    # ---------------- forecast (same signature as your Poisson class) ----------------

    def forecast(
        self,
        Predictant,
        clim_year_start,
        clim_year_end,
        Predictor,
        hindcast_det,
        Predictor_for_year,
        best_code_da=None,
        best_shape_da=None,
        best_loc_da=None,
        best_scale_da=None,
        return_params=False,
    ):
        """
        Returns
        -------
        forecast_expanded : xr.DataArray
            dims (T=1, Y, X) deterministic forecast
        forecast_prob : xr.DataArray
            dims (probability, T=1, Y, X)
        (optional) forecast_params : xr.Dataset
            if return_params=True, includes alpha/pi/pplus/P0/model_code for the forecast step
        """
        chunksize_x = int(np.round(len(Predictant.get_index("X")) / self.nb_cores))
        chunksize_y = int(np.round(len(Predictant.get_index("Y")) / self.nb_cores))

        Predictor["T"] = Predictant["T"]
        Predictant = Predictant.transpose("T", "Y", "X")

        Predictor_for_year_ = Predictor_for_year.squeeze()
        y_test = xr.full_like(Predictant.isel(T=0), np.nan)

        client = None
        if Client is not None and self.nb_cores and self.nb_cores > 1:
            client = Client(n_workers=self.nb_cores, threads_per_worker=1)

        out_size = 7 if return_params else 2

        result = xr.apply_ufunc(
            self.fit_predict,
            Predictor,
            Predictant.chunk({"Y": chunksize_y, "X": chunksize_x}),
            Predictor_for_year_,
            y_test.chunk({"Y": chunksize_y, "X": chunksize_x}),
            kwargs={"return_params": return_params},
            input_core_dims=[("T", "features"), ("T",), ("features",), ()],
            vectorize=True,
            dask="parallelized",
            output_core_dims=[("output",)],
            output_dtypes=["float"],
            dask_gufunc_kwargs={"output_sizes": {"output": out_size}},
        )

        result_ = result.compute()
        if client is not None:
            client.close()

        pred_map = result_.isel(output=1)

        # Build a 1-step time coordinate consistent with your original logic
        year = Predictor_for_year.coords["T"].values[0].astype("datetime64[Y]").astype(int) + 1970
        T0 = Predictant.isel(T=0).coords["T"].values
        month_1 = T0.astype("datetime64[M]").astype(int) % 12 + 1
        new_T_value = np.datetime64(f"{year}-{month_1:02d}-01")

        forecast_expanded = pred_map.expand_dims(T=[new_T_value])
        forecast_expanded["T"] = forecast_expanded["T"].astype("datetime64[ns]")

        # For probabilities
        rainfall_for_tercile = Predictant.sel(T=slice(str(clim_year_start), str(clim_year_end)))
        terciles = rainfall_for_tercile.quantile([0.32, 0.67], dim="T")
        T1_emp = terciles.isel(quantile=0).drop_vars("quantile")
        T2_emp = terciles.isel(quantile=1).drop_vars("quantile")

        error_variance = (Predictant - hindcast_det).var(dim="T")
        dof = max(int(rainfall_for_tercile.sizes["T"]) - 1, 2)

        dm = self.dist_method

        if dm == "bestfit":
            if any(v is None for v in (best_code_da, best_shape_da, best_loc_da, best_scale_da)):
                raise ValueError("dist_method='bestfit' requires best_code_da, best_shape_da, best_loc_da, best_scale_da.")

            T1, T2 = xr.apply_ufunc(
                self._ppf_terciles_from_code,
                best_code_da, best_shape_da, best_loc_da, best_scale_da,
                input_core_dims=[(), (), (), ()],
                output_core_dims=[(), ()],
                vectorize=True,
                dask="parallelized",
                output_dtypes=[float, float],
            )

            forecast_prob = xr.apply_ufunc(
                self.calculate_tercile_probabilities_bestfit,
                forecast_expanded,
                error_variance,
                T1, T2,
                best_code_da,
                input_core_dims=[("T",), (), (), (), ()],
                output_core_dims=[("probability", "T")],
                vectorize=True,
                dask="parallelized",
                kwargs={"dof": dof},
                output_dtypes=[float],
                dask_gufunc_kwargs={"output_sizes": {"probability": 3}, "allow_rechunk": True},
            )

        elif dm == "nonparam":
            error_samples = Predictant - hindcast_det
            forecast_prob = xr.apply_ufunc(
                self.calculate_tercile_probabilities_nonparametric,
                forecast_expanded,
                error_samples,
                T1_emp, T2_emp,
                input_core_dims=[("T",), ("T",), (), ()],
                output_core_dims=[("probability", "T")],
                vectorize=True,
                dask="parallelized",
                output_dtypes=[float],
                dask_gufunc_kwargs={"output_sizes": {"probability": 3}, "allow_rechunk": True},
            )
        else:
            raise ValueError(f"Invalid dist_method: {self.dist_method}")

        forecast_prob = forecast_prob.assign_coords(probability=("probability", ["PB", "PN", "PA"]))
        forecast_prob = forecast_prob.transpose("probability", "T", "Y", "X")

        if not return_params:
            return forecast_expanded, forecast_prob

        params = xr.Dataset(
            {
                "alpha": result_.isel(output=2),
                "pi": result_.isel(output=3),
                "pplus": result_.isel(output=4),
                "P0": result_.isel(output=5),
                "model_code": result_.isel(output=6),
            }
        )
        return forecast_expanded, forecast_prob, params


model = WAS_CountModel_AutoGate(
    nb_cores=8,
    dist_method="bestfit",                 # or "nonparam"
    WAS_count_fit_backend="numpy_nb2",     # switch to "statsmodels" for validation runs
)

# deterministic hindcast (example; depends on how you compute your hindcast_det)
hind_det = ...  # (T,Y,X)

hind_prob = model.compute_prob(Predictant, 1991, 2020, hind_det,
                               best_code_da=best_code_da,
                               best_shape_da=best_shape_da,
                               best_loc_da=best_loc_da,
                               best_scale_da=best_scale_da)

fcst_det, fcst_prob, fcst_params = model.forecast(
    Predictant, 1991, 2020,
    Predictor, hind_det,
    Predictor_for_year,
    best_code_da=best_code_da,
    best_shape_da=best_shape_da,
    best_loc_da=best_loc_da,
    best_scale_da=best_scale_da,
    return_params=True,
)

"""
auto = WAS_CountModel_AutoGate(
    nb_cores=8,
    WAS_count_fit_backend="numpy_nb2",   # switch to "statsmodels" for validation
    pearson_dispersion_threshold=1.2,
    gate_alpha=0.05,
    zero_inflation_threshold=0.35,
    hurdle_threshold=0.45,               # more conservative for hurdle
)

# Hindcast deterministic + parameters
ds_hind = auto.compute_model(X_train=Predictor, y_train=Predictant, X_test=X_test, y_test=y_test, return_params=True)
hind_pred = ds_hind["prediction"]

# One-year forecast deterministic + parameters
ds_fcst = auto.forecast(Predictor=Predictor, Predictant=Predictant, Predictor_for_year=Predictor_for_year, return_params=True)
"""