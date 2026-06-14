"""Kernel CCA variants for seasonal forecasting.

Built on the cca-zoo library, these classes extend
:class:`~wass2s.was_cca.WAS_CCA_base` with non-linear kernels.

Classes
-------
WAS_KernelCCA_base
    Abstract base implementing preprocessing, probabilistic post-processing,
    and Optuna / grid hyperparameter search for any cca-zoo kernel method.
WAS_KCCA
    Kernel CCA (cca-zoo ``KCCA`` / ``LinearCCA`` for the linear kernel).
WAS_KGCCA
    Kernel Generalised CCA (cca-zoo ``KGCCA`` / ``GCCA``).
WAS_EOF_KCCA
    EOF pre-reduction layer on top of ``WAS_KCCA``: reduces each spatial
    predictor field to its leading principal components before applying
    kernel CCA.
"""
# =============================================================================
# WAS_KCCA / WAS_KGCCA
# -----------------------------------------------------------------------------
# Kernel CCA (KCCA) and Kernel Generalized CCA (KGCCA) seasonal-forecast models
# =============================================================================


import numpy as np
import xarray as xr
import matplotlib.pyplot as plt
from scipy.stats import (norm, lognorm, expon, gamma, weibull_min,
                       t, poisson, nbinom)
from scipy.special import gamma as gamma_function
from scipy.optimize import fsolve
from wass2s.utils import * 
from wass2s.was_eof import *


try:
    from cca_zoo.linear import CCA as LinearCCA, GCCA
    from cca_zoo.nonparametric import KCCA, KGCCA
    from cca_zoo.model_selection import GridSearchCV as CCA_GridSearch
    from sklearn.model_selection import KFold
    HAS_CCA_ZOO = True
except ImportError:
    HAS_CCA_ZOO = False
    print("Warning: cca-zoo not installed. CCA/KCCA/KGCCA functionality disabled.")


class WAS_KernelCCA_base:
    """
    Shared base for kernel CCA seasonal-forecast models (KCCA, KGCCA).

    Subclasses only need to implement the model-construction hooks:
        _build_model(params)      -> a fitted-ready cca-zoo estimator
        _grid_estimator()         -> estimator used by GridSearchCV
        _prepare_param_grid()     -> dict for grid search
        _objective(trial, X, Y)   -> float for Optuna (optional)
        _is_linear_kernel()       -> whether interpretable feature weights exist

    Conventions identical to WAS_CCA_base:
        WAS dims:  T, Y, X
        Predictand standardized before CCA, returned in original scale.
        Missing values filled with training-period means (no leakage).
    """

    # Tercile quantiles -- single source of truth (same values as WAS_CCA_base).
    Q_LOW = 0.33
    Q_HIGH = 0.67

    def __init__(
        self,
        n_modes=4,
        kernel="linear",
        dist_method="nonparam",
        standardize=True,
        cv_folds=3,
        random_state=42,
        search_method="grid",
        reg=1e-6,
    ):
        self.n_modes = n_modes
        self.kernel = kernel
        self.dist_method = dist_method
        self.standardize = standardize
        self.cv_folds = cv_folds
        self.random_state = random_state
        self.search_method = search_method
        self.reg = reg  # ridge term for the scores -> field regression

        # Hyperparameters (tuned once, reused everywhere).
        self.best_params = None
        self.is_tuned = False

        # Model + fitted state.
        self.model = None
        self.is_fitted = False

        # Stored training artefacts.
        self.X_coords = None
        self.Y_coords = None
        self.train_time = None
        self.X_train_np = None
        self.Y_train_np = None
        self.zx_train = None
        self.zy_train = None
        self._reg_B = None       # (n_modes, n_features_Y)
        self._reg_mean = None    # (n_features_Y,)
        self.canonical_correlations = None

        # Predictor (X) standardization stats, learned on the TRAIN fold so the
        # same per-feature scaling is applied at fit AND predict. This keeps the
        # RBF/poly kernel scale consistent with compute_hyperparameters (which
        # also standardizes X when self.standardize is True). Gated by
        # self.standardize; for the linear kernel CCA is scale-invariant so this
        # is immaterial, but it removes the tune/fit mismatch for non-linear kernels.
        self._x_mean = None      # (Y, X)
        self._x_std = None       # (Y, X)

    # ------------------------------------------------------------------ #
    # Hooks to be overridden by subclasses
    # ------------------------------------------------------------------ #

    def _build_model(self, params):
        raise NotImplementedError

    def _grid_estimator(self):
        raise NotImplementedError

    def _prepare_param_grid(self, kernel=None):
        raise NotImplementedError

    def _objective(self, trial, X_np, Y_np):
        raise NotImplementedError

    def _is_linear_kernel(self):
        return self.kernel == "linear"

    # ------------------------------------------------------------------ #
    # Internal utilities (mirrors WAS_CCA_base)
    # ------------------------------------------------------------------ #

    @staticmethod
    def _drop_member_dim(da: "xr.DataArray") -> "xr.DataArray":
        if "M" in da.dims:
            da = da.isel(M=0, drop=True)
        return da

    @staticmethod
    def _ensure_tyx(da: "xr.DataArray", name="data") -> "xr.DataArray":
        missing = [d for d in ("T", "Y", "X") if d not in da.dims]
        if missing:
            raise ValueError(f"{name} must contain dimensions T, Y, X. Missing: {missing}")
        return da.transpose("T", "Y", "X")

    @staticmethod
    def _fill_train_data(da: "xr.DataArray") -> "xr.DataArray":
        mean_da = da.mean(dim="T", skipna=True)
        return da.fillna(mean_da).fillna(0)

    @staticmethod
    def _fill_test_data(test_da: "xr.DataArray", train_da: "xr.DataArray") -> "xr.DataArray":
        train_mean = train_da.mean(dim="T", skipna=True)
        return test_da.fillna(train_mean).fillna(0)

    @staticmethod
    def _spatial_mask(Predictant: "xr.DataArray") -> "xr.DataArray":
        return xr.where(np.isfinite(Predictant.isel(T=0)), 1.0, np.nan)

    @staticmethod
    def _normalize_probabilities(prob: "xr.DataArray") -> "xr.DataArray":
        prob = prob.clip(min=0, max=1)
        total = prob.sum(dim="probability", skipna=True)
        prob = prob / xr.where(total > 0, total, np.nan)
        return prob

    @staticmethod
    def _make_forecast_time(Predictant: "xr.DataArray", Predictor_for_year: "xr.DataArray"):
        forecast_year = (
            Predictor_for_year.coords["T"].values.astype("datetime64[Y]").astype(int)[0]
            + 1970
        )
        first_target_time = Predictant.isel(T=0).coords["T"].values
        target_month = first_target_time.astype("datetime64[M]").astype(int) % 12 + 1
        new_time = np.datetime64(f"{forecast_year}-{target_month:02d}-01")
        return np.array([new_time], dtype="datetime64[ns]")


    @staticmethod
    def _stack_spatial(da: "xr.DataArray"):
        """(T, Y, X) DataArray -> (T, feature) ndarray + feature coords."""
        if "X" in da.dims and "Y" in da.dims:
            da = da.rename({"X": "lon", "Y": "lat"})
        stacked = da.stack(feature=["lat", "lon"])
        stacked = stacked.transpose("T", "feature")
        return stacked.values, stacked.coords["feature"]

    @staticmethod
    def _unstack_spatial(data_np, feature_coords, time_coords, time_name="T"):
        """(n, feature) ndarray -> (T, Y, X) DataArray."""
        if data_np.ndim == 1:
            data_np = data_np.reshape(1, -1)
        da = xr.DataArray(
            data_np,
            dims=[time_name, "feature"],
            coords={time_name: time_coords, "feature": feature_coords},
        )
        da = da.unstack("feature")
        return da.rename({"lon": "X", "lat": "Y"})

    # ---- training-fold standardization (leakage-free, used in CV) ----

    @staticmethod
    def _standardize_along_T(da: "xr.DataArray"):
        mean = da.mean(dim="T", skipna=True)
        std = da.std(dim="T", skipna=True)
        std = xr.where(std > 0, std, 1.0)
        return (da - mean) / std, mean, std

    @staticmethod
    def _reverse_along_T(da: "xr.DataArray", mean: "xr.DataArray", std: "xr.DataArray"):
        return da * std + mean

    # ------------------------------------------------------------------ #
    # Preprocessing
    # ------------------------------------------------------------------ #

    def preprocess_data(self, X_train: "xr.DataArray", y_train: "xr.DataArray"):
        X_train = self._drop_member_dim(X_train)
        y_train = self._drop_member_dim(y_train)

        X_train = self._ensure_tyx(X_train, name="X_train")
        y_train = self._ensure_tyx(y_train, name="y_train")

        X_train = self._fill_train_data(X_train)
        y_train = self._fill_train_data(y_train)
        return X_train, y_train

    def preprocess_test_data(self, X_test, y_test, X_train, y_train):
        X_test = self._drop_member_dim(X_test)
        y_test = self._drop_member_dim(y_test)
        X_train = self._drop_member_dim(X_train)
        y_train = self._drop_member_dim(y_train)

        X_test = self._ensure_tyx(X_test, name="X_test")
        y_test = self._ensure_tyx(y_test, name="y_test")
        X_train = self._ensure_tyx(X_train, name="X_train")
        y_train = self._ensure_tyx(y_train, name="y_train")

        X_test = self._fill_test_data(X_test, X_train)
        y_test = self._fill_test_data(y_test, y_train)
        return X_test, y_test

    def preprocess_forecast_data(self, Predictor_for_year, Predictor_train):
        Predictor_for_year = self._drop_member_dim(Predictor_for_year)
        Predictor_train = self._drop_member_dim(Predictor_train)

        Predictor_for_year = self._ensure_tyx(Predictor_for_year, name="Predictor_for_year")
        Predictor_train = self._ensure_tyx(Predictor_train, name="Predictor_train")

        Predictor_for_year = self._fill_test_data(Predictor_for_year, Predictor_train)
        return Predictor_for_year

    # ------------------------------------------------------------------ #
    # Hyperparameter tuning (do this once)
    # ------------------------------------------------------------------ #

    def compute_hyperparameters(self, Predictors, Predictand,
                                clim_year_start=None, clim_year_end=None):
        """Find best kernel hyperparameters; stores them in self.best_params."""
        if not HAS_CCA_ZOO:
            raise ImportError("cca-zoo is required for hyperparameter tuning.")

        if self._is_linear_kernel() and self.search_method != "grid":
            print("Linear kernel has limited / no hyperparameters to tune.")

        print(f"Computing hyperparameters for {self.kernel} "
              f"{type(self).__name__} using {self.search_method} search...")

        X_train_final, y_train_final = self.preprocess_data(Predictors, Predictand)
        if self.standardize:
            X_train_final, _, _ = self._standardize_along_T(X_train_final)
            y_train_final, _, _ = self._standardize_along_T(y_train_final)

        X_np, _ = self._stack_spatial(X_train_final)
        Y_np, _ = self._stack_spatial(y_train_final)

        good = ~(np.any(~np.isfinite(X_np), axis=1) | np.any(~np.isfinite(Y_np), axis=1))
        X_np, Y_np = X_np[good], Y_np[good]

        if self.search_method == "grid":
            param_grid = self._prepare_param_grid()
            if not param_grid:
                self.best_params, self.is_tuned = {}, True
                return {}
            print(f"Performing grid search with parameters: {param_grid}")
            cv = KFold(n_splits=self.cv_folds, shuffle=True, random_state=self.random_state)
            grid = CCA_GridSearch(self._grid_estimator(), param_grid=param_grid,
                                  cv=cv, n_jobs=-1, verbose=1)
            grid.fit([X_np, Y_np])
            self.best_params = grid.best_params_
            print(f"Best parameters found: {self.best_params}")
            print(f"Best cross-validation score: {grid.best_score_:.4f}")

        elif self.search_method == "bayesian":
            try:
                import optuna
                from optuna.samplers import TPESampler
            except ImportError:
                raise ImportError("Optuna is required for Bayesian optimization.")
            print("Performing Bayesian optimization with Optuna...")
            study = optuna.create_study(
                direction="maximize", sampler=TPESampler(seed=self.random_state)
            )
            study.optimize(lambda tr: self._objective(tr, X_np, Y_np), n_trials=50)
            self.best_params = study.best_params
            print(f"Best parameters found: {self.best_params}")
            print(f"Best cross-validation score: {study.best_value:.4f}")

        else:
            print("Using default parameters (no search performed).")
            self.best_params = {}

        self.is_tuned = True
        return self.best_params

    def set_hyperparameters(self, hyperparams):
        self.best_params = hyperparams
        self.is_tuned = True
        print(f"Hyperparameters set: {hyperparams}")

    # ------------------------------------------------------------------ #
    # Model fitting + field prediction
    # ------------------------------------------------------------------ #

    def fit_cca(self, X_train, y_train, best_params=None):
        """
        Fit the kernel CCA model and build the (X canonical scores -> Y field)
        regression used for deterministic prediction.

        `y_train` is used *as given* as the regression target, so callers that
        want predictions in standardized space should pass a standardized
        predictand (see compute_model / forecast).
        """
        if not HAS_CCA_ZOO:
            raise ImportError("cca-zoo is required.")

        if best_params is not None:
            params = best_params
        elif self.best_params is not None:
            params = self.best_params
        else:
            params = {}
            if not self._is_linear_kernel():
                print("Warning: no hyperparameters provided. Using defaults.")

        X_train_final, y_train_final = self.preprocess_data(X_train, y_train)

        # Standardize the predictor on this TRAIN fold (per-feature, along T) and
        # remember the stats so _predict_field applies the SAME scaling to the
        # test/forecast predictor. Mirrors compute_hyperparameters so the kernel
        # bandwidth is tuned and fitted on the same X scale. Leakage-free: stats
        # come from the training fold only.
        if self.standardize:
            X_train_final, self._x_mean, self._x_std = self._standardize_along_T(X_train_final)
        else:
            self._x_mean, self._x_std = None, None

        X_np, X_coords = self._stack_spatial(X_train_final)
        Y_np, Y_coords = self._stack_spatial(y_train_final)

        self.X_coords = X_coords
        self.Y_coords = Y_coords
        self.train_time = X_train_final.coords["T"]
        self.X_train_np = X_np
        self.Y_train_np = Y_np

        self.model = self._build_model(params)
        self.model.fit([X_np, Y_np])
        self.is_fitted = True

        # Canonical scores on the training data (both views present).
        scores = self.model.transform([X_np, Y_np])
        self.zx_train, self.zy_train = scores[0], scores[1]

        # Canonical correlations (cca-zoo kernel models expose no direct method).
        self.canonical_correlations = [
            float(np.corrcoef(self.zx_train[:, i], self.zy_train[:, i])[0, 1])
            for i in range(self.n_modes)
        ]

        # Regression: X canonical scores -> Y field (ridge-stabilized).
        self._reg_mean = Y_np.mean(axis=0)
        Yc = Y_np - self._reg_mean
        Z = self.zx_train
        G = Z.T @ Z + self.reg * np.eye(Z.shape[1])
        self._reg_B = np.linalg.solve(G, Z.T @ Yc)
        return self

    def _x_scores(self, X_np):
        """Transform predictor matrix to X canonical scores (single view)."""
        n = X_np.shape[0]
        dummy = np.zeros((n, self.Y_train_np.shape[1]))
        return self.model.transform([X_np, dummy])[0]

    def _predict_field(self, X_da):
        """Deterministic Y field (T, Y, X) from a predictor DataArray."""
        if not self.is_fitted:
            raise ValueError("Model not fitted. Call fit_cca first.")
        X_da = self._drop_member_dim(X_da)
        X_da = self._ensure_tyx(X_da, name="X_predict")
        # Apply the SAME per-feature standardization learned on the train fold
        # (train stats), so the kernel sees the predictor on the scale it was
        # fitted/tuned on. No-op when self.standardize is False.
        if self.standardize and self._x_mean is not None:
            X_da = (X_da - self._x_mean) / self._x_std
        X_np, _ = self._stack_spatial(X_da)
        zx = self._x_scores(X_np)
        Y_np = zx @ self._reg_B + self._reg_mean
        return self._unstack_spatial(Y_np, self.Y_coords, X_da["T"].values)

    def compute_model(self, X_train, y_train, X_test, y_test, best_params=None):
        """
        Fit on the training fold and predict the test fold. Returns the
        deterministic hindcast in the original predictand scale (T, Y, X).
        """
        if self.standardize:
            y_train_st, y_mean, y_std = self._standardize_along_T(
                self._ensure_tyx(self._drop_member_dim(y_train), name="y_train")
            )
            self.fit_cca(X_train, y_train_st, best_params=best_params)
        else:
            self.fit_cca(X_train, y_train, best_params=best_params)

        X_test_prepared, y_test_prepared = self.preprocess_test_data(
            X_test, y_test, X_train, y_train
        )
        y_pred = self._predict_field(X_test_prepared)
        y_pred = y_pred.assign_coords(T=y_test_prepared["T"])

        if self.standardize:
            y_pred = self._reverse_along_T(y_pred, y_mean, y_std)
        return y_pred

    # ------------------------------------------------------------------ #
    # Probability utilities (identical to WAS_CCA_base)
    # ------------------------------------------------------------------ #

    @classmethod
    def _ppf_terciles_from_code(cls, dist_code, shape, loc, scale):
        if np.isnan(dist_code):
            return np.nan, np.nan
        code = int(dist_code)
        qlo, qhi = cls.Q_LOW, cls.Q_HIGH
        try:
            if code == 1:
                return norm.ppf(qlo, loc=loc, scale=scale), norm.ppf(qhi, loc=loc, scale=scale)
            if code == 2:
                return (lognorm.ppf(qlo, s=shape, loc=loc, scale=scale),
                        lognorm.ppf(qhi, s=shape, loc=loc, scale=scale))
            if code == 3:
                return expon.ppf(qlo, loc=loc, scale=scale), expon.ppf(qhi, loc=loc, scale=scale)
            if code == 4:
                return (gamma.ppf(qlo, a=shape, loc=loc, scale=scale),
                        gamma.ppf(qhi, a=shape, loc=loc, scale=scale))
            if code == 5:
                return (weibull_min.ppf(qlo, c=shape, loc=loc, scale=scale),
                        weibull_min.ppf(qhi, c=shape, loc=loc, scale=scale))
            if code == 6:
                return (t.ppf(qlo, df=shape, loc=loc, scale=scale),
                        t.ppf(qhi, df=shape, loc=loc, scale=scale))
            if code == 7:
                return poisson.ppf(qlo, mu=shape, loc=loc), poisson.ppf(qhi, mu=shape, loc=loc)
            if code == 8:
                return (nbinom.ppf(qlo, n=shape, p=scale, loc=loc),
                        nbinom.ppf(qhi, n=shape, p=scale, loc=loc))
        except Exception:
            return np.nan, np.nan
        return np.nan, np.nan

    @staticmethod
    def weibull_shape_solver(k, M, V):
        if k <= 0:
            return -np.inf
        try:
            g1 = gamma_function(1 + 1 / k)
            g2 = gamma_function(1 + 2 / k)
            implied = (g2 / (g1 ** 2)) - 1
            observed = V / (M ** 2)
            return observed - implied
        except Exception:
            return -np.inf

    @classmethod
    def calculate_tercile_probabilities_bestfit(cls, best_guess, error_variance,
                                                T1, T2, dist_code, dof):
        best_guess = np.asarray(best_guess, dtype=float)
        error_variance = float(error_variance)
        n_time = best_guess.size
        out = np.full((3, n_time), np.nan, dtype=float)

        if (np.all(np.isnan(best_guess)) or not np.isfinite(error_variance)
                or error_variance <= 0 or not np.isfinite(T1)
                or not np.isfinite(T2) or np.isnan(dist_code)):
            return out

        code = int(dist_code)
        try:
            if code == 1:
                s = np.sqrt(error_variance)
                c1 = norm.cdf(T1, loc=best_guess, scale=s)
                c2 = norm.cdf(T2, loc=best_guess, scale=s)
                out[0], out[1], out[2] = c1, c2 - c1, 1.0 - c2

            elif code == 2:
                valid = best_guess > 0
                sigma = np.full(n_time, np.nan)
                mu = np.full(n_time, np.nan)
                sigma[valid] = np.sqrt(np.log(1.0 + error_variance / (best_guess[valid] ** 2)))
                mu[valid] = np.log(best_guess[valid]) - 0.5 * sigma[valid] ** 2
                c1 = lognorm.cdf(T1, s=sigma, scale=np.exp(mu))
                c2 = lognorm.cdf(T2, s=sigma, scale=np.exp(mu))
                out[0], out[1], out[2] = c1, c2 - c1, 1.0 - c2

            elif code == 3:
                scale = np.sqrt(error_variance)
                loc = best_guess - scale
                c1 = expon.cdf(T1, loc=loc, scale=scale)
                c2 = expon.cdf(T2, loc=loc, scale=scale)
                out[0], out[1], out[2] = c1, c2 - c1, 1.0 - c2

            elif code == 4:
                valid = best_guess > 0
                alpha = np.full(n_time, np.nan)
                theta = np.full(n_time, np.nan)
                alpha[valid] = (best_guess[valid] ** 2) / error_variance
                theta[valid] = error_variance / best_guess[valid]
                c1 = gamma.cdf(T1, a=alpha, scale=theta)
                c2 = gamma.cdf(T2, a=alpha, scale=theta)
                out[0], out[1], out[2] = c1, c2 - c1, 1.0 - c2

            elif code == 5:
                for i in range(n_time):
                    M, V = best_guess[i], error_variance
                    if not np.isfinite(M) or M <= 0 or V <= 0:
                        continue
                    k = fsolve(cls.weibull_shape_solver, 2.0, args=(M, V))[0]
                    if not np.isfinite(k) or k <= 0:
                        continue
                    lam = M / gamma_function(1 + 1 / k)
                    c1 = weibull_min.cdf(T1, c=k, loc=0, scale=lam)
                    c2 = weibull_min.cdf(T2, c=k, loc=0, scale=lam)
                    out[0, i], out[1, i], out[2, i] = c1, c2 - c1, 1.0 - c2

            elif code == 6:
                if dof <= 2:
                    return out
                loc = best_guess
                scale = np.sqrt(error_variance * (dof - 2) / dof)
                c1 = t.cdf(T1, df=dof, loc=loc, scale=scale)
                c2 = t.cdf(T2, df=dof, loc=loc, scale=scale)
                out[0], out[1], out[2] = c1, c2 - c1, 1.0 - c2

            elif code == 7:
                valid = best_guess >= 0
                mu = np.where(valid, best_guess, np.nan)
                c1 = poisson.cdf(T1, mu=mu)
                c2 = poisson.cdf(T2, mu=mu)
                out[0], out[1], out[2] = c1, c2 - c1, 1.0 - c2

            elif code == 8:
                valid = error_variance > best_guess
                p = np.where(valid, best_guess / error_variance, np.nan)
                n = np.where(valid, best_guess ** 2 / (error_variance - best_guess), np.nan)
                c1 = nbinom.cdf(T1, n=n, p=p)
                c2 = nbinom.cdf(T2, n=n, p=p)
                out[0], out[1], out[2] = c1, c2 - c1, 1.0 - c2
            else:
                return out
        except Exception:
            return out

        out = np.clip(out, 0, 1)
        total = np.nansum(out, axis=0)
        ok = np.isfinite(total) & (total > 0)
        out[:, ok] = out[:, ok] / total[ok]
        return out

    @staticmethod
    def calculate_tercile_probabilities_nonparametric(best_guess, error_samples,
                                                      first_tercile, second_tercile):
        best_guess = np.asarray(best_guess, dtype=float)
        error_samples = np.asarray(error_samples, dtype=float)
        n_time = best_guess.size
        pred_prob = np.full((3, n_time), np.nan, dtype=float)

        valid_errors = error_samples[np.isfinite(error_samples)]
        if valid_errors.size == 0:
            return pred_prob

        for i in range(n_time):
            if not np.isfinite(best_guess[i]):
                continue
            dist = best_guess[i] + valid_errors
            p_below = np.mean(dist < first_tercile)
            p_normal = np.mean((dist >= first_tercile) & (dist < second_tercile))
            pred_prob[0, i] = p_below
            pred_prob[1, i] = p_normal
            pred_prob[2, i] = 1.0 - p_below - p_normal
        return pred_prob

    def _compute_tercile_probabilities(self, Predictant, deterministic,
                                       clim_year_start, clim_year_end,
                                       error_samples, error_variance,
                                       best_code_da=None, best_shape_da=None,
                                       best_loc_da=None, best_scale_da=None):
        Predictant = self._ensure_tyx(self._drop_member_dim(Predictant), name="Predictant")
        deterministic = self._ensure_tyx(self._drop_member_dim(deterministic),
                                         name="deterministic")
        mask = self._spatial_mask(Predictant)

        clim = Predictant.sel(T=slice(str(clim_year_start), str(clim_year_end)))
        if clim.sizes.get("T", 0) < 3:
            raise ValueError("Not enough years in climatology period for tercile computation.")

        terciles_emp = clim.quantile([self.Q_LOW, self.Q_HIGH], dim="T")
        T1_emp = terciles_emp.isel(quantile=0).drop_vars("quantile")
        T2_emp = terciles_emp.isel(quantile=1).drop_vars("quantile")
        dof = max(int(clim.sizes["T"]) - 1, 2)

        if self.dist_method == "bestfit":
            if any(v is None for v in (best_code_da, best_shape_da, best_loc_da, best_scale_da)):
                raise ValueError("dist_method='bestfit' requires best_code_da, "
                                 "best_shape_da, best_loc_da and best_scale_da.")
            T1, T2 = xr.apply_ufunc(
                self._ppf_terciles_from_code,
                best_code_da, best_shape_da, best_loc_da, best_scale_da,
                input_core_dims=[(), (), (), ()],
                output_core_dims=[(), ()],
                vectorize=True, dask="parallelized",
                output_dtypes=[float, float],
            )
            prob = xr.apply_ufunc(
                self.calculate_tercile_probabilities_bestfit,
                deterministic, error_variance, T1, T2, best_code_da,
                input_core_dims=[("T",), (), (), (), ()],
                output_core_dims=[("probability", "T")],
                vectorize=True, kwargs={"dof": dof},
                dask="parallelized", output_dtypes=[float],
                dask_gufunc_kwargs={"output_sizes": {"probability": 3},
                                    "allow_rechunk": True},
            )
        elif self.dist_method == "nonparam":
            # The error samples form a pool decoupled from the forecast time
            # axis; rename T -> S so xarray does not try to align them (the
            # forecast deterministic has a single, different T value).
            error_pool = error_samples.rename({"T": "S"})
            prob = xr.apply_ufunc(
                self.calculate_tercile_probabilities_nonparametric,
                deterministic, error_pool, T1_emp, T2_emp,
                input_core_dims=[("T",), ("S",), (), ()],
                output_core_dims=[("probability", "T")],
                vectorize=True, dask="parallelized", output_dtypes=[float],
                dask_gufunc_kwargs={"output_sizes": {"probability": 3},
                                    "allow_rechunk": True},
            )
        else:
            raise ValueError(f"Invalid dist_method: {self.dist_method}")

        prob = prob.assign_coords(probability=("probability", ["PB", "PN", "PA"]))
        prob = prob.transpose("probability", "T", "Y", "X")
        prob = self._normalize_probabilities(prob)
        return prob * mask

    def compute_prob(self, Predictant, clim_year_start, clim_year_end, hindcast_det,
                     best_code_da=None, best_shape_da=None,
                     best_loc_da=None, best_scale_da=None):
        Predictant = self._ensure_tyx(self._drop_member_dim(Predictant), name="Predictant")
        hindcast_det = self._ensure_tyx(self._drop_member_dim(hindcast_det),
                                        name="hindcast_det")

        if hindcast_det.sizes["T"] == Predictant.sizes["T"]:
            hindcast_det = hindcast_det.assign_coords(T=Predictant["T"])

        error_samples = Predictant - hindcast_det
        error_variance = error_samples.var(dim="T", skipna=True)

        return self._compute_tercile_probabilities(
            Predictant=Predictant, deterministic=hindcast_det,
            clim_year_start=clim_year_start, clim_year_end=clim_year_end,
            error_samples=error_samples, error_variance=error_variance,
            best_code_da=best_code_da, best_shape_da=best_shape_da,
            best_loc_da=best_loc_da, best_scale_da=best_scale_da,
        )

    # ------------------------------------------------------------------ #
    # Real-time forecast
    # ------------------------------------------------------------------ #

    def forecast(self, Predictant, clim_year_start, clim_year_end, Predictor,
                 hindcast_det, Predictor_for_year, best_params=None,
                 best_code_da=None, best_shape_da=None,
                 best_loc_da=None, best_scale_da=None):
        Predictant = self._ensure_tyx(self._drop_member_dim(Predictant), name="Predictant")
        Predictor = self._ensure_tyx(self._drop_member_dim(Predictor), name="Predictor")
        hindcast_det = self._ensure_tyx(self._drop_member_dim(hindcast_det),
                                        name="hindcast_det")
        mask = self._spatial_mask(Predictant)

        # Standardize the predictand before CCA (climatology window), same as
        # WAS_CCA_base, then fit and predict in standardized space.
        if self.standardize:
            Predictant_fit = standardize_timeseries(
                Predictant, clim_year_start, clim_year_end
            )
        else:
            Predictant_fit = Predictant

        self.fit_cca(Predictor, Predictant_fit, best_params=best_params)

        X_forecast_prepared = self.preprocess_forecast_data(
            Predictor_for_year=Predictor_for_year, Predictor_train=Predictor
        )
        forecast_det = self._predict_field(X_forecast_prepared)

        if self.standardize:
            forecast_det = reverse_standardize(
                forecast_det, Predictant, clim_year_start, clim_year_end
            )

        forecast_time = self._make_forecast_time(Predictant, Predictor_for_year)
        forecast_det = forecast_det.assign_coords(
            T=xr.DataArray(forecast_time, dims=["T"])
        )

        # Residual distribution from the supplied hindcast.
        if hindcast_det.sizes["T"] == Predictant.sizes["T"]:
            hindcast_det = hindcast_det.assign_coords(T=Predictant["T"])
        error_samples = Predictant - hindcast_det
        error_variance = error_samples.var(dim="T", skipna=True)

        forecast_prob = self._compute_tercile_probabilities(
            Predictant=Predictant, deterministic=forecast_det,
            clim_year_start=clim_year_start, clim_year_end=clim_year_end,
            error_samples=error_samples, error_variance=error_variance,
            best_code_da=best_code_da, best_shape_da=best_shape_da,
            best_loc_da=best_loc_da, best_scale_da=best_scale_da,
        )

        forecast_det = xr.where(forecast_det < 0, 0, forecast_det)
        return forecast_det * mask, forecast_prob * mask

    # ------------------------------------------------------------------ #
    # Introspection + plotting
    # ------------------------------------------------------------------ #

    def get_components(self):
        """Spatial patterns (linear kernel only)."""
        if not self.is_fitted:
            raise ValueError("Model not fitted.")
        if not self._is_linear_kernel():
            print("Warning: components are not interpretable for non-linear kernels.")
            return None, None
        weights = self.model.weights
        wX, wY = np.asarray(weights[0]), np.asarray(weights[1])  # (features, modes)
        modes = np.arange(1, self.n_modes + 1)
        X_comp = self._unstack_spatial(wX.T, self.X_coords, modes, time_name="mode")
        Y_comp = self._unstack_spatial(wY.T, self.Y_coords, modes, time_name="mode")
        return X_comp, Y_comp

    def get_scores(self):
        if not self.is_fitted:
            raise ValueError("Model not fitted.")
        modes = range(1, self.n_modes + 1)
        X_scores = xr.DataArray(self.zx_train, dims=["T", "mode"],
                                coords={"T": self.train_time, "mode": list(modes)})
        Y_scores = xr.DataArray(self.zy_train, dims=["T", "mode"],
                                coords={"T": self.train_time, "mode": list(modes)})
        return X_scores, Y_scores

    def get_canonical_correlations(self):
        if not self.is_fitted:
            raise ValueError("Model not fitted.")
        return self.canonical_correlations

    def get_model_info(self):
        info = {
            "model": type(self).__name__,
            "n_modes": self.n_modes,
            "kernel": self.kernel,
            "dist_method": self.dist_method,
            "standardize": self.standardize,
            "is_fitted": self.is_fitted,
            "is_tuned": self.is_tuned,
            "best_params": self.best_params,
        }
        if self.is_fitted:
            info["canonical_correlations"] = self.canonical_correlations
            info["training_samples"] = self.X_train_np.shape[0]
            info["X_features"] = self.X_train_np.shape[1]
            info["Y_features"] = self.Y_train_np.shape[1]
        return info

    def plot_cca_results(self, X=None, Y=None, n_modes=None,
                         clim_year_start=None, clim_year_end=None, best_params=None):
        """Plot canonical scores (always) and spatial modes (linear kernel)."""
        if X is not None and Y is not None:
            X = self._ensure_tyx(self._drop_member_dim(X), name="X")
            Y = self._ensure_tyx(self._drop_member_dim(Y), name="Y")
            if self.standardize and clim_year_start is not None and clim_year_end is not None:
                Y_fit = standardize_timeseries(Y, clim_year_start, clim_year_end)
            else:
                Y_fit = Y
            self.fit_cca(X, Y_fit, best_params=best_params)
            mask = self._spatial_mask(Y)
        elif not self.is_fitted:
            raise ValueError("Model not fitted. Provide X and Y or call fit_cca first.")
        else:
            mask = None

        if n_modes is None:
            n_modes = self.n_modes

        X_scores, Y_scores = self.get_scores()
        X_comp, Y_comp = self.get_components()
        linear = X_comp is not None

        ncols = 3 if linear else 1
        fig, axes = plt.subplots(n_modes, ncols,
                                 figsize=(5 * ncols, 3.5 * n_modes), squeeze=False)

        for i, mode in enumerate(range(1, n_modes + 1)):
            col = 0
            if linear:
                ax = axes[i, col]; col += 1
                X_comp.sel(mode=mode).plot(ax=ax, cmap="RdBu_r")
                ax.set_title(f"X Mode {mode}")

            ax = axes[i, col]; col += 1
            ax.plot(X_scores["T"].dt.year.values, X_scores.sel(mode=mode), label="X score")
            ax.plot(Y_scores["T"].dt.year.values, Y_scores.sel(mode=mode), label="Y score")
            ax.axhline(0, linestyle="--", linewidth=0.8)
            ax.legend()
            cc = self.canonical_correlations[mode - 1]
            ax.set_title(f"Scores Mode {mode} (r = {cc:.2f})")
            ax.set_xlabel("Time")
            ax.set_ylabel("Canonical variate")

            if linear:
                ax = axes[i, col]
                Y_mode = Y_comp.sel(mode=mode)
                if mask is not None:
                    Y_mode = Y_mode * mask
                Y_mode.plot(ax=ax, cmap="RdBu_r")
                ax.set_title(f"Y Mode {mode}")

        if not linear:
            print("Spatial components are not interpretable for the "
                  f"'{self.kernel}' kernel; showing canonical scores only.")
        plt.tight_layout()
        plt.show()


# ===========================================================================
# Concrete model 1: Kernel CCA
# ===========================================================================
class WAS_KCCA(WAS_KernelCCA_base):
    """Kernel Canonical Correlation Analysis (cca-zoo KCCA / LinearCCA)."""

    def _build_model(self, params):
        if self._is_linear_kernel():
            return LinearCCA(latent_dimensions=self.n_modes)
        return KCCA(latent_dimensions=self.n_modes, kernel=self.kernel, **params)

    def _grid_estimator(self):
        if self._is_linear_kernel():
            return LinearCCA(latent_dimensions=self.n_modes)
        return KCCA(latent_dimensions=self.n_modes, kernel=self.kernel)

    def _prepare_param_grid(self, kernel=None):
        kernel = kernel or self.kernel
        if kernel == "rbf":
            return {"gamma": [0.01, 0.05, 0.1, 0.5, 1.0, 2.0, 5.0]}
        if kernel == "poly":
            return {"degree": [2, 3, 4], "gamma": [0.1, 0.5, 1.0, 2.0],
                    "coef0": [0, 1, 2]}
        if kernel == "sigmoid":
            return {"gamma": [0.01, 0.05, 0.1, 0.5, 1.0], "coef0": [0, 1, 2]}
        return {}  # linear

    def _objective(self, trial, X_np, Y_np):
        if self.kernel == "rbf":
            params = {"gamma": trial.suggest_float("gamma", 0.01, 5.0, log=True)}
        elif self.kernel == "poly":
            params = {"degree": trial.suggest_int("degree", 2, 4),
                      "gamma": trial.suggest_float("gamma", 0.1, 5.0, log=True),
                      "coef0": trial.suggest_float("coef0", 0, 5)}
        elif self.kernel == "sigmoid":
            params = {"gamma": trial.suggest_float("gamma", 0.01, 1.0, log=True),
                      "coef0": trial.suggest_float("coef0", 0, 5)}
        else:
            return 0.0
        model = KCCA(latent_dimensions=self.n_modes, kernel=self.kernel, **params)
        cv = KFold(n_splits=self.cv_folds, shuffle=True, random_state=self.random_state)
        corr = []
        for tr, va in cv.split(X_np):
            model.fit([X_np[tr], Y_np[tr]])
            s = model.transform([X_np[va], Y_np[va]])
            corr.append(np.corrcoef(s[0][:, 0], s[1][:, 0])[0, 1])
        return float(np.mean(corr))


# ===========================================================================
# Concrete model 2: Kernel Generalized CCA
# ===========================================================================
class WAS_KGCCA(WAS_KernelCCA_base):
    """Kernel Generalized CCA (cca-zoo KGCCA / GCCA)."""

    def __init__(self, *args, cv_folds=5, **kwargs):
        super().__init__(*args, cv_folds=cv_folds, **kwargs)

    def _build_model(self, params):
        if self._is_linear_kernel():
            if "c" in params:
                return GCCA(latent_dimensions=self.n_modes, c=params["c"])
            return GCCA(latent_dimensions=self.n_modes)
        return KGCCA(latent_dimensions=self.n_modes, kernel=self.kernel, **params)

    def _grid_estimator(self):
        if self._is_linear_kernel():
            return GCCA(latent_dimensions=self.n_modes)
        return KGCCA(latent_dimensions=self.n_modes, kernel=self.kernel)

    def _prepare_param_grid(self, kernel=None):
        kernel = kernel or self.kernel
        if kernel == "rbf":
            return {"gamma": [[0.01, 0.01], [0.1, 0.1], [1.0, 1.0], [10.0, 10.0]],
                    "c": [[0, 0], [0.1, 0.1], [0.5, 0.5]]}
        if kernel == "poly":
            return {"degree": [[2, 2], [3, 3]], "gamma": [[0.1, 0.1], [1.0, 1.0]],
                    "coef0": [[0, 0], [1, 1]]}
        if kernel == "sigmoid":
            return {"gamma": [[0.01, 0.01], [0.1, 0.1], [1.0, 1.0]],
                    "coef0": [[0, 0], [1, 1]]}
        return {"c": [[0, 0], [0.1, 0.1], [0.5, 0.5], [1.0, 1.0]]}  # linear GCCA

    def _objective(self, trial, X_np, Y_np):
        if self.kernel == "rbf":
            g = trial.suggest_float("gamma", 0.01, 10.0, log=True)
            c = trial.suggest_float("c", 0, 1.0)
            params = {"gamma": [g, g], "c": [c, c]}
        elif self.kernel == "poly":
            d = trial.suggest_int("degree", 2, 4)
            g = trial.suggest_float("gamma", 0.1, 5.0, log=True)
            c0 = trial.suggest_float("coef0", 0, 5)
            params = {"degree": [d, d], "gamma": [g, g], "coef0": [c0, c0]}
        elif self.kernel == "sigmoid":
            g = trial.suggest_float("gamma", 0.01, 1.0, log=True)
            c0 = trial.suggest_float("coef0", 0, 5)
            params = {"gamma": [g, g], "coef0": [c0, c0]}
        else:
            c = trial.suggest_float("c", 0, 1.0)
            params = {"c": [c, c]}
        model = (GCCA(latent_dimensions=self.n_modes, **params)
                 if self._is_linear_kernel()
                 else KGCCA(latent_dimensions=self.n_modes, kernel=self.kernel, **params))
        cv = KFold(n_splits=self.cv_folds, shuffle=True, random_state=self.random_state)
        corr = []
        for tr, va in cv.split(X_np):
            model.fit([X_np[tr], Y_np[tr]])
            s = model.transform([X_np[va], Y_np[va]])
            corr.append(np.corrcoef(s[0][:, 0], s[1][:, 0])[0, 1])
        return float(np.mean(corr))


"""
EOF -> kernel-CCA pipeline for MULTIPLE predictor fields (SST, SLP, UWIND, ...).

Why
---
With ~30 years and several predictor fields, feeding the raw grids (hundreds to
thousands of points) into a CCA overfits. 

What this class does
--------------------
WAS_EOF_KCCA subclasses WAS_KCCA (the concrete kernel-CCA). It accepts the
predictor as a LIST of (T, [M], Y, X) DataArrays (one per field). Per fold it:
  1. fits a fold-safe WAS_EOF on each TRAIN field (mean/trend/EOFs learned on the
     training fold; the test fold is PROJECTED onto the training EOFs),
  2. concatenates the per-field PCs into a single pseudo predictor (T, Y, X=1),
  3. delegates to the parent kernel-CCA, which fits on the low-dim predictor and
     returns the deterministic hindcast in PHYSICAL scale.

Everything probabilistic (compute_prob, tercile dressing, forecast residuals) is
inherited unchanged from the kernel-CCA parent. Because the EOF is re-fit inside
compute_model, the reduction is leakage-free in cross-validation.

Two reduction modes
-------------------
  joint=False (default): one EOF PER field, keep eof_n_modes PCs each, concat ->
                         n_fields * eof_n_modes features. Explicit per-field
                         control; what "reduce each field then concatenate" means.
  joint=True           : one MULTIVARIATE EOF over all fields at once (WAS_EOF's
                         native list mode, per-field normalization on) -> a single
                         set of eof_n_modes joint modes. Use when fields are
                         physically coupled and you want shared modes.

"""

class WAS_EOF_KCCA(WAS_KCCA):

    def __init__(self, eof_n_modes=5, joint=False, eof_detrend=True,
                 eof_standardize=False, eof_use_coslat=True, **kcca_kwargs):
        """
        eof_n_modes      PCs kept per field (joint=False) or total joint modes
                         (joint=True).
        joint            False -> one EOF per field then concat PCs (default).
                         True  -> one multivariate EOF over all fields.
        eof_detrend / eof_standardize / eof_use_coslat
                         passed through to WAS_EOF.
        **kcca_kwargs    forwarded to WAS_KCCA / WAS_KernelCCA_base (n_modes, kernel,
                         dist_method, standardize, reg, search_method, ...).
        """
        super().__init__(**kcca_kwargs)
        self.eof_n_modes = eof_n_modes
        self.joint = joint
        self.eof_detrend = eof_detrend
        self.eof_standardize = eof_standardize
        self.eof_use_coslat = eof_use_coslat
        self._eofs = None      # list of fitted WAS_EOF (1 if joint, else 1/field)

    # ------------------------------------------------------------------ utils
    @staticmethod
    def _as_field_list(X):
        return list(X) if isinstance(X, (list, tuple)) else [X]

    @staticmethod
    def _drop_M(f):
        return f.isel(M=0, drop=True) if "M" in f.dims else f

    def _new_eof(self):
        return WAS_EOF(n_modes=self.eof_n_modes, detrend=self.eof_detrend,
                       standardize=self.eof_standardize,
                       use_coslat=self.eof_use_coslat)

    def _pcs_to_pseudo(self, pcs_list):
        """Per-field PCs (each dims (mode, T)) -> pseudo predictor (T, Y, X=1).

        The kernel-CCA treats the predictor as a flat feature vector, so packing
        the concatenated PCs as a (T, Y=feature, X=1) grid is mathematically
        transparent and keeps the parent code untouched."""
        cols, offset = [], 0
        for pcs in pcs_list:
            p = pcs.transpose("T", "mode")
            p = p.assign_coords(mode=np.arange(offset, offset + p.sizes["mode"]))
            offset += p.sizes["mode"]
            cols.append(p)
        X = xr.concat(cols, dim="mode").rename({"mode": "Y"})
        X = X.assign_coords(Y=np.arange(X.sizes["Y"]))
        X = X.expand_dims(X=[0]).transpose("T", "Y", "X")
        return X

    # ------------------------------------------------- fold-safe EOF reduction
    def _reduce_fit(self, X):
        """Fit EOF(s) on this fold's predictor and return the pseudo predictor."""
        fields = [self._drop_M(f) for f in self._as_field_list(X)]
        if self.joint:
            e = self._new_eof()
            e.fit(fields, dim="T")
            self._eofs = [e]
            return self._pcs_to_pseudo([e.transform(fields)])
        self._eofs, pcs = [], []
        for f in fields:
            e = self._new_eof()
            e.fit(f, dim="T")
            self._eofs.append(e)
            pcs.append(e.transform(f))
        return self._pcs_to_pseudo(pcs)

    def _reduce_apply(self, X):
        """Project a test / forecast predictor onto the already-fitted EOF(s)."""
        if self._eofs is None:
            raise ValueError("EOFs not fitted; call _reduce_fit first.")
        fields = [self._drop_M(f) for f in self._as_field_list(X)]
        if self.joint:
            return self._pcs_to_pseudo([self._eofs[0].transform(fields)])
        pcs = [e.transform(f) for f, e in zip(fields, self._eofs)]
        return self._pcs_to_pseudo(pcs)

    # ----------------------------------- overrides: reduce, then delegate up
    def compute_model(self, X_train, y_train, X_test, y_test, best_params=None):
        pcs_train = self._reduce_fit(X_train)     # EOF fit on TRAIN fold (fold-safe)
        pcs_test = self._reduce_apply(X_test)     # TEST projected onto TRAIN EOFs
        return super().compute_model(pcs_train, y_train, pcs_test, y_test,
                                     best_params=best_params)

    def compute_hyperparameters(self, Predictors, Predictand,
                                clim_year_start=None, clim_year_end=None):
        # Global tuning: EOFs fit on the full predictor (mild leakage, same order
        # as climatological terciles). The per-fold compute_model re-fits the EOFs.
        pcs = self._reduce_fit(Predictors)
        return super().compute_hyperparameters(pcs, Predictand,
                                               clim_year_start, clim_year_end)

    def forecast(self, Predictant, clim_year_start, clim_year_end, Predictor,
                 hindcast_det, Predictor_for_year, best_params=None,
                 best_code_da=None, best_shape_da=None,
                 best_loc_da=None, best_scale_da=None):
        pcs_hist = self._reduce_fit(Predictor)              # EOFs on the hindcast predictor
        pcs_year = self._reduce_apply(Predictor_for_year)   # project the forecast year
        return super().forecast(
            Predictant, clim_year_start, clim_year_end, pcs_hist,
            hindcast_det, pcs_year, best_params=best_params,
            best_code_da=best_code_da, best_shape_da=best_shape_da,
            best_loc_da=best_loc_da, best_scale_da=best_scale_da)