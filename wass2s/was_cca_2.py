"""Kernel CCA variants for seasonal forecasting.

Built on the cca-zoo library, these classes extend the conventions of
:mod:`wass2s.was_cca` with non-linear kernels.

Classes
-------
WAS_KernelCCA_base
    Abstract base implementing preprocessing, EOF/PCA pre-reduction,
    probabilistic post-processing and hyperparameter search for any cca-zoo
    kernel method.
WAS_KCCA
    Kernel CCA (cca-zoo ``KCCA`` / ``LinearCCA`` for the linear kernel).
WAS_KGCCA
    Kernel Generalised CCA (cca-zoo ``KGCCA`` / ``GCCA``).
WAS_EOF_KCCA
    EOF pre-reduction layer on top of ``WAS_KCCA``: reduces each spatial
    predictor field to its leading principal components before applying
    kernel CCA. This is the only configuration recommended for operational use.


Why this revision exists
------------------------
Kernel CCA memorises at seasonal-forecast sample sizes. Measured on **pure
noise**, 25 training years, RBF kernel, first canonical correlation on the
training set (cca-zoo 3.1.1):

    features (X / Y)     c=1e-6     c=0.1      c=10
    40 / 20              +0.999     +0.999     +0.966
    20 / 10              +0.998     +0.998     +0.627
    10 /  6              +0.998     +0.993     +0.486
     6 /  4              +0.991     +0.956     +0.156

Two things follow, and they drive most of the changes below.

1. Tikhonov regularisation only bites once the number of features is well below
   the number of years. Adding ``c`` to the search grid is necessary but not
   sufficient: the PCA pre-reduction has to cap the feature count first. A
   variance threshold such as ``pca_explained_variance=0.80`` does not do that --
   on an SST field it can retain 20+ PCs. ``pca_max_feature_ratio`` now caps the
   retained components at a fraction of the training sample size.

2. A training-set canonical correlation is not evidence of coupling. The old
   code stored only the training value and printed it in plot titles as
   ``r = 0.98``. A separate conditional correlation diagnostic is available,
   but forecast skill must come from predictions made by the outer validation
   procedure, with preprocessing fitted inside its training folds.

Other corrections with respect to the previous revision
-------------------------------------------------------
* The whole probability layer is now inherited from
  ``wass2s.was_cca._ProbabilityEngine`` instead of being duplicated here. That
  brings exact 1/3 and 2/3 tercile thresholds (was 0.33 / 0.67), a bounded
  ``brentq`` Weibull solver (was an unbounded ``fsolve``), and shared residual
  calibration. Out-of-sample residuals supply a bias correction to the forecast
  centre and their variance supplies the predictive spread. The shared engine
  also guards against discrete families on a continuous predictand.
* ``get_components`` no longer crashes under the default ``pca_backend="xeofs"``.
  The old code called ``self._x_pca.components_``, an sklearn attribute the
  xeofs model does not have.
* ``compute_model`` and ``forecast`` now standardise the predictand the same
  way -- on the statistics of whatever plays the role of the training set. The
  old code used fold statistics in cross-validation and climatology statistics
  in the forecast, so the residual sample dressing the forecast came from a
  differently normalised model.
* The Optuna objective averages the first ``k`` canonical correlations on the
  validation fold instead of scoring the leading mode alone, and can score the
  reconstructed field instead.
* Non-negativity clipping is conditional on the sign of the observations, and
  the spatial mask keys off the whole record rather than ``T=0``.

The ``dummy`` second view passed to ``model.transform`` is NOT a fragility.
Verified on cca-zoo 3.1.1: X scores are bit-identical whether the second view is
zeros or random data, and ``None`` is rejected by the validator. It stays.
"""

# =============================================================================
# WAS_KCCA / WAS_KGCCA
# -----------------------------------------------------------------------------
# Kernel CCA (KCCA) and Kernel Generalized CCA (KGCCA) seasonal-forecast models
# =============================================================================

import warnings

import numpy as np
import xarray as xr
import matplotlib.pyplot as plt

from scipy.special import gamma as gamma_function  # noqa: F401  (public re-export)

from wass2s.utils import *  # noqa: F401,F403
from wass2s.was_eof import *  # noqa: F401,F403

# Shared machinery. Importing rather than duplicating means a fix to the tercile
# code lands in the linear and the kernel models at the same time.
from wass2s.was_cca import (
    _ProbabilityEngine,
    _pufunc,  # noqa: F401  (kept importable for callers that used it from here)
    _normalize_probabilities as _shared_normalize_probabilities,
    _predictive_error_stats,
    _target_time,
    spatial_mask,
    conditional_nonnegative_clip,
)

try:
    from cca_zoo.linear import CCA as LinearCCA, GCCA
    from cca_zoo.nonparametric import KCCA, KGCCA
    from cca_zoo.model_selection import GridSearchCV as CCA_GridSearch
    from sklearn.model_selection import KFold
    from sklearn.decomposition import PCA
    HAS_CCA_ZOO = True
except ImportError:
    HAS_CCA_ZOO = False
    print("Warning: cca-zoo not installed. CCA/KCCA/KGCCA functionality disabled.")

try:
    import xeofs as xe
    HAS_XEOFS = True
except ImportError:
    xe = None
    HAS_XEOFS = False


class WAS_KernelCCA_base(_ProbabilityEngine):
    """
    Shared base for kernel CCA seasonal-forecast models (KCCA, KGCCA).

    Subclasses only need to implement the model-construction hooks:
        _build_model(params)      -> a fitted-ready cca-zoo estimator
        _grid_estimator()         -> estimator used by GridSearchCV
        _prepare_param_grid()     -> dict for grid search
        _params_for_trial(trial)  -> parameters sampled by Optuna
        _is_linear_kernel()       -> whether interpretable feature weights exist

    Conventions identical to WAS_CCA_base:
        WAS dims:  T, Y, X
        Predictand standardized before CCA, returned in original scale.
        Missing values filled with training-period means (no leakage).

    The probability layer (compute_prob, tercile dressing, distribution
    families) is inherited from wass2s.was_cca._ProbabilityEngine, which needs
    only ``self.dist_method`` and ``self.n_modes``.
    """

    # Kept for backwards compatibility with callers that read them. The shared
    # engine uses exact thirds internally; these are informational only.
    Q_LOW = 1.0 / 3.0
    Q_HIGH = 2.0 / 3.0

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
        pca_explained_variance=0.80,
        pca_min_components=None,
        pca_max_feature_ratio=0.5,
        pca_whiten=False,
        pca_backend="xeofs",
        pca_use_coslat=False,
        pca_solver="auto",
        tuning_objective="canonical",
        warn_memorisation=True,
        cv_gap=0,
    ):
        self.n_modes = n_modes
        self.kernel = kernel
        self.dist_method = dist_method
        self.standardize = standardize
        self.cv_folds = cv_folds
        self.cv_gap = int(cv_gap)
        if self.cv_gap < 0:
            raise ValueError("cv_gap must be nonnegative.")
        self.random_state = random_state
        self.search_method = search_method
        self.reg = reg  # ridge term for the scores -> field regression

        # Optional fold-safe PCA pre-reduction before CCA/KCCA/KGCCA.
        # pca_explained_variance=0.80 keeps the smallest number of PCs
        # explaining at least 80% variance; set to None to disable, or to an
        # integer to request exactly that many components.
        self.pca_explained_variance = pca_explained_variance
        self.pca_min_components = pca_min_components

        # Hard cap on the retained components, as a fraction of the number of
        # training samples. A variance threshold alone does not control the
        # feature-to-sample ratio, and that ratio is what decides whether the
        # kernel generalises or memorises (see the module docstring table).
        # 0.5 means: never more PCs than half the training years. Set to None to
        # restore the old, unguarded behaviour.
        self.pca_max_feature_ratio = pca_max_feature_ratio

        self.pca_whiten = pca_whiten
        if pca_backend not in {"xeofs", "sklearn"}:
            raise ValueError("pca_backend must be 'xeofs' or 'sklearn'.")
        self.pca_backend = pca_backend
        self.pca_use_coslat = pca_use_coslat
        self.pca_solver = pca_solver

        # "canonical": mean of the first n_modes canonical correlations on the
        # validation fold. "reconstruction": correlation of the reconstructed
        # predictand, which is what the forecast actually uses and is therefore
        # the more honest target, at a higher cost.
        if tuning_objective not in {"canonical", "reconstruction"}:
            raise ValueError("tuning_objective must be 'canonical' or 'reconstruction'.")
        self.tuning_objective = tuning_objective
        self.warn_memorisation = warn_memorisation

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

        # In-sample canonical correlations, and their cross-validated
        # counterparts once computed. Never report the first without the
        # second: at 25-35 years the in-sample value approaches 1 even on noise.
        self.canonical_correlations = None
        self.canonical_correlations_cv_ = None

        # Predictand standardization statistics, learned on whatever plays the
        # role of the training set, so compute_model and forecast share one
        # convention (they did not before).
        self._y_mean = None
        self._y_std = None

        # Predictor (X) standardization stats, learned on the TRAIN fold so the
        # same per-feature scaling is applied at fit AND predict. This keeps the
        # RBF/poly kernel scale consistent with compute_hyperparameters (which
        # also standardizes X when self.standardize is True). Gated by
        # self.standardize; for the linear kernel CCA is scale-invariant so this
        # is immaterial, but it removes the tune/fit mismatch for non-linear kernels.
        self._x_mean = None      # (Y, X)
        self._x_std = None       # (Y, X)

        # Per-feature statistics of the PCA-reduced views, learned on the train
        # fold. A kernel measures distances, and PCA scores are NOT unit
        # variance: the leading PC carries far more spread than the trailing
        # ones, and the two views can differ by an order of magnitude (a
        # standardized predictor at ~1 against a predictand in mm at ~40). With
        # gamma defaulting to 1/n_features the RBF kernel then saturates on the
        # wider view -- measured here as validation scores collapsing to a
        # constant, which makes the canonical correlation undefined. Scaling the
        # reduced views puts both on a comparable footing and makes gamma mean
        # the same thing for X and Y.
        self._zx_mean = None
        self._zx_std = None
        self._zy_mean = None
        self._zy_std = None

        # PCA artefacts. They are fitted on the TRAIN fold in fit_cca and
        # reused for the corresponding test / forecast predictor.
        self._x_pca = None
        self._y_pca = None
        self._x_pca_variance = None
        self._y_pca_variance = None
        self._x_pca_modes = None
        self._y_pca_modes = None
        self._X_raw_n_features = None
        self._Y_raw_n_features = None

    # ------------------------------------------------------------------ #
    # Hooks to be overridden by subclasses
    # ------------------------------------------------------------------ #

    def _build_model(self, params):
        raise NotImplementedError

    def _grid_estimator(self):
        raise NotImplementedError

    def _prepare_param_grid(self, kernel=None):
        raise NotImplementedError

    def _params_for_trial(self, trial):
        """Kernel parameters sampled from one Optuna trial."""
        raise NotImplementedError

    def _model_from_params(self, params):
        """Standalone estimator used for scoring. Defaults to _build_model."""
        return self._build_model(params)

    def _objective(self, trial, X_np, Y_np):
        """Optuna objective. Subclasses rarely need to override this."""
        return self._score_params(self._params_for_trial(trial), X_np, Y_np)

    def _is_linear_kernel(self):
        return self.kernel == "linear"

    # ------------------------------------------------------------------ #
    # Internal utilities
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
        """Cells with at least three valid observations.

        The previous implementation tested ``T=0`` only, which discarded a cell
        for the whole record when its first year happened to be missing.
        """
        return spatial_mask(Predictant)

    @staticmethod
    def _normalize_probabilities(prob: "xr.DataArray") -> "xr.DataArray":
        return _shared_normalize_probabilities(prob)

    @staticmethod
    def _make_forecast_time(Predictant: "xr.DataArray", Predictor_for_year: "xr.DataArray"):
        """Forecast year from the predictor, target month from the predictand."""
        return _target_time(Predictant, Predictor_for_year)

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

    # ---- PCA / EOF pre-reduction before CCA/KCCA/KGCCA -----------

    def _use_pca(self):
        return self.pca_explained_variance is not None

    def _feature_cap(self, n_samples):
        """Maximum number of retained components allowed by the sample size."""
        if self.pca_max_feature_ratio is None:
            return None
        cap = int(np.floor(float(self.pca_max_feature_ratio) * int(n_samples)))
        # A claimed hard cap must not silently expand to accommodate n_modes.
        # fit_cca explicitly rejects unsupported latent dimensionality.
        return max(1, cap)

    def _resolve_n_pca_components(self, explained_ratio, n_samples=None, view_name="view"):
        """Return the number of PCs/EOFs to keep for one view.

        For a float in (0, 1], keep enough components to reach that cumulative
        explained variance. For an integer, keep exactly that many components.
        In both cases keep at least n_modes, and never more than the cap implied
        by ``pca_max_feature_ratio``.
        """
        max_components = int(explained_ratio.size)
        if max_components <= 0:
            return 0

        target = self.pca_explained_variance
        if isinstance(target, (float, np.floating)) and 0 < float(target) <= 1:
            n_keep = int(np.searchsorted(np.cumsum(explained_ratio), float(target)) + 1)
        elif isinstance(target, (int, np.integer)) and int(target) >= 1:
            n_keep = int(target)
        else:
            raise ValueError(
                "pca_explained_variance must be None, a float in (0, 1], "
                "or an integer >= 1."
            )

        min_components = self.n_modes if self.pca_min_components is None else int(self.pca_min_components)
        n_keep = max(n_keep, min_components)

        # The dimensionality guard. Without it a variance threshold on a large
        # SST field routinely retains more components than there are years, and
        # the kernel then reproduces the training set exactly.
        if n_samples is not None:
            cap = self._feature_cap(n_samples)
            if cap is not None and n_keep > cap:
                warnings.warn(
                    f"{view_name}: variance target asked for {n_keep} components but "
                    f"only {cap} are allowed at {n_samples} training samples "
                    f"(pca_max_feature_ratio={self.pca_max_feature_ratio}). "
                    "Truncating. Kernel CCA memorises when features approach samples.",
                    RuntimeWarning, stacklevel=3,
                )
                n_keep = cap

        return max(1, min(n_keep, max_components))

    def _check_feature_ratio(self, n_samples, n_x, n_y):
        """Warn when the fitted problem sits in the memorisation regime."""
        if not self.warn_memorisation:
            return
        if self._is_linear_kernel() and int(n_x) + int(n_y) >= int(n_samples) - 1:
            warnings.warn(
                f"Linear CCA joint dimension is {int(n_x) + int(n_y)} "
                f"({n_x} X + {n_y} Y features) for {n_samples} samples. "
                "Separate per-view PCA caps do not control this joint rank: "
                "at or above the centered sample dimension, training "
                "canonical correlations can be artificially near one. "
                "Consider a smaller combined PCA rank; retained dimensions "
                "have not been changed automatically.",
                RuntimeWarning, stacklevel=3,
            )
        worst = max(int(n_x), int(n_y))
        if n_samples > 0 and worst > 0.5 * n_samples:
            warnings.warn(
                f"Kernel CCA fitted with {n_x} X features and {n_y} Y features on "
                f"{n_samples} samples. Above roughly half the sample size the "
                "in-sample canonical correlations approach 1 even on unrelated "
                "fields. Reduce pca_explained_variance or use WAS_EOF_KCCA.",
                RuntimeWarning, stacklevel=3,
            )

    # -- sklearn backend: keeps the old pure-numpy behaviour --------------

    def _fit_one_sklearn_pca(self, A_np, view_name):
        """Fit sklearn PCA for one already-stacked view."""
        if not self._use_pca():
            return None, A_np, None, None

        n_samples, n_features = A_np.shape
        max_components = min(n_samples, n_features)
        if max_components <= 1:
            return None, A_np, None, None

        pca_full = PCA(
            n_components=max_components,
            svd_solver="full",
            whiten=self.pca_whiten,
            random_state=self.random_state,
        )
        # One full SVD supplies both the variance-based rank selection and the
        # retained projection. Refitting a second full-solver PCA after choosing
        # n_keep repeated the same spatial decomposition in every fold.
        A_full = pca_full.fit_transform(A_np)
        n_keep = self._resolve_n_pca_components(
            pca_full.explained_variance_ratio_, n_samples=n_samples, view_name=view_name
        )
        pca = pca_full
        pca.noise_variance_ = (
            float(np.mean(pca.explained_variance_[n_keep:]))
            if n_keep < max_components else 0.0
        )
        for attr in ("components_", "explained_variance_",
                     "explained_variance_ratio_", "singular_values_"):
            setattr(pca, attr, getattr(pca, attr)[:n_keep].copy())
        pca.n_components = pca.n_components_ = n_keep
        A_red = A_full[:, :n_keep].copy()
        variance_kept = float(np.nansum(pca.explained_variance_ratio_))
        print(
            f"{view_name} sklearn PCA: kept {n_keep}/{n_features} components "
            f"({100 * variance_kept:.1f}% explained variance)."
        )
        return pca, A_red, variance_kept, None

    def _fit_sklearn_pca_views(self, X_np, Y_np, store=True):
        """Fit sklearn PCA on X and Y views, optionally storing transforms."""
        x_pca, X_use, x_var, x_modes = self._fit_one_sklearn_pca(X_np, "X")
        y_pca, Y_use, y_var, y_modes = self._fit_one_sklearn_pca(Y_np, "Y")

        if store:
            self._x_pca = x_pca
            self._y_pca = y_pca
            self._x_pca_variance = x_var
            self._y_pca_variance = y_var
            self._x_pca_modes = x_modes
            self._y_pca_modes = y_modes
            self._X_raw_n_features = X_np.shape[1]
            self._Y_raw_n_features = Y_np.shape[1]

        return X_use, Y_use

    # -- xeofs backend: xarray-native EOF/PCA with optional cos(lat) weights --

    @staticmethod
    def _to_xeofs_dims(da: "xr.DataArray") -> "xr.DataArray":
        """Rename WAS spatial dimensions to xeofs-friendly names."""
        rename = {}
        if "Y" in da.dims:
            rename["Y"] = "lat"
        if "X" in da.dims:
            rename["X"] = "lon"
        return da.rename(rename) if rename else da

    @staticmethod
    def _from_xeofs_dims(da: "xr.DataArray") -> "xr.DataArray":
        """Rename xeofs spatial dimensions back to WAS names."""
        rename = {}
        if "lat" in da.dims:
            rename["lat"] = "Y"
        if "lon" in da.dims:
            rename["lon"] = "X"
        return da.rename(rename) if rename else da

    def _new_xeofs_eof(self, n_modes):
        if not HAS_XEOFS:
            raise ImportError(
                "pca_backend='xeofs' requires xeofs. Install with: "
                "conda install -c conda-forge xeofs  # or pip install xeofs"
            )
        return xe.single.EOF(
            n_modes=int(n_modes),
            center=True,
            standardize=False,   # standardization is already controlled by self.standardize
            use_coslat=self.pca_use_coslat,
            check_nans=True,
            compute=True,
            random_state=self.random_state,
            solver=self.pca_solver,
        )

    @staticmethod
    def _scores_to_numpy(scores: "xr.DataArray"):
        """xeofs scores -> ndarray shaped (T, mode)."""
        if "T" not in scores.dims or "mode" not in scores.dims:
            raise ValueError(f"xeofs scores must contain T and mode dimensions, got {scores.dims}")
        return scores.transpose("T", "mode").values

    def _fit_one_xeofs_pca(self, da: "xr.DataArray", view_name):
        """Fit xeofs EOF/PCA for one xarray view and return reduced scores."""
        if not self._use_pca():
            A_np, _ = self._stack_spatial(da)
            return None, A_np, None, None

        da = self._ensure_tyx(self._drop_member_dim(da), name=f"{view_name}_pca_input")
        da_xe = self._to_xeofs_dims(da)
        n_samples = int(da_xe.sizes["T"])
        n_features = int(np.prod([da_xe.sizes[d] for d in da_xe.dims if d != "T"]))
        max_components = min(n_samples, n_features)
        if max_components <= 1:
            A_np, _ = self._stack_spatial(da)
            return None, A_np, None, None

        # Fit once with the maximum admissible rank, then retain the leading
        # modes required to reach the requested variance threshold. The fitted
        # full model is kept because xeofs inverse_transform accepts a subset
        # of scores with a mode dimension.
        model = self._new_xeofs_eof(max_components)
        model.fit(da_xe, dim="T")
        evr = np.asarray(model.explained_variance_ratio().values, dtype=float)
        n_keep = self._resolve_n_pca_components(
            evr, n_samples=n_samples, view_name=view_name
        )
        modes = model.explained_variance_ratio().coords["mode"].isel(mode=slice(0, n_keep))
        scores = model.transform(da_xe, normalized=False).isel(mode=slice(0, n_keep))
        A_red = self._scores_to_numpy(scores)
        variance_kept = float(np.nansum(evr[:n_keep]))

        print(
            f"{view_name} xeofs PCA/EOF: kept {n_keep}/{n_features} components "
            f"({100 * variance_kept:.1f}% explained variance)."
        )
        return model, A_red, variance_kept, modes

    def _fit_xeofs_pca_views(self, X_da, Y_da, store=True):
        """Fit xarray-native xeofs EOF/PCA on X and Y views."""
        x_pca, X_use, x_var, x_modes = self._fit_one_xeofs_pca(X_da, "X")
        y_pca, Y_use, y_var, y_modes = self._fit_one_xeofs_pca(Y_da, "Y")

        if store:
            self._x_pca = x_pca
            self._y_pca = y_pca
            self._x_pca_variance = x_var
            self._y_pca_variance = y_var
            self._x_pca_modes = x_modes
            self._y_pca_modes = y_modes
            self._X_raw_n_features = int(np.prod([X_da.sizes[d] for d in X_da.dims if d != "T"]))
            self._Y_raw_n_features = int(np.prod([Y_da.sizes[d] for d in Y_da.dims if d != "T"]))

        return X_use, Y_use

    def _fit_pca_views(self, X_da, Y_da, X_np=None, Y_np=None, store=True):
        """Fit PCA/EOF on both CCA views using the selected backend."""
        if not self._use_pca():
            if X_np is None:
                X_np, _ = self._stack_spatial(X_da)
            if Y_np is None:
                Y_np, _ = self._stack_spatial(Y_da)
            return X_np, Y_np

        if self.pca_backend == "xeofs":
            return self._fit_xeofs_pca_views(X_da, Y_da, store=store)

        if X_np is None:
            X_np, _ = self._stack_spatial(X_da)
        if Y_np is None:
            Y_np, _ = self._stack_spatial(Y_da)
        return self._fit_sklearn_pca_views(X_np, Y_np, store=store)

    def _transform_x_pca(self, X_da, X_np=None):
        """Apply the fitted X PCA/EOF to a predictor field."""
        if self._x_pca is None:
            if X_np is None:
                X_np, _ = self._stack_spatial(X_da)
            return X_np

        if self.pca_backend == "xeofs":
            X_da = self._ensure_tyx(self._drop_member_dim(X_da), name="X_predict")
            scores = self._x_pca.transform(self._to_xeofs_dims(X_da), normalized=False)
            if self._x_pca_modes is not None:
                scores = scores.sel(mode=self._x_pca_modes)
            return self._scores_to_numpy(scores)

        if X_np is None:
            X_np, _ = self._stack_spatial(X_da)
        return self._x_pca.transform(X_np)

    def _inverse_y_pca(self, Y_np, time_coords):
        """Inverse-transform reduced Y scores back to the original Y grid."""
        if self._y_pca is None:
            return Y_np

        if self.pca_backend == "xeofs":
            if self._y_pca_modes is None:
                raise ValueError("Missing stored Y xeofs modes.")
            scores = xr.DataArray(
                Y_np,
                dims=("T", "mode"),
                coords={"T": time_coords, "mode": self._y_pca_modes},
            )
            rec = self._y_pca.inverse_transform(scores, normalized=False)
            rec = self._from_xeofs_dims(rec)
            rec = self._ensure_tyx(rec, name="Y_reconstructed")
            rec_np, _ = self._stack_spatial(rec)
            return rec_np

        return self._y_pca.inverse_transform(Y_np)

    def _pca_components_matrix(self, which):
        """Loadings of one fitted PCA/EOF as ``(n_features, n_kept)``.

        Backend-aware. The previous implementation reached for the sklearn
        attribute ``components_`` regardless of backend, so ``get_components``
        raised AttributeError under the default ``pca_backend='xeofs'``.
        Returns None when the loadings cannot be expressed on the original grid.
        """
        model = self._x_pca if which == "x" else self._y_pca
        if model is None:
            return None

        if self.pca_backend == "sklearn":
            return np.asarray(model.components_).T  # (features, kept)

        modes = self._x_pca_modes if which == "x" else self._y_pca_modes
        if modes is None:
            return None
        try:
            comps = model.components().sel(mode=modes)      # xeofs: (mode, lat, lon)
            stacked = comps.stack(feature=["lat", "lon"]).transpose("feature", "mode")
            return np.asarray(stacked.values)               # (features, kept)
        except Exception:
            return None

    # ---- kernel-space scaling of the reduced views ----------------------

    @staticmethod
    def _column_stats(A_np):
        """Per-column mean and a strictly positive standard deviation."""
        mean = np.nanmean(A_np, axis=0)
        std = np.nanstd(A_np, axis=0)
        std = np.where(np.isfinite(std) & (std > 0), std, 1.0)
        return mean, std

    def _scale_reduced(self, A_np, mean, std):
        """Apply stored column statistics. No-op when standardize is False."""
        if not self.standardize or mean is None or std is None:
            return A_np
        return (A_np - mean) / std

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
        y_test = self._drop_member_dim(y_test) if y_test is not None else None
        X_train = self._drop_member_dim(X_train)
        y_train = self._drop_member_dim(y_train)

        X_test = self._ensure_tyx(X_test, name="X_test")
        y_test = self._ensure_tyx(y_test, name="y_test") if y_test is not None else None
        X_train = self._ensure_tyx(X_train, name="X_train")
        y_train = self._ensure_tyx(y_train, name="y_train")

        X_test = self._fill_test_data(X_test, X_train)
        y_test = self._fill_test_data(y_test, y_train) if y_test is not None else None
        return X_test, y_test

    def preprocess_forecast_data(self, Predictor_for_year, Predictor_train):
        Predictor_for_year = self._drop_member_dim(Predictor_for_year)
        Predictor_train = self._drop_member_dim(Predictor_train)

        Predictor_for_year = self._ensure_tyx(Predictor_for_year, name="Predictor_for_year")
        Predictor_train = self._ensure_tyx(Predictor_train, name="Predictor_train")

        Predictor_for_year = self._fill_test_data(Predictor_for_year, Predictor_train)
        return Predictor_for_year

    # ------------------------------------------------------------------ #
    # Diagnostics
    # ------------------------------------------------------------------ #

    def _canonical_correlations_cv(self, X_np, Y_np, params, n_splits=None):
        """Conditional diagnostics on already-prepared views, not pipeline skill."""
        warnings.warn(
            "Canonical diagnostics reuse the supplied reduced views. PCA/scaling "
            "are not refitted here: use outer-CV predictions to report forecast skill.",
            RuntimeWarning, stacklevel=2,
        )
        n_samples = X_np.shape[0]
        n_splits = int(n_splits or self.cv_folds)
        if n_samples < 2 * n_splits:
            return None
        cv = KFold(n_splits=n_splits, shuffle=True, random_state=self.random_state)
        per_fold = []
        for tr, va in cv.split(X_np):
            try:
                model = self._model_from_params(params)
                model.fit([X_np[tr], Y_np[tr]])
                s = model.transform([X_np[va], Y_np[va]])
                k = min(self.n_modes, s[0].shape[1], s[1].shape[1])
                per_fold.append([
                    np.corrcoef(s[0][:, i], s[1][:, i])[0, 1] for i in range(k)
                ])
            except Exception:
                continue
        if not per_fold:
            return None
        width = min(len(r) for r in per_fold)
        if width == 0:
            return None
        arr = np.asarray([r[:width] for r in per_fold], dtype=float)
        with np.errstate(invalid="ignore"):
            return [float(np.nanmean(arr[:, i])) for i in range(width)]

    def diagnose_memorisation(self, X_np=None, Y_np=None, params=None, verbose=True):
        """Compare in-sample and cross-validated canonical correlations.

        A large gap means the kernel is reproducing the training years rather
        than learning a relationship. Returns ``(in_sample, cross_validated)``.
        """
        if X_np is None or Y_np is None:
            if self.X_train_np is None:
                raise ValueError("Fit the model first, or pass X_np and Y_np.")
            X_np, Y_np = self.X_train_np, self.Y_train_np
        params = params if params is not None else (self.best_params or {})

        cv_r = self._canonical_correlations_cv(X_np, Y_np, params)
        self.canonical_correlations_cv_ = cv_r

        if cv_r is not None and self.canonical_correlations is not None:
            k = min(len(cv_r), len(self.canonical_correlations))
            train_r = [self.canonical_correlations[i] for i in range(k)]
            if verbose:
                print("  canonical correlations  in-sample: "
                      + ", ".join(f"{v:+.2f}" for v in train_r))
                print("                                 cv: "
                      + ", ".join(f"{v:+.2f}" for v in cv_r[:k]))
            gap = float(np.nanmean(np.abs(train_r)) - np.nanmean(np.abs(cv_r[:k])))
            if gap > 0.4:
                warnings.warn(
                    f"In-sample canonical correlations exceed the cross-validated "
                    f"ones by {gap:.2f} on average. The kernel is largely "
                    "memorising the training years; do not read the in-sample "
                    "values as evidence of coupling.",
                    RuntimeWarning, stacklevel=2,
                )
        return self.canonical_correlations, cv_r

    # ------------------------------------------------------------------ #
    # Shared scoring for hyperparameter search
    # ------------------------------------------------------------------ #

    def _score_params(self, params, X_np, Y_np):
        """Cross-validated score for one parameter set.

        'canonical'      mean of the first k canonical correlations on the
                         validation fold. The previous implementation scored the
                         leading mode alone, so with n_modes > 1 the bandwidth
                         could be tuned to mode 1 at the expense of the rest.
        'reconstruction' correlation of the reconstructed predictand, i.e. the
                         quantity the forecast actually uses. The scores -> field
                         regression is refitted on the training part of each
                         inner fold so nothing from the validation part leaks in.
        """
        cv = KFold(n_splits=self.cv_folds, shuffle=True, random_state=self.random_state)
        scores = []
        for tr, va in cv.split(X_np):
            try:
                model = self._model_from_params(params)
                model.fit([X_np[tr], Y_np[tr]])

                if self.tuning_objective == "canonical":
                    s = model.transform([X_np[va], Y_np[va]])
                    k = min(self.n_modes, s[0].shape[1], s[1].shape[1])
                    r = [np.corrcoef(s[0][:, i], s[1][:, i])[0, 1] for i in range(k)]
                    scores.append(float(np.nanmean(r)))
                else:
                    z_tr = model.transform([X_np[tr], Y_np[tr]])[0]
                    mean = Y_np[tr].mean(axis=0)
                    G = z_tr.T @ z_tr + self.reg * np.eye(z_tr.shape[1])
                    B = np.linalg.solve(G, z_tr.T @ (Y_np[tr] - mean))
                    z_va = model.transform([X_np[va], np.zeros_like(Y_np[va])])[0]
                    pred = z_va @ B + mean
                    obs = Y_np[va]
                    pa = pred - pred.mean(axis=0)
                    oa = obs - obs.mean(axis=0)
                    den = np.sqrt(np.nansum(pa ** 2) * np.nansum(oa ** 2))
                    scores.append(float(np.nansum(pa * oa) / den) if den > 0 else np.nan)
            except Exception:
                continue
        return float(np.nanmean(scores)) if scores else -np.inf

    # ------------------------------------------------------------------ #
    # Hyperparameter tuning (do this once)
    # ------------------------------------------------------------------ #

    def _prepare_inner_views(self, Predictors, Predictand):
        """Cache PCA/scaling once per inner training fold, before kernel search."""
        import copy
        n = Predictand.sizes["T"]
        if self.cv_folds < 2 or n < 2 * self.cv_folds:
            raise ValueError("Kernel tuning needs at least two validation samples per fold.")
        cv = KFold(n_splits=self.cv_folds, shuffle=False)
        prepared = []
        for tr, va in cv.split(np.arange(n)):
            if self.cv_gap:
                tr = tr[(tr < va.min()-self.cv_gap) | (tr > va.max()+self.cv_gap)]
            if len(tr) < 3:
                raise ValueError("Too few kernel tuning samples after cv_gap exclusion.")
            fold = copy.deepcopy(self)
            def take(X, idx):
                return [f.isel(T=idx) for f in X] if isinstance(X, (list, tuple)) else X.isel(T=idx)
            Xt, Xv = take(Predictors, tr), take(Predictors, va)
            yt, yv = Predictand.isel(T=tr), Predictand.isel(T=va)
            # The optional upstream multivariate EOF belongs inside this fold too.
            if hasattr(fold, "_reduce_fit"):
                Xt = fold._reduce_fit(Xt)
                Xv = fold._reduce_apply(Xv)
            # compute_model/forecast scale the observed target before filling
            # gaps. Use that same order here: filling first shrinks a cell's
            # variance according to its missing-data frequency and changes the
            # PCA fitted by tuning relative to the prediction pipeline.
            yt = fold._ensure_tyx(fold._drop_member_dim(yt), name="y_train")
            yv = fold._ensure_tyx(fold._drop_member_dim(yv), name="y_validation")
            if fold.standardize:
                yt, ym, ys = fold._standardize_along_T(yt)
                yv = (yv - ym) / ys
            # The kernel's Y projection needs imputed inputs, but reconstructed
            # fields must only be scored against observations that exist.
            yobs, _ = fold._stack_spatial(yv)
            Xt, yt = fold.preprocess_data(Xt, yt)
            Xv = fold._fill_test_data(fold._ensure_tyx(fold._drop_member_dim(Xv)), Xt)
            yv = fold._fill_test_data(yv, yt)
            if fold.standardize:
                Xt, xm, xs = fold._standardize_along_T(Xt)
                Xv = (Xv - xm) / xs
            Xtr, Ytr = fold._fit_pca_views(Xt, yt, store=True)
            Xva = fold._transform_x_pca(Xv)
            yv_raw, _ = fold._stack_spatial(yv)
            if fold._y_pca is None:
                Yva = yv_raw
            elif fold.pca_backend == "sklearn":
                Yva = fold._y_pca.transform(yv_raw)
            else:
                scores = fold._y_pca.transform(fold._to_xeofs_dims(yv), normalized=False)
                Yva = fold._scores_to_numpy(scores.sel(mode=fold._y_pca_modes))
            Yreg = Ytr.copy()
            if fold.standardize:
                xm, xs = fold._column_stats(Xtr)
                ym, ys = fold._column_stats(Ytr)
                Xtr, Xva = (Xtr-xm)/xs, (Xva-xm)/xs
                Ytr, Yva = (Ytr-ym)/ys, (Yva-ym)/ys
            if min(Xtr.shape[1], Ytr.shape[1]) < self.n_modes:
                raise ValueError("Inner PCA rank is smaller than n_modes; lower n_modes.")
            prepared.append((Xtr, Ytr, Xva, Yva, Yreg, yobs, fold, yv["T"].values))
        return prepared

    def _score_inner_views(self, params, prepared):
        scores = []
        for Xtr, Ytr, Xva, Yva, Yreg, yobs, fold, times in prepared:
            try:
                model = self._model_from_params(params)
                model.fit([Xtr, Ytr])
                zv = model.transform([Xva, Yva])
                if self.tuning_objective == "canonical":
                    k = min(self.n_modes, zv[0].shape[1], zv[1].shape[1])
                    vals = [np.corrcoef(zv[0][:, i], zv[1][:, i])[0, 1] for i in range(k)]
                    scores.append(float(np.nanmean(vals)))
                else:
                    z = model.transform([Xtr, Ytr])[0]
                    zmean, mean = z.mean(axis=0), Yreg.mean(axis=0)
                    zc = z - zmean
                    B = np.linalg.solve(zc.T @ zc + self.reg*np.eye(z.shape[1]), zc.T @ (Yreg-mean))
                    pred = fold._inverse_y_pca((zv[0]-zmean) @ B + mean, times)
                    # Full reconstructed field: low-variance omitted modes still
                    # contribute error, unlike a score restricted to retained PCs.
                    valid = np.isfinite(yobs) & np.isfinite(pred)
                    scores.append(-float(np.mean((pred[valid]-yobs[valid])**2)) if valid.any() else np.nan)
            except (ValueError, np.linalg.LinAlgError, FloatingPointError):
                return -np.inf  # All candidates must be scored on identical folds.
        return float(np.mean(scores)) if scores and np.isfinite(scores).all() else -np.inf

    def compute_hyperparameters(self, Predictors, Predictand,
                                clim_year_start=None, clim_year_end=None):
        """Tune on supplied training data with fold-local EOF/PCA and scaling.

        Call inside each outer training fold for a nested performance estimate.
        ``tuning_score_`` is a model-selection score, never a final skill estimate.
        ``reconstruction`` optimizes negative MSE in standardized target units.
        """
        if not HAS_CCA_ZOO:
            raise ImportError("cca-zoo is required for hyperparameter tuning.")
        if self.search_method not in {"grid", "bayesian"}:
            self.best_params, self.is_tuned = {}, True
            return {}
        grid = self._prepare_param_grid()
        if self.search_method == "grid" and not grid:
            self.best_params, self.is_tuned = {}, True
            return {}
        prepared = self._prepare_inner_views(Predictors, Predictand)
        if self.search_method == "grid":
            from sklearn.model_selection import ParameterGrid
            results = [(params, self._score_inner_views(params, prepared)) for params in ParameterGrid(grid)]
            params, score = max(results, key=lambda r: r[1])
            if not np.isfinite(score):
                raise RuntimeError("No kernel candidate succeeded on every inner fold.")
            self.best_params, self.tuning_score_ = params, score
            self.tuning_results_ = results
        else:
            try:
                import optuna
            except ImportError as exc:
                raise ImportError("Optuna is required for Bayesian optimization.") from exc
            study = optuna.create_study(direction="maximize", sampler=optuna.samplers.TPESampler(seed=self.random_state))
            study.optimize(lambda trial: self._score_inner_views(self._params_for_trial(trial), prepared), n_trials=50)
            if not np.isfinite(study.best_value):
                raise RuntimeError("No kernel candidate succeeded on every inner fold.")
            self.best_params, self.tuning_score_ = study.best_params, study.best_value
        self.is_tuned = True
        self.canonical_correlations_cv_ = None
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

        X_np_raw, X_coords = self._stack_spatial(X_train_final)
        Y_np_raw, Y_coords = self._stack_spatial(y_train_final)

        self.X_coords = X_coords
        self.Y_coords = Y_coords
        self.train_time = X_train_final.coords["T"]

        # PCA is fitted on the TRAIN fold only. CCA/KCCA/KGCCA sees the reduced
        # X/Y views; Y is reconstructed back to the original grid after prediction.
        X_np, Y_np = self._fit_pca_views(
            X_train_final, y_train_final, X_np=X_np_raw, Y_np=Y_np_raw, store=True
        )
        # Kernel space: both reduced views scaled per feature on this fold.
        # The scores -> field regression below still targets the UNSCALED Y
        # view, so the reconstruction path and _inverse_y_pca are untouched.
        if self.standardize:
            self._zx_mean, self._zx_std = self._column_stats(X_np)
            self._zy_mean, self._zy_std = self._column_stats(Y_np)
        else:
            self._zx_mean = self._zx_std = self._zy_mean = self._zy_std = None
        Xk = self._scale_reduced(X_np, self._zx_mean, self._zx_std)
        Yk = self._scale_reduced(Y_np, self._zy_mean, self._zy_std)

        # Diagnostics operate in the space the kernel actually sees.
        self.X_train_np = Xk
        self.Y_train_np = Yk
        self._Y_reduced_train = Y_np
        self._check_feature_ratio(X_np.shape[0], X_np.shape[1], Y_np.shape[1])

        if min(X_np.shape[1], Y_np.shape[1]) < self.n_modes:
            raise ValueError(
                f"n_modes={self.n_modes} is larger than the PCA-reduced view "
                f"dimensions X={X_np.shape[1]}, Y={Y_np.shape[1]}. "
                "Lower n_modes or pca_min_components, or disable PCA."
            )

        self.model = self._build_model(params)
        self.model.fit([Xk, Yk])
        self.is_fitted = True

        # Canonical scores on the training data (both views present).
        scores = self.model.transform([Xk, Yk])
        self.zx_train, self.zy_train = scores[0], scores[1]

        # In-sample canonical correlations. Kept because plotting and external
        # callers expect them, but see diagnose_memorisation: at these sample
        # sizes they are close to 1 by construction.
        n_latent = min(self.n_modes, self.zx_train.shape[1], self.zy_train.shape[1])
        self.canonical_correlations = [
            float(np.corrcoef(self.zx_train[:, i], self.zy_train[:, i])[0, 1])
            for i in range(n_latent)
        ]
        # A fresh fit invalidates any previously computed CV counterpart.
        self.canonical_correlations_cv_ = None

        # Regression: X canonical scores -> reduced Y view (ridge-stabilized).
        # If Y-PCA is active, _predict_field inverse-transforms this back to the
        # original spatial grid.
        self._reg_mean = Y_np.mean(axis=0)
        Yc = Y_np - self._reg_mean
        self._reg_zmean = self.zx_train.mean(axis=0)
        Z = self.zx_train - self._reg_zmean
        G = Z.T @ Z + self.reg * np.eye(Z.shape[1])
        self._reg_B = np.linalg.solve(G, Z.T @ Yc)
        return self

    def _x_scores(self, X_np):
        """Transform a predictor matrix to X canonical scores.

        cca-zoo requires a value for every view and rejects None, so a zero
        matrix stands in for the unobserved predictand. Verified on cca-zoo
        3.1.1: the X scores are identical whether the second view is zeros or
        random data, because each view is projected independently.
        """
        n = X_np.shape[0]
        Xk = self._scale_reduced(X_np, self._zx_mean, self._zx_std)
        dummy = np.zeros((n, self.Y_train_np.shape[1]))
        return self.model.transform([Xk, dummy])[0]

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
        X_np_raw, _ = self._stack_spatial(X_da)
        X_np = self._transform_x_pca(X_da, X_np=X_np_raw)
        zx = self._x_scores(X_np)
        Y_np_reduced = (zx - self._reg_zmean) @ self._reg_B + self._reg_mean
        Y_np = self._inverse_y_pca(Y_np_reduced, X_da["T"].values)
        return self._unstack_spatial(Y_np, self.Y_coords, X_da["T"].values)

    def compute_model(self, X_train, y_train, X_test, y_test=None, best_params=None):
        """
        Fit on the training fold and predict the test fold. Returns the
        deterministic hindcast in the original predictand scale (T, Y, X).

        The predictand is standardized with the statistics of the training fold
        only, and forecast() applies the same rule with the whole record as its
        training set. The two therefore live in the same normalisation, which
        was not the case before: compute_model used fold statistics while
        forecast used the climatology window, so the residual sample produced
        here described a differently scaled model from the one being dressed.
        """
        if self.standardize:
            y_train_st, y_mean, y_std = self._standardize_along_T(
                self._ensure_tyx(self._drop_member_dim(y_train), name="y_train")
            )
            self._y_mean, self._y_std = y_mean, y_std
            self.fit_cca(X_train, y_train_st, best_params=best_params)
        else:
            self._y_mean, self._y_std = None, None
            self.fit_cca(X_train, y_train, best_params=best_params)

        X_test_prepared, y_test_prepared = self.preprocess_test_data(
            X_test, y_test, X_train, y_train
        )
        y_pred = self._predict_field(X_test_prepared)
        if y_test_prepared is not None:
            y_pred = y_pred.assign_coords(T=y_test_prepared["T"])

        if self.standardize:
            y_pred = self._reverse_along_T(y_pred, self._y_mean, self._y_std)
        return y_pred

    # ------------------------------------------------------------------ #
    # Real-time forecast
    # ------------------------------------------------------------------ #

    def forecast(self, Predictant, clim_year_start, clim_year_end, Predictor,
                 hindcast_det, Predictor_for_year, best_params=None,
                 best_code_da=None, best_shape_da=None,
                 best_loc_da=None, best_scale_da=None):
        """Deterministic and probabilistic forecast.

        clim_year_start / clim_year_end define the tercile climatology only. The
        predictand standardization uses the statistics of the fitting set (the
        whole record here), matching what compute_model does per fold.
        """
        Predictant = self._ensure_tyx(self._drop_member_dim(Predictant), name="Predictant")
        Predictor = self._ensure_tyx(self._drop_member_dim(Predictor), name="Predictor")
        hindcast_det = self._ensure_tyx(self._drop_member_dim(hindcast_det),
                                        name="hindcast_det")
        mask = self._spatial_mask(Predictant)

        if self.standardize:
            Predictant_fit, self._y_mean, self._y_std = self._standardize_along_T(Predictant)
        else:
            Predictant_fit = Predictant
            self._y_mean, self._y_std = None, None

        self.fit_cca(Predictor, Predictant_fit, best_params=best_params)

        X_forecast_prepared = self.preprocess_forecast_data(
            Predictor_for_year=Predictor_for_year, Predictor_train=Predictor
        )
        forecast_det = self._predict_field(X_forecast_prepared)

        if self.standardize:
            forecast_det = self._reverse_along_T(forecast_det, self._y_mean, self._y_std)

        forecast_time = self._make_forecast_time(Predictant, Predictor_for_year)
        forecast_det = forecast_det.assign_coords(
            T=xr.DataArray(forecast_time, dims=["T"])
        ).transpose("T", "Y", "X")

        # Same calibrator as the strict cross-validator and the linear CCA
        # classes (``_ProbabilityEngine._calibrated_probabilities``): the
        # out-of-sample residuals ``obs - hindcast`` give a bias-corrected
        # centre and the residual variance (ddof=1); the deterministic field
        # returned is that calibrated mean, clipped by the declared support.
        if hindcast_det.sizes["T"] == Predictant.sizes["T"]:
            hindcast_det = hindcast_det.assign_coords(T=Predictant["T"])
        forecast_det, forecast_prob = self._calibrated_probabilities(
            forecast_det, Predictant, Predictant - hindcast_det,
            clim_year_start, clim_year_end, mask,
            best_code_da, best_shape_da, best_loc_da, best_scale_da,
        )

        # _calibrated_probabilities already masks and renormalizes.
        return forecast_det * mask, forecast_prob

    # ------------------------------------------------------------------ #
    # Introspection + plotting
    # ------------------------------------------------------------------ #

    def get_components(self):
        """Spatial patterns (linear kernel only).

        With a PCA pre-reduction the CCA weights live in PC space; they are
        mapped back to the grid through the PCA loadings, which is exact for a
        linear kernel. Returns (None, None) for non-linear kernels, where no
        such mapping exists, and when the mapping cannot be formed.
        """
        if not self.is_fitted:
            raise ValueError("Model not fitted.")
        if not self._is_linear_kernel():
            print("Warning: components are not interpretable for non-linear kernels.")
            return None, None

        weights = self.model.weights
        wX, wY = np.asarray(weights[0]), np.asarray(weights[1])  # (features, modes)

        # Backend-aware loadings. The previous code used the sklearn-only
        # attribute ``components_`` and therefore raised AttributeError under
        # the default xeofs backend.
        LX = self._pca_components_matrix("x")
        LY = self._pca_components_matrix("y")
        if LX is not None:
            wX = LX @ wX
        if LY is not None:
            wY = LY @ wY

        if wX.shape[0] != len(self.X_coords) or wY.shape[0] != len(self.Y_coords):
            print("Warning: CCA weights cannot be mapped back to the grid with "
                  "the current PCA configuration; skipping spatial modes.")
            return None, None

        modes = np.arange(1, wX.shape[1] + 1)
        X_comp = self._unstack_spatial(wX.T, self.X_coords, modes, time_name="mode")
        Y_comp = self._unstack_spatial(wY.T, self.Y_coords, modes, time_name="mode")
        return X_comp, Y_comp

    def get_scores(self):
        if not self.is_fitted:
            raise ValueError("Model not fitted.")
        modes = list(range(1, self.zx_train.shape[1] + 1))
        X_scores = xr.DataArray(self.zx_train, dims=["T", "mode"],
                                coords={"T": self.train_time, "mode": modes})
        Y_scores = xr.DataArray(self.zy_train, dims=["T", "mode"],
                                coords={"T": self.train_time, "mode": modes})
        return X_scores, Y_scores

    def get_canonical_correlations(self, cross_validated=False):
        """In-sample canonical correlations, or their CV counterpart.

        Prefer ``cross_validated=True``. The in-sample values approach 1 as soon
        as the feature count approaches the number of years.
        """
        if not self.is_fitted:
            raise ValueError("Model not fitted.")
        if not cross_validated:
            return self.canonical_correlations
        if self.canonical_correlations_cv_ is None:
            self.diagnose_memorisation(verbose=False)
        return self.canonical_correlations_cv_

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
            "pca_explained_variance": self.pca_explained_variance,
            "pca_min_components": self.pca_min_components,
            "pca_max_feature_ratio": self.pca_max_feature_ratio,
            "pca_whiten": self.pca_whiten,
            "pca_backend": self.pca_backend,
            "pca_use_coslat": self.pca_use_coslat,
            "pca_solver": self.pca_solver,
            "tuning_objective": self.tuning_objective,
        }
        if self.is_fitted:
            info["canonical_correlations"] = self.canonical_correlations
            info["canonical_correlations_cv"] = self.canonical_correlations_cv_
            info["training_samples"] = self.X_train_np.shape[0]
            info["X_features_raw"] = self._X_raw_n_features
            info["Y_features_raw"] = self._Y_raw_n_features
            info["X_features_cca"] = self.X_train_np.shape[1]
            info["Y_features_cca"] = self.Y_train_np.shape[1]
            # The single most informative number for judging whether this fit
            # can be believed at all.
            info["feature_to_sample_ratio"] = round(
                max(self.X_train_np.shape[1], self.Y_train_np.shape[1])
                / max(1, self.X_train_np.shape[0]), 3
            )
            info["X_pca_variance_kept"] = self._x_pca_variance
            info["Y_pca_variance_kept"] = self._y_pca_variance
        return info

    def plot_cca_results(self, X=None, Y=None, n_modes=None,
                         clim_year_start=None, clim_year_end=None, best_params=None,
                         show_cv_correlations=True):
        """Plot canonical scores (always) and spatial modes (linear kernel).

        Titles carry the cross-validated canonical correlation next to the
        in-sample one. Reporting only the latter, as the previous version did,
        invited reading a memorised r = 0.98 as physical coupling.
        """
        mask = None
        if X is not None and Y is not None:
            X = self._ensure_tyx(self._drop_member_dim(X), name="X")
            Y = self._ensure_tyx(self._drop_member_dim(Y), name="Y")
            # Same standardization convention as compute_model / forecast.
            if self.standardize:
                Y_fit, _, _ = self._standardize_along_T(Y)
            else:
                Y_fit = Y
            self.fit_cca(X, Y_fit, best_params=best_params)
            mask = self._spatial_mask(Y)
        elif not self.is_fitted:
            raise ValueError("Model not fitted. Provide X and Y or call fit_cca first.")

        cv_r = None
        if show_cv_correlations:
            try:
                _, cv_r = self.diagnose_memorisation(verbose=False)
            except Exception:
                cv_r = None

        X_scores, Y_scores = self.get_scores()
        X_comp, Y_comp = self.get_components()
        linear = X_comp is not None

        available = int(X_scores.sizes["mode"])
        n_modes = available if n_modes is None else max(1, min(int(n_modes), available))

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
            title = f"Scores Mode {mode} (r = {cc:.2f} in-sample"
            if cv_r is not None and mode - 1 < len(cv_r):
                title += f", {cv_r[mode - 1]:.2f} CV"
            ax.set_title(title + ")")
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
    """Kernel Canonical Correlation Analysis (cca-zoo KCCA / LinearCCA).

    On a raw spatial grid this model memorises: with more grid points than years
    the in-sample canonical correlations are ~1 regardless of any physical
    relationship. Keep the PCA pre-reduction on, or use WAS_EOF_KCCA.
    """

    def _build_model(self, params):
        if self._is_linear_kernel():
            # LinearCCA has no kernel regularisation parameter; c applies to the
            # kernel path only.
            return LinearCCA(latent_dimensions=self.n_modes)
        return KCCA(latent_dimensions=self.n_modes, kernel=self.kernel, **params)

    def _grid_estimator(self):
        if self._is_linear_kernel():
            return LinearCCA(latent_dimensions=self.n_modes)
        return KCCA(latent_dimensions=self.n_modes, kernel=self.kernel)

    def _prepare_param_grid(self, kernel=None):
        """Search grid, now including the Tikhonov regularisation ``c``.

        cca-zoo's signature is
        ``KCCA(latent_dimensions, center, c, kernel, gamma, degree, coef0, ...)``
        with ``c=0.1`` by default. The previous grid never touched it, so every
        fit ran at that default. Measured caveat: ``c`` only regularises
        effectively once the feature count is well below the sample size (see
        the module docstring), so this grid complements the PCA cap rather than
        replacing it.
        """
        kernel = kernel or self.kernel
        c_grid = [0.01, 0.1, 0.5, 1.0]
        if kernel == "rbf":
            return {"gamma": [0.01, 0.05, 0.1, 0.5, 1.0, 2.0, 5.0], "c": c_grid}
        if kernel == "poly":
            return {"degree": [2, 3, 4], "gamma": [0.1, 0.5, 1.0, 2.0],
                    "coef0": [0, 1, 2], "c": c_grid}
        if kernel == "sigmoid":
            return {"gamma": [0.01, 0.05, 0.1, 0.5, 1.0], "coef0": [0, 1, 2],
                    "c": c_grid}
        return {}  # linear: LinearCCA exposes no tunable parameter here

    def _params_for_trial(self, trial):
        if self.kernel == "rbf":
            return {"gamma": trial.suggest_float("gamma", 0.01, 5.0, log=True),
                    "c": trial.suggest_float("c", 1e-3, 2.0, log=True)}
        if self.kernel == "poly":
            return {"degree": trial.suggest_int("degree", 2, 4),
                    "gamma": trial.suggest_float("gamma", 0.1, 5.0, log=True),
                    "coef0": trial.suggest_float("coef0", 0, 5),
                    "c": trial.suggest_float("c", 1e-3, 2.0, log=True)}
        if self.kernel == "sigmoid":
            return {"gamma": trial.suggest_float("gamma", 0.01, 1.0, log=True),
                    "coef0": trial.suggest_float("coef0", 0, 5),
                    "c": trial.suggest_float("c", 1e-3, 2.0, log=True)}
        return {}

    def _model_from_params(self, params):
        if self._is_linear_kernel():
            return LinearCCA(latent_dimensions=self.n_modes)
        return KCCA(latent_dimensions=self.n_modes, kernel=self.kernel, **params)


# ===========================================================================
# Concrete model 2: Kernel Generalized CCA
# ===========================================================================
class WAS_KGCCA(WAS_KernelCCA_base):
    """Kernel Generalized CCA (cca-zoo KGCCA / GCCA).

    KGCCA takes per-view lists for its kernel parameters, hence the ``[v, v]``
    shape of every grid entry below.
    """

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
        c_grid = [[0.01, 0.01], [0.1, 0.1], [0.5, 0.5], [1.0, 1.0]]
        if kernel == "rbf":
            return {"gamma": [[0.01, 0.01], [0.1, 0.1], [1.0, 1.0], [10.0, 10.0]],
                    "c": c_grid}
        if kernel == "poly":
            return {"degree": [[2, 2], [3, 3]], "gamma": [[0.1, 0.1], [1.0, 1.0]],
                    "coef0": [[0, 0], [1, 1]], "c": c_grid}
        if kernel == "sigmoid":
            return {"gamma": [[0.01, 0.01], [0.1, 0.1], [1.0, 1.0]],
                    "coef0": [[0, 0], [1, 1]], "c": c_grid}
        return {"c": c_grid}  # linear GCCA does expose c

    def _params_for_trial(self, trial):
        if self.kernel == "rbf":
            g = trial.suggest_float("gamma", 0.01, 10.0, log=True)
            c = trial.suggest_float("c", 1e-3, 2.0, log=True)
            return {"gamma": [g, g], "c": [c, c]}
        if self.kernel == "poly":
            d = trial.suggest_int("degree", 2, 4)
            g = trial.suggest_float("gamma", 0.1, 5.0, log=True)
            c0 = trial.suggest_float("coef0", 0, 5)
            c = trial.suggest_float("c", 1e-3, 2.0, log=True)
            return {"degree": [d, d], "gamma": [g, g], "coef0": [c0, c0], "c": [c, c]}
        if self.kernel == "sigmoid":
            g = trial.suggest_float("gamma", 0.01, 1.0, log=True)
            c0 = trial.suggest_float("coef0", 0, 5)
            c = trial.suggest_float("c", 1e-3, 2.0, log=True)
            return {"gamma": [g, g], "coef0": [c0, c0], "c": [c, c]}
        c = trial.suggest_float("c", 1e-3, 2.0, log=True)
        return {"c": [c, c]}

    def _model_from_params(self, params):
        if self._is_linear_kernel():
            return GCCA(latent_dimensions=self.n_modes, **params)
        return KGCCA(latent_dimensions=self.n_modes, kernel=self.kernel, **params)


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

Sizing the reduction
--------------------
The total predictor width is n_fields * eof_n_modes when joint=False. That total,
not eof_n_modes, is what has to stay well below the number of years: five fields
at five PCs each is 25 features, which on a 30-year hindcast puts the kernel back
in the memorisation regime. The width is checked at fit time and a warning is
raised.
"""


class WAS_EOF_KCCA(WAS_KCCA):

    def __init__(self, eof_n_modes=5, joint=False, eof_detrend=None,
                 eof_standardize=False, eof_use_coslat=True, *, detrend=None,
                 **kcca_kwargs):
        """
        eof_n_modes      PCs kept per field (joint=False) or total joint modes
                         (joint=True).
        joint            False -> one EOF per field then concat PCs (default).
                         True  -> one multivariate EOF over all fields.
        detrend          Remove predictor trends in the upstream EOFs; defaults
                         to True. False disables trend fitting and application
                         in tuning, validation and forecasting. The predictand
                         is not detrended by this model.
        eof_detrend      Legacy alias for detrend, including positional calls.
                         Explicitly conflicting aliases raise ValueError.
        eof_standardize / eof_use_coslat
                         passed through to WAS_EOF.
        **kcca_kwargs    forwarded to WAS_KCCA / WAS_KernelCCA_base (n_modes, kernel,
                         dist_method, standardize, reg, search_method, ...).
        """
        for name, value in (("detrend", detrend), ("eof_detrend", eof_detrend)):
            if value is not None and not isinstance(value, (bool, np.bool_)):
                raise TypeError(f"{name} must be a boolean or None.")
        if detrend is not None and eof_detrend is not None and detrend != eof_detrend:
            raise ValueError("detrend and eof_detrend must agree when both are supplied.")
        resolved_detrend = detrend if detrend is not None else eof_detrend
        super().__init__(**kcca_kwargs)
        self.eof_n_modes = eof_n_modes
        self.joint = joint
        self.detrend = True if resolved_detrend is None else resolved_detrend
        self.eof_standardize = eof_standardize
        self.eof_use_coslat = eof_use_coslat
        self._eofs = None      # list of fitted WAS_EOF (1 if joint, else 1/field)

    # ------------------------------------------------------------------ utils
    @property
    def detrend(self):
        """Canonical alias for the EOF setting, applied on each new fit."""
        return self.eof_detrend

    @detrend.setter
    def detrend(self, value):
        if not isinstance(value, (bool, np.bool_)):
            raise TypeError("detrend must be a boolean.")
        self.eof_detrend = bool(value)

    @staticmethod
    def _as_field_list(X):
        return list(X) if isinstance(X, (list, tuple)) else [X]

    @staticmethod
    def _drop_M(f):
        return f.isel(M=0, drop=True) if "M" in f.dims else f

    def _new_eof(self):
        return WAS_EOF(n_modes=self.eof_n_modes, detrend=self.eof_detrend,  # noqa: F405
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

    def _warn_total_width(self, n_fields, n_samples):
        """The concatenated width, not eof_n_modes, is what must stay small."""
        if not self.warn_memorisation:
            return
        width = self.eof_n_modes if self.joint else self.eof_n_modes * int(n_fields)
        if n_samples > 0 and width > 0.5 * n_samples:
            warnings.warn(
                f"EOF pre-reduction yields {width} predictor features "
                f"({n_fields} field(s) x {self.eof_n_modes} PCs) for {n_samples} "
                "samples. Lower eof_n_modes, drop a field, or set joint=True.",
                RuntimeWarning, stacklevel=3,
            )

    # ------------------------------------------------- fold-safe EOF reduction
    def _reduce_fit(self, X):
        """Fit EOF(s) on this fold's predictor and return the pseudo predictor."""
        fields = [self._drop_M(f) for f in self._as_field_list(X)]
        self._warn_total_width(len(fields), int(fields[0].sizes["T"]))
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
    def compute_model(self, X_train, y_train, X_test, y_test=None, best_params=None):
        pcs_train = self._reduce_fit(X_train)     # EOF fit on TRAIN fold (fold-safe)
        pcs_test = self._reduce_apply(X_test)     # TEST projected onto TRAIN EOFs
        return super().compute_model(pcs_train, y_train, pcs_test, y_test,
                                     best_params=best_params)

    def compute_hyperparameters(self, Predictors, Predictand,
                                clim_year_start=None, clim_year_end=None):
        return super().compute_hyperparameters(Predictors, Predictand,
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


# Keep constructor inputs for fresh model instances in nested cross-validation.
from .was_crossfit_adapters import capture_constructors as _capture_constructors
_capture_constructors(globals())
