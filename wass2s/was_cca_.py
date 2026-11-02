"""Canonical Correlation Analysis (CCA) for seasonal forecasting — WASS2S.

Four CCA implementations share a single preprocessing, probability and plotting
engine, and differ **only** by the transformation applied to the predictor and
the predictand:

``WAS_CCA``
    Leading EEOF component removed from X and from the standardized Y.
    The component is persisted at ``T[-eeof_ref_offset]`` for the forecast year,
    on both sides, so fit space and predict space coincide.

``WAS_CCA_base``
    No detrending. X raw, Y standardized over the climatology.

``WAS_CCA_op``
    Linear detrending of X, standardization + linear detrending of Y.

``WAS_CCA_strict``
    Same transformations as ``op``, plus strict spatial gap filling; intended to
    be driven by a cross-validator that refits every transformation per fold.

All classes expose ``compute_model``, ``compute_prob``, ``forecast``,
``plot_cca_results`` and ``tune``.

Changes with respect to the previous revision
---------------------------------------------
* ``tune_truncation`` selects ``(n_modes, n_pca_modes_X, n_pca_modes_Y)`` by
  cross-validation with a one-standard-error parsimony rule (CPT-like).
  ``xe.cross.CCA`` accepts a 2-sequence for ``n_pca_modes``, so X and Y
  truncations are selected independently (verified on xeofs 3.0.4).
* ``WAS_CCA``: the forecast-year predictor is now moved into the same EEOF
  residual space as the training predictor, and the predictand EEOF component
  is added back before ``reverse_standardize``. Previously neither was done.
* ``_spatial_mask`` no longer keys off ``T=0`` alone.
* Predictive spread uses the mean squared error, not the residual variance, so
  a biased hindcast is no longer given an artificially narrow distribution.
* Non-negativity clipping is conditional on the sign of the observations and is
  exposed as a module-level function so the cross-validator applies the very
  same rule to hindcasts.
* Effective degrees of freedom account for the modes consumed by the CCA.
* Discrete families (Poisson, negative binomial) are refused for continuous
  predictands instead of being evaluated at the integer floor of the tercile.
* The duplicated ``@staticmethod`` decorators and the four copies of the
  probability code are gone.
"""

from __future__ import annotations

import os
import warnings
from concurrent.futures import ThreadPoolExecutor

import numpy as np
import xarray as xr
import matplotlib.pyplot as plt

from scipy.stats import gamma, norm, lognorm, expon, weibull_min, t, poisson, nbinom
from scipy.optimize import brentq
from scipy.special import gamma as gamma_function

import xeofs as xe

from wass2s.utils import *  # noqa: F401,F403


__all__ = [
    "WAS_CCA",
    "WAS_CCA_base",
    "WAS_CCA_op",
    "WAS_CCA_strict",
    "tune_truncation",
    "TruncationResult",
    "conditional_nonnegative_clip",
    "spatial_mask",
    "check_tercile_calibration",
]


# ---------------------------------------------------------------------------
# Grid-parallel apply_ufunc
# ---------------------------------------------------------------------------
def _grid_workers():
    try:
        n = int(os.environ.get("WAS_NCORES", "0"))
        if n <= 0:
            n = os.cpu_count() or 1
        return max(1, n)
    except Exception:
        return 1


def _pufunc(func, *args, **kwargs):
    """Drop-in ``xr.apply_ufunc`` that chunks along Y and runs on threads.

    Chunking changes execution, never values. Disable with WAS_GRID_PARALLEL=0.
    """
    if os.environ.get("WAS_GRID_PARALLEL", "1") == "0":
        return xr.apply_ufunc(func, *args, **kwargs)
    n = _grid_workers()
    if n <= 1:
        return xr.apply_ufunc(func, *args, **kwargs)
    try:
        new_args, chunked = [], False
        for a in args:
            if hasattr(a, "dims") and "Y" in getattr(a, "dims", ()):
                ny = int(a.sizes.get("Y", 1))
                if ny >= 2 * n:
                    a = a.chunk({"Y": max(1, ny // n)})
                    chunked = True
            new_args.append(a)
        if not chunked:
            return xr.apply_ufunc(func, *args, **kwargs)
        res = xr.apply_ufunc(func, *new_args, **kwargs)
        return res.compute(scheduler="threads", num_workers=n)
    except Exception:
        return xr.apply_ufunc(func, *args, **kwargs)


_DISTRIBUTION_CODES = {
    "normal": 1, "norm": 1, "gaussian": 1,
    "lognormal": 2, "lognorm": 2,
    "exponential": 3, "expon": 3,
    "gamma": 4,
    "weibull": 5, "weibull_min": 5,
    "t": 6, "student_t": 6, "student-t": 6,
    "poisson": 7,
    "nbinom": 8, "negative_binomial": 8, "negative-binomial": 8,
}

_DISCRETE_CODES = (7, 8)


# ---------------------------------------------------------------------------
# Field helpers
# ---------------------------------------------------------------------------
def _prepare_tyx(da: xr.DataArray, name: str) -> xr.DataArray:
    """Validate a WAS field and return it in ``(T, Y, X)`` order."""
    if not isinstance(da, xr.DataArray):
        raise TypeError(f"{name} must be an xarray.DataArray, got {type(da)!r}.")

    if "M" in da.dims:
        if da.sizes["M"] < 1:
            raise ValueError(f"{name} has an empty M dimension.")
        da = da.isel(M=0, drop=True)

    extra_dims = [d for d in da.dims if d not in ("T", "Y", "X")]
    singleton = [d for d in extra_dims if da.sizes[d] == 1]
    if singleton:
        da = da.squeeze(singleton, drop=True)

    missing = [d for d in ("T", "Y", "X") if d not in da.dims]
    extra = [d for d in da.dims if d not in ("T", "Y", "X")]
    if missing or extra:
        raise ValueError(
            f"{name} must have dimensions T, Y, X after removing M. "
            f"Missing={missing}, extra={extra}, received={da.dims}."
        )
    return da.transpose("T", "Y", "X")


def _to_latlon(da: xr.DataArray) -> xr.DataArray:
    rename = {}
    if "Y" in da.dims:
        rename["Y"] = "lat"
    if "X" in da.dims:
        rename["X"] = "lon"
    if rename:
        da = da.rename(rename)
    order = [d for d in ("T", "lat", "lon") if d in da.dims]
    return da.transpose(*(order + [d for d in da.dims if d not in order]))


def _to_yx(da: xr.DataArray) -> xr.DataArray:
    rename = {}
    if "lat" in da.dims:
        rename["lat"] = "Y"
    if "lon" in da.dims:
        rename["lon"] = "X"
    if rename:
        da = da.rename(rename)
    order = [d for d in ("T", "Y", "X") if d in da.dims]
    return da.transpose(*(order + [d for d in da.dims if d not in order]))


def _as_bool(value, default=False) -> bool:
    try:
        return bool(np.asarray(value).item())
    except Exception:
        return default


def spatial_mask(Predictant: xr.DataArray, min_valid: int = 3) -> xr.DataArray:
    """(Y, X) mask of cells usable as forecast targets.

    A cell is retained when it has at least ``min_valid`` finite observations —
    the minimum needed to estimate terciles and a residual spread. Keying the
    mask off ``T=0`` alone, as before, discarded every cell whose first year
    happened to be missing.
    """
    n_ok = np.isfinite(Predictant).sum(dim="T")
    return xr.where(n_ok >= int(min_valid), 1.0, np.nan)


# Backwards-compatible private alias.
_spatial_mask = spatial_mask


def _coordinates_compatible(reference, candidate, dim: str) -> bool:
    if reference.sizes[dim] != candidate.sizes[dim]:
        return False
    ref = np.asarray(reference[dim].values)
    cand = np.asarray(candidate[dim].values)
    if np.array_equal(ref, cand):
        return True
    if np.issubdtype(ref.dtype, np.number) and np.issubdtype(cand.dtype, np.number):
        return np.allclose(ref, cand, equal_nan=True)
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
        if _as_bool(reordered.isnull().all(), default=False):
            raise ValueError(
                f"Predictant and hindcast_det have no compatible {dim} coordinates."
            )
        hindcast_det = reordered

    if np.array_equal(Predictant["T"].values, hindcast_det["T"].values):
        return Predictant, hindcast_det

    # Cross-validation outputs are commonly positionally correct but carry a
    # synthetic or integer T coordinate.
    if Predictant.sizes["T"] == hindcast_det.sizes["T"]:
        hindcast_det = hindcast_det.assign_coords(T=Predictant["T"])
        return Predictant, hindcast_det

    common = np.intersect1d(Predictant["T"].values, hindcast_det["T"].values)
    if common.size < 2:
        raise ValueError(
            "Predictant and hindcast_det do not have enough common T coordinates "
            "to estimate forecast residuals."
        )
    return Predictant.sel(T=common), hindcast_det.sel(T=common)


def _fill_spatial_gaps_safe(da: xr.DataArray) -> xr.DataArray:
    """Forward/backward spatial filling, avoiding a hard bottleneck dependency."""
    if not _as_bool(da.isnull().any(), default=True):
        return da
    out = da
    try:
        for dim in ("Y", "X", "lat", "lon"):
            if dim in out.dims:
                out = out.ffill(dim=dim).bfill(dim=dim)
    except ModuleNotFoundError:
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
    """Shared (Y, X) validity mask for a field fed to xeofs.

    A cell is kept when it is finite at every training step AND not temporally
    constant. Failing cells are excluded identically in training, validation and
    forecast inputs, so xeofs drops the same features on both sides instead of
    ingesting constant-zero fills.
    """
    finite_all = np.isfinite(field_imputed).all(dim="T")
    non_constant = field_imputed.std(dim="T", skipna=True) > 0
    mask = finite_all & non_constant
    if not _as_bool(mask.any(), default=True):
        mask = finite_all
    return mask


def conditional_nonnegative_clip(
    deterministic: xr.DataArray, Predictant: xr.DataArray
) -> xr.DataArray:
    """Clip at zero only when the observations are themselves non-negative.

    Use this in the cross-validator too. Clipping hindcasts unconditionally
    while clipping forecasts conditionally biases ``error_samples`` and shifts
    the whole predictive distribution.
    """
    try:
        value = float(np.asarray(Predictant.min(skipna=True).values))
    except Exception:
        return deterministic
    if np.isfinite(value) and value >= 0.0:
        return deterministic.clip(min=0.0)
    return deterministic


_conditional_nonnegative_clip = conditional_nonnegative_clip


def _target_time(Predictant: xr.DataArray, Predictor_for_year: xr.DataArray):
    year = int(Predictor_for_year["T"].dt.year.values[0])
    month = int(Predictant["T"].dt.month.values[0])
    return np.array([np.datetime64(f"{year:04d}-{month:02d}-01")], dtype="datetime64[ns]")


def _predictive_error_stats(observed: xr.DataArray, deterministic: xr.DataArray):
    """Residual sample and predictive spread.

    The spread is the **mean squared error**, not the residual variance: a
    hindcast with a systematic bias must not be given a narrow predictive
    distribution centred on that bias.
    """
    resid = observed - deterministic
    mse = (resid ** 2).mean(dim="T", skipna=True)
    return resid, mse


def _effective_dof(n_clim: int, n_modes: int) -> int:
    """Residual degrees of freedom after the CCA has consumed ``n_modes``."""
    return int(max(int(n_clim) - int(n_modes) - 1, 3))


def _looks_discrete(da: xr.DataArray, sample: int = 5000) -> bool:
    values = np.asarray(da.values).ravel()
    values = values[np.isfinite(values)]
    if values.size == 0:
        return False
    if values.size > sample:
        values = values[:: max(1, values.size // sample)]
    return bool(np.allclose(values, np.round(values)))


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


def _solve_weibull_shape(M: float, V: float) -> float:
    """Solve the Weibull shape from a positive mean and variance."""
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


def check_tercile_calibration(
    Predictant: xr.DataArray, T1: xr.DataArray, T2: xr.DataArray,
    clim_year_start=None, clim_year_end=None,
) -> xr.Dataset:
    """Observed frequency of each category defined by (T1, T2).

    With ``dist_method='bestfit'`` the thresholds come from a fitted family
    while verification uses the observations. If this diagnostic departs from
    1/3 the probabilistic forecasts carry a structural reliability bias that no
    amount of renormalization will remove.
    """
    obs = _prepare_tyx(Predictant, "Predictant")
    if clim_year_start is not None and clim_year_end is not None:
        obs = obs.sel(T=slice(str(clim_year_start), str(clim_year_end)))
    T1 = T1.drop_vars("quantile", errors="ignore")
    T2 = T2.drop_vars("quantile", errors="ignore")
    n = np.isfinite(obs).sum(dim="T")
    fb = (obs < T1).sum(dim="T") / n
    fn = ((obs >= T1) & (obs < T2)).sum(dim="T") / n
    fa = (obs >= T2).sum(dim="T") / n
    return xr.Dataset({"freq_below": fb, "freq_near": fn, "freq_above": fa})


# ---------------------------------------------------------------------------
# Truncation handling
# ---------------------------------------------------------------------------
def _resolve_pca_modes(n_pca_modes):
    """Return ``(n_x, n_y)``.

    ``xe.cross.CCA`` accepts ``Sequence[int | float | str]`` for
    ``n_pca_modes``, so X and Y truncations can differ.
    """
    if isinstance(n_pca_modes, (list, tuple, np.ndarray)):
        seq = list(n_pca_modes)
        if len(seq) != 2:
            raise ValueError("n_pca_modes as a sequence must have exactly 2 entries (X, Y).")
        return seq[0], seq[1]
    return n_pca_modes, n_pca_modes


def _safe_mode_counts(n_modes, n_pca_modes, n_samples):
    """Limit CCA/PCA modes to what the training sample can support."""
    if n_samples < 2:
        raise ValueError("CCA requires at least two training samples along T.")
    max_modes = max(1, int(n_samples) - 1)
    nx, ny = _resolve_pca_modes(n_pca_modes)

    def _cap(v):
        if isinstance(v, str) or isinstance(v, float):
            return v  # "all" or a variance fraction: xeofs handles the cap
        return max(1, min(int(v), max_modes))

    safe_x, safe_y = _cap(nx), _cap(ny)
    numeric = [v for v in (safe_x, safe_y) if isinstance(v, (int, np.integer))]
    ceiling = min([max_modes] + numeric) if numeric else max_modes
    safe_cca = max(1, min(int(n_modes), ceiling))
    return safe_cca, [safe_x, safe_y]


def _make_cca(cfg: dict, n_samples: int):
    """Build an ``xe.cross.CCA`` with fold-compatible mode counts."""
    n_modes, n_pca_modes = _safe_mode_counts(
        cfg["n_modes"], cfg["n_pca_modes"], n_samples
    )
    return xe.cross.CCA(
        n_modes=n_modes,
        standardize=cfg.get("standardize", False),
        use_coslat=cfg.get("use_coslat", True),
        use_pca=cfg.get("use_pca", True),
        n_pca_modes=n_pca_modes,
    )


# ---------------------------------------------------------------------------
# Preprocessing (fitted on the training fold, applied to everything else)
# ---------------------------------------------------------------------------
def _fit_preprocessing(X: xr.DataArray, Y: xr.DataArray):
    """Impute partial gaps with the training mean, exclude unusable cells.

    Cells that are not fully observed, or temporally constant, are left as NaN
    so xeofs drops those features rather than ingesting constant-zero fills.
    """
    X = _prepare_tyx(X, "X_train")
    Y = _prepare_tyx(Y, "y_train")
    x_ref = X.mean(dim="T", skipna=True)
    y_ref = Y.mean(dim="T", skipna=True)
    X_imp = X.fillna(x_ref)
    Y_imp = Y.fillna(y_ref)
    state = {
        "x_ref": x_ref,
        "y_ref": y_ref,
        "x_mask": _common_validity_mask(X_imp),
        "y_mask": _common_validity_mask(Y_imp),
    }
    X_final = _to_latlon(X_imp.where(state["x_mask"]))
    Y_final = _to_latlon(Y_imp.where(state["y_mask"]))
    return state, X_final, Y_final


def _apply_preprocessing(da: xr.DataArray, ref, mask, name="field") -> xr.DataArray:
    da = _prepare_tyx(da, name)
    if ref is not None:
        da = da.fillna(ref)
    if mask is not None:
        da = da.where(mask)
    return _to_latlon(da)


def _fit_predict_once(cfg, X_train, y_train, X_test):
    """One standalone fit + predict. Never mutates a model instance."""
    state, Xtr, Ytr = _fit_preprocessing(X_train, y_train)
    n_samples = min(Xtr.sizes["T"], Ytr.sizes["T"])
    cca = _make_cca(cfg, n_samples)
    model = cca.fit(Xtr, Ytr, dim="T")
    Xte = _apply_preprocessing(X_test, state["x_ref"], state["x_mask"], "X_test")
    y_pred = model.predict(Xte)
    y_pred = model.inverse_transform(model.transform(Xte), y_pred)[1]
    return _to_yx(y_pred)


# ---------------------------------------------------------------------------
# Truncation selection
# ---------------------------------------------------------------------------
class TruncationResult(dict):
    """Result of :func:`tune_truncation` (a dict with attribute access)."""

    __getattr__ = dict.get

    def __repr__(self):
        return (
            f"TruncationResult(n_modes={self.get('n_modes')}, "
            f"n_pca_modes={self.get('n_pca_modes')}, "
            f"score={self.get('score'):.4f}, rule={self.get('rule')!r})"
        )


def _area_weights(obs: xr.DataArray) -> xr.DataArray:
    lat = obs["Y"]
    w = np.cos(np.deg2rad(lat.astype(float)))
    return w.clip(min=0.0).broadcast_like(obs.isel(T=0, drop=True))


def _aggregate_score(pred, obs, weights, kind):
    """Area-weighted aggregate skill score. Returns a float."""
    valid = np.isfinite(pred) & np.isfinite(obs)
    if kind == "mse":
        sq = xr.where(valid, (pred - obs) ** 2, np.nan)
        num = (sq * weights).sum(dim=("T", "Y", "X"), skipna=True)
        den = (xr.where(valid, weights, np.nan)).sum(dim=("T", "Y", "X"), skipna=True)
        return float(num / den) if float(den) > 0 else np.nan

    if kind in ("pearson", "spearman"):
        p = xr.where(valid, pred, np.nan)
        o = xr.where(valid, obs, np.nan)
        if kind == "spearman":
            p = p.rank(dim="T")
            o = o.rank(dim="T")
        pa = p - p.mean(dim="T", skipna=True)
        oa = o - o.mean(dim="T", skipna=True)
        cov = (pa * oa).sum(dim="T", skipna=True)
        denom = np.sqrt((pa ** 2).sum(dim="T", skipna=True) * (oa ** 2).sum(dim="T", skipna=True))
        corr = xr.where(denom > 0, cov / denom, np.nan)
        w = xr.where(np.isfinite(corr), weights, np.nan)
        num = (corr * w).sum(skipna=True)
        den = w.sum(skipna=True)
        return float(num / den) if float(den) > 0 else np.nan

    raise ValueError(f"Unknown score {kind!r}. Use 'mse', 'pearson' or 'spearman'.")


def _score_with_jackknife(pred, obs, weights, kind, fold_of):
    """Aggregate score plus its leave-one-fold-out jackknife standard error."""
    full = _aggregate_score(pred, obs, weights, kind)
    folds = np.unique(fold_of)
    if folds.size < 3:
        return full, np.nan
    partial = []
    for f in folds:
        keep = np.where(fold_of != f)[0]
        if keep.size < 2:
            continue
        partial.append(
            _aggregate_score(pred.isel(T=keep), obs.isel(T=keep), weights, kind)
        )
    partial = np.asarray([v for v in partial if np.isfinite(v)], dtype=float)
    if partial.size < 3:
        return full, np.nan
    n = partial.size
    se = float(np.sqrt((n - 1) / n * np.sum((partial - partial.mean()) ** 2)))
    return full, se


def tune_truncation(
    X: xr.DataArray,
    Y: xr.DataArray,
    splits,
    n_modes_grid=(1, 2, 3, 4),
    x_pca_grid=(4, 6, 8, 10),
    y_pca_grid=None,
    template=None,
    score="mse",
    one_se=True,
    n_jobs=None,
    progress=True,
):
    """Select ``(n_modes, n_pca_modes_X, n_pca_modes_Y)`` by cross-validation.

    Parameters
    ----------
    X, Y : xr.DataArray
        Predictor and predictand **already in the space the CCA will be fitted
        in** (detrended, standardized, whatever the class requires). This
        function does not transform them.
    splits : sequence of (train_index, test_index)
        For example ``list(cv.custom_cv.split(X['T'], cv.nb_omit))``.
    n_modes_grid, x_pca_grid : sequence of int
        Canonical modes and X PCA truncations to explore.
    y_pca_grid : sequence of int or None
        ``None`` mirrors ``x_pca_grid`` (2-D search, |grid| = |modes|x|x_pca|).
        Pass an explicit sequence for the full 3-D CPT-like search — it costs
        |y_pca_grid| times more.
    template : WAS_CCA* instance or None
        Its ``standardize`` / ``use_coslat`` / ``use_pca`` settings are reused.
    score : {'mse', 'pearson', 'spearman'}
        ``'mse'`` is the default on purpose: correlation is blind to amplitude,
        and amplitude is exactly what feeds the predictive spread and therefore
        the calibration of the tercile probabilities.
    one_se : bool
        Apply the one-standard-error rule: among the configurations whose score
        is within one jackknife SE of the best, keep the most parsimonious. On
        35-40 years the raw optimum is not stable from one season to the next.

    Returns
    -------
    TruncationResult
        ``n_modes``, ``n_pca_modes`` (a ``[nx, ny]`` list), ``score``, ``se``,
        ``rule`` and ``table`` (every configuration, sorted best first).

    Notes
    -----
    The score reported here is a maximum over the grid and is therefore
    optimistic. Run this **once, before** the outer CV loop for a CPT-like
    workflow, or **inside** each training fold for a strictly nested estimate —
    but never compare a score obtained one way with a score obtained the other.
    """
    X = _prepare_tyx(X, "X")
    Y = _prepare_tyx(Y, "Y")
    splits = [(np.asarray(a), np.asarray(b)) for a, b in splits]
    if not splits:
        raise ValueError("`splits` is empty.")

    cfg_base = {
        "standardize": getattr(template, "standardize", False),
        "use_coslat": getattr(template, "use_coslat", True),
        "use_pca": getattr(template, "use_pca", True),
    }

    y_grid = tuple(x_pca_grid) if y_pca_grid is None else tuple(y_pca_grid)
    tied = y_pca_grid is None

    combos = []
    for nm in sorted(set(int(v) for v in n_modes_grid)):
        for nx in sorted(set(int(v) for v in x_pca_grid)):
            ys = (nx,) if tied else y_grid
            for ny in sorted(set(int(v) for v in ys)):
                if nm <= min(nx, ny):
                    combos.append((nm, nx, ny))
    if not combos:
        raise ValueError("Empty grid: n_modes must be <= min(n_pca_X, n_pca_Y).")

    weights = _area_weights(Y)
    fold_of = np.concatenate([np.full(len(te), i) for i, (_, te) in enumerate(splits)])

    def _evaluate(combo):
        nm, nx, ny = combo
        cfg = dict(cfg_base, n_modes=nm, n_pca_modes=[nx, ny])
        preds = []
        try:
            for train_index, test_index in splits:
                pred = _fit_predict_once(
                    cfg,
                    X.isel(T=train_index),
                    Y.isel(T=train_index),
                    X.isel(T=test_index),
                )
                pred = pred.assign_coords(T=Y["T"].isel(T=test_index))
                preds.append(pred)
        except Exception as exc:  # a truncation the sample cannot support
            return {"n_modes": nm, "n_pca_x": nx, "n_pca_y": ny,
                    "score": np.nan, "se": np.nan, "error": repr(exc)}

        pred_all = xr.concat(preds, dim="T")
        obs_all = Y.sel(T=pred_all["T"])
        value, se = _score_with_jackknife(pred_all, obs_all, weights, score, fold_of)
        return {"n_modes": nm, "n_pca_x": nx, "n_pca_y": ny,
                "score": value, "se": se, "error": None}

    workers = _grid_workers() if n_jobs is None else int(n_jobs)
    if workers > 1 and len(combos) > 1:
        with ThreadPoolExecutor(max_workers=min(workers, len(combos))) as pool:
            rows = list(pool.map(_evaluate, combos))
    else:
        rows = []
        for k, combo in enumerate(combos, start=1):
            if progress:
                print(f"  truncation {k}/{len(combos)} {combo}", flush=True)
            rows.append(_evaluate(combo))

    lower_is_better = score == "mse"
    finite = [r for r in rows if np.isfinite(r["score"])]
    if not finite:
        raise RuntimeError("No truncation could be evaluated; check the grid and the data.")

    rows_sorted = sorted(finite, key=lambda r: r["score"], reverse=not lower_is_better)
    best = rows_sorted[0]
    rule = "best"

    if one_se and np.isfinite(best["se"]):
        if lower_is_better:
            threshold = best["score"] + best["se"]
            within = [r for r in finite if r["score"] <= threshold]
        else:
            threshold = best["score"] - best["se"]
            within = [r for r in finite if r["score"] >= threshold]
        if within:
            chosen = min(within, key=lambda r: (r["n_modes"], r["n_pca_x"] + r["n_pca_y"]))
            if chosen is not best:
                rule = "one-se"
            best = chosen

    return TruncationResult(
        n_modes=int(best["n_modes"]),
        n_pca_modes=[int(best["n_pca_x"]), int(best["n_pca_y"])],
        score=float(best["score"]),
        se=float(best["se"]) if np.isfinite(best["se"]) else np.nan,
        metric=score,
        rule=rule,
        table=rows_sorted,
    )


# ---------------------------------------------------------------------------
# Probability engine
# ---------------------------------------------------------------------------
class _ProbabilityEngine:
    """Tercile probabilities. Shared by all four CCA classes."""

    @staticmethod
    def _ppf_terciles_from_code(dist_code, shape, loc, scale):
        """Exact 1/3 and 2/3 thresholds from fitted parameters."""
        if not np.isfinite(dist_code):
            return np.nan, np.nan
        q1, q2 = 1.0 / 3.0, 2.0 / 3.0
        code = int(dist_code)
        try:
            if code == 1:
                return norm.ppf(q1, loc=loc, scale=scale), norm.ppf(q2, loc=loc, scale=scale)
            if code == 2:
                return (lognorm.ppf(q1, s=shape, loc=loc, scale=scale),
                        lognorm.ppf(q2, s=shape, loc=loc, scale=scale))
            if code == 3:
                return expon.ppf(q1, loc=loc, scale=scale), expon.ppf(q2, loc=loc, scale=scale)
            if code == 4:
                return (gamma.ppf(q1, a=shape, loc=loc, scale=scale),
                        gamma.ppf(q2, a=shape, loc=loc, scale=scale))
            if code == 5:
                return (weibull_min.ppf(q1, c=shape, loc=loc, scale=scale),
                        weibull_min.ppf(q2, c=shape, loc=loc, scale=scale))
            if code == 6:
                return (t.ppf(q1, df=shape, loc=loc, scale=scale),
                        t.ppf(q2, df=shape, loc=loc, scale=scale))
            if code == 7:
                return poisson.ppf(q1, mu=shape, loc=loc), poisson.ppf(q2, mu=shape, loc=loc)
            if code == 8:
                return (nbinom.ppf(q1, n=shape, p=scale, loc=loc),
                        nbinom.ppf(q2, n=shape, p=scale, loc=loc))
        except Exception:
            return np.nan, np.nan
        return np.nan, np.nan

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
    def calculate_tercile_probabilities_bestfit(
        best_guess, error_variance, T1, T2, dist_code, dof, nonneg=True
    ):
        """PB/PN/PA for the requested predictive family.

        ``error_variance`` is the mean squared error of the hindcast, so the
        predictive distribution is centred on ``best_guess`` with a spread that
        includes any systematic bias.
        """
        best_guess = np.asarray(best_guess, dtype=float)
        error_variance = float(error_variance)
        n_time = best_guess.size
        out = np.full((3, n_time), np.nan, dtype=float)

        if (np.all(~np.isfinite(best_guess))
                or not np.isfinite(error_variance) or error_variance <= 0
                or not np.isfinite(T1) or not np.isfinite(T2)
                or not np.isfinite(dist_code)):
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
                sigma[valid] = np.sqrt(np.log1p(error_variance / (best_guess[valid] ** 2)))
                mu[valid] = np.log(best_guess[valid]) - 0.5 * sigma[valid] ** 2
                c1 = lognorm.cdf(T1, s=sigma, scale=np.exp(mu))
                c2 = lognorm.cdf(T2, s=sigma, scale=np.exp(mu))

            elif code == 3:
                if nonneg:
                    # Single-parameter exponential: loc = 0 keeps the support
                    # non-negative and scale = best_guess keeps the mean right.
                    scale = np.where(best_guess > 0, best_guess, np.nan)
                    loc = 0.0
                else:
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
                n = np.where(valid, best_guess ** 2 / (error_variance - best_guess), np.nan)
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
        ok = np.isfinite(total) & (total > 0)
        out[:, ok] /= total[ok]
        return out

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

    # -- engine ------------------------------------------------------------
    def _probability_field(
        self, deterministic, error_samples, error_variance, T1_emp, T2_emp,
        dof, mask, nonneg=True, discrete=False,
        best_code_da=None, best_shape_da=None, best_loc_da=None, best_scale_da=None,
    ):
        deterministic = _prepare_tyx(deterministic, "deterministic")
        dm = str(self.dist_method).strip().lower()

        if dm == "nonparam":
            error_samples = _prepare_tyx(error_samples, "error_samples").rename({"T": "T_error"})
            prob = _pufunc(
                self.calculate_tercile_probabilities_nonparametric,
                deterministic, error_samples, T1_emp, T2_emp,
                input_core_dims=[("T",), ("T_error",), (), ()],
                output_core_dims=[("probability", "T")],
                vectorize=True, dask="parallelized", output_dtypes=[float],
                dask_gufunc_kwargs={"output_sizes": {"probability": 3}, "allow_rechunk": True},
            )
            return _finalize_probabilities(prob, mask)

        if dm == "bestfit":
            if any(v is None for v in (best_code_da, best_shape_da, best_loc_da, best_scale_da)):
                raise ValueError(
                    "dist_method='bestfit' requires best_code_da, best_shape_da, "
                    "best_loc_da and best_scale_da."
                )
            code_da = self._guard_discrete_codes(best_code_da, discrete)
            T1, T2 = _pufunc(
                self._ppf_terciles_from_code,
                code_da, best_shape_da, best_loc_da, best_scale_da,
                input_core_dims=[(), (), (), ()], output_core_dims=[(), ()],
                vectorize=True, dask="parallelized", output_dtypes=[float, float],
            )

        elif dm in _DISTRIBUTION_CODES:
            code = _DISTRIBUTION_CODES[dm]
            if code in _DISCRETE_CODES and not discrete:
                raise ValueError(
                    f"dist_method={self.dist_method!r} is a discrete family but the "
                    "predictand is continuous; its CDF would be evaluated at the "
                    "integer floor of the tercile thresholds."
                )
            T1, T2 = T1_emp, T2_emp
            code_da = xr.full_like(error_variance, code, dtype=float)

        else:
            valid = ["nonparam", "bestfit"] + sorted(set(_DISTRIBUTION_CODES))
            raise ValueError(
                f"Invalid dist_method={self.dist_method!r}. Supported methods: {valid}."
            )

        prob = _pufunc(
            self.calculate_tercile_probabilities_bestfit,
            deterministic, error_variance, T1, T2, code_da,
            input_core_dims=[("T",), (), (), (), ()],
            output_core_dims=[("probability", "T")],
            vectorize=True, kwargs={"dof": dof, "nonneg": nonneg},
            dask="parallelized", output_dtypes=[float],
            dask_gufunc_kwargs={"output_sizes": {"probability": 3}, "allow_rechunk": True},
        )
        return _finalize_probabilities(prob, mask)

    @staticmethod
    def _guard_discrete_codes(code_da, discrete):
        """Fall back to the gamma family where a discrete one was selected."""
        if discrete:
            return code_da
        offending = (code_da == 7) | (code_da == 8)
        if _as_bool(offending.any(), default=False):
            warnings.warn(
                "Poisson / negative-binomial codes selected for a continuous "
                "predictand; falling back to gamma at those grid cells.",
                RuntimeWarning, stacklevel=2,
            )
            return xr.where(offending, 4.0, code_da)
        return code_da

    def _tercile_context(self, Predictant, clim_year_start, clim_year_end):
        clim = Predictant.sel(T=slice(str(clim_year_start), str(clim_year_end)))
        if clim.sizes.get("T", 0) < 3:
            raise ValueError("Not enough years in climatology period for terciles.")
        terciles = clim.quantile([1.0 / 3.0, 2.0 / 3.0], dim="T")
        T1 = terciles.isel(quantile=0, drop=True)
        T2 = terciles.isel(quantile=1, drop=True)
        dof = _effective_dof(clim.sizes["T"], self.n_modes)
        return T1, T2, dof

    def compute_prob(
        self, Predictant, clim_year_start, clim_year_end, hindcast_det,
        best_code_da=None, best_shape_da=None, best_loc_da=None, best_scale_da=None,
    ):
        """Hindcast tercile probabilities, dims (probability, T, Y, X)."""
        Predictant, hindcast_det = _align_observation_and_hindcast(Predictant, hindcast_det)
        mask = spatial_mask(Predictant)
        T1_emp, T2_emp, dof = self._tercile_context(
            Predictant, clim_year_start, clim_year_end
        )
        error_samples, error_variance = _predictive_error_stats(Predictant, hindcast_det)
        return self._probability_field(
            deterministic=hindcast_det,
            error_samples=error_samples,
            error_variance=error_variance,
            T1_emp=T1_emp, T2_emp=T2_emp, dof=dof, mask=mask,
            nonneg=float(np.asarray(Predictant.min(skipna=True).values)) >= 0.0,
            discrete=_looks_discrete(Predictant),
            best_code_da=best_code_da, best_shape_da=best_shape_da,
            best_loc_da=best_loc_da, best_scale_da=best_scale_da,
        )

    # Backwards-compatible alias used by older workflows.
    def _compute_tercile_probabilities(
        self, Predictant, deterministic, clim_year_start, clim_year_end,
        error_samples, error_variance,
        best_code_da=None, best_shape_da=None, best_loc_da=None, best_scale_da=None,
    ):
        Predictant = _prepare_tyx(Predictant, "Predictant")
        mask = spatial_mask(Predictant)
        T1_emp, T2_emp, dof = self._tercile_context(
            Predictant, clim_year_start, clim_year_end
        )
        return self._probability_field(
            deterministic, error_samples, error_variance, T1_emp, T2_emp, dof, mask,
            nonneg=float(np.asarray(Predictant.min(skipna=True).values)) >= 0.0,
            discrete=_looks_discrete(Predictant),
            best_code_da=best_code_da, best_shape_da=best_shape_da,
            best_loc_da=best_loc_da, best_scale_da=best_scale_da,
        )


# ---------------------------------------------------------------------------
# Shared CCA core
# ---------------------------------------------------------------------------
class _CCACore(_ProbabilityEngine):
    """Preprocessing, fitting, prediction, tuning and plotting."""

    def __init__(self, n_modes=4, n_pca_modes=8, standardize=False,
                 use_coslat=True, use_pca=True, dist_method="nonparam"):
        self.n_modes = n_modes
        self.n_pca_modes = n_pca_modes
        self.standardize = standardize
        self.use_coslat = use_coslat
        self.use_pca = use_pca
        self.dist_method = dist_method

        self.cca = None
        self.cca_model = None
        self._prep = None
        self.truncation_ = None

    # -- configuration -----------------------------------------------------
    @property
    def _cfg(self):
        return {
            "n_modes": self.n_modes,
            "n_pca_modes": self.n_pca_modes,
            "standardize": self.standardize,
            "use_coslat": self.use_coslat,
            "use_pca": self.use_pca,
        }

    # -- preprocessing -----------------------------------------------------
    def preprocess_data(self, X, Y):
        self._prep, X_final, Y_final = _fit_preprocessing(X, Y)
        # Historical attribute names kept for external callers.
        self._x_train_ref = self._prep["x_ref"]
        self._y_train_ref = self._prep["y_ref"]
        self._x_valid_mask = self._prep["x_mask"]
        self._y_valid_mask = self._prep["y_mask"]
        return X_final, Y_final

    def preprocess_test_data(self, X_test, y_test=None, X_train=None, y_train=None):
        prep = self._prep or {}
        x_ref = prep.get("x_ref")
        if x_ref is None and X_train is not None:
            x_ref = _prepare_tyx(X_train, "X_train").mean(dim="T", skipna=True)
        X_out = _apply_preprocessing(X_test, x_ref, prep.get("x_mask"), "X_test")

        if y_test is None:
            return X_out, None
        y_ref = prep.get("y_ref")
        if y_ref is None and y_train is not None:
            y_ref = _prepare_tyx(y_train, "y_train").mean(dim="T", skipna=True)
        y_out = _apply_preprocessing(y_test, y_ref, prep.get("y_mask"), "y_test")
        return X_out, y_out

    def preprocess_forecast_data(self, Predictor_for_year, Predictor_train=None):
        return self.preprocess_test_data(Predictor_for_year, X_train=Predictor_train)[0]

    # -- fit / predict -----------------------------------------------------
    def fit_cca(self, X_train, y_train):
        X_final, y_final = self.preprocess_data(X_train, y_train)
        n_samples = min(X_final.sizes["T"], y_final.sizes["T"])
        self.cca = _make_cca(self._cfg, n_samples)
        self.cca_model = self.cca.fit(X_final, y_final, dim="T")
        return self.cca_model

    def _predict_prepared(self, X_prepared):
        y_pred = self.cca_model.predict(X_prepared)
        y_pred = self.cca_model.inverse_transform(
            self.cca_model.transform(X_prepared), y_pred
        )[1]
        return _to_yx(y_pred).transpose("T", "Y", "X")

    def compute_model(self, X_train, y_train, X_test, y_test=None):
        self.fit_cca(X_train, y_train)
        X_prepared, y_prepared = self.preprocess_test_data(
            X_test, y_test, X_train=X_train, y_train=y_train
        )
        y_pred = self._predict_prepared(X_prepared)
        if y_prepared is not None:
            y_pred = y_pred.assign_coords(T=y_prepared["T"])
        return y_pred

    # -- truncation --------------------------------------------------------
    def tune(self, X, Y, splits, apply=True, **kwargs):
        """Select the truncations by CV and (by default) adopt them.

        ``X`` and ``Y`` must already be in the space this class fits in. For a
        CPT-like workflow call this once before the outer CV loop; for a nested
        estimate call it inside each training fold.
        """
        kwargs.setdefault("template", self)
        result = tune_truncation(X, Y, splits, **kwargs)
        self.truncation_ = result
        if apply:
            self.n_modes = result["n_modes"]
            self.n_pca_modes = result["n_pca_modes"]
        return result

    # -- forecast tail -----------------------------------------------------
    def _forecast_probabilities(
        self, forecast_det, Predictant, Predictant_resid, hindcast_resid,
        clim_year_start, clim_year_end, mask,
        best_code_da=None, best_shape_da=None, best_loc_da=None, best_scale_da=None,
    ):
        T1_emp, T2_emp, dof = self._tercile_context(
            Predictant, clim_year_start, clim_year_end
        )
        error_samples, error_variance = _predictive_error_stats(
            Predictant_resid, hindcast_resid
        )
        return self._probability_field(
            forecast_det, error_samples, error_variance, T1_emp, T2_emp, dof, mask,
            nonneg=float(np.asarray(Predictant.min(skipna=True).values)) >= 0.0,
            discrete=_looks_discrete(Predictant),
            best_code_da=best_code_da, best_shape_da=best_shape_da,
            best_loc_da=best_loc_da, best_scale_da=best_scale_da,
        )

    def _forecast_common(self, Predictant, Predictor, Predictor_for_year, hindcast_det):
        Predictant = _prepare_tyx(Predictant, "Predictant")
        Predictor = _prepare_tyx(Predictor, "Predictor")
        Predictor_for_year = _prepare_tyx(Predictor_for_year, "Predictor_for_year")
        Predictant_resid, hindcast_resid = _align_observation_and_hindcast(
            Predictant, hindcast_det
        )
        return (Predictant, Predictor, Predictor_for_year,
                Predictant_resid, hindcast_resid, spatial_mask(Predictant))

    # -- plotting ----------------------------------------------------------
    def _plot_prepare(self, X, Y, clim_year_start, clim_year_end):
        """Transform X and Y into the space this class fits in."""
        raise NotImplementedError

    def plot_cca_results(self, X=None, Y=None, n_modes=None,
                         clim_year_start=None, clim_year_end=None):
        """Plot CCA spatial modes and canonical scores."""
        mask = None
        if X is not None and Y is not None:
            X = _prepare_tyx(X, "X")
            Y = _prepare_tyx(Y, "Y")
            mask = _to_latlon(spatial_mask(Y))
            X_ready, Y_ready = self._plot_prepare(X, Y, clim_year_start, clim_year_end)
            self.fit_cca(X_ready, Y_ready)
        elif self.cca_model is None:
            raise ValueError(
                "The CCA model has not been fitted yet. Provide X and Y, or call fit_cca."
            )

        X_modes, Y_modes = self.cca_model.components()
        X_scores, Y_scores = self.cca_model.scores()
        var_X = self.cca_model.fraction_variance_X_explained_by_X()
        var_Y = self.cca_model.fraction_variance_Y_explained_by_Y()
        var_Y_by_X = self.cca_model.fraction_variance_Y_explained_by_X()

        available = len(X_modes["mode"])
        n_modes = available if n_modes is None else int(n_modes)
        n_modes = max(1, min(n_modes, available))
        mode_indices = X_modes["mode"].values[:n_modes]

        fig, axes = plt.subplots(n_modes, 3, figsize=(15, 3.5 * n_modes), squeeze=False)

        for i, mode in enumerate(mode_indices):
            ax = axes[i, 0]
            X_modes.sel(mode=mode).plot(ax=ax, cmap="RdBu_r")
            ax.set_title(
                f"X Mode {mode} ({float(var_X.sel(mode=mode).values) * 100:.2f}% X variance)"
            )

            ax = axes[i, 1]
            xs = X_scores.sel(mode=mode)
            ys = Y_scores.sel(mode=mode)
            if np.issubdtype(xs["T"].dtype, np.datetime64):
                axis, xlabel = xs["T"].dt.year.values, "Year"
            else:
                axis, xlabel = xs["T"].values, "Time"
            ax.plot(axis, xs.values, label="X score")
            ax.plot(axis, ys.values, label="Y score")
            ax.axhline(0.0, linestyle="--", lw=0.8)
            ax.legend()
            ax.set_title(
                f"Scores Mode {mode} "
                f"({float(var_Y_by_X.sel(mode=mode).values) * 100:.2f}% Y explained by X)"
            )
            ax.set_xlabel(xlabel)
            ax.set_ylabel("Canonical variate")

            ax = axes[i, 2]
            ymode = Y_modes.sel(mode=mode)
            if mask is not None:
                ymode = ymode * mask
            ymode.plot(ax=ax, cmap="RdBu_r")
            ax.set_title(
                f"Y Mode {mode} ({float(var_Y.sel(mode=mode).values) * 100:.2f}% Y variance)"
            )

        plt.tight_layout()
        plt.show()


# ---------------------------------------------------------------------------
# WAS_CCA — leading EEOF component removed from both fields
# ---------------------------------------------------------------------------
class WAS_CCA(_CCACore):
    """CCA on EEOF residuals.

    The leading extended-EOF component is removed from the predictor and from
    the standardized predictand. Because that component cannot be recomputed
    for a year outside the record, it is persisted from ``T[-eeof_ref_offset]``
    — on **both** sides, so the forecast-year predictor enters the same space
    the model was fitted in, and the predictand component is added back before
    ``reverse_standardize``.
    """

    def __init__(self, n_modes=4, n_pca_modes=8, standardize=False, use_coslat=True,
                 use_pca=True, dist_method="nonparam", eeof_ref_offset=3):
        super().__init__(n_modes, n_pca_modes, standardize, use_coslat, use_pca, dist_method)
        self.eeof_ref_offset = int(eeof_ref_offset)

    def _eeof_split(self, field):
        """Return (residual, trend, persistence reference)."""
        trend = extract_leading_eeof_component(field)  # noqa: F405
        idx = max(0, trend.sizes["T"] - self.eeof_ref_offset)
        ref = trend.isel(T=idx, drop=True)
        return field - trend.fillna(ref), trend, ref

    def prepare_eeof_fields(self, Predictor, Predictant, clim_year_start, clim_year_end):
        """Fields in fit space, plus what the cross-validator needs to invert it.

        Returns ``(X_resid, Y_resid, trend_Y, ref_X, ref_Y)``. In cross
        validation, predict with ``X_resid`` on **both** folds and add
        ``trend_Y`` back at the validation year before ``reverse_standardize``.
        """
        Predictor = _prepare_tyx(Predictor, "Predictor")
        Predictant = _prepare_tyx(Predictant, "Predictant")
        X_resid, _, ref_X = self._eeof_split(Predictor)
        Predictant_st = standardize_timeseries(  # noqa: F405
            Predictant, clim_year_start, clim_year_end
        )
        Y_resid, trend_Y, ref_Y = self._eeof_split(Predictant_st)
        return X_resid, Y_resid, trend_Y, ref_X, ref_Y

    def _plot_prepare(self, X, Y, clim_year_start, clim_year_end):
        X_resid, Y_resid, _, _, _ = self.prepare_eeof_fields(
            X, Y, clim_year_start, clim_year_end
        )
        return X_resid, Y_resid

    def forecast(self, Predictant, clim_year_start, clim_year_end, Predictor,
                 hindcast_det, Predictor_for_year, best_code_da=None,
                 best_shape_da=None, best_loc_da=None, best_scale_da=None):
        (Predictant, Predictor, Predictor_for_year,
         Predictant_resid, hindcast_resid, mask) = self._forecast_common(
            Predictant, Predictor, Predictor_for_year, hindcast_det
        )

        X_resid, Y_resid, _, ref_X, ref_Y = self.prepare_eeof_fields(
            Predictor, Predictant, clim_year_start, clim_year_end
        )
        self.fit_cca(X_resid, Y_resid)

        # Same EEOF residual space as the training predictor (this is the step
        # that was missing), then the same validity mask used at fit time.
        X_year = _fill_spatial_gaps_safe(
            Predictor_for_year.fillna(Predictor.mean(dim="T", skipna=True))
        )
        X_year = X_year - ref_X
        X_prepared = self.preprocess_test_data(X_year)[0]

        result = self._predict_prepared(X_prepared)
        result = result.assign_coords(
            T=xr.DataArray(_target_time(Predictant, Predictor_for_year), dims=["T"])
        )
        # Predictand EEOF component added back before leaving standardized space.
        result_st = result + ref_Y

        forecast_det = reverse_standardize(  # noqa: F405
            result_st, Predictant, clim_year_start, clim_year_end
        ).transpose("T", "Y", "X")
        forecast_det = conditional_nonnegative_clip(forecast_det, Predictant)

        forecast_prob = self._forecast_probabilities(
            forecast_det, Predictant, Predictant_resid, hindcast_resid,
            clim_year_start, clim_year_end, mask,
            best_code_da, best_shape_da, best_loc_da, best_scale_da,
        )
        return (forecast_det * mask).transpose("T", "Y", "X"), forecast_prob


# ---------------------------------------------------------------------------
# WAS_CCA_base — no detrending
# ---------------------------------------------------------------------------
class WAS_CCA_base(_CCACore):
    """CCA with no detrending. X raw, Y standardized over the climatology.

    Note that ``standardize_timeseries`` uses the full climatology, so the
    reference statistics of a validation year are not strictly held out. On
    30-40 years the effect is second order, but this class is not
    "leakage-free" in the strict sense — ``WAS_CCA_strict`` is.
    """

    def _plot_prepare(self, X, Y, clim_year_start, clim_year_end):
        if clim_year_start is not None and clim_year_end is not None:
            Y = standardize_timeseries(Y, clim_year_start, clim_year_end)  # noqa: F405
        return X, Y

    def forecast(self, Predictant, clim_year_start, clim_year_end, Predictor,
                 hindcast_det, Predictor_for_year, best_code_da=None,
                 best_shape_da=None, best_loc_da=None, best_scale_da=None):
        (Predictant, Predictor, Predictor_for_year,
         Predictant_resid, hindcast_resid, mask) = self._forecast_common(
            Predictant, Predictor, Predictor_for_year, hindcast_det
        )

        Predictant_st = standardize_timeseries(  # noqa: F405
            Predictant, clim_year_start, clim_year_end
        )
        self.fit_cca(Predictor, Predictant_st)

        X_prepared = self.preprocess_test_data(Predictor_for_year)[0]
        forecast_st = self._predict_prepared(X_prepared)

        forecast_det = reverse_standardize(  # noqa: F405
            forecast_st, Predictant, clim_year_start, clim_year_end
        )
        forecast_det = forecast_det.assign_coords(
            T=xr.DataArray(_target_time(Predictant, Predictor_for_year), dims=["T"])
        ).transpose("T", "Y", "X")
        forecast_det = conditional_nonnegative_clip(forecast_det, Predictant)

        forecast_prob = self._forecast_probabilities(
            forecast_det, Predictant, Predictant_resid, hindcast_resid,
            clim_year_start, clim_year_end, mask,
            best_code_da, best_shape_da, best_loc_da, best_scale_da,
        )
        return (forecast_det * mask).transpose("T", "Y", "X"), forecast_prob


# ---------------------------------------------------------------------------
# WAS_CCA_op — linear detrending
# ---------------------------------------------------------------------------
class WAS_CCA_op(_CCACore):
    """CCA on linearly detrended X and standardized, detrended Y.

    The cross-validator must fit ``detrended_data`` **inside** each fold. Fitting
    the trend on the full record and reusing its coefficients to rebuild the
    hindcast lets the regression see the validation year — negligible on
    rainfall, material on temperature, where the trend carries most of the
    variance. See :meth:`prepare_fold` .
    """

    def __init__(self, n_modes=4, n_pca_modes=8, standardize=False, use_coslat=True,
                 use_pca=True, dist_method="bestfit"):
        super().__init__(n_modes, n_pca_modes, standardize, use_coslat, use_pca, dist_method)

    @staticmethod
    def prepare_fold(X_train, y_train_raw, X_test, clim_year_start, clim_year_end):
        """Fold-local transformation. Use this in the cross-validator.

        Returns ``(X_train_ready, y_train_ready, X_test_ready, coeffs_Y, meta_Y)``.
        Add the predictand trend back with
        ``pred + apply_detrend_data(pred, coeffs_Y, meta_Y)`` at the validation
        year, then ``reverse_standardize`` against ``y_train_raw``.
        """
        X_train = _prepare_tyx(X_train, "X_train")
        X_test = _prepare_tyx(X_test, "X_test")
        y_train_raw = _prepare_tyx(y_train_raw, "y_train")

        X_train_detr, coeffs_X, meta_X = detrended_data(X_train, dim="T")  # noqa: F405
        X_test_detr = X_test - apply_detrend_data(X_test, coeffs_X, meta_X)  # noqa: F405

        y_st = standardize_timeseries(  # noqa: F405
            y_train_raw, clim_year_start, clim_year_end
        )
        y_st = y_st.where(np.isfinite(y_st))
        y_st_detr, coeffs_Y, meta_Y = detrended_data(y_st, dim="T")  # noqa: F405

        return X_train_detr, y_st_detr, X_test_detr, coeffs_Y, meta_Y

    def _plot_prepare(self, X, Y, clim_year_start, clim_year_end):
        X_detr, _, _ = detrended_data(X, dim="T")  # noqa: F405
        Y_st = standardize_timeseries(Y, clim_year_start, clim_year_end)  # noqa: F405
        Y_st = Y_st.where(np.isfinite(Y_st))
        Y_detr, _, _ = detrended_data(Y_st, dim="T")  # noqa: F405
        return X_detr, Y_detr

    def forecast(self, Predictant, clim_year_start, clim_year_end, Predictor,
                 hindcast_det, Predictor_for_year, best_code_da=None,
                 best_shape_da=None, best_loc_da=None, best_scale_da=None):
        (Predictant, Predictor, Predictor_for_year,
         Predictant_resid, hindcast_resid, mask) = self._forecast_common(
            Predictant, Predictor, Predictor_for_year, hindcast_det
        )

        Predictor_detrend, coeffs_X, meta_X = detrended_data(Predictor, dim="T")  # noqa: F405
        Predictant_st = standardize_timeseries(  # noqa: F405
            Predictant, clim_year_start, clim_year_end
        )
        Predictant_st_detrend, coeffs_Y, meta_Y = detrended_data(  # noqa: F405
            Predictant_st, dim="T"
        )

        X_year = Predictor_for_year.fillna(Predictor.mean(dim="T", skipna=True))
        X_year = X_year - apply_detrend_data(X_year, coeffs_X, meta_X)  # noqa: F405

        self.fit_cca(Predictor_detrend, Predictant_st_detrend)
        X_prepared = self.preprocess_test_data(X_year)[0]
        result_detrended = self._predict_prepared(X_prepared)

        # Evaluate the Y trend at the predictand target time, not the init month.
        result_detrended = result_detrended.assign_coords(
            T=xr.DataArray(_target_time(Predictant, Predictor_for_year), dims=["T"])
        )
        result_st = result_detrended + apply_detrend_data(  # noqa: F405
            result_detrended, coeffs_Y, meta_Y
        )

        forecast_det = reverse_standardize(  # noqa: F405
            result_st, Predictant, clim_year_start, clim_year_end
        ).transpose("T", "Y", "X")
        forecast_det = conditional_nonnegative_clip(forecast_det, Predictant)

        forecast_prob = self._forecast_probabilities(
            forecast_det, Predictant, Predictant_resid, hindcast_resid,
            clim_year_start, clim_year_end, mask,
            best_code_da, best_shape_da, best_loc_da, best_scale_da,
        )
        return (forecast_det * mask).transpose("T", "Y", "X"), forecast_prob


# ---------------------------------------------------------------------------
# WAS_CCA_strict — op transformations, strict spatial filling
# ---------------------------------------------------------------------------
class WAS_CCA_strict(WAS_CCA_op):
    """Same transformations as ``op`` with strict spatial gap filling.

    Intended for the fold-safe cross-validation branch, where every transform is
    refitted on the training fold. :meth:`tune` called inside that loop gives a
    fully nested truncation estimate.
    """

    @staticmethod
    def _fill_spatial_gaps(da, ref=None):
        """Fill forecast gaps with the historical mean, then spatial neighbours."""
        out = da
        if ref is not None and "T" in getattr(ref, "dims", ()):
            out = out.fillna(ref.mean(dim="T", skipna=True))
        return _fill_spatial_gaps_safe(out).fillna(0.0)

    @staticmethod
    def _safe_drop_vars(da, names):
        existing = [n for n in names if n in da.coords or n in da.variables]
        return da.drop_vars(existing) if existing else da

    _rename_to_latlon = staticmethod(_to_latlon)
    _rename_to_YX = staticmethod(_to_yx)

    @staticmethod
    def _target_time_from_predictor(Predictant, Predictor_for_year):
        return _target_time(Predictant, Predictor_for_year)[0]

    def forecast(self, Predictant, clim_year_start, clim_year_end, Predictor,
                 hindcast_det, Predictor_for_year, best_code_da=None,
                 best_shape_da=None, best_loc_da=None, best_scale_da=None):
        (Predictant, Predictor, Predictor_for_year,
         Predictant_resid, hindcast_resid, mask) = self._forecast_common(
            Predictant, Predictor, Predictor_for_year, hindcast_det
        )

        Predictor_filled = self._fill_spatial_gaps(Predictor)
        Predictor_detrend, coeffs_X, meta_X = detrended_data(  # noqa: F405
            Predictor_filled, dim="T"
        )

        Predictant_st = standardize_timeseries(  # noqa: F405
            Predictant, clim_year_start, clim_year_end
        )
        Predictant_st = Predictant_st.where(np.isfinite(Predictant_st))
        Predictant_st_detrend, coeffs_Y, meta_Y = detrended_data(  # noqa: F405
            Predictant_st, dim="T"
        )

        X_year = self._fill_spatial_gaps(Predictor_for_year, ref=Predictor_filled)
        X_year = X_year - apply_detrend_data(X_year, coeffs_X, meta_X)  # noqa: F405

        self.fit_cca(Predictor_detrend, Predictant_st_detrend)
        X_prepared = self.preprocess_test_data(X_year)[0]
        result_detrended = self._predict_prepared(X_prepared)

        result_detrended = result_detrended.assign_coords(
            T=xr.DataArray(_target_time(Predictant, Predictor_for_year), dims=["T"])
        )
        result_st = result_detrended + apply_detrend_data(  # noqa: F405
            result_detrended, coeffs_Y, meta_Y
        )

        forecast_det = reverse_standardize(  # noqa: F405
            result_st, Predictant, clim_year_start, clim_year_end
        ).transpose("T", "Y", "X")
        forecast_det = conditional_nonnegative_clip(forecast_det, Predictant)

        forecast_prob = self._forecast_probabilities(
            forecast_det, Predictant, Predictant_resid, hindcast_resid,
            clim_year_start, clim_year_end, mask,
            best_code_da, best_shape_da, best_loc_da, best_scale_da,
        )
        return (forecast_det * mask).transpose("T", "Y", "X"), forecast_prob
