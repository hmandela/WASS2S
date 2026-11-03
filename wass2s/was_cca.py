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

``mode_selection="mdl"`` replaces nested mode-search CV by the training-only
max-min MDL selector of Song et al. (2016). Constructor mode counts are upper
bounds, including a possible rank-zero baseline. The Python API keeps ``cv``
as its compatibility default; the operational CLIs explicitly select ``mdl``.
The probability-calibration folds and fixed climatological events are separate
from this mode-selection choice.

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

Changes of 18 September 2026 (scientific audit of the CCA chain)
----------------------------------------------------------------
* The truncation selected by the nested cross-validation now reaches
  ``forecast``: ``cross_validate`` ends with a final selection on the complete
  record (same outer splits, same fold transformations) stored on the calling
  instance (``truncation_``, ``truncation_folds_`` and the fingerprint cache
  read by ``forecast``). ``forecast`` selects it itself when autotuning is on
  and nothing was cached for these fields. See :meth:`_CCACore.finalize_truncation`.
* One probabilistic calibrator: ``forecast``, ``compute_prob`` and the legacy
  ``_compute_tercile_probabilities`` entry point (kernel CCA, CLI consensus)
  use the convention of the strict cross-validator, ``N(pred + bias, var_resid)`` with
  ``bias = mean(obs - hindcast)``, ``var_resid`` the residual variance
  (ddof=1), ``dof = n - 1`` and the non-negativity rule of
  ``apply_physical_support``. The deterministic forecast returned is the
  bias-corrected mean the probabilities are centred on. The predictive spread
  is no longer the MSE, which double-counted the bias.
* ``WAS_CCA_strict._fill_spatial_gaps`` fills partial gaps only; cells never
  observed on the training record (continents on an SST domain) stay NaN and
  are excluded by the validity mask instead of duplicating coastal columns.
* ``tune_truncation`` evaluates every candidate with the engine of the
  template: a ridge template (``WAS_CCA_Regularized*``) is scored with the
  ridge CCA at its own ``alpha`` rather than with the unregularized xeofs CCA.
  ``n_modes=None`` (ridge default) no longer breaks the default grid.
* ``default_truncation_grid`` accepts an optional independent ``y_pca_grid``.
"""

from __future__ import annotations

import copy
import hashlib
import inspect
import os
import warnings
from concurrent.futures import ThreadPoolExecutor

import numpy as np
import xarray as xr
from .was_probability_reference import climatological_terciles
import matplotlib.pyplot as plt

from scipy.stats import gamma, norm, lognorm, expon, weibull_min, t, poisson, nbinom
from scipy.optimize import brentq
from scipy.special import gamma as gamma_function

import functools

import xeofs as xe

from wass2s.utils import *  # noqa: F401,F403
from wass2s.was_cca_engine import ExactCCA, LazyXeofsCCA, exact_cca_supported
from wass2s.was_cca_mdl import select_cca_mdl


__all__ = [
    "WAS_CCA",
    "WAS_CCA_base",
    "WAS_CCA_op",
    "WAS_CCA_strict",
    "WAS_CCA_Regularized",
    "WAS_CCA_Regularized_strict",
    "tune_truncation",
    "TruncationResult",
    "conditional_nonnegative_clip",
    "spatial_mask",
    "check_tercile_calibration",
]

# ``wass2s/__init__.py`` and ``was_cross_validate.py`` both do
# ``from wass2s.was_cca import *``. Both import ``wass2s.utils`` separately, so
# restricting the star export to ``__all__`` does not remove anything they use.


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
    new_args, chunked = [], False
    for a in args:
        if hasattr(a, "dims") and "Y" in a.dims:
            ny = int(a.sizes.get("Y", 1))
            if ny >= 2 * n:
                a = a.chunk({"Y": max(1, ny // n)})
                chunked = True
        new_args.append(a)
    result = xr.apply_ufunc(func, *new_args, **kwargs)
    if chunked:
        from dask import compute
        return compute(*result, scheduler="threads", num_workers=n) if isinstance(result, tuple) else result.compute(scheduler="threads", num_workers=n)
    return result


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


def conditional_nonnegative_clip(deterministic, Predictant):
    """Use physical support declared by metadata, never held-out observations."""
    from wass2s.was_scientific_utils import apply_physical_support
    return apply_physical_support(deterministic, Predictant)


_conditional_nonnegative_clip = conditional_nonnegative_clip


def _year_month_values(t: xr.DataArray):
    """(year, month) integer arrays for a T coordinate, regardless of dtype.

    ``.dt.year``/``.dt.month`` only work on a datetime64 or cftime axis. A
    GCM-derived T coordinate is not always one of those — plain
    integer/float years, or an object-dtype array of Timestamps, both show
    up in practice — and ``.dt`` then raises ``AttributeError``. This covers
    all of those without changing behaviour on a normal datetime axis: an
    integer/float axis is treated as bare years (month defaults to 1, since
    no month information exists to recover), and object-dtype elements are
    read via their own ``.year``/``.month`` attributes.
    """
    vals = np.asarray(t.values)
    if np.issubdtype(vals.dtype, np.datetime64):
        return np.asarray(t.dt.year.values), np.asarray(t.dt.month.values)
    try:
        return np.asarray(t.dt.year.values), np.asarray(t.dt.month.values)  # cftime
    except (AttributeError, TypeError):
        pass
    if np.issubdtype(vals.dtype, np.number):
        years = vals.astype(int)
        return years, np.ones_like(years, dtype=int)
    years = np.array([getattr(x, "year", 1970) for x in vals], dtype=int)
    months = np.array([getattr(x, "month", 1) for x in vals], dtype=int)
    return years, months


def _target_time(Predictant: xr.DataArray, Predictor_for_year: xr.DataArray):
    years, _ = _year_month_values(Predictor_for_year["T"])
    _, months = _year_month_values(Predictant["T"])
    year = int(years[0])
    month = int(months[0])
    return np.array([np.datetime64(f"{year:04d}-{month:02d}-01")], dtype="datetime64[ns]")


def _forecast_predictor_time(Predictor, Predictor_for_year, Predictant):
    """T at which the forecast-year predictor should be placed.

    The common WASS2S usage stamps the historical predictor with the
    predictand's calendar::

        predictor = predictors[i].isel(T=slice(None, -1))
        predictor["T"] = rainfall.sel(T=slice(...))["T"]

    but ``Predictor_for_year`` — usually ``predictors[i].isel(T=[-1])`` — keeps
    the model initialization month, because that assignment only touched the
    sliced copy. Evaluating the predictor trend at an April stamp when it was
    fitted on July stamps offsets it by a lead time. When the historical
    predictor carries the predictand's months, place the forecast predictor on
    the target time; otherwise leave its own initialization time alone.
    """
    try:
        _, pmonths = _year_month_values(Predictor["T"])
        _, ymonths = _year_month_values(Predictant["T"])
        pm = set(np.unique(pmonths).tolist())
        ym = set(np.unique(ymonths).tolist())
    except Exception:
        return Predictor_for_year["T"].values
    if pm == ym:
        return _target_time(Predictant, Predictor_for_year)
    return Predictor_for_year["T"].values



def _predictive_error_stats(observed: xr.DataArray, deterministic: xr.DataArray):
    """Legacy ``(residuals, MSE)`` pair, kept for ``was_cca_2`` (kernel CCA).

    The linear CCA classes no longer use it: they calibrate with
    :func:`_predictive_moments`, the convention of the strict cross-validator.
    """
    resid = observed - deterministic
    mse = (resid ** 2).mean(dim="T", skipna=True)
    return resid, mse


def _predictive_moments(residuals: xr.DataArray):
    """``(variance, bias)`` of an out-of-sample residual sample ``obs - hindcast``.

    Same calibrator as ``was_cross_validate._training_probabilities``: the
    predictive distribution is ``N(pred + bias, variance)`` with ``bias`` the
    mean residual and ``variance`` the residual variance about that mean
    (ddof=1). The cross-validated probabilistic scores (RPSS, GROC)
    therefore verify the calibration that ``forecast`` emits.
    """
    return (residuals.var(dim="T", ddof=1, skipna=True),
            residuals.mean(dim="T", skipna=True))


def _nonnegative_support(reference) -> bool:
    """Whether ``apply_physical_support`` clips this variable at zero.

    Read from metadata (name / ``support_policy``), never from the observed
    minimum, so the cross-validator and ``forecast`` apply one rule.
    """
    from wass2s.was_scientific_utils import apply_physical_support
    try:
        probe = xr.DataArray(np.array([-1.0]), dims=["probe"])
        return bool(float(apply_physical_support(probe, reference).values[0]) >= 0.0)
    except Exception:
        return False


def _residual_dof(n_samples: int) -> int:
    """Degrees of freedom of a residual sample of ``n_samples`` (as in the CV)."""
    return int(max(int(n_samples) - 1, 3))


def _effective_dof(n_clim: int, n_modes) -> int:
    """Legacy helper. ``n_modes=None`` (ridge default) counts as zero."""
    return int(max(int(n_clim) - int(n_modes or 0) - 1, 3))


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

    Use the same empirical climatological boundaries and tie convention as
    verification: PB <= T1, PN in (T1, T2], PA > T2. Finite-sample frequencies
    need not be exactly 1/3, especially with repeated precipitation values;
    a departure alone is not evidence of forecast miscalibration.
    """
    obs = _prepare_tyx(Predictant, "Predictant")
    if clim_year_start is not None and clim_year_end is not None:
        obs = obs.sel(T=slice(str(clim_year_start), str(clim_year_end)))
    T1 = T1.drop_vars("quantile", errors="ignore")
    T2 = T2.drop_vars("quantile", errors="ignore")
    n = np.isfinite(obs).sum(dim="T")
    fb = (obs <= T1).sum(dim="T") / n
    fn = ((obs > T1) & (obs <= T2)).sum(dim="T") / n
    fa = (obs > T2).sum(dim="T") / n
    return xr.Dataset({"freq_below": fb, "freq_near": fn, "freq_above": fa})


# ---------------------------------------------------------------------------
# Truncation handling
# ---------------------------------------------------------------------------
# Module-level switch. Set WAS_CCA_AUTOTUNE=0 to restore the historical
# behaviour, where the constructor values are used verbatim.
def _autotune_default() -> bool:
    return os.environ.get("WAS_CCA_AUTOTUNE", "1") != "0"


def _default_truncation_grid(n_modes, n_pca_modes, y_pca_grid=None):
    """Search grid implied by the constructor values.

    ``n_modes`` and ``n_pca_modes`` are read as **upper bounds**: a model built
    with ``(3, 8)`` searches 1-3 canonical modes and 2-8 PCA modes. The selected
    configuration is therefore never more complex than what the caller asked
    for, and a caller who never touches this still gets a truncation supported
    by the sample rather than an a priori one.

    ``n_modes=None`` (the ridge engine's default, "keep every canonical
    direction") searches four mode counts spread between 1 and the X PCA
    truncation. ``y_pca_grid`` optionally decouples the Y PCA truncation from
    the X one (CPT-like 3-D search); by default the two stay tied.
    """
    nx, _ny = _resolve_pca_modes(n_pca_modes)
    try:
        nx = int(nx)
    except (TypeError, ValueError):
        nx = 8
    if n_modes is None:
        modes = sorted({1, max(1, nx // 3), max(1, (2 * nx) // 3), max(1, nx)})
    else:
        modes = sorted({m for m in range(1, int(n_modes) + 1)})[-4:]
    pcas = sorted({max(2, nx // 4), max(2, nx // 2), max(3, (3 * nx) // 4), nx})
    grid = {"n_modes_grid": tuple(modes), "x_pca_grid": tuple(pcas),
            "score": "mse", "one_se": True, "progress": False}
    if y_pca_grid is not None:
        grid["y_pca_grid"] = tuple(int(v) for v in y_pca_grid)
    return grid


def _field_fingerprint(*arrays):
    """Order- and coordinate-sensitive digest of the complete training fields.

    Dates affect detrending and latitude affects area weights. Sums and sums
    of squares cannot identify a field: even an independent permutation of X
    years preserves those statistics while destroying its relationship to Y.
    Hashing is linear in input size and does not repeat the decompositions.
    """
    digest = hashlib.sha256()
    for da in arrays:
        digest.update(repr(tuple(da.sizes.items())).encode("utf-8"))
        # Canonical numeric representation makes float32/float64 copies agree.
        values = np.array(da.values, dtype="<f8", order="C", copy=True)
        values[np.isnan(values)] = np.nan  # normalize NaN payloads
        digest.update(memoryview(values).cast("B"))
        for dim in da.dims:
            coordinate = np.asarray(da[dim].values)
            digest.update(dim.encode("utf-8"))
            if np.issubdtype(coordinate.dtype, np.datetime64):
                coordinate = coordinate.astype("datetime64[ns]").astype("<i8")
            elif np.issubdtype(coordinate.dtype, np.number):
                coordinate = coordinate.astype("<f8")
            else:
                digest.update(repr(coordinate.tolist()).encode("utf-8"))
                continue
            digest.update(np.ascontiguousarray(coordinate).tobytes())
    return digest.hexdigest()


def _pair_fingerprint(Predictant, Predictor):
    """Fingerprint of ``(Predictor, Predictant)`` in canonical ``(T, Y, X)`` form.

    ``cross_validate`` and ``forecast`` both go through this, so a singleton
    ``M`` dimension or a different dimension order on one side does not
    prevent ``forecast`` from finding the truncation selected for the fields.
    """
    return _field_fingerprint(_prepare_tyx(Predictor, "Predictor"),
                              _prepare_tyx(Predictant, "Predictant"))


def _loo_splits(t: xr.DataArray, nb_omit=1):
    """Leave-one-out splits with the calendar-year buffer of the cross-validator.

    Same contract as ``CustomTimeSeriesSplit.split(T, nb_omit)``: the years
    within ``ceil(nb_omit / 2)`` of the test year are left out of training.
    """
    years, _ = _year_month_values(t)
    years = np.asarray(years)
    gap = int(np.ceil(int(nb_omit) / 2))
    idx = np.arange(years.size)
    out = []
    for i in range(years.size):
        tr = idx[np.abs(years - years[i]) > gap]
        if tr.size >= 2:
            out.append((tr, np.array([i], dtype=int)))
    return out


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
    # None (ridge default: keep every canonical direction) means the ceiling.
    safe_cca = max(1, min(int(ceiling if n_modes is None else n_modes), ceiling))
    return safe_cca, [safe_x, safe_y]


def use_exact_cca(cfg=None) -> bool:
    """Whether the exact engine (:mod:`wass2s.was_cca_engine`) serves this fit.

    ``WAS_CCA_ENGINE=xeofs`` restores the historical xeofs pipeline (for
    A/B comparisons); float or ``use_pca=False`` configurations always keep it.
    """
    if os.environ.get("WAS_CCA_ENGINE", "exact").strip().lower() == "xeofs":
        return False
    if cfg is None:
        return True
    return exact_cca_supported(cfg.get("use_pca", True), cfg.get("n_pca_modes"))


def seed_xeofs_cca(cca, random_state=42):
    """Seed the two PCA preprocessors, which xeofs 3.x leaves unseeded.

    ``BaseModelCrossSet`` forwards ``random_state`` to the cross SVD only; its
    PCA steps run ``randomized_svd(random_state=None)``, so two identical fits
    differ. Only the xeofs fallback / diagnostics path needs this.
    """
    for pca in (getattr(cca, "pca1", None), getattr(cca, "pca2", None)):
        if pca is not None and hasattr(pca, "random_state"):
            pca.random_state = random_state
    return cca


def _make_cca(cfg: dict, n_samples: int):
    """Build a seeded ``xe.cross.CCA`` with fold-compatible mode counts."""
    n_modes, n_pca_modes = _safe_mode_counts(
        cfg["n_modes"], cfg["n_pca_modes"], n_samples
    )
    return seed_xeofs_cca(xe.cross.CCA(
        n_modes=n_modes,
        standardize=cfg.get("standardize", False),
        use_coslat=cfg.get("use_coslat", True),
        use_pca=cfg.get("use_pca", True),
        n_pca_modes=n_pca_modes,
        random_state=cfg.get("random_state", 42),
    ), cfg.get("random_state", 42))


def _new_safe_cca(owner, X_train: xr.DataArray, y_train: xr.DataArray):
    """Legacy helper kept for compatibility: build a fold-safe xeofs CCA."""
    n_samples = min(X_train.sizes["T"], y_train.sizes["T"])
    return _make_cca(
        {
            "n_modes": owner.n_modes,
            "n_pca_modes": owner.n_pca_modes,
            "standardize": owner.standardize,
            "use_coslat": owner.use_coslat,
            "use_pca": owner.use_pca,
        },
        n_samples,
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
    return _split_predictions(None, cfg, X_train, y_train, X_test,
                              [(cfg["n_modes"], cfg["n_pca_modes"])])[0]


def _split_predictions(template, cfg, X_train, y_train, X_test, candidates):
    """Predictions of ``X_test`` for every ``(n_modes, n_pca_modes)`` candidate.

    The candidate-independent work -- fold preprocessing and, with the exact
    engine, the PCA of both fields -- is done once for the split; a failing
    candidate yields its exception in place of a prediction. Ridge templates
    (:class:`_RidgeCCAEngine`) are scored with their own engine at their own
    ``alpha``: selecting the truncation with the unregularized CCA and then
    applying it to the shrunk solution would validate a model that is never
    used.
    """
    if isinstance(template, _RidgeCCAEngine):
        return template._ridge_candidates(X_train, y_train, X_test, candidates)
    state, Xtr, Ytr = _fit_preprocessing(X_train, y_train)
    n_samples = min(Xtr.sizes["T"], Ytr.sizes["T"])
    Xte = _apply_preprocessing(X_test, state["x_ref"], state["x_mask"], "X_test")
    out = []
    exact = all(use_exact_cca(dict(cfg, n_pca_modes=c[1])) for c in candidates)
    if exact:
        try:
            engine = ExactCCA(Xtr, Ytr, cfg.get("standardize", False),
                              cfg.get("use_coslat", True))
            Zx = engine.bx.project(Xte)
        except Exception as exc:  # e.g. a field without variance in this fold
            return [exc for _ in candidates]
    for n_modes, n_pca in candidates:
        try:
            if exact:
                nm, npca = _safe_mode_counts(n_modes, n_pca, n_samples)
                y_pred = engine.predict(Xte, nm, npca, Zx=Zx)
            else:
                model = _make_cca(dict(cfg, n_modes=n_modes, n_pca_modes=n_pca),
                                  n_samples).fit(Xtr, Ytr, dim="T")
                y_pred = model.predict(Xte)
                y_pred = model.inverse_transform(model.transform(Xte), y_pred)[1]
            out.append(_to_yx(y_pred))
        except Exception as exc:
            out.append(exc)
    if len(candidates) == 1 and isinstance(out[0], Exception):
        raise out[0]
    return out


def _candidate_predictor(template):
    """``(cfg, X_train, y_train, X_test) -> prediction`` for the template's engine.

    Kept for external callers; :func:`tune_truncation` itself scores all
    candidates of a split at once through :func:`_split_predictions`.
    """
    if not isinstance(template, _RidgeCCAEngine):
        return _fit_predict_once

    def _ridge_once(cfg, X_train, y_train, X_test):
        out = _split_predictions(template, cfg, X_train, y_train, X_test,
                                 [(cfg["n_modes"], cfg["n_pca_modes"])])[0]
        if isinstance(out, Exception):
            raise out
        return out

    return _ridge_once


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


def _jackknife_se(partial) -> float:
    """Leave-one-out jackknife standard error from leave-one-fold-out scores."""
    partial = np.asarray(partial, dtype=float)
    partial = partial[np.isfinite(partial)]
    if partial.size < 3:
        return np.nan
    n = partial.size
    return float(np.sqrt((n - 1) / n * np.sum((partial - partial.mean()) ** 2)))


def _score_with_jackknife(pred, obs, weights, kind, fold_of):
    """Aggregate score, its jackknife SE, and the leave-one-fold-out scores.

    The third element is the vector of leave-one-fold-out scores, aligned with
    ``np.unique(fold_of)`` and padded with NaN where a fold could not be
    scored. Because every truncation is evaluated on the same folds, these
    vectors are comparable term by term across candidates, which is what makes
    the *paired* standard error of a difference computable in
    ``tune_truncation``. The marginal SE returned here is the right measure of
    how uncertain one score is; it is the wrong measure of how uncertain the
    gap between two scores is.
    """
    full = _aggregate_score(pred, obs, weights, kind)
    folds = np.unique(fold_of)
    if folds.size < 3:
        return full, np.nan, None
    partial = np.full(folds.size, np.nan, dtype=float)
    for i, f in enumerate(folds):
        keep = np.where(fold_of != f)[0]
        if keep.size < 2:
            continue
        partial[i] = _aggregate_score(
            pred.isel(T=keep), obs.isel(T=keep), weights, kind
        )
    return full, _jackknife_se(partial), partial


def _prediction_budget_bytes() -> int:
    """Memory allowed for the out-of-fold predictions held during a tuning."""
    try:
        return max(1, int(float(os.environ.get("WAS_CCA_TUNING_MEMORY_MB", "1024")) * 2 ** 20))
    except ValueError:
        return 2 ** 30


def _score_candidates_by_split(X, Y, splits, combos, cfg_base, template, weights,
                               score, fold_of, n_jobs=None, progress=False):
    """Out-of-fold predictions of every candidate, split by split, then scores.

    Split-major: each training split is preprocessed and decomposed once and
    serves all candidates (:func:`_split_predictions`). Predictions are kept
    by chunks of candidates sized by ``WAS_CCA_TUNING_MEMORY_MB``
    (default 1024), then scored exactly as before (area-weighted aggregate,
    leave-one-fold-out partial scores for the paired one-SE rule).
    """
    n_test = int(sum(len(te) for _, te in splits))
    cells = int(np.prod([Y.sizes[d] for d in Y.dims if d != "T"]))
    per_candidate = max(1, n_test * cells * 8)
    chunk = max(1, min(len(combos), _prediction_budget_bytes() // per_candidate))
    workers = _grid_workers() if n_jobs is None else max(1, int(n_jobs))
    T_all = Y["T"].values
    template_yx = Y.isel(T=0, drop=True)

    def _one_split(k, chunk_combos):
        train_index, test_index = splits[k]
        return _split_predictions(
            template, cfg_base, X.isel(T=train_index), Y.isel(T=train_index),
            X.isel(T=test_index), [(nm, [nx, ny]) for nm, nx, ny in chunk_combos])

    rows = []
    for start in range(0, len(combos), chunk):
        chunk_combos = combos[start:start + chunk]
        if progress:
            print(f"  truncation candidates {start + 1}-{start + len(chunk_combos)}"
                  f"/{len(combos)} on {len(splits)} splits", flush=True)
        if workers > 1 and len(splits) > 1:
            with ThreadPoolExecutor(max_workers=min(workers, len(splits))) as pool:
                per_split = list(pool.map(lambda k: _one_split(k, chunk_combos),
                                          range(len(splits))))
        else:
            per_split = [_one_split(k, chunk_combos) for k in range(len(splits))]

        for j, (nm, nx, ny) in enumerate(chunk_combos):
            failures = [res[j] for res in per_split if isinstance(res[j], Exception)]
            if failures:  # a truncation the sample cannot support
                rows.append({"n_modes": nm, "n_pca_x": nx, "n_pca_y": ny,
                             "score": np.nan, "se": np.nan, "partial": None,
                             "error": repr(failures[0])})
                continue
            values = np.concatenate(
                [res[j].transpose("T", "Y", "X").values
                 for res in per_split], axis=0)
            times = np.concatenate([T_all[te] for _, te in splits])
            pred_all = xr.DataArray(
                values, dims=("T", "Y", "X"),
                coords={"T": times, "Y": template_yx["Y"], "X": template_yx["X"]})
            obs_all = Y.sel(T=pred_all["T"])
            value, se, partial = _score_with_jackknife(
                pred_all, obs_all, weights, score, fold_of)
            rows.append({"n_modes": nm, "n_pca_x": nx, "n_pca_y": ny,
                         "score": value, "se": se, "partial": partial, "error": None})
            del values, pred_all
        del per_split
    return rows


def aic_truncation(X, Y, n_modes_grid=(1, 2, 3, 4, 5), x_pca_grid=(4, 6, 8, 10, 12),
                   y_pca_grid=None, template=None, **_ignored):
    """Experimental Bartlett-AIC heuristic for CCA/PCA truncation.

    Same grid semantics as :func:`tune_truncation` (``y_pca_grid=None`` ties
    the Y truncation to X) and the same fields (already in the fit space), but
    no cross-validation: one exact decomposition of the training years scores
    every candidate (:func:`wass2s.was_cca_engine.bartlett_aic`). Comparisons
    across different PCA spaces are heuristic, not a common-data likelihood.
    MDL is the recommended training-only selector. The null model is always
    available; saturated/unit-correlation candidates are rejected.
    """
    from wass2s.was_cca_engine import aic_truncation_table
    if template is not None and not use_exact_cca(template._cfg):
        raise ValueError("AIC selection requires the exact CCA engine with integer PCA bounds.")
    X = _prepare_tyx(X, "X")
    Y = _prepare_tyx(Y, "Y")
    tied = y_pca_grid is None
    y_grid = tuple(x_pca_grid) if tied else tuple(y_pca_grid)
    combos = []
    for nm in sorted({0, *(int(v) for v in n_modes_grid)}):
        for nx in sorted(set(int(v) for v in x_pca_grid)):
            for ny in sorted(set(int(v) for v in ((nx,) if tied else y_grid))):
                if 0 <= nm <= min(nx, ny) and min(nx, ny) >= 1:
                    combos.append((nm, nx, ny))
    if not combos:
        raise ValueError("Empty grid: n_modes must be <= min(n_pca_X, n_pca_Y).")
    _, Xtr, Ytr = _fit_preprocessing(X, Y)
    try:
        engine = ExactCCA(Xtr, Ytr, getattr(template, "standardize", False),
                          getattr(template, "use_coslat", True))
    except ValueError as exc:
        if "training field has no variance" not in str(exc):
            raise
        return TruncationResult(n_modes=0, n_pca_modes=[1, 1], score=0.0,
                                se=np.nan, method="aic", metric="bartlett_aic",
                                rule="min", table=[], experimental=True,
                                fallback_reason="constant_training_field")
    n = int(min(Xtr.sizes["T"], Ytr.sizes["T"]))
    rows = aic_truncation_table(engine, [(nm, [nx, ny]) for nm, nx, ny in combos], n,
                                cap=_safe_mode_counts,
                                detrend_dof=getattr(template, "_mdl_detrend_dof", 0))
    order = sorted(range(len(rows)), key=lambda i: (rows[i][2], combos[i][0],
                                                     combos[i][1] + combos[i][2]))
    nm, nx, ny = combos[order[0]]
    table = [{"n_modes": combos[i][0], "n_pca_x": combos[i][1], "n_pca_y": combos[i][2],
              "resolved": list(rows[i][1]), "score": rows[i][2]} for i in order]
    return TruncationResult(n_modes=int(nm), n_pca_modes=[int(nx), int(ny)],
                            score=float(rows[order[0]][2]), se=np.nan,
                            method="aic", metric="bartlett_aic", rule="min", table=table,
                            experimental=True)


def tune_truncation(
    X: xr.DataArray,
    Y: xr.DataArray,
    splits,
    n_modes_grid=(1, 2, 3, 4, 5),
    x_pca_grid=(4, 6, 8, 10, 12),
    y_pca_grid=None,
    template=None,
    score="mse",
    one_se=True,
    paired_se=True,
    se_multiplier=2.0,
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
        A ridge template (``WAS_CCA_Regularized*``) is also scored with its
        own engine and ``alpha`` (see :func:`_candidate_predictor`).
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
    rows = _score_candidates_by_split(
        X, Y, splits, combos, cfg_base, template, weights, score, fold_of,
        n_jobs=n_jobs, progress=progress)

    lower_is_better = score == "mse"
    finite = [r for r in rows if np.isfinite(r["score"])]
    if not finite:
        raise RuntimeError("No truncation could be evaluated; check the grid and the data.")

    rows_sorted = sorted(finite, key=lambda r: r["score"], reverse=not lower_is_better)
    best = rows_sorted[0]
    rule = "best"

    if one_se and np.isfinite(best["se"]):
        # Every candidate is scored on the SAME folds, so their fold-to-fold
        # fluctuations are largely common (a hard year is hard for all of
        # them). The uncertainty that matters when deciding whether candidate k
        # is distinguishable from the best one is the SE of the paired
        # difference, not the marginal SE of either score. Using the marginal
        # SE inflates the tolerance several-fold and admits the whole grid,
        # after which the parsimony clause returns the floor every time,
        # regardless of the data.
        best_partial = best.get("partial")
        paired = bool(paired_se) and best_partial is not None

        def _tolerance(candidate):
            if not paired:
                return best["se"]
            cand_partial = candidate.get("partial")
            if cand_partial is None or len(cand_partial) != len(best_partial):
                return best["se"]
            se_pair = _jackknife_se(np.asarray(cand_partial) - np.asarray(best_partial))
            # A degenerate paired SE (identical or near-identical fold scores)
            # must not silently widen the gate back to the marginal value.
            if not np.isfinite(se_pair):
                return 0.0
            # The paired SE is correctly calibrated but, by construction, a
            # ~1 SE gate follows the noise winner when no truncation is truly
            # better. se_multiplier restores parsimony in that null case
            # without giving up the sensitivity the paired SE buys. Calibrated
            # on synthetic folds (24 years, common fold noise): with k=2.0 a
            # real gain is detected in 100% of draws, while a null case falls
            # back to the floor in 95% and a harmful-complexity case in 100%.
            # The marginal rule this replaces detected a real gain in 0%.
            return float(se_multiplier) * se_pair

        if lower_is_better:
            within = [r for r in finite if r["score"] <= best["score"] + _tolerance(r)]
        else:
            within = [r for r in finite if r["score"] >= best["score"] - _tolerance(r)]
        if within:
            chosen = min(within, key=lambda r: (r["n_modes"], r["n_pca_x"] + r["n_pca_y"]))
            if chosen is not best:
                rule = "one-se-paired" if paired else "one-se"
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
        best_guess, error_samples, first_tercile, second_tercile, nonneg=False
    ):
        """Non-parametric probabilities from the historical residual sample.

        Same dressing as the strict cross-validator (``was_crossfit._probabilities``):
        one residual per hindcast year added to the prediction, the sample
        clipped at zero when the variable is non-negative, and the categories
        ``sample <= T1``, ``T1 < sample <= T2``, ``sample > T2`` — the closed
        lower bound is the one the observed classes use.
        """
        best_guess = np.asarray(best_guess, dtype=float)
        valid_errors = np.asarray(error_samples, dtype=float)
        valid_errors = valid_errors[np.isfinite(valid_errors)]
        out = np.full((3, best_guess.size), np.nan, dtype=float)
        if valid_errors.size == 0 or not (np.isfinite(first_tercile) and np.isfinite(second_tercile)):
            return out
        for i, prediction in enumerate(best_guess):
            if not np.isfinite(prediction):
                continue
            sample = prediction + valid_errors
            if nonneg:
                sample = np.clip(sample, 0.0, None)
            pb = np.mean(sample <= first_tercile)
            pn = np.mean((sample > first_tercile) & (sample <= second_tercile))
            out[0, i] = pb
            out[1, i] = pn
            out[2, i] = 1.0 - pb - pn
        return out

    # -- engine ------------------------------------------------------------
    def _probability_field(
        self, deterministic, error_samples, error_variance, T1_emp, T2_emp,
        dof, mask, nonneg=True, discrete=False,
        best_code_da=None, best_shape_da=None, best_loc_da=None, best_scale_da=None,
        calibrated_mean=None,
    ):
        """PB/PN/PA field.

        ``deterministic`` is the raw prediction, dressed with ``error_samples``
        (raw residuals ``obs - hindcast``) in the non-parametric case. The
        parametric families are centred on ``calibrated_mean`` — the
        bias-corrected prediction — with ``error_variance`` the residual
        variance; when it is omitted the raw prediction is used (legacy
        callers such as the kernel CCA). Zero residual variance yields the
        point-mass category of the calibrated mean, as in the cross-validator.
        """
        deterministic = _prepare_tyx(deterministic, "deterministic")
        dm = str(self.dist_method).strip().lower()

        if dm == "nonparam":
            error_samples = _prepare_tyx(error_samples, "error_samples").rename({"T": "T_error"})
            kernel = self.calculate_tercile_probabilities_nonparametric
            kwargs = {"nonneg": nonneg} if "nonneg" in inspect.signature(kernel).parameters else {}
            prob = _pufunc(
                kernel,
                deterministic, error_samples, T1_emp, T2_emp,
                input_core_dims=[("T",), ("T_error",), (), ()],
                output_core_dims=[("probability", "T")],
                vectorize=True, kwargs=kwargs, dask="parallelized", output_dtypes=[float],
                dask_gufunc_kwargs={"output_sizes": {"probability": 3}, "allow_rechunk": True},
            )
            return _finalize_probabilities(prob, mask)

        centre = deterministic if calibrated_mean is None else _prepare_tyx(
            calibrated_mean, "calibrated_mean")

        if dm == "bestfit":
            if any(v is None for v in (best_code_da, best_shape_da, best_loc_da, best_scale_da)):
                raise ValueError(
                    "dist_method='bestfit' requires best_code_da, best_shape_da, "
                    "best_loc_da and best_scale_da."
                )
            code_da = self._guard_discrete_codes(best_code_da, discrete)
            # Category boundaries stay on the observed climatological reference.
            # The selected distribution controls the predictive CDF only.
            if getattr(self, "tercile_reference", "climatology") == "training":
                T1, T2 = _pufunc(
                    self._ppf_terciles_from_code,
                    code_da, best_shape_da, best_loc_da, best_scale_da,
                    input_core_dims=[(), (), (), ()], output_core_dims=[(), ()],
                    vectorize=True, dask="parallelized", output_dtypes=[float, float],
                )
            else:
                T1, T2 = T1_emp, T2_emp

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
            centre, error_variance, T1, T2, code_da,
            input_core_dims=[("T",), (), (), (), ()],
            output_core_dims=[("probability", "T")],
            vectorize=True, kwargs={"dof": dof, "nonneg": nonneg},
            dask="parallelized", output_dtypes=[float],
            dask_gufunc_kwargs={"output_sizes": {"probability": 3}, "allow_rechunk": True},
        )
        prob = prob.assign_coords(probability=("probability", ["PB", "PN", "PA"]))
        # A residual sample without spread is a point mass, not a NaN field.
        point = xr.concat(
            [(centre <= T1).astype(float), ((centre > T1) & (centre <= T2)).astype(float),
             (centre > T2).astype(float)],
            dim=xr.IndexVariable("probability", ["PB", "PN", "PA"]),
        ).where(np.isfinite(centre) & np.isfinite(T1) & np.isfinite(T2))
        prob = xr.where(error_variance == 0, point, prob)
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

    def _tercile_context(self, Predictant, clim_year_start, clim_year_end, n_resid=None):
        """Empirical terciles of the climatology and the residual dof.

        ``dof`` follows the cross-validator: ``n - 1`` for a residual sample
        of ``n`` out-of-sample years (the climatology length when no residual
        sample is given). The CCA modes are not subtracted: the residuals are
        out of sample, so the fit consumed none of their degrees of freedom.
        """
        clim = Predictant.sel(T=slice(str(clim_year_start), str(clim_year_end)))
        if clim.sizes.get("T", 0) < 3:
            raise ValueError("Not enough years in climatology period for terciles.")
        terciles = climatological_terciles(clim)[0]
        T1 = terciles.isel(quantile=0, drop=True)
        T2 = terciles.isel(quantile=1, drop=True)
        dof = _residual_dof(clim.sizes["T"] if n_resid is None else n_resid)
        return T1, T2, dof

    def _calibrated_probabilities(
        self, deterministic, Predictant, residuals, clim_year_start,
        clim_year_end, mask, best_code_da=None, best_shape_da=None,
        best_loc_da=None, best_scale_da=None,
    ):
        """``(calibrated_mean, probabilities)`` with the cross-validator's calibrator.

        ``deterministic`` is dressed with ``residuals`` (``obs - hindcast``,
        out of sample): mean corrected by the mean residual, spread equal to
        the residual variance (ddof=1), ``dof = n - 1``, non-negativity from
        the declared physical support. The calibrated mean is clipped with
        ``apply_physical_support`` exactly like the cross-validator's
        ``calibrated_mean``; it is what the probabilities are centred on and
        what ``forecast`` returns as its deterministic field.
        """
        residuals = _prepare_tyx(residuals, "residuals")
        T1_emp, T2_emp, dof = self._tercile_context(
            Predictant, clim_year_start, clim_year_end, n_resid=residuals.sizes["T"]
        )
        error_variance, bias = _predictive_moments(residuals)
        calibrated = conditional_nonnegative_clip(deterministic + bias, Predictant)
        prob = self._probability_field(
            deterministic, residuals, error_variance, T1_emp, T2_emp, dof, mask,
            nonneg=_nonnegative_support(Predictant),
            discrete=_looks_discrete(Predictant),
            best_code_da=best_code_da, best_shape_da=best_shape_da,
            best_loc_da=best_loc_da, best_scale_da=best_scale_da,
            calibrated_mean=calibrated,
        )
        return calibrated.transpose("T", "Y", "X"), prob

    def compute_prob(
        self, Predictant, clim_year_start, clim_year_end, hindcast_det,
        best_code_da=None, best_shape_da=None, best_loc_da=None, best_scale_da=None,
    ):
        """Hindcast tercile probabilities, dims (probability, T, Y, X).

        Diagnostic (full-pool) calibration with the same calibrator as
        ``forecast``: ``N(hindcast + bias, var_resid)`` where the residuals are
        those of ``hindcast_det`` itself.
        """
        Predictant, hindcast_det = _align_observation_and_hindcast(Predictant, hindcast_det)
        return self._calibrated_probabilities(
            hindcast_det, Predictant, Predictant - hindcast_det,
            clim_year_start, clim_year_end, spatial_mask(Predictant),
            best_code_da, best_shape_da, best_loc_da, best_scale_da,
        )[1]

    def _compute_tercile_probabilities(
        self, Predictant, deterministic, clim_year_start, clim_year_end,
        error_samples, error_variance,
        best_code_da=None, best_shape_da=None, best_loc_da=None, best_scale_da=None,
    ):
        """Backwards-compatible entry point (kernel CCA, CLI consensus).

        Same calibrator as ``forecast``: ``error_samples`` (residuals
        ``obs - hindcast``) give the bias-corrected centre and the residual
        variance; ``error_variance`` only serves the legacy ``N(pred, MSE)``
        path taken when no residual sample is supplied.
        """
        Predictant = _prepare_tyx(Predictant, "Predictant")
        mask = spatial_mask(Predictant)
        if error_samples is not None:
            return self._calibrated_probabilities(
                _prepare_tyx(deterministic, "deterministic"), Predictant, error_samples,
                clim_year_start, clim_year_end, mask,
                best_code_da, best_shape_da, best_loc_da, best_scale_da,
            )[1]
        T1_emp, T2_emp, dof = self._tercile_context(
            Predictant, clim_year_start, clim_year_end
        )
        return self._probability_field(
            deterministic, error_samples, error_variance, T1_emp, T2_emp, dof, mask,
            nonneg=_nonnegative_support(Predictant),
            discrete=_looks_discrete(Predictant),
            best_code_da=best_code_da, best_shape_da=best_shape_da,
            best_loc_da=best_loc_da, best_scale_da=best_scale_da,
        )


def _compute_probability_field(
    owner, deterministic, error_samples, error_variance, T1_emp, T2_emp,
    dof, mask, best_code_da=None, best_shape_da=None,
    best_loc_da=None, best_scale_da=None,
):
    """Legacy module-level entry point; delegates to the shared engine."""
    return owner._probability_field(
        deterministic, error_samples, error_variance, T1_emp, T2_emp, dof, mask,
        best_code_da=best_code_da, best_shape_da=best_shape_da,
        best_loc_da=best_loc_da, best_scale_da=best_scale_da,
    )


# ---------------------------------------------------------------------------
# Shared CCA core
# ---------------------------------------------------------------------------
class _CCACore(_ProbabilityEngine):
    """Preprocessing, fitting, prediction, tuning and plotting."""

    # 'cv' (cross-validated grid, historical) or 'aic' (Bartlett-corrected AIC
    # on the training years, no inner folds); cross_validate(truncation_selection=...)
    truncation_selection = "cv"

    def __init__(self, n_modes=4, n_pca_modes=8, standardize=False,
                 use_coslat=True, use_pca=True, dist_method="nonparam",
                 auto_truncation=None, mode_selection="cv"):
        self.n_modes = n_modes
        self.n_pca_modes = n_pca_modes
        self.standardize = standardize
        self.use_coslat = use_coslat
        self.use_pca = use_pca
        self.dist_method = dist_method

        # None defers to the module switch (env WAS_CCA_AUTOTUNE).
        self.auto_truncation = auto_truncation
        if mode_selection not in ("cv", "mdl", "aic"):
            raise ValueError("mode_selection must be 'cv', 'mdl' or 'aic'.")
        self.mode_selection = mode_selection
        self._mdl_baseline = None
        self._truncation_cache = {}
        # Read by the strict cross-validator's calibrator (``nonneg`` of the
        # exponential family); set from the predictand's declared support by
        # cross_validate so both sides apply the rule of _nonnegative_support.
        self.positive_support = False

        self.cca = None
        self.cca_model = None
        self._engine = None
        self._engine_modes = None
        self._prep = None
        self.truncation_ = None
        self.truncation_folds_ = None

    # -- configuration -----------------------------------------------------
    def autotune_enabled(self) -> bool:
        return _autotune_default() if self.auto_truncation is None else bool(self.auto_truncation)

    def default_truncation_grid(self, y_pca_grid=None):
        """Mode-selection settings for CV or training-only spectral MDL.

        ``y_pca_grid`` optionally decouples the Y PCA truncation from the X one
        (CPT-like 3-D search); the default keeps them tied.
        """
        if self.mode_selection == "mdl":
            if y_pca_grid is not None:
                raise ValueError("MDL uses integer PCA upper bounds, not a CV grid.")
            return {"method": "mdl", "max_modes": self.n_modes,
                    "max_pca_modes": self.n_pca_modes}
        return _default_truncation_grid(self.n_modes, self.n_pca_modes, y_pca_grid)

    def remember_truncation(self, key, result):
        if hasattr(self, "detrend"):
            result["detrend"] = self.detrend
        self._truncation_cache[key] = result

    def recall_truncation(self, key):
        return self._truncation_cache.get(key)

    @property
    def _cfg(self):
        return {
            "n_modes": self.n_modes,
            "n_pca_modes": self.n_pca_modes,
            "standardize": self.standardize,
            "use_coslat": self.use_coslat,
            "use_pca": self.use_pca,
        }

    # -- legacy helpers kept for API compatibility -------------------------
    _drop_member_dim = staticmethod(lambda da: da.isel(M=0, drop=True) if "M" in da.dims else da)
    _to_xeofs_dims = staticmethod(_to_latlon)
    _from_xeofs_dims = staticmethod(_to_yx)
    _rename_to_latlon = staticmethod(_to_latlon)
    _rename_to_YX = staticmethod(_to_yx)
    _normalize_probabilities = staticmethod(_normalize_probabilities)

    @staticmethod
    def _ensure_tyx(da, name="data"):
        return _prepare_tyx(da, name)

    @staticmethod
    def _spatial_mask(da):
        return spatial_mask(da)

    @staticmethod
    def _fill_train_data(da):
        return da.fillna(da.mean(dim="T", skipna=True)).fillna(0)

    @staticmethod
    def _fill_test_data(test_da, train_da):
        return test_da.fillna(train_da.mean(dim="T", skipna=True)).fillna(0)

    @staticmethod
    def _make_forecast_time(Predictant, Predictor_for_year):
        return _target_time(Predictant, Predictor_for_year)

    @staticmethod
    def _safe_drop_vars(da, names):
        existing = [n for n in names if n in da.coords or n in da.variables]
        return da.drop_vars(existing) if existing else da

    def _new_cca(self):
        """Legacy constructor; prefer ``fit_cca``, which caps modes per fold."""
        return _make_cca(self._cfg, 10 ** 6)

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
        """Fit on the training fold.

        Predictions come from the exact engine (:class:`ExactCCA`, identical
        to xeofs with an exact SVD, deterministic). ``cca_model`` is a
        :class:`LazyXeofsCCA`: the xeofs object behind the diagnostics
        (components, scores, explained variances) is only fitted if one of
        them is requested. ``WAS_CCA_ENGINE=xeofs`` restores the historical
        path.
        """
        X_final, y_final = self.preprocess_data(X_train, y_train)
        self._mdl_baseline = None
        if self.n_modes == 0:
            # A rank-zero MDL model has no predictor-dependent anomaly.
            # Restoring the fold trend/scaling is the caller's responsibility.
            self._mdl_baseline = y_final.mean("T", skipna=True)
            self.cca = self.cca_model = self._engine = self._engine_modes = None
            return None
        n_samples = min(X_final.sizes["T"], y_final.sizes["T"])
        cfg = self._cfg
        self.cca = _make_cca(cfg, n_samples)
        self._engine = None
        if use_exact_cca(cfg):
            self._engine = ExactCCA(X_final, y_final, cfg["standardize"], cfg["use_coslat"])
            self._engine_modes = _safe_mode_counts(cfg["n_modes"], cfg["n_pca_modes"], n_samples)
            self.cca_model = LazyXeofsCCA(functools.partial(_make_cca, cfg, n_samples),
                                          X_final, y_final, engine=self._engine)
        else:
            self.cca_model = self.cca.fit(X_final, y_final, dim="T")
        return self.cca_model

    def _predict_prepared(self, X_prepared):
        if self._mdl_baseline is not None:
            return _to_yx(self._mdl_baseline.expand_dims(T=X_prepared["T"])).transpose("T", "Y", "X")
        engine = getattr(self, "_engine", None)
        if engine is not None:
            y_pred = engine.predict(X_prepared, *self._engine_modes)
            return _to_yx(y_pred).transpose("T", "Y", "X")
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
    def select_mdl(self, X, Y, apply=True, max_modes=None, max_pca_modes=None):
        """Select PCA/CCA ranks from TRAIN only, with no validation refits.

        Inputs are already in this class's fitting space. The same two spatial
        decompositions serve the spectral MDL calculation and prediction.
        See ``was_cca_mdl`` for the real-data criterion and rank guards.
        """
        if isinstance(self, _RidgeCCAEngine):
            raise ValueError("The spectral MDL criterion requires unregularized CCA.")
        if not use_exact_cca(self._cfg):
            raise ValueError("MDL requires the exact CCA engine and integer PCA bounds.")
        max_modes = self.n_modes if max_modes is None else max_modes
        max_pca_modes = self.n_pca_modes if max_pca_modes is None else max_pca_modes
        X_final, Y_final = self.preprocess_data(X, Y)
        self._mdl_baseline = None
        self.cca = self.cca_model = self._engine = self._engine_modes = None
        try:
            self._engine = ExactCCA(X_final, Y_final, self.standardize, self.use_coslat)
        except ValueError as exc:
            if "training field has no variance" not in str(exc):
                raise
            # Constant training fields cannot carry a shared anomaly.
            self._mdl_baseline = Y_final.mean("T", skipna=True)
            result = {"n_modes": 0, "n_pca_modes": [1, 1], "metric": "mdl",
                      "score": 0.0, "rule": "mdl", "table": [],
                      "n_samples": int(Y_final.sizes["T"]),
                      "fallback_reason": "constant_training_field"}
        else:
            result = select_cca_mdl(
                self._engine.bx, self._engine.by, max_modes, max_pca_modes,
                detrend_dof=getattr(self, "_mdl_detrend_dof", 0))
            self._engine_modes = (result["n_modes"], result["n_pca_modes"])
        result = TruncationResult(result)
        result["method"] = "mdl"
        if hasattr(self, "detrend"):
            result["detrend"] = self.detrend
        result["selection_bounds"] = {
            "max_modes": max_modes,
            "max_pca_modes": list(_resolve_pca_modes(max_pca_modes))}
        self.truncation_ = result
        if apply:
            self.apply_truncation(result)
        if result["n_modes"] > 0:
            cfg = dict(self._cfg, n_modes=result["n_modes"],
                       n_pca_modes=result["n_pca_modes"])
            n_samples = int(Y_final.sizes["T"])
            self.cca_model = LazyXeofsCCA(
                functools.partial(_make_cca, cfg, n_samples), X_final, Y_final,
                engine=self._engine)
        return result

    def compute_model_mdl(self, X_train, y_train, X_test, *, max_modes=None,
                          max_pca_modes=None):
        """Select and predict with one pair of training-field decompositions."""
        self.select_mdl(X_train, y_train, max_modes=max_modes,
                        max_pca_modes=max_pca_modes)
        X_prepared, _ = self.preprocess_test_data(X_test)
        return self._predict_prepared(X_prepared)

    def apply_truncation(self, result=None):
        """Adopt a stored :class:`TruncationResult` on this instance."""
        result = result or self.truncation_
        if result is None:
            raise ValueError("No truncation result available; call tune() first.")
        self.n_modes = result["n_modes"]
        self.n_pca_modes = result["n_pca_modes"]
        return self

    def truncation_state(self):
        """Current ``(n_modes, n_pca_modes)``, for save/restore around a loop."""
        return self.n_modes, self.n_pca_modes

    def restore_truncation(self, state):
        self.n_modes, self.n_pca_modes = state
        return self

    def tune(self, X, Y, splits, apply=True, **kwargs):
        """Select the truncations by CV and (by default) adopt them.

        ``X`` and ``Y`` must already be in the space this class fits in
        (use :meth:`finalize_truncation` to start from raw fields). The
        selection is stored in ``self.truncation_``; with ``apply=True`` it
        also replaces ``n_modes`` / ``n_pca_modes`` on this instance.

        How ``cross_validate`` uses it
        ------------------------------
        The nested cross-validation calls ``tune`` on *copies* of the model,
        inside every outer training fold (and again inside every calibration
        block), so the hindcast is produced by a truncation that may differ
        from fold to fold; those per-fold choices are reported in
        ``self.truncation_folds_``. The caller's instance keeps its
        constructor values: a truncation is a property of a *(predictor,
        season, domain)* triple and one instance is commonly reused across a
        loop of predictors. At the end ``cross_validate`` runs
        :meth:`finalize_truncation` on the complete record (same outer splits,
        same transformations), which fills ``self.truncation_`` and caches the
        result under the fingerprint of the fields; :meth:`forecast` applies
        it for those fields only, for the duration of the call.
        """
        kwargs.setdefault("template", self)
        if (self.mode_selection == "aic" or
                str(getattr(self, "truncation_selection", "cv")).lower() == "aic"):
            if isinstance(self, _RidgeCCAEngine):
                warnings.warn("truncation_selection='aic' does not apply to the ridge CCA "
                              "(shrunk correlations); cross-validated selection is used.",
                              RuntimeWarning, stacklevel=2)
                result = tune_truncation(X, Y, splits, **kwargs)
            else:
                result = aic_truncation(X, Y, **kwargs)
        else:
            result = tune_truncation(X, Y, splits, **kwargs)
        if hasattr(self, "detrend"):
            result["detrend"] = self.detrend
        self.truncation_ = result
        if apply:
            self.n_modes = result["n_modes"]
            self.n_pca_modes = result["n_pca_modes"]
        return result

    def _tuning_fields(self, X, Y, clim_year_start, clim_year_end):
        """Raw ``(X, Y)`` into the fit space, as the cross-validator does per fold."""
        X_ready, Y_ready = self._plot_prepare(X, Y, clim_year_start, clim_year_end)
        return X_ready.fillna(0.0), Y_ready.fillna(0.0)

    def finalize_truncation(self, Predictor, Predictant, clim_year_start, clim_year_end,
                            splits=None, nb_omit=1, **grid):
        """Select the truncation on the complete record and remember it for ``forecast``.

        The record is transformed exactly as a training fold is in the
        cross-validator (:meth:`_tuning_fields`: spatial filling for the strict
        class, detrending, standardization), then :meth:`tune` scores the grid
        on ``splits`` — by default leave-one-out with the same calendar buffer
        as ``WAS_Cross_Validator`` (``nb_omit``) — without modifying
        ``n_modes`` / ``n_pca_modes``. The result goes to ``self.truncation_``
        and to the cache keyed by the fingerprint of ``(Predictor, Predictant)``
        in canonical ``(T, Y, X)`` form, which :meth:`forecast` looks up with
        the fields it receives. ``grid`` is the ``tune_truncation`` grid the
        nested selection used (``cross_validate`` passes its own, verbatim);
        without it :meth:`default_truncation_grid` applies. ``nested`` /
        ``inner_max_folds`` are ignored here.
        """
        X = _prepare_tyx(Predictor, "Predictor")
        Y = _prepare_tyx(Predictant, "Predictant")
        # The grid the cross-validator tuned with, verbatim, or the default one.
        kwargs = dict(grid) if grid else dict(self.default_truncation_grid())
        for key in ("nested", "inner_max_folds"):
            kwargs.pop(key, None)
        kwargs.setdefault("progress", False)
        X_ready, Y_ready = self._tuning_fields(X, Y, clim_year_start, clim_year_end)
        if kwargs.pop("method", self.mode_selection) == "mdl":
            kwargs.pop("progress", None)
            result = self.select_mdl(X_ready, Y_ready, apply=False, **kwargs)
            self.remember_truncation(_pair_fingerprint(Predictant, Predictor), result)
            if os.environ.get("WAS_CCA_QUIET_TUNE", "0") != "1":
                print(f"  final MDL selection (training only, no inner CV): "
                      f"n_modes={result['n_modes']} n_pca_modes={result['n_pca_modes']}",
                      flush=True)
            return result
        if splits is None:
            splits = _loo_splits(Y["T"], nb_omit)
        result = self.tune(X_ready, Y_ready, splits, apply=False, **kwargs)
        self.remember_truncation(_pair_fingerprint(Predictant, Predictor), result)
        if os.environ.get("WAS_CCA_QUIET_TUNE", "0") != "1":
            print(f"  final truncation (full record, {len(splits)} splits): "
                  f"n_modes={result['n_modes']} n_pca_modes={result['n_pca_modes']} "
                  f"{result['metric']}={result['score']:.4f} ({result['rule']})", flush=True)
        return result

    # -- forecast tail -----------------------------------------------------
    def _forecast_probabilities(
        self, forecast_det, Predictant, Predictant_resid, hindcast_resid,
        clim_year_start, clim_year_end, mask,
        best_code_da=None, best_shape_da=None, best_loc_da=None, best_scale_da=None,
    ):
        """``(calibrated deterministic, probabilities)`` of a forecast.

        Calibrated against the out-of-sample hindcast with the convention of
        the strict cross-validator (see :meth:`_calibrated_probabilities`).
        """
        return self._calibrated_probabilities(
            forecast_det, Predictant, Predictant_resid - hindcast_resid,
            clim_year_start, clim_year_end, mask,
            best_code_da, best_shape_da, best_loc_da, best_scale_da,
        )

    def _cached_truncation_for(self, Predictant, Predictor):
        """Truncation selected for these very fields during cross-validation."""
        try:
            if not self.autotune_enabled():
                return None
            result = self.recall_truncation(_pair_fingerprint(Predictant, Predictor))
            if result is None:
                return None
            if hasattr(self, "detrend") and result.get("detrend") != self.detrend:
                return None
            method = ("aic" if self.mode_selection == "aic" or
                      getattr(self, "truncation_selection", "cv") == "aic"
                      else self.mode_selection)
            cached_method = result.get("method", "aic" if
                                       result.get("metric") == "bartlett_aic" else "cv")
            if cached_method != method:
                return None
            if self.mode_selection == "mdl":
                bounds = {"max_modes": self.n_modes,
                          "max_pca_modes": list(_resolve_pca_modes(self.n_pca_modes))}
                if result.get("selection_bounds") != bounds:
                    return None
            return result
        except Exception:
            return None

    def _forecast_common(self, Predictant, Predictor, Predictor_for_year, hindcast_det):
        Predictant = _prepare_tyx(Predictant, "Predictant")
        Predictor = _prepare_tyx(Predictor, "Predictor")
        Predictor_for_year = _prepare_tyx(Predictor_for_year, "Predictor_for_year")
        Predictant_resid, hindcast_resid = _align_observation_and_hindcast(
            Predictant, hindcast_det
        )
        return (Predictant, Predictor, Predictor_for_year,
                Predictant_resid, hindcast_resid, spatial_mask(Predictant))

    def forecast(self, Predictant, clim_year_start, clim_year_end, Predictor,
                 hindcast_det, Predictor_for_year, best_code_da=None,
                 best_shape_da=None, best_loc_da=None, best_scale_da=None):
        """Deterministic and probabilistic forecast.

        Truncation: the one ``cross_validate`` (via :meth:`finalize_truncation`)
        selected for these same fields is applied for the duration of the call
        and the instance is restored afterwards, so a single model object can
        be reused across a loop of predictors without any bookkeeping by the
        caller. When autotuning is on and nothing was cached for these fields
        (``forecast`` called without ``cross_validate``, or with other fields),
        the selection is made here on the complete record with the same
        procedure, so the forecast never silently falls back to the
        constructor values while the hindcast was tuned.

        Calibration: the deterministic field returned is the prediction
        corrected by the mean out-of-sample residual of ``hindcast_det`` and
        clipped to the declared physical support; the probabilities are
        centred on it with the residual variance (see
        :meth:`_calibrated_probabilities`) — the convention of the strict
        cross-validator, so the validated scores describe the emitted
        calibration.
        """
        cached = self._cached_truncation_for(Predictant, Predictor)
        if cached is None and self.autotune_enabled():
            try:
                cached = self.finalize_truncation(
                    Predictor, Predictant, clim_year_start, clim_year_end
                )
            except Exception as exc:  # an operational forecast must still be issued
                if self.mode_selection in ("mdl", "aic") or self.truncation_selection == "aic":
                    method = "MDL" if self.mode_selection == "mdl" else "AIC"
                    raise RuntimeError(f"CCA {method} selection failed; refusing an unselected forecast.") from exc
                warnings.warn(
                    f"CCA truncation could not be selected on the full record ({exc!r}); "
                    f"forecast uses the constructor values n_modes={self.n_modes}, "
                    f"n_pca_modes={self.n_pca_modes}.", RuntimeWarning, stacklevel=2)
        if cached is None:
            return self._forecast_impl(
                Predictant, clim_year_start, clim_year_end, Predictor,
                hindcast_det, Predictor_for_year, best_code_da,
                best_shape_da, best_loc_da, best_scale_da,
            )
        state = self.truncation_state()
        try:
            self.apply_truncation(cached)
            return self._forecast_impl(
                Predictant, clim_year_start, clim_year_end, Predictor,
                hindcast_det, Predictor_for_year, best_code_da,
                best_shape_da, best_loc_da, best_scale_da,
            )
        finally:
            self.restore_truncation(state)

    # -- plotting ----------------------------------------------------------
    def _plot_prepare(self, X, Y, clim_year_start, clim_year_end):
        """Transform X and Y into the space this class fits in."""
        raise NotImplementedError

    def plot_cca_results(self, X=None, Y=None, n_modes=None,
                         clim_year_start=None, clim_year_end=None):
        """Plot CCA spatial modes and canonical scores."""
        if self.n_modes == 0:
            print("  MDL retained no shared CCA mode; no canonical patterns to plot.", flush=True)
            return None
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
                 use_pca=True, dist_method="nonparam", eeof_ref_offset=3,
                 auto_truncation=None, mode_selection="cv"):
        super().__init__(n_modes, n_pca_modes, standardize, use_coslat, use_pca,
                         dist_method, auto_truncation=auto_truncation,
                         mode_selection=mode_selection)
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

    def _forecast_impl(self, Predictant, clim_year_start, clim_year_end, Predictor,
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

        forecast_det, forecast_prob = self._forecast_probabilities(
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

    The cross-validator estimates the transformation from each training fold;
    the full training climatology is used for the operational forecast.
    """

    def _plot_prepare(self, X, Y, clim_year_start, clim_year_end):
        if clim_year_start is not None and clim_year_end is not None:
            Y = standardize_timeseries(Y, clim_year_start, clim_year_end)  # noqa: F405
        return X, Y

    def _forecast_impl(self, Predictant, clim_year_start, clim_year_end, Predictor,
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

        forecast_det, forecast_prob = self._forecast_probabilities(
            forecast_det, Predictant, Predictant_resid, hindcast_resid,
            clim_year_start, clim_year_end, mask,
            best_code_da, best_shape_da, best_loc_da, best_scale_da,
        )
        return (forecast_det * mask).transpose("T", "Y", "X"), forecast_prob


# ---------------------------------------------------------------------------
# Linear-detrending family (WAS_CCA_op, WAS_CCA_strict)
# ---------------------------------------------------------------------------
class _DetrendedCCA(_CCACore):
    """Shared optional linear detrending and predictand standardization.

    ``WAS_CCA_strict`` deliberately does **not** subclass ``WAS_CCA_op``:
    ``WAS_Cross_Validator.cross_validate`` tests ``isinstance(model,
    WAS_CCA_op)`` before ``WAS_CCA_strict``, so inheritance between the two
    would silently route every strict model through the op branch.
    """

    @property
    def detrend(self):
        """Whether to remove X/Y linear trends and restore Y at prediction time."""
        return getattr(self, "_detrend", True)

    @detrend.setter
    def detrend(self, value):
        if not isinstance(value, (bool, np.bool_)):
            raise TypeError("detrend must be a boolean.")
        value = bool(value)
        previous = self.detrend
        self._detrend = value
        if previous != value:
            # A fitted basis and its selected dimensions belong to one
            # preprocessing space. Never reuse them after a flag change.
            self._truncation_cache = {}
            for name in ("truncation_", "truncation_folds_",
                         "truncation_diagnostics_folds_", "cca", "cca_model",
                         "_engine", "_engine_modes", "_prep", "_mdl_baseline"):
                setattr(self, name, None)

    @property
    def _mdl_detrend_dof(self):
        return int(self.detrend)

    def _prepare_training_fields(self, X, Y, clim_year_start, clim_year_end):
        """Fit the training-only transforms shared by CV, plots and forecasts."""
        Y_st = standardize_timeseries(Y, clim_year_start, clim_year_end)  # noqa: F405
        Y_st = Y_st.where(np.isfinite(Y_st))
        if not self.detrend:
            return X, Y_st, None, None, None, None
        X_ready, coeffs_X, meta_X = detrended_data(X, dim="T")  # noqa: F405
        Y_ready, coeffs_Y, meta_Y = detrended_data(Y_st, dim="T")  # noqa: F405
        return X_ready, Y_ready, coeffs_X, meta_X, coeffs_Y, meta_Y

    @staticmethod
    def _remove_predictor_trend(X, coeffs, meta):
        return X if coeffs is None else X - apply_detrend_data(X, coeffs, meta)  # noqa: F405

    @staticmethod
    def _restore_predictand_trend(Y, coeffs, meta):
        return Y if coeffs is None else Y + apply_detrend_data(Y, coeffs, meta)  # noqa: F405

    def prepare_fold(self, X_train, y_train_raw, X_test, clim_year_start, clim_year_end):
        """Fold-local transformation. Use this in the cross-validator.

        Returns ``(X_train_ready, y_train_ready, X_test_ready, coeffs_Y, meta_Y)``.
        If ``coeffs_Y`` is not None, restore the predictand trend at the
        validation year, then reverse-standardize against ``y_train_raw``.
        With ``detrend=False`` both trend metadata are None and neither trend
        fitting nor trend application runs; standardization remains active.

        ``utils.detrended_data`` references time to the first step of the array
        it is fitted on and stores that origin in ``meta['x0']``, so applying
        the training-fold coefficients to another fold is well defined.
        """
        X_train = _prepare_tyx(X_train, "X_train")
        X_test = _prepare_tyx(X_test, "X_test")
        y_train_raw = _prepare_tyx(y_train_raw, "y_train")

        X_ready, Y_ready, coeffs_X, meta_X, coeffs_Y, meta_Y = self._prepare_training_fields(
            X_train, y_train_raw, clim_year_start, clim_year_end)
        X_test_ready = self._remove_predictor_trend(X_test, coeffs_X, meta_X)
        return X_ready, Y_ready, X_test_ready, coeffs_Y, meta_Y

    def _plot_prepare(self, X, Y, clim_year_start, clim_year_end):
        return self._prepare_training_fields(X, Y, clim_year_start, clim_year_end)[:2]

    def _tuning_fields(self, X, Y, clim_year_start, clim_year_end):
        """Same transformation as one training fold of ``_cca_cv_detrended``."""
        X_ready, Y_ready, _, _, _ = self.prepare_fold(X, Y, X, clim_year_start, clim_year_end)
        return X_ready.fillna(0.0), Y_ready.fillna(0.0)


class WAS_CCA_op(_DetrendedCCA):
    """CCA on X and standardized Y, optionally linearly detrended.

    ``detrend=True`` preserves the historical transformation; False disables
    all trend fitting, subtraction and restoration, including in CV.

    The cross-validator must fit ``detrended_data`` **inside** each fold. Fitting
    the trend on the full record and reusing its coefficients to rebuild the
    hindcast lets the regression see the validation year — negligible on
    rainfall, material on temperature, where the trend carries most of the
    variance. See :meth:`prepare_fold` .
    """

    def __init__(self, n_modes=4, n_pca_modes=8, standardize=False, use_coslat=True,
                 use_pca=True, dist_method="bestfit", auto_truncation=None,
                 mode_selection="cv", detrend=True):
        super().__init__(n_modes, n_pca_modes, standardize, use_coslat, use_pca,
                         dist_method, auto_truncation=auto_truncation,
                         mode_selection=mode_selection)
        self.detrend = detrend

    def _forecast_impl(self, Predictant, clim_year_start, clim_year_end, Predictor,
                       hindcast_det, Predictor_for_year, best_code_da=None,
                       best_shape_da=None, best_loc_da=None, best_scale_da=None):
        (Predictant, Predictor, Predictor_for_year,
         Predictant_resid, hindcast_resid, mask) = self._forecast_common(
            Predictant, Predictor, Predictor_for_year, hindcast_det
        )

        (Predictor_ready, Predictant_ready, coeffs_X, meta_X,
         coeffs_Y, meta_Y) = self._prepare_training_fields(
            Predictor, Predictant, clim_year_start, clim_year_end)

        X_year = Predictor_for_year.fillna(Predictor.mean(dim="T", skipna=True))
        # Evaluate the predictor trend at the calendar point the training
        # predictor actually uses (see _forecast_predictor_time).
        X_year = X_year.assign_coords(
            T=_forecast_predictor_time(Predictor, Predictor_for_year, Predictant)
        )
        X_year = self._remove_predictor_trend(X_year, coeffs_X, meta_X)

        self.fit_cca(Predictor_ready, Predictant_ready)
        X_prepared = self.preprocess_test_data(X_year)[0]
        result_detrended = self._predict_prepared(X_prepared)

        # Evaluate the Y trend at the predictand target time, not the init month.
        result_detrended = result_detrended.assign_coords(
            T=xr.DataArray(_target_time(Predictant, Predictor_for_year), dims=["T"])
        )
        result_st = self._restore_predictand_trend(result_detrended, coeffs_Y, meta_Y)

        forecast_det = reverse_standardize(  # noqa: F405
            result_st, Predictant, clim_year_start, clim_year_end
        ).transpose("T", "Y", "X")
        forecast_det = conditional_nonnegative_clip(forecast_det, Predictant)

        forecast_det, forecast_prob = self._forecast_probabilities(
            forecast_det, Predictant, Predictant_resid, hindcast_resid,
            clim_year_start, clim_year_end, mask,
            best_code_da, best_shape_da, best_loc_da, best_scale_da,
        )
        return (forecast_det * mask).transpose("T", "Y", "X"), forecast_prob


class _RidgeCCAEngine:
    """Mixin providing ridge-regularized canonical correlation.

    Combine with WAS_CCA_op or WAS_CCA_strict (mixin first, so its
    fit_cca/_predict_prepared/compute_model take priority in the MRO) to
    get the matching _forecast_impl / spatial-gap-filling behaviour of
    whichever sibling you combine it with — see WAS_CCA_Regularized
    (op-style) and WAS_CCA_Regularized_strict (strict spatial filling,
    the right choice for SST domains carrying a land mask).

    Adds covariance shrinkage to the CCA: it shrinks the X/Y covariance
    estimates toward a scaled identity before solving for the canonical
    directions — ridge regression applied to CCA (Vinod, 1976). On a short,
    correlated-predictor hindcast this is typically more stable than a
    mode-count search alone. The truncation search of the cross-validator
    still runs when autotuning is on (``auto_truncation`` / WAS_CCA_AUTOTUNE),
    and every candidate is then scored with this engine at the same
    ``alpha`` (:func:`_candidate_predictor`), never with the unregularized
    xeofs CCA; ``alpha`` itself is not searched.

    Parameters
    ----------
    alpha : float in [0, 1]
        Shrinkage strength. 0 recovers plain PCA + unregularized CCA (the
        ill-posed regime WAS_CCA_op/_strict's mode search works around). 1
        shrinks the covariance fully to a scaled identity. Tune this the
        same way you would a ridge penalty: by out-of-sample CV.
    n_modes : int or None
        Optional cap on canonical directions kept after shrinkage (default:
        keep all min(kx, ky)). Regularization and truncation are not
        mutually exclusive, though shrinkage alone is usually enough. With
        ``None`` the default truncation grid searches four mode counts
        between 1 and the X PCA truncation (see ``_default_truncation_grid``).

    Notes
    -----
    Validated against sklearn.cross_decomposition.CCA at alpha=0 (canonical
    correlations match to 3 decimals on synthetic data), and against a
    plain (alpha=0) fit in a near-singular regime (kx close to n): shrinkage
    around alpha=0.05-0.1 improved out-of-sample skill and cut its variance
    ~4x in that synthetic test, while a null-relationship control confirmed
    shrinkage does not hallucinate skill where none exists.
    """

    def __init__(self, n_modes=None, n_pca_modes=8, alpha=0.1, standardize=False,
                 use_coslat=True, use_pca=True, dist_method="bestfit",
                 auto_truncation=None, detrend=True):
        super().__init__(n_modes, n_pca_modes, standardize, use_coslat, use_pca,
                         dist_method, auto_truncation=auto_truncation, detrend=detrend)
        self.alpha = float(alpha)
        if not 0.0 <= self.alpha <= 1.0:
            raise ValueError("alpha must be in [0, 1].")

    # -- ridge-regularized canonical correlation, replacing xeofs's CCA -----
    @staticmethod
    def _pca_fit(X2d, k):
        mean = X2d.mean(axis=0)
        Xc = X2d - mean
        U, s, Vt = np.linalg.svd(Xc, full_matrices=False)
        k = max(1, min(int(k), Vt.shape[0], X2d.shape[0] - 1))
        comps = Vt[:k]
        return mean, comps, Xc @ comps.T

    @staticmethod
    def _pca_transform(X2d, mean, comps):
        return (X2d - mean) @ comps.T

    @staticmethod
    def _pca_inverse(scores, mean, comps):
        return scores @ comps + mean

    @staticmethod
    def _fit_ridge_cca(Ux, Uy, alpha):
        """Canonical vectors a, b and correlations rho on already-PCA-reduced
        score matrices, via a whitened SVD (numerically stabler than a raw
        generalized eigenproblem on the regularized covariances)."""
        n = Ux.shape[0]
        Sxx = Ux.T @ Ux / max(n - 1, 1)
        Syy = Uy.T @ Uy / max(n - 1, 1)
        Sxy = Ux.T @ Uy / max(n - 1, 1)
        kx, ky = Sxx.shape[0], Syy.shape[0]
        Sxx_r = (1 - alpha) * Sxx + alpha * (np.trace(Sxx) / kx) * np.eye(kx)
        Syy_r = (1 - alpha) * Syy + alpha * (np.trace(Syy) / ky) * np.eye(ky)

        def _inv_sqrt(S):
            w, V = np.linalg.eigh(S)
            w = np.clip(w, 1e-10, None)
            return V @ np.diag(1.0 / np.sqrt(w)) @ V.T

        Sxx_is, Syy_is = _inv_sqrt(Sxx_r), _inv_sqrt(Syy_r)
        K = Sxx_is @ Sxy @ Syy_is
        U_, s, Vt = np.linalg.svd(K, full_matrices=False)
        r = min(kx, ky)
        a = Sxx_is @ U_[:, :r]
        b = Syy_is @ Vt.T[:, :r]
        return a, b, s[:r]

    def fit_cca(self, X_train, y_train):
        # Same fold-safe preprocessing contract as _CCACore.fit_cca — mask,
        # mean-impute, and reference everything on the training fold only.
        X_final, y_final = self.preprocess_data(X_train, y_train)
        self._ridge_solve(self._ridge_prepare(X_final, y_final))
        # Kept truthy for any external code that only checks "has this been
        # fitted yet" — this class has no separate xeofs model object.
        self.cca = self.cca_model = self
        return self.cca_model

    def _ridge_prepare(self, X_final, y_final):
        """Candidate-independent part of the fit: scaling and full SVD of both fields.

        ``_pca_fit`` truncates the same thin SVD for every ``k``, so one
        decomposition serves every PCA truncation of a tuning grid.
        """
        y_dims = {d: y_final[d].values for d in y_final.dims if d != "T"}
        Xv = np.nan_to_num(X_final.values.reshape(X_final.sizes["T"], -1), nan=0.0)
        Yv = np.nan_to_num(y_final.values.reshape(y_final.sizes["T"], -1), nan=0.0)
        if Xv.shape[0] != Yv.shape[0] or Xv.shape[0] < 3:
            raise ValueError("CCA needs at least three paired training samples.")
        def scaling(field, values):
            mean = values.mean(axis=0) if self.standardize else np.zeros(values.shape[1])
            std = values.std(axis=0) if self.standardize else np.ones(values.shape[1])
            std = np.where(std > 1e-12, std, 1.0)
            weight = np.ones_like(mean)
            lat = "lat" if "lat" in field.dims else "Y"
            if self.use_coslat and lat in field.dims:
                w = np.sqrt(np.clip(np.cos(np.deg2rad(field[lat])), 1e-12, None))
                weight = xr.ones_like(field.isel(T=0, drop=True)).__mul__(w).values.ravel()
            return mean, std, weight
        xs, ys = scaling(X_final, Xv), scaling(y_final, Yv)
        Xv = (Xv - xs[0]) / xs[1] * xs[2]
        Yv = (Yv - ys[0]) / ys[1] * ys[2]

        def decompose(V):
            mean = V.mean(axis=0)
            U, sv, Vt = np.linalg.svd(V - mean, full_matrices=False)
            return mean, U, sv, Vt

        return {"y_dims": y_dims, "x_scaling": xs, "y_scaling": ys,
                "n": Xv.shape[0], "px": Xv.shape[1], "py": Yv.shape[1],
                "x": decompose(Xv), "y": decompose(Yv)}

    def _ridge_solve(self, prep):
        """Ridge CCA at this instance's ``(n_modes, n_pca_modes, alpha)``."""
        self._y_dims = prep["y_dims"]
        self._r_xmean, self._r_xstd, self._r_xweight = prep["x_scaling"]
        self._r_ymean, self._r_ystd, self._r_yweight = prep["y_scaling"]
        requested = self.n_pca_modes
        kx, ky = requested if isinstance(requested, (list, tuple)) else (requested, requested)
        kx = prep["px"] if not self.use_pca or kx is None else int(kx)
        ky = prep["py"] if not self.use_pca or ky is None else int(ky)

        def truncate(dec, k):
            # identical to _pca_fit on the same matrix
            mean, U, sv, Vt = dec
            k = max(1, min(int(k), Vt.shape[0], prep["n"] - 1))
            return mean, Vt[:k], U[:, :k] * sv[:k]

        self._mx, self._cx, Sx = truncate(prep["x"], kx)
        self._my, self._cy, Sy = truncate(prep["y"], ky)
        self._a, self._b, self._rho = self._fit_ridge_cca(Sx, Sy, self.alpha)
        if self.n_modes:
            r = max(1, min(int(self.n_modes), self._a.shape[1]))
            self._a, self._b, self._rho = self._a[:, :r], self._b[:, :r], self._rho[:r]
        # Shrinkage-normalized scores do not have unit empirical variance.
        # Learn the forecast reconstruction on training scores, rather than
        # treating shrinkage singular values as empirical correlations.
        Cx = Sx @ self._a
        self._score_to_y = np.linalg.lstsq(Cx, Sy, rcond=None)[0]
        return self

    def _ridge_candidates(self, X_train, y_train, X_test, candidates):
        """Predictions for several ``(n_modes, n_pca_modes)`` from one preparation."""
        base = copy.copy(self)
        base._prep = None
        X_final, y_final = base.preprocess_data(X_train, y_train)
        prep = base._ridge_prepare(X_final, y_final)
        X_prepared, _ = base.preprocess_test_data(X_test, X_train=X_train)
        out = []
        for n_modes, n_pca in candidates:
            try:
                probe = copy.copy(base)
                probe.n_modes, probe.n_pca_modes = n_modes, n_pca
                out.append(probe._ridge_solve(prep)._predict_prepared(X_prepared))
            except Exception as exc:
                out.append(exc)
        return out

    def _predict_prepared(self, X_prepared):
        Xv = np.nan_to_num(X_prepared.values.reshape(X_prepared.sizes["T"], -1), nan=0.0)
        Xv = (Xv - self._r_xmean) / self._r_xstd * self._r_xweight
        Sx = self._pca_transform(Xv, self._mx, self._cx)
        Cx = Sx @ self._a
        Sy_pred = Cx @ self._score_to_y
        Yv_pred = self._pca_inverse(Sy_pred, self._my, self._cy)
        Yv_pred = Yv_pred / self._r_yweight * self._r_ystd + self._r_ymean
        out_shape = (X_prepared.sizes["T"],) + tuple(len(v) for v in self._y_dims.values())
        y_pred = xr.DataArray(
            Yv_pred.reshape(out_shape),
            dims=("T",) + tuple(self._y_dims.keys()),
            coords={"T": X_prepared["T"], **self._y_dims},
        )
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


class WAS_CCA_Regularized(_RidgeCCAEngine, WAS_CCA_op):
    """Ridge-regularized CCA (op-style: no strict spatial gap filling).

    See _RidgeCCAEngine for the shrinkage mechanism and its validation.
    """
    pass


# ---------------------------------------------------------------------------
# WAS_CCA_strict — op transformations, strict spatial filling
# ---------------------------------------------------------------------------
class WAS_CCA_strict(_DetrendedCCA):
    """Same transformations as ``op`` with spatial filling of *partial* gaps.

    A sibling of ``WAS_CCA_op``, not a subclass — see ``_DetrendedCCA``.

    Cells with occasional missing dates are filled from the training record
    and their neighbours (:meth:`_fill_spatial_gaps`); cells never observed
    (land on an SST domain) stay NaN and are masked out, exactly as in ``op``.
    Intended for the fold-safe cross-validation branch, where every transform is
    refitted on the training fold. :meth:`tune` called inside that loop gives a
    fully nested truncation estimate.
    """

    def __init__(self, n_modes=4, n_pca_modes=8, standardize=False, use_coslat=True,
                 use_pca=True, dist_method="bestfit", auto_truncation=None,
                 mode_selection="cv", detrend=True):
        super().__init__(n_modes, n_pca_modes, standardize, use_coslat, use_pca,
                         dist_method, auto_truncation=auto_truncation,
                         mode_selection=mode_selection)
        self.detrend = detrend

    @staticmethod
    def _fill_spatial_gaps(da, ref=None, min_valid=10):
        """Fill *partial* gaps only; cells never observed stay NaN.

        A cell is filled — with the historical mean of ``ref`` when given,
        then by its nearest neighbours along Y and X — only when it is finite
        on at least ``min_valid`` dates of the training record (``ref``, or
        ``da`` itself; capped at the record length). Cells without any
        observation, the continents of an SST domain, are left NaN so that
        ``_common_validity_mask`` excludes them from the CCA. The previous
        unconditional ffill/bfill copied the nearest coastal column over
        every land cell and fed those duplicates to the covariance as real
        features. ``min_valid=10`` matches ``detrended_data``, below which a
        cell gets no trend and is dropped anyway.
        """
        base = ref if ref is not None and "T" in getattr(ref, "dims", ()) else da
        out = da.fillna(base.mean(dim="T", skipna=True)) if base is not da else da
        if "T" in base.dims:
            need = max(1, min(int(min_valid), int(base.sizes["T"])))
            observed = np.isfinite(base).sum(dim="T") >= need
        else:
            observed = np.isfinite(base)
        return _fill_spatial_gaps_safe(out).where(observed, out)

    def _tuning_fields(self, X, Y, clim_year_start, clim_year_end):
        return super()._tuning_fields(self._fill_spatial_gaps(X), Y,
                                      clim_year_start, clim_year_end)

    @staticmethod
    def _target_time_from_predictor(Predictant, Predictor_for_year):
        return _target_time(Predictant, Predictor_for_year)[0]

    def _forecast_impl(self, Predictant, clim_year_start, clim_year_end, Predictor,
                       hindcast_det, Predictor_for_year, best_code_da=None,
                       best_shape_da=None, best_loc_da=None, best_scale_da=None):
        (Predictant, Predictor, Predictor_for_year,
         Predictant_resid, hindcast_resid, mask) = self._forecast_common(
            Predictant, Predictor, Predictor_for_year, hindcast_det
        )

        Predictor_filled = self._fill_spatial_gaps(Predictor)
        (Predictor_ready, Predictant_ready, coeffs_X, meta_X,
         coeffs_Y, meta_Y) = self._prepare_training_fields(
            Predictor_filled, Predictant, clim_year_start, clim_year_end)

        X_year = self._fill_spatial_gaps(Predictor_for_year, ref=Predictor_filled)
        X_year = X_year.assign_coords(
            T=_forecast_predictor_time(Predictor, Predictor_for_year, Predictant)
        )
        X_year = self._remove_predictor_trend(X_year, coeffs_X, meta_X)

        self.fit_cca(Predictor_ready, Predictant_ready)
        X_prepared = self.preprocess_test_data(X_year)[0]
        result_detrended = self._predict_prepared(X_prepared)

        result_detrended = result_detrended.assign_coords(
            T=xr.DataArray(_target_time(Predictant, Predictor_for_year), dims=["T"])
        )
        result_st = self._restore_predictand_trend(result_detrended, coeffs_Y, meta_Y)

        forecast_det = reverse_standardize(  # noqa: F405
            result_st, Predictant, clim_year_start, clim_year_end
        ).transpose("T", "Y", "X")
        forecast_det = conditional_nonnegative_clip(forecast_det, Predictant)

        forecast_det, forecast_prob = self._forecast_probabilities(
            forecast_det, Predictant, Predictant_resid, hindcast_resid,
            clim_year_start, clim_year_end, mask,
            best_code_da, best_shape_da, best_loc_da, best_scale_da,
        )
        return (forecast_det * mask).transpose("T", "Y", "X"), forecast_prob


class WAS_CCA_Regularized_strict(_RidgeCCAEngine, WAS_CCA_strict):
    """Ridge-regularized CCA with strict spatial gap filling.

    The right choice for SST-domain predictors carrying a land mask — the
    same reason WAS_CCA_strict exists instead of WAS_CCA_op for those
    domains. See _RidgeCCAEngine for the shrinkage mechanism and its
    validation.
    """
    pass


# Keep constructor inputs for fresh model instances in nested cross-validation.
from .was_crossfit_adapters import capture_constructors as _capture_constructors
_capture_constructors(globals())
